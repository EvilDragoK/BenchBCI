"""3D 脑网络可视化（BrainNet Viewer / BrainNet Studio 风格）。

设计（对应 BrainNet Studio 工作流的「可视化」环节）：
- 半透明头部模型（球面 + 鼻/耳轮廓）作为解剖参考
- 电极（网络节点）按真实 3D 坐标放置，节点大小 ∝ 节点重要性（连接强度）
- 连接边按强度着色：正连接=红、负连接=蓝，线宽 ∝ |连接强度|
- 支持鼠标旋转/缩放（matplotlib 3D 工具栏），可导出 PNG
"""
from __future__ import annotations

import numpy as np

from PyQt6.QtWidgets import (QCheckBox, QDialog, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QVBoxLayout, QWidget)

from core.brainnet import node_importance
from .plotting import new_canvas_with_toolbar
from .topo_dialog import get_eeg_pos3d


class Brain3DDialog(QDialog):
    def __init__(self, adj, raw, title: str = "3D 脑网络", parent=None):
        super().__init__(parent)
        self.adj = np.asarray(adj, dtype=float)
        self.raw = raw
        self.setWindowTitle(title)
        self.resize(940, 720)

        self.spin_density = QDoubleSpinBox()
        self.spin_density.setRange(0.05, 1.0)
        self.spin_density.setSingleStep(0.05)
        self.spin_density.setValue(0.2)
        self.chk_neg = QCheckBox("显示负连接（蓝）")
        self.chk_neg.setChecked(True)
        self.chk_names = QCheckBox("显示电极名")
        self.chk_names.setChecked(True)
        self.btn_draw = QPushButton("重绘")
        self.btn_png = QPushButton("导出 PNG")
        self.spin_density.setToolTip("只显示最强的 N% 连接边\n0.2 = 最强 20%（231 条边里约 46 条）")
        self.lbl_stat = QLabel("—")
        self.lbl_stat.setStyleSheet("color:#555;")
        self.lbl_stat.setWordWrap(True)

        # 缩放控件
        self.btn_zoom_in = QPushButton("放大 ＋")
        self.btn_zoom_out = QPushButton("缩小 －")
        self.btn_zoom_reset = QPushButton("复位")
        zoom_row = QWidget()
        zl = QHBoxLayout(zoom_row)
        zl.setContentsMargins(0, 0, 0, 0)
        zl.addWidget(self.btn_zoom_in)
        zl.addWidget(self.btn_zoom_out)
        zl.addWidget(self.btn_zoom_reset)
        self.lbl_zoom = QLabel("缩放 1.00×")
        self.lbl_zoom.setStyleSheet("color:#555;")

        form = QFormLayout()
        form.addRow("边密度", self.spin_density)
        form.addRow("当前显示", self.lbl_stat)
        form.addRow(self.chk_neg)
        form.addRow(self.chk_names)
        form.addRow("缩放", zoom_row)
        form.addRow(self.lbl_zoom)
        form.addRow(self.btn_draw)
        form.addRow(self.btn_png)
        form.addRow(QLabel("提示：拖动旋转；滚轮缩放；\n双击画布恢复默认视角。"))
        panel = QWidget()
        panel.setLayout(form)
        panel.setMaximumWidth(240)

        self.fig, self.canvas, self.tb = new_canvas_with_toolbar(9, 6.6)
        body = QHBoxLayout(self)
        body.addWidget(panel)
        right = QVBoxLayout()
        right.addWidget(self.tb)
        right.addWidget(self.canvas, 1)
        body.addLayout(right, 1)

        self.btn_draw.clicked.connect(self.refresh)
        self.btn_png.clicked.connect(self._export)
        self.spin_density.valueChanged.connect(lambda *_: self.refresh())
        self.chk_neg.toggled.connect(lambda *_: self.refresh())
        self.chk_names.toggled.connect(lambda *_: self.refresh())
        self.btn_zoom_in.clicked.connect(lambda: self._zoom_by(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom_by(1 / 1.25))
        self.btn_zoom_reset.clicked.connect(self._zoom_reset)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_double_click)
        self._zoom = 1.0
        self._ax = None
        self._names: list[str] = []
        self.refresh()

    # ------------------------------------------------------------ 缩放
    def _zoom_by(self, factor: float):
        self._zoom = float(np.clip(self._zoom * factor, 0.2, 10.0))
        self._apply_zoom()

    def _zoom_reset(self):
        self._zoom = 1.0
        self._apply_zoom()

    def _on_scroll(self, event):
        ax = getattr(self, "_ax", None)
        if ax is not None and event.inaxes is ax:
            self._zoom_by(1.15 if getattr(event, "button", "") == "up" else 1 / 1.15)

    def _on_double_click(self, event):
        """双击画布 → 恢复默认视角与缩放。"""
        if event.dblclick and getattr(self, "_ax", None) is not None:
            self._zoom = 1.0
            self._ax.view_init(elev=18, azim=-68)
            self._apply_zoom()

    def _apply_zoom(self):
        ax = getattr(self, "_ax", None)
        if ax is None:
            return
        try:
            ax.set_box_aspect((1, 1, 1), zoom=self._zoom)
        except TypeError:  # 旧版 matplotlib 不支持 zoom 参数
            pass
        self.lbl_zoom.setText(f"缩放 {self._zoom:.2f}×")
        self.canvas.draw()

    # ------------------------------------------------------------------
    def refresh(self):
        try:
            idx, pos, names = get_eeg_pos3d(self.raw)
        except Exception as e:  # noqa: BLE001
            from ..workers import exc_to_zh

            QMessageBox.warning(self, "提示", exc_to_zh(e))
            return
        if not len(idx):
            QMessageBox.warning(self, "提示", "通道名无法匹配标准导联，无法绘制 3D 网络")
            return
        self._names = names
        A = self.adj[np.ix_(idx, idx)].copy()
        np.fill_diagonal(A, 0.0)

        # 保留用户当前的旋转视角（重绘不跳回默认角度）
        prev = getattr(self, "_ax", None)
        elev, azim = (prev.elev, prev.azim) if prev is not None else (18, -68)

        fig = self.fig
        fig.clear()
        ax = fig.add_subplot(111, projection="3d")

        # ---- 头部模型 ----
        R = float(np.abs(pos).max()) * 1.06
        u = np.linspace(0, 2 * np.pi, 48)
        v = np.linspace(0, np.pi, 24)
        xs = R * np.cos(u)[None, :] * np.sin(v)[:, None]
        ys = R * np.sin(u)[None, :] * np.sin(v)[:, None]
        zs = R * np.cos(v)[:, None] * np.ones_like(u)[None, :]
        ax.plot_surface(xs, ys, zs, color="lightgray", alpha=0.06,
                        linewidth=0, shade=False, zorder=0)
        # 鼻子（前方 y+）与双耳（左右 x±）
        ax.plot([0, 0], [R * 0.92, R * 1.18], [0, 0], color="gray", lw=1.2)
        ax.plot([0, 0], [R * 1.18, R * 0.92], [0, 0], color="gray", lw=1.2)
        for sx in (-1, 1):
            th = np.linspace(0, 2 * np.pi, 20)
            ax.plot(sx * np.full_like(th, R), 0.18 * R * np.sin(th),
                    0.18 * R * np.cos(th), color="gray", lw=1.0)

        # ---- 连接边（先画边，保证节点/文字在上层）----
        Aabs = np.abs(A)
        tri = Aabs[np.triu_indices(len(A), 1)]
        density = self.spin_density.value()
        k = max(1, int(len(tri) * density))
        thr = np.partition(tri, len(tri) - k)[len(tri) - k] if len(tri) else 0.0
        wmax = tri.max() if len(tri) and tri.max() > 0 else 1.0
        n_pos = n_neg = n_hidden = 0
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                w = A[i, j]
                if Aabs[i, j] < thr or w == 0:
                    continue
                if w < 0 and not self.chk_neg.isChecked():
                    n_hidden += 1
                    continue
                color = "#d62728" if w > 0 else "#1f77b4"
                if w > 0:
                    n_pos += 1
                else:
                    n_neg += 1
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        [pos[i, 2], pos[j, 2]],
                        color=color, lw=0.5 + 3.0 * Aabs[i, j] / wmax,
                        alpha=0.7, zorder=1)

        # ---- 节点（大小 ∝ 重要性）----
        imp = node_importance(A)
        sizes = 30 + 260 * imp / imp.max() if imp.max() > 0 else np.full(len(imp), 40.0)
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=sizes, c="black",
                   depthshade=False, zorder=10)

        # ---- 电极名（沿径向外移，避免被头壳遮挡）----
        if self.chk_names.isChecked():
            for n, p in zip(names, pos):
                norm = np.linalg.norm(p) or 1.0
                pv = p / norm * (norm * 1.10)
                ax.text(pv[0], pv[1], pv[2], n, fontsize=8, color="#222222",
                        ha="center", va="center", zorder=20)

        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        self._ax = ax  # 供缩放/双击回调使用
        ax.set_title(f"3D 脑网络（{len(names)} 节点，边密度={density:.2f}，红=正/蓝=负）",
                     fontsize=10)
        self.lbl_stat.setText(
            f"正连接 {n_pos} 条 / 负连接 {n_neg} 条"
            + (f"\n（因未勾选，隐藏负连接 {n_hidden} 条）" if n_hidden else "")
            + ("\n提示：当前矩阵无负值\n（plv/pli/mi 恒为正，\n只有 corr 才有负连接）"
               if n_pos and n_neg == 0 and n_hidden == 0 else ""))
        self._apply_zoom()

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出图片", "brain3d.png",
                                              "PNG (*.png);;SVG (*.svg)")
        if not path:
            return
        self.fig.savefig(path, dpi=180)
        QMessageBox.information(self, "完成", f"已导出 {path}")
