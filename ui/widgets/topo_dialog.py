"""头皮地形图（脑地形图 / topomap）可视化对话框。

设计方案（针对"每个数据如何映射到头皮图"）：

1. 瞬时幅值模式
   选定时间点 t（可一键取波形浏览器当前窗口中心），取各 EEG 通道在 t 时刻的
   瞬时值（单位跟随当前显示单位：µV 或 z/a.u.），画单张地形图。
   → 回答"某一瞬间能量在头皮上怎么分布"。

2. 频带功率模式
   以波形浏览器当前显示窗口为数据段，计算 δ/θ/α/β/γ 各频带绝对功率
   （复用 core.features.absolute_power），可画全部 5 张或单选一个频带。
   → 回答"这段时间哪个脑区哪个频带活跃"。

3. 电极定位：通道名 → standard_1005 蒙太奇 2D 坐标（只画能匹配到位置的
   通道），空间插值由 MNE plot_topomap 内部的球面插值完成，外圈自动绘制
   头轮廓 + 鼻 + 双耳，与论文图样式一致。

4. 交互：与信号浏览器联动（窗口中心/当前窗口）；可导出 PNG/SVG。
"""
from __future__ import annotations

import numpy as np

from PyQt6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                             QHBoxLayout, QMessageBox, QPushButton, QFormLayout,
                             QVBoxLayout, QWidget)

from core.features import BANDS, absolute_power
from .plotting import new_canvas

#: 电极点样式：MNE plot_topomap 的 sensors 传格式字符串（'ko' = 黑色实心圆，
#: 比 MNE 默认的微型散点 s=0.25 清晰得多）
_SENSORS = "ko"


def get_eeg_pos3d(raw):
    """把 raw 的 EEG 通道映射到 standard_1005 的 3D 坐标（米，头部坐标系）。

    返回 (idx, pos (P,3), names)，供 3D 脑网络可视化使用。
    """
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    ch = [raw.ch_names[i] for i in picks] if picks.size else list(raw.ch_names)
    try:
        ch_pos = mne.channels.make_standard_montage("standard_1005").get_positions().get("ch_pos", {})
    except Exception:  # noqa: BLE001
        ch_pos = {}
    pts = [(i, np.asarray(ch_pos[c], dtype=float))
           for i, c in enumerate(ch)
           if c in ch_pos and np.isfinite(ch_pos[c]).all()]
    idx = [i for i, _ in pts]
    pos = np.array([p for _, p in pts]) if pts else np.zeros((0, 3))
    names = [ch[i] for i in idx]
    return idx, pos, names


def get_eeg_pos(raw):
    """把 raw 的 EEG 通道映射到 standard_1005 2D 坐标。

    返回 (idx, pos, ch_names)：idx 是 raw 中能匹配到位置的通道索引，pos (P,2)，
    ch_names 与之对齐。供脑地形图 / ERP 成分地形图复用。
    """
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    ch = [raw.ch_names[i] for i in picks] if picks.size else list(raw.ch_names)
    try:
        pos3d = mne.channels.make_standard_montage("standard_1005").get_positions().get("ch_pos", {})
    except Exception:  # noqa: BLE001
        pos3d = {}
    pts = [(i, np.asarray(pos3d[c][:2], dtype=float))
           for i, c in enumerate(ch)
           if c in pos3d and np.isfinite(pos3d[c][:2]).all()]
    idx = [i for i, _ in pts]
    pos = np.array([p for _, p in pts]) if pts else np.zeros((0, 2))
    names = [ch[i] for i in idx]
    return idx, pos, names


class TopoDialog(QDialog):
    def __init__(self, raw, browser=None, parent=None):
        super().__init__(parent)
        self.raw = raw
        self.browser = browser
        self._unit = getattr(browser, "_unit", "µV") if browser else "µV"
        self._scale = getattr(browser, "_scale", 1e6) if browser else 1e6
        self.setWindowTitle("脑地形图")
        self.resize(920, 640)

        import mne

        self._fs = float(raw.info["sfreq"])
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        self._eeg_picks = list(picks) if picks.size else list(range(len(raw.ch_names)))
        self._ch = [raw.ch_names[i] for i in self._eeg_picks]

        # ---- 通道 2D 坐标（standard_1005），只保留能匹配的 ----
        try:
            positions = mne.channels.make_standard_montage("standard_1005").get_positions()
            pos3d = positions.get("ch_pos", {})
        except Exception:  # noqa: BLE001
            pos3d = {}
        pts = [(i, np.asarray(pos3d[ch][:2], dtype=float))
               for i, ch in enumerate(self._ch)
               if ch in pos3d and np.isfinite(pos3d[ch][:2]).all()]
        self._idx = [i for i, _ in pts]
        self._pos = np.array([p for _, p in pts]) if pts else np.zeros((0, 2))
        self._ch_names = [self._ch[i] for i in self._idx]  # 与 pos 一一对应的通道名

        # ---- 控件 ----
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["瞬时幅值", "频带功率"])
        self.spin_t = QDoubleSpinBox()
        self.spin_t.setRange(0.0, max(raw.n_times / self._fs, 0.1))
        self.spin_t.setDecimals(2)
        self.spin_t.setSingleStep(0.1)
        self.spin_t.setSuffix(" s")
        self.btn_center = QPushButton("取当前窗口中心")
        self.combo_band = QComboBox()
        self.combo_band.addItems(list(BANDS))  # 默认 delta，单频带显示
        self.btn_draw = QPushButton("绘制")
        self.btn_draw.setStyleSheet("font-weight:bold;")
        self.btn_png = QPushButton("导出 PNG")

        form = QFormLayout()
        form.addRow("模式", self.combo_mode)
        tw = QWidget()
        tl = QHBoxLayout(tw)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.spin_t)
        tl.addWidget(self.btn_center)
        form.addRow("时间点", tw)
        form.addRow("频带", self.combo_band)
        bw = QWidget()
        bl = QHBoxLayout(bw)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.btn_draw)
        bl.addWidget(self.btn_png)
        form.addRow(bw)
        panel = QWidget()
        panel.setLayout(form)
        panel.setMaximumWidth(340)

        self.fig, self.canvas = new_canvas(9.2, 6.4)
        body = QHBoxLayout(self)
        body.addWidget(panel)
        body.addWidget(self.canvas, 1)

        self.btn_center.clicked.connect(self._use_center)
        self.btn_draw.clicked.connect(self.refresh)
        self.btn_png.clicked.connect(self._export_png)
        self.combo_mode.currentIndexChanged.connect(lambda *_: self.refresh())
        self.combo_band.currentIndexChanged.connect(lambda *_: self.refresh())
        self._use_center()
        self.refresh()

    # ------------------------------------------------------------------
    def _center_t(self) -> float:
        if self.browser is not None:
            t0, t1 = self.browser.view_window()
            return (t0 + t1) / 2.0
        return 0.0

    def _use_center(self):
        self.spin_t.setValue(round(self._center_t(), 2))

    def _values_instant(self, t: float) -> np.ndarray:
        i = int(np.clip(t, 0.0, self.raw.n_times - 1) * self._fs)
        col = self.raw.get_data(picks=self._eeg_picks, start=i, stop=i + 1)
        return col[:, 0] * self._scale

    def _values_bandpower(self, band_name: str):
        if self.browser is not None:
            t0, t1 = self.browser.view_window()
        else:
            t0, t1 = 0.0, 2.0
        s = int(t0 * self._fs)
        e = min(int(t1 * self._fs) + 1, self.raw.n_times)
        seg = self.raw.get_data(picks=self._eeg_picks, start=s, stop=e) * self._scale
        ap = absolute_power(seg[None], self._fs)[0]  # (C, 5)
        k = list(BANDS).index(band_name)
        return ap[:, k], band_name

    # ------------------------------------------------------------------
    def refresh(self):
        if not len(self._idx):
            self._show_error("通道名无法匹配 standard_1005 蒙太奇，无可用电极位置。")
            return
        from mne.viz import plot_topomap

        self.fig.clear()
        try:
            if self.combo_mode.currentIndex() == 0:
                v = self._values_instant(self.spin_t.value())[self._idx]
                ax = self.fig.add_subplot(111)
                im, _ = plot_topomap(v, self._pos, names=self._ch_names, sensors=_SENSORS,
                                     axes=ax, cmap="RdBu_r", show=False, contours=4)
                self.fig.colorbar(im, ax=ax, shrink=0.6, aspect=30)
                ax.set_title(f"瞬时幅值 @ {self.spin_t.value():.2f} s（{self._unit}）",
                             fontsize=12)
            else:
                vals, band = self._values_bandpower(self.combo_band.currentText())
                vals = vals[self._idx]
                ax = self.fig.add_subplot(111)
                im, _ = plot_topomap(vals, self._pos, names=self._ch_names, sensors=_SENSORS,
                                     axes=ax, cmap="RdBu_r", show=False, contours=4)
                self.fig.colorbar(im, ax=ax, shrink=0.6, aspect=30)
                ax.set_title(f"{band} 频带功率（{self._unit}²）", fontsize=12)
            self.canvas.draw()
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            from ..workers import exc_to_zh

            self._show_error(exc_to_zh(e))

    def _show_error(self, text: str):
        """绘制失败时不弹模态框（避免阻塞自动化测试），直接画在画布上。"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, f"绘制失败：\n{text}", ha="center", va="center",
                fontsize=11, color="#c8102e")
        ax.set_axis_off()
        self.canvas.draw()

    def _export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出图片", "topomap.png",
                                              "PNG (*.png);;SVG (*.svg)")
        if not path:
            return
        self.fig.savefig(path, dpi=150)
        QMessageBox.information(self, "完成", f"已导出 {path}")
