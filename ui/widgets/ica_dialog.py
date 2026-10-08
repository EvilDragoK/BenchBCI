"""ICA 成分浏览器：地形图 + 时序 + 诊断指标 + 交互勾选剔除。

用法（预处理工作台）::

    dlg = IcaBrowserDialog(raw_before_ica, params, parent=self)
    if dlg.exec():
        params["exclude"] = dlg.exclude      # 写回流水线参数

设计：把"盲剔编号"变成"看着地形图和指标勾选"——每列成分给出峰度、1 s 自相关、
额区（眼电代理）相关、肌电比、幅值比与方差贡献，并标出自动建议的成分与理由。
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QDialog, QHBoxLayout, QLabel, QListWidget,
                             QListWidgetItem, QMessageBox, QProgressDialog,
                             QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from core.preprocess.artifact import component_features, proxy_channels, suggest_exclude
from .plotting import new_canvas_with_toolbar
from .topo_dialog import _SENSORS, get_eeg_pos
from ..workers import FunctionThread


class IcaBrowserDialog(QDialog):
    """浏览/挑选 ICA 成分。``self.exclude`` 为最终选定的成分编号。"""

    def __init__(self, raw, params: dict, parent=None, exclude=None):
        super().__init__(parent)
        self.setWindowTitle("ICA 成分浏览与挑选")
        self.resize(1180, 760)
        self.raw = raw
        self.params = dict(params or {})
        self.exclude = sorted(int(e) for e in
                              (exclude if exclude is not None else self.params.get("exclude") or []))
        self._threads: list[FunctionThread] = []
        self._ica = None
        self._feats: dict | None = None
        self._suggested: list[int] = []
        self._reasons: dict[int, list[str]] = {}
        self._build()

    # ------------------------------------------------------------ 界面
    def _build(self):
        n_comp = int(self.params.get("n_components", 10))
        self.list_comp = QListWidget()
        self.list_comp.setFixedWidth(170)
        self.list_comp.itemChanged.connect(lambda *_: self._redraw())

        self.btn_fit = QPushButton("重新拟合")
        self.btn_auto = QPushButton("采用自动建议")
        self.btn_none = QPushButton("全不选")
        self.btn_apply = QPushButton("应用并写回流水线")
        self.btn_apply.setProperty("accent", True)
        self.chk_multi = QCheckBox("多选对比")
        self.chk_multi.setChecked(True)
        self.spin_span = QSpinBox()
        self.spin_span.setRange(1, 600)
        self.spin_span.setValue(20)
        self.spin_span.setSuffix(" s")
        self.lbl_status = QLabel("正在拟合 ICA…")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setMinimumWidth(120)

        top = QHBoxLayout()
        top.addWidget(self.btn_fit)
        top.addWidget(self.btn_auto)
        top.addWidget(self.btn_none)
        top.addWidget(QLabel("时序显示"))
        top.addWidget(self.spin_span)
        top.addWidget(self.chk_multi)
        top.addStretch(1)
        top.addWidget(self.btn_apply)

        self.fig_topo, self.canvas_topo, self.tb_topo = new_canvas_with_toolbar(7.2, 5.0)
        self.fig_ts, self.canvas_ts, self.tb_ts = new_canvas_with_toolbar(7.2, 2.6)
        self.tbl = QTableWidget(0, 0)
        self.tbl.setMaximumHeight(190)

        mid = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel(f"成分列表（{n_comp} 个，勾选=剔除）"))
        left.addWidget(self.list_comp, 1)
        left.addWidget(self.lbl_status)
        mid.addLayout(left)
        right = QVBoxLayout()
        right.addWidget(self.tb_topo)
        right.addWidget(self.canvas_topo, 3)
        right.addWidget(self.tb_ts)
        right.addWidget(self.canvas_ts, 2)
        mid.addLayout(right, 1)

        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addLayout(mid, 1)
        root.addWidget(QLabel("诊断指标（判定依据：峰度/幅值比需同时「超阈值 + 成分间离群」）"))
        root.addWidget(self.tbl)

        self.btn_fit.clicked.connect(self.fit)
        self.btn_auto.clicked.connect(self.use_suggestion)
        self.btn_none.clicked.connect(lambda: self._check_all(False))
        self.btn_apply.clicked.connect(self._accept)
        self.spin_span.valueChanged.connect(lambda *_: self._redraw())

    # ------------------------------------------------------------ 拟合
    def fit(self):
        dlg = QProgressDialog("正在拟合 ICA（CPU 上可能需要几十秒）…", "", 0, 0, self)
        dlg.setWindowTitle("请稍候")
        dlg.setCancelButton(None)
        dlg.show()

        raw = self.raw
        p = self.params

        def job(log=print):
            from mne.preprocessing import ICA as MneICA

            method = str(p.get("method", "fastica"))
            if method == "picard":
                try:
                    import picard  # noqa: F401
                except ImportError:
                    method = "fastica"
            ica = MneICA(n_components=int(p.get("n_components", 10)), method=method,
                         random_state=int(p.get("random_state", 97)), max_iter="auto")
            ica.fit(raw, verbose="ERROR")
            feats = component_features(raw, ica)
            return ica, feats

        th = FunctionThread(job, parent=self)
        self._threads.append(th)

        def ok(res):
            dlg.close()
            self._ica, self._feats = res
            self._suggested, self._reasons = suggest_exclude(
                self._feats, min_hits=int(p.get("min_hits", 2)))
            self._fill_list()
            self._fill_table()
            proxy = proxy_channels(raw)
            src = "EOG" if self._feats.get("proxy_source") == "eog" else "额区代理"
            self.lbl_status.setText(
                f"拟合完成：{self._feats['n_components']} 个成分，"
                f"伪迹参考 = {src}（{', '.join(proxy) or '无'}）；"
                f"自动建议剔除 {len(self._suggested)} 个。")

        th.ok.connect(ok)
        th.fail.connect(lambda m: (dlg.close(), QMessageBox.critical(self, "拟合失败", m)))
        th.start()

    # ------------------------------------------------------------ 列表/表格
    def _fill_list(self):
        self.list_comp.blockSignals(True)
        self.list_comp.clear()
        for i in range(int(self._feats["n_components"])):
            mark = "★" if i in self._suggested else " "
            it = QListWidgetItem(f"{mark} IC {i:02d}")
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if i in self.exclude else Qt.CheckState.Unchecked)
            if i in self._reasons:
                it.setToolTip("剔除理由：" + "；".join(self._reasons[i]))
            elif i in self._suggested:
                it.setToolTip("自动建议剔除")
            self.list_comp.addItem(it)
        self.list_comp.blockSignals(False)
        self._redraw()

    def _fill_table(self):
        f = self._feats
        n = int(f["n_components"])
        rows = []
        for i in range(n):
            rows.append([
                f"IC {i:02d}",
                f"{f['kurtosis'][i]:.2f}", f"{f['autocorr'][i]:.2f}",
                f"{f['frontal_corr'][i]:.2f}", f"{f['emg_ratio'][i]:.2f}",
                f"{f['amp_ratio'][i]:.1f}", f"{f['var_frac'][i] * 100:.2f}%",
                "剔除：" + "；".join(self._reasons.get(i, [])) if i in self._reasons else "",
            ])
        self.tbl.clear()
        self.tbl.setColumnCount(8)
        self.tbl.setRowCount(len(rows))
        self.tbl.setHorizontalHeaderLabels(
            ["成分", "峰度", "1s自相关", "额区相关", "肌电比", "幅值比", "方差占比", "自动判定理由"])
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                self.tbl.setItem(r, c, QTableWidgetItem(str(v)))
        self.tbl.resizeColumnsToContents()

    def _checked(self) -> list[int]:
        return [i for i in range(self.list_comp.count())
                if self.list_comp.item(i).checkState() == Qt.CheckState.Checked]

    def _check_all(self, checked: bool):
        st = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.list_comp.blockSignals(True)
        for i in range(self.list_comp.count()):
            self.list_comp.item(i).setCheckState(st)
        self.list_comp.blockSignals(False)
        self._redraw()

    def use_suggestion(self):
        self.list_comp.blockSignals(True)
        for i in range(self.list_comp.count()):
            self.list_comp.item(i).setCheckState(
                Qt.CheckState.Checked if i in self._suggested else Qt.CheckState.Unchecked)
        self.list_comp.blockSignals(False)
        self._redraw()

    # ------------------------------------------------------------ 绘图
    def _redraw(self):
        if self._ica is None:
            return
        sel = self._checked()
        if not sel:
            sel = [0]
        sel = sel[:12] if self.chk_multi.isChecked() else sel[-1:]
        self._draw_topo(sel)
        self._draw_ts(sel)

    def _draw_topo(self, sel: list[int]):
        import mne

        fig = self.fig_topo
        fig.clear()
        topo = np.asarray(self._ica.get_components(), dtype=float)   # (n_ch, n_comp)
        try:
            # get_eeg_pos 返回的 idx/pos/names 对齐于「EEG 通道列表」，需映射回 raw 的通道行
            _, pos, names = get_eeg_pos(self.raw)
            picks = mne.pick_types(self.raw.info, eeg=True, exclude=[])
            eeg_names = [self.raw.ch_names[i] for i in picks]
            rows = [picks[eeg_names.index(n)] for n in names]
            topo = topo[rows]
        except Exception as e:  # noqa: BLE001
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"无法获取电极坐标：{e}", ha="center")
            self.canvas_topo.draw()
            return
        n = len(sel)
        ncol = min(n, 4)
        nrow = int(np.ceil(n / ncol))
        from mne.viz import plot_topomap

        vmax = float(np.abs(topo[:, sel]).max()) or 1.0
        for k, comp in enumerate(sel):
            ax = fig.add_subplot(nrow, ncol, k + 1)
            im, _ = plot_topomap(topo[:, comp], pos, names=names, sensors=_SENSORS,
                                 axes=ax, cmap="RdBu_r", show=False, contours=4,
                                 vlim=(-vmax, vmax))
            ax.set_title(f"IC {comp:02d}" + ("（剔除）" if comp in self._checked() else ""),
                         fontsize=9)
            if k == 0:
                fig.colorbar(im, ax=ax, shrink=0.7, aspect=30)
        fig.suptitle("ICA 成分头皮分布（地形图）", fontsize=11)
        self.canvas_topo.draw()

    def _draw_ts(self, sel: list[int]):
        fig = self.fig_ts
        fig.clear()
        src = self._ica.get_sources(self.raw).get_data()             # (n_comp, T)
        fs = float(self.raw.info["sfreq"])
        span = int(self.spin_span.value() * fs)
        span = min(max(span, 8), src.shape[-1])
        t = np.arange(span) / fs
        ax = fig.add_subplot(111)
        for comp in sel:
            y = src[comp, :span]
            scale = np.abs(y).max() or 1.0
            ax.plot(t, y / scale + comp, lw=0.7,
                    color="crimson" if comp in self._checked() else "black")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("成分（按序号堆叠，各自归一化）")
        ax.set_title(f"ICA 成分时序（前 {t[-1]:.1f} s，红=已勾选剔除）")
        ax.grid(alpha=0.3)
        self.canvas_ts.draw()

    # ------------------------------------------------------------ 结果
    def _accept(self):
        self.exclude = sorted(self._checked())
        self.accept()

    def closeEvent(self, e):  # noqa: N802 - Qt 命名
        self.exclude = sorted(self._checked())
        super().closeEvent(e)


def browse_ica(raw, params: dict, parent=None, exclude=None) -> list[int] | None:
    """便捷入口：返回选定的剔除成分列表（取消则返回 None）。"""
    dlg = IcaBrowserDialog(raw, params, parent=parent, exclude=exclude)
    dlg.fit()
    return dlg.exclude if dlg.exec() else None
