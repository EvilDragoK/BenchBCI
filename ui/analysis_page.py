"""信号分析页：频域 / ERP 时域 / 时频 / 非线性熵 / 脑网络。"""
from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                             QDoubleSpinBox, QHBoxLayout, QLabel, QListWidget,
                             QListWidgetItem, QMessageBox, QProgressDialog,
                             QPushButton, QSpinBox, QTabWidget, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from core.segment import make_epochs
from core.erp import (DEFAULT_COMPONENTS, baseline_correct, component_condition_stats,
                      component_peak_values, erp_average, erp_conditions,
                      extract_components, single_trial_components)
from core.features import (BANDS, PSD_LABELS, PSD_METHODS, PSD_SHORT, absolute_power,
                           psd, relative_power)
from core.stats import effect_magnitude, group_report, stars
from core.timefreq import (BASELINE_LABELS, BASELINE_MODES, BASELINE_SHORT, TFR_LABELS,
                           TFR_METHODS, TFR_SHORT, band_tfr_power, band_window_topomap,
                           compute_tfr, tfr_baseline, tfr_channel_stats, tfr_to_rows)
from core.nonlinear import entropy_channels
from core.networks import (CONNECT_LABELS, CONNECT_METHODS, SPARSIFY_LABELS,
                           SPARSIFY_STRATEGIES, binarize, connectivity_matrix,
                           graph_metrics)
from core.brainnet import (biomarkers_by_condition, connection_variability,
                           dynamic_connectivity, dynamic_features)
from .widgets.brain3d_dialog import Brain3DDialog
from .widgets.plotting import new_canvas, new_canvas_with_toolbar
from .widgets.topo_dialog import _SENSORS, get_eeg_pos
from .workers import FunctionThread, exc_to_zh

CH_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


class AnalysisPage(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._threads: list[FunctionThread] = []
        self._ch: list[str] = []
        self._erp_full_wave: np.ndarray | None = None

        # ---- 顶部分段参数 ----
        top = QHBoxLayout()
        top.addWidget(QLabel("分段 tmin"))
        self.spin_tmin = QDoubleSpinBox()
        self.spin_tmin.setRange(-2.0, 5.0)
        self.spin_tmin.setValue(0.5)
        self.spin_tmin.setSuffix(" s")
        top.addWidget(self.spin_tmin)
        top.addWidget(QLabel("tmax"))
        self.spin_tmax = QDoubleSpinBox()
        self.spin_tmax.setRange(-1.0, 8.0)
        self.spin_tmax.setValue(2.5)
        self.spin_tmax.setSuffix(" s")
        top.addWidget(self.spin_tmax)
        top.addWidget(QLabel("最多 epochs"))
        self.spin_max = QSpinBox()
        self.spin_max.setRange(10, 10000)
        self.spin_max.setValue(200)
        top.addWidget(self.spin_max)
        self.lbl_hint = QLabel("提示：以下分析基于当前已加载数据（EEG 通道，µV）。")
        self.lbl_hint.setStyleSheet("color:gray;")
        top.addWidget(self.lbl_hint)
        top.addStretch(1)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_freq(), "频域分析")
        self.tabs.addTab(self._build_erp(), "ERP 时域成分")
        self.tabs.addTab(self._build_tfr(), "时频分析")
        self.tabs.addTab(self._build_nonlinear(), "非线性熵")
        self.tabs.addTab(self._build_net(), "脑网络")

        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addWidget(self.tabs, 1)

    # ================================================================ 通用
    def _require_raw(self) -> bool:
        if self.state.raw is None:
            QMessageBox.information(self, "提示", "请先在「数据管理」页加载数据")
            return False
        return True

    def _segments(self, tmin=None, tmax=None, max_epochs=None, baseline=None):
        import mne

        raw = self.state.raw
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        raw_eeg = raw.copy().pick(list(picks)) if picks.size else raw
        tmin = self.spin_tmin.value() if tmin is None else tmin
        tmax = self.spin_tmax.value() if tmax is None else tmax
        n = self.spin_max.value() if max_epochs is None else max_epochs
        X, labels, times = make_epochs(raw_eeg, tmin, tmax, n, baseline=baseline)
        return X * 1e6, labels, times, raw_eeg  # → µV

    def _ensure_channels(self):
        """从当前数据填充通道下拉（时频等 tab 复用）。"""
        if self.combo_tfr_ch.count() == 0 and self.state.raw is not None:
            import mne

            raw = self.state.raw
            picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            self._ch = [raw.ch_names[i] for i in picks] if picks.size else list(raw.ch_names)
            self.combo_tfr_ch.addItems(self._ch)

    def _busy(self, text):
        dlg = QProgressDialog(text, "", 0, 0, self)
        dlg.setWindowTitle("请稍候")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setCancelButton(None)
        dlg.show()
        QApplication.processEvents()
        return dlg

    def _run(self, fn, on_ok, text):
        dlg = self._busy(text)
        th = FunctionThread(fn, parent=self)
        self._threads.append(th)
        th.ok.connect(lambda res: (dlg.close(), on_ok(res)))
        th.fail.connect(lambda m: (dlg.close(), QMessageBox.critical(self, "计算失败", m)))
        th.start()

    @staticmethod
    def _fill_table(tbl: QTableWidget, headers, rows):
        tbl.clear()
        tbl.setColumnCount(len(headers))
        tbl.setRowCount(len(rows))
        tbl.setHorizontalHeaderLabels(headers)
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                tbl.setItem(i, j, QTableWidgetItem(str(v)))

    # ================================================================ 频域
    def _build_freq(self):
        w = QWidget()
        self.btn_freq = QPushButton("计算通道频谱与频段能量")
        self.btn_freq_all = QPushButton("全选")
        self.btn_freq_none = QPushButton("全不选")
        self.btn_freq_export = QPushButton("导出 CSV")
        self.btn_freq_stats = QPushButton("逐通道统计检验…")
        self.btn_freq_stats.setToolTip(
            "对每个频带做逐通道的条件间检验（Welch/周期图/多窗功率），\n"
            "输出 t、p、FDR 校正后 p 与 Cohen's d，可直接写进论文")
        self.combo_psd = QComboBox()
        self.combo_psd.setToolTip("\n".join(f"{PSD_SHORT[k]}：{PSD_LABELS[k]}"
                                            for k in PSD_METHODS))
        for key in PSD_METHODS:                    # 短标签：长文本会把下拉框撑宽
            self.combo_psd.addItem(PSD_SHORT.get(key, key), userData=key)
        self.chk_rel = QCheckBox("相对功率 (%)")
        self.chk_rel.setToolTip("勾选后频带表显示相对功率（各频带占比，和为 100%）")
        self.list_freq_ch = QListWidget()
        self.list_freq_ch.setFixedWidth(150)
        self.fig_freq, self.canvas_freq, self.tb_freq = new_canvas_with_toolbar(8, 4.5)
        self.tbl_freq = QTableWidget(0, 0)

        top = QHBoxLayout()
        top.addWidget(self.btn_freq)
        top.addWidget(QLabel("谱方法"))
        top.addWidget(self.combo_psd)
        top.addWidget(self.chk_rel)
        top.addWidget(self.btn_freq_all)
        top.addWidget(self.btn_freq_none)
        top.addWidget(self.btn_freq_export)
        top.addWidget(self.btn_freq_stats)
        top.addStretch(1)
        body = QHBoxLayout()
        body.addWidget(self.list_freq_ch)
        body.addWidget(self.canvas_freq, 1)
        lay = QVBoxLayout(w)
        lay.addLayout(top)
        lay.addWidget(self.tb_freq)
        lay.addLayout(body, 1)
        lay.addWidget(self.tbl_freq)

        self.btn_freq.clicked.connect(self.compute_freq)
        self.btn_freq_all.clicked.connect(lambda: self._set_all_freq(True))
        self.btn_freq_none.clicked.connect(lambda: self._set_all_freq(False))
        self.btn_freq_export.clicked.connect(self._export_freq)
        self.btn_freq_stats.clicked.connect(self._freq_stats)
        self.chk_rel.toggled.connect(lambda *_: self._redraw_freq())
        self.list_freq_ch.itemChanged.connect(lambda *_: self._redraw_freq())
        return w

    def _set_all_freq(self, checked: bool):
        st = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.list_freq_ch.blockSignals(True)
        for i in range(self.list_freq_ch.count()):
            self.list_freq_ch.item(i).setCheckState(st)
        self.list_freq_ch.blockSignals(False)
        self._redraw_freq()

    def _psd_method(self) -> str:
        return self.combo_psd.currentData() or "welch"

    def compute_freq(self):
        if not self._require_raw():
            return
        method = self._psd_method()

        def job(log=print):
            X, _, _, raw_eeg = self._segments()
            ch = raw_eeg.ch_names
            fs = float(raw_eeg.info["sfreq"])
            f, p = psd(X, fs, method=method)              # (F,), (N,C,F) µV²/Hz
            psd_mean = p.mean(axis=0)                     # (C,F) µV²
            ap = absolute_power(X, fs).mean(axis=0)       # (C,B) µV²
            rel = relative_power(X, fs).mean(axis=0)      # (C,B) %
            return ch, f, psd_mean, ap, rel

        self._run(job, self._freq_done, f"正在计算频谱（{method}）…")

    def _freq_done(self, res):
        ch, f, psd_mean, ap, rel = res
        self._ch, self._freq_f, self._freq_psd = ch, f, psd_mean
        self._freq_ap, self._freq_rel = ap, rel
        self.list_freq_ch.blockSignals(True)
        self.list_freq_ch.clear()
        for name in ch:
            it = QListWidgetItem(name)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            self.list_freq_ch.addItem(it)
        self.list_freq_ch.blockSignals(False)
        self._fill_freq_table()
        self._redraw_freq()

    def _fill_freq_table(self):
        ch, ap, rel = self._ch, self._freq_ap, self._freq_rel
        bands = list(BANDS)
        if self.chk_rel.isChecked():
            rows = [[ch[i]] + [f"{rel[i, b]:.2f}" for b in range(len(bands))]
                    for i in range(len(ch))]
            self._fill_table(self.tbl_freq, ["通道"] + [b + " (%)" for b in bands], rows)
        else:
            rows = [[ch[i]] + [f"{ap[i, b]:.3g}" for b in range(len(bands))]
                    + [f"{ap[i].sum():.3g}"] for i in range(len(ch))]
            self._fill_table(self.tbl_freq, ["通道"] + bands + ["总功率"], rows)

    def _redraw_freq(self):
        if not hasattr(self, "_freq_psd"):
            return
        f, psd = self._freq_f, self._freq_psd
        if self.chk_rel.isChecked():
            self._fill_freq_table()
        sel = [i for i in range(self.list_freq_ch.count())
               if self.list_freq_ch.item(i).checkState() == Qt.CheckState.Checked]
        fig = self.fig_freq
        fig.clear()
        ax = fig.add_subplot(111)
        for n, i in enumerate(sel):
            ax.plot(f, 10 * np.log10(psd[i] + 1e-30), lw=0.8,
                    color=CH_COLORS[n % len(CH_COLORS)], label=self._ch[i])
        if sel:
            ax.plot(f, 10 * np.log10(psd[sel].mean(0) + 1e-30), "k", lw=2.2,
                    label="平均（勾选）")
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel("功率 (dB, µV²/Hz)")
        ax.set_title(f"逐通道功率谱（{PSD_LABELS.get(self._psd_method(), '')}）")
        ax.grid(alpha=0.3)
        if sel:
            ax.legend(fontsize=6, ncol=2, loc="upper right")
        self.canvas_freq.draw()

    # ================================================================ ERP
    def _build_erp(self):
        w = QWidget()
        row = QHBoxLayout()
        row.addWidget(QLabel("条件"))
        self.combo_cond = QComboBox()
        row.addWidget(self.combo_cond)
        row.addWidget(QLabel("通道"))
        self.combo_erp_ch = QComboBox()
        row.addWidget(self.combo_erp_ch)
        self.btn_erp = QPushButton("计算 ERP 与成分")
        self.btn_erp_topo = QPushButton("成分地形图")
        self.btn_erp_stats = QPushButton("统计分析…")
        self.btn_erp_stats.setToolTip(
            "逐试次提取各成分的幅值/潜伏期 → 条件间检验\n（t 检验 + FDR 校正 + Cohen's d）")
        row.addWidget(self.btn_erp)
        row.addWidget(self.btn_erp_topo)
        row.addWidget(self.btn_erp_stats)
        self.chk_erp_base = QCheckBox("基线校正")
        self.chk_erp_base.setChecked(True)
        self.chk_erp_base.setToolTip("tmin<0 时用 tmin~0 段做基线；tmin≥0 时用前 200 ms 近似")
        self.chk_erp_multi = QCheckBox("多条件对比")
        self.chk_erp_multi.setToolTip("勾选后所有条件的波形叠在同一张图里对比")
        row.addWidget(self.chk_erp_base)
        row.addWidget(self.chk_erp_multi)
        row.addStretch(1)
        # 第二行：时间窗（单独一行，避免控件过多被挤出窗口）
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("ERP tmin"))
        self.spin_erp_tmin = QDoubleSpinBox()
        self.spin_erp_tmin.setRange(-1.0, 2.0)
        self.spin_erp_tmin.setValue(0.0)
        self.spin_erp_tmin.setSuffix(" s")
        row2.addWidget(self.spin_erp_tmin)
        row2.addWidget(QLabel("tmax"))
        self.spin_erp_tmax = QDoubleSpinBox()
        self.spin_erp_tmax.setRange(0.2, 3.0)
        self.spin_erp_tmax.setValue(0.6)
        self.spin_erp_tmax.setSuffix(" s")
        row2.addWidget(self.spin_erp_tmax)
        row2.addStretch(1)

        self.fig_erp, self.canvas_erp, self.tb_erp = new_canvas_with_toolbar(8, 4.5)
        self.tbl_erp = QTableWidget(0, 0)
        lay = QVBoxLayout(w)
        lay.addLayout(row)
        lay.addLayout(row2)
        lay.addWidget(self.tb_erp)
        lay.addWidget(self.canvas_erp, 1)
        lay.addWidget(self.tbl_erp)
        self.btn_erp.clicked.connect(self.compute_erp)
        self.btn_erp_topo.clicked.connect(self.show_erp_topo)
        self.btn_erp_stats.clicked.connect(self._erp_stats)
        self.combo_cond.currentIndexChanged.connect(lambda *_: self._erp_draw())
        self.combo_erp_ch.currentIndexChanged.connect(lambda *_: self._erp_draw())
        self.chk_erp_multi.toggled.connect(lambda *_: self._erp_draw())
        return w

    def compute_erp(self):
        if not self._require_raw():
            return
        use_base = self.chk_erp_base.isChecked()

        def job(log=print):
            t0, t1 = self.spin_erp_tmin.value(), self.spin_erp_tmax.value()
            base = (None, 0) if (use_base and t0 < 0) else None
            X, labels, times, raw_eeg = self._segments(tmin=t0, tmax=t1, baseline=base)
            if use_base and t0 >= 0:      # 窗口全在刺激之后 → 用前 200 ms 近似基线
                X = baseline_correct(X, times, baseline=(times[0], times[0] + 0.2))
            ch = raw_eeg.ch_names
            conds = ["all"] + sorted(set(labels))
            return X, labels, times, ch, conds

        self._run(job, self._erp_seg_done, "正在分段…")

    def _erp_seg_done(self, res):
        X, labels, times, ch, conds = res
        self._erp_X, self._erp_labels, self._erp_times = X, labels, times
        self._ch = ch
        self.combo_cond.blockSignals(True)
        self.combo_cond.clear()
        self.combo_cond.addItems(conds)
        self.combo_cond.blockSignals(False)
        self.combo_erp_ch.blockSignals(True)
        self.combo_erp_ch.clear()
        self.combo_erp_ch.addItem("全通道平均")
        self.combo_erp_ch.addItems(ch)
        self.combo_erp_ch.blockSignals(False)
        self._erp_draw()

    def _erp_draw(self):
        if not hasattr(self, "_erp_X"):
            return
        cond = self.combo_cond.currentText()
        try:
            wave = erp_average(self._erp_X, self._erp_labels, cond)
        except ValueError as e:
            QMessageBox.warning(self, "提示", exc_to_zh(e))
            return
        self._erp_full_wave = wave
        ch = self.combo_erp_ch.currentText()
        ci = self._ch.index(ch) if (ch and ch != "全通道平均") else None

        def _pick(w):
            return w[ci] if ci is not None else w.mean(axis=0)

        times = self._erp_times
        fig, canvas = self.fig_erp, self.canvas_erp
        fig.clear()
        ax = fig.add_subplot(111)

        if self.chk_erp_multi.isChecked():          # 多条件叠加对比
            try:
                waves, counts = erp_conditions(self._erp_X, self._erp_labels)
            except ValueError as e:
                QMessageBox.warning(self, "提示", exc_to_zh(e))
                return
            for n, (name, wv) in enumerate(waves.items()):
                ax.plot(times, _pick(wv), lw=1.5, label=f"{name} (n={counts[name]})",
                        color=CH_COLORS[n % len(CH_COLORS)])
            ax.legend(fontsize=8)
        else:
            ax.plot(times, _pick(wave), "k", lw=1.6)
        ax.axhline(0, color="gray", lw=0.6)

        disp = _pick(wave)                          # 成分表基于当前选中条件
        comps = extract_components(times, disp[None, :])
        rows = []
        for name in DEFAULT_COMPONENTS:
            lat = comps[name]["latency_ms"][0]
            amp = comps[name]["amplitude"][0]
            ax.plot(lat / 1000.0, amp, "o", color="crimson")
            ax.annotate(f"{name} {lat:.0f}ms", (lat / 1000.0, amp),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=8, color="crimson", ha="center")
            rows.append([name, f"{lat:.1f}", f"{amp:.3f}"])
        ax.set_xlabel("时间 (s，相对 cue/刺激)")
        ax.set_ylabel("幅值 (µV)")
        title = f"ERP 叠加平均（条件={cond}，通道={ch}）"
        if self.chk_erp_base.isChecked():
            title += "（已基线校正）"
        ax.set_title(title)
        ax.grid(alpha=0.3)
        canvas.draw()
        self._fill_table(self.tbl_erp, ["成分", "潜伏期 (ms)", "幅值 (µV)"], rows)

    def show_erp_topo(self):
        if self._erp_full_wave is None:
            QMessageBox.information(self, "提示", "请先计算 ERP")
            return
        try:
            _, pos, names = get_eeg_pos(self.state.raw)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "提示", exc_to_zh(e))
            return
        peaks = component_peak_values(self._erp_times, self._erp_full_wave)
        idx = [i for i in range(len(self._ch)) if self._ch[i] in names]
        from mne.viz import plot_topomap

        fig, canvas = new_canvas(10, 4)
        comps = list(DEFAULT_COMPONENTS)
        for k, name in enumerate(comps):
            v = peaks[name][idx]
            ax = fig.add_subplot(1, 4, k + 1)
            im, _ = plot_topomap(v, pos, names=names, sensors=_SENSORS, axes=ax,
                                 cmap="RdBu_r", show=False, contours=4)
            fig.colorbar(im, ax=ax, shrink=0.6, aspect=30)
            ax.set_title(name, fontsize=11)
        canvas.draw()
        from PyQt6.QtWidgets import QDialog, QVBoxLayout

        d = QDialog(self)
        d.setWindowTitle("ERP 成分地形图（各成分峰值时刻）")
        lay = QVBoxLayout(d)
        lay.addWidget(canvas)
        d.resize(1000, 400)
        d.exec()

    # ================================================================ 时频
    def _build_tfr(self):
        w = QWidget()
        row = QHBoxLayout()
        row.addWidget(QLabel("通道"))
        self.combo_tfr_ch = QComboBox()
        row.addWidget(self.combo_tfr_ch)
        row.addWidget(QLabel("方法"))
        self.combo_tfr_method = QComboBox()
        self.combo_tfr_method.setToolTip("\n".join(f"{TFR_SHORT[k]}：{TFR_LABELS[k]}"
                                                   for k in TFR_METHODS))
        for key in TFR_METHODS:
            self.combo_tfr_method.addItem(TFR_SHORT.get(key, key), userData=key)
        row.addWidget(self.combo_tfr_method)
        row.addWidget(QLabel("归一化"))
        self.combo_tfr_norm = QComboBox()
        self.combo_tfr_norm.setToolTip(
            "ERD/ERS 基线归一化方式：\n"
            + "\n".join(f"{BASELINE_SHORT[k]}：{BASELINE_LABELS[k]}" for k in BASELINE_MODES))
        for key in BASELINE_MODES:
            self.combo_tfr_norm.addItem(BASELINE_SHORT.get(key, key), userData=key)
        self.combo_tfr_norm.setCurrentIndex(BASELINE_MODES.index("ers"))
        row.addWidget(self.combo_tfr_norm)
        row.addWidget(QLabel("频段"))
        self.combo_tfr_band = QComboBox()
        for key in BANDS:
            self.combo_tfr_band.addItem(f"{key} ({BANDS[key][0]:g}-{BANDS[key][1]:g} Hz)",
                                        userData=key)
        self.combo_tfr_band.setCurrentIndex(list(BANDS).index("alpha"))
        row.addWidget(self.combo_tfr_band)
        self.btn_tfr = QPushButton("计算时频图")
        self.btn_tfr_topo = QPushButton("频段地形图…")
        self.btn_tfr_topo.setToolTip("「频段 × 时间窗」平均功率的头皮分布")
        self.btn_tfr_stats = QPushButton("统计检验…")
        self.btn_tfr_stats.setToolTip("逐通道检验「检验窗 vs 基线」是否显著（t/置换 + FDR）")
        self.btn_tfr_export = QPushButton("导出 CSV")
        row.addWidget(self.btn_tfr)
        row.addStretch(1)

        # 第二行：时间窗 + 按钮（控件多，分两行才不会被挤出窗口）
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("基线窗"))
        self.spin_base0, self.spin_base1 = QDoubleSpinBox(), QDoubleSpinBox()
        for sp, v in ((self.spin_base0, -0.5), (self.spin_base1, 0.0)):
            sp.setRange(-3.0, 3.0)
            sp.setSingleStep(0.1)
            sp.setValue(v)
            sp.setSuffix(" s")
            row2.addWidget(sp)
        row2.addWidget(QLabel("检验窗"))
        self.spin_test0, self.spin_test1 = QDoubleSpinBox(), QDoubleSpinBox()
        for sp, v in ((self.spin_test0, 0.5), (self.spin_test1, 1.5)):
            sp.setRange(-3.0, 5.0)
            sp.setSingleStep(0.1)
            sp.setValue(v)
            sp.setSuffix(" s")
            row2.addWidget(sp)
        row2.addWidget(self.btn_tfr_topo)
        row2.addWidget(self.btn_tfr_stats)
        row2.addWidget(self.btn_tfr_export)
        row2.addStretch(1)

        self.fig_tfr, self.canvas_tfr, self.tb_tfr = new_canvas_with_toolbar(8, 5)
        lay = QVBoxLayout(w)
        lay.addLayout(row)
        lay.addLayout(row2)
        lay.addWidget(self.tb_tfr)
        lay.addWidget(self.canvas_tfr, 1)
        self.btn_tfr.clicked.connect(self.compute_tfr)
        self.btn_tfr_topo.clicked.connect(self._tfr_topo)
        self.btn_tfr_stats.clicked.connect(self._tfr_stats)
        self.btn_tfr_export.clicked.connect(self._tfr_export)
        self.combo_tfr_norm.currentIndexChanged.connect(lambda *_: self._tfr_plot())
        self.combo_tfr_ch.currentIndexChanged.connect(lambda *_: self._tfr_plot())
        return w

    def _tfr_band(self) -> tuple:
        key = self.combo_tfr_band.currentData() or "alpha"
        return tuple(BANDS[key])

    def compute_tfr(self):
        if not self._require_raw():
            return
        self._ensure_channels()
        if self.combo_tfr_ch.count() == 0:
            QMessageBox.information(self, "提示", "未找到可用通道")
            return
        method = self.combo_tfr_method.currentData() or "morlet"

        def job(log=print):
            X, _, times, raw_eeg = self._segments()
            self._ch = raw_eeg.ch_names
            fs = float(raw_eeg.info["sfreq"])
            P, f, t = compute_tfr(X, fs, times=times, method=method)
            return P, f, t, fs

        self._run(job, self._tfr_done, f"正在计算时频（{method}）…")

    def _tfr_done(self, res):
        self._tfr_P, self._tfr_f, self._tfr_t, self._tfr_fs = res
        self._tfr_plot()

    def _tfr_plot(self):
        if not hasattr(self, "_tfr_P"):
            return
        ch = self.combo_tfr_ch.currentText()
        if ch not in self._ch:
            return
        ci = self._ch.index(ch)
        norm = self.combo_tfr_norm.currentData() or "none"
        base = (self.spin_base0.value(), self.spin_base1.value())
        P = self._tfr_P[:, ci:ci + 1]
        t = self._tfr_t
        try:
            V = tfr_baseline(P, t, baseline=base, mode=norm).mean(axis=0)[0]   # (F,T)
            label = BASELINE_LABELS.get(norm, norm)
        except ValueError as e:              # 基线窗与数据不重叠 → 退化为原始 dB
            QMessageBox.warning(self, "提示", exc_to_zh(e))
            V = 10 * np.log10(P.mean(axis=0)[0] + 1e-30)
            label = "dB"
        method = self.combo_tfr_method.currentData() or "morlet"
        fig, canvas = self.fig_tfr, self.canvas_tfr
        fig.clear()
        ax = fig.add_subplot(111)
        im = ax.pcolormesh(t, self._tfr_f, V, shading="auto", cmap="RdBu_r"
                           if norm != "none" else "jet")
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("频率 (Hz)")
        ax.set_title(f"{TFR_LABELS.get(method, method)}（{ch}，epoch 平均，{label}）")
        canvas.draw()

    # ========================================================= 导出与统计（共用）
    def _pick_save_path(self, default_name: str, title: str = "导出 CSV"):
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(self, title, default_name, "CSV (*.csv)")
        return path or None

    def _write_csv(self, path: str, header, rows) -> int:
        import csv

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(list(header))
            w.writerows([list(r) for r in rows])
        return len(rows)

    def _table_dialog(self, title: str, headers, rows, note: str = "") -> QDialog:
        """统一的结果表对话框（带导出 CSV）。"""
        tbl = QTableWidget(0, 0)
        self._fill_table(tbl, headers, rows)
        d = QDialog(self)
        d.setWindowTitle(title)
        lay = QVBoxLayout(d)
        if note:
            lb = QLabel(note)
            lb.setWordWrap(True)
            lb.setMinimumWidth(120)     # 长中文说明允许换行（否则会撑宽对话框）
            lay.addWidget(lb)
        lay.addWidget(tbl, 1)
        btn = QPushButton("导出 CSV")
        btn.setProperty("accent", True)
        export_name = title.replace(" ", "_").replace("…", "") + ".csv"

        def _do_export():
            p = self._pick_save_path(export_name, "导出统计结果")
            if p:
                n = self._write_csv(p, headers, rows)
                self.lbl_hint.setText(f"已导出 {n} 行 → {p}")

        btn.clicked.connect(_do_export)
        lay.addWidget(btn)
        d.resize(1000, 620)
        return d

    @staticmethod
    def _p_star(p) -> str:
        return str(stars(np.asarray([float(p)]))[0])

    # ---------------------------------------------------------- 频域导出/统计
    def _export_freq(self):
        if not hasattr(self, "_freq_psd"):
            QMessageBox.information(self, "提示", "请先「计算通道频谱与频段能量」")
            return
        from pathlib import Path as _Path

        from core.features import psd_table

        p = self._pick_save_path("psd.csv", "导出频谱与频带功率")
        if not p:
            return
        stem = str(_Path(p).with_suffix(""))
        h, r = psd_table(self._freq_f, self._freq_psd, self._ch)      # 逐频点 dB
        n1 = self._write_csv(f"{stem}_spectrum.csv", h, r)
        bands = list(BANDS)
        rows = [[self._ch[i]]
                + [f"{self._freq_ap[i, b]:.6g}" for b in range(len(bands))]
                + [f"{self._freq_rel[i, b]:.3f}" for b in range(len(bands))]
                for i in range(len(self._ch))]
        n2 = self._write_csv(f"{stem}_bands.csv",
                             ["通道"] + [f"{b} (µV²)" for b in bands]
                             + [f"{b} (%)" for b in bands], rows)
        self.lbl_hint.setText(
            f"已导出 → {stem}_spectrum.csv（{n1} 行）、{stem}_bands.csv（{n2} 行）")

    def _freq_stats(self):
        if not self._require_raw():
            return

        def job(log=print):
            X, labels, _, raw_eeg = self._segments()
            fs = float(raw_eeg.info["sfreq"])
            ch = raw_eeg.ch_names
            conds = sorted(set(labels))
            if len(conds) < 2:
                raise ValueError(f"统计检验需要至少 2 个条件（当前只有 {conds}）")
            a, b = conds[0], conds[1]
            ma = np.array([l == a for l in labels])
            mb = np.array([l == b for l in labels])
            ap = absolute_power(X, fs)                     # (N,C,B) µV²
            rows = []
            for bi, band in enumerate(BANDS):
                va = np.moveaxis(ap[ma][:, :, bi], 0, -1)  # (C, n_a)
                vb = np.moveaxis(ap[mb][:, :, bi], 0, -1)
                rep = group_report(va, vb, paired=False, correct_method="fdr_bh")
                for c in range(len(ch)):
                    rows.append([band, ch[c], f"{rep['mean_a'][c]:.5g}", f"{rep['mean_b'][c]:.5g}",
                                 f"{rep['t'][c]:.2f}", f"{rep['p_corrected'][c]:.4g}",
                                 self._p_star(rep["p_corrected"][c]), f"{rep['d'][c]:.2f}",
                                 str(effect_magnitude(rep["d"][c]))])
            return rows, (a, b), int(ma.sum()), int(mb.sum())

        self._run(job, self._freq_stats_done, "正在做逐通道频带功率检验…")

    def _freq_stats_done(self, res):
        rows, (a, b), na, nb = res
        d = self._table_dialog(
            f"逐通道频带功率检验（{a} vs {b}）",
            ["频带", "通道", f"均值[{a}]", f"均值[{b}]", "t", "p(FDR)", "显著性", "Cohen's d", "效应"],
            rows,
            note=f"独立样本 t 检验 + FDR 校正；n[{a}]={na}，n[{b}]={nb}。"
                 f"单位 µV²；*** p<0.001，** p<0.01，* p<0.05。")
        d.exec()

    # ---------------------------------------------------------- ERP 统计
    def _erp_stats(self):
        if not self._require_raw():
            return

        def job(log=print):
            t0, t1 = self.spin_erp_tmin.value(), self.spin_erp_tmax.value()
            X, labels, times, raw_eeg = self._segments(tmin=t0, tmax=t1)
            ch = raw_eeg.ch_names
            res = component_condition_stats(times, X, labels, paired=False,
                                            correct_method="fdr_bh")
            rows = []
            for comp, rep in res["components"].items():
                for metric, label in (("amplitude", "幅值(µV)"), ("latency", "潜伏期(ms)")):
                    r = rep[metric]
                    for c in range(len(ch)):
                        rows.append([comp, label, ch[c],
                                     f"{r['mean_a'][c]:.4g}", f"{r['mean_b'][c]:.4g}",
                                     f"{r['t'][c]:.2f}", f"{r['p_corrected'][c]:.4g}",
                                     self._p_star(r["p_corrected"][c]), f"{r['d'][c]:.2f}"])
            return rows, res["conds"], res["n"]

        self._run(job, self._erp_stats_done, "正在做 ERP 成分统计检验…")

    def _erp_stats_done(self, res):
        rows, (a, b), n = res
        d = self._table_dialog(
            f"ERP 成分统计检验（{a} vs {b}）",
            ["成分", "指标", "通道", f"均值[{a}]", f"均值[{b}]", "t", "p(FDR)", "显著性", "Cohen's d"],
            rows,
            note=f"逐试次提取成分幅值/潜伏期后做独立样本 t 检验 + FDR 校正；"
                 f"n[{a}]={n.get(a)}，n[{b}]={n.get(b)}。")
        d.exec()

    # ---------------------------------------------------------- 时频地形图/统计/导出
    def _tfr_topo(self):
        if not hasattr(self, "_tfr_P"):
            QMessageBox.information(self, "提示", "请先「计算时频图」")
            return
        band = self._tfr_band()
        t0, t1 = self.spin_test0.value(), self.spin_test1.value()
        try:
            V = band_window_topomap(self._tfr_P, self._tfr_f, self._tfr_t,
                                    band=band, tmin=t0, tmax=t1).mean(axis=0)
        except ValueError as e:
            QMessageBox.warning(self, "提示", exc_to_zh(e))
            return
        try:
            _, pos, names = get_eeg_pos(self.state.raw)
            idx = [i for i in range(len(self._ch)) if self._ch[i] in names]
            v = V[idx]
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "提示", exc_to_zh(e))
            return
        from mne.viz import plot_topomap

        fig, canvas = new_canvas(4.6, 4.0)
        ax = fig.add_subplot(111)
        im, _ = plot_topomap(v, pos, names=names, sensors=_SENSORS, axes=ax,
                             cmap="RdBu_r", show=False, contours=4)
        fig.colorbar(im, ax=ax, shrink=0.8, aspect=30, label="功率 (µV²)")
        ax.set_title(f"{band[0]:g}-{band[1]:g} Hz，{t0:.2f}~{t1:.2f} s 平均功率")
        canvas.draw()
        d = QDialog(self)
        d.setWindowTitle("时频地形图（频段 × 时间窗）")
        lay = QVBoxLayout(d)
        lay.addWidget(canvas)
        d.resize(560, 520)
        d.exec()

    def _tfr_stats(self):
        if not hasattr(self, "_tfr_P"):
            QMessageBox.information(self, "提示", "请先「计算时频图」")
            return
        P, f, t, ch = self._tfr_P, self._tfr_f, self._tfr_t, self._ch
        band = self._tfr_band()
        b0, b1 = self.spin_base0.value(), self.spin_base1.value()
        t0, t1 = self.spin_test0.value(), self.spin_test1.value()

        def job(log=print):
            s = tfr_channel_stats(P, t, f, test_win=(t0, t1), baseline=(b0, b1), band=band)
            rows = [[ch[c], f"{s['base_mean'][c]:.5g}", f"{s['test_mean'][c]:.5g}",
                     f"{s['test_mean'][c] - s['base_mean'][c]:.5g}",
                     f"{s['t'][c]:.2f}", f"{s['p_corrected'][c]:.4g}",
                     self._p_star(s["p_corrected"][c]), f"{s['d'][c]:.2f}"]
                    for c in range(len(ch))]
            return rows, s["n"]

        self._run(job, lambda r: self._tfr_stats_done(r, band, b0, b1, t0, t1),
                  "正在做逐通道时频检验…")

    def _tfr_stats_done(self, res, band, b0, b1, t0, t1):
        rows, n = res
        d = self._table_dialog(
            f"时频统计检验（{band[0]:g}-{band[1]:g} Hz）",
            ["通道", "基线均值", "检验窗均值", "变化量", "t", "p(FDR)", "显著性", "Cohen's d"],
            rows,
            note=f"配对 t 检验（每个 epoch 的检验窗 vs 基线窗）+ FDR 校正；n={n} 个 epoch。"
                 f"基线窗 {b0:.2f}~{b1:.2f} s，检验窗 {t0:.2f}~{t1:.2f} s；单位 µV²。")
        d.exec()

    def _tfr_export(self):
        if not hasattr(self, "_tfr_P"):
            QMessageBox.information(self, "提示", "请先「计算时频图」")
            return
        p = self._pick_save_path("tfr.csv", "导出时频数据")
        if not p:
            return
        header, rows = tfr_to_rows(self._tfr_P, self._tfr_f, self._tfr_t, self._ch)
        n = self._write_csv(p, header, rows)
        self.lbl_hint.setText(
            f"已导出时频数据 {n} 行（通道×频点）→ {p}"
            f"（列 = 时间点，epoch 平均功率）")

    # ================================================================ 非线性
    def _build_nonlinear(self):
        w = QWidget()
        self.btn_nl = QPushButton("计算样本熵 / 近似熵 / 排列熵")
        self.lbl_nl = QLabel("说明：熵计算逐 epoch 逐通道，建议 epochs 不超过 60（约几秒~十几秒）。")
        self.lbl_nl.setStyleSheet("color:gray;")
        self.tbl_nl = QTableWidget(0, 0)
        lay = QVBoxLayout(w)
        lay.addWidget(self.btn_nl)
        lay.addWidget(self.lbl_nl)
        lay.addWidget(self.tbl_nl, 1)
        self.btn_nl.clicked.connect(self.compute_nonlinear)
        return w

    def compute_nonlinear(self):
        if not self._require_raw():
            return
        n = min(self.spin_max.value(), 60)

        def job(log=print):
            X, _, _, raw_eeg = self._segments(max_epochs=n)
            self._ch = raw_eeg.ch_names
            return entropy_channels(X, methods=("sample", "approximate", "permutation"))

        self._run(job, self._nl_done, f"正在计算熵（{n} epochs，约数秒~十几秒）…")

    def _nl_done(self, ent):
        rows = [[self._ch[c], f"{ent['sample'][c]:.4f}", f"{ent['approximate'][c]:.4f}",
                 f"{ent['permutation'][c]:.4f}"] for c in range(len(self._ch))]
        self._fill_table(self.tbl_nl, ["通道", "样本熵", "近似熵", "排列熵"], rows)

    # ================================================================ 网络
    def _build_net(self):
        w = QWidget()
        row = QHBoxLayout()
        row.addWidget(QLabel("连接方法"))
        self.combo_method = QComboBox()
        # 只放下拉短标签（方法名），中文说明放 tooltip：
        # QComboBox 会按最长条目撑大最小宽度，长条目会让整页最小宽度涨到 2000+ px
        for key in CONNECT_METHODS:
            self.combo_method.addItem(key, userData=key)
        self.combo_method.setToolTip("连接方法（键名）：\n" + "\n".join(
            f"{k}：{CONNECT_LABELS.get(k, '')}" for k in CONNECT_METHODS))
        row.addWidget(self.combo_method)
        row.addWidget(QLabel("稀疏化"))
        self.combo_sparsify = QComboBox()
        for key in SPARSIFY_STRATEGIES:      # 短标签（取「（…）」之前的部分）
            self.combo_sparsify.addItem(SPARSIFY_LABELS.get(key, key).split("（")[0],
                                        userData=key)
        self.combo_sparsify.setToolTip("\n".join(
            f"{SPARSIFY_LABELS.get(k, k).split('（')[0]}：{SPARSIFY_LABELS.get(k, k)}"
            for k in SPARSIFY_STRATEGIES))
        row.addWidget(self.combo_sparsify)
        row.addWidget(QLabel("边密度"))
        self.spin_density = QDoubleSpinBox()
        self.spin_density.setRange(0.05, 1.0)
        self.spin_density.setSingleStep(0.05)
        self.spin_density.setValue(0.2)
        row.addWidget(self.spin_density)
        self.btn_net = QPushButton("计算连接与图属性")
        row.addWidget(self.btn_net)
        self.btn_brain3d = QPushButton("3D 脑网络…")
        self.btn_dynamic = QPushButton("动态网络…")
        self.btn_bio = QPushButton("生物标志物挖掘…")
        self.btn_matrix = QPushButton("协变 / 相似性 / 变量关联…")
        self.btn_matrix.setToolTip(
            "表格 / 矩阵类分析的统一入口：\n"
            "① T1 结构协变（灰质体积/皮层厚度表 → 组水平协变矩阵）\n"
            "② ASL-CBF / PET 的 ROI 数值表 → Pearson/Spearman/KL/JSD/MIND 相似性网络\n"
            "③ 时间序列连接（ROI × 时间，12 种方法）\n"
            "④ 变量关联（Linked Heatmap）：随机森林重要性 + Pearson 相关矩阵 +\n"
            "   线性混合效应模型（LMM）效应弧线 —— 例如「临床量表 ~ EEG 指标」\n"
            "⑤ 导入其他软件导出的连接矩阵（.mat / .csv / .npy）继续分析")
        row.addWidget(self.btn_brain3d)
        row.addWidget(self.btn_dynamic)
        row.addWidget(self.btn_bio)
        row.addWidget(self.btn_matrix)
        self.lbl_net = QLabel("granger / low_rank 较慢（十几秒）；wcoherence 有 ~0.2 基线偏差。")
        self.lbl_net.setStyleSheet("color:gray;")
        self.lbl_net.setWordWrap(True)
        self.lbl_net.setMinimumWidth(120)    # 允许换行，不撑宽整页
        row.addWidget(self.lbl_net)
        row.addStretch(1)
        self.fig_net, self.canvas_net, self.tb_net = new_canvas_with_toolbar(10, 5)
        self.tbl_net = QTableWidget(0, 0)
        lay = QVBoxLayout(w)
        lay.addLayout(row)
        lay.addWidget(self.tb_net)
        lay.addWidget(self.canvas_net, 1)
        lay.addWidget(self.tbl_net)
        self.btn_net.clicked.connect(self.compute_net)
        self.btn_brain3d.clicked.connect(self.show_brain3d)
        self.btn_dynamic.clicked.connect(self.show_dynamic)
        self.btn_bio.clicked.connect(self.show_biomarkers)
        self.btn_matrix.clicked.connect(self.show_matrix_workbench)
        return w

    def show_matrix_workbench(self):
        """协变 / 相似性 / 外部矩阵工作台（T1、ASL、PET、外部连接矩阵）。"""
        from .widgets.netmatrix_dialog import NetMatrixDialog

        NetMatrixDialog(self).exec()

    def compute_net(self):
        if not self._require_raw():
            return
        method = self.combo_method.currentData() or self.combo_method.currentText()

        def job(log=print):
            X, _, _, raw_eeg = self._segments()
            self._ch = raw_eeg.ch_names
            fs = float(raw_eeg.info["sfreq"])
            A = connectivity_matrix(X, fs, method=method)
            if A.ndim == 3:
                A = A.mean(axis=0)
            A = np.asarray(A, dtype=float)
            np.fill_diagonal(A, 0.0)
            return A

        self._run(job, lambda A: self._net_done(A, method), "正在计算连接矩阵…")

    def _net_done(self, A, method):
        try:
            _, pos, names = get_eeg_pos(self.state.raw)
            idx = [i for i in range(len(self._ch)) if self._ch[i] in names]
            if not idx:
                raise ValueError("通道名无法匹配标准导联，无法绘制网络拓扑图")
            Asub = A[np.ix_(idx, idx)]
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "提示", exc_to_zh(e))
            return

        strategy = self.combo_sparsify.currentData() or "proportional"
        if strategy in ("absolute", "adaptive"):
            B = binarize(Asub, density=self.spin_density.value(), strategy=strategy,
                         percentile=float(self.spin_density.value()) * 100)
        else:
            B = binarize(Asub, density=self.spin_density.value(), strategy=strategy)

        fig, canvas = self.fig_net, self.canvas_net
        fig.clear()
        ax1 = fig.add_subplot(121)
        im = ax1.imshow(Asub, cmap="RdBu_r", aspect="auto")
        fig.colorbar(im, ax=ax1, shrink=0.7)
        ax1.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
        ax1.set_yticks(range(len(names)), names, fontsize=6)
        ax1.set_title(f"{CONNECT_LABELS.get(method, method)} 连接矩阵")

        ax2 = fig.add_subplot(122)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                if B[a, b]:
                    ax2.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]],
                             "k-", lw=0.5, alpha=0.5)
        ax2.scatter(pos[:, 0], pos[:, 1], c="white", edgecolors="k", s=120, zorder=3)
        for i, n in enumerate(names):
            ax2.text(pos[i, 0], pos[i, 1], n, fontsize=6, ha="center", va="center", zorder=4)
        ax2.set_aspect("equal")
        ax2.set_title(f"二值网络（稀疏化 {strategy}，密度/分位 {self.spin_density.value():.2f}）")
        canvas.draw()

        gm = graph_metrics(B)
        rows = [[names[i], f"{gm['degree'][i]:.0f}", f"{gm['clustering'][i]:.3f}",
                 f"{gm['local_efficiency'][i]:.3f}"] for i in range(len(names))]
        rows.append(["全局", "-", "-", f"{gm['global_efficiency']:.3f}"])
        rows.append(["平均最短路径", f"{gm['avg_path_length']:.3f}", "", ""])
        self._fill_table(self.tbl_net,
                         ["通道", "节点度", "聚类系数", "局部效率"], rows)

    # ------------------------- 3D 脑网络（BrainNet Studio 风格） -------------------------
    def show_brain3d(self):
        if not self._require_raw():
            return
        method = self.combo_method.currentText()

        def job(log=print):
            X, _, _, raw_eeg = self._segments()
            self._ch = raw_eeg.ch_names
            fs = float(raw_eeg.info["sfreq"])
            A = connectivity_matrix(X, fs, method=method)
            if A.ndim == 3:
                A = A.mean(axis=0)
            A = np.asarray(A, dtype=float).copy()
            np.fill_diagonal(A, 0.0)
            return A

        self._run(job, lambda A: self._brain3d_done(A, method), "正在构建 3D 脑网络…")

    def _brain3d_done(self, A, method):
        dlg = Brain3DDialog(A, self.state.raw, f"3D 脑网络（{method}）", self)
        dlg.exec()

    # ------------------------- 动态脑网络 -------------------------
    def show_dynamic(self):
        if not self._require_raw():
            return
        method = self.combo_method.currentText()
        span = max(self.spin_tmax.value() - self.spin_tmin.value(), 0.4)

        def job(log=print):
            X, _, _, raw_eeg = self._segments()
            fs = float(raw_eeg.info["sfreq"])
            ch = raw_eeg.ch_names
            win_s = max(span / 3.0, 0.2)
            step_s = max(win_s / 3.0, 0.05)
            t, adj = dynamic_connectivity(X, fs, method=method, win_s=win_s, step_s=step_s)
            return t, adj, dynamic_features(adj), connection_variability(adj), ch, win_s

        self._run(job, self._dynamic_done, "正在构建动态网络…")

    def _dynamic_done(self, res):
        t, adj, feats, var, ch, win_s = res
        fig, canvas, tb = new_canvas_with_toolbar(10, 4.2)
        ax1 = fig.add_subplot(121)
        ax1.plot(t, feats["global_efficiency"], "-o", ms=3, label="全局效率")
        ax1.plot(t, feats["clustering"], "-s", ms=3, label="平均聚类系数")
        ax1.set_xlabel("窗口中心时间 (s)")
        ax1.set_ylabel("图属性")
        ax1.set_title(f"动态网络属性（{len(t)} 窗口，窗长 {win_s:.2f}s）")
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=8)
        ax2 = fig.add_subplot(122)
        im = fig.colorbar(ax2.imshow(var, cmap="hot"), ax=ax2, shrink=0.8)
        im.set_label("变异性 (std)")
        ax2.set_xticks(range(len(ch)), ch, rotation=90, fontsize=6)
        ax2.set_yticks(range(len(ch)), ch, fontsize=6)
        ax2.set_title("连接时间变异性（越大越动态）")
        canvas.draw()

        d = QDialog(self)
        d.setWindowTitle("动态脑网络")
        lay = QVBoxLayout(d)
        lay.addWidget(tb)
        lay.addWidget(canvas)
        d.resize(1000, 460)
        d.exec()

    # ------------------------- 生物标志物挖掘 -------------------------
    def show_biomarkers(self):
        if not self._require_raw():
            return
        method = self.combo_method.currentText()

        def job(log=print):
            X, labels, _, raw_eeg = self._segments()
            fs = float(raw_eeg.info["sfreq"])
            ch = raw_eeg.ch_names
            return biomarkers_by_condition(X, labels, fs, method=method, top_n=20), ch

        self._run(job, self._bio_done, "正在挖掘判别性连接…")

    def _bio_done(self, res):
        r, ch = res
        conds = r["conds"]
        rows = []
        for rank, (i, j, t, p, m0, m1) in enumerate(r["top_connections"], 1):
            stars = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            rows.append([rank, f"{ch[i]} – {ch[j]}", f"{t:.2f}", f"{p:.4f}{stars}",
                         f"{m0:.4f}", f"{m1:.4f}"])
        tbl_top = QTableWidget(0, 0)
        self._fill_table(tbl_top, ["排名", "通道对", "t 值", "p 值(FDR 校正)",
                                   f"均值[{conds[0]}]", f"均值[{conds[1]}]"], rows)

        ns = r["node_score"]
        order = np.argsort(-ns)[:10]
        rows2 = [[k + 1, ch[i], f"{ns[i]:.2f}"] for k, i in enumerate(order)]
        tbl_node = QTableWidget(0, 0)
        self._fill_table(tbl_node, ["排名", "脑区（电极）", "判别性得分"], rows2)

        d = QDialog(self)
        d.setWindowTitle(f"候选生物标志物（条件 {conds[0]} vs {conds[1]}）")
        lay = QVBoxLayout(d)
        lay.addWidget(QLabel(
            f"判别性连接 Top-20（按 FDR 校正后 p 排序；** p<0.01，* p<0.05）；"
            f"全部 {r['n_significant']} 条边在 {r.get('correct_method', 'fdr_bh')} "
            f"校正后 p<0.05（共检 C(C−1)/2 条）"))
        lay.addWidget(tbl_top)
        lay.addWidget(QLabel("判别性脑区 Top-10（该电极所有连接的 |t| 之和）"))
        lay.addWidget(tbl_node)
        d.resize(820, 650)
        d.exec()
