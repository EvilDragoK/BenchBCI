"""数据管理页：BNCI / PhysioNet 数据集加载 / 通用格式导入 / 元信息 / 信号浏览器 / PSD。"""
from __future__ import annotations

import re
from pathlib import Path

from PyQt6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                             QMessageBox, QPlainTextEdit, QProgressDialog,
                             QPushButton, QSplitter, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)
from PyQt6.QtCore import Qt

from core.io.bnci import list_subjects as bnci_list_subjects
from core.io.bnci import load_subject_raw as bnci_load_raw
from core.io.physionet import DEFAULT_RUNS as PN_DEFAULT_RUNS
from core.io.physionet import list_subjects as pn_list_subjects
from core.io.physionet import load_subject_raw as pn_load_raw
from core.io.loaders import read_raw_any
from core.preprocess.operators import Normalize, OPERATORS
from core import PROJECT_ROOT
from core.data_paths import ENV_VAR as DATA_ENV_VAR
from core.data_paths import bnci_root, physionet_root
from .widgets.feature_dialog import FeatureDialog
from .widgets.psd_dialog import PsdCompareDialog
from .widgets.signal_browser import SignalBrowser
from .widgets.topo_dialog import TopoDialog
from .workers import FunctionThread, exc_to_zh

#: 默认数据集路径
# 默认路径 + 环境变量覆盖（换机器/换系统不用改代码，详见 core/data_paths.py）
BNCI_ROOT = str(bnci_root())
PHYSIONET_ROOT = str(physionet_root())

#: 数据管理页「快速处理」的滤波预设（对应文献 Filtering 列的常见写法）
FILTER_PRESETS = {
    "无": None,
    "带通 0.1-75 Hz": ("bandpass", 0.1, 75.0),
    "带通 0.3-75 Hz": ("bandpass", 0.3, 75.0),
    "带通 0.5-45 Hz": ("bandpass", 0.5, 45.0),
    "带通 1-50 Hz": ("bandpass", 1.0, 50.0),
    "带通 4-40 Hz（MI 常用）": ("bandpass", 4.0, 40.0),
    "带通 8-32 Hz（MI）": ("bandpass", 8.0, 32.0),
    "低通 ≤120 Hz": ("bandpass", 0.0, 120.0),
    "高通 ≥0.1 Hz": ("bandpass", 0.1, 0.0),
    "50 Hz 陷波": ("notch", [50.0]),
    "50&60 Hz 陷波": ("notch", [50.0, 60.0]),
    "高斯平滑 σ=20ms": ("gaussian", 0.02),
}

#: 归一化预设（Alignment 列）→ Normalize 算子 mode
QNORM_PRESETS = {
    "无": None,
    "z-score → N(0,1)": "zscore",
    "Norm → [-1,1]": "minmax_pm",
    "Norm → [0,1]": "minmax_01",
    "Norm → x/P95": "p95",
    "IQR Scaling": "iqr",
    "A-Law 压扩": "alaw",
    "EMA 在线标准化": "ema",
    "去均值 (zero-mean)": "zero_mean",
}

#: 各归一化模式对应的显示单位
QNORM_UNIT = {
    "zscore": "z", "ema": "z", "quantile": "N(0,1)",
    "minmax_pm": "a.u.", "minmax_01": "a.u.", "p95": "a.u.",
    "iqr": "a.u.", "alaw": "a.u.", "zero_mean": "a.u.",
}


class DataPage(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state  # 共享：raw / raw_name / meta
        self._thread = None
        # 最近一次加载的原始数据（用于"还原"，撤销重采样/归一化）
        self._original_raw = None
        self._original_meta = None
        self._original_name = ""
        self._ann = None

        # ---- 顶栏：数据集类型 + 根目录 + 被试 + session/runs ----
        self.combo_kind = QComboBox()
        self.combo_kind.addItems(["BNCI2014001 (2a)", "PhysioNet MI (eegmmidb)"])
        self.edit_root = QLineEdit(BNCI_ROOT)
        self.combo_subject = QComboBox()
        self.combo_session = QComboBox()
        self.combo_session.addItems(["T", "E"])
        self.edit_runs = QLineEdit(",".join(str(r) for r in PN_DEFAULT_RUNS))
        self.edit_runs.setFixedWidth(90)
        self.edit_runs.setEnabled(False)
        self.edit_runs.setToolTip("PhysioNet 要合并的 run，如 5,6,9,10（想象 4 类）")
        self.btn_scan = QPushButton("扫描被试")
        self.btn_load = QPushButton("加载数据")
        self.btn_import = QPushButton("导入文件…")

        top = QHBoxLayout()
        top.addWidget(QLabel("数据集"))
        top.addWidget(self.combo_kind)
        top.addWidget(QLabel("目录"))
        top.addWidget(self.edit_root, 1)
        top.addWidget(self.btn_scan)
        top.addWidget(QLabel("被试"))
        top.addWidget(self.combo_subject)
        top.addWidget(QLabel("Session"))
        top.addWidget(self.combo_session)
        top.addWidget(QLabel("Runs"))
        top.addWidget(self.edit_runs)
        top.addWidget(self.btn_load)
        top.addWidget(self.btn_import)

        # ---- 中部：元信息 + 信号浏览器 ----
        self.txt_meta = QPlainTextEdit()
        self.txt_meta.setReadOnly(True)
        self.txt_meta.setMaximumWidth(340)
        self.browser = SignalBrowser()

        # ---- 底部：事件注释表 + 工具按钮列 ----
        self.tbl_events = QTableWidget(0, 3)
        self.tbl_events.setHorizontalHeaderLabels(["描述", "起始 (s)", "时长 (s)"])
        self.btn_psd = QPushButton("PSD 频谱图")
        self.btn_features = QPushButton("特征提取…")
        self.btn_topo = QPushButton("脑地形图…")
        self.btn_restore = QPushButton("↩ 还原原始数据")
        self.btn_restore.setEnabled(False)

        # 重采样
        row_rs = QHBoxLayout()
        row_rs.addWidget(QLabel("重采样至"))
        self.spin_sfreq = QDoubleSpinBox()
        self.spin_sfreq.setRange(50.0, 2048.0)
        self.spin_sfreq.setDecimals(1)
        self.spin_sfreq.setValue(250.0)
        self.spin_sfreq.setSuffix(" Hz")
        row_rs.addWidget(self.spin_sfreq)
        self.btn_resample = QPushButton("重采样")
        row_rs.addWidget(self.btn_resample)

        # 归一化
        row_nm = QHBoxLayout()
        self.combo_norm_mode = QComboBox()
        self.combo_norm_mode.addItems(["zscore", "minmax"])
        row_nm.addWidget(QLabel("归一化"))
        row_nm.addWidget(self.combo_norm_mode)
        self.btn_normalize = QPushButton("应用归一化")
        row_nm.addWidget(self.btn_normalize)

        # 快速处理：滤波（Filtering）→ 归一化（Alignment），对应文献方案表
        lbl_proc = QLabel("—— 快速处理（滤波 → 归一化）——")
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(list(FILTER_PRESETS))
        self.combo_qnorm = QComboBox()
        self.combo_qnorm.addItems(list(QNORM_PRESETS))
        self.btn_apply_proc = QPushButton("▶ 应用处理并对比")
        row_p1 = QHBoxLayout()
        row_p1.addWidget(QLabel("滤波"))
        row_p1.addWidget(self.combo_filter, 1)
        row_p2 = QHBoxLayout()
        row_p2.addWidget(QLabel("对齐"))
        row_p2.addWidget(self.combo_qnorm, 1)

        side = QVBoxLayout()
        side.addWidget(self.btn_psd)
        side.addWidget(self.btn_topo)
        side.addWidget(self.btn_features)
        side.addLayout(row_rs)
        side.addLayout(row_nm)
        side.addWidget(lbl_proc)
        hint_proc = QLabel("仅影响当前浏览与「信号分析」页，\n不进入训练流水线（训练请用预处理工作台）")
        hint_proc.setStyleSheet("color:#888; font-size:11px;")
        side.addWidget(hint_proc)
        side.addLayout(row_p1)
        side.addLayout(row_p2)
        side.addWidget(self.btn_apply_proc)
        side.addWidget(self.btn_restore)
        side.addStretch(1)

        bottom = QHBoxLayout()
        bottom.addWidget(self.tbl_events, 1)
        bottom.addLayout(side)

        splitter = QSplitter(Qt.Orientation.Vertical)
        mid = QWidget()
        mlay = QHBoxLayout(mid)
        mlay.addWidget(self.txt_meta)
        mlay.addWidget(self.browser, 1)
        splitter.addWidget(mid)
        ev = QWidget()
        ev.setLayout(bottom)
        splitter.addWidget(ev)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addWidget(splitter, 1)

        self.btn_scan.clicked.connect(self.scan)
        self.btn_load.clicked.connect(self.load_subject)
        self.btn_import.clicked.connect(self.import_file)
        self.btn_psd.clicked.connect(self.show_psd)
        self.btn_features.clicked.connect(self.show_features)
        self.btn_topo.clicked.connect(self.show_topo)
        self.btn_resample.clicked.connect(self.resample_raw)
        self.btn_normalize.clicked.connect(self.normalize_raw)
        self.btn_apply_proc.clicked.connect(self.apply_processing)
        self.btn_restore.clicked.connect(self.restore_original)
        self.combo_kind.currentIndexChanged.connect(self._on_kind_changed)
        self.tbl_events.cellClicked.connect(self._on_event_clicked)
        self._threads: list[FunctionThread] = []
        self.scan()

    # ------------------------------------------------------------------
    def _start_busy(self, text: str) -> QProgressDialog:
        """弹出一个模态、无取消按钮的读取进度窗（不确定时长 busy 指示）。"""
        dlg = QProgressDialog(text, "", 0, 0, self)
        dlg.setWindowTitle("请稍候")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setMinimumWidth(320)
        dlg.show()
        QApplication.processEvents()
        return dlg

    def _is_physionet(self) -> bool:
        return self.combo_kind.currentIndex() == 1

    def _on_kind_changed(self):
        pn = self._is_physionet()
        self.combo_session.setEnabled(not pn)
        self.edit_runs.setEnabled(pn)
        self.edit_root.setText(PHYSIONET_ROOT if pn else BNCI_ROOT)
        self.scan()

    def scan(self):
        root = Path(self.edit_root.text())
        pn = self._is_physionet()
        dlg = self._start_busy("正在扫描被试…")

        self._thread = FunctionThread(
            pn_list_subjects if pn else bnci_list_subjects, root, parent=self)
        self._thread.ok.connect(lambda subs: self._scan_done(subs, dlg, pn))
        self._thread.fail.connect(lambda m: self._scan_failed(m, dlg))
        self._thread.start()

    def _scan_done(self, subs, dlg, pn: bool = False):
        dlg.close()
        self.combo_subject.clear()
        self.combo_subject.addItems([f"{s:03d}" if pn else f"{s:02d}" for s in subs])
        if not subs:
            self.txt_meta.setPlainText(f"目录中未发现数据文件：\n{self.edit_root.text()}")

    def _scan_failed(self, msg, dlg):
        dlg.close()
        QMessageBox.warning(self, "扫描失败", msg)

    def load_subject(self):
        if not self.combo_subject.count():
            QMessageBox.information(self, "提示", "请先点「扫描被试」选择被试")
            return
        subj = int(self.combo_subject.currentText())
        root = self.edit_root.text()

        if self._is_physionet():
            runs = [int(x) for x in re.split(r"[,，\s]+", self.edit_runs.text().strip())
                    if x.strip().isdigit()]
            if not runs:
                QMessageBox.information(self, "提示", "请填写要加载的 run 编号，如 5,6,9,10")
                return
            name = f"PhysioNet MI S{subj:03d} runs[{','.join(map(str, runs))}]"
            dlg = self._start_busy(f"正在读取 {name} …")

            def job(log=print):
                return pn_load_raw(root, subj, runs=runs, eeg_only=False)
        else:
            sess = self.combo_session.currentText()
            name = f"BNCI2014001 S{subj:02d}{sess}"
            dlg = self._start_busy(f"正在读取 {name} …")

            def job(log=print):
                return bnci_load_raw(root, subj, sess, eeg_only=False)

        self._thread = FunctionThread(job, parent=self)
        self._thread.ok.connect(lambda res: self._load_done(res, dlg, name))
        self._thread.fail.connect(lambda m: self._load_failed(m, dlg))
        self._thread.start()

    def import_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 EEG 文件", "",
            "EEG 文件 (*.edf *.bdf *.gdf *.vhdr *.fif *.cnt *.set *.mat *.npy);;所有文件 (*)")
        if not path:
            return
        dlg = self._start_busy(f"正在读取 {Path(path).name} …")

        def job(log=print):
            return read_raw_any(path), None

        self._thread = FunctionThread(job, parent=self)
        self._thread.ok.connect(lambda res: self._load_done((res[0], res[1]), dlg, Path(path).name))
        self._thread.fail.connect(lambda m: self._load_failed(m, dlg))
        self._thread.start()

    def _load_done(self, res, dlg, name):
        dlg.close()
        self.btn_load.setEnabled(True)
        try:
            self._on_loaded(res, name)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "显示失败", exc_to_zh(e))

    def restore_original(self):
        """撤销重采样/归一化，回到最近一次加载的原始数据。"""
        if self._original_raw is None:
            QMessageBox.information(self, "提示", "没有可还原的原始数据")
            return
        self._on_loaded((self._original_raw, self._original_meta), self._original_name)
        QMessageBox.information(self, "完成",
                                "已还原到最近一次加载的原始数据（µV 显示、默认增益）。")

    def _load_failed(self, msg, dlg):
        dlg.close()
        self.btn_load.setEnabled(True)
        QMessageBox.critical(self, "加载失败", msg)

    # ------------------------------------------------------------------
    def _on_loaded(self, res, name):
        if isinstance(res, tuple):
            raw, meta = res
        else:
            raw, meta = res
        self.state.raw, self.state.raw_name, self.state.meta = raw, name, meta
        # 保存原始引用（还原功能）与事件引用（点击定位）
        self._original_raw, self._original_meta, self._original_name = raw, meta, name
        self._ann = raw.annotations
        self.btn_restore.setEnabled(True)
        self.browser.set_raw(raw)
        self.btn_load.setEnabled(True)

        lines = [f"来源：{name}", f"采样率：{raw.info['sfreq']:.1f} Hz",
                 f"通道数：{len(raw.ch_names)}",
                 f"时长：{raw.n_times / raw.info['sfreq'] / 60:.1f} 分钟",
                 f"通道：{', '.join(raw.ch_names)}"]
        if meta:
            lines += ["", f"trial 数：{meta.get('n_trials', '-')}",
                      f"类别分布：{meta.get('class_counts', '-')}"]
            if meta.get("runs"):
                lines.append(f"runs：{meta['runs']}")
            if meta.get("line_freq"):
                lines.append(f"工频：{meta['line_freq']:.0f} Hz")
            if meta.get("n_artifact_trials") is not None:
                lines.append(f"BNCI 伪迹标注：{meta['n_artifact_trials']} trials")
        self.txt_meta.setPlainText("\n".join(lines))

        ann = raw.annotations
        rows = min(len(ann), 500)
        self.tbl_events.setRowCount(rows)
        for i in range(rows):
            self.tbl_events.setItem(i, 0, QTableWidgetItem(str(ann.description[i])))
            self.tbl_events.setItem(i, 1, QTableWidgetItem(f"{ann.onset[i]:.2f}"))
            self.tbl_events.setItem(i, 2, QTableWidgetItem(f"{ann.duration[i]:.2f}"))

    def show_psd(self):
        if self.state.raw is None:
            QMessageBox.information(self, "提示", "请先加载数据")
            return
        dlg = PsdCompareDialog(self.state.raw, None, title="PSD 频谱", parent=self)
        dlg.exec()

    # ------------------------------------------------------------------
    def resample_raw(self):
        """对当前已加载的 Raw 重采样（只影响当前会话；批量处理请在预处理流水线配置）。"""
        if self.state.raw is None:
            QMessageBox.information(self, "提示", "请先加载数据")
            return
        target = float(self.spin_sfreq.value())
        cur = float(self.state.raw.info["sfreq"])
        if abs(target - cur) < 1e-6:
            QMessageBox.information(self, "提示", f"当前采样率已是 {cur:g} Hz")
            return
        if target > cur:
            if QMessageBox.question(
                    self, "确认升采样",
                    f"目标 {target:g} Hz 高于当前 {cur:g} Hz（升采样不增加真实信息，"
                    f"文件更大）。继续？") != QMessageBox.StandardButton.Yes:
                return
        dlg = self._start_busy(f"正在重采样 {cur:g} → {target:g} Hz …")

        def job(log=print):
            r = self.state.raw.copy().resample(target)
            return r

        th = FunctionThread(job, parent=self)
        self._threads.append(th)
        th.ok.connect(lambda r: self._resample_done(r, dlg, target))
        th.fail.connect(lambda m: (dlg.close(), QMessageBox.critical(self, "重采样失败", m)))
        th.start()

    def _resample_done(self, raw_new, dlg, target):
        dlg.close()
        self.state.raw = raw_new
        self.state.meta = None
        self.browser.set_raw(raw_new)
        if self.txt_meta.toPlainText():
            self.txt_meta.setPlainText(
                f"（已重采样至 {target:g} Hz，仅当前会话）\n\n" + self.txt_meta.toPlainText())
        QMessageBox.information(
            self, "完成",
            f"已重采样至 {target:g} Hz 并刷新显示。\n注意：仅影响当前会话；"
            f"批量处理请在「预处理工作台」流水线中加入 resample 步骤。")

    def normalize_raw(self):
        """对当前 Raw 做逐通道归一化（z-score / min-max），仅影响当前会话显示。"""
        if self.state.raw is None:
            QMessageBox.information(self, "提示", "请先加载数据")
            return
        mode = self.combo_norm_mode.currentText()
        dlg = self._start_busy(f"正在归一化（{mode}）…")

        def job(log=print):
            return Normalize({"mode": mode}).apply(self.state.raw.copy(), {})

        th = FunctionThread(job, parent=self)
        self._threads.append(th)
        th.ok.connect(lambda r: self._normalize_done(r, dlg, mode))
        th.fail.connect(lambda m: (dlg.close(), QMessageBox.critical(self, "归一化失败", m)))
        th.start()

    def _normalize_done(self, raw_new, dlg, mode):
        dlg.close()
        self.state.raw = raw_new
        self.state.meta = None
        self.browser.set_raw(raw_new)
        # 归一化后数据已无量纲：切换单位（不再 ×1e6）并自动调整通道间距
        unit = "z" if mode == "zscore" else "a.u."
        self.browser.set_display(unit=unit, scale=1.0)
        self.browser.auto_gain()
        QMessageBox.information(
            self, "完成",
            f"已按 {mode} 逐通道归一化并刷新显示（单位已切换为 {unit}，"
            f"增益已自动适配）。\n注意：这是显示/对比用途（统计量来自全段数据）；"
            f"训练请用「训练控制台」的归一化设置，其统计量仅来自训练折，防止数据泄漏。")

    def show_features(self):
        if self.state.raw is None:
            QMessageBox.information(self, "提示", "请先加载数据")
            return
        dlg = FeatureDialog(self.state.raw, parent=self)
        dlg.exec()

    def _on_event_clicked(self, row: int, _col: int = 0):
        """点击事件表某行 → 时间窗定位到该事件并画竖直标记线。"""
        if self._ann is None or row >= len(self._ann):
            return
        t = float(self._ann.onset[row])
        desc = str(self._ann.description[row])
        self.browser.locate(t, label=desc)

    # ------------------------- 快速处理（Filtering → Alignment） -------------------------
    @staticmethod
    def _apply_filter_preset(raw, preset):
        """把滤波预设应用到 raw（复用预处理算子，与流水线/命令行同一实现）。"""
        if preset is None:
            return raw
        kind = preset[0]
        if kind == "bandpass":
            op = OPERATORS["bandpass_filter"]({"l_freq": preset[1], "h_freq": preset[2]})
        elif kind == "notch":
            op = OPERATORS["notch_filter"]({"freqs": preset[1]})
        else:
            op = OPERATORS["gaussian_filter"]({"sigma_s": preset[1]})
        return op.apply(raw, {})

    def apply_processing(self):
        """应用所选滤波与归一化到当前数据，并弹出处理前后 PSD 对比图。"""
        if self.state.raw is None:
            QMessageBox.information(self, "提示", "请先加载数据")
            return
        fkey = self.combo_filter.currentText()
        nkey = self.combo_qnorm.currentText()
        if FILTER_PRESETS.get(fkey) is None and QNORM_PRESETS.get(nkey) is None:
            QMessageBox.information(self, "提示", "请先选择至少一种滤波或归一化方式")
            return
        before = self.state.raw.copy()
        dlg = self._start_busy("正在应用滤波/归一化…")

        def job(log=print):
            r = self.state.raw.copy()
            r = self._apply_filter_preset(r, FILTER_PRESETS[fkey])
            mode = QNORM_PRESETS[nkey]
            if mode:
                r = Normalize({"mode": mode}).apply(r, {})
            return r, mode

        th = FunctionThread(job, parent=self)
        self._threads.append(th)
        th.ok.connect(lambda res: self._process_done(res, dlg, before, fkey, nkey))
        th.fail.connect(lambda m: (dlg.close(), QMessageBox.critical(self, "处理失败", m)))
        th.start()

    def _process_done(self, res, dlg, before, fkey, nkey):
        dlg.close()
        r, mode = res
        self.state.raw = r
        self.state.meta = None
        self.browser.set_raw(r)
        if mode:  # 归一化后切换单位并自动适配增益
            self.browser.set_display(unit=QNORM_UNIT.get(mode, "a.u."), scale=1.0)
            self.browser.auto_gain()
        PsdCompareDialog(before, r, f"处理前后 PSD 对比（{fkey} + {nkey}）", self).exec()
        QMessageBox.information(
            self, "完成",
            f"已应用：{fkey} + {nkey}，显示已刷新（可点击 PSD 对比图回顾）。\n"
            f"提示：仅影响当前会话，点「↩ 还原原始数据」可撤销；"
            f"批量/可复现处理请用「预处理工作台」流水线。")

    def show_topo(self):
        if self.state.raw is None:
            QMessageBox.information(self, "提示", "请先加载数据")
            return
        dlg = TopoDialog(self.state.raw, browser=self.browser, parent=self)
        dlg.exec()
