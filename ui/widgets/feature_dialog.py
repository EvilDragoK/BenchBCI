"""特征提取对话框：选择特征 → 按 cue 分段计算 → 表格预览 → 导出 CSV。"""
from __future__ import annotations

import csv
import time

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog,
                             QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                             QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
                             QTableWidget, QTableWidgetItem, QVBoxLayout)

from core.features import FEATURES, extract_features
from ..workers import FunctionThread


def segment_raw(raw, max_epochs: int = 500, tmin: float = 0.5, tmax: float = 2.5):
    """把 Raw 切成特征提取用的 epoch 数组 (N, C, T)。

    有 cue 事件时按事件分段（与预处理一致）；无事件时用 2s 滑窗（50% 重叠）。
    """
    import mne

    fs = raw.info["sfreq"]
    if len(raw.annotations):
        try:
            events, event_id = mne.events_from_annotations(raw)
        except Exception:
            events, event_id = np.zeros((0, 3), int), {}
        if len(events):
            epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax,
                                baseline=None, preload=True, proj=False,
                                event_repeated="drop")
            if len(epochs):
                return epochs.get_data()[:max_epochs].astype(np.float64)

    # 回退：整段 2s 滑窗（50% 重叠）
    data = raw.get_data()
    win = int((tmax - tmin) * fs)
    step = max(win // 2, 1)
    segs = [data[:, s:s + win] for s in range(0, data.shape[1] - win + 1, step)]
    return np.stack(segs[:max_epochs]).astype(np.float64)


class FeatureDialog(QDialog):
    def __init__(self, raw, parent=None):
        super().__init__(parent)
        self.raw = raw
        self._result = None       # (F, feat_names)
        self._thread = None
        self.resize(1000, 700)
        self.setWindowTitle("特征提取")

        left = QVBoxLayout()
        left.addWidget(QLabel("选择要计算的特征（默认全选，AH/IPH 的 EMD 版需安装 PyEMD）："))
        self.list_feat = QListWidget()
        for name in FEATURES:
            it = QListWidgetItem(name)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            self.list_feat.addItem(it)
        left.addWidget(self.list_feat, 1)
        self.chk_emd = QCheckBox("AH/IPH 使用 EMD 分解（慢，需 PyEMD）")
        left.addWidget(self.chk_emd)

        right = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel("最多 epochs"))
        self.spin_max = QSpinBox()
        self.spin_max.setRange(10, 20000)
        self.spin_max.setValue(500)
        row.addWidget(self.spin_max)
        row.addWidget(QLabel("FC 方法"))
        self.combo_fc = QComboBox()
        self.combo_fc.addItems(["corr", "plv", "coh"])
        row.addWidget(self.combo_fc)
        self.btn_run = QPushButton("计算")
        self.btn_run.setStyleSheet("font-weight:bold;")
        self.btn_csv = QPushButton("导出 CSV")
        self.btn_csv.setEnabled(False)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_csv)
        right.addLayout(row)

        self.lbl_info = QLabel("—")
        right.addWidget(self.lbl_info)
        self.tbl = QTableWidget(0, 0)
        right.addWidget(self.tbl, 1)
        self.txt_names = QPlainTextEdit()
        self.txt_names.setReadOnly(True)
        self.txt_names.setMaximumHeight(120)
        self.txt_names.setPlaceholderText("特征名列表…")
        right.addWidget(self.txt_names)

        root = QHBoxLayout(self)
        root.addLayout(left, 0)
        root.addLayout(right, 1)

        self.btn_run.clicked.connect(self.compute)
        self.btn_csv.clicked.connect(self.export_csv)

    # ------------------------------------------------------------------
    def _selected(self):
        out = []
        for i in range(self.list_feat.count()):
            it = self.list_feat.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(it.text())
        return out

    def compute(self):
        names = self._selected()
        if not names:
            QMessageBox.information(self, "提示", "请至少勾选一个特征")
            return
        max_ep = self.spin_max.value()
        fc = self.combo_fc.currentText()
        use_emd = self.chk_emd.isChecked()

        import mne

        try:
            X = segment_raw(self.raw, max_epochs=max_ep)
        except Exception as e:  # noqa: BLE001
            from ..workers import exc_to_zh

            QMessageBox.critical(self, "分段失败", exc_to_zh(e))
            return
        sfreq = float(self.raw.info["sfreq"])
        ch_names = list(self.raw.ch_names)
        self.lbl_info.setText(f"已切分 {X.shape[0]} epochs × {X.shape[1]} 通道 × "
                              f"{X.shape[2]} 点，计算 {len(names)} 类特征…")

        from PyQt6.QtWidgets import QProgressDialog, QApplication
        from PyQt6.QtCore import Qt

        dlg = QProgressDialog("正在计算特征…", "", 0, 0, self)
        dlg.setWindowTitle("请稍候")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setCancelButton(None)
        dlg.show()
        QApplication.processEvents()

        def job(log=print, progress=None):
            t0 = time.perf_counter()
            F, fnames = extract_features(X, sfreq, names, ch_names=ch_names,
                                         fc_method=fc, use_emd=use_emd)
            return F, fnames, time.perf_counter() - t0

        self._thread = FunctionThread(job, parent=self)
        self._thread.ok.connect(lambda res: self._done(res, dlg))
        self._thread.fail.connect(lambda m: (dlg.close(),
                                             QMessageBox.critical(self, "计算失败", m)))
        self._thread.start()

    def _done(self, res, dlg):
        dlg.close()
        F, feat_names, dt = res
        self._result = (F, feat_names)
        self.btn_csv.setEnabled(True)
        n_show_cols = min(F.shape[1], 30)
        n_show_rows = min(F.shape[0], 100)
        self.tbl.clear()
        self.tbl.setColumnCount(n_show_cols)
        self.tbl.setRowCount(n_show_rows)
        self.tbl.setHorizontalHeaderLabels(feat_names[:n_show_cols])
        for i in range(n_show_rows):
            for j in range(n_show_cols):
                self.tbl.setItem(i, j, QTableWidgetItem(f"{F[i, j]:.4g}"))
        self.lbl_info.setText(
            f"完成（{dt:.1f}s）：{F.shape[0]} epochs × {F.shape[1]} 维特征"
            f"（表格只显示前 {n_show_rows} 行 × 前 {n_show_cols} 列）")
        self.txt_names.setPlainText("  ".join(feat_names))

    def export_csv(self):
        if self._result is None:
            return
        F, feat_names = self._result
        path, _ = QFileDialog.getSaveFileName(self, "导出特征 CSV", "features.csv",
                                              "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch"] + feat_names)
            for i in range(F.shape[0]):
                w.writerow([i] + [f"{v:.6g}" for v in F[i]])
        QMessageBox.information(self, "完成", f"已导出 {path}\n（{F.shape[0]} 行 × {F.shape[1]} 维）")
