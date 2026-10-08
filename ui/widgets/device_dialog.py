"""训练设备选择：**探测本机设备 → 勾选（多卡可多选）→ 一键测速**。

跨平台：Windows / Linux 上会列出各张 NVIDIA 卡，macOS 上列出 Apple Silicon(MPS)，
没有加速设备时用 CPU 兜底；勾选多张 CUDA 卡即多卡并行（DataParallel，
权重保存时会去掉 ``module.`` 前缀，之后单卡/CPU 也能直接加载）。
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QCheckBox, QDialog, QGridLayout,
                             QHBoxLayout, QHeaderView, QLabel, QMessageBox,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from core.train.devices import (device_label, list_devices, probe_device,
                               resolve_device, system_info)


class DeviceDialog(QDialog):
    """选择训练设备（含实时探测与测速）。"""

    def __init__(self, current: str = "auto", parent=None):
        super().__init__(parent)
        self.setWindowTitle("训练设备 — 探测与选择")
        self.resize(760, 560)

        head = QLabel("<b>系统与框架</b>")
        self.lbl_sys = QLabel()
        self.lbl_sys.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_sys.setWordWrap(True)
        self.lbl_sys.setStyleSheet(
            "background:rgba(0,0,0,0.03); border-radius:8px; padding:8px;")

        self.chk_auto = QCheckBox("自动选择（推荐：CUDA > MPS > CPU）")
        self.chk_auto.setToolTip("自动挑当前机器上最快的可用设备；换机器不用改配置")
        self.chk_auto.toggled.connect(self._on_auto_toggled)

        self.tbl = QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(["选择", "设备", "名称", "显存", "说明"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        for col in (0, 1, 3):
            self.tbl.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents)
        self.tbl.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents)
        self.tbl.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Stretch)      # 说明列占满剩余宽度
        self.tbl.itemChanged.connect(self._on_item_changed)

        self.lbl_result = QLabel("点「测试所选设备」会真跑几步矩阵乘 + 反传（同步计时）")
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setStyleSheet("color:#888; font-size:11px;")

        btn_reload = QPushButton("重新检测")
        btn_reload.clicked.connect(lambda: self._reload())
        self.btn_probe = QPushButton("测试所选设备")
        self.btn_probe.clicked.connect(self._probe)
        self.btn_probe.setProperty("accent", True)
        btn_ok = QPushButton("确定")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)

        btns = QHBoxLayout()
        btns.addWidget(btn_reload)
        btns.addWidget(self.btn_probe)
        btns.addStretch(1)
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)

        lay = QVBoxLayout(self)
        lay.addWidget(head)
        lay.addWidget(self.lbl_sys)
        lay.addWidget(self.chk_auto)
        lay.addWidget(self.tbl, 1)
        lay.addWidget(self.lbl_result)
        lay.addLayout(btns)

        self._current = str(current or "auto")
        self._reload()

    # ================================================================ 探测
    def _reload(self):
        info = system_info()
        cells = "　".join(f"<span style='color:#888'>{k}</span> <b>{v}</b>"
                          for k, v in info.items())
        self.lbl_sys.setText(cells)
        self._devices = list_devices()
        self._auto = self._current in ("", "auto") or not self._current
        self.chk_auto.setChecked(self._auto)

        self.tbl.blockSignals(True)
        self.tbl.setRowCount(0)
        want = set(self._checked_ids(self._current))
        for d in self._devices:
            row = self.tbl.rowCount()
            self.tbl.insertRow(row)
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk.setCheckState(Qt.CheckState.Checked if d.id in want
                              else Qt.CheckState.Unchecked)
            chk.setData(Qt.ItemDataRole.UserRole, d.id)
            chk.setData(Qt.ItemDataRole.UserRole + 1, d.kind)
            self.tbl.setItem(row, 0, chk)
            for col, val in enumerate(
                    (d.id, d.name, f"{d.memory_gb:g} GB" if d.memory_gb else "—",
                     d.note or ""), start=1):
                self.tbl.setItem(row, col, QTableWidgetItem(str(val)))
        self.tbl.blockSignals(False)
        self._on_auto_toggled(self._auto)
        self._refresh_summary()

    @staticmethod
    def _checked_ids(spec: str) -> list[str]:
        """把配置写法展开成表格里该勾选的设备 id。"""
        prim, ids = resolve_device(spec)
        if len(ids) > 1:
            return [f"cuda:{i}" for i in ids]
        return [prim]

    def _on_auto_toggled(self, on: bool):
        self.tbl.setEnabled(not on)
        if on:
            self.tbl.blockSignals(True)
            for row in range(self.tbl.rowCount()):
                if self.tbl.item(row, 0) is not None:
                    self.tbl.item(row, 0).setCheckState(Qt.CheckState.Unchecked)
            self.tbl.blockSignals(False)
        self._refresh_summary()

    def _on_item_changed(self, item):
        """勾选联动：CPU/MPS/XPU 只能单选；CUDA 可以多选（多卡并行）。"""
        if item.column() != 0:
            return
        kind = item.data(Qt.ItemDataRole.UserRole + 1)
        if item.checkState() != Qt.CheckState.Checked:
            self._refresh_summary()
            return
        if kind in ("cpu", "mps", "xpu"):
            self.tbl.blockSignals(True)
            for row in range(self.tbl.rowCount()):
                other = self.tbl.item(row, 0)
                if other is not None and other is not item:
                    other.setCheckState(Qt.CheckState.Unchecked)
            self.tbl.blockSignals(False)
            self.chk_auto.setChecked(False)
        else:
            # 勾了 CUDA 就取消 CPU/MPS 这类单体设备
            self.tbl.blockSignals(True)
            for row in range(self.tbl.rowCount()):
                other = self.tbl.item(row, 0)
                if (other is not None and other is not item
                        and other.checkState() == Qt.CheckState.Checked
                        and other.data(Qt.ItemDataRole.UserRole + 1) != "cuda"):
                    other.setCheckState(Qt.CheckState.Unchecked)
            self.tbl.blockSignals(False)
        self._refresh_summary()

    # ================================================================ 结果
    def selected_spec(self) -> str:
        """当前选择 → 配置写法（"auto" / "cpu" / "mps" / "cuda:0" / "cuda:0,1"）。"""
        if self.chk_auto.isChecked():
            return "auto"
        cuda: list[int] = []
        single = ""
        for row in range(self.tbl.rowCount()):
            it = self.tbl.item(row, 0)
            if it is None or it.checkState() != Qt.CheckState.Checked:
                continue
            did = str(it.data(Qt.ItemDataRole.UserRole))
            if str(it.data(Qt.ItemDataRole.UserRole + 1)) == "cuda":
                cuda.append(int(did.split(":")[1]))
            else:
                single = did
        if cuda:
            cuda = sorted(set(cuda))
            return f"cuda:{cuda[0]}" if len(cuda) == 1 else \
                "cuda:" + ",".join(str(i) for i in cuda)
        return single or "auto"

    def _refresh_summary(self):
        spec = self.selected_spec()
        prim, ids = resolve_device(spec)
        extra = f"　多卡并行 {len(ids)} 张（卡号 {ids}）" if len(ids) > 1 else ""
        self.lbl_result.setText(f"将使用：<b>{device_label(spec)}</b>"
                                f"　配置写法 <code>{spec}</code>{extra}")
        self.lbl_result.setStyleSheet("color:#333; font-size:11px;")

    def _probe(self):
        spec = self.selected_spec()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            res = probe_device(spec)
        finally:
            QApplication.restoreOverrideCursor()
        if res["ok"]:
            ms = res["ms_per_step"]
            self.lbl_result.setText(
                f"<span style='color:#1a7f37'>✓ 测试通过</span>　{res['name']}"
                f"　·　512×512 矩阵乘 + 反传 <b>{ms:.2f} ms/步</b>（越小越快）")
        else:
            self.lbl_result.setText(
                f"<span style='color:#c0392b'>✗ 该设备跑不通</span>　{res['device']}\n"
                f"{res['error']}\n可改选其它设备（CPU 一定能跑）。")
            QMessageBox.warning(self, "设备不可用",
                                f"{res['device']} 测试失败：\n{res['error']}")
