"""自定义模型编辑器：**在内置模型定义上改写 → 检查维度 → 保存为新模型**。

训练控制台「模型」旁边的「自定义模型…」打开本窗口：

1. 选一个**来源**（内置模型，或已保存的自定义模型）→ 编辑区载入它的完整源码；
2. 直接在源码上改写（如给 EEGNet 的 block 之间加一个金字塔池化分支）；
3. 点 **检查输入/输出与维度**：按当前数据的 (类别数, 通道数, 采样点) 真跑一遍
   构建 → 前向（逐层记录输入/输出形状）→ 反向，并给出通过/失败原因；
4. 点 **保存为新模型** → 写入 ``models_custom/<名称>.py``，回到训练页即可选择训练。
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QApplication, QComboBox, QDialog, QDoubleSpinBox,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
                             QSplitter, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from core.models import TORCH_MODELS
from core.models.custom import (BLANK_SOURCES, SNIPPETS, blank_template, check_code,
                                list_custom, read_source, sanitize_identifier,
                                save_custom, template_source)

from .code_viewer_dialog import PythonHighlighter


class ModelEditorDialog(QDialog):
    """自定义模型编辑器（写码 → 检查 → 另存为新模型）。"""

    def __init__(self, base: str = "eegnet", n_classes: int = 4, n_ch: int = 22,
                 n_times: int = 501, parent=None, on_saved=None,
                 edit_name: str | None = None, blank: bool = False):
        super().__init__(parent)
        self.setWindowTitle("自定义模型 — 从零设计，或在现有定义上改写")
        self.resize(1220, 820)
        self._base = base if base in TORCH_MODELS else "eegnet"
        self._on_saved = on_saved
        self._result: dict | None = None
        self._blank = bool(blank)
        self._loading = False      # 正在写入编辑器（避免把程序写入当作用户编辑）
        self._pristine = True      # 编辑器内容还是刚生成的骨架（未被手工改过）

        # ---------------- 顶部：来源 / 名称 / 说明 ----------------
        self.combo_src = QComboBox()
        for key, label in BLANK_SOURCES:       # 从零设计排在最前
            self.combo_src.addItem(label, userData=("blank", key))
        for k in TORCH_MODELS:
            self.combo_src.addItem(f"（内置）{k}", userData=("builtin", k))
        for name, meta in list_custom().items():
            self.combo_src.addItem(f"（自定义）{name}", userData=("custom", name))
            idx = self.combo_src.count() - 1
            self.combo_src.setItemData(
                idx, meta.get("base", ""), Qt.ItemDataRole.ToolTipRole)
        self.combo_src.currentIndexChanged.connect(self._load_source)

        self.edit_name = QLineEdit(edit_name or f"{self._base}_custom")
        self.edit_name.setPlaceholderText(
            "模型名（英文标识符）：从零设计时它直接决定类名 class <模型名>(nn.Module)")
        self.lbl_name_hint = QLabel()
        self.lbl_name_hint.setStyleSheet("color:#888; font-size:11px;")
        self.edit_name.textChanged.connect(self._on_name_changed)
        name_row = QWidget()
        nl = QHBoxLayout(name_row)
        nl.setContentsMargins(0, 0, 0, 0)
        nl.setSpacing(6)
        nl.addWidget(self.edit_name, 1)
        nl.addWidget(self.lbl_name_hint)
        self.edit_note = QLineEdit()
        self.edit_note.setPlaceholderText("改动说明（可选），会写进文件头")

        self.combo_snippet = QComboBox()
        self.combo_snippet.addItems(list(SNIPPETS))
        btn_insert = QPushButton("插入片段")
        btn_insert.setToolTip("在当前光标处插入一段可复用的网络模块（含用法注释）")
        btn_insert.clicked.connect(self._insert_snippet)
        btn_reset = QPushButton("重置为来源源码")
        btn_reset.clicked.connect(self._reset_source)

        # ---------------- 中间：编辑区 + 检查结果 ----------------
        self.txt = QPlainTextEdit()
        self.txt.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.txt.setTabStopDistance(28)
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(12)
        self.txt.setFont(mono)
        self._hl = PythonHighlighter(self.txt.document())
        self.txt.textChanged.connect(self._mark_dirty)

        editor = QWidget()
        el = QVBoxLayout(editor)
        el.setContentsMargins(0, 0, 0, 0)
        el.addWidget(QLabel("模型定义（torch.nn.Module 源码，可自由改写）"))
        el.addWidget(self.txt, 1)

        # 结果面板
        self.lbl_sum = QLabel("尚未检查")
        self.lbl_sum.setWordWrap(True)
        self.tbl = QTableWidget(0, 6)
        self.tbl.setHorizontalHeaderLabels(
            ["#", "模块", "层类型", "输入形状", "输出形状", "参数量"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.txt_msg = QPlainTextEdit()
        self.txt_msg.setReadOnly(True)
        self.txt_msg.setPlaceholderText("检查信息（错误 / 警告 / 提示）会显示在这里")

        res = QWidget()
        rl = QVBoxLayout(res)
        rl.setContentsMargins(6, 0, 0, 0)
        rl.addWidget(QLabel("检查结果"))
        rl.addWidget(self.lbl_sum)
        rl.addWidget(self.tbl, 3)
        rl.addWidget(self.txt_msg, 2)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(editor)
        splitter.addWidget(res)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        # ---------------- 底部：维度 + 操作 ----------------
        self.spin_cls = QSpinBox()
        self.spin_cls.setRange(2, 64)
        self.spin_cls.setValue(int(n_classes))
        self.spin_ch = QSpinBox()
        self.spin_ch.setRange(1, 512)
        self.spin_ch.setValue(int(n_ch))
        self.spin_t = QSpinBox()
        self.spin_t.setRange(16, 20000)
        self.spin_t.setValue(int(n_times))
        self.spin_b = QSpinBox()
        self.spin_b.setRange(2, 64)
        self.spin_b.setValue(4)
        for s, tip in ((self.spin_cls, "类别数（= 分类头输出维度）"),
                       (self.spin_ch, "通道数（= 输入的第二个维度）"),
                       (self.spin_t, "每段采样点数（= 输入的最后一维）"),
                       (self.spin_b, "一次前向的 batch 大小（BatchNorm 需要 > 1）")):
            s.setToolTip(tip)
            s.valueChanged.connect(self._on_dims_changed)
        self.lbl_contract = QLabel()
        self.lbl_contract.setStyleSheet("color:#888; font-size:11px;")
        self._update_contract()

        self.btn_check = QPushButton("检查输入 / 输出与维度")
        self.btn_check.setToolTip("真实构建并跑一次前向+反向，逐层核对形状")
        self.btn_check.clicked.connect(self.run_check)
        self.btn_save = QPushButton("保存为新模型")
        self.btn_save.clicked.connect(self.save)
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.accept)

        dims = QHBoxLayout()
        for lb, w in (("类别数", self.spin_cls), ("通道数", self.spin_ch),
                      ("采样点", self.spin_t), ("batch", self.spin_b)):
            dims.addWidget(QLabel(lb))
            dims.addWidget(w)
        dims.addStretch(1)
        dims.addWidget(self.btn_check)
        dims.addWidget(self.btn_save)
        dims.addWidget(self.btn_close)

        top = QHBoxLayout()
        top.addWidget(QLabel("来源"))
        top.addWidget(self.combo_src, 3)
        top.addWidget(QLabel("模型名"))
        top.addWidget(name_row, 3)
        top.addWidget(QLabel("说明"))
        top.addWidget(self.edit_note, 2)

        top2 = QHBoxLayout()
        top2.addWidget(QLabel("代码片段"))
        top2.addWidget(self.combo_snippet, 2)
        top2.addWidget(btn_insert)
        top2.addWidget(btn_reset)
        top2.addWidget(self.lbl_contract, 1)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addLayout(top2)
        lay.addWidget(splitter, 1)
        lay.addLayout(dims)

        if self._blank:                     # 从零设计：名称是第一步
            self.combo_src.setCurrentIndex(0)
            self.edit_name.setText(edit_name or "MyEEGNet")
            self.edit_note.setPlaceholderText("例如：3 层卷积 + 金字塔池化，从零设计")
        self._load_source()
        if self._blank:
            self.edit_name.setFocus()
            self.edit_name.selectAll()

    # ================================================================ 源码
    def _current_kind(self) -> tuple[str, str]:
        return self.combo_src.currentData() or ("builtin", self._base)

    def _update_contract(self):
        """底部提示：当前检查用的输入/输出维度约定。"""
        self.lbl_contract.setText(
            f"维度约定：输入 (B, 1, {self.spin_ch.value()}, {self.spin_t.value()})"
            f" → 输出 (B, {self.spin_cls.value()})")

    def _update_name_hint(self):
        raw = self.edit_name.text().strip()
        cls = sanitize_identifier(raw)
        if not raw:
            self.lbl_name_hint.setText("")
        elif raw == cls:
            self.lbl_name_hint.setText(f"类名 class {cls}(nn.Module)")
        else:
            self.lbl_name_hint.setText(f"将按合法标识符生成：class {cls}(nn.Module)")

    def _on_name_changed(self, *_):
        """从零设计时：名称即类名，边输边重新生成骨架（前提是还没手工改过）。"""
        self._update_name_hint()
        kind, _key = self._current_kind()
        if kind == "blank" and self._pristine:
            self._load_source(force=True)

    def _on_dims_changed(self, *_):
        """维度变化：更新提示；（从零设计且未改过时）骨架里的默认参数跟着变。"""
        self._update_contract()
        kind, _key = self._current_kind()
        if kind == "blank" and self._pristine:
            self._load_source(force=True)

    def _load_source(self, *_ , force: bool = False):
        kind, key = self._current_kind()
        if not force and not self._should_reload():
            return
        self._loading = True
        try:
            if kind == "blank":
                code = blank_template(self.edit_name.text().strip(), self.spin_cls.value(),
                                      self.spin_ch.value(), self.spin_t.value(),
                                      style=key, note=self.edit_note.text().strip())
            elif kind == "custom":
                code = read_source(key)
                self.edit_name.setText(key)
                if not self.edit_note.text():
                    self.edit_note.setText(list_custom().get(key, {}).get("note", ""))
            else:
                code = template_source(key, self.edit_name.text().strip(),
                                       self.edit_note.text().strip())
            self.txt.setPlainText(code)
            self._pristine = True
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "载入失败", f"{type(e).__name__}: {e}")
        finally:
            self._loading = False
        if kind == "builtin":
            self._base = key
        elif kind == "blank":
            self._base = ""          # 从零设计：不记录"基于某个内置模型"
        self._update_name_hint()
        self._update_contract()

    def _should_reload(self) -> bool:
        """用户切换来源时是否允许覆盖编辑器（改过就确认一下）。"""
        if self._pristine or not self.txt.toPlainText().strip():
            return True
        return QMessageBox.question(
            self, "确认", "编辑器里已经有你的改动，切换来源会覆盖它。继续？") \
            == QMessageBox.StandardButton.Yes

    def _mark_dirty(self):
        if not self._loading:
            self._pristine = False

    def _reset_source(self):
        if QMessageBox.question(self, "确认", "放弃当前修改，重新载入来源源码？") \
                != QMessageBox.StandardButton.Yes:
            return
        self._load_source(force=True)

    def _insert_snippet(self):
        name = self.combo_snippet.currentText()
        code = SNIPPETS.get(name, "")
        cur = self.txt.textCursor()
        cur.insertText("\n" + code.strip("\n") + "\n")
        self.txt.setTextCursor(cur)
        self.txt.setFocus()

    # ================================================================ 检查
    def _dims(self) -> dict:
        return {"n_classes": int(self.spin_cls.value()), "n_ch": int(self.spin_ch.value()),
                "n_times": int(self.spin_t.value()), "batch": int(self.spin_b.value())}

    def run_check(self) -> dict:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            res = check_code(self.txt.toPlainText(), **self._dims())
        except Exception as e:  # noqa: BLE001
            res = {"ok": False, "errors": [f"{type(e).__name__}: {e}"],
                   "warnings": [], "notes": [], "steps": [], "classes": [],
                   "target_class": None, "n_params": 0, "n_layers": 0,
                   "input_shape": None, "output_shape": None}
        finally:
            QApplication.restoreOverrideCursor()
        self._result = res
        self._fill_results(res)
        return res

    def _fill_results(self, res: dict):
        ok = bool(res.get("ok"))
        head = ("<span style='color:#1a7f37;font-weight:bold'>检查通过</span>"
                if ok else
                "<span style='color:#c0392b;font-weight:bold'>检查未通过</span>")
        parts = [head]
        if res.get("target_class"):
            parts.append(f"模型类 <b>{res['target_class']}</b>")
        if res.get("input_shape"):
            parts.append(f"输入 <code>{res['input_shape']}</code> → "
                         f"输出 <code>{res.get('output_shape')}</code>")
        parts.append(f"目标输出 (batch={self.spin_b.value()}, "
                     f"n_classes={self.spin_cls.value()})")
        parts.append(f"参数量 {int(res.get('n_params') or 0):,}")
        parts.append(f"逐层核对 {int(res.get('n_layers') or 0)} 层")
        self.lbl_sum.setText("　|　".join(parts))

        steps = res.get("steps") or []
        self.tbl.setRowCount(len(steps))
        for r, s in enumerate(steps):
            vals = [str(s.get("i", r + 1)), str(s.get("name", "")), str(s.get("type", "")),
                    str(s.get("in")), str(s.get("out")), f"{s.get('params', 0):,}"]
            for c, v in enumerate(vals):
                self.tbl.setItem(r, c, QTableWidgetItem(v))

        lines = []
        for e in res.get("errors") or []:
            lines.append(f"[错误] {e}")
        for w in res.get("warnings") or []:
            lines.append(f"[警告] {w}")
        for n in res.get("notes") or []:
            lines.append(f"[提示] {n}")
        if res.get("classes"):
            lines.append(f"[提示] 文件里的 nn.Module 子类：{res['classes']}")
        self.txt_msg.setPlainText("\n".join(lines) or "（无）")

    # ================================================================ 保存
    def save(self):
        name = self.edit_name.text().strip()
        if not name:
            QMessageBox.information(self, "提示", "请先填写模型名称")
            return
        res = self._result or self.run_check()
        if not res.get("ok"):
            if QMessageBox.question(
                    self, "检查未通过",
                    "当前定义未通过检查（见右侧错误列表）。仍要保存吗？\n"
                    "（保存后可在训练页选择，但训练很可能失败）"
            ) != QMessageBox.StandardButton.Yes:
                return
        try:
            path = save_custom(name, self.txt.toPlainText(),
                               base=self._base, note=self.edit_note.text().strip(),
                               shape=self._dims())
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "保存失败", f"{type(e).__name__}: {e}")
            return
        if self._on_saved:
            try:
                self._on_saved(name)
            except Exception:  # noqa: BLE001
                pass
        QMessageBox.information(
            self, "已保存",
            f"自定义模型「{name}」已保存：\n{path}\n\n"
            f"训练页的「模型」下拉框已刷新，可直接选中开始训练。")
        self.accept()
