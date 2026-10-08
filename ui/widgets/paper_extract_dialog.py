"""从论文方法段抽取训练超参（可选 LLM，离线也能用）。

界面分三块：
- 左：论文文本（粘贴 / 从 txt·md·pdf 导入）+ LLM 设置（供应商、模型、密钥、连通性测试）
- 右上：识别结果表（字段 / 识别值 / **我的修正** / 来源 / 原文依据），可直接改
- 右下：将要写入的训练配置预览（YAML）→ 应用到训练控制台 / 导出 YAML
"""
from __future__ import annotations

import inspect
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                             QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QListWidget,
                             QListWidgetItem, QMessageBox, QPlainTextEdit,
                             QPushButton, QSpinBox, QSplitter, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from core import PROJECT_ROOT
from core.llm import (CONFIG_PATH, FIELDS, PROVIDERS, LLMConfig, default_config,
                      extract_train_params, format_report, parse_field_text,
                      save_config_file, test_connection, to_train_config)
from core.paper_text import (SUPPORTED_SUFFIXES, assemble_param_text, is_supported,
                             provenance, read_sources, summarize_provenance)

from ..workers import FunctionThread

#: 一键体验用的示例段落（中英文各半，覆盖全部字段）
SAMPLE = """2.3 Implementation details
We trained EEGNet for 200 epochs with the AdamW optimizer (learning rate = 1e-3,
weight decay = 0.01, cosine annealing schedule). The mini-batch size was set to 64.
All experiments used a fixed random seed of 42. Input signals were band-pass filtered
between 4 and 40 Hz and standardized with a z-score computed per channel. We applied
Euclidean alignment to each subject before training. Data augmentation was used,
including time shift (±50 samples), additive Gaussian noise and channel dropout (p = 0.1).
Evaluation followed a leave-one-subject-out protocol."""


class PaperExtractDialog(QDialog):
    """论文超参抽取对话框。``on_apply(cfg)`` 由训练页提供，用于把结果写回界面。"""

    def __init__(self, parent=None, on_apply=None, base_config=None):
        super().__init__(parent)
        self.setWindowTitle("从论文提取训练参数（可选 LLM，离线也能用）")
        self.resize(1240, 780)
        self._on_apply = on_apply
        self._base_config = base_config or (lambda: {})
        self._thread: FunctionThread | None = None
        self._result: dict | None = None
        self._prov: dict | None = None
        self._files: list = []                  # [(SourceFile, text)]
        self._asm_text = ""                     # 最近一次由文件组装出的文本
        self._asm_info: dict | None = None
        self.cfg = default_config()
        self.setAcceptDrops(True)               # 支持把 PDF 直接拖进这个对话框

        # ---------------- 左：论文文本 ----------------
        self.txt_paper = QPlainTextEdit()
        self.txt_paper.setPlaceholderText(
            "把论文的「实验设置 / Implementation details」段落粘贴到这里…\n"
            "（中英文都支持；也可以点「从文件导入」读取 .txt / .md / .pdf）")
        self.btn_file = QPushButton("从文件导入…")
        self.btn_file.setToolTip("等同于把文件拖进来（可多选：正文 + 补充材料）")
        self.btn_sample = QPushButton("填入示例")
        self.btn_clear = QPushButton("清空")
        self.btn_file.clicked.connect(self._pick_files)
        self.btn_sample.clicked.connect(lambda: self.txt_paper.setPlainText(SAMPLE))
        self.btn_clear.clicked.connect(self.txt_paper.clear)
        row_f = QHBoxLayout()
        row_f.addWidget(self.btn_file)
        row_f.addWidget(self.btn_sample)
        row_f.addWidget(self.btn_clear)
        row_f.addStretch(1)
        box_paper = QGroupBox("论文文本")
        lay_paper = QVBoxLayout(box_paper)
        lay_paper.addWidget(self.txt_paper, 1)
        lay_paper.addLayout(row_f)

        # ---------------- 左：LLM 设置 ----------------
        self.combo_provider = QComboBox()
        for key, spec in PROVIDERS.items():
            self.combo_provider.addItem(spec["label"], userData=key)
        self.edit_model = QLineEdit(self.cfg.model)
        self.edit_base = QLineEdit(self.cfg.base_url)
        self.edit_key = QLineEdit(self.cfg.api_key)
        self.edit_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit_key.setPlaceholderText("留空 = 只用规则抽取；也可用环境变量（见下）")
        self.chk_show_key = QCheckBox("显示")
        self.chk_show_key.toggled.connect(
            lambda on: self.edit_key.setEchoMode(
                QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password))
        self.chk_save_key = QCheckBox("保存到 configs/llm.yaml（已在 .gitignore）")
        self.spin_timeout = QDoubleSpinBox()
        self.spin_timeout.setRange(5, 600)
        self.spin_timeout.setValue(float(self.cfg.timeout))
        self.spin_timeout.setSuffix(" s")
        self.btn_test = QPushButton("测试连接")
        self.btn_test.clicked.connect(self._test_connection)
        self.lbl_key_hint = QLabel("")
        self.lbl_key_hint.setWordWrap(True)
        self.lbl_key_hint.setMinimumWidth(120)
        self.lbl_key_hint.setStyleSheet("color:#888; font-size:11px;")

        grid = QGridLayout()
        grid.addWidget(QLabel("供应商"), 0, 0)
        grid.addWidget(self.combo_provider, 0, 1, 1, 2)
        grid.addWidget(QLabel("模型"), 1, 0)
        grid.addWidget(self.edit_model, 1, 1, 1, 2)
        grid.addWidget(QLabel("接口地址"), 2, 0)
        grid.addWidget(self.edit_base, 2, 1, 1, 2)
        grid.addWidget(QLabel("API 密钥"), 3, 0)
        grid.addWidget(self.edit_key, 3, 1)
        grid.addWidget(self.chk_show_key, 3, 2)
        grid.addWidget(QLabel("超时"), 4, 0)
        grid.addWidget(self.spin_timeout, 4, 1)
        grid.addWidget(self.btn_test, 4, 2)
        grid.addWidget(self.chk_save_key, 5, 1, 1, 2)
        grid.addWidget(self.lbl_key_hint, 6, 0, 1, 3)
        box_llm = QGroupBox("LLM 设置（可选）")
        box_llm.setLayout(grid)

        # ---------------- 左：论文文件（可拖入，正文 + 补充材料） ----------------
        self.list_files = QListWidget()
        self.list_files.setMaximumHeight(96)
        self.list_files.setToolTip(
            "支持 " + " / ".join(SUPPORTED_SUFFIXES) + "\n"
            "BCI/EEG 论文的训练细节常在正文之后的补充材料里（Supplementary / "
            "Appendix / 附录），把正文和补充材料一起拖进来最稳")
        self.btn_add_file = QPushButton("添加文件…")
        self.btn_del_file = QPushButton("移出选中")
        self.btn_clear_file = QPushButton("清空")
        self.btn_add_file.clicked.connect(self._pick_files)
        self.btn_del_file.clicked.connect(self._remove_selected)
        self.btn_clear_file.clicked.connect(self._clear_files)
        row_lf = QHBoxLayout()
        row_lf.addWidget(self.btn_add_file)
        row_lf.addWidget(self.btn_del_file)
        row_lf.addWidget(self.btn_clear_file)
        row_lf.addStretch(1)
        self.chk_prefer_supp = QCheckBox("优先补充材料/实现细节小节")
        self.chk_prefer_supp.setChecked(True)
        self.chk_prefer_supp.setToolTip(
            "开启后会先在每个文件里定位「Implementation details / Experimental setup / "
            "超参数 / 实验设置」这类小节，并让补充材料排在正文之前（正文里的"
            "「详见补充材料」只是线索，真正参数往往不在正文）")
        self.chk_prefer_supp.toggled.connect(lambda *_: self._assemble())
        self.spin_maxchars = QSpinBox()
        self.spin_maxchars.setRange(1000, 200000)
        self.spin_maxchars.setSingleStep(1000)
        self.spin_maxchars.setValue(12000)
        self.spin_maxchars.setSuffix(" 字")
        self.spin_maxchars.setToolTip("组装文本的上限（控制 LLM 成本/上下文）")
        self.spin_maxchars.valueChanged.connect(lambda *_: self._assemble())
        row_cf = QHBoxLayout()
        row_cf.addWidget(self.chk_prefer_supp)
        row_cf.addStretch(1)
        row_cf.addWidget(QLabel("上限"))
        row_cf.addWidget(self.spin_maxchars)
        self.lbl_asm = QLabel("（还没拖入文件；也可以直接把方法段落粘贴到下面）")
        self.lbl_asm.setWordWrap(True)
        self.lbl_asm.setMinimumWidth(120)
        self.lbl_asm.setStyleSheet("color:#666; font-size:11px;")
        box_files = QGroupBox("论文文件（可直接拖进来；正文 + 补充材料一起更好）")
        lay_files = QVBoxLayout(box_files)
        lay_files.addWidget(self.list_files)
        lay_files.addLayout(row_lf)
        lay_files.addLayout(row_cf)
        lay_files.addWidget(self.lbl_asm)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(box_files, 2)
        ll.addWidget(box_paper, 3)
        ll.addWidget(box_llm, 2)

        # ---------------- 右上：结果表 ----------------
        self.tbl = QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(["字段", "识别值", "我的修正", "来源", "原文依据"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setColumnWidth(0, 150)
        self.tbl.setColumnWidth(1, 120)
        self.tbl.setColumnWidth(2, 110)
        self.tbl.setColumnWidth(3, 130)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.lbl_meta = QLabel("提示：直接粘贴方法段落 → 点「开始识别」。没配 API 密钥也能用（内置规则）。")
        self.lbl_meta.setWordWrap(True)
        self.lbl_meta.setMinimumWidth(120)

        self.txt_yaml = QPlainTextEdit()
        self.txt_yaml.setReadOnly(True)
        self.txt_yaml.setStyleSheet("font-size:11px;")

        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(120)
        self.txt_log.setStyleSheet("font-size:11px;")

        self.chk_merge = QCheckBox("合并到训练控制台当前配置（只覆盖识别到的字段）")
        self.chk_merge.setChecked(True)
        self.chk_use_llm = QCheckBox("使用 LLM（需密钥）")
        self.chk_use_llm.setChecked(True)
        self.chk_use_llm.setToolTip("取消勾选 = 只用内置规则（完全离线，不联网）")
        self.btn_run = QPushButton("开始识别")
        self.btn_run.setProperty("accent", True)
        self.btn_run.clicked.connect(self.run_extract)
        self.btn_apply = QPushButton("应用到训练配置")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._apply)
        self.btn_export = QPushButton("导出 YAML…")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export)
        self.btn_copy = QPushButton("复制 YAML")
        self.btn_copy.setEnabled(False)
        self.btn_copy.clicked.connect(
            lambda: QApplication.clipboard().setText(self.txt_yaml.toPlainText()))
        self.chk_bind = QCheckBox("来源写进实验记录")
        self.chk_bind.setChecked(True)
        self.chk_bind.setToolTip(
            "勾选后，抽取来源（文件/小节/每个字段的原文依据/文本 sha256）会随"
            "config.yaml 一起落盘到 experiments/<实验名>/，并出现在 report.md 的"
            "「训练参数来源」一节与实验列表备注里")
        self.btn_prov = QPushButton("查看溯源…")
        self.btn_prov.setEnabled(False)
        self.btn_prov.clicked.connect(self._show_provenance)
        row_btn = QHBoxLayout()
        row_btn.addWidget(self.btn_run)
        row_btn.addWidget(self.chk_use_llm)
        row_btn.addWidget(self.chk_merge)
        row_btn.addWidget(self.chk_bind)
        row_btn.addWidget(self.btn_prov)
        row_btn.addStretch(1)
        row_btn.addWidget(self.btn_apply)
        row_btn.addWidget(self.btn_export)
        row_btn.addWidget(self.btn_copy)

        self.lbl_aug_warn = QLabel("")
        self.lbl_aug_warn.setWordWrap(True)
        self.lbl_aug_warn.setMinimumWidth(120)
        self.lbl_aug_warn.setStyleSheet("color:#b06a00; font-size:11px;")

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(self.lbl_meta)
        rl.addWidget(self.tbl, 3)
        rl.addWidget(QLabel("将要写入的训练配置（YAML 预览）"))
        rl.addWidget(self.lbl_aug_warn)
        rl.addWidget(self.txt_yaml, 2)
        rl.addLayout(row_btn)
        rl.addWidget(self.txt_log)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)
        split.setSizes([470, 770])
        root = QVBoxLayout(self)
        root.addWidget(split, 1)

        self.combo_provider.currentIndexChanged.connect(self._on_provider_changed)
        self._on_provider_changed()
        self._fill_table({})
        self._refresh_preview()

    # ================================================================ 设置
    def _on_provider_changed(self):
        key = self.combo_provider.currentData() or "deepseek"
        spec = PROVIDERS.get(key, {})
        if not self.edit_model.text().strip() or self.edit_model.text() == self.cfg.model:
            self.edit_model.setText(str(spec.get("model", "")))
        if not self.edit_base.text().strip() or self.edit_base.text() == self.cfg.base_url:
            self.edit_base.setText(str(spec.get("base_url", "")))
        env = spec.get("env") or ""
        self.lbl_key_hint.setText(
            f"· 也可用环境变量：{env}（或通用 BCI_LLM_API_KEY）；密钥只存本机，"
            f"配置文件 {CONFIG_PATH.relative_to(PROJECT_ROOT)} 已在 .gitignore 里\n"
            f"· 不填密钥 = 只用内置规则抽取（离线可用，不联网）")

    def current_cfg(self) -> LLMConfig:
        return LLMConfig(
            provider=self.combo_provider.currentData() or "deepseek",
            model=self.edit_model.text().strip(),
            base_url=self.edit_base.text().strip(),
            api_key=self.edit_key.text().strip(),
            timeout=float(self.spin_timeout.value()),
        ).resolved()

    def _test_connection(self):
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "提示", "正在识别中，请稍候")
            return
        cfg = self.current_cfg()
        self.txt_log.appendPlainText(f"测试连接：{cfg.describe()}")

        def job(log=print):
            return test_connection(cfg, log=log)

        th = FunctionThread(job, parent=self)
        self._thread = th
        th.logged.connect(lambda s: self.txt_log.appendPlainText(str(s)))
        th.ok.connect(lambda msg: self.txt_log.appendPlainText(str(msg)))
        th.fail.connect(lambda m: QMessageBox.warning(self, "连接失败", m))
        th.start()

    # ---------------------------------------------------------- 拖拽与文件
    def dragEnterEvent(self, event):                 # noqa: N802 - Qt 命名
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if any(is_supported(u.toLocalFile()) for u in urls):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):                      # noqa: N802
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        keep = [p for p in paths if is_supported(p)]
        skipped = [Path(p).name for p in paths if p and not is_supported(p)]
        if skipped:
            self.txt_log.appendPlainText(
                f"[跳过] 不支持的文件类型：{', '.join(skipped[:4])}"
                f"（支持 {'/'.join(SUPPORTED_SUFFIXES)}）")
        if keep:
            event.acceptProposedAction()
            self._add_paths(keep)

    def _pick_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择论文文件（可多选：正文 + 补充材料）", str(Path.home()),
            "论文文件 (*.pdf *.txt *.md *.markdown *.tex *.rst);;所有文件 (*)")
        self._add_paths(files)

    def _add_paths(self, paths):
        """读入文件（PDF 解析放后台线程，避免界面卡死），然后组装文本。"""
        paths = [p for p in paths if p]
        if not paths:
            return
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "提示", "正在处理上一个任务，请稍候")
            return
        have = {src.path for src, _ in self._files}
        paths = [p for p in paths if str(Path(p)) not in have]
        if not paths:
            return
        self.btn_run.setEnabled(False)
        self.txt_log.appendPlainText(f"正在读取 {len(paths)} 个文件…")

        def job(log=print):
            return read_sources(paths, log=log)

        def done(res):
            srcs, texts = res
            self._files += [(s, texts[s.path]) for s in srcs]
            self.btn_run.setEnabled(True)
            self._refresh_file_list()
            self._assemble()

        th = FunctionThread(job, parent=self)
        self._thread = th
        th.logged.connect(lambda t: self.txt_log.appendPlainText(str(t)))
        th.ok.connect(done)
        th.fail.connect(lambda m: (self.btn_run.setEnabled(True),
                                   QMessageBox.critical(self, "读取失败", m)))
        th.start()

    def _refresh_file_list(self):
        self.list_files.clear()
        for src, _ in self._files:
            it = QListWidgetItem(
                f"{src.name} — {src.kind_label}"
                + (f"，{src.n_pages} 页" if src.n_pages else "")
                + f"，{src.n_chars} 字"
                + (f"  ⚠{src.note}" if src.note else ""))
            it.setToolTip(src.path)
            self.list_files.addItem(it)

    def _remove_selected(self):
        rows = sorted((self.list_files.row(i) for i in self.list_files.selectedItems()),
                      reverse=True)
        for r in rows:
            if 0 <= r < len(self._files):
                self._files.pop(r)
        self._refresh_file_list()
        self._assemble()

    def _clear_files(self):
        self._files.clear()
        self._asm_text, self._asm_info = "", None
        self._refresh_file_list()
        self.lbl_asm.setText("（已清空文件；仍可直接粘贴文本）")

    def _assemble(self):
        """把已读入的文件组装成"补充材料优先"的抽取文本，填进文本框。"""
        if not self._files:
            return
        try:
            text, info = assemble_param_text(
                self._files, max_chars=int(self.spin_maxchars.value()),
                prefer_supplementary=self.chk_prefer_supp.isChecked(),
                log=lambda t: self.txt_log.appendPlainText(str(t)))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "组装失败", str(e))
            return
        self._asm_text, self._asm_info = text, info
        self.txt_paper.setPlainText(text)
        secs = info.get("used_sections") or []
        txt = (f"已组装 {len(self._files)} 个文件 → {info['n_chars']} 字"
               + ("（已截断）" if info.get("truncated") else "")
               + (f"；优先小节：{secs[0]}" if secs else "；未找到明确参数小节，已取开头兜底"))
        if info.get("notes"):
            txt += "\n⚠ " + "；".join(info["notes"][:2])
        self.lbl_asm.setText(txt)

    def _is_assembled_text(self) -> bool:
        """当前文本框内容是否 == 最近一次从文件组装出的内容（手动改过就不算）。"""
        return bool(self._asm_text) and \
            self.txt_paper.toPlainText().strip() == self._asm_text.strip()

    # ================================================================ 识别
    def run_extract(self):
        text = self.txt_paper.toPlainText().strip()
        if len(text) < 20:
            QMessageBox.information(self, "提示", "请先粘贴论文方法段落（至少 20 个字符）")
            return
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "提示", "上一次识别还在进行中")
            return
        cfg = self.current_cfg()
        if self.chk_save_key.isChecked() and cfg.api_key:
            try:
                p = save_config_file(cfg)
                self.txt_log.appendPlainText(f"配置（含密钥）已保存 → {p}")
            except Exception as e:  # noqa: BLE001
                self.txt_log.appendPlainText(f"[警告] 保存配置失败：{e}")
        use_llm = self.chk_use_llm.isChecked()
        self.btn_run.setEnabled(False)
        self.txt_log.clear()
        from_files = self._is_assembled_text()
        self.txt_log.appendPlainText(
            f"开始识别：{len(text)} 字"
            + (f"（来自 {len(self._files)} 个文件）" if from_files else "（粘贴/编辑的文本）")
            + f"，LLM={'开' if use_llm else '关（只用规则）'}；"
            f"{cfg.describe() if use_llm else '内置规则'}")

        def job(log=print):
            return extract_train_params(text, cfg=cfg, use_llm=use_llm, log=log)

        th = FunctionThread(job, parent=self)
        self._thread = th
        th.logged.connect(lambda s: self.txt_log.appendPlainText(str(s)))
        th.ok.connect(self._done)
        th.fail.connect(self._failed)
        th.start()

    def _failed(self, msg: str):
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "识别失败", msg)

    def _done(self, res: dict):
        self.btn_run.setEnabled(True)
        self._result = res
        info = self._asm_info if self._is_assembled_text() else None
        if info is None and self._files:
            info = {"sources": [], "notes": ["文本框内容与组装结果不一致（多半手工改过），"
                                            "溯源只记录文本本身"]}
        self._prov = provenance(res, self.txt_paper.toPlainText(), info)
        if not self._files:
            self._prov["input"]["notes"] = list(self._prov["input"].get("notes") or []) + \
                ["文本来自手工粘贴/编辑，未关联文件"]
        self.btn_prov.setEnabled(True)
        self._fill_table(res["fields"])
        meta = res["meta"]
        txt = (f"{meta['n_chars']} 字 · {meta['elapsed']}s · 规则命中 {meta['n_rules']} 项"
               + (f" · LLM({meta['model']}) 命中 {meta['n_llm']} 项" if meta["used_llm"]
                  else " · 未用 LLM"))
        for w in meta.get("warnings") or []:
            txt += f"\n⚠ {w}"
        missing = [f.label for f in FIELDS
                   if not (res["fields"].get(f.key) or {}).get("value")]
        if missing:
            txt += f"\n论文里没写（保持当前配置）：{'、'.join(missing)}"
        self.lbl_meta.setText(txt)
        self.txt_log.appendPlainText(format_report(res))
        self._refresh_preview()
        self.btn_apply.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_copy.setEnabled(True)
        if self._files:
            self.lbl_asm.setText(self.lbl_asm.text().split("\n⚠")[0]
                                 + f"；已记录来源：{summarize_provenance(self._prov)}")

    # ================================================================ 表格
    def _fill_table(self, fields: dict):
        self.tbl.setRowCount(len(FIELDS))
        for i, f in enumerate(FIELDS):
            item = (fields.get(f.key) or {})
            value = item.get("value")
            show = ("、".join(value) if isinstance(value, list) else
                    ("是" if value is True else "否" if value is False else
                     ("—" if value is None else
                      (f"{value:.6g}" if isinstance(value, float) else str(value)))))
            src = str(item.get("source") or "未识别")
            ev = str(item.get("evidence") or "")
            if item.get("conflict"):
                src += " ⚠冲突"
                ev = f"（修正建议：{item['conflict']}）"
            cells = [f"{f.label}", show, "", src, ev]
            for j, txt in enumerate(cells):
                it = QTableWidgetItem(txt)
                if j == 0:
                    it.setData(Qt.ItemDataRole.UserRole, f.key)
                    if value is None:
                        it.setForeground(Qt.GlobalColor.gray)
                if j == 2:                       # 修正列可编辑
                    it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable)
                    it.setToolTip("留空 = 用左侧识别值；填了就以你填的为准\n"
                                  f"{f.hint or ''}" + (f"可选：{'/'.join(f.options)}" if f.options else ""))
                else:
                    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if j == 1 and value is not None:
                    it.setForeground(Qt.GlobalColor.darkGreen)
                self.tbl.setItem(i, j, it)
        self.tbl.resizeColumnsToContents()
        self.tbl.setColumnWidth(0, max(150, self.tbl.columnWidth(0)))
        self.tbl.horizontalHeader().setStretchLastSection(True)

    def edited_fields(self) -> dict:
        """把表格里的修正合并回字段（修正列优先）。"""
        out = dict((self._result or {}).get("fields") or {})
        for i in range(self.tbl.rowCount()):
            key_item = self.tbl.item(i, 0)
            if key_item is None:
                continue
            key = key_item.data(Qt.ItemDataRole.UserRole)
            fix_item = self.tbl.item(i, 2)
            if not key or fix_item is None or not fix_item.text().strip():
                continue
            val = parse_field_text(key, fix_item.text())
            if val is None and fix_item.text().strip():
                self.txt_log.appendPlainText(f"[警告] 「{key}」的修正值无法解析：{fix_item.text()}")
                continue
            out[key] = {"value": val, "evidence": "人工修正", "source": "人工修正"}
        return out

    def _refresh_preview(self):
        import yaml

        fields = self.edited_fields()
        base = self._base_config() if self.chk_merge.isChecked() else {}
        cfg = to_train_config(fields, base)
        aug = cfg.get("augmentation") or {}
        # 本平台训练器不支持的手段：从待写入的配置里摘掉，但在界面上明确提示（不静默丢弃）
        unsupported = aug.pop("unsupported_types", None)
        self.lbl_aug_warn.setText(
            "" if not unsupported else
            ("⚠ 论文提到的这些增强手段本平台暂不支持，未写入配置：" + "、".join(unsupported)
             + "（可手写进自定义模型/训练脚本）"))
        self.txt_yaml.setPlainText(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        return cfg

    # ================================================================ 应用
    def _apply(self):
        if self._on_apply is None:
            QMessageBox.information(self, "提示", "当前上下文没有可写入的训练配置")
            return
        cfg = self._refresh_preview()
        prov = self._prov if (self.chk_bind.isChecked() and self._prov) else None
        try:                                        # 兼容 1 参数/2 参数两种回调
            n = len(inspect.signature(self._on_apply).parameters)
            self._on_apply(cfg, prov) if n >= 2 else self._on_apply(cfg)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "应用失败", str(e))
            return
        self.txt_log.appendPlainText(
            "已应用到训练控制台 ✓" + (f"（来源已绑定：{summarize_provenance(prov)}）"
                                   if prov else "（未绑定来源）"))

    def _show_provenance(self):
        if not self._prov:
            QMessageBox.information(self, "提示", "先识别一次")
            return
        import yaml

        bind = self.chk_bind.isChecked()
        head = ("将写入实验记录（config.yaml 里" + "）" + "\n\n") if bind else \
            ("未勾选「来源写进实验记录」，本次不落盘；下面仅供参考：\n\n")
        d = QDialog(self)
        d.setWindowTitle("抽取溯源（可随实验记录一起保存）")
        lay = QVBoxLayout(d)
        editor = QPlainTextEdit(head + yaml.safe_dump(self._prov, allow_unicode=True,
                                                     sort_keys=False, width=100))
        editor.setReadOnly(True)
        lay.addWidget(editor, 1)
        b = QPushButton("复制")
        b.clicked.connect(lambda: QApplication.clipboard().setText(editor.toPlainText()))
        lay.addWidget(b)
        d.resize(880, 640)
        d.exec()

    def _export(self):
        import yaml

        cfg = self._refresh_preview()
        path, _ = QFileDialog.getSaveFileName(self, "导出训练配置", "train_from_paper.yaml",
                                              "YAML (*.yaml)")
        if not path:
            return
        Path(path).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                              encoding="utf-8")
        self.txt_log.appendPlainText(f"已导出 → {path}")
