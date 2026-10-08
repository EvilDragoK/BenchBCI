"""预处理工作台：流水线编辑器（增删/排序/启停）+ 参数面板 + 即时预览 + 批量应用。"""
from __future__ import annotations

from pathlib import Path

import yaml
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem, QMenu,
                             QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
                             QSpinBox, QToolButton, QVBoxLayout, QWidget)

from core import PROJECT_ROOT
from core.pipeline import Pipeline, load_config, run_dataset_pipeline

from core.preprocess.operators import OPERATORS, describe_operator
from .widgets.pipeline_preview_dialog import PipelinePreviewDialog
from .widgets.psd_dialog import PsdCompareDialog
from .workers import FunctionThread


class PreprocessPage(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._threads: list[FunctionThread] = []
        self._form_getters: dict[str, object] = {}
        self._form_op = None
        self._steps: list[dict] = []

        # ================= 左侧：步骤列表 =================
        self.list_steps = QListWidget()
        self.list_steps.setMinimumWidth(280)
        btn_add = QToolButton()
        btn_add.setText("＋ 添加步骤")
        btn_add.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(btn_add)
        for name, cls in OPERATORS.items():
            menu.addAction(f"{cls.title}  ({name})", lambda n=name: self._add_step(n))
        btn_add.setMenu(menu)
        self.btn_del = QPushButton("删除")
        self.btn_up = QPushButton("上移")
        self.btn_down = QPushButton("下移")
        self.btn_save = QPushButton("保存 YAML")
        self.btn_loadcfg = QPushButton("加载 YAML")

        left_lay = QVBoxLayout()
        left_lay.addWidget(QLabel("流水线步骤（勾选=启用，顺序自上而下）"))
        hint_pl = QLabel("此处从原始数据独立批量处理（不受「数据管理」页浏览操作的影响），"
                         "结果写入 processed/ 供训练使用，可命令行复现。")
        hint_pl.setWordWrap(True)
        # 中文长句没有空格可断词，wordWrap 标签的最小宽度≈整句宽度，
        # 会把整页的最小宽度撑到上千像素；显式给个小最小宽度即可正常换行。
        hint_pl.setMinimumWidth(120)
        hint_pl.setStyleSheet("color:#888; font-size:11px;")
        left_lay.addWidget(hint_pl)
        left_lay.addWidget(self.list_steps, 1)
        row = QHBoxLayout()
        row.addWidget(btn_add)
        row.addWidget(self.btn_del)
        row2 = QHBoxLayout()
        row2.addWidget(self.btn_up)
        row2.addWidget(self.btn_down)
        row3 = QHBoxLayout()
        row3.addWidget(self.btn_save)
        row3.addWidget(self.btn_loadcfg)
        left_lay.addLayout(row)
        left_lay.addLayout(row2)
        left_lay.addLayout(row3)

        left = QWidget()
        left.setLayout(left_lay)

        # ================= 右上：参数面板 =================
        self.form_holder = QWidget()
        self.form_layout = QFormLayout(self.form_holder)
        self.form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.form_holder)

        # ================= 右下：数据集与执行 =================
        self.edit_root = QLineEdit()
        self.edit_subjects = QLineEdit("1-9")
        self.edit_sessions = QLineEdit("T,E")
        self.chk_eeg_only = QCheckBox("仅保留 EEG 通道")
        self.chk_eeg_only.setChecked(True)
        self.edit_name = QLineEdit("bnci2014001_standard_mi")
        self.edit_out = QLineEdit(str(PROJECT_ROOT / "processed" / "bnci2014001"))
        self.btn_browse_out = QPushButton("…")
        self.btn_preview = QPushButton("预览当前步骤")
        self.btn_preview_all = QPushButton("一键预览整条流水线…")
        self.btn_preview_all.setToolTip("跑完所有步骤，输出执行报告表与最终结果可视化")
        self.btn_apply = QPushButton("▶ 应用到全部受试者")
        self.btn_apply.setStyleSheet("font-weight: bold;")

        ds_box = QGroupBox("数据集与输出")
        f = QFormLayout(ds_box)
        f.addRow("数据集根目录", self.edit_root)
        f.addRow("被试（如 1-9 或 1,3,5）", self.edit_subjects)
        f.addRow("Sessions（T,E）", self.edit_sessions)
        f.addRow("", self.chk_eeg_only)
        f.addRow("流水线名称", self.edit_name)
        out_row = QHBoxLayout()
        out_row.addWidget(self.edit_out, 1)
        out_row.addWidget(self.btn_browse_out)
        w = QWidget()
        w.setLayout(out_row)
        f.addRow("输出目录", w)
        prev_row = QHBoxLayout()
        prev_row.addWidget(self.btn_preview)
        prev_row.addWidget(self.btn_preview_all)
        pw = QWidget()
        pw.setLayout(prev_row)
        f.addRow(pw)
        f.addRow(self.btn_apply)

        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(220)

        right_lay = QVBoxLayout()
        right_lay.addWidget(QLabel("参数设置"))
        right_lay.addWidget(self.scroll, 1)
        right_lay.addWidget(ds_box)
        right_lay.addWidget(QLabel("执行日志"))
        right_lay.addWidget(self.txt_log)

        root = QHBoxLayout(self)
        root.addWidget(left)
        root.addLayout(right_lay, 1)

        # ---- 信号 ----
        self.list_steps.itemChanged.connect(lambda *_: self._sync_enabled())
        self.list_steps.currentRowChanged.connect(self._build_form)
        self.btn_del.clicked.connect(self._remove_step)
        self.btn_up.clicked.connect(lambda: self._move(-1))
        self.btn_down.clicked.connect(lambda: self._move(+1))
        self.btn_save.clicked.connect(self.save_yaml)
        self.btn_loadcfg.clicked.connect(self.load_yaml)
        self.btn_browse_out.clicked.connect(self._browse_out)
        self.btn_preview.clicked.connect(self.preview_step)
        self.btn_preview_all.clicked.connect(self.preview_all)
        self.btn_apply.clicked.connect(self.apply_all)

        self._load_default_pipeline()

    # ================================================================ 步骤管理
    def _add_step(self, name: str):
        cls = OPERATORS[name]
        self._steps.append({"op": name, "params": dict(cls({}).params), "enabled": True})
        self._sync_rows()
        self.list_steps.setCurrentRow(self.list_steps.count() - 1)

    def _sync_rows(self):
        """list widget ←→ _steps 数据同步（重建显示）。"""
        self.list_steps.blockSignals(True)
        self.list_steps.clear()
        for s in self._steps:
            cls = OPERATORS[s["op"]]
            it = QListWidgetItem(cls.title)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if s["enabled"] else Qt.CheckState.Unchecked)
            it.setToolTip(s["op"])
            self.list_steps.addItem(it)
        self.list_steps.blockSignals(False)

    def _sync_enabled(self):
        for i in range(self.list_steps.count()):
            if i < len(self._steps):
                self._steps[i]["enabled"] = (
                    self.list_steps.item(i).checkState() == Qt.CheckState.Checked)

    def _remove_step(self):
        row = self.list_steps.currentRow()
        if row >= 0 and row < len(self._steps):
            self._steps.pop(row)
            self._sync_rows()
            self._build_form(self.list_steps.currentRow())

    def _browse_out(self):
        from PyQt6.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(self, "选择输出目录",
                                                self.edit_out.text().strip() or str(PROJECT_ROOT))
        if path:
            self.edit_out.setText(path)

    def _move(self, d: int):
        row = self.list_steps.currentRow()
        j = row + d
        if 0 <= row < len(self._steps) and 0 <= j < len(self._steps):
            self._steps[row], self._steps[j] = self._steps[j], self._steps[row]
            self._sync_rows()
            self.list_steps.setCurrentRow(j)

    # ================================================================ 参数表单
    def _build_form(self, row: int):
        # 清空旧表单
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._form_getters = {}
        self._form_op = None
        if row < 0 or row >= len(self._steps):
            self.form_layout.addRow(QLabel("（请选择一个步骤）"))
            return
        s = self._steps[row]
        op = OPERATORS[s["op"]](s["params"])
        self._form_op = (row, op)
        title = QLabel(f"【{op.title}】")
        title.setStyleSheet("font-weight:bold; font-size:13px;")
        self.form_layout.addRow(title)
        desc = describe_operator(s["op"])
        if desc:
            lb = QLabel(desc)
            lb.setWordWrap(True)
            lb.setMinimumWidth(120)     # 允许换行，避免长中文说明撑宽整页
            lb.setStyleSheet("color:#333; background:#f2f5f8; padding:6px;"
                             "border-radius:4px; border:1px solid #dde3ea;")
            self.form_layout.addRow(lb)
        if s["op"] == "ica":          # ICA 专属：成分浏览/挑选入口（放在参数之前，保证可见）
            btn_ica = QPushButton("浏览 ICA 成分（地形图 / 时序）…")
            btn_ica.setToolTip(
                "用当前参数在「数据管理」页已加载的数据上拟合 ICA（自动先跑前序步骤），\n"
                "逐个查看成分地形图、时序与诊断指标（峰度/自相关/额区相关/肌电比），\n"
                "勾选后写回「手动剔除成分编号」")
            btn_ica.clicked.connect(self._browse_ica)
            self.form_layout.addRow(btn_ica)

        for k, spec in op.schema.items():
            typ, default, label = spec[0], spec[1], spec[2]
            choices = spec[3] if len(spec) > 3 else None
            widget = self._make_widget(typ, default, choices, op, k)
            self.form_layout.addRow(label, widget)
        hint = QLabel("提示：导联/滤波/重参考/坏段/裁剪/回归/ICA 都作用在分段之前（raw 阶段），"
                      "振幅与 IQR 剔除在分段之后（epochs 阶段）。")
        hint.setWordWrap(True)
        hint.setMinimumWidth(120)
        hint.setStyleSheet("color: gray;")
        self.form_layout.addRow(hint)

    def _make_widget(self, typ, default, choices, op, key):
        if typ == "bool":
            w = QCheckBox()
            w.setChecked(bool(default))
            w.toggled.connect(lambda v, k=key: self._set_param(k, v))
            self._form_getters[key] = w.isChecked
            return w
        if typ == "choice":
            w = QComboBox()
            w.addItems(choices or [])
            w.setCurrentText(str(default))
            w.currentTextChanged.connect(lambda v, k=key: self._set_param(k, v))
            self._form_getters[key] = w.currentText
            return w
        if typ == "int":
            w = QSpinBox()
            w.setRange(-100000, 100000)
            w.setValue(int(default))
            w.valueChanged.connect(lambda v, k=key: self._set_param(k, v))
            self._form_getters[key] = w.value
            return w
        if typ == "float":
            w = QDoubleSpinBox()
            w.setRange(-1e6, 1e6)
            w.setDecimals(4)
            w.setValue(float(default))
            w.valueChanged.connect(lambda v, k=key: self._set_param(k, v))
            self._form_getters[key] = w.value
            return w
        # str / list_*：逗号分隔
        w = QLineEdit()
        if isinstance(default, (list, tuple)):
            w.setText(", ".join(str(x) for x in default))
        else:
            w.setText(str(default))
        w.editingFinished.connect(lambda k=key, w=w: self._set_param(k, w.text()))
        self._form_getters[key] = w.text
        return w

    def _set_param(self, key, value):
        if self._form_op is None:
            return
        row, op = self._form_op
        op.set_param(key, value)
        self._steps[row]["params"] = dict(op.params)

    # ================================================================ ICA 成分浏览
    def _browse_ica(self):
        """在「ICA 之前的数据」上拟合 ICA，打开成分浏览器并写回剔除编号。"""
        if self.state.raw is None:
            QMessageBox.information(self, "提示", "请先在「数据管理」页加载数据")
            return
        row = self.list_steps.currentRow()
        if row < 0 or row >= len(self._steps) or self._steps[row]["op"] != "ica":
            QMessageBox.information(self, "提示", "请先在左侧选择「ICA 伪迹剔除」步骤")
            return
        self._collect_params_from_form()
        prev = self._steps[:row]
        raw, used = self._preview_crop(self.state.raw)
        self._log(f"准备 ICA 浏览数据（取前 {used:.0f} s；先执行 ICA 之前的 "
                  f"{sum(1 for s in prev if s['enabled'])} 个启用步骤）…")

        def job(log=print):
            obj = raw
            if any(s["enabled"] for s in prev):
                pipe = Pipeline([(OPERATORS[s["op"]](s["params"]), s["enabled"]) for s in prev])
                obj, _ = pipe.run(raw)
            return obj

        def done(obj):
            import mne

            if not isinstance(obj, mne.BaseRaw):
                QMessageBox.warning(self, "提示", "ICA 之前已经分段，无法按 raw 浏览成分")
                return
            from .widgets.ica_dialog import IcaBrowserDialog

            params = self._steps[row]["params"]
            dlg = IcaBrowserDialog(obj, params, parent=self,
                                   exclude=params.get("exclude") or [])
            dlg.fit()
            if dlg.exec():
                self._steps[row]["params"]["exclude"] = list(dlg.exclude)
                self._build_form(row)
                self._log(f"ICA 剔除成分已更新为 {dlg.exclude}"
                          f"（记得保存 YAML / 应用到全部受试者）")

        th = FunctionThread(job, parent=self)
        self._threads.append(th)
        th.ok.connect(done)
        th.fail.connect(lambda m: QMessageBox.critical(self, "准备数据失败", m))
        th.start()

    # ================================================================ 配置构建
    def _collect_params_from_form(self):
        """把当前表单值（未触发的编辑）写回步骤数据。"""
        if self._form_op is None:
            return
        row, op = self._form_op
        for k, getter in self._form_getters.items():
            try:
                op.set_param(k, getter())
            except Exception:  # noqa: BLE001
                pass
        self._steps[row]["params"] = dict(op.params)

    def build_config(self) -> dict:
        self._collect_params_from_form()
        sessions = [s.strip().upper() for s in self.edit_sessions.text().split(",") if s.strip()]
        return {
            "name": self.edit_name.text().strip() or "pipeline",
            "dataset": {
                "root": self.edit_root.text().strip(),
                "subjects": self.edit_subjects.text().strip() or "all",
                "sessions": sessions or ["T"],
                "eeg_only": self.chk_eeg_only.isChecked(),
            },
            "steps": [
                {"op": s["op"], "params": s["params"], "enabled": bool(s["enabled"])}
                for s in self._steps
            ],
            "output": {
                "dir": self.edit_out.text().strip(),
                "save_fif": True,
                "save_npy": True,
            },
        }

    # ================================================================ YAML
    def save_yaml(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存流水线", str(PROJECT_ROOT / "configs"),
                                              "YAML (*.yaml *.yml)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.build_config(), f, allow_unicode=True, sort_keys=False)
        self._log(f"流水线已保存 → {path}")

    def load_yaml(self):
        path, _ = QFileDialog.getOpenFileName(self, "加载流水线", str(PROJECT_ROOT / "configs"),
                                              "YAML (*.yaml *.yml);;所有文件 (*)")
        if not path:
            return
        try:
            cfg = load_config(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "加载失败", str(e))
            return
        self._steps = [
            {"op": s["op"], "params": s.get("params") or {}, "enabled": bool(s.get("enabled", True))}
            for s in cfg.get("steps", [])
        ]
        ds = cfg.get("dataset", {})
        self.edit_root.setText(ds.get("root", ""))
        self.edit_subjects.setText(str(ds.get("subjects", "1-9")))
        self.edit_sessions.setText(",".join(ds.get("sessions", ["T"])))
        self.chk_eeg_only.setChecked(bool(ds.get("eeg_only", True)))
        self.edit_name.setText(cfg.get("name", "pipeline"))
        self.edit_out.setText(cfg.get("output", {}).get("dir", self.edit_out.text()))
        self._sync_rows()
        if self.list_steps.count():
            self.list_steps.setCurrentRow(0)
        self._log(f"流水线已加载 ← {path}")

    def _load_default_pipeline(self):
        default_cfg = PROJECT_ROOT / "configs" / "preprocess_bnci2014001.yaml"
        if default_cfg.exists():
            try:
                cfg = load_config(default_cfg)
                self._steps = [
                    {"op": s["op"], "params": s.get("params") or {},
                     "enabled": bool(s.get("enabled", True))}
                    for s in cfg.get("steps", [])
                ]
                ds = cfg.get("dataset", {})
                self.edit_root.setText(ds.get("root", ""))
                self.edit_subjects.setText(str(ds.get("subjects", "1-9")))
                self.edit_sessions.setText(",".join(ds.get("sessions", ["T", "E"])))
                self.chk_eeg_only.setChecked(bool(ds.get("eeg_only", True)))
                self.edit_out.setText(cfg.get("output", {}).get("dir", self.edit_out.text()))
            except Exception:  # noqa: BLE001
                self._steps = []
        else:
            self._steps = []
        self._sync_rows()

    # ================================================================ 预览与执行
    @staticmethod
    def _preview_crop(raw, limit: float = 300.0, min_events: int = 20):
        """裁剪出用于预览的数据：**保证至少含 min_events 个事件**。

        有些数据集开头一大段没有事件（如 BNCI2014-001 前 3 个 run 无 cue，
        第一个 cue 在第 368 s），若简单截前 300 s 会让「分段」步骤无事件可提。
        """
        import mne
        import numpy as np

        total = raw.n_times / raw.info["sfreq"]
        if total <= limit:
            return raw.copy(), total
        tmax = limit
        try:
            events, _ = mne.events_from_annotations(raw)
        except Exception:  # noqa: BLE001
            events = np.zeros((0, 3), dtype=int)
        if len(events):
            t_ev = events[:, 0] / raw.info["sfreq"]
            k = min(len(t_ev), max(int(min_events), 1)) - 1
            tmax = max(limit, min(float(t_ev[k]) + 30.0, total))
        return raw.copy().crop(tmax=tmax), tmax

    def preview_step(self):
        if self.state.raw is None:
            QMessageBox.information(self, "提示", "请先在“数据管理”页加载数据")
            return
        self._collect_params_from_form()
        row = self.list_steps.currentRow()
        steps = self._steps[: row + 1]
        raw, used = self._preview_crop(self.state.raw)
        before = raw.copy()
        import mne  # noqa: F401

        def job(log=print):
            pipe = Pipeline([(OPERATORS[s["op"]](s["params"]), s["enabled"]) for s in steps])
            obj, ctx = pipe.run(raw)
            return obj, ctx

        self._log("预览执行中…")
        th = FunctionThread(job, parent=self)
        self._threads.append(th)

        def done(res):
            obj, ctx = res
            for entry in ctx.get("log", []):
                if not entry.get("skipped"):
                    self._log(f"  · {entry['op']} {entry['seconds']}s")
            import mne

            if isinstance(obj, mne.BaseEpochs):
                import numpy as np

                names = {v: k for k, v in (ctx.get("classes") or {}).items()}
                codes, cnts = np.unique(obj.events[:, 2], return_counts=True)
                pretty = {names.get(int(c), c): int(n) for c, n in zip(codes, cnts)}
                self._log(f"分段结果：{len(obj)} epochs {pretty}；"
                          f"拒绝 {ctx.get('rejected_trials', 0)} trials")
                QMessageBox.information(
                    self, "预览完成",
                    f"分段后 {len(obj)} 个 epoch：\n{pretty}\n"
                    f"剔除 {ctx.get('rejected_trials', 0)} 个 trial")
            else:
                dlg = PsdCompareDialog(before, obj, "预处理前后 PSD 对比", self)
                dlg.exec()

        th.ok.connect(done)
        th.fail.connect(lambda m: QMessageBox.critical(self, "预览失败", m))
        th.start()

    def preview_all(self):
        """一键跑完整条流水线：输出执行报告表 + 最终结果可视化。"""
        if self.state.raw is None:
            QMessageBox.information(self, "提示", "请先在“数据管理”页加载数据")
            return
        self._collect_params_from_form()
        if not self._steps:
            QMessageBox.information(self, "提示", "流水线为空，请先添加步骤")
            return
        steps = [(OPERATORS[s["op"]](s["params"]), s["enabled"]) for s in self._steps]
        raw, used = self._preview_crop(self.state.raw)
        before = raw.copy()
        self._log(f"整条流水线预览中…（取前 {used:.0f} s 数据加速，已保证包含事件）")

        def job(log=print):
            return Pipeline(steps).run(raw)

        th = FunctionThread(job, parent=self)
        self._threads.append(th)

        def done(res):
            obj, ctx = res
            total = sum(e.get("seconds", 0.0) for e in ctx.get("log", [])
                        if not e.get("skipped"))
            for entry in ctx.get("log", []):
                tag = "跳过" if entry.get("skipped") else f"{entry.get('seconds', 0):.3f}s"
                self._log(f"  · {entry['op']} {tag}")
            self._log(f"总耗时 {total:.3f}s → 最终输出 {type(obj).__name__}")
            dlg = PipelinePreviewDialog(before, obj, ctx, "整条流水线预览", self)
            dlg.exec()

        th.ok.connect(done)
        th.fail.connect(lambda m: QMessageBox.critical(self, "预览失败", m))
        th.start()

    def apply_all(self):
        config = self.build_config()
        if not config["steps"]:
            QMessageBox.information(self, "提示", "流水线为空")
            return
        out_dir = Path(config["output"]["dir"])
        if QMessageBox.question(
                self, "确认", f"将对被试 {config['dataset']['subjects']} 执行流水线并写入\n{out_dir}\n继续？"
        ) != QMessageBox.StandardButton.Yes:
            return

        th = FunctionThread(run_dataset_pipeline, config, parent=self)
        self._threads.append(th)
        th.logged.connect(self._log)
        th.ok.connect(lambda res: self._log(
            f"完成：{res['n_epochs']} epochs → {res['out_dir']}（详见 {res['out_dir']}/report.md）"))
        th.fail.connect(lambda m: QMessageBox.critical(self, "流水线失败", m))
        th.start()

    def _log(self, text):
        self.txt_log.appendPlainText(str(text))
