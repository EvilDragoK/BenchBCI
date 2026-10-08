"""训练控制台：配置 → 后台训练 → 实时曲线 → 实验列表。"""
from __future__ import annotations

import queue

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                             QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QMenu, QMessageBox, QPlainTextEdit,
                             QPushButton, QSpinBox, QSplitter, QTableWidget,
                             QToolButton, QVBoxLayout, QWidget)
from PyQt6.QtCore import Qt

from core import PROJECT_ROOT
from core.models import MODEL_NAMES, is_classic, is_custom
from core.models.foundation import is_foundation
from core.train import plot_few_shot_curve, run_few_shot_curve, run_training
from core.paper_text import summarize_provenance
from core.pipeline import load_config
from core.experiment import ExperimentManager
from .widgets.curve_widget import LiveCurves
from .workers import FunctionThread

#: 各「动态参数行」适用的划分方案 —— 不在集合里的方案会隐藏该行
#: （例如 loso 用不到 k，选中 loso 时 k 那一行就不显示）
_ROW_SCHEMES: dict[str, set[str]] = {
    "k": {"kfold", "within", "subject_kfold", "block", "nested_cv", "holdout"},
    "shot": {"within", "few_shot"},
    "test_session": {"loso", "few_shot"},
    "folds": {"custom"},
    "test_dir": {"cross_dataset"},
    "grid": {"nested_cv"},
}

#: 编码器学习率输入框的默认值（判断用户是否改过，用于"改过就保持可见"）
_LR_ENCODER_DEFAULT = 1e-5


class TrainPage(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._thread = None
        self._paper_params: dict | None = None      # 论文抽取来源（写入实验记录）
        self._events: queue.Queue = queue.Queue()

        # ================= 左：配置表单 =================
        self.combo_model = QComboBox()
        self.combo_model.setToolTip("内置模型 / EEG 基础模型 / 你自己的自定义模型")
        self._rebuild_model_combo()
        self.btn_model_code = QPushButton("查看模型定义")
        self.btn_model_code.setToolTip("跳转查看该模型的 PyTorch (nn.Module) 源码定义")
        self.btn_model_new = QToolButton()
        self.btn_model_new.setText("自定义模型 ▾")
        self.btn_model_new.setToolTip(
            "两种做法：① 以内置模型定义为模板改写；② 从零设计（先输英文模型名，\n"
            "自动生成 class <名称>(nn.Module) 骨架）→ 检查输入/输出与逐层维度 → 保存为新模型")
        self.btn_model_new.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        _menu_new = QMenu(self.btn_model_new)
        _act_edit = _menu_new.addAction("基于当前模型改写…")
        _act_edit.setToolTip("以「模型」下拉框里选中的模型源码为模板")
        _act_edit.triggered.connect(lambda: self.open_model_editor())
        _act_blank = _menu_new.addAction("从零设计新模型…")
        _act_blank.setToolTip("先输入英文模型名，自动生成 nn.Module 骨架再逐层检查")
        _act_blank.triggered.connect(lambda: self.open_model_editor(blank=True))
        self.btn_model_new.setMenu(_menu_new)
        self.edit_processed = QLineEdit(str(PROJECT_ROOT / "processed" / "bnci2014001"))
        self.combo_scheme = QComboBox()
        self.combo_scheme.addItems(["session", "within", "kfold", "few_shot", "block",
                                    "loso", "subject_kfold", "session_kfold",
                                    "nested_cv", "cross_dataset", "custom", "holdout"])
        self.combo_scheme.setToolTip(
            "【被试内 within-subject】\n"
            "within ：逐被试按比例随机划分——每人内部随机切 1/k 作测试，重复 n_repeats 次\n"
            "         取平均（最常见的「随机划分」做法）。\n"
            "kfold  ：逐被试分层 k 折——每人内部切 k 折轮流当测试（不同被试不混切）。\n"
            "block  ：按记录块（run）划分——同一时间段内的相邻 trial 不跨训练/测试。\n"
            "few_shot：被试内少样本校准——每类仅 n 个样本训练，回答「校准要多少数据」。\n"
            "【跨时间 cross-session】\n"
            "session：同一被试 T session 训练、E session 测试（标准基准）。\n"
            "session_kfold：同人跨 session 双向——T→E 与 E→T 都测一遍。\n"
            "【跨被试 cross-subject】\n"
            "loso   ：留一名被试完全没参与训练（最难，衡量「换个人还能不能用」）。\n"
            "subject_kfold：跨被试分组 k 折——每次留出一整组被试测试（LOSO 的 k 折版）。\n"
            "【高级】\n"
            "nested_cv：嵌套交叉验证——训练折内部再分折选超参，指标无偏。\n"
            "cross_dataset：跨数据集迁移——在 A 数据集训练、B 数据集测试。\n"
            "custom ：自定义折——每一折手工指定 训练/验证/测试 受试者。\n"
            "【调试】\n"
            "holdout：全体单次随机划分——快速调试用，结果波动大。")
        self.spin_k = QSpinBox()
        self.spin_k.setRange(2, 10)
        self.spin_k.setValue(5)
        self.spin_k.setToolTip(
            "kfold ：每名被试内部切 k 折（共 N 名被试 × k 折）\n"
            "holdout：测试集占 1/k（k=5 → 训练 80% / 测试 20%）")
        self.spin_shot = QSpinBox()
        self.spin_shot.setRange(1, 200)
        self.spin_shot.setValue(5)
        self.spin_shot.setToolTip("few_shot：每类用于训练的样本数（如 1/5/10/20）")
        self.spin_repeats = QSpinBox()
        self.spin_repeats.setRange(1, 100)
        self.spin_repeats.setValue(1)
        self.spin_repeats.setToolTip(
            "within / few_shot：重复随机采样次数\n"
            "1  = 每名被试一折（共 N 折，与 EEG-FM-Benchmark 等主流基准一致）\n"
            ">1 = 折数 = 被试数 × n_repeats（重复采样取平均，更稳但更耗时）")
        self.spin_train_pct = QDoubleSpinBox()
        self.spin_train_pct.setDecimals(2)
        self.spin_train_pct.setSingleStep(0.05)
        self.spin_train_pct.setRange(0.0, 1.0)
        self.spin_train_pct.setValue(0.30)
        self.spin_train_pct.setToolTip(
            "within / few_shot：按「每被试每类」比例抽训练样本\n"
            "（对齐 EEG-FM-Benchmark 的 --train_percentage，其默认 0.3）\n"
            "设为 0 表示不用比例，改用「每类样本数 n_shot」或 1/k")
        self.combo_test_session = QComboBox()
        self.combo_test_session.addItems(["all", "T", "E"])
        self.combo_test_session.setToolTip("loso/few_shot：用哪个 session 作为测试数据")
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 2000)
        self.spin_epochs.setValue(60)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(4, 4096)
        self.spin_batch.setValue(64)
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(6)
        self.spin_lr.setRange(1e-6, 1.0)
        self.spin_lr.setValue(0.001)
        self.combo_opt = QComboBox()
        self.combo_opt.addItems(["adamw", "adam", "sgd"])
        self.combo_sched = QComboBox()
        self.combo_sched.addItems(["cosine", "step", "none"])
        self.spin_wd = QDoubleSpinBox()
        self.spin_wd.setDecimals(6)
        self.spin_wd.setRange(0.0, 1.0)
        self.spin_wd.setValue(0.01)
        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(0, 500)
        self.spin_patience.setValue(12)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(42)
        self.chk_weighted = QCheckBox("类别加权 loss")
        self.chk_weighted.setChecked(True)
        self.combo_norm = QComboBox()
        self.combo_norm.addItems([
            "zscore_channel",   # z-score（训练折统计）
            "zscore_epoch",     # Instance Norm
            "minmax_pm",        # Norm → [-1,1]（训练折统计）
            "minmax_channel",   # Norm → [0,1]（训练折统计）
            "minmax_pm_epoch",  # 逐 epoch [-1,1]
            "minmax_epoch",     # 逐 epoch [0,1]
            "p95",              # x / P95
            "iqr",              # IQR Scaling
            "quantile",         # Quantile Norm（秩→正态）
            "alaw",             # A-Law 压扩
            "ema",              # EMA 在线标准化
            "zero_mean",        # 仅去均值
            "none",
        ])
        self.chk_ea = QCheckBox("欧几里得对齐 (EA)")
        self.spin_shift = QSpinBox()
        self.spin_shift.setRange(0, 200)
        self.spin_shift.setValue(0)
        self.spin_noise = QDoubleSpinBox()
        self.spin_noise.setDecimals(3)
        self.spin_noise.setRange(0.0, 1.0)
        self.spin_noise.setValue(0.0)
        self.spin_chdrop = QDoubleSpinBox()
        self.spin_chdrop.setDecimals(3)
        self.spin_chdrop.setRange(0.0, 0.9)
        self.spin_chdrop.setValue(0.0)
        self.edit_name = QLineEdit("my_run")
        self.edit_note = QLineEdit("")
        self.btn_loadcfg = QPushButton("读取配置…")
        self.btn_paper = QPushButton("从论文提取参数…")
        self.btn_paper.setToolTip(
            "粘贴论文「实验设置 / Implementation details」段落 → 自动识别\n"
            "epochs / batch size / 学习率 / 优化器 / 随机种子 / 标准化 /\n"
            "是否欧几里得对齐 / 是否用数据增强（并给出原文依据供核对）\n"
            "· 配了 LLM 密钥用大模型抽取；不配也能用内置规则（离线）\n"
            "· 识别结果可人工修正，一键写回左侧配置")
        self.btn_curve = QPushButton("Few-Shot 学习曲线…")
        self.btn_curve.setToolTip("自动扫描 n_shot ∈ {1,5,10,20}，画「每类需要多少校准样本」曲线")
        self.btn_start = QPushButton("▶ 开始训练")
        self.btn_start.setStyleSheet("font-weight: bold;")
        self.btn_stop = QPushButton("■ 停止")

        box = QGroupBox("训练配置")
        f = QFormLayout(box)
        self._form = f
        self._dyn_rows: dict[str, int] = {}      # 动态参数行：key → 行号
        model_row = QWidget()
        ml = QHBoxLayout(model_row)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(self.combo_model, 1)
        ml.addWidget(self.btn_model_code)
        ml.addWidget(self.btn_model_new)
        f.addRow("模型", model_row)
        f.addRow("预处理数据目录", self.edit_processed)
        scheme_row = QWidget()
        scl = QHBoxLayout(scheme_row)
        scl.setContentsMargins(0, 0, 0, 0)
        scl.addWidget(self.combo_scheme, 1)
        self.btn_scheme_intro = QPushButton("方案介绍…")
        self.btn_scheme_intro.setToolTip("看图理解每种划分方案如何切分数据（颜色取自真实划分结果）")
        scl.addWidget(self.btn_scheme_intro)
        f.addRow("划分方案", scheme_row)
        # —— 高级方案的附加配置 ——
        self._folds: list[dict] = []
        self.lbl_folds = QLabel("（未配置，默认自动 5 折）")
        self.lbl_folds.setStyleSheet("color:#888;")
        self.btn_edit_folds = QPushButton("编辑折配置…")
        folds_row = QWidget()
        fl = QHBoxLayout(folds_row)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(self.lbl_folds, 1)
        fl.addWidget(self.btn_edit_folds)

        self.edit_test_dir = QLineEdit("")
        self.edit_test_dir.setPlaceholderText("cross_dataset 用：目标域预处理目录")
        self.btn_test_dir = QPushButton("…")
        testdir_row = QWidget()
        tl = QHBoxLayout(testdir_row)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.edit_test_dir, 1)
        tl.addWidget(self.btn_test_dir)

        self.edit_grid = QLineEdit("")
        self.edit_grid.setPlaceholderText("nested_cv 用：如 n_components: 4,6,8")

        f.addRow("自定义折（custom）", folds_row)
        self._dyn_rows["folds"] = f.rowCount() - 1
        f.addRow("目标域目录（cross_dataset）", testdir_row)
        self._dyn_rows["test_dir"] = f.rowCount() - 1
        f.addRow("内层网格（nested_cv）", self.edit_grid)
        self._dyn_rows["grid"] = f.rowCount() - 1
        f.addRow("k（kfold/holdout/within）", self.spin_k)
        self._dyn_rows["k"] = f.rowCount() - 1
        shot_row = QWidget()
        sl = QHBoxLayout(shot_row)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(QLabel("每类样本"))
        sl.addWidget(self.spin_shot)
        sl.addWidget(QLabel("重复"))
        sl.addWidget(self.spin_repeats)
        sl.addWidget(QLabel("每类比例"))
        sl.addWidget(self.spin_train_pct)
        f.addRow("within/few_shot 参数", shot_row)
        self._dyn_rows["shot"] = f.rowCount() - 1
        f.addRow("loso 测试 session", self.combo_test_session)
        self._dyn_rows["test_session"] = f.rowCount() - 1

        # ---- 预训练权重 / 微调策略 / 分层学习率 ----
        self.edit_pretrained = QLineEdit("")
        self.edit_pretrained.setPlaceholderText(
            "auto（按模型自动下载）/ hf:repo[/file] / url:... / 本地路径（留空=从零训练）")
        self.btn_pretrained = QPushButton("…")
        self.btn_pretrained.setToolTip("选择本地权重文件")
        self.btn_download = QPushButton("下载/校验")
        self.btn_download.setToolTip("按上面的地址下载权重到本地缓存（~/.cache/bci_lab/pretrained）")
        pre_row = QWidget()
        pl_pre = QHBoxLayout(pre_row)
        pl_pre.setContentsMargins(0, 0, 0, 0)
        pl_pre.addWidget(self.edit_pretrained, 1)
        pl_pre.addWidget(self.btn_pretrained)
        pl_pre.addWidget(self.btn_download)

        self.combo_freeze = QComboBox()
        self.combo_freeze.addItems(["none", "encoder", "all_but_head", "custom"])
        self.combo_freeze.setToolTip(
            "none         = 全部参数都训练（从零或全参数微调）\n"
            "encoder      = 冻结编码器/主干，只训练分类头（线性探针）\n"
            "all_but_head = 同 encoder（对齐 EEG-FM-Benchmark 的叫法）\n"
            "custom       = 只训练右侧「仅训练模块」里列出的模块")
        self.edit_trainable = QLineEdit("")
        self.edit_trainable.setPlaceholderText("可训练模块（逗号分隔），如 classifier, proj_out")
        self.edit_trainable.setToolTip(
            "按参数名分段匹配：填 classifier 会命中 classifier.weight；\n"
            "留空则由上面的冻结策略决定")
        freeze_row = QWidget()
        fl_free = QHBoxLayout(freeze_row)
        fl_free.setContentsMargins(0, 0, 0, 0)
        fl_free.addWidget(self.combo_freeze, 1)
        fl_free.addWidget(QLabel("仅训练模块"))
        fl_free.addWidget(self.edit_trainable, 2)

        self.spin_lr_encoder = QDoubleSpinBox()
        self.spin_lr_encoder.setDecimals(7)
        self.spin_lr_encoder.setRange(0.0, 1.0)
        self.spin_lr_encoder.setValue(1e-5)
        self.spin_lr_encoder.setToolTip(
            "编码器（预训练主干）的学习率。0 表示与「学习率」相同。\n"
            "典型：头 1e-3、编码器 1e-5（差 100 倍）")
        self.spin_layer_decay = QDoubleSpinBox()
        self.spin_layer_decay.setDecimals(2)
        self.spin_layer_decay.setRange(0.1, 1.0)
        self.spin_layer_decay.setSingleStep(0.05)
        self.spin_layer_decay.setValue(1.0)
        self.spin_layer_decay.setToolTip(
            "逐层衰减系数（<1 时生效）：编码器第 k 层 lr = lr_encoder × decay^(层数-k)，\n"
            "即底层小、高层大（LaBraM/MAE 常见的分层微调）")
        lr_row = QWidget()
        lr_lay = QHBoxLayout(lr_row)
        lr_lay.setContentsMargins(0, 0, 0, 0)
        lr_lay.addWidget(QLabel("学习率"))
        lr_lay.addWidget(self.spin_lr_encoder)
        lr_lay.addWidget(QLabel("逐层衰减"))
        lr_lay.addWidget(self.spin_layer_decay)

        f.addRow("预训练权重", pre_row)
        self._dyn_rows["pretrained"] = f.rowCount() - 1
        f.addRow("冻结策略", freeze_row)
        self._dyn_rows["freeze"] = f.rowCount() - 1
        f.addRow("预训练主干", lr_row)
        self._dyn_rows["enc_lr"] = f.rowCount() - 1

        # ---- 训练设备：自动 / CPU / MPS / CUDA 单卡 / 多卡并行 ----
        self.combo_device = QComboBox()
        self.combo_device.setToolTip(
            "训练用设备：\n"
            "  自动      → 挑本机最快（CUDA > MPS > CPU），换机器不用改配置\n"
            "  cpu       → Windows / Linux / macOS 通用\n"
            "  mps       → Apple Silicon（Metal 加速）\n"
            "  cuda:N    → 指定第 N 张 NVIDIA 卡\n"
            "  cuda:0,1  → 多卡并行（点右边按钮可勾选多张）")
        self.combo_device.currentIndexChanged.connect(self._update_device_label)
        self.btn_device = QPushButton("检测/选择…")
        self.btn_device.setToolTip("探测本机设备、勾选（多卡可多选）并一键测速")
        self.btn_device.clicked.connect(self.pick_device_dialog)
        self.lbl_device = QLabel()
        self.lbl_device.setStyleSheet("color:#888; font-size:11px;")
        self.lbl_device.setWordWrap(True)
        device_row = QWidget()
        dl = QHBoxLayout(device_row)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.addWidget(self.combo_device, 1)
        dl.addWidget(self.btn_device)
        f.addRow("训练设备", device_row)
        f.addRow("", self.lbl_device)
        self._populate_devices()

        f.addRow("epochs", self.spin_epochs)
        f.addRow("batch size", self.spin_batch)
        f.addRow("学习率", self.spin_lr)
        f.addRow("优化器", self.combo_opt)
        f.addRow("调度器", self.combo_sched)
        f.addRow("weight decay", self.spin_wd)
        f.addRow("早停 patience", self.spin_patience)
        f.addRow("随机种子", self.spin_seed)
        f.addRow("", self.chk_weighted)
        f.addRow("标准化", self.combo_norm)
        f.addRow("", self.chk_ea)

        # ---- 数据增强：三种方法折叠起来，点标题展开/收起 ----
        self.btn_aug = QToolButton()
        self.btn_aug.setCheckable(True)
        self.btn_aug.setChecked(False)
        self.btn_aug.setArrowType(Qt.ArrowType.RightArrow)
        self.btn_aug.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.btn_aug.setStyleSheet(
            "QToolButton{border:none;font-weight:bold;padding:3px 0;text-align:left;}")
        self.btn_aug.setToolTip(
            "时间平移 / 高斯噪声 / 通道 dropout —— 每个 epoch 现场随机施加\n"
            "（不落盘、不参与预处理，因此不会造成测试集信息泄漏）")
        f.addRow(self.btn_aug)                    # 跨两列，作为折叠标题
        for key, label, widget in (("aug_shift", "时间平移（采样点）", self.spin_shift),
                                   ("aug_noise", "高斯噪声（σ 比例）", self.spin_noise),
                                   ("aug_chdrop", "通道 dropout（比例）", self.spin_chdrop)):
            f.addRow(label, widget)
            self._dyn_rows[key] = f.rowCount() - 1

        f.addRow("实验名", self.edit_name)
        f.addRow("备注", self.edit_note)
        self.lbl_paper = QLabel("（未使用论文抽取）")
        self.lbl_paper.setWordWrap(True)
        self.lbl_paper.setMinimumWidth(120)
        self.lbl_paper.setStyleSheet("color:#666; font-size:11px;")
        self.btn_paper_clear = QPushButton("清除")
        self.btn_paper_clear.setEnabled(False)
        self.btn_paper_clear.setToolTip(
            "不再把「参数来自哪篇论文」写进实验记录（已应用的超参数数值保持不变）")
        row_pp = QHBoxLayout()
        row_pp.setContentsMargins(0, 0, 0, 0)
        row_pp.addWidget(self.lbl_paper, 1)
        row_pp.addWidget(self.btn_paper_clear)
        w_pp = QWidget()
        w_pp.setLayout(row_pp)
        f.addRow("参数来源", w_pp)
        row = QHBoxLayout()
        row.addWidget(self.btn_loadcfg)
        row.addWidget(self.btn_paper)
        row.addWidget(self.btn_curve)
        w = QWidget()
        w.setLayout(row)
        f.addRow(w)
        row2 = QHBoxLayout()
        row2.addWidget(self.btn_start)
        row2.addWidget(self.btn_stop)
        w2 = QWidget()
        w2.setLayout(row2)
        f.addRow(w2)

        # ================= 右：曲线 + 日志 + 实验表 =================
        self.curves = LiveCurves()
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.tbl_exps = QTableWidget(0, 6)
        self.tbl_exps.setHorizontalHeaderLabels(
            ["id", "时间", "模型", "划分", "acc (%)", "目录"])
        self.tbl_exps.setColumnWidth(5, 320)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(self.curves)          # 曲线：两个图固定高度，不随窗口拉伸
        rl.addWidget(QLabel("训练日志"))
        rl.addWidget(self.txt_log, 1)      # 日志占剩余空间

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(box)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root = QVBoxLayout(self)
        root.addWidget(splitter, 1)
        root.addWidget(QLabel("实验列表（双击回看 → 结果分析页）"))
        root.addWidget(self.tbl_exps)

        # ---- 事件 ----
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_curve.clicked.connect(self.few_shot_curve_dialog)
        self.btn_model_code.clicked.connect(self.show_model_code)
        self.btn_edit_folds.clicked.connect(self.edit_folds)
        self.btn_test_dir.clicked.connect(self._browse_test_dir)
        self.btn_scheme_intro.clicked.connect(self.show_scheme_intro)
        self.btn_pretrained.clicked.connect(self._browse_pretrained)
        self.btn_download.clicked.connect(self._download_weights)
        self.combo_model.currentIndexChanged.connect(self._on_model_changed)
        # 参数行「按需显示」：方案 / 模型 / 权重 / 冻结策略 任一变化都重算
        self.combo_scheme.currentTextChanged.connect(self._refresh_visibility)
        self.combo_model.currentIndexChanged.connect(self._refresh_visibility)
        self.combo_freeze.currentTextChanged.connect(self._refresh_visibility)
        self.edit_pretrained.textChanged.connect(self._refresh_visibility)
        self.spin_lr_encoder.valueChanged.connect(self._refresh_visibility)
        self.spin_layer_decay.valueChanged.connect(self._refresh_visibility)
        # 数据增强折叠
        self.btn_aug.toggled.connect(self._update_aug_summary)
        self.btn_aug.toggled.connect(self._refresh_visibility)
        for _spin in (self.spin_shift, self.spin_noise, self.spin_chdrop):
            _spin.valueChanged.connect(self._update_aug_summary)
        self.btn_loadcfg.clicked.connect(self.load_config_dialog)
        self.btn_paper.clicked.connect(self.extract_from_paper)
        self.btn_paper_clear.clicked.connect(lambda: self._set_paper_params(None))
        self.tbl_exps.itemDoubleClicked.connect(self._open_in_result_page)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._drain_events)
        self._timer.start(300)
        self._update_aug_summary()
        self._refresh_visibility()
        self.refresh_experiments()

    # ------------------------------------------------------------------
    def _current_model(self) -> str:
        """当前选中的模型名（基础模型 / 自定义模型用 userData 存真实名字）。"""
        return str(self.combo_model.currentData() or self.combo_model.currentText())

    def _rebuild_model_combo(self):
        """重建模型下拉框：内置 + EEG 基础模型 + 自定义模型（保存后即时刷新）。"""
        keep = self._current_model() if self.combo_model.count() else ""
        self.combo_model.blockSignals(True)
        self.combo_model.clear()
        self.combo_model.addItems(MODEL_NAMES)
        try:                       # EEG 基础模型（需 braindecode + 预训练权重）
            from core.models.foundation import FOUNDATION_MODELS

            self.combo_model.insertSeparator(self.combo_model.count())
            for key, spec in FOUNDATION_MODELS.items():
                self.combo_model.addItem(f"{key}（{spec.display}）", userData=key)
        except Exception:  # noqa: BLE001
            pass
        try:                       # 自定义模型（models_custom/*.py）
            from core.models.custom import list_custom

            customs = list_custom()
            if customs:
                self.combo_model.insertSeparator(self.combo_model.count())
                for name, meta in customs.items():
                    it = f"{name}（自定义"
                    base = meta.get("base") or ""
                    it += f" · 基于 {base}）" if base else "）"
                    self.combo_model.addItem(it, userData=name)
        except Exception:  # noqa: BLE001
            pass
        self.combo_model.blockSignals(False)
        if keep:
            self._select_model(keep)

    def _select_model(self, name: str) -> bool:
        """按下拉框里的真实模型名选中（自定义模型的显示文案带后缀）。"""
        for i in range(self.combo_model.count()):
            if (self.combo_model.itemData(i) or self.combo_model.itemText(i)) == name:
                self.combo_model.setCurrentIndex(i)
                return True
        return False

    # ------------------------------------------------------------ 训练设备
    def _populate_devices(self, keep: str | None = None):
        """按本机探测结果重建设备下拉框。"""
        from core.train.devices import list_devices

        if keep is None:
            keep = self.combo_device.currentData() if self.combo_device.count() else "auto"
        self.combo_device.blockSignals(True)
        self.combo_device.clear()
        self.combo_device.addItem("自动（推荐：CUDA > MPS > CPU）", userData="auto")
        devices = list_devices()
        if devices:
            self.combo_device.insertSeparator(self.combo_device.count())
        for d in devices:
            mem = f"（{d.memory_gb:g} GB）" if d.memory_gb else ""
            self.combo_device.addItem(f"{d.id} — {d.name}{mem}", userData=d.id)
        self.combo_device.blockSignals(False)
        self._select_device(keep or "auto")
        self._update_device_label()

    def _select_device(self, spec: str) -> bool:
        """选中某个设备写法；下拉框里没有（如配置里的 cuda:0,1）就临时补一项。"""
        spec = str(spec or "auto")
        for i in range(self.combo_device.count()):
            if self.combo_device.itemData(i) == spec:
                self.combo_device.setCurrentIndex(i)
                return True
        if spec == "auto":
            self.combo_device.setCurrentIndex(0)
            return True
        self.combo_device.blockSignals(True)
        self.combo_device.addItem(f"{spec}（来自配置）", userData=spec)
        self.combo_device.setCurrentIndex(self.combo_device.count() - 1)
        self.combo_device.blockSignals(False)
        return False

    def _update_device_label(self, *_):
        """在下拉框下面显示"实际会用哪块设备"。"""
        from core.train.devices import device_label

        spec = self.combo_device.currentData() or "auto"
        try:
            self.lbl_device.setText(f"实际使用：{device_label(spec)}")
        except Exception:  # noqa: BLE001
            self.lbl_device.setText("")

    def pick_device_dialog(self):
        """打开设备选择对话框（探测 + 多卡勾选 + 测速）。"""
        from core.train.devices import device_label
        from .widgets.device_dialog import DeviceDialog

        current = self.combo_device.currentData() or "auto"
        dlg = DeviceDialog(current=current, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        spec = dlg.selected_spec()
        self._populate_devices(keep=spec)
        self._log(f"训练设备已设为 {spec} → {device_label(spec)}")

    def _guess_shape(self) -> dict:
        """当前预处理数据的 (类别数, 通道数, 采样点)——用于自定义模型检查。"""
        import numpy as np

        dims = {"n_classes": 4, "n_ch": 22, "n_times": 501}
        try:
            root = self.edit_processed.text().strip()
            if root:
                from pathlib import Path

                d = Path(root)
                x = np.load(d / "X.npy", mmap_mode="r")
                dims["n_ch"], dims["n_times"] = int(x.shape[1]), int(x.shape[2])
                import json

                classes = json.loads((d / "classes.json").read_text(encoding="utf-8"))
                dims["n_classes"] = max(len(classes), 2)
        except Exception:  # noqa: BLE001 - 数据还没生成时用默认维度
            pass
        return dims

    def open_model_editor(self, edit_name: str | None = None, blank: bool = False):
        """打开自定义模型编辑器。

        blank=False（默认）→ 以当前选中的内置模型源码为模板改写；
        blank=True        → **从零设计**：先输英文模型名，自动生成 nn.Module 骨架。
        """
        from .widgets.model_editor_dialog import ModelEditorDialog

        name = self._current_model()
        try:
            from core.models import TORCH_MODELS

            base = name if name in TORCH_MODELS else "eegnet"
        except Exception:  # noqa: BLE001
            base = "eegnet"
        dims = self._guess_shape()
        self._log(f"打开自定义模型编辑器（{'从零设计' if blank else f'模板 = {base}'}）："
                  f"检查维度 = {dims['n_classes']} 类 / {dims['n_ch']} 通道 / "
                  f"{dims['n_times']} 采样点")

        def _saved(new_name: str):
            self._rebuild_model_combo()
            self._select_model(new_name)
            self._log(f"自定义模型「{new_name}」已保存并选中，可直接开始训练")

        dlg = ModelEditorDialog(base=base, parent=self, on_saved=_saved,
                                edit_name=edit_name, blank=blank, **dims)
        dlg.exec()

    def _on_model_changed(self):
        """切到基础模型时：自动填 auto 并提示权重仓库/来源。"""
        name = self._current_model()
        try:
            from core.models.foundation import FOUNDATION_MODELS

            spec = FOUNDATION_MODELS.get(name)
        except Exception:  # noqa: BLE001
            spec = None
        if spec is None:
            return
        if not self.edit_pretrained.text().strip() and spec.hf_repo:
            self.edit_pretrained.setText("auto")
        self._log(f"[基础模型] {spec.display}：{spec.note}")
        if spec.hf_repo:
            self._log(f"  权重仓库 {spec.hf_repo} → 点「下载/校验」自动获取")
        else:
            self._log(f"  需按官方来源准备权重：{spec.source or spec.arch_repo}")

    def _download_weights(self):
        """按当前地址下载/校验权重，成功后把本地路径回填到输入框。"""
        from core.models.foundation import get_spec, is_foundation, resolve_weights

        ref = self.edit_pretrained.text().strip() or "auto"
        name = self._current_model()
        spec = get_spec(name) if is_foundation(name) else None
        try:
            path = resolve_weights(ref, spec=spec, log=self._log)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "下载失败", f"{type(e).__name__}: {e}")
            return
        if path is None:
            QMessageBox.information(self, "提示", "未指定权重（留空 = 从零训练）")
            return
        if path.is_dir():
            self._log(f"  权重目录：{path}")
            QMessageBox.information(self, "完成", f"权重已就绪（目录）：\n{path}")
        else:
            self.edit_pretrained.setText(str(path))
            QMessageBox.information(
                self, "完成",
                f"权重已就绪：\n{path}\n（{path.stat().st_size / 1e6:.1f} MB）")

    def _browse_pretrained(self):
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, "选择预训练权重", str(PROJECT_ROOT / "experiments"),
            "权重文件 (*.pt *.pth);;所有文件 (*)")
        if path:
            self.edit_pretrained.setText(path)

    def _update_aug_summary(self, *_):
        """折叠标题上显示增强是否启用（收起时也能一眼看到）。"""
        parts = []
        if self.spin_shift.value():
            parts.append(f"平移 {self.spin_shift.value()}")
        if self.spin_noise.value() > 0:
            parts.append(f"噪声 {self.spin_noise.value():.2f}")
        if self.spin_chdrop.value() > 0:
            parts.append(f"dropout {self.spin_chdrop.value():.2f}")
        state = "、".join(parts) if parts else "未启用"
        act = "收起" if self.btn_aug.isChecked() else "展开"
        self.btn_aug.setText(f"数据增强：{state}（点击{act}）")
        self.btn_aug.setArrowType(Qt.ArrowType.DownArrow if self.btn_aug.isChecked()
                                  else Qt.ArrowType.RightArrow)

    def _refresh_visibility(self, *_) -> dict:
        """只显示**与当前设置相关**的参数行，返回可见性字典（便于调试/测试）。

        规则：
        - 划分方案：``_ROW_SCHEMES``（如 loso 不需要 k）
        - 预训练权重 / 冻结策略 / 主干学习率：
            经典基线（CSP+LDA 等）不支持加载权重 → 三行全隐藏；
            非基础模型且未填权重 → 只保留「预训练权重」一行；
            基础模型或已填权重 → 显示「冻结策略」；
            主干可训练（冻结策略为 none/custom）→ 才显示「主干学习率」
        - 数据增强三行：由折叠标题控制展开/收起
        """
        scheme = self.combo_scheme.currentText()
        model = self._current_model()
        classic = bool(model) and is_classic(model)
        custom = bool(model) and is_custom(model)

        has_pre = bool(self.edit_pretrained.text().strip())
        fm = bool(model) and is_foundation(model)
        freeze = self.combo_freeze.currentText()
        # 自定义模型没有登记权重仓库，也不涉及主干微调策略 → 这几行全隐藏
        weights_ctx = (not classic) and (not custom) and (has_pre or fm)
        # 用户手动调过主干学习率/衰减 → 保持可见，避免"被自动隐藏"的困惑
        tuned = (abs(self.spin_lr_encoder.value() - _LR_ENCODER_DEFAULT) > 1e-12
                 or self.spin_layer_decay.value() < 1.0)

        vis: dict[str, bool] = {
            "pretrained": (not classic) and (not custom),
            "freeze": weights_ctx,
            "enc_lr": weights_ctx and (freeze in ("none", "custom") or tuned),
            "aug_shift": self.btn_aug.isChecked(),
            "aug_noise": self.btn_aug.isChecked(),
            "aug_chdrop": self.btn_aug.isChecked(),
        }
        for key, schemes in _ROW_SCHEMES.items():
            vis[key] = scheme in schemes

        for key, row in self._dyn_rows.items():
            self._set_row_visible(row, vis.get(key, True))
        return vis

    def _set_row_visible(self, row: int, visible: bool):
        """兼容不同 Qt 版本的行隐藏。"""
        try:
            self._form.setRowVisible(row, visible)   # Qt >= 6.4
        except AttributeError:                       # 旧版 Qt：退化为隐藏控件
            for role in (QFormLayout.ItemRole.LabelRole,
                         QFormLayout.ItemRole.FieldRole):
                item = self._form.itemAt(row, role)
                if item is not None and item.widget() is not None:
                    item.widget().setVisible(visible)

    def show_scheme_intro(self):
        """弹出「划分方案介绍」：看每种协议在真实数据上的切分示意图。"""
        from .widgets.split_intro_dialog import SplitIntroDialog

        SplitIntroDialog(self.edit_processed.text().strip(), self).exec()

    def _browse_test_dir(self):
        from PyQt6.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(
            self, "选择目标域（测试数据集）预处理目录",
            str(PROJECT_ROOT / "processed"))
        if path:
            self.edit_test_dir.setText(path)

    def edit_folds(self):
        """打开自定义折编辑器（每折指定 train / val / test 受试者）。"""
        from PyQt6.QtWidgets import QDialog

        from .widgets.folds_dialog import FoldsDialog, subjects_from_processed

        processed = self.edit_processed.text().strip()
        subs = subjects_from_processed(processed)
        if not subs:
            QMessageBox.information(
                self, "提示",
                f"未在「{processed}」找到 sid.npy。\n"
                f"请先在「预处理工作台」生成预处理数据，再配置自定义折。")
            return
        dlg = FoldsDialog(subs, self._folds, processed, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._folds = dlg.get_folds()
            brief = "；".join(f"{f['name']}: {len(f['train'])}tr/{len(f['test'])}te"
                              for f in self._folds)
            self.lbl_folds.setText(f"已配置 {len(self._folds)} 折（{brief}）")
            self.lbl_folds.setStyleSheet("color:#1a7f37;")
            self.combo_scheme.setCurrentText("custom")

    @staticmethod
    def _parse_grid(text: str) -> dict:
        """解析 nested_cv 的参数网格：`n_components: 4,6,8; dropout: 0.25,0.5`。"""
        text = (text or "").strip()
        if not text:
            return {}
        grid: dict[str, list] = {}
        for part in text.replace("；", ";").split(";"):
            if ":" not in part:
                continue
            key, _, raw = part.partition(":")
            vals: list = []
            for item in raw.replace("，", ",").split(","):
                item = item.strip()
                if not item:
                    continue
                try:
                    vals.append(int(item))
                except ValueError:
                    try:
                        vals.append(float(item))
                    except ValueError:
                        vals.append(item)
            if key.strip() and vals:
                grid[key.strip()] = vals
        return grid

    def show_model_code(self):
        """跳转查看当前所选模型的 PyTorch (nn.Module) 定义代码。"""
        from .widgets.code_viewer_dialog import (CodeViewerDialog, model_code,
                                                 model_summary)

        name = self._current_model()
        title, code, path = model_code(name)
        dlg = CodeViewerDialog(title, code, path, model_summary(name), self)
        dlg.exec()

    def few_shot_curve_dialog(self):
        """Few-Shot 学习曲线：扫描多个 n_shot，自动训练并画准确率曲线。"""
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLineEdit

        dlg = QDialog(self)
        dlg.setWindowTitle("Few-Shot 学习曲线")
        f = QFormLayout(dlg)
        edit_shots = QLineEdit("1,5,10,20")
        spin_rep = QSpinBox()
        spin_rep.setRange(1, 100)
        spin_rep.setValue(self.spin_repeats.value())
        f.addRow("n_shot 列表（逗号分隔）", edit_shots)
        f.addRow("每次的重复抽样次数", spin_rep)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        f.addRow(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        raw = edit_shots.text().replace("，", ",")
        shots = [int(x) for x in raw.split(",") if x.strip().isdigit()]
        if not shots:
            QMessageBox.information(self, "提示", "请填写至少一个 n_shot（如 1,5,10,20）")
            return

        cfg = self.build_config()
        cfg["split"]["n_repeats"] = int(spin_rep.value())
        cfg["experiment"]["name"] = (cfg["experiment"]["name"] or "fewshot") + "_curve"
        self._log(f"开始 Few-Shot 学习曲线：n_shot={shots}，重复 {spin_rep.value()} 次")

        def job(log=print):
            return run_few_shot_curve(cfg, shots=shots, log=log)

        th = FunctionThread(job, parent=self)
        self._curve_thread = th
        th.logged.connect(self._log)
        th.ok.connect(self._curve_done)
        th.fail.connect(lambda m: QMessageBox.critical(self, "学习曲线失败", m))
        th.start()

    def _curve_done(self, curve):
        if not curve.get("results"):
            msgs = "；".join(f"n_shot={f['n_shot']}: {f['error']}"
                            for f in curve.get("failed", []))
            QMessageBox.information(self, "提示",
                                    "没有产生任何结果。\n" + (msgs or "请检查数据与配置"))
            return
        self._log("学习曲线完成：" + "，".join(
            f"n_shot={r['n_shot']} → {r['acc_mean'] * 100:.2f}%" for r in curve["results"]))
        for f_ in curve.get("failed", []):
            self._log(f"! n_shot={f_['n_shot']} 未成功：{f_['error']}")
        self.refresh_experiments()

        from PyQt6.QtWidgets import QDialog, QFileDialog, QVBoxLayout

        from .widgets.plotting import new_canvas

        fig, canvas = new_canvas(7.0, 5.2)
        ax = fig.add_subplot(111)
        plot_few_shot_curve(curve, ax)
        canvas.draw()
        d = QDialog(self)
        d.setWindowTitle("Few-Shot 学习曲线（每类需要多少校准样本）")
        lay = QVBoxLayout(d)
        lay.addWidget(canvas)
        btn = QPushButton("导出图片")

        def _save():
            path, _ = QFileDialog.getSaveFileName(d, "导出学习曲线", "fewshot_curve.png",
                                                  "PNG (*.png);;SVG (*.svg)")
            if path:
                fig.savefig(path, dpi=160)
                QMessageBox.information(d, "完成", f"已导出 {path}")

        btn.clicked.connect(_save)
        lay.addWidget(btn)
        d.resize(760, 620)
        d.exec()

    # ------------------------------------------------------------------
    def build_config(self) -> dict:
        scheme = self.combo_scheme.currentText()
        cfg = {
            "experiment": {"name": self.edit_name.text().strip() or "run",
                           "note": self.edit_note.text().strip()},
            "data": {"processed_dir": self.edit_processed.text().strip(),
                     "test_processed_dir": self.edit_test_dir.text().strip() or None},
            "model": {"name": self._current_model(), "params": {},
                      "pretrained": self.edit_pretrained.text().strip() or None,
                      "freeze": self.combo_freeze.currentText(),
                      "trainable": [s.strip() for s in
                                    self.edit_trainable.text().replace("，", ",").split(",")
                                    if s.strip()]},
            "split": {"scheme": scheme, "k": self.spin_k.value(),
                      # few_shot 训练样本本就极少，不再切验证集
                      "val_frac": 0.0 if scheme == "few_shot" else 0.15,
                      "seed": self.spin_seed.value(),
                      "test_session": self.combo_test_session.currentText(),
                      "n_shot": self.spin_shot.value(),
                      "n_repeats": self.spin_repeats.value(),
                      # 每类比例：仅 within / few_shot 生效；0 表示不使用
                      "train_percentage": (self.spin_train_pct.value()
                                           if (self.spin_train_pct.value() > 0
                                               and scheme in ("within", "few_shot"))
                                           else None),
                      "folds": self._folds,
                      "param_grid": self._parse_grid(self.edit_grid.text())},
            "normalization": {"mode": self.combo_norm.currentText(),
                              "euclidean_alignment": self.chk_ea.isChecked()},
            "augmentation": {"time_shift": self.spin_shift.value(),
                             "gaussian_noise": self.spin_noise.value(),
                             "channel_dropout": self.spin_chdrop.value()},
            "trainer": {"epochs": self.spin_epochs.value(),
                        "batch_size": self.spin_batch.value(),
                        "lr": self.spin_lr.value(),
                        # 编码器分层学习率：0 表示与 lr 相同；layer_decay<1 启用逐层衰减
                        "lr_encoder": (self.spin_lr_encoder.value() or None),
                        "layer_decay": self.spin_layer_decay.value(),
                        "optimizer": self.combo_opt.currentText(),
                        "weight_decay": self.spin_wd.value(),
                        "scheduler": self.combo_sched.currentText(),
                        "patience": self.spin_patience.value(),
                        "seed": self.spin_seed.value(),
                        "amp": "auto", "deterministic": False,
                        "device": self.combo_device.currentData() or "auto",
                        "num_workers": 0},
            "class_balance": {"weighted_loss": self.chk_weighted.isChecked()},
        }
        if self._paper_params:                      # 论文抽取溯源 → 写进实验记录
            cfg["paper_params"] = self._paper_params
        return cfg

    def _set_paper_params(self, prov: dict | None):
        """记下/清除"本次配置来自哪篇论文哪一节"（会随 config.yaml 一起落盘）。"""
        self._paper_params = prov or None
        if prov:
            self.lbl_paper.setText(
                f"{summarize_provenance(prov)}（将写入实验记录 config.yaml）")
            self.btn_paper_clear.setEnabled(True)
        else:
            self.lbl_paper.setText("（未使用论文抽取）")
            self.btn_paper_clear.setEnabled(False)

    def load_config_dialog(self):
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(self, "读取训练配置",
                                              str(PROJECT_ROOT / "configs"), "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            cfg = load_config(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "读取失败", str(e))
            return
        self._apply_config(cfg)
        self._log(f"配置已加载 ← {path}")

    def extract_from_paper(self):
        """从论文方法段抽取训练超参（可选 LLM；没配密钥用内置规则），并写回界面。"""
        from .widgets.paper_extract_dialog import PaperExtractDialog

        dlg = PaperExtractDialog(parent=self, on_apply=self._apply_from_paper,
                                 base_config=self.build_config)
        dlg.exec()

    def _apply_from_paper(self, cfg: dict, prov: dict | None = None):
        """把抽取到的配置合并进当前界面（未识别到的字段保持原样）。

        ``prov`` 是抽取溯源（来源文件/小节/依据），随 ``config.yaml`` 写入实验记录。
        """
        base = self.build_config()

        def _deep_merge(dst: dict, src: dict) -> dict:
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    _deep_merge(dst[k], v)
                else:
                    dst[k] = v
            return dst

        merged = _deep_merge(base, cfg or {})
        self._apply_config(merged)
        self._set_paper_params(prov)
        changed = [k for k in ("trainer", "normalization", "augmentation", "split")
                   if (cfg or {}).get(k)]
        self._log(f"已从论文文本应用训练参数：{', '.join(changed) or '（无变化）'}"
                  + (f"；来源已绑定实验记录：{summarize_provenance(prov)}" if prov else ""))

    def _apply_config(self, cfg):
        exp = cfg.get("experiment", {})
        self.edit_name.setText(exp.get("name", "run"))
        self.edit_note.setText(exp.get("note", ""))
        self.edit_processed.setText(cfg.get("data", {}).get("processed_dir",
                                    str(PROJECT_ROOT / "processed" / "bnci2014001")))
        model = cfg.get("model", {}).get("name", "eegnet")
        if not self._select_model(model):
            self._log(f"[提示] 配置里的模型 {model} 不在下拉框中（可能已被删除）")
        sp = cfg.get("split", {})
        self.combo_scheme.setCurrentText(sp.get("scheme", "session"))
        self.spin_k.setValue(int(sp.get("k", 5)))
        self.spin_shot.setValue(int(sp.get("n_shot", 5)))
        self.spin_repeats.setValue(int(sp.get("n_repeats", 10)))
        self.combo_test_session.setCurrentText(str(sp.get("test_session", "all")))
        self.spin_seed.setValue(int(sp.get("seed", 42)))
        tr = cfg.get("trainer", {})
        self._populate_devices(keep=str(tr.get("device", "auto") or "auto"))
        self.spin_epochs.setValue(int(tr.get("epochs", 60)))
        self.spin_batch.setValue(int(tr.get("batch_size", 64)))
        self.spin_lr.setValue(float(tr.get("lr", 1e-3)))
        self.combo_opt.setCurrentText(tr.get("optimizer", "adamw"))
        self.combo_sched.setCurrentText(tr.get("scheduler", "cosine"))
        self.spin_wd.setValue(float(tr.get("weight_decay", 0.01)))
        self.spin_patience.setValue(int(tr.get("patience", 12)))
        no = cfg.get("normalization", {})
        self.combo_norm.setCurrentText(no.get("mode", "zscore_channel"))
        self.chk_ea.setChecked(bool(no.get("euclidean_alignment", False)))
        au = cfg.get("augmentation", {})
        self.spin_shift.setValue(int(au.get("time_shift", 0) or 0))
        self.spin_noise.setValue(float(au.get("gaussian_noise", 0) or 0))
        self.spin_chdrop.setValue(float(au.get("channel_dropout", 0) or 0))
        self.chk_weighted.setChecked(bool(cfg.get("class_balance", {}).get("weighted_loss", True)))
        self._set_paper_params(cfg.get("paper_params"))     # 配置里的论文来源也一并还原

    # ------------------------------------------------------------------
    def start(self):
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "提示", "已有训练在进行中")
            return
        config = self.build_config()
        if not config["data"]["processed_dir"]:
            QMessageBox.warning(self, "提示", "请填写预处理数据目录（先在预处理页生成）")
            return
        self.curves.clear_all()
        while True:
            try:
                self._events.get_nowait()
            except queue.Empty:
                break

        th = FunctionThread(run_training, config, parent=self)
        self._thread = th
        th.logged.connect(self._log)
        # on_epoch 回调在工作线程执行 → 队列 → QTimer 消费（线程安全）
        config_with_cb = dict(config)

        def on_epoch(tag, stats):
            self._events.put(("epoch", tag, stats))

        th._kwargs["on_epoch"] = on_epoch
        th.ok.connect(self._train_done)
        th.fail.connect(lambda m: (self._log(f"[错误] {m}"),
                                   QMessageBox.critical(self, "训练失败", m)))
        self.btn_start.setEnabled(False)
        th.start()

    def stop(self):
        if self._thread is not None and self._thread.isRunning():
            self._thread.stop()
            self._log("已请求停止（epoch 边界生效）…")

    def _train_done(self, summary):
        self.btn_start.setEnabled(True)
        agg = summary.get("aggregate", {})
        self._log(f"✔ 完成：acc={agg.get('acc_mean', 0) * 100:.2f}±"
                  f"{agg.get('acc_std', 0) * 100:.2f}%  kappa={agg.get('kappa_mean', 0):.3f}")
        self.refresh_experiments()

    def _drain_events(self):
        """把训练线程推送的 epoch 统计画到实时曲线上。

        曲线只是可视化：这里整体加一层异常保护，任何绘图问题都不该打断训练
        （历史 bug：曲线组件调用不存在的 PlotDataItem.getLists() 导致抛错）。
        """
        drained = 0
        while drained < 200:
            try:
                kind, tag, stats = self._events.get_nowait()
            except queue.Empty:
                break
            drained += 1
            try:
                epoch = stats.get("epoch")
                for key in ("train_loss", "train_acc", "val_loss", "val_acc"):
                    value = stats.get(key)
                    if value is not None:
                        # 按折分开存储：多折训练时曲线不会互相混在一起
                        self.curves.add_point(key, epoch, value, fold=str(tag))
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    def refresh_experiments(self):
        try:
            rows = ExperimentManager().list_experiments()
        except Exception:  # noqa: BLE001
            rows = []
        self.tbl_exps.setRowCount(len(rows))
        for i, r in enumerate(rows):
            acc = r.get("acc_mean")
            vals = [str(r.get("id", "")), str(r.get("ts", ""))[:16], str(r.get("model", "")),
                    str(r.get("scheme", "")),
                    f"{(acc or 0) * 100:.2f}", str(r.get("dir", ""))]
            for j, v in enumerate(vals):
                from PyQt6.QtWidgets import QTableWidgetItem

                self.tbl_exps.setItem(i, j, QTableWidgetItem(v))

    def _open_in_result_page(self, item):
        row = item.row()
        exp_dir = self.tbl_exps.item(row, 5).text()
        result_page = getattr(self.state, "result_page", None)
        if result_page is not None:
            result_page.load_experiment_dir(exp_dir)
            if getattr(self.state, "switch_page", None):
                self.state.switch_page("结果分析")

    def _log(self, text):
        self.txt_log.appendPlainText(str(text))
