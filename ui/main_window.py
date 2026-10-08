"""主窗口：**可折叠左侧步骤栏** + 五个功能页（统一「浅紫粉」默认主题）。

主题在「视图 → 切换主题」里选择，选定后**全局生效**（界面、图表、弹窗一致）：
默认「浅紫粉」，另有「夜兰樱雾 / 海雾松岚 / 赤陶海铜」三套配色卡方案。
"""
from __future__ import annotations

from PyQt6.QtCore import (QEasingCurve, QParallelAnimationGroup, QPropertyAnimation,
                          Qt)
from PyQt6.QtGui import QActionGroup, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel, QListWidget,
                             QListWidgetItem, QMainWindow, QPushButton,
                             QStackedWidget, QStatusBar, QToolButton, QVBoxLayout,
                             QWidget)

from .theme import DEFAULT_THEME, THEMES, apply_theme

#: 主行动按钮的关键词：命中即着主题强调色
_ACCENT_KEYWORDS = ("▶", "应用到", "应用归一化", "开始", "运行", "加载", "绘制",
                    "计算")

#: 每页一句话说明（显示在顶栏副标题）
PAGE_DESC = {
    "数据管理": "加载原始数据，查看波形 / 事件 / 通道",
    "预处理工作台": "批量清洗与分段，落盘 processed 数据",
    "训练控制台": "配置模型与划分方案，训练并实时看曲线",
    "结果分析": "回看实验指标、混淆矩阵与按被试表现",
    "信号分析": "频域 / ERP / 时频 / 熵 / 脑网络分析",
}


class AppState:
    """跨页面共享状态（不写算法，只传引用）。"""

    def __init__(self):
        self.raw = None
        self.raw_name = ""
        self.meta = None
        self.result_page = None
        self.switch_page = None

    def set_raw(self, raw, name, meta):
        self.raw, self.raw_name, self.meta = raw, name, meta


class MainWindow(QMainWindow):
    PAGES = ["数据管理", "预处理工作台", "训练控制台", "结果分析", "信号分析"]
    NAV_W = 210      # 展开宽度
    RAIL_W = 46      # 收起后的窄条宽度

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI Lab — EEG 预处理与深度学习训练平台")
        self.resize(1460, 900)
        self.state = AppState()
        self._collapsed = False
        self._nav_anim: QParallelAnimationGroup | None = None

        # ---------------- 功能页 ----------------
        from .analysis_page import AnalysisPage
        from .data_page import DataPage
        from .preprocess_page import PreprocessPage
        from .result_page import ResultPage
        from .train_page import TrainPage

        self._pages: list[QWidget] = [DataPage(self.state), PreprocessPage(self.state),
                                      TrainPage(self.state), ResultPage(self.state),
                                      AnalysisPage(self.state)]
        (self.page_data, self.page_pre, self.page_train,
         self.page_result, self.page_analysis) = self._pages
        self.stack = QStackedWidget()
        for p in self._pages:
            self.stack.addWidget(p)
        self.state.result_page = self.page_result

        # ---------------- 布局 ----------------
        self._build_nav()
        self._build_topbar()

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self.topbar)
        rl.addWidget(self.stack, 1)

        center = QWidget()
        lay = QHBoxLayout(center)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.nav_box)
        lay.addWidget(right, 1)
        self.setCentralWidget(center)
        self.setStatusBar(QStatusBar())

        # ---------------- 主题 / 菜单 / 事件 ----------------
        self._tag_accent_buttons()   # 先打标记，再套样式表（属性选择器才生效）
        self._build_menus()          # 内部应用默认主题
        self.state.switch_page = self._switch_page
        self.nav.currentRowChanged.connect(self._on_page_changed)
        self._on_page_changed(0)

    # ================================================================ 步骤栏
    def _build_nav(self):
        self.nav_box = QWidget()
        self.nav_box.setObjectName("NavBox")
        self.nav_box.setFixedWidth(self.NAV_W)

        head = QWidget()
        head.setObjectName("NavHeader")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(12, 10, 8, 4)
        hl.setSpacing(4)
        self.lbl_nav_title = QLabel("实验流程")
        self.lbl_nav_title.setObjectName("NavTitle")
        self.btn_nav = QToolButton()
        self.btn_nav.setObjectName("NavToggle")
        self.btn_nav.setText("«")
        self.btn_nav.setFixedSize(26, 26)
        self.btn_nav.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_nav.setToolTip("收起 / 展开左侧步骤栏（Ctrl+B）")
        self.btn_nav.clicked.connect(self.toggle_nav)
        hl.addWidget(self.lbl_nav_title, 1)
        hl.addWidget(self.btn_nav)

        self.nav = QListWidget()
        self.nav.setObjectName("NavList")
        self.nav.setFrameShape(QFrame.Shape.NoFrame)
        self.nav.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for i, name in enumerate(self.PAGES):
            it = QListWidgetItem(f"{i + 1}. {name}")
            it.setToolTip(f"{name}：{PAGE_DESC.get(name, '')}")
            self.nav.addItem(it)

        # 收起后的窄条：纯序号（中性色，不额外引入色彩）
        self.rail = QWidget()
        self.rail.setObjectName("NavRail")
        rl = QVBoxLayout(self.rail)
        rl.setContentsMargins(5, 8, 5, 8)
        rl.setSpacing(7)
        self._nums: list[QPushButton] = []
        for i, name in enumerate(self.PAGES):
            b = QPushButton(str(i + 1))
            b.setObjectName("RailNum")
            b.setFixedSize(30, 30)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setToolTip(f"{name}：{PAGE_DESC.get(name, '')}")
            b.clicked.connect(lambda _=False, idx=i: self._goto(idx))
            rl.addWidget(b, 0, Qt.AlignmentFlag.AlignHCenter)
            self._nums.append(b)
        rl.addStretch(1)
        self.rail.setVisible(False)

        lay = QVBoxLayout(self.nav_box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(head)
        lay.addWidget(self.nav, 1)
        lay.addWidget(self.rail)

    def _build_topbar(self):
        self.topbar = QWidget()
        self.topbar.setObjectName("TopBar")
        self.topbar.setFixedHeight(56)
        lay = QHBoxLayout(self.topbar)
        lay.setContentsMargins(18, 6, 16, 6)
        lay.setSpacing(12)

        self.lbl_crumb = QLabel()
        self.lbl_crumb.setObjectName("Crumb")
        self.lbl_crumb_sub = QLabel()
        self.lbl_crumb_sub.setObjectName("CrumbSub")
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        col.addWidget(self.lbl_crumb)
        col.addWidget(self.lbl_crumb_sub)
        lay.addLayout(col, 1)

    # ================================================================ 折叠
    def toggle_nav(self):
        """收起 / 展开左侧步骤栏。"""
        self._collapsed = not self._collapsed
        self.nav.setVisible(not self._collapsed)
        self.lbl_nav_title.setVisible(not self._collapsed)
        self.rail.setVisible(self._collapsed)
        self.btn_nav.setText("»" if self._collapsed else "«")
        self.btn_nav.setToolTip("展开左侧步骤栏（Ctrl+B）" if self._collapsed
                                else "收起左侧步骤栏（Ctrl+B）")
        self._animate_nav(self.RAIL_W if self._collapsed else self.NAV_W)
        self.statusBar().showMessage(
            "左侧步骤栏已收起（点序号或 Ctrl+B 展开）" if self._collapsed
            else "左侧步骤栏已展开", 4000)

    def _animate_nav(self, target: int):
        grp = QParallelAnimationGroup(self)
        for prop in (b"minimumWidth", b"maximumWidth"):
            anim = QPropertyAnimation(self.nav_box, prop)
            anim.setDuration(150)
            anim.setStartValue(self.nav_box.width())
            anim.setEndValue(target)
            anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
            grp.addAnimation(anim)
        grp.start(QParallelAnimationGroup.DeletionPolicy.DeleteWhenStopped)
        self._nav_anim = grp          # 持有引用，避免被 GC

    def _goto(self, index: int):
        if 0 <= index < len(self._pages):
            self.nav.setCurrentRow(index)
            if self._collapsed:       # 收起态点序号 → 展开看内容
                self.toggle_nav()

    # ================================================================ 页面
    def _on_page_changed(self, index: int):
        if index < 0:
            return
        self.stack.setCurrentIndex(index)
        name = self.PAGES[index]
        self.lbl_crumb.setText(f"{index + 1}. {name}")
        self.lbl_crumb_sub.setText(PAGE_DESC.get(name, ""))
        self._mark_rail(index)
        self.statusBar().showMessage(
            f"{name}　主题：{self._theme.name}　"
            "流程：数据管理 → 预处理工作台 → 训练控制台 → 结果分析 → 信号分析")
        if self.nav.currentRow() != index:
            self.nav.setCurrentRow(index)

    def _mark_rail(self, active: int):
        for i, b in enumerate(self._nums):
            b.setProperty("active", i == active)
            b.style().unpolish(b)
            b.style().polish(b)

    def _tag_accent_buttons(self):
        """把各页的「主行动按钮」标记为强调色（QSS 用 [accent="true"] 命中）。"""
        for page in self._pages:
            for b in page.findChildren(QPushButton):
                if any(k in b.text() for k in _ACCENT_KEYWORDS):
                    b.setProperty("accent", True)

    # ================================================================ 主题
    def _set_theme(self, theme, *, announce: bool = True):
        app = QApplication.instance()
        t = apply_theme(app if app is not None else self, theme)
        self._theme = t
        for act in self._theme_actions:
            act.setChecked(act.data() == t.name)
        for p in self._pages:                     # 图表按新主题重绘
            hook = getattr(p, "on_theme_changed", None)
            if callable(hook):
                try:
                    hook()
                except Exception:  # noqa: BLE001 - 重绘失败不影响使用
                    pass
        if announce:
            self.statusBar().showMessage(f"已切换主题：{t.name}（{t.source}）", 5000)

    # ================================================================ 菜单
    def _build_menus(self):
        m_view = self.menuBar().addMenu("视图(&V)")
        act_nav = m_view.addAction("收起 / 展开步骤栏")
        act_nav.setShortcut(QKeySequence("Ctrl+B"))
        act_nav.triggered.connect(self.toggle_nav)

        m_theme = m_view.addMenu("切换主题")
        self._theme_actions: list = []
        group = QActionGroup(self)
        group.setExclusive(True)
        for t in THEMES:
            act = m_theme.addAction(f"{t.name}（{t.source}）")
            act.setCheckable(True)
            act.setData(t.name)
            act.setChecked(t.name == DEFAULT_THEME.name)
            act.triggered.connect(lambda _=False, th=t: self._set_theme(th))
            group.addAction(act)
            self._theme_actions.append(act)

        self._theme = apply_theme(QApplication.instance(), DEFAULT_THEME)

        m_help = self.menuBar().addMenu("帮助(&H)")
        act_demo = m_help.addAction("新手引导演示…")
        act_demo.setToolTip("一键跑通 数据 → 预处理 → 训练 → 结果，并弹出链路图")
        act_demo.triggered.connect(self._show_demo)
        act_scheme = m_help.addAction("划分方案介绍…")
        act_scheme.setToolTip("看图理解各种划分协议如何切分数据")
        act_scheme.triggered.connect(self._show_scheme_intro)
        m_help.addSeparator()
        act_order = m_help.addAction("使用顺序提示")
        act_order.triggered.connect(lambda: self.statusBar().showMessage(
            "使用顺序：数据管理 → 预处理工作台 → 训练控制台 → 结果分析", 8000))

        QShortcut(QKeySequence("Ctrl+B"), self, activated=self.toggle_nav)

    def _show_demo(self):
        from .widgets.demo_dialog import DemoWizardDialog

        DemoWizardDialog(self.state, self).exec()

    def _show_scheme_intro(self):
        from core import PROJECT_ROOT

        from .widgets.split_intro_dialog import SplitIntroDialog

        processed = PROJECT_ROOT / "processed" / "bnci2014001"
        SplitIntroDialog(str(processed), self).exec()

    def _switch_page(self, name: str):
        if name in self.PAGES:
            self._goto(self.PAGES.index(name))
