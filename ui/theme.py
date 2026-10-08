"""应用主题：**默认「浅紫粉」浅色主题**，其余配色方案可选。

- 默认 「浅紫粉」：整套浅色系（浅紫粉强调色 + 奶白底 + 淡紫侧栏），界面与图表共用。
- 备选 「夜兰樱雾」「海雾松岚」「赤陶海铜」：取自三张 Origin 科研配色卡，
  在「视图 → 切换主题」里切换，选定后全局生效（含图表与弹窗）。

配色卡出处（每套 5 色）：
- 方案 57 夜兰樱雾  #211A3E / #453370 / #A597B6 / #FEF3E8 / #D06C9D
- 方案 58 海雾松岚  #121F34 / #46657E / #BAD1D5 / #F6F1EC / #67AA82
- 方案 60 赤陶海铜  #4B1A18 / #AB352B / #D6904E / #EAE4E0 / #4982AE
"""
from __future__ import annotations

from dataclasses import dataclass

from core.plot_fonts import LATIN_FONT, apply_matplotlib_font  # noqa: F401


# --------------------------------------------------------------------- 颜色工具
def _rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _hex(rgb) -> str:
    return "#%02X%02X%02X" % tuple(max(0, min(255, int(v))) for v in rgb)


def mix(a: str, b: str, t: float) -> str:
    """按比例混合两色（t=0 → a，t=1 → b）。"""
    ra, rb = _rgb(a), _rgb(b)
    return _hex(ra[i] + (rb[i] - ra[i]) * t for i in range(3))


def rgba(color: str, alpha: float) -> str:
    """转成 QSS 可用的 rgba()。"""
    r, g, b = _rgb(color)
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


def rgb_text(color: str) -> str:
    return "RGB%s" % (_rgb(color),)


def _luma(color: str) -> float:
    r, g, b = _rgb(color)
    return 0.299 * r + 0.587 * g + 0.114 * b


# --------------------------------------------------------------------- 主题定义
@dataclass(frozen=True)
class Theme:
    """一套主题配色。

    dark_nav=False → 浅色侧边栏（默认主题用）；True → 深色侧边栏（配色卡方案用）。
    """

    name: str
    source: str
    deep: str            # 深色：标题 / 选中文字 / 深底
    mid: str             # 中间色：边框 / 次级强调
    soft: str            # 浅色：hover / 选中浅底
    bg: str              # 页面底色
    accent: str          # 强调色：主按钮 / 当前项
    link: str = ""       # 第二强调色（可选）
    dark_nav: bool = False

    # ---- 通用派生 ----
    @property
    def card(self) -> str:
        return "#FFFFFF"

    @property
    def text(self) -> str:
        return mix(self.deep, "#101014", 0.45)

    @property
    def dim(self) -> str:
        return self.mid

    @property
    def on_deep(self) -> str:
        return "#FFFFFF" if _luma(self.deep) < 150 else self.deep

    @property
    def border(self) -> str:
        return rgba(self.mid, 0.42)

    @property
    def accent_dark(self) -> str:
        return mix(self.accent, "#000000", 0.22)

    @property
    def link_color(self) -> str:
        return self.link or self.accent

    # ---- 侧边栏 / 顶栏 / 表头 ----
    @property
    def nav_bg(self) -> str:
        return self.deep if self.dark_nav else mix(self.soft, "#FFFFFF", 0.30)

    @property
    def nav_text(self) -> str:
        return self.on_deep if self.dark_nav else self.deep

    @property
    def nav_hover(self) -> str:
        return rgba(self.soft, 0.30) if self.dark_nav else rgba(self.mid, 0.18)

    @property
    def bar_bg(self) -> str:
        """顶栏 / 菜单栏 / 状态栏底色。"""
        return self.deep if self.dark_nav else self.card

    @property
    def bar_text(self) -> str:
        return self.on_deep if self.dark_nav else self.deep

    @property
    def bar_sub(self) -> str:
        return self.soft if self.dark_nav else self.mid

    @property
    def head_bg(self) -> str:
        """分组标题胶囊底色。"""
        return self.deep if self.dark_nav else self.soft

    @property
    def head_text(self) -> str:
        return self.on_deep if self.dark_nav else mix(self.deep, "#000000", 0.12)

    @property
    def thead_bg(self) -> str:
        return self.deep if self.dark_nav else mix(self.soft, "#FFFFFF", 0.10)

    @property
    def thead_text(self) -> str:
        return self.on_deep if self.dark_nav else self.deep


def _t(name, source, deep, mid, soft, bg, accent, link="", dark_nav=False) -> Theme:
    return Theme(name, source, deep, mid, soft, bg, accent, link, dark_nav)


#: 全部可选方案；**第 0 个是默认主题**
THEMES: list[Theme] = [
    # 默认：浅紫粉（浅色系，界面柔和、图表同色）
    _t("浅紫粉", "默认", deep="#4A3B63", mid="#9C8BBA", soft="#EDE7F6",
       bg="#FAF7FD", accent="#B06AA0", link="#8E6FC8"),
    # 备选：三张 Origin 配色卡
    _t("夜兰樱雾", "方案 57", deep="#211A3E", mid="#453370", soft="#A597B6",
       bg="#FEF3E8", accent="#D06C9D", dark_nav=True),
    _t("海雾松岚", "方案 58", deep="#121F34", mid="#46657E", soft="#BAD1D5",
       bg="#F6F1EC", accent="#67AA82", dark_nav=True),
    _t("赤陶海铜", "方案 60", deep="#4B1A18", mid="#AB352B", soft="#D6904E",
       bg="#EAE4E0", accent="#AB352B", link="#4982AE", dark_nav=True),
]

DEFAULT_THEME = THEMES[0]

#: 危险操作（删除）用的红，所有主题共用
DANGER = "#C0392B"
DANGER_DARK = "#96291D"

_current: Theme | None = None


def resolve(theme: "Theme | int | str | None") -> Theme:
    """把主题（对象 / 序号 / 名称）解析成 Theme。"""
    if isinstance(theme, Theme):
        return theme
    if isinstance(theme, int):
        return THEMES[theme % len(THEMES)]
    if isinstance(theme, str):
        for t in THEMES:
            if theme in (t.name, t.source):
                return t
    return DEFAULT_THEME


def current_theme() -> Theme:
    return _current or DEFAULT_THEME


# --------------------------------------------------------------------- QSS 生成
def build_qss(theme: "Theme | int | str") -> str:
    """按主题生成整套样式表。"""
    t = resolve(theme)
    return f"""
/* ============================ 全局 ============================ */
QWidget {{ background: {t.bg}; color: {t.text}; }}
QMainWindow, QDialog {{ background: {t.bg}; }}
QScrollArea, QStackedWidget, QSplitter {{ background: {t.bg}; border: none; }}
QMenuBar {{ background: {t.bar_bg}; color: {t.bar_text}; padding: 1px 4px; }}
QMenuBar::item {{ background: transparent; padding: 5px 11px; border-radius: 6px; }}
QMenuBar::item:selected {{ background: {t.accent}; color: #FFFFFF; }}
QMenu {{ background: {t.card}; border: 1px solid {t.mid}; padding: 5px; }}
QMenu::item {{ padding: 6px 24px; border-radius: 5px; }}
QMenu::item:selected {{ background: {t.accent}; color: #FFFFFF; }}
QMenu::separator {{ height: 1px; background: {t.border}; margin: 4px 8px; }}
QStatusBar {{ background: {t.bar_bg}; color: {t.bar_text};
              border-top: 1px solid {t.border}; }}
QStatusBar::item {{ border: none; }}
QToolTip {{ background: {t.deep}; color: {t.on_deep}; border: 1px solid {t.accent};
            padding: 4px 6px; }}

/* ========================= 顶栏（浅色，无彩色块） ========================= */
#TopBar {{ background: {t.bar_bg}; border-bottom: 1px solid {t.border}; }}
#Crumb {{ background: transparent; color: {t.bar_text}; font-size: 15px;
          font-weight: bold; }}
#CrumbSub {{ background: transparent; color: {t.bar_sub}; font-size: 11px; }}

/* ========================= 左侧步骤栏 ========================= */
#NavBox, #NavHeader, #NavRail {{ background: {t.nav_bg}; }}
#NavTitle {{ background: transparent; color: {t.nav_text}; font-weight: bold;
             padding: 4px 2px; }}
#NavList {{ background: {t.nav_bg}; border: none; outline: none; }}
#NavList::item {{ color: {t.nav_text}; padding: 9px 9px; margin: 2px 8px;
                  border-radius: 8px; }}
#NavList::item:hover {{ background: {t.nav_hover}; }}
#NavList::item:selected {{ background: {t.accent}; color: #FFFFFF; font-weight: bold; }}
#NavToggle {{ background: transparent; color: {t.nav_text};
              border: 1px solid {t.border}; border-radius: 8px;
              font-size: 13px; font-weight: bold; }}
#NavToggle:hover {{ background: {t.accent}; color: #FFFFFF; border-color: {t.accent}; }}
#RailHint {{ background: transparent; color: {t.dim}; font-size: 9px; }}
#RailNum {{ background: {t.card}; color: {t.nav_text}; border: 1px solid {t.border};
            border-radius: 8px; font-weight: bold; }}
#RailNum:hover {{ background: {t.soft}; }}
#RailNum[active="true"] {{ background: {t.accent}; color: #FFFFFF;
                           border-color: {t.accent}; }}

/* ========================== 分组卡片 ========================== */
QGroupBox {{ background: {t.card}; border: 1px solid {t.border}; border-radius: 11px;
             margin-top: 17px; padding-top: 9px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left;
                    left: 12px; padding: 3px 11px; background: {t.head_bg};
                    color: {t.head_text}; border-radius: 8px; font-weight: bold; }}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{ background: {t.border}; }}

/* ============================ 按钮 ============================ */
QPushButton {{ background: {t.card}; color: {t.text}; border: 1px solid {t.border};
               border-radius: 7px; padding: 5px 12px; }}
QPushButton:hover {{ background: {t.soft}; border-color: {t.mid}; }}
QPushButton:pressed {{ background: {t.mid}; color: #FFFFFF; }}
QPushButton:disabled {{ background: {t.bg}; color: {t.dim};
                        border-color: {t.border}; }}
QPushButton[accent="true"] {{ background: {t.accent}; color: #FFFFFF; border: none;
                              font-weight: bold; padding: 6px 14px; }}
QPushButton[accent="true"]:hover {{ background: {t.accent_dark}; }}
QPushButton[accent="true"]:disabled {{ background: {t.soft}; color: {t.dim}; }}
/* 危险操作（删除等）：平时低调，悬停变红 */
QPushButton[danger="true"] {{ background: {t.card}; color: {DANGER};
                              border: 1px solid {rgba(DANGER, 0.45)}; }}
QPushButton[danger="true"]:hover {{ background: {DANGER}; color: #FFFFFF;
                                    border-color: {DANGER}; }}
QPushButton[danger="true"]:pressed {{ background: {DANGER_DARK}; color: #FFFFFF; }}
QToolButton {{ background: transparent; border: 1px solid transparent;
               border-radius: 6px; padding: 3px 6px; }}
QToolButton:hover {{ background: {t.soft}; }}

/* ======================== 输入 / 下拉框 ======================== */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit,
QTextBrowser {{ background: {t.card}; border: 1px solid {t.border};
                border-radius: 6px; padding: 3px 6px;
                selection-background-color: {t.accent}; selection-color: #FFFFFF; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{ border: 1px solid {t.accent}; }}
QLineEdit:read-only, QPlainTextEdit:read-only {{ background: {rgba(t.bg, 0.75)}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox::down-arrow {{ image: none; width: 0; height: 0;
                         border-left: 4px solid transparent;
                         border-right: 4px solid transparent;
                         border-top: 5px solid {t.mid}; margin-right: 6px; }}
QComboBox QAbstractItemView {{ background: {t.card}; border: 1px solid {t.mid};
                               selection-background-color: {t.accent};
                               selection-color: #FFFFFF; outline: none; }}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 15px; background: {t.bg}; border: none; }}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {t.soft}; }}
QCheckBox, QRadioButton {{ background: transparent; spacing: 6px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 15px; height: 15px;
    border: 1px solid {t.mid}; border-radius: 4px; background: {t.card}; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {t.accent}; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {t.accent}; border-color: {t.accent}; }}
QSlider::groove:horizontal {{ height: 4px; background: {t.soft}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 13px; margin: -5px 0; border-radius: 6px;
                              background: {t.accent}; }}

/* ======================= 列表 / 表格 / 树 ======================= */
QListWidget, QTreeWidget, QTableWidget, QTableView {{ background: {t.card};
    border: 1px solid {t.border}; border-radius: 8px; outline: none;
    gridline-color: {t.border}; alternate-background-color: {rgba(t.soft, 0.30)}; }}
QListWidget::item, QTreeWidget::item {{ padding: 4px 6px; border-radius: 5px; }}
QListWidget::item:hover, QTreeWidget::item:hover {{ background: {t.soft}; }}
QListWidget::item:selected, QTreeWidget::item:selected,
QTableWidget::item:selected, QTableView::item:selected {{
    background: {t.accent}; color: #FFFFFF; }}
QTableWidget::item, QTableView::item {{ padding: 3px; }}
QHeaderView {{ background: {t.thead_bg}; }}
QHeaderView::section {{ background: {t.thead_bg}; color: {t.thead_text};
                        border: none; padding: 5px 6px; margin: 0; }}
QHeaderView::section:horizontal {{ border-right: 1px solid {t.border}; }}
QTableCornerButton::section {{ background: {t.thead_bg}; border: none; }}

/* ======================== 标签页 / 进度条 ======================== */
QTabWidget::pane {{ border: 1px solid {t.border}; border-radius: 9px;
                    background: {t.card}; top: -1px; }}
QTabBar::tab {{ background: transparent; color: {t.dim}; padding: 6px 14px;
                margin: 2px; border-radius: 7px; }}
QTabBar::tab:hover {{ background: {t.soft}; }}
QTabBar::tab:selected {{ background: {t.accent}; color: #FFFFFF; font-weight: bold; }}
QProgressBar {{ background: {t.bg}; border: 1px solid {t.border}; border-radius: 6px;
                text-align: center; color: {t.text}; }}
QProgressBar::chunk {{ background: {t.accent}; border-radius: 5px; }}

/* ========================= 滚动条 / 分割 ========================= */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {rgba(t.mid, 0.45)}; border-radius: 5px;
                               min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {t.accent}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {rgba(t.mid, 0.45)}; border-radius: 5px;
                                 min-width: 28px; }}
QScrollBar::handle:horizontal:hover {{ background: {t.accent}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background: {rgba(t.soft, 0.8)}; }}
QSplitter::handle:hover {{ background: {t.accent}; }}
"""


def _apply_plot_colors(t: Theme) -> None:
    """让 matplotlib 图表跟随主题（背景、坐标轴、网格、字体）。"""
    try:
        import matplotlib
    except Exception:  # pragma: no cover
        return
    matplotlib.rcParams.update({
        "figure.facecolor": t.card,
        "axes.facecolor": t.card,
        "axes.edgecolor": t.mid,
        "axes.labelcolor": t.text,
        "axes.titlecolor": t.deep,
        "text.color": t.text,
        "xtick.color": t.mid,
        "ytick.color": t.mid,
        "grid.color": t.soft,
        "legend.facecolor": t.card,
        "legend.edgecolor": t.soft,
        "savefig.facecolor": t.card,
    })


def apply_theme(target, theme: "Theme | int | str" = 0) -> Theme:
    """把主题应用到 QApplication / 任意控件，并同步图表配色。"""
    t = resolve(theme)
    global _current
    _current = t
    try:
        target.setStyleSheet(build_qss(t))
    except AttributeError:  # pragma: no cover
        pass
    _apply_plot_colors(t)
    return t


def theme_cmap(theme: "Theme | int | str | None" = None):
    """按主题生成的连续色带（用于混淆矩阵等热力图）。"""
    from matplotlib.colors import LinearSegmentedColormap

    t = resolve(theme) if theme is not None else current_theme()
    return LinearSegmentedColormap.from_list(
        "theme", [t.card, mix(t.soft, "#FFFFFF", 0.45), t.soft, t.accent, t.deep])


def plot_palette(theme: "Theme | int | str | None" = None) -> list[str]:
    """给图表用的配色序列（主色在前，适合多条曲线）。"""
    t = resolve(theme) if theme is not None else current_theme()
    return [t.accent, t.deep, t.mid, mix(t.accent, "#FFFFFF", 0.35),
            t.link_color, mix(t.deep, "#FFFFFF", 0.3)]
