"""全局字体规范：**数字/字母用 Times New Roman，中文用宋体**。

界面（Qt）与出图（matplotlib）共用同一套字体栈，保证「界面上看到的」和
「图里画出来的」字形一致。字体栈顺序即回退顺序：拉丁字形先走 Times New
Roman，Times 缺的中文字形再回退到宋体族（Songti SC / SimSun / STSong …）。
"""
from __future__ import annotations

#: 拉丁文与数字字体
LATIN_FONT = "Times New Roman"

#: 中文宋体族（按平台优先级：macOS → Windows → Linux）
CJK_FONTS = (
    "Songti SC",          # macOS「宋体」
    "STSong",             # macOS 华文宋体
    "SimSun",             # Windows 宋体
    "NSimSun",
    "Noto Serif CJK SC",  # Linux
    "Source Han Serif SC",
    "Microsoft YaHei",    # 兜底
    "PingFang SC",
)

#: 最终兜底（保证任何平台都有字形）
FALLBACK = ("DejaVu Serif", "DejaVu Sans")


def font_stack() -> list[str]:
    """Qt / matplotlib 通用字体族列表：拉丁在前，中文在后。"""
    return [LATIN_FONT, *CJK_FONTS]


def qt_font(point_size: int = 12):
    """构造符合规范的 QFont（需在导入 PyQt 的环境下调用）。"""
    from PyQt6.QtGui import QFont

    f = QFont()
    try:
        f.setFamilies(font_stack())         # Qt6：多族回退（数字 Times、中文宋体）
    except AttributeError:                   # 兼容旧版：只设首选族
        f.setFamily(LATIN_FONT)
    f.setPointSize(point_size)
    f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return f


def apply_matplotlib_font(size: float = 10.0) -> list[str]:
    """设置 matplotlib 全局字体。

    只保留系统里真实存在的字体族——否则 matplotlib 每画一张图都会打印
    "findfont: Font family ... not found" 的告警。
    """
    try:
        import matplotlib
        from matplotlib import font_manager, rcParams
    except Exception:  # pragma: no cover - 无 matplotlib 环境
        return []

    have = {f.name for f in font_manager.fontManager.ttflist}
    stack = [n for n in font_stack() if n in have]
    stack += [n for n in FALLBACK if n in have and n not in stack]
    if not stack:
        return []

    # 关键：font.family 必须是**具体字体名列表**，逐字形回退才生效。
    # 若写成 "serif"（走 font.serif 别名），matplotlib 只会用列表里的第一个字体，
    # 中文就会变成方框。
    try:
        ver = tuple(int(x) for x in matplotlib.__version__.split(".")[:2])
    except Exception:  # pragma: no cover
        ver = (3, 6)
    if ver >= (3, 6):
        rcParams["font.family"] = stack
    else:  # 老版本不支持逐字形回退 → 优先保证中文不出现方框
        rcParams["font.family"] = [stack[1] if len(stack) > 1 else stack[0], *stack]

    rcParams["font.serif"] = stack
    rcParams["font.sans-serif"] = stack
    rcParams["mathtext.fontset"] = "stix"    # 数学符号与 Times 风格一致
    rcParams["axes.unicode_minus"] = False   # 负号用 ASCII，避免缺字形
    rcParams["font.size"] = size
    # 宋体族常只有 Light 一个字重，matplotlib 会刷 "Failed to find font weight"
    # 的 debug 级日志；这里静音，保持控制台干净。
    import logging

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    return stack
