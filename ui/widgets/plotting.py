"""matplotlib 画布工具（中文友好字体、跨平台）。"""
from __future__ import annotations

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt  # noqa: E402

from core.plot_fonts import apply_matplotlib_font  # noqa: E402

# 数字/字母 Times New Roman、中文宋体（与界面同一套字体栈）
apply_matplotlib_font()

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


def new_canvas(width=8, height=5, dpi=100):
    fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
    canvas = FigureCanvas(fig)
    return fig, canvas


def new_canvas_with_toolbar(width=8, height=5, dpi=100):
    """返回 (fig, canvas, toolbar)。toolbar 提供缩放/平移/复位（可放大横纵轴）。"""
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

    fig, canvas = new_canvas(width, height, dpi)
    toolbar = NavigationToolbar2QT(canvas, canvas)
    return fig, canvas, toolbar
