"""BCI Lab 应用入口。

用法::

    python main.py              # 启动 GUI
    python main.py --selftest   # 无界面自检（构造全部窗口后退出，用于 CI）
"""
from __future__ import annotations

import sys


def main() -> int:
    from pathlib import Path

    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("BCI Lab")
    app.setStyle("Fusion")

    # ---- 全局字体规范：数字/字母 Times New Roman，中文宋体（界面与图表统一）----
    from core.plot_fonts import apply_matplotlib_font, qt_font
    from ui.theme import apply_theme

    app.setFont(qt_font(12))
    apply_matplotlib_font()
    apply_theme(app, 0)      # 启动时用第 1 页（数据管理）的主题

    # 应用图标（跨平台：窗口标题栏/任务栏/Dock）
    icon_path = Path(__file__).resolve().parent / "icon.png"
    icon = QIcon(str(icon_path)) if icon_path.exists() else QIcon()
    if not icon.isNull():
        app.setWindowIcon(icon)

    from ui.main_window import MainWindow

    win = MainWindow()
    if not icon.isNull():
        win.setWindowIcon(icon)
    win.show()

    if "--selftest" in sys.argv:
        # 无头环境自检：构造完成后立即退出
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(500, app.quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
