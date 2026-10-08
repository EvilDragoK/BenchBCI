"""模型定义查看器：展示训练控制台中选中模型的 PyTorch (nn.Module) 源码。

训练页「模型」下拉框旁点「查看模型定义」即可跳转到这里，直接看到该模型的
网络结构定义代码（class xxx(nn.Module)），便于对照论文与调参。
"""
from __future__ import annotations

import inspect
from pathlib import Path

from PyQt6.QtCore import Qt, QRegularExpression, QUrl
from PyQt6.QtGui import (QColor, QDesktopServices, QFont, QSyntaxHighlighter,
                         QTextCharFormat)
from PyQt6.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel,
                             QMessageBox, QPlainTextEdit, QPushButton,
                             QVBoxLayout)


class PythonHighlighter(QSyntaxHighlighter):
    """极简 Python 语法高亮（关键字 / 字符串 / 注释 / 数字 / 装饰器）。"""

    def __init__(self, doc):
        super().__init__(doc)

        def fmt(color: str, bold: bool = False, italic: bool = False) -> QTextCharFormat:
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            if bold:
                f.setFontWeight(QFont.Weight.Bold)
            f.setFontItalic(italic)
            return f

        self.rules: list[tuple[QRegularExpression, QTextCharFormat]] = []
        kw = fmt("#0033b3", bold=True)
        for group in (
            "class|def|return|if|elif|else|for|while|import|from|as|with|try|except|"
            "finally|raise|pass|break|continue|in|is|not|and|or|lambda|None|True|"
            "False|yield|global|assert|del|async|await",
            "self",
        ):
            self.rules.append((QRegularExpression(rf"\b(?:{group})\b"), kw))
        self.rules.append((QRegularExpression(r"\bself\b"), fmt("#945200", italic=True)))
        self.rules.append((QRegularExpression(r"#[^\n]*"), fmt("#8c8c8c", italic=True)))
        for q in ('"""[^"]*"""', r"'''[^']*'''", r"'[^'\n]*'", r'"[^"\n]*"'):
            self.rules.append((QRegularExpression(q), fmt("#067d17")))
        self.rules.append((QRegularExpression(r"\b\d+\.?\d*\b"), fmt("#1750eb")))
        self.rules.append((QRegularExpression(r"@\w+"), fmt("#9e880d")))
        self.rules.append((QRegularExpression(r"\b(?:nn|torch|F|np|mne)\b"),
                           fmt("#7a3e9d")))

    def highlightBlock(self, text):  # noqa: N802 (Qt 命名)
        for pattern, fmt_ in self.rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), fmt_)


def model_code(name: str) -> tuple[str, str, str | None]:
    """按模型名取源码，返回 (标题, 源码, 文件路径)。

    支持四类：自定义模型、本项目 nn.Module、braindecode 基础模型、sklearn 经典基线。
    """
    from core.models import TORCH_MODELS, is_custom

    if is_custom(name):       # 自定义模型：直接读 models_custom/<name>.py
        from core.models.custom import list_custom, read_source

        meta = list_custom().get(name, {})
        base = meta.get("base") or ""
        title = f"{name} — 自定义模型" + (f"（基于 {base} 改写）" if base else "（从零设计）")
        note = meta.get("note") or ""
        return (title + (f"\n{note}" if note else ""), read_source(name),
                meta.get("file"))

    if name in TORCH_MODELS:
        cls = TORCH_MODELS[name]
        return (f"{cls.__name__} — PyTorch nn.Module",
                inspect.getsource(cls), inspect.getsourcefile(cls))

    # EEG 基础模型：源码来自 braindecode
    from core.models.foundation import get_spec, is_foundation

    if is_foundation(name):
        spec = get_spec(name)
        try:
            import braindecode.models as bm
        except ImportError as e:
            raise ImportError(
                f"查看「{spec.display}」的定义需要 braindecode：\n    pip install braindecode") from e
        for cls_name in (spec.cls_name, spec.interp_cls):
            cls = getattr(bm, cls_name or "", None)
            if cls is not None:
                return (f"{cls.__name__} — Braindecode 基础模型（{spec.display}）",
                        inspect.getsource(cls), inspect.getsourcefile(cls))
        raise ImportError(
            f"当前 braindecode 版本未提供 {spec.display} 的实现类；\n"
            f"官方来源：{spec.source or spec.arch_repo or '（见文档）'}")

    from core.models.sklearn_baselines import ClassicBaseline

    return (f"ClassicBaseline —— sklearn 管线（{name}）",
            inspect.getsource(ClassicBaseline),
            inspect.getsourcefile(ClassicBaseline))


def model_summary(name: str) -> str:
    """模型一句话摘要（含参数量），失败时返回空串。"""
    try:
        from core.models import build_model, is_classic, is_custom

        if is_classic(name):
            return f"经典基线：{name}（CSP/FBCSP 特征 + 线性分类器，无神经网络参数）"
        if is_custom(name):
            from core.models.custom import list_custom

            meta = list_custom().get(name, {})
            base = meta.get("base") or ""
            m = build_model(name, 4, 22, 501, {})
            n_par = sum(p.numel() for p in m.parameters())
            origin = f"基于 {base} 改写" if base else "从零设计"
            return (f"自定义模型（{origin}）；"
                    f"参数量 {n_par:,}（示例配置：4 类 / 22 通道 / 501 采样点）")
        import torch

        m = build_model(name, 4, 22, 501, {})
        n_par = sum(p.numel() for p in m.parameters())
        n_lyr = sum(1 for _ in m.modules())
        return (f"参数量 {n_par:,}（示例配置：4 类 / 22 通道 / 501 采样点）；"
                f"子模块 {n_lyr} 个")
    except Exception as e:  # noqa: BLE001
        return f"（无法构建示例模型：{type(e).__name__}）"


class CodeViewerDialog(QDialog):
    """只读代码窗口：模型定义源码 + 语法高亮 + 复制 / 打开所在目录。"""

    def __init__(self, title: str, code: str, file_path: str | None = None,
                 summary: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"模型定义 — {title}")
        self.resize(960, 720)

        head = QLabel(f"<b>{title}</b><br><span style='color:#666'>{summary}</span>")
        head.setWordWrap(True)
        path_lb = QLabel(f"<span style='color:#888; font-size:11px'>{file_path or ''}</span>")
        path_lb.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.txt = QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setPlainText(code)
        self.txt.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(12)
        self.txt.setFont(mono)
        self._hl = PythonHighlighter(self.txt.document())

        btn_copy = QPushButton("复制代码")
        btn_open = QPushButton("在 Finder 中显示")
        btn_close = QPushButton("关闭")
        btn_copy.clicked.connect(self._copy)
        if file_path:
            btn_open.clicked.connect(lambda: self._reveal(file_path))
        else:
            btn_open.setEnabled(False)
        btn_close.clicked.connect(self.accept)

        row = QHBoxLayout()
        row.addWidget(btn_copy)
        row.addWidget(btn_open)
        row.addStretch(1)
        row.addWidget(btn_close)

        lay = QVBoxLayout(self)
        lay.addWidget(head)
        lay.addWidget(path_lb)
        lay.addWidget(self.txt, 1)
        lay.addLayout(row)

    def _copy(self):
        QApplication.clipboard().setText(self.txt.toPlainText())
        QMessageBox.information(self, "完成", "模型定义代码已复制到剪贴板")

    def _reveal(self, path: str):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))
