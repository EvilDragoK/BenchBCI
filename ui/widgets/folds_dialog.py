"""自定义折配置编辑器：每一折自己指定「训练 / 验证 / 测试」的受试者名单。

- 受试者写成 `1,3,5` 或 `1-5` 均可；验证受试者留空时，训练时会自动从该折的
  训练集里按比例划出验证集（绝不触碰测试受试者）。
- 「自动生成 N 折」按受试者轮转平分，便于从 5-fold 起步再手工微调。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QSpinBox, QTableWidget,
                             QTableWidgetItem, QVBoxLayout)

_HEADERS = ["折名", "训练受试者", "验证受试者（可留空）", "测试受试者"]


def subjects_from_processed(processed_dir: str) -> list[int]:
    """从预处理输出目录读取可用受试者（sid.npy）。"""
    path = Path(processed_dir) / "sid.npy"
    if not path.exists():
        return []
    return sorted(int(v) for v in np.unique(np.load(path)))


def auto_kfold(subjects, n_folds: int = 5) -> list[dict]:
    """受试者轮转平分为 n_folds 折（测试集轮转，训练集为补集，验证留空=自动划分）。"""
    subs = sorted(int(s) for s in subjects)
    out: list[dict] = []
    n_folds = max(int(n_folds), 2)
    for i in range(n_folds):
        te = [s for j, s in enumerate(subs) if j % n_folds == i]
        tr = [s for s in subs if s not in te]
        if te and tr:
            out.append({"name": f"fold{i + 1}", "train": tr, "val": [], "test": te})
    return out


def _fmt(ids) -> str:
    return ", ".join(str(int(v)) for v in (ids or []))


def _parse(text: str, available: list[int]) -> list[int]:
    """解析 "1,3,5" / "1-5" / "all" / 空 → 受试者列表。"""
    text = (text or "").strip()
    if not text:
        return []
    if text.lower() in ("all", "全部", "*"):
        return list(available)
    out: list[int] = []
    for chunk in text.replace("，", ",").replace("、", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, _, b = chunk.partition("-")
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                continue
            out.extend(range(min(lo, hi), max(lo, hi) + 1))
        elif chunk.isdigit():
            out.append(int(chunk))
    return sorted(set(out))


class FoldsDialog(QDialog):
    """折配置表格编辑器。"""

    def __init__(self, subjects: list[int], folds: list[dict] | None = None,
                 processed_dir: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("自定义折配置 —— 每折指定 训练 / 验证 / 测试 受试者")
        self.resize(760, 460)
        self._available = sorted(int(s) for s in subjects)

        head = QLabel(
            f"数据中的可用受试者：<b>{_fmt(self._available) or '（未找到，请先在预处理页生成数据）'}</b>"
            f"<br><span style='color:#666'>填法：<code>1,3,5</code> 或 <code>1-5</code>；"
            f"验证受试者留空表示自动从该折训练集中按比例划出。</span>")
        head.setWordWrap(True)

        self.tbl = QTableWidget(0, len(_HEADERS))
        self.tbl.setHorizontalHeaderLabels(_HEADERS)

        self.spin_auto = QSpinBox()
        self.spin_auto.setRange(2, 20)
        self.spin_auto.setValue(5)
        btn_auto = QPushButton("自动生成 N 折")
        btn_add = QPushButton("＋ 添加折")
        btn_del = QPushButton("－ 删除选中")
        btn_clear = QPushButton("清空")
        btn_ok = QPushButton("确定")
        btn_cancel = QPushButton("取消")

        btn_auto.clicked.connect(self._auto)
        btn_add.clicked.connect(lambda: self._append_row({"name": "", "train": [],
                                                          "val": [], "test": []}))
        btn_del.clicked.connect(self._remove_selected)
        btn_clear.clicked.connect(lambda: self.tbl.setRowCount(0))
        btn_ok.clicked.connect(self._accept)
        btn_cancel.clicked.connect(self.reject)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("生成折数"))
        row1.addWidget(self.spin_auto)
        row1.addWidget(btn_auto)
        row1.addWidget(btn_add)
        row1.addWidget(btn_del)
        row1.addWidget(btn_clear)
        row1.addStretch(1)

        row2 = QHBoxLayout()
        row2.addStretch(1)
        row2.addWidget(btn_ok)
        row2.addWidget(btn_cancel)

        lay = QVBoxLayout(self)
        lay.addWidget(head)
        lay.addLayout(row1)
        lay.addWidget(self.tbl, 1)
        lay.addLayout(row2)

        for f in (folds or auto_kfold(self._available, 5)):
            self._append_row(f)

    # ------------------------------------------------------------------
    def _append_row(self, fold: dict):
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        for c, key in enumerate(("name", "train", "val", "test")):
            val = fold.get(key)
            text = str(val) if key == "name" and val else _fmt(val)
            self.tbl.setItem(r, c, QTableWidgetItem(text))

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            self.tbl.removeRow(r)

    def _auto(self):
        self.tbl.setRowCount(0)
        for f in auto_kfold(self._available, int(self.spin_auto.value())):
            self._append_row(f)

    def _accept(self):
        try:
            folds = self.get_folds()
        except ValueError as e:
            QMessageBox.warning(self, "配置有误", str(e))
            return
        if not folds:
            QMessageBox.warning(self, "配置有误", "至少需要配置一折")
            return
        self.accept()

    # ------------------------------------------------------------------
    def get_folds(self) -> list[dict]:
        """读回配置并校验（受试者是否存在、角色是否重叠）。"""
        folds: list[dict] = []
        for r in range(self.tbl.rowCount()):
            def cell(c):
                it = self.tbl.item(r, c)
                return it.text() if it else ""

            name = cell(0).strip() or f"fold{r + 1}"
            tr = _parse(cell(1), self._available)
            va = _parse(cell(2), self._available)
            te = _parse(cell(3), self._available)
            if not tr or not te:
                raise ValueError(f"第 {r + 1} 行「{name}」必须同时填写训练与测试受试者")
            unknown = [s for s in tr + va + te if s not in self._available]
            if unknown:
                raise ValueError(f"第 {r + 1} 行含未知受试者 {sorted(set(unknown))}；"
                                 f"可用：{self._available}")
            overlap = ((set(tr) & set(te)) | (set(va) & set(te)) | (set(tr) & set(va)))
            if overlap:
                raise ValueError(f"第 {r + 1} 行「{name}」的角色重叠：{sorted(overlap)}")
            folds.append({"name": name, "train": tr, "val": va, "test": te})
        return folds
