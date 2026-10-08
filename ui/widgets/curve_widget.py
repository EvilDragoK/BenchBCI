"""训练实时曲线控件（pyqtgraph）：train/val loss 与 accuracy。

要点：
- **按「折」分别存储**：多折（kfold/loso/session…）训练时每折的曲线互不混合，
  可用顶部「折」下拉框 / ◀ ▶ 切换查看；默认自动跟随最新折。
- 两个图**固定高度**，不随窗口拉伸（避免把日志区挤没）。
- 数据由控件自己维护，不依赖 pyqtgraph 的取数 API。
"""
from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel,
                             QPushButton, QVBoxLayout, QWidget)

pg.setConfigOptions(antialias=True, background="w", foreground="k")

_COLORS = {"train": "#1f77b4", "val": "#d62728", "test": "#2ca02c"}
_KEYS = ("train_loss", "val_loss", "train_acc", "val_acc")
_PLOT_MIN_H = 190
_PLOT_MAX_H = 265


class LiveCurves(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- 顶部：折选择工具条 ----
        self.combo_fold = QComboBox()
        self.combo_fold.setMinimumWidth(150)
        self.btn_prev = QPushButton("◀")
        self.btn_next = QPushButton("▶")
        self.btn_prev.setFixedWidth(30)
        self.btn_next.setFixedWidth(30)
        self.chk_follow = QCheckBox("跟随最新折")
        self.chk_follow.setChecked(True)
        self.chk_follow.setToolTip("勾选时自动切到正在训练的折；取消后可固定查看某一折")
        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color:#666; font-size:11px;")

        bar = QHBoxLayout()
        bar.setContentsMargins(2, 2, 2, 0)
        bar.addWidget(QLabel("折"))
        bar.addWidget(self.combo_fold)
        bar.addWidget(self.btn_prev)
        bar.addWidget(self.btn_next)
        bar.addWidget(self.chk_follow)
        bar.addStretch(1)
        bar.addWidget(self.lbl_info)

        # ---- 两个固定高度的图 ----
        self.p1 = pg.PlotWidget(title="Loss")
        self.p2 = pg.PlotWidget(title="Accuracy")
        for p in (self.p1, self.p2):
            p.addLegend(offset=(10, 10))
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel("bottom", "epoch")
            p.setMinimumHeight(_PLOT_MIN_H)
            p.setMaximumHeight(_PLOT_MAX_H)
        self.p1.setLabel("left", "loss")
        self.p2.setLabel("left", "acc (%)")     # 纵轴用百分比，不显示 0~1 小数

        self._curves: dict[str, object] = {}
        for key in _KEYS:
            target = self.p1 if "loss" in key else self.p2
            base = key.split("_")[0]
            dash = (4, 3) if "acc" in key else None
            self._curves[key] = target.plot(
                [], [], pen=pg.mkPen(_COLORS.get(base, "#7f7f7f"), width=2, dash=dash),
                name=key)

        self._folds: dict[str, dict[str, tuple[list[float], list[float]]]] = {}
        self._order: list[str] = []
        self._current = ""

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(self.p1)
        lay.addWidget(self.p2)

        self.combo_fold.currentTextChanged.connect(self._on_combo)
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next.clicked.connect(lambda: self._step(+1))

    # ------------------------------------------------------------------ 数据
    def add_point(self, key: str, x, y, fold: str = "fold"):
        """追加一个数据点。fold = 折标识（训练时传入该折的 tag）。"""
        if x is None or y is None:
            return
        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            return
        if "acc" in key:            # 准确率统一换算成百分比显示
            y *= 100.0
        tag = str(fold or "fold")
        if tag not in self._folds:
            self._folds[tag] = {}
            self._order.append(tag)
            self._sync_combo()
        buf = self._folds[tag].setdefault(key, ([], []))
        buf[0].append(x)
        buf[1].append(y)
        if self.chk_follow.isChecked() or self._current == tag:
            self.set_fold(tag)
        else:
            self._update_info()

    def set_fold(self, tag: str):
        """切换显示某一折。"""
        if tag not in self._folds:
            return
        self._current = tag
        self.combo_fold.blockSignals(True)
        idx = self.combo_fold.findText(tag)
        if idx >= 0:
            self.combo_fold.setCurrentIndex(idx)
        self.combo_fold.blockSignals(False)
        self._refresh()
        self._update_info()

    def clear_all(self):
        self._folds.clear()
        self._order.clear()
        self._current = ""
        self.combo_fold.blockSignals(True)
        self.combo_fold.clear()
        self.combo_fold.blockSignals(False)
        self.lbl_info.setText("")
        self._refresh()

    # ------------------------------------------------------------------ 内部
    def _sync_combo(self):
        self.combo_fold.blockSignals(True)
        self.combo_fold.clear()
        self.combo_fold.addItems(self._order)
        if self._current in self._order:
            self.combo_fold.setCurrentIndex(self._order.index(self._current))
        self.combo_fold.blockSignals(False)

    def _on_combo(self, text: str):
        if text and text in self._folds and text != self._current:
            self._current = text
            self._refresh()
            self._update_info()

    def _step(self, delta: int):
        if not self._order:
            return
        i = self._order.index(self._current) if self._current in self._order else 0
        self.set_fold(self._order[max(0, min(i + delta, len(self._order) - 1))])

    def _refresh(self):
        """按当前折重画全部曲线（没数据的曲线置空）。"""
        data = self._folds.get(self._current, {})
        for key, curve in self._curves.items():
            xs, ys = data.get(key, ([], []))
            curve.setData(xs, ys)

    def _update_info(self):
        data = self._folds.get(self._current, {})
        xs_all = data.get("train_loss", ([], []))[0]
        n = len(xs_all)
        total = len(self._order)
        idx = self._order.index(self._current) + 1 if self._current in self._order else 0
        self.lbl_info.setText(
            f"第 {idx}/{total} 折 · {n} 个 epoch" if n else f"第 {idx}/{total} 折")
