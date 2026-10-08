"""多通道 EEG 信号浏览器（pyqtgraph，支持大时间窗流畅滚动）。

- 通道列表：勾选显示/隐藏，右键标记/取消坏导
- 时间窗翻页、窗长切换、垂直增益滑杆
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
                             QListWidget, QListWidgetItem, QMenu, QPushButton,
                             QSlider, QVBoxLayout, QWidget)

pg.setConfigOptions(antialias=False, background="w", foreground="k")


class SignalBrowser(QWidget):
    badChannelToggled = pyqtSignal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.raw = None
        self._curves: list[pg.PlotDataItem] = []
        self._t0 = 0.0
        self._marker: pg.InfiniteLine | None = None
        # 显示单位与缩放：原始数据为 Volt → 显示 µV（×1e6）；
        # 归一化后改为无量纲（scale=1.0，单位 z / a.u.）
        self._unit = "µV"
        self._scale = 1e6

        # ---- 工具栏 ----
        bar = QHBoxLayout()
        self.btn_prev = QPushButton("◀ 前一窗")
        self.btn_next = QPushButton("后一窗 ▶")
        self.spin_t0 = QDoubleSpinBox()
        self.spin_t0.setDecimals(1)
        self.spin_t0.setSuffix(" s")
        self.spin_t0.setSingleStep(1.0)
        self.combo_win = QComboBox()
        for w in (2, 5, 10, 30, 60):
            self.combo_win.addItem(f"{w} s", w)
        self.combo_win.setCurrentIndex(1)
        self.slider_gain = QSlider(Qt.Orientation.Vertical)
        self.slider_gain.setRange(2, 3000)   # 内部 ×0.1 → 增益 0.2 ~ 300
        self.slider_gain.setValue(400)       # 默认 40.0
        self.lbl_gain = QLabel("40 µV")
        bar.addWidget(QLabel("起始时间"))
        bar.addWidget(self.spin_t0)
        bar.addWidget(QLabel("窗长"))
        bar.addWidget(self.combo_win)
        bar.addWidget(self.btn_prev)
        bar.addWidget(self.btn_next)
        bar.addWidget(self.lbl_gain)
        bar.addStretch(1)

        # ---- 通道列表 + 绘图 ----
        self.list_channels = QListWidget()
        self.list_channels.setFixedWidth(160)
        self.list_channels.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "时间 (s)")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.getPlotItem().setClipToView(True)

        # 通道列：全选/全不选 + 勾选列表
        self.btn_all_ch = QPushButton("全选")
        self.btn_none_ch = QPushButton("全不选")
        self.btn_all_ch.setToolTip("显示全部通道")
        self.btn_none_ch.setToolTip("隐藏全部通道")
        ch_row = QHBoxLayout()
        ch_row.setContentsMargins(0, 0, 0, 0)
        ch_row.addWidget(self.btn_all_ch)
        ch_row.addWidget(self.btn_none_ch)
        left_col = QVBoxLayout()
        left_col.addLayout(ch_row)
        left_col.addWidget(self.list_channels, 1)

        body = QHBoxLayout()
        body.addLayout(left_col)
        body.addWidget(self.plot, 1)
        side = QVBoxLayout()
        side.addWidget(QLabel("增益"))
        side.addWidget(self.slider_gain)
        body.addLayout(side)

        root = QVBoxLayout(self)
        root.addLayout(bar)
        root.addLayout(body, 1)

        self.btn_prev.clicked.connect(lambda: self._page(-1))
        self.btn_next.clicked.connect(lambda: self._page(+1))
        self.spin_t0.valueChanged.connect(self._on_t0)
        self.combo_win.currentIndexChanged.connect(self._refresh)
        self.slider_gain.valueChanged.connect(self._on_gain)
        self.list_channels.itemChanged.connect(lambda *_: self._refresh())
        self.list_channels.customContextMenuRequested.connect(self._channel_menu)
        self.btn_all_ch.clicked.connect(lambda: self._set_all_channels(True))
        self.btn_none_ch.clicked.connect(lambda: self._set_all_channels(False))

    # ------------------------------------------------------------------
    def set_raw(self, raw):
        self.raw = raw
        # 每次装载新数据都回到「µV」显示与默认增益
        self._unit, self._scale = "µV", 1e6
        self.slider_gain.blockSignals(True)
        self.slider_gain.setValue(400)
        self.slider_gain.blockSignals(False)
        self.lbl_gain.setText("40 µV")
        fs = raw.info["sfreq"]
        dur = raw.n_times / fs
        self.spin_t0.blockSignals(True)
        self.spin_t0.setRange(0.0, max(0.0, dur - self._win()))
        self.spin_t0.setValue(0.0)
        self.spin_t0.blockSignals(False)
        self._t0 = 0.0

        self.list_channels.blockSignals(True)
        self.list_channels.clear()
        bads = set(raw.info["bads"])
        for i, name in enumerate(raw.ch_names):
            it = QListWidgetItem(name)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            if name in bads:
                self._mark_bad(it, True)
            self.list_channels.addItem(it)
        self.list_channels.blockSignals(False)

        self.plot.clear()
        self._curves = []
        self._marker = None  # 旧标记线随 plot.clear 移除
        for i, name in enumerate(raw.ch_names):
            c = self.plot.plot([], [], pen=pg.mkPen(_curve_color(i, name in bads), width=1))
            self._curves.append(c)
        self._refresh()

    # ------------------------------------------------------------------
    def view_window(self) -> tuple[float, float]:
        """当前显示窗口 (t0, t1)，单位秒。"""
        return (self._t0, self._t0 + self._win())

    def locate(self, t: float, label: str = ""):
        """把时刻 t 居中显示，并画一条带描述的竖直标记线。"""
        if self._marker is not None:
            try:
                self.plot.removeItem(self._marker)
            except Exception:  # noqa: BLE001
                pass
            self._marker = None
        self._marker = pg.InfiniteLine(
            pos=t, angle=90, movable=False,
            pen=pg.mkPen((230, 110, 10), width=2, style=Qt.PenStyle.DashLine),
            label=label or None,
            labelOpts={"position": 0.06, "color": (200, 90, 0)})
        self.plot.addItem(self._marker)
        self._set_t0(t - self._win() / 2)

    def _set_all_channels(self, checked: bool):
        """一键全选 / 全不选通道显示。"""
        st = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.list_channels.blockSignals(True)
        for i in range(self.list_channels.count()):
            self.list_channels.item(i).setCheckState(st)
        self.list_channels.blockSignals(False)
        self._refresh()

    # ------------------------------------------------------------------
    def _win(self) -> float:
        return float(self.combo_win.currentData() or 5.0)

    def _page(self, direction: int):
        self._set_t0(self._t0 + direction * self._win())

    def _on_t0(self, v):
        self._t0 = float(v)
        self._refresh()

    def _on_gain(self, v):
        g = float(v) / 10.0
        self.lbl_gain.setText(f"{g:.4g} {self._unit}")
        self._refresh()

    # ------------------------------------------------------------------
    def set_display(self, unit: str = "µV", scale: float = 1e6):
        """切换显示单位与缩放（归一化后传入 unit='z'/'a.u.'、scale=1.0）。"""
        self._unit = unit
        self._scale = scale
        self._on_gain(self.slider_gain.value())

    def auto_gain(self):
        """按当前数据中位通道幅度自动设置一个合适的通道间距。"""
        if self.raw is None:
            return
        n = min(self.raw.n_times, int(30.0 * self.raw.info["sfreq"]))
        d = self.raw.get_data(start=0, stop=n) * self._scale
        std = float(np.median(d.std(axis=1)))
        gain = float(np.clip(4.0 * std, 0.2, 300.0))
        self.slider_gain.setValue(int(round(gain * 10.0)))
        self._on_gain(self.slider_gain.value())

    def _set_t0(self, t):
        dur = self.raw.n_times / self.raw.info["sfreq"]
        t = float(np.clip(t, 0.0, max(0.0, dur - self._win())))
        self._t0 = t
        self.spin_t0.blockSignals(True)
        self.spin_t0.setValue(round(t, 2))
        self.spin_t0.blockSignals(False)
        self._refresh()

    def _channel_menu(self, pos):
        it = self.list_channels.itemAt(pos)
        if it is None or self.raw is None:
            return
        name = it.text()
        is_bad = name in self.raw.info["bads"]
        menu = QMenu(self)
        act = menu.addAction("取消坏导" if is_bad else "标记坏导")
        act.triggered.connect(lambda: self._toggle_bad(it))
        menu.exec(self.list_channels.mapToGlobal(pos))

    def _toggle_bad(self, item: QListWidgetItem):
        name = item.text()
        bads = set(self.raw.info["bads"])
        now_bad = name not in bads
        if now_bad:
            bads.add(name)
        else:
            bads.discard(name)
        self.raw.info["bads"] = sorted(bads)
        self._mark_bad(item, now_bad)
        for i, ch in enumerate(self.raw.ch_names):
            self._curves[i].setPen(pg.mkPen(_curve_color(i, ch in bads), width=1))
        self.badChannelToggled.emit(name, now_bad)
        self._refresh()

    def _mark_bad(self, item: QListWidgetItem, bad: bool):
        item.setForeground(QColor(200, 30, 30) if bad else QColor(0, 0, 0))
        item.setText(item.text().split(" ✗")[0] + (" ✗" if bad else ""))

    # ------------------------------------------------------------------
    def _refresh(self):
        if self.raw is None:
            return
        raw = self.raw
        fs = raw.info["sfreq"]
        win = self._win()
        start = int(self._t0 * fs)
        stop = min(int((self._t0 + win) * fs) + 1, raw.n_times)
        if stop <= start:
            return
        data = raw.get_data(start=start, stop=stop) * self._scale  # 按显示单位缩放
        times = np.arange(start, stop) / fs
        gain = float(self.slider_gain.value()) / 10.0
        y0 = -gain * len(raw.ch_names) / 2
        for i, curve in enumerate(self._curves):
            it = self.list_channels.item(i)
            visible = it is None or it.checkState() == Qt.CheckState.Checked
            if not visible:
                curve.setData([], [])
                continue
            curve.setData(times, data[i] + y0 + i * gain)
        self.plot.setYRange(y0 - gain, y0 + gain * (len(raw.ch_names) + 1), padding=0)
        self.plot.setXRange(times[0], times[-1], padding=0.01)


def _curve_color(i: int, bad: bool):
    if bad:
        return "#c8102e"
    return pg.intColor(i, hues=11)
