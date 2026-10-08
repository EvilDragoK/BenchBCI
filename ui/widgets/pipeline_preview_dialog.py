"""整条流水线一键预览：各步骤执行报告表 + 最终输出可视化。

- 最终输出是 Raw → 左上：时域波形对比；右上：处理前后 PSD
- 最终输出是 Epochs → 左：各类 ERP 平均波形；右：各类 trial 数
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel,
                             QMessageBox, QPushButton, QTableWidget,
                             QTableWidgetItem, QVBoxLayout)

from .plotting import new_canvas


class PipelinePreviewDialog(QDialog):
    def __init__(self, before, after, ctx: dict, title: str = "流水线预览", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 720)
        self.fig, self.canvas = new_canvas(10.0, 4.6)

        self.tbl = QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(["#", "步骤", "状态", "耗时 (s)", "参数"])
        self._fill_report(ctx)

        self.lbl_sum = QLabel(self._summary_text(after, ctx))
        self.lbl_sum.setWordWrap(True)
        self.lbl_sum.setStyleSheet(
            "background:#f2f5f8; border:1px solid #dde3ea; border-radius:4px; padding:6px;")
        btn_png = QPushButton("导出预览图")
        btn_png.clicked.connect(self._export)
        top = QHBoxLayout()
        top.addWidget(self.lbl_sum, 1)
        top.addWidget(btn_png)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(QLabel("执行报告（各步骤耗时；灰色=已跳过）"))
        lay.addWidget(self.tbl)
        self._draw(before, after)

    # ---------------------------------------------------------------- 报告
    def _fill_report(self, ctx: dict):
        log = ctx.get("log", [])
        self.tbl.setRowCount(len(log))
        for i, e in enumerate(log):
            skipped = bool(e.get("skipped"))
            vals = [str(i + 1), str(e.get("op", "")),
                    "跳过" if skipped else "执行",
                    "-" if skipped else f"{e.get('seconds', 0.0):.3f}",
                    str(e.get("params", ""))[:90]]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if skipped:
                    it.setForeground(Qt.GlobalColor.gray)
                self.tbl.setItem(i, j, it)
        self.tbl.resizeColumnsToContents()

    @staticmethod
    def _summary_text(after, ctx: dict) -> str:
        total = sum(e.get("seconds", 0.0) for e in ctx.get("log", []) if not e.get("skipped"))
        n_skip = sum(1 for e in ctx.get("log", []) if e.get("skipped"))
        rej = ctx.get("rejected_trials", 0)
        import mne

        if isinstance(after, mne.BaseEpochs):
            desc = (f"最终输出：Epochs — {len(after)} 个试次 × {len(after.ch_names)} 通道 "
                    f"× {len(after.times)} 采样点，采样率 {after.info['sfreq']:g} Hz")
            if rej:
                desc += f"；振幅剔除 {rej} 个试次"
        else:
            desc = (f"最终输出：Raw — {len(after.ch_names)} 通道 × {after.n_times} 采样点，"
                    f"采样率 {after.info['sfreq']:g} Hz，时长 "
                    f"{after.n_times / after.info['sfreq']:.1f} s")
        if n_skip:
            desc += f"；跳过 {n_skip} 步"
        return desc + f"\n流水线总耗时 {total:.3f} s"

    # ---------------------------------------------------------------- 可视化
    def _draw(self, before, after):
        import mne

        self.fig.clear()
        try:
            if isinstance(after, mne.BaseEpochs):
                self._draw_epochs(after)
            else:
                self._draw_raw(before, after)
        except Exception as e:  # noqa: BLE001
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"可视化失败：{type(e).__name__}: {e}",
                    ha="center", va="center", color="#c8102e")
            ax.set_axis_off()
        self.canvas.draw()

    def _draw_epochs(self, epochs):
        X = epochs.get_data()  # (N, C, T)（单位与数据一致）
        codes = np.asarray(epochs.events[:, 2])
        names = {int(v): k for k, v in (epochs.event_id or {}).items()}
        # 判断单位：Volt → µV；归一化后数据保持原样
        scale, unit = (1e6, "µV") if np.median(np.abs(X)) < 1e-3 else (1.0, "a.u.")
        t = epochs.times

        ax1 = self.fig.add_subplot(121)
        for code in np.unique(codes):
            m = codes == code
            if m.sum() == 0:
                continue
            wave = X[m].mean(axis=(0, 1)) * scale  # 跨试次 + 跨通道平均
            ax1.plot(t, wave, lw=1.5, label=names.get(int(code), str(code)))
        ax1.set_xlabel("时间 (s)")
        ax1.set_ylabel(f"幅值 ({unit})")
        ax1.set_title(f"分段后 ERP（{len(epochs)} 试次，各类平均）")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        ax2 = self.fig.add_subplot(122)
        vals, cnts = np.unique(codes, return_counts=True)
        labels = [names.get(int(v), str(v)) for v in vals]
        bars = ax2.bar(labels, cnts, color="#4c78a8")
        for b, c in zip(bars, cnts):
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height(), str(int(c)),
                     ha="center", va="bottom", fontsize=9)
        ax2.set_ylabel("试次数")
        ax2.set_title("各条件 trial 数")
        ax2.tick_params(axis="x", labelrotation=20)
        ax2.grid(alpha=0.3, axis="y")

    def _draw_raw(self, before, after):
        from .psd_dialog import _mean_psd, _wave_compare

        ax1 = self.fig.add_subplot(121)
        _wave_compare(ax1, before, after)

        ax2 = self.fig.add_subplot(122)
        if before is not None:
            f, m, _ = _mean_psd(before)
            ax2.plot(f, m, color="#1f77b4", lw=1.2, label="处理前")
        if after is not None:
            f, m, _ = _mean_psd(after)
            ax2.plot(f, m, color="#d62728", lw=1.2, label="处理后")
        ax2.set_xlabel("频率 (Hz)")
        ax2.set_ylabel("功率 (dB)")
        ax2.set_title("平均功率谱（处理前 vs 处理后）")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

    # ---------------------------------------------------------------- 导出
    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出预览图", "pipeline_preview.png",
                                             "PNG (*.png);;SVG (*.svg)")
        if not path:
            return
        self.fig.savefig(path, dpi=160)
        QMessageBox.information(self, "完成", f"已导出 {path}")
