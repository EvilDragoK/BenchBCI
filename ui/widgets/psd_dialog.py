"""PSD（功率谱）对比对话框：滤波/处理前后叠加查看。"""
from __future__ import annotations

import numpy as np

from PyQt6.QtWidgets import QDialog, QVBoxLayout

from .plotting import new_canvas


def _mean_psd(raw, fmax=80.0):
    """计算 EEG 通道平均功率谱（Welch），返回 (freqs, 平均 dB, 各通道 psd)。"""
    import mne

    if mne.pick_types(raw.info, eeg=True, exclude=[]).size == 0:
        picks = None
    else:
        picks = "eeg"
    spec = raw.compute_psd(method="welch", fmin=0.5, fmax=fmax, picks=picks)
    psds, freqs = spec.get_data(return_freqs=True)
    return freqs, 10 * np.log10(psds.mean(axis=0) + 1e-30), psds


#: 时域对比优先展示的代表性通道
_PREF_CH = ["C3", "Cz", "C4", "FC3", "FC4", "Pz"]


def _wave_compare(ax, before, after, dur: float = 2.0):
    """时域波形对比：取前 dur 秒的 3 条代表性通道（处理后在上层着色）。"""
    ref = after if after is not None else before
    if ref is None:
        return
    fs = float(ref.info["sfreq"])
    n = int(min(dur * fs, ref.n_times))
    if n < 8:
        return
    chans = [c for c in _PREF_CH if c in ref.ch_names][:3]
    if not chans:
        chans = ref.ch_names[:3]
    t = np.arange(n) / fs
    # 用首通道幅度估算通道间距，兼容 µV 与归一化数据
    i0 = ref.ch_names.index(chans[0])
    seg0 = ref.get_data(picks=[i0], start=0, stop=n)[0]
    span = 4.0 * float(np.std(seg0) + 1e-12)
    for k, ch in enumerate(chans):
        i = ref.ch_names.index(ch)
        off = k * span
        if before is not None and ch in before.ch_names:
            j = before.ch_names.index(ch)
            ax.plot(t, before.get_data(picks=[j], start=0, stop=n)[0] + off,
                    color="#9aa0a6", lw=0.9, alpha=0.8,
                    label="处理前" if k == 0 else None)
        if after is not None and ch in after.ch_names:
            j = after.ch_names.index(ch)
            ax.plot(t, after.get_data(picks=[j], start=0, stop=n)[0] + off,
                    color="#d62728", lw=0.9,
                    label="处理后" if k == 0 else None)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("幅值（数据单位）+ 通道偏移")
    ax.set_title(f"时域波形对比（前 {min(dur, ref.n_times / fs):.1f} s，"
                 f"通道 {'/'.join(chans)}）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


class PsdCompareDialog(QDialog):
    """处理前/后对比：上=时域波形，下=功率谱。before/after 可为 None。"""

    def __init__(self, before, after=None, title="处理前后对比", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(820, 660)
        self.fig, self.canvas = new_canvas(8.2, 6.6)
        fig, canvas = self.fig, self.canvas
        ax_wave = fig.add_subplot(211)
        ax_psd = fig.add_subplot(212)
        if before is not None or after is not None:
            try:
                _wave_compare(ax_wave, before, after)
            except Exception:  # noqa: BLE001
                ax_wave.set_title("时域波形对比（不可用）")
        if before is not None:
            f, m, _ = _mean_psd(before)
            ax_psd.plot(f, m, color="#1f77b4", label="处理前", lw=1.2)
        if after is not None:
            f, m, _ = _mean_psd(after)
            ax_psd.plot(f, m, color="#d62728", label="处理后", lw=1.2)
        ax_psd.set_xlabel("频率 (Hz)")
        ax_psd.set_ylabel("功率 (dB)")
        ax_psd.set_title("平均功率谱密度（EEG 通道）")
        ax_psd.legend(fontsize=8)
        ax_psd.grid(alpha=0.3)
        lay = QVBoxLayout(self)
        lay.addWidget(canvas)
