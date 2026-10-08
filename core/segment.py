"""把 Raw 切成 epoch 数组（供分析页 / 特征提取复用）。"""
from __future__ import annotations

import numpy as np


def make_epochs(raw, tmin: float = 0.5, tmax: float = 2.5, max_epochs: int = 500,
                baseline=None):
    """返回 (X, labels, times)。

    - 有事件注释时按事件分段（labels = 事件描述，times = 相对 tmin 的时间轴）
    - 无事件时退化为 2s 滑窗（50% 重叠），labels = 'seg0'...
    X: (N, C, T) float64，单位与 raw 一致（通常 Volt）。
    """
    import mne

    fs = float(raw.info["sfreq"])
    if len(raw.annotations):
        try:
            events, event_id = mne.events_from_annotations(raw)
        except Exception:  # noqa: BLE001
            events, event_id = np.zeros((0, 3), int), {}
        if len(events):
            ep = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax,
                            baseline=baseline, preload=True, proj=False,
                            event_repeated="drop")
            if len(ep):
                X = ep.get_data()[:max_epochs].astype(np.float64)
                inv = {int(v): k for k, v in ep.event_id.items()}
                codes = ep.events[:, 2][:max_epochs]
                labels = [inv.get(int(c), str(int(c))) for c in codes]
                return X, labels, ep.times

    # 滑窗 fallback
    data = raw.get_data()
    win = max(int((tmax - tmin) * fs), 8)
    step = max(win // 2, 1)
    segs = [data[:, s:s + win] for s in range(0, data.shape[1] - win + 1, step)]
    X = np.stack(segs[:max_epochs]).astype(np.float64)
    labels = [f"seg{i}" for i in range(len(X))]
    times = tmin + np.arange(win) / fs
    return X, labels, times
