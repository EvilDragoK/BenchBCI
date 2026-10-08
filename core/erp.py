"""ERP 时域成分分析：叠加平均（可多条件对比）、成分提取、单试次成分与统计检验。

流程对应经典 ERP 分析四步：
1. 数据叠加：``erp_average`` / ``erp_conditions``（多条件对比）
2. 画波形图：返回 (C, T) 波形给 UI 绘制
3. 提取成分：``extract_components``（平均波形峰值）+ ``single_trial_components``（逐试次）
4. 统计分析：``component_condition_stats``（成分幅值/潜伏期的条件间检验 + FDR 校正）
"""
from __future__ import annotations

import numpy as np

#: 默认 ERP 成分定义：(极性, 时间窗下限 s, 上限 s)
DEFAULT_COMPONENTS = {
    "N1": ("neg", 0.05, 0.15),
    "P2": ("pos", 0.12, 0.25),
    "N2": ("neg", 0.18, 0.30),
    "P3": ("pos", 0.25, 0.50),
}


def baseline_correct(X, times, baseline=(-0.2, 0.0)):
    """基线校正：减去每试次每通道在基线窗的均值。X (N,C,T) → 同形。"""
    times = np.asarray(times, dtype=float)
    m = (times >= baseline[0]) & (times <= baseline[1])
    if not m.any():
        raise ValueError(f"基线窗 {baseline} 内没有时间点（可用 {times[0]:.3f}~{times[-1]:.3f}s）")
    return np.asarray(X, dtype=float) - np.asarray(X, dtype=float)[..., m].mean(axis=-1, keepdims=True)


def erp_average(X, labels, condition: str = "all"):
    """对指定条件（或全部）做叠加平均。X (N,C,T) → (C,T)。"""
    if condition and condition != "all":
        m = np.array([l == condition for l in labels])
        X = X[m]
    if len(X) == 0:
        raise ValueError(f"条件 {condition} 下没有 epoch")
    return X.mean(axis=0)


def erp_conditions(X, labels, conds=None):
    """多条件叠加平均 → ``({条件: (C,T)}, {条件: 试次数})``，用于多条件波形对比。"""
    labels = list(labels)
    unique = sorted(set(labels))
    conds = [c for c in (list(conds) if conds else unique) if c in unique]
    if not conds:
        raise ValueError(f"没有任何可用条件（数据里有：{unique[:6]}）")
    X = np.asarray(X, dtype=float)
    waves, counts = {}, {}
    for c in conds:
        m = np.array([l == c for l in labels])
        if not m.any():
            continue
        waves[c] = X[m].mean(axis=0)
        counts[c] = int(m.sum())
    if not waves:
        raise ValueError("所选条件都没有 epoch")
    return waves, counts


def single_trial_components(times, X, components=None):
    """逐试次成分提取（做统计检验的前提）。

    Returns
    -------
    {name: {"amplitude": (N,C), "latency_ms": (N,C)}} —— 每个试次每个通道在成分窗口
    内的极值（负性成分取最小、正性取最大）及其潜伏期。
    """
    comps = components or DEFAULT_COMPONENTS
    times = np.asarray(times, dtype=float)
    X = np.asarray(X, dtype=float)          # (N, C, T)
    out = {}
    for name, (pol, lo, hi) in comps.items():
        m = (times >= lo) & (times <= hi)
        if not m.any():
            continue
        seg = X[:, :, m]                    # (N, C, T')
        k = np.argmin(seg, axis=-1) if pol == "neg" else np.argmax(seg, axis=-1)
        n, c = np.indices(k.shape)
        out[name] = {
            "amplitude": seg[n, c, k],
            "latency_ms": times[np.flatnonzero(m)][k] * 1000.0,
        }
    if not out:
        raise ValueError("时间轴与成分窗口没有交集，请检查 tmin/tmax")
    return out


def component_condition_stats(times, X, labels, conds=None, components=None,
                              paired: bool = True, correct_method: str = "fdr_bh",
                              n_perm: int = 0):
    """成分幅值/潜伏期的**条件间统计检验**（逐通道，含多重比较校正）。

    Returns
    -------
    dict::

        {"conds": (a, b), "components": {
            "P3": {"amplitude": <group_report>, "latency": <group_report>}, ...}}

    其中 ``group_report`` 见 :mod:`core.stats`（含 t、p、校正后 p、Cohen's d）。
    """
    from .stats import group_report

    single = single_trial_components(times, X, components)
    labels = list(labels)
    unique = sorted(set(labels))
    if conds:
        use = [c for c in conds if c in unique]
    else:
        use = unique
    if len(use) < 2:
        raise ValueError(f"统计检验至少需要 2 个条件（数据里有：{unique[:6]}）")
    a, b = use[0], use[1]
    ma = np.array([l == a for l in labels])
    mb = np.array([l == b for l in labels])

    result = {"conds": (a, b), "n": {a: int(ma.sum()), b: int(mb.sum())}, "components": {}}
    for name, d in single.items():
        rep = {}
        for metric in ("amplitude", "latency_ms"):
            V = d[metric]                                  # (N, C)
            va = np.moveaxis(V[ma], 0, -1)                 # (C, n_a)
            vb = np.moveaxis(V[mb], 0, -1)
            rep["amplitude" if metric == "amplitude" else "latency"] = group_report(
                va, vb, paired=paired, correct_method=correct_method, n_perm=n_perm)
        result["components"][name] = rep
    return result


def extract_components(times, wave, components=None):
    """在时间窗内找各成分极值。

    Parameters
    ----------
    times : (T,) 时间轴（秒，相对 tmin）
    wave : (C, T) ERP 平均波形
    components : {name: (polarity, lo, hi)}，默认 DEFAULT_COMPONENTS

    Returns
    -------
    {name: {"latency_ms": (C,) 潜伏期, "amplitude": (C,) 峰值幅值}}
    """
    comps = components or DEFAULT_COMPONENTS
    out = {}
    for name, (pol, lo, hi) in comps.items():
        m = (times >= lo) & (times <= hi)
        idx = np.where(m)[0]
        if pol == "neg":
            k = np.argmin(wave[:, m], axis=1)
        else:
            k = np.argmax(wave[:, m], axis=1)
        k_global = idx[k]  # 每通道在全局时间轴上的索引
        lat_ms = times[k_global] * 1000.0
        amp = wave[np.arange(wave.shape[0]), k_global]
        out[name] = {"latency_ms": lat_ms, "amplitude": amp}
    return out


def component_peak_values(times, wave, components=None):
    """各成分峰值时刻的每通道幅值，用于成分地形图。→ {name: (C,)}"""
    comps = components or DEFAULT_COMPONENTS
    peaks = {}
    for name, (pol, lo, hi) in comps.items():
        m = (times >= lo) & (times <= hi)
        idx = np.where(m)[0]
        k = np.argmin(wave[:, m], axis=1) if pol == "neg" else np.argmax(wave[:, m], axis=1)
        k_global = idx[k]
        peaks[name] = wave[np.arange(wave.shape[0]), k_global]
    return peaks
