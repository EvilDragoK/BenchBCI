"""时频分析：Morlet 小波 / 多窗（multitaper）/ STFT，ERD/ERS 基线归一化、频段提取与统计。

统一入口 ``compute_tfr`` 返回 ``(power (N,C,F,T), freqs, times)``，三种方法输出形状一致，
因此 UI / 统计 / 导出都能共用。
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12

#: 时频方法（UI 下拉用）
TFR_METHODS = ["morlet", "multitaper", "stft"]
TFR_LABELS = {
    "morlet": "Morlet 小波（时频分辨率均衡，最常用）",
    "multitaper": "多窗谱（DPSS，谱估计更平滑，适合功率比较）",
    "stft": "STFT 短时傅里叶（窗长固定，适合瞬态成分/ERP 时频）",
}
#: 下拉框短标签（长文本会把控件撑宽，详细说明放 tooltip）
TFR_SHORT = {"morlet": "Morlet 小波", "multitaper": "多窗 DPSS", "stft": "STFT"}

#: 基线归一化方式（ERD/ERS 相关）
BASELINE_MODES = ["db", "ers", "ratio", "z", "none"]
BASELINE_LABELS = {
    "db": "dB（10·log10 相对基线）",
    "ers": "ERD/ERS (%)（(P-P0)/P0）",
    "ratio": "比值（P/P0）",
    "z": "基线 z 分数",
    "none": "不做基线校正",
}
BASELINE_SHORT = {"db": "dB", "ers": "ERD/ERS (%)", "ratio": "比值",
                  "z": "z 分数", "none": "不校正"}


def _default_freqs(sfreq: float, lo: float = 1.0, hi: float = 45.0) -> np.ndarray:
    return np.arange(lo, min(hi, sfreq / 2 * 0.95), 1.0)


def _trim_freqs(freqs, sfreq: float, n_times: int, n_cycles: float) -> np.ndarray:
    """去掉「小波长度超过信号长度」的频点（小波类方法都受这个限制）。"""
    freqs = np.asarray(freqs, dtype=float)
    freqs = freqs[freqs < sfreq / 2]
    keep = n_cycles / np.maximum(freqs, EPS) * sfreq * 1.8 < n_times
    return freqs[keep]


# ================================================================ 三种时频方法
def morlet_tfr(X, sfreq, freqs=None, n_cycles: float = 4.0):
    """Morlet 小波时频功率。X (N,C,T) → (power (N,C,F,T), freqs (F,))。

    会自动过滤「小波长度超过信号长度」的低频点（避免 mne 报错）。
    """
    import mne

    X = np.asarray(X, dtype=np.float64)
    freqs = _default_freqs(sfreq) if freqs is None else freqs
    freqs = _trim_freqs(freqs, sfreq, X.shape[-1], n_cycles)
    if len(freqs) == 0:
        raise ValueError("没有可用的频率点（小波均长于信号），请降低 n_cycles 或缩短频率范围")
    power = mne.time_frequency.tfr_array_morlet(
        X, sfreq, freqs, n_cycles=n_cycles, output="power", zero_mean=True, verbose="ERROR")
    return np.asarray(power), freqs


def multitaper_tfr(X, sfreq, freqs=None, n_cycles: float = 4.0, time_bandwidth: float = 4.0):
    """多窗（DPSS multitaper）时频功率。→ (power (N,C,F,T), freqs)"""
    import mne

    X = np.asarray(X, dtype=np.float64)
    freqs = _default_freqs(sfreq) if freqs is None else freqs
    freqs = _trim_freqs(freqs, sfreq, X.shape[-1], n_cycles)
    if len(freqs) == 0:
        raise ValueError("没有可用的频率点（小波均长于信号），请缩短频率范围或加长分段")
    power = mne.time_frequency.tfr_array_multitaper(
        X, sfreq, freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth,
        output="power", zero_mean=True, verbose="ERROR")
    return np.asarray(power), freqs


def stft_tfr(X, sfreq, freqs=None, win_s: float = 0.4, step_s: float = 0.05):
    """STFT 短时傅里叶时频功率。→ (power (N,C,F,T), freqs, times)

    用 scipy 的 STFT 计算，再把频率轴插值到目标频点（保证与其它方法同形状）。
    """
    from scipy.signal import stft as _stft

    X = np.asarray(X, dtype=np.float64)
    T = X.shape[-1]
    nperseg = int(min(max(round(win_s * sfreq), 8), T))
    step = max(int(round(step_s * sfreq)), 1)
    noverlap = max(nperseg - step, 0)
    f, t, Z = _stft(X, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=-1)
    power = (np.abs(Z) ** 2)                     # (N,C,F',T')
    target = _default_freqs(sfreq) if freqs is None else np.asarray(freqs, dtype=float)
    # 丢弃 DC 附近：STFT 的 f[0] 通常是 0 Hz，取"最近频点"会落到直流上
    target = target[(target >= max(f[1] if len(f) > 1 else f[0], 1.0)) & (target <= f[-1])]
    if len(target) == 0:
        raise ValueError(f"目标频段 {target[:3]}… 超出 STFT 可分辨范围 [{f[0]:.2f}, {f[-1]:.2f}] Hz")
    idx = np.abs(f[None, :] - target[:, None]).argmin(axis=1)   # 最近频点
    power = power[:, :, np.unique(idx), :]
    freqs = f[np.unique(idx)]
    return power, freqs, t


def compute_tfr(X, sfreq, times=None, method: str = "morlet", freqs=None,
                n_cycles: float = 4.0, win_s: float = 0.4, step_s: float = 0.05,
                time_bandwidth: float = 4.0):
    """统一时频入口 → ``(power (N,C,F,T), freqs, times)``。

    ``times`` 仅对 morlet / multitaper 有效（它们与输入等长）；stft 由自身窗长决定。
    """
    method = str(method).lower()
    if method == "morlet":
        power, f = morlet_tfr(X, sfreq, freqs, n_cycles)
        t = np.asarray(times) if times is not None else np.arange(power.shape[-1]) / sfreq
    elif method == "multitaper":
        power, f = multitaper_tfr(X, sfreq, freqs, n_cycles, time_bandwidth)
        t = np.asarray(times) if times is not None else np.arange(power.shape[-1]) / sfreq
    elif method == "stft":
        power, f, t = stft_tfr(X, sfreq, freqs, win_s, step_s)
    else:
        raise ValueError(f"未知时频方法：{method}（可用：{TFR_METHODS}）")
    return np.asarray(power, dtype=np.float64), np.asarray(f, dtype=float), np.asarray(t, dtype=float)


# ================================================================ 基线 / ERD-ERS
def _safe_ratio(a, b):
    """按元素相除，分母为 0 时取 1（避免 EPS 与数据量级同阶造成的系统偏差）。

    注意：功率的量级取决于输入单位（Volt² 时约 1e-11，µV² 时约 1e-2），
    因此不能用固定 EPS 做除法——那会在大约 10% 的范围内偏掉结果。
    """
    b = np.asarray(b, dtype=float)
    return np.asarray(a, dtype=float) / np.where(np.abs(b) > 0, b, 1.0)


def tfr_baseline(power, times, baseline=(-0.5, 0.0), mode: str = "db"):
    """按基线窗做归一化（ERD/ERS 分析的第一步）。

    ``mode``：``db`` = 10·log10(P/P0)；``ers`` = (P-P0)/P0（%）；
    ``ratio`` = P/P0；``z`` = (P-mean(P0))/std(P0)；``none`` = 原样返回。
    """
    if mode in (None, "none"):
        return np.asarray(power, dtype=float)
    P = np.asarray(power, dtype=float)
    times = np.asarray(times, dtype=float)
    m = (times >= baseline[0]) & (times <= baseline[1])
    if not m.any():
        raise ValueError(f"基线窗 {baseline} 内没有时间点（可用范围 {times[0]:.2f}~{times[-1]:.2f}s）")
    base = P[..., m]
    p0 = base.mean(axis=-1, keepdims=True)
    if mode == "db":
        return 10.0 * np.log10(np.maximum(_safe_ratio(P, p0), 1e-30))
    if mode == "ers":
        return _safe_ratio(P - p0, p0) * 100.0
    if mode == "ratio":
        return _safe_ratio(P, p0)
    if mode == "z":
        sd = base.std(axis=-1, keepdims=True)
        return _safe_ratio(P - p0, sd)
    raise ValueError(f"未知基线方式：{mode}（可用：{BASELINE_MODES}）")


def band_tfr_power(power, freqs, bands=None):
    """把 TFR 功率按频带归并，返回 {band: (N,C,T)}。

    频点用**无重叠划分**（每个频点恰好归属一个频带）；没有频点覆盖的频带直接跳过
    （短分段 + 小波长度限制时 delta/theta 可能取不到，跳过比返回 NaN 更有用）。
    """
    from .features import BANDS, _band_masks

    bands = bands or BANDS
    freqs = np.asarray(freqs, dtype=float)
    out = {}
    for name, m in _band_masks(freqs, bands).items():
        if m.any():
            out[name] = power[:, :, m, :].mean(axis=2)
    return out


def band_window_topomap(power, freqs, times, band=(8.0, 13.0), tmin=0.0, tmax=1.0):
    """「频段 × 时间窗」平均功率 → (N, C)，用于画头皮地形图。

    这是时频分析的"地形图"版本：选一个频段与一段时间窗，看它在大脑上的分布
    （例如 alpha 在 0.5-1.5 s 的后部增强）。
    """
    P = np.asarray(power, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    times = np.asarray(times, dtype=float)
    fm = (freqs >= band[0]) & (freqs <= band[1])
    tm = (times >= tmin) & (times <= tmax)
    if not fm.any():
        raise ValueError(f"频段 {band} 内没有频点（可用 {freqs.min():.1f}~{freqs.max():.1f} Hz）")
    if not tm.any():
        raise ValueError(f"时间窗 [{tmin}, {tmax}] 内没有时间点（可用 {times[0]:.2f}~{times[-1]:.2f}s）")
    return P[:, :, fm, :][:, :, :, tm].mean(axis=(2, 3))


# ================================================================ 统计检验
def tfr_channel_stats(power, times, freqs, test_win=(0.5, 1.5), baseline=(-0.5, 0.0),
                      band=None, method: str = "ttest", correct_method: str = "fdr_bh",
                      n_perm: int = 0, paired: bool = True, random_state: int = 0):
    """逐通道（可选：先按频段归并）检验「检验窗 vs 基线」是否显著。

    Parameters
    ----------
    power : (N, C, F, T) 时频功率（未归一化）
    band : (lo, hi) 时先在该频段上平均，否则对每个频点各做一次检验（(C,F) 结果）
    method : ``ttest``（配对 t）/ ``perm``（配对置换，需 n_perm>0）

    Returns
    -------
    dict：``t`` / ``p`` / ``p_corrected`` / ``d`` / ``n`` 等，形状 (C,) 或 (C,F)
    """
    from .stats import correct, cohens_d, permutation_test, ttest_ind, ttest_rel

    P = np.asarray(power, dtype=float)
    times = np.asarray(times, dtype=float)
    tm = (times >= test_win[0]) & (times <= test_win[1])
    bm = (times >= baseline[0]) & (times <= baseline[1])
    if not tm.any() or not bm.any():
        raise ValueError("检验窗或基线窗为空，请检查时间范围")
    if band is not None:
        fm = (np.asarray(freqs, float) >= band[0]) & (np.asarray(freqs, float) <= band[1])
        if not fm.any():
            raise ValueError(f"频段 {band} 内没有频点")
        test = P[:, :, fm, :][:, :, :, tm].mean(axis=(2, 3))      # (N, C)
        base = P[:, :, fm, :][:, :, :, bm].mean(axis=(2, 3))      # (N, C)
    else:
        test = P[:, :, :, tm].mean(axis=3)                        # (N, C, F)
        base = P[:, :, :, bm].mean(axis=3)

    A = np.moveaxis(test, 0, -1)          # 观测量放最后：(..., N)
    B = np.moveaxis(base, 0, -1)
    if method == "perm":
        t, p = permutation_test(A, B, n_perm=max(int(n_perm), 200), paired=paired,
                                random_state=random_state)
    else:
        t, p = ttest_rel(A, B) if paired else ttest_ind(A, B)
    d = cohens_d(A, B, paired=paired)
    return {
        "t": t, "p": p, "p_corrected": correct(p, correct_method),
        "d": d, "n": A.shape[-1],
        "test_mean": np.moveaxis(test, 0, -1).mean(axis=-1),
        "base_mean": np.moveaxis(base, 0, -1).mean(axis=-1),
    }


# ================================================================ 导出
def tfr_to_rows(power, freqs, times, ch_names=None, agg: str = "mean"):
    """把 TFR 转成可写 CSV 的二维表：每行 = 通道×频点，列 = 时间点。"""
    P = np.asarray(power, dtype=float)
    if agg in ("mean", "median"):
        P = P.mean(axis=0) if agg == "mean" else np.median(P, axis=0)   # (C,F,T)
    else:
        raise ValueError("agg 仅支持 mean / median")
    freqs = np.asarray(freqs, float)
    times = np.asarray(times, float)
    names = list(ch_names) if ch_names is not None else [f"ch{i}" for i in range(P.shape[0])]
    header = ["通道", "频率(Hz)"] + [f"{t:.4f}" for t in times]
    rows = []
    for c in range(P.shape[0]):
        for k in range(P.shape[1]):
            rows.append([names[c], f"{freqs[k]:.4f}"] + [f"{v:.6g}" for v in P[c, k]])
    return header, rows
