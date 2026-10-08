"""EEG 手工特征提取合集（对应特征工程论文中的方法清单）。

所有函数输入 X: (N, C, T) 数组（N=epoch 数, C=通道数, T=采样点）与 sfreq，
输出逐 epoch / 逐通道的特征数组。统一入口 `extract_features` 返回拼接好的
特征矩阵 (N, D) 与特征名列表，可直接喂给 sklearn / PyTorch。

| 缩写 | 方法 | 输出 |
|---|---|---|
| AP      | 绝对功率（各频带积分功率）          | (N, C, B) |
| APR     | θ/β 频带绝对功率比 APβ/APθ          | (N, C)    |
| DE      | 微分熵（逐频带带通后高斯假设）      | (N, C, B) |
| SHE     | 香农熵（信号幅度直方图）            | (N, C)    |
| SPE     | 谱熵（PSD 归一化熵）                | (N, C)    |
| C0      | C0 复杂度（FFT 阈值重构能量比）     | (N, C)    |
| NSI     | 非平稳指数（分窗方差异质性，简化版）| (N, C)    |
| HOC     | 高阶过零特征（k 阶差分过零率）      | (N, C, K) |
| DASM    | 差分不对称性（左右半球 DE 之差）    | (N, P, B) |
| RASM    | 合理不对称性（左右半球 DE 之比）    | (N, P, B) |
| AH_TIMF | Hilbert 变换 IMF 幅值特征           | (N, C, K) |
| IPH_TIMF| Hilbert 变换 IMF 瞬时相位/频率特征  | (N, C, K) |
| FC      | 功能连接（corr / plv / coh）        | (N, C, C) |

B=频带数（默认 5），P=左右对称通道对数，K=IMF/阶数。
"""
from __future__ import annotations

import re

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch

EPS = 1e-12

#: 标准 EEG 频带定义
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

#: extract_features 支持的特征名
FEATURES = ["AP", "APR", "DE", "SHE", "SPE", "C0", "NSI", "HOC",
            "DASM", "RASM", "AH_TIMF", "IPH_TIMF", "FC"]


# ================================================================ 工具
def _band_filter(X: np.ndarray, sfreq: float, lo: float, hi: float, order: int = 4):
    """零相位带通滤波（沿最后一维，支持批量 (N,C,T)）。"""
    nyq = sfreq / 2.0
    b, a = butter(order, [lo / nyq, min(hi / nyq, 0.999)], btype="band")
    return filtfilt(b, a, X, axis=-1)


def _welch(X: np.ndarray, sfreq: float, nperseg: int = 256):
    nperseg = min(nperseg, X.shape[-1])
    return welch(X, fs=sfreq, nperseg=nperseg, axis=-1)


# ================================================================ 频域特征
def absolute_power(X, sfreq, bands=None):
    """AP：各频带绝对功率（PSD 积分）。→ (N, C, B)"""
    bands = bands or BANDS
    f, p = _welch(X, sfreq)
    out = []
    for lo, hi in bands.values():
        m = (f >= lo) & (f <= hi)
        out.append(np.trapezoid(p[..., m], f[m], axis=-1))
    return np.stack(out, axis=-1)


def theta_beta_ratio(X, sfreq):
    """APβ/APθ：theta 与 beta 频带绝对功率之比。→ (N, C)"""
    ap = absolute_power(X, sfreq, bands={"theta": BANDS["theta"], "beta": BANDS["beta"]})
    return ap[..., 0] / (ap[..., 1] + EPS)


#: 功率谱估计方法（UI 下拉用）
PSD_METHODS = ["welch", "periodogram", "multitaper"]
PSD_LABELS = {
    "welch": "Welch（分段平均，方差小，最常用）",
    "periodogram": "周期图（不分段，频率分辨高、方差大）",
    "multitaper": "多窗 DPSS（谱最平滑，适合逐频点比较）",
}
#: 下拉框里用短标签（长文本会按最长条目把控件撑宽），详细说明放 tooltip
PSD_SHORT = {"welch": "Welch", "periodogram": "周期图", "multitaper": "多窗 DPSS"}


def psd(X, sfreq, method: str = "welch", fmin: float = 1.0, fmax: float | None = None,
        nperseg: int = 256):
    """功率谱密度。X (N,C,T) → (freqs (F,), psd (N,C,F))。

    ``method``：``welch``（分段平均）/ ``periodogram``（周期图）/ ``multitaper``（DPSS 多窗）。
    ``fmax`` 默认取 Nyquist 的 95% 以内。
    """
    X = np.asarray(X, dtype=np.float64)
    fs = float(sfreq)
    nyq = fs / 2.0
    fmax = float(fmax) if fmax else min(60.0, nyq * 0.95)
    method = str(method).lower()
    if method == "multitaper":
        import mne

        p, f = mne.time_frequency.psd_array_multitaper(
            X, sfreq=fs, fmin=fmin, fmax=fmax, verbose="ERROR")
        return np.asarray(f), np.asarray(p)
    if method == "periodogram":
        from scipy.signal import periodogram

        f, p = periodogram(X, fs=fs, axis=-1)
    elif method == "welch":
        f, p = _welch(X, fs, nperseg=nperseg)
    else:
        raise ValueError(f"未知谱估计方法：{method}（可用：{PSD_METHODS}）")
    m = (f >= fmin) & (f <= fmax)
    return np.asarray(f[m]), np.asarray(p[..., m])


def _band_masks(freqs, bands):
    """把离散频点**无重叠、无遗漏**地分给各频带（键顺序 = bands 顺序）。

    为什么不用 ``(f >= lo) & (f <= hi)``：离散频点下相邻频带会共享边界点（重复计数），
    用梯形积分还会漏掉边界点之间的区间，各频带功率之和不等于总功率。
    这里让每个频点恰好归属一个频带，配合矩形求和即可严格加和。
    """
    freqs = np.asarray(freqs, dtype=float)
    items = list(bands.items())
    masks = {}
    for i, (name, (lo, hi)) in enumerate(items):
        last = (i == len(items) - 1)
        m = (freqs >= lo) & ((freqs <= hi) if last else (freqs < hi))
        masks[name] = m
    return masks


def relative_power(X, sfreq, bands=None, method: str = "welch"):
    """相对频带功率（各频带占总功率的百分比，各频带之和 = 100%）。→ (N, C, B)"""
    bands = bands or BANDS
    # 分母必须与分子同范围：fmin/fmax 取频带并集，否则会把范围外功率算进分母
    f, p = psd(X, sfreq, method=method,
               fmin=min(lo for lo, _ in bands.values()),
               fmax=max(hi for _, hi in bands.values()))
    masks = _band_masks(f, bands)
    ap = np.stack([p[..., m].sum(axis=-1) for m in masks.values()], axis=-1)  # 矩形求和
    total = p.sum(axis=-1)
    # 不用固定 EPS 做除法：功率量级随单位变化（V²≈1e-11 / µV²≈1e-2），
    # 固定 EPS 会让相对功率在大约 10% 的范围内偏掉。
    denom = np.where(total > 0, total, 1.0)
    return ap / denom[..., None] * 100.0


def psd_table(freqs, psd_mean, ch_names, fmin: float = 1.0, fmax: float | None = None,
              relative: bool = False, bands=None):
    """PSD 表格（供 CSV 导出 / 界面显示）。

    Returns
    -------
    (header, rows)：``rows`` 每行 = 一个通道，列为频点功率（dB）或频带相对功率（%）。
    """
    freqs = np.asarray(freqs, dtype=float)
    P = np.asarray(psd_mean, dtype=float)          # (C, F)
    if relative:
        bands = bands or BANDS
        masks = _band_masks(freqs, bands)
        rel = np.stack([P[:, m].sum(axis=1) for m in masks.values()], axis=1)   # (C, B)
        tot = rel.sum(axis=1, keepdims=True)
        rel = rel / np.where(tot > 0, tot, 1.0) * 100.0
        return ["通道"] + list(bands), [[str(c)] + [f"{v:.2f}" for v in rel[i]]
                                        for i, c in enumerate(ch_names)]
    fmax = float(fmax) if fmax else freqs[-1]
    m = (freqs >= fmin) & (freqs <= fmax)
    return ["通道"] + [f"{v:.2f}" for v in freqs[m]], \
        [[str(c)] + [f"{10 * np.log10(v + 1e-30):.3f}" for v in P[i][m]]
         for i, c in enumerate(ch_names)]


def differential_entropy(X, sfreq, bands=None):
    """DE：逐频带带通滤波后，高斯假设下的微分熵 0.5·ln(2πe·σ²)。→ (N, C, B)"""
    bands = bands or BANDS
    out = []
    for lo, hi in bands.values():
        xb = _band_filter(X, sfreq, lo, hi)
        out.append(0.5 * np.log(2 * np.pi * np.e * np.var(xb, axis=-1) + EPS))
    return np.stack(out, axis=-1)


# ================================================================ 熵特征
def shannon_entropy(X, n_bins: int = 64):
    """SHE：信号幅度分布的香农熵（逐 epoch 逐通道直方图）。→ (N, C)"""
    X = np.asarray(X)
    N, C = X.shape[0], X.shape[1]
    out = np.empty((N, C), dtype=np.float64)
    for i in range(N):
        for j in range(C):
            h, _ = np.histogram(X[i, j], bins=n_bins)
            p = h[h > 0] / X.shape[-1]
            out[i, j] = float(-(p * np.log2(p)).sum())
    return out


def spectral_entropy(X, sfreq, normalize: bool = True):
    """SPE：PSD 归一化为分布后的香农熵。→ (N, C)，normalize 时取值 [0,1]"""
    f, p = _welch(X, sfreq)
    q = (p + EPS) / (p + EPS).sum(-1, keepdims=True)
    H = -(q * np.log(q)).sum(-1)
    if normalize:
        H = H / np.log(q.shape[-1])
    return H


# ================================================================ 复杂度/非平稳
def c0_complexity(X, sfreq=None, M: float = 3.0):
    """C0 复杂度：FFT 中幅值 ≤ M·平均幅值的"不重要"分量重构能量占比。→ (N, C)，[0,1]"""
    F = np.fft.rfft(X, axis=-1)
    amp = np.abs(F)
    thr = M * amp.mean(axis=-1, keepdims=True)
    Fc = np.where(amp > thr, F, 0)
    xhat = np.fft.irfft(Fc, n=X.shape[-1], axis=-1)
    resid = X - xhat
    return (resid ** 2).sum(-1) / ((X ** 2).sum(-1) + EPS)


def nsi(X, n_seg: int = 8):
    """NSI 非平稳指数（简化实现）：epoch 内分 n_seg 窗，各窗方差的变异系数。

    NSI = std(v_i)/mean(v_i)，越大越非平稳。→ (N, C)
    （严格定义基于局部协方差结构，这里用方差异异性做实用近似。）
    """
    N, C, T = X.shape
    T2 = (T // n_seg) * n_seg
    v = X[..., :T2].reshape(N, C, n_seg, T2 // n_seg).var(axis=-1)
    return v.std(axis=-1) / (v.mean(axis=-1) + EPS)


def hoc(X, order: int = 7):
    """HOC 高阶过零特征：对信号逐阶差分，统计各阶符号序列的过零率。→ (N, C, order)"""
    out = []
    d = np.asarray(X)
    for _ in range(order):
        d = np.diff(d, axis=-1)
        s = np.sign(d)
        zc = (s[..., 1:] * s[..., :-1] < 0).sum(-1)
        out.append(zc / d.shape[-1])
    return np.stack(out, axis=-1)


# ================================================================ 半球不对称
def hemisphere_pairs(ch_names):
    """按国际 10-20 命名自动配对左右半球通道（奇数=左，偶数=右，z/中线跳过）。

    返回 [(left_idx, right_idx), ...]
    """
    left, right = {}, {}
    for i, name in enumerate(ch_names):
        m = re.match(r"^([A-Za-z]+)(\d+)", str(name))
        if not m:
            continue
        pre, num = m.group(1).upper(), int(m.group(2))
        if num % 2 == 1:
            left[(pre, num, num + 1)] = i
        else:
            right[(pre, num - 1, num)] = i
    return [(li, right[k]) for k, li in sorted(left.items()) if k in right]


def dasm(X, sfreq, ch_names, bands=None):
    """DASM 差分不对称性：对称通道对的 DE(left) − DE(right)。→ ((N, P, B), pairs)"""
    de = differential_entropy(X, sfreq, bands)
    pairs = hemisphere_pairs(ch_names)
    if not pairs:
        raise ValueError("通道命名中未找到左右对称对（如 C3/C4）")
    return np.stack([de[:, li, :] - de[:, ri, :] for li, ri in pairs], axis=1), pairs


def rasm(X, sfreq, ch_names, bands=None):
    """RASM 合理不对称性：对称通道对的 DE(left) / DE(right)。→ ((N, P, B), pairs)"""
    de = differential_entropy(X, sfreq, bands)
    pairs = hemisphere_pairs(ch_names)
    if not pairs:
        raise ValueError("通道命名中未找到左右对称对（如 C3/C4）")
    return np.stack([de[:, li, :] / (de[:, ri, :] + EPS) for li, ri in pairs], axis=1), pairs


# ================================================================ Hilbert / IMF
def hilbert_timf(X, sfreq=None, n_imf: int = 4, use_emd: bool = False):
    """Hilbert 变换本征模态函数（IMF）特征。

    - use_emd=True 且安装了 PyEMD（`pip install EMD-signal`）时：先经验模态分解取前
      n_imf 个 IMF，再对每个 IMF 做 Hilbert 变换（**很慢**，逐 epoch 逐通道分解）。
    - 否则（默认）：把原信号视为单一模态（退化版本，向量化、极快）。

    Returns
    -------
    (ah, iph)：
        ah  : 各 IMF 解析信号幅值的均值        → (N, C, K)
        iph : 各 IMF 平均瞬时频率 (Hz)（瞬时相位的差分统计）→ (N, C, K)
    """
    X = np.asarray(X, dtype=np.float64)
    if use_emd:
        try:
            from PyEMD import EMD  # noqa: F401
        except ImportError:
            use_emd = False

    if not use_emd:
        ana = hilbert(X, axis=-1)
        amp = np.abs(ana)
        ph = np.angle(ana)
        inst_freq = np.diff(np.unwrap(ph, axis=-1), axis=-1) / (2 * np.pi) * (sfreq or 1.0)
        return amp.mean(-1, keepdims=True), inst_freq.mean(-1, keepdims=True)

    emd = EMD()
    N, C = X.shape[0], X.shape[1]
    ahs, iphs = [], []
    for i in range(N):
        ahs_r, iphs_r = [], []
        for j in range(C):
            imfs = emd.emd(X[i, j], max_imf=n_imf - 1)
            imfs = imfs[:n_imf] if imfs.size else X[i:i + 1, j:j + 1]
            ana = hilbert(imfs, axis=-1)
            amp = np.abs(ana)
            ph = np.angle(ana)
            ifreq = np.diff(np.unwrap(ph, axis=-1), axis=-1) / (2 * np.pi) * (sfreq or 1.0)
            ahs_r.append(amp.mean(-1))
            iphs_r.append(ifreq.mean(-1))
        k = min(len(a) for a in ahs_r)
        ahs.append(np.stack([a[:k] for a in ahs_r], axis=0))
        iphs.append(np.stack([a[:k] for a in iphs_r], axis=0))
    k = min(a.shape[-1] for a in ahs)
    return (np.stack([a[:, :k] for a in ahs]), np.stack([a[:, :k] for a in iphs]))


# ================================================================ 功能连接
def functional_connectivity(X, sfreq, method: str = "corr", fmin: float = 8.0, fmax: float = 30.0):
    """FC 功能连接矩阵。→ (N, C, C) 对称矩阵

    method:
        corr : epoch 内通道两两皮尔逊相关（快）
        plv  : 频带 [fmin,fmax] 滤波 → Hilbert 瞬时相位锁值（较快）
        coh  : Welch 相干（慢，建议限制 epoch 数）
    """
    X = np.asarray(X, dtype=np.float64)
    N, C, T = X.shape
    method = method.lower()

    if method == "corr":
        Xc = X - X.mean(-1, keepdims=True)
        cov = np.einsum("nct,nkt->nck", Xc, Xc) / max(T - 1, 1)
        sig = np.sqrt(np.einsum("nct,nct->nc", Xc, Xc) / max(T - 1, 1) + EPS)
        return np.clip(cov / (sig[..., None] * sig[..., None, :]), -1, 1)

    if method == "plv":
        xb = _band_filter(X, sfreq, fmin, fmax).astype(np.complex64)
        z = np.exp(1j * np.angle(hilbert(xb.real, axis=-1))).astype(np.complex64)
        plv = np.abs(np.matmul(z, np.conj(np.swapaxes(z, 1, 2)))) / T
        return plv

    if method == "coh":
        # 注意：Welch coherence 本身已是 0~1 的幅值平方相干，
        # 频带内做**加权平均**即可；早期版本又除以了自谱乘积（量纲不一致），
        # 会给出 1e12 这种无意义的数值。
        f, _ = welch(X, fs=sfreq, nperseg=min(256, T), axis=-1)
        m = (f >= fmin) & (f <= fmax)
        if int(m.sum()) < 2:
            m = np.ones_like(f, dtype=bool)
        band = max(float(f[m][-1] - f[m][0]), EPS)
        out = np.ones((N, C, C))
        for i in range(N):
            for a in range(C):
                for b in range(a + 1, C):
                    _, Pab = signal_coherence_pair(X[i, a], X[i, b], sfreq, min(256, T))
                    out[i, a, b] = out[i, b, a] = float(np.clip(
                        np.trapezoid(Pab[m], f[m]) / band, 0.0, 1.0))
        return out

    raise ValueError(f"未知功能连接方法：{method}（可选 corr/plv/coh）")


def signal_coherence_pair(x, y, sfreq, nperseg):
    from scipy.signal import coherence

    f, Cxy = coherence(x, y, fs=sfreq, nperseg=nperseg)
    return f, Cxy


# ================================================================ 统一入口
def extract_features(X, sfreq, names, ch_names=None, fc_method="corr",
                     use_emd=False, progress=None) -> tuple[np.ndarray, list[str]]:
    """计算并拼接多个特征。

    Parameters
    ----------
    X : (N, C, T)
    names : 特征名列表（见 FEATURES）
    ch_names : 通道名（DASM/RASM 需要）
    fc_method : corr | plv | coh
    use_emd : AH/IPH 是否用 EMD 分解（需 PyEMD，慢）
    progress : 可选回调 fn(done_ratio)，GUI 进度用

    Returns
    -------
    (F, feat_names)：F (N, D) float32；feat_names 长度 D。
    """
    X = np.asarray(X, dtype=np.float64)
    N, C = X.shape[0], X.shape[1]
    band_names = list(BANDS)
    mats, feat_names = [], []

    todo = [n.upper() for n in names]
    for k, key in enumerate(todo):
        if key == "AP":
            v = absolute_power(X, sfreq)
            feat_names += [f"AP[{b}]_ch{c}" for c in range(C) for b in band_names]
            mats.append(v.reshape(N, -1))
        elif key == "APR":
            v = theta_beta_ratio(X, sfreq)
            feat_names += [f"APb/APt_ch{c}" for c in range(C)]
            mats.append(v)
        elif key == "DE":
            v = differential_entropy(X, sfreq)
            feat_names += [f"DE[{b}]_ch{c}" for c in range(C) for b in band_names]
            mats.append(v.reshape(N, -1))
        elif key == "SHE":
            v = shannon_entropy(X)
            feat_names += [f"SHE_ch{c}" for c in range(C)]
            mats.append(v)
        elif key == "SPE":
            v = spectral_entropy(X, sfreq)
            feat_names += [f"SPE_ch{c}" for c in range(C)]
            mats.append(v)
        elif key == "C0":
            v = c0_complexity(X)
            feat_names += [f"C0_ch{c}" for c in range(C)]
            mats.append(v)
        elif key == "NSI":
            v = nsi(X)
            feat_names += [f"NSI_ch{c}" for c in range(C)]
            mats.append(v)
        elif key == "HOC":
            v = hoc(X)
            K = v.shape[-1]
            feat_names += [f"HOC{k}_ch{c}" for c in range(C) for k in range(1, K + 1)]
            mats.append(v.reshape(N, -1))
        elif key in ("DASM", "RASM"):
            if ch_names is None:
                raise ValueError(f"{key} 需要 ch_names（用于自动配对左右半球通道）")
            v, pairs = (dasm if key == "DASM" else rasm)(X, sfreq, ch_names)
            for p in range(len(pairs)):
                feat_names += [f"{key}[{b}]_pair{p}" for b in band_names]
            mats.append(v.reshape(N, -1))
        elif key == "AH_TIMF":
            v, _ = hilbert_timf(X, sfreq, use_emd=use_emd)
            K = v.shape[-1]
            feat_names += [f"AH_imf{i}_ch{c}" for c in range(C) for i in range(1, K + 1)]
            mats.append(v.reshape(N, -1))
        elif key == "IPH_TIMF":
            _, v = hilbert_timf(X, sfreq, use_emd=use_emd)
            K = v.shape[-1]
            feat_names += [f"IPH_imf{i}_ch{c}" for c in range(C) for i in range(1, K + 1)]
            mats.append(v.reshape(N, -1))
        elif key == "FC":
            v = functional_connectivity(X, sfreq, method=fc_method)
            iu = np.triu_indices(C, 1)
            feat_names += [f"FC[{fc_method}]_{i}-{j}" for i, j in zip(*iu)]
            mats.append(v[:, iu[0], iu[1]])
        else:
            raise KeyError(f"未知特征：{key}（可用：{FEATURES}）")

        if progress:
            progress((k + 1) / len(todo))

    F = np.concatenate([np.asarray(m).reshape(N, -1) for m in mats], axis=1)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return F, feat_names
