"""脑网络：连接指标与图属性。

**连接方法（12 种）**

时间序列类（EEG / BOLD ROI 时序通用，输入 (N,C,T) 或 (C,T)）::

    corr         Pearson 相关（epoch 内逐 trial，→ (N,C,C)）
    spearman     Spearman 秩相关（跨 epoch 拼接，→ (C,C)）
    partial      偏相关（协方差求逆 → -Pij/√(Pii·Pjj)，控制其余通道）
    mi           互信息（直方图估计 → (C,C)）
    coh          Welch 相干（→ (N,C,C)）
    plv          相位锁值（→ (N,C,C)）
    pli          相位滞后指数（→ (N,C,C)）
    wcoherence   小波相干（Morlet CWT，频带内平均 → (C,C)）
    hofc         高阶功能连接（边×边的边序列相关 → (E,E)，E=C(C-1)/2）
    sparse_repr  稀疏表示（L1 回归：每个通道用其余通道稀疏重构 → (C,C)）
    low_rank     低秩表示（LRR，ADMM 求解 Z，|Z| 对称化 → (C,C)）
    granger      格兰杰因果（statsmodels F 检验，-log10(p) → (C,C)）

**稀疏化策略（5 种）**，见 :func:`sparsify`：``none`` / ``proportional`` / ``absolute`` /
``adaptive`` / ``mst``。
"""
from __future__ import annotations

import numpy as np

from .features import _band_filter, functional_connectivity

#: 下拉框用：时间序列类连接方法
CONNECT_METHODS = ["corr", "spearman", "partial", "mi", "coh", "plv", "pli",
                   "wcoherence", "hofc", "sparse_repr", "low_rank", "granger"]

#: 方法中文名（GUI 显示）
CONNECT_LABELS = {
    "corr": "Pearson 相关",
    "spearman": "Spearman 秩相关",
    "partial": "偏相关（控制其余通道）",
    "mi": "互信息",
    "coh": "Welch 相干",
    "plv": "相位锁值 PLV",
    "pli": "相位滞后指数 PLI",
    "wcoherence": "小波相干（Morlet）",
    "hofc": "高阶功能连接 HOFC（边×边）",
    "sparse_repr": "稀疏表示（L1 回归）",
    "low_rank": "低秩表示 LRR",
    "granger": "格兰杰因果 GCA",
}

#: 稀疏化策略与中文名
SPARSIFY_STRATEGIES = ["none", "proportional", "absolute", "adaptive", "mst"]
SPARSIFY_LABELS = {
    "none": "不稀疏化（保留全部权重）",
    "proportional": "比例阈值（保留最强 x% 的边）",
    "absolute": "绝对阈值（|权重| ≥ 阈值）",
    "adaptive": "自适应阈值（每个节点按自身强度分位数）",
    "mst": "最小生成树骨架（每节点至少一枝，保留 C−1 条最强边）",
}


def _as_nct(X) -> np.ndarray:
    """(C,T) 或 (N,C,T) → 至少三维 (N,C,T)。"""
    X = np.asarray(X, dtype=np.float64)
    return X[None] if X.ndim == 2 else X


def _flat(X, max_samples: int | None = None) -> np.ndarray:
    """跨 epoch 拼接成 (C, N*T)（可选截断，控制耗时）。"""
    X = _as_nct(X)
    data = X.transpose(1, 0, 2).reshape(X.shape[1], -1)
    if max_samples and data.shape[1] > max_samples:
        data = data[:, :max_samples]
    return data


def _symmetrize(A, diag: float = 0.0) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, diag)
    return A


def pli_matrix(X, sfreq, fmin: float = 8.0, fmax: float = 30.0):
    """相位滞后指数 PLI。X (N,C,T) → (N,C,C)（对角=1）；(C,T) → (C,C)。"""
    from scipy.signal import hilbert

    X = np.asarray(X, dtype=np.float64)
    squeeze = X.ndim == 2            # 单段 (C,T)：按单 epoch 处理并降回二维
    X = _as_nct(X)
    xb = _band_filter(X, sfreq, fmin, fmax)
    ph = np.angle(hilbert(xb, axis=-1))
    N, C = ph.shape[0], ph.shape[1]
    pli = np.ones((N, C, C))
    for a in range(C):
        for b in range(a + 1, C):
            dph = ph[:, a] - ph[:, b]
            pli[:, a, b] = pli[:, b, a] = np.abs(np.sign(np.sin(dph)).mean(axis=-1))
    return pli[0] if squeeze else pli


def mi_matrix(X, bins: int = 16):
    """互信息（跨 epoch 拼接后逐通道对计算）。X (N,C,T) 或 (C,T) → (C,C)，对角=0。"""
    data = _flat(X)                             # (C, N*T)，兼容二维输入
    C = data.shape[0]
    mi = np.zeros((C, C))
    for a in range(C):
        for b in range(a + 1, C):
            h, _, _ = np.histogram2d(data[a], data[b], bins=bins)
            p = h / h.sum()
            px = p.sum(axis=1, keepdims=True)
            py = p.sum(axis=0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = p / (px * py)
                ratio = np.where((p > 0), ratio, 0.0)
            mi[a, b] = mi[b, a] = float((p * np.log(ratio + 1e-30)).sum())
    return mi


def granger_matrix(X, sfreq, maxlag: int = 3, max_samples: int = 30000):
    """格兰杰因果（statsmodels，F 检验）。X (N,C,T) → (C,C)，G[i,j] = j→i 的 -log10(p)。"""
    import contextlib
    import io
    import warnings

    from statsmodels.tsa.stattools import grangercausalitytests

    C = X.shape[1]
    x = X.transpose(1, 0, 2).reshape(C, -1)[:, :max_samples]
    G = np.zeros((C, C))
    for a in range(C):
        for b in range(C):
            if a == b:
                continue
            data = np.column_stack([x[a], x[b]])
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = grangercausalitytests(data, maxlag=maxlag)
            p = min(res[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1))
            G[a, b] = float(-np.log10(max(p, 1e-30)))  # b → a 强度
    return G


def spearman_matrix(X, max_samples: int | None = None):
    """Spearman 秩相关：先做秩变换再算 Pearson。X (N,C,T) → (C,C)，对角=1。"""
    from scipy.stats import rankdata

    data = _flat(X, max_samples)
    ranked = np.apply_along_axis(rankdata, 1, data)
    C = ranked.shape[0]
    out = np.ones((C, C))
    for a in range(C):
        for b in range(a + 1, C):
            out[a, b] = out[b, a] = float(np.corrcoef(ranked[a], ranked[b])[0, 1])
    return np.nan_to_num(out, nan=0.0)


def partial_corr_matrix(X, alpha: float = 1e-3, max_samples: int | None = None):
    """偏相关：协方差矩阵求逆（精度矩阵）→ ``-Pij/√(Pii·Pjj)``。

    ``alpha`` 为对角岭正则，保证通道数较多时数值稳定（不会奇异）。
    X (N,C,T) → (C,C)，对角=0。
    """
    data = _flat(X, max_samples)
    C = data.shape[0]
    cov = np.cov(data)
    cov = np.atleast_2d(cov)
    cov += np.eye(C) * (alpha * np.trace(cov) / max(C, 1))
    prec = np.linalg.pinv(cov)
    d = np.sqrt(np.clip(np.diag(prec), 1e-12, None))
    pcorr = -prec / np.outer(d, d)
    return _symmetrize(np.clip(pcorr, -1, 1))


def wavelet_coherence_matrix(X, sfreq, fmin: float = 8.0, fmax: float = 30.0,
                             n_freqs: int = 12, smooth_s: float = 0.5,
                             max_samples: int = 8000):
    """小波相干（Morlet 连续小波变换 → 频带内幅值平方相干）。

    手写 CWT（scipy ≥1.15 移除了 ``cwt``/``morlet2``），只依赖 numpy。
    X (N,C,T) → (C,C)，取值 0~1（越大越同步），对角=1。

    .. note:: 与 Welch 相干一样存在**小样本基线偏差**：完全独立的通道也会拿到
       ~0.2 上下的相干值（数据越短越高）。所以看**相对大小**（哪对更高），
       不要拿绝对值跟 0 比。
    """
    data = _flat(X, max_samples)                      # (C, T')
    # 逐通道标准化：小波相干本身与幅值无关，但**归一化能避免绝对 epsilon
    # 在小幅值（真实 EEG 是 1e-5 V 量级）时把相干值压成 0**。
    sd = data.std(axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    data = (data - data.mean(axis=1, keepdims=True)) / sd
    C, T = data.shape
    freqs = np.geomspace(max(fmin, 0.5), max(fmax, fmin * 1.2), int(max(n_freqs, 3)))
    W = np.empty((C, len(freqs), T), dtype=complex)
    n_fft = 1 << int(np.ceil(np.log2(T + 1024)))
    for ch in range(C):
        Xf = np.fft.fft(data[ch] - data[ch].mean(), n_fft)
        for k, f in enumerate(freqs):
            scale = 6.0 * sfreq / (2 * np.pi * f)     # Morlet ω0=6
            half = max(int(4 * scale), 4)
            t = np.arange(-half, half + 1)
            psi = ((np.pi ** -0.25) * np.exp(1j * 6.0 * t / scale)
                   * np.exp(-(t ** 2) / (2 * scale ** 2)) / np.sqrt(scale))
            conv = np.fft.ifft(Xf * np.conj(np.fft.fft(psi, n_fft)))
            W[ch, k] = conv[half:half + T]            # 取与输入对齐的中心段

    # 时间维平滑（移动平均 = 频域乘积，快）
    win = max(int(smooth_s * sfreq), 3)
    kern = np.ones(win) / win
    def smooth(a):
        return np.apply_along_axis(
            lambda v: np.convolve(v, kern, mode="same"), -1, a)

    power = smooth(np.abs(W) ** 2)                    # (C, F, T)
    floor = 1e-12 * max(float(np.max(power)) ** 2, 1e-30)   # 相对 epsilon（尺度无关）
    out = np.ones((C, C))
    for a in range(C):
        for b in range(a + 1, C):
            cross = smooth(W[a] * np.conj(W[b]))
            num = np.abs(cross) ** 2
            den = power[a] * power[b] + floor
            out[a, b] = out[b, a] = float(np.clip(np.mean(num / den), 0.0, 1.0))
    return out


def hofc_matrix(X, max_samples: int = 4000, win_s: float | None = None, sfreq: float = 250.0):
    """高阶功能连接 HOFC：**边 × 边**（E=C(C−1)/2）。

    做法：把数据切成若干窗，每窗算一个 FC（低阶），得到「边的时序」；
    再对边与边之间求相关 → (E,E)。它描述"连接之间的连接"，
    所以不能直接当节点图用（图属性会按边图计算，界面上会提示）。
    """
    X = _as_nct(X)
    N, C, T = X.shape
    data = X.reshape(-1, C, T) if N > 1 else X

    def _windows(win_s_: float):
        step = max(int(0.25 * sfreq), 1)
        win = max(int(win_s_ * sfreq), 16)
        return [data[0][:, s:s + win] for s in range(0, max(T - win + 1, 1), step)]

    if win_s:
        segs = _windows(float(win_s))
    else:
        segs = [ep for ep in data]
        if len(segs) < 3:                     # 单条长时序 → 自动按 2 s 窗滑窗
            segs = _windows(2.0)
    if len(segs) < 3:
        raise ValueError(f"高阶功能连接至少需要 3 个窗口/试次（当前 {len(segs)}，"
                         f"数据长 {T} 点：请加长数据或减小窗口）")

    iu = np.triu_indices(C, 1)
    edge_series = np.stack([np.corrcoef(seg)[iu] for seg in segs])    # (W, E)
    E = edge_series.shape[1]
    out = np.eye(E)
    for i in range(E):
        for j in range(i + 1, E):
            v = float(np.corrcoef(edge_series[:, i], edge_series[:, j])[0, 1])
            out[i, j] = out[j, i] = 0.0 if not np.isfinite(v) else v
    return out


def sparse_repr_matrix(X, alpha: float = 0.01, max_samples: int = 20000):
    """稀疏表示：每个通道用其余通道做 **L1 正则回归** 稀疏重构。

    ``A[i,j]`` = 用通道 j 重构通道 i 时的系数绝对值（越大表示 j 对 i 越"基"）。
    X (N,C,T) → (C,C)，对角=0。alpha 越大越稀疏。

    实现要点：先把每个通道 **z-score 标准化**再回归——Lasso 的目标函数按样本数归一，
    直接喂 1e-5 量级的 EEG 电压会让惩罚项压倒数据项，系数全变 0。
    """
    from sklearn.linear_model import Lasso

    data = _flat(X, max_samples)
    C = data.shape[0]
    scale = data.std(axis=1, keepdims=True)
    scale[scale < 1e-12] = 1.0
    Z = (data - data.mean(axis=1, keepdims=True)) / scale    # (C, T) 标准化
    out = np.zeros((C, C))
    for i in range(C):
        others = [j for j in range(C) if j != i]
        model = Lasso(alpha=float(alpha), fit_intercept=False, max_iter=5000)
        model.fit(Z[others].T, Z[i])                          # (T, C-1) → (T,)
        out[i, others] = np.abs(model.coef_)
    return _symmetrize(out)


def low_rank_matrix(X, lam: float = 0.05, max_iter: int = 60,
                    max_samples: int = 20000):
    """低秩表示 LRR：``min ||Z||_* + λ||E||_1  s.t. X = XZ + E``（LADMAP-ADMM）。

    数据矩阵 ``X`` 取 **(T, C)**（每一列是一个通道的整段时间序列），因此 ``Z`` 是
    ``(C, C)``——"每个通道能被其它通道的低秩子空间表示多少"。用 ``|Z|`` 对称化作为
    亲和矩阵（低秩表示偏"全局子空间"，比稀疏表示更抗噪）。
    X (N,C,T) → (C,C)，对角=0。
    """
    data = _flat(X, max_samples)                        # (C, T)
    D = data.T                                          # (T, C) 每列一个通道
    norms = np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
    Xn = D / norms                                      # 列归一化（量纲无关）
    T, C = Xn.shape
    lip = max(float(np.linalg.norm(Xn, 2)) ** 2, 1e-8)  # LADMAP 的 Lipschitz 常数
    Z = np.zeros((C, C))
    E = np.zeros_like(Xn)
    Y = np.zeros_like(Xn)
    mu, rho, max_mu = 1.0, 1.5, 1e8
    prev_norm = None
    for _ in range(int(max_iter)):
        # Z 更新：线性化 + 奇异值阈值（LADMAP）
        Z = _svt(Z + (Xn.T @ (Xn - Xn @ Z - E + Y / mu)) / (mu * lip), 1.0 / (mu * lip))
        XZ = Xn @ Z
        # E 更新：软阈值
        E = _shrink(Xn - XZ + Y / mu, lam / mu)
        res = Xn - XZ - E
        Y = Y + mu * res
        err = float(np.linalg.norm(res)) / max(float(np.linalg.norm(Xn)), 1e-12)
        mu = min(mu * rho, max_mu)
        if err < 1e-6 or (prev_norm is not None and abs(prev_norm - err) < 1e-9):
            break
        prev_norm = err
    return _symmetrize(np.abs(Z))


def _svt(M, thresh: float) -> np.ndarray:
    """奇异值阈值算子（Soft-threshold on singular values）。"""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return (U * np.clip(s - thresh, 0, None)) @ Vt


def _shrink(M, thresh: float) -> np.ndarray:
    """软阈值（L1 近端算子）。"""
    return np.sign(M) * np.clip(np.abs(M) - thresh, 0, None)


def connectivity_matrix(X, sfreq, method: str = "corr", **kw):
    """统一入口。

    - 逐 trial 的：``corr`` / ``coh`` / ``plv`` / ``pli`` → (N,C,C)
    - 跨 trial 汇总：``spearman`` / ``partial`` / ``mi`` / ``wcoherence`` /
      ``sparse_repr`` / ``low_rank`` / ``granger`` → (C,C)
    - ``hofc`` → (E,E)，E=C(C−1)/2（边图）

    ``X`` 同时接受 (N,C,T) 与单段 (C,T)：单段输入时逐 trial 方法返回 (C,C) 而不是
    (1,C,C)（入口统一升维、出口按需降维，避免各子函数各写一套）。
    """
    method = method.lower()
    X = np.asarray(X, dtype=np.float64)
    squeeze = X.ndim == 2                      # 单段输入 → 结果降回二维
    X = _as_nct(X)
    fmin = float(kw.get("fmin", 8.0))
    fmax = float(kw.get("fmax", 30.0))
    if method in ("corr", "plv", "coh"):
        out = functional_connectivity(X, sfreq, method=method, fmin=fmin, fmax=fmax)
    elif method == "pli":
        out = pli_matrix(X, sfreq, fmin=fmin, fmax=fmax)
    elif method == "mi":
        out = mi_matrix(X, bins=int(kw.get("bins", 16)))
    elif method == "granger":
        out = granger_matrix(X, sfreq, maxlag=int(kw.get("maxlag", 3)),
                             max_samples=int(kw.get("max_samples", 30000)))
    elif method == "spearman":
        out = spearman_matrix(X, max_samples=kw.get("max_samples"))
    elif method in ("partial", "partial_corr"):
        out = partial_corr_matrix(X, alpha=float(kw.get("alpha", 1e-3)))
    elif method in ("wcoherence", "wcoh", "wavelet"):
        out = wavelet_coherence_matrix(X, sfreq, fmin=fmin, fmax=fmax,
                                       n_freqs=int(kw.get("n_freqs", 12)))
    elif method in ("hofc", "high_order"):
        out = hofc_matrix(X, sfreq=float(sfreq),
                          win_s=kw.get("win_s"), max_samples=int(kw.get("max_samples", 4000)))
    elif method in ("sparse_repr", "sparse"):
        out = sparse_repr_matrix(X, alpha=float(kw.get("alpha", 0.01)))
    elif method in ("low_rank", "lrr"):
        out = low_rank_matrix(X, lam=float(kw.get("lam", 0.05)))
    else:
        raise ValueError(f"未知连接方法：{method}（可选 {CONNECT_METHODS}）")

    out = np.asarray(out)
    if squeeze and out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
    return out


# ---------------------------------------------------------------- 稀疏化
def sparsify(adj, strategy: str = "proportional", density: float = 0.2,
             threshold: float | None = None, percentile: float = 80.0,
             symmetrize: bool = True):
    """网络稀疏化：只保留能代表神经过程的强连接，其余置 0。

    Parameters
    ----------
    strategy : none / proportional / absolute / adaptive / mst
        - ``none``         只对称化 + 去对角
        - ``proportional`` 保留最强 ``density`` 比例的边（等价旧 ``binarize`` 的阈值）
        - ``absolute``     ``|w| ≥ threshold``（未给阈值时用 ``percentile`` 分位数）
        - ``adaptive``     每个节点按**自身**连接强度的分位数保留 → 强弱节点一视同仁
        - ``mst``          最大生成树骨架（C−1 条最强边，保证全图连通，便于算路径长度）
    symmetrize : 是否先对称化（图属性要求无向；有向矩阵如 GCA 会被对称化）

    Returns
    -------
    ndarray (C,C)：**加权**稀疏矩阵（被剔除的边为 0）；要二值图用 :func:`binarize`。
    """
    A = np.asarray(adj, dtype=float)
    if symmetrize:
        A = (A + A.T) / 2.0
    A = A.copy()
    np.fill_diagonal(A, 0.0)
    C = A.shape[0]
    if C < 2:
        return A
    iu = np.triu_indices(C, 1)
    strategy = (strategy or "proportional").lower()

    if strategy == "none":
        keep = A != 0.0
    elif strategy == "proportional":
        tri = np.abs(A[iu])
        k = max(1, int(len(tri) * float(np.clip(density, 0.01, 1.0))))
        thr = np.partition(tri, len(tri) - k)[len(tri) - k]
        keep = np.abs(A) >= thr
    elif strategy == "absolute":
        if threshold is None:
            threshold = float(np.percentile(np.abs(A[iu]), percentile))
        keep = np.abs(A) >= float(threshold)
    elif strategy == "adaptive":
        keep = np.zeros_like(A, dtype=bool)
        for i in range(C):
            row = np.abs(A[i])
            nz = row[row > 0]
            if nz.size == 0:
                continue
            keep[i] = row >= float(np.percentile(nz, percentile))
        keep = keep | keep.T
    elif strategy == "mst":
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree

        neg = csr_matrix(-np.abs(A) - 1e-12)         # 取负 → 最小生成树 = 最大生成树
        m = minimum_spanning_tree(neg).toarray()
        keep = (m != 0) | (m.T != 0)
    else:
        raise ValueError(f"未知稀疏化策略：{strategy}（可选 {SPARSIFY_STRATEGIES}）")

    np.fill_diagonal(keep, False)
    return np.where(keep, A, 0.0)


# ---------------------------------------------------------------- 图属性
def binarize(adj, density: float = 0.2, strategy: str = "proportional", **kw):
    """对称化 + 稀疏化 → 二值邻接矩阵 (C,C)（图属性的输入）。"""
    W = sparsify(adj, strategy=strategy, density=density, **kw)
    return (np.abs(W) > 0).astype(int)


def graph_metrics(adj_binary):
    """图属性：节点度、聚类系数、全局/局部效率、平均最短路径长度。"""
    import networkx as nx

    G = nx.from_numpy_array(np.asarray(adj_binary))
    deg = np.array([d for _, d in G.degree()], dtype=float)
    clus = np.array([nx.clustering(G, n) for n in G.nodes()], dtype=float)
    ge = float(nx.global_efficiency(G))

    def _node_local_eff(n):
        nb = list(G[n])
        if len(nb) < 2:
            return 0.0
        return float(nx.global_efficiency(G.subgraph(nb)))

    le = np.array([_node_local_eff(n) for n in G.nodes()], dtype=float)
    if nx.is_connected(G):
        apl = float(nx.average_shortest_path_length(G))
    else:
        comps = [G.subgraph(c) for c in nx.connected_components(G)]
        lens = [nx.average_shortest_path_length(c) for c in comps if len(c) > 1]
        apl = float(np.mean(lens)) if lens else float("nan")
    return {
        "degree": deg,
        "clustering": clus,
        "local_efficiency": le,
        "global_efficiency": ge,
        "avg_path_length": apl,
    }
