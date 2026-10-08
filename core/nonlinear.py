"""非线性分析：样本熵、近似熵、排列熵。"""
from __future__ import annotations

import numpy as np


def _templates(x, m):
    from numpy.lib.stride_tricks import sliding_window_view

    return sliding_window_view(x, m)


def _tri_count(rows, r):
    """上三角（不含自匹配）中 Chebyshev 距离 < r 的模板对数。"""
    from scipy.spatial.distance import pdist

    return int((pdist(rows, metric="chebyshev") < r).sum())


def _phi(rows, r):
    """近似熵的 φ 统计量（含自匹配）。"""
    from scipy.spatial.distance import pdist, squareform

    n = len(rows)
    D = squareform(pdist(rows, metric="chebyshev"))  # (n,n)，对角 0
    c = (D < r).sum(axis=1)  # 含自身（对角 0 < r）
    return float(np.log(np.clip(c, 1, None) / n).mean())


def sample_entropy(x, m: int = 2, r: float | None = None):
    """样本熵 SampEn（Richman & Moorman 2000，不含自匹配）。r 默认 0.2 × 标准差。

    要点（原来实现的两处偏差已修）：
    - B 与 A 必须基于**同一批** i=1..N-m 模板（原来 B 用了 N-m+1 个，结果系统性偏高）；
    - A=0（没有任何 m+1 匹配对）时真值是 +∞，原实现返回 0.0（熵的最小值）方向相反；
      这里取理论上界 ``log(B)``（等价于"恰好只有 1 对 m+1 匹配"），既保守又便于聚合。
    信号过短返回 ``nan``（由调用方按 nan 处理，不再静默当成 0）。
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    N = len(x)
    if r is None:
        r = 0.2 * x.std()
    if r <= 0 or N < m + 2:
        return 0.0 if r <= 0 else float("nan")
    B = _tri_count(_templates(x, m)[: N - m], r)      # 与 m+1 同源：前 N-m 个模板
    A = _tri_count(_templates(x, m + 1), r)
    if B == 0:
        return float("nan")
    return float(-np.log(max(A, 1) / B))


def approximate_entropy(x, m: int = 2, r: float | None = None):
    """近似熵（含自匹配，Pincus 1991）。"""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    N = len(x)
    if r is None:
        r = 0.2 * x.std()
    if r <= 0 or N < m + 1:
        return 0.0
    return float(_phi(_templates(x, m), r) - _phi(_templates(x, m + 1), r))


def permutation_entropy(x, order: int = 3, delay: int = 1):
    """排列熵（Bandt & Pompe 2002）。"""
    from itertools import permutations

    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < order * delay:
        return 0.0
    perms = list(permutations(range(order)))
    lookup = {p: i for i, p in enumerate(perms)}
    counts = np.zeros(len(perms))
    for t in range(N - (order - 1) * delay):
        idx = np.argsort(x[t:t + order * delay:delay])
        counts[lookup[tuple(idx)]] += 1
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(perms)))


def entropy_channels(X, methods=("sample", "approximate", "permutation"),
                     m: int = 2, order: int = 3):
    """逐通道计算熵。X (N,C,T) → dict {method: (C,) 跨 epoch 平均}。"""
    X = np.asarray(X, dtype=np.float64)
    N, C, T = X.shape
    out = {}
    for method in methods:
        vals = np.zeros(C)
        for c in range(C):
            seg = []
            for n in range(N):
                if method == "sample":
                    seg.append(sample_entropy(X[n, c], m=m))
                elif method == "approximate":
                    seg.append(approximate_entropy(X[n, c], m=m))
                else:
                    seg.append(permutation_entropy(X[n, c], order=order))
            seg = np.asarray(seg, dtype=float)
            finite = np.isfinite(seg)
            # 过短/退化 epoch 返回 nan，这里按有效值平均（全为 nan 时记 0）
            vals[c] = float(seg[finite].mean()) if finite.any() else 0.0
        out[method] = vals
    return out
