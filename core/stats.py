"""统计检验工具（向量化）：t 检验、置换检验、多重比较校正、效应量。

约定
----
数组**最后一维是「观测」**（试次 / epoch / 被试），其余维是「单元」（通道 / 频点 / 时间点）。
所有函数沿最后一维检验，返回与「单元维」同形的统计量，因此可以一次性算出
``(C,)``（逐通道）、``(C, F)``（逐通道逐频点）的全部 p 值。

示例
----
>>> t, p = ttest_ind(A, B)          # A, B: (C, N)
>>> q = correct(p, "fdr_bh")        # 多重比较校正（展平→校正→还原形状）
>>> d = cohens_d(A, B)
>>> r = group_report(A, B)          # 一步得到 UI 表格需要的全部列
"""
from __future__ import annotations

import numpy as np
from scipy import stats

EPS = 1e-12

#: 多重比较校正方法（UI 下拉用）
CORRECT_METHODS = ["none", "fdr_bh", "fdr_by", "bonferroni", "holm"]
CORRECT_LABELS = {
    "none": "不校正",
    "fdr_bh": "FDR（Benjamini-Hochberg）",
    "fdr_by": "FDR（Benjamini-Yekutieli）",
    "bonferroni": "Bonferroni",
    "holm": "Holm（逐步降阶）",
}


# ================================================================ 内部工具
def _as_matrix(a) -> tuple[np.ndarray, tuple]:
    """(..., N) → ((U, N), 单元形状)。"""
    arr = np.asarray(a, dtype=float)
    if arr.ndim < 2:
        raise ValueError(f"数据至少需要 2 维（单元 × 观测），当前 {arr.shape}")
    shape = arr.shape[:-1]
    return arr.reshape(-1, arr.shape[-1]), shape


# ================================================================ 参数检验
def ttest_ind(a, b, equal_var: bool = False):
    """独立样本 t 检验（默认 Welch，不假设方差齐性）。→ (t, p)，形状 = 单元维。"""
    A, shape = _as_matrix(a)
    B, _ = _as_matrix(b)
    t, p = stats.ttest_ind(A, B, axis=-1, equal_var=equal_var)
    return np.asarray(t).reshape(shape), np.asarray(p).reshape(shape)


def ttest_rel(a, b):
    """配对 t 检验。要求两组形状完全一致。→ (t, p)"""
    A, shape = _as_matrix(a)
    B, bshape = _as_matrix(b)
    if A.shape != B.shape:
        raise ValueError(f"配对检验要求形状一致：{a.shape} vs {b.shape}")
    t, p = stats.ttest_rel(A, B, axis=-1)
    return np.asarray(t).reshape(shape), np.asarray(p).reshape(shape)


def ttest_1samp(a, popmean: float = 0.0):
    """单样本 t 检验（如「某频段功率是否显著不为 0」）。→ (t, p)"""
    A, shape = _as_matrix(a)
    t, p = stats.ttest_1samp(A, popmean, axis=-1)
    return np.asarray(t).reshape(shape), np.asarray(p).reshape(shape)


def anova_oneway(*groups):
    """单因素方差分析（>2 组时用）。→ (F, p)，形状 = 单元维。"""
    mats = [_as_matrix(g) for g in groups]
    shape = mats[0][1]
    F, p = stats.f_oneway(*[m[0] for m in mats], axis=-1)
    return np.asarray(F).reshape(shape), np.asarray(p).reshape(shape)


# ================================================================ 效应量
def cohens_d(a, b, paired: bool = False):
    """Cohen's d（独立样本用合并标准差；配对用 dz 的等价形式）。→ 形状 = 单元维。

    零方差保护：若两组完全无差异（标准差为 0），返回 0 而不是除零产生天文数字。
    """
    A, shape = _as_matrix(a)
    B, _ = _as_matrix(b)
    if paired:
        D = A - B
        sd = D.std(axis=-1, ddof=1)
        return np.where(sd > 0, D.mean(axis=-1) / np.where(sd > 0, sd, 1.0), 0.0).reshape(shape)
    n1, n2 = A.shape[-1], B.shape[-1]
    var = (((n1 - 1) * A.var(axis=-1, ddof=1) + (n2 - 1) * B.var(axis=-1, ddof=1))
           / max(n1 + n2 - 2, 1))
    sd = np.sqrt(np.clip(var, 0.0, None))
    return np.where(sd > 0, (A.mean(axis=-1) - B.mean(axis=-1)) / np.where(sd > 0, sd, 1.0),
                    0.0).reshape(shape)


def effect_magnitude(d):
    """按 |d| 给出定性描述（Cohen 经验阈值），输入可为标量或数组。"""
    ad = np.abs(np.asarray(d, dtype=float))
    return np.where(ad < 0.2, "可忽略",
                    np.where(ad < 0.5, "小",
                             np.where(ad < 0.8, "中", "大")))


# ================================================================ 置换检验
def permutation_test(a, b=None, n_perm: int = 2000, paired: bool | None = None,
                     statistic: str = "mean_diff", random_state: int = 0):
    """置换 / 随机化检验（不依赖分布假设，适合小样本、偏态数据）。

    - ``b=None``：单样本符号翻转（检验均值是否显著非 0）
    - ``b`` 且 ``paired=True``：配对符号翻转
    - ``b`` 且 ``paired=False``：两组标签置换（要求两组样本量相同）

    Returns
    -------
    (stat, p) —— stat 为观测统计量，p 为双尾经验 p 值（含 +1 校正）。
    """
    A, shape = _as_matrix(a)
    rng = np.random.default_rng(random_state)
    N = A.shape[-1]

    if b is None:
        obs = A.mean(axis=-1)
        signs = rng.choice([-1.0, 1.0], size=(n_perm, N))
        null = np.einsum("pn,un->pu", signs, A) / N
    else:
        B, _ = _as_matrix(b)
        if paired is None:
            paired = A.shape == B.shape
        if paired:
            if A.shape != B.shape:
                raise ValueError("配对置换检验要求两组形状一致")
            D = A - B
            obs = D.mean(axis=-1)
            signs = rng.choice([-1.0, 1.0], size=(n_perm, N))
            null = np.einsum("pn,un->pu", signs, D) / N
        else:
            if A.shape[-1] != B.shape[-1]:
                raise ValueError("独立两组置换检验要求样本量相同（可先做随机下采样）")
            C = np.concatenate([A, B], axis=-1)
            n2 = C.shape[-1]
            idx = np.argsort(rng.random((n_perm, n2)), axis=1)[:, :N]
            M = np.zeros((n_perm, n2))
            np.put_along_axis(M, idx, 1.0, axis=1)
            W = M / N - (1.0 - M) / (n2 - N)
            obs = C[:, :N].mean(-1) - C[:, N:].mean(-1)
            null = np.einsum("pn,un->pu", W, C)

    if statistic == "mean_diff":
        stat = obs
    else:  # pragma: no cover - 预留其它统计量
        raise ValueError(f"未知统计量：{statistic}")

    cnt = (np.abs(null) >= np.abs(obs)[None, :] - 1e-12).sum(axis=0)
    p = (cnt + 1.0) / (n_perm + 1.0)
    return stat.reshape(shape), p.reshape(shape)


# ================================================================ 多重比较校正
def correct(p, method: str = "fdr_bh"):
    """多重比较校正。输入任意形状 p 值数组（展平后校正，再还原形状）。"""
    P = np.asarray(p, dtype=float)
    shape = P.shape
    flat = P.ravel()
    n = flat.size
    if n == 0 or method in (None, "none"):
        return P.copy()
    if method == "bonferroni":
        return np.clip(flat * n, 0.0, 1.0).reshape(shape)

    order = np.argsort(flat)
    ranked = flat[order]
    k = np.arange(1, n + 1)
    if method == "holm":
        adj = np.maximum.accumulate((n - k + 1) * ranked)
    elif method in ("fdr_bh", "fdr_by"):
        factor = n / k
        if method == "fdr_by":                      # 更保守，允许任意相关结构
            factor = factor * np.sum(1.0 / k)
        adj = np.minimum.accumulate((ranked * factor)[::-1])[::-1]
    else:
        raise ValueError(f"未知校正方法：{method}（可用：{CORRECT_METHODS}）")
    out = np.empty(n)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out.reshape(shape)


def stars(p) -> np.ndarray:
    """p 值 → 显著性星号（ns / * / ** / ***）。"""
    P = np.asarray(p, dtype=float)
    return np.where(P < 0.001, "***", np.where(P < 0.01, "**",
                    np.where(P < 0.05, "*", "ns")))


# ================================================================ 一步式报告
def group_report(a, b=None, paired: bool = True, correct_method: str = "fdr_bh",
                 n_perm: int = 0, random_state: int = 0) -> dict:
    """条件间（或单条件 vs 0）的逐单元统计报告，直接喂 UI 表格。

    Parameters
    ----------
    a, b : (..., N) 数组。b=None 时做单样本 t 检验（vs 0）
    n_perm : >0 时另外做置换检验，返回 perm_p

    Returns
    -------
    dict，键均为与单元维同形的数组：mean_a, std_a, mean_b, std_b, t, p,
    p_corrected, d, n_a, n_b, n_perm(可选), perm_p(可选)
    """
    A, shape = _as_matrix(a)
    out: dict[str, np.ndarray] = {}
    out["mean_a"] = A.mean(axis=-1).reshape(shape)
    out["std_a"] = A.std(axis=-1, ddof=1).reshape(shape)
    out["n_a"] = np.full(shape, A.shape[-1], dtype=int)

    if b is None:
        t, p = ttest_1samp(a, 0.0)
        out["mean_b"] = np.zeros(shape)
        out["std_b"] = np.zeros(shape)
        out["n_b"] = np.zeros(shape, dtype=int)
        out["d"] = (out["mean_a"] / (out["std_a"] + EPS))
    else:
        B, _ = _as_matrix(b)
        out["mean_b"] = B.mean(axis=-1).reshape(shape)
        out["std_b"] = B.std(axis=-1, ddof=1).reshape(shape)
        out["n_b"] = np.full(shape, B.shape[-1], dtype=int)
        t, p = ttest_rel(a, b) if paired else ttest_ind(a, b)
        out["d"] = cohens_d(a, b, paired=paired)

    out["t"] = t
    out["p"] = p
    out["p_corrected"] = correct(p, correct_method)
    if n_perm and int(n_perm) > 0:
        _, pp = permutation_test(a, b, n_perm=int(n_perm), paired=paired,
                                 random_state=random_state)
        out["perm_p"] = pp
        out["perm_p_corrected"] = correct(pp, correct_method)
    return out


def report_rows(names, rep: dict, unit_label: str = "单元",
                fmt_effect: bool = True) -> list[list[str]]:
    """把 group_report 的结果转成 UI 表格行（字符串）。"""
    keys = ("mean_a", "std_a", "mean_b", "std_b", "t", "p", "p_corrected", "d")
    rows = []
    for i, name in enumerate(names):
        row = [str(name)]
        for k in keys:
            if k not in rep:
                continue
            v = np.asarray(rep[k]).reshape(-1)[i]
            if k in ("p", "p_corrected") or k.startswith("perm"):
                row.append(f"{v:.4g}{' ' + stars(v) if k != 't' else ''}")
            else:
                row.append(f"{v:.3g}")
        if fmt_effect and "d" in rep:
            row.append(str(effect_magnitude(np.asarray(rep["d"]).reshape(-1)[i])))
        rows.append(row)
    return rows


REPORT_HEADERS = ["单元", "均值[A]", "SD[A]", "均值[B]", "SD[B]", "t", "p", "p(校正)", "Cohen's d", "效应"]
