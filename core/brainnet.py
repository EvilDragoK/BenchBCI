"""脑网络构建、特征提取与生物标志物挖掘（对齐 BrainNet Studio 流程）。

对应 BrainNet Studio（arXiv:2609.37956）六步工作流中的后四步：
    静态/动态网络构建 → 特征提取 → 可视化 → 候选生物标志物识别
本模块为纯计算层（不依赖 PyQt），可视化在 ui/widgets/brain3d_dialog.py。
"""
from __future__ import annotations

import numpy as np

from .networks import binarize, connectivity_matrix, graph_metrics


def static_connectivity(X, sfreq, method: str = "corr", **kw):
    """静态网络：全时间段/跨 trial 的连接矩阵（(N,C,C) 或 (C,C)）。"""
    return connectivity_matrix(X, sfreq, method=method, **kw)


def dynamic_connectivity(X, sfreq, method: str = "corr", win_s: float = 1.0,
                         step_s: float = 0.25, fmin: float = 8.0, fmax: float = 30.0,
                         max_windows: int = 200):
    """动态网络：滑窗构建连接矩阵序列。

    X (N,C,T) 分段数据；对每个时间窗内的数据计算连接矩阵并跨 trial 平均。

    Returns
    -------
    (times (W,), adj (W,C,C))：窗口中心时刻（秒）与各窗口连接矩阵。
    """
    N, C, T = X.shape
    win = max(int(win_s * sfreq), 8)
    step = max(int(step_s * sfreq), 1)
    starts = list(range(0, max(T - win + 1, 1), step))[:max_windows]
    times, mats = [], []
    for s in starts:
        seg = X[:, :, s:s + win]
        if seg.shape[-1] < 8:
            continue
        A = connectivity_matrix(seg, sfreq, method=method, fmin=fmin, fmax=fmax)
        if A.ndim == 3:
            A = A.mean(axis=0)
        A = np.asarray(A, dtype=float).copy()
        np.fill_diagonal(A, 0.0)
        mats.append(A)
        times.append((s + win / 2.0) / sfreq)
    if not mats:
        raise ValueError("时间窗太宽，无法构建动态网络（请减小窗长）")
    return np.asarray(times), np.stack(mats)


def dynamic_features(adj_seq, density: float = 0.2):
    """动态网络特征：逐窗口图属性序列 + 节点强度序列。

    Returns
    -------
    {global_efficiency (W,), clustering (W,), avg_path_length (W,), strength (W,C)}
    """
    ge, cl, pl, st = [], [], [], []
    for A in adj_seq:
        gm = graph_metrics(binarize(A, density))
        ge.append(gm["global_efficiency"])
        cl.append(float(gm["clustering"].mean()))
        pl.append(gm["avg_path_length"])
        st.append(np.abs(A).sum(axis=1))
    return {
        "global_efficiency": np.asarray(ge),
        "clustering": np.asarray(cl),
        "avg_path_length": np.asarray(pl),
        "strength": np.asarray(st),
    }


def connection_variability(adj_seq):
    """连接的时间变异性（动态性指标）：跨窗口标准差 → (C,C)。"""
    return np.asarray(adj_seq).std(axis=0)


def node_importance(adj, metric: str = "strength"):
    """节点重要性：strength=连接强度之和；degree=二值化后的度。"""
    A = np.asarray(adj, dtype=float).copy()
    np.fill_diagonal(A, 0.0)
    if metric == "degree":
        return binarize(A, 0.2).sum(axis=1).astype(float)
    return np.abs(A).sum(axis=1)


def biomarkers_by_condition(X, labels, sfreq, method: str = "corr",
                            fmin: float = 8.0, fmax: float = 30.0,
                            top_n: int = 20, correct_method: str = "fdr_bh"):
    """候选生物标志物挖掘：按条件比较连接，找出判别性脑连接与脑区。

    对每对通道 (i,j)，用各条件的逐 trial 连接值做 Welch t 检验（one-vs-rest 取最强）。
    **p 值做多重比较校正**（默认 FDR-BH；检验次数 = C(C−1)/2 条边），
    ``top_connections`` 里给出的 p 是**校正后** p（原始 p 见 ``p_val`` 矩阵）。

    Returns
    -------
    {conds, A_by_cond {cond:(C,C)}, t_stat (C,C), p_val (C,C), p_corrected (C,C),
     n_significant, top_connections [(i,j,t,p_corrected,mean_c0,mean_c1)], node_score (C,)}
    """
    from scipy import stats

    from .stats import correct

    labels = np.asarray(labels)
    conds = sorted(set(labels.tolist()))
    if len(conds) < 2:
        raise ValueError(f"生物标志物挖掘需要至少 2 个条件（当前 {len(conds)} 个）")

    A_all = connectivity_matrix(X, sfreq, method=method, fmin=fmin, fmax=fmax)
    if A_all.ndim == 2:
        raise ValueError(f"{method} 不支持逐 trial 连接计算，请改用 corr/plv/coh/pli")
    A_all = np.asarray(A_all, dtype=float)
    idx_by_cond = {c: np.where(labels == c)[0] for c in conds}
    A_by_cond = {c: A_all[idx_by_cond[c]].mean(axis=0) for c in conds}

    C = A_all.shape[1]
    t_mat = np.zeros((C, C))
    p_mat = np.ones((C, C))
    all_idx = np.arange(len(labels))
    pairs = [(i, j) for i in range(C) for j in range(i + 1, C)]
    raw_p = np.ones(len(pairs))
    for k, (i, j) in enumerate(pairs):
        best_t, best_p = 0.0, 1.0
        for c in conds:
            m = idx_by_cond[c]
            rest = np.setdiff1d(all_idx, m)
            a, b = A_all[m, i, j], A_all[rest, i, j]
            if len(a) < 2 or len(b) < 2:
                continue
            t, p = stats.ttest_ind(a, b, equal_var=False)
            if np.isfinite(p) and p < best_p:
                best_t, best_p = float(t), float(p)
        t_mat[i, j] = t_mat[j, i] = best_t
        p_mat[i, j] = p_mat[j, i] = best_p
        raw_p[k] = best_p

    # 多重比较校正：不校正时，C=22 → 231 条边里纯随机也常冒出 p<0.05
    corr_p = correct(raw_p, correct_method)
    pcm = np.ones((C, C))
    for k, (i, j) in enumerate(pairs):
        pcm[i, j] = pcm[j, i] = float(corr_p[k])

    # 排序：先按校正后 p 升序，再按 |t| 降序（p 相同时以效应量排）
    tri = sorted(((i, j, float(corr_p[k]), float(t_mat[i, j]))
                  for k, (i, j) in enumerate(pairs)),
                 key=lambda x: (x[2], -abs(x[3])))
    top = [(i, j, t, pc, float(A_by_cond[conds[0]][i, j]), float(A_by_cond[conds[1]][i, j]))
           for i, j, pc, t in tri[:top_n]]
    return {
        "conds": conds,
        "A_by_cond": A_by_cond,
        "t_stat": t_mat,
        "p_val": p_mat,                       # 原始 p
        "p_corrected": pcm,                   # 多重比较校正后 p
        "n_significant": int((corr_p < 0.05).sum()),
        "correct_method": correct_method,
        "top_connections": top,
        "node_score": np.abs(t_mat).sum(axis=1),
    }
