"""变量关联分析（Linked Heatmap）：随机森林重要性 + Pearson 相关 + 线性混合效应模型。

面向**表格型**数据（行 = 观测/被试，列 = 变量，另指定一个目标变量），一张图回答三件事：

1. **RF 重要性**（顶部柱条）—— 哪些变量对目标变量的预测更重要
   （置换重要性：打乱第 j 个变量后模型误差增加多少）
2. **Pearson 相关**（中间三角矩阵）—— 变量两两之间如何共同变化
   （颜色 = 方向，方块边长与颜色深浅 = |r|；**相关性不是因果**）
3. **LMM 效应**（底部弧线）—— 控制"同一被试/同一地点重复观测"这类层级结构后，
   哪些变量仍与目标变量显著正/负相关
   （颜色 = 方向，线宽 ∝ |标准化 β|，实线/虚线 = p<阈值 / 不显著）

对应数据形态与平台里其它表格分析一致（如"被试 × EEG 指标 + 一个临床量表得分"、
"样点 × 环境变量 + 抗生素抗性丰度"）。

设计要点
--------
- **不依赖 Qt**：只用到 numpy / scipy / scikit-learn / statsmodels（绘图可选 matplotlib），
  返回 ``(fig, ax)``，既能嵌进 GUI 也能在脚本里批量出图。
- **置换重要性在留出集上算**：森林本身容易高估训练集变量，置换重要性 + 独立测试集
  才反映"对新样本的预测贡献"；同时返回测试集 R²，便于判断这次回归是否可信。
- **标准化 β**：LMM/OLS 前把所有变量按训练样本 z-score，β 之间才可比（弧线宽度才有意义）。
- **返回三张整洁表**：重要性表 / 相关表 / 效应表，可直接导出 CSV 写进论文附录。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: 随机森林默认参数（回归任务）
DEFAULT_RF = {
    "n_estimators": 500,
    "min_samples_leaf": 2,
    "max_features": 1.0 / 3.0,
    "random_state": 0,
    "n_jobs": -1,
}

#: 绘图默认样式（与参考图的风格一致；GUI 里可整体替换）
PLOT_STYLE = {
    "matrix_cmap": "RdBu",
    "positive_color": "#1584b3",
    "negative_color": "#d66a00",
    "importance_color": "#4c78a8",
    "hub_color": "#253238",
    "hub_x": 0.36,
    "hub_y": None,                 # None → len(predictors) * 0.70
    "legend_anchor": None,         # None → 自动：变量多时 (0.38, 0.18)，少时下移避免压住末行标签
    "row_label_offset": 0.58,
    "colorbar_position": (0.705, 0.29, 0.016, 0.43),
    "top_corr_color": "#3f3f3f",
    "show_link_endpoints": False,
    "annotate_top_corr": False,
    "top_corr_count": 3,
    "annotate_links": False,
    "link_value_column": "p_value",
    "link_value_absolute": False,
    "link_value_format": ".4g",
    "link_label_significant_only": True,
    "show_importance_values": False,
    "sig_level": 0.05,
}


# ================================================================ 数据整理
def prepare_table(table, col_names, response: str, predictors=None):
    """校验并整理表格：丢掉含缺失的行、常数列，返回 (X, y, names, meta)。

    - 缺失值：**整行丢弃**（并用 dropped_rows 告知丢了几行）
    - 常数列（标准差为 0）：无法标准化/回归，直接剔除并记进 dropped_cols
    """
    table = np.asarray(table, dtype=float)
    if table.ndim != 2:
        raise ValueError(f"需要二维表格，当前形状 {table.shape}")
    col_names = [str(c) for c in col_names]
    if len(col_names) != table.shape[1]:
        raise ValueError(f"列名数（{len(col_names)}）与表格列数（{table.shape[1]}）不一致")

    if response not in col_names:
        raise ValueError(f"目标变量「{response}」不在列名里：{col_names}")
    cols = list(predictors) if predictors else [c for c in col_names if c != response]
    missing = [c for c in cols if c not in col_names]
    if missing:
        raise ValueError(f"这些自变量不在列名里：{missing}")
    if len(cols) < 2:
        raise ValueError("至少需要 2 个自变量")

    idx = [col_names.index(c) for c in cols]
    y = table[:, col_names.index(response)]
    X = table[:, idx]

    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    dropped_rows = int((~valid).sum())
    X, y = X[valid], y[valid]
    if X.shape[0] < 3:
        raise ValueError(f"有效样本只剩 {X.shape[0]} 行，无法做回归/森林")

    keep = X.std(axis=0) > 0
    dropped_cols = [c for c, k in zip(cols, keep) if not k]
    X = X[:, keep]
    cols = [c for c, k in zip(cols, keep) if k]
    if len(cols) < 2:
        raise ValueError(f"剔除常数列后自变量不足 2 个（剔除：{dropped_cols}）")

    return X, y, cols, {"n_obs": int(X.shape[0]), "dropped_rows": dropped_rows,
                        "dropped_cols": dropped_cols}


# ============================================================ 随机森林重要性
def rf_importance(X, y, names, test_size: float = 0.25, n_repeats: int = 10,
                  seed: int = 0, n_estimators: int = 500, rf_params: dict | None = None):
    """随机森林 + **留出集上的置换重要性**。

    返回 ``(rows, meta)``：rows 为 ``[{variable, importance, importance_std, rank}, ...]``
    （按 importance 降序）；meta 含 ``r2_train`` / ``r2_test`` / ``n_test``。

    为什么用置换重要性：森林自带的 impurity importance 偏向高基数变量且在训练集上
    算；打乱变量后看**测试集**误差增加多少，才是"对新样本的预测贡献"。
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 10:
        raise ValueError(f"样本太少（{n}），随机森林至少需要 10 行")
    if int(n * test_size) < 2:                    # 小样本时保证测试集至少 2 行
        test_size = 2.0 / n

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    params = {**DEFAULT_RF, "n_estimators": int(n_estimators), **(rf_params or {})}
    rf = RandomForestRegressor(**params).fit(Xtr, ytr)
    r2_train = float(rf.score(Xtr, ytr))
    r2_test = float(rf.score(Xte, yte))

    pi = permutation_importance(rf, Xte, yte, n_repeats=int(n_repeats),
                               random_state=seed, n_jobs=params.get("n_jobs", -1))
    rows = [{"variable": names[i],
             "importance": float(pi.importances_mean[i]),
             "importance_std": float(pi.importances_std[i])}
            for i in range(len(names))]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    for k, r in enumerate(rows, 1):
        r["rank"] = k
    meta = {"r2_train": r2_train, "r2_test": r2_test, "n_test": int(len(yte)),
            "n_estimators": int(params["n_estimators"])}
    return rows, meta


# ================================================================ Pearson
def pearson_matrix(table, names, min_n: int = 3):
    """变量两两 Pearson 相关矩阵 → (C,C)，对角 = 1。"""
    M = np.asarray(table, dtype=float)
    if M.shape[1] != len(names):
        raise ValueError(f"列数（{M.shape[1]}）与变量名数（{len(names)}）不一致")
    if M.shape[0] < min_n:
        raise ValueError(f"样本太少（{M.shape[0]} 行），相关系数不可靠")
    C = M.shape[1]
    out = np.eye(C)
    for i in range(C):
        for j in range(i + 1, C):
            a, b = M[:, i], M[:, j]
            if a.std() == 0 or b.std() == 0:
                r = 0.0
            else:
                r = float(np.corrcoef(a, b)[0, 1])
            out[i, j] = out[j, i] = 0.0 if not np.isfinite(r) else r
    return out


# ======================================================= 线性混合效应模型
def fit_effects(X, y, names, groups=None, standardize: bool = True,
                method: str = "lmm", max_iter: int = 200):
    """目标变量 ~ 各自变量（LMM 或 OLS）→ 整洁的效应表。

    - ``method='lmm'`` 且给了 ``groups``：``MixedLM(y, X, groups=groups)``（随机截距），
      适合"同一被试/地点/年份重复观测" —— 忽略它会把组间差异错算成变量效应。
    - 其它情况退回 OLS，并在 meta 里说明原因。
    - ``standardize=True`` 时先对 y 与每个 x 做 z-score，于是 β 可比（弧线宽度才有意义）。

    返回 ``(rows, meta)``，rows 为
    ``[{variable, beta, standardized_beta, se, statistic, p_value, ci_low, ci_high}, ...]``。
    """
    import statsmodels.api as sm

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if standardize:
        yv = (y - y.mean()) / (y.std(ddof=1) or 1.0)
        mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
        sd = np.where(sd > 0, sd, 1.0)
        Xv = (X - mu) / sd
    else:
        yv, Xv = y, X

    notes: list[str] = []
    used = method
    groups_arr = None
    if groups is not None:
        groups_arr = np.asarray(groups)
        if len(groups_arr) != n:
            raise ValueError(f"分组变量长度（{len(groups_arr)}）与样本数（{n}）不一致")
        if len(np.unique(groups_arr)) < 2:
            notes.append("分组变量只有 1 个取值 → 无法估计随机效应，已改用 OLS")
            groups_arr, used = None, "ols"

    res, model_kind = None, "ols"
    if used == "lmm" and groups_arr is not None:
        try:
            import warnings as _w

            md = sm.MixedLM(yv, sm.add_constant(Xv), groups=groups_arr)
            with _w.catch_warnings(record=True) as caught:   # 捕获"随机效应奇异"等告警
                _w.simplefilter("always")
                fit = md.fit(reml=True, method="lbfgs", maxiter=int(max_iter))
            if not bool(getattr(fit, "converged", True)):
                raise RuntimeError("MixedLM 未收敛")
            res, model_kind = fit, "lmm"
            cov_re = np.asarray(getattr(fit, "cov_re", [[0.0]]), dtype=float)
            notes.append(f"随机截距方差 = {float(np.ravel(cov_re)[0]):.3g}"
                         f"（{len(np.unique(groups_arr))} 组 × 平均 "
                         f"{n / max(len(np.unique(groups_arr)), 1):.1f} 次观测）")
            if any("singular" in str(w.message).lower() for w in caught):
                notes.append("随机效应协方差接近奇异：说明组间差异很小，"
                             "效应估计与 OLS 基本相同（结果仍可用，但不必强调随机效应）")
        except Exception as e:  # noqa: BLE001 - 收敛失败/奇异 → 退回 OLS，但不静默
            notes.append(f"LMM 拟合失败（{type(e).__name__}: {str(e).splitlines()[0][:80]}）→ 已改用 OLS")
            res = None
    if res is None:
        res = sm.OLS(yv, sm.add_constant(Xv)).fit()
        model_kind = "ols"
        if groups_arr is not None and used == "lmm":
            notes.append("OLS 未考虑分组结构：若组间存在系统差异，效应估计可能有偏")

    params = np.asarray(res.params, dtype=float)[1:]      # 去掉截距
    bse = np.asarray(res.bse, dtype=float)[1:]
    stat = np.asarray(res.tvalues, dtype=float)[1:]
    pval = np.asarray(res.pvalues, dtype=float)[1:]
    ci = np.asarray(res.conf_int(), dtype=float)[1:]
    rows = []
    for i, name in enumerate(names):
        beta = float(params[i])
        rows.append({
            "variable": name,
            "beta": beta,
            "standardized_beta": beta if standardize else float("nan"),
            "se": float(bse[i]),
            "statistic": float(stat[i]),
            "p_value": float(pval[i]) if np.isfinite(pval[i]) else float("nan"),
            "ci_low": float(ci[i, 0]),
            "ci_high": float(ci[i, 1]),
        })
    meta = {"model": model_kind, "n_obs": int(n),
            "n_groups": int(len(np.unique(groups_arr))) if groups_arr is not None else 0,
            "standardized": bool(standardize), "notes": notes}
    return rows, meta


# ================================================================ 一键分析
def linked_analysis(table, response: str, col_names, predictors=None, groups=None,
                    n_estimators: int = 500, n_repeats: int = 10, seed: int = 0,
                    method: str = "lmm", standardize: bool = True,
                    log_transform: bool = False, log=print) -> dict:
    """完整流程：整理 → （可选）对数变换 → 森林重要性 → 相关矩阵 → LMM/OLS 效应。

    ``log_transform=True`` 时对**目标变量与所有自变量**取 log10（要求全为正数）：
    脑电频带功率、物种丰度这类**正偏重尾**指标不取对数时，少数极端样本会主导
    MSE，随机森林的 R² 甚至为负（这时日志会提示"重要性排序仅供参考"）。
    实测：BNCI 的 α 功率当目标变量时，R² 从 −0.001 提升到 0.83。
    注意收益主要来自**目标变量**的重尾被压缩 —— 随机森林对自变量做单调变换
    本来就不敏感（树只看大小顺序），但变换会同时用于相关矩阵与回归，
    保证三联图三部分的量纲一致。

    Returns
    -------
    dict::

        {response, predictors, importance[], corr (C,C), effects[], meta{...}}
    """
    X, y, names, prep = prepare_table(table, col_names, response, predictors)
    log(f"变量关联分析：{prep['n_obs']} 行 × {len(names)} 个自变量"
        + (f"（丢弃 {prep['dropped_rows']} 行含缺失、{len(prep['dropped_cols'])} 个常数列）"
           if prep["dropped_rows"] or prep["dropped_cols"] else ""))
    prep["log10"] = False
    if log_transform:
        if (X > 0).all() and (y > 0).all():
            X, y = np.log10(X), np.log10(y)
            prep["log10"] = True
            log("已对目标变量与自变量取 log10（正偏重尾指标建议开启）")
        else:
            log("  [提示] 存在非正值，跳过 log10 变换（对数只对全正数据有意义）")

    importance, rf_meta = rf_importance(X, y, names, n_repeats=n_repeats, seed=seed,
                                        n_estimators=int(n_estimators))
    log(f"随机森林（{rf_meta['n_estimators']} 棵树）：R²(训练)={rf_meta['r2_train']:.3f}，"
        f"R²(留出 {rf_meta['n_test']} 行)={rf_meta['r2_test']:.3f}；"
        f"Top-3 重要变量 = {[r['variable'] for r in importance[:3]]}")
    if rf_meta["r2_test"] < 0:
        log("  [注意] 留出集 R² < 0 → 这些变量线性/非线性都预测不了目标变量，"
            "重要性排序仅供参考")

    corr = pearson_matrix(X, names)
    pairs = [(abs(float(corr[i, j])), i, j)
             for i in range(len(names)) for j in range(i + 1, len(names))]
    if pairs:
        _, bi, bj = max(pairs, key=lambda p: p[0])
        log(f"相关矩阵：|r| 最大的一对 = {names[bi]}–{names[bj]} "
            f"r={corr[bi, bj]:+.2f}")

    eff_groups = None
    if groups is not None:
        g = np.asarray(groups)
        if g.ndim == 2 and g.shape[1] == 1:
            g = g.ravel()
        # 与 prepare_table 同样的行过滤（缺失行/常数列不影响行过滤，但需与 X 对齐）
        tbl = np.asarray(table, dtype=float)
        names_all = [str(c) for c in col_names]
        cols = list(predictors) if predictors else [c for c in names_all if c != response]
        valid = (np.isfinite(tbl[:, [names_all.index(c) for c in cols]]).all(axis=1)
                 & np.isfinite(tbl[:, names_all.index(response)]))
        eff_groups = g[valid]
    effects, eff_meta = fit_effects(X, y, names, groups=eff_groups,
                                    standardize=standardize, method=method)
    sig = [r for r in effects if np.isfinite(r["p_value"]) and r["p_value"] < 0.05]
    detail = "、".join(f"{r['variable']}(β={r['beta']:+.2f})" for r in sig[:5])
    log(f"{eff_meta['model'].upper()} 效应：显著（p<0.05）{len(sig)}/{len(effects)} 个变量"
        + (f" → {detail}" if detail else ""))
    for note in eff_meta["notes"]:
        log(f"  [效应模型] {note}")

    return {"response": response, "predictors": names, "importance": importance,
            "corr": corr, "effects": effects,
            "meta": {"prepare": prep, "rf": rf_meta, "effects": eff_meta}}


# ================================================================ 表导出
def export_tables(result: dict) -> dict[str, tuple[list[str], list[list[str]]]]:
    """把分析结果整理成三张表（``{文件名: (表头, 行)}``），供 CSV 导出。"""
    from .stats import stars

    imp = result["importance"]
    names = [r["variable"] for r in imp]
    corr = np.asarray(result["corr"])
    idx = result["predictors"]

    t_imp = (["variable", "importance(permutation)", "std", "rank"],
             [[r["variable"], f"{r['importance']:.6g}", f"{r['importance_std']:.4g}", str(r["rank"])]
              for r in imp])
    order = [idx.index(n) for n in names]
    t_corr = (["variable"] + names,
              [[names[i]] + [f"{corr[order[i], order[j]]:+.4f}" for j in range(len(names))]
               for i in range(len(names))])
    # 效应表顺带给出 FDR 校正结果（逐边多重比较在变量层面的对应物）
    from .stats import correct

    pvals = np.array([r["p_value"] for r in result["effects"]], dtype=float)
    ok = np.isfinite(pvals)
    padj = np.full_like(pvals, np.nan)
    if ok.any():
        padj[ok] = correct(pvals[ok], "fdr_bh")
    rows = []
    for r, pf in zip(result["effects"], padj):
        p = r["p_value"]
        rows.append([r["variable"], f"{r['beta']:+.4f}", f"{r['se']:.4f}",
                     f"{r['statistic']:+.3f}",
                     f"{p:.4g}" if np.isfinite(p) else "nan",
                     f"{pf:.4g}" if np.isfinite(pf) else "nan",
                     stars(p) if np.isfinite(p) else "nan",
                     f"{r['ci_low']:+.4f}", f"{r['ci_high']:+.4f}",
                     r["variable"]])
    t_eff = (["variable", "beta(std)", "se", "statistic", "p_value", "p_fdr", "sig",
              "ci_low", "ci_high", "label"], rows)
    return {"importance.csv": t_imp, "pearson.csv": t_corr, "effects.csv": t_eff}


# ================================================================ 绘图
def plot_linked_heatmap(corr, names, importance, effects, response: str, *, fig=None,
                        output: str | Path | None = None, title: str | None = None,
                        figsize: tuple[float, float] | None = None, dpi: int = 300,
                        font_family=None, base_font_size: float = 12,
                        title_font_size: float | None = None,
                        top_label_size: float | None = None,
                        row_label_size: float | None = None,
                        legend_font_size: float | None = None,
                        link_label_size: float | None = None, **style):
    """Nature 风格三联图：顶部 RF 重要性 + 中间 Pearson 三角矩阵 + 底部 LMM 弧线。

    ``importance`` / ``effects`` 接受 ``[{...}, ...]``（:func:`linked_analysis` 的输出）
    或 pandas DataFrame。变量顺序**按重要性降序**重排。

    所有样式项见模块级 :data:`PLOT_STYLE`，可通过关键字覆盖。
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, PathPatch, Rectangle
    from matplotlib.path import Path as MplPath

    st = {**PLOT_STYLE, **(style or {})}
    corr = np.asarray(corr, dtype=float)
    names = [str(x) for x in names]
    imp_rows = _as_rows(importance, ("variable", "importance"))
    eff_rows = _as_rows(effects, ("variable", "standardized_beta", "p_value"))
    eff_by_name = {r["variable"]: r for r in eff_rows}
    imp_by_name = {r["variable"]: r for r in imp_rows}

    columns = sorted(names, key=lambda x: float(imp_by_name.get(x, {}).get("importance", 0.0)),
                     reverse=True)
    n = len(columns)
    if n < 2:
        raise ValueError("至少需要 2 个自变量")
    pos = {c: i for i, c in enumerate(names)}

    if font_family is None:                       # 默认用平台统一字体栈（含中文宋体回退）
        try:
            from .plot_fonts import apply_matplotlib_font, font_stack

            apply_matplotlib_font()               # 同时把 matplotlib 的 findfont 告警静音
            font_family = font_stack()
        except Exception:  # noqa: BLE001
            font_family = "Times New Roman"
    if figsize is None:
        figsize = (max(11, n * 0.62 + 5.5), max(8, n * 0.54 + 2.8))
    title_font_size = title_font_size or base_font_size + 4
    top_label_size = top_label_size or base_font_size - 2
    row_label_size = row_label_size or base_font_size - 1
    legend_font_size = legend_font_size or base_font_size - 1
    link_label_size = link_label_size or base_font_size - 3

    # matplotlib >= 3.9 移除了 cm.get_cmap，这里用 colormaps 注册表 + 兜底
    cmap = matplotlib.colormaps.get(st["matrix_cmap"]) if hasattr(matplotlib, "colormaps") \
        else plt.get_cmap(st["matrix_cmap"])
    if cmap is None:                              # pragma: no cover - 名字写错时兜底
        cmap = matplotlib.colormaps["RdBu"]
    norm = colors.Normalize(vmin=-1, vmax=1)

    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig.clear()
        ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.035, right=0.78, top=0.91, bottom=0.10)
    ax.set_xlim(-2.7, n + 1.45)
    ax.set_ylim(-1.68, n + 0.95)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        fig.suptitle(title, x=0.04, y=0.97, ha="left", fontsize=title_font_size,
                     fontweight="bold", fontfamily=font_family)

    # ---- 顶部：RF 重要性（原始数值按最大值缩放为条高；负值夹到 0 只影响显示）----
    raw_scores = np.array([float(imp_by_name.get(c, {}).get("importance", 0.0)) for c in columns])
    display = np.clip(raw_scores, 0.0, None)
    denom = display.max() if display.max() else 1.0
    bar_scores = display / denom
    for j, (raw_score, score) in enumerate(zip(raw_scores, bar_scores)):
        ax.add_patch(Rectangle((j, -0.90), 0.88, 0.42, facecolor="#e6e6e6",
                               edgecolor="white", lw=0.5))
        ax.add_patch(Rectangle((j + 0.06, -0.84 + 0.31 * (1 - score)), 0.76, 0.31 * score,
                               facecolor=st["importance_color"], edgecolor="none"))
        if st["show_importance_values"]:
            ax.text(j + 0.44, -1.04, f"{raw_score:.3g}", ha="center", va="top",
                    fontsize=base_font_size - 4, color="#444", fontfamily=font_family)
    ax.text(-1.45, -0.66, "RF\nimportance", ha="right", va="center",
            fontsize=base_font_size - 3, color="#333", fontfamily=font_family)

    # ---- 中间：相关三角矩阵（方块边长 ∝ √|r|，颜色 = 方向）----
    top_pairs = set()
    if st["annotate_top_corr"] and int(st["top_corr_count"]) > 0:
        pair_values = []
        for i in range(n):
            for j in range(i + 1, n):
                v = float(corr[pos[columns[i]], pos[columns[j]]])
                pair_values.append((abs(v), i, j, v))
        top_pairs = {(i, j) for _, i, j, _ in sorted(pair_values, reverse=True)[:int(st["top_corr_count"])]}
    for i in range(n):
        for j in range(i, n):
            if i == j:
                continue                          # 对角 r=1 省略，保持三角区清爽
            value = float(corr[pos[columns[i]], pos[columns[j]]])
            ax.add_patch(Rectangle((j, i), 0.88, 0.88, facecolor="white",
                                   edgecolor="#c7cdd2", lw=0.55))
            side = 0.10 + 0.68 * np.sqrt(abs(value))
            ax.add_patch(Rectangle((j + 0.44 - side / 2, i + 0.44 - side / 2), side, side,
                                   facecolor=cmap(norm(value)), edgecolor="#71808a", lw=0.45))
            if (i, j) in top_pairs:
                ax.text(j + 0.44, i + 0.44, format(value, ".4g"), ha="center", va="center",
                        fontsize=base_font_size - 3, color=st["top_corr_color"],
                        fontweight="bold", fontfamily=font_family, zorder=7)

    labels = [c.replace("_", " ") for c in columns]
    for i, label in enumerate(labels):
        ax.text(i + 0.44, -1.34, label, rotation=48, ha="left", va="bottom",
                fontsize=top_label_size, fontfamily=font_family)
        ax.text(i + st["row_label_offset"], i + 0.44, label, ha="right", va="center",
                fontsize=row_label_size, fontfamily=font_family)

    # ---- 底部：LMM/OLS 效应弧线（颜色=方向，线宽∝|β|，实/虚=显著性）----
    hub_y = n * 0.70 if st["hub_y"] is None else float(st["hub_y"])
    ax.add_patch(Circle((st["hub_x"], hub_y), 0.16, color=st["hub_color"], ec="white",
                        lw=0.8, zorder=10))
    ax.text(st["hub_x"] - 0.02, hub_y + 0.31, response.replace("_", " "), ha="center",
            va="top", fontsize=base_font_size, fontweight="bold", fontfamily=font_family)
    for i, name in enumerate(columns):
        row = eff_by_name.get(name)
        if not row:
            continue
        beta = float(row.get("standardized_beta", row.get("beta", 0.0)) or 0.0)
        if not np.isfinite(beta):
            beta = float(row.get("beta", 0.0) or 0.0)
        pvalue = float(row.get("p_value", 1.0))
        target_x, target_y = i + st["row_label_offset"] + 0.02, i + 0.44
        ratio = i / max(n - 1, 1)
        bend = target_x - (2.30 - 1.20 * ratio)
        vertices = [(st["hub_x"], hub_y),
                    (st["hub_x"] - 0.20, hub_y - 1.35 * (1 - ratio)),
                    (bend, target_y), (target_x, target_y)]
        path = MplPath(vertices, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        color = st["positive_color"] if beta >= 0 else st["negative_color"]
        width = 0.7 + min(abs(beta), 0.9) * 5.2
        style_ = "-" if (np.isfinite(pvalue) and pvalue < float(st["sig_level"])) else (0, (2.2, 2.0))
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color, lw=width,
                               linestyle=style_, alpha=0.86, zorder=2))
        if (st["annotate_links"] and st["link_value_column"] in row
                and (not st["link_label_significant_only"]
                     or (np.isfinite(pvalue) and pvalue < float(st["sig_level"])))):
            link_value = float(row[st["link_value_column"]])
            if st["link_value_absolute"]:
                link_value = abs(link_value)
            p0, p1, p2, p3 = np.asarray(vertices, dtype=float)
            t = 0.52
            point = ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
                     + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)
            tangent = (3 * (1 - t) ** 2 * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1)
                       + 3 * t ** 2 * (p3 - p2))
            angle = float(np.degrees(np.arctan2(tangent[1], tangent[0])))
            if angle < -90 or angle > 90:
                angle += 180
            normal = np.array([-tangent[1], tangent[0]], dtype=float)
            norm_len = np.linalg.norm(normal) or 1.0
            point = point + normal / norm_len * 0.11
            ax.text(point[0], point[1], format(link_value, st["link_value_format"]),
                    ha="center", va="center", rotation=angle, rotation_mode="anchor",
                    fontsize=link_label_size, color=color, fontfamily=font_family, zorder=6)
        if st["show_link_endpoints"]:
            ax.add_patch(Circle((target_x, target_y), 0.045, color=color, ec="white",
                                lw=0.4, zorder=9))

    # ---- 色条 + 图例 ----
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cax = fig.add_axes(st["colorbar_position"])
    cbar = fig.colorbar(scalar, cax=cax)
    cbar.set_label("Pearson's r", fontsize=base_font_size - 1, fontfamily=font_family)
    cbar.ax.tick_params(labelsize=base_font_size - 2)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily(font_family)

    legend = [
        Line2D([0], [0], color=st["positive_color"], lw=2.5, label="Positive"),
        Line2D([0], [0], color=st["negative_color"], lw=2.5, label="Negative"),
        Line2D([0], [0], color="#555555", lw=2.2, label=f"p < {st['sig_level']:g}"),
        Line2D([0], [0], color="#555555", lw=2.2, ls="--", label=f"p >= {st['sig_level']:g}"),
    ]
    anchor = st["legend_anchor"] or ((0.38, 0.18) if n >= 12 else (0.36, 0.085))
    legend_artist = fig.legend(handles=legend, loc="lower center",
                               bbox_to_anchor=anchor, ncol=4, frameon=False,
                               prop={"family": font_family[0] if isinstance(font_family, (list, tuple))
                                     else font_family, "size": legend_font_size},
                               title="LMM links", title_fontsize=legend_font_size + 1)
    legend_artist.get_title().set_fontfamily(font_family)

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig, ax


def _as_rows(obj, keys: tuple[str, ...]) -> list[dict]:
    """把 ``[{...}]`` 或 DataFrame 统一成 list[dict]。"""
    if obj is None:
        return []
    if hasattr(obj, "to_dict") and hasattr(obj, "columns"):        # pandas DataFrame
        return obj.to_dict("records")
    out = []
    for r in obj:
        if isinstance(r, dict):
            out.append(dict(r))
        else:                                                       # 序列 → 按 keys 对齐
            out.append({k: v for k, v in zip(keys, r)})
    return out
