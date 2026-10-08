"""变量关联分析（Linked Heatmap）单测：RF 重要性 / Pearson / LMM 与数据读取。

对应参考图（Nature 2026, 654:398–407）里那张三联图：
顶部 RF 重要性、中间 Pearson 三角矩阵、底部 LMM 效应弧线。
"""
import os

import numpy as np
import pytest

from core.linked import (PLOT_STYLE, export_tables, fit_effects, linked_analysis,
                         pearson_matrix, plot_linked_heatmap, prepare_table,
                         rf_importance)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _data(n=240, n_groups=8, seed=0, y_effect=(1.6, -1.2), noise=0.8):
    """合成：x1 正效应、x2 负效应、x3 弱正效应、x4~x8 纯噪声，另有分组随机截距。"""
    rng = np.random.default_rng(seed)
    per = max(n // n_groups, 1)                     # 保证能整除，n 取整到组数倍数
    n = per * n_groups
    groups = np.repeat(np.arange(n_groups), per)
    g_eff = rng.normal(0, 2.0, n_groups)[groups]
    X = rng.normal(0, 1, (n, 8))
    y = (y_effect[0] * X[:, 0] + y_effect[1] * X[:, 1] + 0.3 * X[:, 2]
         + g_eff + rng.normal(0, noise, n))
    names = [f"x{i + 1}" for i in range(8)]
    table = np.column_stack([X, y])
    return table, names + ["y"], names, groups


# ---------------------------------------------------------------- 表读取
def test_load_table_header_only_csv(tmp_path):
    """回归：纯表头 CSV（无行标签列）曾把第一个列名当"角落单元格"丢掉 → 列名整体错位。"""
    from core.netcov import load_table_file, load_matrix_file

    p = tmp_path / "t.csv"
    p.write_text("Temperature,NO3-N,ARG_abundance\n"
                 "1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8")
    M, cols, _rows, nn = load_table_file(p)
    assert cols == ["Temperature", "NO3-N", "ARG_abundance"], "列名不能错位"
    assert M.shape == (3, 3) and not nn
    M2, labels2 = load_matrix_file(p)                 # 矩阵读取器也不能错位
    assert labels2 == ["Temperature", "NO3-N", "ARG_abundance"]
    assert M2.shape == (3, 3)


def test_load_table_text_group_column(tmp_path):
    """文本列（被试编号）要单独返回，供 LMM 当分组变量，而不是让解析崩掉。"""
    from core.netcov import load_table_file

    p = tmp_path / "g.csv"
    p.write_text("subject,alpha,beta,y\nS01,1,2,3\nS01,2,3,5\nS02,3,4,7\n", encoding="utf-8")
    M, cols, _rows, nn = load_table_file(p)
    assert cols == ["alpha", "beta", "y"] and M.shape == (3, 3)
    assert "subject" in nn and list(nn["subject"]) == ["S01", "S01", "S02"]


def test_load_table_corner_layout(tmp_path):
    """"角落单元格 + 行标签"排版（save_matrix_file 的输出）仍要正确解析。"""
    from core.netcov import load_table_file, save_matrix_file

    p = tmp_path / "m.csv"
    A = np.arange(9, dtype=float).reshape(3, 3)
    save_matrix_file(p, A, labels=["r1", "r2", "r3"])
    M, cols, rows, nn = load_table_file(p)
    np.testing.assert_allclose(M, A)
    assert cols == ["r1", "r2", "r3"] and rows == ["r1", "r2", "r3"]


# ---------------------------------------------------------------- 各分析步
def test_prepare_table_drops_missing_and_constant():
    table, cols, names, _ = _data(n=40, n_groups=4)
    table = table.copy()
    table[0, 0] = np.nan                       # 缺一行
    table[:, names.index("x5")] = 1.0          # 常数列
    X, y, kept, meta = prepare_table(table, cols, "y")
    assert meta["dropped_rows"] == 1
    assert meta["dropped_cols"] == ["x5"]
    assert kept == [n for n in names if n != "x5"]
    assert X.shape == (39, 7)
    with pytest.raises(ValueError):
        prepare_table(table, cols, "not_a_column")


def test_pearson_matrix_properties():
    table, cols, names, _ = _data(n=100, n_groups=5, y_effect=(0, 0), noise=1.0)
    corr = pearson_matrix(table[:, :8], names)
    np.testing.assert_allclose(np.diag(corr), 1.0)
    np.testing.assert_allclose(corr, corr.T)
    assert (np.abs(corr) <= 1).all()
    # 逐个元素与 np.corrcoef 核对（含自动构造出的相关结构）
    for i in range(4):
        for j in range(4):
            expect = float(np.corrcoef(table[:, i], table[:, j])[0, 1])
            assert abs(corr[i, j] - expect) < 1e-10
    # 常数列（方差 0）不应产生 nan
    flat = np.column_stack([table[:, 0], np.ones(len(table))])
    c2 = pearson_matrix(flat, ["a", "const"])
    assert np.isfinite(c2).all() and c2[0, 1] == 0.0


def test_rf_importance_ranks_true_drivers_and_reports_r2():
    table, cols, names, _ = _data()
    X, y, kept, _ = prepare_table(table, cols, "y")
    rows, meta = rf_importance(X, y, kept, n_estimators=300, n_repeats=5, seed=0)
    top2 = {r["variable"] for r in rows[:2]}
    assert top2 == {"x1", "x2"}, f"真驱动变量应为 x1/x2，实际 top2={top2}"
    imp = {r["variable"]: r["importance"] for r in rows}
    noise = [imp[f"x{i}"] for i in (4, 5, 6, 7, 8)]
    assert imp["x1"] > max(noise) and imp["x2"] > max(noise)
    assert meta["r2_test"] > 0.3 and meta["n_estimators"] == 300
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    with pytest.raises(ValueError):                       # 样本太少要报错而不是硬跑
        rf_importance(X[:5], y[:5], kept[:5])


def test_lmm_recovers_effects_and_controls_group_structure():
    table, cols, names, groups = _data()
    X, y, kept, _ = prepare_table(table, cols, "y")
    rows, meta = fit_effects(X, y, kept, groups=groups, standardize=True, method="lmm")
    eff = {r["variable"]: r for r in rows}
    assert meta["model"] == "lmm" and meta["n_groups"] == 8
    # 真效应：方向正确、显著
    assert eff["x1"]["beta"] > 0.3 and eff["x1"]["p_value"] < 0.01
    assert eff["x2"]["beta"] < -0.3 and eff["x2"]["p_value"] < 0.01
    # 纯噪声变量不应显著（这是"控制分组 + 显著性"这一步的核心价值）
    for v in ("x4", "x5", "x6", "x7", "x8"):
        assert eff[v]["p_value"] > 0.05, f"{v} 是纯噪声却被判显著"
    assert any("随机截距方差" in n for n in meta["notes"])
    # 标准化后 |β| 可比：真效应明显大于噪声
    assert abs(eff["x1"]["beta"]) > 3 * max(abs(eff[v]["beta"]) for v in ("x4", "x5"))
    # 不给分组 / 只有 1 组 → 退回 OLS 并说明
    rows_ols, meta_ols = fit_effects(X, y, kept, groups=None, method="lmm")
    assert meta_ols["model"] == "ols"
    _, meta_one = fit_effects(X, y, kept, groups=np.zeros(len(y), dtype=int), method="lmm")
    assert meta_one["model"] == "ols" and any("只有 1 个取值" in n for n in meta_one["notes"])


def test_lmm_beats_ols_when_group_effect_confounds_predictor():
    """构造"分组随机截距与自变量相关"的数据：忽略分组会把组间差异错算成变量效应。"""
    rng = np.random.default_rng(1)
    n_groups, per = 30, 12
    g = np.repeat(np.arange(n_groups), per)
    gshift = rng.normal(0, 3.0, n_groups)[g]              # 组间差异很大
    x_conf = gshift + rng.normal(0, 1.0, len(g))          # 自变量被组间差异污染
    x_true = rng.normal(0, 1, len(g))
    y = 1.0 * x_true + gshift + rng.normal(0, 0.5, len(g))   # y 其实只由 x_true 驱动
    X = np.column_stack([x_conf, x_true])
    names = ["x_confounded", "x_true"]
    rows_lmm, _ = fit_effects(X, y, names, groups=g, method="lmm")
    rows_ols, _ = fit_effects(X, y, names, groups=None)
    lmm = {r["variable"]: r for r in rows_lmm}
    ols = {r["variable"]: r for r in rows_ols}
    assert lmm["x_true"]["p_value"] < 0.01                # 真效应
    assert abs(lmm["x_confounded"]["beta"]) < abs(ols["x_confounded"]["beta"])   # 伪相关被压下去


# ------------------------------------------------------------ 一键 + 导出 + 绘图
def test_linked_analysis_bundle_and_export():
    table, cols, names, groups = _data()
    res = linked_analysis(table, "y", cols, groups=groups, n_estimators=200,
                          log=lambda *_: None)
    assert set(res) >= {"response", "predictors", "importance", "corr", "effects", "meta"}
    assert res["corr"].shape == (8, 8)
    assert len(res["importance"]) == 8 and len(res["effects"]) == 8
    tabs = export_tables(res)
    assert set(tabs) == {"importance.csv", "pearson.csv", "effects.csv"}
    head, rows = tabs["effects.csv"]
    assert {"variable", "beta(std)", "se", "p_value", "p_fdr", "sig"} <= set(head)
    assert len(rows) == 8
    assert len(tabs["pearson.csv"][0]) == 9 and len(tabs["pearson.csv"][1]) == 8
    # 分组变量本身不应出现在自变量里
    res2 = linked_analysis(table, "y", cols, predictors=[n for n in names if n != "x1"],
                           log=lambda *_: None)
    assert "x1" not in res2["predictors"]


@pytest.mark.parametrize("kwargs", [
    {},
    {"annotate_links": True, "show_importance_values": True, "annotate_top_corr": True},
    {"font_family": "DejaVu Sans", "base_font_size": 9},
])
def test_plot_linked_heatmap(tmp_path, kwargs):
    import matplotlib

    matplotlib.use("Agg")
    table, cols, names, groups = _data(n=160, n_groups=6)
    res = linked_analysis(table, "y", cols, groups=groups, n_estimators=150,
                          log=lambda *_: None)
    out = tmp_path / "linked.png"
    fig, ax = plot_linked_heatmap(res["corr"], res["predictors"], res["importance"],
                                  res["effects"], "y", output=out, title="demo", **kwargs)
    assert out.exists() and out.stat().st_size > 5000
    assert len(fig.axes) >= 2                          # 主图 + 色条
    assert ax.texts                                 # 有标签/数值文本
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_accepts_pandas_dataframe():
    """参考实现里 importance/effects 是 DataFrame —— 保持兼容。"""
    pd = pytest.importorskip("pandas")
    import matplotlib

    matplotlib.use("Agg")
    table, cols, names, groups = _data(n=120, n_groups=5)
    res = linked_analysis(table, "y", cols, groups=groups, n_estimators=100,
                          log=lambda *_: None)
    fig, _ax = plot_linked_heatmap(res["corr"], res["predictors"],
                                   pd.DataFrame(res["importance"]),
                                   pd.DataFrame(res["effects"]), "y")
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_log_transform_helps_skewed_heavy_tailed_target():
    """正偏重尾数据（如频带功率）：取 log10 后森林 R² 应显著变好。"""
    rng = np.random.default_rng(0)
    n = 300
    x1 = np.exp(rng.normal(0, 1.0, n))            # 对数正态（正偏）
    x2 = np.exp(rng.normal(0, 0.5, n))            # 也保持正值（log10 只对全正数据可用）
    # 目标变量跨几个数量级：原始尺度上少数大值主导 MSE，R² 会掉到负值；
    # 取 log10 后重尾被压缩，森林才能学到结构
    # （注：随机森林对**自变量**的单调变换不敏感，收益主要来自目标变量）
    y = 10.0 ** (1.0 * np.log10(x1) - 0.8 * np.log10(x2) + rng.normal(0, 0.8, n))
    base = y
    table = np.column_stack([x1, x2, base])
    raw = linked_analysis(table, "c", ["x1", "x2", "c"], n_estimators=200,
                          log=lambda *_: None)
    logged = linked_analysis(table, "c", ["x1", "x2", "c"], n_estimators=200,
                             log_transform=True, log=lambda *_: None)
    assert logged["meta"]["prepare"]["log10"] is True
    assert raw["meta"]["prepare"]["log10"] is False
    # 原始尺度上重尾把 MSE 拉爆，森林只能学到很少结构；取对数后明显变好
    assert raw["meta"]["rf"]["r2_test"] < 0.2
    assert logged["meta"]["rf"]["r2_test"] > raw["meta"]["rf"]["r2_test"] + 0.15
    # 非正值 → 自动跳过并说明
    t2 = table.copy()
    t2[0, 0] = 0.0
    skipped = linked_analysis(t2, "c", ["x1", "x2", "c"], n_estimators=100,
                              log_transform=True, log=lambda *_: None)
    assert skipped["meta"]["prepare"]["log10"] is False


def test_singular_random_effect_becomes_chinese_note():
    """随机效应接近奇异时，不能把 statsmodels 的英文告警直接甩给用户。"""
    rng = np.random.default_rng(0)
    n_groups, per = 6, 7
    n = n_groups * per
    g = np.repeat(np.arange(n_groups), per)
    X = rng.normal(0, 1, (n, 2))
    y = 1.0 * X[:, 0] + rng.normal(0, 1.0, n)     # 无组间差异 → 随机效应方差≈0
    rows, meta = fit_effects(X, y, ["a", "b"], groups=g, method="lmm")
    assert meta["model"] in ("lmm", "ols")
    assert any("随机" in note for note in meta["notes"])


def test_plot_style_defaults_match_reference():
    """默认样式（参考图的配色/几何）不该被改动。"""
    assert PLOT_STYLE["matrix_cmap"] == "RdBu"
    assert PLOT_STYLE["positive_color"] == "#1584b3"
    assert PLOT_STYLE["negative_color"] == "#d66a00"
    assert PLOT_STYLE["annotate_links"] is False
    assert PLOT_STYLE["sig_level"] == 0.05
