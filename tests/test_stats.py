"""统计检验工具单测：t / 置换 / 多重比较校正 / 效应量 / 报告组装。"""
import numpy as np
import pytest

from core.stats import (CORRECT_METHODS, cohens_d, correct, effect_magnitude,
                        group_report, permutation_test, stars, ttest_1samp,
                        ttest_ind, ttest_rel)

rng = np.random.default_rng(0)


def test_ttest_shapes_and_direction():
    a = rng.normal(0.0, 1.0, (3, 50))
    b = rng.normal(0.8, 1.0, (3, 50))
    t, p = ttest_ind(a, b)
    assert t.shape == p.shape == (3,)
    assert (t < 0).all() and (p < 0.01).all()           # a 明显小于 b
    t2, p2 = ttest_1samp(a, 0.0)
    assert t2.shape == (3,) and (np.abs(t2) < 3.0).all()   # 均值接近 0 → t 很小
    t3, p3 = ttest_rel(a, a + 0.5 + rng.normal(0, 1.0, a.shape))
    assert (t3 < 0).all() and (p3 < 0.01).all()


def test_ttest_rel_requires_same_shape():
    with pytest.raises(ValueError):
        ttest_rel(rng.normal(0, 1, (2, 10)), rng.normal(0, 1, (2, 12)))


def test_correct_methods_monotone_and_shape():
    p = np.array([[0.001, 0.02, 0.2, 0.9], [0.3, 0.04, 0.5, 0.6]])
    for m in CORRECT_METHODS:
        q = correct(p, m)
        assert q.shape == p.shape
        assert ((q >= p - 1e-12) & (q <= 1.0)).all()
    assert np.allclose(correct(p, "none"), p)
    fdr, bonf, holm = correct(p, "fdr_bh"), correct(p, "bonferroni"), correct(p, "holm")
    assert (bonf >= fdr - 1e-12).all()                  # Bonferroni 更保守
    assert (holm >= fdr - 1e-12).all()
    # 单调性：原始 p 更小的，校正后 q 不应更大
    order = np.argsort(p, axis=None)
    assert np.all(np.diff(fdr.ravel()[order]) >= -1e-12)
    with pytest.raises(ValueError):
        correct(p, "no-such-method")


def test_cohens_d_known_values():
    a = rng.normal(0.0, 1.0, (1, 200))
    b = rng.normal(1.0, 1.0, (1, 200))
    d = cohens_d(a, b)
    assert -1.3 < d[0] < -0.7                            # 均值差 1、SD 1 → d≈-1
    d_paired = cohens_d(a, a + rng.normal(0.5, 1.0, a.shape), paired=True)
    assert -1.0 < d_paired[0] < -0.2                     # dz = 0.5/SD(diff) ≈ 0.5
    assert np.isfinite(cohens_d(a, a, paired=True)).all()  # 零方差保护，不炸成天文数字
    assert list(effect_magnitude(np.array([0.1, 0.3, 0.6, 1.0]))) == \
        ["可忽略", "小", "中", "大"]


def test_permutation_test():
    a = rng.normal(0.0, 1.0, (2, 30))
    b = rng.normal(1.2, 1.0, (2, 30))
    stat, p = permutation_test(a, b, n_perm=1000, paired=False)
    assert stat.shape == p.shape == (2,)
    assert (p < 0.01).all()
    # 同一份数据 → p 值应接近均匀分布（不系统性显著）
    _, p_null = permutation_test(a, a + rng.normal(0, 1e-6, a.shape), n_perm=1000, paired=True)
    assert p_null.mean() > 0.2
    # 单样本符号翻转：均值远离 0 → 显著
    _, p1 = permutation_test(a + 3.0, n_perm=1000)
    assert (p1 < 0.01).all()


def test_group_report_and_stars():
    a = rng.normal(0.0, 1.0, (4, 40))
    b = rng.normal(0.7, 1.0, (4, 40))
    rep = group_report(a, b, paired=False, correct_method="fdr_bh")
    for k in ("mean_a", "std_a", "mean_b", "t", "p", "p_corrected", "d", "n_a", "n_b"):
        assert rep[k].shape == (4,)
    assert np.all(rep["p_corrected"] >= rep["p"] - 1e-12)
    assert (rep["n_a"] == 40).all() and (rep["n_b"] == 40).all()
    rep2 = group_report(a, n_perm=200)                   # 单样本 + 置换
    assert "perm_p" in rep2 and rep2["perm_p"].shape == (4,)
    assert list(stars(np.array([0.0001, 0.005, 0.04, 0.4]))) == ["***", "**", "*", "ns"]
