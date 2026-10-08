"""脑网络构建 / 动态网络 / 生物标志物挖掘单测（对齐 BrainNet Studio 流程）。"""
import numpy as np

from core.brainnet import (biomarkers_by_condition, connection_variability,
                           dynamic_connectivity, dynamic_features,
                           node_importance, static_connectivity)

FS = 250.0
rng = np.random.default_rng(0)


def test_static_connectivity():
    X = rng.standard_normal((10, 6, 300))
    A = static_connectivity(X, FS, method="corr")
    assert A.shape == (10, 6, 6)


def test_dynamic_connectivity_and_features():
    X = rng.standard_normal((10, 5, 300))
    t, adj = dynamic_connectivity(X, FS, method="corr", win_s=0.4, step_s=0.2)
    assert adj.shape[1:] == (5, 5)
    assert len(t) == adj.shape[0]
    assert adj.shape[0] >= 2
    feats = dynamic_features(adj)
    assert feats["global_efficiency"].shape == (adj.shape[0],)
    assert feats["clustering"].shape == (adj.shape[0],)
    assert feats["strength"].shape == (adj.shape[0], 5)
    var = connection_variability(adj)
    assert var.shape == (5, 5)
    assert (var >= 0).all()


def test_node_importance():
    X = rng.standard_normal((10, 5, 300))
    A = static_connectivity(X, FS).mean(axis=0)
    np.fill_diagonal(A, 0)
    s = node_importance(A, "strength")
    d = node_importance(A, "degree")
    assert s.shape == (5,) and d.shape == (5,)
    assert (s > 0).all()


def test_biomarkers_by_condition():
    # 两组各 10 trials，给 b 组注入通道 1-2 的强相关
    X = rng.standard_normal((20, 6, 200))
    labels = ["a"] * 10 + ["b"] * 10
    X[10:, 1] = 0.9 * X[10:, 2] + 0.1 * rng.standard_normal((10, 200))

    r = biomarkers_by_condition(X, labels, FS, method="corr", top_n=5)
    assert r["conds"] == ["a", "b"]
    assert r["t_stat"].shape == (6, 6)
    np.testing.assert_allclose(r["t_stat"], r["t_stat"].T)  # 对称
    assert len(r["top_connections"]) == 5
    assert r["node_score"].shape == (6,)
    pairs = [{i, j} for i, j, *_ in r["top_connections"]]
    assert {1, 2} in pairs  # 注入的判别连接应被检出（Top-5 内）
    for i, j, t, p, m0, m1 in r["top_connections"]:
        assert 0.0 <= p <= 1.0


def test_biomarkers_requires_two_conditions():
    X = rng.standard_normal((6, 4, 200))
    try:
        biomarkers_by_condition(X, ["a"] * 6, FS)
    except ValueError as e:
        assert "条件" in str(e)
    else:  # pragma: no cover
        raise AssertionError("单条件应报错")
