"""信号分析模块单测：ERP / 时频 / 非线性熵 / 脑网络。"""
import numpy as np
import pytest

from core.erp import DEFAULT_COMPONENTS, component_peak_values, erp_average, extract_components
from core.timefreq import morlet_tfr
from core.nonlinear import (approximate_entropy, entropy_channels,
                            permutation_entropy, sample_entropy)
from core.networks import (binarize, connectivity_matrix, graph_metrics,
                           mi_matrix, pli_matrix)

FS = 250.0
rng = np.random.default_rng(0)


def _synthetic_erp(N=40, C=4, T=200):
    """带一个正峰（~100ms）与一个负峰（~200ms）的合成 ERP + 噪声。"""
    t = np.arange(T) / FS
    wave = np.exp(-((t - 0.1) ** 2) / (2 * 0.02 ** 2)) - 0.7 * np.exp(-((t - 0.2) ** 2) / (2 * 0.03 ** 2))
    return (wave[None, None, :] + 0.1 * rng.standard_normal((N, C, T)))


def test_erp_components():
    X = _synthetic_erp()
    labels = ["a"] * len(X)
    wave = erp_average(X, labels)  # (C,T)
    times = np.arange(X.shape[-1]) / FS
    comps = extract_components(times, wave, components={"P1": ("pos", 0.05, 0.15),
                                                        "N1": ("neg", 0.15, 0.30)})
    assert 80 < comps["P1"]["latency_ms"][0] < 120   # 正峰 ~100ms
    assert 180 < comps["N1"]["latency_ms"][0] < 220  # 负峰 ~200ms
    assert comps["P1"]["amplitude"][0] > 0
    assert comps["N1"]["amplitude"][0] < 0


def test_erp_average_condition():
    X = _synthetic_erp(N=20)
    labels = ["a"] * 10 + ["b"] * 10
    wa = erp_average(X, labels, "a")
    wb = erp_average(X, labels, "b")
    assert wa.shape == wb.shape == (4, 200)


def test_morlet_tfr():
    t = np.arange(501) / FS
    x = np.sin(2 * np.pi * 10 * t)[None, None, :]
    power, freqs = morlet_tfr(x, FS, freqs=np.arange(1, 40, 2))
    assert power.shape[0] == 1 and power.shape[1] == 1
    # 10Hz 频点功率应最大
    band = freqs[(freqs >= 9) & (freqs <= 11)]
    k = int(np.argmin(np.abs(freqs - 10)))
    assert np.argmax(power[0, 0, :, :].mean(axis=1)) == k


def test_entropy_order():
    """正弦（规则）的熵应低于白噪声（随机）。"""
    t = np.arange(500) / FS
    sine = np.sin(2 * np.pi * 10 * t)
    noise = rng.standard_normal(500)
    assert sample_entropy(noise) > sample_entropy(sine)
    assert approximate_entropy(noise) > approximate_entropy(sine)
    assert permutation_entropy(noise) > permutation_entropy(sine)


def test_entropy_channels_shape():
    X = rng.standard_normal((5, 4, 300))
    ent = entropy_channels(X)
    assert set(ent) == {"sample", "approximate", "permutation"}
    for v in ent.values():
        assert v.shape == (4,)
        assert np.isfinite(v).all()


def test_pli_matrix():
    X = rng.standard_normal((4, 6, 501))
    pli = pli_matrix(X, FS)
    assert pli.shape == (4, 6, 6)
    np.testing.assert_allclose(np.diagonal(pli, axis1=1, axis2=2), 1.0)
    assert (pli >= 0).all() and (pli <= 1).all()


def test_mi_matrix():
    X = rng.standard_normal((4, 6, 501))
    mi = mi_matrix(X)
    assert mi.shape == (6, 6)
    np.testing.assert_allclose(np.diagonal(mi), 0.0, atol=1e-12)
    assert (mi >= 0).all()


def test_connectivity_dispatch():
    X = rng.standard_normal((4, 6, 501))
    for method in ("corr", "plv", "pli"):
        A = connectivity_matrix(X, FS, method=method)
        assert A.ndim == 3 and A.shape[1:] == (6, 6)
    A = connectivity_matrix(X, FS, method="mi")
    assert A.shape == (6, 6)


def test_binarize_and_graph_metrics():
    C = 8
    A = rng.standard_normal((C, C))
    A = (A + A.T) / 2
    B = binarize(A, density=0.3)
    assert B.shape == (C, C)
    assert (B == B.T).all()
    np.testing.assert_allclose(np.diagonal(B), 0)
    gm = graph_metrics(B)
    assert gm["degree"].shape == (C,)
    assert 0 <= gm["global_efficiency"] <= 1
    assert gm["clustering"].shape == (C,)
    assert np.isfinite(gm["avg_path_length"])
