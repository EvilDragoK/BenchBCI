"""归一化/对齐方案单测（对应文献 Alignment 对照表）。"""
import numpy as np
import pytest

from core.datasets import (NORM_MODES, apply_normalization, compute_norm_stats,
                           _ema_normalize, _rank_normal)


def _data(seed=0, N=8, C=6, T=501):
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 250.0
    return (10e-6 * (np.sin(2 * np.pi * 10 * t) * 3
                     + rng.standard_normal((N, C, T)))).astype(np.float64)


ALL = [m for m in NORM_MODES if m != "none"]


@pytest.mark.parametrize("mode", ALL)
def test_all_modes_finite_and_shape(mode):
    X = _data()
    stats = compute_norm_stats(X, mode)
    Y = apply_normalization(X, mode, stats)
    assert Y.shape == X.shape
    assert np.isfinite(Y).all()


def test_zscore_channel_stats():
    X = _data()
    stats = compute_norm_stats(X[:6], "zscore_channel")
    Ztr = apply_normalization(X[:6], "zscore_channel", stats)
    np.testing.assert_allclose(Ztr.mean(), 0.0, atol=1e-5)
    np.testing.assert_allclose(Ztr.std(), 1.0, atol=1e-2)
    # 测试集用训练统计量（防泄漏哨兵：不应恰好 0/1）
    Zte = apply_normalization(X[6:], "zscore_channel", stats)
    assert not np.allclose(Zte.mean(), 0.0, atol=1e-6)


def test_minmax_ranges():
    X = _data()
    s = compute_norm_stats(X, "minmax_pm")
    Y = apply_normalization(X, "minmax_pm", s)
    assert Y.min() >= -1.0 - 1e-6 and Y.max() <= 1.0 + 1e-6
    Ye = apply_normalization(X, "minmax_pm_epoch")
    assert Ye.min() >= -1.0 - 1e-6 and Ye.max() <= 1.0 + 1e-6
    Y01 = apply_normalization(X, "minmax_epoch")
    assert Y01.min() >= 0.0 - 1e-6 and Y01.max() <= 1.0 + 1e-6


def test_instance_norm_zscore_epoch():
    Z = apply_normalization(_data(), "zscore_epoch")
    np.testing.assert_allclose(Z.mean(axis=2), 0.0, atol=1e-5)
    np.testing.assert_allclose(Z.std(axis=2), 1.0, atol=1e-2)


def test_p95_scaling():
    X = _data()
    s = compute_norm_stats(X, "p95")
    Y = apply_normalization(X, "p95", s)
    # 95% 的样本绝对值应 ≤ 1（p95 定义）
    assert (np.abs(Y) <= 1.0).mean() >= 0.94


def test_iqr_robust():
    X = _data()
    # 加入极端伪迹
    X[0, 0, :50] += 1.0
    s = compute_norm_stats(X, "iqr")
    Y = apply_normalization(X, "iqr", s)
    med = np.median(Y)
    assert abs(med) < 0.5  # 中位数居中，且不受极端值拉动
    # IQR 比 std 更抗极值：IQR 缩放后 99 分位远小于 z-score 的 99 分位
    Z = apply_normalization(X, "zscore_channel", compute_norm_stats(X, "zscore_channel"))
    assert np.percentile(np.abs(Y), 99) < np.percentile(np.abs(Z), 99)


def test_x_div():
    Y = apply_normalization(_data(), "x_div", {"constant": 100.0})
    X = _data()
    np.testing.assert_allclose(Y, X / 100.0, rtol=1e-5)


def test_quantile_transform():
    Z = apply_normalization(_data(), "quantile")
    # 秩→正态：均值≈0、方差≈1，且保序（秩不变）
    np.testing.assert_allclose(Z.mean(axis=2), 0.0, atol=1e-6)
    np.testing.assert_allclose(Z.std(axis=2), 1.0, atol=1e-2)
    x = _data()[0, 0]
    order_x = np.argsort(x)
    order_z = np.argsort(_rank_normal(x[None, None])[0, 0])
    np.testing.assert_array_equal(order_x, order_z)


def test_alaw():
    X = _data()
    Y = apply_normalization(X, "alaw")
    A = 87.56
    u = np.abs(X).max(axis=2, keepdims=True)
    ref = np.sign(X) * np.log(1 + A * np.abs(X) / u) / np.log(1 + A)
    np.testing.assert_allclose(Y, ref, rtol=1e-4)
    assert np.abs(Y).max() <= 1.0 + 1e-6
    # 压缩性质：最大幅度处 → ±1
    i = np.unravel_index(np.abs(X).argmax(), X.shape)
    assert abs(Y[i]) > 0.99


def test_ema():
    Y = _ema_normalize(_data(), alpha=0.05)
    assert np.isfinite(Y).all()
    # EMA 去除了缓变趋势：前 100 点的均值应接近 0
    assert abs(Y[:, :, :100].mean()) < 1.0


def test_zero_mean():
    Y = apply_normalization(_data(), "zero_mean")
    np.testing.assert_allclose(Y.mean(axis=2), 0.0, atol=1e-10)


def test_backward_compat_mean_std():
    X = _data()
    mean, std = X.mean(axis=(0, 2), keepdims=True), X.std(axis=(0, 2), keepdims=True)
    Y = apply_normalization(X, "zscore_channel", mean=mean, std=std)
    assert np.isfinite(Y).all()


def test_unknown_mode():
    with pytest.raises(ValueError):
        apply_normalization(_data(), "bad_mode")
