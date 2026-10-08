"""特征提取单测：每个特征的形状、取值范围与基本性质。"""
import numpy as np
import pytest

from core.features import (FEATURES, BANDS, absolute_power, c0_complexity,
                           differential_entropy, extract_features,
                           functional_connectivity, hemisphere_pairs,
                           hilbert_timf, hoc, nsi, shannon_entropy,
                           spectral_entropy, theta_beta_ratio, dasm, rasm)

# BNCI 2a 的 22 通道名
CH_2A = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
         "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "P1", "Pz"]

FS = 250.0
rng = np.random.default_rng(0)
# 带一点频结构的合成信号：低频振荡 + 白噪声
t = np.arange(501) / FS
X = (0.3 * np.sin(2 * np.pi * 10 * t) + 0.1 * rng.standard_normal((6, 22, 501))
     + 0.05 * np.sin(2 * np.pi * 22 * t)).astype(np.float64)


def test_shapes():
    assert absolute_power(X, FS).shape == (6, 22, len(BANDS))
    assert theta_beta_ratio(X, FS).shape == (6, 22)
    assert differential_entropy(X, FS).shape == (6, 22, len(BANDS))
    assert shannon_entropy(X).shape == (6, 22)
    assert spectral_entropy(X, FS).shape == (6, 22)
    assert c0_complexity(X).shape == (6, 22)
    assert nsi(X).shape == (6, 22)
    assert hoc(X, order=5).shape == (6, 22, 5)
    ah, iph = hilbert_timf(X, FS)
    assert ah.shape == iph.shape == (6, 22, 1)


def test_value_ranges():
    ap = absolute_power(X, FS)
    assert (ap > 0).all()
    # 10Hz 信号 → alpha 频带功率应最大
    band_names = list(BANDS)
    assert band_names[int(ap[0].mean(0).argmax())] == "alpha"
    assert (theta_beta_ratio(X, FS) > 0).all()
    she = shannon_entropy(X)
    assert (she > 0).all()
    spe = spectral_entropy(X, FS)
    assert ((spe >= 0) & (spe <= 1 + 1e-9)).all()
    c0 = c0_complexity(X)
    assert ((c0 >= 0) & (c0 <= 1 + 1e-9)).all()
    assert (nsi(X) >= 0).all()
    assert np.isfinite(differential_entropy(X, FS)).all()


def test_c0_detects_structure():
    """正弦+小噪声（结构强）的 C0 应明显低于纯白噪声。"""
    white = rng.standard_normal((1, 22, 501))
    assert c0_complexity(white).mean() > c0_complexity(X).mean()


def test_hemisphere_pairs_2a():
    pairs = hemisphere_pairs(CH_2A)
    # FC1-2, FC3-4, C1-2, C3-4, C5-6, CP1-2, CP3-4, CP5-6 = 8 对
    # （2a 数据只有 22 通道，P1 无 P2 配对，中线 z 通道跳过）
    assert len(pairs) == 8
    names = {CH_2A[li] for li, ri in pairs}
    assert "C3" in names and "FC1" in names and "CP5" in names
    assert "Fz" not in names and "Cz" not in names


def test_dasm_rasm():
    d, pairs = dasm(X, FS, CH_2A)
    r, _ = rasm(X, FS, CH_2A)
    assert d.shape == r.shape == (6, 8, len(BANDS))
    assert (r > 0).all()


def test_functional_connectivity():
    for method in ("corr", "plv"):
        fc = functional_connectivity(X, FS, method=method)
        assert fc.shape == (6, 22, 22)
        np.testing.assert_allclose(np.diagonal(fc, axis1=1, axis2=2), 1.0, atol=1e-4)
        np.testing.assert_allclose(fc, np.swapaxes(fc, 1, 2), atol=1e-5)


def test_extract_features_all():
    F, names = extract_features(X, FS, FEATURES, ch_names=CH_2A, fc_method="corr")
    assert F.shape[0] == 6
    assert F.shape[1] == len(names) > 0
    assert np.isfinite(F).all()
    # 每个特征名前缀都出现过
    joined = " ".join(names)
    for key in ["AP[", "APb", "DE[", "SHE", "SPE", "C0", "NSI", "HOC",
                "DASM", "RASM", "AH_", "IPH_", "FC["]:
        assert key in joined, key


def test_extract_features_unknown():
    with pytest.raises(KeyError):
        extract_features(X, FS, ["not_a_feature"])
