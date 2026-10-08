"""模型前向形状单测（无需数据集）。"""
import numpy as np
import pytest
import torch

from core.models import MODEL_NAMES, build_model, is_classic


@pytest.mark.parametrize("name", [m for m in MODEL_NAMES if not is_classic(m)])
def test_forward_shapes(name):
    n_ch, n_times, n_classes, B = 22, 501, 4, 3
    params = {}
    if name == "eegnet":
        params = {"kernel_length": 64}  # 小核加快测试
    if name == "conformer":
        params = {"d_model": 32, "depth": 1, "heads": 4}
    model = build_model(name, n_classes, n_ch, n_times, params)
    x = torch.randn(B, 1, n_ch, n_times)
    out = model(x)
    assert out.shape == (B, n_classes)
    out.sum().backward()  # 反传可用


def test_classic_baseline():
    from core.models.sklearn_baselines import ClassicBaseline

    rng = np.random.default_rng(0)
    N, C, T = 60, 8, 100
    X = rng.standard_normal((N, C, T)).astype(np.float32)
    y = rng.integers(0, 2, N)
    clf = ClassicBaseline(kind="csp_lda", n_components=2)
    clf.fit(X, y)
    acc = clf.score(X, y)
    assert 0.0 <= acc <= 1.0
