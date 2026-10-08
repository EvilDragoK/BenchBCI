"""Few-Shot 学习曲线单测（依赖已预处理数据）。"""
import shutil
from pathlib import Path

import numpy as np
import pytest

PROC = Path("processed/bnci2014001")
pytestmark = pytest.mark.skipif(not (PROC / "X.npy").exists(),
                                reason="需要先运行预处理流水线（processed/bnci2014001）")


def _cfg(n_repeats=2):
    return {
        "experiment": {"name": "unittest_curve", "note": ""},
        "data": {"processed_dir": str(PROC)},
        "model": {"name": "csp_lda", "params": {"n_components": 4}},
        "split": {"scheme": "few_shot", "n_shot": 4, "n_repeats": n_repeats,
                  "test_session": "T", "val_frac": 0.0, "seed": 42},
        "normalization": {"mode": "zscore_channel", "euclidean_alignment": False},
        "augmentation": {},
        "trainer": {"epochs": 1, "batch_size": 64, "device": "cpu"},
        "class_balance": {"weighted_loss": False},
    }


def test_few_shot_curve_structure():
    from core.train import run_few_shot_curve

    created = []
    try:
        curve = run_few_shot_curve(_cfg(), shots=(4, 8), log=lambda *_: None, save=False)
        assert curve["shots"] == [4, 8]
        assert len(curve["results"]) == 2
        for r in curve["results"]:
            assert 0.0 <= r["acc_mean"] <= 1.0
            assert r["n_folds"] > 0
            assert r["n_subjects"] > 0
            assert r["per_subject_acc_mean"] is not None
            assert 0.0 <= r["per_subject_acc_mean"] <= 1.0
            if r.get("exp_dir"):
                created.append(r["exp_dir"])
        # 两个 shot 的折数应一致（同一批被试 × 同一重复次数）
        assert curve["results"][0]["n_folds"] == curve["results"][1]["n_folds"]
    finally:
        for d in created:
            shutil.rmtree(d, ignore_errors=True)


def test_plot_few_shot_curve():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from core.train import plot_few_shot_curve

    curve = {"shots": [1, 5, 10], "acc_mean": [0.30, 0.40, 0.45],
             "acc_std": [0.05, 0.04, 0.03],
             "per_subject_acc_mean": [0.32, 0.41, 0.46],
             "per_subject_acc_std": [0.01, 0.01, 0.01]}
    fig, ax = plt.subplots()
    plot_few_shot_curve(curve, ax)
    assert "n_shot" in ax.get_xlabel()
    assert "准确率" in ax.get_ylabel()
    plt.close(fig)


def test_plot_few_shot_curve_empty():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from core.train import plot_few_shot_curve

    fig, ax = plt.subplots()
    plot_few_shot_curve({"shots": [], "acc_mean": []}, ax)  # 不应抛异常
    plt.close(fig)
