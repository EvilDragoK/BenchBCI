"""评估指标、跨折聚合与标签编码回归测试。

回归背景：预处理输出的类别码是 1..4（BNCI 约定），而模型输出索引是 0..K-1。
若不重编码，CPU 上 CrossEntropyLoss 会报 "Target 4 is out of bounds"，
MPS 上则静默越界 → 混淆矩阵最后一列恒为 0、指标失真；不同折形状不一致还会
让跨折聚合抛 "inhomogeneous shape"。
"""
from pathlib import Path

import numpy as np
import pytest

PROC = Path("processed/bnci2014001")


# --------------------------------------------------------------------- 指标
def test_compute_metrics_fixed_labels_shape():
    """显式 labels 时混淆矩阵形状固定，不随这批数据是否缺类而变。"""
    from core.train.evaluation import compute_metrics

    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]  # 没有预测出类 2
    m = compute_metrics(y_true, y_pred, ["a", "b", "c"], labels=[0, 1, 2])
    assert np.asarray(m["confusion"]).shape == (3, 3)
    assert m["per_class"]["labels"] == ["0", "1", "2"]


def test_aggregate_consistent_shapes():
    """各折用同一 labels → 聚合不再报 inhomogeneous shape。"""
    from core.train.evaluation import aggregate, compute_metrics

    folds = [
        compute_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["a", "b", "c"], labels=[0, 1, 2]),
        compute_metrics([0, 0, 1, 1], [0, 0, 0, 0], ["a", "b", "c"], labels=[0, 1, 2]),
    ]
    agg = aggregate(folds)
    assert np.asarray(agg["confusion_total"]).shape == (3, 3)
    assert agg["n_folds"] == 2


def test_export_report_without_per_subject(tmp_path):
    """per_subject 为空的实验（如 subject_kfold 留出多名被试）也要能生成报告。"""
    from core.train.evaluation import export_markdown_report

    summary = {
        "folds": [{"fold": 0, "tag": "subj-kfold-1/2[S01,S02]", "acc": 0.5, "kappa": 0.3}],
        "aggregate": {"n_folds": 1, "acc_mean": 0.5, "acc_std": 0.0,
                      "kappa_mean": 0.3, "kappa_std": 0.0,
                      "confusion_total": [[3, 1], [2, 4]]},
        "class_names": ["a", "b"],
        "per_subject": [],
    }
    out = export_markdown_report(tmp_path, summary, {"experiment": {"name": "x"}})
    text = out.read_text(encoding="utf-8")
    assert "混淆矩阵" in text and "| 真实" in text


PROC_T = PROC.parent / "bnci2014001_T"
PROC_E = PROC.parent / "bnci2014001_E"


@pytest.mark.skipif(not (PROC_T / "X.npy").exists() or not (PROC_E / "X.npy").exists(),
                    reason="需要 T/E 两份预处理数据（见 configs/train_cross_dataset.yaml）")
def test_cross_dataset_label_alignment():
    """跨数据集回归：目标域标签必须与源域做同样重编码。

    否则目标域标签会停留在原始码（1..4）、与预测（0..3）错位，
    典型症状是混淆矩阵某一行整行为 0、准确率跌到随机以下。
    """
    import shutil

    from core.train import run_training

    cfg = {
        "experiment": {"name": "unittest_cross_ds"},
        "data": {"processed_dir": str(PROC_T), "test_processed_dir": str(PROC_E)},
        "model": {"name": "csp_lda", "params": {"n_components": 4}},
        "split": {"scheme": "cross_dataset", "val_frac": 0.15, "seed": 0},
        "normalization": {"mode": "zscore_channel", "euclidean_alignment": False},
        "augmentation": {},
        "trainer": {"epochs": 1, "batch_size": 64, "device": "cpu"},
        "class_balance": {"weighted_loss": False},
    }
    created = []
    try:
        summary = run_training(cfg, log=lambda *_: None)
        created.append(summary["exp_dir"])
        conf = np.asarray(summary["aggregate"]["confusion_total"])
        assert conf.shape[0] == conf.shape[1] == 4
        assert (conf.sum(axis=1) > 0).all(), "某真实类别整行为 0：目标域标签疑未对齐"
    finally:
        for d in created:
            shutil.rmtree(d, ignore_errors=True)


def test_label_recoding_mapping():
    """1..4 的类别码应映射为 0..3 且顺序稳定。"""
    codes = [1, 2, 3, 4]
    y = np.array([4, 1, 3, 2, 2, 4])
    y2 = np.searchsorted(codes, y).astype(np.int64)
    assert sorted(set(y2.tolist())) == [0, 1, 2, 3]
    assert y2.tolist() == [3, 0, 2, 1, 1, 3]


# --------------------------------------------------------- 端到端（需数据）
@pytest.mark.skipif(not (PROC / "X.npy").exists(),
                    reason="需要先运行预处理流水线（processed/bnci2014001）")
def test_eegnet_training_labels_encoded_end_to_end():
    """CPU 上跑一折 EEGNet：标签未重编码会直接崩；修复后每列都有预测。"""
    import shutil

    from core.train import run_training

    cfg = {
        "experiment": {"name": "unittest_label_encode"},
        "data": {"processed_dir": str(PROC)},
        "model": {"name": "eegnet",
                  "params": {"F1": 4, "F2": 8, "D": 2, "kernel_length": 64}},
        "split": {"scheme": "few_shot", "n_shot": 8, "n_repeats": 1,
                  "test_session": "T", "val_frac": 0.0, "seed": 0},
        "normalization": {"mode": "zscore_channel", "euclidean_alignment": False},
        "augmentation": {},
        "trainer": {"epochs": 1, "batch_size": 16, "device": "cpu"},
        "class_balance": {"weighted_loss": True},
    }
    created = []
    try:
        summary = run_training(cfg, log=lambda *_: None)
        created.append(summary["exp_dir"])
        conf = np.asarray(summary["aggregate"]["confusion_total"])
        assert conf.shape[0] == conf.shape[1] == 4
        # 标签错位时最后一列恒为 0 —— 这是本次 bug 的典型症状
        assert (conf.sum(axis=0) > 0).all(), "存在从未被预测的类别，疑标签编码错误"
        assert 0.0 <= summary["aggregate"]["acc_mean"] <= 1.0
    finally:
        for d in created:
            shutil.rmtree(d, ignore_errors=True)
