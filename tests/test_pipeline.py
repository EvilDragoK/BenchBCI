"""流水线单测：合成数据、确定性（GUI 与命令行同构保证）。"""
import numpy as np
import pytest

import mne


def _make_raw(n_trials=20, fs=250.0, n_ch=4, seed=0):
    rng = np.random.default_rng(seed)
    n_times = int(fs * 40)
    data = rng.standard_normal((n_ch, n_times)) * 50e-6
    info = mne.create_info([f"E{i + 1}" for i in range(n_ch)], fs, "eeg")
    raw = mne.io.RawArray(data, info)
    onsets = np.arange(1, n_trials + 1)  # 保证 tmin/tmax 窗口不越界
    desc = ["a" if i % 2 == 0 else "b" for i in range(n_trials)]
    raw.set_annotations(mne.Annotations(onsets, np.zeros(n_trials), desc))
    return raw


CFG = [
    {"op": "bandpass_filter", "params": {"l_freq": 4.0, "h_freq": 40.0},
     "enabled": True},
    {"op": "re_reference", "params": {"mode": "average"}, "enabled": True},
    {"op": "epoching", "params": {"tmin": 0.0, "tmax": 2.0, "baseline": False},
     "enabled": True},
    {"op": "amplitude_reject", "params": {"threshold_uv": 1000.0}, "enabled": True},
]


def test_pipeline_shapes_and_labels():
    from core.pipeline import Pipeline

    raw = _make_raw()
    pipe = Pipeline.from_config(CFG)
    epochs, ctx = pipe.run(raw.copy())
    assert len(epochs) == 20
    assert epochs.get_data().shape == (20, 4, 501)
    assert set(epochs.events[:, 2]) == {1, 2}
    assert ctx["classes"] == {"a": 1, "b": 2}


def test_pipeline_deterministic():
    from core.pipeline import Pipeline

    outs = []
    for _ in range(2):
        pipe = Pipeline.from_config(CFG)
        epochs, _ = pipe.run(_make_raw(seed=3).copy())
        outs.append(epochs.get_data())
    np.testing.assert_array_equal(outs[0], outs[1])


def test_disabled_step_skipped():
    from core.pipeline import Pipeline

    cfg = [dict(CFG[0], enabled=False), CFG[2]]
    pipe = Pipeline.from_config(cfg)
    epochs, ctx = pipe.run(_make_raw().copy())
    assert len(epochs) == 20
    assert ctx["log"][0].get("skipped") is True


def test_splits_no_overlap():
    from core.train.splits import make_splits

    # 每名被试同时含 T 与 E session（逐被试 T→E 协议）
    y = np.repeat([0, 1, 2, 3], 10)
    sid = np.array([1, 2] * 20)
    sess = np.array(["T"] * 20 + ["E"] * 20)
    sp = make_splits(y, sid, sess, scheme="session")
    assert len(sp) == 2
    for s in sp:
        assert not np.intersect1d(s.train, s.test).size
    sp = make_splits(y, sid, np.array(["T"] * 40), scheme="kfold", k=4)
    assert len(sp) == 8  # 2 名被试 × 4 折（逐被试切分，不同被试绝不混切）
    for s in sp:
        assert not np.intersect1d(s.train, s.test).size
        assert np.all(sid[s.train] == sid[s.test][0])  # 训练/测试同属一名被试
    sp = make_splits(y, sid, np.array(["T"] * 40), scheme="loso")
    assert len(sp) == 2  # 2 名被试
    for s in sp:
        assert np.all(sid[s.train] != sid[s.test][0])  # LOSO：训练集不含测试被试
    # within：逐被试按比例随机划分（每人内部随机切 1/k，重复 n_repeats 次）
    sp = make_splits(y, sid, np.array(["T"] * 40), scheme="within", k=4, n_repeats=2)
    assert len(sp) == 4  # 2 名被试 × 2 次重复
    for s in sp:
        assert not np.intersect1d(s.train, s.test).size
        assert np.all(sid[s.train] == sid[s.test][0])  # 同一被试内部划分
        total = len(s.train) + len(s.val) + len(s.test)
        assert abs(len(s.test) / total - 0.25) < 0.02  # 测试占比 1/k
    # subject_kfold：跨被试分组 k 折（每次留出一整组被试）
    sp = make_splits(y, sid, np.array(["T"] * 40), scheme="subject_kfold", k=2)
    assert len(sp) == 2
    for s in sp:
        assert np.all(sid[s.train] != sid[s.test][0])  # 训练集不含测试被试
    # holdout：测试集应占 1/k（k=5 → 20%），而不是反过来
    sp = make_splits(y, sid, np.array(["T"] * 40), scheme="holdout", k=5)
    assert len(sp) == 1
    assert abs(len(sp[0].test) / len(y) - 0.2) < 0.06


def test_within_default_one_fold_per_subject():
    """within 默认「每名被试一折」（对齐 EEG-FM-Benchmark），而非 被试数×重复次数。"""
    from core.train.splits import make_splits

    y = np.repeat([0, 1, 2, 3], 10)
    sid = np.repeat([1, 2], 20)      # 2 名被试，各 20 trial（每类 5 个）
    sess = np.array(["T"] * 40)

    sp = make_splits(y, sid, sess, scheme="within")          # 默认 n_repeats=1
    assert len(sp) == 2                                      # = 被试数
    for s in sp:
        assert not np.intersect1d(s.train, s.test).size
        assert np.all(sid[s.train] == sid[s.test][0])        # 同一被试内部划分

    # 比例模式：每类取 50% 作训练 → 测试样本应多于训练
    sp = make_splits(y, sid, sess, scheme="within", train_percentage=0.5)
    assert len(sp) == 2
    for s in sp:
        assert len(s.test) > len(s.train)

    # n_repeats>1 才产生重复采样：折数 = 被试数 × n_repeats
    sp = make_splits(y, sid, sess, scheme="within", n_repeats=3)
    assert len(sp) == 6


def test_splits_new_protocols():
    """custom / block / session_kfold / cross_dataset / nested_cv 的划分正确性。"""
    import pytest

    from core.train.splits import make_splits

    n_sub, per = 4, 40
    y = np.tile(np.repeat([0, 1, 2, 3], per // 4), n_sub)
    sid = np.repeat([1, 2, 3, 4], per)
    sess = np.tile(["T"] * (per // 2) + ["E"] * (per // 2), n_sub)
    block = np.concatenate([np.repeat([i * 2, i * 2 + 1], per // 2) for i in range(n_sub)])

    # ---- custom：val 留空 → 自动从 train 划出；显式 val 生效；非法输入报错
    folds = [{"name": "f1", "train": [1, 2], "val": [], "test": [3]},
             {"name": "f2", "train": [3, 4], "val": [1], "test": [2]}]
    sp = make_splits(y, sid, sess, scheme="custom", custom_folds=folds)
    assert len(sp) == 2
    assert len(sp[0].val) > 0                       # 自动划出验证集
    assert set(sid[sp[1].val].tolist()) == {1}      # 显式指定的验证受试者
    for s in sp:
        assert not np.intersect1d(s.train, s.test).size
        assert not np.intersect1d(s.val, s.test).size
    with pytest.raises(ValueError):
        make_splits(y, sid, sess, scheme="custom",
                    custom_folds=[{"name": "bad", "train": [99], "test": [1]}])
    with pytest.raises(ValueError):
        make_splits(y, sid, sess, scheme="custom",
                    custom_folds=[{"name": "ov", "train": [1], "test": [1]}])

    # ---- block：按记录块划分；缺 block 信息时报错
    sp = make_splits(y, sid, sess, scheme="block", block=block, k=2)
    assert len(sp) == n_sub * 2
    for s in sp:
        assert not np.intersect1d(s.train, s.test).size
        assert set(block[s.train].tolist()).isdisjoint(set(block[s.test].tolist()))
    with pytest.raises(ValueError):
        make_splits(y, sid, sess, scheme="block", block=None)

    # ---- session_kfold：同人跨 session 双向（4 人 × 2 方向）
    sp = make_splits(y, sid, sess, scheme="session_kfold")
    assert len(sp) == n_sub * 2
    for s in sp:
        assert np.all(sid[s.train] == sid[s.test][0])

    # ---- cross_dataset：按域切开（前 80 训练域 / 后 80 目标域）
    sp = make_splits(y, sid, sess, scheme="cross_dataset", domain_sizes=(80, 80))
    assert len(sp) == 1
    assert sp[0].train.max() < 80 and sp[0].test.min() >= 80
    with pytest.raises(ValueError):
        make_splits(y, sid, sess, scheme="cross_dataset", domain_sizes=None)

    # ---- nested_cv：外层折与 kfold 同构（内层选参在 run_training 内完成）
    sp = make_splits(y, sid, sess, scheme="nested_cv", k=2)
    assert len(sp) == n_sub * 2
    assert all("nested" in s.tag for s in sp)


def test_normalization_no_leakage():
    from core.datasets import apply_normalization, compute_channel_stats

    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((50, 4, 100)).astype(np.float32) * 2 + 1
    Xte = rng.standard_normal((10, 4, 100)).astype(np.float32)
    mean, std = compute_channel_stats(Xtr)
    stats = {"mean": mean, "std": std}
    Z = apply_normalization(Xte, "zscore_channel", stats)
    # 测试集用训练统计量 → 不应恰好零均值单位方差（泄漏检测的朴素哨兵）
    assert not np.allclose(Z.mean(), 0.0, atol=1e-6)
    Ztr = apply_normalization(Xtr, "zscore_channel", stats)
    np.testing.assert_allclose(Ztr.mean(), 0.0, atol=1e-5)


def test_euclidean_alignment():
    from core.datasets import euclidean_alignment

    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3))
    X = rng.standard_normal((30, 3, 200))
    Xa = A @ X  # 人为加一个公共空间混合
    out = euclidean_alignment(Xa, shrink=0.0)
    cov = np.einsum("nct,nkt->nck", out, out) / out.shape[2]
    np.testing.assert_allclose(cov.mean(0), np.eye(3), atol=1e-8)
