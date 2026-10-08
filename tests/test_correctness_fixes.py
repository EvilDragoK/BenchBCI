"""正确性回归测试：针对一次全量审计中确认并修复的实现错误（每条都对应旧行为）。

覆盖：
1. 划分泄漏        —— session / loso 曾把验证集留在训练集内
2. PhysioNet 映射  —— run→任务 曾按"相邻两两"分组（半数样本标签是错的）
3. BNCI 电极表     —— 曾多出 CP5/CP6、缺 P2/POz，导致 13 号起整体错位
4. EA 逐被试       —— 曾在全体被试上做一次全局对齐
5. 跨折 ckpt 复用  —— 本折无验证集时曾载入上一折的 best 权重
6. 样本熵定义      —— B 的模板数曾与 A 不一致；A=0 曾静默返回 0（方向相反）
7. minmax 常数通道 —— 曾除零得到 inf/nan
8. MI/PLI 二维输入 —— 文档承诺支持 (C,T) 但实现会崩
9. CLI 退出码      —— 成功也曾以退出码 1 结束
10. 生物标志物校正 —— 逐边检验曾无多重比较校正
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ 1 划分泄漏
def _dataset(n_subj=3, n_class=4, per_class=15, sessions=("T", "E")):
    y, sid, sess = [], [], []
    for s in range(1, n_subj + 1):
        for sn in sessions:
            for c in range(n_class):
                y += [c] * per_class
                sid += [s] * per_class
                sess += [sn] * per_class
    return np.array(y), np.array(sid), np.array(sess)


def test_no_val_leak_in_any_scheme():
    """所有划分方案都必须满足 train ∩ val = val ∩ test = train ∩ test = ∅。"""
    from core.train.splits import make_splits

    y, sid, sess = _dataset()
    n = len(y)
    block = np.repeat(np.arange(6), n // 6 + 1)[:n]
    cases = [
        ("session", {}), ("loso", {}), ("session_kfold", {}),
        ("kfold", {"k": 3}), ("holdout", {"k": 3}), ("within", {"k": 3}),
        ("few_shot", {"n_shot": 5}), ("subject_kfold", {"k": 3}),
        ("block", {"block": block}),
        ("custom", {"custom_folds": [{"name": "f1", "train": [1, 2], "test": [3]}]}),
    ]
    for scheme, kw in cases:
        splits = make_splits(y, sid, sess, scheme, **kw)
        assert splits, f"{scheme} 没有产生折"
        for sp in splits:
            assert not np.intersect1d(sp.train, sp.val).size, f"{scheme}: 验证集泄漏进训练集"
            assert not np.intersect1d(sp.val, sp.test).size, f"{scheme}: val/test 重叠"
            assert not np.intersect1d(sp.train, sp.test).size, f"{scheme}: train/test 重叠"


def test_session_and_loso_carve_val_out_of_train():
    """具体数值检查：val 非空且训练集确实少掉了 val 那些样本。"""
    from core.train.splits import make_splits

    y, sid, sess = _dataset(n_subj=3)
    for scheme in ("session", "loso"):
        for sp in make_splits(y, sid, sess, scheme, val_frac=0.2):
            assert len(sp.val) > 0
            assert len(sp.train) + len(sp.val) > 0
            # 训练 + 验证 应恰好等于"该折可用训练样本"（无重复计数）
            assert len(np.union1d(sp.train, sp.val)) == len(sp.train) + len(sp.val)


# ------------------------------------------------------- 2 PhysioNet run→任务
def test_physionet_run_task_mapping():
    from core.io.physionet import DEFAULT_RUNS, RUN_TASKS

    assert RUN_TASKS[5] == ("real", "both_fists_real", "both_feet_real")   # 旧实现写成 imagined
    assert RUN_TASKS[9] == ("real", "both_fists_real", "both_feet_real")
    assert RUN_TASKS[4] == RUN_TASKS[8] == RUN_TASKS[12]
    assert all(RUN_TASKS[r][0] == "imagined" for r in DEFAULT_RUNS)


# ------------------------------------------------------------- 3 BNCI 电极表
def test_bnci_channel_names_official():
    """官方 22 导（MOABB `_load_data_001_2014`）：无 Oz、无 CP5/CP6；含 FCz、P2、POz。"""
    from core.io.bnci import EEG_CH_NAMES, EOG_CH_NAMES

    assert EEG_CH_NAMES == [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
        "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
    ]
    assert len(EEG_CH_NAMES) == 22 and len(set(EEG_CH_NAMES)) == 22
    for bad in ("CP5", "CP6", "Oz"):
        assert bad not in EEG_CH_NAMES, f"{bad} 不在官方 2a 导联里"
    assert EOG_CH_NAMES == ["EOG1", "EOG2", "EOG3"]


def test_bnci_montage_all_matched():
    """22 个官方导联名都必须在 standard_1005 里有坐标（否则地形图会缺电极）。"""
    import mne

    from core.io.bnci import EEG_CH_NAMES

    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    missing = [c for c in EEG_CH_NAMES if c not in pos]
    assert not missing, f"这些导联在 standard_1005 中找不到坐标：{missing}"


# ---------------------------------------------------------------- 4 EA 逐被试
def test_ea_per_subject_normalizes_each_subject():
    """逐被试 EA 后，每名被试自己的协方差都应≈单位阵（全局 EA 做不到这点）。"""
    from core.datasets import euclidean_alignment

    rng = np.random.default_rng(0)
    A = rng.standard_normal((40, 6, 128)) * 3.0          # 被试 1：幅值大
    B = rng.standard_normal((40, 6, 128)) * 0.4          # 被试 2：幅值小
    sid = np.array([1] * 40 + [2] * 40)
    X = np.concatenate([A, B])

    per = X.copy()
    for s in (1, 2):
        m = sid == s
        per[m] = euclidean_alignment(per[m])

    def cov_err(Z):
        v = Z.transpose(1, 0, 2).reshape(Z.shape[1], -1)
        C = np.cov(v)
        return float(np.abs(C - np.eye(Z.shape[1])).max())

    for s in (1, 2):
        assert cov_err(per[sid == s]) < 0.15, f"被试 {s} 对齐后协方差偏离单位阵"
    # 全局一次性对齐：至少有一名被试仍然明显偏离单位阵（说明旧写法确实错）
    glob = euclidean_alignment(X)
    assert max(cov_err(glob[sid == s]) for s in (1, 2)) > 0.15


# ------------------------------------------------------- 5 跨折 ckpt 不能复用
def test_best_ckpt_not_reused_across_folds(tmp_path):
    from core.train import _drop_stale_best, _restore_best_if_any

    class _FakeTrainer:
        def __init__(self):
            self.calls = []

        def load_ckpt(self, name="ckpt_best.pt"):
            self.calls.append(name)

    ckpt = tmp_path / "ckpt_best.pt"
    t = _FakeTrainer()
    # 本折没有验证集（fit 返回 -1.0）→ 绝不能载入
    assert _restore_best_if_any(t, -1.0, ckpt) is False
    # 本折有 best 但没写出文件 → 不载入
    assert _restore_best_if_any(t, 0.9, ckpt) is False
    assert t.calls == []
    # 本折选出 best 且文件在 → 载入
    ckpt.write_text("x")
    assert _restore_best_if_any(t, 0.9, ckpt) is True
    assert t.calls == ["ckpt_best.pt"]
    # 新折开始前必须把上一折残留的 best 删掉
    assert _drop_stale_best(tmp_path) is True and not ckpt.exists()
    assert _drop_stale_best(tmp_path) is False


# ---------------------------------------------------------------- 6 样本熵
def test_sample_entropy_definition():
    from core.nonlinear import sample_entropy

    # B 与 A 必须同源（都基于 i=1..N-m 模板）→ 手工核对一个可解析的小例子
    x = np.array([0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0])
    B, A = 3, 0                       # 长度 2 的 [0,1] 出现 3 次；长度 3 全不同
    got = sample_entropy(x, m=2, r=1e-9)
    assert got == pytest.approx(np.log(B), abs=1e-9)   # A=0 → 理论上界 log(B)，不是 0
    assert got > 0

    # 过长/过短信号分别给出有限值 / nan（不再静默当 0）
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(300)
    sine = np.sin(2 * np.pi * 10 * np.arange(300) / 100)
    assert sample_entropy(noise) > sample_entropy(sine)     # 噪声更不规则
    assert np.isnan(sample_entropy(np.array([1.0, 2.0])))
    assert sample_entropy(np.ones(200)) == 0.0              # 常数 → 0（合法）


def test_entropy_channels_handles_nan():
    from core.nonlinear import entropy_channels

    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 3, 120))
    X[0, 1, :] = 0.0                     # 常数通道
    ent = entropy_channels(X, methods=("sample", "approximate", "permutation"))
    for v in ent.values():
        assert v.shape == (3,) and np.isfinite(v).all()


# ------------------------------------------------------------- 7 minmax 除零
def test_minmax_constant_channel_is_finite():
    from core.datasets import apply_normalization

    X = np.ones((4, 3, 32), dtype=np.float32)      # 全常数 → hi-lo = 0
    for mode in ("minmax_channel", "minmax_pm", "minmax_epoch", "minmax_pm_epoch"):
        out = apply_normalization(X, mode)
        assert np.isfinite(out).all(), f"{mode} 出现 inf/nan"


# ------------------------------------------------------- 8 连接矩阵二维输入
@pytest.mark.parametrize("method", ["corr", "spearman", "partial", "mi", "coh", "plv", "pli",
                                    "wcoherence", "sparse_repr", "low_rank", "granger"])
def test_connectivity_accepts_2d(method):
    """模块文档承诺支持单段 (C,T) 输入 → 不应抛异常，输出 (C,C)。"""
    from core.networks import connectivity_matrix

    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 400))              # (C,T)
    A = connectivity_matrix(X, 250.0, method=method, fmin=8, fmax=30)
    assert A.shape == (5, 5) and np.isfinite(A).all()


# ---------------------------------------------------------------- 9 CLI 退出码
def test_cli_exit_code_zero_on_success():
    r = subprocess.run([sys.executable, "-m", "core.cli", "list"], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"成功命令退出码应为 0，实际 {r.returncode}；stderr={r.stderr[:200]}"


# --------------------------------------------------------- 10 生物标志物校正
def test_matrix_roundtrip_all_formats(tmp_path):
    """回归：save_matrix_file 写 .mat 曾落到 CSV 分支 → 后缀 .mat 的文本文件，
    再用 loadmat 读直接报 "Unknown mat file type"（往返静默损坏）。"""
    from core.netcov import load_matrix_file, save_matrix_file

    rng = np.random.default_rng(0)
    M = rng.random((6, 6))
    labels = [f"ROI{i}" for i in range(6)]
    for ext in (".npy", ".csv", ".mat"):
        p = tmp_path / f"conn{ext}"
        save_matrix_file(p, M, labels=labels)
        M2, lab2 = load_matrix_file(p)
        np.testing.assert_allclose(M2, M, atol=1e-6)
        assert M2.shape == (6, 6)
        assert lab2[:6] == labels, f"{ext} 的 ROI 标签没被保留"


@pytest.mark.parametrize("shape", [(6, 4, 64), (6, 1, 4, 64), (5, 7, 100), (1, 3, 32)])
def test_augment_batch_shapes(shape):
    """回归：augment_batch 的三个分支都曾对不上形状。

    - 时间平移：`idx.expand(*x.shape[:-1], T)` 只在 batch == 通道数时碰巧成立，
      否则抛 "expanded size of the tensor must match…"；4D 输入同样错。
    - 高斯噪声：4D 分支形状算错，相加时广播成 5 维张量。
    - generator：只收 torch.Generator（int / numpy Generator 直接崩）。
    """
    from core.datasets import augment_batch

    rng = np.random.default_rng(0)
    X = rng.standard_normal(shape).astype("float32")
    for gen in (None, 1, np.random.default_rng(0)):
        for kw in ({"time_shift": 4}, {"noise_std": 0.5}, {"channel_dropout": 0.5},
                   {"time_shift": 4, "noise_std": 0.5, "channel_dropout": 0.5}):
            out = np.asarray(augment_batch(X, generator=gen, **kw))
            assert out.shape == shape, f"{kw} 改变了形状：{out.shape}"
            assert np.isfinite(out).all()


def test_augment_time_shift_is_circular_roll():
    """时间平移必须是「循环移位」（每个样本一个随机位移，位移在 ±time_shift 内）。"""
    import torch

    from core.datasets import augment_batch

    rng = np.random.default_rng(0)
    for shape in ((4, 3, 40), (4, 1, 3, 40)):
        X = rng.standard_normal(shape).astype("float32")
        out = np.asarray(augment_batch(X, time_shift=5,
                                       generator=torch.Generator().manual_seed(0)))
        assert out.shape == shape
        for b in range(shape[0]):                 # 逐样本找位移 k：out == roll(x, k)
            hits = [k for k in range(-5, 6)
                    if np.allclose(np.roll(X[b], k, axis=-1), out[b])]
            assert hits, f"{shape} 第 {b} 个样本不是任何循环移位（时间平移实现不对）"


def test_augment_noise_and_dropout_take_effect():
    from core.datasets import augment_batch

    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 6, 64)).astype("float32")
    noisy = np.asarray(augment_batch(X, noise_std=0.5, generator=1))
    assert not np.allclose(noisy, X)
    # 噪声幅度应大致与 noise_std × 每样本标准差 同量级
    got = float(np.std(noisy - X))
    want = 0.5 * float(np.std(X))
    assert 0.4 * want < got < 2.5 * want, f"噪声幅度异常：{got:.3f} vs {want:.3f}"
    dropped = np.asarray(augment_batch(X, channel_dropout=0.9, generator=1))
    assert (dropped == 0).any(), "通道 dropout 没有把任何通道置零"
    # 同一 seed 可复现（三种 generator 写法都行）
    a = np.asarray(augment_batch(X, time_shift=4, noise_std=0.5, generator=7))
    b = np.asarray(augment_batch(X, time_shift=4, noise_std=0.5, generator=7))
    np.testing.assert_allclose(a, b)


def test_model_short_window_error_is_actionable():
    """回归：短时窗曾抛出难懂的 BatchNorm 报错；现在要给出可操作提示。"""
    from core.models import build_model

    with pytest.raises(ValueError) as ei:
        build_model("deepconvnet", 4, 8, 64, {})
    msg = str(ei.value)
    assert "最小时长" in msg and "n_times" in msg


def test_check_code_explains_head_mismatch():
    """回归：分类头与特征维数不一致时，报错要翻译成可操作的中文提示。"""
    from core.models.custom import check_code, template_source

    code = template_source("eegnet", "probe_head_mismatch")
    head, _, tail = code.rpartition("n_classes)")
    res = check_code(head + "n_classes + 1)" + tail, n_classes=4, n_ch=8, n_times=256, batch=2)
    assert not res["ok"]
    joined = " ".join(res["errors"])
    assert "输出维度不对" in joined or "分类头" in joined


def test_biomarkers_p_values_corrected():
    from core.brainnet import biomarkers_by_condition

    rng = np.random.default_rng(0)
    X = rng.standard_normal((24, 6, 200))
    labels = ["a"] * 12 + ["b"] * 12
    X[12:, 1] = 0.9 * X[12:, 2] + 0.1 * rng.standard_normal((12, 200))   # 注入判别连接
    r = biomarkers_by_condition(X, labels, 250.0, method="corr", top_n=8)

    assert r["p_corrected"].shape == (6, 6)
    # 校正后 p 不小于原始 p（FDR/Bonferroni 都是放大）
    upper = np.triu_indices(6, 1)
    assert (r["p_corrected"][upper] >= r["p_val"][upper] - 1e-12).all()
    assert r["n_significant"] >= 1
    assert r["correct_method"] == "fdr_bh"
    # 注入的连接应在 Top 里，且位置对称
    pairs = [{i, j} for i, j, *_ in r["top_connections"]]
    assert {1, 2} in pairs
    np.testing.assert_allclose(r["p_corrected"], r["p_corrected"].T)
