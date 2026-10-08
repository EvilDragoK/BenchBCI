"""脑网络扩展：12 种连接方法、5 种稀疏化、协变/相似性网络、滞后矩阵、矩阵导入。

数据用合成信号，**已知正确答案**：
- 通道 0-1 强耦合（0.9 线性），2-3 中等，4-5 独立 → 耦合对权重必须显著更高；
- 通道 1 相对通道 0 延迟 20 ms → 滞后矩阵要能把 20 ms 找回来；
- ROI 1 与 ROI 0 共变 → 协变/相似性矩阵 (0,1) 必须高于 (0,4)。
"""
from __future__ import annotations

import numpy as np
import pytest

from core.netcov import (SIMILARITY_METHODS, group_covariance, individual_similarity,
                         lag_matrix, load_matrix_file, mind_matrix, save_matrix_file)
from core.networks import (CONNECT_METHODS, SPARSIFY_STRATEGIES, binarize,
                           connectivity_matrix, graph_metrics, sparsify)

SFREQ, T, C = 250.0, 500, 6


@pytest.fixture(scope="module")
def synth():
    """(N,C,T) 合成 EEG：0-1 强耦合、2-3 中等、4-5 独立。"""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, C, T))
    X[:, 1] = 0.9 * X[:, 0] + 0.1 * rng.standard_normal((8, T))
    X[:, 3] = 0.7 * X[:, 2] + 0.3 * rng.standard_normal((8, T))
    return X * 1e-5                                   # 贴近真实 EEG 电压量级


# ------------------------------------------------------------------ 连接方法
def test_all_methods_shape_and_symmetry(synth):
    E = C * (C - 1) // 2
    for m in CONNECT_METHODS:
        A = np.asarray(connectivity_matrix(synth, SFREQ, method=m))
        assert A.ndim in (2, 3), m
        A2 = A.mean(axis=0) if A.ndim == 3 else A
        if m == "hofc":
            assert A2.shape == (E, E), "HOFC 应是边×边矩阵"
        else:
            assert A2.shape == (C, C), m
        if m != "granger":
            # GCA 是**有向**指标（j→i 的因果强度），本就不对称
            assert np.allclose(A2, A2.T, atol=1e-6), f"{m} 应对称"
        assert np.isfinite(A2).all(), f"{m} 出现非有限值"


@pytest.mark.parametrize("method", ["corr", "spearman", "partial", "mi", "coh",
                                    "plv", "wcoherence", "sparse_repr", "low_rank"])
def test_coupled_pair_stands_out(synth, method):
    """零延迟耦合类方法：耦合对(0,1) 必须明显高于独立对(0,4)。"""
    A = np.asarray(connectivity_matrix(synth, SFREQ, method=method))
    A2 = A.mean(axis=0) if A.ndim == 3 else A
    coupled, separate = abs(A2[0, 1]), abs(A2[0, 4])
    assert coupled > separate * 1.5, f"{method}: 耦合 {coupled:.4f} 未明显高于独立 {separate:.4f}"


def test_pli_ignores_zero_lag(synth):
    """PLI 的设计就是**忽略零延迟**同步（对容积传导稳健）→ 相同信号不该给高值。"""
    from core.networks import pli_matrix

    dup = np.stack([synth[:, 0], synth[:, 0], synth[:, 4]], axis=1)
    P = pli_matrix(dup, SFREQ).mean(axis=0)
    assert P[0, 1] < 0.2, f"零延迟同步的 PLI 应很低，实际 {P[0,1]:.3f}"


def test_wavelet_coherence_sane(synth):
    """小波相干：相同信号≈1；独立信号在基线偏差范围内。"""
    from core.networks import wavelet_coherence_matrix

    dup = np.stack([synth[:, 0], synth[:, 0], synth[:, 4]], axis=1)
    W = wavelet_coherence_matrix(dup, SFREQ)
    assert W[0, 1] > 0.9, f"相同信号的小波相干应接近 1，实际 {W[0,1]:.3f}"
    assert 0.0 <= W[0, 2] <= 1.0 and W[0, 1] > W[0, 2]


def test_sparse_and_low_rank_are_sparse(synth):
    """稀疏表示应比低秩表示更稀疏（这是两者的核心差别）。"""
    S = np.asarray(connectivity_matrix(synth, SFREQ, method="sparse_repr"))
    L = np.asarray(connectivity_matrix(synth, SFREQ, method="low_rank"))
    assert S[0, 1] > S[0, 4] * 3, "稀疏表示的耦合对应远高于噪声对"
    assert L[0, 1] > L[0, 4] * 3, "低秩表示的耦合对应高于噪声对"
    assert (S > 1e-6).mean() < (L > 1e-6).mean() + 0.2   # 稀疏 ≤ 低秩（允许小幅波动）


def test_coh_is_bounded(synth):
    """回归：coh 曾经因为除以自谱乘积给出 1e12 这种无意义数值。"""
    A = np.asarray(connectivity_matrix(synth, SFREQ, method="coh")).mean(axis=0)
    assert A.max() <= 1.0 + 1e-6, f"相干应在 0~1，实际最大 {A.max():.3e}"


# ------------------------------------------------------------------ 稀疏化
def test_sparsify_strategies(synth):
    A = np.asarray(connectivity_matrix(synth, SFREQ, method="corr")).mean(axis=0)
    A = np.abs(A)
    tot = C * (C - 1) // 2

    none = sparsify(A, "none")
    assert (none != 0).sum() == (A != 0).sum() - C   # 只去掉了对角

    prop = sparsify(A, "proportional", density=0.3)
    edges = (prop != 0).sum() // 2
    assert abs(edges / tot - 0.3) <= 0.15, f"比例阈值应保留约 30%，实际 {edges}/{tot}"

    hard = sparsify(A, "absolute", threshold=0.5)
    assert (hard != 0).sum() // 2 <= edges + 3

    adapt = sparsify(A, "adaptive", percentile=70)
    per_node = ((adapt != 0).sum(axis=1) > 0).all()
    assert per_node, "自适应阈值应保证每个节点都留有连接"

    mst = sparsify(A, "mst")
    assert (mst != 0).sum() // 2 == C - 1, "MST 骨架应恰好保留 C−1 条边"
    assert graph_metrics(((mst != 0).astype(int)))["avg_path_length"] > 0


def test_binarize_backward_compatible(synth):
    A = np.asarray(connectivity_matrix(synth, SFREQ, method="corr")).mean(axis=0)
    B = binarize(A, 0.2)
    assert set(np.unique(B)) <= {0, 1} and B.shape == (C, C)
    assert (np.diag(B) == 0).all()


def test_sparsify_unknown_strategy(synth):
    with pytest.raises(ValueError, match="未知稀疏化策略"):
        sparsify(np.ones((3, 3)), "whatever")


# ------------------------------------------------------------------ 滞后矩阵
def test_lag_matrix_recovers_delay():
    rng = np.random.default_rng(1)
    sig = rng.standard_normal((1, 3, 2000))
    shift = int(0.020 * SFREQ)                      # 20 ms
    sig[0, 1, shift:] = sig[0, 0, :-shift]          # 通道 1 落后通道 0
    sig[0, 2] = sig[0, 0]                           # 通道 2 与 0 同步
    for interp in ("parabolic", "cubic"):
        L = lag_matrix(sig, SFREQ, max_lag_ms=80, interp=interp)
        assert L.shape == (3, 3)
        assert np.allclose(L, -L.T), "滞后矩阵应反对称"
        assert abs(np.diag(L)).max() == 0
        assert abs(L[1, 0] - 20.0) < 2.0, f"{interp}: 应找回 20 ms，实际 {L[1,0]:.2f}"
        assert abs(L[2, 0]) < 3.0, f"{interp}: 同步通道滞后应≈0，实际 {L[2,0]:.2f}"


def test_lag_matrix_rejects_short_data():
    with pytest.raises(ValueError, match="太短"):
        lag_matrix(np.zeros((2, 8)), SFREQ)


# ------------------------------------------------------------------ 协变/相似性
def test_group_covariance_detects_co_variation():
    rng = np.random.default_rng(2)
    tab = rng.standard_normal((30, 5))                  # 30 名被试 × 5 个 ROI
    tab[:, 1] = 1.2 * tab[:, 0] + 0.3 * rng.standard_normal(30)
    G = group_covariance(tab)
    assert G.shape == (5, 5) and np.allclose(G, G.T)
    assert G[0, 1] > 0.85, "共变的两个 ROI 应高相关"
    assert abs(G[0, 4]) < 0.4, "无关 ROI 应接近 0"


def test_group_covariance_needs_subjects():
    with pytest.raises(ValueError, match="至少需要 3 名被试"):
        group_covariance(np.ones((2, 4)))


@pytest.mark.parametrize("method", SIMILARITY_METHODS)
def test_individual_similarity_methods(method):
    rng = np.random.default_rng(3)
    tab = np.abs(rng.standard_normal((5, 12))) + 1.0    # ROI × 指标（非负，KL/JSD 可直接用）
    tab[1] = tab[0] * 0.95 + 0.05 * np.abs(rng.standard_normal(12))
    S = individual_similarity(tab, method=method, n_metrics=3)
    assert S.shape == (5, 5)
    assert np.allclose(S, S.T, atol=1e-9)
    assert np.allclose(np.diag(S), 1.0)
    assert S[0, 1] > S[0, 4], f"{method}: 相似 ROI 应高于无关 ROI"


def test_mind_weights_matter():
    rng = np.random.default_rng(4)
    tab = np.abs(rng.standard_normal((4, 6))) + 1.0
    tab[1] = tab[0] * 0.9
    w1 = mind_matrix(tab, n_metrics=2, weights=[1, 0])
    w2 = mind_matrix(tab, n_metrics=2, weights=[0, 1])
    assert not np.allclose(w1, w2), "指标权重不同结果应不同"
    # 权重长度与指标数不符时退化为整体相似性，不应报错
    assert mind_matrix(tab, n_metrics=5).shape == (4, 4)


# ------------------------------------------------------------------ 矩阵导入
@pytest.mark.parametrize("ext", [".npy", ".csv"])
def test_save_load_matrix_roundtrip(tmp_path, ext):
    M = np.arange(25, dtype=float).reshape(5, 5) / 7
    labels = [f"ROI{i}" for i in range(5)]
    p = tmp_path / f"net{ext}"
    save_matrix_file(p, M, labels=labels)
    M2, lab = load_matrix_file(p)
    assert M2.shape == (5, 5)
    assert np.allclose(M2, M, atol=1e-6)
    assert lab[:5] == labels


def test_load_mat_file(tmp_path):
    from scipy.io import savemat

    M = np.eye(4) * 2.5
    p = tmp_path / "conn.mat"
    savemat(p, {"A": M, "not_matrix": np.zeros(3), "labels": ["A", "B", "C", "D"]})
    M2, labels = load_matrix_file(p)
    assert M2.shape == (4, 4) and np.allclose(M2, M)
    assert labels == ["A", "B", "C", "D"]


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_matrix_file("/tmp/definitely_not_here_12345.npy")


# ------------------------------------------------------------------ 界面
def test_netmatrix_dialog_workbench(tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])   # noqa: F841

    from ui.widgets.netmatrix_dialog import NetMatrixDialog

    dlg = NetMatrixDialog()
    # ① 组水平协变（行=被试）
    rng = np.random.default_rng(5)
    tab = rng.standard_normal((20, 4))
    tab[:, 1] = tab[:, 0] * 1.1
    dlg._table = tab
    dlg.rb_table.setChecked(True)
    dlg.combo_rows.setCurrentIndex(0)
    dlg.compute()
    assert dlg._matrix is not None and dlg._matrix.shape == (4, 4)
    assert dlg.tbl.rowCount() == 4
    assert "组水平结构协变" in dlg.txt.toPlainText()

    # ② 个体相似性（行=ROI，MIND 多指标）
    dlg._table = np.abs(rng.standard_normal((5, 6))) + 1.0
    dlg.combo_rows.setCurrentIndex(1)
    dlg.combo_sim.setCurrentIndex(
        [dlg.combo_sim.itemData(i) for i in range(dlg.combo_sim.count())].index("mind"))
    dlg.spin_metrics.setValue(3)
    dlg.compute()
    assert dlg._matrix.shape == (5, 5)
    assert "个体水平相似性" in dlg.txt.toPlainText()

    # ③ 外部矩阵文件（含稀疏化）
    p = tmp_path / "external.csv"
    save_matrix_file(p, np.abs(rng.standard_normal((6, 6))), labels=[f"R{i}" for i in range(6)])
    M, _ = load_matrix_file(p)
    dlg.rb_file.setChecked(True)
    dlg._matrix = M
    dlg.combo_sparsify.setCurrentIndex(4)              # mst
    before = dlg.fig.axes[0].get_title() if dlg.fig.axes else ""
    dlg.compute()
    assert dlg._matrix.shape == (6, 6)
    assert "稀疏化后网络" in dlg.fig.axes[0].get_title()
    assert before != dlg.fig.axes[0].get_title()
