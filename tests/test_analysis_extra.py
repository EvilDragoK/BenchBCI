"""分析补齐项单测：ERP 多条件与统计、时频方法/ERD-ERS/地形图/统计、PSD 方法与导出。"""
import os

import numpy as np
import pytest

from core.erp import (baseline_correct, component_condition_stats, erp_conditions,
                      single_trial_components)
from core.features import psd, psd_table, relative_power
from core.timefreq import (band_tfr_power, band_window_topomap, compute_tfr,
                           tfr_baseline, tfr_channel_stats, tfr_to_rows)

FS = 250.0
rng = np.random.default_rng(0)


def _erp_data(n_per_cond=40, C=4, T=250, amp=25e-6, seed=7):
    """两条件合成 ERP：target 的 P3（0.25-0.5 s）更强（固定种子，测试可复现）。"""
    rng_local = np.random.default_rng(seed)
    times = np.arange(T) / FS - 0.2
    rows, labels = [], []
    for i in range(n_per_cond * 2):
        cond = i % 2
        x = rng_local.normal(0, 3e-6, (C, T)) + 2e-6 * np.sin(2 * np.pi * 10 * times)
        a = amp if cond else 5e-6
        x[:, (times >= 0.25) & (times <= 0.5)] += a
        rows.append(x)
        labels.append("target" if cond else "std")
    return np.stack(rows), labels, times


# ---------------------------------------------------------------- ERP
def test_erp_conditions_and_counts():
    X, labels, _ = _erp_data(n_per_cond=10)
    waves, counts = erp_conditions(X, labels)
    assert set(waves) == {"target", "std"} and set(counts) == {"target", "std"}
    assert waves["target"].shape == (4, 250) and counts["std"] == 10
    with pytest.raises(ValueError):
        erp_conditions(X, labels, conds=["nope"])


def test_baseline_correct_zeroes_baseline():
    X, _, times = _erp_data(n_per_cond=5)
    Xb = baseline_correct(X, times, baseline=(-0.2, 0.0))
    m = (times >= -0.2) & (times <= 0.0)
    assert np.abs(Xb[..., m].mean(axis=-1)).max() < 1e-15
    with pytest.raises(ValueError):
        baseline_correct(X, times, baseline=(5.0, 6.0))


def test_single_trial_components_shapes():
    X, labels, times = _erp_data(n_per_cond=6)
    st = single_trial_components(times, X)
    assert set(st) == {"N1", "P2", "N2", "P3"}
    for v in st.values():
        assert v["amplitude"].shape == (12, 4)
        assert v["latency_ms"].shape == (12, 4)
    # 正性成分 P3 只在 0.25-0.5 s 内取到
    lat = st["P3"]["latency_ms"]
    assert ((lat >= 250) & (lat <= 500)).all()


def test_component_condition_stats_detects_p3():
    X, labels, times = _erp_data(n_per_cond=30)
    res = component_condition_stats(times, X, labels, paired=False, correct_method="fdr_bh")
    assert res["conds"] == ("std", "target")
    p3 = res["components"]["P3"]["amplitude"]
    assert (p3["t"] < -5).all()                     # t = 均值[std] - 均值[target] < 0
    assert (p3["p_corrected"] < 0.001).all()
    assert (p3["d"] < -1).all()
    # 无差异的 N1 效应量应远小于 P3
    n1 = res["components"]["N1"]["amplitude"]
    assert (np.abs(n1["d"]) < 1.0).all()
    assert (np.abs(n1["d"]) < np.abs(p3["d"])).all()
    with pytest.raises(ValueError):                 # 只有一个条件 → 报错
        component_condition_stats(times, X, ["a"] * len(X))


# ---------------------------------------------------------------- 时频
@pytest.mark.parametrize("method", ["morlet", "multitaper", "stft"])
def test_compute_tfr_methods(method):
    X, _, times = _erp_data(n_per_cond=8, T=250)
    P, f, t = compute_tfr(X, FS, times=times, method=method)
    assert P.shape[0] == 16 and P.shape[1] == 4 and P.shape[2] == len(f)
    assert P.shape[3] == len(t)
    assert (P >= 0).all()
    assert f.min() >= 1.0 and f.max() <= 45.0
    if method != "stft":
        assert len(t) == len(times)                  # 与输入等长


def test_compute_tfr_unknown_method():
    X, _, times = _erp_data(n_per_cond=2, T=200)
    with pytest.raises(ValueError):
        compute_tfr(X, FS, times=times, method="nope")


def test_tfr_baseline_modes():
    X, _, times = _erp_data(n_per_cond=6, T=250)
    P, f, t = compute_tfr(X, FS, times=times, method="morlet")
    bm = (t >= -0.2) & (t <= 0.0)
    Bers = tfr_baseline(P, t, baseline=(-0.2, 0.0), mode="ers")
    assert abs(float(Bers[..., bm].mean())) < 1e-9              # (P-P0)/P0 在基线上均值为 0
    Bratio = tfr_baseline(P, t, baseline=(-0.2, 0.0), mode="ratio")
    assert abs(float(Bratio[..., bm].mean()) - 1.0) < 1e-9      # 比值在基线上均值为 1
    Bdb = tfr_baseline(P, t, baseline=(-0.2, 0.0), mode="db")
    # dB 是几何量：还原回比值后基线上应恰好为 1
    assert abs(float((10 ** (Bdb[..., bm] / 10)).mean()) - 1.0) < 1e-6
    Bz = tfr_baseline(P, t, baseline=(-0.2, 0.0), mode="z")
    assert abs(float(Bz[..., bm].mean())) < 1e-6
    assert tfr_baseline(P, t, mode="none").shape == P.shape
    with pytest.raises(ValueError):
        tfr_baseline(P, t, baseline=(99.0, 100.0))


def test_band_tfr_power_and_topomap():
    X, _, times = _erp_data(n_per_cond=6, T=250)          # 12 个 epoch
    P, f, t = compute_tfr(X, FS, times=times, method="morlet")
    bp = band_tfr_power(P, f)
    assert "alpha" in bp and bp["alpha"].shape == (12, 4, len(t))
    # 短分段下低频小波超长 → delta 取不到，应被跳过（而不是返回 NaN）
    assert all(np.isfinite(v).all() for v in bp.values())
    topo = band_window_topomap(P, f, t, band=(8, 13), tmin=0.0, tmax=0.4)
    assert topo.shape == (12, 4)
    with pytest.raises(ValueError):
        band_window_topomap(P, f, t, band=(100, 120))
    with pytest.raises(ValueError):
        band_window_topomap(P, f, t, tmin=50.0, tmax=60.0)


def test_tfr_channel_stats_and_export():
    X, _, times = _erp_data(n_per_cond=12, T=250)
    P, f, t = compute_tfr(X, FS, times=times, method="morlet")
    s = tfr_channel_stats(P, t, f, test_win=(0.25, 0.5), baseline=(-0.2, 0.0), band=(8, 13))
    assert s["t"].shape == (4,) and (s["p_corrected"] >= s["p"] - 1e-12).all()
    s2 = tfr_channel_stats(P, t, f, test_win=(0.25, 0.5), baseline=(-0.2, 0.0),
                           band=(8, 13), method="perm", n_perm=200)
    assert s2["t"].shape == (4,) and np.isfinite(s2["p_corrected"]).all()
    h, rows = tfr_to_rows(P, f, t, [f"ch{i}" for i in range(4)])
    assert len(rows) == 4 * len(f) and len(h) == 2 + len(t)
    with pytest.raises(ValueError):
        tfr_to_rows(P, f, t, agg="bogus")


# ---------------------------------------------------------------- PSD
@pytest.mark.parametrize("method", ["welch", "periodogram", "multitaper"])
def test_psd_methods(method):
    X, _, _ = _erp_data(n_per_cond=4, T=500)
    f, P = psd(X, FS, method=method, fmin=1.0, fmax=45.0)
    assert P.shape == (8, 4, len(f)) and f[0] >= 1.0 and f[-1] <= 45.0
    assert (P >= 0).all()


def test_psd_unknown_method_and_relative_power():
    X, _, _ = _erp_data(n_per_cond=3, T=500)
    with pytest.raises(ValueError):
        psd(X, FS, method="nope")
    rel = relative_power(X, FS)
    assert rel.shape == (6, 4, 5)
    # 各频带相对功率之和 = 100%
    np.testing.assert_allclose(rel.sum(axis=-1), 100.0, atol=1e-6)
    assert (rel >= 0).all()


def test_psd_table_absolute_and_relative():
    X, _, _ = _erp_data(n_per_cond=3, T=500)
    f, P = psd(X, FS)
    ch = [f"ch{i}" for i in range(4)]
    h, rows = psd_table(f, P.mean(0), ch, fmin=1.0, fmax=20.0)
    assert h[0] == "通道" and len(rows) == 4 and len(rows[0]) == len(h)
    h2, rows2 = psd_table(f, P.mean(0), ch, relative=True)
    assert h2[0] == "通道" and set(h2[1:]) == {"delta", "theta", "alpha", "beta", "gamma"}
    vals = np.array([float(v) for v in rows2[0][1:]])
    assert abs(vals.sum() - 100.0) < 0.05           # 表格按 2 位小数显示，允许四舍五入误差


# ---------------------------------------------------------------- 界面接线
def test_analysis_page_controls():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])   # noqa: F841
    from ui.analysis_page import AnalysisPage
    from ui.main_window import AppState

    page = AnalysisPage(AppState())
    # 频域：谱方法 / 相对功率 / 导出 / 统计
    assert page.combo_psd.count() == len(["welch", "periodogram", "multitaper"])
    assert page._psd_method() in ("welch", "periodogram", "multitaper")
    assert page.btn_freq_stats.isEnabled() and page.btn_freq_export.isEnabled()
    # ERP：基线校正 / 多条件 / 统计
    assert page.chk_erp_base.isChecked() and page.btn_erp_stats.isEnabled()
    # 时频：方法 / 归一化 / 频段 / 地形图 / 统计 / 导出
    assert page.combo_tfr_method.count() == 3
    assert page.combo_tfr_norm.count() == 5
    assert page._tfr_band() == (8.0, 13.0)              # 默认 alpha
    assert page.btn_tfr_topo.isEnabled() and page.btn_tfr_export.isEnabled()
    assert AnalysisPage._p_star(0.0005) == "***"
    assert AnalysisPage._p_star(0.03) == "*"
