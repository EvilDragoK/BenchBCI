"""预处理补齐项单测：导联定位 / 重参考（含 REST） / 坏段 / 去极值 / 伪迹识别与回归。"""
import os

import numpy as np
import pytest

from core.pipeline import Pipeline
from core.preprocess.artifact import (component_features, proxy_channels, front_most_channels,
                                      suggest_exclude)
from core.preprocess.operators import (OPERATORS, build_operator, channel_positions,
                                       resolve_montage_file)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FS = 100.0
NAMES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
         "C4", "T8", "P3", "Pz", "P4", "O1", "Oz", "O2", "Fpz", "C1"]
rng = np.random.default_rng(0)


def _raw(dur=30.0, blink=True, spike=True, big_segment=False, flat=False, set_montage=True):
    """合成带眨眼 / 尖峰 / 大干扰段 / 平坦段的 EEG。"""
    import mne

    T = int(dur * FS)
    t = np.arange(T) / FS
    data = rng.standard_normal((len(NAMES), T)) * 5e-6
    data += 8e-6 * np.sin(2 * np.pi * 10 * t)[None, :]
    if blink:                                   # 额区最强的眨眼
        b = np.exp(-((t % 5.0 - 2.5) ** 2) / 0.01)
        for i, nm in enumerate(NAMES):
            w = 1.0 if nm.startswith(("Fp", "AF")) else (0.4 if nm.startswith("F") else 0.2)
            data[i] += w * b * 90e-6
    if spike:
        data[3, min(int(7 * FS), T - 1)] = 4e-3  # 电极尖峰
    if big_segment:
        data[:, int(20 * FS):int(22 * FS)] *= 30
    if flat:
        data[:, int(26 * FS):int(26.5 * FS)] = 0.0
    info = mne.create_info(NAMES, FS, "eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    if set_montage:
        raw.set_montage("standard_1020", on_missing="ignore")
    ann = mne.Annotations(onset=np.arange(1.0, dur - 1.0, 1.0), duration=[0.1] * (int(dur) - 2),
                          description=["left" if i % 2 else "right" for i in range(int(dur) - 2)])
    raw.set_annotations(ann)
    return raw


# ---------------------------------------------------------------- 导联定位
def test_montage_positions_and_file_resolution():
    raw = _raw(dur=6.0, blink=False, spike=False)
    ctx = {}
    build_operator("set_montage", {"montage": "standard_1020"}).apply(raw, ctx)
    pos = channel_positions(raw)
    assert len(pos) >= 15
    assert all(len(v) == 3 and np.isfinite(v).all() for v in pos.values())
    assert ctx["channel_positions"] and ctx["montage"]["source"] == "standard"
    assert ctx["montage"]["n_matched"] == len(pos)
    # 名字不是文件 → None；不存在的文件 → None
    assert resolve_montage_file("standard_1005") is None
    assert resolve_montage_file("/definitely/not/here.elc") is None
    assert resolve_montage_file("montage.elc") is None
    assert resolve_montage_file("") is None


def test_montage_from_file(tmp_path):
    """自定义坐标文件（.csv：name,x,y,z）应被读取并匹配。"""
    p = tmp_path / "my_montage.csv"
    lines = ["name,x,y,z"] + [f"{n},{0.01*i:.4f},{-0.02 - 0.001*i:.4f},{0.03:.4f}"
                              for i, n in enumerate(NAMES)]
    p.write_text("\n".join(lines), encoding="utf-8")
    ctx = {}
    raw = _raw(dur=4.0, blink=False, spike=False, set_montage=False)
    op = build_operator("set_montage", {"montage": str(p)})
    assert resolve_montage_file(str(p)) == p
    op.apply(raw, ctx)
    assert ctx["montage"]["source"] == "file"
    assert ctx["montage"]["n_matched"] >= 15        # 坐标被套上


# ---------------------------------------------------------------- 重参考
@pytest.mark.parametrize("mode", ["average", "channels", "rest"])
def test_rereference_modes(mode):
    ctx = {}
    raw = _raw(dur=8.0, blink=False, spike=False)
    data0 = raw.get_data().copy()
    op = build_operator("re_reference", {"mode": mode, "channels": ["Cz"]})
    out = op.apply(raw, ctx)
    d = out.get_data()
    assert np.isfinite(d).all()
    assert not np.allclose(d, data0)                 # 数据确实变了
    if mode == "average":
        assert np.abs(d.mean(axis=0)).max() < 1e-12  # 全通道均值 ≈ 0


def test_rereference_bipolar_adds_channels():
    ctx = {}
    raw = _raw(dur=5.0, blink=False, spike=False)
    n0 = len(raw.ch_names)
    out = build_operator("re_reference",
                         {"mode": "bipolar", "pairs": ["Fp1-Fp2", "C3-C4"]}).apply(raw, ctx)
    assert len(out.ch_names) == n0 + 2
    assert "Fp1-Fp2" in out.ch_names and ctx["bipolar_channels"] == ["Fp1-Fp2", "C3-C4"]


def test_rereference_channels_needs_valid_name():
    with pytest.raises(ValueError):
        build_operator("re_reference", {"mode": "channels", "channels": ["NOT_A_CH"]}) \
            .apply(_raw(dur=4.0, blink=False, spike=False), {})


# ---------------------------------------------------------------- 坏段
def test_bad_segment_detection_and_splitting():
    raw = _raw(dur=30.0, blink=False, spike=False, big_segment=True, flat=True)
    ctx = {}
    out = build_operator("reject_bad_segments",
                         {"win_s": 1.0, "step_s": 0.5, "thresh_uv": 200.0,
                          "flat_uv": 0.5, "min_dur_s": 0.3, "ch_frac": 0.5}).apply(raw, ctx)
    assert ctx["bad_segment_seconds"] > 0
    segs = ctx["bad_segments"]
    assert any(s[0] < 22.0 and s[1] > 20.0 for s in segs)          # 抓到 20-22 s 大干扰段
    assert any("BAD" in str(d) for d in out.annotations.description)

    steps = [("set_montage", {"montage": "standard_1020"}),
             ("reject_bad_segments", {"thresh_uv": 200.0, "flat_uv": 0.5}),
             ("epoching", {"tmin": 0.0, "tmax": 0.6, "baseline": False,
                           "reject_by_annotation": True})]
    obj, ctx2 = Pipeline([(build_operator(n, p), True) for n, p in steps]).run(raw.copy())
    assert ctx2.get("dropped_bad_segments", 0) > 0
    assert "BAD_segment" not in (ctx2.get("classes") or {})        # 坏段不能进类别表

    steps[2][1]["reject_by_annotation"] = False                    # 关掉 → 不按注释剔
    _, ctx3 = Pipeline([(build_operator(n, p), True) for n, p in steps]).run(raw.copy())
    assert ctx3.get("dropped_bad_segments", 0) == 0


def test_bad_segment_clean_data_finds_nothing():
    ctx = {}
    build_operator("reject_bad_segments", {"thresh_uv": 200.0, "flat_uv": 0.5}) \
        .apply(_raw(dur=10.0, blink=False, spike=False), ctx)
    assert ctx["bad_segments"] == [] and ctx["bad_segment_seconds"] == 0


# ---------------------------------------------------------------- 去极值
def test_clip_outliers_removes_spike_with_floor():
    raw = _raw(dur=10.0, blink=False, spike=True)
    ctx = {}
    out = build_operator("clip_outliers",
                         {"mode": "mad", "k": 12.0, "min_uv": 200.0}).apply(raw, ctx)
    assert ctx["clipped_samples"] > 0
    assert np.abs(out.get_data()).max() < np.abs(raw.get_data()).max()   # 尖峰被压低
    assert np.abs(out.get_data()).max() <= 200e-6 + 1e-9
    # 干净数据（无尖峰、无超限值）不应被裁剪
    ctx2 = {}
    clean = _raw(dur=10.0, blink=False, spike=False)
    build_operator("clip_outliers", {"mode": "mad", "k": 12.0, "min_uv": 200.0}) \
        .apply(clean, ctx2)
    assert ctx2["clipped_samples"] == 0


@pytest.mark.parametrize("mode", ["mad", "sigma", "quantile"])
def test_clip_outliers_modes_run(mode):
    ctx = {}
    out = build_operator("clip_outliers", {"mode": mode, "k": 6.0, "quantile": 99.0}) \
        .apply(_raw(dur=6.0, blink=False, spike=True), ctx)
    assert np.isfinite(out.get_data()).all()


def test_reject_outlier_epochs():
    steps = [("set_montage", {"montage": "standard_1020"}),
             ("epoching", {"tmin": 0.0, "tmax": 0.6, "baseline": False}),
             ("reject_outlier_epochs", {"metric": "ptp", "iqr_k": 1.5})]
    raw = _raw(dur=30.0, blink=False, spike=False)
    # 放大最后 4 s（约 13% 的 trial）→ 这些 trial 的峰峰值远超其余
    raw2 = raw.copy()
    d = raw2.get_data()
    d[:, -int(4 * FS):] *= 6
    raw2._data[:] = d

    obj_all, _ = Pipeline([(build_operator(n, p), True) for n, p in steps[:-1]]).run(raw2.copy())
    obj2, ctx2 = Pipeline([(build_operator(n, p), True) for n, p in steps]).run(raw2)
    peak_all = float(np.abs(obj_all.get_data()).max())
    peak_keep = float(np.abs(obj2.get_data()).max())
    assert ctx2["outlier_epochs"] > 0
    assert ctx2["rejected_trials"] >= ctx2["outlier_epochs"]
    assert peak_keep < peak_all / 3                 # 极端 trial 被剔除，剩余幅值回到正常量级


# ---------------------------------------------------------------- 伪迹：代理通道 / 回归 / ICA 判定
def test_proxy_channels_prefers_frontal():
    raw = _raw(dur=5.0, blink=False, spike=False)
    proxy = proxy_channels(raw)
    assert set(proxy) <= set(NAMES)
    assert all(c.startswith(("Fp", "AF")) for c in proxy)   # 优先最靠前的 Fp/AF
    assert len(proxy) <= 4                                  # 不会取太多
    # 没有 Fp/AF 时退到 F 开头
    raw2 = raw.copy().pick([c for c in NAMES if not c.startswith(("Fp", "AF"))])
    fm = front_most_channels(raw2)
    assert fm and all(c.startswith("F") for c in fm) and "F3" in fm


def test_artifact_regression_reduces_blink_variance():
    raw = _raw(dur=20.0, blink=True, spike=False)
    ctx = {}
    out = build_operator("artifact_regression",
                         {"proxies": ["Fp1", "Fp2"], "band_lower": 0.5,
                          "band_upper": 10.0}).apply(raw, ctx)
    d0, d1 = raw.get_data(), out.get_data()
    assert d1.shape == d0.shape and np.isfinite(d1).all()
    # 非代理通道（眨眼经容积传导扩散到这里）的方差应明显下降
    i = NAMES.index("C3")
    assert d1[i].var() < d0[i].var() * 0.9
    assert ctx["artifact_regression"]["proxies"] == ["Fp1", "Fp2"]
    assert ctx["artifact_regression"]["removed_var_mean"] > 0
    # 代理通道自身不被校正（它们是伪迹参考，不是干净的 EEG）
    j = NAMES.index("Fp1")
    np.testing.assert_allclose(d1[j], d0[j])


def test_artifact_regression_needs_proxies():
    raw = _raw(dur=4.0, blink=False, spike=False)
    with pytest.raises(ValueError):
        build_operator("artifact_regression", {"proxies": ["NOPE"]}).apply(raw, {})


def test_ica_component_features_and_suggestion():
    from mne.preprocessing import ICA

    raw = _raw(dur=40.0, blink=True, spike=False)
    raw.pick(NAMES)
    ica = ICA(n_components=8, method="fastica", random_state=0, max_iter=200)
    ica.fit(raw, verbose="ERROR")
    feats = component_features(raw, ica)
    for k in ("kurtosis", "autocorr", "frontal_corr", "emg_ratio", "amp_ratio", "var_frac"):
        assert feats[k].shape == (8,)
    assert np.isfinite(feats["kurtosis"]).all()
    assert abs(float(np.sum(feats["var_frac"])) - 1.0) < 1e-6        # 方差贡献归一
    idx, reasons = suggest_exclude(feats, min_hits=2, max_frac=0.4)
    assert len(idx) <= max(int(8 * 0.4), 1)                          # 安全阀：≤40%
    assert all(reasons[i] for i in idx)                              # 每个都有理由
    assert idx == sorted(idx)


def test_ica_operator_autodetect_records_reasons():
    steps = [("set_montage", {"montage": "standard_1020"}),
             ("ica", {"n_components": 6, "method": "fastica", "random_state": 0,
                      "autodetect_artifact": True, "min_hits": 2}),
             ("epoching", {"tmin": 0.0, "tmax": 0.6, "baseline": False})]
    raw = _raw(dur=40.0, blink=True, spike=False)
    _, ctx = Pipeline([(build_operator(n, p), True) for n, p in steps]).run(raw)
    assert "ica_features" in ctx
    assert isinstance(ctx.get("ica_suggested", []), list)
    assert len(ctx.get("ica_exclude", [])) <= 3                      # 不会把成分删光


def test_new_operators_registered():
    for name in ("reject_bad_segments", "clip_outliers", "artifact_regression",
                 "reject_outlier_epochs"):
        assert name in OPERATORS
        assert OPERATORS[name].title and OPERATORS[name].schema
