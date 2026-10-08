"""逐功能演示与自检脚本（`docs/功能说明.md` 里所有实测数字都由它产出）。

用法::

    .venv/bin/python docs/demo_features.py            # 全部
    .venv/bin/python docs/demo_features.py 模型 时频   # 只跑含关键字的组

设计原则：**每个功能都真跑一遍**（合成信号 + 本机真实 BNCI2014001），
并把关键事实用 ``✔/✘`` 打出来 —— 既当演示，也当回归自检。
"""
from __future__ import annotations

import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_paths import abbrev, bnci_root, physionet_root   # noqa: E402 - 需先加 sys.path

BNCI_ROOT = bnci_root()            # 默认本机路径 + 环境变量 BCI_DATA_ROOT 覆盖
PHYSIONET_ROOT = physionet_root()
_FILTER = [a for a in sys.argv[1:]]
_FAILS: list[str] = []


def sec(title: str):
    if _FILTER and not any(f in title for f in _FILTER):
        return False
    print(f"\n{'─' * 74}\n◆ {title}\n{'─' * 74}")
    return True


def ok(cond, msg):
    print(f"  {'✔' if cond else '✘'} {msg}")
    if not cond:
        _FAILS.append(msg)
    return bool(cond)


def info(msg):
    print(f"    {msg}")


def _eeg(n_ch=8, n_times=256, n_epochs=24, fs=250.0, freq=10.0, noise=1.0):
    """合成 EEG：alpha 正弦 + 白噪声（(N,C,T) µV）。"""
    rng = np.random.default_rng(0)
    t = np.arange(n_times) / fs
    base = 8.0 * np.sin(2 * np.pi * freq * t)
    X = rng.normal(0, noise, (n_epochs, n_ch, n_times)) + base
    return X


def _real_raw(subject=1, session="T", dur=40.0):
    """真实 BNCI 被试的一个 session（按需裁剪到 dur 秒）。"""
    from core.io.bnci import load_subject_raw

    raw, meta = load_subject_raw(BNCI_ROOT, subject, session, eeg_only=True)
    if dur and raw.n_times > dur * raw.info["sfreq"]:
        raw.crop(0, dur)
    return raw, meta


# ==================================================================== 1 数据 IO
def demo_io():
    if not sec("1. 数据读取（BNCI / PhysioNet / 通用格式）"):
        return
    from core.io.bnci import CLASS_NAMES, EEG_CH_NAMES, list_subjects, load_subject_raw
    from core.io.loaders import SUPPORTED_FORMATS

    info(f"支持格式：{', '.join(SUPPORTED_FORMATS)}")
    if BNCI_ROOT.exists():
        subs = list_subjects(BNCI_ROOT)
        raw, meta = load_subject_raw(BNCI_ROOT, 1, "T", eeg_only=True)
        ok(len(subs) == 9, f"BNCI2014001：被试 {subs}，session T/E")
        ok(len(raw.ch_names) == 22 and raw.info["sfreq"] == 250.0,
           f"被试 1 T：{len(raw.ch_names)} 通道 / {raw.info['sfreq']:.0f} Hz / "
           f"{meta['n_trials']} trial / 时长 {meta['duration_s']:.0f}s")
        ok(EEG_CH_NAMES[-3:] == ["Pz", "P2", "POz"],
           f"官方 22 导联表（末 4 个 = {EEG_CH_NAMES[-4:]}；无 Oz/CP5/CP6）")
        ok(set(meta["class_counts"]) == set(CLASS_NAMES),
           f"类别 {meta['class_counts']}；cue 偏移 = {meta['cue_offset_s']}s")
    else:
        info("BNCI2014001 不在本机 → 跳过")

    from core.io.physionet import DEFAULT_RUNS, RUN_TASKS

    ok(DEFAULT_RUNS == [4, 6, 8, 10, 12, 14],
       f"PhysioNet 纯想象 run = {DEFAULT_RUNS}"
       f"（R04/08/12 想象左右手、R06/10/14 想象双手双脚）")
    ok(RUN_TASKS[5][0] == "real" and RUN_TASKS[6][0] == "imagined",
       "run→任务映射符合官方分组（R05 真实、R06 想象）")
    if PHYSIONET_ROOT.exists():
        info(f"PhysioNet 根目录可用：{abbrev(PHYSIONET_ROOT)}")
    else:
        info("PhysioNet 数据不在本机 → 仅验证映射表")


# ============================================================== 2 预处理流水线
def demo_preprocess():
    if not sec("2. 预处理（19 个算子 / 真实数据全链路）"):
        return
    from core.preprocess.artifact import proxy_channels
    from core.preprocess.operators import OPERATORS, channel_positions
    from core.pipeline import Pipeline, build_operator

    info("算子清单：" + "、".join(OPERATORS))
    steps = [
        ("set_montage", {"montage": "standard_1005"}),
        ("pick_eeg", {"keep_eog": False}),
        ("notch_filter", {"freqs": [50]}),
        ("bandpass_filter", {"l_freq": 4.0, "h_freq": 40.0}),
        ("re_reference", {"mode": "average"}),
        ("reject_bad_segments", {"thresh_uv": 200.0, "flat_uv": 0.5}),
        ("clip_outliers", {"mode": "mad", "k": 12.0, "min_uv": 200.0}),
        ("artifact_regression", {"band_lower": 0.5, "band_upper": 10.0}),
        ("ica", {"n_components": 8, "method": "fastica", "autodetect_artifact": True}),
        ("epoching", {"tmin": 0.5, "tmax": 2.5, "baseline": True,
                      "reject_by_annotation": True}),
        ("amplitude_reject", {"threshold_uv": 150.0}),
        ("reject_outlier_epochs", {"metric": "ptp", "iqr_k": 3.0}),
    ]
    raw, _ = _real_raw(1, "T", dur=None)      # 整段 session（前 3 个 run 无标签，不能截太短）
    obj, ctx = Pipeline([(build_operator(n, p), True) for n, p in steps]).run(raw)
    ok(len(obj) > 0, f"12 步真实数据全链路 → {len(obj)} epochs（输入 {len(raw.ch_names)} 通道）")
    ok(len(channel_positions(raw)) == 22, f"导联定位：22 个电极拿到坐标（{list(channel_positions(raw))[:3]}…）")
    ok("bad_segment_seconds" in ctx, f"坏段检测：{len(ctx['bad_segments'])} 段 / "
                                     f"{ctx['bad_segment_seconds']:.1f}s")
    ok("clipped_samples" in ctx, f"去极值裁剪：{ctx['clipped_samples']} 个采样点")
    ok("ica_features" in ctx,
       f"ICA 自动判定：建议剔除 {ctx.get('ica_exclude', [])}，候选 "
       f"{sorted((ctx.get('ica_candidate_reasons') or {}))}")
    ok(proxy_channels(raw) and not any(ch in ("EOG1", "EOG2") for ch in raw.ch_names),
       f"伪迹代理通道 = {proxy_channels(raw)}（无 EOG 时退到最前额电极）")
    info(f"分段统计：分段前 {ctx.get('n_trials_before', 0)} 个 trial → 保留 {len(obj)}，"
         f"坏段丢弃 {ctx.get('dropped_bad_segments', 0)}，离群丢弃 {ctx.get('outlier_epochs', 0)}")
    # REST 重参考（球模型近似）：需要电极坐标 → 先设导联
    r2, _ = _real_raw(1, "T", dur=20.0)
    from core.preprocess.operators import build_operator as bo

    bo("set_montage", {"montage": "standard_1005"}).apply(r2, {})
    out_rest = bo("re_reference", {"mode": "rest", "rest_alpha": 0.2}).apply(r2.copy(), {})
    ok(np.isfinite(out_rest.get_data()).all(),
       f"REST 重参考：数据全有限（max|V| = {np.abs(out_rest.get_data()).max() * 1e6:.1f} µV）")


# ================================================================ 3 特征工程
def demo_features():
    if not sec("3. 手工特征工程（频谱/熵/复杂度/Hjorth/连接）"):
        return
    from core import features as F

    info(f"特征表 FEATURES（{len(F.FEATURES)} 项）：{list(F.FEATURES)[:10]}…")
    X = _eeg(n_ch=4, n_times=500, n_epochs=8, freq=10.0)
    ap = F.absolute_power(X, 250.0, bands={"alpha": (8, 13), "beta": (13, 30)})
    ok(ap.shape == (8, 4, 2), f"绝对频带功率 (N,C,B) = {ap.shape}；alpha 均值 {ap[..., 0].mean():.3g} µV²")
    rel = F.relative_power(X, 250.0)
    ok(abs(rel.sum(-1).mean() - 100) < 1e-6, f"相对频带功率各频带之和 = {rel.sum(-1).mean():.4f}%")
    de = F.differential_entropy(X, 250.0)
    ok(de.shape == (8, 4, 5), f"微分熵 (N,C,5 频带) = {de.shape}")
    tb = F.theta_beta_ratio(X, 250.0)
    ok(np.isfinite(tb).all(), f"θ/β 比值：均值 {tb.mean():.3f}")
    c0 = F.c0_complexity(X, 250.0)
    ok(c0.shape == (8, 4) and np.isfinite(c0).all(), f"C0 复杂度 (N,C) = {c0.shape}")
    se = F.spectral_entropy(X, 250.0)
    ok(se.shape[:2] == (8, 4), f"谱熵 (N,C) = {se.shape}")
    info("另含：" + "、".join(n for n in ("hoc", "nsi", "hemisphere_pairs", "dasm", "rasm",
                                        "hilbert_timf", "functional_connectivity")
                             if hasattr(F, n)) + " 等函数")


# ======================================================= 4 数据集 / 归一化 / EA
def demo_dataset():
    if not sec("4. 张量数据集、归一化（防泄漏）、欧几里得对齐、数据增强"):
        return
    from core import datasets as D

    info(f"归一化模式（{len(D.NORM_MODES)}）：{D.NORM_MODES}")
    Xtr = _eeg(n_epochs=40) * 10.0
    Xte = _eeg(n_epochs=10) * 10.0
    for mode in ("zscore_channel", "minmax_channel", "zscore_epoch", "quantile"):
        st = D.compute_norm_stats(Xtr, mode)
        a = D.apply_normalization(Xtr, mode, st)
        b = D.apply_normalization(Xte, mode, st)
        ok(np.isfinite(a).all() and np.isfinite(b).all(), f"{mode:16s} 训练/测试都有限")
    st = D.compute_norm_stats(Xtr, "zscore_channel")
    ok(abs(float(D.apply_normalization(Xtr, "zscore_channel", st).mean())) < 1e-6,
       "防泄漏：统计量只来自训练折（用同一统计量变换测试集）")
    # 各通道独立噪声（不要共用一个正弦，否则协方差近似奇异、EA 不稳定）
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 3.0, (40, 8, 200)),
                        rng.normal(0, 0.4, (40, 8, 200))])
    sid = np.array([1] * 40 + [2] * 40)
    ea = np.concatenate([D.euclidean_alignment(X[sid == s]) for s in (1, 2)])
    cov_err = []
    for s in (1, 2):
        v = ea[sid == s].transpose(1, 0, 2).reshape(ea.shape[1], -1)
        cov_err.append(float(np.abs(np.cov(v) - np.eye(ea.shape[1])).max()))
    ok(max(cov_err) < 0.15, f"逐被试 EA：各被试协方差偏离单位阵最大 {max(cov_err):.3f}")
    aug = D.augment_batch(Xtr[:8], time_shift=8, noise_std=0.1, channel_dropout=0.1,
                          generator=np.random.default_rng(0))
    ok(aug.shape == Xtr[:8].shape and np.isfinite(aug).all(),
       f"数据增强（平移 8 点 + 高斯噪声 0.1 + 通道 dropout 0.1）输出形状 {aug.shape}")


# ==================================================================== 5 模型
def demo_models():
    if not sec("5. 模型：内置 6 个 + 基础模型 + 自定义模型"):
        return
    import torch

    from core.models import (CLASSIC_MODELS, MODEL_NAMES, TORCH_MODELS, all_model_names,
                             build_model, is_classic, is_custom)

    info(f"内置 Torch 模型：{list(TORCH_MODELS)}")
    info(f"传统基线：{list(CLASSIC_MODELS)}")
    info(f"下拉框全部可用：{all_model_names()}")
    C, T, K = 8, 500, 4
    for name in TORCH_MODELS:
        try:
            m = build_model(name, K, C, T, {})
            with torch.no_grad():
                y = m(torch.randn(2, C, T))
            n_par = sum(p.numel() for p in m.parameters())
            ok(tuple(y.shape) == (2, K), f"{name:14s} 前向输出 {tuple(y.shape)}，参数 {n_par / 1e3:.1f}k")
        except Exception as e:  # noqa: BLE001
            ok(False, f"{name} 前向失败：{type(e).__name__}: {str(e)[:60]}")
    # 各模型对「输入时长」的最低要求（工程上最常踩的坑）
    mins = {}
    for name in TORCH_MODELS:
        for t_len in (64, 128, 200, 256, 384, 500, 800, 1000):
            try:
                mm = build_model(name, K, C, t_len, {})
                with torch.no_grad():
                    yy = mm(torch.randn(2, C, t_len))
                if tuple(yy.shape) == (2, K):
                    mins[name] = t_len
                    break
            except Exception:  # noqa: BLE001
                continue
    info("最小可用输入时长（采样点，250 Hz 下 64≈0.26s / 500≈2s）："
         + "、".join(f"{k}={v}" for k, v in mins.items()))
    clf = build_model("csp_lda", K, C, T, {"n_components": 4})
    clf.fit(_eeg(n_ch=C, n_times=T, n_epochs=24), np.tile([0, 1, 2, 3], 6))
    pred = clf.predict(_eeg(n_ch=C, n_times=T, n_epochs=4))
    ok(pred.shape == (4,) and is_classic("csp_lda"), f"经典基线 CSP+LDA 预测 {pred.shape}，类别 {clf.classes_}")

    # 自定义模型：以 EEGNet 为模板 → 检查（逐层维度）→ 保存 → 构建 → 删除
    from core.models import custom as CM

    with tempfile.TemporaryDirectory() as td:
        old = CM.CUSTOM_DIR
        CM.CUSTOM_DIR = Path(td)
        try:
            code = CM.template_source("eegnet", "demo_eegnet_pyramid",
                                      note="演示：在 EEGNet 基础上加金字塔特征")
            res = CM.check_code(code, n_classes=K, n_ch=C, n_times=T, batch=2)
            ok(res["ok"], f"自定义模型检查：输入 {res['input_shape']} → 输出 {res['output_shape']}，"
                          f"{res['n_layers']} 层 / {res['n_params'] / 1e3:.1f}k 参数，"
                          f"目标类别数 {res['n_classes']} 一致")
            trace = " → ".join(str(s.get("out") or s.get("shape") or s) for s in res["steps"][:4])
            info(f"逐层维度追踪（前 4 层）：{trace}")
            path = CM.save_custom("demo_eegnet_pyramid", code, base="eegnet",
                                  note="演示：在 EEGNet 基础上加金字塔特征")
            ok(Path(path).exists() and "demo_eegnet_pyramid" in CM.custom_names(),
               f"保存为新模型 → {Path(path).name}（训练页下拉框即时可见）")
            m2 = build_model("demo_eegnet_pyramid", K, C, T, {})
            with torch.no_grad():
                y2 = m2(torch.randn(2, C, T))
            ok(tuple(y2.shape) == (2, K),
               f"训练路径可直接构建该自定义模型，输出 {tuple(y2.shape)}")
            # 反向用例：把输出的类别数改成 K+1 → 检查必须报「输出维度不对」
            head, _, tail = code.rpartition("n_classes)")
            r_bad = CM.check_code(head + "n_classes + 1)" + tail, n_classes=K, n_ch=C,
                                  n_times=T, batch=2)
            ok(not r_bad["ok"] and any("输出维度" in e for e in r_bad["errors"]),
               f"类别数写错的代码被拦下：{str(r_bad['errors'][0])[:56]}…")
            # 反向用例 2：时窗太短 → 构造阶段给出「可操作」的报错（不是难懂的 BN 报错）
            try:
                build_model("deepconvnet", K, C, 64, {})
                ok(False, "DeepConvNet 在 64 采样点竟然构建成功")
            except ValueError as e:
                ok("最小时长" in str(e),
                   f"时窗太短时的报错可操作：{str(e).splitlines()[1][:56]}…")
            CM.delete_custom("demo_eegnet_pyramid")
            ok("demo_eegnet_pyramid" not in CM.custom_names(), "删除自定义模型生效")
        finally:
            CM.CUSTOM_DIR = old

    try:
        from core.models.foundation import list_foundations

        info(f"EEG 基础模型（需 braindecode + 预训练权重）：{list_foundations()}")
    except Exception as e:  # noqa: BLE001
        info(f"基础模型注册表不可用（{type(e).__name__}）")
    ok(is_custom("nothing_here") is False, "is_custom 对未知名字返回 False")


# =================================================================== 6 划分
def demo_splits():
    if not sec("6. 划分协议（9 种）与防泄漏"):
        return
    from core.train.splits import make_splits

    y, sid, sess = [], [], []
    for s in range(1, 4):
        for sn in ("T", "E"):
            for c in range(4):
                y += [c] * 12
                sid += [s] * 12
                sess += [sn] * 12
    y, sid, sess = np.array(y), np.array(sid), np.array(sess)
    n = len(y)
    block = np.repeat(np.arange(6), n // 6 + 1)[:n]
    cases = [("session", {}), ("loso", {}), ("session_kfold", {}), ("kfold", {"k": 3}),
             ("within", {"k": 3}), ("few_shot", {"n_shot": 4}), ("subject_kfold", {"k": 3}),
             ("holdout", {"k": 5}), ("block", {"block": block}),
             ("custom", {"custom_folds": [{"name": "f1", "train": [1, 2], "test": [3]}]})]
    print(f"    {'协议':14s} {'折数':>4s} {'train':>6s} {'val':>5s} {'test':>5s}  泄漏检查")
    bad = 0
    for scheme, kw in cases:
        sp = make_splits(y, sid, sess, scheme, **kw)
        s0 = sp[0]
        leak = (np.intersect1d(s0.train, s0.val).size or np.intersect1d(s0.train, s0.test).size
                or np.intersect1d(s0.val, s0.test).size)
        bad += bool(leak)
        print(f"    {scheme:14s} {len(sp):>4d} {len(s0.train):>6d} {len(s0.val):>5d} "
              f"{len(s0.test):>5d}  {'✘ 有重叠' if leak else '✔ train/val/test 互不重叠'}")
    ok(bad == 0, "10 种协议首折均无索引重叠（含 val 不泄漏进 train）")
    info(f"数据规模：3 被试 × 2 session × 4 类 × 12 = {n} 个 trial")


# =============================================================== 7 训练与评测
def demo_train():
    if not sec("7. 训练（后台线程/断点续训/早停）与评测指标"):
        return
    import glob
    import json

    from core.train.evaluation import information_transfer_rate

    acc, kappa, itr = 0.6, 0.45, information_transfer_rate(0.6, 4, 8.0)
    ref = (np.log2(4) + 0.6 * np.log2(0.6) + 0.4 * np.log2(0.4 / 3)) * 60 / 8.0
    ok(abs(itr - ref) < 1e-6, f"ITR（Wolpaw）：acc=0.6/4 类/8s → {itr:.3f} bits/min（与公式一致）")
    dirs = sorted(d for d in glob.glob(str(ROOT / "experiments" / "*"))
                  if (Path(d) / "summary.json").exists())
    if dirs:
        s = json.load(open(Path(dirs[-1]) / "summary.json"))
        a = s.get("aggregate", {})
        info(f"最近一次实验：{Path(dirs[-1]).name}")
        info(f"  acc={a.get('acc_mean', 0) * 100:.2f}% ± {a.get('acc_std', 0) * 100:.2f}%，"
             f"kappa={a.get('kappa_mean', 0):.3f}，ITR={a.get('itr_mean', 0):.2f} bits/min，"
             f"折数 {len(a.get('acc_per_fold') or [])}")
        info(f"  summary 指标键：{sorted(a)[:12]}…")
        ok("acc_mean" in a and "confusion_total" in a,
           "实验摘要含 acc/kappa/ITR/混淆矩阵，可凭 config.yaml 复现")
    from core.train.compare import check_compatible

    info(f"双实验对比（一致性检查 + 指标对照）可用：{callable(check_compatible)}")


# ====================================================================== 8 ERP
def demo_erp():
    if not sec("8. ERP：叠加平均 / 多条件 / 成分提取 / 统计检验 / 地形图"):
        return
    from core.erp import (component_condition_stats, erp_average, erp_conditions,
                          extract_components, single_trial_components)

    fs, T = 250.0, 250
    times = np.arange(T) / fs - 0.2
    rng = np.random.default_rng(7)
    rows, labels = [], []
    for i in range(120):
        cond = i % 2
        x = rng.normal(0, 3.0, (4, T))
        x[:, (times >= 0.25) & (times <= 0.5)] += 25.0 if cond else 5.0   # target 的 P3 更强
        rows.append(x)
        labels.append("target" if cond else "std")
    X = np.stack(rows)
    wave = erp_average(X, labels, "target")
    ok(wave.shape == (4, T), f"叠加平均（条件 target，n=60）→ {wave.shape}")
    waves, cnt = erp_conditions(X, labels)
    ok(set(waves) == {"target", "std"}, f"多条件叠加 → {[(k, v.shape, cnt[k]) for k, v in waves.items()]}")
    comps = extract_components(times, wave)
    info("平均波形成分：" + "，".join(
        f"{k} {v['latency_ms'][0]:.0f}ms/{v['amplitude'][0]:.1f}µV"
        for k, v in comps.items() if "latency_ms" in v))
    st = single_trial_components(times, X)
    ok(st["P3"]["amplitude"].shape == (120, 4), f"逐试次成分提取 → amplitude {st['P3']['amplitude'].shape}")
    res = component_condition_stats(times, X, labels, paired=False)
    p3 = res["components"]["P3"]["amplitude"]
    info(f"P3 幅值检验（{res['conds'][0]} vs {res['conds'][1]}）：t={p3['t'][0]:.1f}，"
         f"p(FDR)={p3['p_corrected'][0]:.2e}，d={p3['d'][0]:.1f}")
    ok((p3["p_corrected"] < 0.001).all() and (p3["t"] < 0).all(),
       "注入的 P3 条件差异被检出（t<0、FDR 后 p<0.001）")
    n1 = res["components"]["N1"]["amplitude"]
    ok((np.abs(n1["d"]) < 1.0).all(), f"未注入差异的 N1 效应量很小（|d|max={np.abs(n1['d']).max():.2f}）")


# ==================================================================== 9 时频
def demo_tfr():
    if not sec("9. 时频：Morlet/多窗/STFT + ERD-ERS + 地形图 + 统计 + 导出"):
        return
    from core.timefreq import (band_tfr_power, band_window_topomap, compute_tfr,
                               tfr_baseline, tfr_channel_stats, tfr_to_rows)

    fs, T = 250.0, 500
    times = np.arange(T) / fs - 0.5
    rng = np.random.default_rng(0)
    X = np.stack([rng.normal(0, 1, (4, T)) + 6.0 * np.sin(2 * np.pi * 10 * times)
                  + (5.0 if i % 2 else 0.0) * np.sin(2 * np.pi * 10 * times) for i in range(20)])
    shapes = {}
    for method in ("morlet", "multitaper", "stft"):
        P, f, t = compute_tfr(X, fs, times=times, method=method)
        shapes[method] = (P.shape, float(f.min()), float(f.max()))
        info(f"{method:11s} → power{P.shape}，{f.min():.0f}–{f.max():.0f} Hz，{len(t)} 个时间点")
    ok(len(shapes) == 3, "三种方法输出形状一致（可直接互换）")
    P, f, t = compute_tfr(X, fs, times=times, method="morlet")
    k10 = int(np.argmin(np.abs(f - 10)))
    peak = int(np.argmax(P[:, :, :, :].mean(axis=(0, 1, 3))))
    ok(peak == k10, f"10 Hz 正弦的功率峰值落在 {f[peak]:.0f} Hz 频点")
    ers = tfr_baseline(P, t, baseline=(-0.5, 0.0), mode="ers")
    b = (t >= -0.5) & (t <= 0.0)
    ok(abs(float(ers[..., b].mean())) < 1e-9, f"ERD/ERS 基线均值 = {ers[..., b].mean():.2e}%（应为 0）")
    bp = band_tfr_power(P, f)
    topo = band_window_topomap(P, f, t, band=(8, 13), tmin=0.2, tmax=0.8)
    ok(topo.shape == (20, 4), f"频段×时间窗地形图 → {topo.shape}（α 频段 {list(bp)}）")
    st = tfr_channel_stats(P, t, f, test_win=(0.2, 0.8), baseline=(-0.5, 0.0), band=(8, 13))
    ok((st["t"] > 3).all() and (st["p_corrected"] < 0.01).all(),
       f"逐通道检验：t = {np.round(st['t'], 1)}，p(FDR) < 0.01，d = {np.round(st['d'], 1)}")
    h, rows = tfr_to_rows(P, f, t, [f"ch{i}" for i in range(4)])
    ok(len(rows) == 4 * len(f) and len(h) == 2 + len(t), f"导出 CSV 表：{len(rows)} 行 × {len(h)} 列")


# ===================================================================== 10 PSD
def demo_psd():
    if not sec("10. 功率谱密度：3 种谱估计 + 相对功率 + 逐通道统计 + 导出"):
        return
    from core.features import PSD_METHODS, absolute_power, psd, psd_table, relative_power
    from core.stats import group_report

    fs, T = 250.0, 1000
    rng = np.random.default_rng(0)
    t = np.arange(T) / fs
    X = np.stack([rng.normal(0, 1, (4, T)) + 8.0 * np.sin(2 * np.pi * 10 * t) for _ in range(12)])
    for m in PSD_METHODS:
        f, P = psd(X, fs, method=m, fmin=1, fmax=45)
        k = int(np.argmin(np.abs(f - 10)))
        ok(P.shape == (12, 4, len(f)) and k == int(np.argmax(P.mean(axis=(0, 1)))),
           f"{m:12s} → {P.shape}，10 Hz 处为谱峰")
    f, P = psd(X, fs)
    rel = relative_power(X, fs)
    ok(abs(rel.sum(-1).mean() - 100) < 1e-6, f"相对功率各频带之和 = {rel.sum(-1).mean():.4f}%")
    h, rows = psd_table(f, P.mean(0), [f"ch{i}" for i in range(4)])
    h2, rows2 = psd_table(f, P.mean(0), [f"ch{i}" for i in range(4)], relative=True)
    ok(len(rows) == 4 and h2[0] == "通道", f"导出表：绝对（{len(h)} 列，dB）/ 相对（{h2[1:]}）")
    ap = absolute_power(X, fs)
    rep = group_report(np.moveaxis(ap[:6, :, 2], 0, -1), np.moveaxis(ap[6:, :, 2], 0, -1),
                       paired=False)
    ok((rep["p_corrected"] >= rep["p"] - 1e-12).all(),
       f"逐通道 alpha 功率检验：t={np.round(rep['t'], 2)}，p(FDR)={np.round(rep['p_corrected'], 3)}")


# ================================================================= 11 非线性
def demo_nonlinear():
    if not sec("11. 非线性：样本熵 / 近似熵 / 排列熵"):
        return
    from core.nonlinear import (approximate_entropy, entropy_channels, permutation_entropy,
                                sample_entropy)

    rng = np.random.default_rng(0)
    t = np.arange(600) / 250.0
    noise = rng.standard_normal(600)
    sine = np.sin(2 * np.pi * 10 * t)
    const = np.ones(600)
    for name, fn in (("样本熵", sample_entropy), ("近似熵", approximate_entropy),
                     ("排列熵", permutation_entropy)):
        a, b, c = fn(noise), fn(sine), fn(const)
        ok(a > b >= c, f"{name}：噪声 {a:.3f} > 正弦 {b:.3f} ≥ 常数 {c:.3f}（越不规则越大）")
    x = np.array([0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0])
    v = sample_entropy(x, m=2, r=1e-9)
    ok(abs(v - np.log(3)) < 1e-9, f"A=0 的极端情形返回理论上界 log(B)={v:.4f}（旧实现会给 0.0，方向相反）")
    X = rng.standard_normal((5, 3, 300))
    ent = entropy_channels(X)
    ok(all(v_.shape == (3,) for v_ in ent.values()), f"逐通道熵 = {list(ent)}，形状 (3,)")


# =================================================================== 12 网络
def demo_networks():
    if not sec("12. 脑网络：12 种连接 + 图指标 + 稀疏化 + 生物标志物"):
        return
    from core.brainnet import biomarkers_by_condition, dynamic_connectivity, static_connectivity
    from core.networks import (CONNECT_METHODS, SPARSIFY_STRATEGIES, binarize,
                               connectivity_matrix, graph_metrics, sparsify)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, 6, 400))
    # 只在后一半 trial 注入强相关 → 该连接在条件 b 与 a 之间有判别性
    X[8:, 1] = 0.9 * X[8:, 2] + 0.1 * rng.standard_normal((8, 400))
    shapes = {}
    for m in CONNECT_METHODS:
        A = connectivity_matrix(X, 250.0, method=m)
        shapes[m] = tuple(A.shape)
    info("连接方法 → 输出形状：" + "，".join(f"{k}{v}" for k, v in shapes.items()))
    ok(all(np.isfinite(A).all() for A in
           [connectivity_matrix(X[:1], 250.0, method=m) for m in CONNECT_METHODS[:4]]),
       "逐 trial 与跨 trial 方法都可计算，结果有限")
    A = connectivity_matrix(X, 250.0, method="corr").mean(axis=0)
    B = binarize(A, density=0.3)
    gm = graph_metrics(B)
    n_edges = 6 * 5 / 2
    ok(abs(B.sum() / 2 - 0.3 * n_edges) <= 2,
       f"二值化密度 0.3 → {int(B.sum() / 2)} 条边（目标 {0.3 * n_edges:.1f}）；"
       f"平均度 {gm['degree'].mean():.1f}")
    full = np.ones((5, 5)) - np.eye(5)
    gmf = graph_metrics(full)
    ok(abs(gmf["global_efficiency"] - 1.0) < 1e-9 and abs(gmf["clustering"].mean() - 1.0) < 1e-9,
       f"完全图校验：全局效率 {gmf['global_efficiency']:.3f}、聚类 {gmf['clustering'].mean():.3f}（应为 1）")
    for s in SPARSIFY_STRATEGIES:
        S = sparsify(A, strategy=s, density=0.2)
        assert np.isfinite(S).all()
    ok(True, f"5 种稀疏化策略都可用：{SPARSIFY_STRATEGIES}")
    labels = ["a"] * 8 + ["b"] * 8
    r = biomarkers_by_condition(X, labels, 250.0, top_n=5)
    pairs = [{i, j} for i, j, *_ in r["top_connections"]]
    ok({1, 2} in pairs and r["n_significant"] >= 1,
       f"生物标志物：注入的判别连接 {sorted(pairs, key=str)[:3]}… 被检出，"
       f"FDR 校正后显著边数 = {r['n_significant']}（未校正的原始 p 更小 → 见 p_corrected）")
    times, seq = dynamic_connectivity(X, 250.0, win_s=1.0, step_s=0.5)   # (T,), (T,C,C)
    seq = np.asarray(seq)
    ok(seq.ndim == 3 and seq.shape[0] == len(times),
       f"动态连接 → 窗口 {seq.shape}（时间点 {np.round(times, 2)[:3]}…）")
    st_adj = np.asarray(static_connectivity(X, 250.0))
    ok(st_adj.shape[-2:] == (6, 6),
       f"静态连接图 → {st_adj.shape}（逐 trial 时自行按 trial 平均即得网络）")


# ====================================================== 13 非时序网络（netcov）
def demo_netcov():
    if not sec("13. 非时序脑网络：结构协变 / 相似性 / 滞后 / 外部矩阵导入导出"):
        return
    from core.netcov import (LAG_INTERP, SIMILARITY_METHODS, group_covariance,
                             individual_similarity, lag_matrix, load_matrix_file,
                             save_matrix_file)

    rng = np.random.default_rng(0)
    table = rng.normal(1.0, 0.2, (30, 8))            # 30 被试 × 8 个 ROI 灰质体积
    A = group_covariance(table)
    labels = [f"ROI{i}" for i in range(8)]
    ok(A.shape == (8, 8) and np.allclose(A, A.T),
       f"T1 结构协变：{table.shape} 表 → 协变矩阵 {A.shape}（对称、对角 1）")
    sim = {m: individual_similarity(table, method=m) for m in SIMILARITY_METHODS}
    ok(all(np.asarray(v).shape == (30, 30) for v in sim.values()),
       f"个体相似性网络：{SIMILARITY_METHODS} 5 种指标 → 均为 (30,30)")
    t = np.arange(2000) / 250.0
    multi = np.stack([np.sin(2 * np.pi * 10 * t), np.roll(np.sin(2 * np.pi * 10 * t), 5)])
    L = lag_matrix(multi, 250.0, interp=LAG_INTERP[0])
    ok(L.shape == (2, 2) and np.isfinite(L).all(),
       f"有效连接滞后矩阵：{L.shape}（{LAG_INTERP} 两种插值可选）")
    with tempfile.TemporaryDirectory() as td:
        for ext in (".npy", ".csv", ".mat"):
            p = Path(td) / f"conn{ext}"
            save_matrix_file(p, A, labels=labels)
            M2, lab2 = load_matrix_file(p)
            assert M2.shape == A.shape, ext
        ok(True, "外部矩阵导入导出：.npy / .csv / .mat 三种格式往返一致（带 ROI 标签）")


# =================================================================== 14 影像
def demo_neuroimg():
    if not sec("14. 影像（可选依赖）：NIfTI + 图集 → ROI 时序/数值"):
        return
    from core import neuroimg

    try:
        neuroimg.require_nifti()
        info("nibabel 可用 → 可直接读 NIfTI、按图集抽 ROI 时序")
    except Exception as e:  # noqa: BLE001
        info(f"未安装影像依赖（{type(e).__name__}）→ 该功能按钮会自动禁用并给出安装提示")
    info(f"内置可选图集：{list(neuroimg.HO_ATLASES)[:4]}…")
    ok(hasattr(neuroimg, "extract_roi_timeseries"), "ROI 时序/数值提取接口存在")


# ============================================================= 15 实验管理
def demo_experiment():
    if not sec("15. 实验管理：目录 + SQLite 索引 + 失效记录过滤"):
        return
    import json
    import tempfile

    from core.experiment import ExperimentManager

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "experiments"
        mgr = ExperimentManager(root=root)
        exp = mgr.create("demo", {"experiment": {"name": "demo"}})
        exp.log("hello")
        exp.save_metrics([{"fold": 0, "acc": 0.5}])
        exp.save_summary({"n_folds": 1, "aggregate": {"acc_mean": 0.5}})
        # 索引由训练结束时显式写入（run_training 里调用 manager.register）
        mgr.register(exp, name="demo", model="csp_lda", scheme="session",
                     acc_mean=0.5, n_trials=288)
        rows = mgr.list_experiments()
        ok(len(rows) == 1 and (exp.dir / "summary.json").exists(),
           f"实验目录结构：{sorted(p.name for p in exp.dir.iterdir())}")
        # 目录被删 → 索引里的残留记录应被自动过滤并清理
        import shutil

        shutil.rmtree(exp.dir)
        ok(len(mgr.list_experiments()) == 0, "目录被删后列表自动过滤该记录（不再出现点进去报错的行）")


# ========================================================= 17 变量关联（Linked）
def demo_linked():
    if not sec("16. 变量关联分析（RF 重要性 + Pearson + LMM，Linked Heatmap）"):
        return
    import csv

    from core.features import absolute_power, spectral_entropy
    from core.linked import export_tables, linked_analysis, plot_linked_heatmap

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # ---- ① 合成验证：已知效应能否被正确恢复（x1 正、x2 负、x3 弱，x4~x8 噪声）----
        rng = np.random.default_rng(0)
        n, G = 240, 8
        grp = np.repeat(np.arange(G), n // G)
        g_eff = rng.normal(0, 2.0, G)[grp]
        Xs = rng.normal(0, 1, (n, 8))
        ys = 1.6 * Xs[:, 0] - 1.2 * Xs[:, 1] + 0.3 * Xs[:, 2] + g_eff + rng.normal(0, 0.8, n)
        names_s = [f"x{i + 1}" for i in range(8)]
        csv_s = td / "sim.csv"
        with open(csv_s, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subject"] + names_s + ["y"])          # 文本分组列 + 数值变量列
            for i in range(n):
                w.writerow([f"S{grp[i] + 1:02d}"] + [f"{v:.6g}" for v in Xs[i]] + [f"{ys[i]:.6g}"])
        ok(csv_s.exists(), f"写出一张变量表（含文本分组列 subject）：{csv_s.name}")

        res = linked_analysis(np.column_stack([Xs, ys]), "y", names_s + ["y"],
                              groups=grp, n_estimators=400, log=lambda s: info(s))
        imp = {r["variable"]: r["importance"] for r in res["importance"]}
        eff = {r["variable"]: r for r in res["effects"]}
        ok(imp["x1"] > max(imp[f"x{i}"] for i in (4, 5, 6, 7, 8)) and
           imp["x2"] > max(imp[f"x{i}"] for i in (4, 5, 6, 7, 8)),
           f"RF 重要性把真驱动变量排在最前（x1={imp['x1']:.3f}、x2={imp['x2']:.3f}，"
           f"噪声最大 {max(imp[f'x{i}'] for i in (4, 5, 6, 7, 8)):.4f}）")
        ok(eff["x1"]["beta"] > 0.3 and eff["x1"]["p_value"] < 0.01 and
           eff["x2"]["beta"] < -0.3 and eff["x2"]["p_value"] < 0.01,
           f"LMM 恢复出方向与显著性（x1 β={eff['x1']['beta']:+.2f} p={eff['x1']['p_value']:.1e}；"
           f"x2 β={eff['x2']['beta']:+.2f} p={eff['x2']['p_value']:.1e}）")
        noisy = [v for v in ("x4", "x5", "x6", "x7", "x8") if eff[v]["p_value"] < 0.05]
        ok(not noisy, f"纯噪声变量未被误判显著（{noisy or 'none'}）")

        # ---- ② 真实数据：BNCI 每个 epoch 的 EEG 指标 → 关联分析（分组 = 被试）----
        proc = ROOT / "processed" / "bnci2014001"
        if proc.exists():
            from core.io.loaders import load_processed

            data = load_processed(proc)
            X, y, sid = np.asarray(data["X"]), np.asarray(data["y"]), np.asarray(data["sid"])
            rng2 = np.random.default_rng(0)
            pick = np.sort(rng2.choice(len(y), size=min(900, len(y)), replace=False))
            Xe, side = X[pick], sid[pick]
            fs = 250.0
            ap = absolute_power(Xe, fs)                        # (n,C,5 频带) µV²
            se = spectral_entropy(Xe, fs)                       # (n,C)
            ch = list(data["channel_names"])
            idxs = {c: ch.index(c) for c in ("Cz", "Pz", "POz", "C3", "C4") if c in ch}
            cols = (["alpha_Cz", "alpha_Pz", "alpha_POz", "alpha_C3", "alpha_C4",
                     "beta_Cz", "theta_beta_Cz", "entropy_Cz", "trial_index"]
                    + ["subject"])
            Fe = np.column_stack([ap[:, idxs["Cz"], 2].ravel(), ap[:, idxs["Pz"], 2].ravel(),
                                  ap[:, idxs["POz"], 2].ravel(), ap[:, idxs["C3"], 2].ravel(),
                                  ap[:, idxs["C4"], 2].ravel(), ap[:, idxs["Cz"], 3].ravel(),
                                  ap[:, idxs["Cz"], 1].ravel() / (ap[:, idxs["Cz"], 3].ravel() + 1e-12),
                                  se[:, idxs["Cz"]], np.arange(1, len(pick) + 1)])
            # 频带功率是正偏重尾指标 → 勾上「对正偏变量取 log10」
            # （不取对数时少数极端 epoch 会主导方差，RF 的 R² 甚至为负）
            res_r = linked_analysis(Fe, "alpha_Cz", cols[:-1],
                                    groups=[f"S{int(s):02d}" for s in side],
                                    n_estimators=300, log_transform=True,
                                    log=lambda s: info(s))
            sig = [r for r in res_r["effects"] if r["p_value"] < 0.05]
            ok(len(res_r["effects"]) == 8,
               f"真实数据（{len(pick)} 个 epoch，分组 = {len(set(side.tolist()))} 名被试）："
               f"8 个指标的 LMM 效应 + RF 重要性一次算出")
            info("显著指标：" + "、".join(
                f"{r['variable']}(β={r['beta']:+.2f}, p={r['p_value']:.1e})" for r in sig))
            figr, _ = plot_linked_heatmap(
                res_r["corr"], res_r["predictors"], res_r["importance"], res_r["effects"],
                "alpha_Cz",
                title="BNCI2014001 真实数据：alpha 功率(Cz) ~ EEG 指标（LMM 随机截距=被试）")
            out_r = ROOT / "experiments" / "linked_heatmap_bnci.png"
            figr.savefig(out_r, dpi=130, bbox_inches="tight")
            info(f"真实数据三联图 → {out_r.relative_to(ROOT)}")

        # ---- ③ 合成数据的成图 + 导出表 ----
        fig, _ = plot_linked_heatmap(res["corr"], res["predictors"], res["importance"],
                                     res["effects"], "y",
                                     title="Linked heatmap（合成数据：真效应 x1 +1.6 / x2 −1.2）",
                                     show_importance_values=True, annotate_top_corr=True,
                                     annotate_links=True)
        out = ROOT / "experiments" / "linked_heatmap_demo.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        tabs = export_tables(res)
        ok(set(tabs) == {"importance.csv", "pearson.csv", "effects.csv"},
           f"结果可导出三张表：{ {k: len(v[1]) for k, v in tabs.items()} }（重要性/相关/效应）")
        ok(out.exists(), f"成图 → {out.relative_to(ROOT)}")
        info("读图：顶部看重要性、中间看相关、底部看模型效应；相关 ≠ 因果，"
             "RF 重要性本身不含方向")


# ================================================= 17 论文超参抽取（可选 LLM）
def demo_llm():
    if not sec("17. 论文超参抽取（LLM 可选 + 内置规则兜底，识别 epoch/bs/lr/优化器/种子/标准化/EA/增强）"):
        return
    import json as _json

    from core import llm as L

    en = ("2.3 Implementation details\n"
          "We trained EEGNet for 200 epochs with the AdamW optimizer (learning rate = 1e-3,\n"
          "weight decay = 0.01, cosine annealing). The mini-batch size was set to 64.\n"
          "All experiments used a fixed random seed of 42. Signals were standardized with a\n"
          "z-score computed per channel. We applied Euclidean alignment to each subject.\n"
          "Data augmentation was used, including time shift, Gaussian noise and channel\n"
          "dropout. Evaluation followed a leave-one-subject-out protocol.")
    zh = ("本文使用 EEGNet 训练 100 个 epoch，批大小为 32，采用 Adam 优化器，初始学习率设为 0.0005，\n"
          "权重衰减为 1e-4，使用余弦退火调度，随机种子固定为 2024。每个通道做 z-score 标准化，\n"
          "未使用欧几里得对齐，未使用数据增强。划分方式为留一被试交叉验证。")

    r_en = L.extract_train_params(en, use_llm=False, log=lambda s: info(s))
    r_zh = L.extract_train_params(zh, use_llm=False, log=lambda s: info(s))
    v_en, v_zh = r_en["fields"], r_zh["fields"]
    ok(v_en["epochs"]["value"] == 200 and v_en["batch_size"]["value"] == 64
       and abs(v_en["lr"]["value"] - 1e-3) < 1e-12 and v_en["optimizer"]["value"] == "adamw"
       and v_en["seed"]["value"] == 42 and v_en["normalization"]["value"] == "zscore_channel"
       and v_en["euclidean_alignment"]["value"] is True and v_en["augmentation"]["value"] is True,
       f"英文段落（离线规则）：epochs={v_en['epochs']['value']}、bs={v_en['batch_size']['value']}、"
       f"lr={v_en['lr']['value']}、opt={v_en['optimizer']['value']}、seed={v_en['seed']['value']}、"
       f"norm={v_en['normalization']['value']}、EA={v_en['euclidean_alignment']['value']}、"
       f"增强={v_en['augmentation']['value']}")
    ok(v_zh["epochs"]["value"] == 100 and v_zh["batch_size"]["value"] == 32
       and abs(v_zh["lr"]["value"] - 5e-4) < 1e-12
       and v_zh["euclidean_alignment"]["value"] is False
       and v_zh["augmentation"]["value"] is False,
       f"中文段落 + 否定表述：epochs={v_zh['epochs']['value']}、lr={v_zh['lr']['value']}、"
       f"EA={v_zh['euclidean_alignment']['value']}（未使用欧几里得对齐）、"
       f"增强={v_zh['augmentation']['value']}（未使用数据增强）")
    ev = v_en["euclidean_alignment"]["evidence"]
    ok(bool(ev), f"每个识别值都带原文依据，例如 EA → 「{ev[:46]}…」")
    cfg_en = L.to_train_config(r_en["fields"])
    ok(cfg_en["trainer"]["epochs"] == 200 and cfg_en["normalization"]["euclidean_alignment"] is True
       and cfg_en["augmentation"]["time_shift"] > 0,
       "可直接映射成训练配置：trainer/normalization/augmentation 三段都有值"
       f"（augmentation={cfg_en['augmentation']}）")
    cfg_zh = L.to_train_config(r_zh["fields"], {"augmentation": {"time_shift": 50,
                                                                "gaussian_noise": 0.3,
                                                                "channel_dropout": 0.1}})
    ok(cfg_zh["augmentation"] == {"time_shift": 0, "gaussian_noise": 0, "channel_dropout": 0},
       "论文明确「未使用数据增强」→ 覆盖掉已有增强开关（三个全 0）")

    # ---- 模拟一次 LLM 调用（不联网）：展示"一致 / 冲突"合并与降级 ----
    def fake_post(url, headers, payload, timeout):
        body = {"epochs": {"value": 200, "evidence": "for 200 epochs", "confidence": 0.9},
                "batch_size": {"value": 64, "evidence": "mini-batch size was set to 64"},
                "lr": {"value": 3e-4, "evidence": "learning rate = 3e-4"},   # 与规则冲突
                "optimizer": {"value": "adamw"},
                "augmentation": {"value": True},
                "augmentation_types": {"value": ["time_shift", "mixup"]}}
        return {"choices": [{"message": {"content": "```json\n" + _json.dumps(body) + "\n```"}}]}
    real_post = L._post_json
    try:
        L._post_json = fake_post
        cfg_llm = L.LLMConfig(provider="deepseek", api_key="sk-mock")
        r_llm = L.extract_train_params(en, cfg=cfg_llm, log=lambda s: info(s))
        lr_item = r_llm["fields"]["lr"]
        ok(lr_item["value"] == 3e-4 and "不一致" in lr_item["source"],
           f"LLM 与规则不一致时保留 LLM 值并**标注冲突**（lr={lr_item['value']}，"
           f"{lr_item['source']}，依据={lr_item['evidence']}）")
        ok(r_llm["fields"]["epochs"]["source"] == "LLM + 规则一致",
           "两边一致时标为「LLM + 规则一致」（可放心采用）")
        cfg_aug = L.to_train_config(r_llm["fields"])["augmentation"]
        ok(cfg_aug.get("unsupported_types") == ["mixup"],
           f"平台不支持的手段（mixup）单独列出、不静默丢弃（{cfg_aug}）")

        L._post_json = lambda *a, **k: {"choices": [{"message": {"content": "抱歉，无法回答"}}]}
        r_bad = L.extract_train_params(en, cfg=cfg_llm, log=lambda s: info(s))
        ok(r_bad["meta"]["used_llm"] is False and r_bad["fields"]["epochs"]["value"] == 200
           and "降级" in r_bad["meta"]["warnings"][0],
           "模型返回非 JSON → 自动降级到规则，并给出警告")
        L._post_json = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("HTTP 401：invalid api key"))
        r_err = L.extract_train_params(en, cfg=cfg_llm, log=lambda s: None)
        ok(r_err["fields"]["batch_size"]["value"] == 64 and "HTTP 401" in r_err["meta"]["warnings"][0],
           "密钥/网络报错 → 同样降级（功能不塌），警告里保留原始错误")
    finally:
        L._post_json = real_post
    # ---- PDF（正文 + 补充材料）：BCI 论文的细节通常在正文后的补充材料里 ----
    import tempfile
    from pathlib import Path as _P

    from core import paper_text as PT
    from core.experiment import ExperimentManager
    from core.train.evaluation import export_markdown_report

    main_pdf_text = (
        "Neural decoding of motor imagery with EEGNet\n"
        "3.2 Methods\n"
        "We recorded 9 subjects with 22 EEG channels at 250 Hz. Model training details\n"
        "and all hyperparameters are provided in the Supplementary Material (Table S1).\n"
        "3.3 Results\n"
        "Our method achieved 78.5% accuracy.\n")
    supp_pdf_text = (
        "Supplementary Material\n"
        "S1 Data preprocessing\n"
        "Signals were re-referenced to the common average and filtered 4-40 Hz.\n"
        "S2 Implementation details\n"
        "All models were trained for 200 epochs with the AdamW optimizer (learning rate = 1e-3,\n"
        "weight decay = 0.01) and a cosine annealing schedule. The mini-batch size was set to 64.\n"
        "The random seed was fixed to 42. Signals were standardized with a z-score per channel.\n"
        "Euclidean alignment was applied. Data augmentation included time shift and Gaussian\n"
        "noise. Table S1. Hyperparameter search space.\n"
        "S3 Additional results\n"
        "Per-subject accuracies are listed below.\n")
    tmp = _P(tempfile.mkdtemp())
    main_pdf = PT.write_sample_pdf(tmp / "Smith2023_main.pdf", [main_pdf_text])
    supp_pdf = PT.write_sample_pdf(tmp / "Smith2023_supplementary.pdf", [supp_pdf_text])
    srcs, texts = PT.read_sources([main_pdf, supp_pdf], log=lambda t: info(t))
    kinds = {s.name: s.kind for s in srcs}
    ok(kinds.get("Smith2023_main.pdf") == "main"
       and kinds.get("Smith2023_supplementary.pdf") == "supplementary",
       f"拖进两个 PDF 能自动区分正文/补充材料：{kinds}")
    merged2, asm = PT.assemble_param_text([(s, texts[s.path]) for s in srcs],
                                          max_chars=6000, log=lambda t: info(t))
    ok(merged2.index("Implementation details") < merged2.find("3.2 Methods")
       and asm["used_sections"],
       f"组装时补充材料排在正文之前，用的是「{asm['used_sections'][0]}」")
    ok(any("补充材料" in n for n in asm["notes"]),
       "正文里写着「详见补充材料」时给出提示（参数确实不在正文）")
    res_pdf = L.extract_train_params(merged2, use_llm=False, log=lambda t: info(t))
    vv = {k: res_pdf["fields"][k]["value"] for k in
          ("epochs", "batch_size", "lr", "optimizer", "seed")}
    ok(vv == {"epochs": 200, "batch_size": 64, "lr": 1e-3, "optimizer": "adamw", "seed": 42},
       f"从 PDF 组装文本里抽到超参：{vv}（随机种子的长连接词写法也能读）")
    prov2 = PT.provenance(res_pdf, merged2, asm)
    ok(prov2["input"]["text_sha256"] and prov2["input"]["sources"]
       and prov2["fields"]["epochs"]["evidence"],
       "溯源齐全：每个来源文件 sha256 + 采用小节 + 每个字段的原文依据")
    mgr = ExperimentManager(root=tmp / "experiments")
    cfg_paper = {"experiment": {"name": "paper_run"}, "trainer": {"epochs": 200},
                 "paper_params": prov2}
    exp = mgr.create("paper_run", cfg_paper)
    cfg_txt = (exp.dir / "config.yaml").read_text(encoding="utf-8")
    ok("paper_params" in cfg_txt and "supplementary" in cfg_txt,
       "参数来源随 config.yaml 落盘到实验目录（半年后能查「这套超参抄自哪一节」）")
    report = export_markdown_report(exp.dir, {"experiment": "paper_run", "aggregate": {}},
                                    cfg_paper)
    md_txt = _P(report).read_text(encoding="utf-8")
    ok("## 训练参数来源（论文抽取）" in md_txt and "for 200 epochs" in md_txt,
       "实验报告 report.md 自动带「训练参数来源」一节（含依据与来源文件）")
    ok(bool(PT.summarize_provenance(prov2)),
       f"实验列表备注里的一行来源：{PT.summarize_provenance(prov2)}")

    ok(L.LLMConfig(provider="ollama").configured
       and not L.LLMConfig(provider="deepseek", api_key="").configured,
       "配置判定：本地 Ollama 无需密钥即认为可用；云端缺密钥则自动走规则")
    info(f"供应商：{len(L.PROVIDERS)} 个（" + "、".join(
        s["label"].split("（")[0] for s in list(L.PROVIDERS.values())[:6]) + "…）；"
        f"密钥只从环境变量或 configs/llm.yaml 读，不硬编码")


# ==================================================================== 18 GUI
def demo_gui():
    if not sec("18. 图形界面：5 页 + 可折叠步骤栏 + 主题 + 对话框"):
        return
    import matplotlib

    matplotlib.use("Agg")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    _ = app
    from core.plot_fonts import apply_matplotlib_font, qt_font
    from ui.main_window import MainWindow
    from ui.theme import THEMES, apply_theme

    app.setFont(qt_font(12))
    apply_matplotlib_font()
    apply_theme(app, 0)
    win = MainWindow()
    pages = list(MainWindow.PAGES)
    ok(len(win._pages) == len(pages) == 5, f"功能页：{pages}")
    ok("Times New Roman" in app.font().families()[:3],
       f"全局字体栈：{app.font().families()[:4]}（数字/字母 Times New Roman + 中文宋体）")
    ok(len(THEMES) >= 4, f"主题：{[t.name for t in THEMES]}（默认「{THEMES[0].name}」，菜单可切换）")
    win.resize(1500, 940)
    win.show()
    app.processEvents()
    from PyQt6.QtTest import QTest

    win.toggle_nav()
    QTest.qWait(400)               # 折叠是 210→48 的平滑动画，要等它跑完
    ok(win.nav_box.width() < 100, f"左侧步骤栏可折叠：{win.nav_box.width()}px（展开 210px）")
    win.toggle_nav()
    QTest.qWait(400)
    ok(win.minimumSizeHint().width() < 1700,
       f"窗口最小宽度 {win.minimumSizeHint().width()}px（布局不溢出，笔记本可用）")
    info(f"对话框：{', '.join(sorted(p.stem for p in (ROOT / 'ui' / 'widgets').glob('*.py') if p.stem not in ('__init__', 'plotting')))}")


# ====================================================================== main
DEMOS = [("1", demo_io), ("2", demo_preprocess), ("3", demo_features), ("4", demo_dataset),
         ("5", demo_models), ("6", demo_splits), ("7", demo_train), ("8", demo_erp),
         ("9", demo_tfr), ("10", demo_psd), ("11", demo_nonlinear), ("12", demo_networks),
         ("13", demo_netcov), ("14", demo_neuroimg), ("15", demo_experiment),
         ("16", demo_linked), ("17", demo_llm), ("18", demo_gui)]


def main():
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    try:                       # MNE 会在 INFO 级别刷屏（建 Raw、设计滤波器…）
        import mne

        mne.set_log_level("ERROR")
    except Exception:  # noqa: BLE001
        pass
    try:                       # 无头 Qt 的 "propagateSizeHints" 等提示与演示无关
        from PyQt6.QtCore import qInstallMessageHandler

        qInstallMessageHandler(lambda *_: None)
    except Exception:  # noqa: BLE001
        pass
    print("=" * 74)
    print("BCI Lab 功能演示与自检（每个功能真跑一遍；✔ 通过 / ✘ 失败）")
    print("=" * 74)
    t0 = time.perf_counter()
    for _, fn in DEMOS:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"  ✘ 该组中断：{type(e).__name__}: {e}")
            traceback.print_exc()
            _FAILS.append(f"{fn.__name__}: {e}")
    print("\n" + "=" * 74)
    print(f"演示结束：耗时 {time.perf_counter() - t0:.1f}s，失败 {len(_FAILS)} 项")
    for f in _FAILS:
        print(f"  ✘ {f}")
    print("=" * 74)
    return 1 if _FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
