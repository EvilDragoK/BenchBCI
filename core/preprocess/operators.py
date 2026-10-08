"""预处理算子：每一步一个类，参数自描述（schema 驱动 GUI 表单生成）。

stage 说明：
- 'raw'   : 作用于 mne.io.Raw（滤波、重参考、重采样、ICA 等）
- 'epochs': 作用于 mne.Epochs（分段后的剔除等）

Epoching 算子负责 Raw → Epochs 的转换，其后只能接 'epochs' 算子。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

#: 可识别的电极坐标文件后缀（mne.channels.read_custom_montage 支持）
MONTAGE_SUFFIXES = (".elc", ".csd", ".sfp", ".elp", ".bvef", ".loc", ".hpts",
                    ".csv", ".tsv", ".txt")


def _as_list(v) -> list:
    if v is None or v == "" or v == []:
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    if isinstance(v, str):
        return [s.strip() for s in v.replace(";", ",").split(",") if s.strip()]
    return [v]


def _as_float_list(v) -> list[float]:
    return [float(x) for x in _as_list(v)]


def _as_int_list(v) -> list[int]:
    return [int(float(x)) for x in _as_list(v)]


def resolve_montage_file(spec: str) -> Path | None:
    """把「导联系统」参数解析为坐标文件路径；不是文件时返回 None。

    支持相对路径（相对项目根目录）与 ``~`` 展开。
    """
    txt = str(spec or "").strip().strip("\"'")
    if not txt or not txt.lower().endswith(MONTAGE_SUFFIXES):
        return None
    from .. import PROJECT_ROOT

    p = Path(txt).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p if p.is_file() else None


def channel_positions(raw) -> dict[str, list[float]]:
    """取各 EEG 通道的三维坐标（米，头部坐标系）；未设导联时返回空字典。"""
    from mne.io.constants import FIFF

    out: dict[str, list[float]] = {}
    for ch in raw.info["chs"]:
        if ch["kind"] != FIFF.FIFFV_EEG_CH:
            continue
        xyz = [float(v) for v in ch["loc"][:3]]
        if np.all(np.isfinite(xyz)) and any(abs(v) > 1e-12 for v in xyz):
            out[ch["ch_name"]] = xyz
    return out


def _rest_reference(raw, spacing_mm: float = 20.0):
    """REST（无穷远参考）——用球模型近似头几何算前向解，再做参考变换。

    相比 CAR，REST 不会把参考电极自身的活动混进所有通道，跨被试比较更稳。
    需要电极坐标（流水线里先加「设置导联系统」）。

    实现要点：球模型**按当前电极分布自适应**取球心与半径——固定半径（如 0.09 m）
    会把 Fp1 这类前额电极甩到球外，前向矩阵出现 NaN，REST 直接失败。
    """
    import mne

    pos_map = channel_positions(raw)
    if len(pos_map) < 8:
        raise ValueError("REST 需要电极坐标：请先添加「设置导联系统」并确保通道名匹配")
    pos = np.asarray(list(pos_map.values()), dtype=float)
    center = pos.mean(axis=0)
    radius = float(np.linalg.norm(pos - center, axis=1).max()) * 1.05

    sphere = mne.make_sphere_model(r0=tuple(center), head_radius=radius,
                                   verbose="ERROR")
    src = mne.setup_volume_source_space(pos=float(spacing_mm), sphere=sphere,
                                        mindist=2.0, verbose="ERROR")
    fwd = mne.make_forward_solution(raw.info, trans=None, src=src, bem=sphere,
                                    eeg=True, meg=False, verbose="ERROR")
    lead = np.asarray(fwd["sol"]["data"], dtype=float)
    if not np.isfinite(lead).all():        # 兜底：球再放大一点重算一次
        sphere = mne.make_sphere_model(r0=tuple(center), head_radius=radius * 1.15,
                                       verbose="ERROR")
        src = mne.setup_volume_source_space(pos=float(spacing_mm), sphere=sphere,
                                           mindist=2.0, verbose="ERROR")
        fwd = mne.make_forward_solution(raw.info, trans=None, src=src, bem=sphere,
                                        eeg=True, meg=False, verbose="ERROR")
    raw.set_eeg_reference("REST", forward=fwd, verbose="ERROR")


class Operator:
    """算子基类。schema: {参数名: (类型, 默认值, 中文说明[, 选项列表])}"""

    name = "operator"
    title = "算子"
    stage = "raw"
    schema: dict[str, tuple] = {}

    def __init__(self, params: dict | None = None):
        self.params = {k: self.schema[k][1] for k in self.schema}
        if params:
            for k, v in params.items():
                if k in self.schema:
                    self.params[k] = v

    # -- 类型转换钩子：把 YAML / GUI 字符串转回参数类型 -------------------
    def coerce(self, k: str, v):
        t = self.schema[k][0]
        if t == "int":
            return int(float(v))
        if t == "float":
            return float(v)
        if t == "bool":
            return bool(v) if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
        if t == "list_float":
            return _as_float_list(v)
        if t == "list_int":
            return _as_int_list(v)
        if t == "list_str":
            return [str(x) for x in _as_list(v)]
        return str(v)

    def set_param(self, k: str, v):
        if k in self.schema:
            self.params[k] = self.coerce(k, v)

    def apply(self, obj, ctx: dict):
        raise NotImplementedError

    def describe(self) -> str:
        parts = []
        for k, spec in self.schema.items():
            parts.append(f"{k}={self.params[k]}")
        return f"{self.title}({', '.join(parts)})"


# ============================== Raw 阶段算子 ==============================

class SetMontage(Operator):
    """设置导联（电极）位置：标准导联系统名 **或** 坐标文件路径。

    坐标会写进 raw.info，供地形图、插值坏导、REST 重参考使用；
    流水线还会把坐标存进 meta.json（导联定位可追溯/可复现）。
    """
    name, title = "set_montage", "设置导联系统"
    schema = {"montage": ("str", "standard_1005", "导联系统名或坐标文件路径")}

    def apply(self, raw, ctx):
        spec = str(self.params["montage"]).strip()
        path = resolve_montage_file(spec)
        if path is not None:
            from mne.channels import read_custom_montage

            mont = read_custom_montage(str(path))
            raw.set_montage(mont, on_missing="ignore")
            matched = len(set(mont.ch_names) & set(raw.ch_names))
            ctx["montage"] = {"source": "file", "path": str(path),
                              "n_in_file": len(mont.ch_names)}
            if matched == 0:
                ctx.setdefault("warnings", []).append(
                    f"坐标文件 {path.name} 中没有任何通道名与数据匹配（检查电极命名）")
        else:
            name = spec or "standard_1005"
            raw.set_montage(name, on_missing="ignore")
            ctx["montage"] = {"source": "standard", "name": name}

        pos = channel_positions(raw)
        ctx["channel_positions"] = pos
        ctx["montage"]["n_matched"] = len(pos)
        return raw


class PickEEG(Operator):
    name, title = "pick_eeg", "仅保留 EEG 通道"
    schema = {"keep_eog": ("bool", False, "同时保留 EOG 通道")}

    def apply(self, raw, ctx):
        import mne

        picks = mne.pick_types(raw.info, eeg=True, eog=self.params["keep_eog"], exclude=[])
        picked = [raw.ch_names[i] for i in picks]
        if picked:
            raw.pick(picked)
        return raw


class DropChannels(Operator):
    name, title = "drop_channels", "删除通道"
    schema = {"ch_names": ("list_str", [], "通道名列表")}

    def apply(self, raw, ctx):
        names = [c for c in self.params["ch_names"] if c in raw.ch_names]
        if names:
            raw.drop_channels(names)
        return raw


class MarkBadChannels(Operator):
    name, title = "mark_bad_channels", "标记坏导"
    schema = {"ch_names": ("list_str", [], "坏导名列表")}

    def apply(self, raw, ctx):
        names = [c for c in self.params["ch_names"] if c in raw.ch_names]
        raw.info["bads"] = sorted(set(raw.info["bads"]) | set(names))
        ctx.setdefault("bad_channels", []).extend(names)
        return raw


class InterpolateBads(Operator):
    name, title = "interpolate_bads", "插值坏导"
    schema = {"reset_bads": ("bool", True, "插值后清空坏导标记")}

    def apply(self, raw, ctx):
        if raw.info["bads"]:
            raw.interpolate_bads(reset_bads=self.params["reset_bads"])
        return raw


class NotchFilter(Operator):
    name, title = "notch_filter", "陷波滤波（工频）"
    schema = {"freqs": ("list_float", [50.0], "陷波频率 (Hz)")}

    def apply(self, raw, ctx):
        freqs = self.params["freqs"]
        if freqs:
            raw.notch_filter(freqs=freqs, picks=None)
        return raw


class BandpassFilter(Operator):
    """带通/低通/高通滤波：l_freq=0 → 仅低通；h_freq=0 → 仅高通。"""
    name, title = "bandpass_filter", "滤波（带通/低通/高通）"
    schema = {
        "l_freq": ("float", 4.0, "低截止 (Hz)，0=仅低通"),
        "h_freq": ("float", 40.0, "高截止 (Hz)，0=仅高通"),
        "method": ("choice", "fir", "滤波器类型", ["fir", "iir"]),
        "iir_order": ("int", 4, "IIR 阶数（Butterworth）"),
    }

    def apply(self, raw, ctx):
        lf, hf = float(self.params["l_freq"]), float(self.params["h_freq"])
        method = self.params["method"]
        iir = {"order": int(self.params["iir_order"]), "ftype": "butter"} if method == "iir" else None
        if lf > 0 and hf > 0:  # 带通
            raw.filter(lf, hf, method=method, iir_params=iir, phase="zero" if method == "fir" else None)
        elif hf > 0:  # 仅低通（如 ≤120 Hz、≤45 Hz）
            raw.filter(None, hf, method=method, iir_params=iir, phase="zero" if method == "fir" else None)
        elif lf > 0:  # 仅高通（如 ≥0.1 Hz）
            raw.filter(lf, None, method=method, iir_params=iir, phase="zero" if method == "fir" else None)
        return raw


class GaussianSmooth(Operator):
    """高斯平滑（低通）：σ 以秒为单位，按采样率换算为采样点。"""
    name, title = "gaussian_filter", "高斯滤波（低通平滑）"
    schema = {"sigma_s": ("float", 0.02, "高斯核 σ (秒)")}

    def apply(self, raw, ctx):
        from scipy.ndimage import gaussian_filter1d

        sigma = max(int(round(float(self.params["sigma_s"]) * raw.info["sfreq"])), 1)
        raw._data[:] = gaussian_filter1d(raw.get_data(), sigma, axis=-1)
        return raw


class Resample(Operator):
    name, title = "resample", "重采样"
    schema = {"sfreq": ("float", 250.0, "目标采样率 (Hz)")}

    def apply(self, raw, ctx):
        raw.resample(float(self.params["sfreq"]))
        return raw


class Normalize(Operator):
    """归一化（多方案，对应文献常见 Alignment 方案）。

    注意：此处统计量来自整段数据（含训练+测试），仅适合显示与对比实验；
    严格防泄漏的训练归一化请在训练配置 normalization 里设置（只用训练折统计量）。
    """
    name, title = "normalize", "归一化（多方案）"
    schema = {
        "mode": ("choice", "zscore", "方式",
                 ["zscore", "minmax_01", "minmax_pm", "p95", "x_div", "iqr",
                  "quantile", "alaw", "ema", "zero_mean"]),
        "p": ("float", 95.0, "p95 模式：百分位"),
        "constant": ("float", 100.0, "x_div 模式：除数（如 x/100）"),
        "A": ("float", 87.56, "alaw 压扩系数"),
        "alpha": ("float", 0.01, "ema 平滑系数（新样本权重）"),
    }

    def apply(self, raw, ctx):
        import mne

        data = raw.get_data()  # (C, T)
        mode = self.params["mode"]
        data = _normalize_2d(data, mode, p=float(self.params["p"]),
                             constant=float(self.params["constant"]),
                             A=float(self.params["A"]), alpha=float(self.params["alpha"]))
        info = raw.info.copy()
        ann = raw.annotations
        new = mne.io.RawArray(data, info)
        new.set_annotations(ann)
        return new


def _normalize_2d(data, mode, p=95.0, constant=100.0, A=87.56, alpha=0.01):
    """对 (C, T) 连续数据沿时间维做逐通道归一化（供 Normalize 算子复用）。"""
    eps = 1e-12
    if mode == "zscore":
        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + eps)
    elif mode == "minmax_01":
        lo = data.min(axis=1, keepdims=True)
        data = (data - lo) / (data.max(axis=1, keepdims=True) - lo + eps)
    elif mode == "minmax_pm":
        lo = data.min(axis=1, keepdims=True)
        data = (data - lo) / (data.max(axis=1, keepdims=True) - lo + eps) * 2 - 1
    elif mode == "p95":
        p95 = np.percentile(np.abs(data), p, axis=1, keepdims=True)
        data = data / (p95 + eps)
    elif mode == "x_div":
        data = data / (abs(constant) + eps)
    elif mode == "iqr":
        med = np.median(data, axis=1, keepdims=True)
        q75, q25 = np.percentile(data, [75, 25], axis=1, keepdims=True)
        data = (data - med) / (q75 - q25 + eps)
    elif mode == "quantile":  # 秩变换 → 标准正态分位数
        from scipy.stats import norm

        T = data.shape[-1]
        order = np.argsort(data, axis=-1)
        ranks = np.empty_like(order, dtype=np.float64)
        np.put_along_axis(ranks, order, np.arange(1, T + 1, dtype=np.float64), axis=-1)
        data = norm.ppf((ranks - 0.5) / T)
    elif mode == "alaw":
        u = np.abs(data).max(axis=1, keepdims=True)
        u = np.where(u > 0, u, 1.0)  # 全零信号安全
        data = np.sign(data) * np.log(1 + A * np.abs(data) / u) / np.log(1 + A)
    elif mode == "ema":
        from scipy.signal import lfilter

        alpha = min(max(alpha, 1e-4), 1.0)
        b, a = [alpha], [1.0, -(1.0 - alpha)]
        x0 = data[..., :1]
        mu = lfilter(b, a, data - x0, axis=-1) + x0
        ex2 = lfilter(b, a, (data - x0) ** 2, axis=-1) + x0 ** 2
        var = np.clip(ex2 - mu ** 2, 0.0, None)
        data = (data - mu) / np.sqrt(var + eps)
    elif mode == "zero_mean":
        data = data - data.mean(axis=1, keepdims=True)
    else:
        raise ValueError(f"未知归一化方式：{mode}")
    return data


class ReReference(Operator):
    """重参考：共平均（CAR）/ 指定电极 / REST（无穷远）/ 双极导联。

    - ``average``：共平均参考，最常用，去共模噪声
    - ``channels``：指定参考电极（如乳突 TP9/TP10，可多个）
    - ``rest``：无穷远参考（球模型近似），跨被试比较更稳，需要电极坐标
    - ``bipolar``：按电极对生成双极导联（衍生通道，原通道保留）
    """
    name, title = "re_reference", "重参考"
    schema = {
        "mode": ("choice", "average", "参考方式",
                 ["average", "channels", "rest", "bipolar"]),
        "channels": ("list_str", [], "参考通道（channels 模式，可多个）"),
        "pairs": ("list_str", [], "双极对（bipolar 模式，如 Fp1-Fp2, C3-C4）"),
        "projection": ("bool", False, "前馈投影（不改变数据本身，可逆；REST 不适用）"),
    }

    def apply(self, raw, ctx):
        mode = self.params["mode"]
        proj = bool(self.params["projection"])
        if mode == "average":
            raw.set_eeg_reference("average", projection=proj, verbose="ERROR")
            ctx["reference"] = "average(CAR)"
        elif mode == "channels":
            chs = [c for c in self.params["channels"] if c in raw.ch_names]
            if not chs:
                raise ValueError("channels 参考模式需要至少一个有效参考通道名")
            raw.set_eeg_reference(chs, projection=proj, verbose="ERROR")
            ctx["reference"] = "channels:" + ",".join(chs)
        elif mode == "rest":
            _rest_reference(raw)
            ctx["reference"] = "REST（球模型近似）"
        else:  # bipolar
            import mne

            pairs = [str(p) for p in self.params["pairs"] if "-" in str(p)]
            if not pairs:
                raise ValueError("bipolar 模式需要形如 Fp1-Fp2, C3-C4 的电极对")
            created = []
            for pair in pairs:
                a, b = [s.strip() for s in pair.split("-", 1)]
                if a in raw.ch_names and b in raw.ch_names:
                    mne.set_bipolar_reference(raw, a, [b], ch_name=f"{a}-{b}",
                                              drop_refs=False, copy=False,
                                              verbose="ERROR")
                    created.append(f"{a}-{b}")
            if not created:
                raise ValueError("双极对里没有任何「两端都存在」的通道名")
            ctx["reference"] = "bipolar"
            ctx["bipolar_channels"] = created
        return raw


class ICA(Operator):
    """ICA 伪迹剔除：拟合 + 手动/自动剔除成分，作用于 Raw（分段前）。

    自动识别两种模式（可同时开）：
    - ``autodetect_eog``：需要真正的 EOG 通道
    - ``autodetect_artifact``：**无需 EOG**——用峰度 / 1s 自相关 / 额区通道代理相关 /
      肌电高频占比 / 幅值比五个判据识别眨眼、眼动、肌电、电极尖峰，并给出理由

    注意：ICA 在整名被试数据上拟合（与标签无关），如需严格折内拟合请在训练
    阶段对训练折单独处理——GUI 默认在流水线中按被试整体拟合。
    """
    name, title = "ica", "ICA 伪迹剔除"
    schema = {
        "n_components": ("int", 10, "成分数"),
        "method": ("choice", "fastica", "算法", ["fastica", "picard", "infomax"]),
        "random_state": ("int", 97, "随机种子"),
        "exclude": ("list_int", [], "手动剔除成分编号"),
        "autodetect_eog": ("bool", False, "自动检测眼电成分（需 EOG 通道）"),
        "autodetect_artifact": ("bool", False, "自动识别伪迹成分（无需 EOG）"),
        "min_hits": ("int", 2, "自动识别需同时命中的判据数（1=更激进，2=推荐）"),
    }

    def apply(self, raw, ctx):
        from mne.preprocessing import ICA as MneICA

        from .artifact import component_features, suggest_exclude

        method = self.params["method"]
        if method == "picard":
            try:
                import picard  # noqa: F401
            except ImportError:
                method = "fastica"
        ica = MneICA(n_components=int(self.params["n_components"]),
                     method=method, random_state=int(self.params["random_state"]),
                     max_iter="auto")
        ica.fit(raw, verbose="ERROR")

        # 成分诊断：无论是否自动剔除都算一遍，GUI 可用来浏览/复核
        feats = None
        try:
            feats = component_features(raw, ica)
            ctx["ica_features"] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in feats.items()
            }
        except Exception as e:  # noqa: BLE001 - 诊断失败不影响主流程
            ctx.setdefault("warnings", []).append(f"ICA 成分诊断失败：{e}")

        exclude = [int(e) for e in self.params["exclude"]]
        if self.params["autodetect_eog"]:
            import mne

            if mne.pick_types(raw.info, eog=True, exclude=[]).size:
                auto, _ = ica.find_bads_eog(raw, verbose="ERROR")
                exclude = sorted(set(exclude) | set(int(x) for x in auto))
        if self.params["autodetect_artifact"] and feats is not None:
            auto, reasons = suggest_exclude(feats, min_hits=int(self.params["min_hits"]))
            # 命中 1 条判据的作为"候选"报告（不剔除），方便人工在浏览器里复核
            cand, cand_reasons = suggest_exclude(feats, min_hits=1)
            exclude = sorted(set(exclude) | set(auto))
            ctx["ica_suggested"] = auto
            ctx["ica_reasons"] = {int(k): v for k, v in reasons.items()}
            ctx["ica_candidates"] = cand
            ctx["ica_candidate_reasons"] = {int(k): v for k, v in cand_reasons.items()}

        ica.exclude = sorted(set(int(e) for e in exclude))
        ica.apply(raw, verbose="ERROR")
        ctx.setdefault("ica_exclude", []).extend(ica.exclude)
        ctx["ica"] = ica  # 供 GUI 浏览地形图/时序
        return raw


# ============================ Raw → Epochs 转换 ============================

class RejectBadSegments(Operator):
    """坏**时间段**检测：滑动窗找「大振幅段 / 平坦段」，写成 ``BAD_segment`` 注释。

    与「振幅阈值剔除」的区别：那个是逐 trial 事后剔除，这个是**在连续数据上**
    定位坏时间段（含跨 trial 的连续干扰），分段时按注释自动丢弃重叠的 epoch。

    判据（滑动窗内）：
    - 峰峰值 > ``thresh_uv`` 的通道占比 ≥ ``ch_frac`` → 大振幅段（运动/肌电爆发）
    - 峰峰值 < ``flat_uv`` 的通道占比 ≥ ``ch_frac`` → 平坦段（电极脱落/饱和）
    """
    name, title = "reject_bad_segments", "坏段检测（连续时间段）"
    stage = "raw"
    schema = {
        "win_s": ("float", 1.0, "滑动窗长 (s)"),
        "step_s": ("float", 0.5, "滑动步长 (s)"),
        "thresh_uv": ("float", 200.0, "大振幅阈值 (µV)，0=不查"),
        "flat_uv": ("float", 0.5, "平坦阈值 (µV)，0=不查"),
        "min_dur_s": ("float", 0.3, "坏段最短时长 (s)"),
        "ch_frac": ("float", 0.5, "超阈值通道占比达到该值即判坏"),
    }

    def apply(self, raw, ctx):
        import mne
        from numpy.lib.stride_tricks import sliding_window_view

        data = raw.get_data() * 1e6                       # µV (C, T)
        fs = float(raw.info["sfreq"])
        win = max(int(round(float(self.params["win_s"]) * fs)), 2)
        step = max(int(round(float(self.params["step_s"]) * fs)), 1)
        if data.shape[-1] < win:
            ctx.setdefault("bad_segments", [])
            return raw

        # (C, n_win, win) 视图 → 峰峰值 (C, n_win)
        view = sliding_window_view(data, win, axis=-1)[:, ::step]
        ptp = view.max(-1) - view.min(-1)
        n_ch = ptp.shape[0]
        frac = np.zeros(ptp.shape[1])
        th, fl = float(self.params["thresh_uv"]), float(self.params["flat_uv"])
        if th > 0:
            frac = np.maximum(frac, (ptp > th).sum(axis=0) / n_ch)
        if fl > 0:
            frac = np.maximum(frac, (ptp < fl).sum(axis=0) / n_ch)
        bad = frac >= float(self.params["ch_frac"])

        segs, i = [], 0
        while i < len(bad):
            if bad[i]:
                j = i
                while j + 1 < len(bad) and bad[j + 1]:
                    j += 1
                t0 = i * step / fs
                t1 = min((j * step + win) / fs, raw.n_times / fs)
                if t1 - t0 >= float(self.params["min_dur_s"]):
                    segs.append((round(t0, 3), round(t1, 3)))
                i = j + 1
            else:
                i += 1

        if segs:
            ann = mne.Annotations(onset=[s[0] for s in segs],
                                  duration=[s[1] - s[0] for s in segs],
                                  description=["BAD_segment"] * len(segs))
            raw.set_annotations(raw.annotations + ann)
        ctx["bad_segments"] = segs
        ctx["bad_segment_seconds"] = round(sum(s[1] - s[0] for s in segs), 2)
        return raw


class ClipOutliers(Operator):
    """去极值（幅值裁剪）：把个别极端采样点夹到稳健范围内，不改变整体波形。

    真实数据里总有零星尖峰（电极移动、吞咽、数字伪迹），它们会主导后续的
    统计量（尤其 z-score 与协方差）。裁剪比"整段丢弃"温和得多。

    方式：``mad`` = 中位 ± k·MAD（推荐）；``sigma`` = 均值 ± k·σ；
    ``quantile`` = 逐通道 |x| 的 p 分位。

    ``min_uv`` 是**下限保护**：裁剪范围不小于该值，避免在低噪声记录上把正常
    脑电（alpha 波、眨眼）也削平——只治真正的极端采样点。
    """
    name, title = "clip_outliers", "去极值（幅值裁剪）"
    stage = "raw"
    schema = {
        "mode": ("choice", "mad", "方式", ["mad", "sigma", "quantile", "none"]),
        "k": ("float", 12.0, "倍数（mad / sigma 方式）"),
        "min_uv": ("float", 200.0, "裁剪下限（µV，裁剪范围不小于此值）"),
        "quantile": ("float", 99.9, "分位（quantile 方式，如 99.9）"),
        "scope": ("choice", "channel", "统计范围", ["channel", "global"]),
    }

    def apply(self, raw, ctx):
        import mne

        data = raw.get_data()
        mode = str(self.params["mode"])
        if mode == "none":
            ctx["clipped_samples"] = 0
            return raw
        per_ch = self.params["scope"] == "channel"
        if mode == "mad":
            k = float(self.params["k"])
            if per_ch:
                med = np.median(data, axis=-1, keepdims=True)
                mad = np.median(np.abs(data - med), axis=-1, keepdims=True) * 1.4826
            else:
                med = float(np.median(data))
                mad = float(np.median(np.abs(data - med))) * 1.4826
            lo, hi = med - k * mad, med + k * mad
        elif mode == "sigma":
            k = float(self.params["k"])
            mu = data.mean(-1, keepdims=True) if per_ch else float(data.mean())
            sd = data.std(-1, keepdims=True) if per_ch else float(data.std())
            lo, hi = mu - k * sd, mu + k * sd
        else:  # quantile
            q = float(self.params["quantile"])
            if per_ch:
                a = np.percentile(np.abs(data), q, axis=-1, keepdims=True)
            else:
                a = float(np.percentile(np.abs(data), q))
            lo, hi = -a, a

        floor = abs(float(self.params["min_uv"])) * 1e-6
        if floor > 0:
            lo = np.minimum(lo, -floor)
            hi = np.maximum(hi, floor)
        clipped = np.clip(data, lo, hi)
        n_clip = int(np.count_nonzero(clipped != data))
        ctx["clipped_samples"] = n_clip
        ctx["clipped_fraction"] = round(n_clip / max(data.size, 1), 8)
        if n_clip == 0:
            return raw

        info = raw.info.copy()
        new = mne.io.RawArray(clipped, info)
        new.set_annotations(raw.annotations)
        return new


class ArtifactRegression(Operator):
    """回归法去伪迹：用代理通道（EOG / 额区 / 自定）线性回归去除其成分。

    与 ICA 互补——ICA 是盲源分离（适合眨眼/心电），回归法需要明确的参考通道，
    胜在**快、可解释、无随机性**。没有 EOG 通道时自动用额区通道做代理。
    """
    name, title = "artifact_regression", "回归法去伪迹"
    stage = "raw"
    schema = {
        "proxies": ("list_str", [], "代理通道（空=自动：EOG→额区）"),
        "band_lower": ("float", 0.5, "回归频段下限 (Hz)，0=不限"),
        "band_upper": ("float", 10.0, "回归频段上限 (Hz)，0=不限"),
    }

    def apply(self, raw, ctx):
        from .artifact import regress_artifacts

        proxies = _as_list(self.params["proxies"]) or None
        lo = float(self.params["band_lower"]) or None
        hi = float(self.params["band_upper"]) or None
        return regress_artifacts(raw, proxies, lower=lo, upper=hi, ctx=ctx)


class Epoching(Operator):
    """事件提取与分段：从 annotations 提取事件（BNCI 加载器已写入）。"""
    name, title = "epoching", "事件提取与分段"
    stage = "raw2epochs"
    schema = {
        "tmin": ("float", 0.5, "tmin (s，相对 cue)"),
        "tmax": ("float", 2.5, "tmax (s，相对 cue)"),
        "baseline": ("bool", True, "基线校正（tmin~0，需 tmin<0）"),
        "classes": ("list_str", [], "保留类别（空=全部）"),
        "reject_by_annotation": ("bool", True, "按注释剔除坏段（BAD_* 标注）"),
    }

    def apply(self, raw, ctx):
        import mne

        # 坏段/边界注释不是"类别"，必须排除，否则类别表里会混进 BAD_segment
        valid = [str(d) for d in sorted(set(raw.annotations.description))
                 if not str(d).lower().startswith(("bad", "boundary"))]
        keep = _as_list(self.params["classes"])
        if keep:
            keep = [c for c in keep if c in valid]
            if not keep:
                raise ValueError(f"annotations 中不含指定类别：{valid}")
            event_id = {c: i + 1 for i, c in enumerate(keep)}
        elif valid:
            event_id = {c: i + 1 for i, c in enumerate(valid)}
        else:
            raise ValueError(f"未找到可用事件注释（只有 {sorted(set(map(str, raw.annotations.description)))}）")

        events, ev_map = mne.events_from_annotations(raw, event_id=event_id, verbose="ERROR")
        if len(events) == 0:
            raise ValueError("未从 annotations 中提取到任何事件，无法分段")
        tmin, tmax = float(self.params["tmin"]), float(self.params["tmax"])
        baseline = (None, 0) if (self.params["baseline"] and tmin < 0) else None
        reject_ann = bool(self.params.get("reject_by_annotation", True))
        n_sel = int(np.isin(events[:, 2], list(ev_map.values())).sum())
        epochs = mne.Epochs(raw, events, event_id=ev_map, tmin=tmin, tmax=tmax,
                            baseline=baseline, preload=True, proj=False,
                            reject=None, event_repeated="drop",
                            reject_by_annotation=reject_ann, verbose="ERROR")
        if reject_ann and len(epochs) < n_sel:
            # 被 BAD_* 注释（坏段检测）丢弃的 trial 数
            ctx["dropped_bad_segments"] = (ctx.get("dropped_bad_segments", 0)
                                           + (n_sel - len(epochs)))
        ctx["classes"] = {k: int(v) for k, v in ev_map.items()}
        ctx.setdefault("n_trials_before", 0)
        ctx["n_trials_before"] += len(epochs)
        return epochs


# ============================== Epochs 阶段算子 ==============================

class AmplitudeReject(Operator):
    """振幅阈值剔除：epoch 内 EEG 峰峰值超过阈值即剔除。"""
    name, title = "amplitude_reject", "振幅阈值剔除"
    stage = "epochs"
    schema = {"threshold_uv": ("float", 150.0, "峰峰值阈值 (µV)")}

    def apply(self, epochs, ctx):
        n_before = len(epochs)
        epochs.drop_bad(reject={"eeg": float(self.params["threshold_uv"]) * 1e-6})
        n_after = len(epochs)
        ctx.setdefault("rejected_trials", 0)
        ctx["rejected_trials"] += n_before - n_after
        ctx["n_trials_after"] = n_after  # 注意：n_trials_before 由 Epoching 负责累加
        return epochs


class RejectOutlierEpochs(Operator):
    """离群 epoch 剔除：用数据**自身分布**（Q3 + k·IQR）判定离群 trial。

    与「振幅阈值剔除」的区别：不需要人工猜 µV 阈值，自动适应不同被试/不同
    设备的幅值尺度；指标可选峰峰值（默认）、标准差或峰度（对尖峰更敏感）。
    """
    name, title = "reject_outlier_epochs", "离群 epoch 剔除（IQR）"
    stage = "epochs"
    schema = {
        "metric": ("choice", "ptp", "指标", ["ptp", "std", "kurtosis"]),
        "iqr_k": ("float", 3.0, "IQR 倍数（超过 Q3+k·IQR 即剔除）"),
        "per_channel": ("bool", True, "按通道分别判定（任一通道超限即剔除）"),
    }

    def apply(self, epochs, ctx):
        from scipy.stats import kurtosis as _kurt

        X = epochs.get_data() * 1e6                     # µV (N, C, T)
        metric = str(self.params["metric"])
        if metric == "ptp":
            m = X.max(-1) - X.min(-1)
        elif metric == "std":
            m = X.std(-1)
        else:
            m = _kurt(X, axis=-1)
        k = float(self.params["iqr_k"])
        if self.params["per_channel"]:
            q3, q1 = np.percentile(m, 75, axis=0), np.percentile(m, 25, axis=0)
            bad = (m > (q3 + k * (q3 - q1))[None, :]).any(axis=1)
        else:
            mm = m.mean(-1)
            q3, q1 = np.percentile(mm, [75, 25])
            bad = mm > q3 + k * (q3 - q1)

        n_before = len(epochs)
        idx = np.flatnonzero(bad)
        if idx.size:
            epochs.drop(idx.tolist(), reason="outlier", verbose="ERROR")
        ctx.setdefault("rejected_trials", 0)
        ctx["rejected_trials"] += int(n_before - len(epochs))
        ctx["outlier_epochs"] = int(idx.size)
        ctx["n_trials_after"] = len(epochs)
        return epochs


class EuclideanAlignment(Operator):
    """欧几里得对齐（EA）：把 epochs 的平均协方差对齐到单位阵（跨被试场景常用）。

    注意：此处对当前全部 epochs 做无标签对齐（不涉及类别信息，跨被试合法）；
    严格折内 EA 请使用训练配置 normalization.euclidean_alignment。
    """
    name, title = "euclidean_alignment", "欧几里得对齐 (EA)"
    stage = "epochs"
    schema = {"shrink": ("float", 0.1, "协方差收缩正则 (0~1)")}

    def apply(self, epochs, ctx):
        from ..datasets import euclidean_alignment

        X = euclidean_alignment(epochs.get_data(), shrink=float(self.params["shrink"]))
        if not getattr(epochs, "preload", False):
            epochs.load_data()
        epochs._data[:] = X  # get_data() 返回副本，写回内部数组
        return epochs


OPERATORS: dict[str, type[Operator]] = {
    op.name: op for op in [
        SetMontage, PickEEG, DropChannels, MarkBadChannels, InterpolateBads,
        NotchFilter, BandpassFilter, GaussianSmooth, Resample, Normalize,
        ReReference, RejectBadSegments, ClipOutliers, ArtifactRegression, ICA,
        Epoching, AmplitudeReject, RejectOutlierEpochs, EuclideanAlignment,
    ]
}


#: 算子说明（GUI 参数面板顶部显示：作用 / 推荐参数 / 注意事项）
DESCRIPTIONS: dict[str, str] = {
    "set_montage": "设置电极位置（标准导联系统）。后续的地形图、插值坏导都依赖它。"
                   "推荐 standard_1005（64 导联）或 standard_1020，建议放在流水线第一步。",
    "pick_eeg": "只保留 EEG 通道（丢弃 EOG 等）。若后续要用 ICA 自动检测眼电成分，"
                "请保持「同时保留 EOG」为勾选。",
    "drop_channels": "按名字删除通道（坏导、无关通道）。注意：删除后无法插值恢复，"
                     "更推荐用「标记坏导 + 插值坏导」。",
    "mark_bad_channels": "把指定通道标记为坏导（不删数据，供后续插值或排查使用）。",
    "interpolate_bads": "用球面样条插值修复坏导（需先设置导联）。插值后建议清空坏导标记。",
    "notch_filter": "陷波去除工频干扰。中国/欧洲数据用 50 Hz，美国数据（PhysioNet）用 60 Hz；"
                    "可同时滤谐波，如 [50, 100]。",
    "bandpass_filter": "带通/低通/高通滤波。运动想象常用 4-40 Hz 或 8-32 Hz；P300 常用 0.1-30 Hz。"
                       "l_freq 填 0 = 仅低通，h_freq 填 0 = 仅高通。FIR 零相位无失真，IIR 更快。",
    "gaussian_filter": "高斯核平滑（低通），σ 以秒为单位。用于轻度去高频噪声，会轻微模糊波形。",
    "resample": "重采样（自动带抗混叠滤波）。常用 500→250 Hz 或统一到 250/128 Hz，"
                "可加速训练并统一不同数据集的采样率。降采样安全，升采样不增加信息。",
    "normalize": "归一化（多方案）：z-score / [-1,1] / [0,1] / x÷P95 / IQR / A-Law / EMA / 去均值。"
                 "注意此处统计量来自整段数据，仅适合探索与对比实验；"
                 "训练用归一化请在「训练控制台」配置（只用训练折统计量，防数据泄漏）。",
    "re_reference": "重参考：average = 共平均参考（CAR，最常用）；channels = 指定参考电极"
                    "（如乳突 TP9/TP10）；rest = 无穷远参考（球模型近似，跨被试更稳，需电极坐标）；"
                    "bipolar = 按电极对生成双极导联（衍生通道）。projection 勾选后只做投影、不改数据，可逆。",
    "reject_bad_segments": "坏段检测（连续时间段）：滑动窗找大振幅段与平坦段（电极脱落/饱和），"
                           "写成 BAD_segment 注释；分段步骤会自动丢弃与之重叠的 trial。"
                           "比逐 trial 阈值剔除更早发现问题，建议放在滤波之后、ICA 之前。",
    "clip_outliers": "去极值（幅值裁剪）：把零星尖峰夹到稳健范围（中位 ± k·MAD，k 取 6~10）。"
                     "尖峰会污染 z-score 与协方差，裁剪比整段丢弃温和得多。"
                     "建议与「坏段检测」配合：大范围干扰交给坏段，个别尖峰交给裁剪。",
    "artifact_regression": "回归法去伪迹：用 EOG（无 EOG 则用额区通道代理）线性回归掉眨眼/眼动。"
                           "与 ICA 互补：更快、可解释、无随机性；建议只回归伪迹所在频段（0.5-10 Hz）。",
    "ica": "ICA 盲源分离去伪迹（眼电/肌电/心电）。先拟合再剔除成分：可开「自动检测眼电成分」"
           "（需 EOG 通道），或开「自动识别伪迹成分」——无需 EOG，用峰度 / 1s 自相关 / 额区代理相关 / "
           "肌电高频占比 / 幅值比五个判据自动判定，并在日志里给出理由；也可手动填编号"
           "（建议用参数面板上的「浏览 ICA 成分…」逐个看地形图确认）。CPU 上较慢，大文件建议先降采样。",
    "epoching": "事件提取与分段：把连续数据按 cue 切成固定长度小段（epoch）。"
                "tmin/tmax 是相对事件的窗口（如 MI 用 0.5→2.5 s）。这是「样本单位」的来源，"
                "必须放在所有 raw 阶段算子之后。",
    "amplitude_reject": "振幅阈值剔除：epoch 内峰峰值超过阈值即丢弃（简单有效）。"
                        "BNCI 常用 150 µV；PhysioNet 原始数据伪迹多，用 300-500 µV。"
                        "剔除比例过高时建议改用 ICA 而非继续放宽阈值。",
    "reject_outlier_epochs": "离群 epoch 剔除（IQR）：用数据自身分布（Q3+k·IQR）判定离群，"
                             "不用猜 µV 阈值，自动适应各被试幅值尺度。指标可选峰峰值/标准差/峰度，"
                             "与「振幅阈值剔除」二选一或叠加使用（先阈值、后 IQR 温和收尾）。",
    "euclidean_alignment": "欧几里得对齐（EA）：把各被试协方差对齐到单位阵，消除跨被试尺度差异。"
                           "跨被试（LOSO）强烈建议开启。此处对当前数据无标签对齐；"
                           "训练配置里也有等价开关。",
}


def describe_operator(name: str) -> str:
    """返回算子说明文本（GUI 用）。"""
    return DESCRIPTIONS.get(name, "")


def build_operator(name: str, params: dict | None = None) -> Operator:
    if name not in OPERATORS:
        raise KeyError(f"未知算子：{name}（可用：{sorted(OPERATORS)}）")
    return OPERATORS[name](params)
