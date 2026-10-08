"""伪迹处理：ICA 成分自动判定 + 回归法去伪迹。

设计要点
--------
1. **不引入新依赖**：只用 numpy / scipy / mne（项目现有依赖）。
2. **无需 EOG 也能判定眼动成分**：没有 EOG 通道时，用**额区通道**（Fp/AF/F 开头）
   作为眼电代理——眨眼在前额最强，这一招在真实数据里非常有效。
3. 判定给出**理由**（kurtosis / 自相关 / 额区相关 / 肌电比 / 幅值比），
   UI 里可以逐条展示，用户勾选有据可依，而不是"盲剔编号"。

判据说明（每个 ICA 成分）：见 ``component_features`` 的返回字段。
"""
from __future__ import annotations

import re

import numpy as np

EPS = 1e-12

#: 额区（眼电代理）通道名前缀，覆盖 10-20 / 10-05 命名（Fp1、AF7、F3、Fz…）
FRONTAL_PREFIXES = ("fp", "af", "f")

#: 默认判定阈值
DEFAULT_THRESHOLDS = {
    "kurt": 4.0,          # 峰度（尖峰/眨眼）
    "autocorr": 0.90,     # 1 s 自相关（缓慢漂移/眼动）
    "frontal_corr": 0.80,  # 与额区/眼电代理的最大相关（实测眼动成分约 0.8~0.95）
    "emg": 2.0,           # 20-45 Hz 功率占比（肌电）
    "amp_ratio": 6.0,     # 峰峰值 / 鲁棒标准差（超大幅值）
}


# ================================================================ 通道归类
def eog_channels(raw) -> list[str]:
    """已标记为 EOG 类型的通道名。"""
    return [ch for ch in raw.ch_names
            if raw.get_channel_types([ch])[0] in ("eog", "ecg", "emg")]


#: 标准电极名：F/Fp/AF + 数字或 z（F3、Fz、Fp1、AF7），避免误收 Fs/Filter 等
_FRONTAL_RE = re.compile(r"^(fp|af|f)(z|\d+)[a-z0-9]*$")
#: 最靠前的两簇：Fp（前额极）与 AF（前额），眨眼/眼动在此最强
_FP_AF_RE = re.compile(r"^(fp|af)(z|\d+)[a-z0-9]*$")
_F_RE = re.compile(r"^f(z|\d+)[a-z0-9]*$")


def frontal_channels(raw, extra: tuple[str, ...] = ()) -> list[str]:
    """按名字匹配额区通道（F / Fp / AF 开头）。"""
    out = [ch for ch in raw.ch_names if _FRONTAL_RE.match(ch.strip().lower())]
    if extra:                                  # 允许用户补充自定义前缀
        low_extra = tuple(p.lower() for p in extra)
        out += [ch for ch in raw.ch_names
                if ch.lower().startswith(low_extra) and ch not in out]
    return out


def front_most_channels(raw) -> list[str]:
    """最靠前的电极：优先 Fp/AF（眨眼最强处），没有才退到 F。"""
    names = [ch.strip().lower() for ch in raw.ch_names]
    fp = [ch for ch, low in zip(raw.ch_names, names) if _FP_AF_RE.match(low)]
    if fp:
        return fp
    return [ch for ch, low in zip(raw.ch_names, names) if _F_RE.match(low)]


def proxy_channels(raw, max_n: int = 4) -> list[str]:
    """伪迹检测/回归用的代理通道：优先 EOG，其次最靠前的额区电极。

    限制最多 ``max_n`` 个：代理太多会把真实额区脑电一起回归掉。
    """
    eog = eog_channels(raw)
    cand = eog if eog else front_most_channels(raw)
    return cand[:max_n] if max_n else cand


# ================================================================ 成分诊断
def component_features(raw, ica, proxies: list[str] | None = None) -> dict:
    """计算每个 ICA 成分的诊断指标。

    Returns
    -------
    dict，均为 (n_components,) 数组：

    - ``kurtosis``：超额峰度（正常≈0；眨眼/尖峰成分显著 >0）
    - ``autocorr``：1 s 滞后自相关（眼动/漂移成分偏高）
    - ``frontal_corr``：与代理通道（EOG 或额区）的最大 |相关|
    - ``emg_ratio``：20-45 Hz 功率 / 1-45 Hz 功率（肌电成分偏高）
    - ``amp_ratio``：峰峰值 / 1.4826·MAD（超大幅值/尖峰）
    - ``var_frac``：该成分解释的方差占比
    - ``proxy_source``：'eog' / 'frontal' / 'none'（相关判据的来源）
    """
    from scipy.signal import welch
    from scipy.stats import kurtosis

    src = ica.get_sources(raw).get_data()          # (n_comp, T)
    fs = float(raw.info["sfreq"])
    n_comp = src.shape[0]
    out: dict[str, object] = {"n_components": n_comp}

    out["kurtosis"] = np.asarray(kurtosis(src, axis=-1, fisher=True), dtype=float)

    # --- 1 s 自相关（缓慢成分 → 眼动/漂移/电极漂移）---
    lag = max(int(round(fs)), 1)
    if src.shape[-1] > lag + 8:
        a = src[:, :-lag] - src[:, :-lag].mean(-1, keepdims=True)
        b = src[:, lag:] - src[:, lag:].mean(-1, keepdims=True)
        den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1)) + EPS
        out["autocorr"] = (a * b).sum(-1) / den
    else:  # 数据太短，退化为 1 采样点滞后
        a = src[:, :-1] - src[:, :-1].mean(-1, keepdims=True)
        b = src[:, 1:] - src[:, 1:].mean(-1, keepdims=True)
        out["autocorr"] = (a * b).sum(-1) / (np.sqrt((a * a).sum(-1) * (b * b).sum(-1)) + EPS)

    # --- 与代理通道的相关（眨眼/眼动最直接的特征）---
    proxies = proxies if proxies is not None else proxy_channels(raw)
    proxies = [p for p in proxies if p in raw.ch_names]
    if proxies:
        P = raw.copy().pick(proxies).get_data()     # (k, T)
        Sc = src - src.mean(-1, keepdims=True)
        Pc = P - P.mean(-1, keepdims=True)
        num = Sc @ Pc.T                             # (n_comp, k)
        den = (np.linalg.norm(Sc, axis=-1)[:, None] * np.linalg.norm(Pc, axis=-1)[None, :]) + EPS
        out["frontal_corr"] = np.abs(num / den).max(axis=-1)
        out["proxy_source"] = "eog" if eog_channels(raw) else "frontal"
    else:
        out["frontal_corr"] = np.zeros(n_comp)
        out["proxy_source"] = "none"

    # --- 肌电：高频功率占比 ---
    nperseg = int(min(256, src.shape[-1]))
    f, p = welch(src, fs=fs, nperseg=max(nperseg, 8), axis=-1)
    hi = (f >= 20.0) & (f <= min(45.0, fs / 2 * 0.95))
    lo = (f >= 1.0) & (f <= min(45.0, fs / 2 * 0.95))
    out["emg_ratio"] = p[:, hi].sum(-1) / (p[:, lo].sum(-1) + EPS)

    # --- 幅值比（尖峰/运动伪迹）---
    ptp = src.max(-1) - src.min(-1)
    med = np.median(src, axis=-1)
    mad = np.median(np.abs(src - med[:, None]), axis=-1) * 1.4826
    out["amp_ratio"] = ptp / (mad + EPS)

    # --- 方差贡献：ICA 成分是白化的（各成分方差相等），必须用混合矩阵的列范数 ---
    try:
        mix = np.asarray(ica.mixing_matrix_, dtype=float)      # (n_ch, n_comp)
        contrib = (mix ** 2).sum(axis=0)
        out["var_frac"] = contrib / (contrib.sum() + EPS)
    except Exception:  # noqa: BLE001 - 极端情况下退化为均分
        out["var_frac"] = np.full(n_comp, 1.0 / max(n_comp, 1))
    return out


def _relative_outlier(v) -> np.ndarray:
    """相对离群掩码：≥ 中位 + 2·MAD（MAD=0 时全 False）。

    用途：像峰度、幅值比这类**没有通用绝对阈值**的指标，必须同时是"成分里的离群者"
    才判定，否则一旦整段数据都脏（例如满屏运动伪迹），会把全部成分一次标光。
    """
    v = np.asarray(v, dtype=float)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    if mad <= 0:
        return np.zeros(v.shape, dtype=bool)
    return v >= med + 2.0 * mad


def suggest_exclude(feats: dict, thresholds: dict | None = None,
                    min_hits: int = 2, max_frac: float = 0.4
                    ) -> tuple[list[int], dict[int, list[str]]]:
    """按诊断指标给出建议剔除的成分与理由（带两道安全阀）。

    判据：
    - ``kurtosis`` / ``amp_ratio``：需**同时**超过绝对阈值 + 成分间相对离群
    - ``autocorr`` / ``frontal_corr`` / ``emg_ratio``：绝对阈值（物理含义明确）

    安全阀：
    1. ``min_hits``：至少命中 N 条判据才算（默认 2，避免"单指标误伤"）
    2. ``max_frac``：建议剔除数不超过成分总数的该比例（超出则按命中数取前几名），
       防止把整份数据判成伪迹

    Returns
    -------
    (indices, reasons)：``reasons[comp] = ["额区相关 0.95", "峰度 6.2", ...]``
    """
    th = dict(DEFAULT_THRESHOLDS)
    th.update(thresholds or {})
    n = int(feats["n_components"])
    kurt_rel = _relative_outlier(feats["kurtosis"])
    amp_rel = _relative_outlier(feats["amp_ratio"])

    reasons: dict[int, list[str]] = {}
    for i in range(n):
        hits = []
        if feats["kurtosis"][i] >= th["kurt"] and kurt_rel[i]:
            hits.append(f"峰度 {feats['kurtosis'][i]:.1f}")
        if feats["autocorr"][i] >= th["autocorr"]:
            hits.append(f"1s自相关 {feats['autocorr'][i]:.2f}")
        if feats["frontal_corr"][i] >= th["frontal_corr"]:
            tag = "眼电相关" if feats.get("proxy_source") == "eog" else "额区相关"
            hits.append(f"{tag} {feats['frontal_corr'][i]:.2f}")
        if feats["emg_ratio"][i] >= th["emg"]:
            hits.append(f"肌电比 {feats['emg_ratio'][i]:.1f}")
        if feats["amp_ratio"][i] >= th["amp_ratio"] and amp_rel[i]:
            hits.append(f"幅值比 {feats['amp_ratio'][i]:.1f}")
        if len(hits) >= max(int(min_hits), 1):
            reasons[i] = hits

    limit = max(int(n * float(max_frac)), 1)
    if len(reasons) > limit:
        keep = sorted(reasons, key=lambda k: (-len(reasons[k]), k))[:limit]
        dropped = len(reasons) - len(keep)
        reasons = {k: reasons[k] + [f"（另 {dropped} 个候选因超上限被忽略）"] if k == keep[0]
                   else reasons[k] for k in keep}
    return sorted(reasons), reasons


# ================================================================ 回归法去伪迹
def regress_artifacts(raw, proxies=None, lower: float | None = None,
                      upper: float | None = None, ctx: dict | None = None):
    """回归法去伪迹：从每个 EEG 通道中减去由代理通道线性预测的部分。

    与 ICA 互补：ICA 是盲源分离（适合眨眼/心电），回归法适合有明确参考通道的
    伪迹（EOG 眼动、EMG 颌肌、ECG 心电）。

    Parameters
    ----------
    raw : mne.io.BaseRaw
    proxies : 代理通道名列表；默认 EOG（无则额区）
    lower, upper : 可选带通（Hz），只回归该频段的伪迹（如眨眼用 0.5-10 Hz）
    ctx : 流水线上下文，写入 ``artifact_regression``

    Notes
    -----
    代理通道自身不做校正（它们是伪迹参考，不是干净的 EEG）。
    """
    import mne

    if not getattr(raw, "preload", False):
        raw.load_data()
    proxies = [p for p in (proxies or proxy_channels(raw)) if p in raw.ch_names]
    if not proxies:
        raise ValueError("未找到代理通道：请指定通道名，或确保存在 EOG/额区通道")
    if ctx is not None and not eog_channels(raw) and not any(
            p.lower().startswith(("fp", "af")) for p in proxies):
        # 只有普通额区电极（如 BNCI 的 Fz）时，代理信号里混有真实额区脑电，
        # 回归会连同它一起去掉——明确提示，交由用户判断是否启用/换代理
        ctx.setdefault("warnings", []).append(
            f"回归去伪迹用的代理 {proxies} 都是普通额区电极（无 Fp/AF、无 EOG），"
            f"可能连真实额区脑电一起去除；建议指定 EOG 通道或改用 ICA")

    data = raw.get_data()                       # (C, T)
    P = raw.copy().pick(proxies).get_data()     # (k, T)
    if lower or upper:
        from scipy.signal import butter, filtfilt

        nyq = float(raw.info["sfreq"]) / 2.0
        lo = (max(float(lower), 0.1) / nyq) if lower else None
        hi = (min(float(upper), nyq * 0.95) / nyq) if upper else None
        btype = "band" if (lo and hi) else ("highpass" if lo else "lowpass")
        b, a = butter(4, [lo, hi] if btype == "band" else (lo or hi), btype=btype)
        P = filtfilt(b, a, P, axis=-1)

    design = np.vstack([P, np.ones((1, P.shape[-1]))])          # 带截距
    G = design @ design.T
    beta = data @ design.T @ np.linalg.pinv(G)                  # (C, k+1)
    clean = data - beta @ design

    # 代理通道原样保留（不校正）
    idx = [raw.ch_names.index(p) for p in proxies]
    clean[idx] = data[idx]

    r2 = 1.0 - (clean.var(-1) / (data.var(-1) + EPS))
    info = raw.info.copy()
    new = mne.io.RawArray(clean, info)
    new.set_annotations(raw.annotations)
    if ctx is not None:
        ctx["artifact_regression"] = {
            "proxies": proxies,
            "band": [lower, upper],
            "removed_var_mean": float(np.mean(r2)),
        }
    return new
