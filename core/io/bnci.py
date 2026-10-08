"""BNCI 2014-001（BCI Competition IV Dataset 2a）.mat 读取器。

文件结构（scipy.io.loadmat, squeeze_me=True, struct_as_record=False）::

    data : (9,) struct 数组，每个 run 一个元素，字段包括
        X         : (n_times, 25)  25 = 22 EEG + 3 EOG，单位 µV
        trial     : (n_trials,)    cue 起始采样点
        y         : (n_trials,)    类别 1..4（left_hand / right_hand / feet / tongue）
        fs        : 采样率（250 Hz）
        classes   : 类别名列表
        artifacts : (n_trials,) 该 trial 是否含 EOG 伪迹（BNCI 官方标注）

本模块把一个 .mat 文件的所有 run 拼接为单个 `mne.io.Raw`，
事件写入 annotations（description = left_hand / right_hand / feet / tongue），
数据转换为 MNE 要求的 Volt 单位。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

# BCI Competition IV 2a：22 个 EEG 电极（国际 10-20 系统）+ 3 个 EOG。
# 顺序与 MOABB `_load_data_001_2014` 的官方定义逐位一致（**无 Oz、无 CP5/CP6**，
# 含 FCz 与 P2、POz）。顺序必须与 .mat 里的列一一对应，写错会让地形图/脑区分析
# 整体错位（且 set_montage(on_missing="ignore") 不会报错，属静默错误）。
EEG_CH_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]
EOG_CH_NAMES = ["EOG1", "EOG2", "EOG3"]
# y=1..4 对应顺序（与文件内 classes 字段一致）
CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]

# BNCI2014-001 的 `trial` 字段标记「试次起点(fixation cross)」，cue 箭头在其后 2 s。
# 因此把 annotations（cue）放在 trial + 2 s，这样标准 MI 分段（cue 后 0.5~2.5 s）
# 即对应 trial 后 2.5~4.5 s（经验上判别信号最强处）。
CUE_OFFSET_S = 2.0


def _load_mat_runs(path: Path) -> list[Any]:
    from scipy.io import loadmat

    data = loadmat(str(path), squeeze_me=True, struct_as_record=False)["data"]
    return list(np.atleast_1d(data))


def _run_to_raw(run: Any, eeg_only: bool):
    """单个 run struct → (mne.io.RawArray, run 元信息)。"""
    import mne

    X = np.asarray(run.X, dtype=np.float64).T  # (n_ch, n_times)，µV
    X *= 1e-6  # µV → V（MNE 约定）
    fs = float(np.asarray(run.fs).ravel()[0])

    n_ch = X.shape[0]
    if n_ch == len(EEG_CH_NAMES) + len(EOG_CH_NAMES):  # 25
        eeg, eog = EEG_CH_NAMES, EOG_CH_NAMES
    else:  # 通道数不匹配时退化为通用命名
        eeg = [f"EEG{i + 1}" for i in range(n_ch)]
        eog = []
    ch_names = eeg + eog
    ch_types = ["eeg"] * len(eeg) + ["eog"] * len(eog)

    info = mne.create_info(ch_names, fs, ch_types)
    raw = mne.io.RawArray(X, info)

    trial = np.atleast_1d(np.asarray(run.trial)).ravel().astype(int)
    y = np.atleast_1d(np.asarray(run.y)).ravel().astype(int)
    n = min(len(trial), len(y))
    trial, y = trial[:n], y[:n]
    onset = trial / fs + CUE_OFFSET_S  # trial 起点 → cue 时刻
    desc = [CLASS_NAMES[int(v) - 1] if 1 <= int(v) <= len(CLASS_NAMES) else f"class_{v}" for v in y]
    if len(onset):
        raw.set_annotations(mne.Annotations(onset=onset, duration=np.zeros(len(onset)), description=desc))

    art = np.atleast_1d(np.asarray(getattr(run, "artifacts", []))).ravel()
    meta = {
        "n_samples": int(X.shape[1]),
        "duration_s": X.shape[1] / fs,
        "n_trials": int(len(y)),
        "class_counts": {CLASS_NAMES[int(v) - 1]: int((y == v).sum()) for v in np.unique(y) if 1 <= v <= 4},
        "n_artifact_trials": int(np.asarray(art, dtype=bool).sum()) if art.size else 0,
    }
    return raw, meta


def list_subjects(root: str | Path) -> list[int]:
    """扫描目录，返回可用的被试编号（A01T.mat / A01E.mat ...）。"""
    root = Path(root)
    subj = set()
    for f in root.glob("A*.mat"):
        m = re.match(r"A(\d+)[TE]\.mat", f.name, re.IGNORECASE)
        if m:
            subj.add(int(m.group(1)))
    return sorted(subj)


def load_subject_raw(root: str | Path, subject: int, session: str = "T", eeg_only: bool = False):
    """加载一名被试一个 session 的全部 run，拼接为单个 Raw。

    Parameters
    ----------
    root : 数据集根目录（含 A01T.mat 等）。
    subject : 被试编号 1..9。
    session : 'T'（训练）或 'E'（评估）。
    eeg_only : 若 True 只保留 22 个 EEG 通道（丢弃 EOG）。

    Returns
    -------
    (raw, meta) : meta 含被试/采样率/trial 数/类别分布/伪迹标注等信息。
    """
    import mne

    root = Path(root)
    fname = root / f"A{int(subject):02d}{session.upper()}.mat"
    if not fname.exists():
        fname = root / f"A{int(subject):02d}{session.lower()}.mat"
    if not fname.exists():
        raise FileNotFoundError(f"找不到被试 {subject} session {session} 的文件：{fname}")

    runs = _load_mat_runs(fname)
    raws, run_meta = [], []
    for r in runs:
        raw, meta = _run_to_raw(r, eeg_only)
        raws.append(raw)
        run_meta.append(meta)

    raw = mne.concatenate_raws(raws)  # annotations 自动按时间偏移合并
    if eeg_only:
        picks = [ch for ch in raw.ch_names if ch in EEG_CH_NAMES]
        raw.pick(picks)

    n_trials = sum(m["n_trials"] for m in run_meta)
    class_counts: dict[str, int] = {}
    for m in run_meta:
        for k, v in m["class_counts"].items():
            class_counts[k] = class_counts.get(k, 0) + v
    meta = {
        "source": str(fname),
        "subject": int(subject),
        "session": session.upper(),
        "sfreq": float(raw.info["sfreq"]),
        "n_channels": len(raw.ch_names),
        "ch_names": list(raw.ch_names),
        "duration_s": raw.n_times / raw.info["sfreq"],
        "n_runs": len(run_meta),
        "runs": run_meta,
        "n_trials": n_trials,
        "class_counts": class_counts,
        "n_artifact_trials": sum(m["n_artifact_trials"] for m in run_meta),
        "classes": CLASS_NAMES,
        "cue_offset_s": CUE_OFFSET_S,
    }
    return raw, meta
