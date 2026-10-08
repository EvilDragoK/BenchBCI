"""PhysioNet EEG Motor Movement/Imagery Dataset (eegmmidb) 读取器。

数据集结构::

    PhysioNet_MI/
    ├── S001/S001R01.edf  ... S001R14.edf   # 每被试 14 个 run（EDF+，含 T0/T1/T2 标注）
    ├── S002/...                            # 共 109 名被试
    └── RECORDS

- 64 通道 EEG，采样率 160 Hz，**美国数据（工频 60 Hz）**
- 通道名为 WFDB 风格（`Fc5.` / `C3..` / `Fz..`），本模块统一规范化为标准 10-05 命名
- 事件：`T0`=休息开始，`T1`/`T2`=任务提示（含义随 run 变化，见 RUN_TASKS）

各 run 任务（T1 / T2 的语义）——PhysioNet 官方按「同一任务每 3 个 run」分组，
因此是 R03/R07/R11 同一任务，而不是相邻两两配对::

    R01 睁眼基线   R02 闭眼基线（仅 T0，无任务）
    Task1 真实左/右手握拳： R03 / R07 / R11
    Task2 想象左/右手握拳： R04 / R08 / R12
    Task3 真实双手握拳/双脚踏步： R05 / R09 / R13
    Task4 想象双手握拳/双脚踏步： R06 / R10 / R14
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

#: 各 run 的 (模式, T1 标签, T2 标签)；基线 run 无任务
RUN_TASKS: dict[int, tuple[str, str | None, str | None]] = {
    1: ("baseline_eyesopen", None, None),
    2: ("baseline_eyesclosed", None, None),
    3: ("real", "left_fist_real", "right_fist_real"),
    4: ("imagined", "left_fist", "right_fist"),
    5: ("real", "both_fists_real", "both_feet_real"),
    6: ("imagined", "both_fists", "both_feet"),
    7: ("real", "left_fist_real", "right_fist_real"),
    8: ("imagined", "left_fist", "right_fist"),
    9: ("real", "both_fists_real", "both_feet_real"),
    10: ("imagined", "both_fists", "both_feet"),
    11: ("real", "left_fist_real", "right_fist_real"),
    12: ("imagined", "left_fist", "right_fist"),
    13: ("real", "both_fists_real", "both_feet_real"),
    14: ("imagined", "both_fists", "both_feet"),
}

#: 默认运动想象 4 分类（**纯想象**）：左手 / 右手 / 双手 / 双脚
#: = Task2（R04/R08/R12）+ Task4（R06/R10/R14），共 6 个 run
DEFAULT_RUNS = [4, 6, 8, 10, 12, 14]
#: 全部含任务的 run
TASK_RUNS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


def normalize_ch_name(name: str) -> str:
    """WFDB 风格通道名 → 标准 10-05 命名。

    例：``Fc5.`` → ``FC5``，``C3..`` → ``C3``，``Fz..`` → ``Fz``，
    ``Fp1.`` → ``Fp1``，``Poz.`` → ``POz``，``Iz..`` → ``Iz``。
    """
    n = str(name).replace(".", "").replace(" ", "")
    # 前缀用非贪婪匹配，避免把结尾的 z 吞进字母部分（Fz → F + z）
    m = re.match(r"^([A-Za-z]+?)(\d*)(z?)$", n)
    if not m:
        return n
    letters, digits, z = m.groups()
    letters = letters.upper()
    if letters == "FP":  # Fp1 / Fpz
        letters = "Fp"
    return f"{letters}{digits}{z.lower()}"


def list_subjects(root: str | Path) -> list[int]:
    """扫描数据集目录，返回可用被试编号（1..109）。"""
    root = Path(root)
    subs = []
    for p in sorted(root.glob("S*")):
        m = re.fullmatch(r"[Ss](\d+)", p.name)
        if m and p.is_dir():
            subs.append(int(m.group(1)))
    return sorted(subs)


def list_runs(root: str | Path, subject: int) -> list[int]:
    """返回某被试可用的 run 编号。"""
    d = Path(root) / f"S{int(subject):03d}"
    runs = []
    for f in sorted(d.glob(f"S{int(subject):03d}R*.edf")):
        m = re.search(r"R(\d+)\.edf$", f.name)
        if m:
            runs.append(int(m.group(1)))
    return sorted(runs)


def load_subject_raw(root: str | Path, subject: int, runs=None, eeg_only: bool = True,
                     keep_rest: bool = False):
    """加载一名被试的若干 run 并合并为单个 Raw（事件已映射为语义标签）。

    Parameters
    ----------
    root : 数据集根目录（含 S001/ ...）。
    subject : 被试编号 1..109。
    runs : run 编号列表（如 [5, 6, 9, 10]），默认 DEFAULT_RUNS。
    eeg_only : 保留以兼容统一接口（本数据集 64 通道全为 EEG）。
    keep_rest : 是否保留 T0（休息）标注为 'rest'；默认 False（只保留任务事件）。

    Returns
    -------
    (raw, meta)
    """
    import mne

    root = Path(root)
    runs = list(runs) if runs else list(DEFAULT_RUNS)
    mode = "physionet_mi"

    raws, run_meta = [], []
    for r in runs:
        fname = root / f"S{int(subject):03d}" / f"S{int(subject):03d}R{int(r):02d}.edf"
        if not fname.exists():
            continue
        raw = mne.io.read_raw_edf(str(fname), preload=True, verbose="ERROR")
        # 通道名规范化
        raw.rename_channels({ch: normalize_ch_name(ch) for ch in raw.ch_names})
        raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

        _, t1_label, t2_label = RUN_TASKS.get(int(r), ("unknown", None, None))
        onsets, descs = [], []
        for d, on in zip(raw.annotations.description, raw.annotations.onset):
            d = str(d).strip()
            if d == "T1" and t1_label:
                onsets.append(float(on))
                descs.append(t1_label)
            elif d == "T2" and t2_label:
                onsets.append(float(on))
                descs.append(t2_label)
            elif d == "T0" and keep_rest:
                onsets.append(float(on))
                descs.append("rest")
        raw.set_annotations(mne.Annotations(onsets, np.zeros(len(onsets)), descs))

        run_meta.append({
            "run": int(r), "task": RUN_TASKS.get(int(r), ("unknown",))[0],
            "duration_s": raw.n_times / raw.info["sfreq"], "n_events": len(onsets),
        })
        raws.append(raw)

    if not raws:
        raise FileNotFoundError(
            f"未找到被试 S{int(subject):03d} 的 run {runs}（目录：{root}）")

    raw = mne.concatenate_raws(raws)

    n_trials = sum(m["n_events"] for m in run_meta)
    class_counts: dict[str, int] = {}
    for d in raw.annotations.description:
        d = str(d)
        if "boundary" in d.lower():  # 跳过 run 拼接产生的边界标注
            continue
        class_counts[d] = class_counts.get(d, 0) + 1

    meta = {
        "source": str(root / f"S{int(subject):03d}"),
        "dataset": mode,
        "subject": int(subject),
        "session": "R" + "+".join(str(r) for r in runs),
        "runs": runs,
        "run_info": run_meta,
        "sfreq": float(raw.info["sfreq"]),
        "n_channels": len(raw.ch_names),
        "ch_names": list(raw.ch_names),
        "duration_s": raw.n_times / raw.info["sfreq"],
        "n_trials": n_trials,
        "class_counts": class_counts,
        "classes": sorted(class_counts),
        "line_freq": 60.0,  # 美国数据：工频 60 Hz
    }
    return raw, meta
