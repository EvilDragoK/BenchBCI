"""通用数据读取：EDF/BDF/GDF/BrainVision/FIF/CNT/EEGLAB + 预处理输出(.npy)。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SUPPORTED_FORMATS = [".edf", ".bdf", ".gdf", ".vhdr", ".fif", ".cnt", ".set", ".mat", ".npy"]


def read_raw_any(path: str | Path, preload: bool = True):
    """按扩展名自动分发到 mne.io.read_raw_*。

    .mat/.npy 需为 (n_ch, n_times) 的数组文件（会假定全部为 EEG 通道）。
    """
    import mne

    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".edf":
        return mne.io.read_raw_edf(path, preload=preload)
    if ext == ".bdf":
        return mne.io.read_raw_bdf(path, preload=preload)
    if ext == ".gdf":
        return mne.io.read_raw_gdf(path, preload=preload)
    if ext == ".vhdr":
        return mne.io.read_raw_brainvision(path, preload=preload)
    if ext == ".cnt":
        return mne.io.read_raw_cnt(path, preload=preload)
    if ext == ".set":
        return mne.io.read_raw_eeglab(path, preload=preload)
    if ext == ".fif" or ext == ".fif.gz":
        return mne.io.read_raw_fif(path, preload=preload, allow_maxshield=True)
    if ext in (".mat", ".npy"):
        data = _load_array(path)
        if data.ndim != 2:
            raise ValueError(f"{path} 应为 (n_ch, n_times) 二维数组，当前 shape={data.shape}")
        sfreq = 250.0
        info = mne.create_info([f"EEG{i + 1}" for i in range(data.shape[0])], sfreq, "eeg")
        return mne.io.RawArray(data.astype(np.float64), info)
    raise ValueError(f"不支持的格式：{ext}（支持 {SUPPORTED_FORMATS}）")


def _load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".mat":
        from scipy.io import loadmat

        d = loadmat(str(path))
        for k, v in d.items():
            if not k.startswith("__") and isinstance(v, np.ndarray):
                return np.asarray(v)
        raise ValueError(f"{path} 中未找到数值数组")
    return np.load(path)


def read_epochs_fif(path: str | Path):
    """读取预处理保存的 Epochs FIF。"""
    import mne

    return mne.read_epochs(str(path), preload=True)


def load_processed(processed_dir: str | Path) -> dict:
    """读取 `core.pipeline` 生成的预处理输出。

    Returns
    -------
    dict(X, y, sid, sess, classes, channel_names, sfreq, meta)
        X: float32 [N, C, T]（Volt）；y: int64 类别码；
        sid / sess: 每条 epoch 的被试编号与 session 标记。
    """
    d = Path(processed_dir)
    X = np.load(d / "X.npy")
    y = np.load(d / "y.npy").astype(np.int64)
    sid = np.load(d / "sid.npy").astype(np.int64)
    sess = np.load(d / "sess.npy")
    # 记录块编号（新流水线才会输出；旧数据没有 → None，block 划分会给出提示）
    block = np.load(d / "block.npy").astype(np.int64) if (d / "block.npy").exists() else None
    with open(d / "classes.json", encoding="utf-8") as f:
        classes = json.load(f)  # {class_name: code}
    meta = {}
    if (d / "meta.json").exists():
        with open(d / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
    return {
        "X": X, "y": y, "sid": sid, "sess": sess, "block": block,
        "classes": classes, "meta": meta,
        "channel_names": meta.get("channel_names", []),
        "sfreq": meta.get("sfreq", 250.0),
    }


def class_names_from_processed(classes: dict) -> list[str]:
    """{'feet':1,...} → 按码排序的类名列表。"""
    inv = {int(v): k for k, v in classes.items()}
    return [inv[i] for i in sorted(inv)]
