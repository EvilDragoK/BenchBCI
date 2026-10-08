"""BNCI2014001 数据集读取单测（数据缺失时自动跳过）。"""
import pytest

from core.data_paths import bnci_root      # 默认本机路径 + 环境变量 BCI_DATA_ROOT 覆盖
from core.io.bnci import load_subject_raw

DATA_ROOT = bnci_root()
pytestmark = pytest.mark.skipif(not (DATA_ROOT / "A01T.mat").exists(),
                                reason="BNCI2014001 数据集不可用")


def test_load_subject():
    raw, meta = load_subject_raw(DATA_ROOT, 1, "T", eeg_only=True)
    assert len(raw.ch_names) == 22
    assert meta["sfreq"] == 250.0
    assert meta["n_trials"] == 288
    assert set(meta["class_counts"]) == {"left_hand", "right_hand", "feet", "tongue"}
    # 事件标注可被 epoching 使用
    import mne

    events, ev = mne.events_from_annotations(raw)
    assert len(events) == 288


def test_list_subjects():
    from core.io.bnci import list_subjects

    assert list_subjects(DATA_ROOT) == list(range(1, 10))
