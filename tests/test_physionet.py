"""PhysioNet MI 数据集读取单测（数据缺失时自动跳过）。"""
import pytest

from core.data_paths import physionet_root   # 默认本机路径 + 环境变量覆盖
from core.io.physionet import normalize_ch_name

DATA_ROOT = physionet_root()
pytestmark = pytest.mark.skipif(not (DATA_ROOT / "S001").exists(),
                                reason="PhysioNet_MI 数据集不可用")


def test_normalize_ch_name():
    assert normalize_ch_name("Fc5.") == "FC5"
    assert normalize_ch_name("C3..") == "C3"
    assert normalize_ch_name("Fz..") == "Fz"
    assert normalize_ch_name("Fp1.") == "Fp1"
    assert normalize_ch_name("Fpz.") == "Fpz"
    assert normalize_ch_name("Poz.") == "POz"
    assert normalize_ch_name("Iz..") == "Iz"
    assert normalize_ch_name("Fcz.") == "FCz"
    assert normalize_ch_name("Af7.") == "AF7"
    assert normalize_ch_name("T9..") == "T9"


def test_list_subjects_and_runs():
    from core.io.physionet import list_runs, list_subjects

    subs = list_subjects(DATA_ROOT)
    assert len(subs) >= 1 and subs[0] == 1
    runs = list_runs(DATA_ROOT, 1)
    assert runs == list(range(1, 15))


def test_load_subject_raw():
    from core.io.physionet import DEFAULT_RUNS, load_subject_raw

    raw, meta = load_subject_raw(DATA_ROOT, 1, runs=DEFAULT_RUNS)
    assert len(raw.ch_names) == 64
    assert raw.info["sfreq"] == 160.0
    assert meta["line_freq"] == 60.0
    assert meta["runs"] == DEFAULT_RUNS == [4, 6, 8, 10, 12, 14]   # 纯想象 6 个 run
    # 四类**想象**任务标签（R04/R08/R12 = 想象左/右手，R06/R10/R14 = 想象双手/双脚）
    assert set(meta["class_counts"]) == {"left_fist", "right_fist", "both_fists", "both_feet"}
    # 通道名已规范化为标准 10-05（可用于 montage 匹配 / 地形图）
    assert "C3" in raw.ch_names and "FC5" in raw.ch_names and "POz" in raw.ch_names

    import mne

    events, ev = mne.events_from_annotations(raw)
    assert len(events) == meta["n_trials"]
    assert len(ev) == 4


def test_run_task_mapping_is_official():
    """回归：PhysioNet 每个任务占每 4 个 run 中的同一个（R03/R07/R11 同任务），
    而不是相邻两两一组——旧实现把 R05/R09（真实动作）当成想象任务，标签全错。"""
    from core.io.physionet import DEFAULT_RUNS, RUN_TASKS, TASK_RUNS

    # Task1 真实左/右手：R03/R07/R11
    for r in (3, 7, 11):
        assert RUN_TASKS[r] == ("real", "left_fist_real", "right_fist_real")
    # Task2 想象左/右手：R04/R08/R12
    for r in (4, 8, 12):
        assert RUN_TASKS[r] == ("imagined", "left_fist", "right_fist")
    # Task3 真实双手/双脚：R05/R09/R13
    for r in (5, 9, 13):
        assert RUN_TASKS[r] == ("real", "both_fists_real", "both_feet_real")
    # Task4 想象双手/双脚：R06/R10/R14
    for r in (6, 10, 14):
        assert RUN_TASKS[r] == ("imagined", "both_fists", "both_feet")
    assert all(RUN_TASKS[r][0] == "imagined" for r in DEFAULT_RUNS)
    assert sorted(DEFAULT_RUNS) == DEFAULT_RUNS
    assert set(DEFAULT_RUNS) <= set(TASK_RUNS)


def test_baseline_run_no_task_events():
    from core.io.physionet import load_subject_raw

    raw, meta = load_subject_raw(DATA_ROOT, 1, runs=[1], keep_rest=False)
    assert meta["n_trials"] == 0  # 基线 run 只有 T0，被丢弃
    raw2, meta2 = load_subject_raw(DATA_ROOT, 1, runs=[1], keep_rest=True)
    assert meta2["n_trials"] > 0 and "rest" in meta2["class_counts"]
