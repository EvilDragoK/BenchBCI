"""数据划分方案单测：session / kfold / few_shot / loso / holdout。"""
import numpy as np
import pytest

from core.train.splits import make_splits


def _dataset(n_subj=3, n_class=4, per_class=10, sessions=("T", "E")):
    """构造：n_subj 名被试 × 2 session × n_class 类 × per_class 样本。"""
    y, sid, sess = [], [], []
    for s in range(1, n_subj + 1):
        for sn in sessions:
            for c in range(n_class):
                y += [c] * per_class
                sid += [s] * per_class
                sess += [sn] * per_class
    return np.array(y), np.array(sid), np.array(sess)


def test_session_scheme():
    y, sid, sess = _dataset()
    sp = make_splits(y, sid, sess, "session")
    assert len(sp) == 3                      # 每名被试一折
    for s in sp:
        assert s.subject > 0
        assert not np.intersect1d(s.train, s.test).size
        assert set(sess[s.train]) == {"T"} and set(sess[s.test]) == {"E"}


def test_kfold_scheme():
    y, sid, sess = _dataset(n_subj=1)
    sp = make_splits(y, sid, sess, "kfold", k=5)
    assert len(sp) == 5
    for s in sp:
        assert not np.intersect1d(s.train, s.test).size


def test_loso_scheme():
    y, sid, sess = _dataset(n_subj=3)
    sp = make_splits(y, sid, sess, "loso")
    assert len(sp) == 3                       # 每名被试被留出一次
    assert sorted(s.subject for s in sp) == [1, 2, 3]
    for s in sp:
        # 测试集只含留出被试；训练集不含该被试，且覆盖其余被试的全部 session
        assert set(sid[s.test]) == {s.subject}
        assert s.subject not in set(sid[s.train])
        assert set(sid[s.train]) == {1, 2, 3} - {s.subject}
        assert set(sess[s.train]) == {"T", "E"}   # 其他被试全部 session 参与训练
        assert not np.intersect1d(s.train, s.test).size


def test_loso_test_session_filter():
    y, sid, sess = _dataset(n_subj=3)
    sp = make_splits(y, sid, sess, "loso", test_session="T")
    for s in sp:
        assert set(sess[s.test]) == {"T"}


def test_few_shot_scheme():
    y, sid, sess = _dataset(n_subj=2, per_class=10)
    sp = make_splits(y, sid, sess, "few_shot", n_shot=3, n_repeats=2)
    assert len(sp) == 2 * 2                   # 2 名被试 × 2 次重复
    for s in sp:
        # 每类训练样本数 = n_shot（4 类 → 12 个）
        vals, cnts = np.unique(y[s.train], return_counts=True)
        assert len(vals) == 4 and set(cnts.tolist()) == {3}
        assert len(s.train) == 12
        # 测试集为同类剩余样本，且与训练集无重叠、同属一名被试
        assert not np.intersect1d(s.train, s.test).size
        assert len(set(sid[s.train]) | set(sid[s.test])) == 1
        assert len(s.train) + len(s.test) == 80   # 2 session × 4 类 × 10


def test_few_shot_repeats_differ():
    y, sid, sess = _dataset(n_subj=1, per_class=20)
    sp = make_splits(y, sid, sess, "few_shot", n_shot=5, n_repeats=3, seed=1)
    trains = [tuple(sorted(s.train.tolist())) for s in sp]
    assert len(set(trains)) > 1               # 不同重复抽样结果应不同


def test_few_shot_insufficient_samples():
    y, sid, sess = _dataset(n_subj=1, per_class=1, sessions=("T",))  # 每类仅 1 个
    with pytest.raises(ValueError):
        make_splits(y, sid, sess, "few_shot", n_shot=1)


def test_loso_needs_two_subjects():
    y, sid, sess = _dataset(n_subj=1)
    with pytest.raises(ValueError):
        make_splits(y, sid, sess, "loso")


def test_unknown_scheme():
    y, sid, sess = _dataset()
    with pytest.raises(ValueError):
        make_splits(y, sid, sess, "not_a_scheme")
