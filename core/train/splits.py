"""数据划分方案。

支持 7 种协议（按 MOABB 生态的常用分类组织）：

**被试内 within-subject**
- ``within``        : 被试内划分（对齐 EEG-FM-Benchmark）——**每名被试一折**
                      （共 N 折）；训练集按 `train_percentage`（每被试每类比例，
                      该基准默认 0.3）抽取，未给该参数时按 1/k 分层随机切。
                      `n_repeats > 1` 时折数 = 被试数 × n_repeats（重复采样取平均）
- ``kfold``         : 逐被试分层 k 折（每人切 k 折轮流当测试，跨被试聚合）；
                      不同被试的数据绝不混切
- ``few_shot``      : 被试内少样本校准——每类用 n_shot 个 trial 训练（或用
                      `train_percentage` 按比例），其余测试

**跨时间 cross-session**
- ``session``       : 逐被试 T(训练) → E(测试)，同被试跨时间稳定性
                      （标准 BCI 基准）

**跨被试 cross-subject**
- ``loso``          : Leave-One-Subject-Out——每次留出一名被试做测试
- ``subject_kfold`` : 跨被试分组 k 折——每次留出一整组被试做测试

**快速调试**
- ``holdout``       : 全体数据单次随机划分（测试集占 1/k，k=5 → 80%/20%）

每个 Split 带 ``subject`` 字段，便于按被试聚合（LOSO / few_shot / session 场景）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split


@dataclass
class Split:
    fold: int
    tag: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    subject: int = -1   # 该折对应的被试编号（-1=不区分，用于按被试聚合）
    repeat: int = 0     # few_shot 的重复抽样序号


def make_splits(y: np.ndarray, sid: np.ndarray, sess: np.ndarray, scheme: str,
                k: int = 5, val_frac: float = 0.15, seed: int = 42,
                session_train: str = "T", session_test: str = "E",
                test_session: str = "all", n_shot: int = 5,
                n_repeats: int = 1, custom_folds: list[dict] | None = None,
                block: np.ndarray | None = None,
                domain_sizes: tuple[int, ...] | None = None,
                train_percentage: float | None = None) -> list[Split]:
    """按协议生成划分。

    Parameters
    ----------
    scheme : session | within | kfold | few_shot | loso | subject_kfold |
             holdout | custom | block | session_kfold
    test_session : loso / few_shot 用——测试数据取哪个 session（all 表示全部 session）
    n_shot : 仅 few_shot 用——每类训练样本数
    n_repeats : 仅 few_shot / within 用——随机重复抽样次数
    custom_folds : 仅 custom 用——[{name, train:[受试者号], val:[...], test:[...]}, ...]
                   val 缺省/为空时，自动从该折 train 中按 val_frac 划出
    block : 仅 block 用——每个 trial 所属的记录块编号（预处理时写入）
    """
    y = np.asarray(y)
    sid = np.asarray(sid)
    sess = np.asarray(sess)
    splits: list[Split] = []

    if scheme == "session":
        # 标准 BCI 评测协议：逐被试 用 T(训练) session 训练、E(评估) session 测试；
        # 多被试时每名被试一个折，聚合结果即“均值±标准差”。
        subjects = sorted(set(sid.tolist())) or [None]
        for i, s in enumerate(subjects):
            if s is None:
                tr = np.where(sess == session_train)[0]
                te = np.where(sess == session_test)[0]
                tag = f"train:{session_train}->test:{session_test}"
            else:
                tr = np.where((sess == session_train) & (sid == s))[0]
                te = np.where((sess == session_test) & (sid == s))[0]
                tag = f"S{int(s):02d}:{session_train}->{session_test}"
            if len(tr) == 0 or len(te) == 0:
                continue
            # 注意用 _carve_val 返回的 tr2：验证集必须从训练集里**真正切出去**
            # （否则 val ⊂ train，早停/选权重就建立在训练过的样本上）
            va, tr = _carve_val(tr, y, val_frac, seed + i)
            splits.append(Split(i, tag, tr, va, te,
                                subject=int(s) if s is not None else -1))
        if not splits:
            raise ValueError(
                f"session 划分需要同时含 {session_train}(训练) 与 {session_test}(测试) 数据，"
                f"请在预处理配置 sessions 中包含两者。")

    elif scheme in ("kfold", "nested_cv"):
        # 逐被试分层 k 折（衡量"被试内性能上限"）：每名被试内部独立切 k 折，
        # 再跨被试聚合。**不能**把所有被试混在一起切——同一被试相邻 trial 高度
        # 相关，混切会让它同时出现在训练/测试集，结果被显著高估（数据泄漏）。
        # nested_cv 的外层折与此相同，差别在于 run_training 会在每个训练折内部
        # 再做 inner_k 折来选择超参（消除"用测试集调参"的乐观偏差）。
        label = "nested" if scheme == "nested_cv" else "kfold"
        subjects = sorted(set(sid.tolist())) or [None]
        fold = 0
        for s in subjects:
            idx_s = np.arange(len(y)) if s is None else np.where(sid == s)[0]
            if len(idx_s) < k or len(np.unique(y[idx_s])) < 2:
                continue
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            for i, (a, b) in enumerate(skf.split(np.zeros(len(idx_s)), y[idx_s])):
                tr, te = idx_s[a], idx_s[b]
                va, tr2 = _carve_val(tr, y, val_frac, seed + fold)
                prefix = f"S{int(s):02d}-" if s is not None else ""
                splits.append(Split(fold, f"{prefix}{label}-{i + 1}/{k}", tr2, va, te,
                                    subject=int(s) if s is not None else -1))
                fold += 1
        if not splits:
            raise ValueError(
                f"{scheme} 划分失败：每名被试至少需要 {k} 个样本（含至少 2 类）")

    elif scheme == "few_shot":
        splits = _few_shot_splits(y, sid, sess, n_shot, n_repeats, val_frac,
                                  seed, test_session, train_percentage)

    elif scheme == "loso":
        splits = _loso_splits(y, sid, sess, val_frac, seed, test_session)

    elif scheme == "within":
        # 被试内划分（within-subject）—— 对齐 EEG-FM-Benchmark 的做法：
        #   · 每名被试**一折**（共 N 折），而不是"被试数 × 重复次数"
        #   · 训练集按「每被试每类比例」抽取（train_percentage，该基准默认 0.3）；
        #     未给该参数时退化为按 1/k 分层随机切分
        #   · 需要重复随机采样取平均时，显式设 n_repeats > 1
        #     （折数 = 被试数 × n_repeats；该基准是用多个 seed 达到同样效果）
        subjects = sorted(set(sid.tolist())) or [None]
        n_rep = max(int(n_repeats), 1)
        fold = 0
        for s in subjects:
            idx_s = np.arange(len(y)) if s is None else np.where(sid == s)[0]
            if len(idx_s) < 2 or len(np.unique(y[idx_s])) < 2:
                continue
            for r in range(n_rep):
                tr_all, te = _per_subject_split(idx_s, y[idx_s], k, train_percentage,
                                                seed + fold)
                if len(tr_all) == 0 or len(te) == 0:
                    continue
                va, tr = _carve_val(tr_all, y, val_frac, seed + fold)
                prefix = f"S{int(s):02d}-" if s is not None else ""
                splits.append(Split(fold, f"{prefix}within-{r + 1}/{n_rep}", tr, va, te,
                                    subject=int(s) if s is not None else -1, repeat=r))
                fold += 1
        if not splits:
            raise ValueError("within 划分失败：每名被试至少需要 2 个样本（含至少 2 类）")

    elif scheme == "subject_kfold":
        # 跨被试分组 k 折（subject-independent k-fold）：把被试分成 k 组，
        # 每次留出一整组被试做测试——训练集里绝不出现测试被试的任何样本。
        # 相当于 LOSO 的 k 折版本，论文中常写作 "5-fold cross-subject"。
        subjects = sorted(set(sid.tolist()))
        if len(subjects) < k:
            raise ValueError(
                f"subject_kfold 需要至少 {k} 名被试（当前 {len(subjects)} 名）；"
                f"被试较少时请用 loso")
        for i, (tr, te) in enumerate(GroupKFold(n_splits=k).split(np.zeros(len(y)),
                                                                 y, groups=sid)):
            va, tr2 = _carve_val(tr, y, val_frac, seed + i)
            held = ",".join(f"S{v:02d}" for v in sorted(set(sid[te].tolist())))
            splits.append(Split(i, f"subj-kfold-{i + 1}/{k}[{held}]", tr2, va, te))
        if not splits:
            raise ValueError("subject_kfold 划分失败")

    elif scheme == "custom":
        splits = _custom_splits(y, sid, custom_folds or [], val_frac, seed)

    elif scheme == "block":
        splits = _block_splits(y, sid, block, val_frac, seed, k)

    elif scheme == "session_kfold":
        splits = _session_kfold_splits(y, sid, sess, val_frac, seed)

    elif scheme == "cross_dataset":
        # 跨数据集迁移：训练域与测试域是两份**不同**的预处理数据（如 2a→2b、
        # PhysioNet→2a）。run_training 会把两份数据首尾拼接后传入 domain_sizes
        # = (源域样本数, 目标域样本数)，这里只需按域切一刀。
        if not domain_sizes or len(domain_sizes) < 2:
            raise ValueError("cross_dataset 需要两个数据集（训练域 + 测试域）")
        n_a = int(domain_sizes[0])
        tr = np.arange(n_a)
        te = np.arange(n_a, len(y))
        if len(tr) == 0 or len(te) == 0:
            raise ValueError("cross_dataset 的两个域都不能为空")
        va, tr2 = _carve_val(tr, y, val_frac, seed)
        splits = [Split(0, f"cross-dataset[{n_a}→{len(te)}]", tr2, va, te)]

    elif scheme == "holdout":
        # 单次随机划分：测试集占 1/k（k=5 → 训练 80% / 测试 20%）
        tr, te = train_test_split(np.arange(len(y)), test_size=1.0 / max(k, 2),
                                  stratify=y, random_state=seed)
        va, tr2 = _carve_val(tr, y, val_frac, seed)
        splits.append(Split(0, f"holdout-{k - 1}:1", tr2, va, te))
    else:
        raise ValueError(f"未知划分方案：{scheme}")

    for sp in splits:
        assert not np.intersect1d(sp.train, sp.test).size, "训练/测试索引重叠"
        assert not np.intersect1d(sp.train, sp.val).size, "验证集泄漏进训练集"
        assert not np.intersect1d(sp.val, sp.test).size, "验证/测试索引重叠"
    return splits


# ---------------------------------------------------------------- 具体协议
def _loso_splits(y, sid, sess, val_frac, seed, test_session) -> list[Split]:
    """Leave-One-Subject-Out：每次留出一名被试（其指定 session）做测试。"""
    subjects = sorted(set(sid.tolist()))
    if len(subjects) < 2:
        raise ValueError("LOSO 需要至少 2 名被试的数据（跨被试泛化）")
    ts = str(test_session or "all").strip().upper()
    splits = []
    for i, s in enumerate(subjects):
        te = np.where(sid == s)[0]
        tr = np.where(sid != s)[0]          # 其余被试的**全部** session 都参与训练
        if ts not in ("ALL", ""):
            te = te[np.asarray(sess)[te] == ts]
        if len(tr) == 0 or len(te) == 0:
            continue
        va, tr = _carve_val(tr, y, val_frac, seed + i)   # 同 session：tr2 必须生效
        splits.append(Split(i, f"loso-S{int(s):03d}", tr, va, te, subject=int(s)))
    if not splits:
        raise ValueError(f"LOSO 划分失败：没有可用的测试被试（test_session={test_session}）")
    return splits


def _per_subject_split(idx, y_sub, k, train_percentage, seed):
    """被试内切出 (训练索引, 测试索引)（全局索引、已排序）。

    - 给了 ``train_percentage``：按类别各取 `int(每类样本数 × 比例)` 作训练，
      其余作测试 —— 对齐 EEG-FM-Benchmark 的 few-shot 切法（默认 0.3）
    - 未给：按 ``1/k`` 分层随机切
    """
    if train_percentage is not None:
        tp = float(train_percentage)
        rng = np.random.default_rng(seed)
        tr: list[int] = []
        te: list[int] = []
        for c in np.unique(y_sub):
            sel = idx[y_sub == c]
            n = len(sel)
            if n < 2:
                continue
            n_tr = max(1, min(int(n * tp), n - 1))   # 保证 train/test 都非空
            perm = rng.permutation(n)
            tr.extend(sel[perm[:n_tr]].tolist())
            te.extend(sel[perm[n_tr:]].tolist())
        return (np.array(sorted(tr), dtype=int), np.array(sorted(te), dtype=int))
    tr, te = train_test_split(idx, test_size=1.0 / max(k, 2), stratify=y_sub,
                              random_state=seed)
    return np.sort(tr), np.sort(te)


def _few_shot_splits(y, sid, sess, n_shot, n_repeats, val_frac, seed,
                     test_session, train_percentage=None) -> list[Split]:
    """被试内少样本：每被试每类抽 n_shot 个训练（或用 train_percentage 按比例），
    其余测试；重复 n_repeats 次。

    先对被试内重复取平均、再跨被试平均（与 LOSO 聚合方式一致）。
    """
    subjects = sorted(set(sid.tolist())) or [None]
    classes = sorted(set(np.asarray(y).tolist()))
    rng = np.random.default_rng(seed)
    ts = str(test_session or "all").strip().upper()
    n_shot = max(int(n_shot), 1)
    n_repeats = max(int(n_repeats), 1)
    splits: list[Split] = []
    fold = 0
    for s in subjects:
        base = (np.arange(len(y)) if s is None
                else np.where(np.asarray(sid) == s)[0])
        if ts not in ("ALL", ""):
            base = base[np.asarray(sess)[base] == ts]
        if len(base) == 0:
            continue
        for r in range(n_repeats):
            tr, te = [], []
            for c in classes:
                idx_c = base[np.asarray(y)[base] == c]
                if len(idx_c) < 2:      # 至少留 1 个做测试
                    continue
                perm = rng.permutation(len(idx_c))
                if train_percentage is not None:
                    # 按「每被试每类比例」取训练样本（对齐 EEG-FM-Benchmark）
                    k_use = max(1, min(int(len(idx_c) * float(train_percentage)),
                                       len(idx_c) - 1))
                else:
                    k_use = min(max(int(n_shot), 1), len(idx_c) - 1)
                tr += idx_c[perm[:k_use]].tolist()
                te += idx_c[perm[k_use:]].tolist()
            if not tr or not te:
                continue
            tr_arr = np.array(sorted(tr), dtype=int)
            va, tr_arr = _carve_val(tr_arr, np.asarray(y), val_frac, seed + fold)
            tag = f"{'S%03d-' % int(s) if s is not None else ''}shot{n_shot}-r{r + 1}"
            splits.append(Split(fold, tag, tr_arr, va, np.array(sorted(te), dtype=int),
                                subject=int(s) if s is not None else -1, repeat=r))
            fold += 1
    if not splits:
        raise ValueError(
            f"few_shot 划分失败：每类至少需要 2 个 trial（当前每类样本不足，"
            f"或 test_session={test_session} 无数据）")
    return splits


def _custom_splits(y, sid, folds, val_frac, seed) -> list[Split]:
    """自定义折：每折显式指定 train / val / test 的受试者名单。

    - 受试者号会用实际数据校验（未知编号直接报错，避免"静默空折"）
    - 同一折内 train/val/test 角色不得重叠
    - 未指定 val（或留空）时，**自动从该折 train 中按 val_frac 分层划出**，
      绝不会触碰 test
    """
    if not folds:
        raise ValueError("custom 划分需要提供 folds 列表（每折含 train[必填]/val/test）")
    known = {int(v) for v in np.unique(sid)}
    splits: list[Split] = []

    for i, spec in enumerate(folds):
        name = str(spec.get("name") or f"fold{i + 1}")

        def _ids(key, _spec=spec, _name=name):
            vals = _spec.get(key) or []
            if isinstance(vals, (int, np.integer)):
                vals = [vals]
            out = [int(v) for v in vals]
            unknown = [v for v in out if v not in known]
            if unknown:
                raise ValueError(
                    f"自定义折「{_name}」的 {key} 含未知受试者 {unknown}；"
                    f"数据中仅有的受试者：{sorted(known)}")
            return out

        tr_s, va_s, te_s = _ids("train"), _ids("val"), _ids("test")
        if not tr_s or not te_s:
            raise ValueError(f"自定义折「{name}」必须同时指定 train 与 test 受试者")
        overlap = ((set(tr_s) & set(te_s)) | (set(va_s) & set(te_s))
                   | (set(tr_s) & set(va_s)))
        if overlap:
            raise ValueError(f"自定义折「{name}」的受试者角色重叠：{sorted(overlap)}")

        tr = np.where(np.isin(sid, tr_s))[0]
        te = np.where(np.isin(sid, te_s))[0]
        if len(tr) == 0 or len(te) == 0:
            raise ValueError(f"自定义折「{name}」的 train/test 在数据中没有任何 trial")
        if va_s:
            va = np.where(np.isin(sid, va_s))[0]
        else:
            # 未指定验证受试者 → 自动从训练集里分层划出（不碰测试集）
            va, tr = _carve_val(tr, y, val_frac, seed + i)

        va_txt = f"va{len(va_s)}" if va_s else "va自动"
        tag = f"{name}(tr{len(tr_s)}/{va_txt}/te{len(te_s)})"
        splits.append(Split(i, tag, tr, va, te,
                            subject=te_s[0] if len(te_s) == 1 else -1))
    return splits


def _block_splits(y, sid, block, val_frac, seed, k) -> list[Split]:
    """按记录块（run/block）划分——同一 block 内的相邻 trial 绝不跨训练/测试。

    每名被试内部把 block 分成 k 组，每次留出一组做测试。比随机切 trial 更严格：
    连续采集的相邻试次高度相关，随机切会高估性能。
    """
    if block is None:
        raise ValueError(
            "block 划分需要预处理数据带有 block 信息；请用新版流水线重新生成 "
            "processed 数据（输出目录会多出 block.npy）")
    block = np.asarray(block)
    subjects = sorted(set(sid.tolist())) or [None]
    splits: list[Split] = []
    fold = 0
    for s in subjects:
        m = np.arange(len(y)) if s is None else np.where(sid == s)[0]
        blocks = sorted({int(b) for b in block[m]})
        if len(blocks) < 2:
            continue
        groups = np.array_split(np.array(blocks, dtype=int),
                                min(max(int(k), 2), len(blocks)))
        for i, g in enumerate(groups):
            sel = np.isin(block[m], g)
            te, tr = m[sel], m[~sel]
            if len(te) == 0 or len(tr) == 0:
                continue
            va, tr2 = _carve_val(tr, y, val_frac, seed + fold)
            prefix = f"S{int(s):02d}-" if s is not None else ""
            splits.append(Split(fold, f"{prefix}block-{i + 1}/{len(groups)}",
                                tr2, va, te,
                                subject=int(s) if s is not None else -1))
            fold += 1
    if not splits:
        raise ValueError("block 划分失败：每名被试至少需要 2 个记录块（block）")
    return splits


def _session_kfold_splits(y, sid, sess, val_frac, seed) -> list[Split]:
    """同被试跨 session：每个 (被试, 训练session → 测试session) 组合一折。

    与 ``session`` 方案的区别：``session`` 只做 T→E；本方案把所有有序组合都
    测一遍（T/E 两个 session 时即 T→E 与 E→T，每名被试 2 折），避免"方向偏好"。
    """
    subjects = sorted(set(sid.tolist())) or [None]
    sessions = sorted({str(v) for v in sess})
    if len(sessions) < 2:
        raise ValueError(f"session_kfold 需要至少 2 个 session（当前只有 {sessions}）")
    splits: list[Split] = []
    fold = 0
    for s in subjects:
        base = np.ones(len(y), dtype=bool) if s is None else (sid == s)
        for tr_s in sessions:
            for te_s in sessions:
                if tr_s == te_s:
                    continue
                tr = np.where(base & (sess == tr_s))[0]
                te = np.where(base & (sess == te_s))[0]
                if len(tr) == 0 or len(te) == 0:
                    continue
                va, tr2 = _carve_val(tr, y, val_frac, seed + fold)
                prefix = f"S{int(s):02d}-" if s is not None else ""
                splits.append(Split(fold, f"{prefix}{tr_s}->{te_s}", tr2, va, te,
                                    subject=int(s) if s is not None else -1))
                fold += 1
    if not splits:
        raise ValueError("session_kfold 划分失败：没有可用的 session 组合")
    return splits


def _carve_val(train_idx, y, val_frac, seed):
    """从训练索引中划出验证集（分层，用于早停/调参，绝不触碰测试折）。

    样本太少或某类样本不足导致分层切分失败时，退回为「不设验证集」，
    避免训练流程因小数据集直接报错。
    """
    idx = np.asarray(train_idx)
    if not (val_frac and val_frac > 0) or len(idx) <= 4:
        return np.array([], dtype=int), idx
    y_tr = y[idx]
    counts = np.bincount(y_tr, minlength=max(y_tr.max(), 0) + 1)
    if np.count_nonzero(counts) <= 1 or np.min(counts[counts > 0]) < 2:
        return np.array([], dtype=int), idx
    try:
        tr2, va = train_test_split(idx, test_size=val_frac, stratify=y_tr,
                                   random_state=seed)
        return va, tr2
    except ValueError:
        return np.array([], dtype=int), idx
