"""两个实验的对比：**先做一致性检查，再做指标对照**。

结果分析页选中两个实验时调用：

1. ``check_compatible`` —— 先判断能不能比：数据集必须一致（否则指标没有可比性），
   类别集合必须一致（否则混淆矩阵无法对照）；划分方案/模型/样本数不同只给提示；
2. ``compare_metrics`` —— 逐项指标对照（A / B / 差值 / 谁更好，自动区分
   "越大越好"与"越小越好"的指标）；
3. ``compare_per_subject`` —— 按被试逐人对照（用来看"谁对谁更好"）。
"""
from __future__ import annotations

from pathlib import Path

#: 对照表用的指标：(键, 显示名, 是否百分比, 是否越小越好)
METRICS: list[tuple[str, str, bool, bool | None]] = [
    ("acc", "Accuracy（准确率）", True, False),
    ("balanced_acc", "Balanced accuracy", True, False),
    ("kappa", "Cohen's kappa", False, False),
    ("mcc", "MCC（Matthews）", False, False),
    ("f1_macro", "F1 (macro)", False, False),
    ("precision_macro", "Precision (macro)", False, False),
    ("recall_macro", "Recall / 敏感度 (macro)", False, False),
    ("specificity_macro", "Specificity（特异度）", False, False),
    ("f1_weighted", "F1 (weighted)", False, False),
    ("auc_macro", "AUC-ROC (macro, OvR)", False, False),
    ("auc_pr_macro", "AUC-PR (macro)", False, False),
    ("brier", "Brier score", False, True),
    ("ece", "ECE（期望校准误差）", False, True),
    ("itr", "ITR（信息传输率 bits/min）", False, False),
    ("n_folds", "折数", False, None),
    ("n_trials", "总试次", False, None),
]


def dataset_key(summary: dict | None) -> str:
    """数据集标识：优先 ``data_dir`` 末段（如 bnci2014001）。"""
    s = summary or {}
    v = s.get("data_dir") or ""
    if not v:
        cfg = s.get("config") or {}
        v = ((cfg.get("data") or {}).get("processed_dir")) or ""
    if not v:
        v = s.get("dataset") or ""
    return Path(str(v)).name if str(v).strip() else "（未知）"


def class_key(summary: dict | None) -> tuple:
    s = summary or {}
    names = tuple(str(n) for n in (s.get("class_names") or []))
    return names or (int(s.get("n_classes") or 0),)


def _fmt_classes(summary: dict | None) -> str:
    s = summary or {}
    names = s.get("class_names") or []
    if names:
        return "、".join(map(str, names))
    return f"{s.get('n_classes', '?')} 类"


def check_compatible(a: dict, b: dict) -> dict:
    """对比前的一致性检查。

    Returns
    -------
    dict : ``{ok, checks[], notes[], blocking[]}``；``checks`` 每项为
    ``{name, ok, blocking, a, b, detail}``，``blocking=True and not ok`` 即拒绝对比。
    """
    checks: list[dict] = []

    da, db = dataset_key(a), dataset_key(b)
    checks.append({
        "name": "数据集", "ok": da == db, "blocking": True, "a": da, "b": db,
        "detail": "" if da == db else
                  f"数据集不一致（{da} vs {db}），两次实验的指标没有可比性"})

    ca, cb = class_key(a), class_key(b)
    checks.append({
        "name": "类别集合", "ok": ca == cb, "blocking": True,
        "a": _fmt_classes(a), "b": _fmt_classes(b),
        "detail": "" if ca == cb else "类别集合不一致，混淆矩阵与逐类指标无法对照"})

    sa, sb = a.get("scheme"), b.get("scheme")
    checks.append({
        "name": "划分方案", "ok": sa == sb, "blocking": False, "a": sa, "b": sb,
        "detail": "" if sa == sb else
                  f"划分方案不同（{sa} vs {sb}），准确率口径不同，只宜看趋势"})

    na, nb = a.get("n_trials"), b.get("n_trials")
    checks.append({
        "name": "样本数", "ok": na == nb, "blocking": False, "a": na, "b": nb,
        "detail": "" if na == nb else
                  f"样本数不同（{na} vs {nb}），预处理数据可能已改动"})

    checks.append({
        "name": "模型", "ok": True, "blocking": False,
        "a": a.get("model"), "b": b.get("model"), "detail": ""})

    blocking = [c for c in checks if c["blocking"] and not c["ok"]]
    return {"ok": not blocking, "checks": checks, "blocking": blocking,
            "notes": [c["detail"] for c in checks if c["detail"]]}


def compare_metrics(a: dict, b: dict) -> list[dict]:
    """逐项指标对照：A=基准（左）/ B=新实验，diff = B − A。"""
    aa, ab = a.get("aggregate") or {}, b.get("aggregate") or {}
    rows: list[dict] = []
    for key, label, pct, lower_better in METRICS:
        if key == "n_folds":
            ma = aa.get("n_folds") if aa else a.get("n_folds")
            mb = ab.get("n_folds") if ab else b.get("n_folds")
            sa = sb = None
        elif key == "n_trials":
            ma, mb = a.get("n_trials"), b.get("n_trials")
            sa = sb = None
        else:
            ma, sa = aa.get(f"{key}_mean"), aa.get(f"{key}_std")
            mb, sb = ab.get(f"{key}_mean"), ab.get(f"{key}_std")
        if ma is None and mb is None:
            continue
        diff = None if (ma is None or mb is None) else float(mb) - float(ma)
        if diff is None or lower_better is None:
            better = None if diff is None else "="
        elif abs(diff) < 1e-12:
            better = "="
        elif (diff < 0) if lower_better else (diff > 0):
            better = "B"
        else:
            better = "A"
        rows.append({"key": key, "label": label, "pct": bool(pct),
                     "lower_better": lower_better,
                     "a": None if ma is None else float(ma),
                     "a_std": None if sa is None else float(sa),
                     "b": None if mb is None else float(mb),
                     "b_std": None if sb is None else float(sb),
                     "diff": diff, "better": better})
    return rows


def winner_summary(rows: list[dict]) -> dict:
    """统计谁赢的指标更多（只看有明确方向的指标）。"""
    a = sum(1 for r in rows if r["better"] == "A" and r["lower_better"] is not None)
    b = sum(1 for r in rows if r["better"] == "B" and r["lower_better"] is not None)
    tie = sum(1 for r in rows if r["better"] == "=" and r["lower_better"] is not None)
    return {"a": a, "b": b, "tie": tie, "n": a + b + tie}


def compare_per_subject(a: dict, b: dict) -> list[dict]:
    """按被试逐人对照（缺失的一方记 None）。"""
    pa = {int(p["subject"]): p for p in (a.get("per_subject") or [])}
    pb = {int(p["subject"]): p for p in (b.get("per_subject") or [])}
    out = []
    for s in sorted(set(pa) | set(pb)):
        am = (pa.get(s) or {}).get("acc_mean")
        bm = (pb.get(s) or {}).get("acc_mean")
        out.append({"subject": s,
                    "a": None if am is None else float(am),
                    "b": None if bm is None else float(bm),
                    "diff": None if (am is None or bm is None) else float(bm) - float(am)})
    return out


def fold_accs(summary: dict) -> list[float]:
    folds = summary.get("folds") or []
    return [float(f.get("acc")) for f in folds if f.get("acc") is not None]
