"""评估指标、跨折聚合、统计检验与报告导出。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, cohen_kappa_score,
                             confusion_matrix, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)


def compute_metrics(y_true, y_pred, class_names=None, labels=None, y_prob=None) -> dict:
    """分类指标（尽量全）。

    Parameters
    ----------
    labels : 显式类标签列表（推荐由调用方传全局类别，如 0..K-1）。
             若不传则按当前这批数据推断——**注意**这会因某折恰好缺类而使混淆
             矩阵形状在不同折之间不一致，进而导致跨折聚合失败。
    y_prob : shape (N, K) 预测概率，列顺序需与 ``labels`` 对齐。只有给了它才会
             计算 AUC / AUC-PR / Brier / ECE 这类**需要概率**的指标。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    else:
        # 固定为全局类别，保证各折混淆矩阵形状一致（跨折可聚合）
        labels = [int(v) for v in labels]
    names = [str(n) for n in (class_names or [str(l) for l in labels])]

    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # 每类特异度（one-vs-rest）：TN / (TN + FP)
    total = float(cm.sum())
    spec = []
    for i in range(len(labels)):
        tp = float(cm[i, i])
        fn = float(cm[i, :].sum()) - tp
        fp = float(cm[:, i].sum()) - tp
        tn = total - tp - fn - fp
        spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    out = {
        "acc": float((y_true == y_pred).mean()),
        "balanced_acc": float(r_mac),          # 宏平均召回 = balanced accuracy
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "mcc": (float(matthews_corrcoef(y_true, y_pred)) if len(labels) > 1 else 0.0),
        "precision_macro": float(p_mac),
        "recall_macro": float(r_mac),
        "f1_macro": float(f1_mac),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),
        "specificity_macro": float(np.mean(spec)) if spec else 0.0,
        "auc_macro": None,
        "auc_pr_macro": None,
        "brier": None,                          # 越小越好
        "ece": None,                            # 期望校准误差，越小越好
        "per_class": {
            "labels": [str(l) for l in labels],
            "names": names,
            "precision": [round(float(x), 4) for x in p],
            "recall": [round(float(x), 4) for x in r],
            "specificity": [round(float(x), 4) for x in spec],
            "f1": [round(float(x), 4) for x in f1],
            "support": [int(x) for x in sup],
            "auc": [None] * len(labels),
        },
        "confusion": cm.tolist(),
    }

    # ---- 需要概率的指标 ----
    prob = np.asarray(y_prob, dtype=float) if y_prob is not None else None
    if prob is not None and prob.ndim == 2 and prob.shape[1] == len(labels):
        idx = np.searchsorted(labels, y_true)
        onehot = np.eye(len(labels))[idx]
        try:
            if len(labels) == 2:
                out["auc_macro"] = float(roc_auc_score(idx, prob[:, 1]))
                out["auc_pr_macro"] = float(average_precision_score(idx, prob[:, 1]))
            else:
                out["auc_macro"] = float(roc_auc_score(
                    idx, prob, multi_class="ovr", average="macro",
                    labels=list(range(len(labels)))))
                out["auc_pr_macro"] = float(average_precision_score(
                    onehot, prob, average="macro"))
            out["per_class"]["auc"] = [
                round(float(roc_auc_score(onehot[:, i], prob[:, i])), 4)
                for i in range(len(labels))]
        except Exception:  # noqa: BLE001  （某类样本缺失等情况）
            pass
        # Brier（多类 = 每类平方误差之和的均值）
        out["brier"] = float(np.mean(np.sum((prob - onehot) ** 2, axis=1)))
        # ECE：10 分箱的置信度-准确率加权偏差
        conf, pred_idx = prob.max(1), prob.argmax(1)
        ok = (pred_idx == idx)
        ece = 0.0
        for lo in np.linspace(0.0, 0.9, 10):
            m = (conf > lo) & (conf <= lo + 0.1)
            if m.sum():
                ece += float(m.mean()) * abs(float(ok[m].mean()) - float(conf[m].mean()))
        out["ece"] = float(ece)
    return out


def _mean_std(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None
    return (float(np.mean(vals)),
            float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)


def information_transfer_rate(acc, n_classes, trial_seconds) -> float | None:
    """BCI 信息传输率 ITR（bits/min）。

    ITR = (log2(N) + P·log2(P) + (1-P)·log2((1-P)/(N-1))) · 60 / T
    其中 P=准确率、N=类别数、T=一次决策占用的秒数（含 cue+想象+休息，
    BNCI2014-001 约 8 s，可用 ``evaluation.trial_seconds`` 配置）。
    """
    try:
        n_classes, trial_seconds = int(n_classes), float(trial_seconds)
    except (TypeError, ValueError):
        return None
    if n_classes < 2 or trial_seconds <= 0:
        return None
    p = min(max(float(acc), 1e-6), 1 - 1e-6)
    bits = (np.log2(n_classes) + p * np.log2(p)
            + (1 - p) * np.log2((1 - p) / (n_classes - 1)))
    return float(max(bits, 0.0) * 60.0 / trial_seconds)


#: 参与跨折聚合的标量指标
_AGG_KEYS = ("acc", "balanced_acc", "kappa", "mcc", "precision_macro", "recall_macro",
             "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted",
             "specificity_macro", "auc_macro", "auc_pr_macro", "brier", "ece")


def aggregate(fold_metrics: list[dict], n_classes: int | None = None,
              trial_seconds: float | None = None) -> dict:
    """跨折聚合：每个标量指标都给出 mean ± std（None 的指标自动跳过）。"""
    conf = np.sum([np.asarray(m["confusion"]) for m in fold_metrics], axis=0)
    out: dict = {
        "n_folds": len(fold_metrics),
        "confusion_total": conf.tolist(),
        "acc_per_fold": [round(float(m["acc"]), 4) for m in fold_metrics],
    }
    for key in _AGG_KEYS:
        mean, std = _mean_std([m.get(key) for m in fold_metrics])
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std
    # ITR（BCI 特有）：用每折 acc 各自算，再取均值±std
    if n_classes and trial_seconds:
        itrs = [information_transfer_rate(m["acc"], n_classes, trial_seconds)
                for m in fold_metrics]
        out["itr_mean"], out["itr_std"] = _mean_std(itrs)
    else:
        out["itr_mean"], out["itr_std"] = None, None
    return out


def per_subject_summary(fold_metrics: list[dict]) -> list[dict]:
    """按被试聚合：被试内先平均（few_shot 的多次重复 / session 的多折），再跨被试。

    用于 LOSO / few_shot / session 场景——这些场景下"每名被试一个准确率"
    比"所有折混在一起"更有意义（与论文报告方式一致）。
    """
    from collections import defaultdict

    groups: dict[int, list[float]] = defaultdict(list)
    for f in fold_metrics:
        s = int(f.get("subject", -1))
        if s >= 0:
            groups[s].append(float(f["acc"]))
    return [{"subject": s, "acc_mean": float(np.mean(v)),
             "acc_std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
             "n_folds": len(v)}
            for s, v in sorted(groups.items())]


def paired_tests(accs_a, accs_b) -> dict:
    """模型间配对检验（同一折/同一被试上的两组准确率）。"""
    from scipy import stats

    a, b = np.asarray(accs_a, float), np.asarray(accs_b, float)
    if len(a) != len(b) or len(a) < 2:
        return {"ttest_p": None, "wilcoxon_p": None, "note": "样本不足或长度不等"}
    try:
        t = stats.ttest_rel(a, b)
        ttest_p = float(t.pvalue)
    except Exception:
        ttest_p = None
    try:
        w = stats.wilcoxon(a, b)
        wilcoxon_p = float(w.pvalue)
    except Exception:
        wilcoxon_p = None
    return {"ttest_p": ttest_p, "wilcoxon_p": wilcoxon_p}


def export_markdown_report(exp_dir: str | Path, summary: dict, config: dict) -> Path:
    """把 summary 导出为 Markdown 实验报告（含配置、指标、混淆矩阵表）。"""
    exp_dir = Path(exp_dir)
    lines = [f"# 实验报告 — {summary.get('experiment', '')}", "",
             f"- 时间：{summary.get('timestamp', '')}",
             f"- 模型：`{summary.get('model', '')}`  划分：`{summary.get('scheme', '')}`",
             f"- 数据：`{summary.get('data_dir', '')}` "
             f"（{summary.get('n_trials', '?')} trials, {summary.get('n_classes', '?')} 类）",
             ""]
    agg = summary.get("aggregate", {})
    if agg:
        def _fmt(key, pct=False, nd=3):
            m, s = agg.get(f"{key}_mean"), agg.get(f"{key}_std")
            if m is None:
                return "—"
            if pct:
                return f"{m * 100:.2f} ± {(s or 0) * 100:.2f} %"
            return f"{m:.{nd}f} ± {(s or 0):.{nd}f}"

        itr = agg.get("itr_mean")
        itr_txt = (f"{itr:.2f} ± {(agg.get('itr_std') or 0):.2f}" if itr is not None
                   else "—（需配置 evaluation.trial_seconds）")
        lines += [
            "## 总体结果", "",
            f"共 **{agg.get('n_folds', '?')}** 折。", "",
            "| 指标 | 均值 ± 标准差 | 说明 |", "|---|---|---|",
            f"| Accuracy | {_fmt('acc', pct=True)} | 总体正确率 |",
            f"| Balanced accuracy | {_fmt('balanced_acc', pct=True)} | 各类召回的平均（不平衡时更可靠） |",
            f"| Cohen's kappa | {_fmt('kappa')} | 扣除随机一致后的吻合度 |",
            f"| MCC | {_fmt('mcc')} | Matthews 相关系数（-1 ~ 1） |",
            f"| Precision (macro) | {_fmt('precision_macro')} | 各类精确率的平均 |",
            f"| Recall / Sensitivity (macro) | {_fmt('recall_macro')} | 各类召回率的平均 |",
            f"| Specificity (macro) | {_fmt('specificity_macro')} | 各类特异度的平均 |",
            f"| F1 (macro) | {_fmt('f1_macro')} | 精确率/召回的调和平均 |",
            f"| F1 (weighted) | {_fmt('f1_weighted')} | 按样本数加权的 F1 |",
            f"| AUC-ROC (macro, OvR) | {_fmt('auc_macro')} | 排序能力（0.5 = 随机） |",
            f"| AUC-PR (macro) | {_fmt('auc_pr_macro')} | PR 曲线面积（不平衡数据更敏感） |",
            f"| Brier score ↓ | {_fmt('brier')} | 概率预测均方误差，越小越好 |",
            f"| ECE ↓ | {_fmt('ece')} | 期望校准误差，越小越好 |",
            f"| ITR (bits/min) | {itr_txt} | 信息传输率（BCI 特有指标） |",
            "",
        ]
    lines += ["## 各折明细", "",
              "| 折 | tag | acc | balanced | kappa | MCC | F1(macro) | AUC | specificity |",
              "|---|---|---|---|---|---|---|---|---|"]

    def _g(fold: dict, key: str, nd: int = 4) -> str:
        v = fold.get(key)
        return "—" if v is None else f"{float(v):.{nd}f}"

    for f in summary.get("folds", []):
        lines.append(
            f"| {f['fold']} | {f['tag']} | {_g(f, 'acc')} | {_g(f, 'balanced_acc')} | "
            f"{_g(f, 'kappa')} | {_g(f, 'mcc')} | {_g(f, 'f1_macro')} | "
            f"{_g(f, 'auc_macro')} | {_g(f, 'specificity_macro')} |")

    import numpy as np

    per_subj = summary.get("per_subject") or []
    if len(per_subj) > 1:
        accs = [p["acc_mean"] for p in per_subj]
        lines += ["", "## 按被试汇总（LOSO / few-shot / session 场景）", "",
                  "| 被试 | acc | 折数 |", "|---|---|---|"]
        for p in per_subj:
            lines.append(f"| S{p['subject']:03d} | {p['acc_mean']:.4f} | {p['n_folds']} |")
        lines += ["", f"**被试均值：{np.mean(accs) * 100:.2f} ± "
                      f"{np.std(accs, ddof=1) * 100:.2f} %**（{len(accs)} 名被试）"]

    conf = np.asarray(agg.get("confusion_total", []))
    if conf.size:
        names = summary.get("class_names") or [str(i) for i in range(len(conf))]
        lines += ["", "## 混淆矩阵（合计）", "",
                  "| 真实\\预测 | " + " | ".join(map(str, names)) + " |",
                  "|---|" + "---|" * len(names)]
        for i, row in enumerate(conf):
            lines.append(f"| {names[i] if i < len(names) else i} | "
                         + " | ".join(str(int(v)) for v in row) + " |")

    # 训练参数来源（从论文文本抽取时才有）：把"这套超参抄自哪篇论文哪一节"写进报告
    prov = (config or {}).get("paper_params")
    if prov:
        try:
            from ..paper_text import markdown_provenance

            lines += ["", markdown_provenance(prov)]
        except Exception:  # noqa: BLE001 - 溯源渲染失败不该影响报告生成
            lines += ["", "## 训练参数来源（论文抽取）", "",
                      f"- 输入：{(prov.get('input') or {}).get('source_summary', '')}",
                      f"- 方式：{prov.get('method', '')}", ""]

    lines += ["", "## 配置", "", "```yaml", json.dumps(config, ensure_ascii=False, indent=2,
                                                       default=str), "```", ""]
    out = exp_dir / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
