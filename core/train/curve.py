"""Few-Shot 学习曲线：扫描不同校准样本数 n_shot，得到准确率曲线。

回答的问题：**每类需要多少个校准样本，被试内性能才够用？**

用法::

    from core.train import run_few_shot_curve
    curve = run_few_shot_curve(config, shots=(1, 5, 10, 20))
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np


def run_few_shot_curve(config: dict, shots=(1, 5, 10, 20), log=print,
                       stop_event=None, save: bool = True) -> dict:
    """对每个 n_shot 跑一次 Within-Subject Few-Shot 训练，汇总成学习曲线。

    Returns
    -------
    dict: {base_name, shots, acc_mean, acc_std, per_subject_acc_mean,
           per_subject_acc_std, results: [...每个 shot 的明细...], json_path}
    """
    from . import run_training  # 延迟导入，避免与包初始化循环

    shots = [int(s) for s in shots]
    base_name = str((config.get("experiment") or {}).get("name", "fewshot"))
    results: list[dict] = []
    failed: list[dict] = []
    for k in shots:
        if stop_event is not None and stop_event.is_set():
            log("已停止：提前结束学习曲线扫描。")
            break
        cfg = copy.deepcopy(config)
        cfg.setdefault("split", {})
        cfg["split"].update({"scheme": "few_shot", "n_shot": int(k)})
        cfg.setdefault("experiment", {})
        cfg["experiment"]["name"] = f"{base_name}_shot{k}"
        cfg["experiment"]["note"] = f"Few-Shot 学习曲线：每类 {k} 个校准样本"
        log(f"\n===== n_shot = {k} =====")
        try:
            summary = run_training(cfg, log=log, stop_event=stop_event)
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            log(f"n_shot={k} 训练失败，跳过：{msg}")
            failed.append({"n_shot": int(k), "error": msg})
            continue

        agg = summary.get("aggregate", {}) or {}
        per_subj = summary.get("per_subject") or []
        subj_acc = [float(p["acc_mean"]) for p in per_subj]
        results.append({
            "n_shot": int(k),
            "acc_mean": float(agg.get("acc_mean", 0.0)),
            "acc_std": float(agg.get("acc_std", 0.0)),
            "n_folds": int(agg.get("n_folds", 0)),
            "n_subjects": len(per_subj),
            "per_subject_acc_mean": float(np.mean(subj_acc)) if subj_acc else None,
            "per_subject_acc_std": (float(np.std(subj_acc, ddof=1))
                                    if len(subj_acc) > 1 else 0.0),
            "exp_dir": summary.get("exp_dir"),
        })

    curve = {
        "base_name": base_name,
        "shots": [r["n_shot"] for r in results],
        "acc_mean": [r["acc_mean"] for r in results],
        "acc_std": [r["acc_std"] for r in results],
        "per_subject_acc_mean": [r["per_subject_acc_mean"] for r in results],
        "per_subject_acc_std": [r["per_subject_acc_std"] for r in results],
        "results": results,
        "failed": failed,
        "json_path": None,
    }
    if failed:
        log("以下 n_shot 未成功（属方法本身限制）：")
        for f_ in failed:
            log(f"  · n_shot={f_['n_shot']}：{f_['error']}")
        log("提示：样本数 ≤ 类别数时 LDA 无法拟合，1-shot 四分类请改用神经网络模型，"
            "或减少类别数（如只做左右手二分类）。")
    if save and results:
        out = Path("experiments") / f"fewshot_curve_{base_name}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(curve, ensure_ascii=False, indent=2), encoding="utf-8")
        curve["json_path"] = str(out.resolve())
        log(f"学习曲线数据已保存 → {curve['json_path']}")
    return curve


def plot_few_shot_curve(curve: dict, ax=None):
    """把学习曲线画到 matplotlib Axes 上（GUI/CLI 共用）。"""
    import matplotlib.pyplot as plt

    # 统一字体：数字/字母 Times New Roman、中文宋体（跨平台，避免方框）
    from ..plot_fonts import apply_matplotlib_font

    apply_matplotlib_font()

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    x = np.asarray(curve.get("shots", []), dtype=float)
    y = np.asarray(curve.get("acc_mean", []), dtype=float) * 100
    e = np.asarray(curve.get("acc_std", []), dtype=float) * 100
    if len(x) == 0:
        ax.set_title("（无数据）")
        return ax
    ax.errorbar(x, y, yerr=e, marker="o", capsize=4, lw=1.6,
                color="#4c78a8", label="所有折平均")
    psm = curve.get("per_subject_acc_mean") or []
    if psm and all(v is not None for v in psm):
        ax.plot(x, np.asarray(psm, dtype=float) * 100, marker="s", ls="--", lw=1.4,
                color="#d62728", label="按被试平均")
    ax.set_xlabel("每类校准样本数 n_shot")
    ax.set_ylabel("准确率 (%)")
    ax.set_title("Within-Subject Few-Shot 学习曲线")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.minorticks_off()          # 显式刻度已给定，去掉次刻度的 3×10⁰ 之类标签
    ax.grid(alpha=0.3, which="major")
    ax.legend(fontsize=8)
    failed = curve.get("failed") or []
    if failed:
        ax.text(0.02, 0.03,
                "未成功：" + "，".join(f"n_shot={f['n_shot']}" for f in failed)
                + "（样本数 ≤ 类别数，LDA 无法拟合）",
                transform=ax.transAxes, fontsize=7, color="#c8102e")
    return ax
