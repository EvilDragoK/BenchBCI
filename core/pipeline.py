"""预处理流水线编排器 + 命令行入口。

流水线配置（YAML）示例见 configs/preprocess_bnci2014001.yaml。
命令行运行（与 GUI 结果完全一致，保证可复现）::

    python -m core.pipeline configs/preprocess_bnci2014001.yaml \
        [--root /path/to/BNCI2014001] [--out processed/xxx] [--subjects 1-9] [--sessions T,E]

输出目录结构::

    <out>/
    ├── S01T-epo.fif ... S09E-epo.fif   # 每被试每 session 的 Epochs（可回看）
    ├── X.npy y.npy sid.npy sess.npy    # 训练用张量（float32, Volt）
    ├── classes.json meta.json          # 类别映射与全局元信息
    ├── pipeline.yaml                   # 配置快照（复现用）
    └── report.md                       # 预处理报告（可直接写进论文方法部分）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import yaml

from .io.bnci import load_subject_raw, list_subjects
from .preprocess.operators import build_operator

from . import PROJECT_ROOT


# ---------------------------------------------------------------- 工具函数
def parse_subjects(spec, available=None) -> list[int]:
    """'1-9' / '1,3,5' / 'all' → [1, ..., 9]。"""
    spec = str(spec).strip().lower()
    if spec in ("all", ""):
        return list(available) if available is not None else []
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")[:2]
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(float(part)))
    if available is not None:
        out = {s for s in out if s in set(available)}
    return sorted(out)


def resolve_path(p, base: Path | None = None) -> Path:
    p = Path(p).expanduser()
    if p.is_absolute():
        return p
    return (base or PROJECT_ROOT) / p


# ---------------------------------------------------------------- 流水线
class Pipeline:
    """有序步骤链。steps = [(Operator, enabled)]。"""

    def __init__(self, steps: list[tuple] | None = None):
        self.steps = steps or []

    @classmethod
    def from_config(cls, cfg: list[dict]) -> "Pipeline":
        steps = []
        for s in cfg:
            op = build_operator(s["op"], s.get("params") or {})
            steps.append((op, bool(s.get("enabled", True))))
        return cls(steps)

    def to_config(self) -> list[dict]:
        return [
            {"op": op.name, "params": {k: _plain(v) for k, v in op.params.items()},
             "enabled": bool(enabled)}
            for op, enabled in self.steps
        ]

    def run(self, raw, ctx: dict | None = None):
        """依次执行启用的算子，返回 (Raw|Epochs, ctx)。"""
        ctx = ctx if ctx is not None else {"report": [], "log": []}
        obj = raw
        for op, enabled in self.steps:
            if not enabled:
                ctx["log"].append({"op": op.name, "params": op.params, "seconds": 0.0, "skipped": True})
                continue
            t0 = time.perf_counter()
            obj = op.apply(obj, ctx)
            ctx["log"].append({"op": op.name, "params": dict(op.params),
                               "seconds": round(time.perf_counter() - t0, 3)})
        return obj, ctx


def _plain(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.ndarray, list, tuple)):
        return [ _plain(x) for x in v ]
    return v


# ---------------------------------------------------------------- 数据集级执行
def run_dataset_pipeline(config: dict, log=print) -> dict:
    """对数据集全部（指定）被试执行流水线并保存输出。"""
    ds = config["dataset"]
    root = resolve_path(ds["root"])
    kind = str(ds.get("kind", "bnci2014001")).lower()
    is_pn = kind in ("physionet_mi", "physionet", "eegmmidb")
    if is_pn:
        from .io.physionet import DEFAULT_RUNS
        from .io.physionet import list_subjects as _list_subjects
        from .io.physionet import load_subject_raw as _load_raw
    else:
        from .io.bnci import list_subjects as _list_subjects
        from .io.bnci import load_subject_raw as _load_raw

    available = _list_subjects(root)
    if not available:
        raise FileNotFoundError(f"目录中未找到 {kind} 数据：{root}")
    subjects = parse_subjects(ds.get("subjects", "all"), available)
    if not subjects:
        subjects = available
    eeg_only = bool(ds.get("eeg_only", False))
    # PhysioNet MI：sessions 表示要合并的 run 列表（一次加载得到多类任务）
    if is_pn:
        raw_runs = ds.get("runs") or [int(str(s).lstrip("Rr"))
                                      for s in _as_list(ds.get("sessions", []))] or list(DEFAULT_RUNS)
        runs = [int(r) for r in raw_runs]
        sessions = [f"R{'+'.join(str(r) for r in runs)}"]
    else:
        runs = None
        sessions = [str(s).upper() for s in _as_list(ds.get("sessions", ["T"]))]

    out_cfg = config.get("output", {})
    out_dir = resolve_path(out_cfg.get("dir", "processed/run"))
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline.from_config(config.get("steps", []))

    Xs, ys, sids, sesss, blocks = [], [], [], [], []
    block_offset = 0   # 让 block 编号跨 (被试, session) 全局唯一
    classes_map: dict | None = None
    ch_names, sfreq = None, None
    t0 = time.perf_counter()
    report_lines = [
        f"# 预处理报告 — {config.get('name', 'pipeline')}",
        "",
        f"- 数据集根目录：`{root}`",
        f"- 被试：{subjects}",
        f"- Sessions：{sessions}",
        f"- 流水线步骤：{len(pipeline.steps)}",
        "",
        "| 被试/Session | 原始 trial 数 | 保留 trial 数 | 剔除比例 | 坏导 | ICA 剔除成分 | 耗时 (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    # 数据质量 / 伪迹处理统计（跨被试累加）
    bad_seg_total, clipped_total = 0.0, 0
    reg_notes: list[str] = []
    ch_positions = None
    montage_info = None

    for subj in subjects:
        for sess in sessions:
            try:
                if is_pn:
                    raw, meta = _load_raw(root, subj, runs=runs, eeg_only=eeg_only)
                else:
                    raw, meta = _load_raw(root, subj, sess, eeg_only=eeg_only)
            except FileNotFoundError as e:
                log(f"[跳过] {e}")
                continue
            t_s = time.perf_counter()
            epochs, ctx = pipeline.run(raw.copy())
            X = epochs.get_data().astype(np.float32)
            y = epochs.events[:, 2].astype(np.int64)
            if X.shape[0] == 0:
                log(f"[警告] S{subj:02d}{sess} 分段后无 trial，跳过保存")
                continue
            if classes_map is None:
                classes_map = ctx.get("classes", {})
                ch_names = list(epochs.ch_names)
                sfreq = float(epochs.info["sfreq"])
            Xs.append(X)
            ys.append(y)
            sids.append(np.full(len(y), subj, dtype=np.int64))
            sesss.append(np.array([sess] * len(y)))
            seg = _epoch_blocks(raw, epochs)
            blocks.append(seg + block_offset)
            block_offset += int(seg.max()) + 1 if len(seg) else 0

            if out_cfg.get("save_fif", True):
                epochs.save(str(out_dir / f"S{subj:02d}{sess}-epo.fif"), overwrite=True)

            n_before = ctx.get("n_trials_before", len(y))
            n_rej = ctx.get("rejected_trials", 0)
            bads = ",".join(ctx.get("bad_channels", [])) or "-"
            ica_ex = ",".join(map(str, ctx.get("ica_exclude", []))) or "-"
            dt = time.perf_counter() - t_s
            report_lines.append(
                f"| S{subj:02d}{sess} | {n_before} | {len(y)} | "
                f"{n_rej / max(n_before, 1) * 100:.1f}% | {bads} | {ica_ex} | {dt:.1f} |")
            log(f"S{subj:02d}{sess}: {n_before} → {len(y)} trials "
                f"(剔除 {n_rej}, {dt:.1f}s)")
            if n_before and n_rej / n_before > 0.5:
                log(f"    [警告] 剔除比例 {n_rej / n_before * 100:.0f}% 偏高，"
                    f"建议放宽振幅阈值或启用 ICA")

            # --- 数据质量与伪迹处理统计 ---
            bad_seg_total += float(ctx.get("bad_segment_seconds", 0.0) or 0.0)
            clipped_total += int(ctx.get("clipped_samples", 0) or 0)
            if ctx.get("dropped_bad_segments"):
                log(f"    [坏段] 按 BAD_* 注释丢弃 {ctx['dropped_bad_segments']} 个 trial")
            if ctx.get("outlier_epochs"):
                log(f"    [离群] IQR 准则剔除 {ctx['outlier_epochs']} 个 trial")
            if ctx.get("ica_candidate_reasons"):
                dropped = set(ctx.get("ica_exclude") or [])
                for k, why in sorted(ctx["ica_candidate_reasons"].items()):
                    tag = "已剔除" if k in dropped else "候选（未剔除）"
                    log(f"    [ICA] 成分 {k} {tag}：{'；'.join(why)}")
            elif "ica_features" in ctx:
                log("    [ICA] 未发现明显伪迹成分（各判据均未达阈值）")
            if ctx.get("bad_segments"):
                log(f"    [坏段] 共 {len(ctx['bad_segments'])} 段 / "
                    f"{ctx.get('bad_segment_seconds', 0)} s")
            if ch_positions is None and ctx.get("channel_positions"):
                ch_positions = ctx["channel_positions"]
                montage_info = ctx.get("montage")
            reg = ctx.get("artifact_regression")
            if reg:
                reg_notes.append(f"S{subj:02d}{sess}→{','.join(reg['proxies'])}")
            for msg in ctx.get("warnings", []):
                log(f"    [提示] {msg}")

            # 每步耗时写日志
            for entry in ctx.get("log", []):
                if not entry.get("skipped"):
                    log(f"    · {entry['op']} {entry['seconds']}s params={entry['params']}")

    if not Xs:
        raise RuntimeError("流水线未产生任何 epoch，请检查事件/分段参数")

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    sid = np.concatenate(sids, axis=0)
    sess = np.concatenate(sesss, axis=0)
    block = np.concatenate(blocks, axis=0)

    if out_cfg.get("save_npy", True):
        np.save(out_dir / "X.npy", X)
        np.save(out_dir / "y.npy", y)
        np.save(out_dir / "sid.npy", sid)
        np.save(out_dir / "sess.npy", sess)
        np.save(out_dir / "block.npy", block)   # 供 block（按记录块）划分使用
    with open(out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes_map, f, ensure_ascii=False, indent=2)
    meta_out = {
        "dataset_root": str(root),
        "pipeline_name": config.get("name", ""),
        "subjects": subjects,
        "sessions": sessions,
        "sfreq": sfreq,
        "channel_names": ch_names,
        "n_epochs": int(len(y)),
        "n_times": int(X.shape[2]),
        "tmin": float(epochs.tmin),
        "tmax": float(epochs.tmax),
        "class_counts": {str(k): int((y == v).sum()) for k, v in classes_map.items()},
        "n_blocks": int(len(np.unique(block))),
        "total_seconds": round(time.perf_counter() - t0, 1),
        "steps": pipeline.to_config(),
        # 导联定位（电极坐标，米/头部坐标系）+ 数据质量统计，便于追溯与复现
        "montage": montage_info or {},
        "channel_positions": ch_positions or {},
        "bad_segment_seconds": round(bad_seg_total, 1),
        "clipped_samples": int(clipped_total),
        "artifact_regression": reg_notes,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)
    with open(out_dir / "pipeline.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    montage_txt = "未设置"
    if montage_info:
        montage_txt = (f"{montage_info.get('name') or montage_info.get('path') or '自定义'}"
                       f"（{len(ch_positions or {})} 个通道有坐标）")
    report_lines += [
        "",
        f"**总计**：{len(y)} 个 epoch，{X.shape[1]} 通道 × {X.shape[2]} 采样点，"
        f"采样率 {sfreq:g} Hz，总耗时 {meta_out['total_seconds']}s。",
        "",
        "## 数据质量与伪迹处理",
        f"- 导联定位：{montage_txt}",
        f"- 坏段检测（连续时间段）：共 {bad_seg_total:.1f} s",
        f"- 去极值裁剪：{clipped_total} 个采样点",
        f"- 回归法去伪迹：{'; '.join(reg_notes) if reg_notes else '未启用'}",
        "",
        "## 流水线配置（快照）",
        "```yaml",
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False).strip(),
        "```",
    ]
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    log(f"流水线完成 → {out_dir}（{len(y)} epochs, 总耗时 {meta_out['total_seconds']}s）")
    return {"out_dir": str(out_dir), "n_epochs": int(len(y)), "meta": meta_out}


def _epoch_blocks(raw, epochs) -> np.ndarray:
    """判断每个 epoch 落在原始（拼接后）数据的哪一段——用于 block 划分。

    BNCI2014001 的 T/E session 各由多个 run 拼成，MNE 会在拼接处插入
    boundary 标注；据此即可把 trial 归到所属记录块，避免同一段内的相邻
    trial 被随机拆到训练/测试集（会高估性能）。
    """
    fs = float(raw.info["sfreq"])
    desc = np.asarray([str(d) for d in raw.annotations.description])
    ons = np.asarray(raw.annotations.onset, dtype=float)
    mask = np.array([("boundary" in d.lower()) for d in desc])
    cuts = np.unique(np.concatenate([[0.0], ons[mask]])) if mask.any() else np.array([0.0])
    t = np.asarray(epochs.events[:, 0], dtype=float) / fs
    idx = np.searchsorted(cuts, t, side="right") - 1
    return np.clip(idx, 0, max(len(cuts) - 1, 0)).astype(np.int64)


def _as_list(v) -> list:
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        return [s.strip().upper() for s in v.split(",") if s.strip()]
    return [v]


# ---------------------------------------------------------------- CLI
#: 配置里的环境变量占位符：``${VAR}`` 或 ``${VAR:-默认值}``
_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value, env: dict | None = None):
    """递归展开配置字符串里的 ``${VAR}`` / ``${VAR:-默认值}``。

    这样配置文件可以写成 ``root: ${BCI_DATA_ROOT:-~/Desktop/DL/BCI/data}/BNCI2014001``：
    本机不设环境变量就用默认路径，换到 Windows/别人的机器只设一个变量即可，不用改 YAML。
    """
    env = os.environ if env is None else env
    if isinstance(value, str):
        out = _ENV_PLACEHOLDER.sub(
            lambda m: env.get(m.group(1),
                              m.group(2) if m.group(2) is not None else ""), value)
        return os.path.expanduser(out) if out.startswith("~") else out
    if isinstance(value, dict):
        return {k: expand_env(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v, env) for v in value]
    return value


def load_config(path: str | Path) -> dict:
    """读 YAML 配置（自动展开 ``${VAR}`` / ``${VAR:-默认值}``）。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return expand_env(cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="core.pipeline", description="命令行执行预处理流水线")
    ap.add_argument("config", help="流水线 YAML 配置路径")
    ap.add_argument("--root", help="覆盖数据集根目录")
    ap.add_argument("--out", help="覆盖输出目录")
    ap.add_argument("--subjects", help="覆盖被试，如 1-9 或 1,3,5")
    ap.add_argument("--sessions", help="覆盖 session，如 T,E")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    if args.root:
        config["dataset"]["root"] = args.root
    if args.out:
        config.setdefault("output", {})["dir"] = args.out
    if args.subjects:
        config["dataset"]["subjects"] = args.subjects
    if args.sessions:
        config["dataset"]["sessions"] = [s.upper() for s in args.sessions.split(",")]

    return run_dataset_pipeline(config, log=print)


if __name__ == "__main__":
    main()
