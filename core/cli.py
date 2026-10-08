"""统一命令行入口::

    python -m core.cli pipeline configs/preprocess_bnci2014001.yaml [--subjects 1-9]
    python -m core.cli train    configs/train_eegnet.yaml [--epochs N] [--device cpu]
    python -m core.cli list     [关键字]
    python -m core.cli report   experiments/<exp_dir>      # 重新生成/查看摘要
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="core.cli", description="BCI Lab 命令行工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("pipeline", help="执行预处理流水线")
    p1.add_argument("config")
    p1.add_argument("--root")
    p1.add_argument("--out")
    p1.add_argument("--subjects")
    p1.add_argument("--sessions")

    p2 = sub.add_parser("train", help="执行训练")
    p2.add_argument("config")
    p2.add_argument("--epochs", type=int, help="覆盖训练轮数")
    p2.add_argument("--device",
                    help="覆盖设备：auto/cpu/mps/cuda:N/cuda:0,1（多卡并行）")
    p2.add_argument("--processed-dir", help="覆盖预处理数据目录")
    p2.add_argument("--scheme", help="覆盖划分方案")

    p3 = sub.add_parser("list", help="列出实验")
    p3.add_argument("keyword", nargs="?", default="")

    p4 = sub.add_parser("report", help="打印实验摘要")
    p4.add_argument("exp_dir")

    p5 = sub.add_parser("fewshot-curve",
                        help="Few-Shot 学习曲线：扫描不同 n_shot 的准确率")
    p5.add_argument("config")
    p5.add_argument("--shots", default="1,5,10,20", help="逗号分隔的 n_shot 列表")
    p5.add_argument("--repeats", type=int, help="覆盖 few_shot 的重复抽样次数")
    p5.add_argument("--device", help="覆盖设备（同 train）")
    p5.add_argument("--processed-dir", help="覆盖预处理数据目录")

    p6 = sub.add_parser("devices", help="列出本机可用训练设备并测速")
    p6.add_argument("--probe", action="store_true",
                    help="对每个设备真跑几步矩阵乘测速（MPS/CUDA 上更有意义）")

    args = ap.parse_args(argv)

    if args.cmd == "devices":
        from .train.devices import device_label, list_devices, probe_device, system_info

        for k, v in system_info().items():
            print(f"  {k:<16} {v}")
        print("\n可用设备（配置里 trainer.device 的写法）：")
        print(f"  {'auto':<12} 自动（推荐：CUDA > MPS > CPU）")
        for d in list_devices():
            mem = f"，{d.memory_gb:g} GB" if d.memory_gb else ""
            print(f"  {d.id:<12} {d.name}{mem}"
                  + (f"　— {d.note}" if d.note else ""))
        if args.probe:
            print("\n测速（512×512 矩阵乘 + 反传）：")
            for spec in ["auto", "cpu"] + [d.id for d in list_devices() if d.kind == "cuda"]:
                r = probe_device(spec)
                if r["ok"]:
                    print(f"  {spec:<12} {r['name']}　{r['ms_per_step']:.2f} ms/步")
                else:
                    print(f"  {spec:<12} ✗ {r['error']}")
        best = device_label("auto")
        print(f"\n当前 auto 会选：{best}")
        return None

    if args.cmd == "pipeline":
        from .pipeline import main as pipe_main

        kw = {k: v for k, v in vars(args).items() if k != "cmd" and v is not None}
        return pipe_main([kw["config"]] + sum(
            [[f"--{k.replace('_', '-')}", str(v)] for k, v in kw.items() if k != "config"], []))

    if args.cmd == "train":
        from .pipeline import load_config
        from .train import run_training

        config = load_config(args.config)
        if args.epochs:
            config.setdefault("trainer", {})["epochs"] = args.epochs
        if args.device:
            config.setdefault("trainer", {})["device"] = args.device
        if args.processed_dir:
            config.setdefault("data", {})["processed_dir"] = args.processed_dir
        if args.scheme:
            config.setdefault("split", {})["scheme"] = args.scheme
        summary = run_training(config, log=print)
        print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
        return summary

    if args.cmd == "list":
        from .experiment import ExperimentManager

        rows = ExperimentManager().list_experiments(args.keyword)
        if not rows:
            print("（暂无实验）")
        for r in rows:
            acc = r["acc_mean"]
            print(f"{r['id']:3d}  {r['ts']}  {r['model']:<14s} acc="
                  f"{(acc or 0) * 100:5.2f}%  {r['dir']}")
        return rows

    if args.cmd == "report":
        from pathlib import Path

        p = Path(args.exp_dir) / "summary.json"
        print(p.read_text(encoding="utf-8"))
        return None

    if args.cmd == "fewshot-curve":
        from .pipeline import load_config
        from .train import run_few_shot_curve

        config = load_config(args.config)
        if args.repeats:
            config.setdefault("split", {})["n_repeats"] = args.repeats
        if args.device:
            config.setdefault("trainer", {})["device"] = args.device
        if args.processed_dir:
            config.setdefault("data", {})["processed_dir"] = args.processed_dir
        shots = [int(x) for x in args.shots.split(",") if x.strip()]
        curve = run_few_shot_curve(config, shots=shots, log=print)
        print("\n" + "=" * 62)
        print(f"{'n_shot':>6} | {'所有折平均':>14} | {'按被试平均':>12} | 折数")
        print("-" * 62)
        for r in curve["results"]:
            psm = r["per_subject_acc_mean"]
            ps = f"{psm * 100:6.2f}%" if psm is not None else "    -  "
            print(f"{r['n_shot']:>6} | {r['acc_mean'] * 100:6.2f}% ± "
                  f"{r['acc_std'] * 100:4.2f} | {ps:>12} | {r['n_folds']}")
        print("=" * 62)
        for f_ in curve.get("failed", []):
            print(f"! n_shot={f_['n_shot']} 未成功：{f_['error']}")
        if curve.get("failed"):
            print("提示：样本数 ≤ 类别数时 LDA 无法拟合；1-shot 四分类请改神经网络模型"
                  "或减少类别数。")
        if curve.get("json_path"):
            print(f"曲线数据 → {curve['json_path']}")
        return curve


if __name__ == "__main__":
    # main() 对 train/list/fewshot-curve 会返回 dict / list（不是错误码），
    # 直接 sys.exit(返回值) 会：① 成功却以退出码 1 结束 ② 把结果打到 stderr。
    _rc = main()
    sys.exit(_rc if isinstance(_rc, int) else 0)
