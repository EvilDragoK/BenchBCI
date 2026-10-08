"""训练入口：run_training(config) —— GUI 与命令行共用同一条路径，保证一致。"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import PROJECT_ROOT
from ..datasets import (apply_normalization, build_dataloader,
                        compute_norm_stats, euclidean_alignment,
                        set_global_seed)
from ..experiment import ExperimentManager
from ..io.loaders import load_processed
from ..models import build_model, is_classic
from ..pipeline import resolve_path
from .evaluation import (aggregate, compute_metrics, export_markdown_report,
                         per_subject_summary)
from .splits import make_splits
from .trainer import Trainer

DEFAULTS = {
    "experiment": {"name": "run", "note": ""},
    "data": {"processed_dir": "processed/bnci2014001"},
    "model": {"name": "eegnet", "params": {},
              # 预训练权重：None=从零训练 / auto=按注册表自动下载 /
              # hf:<repo>[/file] / url:<...> / 本地路径
              "pretrained": None,
              # 微调策略：none | encoder(≈all_but_head) | custom（配合 trainable 白名单）
              "freeze": "none", "trainable": []},
    "split": {"scheme": "session", "k": 5, "val_frac": 0.15, "seed": 42,
              "test_session": "all",   # loso / few_shot：测试数据用哪个 session
              "n_shot": 5,             # few_shot：每类训练样本数
              # within / few_shot：重复随机采样次数。1 = 每名被试一折（共 N 折，
              # 与 EEG-FM-Benchmark 等主流基准一致）；>1 则折数 = 被试数 × n_repeats
              "n_repeats": 1,
              # within / few_shot：按「每被试每类」比例切训练集（对齐 EEG-FM-Benchmark
              # 的 --train_percentage，默认 0.3）；为 None 时改用 n_shot 或 1/k
              "train_percentage": None},
    "normalization": {"mode": "zscore_channel", "euclidean_alignment": False},
    "augmentation": {"time_shift": 0, "gaussian_noise": 0.0, "channel_dropout": 0.0},
    "trainer": {"epochs": 60, "batch_size": 64, "lr": 1e-3, "optimizer": "adamw",
                "weight_decay": 0.01, "scheduler": "cosine", "patience": 12,
                # 编码器（预训练主干）单独的学习率；None = 与 lr 相同。
                # layer_decay < 1 时对编码器按层做指数衰减（底层小、高层大）
                "lr_encoder": None, "layer_decay": 1.0,
                "amp": "auto", "seed": 42, "deterministic": False,
                "device": "auto", "num_workers": 0},
    "class_balance": {"weighted_loss": True},
    # ITR（信息传输率）需要"一次决策占用的秒数"：BNCI2014-001 一个 trial 约 8 s
    "evaluation": {"trial_seconds": 8.0},
}


def _apply_pretrained(model, cfg: dict, log) -> None:
    """加载预训练权重（支持自动下载）并应用微调策略（冻结 / 可训练白名单）。

    - ``model.pretrained``：``auto``（按基础模型注册表下载）/ ``hf:<repo>`` /
      ``hf:<repo>/<file>`` / ``url:<...>`` / 本地文件或目录
    - ``model.freeze`` / ``model.trainable``：见 ``core.models.finetune``
    - 兼容旧写法 ``model.finetune: full | head_only``
    """
    import torch

    from ..models.foundation import get_spec, is_foundation, resolve_weights
    from ..models.finetune import apply_finetune_policy

    name = str(cfg["model"].get("name", "")).lower()
    spec = get_spec(name) if is_foundation(name) else None
    ref = cfg["model"].get("pretrained")

    if ref:
        path = resolve_weights(ref, spec=spec, log=log)     # 必要时自动下载
        if path is not None:
            if path.is_dir():                               # 整仓下载 → 找权重文件
                files = sorted(p for p in path.rglob("*")
                               if p.suffix in (".safetensors", ".bin", ".pt", ".pth"))
                if not files:
                    raise FileNotFoundError(f"该目录里没有找到权重文件：{path}")
                preferred = [p for p in files if "pretrain" in p.name.lower()]
                path = (preferred or files)[0]
            size_mb = path.stat().st_size / 1e6
            try:
                if path.suffix == ".safetensors":
                    from safetensors.torch import load_file

                    state = load_file(str(path))
                else:
                    state = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(state, dict):
                    state = (state.get("model") or state.get("state_dict")
                             or state.get("model_state_dict") or state)
                missing, unexpected = model.load_state_dict(state, strict=False)
                hit = max(len(state) - len(unexpected), 0)
                # 命中率过低 → 模型可能被包了一层（backbone./main_model. 前缀）
                if state and hit < 0.2 * len(state):
                    for prefix in ("backbone.", "main_model.", "encoder."):
                        alt = {f"{prefix}{k}": v for k, v in state.items()}
                        m2, u2 = model.load_state_dict(alt, strict=False)
                        hit2 = max(len(alt) - len(u2), 0)
                        if hit2 > hit:
                            missing, unexpected, state, hit = m2, u2, alt, hit2
                            log(f"    （按前缀 '{prefix}' 对齐了权重键名）")
                            break
                log(f"  已加载权重 {Path(path).name}（{size_mb:.1f} MB）："
                    f"命中 {hit}/{len(state)} 项，未命中 {len(missing)} 项"
                    f"（通常是新增的分类头）")
            except Exception as e:  # noqa: BLE001
                log(f"  [警告] 权重加载失败（{type(e).__name__}: {e}），本次从零训练")

    # 微调策略（必须在加载权重之后应用）
    legacy = str(cfg["model"].get("finetune", "") or "").lower()
    freeze = cfg["model"].get("freeze", "none")
    if legacy == "head_only":
        freeze = "encoder"
    elif legacy == "full" and not cfg["model"].get("freeze"):
        freeze = "none"
    apply_finetune_policy(model, freeze, cfg["model"].get("trainable"), log=log)


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _nested_select_params(cfg: dict, Xtr, ytr, n_classes: int, exp, log) -> dict:
    """嵌套 CV 的内层：在**训练折内部**再切 inner_k 折，挑出验证最优的超参组合。

    这样做的好处是超参不是"在测试集上试出来的"，因此外层指标无偏（这是嵌套
    交叉验证的意义）。候选组合来自 ``split.param_grid``，例如::

        split:
          scheme: nested_cv
          k: 5                 # 外层折
          inner_k: 3           # 内层折
          param_grid: {n_components: [4, 6, 8]}      # 经典基线
          # param_grid: {F1: [8, 16], dropout: [0.25, 0.5]}   # 神经网络
    """
    import itertools

    grid = cfg["split"].get("param_grid") or {}
    if not grid:
        log("  [提示] nested_cv 未提供 split.param_grid，跳过内层选择（等价于 kfold）")
        return {}
    from sklearn.model_selection import StratifiedKFold

    inner_k = max(int(cfg["split"].get("inner_k", 3)), 2)
    seed = int(cfg["split"].get("seed", 42))
    model_name = cfg["model"]["name"]
    base = dict(cfg["model"].get("params") or {})
    keys = list(grid)
    combos = [dict(zip(keys, vals))
              for vals in itertools.product(*[grid[k] for k in keys])]
    inner_epochs = int(cfg["split"].get(
        "inner_epochs", min(int(cfg["trainer"].get("epochs", 20)), 20)))
    folds = list(StratifiedKFold(n_splits=inner_k, shuffle=True,
                                 random_state=seed).split(np.zeros(len(ytr)), ytr))
    best_params, best_score = {}, -1.0
    for combo in combos:
        params = {**base, **combo}
        scores: list[float] = []
        for a, b in folds:
            try:
                if is_classic(model_name):
                    clf = build_model(model_name, n_classes, Xtr.shape[1],
                                      Xtr.shape[2], params)
                    clf.fit(Xtr[a], ytr[a])
                    scores.append(float((clf.predict(Xtr[b]) == ytr[b]).mean()))
                else:
                    from ..datasets import build_dataloader
                    from .trainer import Trainer

                    model = build_model(model_name, n_classes, Xtr.shape[1],
                                        Xtr.shape[2], params)
                    trainer = Trainer(model,
                                      {**cfg["trainer"], "epochs": inner_epochs,
                                       "augmentation": cfg["augmentation"]},
                                      exp.ckpt_dir / "inner", log=lambda *_: None)
                    trainer.fit(build_dataloader(
                        Xtr[a], ytr[a], int(cfg["trainer"].get("batch_size", 64)),
                        shuffle=True, seed=seed, num_workers=0), None)
                    yt, yp, _ = trainer.evaluate(build_dataloader(
                        Xtr[b], ytr[b], int(cfg["trainer"].get("batch_size", 64)),
                        shuffle=False))
                    scores.append(float((np.asarray(yt) == np.asarray(yp)).mean()))
            except Exception as e:  # noqa: BLE001
                log(f"    内层 {combo} 有一折失败（{type(e).__name__}: {e}），跳过")
        mean_acc = float(np.mean(scores)) if scores else 0.0
        log(f"    内层评估 {combo} → 平均 acc {mean_acc:.4f}（{len(scores)} 折）")
        if mean_acc > best_score:
            best_score, best_params = mean_acc, dict(combo)
    return best_params


def _probe_device(model, X_sample, tr_cfg: dict, log) -> dict:
    """设备自检：用一个小 batch 试跑，若不支持则自动回退 CPU。

    典型场景：Apple MPS 目前不支持带 dropout 的
    ``scaled_dot_product_attention``，很多基础模型（CBraMod 等）的注意力层会命中，
    直接跑会在训练中途抛 NotImplementedError。
    """
    import torch

    from .devices import device_label, resolve_device

    device, _ids = resolve_device(tr_cfg.get("device", "auto"))
    log(f"  训练设备：{device_label(tr_cfg.get('device', 'auto'))}")
    if device == "cpu":
        return tr_cfg
    try:
        n = min(4, len(X_sample))
        probe = torch.as_tensor(X_sample[:n], dtype=torch.float32)
        model.to(device)
        model.train()
        with torch.no_grad():
            model(probe.to(device))
        model.to("cpu")
        return tr_cfg
    except Exception as e:  # noqa: BLE001
        log(f"  [提示] 设备 {device} 跑不通该模型（{type(e).__name__}: "
            f"{str(e).splitlines()[0][:70]}）→ 自动改用 CPU")
        try:
            model.to("cpu")
        except Exception:  # noqa: BLE001
            pass
        return {**tr_cfg, "device": "cpu"}


def _drop_stale_best(ckpt_dir) -> bool:
    """折开始前删掉上一折的 ``ckpt_best.pt``（避免跨折误用）。

    所有折共用同一个 ``exp.ckpt_dir``，而"本折没有验证集"时永远不会写新 best
    （val_acc 恒为 nan）——若不删，下一折就会载入**上一折**的最佳权重并据此评估
    测试集（静默用错权重）。
    """
    p = Path(ckpt_dir) / "ckpt_best.pt"
    if not p.exists():
        return False
    try:
        p.unlink()
        return True
    except OSError:  # pragma: no cover - 只读挂载等极端情况
        return False


def _restore_best_if_any(trainer, best_val, best_ckpt) -> bool:
    """按需载回本折最佳权重：仅当本折确实选出过 best 且文件存在。

    ``best_val`` 为 ``-1.0``（无验证集，Trainer.fit 的初值）或 None 时不载入。
    """
    best_val = -1.0 if best_val is None else float(best_val)
    p = Path(best_ckpt)
    if best_val > -1.0 and p.exists():
        trainer.load_ckpt("ckpt_best.pt")
        return True
    return False


def run_training(config: dict, log=print, on_epoch=None, stop_event=None) -> dict:
    """执行完整训练（多折）并把一切写入实验目录。

    Parameters
    ----------
    on_epoch : callable(fold_tag, stats_dict) —— GUI 实时曲线回调（在工作线程中调用）。
    stop_event : threading.Event —— 置位后训练在 epoch 边界提前停止。

    Returns
    -------
    summary dict（同时写入 experiments/<...>/summary.json）。
    """
    cfg = _merge(DEFAULTS, config)
    data_dir = resolve_path(cfg["data"]["processed_dir"])
    log(f"加载预处理数据：{data_dir}")
    data = load_processed(data_dir)
    X, y, sid, sess = data["X"], data["y"], data["sid"], data["sess"]
    # ---- 标签重编码：原始类别码（如 BNCI 的 1..4）→ 连续索引 0..K-1 ----
    # 模型与 CrossEntropyLoss 要求目标 ∈ [0, K)：直接用 1..4 时 CPU 会报
    # "Target 4 is out of bounds"，而 MPS 上会静默越界，导致指标失真
    # （典型症状：混淆矩阵最后一列恒为 0）。
    codes = sorted(int(v) for v in np.unique(y))
    inv = {int(v): k for k, v in (data.get("classes") or {}).items()}
    class_names = [inv.get(c, f"class_{c}") for c in codes]
    n_classes = len(class_names)
    if codes != list(range(n_classes)):
        y = np.searchsorted(codes, y).astype(np.int64)
        log(f"  标签重编码：{codes} → 0..{n_classes - 1}（模型输出索引）")
    log(f"  {X.shape[0]} trials × {X.shape[1]} ch × {X.shape[2]} samples, "
        f"{n_classes} 类 {class_names}")

    block = data.get("block")
    # ---- 跨数据集迁移：再加载一份目标域数据，与源域首尾拼接 ----
    domain_sizes = None
    scheme_name = str(cfg["split"].get("scheme", "session"))
    if scheme_name == "cross_dataset":
        test_dir_cfg = cfg["data"].get("test_processed_dir")
        if not test_dir_cfg:
            raise ValueError(
                "cross_dataset 需要在 data.test_processed_dir 指定目标域（测试数据集）"
                "的预处理输出目录")
        test_dir = resolve_path(test_dir_cfg)
        log(f"加载目标域数据（跨数据集迁移）：{test_dir}")
        data_b = load_processed(test_dir)
        Xb, yb = data_b["X"], data_b["y"]
        if Xb.shape[1:] != X.shape[1:]:
            raise ValueError(
                f"两个数据集的通道/采样点不一致：{X.shape[1:]} vs {Xb.shape[1:]}"
                f"（需保证通道数与 epoch 时长相同）")
        codes_b = sorted(int(v) for v in np.unique(yb))
        if not set(codes_b) <= set(codes):
            raise ValueError(f"目标域类别码 {codes_b} 与源域 {codes} 无法对齐")
        # 源域此前已按 codes 重编码为 0..K-1；目标域必须做**同样**的映射，
        # 否则目标域标签停留在原始码（如 1..4），与预测（0..3）系统性错位。
        yb = np.searchsorted(codes, yb).astype(np.int64)
        n_a = len(y)
        X = np.concatenate([X, Xb], axis=0)
        y = np.concatenate([y, yb], axis=0)
        # 目标域被试号加偏移，避免与源域混淆（block 同理）
        sid = np.concatenate([sid, np.asarray(data_b["sid"], dtype=np.int64) + 1000])
        sess = np.concatenate([sess, np.asarray(data_b["sess"])])
        blk_b = data_b.get("block")
        if block is not None and blk_b is not None:
            block = np.concatenate([block, np.asarray(blk_b, dtype=np.int64)
                                    + int(np.max(block)) + 1])
        domain_sizes = (n_a, len(yb))
        log(f"  训练域 {n_a} trials → 测试域 {len(yb)} trials（标签/被试号已对齐）")

    split_cfg = cfg["split"]
    seed = int(split_cfg.get("seed", 42))
    set_global_seed(seed, bool(cfg["trainer"].get("deterministic", False)))
    splits = make_splits(y, sid, sess, scheme=split_cfg["scheme"],
                         k=int(split_cfg.get("k", 5)),
                         val_frac=float(split_cfg.get("val_frac", 0.15)),
                         seed=seed,
                         test_session=str(split_cfg.get("test_session", "all")),
                         n_shot=int(split_cfg.get("n_shot", 5)),
                         n_repeats=int(split_cfg.get("n_repeats", 1)),
                         train_percentage=split_cfg.get("train_percentage"),
                         custom_folds=split_cfg.get("folds"),
                         block=block,
                         domain_sizes=domain_sizes)
    scheme = split_cfg["scheme"]
    if scheme == "loso":
        log(f"划分方案 LOSO（跨被试泛化）：{len(splits)} 折，"
            f"每折留出 1 名被试（test_session={split_cfg.get('test_session', 'all')}）")
    elif scheme == "few_shot":
        log(f"划分方案 Within-Subject Few-Shot：每被试每类 "
            f"{split_cfg.get('n_shot', 5)} 个训练样本 × "
            f"{split_cfg.get('n_repeats', 10)} 次重复抽样 → {len(splits)} 折")
    else:
        log(f"划分方案 {scheme}：{len(splits)} 折")

    manager = ExperimentManager()
    exp = manager.create(cfg["experiment"]["name"], config)
    exp.log(f"训练开始 {datetime.now().isoformat(timespec='seconds')}")

    model_name = cfg["model"]["name"]
    t_start = time.perf_counter()

    # ---- 欧几里得对齐（EA）：逐被试、无标签，与折无关 → 只做一次 ----
    # 必须**按被试分别**对齐（euclidean_alignment 用整段输入的协方差均值，
    # 一次性喂全体被试等于把它们对齐到同一个"总体协方差"，被试间差异并未消除）
    if cfg["normalization"].get("euclidean_alignment"):
        log("应用欧几里得对齐（逐被试、无标签、与折无关，只做一次）…")
        for _s in sorted(set(sid.tolist())):
            _m = sid == _s
            X[_m] = euclidean_alignment(X[_m])
        log(f"  EA 完成（{len(np.unique(sid))} 名被试各自对齐到单位阵）")

    fold_results, metric_rows = [], []
    for sp in splits:
        if stop_event is not None and stop_event.is_set():
            log("停止：不再开始新的折。")
            break
        log(f"— 折 {sp.fold}（{sp.tag}）：train={len(sp.train)} val={len(sp.val)} "
            f"test={len(sp.test)}")
        tr, va, te = sp.train, sp.val, sp.test

        # ---- 标准化：统计量只来自训练折（防泄漏）----
        norm_cfg = cfg["normalization"]
        mode = norm_cfg.get("mode", "zscore_channel")
        stats = compute_norm_stats(X[tr], mode)  # 训练折统计量；epoch 级模式返回 {}
        Xtr = apply_normalization(X[tr], mode, stats)
        Xva = apply_normalization(X[va], mode, stats) if len(va) else None
        Xte = apply_normalization(X[te], mode, stats)

        # ---- 嵌套 CV：在训练折内部再切 inner_k 折来选超参（不看测试折）----
        model_params = dict(cfg["model"].get("params") or {})
        if scheme == "nested_cv":
            picked = _nested_select_params(cfg, Xtr, y[tr], n_classes, exp, log)
            if picked:
                log(f"  内层 CV 选中超参：{picked}")
                model_params.update(picked)

        if is_classic(model_name):
            if cfg["model"].get("pretrained"):
                log("  [提示] 经典基线（CSP/LDA 等）不支持加载预训练权重，已忽略该配置")
            clf = build_model(model_name, n_classes, X.shape[1], X.shape[2], model_params)
            t0 = time.perf_counter()
            clf.fit(Xtr, y[tr])
            fit_s = time.perf_counter() - t0
            yp = clf.predict(Xte)
            prob = None
            try:                       # 概率用于 AUC / Brier / ECE（个别基线没有）
                prob = clf.predict_proba(Xte)
            except Exception as e:  # noqa: BLE001
                if sp.fold == 0:
                    log(f"  [提示] 该基线未提供预测概率（{type(e).__name__}），"
                        f"AUC / Brier / ECE 将留空")
            m = compute_metrics(y[te], yp, class_names, labels=list(range(n_classes)),
                                y_prob=prob)
            m["best_val_acc"] = None
            extra = f" AUC={m['auc_macro']:.4f}" if m["auc_macro"] is not None else ""
            log(f"  {model_name} 拟合 {fit_s:.1f}s → 测试 acc={m['acc']:.4f} "
                f"kappa={m['kappa']:.3f} balanced={m['balanced_acc']:.4f}{extra}")
            exp.log(f"fold {sp.fold} ({sp.tag}) acc={m['acc']:.4f} kappa={m['kappa']:.3f}")
        else:
            import torch

            from ..datasets import EpochTensorDataset  # noqa: F401

            tr_cfg = dict(cfg["trainer"])
            from ..models.foundation import build_foundation_model, is_foundation

            if is_foundation(model_name):
                # EEG 基础模型（LaBraM / CBraMod / EEGPT / BIOT …）：走 braindecode
                model = build_foundation_model(
                    model_name, n_classes, X.shape[1], X.shape[2],
                    sfreq=float(data.get("sfreq") or 250.0),
                    params=model_params, ch_names=data.get("channel_names"), log=log)
            else:
                model = build_model(model_name, n_classes, X.shape[1], X.shape[2],
                                    model_params)
            _apply_pretrained(model, cfg, log)      # 预训练权重 + 微调策略
            tr_cfg = _probe_device(model, Xtr, tr_cfg, log)   # 设备自检（MPS 等）
            n_params = sum(p.numel() for p in model.parameters())
            log(f"  模型 {model_name}（{n_params / 1e3:.1f}k 参数）")
            bs = int(tr_cfg.get("batch_size", 64))
            nw = int(tr_cfg.get("num_workers", 0))
            train_loader = build_dataloader(Xtr, y[tr], bs, shuffle=True, seed=seed, num_workers=nw)
            val_loader = (build_dataloader(Xva, y[va], bs, shuffle=False, num_workers=nw)
                          if Xva is not None else None)
            test_loader = build_dataloader(Xte, y[te], bs, shuffle=False, num_workers=nw)

            trainer = Trainer(model, {**tr_cfg, "augmentation": cfg["augmentation"]},
                              exp.ckpt_dir, log=log, stop_event=stop_event)
            if cfg["class_balance"].get("weighted_loss", True):
                trainer.set_class_weights(y[tr])
            from torch.utils.data import DataLoader  # noqa: F401

            # 折与折共用同一个 ckpt 目录：先把上一折的 best 删掉，避免"本折没有
            # 验证集（val_acc 恒为 nan → 从不保存 best）"时误载上一折的最佳权重。
            _drop_stale_best(exp.ckpt_dir)

            history, best_val = trainer.fit(
                train_loader, val_loader,
                on_epoch=(lambda st, tag=sp.tag: on_epoch(tag, st)) if on_epoch else None)
            for st in history:
                metric_rows.append({"fold": sp.fold, "tag": sp.tag, **st})
            _restore_best_if_any(trainer, best_val, exp.ckpt_dir / "ckpt_best.pt")
            yt, yp, prob = trainer.evaluate(test_loader)
            m = compute_metrics(yt, yp, class_names, labels=list(range(n_classes)),
                                y_prob=prob)
            m["best_val_acc"] = best_val
            log(f"  测试 acc={m['acc']:.4f} kappa={m['kappa']:.3f} "
                f"(best_val={best_val})")
            exp.log(f"fold {sp.fold} ({sp.tag}) acc={m['acc']:.4f} kappa={m['kappa']:.3f}")

        fold_results.append({"fold": sp.fold, "tag": sp.tag,
                             "subject": int(getattr(sp, "subject", -1)), **m})

    summary = {
        "experiment": exp.dir.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "scheme": split_cfg["scheme"],
        "data_dir": str(data_dir),
        "n_trials": int(len(y)),
        "n_classes": n_classes,
        "class_names": class_names,
        "folds": fold_results,
        "aggregate": aggregate(
            fold_results, n_classes=n_classes,
            trial_seconds=(cfg.get("evaluation") or {}).get("trial_seconds")),
        "per_subject": per_subject_summary(fold_results),
        "total_seconds": round(time.perf_counter() - t_start, 1),
        "exp_dir": str(exp.dir),
    }
    exp.save_metrics(metric_rows)
    exp.save_summary(summary)
    report = export_markdown_report(exp.dir, summary, config)
    note = str(cfg["experiment"].get("note", ""))
    prov = (cfg.get("paper_params") or {})           # 论文抽取来源 → 记进实验列表备注
    if prov:
        try:
            from ..paper_text import summarize_provenance

            src = summarize_provenance(prov)
            if src:
                note = (note + " · " if note else "") + f"参数来源: {src}"
        except Exception:  # noqa: BLE001
            pass
    manager.register(exp, name=exp.dir.name, dataset=str(data_dir), model=model_name,
                     scheme=split_cfg["scheme"], n_trials=len(y),
                     acc_mean=summary["aggregate"]["acc_mean"],
                     acc_std=summary["aggregate"]["acc_std"],
                     kappa_mean=summary["aggregate"]["kappa_mean"],
                     note=note)
    agg = summary["aggregate"]

    def _txt(key, pct=False, nd=3):
        m, s = agg.get(f"{key}_mean"), agg.get(f"{key}_std")
        if m is None:
            return "—"
        fmt = (lambda v: f"{v * 100:.2f}") if pct else (lambda v: f"{v:.{nd}f}")
        return f"{fmt(m)} ± {fmt(s or 0)}"

    log(f"完成：acc = {_txt('acc', pct=True)} %，kappa = {_txt('kappa')}，"
        f"balanced = {_txt('balanced_acc', pct=True)} %，MCC = {_txt('mcc')}")
    log(f"      F1(macro) = {_txt('f1_macro')}，specificity = {_txt('specificity_macro')}，"
        f"AUC = {_txt('auc_macro')}")
    if agg.get("itr_mean") is not None:
        log(f"      ITR = {agg['itr_mean']:.2f} ± {agg.get('itr_std') or 0:.2f} bits/min")
    log(f"      报告 → {report}")
    return summary


# 延迟到包初始化末尾再导入（curve 内部按需导入 run_training，避免循环依赖）
from .curve import plot_few_shot_curve, run_few_shot_curve  # noqa: E402
