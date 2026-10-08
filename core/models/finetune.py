"""微调策略：决定「哪些参数训练、哪些冻结」。

配置（训练 YAML / GUI）：

.. code-block:: yaml

    model:
      freeze: encoder        # none | encoder | all_but_head | custom
      trainable: []          # 白名单（模块名片段，优先级高于 freeze）

策略含义
--------
- ``none``         ：全部参数可训练
- ``encoder``      ：冻结主干/编码器，只训练分类头（= 线性探针 / head_only）
- ``all_but_head`` ：``encoder`` 的别名（对齐 EEG-FM-Benchmark 的叫法）
- ``custom``       ：只训练 ``trainable`` 里列出的模块，其余全部冻结

``trainable`` 的匹配方式是**按参数名分段匹配**：例如 ``trainable: [classifier]``
会命中 ``classifier.weight``，``trainable: [proj_out, task_head]`` 会命中
``main_model.proj_out.weight`` 与 ``task_head.bias``。
"""
from __future__ import annotations

#: 视作"分类头 / 任务头"的模块名片段
HEAD_HINTS = ("classifier", "task_head", "head", "fc", "proj_out", "output",
              "final_layer", "unembedder", "logits")

#: 视作"编码器 / 主干"的模块名片段
ENCODER_HINTS = ("encoder", "backbone", "main_model", "blocks", "layers",
                 "patch_embed", "transformer", "embedder", "module")


def _parts(name: str) -> set[str]:
    return set(name.split("."))


def is_head_param(name: str) -> bool:
    """参数是否属于分类头/任务头。"""
    return bool(_parts(name) & set(HEAD_HINTS))


def is_encoder_param(name: str) -> bool:
    """参数是否属于编码器/主干。"""
    return bool(_parts(name) & set(ENCODER_HINTS))


def decide_keep(name: str, freeze: str, whitelist: list[str]) -> bool:
    """单个参数是否可训练。"""
    if whitelist:
        return bool(_parts(name) & set(whitelist))
    if freeze in ("encoder", "all_but_head"):
        return is_head_param(name)
    if freeze == "none":
        return True
    raise ValueError(
        f"未知冻结策略：{freeze}（可用 none / encoder / all_but_head / custom）")


def apply_finetune_policy(model, freeze: str = "none", trainable=None, log=print) -> dict:
    """按策略设置 ``requires_grad``，返回统计信息（可训练参数量、模块清单等）。"""
    freeze = str(freeze or "none").lower()
    whitelist = [str(x).strip() for x in (trainable or []) if str(x).strip()]
    if whitelist and freeze == "none":
        freeze = "custom"          # 给了白名单就视为自定义策略

    n_train = n_frozen = 0
    train_mods: set[str] = set()
    frozen_mods: set[str] = set()
    for name, param in model.named_parameters():
        keep = decide_keep(name, freeze, whitelist)
        param.requires_grad = bool(keep)
        module = name.rsplit(".", 1)[0]
        if keep:
            n_train += param.numel()
            train_mods.add(module)
        else:
            n_frozen += param.numel()
            frozen_mods.add(module)

    info = {
        "freeze": freeze,
        "n_trainable": n_train,
        "n_frozen": n_frozen,
        "trainable_modules": sorted(train_mods),
        "frozen_modules": sorted(frozen_mods),
    }
    if freeze != "none" or whitelist:
        total = max(n_train + n_frozen, 1)
        log(f"  微调策略 {freeze}"
            + (f"（白名单 {whitelist}）" if whitelist else "")
            + f"：可训练 {n_train:,}/{total:,} 参数（{n_train / total * 100:.1f}%）")
        if train_mods:
            preview = ", ".join(sorted(train_mods)[:6])
            log(f"    可训练模块示例：{preview}"
                + (" …" if len(train_mods) > 6 else ""))
    if n_train == 0:
        raise ValueError("冻结策略把**所有**参数都冻结了，训练将无法进行；"
                         "请检查 model.freeze / model.trainable 配置")
    return info


def param_groups_by_role(model, lr_head: float, lr_encoder: float | None = None,
                         layer_decay: float = 1.0, weight_decay: float = 0.0,
                         log=print) -> list[dict]:
    """把可训练参数分成「分类头」与「编码器」两组，可分别设学习率。

    - 头组用 ``lr_head``；编码器组用 ``lr_encoder``（为 None 时与头组相同）
    - ``layer_decay < 1`` 时对编码器按层号做指数衰减（底层小、高层大），
      并把缩放**直接写进 group["lr"]**（不同于参考实现只写不生效的 lr_scale）
    - 权重衰减只加在 ≥2 维参数上（bias / norm 参数不带 wd）
    """
    lr_encoder = float(lr_head if lr_encoder is None else lr_encoder)
    same_lr = abs(lr_encoder - float(lr_head)) < 1e-12 and layer_decay >= 1.0

    head_decay: list = []
    head_nodecay: list = []
    enc: list[tuple[str, object]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = is_head_param(name) or not is_encoder_param(name)
        if is_head:
            (head_nodecay if param.ndim <= 1 else head_decay).append(param)
        else:
            enc.append((name, param))

    groups: list[dict] = []
    if head_decay:
        groups.append({"params": head_decay, "lr": float(lr_head),
                       "weight_decay": weight_decay, "_role": "head"})
    if head_nodecay:
        groups.append({"params": head_nodecay, "lr": float(lr_head),
                       "weight_decay": 0.0, "_role": "head"})

    if enc:
        if same_lr and layer_decay >= 1.0:
            decay_p = [p for _n, p in enc if p.ndim > 1]
            nodecay_p = [p for _n, p in enc if p.ndim <= 1]
            if decay_p:
                groups.append({"params": decay_p, "lr": lr_encoder,
                               "weight_decay": weight_decay, "_role": "encoder"})
            if nodecay_p:
                groups.append({"params": nodecay_p, "lr": lr_encoder,
                               "weight_decay": 0.0, "_role": "encoder"})
        else:
            # 按「层号」分组以实现逐层衰减
            buckets: dict[int, dict] = {}
            max_layer = 0
            for name, param in enc:
                lid = _layer_id(name)
                max_layer = max(max_layer, lid)
                b = buckets.setdefault(lid, {"decay": [], "nodecay": []})
                b["nodecay" if param.ndim <= 1 else "decay"].append(param)
            for lid, b in sorted(buckets.items()):
                scale = float(layer_decay) ** max(max_layer - lid, 0) if layer_decay < 1.0 else 1.0
                for tag, key in (("decay", "decay"), ("nodecay", "nodecay")):
                    if b[key]:
                        groups.append({
                            "params": b[key],
                            "lr": lr_encoder * scale,
                            "weight_decay": weight_decay if tag == "decay" else 0.0,
                            "_role": "encoder", "_layer": lid, "_scale": scale})
    if not groups:
        raise ValueError("没有可训练参数，无法构建优化器")
    info = ", ".join(
        f"{g['_role']}{f'(layer{g.get('_layer')}×{g.get('_scale', 1):.2f})' if g.get('_layer') is not None else ''}"
        f":lr={g['lr']:.2e},{sum(p.numel() for p in g['params'])}参数" for g in groups)
    log(f"  优化器参数组 → {info}")
    return groups


def _layer_id(name: str) -> int:
    """从参数名里解析层号（blocks.3.* / layers.5.* / encoder.layers.7.*）。"""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p in ("blocks", "layers", "block", "layer") and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                continue
    return 0
