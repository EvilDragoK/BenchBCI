"""PyTorch 训练循环：AMP、调度器、早停、断点续训、类别加权、增强。"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def pick_device(spec: str = "auto") -> str:
    """把设备写法解析成 torch 设备字符串（详见 core.train.devices）。

    兼容旧接口：``auto`` → CUDA > MPS > XPU > CPU；不可用的设备优雅回退。
    """
    from .devices import resolve_device

    return resolve_device(spec)[0]


class Trainer:
    def __init__(self, model, cfg: dict, ckpt_dir: Path, log=print, stop_event=None):
        import torch
        from torch import nn

        self.torch = torch
        self.model = model
        self.cfg = cfg
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.stop_event = stop_event
        # 设备：解析 "auto / cpu / mps / cuda:0 / cuda:0,1"，多卡自动包 DataParallel
        from .devices import resolve_device, wrap_multi_gpu

        self.device, self.device_ids = resolve_device(cfg.get("device", "auto"))
        self.model = wrap_multi_gpu(model, self.device_ids, self.device)
        self.model.to(self.device)

        self.epochs = int(cfg.get("epochs", 50))
        lr = float(cfg.get("lr", 1e-3))
        wd = float(cfg.get("weight_decay", 0.0))
        lr_encoder = cfg.get("lr_encoder")
        layer_decay = float(cfg.get("layer_decay", 1.0) or 1.0)
        opt_name = str(cfg.get("optimizer", "adamw")).lower()

        # 参数分组：分类头与编码器可分别设学习率（并支持逐层衰减）；
        # 冻结参数（requires_grad=False）不会进入任何一组
        from ..models.finetune import param_groups_by_role

        try:
            groups = param_groups_by_role(model, lr, lr_encoder, layer_decay,
                                          weight_decay=wd, log=self.log)
        except Exception as e:  # noqa: BLE001  分组失败退化为单组，不阻断训练
            self.log(f"  [提示] 参数分组失败（{type(e).__name__}: {e}），改用单一学习率")
            groups = [{"params": [p for p in model.parameters() if p.requires_grad]
                       or list(model.parameters()), "lr": lr, "weight_decay": wd}]
        # 私有字段（_role/_layer/_scale）不能传给优化器
        groups = [{k: v for k, v in g.items() if not k.startswith("_")} for g in groups]

        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(groups, lr=lr)
        elif opt_name == "sgd":
            self.optimizer = torch.optim.SGD(groups, lr=lr, momentum=0.9)
        else:
            self.optimizer = torch.optim.AdamW(groups, lr=lr)

        sched = str(cfg.get("scheduler", "cosine")).lower()
        if sched == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)
        elif sched == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 10, 0.5)
        else:
            self.scheduler = None

        amp = cfg.get("amp", "auto")
        self.use_amp = (torch.cuda.is_available() and self.device.startswith("cuda")
                        and (amp is True or amp == "auto"))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.criterion = nn.CrossEntropyLoss()

        aug = cfg.get("augmentation", {}) or {}
        self.aug = {
            "time_shift": int(aug.get("time_shift", 0) or 0),
            "gaussian_noise": float(aug.get("gaussian_noise", 0.0) or 0.0),
            "channel_dropout": float(aug.get("channel_dropout", 0.0) or 0.0),
        }

    # ------------------------------------------------------------------
    def set_class_weights(self, y_train):
        """类别不平衡 → 加权交叉熵（权重只来自训练折标签）。"""
        y_train = np.asarray(y_train)
        labels = np.unique(y_train)
        counts = np.array([(y_train == l).sum() for l in labels], dtype=np.float64)
        counts = np.maximum(counts, 1.0)
        w = counts.sum() / (len(labels) * counts)
        import torch

        w = torch.as_tensor(w, dtype=torch.float32, device=self.device)
        from torch import nn

        self.criterion = nn.CrossEntropyLoss(weight=w)

    def fit(self, train_loader, val_loader, on_epoch=None):
        best_val_acc, best_epoch, wait = -1.0, -1, 0
        patience = int(self.cfg.get("patience", 10))
        history = []
        for epoch in range(1, self.epochs + 1):
            if self.stop_event is not None and self.stop_event.is_set():
                self.log("收到停止信号，提前结束训练。")
                break
            t0 = time.perf_counter()
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = 0.0, float("nan")
            if val_loader is not None and len(val_loader):
                val_loss, val_acc = self._run_epoch(val_loader, train=False)[:2]
            if self.scheduler is not None:
                self.scheduler.step()
            stats = {
                "epoch": epoch, "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4) if val_loss == val_loss else None,
                "val_acc": round(val_acc, 4) if val_acc == val_acc else None,
                "lr": self.optimizer.param_groups[0]["lr"],
                "seconds": round(time.perf_counter() - t0, 2),
            }
            history.append(stats)
            marker = ""
            if stats["val_acc"] is not None and stats["val_acc"] > best_val_acc:
                best_val_acc, best_epoch, wait = stats["val_acc"], epoch, 0
                self._save_ckpt("ckpt_best.pt", epoch, best_val_acc)
                marker = " *"
            else:
                wait += 1
            self._save_ckpt("ckpt_last.pt", epoch, best_val_acc)
            self.log(f"[epoch {epoch:3d}] train_loss={stats['train_loss']:.4f} "
                     f"train_acc={stats['train_acc']:.3f} "
                     f"val_loss={stats['val_loss']} val_acc={stats['val_acc']}"
                     f" ({stats['seconds']}s){marker}")
            if on_epoch:
                on_epoch(stats)
            if patience and stats["val_acc"] is not None and wait >= patience:
                self.log(f"早停：{patience} 个 epoch 无提升（best epoch {best_epoch}）。")
                break
        return history, best_val_acc

    # ------------------------------------------------------------------
    def _run_epoch(self, loader, train: bool, return_prob: bool = False):
        import torch

        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        y_true, y_pred = [], []
        probs: list = []          # 评估时收集 softmax 概率（供 AUC/Brier/ECE 使用）
        ctx = (torch.autocast("cuda", enabled=self.use_amp)
               if train and self.use_amp else torch.autocast("cpu", enabled=False))
        with ctx:
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if train:
                    from ..datasets import augment_batch

                    x = augment_batch(x, generator=None,
                                      time_shift=self.aug["time_shift"],
                                      noise_std=self.aug["gaussian_noise"],
                                      channel_dropout=self.aug["channel_dropout"])
                    with ctx:
                        out = self.model(x)
                        loss = self.criterion(out, y)
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                else:
                    with torch.no_grad():
                        out = self.model(x)
                        loss = self.criterion(out, y)
                total_loss += float(loss.detach()) * len(y)
                correct += int((out.argmax(1) == y).sum())
                total += len(y)
                y_true.extend(y.tolist())
                y_pred.extend(out.argmax(1).tolist())
                if (not train) and return_prob:
                    probs.append(torch.softmax(out.detach().float(), dim=1).cpu().numpy())
        mean_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        if not train:
            if return_prob and probs:
                return (mean_loss, acc, np.array(y_true), np.array(y_pred),
                        np.concatenate(probs, 0))
            return mean_loss, acc, np.array(y_true), np.array(y_pred)
        return mean_loss, acc

    def train_epoch(self, loader):
        return self._run_epoch(loader, train=True)

    def evaluate(self, loader):
        """测试集评估：返回 (y_true, y_pred, y_prob)。"""
        _, _, yt, yp, prob = self._run_epoch(loader, train=False, return_prob=True)
        return yt, yp, prob

    def load_ckpt(self, name="ckpt_best.pt"):
        import torch

        from .devices import adapt_state_dict

        path = self.ckpt_dir / name
        sd = torch.load(path, map_location=self.device, weights_only=False)
        # 权重文件不带 module. 前缀（与设备无关），这里按当前是否多卡自适应
        self.model.load_state_dict(adapt_state_dict(sd["model"], self.model))
        return sd

    def _save_ckpt(self, name, epoch, best_val_acc):
        import random

        import numpy as np

        from .devices import unwrap_state_dict

        torch = self.torch
        state = {
            # 去掉 DataParallel 的 module. 前缀：多卡训的权重也能单卡/CPU 载入
            "model": unwrap_state_dict(self.model.state_dict()),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        torch.save(state, self.ckpt_dir / name)
