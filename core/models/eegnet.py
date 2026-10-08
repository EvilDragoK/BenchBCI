"""EEGNet (Lawhern et al., 2018) 的 PyTorch 复现，参数量全可调。

实现细节：卷积一律用**奇数 kernel + 显式对称填充**，而不是 ``padding="same"``。
原因：当 kernel 为偶数且 dilation 为奇数时，PyTorch 的 ``padding="same"`` 需要
额外拷贝一份零填充输入（并给出 UserWarning），且左右填充不对称会让边界语义
偏移；显式对称填充输出长度完全不变，也没有额外开销。
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _odd(k: int) -> int:
    """kernel 长度向上取奇数（偶数时 +1）。"""
    k = int(k)
    return k if k % 2 == 1 else k + 1


def _same_padding(k: int) -> tuple[int, int]:
    """stride=1 时与 "same" 等价的对称填充（要求 k 为奇数）。"""
    return (0, (k - 1) // 2)


class EEGNet(nn.Module):
    def __init__(self, n_classes: int = 4, n_ch: int = 22, n_times: int = 501,
                 F1: int = 8, F2: int = 16, D: int = 2, kernel_length: int = 125,
                 dropout: float = 0.25, kernel_length2: int = 16):
        super().__init__()
        k1, k2 = _odd(kernel_length), _odd(kernel_length2)
        self.n_ch, self.n_times = n_ch, n_times
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=_same_padding(k1), bias=False),
            nn.BatchNorm2d(F1),
            # depthwise（跨通道）卷积，depth multiplier D
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, k2), groups=F1 * D,
                      padding=_same_padding(k2), bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        # 分类头 in_features 自动探测（eval 模式 + 失败时给出可操作提示）。
        # 注意用**绝对导入**：本文件会被 custom.template_source 原样复制成用户的自定义
        # 模型（在 models_custom/ 下以独立模块执行），相对导入在那里会直接 ImportError。
        from core.models._shape import infer_flat_dim

        feat = infer_flat_dim(nn.Sequential(self.block1, self.block2), n_ch, n_times,
                              tag="EEGNet")
        self.classifier = nn.Linear(feat, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:  # (B, C, T) → (B, 1, C, T)
            x = x.unsqueeze(1)
        return self.classifier(self.block2(self.block1(x)).flatten(1))
