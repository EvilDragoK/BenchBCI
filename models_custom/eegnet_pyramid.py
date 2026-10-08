from __future__ import annotations

"""自定义模型「eegnet_pyramid」——基于 eegnet（EEGNet）改写。

在 block1/block2 之间加入金字塔（多尺度）池化特征

要点（改完点「检查」会按当前数据的维度真跑一遍）：
- 构造函数请保留 n_classes / n_ch / n_times 三个维度参数；
- 层的输出形状必须能一路走到分类头，最后输出 (B, n_classes)；
- 下面用 forward hook 逐层核对输入/输出形状，维度对不上会直接报出来。
"""

# 对外暴露的模型类（改名时请同步修改）
__model__ = "eegnet_pyramid"

import torch
import torch.nn as nn


def _odd(k: int) -> int:
    """kernel 长度向上取奇数（偶数时 +1）。"""
    k = int(k)
    return k if k % 2 == 1 else k + 1


def _same_padding(k: int) -> tuple[int, int]:
    """stride=1 时与 "same" 等价的对称填充（要求 k 为奇数）。"""
    return (0, (k - 1) // 2)



class PyramidPool(nn.Module):
    """金字塔（多尺度）池化：多尺度时间池化后展平拼接。"""

    def __init__(self, scales=(1, 2, 4)):
        super().__init__()
        self.scales = tuple(int(s) for s in scales)

    def forward(self, x):
        levels = [nn.functional.adaptive_avg_pool2d(x, (1, s)) for s in self.scales]
        return torch.cat([lv.flatten(1) for lv in levels], dim=1)

class eegnet_pyramid(nn.Module):
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
        # 金字塔特征：对 block1 输出做多尺度池化（1/2/4 点）
        self.pyramid = PyramidPool((1, 2, 4))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, k2), groups=F1 * D,
                      padding=_same_padding(k2), bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            f1 = self.block1(torch.zeros(1, 1, n_ch, n_times))
            feat = self.pyramid(f1).shape[1] + self.block2(f1).flatten(1).shape[1]
        self.classifier = nn.Linear(feat, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:  # (B, C, T) → (B, 1, C, T)
            x = x.unsqueeze(1)
        f1 = self.block1(x)
        pyramid = self.pyramid(f1)          # (B, C×Σscales) 金字塔特征
        deep = self.block2(f1).flatten(1)   # (B, F2) block2 深度特征
        return self.classifier(torch.cat([pyramid, deep], dim=1))

