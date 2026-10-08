"""FBCNet 风格模型（参考 Mane et al., 2021 的精简复现）。

结构：多频带时间卷积（各带不同核长）→ 每带空间卷积 → 时间维方差池化 →
带权 log 变换 → 线性分类层。与原文 FBCSP-style 滤波器组思想一致。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class FBCNet(nn.Module):
    def __init__(self, n_classes: int = 4, n_ch: int = 22, n_times: int = 501,
                 n_bands: int = 6, filters_per_band: int = 8, dropout: float = 0.3):
        super().__init__()
        ks = np.unique(np.geomspace(8, max(16, min(128, n_times // 4)), int(n_bands)).round().astype(int))
        n_bands = len(ks)
        self.temporal = nn.ModuleList(
            [nn.Conv2d(1, filters_per_band, (1, int(k)), padding=(0, int(k) // 2), bias=False)
             for k in ks])
        self.spatial = nn.ModuleList(
            [nn.Conv2d(filters_per_band, filters_per_band, (n_ch, 1), bias=False)
             for _ in ks])
        self.bn = nn.BatchNorm1d(n_bands * filters_per_band)
        self.weight = nn.Parameter(torch.ones(n_bands * filters_per_band))  # FBCNet 的 w-log
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(n_bands * filters_per_band, n_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        feats = []
        for tc, sc in zip(self.temporal, self.spatial):
            f = sc(tc(x))                       # (B, F, 1, T')
            feats.append(f.squeeze(2).var(dim=2))  # 方差池化 → (B, F)
        f = torch.cat(feats, dim=1)
        f = self.bn(f)
        f = self.weight * torch.log(f.clamp_min(1e-8))
        return self.classifier(self.dropout(f))
