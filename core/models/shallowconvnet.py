"""ShallowConvNet (Schirrmeister et al., 2017)。"""
from __future__ import annotations

import torch
import torch.nn as nn


class ShallowConvNet(nn.Module):
    def __init__(self, n_classes: int = 4, n_ch: int = 22, n_times: int = 501,
                 n_filters: int = 40, kernel_s: int = 25, pool_s: int = 75,
                 pool_stride: int = 15, dropout: float = 0.5):
        super().__init__()
        self.conv_time = nn.Conv2d(1, n_filters, (1, kernel_s), bias=False)
        self.conv_space = nn.Conv2d(n_filters, n_filters, (n_ch, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.pool = nn.AvgPool2d((1, pool_s), stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)
        # 绝对导入：本文件会被原样复制为自定义模型模板（见 eegnet.py 同处注释）
        from core.models._shape import infer_flat_dim

        feat = infer_flat_dim(
            nn.Sequential(self.conv_time, self.conv_space, self.bn, self.pool),
            n_ch, n_times, tag="ShallowConvNet")
        self.classifier = nn.Linear(feat, n_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.conv_time(x)
        x = self.conv_space(x)
        x = self.bn(x)
        x = self.pool(x ** 2)            # 原文：平方
        x = torch.log(x.clamp_min(1e-8))  # 原文：对数
        x = self.dropout(x.flatten(1))
        return self.classifier(x)
