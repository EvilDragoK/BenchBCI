"""DeepConvNet (Schirrmeister et al., 2017)。"""
from __future__ import annotations

import torch
import torch.nn as nn


class DeepConvNet(nn.Module):
    def __init__(self, n_classes: int = 4, n_ch: int = 22, n_times: int = 501,
                 dropout: float = 0.5):
        super().__init__()
        def block(cin, cout, k=(1, 10), pool=True, first_spatial=False):
            layers = [nn.Conv2d(cin, cout, k, bias=not first_spatial)]
            if first_spatial:
                layers.append(nn.Conv2d(cout, cout, (n_ch, 1), bias=False))
            layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.ELU())
            if pool:
                layers.append(nn.MaxPool2d((1, 3), stride=(1, 3)))
            layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        self.b0 = block(1, 25, first_spatial=True)   # 时间卷积 + 空间卷积
        self.b1 = block(25, 50)
        self.b2 = block(50, 100)
        self.b3 = block(100, 200)
        # 绝对导入：本文件会被原样复制为自定义模型模板（见 eegnet.py 同处注释）
        from core.models._shape import infer_flat_dim

        feat = infer_flat_dim(nn.Sequential(self.b0, self.b1, self.b2, self.b3),
                              n_ch, n_times, tag="DeepConvNet")
        self.classifier = nn.Linear(feat, n_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.b3(self.b2(self.b1(self.b0(x))))
        return self.classifier(x.flatten(1))
