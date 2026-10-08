"""EEG Conformer (Song et al., 2022)：CNN patch 嵌入 + Transformer 编码器。"""
from __future__ import annotations

import torch
import torch.nn as nn


class EEGConformer(nn.Module):
    def __init__(self, n_classes: int = 4, n_ch: int = 22, n_times: int = 501,
                 patch: int = 25, d_model: int = 64, depth: int = 2, heads: int = 4,
                 mlp_ratio: int = 2, dropout: float = 0.25):
        super().__init__()
        self.n_patches = n_times // patch
        self.patch_embed = nn.Conv2d(1, d_model, (1, patch), stride=(1, patch))
        n_tokens = self.n_patches * n_ch
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * mlp_ratio,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_tokens * d_model, n_classes),
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.patch_embed(x)                       # (B, d, C, P)
        x = x.flatten(2).transpose(1, 2)              # (B, C*P, d)
        x = x + self.pos_embed[:, : x.shape[1]]
        x = self.encoder(x)
        x = self.norm(x)
        return self.head(x.flatten(1))
