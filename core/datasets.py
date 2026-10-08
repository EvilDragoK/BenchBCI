"""张量数据集、标准化（防泄漏）、欧几里得对齐与数据增强。"""
from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int, deterministic: bool = False):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:  # 某些算子不支持确定性实现
            pass


class EpochTensorDataset:
    """X: (N, C, T) float32；y: (N,) int64。轻量 Dataset。"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        import torch

        self.X = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y, dtype=np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def build_dataloader(X, y, batch_size: int, shuffle: bool, seed: int = 0, num_workers: int = 0):
    import torch
    from torch.utils.data import DataLoader

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(EpochTensorDataset(X, y), batch_size=batch_size,
                      shuffle=shuffle, generator=g if shuffle else None,
                      num_workers=num_workers)


# ---------------------------------------------------------------- 标准化
def compute_channel_stats(X: np.ndarray):
    """逐通道均值/方差，仅应由训练折调用（防数据泄漏）。"""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=(0, 2), keepdims=True)            # (1, C, 1)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


# 训练归一化模式（对应文献 Filtering/Alignment 对照表中的 Alignment 列）
NORM_MODES = [
    "zscore_channel",    # z-score（训练折逐通道统计）→ N(0,1)
    "zscore_epoch",      # Instance Norm：逐 epoch 逐通道
    "minmax_channel",    # 训练折逐通道 min-max → [0,1]
    "minmax_pm",         # 训练折逐通道 min-max → [-1,1]
    "minmax_epoch",      # 逐 epoch → [0,1]
    "minmax_pm_epoch",   # 逐 epoch → [-1,1]
    "p95",               # x / P95（训练折逐通道 |x| 的 95 百分位）
    "x_div",             # x / 常数（stats['constant']，默认 100）
    "iqr",               # (x-中位数)/IQR（训练折逐通道，鲁棒 z-score）
    "quantile",          # 秩变换 → 标准正态分位数（逐 epoch 逐通道）
    "alaw",              # A 律压扩（逐 epoch 逐通道，u=epoch 内 max|x|）
    "ema",               # 指数滑动平均在线标准化（沿时间维）
    "zero_mean",         # 仅去均值（逐 epoch 逐通道）
    "none",
]
# 这些模式的统计量来自训练折；其余模式逐 epoch/无参数，无需统计量
_TRAIN_STATS_MODES = {"zscore_channel", "minmax_channel", "minmax_pm",
                      "p95", "x_div", "iqr"}


def compute_norm_stats(X: np.ndarray, mode: str, constant: float = 100.0,
                       p: float = 95.0) -> dict:
    """按模式计算训练折统计量（仅应由训练折调用，防数据泄漏）。"""
    X = np.asarray(X, dtype=np.float64)
    stats: dict = {"mode": mode}
    if mode == "zscore_channel":
        stats["mean"] = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        stats["std"] = (X.std(axis=(0, 2), keepdims=True) + 1e-8).astype(np.float32)
    elif mode in ("minmax_channel", "minmax_pm"):
        stats["lo"] = X.min(axis=(0, 2), keepdims=True).astype(np.float32)
        stats["hi"] = X.max(axis=(0, 2), keepdims=True).astype(np.float32)
    elif mode == "p95":
        stats["p"] = np.percentile(np.abs(X), p, axis=(0, 2), keepdims=True).astype(np.float32) + 1e-8
    elif mode == "iqr":
        stats["med"] = np.median(X, axis=(0, 2), keepdims=True).astype(np.float32)
        q75, q25 = np.percentile(X, [75, 25], axis=(0, 2), keepdims=True)
        stats["iqr"] = (q75 - q25 + 1e-8).astype(np.float32)
    elif mode == "x_div":
        stats["constant"] = float(constant)
    return stats


def _rank_normal(x: np.ndarray) -> np.ndarray:
    """秩变换 → 标准正态分位数（沿最后一维）。"""
    from scipy.stats import norm

    T = x.shape[-1]
    order = np.argsort(x, axis=-1)
    ranks = np.empty_like(order, dtype=np.float64)
    np.put_along_axis(ranks, order, np.arange(1, T + 1, dtype=np.float64), axis=-1)
    return norm.ppf((ranks - 0.5) / T)


def _ema_normalize(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """指数滑动平均在线标准化（沿最后一维时间轴）。"""
    from scipy.signal import lfilter

    alpha = min(max(alpha, 1e-4), 1.0)
    b, a = [alpha], [1.0, -(1.0 - alpha)]
    x0 = x[..., :1]
    mu = lfilter(b, a, x - x0, axis=-1) + x0
    ex2 = lfilter(b, a, (x - x0) ** 2, axis=-1) + x0 ** 2
    var = np.clip(ex2 - mu ** 2, 0.0, None)
    return (x - mu) / np.sqrt(var + 1e-12)


def apply_normalization(X: np.ndarray, mode: str, stats: dict | None = None,
                        mean=None, std=None) -> np.ndarray:
    """按模式归一化。

    stats 由 `compute_norm_stats(X_train, mode)` 生成（只用训练折统计量，防泄漏）；
    epoch 级模式（*_epoch / quantile / alaw / ema / zero_mean）无需跨样本统计量。
    兼容旧签名 apply_normalization(X, 'zscore_channel', mean, std)。
    """
    X = np.asarray(X, dtype=np.float32)
    eps = 1e-8
    if stats is not None and not isinstance(stats, dict):
        # 兼容旧位置调用 apply_normalization(X, mode, mean, std)
        stats = {"mean": stats}
    if stats is None and (mean is not None or std is not None):
        stats = {"mean": mean, "std": std}

    if mode == "zscore_channel":
        if not stats or "mean" not in stats:
            stats = compute_norm_stats(X, mode)
        return (X - stats["mean"]) / stats["std"]
    if mode == "zscore_epoch":  # Instance Norm
        mu = X.mean(axis=2, keepdims=True)
        return (X - mu) / (X.std(axis=2, keepdims=True) + eps)
    if mode == "minmax_channel":
        if not stats or "lo" not in stats:
            stats = compute_norm_stats(X, mode)
        return (X - stats["lo"]) / (stats["hi"] - stats["lo"] + eps)   # 常数通道不除零
    if mode == "minmax_pm":
        if not stats or "lo" not in stats:
            stats = compute_norm_stats(X, mode)
        return (X - stats["lo"]) / (stats["hi"] - stats["lo"] + eps) * 2 - 1
    if mode == "minmax_epoch":
        lo = X.min(axis=2, keepdims=True)
        return (X - lo) / (X.max(axis=2, keepdims=True) - lo + eps)
    if mode == "minmax_pm_epoch":
        lo = X.min(axis=2, keepdims=True)
        return (X - lo) / (X.max(axis=2, keepdims=True) - lo + eps) * 2 - 1
    if mode == "p95":
        if not stats or "p" not in stats:
            stats = compute_norm_stats(X, mode)
        return X / stats["p"]
    if mode == "x_div":
        c = float((stats or {}).get("constant", 100.0))
        return X / (abs(c) + eps)
    if mode == "iqr":
        if not stats or "med" not in stats:
            stats = compute_norm_stats(X, mode)
        return (X - stats["med"]) / stats["iqr"]
    if mode == "quantile":
        return _rank_normal(X).astype(np.float32)
    if mode == "alaw":
        A = 87.56
        u = np.abs(X).max(axis=2, keepdims=True)
        u = np.where(u > 0, u, 1.0)  # 全零 epoch 安全
        return (np.sign(X) * np.log(1 + A * np.abs(X) / u) / np.log(1 + A)).astype(np.float32)
    if mode == "ema":
        return _ema_normalize(X.astype(np.float64)).astype(np.float32)
    if mode == "zero_mean":
        return X - X.mean(axis=2, keepdims=True)
    if mode in (None, "none"):
        return X
    raise ValueError(f"未知归一化模式：{mode}（可用：{NORM_MODES}）")


def euclidean_alignment(X: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """欧几里得对齐（EA）：把每个 epoch 的平均协方差对齐到单位阵。

    应逐被试调用（使用该被试自身无标签数据，跨被试/LOSO 场景合法）。
    """
    X = np.asarray(X, dtype=np.float64)
    N, C, T = X.shape
    cov = np.einsum("nct,nkt->nck", X, X) / T          # (N, C, C)
    R = cov.mean(axis=0)
    # 收缩正则保证可逆
    trace = np.trace(R) / C
    R = (1 - shrink) * R + shrink * trace * np.eye(C)
    w, V = np.linalg.eigh(R)
    R_inv_half = (V * (1.0 / np.sqrt(np.clip(w, 1e-12, None)))) @ V.T
    return (R_inv_half @ X).astype(np.float32)


# ---------------------------------------------------------------- 增强
def augment_batch(x, time_shift: int = 0, noise_std: float = 0.0,
                  channel_dropout: float = 0.0, generator=None):
    """在一个 batch 张量 (B, C, T 或 B, 1, C, T) 上做在线增强。

    ``generator`` 兼容三种写法：``torch.Generator``（原样使用）、``int`` 种子、
    ``numpy`` 的 ``Generator``（借它的一个随机种子）——之前只接受 torch 生成器，
    传 numpy 的会抛 "randint() received an invalid combination of arguments"。
    """
    import torch

    if not isinstance(x, torch.Tensor):           # 兼容 numpy 输入
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    gen = generator
    if gen is not None and not isinstance(gen, torch.Generator):
        if isinstance(gen, (int, np.integer)):
            seed = int(gen)
        else:                                    # numpy / random 的 Generator
            seed = int(gen.integers(0, 2 ** 31 - 1))
        try:
            gen = torch.Generator(device=x.device).manual_seed(seed)
        except Exception:  # noqa: BLE001 - 某些后端（如 mps）不支持带 device 的生成器
            gen = torch.Generator().manual_seed(seed)

    if time_shift and time_shift > 0:
        B = x.shape[0]
        T = x.shape[-1]
        shifts = torch.randint(-int(time_shift), int(time_shift) + 1, (B,),
                               generator=gen, device=x.device)
        idx = (torch.arange(T, device=x.device).unsqueeze(0)
               + shifts.unsqueeze(1)) % T
        # (B, T) 要显式插轴再扩成 (B, …, T)。原来直接 idx.expand(*x.shape[:-1], T)
        # 只在 batch == 通道数时"碰巧"对齐成功（(B,T) 被当成 (C,T) 去广播），
        # 其它情况会抛 "expanded size of the tensor must match…" —— 时间平移等于不可用。
        view = (B,) + (1,) * (x.ndim - 2) + (T,)
        x = x.gather(-1, idx.reshape(view).expand(*x.shape[:-1], T))
    if noise_std and noise_std > 0:
        # 标准差都在「(通道, 时间)」二维上取：
        #   3D (B,C,T)   → dims=(1,2) → (B,1,1)
        #   4D (B,1,C,T) → dims=(2,3) → (B,1,1,1)
        # 原来 4D 分支写成 std(dim=(1,3)).unsqueeze(2)，形状对不上 x，相加时广播成
        # 5 维张量（(B,1,4,64)+(B,1,1,1,1) → (B,B,1,4,64)），噪声增强等于废掉。
        dims = (1, 2) if x.ndim == 3 else (2, 3)
        std = x.detach().std(dim=dims, keepdim=True)
        noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
        x = x + noise_std * std * noise
    if channel_dropout and channel_dropout > 0:
        if x.ndim == 3:
            mask = (torch.rand(x.shape[:2], device=x.device, generator=gen)
                    > channel_dropout).unsqueeze(-1)
        else:
            mask = (torch.rand((x.shape[0], x.shape[2]), device=x.device,
                               generator=gen) > channel_dropout)
            mask = mask.unsqueeze(1).unsqueeze(-1)
        x = x * mask
    return x
