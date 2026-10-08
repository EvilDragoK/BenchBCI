"""模型内部小工具：用一次「哑前向」推断分类头的输入维数。

放在单独模块里，避免 ``core.models.__init__`` 与各模型文件互相 import 成环。
"""
from __future__ import annotations


def infer_flat_dim(block, n_ch: int, n_times: int, tag: str = "") -> int:
    """跑一次哑前向，返回展平后的特征维数（= 分类头 ``in_features``）。

    两个细节很重要：

    1. **在 ``eval()`` 下跑**：``BatchNorm`` 在 ``train()`` 且 batch=1、每个通道只有
       1 个值时，无论输入多长都会抛 "Expected more than 1 value per channel" ——
       这会把真正的原因（时窗太短、时间维被池化压没了）藏在后面，报错完全看不懂。
    2. **失败时给出可操作建议**：直接告诉用户各类模型的经验最小时长。
    """
    import torch

    was_training = block.training
    block.eval()
    try:
        with torch.no_grad():
            out = block(torch.zeros(1, 1, n_ch, n_times))
        return int(out.flatten(1).shape[1])
    except Exception as e:  # noqa: BLE001
        first = str(e).splitlines()[0][:120]
        raise ValueError(
            f"推断{tag or '模型'}特征维数失败（n_ch={n_ch}, n_times={n_times}）：{first}\n"
            f"→ 最常见原因是**输入时窗太短**：卷积 + 池化后时间维被压到 0/1。"
            f"经验最小时长（采样点）：EEGNet/FBCNet/Conformer ≥ 64、"
            f"ShallowConvNet ≥ 128、DeepConvNet ≥ 500；"
            f"也可减小 kernel_length 或去掉一层池化。") from e
    finally:
        block.train(was_training)
