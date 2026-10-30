"""不依赖音频重型库的 scalar-loss 占位 Loss。"""

from __future__ import annotations

import torch.nn as nn

from core.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("ScalarPassthroughLoss")
class ScalarPassthroughLoss(nn.Module):
    """供模型自身返回 scalar loss 的专用 Trainer 满足配置构造合同。"""

    def forward(self, enhanced, clean, lengths=None, **kwargs):
        del clean, lengths, kwargs
        return enhanced
