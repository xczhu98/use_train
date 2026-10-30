"""归一化层集合（移植自 wesep ``wesep/modules/common/norm.py``，仅保留用到的部分）.

- ``select_norm``：按名字构造归一化层（cLN/LN/gLN/GN/BN）；
- ``ChannelWiseLayerNorm``：通道维 LN（``(B, C, T)`` 输入）；
- ``GlobalChannelLayerNorm``：全局通道 LN（ConvTasNet 风格）。

Author: Xincheng Zhu
"""

from __future__ import annotations

import torch
import torch.nn as nn


def select_norm(norm: str, dim, eps: float = 1e-5, group: int = 1):
    """按名字构造归一化层。LN 代价比 BN 大。"""
    if norm not in ["cLN", "LN", "gLN", "GN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, eps, elementwise_affine=True)
    elif norm == "LN":
        # dim 可以是 int 或 tuple
        return nn.LayerNorm(dim, eps, elementwise_affine=True)
    elif norm == "GN":
        return nn.GroupNorm(group, dim, eps)
    elif norm == "BN":
        return nn.BatchNorm1d(dim, eps)
    else:
        return GlobalChannelLayerNorm(dim, eps, elementwise_affine=True)


class ChannelWiseLayerNorm(nn.LayerNorm):
    """通道维 LayerNorm，输入 ``(B, C, T)``。"""

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        x = torch.transpose(x, 1, -1)
        x = super().forward(x)
        x = torch.transpose(x, 1, -1)
        return x


class GlobalChannelLayerNorm(nn.Module):
    """全局 LayerNorm（在 (C, T) 维上做统计）。"""

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # x = N x C x L
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__class__.__name__))
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
        if self.elementwise_affine:
            x = (self.weight * (x - mean) / torch.sqrt(var + self.eps) +
                 self.bias)
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x
