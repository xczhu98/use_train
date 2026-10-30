# SPDX-License-Identifier: Apache-2.0
"""WeSep cyclic angle encoding，移植自 commit ``dc11cb18``。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CycPosEncoding(nn.Module):
    """将任意角张量 ``[...]`` 编码为 ``[...,D]``，其中 ``D`` 必须为偶数。"""

    def __init__(self, embed_dim: int, alpha: float = 1.0):
        super().__init__()
        if type(embed_dim) is not int or embed_dim <= 0 or embed_dim % 2:
            raise ValueError(f"embed_dim must be a positive even int, got {embed_dim!r}")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a finite positive number")
        alpha = float(alpha)
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be a finite positive number")
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.register_buffer("div_term", div_term)

    def forward(self, angle: torch.Tensor) -> torch.Tensor:
        if not isinstance(angle, torch.Tensor) or not angle.dtype.is_floating_point:
            raise TypeError("angle must be a floating-point tensor in radians")
        if not bool(torch.isfinite(angle).all().item()):
            raise ValueError("angle must contain only finite radians")
        phase = angle.unsqueeze(-1) * self.alpha * self.div_term.to(angle.dtype)
        return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1).flatten(-2)


__all__ = ["CycPosEncoding"]
