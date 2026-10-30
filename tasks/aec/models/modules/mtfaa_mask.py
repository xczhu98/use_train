"""MTFAA 风格两阶段频率邻域 mask head。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MTFAAMaskHead(nn.Module):
    """频率邻域幅度估计与复数 mask 组合的两阶段输出头。

    输入特征采用 ``(B, C, T, F)``，输入与输出频谱采用
    ``(B, F, T, 2)``。幅度阶段在每帧聚合相邻三个频点；预测卷积只读取
    当前及过去两个时间帧，因此不会引入未来帧依赖。
    """

    def __init__(self, ch: int):
        super().__init__()
        self.time_pad = 2
        self.mag_weight = nn.Conv2d(ch, 3, kernel_size=(3, 1))
        kernel = torch.eye(3).reshape(3, 1, 1, 3)
        self.register_buffer("freq_unfold_kernel", kernel)
        self.real_mask = nn.Conv2d(ch, 1, kernel_size=(3, 1))
        self.imag_mask = nn.Conv2d(ch, 1, kernel_size=(3, 1))

    def _causal_time_conv(self, conv: nn.Conv2d, x: Tensor) -> Tensor:
        return conv(F.pad(x, (0, 0, self.time_pad, 0)))

    def forward(self, decoder_out: Tensor, spec: Tensor) -> Tensor:
        """根据 decoder 特征增强复频谱。"""
        spec_2ch = spec.permute(0, 3, 2, 1)
        magnitude = torch.sqrt(
            spec_2ch[:, 0].square() + spec_2ch[:, 1].square() + 1e-12
        )
        weights = self._causal_time_conv(self.mag_weight, decoder_out).sigmoid()
        magnitude_pad = F.pad(magnitude.unsqueeze(1), (1, 1, 0, 0))
        magnitude_neighbors = F.conv2d(magnitude_pad, self.freq_unfold_kernel)
        smoothed_magnitude = (magnitude_neighbors * weights).sum(dim=1)

        real_mask = self._causal_time_conv(self.real_mask, decoder_out).squeeze(1)
        imag_mask = self._causal_time_conv(self.imag_mask, decoder_out).squeeze(1)
        mask_magnitude = torch.sqrt(
            torch.clamp(real_mask.square() + imag_mask.square(), min=1e-7)
        )
        mask_phase = torch.atan2(imag_mask, real_mask + 1e-7)

        enhanced_magnitude = smoothed_magnitude * torch.tanh(mask_magnitude)
        phase = torch.atan2(spec_2ch[:, 1], spec_2ch[:, 0]) + mask_phase
        output = torch.stack(
            [enhanced_magnitude * torch.cos(phase), enhanced_magnitude * torch.sin(phase)],
            dim=1,
        )
        return output.permute(0, 3, 2, 1)
