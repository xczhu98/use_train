"""Fast-ULCNet 使用的 Complex Ratio Mask。

基于 ``narrietal/CRM_pytorch`` 固定版本复制并适配；commit、适配范围和 MIT
许可见本目录 ``PROVENANCE`` 与 ``LICENSE.CRM_pytorch``。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ComplexRatioMask(nn.Module):
    """按 CRM_pytorch 的 E/C/R 语义应用复数掩码。"""

    def __init__(self, masking_mode: str = "E"):
        super().__init__()
        self.masking_mode = masking_mode.upper()
        self.eps = 1e-8
        if self.masking_mode not in ("E", "C", "R"):
            raise ValueError("masking_mode must be one of 'E', 'C', 'R'")

    def forward(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        m_real: torch.Tensor,
        m_imag: torch.Tensor,
    ) -> torch.Tensor:
        """返回与输入形状一致的 complex enhanced spectrum。"""
        if self.masking_mode == "E":
            input_magnitude = torch.sqrt(real**2 + imag**2)
            input_phase = torch.atan2(imag, real)
            safe_mask_real = m_real + self.eps
            mask_magnitude = torch.tanh(
                torch.sqrt(safe_mask_real**2 + m_imag**2)
            )
            mask_phase = torch.atan2(m_imag, safe_mask_real)
            estimated_magnitude = mask_magnitude * input_magnitude
            estimated_phase = input_phase + mask_phase
            out_real = estimated_magnitude * torch.cos(estimated_phase)
            out_imag = estimated_magnitude * torch.sin(estimated_phase)
        elif self.masking_mode == "C":
            out_real = real * m_real - imag * m_imag
            out_imag = real * m_imag + imag * m_real
        else:  # R：CRM_pytorch 分别缩放 real / imag 分量
            out_real = real * m_real
            out_imag = imag * m_imag
        return torch.complex(out_real, out_imag)
