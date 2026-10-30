"""Composite loss for MP-SENet / MambAttention / RWSAMamba-UNet MetricGAN models.

已注册任务专用，仅供 MetricGANCompositeTrainer 使用，
签名非标准（forward(forward_dict: dict) 而非标准 (enhanced, clean, lengths)）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from core.losses.utils import create_frame_mask, prepare_centered_stft_inputs
from core.utils.registry import LOSS_REGISTRY


def anti_wrapping_function(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x - torch.round(x / (2.0 * np.pi)) * 2.0 * np.pi)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """按广播后的有效元素数归约，空差分维保持可反向的图零。"""
    expanded_mask = mask.expand_as(value).to(value.dtype)
    return (value * expanded_mask).sum() / expanded_mask.sum().clamp(min=1)


def phase_losses(
    phase_r: torch.Tensor,
    phase_g: torch.Tensor,
    frame_mask: torch.Tensor | None = None,
):
    """IP + GD + IAF anti-wrapping phase losses。"""
    ip_error = anti_wrapping_function(phase_r - phase_g)
    gd_error = anti_wrapping_function(
        torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)
    )
    iaf_error = anti_wrapping_function(
        torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)
    )
    if frame_mask is None:
        return ip_error.mean(), gd_error.mean(), iaf_error.mean()

    spec_mask = frame_mask.unsqueeze(1)
    ip_loss = _masked_mean(ip_error, spec_mask)
    gd_loss = _masked_mean(gd_error, spec_mask)
    iaf_loss = _masked_mean(iaf_error, spec_mask[..., 1:])
    return ip_loss, gd_loss, iaf_loss


@LOSS_REGISTRY.register("MPSEnetCompositeLoss")
class MPSEnetCompositeLoss(nn.Module):
    """Composite loss for MP-SENet-style MetricGAN models.

    Usage in trainer._compute_reconstruction_loss:
        loss = self.composite_loss(forward_dict)
    """

    # 非标准签名标注: forward(forward_dict: dict), 供 registry 遍历/冒烟测试跳过
    SIGNATURE = "forward_dict"

    def __init__(
        self,
        weight_mag: float = 0.9,
        weight_pha: float = 0.3,
        weight_com: float = 0.1,
        weight_time: float = 0.2,
        weight_stft: float = 0.1,
        stft_n_fft: int = 400,
        stft_hop_length: int = 100,
        stft_win_length: int = 400,
        compress_factor: float = 0.3,
        window: str = "hann",
    ):
        super().__init__()
        self.weights = {
            "mag": weight_mag,
            "pha": weight_pha,
            "com": weight_com,
            "time": weight_time,
            "stft": weight_stft,
        }
        self.n_fft = stft_n_fft
        self.hop_length = stft_hop_length
        self.win_length = stft_win_length
        self.compress_factor = compress_factor
        # 与 core/stft 的有效窗名集合保持一致：未知窗名必须 fail-fast，
        # 不能静默回退 hann（配置拼写错误会静默改变损失数值）。
        if not isinstance(window, str) or window.lower() not in ("hann", "hamming"):
            raise ValueError(
                f"window must be one of: hann, hamming; got {window!r}"
            )
        if window.lower() == "hamming":
            win = torch.hamming_window(stft_win_length)
        else:
            win = torch.hann_window(stft_win_length)
        self.register_buffer("window", win)

    def forward(self, forward_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute composite loss from forward_dict (output of _forward_generator)."""
        clean_mag = forward_dict["clean_mag"]          # compressed
        clean_pha = forward_dict["clean_pha"]
        enhanced_mag = forward_dict["enhanced_mag"]    # compressed
        enhanced_pha = forward_dict["enhanced_pha"]
        raw_spec = forward_dict.get("loss_enhanced_spectra")
        if raw_spec is not None:
            raw_complex = torch.view_as_complex(raw_spec.contiguous())
            enhanced_mag = torch.abs(raw_complex).pow(self.compress_factor)
            enhanced_pha = torch.angle(raw_complex)
        enhanced = forward_dict["enhanced"]            # waveform
        clean = forward_dict["clean"]                  # waveform
        provided_lengths = forward_dict.get("lengths")

        (enhanced_stft_input, _), lengths, sample_mask = prepare_centered_stft_inputs(
            enhanced, clean, lengths=provided_lengths, n_fft=self.n_fft
        )
        frame_mask = create_frame_mask(
            lengths,
            clean_mag.shape[-1],
            self.n_fft,
            self.win_length,
            self.hop_length,
            clean_mag.device,
        )
        spec_mask = frame_mask.unsqueeze(1)

        def _masked_mse(a, b, mask):
            return _masked_mean((a - b).square(), mask)

        # 1. Magnitude loss (on compressed magnitude)
        loss_mag = _masked_mse(enhanced_mag, clean_mag, spec_mask)

        # 2. Phase loss (IP + GD + IAF)
        ip_loss, gd_loss, iaf_loss = phase_losses(
            clean_pha, enhanced_pha, frame_mask
        )
        loss_pha = ip_loss + gd_loss + iaf_loss

        # 3. Complex loss（官方 mag_phase_stft 的压缩复谱域）
        clean_com = torch.stack([
            clean_mag * torch.cos(clean_pha),
            clean_mag * torch.sin(clean_pha),
        ], dim=-1)
        enh_com = torch.stack([
            enhanced_mag * torch.cos(enhanced_pha),
            enhanced_mag * torch.sin(enhanced_pha),
        ], dim=-1)
        com_mask = spec_mask.unsqueeze(-1)
        loss_com = _masked_mse(enh_com, clean_com, com_mask) * 2.0

        # 4. Time loss (L1 on waveform)
        loss_time = _masked_mean((enhanced - clean).abs(), sample_mask)

        # 5. STFT consistency loss（重建波形重新分析后的压缩复谱域）
        win = self.window.to(device=enhanced.device, dtype=enhanced.dtype)
        enh_stft = torch.stft(
            enhanced_stft_input,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=win,
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        enh_stft = enh_stft[..., :enh_com.shape[-2]]
        # 官方 consistency 分支调用 mag_phase_stft(..., addeps=True)。
        eps = 1e-10
        enh_stft_mag = torch.sqrt(
            enh_stft.real.square() + enh_stft.imag.square() + eps
        ).pow(self.compress_factor)
        enh_stft_pha = torch.atan2(enh_stft.imag + eps, enh_stft.real + eps)
        enh_stft_com = torch.stack(
            [
                enh_stft_mag * torch.cos(enh_stft_pha),
                enh_stft_mag * torch.sin(enh_stft_pha),
            ],
            dim=-1,
        )
        loss_stft = _masked_mse(enh_com, enh_stft_com, com_mask) * 2.0

        total = (
            self.weights["mag"] * loss_mag
            + self.weights["pha"] * loss_pha
            + self.weights["com"] * loss_com
            + self.weights["time"] * loss_time
            + self.weights["stft"] * loss_stft
        )
        return total
