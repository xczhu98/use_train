"""MossFormerGAN Generator wrapper for use_train framework.

Reference: ClearerVoice-Studio MossFormerGAN (ModelScope).
Upstream: https://github.com/modelscope/ClearerVoice-Studio

Wrapper responsibilities:
  - Receive real-valued complex spectrogram ``(B, F, T, 2)`` from framework STFT.
  - Apply ``power_compress`` (compress_factor=0.3) and permute to ``(B, 2, T, F)``
    to match upstream SyncANet's expected layout.
  - Feed into upstream ``SyncANet`` (DenseEncoder + SyncANetBlock stack +
    MaskDecoder + ComplexDecoder).
  - Permute outputs back and apply ``power_uncompress`` to rebuild ``(B, F, T, 2)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tasks.se.models.modules.clearervoice_impl.mossformer_gan.generator import SyncANet
from core.utils.registry import MODEL_REGISTRY


def _power_compress(x: torch.Tensor, compress_factor: float = 0.3) -> torch.Tensor:
    """(B, F, T, 2) -> (B, 2, F, T) compressed real/imag channels."""
    real, imag = x[..., 0], x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** compress_factor
    real_c = mag * torch.cos(phase)
    imag_c = mag * torch.sin(phase)
    return torch.stack([real_c, imag_c], dim=1)


def _power_uncompress(
    real: torch.Tensor, imag: torch.Tensor, compress_factor: float = 0.3
) -> torch.Tensor:
    """Compressed ``real``/``imag`` tensors -> (..., 2) last dim [Re, Im]."""
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1.0 / compress_factor)
    real_u = mag * torch.cos(phase)
    imag_u = mag * torch.sin(phase)
    return torch.stack([real_u, imag_u], dim=-1)


@MODEL_REGISTRY.register("MossFormerGANGenerator")
class MossFormerGANGenerator(nn.Module):
    """Framework-compatible wrapper for MossFormerGAN's SyncANet generator.

    Args:
        n_fft: FFT size (must match framework STFT).
        hop_length: Hop length (stored for info).
        win_length: Window length (stored for info).
        num_channel: DenseEncoder channel width (default 64).
        compress_factor: Power compression exponent (default 0.3).
    """

    def __init__(
        self,
        n_fft: int = 400,
        hop_length: int = 100,
        win_length: int = 400,
        num_channel: int = 64,
        compress_factor: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        for key in ("window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"MossFormerGANGenerator got unexpected keyword arguments: {sorted(kwargs)}")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.compress_factor = float(compress_factor)
        num_features = n_fft // 2 + 1
        self.core = SyncANet(num_channel=num_channel, num_features=num_features)

    def forward(
        self, noisy_spec: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            noisy_spec: ``(B, F, T, 2)`` real-valued complex spectrogram.
            lengths: Unused (API compatibility).

        Returns:
            enhanced_spec: ``(B, F, T, 2)`` enhanced complex spectrogram.
        """
        if noisy_spec.dim() != 4 or noisy_spec.shape[-1] != 2:
            raise ValueError(f"Expected [B, F, T, 2], got {tuple(noisy_spec.shape)}")
        if noisy_spec.shape[1] != self.n_fft // 2 + 1:
            raise ValueError(f"Expected F={self.n_fft // 2 + 1}, got F={noisy_spec.shape[1]}")
        if not noisy_spec.is_floating_point():
            raise TypeError("MossFormerGANGenerator noisy_spec must use a floating-point dtype")
        # SyncANet expects (B, 2, T, F) layout with compressed real/imag channels
        z = _power_compress(noisy_spec, self.compress_factor).permute(0, 1, 3, 2)
        out_list = self.core(z)
        pred_real = out_list[0].permute(0, 1, 3, 2)  # (B, 1, F, T)
        pred_imag = out_list[1].permute(0, 1, 3, 2)  # (B, 1, F, T)
        enh = _power_uncompress(pred_real, pred_imag, self.compress_factor).squeeze(1)
        return enh
