"""Discriminators for GAN-based speech enhancement and AEC.

All discriminators follow the MetricGAN paradigm:
- Input: concatenated spectral features of (reference, enhanced)
- Output: scalar quality score (trained to approximate perceptual metrics like PESQ)
- Loss: MSE against real metric scores (not binary 0/1 classification)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from core.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("MetricDiscriminator")
class MetricDiscriminator(nn.Module):
    """PESQ-guided metric discriminator for GAN-based speech enhancement/AEC.

    Architecture: SpectralNorm Conv2d encoder that maps concatenated (ref, enhanced)
    spectral features to a scalar quality score.

    Input: ``ref_mag`` `(B,1,F,T)`, ``ref_pha`` `(B,1,F,T)`,
           ``enh_mag`` `(B,1,F,T)`, ``enh_pha`` `(B,1,F,T)`
    Output: ``score`` `(B,)` scalar in approximate range [0, 1]

    Reference: MetricGAN / MP-SENet / MambAttention style discriminator.
    """

    def __init__(
        self,
        input_features: int = 4,
        channels: tuple[int, ...] = (32, 32, 32, 1),
        kernel_size: int = 3,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = input_features
        for i, out_ch in enumerate(channels):
            stride = (1, 2) if i < len(channels) - 1 else 1
            conv = nn.Conv2d(
                in_ch, out_ch, kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
            if use_spectral_norm:
                conv = spectral_norm(conv)
            layers.append(conv)
            if i < len(channels) - 1:
                layers.append(nn.PReLU())
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)

    def forward(
        self,
        ref_mag: torch.Tensor,
        ref_pha: torch.Tensor,
        enh_mag: torch.Tensor,
        enh_pha: torch.Tensor,
    ) -> torch.Tensor:
        """Return scalar quality score for each sample in batch.

        All inputs expected shape ``(B, 1, F, T)`` or ``(B, F, T)``.
        """
        # Ensure channel dim
        def _ensure_ch2(x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 3:
                return x.unsqueeze(1)
            return x

        ref_mag = _ensure_ch2(ref_mag)
        ref_pha = _ensure_ch2(ref_pha)
        enh_mag = _ensure_ch2(enh_mag)
        enh_pha = _ensure_ch2(enh_pha)

        x = torch.cat([ref_mag, ref_pha, enh_mag, enh_pha], dim=1)  # (B, 4, F, T)
        score = self.encoder(x)  # (B, 1, F', T')
        return score.mean(dim=[1, 2, 3])  # (B,)


@MODEL_REGISTRY.register("SpectralDiscriminator")
class SpectralDiscriminator(nn.Module):
    """Lightweight spectral discriminator using only magnitude (no phase).

    Useful when phase is not available or when you want a simpler discriminator.
    """

    def __init__(
        self,
        input_features: int = 2,  # ref_mag, enh_mag
        channels: tuple[int, ...] = (16, 16, 16, 1),
        kernel_size: int = 3,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = input_features
        for i, out_ch in enumerate(channels):
            stride = (1, 2) if i < len(channels) - 1 else 1
            conv = nn.Conv2d(
                in_ch, out_ch, kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
            if use_spectral_norm:
                conv = spectral_norm(conv)
            layers.append(conv)
            if i < len(channels) - 1:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)

    def forward(
        self,
        ref_mag: torch.Tensor,
        ref_pha: torch.Tensor,
        enh_mag: torch.Tensor,
        enh_pha: torch.Tensor,
    ) -> torch.Tensor:
        """Forward compatible with 4-arg GANTrainer interface (phase ignored)."""
        def _ensure_ch2(x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 3:
                return x.unsqueeze(1)
            return x

        ref_mag = _ensure_ch2(ref_mag)
        enh_mag = _ensure_ch2(enh_mag)
        x = torch.cat([ref_mag, enh_mag], dim=1)
        score = self.encoder(x)
        return score.mean(dim=[1, 2, 3])
