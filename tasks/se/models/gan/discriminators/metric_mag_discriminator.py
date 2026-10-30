"""MetricGAN mag-only discriminator wrapper.

Adapts the original MP-SENet / MambAttention / RWSAMamba-UNet MetricDiscriminator
(2-channel mag input) to the framework's 4-argument discriminator interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tasks.se.models.modules.mp_senet_impl.discriminator import MetricDiscriminator as _OriginalMetricDiscriminator
from core.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("MetricMagDiscriminator")
class MetricMagDiscriminator(nn.Module):
    """Mag-only metric discriminator for MetricGAN training.

    Architecture (from original repos):
    - 4× SpectralNorm Conv2d (4×4, stride 2) + InstanceNorm2d + PReLU
    - AdaptiveMaxPool2d(1) → Flatten
    - Linear(dim*8 → dim*4) + Dropout(0.3) + PReLU
    - Linear(dim*4 → 1) + LearnableSigmoid1D

    Input: framework-style 4 tensors (ref_mag, ref_pha, enh_mag, enh_pha)
    Output: scalar score per sample (B,)
    """

    def __init__(self, dim: int = 16):
        super().__init__()
        self.inner = _OriginalMetricDiscriminator(dim=dim, in_channel=2)

    def forward(
        self,
        ref_mag: torch.Tensor,
        ref_pha: torch.Tensor,
        enh_mag: torch.Tensor,
        enh_pha: torch.Tensor,
    ) -> torch.Tensor:
        # Original discriminator only uses magnitude (ignores phase)
        # Inner MetricDiscriminator.stack produces (B, 2, F, T) from 3-D inputs.
        # If callers pass 4-D (B, 1, F, T) we must squeeze first to avoid 5-D output.
        if ref_mag.dim() == 4:
            ref_mag = ref_mag.squeeze(1)
        if enh_mag.dim() == 4:
            enh_mag = enh_mag.squeeze(1)
        return self.inner(ref_mag, enh_mag).squeeze(-1)
