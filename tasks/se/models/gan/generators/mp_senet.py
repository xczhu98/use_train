"""MP-SENet Generator wrapper for use_train framework.

Reference: "MP-SENet: A Speech Enhancement Model with Parallel Denoising of
Magnitude and Phase Spectra" (Interspeech 2023).
Upstream: https://github.com/yxlu-0102/MP-SENet

Wrapper responsibilities:
  - Receive real-valued complex spectrogram ``(B, F, T, 2)`` from framework STFT.
  - Extract magnitude / phase, compress magnitude by ``compress_factor``.
  - Feed compressed (mag, pha) into upstream ``MPNet``.
  - Decompress MPNet's output magnitude and rebuild ``(B, F, T, 2)``.

The upstream ``MPNet`` itself is spectrum-in / spectrum-out (mag/pha only);
compression / decompression lives in this wrapper so the framework always
sees ``(B, F, T, 2)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tasks.se.models.modules.mp_senet_impl.generator import MPNet
from core.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("MPSEnetGenerator")
class MPSEnetGenerator(nn.Module):
    """Framework-compatible wrapper for MP-SENet's MPNet.

    Args:
        n_fft: FFT size (must match framework STFT).
        hop_length: Hop length (stored for info, not used inside MPNet).
        win_length: Window length (stored for info, not used inside MPNet).
        compress_factor: Power-law compression exponent (default 0.3).
        num_tsblocks: Number of Transformer blocks in MPNet.
        dense_channel: DenseEncoder channel width.
        beta: Sigmoid slope factor for mask decoder.
    """

    def __init__(
        self,
        n_fft: int = 400,
        hop_length: int = 100,
        win_length: int = 400,
        compress_factor: float = 0.3,
        num_tsblocks: int = 4,
        dense_channel: int = 64,
        beta: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        for key in ("window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"MPSEnetGenerator got unexpected keyword arguments: {sorted(kwargs)}")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.compress_factor = compress_factor

        # MPNet expects an ``h`` namespace object with specific attributes.
        class H:
            pass

        h = H()
        h.n_fft = n_fft
        h.hop_length = hop_length
        h.win_length = win_length
        h.dense_channel = dense_channel
        h.beta = beta

        self.mpnet = MPNet(h, num_tsblocks=num_tsblocks)

    def forward(self, noisy_spec: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
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
            raise TypeError("MPSEnetGenerator noisy_spec must use a floating-point dtype")
        # Extract magnitude and phase from real-valued complex representation
        mag = torch.sqrt(noisy_spec[..., 0] ** 2 + noisy_spec[..., 1] ** 2 + 1e-9)
        pha = torch.atan2(noisy_spec[..., 1] + 1e-10, noisy_spec[..., 0] + 1e-5)

        # Compress magnitude before feeding to MPNet
        mag_c = torch.pow(mag, self.compress_factor)

        # MPNet forward: outputs compressed magnitude + phase
        denoised_amp_c, denoised_pha, _ = self.mpnet(mag_c, pha)

        # Decompress magnitude back to original domain
        denoised_amp = torch.pow(denoised_amp_c, 1.0 / self.compress_factor)

        # Rebuild complex spectrogram
        real = denoised_amp * torch.cos(denoised_pha)
        imag = denoised_amp * torch.sin(denoised_pha)
        enhanced_spec = torch.stack([real, imag], dim=-1)

        return enhanced_spec
