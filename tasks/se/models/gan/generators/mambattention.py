"""MambAttention Generator wrapper for use_train framework.

Reference: "MambAttention: Mamba With Multi-Head Attention for Generalizable
Single-Channel Speech Enhancement" (IEEE/ACM TASLP 2026).
Upstream: https://github.com/NikolaiKyhne/MambAttention

Wrapper responsibilities:
  - Receive real-valued complex spectrogram ``(B, F, T, 2)`` from framework STFT.
  - Extract magnitude / phase, compress magnitude.
  - Feed into upstream ``MambAttention`` (DenseEncoder + MambAttentionBlock
    stack + MagDecoder / PhaseDecoder).
  - Decompress output magnitude and rebuild ``(B, F, T, 2)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tasks.se.models.modules.mambattention_impl.generator import MambAttention
from core.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("MambAttentionGenerator")
class MambAttentionGenerator(nn.Module):
    """Framework-compatible wrapper for MambAttention generator.

    Args:
        n_fft: FFT size (must match framework STFT).
        hop_length: Hop length (stored for info).
        win_length: Window length (stored for info).
        compress_factor: Power-law compression exponent (default 0.3).
        hid_feature: Hidden feature dimension (default 64, matching upstream ``m`` variant).
        num_tfmamba: Number of TFMamba blocks.
        d_state: Mamba SSM state dimension.
        d_conv: Mamba causal convolution width.
        expand: Mamba expansion factor.
        norm_epsilon: LayerNorm epsilon.
        beta: Sigmoid slope factor for mask decoder.
    """

    def __init__(
        self,
        n_fft: int = 400,
        hop_length: int = 100,
        win_length: int = 400,
        compress_factor: float = 0.3,
        hid_feature: int = 64,
        num_tfmamba: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 4,
        norm_epsilon: float = 1e-5,
        beta: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        for key in ("window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"MambAttentionGenerator got unexpected keyword arguments: {sorted(kwargs)}")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.compress_factor = compress_factor

        cfg = {
            "stft_cfg": {
                "n_fft": n_fft,
                "hop_size": hop_length,
                "win_size": win_length,
                "compress_factor": compress_factor,
            },
            "model_cfg": {
                "hid_feature": hid_feature,
                "compress_factor": compress_factor,
                "num_tfmamba": num_tfmamba,
                "d_state": d_state,
                "d_conv": d_conv,
                "expand": expand,
                "norm_epsilon": norm_epsilon,
                "beta": beta,
                "input_channel": 2,
                "output_channel": 1,
            },
        }
        self.model = MambAttention(cfg)

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
            raise TypeError("MambAttentionGenerator noisy_spec must use a floating-point dtype")
        mag = torch.sqrt(noisy_spec[..., 0] ** 2 + noisy_spec[..., 1] ** 2 + 1e-9)
        pha = torch.atan2(noisy_spec[..., 1] + 1e-10, noisy_spec[..., 0] + 1e-5)
        mag_c = torch.pow(mag, self.compress_factor)
        denoised_amp_c, denoised_pha, _ = self.model(mag_c, pha)
        denoised_amp = torch.pow(denoised_amp_c, 1.0 / self.compress_factor)
        real = denoised_amp * torch.cos(denoised_pha)
        imag = denoised_amp * torch.sin(denoised_pha)
        return torch.stack([real, imag], dim=-1)
