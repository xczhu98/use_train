"""RWSA-MambaUNet generator wrapper for the use_train framework.

Reference: "Exploring Resolution-Wise Shared Attention in Hybrid Mamba-U-Nets
for Improved Cross-Corpus Speech Enhancement" (ICASSP 2026).
Upstream: https://github.com/NikolaiKyhne/RWSAMamba-UNet
Pinned commit: 02635ab7e92a0584fde729f8a55f398c2379da82
License: MIT, Copyright (c) 2025 Nikolai Lund Kühne.

Wrapper responsibilities:
  - Receive real-valued complex spectrogram ``(B, F, T, 2)`` from framework STFT.
  - Extract magnitude / phase with the upstream ``torch.abs`` / ``torch.angle`` semantics.
  - Feed compressed magnitude and phase into upstream ``MambAttentionSEUNet``.
  - Decompress output magnitude and rebuild ``(B, F, T, 2)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.registry import MODEL_REGISTRY
from tasks.se.models.modules.rwsa_mamba_impl.generator import MambAttentionSEUNet


@MODEL_REGISTRY.register("RWSAMambaUNetGenerator")
class RWSAMambaUNetGenerator(nn.Module):
    """Framework-compatible wrapper for the official RWSA-MambaUNet generator.

    Args:
        n_fft: FFT size injected from the top-level STFT config.
        hop_length: Hop length injected from the top-level STFT config.
        win_length: Window length injected from the top-level STFT config.
        window: STFT window name, retained to bind the wrapper to the top-level contract.
        center: STFT centering flag, retained to bind the wrapper to the top-level contract.
        compress_factor: Power-law compression exponent (default 0.3).
        hid_feature: Hidden feature dimension (16/16/24 for official XS/S/M variants).
        num_tfmamba: Number of TF-Mamba blocks (2/4/4 for official XS/S/M variants).
        d_state: Mamba SSM state dimension.
        d_conv: Mamba causal convolution width.
        expand: Mamba expansion factor.
        norm_epsilon: LayerNorm epsilon.
        beta: Sigmoid slope factor for mask decoder.
    """

    def __init__(
        self,
        n_fft: int = 510,
        hop_length: int = 120,
        win_length: int = 510,
        window: str = "hann",
        center: bool = True,
        compress_factor: float = 0.3,
        hid_feature: int = 16,
        num_tfmamba: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 4,
        norm_epsilon: float = 1e-5,
        beta: float = 2.0,
    ):
        super().__init__()
        for name, value in {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "hid_feature": hid_feature,
            "num_tfmamba": num_tfmamba,
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if win_length > n_fft:
            raise ValueError(f"win_length must not exceed n_fft, got {win_length} > {n_fft}")
        if not isinstance(window, str) or not window:
            raise ValueError(f"window must be a non-empty string, got {window!r}")
        if not isinstance(center, bool):
            raise TypeError(f"center must be bool, got {type(center).__name__}")
        for name, value in {
            "compress_factor": compress_factor,
            "norm_epsilon": norm_epsilon,
            "beta": beta,
        }.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        if hid_feature % 8 != 0:
            raise ValueError(
                f"hid_feature must be divisible by 8 for the official attention heads, got {hid_feature}"
            )

        expected_freq_bins = n_fft // 2 + 1
        if expected_freq_bins % 8 != 0:
            raise ValueError(
                "RWSA-MambaUNet requires F=n_fft//2+1 to be divisible by 8 "
                "for exact DenseEncoder/PixelShuffle reconstruction, "
                f"got F={expected_freq_bins} from n_fft={n_fft}"
            )

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.compress_factor = float(compress_factor)

        cfg = {
            "stft_cfg": {
                "n_fft": n_fft,
                "hop_size": hop_length,
                "win_size": win_length,
                "compress_factor": self.compress_factor,
            },
            "model_cfg": {
                "hid_feature": hid_feature,
                "compress_factor": self.compress_factor,
                "num_tfmamba": num_tfmamba,
                "d_state": d_state,
                "d_conv": d_conv,
                "expand": expand,
                "norm_epsilon": float(norm_epsilon),
                "beta": float(beta),
                "input_channel": 2,
                "output_channel": 1,
            },
        }
        self.model = MambAttentionSEUNet(cfg)

    def forward(self, noisy_spec: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Enhance a real-view complex spectrogram.

        Args:
            noisy_spec: Floating-point tensor shaped ``(B, F, T, 2)``.
            lengths: Unused; accepted for the shared SE model call contract.

        Returns:
            Enhanced real-view complex spectrogram with the same shape as ``noisy_spec``.
        """
        if not isinstance(noisy_spec, torch.Tensor):
            raise TypeError(f"noisy_spec must be a torch.Tensor, got {type(noisy_spec).__name__}")
        if noisy_spec.ndim != 4 or noisy_spec.shape[-1] != 2:
            raise ValueError(
                "noisy_spec must have shape (B, F, T, 2), "
                f"got {tuple(noisy_spec.shape)}"
            )
        if not noisy_spec.is_floating_point():
            raise TypeError(f"noisy_spec must be floating point, got {noisy_spec.dtype}")

        batch, freq_bins, time_frames, _ = noisy_spec.shape
        if batch <= 0 or time_frames <= 0:
            raise ValueError(
                "noisy_spec batch and time dimensions must be non-empty, "
                f"got {tuple(noisy_spec.shape)}"
            )
        expected_freq_bins = self.n_fft // 2 + 1
        if freq_bins != expected_freq_bins:
            raise ValueError(
                f"noisy_spec has F={freq_bins}, expected {expected_freq_bins} "
                f"for n_fft={self.n_fft}"
            )

        # 两级 PixelUnshuffle(2) 要求时间轴为 4 的倍数；输出随后裁回原长度。
        pad_t = (-time_frames) % 4
        if pad_t:
            noisy_spec = F.pad(noisy_spec, (0, 0, 0, pad_t))

        noisy_complex = torch.view_as_complex(noisy_spec.contiguous())
        magnitude = torch.abs(noisy_complex)
        phase = torch.angle(noisy_complex)
        compressed_magnitude = torch.pow(magnitude, self.compress_factor)
        enhanced_magnitude_c, enhanced_phase, _ = self.model(compressed_magnitude, phase)
        enhanced_magnitude = torch.pow(enhanced_magnitude_c, 1.0 / self.compress_factor)
        enhanced = torch.stack(
            [
                enhanced_magnitude * torch.cos(enhanced_phase),
                enhanced_magnitude * torch.sin(enhanced_phase),
            ],
            dim=-1,
        )
        return enhanced[:, :, :time_frames, :] if pad_t else enhanced
