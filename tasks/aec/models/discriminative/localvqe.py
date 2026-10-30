"""LocalVQE — streaming-friendly AEC model with S4D bottleneck and CausalGroupNorm.

Reference: LocalVQE, a lightweight joint AEC / NS / dereverberation model
(https://github.com/localai-org/LocalVQE).  Adapted to the spectrum-in /
spectrum-out training framework used by all AEC models in this codebase.

Key architectural differences from DeepVQE:
  1. **CausalGroupNorm** (per-frame normalization over (C, F)) instead of
     BatchNorm2d → streaming-friendly, no train/eval mismatch.
  2. **S4D Bottleneck** (Diagonal State Space Model) instead of GRU
     → complex diagonal SSM with polar parameterization for guaranteed stability.
  3. **SiLU** activation (arch_version=3) instead of ELU.
  4. **(4, 4)** kernel size instead of DeepVQE's (4, 3).
  5. **Narrower channels**: [32, 40, 40, 40, 40] vs DeepVQE base [64, 128, 128, 128, 128]
     → 当前实现约 0.78M params vs ~8M.
  6. **AlignBlock** with separate hidden_channels (32) and in_channels (40)
     → Q/K projected to lower dimension, but weighted sum uses full feature channels.
  7. **Temperature** parameter in AlignBlock for sharper/fuzzier delay distributions.

Architecture (spectrum-in / spectrum-out):
  FE (power-law c=0.3) →
  Mic:  enc1(2→32) → enc2(32→40) → [concat aligned far] →
        enc3(80→40) → enc4(40→40) → enc5(40→40) →
  Far:  enc_far1(2→32) → enc_far2(32→40) → AlignBlock(hidden=32) →
  S4D Bottleneck (diagonal SSM) →
  Dec:  dec5(40→40) → dec4(40→40) → dec3(40→40) → dec2(40→32) → dec1(32→27) →
  CCM (3×3 complex convolving mask) → enhanced spectrum

Usage (YAML):
  model:
    type: LocalVQE
    args:
      n_fft: 512
      align_delay: 64
      mic_channels: [2, 32, 40, 40, 40, 40]
      far_channels: [2, 32, 40]
      align_hidden: 32
      compression: 0.3
      kernel_size: [4, 4]
      bottleneck_hidden: 162
      arch_version: 3

Run directly to test params and compute:
  python tasks/aec/models/discriminative/localvqe.py [--sr 16000] [--dur 1.0] [--device cpu]

Author: Xincheng Zhu
"""

from __future__ import annotations

import math
import sys as _sys
import os as _os

if __name__ == "__main__":
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", ".."))

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.utils.registry import MODEL_REGISTRY


# ═══════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════

def _causal_pad(kernel_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Compute ZeroPad2d args for causal convolution.

    Time: pad KH-1 on top (past only).
    Freq: symmetric-ish, preserves size at stride=1, halves at stride=2.
    Returns (freq_left, freq_right, time_top, time_bottom).
    """
    kh, kw = kernel_size
    pad_left = (kw - 1) // 2
    pad_right = kw - 1 - pad_left
    return (pad_left, pad_right, kh - 1, 0)


def _compute_freq_progression(n_freqs: int, kernel_size: Tuple[int, int], n_stages: int = 5) -> List[int]:
    """Compute frequency bin count at each encoder stage.

    Each encoder block halves frequency via stride=2 with causal padding.
    Returns list of length n_stages+1: [f0, f1, f2, f3, f4, f5].
    """
    _kh, kw = kernel_size
    pad_left = (kw - 1) // 2
    pad_right = kw - 1 - pad_left
    f = n_freqs
    freqs = [f]
    for _ in range(n_stages):
        f = (f + pad_left + pad_right - kw) // 2 + 1
        freqs.append(f)
    return freqs


def _arch_activation(arch_version: int) -> nn.Module:
    """ReLU6 for arch_version=2 (v1.1), SiLU for arch_version=3 (v1.2)."""
    if arch_version == 2:
        return nn.ReLU6()
    if arch_version == 3:
        return nn.SiLU()
    raise ValueError(f"Unsupported arch_version={arch_version}")


# ═══════════════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════════════

class CausalGroupNorm(nn.Module):
    """GroupNorm-like norm that is causal across the time axis.

    nn.GroupNorm(C, C) on a (B, C, T, F) tensor reduces across (C, T, F)
    per sample, making the normalization at frame t depend on frames
    > t — fatal for streaming inference.  CausalGroupNorm reduces across
    (C, F) per (B, T) instead, so each frame's stats depend only on
    that frame.  Equivalent to nn.LayerNorm([C, F]) when F is fixed,
    but F changes across encoder/decoder stages so we compute the
    reduction explicitly.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"CausalGroupNorm expects 4-D input, got ndim={x.ndim}")
        mean = x.mean(dim=(1, 3), keepdim=True)
        var = x.var(dim=(1, 3), keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class _FE(nn.Module):
    """Power-law feature extraction.

    Compresses magnitude via power-law, preserving phase direction.
    Output has same real/imag structure but with compressed magnitudes.
    Layout: (B, F, T, 2) → (B, 2, T, F).
    """

    def __init__(self, c: float = 0.3):
        super().__init__()
        self.c = c

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, F, T, 2) → (B, 2, T, F)"""
        x_mag = torch.sqrt(x[..., [0]] ** 2 + x[..., [1]] ** 2 + 1e-12)
        x_c = torch.div(x, x_mag.pow(1 - self.c) + 1e-12)
        return x_c.permute(0, 3, 2, 1).contiguous()


class _ResidualBlock(nn.Module):
    """norm → pad → conv → act + skip（pre-norm；act=ReLU6 for v2, SiLU for v3）。"""

    def __init__(self, channels: int, kernel_size: Tuple[int, int] = (4, 4), arch_version: int = 3):
        super().__init__()
        self.arch_version = arch_version
        self.pad = nn.ZeroPad2d(_causal_pad(kernel_size))
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size)
        self.norm = CausalGroupNorm(channels)
        self.act = _arch_activation(arch_version)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv(self.pad(self.norm(x)))) + x


class _EncoderBlock(nn.Module):
    """norm → pad → conv (stride=(1,2)) → act → ResidualBlock。

    Pre-norm normalizes the INPUT before padding and convolution, which is both
    more stable and streaming-compatible.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (4, 4),
        stride: Tuple[int, int] = (1, 2),
        arch_version: int = 3,
    ):
        super().__init__()
        self.arch_version = arch_version
        self.pad = nn.ZeroPad2d(_causal_pad(kernel_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        self.norm = CausalGroupNorm(in_channels)
        self.act = _arch_activation(arch_version)
        self.resblock = _ResidualBlock(out_channels, kernel_size=kernel_size, arch_version=arch_version)

    def forward(self, x: Tensor) -> Tensor:
        return self.resblock(self.act(self.conv(self.pad(self.norm(x)))))


class _SubpixelConv2d(nn.Module):
    """norm → pad → conv → reshape (×2 freq)（pre-norm）。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (4, 4),
                 arch_version: int = 3):
        super().__init__()
        self.arch_version = arch_version
        if arch_version not in (2, 3):
            raise ValueError(f"Unsupported arch_version={arch_version}")
        self.norm = CausalGroupNorm(in_channels)
        self.pad = nn.ZeroPad2d(_causal_pad(kernel_size))
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(self.pad(self.norm(x)))
        batch, doubled_channels, time, frequency = y.shape
        channels = doubled_channels // 2
        return (
            y.reshape(batch, 2, channels, time, frequency)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
            .reshape(batch, channels, time, frequency * 2)
        )


class _DecoderBlock(nn.Module):
    """skip_norm → skip_conv → +x → resblock → deconv → act (if not last)。

    Act is ReLU6 for v2 and SiLU for v3.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (4, 4),
        is_last: bool = False,
        enc_channels: int | None = None,
        arch_version: int = 3,
    ):
        super().__init__()
        self.arch_version = arch_version
        if enc_channels is None:
            enc_channels = in_channels
        if arch_version not in (2, 3):
            raise ValueError(f"Unsupported arch_version={arch_version}")
        self.skip_norm = CausalGroupNorm(enc_channels)
        self.skip_conv = nn.Conv2d(enc_channels, in_channels, 1)
        self.resblock = _ResidualBlock(in_channels, kernel_size=kernel_size, arch_version=arch_version)
        self.deconv = _SubpixelConv2d(in_channels, out_channels, kernel_size, arch_version=arch_version)
        self.is_last = is_last
        if not is_last:
            self.act = _arch_activation(arch_version)

    def forward(self, x: Tensor, x_en: Tensor) -> Tensor:
        x_en = self.skip_norm(x_en)
        y = x + self.skip_conv(x_en)
        y = self.deconv(self.resblock(y))
        if not self.is_last:
            y = self.act(y)
        return y


class _S4DBottleneck(nn.Module):
    """Diagonal State Space Model (S4D) bottleneck.

    Complex diagonal SSM with polar parameterization for guaranteed stability.
    A = r * exp(j*theta) where r = exp(-softplus(rate)) < 1.
    Each state channel is independently prunable since A is diagonal.

    Reference: Saon et al., "Diagonal State Spaces are as Effective as
    Structured State Spaces" (NeurIPS 2022).
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input/output projections
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, input_size)

        # Diagonal A in polar form: A = r * exp(j*theta)
        # r = exp(-softplus(log_rate)) guarantees r in (0, 1)
        self.A_log_rate = nn.Parameter(torch.empty(hidden_size))
        self.A_theta = nn.Parameter(torch.empty(hidden_size))

        # Per-channel input coupling B = B_real + j * B_imag
        self.B_real = nn.Parameter(torch.empty(hidden_size))
        self.B_imag = nn.Parameter(torch.empty(hidden_size))

        # Per-channel output coupling C = C_real + j * C_imag
        self.C_real = nn.Parameter(torch.empty(hidden_size))
        self.C_imag = nn.Parameter(torch.empty(hidden_size))

        # Skip connection
        self.D = nn.Parameter(torch.empty(input_size))

        self._init_parameters()

    def _init_parameters(self):
        # log_rate: uniform so softplus(log_rate) covers ~0.001 to ~7
        # => r = exp(-softplus) covers ~0.999 (slow decay) to ~0.001 (fast decay)
        nn.init.uniform_(self.A_log_rate, -6.0, 2.0)
        # theta: uniform frequency coverage [0, pi]
        with torch.no_grad():
            self.A_theta.copy_(torch.linspace(0, math.pi, self.hidden_size))
        # B, C: small random
        nn.init.normal_(self.B_real, std=0.01)
        nn.init.normal_(self.B_imag, std=0.01)
        nn.init.normal_(self.C_real, std=0.01)
        nn.init.normal_(self.C_imag, std=0.01)
        # D: identity skip
        nn.init.ones_(self.D)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, T, F) → (B, C, T, F)"""
        B_dim, C, T, F_in = x.shape
        u = x.permute(0, 2, 1, 3).reshape(B_dim, T, C * F_in)  # (B, T, H)

        v = self.input_proj(u)  # (B, T, N)

        # Polar → Cartesian: A = r * (cos(theta) + j*sin(theta))
        # Clamp softplus output to ensure minimum decay rate (prevents r → 1 instability)
        sp = F.softplus(self.A_log_rate).clamp(min=0.01)
        r = torch.exp(-sp)  # (N,) in (0, ~0.99)
        a_real = r * torch.cos(self.A_theta)  # (N,)
        a_imag = r * torch.sin(self.A_theta)  # (N,)

        # Run recurrence in float32 for numerical stability under AMP
        v_f = v.float()  # (B, T, N)
        h_real = torch.zeros(B_dim, self.hidden_size, dtype=torch.float32, device=v.device)
        h_imag = torch.zeros(B_dim, self.hidden_size, dtype=torch.float32, device=v.device)
        y = torch.empty(B_dim, T, self.hidden_size, dtype=torch.float32, device=v.device)
        a_real_f = a_real.float()
        a_imag_f = a_imag.float()
        B_real_f = self.B_real.float()
        B_imag_f = self.B_imag.float()
        C_real_f = self.C_real.float()
        C_imag_f = self.C_imag.float()

        for t in range(T):
            v_t = v_f[:, t, :]  # (B, N)
            h_real_new = a_real_f * h_real - a_imag_f * h_imag + B_real_f * v_t
            h_imag_new = a_real_f * h_imag + a_imag_f * h_real + B_imag_f * v_t
            h_real = h_real_new
            h_imag = h_imag_new
            y[:, t, :] = C_real_f * h_real - C_imag_f * h_imag

        y = y.to(u.dtype)  # back to original dtype for projections
        out = self.output_proj(y) + self.D * u  # (B, T, H)
        return out.reshape(B_dim, T, C, F_in).permute(0, 2, 1, 3).contiguous()


class _AlignBlock(nn.Module):
    """Cross-attention soft delay alignment.

    Estimates a delay distribution over [0, dmax) for each frame, then
    applies it to align the reference signal to the microphone signal.

    Key difference from DeepVQE's AlignBlock:
      - Q/K projections use `hidden_channels` (e.g. 32), but the weighted
        sum operates on the full `in_channels` (e.g. 40) reference features.
      - Temperature parameter for sharper/fuzzier delay distributions.
      - Dot product scaled by √F to prevent softmax saturation.
    """

    def __init__(self, in_channels: int, hidden_channels: int, dmax: int = 64, temperature: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dmax = dmax
        self.register_buffer("temperature", torch.tensor(float(temperature)))

        # Pointwise projections for Q and K (into lower-dimensional space)
        self.pconv_mic = nn.Conv2d(in_channels, hidden_channels, 1)
        self.pconv_ref = nn.Conv2d(in_channels, hidden_channels, 1)

        # Causal unfold: pad dmax-1 frames on top (past), zero on bottom
        self.unfold_k = nn.Sequential(
            nn.ZeroPad2d([0, 0, dmax - 1, 0]),
            nn.Unfold((dmax, 1)),
        )

        # Smoothing conv over the similarity scores (causal in time)
        self.conv = nn.Sequential(
            nn.ZeroPad2d([1, 1, 4, 0]),  # causal time, symmetric freq
            nn.Conv2d(hidden_channels, 1, (5, 3)),
        )

    def fold_temperature(self, temperature: float | None = None):
        """Bake the AlignBlock softmax temperature into the smoothing-conv weights.

        Call after ``load_state_dict``.  With no argument, reads from
        ``self.temperature`` (a buffer carried by the checkpoint).
        The original GGML graph has no temperature parameter, so PyTorch
        inference must call this to match its behavior.
        """
        t = float(temperature if temperature is not None else self.temperature)
        if t == 1.0:
            return
        with torch.no_grad():
            self.conv[1].weight.div_(t)
            self.conv[1].bias.div_(t)
            self.temperature.fill_(1.0)

    def forward(self, x_mic: Tensor, x_ref: Tensor, return_delay: bool = False):
        """x_mic, x_ref: (B, C, T, F) → aligned: (B, C, T, F)"""
        # Force FP32: the dot-product sum over frequency bins can overflow
        # in FP16 (hidden=32 × F=257 ≈ 8200 terms, easily > 65504).
        _need_fp32 = (x_mic.dtype != torch.float32)
        if _need_fp32:
            x_mic = x_mic.float()
            x_ref = x_ref.float()

        B, C, T, F_in = x_ref.shape

        # Compute Q and K in lower-dimensional space
        Q = self.pconv_mic(x_mic)  # (B, H, T, F)
        K = self.pconv_ref(x_ref)  # (B, H, T, F)

        # Unfold K along time: creates dmax delayed copies
        Ku = self.unfold_k(K)  # (B, H*dmax, T*F)
        Ku = Ku.view(B, self.hidden_channels, self.dmax, T, F_in)
        Ku = Ku.permute(0, 1, 3, 2, 4).contiguous()  # (B, H, T, dmax, F)

        # Cross-attention similarity: sum over frequency, scaled by √F
        V = torch.sum(Q.unsqueeze(-2) * Ku, dim=-1)  # (B, H, T, dmax)
        V = V / math.sqrt(F_in)

        # Smooth and reduce to single-head attention
        V = self.conv(V)  # (B, 1, T, dmax)

        # Softmax over delay dimension
        A = torch.softmax(V / self.temperature, dim=-1)  # (B, 1, T, dmax)

        # Unfold x_ref (full channels) for the weighted sum
        # NOTE: uses in_channels (C), not hidden_channels — this is the
        # key difference from DeepVQE's AlignBlock.
        unfold_ref = F.pad(x_ref, [0, 0, self.dmax - 1, 0])
        ref_unfolded = unfold_ref.unfold(2, self.dmax, 1)  # (B, C, T, F, dmax)
        ref_unfolded = ref_unfolded.permute(0, 1, 2, 4, 3).contiguous()  # (B, C, T, dmax, F)

        # Weighted sum using attention weights
        A_expanded = A[:, :, :, :, None]  # (B, 1, T, dmax, 1)
        aligned = torch.sum(ref_unfolded * A_expanded, dim=-2)  # (B, C, T, F)

        if return_delay:
            return aligned, A.squeeze(1)  # (B, T, dmax)
        return aligned


class _CCM(nn.Module):
    """Complex Convolving Mask — real-valued implementation.

    Decomposes the cube-root-of-unity basis into explicit real/imag parts.
    Avoids torch.complex for ONNX/GGML compatibility.

    Input mask: 27 channels = 3 (cube-root basis) × 9 (3×3 convolution kernel).
    Output: complex-valued enhanced STFT via real-valued arithmetic.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "v_real",
            torch.tensor([1.0, -0.5, -0.5], dtype=torch.float32),
        )
        self.register_buffer(
            "v_imag",
            torch.tensor([0.0, math.sqrt(3) / 2, -math.sqrt(3) / 2], dtype=torch.float32),
        )
        self.unfold = nn.Sequential(
            nn.ZeroPad2d([1, 1, 2, 0]),  # causal: pad top only
            nn.Unfold(kernel_size=(3, 3)),
        )

    def forward(self, m: Tensor, x: Tensor) -> Tensor:
        """
        m: (B, 27, T, F) — mask from decoder
        x: (B, F, T, 2)  — input spectrum (real, imag)
        Returns: (B, F, T, 2) — enhanced spectrum
        """
        # Force FP32 for numerical stability under AMP
        _need_fp32 = (m.dtype != torch.float32)
        if _need_fp32:
            m = m.float()
            x = x.float()

        # Decompose mask into real/imag via cube-root basis
        batch, _, time, frequency = m.shape
        m = m.reshape(batch, 3, 9, time, frequency)
        H_real = torch.sum(self.v_real[None, :, None, None, None] * m, dim=1)  # (B, 9, T, F)
        H_imag = torch.sum(self.v_imag[None, :, None, None, None] * m, dim=1)  # (B, 9, T, F)

        M_real = H_real.reshape(batch, 3, 3, time, frequency)
        M_imag = H_imag.reshape(batch, 3, 3, time, frequency)

        # Unfold input spectrum
        x_perm = x.permute(0, 3, 2, 1).contiguous()  # (B, 2, T, F)
        x_unfold = self.unfold(x_perm)
        x_unfold = x_unfold.reshape(batch, 2, 3, 3, time, frequency)

        # Complex multiplication via real-valued arithmetic
        x_enh_real = torch.sum(M_real * x_unfold[:, 0] - M_imag * x_unfold[:, 1], dim=(1, 2))  # (B, T, F)
        x_enh_imag = torch.sum(M_real * x_unfold[:, 1] + M_imag * x_unfold[:, 0], dim=(1, 2))  # (B, T, F)

        x_enh = torch.stack([x_enh_real, x_enh_imag], dim=3).transpose(1, 2).contiguous()  # (B, F, T, 2)
        return x_enh


# ═══════════════════════════════════════════════════════════════════════
# LocalVQE model
# ═══════════════════════════════════════════════════════════════════════

@MODEL_REGISTRY.register("LocalVQE")
class LocalVQE(nn.Module):
    """LocalVQE: joint AEC / NS / dereverberation with S4D bottleneck.

    Spectrum-in / spectrum-out: accepts mic_spectra, farend_spectra (B, F, T, 2).
    STFT/iSTFT and window are done by the trainer (YAML ``stft`` block).

    Component order:
      FE (power-law) → mic encoder (5 blocks) + far-end encoder (2 blocks)
      → AlignBlock (soft delay cross-attention) → fused encoder block
      → S4D bottleneck → decoder (5 sub-pixel blocks) → mask head
      → CCM (3×3 complex convolving mask, real-valued) → enhanced spectrum.

    Args:
        n_fft: STFT size (from YAML stft block).  ``n_fft=512`` produces a
            257-bin one-sided STFT.
        align_delay: Number of AlignBlock delay candidates including the current
            frame.  With 64 candidates, the maximum lag is 63 frames.
        mic_channels: Mic encoder channel progression [in, enc1, ..., enc5].
        far_channels: Far-end encoder channel progression [in, enc1, enc2].
        align_hidden: Hidden channels for AlignBlock Q/K projections.
        compression: Power-law compression exponent (c=0.3).
        kernel_size: Conv2d kernel size (kh, kw).  Default (4, 4).
        bottleneck_hidden: S4D hidden state size.  Default 162 matches original.
        arch_version: 2 (ReLU6) or 3 (SiLU).  This selects block/activation
            structure only and does not imply official checkpoint compatibility.
    """

    aec_input_capabilities = frozenset({"mic_spectra", "farend_spectra"})

    def __init__(
        self,
        n_fft: int = 512,
        align_delay: int = 64,
        mic_channels: List[int] | None = None,
        far_channels: List[int] | None = None,
        align_hidden: int = 32,
        compression: float = 0.3,
        kernel_size: Tuple[int, int] | List[int] = (4, 4),
        bottleneck_hidden: int = 162,
        arch_version: int = 3,
        **kwargs,
    ):
        super().__init__()
        if mic_channels is None:
            mic_channels = [2, 32, 40, 40, 40, 40]
        if far_channels is None:
            far_channels = [2, 32, 40]

        self.n_fft = n_fft
        self.n_freqs = n_fft // 2 + 1
        self.align_delay = align_delay
        self.arch_version = arch_version
        ks = tuple(kernel_size)
        av = arch_version

        # 保存图外 STFT 契约供 streaming runtime metadata 使用。
        self.hop_length = kwargs.pop("hop_length", n_fft // 2)
        self.win_length = kwargs.pop("win_length", n_fft)
        self.window = kwargs.pop("window", "hann")
        self.center = kwargs.pop("center", True)

        # Reject unknown kwargs
        if kwargs:
            raise TypeError(f"LocalVQE got unexpected keyword arguments: {sorted(kwargs)}")
        int_params = {
            "n_fft": n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "align_delay": align_delay,
            "align_hidden": align_hidden,
            "bottleneck_hidden": bottleneck_hidden,
        }
        invalid = {
            name: value for name, value in int_params.items()
            if not isinstance(value, int) or isinstance(value, bool) or value < 1
        }
        if invalid:
            raise ValueError(f"LocalVQE expects positive integer parameters, got {invalid}")
        if not self.hop_length <= self.win_length <= n_fft:
            raise ValueError("LocalVQE requires hop_length <= win_length <= n_fft")
        if not isinstance(self.window, str) or self.window.lower() != "hann":
            raise ValueError(f"LocalVQE supports only window='hann', got {self.window!r}")
        if not isinstance(self.center, bool):
            raise TypeError(f"LocalVQE center must be bool, got {self.center!r}")
        if isinstance(compression, bool) or not isinstance(compression, (float, int)):
            raise TypeError(f"LocalVQE compression must be numeric in (0, 1], got {compression!r}")
        if not 0 < compression <= 1:
            raise ValueError(f"LocalVQE compression must be in (0, 1], got {compression!r}")
        if arch_version not in (2, 3):
            raise ValueError(f"LocalVQE arch_version must be 2 or 3, got {arch_version!r}")
        if len(ks) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in ks
        ):
            raise ValueError(f"LocalVQE kernel_size must contain two positive integers, got {kernel_size!r}")
        if len(mic_channels) != 6 or len(far_channels) != 3:
            raise ValueError("LocalVQE requires mic_channels length 6 and far_channels length 3")
        for name, channels in (("mic_channels", mic_channels), ("far_channels", far_channels)):
            if any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in channels
            ):
                raise ValueError(f"LocalVQE {name} must contain positive integers, got {channels!r}")
        if mic_channels[0] != 2 or far_channels[0] != 2:
            raise ValueError("LocalVQE mic_channels/far_channels must start with complex input size 2")
        if far_channels[2] != mic_channels[2]:
            raise ValueError("LocalVQE far_channels[2] must equal mic_channels[2] for alignment")

        # Feature extraction (power-law compression)
        self.fe_mic = _FE(c=compression)
        self.fe_ref = _FE(c=compression)

        # ── Mic encoder blocks 1-2 (pre-alignment) ─────────────────────
        self.mic_enc1 = _EncoderBlock(mic_channels[0], mic_channels[1], kernel_size=ks, arch_version=av)
        self.mic_enc2 = _EncoderBlock(mic_channels[1], mic_channels[2], kernel_size=ks, arch_version=av)

        # ── Far-end encoder blocks 1-2 ─────────────────────────────────
        self.far_enc1 = _EncoderBlock(far_channels[0], far_channels[1], kernel_size=ks, arch_version=av)
        self.far_enc2 = _EncoderBlock(far_channels[1], far_channels[2], kernel_size=ks, arch_version=av)

        # ── Alignment ───────────────────────────────────────────────────
        self.align = _AlignBlock(
            in_channels=mic_channels[2],
            hidden_channels=align_hidden,
            dmax=align_delay,
        )

        # ── Mic encoder blocks 3-5 (post-alignment) ────────────────────
        # enc3 input = concat(mic_enc2, aligned_far) → 2× channels
        self.mic_enc3 = _EncoderBlock(mic_channels[2] * 2, mic_channels[3], kernel_size=ks, arch_version=av)
        self.mic_enc4 = _EncoderBlock(mic_channels[3], mic_channels[4], kernel_size=ks, arch_version=av)
        self.mic_enc5 = _EncoderBlock(mic_channels[4], mic_channels[5], kernel_size=ks, arch_version=av)

        # ── S4D Bottleneck ──────────────────────────────────────────────
        freqs = _compute_freq_progression(self.n_freqs, ks)
        bn_input = mic_channels[5] * freqs[5]
        bn_hidden = bottleneck_hidden if bottleneck_hidden > 0 else bn_input // 2
        self.bottleneck = _S4DBottleneck(bn_input, bn_hidden)

        # ── Decoder blocks (mirror encoder) ─────────────────────────────
        self.dec5 = _DecoderBlock(mic_channels[5], mic_channels[4], kernel_size=ks, arch_version=av)
        self.dec4 = _DecoderBlock(mic_channels[4], mic_channels[3], kernel_size=ks, arch_version=av)
        self.dec3 = _DecoderBlock(mic_channels[3], mic_channels[2], kernel_size=ks, arch_version=av)
        self.dec2 = _DecoderBlock(mic_channels[2], mic_channels[1], kernel_size=ks, arch_version=av)
        self.dec1 = _DecoderBlock(mic_channels[1], 27, kernel_size=ks, is_last=True, arch_version=av)

        # ── CCM mask head ───────────────────────────────────────────────
        self.ccm = _CCM()
        self._init_ccm_identity()

    def _init_ccm_identity(self):
        """Initialize dec1 deconv bias so the CCM mask starts as identity (passthrough).

        The 27-ch mask is reshaped as (3 basis, 9 kernel).  Basis vectors
        v_real=[1,-0.5,-0.5] and v_imag=[0,√3/2,-√3/2] sum to zero, so
        default init (similar values across the 3 groups) produces near-zero
        mask magnitude.  Fix: set the current-frame kernel element of the
        first basis (r=0, v_real=1, v_imag=0) to 1, giving H_real[center]=1.

        CCM uses causal ZeroPad2d([1,1,2,0]) with 3×3 kernel:
          m=0: t-2, m=1: t-1, m=2: t (current frame)
          n=0: f-1, n=1: f (current freq), n=2: f+1
        So current (t, f) = kernel index 2*3+1 = 7.

        SubpixelConv2d stores 54 channels (27×2 for sub-pixel shuffle).
        Output channel c comes from conv channels c (even freq) and c+27
        (odd freq), so we set bias[7] = bias[34] = 1.
        """
        conv = self.dec1.deconv.conv
        with torch.no_grad():
            conv.bias.zero_()
            conv.bias[7] = 1.0   # r=0, current (t,f), even freq bins
            conv.bias[34] = 1.0  # r=0, current (t,f), odd freq bins

    # ── Forward ─────────────────────────────────────────────────────────
    def forward_spectra(self, mic_spec: Tensor, far_spec: Tensor) -> Tensor:
        """Spectra-domain forward pass.

        Args:
            mic_spec: (B, F, T, 2) microphone spectrum (real/imag).
            far_spec:  (B, F, T, 2) far-end reference spectrum.
        Returns:
            nearend_est: (B, F, T, 2) estimated near-end (speech) spectrum.
        """
        for name, spec in (("mic_spec", mic_spec), ("far_spec", far_spec)):
            if not isinstance(spec, Tensor):
                raise TypeError(f"LocalVQE {name} must be a torch.Tensor, got {type(spec).__name__}")
            if spec.ndim != 4 or spec.shape[-1] != 2:
                raise ValueError(
                    f"LocalVQE expects {name} (B, F, T, 2), got {tuple(spec.shape)}"
                )
            if not spec.is_floating_point():
                raise TypeError(f"LocalVQE {name} must use a floating dtype, got {spec.dtype}")
        if mic_spec.dtype != far_spec.dtype:
            raise TypeError(
                f"LocalVQE mic_spec and far_spec dtypes must match, got "
                f"{mic_spec.dtype} and {far_spec.dtype}"
            )
        if mic_spec.device != far_spec.device:
            raise ValueError(
                f"LocalVQE mic_spec and far_spec devices must match, got "
                f"{mic_spec.device} and {far_spec.device}"
            )
        if far_spec.shape != mic_spec.shape:
            raise ValueError(
                f"LocalVQE requires far_spec shape to exactly match mic_spec, "
                f"got {far_spec.shape} and {mic_spec.shape}"
            )
        if mic_spec.shape[1] != self.n_freqs:
            raise ValueError(
                f"LocalVQE n_freqs={self.n_freqs} (n_fft={self.n_fft}), "
                f"but mic_spec has F={mic_spec.shape[1]}. "
                f"Check that the YAML stft.n_fft matches the model's n_fft."
            )
        model_param = next(self.parameters())
        if mic_spec.dtype != model_param.dtype:
            raise TypeError(
                f"LocalVQE spectra dtype must match model parameters, got "
                f"{mic_spec.dtype} and {model_param.dtype}"
            )
        if mic_spec.device != model_param.device:
            raise ValueError(
                f"LocalVQE spectra device must match model parameters, got "
                f"{mic_spec.device} and {model_param.device}"
            )

        # ── Feature extraction ──────────────────────────────────────────
        mic_fe = self.fe_mic(mic_spec)  # (B, 2, T, F)
        ref_fe = self.fe_ref(far_spec)  # (B, 2, T, F)

        # ── Mic encoder 1-2 (pre-alignment) ─────────────────────────────
        mic_e1 = self.mic_enc1(mic_fe)   # (B, 32, T, F1)
        mic_e2 = self.mic_enc2(mic_e1)   # (B, 40, T, F2)

        # ── Far-end encoder 1-2 ─────────────────────────────────────────
        far_e1 = self.far_enc1(ref_fe)   # (B, 32, T, F1)
        far_e2 = self.far_enc2(far_e1)   # (B, 40, T, F2)

        # ── Alignment ───────────────────────────────────────────────────
        aligned_far = self.align(mic_e2, far_e2)  # (B, 40, T, F2)

        # ── Concat mic + aligned far-end ────────────────────────────────
        concat = torch.cat([mic_e2, aligned_far], dim=1)  # (B, 80, T, F2)

        # ── Mic encoder 3-5 (post-alignment) ────────────────────────────
        mic_e3 = self.mic_enc3(concat)  # (B, 40, T, F3)
        mic_e4 = self.mic_enc4(mic_e3)  # (B, 40, T, F4)
        mic_e5 = self.mic_enc5(mic_e4)  # (B, 40, T, F5)

        # ── S4D Bottleneck ──────────────────────────────────────────────
        bn = self.bottleneck(mic_e5)  # (B, 40, T, F5)

        # ── Decoder with skip connections ───────────────────────────────
        # Trim freq to match encoder output (off-by-one from sub-pixel upsampling)
        d5 = self.dec5(bn, mic_e5)[..., :mic_e4.shape[-1]]    # (B, 40, T, F4)
        d4 = self.dec4(d5, mic_e4)[..., :mic_e3.shape[-1]]    # (B, 40, T, F3)
        d3 = self.dec3(d4, mic_e3)[..., :mic_e2.shape[-1]]    # (B, 40, T, F2)
        d2 = self.dec2(d3, mic_e2)[..., :mic_e1.shape[-1]]    # (B, 32, T, F1)
        d1 = self.dec1(d2, mic_e1)[..., :mic_fe.shape[-1]]    # (B, 27, T, F)

        # ── CCM: produce enhanced near-end spectrum ─────────────────────
        return self.ccm(d1, mic_spec)  # (B, F, T, 2)

    def forward(self, batch: dict) -> dict:
        """Dict-based forward (compatible with AEC training loop).

        Expected keys: mic_spectra, farend_spectra (B, F, T, 2).
        Sets: nearend_estimate_spectra.
        """
        batch["nearend_estimate_spectra"] = self.forward_spectra(
            batch["mic_spectra"], batch["farend_spectra"]
        )
        return batch


# ═══════════════════════════════════════════════════════════════════════
# Profiling block
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from core.utils.model_profile import profile_model, make_profile_parser
    from core.stft import build_stft_istft

    args = make_profile_parser(default_sr=16000).parse_args()

    STFT_CFG = {
        "n_fft": 512,
        "hop_length": 256,
        "win_length": 512,
        "window": "hann",
    }
    model = LocalVQE(n_fft=STFT_CFG["n_fft"])
    stft_module = build_stft_istft(STFT_CFG)

    class _Wrapper(nn.Module):
        def __init__(self, net, stft_m):
            super().__init__()
            self.net, self.stft = net, stft_m

        def forward(self, wav):
            # For profiling we duplicate the mono input as both mic and far.
            mic_spec = self.stft(wav)
            far_spec = self.stft(wav)
            out_spec = self.net.forward_spectra(mic_spec, far_spec)
            return self.stft.inverse(out_spec, wav.shape[-1])

    wrapped = _Wrapper(model, stft_module)
    print("\n[Config] STFT:", STFT_CFG, "| sample_rate:", args.sr)
    profile_model(
        wrapped,
        sample_rate=args.sr,
        duration=args.dur,
        batch_size=args.batch,
        device=args.device,
        repeats=args.repeats,
        label="LocalVQE (AEC, mic+far dual-input, S4D bottleneck)",
    )
