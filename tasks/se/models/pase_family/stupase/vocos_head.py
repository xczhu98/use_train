# Copyright (c) 2025 jishengpeng. MIT license.
# Source: https://github.com/jishengpeng/WavTokenizer
# Ported into use_train framework for StuPASE reproduction.
#
# 数值行为不同于 tasks/se/models/pase_family/common/vocos/head.py
# (PASE's head): the log-magnitude clamp range differs (-20,5 here vs an
# exp-magnitude clamp of max=1e2 in PASE's port), kept as a separate file to
# preserve StuPASE's exact numerics.

import torch
import torch.nn as nn


class ISTFT(nn.Module):
    """Custom ISTFT supporting 'same' padding (no center requirement).

    Args:
        n_fft: FFT size.
        hop_length: Hop between frames.
        win_length: Window length.
        padding: 'center' (delegate to torch.istft) or 'same'.
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.padding == "center":
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
        pad = (self.win_length - self.hop_length) // 2
        assert spec.dim() == 3, "Expected 3D tensor"
        B, N, T = spec.shape
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft, output_size=(1, output_size),
            kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq, output_size=(1, output_size),
            kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope
        return y


class ISTFTHead(nn.Module):
    """ISTFT Head predicting magnitude and phase from hidden features.

    Args:
        dim: Hidden dimension.
        n_fft: FFT size (also used as win_length).
        hop_length: Should align with the Mel-vocoder's frame rate.
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        """Args: x (B, D, L). Returns: audio (B, T), or pre-ISTFT hidden if return_hidden."""
        x = self.out(x.transpose(1, 2)).transpose(1, 2)
        if return_hidden:
            return x
        mag, p = x.chunk(2, dim=1)
        mag = torch.clip(mag, min=-20, max=5)  # safeguard to prevent excessively large magnitudes
        mag = torch.exp(mag)
        x = torch.cos(p)
        y = torch.sin(p)
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio
