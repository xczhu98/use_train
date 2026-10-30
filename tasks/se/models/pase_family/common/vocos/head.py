# Copyright (c) 2025 jishengpeng. MIT license.
# Source: https://github.com/jishengpeng/WavTokenizer
# Ported into use_train framework for PASE reproduction.

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
        hop_length: Should align with encoder downsampling (e.g., 320 for WavLM).
        mag_clip: 幅度裁剪策略。"post_exp"（默认）为 PASE/WavTokenizer 原行为
            （先 exp 再裁剪上限 1e2）；"log" 为官方 UniPASE 行为（先把 log-mag
            裁剪到 (-20, 5) 再 exp，见 unipase/models/vocoder/vocos/head.py:111-128）。
            该参数只影响前向数值，不引入新的 state_dict key。
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same",
                 mag_clip: str = "post_exp"):
        super().__init__()
        if mag_clip not in ("post_exp", "log"):
            raise ValueError(f"mag_clip must be 'post_exp' or 'log', got {mag_clip!r}")
        self.mag_clip = mag_clip
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        """Args: x (B, D, L). Returns: audio (B, T)；return_hidden=True 时返回
        chunk 前的 Linear 输出 (B, n_fft+2, L)（对齐官方 UniPASE head；
        官方保留给特征损失用，本框架目前无调用方）。"""
        x = self.out(x.transpose(1, 2)).transpose(1, 2)
        if return_hidden:
            return x
        mag, p = x.chunk(2, dim=1)
        if self.mag_clip == "log":
            # 官方 UniPASE：先裁剪 log-mag 再 exp（防止过大幅度）
            mag = torch.clip(mag, min=-20, max=5)
            mag = torch.exp(mag)
        else:
            # PASE/WavTokenizer 原行为：先 exp 再裁剪上限
            mag = torch.exp(mag)
            mag = torch.clip(mag, max=1e2)
        x = torch.cos(p)
        y = torch.sin(p)
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio
