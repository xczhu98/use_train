"""Offline multi-channel SpatialNet for spectrum-domain speech separation."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from core.models.io_contract import ModelIOContract, TensorField
from core.utils.registry import MODEL_REGISTRY


class _SeqLastNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _GroupedLinear(nn.Module):
    def __init__(self, features: int, groups: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(groups, features, features))
        self.bias = nn.Parameter(torch.zeros(groups, features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("...gf,gkf->...gk", x, self.weight) + self.bias


class _SpatialLayer(nn.Module):
    def __init__(
        self, hidden: int, ffn: int, squeeze: int, freqs: int, heads: int,
        kernel: Sequence[int], groups: Sequence[int], dropout: Sequence[float],
        full: nn.Module | None,
    ) -> None:
        super().__init__()
        fg, tg = groups
        self.fconv1 = nn.Sequential(_SeqLastNorm(hidden), nn.Conv1d(hidden, hidden, kernel[0], groups=fg, padding="same"), nn.PReLU(hidden))
        self.fnorm = nn.LayerNorm(hidden)
        self.squeeze = nn.Sequential(nn.Conv1d(hidden, squeeze, 1), nn.SiLU())
        self.full = full if full is not None else _GroupedLinear(freqs, squeeze)
        self.unsqueeze = nn.Sequential(nn.Conv1d(squeeze, hidden, 1), nn.SiLU())
        self.fconv2 = nn.Sequential(_SeqLastNorm(hidden), nn.Conv1d(hidden, hidden, kernel[0], groups=fg, padding="same"), nn.PReLU(hidden))
        self.attn_norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=dropout[0])
        self.ffn_norm = nn.LayerNorm(hidden)
        self.tconv = nn.Sequential(
            nn.Conv1d(hidden, ffn, 1), nn.SiLU(),
            nn.Conv1d(ffn, ffn, kernel[1], groups=tg, padding="same"), nn.SiLU(),
            nn.Conv1d(ffn, ffn, kernel[1], groups=tg, padding="same"), nn.SiLU(),
            nn.Conv1d(ffn, hidden, 1),
        )
        self.dropout = nn.Dropout(dropout[1])

    @staticmethod
    def _conv(block: nn.Module, x: Tensor) -> Tensor:
        b, f, t, h = x.shape
        y = x.permute(0, 2, 3, 1).reshape(b * t, h, f)
        y = block(y).reshape(b, t, h, f).permute(0, 3, 1, 2)
        return y

    def _full(self, x: Tensor) -> Tensor:
        b, f, t, h = x.shape
        y = self.fnorm(x).permute(0, 2, 3, 1).reshape(b * t, h, f)
        y = self.squeeze(y)
        y = self.full(y)
        y = self.unsqueeze(y)
        return y.reshape(b, t, h, f).permute(0, 3, 1, 2)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self._conv(self.fconv1, x)
        x = x + self._full(x)
        x = x + self._conv(self.fconv2, x)
        b, f, t, h = x.shape
        y = self.attn_norm(x).reshape(b * f, t, h)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y.reshape(b, f, t, h)
        y = self.ffn_norm(x).transpose(-1, -2).reshape(b * f, h, t)
        x = x + self.dropout(self.tconv(y).reshape(b, f, h, t).transpose(-1, -2))
        return x


@MODEL_REGISTRY.register("SpatialNet")
class SpatialNet(nn.Module):
    """Offline SpatialNet: ``[B,C,F,T,2]`` mixture to ``[B,K,F,T,2]`` sources."""

    input_domain = "spectrum"
    output_domain = "spectrum"
    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({"ss"}),
        inputs=(TensorField("mixture_spectra", "spectrum", ("batch", "microphone", "frequency", "frame", "complex")),),
        outputs=(TensorField("est_sources_spectra", "spectrum", ("batch", "source", "frequency", "frame", "complex"), multiple=True),),
        forward_kind="estimate",
        frontend_owner="runtime",
    )

    def __init__(
        self, n_channel: int = 6, num_sources: int = 2, ref_channel: int = 0,
        dim_hidden: int = 96, dim_ffn: int = 192, dim_squeeze: int = 8,
        num_layers: int = 8, num_heads: int = 4, num_freqs: int = 129,
        encoder_kernel_size: int = 5, kernel_size: Sequence[int] = (5, 3),
        conv_groups: Sequence[int] = (8, 8),
        dropout: Sequence[float] = (0, 0, 0), full_share: int = 0,
        n_fft: int | None = None, hop_length: int | None = None, win_length: int | None = None,
        window: str | None = None, center: bool | None = None,
    ) -> None:
        super().__init__()
        if type(n_channel) is not int or n_channel < 1:
            raise ValueError("n_channel must be a positive int")
        if type(num_sources) is not int or num_sources < 1:
            raise ValueError("num_sources must be a positive int")
        if type(ref_channel) is not int or not 0 <= ref_channel < n_channel:
            raise ValueError("ref_channel must index an available microphone")
        if n_fft is not None and n_fft // 2 + 1 != num_freqs:
            raise ValueError("SpatialNet num_freqs must match stft.n_fft")
        if hop_length is not None and hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if win_length is not None and win_length <= 0:
            raise ValueError("win_length must be positive")
        if window is not None and window.lower() not in {"hann", "hamming"}:
            raise ValueError("SpatialNet supports hann or hamming windows")
        if center is not None and type(center) is not bool:
            raise TypeError("center must be bool")
        if dim_hidden % conv_groups[0] or dim_ffn % conv_groups[1] or dim_hidden % num_heads:
            raise ValueError("hidden/ffn dimensions must divide convolution groups and heads")
        self.n_channel, self.num_sources, self.ref_channel, self.num_freqs = n_channel, num_sources, ref_channel, num_freqs
        self.encoder = nn.Conv1d(n_channel * 2, dim_hidden, encoder_kernel_size, padding="same")
        layers = []
        full = None
        for index in range(num_layers):
            layer = _SpatialLayer(dim_hidden, dim_ffn, dim_squeeze, num_freqs, num_heads, kernel_size, conv_groups, dropout, full if index > full_share else None)
            if index >= full_share:
                full = layer.full
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.decoder = nn.Linear(dim_hidden, num_sources * 2)

    def forward(self, mixture_spectra: Tensor, lengths=None, *, fs=None):
        del lengths, fs
        if self.n_channel == 1 and mixture_spectra.ndim == 4:
            mixture_spectra = mixture_spectra.unsqueeze(1)
        if mixture_spectra.ndim != 5 or mixture_spectra.shape[1] != self.n_channel or mixture_spectra.shape[-1] != 2:
            raise ValueError("SpatialNet expects mixture_spectra with shape [B, C, F, T, 2]")
        if mixture_spectra.shape[2] != self.num_freqs:
            raise ValueError(f"SpatialNet expected {self.num_freqs} frequency bins, got {mixture_spectra.shape[2]}")
        b, c, f, t, _ = mixture_spectra.shape
        x = mixture_spectra.float().permute(0, 2, 3, 1, 4).reshape(b, f, t, c * 2)
        x = self.encoder(x.reshape(b * f, t, c * 2).transpose(1, 2)).transpose(1, 2).reshape(b, f, t, -1)
        for layer in self.layers:
            x = layer(x)
        output = self.decoder(x).reshape(b, f, t, self.num_sources, 2).permute(0, 3, 1, 2, 4)
        return {"est_sources_spectra": output}
