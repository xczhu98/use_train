"""Offline SpatialNet for multi-channel speech enhancement.

The network is adapted from NBSS SpatialNet as a spectrum-in/spectrum-out
module. STFT/iSTFT remain owned by the framework; the model consumes
``(B, C, F, T, 2)`` and returns the enhanced reference-channel spectrum in
``(B, F, T, 2)``. Mono ``(B, F, T, 2)`` input remains supported for ``C=1``.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from core.models.io_contract import ModelIOContract, TensorField
from core.utils.registry import MODEL_REGISTRY
from tasks.se.models.base_model import BaseSEModel


class _GroupLinear(nn.Module):
    def __init__(self, features: int, groups: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(groups, features, features))
        self.bias = nn.Parameter(torch.zeros(groups, features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("...gf,gkf->...gk", x, self.weight) + self.bias


class _SpatialLayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        ffn: int,
        squeeze: int,
        freqs: int,
        heads: int,
        dropout: Sequence[float],
        kernel_size: Sequence[int],
        conv_groups: Sequence[int],
        norms: Sequence[str],
        full: nn.Module | None,
    ) -> None:
        super().__init__()
        fg, tg = conv_groups
        fk, tk = kernel_size
        self.fconv1 = nn.Sequential(
            self._norm(norms[3], hidden, fg, True),
            nn.Conv1d(hidden, hidden, fk, padding="same", groups=fg),
            nn.PReLU(hidden),
        )
        self.norm_full = self._norm(norms[5], hidden, fg, False)
        self.squeeze = nn.Sequential(nn.Conv1d(hidden, squeeze, 1), nn.SiLU())
        self.full = full if full is not None else _GroupLinear(freqs, squeeze)
        self.unsqueeze = nn.Sequential(nn.Conv1d(squeeze, hidden, 1), nn.SiLU())
        self.fconv2 = nn.Sequential(
            self._norm(norms[4], hidden, fg, True),
            nn.Conv1d(hidden, hidden, fk, padding="same", groups=fg),
            nn.PReLU(hidden),
        )
        self.norm_attn = self._norm(norms[0], hidden, tg, False)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.drop_attn = nn.Dropout(dropout[0])
        self.norm_ffn = self._norm(norms[1], hidden, tg, True)
        self.tconv = nn.Sequential(
            nn.Conv1d(hidden, ffn, 1), nn.SiLU(),
            nn.Conv1d(ffn, ffn, tk, padding="same", groups=tg), nn.SiLU(),
            nn.Conv1d(ffn, ffn, tk, padding="same", groups=tg),
            self._norm(norms[2], ffn, tg, True), nn.SiLU(),
            nn.Conv1d(ffn, ffn, tk, padding="same", groups=tg), nn.SiLU(),
            nn.Conv1d(ffn, hidden, 1),
        )
        self.drop_ffn = nn.Dropout(dropout[1])

    @staticmethod
    def _norm(kind: str, channels: int, groups: int, seq_last: bool) -> nn.Module:
        if kind.upper() == "LN":
            return _LastLayerNorm(channels, seq_last)
        if kind.upper() == "GN":
            return _ChannelGroupNorm(groups, channels, seq_last)
        raise ValueError(f"Unsupported SpatialNet norm: {kind}")

    def _fconv(self, block: nn.Module, x: Tensor) -> Tensor:
        b, f, t, h = x.shape
        y = block(x.permute(0, 2, 3, 1).reshape(b * t, h, f))
        return y.reshape(b, t, h, f).permute(0, 3, 1, 2)

    def _full(self, x: Tensor) -> Tensor:
        b, f, t, h = x.shape
        y = self.norm_full(x).permute(0, 2, 3, 1).reshape(b * t, h, f)
        y = self.squeeze(y)
        y = self.full(y)
        y = self.unsqueeze(y)
        return y.reshape(b, t, h, f).permute(0, 3, 1, 2)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self._fconv(self.fconv1, x)
        x = x + self._full(x)
        x = x + self._fconv(self.fconv2, x)
        b, f, t, h = x.shape
        y = self.norm_attn(x).reshape(b * f, t, h)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop_attn(y.reshape(b, f, t, h))
        y = x.transpose(-1, -2).reshape(b * f, h, t)
        y = self.norm_ffn(y)
        y = self.tconv(y).reshape(b, f, h, t).transpose(-1, -2)
        return x + self.drop_ffn(y)


class _LastLayerNorm(nn.Module):
    def __init__(self, channels: int, seq_last: bool) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.seq_last = seq_last

    def forward(self, x: Tensor) -> Tensor:
        if self.seq_last:
            return self.norm(x.transpose(1, -1)).transpose(1, -1)
        return self.norm(x)


class _ChannelGroupNorm(nn.Module):
    def __init__(self, groups: int, channels: int, seq_last: bool) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.seq_last = seq_last

    def forward(self, x: Tensor) -> Tensor:
        if self.seq_last:
            return self.norm(x)
        return self.norm(x.transpose(1, -1)).transpose(1, -1)


class _SpatialNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_freqs: int, num_layers: int,
                 dim_hidden: int, dim_ffn: int, dim_squeeze: int, num_heads: int,
                 dropout: Sequence[float], kernel_size: Sequence[int],
                 conv_groups: Sequence[int], norms: Sequence[str], full_share: int) -> None:
        super().__init__()
        self.encoder = nn.Conv1d(input_dim, dim_hidden, 5, padding="same")
        layers = []
        shared = None
        for index in range(num_layers):
            layer = _SpatialLayer(dim_hidden, dim_ffn, dim_squeeze, num_freqs, num_heads,
                                  dropout, kernel_size, conv_groups, norms,
                                  shared if index > full_share else None)
            if index >= full_share:
                shared = layer.full
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.decoder = nn.Linear(dim_hidden, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        b, f, t, _ = x.shape
        y = self.encoder(x.reshape(b * f, t, -1).transpose(1, 2)).transpose(1, 2)
        y = y.reshape(b, f, t, -1)
        for layer in self.layers:
            y = layer(y)
        return self.decoder(y)


@MODEL_REGISTRY.register("SpatialNet_SE")
class SpatialNetSE(BaseSEModel):
    """SpatialNet SE: ``[B,C,F,T,2]`` microphones to one reference spectrum."""

    supports_multichannel_input = True
    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({"se"}),
        inputs=(TensorField("noisy_spec", "spectrum", ("batch", "microphone", "frequency", "frame", "complex")),),
        outputs=(TensorField("enhanced_spec", "spectrum", ("batch", "frequency", "frame", "complex")),),
        forward_kind="estimate",
        frontend_owner="runtime",
    )

    def __init__(self, n_fft: int = 512, num_layers: int = 8, dim_hidden: int = 96,
                 dim_ffn: int = 192, dim_squeeze: int = 8, num_heads: int = 4,
                 encoder_kernel_size: int = 5, dropout: Sequence[float] = (0, 0, 0),
                 kernel_size: Sequence[int] = (5, 3), conv_groups: Sequence[int] = (8, 8),
                 norms: Sequence[str] = ("LN", "LN", "GN", "LN", "LN", "LN"),
                 full_share: int = 0, hop_length: int | None = None,
                 win_length: int | None = None, window: str | None = None,
                 center: bool | None = None, n_channel: int = 1,
                 ref_channel: int = 0) -> None:
        super().__init__()
        if encoder_kernel_size != 5:
            raise ValueError("SpatialNetSE currently uses encoder_kernel_size=5")
        if len(norms) != 6 or len(kernel_size) != 2 or len(conv_groups) != 2:
            raise ValueError("Invalid SpatialNet configuration lengths")
        if type(n_channel) is not int or n_channel < 1:
            raise ValueError("n_channel must be a positive int")
        if type(ref_channel) is not int or not 0 <= ref_channel < n_channel:
            raise ValueError("ref_channel must index an available microphone")
        self.n_fft = n_fft
        self.n_channel = n_channel
        self.ref_channel = ref_channel
        self.net = _SpatialNet(n_channel * 2, 2, n_fft // 2 + 1, num_layers, dim_hidden, dim_ffn,
                               dim_squeeze, num_heads, dropout, kernel_size, conv_groups,
                               norms, full_share)

    def forward(self, noisy_spec: Tensor, lengths=None, *, fs: int = None) -> Tensor:
        del lengths, fs
        if noisy_spec.ndim == 4:
            if self.n_channel != 1:
                raise ValueError("SpatialNetSE expects [B,C,F,T,2] for n_channel > 1")
            noisy_spec = noisy_spec.unsqueeze(1)
        if noisy_spec.ndim != 5 or noisy_spec.shape[1] != self.n_channel or noisy_spec.shape[-1] != 2:
            raise ValueError("SpatialNetSE expects spectrum shape (B, C, F, T, 2)")
        expected = self.n_fft // 2 + 1
        if noisy_spec.shape[2] != expected:
            raise ValueError(f"SpatialNetSE expects F={expected}, got {noisy_spec.shape[2]}")
        b, _, f, t, _ = noisy_spec.shape
        x = noisy_spec.float().permute(0, 2, 3, 1, 4).reshape(b, f, t, self.n_channel * 2)
        return self.net(x)
