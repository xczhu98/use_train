"""FastEnhancer - unified speech enhancement model supporting all variants (T/B/S/M/L).

Architecture: StridedConv encoder -> RNNFormer (GRU + Attention) -> ConvTranspose decoder
Spectrum-in / spectrum-out; STFT/iSTFT and window from YAML stft (trainer).

论文: "FastEnhancer: Speed-Optimized Streaming Neural Speech Enhancement",
      Sunghwan Ahn, Jinmo Han, Beom Jun Woo, Nam Soo Kim, arXiv:2509.21867.
官方代码: https://github.com/aask1357/fastenhancer

5 model variants from Table 1 of the paper:
  ┌──────────┬──────┬────────────────────┬────┬────┬────┬────┬─────────┐
  │ Variant  │  H   │   kernel_size      │ L  │ K  │ C1 │ C2 │  F      │
  ├──────────┼──────┼────────────────────┼────┼────┼────┼────┼─────────┤
  │ Tiny (T) │ 256  │ [8, 3, 3]          │  2 │  2 │ 24 │ 20 │  16     │
  │ Base (B) │ 256  │ [8, 3, 3]          │  2 │  3 │ 48 │ 36 │  24     │
  │ Small(S) │ 256  │ [8, 3, 3, 3]       │  3 │  3 │ 64 │ 48 │  36     │
  │ Med. (M) │ 160  │ [8, 3, 3, 3]       │  3 │  4 │ 96 │ 72 │  48     │
  │ Large(L) │ 100  │ [8, 3, 3, 3, 3]    │  4 │  5 │128 │ 96 │  64     │
  └──────────┴──────┴────────────────────┴────┴────┴────┴────┴─────────┘
  H=hop_size, L=num encoder blocks, K=num RNNFormer blocks,
  C1=encoder channels, C2=RNNFormer channels, F=RNNFormer freq

Usage in YAML config:
    model:
      type: FastEnhancer
      args:
        variant: B               # or T, S, M, L
        # any parameter below can override the variant defaults

Run directly to test params and compute:
  python tasks/se/models/fastenhancer.py [--sr 16000] [--dur 1.0] [--device cpu]
"""

from __future__ import annotations
import sys as _sys, os as _os
if __name__ == "__main__":
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))


import math
import typing as tp
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm as weight_norm_fn

from core.utils.registry import MODEL_REGISTRY
from tasks.se.models.base_model import BaseSEModel


# ──────────────────────────────────────────────────────────────
# Variant presets (Table 1 of FastEnhancer paper)
# ──────────────────────────────────────────────────────────────
_VARIANT_PRESETS: dict[str, dict[str, tp.Any]] = {
    "T": dict(
        channels=24, kernel_size=[8, 3, 3], hop_size=256,
        rnnformer_kwargs=dict(num_blocks=2, channels=20, freq=16, num_heads=4),
    ),
    "B": dict(
        channels=48, kernel_size=[8, 3, 3], hop_size=256,
        rnnformer_kwargs=dict(num_blocks=3, channels=36, freq=24, num_heads=4),
    ),
    "S": dict(
        channels=64, kernel_size=[8, 3, 3, 3], hop_size=256,
        rnnformer_kwargs=dict(num_blocks=3, channels=48, freq=36, num_heads=4),
    ),
    "M": dict(
        channels=96, kernel_size=[8, 3, 3, 3], hop_size=160,
        rnnformer_kwargs=dict(num_blocks=4, channels=72, freq=48, num_heads=4),
    ),
    "L": dict(
        channels=128, kernel_size=[8, 3, 3, 3, 3], hop_size=100,
        rnnformer_kwargs=dict(num_blocks=5, channels=96, freq=64, num_heads=4),
    ),
}


# ──────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────
class StridedConv1d(nn.Conv1d):
    """Conv1d where stride is absorbed into channel reshaping (efficient MACs)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
        if stride < 1 or kernel_size % stride != 0:
            raise ValueError(
                f"kernel_size ({kernel_size}) must be a positive multiple of stride ({stride})"
            )
        self._orig_stride = stride
        self._orig_padding = padding
        super().__init__(
            in_channels * stride, out_channels, kernel_size // stride,
            stride=1, padding=0, **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        s = self._orig_stride
        x = F.pad(x, (self._orig_padding, self._orig_padding))
        B, C, T = x.shape
        x = x.view(B, C, T // s, s).permute(0, 3, 1, 2).reshape(B, C * s, T // s)
        return super().forward(x)


class ScaledConvTranspose1d(nn.ConvTranspose1d):
    """ConvTranspose1d with learnable scale and optional weight normalization."""

    def __init__(self, *args, normalize: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize = normalize
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        if self.normalize:
            w = F.normalize(self.weight, dim=(0, 1, 2)).mul_(self.scale)
        else:
            w = self.weight * self.scale
        return F.conv_transpose1d(
            x, w, self.bias, stride=self.stride,
            padding=self.padding, output_padding=self.output_padding,
            groups=self.groups, dilation=self.dilation,
        )


class _CLBatchNorm(nn.BatchNorm1d):
    """BatchNorm operating on channels-last layout [T, B, F, C]."""

    def forward(self, x: Tensor) -> Tensor:
        T, B, F, C = x.shape
        return super().forward(x.view(T * B * F, C, 1)).view(T, B, F, C)


def _positional_embedding(channels: int, freq: int) -> Tensor:
    freq = max(freq, 2)
    f = torch.arange(1, freq + 1, dtype=torch.float32) * (math.pi / freq)
    c = torch.linspace(math.log(1), math.log(max(freq - 1, 1)), channels // 2,
                        dtype=torch.float32).exp()
    grid = f.view(-1, 1) * c.view(1, -1)
    return torch.cat((grid.sin(), grid.cos()), dim=1)


class _Attention(nn.Module):
    def __init__(self, channels: int, num_heads: int, attn_bias: bool = False):
        super().__init__()
        self.head_dim = channels // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(channels, channels * 3, bias=attn_bias)

    def forward(self, x: Tensor) -> Tensor:
        TB, FREQ, _ = x.shape
        qkv = self.qkv(x).reshape(TB, FREQ, self.num_heads, -1).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(TB, FREQ, -1)


class RNNFormerBlock(nn.Module):
    """GRU (time) + Multi-head Attention (frequency) block."""

    def __init__(
        self, channels: int, freq: int, num_heads: int,
        weight_norm: bool = False,
        positional_embedding: str | None = None,
        attn_bias: bool = False, eps: float = 1e-8,
    ):
        super().__init__()
        self.channels = channels
        self.freq = freq
        BN = _CLBatchNorm

        self.rnn = nn.GRU(channels, channels, batch_first=False)
        self.rnn_fc = nn.Linear(channels, channels, bias=False)
        self.rnn_post_norm = BN(channels, eps)

        self.attn = _Attention(channels, num_heads, attn_bias)
        self.attn_fc = nn.Linear(channels, channels, bias=False)
        self.attn_post_norm = BN(channels, eps)

        if positional_embedding is not None:
            pe = _positional_embedding(channels, freq)
            if positional_embedding == "fixed":
                self.register_buffer("pe", pe)
            elif positional_embedding in ("learnable", "train"):
                self.pe = nn.Parameter(pe)
        else:
            self.pe = None

        if weight_norm:
            self.rnn = weight_norm_fn(self.rnn, name="weight_ih_l0")
            self.rnn = weight_norm_fn(self.rnn, name="weight_hh_l0")
            self.attn.qkv = weight_norm_fn(self.attn.qkv)

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x: Tensor, h: Tensor | None) -> tuple[Tensor, Tensor]:
        TIME, BATCH, FREQ, CH = x.shape

        x_in = x
        x = x.view(TIME, FREQ * BATCH, CH)
        x, h = self.rnn(x, h)
        x = x.view(TIME, BATCH, FREQ, CH)
        x = self.rnn_fc(x)
        x = self.rnn_post_norm(x)
        x = x.add_(x_in)

        if self.pe is not None:
            x = x.add_(self.pe)
        x_in = x
        x = x.view(TIME * BATCH, FREQ, CH)
        x = self.attn(x)
        x = x.view(TIME, BATCH, FREQ, CH)
        x = self.attn_fc(x)
        x = self.attn_post_norm(x)
        x = x.add_(x_in)
        return x, h


@dataclass
class _RNNFormerConfig:
    num_blocks: int = 3
    channels: int = 36
    freq: int = 24
    num_heads: int = 4
    eps: float = 1e-5
    positional_embedding: str | None = "learnable"
    attn_bias: bool = False


def _build_freq_linear(n_freq: int, n_filter: int, init: str = "linear_fixed"):
    """Frequency compression/expansion linear layers with triangular filter init."""
    if n_filter < 2:
        raise ValueError(f"n_filter must be >= 2, got {n_filter}")
    pre = nn.Linear(n_freq, n_filter, bias=False)
    post = nn.Linear(n_filter, n_freq, bias=False)

    delta = (n_freq - 1) / (n_filter - 1)
    f_filter = torch.linspace(0, n_freq - 1, n_filter)
    f_freqs = torch.linspace(0, n_freq - 1, n_freq)

    down = f_filter[1:, None] - f_freqs[None, :]
    up = f_freqs[None, :] - f_filter[:-1, None]
    down, up = down / delta, up / delta
    down = F.pad(down, (0, 0, 0, 1), value=1.0)
    up = F.pad(up, (0, 0, 1, 0), value=1.0)
    pre_w = torch.max(up.new_zeros(1), torch.min(down, up))
    pre_w = pre_w / pre_w.sum(dim=1, keepdim=True)
    post_w = pre_w.T / pre_w.T.sum(dim=1, keepdim=True)

    if init.endswith("_fixed"):
        delattr(pre, "weight")
        delattr(post, "weight")
        pre.register_buffer("weight", pre_w.contiguous().clone())
        post.register_buffer("weight", post_w.contiguous().clone())
    else:
        pre.weight.data.copy_(pre_w)
        post.weight.data.copy_(post_w)
    return pre, post


# ──────────────────────────────────────────────────────────────
# FastEnhancer Model (supports T/B/S/M/L variants)
# ──────────────────────────────────────────────────────────────
@MODEL_REGISTRY.register("FastEnhancer")
class FastEnhancer(BaseSEModel):
    """FastEnhancer: RNNFormer-based lightweight speech enhancement.

    Supports all 5 paper variants via ``variant`` parameter or explicit args.
    If ``variant`` is given, its preset values are used as defaults, but any
    explicitly provided argument will override the preset.

    Args:
        variant: One of 'T'/'B'/'S'/'M'/'L' (case-insensitive). Sets defaults
                 for channels, kernel_size, rnnformer_kwargs.
                 None means use parameter defaults directly (equivalent to 'B').
        channels: Encoder/decoder channel width C1.
        kernel_size: List of kernel sizes [enc_pre, enc1, ...].
        stride: Strided conv stride (default 4).
        input_compression: Magnitude compression exponent (default 0.3).
        rnnformer_kwargs: Dict of RNNFormer config overrides
            (num_blocks, channels, freq, num_heads, eps, positional_embedding, attn_bias).
        weight_norm: Apply weight normalization to GRU and attention.
        normalize_final_conv: Apply normalization to final transposed conv.
        pre_post_init: Frequency linear init method.
    """

    def __init__(
        self,
        variant: str | None = None,
        channels: int | None = None,
        kernel_size: list[int] | None = None,
        stride: int = 4,
        input_compression: float = 0.3,
        rnnformer_kwargs: dict[str, tp.Any] | None = None,
        activation: str = "SiLU",
        weight_norm: bool = True,
        normalize_final_conv: bool = True,
        pre_post_init: str = "linear_fixed",
        **kwargs,
    ):
        super().__init__()

        preset = {}
        if variant is not None:
            key = variant.upper()
            if key not in _VARIANT_PRESETS:
                raise ValueError(f"Unknown variant '{variant}'. "
                                 f"Available: {list(_VARIANT_PRESETS.keys())}")
            preset = _VARIANT_PRESETS[key]

        channels = channels if channels is not None else preset.get("channels", 48)
        kernel_size = kernel_size if kernel_size is not None else preset.get("kernel_size", [8, 3, 3])

        preset_rf = preset.get("rnnformer_kwargs", {})
        if rnnformer_kwargs is not None:
            merged_rf = {**preset_rf, **rnnformer_kwargs}
        else:
            merged_rf = preset_rf

        n_fft = kwargs.pop("n_fft", 512)
        for key in ("hop_length", "win_length", "window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"FastEnhancer got unexpected keyword arguments: {sorted(kwargs)}")
        self.input_compression = input_compression
        self.n_fft = n_fft
        self._eps = 1e-5

        rf_cfg = _RNNFormerConfig(**merged_rf)
        self.rf_ch = rf_cfg.channels
        self.rf_freq = rf_cfg.freq

        Act = getattr(nn, activation)
        BN = nn.BatchNorm1d

        pad = (kernel_size[0] - stride) // 2
        self.enc_pre = nn.Sequential(
            StridedConv1d(2, channels, kernel_size[0], stride=stride, padding=pad, bias=False),
            BN(channels), Act(inplace=True),
        )

        self.encoder = nn.ModuleList()
        for k in kernel_size[1:]:
            self.encoder.append(nn.Sequential(
                nn.Conv1d(channels, channels, k, padding=(k - 1) // 2, bias=False),
                BN(channels), Act(inplace=True),
            ))

        freq = n_fft // 2 // stride
        rf_pre_lin, rf_post_lin = _build_freq_linear(freq, self.rf_freq, pre_post_init)
        self.rf_pre = nn.Sequential(
            rf_pre_lin,
            nn.Conv1d(channels, self.rf_ch, 1, bias=False),
            BN(self.rf_ch),
        )

        pe = rf_cfg.positional_embedding
        blocks = []
        for _ in range(rf_cfg.num_blocks):
            blocks.append(RNNFormerBlock(
                self.rf_ch, self.rf_freq, rf_cfg.num_heads,
                weight_norm=weight_norm,
                positional_embedding=pe, attn_bias=rf_cfg.attn_bias, eps=rf_cfg.eps,
            ))
            pe = None
        self.rf_block = nn.ModuleList(blocks)

        self.rf_post = nn.Sequential(
            rf_post_lin,
            nn.Conv1d(self.rf_ch, channels, 1, bias=False),
            BN(channels),
        )

        self.decoder = nn.ModuleList()
        for k in reversed(kernel_size[1:]):
            self.decoder.append(nn.Sequential(
                nn.Conv1d(channels * 2, channels, 1, bias=False),
                BN(channels), Act(inplace=True),
                nn.Conv1d(channels, channels, k, padding=(k - 1) // 2, bias=False),
                BN(channels), Act(inplace=True),
            ))

        self.dec_post = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 1, bias=False),
            BN(channels), Act(inplace=True),
            ScaledConvTranspose1d(
                channels, 2, kernel_size[0], stride=stride,
                padding=pad, bias=True, normalize=normalize_final_conv,
            ),
        )

    def _model_forward(self, spec: Tensor) -> Tensor:
        """Core model: compressed spec [B,F,T,2] -> mask [B,F,T,2]."""
        B, FREQ, T, _ = spec.shape
        x = spec.permute(0, 2, 3, 1).reshape(B * T, 2, FREQ)

        x = self.enc_pre(x)
        enc_outs = [x]
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)

        x = self.rf_pre(x)
        _, C, _F = x.shape
        x = x.view(B, T, C, _F).permute(1, 0, 3, 2).contiguous()
        for block in self.rf_block:
            x, _ = block(x, None)
        x = x.permute(1, 0, 3, 2).reshape(B * T, C, _F)
        x = self.rf_post(x)

        for dec in self.decoder:
            x = torch.cat([x, enc_outs.pop()], dim=1)
            x = dec(x)

        x = torch.cat([x, enc_outs.pop()], dim=1)
        x = self.dec_post(x)
        return x.reshape(B, T, 2, FREQ).permute(0, 3, 1, 2)

    def forward(self, noisy_spec: Tensor, lengths: Tensor | None = None) -> Tensor:
        """Spectrum-in / spectrum-out. STFT/iSTFT must be done by the caller.

        Args:
            noisy_spec: (B, F, T, 2) real-valued complex spectrum.

        Returns:
            enhanced_spec: (B, F, T, 2).
        """
        if noisy_spec.dim() != 4 or noisy_spec.shape[-1] != 2:
            raise ValueError(
                "FastEnhancer accepts only spectrum input (B, F, T, 2). "
                "Do STFT in the caller (trainer/inference) and pass the result here."
            )
        if noisy_spec.shape[1] != self.n_fft // 2 + 1:
            raise ValueError(f"Expected F={self.n_fft // 2 + 1}, got F={noisy_spec.shape[1]}")
        if noisy_spec.shape[0] == 0 or noisy_spec.shape[2] == 0:
            raise ValueError("FastEnhancer requires non-empty batch and time dimensions")
        if not noisy_spec.is_floating_point():
            raise TypeError("FastEnhancer noisy_spec must use a floating-point dtype")
        return self._forward_spectrum(noisy_spec)

    def _forward_spectrum(self, noisy_spec: Tensor) -> Tensor:
        """Spectrum-in / spectrum-out. (B, F, T, 2) -> (B, F, T, 2). Drops Nyquist for encoder then pads back."""
        c = self.input_compression
        eps = self._eps
        spec = noisy_spec[:, :-1, :, :]
        mag = torch.linalg.norm(spec, dim=-1, keepdim=True).clamp(min=eps)
        spec_c = spec * mag.pow(c - 1.0)
        mask = self._model_forward(spec_c)
        spec_c = torch.view_as_complex(spec_c.contiguous())
        mask_c = torch.view_as_complex(mask.contiguous())
        out_c = spec_c * mask_c
        out_c = out_c * out_c.abs().clamp(min=eps).pow(1.0 / c - 1.0)
        out_c = F.pad(out_c, (0, 0, 0, 1))
        return torch.view_as_real(out_c.contiguous())


FastEnhancerB = FastEnhancer
MODEL_REGISTRY.register("FastEnhancerB")(FastEnhancer)


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 验证本模型参数量与算力。修改下面配置后运行: python tasks/se/models/fastenhancer.py [--sr 16000] [--dur 1.0] [--device cpu]
    # 变体: variant 取 "T"|"B"|"S"|"M"|"L"，或留空用 MODEL_ARGS 覆盖预设。
    # -------------------------------------------------------------------------
    from core.utils.model_profile import profile_model, make_profile_parser
    from core.stft import build_stft_istft

    args = make_profile_parser(default_sr=16000).parse_args()

    # 完整配置：STFT、采样率、模型全部参数（与 YAML 对齐）
    SAMPLE_RATE = args.sr
    STFT_CFG = {
        "n_fft": 512,
        "hop_length": 256,
        "win_length": 512,
        "window": "hann",
    }
    MODEL_ARGS = {
        "variant": "B",
        "n_fft": STFT_CFG["n_fft"],
        # 可选覆盖预设: channels, kernel_size, hop_size, rnnformer_kwargs, ...
    }

    model = FastEnhancer(**MODEL_ARGS)
    stft_module = build_stft_istft(STFT_CFG)

    class _W(nn.Module):
        def __init__(self, m, stft_m):
            super().__init__()
            self._m, self._stft = m, stft_m
        def forward(self, wav):
            spec = self._stft(wav)
            return self._stft.inverse(self._m(spec), wav.shape[-1])

    wrapped = _W(model, stft_module)
    print("\n[Config] STFT:", STFT_CFG, "| sample_rate:", SAMPLE_RATE, "| model args:", MODEL_ARGS)
    profile_model(
        wrapped,
        sample_rate=SAMPLE_RATE,
        duration=args.dur,
        batch_size=args.batch,
        device=args.device,
        repeats=args.repeats,
        label="FastEnhancer",
    )
