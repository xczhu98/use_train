"""BS-RoFormer: Band-Split RoFormer for Speech Enhancement (频谱进/频谱出).

论文: "Music Source Separation with Band-Split RoFormer",
      Ju-Chiang Wang, Wei-Tsung Lu, Minz Won, arXiv: 2309.02612.
参考实现:
      - lucidrains: https://github.com/lucidrains/BS-RoFormer (简洁独立实现)
      - CPJKU: https://github.com/CPJKU/music-source-restoration/blob/main/msr_separation/src/models/bs_roformer

本实现将原始 waveform-in/out 的 BS-RoFormer 改为 spectrum-in/out，
去除内部 STFT/iSTFT，适配本仓库 SE 框架。RoPE 自行实现，不引入外部依赖。

模块分区（速查）:
  - RotaryEmbedding: 旋转位置编码（自行实现，无额外依赖）
  - RMSNorm: RMS 归一化（替代 LayerNorm，lucidrains 风格）
  - FeedForward: GLU-style 前馈网络
  - Attention: 带 RoPE 的多头注意力 + 门控输出
  - Transformer: 注意力 + FFN 堆叠
  - BandSplit: 将频谱按子带分组并独立嵌入
  - MaskEstimator: 从子带特征估计频率掩膜
  - BSRoFormer: 顶层 SE 模型

谱域接口:
  输入 noisy_spec (B, F, T, 2)；输出 enhanced_spec (B, F, T, 2)。
  STFT/iSTFT 由训练框架完成。

脚本: python tasks/se/models/bs_roformer.py [--sr 16000] [--dur 1.0] [--device cpu]
"""

# When run directly, add project root to sys.path
if __name__ == "__main__":
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange

from tasks.se.models.amp_utils import fp32_forward
from tasks.se.models.base_model import BaseSEModel
from core.utils.registry import MODEL_REGISTRY


# =============================================================================
# Rotary Positional Embedding (自行实现, 无外部依赖)
# =============================================================================


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE, Su et al., 2021).

    Self-contained implementation to avoid dependency on ``rotary_embedding_torch``.

    Args:
        dim: Rotation dimension (= dim_head typically).
        base: Base for frequency computation. Default 10000.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # 按 (seq_len, device) 缓存 freqs，避免每层每步重复 outer/cat；不参与 state_dict。
        self._freqs_cache: dict[tuple[int, torch.device], Tensor] = {}

    @fp32_forward
    def forward(self, seq_len: int, device: torch.device) -> Tensor:
        """Return cos/sin caches of shape (seq_len, dim)."""
        # inference_mode 下生成的张量不能留给后续训练步 backward，
        # 因此验证/推理期间跳过缓存（计算本身很便宜）。
        if not torch.is_inference_mode_enabled():
            key = (seq_len, device)
            freqs = self._freqs_cache.get(key)
            if freqs is not None:
                return freqs
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim//2)
        freqs = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        if not torch.is_inference_mode_enabled():
            # 动态混音下 seq_len 变化，缓存设上限防止无界增长。
            if len(self._freqs_cache) >= 32:
                self._freqs_cache.clear()
            self._freqs_cache[key] = freqs.detach()
        return freqs

    @staticmethod
    def rotate_half(x: Tensor) -> Tensor:
        """Rotate half of the features."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary(self, x: Tensor, freqs: Tensor) -> Tensor:
        """Apply RoPE to tensor x: (B, H, T, D)."""
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        return x * cos + self.rotate_half(x) * sin


# =============================================================================
# RMSNorm
# =============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.gamma


# =============================================================================
# FeedForward — GELU 前馈 (论文中 Transformer FFN)
# =============================================================================


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation.

    Args:
        dim: Input/output dimension.
        mult: Inner expansion ratio. Default 4.
        dropout: Dropout rate.
    """

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# =============================================================================
# Attention — 带 RoPE 的多头注意力 + 门控输出
# =============================================================================


class Attention(nn.Module):
    """Multi-head attention with Rotary Position Embedding and gated output.

    论文 Section 3.2: 时间/频率 Transformer 层各自使用独立 RoPE。
    门控机制: output = attn_out * sigmoid(gate), gate 为可学习线性投影。

    Args:
        dim: Model dimension.
        heads: Number of attention heads. Default 8.
        dim_head: Per-head dimension. Default 64.
        dropout: Attention dropout. Default 0.0.
        rotary_emb: Shared RotaryEmbedding instance.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotary_emb: RotaryEmbedding | None = None,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        dim_inner = heads * dim_head

        self.rotary_emb = rotary_emb
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gate = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout),
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = dim_head ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, D) -> (B, T, D)."""
        x_norm = self.norm(x)
        q, k, v = rearrange(
            self.to_qkv(x_norm), "b n (qkv h d) -> qkv b h n d",
            qkv=3, h=self.heads,
        )

        # Apply RoPE
        if self.rotary_emb is not None:
            freqs = self.rotary_emb(q.shape[2], q.device)
            q = self.rotary_emb.apply_rotary(q, freqs)
            k = self.rotary_emb.apply_rotary(k, freqs)

        # Scaled dot-product attention（SDPA 与 matmul→softmax→matmul 数值等价，
        # scale 显式传入保持与原始实现一致）
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            scale=self.scale,
        )

        # Gated output
        gate = self.to_gate(x_norm)
        out = out * rearrange(gate, "b n h -> b h n 1").sigmoid()
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# =============================================================================
# Transformer — Attention + FFN 深度堆叠
# =============================================================================


class Transformer(nn.Module):
    """Transformer encoder with RoPE: interleaved Attention + FFN layers.

    Args:
        dim: Model dimension.
        depth: Number of layers.
        heads: Attention heads. Default 8.
        dim_head: Per-head dimension. Default 64.
        attn_dropout: Attention dropout.
        ff_dropout: FFN dropout.
        ff_mult: FFN expansion. Default 4.
        rotary_emb: Shared RotaryEmbedding instance.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int = 8,
        dim_head: int = 64,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_mult: int = 4,
        rotary_emb: RotaryEmbedding | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, dim_head, attn_dropout, rotary_emb),
                FeedForward(dim, ff_mult, ff_dropout),
            ]))
        self.norm = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# =============================================================================
# BandSplit — 将频谱按子带分组并独立嵌入 (论文 Section 3.1)
# =============================================================================


class BandSplit(nn.Module):
    """Band-Split module: group frequency bins into bands, embed each independently.

    论文 Section 3.1: 将频谱分为 ~60 个子带，每个子带通过独立的 Norm + Linear
    映射到统一的 embedding 维度。

    Args:
        dim: Output embedding dimension per band.
        dim_inputs: Tuple of frequency bin counts for each band.
    """

    def __init__(self, dim: int, dim_inputs: tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = nn.ModuleList([])
        for dim_in in dim_inputs:
            self.to_features.append(nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim),
            ))

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, F_total) -> (B, T, n_bands, D)."""
        splits = x.split(self.dim_inputs, dim=-1)
        outs = []
        for split_input, to_feature in zip(splits, self.to_features):
            outs.append(to_feature(split_input))
        return torch.stack(outs, dim=-2)


# =============================================================================
# MaskEstimator — 从子带特征估计频率掩膜 (论文 Section 3.3)
# =============================================================================


def _MLP(dim_in: int, dim_out: int, dim_hidden: int | None = None, depth: int = 1):
    """Simple MLP with Tanh activations between layers."""
    dim_hidden = dim_hidden or dim_in
    layers = []
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)
    for ind, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(d_in, d_out))
        if ind < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class MaskEstimator(nn.Module):
    """Estimate complex mask per band then reconstruct full frequency mask.

    Per-band MLP + GLU produces mask values for each band's frequency bins.

    Args:
        dim: Input embedding dimension.
        dim_inputs: Tuple of frequency bin counts per band (same as BandSplit).
        depth: MLP depth.
        mlp_expansion_factor: MLP hidden expansion. Default 4.
    """

    def __init__(self, dim: int, dim_inputs: tuple[int, ...],
                 depth: int = 2, mlp_expansion_factor: int = 4):
        super().__init__()
        self.dim_inputs = dim_inputs
        dim_hidden = dim * mlp_expansion_factor
        self.to_freqs = nn.ModuleList([])
        for dim_in in dim_inputs:
            mlp = nn.Sequential(
                _MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1),
            )
            self.to_freqs.append(mlp)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, n_bands, D) -> (B, T, F_total)."""
        bands = x.unbind(dim=-2)
        outs = []
        for band_features, mlp in zip(bands, self.to_freqs):
            outs.append(mlp(band_features))
        return torch.cat(outs, dim=-1)


# =============================================================================
# Default band split configuration (16kHz, n_fft=512 → 257 freq bins, mono)
# =============================================================================

# For 257 freq bins (n_fft=512), divide into bands:
# 12×2 + 12×2 + 4×4 + 4×8 + 3×16 + 4×16 + 1×49 = 257
DEFAULT_FREQS_PER_BANDS_257 = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,   # 12 × 2 = 24  (low freq, fine resolution)
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,   # 12 × 2 = 24
    4, 4, 4, 4,                             # 4 × 4 = 16
    8, 8, 8, 8,                             # 4 × 8 = 32
    16, 16, 16,                             # 3 × 16 = 48
    16, 16, 16, 16,                         # 4 × 16 = 64
    49,                                     # 1 × 49 = 49  (highest freq, coarse)
)

# For 961 freq bins (n_fft=1920, 48kHz), a broader division
# 12×2 + 12×2 + 8×4 + 8×8 + 8×16 + 8×32 + 8×48 + 1×49 = 961
DEFAULT_FREQS_PER_BANDS_961 = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,   # 12 × 2 = 24  (low freq, fine resolution)
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,   # 12 × 2 = 24
    4, 4, 4, 4, 4, 4, 4, 4,               # 8 × 4 = 32
    8, 8, 8, 8, 8, 8, 8, 8,               # 8 × 8 = 64
    16, 16, 16, 16, 16, 16, 16, 16,       # 8 × 16 = 128
    32, 32, 32, 32, 32, 32, 32, 32,       # 8 × 32 = 256
    48, 48, 48, 48, 48, 48, 48, 48,       # 8 × 48 = 384
    49,                                     # 1 × 49 = 49  (highest freq, coarse)
)


def _get_default_bands(n_freqs: int) -> tuple[int, ...]:
    """Select or construct default band split for given number of freq bins."""
    if n_freqs == 257:
        return DEFAULT_FREQS_PER_BANDS_257
    if n_freqs == 961:
        return DEFAULT_FREQS_PER_BANDS_961
    # Fallback: generate bands automatically
    bands = []
    remaining = n_freqs
    band_size = 2
    while remaining > 0:
        if remaining <= band_size:
            bands.append(remaining)
            remaining = 0
        else:
            bands.append(band_size)
            remaining -= band_size
            # Gradually increase band size
            if len(bands) % 12 == 0 and band_size < 64:
                band_size = min(band_size * 2, 64)
    return tuple(bands)


# =============================================================================
# BSRoFormer — 顶层 SE 模型（频谱进/频谱出, 复数掩膜）
# =============================================================================


@MODEL_REGISTRY.register("BSRoFormer")
class BSRoFormer(BaseSEModel):
    """Band-Split RoFormer for speech enhancement (spectrum-in / spectrum-out).

    Input:  noisy_spec (B, F, T, 2) from framework STFT.
    Output: enhanced_spec (B, F, T, 2).

    Architecture (论文 Figure 1):
      1. BandSplit: split freq into bands, embed each -> (B, T, n_bands, dim)
      2. N × (Time-Transformer + Freq-Transformer) with RoPE
      3. MaskEstimator: estimate complex mask per band -> reconstruct
      4. Complex-multiply mask with input spectrum

    Args:
        n_fft: FFT size. Default 512.
        dim: Embedding dimension per band. Default 128.
        depth: Number of (time+freq) transformer pairs. Default 6.
        time_transformer_depth: Layers per time transformer. Default 2.
        freq_transformer_depth: Layers per freq transformer. Default 2.
        heads: Number of attention heads. Default 8.
        dim_head: Per-head dimension. Default 64.
        attn_dropout: Attention dropout. Default 0.0.
        ff_dropout: FFN dropout. Default 0.0.
        ff_mult: FFN expansion ratio. Default 4.
        freqs_per_bands: Band frequency split. Auto-detected if None.
        mask_estimator_depth: MLP depth in mask estimator. Default 2.
    """

    def __init__(
        self,
        n_fft: int = 512,
        dim: int = 128,
        depth: int = 6,
        time_transformer_depth: int = 2,
        freq_transformer_depth: int = 2,
        heads: int = 8,
        dim_head: int = 64,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_mult: int = 4,
        freqs_per_bands: tuple[int, ...] | None = None,
        mask_estimator_depth: int = 2,
        **kwargs,
    ):
        super().__init__()
        for key in ("hop_length", "win_length", "window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"BSRoFormer got unexpected keyword arguments: {sorted(kwargs)}")
        self.n_fft = n_fft
        n_freqs = n_fft // 2 + 1

        # Determine band split
        if freqs_per_bands is None:
            freqs_per_bands = _get_default_bands(n_freqs)
        if sum(freqs_per_bands) != n_freqs:
            raise ValueError(
                f"Band split sum {sum(freqs_per_bands)} != freq bins {n_freqs}"
            )

        # Band-split input dims: each band has (freq_bins * 2) input features (real + imag)
        freqs_per_bands_with_complex = tuple(2 * f for f in freqs_per_bands)

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        # Shared RoPE for time and freq transformers
        time_rotary = RotaryEmbedding(dim=dim_head)
        freq_rotary = RotaryEmbedding(dim=dim_head)

        transformer_kwargs = dict(
            dim=dim, heads=heads, dim_head=dim_head,
            attn_dropout=attn_dropout, ff_dropout=ff_dropout, ff_mult=ff_mult,
        )

        # Hierarchical time + freq transformer layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(depth=time_transformer_depth, rotary_emb=time_rotary, **transformer_kwargs),
                Transformer(depth=freq_transformer_depth, rotary_emb=freq_rotary, **transformer_kwargs),
            ]))

        self.final_norm = RMSNorm(dim)

        # Mask estimator (complex mask: real + imag per band)
        self.mask_estimator = MaskEstimator(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex,
            depth=mask_estimator_depth,
        )

    @fp32_forward
    def forward(self, noisy_spec: torch.Tensor, lengths=None, **kw) -> torch.Tensor:
        """noisy_spec: (B, F, T, 2). Returns enhanced_spec (B, F, T, 2)."""
        if noisy_spec.dim() != 4 or noisy_spec.shape[-1] != 2:
            raise ValueError(f"Expected [B, F, T, 2], got {tuple(noisy_spec.shape)}")
        if noisy_spec.shape[1] != self.n_fft // 2 + 1:
            raise ValueError(f"Expected F={self.n_fft // 2 + 1}, got F={noisy_spec.shape[1]}")
        if noisy_spec.shape[0] == 0 or noisy_spec.shape[2] == 0:
            raise ValueError("BSRoFormer requires non-empty batch and time dimensions")
        if not noisy_spec.is_floating_point():
            raise TypeError("BSRoFormer noisy_spec must use a floating-point dtype")
        noisy_spec = noisy_spec.float()
        B, F_bins, T, _ = noisy_spec.shape

        # (B, F, T, 2) -> (B, T, F*2) : concat real+imag along freq
        stft_repr = noisy_spec.permute(0, 2, 1, 3)  # (B, T, F, 2)
        x = rearrange(stft_repr, "b t f c -> b t (f c)")

        # BandSplit: (B, T, F*2) -> (B, T, n_bands, dim)
        x = self.band_split(x)

        # Hierarchical time/freq transformers
        for time_transformer, freq_transformer in self.layers:
            # Time transformer: operate on time dim for each band
            x = rearrange(x, "b t f d -> (b f) t d")
            x = time_transformer(x)
            x = rearrange(x, "(b f) t d -> b t f d", b=B)

            # Freq transformer: operate on band dim for each time step
            x = rearrange(x, "b t f d -> (b t) f d")
            x = freq_transformer(x)
            x = rearrange(x, "(b t) f d -> b t f d", b=B)

        x = self.final_norm(x)

        # Estimate mask: (B, T, n_bands, dim) -> (B, T, F*2)
        mask = self.mask_estimator(x)
        mask = rearrange(mask, "b t (f c) -> b t f c", c=2)  # (B, T, F, 2)

        # Complex multiplication: stft_repr * mask
        # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        stft_real = stft_repr[..., 0]
        stft_imag = stft_repr[..., 1]
        mask_real = mask[..., 0]
        mask_imag = mask[..., 1]

        enh_real = stft_real * mask_real - stft_imag * mask_imag
        enh_imag = stft_real * mask_imag + stft_imag * mask_real

        # (B, T, F, 2) -> (B, F, T, 2)
        enhanced = torch.stack([enh_real, enh_imag], dim=-1).permute(0, 2, 1, 3)
        return enhanced


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 验证本模型参数量与算力。运行: python tasks/se/models/bs_roformer.py [--sr 16000] [--dur 1.0] [--device cpu]
    # -------------------------------------------------------------------------
    from core.utils.model_profile import profile_model, make_profile_parser
    from core.stft import build_stft_istft

    args = make_profile_parser(default_sr=16000).parse_args()

    STFT_CFG = {
        "n_fft": 512,
        "hop_length": 256,
        "win_length": 512,
        "window": "hann",
    }
    SAMPLE_RATE = args.sr
    MODEL_ARGS = {
        "n_fft": STFT_CFG["n_fft"],
        "dim": 128,
        "depth": 6,
        "time_transformer_depth": 2,
        "freq_transformer_depth": 2,
        "heads": 8,
        "dim_head": 64,
        "mask_estimator_depth": 2,
    }

    model = BSRoFormer(**MODEL_ARGS)
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
        label="BSRoFormer",
    )
