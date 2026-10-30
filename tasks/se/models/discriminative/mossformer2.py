"""MossFormer2: Gated Single-head Transformer with FSMN for Speech Enhancement (频谱进/频谱出).

论文: "MossFormer2: Combining Transformer and RNN-Free Recurrent Network for
      Enhanced Time-Domain Monaural Speech Separation", S. Zhao, B. Ma,
      ICASSP 2024.
上游代码:
      - ClearerVoice-Studio: https://github.com/modelscope/ClearerVoice-Studio/tree/main/train/speech_enhancement/models/mossformer2
      - FLASH attention 源自: https://github.com/lucidrains/FLASH-pytorch

本文件为干净的纯频谱进出版本，不依赖 clearervoice_impl 或 ConvSTFT，
直接处理 (B, F, T, 2) 频谱。

模块分区（速查）:
  - ScaleNorm: 维度归一化（FLASH 论文）
  - ScaledSinuEmbedding: 可学习缩放正弦位置编码
  - OffsetScale: 逐头仿射变换（Q/K 分头后）
  - UniDeepFsmn: 有限状态记忆网络（论文中 RNN-Free Recurrent）
  - GatedFSMN: Gated FSMN block（v * fsmn(u) + residual）
  - FLASHAttention: 分组二次注意力 + 线性注意力 + 门控（论文核心）
  - MossFormerBlock: FLASH + GatedFSMN 交替堆叠（MossFormer2 的创新）
  - MossFormer2MaskNet: 编码器 → MossFormer blocks → 解码器 → 掩膜
  - MossFormer2: 顶层 SE 模型

谱域接口:
  输入 noisy_spec (B, F, T, 2)；输出 enhanced_spec (B, F, T, 2)。
  STFT/iSTFT 由训练框架完成。

脚本: python tasks/se/models/mossformer2.py [--sr 16000] [--dur 1.0] [--device cpu]
"""

# When run directly, add project root to sys.path
if __name__ == "__main__":
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops import rearrange

from tasks.se.models.amp_utils import fp32_forward
from tasks.se.models.base_model import BaseSEModel
from core.utils.registry import MODEL_REGISTRY


# =============================================================================
# Utilities
# =============================================================================

def _padding_to_multiple_of(n: int, mult: int) -> int:
    remainder = n % mult
    return 0 if remainder == 0 else mult - remainder


# =============================================================================
# ScaleNorm (FLASH 论文)
# =============================================================================


class ScaleNorm(nn.Module):
    """L2 normalization with learnable scale (替代 LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# =============================================================================
# ScaledSinuEmbedding — 可学习缩放正弦位置编码
# =============================================================================


class ScaledSinuEmbedding(nn.Module):
    """Sinusoidal positional encoding with learnable scale factor."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> positional embedding (T, D)."""
        T = x.shape[1]
        t = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        return emb * self.scale


# =============================================================================
# OffsetScale — 逐头仿射变换 (Q/K 分4头用)
# =============================================================================


class OffsetScale(nn.Module):
    """Per-head affine transform (gamma, beta) for Q/K projections."""

    def __init__(self, dim: int, heads: int = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple:
        """x: (..., D) -> tuple of (heads,) tensors each (..., D)."""
        out = einsum("... d, h d -> ... h d", x, self.gamma) + self.beta
        return out.unbind(dim=-2)


# =============================================================================
# UniDeepFsmn — 有限状态记忆网络 (Feedforward Sequential Memory Network)
# =============================================================================


class UniDeepFsmn(nn.Module):
    """Uni-directional Deep FSMN: linear + conv FIR filter + residual.

    Provides efficient local temporal modeling (论文中的 RNN-free recurrent module).

    Args:
        input_dim: Input/output dimension.
        hidden_dim: Hidden projection dimension.
        order: FIR filter order (conv kernel size). Default 10.
    """

    def __init__(self, input_dim: int, hidden_dim: int, order: int = 10):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, input_dim, bias=False)
        self.conv = nn.Conv2d(
            hidden_dim, hidden_dim, (order, 1),
            padding=(order // 2, 0), groups=hidden_dim, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)."""
        residual = x
        T = x.shape[1]
        p = self.linear(x)  # (B, T, H)
        p = p.unsqueeze(-1).permute(0, 2, 1, 3)  # (B, H, T, 1)
        p = self.conv(p)[:, :, :T, :]  # (B, H, T, 1)  trim to T
        p = p.squeeze(-1).transpose(1, 2)  # (B, T, H)
        return self.project(p) + residual


# =============================================================================
# GatedFSMN — 门控 FSMN block
# =============================================================================


class GatedFSMN(nn.Module):
    """Gated FSMN: split input into (v, u), apply FSMN to u, gate with v.

    论文 MossFormer2 的关键组件: 在每个 FLASH 注意力层后插入 GatedFSMN，
    提供高效的局部时间建模。

    Args:
        dim: Input/output dimension.
        hidden_dim: FSMN hidden dimension.
        order: FSMN filter order.
    """

    def __init__(self, dim: int, hidden_dim: int, order: int = 10):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, dim * 2)
        self.fsmn = UniDeepFsmn(dim, hidden_dim, order)
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)."""
        residual = x
        x = self.norm(x)
        v, u = self.proj_in(x).chunk(2, dim=-1)
        u = self.fsmn(u)
        x = v * torch.sigmoid(u)
        return self.proj_out(x) + residual


# =============================================================================
# ConvModule — Conformer-style 1D convolution + GLU
# =============================================================================


class ConvModule(nn.Module):
    """Lightweight 1D convolution module used inside FLASH to_hidden / to_qk.

    DepthwiseConv1d + PointwiseConv1d + BatchNorm + Swish, with residual.

    Args:
        dim: Input dimension.
        expansion_factor: Hidden expansion. Default 2.
        kernel_size: Depthwise conv kernel. Default 31.
        dropout: Dropout rate.
    """

    def __init__(self, dim: int, expansion_factor: int = 2, kernel_size: int = 31,
                 dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2),
            nn.GLU(dim=-1),
            nn.Conv1d(inner_dim, inner_dim, kernel_size, padding=kernel_size // 2, groups=inner_dim),
            nn.BatchNorm1d(inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)."""
        B, T, D = x.shape
        # Linear + GLU
        x = self.net[0](x)
        x = self.net[1](x)
        # Conv (need B, C, T)
        x = x.transpose(1, 2)
        x = self.net[2](x)
        x = self.net[3](x)
        x = self.net[4](x)
        x = x.transpose(1, 2)
        # Linear + Dropout
        x = self.net[5](x)
        x = self.net[6](x)
        return x


# =============================================================================
# FFConvM — LayerNorm → Linear → SiLU → ConvModule → Dropout (FLASH 前馈)
# =============================================================================


class FFConvM(nn.Module):
    """Feed-forward with ConvModule used in FLASH attention's to_hidden/to_qk projections.

    Args:
        dim: Input/output dimension.
        dim_out: Output dimension. Default same as dim.
        mult: Inner expansion ratio. Default 2.
        dropout: Dropout rate.
    """

    def __init__(self, dim: int, dim_out: int | None = None, mult: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        dim_out = dim_out or dim
        dim_hidden = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_hidden),
            nn.SiLU(),
            ConvModule(dim_hidden, dropout=dropout),
            nn.Linear(dim_hidden, dim_out),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# FLASHAttention — 分组二次注意力 + 线性注意力 + 门控 (论文核心注意力)
# =============================================================================


class FLASHAttention(nn.Module):
    """FLASH: Fast Linear Attention with a Single Head (Hua et al., 2022).

    MossFormer2 使用此机制替代标准多头注意力：
    - 将序列分为 group_size 大小的组
    - 组内使用二次注意力 (quadratic): ReLU(QK^T/g)^2
    - 组间使用线性注意力 (linear): 通过累积 K^T·V 实现全局上下文
    - 门控输出: (att_u * v) * sigmoid(att_v * u)

    Args:
        dim: Model dimension.
        group_size: Attention group size (default 256).
        query_key_dim: Q/K dimension (default 128).
        expansion_factor: Hidden expansion. Default 2.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        group_size: int = 256,
        query_key_dim: int = 128,
        expansion_factor: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.query_key_dim = query_key_dim
        self.norm = ScaleNorm(dim)

        # Projections for hidden and query/key
        self.to_hidden = FFConvM(dim, dim_out=hidden_dim * 2, mult=2, dropout=dropout)
        self.to_qk = FFConvM(dim, dim_out=query_key_dim, mult=2, dropout=dropout)

        # Per-head affine: splits into (quad_q, lin_q, quad_k, lin_k)
        self.offset_scale = OffsetScale(query_key_dim, heads=4)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Dropout(dropout),
        )

        # Rotary embedding for positional encoding within groups
        self.rotary_emb = None  # optional, can add if needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)."""
        B, T, D = x.shape
        residual = x

        # Token shift: shift half channels by 1 position for local mixing
        x_shifted = x.clone()
        x_shifted[:, 1:, D // 2:] = x[:, :-1, D // 2:]
        x = x_shifted

        normed = self.norm(x)

        # Hidden projections -> split into v, u
        hidden = self.to_hidden(normed)
        v, u = hidden.chunk(2, dim=-1)

        # Q/K projections -> 4-head split
        qk = self.to_qk(normed)
        quad_q, lin_q, quad_k, lin_k = self.offset_scale(qk)

        # Pad sequence to multiple of group_size
        pad_len = _padding_to_multiple_of(T, self.group_size)
        if pad_len > 0:
            quad_q = F.pad(quad_q, (0, 0, 0, pad_len))
            quad_k = F.pad(quad_k, (0, 0, 0, pad_len))
            lin_q = F.pad(lin_q, (0, 0, 0, pad_len))
            lin_k = F.pad(lin_k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            u = F.pad(u, (0, 0, 0, pad_len))

        # Reshape into groups
        g = self.group_size
        quad_q = rearrange(quad_q, "b (n g) d -> b n g d", g=g)
        quad_k = rearrange(quad_k, "b (n g) d -> b n g d", g=g)
        lin_q = rearrange(lin_q, "b (n g) d -> b n g d", g=g)
        lin_k = rearrange(lin_k, "b (n g) d -> b n g d", g=g)
        v = rearrange(v, "b (n g) d -> b n g d", g=g)
        u = rearrange(u, "b (n g) d -> b n g d", g=g)

        # --- Quadratic attention (intra-group) ---
        sim = einsum("b n i d, b n j d -> b n i j", quad_q, quad_k) / g
        attn = F.relu(sim) ** 2  # ReLU-squared attention
        quad_out = einsum("b n i j, b n j d -> b n i d", attn, v)

        # --- Linear attention (inter-group global context) ---
        lin_kv = einsum("b n g d, b n g e -> b n d e", lin_k, v) / g
        # Cumulative sum across groups for causal linear attention
        lin_kv_cumsum = lin_kv.cumsum(dim=1)
        lin_out = einsum("b n g d, b n d e -> b n g e", lin_q, lin_kv_cumsum)

        # Combine quadratic + linear attention, apply gating
        att_u = quad_out + lin_out
        att_v = quad_out  # gate uses quadratic only

        # Flatten back
        att_u = rearrange(att_u, "b n g d -> b (n g) d")[:, :T]
        att_v = rearrange(att_v, "b n g d -> b (n g) d")[:, :T]
        u_orig = rearrange(u, "b n g d -> b (n g) d")[:, :T]
        v_orig = rearrange(v, "b n g d -> b (n g) d")[:, :T]

        # Gated output: (att_u * v) * sigmoid(att_v * u)
        out = (att_u * v_orig) * torch.sigmoid(att_v * u_orig)
        out = self.to_out(out)

        return out + residual


# =============================================================================
# MossFormerBlock — FLASH attention + GatedFSMN 交替堆叠
# =============================================================================


class MossFormerBlock(nn.Module):
    """MossFormer2 processing block: alternating FLASH attention and Gated FSMN layers.

    MossFormer2 的核心创新在于每个 FLASH 注意力层后插入一个 Gated FSMN block，
    提供局部时间建模补充全局注意力的不足。

    Args:
        dim: Model dimension.
        depth: Number of (FLASH + FSMN) layer pairs.
        group_size: FLASH attention group size.
        query_key_dim: FLASH Q/K dimension.
        expansion_factor: Hidden expansion.
        fsmn_hidden_dim: FSMN hidden projection dimension.
        fsmn_order: FSMN filter order.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int = 512,
        depth: int = 24,
        group_size: int = 256,
        query_key_dim: int = 128,
        expansion_factor: float = 2.0,
        fsmn_hidden_dim: int = 256,
        fsmn_order: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                FLASHAttention(
                    dim=dim,
                    group_size=group_size,
                    query_key_dim=query_key_dim,
                    expansion_factor=expansion_factor,
                    dropout=dropout,
                ),
                GatedFSMN(
                    dim=dim,
                    hidden_dim=fsmn_hidden_dim,
                    order=fsmn_order,
                ),
            ]))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)."""
        for flash_attn, gated_fsmn in self.layers:
            x = flash_attn(x)
            x = gated_fsmn(x)
        return self.norm(x)


# =============================================================================
# MossFormer2MaskNet — 编码器 → MossFormer blocks → 解码器 → 掩膜
# =============================================================================


class MossFormer2MaskNet(nn.Module):
    """MossFormer2 mask estimation network.

    Architecture (from ClearerVoice-Studio):
      1. LayerNorm + Conv1d encoder: input_dim -> hidden_dim
      2. ScaledSinuEmbedding positional encoding
      3. Computation blocks (MossFormerBlock with skip connections)
      4. PReLU + Conv1d -> gated output (Tanh × Sigmoid)
      5. Conv1d decoder: hidden_dim -> output_dim (freq bins)

    Args:
        input_dim: Input feature dimension (F*2 for real+imag).
        hidden_dim: Internal processing dimension. Default 512.
        output_dim: Output dimension (n_fft//2+1 for mag mask). Not used for CRM.
        num_blocks: Number of MossFormer blocks. Default 24.
        group_size: FLASH group size. Default 256.
        query_key_dim: FLASH Q/K dim. Default 128.
        fsmn_hidden_dim: FSMN hidden. Default 256.
        fsmn_order: FSMN filter order. Default 10.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int | None = None,
        num_blocks: int = 24,
        group_size: int = 256,
        query_key_dim: int = 128,
        fsmn_hidden_dim: int = 256,
        fsmn_order: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()
        output_dim = output_dim or input_dim

        # Encoder
        self.norm = nn.LayerNorm(input_dim)  # operates on last dim
        self.encoder = nn.Conv1d(input_dim, hidden_dim, 1, bias=False)
        self.pos_enc = ScaledSinuEmbedding(hidden_dim)

        # MossFormer computation block with skip connection
        self.mossformer = MossFormerBlock(
            dim=hidden_dim,
            depth=num_blocks,
            group_size=group_size,
            query_key_dim=query_key_dim,
            expansion_factor=2.0,
            fsmn_hidden_dim=fsmn_hidden_dim,
            fsmn_order=fsmn_order,
            dropout=dropout,
        )
        self.skip_norm = nn.LayerNorm(hidden_dim)

        # Decoder
        self.act = nn.PReLU()
        self.decoder_proj = nn.Conv1d(hidden_dim, hidden_dim * 2, 1)  # for gating
        self.decoder_out = nn.Conv1d(hidden_dim, output_dim, 1, bias=False)
        self.output_act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D_in, T) -> mask: (B, T, D_out)."""
        # Encoder
        x = x.transpose(1, 2)  # (B, T, D_in)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, D_in, T)
        e = self.encoder(x)  # (B, H, T)
        e = e.transpose(1, 2)  # (B, T, H)

        # Add positional encoding
        e = e + self.pos_enc(e)

        # MossFormer blocks with skip connection
        skip = e
        e = self.mossformer(e)
        e = self.skip_norm(e + skip)

        # Decoder with gating
        e = self.act(e)
        e = e.transpose(1, 2)  # (B, H, T)
        gate = self.decoder_proj(e)  # (B, H*2, T)
        g1, g2 = gate.chunk(2, dim=1)
        e = torch.tanh(g1) * torch.sigmoid(g2)  # (B, H, T)
        mask = self.decoder_out(e)  # (B, D_out, T)
        mask = self.output_act(mask)  # (B, D_out, T)
        return mask.transpose(1, 2)  # (B, T, D_out)


# =============================================================================
# MossFormer2 — 顶层 SE 模型（频谱进/频谱出, PSM 掩膜）
# =============================================================================


@MODEL_REGISTRY.register("MossFormer2")
class MossFormer2(BaseSEModel):
    """MossFormer2-based speech enhancement (spectrum-in / spectrum-out).

    Input:  noisy_spec (B, F, T, 2) from framework STFT.
    Output: enhanced_spec (B, F, T, 2).

    Architecture:
      - Extract magnitude and phase from complex spectrum
      - MossFormer2MaskNet estimates magnitude mask from power-compressed features
      - Apply mask to magnitude, reconstruct complex spectrum with original phase

    默认使用 power-compression masking (与 ClearerVoice-Studio MossFormer2 一致):
      - 输入: power_compress(mag) + 相位信息
      - 输出: magnitude mask (ReLU 激活, 非负)

    Args:
        n_fft: FFT size. Default 512.
        hidden_dim: MossFormer hidden dimension. Default 512.
        num_blocks: MossFormer depth. Default 24.
        group_size: FLASH attention group size. Default 256.
        query_key_dim: FLASH Q/K dimension. Default 128.
        fsmn_hidden_dim: FSMN hidden dimension. Default 256.
        fsmn_order: FSMN filter order. Default 10.
        compress_factor: Power compression factor. Default 0.3.
        dropout: Dropout rate. Default 0.0.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hidden_dim: int = 512,
        num_blocks: int = 24,
        group_size: int = 256,
        query_key_dim: int = 128,
        fsmn_hidden_dim: int = 256,
        fsmn_order: int = 10,
        compress_factor: float = 0.3,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        for key in ("hop_length", "win_length", "window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"MossFormer2 got unexpected keyword arguments: {sorted(kwargs)}")
        self.n_fft = n_fft
        self.compress_factor = compress_factor
        freq_dim = n_fft // 2 + 1

        # Input: power-compressed mag + real + imag = 3 * freq_dim features
        input_dim = freq_dim * 3

        self.mask_net = MossFormer2MaskNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=freq_dim,  # magnitude mask
            num_blocks=num_blocks,
            group_size=group_size,
            query_key_dim=query_key_dim,
            fsmn_hidden_dim=fsmn_hidden_dim,
            fsmn_order=fsmn_order,
            dropout=dropout,
        )

    @staticmethod
    def _power_compress(mag: torch.Tensor, factor: float = 0.3) -> torch.Tensor:
        """Power compression: mag^factor."""
        return mag.pow(factor)

    @fp32_forward
    def forward(self, noisy_spec: torch.Tensor, lengths=None, **kw) -> torch.Tensor:
        """noisy_spec: (B, F, T, 2). Returns enhanced_spec (B, F, T, 2)."""
        if noisy_spec.dim() != 4 or noisy_spec.shape[-1] != 2:
            raise ValueError(f"Expected [B, F, T, 2], got {tuple(noisy_spec.shape)}")
        if noisy_spec.shape[1] != self.n_fft // 2 + 1:
            raise ValueError(f"Expected F={self.n_fft // 2 + 1}, got F={noisy_spec.shape[1]}")
        if noisy_spec.shape[0] == 0 or noisy_spec.shape[2] == 0:
            raise ValueError("MossFormer2 requires non-empty batch and time dimensions")
        if not noisy_spec.is_floating_point():
            raise TypeError("MossFormer2 noisy_spec must use a floating-point dtype")
        noisy_spec = noisy_spec.float()
        B, F, T, _ = noisy_spec.shape

        # Extract real, imag, magnitude
        real = noisy_spec[..., 0]  # (B, F, T)
        imag = noisy_spec[..., 1]
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-12)

        # Power compressed magnitude
        cmag = self._power_compress(mag, self.compress_factor)  # (B, F, T)

        # Input features: compressed_mag, real, imag concatenated along freq
        # (B, F, T) x 3 -> (B, F*3, T) -> transpose to (B, T, F*3) for MaskNet
        feat = torch.cat([cmag, real, imag], dim=1)  # (B, 3F, T)

        # Estimate mask: (B, 3F, T) -> (B, T, F)
        mask = self.mask_net(feat)  # (B, T, F)

        # Apply mask to magnitude and reconstruct with phase
        mask = mask.permute(0, 2, 1)  # (B, F, T)
        phase = torch.atan2(imag, real + 1e-8)
        enh_mag = mag * mask
        enh_real = enh_mag * torch.cos(phase)
        enh_imag = enh_mag * torch.sin(phase)

        return torch.stack([enh_real, enh_imag], dim=-1)


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 验证本模型参数量与算力。运行: python tasks/se/models/mossformer2.py [--sr 16000] [--dur 1.0] [--device cpu]
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
        "hidden_dim": 256,      # reduced for 16kHz (full model uses 512)
        "num_blocks": 6,        # reduced for testing (full uses 24)
        "group_size": 256,
        "query_key_dim": 128,
        "fsmn_hidden_dim": 128,
    }

    model = MossFormer2(**MODEL_ARGS)
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
        label="MossFormer2",
    )
