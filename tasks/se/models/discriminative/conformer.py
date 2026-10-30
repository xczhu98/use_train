"""Conformer: Conformer-based Speech Enhancement (频谱进/频谱出).

论文: "Conformer: Convolution-augmented Transformer for Speech Recognition",
      Anmol Gulati, James Qin, Chung-Cheng Chiu, et al., Interspeech 2020.
      arXiv: 2005.08100.

增强适配参考: ESPnet conformer_separator.py
      https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/conformer_separator.py

模块分区（速查）:
  - RelPositionalEncoding: 相对位置编码（Transformer-XL 式）
  - MultiHeadedSelfAttention: 含相对位置偏置的多头自注意力（论文 Figure 1, MHSA Module）
  - ConvolutionModule: Pointwise Conv → GLU → Depthwise Conv → BN → Swish → Pointwise Conv（论文 Figure 1, Conv Module）
  - FeedForwardModule: Macaron-style 半步前馈（论文 Figure 1, Feed Forward Module）
  - ConformerBlock: FFN(1/2) → MHSA → Conv → FFN(1/2) → LayerNorm（论文 Figure 1 完整 block）
  - Conformer: 顶层 SE 模型，Linear 嵌入 + N × ConformerBlock + Linear 掩膜估计

谱域接口:
  输入 noisy_spec (B, F, T, 2)；输出 enhanced_spec (B, F, T, 2)。
  STFT/iSTFT 由训练框架完成。

脚本: python tasks/se/models/conformer.py [--sr 16000] [--dur 1.0] [--device cpu]
"""

# When run directly, add project root to sys.path
if __name__ == "__main__":
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks.se.models.amp_utils import fp32_forward
from tasks.se.models.base_model import BaseSEModel
from core.utils.registry import MODEL_REGISTRY


# =============================================================================
# Relative Positional Encoding (Transformer-XL / Conformer 论文 Section 2.2)
# =============================================================================


class RelPositionalEncoding(nn.Module):
    """Sinusoidal relative positional encoding.

    Generates positional encodings for both positive and negative relative positions,
    used by RelMultiHeadedAttention.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length.
        dropout: Dropout rate applied to embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", torch.empty(0), persistent=False)

    def _create_pe(self, length: int, device: torch.device, dtype: torch.dtype):
        """Create sinusoidal positional encoding table."""
        pe = torch.zeros(length, self.d_model, device=device, dtype=dtype)
        position = torch.arange(0, length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, length, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply positional encoding.

        Args:
            x: (B, T, D) input tensor.

        Returns:
            x_scaled: (B, T, D) with sqrt(d_model) scaling.
            pos_emb: (1, T, D) positional encoding.
        """
        T = x.size(1)
        if self.pe.numel() == 0 or self.pe.size(1) < T:
            self.pe = self._create_pe(T, x.device, x.dtype)
        elif self.pe.device != x.device or self.pe.dtype != x.dtype:
            self.pe = self.pe.to(device=x.device, dtype=x.dtype)
        pos_emb = self.pe[:, :T]
        x = x * math.sqrt(self.d_model)
        x = self.dropout(x)
        return x, pos_emb


# =============================================================================
# Multi-Headed Self-Attention with Relative Position Bias (论文 Section 2.2)
# =============================================================================


class MultiHeadedSelfAttention(nn.Module):
    """Multi-headed self-attention with relative positional encoding.

    Implements the attention from "Transformer-XL: Attentive Language Models
    Beyond a Fixed-Length Context" (Dai et al., 2019), used in Conformer.

    Args:
        n_heads: Number of attention heads.
        d_model: Model dimension.
        dropout: Attention dropout rate.
    """

    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        if n_heads < 1 or d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.causal = causal
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        # Learnable bias for content and position (Transformer-XL style)
        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u.unsqueeze(0))
        nn.init.xavier_uniform_(self.pos_bias_v.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        """Relative position shift to align positional scores."""
        B, H, T, _ = x.shape
        # Pad and reshape to shift
        x = F.pad(x, (1, 0))  # (B, H, T, T+1)
        x = x.view(B, H, -1, T)[:, :, 1:].view_as(x[:, :, :, :T])
        return x

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            pos_emb: (1, T, D)

        Returns:
            out: (B, T, D)
        """
        B, T, _ = x.shape
        q = self.linear_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(1, T, self.n_heads, self.d_k).transpose(1, 2)

        # Content-based attention + position-based attention (Transformer-XL Eq. 3)
        q_with_u = q + self.pos_bias_u.unsqueeze(1)  # (B, H, T, d_k)
        q_with_v = q + self.pos_bias_v.unsqueeze(1)

        AC = torch.matmul(q_with_u, k.transpose(-2, -1))      # content attention
        BD = torch.matmul(q_with_v, p.transpose(-2, -1))       # positional attention
        BD = self._rel_shift(BD)

        scale = self.d_k ** -0.5
        attn = (AC + BD) * scale
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(T, T, device=attn.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)  # (B, H, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.linear_out(out)


# =============================================================================
# Convolution Module (论文 Figure 1, Conv Module; Section 2.1)
# =============================================================================


class ConvolutionModule(nn.Module):
    """Conformer convolution module.

    Pointwise Conv → GLU → Depthwise Conv → BatchNorm → Swish → Pointwise Conv → Dropout.

    Args:
        channels: Number of channels (= d_model).
        kernel_size: Depthwise convolution kernel size. Must be odd.
        dropout: Dropout rate.
        causal: If True, use causal (left-only) padding. Default False.
    """

    def __init__(self, channels: int, kernel_size: int = 31, dropout: float = 0.1,
                 causal: bool = False):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        self.causal = causal
        self.pointwise1 = nn.Conv1d(channels, 2 * channels, 1)
        self.glu = nn.GLU(dim=1)
        if causal:
            self.pad = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
            self.depthwise = nn.Conv1d(channels, channels, kernel_size, groups=channels)
        else:
            self.depthwise = nn.Conv1d(
                channels, channels, kernel_size, padding=kernel_size // 2, groups=channels
            )
            self.pad = nn.Identity()
        self.bn = nn.BatchNorm1d(channels)
        self.swish = nn.SiLU()
        self.pointwise2 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, T, C)."""
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.pointwise1(x)
        x = self.glu(x)
        x = self.pad(x)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.swish(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


# =============================================================================
# Feed-Forward Module (论文 Section 2.3, Macaron-style half-step)
# =============================================================================


class FeedForwardModule(nn.Module):
    """Feed-forward module for Conformer block.

    LayerNorm → Linear → Swish → Dropout → Linear → Dropout.

    Args:
        d_model: Model dimension.
        d_ff: Inner feed-forward dimension.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.swish = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)."""
        x = self.norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return self.dropout2(x)


# =============================================================================
# Conformer Block (论文 Figure 1: FFN(1/2) → MHSA → Conv → FFN(1/2) → LN)
# =============================================================================


class ConformerBlock(nn.Module):
    """Single Conformer block (Gulati et al., 2020, Figure 1).

    Macaron-style: x + 1/2 FFN → MHSA + residual → ConvModule + residual → 1/2 FFN + residual → LayerNorm.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: FFN hidden dimension.
        depthwise_conv_kernel_size: Kernel size for depthwise convolution.
        dropout: Dropout rate.
        causal: Use causal convolution.
        use_macaron: Use Macaron-style dual half-step FFN (recommended).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.1,
        causal: bool = False,
        use_macaron: bool = True,
    ):
        super().__init__()
        self.use_macaron = use_macaron
        if use_macaron:
            self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
            self.ff1_scale = 0.5
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadedSelfAttention(n_heads, d_model, dropout, causal=causal)
        self.conv = ConvolutionModule(d_model, depthwise_conv_kernel_size, dropout, causal)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        self.ff2_scale = 0.5 if use_macaron else 1.0
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            pos_emb: (1, T, D) relative positional encoding

        Returns:
            (B, T, D)
        """
        # Macaron-style first half-step FFN
        if self.use_macaron:
            x = x + self.ff1_scale * self.ff1(x)

        # Multi-head self-attention with relative positional encoding
        residual = x
        x_norm = self.attn_norm(x)
        x = residual + self.attn(x_norm, pos_emb)

        # Convolution module
        x = x + self.conv(x)

        # Second half-step FFN
        x = x + self.ff2_scale * self.ff2(x)

        # Final LayerNorm
        x = self.final_norm(x)
        return x


# =============================================================================
# ConformerSE — 顶层 SE 模型（频谱进/频谱出, 复数掩膜）
# =============================================================================


@MODEL_REGISTRY.register("Conformer")
class Conformer(BaseSEModel):
    """Conformer-based speech enhancement model (spectrum-in / spectrum-out).

    Input:  noisy_spec (B, F, T, 2) from framework STFT.
    Output: enhanced_spec (B, F, T, 2).

    Architecture:
      - Input: (B, F, T, 2) -> (B, T, F*2) via reshape
      - Linear embedding to d_model
      - Positional encoding (relative sinusoidal)
      - N × ConformerBlock
      - Linear output to F*2 -> complex mask (real + imag)
      - Apply mask: noisy_spec * mask

    Args:
        n_fft: FFT size (injected from YAML stft block). Default 512.
        num_blocks: Number of Conformer blocks. Default 6.
        d_model: Model dimension. Default 256.
        n_heads: Number of attention heads. Default 4.
        d_ff: FFN hidden dimension. Default 1024.
        depthwise_conv_kernel_size: Depthwise conv kernel. Default 31.
        dropout: Dropout rate. Default 0.1.
        causal: Causal convolution. Default False.
        use_macaron: Macaron-style FFN. Default True.
        nonlinear: Mask activation ('sigmoid', 'relu', 'tanh', 'none'). Default 'none' (CRM).
    """

    def __init__(
        self,
        n_fft: int = 512,
        num_blocks: int = 6,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.1,
        causal: bool = False,
        use_macaron: bool = True,
        nonlinear: str = "none",
        **kwargs,
    ):
        super().__init__()
        for key in ("hop_length", "win_length", "window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"Conformer got unexpected keyword arguments: {sorted(kwargs)}")
        self.n_fft = n_fft
        freq_dim = n_fft // 2 + 1
        input_dim = freq_dim * 2  # real + imag concatenated

        # Input projection: (B, T, F*2) -> (B, T, d_model)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = RelPositionalEncoding(d_model, dropout=dropout)

        # Conformer blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                dropout=dropout,
                causal=causal,
                use_macaron=use_macaron,
            )
            for _ in range(num_blocks)
        ])

        # Output projection: (B, T, d_model) -> (B, T, F*2) -> complex mask
        self.output_proj = nn.Linear(d_model, input_dim)

        # Mask activation
        if nonlinear == "sigmoid":
            self.mask_act = nn.Sigmoid()
        elif nonlinear == "relu":
            self.mask_act = nn.ReLU()
        elif nonlinear == "tanh":
            self.mask_act = nn.Tanh()
        else:
            self.mask_act = nn.Identity()  # CRM: no activation

    @fp32_forward
    def forward(self, noisy_spec: torch.Tensor, lengths=None, **kw) -> torch.Tensor:
        """noisy_spec: (B, F, T, 2). Returns enhanced_spec (B, F, T, 2)."""
        if noisy_spec.dim() != 4 or noisy_spec.shape[-1] != 2:
            raise ValueError(f"Expected [B, F, T, 2], got {tuple(noisy_spec.shape)}")
        if noisy_spec.shape[1] != self.n_fft // 2 + 1:
            raise ValueError(f"Expected F={self.n_fft // 2 + 1}, got F={noisy_spec.shape[1]}")
        if noisy_spec.shape[0] == 0 or noisy_spec.shape[2] == 0:
            raise ValueError("Conformer requires non-empty batch and time dimensions")
        if not noisy_spec.is_floating_point():
            raise TypeError("Conformer noisy_spec must use a floating-point dtype")
        noisy_spec = noisy_spec.float()
        B, F, T, _ = noisy_spec.shape

        # (B, F, T, 2) -> (B, T, F*2)
        x = noisy_spec.permute(0, 2, 1, 3).reshape(B, T, F * 2)

        # Input projection + positional encoding
        x = self.input_proj(x)
        x, pos_emb = self.pos_enc(x)

        # Conformer blocks
        for block in self.blocks:
            x = block(x, pos_emb)

        # Output mask: (B, T, F*2) -> (B, T, F, 2)
        mask = self.output_proj(x).view(B, T, F, 2)
        mask = self.mask_act(mask)

        # Permute mask back to (B, F, T, 2) and apply complex mask
        mask = mask.permute(0, 2, 1, 3)
        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        enh_real = noisy_spec[..., 0] * mask[..., 0] - noisy_spec[..., 1] * mask[..., 1]
        enh_imag = noisy_spec[..., 0] * mask[..., 1] + noisy_spec[..., 1] * mask[..., 0]
        return torch.stack([enh_real, enh_imag], dim=-1)


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 验证本模型参数量与算力。运行: python tasks/se/models/conformer.py [--sr 16000] [--dur 1.0] [--device cpu]
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
        "num_blocks": 6,
        "d_model": 256,
        "n_heads": 4,
        "d_ff": 1024,
        "depthwise_conv_kernel_size": 31,
    }

    model = Conformer(**MODEL_ARGS)
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
        label="Conformer",
    )
