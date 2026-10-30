"""Zipformer2 encoder layer and building blocks for ZipEnhancer.

Implements the Zipformer2EncoderLayer with:
  - Multi-Head Attention Weights (MHAW) shared across SA and NLA
  - Non-Linear Attention (NLA)
  - Two Self-Attention modules
  - Three Feed-Forward modules
  - Two Convolution modules
  - Bypass modules and BiasNorm

All parameter dimensions and initializations are strictly aligned with the
original Zipformer2 (ICLR 2024) to maintain the 2.04 M parameter budget.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from tasks.se.models.modules.zipenhancer_impl.scaling import (
    ActivationDropoutAndLinear,
    BiasNorm,
    BypassModule,
    ChunkCausalDepthwiseConv1d,
    Dropout2,
    FloatLike,
    ScheduledFloat,
    ScaledLinear,
    SimpleDownsample,
    SimpleUpsample,
    convert_num_channels,
    penalize_abs_values_gt,
)


# ---------------------------------------------------------------------------
# FeedforwardModule
# ---------------------------------------------------------------------------

class FeedforwardModule(nn.Module):
    """FFN with SwooshL activation and dropout."""

    def __init__(self, embed_dim: int, feedforward_dim: int, dropout: FloatLike):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, feedforward_dim)
        self.out_proj = ActivationDropoutAndLinear(
            feedforward_dim,
            embed_dim,
            activation="SwooshL",
            dropout_p=dropout,
            dropout_shared_dim=0,
            bias=True,
            initial_scale=0.1,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.out_proj(self.in_proj(x))


# ---------------------------------------------------------------------------
# SelfAttention: consumes pre-computed attention weights
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Simple attention using externally-provided attention weights."""

    def __init__(self, embed_dim: int, num_heads: int, value_head_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, num_heads * value_head_dim, bias=True)
        self.out_proj = ScaledLinear(
            num_heads * value_head_dim, embed_dim, bias=True, initial_scale=0.05
        )

    def forward(self, x: Tensor, attn_weights: Tensor) -> Tensor:
        seq_len, batch_size, embed_dim = x.shape
        num_heads = attn_weights.shape[0]
        x = self.in_proj(x)
        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        value_head_dim = x.shape[-1]
        x = torch.matmul(attn_weights, x)
        x = (
            x.permute(2, 1, 0, 3)
            .contiguous()
            .view(seq_len, batch_size, num_heads * value_head_dim)
        )
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# NonlinAttention: attention-based non-linear convolution replacement
# ---------------------------------------------------------------------------

class NonlinAttention(nn.Module):
    """Non-Linear Attention: O = linear(A * attention(tanh(B) * C))."""

    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.in_proj = nn.Linear(channels, hidden_channels * 3, bias=True)
        self.tanh = nn.Tanh()
        self.out_proj = ScaledLinear(
            hidden_channels, channels, bias=True, initial_scale=0.05
        )

    def forward(self, x: Tensor, attn_weights: Tensor) -> Tensor:
        seq_len, batch_size, _ = x.shape
        x = self.in_proj(x)
        s, x, y = x.chunk(3, dim=2)
        s = self.tanh(s)
        s = s.unsqueeze(-1).reshape(seq_len, batch_size, self.hidden_channels)
        x = x * s
        num_heads = attn_weights.shape[0]
        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        x = torch.matmul(attn_weights, x)
        x = x.permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1)
        x = x * y
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# ConvolutionModule
# ---------------------------------------------------------------------------

class ConvolutionModule(nn.Module):
    """Conformer-style convolution with GLU gating and depthwise conv."""

    def __init__(self, embed_dim: int, kernel_size: int, causal: bool = False):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        self.causal = causal
        self.in_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.sigmoid = nn.Sigmoid()
        self.depthwise_conv = (
            ChunkCausalDepthwiseConv1d(
                channels=embed_dim, kernel_size=kernel_size
            )
            if causal
            else nn.Conv1d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                groups=embed_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )
        self.out_proj = ActivationDropoutAndLinear(
            embed_dim,
            embed_dim,
            activation="SwooshR",
            dropout_p=0.0,
            initial_scale=0.05,
        )

    def forward(
        self, x: Tensor, src_key_padding_mask: Optional[Tensor] = None, chunk_size: int = -1
    ) -> Tensor:
        x = self.in_proj(x)
        x, s = x.chunk(2, dim=2)
        s = self.sigmoid(s)
        x = x * s
        x = x.permute(1, 2, 0)  # (batch, channels, time)
        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)
        if self.causal and chunk_size >= 0 and not torch.jit.is_scripting():
            x = self.depthwise_conv(x, chunk_size=chunk_size)
        else:
            x = self.depthwise_conv(x)
        x = x.permute(2, 0, 1)  # (time, batch, channels)
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# Compact relative positional encoding
# ---------------------------------------------------------------------------

class CompactRelPositionalEncoding(nn.Module):
    """Compact relative positional encoding using atan + Fourier features."""

    def __init__(
        self,
        embed_dim: int,
        dropout_rate: FloatLike,
        max_len: int = 1000,
        length_factor: float = 1.0,
    ):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.dropout = Dropout2(dropout_rate)
        self.length_factor = length_factor
        self.pe: Optional[Tensor] = None
        self.extend_pe(torch.tensor(0.0).expand(max_len))

    def extend_pe(self, x: Tensor, left_context_len: int = 0) -> None:
        T = x.size(0) + left_context_len
        if self.pe is not None and self.pe.size(0) >= T * 2 - 1:
            self.pe = self.pe.to(dtype=x.dtype, device=x.device)
            return
        x = torch.arange(-(T - 1), T, device=x.device).to(torch.float32).unsqueeze(1)
        freqs = 1 + torch.arange(self.embed_dim // 2, device=x.device)
        compression_length = self.embed_dim**0.5
        _tmp = ((x.abs() + compression_length).log() - math.log(compression_length))
        x_compressed = compression_length * x.sign() * _tmp
        length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)
        x_atan = (x_compressed / length_scale).atan()
        cosines = (x_atan * freqs).cos()
        sines = (x_atan * freqs).sin()
        pe = torch.zeros(x.shape[0], self.embed_dim, device=x.device)
        pe[:, 0::2] = cosines
        pe[:, 1::2] = sines
        pe[:, -1] = 1.0
        self.pe = pe.to(dtype=x.dtype)

    def forward(self, x: Tensor, left_context_len: int = 0) -> Tensor:
        self.extend_pe(x, left_context_len)
        x_size_left = x.size(0) + left_context_len
        pos_emb = self.pe[
            self.pe.size(0) // 2 - x_size_left + 1 : self.pe.size(0) // 2 + x.size(0)
        ]
        pos_emb = pos_emb.unsqueeze(0)
        return self.dropout(pos_emb)


# ---------------------------------------------------------------------------
# RelPositionMultiheadAttentionWeights: computes shared attention weights
# ---------------------------------------------------------------------------

class RelPositionMultiheadAttentionWeights(nn.Module):
    """Compute multi-head attention weights with relative position encoding.

    The computed weights are shared by SelfAttention and NonlinAttention.
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        dropout: float = 0.0,
        pos_emb_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5), (4000.0, 0.0)),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.dropout = dropout
        self.pos_emb_skip_rate = pos_emb_skip_rate
        key_head_dim = query_head_dim
        in_proj_dim = (query_head_dim + key_head_dim + pos_head_dim) * num_heads
        self.in_proj = ScaledLinear(
            embed_dim, in_proj_dim, bias=True, initial_scale=query_head_dim**-0.25
        )
        self.linear_pos = ScaledLinear(
            pos_dim, num_heads * pos_head_dim, bias=False, initial_scale=0.05
        )

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.in_proj(x)
        query_head_dim = self.query_head_dim
        pos_head_dim = self.pos_head_dim
        num_heads = self.num_heads
        seq_len, batch_size, _ = x.shape
        query_dim = query_head_dim * num_heads
        q = x[..., 0:query_dim]
        k = x[..., query_dim : 2 * query_dim]
        p = x[..., 2 * query_dim :]
        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        p = p.reshape(seq_len, batch_size, num_heads, pos_head_dim)
        k = k.reshape(seq_len, batch_size, num_heads, query_head_dim)
        q = q.permute(2, 1, 0, 3)  # (head, batch, time1, d_q)
        p = p.permute(2, 1, 0, 3)  # (head, batch, time1, d_p)
        k = k.permute(2, 1, 3, 0)  # (head, batch, d_k, time2)
        attn_scores = torch.matmul(q, k)
        # positional scores
        use_pos_scores = False
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            use_pos_scores = True
        elif not self.training or torch.rand(1).item() >= float(self.pos_emb_skip_rate):
            use_pos_scores = True

        if use_pos_scores:
            pos_emb = self.linear_pos(pos_emb)
            seq_len2 = 2 * seq_len - 1
            pos_emb = pos_emb.reshape(-1, seq_len2, num_heads, pos_head_dim).permute(
                2, 0, 3, 1
            )
            pos_scores = torch.matmul(p, pos_emb)
            # convert relative to absolute positions
            if torch.jit.is_tracing():
                num_heads_, batch_size_, time1_, n = pos_scores.shape
                rows = torch.arange(start=time1_ - 1, end=-1, step=-1)
                cols = torch.arange(seq_len)
                rows = rows.repeat(batch_size_ * num_heads_).unsqueeze(-1)
                indexes = rows + cols
                pos_scores = pos_scores.reshape(-1, n)
                pos_scores = torch.gather(pos_scores, dim=1, index=indexes)
                pos_scores = pos_scores.reshape(num_heads_, batch_size_, time1_, seq_len)
            else:
                # Convert relative-position scores (shape: head, batch, time1, 2*time-1)
                # to absolute-position scores (shape: head, batch, time1, time2).
                #
                # The last axis of pos_scores represents relative offsets:
                #   index 0       -> offset = -(time1-1)
                #   index time1-1 -> offset = 0
                #   index 2*time1-2 -> offset = time1-1
                #
                # For absolute position (t1, t2), relative offset = t2 - t1.
                # We want pos_scores[..., t1, t2] = rel_scores[..., t1, offset=t2-t1]
                #   = rel_scores[..., t1, (time1-1) + (t2-t1)]
                #   = rel_scores[..., t1, (time1-1) + t2 - t1]
                #
                # Using as_strided with:
                #   stride[-2] = stride(2) - stride(3)  (step t1 and offset together)
                #   storage_offset = stride(3) * (time1 - 1)  (start at offset=0)
                pos_scores = pos_scores.as_strided(
                    (num_heads, batch_size, seq_len, seq_len),
                    (
                        pos_scores.stride(0),
                        pos_scores.stride(1),
                        pos_scores.stride(2) - pos_scores.stride(3),
                        pos_scores.stride(3),
                    ),
                    storage_offset=pos_scores.stride(3) * (seq_len - 1),
                )
            attn_scores = attn_scores + pos_scores

        if self.training and torch.rand(1).item() < 0.1:
            attn_scores = penalize_abs_values_gt(
                attn_scores, limit=25.0, penalty=1.0e-04, name=None)

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask, -1000)
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1), -1000
            )
        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        return attn_weights


# ---------------------------------------------------------------------------
# Zipformer2EncoderLayer
# ---------------------------------------------------------------------------

class Zipformer2EncoderLayer(nn.Module):
    """Single Zipformer2 encoder layer.

    Architecture (see Fig. 2 in ZipEnhancer paper):
      FFN1 -> NLA (with shared attn weights) -> SA1 -> Conv1 -> FFN2
      -> bypass_mid -> SA2 -> Conv2 -> FFN3 -> BiasNorm -> bypass
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        value_head_dim: int,
        feedforward_dim: int,
        dropout: FloatLike = 0.1,
        cnn_module_kernel: int = 31,
        causal: bool = False,
        attention_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0),
        conv_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0),
        const_attention_rate: FloatLike = ScheduledFloat(
            (0.0, 0.25), (4000.0, 0.025), default=0),
        ff2_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)),
        ff3_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)),
        bypass_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.5), (4000.0, 0.02), default=0),
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.bypass = BypassModule(
            embed_dim, skip_rate=bypass_skip_rate, straight_through_rate=0)
        self.bypass_mid = BypassModule(embed_dim, straight_through_rate=0)
        self.attention_skip_rate = attention_skip_rate
        self.conv_skip_rate = conv_skip_rate
        self.const_attention_rate = const_attention_rate
        self.ff2_skip_rate = ff2_skip_rate
        self.ff3_skip_rate = ff3_skip_rate
        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim,
            pos_dim=pos_dim,
            num_heads=num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            dropout=0.0,
        )
        self.self_attn1 = SelfAttention(embed_dim, num_heads, value_head_dim)
        self.self_attn2 = SelfAttention(embed_dim, num_heads, value_head_dim)
        self.feed_forward1 = FeedforwardModule(
            embed_dim, (feedforward_dim * 3) // 4, dropout
        )
        self.feed_forward2 = FeedforwardModule(embed_dim, feedforward_dim, dropout)
        self.feed_forward3 = FeedforwardModule(
            embed_dim, (feedforward_dim * 5) // 4, dropout
        )
        self.nonlin_attention = NonlinAttention(
            embed_dim, hidden_channels=3 * embed_dim // 4
        )
        self.conv_module1 = ConvolutionModule(
            embed_dim, cnn_module_kernel, causal=causal
        )
        self.conv_module2 = ConvolutionModule(
            embed_dim, cnn_module_kernel, causal=causal
        )
        # NOTE: 该参数不参与 forward，仅为对齐官方 checkpoint 的 state_dict（严格加载）。
        # 必须关闭梯度：未使用参数在 DDP（find_unused_parameters=False）下会让
        # backward 报错，单卡也只会浪费优化器状态。
        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5), requires_grad=False)
        self.norm = BiasNorm(embed_dim)

    def get_sequence_dropout_mask(self, x: Tensor, dropout_rate: float) -> Optional[Tensor]:
        if dropout_rate == 0.0 or not self.training or torch.jit.is_scripting() or torch.jit.is_tracing():
            return None
        batch_size = x.shape[1]
        mask = (torch.rand(batch_size, 1, device=x.device) > dropout_rate).to(x.dtype)
        return mask

    def sequence_dropout(self, x: Tensor, dropout_rate: float) -> Tensor:
        dropout_mask = self.get_sequence_dropout_mask(x, dropout_rate)
        if dropout_mask is None:
            return x
        return x * dropout_mask

    @staticmethod
    def _scheduled_rate(schedule: FloatLike) -> float:
        """Return scheduled rate, handling JIT and eval mode."""
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return 0.0
        return float(schedule)

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        chunk_size: int = -1,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        src_orig = src

        attention_skip_rate = self._scheduled_rate(self.attention_skip_rate) if self.training else 0.0

        attn_weights = self.self_attn_weights(
            src,
            pos_emb=pos_emb,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
        )

        src = src + self.feed_forward1(src)

        self_attn_dropout_mask = self.get_sequence_dropout_mask(src, attention_skip_rate)

        selected_attn_weights = attn_weights[0:1]
        if not torch.jit.is_scripting() and not torch.jit.is_tracing() and self.training and torch.rand(1).item() < float(self.const_attention_rate):
            selected_attn_weights = selected_attn_weights[0:1]
            selected_attn_weights = (selected_attn_weights > 0.0).to(selected_attn_weights.dtype)
            selected_attn_weights = selected_attn_weights * (1.0 / selected_attn_weights.sum(dim=-1, keepdim=True))

        na = self.nonlin_attention(src, selected_attn_weights)
        src = src + (na if self_attn_dropout_mask is None else na * self_attn_dropout_mask)

        self_attn = self.self_attn1(src, attn_weights)
        src = src + (self_attn if self_attn_dropout_mask is None else self_attn * self_attn_dropout_mask)

        conv_skip_rate = self._scheduled_rate(self.conv_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(
            self.conv_module1(src, src_key_padding_mask=src_key_padding_mask),
            conv_skip_rate,
        )

        ff2_skip_rate = self._scheduled_rate(self.ff2_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(self.feed_forward2(src), ff2_skip_rate)

        src = self.bypass_mid(src_orig, src)

        self_attn = self.self_attn2(src, attn_weights)
        src = src + (self_attn if self_attn_dropout_mask is None else self_attn * self_attn_dropout_mask)

        conv_skip_rate = self._scheduled_rate(self.conv_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(
            self.conv_module2(src, src_key_padding_mask=src_key_padding_mask),
            conv_skip_rate,
        )

        ff3_skip_rate = self._scheduled_rate(self.ff3_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(self.feed_forward3(src), ff3_skip_rate)

        src = self.norm(src)
        src = self.bypass(src_orig, src)
        return src
