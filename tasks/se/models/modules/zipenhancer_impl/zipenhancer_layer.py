"""Dual-Path Zipformer2 encoder with Down-Up sampling for ZipEnhancer.

This module stacks Zipformer2EncoderLayers in a dual-path fashion,
alternating between frequency-domain and time-domain modeling, and
supports symmetric down-up sampling in both axes to reduce FLOPs.

论文参考:
  - Z. Yao et al., "Zipformer: A faster and better encoder for ASR", ICLR 2024.
  - H. Wang & B. Tian, "ZipEnhancer", arXiv:2501.05183.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from tasks.se.models.modules.zipenhancer_impl.scaling import (
    BypassModule,
    FloatLike,
    ScheduledFloat,
    SimpleDownsample,
    SimpleUpsample,
    convert_num_channels,
)
from tasks.se.models.modules.zipenhancer_impl.zipformer import (
    CompactRelPositionalEncoding,
    Zipformer2EncoderLayer,
)


# ---------------------------------------------------------------------------
# DualPathZipformer2Encoder: alternating F/T Zipformer layers
# ---------------------------------------------------------------------------

class DualPathZipformer2Encoder(nn.Module):
    """Dual-path encoder: F-ZipformerBlock -> T-ZipformerBlock per layer.

    Input / output shape: ``(B, C, T, F)``.
    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        pos_dim: int,
        dropout: float,
        warmup_begin: float,
        warmup_end: float,
        initial_layerdrop_rate: float = 0.5,
        final_layerdrop_rate: float = 0.05,
        bypass_layer: Optional[nn.Module] = None,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.encoder_pos = CompactRelPositionalEncoding(
            pos_dim, dropout_rate=0.15, length_factor=1.0
        )
        self.f_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.t_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.bypass_layers = nn.ModuleList(
            [copy.deepcopy(bypass_layer) for _ in range(num_layers * 2)]
        )
        self.num_layers = num_layers

        delta = (1.0 / num_layers) * (warmup_end - warmup_begin)
        cur_begin = warmup_begin
        for i in range(num_layers):
            cur_end = cur_begin + delta
            self.f_layers[i].bypass.skip_rate = ScheduledFloat(
                (cur_begin, initial_layerdrop_rate),
                (cur_end, final_layerdrop_rate),
                default=0.0,
            )
            self.t_layers[i].bypass.skip_rate = ScheduledFloat(
                (cur_begin, initial_layerdrop_rate),
                (cur_end, final_layerdrop_rate),
                default=0.0,
            )
            cur_begin = cur_end

    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        feature_mask: Union[Tensor, float] = 1.0,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        b, c, t, f = src.size()
        # Flatten batch+other_axis for attention
        src_f = src.permute(3, 0, 2, 1).contiguous().view(f, b * t, c)
        src_t = src.permute(2, 0, 3, 1).contiguous().view(t, b * f, c)
        pos_emb_f = self.encoder_pos(src_f)
        pos_emb_t = self.encoder_pos(src_t)

        output = src
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            output = output * feature_mask

        # Generate causal mask for time path (seq_len = t, batch = b*f)
        causal_mask = None
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(t, t, device=src.device, dtype=torch.bool), diagonal=1
            )
            causal_mask = causal_mask.unsqueeze(0).expand(b * f, t, t)

        for i in range(self.num_layers):
            # F-path (frequency axis: no causal mask needed)
            output_f_org = output.permute(3, 2, 0, 1).contiguous()  # (f, t, b, c)
            output_f = output_f_org.view(f, t * b, c)
            output_f = self.f_layers[i](
                output_f,
                pos_emb_f,
                attn_mask=attn_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
            output_f = output_f.view(f, t, b, c)
            output_f = self.bypass_layers[i * 2](output_f_org, output_f)
            output = output_f.permute(2, 3, 1, 0).contiguous()  # (b, c, t, f)

            # T-path (time axis: apply causal mask)
            output_t_org = output.permute(2, 3, 0, 1).contiguous()  # (t, f, b, c)
            output_t = output_t_org.view(t, f * b, c)
            output_t = self.t_layers[i](
                output_t,
                pos_emb_t,
                attn_mask=causal_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
            output_t = output_t.view(t, f, b, c)
            output_t = self.bypass_layers[i * 2 + 1](output_t_org, output_t)
            output = output_t.permute(2, 3, 0, 1).contiguous()  # (b, c, t, f)

            if not torch.jit.is_scripting() and not torch.jit.is_tracing():
                output = output * feature_mask

        return output


# ---------------------------------------------------------------------------
# DualPathDownsampledZipformer2Encoder: down-up wrapper
# ---------------------------------------------------------------------------

class DualPathDownsampledZipformer2Encoder(nn.Module):
    """Wraps a DualPathZipformer2Encoder with T/F down-up sampling.

    Downsamples before the encoder, upsamples after, and combines with
    the original input via a BypassModule.
    """

    def __init__(
        self,
        encoder: nn.Module,
        dim: int,
        t_downsample: int,
        f_downsample: int,
        dropout: FloatLike,
    ):
        super().__init__()
        self.t_downsample_factor = t_downsample
        self.f_downsample_factor = f_downsample
        if self.t_downsample_factor != 1:
            self.downsample_t = SimpleDownsample(dim, t_downsample, dropout)
            self.upsample_t = SimpleUpsample(dim, t_downsample)
        if self.f_downsample_factor != 1:
            self.downsample_f = SimpleDownsample(dim, f_downsample, dropout)
            self.upsample_f = SimpleUpsample(dim, f_downsample)
        self.encoder = encoder
        self.out_combiner = BypassModule(dim, straight_through_rate=0)

    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        feature_mask: Union[Tensor, float] = 1.0,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        b, c, t, f = src.size()
        src_orig = src.permute(2, 3, 0, 1)  # (t, f, b, c)

        # ---- T-downsample: (b, c, t, f) -> (b, c, t//ds, f) -----------------
        # Step 1: flatten batch so each token sequence is independent.
        src_t = src.permute(2, 0, 3, 1).contiguous().view(t, b * f, c)
        if self.t_downsample_factor != 1:
            src_t = self.downsample_t(src_t)          # (t//ds, b*f, c)
        downsample_t = src_t.size(0)
        # Step 2: reshape to (f, b*downsample_t, c) so we can F-downsample next.
        src_t = (
            src_t.view(downsample_t, b, f, c)
            .permute(2, 1, 0, 3)
            .contiguous()
            .view(f, b * downsample_t, c)
        )

        # ---- F-downsample: (f, b*downsample_t, c) -> (f//ds, b*downsample_t, c)
        if self.f_downsample_factor != 1:
            src_t = self.downsample_f(src_t)          # (f//ds, b*downsample_t, c)
        downsample_f = src_t.size(0)
        # Step 3: reshape back to 4D image for the dual-path encoder.
        src_t = (
            src_t.view(downsample_f, b, downsample_t, c)
            .permute(1, 3, 2, 0)
            .contiguous()
        )  # (b, c, downsample_t, downsample_f)

        # ---- Dual-path encoder -----------------------------------------------
        src_t = self.encoder(
            src_t,
            chunk_size=chunk_size,
            feature_mask=feature_mask,
            attn_mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )  # (b, c, downsample_t, downsample_f)

        # ---- F-upsample: (b, c, downsample_t, downsample_f) -> (b, c, downsample_t, f)
        src_t = (
            src_t.permute(3, 0, 2, 1)
            .contiguous()
            .view(downsample_f, b * downsample_t, c)
        )
        if self.f_downsample_factor != 1:
            src_t = self.upsample_f(src_t)            # (f, b*downsample_t, c)
        src_t = (
            src_t[:f]
            .view(f, b, downsample_t, c)
            .permute(2, 1, 0, 3)
            .contiguous()
            .view(downsample_t, b * f, c)
        )

        # ---- T-upsample: (downsample_t, b*f, c) -> (t, f, b, c) --------------
        if self.t_downsample_factor != 1:
            src_t = self.upsample_t(src_t)            # (t, b*f, c)
        src_t = (
            src_t[:t]
            .view(t, b, f, c)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # (t, f, b, c)

        out = self.out_combiner(src_orig, src_t)
        return out.permute(2, 3, 0, 1).contiguous()  # (b, c, t, f)


# ---------------------------------------------------------------------------
# Zipformer2DualPathEncoder: top-level encoder with stacked blocks
# ---------------------------------------------------------------------------

class Zipformer2DualPathEncoder(nn.Module):
    """Top-level dual-path encoder assembling multiple downsampled blocks.

    Args:
        output_downsampling_factor: final downsample factor (unused in ZipEnhancer).
        downsampling_factor: tuple of T-downsample ratios per block.
        f_downsampling_factor: tuple of F-downsample ratios per block.
        encoder_dim: embedding dimension per block.
        num_encoder_layers: number of dual-path layers per block.
        encoder_unmasked_dim: unused in this simplified version (kept for API).
        query_head_dim: query dimension per head.
        pos_head_dim: positional encoding projection dim per head.
        value_head_dim: value dimension per head.
        num_heads: number of attention heads.
        feedforward_dim: FFN hidden dimension per block.
        cnn_module_kernel: depthwise conv kernel size per block.
        pos_dim: relative positional encoding dimension.
        dropout: dropout schedule.
        warmup_batches: warmup period for layerdrop scheduling.
        causal: whether to use causal convolutions.
    """

    def __init__(
        self,
        output_downsampling_factor: int = 2,
        downsampling_factor: Tuple[int] = (2, 4),
        f_downsampling_factor: Optional[Tuple[int]] = None,
        encoder_dim: Union[int, Tuple[int]] = 384,
        num_encoder_layers: Union[int, Tuple[int]] = 4,
        encoder_unmasked_dim: Union[int, Tuple[int]] = 256,
        query_head_dim: Union[int, Tuple[int]] = 24,
        pos_head_dim: Union[int, Tuple[int]] = 4,
        value_head_dim: Union[int, Tuple[int]] = 12,
        num_heads: Union[int, Tuple[int]] = 8,
        feedforward_dim: Union[int, Tuple[int]] = 1536,
        cnn_module_kernel: Union[int, Tuple[int]] = 31,
        pos_dim: int = 192,
        dropout: FloatLike = None,
        warmup_batches: float = 4000.0,
        causal: bool = False,
        chunk_size: Tuple[int] = (-1,),
        left_context_frames: Tuple[int] = (-1,),
    ):
        super().__init__()
        if dropout is None:
            dropout = ScheduledFloat((0.0, 0.3), (20000.0, 0.1))

        def _to_tuple(x):
            if isinstance(x, int):
                x = (x,)
            if len(x) == 1:
                x = x * len(downsampling_factor)
            else:
                assert len(x) == len(downsampling_factor) and isinstance(x[0], int)
            return x

        self.output_downsampling_factor = output_downsampling_factor
        self.downsampling_factor = downsampling_factor
        if f_downsampling_factor is None:
            f_downsampling_factor = downsampling_factor
        self.f_downsampling_factor = _to_tuple(f_downsampling_factor)
        self.encoder_dim = encoder_dim = _to_tuple(encoder_dim)
        self.encoder_unmasked_dim = encoder_unmasked_dim = _to_tuple(
            encoder_unmasked_dim
        )
        num_encoder_layers = _to_tuple(num_encoder_layers)
        self.num_encoder_layers = num_encoder_layers
        self.query_head_dim = query_head_dim = _to_tuple(query_head_dim)
        self.value_head_dim = value_head_dim = _to_tuple(value_head_dim)
        pos_head_dim = _to_tuple(pos_head_dim)
        self.num_heads = num_heads = _to_tuple(num_heads)
        feedforward_dim = _to_tuple(feedforward_dim)
        self.cnn_module_kernel = cnn_module_kernel = _to_tuple(cnn_module_kernel)
        self.causal = causal
        self.chunk_size = chunk_size
        self.left_context_frames = left_context_frames

        for u, d in zip(encoder_unmasked_dim, encoder_dim):
            assert u <= d

        encoders = []
        num_encoders = len(downsampling_factor)
        for i in range(num_encoders):
            encoder_layer = Zipformer2EncoderLayer(
                embed_dim=encoder_dim[i],
                pos_dim=pos_dim,
                num_heads=num_heads[i],
                query_head_dim=query_head_dim[i],
                pos_head_dim=pos_head_dim[i],
                value_head_dim=value_head_dim[i],
                feedforward_dim=feedforward_dim[i],
                dropout=dropout,
                cnn_module_kernel=cnn_module_kernel[i],
                causal=causal,
            )
            encoder = DualPathZipformer2Encoder(
                encoder_layer,
                num_encoder_layers[i],
                pos_dim=pos_dim,
                dropout=dropout,
                warmup_begin=warmup_batches * (i + 1) / (num_encoders + 1),
                warmup_end=warmup_batches * (i + 2) / (num_encoders + 1),
                final_layerdrop_rate=0.035 * (downsampling_factor[i] ** 0.5),
                bypass_layer=BypassModule(encoder_dim[i], straight_through_rate=0),
                causal=causal,
            )
            if downsampling_factor[i] != 1 or self.f_downsampling_factor[i] != 1:
                encoder = DualPathDownsampledZipformer2Encoder(
                    encoder,
                    dim=encoder_dim[i],
                    t_downsample=downsampling_factor[i],
                    f_downsample=self.f_downsampling_factor[i],
                    dropout=dropout,
                )
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)
        self.downsample_output = SimpleDownsample(
            max(encoder_dim), downsample=output_downsampling_factor, dropout=dropout
        )

    def forward(self, x: Tensor) -> Tensor:
        feature_masks = [1.0] * len(self.encoder_dim)
        b, c, t, f = x.shape
        causal_mask = None
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1
            )
            causal_mask = causal_mask.unsqueeze(0).expand(b * f, t, t)
        for i, module in enumerate(self.encoders):
            x = convert_num_channels(x, self.encoder_dim[i])
            x = module(
                x,
                chunk_size=-1,
                feature_mask=feature_masks[i],
                src_key_padding_mask=None,
                attn_mask=causal_mask,
            )
        return x
