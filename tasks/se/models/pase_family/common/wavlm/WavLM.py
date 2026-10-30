# --------------------------------------------------------
# WavLM: Large-Scale Self-Supervised Pre-training for Full Stack Speech Processing
# Github source: https://github.com/microsoft/unilm/tree/master/wavlm
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# Based on fairseq code bases: https://github.com/pytorch/fairseq
# --------------------------------------------------------
# Ported into use_train framework for PASE reproduction.

import ast
import math
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from .modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    MultiheadAttention,
    SamePad,
    init_bert_params,
    get_activation_fn,
    TransposeLast,
    GLU_Linear,
)

logger = logging.getLogger(__name__)


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
) -> np.ndarray:
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)
    all_num_mask = int(mask_prob * all_sz / float(mask_length) + np.random.rand())
    all_num_mask = max(min_masks, all_num_mask)
    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(mask_prob * sz / float(mask_length) + np.random.rand())
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask
        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)
        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)
        if no_overlap:
            mask_idc = []
            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))
                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts
            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts), np.int64,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1
            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)
            mask_idc = np.asarray([
                mask_idc[j] + offset
                for j in range(len(mask_idc))
                for offset in range(lengths[j])
            ])
        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))
    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True
    return mask


class WavLMConfig:
    def __init__(self, cfg=None):
        self.extractor_mode: str = "default"
        self.encoder_layers: int = 12
        self.encoder_embed_dim: int = 768
        self.encoder_ffn_embed_dim: int = 3072
        self.encoder_attention_heads: int = 12
        self.activation_fn: str = "gelu"
        self.layer_norm_first: bool = False
        self.conv_feature_layers: str = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"
        self.conv_bias: bool = False
        self.feature_grad_mult: float = 1.0
        self.normalize: bool = False
        self.dropout: float = 0.1
        self.attention_dropout: float = 0.1
        self.activation_dropout: float = 0.0
        self.encoder_layerdrop: float = 0.0
        self.dropout_input: float = 0.0
        self.dropout_features: float = 0.0
        self.mask_length: int = 10
        self.mask_prob: float = 0.65
        self.mask_selection: str = "static"
        self.mask_other: float = 0
        self.no_mask_overlap: bool = False
        self.mask_min_space: int = 1
        self.mask_channel_length: int = 10
        self.mask_channel_prob: float = 0.0
        self.mask_channel_selection: str = "static"
        self.mask_channel_other: float = 0
        self.no_mask_channel_overlap: bool = False
        self.mask_channel_min_space: int = 1
        self.conv_pos: int = 128
        self.conv_pos_groups: int = 16
        self.relative_position_embedding: bool = False
        self.num_buckets: int = 320
        self.max_distance: int = 1280
        self.gru_rel_pos: bool = False
        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


def _eval_conv_layer_expression(node: ast.AST, source: str):
    """受限求值：只执行列表/元组字面量、int 常量与 ``+`` / ``*``。"""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"WavLM cfg.conv_feature_layers 含非法常量: {node.value!r}")
    if isinstance(node, ast.List):
        return [_eval_conv_layer_expression(elt, source) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_conv_layer_expression(elt, source) for elt in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
        left = _eval_conv_layer_expression(node.left, source)
        right = _eval_conv_layer_expression(node.right, source)
        if isinstance(node.op, ast.Add):
            if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
                raise ValueError(
                    f"WavLM cfg.conv_feature_layers 的 + 只支持序列拼接: {source!r}"
                )
            return left + right
        if isinstance(left, int) and isinstance(right, (list, tuple)):
            return right * left
        if isinstance(left, (list, tuple)) and isinstance(right, int):
            return left * right
        raise ValueError(
            f"WavLM cfg.conv_feature_layers 的 * 只支持 序列 × int: {source!r}"
        )
    raise ValueError(
        f"WavLM cfg.conv_feature_layers 含不支持的表达式 {type(node).__name__}: {source!r}"
    )


def parse_conv_feature_layers(value: object) -> Tuple[Tuple[int, int, int], ...]:
    """解析 checkpoint 的 ``conv_feature_layers``，禁止 ``eval`` 执行任意代码。

    官方 checkpoint 把该字段存成字符串（默认
    ``"[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"``），上游用 ``eval``
    求值；这里改为受限 AST 求值并校验结构，防止不可信 checkpoint 触发代码执行。
    """
    if isinstance(value, str):
        try:
            tree = ast.parse(value, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"WavLM cfg.conv_feature_layers 无法解析: {value!r}") from exc
        parsed = _eval_conv_layer_expression(tree.body, value)
    else:
        parsed = value
    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise ValueError(f"WavLM cfg.conv_feature_layers 必须是非空列表: {parsed!r}")
    layers = []
    for layer in parsed:
        if (
            not isinstance(layer, (list, tuple))
            or len(layer) != 3
            or not all(isinstance(item, int) and not isinstance(item, bool) for item in layer)
        ):
            raise ValueError(
                f"WavLM conv layer 必须是 (dim, kernel, stride) 整数三元组: {layer!r}"
            )
        layers.append(tuple(layer))
    return tuple(layers)


class WavLM(nn.Module):
    def __init__(self, cfg: WavLMConfig) -> None:
        super().__init__()
        logger.info(f"WavLM Config: {cfg.__dict__}")
        self.cfg = cfg
        feature_enc_layers = parse_conv_feature_layers(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )
        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space
        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)
        self.feature_grad_mult = cfg.feature_grad_mult
        self.mask_emb = nn.Parameter(torch.FloatTensor(cfg.encoder_embed_dim).uniform_())
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

    def apply_mask(self, x, mask_indices=None, padding_mask=None):
        B, T, C = x.shape
        if mask_indices is not None:
            if isinstance(mask_indices, np.ndarray):
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            # autocast 下 x 可能是 bf16/fp16，而 mask_emb 保持 fp32，索引赋值要求两侧 dtype 一致
            x[mask_indices] = self.mask_emb.to(device=x.device, dtype=x.dtype)
        else:
            if self.mask_prob > 0:
                mask_indices = compute_mask_indices(
                    (B, T), padding_mask, self.mask_prob, self.mask_length,
                    self.mask_selection, self.mask_other, min_masks=2,
                    no_overlap=self.no_mask_overlap, min_space=self.mask_min_space,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
                x[mask_indices] = self.mask_emb.to(device=x.device, dtype=x.dtype)
            else:
                mask_indices = None
            if self.mask_channel_prob > 0:
                mask_channel_indices = compute_mask_indices(
                    (B, C), None, self.mask_channel_prob, self.mask_channel_length,
                    self.mask_channel_selection, self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap, min_space=self.mask_channel_min_space,
                )
                mask_channel_indices = (
                    torch.from_numpy(mask_channel_indices).to(x.device).unsqueeze(1).expand(-1, T, -1)
                )
                x[mask_channel_indices] = 0
        return x, mask_indices

    def forward_padding_mask(self, features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        mask_indices=None,
        output_layer: Optional[int] = None,
    ):
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        features = self.dropout_input(features)
        if mask:
            x, mask_indices = self.apply_mask(features, mask_indices, padding_mask)
        else:
            x = features
            mask_indices = None
        x, layer_results = self.encoder(
            x, padding_mask=padding_mask, layer=None if output_layer is None else output_layer - 1
        )
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        res = {"x": x, "padding_mask": padding_mask, "features": features, "layer_reps": layer_reps}
        return res, mask_indices


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        conv_type: str = "default"
    ):
        super().__init__()
        assert mode in {"default", "layer_norm"}

        def block(n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            assert (is_layer_norm and is_group_norm) == False
            if is_layer_norm:
                return nn.Sequential(
                    make_conv(), nn.Dropout(p=dropout),
                    nn.Sequential(TransposeLast(), Fp32LayerNorm(dim, elementwise_affine=True), TransposeLast()),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), Fp32GroupNorm(dim, dim, affine=True), nn.GELU())
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        self.conv_type = conv_type
        if self.conv_type == "default":
            in_d = 1
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl
                self.conv_layers.append(
                    block(in_d, dim, k, stride, is_layer_norm=mode == "layer_norm",
                          is_group_norm=mode == "default" and i == 0, conv_bias=conv_bias)
                )
                in_d = dim
        elif self.conv_type == "conv2d":
            in_d = 1
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl
                self.conv_layers.append(torch.nn.Conv2d(in_d, dim, k, stride))
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
        elif self.conv_type == "custom":
            in_d = 1
            idim = 80
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl
                self.conv_layers.append(torch.nn.Conv2d(in_d, dim, k, stride, padding=1))
                self.conv_layers.append(torch.nn.LayerNorm([dim, idim]))
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
                if (i + 1) % 2 == 0:
                    self.conv_layers.append(torch.nn.MaxPool2d(2, stride=2, ceil_mode=True))
                    idim = int(math.ceil(idim / 2))
        else:
            pass

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        if self.conv_type == "custom":
            for conv in self.conv_layers:
                if isinstance(conv, nn.LayerNorm):
                    x = x.transpose(1, 2)
                    x = conv(x).transpose(1, 2)
                else:
                    x = conv(x)
            x = x.transpose(2, 3).contiguous()
            x = x.view(x.size(0), -1, x.size(-1))
        else:
            for conv in self.conv_layers:
                x = conv(x)
            if self.conv_type == "conv2d":
                b, c, t, f = x.size()
                x = x.transpose(2, 3).contiguous().view(b, c * f, t)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.pos_conv = nn.Conv1d(
            self.embedding_dim, self.embedding_dim,
            kernel_size=args.conv_pos, padding=args.conv_pos // 2, groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)
        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())
        if hasattr(args, "relative_position_embedding"):
            self.relative_position_embedding = args.relative_position_embedding
            self.num_buckets = args.num_buckets
            self.max_distance = args.max_distance
        else:
            self.relative_position_embedding = False
            self.num_buckets = 0
            self.max_distance = 0
        self.layers = nn.ModuleList([
            TransformerSentenceEncoderLayer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=self.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                activation_fn=args.activation_fn,
                layer_norm_first=args.layer_norm_first,
                has_relative_attention_bias=(self.relative_position_embedding and i == 0),
                num_buckets=self.num_buckets,
                max_distance=self.max_distance,
                gru_rel_pos=args.gru_rel_pos,
            )
            for i in range(args.encoder_layers)
        ])
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop
        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, streaming_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, streaming_mask, layer)
        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)
        return x, layer_results

    def extract_features(self, x, padding_mask=None, streaming_mask=None, tgt_layer=None):
        if padding_mask is not None:
            x[padding_mask] = 0
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv
        if not self.layer_norm_first:
            x = self.layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        layer_results = []
        z = None
        if tgt_layer is not None:
            layer_results.append((x, z))
        r = None
        pos_bias = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z, pos_bias = layer(x, self_attn_padding_mask=padding_mask, need_weights=False,
                                       self_attn_mask=streaming_mask, pos_bias=pos_bias)
            if tgt_layer is not None:
                layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break
        if r is not None:
            x = r
        x = x.transpose(0, 1)
        return x, layer_results


class TransformerSentenceEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 0,
        max_distance: int = 0,
        rescale_init: bool = False,
        gru_rel_pos: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim, num_attention_heads, dropout=attention_dropout,
            self_attention=True, has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets, max_distance=max_distance, rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm_first = layer_norm_first
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        if self.activation_name == "glu":
            self.fc1 = GLU_Linear(self.embedding_dim, ffn_embedding_dim, "swish")
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None, need_weights=False, pos_bias=None):
        residual = x
        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn, pos_bias = self.self_attn(
                query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask,
                need_weights=False, attn_mask=self_attn_mask, position_bias=pos_bias
            )
            x = self.dropout1(x)
            x = residual + x
            residual = x
            x = self.final_layer_norm(x)
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn, pos_bias = self.self_attn(
                query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights, attn_mask=self_attn_mask, position_bias=pos_bias
            )
            x = self.dropout1(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)
            residual = x
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)
        return x, attn, pos_bias
