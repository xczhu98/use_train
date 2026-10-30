# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
"""KCE（keyword-conditioned encoder）的 canonical TSE 文本 cue 前端。

来源：WeSep commit ``dc11cb18`` 的 ``wesep/modules/textual/kce.py``（Apache-2.0，
Copyright (c) 2026 Ke Zhang）。上游架构来自 GnafiY/DAE-TSE 的 KCE 实现：
https://github.com/GnafiY/DAE-TSE 。本文件保留上游模块结构、参数名与计算顺序，
仅做以下适配：

- **权重来源显式化**：backbone 权重只能来自 ``checkpoint``（经
  ``core.utils.safe_load.safe_torch_load``，``weights_only=True``）或显式注入的
  state dict，并做严格 key/shape 校验；两者都没有时构造即报错，``defer_pretrained``
  延迟加载在取得权重前 ``forward`` 报错。不存在随机初始化推理路径。
- **输入合同 typed cue 化**：phoneme id 行 ``[R, L]``、presence mask ``[R]`` 与
  per-row length 显式传入；缺失行仍走同一条计算图，只在输出处置零。
- **配置严格化**：官方 ``model.yaml`` / ``data.yaml`` 先映射到推理所需字段；仅显式
  丢弃 DAE 训练专用字段并补入上游构造器默认值，其他未知键仍直接拒绝。
- 上游的裸 ``torch.load`` 未复制。

IO 合同
-------
输入（全部在同一 device）：

- ``mixture``：mixture 参考通道波形 ``[R, T]``，16 kHz，浮点。
- ``phoneme_ids``：``[R, L]`` ``torch.int64``，行内有效段为 phoneme id。
- ``present``：``[R]`` ``torch.bool``，cue 是否可用。
- ``textual_lengths``：``[R]`` 整数，present 行取 ``[1, L]``，缺失行必须为 0。
- ``mixture_lengths``：``[R]`` 整数，每行有效波形样本数，取 ``[1, T]``。

输出：句级 cue ``[R, output_dim]``，缺失行为 0。

**model-owned fbank 例外（已登记）**：KCE 自带 Kaldi fbank 前端，直接对 mixture
波形调用 ``torchaudio.compliance.kaldi.fbank``（与上游逐帧一致；该接口不做
``2^15`` 缩放，与 ``tasks/tse/models/speaker/fbank.py`` 的缩放版本不同）。这是
文本 cue 前端的 model-owned 特征提取例外：本模块不创建 window、不执行 separator
STFT/iSTFT，也不消费框架 STFT 上下文。
"""

from __future__ import annotations

import inspect
import logging
import math
import os
import string
import unicodedata
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
import yaml
from torch.nn.utils.rnn import pad_sequence

from core.utils.safe_load import safe_torch_load


logger = logging.getLogger(__name__)

DEFAULT_KCE_MODEL_CONFIG = {
    "audio_net_config": {
        "num_mel_bins": 80,
        "input_trans": {
            "dim": [256, 256]
        },
        "transformer_config": {
            "size": 256,
            "self_att": "MultiHeadAtt",
            "self_att_config": {
                "n_head": 4,
                "n_feats": 256
            },
            "cross_att": "MultiHeadCrossAtt",
            "corss_att_config": {
                "n_head": 4,
                "n_feats": 256,
                "norm": "LayerNorm",
            },
            "feed_forward_config": {
                "dim": [256, 1024, 256]
            },
            "conv_config": {
                "channels": 256,
                "kernel_size": 15
            },
        },
    },
    "kw_net_config": {
        "num_phn_token": 73,
        "input_trans": {
            "dim": [256, 256]
        },
        "transformer_config": {
            "size": 256,
            "self_att": "MultiHeadAtt",
            "self_att_config": {
                "n_head": 4,
                "n_feats": 256
            },
            "feed_forward_config": {
                "dim": [256, 1024, 256]
            },
        },
    },
    "num_audio_block": 8,
    "num_kw_block": 3,
    "sv_net_config": {
        "num_classes": 251,
        "front_output_size": 256,
        "pooling_conf": {
            "type": "learnable_weights",
            "layer_indices": list(range(8)),
            "use_softmax": True,
        },
    },
}

DEFAULT_KCE_FBANK_CONFIG = {
    "num_mel_bins": 80,
    "frame_length": 25.0,
    "frame_shift": 10.0,
    "dither": 0.0,
    "sample_frequency": 16000.0,
    "window_type": "hamming",
    "use_energy": False,
}

KCE_SAMPLE_RATE = 16000
_KCE_CHECKPOINT_PURPOSE = "KCE official backbone checkpoint (trusted DAE-TSE release)"
_CHECKPOINT_KEY = "model"

# 前端只支持 16 kHz KCE 预训练前端；其余采样率必须在配置边界被拒绝。
_SUPPORTED_SAMPLE_RATES = frozenset({KCE_SAMPLE_RATE})

_MODEL_CONFIG_KEYS = frozenset(
    {"audio_net_config", "kw_net_config", "num_audio_block", "num_kw_block",
     "loss_weight", "sv_net_config"}
)
_AUDIO_NET_KEYS = frozenset({"num_mel_bins", "input_trans", "transformer_config"})
_KW_NET_KEYS = frozenset({"num_phn_token", "input_trans", "transformer_config"})
_AUDIO_TRANSFORMER_KEYS = frozenset(
    {"size", "self_att", "self_att_config", "cross_att", "corss_att_config",
     "feed_forward_config", "conv_config"}
)
_KW_TRANSFORMER_KEYS = frozenset(
    {"size", "self_att", "self_att_config", "feed_forward_config"}
)
_SELF_ATT_CONFIG_KEYS = frozenset({"n_head", "n_feats"})
_CROSS_ATT_CONFIG_KEYS = frozenset({"n_head", "n_feats", "norm"})
_INPUT_TRANS_KEYS = frozenset({"dim", "num_block", "bias", "norm", "act"})
_FEED_FORWARD_KEYS = frozenset({"dim", "num_block", "bias", "norm", "act"})
_CONV_CONFIG_KEYS = frozenset(
    {"channels", "kernel_size", "activation", "norm", "causal", "bias"}
)
_SV_NET_KEYS = frozenset(
    {"num_classes", "front_output_size", "pooling_conf", "use_reg_loss",
     "reg_loss_weight", "reduce"}
)
_POOLING_CONF_KEYS = frozenset({"type", "layer_indices", "use_softmax"})

_KCE_CONFIG_KEYS = frozenset(
    {"checkpoint", "model_config", "fbank", "eps", "adapter", "padding_id",
     "bos_id", "eos_id"}
)
_ADAPTER_KEYS = frozenset({"input_dim", "output_dim", "num_layers"})
_OFFICIAL_UNUSED_MODEL_KEYS = frozenset(
    {"decoder_net_config", "eos", "num_doecoder_block", "sos"}
)
_OFFICIAL_UNUSED_KW_KEYS = frozenset({"num_bpe_token"})

_SELF_ATT_NAME = "MultiHeadAtt"
_CROSS_ATT_NAME = "MultiHeadCrossAtt"


class KCEWeightsError(RuntimeError):
    """KCE backbone 权重缺失或与当前结构不匹配。"""


def _reject_unknown(value: Mapping[str, object], allowed: frozenset[str], path: str) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"{path} contains unknown keys: {unknown}")


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return value


def _exact_int(value: object, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{path} must be an exact int >= {minimum}, got {value!r}")
    return value


def _finite_number(value: object, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{path} must be a finite number, got {value!r}")
    resolved = float(value)
    if not math.isfinite(resolved) or (positive and resolved <= 0):
        raise ValueError(f"{path} must be finite{'> 0' if positive else ''}, got {value!r}")
    return resolved


def _dim_sequence(value: object, path: str, *, lengths: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) not in lengths:
        raise ValueError(f"{path} must be a sequence of {lengths} ints, got {value!r}")
    return tuple(_exact_int(item, f"{path}[{index}]", minimum=1) for index, item in enumerate(value))


def _feed_forward_dim(value: object, path: str) -> None:
    if type(value) is int:
        _exact_int(value, path, minimum=1)
        return
    _dim_sequence(value, path, lengths=(2, 3))


def _plain_config(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain_config(item) for item in value)
    return deepcopy(value)


def _normalize_kce_model_config(value: object) -> dict:
    """将官方 DAE 训练配置规范化为 KCE 推理子集的参数化。"""
    config = _plain_config(_mapping(value, "model_config"))
    for key in _OFFICIAL_UNUSED_MODEL_KEYS:
        config.pop(key, None)

    audio = config.get("audio_net_config")
    if isinstance(audio, Mapping):
        audio = dict(audio)
        audio.setdefault("num_mel_bins", 80)
        config["audio_net_config"] = audio
        transformer = audio.get("transformer_config")
        if isinstance(transformer, Mapping):
            transformer = dict(transformer)
            conv = transformer.get("conv_config")
            if isinstance(conv, Mapping):
                conv = dict(conv)
                conv.setdefault("kernel_size", 15)
                transformer["conv_config"] = conv
            config["audio_net_config"]["transformer_config"] = transformer

    keyword = config.get("kw_net_config")
    if isinstance(keyword, Mapping):
        keyword = dict(keyword)
        for key in _OFFICIAL_UNUSED_KW_KEYS:
            keyword.pop(key, None)
        config["kw_net_config"] = keyword

    # DAE 的 FeedForward 实际忽略 norm 配置；WeSep 官方 YAML 写 LayerNorm，
    # 但它不在该前馈层中创建归一化参数。映射时仅移除这个已知无效值。
    feed_forward_paths = (
        ("audio_net_config", "input_trans"),
        ("audio_net_config", "transformer_config", "feed_forward_config"),
        ("kw_net_config", "input_trans"),
        ("kw_net_config", "transformer_config", "feed_forward_config"),
    )
    for path in feed_forward_paths:
        parent = config
        for key in path[:-1]:
            parent = parent.get(key)
            if not isinstance(parent, Mapping):
                break
        else:
            node = parent.get(path[-1])
            if isinstance(node, Mapping) and node.get("norm") == "LayerNorm":
                normalized = dict(node)
                normalized.pop("norm")
                parent[path[-1]] = normalized
    return config


def _normalize_kce_fbank_config(value: object, *, sample_rate: int) -> dict:
    """补齐官方 ``feats_config`` 依赖 torchaudio 默认值的采样率字段。"""
    config = dict(_mapping(value, "kce fbank"))
    config.setdefault("sample_frequency", float(sample_rate))
    return config


class DAEPhonemeTokenizer:
    """把英文文本转成 DAE KCE 使用的 phoneme id 序列（离线 cue 准备用）。

    ``g2p_en`` 与 ``nltk`` 是可选依赖，只在 lexicon 未覆盖某个词时才构造 G2p，
    因此纯 lexicon 路径不需要安装 ``g2p_en``。
    """

    def __init__(self, phoneme_map, lexicon=None):
        if type(phoneme_map) is not str or not phoneme_map:
            raise ValueError("phoneme_map must be a non-empty path")
        if lexicon is not None and (type(lexicon) is not str or not lexicon):
            raise ValueError("lexicon must be a non-empty path or None")
        self.g2p = None
        self.phoneme_to_id = {}
        with open(phoneme_map, encoding="utf-8") as stream:
            for line in stream:
                fields = line.strip().split()
                if len(fields) == 2:
                    self.phoneme_to_id[fields[0]] = int(fields[1])

        self.lexicon = {}
        if lexicon and os.path.isfile(lexicon):
            with open(lexicon, encoding="utf-8") as stream:
                for line in stream:
                    fields = line.strip().split(maxsplit=1)
                    if len(fields) == 2:
                        self.lexicon[fields[0]] = [
                            int(value) for value in fields[1].split()
                        ]

    def _g2p(self):
        if self.g2p is None:
            try:
                import g2p_en
            except ImportError as error:
                raise ImportError(
                    "DAE text preparation requires g2p_en for out-of-lexicon words. "
                    "Install it with: pip install g2p_en"
                ) from error
            self.g2p = g2p_en.G2p()
        return self.g2p

    @staticmethod
    def normalize(text):
        """Normalize English text while preserving word apostrophes."""
        text = (text or "").strip().lower()
        for apostrophe in ("\u2019", "\u2018", "\u0060", "\u00b4"):
            text = text.replace(apostrophe, "'")
        cleaned = []
        for char in text:
            if char == "'":
                cleaned.append(char)
            elif char in string.punctuation or unicodedata.category(
                    char).startswith("P"):
                cleaned.append(" ")
            else:
                cleaned.append(char)
        return " ".join("".join(cleaned).split())

    def encode_words(self, text):
        """Return normalized words and one phoneme-ID sequence per word."""
        normalized = self.normalize(text)
        words = normalized.split()
        if not words:
            raise ValueError("Text is empty after DAE normalization")
        labels = []
        for word in words:
            ids = self.lexicon.get(word)
            if ids is None:
                phones = self._g2p()(word)
                ids = [
                    self.phoneme_to_id[phone] for phone in phones
                    if phone in self.phoneme_to_id
                ]
                self.lexicon[word] = ids
            if not ids:
                raise ValueError(f"No DAE phoneme IDs generated for: {word}")
            labels.append(ids)
        return normalized, words, labels

    def encode(self, text):
        """Return one flattened phoneme-ID sequence for direct inference."""
        _, _, labels = self.encode_words(text)
        return [value for word in labels for value in word]


class WordEmbedding(nn.Module):

    def __init__(self, num_tokens, dim, padding_idx=0):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim, padding_idx=padding_idx)

    def forward(self, x):
        return self.emb(x)


class PositionalEncoding(nn.Module):

    def __init__(self, model_dim, max_len=5000):
        super().__init__()
        self.d_model = model_dim
        self.xcale = math.sqrt(model_dim)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32) *
            -(math.log(10000.0) / model_dim))
        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x * self.xcale + self.pe[:, :x.shape[1]]


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, n_feats):
        super().__init__()
        if n_feats % n_head:
            raise ValueError("Attention dimension must be divisible by heads")
        self.d_k = n_feats // n_head
        self.n_head = n_head
        self.q = nn.Linear(n_feats, n_feats)
        self.k = nn.Linear(n_feats, n_feats)
        self.v = nn.Linear(n_feats, n_feats)
        self.linear_out = nn.Linear(n_feats, n_feats)

    def forward(self, query, key, value, mask=None):
        batch = query.shape[0]
        query = self.q(query).view(batch, -1, self.n_head,
                                   self.d_k).transpose(1, 2)
        key = self.k(key).view(batch, -1, self.n_head,
                               self.d_k).transpose(1, 2)
        value = self.v(value).view(batch, -1, self.n_head,
                                   self.d_k).transpose(1, 2)
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.d_k)
        if mask is not None:
            invalid = mask.unsqueeze(1).eq(0) if mask.ndim < 4 else mask.eq(0)
            score = score.masked_fill(invalid, -torch.inf)
        weight = torch.softmax(score, dim=-1)
        if mask is not None:
            weight = weight.masked_fill(invalid, 0.0)
        context = torch.matmul(weight, value).transpose(1, 2).contiguous()
        context = context.view(batch, query.shape[2], -1)
        return self.linear_out(context), weight


class MultiHeadCrossAttention(nn.Module):

    def __init__(self, n_head, n_feats, norm=None):
        super().__init__()
        normalizer = nn.LayerNorm if norm == "LayerNorm" else nn.Identity
        self.q = nn.Sequential(normalizer(n_feats),
                               nn.Linear(n_feats, n_feats))
        self.k = nn.Sequential(normalizer(n_feats),
                               nn.Linear(n_feats, n_feats))
        self.v = nn.Sequential(normalizer(n_feats),
                               nn.Linear(n_feats, n_feats))
        self.linear_out = nn.Linear(n_feats, n_feats)
        self.n_head = n_head
        self.n_feats = n_feats

    def forward(self, query, key, value, mask=None, aux_score=None):
        query, key, value = self.q(query), self.k(key), self.v(value)
        batch, query_len = query.shape[:2]
        query = query.view(batch, query_len, self.n_head, -1).transpose(1, 2)
        key = key.view(batch, key.shape[1], self.n_head, -1).transpose(1, 2)
        value = value.view(batch, value.shape[1], self.n_head,
                           -1).transpose(1, 2)
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            query.shape[-1])
        if aux_score is not None:
            score = score * aux_score[:, None, None, :]
        if mask is not None:
            invalid = mask.unsqueeze(1).eq(0) if mask.ndim < 4 else mask.eq(0)
            score = score.masked_fill(invalid, -torch.inf)
        weight = torch.softmax(score, dim=-1)
        if mask is not None:
            weight = weight.masked_fill(invalid, 0.0)
        context = torch.matmul(weight, value).transpose(1, 2).contiguous()
        context = context.view(batch, query_len, -1)
        return self.linear_out(context), weight


class FeedForward(nn.Module):
    """官方 KCE 的 FFN 参数化：只支持带 bias 的 1/2/3 段 Linear + ReLU。"""

    def __init__(self, dim, num_block=1, bias=True, norm=None, act="ReLU"):
        super().__init__()
        if type(num_block) is not int or num_block != 1:
            raise ValueError("FeedForward only supports num_block=1")
        if bias is not True:
            raise ValueError("FeedForward only supports bias=True")
        if norm is not None:
            raise ValueError("FeedForward only supports norm=None")
        if act != "ReLU":
            raise ValueError("FeedForward only supports act='ReLU'")
        if isinstance(dim, (list, tuple)):
            if len(dim) not in (2, 3):
                raise ValueError(f"FeedForward dim must have 2 or 3 values, got {dim!r}")
            input_dim = dim[0]
            hidden_dim = dim[1] if len(dim) == 3 else dim[0]
            output_dim = dim[-1]
        else:
            input_dim = hidden_dim = output_dim = dim
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.w2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


class DepthWiseConv(nn.Module):
    """GLU + depthwise conv 残差块；官方配置只用 BatchNorm1d 的 non-causal 形态。"""

    def __init__(self,
                 channels,
                 kernel_size=15,
                 activation=None,
                 norm="batch_norm",
                 causal=False,
                 bias=True):
        super().__init__()
        if activation is not None:
            raise ValueError("DepthWiseConv only supports activation=None (ReLU)")
        if norm not in {"batch_norm", "layer_norm"}:
            raise ValueError(f"unsupported DepthWiseConv norm: {norm!r}")
        if type(causal) is not bool:
            raise TypeError("DepthWiseConv causal must be a bool")
        if type(bias) is not bool:
            raise TypeError("DepthWiseConv bias must be a bool")
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, 1, bias=bias)
        padding = 0 if causal else (kernel_size - 1) // 2
        self.lorder = kernel_size - 1 if causal else 0
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=bias,
        )
        self.use_layer_norm = norm == "layer_norm"
        self.norm = (nn.LayerNorm(channels)
                     if self.use_layer_norm else nn.BatchNorm1d(channels))
        self.pointwise_conv2 = nn.Conv1d(channels, channels, 1, bias=bias)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = F.glu(self.pointwise_conv1(x.transpose(1, 2)), dim=1)
        if self.lorder:
            x = F.pad(x, (self.lorder, 0))
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = self.activation(self.norm(x.transpose(1, 2))).transpose(1, 2)
        else:
            x = self.activation(self.norm(x))
        return self.pointwise_conv2(x).transpose(1, 2)


class TransformerLayer(nn.Module):

    def __init__(self,
                 self_att,
                 feed_forward,
                 macaron_layer=None,
                 conv_layer=None,
                 cross_att=None,
                 size=256):
        super().__init__()
        self.self_att = self_att
        self.feed_forward = feed_forward
        self.macaron_layer = macaron_layer
        self.cross_att = cross_att
        self.input_norm = nn.LayerNorm(size, eps=1e-5)
        self.fnn_norm = nn.LayerNorm(size, eps=1e-5)
        self.size = size
        self.conv_layer = conv_layer
        if conv_layer is not None:
            self.conv_norm = nn.LayerNorm(size, eps=1e-5)
        if macaron_layer is not None:
            self.macaron_norm = nn.LayerNorm(size, eps=1e-5)
            self.macaron_factor = 0.5

    def forward(self, x, mask, cross_input=None):
        if self.macaron_layer is not None:
            x = x + self.macaron_factor * self.macaron_layer(
                self.macaron_norm(x))
        residual = x
        context, attention = self.self_att(self.input_norm(x),
                                           self.input_norm(x),
                                           self.input_norm(x), mask)
        x = residual + context
        if self.cross_att is not None:
            key, value, cross_mask = cross_input
            context, attention = self.cross_att(x, key, value, cross_mask)
            x = x + context
        if self.conv_layer is not None:
            x = x + self.conv_layer(self.conv_norm(x))
        return x + self.feed_forward(self.fnn_norm(x)), attention


class CTC(nn.Module):
    """保留上游 CTC 头以对齐官方 checkpoint 键；推理子集不调用。"""

    def __init__(self, num_tokens, front_output_size, reduce=True):
        super().__init__()
        self.linear_project = nn.Linear(front_output_size, num_tokens)
        self.ctc_loss = nn.CTCLoss(reduction="sum" if reduce else "none")


class SpeakerPooling(nn.Module):
    """保留上游句级 pooling 头与 CE criterion 以对齐官方 checkpoint 键。"""

    def __init__(self,
                 num_classes,
                 front_output_size,
                 pooling_conf,
                 use_reg_loss=True,
                 reg_loss_weight=0.01,
                 reduce=True):
        super().__init__()
        self.linear_project = nn.Linear(front_output_size, num_classes)
        self.ce_loss = nn.CrossEntropyLoss(
            reduction="sum" if reduce else "none", ignore_index=-100)
        self.pooling_type = pooling_conf["type"]
        self.layer_indices = torch.tensor(pooling_conf.get(
            "layer_indices", []),
                                          dtype=torch.long)
        self.use_softmax = pooling_conf["use_softmax"]
        self.use_reg_loss = use_reg_loss
        self.reg_loss_weight = reg_loss_weight
        if self.pooling_type == "learnable_weights":
            count = len(self.layer_indices)
            self.weight = nn.Parameter(torch.full((count, ), 1.0 / count))

    def forward_pooling(self, hidden_states, lengths):
        states = torch.stack(hidden_states)[self.layer_indices.to(
            hidden_states[0].device)]
        if self.pooling_type == "mean":
            states = states.mean(dim=0)
        else:
            weight = F.softmax(self.weight, dim=-1) if self.use_softmax \
                else self.weight
            states = sum(w * state for w, state in zip(weight, states))
        return torch.stack(
            [state[:length].mean(0) for state, length in zip(states, lengths)])


def make_mask(lengths, max_len=None):
    max_len = int(max_len or lengths.max().item())
    positions = torch.arange(max_len, device=lengths.device)
    return positions.unsqueeze(0) >= lengths.unsqueeze(1)


def _conv_output_length(length):
    return (length - 3) // 2 + 1


def _minimum_fbank_frames():
    """au_conv 两级 3x3 stride-2 卷积后仍至少剩一帧所需的最小 fbank 帧数。"""
    frames = 1
    while True:
        reduced = frames
        for _ in range(2):
            reduced = _conv_output_length(reduced)
        if reduced >= 1:
            return frames
        frames += 1


def combine_mask(mask1, mask2):
    return (~(mask1.unsqueeze(2) & mask2.unsqueeze(1))).unsqueeze(1)


def _validate_self_att_config(value: object, path: str) -> Mapping[str, object]:
    config = _mapping(value, path)
    _reject_unknown(config, _SELF_ATT_CONFIG_KEYS, path)
    for key in _SELF_ATT_CONFIG_KEYS:
        _exact_int(config.get(key), f"{path}.{key}", minimum=1)
    if config["n_feats"] % config["n_head"]:
        raise ValueError(f"{path}.n_feats must be divisible by n_head")
    return config


def _validate_cross_att_config(value: object, path: str) -> Mapping[str, object]:
    config = _mapping(value, path)
    _reject_unknown(config, _CROSS_ATT_CONFIG_KEYS, path)
    _validate_self_att_config({key: config[key] for key in _SELF_ATT_CONFIG_KEYS}, path)
    if config.get("norm") not in {None, "LayerNorm"}:
        raise ValueError(f"{path}.norm must be 'LayerNorm' or None")
    return config


def _validate_feed_forward_config(value: object, path: str) -> Mapping[str, object]:
    config = _mapping(value, path)
    _reject_unknown(config, _FEED_FORWARD_KEYS, path)
    _feed_forward_dim(config.get("dim"), f"{path}.dim")
    return config


def _validate_input_trans(value: object, path: str) -> Mapping[str, object]:
    config = _mapping(value, path)
    _reject_unknown(config, _INPUT_TRANS_KEYS, path)
    _feed_forward_dim(config.get("dim"), f"{path}.dim")
    return config


def _validate_conv_config(value: object, path: str) -> Mapping[str, object]:
    config = _mapping(value, path)
    _reject_unknown(config, _CONV_CONFIG_KEYS, path)
    _exact_int(config.get("channels"), f"{path}.channels", minimum=1)
    _exact_int(config.get("kernel_size"), f"{path}.kernel_size", minimum=1)
    if "activation" in config and config["activation"] is not None:
        raise ValueError(f"{path}.activation must be None (ReLU is fixed)")
    return config


def _validate_transformer_config(
    value: object, path: str, *, with_cross: bool
) -> Mapping[str, object]:
    config = _mapping(value, path)
    allowed = _AUDIO_TRANSFORMER_KEYS if with_cross else _KW_TRANSFORMER_KEYS
    _reject_unknown(config, allowed, path)
    _exact_int(config.get("size"), f"{path}.size", minimum=1)
    if config.get("self_att") != _SELF_ATT_NAME:
        raise ValueError(
            f"{path}.self_att must be {_SELF_ATT_NAME!r}, got {config.get('self_att')!r}"
        )
    _validate_self_att_config(config.get("self_att_config"), f"{path}.self_att_config")
    _validate_feed_forward_config(
        config.get("feed_forward_config"), f"{path}.feed_forward_config"
    )
    if with_cross:
        if config.get("cross_att") != _CROSS_ATT_NAME:
            raise ValueError(
                f"{path}.cross_att must be {_CROSS_ATT_NAME!r}, "
                f"got {config.get('cross_att')!r}"
            )
        _validate_cross_att_config(config.get("corss_att_config"), f"{path}.corss_att_config")
        _validate_conv_config(config.get("conv_config"), f"{path}.conv_config")
    return config


def _validate_sv_net_config(value: object, path: str) -> Mapping[str, object]:
    config = _mapping(value, path)
    _reject_unknown(config, _SV_NET_KEYS, path)
    _exact_int(config.get("num_classes"), f"{path}.num_classes", minimum=1)
    _exact_int(config.get("front_output_size"), f"{path}.front_output_size", minimum=1)
    pooling = _mapping(config.get("pooling_conf"), f"{path}.pooling_conf")
    _reject_unknown(pooling, _POOLING_CONF_KEYS, path=f"{path}.pooling_conf")
    for key in ("type", "use_softmax"):
        if key not in pooling:
            raise ValueError(f"{path}.pooling_conf requires {key}")
    if pooling["type"] not in {"learnable_weights", "mean"}:
        raise ValueError(f"{path}.pooling_conf.type must be 'learnable_weights' or 'mean'")
    if type(pooling["use_softmax"]) is not bool:
        raise TypeError(f"{path}.pooling_conf.use_softmax must be a bool")
    indices = pooling.get("layer_indices")
    if not isinstance(indices, (list, tuple)) or not indices:
        raise ValueError(f"{path}.pooling_conf.layer_indices must be a non-empty sequence")
    for index, item in enumerate(indices):
        _exact_int(item, f"{path}.pooling_conf.layer_indices[{index}]", minimum=0)
    return config


def validate_kce_model_config(value: object, path: str = "model_config") -> dict:
    """严格校验官方 KCE backbone 配置，未知键直接报错。"""
    config = _mapping(value, path)
    _reject_unknown(config, _MODEL_CONFIG_KEYS, path)
    audio = _mapping(config.get("audio_net_config"), f"{path}.audio_net_config")
    _reject_unknown(audio, _AUDIO_NET_KEYS, f"{path}.audio_net_config")
    _exact_int(audio.get("num_mel_bins"), f"{path}.audio_net_config.num_mel_bins", minimum=1)
    _validate_input_trans(audio.get("input_trans"), f"{path}.audio_net_config.input_trans")
    _validate_transformer_config(
        audio.get("transformer_config"), f"{path}.audio_net_config.transformer_config",
        with_cross=True,
    )
    keyword = _mapping(config.get("kw_net_config"), f"{path}.kw_net_config")
    _reject_unknown(keyword, _KW_NET_KEYS, f"{path}.kw_net_config")
    _exact_int(keyword.get("num_phn_token"), f"{path}.kw_net_config.num_phn_token", minimum=2)
    _validate_input_trans(keyword.get("input_trans"), f"{path}.kw_net_config.input_trans")
    _validate_transformer_config(
        keyword.get("transformer_config"), f"{path}.kw_net_config.transformer_config",
        with_cross=False,
    )
    _exact_int(config.get("num_audio_block"), f"{path}.num_audio_block", minimum=1)
    _exact_int(config.get("num_kw_block"), f"{path}.num_kw_block", minimum=1)
    if config.get("loss_weight") is not None:
        loss_weight = config["loss_weight"]
        if isinstance(loss_weight, (list, tuple)):
            if not loss_weight:
                raise ValueError(f"{path}.loss_weight must not be empty")
            for index, value in enumerate(loss_weight):
                _finite_number(value, f"{path}.loss_weight[{index}]")
        else:
            _finite_number(loss_weight, f"{path}.loss_weight")
    if config.get("sv_net_config") is not None:
        _validate_sv_net_config(config["sv_net_config"], f"{path}.sv_net_config")
    return dict(config)


class KCEBackbone(nn.Module):
    """AEDKWSASRPhone 的 checkpoint 兼容推理子集。

    ``loss_weight`` 与 ``CTC``/``SpeakerPooling`` 的 criterion 保留是为了对齐官方
    checkpoint 的键名与训练路径，推理子集不调用它们。整个 backbone 始终 ``eval``
    （含 BatchNorm running statistics）。
    """

    def __init__(self,
                 *,
                 audio_net_config,
                 kw_net_config,
                 num_audio_block=8,
                 num_kw_block=3,
                 loss_weight=None,
                 sv_net_config=None):
        super().__init__()
        audio_transformer = audio_net_config["transformer_config"]
        keyword_transformer = kw_net_config["transformer_config"]
        audio_dim = audio_transformer["size"]
        keyword_dim = keyword_transformer["size"]

        self.au_conv = nn.Sequential(
            nn.Conv2d(1, audio_dim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(audio_dim, audio_dim, 3, 2),
            nn.ReLU(),
        )
        self.num_mel_bins = audio_net_config["num_mel_bins"]
        reduced_mels = ((self.num_mel_bins - 1) // 2 - 1) // 2
        self.au_conv_trans = nn.Linear(audio_dim * reduced_mels, audio_dim)
        self.speech_input_projection = FeedForward(
            **audio_net_config["input_trans"])
        self.speech_pe_module = PositionalEncoding(audio_dim)
        self.speech_transformer = nn.ModuleList([
            TransformerLayer(
                size=audio_dim,
                self_att=MultiHeadAttention(
                    **audio_transformer["self_att_config"]),
                cross_att=MultiHeadCrossAttention(
                    **audio_transformer["corss_att_config"]),
                feed_forward=FeedForward(
                    **audio_transformer["feed_forward_config"]),
                macaron_layer=FeedForward(
                    **audio_transformer["feed_forward_config"]),
                conv_layer=DepthWiseConv(**audio_transformer["conv_config"]),
            ) for _ in range(num_audio_block)
        ])

        self.phn_emb = WordEmbedding(kw_net_config["num_phn_token"],
                                     keyword_dim)
        self.keyword_pe_module = PositionalEncoding(keyword_dim)
        self.keyword_input_projection = FeedForward(
            **kw_net_config["input_trans"])
        self.keyword_transformer = nn.ModuleList([
            TransformerLayer(
                size=keyword_dim,
                self_att=MultiHeadAttention(
                    **keyword_transformer["self_att_config"]),
                feed_forward=FeedForward(
                    **keyword_transformer["feed_forward_config"]),
            ) for _ in range(num_kw_block)
        ])
        self.kw_au_link = (nn.Linear(keyword_dim, audio_dim)
                           if keyword_dim != audio_dim else nn.Identity())
        self.loss_weight = loss_weight
        self.asr_phn_criterion = CTC(kw_net_config["num_phn_token"], audio_dim)
        self.use_sv = sv_net_config is not None
        if self.use_sv:
            self.sv_ce_crit = SpeakerPooling(**sv_net_config)

    @staticmethod
    def forward_transformer(layers,
                            hidden,
                            mask=None,
                            cross=None,
                            collect=False):
        states = []
        for layer in layers:
            hidden, _ = layer(hidden, mask, cross_input=cross)
            if collect:
                states.append(hidden)
        return (hidden, states) if collect else hidden

    def forward(self, speech, speech_lengths, keyword, keyword_lengths):
        speech_lengths = torch.div(
            speech_lengths - 3, 2, rounding_mode="floor") + 1
        speech_lengths = torch.div(
            speech_lengths - 3, 2, rounding_mode="floor") + 1
        speech_mask = ~make_mask(speech_lengths).unsqueeze(1)
        keyword_mask = ~make_mask(keyword_lengths).unsqueeze(1)
        cross_mask = ~combine_mask(speech_mask.squeeze(1),
                                   keyword_mask.squeeze(1))

        speech = self.au_conv(speech.unsqueeze(1))
        batch, channels, frames, bins = speech.shape
        speech = speech.transpose(1,
                                  2).contiguous().view(batch, frames,
                                                       channels * bins)
        speech = self.speech_pe_module(
            self.speech_input_projection(self.au_conv_trans(speech)))
        keyword = self.keyword_pe_module(
            self.keyword_input_projection(self.phn_emb(keyword.long())))
        keyword = self.forward_transformer(self.keyword_transformer, keyword,
                                           keyword_mask)
        keyword = self.kw_au_link(keyword)
        speech, states = self.forward_transformer(
            self.speech_transformer,
            speech,
            speech_mask,
            cross=(keyword, keyword, cross_mask),
            collect=True,
        )
        if self.use_sv:
            return self.sv_ce_crit.forward_pooling(states, speech_lengths)
        return torch.stack([
            value[:length].mean(0)
            for value, length in zip(speech, speech_lengths)
        ])


class KCE(nn.Module):
    """把 mixture 波形与 phoneme id 编码成适配后的 DAE 文本 cue。

    Args:
        config: 只接受 ``checkpoint`` / ``model_config`` / ``fbank`` / ``eps`` /
            ``adapter`` / ``padding_id`` / ``bos_id`` / ``eos_id``。
        sample_rate: mixture 采样率，必须为 16000。
        backbone_state_dict: 显式注入的 backbone state dict（与 ``checkpoint`` 二选一）。
        defer_pretrained: True 时不在构造期取权重，改由外层 ``load_state_dict``
            的 pre-hook 填充；在此之前 ``forward`` 报错。注意
            ``evaluate/checkpoint_loading.load_state_dict_checked`` 会在调用
            ``load_state_dict`` 前按键集预校验，因此该路径下 backbone 键必须已包含在
            外层 checkpoint 中。
        allow_unsafe_checkpoint: 官方 DAE-TSE 训练 checkpoint 需要 pickle 时显式
            放开；默认保持 ``weights_only=True``。
    """

    def __init__(
        self,
        config=None,
        *,
        sample_rate=KCE_SAMPLE_RATE,
        backbone_state_dict=None,
        defer_pretrained=False,
        allow_unsafe_checkpoint=False,
    ):
        super().__init__()
        config = {} if config is None else dict(_mapping(config, "kce config"))
        _reject_unknown(config, _KCE_CONFIG_KEYS, "kce config")
        if type(defer_pretrained) is not bool:
            raise TypeError("defer_pretrained must be a bool")
        if type(allow_unsafe_checkpoint) is not bool:
            raise TypeError("allow_unsafe_checkpoint must be a bool")
        if type(sample_rate) is not int or sample_rate not in _SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"KCE requires sample_rate={KCE_SAMPLE_RATE}, got {sample_rate!r}"
            )

        self.sample_rate = sample_rate
        self.checkpoint = config.get("checkpoint")
        if self.checkpoint is not None and (
            type(self.checkpoint) is not str or not self.checkpoint
        ):
            raise ValueError("kce checkpoint must be a non-empty path or None")
        self.allow_unsafe_checkpoint = allow_unsafe_checkpoint
        self.pretrained_loaded = False
        self.fallback_reported = False
        eps = _finite_number(
            config.get("eps", torch.finfo(torch.float32).eps), "kce eps", positive=True
        )

        model_config = config.get("model_config")
        fbank_config = config.get("fbank")
        # 官方 checkpoint 目录旁可能带 model.yaml / data.yaml sidecar。
        if self.checkpoint:
            directory = os.path.dirname(os.path.expanduser(self.checkpoint))
            if model_config is None:
                model_path = os.path.join(directory, "model.yaml")
                if os.path.isfile(model_path):
                    with open(model_path, encoding="utf-8") as stream:
                        model_config = yaml.safe_load(stream)
            if fbank_config is None:
                data_path = os.path.join(directory, "data.yaml")
                if os.path.isfile(data_path):
                    with open(data_path, encoding="utf-8") as stream:
                        data_config = yaml.safe_load(stream)
                    speech_config = _mapping(
                        _mapping(data_config, "data.yaml").get("speech_config"),
                        "data.yaml.speech_config",
                    )
                    if "feats_config" not in speech_config:
                        raise ValueError(
                            f"{data_path} speech_config must provide feats_config for KCE fbank"
                        )
                    fbank_config = speech_config["feats_config"]

        model_config = validate_kce_model_config(
            deepcopy(DEFAULT_KCE_MODEL_CONFIG)
            if model_config is None
            else _normalize_kce_model_config(model_config)
        )
        self.model_config = model_config
        self.fbank_config = self._validate_fbank_config(
            deepcopy(DEFAULT_KCE_FBANK_CONFIG)
            if fbank_config is None
            else _normalize_kce_fbank_config(fbank_config, sample_rate=sample_rate),
            num_mel_bins=model_config["audio_net_config"]["num_mel_bins"],
        )
        backbone_dim = model_config["audio_net_config"]["transformer_config"]["size"]
        self.backbone_dim = backbone_dim
        self.num_phn_token = model_config["kw_net_config"]["num_phn_token"]
        self.model = KCEBackbone(**model_config)
        self.model.requires_grad_(False)

        adapter = dict(_mapping(config.get("adapter", {}), "kce adapter"))
        _reject_unknown(adapter, _ADAPTER_KEYS, "kce adapter")
        input_dim = _exact_int(
            adapter.get("input_dim", backbone_dim), "kce adapter.input_dim", minimum=1
        )
        if input_dim != backbone_dim:
            raise ValueError(
                f"kce adapter.input_dim={input_dim} conflicts with backbone dim {backbone_dim}"
            )
        self.output_dim = _exact_int(
            adapter.get("output_dim", 128), "kce adapter.output_dim", minimum=1
        )
        num_layers = _exact_int(
            adapter.get("num_layers", 1), "kce adapter.num_layers", minimum=1
        )
        hidden_layers = [
            layer
            for _ in range(num_layers - 1)
            for layer in (nn.Linear(input_dim, input_dim), nn.LayerNorm(input_dim, eps=eps))
        ]
        self.adapter = nn.Sequential(
            *hidden_layers,
            nn.Linear(input_dim, self.output_dim),
            nn.LayerNorm(self.output_dim, eps=eps),
        )

        self.padding_id = _exact_int(
            config.get("padding_id", 0), "kce padding_id", minimum=0
        )
        self.bos_id = _exact_int(config.get("bos_id", 71), "kce bos_id", minimum=0)
        self.eos_id = _exact_int(config.get("eos_id", 72), "kce eos_id", minimum=0)
        if self.bos_id == self.eos_id:
            raise ValueError("kce bos_id and eos_id must differ")
        for name, value in (("bos_id", self.bos_id), ("eos_id", self.eos_id)):
            if value >= self.num_phn_token:
                raise ValueError(
                    f"kce {name}={value} must be smaller than num_phn_token={self.num_phn_token}"
                )

        if backbone_state_dict is not None:
            if self.checkpoint:
                raise ValueError(
                    "pass either a kce checkpoint path or an injected backbone state dict"
                )
            if defer_pretrained:
                raise ValueError(
                    "defer_pretrained=True conflicts with an injected backbone state dict"
                )
            self.load_backbone_state_dict(backbone_state_dict)
        elif not defer_pretrained:
            self.load_pretrained()
        # torch>=2.4 的 pre-hook 签名带 module 参数。
        self._register_load_state_dict_pre_hook(self._prepare_pretrained_state, with_module=True)
        self.model.eval()

    @property
    def weights_loaded(self) -> bool:
        """backbone 权重是否已确定（构造期加载或由外层 load_state_dict 注入）。"""
        return self.pretrained_loaded

    def _validate_fbank_config(
        self, value: object, *, num_mel_bins: int
    ) -> dict:
        config = dict(_mapping(value, "kce fbank"))
        parameters = set(inspect.signature(kaldi.fbank).parameters)
        unknown = sorted(set(config) - parameters)
        if unknown:
            raise ValueError(f"kce fbank contains unknown Kaldi fbank arguments: {unknown}")
        for key in ("num_mel_bins", "frame_length", "frame_shift", "sample_frequency"):
            if key not in config:
                raise ValueError(f"kce fbank requires {key}")
        config["num_mel_bins"] = _exact_int(
            config["num_mel_bins"], "kce fbank.num_mel_bins", minimum=1
        )
        if config["num_mel_bins"] != num_mel_bins:
            raise ValueError(
                f"kce fbank.num_mel_bins={config['num_mel_bins']} conflicts with "
                f"backbone num_mel_bins={num_mel_bins}"
            )
        sample_frequency = _finite_number(
            config["sample_frequency"], "kce fbank.sample_frequency", positive=True
        )
        if int(sample_frequency) != self.sample_rate:
            raise ValueError(
                f"kce fbank.sample_frequency={sample_frequency} conflicts with "
                f"sample_rate={self.sample_rate}"
            )
        config["sample_frequency"] = sample_frequency
        config["frame_length"] = _finite_number(
            config["frame_length"], "kce fbank.frame_length", positive=True
        )
        config["frame_shift"] = _finite_number(
            config["frame_shift"], "kce fbank.frame_shift", positive=True
        )
        if "dither" in config:
            config["dither"] = _finite_number(config["dither"], "kce fbank.dither")
        self.window_size = int(self.sample_rate * config["frame_length"] * 0.001)
        self.window_shift = int(self.sample_rate * config["frame_shift"] * 0.001)
        if self.window_size < 3 or self.window_shift < 1:
            raise ValueError("kce fbank frame geometry is too short for the KCE conv stack")
        self.minimum_fbank_frames = _minimum_fbank_frames()
        self.minimum_samples = self.window_size + (
            self.minimum_fbank_frames - 1
        ) * self.window_shift
        return config

    def load_pretrained(self):
        """加载外部 KCE backbone checkpoint（最多一次）。"""
        if self.pretrained_loaded:
            return
        if not self.checkpoint:
            raise KCEWeightsError(
                "KCE backbone weights are absent: pass an explicit kce checkpoint path "
                "or inject a backbone state dict; random initialization is not allowed."
            )
        payload = safe_torch_load(
            self.checkpoint,
            map_location="cpu",
            allow_unsafe=self.allow_unsafe_checkpoint,
            purpose=_KCE_CHECKPOINT_PURPOSE,
        )
        state = payload
        if isinstance(payload, Mapping) and _CHECKPOINT_KEY in payload:
            state = payload[_CHECKPOINT_KEY]
        self.load_backbone_state_dict(state)

    def load_backbone_state_dict(self, state):
        """严格校验 key/shape 后加载 backbone 权重。"""
        if not isinstance(state, Mapping):
            raise KCEWeightsError(
                f"KCE backbone state dict must be a mapping, got {type(state).__name__}"
            )
        reference = self.model.state_dict()
        missing = sorted(set(reference) - set(state))
        unexpected = sorted(set(state) - set(reference))
        mismatched = []
        for key in sorted(set(reference) & set(state)):
            value = state[key]
            if not isinstance(value, torch.Tensor):
                mismatched.append(f"{key}: expected tensor, got {type(value).__name__}")
                continue
            if tuple(value.shape) != tuple(reference[key].shape):
                mismatched.append(
                    f"{key}: {tuple(value.shape)} != {tuple(reference[key].shape)}"
                )
        if missing or unexpected or mismatched:
            raise KCEWeightsError(
                "KCE backbone weights do not match the configured architecture: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
                f"shape_or_type={mismatched[:8]}"
            )
        self.model.load_state_dict(dict(state), strict=True)
        self.pretrained_loaded = True

    def _prepare_pretrained_state(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """外层 checkpoint 未包含 backbone 时，用外部 checkpoint 补齐。"""
        model_prefix = prefix + "model."
        if any(key.startswith(model_prefix) for key in state_dict):
            self.pretrained_loaded = True
            return
        self.load_pretrained()
        state_dict.update({
            model_prefix + key: value
            for key, value in self.model.state_dict().items()
        })
        if not self.fallback_reported:
            logger.warning(
                "KCE is absent from the outer checkpoint; loaded %s.", self.checkpoint
            )
            self.fallback_reported = True

    def train(self, mode=True):
        """只训练 adapter；KCE backbone 始终 eval，BatchNorm 统计不更新。"""
        super().train(mode)
        self.model.eval()
        return self

    @staticmethod
    def _integer_lengths(value, reference, name):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must be int32/int64, got {value.dtype}")
        if value.ndim != 1 or value.shape != reference.shape:
            raise ValueError(f"{name} must have shape {tuple(reference.shape)}")
        if value.device != reference.device:
            raise ValueError(f"{name} must use the cue device")
        return value

    def _validate_inputs(self, mixture, phoneme_ids, present, textual_lengths,
                         mixture_lengths):
        if not isinstance(mixture, torch.Tensor) or mixture.ndim != 2:
            raise ValueError("mixture must be a [R, T] waveform tensor")
        if not mixture.is_floating_point():
            raise TypeError("mixture must be a floating-point waveform")
        if mixture.shape[1] < self.minimum_samples:
            raise ValueError(
                f"mixture must provide at least {self.minimum_samples} samples for the "
                f"KCE conv stack, got T={mixture.shape[1]}"
            )
        if not isinstance(phoneme_ids, torch.Tensor) or phoneme_ids.ndim != 2:
            raise ValueError("phoneme_ids must be a [R, L] tensor")
        if phoneme_ids.dtype != torch.int64:
            raise TypeError(f"phoneme_ids must be torch.int64, got {phoneme_ids.dtype}")
        if phoneme_ids.shape[1] < 1:
            raise ValueError("phoneme_ids must provide at least one token column")
        if not isinstance(present, torch.Tensor) or present.ndim != 1:
            raise ValueError("present must be a [R] tensor")
        if present.dtype != torch.bool:
            raise TypeError(f"present must be torch.bool, got {present.dtype}")
        rows = present.numel()
        if mixture.shape[0] != rows or phoneme_ids.shape[0] != rows:
            raise ValueError(
                "mixture, phoneme_ids and present must share the row axis, got "
                f"{mixture.shape[0]}, {phoneme_ids.shape[0]} and {rows}"
            )
        devices = {mixture.device, phoneme_ids.device, present.device}
        if len(devices) != 1:
            raise ValueError("mixture, phoneme_ids and present must share one device")
        lengths = self._integer_lengths(textual_lengths, present, "textual_lengths")
        mixture_lengths = self._integer_lengths(mixture_lengths, present, "mixture_lengths")
        if len({*devices, lengths.device, mixture_lengths.device}) != 1:
            raise ValueError("all KCE inputs must share one device")

        token_capacity = phoneme_ids.shape[1]
        invalid = (lengths < 0) | (lengths > token_capacity) | (
            present & (lengths < 1)
        ) | (~present & (lengths != 0))
        if bool(invalid.any().item()):
            raise ValueError(
                "textual_lengths must be in [1, L] for present rows and exactly 0 for "
                "missing rows"
            )
        mixture_invalid = (mixture_lengths < self.minimum_samples) | (
            mixture_lengths > mixture.shape[1]
        )
        if bool(mixture_invalid.any().item()):
            raise ValueError(
                f"mixture_lengths must be in [{self.minimum_samples}, {mixture.shape[1]}]"
            )
        if bool(((phoneme_ids < 0) | (phoneme_ids >= self.num_phn_token)).any().item()):
            raise ValueError(
                f"phoneme_ids must be in [0, {self.num_phn_token}), got out-of-range tokens"
            )
        for row, length in enumerate(lengths.tolist()):
            tail = phoneme_ids[row, length:]
            if bool(tail.ne(self.padding_id).any().item()):
                raise ValueError(
                    f"phoneme_ids[{row}] must be padded with padding_id={self.padding_id} "
                    f"after textual_lengths={length}; the canonical cue contract "
                    "zero-fills padding, so padding_id must be 0 for dataset batches"
                )
        return lengths, mixture_lengths

    def _build_keywords(self, phoneme_ids, lengths):
        boundary = phoneme_ids.new_tensor([self.bos_id, self.eos_id], dtype=torch.long)
        keywords = []
        for row, length in enumerate(lengths.tolist()):
            valid = phoneme_ids[row, :length]
            valid = valid[valid.ne(self.padding_id)]
            keywords.append(torch.cat([boundary[:1], valid, boundary[1:]]))
        keyword_lengths = torch.tensor(
            [value.numel() for value in keywords], device=phoneme_ids.device
        )
        return pad_sequence(
            keywords, batch_first=True, padding_value=self.padding_id
        ), keyword_lengths

    def _compute_fbanks(self, mixture, mixture_lengths):
        fbanks = []
        for row, length in enumerate(mixture_lengths.tolist()):
            features = kaldi.fbank(
                mixture[row, :length].float().unsqueeze(0), **self.fbank_config
            )
            if features.shape[0] < self.minimum_fbank_frames:
                raise ValueError(
                    f"mixture row {row} yields {features.shape[0]} fbank frames; the KCE "
                    f"conv stack needs at least {self.minimum_fbank_frames}"
                )
            fbanks.append(features)
        speech_lengths = torch.tensor(
            [value.shape[0] for value in fbanks], device=mixture.device
        )
        return pad_sequence(fbanks, batch_first=True), speech_lengths

    def forward(self, mixture, phoneme_ids, *, present, textual_lengths,
                mixture_lengths):
        """返回 ``[R, output_dim]`` 句级 cue；缺失行输出精确为 0。"""
        lengths, mixture_lengths = self._validate_inputs(
            mixture, phoneme_ids, present, textual_lengths, mixture_lengths
        )
        if not self.pretrained_loaded:
            raise KCEWeightsError(
                "KCE backbone weights were never loaded; the deferred external "
                "checkpoint was not applied and random weights must not be used."
            )
        if self.model.training:
            raise RuntimeError("KCE backbone must stay in eval mode (BatchNorm statistics)")
        keywords, keyword_lengths = self._build_keywords(phoneme_ids, lengths)
        fbanks, speech_lengths = self._compute_fbanks(mixture, mixture_lengths)
        with torch.no_grad():
            cue = self.model(fbanks, speech_lengths, keywords, keyword_lengths)
        cue = self.adapter(cue)
        return torch.where(present.reshape(-1, 1), cue, cue.new_zeros(()))


__all__ = [
    "DEFAULT_KCE_FBANK_CONFIG",
    "DEFAULT_KCE_MODEL_CONFIG",
    "DAEPhonemeTokenizer",
    "KCE",
    "KCEBackbone",
    "KCEWeightsError",
    "KCE_SAMPLE_RATE",
    "combine_mask",
    "make_mask",
    "validate_kce_model_config",
]
