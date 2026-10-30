# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
"""文本 cue 前端：KCE 计算 + 句级 cue 输出（WeSep dc11cb18 移植）。

来源：WeSep commit ``dc11cb18`` 的 ``wesep/modules/textual/textual_frontend.py``
（Apache-2.0，Copyright (c) 2026 Ke Zhang）。保留上游的 feature 配置形状
（``features.dae_kce``）、``compute``/``post`` 接口与每 stage 融合层，适配点：

- 配置解析沿用仓库内 ``deep_update``，未知键直接报错；
- ``compute`` 返回带 ``present`` 的 ``TextualFrontendOutput``，缺失行 cue 精确为 0，
  并在 ``post`` 中把缺失行还原成原 latent（等价 ``CueComposer.gate_fusion``）；
- ``mix_dim`` / ``num_stages`` 与上游 ``TSE_BSRNN_TEXTUAL`` 一样由 separator 决定，
  作为显式构造参数传入。

IO 合同
-------
输入：mixture 波形 ``[R, T]``（16 kHz）、``phoneme_ids`` ``[R, L]`` int64、
``present`` ``[R]`` bool、``textual_lengths`` ``[R]``、``mixture_lengths`` ``[R]``。
输出：``cue`` ``[R, output_dim]`` 句级（广播）特征与 ``present``。cue 可直接交给
``CueComposer.fuse_latent(..., embedding=cue)``（Composer 需以
``spkemb={"fusion": ...}`` 与 ``embedding_dim=output_dim`` 构造）做句级广播融合，
也可用 ``post`` 在每个 separator block 前逐 stage 融合。KCE 自带 Kaldi fbank 前端
（model-owned 特征提取例外，见 ``kce.py``），本模块不引入 separator STFT/iSTFT。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from torch import Tensor, nn

from tasks.tse.models.modules.common.deep_update import deep_update
from tasks.tse.models.modules.composition import gate_fusion
from tasks.tse.models.modules.fusion.speech import SpeakerFuseLayer
from tasks.tse.models.modules.textual.kce import (
    KCE,
    KCE_SAMPLE_RATE,
    DAEPhonemeTokenizer,
)

_FEATURE_NAMES = frozenset({"dae_kce"})
_TOKENIZER_KEYS = frozenset({"phoneme_map", "lexicon"})

DEFAULT_TEXTUAL_CONFIG = {
    "features": {
        "dae_kce": {
            "enabled": True,
            "padding_id": 0,
            "bos_id": 71,
            "eos_id": 72,
            "tokenizer": None,
            "adapter": {
                "input_dim": None,
                "output_dim": 128,
                "num_layers": 1,
            },
            "fusion": "FiLM",
            "multi_fuse": True,
            "checkpoint": None,
            "model_config": None,
            "fbank": None,
            "eps": None,
        }
    },
}


@dataclass(frozen=True)
class TextualFrontendOutput:
    """文本前端输出：句级 cue 与 presence mask。

    ``cue`` 为 ``[R, output_dim]``，缺失行精确为 0；``present`` 为 ``[R]``
    ``torch.bool``。前端不持有 separator，也不与 RI 谱或 latent 拼接。
    """

    cue: Tensor
    present: Tensor


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return value


def _positive_int(value: object, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{path} must be an exact positive int, got {value!r}")
    return value


def _resolve_tokenizer_config(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    config = _mapping(value, "textual features.dae_kce.tokenizer")
    unknown = sorted(set(config) - _TOKENIZER_KEYS)
    if unknown:
        raise ValueError(f"textual tokenizer contains unknown keys: {unknown}")
    if type(config.get("phoneme_map")) is not str or not config["phoneme_map"]:
        raise ValueError("textual tokenizer.phoneme_map must be a non-empty path")
    if "lexicon" in config and (
        type(config["lexicon"]) is not str or not config["lexicon"]
    ):
        raise ValueError("textual tokenizer.lexicon must be a non-empty path")
    return config


def _resolve_feature(value: object, *, mix_dim: int, num_stages: int) -> dict:
    """按上游默认值补全 feature 配置，未知键直接报错。"""
    feature = deep_update(
        DEFAULT_TEXTUAL_CONFIG["features"]["dae_kce"],
        dict(_mapping(value, "textual features.dae_kce")),
    )
    if feature["enabled"] is not True:
        raise ValueError("TextualFrontend requires dae_kce to be enabled")
    if type(feature["multi_fuse"]) is not bool:
        raise TypeError("textual features.dae_kce.multi_fuse must be a bool")
    feature["mix_dim"] = mix_dim
    feature["num_stages"] = num_stages
    return feature


class DAEKCEFeature(nn.Module):
    """把 KCE 计算与 separator 融合层合成一个文本特征。

    ``fusions`` 数量与上游一致：``multi_fuse=True`` 时为 ``num_stages`` 个，
    否则 1 个。``post`` 对缺失 cue 行返回原 latent，调用方不需要额外 gate。
    """

    def __init__(
        self,
        config,
        *,
        mix_dim,
        num_stages,
        sample_rate=KCE_SAMPLE_RATE,
        defer_pretrained=False,
        backbone_state_dict=None,
        allow_unsafe_checkpoint=False,
    ):
        super().__init__()
        self.mix_dim = _positive_int(mix_dim, "mix_dim")
        self.num_stages = _positive_int(num_stages, "num_stages")
        self.sample_rate = sample_rate
        feature = _resolve_feature(config, mix_dim=self.mix_dim, num_stages=self.num_stages)
        self.config = feature
        self.multi_fuse = feature["multi_fuse"]
        self.fusion = feature["fusion"]

        kce_config = {
            "checkpoint": feature["checkpoint"],
            "adapter": {
                key: value
                for key, value in _mapping(
                    feature["adapter"], "textual features.dae_kce.adapter"
                ).items()
                if value is not None
            },
            "padding_id": feature["padding_id"],
            "bos_id": feature["bos_id"],
            "eos_id": feature["eos_id"],
        }
        for key in ("model_config", "fbank", "eps"):
            if feature[key] is not None:
                kce_config[key] = feature[key]
        self.kce = KCE(
            kce_config,
            sample_rate=sample_rate,
            backbone_state_dict=backbone_state_dict,
            defer_pretrained=defer_pretrained,
            allow_unsafe_checkpoint=allow_unsafe_checkpoint,
        )
        self.output_dim = self.kce.output_dim
        self.tokenizer_config = _resolve_tokenizer_config(feature["tokenizer"])
        num_fusions = self.num_stages if self.multi_fuse else 1
        self.fusions = nn.ModuleList([
            SpeakerFuseLayer(
                embed_dim=self.output_dim,
                feat_dim=self.mix_dim,
                fuse_type=self.fusion,
            ) for _ in range(num_fusions)
        ])

    def compute(
        self,
        mixture,
        phoneme_ids,
        *,
        present,
        textual_lengths,
        mixture_lengths,
    ) -> TextualFrontendOutput:
        """计算一条句级 KCE cue ``[R, output_dim]``。"""
        cue = self.kce(
            mixture,
            phoneme_ids,
            present=present,
            textual_lengths=textual_lengths,
            mixture_lengths=mixture_lengths,
        )
        return TextualFrontendOutput(cue=cue, present=present)

    def post(self, mix_latent, output, stage):
        """在指定 separator block 前融合句级 cue。

        ``mix_latent`` 为 ``[R, nband, E, T]`` 或 ``[R, E, T]``；缺失 cue 行返回
        原 ``mix_latent``。
        """
        if not isinstance(output, TextualFrontendOutput):
            raise TypeError("post expects the TextualFrontendOutput from compute")
        if type(stage) is not int or not 0 <= stage < len(self.fusions):
            raise ValueError(
                f"stage must be an int in [0, {len(self.fusions)}), got {stage!r}"
            )
        if not isinstance(mix_latent, Tensor):
            raise TypeError("mix_latent must be a tensor")
        fused = self.fusions[stage](
            mix_latent, output.cue.unsqueeze(1).unsqueeze(-1)
        )
        return gate_fusion(mix_latent, fused, output.present)

    def build_tokenizer(self) -> DAEPhonemeTokenizer:
        """按配置构造离线文本 tokenizer（需要 ``g2p_en`` 时才导入）。"""
        if self.tokenizer_config is None:
            raise RuntimeError(
                "textual features.dae_kce.tokenizer is not configured; offline text "
                "tokenization is unavailable"
            )
        return DAEPhonemeTokenizer(
            self.tokenizer_config["phoneme_map"],
            self.tokenizer_config.get("lexicon"),
        )


class TextualFrontend(nn.Module):
    """在统一小接口后构造启用的文本特征（当前仅 ``dae_kce``）。"""

    def __init__(
        self,
        config,
        *,
        mix_dim,
        num_stages,
        sample_rate=KCE_SAMPLE_RATE,
        defer_pretrained=False,
        backbone_state_dict=None,
        allow_unsafe_checkpoint=False,
    ):
        super().__init__()
        merged = deep_update(DEFAULT_TEXTUAL_CONFIG, dict(_mapping(config, "textual config")))
        features = _mapping(merged["features"], "textual config.features")
        unknown = sorted(set(features) - _FEATURE_NAMES)
        if unknown:
            raise ValueError(f"textual config.features contains unknown features: {unknown}")
        self.mix_dim = _positive_int(mix_dim, "mix_dim")
        self.num_stages = _positive_int(num_stages, "num_stages")
        self.sample_rate = sample_rate
        self.dae_kce = DAEKCEFeature(
            features["dae_kce"],
            mix_dim=self.mix_dim,
            num_stages=self.num_stages,
            sample_rate=sample_rate,
            defer_pretrained=defer_pretrained,
            backbone_state_dict=backbone_state_dict,
            allow_unsafe_checkpoint=allow_unsafe_checkpoint,
        )
        self.output_dim = self.dae_kce.output_dim
        self.fusion = self.dae_kce.fusion
        self.multi_fuse = self.dae_kce.multi_fuse

    def compute(
        self,
        mixture,
        phoneme_ids,
        *,
        present,
        textual_lengths,
        mixture_lengths,
    ) -> TextualFrontendOutput:
        return self.dae_kce.compute(
            mixture,
            phoneme_ids,
            present=present,
            textual_lengths=textual_lengths,
            mixture_lengths=mixture_lengths,
        )

    def forward(
        self,
        mixture,
        phoneme_ids,
        *,
        present,
        textual_lengths,
        mixture_lengths,
    ) -> TextualFrontendOutput:
        return self.compute(
            mixture,
            phoneme_ids,
            present=present,
            textual_lengths=textual_lengths,
            mixture_lengths=mixture_lengths,
        )

    def post(self, mix_latent, output, stage):
        return self.dae_kce.post(mix_latent, output, stage)

    def build_tokenizer(self) -> DAEPhonemeTokenizer:
        return self.dae_kce.build_tokenizer()


__all__ = [
    "DAEKCEFeature",
    "DEFAULT_TEXTUAL_CONFIG",
    "TextualFrontend",
    "TextualFrontendOutput",
]
