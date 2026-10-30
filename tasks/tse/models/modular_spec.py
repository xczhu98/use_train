"""TSE_Modular 的纯配置解析与能力合同。"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from tasks.tse.models.modules.frontend_registry import (
    resolve_cue_frontend,
    validate_cue_frontend_capabilities,
)


Backbone = Literal["bsrnn", "tfgridnet", "nbc2", "dpccn", "spex_plus"]
CueKind = Literal["speaker", "spatial", "visual", "textual"]
InputDomain = Literal["spectrum", "waveform"]


@dataclass(frozen=True)
class STFTSpec:
    n_fft: int
    hop_length: int
    win_length: int
    window: str
    center: bool


@dataclass(frozen=True)
class CueSpec:
    """模型 frontend 消费的表示；不声明或排斥数据合同中的其他 cue slot。"""

    kind: CueKind
    frontend: str
    representation: str
    features: tuple[str, ...]
    options: Mapping[str, object]


@dataclass(frozen=True)
class PublicCombination:
    upstream_name: str
    backbone: Backbone
    cues: frozenset[CueKind]
    phase: str


@dataclass(frozen=True)
class ModularSpec:
    backbone: Backbone
    separator: Mapping[str, object]
    cues: Mapping[CueKind, CueSpec]
    composition: Mapping[str, object]
    sample_rate: int
    n_channel: int
    ref_channel: int
    separator_input_domain: InputDomain
    required_mixture_domains: frozenset[InputDomain]
    frontend_owner: Literal["runtime", "model"]
    stft: STFTSpec | None
    requires_geometry: bool
    upstream_name: str | None
    implementation_available: bool
    local_extension: bool


_PUBLIC_COMBINATION_ROWS = (
    PublicCombination("TSE_BSRNN_SPK", "bsrnn", frozenset({"speaker"}), "P2"),
    PublicCombination("TSE_BSRNN_SPATIAL", "bsrnn", frozenset({"spatial"}), "P2"),
    PublicCombination(
        "TSE_BSRNN_SPK_SPATIAL",
        "bsrnn",
        frozenset({"speaker", "spatial"}),
        "P2",
    ),
    PublicCombination("TSE_TFGRIDNET_SPK", "tfgridnet", frozenset({"speaker"}), "P3"),
    PublicCombination("TSE_NBC2_SPATIAL", "nbc2", frozenset({"spatial"}), "P3"),
    PublicCombination("TSE_DPCCN_SPK", "dpccn", frozenset({"speaker"}), "P3"),
    PublicCombination("TSE_SPEX_PLUS", "spex_plus", frozenset({"speaker"}), "P3"),
    PublicCombination("TSE_BSRNN_TEXTUAL", "bsrnn", frozenset({"textual"}), "P4"),
    PublicCombination("TSE_BSRNN_VISUAL", "bsrnn", frozenset({"visual"}), "P4"),
    PublicCombination("TSE_TFGRIDNET_VISUAL", "tfgridnet", frozenset({"visual"}), "P4"),
)
PUBLIC_COMBINATIONS: Mapping[str, PublicCombination] = MappingProxyType(
    {row.upstream_name: row for row in _PUBLIC_COMBINATION_ROWS}
)
_COMBINATION_BY_KEY = {
    (row.backbone, row.cues): row for row in _PUBLIC_COMBINATION_ROWS
}

# 仅登记上游 parity 实现（BSRNN speaker/spatial 组合）的真实可用性；本地扩展
# （textual/visual 接线、USEF append、track spatial）由 ``implementation_available``
# 与 ``upstream_name is None`` 表达，具体可用性另按 feature 判断，listen 尚未接线。
IMPLEMENTED_PUBLIC_COMBINATIONS: frozenset[str] = frozenset({
    "TSE_BSRNN_SPK", "TSE_BSRNN_SPATIAL", "TSE_BSRNN_SPK_SPATIAL",
})

_BACKBONES = frozenset({"bsrnn", "tfgridnet", "nbc2", "dpccn", "spex_plus"})
_CUE_KINDS = frozenset({"speaker", "spatial", "visual", "textual"})
_SPECTRAL_BACKBONES = frozenset({"bsrnn", "tfgridnet", "nbc2", "dpccn"})

_SEPARATOR_KEYS = {
    "bsrnn": frozenset({"feature_dim", "num_repeat", "causal", "nspk", "norm_type", "band_width", "rnn_norm_eps"}),
    "tfgridnet": frozenset(
        {
            "nspk",
            "n_layers",
            "hidden_channels",
            "n_head",
            "approx_qk_dim",
            "emb_dim",
            "emb_ks",
            "emb_hs",
            "eps",
            "causal",
            "estimation",
            "attention_context",
            "attention_output_norm",
        }
    ),
    "nbc2": frozenset({"nspk", "n_layers", "dim_hidden", "dim_ffn", "block"}),
    "dpccn": frozenset({"nspk", "tcn_layers", "tcn_blocks", "pool_sizes", "causal"}),
    "spex_plus": frozenset(
        {
            "enc_dim",
            "kernel_sizes",
            "stride",
            "bottleneck_dim",
            "conv_dim",
            "conv_kernel_size",
            "num_blocks",
            "num_repeats",
            "causal",
            "skip_connection",
            "mask_activation",
            "num_speakers",
        }
    ),
}
_NBC2_BLOCK_KEYS = frozenset(
    {"n_heads", "dropout", "conv_kernel_size", "n_conv_groups", "norms"}
)

_SPEAKER_FEATURE_KEYS = {
    "listen": frozenset({"glue"}),
    "usef": frozenset({"latent_dim", "approx_qk_dim", "num_heads", "time_kernel_size"}),
    "tfmap": frozenset({"type"}),
    "context": frozenset({"embedding_dim", "attention_dim", "num_heads", "fusion", "source", "injection"}),
    "spkemb": frozenset({"fusion", "injection"}),
    "spex_plus": frozenset({"embedding_dim"}),
}
_SPATIAL_FEATURE_KEYS = {
    "ipd": frozenset(),
    "cdf": frozenset(),
    "sdf": frozenset(),
    "delta_stft": frozenset(),
    "cyc_doaemb": frozenset({"alpha", "embedding_dim", "use_elevation", "fusion"}),
    "initstate_emb": frozenset({"alpha", "embedding_dim", "use_elevation"}),
}
_VISUAL_FEATURE_KEYS = {
    "muse_visual": frozenset(
        {"checkpoint", "freeze_frontend", "upsample", "fusion", "multi_fuse", "adapter"}
    )
}
_TEXTUAL_FEATURE_KEYS = {
    "dae_kce": frozenset(
        {
            "padding_id",
            "bos_id",
            "eos_id",
            "tokenizer",
            "adapter",
            "fusion",
            "multi_fuse",
            "checkpoint",
            "model_config",
            "fbank",
            "eps",
        }
    )
}
_FUSIONS = frozenset({"additive", "concat", "multiply", "FiLM"})


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    if any(type(key) is not str or not key for key in value):
        raise TypeError(f"{path} keys must be non-empty strings")
    return value


def _reject_unknown(
    value: Mapping[str, object], allowed: frozenset[str] | set[str], path: str
) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"{path} contains unknown keys: {unknown}")


def _positive_int(value: object, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{path} must be an exact positive int, got {value!r}")
    return value


def _non_negative_int(value: object, path: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{path} must be an exact non-negative int, got {value!r}")
    return value


def _bool(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{path} must be a bool, got {value!r}")
    return value


def _enum(value: object, allowed: frozenset[str] | set[str], path: str) -> str:
    if type(value) is not str or value not in allowed:
        raise ValueError(f"{path} must be one of {sorted(allowed)}, got {value!r}")
    return value


def _finite_number(value: object, path: str, *, positive: bool = False) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{path} must be a finite number, got {value!r}")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise ValueError(f"{path} must be finite, got {value!r}")
    if positive and result <= 0:
        raise ValueError(f"{path} must be positive, got {value!r}")
    return result


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _validate_int_sequence(
    value: object,
    path: str,
    *,
    exact_length: int | None = None,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path} must be a sequence of positive ints")
    result = tuple(_positive_int(item, f"{path}[{index}]") for index, item in enumerate(value))
    if not result:
        raise ValueError(f"{path} must not be empty")
    if exact_length is not None and len(result) != exact_length:
        raise ValueError(f"{path} must contain exactly {exact_length} values")
    return result


def _validate_separator(backbone: Backbone, value: object) -> Mapping[str, object]:
    separator = _mapping(value, "separator")
    _reject_unknown(separator, _SEPARATOR_KEYS[backbone], "separator")

    for key in (
        "feature_dim",
        "num_repeat",
        "nspk",
        "n_layers",
        "hidden_channels",
        "n_head",
        "approx_qk_dim",
        "emb_dim",
        "emb_ks",
        "emb_hs",
        "dim_hidden",
        "dim_ffn",
        "tcn_layers",
        "tcn_blocks",
        "enc_dim",
        "stride",
        "bottleneck_dim",
        "conv_dim",
        "conv_kernel_size",
        "num_blocks",
        "num_repeats",
        "num_speakers",
    ):
        if key in separator:
            _positive_int(separator[key], f"separator.{key}")
    for key in ("causal", "skip_connection"):
        if key in separator:
            _bool(separator[key], f"separator.{key}")
    if separator.get("nspk", 1) != 1:
        raise ValueError("separator.nspk must be 1 for target extraction")
    if "eps" in separator:
        _finite_number(separator["eps"], "separator.eps", positive=True)
    if "rnn_norm_eps" in separator:
        _finite_number(separator["rnn_norm_eps"], "separator.rnn_norm_eps", positive=True)
    if "norm_type" in separator:
        _enum(separator["norm_type"], {"GN", "cLN", "gLN", "BN"}, "separator.norm_type")
    if "estimation" in separator:
        _enum(separator["estimation"], {"spec", "mask"}, "separator.estimation")
    if "attention_output_norm" in separator:
        _enum(
            separator["attention_output_norm"],
            {"channel", "channel_frequency"},
            "separator.attention_output_norm",
        )
    if separator.get("attention_context") is not None:
        _positive_int(separator["attention_context"], "separator.attention_context")
    if "mask_activation" in separator:
        _enum(separator["mask_activation"], {"relu", "sigmoid"}, "separator.mask_activation")
    if "band_width" in separator:
        _validate_int_sequence(separator["band_width"], "separator.band_width")
    if "pool_sizes" in separator:
        _validate_int_sequence(separator["pool_sizes"], "separator.pool_sizes")
    if "kernel_sizes" in separator:
        _validate_int_sequence(separator["kernel_sizes"], "separator.kernel_sizes", exact_length=3)

    if "block" in separator:
        block = _mapping(separator["block"], "separator.block")
        _reject_unknown(block, _NBC2_BLOCK_KEYS, "separator.block")
        for key in ("n_heads", "conv_kernel_size", "n_conv_groups"):
            if key in block:
                _positive_int(block[key], f"separator.block.{key}")
        if "dropout" in block:
            dropout = _finite_number(block["dropout"], "separator.block.dropout")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("separator.block.dropout must be in [0, 1)")
        if "norms" in block:
            norms = block["norms"]
            if isinstance(norms, (str, bytes)) or not isinstance(norms, Sequence):
                raise TypeError("separator.block.norms must be a sequence")
            for index, norm in enumerate(norms):
                _enum(norm, {"LN", "BN", "GN", "GBN"}, f"separator.block.norms[{index}]")
    return _freeze(separator)  # type: ignore[return-value]


def _validate_feature_map(
    value: object,
    schemas: Mapping[str, frozenset[str]],
    path: str,
) -> tuple[tuple[str, ...], dict[str, Mapping[str, object]]]:
    features = _mapping(value, path)
    if not features:
        raise ValueError(f"{path} must enable at least one feature")
    unknown_features = sorted(set(features) - set(schemas))
    if unknown_features:
        raise ValueError(f"{path} contains unsupported features: {unknown_features}")
    validated: dict[str, Mapping[str, object]] = {}
    for name, raw_config in features.items():
        config = _mapping(raw_config, f"{path}.{name}")
        _reject_unknown(config, schemas[name], f"{path}.{name}")
        validated[name] = config
    return tuple(sorted(validated)), validated


def _validate_fusion_options(features: Mapping[str, Mapping[str, object]], path: str) -> None:
    for name, config in features.items():
        if "fusion" in config:
            _enum(config["fusion"], _FUSIONS, f"{path}.{name}.fusion")
        for key in (
            "glue",
            "latent_dim",
            "approx_qk_dim",
            "num_heads",
            "time_kernel_size",
            "embedding_dim",
            "attention_dim",
        ):
            if key in config:
                _positive_int(config[key], f"{path}.{name}.{key}")


def _resolve_speaker_cue(value: object) -> CueSpec:
    cue = _mapping(value, "cues.speaker")
    _reject_unknown(cue, {"frontend", "representation", "features", "encoder"}, "cues.speaker")
    frontend = resolve_cue_frontend("speaker", cue.get("frontend"))
    representation = _enum(
        cue.get("representation"), {"waveform", "embedding"}, "cues.speaker.representation"
    )
    feature_names, features = _validate_feature_map(
        cue.get("features"), _SPEAKER_FEATURE_KEYS, "cues.speaker.features"
    )
    validate_cue_frontend_capabilities(
        frontend, representation=representation, features=feature_names
    )
    _validate_fusion_options(features, "cues.speaker.features")
    if "tfmap" in features and "type" in features["tfmap"]:
        _enum(
            features["tfmap"]["type"],
            {"spec", "embedding"},
            "cues.speaker.features.tfmap.type",
        )
    if "tfmap" in features and features["tfmap"].get("type", "spec") == "embedding":
        if representation != "waveform":
            raise ValueError(
                "cues.speaker.features.tfmap.type='embedding' requires "
                "cues.speaker.representation='waveform'"
            )
        if "context" not in features:
            raise ValueError(
                "cues.speaker.features.tfmap.type='embedding' requires the "
                "context feature to consume frame-level speaker conditions"
            )
        context = features["context"]
        source = context.get("source")
        _enum(
            source,
            {"tfmap_embedding"},
            "cues.speaker.features.context.source",
        )
    elif "context" in features and "source" in features["context"]:
        _enum(features["context"]["source"], {"speaker_encoder"}, "cues.speaker.features.context.source")
    for name in ("context", "spkemb"):
        if name in features:
            _enum(features[name].get("injection", "latent"), {"latent", "block"}, f"cues.speaker.features.{name}.injection")
    waveform_only = set(feature_names) - {"spkemb"}
    if representation == "embedding" and waveform_only:
        raise ValueError(
            "speaker embedding representation only supports the spkemb feature; "
            f"got {sorted(waveform_only)}"
        )

    encoder: Mapping[str, object] = {}
    if "encoder" in cue:
        encoder = _mapping(cue["encoder"], "cues.speaker.encoder")
        _reject_unknown(
            encoder,
            {
                "name",
                "checkpoint",
                "pretrained_prefix",
                "embedding_dim",
                "feature_dim",
                "pooling",
                "fbank",
                "freeze",
            },
            "cues.speaker.encoder",
        )
        if "freeze" in encoder:
            _bool(encoder["freeze"], "cues.speaker.encoder.freeze")
        if encoder.get("freeze") is True and not encoder.get("checkpoint"):
            raise ValueError("cues.speaker.encoder.freeze=True requires checkpoint")
        for key in ("name", "checkpoint", "pretrained_prefix", "pooling"):
            if key in encoder and (type(encoder[key]) is not str or not encoder[key]):
                raise ValueError(f"cues.speaker.encoder.{key} must be a non-empty string")
            if key == "pretrained_prefix" and key in encoder and encoder[key] != encoder[key].strip():
                raise ValueError(
                    "cues.speaker.encoder.pretrained_prefix must be a trimmed string"
                )
        for key in ("embedding_dim", "feature_dim"):
            if key in encoder:
                _positive_int(encoder[key], f"cues.speaker.encoder.{key}")
        if "fbank" in encoder:
            fbank = _mapping(encoder["fbank"], "cues.speaker.encoder.fbank")
            _reject_unknown(
                fbank,
                {"num_mel_bins", "frame_shift_ms", "frame_length_ms", "dither"},
                "cues.speaker.encoder.fbank",
            )
            if "num_mel_bins" in fbank:
                _positive_int(fbank["num_mel_bins"], "cues.speaker.encoder.fbank.num_mel_bins")
            for key in ("frame_shift_ms", "frame_length_ms"):
                if key in fbank:
                    _finite_number(fbank[key], f"cues.speaker.encoder.fbank.{key}", positive=True)
            if "dither" in fbank:
                _finite_number(fbank["dither"], "cues.speaker.encoder.fbank.dither")

    if encoder:
        if representation == "embedding":
            _reject_unknown(encoder, {"embedding_dim"}, "cues.speaker.encoder for precomputed embedding")
        elif not {"context", "spkemb"}.intersection(feature_names):
            raise ValueError("cues.speaker.encoder is unused without context/spkemb")
        elif "name" in encoder:
            _enum(encoder["name"], {"ECAPA_TDNN_GLOB_c512", "ResNet34"}, "cues.speaker.encoder.name")
    if "usef" in features:
        usef = features["usef"]
        if usef.get("latent_dim", 128) % usef.get("num_heads", 4):
            raise ValueError("speaker USEF latent_dim must be divisible by num_heads")
        if usef.get("time_kernel_size", 1) % 2 == 0:
            raise ValueError("speaker USEF time_kernel_size must be odd")
    options = {"features": features}
    if encoder:
        options["encoder"] = encoder
    return CueSpec(
        kind="speaker",
        frontend=frontend,
        representation=representation,
        features=feature_names,
        options=_freeze(options),  # type: ignore[arg-type]
    )


def _resolve_pairs(value: object, n_channel: int) -> tuple[tuple[int, int], ...]:
    if value is None:
        return tuple(itertools.combinations(range(n_channel), 2))
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("cues.spatial.pairs must be a sequence of microphone pairs")
    pairs: list[tuple[int, int]] = []
    for pair_index, pair in enumerate(value):
        if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError(f"cues.spatial.pairs[{pair_index}] must contain two indices")
        left = _non_negative_int(pair[0], f"cues.spatial.pairs[{pair_index}][0]")
        right = _non_negative_int(pair[1], f"cues.spatial.pairs[{pair_index}][1]")
        if left == right:
            raise ValueError(f"cues.spatial.pairs[{pair_index}] must use distinct microphones")
        if left >= n_channel or right >= n_channel:
            raise ValueError(
                f"cues.spatial.pairs[{pair_index}] exceeds n_channel={n_channel}"
            )
        normalized = (left, right)
        if normalized in pairs:
            raise ValueError(f"cues.spatial.pairs contains duplicate pair {normalized}")
        pairs.append(normalized)
    if not pairs:
        raise ValueError("cues.spatial.pairs must not be empty")
    return tuple(pairs)


def _resolve_spatial_cue(value: object, n_channel: int) -> CueSpec:
    cue = _mapping(value, "cues.spatial")
    _reject_unknown(cue, {"frontend", "representation", "features", "pairs"}, "cues.spatial")
    frontend = resolve_cue_frontend("spatial", cue.get("frontend"))
    representation = _enum(
        cue.get("representation"), {"static", "track"}, "cues.spatial.representation"
    )
    feature_names, features = _validate_feature_map(
        cue.get("features"), _SPATIAL_FEATURE_KEYS, "cues.spatial.features"
    )
    validate_cue_frontend_capabilities(
        frontend, representation=representation, features=feature_names
    )
    _validate_fusion_options(features, "cues.spatial.features")
    for name in ("cyc_doaemb", "initstate_emb"):
        config = features.get(name)
        if config is None:
            continue
        if "alpha" in config:
            _finite_number(config["alpha"], f"cues.spatial.features.{name}.alpha", positive=True)
        if "use_elevation" in config:
            _bool(config["use_elevation"], f"cues.spatial.features.{name}.use_elevation")
    pairs = _resolve_pairs(cue.get("pairs"), n_channel)
    return CueSpec(
        kind="spatial",
        frontend=frontend,
        representation=representation,
        features=feature_names,
        options=_freeze({"features": features, "pairs": pairs}),  # type: ignore[arg-type]
    )


def _resolve_visual_cue(value: object) -> CueSpec:
    cue = _mapping(value, "cues.visual")
    _reject_unknown(cue, {"frontend", "representation", "features"}, "cues.visual")
    frontend = resolve_cue_frontend("visual", cue.get("frontend"))
    representation = _enum(
        cue.get("representation"), {"features", "raw"}, "cues.visual.representation"
    )
    feature_names, features = _validate_feature_map(
        cue.get("features"), _VISUAL_FEATURE_KEYS, "cues.visual.features"
    )
    validate_cue_frontend_capabilities(
        frontend, representation=representation, features=feature_names
    )
    _validate_fusion_options(features, "cues.visual.features")
    muse = features["muse_visual"]
    for key in ("freeze_frontend", "upsample", "multi_fuse"):
        if key in muse:
            _bool(muse[key], f"cues.visual.features.muse_visual.{key}")
    if "checkpoint" in muse and (type(muse["checkpoint"]) is not str or not muse["checkpoint"]):
        raise ValueError("cues.visual.features.muse_visual.checkpoint must be a non-empty string")
    if "adapter" in muse:
        adapter = _mapping(muse["adapter"], "cues.visual.features.muse_visual.adapter")
        _reject_unknown(
            adapter,
            {"type", "input_dim", "output_dim", "hidden_dim", "num_layers"},
            "cues.visual.features.muse_visual.adapter",
        )
        if "type" in adapter:
            _enum(
                adapter["type"],
                {"direct", "muse", "clearer"},
                "cues.visual.features.muse_visual.adapter.type",
            )
        for key in ("input_dim", "output_dim", "hidden_dim", "num_layers"):
            if key in adapter:
                _positive_int(adapter[key], f"cues.visual.features.muse_visual.adapter.{key}")
    return CueSpec(
        kind="visual",
        frontend=frontend,
        representation=representation,
        features=feature_names,
        options=_freeze({"features": features}),  # type: ignore[arg-type]
    )


def _resolve_textual_cue(value: object) -> CueSpec:
    cue = _mapping(value, "cues.textual")
    _reject_unknown(cue, {"frontend", "representation", "features"}, "cues.textual")
    frontend = resolve_cue_frontend("textual", cue.get("frontend"))
    representation = _enum(
        cue.get("representation"), {"phoneme_ids"}, "cues.textual.representation"
    )
    feature_names, features = _validate_feature_map(
        cue.get("features"), _TEXTUAL_FEATURE_KEYS, "cues.textual.features"
    )
    validate_cue_frontend_capabilities(
        frontend, representation=representation, features=feature_names
    )
    kce = features["dae_kce"]
    if "fusion" in kce:
        _enum(kce["fusion"], _FUSIONS, "cues.textual.features.dae_kce.fusion")
    for key in ("padding_id", "bos_id", "eos_id"):
        if key in kce:
            _non_negative_int(kce[key], f"cues.textual.features.dae_kce.{key}")
    if "multi_fuse" in kce:
        _bool(kce["multi_fuse"], "cues.textual.features.dae_kce.multi_fuse")
    if "checkpoint" in kce and (type(kce["checkpoint"]) is not str or not kce["checkpoint"]):
        raise ValueError("cues.textual.features.dae_kce.checkpoint must be a non-empty string")
    if "tokenizer" in kce:
        tokenizer = _mapping(kce["tokenizer"], "cues.textual.features.dae_kce.tokenizer")
        _reject_unknown(
            tokenizer,
            {"phoneme_map", "lexicon"},
            "cues.textual.features.dae_kce.tokenizer",
        )
        for key, item in tokenizer.items():
            if type(item) is not str or not item:
                raise ValueError(
                    f"cues.textual.features.dae_kce.tokenizer.{key} must be a non-empty string"
                )
    if "adapter" in kce:
        adapter = _mapping(kce["adapter"], "cues.textual.features.dae_kce.adapter")
        _reject_unknown(
            adapter,
            {"output_dim", "num_layers"},
            "cues.textual.features.dae_kce.adapter",
        )
        for key, item in adapter.items():
            _positive_int(item, f"cues.textual.features.dae_kce.adapter.{key}")
    for key in ("model_config", "fbank"):
        if key in kce:
            _mapping(kce[key], f"cues.textual.features.dae_kce.{key}")
    if "eps" in kce:
        _finite_number(kce["eps"], "cues.textual.features.dae_kce.eps", positive=True)
    return CueSpec(
        kind="textual",
        frontend=frontend,
        representation=representation,
        features=feature_names,
        options=_freeze({"features": features}),  # type: ignore[arg-type]
    )


def _resolve_cues(value: object, n_channel: int) -> Mapping[CueKind, CueSpec]:
    cues = _mapping(value, "cues")
    _reject_unknown(cues, _CUE_KINDS, "cues")
    if not cues:
        raise ValueError("cues must select at least one cue")
    resolved: dict[CueKind, CueSpec] = {}
    for kind, raw_cue in cues.items():
        if kind == "speaker":
            resolved["speaker"] = _resolve_speaker_cue(raw_cue)
        elif kind == "spatial":
            resolved["spatial"] = _resolve_spatial_cue(raw_cue, n_channel)
        elif kind == "visual":
            resolved["visual"] = _resolve_visual_cue(raw_cue)
        else:
            resolved["textual"] = _resolve_textual_cue(raw_cue)
    return MappingProxyType(resolved)


def _resolve_composition(
    value: object | None,
    cues: Mapping[CueKind, CueSpec],
) -> Mapping[str, object]:
    composition = {} if value is None else _mapping(value, "composition")
    _reject_unknown(composition, {"usef_mode", "allow_all_missing"}, "composition")
    allow_all_missing = _bool(
        composition.get("allow_all_missing", False), "composition.allow_all_missing"
    )
    speaker = cues.get("speaker")
    has_usef = speaker is not None and "usef" in speaker.features
    usef_mode = composition.get("usef_mode")
    if has_usef:
        usef_mode = _enum(usef_mode, {"replace", "append"}, "composition.usef_mode")
        if "spatial" in cues and usef_mode != "append":
            raise ValueError(
                "speaker USEF with spatial cues requires composition.usef_mode='append'; "
                "the upstream joint path does not implement USEF replacement"
            )
    elif usef_mode is not None:
        raise ValueError("composition.usef_mode is only valid when speaker USEF is selected")
    return MappingProxyType(
        {"allow_all_missing": allow_all_missing, "usef_mode": usef_mode}
    )


def _resolve_stft(
    *,
    required: bool,
    n_fft: object | None,
    hop_length: object | None,
    win_length: object | None,
    window: object | None,
    center: object | None,
) -> STFTSpec | None:
    values = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "window": window,
        "center": center,
    }
    if not required:
        supplied = sorted(key for key, value in values.items() if value is not None)
        if supplied:
            raise ValueError(
                "waveform-native backbone rejects unused top-level STFT fields: "
                f"{supplied}"
            )
        return None
    missing = sorted(key for key, value in values.items() if value is None)
    if missing:
        raise ValueError(f"spectrum backbone requires top-level STFT fields: {missing}")
    resolved_n_fft = _positive_int(n_fft, "n_fft")
    resolved_hop = _positive_int(hop_length, "hop_length")
    resolved_win = _positive_int(win_length, "win_length")
    if not resolved_hop <= resolved_win <= resolved_n_fft:
        raise ValueError("STFT geometry must satisfy hop_length <= win_length <= n_fft")
    resolved_window = _enum(window, {"hann", "hamming"}, "window")
    resolved_center = _bool(center, "center")
    if not resolved_center and resolved_win < resolved_n_fft:
        raise ValueError("center=False requires win_length == n_fft")
    if not resolved_center and resolved_window == "hann":
        raise ValueError("center=False does not support periodic Hann at the left boundary")
    return STFTSpec(
        n_fft=resolved_n_fft,
        hop_length=resolved_hop,
        win_length=resolved_win,
        window=resolved_window,
        center=resolved_center,
    )


def _validate_backbone_cues(
    backbone: Backbone,
    cues: Mapping[CueKind, CueSpec],
    *,
    sample_rate: int,
    n_channel: int,
    composition: Mapping[str, object],
) -> None:
    cue_kinds = frozenset(cues)
    if (backbone, cue_kinds) not in _COMBINATION_BY_KEY:
        raise ValueError(
            f"unsupported TSE_Modular combination: backbone={backbone!r}, "
            f"cues={sorted(cue_kinds)}"
        )

    speaker = cues.get("speaker")
    if speaker is not None:
        allowed_features = {
            "bsrnn": {"listen", "usef", "tfmap", "context", "spkemb"},
            "tfgridnet": {"listen", "usef", "tfmap", "context", "spkemb"},
            "dpccn": {"spkemb"},
            "spex_plus": {"spex_plus"},
        }[backbone]
        unsupported = sorted(set(speaker.features) - allowed_features)
        if unsupported:
            raise ValueError(f"{backbone} does not support speaker features: {unsupported}")
        if backbone == "dpccn" and speaker.features != ("spkemb",):
            raise ValueError("dpccn requires exactly the spkemb speaker feature")
        if backbone == "spex_plus":
            if speaker.features != ("spex_plus",) or speaker.representation != "waveform":
                raise ValueError("spex_plus requires waveform speaker cue with spex_plus feature")
        if "listen" in speaker.features and n_channel != 1:
            raise ValueError("speaker listen requires n_channel=1")
        if "usef" in speaker.features and composition["usef_mode"] == "replace" and n_channel != 1:
            raise ValueError("USEF replacement requires n_channel=1")

    if backbone in {"dpccn", "spex_plus"} and n_channel != 1:
        raise ValueError(f"{backbone} requires n_channel=1")

    spatial = cues.get("spatial")
    if spatial is not None:
        if n_channel < 2:
            raise ValueError("spatial cue requires n_channel >= 2")
        target_conditioned = {"cdf", "sdf", "cyc_doaemb", "initstate_emb"}
        if not target_conditioned.intersection(spatial.features):
            raise ValueError(
                "spatial cue requires at least one target-conditioned feature: "
                "cdf, sdf, cyc_doaemb, or initstate_emb"
            )
        if backbone == "nbc2" and "initstate_emb" in spatial.features:
            raise ValueError("nbc2 does not support spatial initstate_emb")

    if "textual" in cues:
        if n_channel != 1:
            raise ValueError("dae_kce textual combination requires n_channel=1")
        if sample_rate != 16000:
            raise ValueError("dae_kce textual combination requires sample_rate=16000")


def _resolve_upstream_name(
    backbone: Backbone,
    cues: Mapping[CueKind, CueSpec],
    composition: Mapping[str, object],
) -> str | None:
    row = _COMBINATION_BY_KEY[(backbone, frozenset(cues))]
    speaker = cues.get("speaker")
    spatial = cues.get("spatial")
    if speaker is not None:
        features = speaker.options["features"]
        if features.get("tfmap", {}).get("type") == "embedding" or any(
            features.get(name, {}).get("injection") == "block" for name in ("context", "spkemb")
        ):
            return None
    # visual/textual 沿用上游注册名作为组合键，但本仓库只提供本地接线，不承诺
    # WeSep upstream checkpoint 的键结构或参数布局，因此不声明 upstream parity。
    if {"visual", "textual"}.intersection(cues):
        return None
    if speaker is not None and "usef" in speaker.features:
        if spatial is not None or composition["usef_mode"] == "append":
            return None
    if spatial is not None and spatial.representation != "static":
        return None
    return row.upstream_name


def resolve_modular_spec(
    *,
    backbone: str,
    separator: Mapping[str, object],
    cues: Mapping[str, object],
    composition: Mapping[str, object] | None,
    sample_rate: int,
    n_channel: int,
    ref_channel: int,
    n_fft: int | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str | None = None,
    center: bool | None = None,
    require_implemented: bool = False,
) -> ModularSpec:
    """严格解析一个 TSE_Modular 配置，不导入或构造任何模型/前端。"""
    if type(require_implemented) is not bool:
        raise TypeError("require_implemented must be a bool")
    resolved_backbone = _enum(backbone, _BACKBONES, "backbone")
    resolved_sample_rate = _positive_int(sample_rate, "sample_rate")
    resolved_n_channel = _positive_int(n_channel, "n_channel")
    resolved_ref_channel = _non_negative_int(ref_channel, "ref_channel")
    if resolved_ref_channel >= resolved_n_channel:
        raise ValueError(
            f"ref_channel={resolved_ref_channel} must be smaller than n_channel={resolved_n_channel}"
        )

    resolved_separator = _validate_separator(resolved_backbone, separator)
    resolved_cues = _resolve_cues(cues, resolved_n_channel)
    resolved_composition = _resolve_composition(composition, resolved_cues)
    _validate_backbone_cues(
        resolved_backbone,
        resolved_cues,
        sample_rate=resolved_sample_rate,
        n_channel=resolved_n_channel,
        composition=resolved_composition,
    )

    spectrum_backbone = resolved_backbone in _SPECTRAL_BACKBONES
    stft = _resolve_stft(
        required=spectrum_backbone,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
    )
    speaker = resolved_cues.get("speaker")
    if speaker is not None:
        features = speaker.options["features"]
        block_sources = sum(features.get(name, {}).get("injection", "latent") == "block" for name in ("context", "spkemb"))
        spatial = resolved_cues.get("spatial")
        block_sources += int(spatial is not None and "cyc_doaemb" in spatial.features)
        if block_sources > 1:
            raise ValueError("only one per-block cue source can be active")
        if any(features.get(name, {}).get("injection") == "block" for name in ("context", "spkemb")) and resolved_backbone != "bsrnn":
            raise ValueError("speaker block injection requires backbone='bsrnn'")
        if features.get("tfmap", {}).get("type") == "embedding":
            if resolved_backbone != "bsrnn" or resolved_n_channel != 1 or set(resolved_cues) != {"speaker"}:
                raise ValueError("tfmap embedding requires mono speaker-only BSRNN")
            if resolved_separator.get("causal", False):
                raise ValueError("tfmap embedding does not support causal mode")
            if {"listen", "usef"}.intersection(features):
                raise ValueError("tfmap embedding cannot be combined with listen/usef")
            if stft is None or not stft.center or stft.window != "hann" or stft.win_length != stft.n_fft:
                raise ValueError("tfmap embedding requires centered Hann STFT with win_length=n_fft")
            encoder = speaker.options.get("encoder", {})
            if encoder.get("name", "ECAPA_TDNN_GLOB_c512") != "ECAPA_TDNN_GLOB_c512":
                raise ValueError("tfmap embedding requires the frame-preserving ECAPA encoder")
            fbank = encoder.get("fbank", {})
            for key, expected in (("frame_length_ms", stft.n_fft * 1000.0 / resolved_sample_rate),
                                  ("frame_shift_ms", stft.hop_length * 1000.0 / resolved_sample_rate), ("dither", 0.0)):
                if key in fbank and fbank[key] != expected:
                    raise ValueError(f"tfmap embedding requires encoder.fbank.{key}={expected}")
    domains: set[InputDomain] = {"spectrum" if spectrum_backbone else "waveform"}
    speaker = resolved_cues.get("speaker")
    if speaker is not None and "listen" in speaker.features:
        domains.add("waveform")
    if "textual" in resolved_cues:
        domains.add("waveform")
    if speaker is not None and speaker.options["features"].get("tfmap", {}).get("type") == "embedding":
        domains.add("waveform")

    upstream_name = _resolve_upstream_name(
        resolved_backbone, resolved_cues, resolved_composition
    )
    visual = resolved_cues.get("visual")
    implementation_available = (
        resolved_backbone == "bsrnn"
        and set(resolved_cues) <= {"speaker", "spatial", "visual", "textual"}
        and (speaker is None or set(speaker.features) <= {"usef", "tfmap", "context", "spkemb"})
        # raw video 需要 MuSE encoder 与视频解码，尚未接线；只接受预提取帧级特征。
        and (visual is None or visual.representation == "features")
    )
    spec = ModularSpec(
        backbone=resolved_backbone,  # type: ignore[arg-type]
        separator=resolved_separator,
        cues=resolved_cues,
        composition=resolved_composition,
        sample_rate=resolved_sample_rate,
        n_channel=resolved_n_channel,
        ref_channel=resolved_ref_channel,
        separator_input_domain="spectrum" if spectrum_backbone else "waveform",
        required_mixture_domains=frozenset(domains),
        frontend_owner="runtime" if spectrum_backbone else "model",
        stft=stft,
        requires_geometry="spatial" in resolved_cues,
        upstream_name=upstream_name,
        implementation_available=implementation_available,
        local_extension=implementation_available and upstream_name is None,
    )
    if require_implemented and not implementation_available:
        label = upstream_name or f"local:{resolved_backbone}+{'+'.join(sorted(resolved_cues))}"
        detail = ""
        if visual is not None and visual.representation == "raw":
            detail = ": raw video input is not wired; use visual representation='features'"
        raise NotImplementedError(
            f"TSE_Modular implementation is not available for {label}{detail}"
        )
    return spec


__all__ = [
    "CueSpec",
    "IMPLEMENTED_PUBLIC_COMBINATIONS",
    "ModularSpec",
    "PUBLIC_COMBINATIONS",
    "PublicCombination",
    "STFTSpec",
    "resolve_modular_spec",
]
