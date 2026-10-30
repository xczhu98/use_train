"""Typed SS Dataset configuration and split-specific validation."""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass
from typing import Literal

from core.data.constants import normalize_data_mode
from core.data.config_flags import resolve_placeholder_num_samples
from core.data.dataset_contract import DatasetContract
from core.models.io_contract import TensorField


_SS_KEYS = frozenset({"num_sources", "sir_range"})
_COMMON_SPLIT_KEYS = frozenset({
    "num_per_epoch", "filter_by_duration", "filter_min_duration",
    "filter_max_duration", "speech_length_file", "target_sr",
    "sampling", "repeat_to_max_duration", "repeat_tile_silence_gap",
    "repeat_tile_silence_gap_range", "placeholder_num_samples", "strict_load",
    "use_dynamic", "inherit_simulation",
    "inherit_train_simulation", "mode",
})
_OFFLINE_SPLIT_KEYS = _COMMON_SPLIT_KEYS | {"manifest", "max_duration_seconds", "simulation_cfg", "rir_scp"}
_DYNAMIC_SPLIT_KEYS = _COMMON_SPLIT_KEYS | {"speech_scp", "segment_seconds", "max_duration_seconds", "noise_scp", "rir_scp", "windnoise_scp", "music_scp", "simulation_cfg", "export_simulation_stems"}


@dataclass(frozen=True)
class SSTaskOptions:
    """仅保留 SS 自己拥有的构造参数；公共 split 字段由 core builder 解析。"""

    num_sources: int
    sir_range: tuple[float, float]


def resolve_ss_task_options(config: dict) -> SSTaskOptions:
    """解析 SS 专属字段，不重复解析 data split 的公共生命周期配置。"""
    if not isinstance(config, dict):
        raise TypeError("SS config must be a mapping")
    ss = config.get("ss")
    if not isinstance(ss, dict):
        raise ValueError("SS config requires a top-level 'ss' mapping")
    _reject_unknown_keys(ss, _SS_KEYS, "ss")
    num_sources = _exact_positive_int(ss.get("num_sources"), "ss.num_sources")
    raw_sir = ss.get("sir_range", [-5.0, 5.0])
    if not isinstance(raw_sir, (list, tuple)) or len(raw_sir) != 2:
        raise ValueError("ss.sir_range must be [low, high]")
    sir_low = _finite_number(raw_sir[0], "ss.sir_range[0]")
    sir_high = _finite_number(raw_sir[1], "ss.sir_range[1]")
    if sir_low > sir_high:
        raise ValueError("ss.sir_range must be [low, high]")
    return SSTaskOptions(num_sources=num_sources, sir_range=(sir_low, sir_high))


def _reject_unknown_keys(mapping: dict, allowed: frozenset[str], location: str) -> None:
    unknown = sorted(key for key in mapping if key not in allowed)
    if unknown:
        raise KeyError(f"{location} contains unknown keys: {unknown}")


def _exact_positive_int(value, location: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{location} must be an exact positive int")
    return value


def _exact_int(value, location: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{location} must be an exact int")
    return value


def _exact_bool(value, location: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{location} must be an exact bool")
    return value


def _finite_number(value, location: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise TypeError(f"{location} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{location} must be positive")
    return float(value)


def build_ss_dataset_contract(
    *, sample_rate: int, num_sources: int
) -> DatasetContract:
    return DatasetContract(
        task="ss",
        fields=(
            TensorField("mixture", "waveform", ("batch", "time")),
            TensorField(
                "sources",
                "waveform",
                ("batch", "source", "time"),
                multiple=True,
            ),
            TensorField("lengths", "label", ("batch",)),
            TensorField("source_mask", "mask", ("batch", "source")),
        ),
        sample_rate=sample_rate,
        cardinality="fixed_multi",
        num_sources=num_sources,
        spatial_observation_fields=("mixture",),
    )


@dataclass(frozen=True)
class SSDatasetConfig:
    """SS Dataset Adapter 构造所需的完整 split 配置。"""

    split: Literal["train", "validation"]
    storage: Literal["dynamic", "offline"]
    num_sources: int
    sample_rate: int
    seed: int
    strict_load: bool
    placeholder_num_samples: int = 1024
    manifest: str | None = None
    speech_scp: str | None = None
    max_duration_seconds: float | None = None
    segment_seconds: float = 3.0
    sir_range: tuple[float, float] = (-5.0, 5.0)
    dynamic_kwargs: dict | None = None
    filter_by_duration: bool = True
    filter_min_duration: float = 0.5
    filter_max_duration: float = 30.0

    @classmethod
    def from_config(
        cls,
        config: dict,
        split: Literal["train", "validation"],
    ) -> "SSDatasetConfig":
        if split not in {"train", "validation"}:
            raise ValueError("SS split must be 'train' or 'validation'")
        if not isinstance(config, dict):
            raise TypeError("SS config must be a mapping")
        data = config.get("data")
        meta = config.get("meta", {})
        if not isinstance(data, dict):
            raise ValueError("SS config requires top-level data and ss mappings")
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            raise TypeError("meta must be a mapping")
        mode = normalize_data_mode(data.get("mode"))
        if mode.task != "ss":
            raise ValueError(f"data.mode belongs to task {mode.task!r}, expected 'ss'")
        task_options = resolve_ss_task_options(config)
        section = data.get(split)
        if not isinstance(section, dict):
            raise ValueError(f"data.{split} must be a mapping")
        section = copy.deepcopy(section)
        # 兼容旧 typed-config facade 的精确异常类型；生产 build_dataset 仍由
        # shared builder 统一执行同一字段的语义校验。
        if "strict_load" in data:
            _exact_bool(data["strict_load"], "data.strict_load")
        if "max_duration_seconds" in section:
            _finite_number(
                section["max_duration_seconds"],
                f"data.{split}.max_duration_seconds",
                positive=True,
            )
        allowed_split_keys = (
            _OFFLINE_SPLIT_KEYS if mode.storage == "offline" else _DYNAMIC_SPLIT_KEYS
        )
        _reject_unknown_keys(section, allowed_split_keys, f"data.{split}")

        # 公共生命周期（duration、sampling、resource path、strict/placeholder 和
        # validation inheritance）必须由 core DatasetBuildSpec 解析；这里仅保留
        # SSDatasetConfig 这个旧校验入口的结构化兼容返回值。
        from core.data.dataset_build import (
            DatasetBuildSpec,
            SamplingCapabilities,
            apply_duration_filter_config,
            resolve_dataset_build_options,
            resolve_dynamic_dataset_build_request,
            validate_layer_a_runtime_config,
            _resolve_effective_sampling,
        )

        common_spec = DatasetBuildSpec(
            task="ss",
            dynamic_dataset_cls=lambda **kwargs: kwargs,
            simulation_cfg_strict_from_cfg=True,
            sampling_capabilities=SamplingCapabilities(True, True, True, True),
            dynamic_validation_default=True,
        )
        dynamic_kwargs = None
        if mode.storage == "dynamic":
            request = resolve_dynamic_dataset_build_request(
                config,
                "train" if split == "train" else "val",
                meta=meta,
                spec=common_spec,
            )
            dynamic_kwargs = dict(request.dataset_kwargs)
            section = dict(dynamic_kwargs)
            sample_rate = _exact_positive_int(
                dynamic_kwargs.get("target_sr"), "data.sample_rate"
            )
            strict_load = dynamic_kwargs["strict_load"]
            placeholder_num_samples = dynamic_kwargs["placeholder_num_samples"]
            # 旧 SSDatasetConfig facade 的调用方仍显式传入这些 canonical
            # 构造参数；shared build_dataset 生产路径不依赖该 facade。
            for key in ("speech_scp", "target_sr", "strict_load", "placeholder_num_samples", "segment_seconds"):
                dynamic_kwargs.pop(key, None)
        else:
            options = resolve_dataset_build_options(
                config,
                "train" if split == "train" else "val",
                expected_task="ss",
            )
            section = dict(options.section)
            apply_duration_filter_config(section)
            _resolve_effective_sampling(
                data[split].get("sampling"),
                capabilities=common_spec.sampling_capabilities,
                mode="offline",
                is_validation=split == "validation",
                explicitly_configured="sampling" in data[split],
            )
            validate_layer_a_runtime_config(section, config_path=f"data.{split}")
            sample_rate = _exact_positive_int(
                options.target_sr, "data.sample_rate"
            )
            strict_load = options.strict_load
            placeholder_num_samples = options.placeholder_num_samples

        manifest = section.get("manifest")
        speech_scp = section.get("speech_scp")
        if mode.storage == "offline":
            if not isinstance(manifest, str) or not manifest.strip():
                raise ValueError(f"data.{split}.manifest must be a non-empty string")
        elif not isinstance(speech_scp, str) or not speech_scp.strip():
            raise ValueError(f"data.{split}.speech_scp must be a non-empty string")
        max_duration = section.get("max_duration_seconds")
        if max_duration is not None:
            max_duration = _finite_number(
                max_duration,
                f"data.{split}.max_duration_seconds",
                positive=True,
            )
        segment_seconds = _finite_number(
            section.get("segment_seconds", 3.0),
            f"data.{split}.segment_seconds",
            positive=True,
        )
        seed = _exact_int(meta.get("seed", 0), "meta.seed")
        filter_by_duration = _exact_bool(
            section.get("filter_by_duration", False), f"data.{split}.filter_by_duration"
        )
        filter_min_duration = _finite_number(
            section.get("filter_min_duration", 0.5),
            f"data.{split}.filter_min_duration",
        )
        filter_max_duration = _finite_number(
            section.get("filter_max_duration", 30.0),
            f"data.{split}.filter_max_duration",
        )
        if filter_min_duration < 0 or (
            filter_max_duration > 0 and filter_min_duration > filter_max_duration
        ):
            raise ValueError("SS duration filter requires 0 <= min <= max")
        return cls(
            split=split,
            storage=mode.storage,
            num_sources=task_options.num_sources,
            sample_rate=sample_rate,
            seed=seed,
            strict_load=strict_load,
            placeholder_num_samples=placeholder_num_samples,
            manifest=manifest.strip() if isinstance(manifest, str) else None,
            speech_scp=speech_scp.strip() if isinstance(speech_scp, str) else None,
            max_duration_seconds=(
                max_duration if max_duration is not None else None
            ),
            segment_seconds=segment_seconds,
            sir_range=task_options.sir_range,
            dynamic_kwargs=dynamic_kwargs,
            filter_by_duration=filter_by_duration,
            filter_min_duration=filter_min_duration,
            filter_max_duration=filter_max_duration,
        )
