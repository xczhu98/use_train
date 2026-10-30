"""Shared dataset-build helpers (SE & AEC).

Constants and utilities used by both ``tasks.se.data.build_dataset`` and
``tasks.aec.data.build_dataset`` / ``tasks.aec.data.dynamic_aec_build``.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable, Optional

from core.data.config_flags import (
    resolve_placeholder_num_samples,
    resolve_simulation_cfg_strict,
    resolve_strict_load,
)
from core.data.constants import NormalizedDataMode, normalize_data_mode
from core.data.offline_dataset import OfflineDataset
from core.data.resource_paths import resolve_data_section_resource_paths
from core.data.simulation_config_contract import validate_simulation_config_contract
from core.utils.validation import require_positive_int

if TYPE_CHECKING:
    from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
VALIDATION_DYNAMIC_DATA_SOURCE_KEYS = (
    "speech_scp",
    "noise_scp",
    "music_scp",
    "rir_scp",
    "windnoise_scp",
    "speech_length_file",
    "filter_min_duration",
    "filter_max_duration",
    "target_sr",
    "speech_min_sample_rate",
    "speech_selected_sets",
    "max_duration_seconds",
    "max_duration",
)

VALIDATION_META_KEYS = ("mode", "use_dynamic", "inherit_simulation")

OFFLINE_ONLY_KEYS = (
    "clean_scp",
    "noisy_scp",
    "sample_rate",
    "target_sr",
)

# 这些键仅供 builder、采样器或 split 合并使用，绝不能传入严格 Dataset 构造器。
BUILDER_ONLY_SECTION_KEYS = frozenset(
    {
        "collate_fn",
        "dataloader",
        "inherit_simulation",
        "inherit_train_simulation",
        "mode",
        "num_per_epoch",
        "placeholder_num_samples",
        "return_noise",
        "sampling",
        "simulation_cfg_strict",
        "strict_load",
        "use_dynamic",
    }
)


def strip_builder_only_section_keys(section: dict) -> None:
    """原地剥离所有 Dataset 不应接收的 builder 元数据。"""
    for key in BUILDER_ONLY_SECTION_KEYS:
        section.pop(key, None)


DATASET_LIFECYCLE_KEYS = frozenset(
    {
        "max_duration_seconds",
        "max_duration",
        "repeat_to_max_duration",
        "repeat_tile_silence_gap",
        "repeat_tile_silence_gap_range",
    }
)
DATA_RESOURCE_KEYS_IN_SPLIT = frozenset(
    {
        "speech_scp",
        "noise_scp",
        "music_scp",
        "windnoise_scp",
        "rir_scp",
        "spk2enroll_json",
        "real_farend_echo_pair_json",
    }
)

SE_PASE_DATASET_KEYS = frozenset({"pase_preprocess"})


AEC_REF_ECHO_DATASET_KEYS = frozenset(
    {
        "aec_ref_echo_fine_gcc_align",
        "aec_ref_echo_gcc_bias_seconds",
        "aec_ref_echo_gcc_max_tau_seconds",
        "aec_ref_echo_gcc_max_pad_seconds",
        "aec_ref_echo_gcc_farend_pad_policy",
        "aec_ref_echo_gcc_min_pad_samples",
        "aec_ref_echo_gcc_peak_threshold",
        "aec_ref_echo_gcc_prominence_db",
        "aec_ref_echo_gcc_tv_iqr_ms",
        "aec_ref_echo_gcc_max_windows",
        "aec_ref_echo_gcc_downsample",
        "aec_ref_echo_random_delay_prob",
        "aec_ref_echo_random_delay_mode",
        "aec_ref_echo_random_delay_distribution",
        "aec_ref_echo_random_delay_beta_a",
        "aec_ref_echo_random_delay_beta_b",
        "aec_ref_echo_random_delay_uniform_prob",
        "aec_ref_echo_delay_range_ms",
        "aec_time_delay_jitter",
        "aec_echo_device_eq",
        "aec_echo_clipping",
    }
)

TSE_DATASET_KEYS = frozenset(
    {
        "enrollment_duration",
        "enrollment_db",
        "enroll_policy",
        # TSE-local 无状态注册音增强（不进入 SimulationConfigs）。
        "enrollment_augment",
        # speaker-axis 多目标（省内存版，见 TSEDynamicMixingDataset）
        "targets_per_mixture",
    }
)

_TASK_DATASET_KEYS = {
    "se": frozenset(),
    "aec": AEC_REF_ECHO_DATASET_KEYS,
    "tse": TSE_DATASET_KEYS,
    "ss": frozenset(),
}


@dataclass(frozen=True)
class SimulationProjectionProvenance:
    """一次 flat projection 的来源与输出键审计。"""

    task: str
    source_keys: tuple[str, ...]
    simulation_keys: tuple[str, ...]
    dataset_keys: tuple[str, ...]


@dataclass(frozen=True)
class SimulationConfigProjection:
    """不修改原 mapping 的 simulation schema 与 Dataset kwargs 投影。"""

    simulation: dict
    dataset_kwargs: dict
    provenance: SimulationProjectionProvenance


def _project_simulation_keys(
    raw: Mapping,
    *,
    task: str,
    dataset_keys: frozenset[str],
    omit_none_dataset_keys: frozenset[str] = frozenset(),
) -> SimulationConfigProjection:
    """复制 flat ``simulation_cfg`` 并提升指定 Dataset 构造参数。"""
    if not isinstance(raw, Mapping):
        raise TypeError("simulation_cfg must be a mapping")
    misplaced_lifecycle = sorted(DATASET_LIFECYCLE_KEYS.intersection(raw))
    if misplaced_lifecycle:
        raise ValueError(
            "Dataset lifecycle keys must be configured directly under data split, "
            f"not simulation_cfg: {misplaced_lifecycle}"
        )
    misplaced_resources = sorted(DATA_RESOURCE_KEYS_IN_SPLIT.intersection(raw))
    if misplaced_resources:
        raise ValueError(
            f"Dataset resource keys {misplaced_resources} must be configured "
            "directly under data split, not simulation_cfg"
        )
    if "layer_a" in raw:
        raise ValueError(
            "simulation_cfg.layer_a is no longer supported; move its child keys "
            "to the top level of simulation_cfg"
        )
    simulation = copy.deepcopy(dict(raw))
    source_keys = tuple(sorted(str(key) for key in simulation))

    dataset_kwargs = {}
    for key in sorted(dataset_keys):
        if key not in simulation:
            continue
        value = simulation.pop(key)
        if key in omit_none_dataset_keys and value is None:
            continue
        dataset_kwargs[key] = value
    provenance = SimulationProjectionProvenance(
        task=task,
        source_keys=source_keys,
        simulation_keys=tuple(sorted(str(key) for key in simulation)),
        dataset_keys=tuple(sorted(dataset_kwargs)),
    )
    return SimulationConfigProjection(
        simulation=simulation,
        dataset_kwargs=dataset_kwargs,
        provenance=provenance,
    )


def project_flat_simulation_cfg(
    raw: Mapping, *, task: str
) -> SimulationConfigProjection:
    """将外部 flat ``simulation_cfg`` 非破坏投影为 schema 与 Dataset 两部分。

    这里只做结构投影，不调用 ``SimulationConfigs`` parser，因此 strict unknown-key、
    未知键 warning、默认值与值域校验仍在既有 parser 时点发生。
    """
    if task not in _TASK_DATASET_KEYS:
        raise ValueError(f"simulation config task must be se/aec/tse/ss, got {task!r}")
    from core.data.simulation_schema import normalize_simulation_cfg

    raw = normalize_simulation_cfg(raw)
    projection = _project_simulation_keys(
        raw,
        task=task,
        dataset_keys=_TASK_DATASET_KEYS[task],
    )
    for owner, keys in _TASK_DATASET_KEYS.items():
        if owner == task:
            continue
        foreign = tuple(sorted(keys.intersection(projection.simulation)))
        if foreign:
            raise ValueError(
                f"simulation_cfg Dataset keys {foreign!r} are {owner.upper()}-only "
                f"and cannot be projected for task {task!r}"
            )
    return projection


# ---------------------------------------------------------------------------
# Shared functions
# ---------------------------------------------------------------------------
def deep_merge(base: dict, override: dict) -> None:
    """In-place deep merge: keys in *override* overwrite *base*; nested dicts merge recursively."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def _finite_real(value: object) -> bool:
    return type(value) is not bool and isinstance(value, Real) and math.isfinite(float(value))


def _validate_target_sr(value: object, *, config_path: str) -> Optional[int]:
    if value is None:
        return None
    return require_positive_int(value, config_path)


def validate_layer_a_runtime_config(section: Mapping, *, config_path: str) -> None:
    """校验 Dynamic Dataset 直接消费的 Layer-A 构造参数，不做隐式转换。"""
    for key in (
        "repeat_to_max_duration",
        "repeat_tile_silence_gap",
        "filter_by_duration",
        "export_simulation_stems",
    ):
        if key in section and type(section[key]) is not bool:
            raise ValueError(f"{config_path}.{key} must be a bool, got {section[key]!r}")

    target_sr = section.get("target_sr")
    if target_sr is not None and (type(target_sr) is not int or target_sr <= 0):
        raise ValueError(
            f"{config_path}.target_sr must be an exact int > 0, got {target_sr!r}"
        )
    max_duration = section.get("max_duration")
    if max_duration is not None and type(max_duration) is not int:
        raise ValueError(
            f"{config_path}.max_duration must be an exact int, got {max_duration!r}"
        )
    max_duration_seconds = section.get("max_duration_seconds")
    if max_duration_seconds is not None and (
        not _finite_real(max_duration_seconds) or max_duration_seconds <= 0
    ):
        raise ValueError(
            f"{config_path}.max_duration_seconds must be a finite number > 0, "
            f"got {max_duration_seconds!r}"
        )
    min_duration = section.get("filter_min_duration")
    if min_duration is not None and (
        not _finite_real(min_duration) or min_duration < 0
    ):
        raise ValueError(
            f"{config_path}.filter_min_duration must be a finite number >= 0, "
            f"got {min_duration!r}"
        )
    max_filter_duration = section.get("filter_max_duration")
    if max_filter_duration is not None and not _finite_real(max_filter_duration):
        raise ValueError(
            f"{config_path}.filter_max_duration must be a finite number, "
            f"got {max_filter_duration!r}"
        )
    if (
        section.get("filter_min_duration") is not None
        and section.get("filter_max_duration") is not None
        and section["filter_max_duration"] > 0
        and section["filter_min_duration"] > section["filter_max_duration"]
    ):
        raise ValueError(
            f"{config_path}.filter_min_duration must be <= filter_max_duration"
        )

    gap_range = section.get("repeat_tile_silence_gap_range")
    if gap_range is not None:
        if (
            not isinstance(gap_range, (list, tuple))
            or len(gap_range) != 2
            or any(not _finite_real(value) for value in gap_range)
            or gap_range[0] < 0
            or gap_range[0] > gap_range[1]
        ):
            raise ValueError(
                f"{config_path}.repeat_tile_silence_gap_range must be an ordered "
                f"2-element finite non-negative range, got {gap_range!r}"
            )


def apply_duration_filter_config(section: dict) -> None:
    """Enable duration filtering if filter_min/max_duration is present in *section* (in-place).

    显式 ``filter_by_duration`` 不得与区间键冲突或被静默改写：非 bool 值直接报错，
    显式 ``false`` 与 ``filter_min/max_duration`` 同时出现时报错。``filter_max_duration``
    必须是正数，不使用 ``1e6`` 隐式哨兵取代用户的非正配置。
    """
    if not isinstance(section, dict):
        return
    has_min = "filter_min_duration" in section
    has_max = "filter_max_duration" in section
    if not has_min and not has_max:
        return
    explicit_filter = section.get("filter_by_duration")
    if explicit_filter is not None and type(explicit_filter) is not bool:
        raise ValueError(
            "data section.filter_by_duration must be a bool, "
            f"got {explicit_filter!r}"
        )
    if explicit_filter is False:
        raise ValueError(
            "data section.filter_by_duration=false conflicts with "
            "filter_min_duration/filter_max_duration; remove the duration keys "
            "or set filter_by_duration: true"
        )
    section["filter_by_duration"] = True
    section.setdefault("filter_min_duration", 0.0)
    section.setdefault("filter_max_duration", 1e6)
    validate_layer_a_runtime_config(section, config_path="data section")
    if section["filter_max_duration"] <= 0:
        raise ValueError(
            "data section.filter_max_duration must be > 0, got "
            f"{section['filter_max_duration']!r}; use filter_by_duration: false "
            "to disable duration filtering instead of a sentinel value"
        )


def _apply_simulation_projection(
    section: dict, projection: SimulationConfigProjection
) -> None:
    """将 projection 写回 builder section，保持 public facade in-place。"""
    simulation = section["simulation_cfg"]
    simulation.clear()
    simulation.update(projection.simulation)
    section.update(projection.dataset_kwargs)


def expand_simulation_cfg_layer_a(section: dict) -> None:
    """拒绝把 Dataset lifecycle 键误放入 simulation_cfg。"""
    if not isinstance(section, dict):
        return
    sim = section.get("simulation_cfg")
    if not isinstance(sim, dict):
        raise ValueError("mode=dynamic requires data.train.simulation_cfg (mapping).")
    if "layer_a" in sim:
        raise ValueError(
            "simulation_cfg.layer_a is no longer supported; move its child keys "
            "to the top level of simulation_cfg"
        )
    misplaced = sorted(DATASET_LIFECYCLE_KEYS.intersection(sim))
    if misplaced:
        raise ValueError(
            "Dataset lifecycle keys must be configured directly under "
            f"data split, not simulation_cfg: {misplaced}"
        )


def expand_simulation_cfg_se_pase_dataset_keys(section: dict) -> None:
    """提升 SE 专属 PASE 预处理配置，不让它进入公共 SimulationConfigs。"""
    if not isinstance(section, dict):
        return
    sim = section.get("simulation_cfg")
    if not isinstance(sim, dict):
        return
    projection = _project_simulation_keys(
        sim,
        task="se",
        dataset_keys=SE_PASE_DATASET_KEYS,
    )
    _apply_simulation_projection(section, projection)


def expand_simulation_cfg_aec_ref_echo_dataset_keys(section: dict) -> None:
    """Hoist AEC ref/echo alignment keys from ``simulation_cfg`` onto *section* (in-place)."""
    if not isinstance(section, dict):
        return
    sim = section.get("simulation_cfg")
    if not isinstance(sim, dict):
        return
    projection = _project_simulation_keys(
        sim,
        task="aec",
        dataset_keys=AEC_REF_ECHO_DATASET_KEYS,
    )
    _apply_simulation_projection(section, projection)


def expand_simulation_cfg_tse_dataset_keys(section: dict) -> None:
    """Hoist TSE dataset-constructor keys from ``simulation_cfg`` onto *section* (in-place).

    TSE resource ``spk2enroll_json`` / ``music_scp`` 必须直接位于 split；只有
    ``enrollment_duration`` / ``enrollment_db`` / ``enroll_policy`` /
    ``enrollment_augment`` / ``targets_per_mixture`` 仍从 ``simulation_cfg``
    顶层提升为 Dataset 构造 kwargs。仅 TSE 路径调用。
    """
    if not isinstance(section, dict):
        return
    sim = section.get("simulation_cfg")
    if not isinstance(sim, dict):
        return
    projection = _project_simulation_keys(
        sim,
        task="tse",
        dataset_keys=TSE_DATASET_KEYS,
    )
    _apply_simulation_projection(section, projection)


def _reject_retired_val_section(plane: Mapping) -> None:
    """``val`` split 别名已退役，validation 只接受 ``data.validation``。"""
    if "val" in plane:
        raise ValueError(
            "data.val is no longer supported; configure the validation split as "
            "data.validation"
        )


def data_plane_for_build(cfg: dict) -> dict:
    """Resolve the data plane from a config dict (handles nested ``cfg["data"]``)."""
    outer = cfg.get("data")
    plane = cfg
    if isinstance(outer, dict) and (
        "train" in outer
        or "validation" in outer
        or outer.get("mode") is not None
        or outer.get("sample_rate") is not None
    ):
        plane = outer
    _reject_retired_val_section(plane)
    return plane


def get_train_section(cfg: dict):
    """Return explicit ``data.train`` after validating the required data.mode."""
    plane = data_plane_for_build(cfg)
    normalize_data_mode(plane.get("mode"))
    return plane.get("train")


@dataclass(frozen=True)
class DatasetBuildOptions:
    """所有 Dataset builder 共用的 split 与容错选项。"""

    plane: Mapping
    mode: NormalizedDataMode
    split: str
    section: dict
    target_sr: Optional[int]
    strict_load: bool
    placeholder_num_samples: int


def merge_validation_section(
    train_section: Mapping,
    validation_section: Mapping,
    *,
    data_source_keys: tuple[str, ...] = VALIDATION_DYNAMIC_DATA_SOURCE_KEYS,
) -> dict:
    """按统一 inheritance 规则合并动态 validation，且不修改输入。

    ``inherit=True`` 时只从 validation 回填 ``data_source_keys`` 与 ``simulation_cfg``，
    其余显式 validation 键沿用 train 值；被忽略的键在此记 warning，避免静默丢弃。
    """
    validation = copy.deepcopy(dict(validation_section))
    inherit = validation.pop(
        "inherit_train_simulation",
        validation.pop("inherit_simulation", True),
    )
    if type(inherit) is not bool:
        raise ValueError(
            "data.validation.inherit_simulation must be a bool, "
            f"got {inherit!r}"
        )
    merged = copy.deepcopy(dict(train_section))
    if inherit:
        inherited_only = sorted(
            key
            for key in validation
            if key not in data_source_keys
            and key != "simulation_cfg"
            and key not in VALIDATION_META_KEYS
            and key not in BUILDER_ONLY_SECTION_KEYS
        )
        if inherited_only:
            logger.warning(
                "data.validation keys %s are ignored because validation inherits "
                "the train section; use data.train or set "
                "inherit_train_simulation: false to apply them",
                inherited_only,
            )
        for key in data_source_keys:
            if key in validation:
                merged[key] = copy.deepcopy(validation[key])
        val_sim = validation.get("simulation_cfg")
        if isinstance(val_sim, Mapping):
            base_sim = merged.get("simulation_cfg")
            if not isinstance(base_sim, dict):
                merged["simulation_cfg"] = copy.deepcopy(dict(val_sim))
            else:
                deep_merge(base_sim, dict(val_sim))
        return merged
    for key, value in validation.items():
        if key in VALIDATION_META_KEYS or key == "inherit_train_simulation":
            continue
        if (
            key == "simulation_cfg"
            and isinstance(value, Mapping)
            and isinstance(merged.get(key), dict)
        ):
            deep_merge(merged[key], dict(value))
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_dataset_build_options(
    cfg: Mapping,
    split: str,
    *,
    expected_task: Optional[str] = None,
    inherit_validation: bool = False,
    validation_data_source_keys: tuple[str, ...] = VALIDATION_DYNAMIC_DATA_SOURCE_KEYS,
) -> DatasetBuildOptions:
    """统一解析 mode、train/validation section 与通用 Dataset 选项。"""
    if split not in ("train", "val"):
        raise ValueError(f"dataset split must be 'train' or 'val', got {split!r}")
    plane = data_plane_for_build(cfg)
    normalized = normalize_data_mode(plane.get("mode"))
    _reject_retired_offline_sidecar_keys(plane, normalized)
    _validate_validation_mode_task(plane, expected_task or normalized.task)
    if (
        expected_task is not None
        and normalized.task != expected_task
        and normalized.task_mode not in ("dynamic", "offline")
    ):
        raise ValueError(
            f"data.mode {normalized.task_mode!r} belongs to task "
            f"{normalized.task!r}, not {expected_task!r}"
        )
    if split == "train":
        raw_section = plane.get("train")
    else:
        raw_section = plane.get("validation")
    if not isinstance(raw_section, Mapping) or not raw_section:
        section_name = "train" if split == "train" else "validation"
        raise ValueError(
            f"Missing data.{section_name} config for {normalized.task_mode} dataset"
        )
    section = copy.deepcopy(dict(raw_section))
    if split == "val" and inherit_validation:
        train_section = plane.get("train")
        if isinstance(train_section, Mapping):
            section = merge_validation_section(
                train_section,
                section,
                data_source_keys=validation_data_source_keys,
            )
    resolve_data_section_resource_paths(section)
    strip_builder_only_section_keys(section)
    target_sr = section.pop("target_sr", None)
    target_sr_path = "data.validation.target_sr" if split == "val" else "data.train.target_sr"
    if target_sr is None:
        target_sr = plane.get("sample_rate")
        target_sr_path = "data.sample_rate"
    target_sr = _validate_target_sr(target_sr, config_path=target_sr_path)
    return DatasetBuildOptions(
        plane=plane,
        mode=normalized,
        split=split,
        section=section,
        target_sr=target_sr,
        strict_load=resolve_strict_load(cfg),
        placeholder_num_samples=resolve_placeholder_num_samples(cfg),
    )


def _sampling_is_nonempty(sampling: object) -> bool:
    return not (
        sampling is None or (isinstance(sampling, Mapping) and not sampling)
    )


@dataclass(frozen=True)
class SamplingCapabilities:
    """Per-task sampling controller capability descriptor。

    每个 speech 子块（language/dataset/speaker）与 noise 独立标记；
    capability 检查逐块进行，一个允许的 controller 不掩盖另一个非法的。
    """

    speech_language: bool
    speech_dataset: bool
    speech_speaker: bool
    noise: bool


_DEFAULT_SAMPLING_CAPABILITIES = SamplingCapabilities(
    speech_language=True,
    speech_dataset=True,
    speech_speaker=True,
    noise=True,
)


@dataclass(frozen=True)
class OfflineDatasetRequest:
    """共享 builder 交给 task offline factory 的已解析构造请求。"""

    section: dict
    target_sr: Optional[int]
    strict_load: bool
    simulation_cfg_strict: bool
    placeholder_num_samples: int
    seed: int


def _core_offline_dataset_factory(request: OfflineDatasetRequest):
    """SE/core Adapter：仅构造由成对四列 SCP 支撑的离线数据集。"""
    return OfflineDataset(
        **request.section,
        target_sr=request.target_sr,
        strict_load=request.strict_load,
        simulation_cfg_strict=request.simulation_cfg_strict,
        placeholder_num_samples=request.placeholder_num_samples,
        seed=request.seed,
    )


@dataclass(frozen=True)
class DatasetBuildSpec:
    """任务 Dataset 构造策略，替代继续增加 ``expand_*`` 布尔开关。"""

    task: str
    dynamic_dataset_cls: Callable[..., "Dataset"]
    dynamic_dataset_factory: Optional[
        Callable[["DynamicDatasetBuildRequest"], "Dataset"]
    ] = None
    offline_dataset_factory: Callable[[OfflineDatasetRequest], "Dataset"] = (
        _core_offline_dataset_factory
    )
    simulation_cfg_expanders: tuple[Callable[[dict], None], ...] = (
        expand_simulation_cfg_layer_a,
    )
    simulation_cfg_strict_from_cfg: bool = False
    sampling_capabilities: SamplingCapabilities = _DEFAULT_SAMPLING_CAPABILITIES
    validate_dynamic_train: Optional[Callable[[Mapping], object]] = None
    dynamic_validation_default: bool = False
    validation_data_source_keys: tuple[str, ...] = VALIDATION_DYNAMIC_DATA_SOURCE_KEYS

    def __post_init__(self) -> None:
        if self.task not in ("se", "aec", "tse", "ss"):
            raise ValueError(f"DatasetBuildSpec.task must be se/aec/tse/ss, got {self.task!r}")
        if not callable(self.dynamic_dataset_cls):
            raise TypeError("DatasetBuildSpec.dynamic_dataset_cls must be callable")
        if self.dynamic_dataset_factory is not None and not callable(
            self.dynamic_dataset_factory
        ):
            raise TypeError("DatasetBuildSpec.dynamic_dataset_factory must be callable")
        if not callable(self.offline_dataset_factory):
            raise TypeError("DatasetBuildSpec.offline_dataset_factory must be callable")
        if not self.simulation_cfg_expanders:
            raise ValueError("DatasetBuildSpec requires at least one simulation expander")
        if not all(callable(expander) for expander in self.simulation_cfg_expanders):
            raise TypeError("DatasetBuildSpec.simulation_cfg_expanders must be callables")


def _sampling_config_is_effective(config) -> bool:
    """有效配置 = 任一 speech controller 存在，或 noise.balance_categories 为 True。"""
    speech = config.speech
    if speech is not None and speech.has_controllers:
        return True
    noise = config.noise
    return noise is not None and noise.balance_categories


def _check_sampling_capabilities(config, capabilities: SamplingCapabilities) -> None:
    """逐个子块检查 capability；schema 错误已由 parser 先行抛出。"""
    speech = config.speech
    if speech is not None:
        if speech.language is not None and not capabilities.speech_language:
            raise ValueError(
                "language sampling is not supported for this dataset builder"
            )
        if speech.dataset is not None and not capabilities.speech_dataset:
            raise ValueError(
                "dataset sampling is not supported for this dataset builder"
            )
        if speech.speaker is not None and not capabilities.speech_speaker:
            raise ValueError(
                "speaker sampling is not supported for this dataset builder"
            )
    noise = config.noise
    if noise is not None and noise.balance_categories and not capabilities.noise:
        raise ValueError("sampling.noise is not supported for TSE")


def _resolve_effective_sampling(
    raw_sampling,
    *,
    capabilities: SamplingCapabilities,
    mode: str,
    is_validation: bool,
    explicitly_configured: bool,
):
    """统一 sampling 解析与 capability / validation 语义。

    - schema 错误（非 mapping、未知键、非法数值）由 parser 直接抛出；
    - train：capability 逐块检查（先于 offline 的 dynamic-only 判定），
      effective-empty（``speech: {}`` / disabled noise / ``{}``）归一为
      ``None``；offline 出现任一有效配置报 ``dynamic-only``；
    - validation：只对原始 section 的**显式**值生效（调用方须在 merge 前
      传入）；显式有效配置报 validation capability error，显式
      effective-empty 与缺失/继承值一律返回 ``None``（由调用方剥离）。
    """
    from core.data.sampling import SamplingConfig

    if is_validation:
        if not explicitly_configured or not _sampling_is_nonempty(raw_sampling):
            return None
        config = SamplingConfig.from_dict(raw_sampling)
        if _sampling_config_is_effective(config):
            raise ValueError(
                "validation sampling is not supported; configure sampling only under data.train"
            )
        return None
    if not _sampling_is_nonempty(raw_sampling):
        return None
    config = SamplingConfig.from_dict(raw_sampling)
    _check_sampling_capabilities(config, capabilities)
    if not _sampling_config_is_effective(config):
        return None
    if mode == "offline":
        raise ValueError(
            "sampling is dynamic-only and cannot be used with offline training"
        )
    return config


def validate_tse_resource_path(value: object, *, config_path: str) -> str:
    """校验 TSE 配置层资源文件路径；不处理 enrollment JSON 内的音频路径。"""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{config_path} must be a non-empty trimmed path string, got {value!r}"
        )
    return value


def resolve_tse_train_speech_scp(train_section: Mapping) -> str:
    """校验必需的 train manifest，并按完整字段路径 fail-fast。"""
    if "speech_scp" not in train_section:
        raise ValueError(
            "data.train.speech_scp must be a non-empty trimmed path string, got missing"
        )
    return validate_tse_resource_path(
        train_section["speech_scp"],
        config_path="data.train.speech_scp",
    )


def resolve_tse_validation_speech_scp(
    validation_section: Mapping,
    train_section: Mapping,
) -> object:
    """解析 validation manifest；仅 key 缺失时继承已校验的 train 值。"""
    train_speech = resolve_tse_train_speech_scp(train_section)
    if "speech_scp" not in validation_section:
        return train_speech
    return validate_tse_resource_path(
        validation_section["speech_scp"],
        config_path="data.validation.speech_scp",
    )


def _tse_resource_from_sections(
    key: str,
    validation_section: Mapping,
    train_section: Mapping,
    *,
    require_validation_local: bool,
    location: str,
    train_location: str,
) -> tuple[object, str]:
    if key in validation_section:
        return validation_section[key], f"{location}.{key}"
    if require_validation_local:
        raise ValueError(
            f"{location}.{key} must be explicitly provided "
            "as a validation-local non-empty trimmed path string"
        )
    if key in train_section:
        return train_section[key], f"{train_location}.{key}"
    raise ValueError(
        f"{location}.{key} requires an effective non-empty "
        "trimmed path string (explicit in validation or inherited from train)"
    )


def validate_tse_train_resource_paths(train_section: Mapping) -> bool:
    """在 entry 与 Dataset builder 边界校验 strict TSE train 资源。"""
    from core.data.simulation_utils import is_strict_jsonl_manifest

    speech_scp = resolve_tse_train_speech_scp(train_section)
    if not is_strict_jsonl_manifest(speech_scp):
        raise ValueError(
            "data.train.speech_scp must be a strict JSONL manifest for TSE dynamic "
            "training; SCP and JSON manifests are not supported"
        )
    if "spk2enroll_json" not in train_section:
        raise ValueError(
            "data.train.spk2enroll_json must be a non-empty trimmed "
            "path string"
        )
    validate_tse_resource_path(
        train_section["spk2enroll_json"],
        config_path="data.train.spk2enroll_json",
    )
    return True


def validate_tse_validation_resource_provenance(
    validation_section: Mapping,
    train_section: Mapping,
) -> bool:
    """校验 strict TSE validation manifest 与有效资源继承。"""
    from core.data.simulation_utils import is_strict_jsonl_manifest

    if (
        "simulation_cfg" in validation_section
        and validation_section["simulation_cfg"] is not None
        and not isinstance(validation_section["simulation_cfg"], Mapping)
    ):
        raise ValueError("data.validation.simulation_cfg must be a mapping")

    effective_speech = resolve_tse_validation_speech_scp(
        validation_section, train_section
    )
    if not is_strict_jsonl_manifest(effective_speech):
        raise ValueError(
            "data.validation.speech_scp must be a strict JSONL manifest for TSE "
            "dynamic validation; SCP and JSON manifests are not supported"
        )
    inherit = validation_section.get("inherit_simulation", True)
    require_local = inherit is False
    enrollment, enrollment_path = _tse_resource_from_sections(
        "spk2enroll_json",
        validation_section,
        train_section,
        require_validation_local=require_local,
        location="data.validation",
        train_location="data.train",
    )
    validate_tse_resource_path(enrollment, config_path=enrollment_path)
    return True


def _apply_tse_validation_resource_provenance(
    validation_section: Mapping,
    train_section: Mapping,
) -> None:
    """校验 validation provenance。"""
    validate_tse_validation_resource_provenance(validation_section, train_section)


def _validate_validation_mode_task(plane: Mapping, expected_task: str) -> None:
    """拒绝 validation 使用其他任务的显式 task-suffixed mode。"""
    validation = plane.get("validation")
    if not isinstance(validation, Mapping) or validation.get("mode") is None:
        return
    normalized = normalize_data_mode(validation.get("mode"))
    if normalized.task_mode not in ("dynamic", "offline") and normalized.task != expected_task:
        raise ValueError(
            f"data.validation.mode {normalized.task_mode!r} belongs to task "
            f"{normalized.task!r}, not {expected_task!r}"
        )


def _reject_retired_offline_sidecar_keys(
    plane: Mapping, normalized_mode: NormalizedDataMode
) -> None:
    """仅在 SE/AEC offline split 拒绝退役 exact key 与采样率 alias。"""
    if normalized_mode.task not in ("se", "aec"):
        return

    train_section = plane.get("train")
    validation = plane.get("validation")
    all_sections = [("train", train_section), ("validation", validation)]
    for section_name, section in all_sections:
        if not isinstance(section, Mapping):
            continue
        for key in ("clean_list", "noisy_list", "clean_json", "noisy_json", "list_file_sample_rate"):
            if key in section:
                raise ValueError(
                    f"data.{section_name}.{key} is no longer supported; rebuild "
                    "paired clean/noisy indexes as four-column 'uid fs length path' "
                    "SCPs with prepare_offline.py"
                )

    if normalized_mode.storage != "offline":
        return

    for section_name, section in all_sections:
        if not isinstance(section, Mapping):
            continue
        for key in ("utt2fs", "speech_length"):
            if key in section:
                raise ValueError(
                    f"data.{section_name}.{key} is no longer supported; rebuild "
                    "clean/noisy indexes as four-column 'uid fs length path' SCPs "
                    "with prepare_scp.py"
                )
        if "sample_rate" in section:
            raise ValueError(
                f"data.{section_name}.sample_rate is no longer supported; use "
                "top-level data.sample_rate for target SR"
            )


def _reject_removed_noise_manifest_option(value, path: str = "data") -> None:
    """旧 runtime admission 逃生键已删除，任何配置位置都必须 fail-fast。"""
    if isinstance(value, Mapping):
        if "allow_unverified_noise_manifest" in value:
            raise ValueError(
                f"{path}.allow_unverified_noise_manifest is no longer supported; "
                "noise training runtime does not enforce strict manifest admission"
            )
        for key, item in value.items():
            _reject_removed_noise_manifest_option(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_removed_noise_manifest_option(item, f"{path}[{index}]")


def _dataset_seed(cfg: Mapping, meta: Optional[Mapping]) -> int:
    """按既有优先级解析 Dataset 样本内 seed。"""
    if isinstance(meta, Mapping):
        return int(meta.get("seed", 0))
    if isinstance(cfg.get("meta"), Mapping):
        return int(cfg["meta"].get("seed", 0))
    return int(cfg.get("seed", 0) or 0)


@dataclass(frozen=True)
class DynamicDatasetBuildRequest:
    """core 已解析的动态 Dataset 构造请求，供任务 Adapter 补充专属 kwargs。"""

    split: str
    dataset_kwargs: dict[str, Any]
    seed: int


def _dynamic_validation_requested(
    *,
    spec: DatasetBuildSpec,
    storage: str,
    raw_validation: Mapping,
) -> bool:
    if storage != "dynamic":
        return False
    if spec.dynamic_validation_default:
        return True
    validation_mode = raw_validation.get("mode")
    validation_storage = (
        normalize_data_mode(validation_mode).storage
        if validation_mode is not None
        else None
    )
    return validation_storage == "dynamic" or raw_validation.get("use_dynamic") is True


def resolve_dynamic_dataset_build_request(
    cfg: Mapping,
    split: str,
    *,
    meta: Optional[Mapping] = None,
    spec: DatasetBuildSpec,
) -> DynamicDatasetBuildRequest:
    """解析动态 Dataset 共有生命周期，尚不构造 Dataset 实例。

    任务 Adapter 只能在返回的 ``dataset_kwargs`` 上追加其专属 manifest、退役键、
    state/STFT 或 schema validator 差异；mode、split、validation 继承、资源路径、
    simulation 投影、通用 runtime 字段与 seed 均由 core 统一持有。
    """
    _reject_removed_noise_manifest_option(cfg)
    options = resolve_dataset_build_options(
        cfg,
        split,
        expected_task=spec.task,
        inherit_validation=split == "val",
        validation_data_source_keys=spec.validation_data_source_keys,
    )
    if options.mode.storage != "dynamic":
        raise ValueError(
            f"data.mode {options.mode.task_mode!r} is not a dynamic {spec.task.upper()} mode"
        )

    plane = options.plane
    raw_train = plane.get("train")
    if not isinstance(raw_train, Mapping) or not raw_train:
        raise ValueError("Missing data.train config for dynamic dataset")
    raw_validation = plane.get("validation") or {} if split == "val" else {}
    if split == "val" and not isinstance(raw_validation, Mapping):
        raise ValueError("Missing data.validation config for dataset split 'val'")
    if split == "val" and not _dynamic_validation_requested(
        spec=spec,
        storage=options.mode.storage,
        raw_validation=raw_validation,
    ):
        raise ValueError(
            f"data.validation does not request a dynamic {spec.task.upper()} dataset"
        )

    if split == "val":
        _resolve_effective_sampling(
            raw_validation.get("sampling"),
            capabilities=spec.sampling_capabilities,
            mode="dynamic",
            is_validation=True,
            explicitly_configured="sampling" in raw_validation,
        )
        validate_simulation_config_contract(
            task=spec.task,
            split="val",
            train_section=raw_train,
            validation_section=raw_validation,
        )
    else:
        validate_simulation_config_contract(
            task=spec.task,
            split="train",
            train_section=raw_train,
        )

    section = options.section
    # 直接调用 dataset builder 的测试/工具路径也必须走同一份分层配置
    # 归一化，不能依赖 trainer entry 恰好先执行过一次。
    if isinstance(section.get("simulation_cfg"), Mapping):
        from core.data.simulation_schema import normalize_simulation_cfg

        section["simulation_cfg"] = normalize_simulation_cfg(section["simulation_cfg"])
    if split == "val":
        for key in OFFLINE_ONLY_KEYS:
            section.pop(key, None)
    sampling_cfg = None
    if split == "train":
        sampling_cfg = _resolve_effective_sampling(
            raw_train.get("sampling"),
            capabilities=spec.sampling_capabilities,
            mode="dynamic",
            is_validation=False,
            explicitly_configured="sampling" in raw_train,
        )
        if spec.validate_dynamic_train is not None:
            spec.validate_dynamic_train(section)
    elif spec.task == "tse":
        _apply_tse_validation_resource_provenance(raw_validation, raw_train)

    apply_duration_filter_config(section)
    for expander in spec.simulation_cfg_expanders:
        expander(section)
    validate_simulation_config_contract(
        task=spec.task,
        split="train",
        train_section={"simulation_cfg": section.get("simulation_cfg", {})},
    )
    if options.target_sr is not None and "target_sr" not in section:
        section["target_sr"] = options.target_sr
    if (section.get("simulation_cfg") or {}).get("rir_backend") == "online":
        section.setdefault("rir_scp", None)
    validate_layer_a_runtime_config(section, config_path=f"data.{split}")
    strip_builder_only_section_keys(section)

    dataset_kwargs: dict[str, Any] = {
        **section,
        "strict_load": options.strict_load,
        "placeholder_num_samples": options.placeholder_num_samples,
    }
    if spec.simulation_cfg_strict_from_cfg:
        dataset_kwargs["simulation_cfg_strict"] = resolve_simulation_cfg_strict(cfg)
    if sampling_cfg is not None:
        dataset_kwargs["sampling_cfg"] = sampling_cfg
    return DynamicDatasetBuildRequest(
        split=split,
        dataset_kwargs=dataset_kwargs,
        seed=_dataset_seed(cfg, meta),
    )


def instantiate_dynamic_dataset(
    spec: DatasetBuildSpec,
    dataset_kwargs: Mapping[str, Any],
    *,
    request: Optional[DynamicDatasetBuildRequest] = None,
    validator: Optional[Callable[..., object]] = None,
):
    """经可选 task schema validator 后实例化动态 Dataset。"""
    kwargs = dict(dataset_kwargs)
    if validator is not None:
        validator(**kwargs)
    if spec.dynamic_dataset_factory is not None:
        if request is None:
            raise ValueError(
                "dynamic_dataset_factory requires the resolved build request"
            )
        return spec.dynamic_dataset_factory(request)
    return spec.dynamic_dataset_cls(**kwargs)


def _build_offline_from_spec(
    *,
    spec: DatasetBuildSpec,
    section: dict,
    target_sr: object,
    strict_load: bool,
    simulation_cfg_strict: bool,
    placeholder_num_samples: int,
    seed: int,
):
    """只共享 lifecycle 与 strict simulation request；task factory 保有 manifest/kwargs ownership。"""
    return spec.offline_dataset_factory(
        OfflineDatasetRequest(
            section=section,
            target_sr=(int(target_sr) if target_sr is not None else None),
            strict_load=strict_load,
            simulation_cfg_strict=simulation_cfg_strict,
            placeholder_num_samples=placeholder_num_samples,
            seed=seed,
        )
    )


# ---------------------------------------------------------------------------
# Shared build_dataset body
# ---------------------------------------------------------------------------
def build_dataset(
    cfg: dict,
    mode: str = "train",
    meta: Optional[dict] = None,
    *,
    spec: DatasetBuildSpec,
) -> "Dataset":
    """按显式 ``DatasetBuildSpec`` 构建 Dataset。"""
    _reject_removed_noise_manifest_option(cfg)
    if mode not in ("train", "val"):
        raise ValueError(f"dataset split must be 'train' or 'val', got {mode!r}")
    plane = data_plane_for_build(cfg)
    normalized_mode = normalize_data_mode(plane.get("mode"))
    _reject_retired_offline_sidecar_keys(plane, normalized_mode)
    _validate_validation_mode_task(plane, spec.task)
    if (
        normalized_mode.task != spec.task
        and normalized_mode.task_mode not in ("dynamic", "offline")
    ):
        raise ValueError(
            f"data.mode {normalized_mode.task_mode!r} belongs to task "
            f"{normalized_mode.task!r}, not {spec.task!r}"
        )
    storage = normalized_mode.storage
    capabilities = spec.sampling_capabilities
    target_sr = _validate_target_sr(
        plane.get("sample_rate", cfg.get("sample_rate", None)),
        config_path="data.sample_rate",
    )
    strict_load = resolve_strict_load(cfg)
    simulation_cfg_strict = (
        resolve_simulation_cfg_strict(cfg)
        if spec.simulation_cfg_strict_from_cfg
        else True
    )
    ph = resolve_placeholder_num_samples(cfg)

    if mode == "train" and storage == "dynamic":
        request = resolve_dynamic_dataset_build_request(cfg, "train", meta=meta, spec=spec)
        train_ds = instantiate_dynamic_dataset(
            spec, request.dataset_kwargs, request=request
        )
        train_ds._simulation_split = "train"
        return train_ds

    if mode == "val":
        raw_validation_section = plane.get("validation")
        if not raw_validation_section:
            raise ValueError("Missing data.validation config for dataset split 'val'")
        if _dynamic_validation_requested(
            spec=spec,
            storage=storage,
            raw_validation=raw_validation_section,
        ):
            request = resolve_dynamic_dataset_build_request(
                cfg, "val", meta=meta, spec=spec
            )
            val_ds = instantiate_dynamic_dataset(
                spec, request.dataset_kwargs, request=request
            )
            val_ds._simulation_split = "validation"
            if hasattr(val_ds, "set_epoch"):
                val_ds.set_epoch(0, base_seed=request.seed)
            return val_ds

        section = copy.deepcopy(dict(raw_validation_section))
        resolve_data_section_resource_paths(section)
        for key in ("use_dynamic", "inherit_simulation", "inherit_train_simulation"):
            if key in section and type(section[key]) is not bool:
                raise ValueError(
                    f"data.validation.{key} must be a bool, got {section[key]!r}"
                )
        _resolve_effective_sampling(
            section.get("sampling"),
            capabilities=capabilities,
            mode=storage,
            is_validation=True,
            explicitly_configured="sampling" in section,
        )
        apply_duration_filter_config(section)
        validate_layer_a_runtime_config(section, config_path="data.validation")
        val_target_sr = _validate_target_sr(
            section.pop("target_sr", target_sr),
            config_path="data.validation.target_sr",
        )
        strip_builder_only_section_keys(section)
        return _build_offline_from_spec(
            spec=spec,
            section=section,
            target_sr=val_target_sr,
            strict_load=strict_load,
            simulation_cfg_strict=simulation_cfg_strict,
            placeholder_num_samples=ph,
            seed=_dataset_seed(cfg, meta),
        )

    section = get_train_section(cfg)
    if not section:
        raise ValueError("Missing data.train config for offline dataset")
    section = copy.deepcopy(dict(section))
    resolve_data_section_resource_paths(section)
    _resolve_effective_sampling(
        section.pop("sampling", None),
        capabilities=capabilities,
        mode="offline",
        is_validation=False,
        explicitly_configured=True,
    )
    apply_duration_filter_config(section)
    validate_layer_a_runtime_config(section, config_path="data.train")
    train_target_sr = _validate_target_sr(
        section.pop("target_sr", target_sr),
        config_path="data.train.target_sr",
    )
    strip_builder_only_section_keys(section)
    return _build_offline_from_spec(
        spec=spec,
        section=section,
        target_sr=train_target_sr,
        strict_load=strict_load,
        simulation_cfg_strict=simulation_cfg_strict,
        placeholder_num_samples=ph,
        seed=_dataset_seed(cfg, meta),
    )
