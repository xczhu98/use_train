"""Audio Filter Pipeline configuration."""
from __future__ import annotations

import copy
import hashlib
import hmac
import logging
import math
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import List, Dict, Optional, Union, Any, Literal, Mapping, Tuple

from ..processors.mos_contract import (
    MOS_MODEL_ORDER,
    MOS_SHIPPED_MODELS,
    MOS_THRESHOLD_DIMENSIONS,
)
from ..processors.optional_runtime import OptionalRuntimeConfigError, WorkerLimits
from .selection_policy import (
    QUALITY_PROFILES,
    MosScoringProjection,
    ResolvedSelectionPolicy,
    SelectionPolicyBook,
    SelectionRoute,
    canonical_policy_value as _canonical_policy_value,
    freeze_snapshot_value as _freeze_snapshot_value,
    normalize_language as _normalize_policy_language,
)
from ..utils.export_roots import validate_export_root_boundaries


class ValidationContext(str, Enum):
    """配置校验阶段；保留旧上下文名称并显式提供 generation-3 Step0–Step6 上下文。"""

    STRUCTURAL = "structural"
    STEP1_EXECUTION = "step1_execution"
    STEP2_EXECUTION = "step2_execution"
    STEP3_EXECUTION = "step3_execution"
    V3_STEP0_EXECUTION = "v3_step0_execution"
    V3_STEP1_EXECUTION = "v3_step1_execution"
    V3_STEP2_EXECUTION = "v3_step2_execution"
    V3_STEP3_EXECUTION = "v3_step3_execution"
    V3_STEP4_EXECUTION = "v3_step4_execution"
    V3_STEP5_EXECUTION = "v3_step5_execution"
    V3_STEP6_EXECUTION = "v3_step6_execution"


def _validate_pipeline_generation(pipeline_generation: Any) -> int:
    """校验并返回受支持的管线 generation（当前仅支持 generation-3）。"""
    if type(pipeline_generation) is not int or pipeline_generation != 3:
        raise ValueError(
            f"pipeline_generation must be 3, got {pipeline_generation!r}"
        )
    return pipeline_generation


def validation_context_for_step(
    step: Optional[str], pipeline_generation: int = 3,
) -> ValidationContext:
    """返回 generation-3 对应的 step 校验上下文。"""
    pipeline_generation = _validate_pipeline_generation(pipeline_generation)
    return {
        "step0": ValidationContext.V3_STEP0_EXECUTION,
        "step1": ValidationContext.V3_STEP1_EXECUTION,
        "step2": ValidationContext.V3_STEP2_EXECUTION,
        "step3": ValidationContext.V3_STEP3_EXECUTION,
        "step4": ValidationContext.V3_STEP4_EXECUTION,
        "step5": ValidationContext.V3_STEP5_EXECUTION,
        "step6": ValidationContext.V3_STEP6_EXECUTION,
        "all": ValidationContext.V3_STEP6_EXECUTION,
    }.get(step, ValidationContext.STRUCTURAL)


def _validate_export_root_boundaries(
    output_root: Optional[str],
    dataset_specs: Mapping[str, "DatasetSpec"],
    export_roots: Mapping[str, Optional[str]],
) -> None:
    """兼容旧私有入口；实际规则由共用纯函数提供。"""
    validate_export_root_boundaries(
        output_root,
        (
            (f"dataset {dataset_id!r} input_root", spec.input_root)
            for dataset_id, spec in dataset_specs.items()
        ),
        export_roots,
    )


SEMANTIC_DATASET_CATEGORIES = frozenset({
    "read_speech", "expressive", "elderly", "singing", "game_voice",
})
DATASET_SOURCE_LAYOUTS = frozenset({"regular_file", "streaming_flac_aggregate"})
# quality_profiles 的构造校验只需要稳定的 selection 元数据，不应为此加载
# processors.mos 及其 torch/core 运行时依赖。
@dataclass(frozen=True)
class _SelectionMosSpec:
    selectable: bool
    validate_threshold: Optional[Tuple[str, ...]] = None


_SELECTION_MOS_SPECS = MappingProxyType({
    model: _SelectionMosSpec(
        selectable=True,
        validate_threshold=MOS_THRESHOLD_DIMENSIONS[model] or None,
    )
    for model in MOS_MODEL_ORDER
})


def _get_selection_mos_spec(name: str) -> _SelectionMosSpec:
    """返回 selection policy 所需的轻量 MOS 元数据。"""
    normalized = str(name).strip().lower()
    try:
        return _SELECTION_MOS_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unknown MOS type: {name}. Available: {sorted(_SELECTION_MOS_SPECS)}"
        ) from exc


_V3_LEGACY_DEFAULT_ATTRIBUTES = {
    "mos_types": list(MOS_SHIPPED_MODELS),
    "urgent_pk_checkpoint": None,
    "urgent_pk_source_dir": None,
    "urgent_pk_encoder": None,
    "urgent_pk_backbone": None,
    "urgent_pk_score_range": None,
    "urgent_pk_expected_sha256": None,
    "urgent_pk_expected_commit": None,
    "scoreq_model_dir": None,
    "dnsmos_model_dir": None,
    "nisqa_model_path": None,
    "nisqa_model_version": None,
    "nisqa_model_sha256": None,
    "utmos_model_path": None,
    "utmos_model_version": None,
    "utmos_model_sha256": None,
    "step2_direct_from_input": False,
    "step2_mos_early_exit_threshold": None,
    "mos_threshold": 3.0,
    "mos_thresholds": {},
    "mos_fusion": "average",
    "enable_multi_mos": True,
    "min_mos_models": 3,
    "enable_signal_integrity_check": True,
    "signal_integrity_max_zero_ratio": 0.99,
    "signal_integrity_min_rms_dbfs": -60.0,
    "signal_integrity_max_dc_offset": 0.5,
    "signal_integrity_clip_threshold": 1.0,
    "dataset_name": None,
}

_V3_SELECTION_POLICY_DEFAULTS = MappingProxyType({
    "mos_models": tuple(_V3_LEGACY_DEFAULT_ATTRIBUTES["mos_types"]),
    "thresholds": MappingProxyType({}),
    "min_mos_models": _V3_LEGACY_DEFAULT_ATTRIBUTES["min_mos_models"],
    "mos_fusion": _V3_LEGACY_DEFAULT_ATTRIBUTES["mos_fusion"],
    "early_exit": None,
})
_UNSET_LEGACY_SELECTION_ARGUMENT = object()
# Step1 native TEN VAD 进程池工作进程数上限。VAD 本身极快（RTF ~0.009），
# 更多 worker 主要换解码/IO 并发；主机 124 核且训练仅占 ~30% CPU，
# 上限按主机核数放开到 64（每 worker 已钉死单线程，避免超订）。
STEP1_VAD_MAX_WORKERS = 64


@dataclass(frozen=True)
class SpeakerMetadataSpec:
    """Speaker resolver metadata；``mapping_path`` 是 B2 解析的 opaque 字符串。

    PipelineConfig 不读取 mapping 文件，也不解析相对路径；B2 必须基于配置加载
    上下文解析路径，并负责文件完整性与内容 digest。
    """

    source: Literal["path_component", "regex", "mapping_file", "none"]
    path_component: Optional[int] = None
    regex: Optional[str] = None
    mapping_path: Optional[str] = None
    regex_group: str = "speaker_id"
    required: bool = True


@dataclass(frozen=True)
class SourceSelectorSpec:
    sample_rates_hz: Tuple[int, ...] = ()
    include_speaker_ids: Tuple[str, ...] = ()
    exclude_relative_paths: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    input_root: str
    language: str
    category: Optional[str] = None
    quality_profile: Optional[str] = field(default=None, kw_only=True)
    speaker: SpeakerMetadataSpec = SpeakerMetadataSpec(
        source="none", required=False,
    )
    source_selector: SourceSelectorSpec = field(
        default=SourceSelectorSpec(), kw_only=True,
    )
    source_layout: Literal["regular_file", "streaming_flac_aggregate"] = field(
        default="regular_file", kw_only=True,
    )

    def __post_init__(self) -> None:
        if type(self.source_layout) is not str or self.source_layout not in DATASET_SOURCE_LAYOUTS:
            raise ValueError(
                "source_layout must be one of "
                f"{sorted(DATASET_SOURCE_LAYOUTS)!r}, got {self.source_layout!r}"
            )


class PipelineConfig:
    """Generation-3 的 Step0–Step6 音频质量筛选配置。

    构造函数只保留 legacy selection 的 Python 签名与属性形状兼容；显式传入
    旧 selection 参数会 fail-closed，generation-3 必须使用
    ``selection_policies``。其余 generation-3 字段由各 Step 的显式校验与
    不可变快照消费。
    """

    @property
    def selection_policies(self) -> Mapping[str, Any]:
        return self._selection_policies_snapshot

    @selection_policies.setter
    def selection_policies(self, value: Mapping[str, Any]) -> None:
        raise AttributeError(
            "selection_policies is read-only for pipeline generation 3"
        )

    @property
    def speaker_consistency(self) -> Mapping[str, Any]:
        return self._speaker_consistency_snapshot

    @speaker_consistency.setter
    def speaker_consistency(self, value: Mapping[str, Any]) -> None:
        raise AttributeError(
            "speaker_consistency is read-only for pipeline generation 3"
        )

    @property
    def min_source_sample_rate_hz(self) -> Optional[int]:
        """返回构造时固定的 source sample-rate floor。"""
        return self._min_source_sample_rate_hz

    @property
    def step6_mode(self) -> str:
        """Generation-3 Step6 的唯一最终筛选执行策略。"""
        return self._step6_mode

    @property
    def step4_mode(self) -> str:
        """Generation-3 Step4 MOS 筛选执行策略：dry_run（仅统计）或 filter（发布后导出）。"""
        return self._step4_mode

    @staticmethod
    def _canonical_speaker_consistency(raw: Any) -> Mapping[str, Any]:
        """构造并冻结 generation-3 Step5/Step6 speaker consistency snapshot。"""
        if raw is None:
            return _freeze_snapshot_value({"enabled": False})
        if not isinstance(raw, Mapping):
            raise ValueError("speaker_consistency must be a mapping")
        allowed_fields = {
            "enabled", "runtime", "model_dir", "model_version", "model_sha256", "runtime_version",
            "target_sample_rate", "enrollment_bundle_path", "enrollment_bundle_sha256",
            "enrollment_exclusion_path", "enrollment_exclusion_sha256",
            "min_cosine_similarity", "gpu_ids", "worker_timeout_seconds",
            "worker_stderr_tail_bytes", "worker_limits",
        }
        invalid_keys = [key for key in raw if type(key) is not str or not key.strip()]
        if invalid_keys:
            raise ValueError(
                "speaker_consistency field names must be non-empty strings, "
                f"got {invalid_keys!r}"
            )
        unknown_fields = set(raw) - allowed_fields
        if unknown_fields:
            raise ValueError(
                "speaker_consistency has unknown fields: "
                f"{sorted(unknown_fields)!r}"
            )
        enabled = raw.get("enabled", False)
        if type(enabled) is not bool:
            raise ValueError("speaker_consistency.enabled must be bool")
        canonical = {"enabled": enabled}
        if not enabled:
            return _freeze_snapshot_value(canonical)

        required_fields = (
            "runtime", "model_dir", "model_version", "model_sha256", "runtime_version",
            "target_sample_rate", "enrollment_bundle_path", "enrollment_bundle_sha256",
            "enrollment_exclusion_path", "enrollment_exclusion_sha256",
        )
        missing = [field for field in required_fields if field not in raw]
        if missing:
            raise ValueError(
                "speaker_consistency enabled requires fields: "
                f"{missing!r}"
            )
        if raw["runtime"] != "wespeaker":
            raise ValueError(
                "speaker_consistency.runtime must be exactly 'wespeaker' when enabled"
            )
        model_dir = raw["model_dir"]
        if not isinstance(model_dir, str) or not model_dir.strip():
            raise ValueError("speaker_consistency.model_dir must be a non-empty string")
        model_dir = os.path.normpath(model_dir)
        if not os.path.isabs(model_dir):
            raise ValueError("speaker_consistency.model_dir must be an absolute path")
        model_version = raw["model_version"]
        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError(
                "speaker_consistency.model_version must be a non-empty string"
            )
        model_sha256 = raw["model_sha256"]
        if not isinstance(model_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", model_sha256
        ):
            raise ValueError(
                "speaker_consistency.model_sha256 must be lowercase 64-character hex"
            )
        runtime_version = raw["runtime_version"]
        if not isinstance(runtime_version, str) or not runtime_version.strip():
            raise ValueError(
                "speaker_consistency.runtime_version must be a non-empty string"
            )
        target_sample_rate = raw["target_sample_rate"]
        if type(target_sample_rate) is not int or target_sample_rate <= 0:
            raise ValueError(
                "speaker_consistency.target_sample_rate must be a positive integer"
            )
        enrollment_bundle_path = raw["enrollment_bundle_path"]
        if not isinstance(enrollment_bundle_path, str) or not enrollment_bundle_path.strip():
            raise ValueError("speaker_consistency.enrollment_bundle_path must be a non-empty string")
        enrollment_bundle_path = os.path.normpath(enrollment_bundle_path)
        if not os.path.isabs(enrollment_bundle_path):
            raise ValueError("speaker_consistency.enrollment_bundle_path must be an absolute path")
        if not enrollment_bundle_path.endswith(".jsonl"):
            raise ValueError("speaker_consistency.enrollment_bundle_path must name a .jsonl bundle")
        enrollment_bundle_sha256 = raw["enrollment_bundle_sha256"]
        if not isinstance(enrollment_bundle_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", enrollment_bundle_sha256
        ):
            raise ValueError(
                "speaker_consistency.enrollment_bundle_sha256 must be lowercase 64-character hex"
            )
        enrollment_exclusion_path = raw["enrollment_exclusion_path"]
        if not isinstance(enrollment_exclusion_path, str) or not enrollment_exclusion_path.strip():
            raise ValueError("speaker_consistency.enrollment_exclusion_path must be a non-empty string")
        enrollment_exclusion_path = os.path.normpath(enrollment_exclusion_path)
        if not os.path.isabs(enrollment_exclusion_path) or not enrollment_exclusion_path.endswith(".jsonl"):
            raise ValueError("speaker_consistency.enrollment_exclusion_path must be an absolute .jsonl path")
        enrollment_exclusion_sha256 = raw["enrollment_exclusion_sha256"]
        if not isinstance(enrollment_exclusion_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", enrollment_exclusion_sha256
        ):
            raise ValueError(
                "speaker_consistency.enrollment_exclusion_sha256 must be lowercase 64-character hex"
            )
        gpu_ids = raw.get("gpu_ids", [])
        if (not isinstance(gpu_ids, list) or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in gpu_ids)
                or len(set(gpu_ids)) != len(gpu_ids)):
            raise ValueError("speaker_consistency.gpu_ids must be a unique list of nonnegative integers")
        canonical.update({
            "runtime": "wespeaker",
            "model_dir": model_dir,
            "model_version": model_version.strip(),
            "model_sha256": model_sha256,
            "runtime_version": runtime_version.strip(),
            "target_sample_rate": target_sample_rate,
            "enrollment_bundle_path": enrollment_bundle_path,
            "enrollment_bundle_sha256": enrollment_bundle_sha256,
            "enrollment_exclusion_path": enrollment_exclusion_path,
            "enrollment_exclusion_sha256": enrollment_exclusion_sha256,
        })
        if gpu_ids:
            canonical["gpu_ids"] = list(gpu_ids)
        worker_timeout_seconds = raw.get("worker_timeout_seconds", 900.0)
        if (isinstance(worker_timeout_seconds, bool)
                or not isinstance(worker_timeout_seconds, (int, float))
                or not math.isfinite(float(worker_timeout_seconds))
                or not 0.0 < float(worker_timeout_seconds) <= 3_600.0):
            raise ValueError(
                "speaker_consistency.worker_timeout_seconds must be finite and in (0, 3600]"
            )
        worker_stderr_tail_bytes = raw.get("worker_stderr_tail_bytes", 4_096)
        if (type(worker_stderr_tail_bytes) is not int
                or not 0 < worker_stderr_tail_bytes <= 65_536):
            raise ValueError(
                "speaker_consistency.worker_stderr_tail_bytes must be an integer in [1, 65536]"
            )
        default_limits = WorkerLimits(
            timeout_ms=max(1, math.ceil(float(worker_timeout_seconds) * 1000.0)),
            terminate_grace_ms=500,
            kill_wait_ms=500,
            rlimit_cpu_seconds=60,
            rlimit_nofile=256,
            rlimit_fsize_bytes=128 * 1024 * 1024,
            rlimit_as_bytes=32 * 1024**3,
            max_stderr_bytes=worker_stderr_tail_bytes,
        )
        worker_limit_fields = tuple(default_limits.__dataclass_fields__)
        worker_limits_raw = raw.get("worker_limits", {})
        if not isinstance(worker_limits_raw, Mapping):
            raise ValueError("speaker_consistency.worker_limits must be a mapping")
        if any(type(field) is not str for field in worker_limits_raw):
            raise ValueError(
                "speaker_consistency.worker_limits field names must be strings"
            )
        unknown_worker_limits = set(worker_limits_raw) - set(worker_limit_fields)
        if unknown_worker_limits:
            raise ValueError(
                "speaker_consistency.worker_limits has unknown fields: "
                f"{sorted(unknown_worker_limits)!r}"
            )
        limit_values = {
            field: getattr(default_limits, field) for field in worker_limit_fields
        }
        limit_values.update(worker_limits_raw)
        try:
            worker_limits = WorkerLimits(**limit_values)
        except (OptionalRuntimeConfigError, TypeError) as exc:
            raise ValueError("speaker_consistency.worker_limits is invalid") from exc
        if worker_limits.timeout_ms > 3_600_000:
            raise ValueError(
                "speaker_consistency.worker_limits.timeout_ms must not exceed 3600000"
            )
        if worker_limits.max_stderr_bytes > 65_536:
            raise ValueError(
                "speaker_consistency.worker_limits.max_stderr_bytes must not exceed 65536"
            )
        if ("worker_limits" in raw and "timeout_ms" in worker_limits_raw
                and "worker_timeout_seconds" in raw
                and worker_limits.timeout_ms
                != max(1, math.ceil(float(worker_timeout_seconds) * 1000.0))):
            raise ValueError(
                "speaker_consistency.worker_limits.timeout_ms conflicts with "
                "worker_timeout_seconds"
            )
        if ("worker_limits" in raw and "max_stderr_bytes" in worker_limits_raw
                and "worker_stderr_tail_bytes" in raw
                and worker_limits.max_stderr_bytes != worker_stderr_tail_bytes):
            raise ValueError(
                "speaker_consistency.worker_limits.max_stderr_bytes conflicts with "
                "worker_stderr_tail_bytes"
            )
        canonical["worker_timeout_seconds"] = worker_limits.timeout_ms / 1000.0
        canonical["worker_stderr_tail_bytes"] = worker_limits.max_stderr_bytes
        canonical["worker_limits"] = {
            field: getattr(worker_limits, field) for field in worker_limit_fields
        }
        if "min_cosine_similarity" in raw:
            value = raw["min_cosine_similarity"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not -1.0 <= float(value) <= 1.0
            ):
                raise ValueError(
                    "speaker_consistency.min_cosine_similarity must be finite and in [-1, 1]"
                )
            canonical["min_cosine_similarity"] = float(value)
        return _freeze_snapshot_value(canonical)

    def __init__(self,
                 # ------------------------------------------------------------------
                 # Step 通用
                 # ------------------------------------------------------------------
                 step: str = "step1",

                 # ------------------------------------------------------------------
                 # Step3：MOS 可选资产与 legacy Python 签名兼容参数
                 # ------------------------------------------------------------------
                 mos_types: Any = _UNSET_LEGACY_SELECTION_ARGUMENT,
                 urgent_pk_checkpoint: Optional[str] = None,
                 urgent_pk_source_dir: Optional[str] = None,
                 urgent_pk_encoder: Optional[str] = None,
                 urgent_pk_backbone: Optional[str] = None,
                 urgent_pk_score_range: Optional[Union[str, Dict[str, float]]] = None,
                 # URGENT-PK 仅在 selection_policies 选中后参与 preflight。可信
                 # pickle 加载要求 checkpoint SHA-256 与 source git commit 同时匹配，
                 # 但这只是必要条件：source tree 的 symlink 等不安全 entry 仍会被
                 # scanner fail-closed 拒绝。shipped YAML/生成器与 processors/mos.py
                 # 的 runtime 兼容常量分别维护并由测试核对；生成器不是 runtime
                 # 常量的单一来源。
                 urgent_pk_expected_sha256: Optional[str] = None,
                 urgent_pk_expected_commit: Optional[str] = None,
                 scoreq_model_dir: Optional[str] = None,
                 dnsmos_model_dir: Optional[str] = None,

                 # 只保留 Python 签名/属性形状兼容；sentinel 区分未传与显式传入，
                 # generation-3 对任何显式 legacy selection 参数均 fail-closed。
                 step2_direct_from_input: Any = _UNSET_LEGACY_SELECTION_ARGUMENT,
                 step2_mos_early_exit_threshold: Any = _UNSET_LEGACY_SELECTION_ARGUMENT,

                 # ------------------------------------------------------------------
                 # Step4/Step6 legacy selection Python 签名兼容参数（显式传入即拒绝）
                 # ------------------------------------------------------------------
                 mos_threshold: Any = _UNSET_LEGACY_SELECTION_ARGUMENT,
                 mos_thresholds: Any = _UNSET_LEGACY_SELECTION_ARGUMENT,
                 mos_fusion: Any = _UNSET_LEGACY_SELECTION_ARGUMENT,
                 enable_multi_mos: Any = _UNSET_LEGACY_SELECTION_ARGUMENT,
                 min_mos_models: Any = _UNSET_LEGACY_SELECTION_ARGUMENT,

                 # ------------------------------------------------------------------
                 # 数据集 registry 与 selection policies
                 # ------------------------------------------------------------------
                 datasets: Optional[Dict[str, Any]] = None,
                 selection_policies: Optional[Dict[str, Any]] = None,

                 # ------------------------------------------------------------------
                 # 输出策略
                 # ------------------------------------------------------------------

                 # ------------------------------------------------------------------
                 # 信号完整性检查
                 # ------------------------------------------------------------------
                 enable_signal_integrity_check: bool = True,
                 signal_integrity_max_zero_ratio: float = 0.99,
                 signal_integrity_min_rms_dbfs: float = -60.0,
                 signal_integrity_max_dc_offset: float = 0.5,
                 signal_integrity_clip_threshold: float = 1.0,

                 # ------------------------------------------------------------------
                 # 通用输出根（保留既有 positional slot）
                 # ------------------------------------------------------------------
                 output_root: Optional[str] = None,

                 # ------------------------------------------------------------------
                 # 并行 / 资源 / GPU
                 # ------------------------------------------------------------------

                 # ------------------------------------------------------------------
                 # 进度 / 断点 / 临时目录 / 可复现
                 # ------------------------------------------------------------------

                 # ------------------------------------------------------------------
                 # 高级 / Stage 自定义
                 # ------------------------------------------------------------------
                 # ------------------------------------------------------------------
                 # 运行时 / 非持久化字段
                 # ------------------------------------------------------------------
                 dataset_name: Optional[str] = None,
                 # 新增持久化字段追加在末尾，保持全部既有 positional slot 不变
                 pipeline_generation: int = 3,
                 speaker_consistency: Optional[Mapping[str, Any]] = None,
                 step5_apply_speaker_filter: Optional[bool] = None,
                 step6_mode: Optional[str] = None,
                 audio_export_workers: int = 32,
                 audio_export_max_pending: int = 64,
                 audio_export_max_inflight_bytes: int = 4 * 1024 * 1024 * 1024,
                 step4_mode: Optional[str] = None,
                 step4_export_root: Optional[str] = None,
                 step6_export_root: Optional[str] = None,
                 step3_gpu_ids: Union[List[int], Tuple[int, ...]] = (),
                 step3_worker_stderr_tail_bytes: int = 4_096,
                 step1_vad_workers: Optional[int] = None,
                 step1_vad_max_pending: Optional[int] = None,
                 output_sample_rate_hz: Optional[int] = None,
                 min_source_sample_rate_hz: Optional[int] = None,
                 nisqa_model_path: Optional[str] = None,
                 nisqa_model_version: Optional[str] = None,
                 nisqa_model_sha256: Optional[str] = None,
                 utmos_model_path: Optional[str] = None,
                 utmos_model_version: Optional[str] = None,
                 utmos_model_sha256: Optional[str] = None,
                 step3_worker_idle_timeout_seconds: Optional[float] = None,
                 step3_worker_hard_timeout_seconds: Optional[float] = None,
                 step2_max_segment_duration_sec: float = 60.0,
                 step2_min_segment_duration_sec: float = 0.0):

        # ------------------------------------------------------------------
        # Step 通用
        # ------------------------------------------------------------------
        self.pipeline_generation = _validate_pipeline_generation(pipeline_generation)
        if step is not None and type(step) is not str:
            raise ValueError("step must be a string")
        self.step = (step or "step1").strip().lower()
        self._explicit_quality_profile_dataset_ids = frozenset(
            str(dataset_id)
            for dataset_id, raw in (datasets or {}).items()
            if isinstance(raw, Mapping) and "quality_profile" in raw
        )
        if self.pipeline_generation == 3:
            legacy_selection_arguments = {
                "mos_types": mos_types,
                "mos_threshold": mos_threshold,
                "mos_thresholds": mos_thresholds,
                "enable_multi_mos": enable_multi_mos,
                "min_mos_models": min_mos_models,
                "mos_fusion": mos_fusion,
                "step2_direct_from_input": step2_direct_from_input,
                "step2_mos_early_exit_threshold": step2_mos_early_exit_threshold,
            }
            explicit_legacy_arguments = sorted(
                name for name, value in legacy_selection_arguments.items()
                if value is not _UNSET_LEGACY_SELECTION_ARGUMENT
            )
            if explicit_legacy_arguments:
                raise ValueError(
                    "pipeline generation 3 does not accept legacy selection "
                    f"argument(s) {explicit_legacy_arguments!r}; configure "
                    "selection_policies instead"
                )
            for name, default in _V3_LEGACY_DEFAULT_ATTRIBUTES.items():
                setattr(self, name, copy.deepcopy(default))
            if output_root is None:
                self.output_root = None
            else:
                normalized_output_root = os.fspath(output_root)
                if (
                    type(normalized_output_root) is not str
                    or not normalized_output_root.strip()
                ):
                    raise ValueError("output_root must be a non-empty string")
                self.output_root = normalized_output_root
            if min_source_sample_rate_hz is not None and (
                type(min_source_sample_rate_hz) is not int
                or min_source_sample_rate_hz <= 0
            ):
                raise ValueError(
                    "min_source_sample_rate_hz must be a positive Python integer or None"
                )
            self._min_source_sample_rate_hz = min_source_sample_rate_hz
            self.datasets = None if datasets is None else copy.deepcopy(dict(datasets))
            self._validate_dataset_identities(self.datasets)
            self._dataset_specs_snapshot = MappingProxyType(
                self._parse_dataset_specs(self.datasets)
            )
            if self.datasets is not None:
                self.datasets = _freeze_snapshot_value(self.datasets)
            self._selection_policy_book = self._build_selection_policy_book(
                selection_policies
            )
            self._selection_policies_snapshot = self._selection_policy_book.snapshot
            if (
                step5_apply_speaker_filter is not None
                and type(step5_apply_speaker_filter) is not bool
            ):
                raise ValueError("step5_apply_speaker_filter must be bool or None")
            inferred_mode = "filter"
            requested_mode = step6_mode if step6_mode is not None else inferred_mode
            if requested_mode not in {"dry_run", "filter"}:
                raise ValueError("step6_mode must be exactly 'dry_run' or 'filter'")
            if step5_apply_speaker_filter and requested_mode != "filter":
                raise ValueError("step5_apply_speaker_filter=True conflicts with step6_mode")
            self._speaker_consistency_snapshot = self._canonical_speaker_consistency(
                speaker_consistency,
            )
            if requested_mode == "filter" and self._speaker_consistency_snapshot["enabled"]:
                # 声纹启用时 filter 才需要 enrollment 阈值；声纹未启用时
                # filter 等价纯 MOS 筛选导出（export.py 的 final_selection 已支持）。
                if "min_cosine_similarity" not in self._speaker_consistency_snapshot:
                    raise ValueError(
                        "step6_mode='filter' requires speaker_consistency.min_cosine_similarity"
                    )
            self._step6_mode = requested_mode
            # 未显式给出时按 step6_mode 推导；显式给出（含 False）必须被尊重，
            # 不得静默改写。
            self.step5_apply_speaker_filter = (
                requested_mode == "filter"
                if step5_apply_speaker_filter is None
                else step5_apply_speaker_filter
            )
            if step4_mode is None:
                step4_mode = "filter"
            if step4_mode not in {"dry_run", "filter"}:
                raise ValueError("step4_mode must be exactly 'dry_run' or 'filter'")
            self._step4_mode = step4_mode
            if type(audio_export_workers) is not int or audio_export_workers <= 0:
                raise ValueError("audio_export_workers must be a positive Python integer")
            if type(audio_export_max_pending) is not int or audio_export_max_pending < audio_export_workers:
                raise ValueError(
                    "audio_export_max_pending must be a Python integer >= audio_export_workers"
                )
            if (type(audio_export_max_inflight_bytes) is not int
                    or audio_export_max_inflight_bytes <= 0):
                raise ValueError(
                    "audio_export_max_inflight_bytes must be a positive Python integer"
                )
            self.audio_export_workers = audio_export_workers
            self.audio_export_max_pending = audio_export_max_pending
            self.audio_export_max_inflight_bytes = audio_export_max_inflight_bytes
            for field_name, value in (
                ("step4_export_root", step4_export_root),
                ("step6_export_root", step6_export_root),
            ):
                if value is None:
                    setattr(self, field_name, None)
                    continue
                normalized = os.fspath(value)
                if type(normalized) is not str or not normalized.strip():
                    raise ValueError(f"{field_name} must be a non-empty string or None")
                setattr(self, field_name, normalized)
            _validate_export_root_boundaries(
                self.output_root,
                self._dataset_specs_snapshot,
                {
                    "step4_export_root": self.step4_export_root,
                    "step6_export_root": self.step6_export_root,
                },
            )
            default_step1_vad_workers = min(
                STEP1_VAD_MAX_WORKERS, max(1, os.cpu_count() or 1)
            )
            if step1_vad_workers is None:
                step1_vad_workers = default_step1_vad_workers
            if step1_vad_max_pending is None:
                step1_vad_max_pending = step1_vad_workers * 2
            if (
                type(step1_vad_workers) is not int
                or not 0 < step1_vad_workers <= STEP1_VAD_MAX_WORKERS
            ):
                raise ValueError(
                    "step1_vad_workers must be a positive Python integer "
                    f"<= {STEP1_VAD_MAX_WORKERS}"
                )
            if (
                type(step1_vad_max_pending) is not int
                or not step1_vad_workers <= step1_vad_max_pending <= step1_vad_workers * 2
            ):
                raise ValueError(
                    "step1_vad_max_pending must be a Python integer in "
                    "[step1_vad_workers, step1_vad_workers * 2]"
                )
            self.step1_vad_workers = step1_vad_workers
            self.step1_vad_max_pending = step1_vad_max_pending
            if not isinstance(step3_gpu_ids, (list, tuple)):
                raise ValueError("step3_gpu_ids must be a list or tuple of unique nonnegative integers")
            if (any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in step3_gpu_ids)
                    or len(set(step3_gpu_ids)) != len(step3_gpu_ids)):
                raise ValueError("step3_gpu_ids must be a list or tuple of unique nonnegative integers")
            self.step3_gpu_ids = tuple(step3_gpu_ids)
            if step3_worker_idle_timeout_seconds is None:
                step3_worker_idle_timeout_seconds = 900.0
            if (
                isinstance(step3_worker_idle_timeout_seconds, bool)
                or not isinstance(step3_worker_idle_timeout_seconds, (int, float))
                or not math.isfinite(float(step3_worker_idle_timeout_seconds))
                or float(step3_worker_idle_timeout_seconds) <= 0.0
            ):
                raise ValueError(
                    "step3_worker_idle_timeout_seconds must be a positive finite number"
                )
            if step3_worker_hard_timeout_seconds is not None and (
                isinstance(step3_worker_hard_timeout_seconds, bool)
                or not isinstance(step3_worker_hard_timeout_seconds, (int, float))
                or not math.isfinite(float(step3_worker_hard_timeout_seconds))
                or float(step3_worker_hard_timeout_seconds) <= 0.0
            ):
                raise ValueError(
                    "step3_worker_hard_timeout_seconds must be a positive finite "
                    "number or None"
                )
            if (type(step3_worker_stderr_tail_bytes) is not int
                    or not 0 < step3_worker_stderr_tail_bytes <= 65_536):
                raise ValueError(
                    "step3_worker_stderr_tail_bytes must be an integer in [1, 65536]"
                )
            if (
                isinstance(step2_max_segment_duration_sec, bool)
                or not isinstance(step2_max_segment_duration_sec, (int, float))
                or not math.isfinite(float(step2_max_segment_duration_sec))
                or float(step2_max_segment_duration_sec) <= 0.0
            ):
                raise ValueError(
                    "step2_max_segment_duration_sec must be a positive finite number"
                )
            self.step2_max_segment_duration_sec = float(
                step2_max_segment_duration_sec
            )
            if (
                isinstance(step2_min_segment_duration_sec, bool)
                or not isinstance(step2_min_segment_duration_sec, (int, float))
                or not math.isfinite(float(step2_min_segment_duration_sec))
                or float(step2_min_segment_duration_sec) < 0.0
            ):
                raise ValueError(
                    "step2_min_segment_duration_sec must be a nonnegative finite number"
                )
            if self.step2_max_segment_duration_sec < float(
                step2_min_segment_duration_sec
            ):
                raise ValueError(
                    "step2_min_segment_duration_sec must not exceed "
                    "step2_max_segment_duration_sec"
                )
            self.step2_min_segment_duration_sec = float(
                step2_min_segment_duration_sec
            )
            self.step3_worker_idle_timeout_seconds = float(
                step3_worker_idle_timeout_seconds
            )
            self.step3_worker_hard_timeout_seconds = (
                None if step3_worker_hard_timeout_seconds is None
                else float(step3_worker_hard_timeout_seconds)
            )
            self.step3_worker_stderr_tail_bytes = step3_worker_stderr_tail_bytes
            if output_sample_rate_hz is not None and (
                type(output_sample_rate_hz) is not int
                or output_sample_rate_hz not in {16_000, 32_000}
            ):
                raise ValueError("output_sample_rate_hz must be 16000, 32000, or None")
            self.output_sample_rate_hz = output_sample_rate_hz
            # generation-3 MOS 资产统一收进本次 PipelineConfig 快照，供 validate
            # 与 stages_v3 工厂按 selected model 读取；未选资产保留但不 preflight。
            # 这只是单次 runtime 配置的权威来源，不表示配置生成器与 runtime
            # 兼容常量共享同一份源码常量。
            self.urgent_pk_checkpoint = urgent_pk_checkpoint
            self.urgent_pk_source_dir = urgent_pk_source_dir
            self.urgent_pk_encoder = urgent_pk_encoder
            self.urgent_pk_backbone = urgent_pk_backbone
            self.urgent_pk_score_range = urgent_pk_score_range
            self.urgent_pk_expected_sha256 = urgent_pk_expected_sha256
            self.urgent_pk_expected_commit = urgent_pk_expected_commit
            self.scoreq_model_dir = scoreq_model_dir
            self.dnsmos_model_dir = dnsmos_model_dir
            self.nisqa_model_path = nisqa_model_path
            self.nisqa_model_version = nisqa_model_version
            self.nisqa_model_sha256 = nisqa_model_sha256
            self.utmos_model_path = utmos_model_path
            self.utmos_model_version = utmos_model_version
            self.utmos_model_sha256 = utmos_model_sha256
            return

    @staticmethod
    def _normalize_language(language: Optional[str]) -> str:
        """兼容旧私有入口；实际规范化由纯策略模块提供。"""
        return _normalize_policy_language(language)

    @staticmethod
    def _speaker_metadata_spec(
        dataset_id: Any, raw: Any,
    ) -> SpeakerMetadataSpec:
        """解析并校验单个数据集的显式 speaker metadata。"""
        if not isinstance(raw, Mapping):
            raise ValueError(f"dataset {dataset_id!r} speaker must be a mapping")
        allowed_fields = {
            "source", "path_component", "regex", "mapping_path",
            "regex_group", "required",
        }
        non_string_fields = [field for field in raw if type(field) is not str]
        if non_string_fields:
            raise ValueError(
                f"dataset {dataset_id!r} speaker field names must be strings, "
                f"got {non_string_fields!r}"
            )
        unknown_fields = set(raw) - allowed_fields
        if unknown_fields:
            raise ValueError(
                f"dataset {dataset_id!r} speaker has unknown fields: "
                f"{sorted(unknown_fields)!r}"
            )

        source = raw.get("source")
        allowed_sources = {"path_component", "regex", "mapping_file", "none"}
        if not isinstance(source, str) or source not in allowed_sources:
            raise ValueError(
                f"dataset {dataset_id!r} speaker source must be one of "
                f"{sorted(allowed_sources)!r}, got {source!r}"
            )
        required = raw.get("required", True)
        if type(required) is not bool:
            raise ValueError(
                f"dataset {dataset_id!r} speaker required must be bool, "
                f"got {required!r}"
            )

        source_fields = {"path_component", "regex", "mapping_path", "regex_group"}
        permitted_fields = {
            "path_component": {"path_component"},
            "regex": {"regex", "regex_group"},
            "mapping_file": {"mapping_path"},
            "none": set(),
        }[source]
        contradictory_fields = (set(raw) & source_fields) - permitted_fields
        if contradictory_fields:
            raise ValueError(
                f"dataset {dataset_id!r} speaker source {source!r} does not allow "
                f"fields {sorted(contradictory_fields)!r}"
            )

        if source == "path_component":
            path_component = raw.get("path_component")
            if type(path_component) is not int or path_component < 0:
                raise ValueError(
                    f"dataset {dataset_id!r} speaker path_component must be a "
                    f"non-negative integer, got {path_component!r}"
                )
            return SpeakerMetadataSpec(
                source=source,
                path_component=path_component,
                required=required,
            )

        if source == "regex":
            pattern = raw.get("regex")
            if not isinstance(pattern, str) or not pattern.strip():
                raise ValueError(
                    f"dataset {dataset_id!r} speaker regex must be a non-empty string"
                )
            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"dataset {dataset_id!r} speaker regex must be compilable: {exc}"
                ) from exc
            regex_group = raw.get("regex_group", "speaker_id")
            if not isinstance(regex_group, str) or not regex_group.strip():
                raise ValueError(
                    f"dataset {dataset_id!r} speaker regex_group must be a "
                    "non-empty string"
                )
            if regex_group not in compiled.groupindex:
                raise ValueError(
                    f"dataset {dataset_id!r} speaker regex_group {regex_group!r} "
                    "must name a group in regex"
                )
            return SpeakerMetadataSpec(
                source=source,
                regex=pattern,
                regex_group=regex_group,
                required=required,
            )

        if source == "mapping_file":
            mapping_path = raw.get("mapping_path")
            if not isinstance(mapping_path, str) or not mapping_path.strip():
                raise ValueError(
                    f"dataset {dataset_id!r} speaker mapping_path must be a "
                    "non-empty string"
                )
            return SpeakerMetadataSpec(
                source=source,
                mapping_path=mapping_path,
                required=required,
            )

        if required is not False:
            raise ValueError(
                f"dataset {dataset_id!r} speaker source 'none' requires "
                "required=False"
            )
        return SpeakerMetadataSpec(source="none", required=False)

    def _source_selector_spec(
        self, dataset_id: Any, raw: Any,
    ) -> SourceSelectorSpec:
        """解析并规范化单个数据集的可证明互斥 source selector。"""
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"dataset {dataset_id!r} source_selector must be a mapping"
            )
        allowed_fields = {
            "sample_rates_hz", "include_speaker_ids", "exclude_relative_paths",
        }
        non_string_fields = [field for field in raw if type(field) is not str]
        if non_string_fields:
            raise ValueError(
                f"dataset {dataset_id!r} source_selector field names must be "
                f"strings, got {non_string_fields!r}"
            )
        unknown_fields = set(raw) - allowed_fields
        if unknown_fields:
            raise ValueError(
                f"dataset {dataset_id!r} source_selector has unknown fields: "
                f"{sorted(unknown_fields)!r}"
            )

        sample_rates = raw.get("sample_rates_hz", ())
        if not isinstance(sample_rates, (list, tuple)):
            raise ValueError(
                f"dataset {dataset_id!r} source_selector.sample_rates_hz "
                "must be a list or tuple"
            )
        if any(type(rate) is not int or rate <= 0 for rate in sample_rates):
            raise ValueError(
                f"dataset {dataset_id!r} source_selector.sample_rates_hz "
                "entries must be positive integers"
            )
        if (
            self.min_source_sample_rate_hz is not None
            and any(
                rate < self.min_source_sample_rate_hz for rate in sample_rates
            )
        ):
            raise ValueError(
                f"dataset {dataset_id!r} source_selector.sample_rates_hz entries "
                "must be greater than or equal to min_source_sample_rate_hz "
                f"{self.min_source_sample_rate_hz}"
            )

        speaker_ids = raw.get("include_speaker_ids", ())
        if not isinstance(speaker_ids, (list, tuple)):
            raise ValueError(
                f"dataset {dataset_id!r} source_selector.include_speaker_ids "
                "must be a list or tuple"
            )
        if any(
            type(speaker_id) is not str
            or not speaker_id
            or speaker_id != speaker_id.strip()
            for speaker_id in speaker_ids
        ):
            raise ValueError(
                f"dataset {dataset_id!r} source_selector.include_speaker_ids "
                "entries must be non-empty strings without leading or trailing whitespace"
            )

        excluded_paths = raw.get("exclude_relative_paths", ())
        if not isinstance(excluded_paths, (list, tuple)):
            raise ValueError(
                f"dataset {dataset_id!r} source_selector.exclude_relative_paths "
                "must be a list or tuple"
            )
        for relative_path in excluded_paths:
            if (
                type(relative_path) is not str
                or not relative_path
                or "\x00" in relative_path
                or "\\" in relative_path
                or relative_path.startswith("/")
                or any(component in ("", ".", "..") for component in relative_path.split("/"))
            ):
                raise ValueError(
                    f"dataset {dataset_id!r} source_selector.exclude_relative_paths "
                    "entries must be normalized relative POSIX paths"
                )

        selector = SourceSelectorSpec(
            sample_rates_hz=tuple(sorted(set(sample_rates))),
            include_speaker_ids=tuple(sorted(set(speaker_ids))),
            exclude_relative_paths=tuple(sorted(set(excluded_paths))),
        )
        if (
            not selector.sample_rates_hz
            and not selector.include_speaker_ids
            and not selector.exclude_relative_paths
        ):
            raise ValueError(
                f"dataset {dataset_id!r} source_selector requires at least one "
                "non-empty field"
            )
        return selector

    @staticmethod
    def _validate_source_selector_exclusivity(
        specs: Mapping[str, DatasetSpec],
    ) -> None:
        """同一 canonical root 仅接受可静态证明两两互斥的 specs。"""
        by_root: Dict[str, List[DatasetSpec]] = {}
        for spec in specs.values():
            canonical_root = os.path.realpath(os.path.abspath(spec.input_root))
            by_root.setdefault(canonical_root, []).append(spec)

        for canonical_root, grouped_specs in by_root.items():
            for index, left in enumerate(grouped_specs):
                for right in grouped_specs[index + 1:]:
                    left_selector = left.source_selector
                    right_selector = right.source_selector
                    rates_disjoint = bool(
                        left_selector.sample_rates_hz
                        and right_selector.sample_rates_hz
                        and set(left_selector.sample_rates_hz).isdisjoint(
                            right_selector.sample_rates_hz
                        )
                    )
                    same_speaker_resolver = bool(
                        (
                            left.speaker.source == right.speaker.source == "path_component"
                            and left.speaker.path_component == right.speaker.path_component
                        )
                        or (
                            left.speaker.source == right.speaker.source == "regex"
                            and left.speaker.regex == right.speaker.regex
                            and left.speaker.regex_group == right.speaker.regex_group
                        )
                    )
                    speakers_disjoint = bool(
                        same_speaker_resolver
                        and left_selector.include_speaker_ids
                        and right_selector.include_speaker_ids
                        and set(left_selector.include_speaker_ids).isdisjoint(
                            right_selector.include_speaker_ids
                        )
                    )
                    if not rates_disjoint and not speakers_disjoint:
                        raise ValueError(
                            f"datasets {left.dataset_id!r} and {right.dataset_id!r} "
                            f"share canonical input_root {canonical_root!r}, but "
                            "their source_selectors are not provably disjoint; "
                            "speaker-set exclusivity requires identical path_component "
                            "or regex speaker metadata"
                        )

    @staticmethod
    def _validate_dataset_identities(
        datasets: Optional[Mapping[str, Any]],
    ) -> None:
        """仅校验 generation-3 dataset registry 的 identity 键。"""
        if datasets is None:
            return
        if not isinstance(datasets, Mapping):
            raise ValueError("datasets must be a mapping")
        for dataset_id in datasets:
            if type(dataset_id) is not str or not dataset_id.strip():
                raise ValueError(
                    f"dataset id must be a non-empty string, got {dataset_id!r}"
                )

    def _parse_dataset_specs(
        self, datasets: Optional[Mapping[str, Any]],
    ) -> Dict[str, DatasetSpec]:
        """解析 dataset registry（generation-3）。"""
        self._validate_dataset_identities(datasets)
        if datasets is None:
            return {}
        allowed_v3_fields = {
            "input_root", "language", "category", "quality_profile",
            "speaker", "source_selector", "source_layout",
        }
        specs = {}
        for raw_dataset_id, raw in datasets.items():
            dataset_id = raw_dataset_id
            if dataset_id in specs:
                raise ValueError(f"duplicate dataset identity: {dataset_id!r}")
            if not isinstance(raw, Mapping):
                raise ValueError(f"dataset {dataset_id!r} must be a mapping")
            non_string_fields = [field for field in raw if type(field) is not str]
            if non_string_fields:
                raise ValueError(
                    f"dataset {dataset_id!r} field names must be strings, "
                    f"got {non_string_fields!r}"
                )
            unknown_fields = set(raw) - allowed_v3_fields
            if unknown_fields:
                raise ValueError(
                    f"dataset {dataset_id!r} has unknown fields: "
                    f"{sorted(unknown_fields)!r}"
                )
            input_root = raw.get("input_root")
            if not isinstance(input_root, str) or not input_root.strip():
                raise ValueError(
                    f"dataset {dataset_id!r} input_root must be a non-empty string"
                )
            if not os.path.isabs(input_root):
                raise ValueError(
                    f"dataset {dataset_id!r} input_root must be an absolute path "
                    "for pipeline generation 3"
                )
            raw_language = raw.get("language")
            if type(raw_language) is not str or not raw_language.strip():
                raise ValueError(
                    f"dataset {dataset_id!r} language must be a non-empty string"
                )
            language = self._normalize_language(raw_language)
            if not language:
                raise ValueError(f"dataset {dataset_id!r} missing language")
            category = raw.get("category")
            source_layout = raw.get("source_layout", "regular_file")
            if type(source_layout) is not str or source_layout not in DATASET_SOURCE_LAYOUTS:
                raise ValueError(
                    f"dataset {dataset_id!r} source_layout must be one of "
                    f"{sorted(DATASET_SOURCE_LAYOUTS)!r}, got {source_layout!r}"
                )
            quality_profile = raw.get("quality_profile")
            if category is not None and (
                type(category) is not str
                or category not in SEMANTIC_DATASET_CATEGORIES
            ):
                raise ValueError(
                    f"dataset {dataset_id!r} category must be None or an exact "
                    f"semantic category: {sorted(SEMANTIC_DATASET_CATEGORIES)!r}; "
                    "legacy quality categories must move to quality_profile"
                )
            if "quality_profile" not in raw:
                raise ValueError(
                    f"dataset {dataset_id!r} missing explicit quality_profile"
                )
            if quality_profile not in QUALITY_PROFILES:
                raise ValueError(
                    f"dataset {dataset_id!r} quality_profile must be one of "
                    f"{QUALITY_PROFILES!r}, got {quality_profile!r}"
                )
            if "speaker" not in raw:
                raise ValueError(
                    f"dataset {dataset_id!r} missing explicit speaker"
                )
            speaker = self._speaker_metadata_spec(dataset_id, raw["speaker"])
            source_selector = (
                self._source_selector_spec(dataset_id, raw["source_selector"])
                if "source_selector" in raw
                else SourceSelectorSpec()
            )
            if source_selector.include_speaker_ids and speaker.source == "none":
                raise ValueError(
                    f"dataset {dataset_id!r} source_selector.include_speaker_ids "
                    "requires speaker source other than 'none'"
                )

            specs[dataset_id] = DatasetSpec(
                dataset_id=dataset_id,
                input_root=input_root,
                language=language,
                category=category,
                quality_profile=quality_profile,
                speaker=speaker,
                source_selector=source_selector,
                source_layout=source_layout,
            )
        self._validate_source_selector_exclusivity(specs)
        return {dataset_id: specs[dataset_id] for dataset_id in sorted(specs)}

    def _dataset_specs(self) -> Dict[str, DatasetSpec]:
        """兼容内部旧调用；generation-3 返回 snapshot。"""
        return dict(self._dataset_specs_snapshot)

    def dataset_specs(self) -> Mapping[str, DatasetSpec]:
        """返回只读 registry（构造时固定）。"""
        return self._dataset_specs_snapshot

    def dataset_spec(self, dataset_id: str) -> DatasetSpec:
        """按精确 dataset identity 返回 immutable spec，未知值 fail closed。"""
        if type(dataset_id) is not str or not dataset_id.strip():
            raise ValueError(
                f"dataset id must be a non-empty string, got {dataset_id!r}"
            )
        try:
            return self.dataset_specs()[dataset_id]
        except KeyError as exc:
            raise ValueError(f"unknown dataset: {dataset_id}") from exc

    @staticmethod
    def _serialized_speaker(spec: SpeakerMetadataSpec) -> Dict[str, Any]:
        payload = {"source": spec.source, "required": spec.required}
        if spec.source == "path_component":
            payload["path_component"] = spec.path_component
        elif spec.source == "regex":
            payload["regex"] = spec.regex
            payload["regex_group"] = spec.regex_group
        elif spec.source == "mapping_file":
            payload["mapping_path"] = spec.mapping_path
        return payload

    def _serialized_datasets(self) -> Dict[str, Any]:
        """generation-3 从 canonical spec 序列化。"""
        serialized = {}
        for dataset_id, spec in self._dataset_specs_snapshot.items():
            payload = {
                "input_root": spec.input_root,
                "language": spec.language,
                "category": spec.category,
                "quality_profile": spec.quality_profile,
                "speaker": self._serialized_speaker(spec.speaker),
            }
            if spec.source_layout != "regular_file":
                payload["source_layout"] = spec.source_layout
            if (
                spec.source_selector.sample_rates_hz
                or spec.source_selector.include_speaker_ids
                or spec.source_selector.exclude_relative_paths
            ):
                selector_payload = {}
                if spec.source_selector.sample_rates_hz:
                    selector_payload["sample_rates_hz"] = list(
                        spec.source_selector.sample_rates_hz
                    )
                if spec.source_selector.include_speaker_ids:
                    selector_payload["include_speaker_ids"] = list(
                        spec.source_selector.include_speaker_ids
                    )
                if spec.source_selector.exclude_relative_paths:
                    selector_payload["exclude_relative_paths"] = list(
                        spec.source_selector.exclude_relative_paths
                    )
                payload["source_selector"] = selector_payload
            serialized[dataset_id] = payload
        return serialized

    def _build_selection_policy_book(self, raw_policies: Any) -> SelectionPolicyBook:
        """把 dataset registry 投影为纯策略模块所需的不可变 route。"""
        routes = {
            dataset_id: SelectionRoute(
                dataset_id=dataset_id,
                language=spec.language,
                category=spec.category,
                quality_profile=spec.quality_profile,
                quality_profile_explicit=(
                    dataset_id in self._explicit_quality_profile_dataset_ids
                ),
            )
            for dataset_id, spec in self._dataset_specs_snapshot.items()
        }
        return SelectionPolicyBook(
            raw_policies,
            routes=routes,
            serialized_datasets=self._serialized_datasets(),
            default_policy=_V3_SELECTION_POLICY_DEFAULTS,
            get_mos_spec=_get_selection_mos_spec,
        )

    def _serialized_selection_policies(self) -> Dict[str, Any]:
        return self._selection_policy_book.serialized()

    def resolve_selection_policy(
        self,
        dataset_id: str,
        language: Optional[str] = None,
        *,
        source_relative_path: Optional[str] = None,
    ) -> ResolvedSelectionPolicy:
        """保留既有 public facade，并委托纯 ``SelectionPolicyBook``。"""
        return self._selection_policy_book.resolve(
            dataset_id,
            language,
            source_relative_path=source_relative_path,
        )

    def mos_scoring_projection(
        self,
        dataset_id: str,
        language: Optional[str] = None,
        *,
        source_relative_path: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """保留既有 mapping facade，并委托纯 scoring projection。"""
        projection: MosScoringProjection = self._selection_policy_book.scoring_projection(
            dataset_id,
            language,
            source_relative_path=source_relative_path,
        )
        return projection.as_mapping()

    def dataset_mos_scoring_binding(self, dataset_id: str) -> Mapping[str, Any]:
        """返回仅影响指定数据集 Step3 评分的只读策略绑定。"""
        self.dataset_spec(dataset_id)
        return self._selection_policy_book.dataset_scoring_binding(dataset_id)

    def dataset_mos_decision_binding(self, dataset_id: str) -> Mapping[str, Any]:
        """返回仅影响指定数据集 Step4/Step6 决策的只读策略绑定。"""
        self.dataset_spec(dataset_id)
        return self._selection_policy_book.dataset_decision_binding(dataset_id)

    def _validate_selection_policy_structure(self) -> None:
        """保留既有校验入口，并委托构造时固定的纯策略簿。"""
        self._selection_policy_book.validate_structure()

    def _validate_v3_selection_policy_semantics(self) -> None:
        """保留既有语义校验入口，并委托纯策略簿。"""
        self._selection_policy_book.validate_semantics()

    def validate(
        self,
        require_scores_file: bool = True,
        context: Optional[ValidationContext] = None,
        stage_names: Optional[List[str]] = None,
    ) -> None:
        """按结构或执行阶段校验（generation-3）。

        ``stage_names=None`` 保持生产入口的保守语义：执行校验包含 MOS 资产
        preflight。显式传入实际 Stage chain 时，仅当 chain 含 ``mos`` 才访问
        MOS 资产；其它配置、数据集与策略校验不弱化。Step5/Step6 execution
        context 在任何 stage/factory 启动前始终校验已启用的 speaker 资产。
        """
        self.pipeline_generation = _validate_pipeline_generation(
            self.pipeline_generation
        )
        if context is None:
            context = validation_context_for_step(self.step, self.pipeline_generation)
        elif not isinstance(context, ValidationContext):
            context = ValidationContext(context)
        validate_mos_assets = stage_names is None or "mos" in stage_names
        v3_contexts = {
            ValidationContext.V3_STEP0_EXECUTION,
            ValidationContext.V3_STEP1_EXECUTION,
            ValidationContext.V3_STEP2_EXECUTION,
            ValidationContext.V3_STEP3_EXECUTION,
            ValidationContext.V3_STEP4_EXECUTION,
            ValidationContext.V3_STEP5_EXECUTION,
            ValidationContext.V3_STEP6_EXECUTION,
        }
        valid_steps = ("step0", "step1", "step2", "step3", "step4", "step5", "step6", "all")
        if context is not ValidationContext.STRUCTURAL and context not in v3_contexts:
            raise ValueError(
                f"validation context {context.value!r} is not valid for pipeline "
                f"generation {self.pipeline_generation}"
            )
        if self.step not in valid_steps:
            raise ValueError(
                f"step {self.step!r} is not valid for pipeline generation "
                f"{self.pipeline_generation}; expected one of {valid_steps!r}"
            )
        if self.step2_direct_from_input:
            raise ValueError(
                "step2_direct_from_input=True is not supported for pipeline generation 3"
            )
        if context in v3_contexts and (
            not isinstance(self.datasets, Mapping) or not self.datasets
        ):
            raise ValueError("datasets must be a non-empty mapping in execution contexts")
        if context in v3_contexts and (
            type(self.output_root) is not str or not self.output_root.strip()
        ):
            raise ValueError("output_root must be a non-empty string in execution contexts")
        self._validate_selection_policy_structure()
        self._validate_v3_selection_policy_semantics()
        if validate_mos_assets and context in {
            ValidationContext.V3_STEP3_EXECUTION,
            ValidationContext.V3_STEP4_EXECUTION,
            ValidationContext.V3_STEP5_EXECUTION,
            ValidationContext.V3_STEP6_EXECUTION,
        }:
            self._validate_mos_assets()
        if context in {
            ValidationContext.V3_STEP5_EXECUTION,
            ValidationContext.V3_STEP6_EXECUTION,
        }:
            self._validate_speaker_assets()

    @staticmethod
    def _verify_speaker_regular_file(
        path: str, expected_sha256: str, label: str,
    ) -> None:
        """通过绑定 descriptor 校验 enrollment regular file 与内容摘要。"""
        from ..utils.strict_jsonl import file_signature, open_regular_stream

        stream = None
        try:
            try:
                stream, initial = open_regular_stream(path, label)
            except FileNotFoundError:
                raise ValueError(f"{label} does not exist") from None
            digest = hashlib.sha256()
            total = 0
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > 32 * 1024 * 1024:
                    raise ValueError(f"{label} exceeds size limit")
                digest.update(chunk)
            final = os.fstat(stream.fileno())
            if file_signature(final) != file_signature(initial) or total != final.st_size:
                raise ValueError(f"{label} changed while being read")
        except OSError as exc:
            raise ValueError(f"cannot safely read {label}") from exc
        finally:
            if stream is not None:
                stream.close()
        actual_sha256 = digest.hexdigest()
        if not hmac.compare_digest(actual_sha256, expected_sha256):
            raise RuntimeError(
                f"{label} digest mismatch: expected {expected_sha256}, "
                f"actual {actual_sha256}"
            )

    def _validate_speaker_assets(self) -> None:
        """Step5/6 execution 前验证 WeSpeaker 与 enrollment 资产绑定。"""
        from ..processors.optional_runtime import (
            LocalModelAsset,
            verify_local_model_asset,
        )

        policy = self._speaker_consistency_snapshot
        if not policy["enabled"]:
            return
        placeholder_fields = {
            "model_dir": "REPLACE_ME" in policy["model_dir"].split(os.sep),
            "enrollment_bundle_path": "REPLACE_ME" in policy["enrollment_bundle_path"].split(os.sep),
            "enrollment_exclusion_path": "REPLACE_ME" in policy["enrollment_exclusion_path"].split(os.sep),
            "model_version": policy["model_version"] == "replace-me",
            "runtime_version": policy["runtime_version"] == "replace-me",
            "model_sha256": policy["model_sha256"] == "0" * 64,
            "enrollment_bundle_sha256": policy["enrollment_bundle_sha256"] == "0" * 64,
            "enrollment_exclusion_sha256": policy["enrollment_exclusion_sha256"] == "0" * 64,
        }
        for field, is_placeholder in placeholder_fields.items():
            if is_placeholder:
                raise ValueError(
                    f"speaker_consistency.{field} is a non-executable placeholder"
                )
        verify_local_model_asset(LocalModelAsset(
            local_path=policy["model_dir"],
            version=policy["model_version"],
            sha256=policy["model_sha256"],
        ))
        self._verify_speaker_regular_file(
            policy["enrollment_bundle_path"],
            policy["enrollment_bundle_sha256"],
            "speaker_consistency enrollment bundle",
        )
        self._verify_speaker_regular_file(
            policy["enrollment_exclusion_path"],
            policy["enrollment_exclusion_sha256"],
            "speaker_consistency enrollment exclusion",
        )

    def selected_mos_models(self) -> Tuple[str, ...]:
        """保留既有 public facade，并委托纯策略簿。"""
        return self._selection_policy_book.selected_mos_models()

    def _validate_mos_assets(self) -> None:
        """按所有 baseline/exact-source selected model 并集校验本地资产。"""
        from ..processors.mos import normalize_urgent_pk_score_range
        from ..processors.optional_runtime import LocalModelAsset, verify_local_model_asset

        selected_models = set(self.selected_mos_models())

        if "urgent_pk" in selected_models:
            urgent_pk_range = normalize_urgent_pk_score_range(self.urgent_pk_score_range)
            if urgent_pk_range is None:
                raise ValueError("启用 urgent_pk 时必须显式配置 urgent_pk_score_range")
            for field, value, hex_len in (
                ("urgent_pk_expected_sha256", self.urgent_pk_expected_sha256, 64),
                ("urgent_pk_expected_commit", self.urgent_pk_expected_commit, 40),
            ):
                if value is not None and str(value).strip():
                    normalized = str(value).strip()
                    if len(normalized) != hex_len or any(
                        c not in "0123456789abcdefABCDEF" for c in normalized
                    ):
                        raise ValueError(
                            f"{field} 必须是 {hex_len} 位十六进制字符串，got {value!r}"
                        )
            required = {
                "urgent_pk_checkpoint": self.urgent_pk_checkpoint,
                "urgent_pk_source_dir": self.urgent_pk_source_dir,
                "urgent_pk_encoder": self.urgent_pk_encoder,
                "urgent_pk_backbone": self.urgent_pk_backbone,
            }
            for field, value in required.items():
                if not value:
                    raise ValueError(f"启用 urgent_pk 时必须配置 {field}")
            if not os.path.isfile(self.urgent_pk_checkpoint):
                raise ValueError(
                    f"urgent_pk_checkpoint 不存在或不是文件: {self.urgent_pk_checkpoint!r}"
                )
            if not os.path.isdir(self.urgent_pk_source_dir):
                raise ValueError(
                    f"urgent_pk_source_dir 不存在或不是目录: {self.urgent_pk_source_dir!r}"
                )
        if "scoreq" in selected_models and (
            not self.scoreq_model_dir or not os.path.isdir(self.scoreq_model_dir)
        ):
            raise ValueError(
                f"启用 scoreq 时必须配置有效的 scoreq_model_dir: {self.scoreq_model_dir!r}"
            )
        if "dnsmos" in selected_models and (
            not self.dnsmos_model_dir or not os.path.isdir(self.dnsmos_model_dir)
        ):
            raise ValueError(
                f"启用 dnsmos 时必须配置有效的 dnsmos_model_dir: {self.dnsmos_model_dir!r}"
            )

        for model in ("nisqa", "utmos"):
            if model not in selected_models:
                continue
            path = getattr(self, f"{model}_model_path")
            version = getattr(self, f"{model}_model_version")
            sha256 = getattr(self, f"{model}_model_sha256")
            if not path or not version or not sha256:
                raise ValueError(
                    f"启用 {model} 时必须配置 {model}_model_path/version/sha256"
                )
            canonical_path = os.path.realpath(os.path.abspath(path))
            if path != canonical_path or not os.path.isfile(path):
                raise ValueError(
                    f"{model}_model_path 必须是 canonical existing file: {path!r}"
                )
            asset_root = os.path.dirname(canonical_path)
            verify_local_model_asset(LocalModelAsset(
                local_path=asset_root,
                version=version,
                sha256=sha256,
            ))

    def to_dict(self) -> Dict:
        """导出 generation-3 canonical serialized view。"""
        serialized = {
            "pipeline_generation": 3,
            "step": self.step,
            "output_root": self.output_root,
            "output_sample_rate_hz": self.output_sample_rate_hz,
            "datasets": self._serialized_datasets(),
            "selection_policies": self._serialized_selection_policies(),
            "speaker_consistency": _canonical_policy_value(
                self._speaker_consistency_snapshot
            ),
            "step3_gpu_ids": list(self.step3_gpu_ids),
            "step3_worker_idle_timeout_seconds": (
                self.step3_worker_idle_timeout_seconds
            ),
            "step3_worker_hard_timeout_seconds": (
                self.step3_worker_hard_timeout_seconds
            ),
            "step3_worker_stderr_tail_bytes": self.step3_worker_stderr_tail_bytes,
            "step6_mode": self.step6_mode,
            "step4_mode": self.step4_mode,
            "step4_export_root": self.step4_export_root,
            "step6_export_root": self.step6_export_root,
            "audio_export_workers": self.audio_export_workers,
            "audio_export_max_pending": self.audio_export_max_pending,
            "audio_export_max_inflight_bytes": self.audio_export_max_inflight_bytes,
            "step1_vad_workers": self.step1_vad_workers,
            "step1_vad_max_pending": self.step1_vad_max_pending,
            "step2_max_segment_duration_sec": self.step2_max_segment_duration_sec,
            "step2_min_segment_duration_sec": self.step2_min_segment_duration_sec,
            "urgent_pk_checkpoint": getattr(self, "urgent_pk_checkpoint", None),
            "urgent_pk_source_dir": getattr(self, "urgent_pk_source_dir", None),
            "urgent_pk_encoder": getattr(self, "urgent_pk_encoder", None),
            "urgent_pk_backbone": getattr(self, "urgent_pk_backbone", None),
            "urgent_pk_score_range": getattr(self, "urgent_pk_score_range", None),
            "urgent_pk_expected_sha256": getattr(self, "urgent_pk_expected_sha256", None),
            "urgent_pk_expected_commit": getattr(self, "urgent_pk_expected_commit", None),
            "scoreq_model_dir": getattr(self, "scoreq_model_dir", None),
            "dnsmos_model_dir": getattr(self, "dnsmos_model_dir", None),
            "nisqa_model_path": getattr(self, "nisqa_model_path", None),
            "nisqa_model_version": getattr(self, "nisqa_model_version", None),
            "nisqa_model_sha256": getattr(self, "nisqa_model_sha256", None),
            "utmos_model_path": getattr(self, "utmos_model_path", None),
            "utmos_model_version": getattr(self, "utmos_model_version", None),
            "utmos_model_sha256": getattr(self, "utmos_model_sha256", None),
        }
        if self.min_source_sample_rate_hz is not None:
            serialized["min_source_sample_rate_hz"] = self.min_source_sample_rate_hz
        return serialized
