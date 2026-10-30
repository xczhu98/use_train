"""Generation-3 Step6 metadata-only multirate export routing."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any, Optional

from ...config import PipelineConfig
from ..mos import normalize_selectable_mos_models
from ...utils.artifacts import ArtifactIndex, PipelineStep
from ...utils.identity_index import unique_identity_index
from ...utils.speaker_metadata import validate_source_relative_path
from .base import ArtifactStage, StageRunContext
from .mos_filter import passed_models
from .select import STEP3_ROW_FIELDS, Step3MosSelectionStage


_STANDARD_SAMPLE_RATES_HZ = (8_000, 16_000, 24_000, 32_000, 44_100, 48_000)
_MINIMUM_WIDEBAND_TARGET_SAMPLE_RATE_HZ = 16_000
EXPORT_ROUTING_VERSION = 3
EXPORT_MANIFEST_SCHEMA = "audio_filter_pipeline.export_manifest.v2"
EXPORT_MANIFEST_FILENAME = "_export_manifest.v2.jsonl"
EXPORT_TRANSACTION_SCHEMA = "audio_filter_pipeline.audio_export_transaction.v2"
EXPORT_LAYOUT_VERSION = 3
EXPORT_LAYOUT_TEMPLATE = (
    "{step}/{language}/{dataset_id}/{artifact_digest}/{manifest_or_ratehz_path}"
)
_EFFECTIVE_BANDWIDTH_RULE = {
    "version": 3,
    "formula": "0 if bandwidth_hz == 0 else min(bandwidth_hz, estimated_sr / 2, source_sample_rate / 2)",
    "zero_bandwidth_estimated_sr": None,
    "target_rule": "source_below_16000_native_else_lowest_standard_rate_at_least_max_16000_and_twice_effective_bandwidth_without_upsampling",
    "minimum_target_sample_rate_hz": _MINIMUM_WIDEBAND_TARGET_SAMPLE_RATE_HZ,
    "source_below_minimum_rule": "native",
    "upsampling_allowed": False,
    "standard_sample_rates_hz": list(_STANDARD_SAMPLE_RATES_HZ),
}
# Step5 的正式行契约由 Step3 全字段追加唯一 speaker_shadow 字段。
STEP5_ROW_FIELDS = STEP3_ROW_FIELDS + ("speaker_shadow",)
STEP6_APPENDED_FIELDS = (
    "source_sample_rate",
    "segment_sample_rate",
    "effective_bandwidth_hz",
    "target_sample_rate",
    "final_selection",
    "export_routing",
)
STEP6_ROW_FIELDS = STEP5_ROW_FIELDS + STEP6_APPENDED_FIELDS
_FINAL_SELECTION_VERSION = 3
_ENROLLMENT_ALGORITHM = "l2_normalized_segment_embedding_to_frozen_enrollment_cosine_v1"
_UNAVAILABLE_SHADOW_STATUS = "unavailable"
logger = logging.getLogger(__name__)


def _finite_nonnegative_real(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a non-bool real")
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite nonnegative real") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{field} must be a finite nonnegative real")
    return value


def target_sample_rate_for_effective_bandwidth(
    effective_bandwidth_hz: float,
    source_sample_rate: int,
) -> int:
    """按 16 kHz 下限路由宽带源，低采样率源保持 native 且不升采样。"""
    effective_bandwidth_hz = _finite_nonnegative_real(
        effective_bandwidth_hz, "effective_bandwidth_hz"
    )
    if type(source_sample_rate) is not int or source_sample_rate <= 0:
        raise ValueError("source_sample_rate must be a positive Python integer")
    if effective_bandwidth_hz > source_sample_rate / 2:
        raise ValueError("effective_bandwidth_hz exceeds source Nyquist frequency")
    if source_sample_rate < _MINIMUM_WIDEBAND_TARGET_SAMPLE_RATE_HZ:
        return source_sample_rate
    required_sample_rate = max(
        2.0 * effective_bandwidth_hz,
        _MINIMUM_WIDEBAND_TARGET_SAMPLE_RATE_HZ,
    )
    for candidate in _STANDARD_SAMPLE_RATES_HZ:
        if candidate <= source_sample_rate and candidate >= required_sample_rate:
            return candidate
    return source_sample_rate


def _validate_dataset_and_source_identity(
    row: Mapping[str, Any],
    dataset_id: str,
    config: PipelineConfig | None,
    *,
    label: str,
    require_original_relative_path: bool = False,
) -> None:
    """校验 DatasetSpec identity 与不可变 source descriptor 的绑定。"""
    if row.get("dataset_id") != dataset_id:
        raise ValueError(f"{label} dataset_id does not match context")
    for field in (
        "language", "source_key", "segment_id", "input_root", "original_path",
        "source_relative_path", "format", "subtype",
    ):
        if type(row.get(field)) is not str or not row[field].strip():
            raise ValueError(f"{label} {field} must be a nonblank exact string")
    for field in ("category", "quality_profile"):
        if field not in row:
            raise ValueError(f"{label} is missing required identity field {field}")
    if row["category"] is not None and (
        type(row["category"]) is not str or not row["category"].strip()
    ):
        raise ValueError(f"{label} category must be None or a nonblank exact string")
    if type(row["quality_profile"]) is not str or not row["quality_profile"].strip():
        raise ValueError(f"{label} quality_profile must be a nonblank exact string")
    if config is not None:
        spec = config.dataset_spec(dataset_id)
        if row["language"] != spec.language:
            raise ValueError(f"{label} language does not exactly match DatasetSpec")
        if row["category"] != spec.category:
            raise ValueError(f"{label} category does not exactly match DatasetSpec")
        if row["quality_profile"] != spec.quality_profile:
            raise ValueError(f"{label} quality_profile does not exactly match DatasetSpec")
        expected_root = os.path.realpath(os.path.abspath(spec.input_root))
        actual_root = os.path.realpath(os.path.abspath(row["input_root"]))
        if actual_root != expected_root:
            raise ValueError(f"{label} input_root does not exactly match DatasetSpec")
    else:
        actual_root = os.path.realpath(os.path.abspath(row["input_root"]))
    if row["input_root"] != actual_root:
        raise ValueError(f"{label} input_root must be canonical")
    actual_path = os.path.realpath(os.path.abspath(row["original_path"]))
    if row["original_path"] != actual_path:
        raise ValueError(f"{label} original_path must be canonical")
    relative_path = validate_source_relative_path(row["source_relative_path"])
    # 仅通过冻结 metadata 的相互绑定判型，源文件删除后仍可完成验证。
    single_file_relative = actual_root.replace(os.sep, "/").lstrip("/")
    single_file_binding = (
        relative_path == single_file_relative and actual_path == actual_root
    )
    directory_expected_path = os.path.realpath(os.path.join(actual_root, relative_path))
    directory_binding = actual_path == directory_expected_path
    if single_file_binding == directory_binding:
        raise ValueError(
            f"{label} original_path is not unambiguously bound to "
            "input_root/source_relative_path"
        )
    single_file_input = single_file_binding
    if require_original_relative_path:
        original_relative_path = row.get("original_relative_path")
        if type(original_relative_path) is not str:
            raise ValueError(f"{label} original_relative_path is invalid")
        if original_relative_path:
            components = original_relative_path.split("/")
            if (
                "\\" in original_relative_path
                or "\x00" in original_relative_path
                or original_relative_path.startswith("/")
                or any(component in ("", ".", "..") for component in components)
            ):
                raise ValueError(f"{label} original_relative_path is invalid")
        expected_original_relative_path = (
            ""
            if single_file_input
            else relative_path.rsplit("/", 1)[0] if "/" in relative_path else ""
        )
        if original_relative_path != expected_original_relative_path:
            raise ValueError(f"{label} original_relative_path is not bound to source_relative_path")

    for field in ("source_size_bytes", "sample_rate", "frames", "channels", "start_sample", "end_sample"):
        if type(row.get(field)) is not int:
            raise ValueError(f"{label} {field} must be a Python integer")
    if row["source_size_bytes"] < 0 or row["sample_rate"] <= 0 or row["frames"] <= 0 or row["channels"] <= 0:
        raise ValueError(f"{label} source descriptor header must be valid")
    if (type(row["source_sha256"]) is not str or len(row["source_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in row["source_sha256"])):
        raise ValueError(f"{label} source_sha256 must be lowercase SHA-256 hex")
    if not 0 <= row["start_sample"] < row["end_sample"] <= row["frames"]:
        raise ValueError(f"{label} segment sample interval must be nonempty and half-open")
    duration = row.get("duration_sec")
    if isinstance(duration, bool) or not isinstance(duration, Real) or not math.isfinite(float(duration)):
        raise ValueError(f"{label} duration_sec must be finite")
    if float(duration) != row["frames"] / row["sample_rate"]:
        raise ValueError(f"{label} duration_sec must exactly match frames / sample_rate")


def _validate_step5_row(
    row: Mapping[str, Any],
    dataset_id: str,
    config: PipelineConfig | None = None,
) -> None:
    if not isinstance(row, Mapping):
        raise ValueError("Step5 upstream row must be a mapping")
    if set(row) != set(STEP5_ROW_FIELDS):
        raise ValueError("Step5 upstream row fields do not match the exact contract")
    _validate_dataset_and_source_identity(
        row, dataset_id, config, label="Step5 upstream row",
        require_original_relative_path=True,
    )
    for field in ("format", "subtype"):
        if not row[field].strip():
            raise ValueError(f"{field} must be a nonblank exact Python string")
    bandwidth_hz = _finite_nonnegative_real(row["bandwidth_hz"], "bandwidth_hz")
    if bandwidth_hz > row["sample_rate"] / 2:
        raise ValueError("bandwidth_hz exceeds source Nyquist frequency")
    estimated_sr = row["estimated_sr"]
    if bandwidth_hz == 0.0:
        if estimated_sr is not None:
            raise ValueError("estimated_sr must be None when bandwidth_hz is zero")
    elif type(estimated_sr) is not int or not 0 < estimated_sr <= row["sample_rate"]:
        raise ValueError(
            "estimated_sr must be a positive Python integer when bandwidth_hz is nonzero"
        )
    # Step6 独立消费 Step3 分数执行 MOS 筛选（与 Step4 解耦）。
    models = normalize_selectable_mos_models(row["mos_models"])
    Step3MosSelectionStage._validate_scores(row["mos_scores"], models)
    if config is not None:
        policy = config.resolve_selection_policy(
            row["dataset_id"],
            row["language"],
            source_relative_path=row["source_relative_path"],
        )
        projection = config.mos_scoring_projection(
            row["dataset_id"],
            row["language"],
            source_relative_path=row["source_relative_path"],
        )
        if (
            models != policy.mos_models
            or row["mos_scoring_policy_hash"] != projection["scoring_policy_hash"]
        ):
            raise ValueError("Step5 upstream row MOS scoring projection is stale")
    shadow = row["speaker_shadow"]
    if not isinstance(shadow, Mapping) or type(shadow.get("status")) is not str:
        raise ValueError("Step5 upstream row speaker_shadow status is invalid")


def _effective_bandwidth_hz(row: Mapping[str, Any]) -> float:
    if row["bandwidth_hz"] == 0.0:
        return 0.0
    return min(
        float(row["bandwidth_hz"]),
        row["estimated_sr"] / 2.0,
        row["sample_rate"] / 2.0,
    )


def _final_selection(row: Mapping[str, Any], config: PipelineConfig) -> Mapping[str, Any]:
    """按 Step6 模式独立执行 MOS 筛选并联合声纹结果生成唯一导出依据。

    Step6 直接消费 Step3 分数按当前动态 K/N 阈值筛选（与 Step4 解耦：
    各自独立重建，改阈值不触发对方重算）。
    """
    policy = config.resolve_selection_policy(
        row["dataset_id"],
        row["language"],
        source_relative_path=row["source_relative_path"],
    )
    passed = passed_models(row["mos_scores"], policy)
    if passed < policy.min_mos_models:
        return {"version": _FINAL_SELECTION_VERSION, "export_eligible": False, "reason": "mos_rejected"}
    speaker_policy = config.speaker_consistency
    shadow = row["speaker_shadow"]
    if not isinstance(shadow, Mapping) or type(shadow.get("status")) is not str:
        raise ValueError("Step5 upstream row speaker_shadow status is invalid")
    if not speaker_policy.get("enabled", False):
        # 声纹未启用：只按 MOS 过滤（Step6 独立筛选通过即导出）。
        return {"version": _FINAL_SELECTION_VERSION, "export_eligible": True, "reason": "mos_selected"}
    if shadow["status"] != "computed":
        # 声纹已启用：只有完成 enrollment cosine 验证（status=computed）的行才可导出；
        # not_trial / disabled / not_enrolled / unavailable / insufficient_group /
        # unsupported / 未知状态一律拒绝，绝不按 MOS 静默放行。
        return {"version": _FINAL_SELECTION_VERSION, "export_eligible": False, "reason": "speaker_shadow_not_exportable"}
    # dry_run 使用候选阈值计算与 filter 完全一致的联合保留统计，但绝不导出音频。
    if config.step6_mode == "dry_run" and "min_cosine_similarity" not in speaker_policy:
        return {"version": _FINAL_SELECTION_VERSION, "export_eligible": True, "reason": "mos_selected"}
    similarity = shadow.get("cosine_similarity_to_enrollment")
    if isinstance(similarity, bool) or not isinstance(similarity, Real) or not math.isfinite(float(similarity)):
        raise ValueError("computed speaker_shadow cosine_similarity_to_enrollment is invalid")
    provenance = shadow.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("computed speaker_shadow provenance is invalid")
    if provenance.get("algorithm") != _ENROLLMENT_ALGORITHM:
        raise ValueError("computed speaker_shadow must use frozen enrollment algorithm")
    if provenance.get("enrollment_bundle_sha256") != speaker_policy["enrollment_bundle_sha256"]:
        raise ValueError("computed speaker_shadow enrollment bundle does not match configuration")
    threshold = speaker_policy["min_cosine_similarity"]
    if float(similarity) < threshold:
        return {
            "version": _FINAL_SELECTION_VERSION,
            "export_eligible": False,
            "reason": "speaker_inconsistent",
        }
    return {
        "version": _FINAL_SELECTION_VERSION,
        "export_eligible": True,
        "reason": "speaker_consistent",
    }


class Step6MultirateExportStage(ArtifactStage):
    """发布 Step5 行的多采样率导出路由，不读取或写入音频。

    Step6 独立消费 Step3 分数执行 MOS 筛选；仅当前数据集可达策略进入
    config digest，与 Step4 的 SE 筛选完全解耦。
    """

    step = PipelineStep.STEP6
    upstream_step = PipelineStep.STEP5

    def __init__(self, *, dataset_id: str | None = None) -> None:
        if dataset_id is not None and (
            type(dataset_id) is not str or not dataset_id.strip()
        ):
            raise ValueError("Step6 dataset_id must be a non-empty string or None")
        self._dataset_id = dataset_id

    def _validate_contract(self, context: StageRunContext) -> None:
        super()._validate_contract(context)
        if self._dataset_id is not None and self._dataset_id != context.dataset_id:
            raise ValueError("Step6 stage dataset_id does not match context")

    def config_payload(self, config: PipelineConfig) -> Mapping[str, Any]:
        return {
            "row_contract": {
                "version": 5,
                "fields": list(STEP6_ROW_FIELDS),
            },
            "upstream_row_contract": {
                "version": 5,
                "fields": list(STEP5_ROW_FIELDS),
            },
            "effective_bandwidth_rule": _EFFECTIVE_BANDWIDTH_RULE,
            "export": {
                "routing_version": EXPORT_ROUTING_VERSION,
                "manifest_schema": EXPORT_MANIFEST_SCHEMA,
                "manifest_filename": EXPORT_MANIFEST_FILENAME,
                "transaction_schema": EXPORT_TRANSACTION_SCHEMA,
                "layout_version": EXPORT_LAYOUT_VERSION,
                "layout_template": EXPORT_LAYOUT_TEMPLATE,
                "mode": "metadata_only",
                "audio_read": False,
                "audio_write": False,
                **(
                    {"configured_target_sample_rate_hz": config.output_sample_rate_hz}
                    if config.output_sample_rate_hz is not None
                    else {}
                ),
                "minimum_target_sample_rate_hz": _MINIMUM_WIDEBAND_TARGET_SAMPLE_RATE_HZ,
                "source_below_minimum_rule": "native",
                "upsampling_allowed": False,
            },
            "artifact_bound_export": {
                "producer_identity_fields": [
                    "step", "dataset_id", "config_digest", "lineage_digest",
                    "rows_sha256", "artifact_digest",
                ],
            },
            "final_selection": {
                "version": _FINAL_SELECTION_VERSION,
                "mode": config.step6_mode,
                "min_cosine_similarity": config.speaker_consistency.get("min_cosine_similarity"),
                "speaker_shadow_algorithm": _ENROLLMENT_ALGORITHM,
                "enrollment_bundle_sha256": config.speaker_consistency.get("enrollment_bundle_sha256"),
            },
            **(
                {"selection_policies": config.to_dict()["selection_policies"]}
                if self._dataset_id is None
                else {"selection": {
                    "resolver": "PipelineConfig.dataset_mos_decision_binding",
                    "dataset_binding": config.dataset_mos_decision_binding(
                        self._dataset_id
                    ),
                }}
            ),
        }

    def produce_rows(
        self,
        context: StageRunContext,
        upstream_index: Optional[ArtifactIndex],
        upstream_rows: Iterable[Mapping[str, Any]],
    ) -> Iterable[Mapping[str, Any]]:
        if upstream_index is None or upstream_index.step != PipelineStep.STEP5.value:
            raise ValueError("Step6 export requires a Step5 upstream artifact")
        unavailable_shadow_rows = 0
        produced_rows = 0
        # 重复检测走临时 sqlite：流式设计下不累积全量 segment_id。
        with unique_identity_index() as reserve_identity:
            for upstream_row in upstream_rows:
                if not isinstance(upstream_row, Mapping):
                    raise ValueError("Step5 upstream row must be a mapping")
                _validate_step5_row(upstream_row, context.dataset_id, context.config)
                segment_id = upstream_row["segment_id"]
                if not reserve_identity("segment_id", segment_id):
                    raise ValueError(f"duplicate segment_id in Step6 artifact: {segment_id!r}")
                shadow = upstream_row["speaker_shadow"]
                if (isinstance(shadow, Mapping)
                        and shadow.get("status") == _UNAVAILABLE_SHADOW_STATUS):
                    unavailable_shadow_rows += 1
                produced_rows += 1
                effective_bandwidth_hz = _effective_bandwidth_hz(upstream_row)
                target_sample_rate = context.config.output_sample_rate_hz
                if target_sample_rate is None:
                    target_sample_rate = target_sample_rate_for_effective_bandwidth(
                        effective_bandwidth_hz, upstream_row["sample_rate"]
                    )
                if target_sample_rate > upstream_row["sample_rate"]:
                    raise ValueError("output_sample_rate_hz must not upsample any selected dataset")
                output_row = {
                    **{field: upstream_row[field] for field in STEP5_ROW_FIELDS},
                    "source_sample_rate": upstream_row["sample_rate"],
                    "segment_sample_rate": upstream_row["sample_rate"],
                    "effective_bandwidth_hz": effective_bandwidth_hz,
                    "target_sample_rate": target_sample_rate,
                    "final_selection": _final_selection(upstream_row, context.config),
                    "export_routing": {
                        "version": EXPORT_ROUTING_VERSION,
                        "layout_version": EXPORT_LAYOUT_VERSION,
                        "layout_template": EXPORT_LAYOUT_TEMPLATE,
                        "manifest_schema": EXPORT_MANIFEST_SCHEMA,
                        "manifest_filename": EXPORT_MANIFEST_FILENAME,
                        "transaction_schema": EXPORT_TRANSACTION_SCHEMA,
                        "mode": "metadata_only",
                        "audio_write": False,
                        **(
                            {"configured_target_sample_rate_hz": target_sample_rate}
                            if context.config.output_sample_rate_hz is not None
                            else {}
                        ),
                        "minimum_target_sample_rate_hz": _MINIMUM_WIDEBAND_TARGET_SAMPLE_RATE_HZ,
                        "source_below_minimum_rule": "native",
                        "upsampling_allowed": False,
                        "source_path": upstream_row["original_path"],
                        "segment_interval_samples": [
                            upstream_row["start_sample"], upstream_row["end_sample"],
                        ],
                    },
                }
                yield output_row
        if (
            produced_rows
            and unavailable_shadow_rows == produced_rows
            and context.config.step6_mode == "filter"
            and context.config.speaker_consistency.get("enabled", False)
        ):
            # 声纹 runtime 不可用（Step5 无 extractor 或 embedding runtime 失败）时
            # Step5 逐行降级为 unavailable，Step6 会把整份 artifact 判为不可导出；
            # filter 模式下发不出任何音频，必须 fail-closed 而不是正常返回 0 导出。
            raise ValueError(
                "speaker shadow is unavailable for every Step5 row; refusing to "
                "publish a filter-mode export without any speaker-verified row"
            )
        if unavailable_shadow_rows:
            logger.warning(
                "Step6 dataset=%s speaker shadow unavailable for %d/%d rows; "
                "these rows cannot be exported",
                context.dataset_id, unavailable_shadow_rows, produced_rows,
            )
