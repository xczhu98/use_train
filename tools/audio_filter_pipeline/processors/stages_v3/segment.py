"""Artifact-backed Step2 continuous source-domain segmentation."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any, Optional

from ...config import PipelineConfig
from ...utils.algorithm_identity import analyzer_algorithm_payload
from ...utils.artifacts import ArtifactIndex, PipelineStep
from ...utils.bandwidth import get_optimal_sample_rate
from ...utils.identity_index import unique_identity_index
from ..segmentation import (
    SegmentBoundary,
    _validate_stable_silence_timeline,
    _validate_timeline,
    plan_single_speaker_segments,
)
from .analyze import STEP1_ROW_FIELDS
from .base import ArtifactStage, StageRunContext


STEP2_ROW_FIELDS: tuple[str, ...] = (
    "source_key", "dataset_id", "language", "category", "quality_profile",
    "input_root", "original_path", "original_relative_path",
    "source_relative_path", "source_size_bytes", "source_sha256",
    "sample_rate", "frames", "channels", "duration_sec", "format", "subtype",
    "speaker_id", "speaker_scope",
    "source_start_sample", "source_end_sample", "signal_finite",
    "signal_zero_ratio", "signal_peak", "signal_clipping_ratio",
    "signal_dc_offset", "signal_rms_dbfs", "signal_num_samples",
    "bandwidth_hz", "estimated_sr", "vad_speech_ratio",
    "vad_timeline_sample_rate", "vad_timeline_num_samples",
    "segment_id", "start_sample", "end_sample", "boundary_reason",
)
_GENERATED_FIELDS = {
    "segment_id", "start_sample", "end_sample", "boundary_reason"
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
# 超过 step2_max_segment_duration_sec 的连续不可切段：planner 保留覆盖边界，
# stage 不为它产生 segment 行（不打分、不导出）。
_SKIPPED_OVERLONG_REASON = "skipped_overlong"
# 低于 step2_min_segment_duration_sec 的段：planner 保留覆盖边界，
# stage 不为它产生 segment 行（不打分、不导出）。
_SKIPPED_SHORT_REASON = "skipped_short"

logger = logging.getLogger(__name__)


def _step2_row(**values: Any) -> dict[str, Any]:
    if set(values) != set(STEP2_ROW_FIELDS):
        raise RuntimeError("Step2 row fields do not match the declared row contract")
    return {field: values[field] for field in STEP2_ROW_FIELDS}


def _segment_id(row: Mapping[str, Any], boundary: SegmentBoundary) -> str:
    identity = [
        "audio_filter_pipeline.segment",
        1,
        row["dataset_id"],
        row["source_key"],
        row["source_sha256"],
        row["sample_rate"],
        boundary.start_sample,
        boundary.end_sample,
    ]
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_fact(row: Mapping[str, Any], field: str) -> float:
    value = row[field]
    if type(value) is not float:
        raise ValueError(
            f"Step1 upstream row {field} must be a non-bool real exact Python float"
        )
    if not math.isfinite(value):
        raise ValueError(f"Step1 upstream row {field} must be finite")
    return value


def _validate_step1_row(
    row: Mapping[str, Any],
    config: PipelineConfig,
    dataset_id: str,
) -> None:
    """完整验证直接前代 Step1 行，避免篡改事实进入 planner。"""

    if set(row) != set(STEP1_ROW_FIELDS):
        raise ValueError("Step1 upstream row fields do not match the exact contract")
    dataset = config.dataset_spec(dataset_id)
    if type(row["dataset_id"]) is not str or row["dataset_id"] != dataset_id:
        raise ValueError("Step1 upstream row dataset_id does not match context")
    for field, expected in (
        ("language", dataset.language),
        ("category", dataset.category),
        ("quality_profile", dataset.quality_profile),
        ("input_root", dataset.input_root),
    ):
        if type(row[field]) is not type(expected) or row[field] != expected:
            raise ValueError(
                f"Step1 upstream row {field} does not match dataset config identity"
            )
    for field in ("source_key", "original_path", "source_relative_path"):
        if type(row[field]) is not str or not row[field].strip():
            raise ValueError(
                f"Step1 upstream row {field} must be an exact nonblank string"
            )
    if type(row["original_relative_path"]) is not str:
        raise ValueError(
            "Step1 upstream row original_relative_path must be an exact string"
        )

    source_size = row["source_size_bytes"]
    if type(source_size) is not int or source_size < 0:
        raise ValueError("source_size_bytes must be a nonnegative Python integer")
    source_sha = row["source_sha256"]
    if type(source_sha) is not str or _SHA256_RE.fullmatch(source_sha) is None:
        raise ValueError("source_sha256 must be lowercase SHA-256 hex")
    for field in ("sample_rate", "frames", "channels"):
        value = row[field]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field} must be a positive Python integer")
    for field in ("format", "subtype"):
        value = row[field]
        if type(value) is not str or not value.strip():
            raise ValueError(f"{field} must be a nonblank exact Python string")

    duration = _finite_fact(row, "duration_sec")
    if duration != row["frames"] / row["sample_rate"]:
        raise ValueError("duration_sec must exactly match frames / sample_rate")
    start = row["source_start_sample"]
    end = row["source_end_sample"]
    if type(start) is not int or start != 0:
        raise ValueError("source_start_sample must be exact integer zero")
    if type(end) is not int or end != row["frames"]:
        raise ValueError("source_end_sample must exactly equal frames")

    if row["signal_finite"] is not True:
        raise ValueError("Step1 upstream row signal_finite must be exact true")
    zero_ratio = _finite_fact(row, "signal_zero_ratio")
    peak = _finite_fact(row, "signal_peak")
    clipping_ratio = _finite_fact(row, "signal_clipping_ratio")
    _finite_fact(row, "signal_dc_offset")
    if not 0.0 <= zero_ratio <= 1.0:
        raise ValueError("Step1 upstream row signal_zero_ratio must be in [0, 1]")
    if peak < 0.0:
        raise ValueError("Step1 upstream row signal_peak must be nonnegative")
    if not 0.0 <= clipping_ratio <= 1.0:
        raise ValueError("Step1 upstream row signal_clipping_ratio must be in [0, 1]")
    rms_dbfs = row["signal_rms_dbfs"]
    if rms_dbfs is None:
        if peak != 0.0:
            raise ValueError("signal_rms_dbfs may be None only when signal_peak is zero")
    else:
        _finite_fact(row, "signal_rms_dbfs")
        if peak == 0.0:
            raise ValueError("zero signal_peak requires signal_rms_dbfs None")
    if (
        type(row["signal_num_samples"]) is not int
        or row["signal_num_samples"] != row["frames"] * row["channels"]
    ):
        raise ValueError("signal_num_samples must exactly equal frames * channels")

    bandwidth_hz = _finite_fact(row, "bandwidth_hz")
    if not 0.0 <= bandwidth_hz <= row["sample_rate"] / 2:
        raise ValueError("bandwidth_hz must be within source Nyquist bounds")
    estimated_sr = row["estimated_sr"]
    if bandwidth_hz == 0.0:
        if estimated_sr is not None:
            raise ValueError("zero bandwidth_hz requires estimated_sr None")
    elif (
        type(estimated_sr) is not int
        or estimated_sr != get_optimal_sample_rate(bandwidth_hz, row["sample_rate"])
    ):
        raise ValueError("estimated_sr does not match bandwidth_hz and sample_rate")

    vad_speech_ratio = _finite_fact(row, "vad_speech_ratio")
    if not 0.0 <= vad_speech_ratio <= 1.0:
        raise ValueError("vad_speech_ratio must be within [0, 1]")
    if row["vad_timeline_sample_rate"] != 16_000:
        raise ValueError("vad_timeline_sample_rate must be exactly 16000")
    vad_num_samples = row["vad_timeline_num_samples"]
    if type(vad_num_samples) is not int or vad_num_samples <= 0:
        raise ValueError("vad_timeline_num_samples must be a positive Python integer")
    scaled = row["frames"] * 16_000
    if row["sample_rate"] == 16_000:
        valid_vad_length = vad_num_samples == row["frames"]
    else:
        minimum = max(1, scaled // row["sample_rate"])
        maximum = (scaled + row["sample_rate"] - 1) // row["sample_rate"] + 1
        valid_vad_length = minimum <= vad_num_samples <= maximum
    if not valid_vad_length:
        raise ValueError("vad_timeline_num_samples does not match source duration")
    _validate_timeline(row["vad_timeline"], vad_num_samples)
    _validate_stable_silence_timeline(
        row["stable_silence_timeline"], row["frames"]
    )

    speaker_id = row["speaker_id"]
    speaker_scope = row["speaker_scope"]
    unknown = speaker_id is None and speaker_scope is None
    explicit = (
        type(speaker_id) is str
        and bool(speaker_id.strip())
        and type(speaker_scope) is str
        and speaker_scope == dataset_id
    )
    if not (unknown or explicit):
        raise ValueError("speaker_id and speaker_scope are not a valid pair")


def _validate_boundaries(
    boundaries: object,
    source_num_samples: int,
) -> tuple[SegmentBoundary, ...]:
    if not isinstance(boundaries, tuple):
        raise TypeError("planner boundaries must be a tuple")
    if not boundaries:
        raise ValueError("planner must return at least one segment")
    previous_end = None
    for index, boundary in enumerate(boundaries):
        if not isinstance(boundary, SegmentBoundary):
            raise TypeError("planner boundary must be a SegmentBoundary")
        if not 0 <= boundary.start_sample < boundary.end_sample <= source_num_samples:
            raise ValueError("planner boundary is outside source bounds")
        if previous_end is not None and boundary.start_sample != previous_end:
            raise ValueError("planner boundaries must be contiguous")
        if boundary.boundary_reason == "no_stable_silence" and (
            index != len(boundaries) - 1
            or boundary.end_sample != source_num_samples
        ):
            raise ValueError(
                "no_stable_silence boundary must be terminal and cover source end"
            )
        previous_end = boundary.end_sample
    if boundaries[0].start_sample != 0:
        raise ValueError("planner boundaries must start at zero")
    if boundaries[-1].end_sample != source_num_samples:
        raise ValueError("planner boundaries must cover source end")
    return boundaries


class Step2SegmentStage(ArtifactStage):
    """Publish continuous source-domain Step2 segment rows without waveform I/O."""

    step = PipelineStep.STEP2
    upstream_step = PipelineStep.STEP1
    # 1 个源可切出多个 segment，段数在处理前无法预知，不能以上游源数作
    # total 上限（否则进度条按输出段推进会超过 100%）；覆盖 progress_total
    # 返回 None → 外壳使用不确定模式进度条（只显示已处理段数计数）。
    progress_unit = "segment"

    def progress_total(self, context, upstream_index):
        """段数无法预知，禁用上限 → 不确定模式进度条。"""
        return None

    def config_payload(self, config: PipelineConfig) -> Mapping[str, Any]:
        return {
            "row_contract": {"version": 5, "fields": list(STEP2_ROW_FIELDS)},
            # 手工语义版本必须叠加算法源码哈希：只改算法实现而不改这里的
            # 字面量时，旧 Step2 artifact 仍会失配而重算。
            "analysis_algorithm_source": analyzer_algorithm_payload(),
            "upstream_row_contract": {
                "version": 6,
                "fields": list(STEP1_ROW_FIELDS),
                "validation": "complete_inherited_facts",
            },
            "segmentation": {
                "version": 5,
                "planner": "plan_single_speaker_segments",
                "source_domain": "original_sample_rate",
                "vad_timeline_sample_rate": 16000,
                "stable_silence_source_domain": True,
                "safe_intersection": "vad_silence_and_stable_silence",
                "guard_ms_each_side": 100,
                "target_duration_sec": 10,
                "silence_search_radius_sec": 1,
                "forward_safe_search": True,
                "hard_cut": False,
                "short_tail_sec": 2,
                "max_merged_duration_sec": 12,
                "max_segment_duration_sec": config.step2_max_segment_duration_sec,
                "min_segment_duration_sec": config.step2_min_segment_duration_sec,
                "overlong_policy": "skip_without_row_emission",
                "short_policy": "skip_without_row_emission",
                "interval": "half_open",
                "coverage": "contiguous_full_source",
                "boundary_reason_semantics": {
                    "whole": "unsplit_source_or_regular_terminal_remainder",
                    "stable_silence": "ends_at_guarded_vad_and_stable_silence_cut",
                    "no_stable_silence": "terminal_remainder_without_future_safe_cut",
                    "tail_merged": "short_terminal_remainder_merged_with_previous",
                    "short_tail_unmerged": "short_terminal_remainder_exceeds_merge_cap",
                    "skipped_overlong": "stretch_exceeds_max_segment_duration_sec_not_emitted",
                    "skipped_short": "stretch_below_min_segment_duration_sec_not_emitted",
                },
            },
            "segment_id": {
                "version": 1,
                "algorithm": "sha256_canonical_json_array",
            },
            "waveform_policy": {
                "read": False,
                "write": False,
                "trim": False,
                "concatenate": False,
            },
        }

    def produce_rows(
        self,
        context: StageRunContext,
        upstream_index: Optional[ArtifactIndex],
        upstream_rows: Iterable[Mapping[str, Any]],
    ) -> Iterable[Mapping[str, Any]]:
        if upstream_index is None or upstream_index.step != PipelineStep.STEP1.value:
            raise ValueError("Step2 segmentation requires a Step1 upstream artifact")
        max_segment_duration_sec = (
            context.config.step2_max_segment_duration_sec
        )
        min_segment_duration_sec = (
            context.config.step2_min_segment_duration_sec
        )
        skipped_segments = 0
        skipped_seconds = 0.0
        skipped_short_segments = 0
        skipped_short_seconds = 0.0
        # 重复检测走临时 sqlite：流式设计下不累积全量 source_key/segment_id。
        with unique_identity_index() as reserve_identity:
            for upstream_row in upstream_rows:
                if not isinstance(upstream_row, Mapping):
                    raise ValueError("Step1 upstream row must be a mapping")
                _validate_step1_row(
                    upstream_row, context.config, context.dataset_id
                )
                source_key = upstream_row["source_key"]
                if not reserve_identity("source_key", source_key):
                    raise ValueError(f"duplicate source_key in Step2 artifact: {source_key!r}")
                boundaries = plan_single_speaker_segments(
                    source_num_samples=upstream_row["frames"],
                    source_sample_rate=upstream_row["sample_rate"],
                    vad_timeline=upstream_row["vad_timeline"],
                    vad_timeline_sample_rate=upstream_row["vad_timeline_sample_rate"],
                    vad_timeline_num_samples=upstream_row["vad_timeline_num_samples"],
                    stable_silence_timeline=upstream_row["stable_silence_timeline"],
                    max_segment_duration_sec=max_segment_duration_sec,
                    min_segment_duration_sec=min_segment_duration_sec,
                )
                boundaries = _validate_boundaries(boundaries, upstream_row["frames"])
                inherited = {
                    field: upstream_row[field]
                    for field in STEP2_ROW_FIELDS
                    if field not in _GENERATED_FIELDS
                }
                for boundary in boundaries:
                    if boundary.boundary_reason == _SKIPPED_OVERLONG_REASON:
                        skipped_segments += 1
                        skipped_seconds += (
                            (boundary.end_sample - boundary.start_sample)
                            / upstream_row["sample_rate"]
                        )
                        continue
                    if boundary.boundary_reason == _SKIPPED_SHORT_REASON:
                        skipped_short_segments += 1
                        skipped_short_seconds += (
                            (boundary.end_sample - boundary.start_sample)
                            / upstream_row["sample_rate"]
                        )
                        continue
                    segment_id = _segment_id(upstream_row, boundary)
                    if not reserve_identity("segment_id", segment_id):
                        raise ValueError(
                            f"duplicate segment_id in Step2 artifact: {segment_id!r}"
                        )
                    yield _step2_row(
                        **inherited,
                        segment_id=segment_id,
                        start_sample=boundary.start_sample,
                        end_sample=boundary.end_sample,
                        boundary_reason=boundary.boundary_reason,
                    )
        if skipped_segments:
            logger.info(
                "Step2 dataset=%s skipped %d overlong segment(s), %.3fs of audio "
                "beyond max_segment_duration_sec=%s",
                context.dataset_id,
                skipped_segments,
                skipped_seconds,
                max_segment_duration_sec,
            )
        if skipped_short_segments:
            logger.info(
                "Step2 dataset=%s skipped %d short segment(s), %.3fs of audio "
                "below min_segment_duration_sec=%s",
                context.dataset_id,
                skipped_short_segments,
                skipped_short_seconds,
                min_segment_duration_sec,
            )
