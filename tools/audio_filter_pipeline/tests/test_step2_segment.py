"""Generation-3 Step2 segment stage contract tests."""

import hashlib
import json
import math
import re
from pathlib import Path

import pytest

import audio_filter_pipeline.processors.stages_v3.segment as segment_module
from audio_filter_pipeline.config import PipelineConfig
from audio_filter_pipeline.processors.segmentation import SegmentBoundary
from audio_filter_pipeline.processors.stages_v3.analyze import STEP1_ROW_FIELDS
from audio_filter_pipeline.processors.stages_v3.base import (
    ArtifactExpectation,
    StageRunContext,
)
from audio_filter_pipeline.processors.stages_v3.segment import (
    STEP2_ROW_FIELDS,
    Step2SegmentStage,
)
from audio_filter_pipeline.utils.algorithm_identity import analyzer_algorithm_payload
from audio_filter_pipeline.utils.artifacts import (
    ArtifactWriter,
    PipelineStep,
    iter_artifact_rows,
)
from audio_filter_pipeline.utils.bandwidth import get_optimal_sample_rate
from audio_filter_pipeline.utils.path_utils import v3_artifact_dir


from stage_test_helpers import runner_expectation_for_test


pytestmark = pytest.mark.unit
EXPECTED_STEP2_FIELDS = (
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
GENERATED_FIELDS = {
    "segment_id", "start_sample", "end_sample", "boundary_reason"
}
CONTRACT_FILES = {"rows.jsonl", "index.json", "_SUCCESS"}


class _StrSubclass(str):
    pass


def _quality_profiles():
    """generation-3 封闭 quality_profiles registry（Task 1 精确结构）。"""
    return {
        "quality_balanced": {"thresholds": {
            "dnsmos": {"ovrl": 3.2, "sig": 3.0, "bak": 3.8, "p808": 3.6},
            "scoreq": 3.6,
            "urgent_pk": 3.5,
        }},
        "quality_high": {"thresholds": {
            "dnsmos": {"ovrl": 3.3, "sig": 3.4, "bak": 4.0, "p808": 3.8},
            "scoreq": 4.2,
            "urgent_pk": 3.5,
        }},
        "quality_premium": {"thresholds": {
            "dnsmos": {"ovrl": 3.4, "sig": 3.6, "bak": 4.2, "p808": 4.0},
            "scoreq": 4.5,
            "urgent_pk": 3.5,
        }},
    }


def _config(output_root, *, threshold=3.0, dataset_id="alpha",
            step2_min_segment_duration_sec=0.0):
    return PipelineConfig(
        pipeline_generation=3,
        step="step2",
        output_root=str(output_root),
        step2_min_segment_duration_sec=step2_min_segment_duration_sec,
        datasets={
            dataset_id: {
                "input_root": "/definitely/missing/audio",
                "language": "en",
                "category": "read_speech",
                "quality_profile": "quality_high",
                "speaker": {"source": "none", "required": False},
            }
        },
        selection_policies={
            "global": {
                "mos_models": ["dnsmos"],
                "thresholds": {"dnsmos": threshold},
                "min_mos_models": 1,
            },
            "quality_profiles": _quality_profiles(),
        },
    )


def _context(config, upstream=None, *, dataset_id="alpha"):
    return StageRunContext(
        config.output_root, dataset_id, config, upstream=upstream
    )


def _expectation(index):
    return runner_expectation_for_test(index)


def _artifact_path(output_root, step="step2"):
    return Path(v3_artifact_dir(str(output_root), step, "alpha"))


def _assert_no_step2_or_staging(output_root):
    target = _artifact_path(output_root)
    assert not target.exists()
    if target.parent.exists():
        assert not list(target.parent.glob(".alpha.stage-*"))


def _vad_length(frames, sample_rate):
    return max(1, round(frames * 16000 / sample_rate))


def _row(
    *, source_key="source-1", frames=80000, sample_rate=16000,
    speaker_id=None, speaker_scope=None, timeline=None, stable_timeline=None,
    source_sha256=None, dataset_id="alpha",
):
    vad_num_samples = _vad_length(frames, sample_rate)
    if timeline is None:
        timeline = []
    if stable_timeline is None:
        stable_timeline = [{"start_sample": 0, "end_sample": frames}]
    values = {
        "source_key": source_key,
        "dataset_id": dataset_id,
        "language": "en",
        "category": "read_speech",
        "quality_profile": "quality_high",
        "input_root": "/definitely/missing/audio",
        "original_path": "/definitely/missing/audio/spk-bait/source.wav",
        "original_relative_path": "spk-bait/source.wav",
        "source_relative_path": "spk-bait/source.wav",
        "source_size_bytes": 123,
        "source_sha256": source_sha256 or hashlib.sha256(source_key.encode()).hexdigest(),
        "sample_rate": sample_rate,
        "frames": frames,
        "channels": 1,
        "duration_sec": frames / sample_rate,
        "format": "WAV",
        "subtype": "PCM_16",
        "speaker_id": speaker_id,
        "speaker_scope": speaker_scope,
        "source_start_sample": 0,
        "source_end_sample": frames,
        "signal_finite": True,
        "signal_zero_ratio": 0.0,
        "signal_peak": 0.5,
        "signal_clipping_ratio": 0.0,
        "signal_dc_offset": 0.0,
        "signal_rms_dbfs": -20.0,
        "signal_num_samples": frames,
        "bandwidth_hz": 3500.0,
        "estimated_sr": get_optimal_sample_rate(3500.0, sample_rate),
        "vad_speech_ratio": 0.0,
        "vad_timeline_sample_rate": 16000,
        "vad_timeline_num_samples": vad_num_samples,
        "vad_timeline": timeline,
        "stable_silence_timeline": stable_timeline,
    }
    assert tuple(values) == STEP1_ROW_FIELDS
    return values


def _publish_step1(
    output_root, rows, *, config_digest="1" * 64, lineage="2" * 64,
    dataset_id="alpha",
):
    writer = ArtifactWriter(
        str(output_root), PipelineStep.STEP1, dataset_id, config_digest, lineage
    )
    for row in rows:
        writer.append(row)
    return writer.finalize()


def _run(tmp_path, rows, *, config=None):
    output_root = tmp_path / "out"
    dataset_id = rows[0]["dataset_id"] if rows else "alpha"
    config = config or _config(output_root, dataset_id=dataset_id)
    upstream = _publish_step1(output_root, rows, dataset_id=dataset_id)
    context = _context(
        config, _expectation(upstream), dataset_id=dataset_id
    )
    index = Step2SegmentStage().run(context)
    return index, list(iter_artifact_rows(index)), context, upstream


def _direct_rows(tmp_path, rows, *, dataset_id="alpha"):
    output_root = tmp_path / "out"
    index = _publish_step1(output_root, [], dataset_id=dataset_id)
    config = _config(output_root, dataset_id=dataset_id)
    context = _context(config, dataset_id=dataset_id)
    return list(Step2SegmentStage().produce_rows(context, index, rows))


def _assert_contiguous(rows, frames):
    assert rows[0]["start_sample"] == 0
    assert rows[-1]["end_sample"] == frames
    assert all(
        left["end_sample"] == right["start_sample"]
        for left, right in zip(rows, rows[1:])
    )


def test_exact_short_source_artifact_row_contract_and_lineage(tmp_path):
    upstream_row = _row()
    index, rows, context, upstream = _run(tmp_path, [upstream_row])
    assert STEP2_ROW_FIELDS == EXPECTED_STEP2_FIELDS
    assert (index.step, index.dataset_id) == ("step2", "alpha")
    assert index.config_digest == Step2SegmentStage().config_digest(context.config)
    assert index.lineage_digest == upstream.artifact_digest
    assert index.row_count == 1
    row = rows[0]
    assert set(row) == set(EXPECTED_STEP2_FIELDS)
    for field in EXPECTED_STEP2_FIELDS[:-4]:
        assert row[field] == upstream_row[field]
    assert set(STEP1_ROW_FIELDS) - set(row) == {
        "vad_timeline", "stable_silence_timeline"
    }
    # 身份字段必须逐字透传 Step1 → Step2
    assert row["category"] == upstream_row["category"] == "read_speech"
    assert row["quality_profile"] == upstream_row["quality_profile"] == "quality_high"
    assert (row["start_sample"], row["end_sample"], row["boundary_reason"]) == (
        0, upstream_row["frames"], "whole"
    )


def _tail_scenario_rows():
    """20.5s 源 + 1s 源：前者尾部 1.6s 独立落地，后者整源低于阈值。"""
    frames = 328_000  # 20.5s @16k
    speech = [
        {"start_sample": 0, "end_sample": 300_800, "mean_probability": 0.75},
        {"start_sample": 307_200, "end_sample": frames, "mean_probability": 0.75},
    ]
    stable = [{"start_sample": 300_800, "end_sample": 307_200}]
    return [
        _row(frames=frames, timeline=speech, stable_timeline=stable),
        _row(source_key="source-2", frames=16_000),
    ]


def test_min_segment_duration_zero_keeps_all_segment_rows(tmp_path):
    output_root = tmp_path / "out"
    config = _config(output_root)
    upstream_rows = _tail_scenario_rows()
    index = _publish_step1(output_root, upstream_rows)
    context = _context(config, _expectation(index))
    rows = list(
        Step2SegmentStage().produce_rows(context, index, upstream_rows)
    )
    assert [
        (row["start_sample"], row["end_sample"], row["boundary_reason"])
        for row in rows
    ] == [
        (0, 302_400, "stable_silence"),
        (302_400, 328_000, "short_tail_unmerged"),
        (0, 16_000, "whole"),
    ]


def test_min_segment_duration_drops_short_tail_and_short_source(tmp_path):
    output_root = tmp_path / "out"
    config = _config(output_root, step2_min_segment_duration_sec=2.0)
    upstream_rows = _tail_scenario_rows()
    index = _publish_step1(output_root, upstream_rows)
    context = _context(config, _expectation(index))
    rows = list(
        Step2SegmentStage().produce_rows(context, index, upstream_rows)
    )
    # 1.6s 尾部与 1s 整源均低于 2s：不产生 segment 行。
    assert [
        (row["start_sample"], row["end_sample"], row["boundary_reason"])
        for row in rows
    ] == [
        (0, 302_400, "stable_silence"),
    ]
    payload = Step2SegmentStage().config_payload(config)
    assert payload["segmentation"]["min_segment_duration_sec"] == 2.0


def test_min_segment_duration_boundary_tail_is_kept(tmp_path):
    output_root = tmp_path / "out"
    config = _config(output_root, step2_min_segment_duration_sec=1.0)
    upstream_rows = _tail_scenario_rows()
    index = _publish_step1(output_root, upstream_rows)
    context = _context(config, _expectation(index))
    rows = list(
        Step2SegmentStage().produce_rows(context, index, upstream_rows)
    )
    # 1.6s 尾部 ≥ 1s 保留；1s 整源恰好等于阈值也保留（跳过条件是严格小于）。
    assert [
        (row["start_sample"], row["end_sample"], row["boundary_reason"])
        for row in rows
    ] == [
        (0, 302_400, "stable_silence"),
        (302_400, 328_000, "short_tail_unmerged"),
        (0, 16_000, "whole"),
    ]


@pytest.mark.parametrize("frames", [1, 159999, 160000, 160001])
def test_every_single_speaker_source_invokes_planner_with_original_values(
    tmp_path, monkeypatch, frames
):
    row = _row(frames=frames)
    timeline = row["vad_timeline"]
    stable_timeline = row["stable_silence_timeline"]
    calls = []

    def spy(**kwargs):
        calls.append(kwargs)
        return (SegmentBoundary(0, frames, "whole"),)

    monkeypatch.setattr(segment_module, "plan_single_speaker_segments", spy)
    produced = _direct_rows(tmp_path, [row])
    assert len(produced) == 1
    assert calls == [{
        "source_num_samples": row["frames"],
        "source_sample_rate": row["sample_rate"],
        "vad_timeline": timeline,
        "vad_timeline_sample_rate": row["vad_timeline_sample_rate"],
        "vad_timeline_num_samples": row["vad_timeline_num_samples"],
        "stable_silence_timeline": stable_timeline,
        "max_segment_duration_sec": 60.0,
        "min_segment_duration_sec": 0.0,
    }]
    assert calls[0]["vad_timeline"] is timeline
    assert calls[0]["stable_silence_timeline"] is stable_timeline


@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000, 192000])
@pytest.mark.parametrize("duration_delta", [None, -1, 0])
def test_real_planner_short_threshold_is_inclusive(tmp_path, sample_rate, duration_delta):
    frames = 1 if duration_delta is None else 10 * sample_rate + duration_delta
    rows = _direct_rows(tmp_path, [_row(frames=frames, sample_rate=sample_rate)])
    assert len(rows) == 1
    assert (rows[0]["start_sample"], rows[0]["end_sample"]) == (0, frames)
    assert rows[0]["boundary_reason"] == "whole"


@pytest.mark.parametrize(
    ("frames", "timeline", "stable_timeline", "reasons"),
    [
        (22 * 16000, [{"start_sample": 0, "end_sample": 22 * 16000,
                       "mean_probability": 0.9}],
         [{"start_sample": 0, "end_sample": 22 * 16000}],
         ["no_stable_silence"]),
        (22 * 16000, [{"start_sample": 0, "end_sample": 9 * 16000,
                       "mean_probability": 0.9},
                      {"start_sample": 11 * 16000, "end_sample": 22 * 16000,
                       "mean_probability": 0.9}],
         [{"start_sample": 9 * 16000, "end_sample": 11 * 16000}],
         ["stable_silence", "no_stable_silence"]),
        (12 * 16000, [],
         [{"start_sample": 9 * 16000, "end_sample": 11 * 16000}],
         ["stable_silence", "whole"]),
    ],
)
def test_real_long_planner_preserves_reasons_and_full_timeline(
    tmp_path, frames, timeline, stable_timeline, reasons
):
    rows = _direct_rows(
        tmp_path,
        [_row(frames=frames, timeline=timeline, stable_timeline=stable_timeline)],
    )
    assert [row["boundary_reason"] for row in rows] == reasons
    _assert_contiguous(rows, frames)
    assert sum(row["end_sample"] - row["start_sample"] for row in rows) == frames


@pytest.mark.parametrize(
    ("speaker_id", "speaker_scope"), [("spk-1", "alpha"), (None, None)]
)
def test_speaker_metadata_is_inherited_without_path_inference(
    tmp_path, speaker_id, speaker_scope
):
    rows = _direct_rows(
        tmp_path,
        [_row(frames=22 * 16000, speaker_id=speaker_id, speaker_scope=speaker_scope)],
    )
    assert {(row["speaker_id"], row["speaker_scope"]) for row in rows} == {
        (speaker_id, speaker_scope)
    }


def test_empty_upstream_publishes_empty_artifact_without_planner(
    tmp_path, monkeypatch
):
    def forbidden(**kwargs):
        raise AssertionError("planner must not be called")

    monkeypatch.setattr(segment_module, "plan_single_speaker_segments", forbidden)
    index, rows, _, _ = _run(tmp_path, [])
    assert index.row_count == 0
    assert rows == []


def test_stage_never_requires_source_or_writes_waveform(tmp_path):
    index, _, _, _ = _run(tmp_path, [_row()])
    artifact = Path(index.artifact_dir)
    assert {item.name for item in artifact.iterdir()} == CONTRACT_FILES
    assert not list(artifact.rglob("*.wav"))
    assert not list(artifact.rglob("*.flac"))


def test_step2_requires_direct_step1_predecessor(tmp_path):
    stage = Step2SegmentStage()
    config = _config(tmp_path / "out")
    wrong = _publish_step1(tmp_path / "out", [])
    object.__setattr__(wrong, "step", PipelineStep.STEP0.value)
    with pytest.raises(ValueError, match="requires a Step1 upstream artifact"):
        list(stage.produce_rows(_context(config), wrong, []))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.pop("language"),
        lambda row: row.update(extra=True),
        lambda row: row.update(snr_db=20.0),
        lambda row: row.update(dataset_id="other"),
        lambda row: row.update(source_key=""),
        lambda row: row.update(source_key="  "),
        lambda row: row.update(source_key=1),
    ],
)
def test_step1_schema_and_identity_fail_before_planner(tmp_path, monkeypatch, mutation):
    row = _row()
    mutation(row)
    monkeypatch.setattr(
        segment_module, "plan_single_speaker_segments",
        lambda **kwargs: pytest.fail("planner called"),
    )
    with pytest.raises(ValueError):
        _direct_rows(tmp_path, [row])


def test_non_mapping_step1_row_fails_before_planner(tmp_path, monkeypatch):
    monkeypatch.setattr(
        segment_module, "plan_single_speaker_segments",
        lambda **kwargs: pytest.fail("planner called"),
    )
    with pytest.raises(ValueError, match="Step1 upstream row.*mapping"):
        _direct_rows(tmp_path, [object()])


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("language", "zh", "language.*dataset config identity"),
        ("category", "singing", "category.*dataset config identity"),
        ("quality_profile", "quality_premium", "quality_profile.*dataset config identity"),
        ("input_root", "/other", "input_root.*dataset config identity"),
        ("signal_finite", False, "signal_finite.*exact true"),
        ("signal_zero_ratio", 1.1, r"signal_zero_ratio.*\[0, 1\]"),
        ("signal_peak", -0.1, "signal_peak.*nonnegative"),
        ("signal_clipping_ratio", math.nan, "signal_clipping_ratio.*finite"),
        ("signal_dc_offset", True, "signal_dc_offset.*non-bool real"),
        ("signal_rms_dbfs", None, "rms_dbfs may be None"),
        ("signal_num_samples", 79999, r"signal_num_samples.*frames \* channels"),
        ("bandwidth_hz", 8000.1, "bandwidth_hz.*Nyquist"),
        ("estimated_sr", 16000, "estimated_sr does not match"),
        ("vad_speech_ratio", -0.1, r"vad_speech_ratio.*\[0, 1\]"),
    ],
)
def test_inherited_identity_signal_bandwidth_and_vad_facts_fail_before_planner(
    tmp_path, monkeypatch, field, value, error
):
    row = _row()
    row[field] = value
    monkeypatch.setattr(
        segment_module, "plan_single_speaker_segments",
        lambda **kwargs: pytest.fail("planner called"),
    )
    with pytest.raises(ValueError, match=error):
        _direct_rows(tmp_path, [row])


@pytest.mark.parametrize(
    ("field", "value", "row_kwargs"),
    [
        pytest.param("channels", -1, {}, id="channels-negative"),
        pytest.param("channels", 1.0, {}, id="channels-float"),
        pytest.param("source_size_bytes", 123.0, {}, id="source-size-float"),
        pytest.param("source_size_bytes", "123", {}, id="source-size-string"),
        pytest.param("format", None, {}, id="format-non-string"),
        pytest.param("format", _StrSubclass("WAV"), {}, id="format-string-subclass"),
        pytest.param("subtype", 1, {}, id="subtype-non-string"),
        pytest.param(
            "subtype", _StrSubclass("PCM_16"), {}, id="subtype-string-subclass"
        ),
        pytest.param("duration_sec", True, {}, id="duration-bool"),
        pytest.param("duration_sec", "5.0", {}, id="duration-non-real"),
        pytest.param(
            "speaker_id", _StrSubclass("spk"),
            {"speaker_id": "spk", "speaker_scope": "alpha"},
            id="speaker-id-string-subclass",
        ),
        pytest.param(
            "speaker_scope", _StrSubclass("alpha"),
            {"speaker_id": "spk", "speaker_scope": "alpha"},
            id="speaker-scope-string-subclass",
        ),
    ],
)
def test_exact_type_and_mode_policy_invalid_rows_fail_closed_before_planner(
    tmp_path, monkeypatch, field, value, row_kwargs
):
    row = _row(**row_kwargs)
    row[field] = value
    calls = []

    def planner_spy(**kwargs):
        calls.append(kwargs)
        return (SegmentBoundary(0, row["frames"], "whole"),)

    monkeypatch.setattr(segment_module, "plan_single_speaker_segments", planner_spy)
    with pytest.raises(ValueError):
        _direct_rows(tmp_path, [row])
    assert calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frames", 0), ("frames", -1), ("frames", True), ("frames", 1.5),
        ("sample_rate", 0), ("sample_rate", -1), ("sample_rate", True),
        ("sample_rate", 1.5), ("channels", 0), ("channels", True),
        ("source_size_bytes", -1), ("source_size_bytes", True),
        ("source_sha256", "A" * 64), ("source_sha256", "a" * 63),
        ("format", " "), ("subtype", ""),
        ("duration_sec", math.nan), ("duration_sec", math.inf),
        ("duration_sec", 999.0),
    ],
)
def test_source_header_validation_precedes_planner(
    tmp_path, monkeypatch, field, value
):
    row = _row()
    row[field] = value
    monkeypatch.setattr(
        segment_module, "plan_single_speaker_segments",
        lambda **kwargs: pytest.fail("planner called"),
    )
    with pytest.raises((TypeError, ValueError)):
        _direct_rows(tmp_path, [row])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_start_sample", False), ("source_start_sample", 1),
        ("source_end_sample", 1.0), ("source_end_sample", 79999),
    ],
)
def test_full_source_bounds_fail_before_planner(tmp_path, monkeypatch, field, value):
    row = _row()
    row[field] = value
    monkeypatch.setattr(
        segment_module, "plan_single_speaker_segments",
        lambda **kwargs: pytest.fail("planner called"),
    )
    with pytest.raises(ValueError):
        _direct_rows(tmp_path, [row])


@pytest.mark.parametrize(
    ("speaker_id", "speaker_scope"),
    [(None, "alpha"), ("spk", None), ("spk", "other"), ("", "alpha"),
     (" ", "alpha"), (1, "alpha"), ("spk", 1)],
)
def test_invalid_speaker_pair_fails_before_planner(
    tmp_path, monkeypatch, speaker_id, speaker_scope
):
    row = _row(speaker_id=speaker_id, speaker_scope=speaker_scope)
    monkeypatch.setattr(
        segment_module, "plan_single_speaker_segments",
        lambda **kwargs: pytest.fail("planner called"),
    )
    with pytest.raises(ValueError):
        _direct_rows(tmp_path, [row])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(vad_timeline_sample_rate=8000),
        lambda row: row.update(vad_timeline_num_samples=10),
        lambda row: row.update(vad_timeline="bad"),
        lambda row: row.update(vad_timeline=[{"start_sample": 0, "end_sample": 1}]),
        lambda row: row.update(vad_timeline=[{
            "start_sample": 0, "end_sample": 1, "mean_probability": 0.5, "x": 1
        }]),
        lambda row: row.update(vad_timeline=[
            {"start_sample": 0, "end_sample": 10, "mean_probability": 0.5},
            {"start_sample": 9, "end_sample": 11, "mean_probability": 0.5},
        ]),
        lambda row: row.update(vad_timeline=[{
            "start_sample": 0, "end_sample": 80001, "mean_probability": 0.5
        }]),
        lambda row: row.update(vad_timeline=[{
            "start_sample": 0, "end_sample": 1, "mean_probability": math.nan
        }]),
        lambda row: row.update(vad_timeline=[{
            "start_sample": 0, "end_sample": 1, "mean_probability": 1.1
        }]),
    ],
)
def test_d1_timeline_failures_propagate(tmp_path, mutation):
    row = _row()
    mutation(row)
    with pytest.raises((TypeError, ValueError)):
        _direct_rows(tmp_path, [row])


@pytest.mark.parametrize(
    "stable_timeline",
    [
        "bad",
        [{"start_sample": 0, "end_sample": 1, "extra": True}],
        [{"start_sample": True, "end_sample": 1}],
        [{"start_sample": 0, "end_sample": 0}],
        [{"start_sample": 0, "end_sample": 80001}],
        [
            {"start_sample": 0, "end_sample": 10},
            {"start_sample": 9, "end_sample": 11},
        ],
    ],
)
def test_malformed_stable_timeline_fails_before_planner(
    tmp_path, monkeypatch, stable_timeline
):
    row = _row(stable_timeline=stable_timeline)
    monkeypatch.setattr(
        segment_module, "plan_single_speaker_segments",
        lambda **kwargs: pytest.fail("planner called"),
    )
    with pytest.raises((TypeError, ValueError), match="stable|start_sample"):
        _direct_rows(tmp_path, [row])


@pytest.mark.parametrize(
    "boundaries",
    [
        [SegmentBoundary(0, 80000, "whole")],
        (),
        (object(),),
        (SegmentBoundary(1, 80000, "whole"),),
        (SegmentBoundary(0, 100, "whole"), SegmentBoundary(101, 80000, "whole")),
        (SegmentBoundary(0, 101, "whole"), SegmentBoundary(100, 80000, "whole")),
        (SegmentBoundary(0, 79999, "whole"),),
        (SegmentBoundary(0, 80001, "whole"),),
        (
            SegmentBoundary(0, 40000, "no_stable_silence"),
            SegmentBoundary(40000, 80000, "whole"),
        ),
        (SegmentBoundary(0, 79999, "no_stable_silence"),),
    ],
)
def test_planner_output_must_be_ordered_contiguous_full_coverage(
    tmp_path, monkeypatch, boundaries
):
    monkeypatch.setattr(
        segment_module, "plan_single_speaker_segments", lambda **kwargs: boundaries
    )
    with pytest.raises((TypeError, ValueError)):
        _direct_rows(tmp_path, [_row()])


def test_duplicate_source_key_fails_before_second_planner_call(tmp_path, monkeypatch):
    calls = []

    def spy(**kwargs):
        calls.append(kwargs)
        return (SegmentBoundary(0, 80000, "whole"),)

    monkeypatch.setattr(segment_module, "plan_single_speaker_segments", spy)
    with pytest.raises(ValueError, match="duplicate source_key"):
        _direct_rows(tmp_path, [_row(), _row()])
    assert len(calls) == 1


def test_duplicate_segment_id_aborts_without_suffix(tmp_path, monkeypatch):
    frames = 20 * 16000
    monkeypatch.setattr(
        segment_module, "plan_single_speaker_segments",
        lambda **kwargs: (
            SegmentBoundary(0, 10 * 16000, "stable_silence"),
            SegmentBoundary(10 * 16000, frames, "whole"),
        ),
    )
    monkeypatch.setattr(segment_module, "_segment_id", lambda row, boundary: "x" * 64)
    output_root = tmp_path / "out"
    config = _config(output_root)
    upstream = _publish_step1(output_root, [_row(frames=frames)])
    with pytest.raises(ValueError, match="duplicate segment_id"):
        Step2SegmentStage().run(_context(config, _expectation(upstream)))
    _assert_no_step2_or_staging(output_root)


def _id_oracle(row, start, end):
    identity = [
        "audio_filter_pipeline.segment", 1, row["dataset_id"], row["source_key"],
        row["source_sha256"], row["sample_rate"], start, end,
    ]
    encoded = json.dumps(
        identity, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_segment_id_matches_production_for_each_identity_input(tmp_path):
    fixed_sha = "a" * 64
    rows = [
        _row(source_key="base", source_sha256=fixed_sha),
        _row(
            dataset_id='数据|set:"[]', source_key="base",
            source_sha256=fixed_sha,
        ),
        _row(source_key='源|key:"[]', source_sha256=fixed_sha),
        _row(source_key="base", source_sha256="f" * 64),
        _row(source_key="base", source_sha256=fixed_sha, sample_rate=8000),
    ]
    actual_ids = []
    for index, row in enumerate(rows):
        produced = _direct_rows(
            tmp_path / str(index), [row], dataset_id=row["dataset_id"]
        )[0]
        expected = _id_oracle(row, 0, row["frames"])
        assert produced["segment_id"] == expected
        assert re.fullmatch(r"[0-9a-f]{64}", produced["segment_id"])
        actual_ids.append(produced["segment_id"])
    assert len(set(actual_ids)) == len(rows)


def test_segment_id_matches_production_for_start_and_end_bounds(
    tmp_path, monkeypatch
):
    frames = 80000
    row = _row(frames=frames, source_sha256="a" * 64)

    def produce_with_cut(name, cut):
        monkeypatch.setattr(
            segment_module,
            "plan_single_speaker_segments",
            lambda **kwargs: (
                SegmentBoundary(0, cut, "stable_silence"),
                SegmentBoundary(cut, frames, "whole"),
            ),
        )
        produced = _direct_rows(tmp_path / name, [row])
        assert [item["segment_id"] for item in produced] == [
            _id_oracle(row, 0, cut),
            _id_oracle(row, cut, frames),
        ]
        return produced

    baseline = produce_with_cut("baseline", 40000)
    changed = produce_with_cut("changed", 40001)
    assert baseline[0]["segment_id"] != changed[0]["segment_id"]
    assert baseline[1]["segment_id"] != changed[1]["segment_id"]


def test_fresh_runs_are_byte_and_digest_deterministic(tmp_path):
    results = []
    for name in ("one", "two"):
        index, rows, _, _ = _run(tmp_path / name, [_row()])
        results.append((Path(index.rows_path).read_bytes(), index.rows_sha256,
                        index.artifact_digest, rows[0]["segment_id"]))
    assert results[0] == results[1]


def test_direct_row_order_follows_sources_and_planner_tuple(tmp_path, monkeypatch):
    def planner(**kwargs):
        frames = kwargs["source_num_samples"]
        return (
            SegmentBoundary(0, frames // 2, "stable_silence"),
            SegmentBoundary(frames // 2, frames, "whole"),
        )

    monkeypatch.setattr(segment_module, "plan_single_speaker_segments", planner)
    rows = _direct_rows(tmp_path, [_row(source_key="b"), _row(source_key="a")])
    assert [(row["source_key"], row["start_sample"]) for row in rows] == [
        ("b", 0), ("b", 40000), ("a", 0), ("a", 40000)
    ]
    assert all(tuple(row) == STEP2_ROW_FIELDS for row in rows)


def test_valid_resume_skips_planner_and_preserves_artifact(tmp_path, monkeypatch):
    index, rows, context, _ = _run(tmp_path, [_row()])
    monkeypatch.setattr(
        segment_module, "plan_single_speaker_segments",
        lambda **kwargs: pytest.fail("planner called during resume"),
    )
    resumed = Step2SegmentStage().run(context)
    assert resumed.artifact_digest == index.artifact_digest
    assert list(iter_artifact_rows(resumed)) == rows


@pytest.mark.parametrize(
    ("filename", "error"),
    [
        ("rows.jsonl", "invalid artifact row 2: missing LF newline"),
        ("index.json", "invalid index JSON"),
        ("_SUCCESS", "marker size mismatch"),
    ],
)
def test_step2_tamper_fails_resume_without_planner_or_overwrite(
    tmp_path, monkeypatch, filename, error
):
    index, _, context, _ = _run(tmp_path, [_row()])
    path = Path(index.artifact_dir) / filename
    original = path.read_bytes()
    path.write_bytes(original + b"tamper")
    tampered = path.read_bytes()
    monkeypatch.setattr(
        segment_module, "plan_single_speaker_segments",
        lambda **kwargs: pytest.fail("planner called after tamper"),
    )
    with pytest.raises(ValueError, match=error):
        Step2SegmentStage().run(context)
    assert path.read_bytes() == tampered


def test_step1_tamper_is_detected_before_step2_resume(tmp_path, monkeypatch):
    _, _, context, upstream = _run(tmp_path, [_row()])
    path = Path(upstream.rows_path)
    path.write_bytes(path.read_bytes() + b"tamper")
    monkeypatch.setattr(
        segment_module, "plan_single_speaker_segments",
        lambda **kwargs: pytest.fail("planner called after upstream tamper"),
    )
    with pytest.raises(ValueError):
        Step2SegmentStage().run(context)


def test_changed_upstream_lineage_rejects_existing_step2_before_planner(
    tmp_path, monkeypatch
):
    _, _, _, first_upstream = _run(tmp_path, [_row(source_key="first")])
    first_dir = Path(first_upstream.artifact_dir)
    parked_dir = first_dir.with_name("alpha-first")
    first_dir.rename(parked_dir)
    second_upstream = _publish_step1(tmp_path / "out", [_row(source_key="second")])
    config = _config(tmp_path / "out")
    monkeypatch.setattr(
        segment_module, "plan_single_speaker_segments",
        lambda **kwargs: pytest.fail("planner called for stale lineage"),
    )
    with pytest.raises(ValueError):
        Step2SegmentStage().run(_context(config, _expectation(second_upstream)))


def test_exact_config_payload_is_json_safe_and_selection_policy_isolated(tmp_path):
    expected = {
        "row_contract": {"version": 5, "fields": list(EXPECTED_STEP2_FIELDS)},
        "upstream_row_contract": {
            "version": 6,
            "fields": list(STEP1_ROW_FIELDS),
            "validation": "complete_inherited_facts",
        },
        "segmentation": {
            "version": 5, "planner": "plan_single_speaker_segments",
            "source_domain": "original_sample_rate", "vad_timeline_sample_rate": 16000,
            "stable_silence_source_domain": True,
            "safe_intersection": "vad_silence_and_stable_silence",
            "guard_ms_each_side": 100,
            "target_duration_sec": 10, "silence_search_radius_sec": 1,
            "forward_safe_search": True, "hard_cut": False,
            "short_tail_sec": 2, "max_merged_duration_sec": 12,
            "max_segment_duration_sec": 60.0,
            "min_segment_duration_sec": 0.0,
            "overlong_policy": "skip_without_row_emission",
            "short_policy": "skip_without_row_emission",
            "interval": "half_open", "coverage": "contiguous_full_source",
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
        "segment_id": {"version": 1, "algorithm": "sha256_canonical_json_array"},
        "waveform_policy": {
            "read": False, "write": False, "trim": False, "concatenate": False
        },
    }
    stage = Step2SegmentStage()
    payload = dict(stage.config_payload(_config(tmp_path / "one")))
    assert payload.pop("analysis_algorithm_source") == analyzer_algorithm_payload()
    assert payload == expected
    json.dumps(expected, allow_nan=False)
    assert stage.config_digest(_config(tmp_path / "one", threshold=3.0)) == \
        stage.config_digest(_config(tmp_path / "two", threshold=4.0))


class _VersionedStage(Step2SegmentStage):
    def config_payload(self, config):
        payload = dict(super().config_payload(config))
        payload["segment_id"] = {"version": 2, "algorithm": "sha256_canonical_json_array"}
        return payload


def test_changed_stage_payload_rejects_existing_artifact_before_planner(
    tmp_path, monkeypatch
):
    _, _, context, _ = _run(tmp_path, [_row()])
    monkeypatch.setattr(
        segment_module, "plan_single_speaker_segments",
        lambda **kwargs: pytest.fail("planner called for stale output"),
    )
    with pytest.raises(ValueError):
        _VersionedStage().run(context)


@pytest.mark.parametrize("failure_kind", ["first", "second", "malformed_later"])
def test_producer_failures_are_atomic(tmp_path, monkeypatch, failure_kind):
    calls = []

    def planner(**kwargs):
        calls.append(kwargs)
        if failure_kind == "first" or (failure_kind == "second" and len(calls) == 2):
            raise RuntimeError("planner boom")
        if failure_kind == "malformed_later" and len(calls) == 2:
            return (SegmentBoundary(1, kwargs["source_num_samples"], "whole"),)
        return (SegmentBoundary(0, kwargs["source_num_samples"], "whole"),)

    monkeypatch.setattr(segment_module, "plan_single_speaker_segments", planner)
    output_root = tmp_path / "out"
    config = _config(output_root)
    upstream = _publish_step1(output_root, [_row(source_key="a"), _row(source_key="b")])
    with pytest.raises((RuntimeError, ValueError)):
        Step2SegmentStage().run(_context(config, _expectation(upstream)))
    _assert_no_step2_or_staging(output_root)


def test_valid_row_then_malformed_row_is_atomic(tmp_path, monkeypatch):
    rows = [_row(source_key="a"), _row(source_key="b")]
    rows[1]["frames"] = 0
    output_root = tmp_path / "out"
    config = _config(output_root)
    upstream = _publish_step1(output_root, rows)
    with pytest.raises(ValueError):
        Step2SegmentStage().run(_context(config, _expectation(upstream)))
    _assert_no_step2_or_staging(output_root)


def test_upstream_iterator_failure_is_atomic(tmp_path, monkeypatch):
    import audio_filter_pipeline.processors.stages_v3.base as stage_base

    output_root = tmp_path / "out"
    config = _config(output_root)
    upstream = _publish_step1(output_root, [_row()])
    real_open_reader = stage_base.open_artifact_reader

    def broken_reader(*args, **kwargs):
        reader = real_open_reader(*args, **kwargs)

        class BrokenReader:
            index = reader.index

            def __init__(self):
                self._seen = False

            def __iter__(self):
                return self

            def __next__(self):
                if self._seen:
                    raise RuntimeError("upstream boom")
                self._seen = True
                return next(reader)

            def close(self):
                reader.close()

        return BrokenReader()

    monkeypatch.setattr(stage_base, "open_artifact_reader", broken_reader)
    with pytest.raises(RuntimeError, match="upstream boom"):
        Step2SegmentStage().run(_context(config, _expectation(upstream)))
    _assert_no_step2_or_staging(output_root)


def test_previous_row_contract_version_resume_fails_before_planner(
    tmp_path, monkeypatch,
):
    """旧 Step2 row_contract（version 4）artifact 必须拒绝恢复且不改写。"""
    upstream_row = _row()
    output_root = tmp_path / "out"
    upstream = _publish_step1(output_root, [upstream_row])
    config = _config(output_root)
    stage = Step2SegmentStage()
    previous_payload = dict(stage.config_payload(config))
    previous_payload["row_contract"] = {
        "version": 4, "fields": list(STEP2_ROW_FIELDS),
    }
    previous_digest = hashlib.sha256(
        json.dumps(
            previous_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert previous_digest != stage.config_digest(config)
    writer = ArtifactWriter(
        output_root,
        PipelineStep.STEP2,
        "alpha",
        previous_digest,
        upstream.artifact_digest,
    )
    writer.append({"stale": True})
    writer.finalize()
    artifact_dir = _artifact_path(output_root)
    rows_before = (artifact_dir / "rows.jsonl").read_bytes()

    monkeypatch.setattr(
        segment_module, "plan_single_speaker_segments",
        lambda **kwargs: pytest.fail("planner called for previous row-contract"),
    )
    with pytest.raises(ValueError, match="expected config digest mismatch"):
        stage.run(_context(config, _expectation(upstream)))

    assert (artifact_dir / "rows.jsonl").read_bytes() == rows_before


def test_profile_survives_step0_to_step1_to_step2_exactly(tmp_path, monkeypatch):
    """quality_profile/category 必须逐字贯穿真实 Step0→Step1→Step2 三阶段行。"""
    import numpy as np
    import soundfile as sf
    from audio_filter_pipeline.config import DatasetSpec, SpeakerMetadataSpec
    from audio_filter_pipeline.processors.stages_v3.analyze import Step1AnalyzeStage
    from audio_filter_pipeline.processors.stages_v3.ingest import Step0IngestStage
    from audio_filter_pipeline.processors.vad_filter import SpeechInterval, VadAnalysis

    frames = 80000
    source = tmp_path / "input" / "source.wav"
    source.parent.mkdir(parents=True)
    time = np.arange(frames, dtype=np.float32) / 16000
    sf.write(
        str(source),
        (0.1 * np.sin(2 * np.pi * 440 * time)).astype(np.float32),
        16000,
        subtype="PCM_16",
    )
    raw = {
        "input_root": str(source.parent),
        "language": "en",
        "category": "read_speech",
        "quality_profile": "quality_high",
        "speaker": {"source": "none", "required": False},
    }
    spec = DatasetSpec(
        dataset_id="alpha",
        input_root=str(source.parent),
        language="en",
        category="read_speech",
        quality_profile="quality_high",
        speaker=SpeakerMetadataSpec(source="none", required=False),
    )
    output_root = tmp_path / "out"
    step0_config = PipelineConfig(
        pipeline_generation=3,
        step="step0",
        output_root=str(output_root),
        datasets={"alpha": raw},
        selection_policies={
            "global": {
                "mos_models": ["dnsmos"],
                "thresholds": {"dnsmos": 3.0},
                "min_mos_models": 1,
            },
            "quality_profiles": _quality_profiles(),
        },
    )
    step0 = Step0IngestStage(spec).run(
        StageRunContext(str(output_root), "alpha", step0_config)
    )
    step0_row = list(iter_artifact_rows(step0))[0]

    stage1 = Step1AnalyzeStage()

    class _FakeVad:
        def analyze(self, waveform, sample_rate):
            num_samples = len(waveform)
            return VadAnalysis(
                speech_ratio=0.75,
                timeline=(SpeechInterval(0, num_samples, 0.75),),
                sample_rate=16000,
                num_samples=num_samples,
            )

    monkeypatch.setattr(stage1, "_get_vad_processor", lambda: _FakeVad())
    step1 = stage1.run(_context(step0_config, upstream=_expectation(step0)))
    step1_row = list(iter_artifact_rows(step1))[0]

    step2 = Step2SegmentStage().run(
        _context(step0_config, upstream=_expectation(step1))
    )
    step2_row = list(iter_artifact_rows(step2))[0]

    assert (
        step0_row["category"]
        == step1_row["category"]
        == step2_row["category"]
        == "read_speech"
    )
    assert (
        step0_row["quality_profile"]
        == step1_row["quality_profile"]
        == step2_row["quality_profile"]
        == "quality_high"
    )


def _full_speech_timeline(frames):
    return [{"start_sample": 0, "end_sample": frames, "mean_probability": 0.9}]


def test_overlong_uninterrupted_source_produces_no_rows(tmp_path):
    frames = 100 * 16000
    row = _row(
        frames=frames,
        timeline=_full_speech_timeline(frames),
        stable_timeline=[],
    )
    assert _direct_rows(tmp_path, [row]) == []


def test_overlong_skip_resumes_after_safe_cut(tmp_path):
    frames = 90 * 16000
    row = _row(
        frames=frames,
        timeline=[
            {"start_sample": 0, "end_sample": 70 * 16000, "mean_probability": 0.9},
            {"start_sample": 71 * 16000, "end_sample": frames, "mean_probability": 0.9},
        ],
        stable_timeline=[{"start_sample": 70 * 16000, "end_sample": 71 * 16000}],
    )
    produced = _direct_rows(tmp_path, [row])
    assert [
        (r["start_sample"], r["end_sample"], r["boundary_reason"]) for r in produced
    ] == [(1_121_600, frames, "no_stable_silence")]


def test_overlong_cap_comes_from_config(tmp_path):
    frames = 100 * 16000
    row = _row(
        frames=frames,
        timeline=_full_speech_timeline(frames),
        stable_timeline=[],
    )
    output_root = tmp_path / "out"
    index = _publish_step1(output_root, [])
    config = _config(output_root)
    config.step2_max_segment_duration_sec = 200.0
    context = _context(config)
    produced = list(Step2SegmentStage().produce_rows(context, index, [row]))
    assert [
        (r["start_sample"], r["end_sample"], r["boundary_reason"]) for r in produced
    ] == [(0, frames, "no_stable_silence")]
