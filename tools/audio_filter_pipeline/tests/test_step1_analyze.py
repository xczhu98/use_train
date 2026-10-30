"""Artifact-backed Step1 完整源音频事实分析契约测试。"""

import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import audio_filter_pipeline.processors.stages_v3.analyze as analyze_module
import audio_filter_pipeline.processors.stages_v3.ingest as ingest_module
import audio_filter_pipeline.processors.stages_v3.source_io as source_io_module
import audio_filter_pipeline.processors.vad_filter as vad_module
import audio_filter_pipeline.utils.algorithm_identity as algorithm_identity
import audio_filter_pipeline.utils.bandwidth as bandwidth_module
from audio_filter_pipeline.config import DatasetSpec, PipelineConfig, SpeakerMetadataSpec
from audio_filter_pipeline.processors.stages_v3.analyze import (
    STEP1_ANALYSIS_FIELDS,
    STEP1_ROW_FIELDS,
    Step1AnalyzeStage,
)
from audio_filter_pipeline.processors.stages_v3.base import (
    EMPTY_LINEAGE_DIGEST,
    ArtifactExpectation,
    StageRunContext,
)
from audio_filter_pipeline.processors.stages_v3.ingest import (
    STEP0_ROW_FIELDS,
    Step0IngestStage,
)
from audio_filter_pipeline.processors.stages_v3.ten_vad import TenVadRuntimeOptions
from audio_filter_pipeline.processors.vad_filter import SpeechInterval, VadAnalysis
from audio_filter_pipeline.utils.algorithm_identity import (
    analyzer_algorithm_payload,
    analyzer_algorithm_source_sha256,
)
from audio_filter_pipeline.utils.artifacts import (
    ArtifactWriter,
    PipelineStep,
    iter_artifact_rows,
)
from audio_filter_pipeline.utils.path_utils import cleanup_expired_temp, v3_artifact_dir
from stage_test_helpers import runner_expectation_for_test


pytestmark = pytest.mark.unit
CONTRACT_FILES = {"rows.jsonl", "index.json", "_SUCCESS"}


def _write_wav(path, *, sample_rate=16000, frames=1600, data=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if data is None:
        time = np.arange(frames, dtype=np.float32) / sample_rate
        data = (0.1 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    sf.write(str(path), data, sample_rate, subtype="PCM_16")


def _dataset(input_root, dataset_id="alpha"):
    return DatasetSpec(
        dataset_id=dataset_id,
        input_root=str(input_root),
        language="en",
        category="read_speech",
        quality_profile="quality_high",
        speaker=SpeakerMetadataSpec(source="none", required=False),
    )


def _raw_dataset(spec):
    values = {
        "input_root": spec.input_root,
        "language": spec.language,
        "category": spec.category,
        "quality_profile": spec.quality_profile,
        "speaker": {"source": "none", "required": False},
    }
    if spec.source_layout != "regular_file":
        values["source_layout"] = spec.source_layout
    return values


def _quality_profiles():
    return {
        "quality_balanced": {"thresholds": {"dnsmos": {"ovrl": 3.2, "sig": 3.0, "bak": 3.8, "p808": 3.6}, "scoreq": 3.6, "urgent_pk": 3.5}},
        "quality_high": {"thresholds": {"dnsmos": {"ovrl": 3.3, "sig": 3.4, "bak": 4.0, "p808": 3.8}, "scoreq": 4.2, "urgent_pk": 3.5}},
        "quality_premium": {"thresholds": {"dnsmos": {"ovrl": 3.4, "sig": 3.6, "bak": 4.2, "p808": 4.0}, "scoreq": 4.5, "urgent_pk": 3.5}},
    }


def _config(output_root, spec, *, threshold=3.0):
    return PipelineConfig(
        pipeline_generation=3,
        step="step1",
        output_root=str(output_root),
        datasets={spec.dataset_id: _raw_dataset(spec)},
        selection_policies={
            "global": {"mos_models": ["dnsmos"], "thresholds": {"dnsmos": threshold}, "min_mos_models": 1},
            "quality_profiles": _quality_profiles(),
        },
    )


def _context(config, *, upstream=None):
    return StageRunContext(config.output_root, "alpha", config, upstream=upstream)


def _expectation(index):
    return runner_expectation_for_test(index)


def _artifact_path(output_root, step="step1"):
    return Path(v3_artifact_dir(str(output_root), step, "alpha"))


def _assert_no_step1_or_staging(output_root):
    target = _artifact_path(output_root)
    assert not target.exists()
    if target.parent.exists():
        assert not list(target.parent.glob(".alpha.stage-*"))


def _publish_real_step0(tmp_path, *, data=None, frames=1600, sample_rate=16000):
    source = tmp_path / "input" / "source.wav"
    _write_wav(source, frames=frames, data=data, sample_rate=sample_rate)
    spec = _dataset(source.parent)
    config = _config(tmp_path / "out", spec)
    index = Step0IngestStage(spec).run(StageRunContext(config.output_root, "alpha", config))
    return source, spec, config, index


def _publish_rows(output_root, rows, *, config_digest="1" * 64):
    writer = ArtifactWriter(output_root, PipelineStep.STEP0, "alpha", config_digest, EMPTY_LINEAGE_DIGEST)
    for row in rows:
        writer.append(row)
    return writer.finalize()


def _fake_analysis(waveform, sample_rate):
    num_samples = len(waveform)
    if num_samples >= 160000:
        timeline = (SpeechInterval(0, 64000, 0.9), SpeechInterval(96000, num_samples, 0.8))
    elif num_samples:
        timeline = (SpeechInterval(0, num_samples, 0.75),)
    else:
        timeline = ()
    covered = sum(item.end_sample - item.start_sample for item in timeline)
    return VadAnalysis(covered / num_samples if num_samples else 0.0, timeline, sample_rate, num_samples)


class RecordingVad:
    def __init__(self, analysis=_fake_analysis):
        self._analysis = analysis
        self.calls = []

    def analyze(self, waveform, sample_rate):
        self.calls.append((waveform.copy(), sample_rate))
        return self._analysis(waveform, sample_rate)


def _stage_with_fake(monkeypatch, fake=None, **kwargs):
    stage = Step1AnalyzeStage(**kwargs)
    fake = fake or RecordingVad()
    calls = []

    def get_processor():
        calls.append(True)
        return fake

    monkeypatch.setattr(stage, "_get_vad_processor", get_processor)
    monkeypatch.setattr(
        analyze_module,
        "estimate_audio_bandwidth",
        lambda _audio, sr, **_kwargs: (sr, 100.0),
    )
    return stage, fake, calls


def test_step1_publishes_exact_full_source_artifact_and_lineage(tmp_path, monkeypatch):
    source, _spec, config, step0 = _publish_real_step0(tmp_path)
    step0_row = list(iter_artifact_rows(step0))[0]
    source_bytes = source.read_bytes()
    stage, fake, processor_calls = _stage_with_fake(monkeypatch)

    step1 = stage.run(_context(config, upstream=_expectation(step0)))
    rows = list(iter_artifact_rows(step1))

    assert STEP1_ROW_FIELDS == STEP0_ROW_FIELDS + STEP1_ANALYSIS_FIELDS
    assert len(STEP0_ROW_FIELDS) == 19
    assert len(STEP1_ANALYSIS_FIELDS) == 16
    assert len(STEP1_ROW_FIELDS) == 35
    assert step1.lineage_digest == step0.artifact_digest
    assert step1.row_count == 1
    row = rows[0]
    assert [row[field] for field in STEP0_ROW_FIELDS] == [step0_row[field] for field in STEP0_ROW_FIELDS]
    assert set(row) == set(STEP1_ROW_FIELDS)
    assert row["source_start_sample"] == 0
    assert row["source_end_sample"] == step0_row["frames"]
    assert row["vad_timeline_num_samples"] == step0_row["frames"]
    json.dumps(row, allow_nan=False)
    assert source.read_bytes() == source_bytes
    assert processor_calls == [True]
    assert len(fake.calls) == 1
    artifact_dir = _artifact_path(config.output_root)
    assert {path.name for path in artifact_dir.iterdir()} == CONTRACT_FILES
    assert not any(path.suffix.lower() in {".wav", ".flac"} for path in artifact_dir.rglob("*"))


def test_step1_real_wav_uses_real_bandwidth_analysis_in_published_row(
    tmp_path, monkeypatch,
):
    sample_rate = 16_000
    frames = 4096
    time_axis = np.arange(frames, dtype=np.float32) / sample_rate
    data = (0.25 * np.sin(2 * np.pi * 1_000 * time_axis)).astype(np.float32)
    source, _spec, config, step0 = _publish_real_step0(
        tmp_path, data=data, frames=frames, sample_rate=sample_rate,
    )
    decoded, decoded_rate = sf.read(
        source, dtype="float32", always_2d=True,
    )
    expected_rate, expected_bandwidth = bandwidth_module.estimate_audio_bandwidth(
        decoded, sr=decoded_rate,
    )
    vad = RecordingVad()
    stage = Step1AnalyzeStage()
    monkeypatch.setattr(stage, "_get_vad_processor", lambda: vad)

    step1 = stage.run(_context(config, upstream=_expectation(step0)))
    row = list(iter_artifact_rows(step1))[0]

    assert expected_rate == sample_rate
    assert expected_bandwidth > 0.0
    assert row["bandwidth_hz"] == expected_bandwidth
    assert row["estimated_sr"] == bandwidth_module.get_optimal_sample_rate(
        expected_bandwidth, sample_rate,
    )
    assert len(vad.calls) == 1


def test_producer_builds_row_in_declared_field_order(tmp_path, monkeypatch):
    _source, _spec, config, step0 = _publish_real_step0(tmp_path)
    row = list(iter_artifact_rows(step0))[0]
    stage, _fake, _calls = _stage_with_fake(monkeypatch)
    produced = list(stage.produce_rows(_context(config, upstream=_expectation(step0)), step0, [row]))[0]
    assert tuple(produced) == STEP1_ROW_FIELDS


def test_two_sources_share_one_ten_processor_and_analyze_twice(tmp_path, monkeypatch):
    root = tmp_path / "input"
    _write_wav(root / "a.wav")
    _write_wav(root / "b.wav")
    spec = _dataset(root)
    config = _config(tmp_path / "out", spec)
    step0 = Step0IngestStage(spec).run(StageRunContext(config.output_root, "alpha", config))
    stage, fake, calls = _stage_with_fake(monkeypatch)
    result = stage.run(_context(config, upstream=_expectation(step0)))
    assert result.row_count == 2
    assert calls == [True]
    assert len(fake.calls) == 2


@pytest.mark.parametrize(
    ("workers", "max_pending"),
    [(65, 65), (2, 1), (2, 5)],
)
def test_step1_stage_rejects_unsafe_cpu_concurrency(workers, max_pending):
    with pytest.raises(ValueError, match="workers|max_pending"):
        Step1AnalyzeStage(workers=workers, max_pending=max_pending)


def test_cpu_pool_is_bounded_and_emits_upstream_order(tmp_path, monkeypatch):
    root = tmp_path / "input"
    for name in ("a.wav", "b.wav", "c.wav"):
        _write_wav(root / name)
    spec = _dataset(root)
    config = _config(tmp_path / "out", spec)
    step0 = Step0IngestStage(spec).run(StageRunContext(config.output_root, "alpha", config))
    rows = list(iter_artifact_rows(step0))
    submitted, shutdowns = [], []

    class Future:
        def __init__(self, row): self.row = row
        def result(self): return {field: self.row[field] if field in self.row else ([] if field == "vad_timeline" else 0) for field in STEP1_ROW_FIELDS}
        def cancel(self): pass

    class Executor:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 2
            kwargs["initializer"](*kwargs["initargs"])
        def submit(self, _function, row):
            submitted.append(row["source_key"])
            return Future(row)
        def shutdown(self, **kwargs): shutdowns.append(kwargs)

    monkeypatch.setattr(analyze_module, "_init_step1_analyze_worker", lambda *_args: None)
    stage = Step1AnalyzeStage(workers=2, max_pending=2, executor_factory=Executor)
    produced = stage.produce_rows(_context(config, upstream=_expectation(step0)), step0, rows)
    assert next(produced)["source_key"] == rows[0]["source_key"]
    assert submitted == [rows[0]["source_key"], rows[1]["source_key"]]
    assert [row["source_key"] for row in produced] == [rows[1]["source_key"], rows[2]["source_key"]]
    assert submitted == [row["source_key"] for row in rows]
    assert shutdowns == [{"wait": True, "cancel_futures": True}]


def test_cpu_pool_first_completed_reverse_completion_preserves_upstream_order(tmp_path, monkeypatch):
    root = tmp_path / "input"
    for name in ("a.wav", "b.wav"):
        _write_wav(root / name)
    spec = _dataset(root)
    config = _config(tmp_path / "out", spec)
    step0 = Step0IngestStage(spec).run(StageRunContext(config.output_root, "alpha", config))
    rows = list(iter_artifact_rows(step0))
    futures = []
    wait_calls = []

    class Future:
        def __init__(self, row):
            self.row = row

        def done(self):
            return True

        def result(self):
            return {
                field: self.row[field] if field in self.row else ([] if field == "vad_timeline" else 0)
                for field in STEP1_ROW_FIELDS
            }

        def cancel(self):
            pass

    class Executor:
        def __init__(self, **kwargs):
            kwargs["initializer"](*kwargs["initargs"])

        def submit(self, _function, row):
            future = Future(row)
            futures.append(future)
            return future

        def shutdown(self, **_kwargs):
            pass

    def reverse_first_completed(submitted, *, return_when):
        wait_calls.append((tuple(submitted), return_when))
        if len(wait_calls) == 1:
            assert tuple(submitted) == (futures[0], futures[1])
            return {futures[1]}, {futures[0]}
        assert tuple(submitted) == (futures[0],)
        return {futures[0]}, set()

    monkeypatch.setattr(analyze_module, "_init_step1_analyze_worker", lambda *_args: None)
    monkeypatch.setattr(analyze_module, "wait", reverse_first_completed)
    stage = Step1AnalyzeStage(workers=2, max_pending=2, executor_factory=Executor)

    produced = list(stage.produce_rows(_context(config, upstream=_expectation(step0)), step0, rows))

    assert [row["source_key"] for row in produced] == [row["source_key"] for row in rows]
    assert [call[0] for call in wait_calls] == [(futures[0], futures[1]), (futures[0],)]
    assert all(call[1] is analyze_module.FIRST_COMPLETED for call in wait_calls)


def test_working_set_excludes_disk_spool_for_streaming_aggregate_layout(tmp_path):
    source, _spec, config, step0 = _publish_real_step0(tmp_path)
    row = next(iter_artifact_rows(step0))
    stage = Step1AnalyzeStage()
    regular = analyze_module._step1_working_set_bytes(
        row,
        source_decode_parameters=stage.source_decode_parameters,
        bandwidth_parameters=stage.bandwidth_parameters,
        vad_parameters=stage.vad_parameters,
        raw_source_spooled=True,
    )
    aggregate = analyze_module._step1_working_set_bytes(
        row,
        source_decode_parameters=stage.source_decode_parameters,
        bandwidth_parameters=stage.bandwidth_parameters,
        vad_parameters=stage.vad_parameters,
        raw_source_spooled=False,
    )
    aggregate_spec = DatasetSpec(
        dataset_id="alpha",
        input_root=str(source.parent),
        language="en",
        category="read_speech",
        quality_profile="quality_high",
        speaker=SpeakerMetadataSpec(source="none", required=False),
        source_layout="streaming_flac_aggregate",
    )
    aggregate_config = _config(tmp_path / "aggregate-out", aggregate_spec)

    assert regular - aggregate == 1024 * 1024
    assert stage._raw_source_spooled(_context(config, upstream=_expectation(step0))) is True
    assert stage._raw_source_spooled(
        _context(aggregate_config, upstream=_expectation(step0))
    ) is False


def test_cpu_pool_global_inflight_gate_releases_completed_reservations(tmp_path, monkeypatch):
    root = tmp_path / "input"
    for name in ("a.wav", "b.wav"):
        _write_wav(root / name)
    spec = _dataset(root)
    config = _config(tmp_path / "out", spec)
    step0 = Step0IngestStage(spec).run(StageRunContext(config.output_root, "alpha", config))
    rows = list(iter_artifact_rows(step0))
    stage = Step1AnalyzeStage(workers=2, max_pending=2)
    reservations = [
        analyze_module._step1_working_set_bytes(
            row,
            source_decode_parameters=stage.source_decode_parameters,
            bandwidth_parameters=stage.bandwidth_parameters,
            vad_parameters=stage.vad_parameters,
        )
        for row in rows
    ]
    submission_counts = []
    submitted = []

    class Future:
        def __init__(self, row):
            self.row = row

        def result(self):
            submission_counts.append(len(submitted))
            return {
                field: self.row[field] if field in self.row else ([] if field == "vad_timeline" else 0)
                for field in STEP1_ROW_FIELDS
            }

        def cancel(self):
            pass

    class Executor:
        def __init__(self, **kwargs):
            kwargs["initializer"](*kwargs["initargs"])

        def submit(self, _function, row):
            submitted.append(row["source_key"])
            return Future(row)

        def shutdown(self, **_kwargs):
            pass

    monkeypatch.setattr(analyze_module, "_init_step1_analyze_worker", lambda *_args: None)
    monkeypatch.setattr(
        analyze_module,
        "STEP1_ANALYSIS_MAX_INFLIGHT_BYTES",
        max(reservations),
    )
    stage.executor_factory = Executor

    produced = list(stage.produce_rows(_context(config, upstream=_expectation(step0)), step0, rows))

    assert [row["source_key"] for row in produced] == [row["source_key"] for row in rows]
    assert submission_counts == [1, 2]


def test_cpu_pool_rejects_single_source_over_global_inflight_gate(tmp_path, monkeypatch):
    _source, _spec, config, step0 = _publish_real_step0(tmp_path)
    row = next(iter_artifact_rows(step0))
    submitted = []

    class Executor:
        def __init__(self, **kwargs):
            kwargs["initializer"](*kwargs["initargs"])

        def submit(self, *_args):
            submitted.append(True)
            pytest.fail("oversized source must not be submitted")

        def shutdown(self, **_kwargs):
            pass

    stage = Step1AnalyzeStage(workers=2, max_pending=2, executor_factory=Executor)
    reservation = analyze_module._step1_working_set_bytes(
        row,
        source_decode_parameters=stage.source_decode_parameters,
        bandwidth_parameters=stage.bandwidth_parameters,
        vad_parameters=stage.vad_parameters,
    )
    monkeypatch.setattr(analyze_module, "_init_step1_analyze_worker", lambda *_args: None)
    monkeypatch.setattr(analyze_module, "STEP1_ANALYSIS_MAX_INFLIGHT_BYTES", reservation - 1)

    with pytest.raises(ValueError, match="working set exceeds"):
        list(stage.produce_rows(_context(config, upstream=_expectation(step0)), step0, [row]))
    assert submitted == []


def test_cpu_pool_failure_cancels_pending_and_propagates(tmp_path):
    _source, _spec, config, step0 = _publish_real_step0(tmp_path)
    row = list(iter_artifact_rows(step0))[0]
    second_row = {**row, "source_key": "second"}
    cancelled, shutdowns = [], []

    class Future:
        def result(self): raise RuntimeError("worker boom")
        def cancel(self): cancelled.append(True)
    class Executor:
        def __init__(self, **_kwargs): pass
        def submit(self, _function, _row): return Future()
        def shutdown(self, **kwargs): shutdowns.append(kwargs)

    stage = Step1AnalyzeStage(workers=2, max_pending=2, executor_factory=Executor)
    with pytest.raises(RuntimeError, match="worker boom"):
        list(stage.produce_rows(_context(config, upstream=_expectation(step0)), step0, [row, second_row]))
    assert cancelled == [True]
    assert shutdowns == [{"wait": False, "cancel_futures": True}]


def test_empty_upstream_still_constructs_processor_once(tmp_path, monkeypatch):
    root = tmp_path / "input"
    root.mkdir()
    spec = _dataset(root)
    config = _config(tmp_path / "out", spec)
    step0 = _publish_rows(config.output_root, [])
    stage, fake, calls = _stage_with_fake(monkeypatch)
    result = stage.run(_context(config, upstream=_expectation(step0)))
    assert result.row_count == 0
    assert calls == [True]
    assert fake.calls == []


def test_duplicate_source_key_fails_closed_without_second_decode(tmp_path, monkeypatch):
    source, _spec, _config_seed, seed = _publish_real_step0(tmp_path / "seed")
    row = list(iter_artifact_rows(seed))[0]
    config = _config(tmp_path / "out", _dataset(source.parent))
    step0 = _publish_rows(config.output_root, [row, dict(row)])
    stage, _fake, _calls = _stage_with_fake(monkeypatch)
    real_decode = analyze_module.decode_verified_source
    calls = []
    def tracked_decode(upstream_row, **kwargs):
        calls.append(upstream_row["source_key"])
        return real_decode(upstream_row, **kwargs)
    monkeypatch.setattr(analyze_module, "decode_verified_source", tracked_decode)
    with pytest.raises(ValueError, match="duplicate source_key"):
        stage.run(_context(config, upstream=_expectation(step0)))
    assert len(calls) == 1
    _assert_no_step1_or_staging(config.output_root)


def test_ten_second_pause_preserves_full_source_bounds_header_and_bytes(tmp_path, monkeypatch):
    sample_rate = 16000
    data = np.zeros(sample_rate * 10, dtype=np.float32)
    tone = (0.1 * np.sin(2 * np.pi * 220 * np.arange(sample_rate * 4, dtype=np.float32) / sample_rate)).astype(np.float32)
    data[:sample_rate * 4], data[sample_rate * 6:] = tone, tone
    source, _spec, config, step0 = _publish_real_step0(tmp_path, data=data, frames=len(data))
    before, upstream = source.read_bytes(), list(iter_artifact_rows(step0))[0]
    stage, _fake, _calls = _stage_with_fake(monkeypatch)
    row = list(iter_artifact_rows(stage.run(_context(config, upstream=_expectation(step0)))))[0]
    assert row["source_start_sample"] == 0
    assert row["source_end_sample"] == upstream["frames"] == sample_rate * 10
    assert row["duration_sec"] == upstream["duration_sec"] == 10.0
    assert row["vad_timeline"] == [{"start_sample": 0, "end_sample": 64000, "mean_probability": 0.9}, {"start_sample": 96000, "end_sample": 160000, "mean_probability": 0.8}]
    assert source.read_bytes() == before


def test_all_zero_low_vad_is_still_published_without_policy_fields(tmp_path, monkeypatch):
    source, _spec, config, step0 = _publish_real_step0(tmp_path, data=np.zeros(1600, dtype=np.float32), frames=1600)
    fake = RecordingVad(lambda waveform, sample_rate: VadAnalysis(0.0, (), sample_rate, len(waveform)))
    stage, _fake, _calls = _stage_with_fake(monkeypatch, fake=fake)
    row = list(iter_artifact_rows(stage.run(_context(config, upstream=_expectation(step0)))))[0]
    assert row["signal_zero_ratio"] == 1.0
    assert row["vad_speech_ratio"] == 0.0
    assert row["vad_timeline"] == []
    assert not ({"accepted", "flags", "waveform", "chunk", "path", "policy_decision"} & set(row))
    assert source.exists()


@pytest.mark.parametrize("mutation", ["deleted", "size", "same_size", "symlink"])
def test_source_mutation_after_step0_fails_closed(tmp_path, monkeypatch, mutation):
    source, _spec, config, step0 = _publish_real_step0(tmp_path)
    replacement = tmp_path / "replacement.wav"
    _write_wav(replacement, data=np.full(1600, 0.02, dtype=np.float32))
    if mutation == "deleted": source.unlink()
    elif mutation == "size": source.write_bytes(source.read_bytes() + b"x")
    elif mutation == "same_size":
        raw = bytearray(source.read_bytes()); raw[-1] ^= 1; source.write_bytes(raw)
    else:
        source.unlink(); source.symlink_to(replacement)
    stage, _fake, _calls = _stage_with_fake(monkeypatch)
    with pytest.raises(ValueError):
        stage.run(_context(config, upstream=_expectation(step0)))
    _assert_no_step1_or_staging(config.output_root)


@pytest.mark.parametrize("fault", ["growth", "early_eof"])
def test_source_descriptor_hash_is_size_bounded_before_decode_and_publish(tmp_path, monkeypatch, fault):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io
    source, _spec, config, step0 = _publish_real_step0(tmp_path)
    source_bytes, expected_size = source.read_bytes(), source.stat().st_size
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    stage, _fake, _calls = _stage_with_fake(monkeypatch)
    real_pread, calls = source_io.os.pread, []
    def faulting_pread(descriptor, count, offset):
        snapshot = os.fstat(descriptor)
        if (snapshot.st_dev, snapshot.st_ino) == source_identity:
            calls.append((offset, count))
            payload = source_bytes + b"X" * 32 if fault == "growth" else source_bytes[:7]
            return payload[offset:offset + count]
        return real_pread(descriptor, count, offset)
    monkeypatch.setattr(source_io.os, "pread", faulting_pread)
    monkeypatch.setattr(source_io, "_decode_source_descriptor", lambda _fd: (_ for _ in ()).throw(AssertionError("must not decode")))
    with pytest.raises(ValueError, match="size|EOF|extra"):
        stage.run(_context(config, upstream=_expectation(step0)))
    assert calls == ([(0, expected_size), (expected_size, 1)] if fault == "growth" else [(0, expected_size), (7, expected_size - 7)])
    _assert_no_step1_or_staging(config.output_root)


@pytest.mark.parametrize("mutation,error", [
    (lambda row: {key: value for key, value in row.items() if key != "language"}, "fields"),
    (lambda row: {**row, "extra": True}, "fields"),
    (lambda row: {**row, "dataset_id": "other"}, "dataset_id"),
    (lambda row: {**row, "source_key": ""}, "source_key"),
])
def test_malformed_upstream_row_fails_before_decode(tmp_path, monkeypatch, mutation, error):
    source, _spec, _seed_config, seed = _publish_real_step0(tmp_path / "seed")
    config = _config(tmp_path / "out", _dataset(source.parent))
    step0 = _publish_rows(config.output_root, [mutation(list(iter_artifact_rows(seed))[0])])
    stage, _fake, _calls = _stage_with_fake(monkeypatch)
    monkeypatch.setattr(analyze_module, "decode_verified_source", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not decode")))
    with pytest.raises(ValueError, match=error): stage.run(_context(config, upstream=_expectation(step0)))
    _assert_no_step1_or_staging(config.output_root)


def test_late_invalid_upstream_row_fails_before_runtime_initialization(
    tmp_path, monkeypatch,
):
    source, _spec, config, step0 = _publish_real_step0(tmp_path)
    artifact_dir = Path(step0.artifact_dir)
    first_row = list(iter_artifact_rows(step0))[0]
    raw_rows = (
        json.dumps(
            first_row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        + b"\nnot-json\n"
    )
    (artifact_dir / "rows.jsonl").write_bytes(raw_rows)
    payload = json.loads((artifact_dir / "index.json").read_text(encoding="utf-8"))
    payload["rows_sha256"] = hashlib.sha256(raw_rows).hexdigest()
    payload["row_count"] = 2
    without_digest = {
        key: value for key, value in payload.items() if key != "artifact_digest"
    }
    payload["artifact_digest"] = hashlib.sha256(
        json.dumps(
            without_digest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    (artifact_dir / "index.json").write_bytes(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    (artifact_dir / "_SUCCESS").write_text(
        payload["artifact_digest"] + "\n", encoding="ascii",
    )

    stage = Step1AnalyzeStage()
    monkeypatch.setattr(
        stage,
        "_get_vad_processor",
        lambda: pytest.fail("runtime initialized before upstream validation"),
    )
    monkeypatch.setattr(
        analyze_module,
        "decode_verified_source",
        lambda *_args, **_kwargs: pytest.fail("producer ran before upstream validation"),
    )

    with pytest.raises(ValueError, match="invalid artifact row 2"):
        stage.run(_context(config, upstream=_expectation(step0)))

    assert source.exists()
    _assert_no_step1_or_staging(config.output_root)


@pytest.mark.parametrize("case", ["missing", "wrong_step", "wrong_config", "wrong_lineage", "tampered"])
def test_upstream_expectation_and_integrity_fail_closed(tmp_path, monkeypatch, case):
    _source, _spec, config, step0 = _publish_real_step0(tmp_path)
    stage, _fake, calls = _stage_with_fake(monkeypatch)
    expectation = _expectation(step0)
    if case == "missing": expectation = None
    elif case == "wrong_step": expectation = ArtifactExpectation(PipelineStep.STEP2, step0.config_digest, step0.lineage_digest)
    elif case == "wrong_config": expectation = ArtifactExpectation(PipelineStep.STEP0, "f" * 64, step0.lineage_digest)
    elif case == "wrong_lineage": expectation = ArtifactExpectation(PipelineStep.STEP0, step0.config_digest, "f" * 64)
    else: Path(step0.rows_path).write_bytes(b'{"tampered":true}\n')
    error_type = RuntimeError if case in {"wrong_config", "wrong_lineage"} else ValueError
    with pytest.raises(error_type):
        stage.run(_context(config, upstream=expectation))
    assert calls == []
    _assert_no_step1_or_staging(config.output_root)


def test_resume_uses_artifact_after_source_is_deleted(tmp_path, monkeypatch):
    source, _spec, config, step0 = _publish_real_step0(tmp_path)
    stage, fake, calls = _stage_with_fake(monkeypatch)
    first = stage.run(_context(config, upstream=_expectation(step0)))
    rows = list(iter_artifact_rows(first))
    source.unlink()
    monkeypatch.setattr(stage, "_get_vad_processor", lambda: (_ for _ in ()).throw(AssertionError("resume runtime access")))
    monkeypatch.setattr(analyze_module, "decode_verified_source", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("resume source access")))
    resumed = stage.run(_context(config, upstream=_expectation(step0)))
    assert resumed.artifact_digest == first.artifact_digest
    assert list(iter_artifact_rows(resumed)) == rows
    assert calls == [True]
    assert len(fake.calls) == 1


def test_changed_resample_method_rejects_old_artifact_before_dependencies(tmp_path, monkeypatch):
    source, _spec, config, step0 = _publish_real_step0(tmp_path)
    baseline, _fake, _calls = _stage_with_fake(monkeypatch)
    baseline.run(_context(config, upstream=_expectation(step0)))
    changed_parameters = replace(vad_module.STEP1_VAD_PARAMETERS, resample_method="scipy")
    monkeypatch.setattr(vad_module, "STEP1_VAD_PARAMETERS", changed_parameters)
    changed, _fake, changed_calls = _stage_with_fake(monkeypatch)
    source.unlink()
    monkeypatch.setattr(changed, "_get_vad_processor", lambda: (_ for _ in ()).throw(AssertionError("must not initialize")))
    with pytest.raises(ValueError, match="config digest"):
        changed.run(_context(config, upstream=_expectation(step0)))
    assert changed_calls == []


def test_existing_step1_rejects_changed_upstream_artifact_lineage(tmp_path, monkeypatch):
    source, _spec, config, first_step0 = _publish_real_step0(tmp_path)
    stage, _fake, _calls = _stage_with_fake(monkeypatch)
    stage.run(_context(config, upstream=_expectation(first_step0)))
    _artifact_path(config.output_root, "step0").rename(tmp_path / "old-step0")
    second_step0 = _publish_rows(config.output_root, [])
    with pytest.raises(ValueError, match="lineage digest"):
        stage.run(_context(config, upstream=_expectation(second_step0)))


def test_vad_budget_parameter_source_drives_payload_execution_and_resume_identity(tmp_path, monkeypatch):
    source, _spec, config, step0 = _publish_real_step0(tmp_path, frames=8)
    baseline, _fake, _calls = _stage_with_fake(monkeypatch)
    baseline.run(_context(config, upstream=_expectation(step0)))
    changed_parameters = source_io_module.SourceDecodeParameters(
        512 * 1024 * 1024,
        256 * 1024 * 1024 + 4,
        8000,
        192000,
        512 * 1024 * 1024,
    )
    monkeypatch.setattr(source_io_module, "STEP1_SOURCE_DECODE_PARAMETERS", changed_parameters)
    changed, _fake, calls = _stage_with_fake(monkeypatch)
    assert changed.source_decode_parameters is changed_parameters
    assert changed.config_payload(config)["analysis_resources"]["max_vad_analysis_bytes"] == 256 * 1024 * 1024 + 4
    with pytest.raises(ValueError, match="config digest"):
        changed.run(_context(config, upstream=_expectation(step0)))
    assert calls == []
    other_config = _config(tmp_path / "changed-out", _dataset(source.parent))
    other_step0 = Step0IngestStage(_dataset(source.parent)).run(StageRunContext(other_config.output_root, "alpha", other_config))
    changed.run(_context(other_config, upstream=_expectation(other_step0)))


def test_raw_source_budget_drives_payload_and_resume_identity(tmp_path, monkeypatch):
    source, _spec, config, step0 = _publish_real_step0(tmp_path, frames=8)
    baseline, _fake, _calls = _stage_with_fake(monkeypatch)
    baseline.run(_context(config, upstream=_expectation(step0)))
    changed_parameters = source_io_module.SourceDecodeParameters(
        512 * 1024 * 1024,
        256 * 1024 * 1024,
        8000,
        192000,
        512 * 1024 * 1024 + 1,
    )
    monkeypatch.setattr(source_io_module, "STEP1_SOURCE_DECODE_PARAMETERS", changed_parameters)
    changed, _fake, calls = _stage_with_fake(monkeypatch)

    assert changed.config_payload(config)["source_decode"]["max_raw_source_bytes"] == 512 * 1024 * 1024 + 1
    with pytest.raises(ValueError, match="config digest"):
        changed.run(_context(config, upstream=_expectation(step0)))
    assert calls == []
    other_config = _config(tmp_path / "changed-raw-out", _dataset(source.parent))
    other_step0 = Step0IngestStage(_dataset(source.parent)).run(
        StageRunContext(other_config.output_root, "alpha", other_config)
    )
    changed.run(_context(other_config, upstream=_expectation(other_step0)))


@pytest.mark.parametrize("limit_kind", ["decoded", "vad"])
def test_resource_budget_rejects_before_read_facts_vad_and_publish(tmp_path, monkeypatch, limit_kind):
    source, _spec, config, step0 = _publish_real_step0(tmp_path, frames=8)
    parameters = source_io_module.SourceDecodeParameters(
        max_decoded_bytes=(7 if limit_kind == "decoded" else 512 * 1024 * 1024),
        max_raw_source_bytes=512 * 1024 * 1024,
        max_vad_analysis_bytes=(7 if limit_kind == "vad" else 256 * 1024 * 1024),
        min_source_sample_rate=8000,
        max_source_sample_rate=192000,
    )
    monkeypatch.setattr(source_io_module, "STEP1_SOURCE_DECODE_PARAMETERS", parameters)
    stage, fake, _calls = _stage_with_fake(monkeypatch)
    monkeypatch.setattr(analyze_module, "analyze_decoded_source", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not analyze")))
    with pytest.raises(ValueError, match="resource limit exceeded"):
        stage.run(_context(config, upstream=_expectation(step0)))
    assert fake.calls == []
    _assert_no_step1_or_staging(config.output_root)


def test_config_payload_is_exact_and_policy_independent(tmp_path):
    root = tmp_path / "input"; root.mkdir()
    spec = _dataset(root)
    low, high = _config(tmp_path / "out", spec, threshold=1.0), _config(tmp_path / "out", spec, threshold=5.0)
    stage = Step1AnalyzeStage()
    payload = dict(stage.config_payload(low))
    assert payload.pop("analysis_algorithm_source") == analyzer_algorithm_payload()
    assert payload == {
        "row_contract": {"version": 7, "fields": list(STEP1_ROW_FIELDS)},
        "source_decode": {"version": 5, "descriptor_bound": True, "verify_source_size": True, "verify_source_sha256": True, "raw_snapshot": "bounded_disk_spool_shared_with_hash_and_decode", "dtype": "float32", "retained_layout": "first_channel_mono_1d", "decode_block_frames": 262144, "max_decoded_bytes": 512 * 1024 * 1024, "max_raw_source_bytes": 512 * 1024 * 1024},
        "analysis_resources": {"version": 3, "channel_policy": "first_channel_index_0", "max_inflight_bytes": 4 * 1024 * 1024 * 1024, "inflight_reservation": "raw_decoded_vad_fft_signal_stable_silence", "max_vad_analysis_bytes": 256 * 1024 * 1024, "min_source_sample_rate": 8000, "max_source_sample_rate": 192000, "vad_capacity_rule": "ceil_ratio_plus_one_non_identity"},
        "signal_metrics": {"version": 2, "channel_policy": "first_channel_index_0"},
        "bandwidth": {"version": 4, "channel_policy": "first_channel_index_0", "fft_size": 2048, "threshold_db": -40.0, "short_nonempty_window": "zero_pad_to_fft_size", "zero_bandwidth_estimated_sr": None},
        "vad": {"version": 5, "channel_policy": "first_channel_index_0", "native_cpu": True, "native_handler_lifecycle": "per_source", "asset": None, "hop_size": 256, "native_threshold": 0.5, "workers": 1, "max_pending": 1, "analysis_sample_rate": 16000, "resample_method": "soxr_hq", "speech_threshold": 0.5, "min_speech_ms": 100, "max_silence_gap_ms": 200, "padding_ms": 100},
        "stable_silence": {
            "version": 3,
            "source_domain": "original_sample_rate",
            "channel_policy": "first_channel_index_0",
            "working_set_max_bytes": 8 * 1024 * 1024,
            "algorithm": "float64_inplace_square_reduceat",
            "frame_ms": 10,
            "rms_threshold_dbfs": -45.0,
            "peak_threshold_dbfs": -30.0,
            "min_silence_ms": 200,
            "guard_ms": 100,
            "interval": "half_open",
        },
        "waveform_policy": {"full_source": True, "trim": False, "concatenate": False},
    }
    json.dumps(stage.config_payload(low), allow_nan=False)
    assert stage.config_digest(low) == stage.config_digest(high)


def test_analysis_algorithm_source_identity_hashes_live_algorithm_files(tmp_path):
    """算法源码哈希必须来自当前源文件，而非可手改的字面量。"""
    package_root = Path(algorithm_identity.__file__).resolve().parents[1]
    source = analyzer_algorithm_source_sha256()
    assert set(source) == {
        "processors/segmentation.py",
        "processors/signal_metrics.py",
        "processors/vad_filter.py",
        "utils/bandwidth.py",
    }
    for relative_path, digest in source.items():
        assert digest == hashlib.sha256(
            (package_root / relative_path).read_bytes()
        ).hexdigest()
    assert analyzer_algorithm_payload() == {
        "version": 1, "source_sha256": dict(source),
    }
    root = tmp_path / "identity-input"; root.mkdir()
    stage_payload = Step1AnalyzeStage().config_payload(
        _config(tmp_path / "identity-out", _dataset(root))
    )
    assert stage_payload["analysis_algorithm_source"] == analyzer_algorithm_payload()


def test_ten_runtime_options_drive_payload_digest_and_execution(tmp_path, monkeypatch):
    _source, _spec, config, step0 = _publish_real_step0(tmp_path)
    baseline = Step1AnalyzeStage()
    options = TenVadRuntimeOptions(hop_size=128, threshold=0.7)
    fake = RecordingVad()
    stage, _fake, calls = _stage_with_fake(monkeypatch, fake=fake, ten_vad_options=options, workers=2, max_pending=3)
    payload = stage.config_payload(config)
    assert payload["vad"]["asset"] is None
    assert payload["vad"]["native_cpu"] is True
    assert payload["vad"]["hop_size"] == 128
    assert payload["vad"]["native_threshold"] == 0.7
    assert payload["vad"]["speech_threshold"] == 0.7
    assert payload["vad"]["workers"] == 2
    assert payload["vad"]["max_pending"] == 3
    assert stage.config_digest(config) != baseline.config_digest(config)
    # 单进程 fake 仅验证 stage runtime 参数和无资产调用约定；pool 由独立测试覆盖。
    serial, serial_fake, serial_calls = _stage_with_fake(monkeypatch, fake=fake, ten_vad_options=options)
    serial.run(_context(config, upstream=_expectation(step0)))
    assert serial.ten_vad_options == options
    assert serial_calls == [True]
    assert len(serial_fake.calls) == 1
    assert calls == []


def _payload_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def test_previous_row_contract_version_resume_fails_before_decode(tmp_path, monkeypatch):
    _source, _spec, config, step0 = _publish_real_step0(tmp_path)
    stage, _fake, _calls = _stage_with_fake(monkeypatch)
    previous_payload = dict(stage.config_payload(config))
    previous_payload["row_contract"] = {"version": 4, "fields": list(STEP1_ROW_FIELDS)}
    writer = ArtifactWriter(config.output_root, PipelineStep.STEP1, "alpha", _payload_digest(previous_payload), step0.artifact_digest)
    writer.append({"stale": True}); writer.finalize()
    rows_before = (_artifact_path(config.output_root) / "rows.jsonl").read_bytes()
    monkeypatch.setattr(analyze_module, "decode_verified_source", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not decode")))
    with pytest.raises(ValueError, match="expected config digest mismatch"):
        stage.run(_context(config, upstream=_expectation(step0)))
    assert (_artifact_path(config.output_root) / "rows.jsonl").read_bytes() == rows_before


def test_parameter_sources_drive_payload_and_real_stage_execution(tmp_path, monkeypatch):
    _source, _spec, config, step0 = _publish_real_step0(tmp_path, frames=512, sample_rate=8000)
    upstream_row = list(iter_artifact_rows(step0))[0]
    default_bandwidth, default_vad = bandwidth_module.STEP1_BANDWIDTH_PARAMETERS, vad_module.STEP1_VAD_PARAMETERS
    changed_bandwidth = replace(default_bandwidth, fft_size=1024, threshold_db=-30.0)
    changed_vad = replace(default_vad, resample_method="scipy", speech_threshold=0.8, min_speech_ms=0, max_silence_gap_ms=0, padding_ms=0)
    monkeypatch.setattr(bandwidth_module, "STEP1_BANDWIDTH_PARAMETERS", changed_bandwidth)
    monkeypatch.setattr(vad_module, "STEP1_VAD_PARAMETERS", changed_vad)
    fake = RecordingVad(lambda waveform, sample_rate: VadAnalysis(0.5, (SpeechInterval(0, len(waveform), 0.9),), sample_rate, len(waveform)))
    stage, _fake, calls = _stage_with_fake(monkeypatch, fake=fake)
    captured = {}
    real_resample = analyze_module.resample_audio
    def recording_resample(waveform, orig_sr, target_sr, *, method):
        captured["resample"] = (orig_sr, target_sr, method)
        return real_resample(waveform, orig_sr, target_sr, method=method)
    monkeypatch.setattr(analyze_module, "resample_audio", recording_resample)
    row = list(stage.produce_rows(_context(config, upstream=_expectation(step0)), step0, [upstream_row]))[0]
    assert stage.config_payload(config)["bandwidth"] == {"version": 4, "channel_policy": "first_channel_index_0", "fft_size": 1024, "threshold_db": -30.0, "short_nonempty_window": "zero_pad_to_fft_size", "zero_bandwidth_estimated_sr": None}
    assert stage.config_payload(config)["vad"]["resample_method"] == "scipy"
    assert captured["resample"] == (8000, 16000, "scipy")
    assert calls == [True]
    assert row["vad_speech_ratio"] == 0.5


@pytest.mark.parametrize("failure", ["processor", "decode", "analyze"])
def test_processing_failure_aborts_step1_without_staging(tmp_path, monkeypatch, failure):
    _source, _spec, config, step0 = _publish_real_step0(tmp_path)
    stage, _fake, _calls = _stage_with_fake(monkeypatch)
    if failure == "processor": monkeypatch.setattr(stage, "_get_vad_processor", lambda: (_ for _ in ()).throw(RuntimeError("processor boom")))
    elif failure == "decode": monkeypatch.setattr(analyze_module, "decode_verified_source", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("decode boom")))
    else: monkeypatch.setattr(analyze_module, "analyze_decoded_source", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("analyze boom")))
    with pytest.raises(RuntimeError, match=f"{failure} boom"):
        stage.run(_context(config, upstream=_expectation(step0)))
    _assert_no_step1_or_staging(config.output_root)


def test_non_16k_vad_domain_aborts_without_publishing_artifact(tmp_path, monkeypatch):
    _source, _spec, config, step0 = _publish_real_step0(tmp_path)
    fake = RecordingVad(lambda waveform, _sample_rate: VadAnalysis(0.0, (), 8000, len(waveform)))
    stage, _fake, _calls = _stage_with_fake(monkeypatch, fake=fake)
    with pytest.raises(ValueError, match="VAD sample_rate must be 16000"):
        stage.run(_context(config, upstream=_expectation(step0)))
    assert len(fake.calls) == 1
    _assert_no_step1_or_staging(config.output_root)


def test_analysis_field_mismatch_aborts_without_publishing(tmp_path, monkeypatch):
    _source, _spec, config, step0 = _publish_real_step0(tmp_path)
    stage, _fake, _calls = _stage_with_fake(monkeypatch)
    monkeypatch.setattr(analyze_module, "analyze_decoded_source", lambda *_args, **_kwargs: {"unexpected": True})
    with pytest.raises(ValueError, match="analysis.*fields"):
        stage.run(_context(config, upstream=_expectation(step0)))
    _assert_no_step1_or_staging(config.output_root)


def test_cleanup_expired_temp_removes_only_old_pipeline_artifacts(tmp_path):
    root = tmp_path / "temp"
    root.mkdir()
    old_file = root / "audio-filter-old-file"
    old_directory = root / "audio-filter-old-directory"
    old_directory.mkdir()
    (old_directory / "payload").write_text("payload", encoding="utf-8")
    new_file = root / "audio-filter-new-file"
    unrelated = root / "notes"
    old_file.write_text("old", encoding="utf-8")
    new_file.write_text("new", encoding="utf-8")
    unrelated.write_text("keep", encoding="utf-8")
    old_timestamp = time.time() - 48 * 3600
    for path in (old_file, old_directory, unrelated):
        os.utime(path, (old_timestamp, old_timestamp))

    assert cleanup_expired_temp(root, max_age_hours=24) == 2
    assert not old_file.exists()
    assert not old_directory.exists()
    assert new_file.exists()
    assert unrelated.exists()


def test_cleanup_expired_temp_accepts_explicit_prefix_and_counts_top_level_artifacts(tmp_path):
    root = tmp_path / "temp"
    root.mkdir()
    custom_old = root / "custom-old"
    default_old = root / "audio-filter-old"
    custom_old.write_text("old", encoding="utf-8")
    default_old.write_text("keep", encoding="utf-8")
    old_timestamp = time.time() - 48 * 3600
    os.utime(custom_old, (old_timestamp, old_timestamp))
    os.utime(default_old, (old_timestamp, old_timestamp))

    assert cleanup_expired_temp(root, max_age_hours=24, prefixes=("custom-",)) == 1
    assert not custom_old.exists()
    assert default_old.exists()


@pytest.mark.parametrize("max_age_hours", [-1, True, 1.0, "1"])
def test_cleanup_expired_temp_rejects_invalid_age_types(tmp_path, max_age_hours):
    with pytest.raises(ValueError, match="max_age_hours"):
        cleanup_expired_temp(tmp_path / "missing", max_age_hours=max_age_hours)


@pytest.mark.parametrize(
    "prefixes",
    [
        (),
        "audio-filter-",
        ("",),
        (" bad",),
        ("bad ",),
        ("bad/name",),
        (".",),
        ("..",),
        ("same-", "same-"),
        123,
    ],
)
def test_cleanup_expired_temp_rejects_invalid_prefixes(tmp_path, prefixes):
    with pytest.raises(ValueError, match="prefix"):
        cleanup_expired_temp(tmp_path / "missing", prefixes=prefixes)


def test_cleanup_expired_temp_rejects_symlink_roots_and_parent_paths(tmp_path):
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        cleanup_expired_temp(linked_root)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    nested_root = linked_parent / "nested"
    nested_root.mkdir()
    with pytest.raises(ValueError, match="symlink"):
        cleanup_expired_temp(nested_root)


def test_cleanup_expired_temp_rejects_symlink_and_special_matching_entries(tmp_path):
    root = tmp_path / "temp"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")
    linked = root / "audio-filter-link"
    linked.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        cleanup_expired_temp(root)
    assert linked.is_symlink()
    assert outside.exists()

    linked.unlink()
    nested = root / "audio-filter-directory"
    nested.mkdir()
    (nested / "escape").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        cleanup_expired_temp(root)
    assert nested.exists()
    assert outside.exists()

    special_root = tmp_path / "special-temp"
    special_root.mkdir()
    os.mkfifo(special_root / "audio-filter-fifo")
    with pytest.raises(ValueError, match="unsupported type"):
        cleanup_expired_temp(special_root)


def test_cleanup_expired_temp_rejects_non_directory_root(tmp_path):
    root = tmp_path / "not-a-directory"
    root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        cleanup_expired_temp(root)


def test_bounded_executor_shutdown_terminates_visible_children_and_preserves_primary_error():
    shutdowns = []
    cancellations = []
    waits = []
    child_actions = []

    class Future:
        def cancel(self):
            cancellations.append(True)
            return True

    class ChildProcess:
        def __init__(self):
            self.alive = True

        def poll(self):
            return None if self.alive else 0

        def terminate(self):
            child_actions.append("terminate")

        def kill(self):
            child_actions.append("kill")
            self.alive = False

        def wait(self, *, timeout):
            waits.append(timeout)

    class WorkerThread:
        def is_alive(self):
            return True

        def join(self, *, timeout):
            waits.append(timeout)

    child = ChildProcess()
    thread = WorkerThread()

    class Executor:
        _processes = {1: child}
        _threads = {thread}

        def shutdown(self, **kwargs):
            shutdowns.append(kwargs)

    primary = RuntimeError("primary failure")
    ingest_module._shutdown_executor_bounded(
        Executor(),
        pending_futures=[Future()],
        stage="step0-ingest",
        workers=2,
        primary_error=primary,
        timeout_seconds=0.01,
    )

    assert cancellations == [True]
    assert shutdowns == [{"wait": False, "cancel_futures": True}]
    assert child_actions == ["terminate", "kill"]
    assert waits and all(timeout is not None for timeout in waits)
    assert any(timeout <= 0.01 for timeout in waits)
    assert "bounded executor shutdown" in "\n".join(primary.__notes__)
    assert "timeout=0.010s" in "\n".join(primary.__notes__)
    assert "worker thread remained alive" in "\n".join(primary.__notes__)
