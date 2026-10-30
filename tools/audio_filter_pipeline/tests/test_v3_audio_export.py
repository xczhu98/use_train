"""Generation-3 Step6 实际多采样率音频导出测试。"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import threading
from pathlib import Path
from threading import Barrier, Event, Lock, Thread
import time

import numpy as np
import pytest
import soundfile as sf

from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter
from audio_filter_pipeline.tests.audio_export_test_helpers import (
    install_export_rows_stub,
)


_TEST_PRODUCER = {
    "step": "step6", "dataset_id": "alpha", "config_digest": "a" * 64,
    "lineage_digest": "b" * 64, "rows_sha256": "c" * 64,
    "artifact_digest": "d" * 64,
}
pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _export_rows_stub(monkeypatch):
    install_export_rows_stub(monkeypatch, Step6AudioExporter)


def _step6_row(source: Path, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source_key": "source-1",
        "dataset_id": "alpha",
        "language": "en",
        "category": "read_speech",
        "quality_profile": "quality_high",
        "input_root": str(source.parent.resolve()),
        "original_path": str(source),
        "original_relative_path": "",
        "source_relative_path": source.name,
        "source_size_bytes": source.stat().st_size,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sample_rate": 16_000,
        "frames": 16_000,
        "channels": 1,
        "duration_sec": 1.0,
        "format": "WAV",
        "subtype": "FLOAT",
        "speaker_id": None,
        "speaker_scope": None,
        "source_start_sample": 0,
        "source_end_sample": 16_000,
        "signal_finite": True,
        "signal_zero_ratio": 0.0,
        "signal_peak": 0.5,
        "signal_clipping_ratio": 0.0,
        "signal_dc_offset": 0.0,
        "signal_rms_dbfs": -12.0,
        "signal_num_samples": 16_000,
        "bandwidth_hz": 3_000.0,
        "estimated_sr": 8_000,
        "vad_speech_ratio": 1.0,
        "vad_timeline_sample_rate": 16_000,
        "vad_timeline_num_samples": 16_000,
        "segment_id": "segment-1",
        "start_sample": 4_000,
        "end_sample": 12_000,
        "boundary_reason": "silence",
        "mos_models": ["scoreq"],
        "mos_scoring_policy_hash": "e" * 64,
        "mos_scores": {"scoreq": 4.5},
        "speaker_shadow": {
            "status": "disabled",
            "provenance": {
                "reason": "speaker_consistency_disabled",
                "inference_performed": False,
            },
        },
        "source_sample_rate": 16_000,
        "segment_sample_rate": 16_000,
        "effective_bandwidth_hz": 3_000.0,
        "target_sample_rate": 16_000,
        "final_selection": {
            "version": 3,
            "export_eligible": True,
            "reason": "mos_selected",
        },
        "export_routing": {
            "version": 3,
            "layout_version": 3,
            "layout_template": "{step}/{language}/{dataset_id}/{artifact_digest}/{manifest_or_ratehz_path}",
            "manifest_schema": "audio_filter_pipeline.export_manifest.v2",
            "manifest_filename": "_export_manifest.v2.jsonl",
            "transaction_schema": "audio_filter_pipeline.audio_export_transaction.v2",
            "mode": "metadata_only",
            "audio_write": False,
            "minimum_target_sample_rate_hz": 16_000,
            "source_below_minimum_rule": "native",
            "upsampling_allowed": False,
            "source_path": str(source),
            "segment_interval_samples": [4_000, 12_000],
        },
    }
    row.update(overrides)
    return row


def _step4_row(source: Path, **overrides: object) -> dict[str, object]:
    row = _step6_row(source)
    for field in (
        "speaker_shadow", "source_sample_rate", "segment_sample_rate",
        "effective_bandwidth_hz", "target_sample_rate", "final_selection",
        "export_routing",
    ):
        row.pop(field)
    row["mos_selection"] = {
        "decision": "keep",
        "passed_models": 1,
        "required_models": 1,
        "policy_id": "alpha:en",
        "policy_hash": "f" * 64,
    }
    row.update(overrides)
    return row


def _write_source(path: Path, sample_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.linspace(-0.5, 0.5, sample_rate, dtype=np.float32)
    sf.write(path, samples, sample_rate, subtype="FLOAT")


def test_exporter_resampler_matches_ceil_frame_contract_for_44k1_source(tmp_path):
    source = tmp_path / "input" / "source.wav"
    _write_source(source, 44_100)
    row = _step6_row(source)
    row.update({
        "sample_rate": 44_100,
        "frames": 44_100,
        "source_end_sample": 44_100,
        "signal_num_samples": 44_100,
        "estimated_sr": 8_000,
        "start_sample": 0,
        "end_sample": 8_000,
        "source_sample_rate": 44_100,
        "segment_sample_rate": 44_100,
        "target_sample_rate": 16_000,
        "export_routing": dict(
            row["export_routing"], segment_interval_samples=[0, 8_000],
        ),
    })

    exported = Step6AudioExporter(str(tmp_path / "out"), workers=1)._export_test_rows([row])
    audio, sample_rate = sf.read(
        exported[0].output_path, dtype="float32", always_2d=True,
    )

    assert sample_rate == 16_000
    assert audio.shape == (2_903, 1)


def test_exporter_clips_resampler_overshoot_at_full_scale_boundary(tmp_path):
    source_rate = 44_100
    frames = source_rate
    source = tmp_path / "input" / "source.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    samples = np.linspace(-0.75, 0.75, frames, dtype=np.float32)
    center = frames // 2
    samples[center - 16:center] = 0.999_969_482_421_875
    samples[center:center + 16] = -0.999_969_482_421_875
    sf.write(source, samples, source_rate, subtype="FLOAT")

    row = _step6_row(source)
    row.update({
        "sample_rate": source_rate,
        "frames": frames,
        "source_end_sample": frames,
        "signal_num_samples": frames,
        "estimated_sr": 8_000,
        "start_sample": 0,
        "end_sample": frames,
        "source_sample_rate": source_rate,
        "segment_sample_rate": source_rate,
        "target_sample_rate": 16_000,
        "export_routing": dict(
            row["export_routing"], segment_interval_samples=[0, frames],
        ),
    })
    exported = Step6AudioExporter(str(tmp_path / "out"), workers=1)._export_test_rows([row])
    audio, sample_rate = sf.read(
        exported[0].output_path, dtype="float32", always_2d=True,
    )

    assert sample_rate == 16_000
    assert audio.shape == (16_000, 1)
    assert float(np.max(np.abs(audio))) <= 1.0


def test_resampler_normalizes_significant_overshoot_instead_of_clipping():
    from audio_filter_pipeline.utils.resample import _normalize_resampled_length

    resampled = np.array([[-0.5], [1.02]], dtype=np.float32)

    normalized = _normalize_resampled_length(resampled, 2, 16_000, 16_000)

    np.testing.assert_allclose(
        normalized,
        np.array([[-0.5 / 1.02], [1.0]], dtype=np.float32),
        rtol=1e-7,
    )


def test_exporter_writes_exact_segment_at_routed_rate_under_v3_layout(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)

    result = Step6AudioExporter(str(tmp_path / "out"))._export_test_rows([
        _step6_row(source),
    ])

    assert len(result) == 1
    exported = result[0]
    destination = Path(exported.output_path)
    assert destination.is_file()
    relative_parts = destination.relative_to(tmp_path / "out").parts
    assert relative_parts[:5] == (
        "audio_exports", "step6", "en", "alpha", exported.output_relative_path.split("/")[3],
    )
    assert len(relative_parts[4]) == 64
    assert relative_parts[5:] == ("16000hz", "source--segment-1.flac")
    assert exported.segment_id == "segment-1"
    assert exported.target_sample_rate == 16_000
    audio, sample_rate = sf.read(destination, dtype="float32", always_2d=True)
    assert sample_rate == 16_000
    assert audio.shape == (8_000, 1)
    info = sf.info(destination)
    assert info.format == "FLAC"
    assert info.subtype == "PCM_24"


def test_multichannel_source_exports_first_channel_mono(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    sample_rate = 48_000
    frames = 12_000
    source = tmp_path / "input" / "stereo.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    audio = np.zeros((frames, 2), dtype=np.float32)
    audio[:, 0] = 0.25
    audio[:, 1] = -0.5
    sf.write(str(source), audio, sample_rate, subtype="FLOAT")
    row = _step6_row(
        source,
        sample_rate=sample_rate,
        frames=frames,
        channels=2,
        duration_sec=frames / sample_rate,
        source_sample_rate=sample_rate,
        segment_sample_rate=sample_rate,
        bandwidth_hz=7_000.0,
        effective_bandwidth_hz=7_000.0,
        estimated_sr=sample_rate,
        signal_num_samples=frames * 2,
    )

    result = Step6AudioExporter(str(tmp_path / "out"))._export_test_rows([row])

    assert len(result) == 1
    exported = result[0]
    destination = Path(exported.output_path)
    assert destination.is_file()
    assert exported.output_relative_path.endswith(".flac")
    out_audio, out_rate = sf.read(destination, dtype="float32", always_2d=True)
    assert out_rate == 16_000
    assert out_audio.shape[1] == 1
    expected_frames = ((row["end_sample"] - row["start_sample"]) * 16_000 + sample_rate - 1) // sample_rate
    assert out_audio.shape[0] == expected_frames
    np.testing.assert_allclose(out_audio[:, 0], np.full(expected_frames, 0.25), atol=0.1)
    manifest = _step6_manifest(tmp_path / "out")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["audio"]["channels"] == 1
    assert record["source"]["channels"] == 2


@pytest.mark.parametrize(
    ("source_sample_rate", "bandwidth_hz"), [(8_000, 3_900.0), (11_025, 5_000.0)],
)
def test_default_decoder_exports_real_low_rate_sources_natively(
        tmp_path, source_sample_rate, bandwidth_hz):
    source = tmp_path / "input" / "source.wav"
    _write_source(source, source_sample_rate)
    row = _step6_row(source)
    row.update({
        "sample_rate": source_sample_rate,
        "frames": source_sample_rate,
        "start_sample": 0,
        "end_sample": source_sample_rate,
        "bandwidth_hz": bandwidth_hz,
        "estimated_sr": source_sample_rate,
        "effective_bandwidth_hz": bandwidth_hz,
        "source_sample_rate": source_sample_rate,
        "segment_sample_rate": source_sample_rate,
        "target_sample_rate": source_sample_rate,
    })
    row["export_routing"] = dict(
        row["export_routing"], segment_interval_samples=[0, source_sample_rate],
    )

    exporter = Step6AudioExporter(str(tmp_path / "out"), workers=1)
    exported = exporter._export_test_rows([row])

    audio, actual_sample_rate = sf.read(
        exported[0].output_path, dtype="float32", always_2d=True,
    )
    assert exported[0].target_sample_rate == source_sample_rate
    assert actual_sample_rate == source_sample_rate
    assert audio.shape == (source_sample_rate, 1)
    assert exporter.last_io_counts["source_sessions_opened"] == 1
    assert exporter.last_io_counts["interval_decode_frames"] == source_sample_rate


def test_exporter_rejects_invalid_v3_target_and_legacy_routing(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "source.wav"
    _write_source(source)
    exporter = Step6AudioExporter(str(tmp_path / "out"), workers=1)
    row = _step6_row(
        source, sample_rate=48_000, frames=48_000,
        source_sample_rate=48_000, segment_sample_rate=48_000,
        target_sample_rate=8_000,
    )
    with pytest.raises(ValueError, match="target_sample_rate.*version 3 policy"):
        exporter._validate_row(row)

    row = _step6_row(source)
    row["export_routing"] = dict(row["export_routing"], version=2)
    with pytest.raises(ValueError, match="version 3 contract"):
        exporter._validate_row(row)


def test_exporter_rejects_duplicate_but_does_not_touch_legacy_layout(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    exporter = Step6AudioExporter(str(tmp_path / "out"))
    row = _step6_row(source)

    with pytest.raises(ValueError, match="duplicate destination"):
        exporter._export_test_rows([row, row])

    destination = (
        tmp_path / "out" / "audio_exports" / "step6" / "alpha" / "16000hz"
        / "source--segment-1.flac"
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"must remain untouched")
    exported = exporter._export_test_rows([row])
    assert Path(exported[0].output_path).is_file()
    assert Path(exported[0].output_path) != destination
    assert destination.read_bytes() == b"must remain untouched"


def test_exporter_uses_final_selection_before_collision_decode_and_write(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    calls: list[str] = []

    def decoder(row):
        calls.append(f"decode:{row['segment_id']}")
        return np.zeros((8_000, 1), dtype=np.float32), 16_000

    def writer(path, audio, sample_rate):
        calls.append(f"write:{sample_rate}")
        sf.write(path, audio, sample_rate, format="FLAC", subtype="PCM_24")

    keep = _step6_row(source)
    reject = _step6_row(source, final_selection={
        "version": 3, "export_eligible": False, "reason": "speaker_inconsistent",
    })
    exported = Step6AudioExporter(
        str(tmp_path / "out"), decoder=decoder, writer=writer,
    )._export_test_rows([keep, reject])

    assert [item.segment_id for item in exported] == ["segment-1"]
    assert calls == ["decode:segment-1", "write:16000"]
    assert Path(exported[0].output_path).is_file()


@pytest.mark.parametrize(
    ("final_selection", "remove_final_selection"),
    [
        pytest.param(None, True, id="missing"),
        pytest.param(None, False, id="none"),
        pytest.param([], False, id="non_mapping"),
        pytest.param({}, False, id="missing_fields"),
        pytest.param({"version": 1, "export_eligible": True, "reason": "x"}, False, id="unknown_version"),
        pytest.param({"version": 2, "export_eligible": "true", "reason": "x"}, False, id="non_bool"),
        pytest.param({"version": 2, "export_eligible": True, "reason": ""}, False, id="blank_reason"),
    ],
)
def test_exporter_fails_closed_for_invalid_final_selection(tmp_path, final_selection, remove_final_selection):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    row = _step6_row(source, final_selection=final_selection)
    if remove_final_selection:
        row.pop("final_selection")

    with pytest.raises(ValueError, match="final_selection"):
        Step6AudioExporter(str(tmp_path / "out"))._export_test_rows([row])


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_exporter_rejects_step6_rows_outside_exact_contract_even_when_rejected(
    tmp_path, mutation,
):
    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    row = _step6_row(source, final_selection={
        "version": 3, "export_eligible": False, "reason": "mos_rejected",
    })
    if mutation == "missing":
        row.pop("signal_peak")
    else:
        row["legacy_passthrough"] = {"opaque": True}

    with pytest.raises(ValueError, match="row fields do not match the exact contract"):
        Step6AudioExporter(str(tmp_path / "out"))._export_test_rows([row])


@pytest.mark.parametrize("nested_field", ["final_selection", "export_routing"])
def test_exporter_rejects_extra_nested_step6_fields(tmp_path, nested_field):
    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    row = _step6_row(source)
    row[nested_field] = dict(row[nested_field], legacy=True)

    with pytest.raises(ValueError, match=f"{nested_field} fields.*exact contract"):
        Step6AudioExporter(str(tmp_path / "out"))._export_test_rows([row])


def test_step4_exporter_accepts_exact_row_contract(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step4AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)

    exported = Step4AudioExporter(str(tmp_path / "out"))._export_test_rows([
        _step4_row(source),
    ])

    assert len(exported) == 1
    assert exported[0].segment_id == "segment-1"
    assert exported[0].target_sample_rate == 16_000


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_step4_exporter_rejects_rows_outside_exact_contract(tmp_path, mutation):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step4AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    row = _step4_row(source)
    if mutation == "missing":
        row.pop("signal_peak")
    else:
        row["legacy_passthrough"] = True

    with pytest.raises(ValueError, match="step4 row fields.*exact contract"):
        Step4AudioExporter(str(tmp_path / "out"))._export_test_rows([row])


def test_step4_exporter_rejects_extra_nested_mos_selection_field(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step4AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    row = _step4_row(source)
    row["mos_selection"] = dict(row["mos_selection"], legacy=True)

    with pytest.raises(ValueError, match="mos_selection fields.*exact contract"):
        Step4AudioExporter(str(tmp_path / "out"))._export_test_rows([row])


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"end_sample": 16_001}, "interval"),
        ({"source_relative_path": "../escape.wav"}, "source_relative_path"),
        ({"dataset_id": "../escape"}, "dataset_id"),
        ({"dataset_id": "alpha\\escape"}, "dataset_id"),
        ({"segment_id": "nested/segment"}, "segment_id"),
        ({"segment_id": ".."}, "segment_id"),
        ({"sample_rate": 8_000}, "sample_rate"),
    ],
)
def test_exporter_fails_closed_for_invalid_source_binding(tmp_path, overrides, error):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)

    with pytest.raises(ValueError, match=error):
        Step6AudioExporter(str(tmp_path / "out"))._export_test_rows([
            _step6_row(source, **overrides),
        ])


@pytest.mark.parametrize("workers, max_pending", [(0, 1), (1, 0), (True, 1), (2, 1)])
def test_exporter_rejects_invalid_concurrency_limits(tmp_path, workers, max_pending):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    with pytest.raises(ValueError):
        Step6AudioExporter(str(tmp_path / "out"), workers=workers, max_pending=max_pending)


@pytest.mark.parametrize("max_inflight_bytes", [0, True, 1.0])
def test_exporter_rejects_invalid_inflight_byte_budget(tmp_path, max_inflight_bytes):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    with pytest.raises(ValueError, match="max_inflight_bytes"):
        Step6AudioExporter(
            str(tmp_path / "out"), max_inflight_bytes=max_inflight_bytes,
        )


def test_exporter_parallelism_is_bounded_and_results_keep_input_order(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    active = 0
    peak_active = 0
    lock = Lock()

    def decoder(row):
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.02 * (4 - int(row["segment_id"].rsplit("-", 1)[1])))
        with lock:
            active -= 1
        return np.zeros((8_000, 1), dtype=np.float32), 16_000

    rows = [_step6_row(source, segment_id=f"segment-{index}") for index in range(4)]
    exported = Step6AudioExporter(
        str(tmp_path / "out"), decoder=decoder, workers=2, max_pending=2,
    )._export_test_rows(rows)

    assert peak_active <= 2
    assert [item.segment_id for item in exported] == ["segment-0", "segment-1", "segment-2", "segment-3"]


def test_exporter_strict_inflight_byte_gate_allows_exactly_one_task_at_a_time(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    active = 0
    peak_active = 0
    lock = Lock()

    def decoder(_row):
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return np.zeros((8_000, 1), dtype=np.float32), 16_000

    rows = [_step6_row(source, segment_id=f"segment-{index}") for index in range(3)]
    estimated = Step6AudioExporter._estimated_inflight_bytes(rows[0])
    exported = Step6AudioExporter(
        str(tmp_path / "out"), decoder=decoder, workers=3, max_pending=3,
        max_inflight_bytes=estimated,
    )._export_test_rows(rows)

    assert peak_active == 1
    assert [item.segment_id for item in exported] == ["segment-0", "segment-1", "segment-2"]


def test_exporter_parallel_failure_stops_new_submissions_and_preserves_outputs(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    second_started = Event()
    decoded: list[str] = []
    lock = Lock()

    def decoder(row):
        segment_id = row["segment_id"]
        with lock:
            decoded.append(segment_id)
        if segment_id == "segment-1":
            second_started.set()
            return np.zeros((8_000, 1), dtype=np.float32), 16_000
        if segment_id == "segment-0":
            assert second_started.wait(timeout=1)
            raise RuntimeError("decode failed")
        pytest.fail("rows after the bounded submitted batch must not decode")

    rows = [_step6_row(source, segment_id=f"segment-{index}") for index in range(3)]
    exporter = Step6AudioExporter(
        str(tmp_path / "out"), decoder=decoder, workers=2, max_pending=2,
    )
    with pytest.raises(RuntimeError, match="decode failed"):
        exporter._export_test_rows(rows)

    assert set(decoded) == {"segment-0", "segment-1"}
    destinations = list((tmp_path / "out" / "audio_exports").glob(
        "step6/en/alpha/*/16000hz/source--segment-1.flac"
    ))
    assert len(destinations) == 1 and destinations[0].is_file()


def _step6_destination(root: Path) -> Path:
    matches = list((root / "audio_exports").glob(
        "step6/en/alpha/*/16000hz/source--segment-1.flac"
    ))
    assert len(matches) == 1
    return matches[0]


def _step6_manifest(root: Path) -> Path:
    matches = list((root / "audio_exports").glob(
        "step6/en/alpha/*/_export_manifest.v2.jsonl"
    ))
    assert len(matches) == 1
    return matches[0]


def _transaction_marker(destination: Path) -> Path:
    matches = list(destination.parent.glob(".audio-export-staging-*/marker.json"))
    assert len(matches) == 1
    return matches[0]


def test_manifest_load_keeps_closing_descriptor_after_unlock_failure(
        tmp_path, monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import audio_export
    from audio_filter_pipeline.processors.stages_v3.audio_export import _ManifestState

    state = _ManifestState(tmp_path, "step6", "alpha", _TEST_PRODUCER, language="en")
    state._exclusive_file_lock = lambda: 123

    def fail_load():
        raise RuntimeError("manifest business failure")

    state._load_unlocked = fail_load
    closed: list[int] = []

    def fail_unlock(_descriptor, operation):
        assert operation == audio_export.fcntl.LOCK_UN
        raise OSError("unlock failure")

    monkeypatch.setattr(audio_export.fcntl, "flock", fail_unlock)
    monkeypatch.setattr(audio_export.os, "close", closed.append)

    with pytest.raises(RuntimeError, match="manifest business failure") as raised:
        state.load()

    assert closed == [123]
    assert any("manifest lock unlock close failed" in note for note in raised.value.__notes__)


def test_manifest_lock_closes_safe_path_after_descriptor_close_failure(
        tmp_path, monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import audio_export
    from audio_filter_pipeline.processors.stages_v3.audio_export import _ManifestState

    lock_path = tmp_path / "lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    real_close = audio_export.os.close
    close_calls: list[int] = []
    flock_calls: list[int] = []
    safe_closed: list[bool] = []

    class FakeSafe:
        def target_stat(self):
            return os.fstat(descriptor)

        def open_target(self, *_args):
            return descriptor

        def close(self):
            safe_closed.append(True)

    def fail_lock(value, operation):
        flock_calls.append(operation)
        if operation == audio_export.fcntl.LOCK_EX:
            raise RuntimeError("lock business failure")

    def fail_descriptor_close(value):
        close_calls.append(value)
        if value == descriptor:
            raise OSError("descriptor close failure")
        return real_close(value)

    try:
        monkeypatch.setattr(
            audio_export, "open_safe_output_path", lambda *_args, **_kwargs: FakeSafe(),
        )
        monkeypatch.setattr(audio_export.fcntl, "flock", fail_lock)
        monkeypatch.setattr(audio_export.os, "close", fail_descriptor_close)
        state = _ManifestState(tmp_path, "step6", "alpha", _TEST_PRODUCER, language="en")

        with pytest.raises(RuntimeError, match="lock business failure") as raised:
            state._exclusive_file_lock()
    finally:
        real_close(descriptor)

    assert flock_calls == [audio_export.fcntl.LOCK_EX, audio_export.fcntl.LOCK_UN]
    assert close_calls == [descriptor]
    assert safe_closed == [True]
    assert any("manifest lock descriptor close failed" in note for note in raised.value.__notes__)


def test_exporter_resume_skips_decode_and_write_after_manifest_publication(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    row = _step6_row(source)
    output_root = tmp_path / "out"
    first = Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])
    manifest = _step6_manifest(output_root)
    assert manifest.is_file()
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 1
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["schema"] == "audio_filter_pipeline.export_manifest.v2"
    assert record["status"] == "published"
    assert set(record["producer"]) == {
        "step", "dataset_id", "config_digest", "lineage_digest",
        "rows_sha256", "artifact_digest",
    }
    assert manifest.parent.name == record["producer"]["artifact_digest"]
    assert record["step"] == "step6"
    assert record["dataset_id"] == row["dataset_id"]
    assert record["segment_id"] == row["segment_id"]
    assert record["output_relative_path"] == first[0].output_relative_path
    assert record["source"]["source_sha256"] == row["source_sha256"]
    assert record["source"]["start_sample"] == row["start_sample"]
    assert record["source"]["end_sample"] == row["end_sample"]
    assert record["audio"] == {
        "format": "FLAC", "subtype": "PCM_24", "sample_rate": 16_000,
        "frames": 8_000, "channels": 1,
    }
    assert record["output_sha256"] == hashlib.sha256(
        Path(first[0].output_path).read_bytes()
    ).hexdigest()

    def should_not_decode(_row):
        raise AssertionError("resume must not decode")

    def should_not_write(_path, _audio, _sample_rate):
        raise AssertionError("resume must not write")

    resumed = Step6AudioExporter(
        str(output_root), decoder=should_not_decode, writer=should_not_write, workers=1,
    )._export_test_rows([row])

    assert resumed == first
    assert resumed[0].output_path == str(_step6_destination(output_root))
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 1


def test_distinct_producer_artifacts_use_isolated_digest_namespaces(tmp_path):
    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    first_row = _step6_row(source)
    second_row = _step6_row(source, final_selection={
        "version": 3, "export_eligible": True, "reason": "independently_selected",
    })

    first = Step6AudioExporter(str(output_root), workers=1)._export_test_rows([first_row])
    second = Step6AudioExporter(str(output_root), workers=1)._export_test_rows([second_row])

    first_path = Path(first[0].output_path)
    second_path = Path(second[0].output_path)
    assert first_path != second_path
    assert first_path.parent.parent.name != second_path.parent.parent.name
    assert first_path.is_file() and second_path.is_file()
    manifests = sorted((output_root / "audio_exports").glob(
        "step6/en/alpha/*/_export_manifest.v2.jsonl"
    ))
    assert len(manifests) == 2
    for manifest in manifests:
        record = json.loads(manifest.read_text(encoding="utf-8"))
        assert manifest.parent.name == record["producer"]["artifact_digest"]
        assert record["output_relative_path"].startswith(
            f"step6/en/alpha/{record['producer']['artifact_digest']}/"
        )


def test_exporter_resume_rejects_manifest_without_destination(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    row = _step6_row(source)
    output_root = tmp_path / "out"
    Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])
    _step6_destination(output_root).unlink()

    with pytest.raises(FileNotFoundError, match="destination is missing"):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])


@pytest.mark.parametrize("field", ["output_sha256", "source"])
def test_exporter_resume_rejects_tampered_manifest(tmp_path, field):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    row = _step6_row(source)
    output_root = tmp_path / "out"
    Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])
    manifest = _step6_manifest(output_root)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if field == "output_sha256":
        record[field] = "0" * 64
    else:
        record[field]["source_key"] = "tampered"
    manifest.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest/source metadata mismatch|hash"):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])


def test_exporter_rejects_unsafe_language_path_component(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    exporter = Step6AudioExporter(str(tmp_path / "out"), workers=1)

    for language in ("../zh", "en/us", r"en\\us", ".", ".."):
        with pytest.raises(ValueError, match="language.*safe single path component"):
            exporter._validate_row(_step6_row(source, language=language))


def test_exporter_resume_rejects_manifest_with_wrong_language_prefix(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    Step6AudioExporter(str(output_root), workers=1)._export_test_rows([_step6_row(source)])
    manifest = _step6_manifest(output_root)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["output_relative_path"] = record["output_relative_path"].replace(
        "step6/en/alpha/", "step6/zh/alpha/", 1,
    )
    manifest.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe output path"):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([_step6_row(source)])


def test_exporter_resume_rejects_output_content_tampering_with_same_metadata(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    row = _step6_row(source)
    output_root = tmp_path / "out"
    Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])
    destination = _step6_destination(output_root)
    audio, sample_rate = sf.read(destination, dtype="float32", always_2d=True)
    audio[0, 0] = -audio[0, 0]
    sf.write(destination, audio, sample_rate, format="FLAC", subtype="PCM_24")

    with pytest.raises(ValueError, match="hash"):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])


def test_exporter_explicit_root_must_be_isolated_from_output_and_input_roots(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    source.parent.mkdir()
    _write_source(source)
    with pytest.raises(ValueError, match="overlap output_root"):
        Step6AudioExporter(str(tmp_path / "out"), export_root=str(tmp_path / "out" / "exports"))
    with pytest.raises(ValueError, match="input_root"):
        Step6AudioExporter(str(tmp_path / "out"), export_root=str(source.parent / "exports"), workers=1)._export_test_rows([
            _step6_row(source),
        ])


def test_exporter_writer_failure_leaves_no_destination_or_manifest(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)

    def failing_writer(path, audio, sample_rate):
        sf.write(path, audio, sample_rate, format="FLAC", subtype="PCM_24")
        raise RuntimeError("writer failed after partial write")

    output_root = tmp_path / "out"
    with pytest.raises(RuntimeError, match="writer failed"):
        Step6AudioExporter(str(output_root), writer=failing_writer, workers=1)._export_test_rows([
            _step6_row(source),
        ])

    assert not list((output_root / "audio_exports").glob(
        "step6/en/alpha/*/16000hz/source--segment-1.flac"
    ))
    assert not list((output_root / "audio_exports").glob(
        "step6/en/alpha/*/_export_manifest.v2.jsonl"
    ))


def test_default_decoder_rejects_source_sha256_mutation(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    row = _step6_row(source)
    payload = bytearray(source.read_bytes())
    payload[-1] ^= 1
    source.write_bytes(payload)

    with pytest.raises(ValueError, match="sha256"):
        Step6AudioExporter(str(tmp_path / "out"), workers=1)._export_test_rows([row])
def test_exporter_resume_with_explicit_root_skips_decode_and_write(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    input_root = tmp_path / "input"
    input_root.mkdir()
    source = input_root / "source.wav"
    _write_source(source)
    output_root = tmp_path / "output"
    export_root = tmp_path / "exports"
    row = _step6_row(source)
    first = Step6AudioExporter(
        str(output_root), export_root=str(export_root), workers=1, max_pending=1,
    )._export_test_rows([row])

    def forbidden_decoder(_row):
        raise AssertionError("resume must not decode a published segment")

    def forbidden_writer(_path, _audio, _sample_rate):
        raise AssertionError("resume must not write a published segment")

    resumed = Step6AudioExporter(
        str(output_root),
        export_root=str(export_root),
        decoder=forbidden_decoder,
        writer=forbidden_writer,
        workers=1,
        max_pending=1,
    )._export_test_rows([row])

    assert resumed == first
    manifest = next(export_root.glob("step6/en/alpha/*/_export_manifest.v2.jsonl"))
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 1


def test_exporter_resume_retries_only_segments_missing_after_partial_failure(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    input_root = tmp_path / "input"
    input_root.mkdir()
    source = input_root / "source.wav"
    _write_source(source)
    output_root = tmp_path / "output"
    export_root = tmp_path / "exports"
    rows = [
        _step6_row(source, segment_id="segment-0"),
        _step6_row(source, segment_id="segment-1"),
    ]
    first_calls: list[str] = []

    def first_decoder(row):
        first_calls.append(row["segment_id"])
        if row["segment_id"] == "segment-1":
            raise RuntimeError("second segment failed")
        return np.zeros((8_000, 1), dtype=np.float32), 16_000

    exporter = Step6AudioExporter(
        str(output_root),
        export_root=str(export_root),
        decoder=first_decoder,
        workers=1,
        max_pending=1,
    )
    with pytest.raises(RuntimeError, match="second segment failed"):
        exporter._export_test_rows(rows)

    first_destination = next(export_root.glob(
        "step6/en/alpha/*/16000hz/source--segment-0.flac"
    ))
    assert first_destination.is_file()
    manifest = next(export_root.glob("step6/en/alpha/*/_export_manifest.v2.jsonl"))
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 1

    retry_calls: list[str] = []

    def retry_decoder(row):
        retry_calls.append(row["segment_id"])
        if row["segment_id"] == "segment-0":
            raise AssertionError("a published segment must not be decoded again")
        return np.zeros((8_000, 1), dtype=np.float32), 16_000

    result = Step6AudioExporter(
        str(output_root),
        export_root=str(export_root),
        decoder=retry_decoder,
        workers=1,
        max_pending=1,
    )._export_test_rows(rows)

    assert retry_calls == ["segment-1"]
    assert [item.segment_id for item in result] == ["segment-0", "segment-1"]
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.parametrize("tamper", ["manifest_hash", "output_metadata"])
def test_exporter_rejects_tampered_resume_state(tmp_path, tamper):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    input_root = tmp_path / "input"
    input_root.mkdir()
    source = input_root / "source.wav"
    _write_source(source)
    output_root = tmp_path / "output"
    export_root = tmp_path / "exports"
    row = _step6_row(source)
    exported = Step6AudioExporter(
        str(output_root), export_root=str(export_root), workers=1,
    )._export_test_rows([row])
    destination = Path(exported[0].output_path)
    manifest = next(export_root.glob("step6/en/alpha/*/_export_manifest.v2.jsonl"))

    if tamper == "manifest_hash":
        record = json.loads(manifest.read_text(encoding="utf-8"))
        record["output_sha256"] = "0" * 64
        manifest.write_text(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    else:
        sf.write(
            destination,
            np.zeros((4_000, 1), dtype=np.float32),
            8_000,
            format="WAV",
            subtype="PCM_16",
        )

    def forbidden_decoder(_row):
        raise AssertionError("tampered output must fail before decode")

    with pytest.raises(ValueError, match="hash|metadata"):
        Step6AudioExporter(
            str(output_root),
            export_root=str(export_root),
            decoder=forbidden_decoder,
            workers=1,
        )._export_test_rows([row])


def test_exporter_rejects_decoder_amplitude_and_leaves_no_final_output(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    input_root = tmp_path / "input"
    input_root.mkdir()
    source = input_root / "source.wav"
    _write_source(source)
    output_root = tmp_path / "output"
    export_root = tmp_path / "exports"
    row = _step6_row(source)

    def decoder(_row):
        return np.full((8_000, 1), 1.1, dtype=np.float32), 16_000

    with pytest.raises(ValueError, match="amplitude"):
        Step6AudioExporter(
            str(output_root), export_root=str(export_root), decoder=decoder, workers=1,
        )._export_test_rows([row])

    assert not list(export_root.glob(
        "step6/en/alpha/*/16000hz/source--segment-1.flac"
    ))


@pytest.mark.parametrize("stage", ["decoder", "resampler"])
def test_exporter_requires_float32_waveforms(tmp_path, stage):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    input_root = tmp_path / "input"
    input_root.mkdir()
    source = input_root / "source.wav"
    _write_source(source)
    row = _step6_row(source)

    def decoder(_row):
        return np.zeros((8_000, 1), dtype=np.float64 if stage == "decoder" else np.float32), 16_000

    def resampler(_audio, _source_rate, _target_rate):
        return np.zeros((4_000, 1), dtype=np.float64)

    with pytest.raises(ValueError, match=f"{stage}.*float32"):
        Step6AudioExporter(
            str(tmp_path / "output"),
            decoder=decoder,
            resampler=resampler,
            workers=1,
        )._export_test_rows([row])


@pytest.mark.parametrize("writer_kind", ["format", "frames", "channels"])
def test_exporter_rejects_writer_output_mismatch_before_publish(tmp_path, writer_kind):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    input_root = tmp_path / "input"
    input_root.mkdir()
    source = input_root / "source.wav"
    _write_source(source)
    output_root = tmp_path / "output"
    export_root = tmp_path / "exports"
    row = _step6_row(source)

    def writer(path, audio, sample_rate):
        if writer_kind == "format":
            sf.write(path, audio, sample_rate, format="WAV", subtype="PCM_16")
        elif writer_kind == "frames":
            sf.write(path, audio[:-1], sample_rate, format="FLAC", subtype="PCM_24")
        else:
            sf.write(
                path,
                np.column_stack((audio[:, 0], audio[:, 0])),
                sample_rate,
                format="FLAC",
                subtype="PCM_24",
            )

    with pytest.raises(ValueError, match="metadata"):
        Step6AudioExporter(
            str(output_root), export_root=str(export_root), writer=writer, workers=1,
        )._export_test_rows([row])

    assert not list(export_root.glob(
        "step6/en/alpha/*/16000hz/source--segment-1.flac"
    ))


def test_exporter_rejects_symlinked_output_path_components(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    input_root = tmp_path / "input"
    input_root.mkdir()
    source = input_root / "source.wav"
    _write_source(source)
    output_root = tmp_path / "output"
    export_root = tmp_path / "exports"
    export_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (export_root / "step6").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        Step6AudioExporter(
            str(output_root), export_root=str(export_root), workers=1,
        )._export_test_rows([_step6_row(source)])
    assert not list(outside.rglob("*.flac"))


def test_exporter_does_not_read_or_overwrite_legacy_symlink_layout(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    input_root = tmp_path / "input"
    input_root.mkdir()
    source = input_root / "source.wav"
    _write_source(source)
    output_root = tmp_path / "output"
    export_root = tmp_path / "exports"
    outside = tmp_path / "outside"
    outside.mkdir()
    row = _step6_row(source)

    manifest_parent = export_root / "step6" / "alpha"
    manifest_parent.mkdir(parents=True)
    manifest_target = outside / "manifest.jsonl"
    manifest_target.write_text("", encoding="utf-8")
    legacy_manifest = manifest_parent / "_export_manifest.jsonl"
    legacy_manifest.symlink_to(manifest_target)
    exported = Step6AudioExporter(
        str(output_root), export_root=str(export_root), workers=1,
    )._export_test_rows([row])
    assert Path(exported[0].output_path).is_file()
    assert legacy_manifest.is_symlink()
    assert manifest_target.read_bytes() == b""

    legacy_destination = (
        export_root / "step6" / "alpha" / "16000hz" / "source--segment-1.flac"
    )
    legacy_destination.parent.mkdir(parents=True, exist_ok=True)
    destination_target = outside / "destination.flac"
    destination_target.write_bytes(b"untouched")
    legacy_destination.symlink_to(destination_target)
    resumed = Step6AudioExporter(
        str(output_root), export_root=str(export_root), workers=1,
    )._export_test_rows([row])
    assert resumed == exported
    assert destination_target.read_bytes() == b"untouched"


def test_exporter_rejects_overlapping_explicit_export_and_input_roots(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    output_root = tmp_path / "output"
    with pytest.raises(ValueError, match="overlap.*output_root"):
        Step6AudioExporter(
            str(output_root), export_root=str(output_root / "exports"), workers=1,
        )

    export_root = tmp_path / "isolated-exports"
    export_root.mkdir()
    source = export_root / "source.wav"
    _write_source(source)
    with pytest.raises(ValueError, match="input_root overlaps"):
        Step6AudioExporter(
            str(output_root), export_root=str(export_root), workers=1,
        )._export_test_rows([_step6_row(source)])


def test_exporter_stops_new_decodes_after_bounded_failure_batch(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    decoded: list[str] = []

    def rows():
        for index in range(4):
            yield _step6_row(source, segment_id=f"segment-{index}")

    def decoder(row):
        decoded.append(row["segment_id"])
        if row["segment_id"] == "segment-0":
            raise RuntimeError("bounded batch failure")
        return np.zeros((8_000, 1), dtype=np.float32), 16_000

    with pytest.raises(RuntimeError, match="bounded batch failure"):
        Step6AudioExporter(
            str(tmp_path / "out"), decoder=decoder, workers=2, max_pending=2,
        )._export_test_rows(rows())
    assert set(decoded) <= {"segment-0", "segment-1"}
    assert "segment-0" in decoded


def test_manifest_append_failure_preserves_publication_for_recovery(
        tmp_path, monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import audio_export
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    row = _step6_row(source)

    def failing_append(_manifest, _record):
        raise ValueError("manifest append failed")

    real_append = audio_export._ManifestState.append
    monkeypatch.setattr(audio_export._ManifestState, "append", failing_append)
    with pytest.raises(ValueError, match="manifest append failed"):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])

    destination = _step6_destination(output_root)
    assert destination.is_file()
    assert _transaction_marker(destination).is_file()
    assert not list((output_root / "audio_exports").glob(
        "step6/en/alpha/*/_export_manifest.v2.jsonl"
    ))

    monkeypatch.setattr(audio_export._ManifestState, "append", real_append)
    retry = Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])
    assert [item.segment_id for item in retry] == [row["segment_id"]]
    assert destination.is_file()
    assert _step6_manifest(output_root).is_file()


def test_manifest_append_cleanup_failure_preserves_committed_audio_and_marker(
        tmp_path, monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import audio_export
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    row = _step6_row(source)
    real_release = audio_export._ManifestState._release_exclusive_file_lock
    release_calls = 0

    def release_then_fail(state, descriptor, primary):
        nonlocal release_calls
        result = real_release(state, descriptor, primary)
        release_calls += 1
        if primary is None and release_calls == 2:
            raise OSError("manifest append lock release failed")
        return result

    monkeypatch.setattr(
        audio_export._ManifestState, "_release_exclusive_file_lock", release_then_fail,
    )
    with pytest.raises(OSError, match="lock release failed"):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])

    destination = _step6_destination(output_root)
    assert destination.is_file()
    assert _transaction_marker(destination).is_file()
    assert len(_step6_manifest(output_root).read_text(encoding="utf-8").splitlines()) == 1

    monkeypatch.setattr(
        audio_export._ManifestState, "_release_exclusive_file_lock", real_release,
    )
    resumed = Step6AudioExporter(
        str(output_root),
        decoder=lambda _row: pytest.fail("committed record must resume without decode"),
        workers=1,
    )._export_test_rows([row])
    assert [item.segment_id for item in resumed] == [row["segment_id"]]
    assert not list(destination.parent.glob(".audio-export-staging-*/marker.json"))


def test_private_incomplete_marker_without_owner_is_preserved_fail_closed(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    row = _step6_row(source)
    first = Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])
    destination = Path(first[0].output_path)
    digest = hashlib.sha256(str(destination).encode("utf-8")).hexdigest()
    marker = destination.parent / f".audio-export-staging-{digest}" / "marker.json"
    marker.parent.mkdir()
    marker.write_bytes(b'{"schema":"incomplete')
    fake_temporary = marker.parent / f".{destination.stem}.{'a' * 32}.flac"
    fake_temporary.write_bytes(b"not exporter-owned")

    with pytest.raises(ValueError, match="canonical JSON line"):
        Step6AudioExporter(
            str(output_root),
            decoder=lambda _row: pytest.fail("untrusted staging must not decode"),
            workers=1,
        )._export_test_rows([row])

    assert marker.is_file()
    assert fake_temporary.read_bytes() == b"not exporter-owned"


def _crash_export_after_phase(
    output_root: str, row: dict[str, object], phase: str,
) -> None:
    from audio_filter_pipeline.processors.stages_v3 import audio_export
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    if phase == "audio_write":
        def crash_after_partial_audio(path, _audio, _sample_rate):
            with open(path, "wb") as stream:
                stream.write(b"fLaC")
                stream.flush()
                os.fsync(stream.fileno())
            os._exit(80)

        exporter = Step6AudioExporter(output_root, writer=crash_after_partial_audio, workers=1)
        exporter._export_test_rows([row])
        return
    if phase == "marker_write":
        original = Step6AudioExporter._write_all

        def crash_after_partial_write(descriptor, payload):
            if b'"producer"' not in payload:
                return original(descriptor, payload)
            os.write(descriptor, payload[: max(1, len(payload) // 2)])
            os._exit(81)

        Step6AudioExporter._write_all = staticmethod(crash_after_partial_write)
    elif phase == "marker_rename":
        original = audio_export.SafeOutputPath.rename_from_noreplace

        def crash_after_marker_rename(safe, source_safe):
            original(safe, source_safe)
            if safe.target_name == "marker.json":
                os._exit(82)

        audio_export.SafeOutputPath.rename_from_noreplace = crash_after_marker_rename
    elif phase == "audio_rename":
        original = audio_export.SafeOutputPath.rename_from_noreplace

        def crash_after_audio_rename(safe, source_safe):
            original(safe, source_safe)
            if safe.target_name.endswith(".flac"):
                os._exit(83)

        audio_export.SafeOutputPath.rename_from_noreplace = crash_after_audio_rename
    else:
        raise ValueError(f"unknown crash phase: {phase}")
    Step6AudioExporter(output_root, workers=1)._export_test_rows([row])


@pytest.mark.parametrize(
    "phase", ["audio_write", "marker_write", "marker_rename", "audio_rename"],
)
def test_cpu_subprocess_crash_matrix_recovers_private_export_state(tmp_path, phase):
    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    row = _step6_row(source)
    process = mp.get_context("fork").Process(
        target=_crash_export_after_phase, args=(str(output_root), row, phase),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == {
        "audio_write": 80, "marker_write": 81, "marker_rename": 82, "audio_rename": 83,
    }[phase]

    resumed = Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])
    destination = Path(resumed[0].output_path)
    assert destination.is_file()
    assert len(_step6_manifest(output_root).read_text(encoding="utf-8").splitlines()) == 1
    assert not list(destination.parent.glob(".audio-export-staging-*"))


def test_concurrent_exporters_share_destination_lock_and_publish_once(tmp_path):
    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    row = _step6_row(source)
    barrier = Barrier(2)
    results: list[tuple] = []
    failures: list[BaseException] = []

    def run_export() -> None:
        try:
            barrier.wait(timeout=3)
            results.append(Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row]))
        except BaseException as exc:
            failures.append(exc)

    first = Thread(target=run_export)
    second = Thread(target=run_export)
    first.start()
    second.start()
    first.join(timeout=10)
    second.join(timeout=10)

    assert not first.is_alive() and not second.is_alive()
    assert failures == []
    assert len(results) == 2
    destination = _step6_destination(output_root)
    assert destination.is_file()
    assert len(_step6_manifest(output_root).read_text(encoding="utf-8").splitlines()) == 1
    lock_domain = next(destination.parent.glob(".audio-export-lock-*"))
    assert (lock_domain / "anchor").is_file()
    assert (lock_domain / "anchor").stat().st_nlink == 1


def test_destination_lock_anchor_replacement_cannot_split_held_critical_section(tmp_path):
    output_root = tmp_path / "out"
    destination = output_root / "audio_exports" / "locks" / "target.flac"
    exporter = Step6AudioExporter(str(output_root), workers=1)
    first_entered = Event()
    release_first = Event()
    second_entered = Event()
    failures: list[BaseException] = []

    def hold_first() -> None:
        try:
            with exporter._destination_lock(destination):
                first_entered.set()
                assert release_first.wait(timeout=5)
        except BaseException as exc:
            failures.append(exc)

    def acquire_second() -> None:
        try:
            with exporter._destination_lock(destination):
                second_entered.set()
        except BaseException as exc:
            failures.append(exc)

    first = Thread(target=hold_first)
    first.start()
    assert first_entered.wait(timeout=5)
    lock_domain = next(destination.parent.glob(".audio-export-lock-*"))
    replacement = lock_domain / "anchor.replacement"
    replacement.write_bytes(b"replacement")
    os.replace(replacement, lock_domain / "anchor")

    second = Thread(target=acquire_second)
    second.start()
    assert not second_entered.wait(timeout=0.2)
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert failures == []
    assert second_entered.is_set()


def test_destination_lock_domains_keep_distinct_destinations_parallel(tmp_path):
    output_root = tmp_path / "out"
    first_destination = output_root / "audio_exports" / "locks" / "first.flac"
    second_destination = output_root / "audio_exports" / "locks" / "second.flac"
    exporter = Step6AudioExporter(str(output_root), workers=1)
    first_entered = Event()
    release_first = Event()
    second_entered = Event()

    def hold_first() -> None:
        with exporter._destination_lock(first_destination):
            first_entered.set()
            assert release_first.wait(timeout=5)

    def acquire_second() -> None:
        with exporter._destination_lock(second_destination):
            second_entered.set()

    first = Thread(target=hold_first)
    second = Thread(target=acquire_second)
    first.start()
    assert first_entered.wait(timeout=5)
    second.start()
    assert second_entered.wait(timeout=2)
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()


def test_exporter_writer_uses_descriptor_or_owned_hidden_temporary_path(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    writer_paths: list[str] = []

    def writer(path, audio, sample_rate):
        writer_paths.append(path)
        sf.write(path, audio, sample_rate, format="FLAC", subtype="PCM_24")

    exported = Step6AudioExporter(
        str(output_root), writer=writer, workers=1,
    )._export_test_rows([_step6_row(source)])

    assert len(writer_paths) == 1
    writer_path = writer_paths[0]
    destination = Path(exported[0].output_path)
    assert writer_path.startswith("/proc/self/fd/") or (
        Path(writer_path).name.startswith(".")
        and Path(writer_path).parent == destination.parent
    )
    assert writer_path != str(destination)
    assert not [
        path for path in destination.parent.iterdir()
        if path.name.startswith(".")
        and not path.name.startswith(".audio-export-lock-")
    ]


def test_audio_verification_does_not_use_full_soundfile_read(tmp_path, monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import audio_export
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    row = _step6_row(source)

    def forbidden_read(*_args, **_kwargs):
        raise AssertionError("verification must use SoundFile streaming")

    monkeypatch.setattr(audio_export.sf, "read", forbidden_read)
    exporter = Step6AudioExporter(
        str(tmp_path / "out"),
        decoder=lambda _row: (np.zeros((8_000, 1), dtype=np.float32), 16_000),
        resampler=lambda _audio, _source_rate, _target_rate: np.zeros(
            (8_000, 1), dtype=np.float32,
        ),
        workers=1,
    )

    result = exporter._export_test_rows([row])

    assert len(result) == 1
    assert Path(result[0].output_path).is_file()


def test_configured_single_task_budget_rejects_limit_minus_one_before_decode(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    row = _step6_row(source)
    estimated = Step6AudioExporter._estimated_inflight_bytes(row)
    decoder_calls: list[str] = []

    def decoder(_row):
        decoder_calls.append("called")
        return np.zeros((8_000, 1), dtype=np.float32), 16_000

    with pytest.raises(ValueError, match="configured estimated waveform/verification"):
        Step6AudioExporter(
            str(tmp_path / "rejected"), decoder=decoder, workers=1,
            max_inflight_bytes=estimated - 1,
        )._export_test_rows([row])
    assert decoder_calls == []


def test_configured_single_task_budget_allows_exact_estimate(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    row = _step6_row(source)
    estimated = Step6AudioExporter._estimated_inflight_bytes(row)
    decoder_calls: list[str] = []

    def decoder(_row):
        decoder_calls.append("called")
        return np.zeros((8_000, 1), dtype=np.float32), 16_000

    exported = Step6AudioExporter(
        str(tmp_path / "allowed"), decoder=decoder, workers=1,
        max_inflight_bytes=estimated,
    )._export_test_rows([row])

    assert len(exported) == 1
    assert decoder_calls == ["called"]


def test_estimated_inflight_budget_includes_buffered_verification_snapshot(
        tmp_path, monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import audio_export
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    row = _step6_row(source)

    monkeypatch.setattr(audio_export, "_AUDIO_VERIFY_BUFFERED_MAX_BYTES", 0)
    without_snapshot = Step6AudioExporter._estimated_inflight_bytes(row)
    snapshot_budget = 8 * 1024 * 1024
    monkeypatch.setattr(
        audio_export, "_AUDIO_VERIFY_BUFFERED_MAX_BYTES", snapshot_budget,
    )
    with_snapshot = Step6AudioExporter._estimated_inflight_bytes(row)

    assert with_snapshot - without_snapshot == snapshot_budget


def test_single_task_budget_includes_verification_margin_and_rejects_before_decode(
        tmp_path, monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import audio_export
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    row = _step6_row(source)
    exporter = Step6AudioExporter(str(tmp_path / "out"), workers=1)
    estimated = exporter._estimated_inflight_bytes(row)

    assert estimated >= audio_export._AUDIO_VERIFY_MARGIN_BYTES
    decoder_calls: list[str] = []

    def decoder(_row):
        decoder_calls.append("called")
        return np.zeros((8_000, 1), dtype=np.float32), 16_000

    monkeypatch.setattr(
        audio_export, "_MAX_SINGLE_TASK_INFLIGHT_BYTES", estimated - 1,
    )
    with pytest.raises(ValueError, match="single export task exceeds"):
        Step6AudioExporter(
            str(tmp_path / "out"), decoder=decoder, workers=1,
        )._export_test_rows([row])
    assert decoder_calls == []


@pytest.mark.parametrize("invalid_kind", ["duplicate", "nan", "infinity"])
def test_manifest_rejects_duplicate_keys_and_nonfinite_json(tmp_path, invalid_kind):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    Step6AudioExporter(str(output_root), workers=1)._export_test_rows([_step6_row(source)])
    manifest = _step6_manifest(output_root)
    raw = manifest.read_text(encoding="utf-8")

    if invalid_kind == "duplicate":
        field_start = raw.index('"schema":')
        field_end = raw.index(",", field_start)
        field = raw[field_start:field_end]
        raw = raw[:field_end] + "," + field + raw[field_end:]
    else:
        token = "NaN" if invalid_kind == "nan" else "Infinity"
        marker = '"frames":8000'
        assert marker in raw
        raw = raw.replace(marker, f'"frames":{token}', 1)
    manifest.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="strict JSON"):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([_step6_row(source)])


def test_manifest_truncates_only_incomplete_eof_tail_during_resume(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    row = _step6_row(source)
    Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])
    manifest = _step6_manifest(output_root)
    original = manifest.read_bytes()
    manifest.write_bytes(original + b'{"schema":"incomplete-tail"')

    def forbidden_decoder(_row):
        raise AssertionError("complete manifest record must resume without decode")

    resumed = Step6AudioExporter(
        str(output_root), decoder=forbidden_decoder, workers=1,
    )._export_test_rows([row])

    assert [item.segment_id for item in resumed] == [row["segment_id"]]
    assert manifest.read_bytes() == original


def test_manifest_enforces_bounded_bytes_and_record_count(tmp_path, monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import audio_export
    from audio_filter_pipeline.processors.stages_v3.audio_export import (
        Step6AudioExporter,
        _ManifestState,
    )

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    Step6AudioExporter(str(output_root), workers=1)._export_test_rows([_step6_row(source)])
    record = json.loads(_step6_manifest(output_root).read_text(encoding="utf-8"))
    payload_size = len(audio_export._canonical_json_line(record))

    bytes_state = _ManifestState(
        tmp_path / "bytes-root", "step6", "alpha", record["producer"], language="en",
    )
    monkeypatch.setattr(audio_export, "_MANIFEST_MAX_BYTES", payload_size - 1)
    with pytest.raises(ValueError, match="total size"):
        bytes_state.append(record)

    records_state = _ManifestState(
        tmp_path / "records-root", "step6", "alpha", record["producer"], language="en",
    )
    monkeypatch.setattr(audio_export, "_MANIFEST_MAX_BYTES", payload_size + 1)
    monkeypatch.setattr(audio_export, "_MANIFEST_MAX_RECORDS", 0)
    with pytest.raises(ValueError, match="record count"):
        records_state.append(record)


def test_manifest_keyboard_interrupt_leaves_journal_for_recovery(tmp_path, monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import audio_export
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    row = _step6_row(source)
    real_append = audio_export._ManifestState.append
    append_calls = 0

    def interrupt_once(manifest, record):
        nonlocal append_calls
        append_calls += 1
        if append_calls == 1:
            raise KeyboardInterrupt("simulated manifest interruption")
        return real_append(manifest, record)

    monkeypatch.setattr(audio_export._ManifestState, "append", interrupt_once)
    with pytest.raises(KeyboardInterrupt, match="simulated manifest interruption"):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])

    destination = _step6_destination(output_root)
    marker = _transaction_marker(destination)
    assert destination.is_file()
    assert not list((output_root / "audio_exports").glob(
        "step6/en/alpha/*/_export_manifest.v2.jsonl"
    ))
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    assert marker_value["record"]["output_relative_path"].endswith(
        "source--segment-1.flac"
    )

    resumed = Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])

    assert [item.segment_id for item in resumed] == [row["segment_id"]]
    assert _step6_manifest(output_root).is_file()
    assert len(_step6_manifest(output_root).read_text(encoding="utf-8").splitlines()) == 1
    assert not list(destination.parent.glob(".audio-export-staging-*/marker.json"))


def test_transaction_recovery_preserves_foreign_destination_inode(tmp_path, monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import audio_export
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    row = _step6_row(source)
    real_append = audio_export._ManifestState.append

    def interrupt_append(_manifest, _record):
        raise KeyboardInterrupt("simulated manifest interruption")

    monkeypatch.setattr(audio_export._ManifestState, "append", interrupt_append)
    with pytest.raises(KeyboardInterrupt):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])

    destination = _step6_destination(output_root)
    marker = _transaction_marker(destination)
    foreign_path = tmp_path / "foreign-destination.flac"
    foreign_path.write_bytes(b"foreign destination")
    os.replace(foreign_path, destination)
    foreign_stat = os.stat(destination)
    foreign_identity = (foreign_stat.st_dev, foreign_stat.st_ino)
    monkeypatch.setattr(audio_export._ManifestState, "append", real_append)

    with pytest.raises(ValueError, match="foreign destination inode"):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])

    assert destination.read_bytes() == b"foreign destination"
    assert (os.stat(destination).st_dev, os.stat(destination).st_ino) == foreign_identity
    assert marker.is_file()


def test_transaction_rejects_forged_temporary_name_without_deleting_foreign_file(
        tmp_path, monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import audio_export
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    row = _step6_row(source)
    real_append = audio_export._ManifestState.append

    def interrupt_append(_manifest, _record):
        raise KeyboardInterrupt("simulated manifest interruption")

    monkeypatch.setattr(audio_export._ManifestState, "append", interrupt_append)
    with pytest.raises(KeyboardInterrupt):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])

    destination = _step6_destination(output_root)
    marker = _transaction_marker(destination)
    forged_temporary = destination.parent / ".user-owned"
    os.link(destination, forged_temporary)
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    marker_value["temp_name"] = forged_temporary.name
    marker.write_bytes(audio_export._canonical_json_line(marker_value))
    foreign_identity = (os.stat(forged_temporary).st_dev, os.stat(forged_temporary).st_ino)
    monkeypatch.setattr(audio_export._ManifestState, "append", real_append)

    with pytest.raises(ValueError, match="temporary name"):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])

    assert forged_temporary.is_file()
    assert (os.stat(forged_temporary).st_dev, os.stat(forged_temporary).st_ino) == foreign_identity
    assert marker.is_file()


def test_transaction_recovery_renames_verified_temporary_when_destination_is_missing(
        tmp_path, monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import audio_export
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    row = _step6_row(source)
    real_rename = audio_export.SafeOutputPath.rename_from_noreplace

    def interrupt_rename(safe, source_safe):
        if safe.target_name.endswith(".flac"):
            raise KeyboardInterrupt("simulated publish interruption")
        return real_rename(safe, source_safe)

    monkeypatch.setattr(audio_export.SafeOutputPath, "rename_from_noreplace", interrupt_rename)
    with pytest.raises(KeyboardInterrupt):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])

    marker = next((output_root / "audio_exports").glob(
        "step6/en/alpha/*/16000hz/.audio-export-staging-*/marker.json"
    ))
    destination = Path(json.loads(marker.read_text(encoding="utf-8"))["target_path"])
    temporary = next(
        path for path in marker.parent.iterdir()
        if path.name.startswith(f".{destination.stem}.")
        and path.name.endswith(destination.suffix)
    )
    temporary_bytes = temporary.read_bytes()
    monkeypatch.setattr(audio_export.SafeOutputPath, "rename_from_noreplace", real_rename)

    Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])

    assert destination.read_bytes() == temporary_bytes
    assert not marker.exists()
    assert not temporary.exists()
    assert len(_step6_manifest(output_root).read_text(encoding="utf-8").splitlines()) == 1


def test_transaction_recovery_preserves_scene_when_destination_races_rename(
        tmp_path, monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import audio_export
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    row = _step6_row(source)
    real_rename = audio_export.SafeOutputPath.rename_from_noreplace

    def interrupt_rename(safe, source_safe):
        if safe.target_name.endswith(".flac"):
            raise KeyboardInterrupt("simulated publish interruption")
        return real_rename(safe, source_safe)

    monkeypatch.setattr(audio_export.SafeOutputPath, "rename_from_noreplace", interrupt_rename)
    with pytest.raises(KeyboardInterrupt):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])

    marker = next((output_root / "audio_exports").glob(
        "step6/en/alpha/*/16000hz/.audio-export-staging-*/marker.json"
    ))
    destination = Path(json.loads(marker.read_text(encoding="utf-8"))["target_path"])
    temporary = next(
        path for path in marker.parent.iterdir()
        if path.name.startswith(f".{destination.stem}.")
        and path.name.endswith(destination.suffix)
    )
    temporary_identity = (temporary.stat().st_dev, temporary.stat().st_ino)
    foreign_bytes = b"foreign destination"

    def racing_rename(safe, _source):
        descriptor = os.open(
            safe.target_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
            dir_fd=safe.parent_fd,
        )
        try:
            os.write(descriptor, foreign_bytes)
        finally:
            os.close(descriptor)
        raise FileExistsError("simulated competing destination")

    monkeypatch.setattr(audio_export.SafeOutputPath, "rename_from_noreplace", racing_rename)
    with pytest.raises(FileExistsError, match="competing"):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])

    assert destination.read_bytes() == foreign_bytes
    assert (temporary.stat().st_dev, temporary.stat().st_ino) == temporary_identity
    assert marker.is_file()


def test_manifest_rejects_hard_linked_manifest_and_lock(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    row = _step6_row(source)
    Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])
    manifest = _step6_manifest(output_root)
    lock = manifest.with_name(f"{manifest.name}.lock")

    manifest_peer = tmp_path / "manifest-peer"
    os.link(manifest, manifest_peer)
    with pytest.raises(ValueError, match="hard-linked"):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])
    manifest_peer.unlink()

    lock_peer = tmp_path / "lock-peer"
    os.link(lock, lock_peer)
    with pytest.raises(ValueError, match="hard-linked"):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])


def test_manifest_rejects_complete_eof_json_without_lf(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    row = _step6_row(source)
    Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])
    manifest = _step6_manifest(output_root)
    original = manifest.read_bytes()
    assert original.endswith(b"\n")
    manifest.write_bytes(original[:-1])

    with pytest.raises(ValueError, match="complete JSON.*LF"):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])

    assert manifest.read_bytes() == original[:-1]


def test_manifest_rollback_preserves_foreign_append(tmp_path, monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import audio_export
    from audio_filter_pipeline.processors.stages_v3.audio_export import (
        Step6AudioExporter,
        _ManifestState,
    )

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    row = _step6_row(source)
    Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])
    manifest_path = _step6_manifest(output_root)
    original = manifest_path.read_bytes()
    first_record = json.loads(original)
    second_record = dict(first_record)
    second_record["segment_id"] = "segment-2"
    second_record["output_relative_path"] = (
        f"step6/en/alpha/{first_record['producer']['artifact_digest']}/"
        "16000hz/source--segment-2.flac"
    )
    payload = audio_export._canonical_json_line(second_record)
    target_stat = manifest_path.stat()
    real_fsync = audio_export.os.fsync
    injected = False

    def fsync_with_foreign_append(descriptor):
        nonlocal injected
        result = real_fsync(descriptor)
        current = os.fstat(descriptor)
        if not injected and (current.st_dev, current.st_ino) == (
                target_stat.st_dev, target_stat.st_ino):
            foreign_descriptor = os.open(
                manifest_path, os.O_WRONLY | os.O_APPEND,
            )
            try:
                os.write(foreign_descriptor, b"foreign-append")
            finally:
                os.close(foreign_descriptor)
            injected = True
        return result

    monkeypatch.setattr(audio_export.os, "fsync", fsync_with_foreign_append)
    state = _ManifestState(
        output_root / "audio_exports", "step6", "alpha", second_record["producer"], language="en",
    )
    with pytest.raises(ValueError, match="changed after append") as raised:
        state.append(second_record)

    assert injected
    assert any("rollback failed" in note for note in raised.value.__notes__)
    assert manifest_path.read_bytes() == original + payload + b"foreign-append"


def test_default_decoder_reuses_verified_session_and_reports_full_read_counts(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    first = _step6_row(source, segment_id="segment-1", start_sample=0, end_sample=8_000)
    first["export_routing"] = dict(
        first["export_routing"], segment_interval_samples=[0, 8_000],
    )
    second = _step6_row(
        source, segment_id="segment-2", start_sample=8_000, end_sample=16_000,
    )
    second["export_routing"] = dict(
        second["export_routing"], segment_interval_samples=[8_000, 16_000],
    )
    rows = [first, second]
    exporter = Step6AudioExporter(
        str(tmp_path / "out"), workers=2, max_pending=2,
    )

    exported = exporter._export_test_rows(rows)

    assert [item.segment_id for item in exported] == ["segment-1", "segment-2"]
    assert exporter.last_io_counts == {
        "source_sessions_opened": 1,
        "source_full_sha_reads": 1,
        "source_full_sha_bytes": source.stat().st_size,
        "output_verify_calls": 4,
        "output_full_read_passes": 4,
        "output_estimated_full_read_bytes": sum(
            2 * Path(item.output_path).stat().st_size for item in exported
        ),
        "interval_decode_calls": 2,
        "interval_decode_frames": 16_000,
        "interval_decode_bytes": 16_000 * np.dtype(np.float32).itemsize,
        "source_sessions_closed": 1,
        "output_verify_calls_by_phase": {
            "published_destination": 2,
            "temporary_publish": 2,
        },
    }


def test_default_decoder_closes_verified_session_after_downstream_failure(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)

    def fail_resample(*_args, **_kwargs):
        raise RuntimeError("resample failed")

    exporter = Step6AudioExporter(
        str(tmp_path / "out"),
        resampler=fail_resample,
        workers=1,
        max_pending=1,
    )

    with pytest.raises(RuntimeError, match="resample failed"):
        exporter._export_test_rows([_step6_row(source)])

    assert exporter.last_io_counts["source_sessions_opened"] == 1
    assert exporter.last_io_counts["source_sessions_closed"] == 1
    assert exporter.last_io_counts["source_full_sha_reads"] == 1
    assert exporter.last_io_counts["interval_decode_calls"] == 1


def test_manifest_load_uses_signature_fast_path_and_reloads_external_change(
        tmp_path, monkeypatch):
    from audio_filter_pipeline.processors.stages_v3.audio_export import (
        Step6AudioExporter,
        _ManifestState,
    )

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    Step6AudioExporter(str(output_root), workers=1)._export_test_rows([_step6_row(source)])
    manifest_path = _step6_manifest(output_root)
    producer = json.loads(manifest_path.read_text(encoding="utf-8"))["producer"]
    state = _ManifestState(
        output_root / "audio_exports", "step6", "alpha", producer, language="en",
    )
    real_load_unlocked = state._load_unlocked
    load_calls = 0

    def recording_load():
        nonlocal load_calls
        load_calls += 1
        return real_load_unlocked()

    monkeypatch.setattr(state, "_load_unlocked", recording_load)
    state.load()
    state.load()
    assert load_calls == 1

    original = manifest_path.read_bytes()
    manifest_path.write_bytes(original)
    state.load()

    assert load_calls == 2
    relative = json.loads(original)["output_relative_path"]
    assert state.get_cached(relative) is not None


def test_manifest_digest_tracks_load_append_and_reload(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import (
        Step6AudioExporter,
        _ManifestState,
    )

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    Step6AudioExporter(str(output_root), workers=1)._export_test_rows([_step6_row(source)])
    manifest_path = _step6_manifest(output_root)
    first_record = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = _ManifestState(
        output_root / "audio_exports", "step6", "alpha", first_record["producer"], language="en",
    )

    state.load()
    assert state._manifest_digest is not None
    assert state._manifest_digest.hexdigest() == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    second_record = dict(first_record)
    second_record["segment_id"] = "segment-2"
    second_record["output_relative_path"] = (
        first_record["output_relative_path"].rsplit("/", 1)[0]
        + "/source--segment-2.flac"
    )
    state.append(second_record)
    manifest_bytes = manifest_path.read_bytes()
    assert state._manifest_digest is not None
    assert state._manifest_digest.hexdigest() == hashlib.sha256(manifest_bytes).hexdigest()

    rebuilt = _ManifestState(
        output_root / "audio_exports", "step6", "alpha", first_record["producer"], language="en",
    )
    rebuilt.load()
    assert rebuilt._manifest_digest is not None
    assert rebuilt._manifest_digest.hexdigest() == hashlib.sha256(manifest_bytes).hexdigest()
    assert rebuilt.get_cached(second_record["output_relative_path"]) == second_record


def test_manifest_append_success_uses_cached_digest_without_prefix_rehash(
        tmp_path, monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import audio_export
    from audio_filter_pipeline.processors.stages_v3.audio_export import (
        Step6AudioExporter,
        _ManifestState,
    )

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    Step6AudioExporter(str(output_root), workers=1)._export_test_rows([_step6_row(source)])
    manifest_path = _step6_manifest(output_root)
    first_record = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = _ManifestState(
        output_root / "audio_exports", "step6", "alpha", first_record["producer"], language="en",
    )
    state.load()
    second_record = dict(first_record)
    second_record["segment_id"] = "segment-2"
    second_record["output_relative_path"] = (
        first_record["output_relative_path"].rsplit("/", 1)[0]
        + "/source--segment-2.flac"
    )

    def forbidden_prefix_hash(*_args, **_kwargs):
        raise AssertionError("normal append must use the cached manifest digest")

    monkeypatch.setattr(audio_export, "_hash_descriptor_region", forbidden_prefix_hash)
    state.append(second_record)

    assert len(manifest_path.read_bytes().splitlines()) == 2


def test_export_artifact_rejects_hand_constructed_index_without_capability(tmp_path):
    from audio_filter_pipeline.utils.artifacts import ArtifactIndex

    index = ArtifactIndex(
        schema_version=2,
        layout_version=3,
        step="step6",
        dataset_id="alpha",
        config_digest="a" * 64,
        lineage_digest="b" * 64,
        row_count=0,
        rows_sha256=hashlib.sha256(b"").hexdigest(),
        artifact_digest="c" * 64,
        artifact_dir=str(tmp_path / "step6" / "alpha"),
        output_root=str(tmp_path),
    )

    with pytest.raises(ValueError, match="production-bound ArtifactIndex"):
        Step6AudioExporter(str(tmp_path / "out"), workers=1).export_artifact(index)


def test_export_artifact_rejects_inspect_unbound_index(tmp_path):
    from audio_filter_pipeline.utils.artifacts import (
        ArtifactWriter,
        PipelineStep,
        inspect_artifact_unbound,
    )

    artifact_root = tmp_path / "artifacts"
    writer = ArtifactWriter(
        artifact_root, PipelineStep.STEP6, "alpha", "a" * 64, "b" * 64,
    )
    writer.append({"dataset_id": "alpha"})
    writer.finalize()
    index = inspect_artifact_unbound(artifact_root, PipelineStep.STEP6, "alpha")

    with pytest.raises(ValueError, match="production-bound ArtifactIndex"):
        Step6AudioExporter(str(tmp_path / "out"), workers=1).export_artifact(index)


def test_prepare_candidate_deep_snapshots_mutable_row_before_prefetch(tmp_path):
    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    row = _step6_row(source)
    exporter = Step6AudioExporter(str(tmp_path / "out"), workers=1)

    candidate = exporter._prepare_candidate(row)
    row["export_routing"]["segment_interval_samples"][0] = 123
    row["final_selection"]["reason"] = "mutated-after-snapshot"

    assert candidate["export_routing"]["segment_interval_samples"] == [4_000, 12_000]
    assert candidate["final_selection"]["reason"] == "mos_selected"
    assert candidate["export_routing"] is not row["export_routing"]


def test_verify_audio_descriptor_buffered_snapshot_matches_streaming_fallback(
    tmp_path, monkeypatch,
):
    from audio_filter_pipeline.processors.stages_v3 import audio_export

    target = tmp_path / "published.flac"
    audio = np.linspace(-0.25, 0.25, 8_000, dtype=np.float32).reshape(-1, 1)
    sf.write(target, audio, 16_000, format="FLAC", subtype="PCM_24")
    expected = {
        "format": "FLAC", "subtype": "PCM_24", "sample_rate": 16_000,
        "frames": 8_000, "channels": 1,
    }
    digest = hashlib.sha256(target.read_bytes()).hexdigest()

    descriptor = os.open(target, os.O_RDONLY)
    try:
        metadata, buffered_digest = audio_export._verify_audio_descriptor(
            descriptor, expected, str(target),
        )
    finally:
        os.close(descriptor)
    assert metadata == expected
    assert buffered_digest == digest

    # 阈值归零强制走流式 decode + 分块 hash 的两遍回退路径，结果必须一致。
    monkeypatch.setattr(audio_export, "_AUDIO_VERIFY_BUFFERED_MAX_BYTES", 0)
    descriptor = os.open(target, os.O_RDONLY)
    try:
        fallback_metadata, fallback_digest = audio_export._verify_audio_descriptor(
            descriptor, expected, str(target),
        )
    finally:
        os.close(descriptor)
    assert fallback_metadata == metadata
    assert fallback_digest == digest


def test_resume_verification_runs_on_worker_threads_and_keeps_input_order(
    tmp_path, monkeypatch,
):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    rows = [
        _step6_row(source, segment_id=f"segment-{index}") for index in range(4)
    ]
    first = Step6AudioExporter(
        str(tmp_path / "out"), workers=2, max_pending=2,
    )._export_test_rows(rows)

    main_thread = threading.get_ident()
    resume_threads: set[int] = set()
    active_resume = 0
    peak_resume = 0
    lock = Lock()
    resume_barrier = Barrier(2)
    original = Step6AudioExporter._resume_existing

    def recording_resume(self, row, destination, relative, manifest, counters):
        nonlocal active_resume, peak_resume
        with lock:
            resume_threads.add(threading.get_ident())
            active_resume += 1
            peak_resume = max(peak_resume, active_resume)
        try:
            resume_barrier.wait(timeout=2)
            return original(self, row, destination, relative, manifest, counters)
        finally:
            with lock:
                active_resume -= 1

    def should_not_decode(_row):
        raise AssertionError("resume must not decode")

    monkeypatch.setattr(Step6AudioExporter, "_resume_existing", recording_resume)
    resumed = Step6AudioExporter(
        str(tmp_path / "out"), decoder=should_not_decode, workers=2, max_pending=2,
    )._export_test_rows(rows)

    assert resumed == first
    assert [item.segment_id for item in resumed] == [
        f"segment-{index}" for index in range(4)
    ]
    # resume 复验必须分发到 worker 线程，不再占用 producer 主线程串行执行。
    assert resume_threads
    assert main_thread not in resume_threads
    assert peak_resume == 2


def test_default_export_verifies_source_binding_before_publish(tmp_path):
    source = tmp_path / "input" / "source.wav"
    replacement = tmp_path / "input" / "replacement.wav"
    _write_source(source)
    _write_source(replacement)
    calls = []

    def mutate_source_after_decode(audio, sample_rate, target_rate):
        source.rename(tmp_path / "input" / "held-source.wav")
        replacement.rename(source)
        return audio

    def forbidden_writer(_path, _audio, _sample_rate):
        calls.append("write")

    with pytest.raises(ValueError, match="binding"):
        Step6AudioExporter(
            str(tmp_path / "out"),
            resampler=mutate_source_after_decode,
            writer=forbidden_writer,
            workers=1,
            max_pending=1,
        )._export_test_rows([_step6_row(source)])

    assert calls == []
    assert not list((tmp_path / "out" / "audio_exports").glob(
        "step6/en/alpha/*/16000hz/*.flac"
    ))


def _legacy_v2_marker_path(destination: Path) -> Path:
    return destination.parent / (
        f".audio-export-transaction-"
        f"{hashlib.sha256(str(destination).encode()).hexdigest()}.json"
    )


def _leave_interrupted_export(output_root: str, row: dict[str, object]) -> Path:
    """让一次导出停在 manifest 落账前，返回留下的 v3 事务 marker 路径。"""
    from audio_filter_pipeline.processors.stages_v3 import audio_export

    def interrupt_append(_manifest, _record):
        raise KeyboardInterrupt("leave published transaction for inspection")

    original = audio_export._ManifestState.append
    audio_export._ManifestState.append = interrupt_append
    try:
        with pytest.raises(KeyboardInterrupt):
            Step6AudioExporter(output_root, workers=1)._export_test_rows([row])
    finally:
        audio_export._ManifestState.append = original
    return _transaction_marker(_step6_destination(Path(output_root)))


def test_legacy_v2_transaction_marker_is_rejected(tmp_path):
    """残留的 v2 export marker 不再读取：导出 fail-fast 且不改写任何现场文件。"""
    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    row = _step6_row(source)
    private_marker = _leave_interrupted_export(str(output_root), row)
    destination = _step6_destination(output_root)
    legacy_marker = _legacy_v2_marker_path(destination)
    legacy_marker.write_bytes(
        b'{"schema":"audio_filter_pipeline.audio_export_transaction.v2"}\n'
    )

    with pytest.raises(ValueError, match="legacy v2 export transaction marker"):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])

    assert legacy_marker.is_file(), "拒绝路径不得删除 v2 marker"
    assert private_marker.is_file(), "拒绝路径不得清理 v3 staging 现场"
    assert destination.is_file()


def test_legacy_v2_marker_rejected_before_incomplete_private_cleanup(tmp_path):
    """v2 marker 优先于 private marker 修复路径：即使 private marker 残缺也直接拒绝。"""
    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    row = _step6_row(source)
    private_marker = _leave_interrupted_export(str(output_root), row)
    private_marker.write_bytes(b'{"schema":"incomplete')
    destination = _step6_destination(output_root)
    legacy_marker = _legacy_v2_marker_path(destination)
    legacy_marker.write_bytes(
        b'{"schema":"audio_filter_pipeline.audio_export_transaction.v2"}\n'
    )

    with pytest.raises(ValueError, match="legacy v2 export transaction marker"):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])

    assert legacy_marker.is_file()
    assert private_marker.is_file()
    assert private_marker.parent.is_dir()
