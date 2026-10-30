"""汇合后并发、缓存、锁与观测计数专项回归测试。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Event

import numpy as np
import pytest
import soundfile as sf

import audio_filter_pipeline.processors.stages_v3.audio_export as audio_export
import audio_filter_pipeline.processors.stages_v3.audio_export_scheduler as scheduler
import audio_filter_pipeline.processors.stages_v3.mos_gpu_job as mos_gpu_job
import audio_filter_pipeline.processors.workers.step3_mos_worker as mos_worker
from audio_filter_pipeline.processors.stages_v3.audio_export import (
    Step4AudioExporter,
    Step6AudioExporter,
)
from audio_filter_pipeline.processors.stages_v3.audio_export_scheduler import (
    ExportIOCounters,
    VerifiedSourceSessionPool,
)
from audio_filter_pipeline.processors.stages_v3.export import STEP6_ROW_FIELDS
from audio_filter_pipeline.processors.stages_v3.mos_filter import STEP4_ROW_FIELDS
from audio_filter_pipeline.utils.run_lock import RunLock, RunLockError
from audio_filter_pipeline.tests.audio_export_test_helpers import (
    install_export_rows_stub,
)


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _export_rows_stub(monkeypatch):
    install_export_rows_stub(monkeypatch, Step4AudioExporter, Step6AudioExporter)


def _write_source(path: Path, frames: int = 16_000) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.linspace(-0.75, 0.75, frames, dtype=np.float32)
    sf.write(path, samples, 16_000, subtype="FLOAT")
    return samples


def _step3_row(
    source: Path, segment_id: str, start: int, end: int,
) -> dict[str, object]:
    return {
        "source_key": "shared-source",
        "dataset_id": "alpha",
        "language": "en",
        "category": "speech",
        "quality_profile": "quality_high",
        "input_root": str(source.parent.resolve()),
        "original_path": str(source.resolve()),
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
        "signal_peak": 0.75,
        "signal_clipping_ratio": 0.0,
        "signal_dc_offset": 0.0,
        "signal_rms_dbfs": -7.3,
        "signal_num_samples": 16_000,
        "bandwidth_hz": 3_000.0,
        "estimated_sr": 8_000,
        "vad_speech_ratio": 1.0,
        "vad_timeline_sample_rate": 16_000,
        "vad_timeline_num_samples": 16_000,
        "segment_id": segment_id,
        "start_sample": start,
        "end_sample": end,
        "boundary_reason": "test_interval",
        "mos_models": ["scoreq"],
        "mos_scoring_policy_hash": "e" * 64,
        "mos_scores": {"scoreq": 4.5},
    }


def _step4_row(
    source: Path, segment_id: str, start: int, end: int,
) -> dict[str, object]:
    return {
        **_step3_row(source, segment_id, start, end),
        "mos_selection": {
            "decision": "keep",
            "passed_models": 1,
            "required_models": 1,
            "policy_id": "alpha:en",
            "policy_hash": "f" * 64,
        },
    }


def _row(source: Path, segment_id: str, start: int, end: int) -> dict[str, object]:
    return {
        **_step3_row(source, segment_id, start, end),
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
            "source_path": str(source.resolve()),
            "segment_interval_samples": [start, end],
        },
    }


def test_step4_and_step6_rows_use_exact_incompatible_artifact_contracts(tmp_path):
    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    step4_row = _step4_row(source, "segment-00", 0, 500)
    step6_row = _row(source, "segment-00", 0, 500)

    assert tuple(step4_row) == STEP4_ROW_FIELDS
    assert tuple(step6_row) == STEP6_ROW_FIELDS
    with pytest.raises(ValueError, match="step6 row fields.*exact contract"):
        Step6AudioExporter(str(tmp_path / "step6-out"), workers=1)._export_test_rows([
            step4_row,
        ])
    with pytest.raises(ValueError, match="step4 row fields.*exact contract"):
        Step4AudioExporter(str(tmp_path / "step4-out"), workers=1)._export_test_rows([
            step6_row,
        ])


def test_thread_pool_shared_session_has_no_offset_race_and_hashes_source_once(tmp_path):
    source = tmp_path / "input" / "source.wav"
    expected = _write_source(source)
    interval_frames = 500
    rows = [
        _row(source, f"segment-{index:02d}", start, start + interval_frames)
        for index, start in enumerate(range(0, len(expected), interval_frames))
    ]
    exporter = Step6AudioExporter(
        str(tmp_path / "out"), workers=8, max_pending=16,
    )

    exported = exporter._export_test_rows(rows)

    decoded = []
    for item in exported:
        audio, sample_rate = sf.read(item.output_path, dtype="float32", always_2d=True)
        assert sample_rate == 16_000
        decoded.append(audio[:, 0])
    np.testing.assert_allclose(
        np.concatenate(decoded), expected, rtol=0.0, atol=2.0 / (1 << 24),
    )
    counts = exporter.last_io_counts
    assert counts["source_sessions_opened"] == 1
    assert counts["source_sessions_closed"] == 1
    assert counts["source_full_sha_reads"] == 1
    assert counts["source_full_sha_bytes"] == source.stat().st_size
    assert counts["interval_decode_calls"] == len(rows)
    assert counts["interval_decode_frames"] == len(expected)
    assert counts["interval_decode_bytes"] == len(expected) * np.dtype(np.float32).itemsize
    assert counts["output_verify_calls"] == 2 * len(rows)
    assert counts["output_full_read_passes"] == 2 * len(rows)


def test_exporter_lookahead_submits_distinct_source_while_first_source_is_blocked(
    tmp_path, monkeypatch,
):
    source_a = tmp_path / "input" / "a.wav"
    source_b = tmp_path / "input" / "b.wav"
    _write_source(source_a)
    _write_source(source_b)
    rows = [
        _row(source_a, "a-0", 0, 8_000),
        _row(source_a, "a-1", 8_000, 16_000),
        _row(source_b, "b-0", 0, 8_000),
    ]
    source_a_waiting = Event()
    source_b_decoded = Event()

    class FakeSession:
        def __init__(self, row):
            self.source_relative_path = row["source_relative_path"]
            self._row = row

        @property
        def io_counts(self):
            return {
                "source_full_sha_reads": 1,
                "source_full_sha_bytes": self._row["source_size_bytes"],
                "interval_decode_calls": 1,
                "interval_decode_frames": 8_000,
                "interval_decode_bytes": 8_000 * 4,
            }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def decode_interval(self, _start_sample, _end_sample):
            if self.source_relative_path == source_a.name:
                source_a_waiting.set()
                if not source_b_decoded.wait(timeout=2):
                    raise AssertionError(
                        "a blocked the worker pool before b reached a worker"
                    )
            else:
                source_b_decoded.set()
            return np.zeros((8_000, 1), dtype=np.float32), 16_000

        def verify_bindings(self):
            return None

    monkeypatch.setattr(
        scheduler,
        "open_verified_source_session",
        lambda row: FakeSession(row),
    )
    exported = Step6AudioExporter(
        str(tmp_path / "out"), workers=2, max_pending=2,
    )._export_test_rows(rows)

    assert source_a_waiting.is_set()
    assert source_b_decoded.is_set()
    assert [item.segment_id for item in exported] == ["a-0", "a-1", "b-0"]


def test_verified_source_session_pool_evicts_lru_and_closes_every_session(
    monkeypatch,
):
    events: list[tuple[str, str]] = []

    class FakeSession:
        def __init__(self, identity: str) -> None:
            self.identity = identity
            self.closed = False

        @property
        def io_counts(self):
            return {
                "source_full_sha_reads": 1,
                "source_full_sha_bytes": 10,
                "interval_decode_calls": 0,
                "interval_decode_frames": 0,
                "interval_decode_bytes": 0,
            }

        def __enter__(self):
            events.append(("open", self.identity))
            return self

        def __exit__(self, *_args):
            assert not self.closed
            self.closed = True
            events.append(("close", self.identity))

    monkeypatch.setattr(
        scheduler,
        "open_verified_source_session",
        lambda row: FakeSession(str(row["source_key"])),
    )
    counters = ExportIOCounters()
    pool = VerifiedSourceSessionPool(2, counters)

    def identity_row(name: str) -> dict[str, object]:
        return {
            "source_key": name,
            "input_root": "/input",
            "original_path": f"/input/{name}.wav",
            "source_relative_path": f"{name}.wav",
            "source_size_bytes": 10,
            "source_sha256": name * 64,
            "sample_rate": 16_000,
            "frames": 1,
            "channels": 1,
            "format": "WAV",
            "subtype": "FLOAT",
            "duration_sec": 1 / 16_000,
        }

    with pool.acquire(identity_row("a")):
        pass
    with pool.acquire(identity_row("b")):
        pass
    with pool.acquire(identity_row("a")):
        pass
    with pool.acquire(identity_row("c")):
        pass

    assert events == [("open", "a"), ("open", "b"), ("close", "b"), ("open", "c")]
    pool.close()
    assert events == [
        ("open", "a"), ("open", "b"), ("close", "b"), ("open", "c"),
        ("close", "a"), ("close", "c"),
    ]
    assert counters.snapshot()["source_sessions_opened"] == 3
    assert counters.snapshot()["source_sessions_closed"] == 3


def test_run_lock_nested_context_fails_without_releasing_outer_lock(tmp_path):
    output_root = tmp_path / "out"
    lock = RunLock(str(output_root), "alpha")

    with lock:
        with pytest.raises(RunLockError, match="already acquired"):
            with lock:
                pass
        with pytest.raises(RunLockError, match="already held"):
            RunLock(str(output_root), "alpha").acquire()

    with RunLock(str(output_root), "alpha"):
        pass


@pytest.mark.parametrize(
    "payload",
    [
        b'{"duplicate":1,"duplicate":2}\n',
        b'{"value":NaN}\n',
        b'{"value":1e999}\n',
        b'{"incomplete":true}',
    ],
)
def test_strict_jsonl_parent_and_worker_reject_the_same_invalid_frames(payload):
    with pytest.raises(ValueError):
        mos_gpu_job._decode_json_line(payload, "parent row")
    with pytest.raises(ValueError):
        mos_worker._decode_json_line(payload, "worker row")


def test_strict_jsonl_parent_worker_round_trip_is_bidirectionally_compatible():
    value = {"ordinal": 7, "nested": {"finite": 1.25, "text": "中文"}}
    parent_payload = mos_gpu_job._encode_json_line(value, "parent row")
    worker_payload = mos_worker._encode_json_line(value, "worker row")

    assert parent_payload == worker_payload
    assert mos_worker._decode_json_line(parent_payload, "worker row") == value
    assert mos_gpu_job._decode_json_line(worker_payload, "parent row") == value


def test_io_counter_hook_cannot_bypass_resume_content_verification(tmp_path, monkeypatch):
    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    row = _row(source, "segment-00", 0, 500)
    exported = Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])
    destination = Path(exported[0].output_path)
    damaged = bytearray(destination.read_bytes())
    damaged[len(damaged) // 2] ^= 0x01
    destination.write_bytes(damaged)

    monkeypatch.setattr(
        audio_export.ExportIOCounters,
        "record_output_verify",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(ValueError, match="cannot be read|hash|metadata|changed"):
        Step6AudioExporter(str(output_root), workers=1)._export_test_rows([row])
