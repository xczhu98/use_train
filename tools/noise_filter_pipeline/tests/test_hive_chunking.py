"""
Tests for ChunkingProcessor hive mode (formerly HiveChunkingProcessor / Stage B0).
"""
import os
import tempfile

import numpy as np
import soundfile as sf
import pytest

from noise_filter_pipeline.core.manifest_io import AudioRecord, read_manifest, write_manifest
from noise_filter_pipeline.processors.chunking import ChunkingProcessor

pytestmark = pytest.mark.pipeline


def _make_wav(path, duration=2.0, sr=16000, scale=0.05):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wav = np.random.randn(int(duration * sr)) * scale
    sf.write(path, wav, sr)


def _make_record(wav_path, duration, sr=16000):
    rec = AudioRecord(
        record_id="ds__src__source",
        dataset_id="ds",
        source_id="src",
        chunk_id="source",
        audio={"path": wav_path, "sr": sr, "duration": duration, "channels": 1},
        labels={"original": "noise"},
    )
    rec.mark_stage("A", "passed")
    return rec


def test_hive_chunking_long_audio():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "src.wav")
        _make_wav(wav_path, duration=25.0)

        manifest = os.path.join(tmpdir, "manifest.jsonl")
        write_manifest(iter([_make_record(wav_path, 25.0)]), manifest)

        proc = ChunkingProcessor({
            "mode": "hive",
            "hive_window_sec": 10.0,
            "hive_step_sec": 5.0,
            "hive_energy_threshold": 0.0005,
            "hive_min_tail_sec": 5.0,
        })
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "out"))

        recs = list(read_manifest(out_manifest))
        assert len(recs) >= 3
        for r in recs:
            assert r.is_stage_passed("B1")
            assert r.audio["duration"] <= 10.0


def test_chunking_none_mode_passthrough():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "src.wav")
        _make_wav(wav_path, duration=8.0)

        manifest = os.path.join(tmpdir, "manifest.jsonl")
        write_manifest(iter([_make_record(wav_path, 8.0)]), manifest)

        proc = ChunkingProcessor({"mode": "none"})
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "out"))

        recs = list(read_manifest(out_manifest))
        assert len(recs) == 1
        assert recs[0].audio["path"] == wav_path


def test_hive_chunking_short_low_energy_filtered():
    """Hive 短于 window 的输入也必须应用 MSE 能量阈值。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "quiet.wav")
        _make_wav(wav_path, duration=2.0, scale=0.0001)
        manifest = os.path.join(tmpdir, "manifest.jsonl")
        write_manifest(iter([_make_record(wav_path, 2.0)]), manifest)

        out_manifest = ChunkingProcessor({
            "mode": "hive", "hive_window_sec": 10.0, "hive_energy_threshold": 0.01,
        }).process(manifest, os.path.join(tmpdir, "out"))
        recs = list(read_manifest(out_manifest))
        assert len(recs) == 1
        assert not recs[0].is_stage_passed("B1")


def test_hive_chunking_low_energy_filtered():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "src.wav")
        _make_wav(wav_path, duration=25.0, scale=0.0001)

        manifest = os.path.join(tmpdir, "manifest.jsonl")
        write_manifest(iter([_make_record(wav_path, 25.0)]), manifest)

        proc = ChunkingProcessor({
            "mode": "hive",
            "hive_energy_threshold": 0.01,
        })
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "out"))

        recs = list(read_manifest(out_manifest))
        assert len(recs) == 1
        assert not recs[0].is_stage_passed("B1")
