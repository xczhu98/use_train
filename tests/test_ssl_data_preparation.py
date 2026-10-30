"""SSL 本地清单整理器的输入路径与元数据门禁。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import soundfile as sf
import numpy as np

from core.data.noise_manifest_adapter import read_noise_source_manifest
from tools.prepare_data_tools.generators.ssl.prepare_manifests import (
    AudioRecord,
    _classify_noise_manifest,
    prepare_manifests,
    read_noise_scp,
    read_speech_jsonl,
    scan_librispeech_split,
)


pytestmark = [pytest.mark.unit]


def _write_audio(path: Path, *, sr: int = 16000, frames: int = 640) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(frames, dtype=np.float32), sr, format="FLAC")
    return path


def test_scan_librispeech_split_and_publish_relative_manifests(tmp_path: Path) -> None:
    corpus = tmp_path / "LibriSpeech"
    train_a = _write_audio(corpus / "train-clean-100" / "10" / "20" / "10-20-0001.flac")
    train_b = _write_audio(corpus / "train-clean-100" / "11" / "21" / "11-21-0001.flac")
    validation = _write_audio(corpus / "dev-clean" / "12" / "22" / "12-22-0001.flac")
    test = _write_audio(corpus / "test-clean" / "13" / "23" / "13-23-0001.flac")
    train = scan_librispeech_split(corpus, "train-clean-100")
    val = scan_librispeech_split(corpus, "dev-clean")
    test_records = scan_librispeech_split(corpus, "test-clean")
    assert len(train) == 2
    assert len(val) == 1
    assert len(test_records) == 1
    assert train[0].path in {train_a.resolve(), train_b.resolve()}
    noise_train = [_record("n1", _write_audio(tmp_path / "noise" / "n1.flac"))]
    noise_val = [_record("n2", _write_audio(tmp_path / "noise" / "n2.flac"))]

    out = tmp_path / "data" / "ssl"
    prepare_manifests(
        output_dir=out,
        speech_train=train,
        speech_validation=val,
        noise_train=noise_train,
        noise_validation=noise_val,
        speech_test=test_records,
        source_info={"speech": {"type": "test"}},
    )
    parsed = read_noise_source_manifest(out / "speech" / "train.scp")
    assert len(parsed.flat) == 2
    assert Path(next(iter(parsed.flat.values()))).is_file()
    assert (out / "speech" / "test.scp").is_file()
    assert json.loads((out / "provenance.json").read_text())["audio_copied"] is False


def _record(uid: str, path: Path):
    info = sf.info(path)
    return AudioRecord(uid, int(info.samplerate), int(info.frames), path.resolve())


def test_noise_scp_rejects_duplicate_uid_and_audio_mismatch(tmp_path: Path) -> None:
    audio = _write_audio(tmp_path / "n.flac")
    scp = tmp_path / "noise.scp"
    scp.write_text(
        f"same 16000 640 {audio}\nsame 16000 640 {audio}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="重复 uid"):
        read_noise_scp(scp, limit=1)

    scp.write_text(f"n 16000 641 {audio}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="帧数不匹配"):
        read_noise_scp(scp, limit=1)


def test_libri2mix_jsonl_rejects_duplicate_uid_and_path(tmp_path: Path) -> None:
    audio = _write_audio(tmp_path / "speech.flac")
    record = {"uid": "s1:one", "path": str(audio), "fs": 16000, "length": 640}
    manifest = tmp_path / "speech.jsonl"
    manifest.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="重复 uid"):
        read_speech_jsonl(manifest, expected_sample_rate=16000)


def test_missing_noise_audio_fails_closed(tmp_path: Path) -> None:
    scp = tmp_path / "noise.scp"
    scp.write_text("missing 16000 640 absent.flac\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="音频路径不存在"):
        read_noise_scp(scp, limit=1)


def test_noise_manifest_provenance_classifies_noise92_subset() -> None:
    noise92_type, noise92_name = _classify_noise_manifest(
        Path("data/ssl/noise92/noise92.scp")
    )
    hq_type, hq_name = _classify_noise_manifest(
        Path("data/shared/noise/filter/v2/train_noise_sources.scp")
    )
    assert noise92_type == "noise92_scp"
    assert "Noise92 local subset" in noise92_name
    assert hq_type == "local_hq_noise_scp"
    assert hq_name == "HQ noise source pool; not Noise92"
