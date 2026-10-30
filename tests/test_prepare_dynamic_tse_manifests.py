"""dynamic_tse strict speech/nested enrollment producer tests。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools.prepare_data_tools.generators.tse import prepare_dynamic_tse_manifests as producer

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.linspace(-0.2, 0.2, 160, dtype=np.float32), 16000)


def test_producer_publishes_strict_speech_and_nested_enrollment(tmp_path: Path) -> None:
    speech = tmp_path / "source" / "speech.wav"
    enroll_a = tmp_path / "source" / "enroll-a.wav"
    enroll_b = tmp_path / "source" / "enroll-b.wav"
    for path in (speech, enroll_a, enroll_b):
        _wav(path)
    source = tmp_path / "source" / "records.jsonl"
    source.write_text(json.dumps({
        "uid": "utt-1",
        "path": "speech.wav",
        "language": "en",
        "dataset_id": "clean-set",
        "speaker_id": "speaker-1",
        "enrollments": ["enroll-a.wav", "enroll-b.wav"],
    }) + "\n", encoding="utf-8")
    speech_output = tmp_path / "published" / "speech.jsonl"
    enrollment_output = tmp_path / "published" / "spk2enroll.json"

    assert producer.main([
        "--source-jsonl", str(source),
        "--speech-jsonl", str(speech_output),
        "--spk2enroll-json", str(enrollment_output),
    ]) == 0

    record = json.loads(speech_output.read_text(encoding="utf-8"))
    assert record["fs"] == 16000
    assert record["length"] == 160
    assert record["dataset_id"] == "clean-set"
    nested = json.loads(enrollment_output.read_text(encoding="utf-8"))
    assert nested == {
        "clean-set": {"speaker-1": [str(enroll_a.resolve()), str(enroll_b.resolve())]}
    }


def test_producer_permanently_rejects_realt_assets(tmp_path: Path) -> None:
    realt = tmp_path / "data" / "REAL-TSE" / "REAL-T-dev"
    speech = realt / "mixtures" / "mix.wav"
    enroll = realt / "enrolment_speakers" / "enroll.wav"
    _wav(speech)
    _wav(enroll)
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps({
        "uid": "trial",
        "path": str(speech),
        "language": "en",
        "dataset_id": "REAL-T",
        "speaker_id": "speaker-1",
        "enrollments": [str(enroll)],
    }) + "\n", encoding="utf-8")

    with pytest.raises(PermissionError, match="REAL-T evaluation assets"):
        producer.main([
            "--source-jsonl", str(source),
            "--speech-jsonl", str(tmp_path / "speech.jsonl"),
            "--spk2enroll-json", str(tmp_path / "spk2enroll.json"),
        ])

    assert not (tmp_path / "speech.jsonl").exists()
    assert not (tmp_path / "spk2enroll.json").exists()
