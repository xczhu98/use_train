"""TSE 原始录音身份从资源生成到 enrollment 排除的合同。"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from test_tse_dynamic_dataset import tse_data, _build_dataset
from tasks.tse.data.dynamic_dataset import EnrollmentAugmentationProviderError
from tasks.tse.manifest_contract import build_recording_ids
from tools.prepare_data_tools.generators.tse import prepare_dynamic_tse_manifests as manifests
from tools.prepare_data_tools.generators.tse import prepare_librimix_dynamic_source as librimix

pytestmark = [pytest.mark.unit, pytest.mark.contract]


@pytest.fixture
def recording_resources(tse_data):
    speech = Path(tse_data["speech_scp"])
    rows = [json.loads(line) for line in speech.read_text().splitlines()]
    nested = json.loads(Path(tse_data["spk2enroll_json"]).read_text())
    for row in rows:
        row["recording_id"] = row["uid"]
    for speaker, paths in nested["fixture"].items():
        for index, path in enumerate(paths):
            # 第一个候选虽路径不同，但来自 mixture 中同一原始录音。
            recording = f"{speaker}_001" if index == 0 else f"{speaker}_independent"
            rows.append({"uid": f"{speaker}-enroll-{index}", "fs": 16000,
                         "length": sf.info(path).frames, "path": path, "language": "zh",
                         "dataset_id": "fixture", "speaker_id": speaker,
                         "recording_id": recording})
    speech.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return tse_data, rows


def test_same_recording_different_paths_are_excluded(recording_resources):
    data, rows = recording_resources
    ds = _build_dataset(data, enroll_policy="fixed")
    with patch.object(ds, "_read_one_speech", wraps=ds._read_one_speech) as read:
        ds._build_enrollment("spk1_001", 16000, np.random.default_rng(1),
                             excluded_paths={rows[0]["path"]})
    assert read.call_args.args[0] == ds.spk2enroll[("fixture", "spk1")][1]


def test_no_independent_recording_fails(recording_resources):
    data, rows = recording_resources
    ds = _build_dataset(data, enroll_policy="fixed")
    with pytest.raises(EnrollmentAugmentationProviderError, match="no independent"):
        ds._build_enrollment("spk2_001", 16000, np.random.default_rng(1),
                             excluded_paths={rows[1]["path"]})


@pytest.mark.parametrize("fault", ["partial", "conflict", "unknown_enrollment"])
def test_recording_metadata_must_cover_dataset_and_enrollment(tmp_path, fault):
    rows = {"a": {"path": "a.wav", "dataset_id": "a", "recording_id": "r1"},
            "b": {"path": "b.wav", "dataset_id": "a", "recording_id": "r2"}}
    enrollment = {("a", "s"): (str(tmp_path / "b.wav"),)}
    if fault == "partial":
        del rows["b"]["recording_id"]
    elif fault == "conflict":
        rows["b"]["path"] = "a.wav"
    else:
        enrollment[("a", "s")] = (str(tmp_path / "unknown.wav"),)
    with pytest.raises(ValueError, match="recording_id"):
        build_recording_ids(rows, enrollment, str(tmp_path / "speech.jsonl"))


def test_recording_identity_is_namespaced_and_optional(tmp_path):
    rows = {"a": {"path": "a.wav", "dataset_id": "a", "recording_id": "same"},
            "b": {"path": "b.wav", "dataset_id": "b", "recording_id": "same"},
            "c": {"path": "c.wav", "dataset_id": "legacy"}}
    ids = build_recording_ids(rows, {}, str(tmp_path / "speech.jsonl"))
    assert ids[str(tmp_path / "a.wav")] != ids[str(tmp_path / "b.wav")]
    assert str(tmp_path / "c.wav") not in ids


def test_librimix_producer_preserves_ids_and_filters_duplicate_recordings(tmp_path):
    names = ["1-10-001_2-20-001", "1-10-001_2-20-002", "1-10-002_2-20-003"]
    for name in names:
        for role in ("s1", "s2"):
            path = tmp_path / "train" / role / f"{name}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, np.ones(160, dtype=np.float32) * .1, 16000)
    records = librimix.build_source_records(tmp_path, "train")
    ids = {row["path"]: row["recording_id"] for row in records}
    for row in records:
        assert row["generator_version"] == 2
        assert all(ids[path] != row["recording_id"] for path in row["enrollments"])
    source, speech, enrollment = (tmp_path / name for name in ("source.jsonl", "speech.jsonl", "enroll.json"))
    librimix.write_source_jsonl(records, source)
    manifests.main(["--source-jsonl", str(source), "--speech-jsonl", str(speech),
                    "--spk2enroll-json", str(enrollment)])
    output = [json.loads(line) for line in speech.read_text().splitlines()]
    assert {row["path"]: row["recording_id"] for row in output} == ids
    assert all(row["generator_version"] == 2 for row in output)
