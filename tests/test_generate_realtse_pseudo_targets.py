"""REAL-TSE pseudo-target manifest producer contracts."""

from pathlib import Path

import pytest

from tools.prepare_data_tools.generators.tse.generate_realtse_pseudo_targets import (
    _write_wesep_manifest,
)

pytestmark = pytest.mark.unit


def test_write_wesep_manifest_creates_split_directory_on_first_run(tmp_path: Path):
    target_dir = tmp_path / "pseudo_targets"
    target_dir.mkdir()
    target_path = target_dir / "mix--enroll.wav"
    target_path.touch()
    mixture = tmp_path / "mixture.wav"
    enrollment = tmp_path / "enrollment.wav"
    trials = [
        {
            "uid": "mix--enroll",
            "speaker": "speaker-1",
            "mix_path": mixture,
            "enr_path": enrollment,
        }
    ]
    split_dir = tmp_path / "offline_manifest" / "train"

    _write_wesep_manifest(trials, target_dir, split_dir, policy="random")

    assert (split_dir / "raw.list").is_file()
    assert (split_dir / "samples.jsonl").is_file()
    assert (split_dir / "cues.yaml").is_file()
    assert (split_dir / "cues" / "audio.json").is_file()
