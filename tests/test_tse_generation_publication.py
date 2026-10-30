"""REAL-TSE GenerationManifest v2 publication 与合规门禁。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools.prepare_data_tools.generators.tse import generate_realtse_pseudo_targets as generator
from tools.prepare_data_tools.generation_manifest import load_generation_manifest
from tasks.tse.data.offline_dataset import TSEOfflineDataset

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.linspace(-0.2, 0.2, 80, dtype=np.float32), 16000)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "REAL-T-dev"
    rows = []
    for index in range(6):
        mix = f"session-{index:02d}_mixture_0_80"
        enroll = f"enroll-{index:02d}"
        rows.append({
            "mixture_utterance": mix,
            "enrolment_speakers_utterance": enroll,
            "speaker": "speaker-1",
            "source": "source-a",
        })
        _write_wav(root / "mixtures" / f"{mix}.wav")
        _write_wav(root / "enrolment_speakers" / f"{enroll}.wav")
        _write_wav(root / "pseudo_targets" / f"{mix}--{enroll}.wav")
    csv_path = root / "DEV" / "source-a_meta.csv"
    csv_path.parent.mkdir(parents=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root, root / "pseudo_targets", root / "offline_manifest"


def _argv(root: Path, targets: Path, manifest: Path, *extra: str) -> list[str]:
    return [
        "--realt_root", str(root),
        "--output-dir", str(targets),
        "--manifest-dir", str(manifest),
        "--no-teacher",
        "--val-session-mod", "2",
        "--acknowledge-real-t-evaluation-only", generator.REAL_T_ACKNOWLEDGEMENT,
        *extra,
    ]


def test_max_trials_remains_probe_and_never_publishes(tmp_path: Path) -> None:
    root, targets, manifest = _fixture(tmp_path)
    assert generator.main(_argv(root, targets, manifest, "--max-trials", "1")) == 0
    assert not manifest.exists()


def test_tse_offline_publication_v2_full_reloads_and_receipts(tmp_path: Path) -> None:
    root, targets, manifest = _fixture(tmp_path)
    assert generator.main(_argv(root, targets, manifest)) == 0

    envelope = load_generation_manifest(manifest)
    receipt = json.loads(
        (manifest / "generation_gate_receipt.json").read_text(encoding="utf-8")
    )
    assert envelope.task == "tse"
    assert envelope.generator.version == generator.GENERATOR_VERSION
    assert envelope.counts.requested == envelope.counts.published == 6
    assert receipt["reload_report"]["publish_eligible"] is True
    assert {gate["adapter"] for gate in receipt["reload_report"]["gates"]} == {
        "tse-wesep-offline-v1"
    }
    reader_options = {
        "manifest_format": "path_list",
        "path_base": "manifest_relative",
        "target_selection": "all",
        "mix_variant": "default",
        "cue_schema": "realtse_raw",
        "enabled_cues": ["audio"],
        "strict_load": True,
    }
    assert len(TSEOfflineDataset(
        samples_list=manifest / "train" / "raw.list",
        cues_yaml=manifest / "train" / "cues.yaml",
        **reader_options,
    )) == 3
    assert len(TSEOfflineDataset(
        samples_list=manifest / "val" / "raw.list",
        cues_yaml=manifest / "val" / "cues.yaml",
        **reader_options,
    )) == 3


def test_official_realt_defaults_to_fail_closed_before_writes(tmp_path: Path) -> None:
    root, targets, manifest = _fixture(tmp_path)
    argv = _argv(root, targets, manifest)
    ack_index = argv.index("--acknowledge-real-t-evaluation-only")
    del argv[ack_index:ack_index + 2]

    with pytest.raises(PermissionError, match="evaluation-only"):
        generator.main(argv)

    assert not manifest.exists()


@pytest.mark.parametrize("split", ["EVAL1", "EVAL2"])
def test_eval_splits_are_never_publishable(tmp_path: Path, split: str) -> None:
    root = tmp_path / f"REAL-T-{split.lower()}"
    (root / split).mkdir(parents=True)

    with pytest.raises(PermissionError, match="final-report-only"):
        generator.main([
            "--realt_root", str(root),
            "--no-teacher",
            "--acknowledge-real-t-evaluation-only", generator.REAL_T_ACKNOWLEDGEMENT,
        ])


def test_split_is_scoped_by_source_when_session_names_collide() -> None:
    trials = [
        {"uid": f"a-{index}", "source": "a", "session": f"s{index}"}
        for index in range(5)
    ] + [
        {"uid": f"b-{index}", "source": "b", "session": f"s{index}"}
        for index in range(4)
    ]
    train, val = generator._split_train_val(trials, 5)
    assert [trial["uid"] for trial in val] == ["a-4"]
    assert {trial["uid"] for trial in train} == {
        *(f"a-{index}" for index in range(4)),
        *(f"b-{index}" for index in range(4)),
    }
