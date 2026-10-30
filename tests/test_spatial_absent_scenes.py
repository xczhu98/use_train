"""TSE target-absent 与 SS noise-only spatial scene 回归。"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tasks.ss.data.dynamic_dataset import SSDynamicMixingDataset
from tasks.tse.data.collate import tse_collate_fn
from tasks.tse.data.dynamic_dataset import TSEDynamicMixingDataset

pytestmark = pytest.mark.unit


@pytest.fixture
def spatial_scene_assets(tmp_path: Path) -> dict[str, str]:
    sr = 8000
    length = 800
    time = np.arange(length, dtype=np.float32) / sr
    speech_paths = {}
    enroll_paths = {}
    for index, uid in enumerate(("spk_a", "spk_b")):
        speech = (0.05 * np.sin(2 * np.pi * (250 + index * 100) * time)).astype(np.float32)
        speech_path = tmp_path / f"{uid}.wav"
        enroll_path = tmp_path / f"{uid}_enroll.wav"
        sf.write(speech_path, speech, sr, subtype="FLOAT")
        sf.write(enroll_path, speech, sr, subtype="FLOAT")
        speech_paths[uid] = speech_path
        enroll_paths[uid] = enroll_path

    speech_manifest = tmp_path / "speech.jsonl"
    speech_manifest.write_text(
        "".join(
            json.dumps(
                {
                    "uid": uid,
                    "fs": sr,
                    "length": length,
                    "path": str(path),
                    "language": "en",
                    "dataset_id": "fixture",
                    "speaker_id": uid,
                }
            )
            + "\n"
            for uid, path in speech_paths.items()
        ),
        encoding="utf-8",
    )
    speech_scp = tmp_path / "speech.scp"
    speech_scp.write_text(
        "".join(f"{uid} {sr} {length} {path}\n" for uid, path in speech_paths.items()),
        encoding="utf-8",
    )
    noise_path = tmp_path / "noise.wav"
    sf.write(noise_path, np.full(length, 0.01, dtype=np.float32), sr, subtype="FLOAT")
    noise_scp = tmp_path / "noise.scp"
    noise_scp.write_text(f"noise {sr} {length} {noise_path}\n", encoding="utf-8")
    enroll_manifest = tmp_path / "enroll.json"
    enroll_manifest.write_text(
        json.dumps({"fixture": {uid: [str(path)] for uid, path in enroll_paths.items()} }),
        encoding="utf-8",
    )

    rir_paths = {}
    for uid in speech_paths:
        rir_path = tmp_path / f"{uid}.npy"
        np.save(rir_path, np.asarray([[1.0], [0.5]], dtype=np.float32))
        rir_paths[uid] = rir_path
    spatial_manifest = tmp_path / "spatial.json"
    entries, identities = {}, {}
    for uid, path in rir_paths.items():
        wave = np.load(path)
        entries[uid], identities[uid] = {}, {}
        for kind, gain in (("full", 1.0), ("direct", .8), ("early", .9)):
            value = (wave * gain).astype('<f4')
            asset = tmp_path / f"{uid}_{kind}.npy"
            np.save(asset, value)
            entries[uid][kind] = str(asset)
            identities[uid][kind] = hashlib.sha256(value.tobytes()).hexdigest()
    spatial_manifest.write_text(json.dumps({"entries": entries, "metadata": {
        "schema_version": 1, "scene_uid": "absent-scene", "sample_rate": sr,
        "actor_order": list(rir_paths), "room": {"room_uid": "absent-room"},
        "actor_positions_m": {uid: [i + 1., 1., 1.5] for i, uid in enumerate(rir_paths)},
        "geometry": {"positions_m": [[-.04, 0, 0], [.04, 0, 0]], "reference_channel": 0},
        "generator_version": "absent-fixture-v1", "track_identities": identities,
    }}), encoding="utf-8")
    return {
        "speech_jsonl": str(speech_manifest),
        "speech_scp": str(speech_scp),
        "noise_scp": str(noise_scp),
        "enroll": str(enroll_manifest),
        "spatial": str(spatial_manifest),
    }


def _spatial_cfg(**overrides: object) -> dict[str, object]:
    cfg: dict[str, object] = {
        "spatial_enabled": True,
        "array_profile": "linear_2ch_80mm",
        "reference_channel": 0,
        "prob_reverberation": 1.0,
        "prob_channel_enabled": False,
        "peak_normalization_mode": "off",
        "snr_low": 5.0,
        "snr_high": 5.0,
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.parametrize(
    "branch",
    ("prob_interference_only", "prob_noise_only", "prob_interference_noise_only"),
)
def test_tse_spatial_target_absent_has_zero_target_and_observed_interferers(
    spatial_scene_assets: dict[str, str], branch: str
) -> None:
    cfg = _spatial_cfg(
        **{
            **{
                name: float(name == branch)
                for name in (
                    "prob_target_only",
                    "prob_interference_only",
                    "prob_noise_only",
                    "prob_interference_noise_only",
                    "prob_target_noise",
                    "prob_target_interference",
                    "prob_target_interference_noise",
                )
            },
            "num_interferers": {1: 1.0},
            "sir_range": [0.0, 0.0],
            "snr_reference": "primary",
        }
    )
    dataset = TSEDynamicMixingDataset(
        spatial_scene_assets["speech_jsonl"],
        noise_scp=spatial_scene_assets["noise_scp"],
        spk2enroll_json=spatial_scene_assets["enroll"],
        rir_scp=spatial_scene_assets["spatial"],
        simulation_cfg=cfg,
        target_sr=8000,
        max_duration_seconds=0.05,
        enrollment_duration=0.05,
        filter_by_duration=False,
        enroll_policy="fixed",
    )

    traced = dataset.get_sample_with_trace(0, capture_stems=True)
    sample = traced.sample
    target = sample["targets"][0]
    cue = target["cues"].speaker_waveform
    assert sample["mixture"].shape == (2, 400)
    assert sample["geometry"].channels == 2
    assert target["target"].shape == (400,)
    assert target["target_present"] is False
    assert np.count_nonzero(target["target"]) == 0
    assert np.count_nonzero(sample["mixture"]) > 0
    assert cue is not None and cue.present and cue.value is not None
    batch = tse_collate_fn([sample], cue_specs=dataset.cue_specs)
    speaker_waveform = batch.cues.speaker_waveform
    assert speaker_waveform is not None
    assert batch.mixture.shape == (1, 2, 400)
    assert batch.target_present.tolist() == [False]
    assert speaker_waveform.present.tolist() == [True]
    facts = traced.trace.metadata_dict()["scene"]
    assert facts["branch"] == branch
    assert facts["target_present"] is False
    assert bool(facts["source_uids"]) is ("interference" in branch)
    assert facts["latent_target_removed_from_observation"] is True
    assert facts["decomposable_final"] is True
    assert facts["scene_spec"]["source_count"] == int("interference" in branch)


def test_ss_spatial_noise_only_has_zero_targets_and_spatial_background(
    spatial_scene_assets: dict[str, str],
) -> None:
    dataset = SSDynamicMixingDataset(
        spatial_scene_assets["speech_scp"],
        noise_scp=spatial_scene_assets["noise_scp"],
        num_sources=2,
        sample_rate=8000,
        segment_seconds=0.05,
        rir_scp=spatial_scene_assets["spatial"],
        simulation_cfg={
            **_spatial_cfg(
                prob_target_only=0.0,
                prob_target_noise=0.0,
                prob_noise_only=1.0,
            ),
        },
    )

    traced = dataset.get_sample_with_trace(
        0, force_branch="prob_noise_only", capture_stems=True
    )
    sample = traced.sample
    assert sample["mixture"].shape == (2, 400)
    assert all(source.shape == (400,) for source in sample["sources"])
    assert all(np.count_nonzero(source) == 0 for source in sample["sources"])
    assert np.count_nonzero(sample["mixture"]) > 0
    facts = traced.trace.metadata_dict()["scene"]
    assert facts["scene_spec"]["source_count"] == 0
    assert facts["activity"]["active_source_count"] == 0
    assert facts["latent_snr_reference_removed_from_observation"] is True
    assert facts["decomposable_final"] is True
