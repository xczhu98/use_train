"""SE/SS supplied-RIR 数据集 CPU smoke tests。"""

import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from tasks.se.data.dynamic_dataset import DynamicMixingDataset
from tasks.ss.data.collate import ss_collate_fn
from tasks.ss.data.offline_dataset import SSOfflineDataset


def _write_typed_spatial_manifest(tmp_path, full_by_actor):
    entries = {}
    identities = {}
    for actor, full in full_by_actor.items():
        tracks = {}
        actor_identities = {}
        for track, value in (
            ("full", full),
            ("direct", full * 0.9),
            ("early", full * 0.8),
        ):
            filename = f"{actor}_{track}.npy"
            np.save(tmp_path / filename, value)
            tracks[track] = filename
            actor_identities[track] = hashlib.sha256(
                np.ascontiguousarray(value, dtype="<f4").tobytes()
            ).hexdigest()
        entries[actor] = tracks
        identities[actor] = actor_identities
    path = tmp_path / "spatial.json"
    path.write_text(
        json.dumps(
            {
                "entries": entries,
                "metadata": {
                    "schema_version": 1,
                    "scene_uid": "scene-fixture",
                    "actor_order": list(full_by_actor),
                    "sample_rate": 8000,
                    "actor_positions_m": {actor: [float(i + 1), 1.0, 1.5]
                                          for i, actor in enumerate(full_by_actor)},
                    "room": {"room_uid": "room-fixture"},
                    "geometry": {
                        "positions_m": [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]]
                    },
                    "generator_version": "fixture-v1",
                    "track_identities": identities,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.unit
def test_ss_offline_supplied_rir_projects_multichannel_observation(tmp_path):
    sr = 8000
    waveform = np.arange(32, dtype=np.float32) / 32
    for name, value in (("mix", waveform), ("a", waveform), ("b", waveform * 2)):
        sf.write(tmp_path / f"{name}.wav", value, sr, subtype="FLOAT")
    spatial_manifest = _write_typed_spatial_manifest(
        tmp_path,
        {
            "a": np.array([[1, 0], [0, 1]], dtype=np.float32),
            "b": np.array([[1, 0], [0, 2]], dtype=np.float32),
        },
    )
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(
            {
                "uid": "u",
                "fs": sr,
                "length": len(waveform),
                "mixture": "mix.wav",
                "sources": [
                    {"source_id": "a", "path": "a.wav", "length": len(waveform)},
                    {"source_id": "b", "path": "b.wav", "length": len(waveform)},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = SSOfflineDataset(
        tmp_path / "manifest.jsonl",
        num_sources=2,
        rir_scp=spatial_manifest,
        simulation_cfg={
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "prob_reverberation": 0.0,
        },
    )
    sample = dataset[0]
    assert sample["mixture"].shape == (2, len(waveform))
    assert all(source.shape == (len(waveform),) for source in sample["sources"])
    batch = ss_collate_fn([sample])
    assert batch.mixture.shape == (1, 2, len(waveform))
    assert batch.sources.shape == (1, 2, len(waveform))


@pytest.mark.unit
def test_se_dynamic_supplied_rir_keeps_target_mono_and_observation_spatial(tmp_path):
    sr = 8000
    waveform = np.sin(np.linspace(0, 3, 128)).astype(np.float32)
    for name, value in (("speech", waveform), ("noise", waveform), ("rir", np.array([1, 0], np.float32))):
        sf.write(tmp_path / f"{name}.wav", value, sr, subtype="FLOAT")
    (tmp_path / "speech.scp").write_text("u 8000 128 speech.wav\n", encoding="utf-8")
    (tmp_path / "noise.scp").write_text("n 8000 128 noise.wav\n", encoding="utf-8")
    (tmp_path / "rir.scp").write_text("r 8000 2 rir.wav\n", encoding="utf-8")
    spatial_manifest = _write_typed_spatial_manifest(
        tmp_path,
        {"u": np.array([[1, 0], [0, 0.5]], dtype=np.float32)},
    )
    dataset = DynamicMixingDataset(
        str(tmp_path / "speech.scp"),
        str(tmp_path / "noise.scp"),
        str(spatial_manifest),
        max_duration_seconds=128 / sr,
        filter_by_duration=False,
        simulation_cfg={
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "prob_reverberation": 0.0,
            "prob_target_only": 1.0,
            "prob_noise_only": 0.0,
            "prob_target_noise": 0.0,
            "prob_channel_enabled": False,
        },
    )
    clean, noisy, _, _, *_ = dataset[0]
    assert clean.shape == (128,)
    assert noisy.shape == (2, 128)
    assert not np.array_equal(noisy[0], noisy[1])


@pytest.mark.unit
def test_spatial_mode_requires_explicit_supplied_rir():
    with pytest.raises(ValueError, match="array_profile or custom_geometry"):
        DynamicMixingDataset(
            "missing.speech.scp",
            "missing.noise.scp",
            "missing.rir.scp",
            simulation_cfg={"spatial_enabled": True, "reference_channel": 0},
        )


@pytest.mark.unit
def test_ss_offline_direct_constructor_rejects_spatial_typo(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "uid": "u",
                "fs": 8000,
                "length": 8,
                "mixture": "missing.wav",
                "sources": [
                    {"source_id": "s", "path": "missing.wav", "length": 8}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SimulationConfigs unknown keys"):
        SSOfflineDataset(
            manifest,
            num_sources=1,
            simulation_cfg={"spatial_enabld": True},
        )
