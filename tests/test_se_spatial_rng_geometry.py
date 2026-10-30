"""SE spatial RNG 与 supplied geometry 合同测试。"""

import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from tasks.se.data.dynamic_dataset import DynamicMixingDataset


def _write_typed_spatial_manifest(tmp_path):
    arrays = {
        "full": np.asarray([[1.0, 0.0], [0.0, 0.5]], dtype=np.float32),
        "direct": np.asarray([[1.0, 0.0], [0.0, 0.25]], dtype=np.float32),
        "early": np.asarray([[1.0, 0.0], [0.0, 0.4]], dtype=np.float32),
    }
    entries = {}
    identities = {}
    for track, value in arrays.items():
        filename = f"spatial_{track}.npy"
        np.save(tmp_path / filename, value)
        entries.setdefault("u", {})[track] = filename
        identities.setdefault("u", {})[track] = hashlib.sha256(
            np.ascontiguousarray(value, dtype="<f4").tobytes()
        ).hexdigest()
    manifest = tmp_path / "spatial.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": entries,
                "metadata": {
                    "schema_version": 1,
                    "scene_uid": "scene-u",
                    "actor_order": ["u"],
                    "sample_rate": 8000,
                    "actor_positions_m": {"u": [1.0, 1.0, 1.5]},
                    "room": {"room_uid": "room-u"},
                    "geometry": {
                        "positions_m": [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]],
                        "spacing_m": 0.08,
                        "coordinate_system": "cartesian_xyz_m",
                    },
                    "generator_version": "fixture-v1",
                    "track_identities": identities,
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


@pytest.mark.unit
def test_se_spatial_uses_supplied_geometry_for_background_without_duplicate_config(tmp_path):
    sample_rate = 8000
    waveform = (0.05 * np.sin(np.linspace(0.0, 3.0, 128))).astype(np.float32)
    for name in ("speech", "noise"):
        sf.write(tmp_path / f"{name}.wav", waveform, sample_rate, subtype="FLOAT")
    np.save(
        tmp_path / "spatial.npy",
        np.asarray([[1.0, 0.0], [0.0, 0.5]], dtype=np.float32),
    )
    (tmp_path / "speech.scp").write_text("u 8000 128 speech.wav\n", encoding="utf-8")
    (tmp_path / "noise.scp").write_text("n 8000 128 noise.wav\n", encoding="utf-8")
    (tmp_path / "rir.scp").write_text("r 8000 2 speech.wav\n", encoding="utf-8")
    (tmp_path / "spatial.json").write_text(
        json.dumps(
            {
                "entries": {"u": "spatial.npy"},
                "metadata": {
                    "geometry": {
                        "positions_m": [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]],
                        "spacing_m": 0.08,
                        "coordinate_system": "cartesian_xyz_m",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    _write_typed_spatial_manifest(tmp_path)
    dataset = DynamicMixingDataset(
        str(tmp_path / "speech.scp"),
        str(tmp_path / "noise.scp"),
        str(tmp_path / "spatial.json"),
        max_duration_seconds=128 / sample_rate,
        filter_by_duration=False,
        simulation_cfg={
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "prob_reverberation": 0.0,
            "prob_target_only": 0.0,
            "prob_noise_only": 0.0,
            "prob_target_noise": 1.0,
            "snr_low": 6.0,
            "snr_high": 6.0,
            "source_volume_normalization_enabled": False,
            "peak_normalization_mode": "off",
            "prob_channel_enabled": False,
            "noisy_post_mix_degradations_enabled": False,
        },
    )

    traced = dataset.get_sample_with_trace(0)
    target, observation = traced.sample[:2]
    assert target.shape == (128,)
    assert observation.shape == (2, 128)
    scene_facts = traced.trace.metadata_dict()["sources"]["scene"]["scene"]
    assert scene_facts["geometry"]["positions_m"] == [
        [-0.04, 0.0, 0.0],
        [0.04, 0.0, 0.0],
    ]
    assert scene_facts["noise_image_provenance"]["source_identity"] == "noise_scp:n"
    assert scene_facts["background_snr_nominal_values"] == [6.0]
    assert scene_facts["background_snr_measure_point"] == "post_render_full_image"
    speech_image = np.stack((target, np.pad(target[:-1], (1, 0)) * .5))
    noise_image = observation - speech_image
    actual_snr = 10 * np.log10(np.sum(speech_image.astype(np.float64) ** 2)
                               / np.sum(noise_image.astype(np.float64) ** 2))
    assert actual_snr == pytest.approx(6.0, abs=1e-4)
    reference_snr = 10 * np.log10(np.sum(speech_image[0].astype(np.float64) ** 2)
                                / np.sum(noise_image[0].astype(np.float64) ** 2))
    assert scene_facts["background_snr_reference_channel_db"] == pytest.approx(reference_snr)
    provenance = scene_facts["noise_image_provenance"]
    assert provenance["snr_policy"] == "full_image_energy"
    assert provenance["snr_reference"] == "all_speech"
    assert provenance["snr_nominal_db"] == pytest.approx(6.0)
    assert provenance["snr_realized_db"] == pytest.approx(6.0, abs=1e-5)
    assert provenance["snr_gain"] > 0.0


@pytest.mark.unit
def test_se_spatial_pipeline_consumes_one_rng_stream(tmp_path, monkeypatch):
    sample_rate = 8000
    waveform = np.sin(np.linspace(0.0, 3.0, 128)).astype(np.float32)
    for name in ("speech", "noise"):
        sf.write(tmp_path / f"{name}.wav", waveform, sample_rate, subtype="FLOAT")
    np.save(
        tmp_path / "spatial.npy",
        np.asarray([[1.0, 0.0], [0.0, 0.5]], dtype=np.float32),
    )
    (tmp_path / "speech.scp").write_text("u 8000 128 speech.wav\n", encoding="utf-8")
    (tmp_path / "noise.scp").write_text("n 8000 128 noise.wav\n", encoding="utf-8")
    (tmp_path / "rir.scp").write_text("r 8000 2 speech.wav\n", encoding="utf-8")
    (tmp_path / "spatial.json").write_text(
        json.dumps({"u": "spatial.npy"}), encoding="utf-8"
    )
    _write_typed_spatial_manifest(tmp_path)
    dataset = DynamicMixingDataset(
        str(tmp_path / "speech.scp"),
        str(tmp_path / "noise.scp"),
        str(tmp_path / "spatial.json"),
        max_duration_seconds=128 / sample_rate,
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
    streams = []

    def get_rng(index=None):
        stream = np.random.default_rng(1234 + int(index or 0))
        streams.append(stream)
        return stream

    monkeypatch.setattr(dataset, "_get_rng", get_rng)
    dataset.get_sample_with_trace(0)
    assert len(streams) == 1
