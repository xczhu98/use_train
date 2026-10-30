"""SE target-only spatial observation contracts for C=2 and C=6 arrays."""

import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from tasks.se.data.dynamic_dataset import DynamicMixingDataset


pytestmark = pytest.mark.unit


def _build_target_only_dataset(tmp_path, num_channels):
    sample_rate = 8000
    num_samples = 128
    waveform = np.sin(np.linspace(0.0, 5.0, num_samples)).astype(np.float32)
    for name in ("speech", "noise", "rir"):
        sf.write(tmp_path / f"{name}.wav", waveform, sample_rate, subtype="FLOAT")
    (tmp_path / "speech.scp").write_text(
        f"speech 8000 {num_samples} speech.wav\n", encoding="utf-8"
    )
    (tmp_path / "noise.scp").write_text(
        f"noise 8000 {num_samples} noise.wav\n", encoding="utf-8"
    )
    (tmp_path / "rir.scp").write_text("rir 8000 128 rir.wav\n", encoding="utf-8")

    spatial_rirs = np.zeros((num_channels, 4), dtype=np.float32)
    spatial_rirs[:, 0] = np.linspace(1.0, 0.25, num_channels, dtype=np.float32)
    spatial_rirs[:, 1] = np.linspace(0.0, 0.15, num_channels, dtype=np.float32)
    tracks = {}
    identities = {}
    for track, value in (
        ("full", spatial_rirs),
        ("direct", spatial_rirs * 0.9),
        ("early", spatial_rirs * 0.8),
    ):
        filename = f"spatial_{track}.npy"
        np.save(tmp_path / filename, value)
        tracks[track] = filename
        identities[track] = hashlib.sha256(
            np.ascontiguousarray(value, dtype="<f4").tobytes()
        ).hexdigest()
    spacing = 0.08 if num_channels == 2 else 0.04
    positions = [
        [spacing * (index - (num_channels - 1) / 2.0), 0.0, 0.0]
        for index in range(num_channels)
    ]
    (tmp_path / "spatial.json").write_text(
        json.dumps(
            {
                "entries": {"speech": tracks},
                "metadata": {
                    "schema_version": 1,
                    "scene_uid": "scene-speech",
                    "actor_order": ["speech"],
                    "sample_rate": sample_rate,
                    "actor_positions_m": {"speech": [1.0, 1.0, 1.5]},
                    "room": {"room_uid": "room-speech"},
                    "geometry": {"positions_m": positions},
                    "generator_version": "fixture-v1",
                    "track_identities": {"speech": identities},
                },
            }
        ),
        encoding="utf-8",
    )

    return DynamicMixingDataset(
        str(tmp_path / "speech.scp"),
        str(tmp_path / "noise.scp"),
        str(tmp_path / "spatial.json"),
        max_duration_seconds=num_samples / sample_rate,
        filter_by_duration=False,
        simulation_cfg={
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm" if num_channels == 2 else "linear_6ch_40mm",
            "reference_channel": 0,
            "prob_reverberation": 0.0,
            "prob_target_only": 1.0,
            "prob_noise_only": 0.0,
            "prob_target_noise": 0.0,
            "prob_channel_enabled": False,
            "peak_normalization_mode": "off",
            "noisy_post_mix_degradations_enabled": False,
        },
    )


@pytest.mark.parametrize("num_channels", (2, 6), ids=("C2", "C6"))
def test_se_target_only_keeps_source_microphone_observation_and_mono_target(
    tmp_path, num_channels
):
    dataset = _build_target_only_dataset(tmp_path, num_channels)

    traced = dataset.get_sample_with_trace(0)
    target, observation = traced.sample[:2]
    scene_facts = traced.trace.metadata_dict()["sources"]["scene"]["scene"]

    assert target.shape == (128,)
    assert observation.shape == (num_channels, 128)
    # offline early 是独立资产，target 必须消费它，不能冒充 full reference。
    np.testing.assert_allclose(observation[0] * .8, target, rtol=1e-5, atol=1e-7)
    assert not np.allclose(observation, np.repeat(target[None, :], num_channels, axis=0))
    assert np.max(np.abs(observation[1:] - target[None, :])) > 1e-4
    assert scene_facts["decomposable_final"] is True
    assert scene_facts["spatial"]["num_channels"] == num_channels
