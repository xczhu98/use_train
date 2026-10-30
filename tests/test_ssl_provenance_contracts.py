"""SSL 数据投影的 provenance、generator identity 与 profile 边界。"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from core.data.acoustic_scene_plan import (
    AcousticActorPlan,
    AcousticScenePlan,
    NoisePlan,
    RIRPlan,
    SourceGainPlan,
)
from core.data.simulation.array_profiles import resolve_microphone_array
from core.data.simulation.rng import SimulationRNG
from core.data.simulation.scene_context import (
    DatasetAcousticRuntimeAdapter,
    SceneRenderContext,
)
from core.data.simulation.scene_manifest import build_track_identity
from core.data.simulation.rir_provider import MovingSpatialRIRTrackSet
from core.data.simulation.trajectory import SourceTrajectory, TrajectoryTimebase
from tasks.ssl.data import SSLProjectionConfig, SSLSceneInput, prepare_ssl_scene, project_ssl_scene
from tasks.ssl.data.projection import _stft
from tasks.ssl.data.dynamic_dataset import SSLDynamicMixingDataset
from tasks.ssl.data import spatial_adapter


pytestmark = pytest.mark.unit


def _scene(*, target_representation: str = "dpipd_far_field") -> SSLSceneInput:
    length = 64
    source = torch.randn(length, generator=torch.Generator().manual_seed(7))
    direct = torch.stack((source, torch.roll(source, 1))).unsqueeze(0)
    mic_positions = torch.tensor([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]])
    source_positions = torch.tensor([[1.0, 0.0, 1.0]])
    return SSLSceneInput(
        observation=direct[0],
        direct_tracks=direct,
        mic_positions=mic_positions,
        source_positions=source_positions,
        sample_rate=16000,
        provenance={"backend": "fixture"},
        coordinate_frame="array_local",
    )


def _rir_provenance(direct: torch.Tensor, *, generator_version: str = "rir-fixture-v7"):
    digest = hashlib.sha256(
        np.ascontiguousarray(direct.detach().cpu().numpy(), dtype="<f4").tobytes()
    ).hexdigest()
    return {
        "source_order": [f"source_{index}" for index in range(int(direct.shape[0]))],
        "rir": {
            "generator_version": generator_version,
            "rir_sha256": digest,
        }
    }


def _offline_fixture_dataset(
    root: Path,
    *,
    layer_e: bool = False,
    layer_f: bool = False,
    seed: int = 0,
    normalization: str = "none",
    sound_speed: float | None = None,
    simulation_profile: str | None = None,
) -> SSLDynamicMixingDataset:
    """构造可重复的 typed offline fixture，覆盖完整 Dataset provenance 链。"""
    root.mkdir(parents=True, exist_ok=True)
    sample_rate = 8000
    length = 200
    time = np.arange(length, dtype=np.float32) / sample_rate
    speech_paths = []
    for index, frequency in enumerate((240.0, 420.0)):
        path = root / f"speech_{index}.wav"
        sf.write(
            path,
            np.sin(2 * np.pi * frequency * time).astype(np.float32),
            sample_rate,
            subtype="FLOAT",
        )
        speech_paths.append(path)
    speech_scp = root / "speech.scp"
    speech_scp.write_text(
        "\n".join(
            f"source_{index} {sample_rate} {length} {path}"
            for index, path in enumerate(speech_paths)
        )
        + "\n",
        encoding="utf-8",
    )

    rir_paths = []
    direct_paths = []
    for index, delay in enumerate((0, 1)):
        rir = np.zeros((2, 8), dtype=np.float32)
        rir[0, delay] = 1.0
        rir[1, delay + 1] = 0.8
        rir_path = root / f"rir_{index}.npy"
        np.save(rir_path, rir)
        rir_paths.append(rir_path)
        direct = rir.copy()
        direct[0, delay] *= 0.9
        direct_path = root / f"direct_{index}.npy"
        np.save(direct_path, direct)
        direct_paths.append(direct_path)

    rir_manifest = root / "spatial_rir.json"
    rir_manifest.write_text(
        json.dumps(
            {
                "entries": {
                    f"source_{index}": {
                        "full": str(rir_paths[index]),
                        "direct": str(direct_paths[index]),
                        "early": str(direct_paths[index]),
                    }
                    for index in range(2)
                },
                "metadata": {
                    "schema_version": 1,
                    "sample_rate": sample_rate,
                    "scene_uid": "ssl-provenance-fixture",
                    "actor_order": ["source_0", "source_1"],
                    "room": {"room_uid": "ssl-room", "room_dim_m": [6, 6, 3]},
                    "actor_positions_m": {
                        "source_0": [1, 1, 1.5],
                        "source_1": [2, 2, 1.5],
                    },
                    "generator_version": "ssl-fixture-v1",
                    "track_identities": {
                        f"source_{index}": {
                            "full": build_track_identity(rir_paths[index]),
                            "direct": build_track_identity(direct_paths[index]),
                            "early": build_track_identity(direct_paths[index]),
                        }
                        for index in range(2)
                    },
                    "geometry": {
                        "positions_m": [[2.0, 2.0, 1.0], [2.08, 2.0, 1.0]],
                        "reference_channel": 0,
                        "array_profile": "linear_2ch_80mm",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return SSLDynamicMixingDataset(
        str(speech_scp),
        tracks=2,
        sample_rate=sample_rate,
        segment_seconds=length / sample_rate,
        rir_scp=str(rir_manifest),
        seed=seed,
        simulation_cfg={
            "spatial_enabled": True,
            "reference_channel": 0,
            "prob_reverberation": 1.0,
            "prob_channel_enabled": layer_e,
            "num_augmentations": {0: 1.0},
            "noisy_post_mix_degradations_enabled": layer_f,
            "peak_normalization_mode": "off",
        },
        custom_geometry={
            "positions_m": [[2.0, 2.0, 1.0], [2.08, 2.0, 1.0]]
        },
        n_fft=16,
        hop_length=8,
        win_length=16,
        window="hamming",
        center=False,
        input_representation="multichannel_stft_real_imag",
        normalization=normalization,
        sound_speed=sound_speed,
        simulation_profile=simulation_profile,
    )


def _identity_snapshot(item: dict) -> dict:
    provenance = item["provenance"]
    return {
        key: provenance.get(key)
        for key in (
            "scene_uid",
            "room",
            "geometry",
            "geometry_identity",
            "track_identity",
            "track_facts",
            "rir",
            "source_order",
            "generator_version",
            "rir_sha256",
            "coordinate_frame",
            "stft",
            "sound_speed_mps",
            "normalization",
            "simulation_profile",
        )
    }


def test_prepare_preserves_nested_rir_generator_identity_and_adds_canonical_provenance():
    scene = _scene()
    direct = torch.as_tensor(scene.direct_tracks)
    provenance = _rir_provenance(direct)
    item = prepare_ssl_scene(
        scene.observation.numpy(),
        direct.numpy(),
        sample_rate=scene.sample_rate,
        mic_positions=scene.mic_positions.numpy(),
        source_positions=scene.source_positions.numpy(),
        reference_channel=0,
        n_fft=16,
        hop_length=8,
        win_length=16,
        window="hamming",
        center=False,
        expected_channels=2,
        expected_tracks=1,
        expected_freqs=9,
        expected_input_size=4,
        backend="offline",
        provenance=provenance,
        coordinate_frame=scene.coordinate_frame,
    )
    assert item["provenance"]["generator_version"] == "rir-fixture-v7"
    assert item["provenance"]["generator_version"] != "scene-projection-v2"
    assert item["provenance"]["rir_sha256"] == provenance["rir"]["rir_sha256"]
    assert item["provenance"]["coordinate_frame"] == "array_local"
    assert item["provenance"]["sample_rate"] == 16000
    assert item["provenance"]["geometry"]["positions_m"] == scene.mic_positions.tolist()


def test_prepare_reads_per_track_moving_rir_identity_facts():
    scene = _scene()
    item = prepare_ssl_scene(
        scene.observation.numpy(),
        scene.direct_tracks.numpy(),
        sample_rate=scene.sample_rate,
        mic_positions=scene.mic_positions.numpy(),
        source_positions=scene.source_positions.numpy(),
        reference_channel=0,
        n_fft=16,
        hop_length=8,
        win_length=16,
        window="hamming",
        center=False,
        expected_channels=2,
        expected_tracks=1,
        expected_freqs=9,
        expected_input_size=4,
        backend="offline",
        provenance={
            "rir": {
                "track_facts": {
                    "full": {"generator_version": "moving-rir-v3"},
                    "direct": {"generator_version": "moving-rir-v3"},
                },
                "track_set": {"full": "a" * 64},
            },
            "source_order": ["source_0"],
        },
        coordinate_frame=scene.coordinate_frame,
    )
    assert item["provenance"]["generator_version"] == "moving-rir-v3"
    assert item["provenance"]["rir_sha256"] == "a" * 64


def test_prepare_reads_nested_track_identity_full_hash():
    scene = _scene()
    item = prepare_ssl_scene(
        scene.observation.numpy(),
        scene.direct_tracks.numpy(),
        sample_rate=scene.sample_rate,
        mic_positions=scene.mic_positions.numpy(),
        source_positions=scene.source_positions.numpy(),
        reference_channel=0,
        n_fft=16,
        hop_length=8,
        win_length=16,
        window="hamming",
        center=False,
        expected_channels=2,
        expected_tracks=1,
        expected_freqs=9,
        expected_input_size=4,
        backend="offline",
        provenance={
            "generator_version": "nested-rir-v4",
            "source_order": ["source_0"],
            "track_identity": {
                "track_set": {
                    "full": {
                        "track_kind": "full",
                        "content_hash": "a" * 64,
                    }
                }
            },
        },
        coordinate_frame=scene.coordinate_frame,
    )
    assert item["provenance"]["rir_sha256"] == "a" * 64


@pytest.mark.parametrize(
    "provenance, match",
    [
        ({"rir": {"rir_sha256": "0" * 64}}, "generator_version"),
        ({"rir": {"generator_version": "rir-fixture-v7"}}, "RIR identity hash"),
        (
            {
                "generator_version": "projection-v1",
                "rir": {
                    "generator_version": "rir-fixture-v7",
                    "rir_sha256": "0" * 64,
                },
            },
            "generator_version",
        ),
    ],
)
def test_prepare_rejects_incomplete_or_conflicting_rir_identity(provenance, match):
    scene = _scene()
    provenance = {"source_order": ["source_0"], **provenance}
    with pytest.raises(ValueError, match=match):
        prepare_ssl_scene(
            scene.observation.numpy(),
            scene.direct_tracks.numpy(),
            sample_rate=scene.sample_rate,
            mic_positions=scene.mic_positions.numpy(),
            source_positions=scene.source_positions.numpy(),
            reference_channel=0,
            n_fft=16,
            hop_length=8,
            win_length=16,
            window="hamming",
            center=False,
            expected_channels=2,
            expected_tracks=1,
            expected_freqs=9,
            expected_input_size=4,
            backend="offline",
            provenance=provenance,
            coordinate_frame=scene.coordinate_frame,
        )


def test_prepare_rejects_conflicting_backend_provenance():
    scene = _scene()
    provenance = {**_rir_provenance(scene.direct_tracks), "backend": "online"}
    with pytest.raises(ValueError, match="backend"):
        prepare_ssl_scene(
            scene.observation.numpy(),
            scene.direct_tracks.numpy(),
            sample_rate=scene.sample_rate,
            mic_positions=scene.mic_positions.numpy(),
            source_positions=scene.source_positions.numpy(),
            reference_channel=0,
            n_fft=16,
            hop_length=8,
            win_length=16,
            window="hamming",
            center=False,
            expected_channels=2,
            expected_tracks=1,
            expected_freqs=9,
            expected_input_size=4,
            backend="offline",
            provenance=provenance,
            coordinate_frame=scene.coordinate_frame,
        )


@pytest.mark.parametrize(
    "source_order",
    [
        [],
        ["source_0", "source_0"],
        [""],
        ["source_0", 1],
    ],
)
def test_prepare_rejects_malformed_source_order(source_order):
    scene = _scene()
    provenance = {**_rir_provenance(scene.direct_tracks), "source_order": source_order}
    with pytest.raises(ValueError, match="source_order"):
        prepare_ssl_scene(
            scene.observation.numpy(),
            scene.direct_tracks.numpy(),
            sample_rate=scene.sample_rate,
            mic_positions=scene.mic_positions.numpy(),
            source_positions=scene.source_positions.numpy(),
            reference_channel=0,
            n_fft=16,
            hop_length=8,
            win_length=16,
            window="hamming",
            center=False,
            expected_channels=2,
            expected_tracks=1,
            expected_freqs=9,
            expected_input_size=4,
            backend="offline",
            provenance=provenance,
            coordinate_frame=scene.coordinate_frame,
        )


def test_near_field_distance_and_coordinate_frame_are_mirrored_in_provenance():
    item = project_ssl_scene(
        _scene(target_representation="dpipd_near_field"),
        SSLProjectionConfig(
            n_fft=16,
            hop_length=8,
            win_length=16,
            target_representation="dpipd_near_field",
        ),
    )
    assert item["coordinate_frame"] == "array_local"
    assert item["provenance"]["coordinate_frame"] == "array_local"
    assert item["provenance"]["source_distances"] == item["source_distances"].tolist()
    torch.testing.assert_close(
        item["source_distances"],
        torch.full_like(item["source_distances"], 2.0**0.5),
    )


def test_forgetting_laplace_uses_official_global_mean_recurrence():
    generator = torch.Generator().manual_seed(23)
    waveform = torch.randn(2, 32, generator=generator)
    raw_config = SSLProjectionConfig(
        n_fft=8, hop_length=4, win_length=8, normalization="none"
    )
    normalized_config = SSLProjectionConfig(
        n_fft=8,
        hop_length=4,
        win_length=8,
        normalization="forgetting_laplace",
        forgetting_norm_sample_length=3,
    )
    raw = _stft(waveform, raw_config)
    normalized = _stft(waveform, normalized_config)
    flat = raw.abs().reshape(1, -1, raw.shape[-1])
    alpha = (3 - 1) / (3 + 1)
    running = torch.zeros(1, 1)
    factors = []
    for frame in range(raw.shape[-1]):
        local = min((frame - 1) / (frame + 1), alpha) if frame < 3 else alpha
        running = local * running + (1.0 - local) * flat[:, :, frame].mean(
            dim=1, keepdim=True
        )
        factors.append(running)
    expected = raw / torch.cat(factors, dim=-1).reshape(1, 1, -1).clamp_min(
        torch.finfo(raw.real.dtype).eps
    )
    torch.testing.assert_close(normalized, expected)


def test_global_mean_normalization_uses_one_cft_factor_and_keeps_target_unscaled():
    generator = torch.Generator().manual_seed(29)
    waveform = torch.randn(2, 64, generator=generator)
    raw_config = SSLProjectionConfig(
        n_fft=16, hop_length=8, win_length=16, center=False, normalization="none"
    )
    global_config = SSLProjectionConfig(
        n_fft=16,
        hop_length=8,
        win_length=16,
        center=False,
        normalization="global_mean",
    )
    raw = _stft(waveform, raw_config)
    normalized = _stft(waveform, global_config)
    expected = raw / raw.abs().mean().clamp_min(torch.finfo(raw.real.dtype).eps)
    torch.testing.assert_close(normalized, expected)

    scene = _scene()
    raw_item = project_ssl_scene(scene, raw_config)
    global_item = project_ssl_scene(scene, global_config)
    torch.testing.assert_close(global_item["target_dpipd"], raw_item["target_dpipd"])
    assert not torch.allclose(global_item["features"], raw_item["features"])
    assert global_item["provenance"]["normalization"] == {
        "mode": "global_mean",
        "forgetting_norm_sample_length": None,
        "aggregation": "global_c_f_t_mean",
    }


def test_projection_provenance_records_stft_sound_speed_and_normalization_contract():
    item = project_ssl_scene(
        _scene(),
        SSLProjectionConfig(
            n_fft=16,
            hop_length=8,
            win_length=16,
            window="hamming",
            center=False,
            exclude_dc=True,
            normalization="global_mean",
            sound_speed=340.0,
        ),
    )
    assert item["provenance"]["stft"] == {
        "n_fft": 16,
        "hop_length": 8,
        "win_length": 16,
        "window": "hamming",
        "center": False,
        "exclude_dc": True,
    }
    assert item["provenance"]["sound_speed_mps"] == 340.0
    assert item["provenance"]["normalization"]["mode"] == "global_mean"
    assert item["provenance"]["normalization"]["aggregation"] == "global_c_f_t_mean"


def test_projection_records_explicit_simulation_profile_without_inference():
    item = project_ssl_scene(
        _scene(),
        SSLProjectionConfig(
            n_fft=16,
            hop_length=8,
            win_length=16,
            simulation_profile="ipdnet_official",
        ),
    )
    assert item["provenance"]["simulation_profile"] == "ipdnet_official"

    scene = _scene()
    scene = SSLSceneInput(
        observation=scene.observation,
        direct_tracks=scene.direct_tracks,
        mic_positions=scene.mic_positions,
        source_positions=scene.source_positions,
        sample_rate=scene.sample_rate,
        provenance={
            "backend": "fixture",
            "simulation_profile": "fnssl_official",
        },
        coordinate_frame=scene.coordinate_frame,
    )
    with pytest.raises(ValueError, match="simulation_profile"):
        project_ssl_scene(
            scene,
            SSLProjectionConfig(
                n_fft=16,
                hop_length=8,
                win_length=16,
                simulation_profile="ipdnet_official",
            ),
        )


def test_projection_rejects_stft_provenance_mismatch():
    scene = _scene()
    scene = SSLSceneInput(
        observation=scene.observation,
        direct_tracks=scene.direct_tracks,
        mic_positions=scene.mic_positions,
        source_positions=scene.source_positions,
        sample_rate=scene.sample_rate,
        provenance={"backend": "fixture", "stft": {"n_fft": 32}},
        coordinate_frame=scene.coordinate_frame,
    )
    with pytest.raises(ValueError, match="provenance.stft.n_fft"):
        project_ssl_scene(
            scene,
            SSLProjectionConfig(n_fft=16, hop_length=8, win_length=16),
        )


def test_projection_rejects_sound_speed_provenance_mismatch():
    scene = _scene()
    scene = SSLSceneInput(
        observation=scene.observation,
        direct_tracks=scene.direct_tracks,
        mic_positions=scene.mic_positions,
        source_positions=scene.source_positions,
        sample_rate=scene.sample_rate,
        provenance={"backend": "fixture", "sound_speed_mps": 340.0},
        coordinate_frame=scene.coordinate_frame,
    )
    with pytest.raises(ValueError, match="sound_speed"):
        project_ssl_scene(
            scene,
            SSLProjectionConfig(
                n_fft=16,
                hop_length=8,
                win_length=16,
                sound_speed=343.0,
            ),
        )


def test_prepare_rejects_conflicting_rir_hash_declarations():
    scene = _scene()
    provenance = {
        "source_order": ["source_0"],
        "rir": {
            "generator_version": "rir-fixture-v7",
            "rir_sha256": "a" * 64,
        },
        "track_facts": {"rir_sha256": "b" * 64},
    }
    with pytest.raises(ValueError, match="hash declarations disagree"):
        prepare_ssl_scene(
            scene.observation.numpy(),
            scene.direct_tracks.numpy(),
            sample_rate=scene.sample_rate,
            mic_positions=scene.mic_positions.numpy(),
            source_positions=scene.source_positions.numpy(),
            reference_channel=0,
            n_fft=16,
            hop_length=8,
            win_length=16,
            window="hamming",
            center=False,
            expected_channels=2,
            expected_tracks=1,
            expected_freqs=9,
            expected_input_size=4,
            backend="offline",
            provenance=provenance,
            coordinate_frame=scene.coordinate_frame,
        )


def test_prepare_rejects_malformed_rir_hash_declaration():
    scene = _scene()
    provenance = {
        "source_order": ["source_0"],
        "rir": {"generator_version": "rir-fixture-v7", "rir_sha256": None},
    }
    with pytest.raises(ValueError, match="strict lowercase"):
        prepare_ssl_scene(
            scene.observation.numpy(),
            scene.direct_tracks.numpy(),
            sample_rate=scene.sample_rate,
            mic_positions=scene.mic_positions.numpy(),
            source_positions=scene.source_positions.numpy(),
            reference_channel=0,
            n_fft=16,
            hop_length=8,
            win_length=16,
            window="hamming",
            center=False,
            expected_channels=2,
            expected_tracks=1,
            expected_freqs=9,
            expected_input_size=4,
            backend="offline",
            provenance=provenance,
            coordinate_frame=scene.coordinate_frame,
        )


def test_dataset_forwards_global_mean_and_sound_speed_to_provenance(tmp_path):
    dataset = _offline_fixture_dataset(
        tmp_path,
        seed=31,
        normalization="global_mean",
        sound_speed=340.0,
    )
    item = dataset[0]
    assert item["provenance"]["normalization"]["mode"] == "global_mean"
    assert item["provenance"]["normalization"]["aggregation"] == "global_c_f_t_mean"
    assert item["provenance"]["sound_speed_mps"] == 340.0


def test_dataset_forwards_explicit_simulation_profile(tmp_path):
    dataset = _offline_fixture_dataset(
        tmp_path,
        seed=37,
        simulation_profile="ipdnet_official",
    )
    item = dataset[0]
    assert item["provenance"]["simulation_profile"] == "ipdnet_official"


def test_forgetting_norm_sample_length_is_explicit_in_ipdnet2_dataset_config():
    from core.utils.config import load_merged_config
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    root = Path(__file__).resolve().parents[1]
    cfg = load_merged_config(root / "configs/ssl/ipdnet2_dynamic.yaml")
    section = SSLTrainingEntry._dataset_config(cfg, "train")
    assert section["normalization"] == "forgetting_laplace"
    assert section["forgetting_norm_sample_length"] == 249
    malformed = deepcopy(cfg)
    malformed["ssl"]["forgetting_norm_sample_length"] = 0
    with pytest.raises(ValueError, match="forgetting_norm_sample_length"):
        SSLTrainingEntry._dataset_config(malformed, "train")


def test_projection_rejects_conflicting_coordinate_frame_provenance():
    scene = _scene()
    scene = SSLSceneInput(
        observation=scene.observation,
        direct_tracks=scene.direct_tracks,
        mic_positions=scene.mic_positions,
        source_positions=scene.source_positions,
        sample_rate=scene.sample_rate,
        provenance={"backend": "fixture", "coordinate_frame": "room_xyz_m"},
        coordinate_frame="array_local",
    )
    with pytest.raises(ValueError, match="coordinate_frame"):
        project_ssl_scene(scene, SSLProjectionConfig(n_fft=16, hop_length=8, win_length=16))


def test_projection_rejects_geometry_provenance_without_positions():
    scene = _scene()
    scene = SSLSceneInput(
        observation=scene.observation,
        direct_tracks=scene.direct_tracks,
        mic_positions=scene.mic_positions,
        source_positions=scene.source_positions,
        sample_rate=scene.sample_rate,
        provenance={"backend": "fixture", "geometry": {"reference_channel": 0}},
        coordinate_frame=scene.coordinate_frame,
    )
    with pytest.raises(ValueError, match="positions_m"):
        project_ssl_scene(scene, SSLProjectionConfig(n_fft=16, hop_length=8, win_length=16))


def test_same_seed_keeps_room_array_and_rir_identity_stable(tmp_path):
    first_dataset = _offline_fixture_dataset(tmp_path, seed=17)
    second_dataset = _offline_fixture_dataset(tmp_path, seed=17)
    first = first_dataset[0]
    second = second_dataset[0]

    for field in ("features", "target_dpipd", "doa", "vad", "geometry", "source_positions"):
        torch.testing.assert_close(first[field], second[field])
    assert _identity_snapshot(first) == _identity_snapshot(second)


def test_layer_ef_switches_do_not_modify_ssl_targets_or_geometry(tmp_path):
    disabled = _offline_fixture_dataset(tmp_path / "disabled", seed=19)
    enabled = _offline_fixture_dataset(
        tmp_path / "enabled", layer_e=True, layer_f=True, seed=19
    )
    disabled_item = disabled[0]
    enabled_item = enabled[0]

    for field in ("target_dpipd", "doa", "vad", "geometry", "source_positions"):
        torch.testing.assert_close(disabled_item[field], enabled_item[field])
    assert _identity_snapshot(disabled_item) == _identity_snapshot(enabled_item)
    assert disabled_item["provenance"]["layer_e"]["channel_augmentation_enabled"] is False
    assert disabled_item["provenance"]["layer_f"]["noisy_post_mix_degradations_enabled"] is False
    assert enabled_item["provenance"]["layer_e"]["channel_augmentation_enabled"] is True
    assert enabled_item["provenance"]["layer_f"]["noisy_post_mix_degradations_enabled"] is True


def test_moving_timebase_alignment_is_repeatable_for_declared_frame_samples():
    dataset = object.__new__(SSLDynamicMixingDataset)
    dataset.tracks, dataset.sample_rate = 2, 8000
    dataset.hop_length, dataset.n_fft, dataset.center = 8, 16, False
    positions = np.asarray(
        [
            [[1.0, 1.0, 1.5], [2.0, 2.0, 1.5]],
            [[1.2, 1.0, 1.5], [2.2, 2.0, 1.5]],
        ],
        dtype=np.float32,
    )
    provenance = {
        "rir": {
            "trajectory": positions.tolist(),
            "trajectory_timebase": {
                "sample_rate": 8000,
                "num_samples": 32,
                "hop_samples": 16,
                "frame_count": 2,
                "start_sample": 0,
                "frame_samples": [0, 16],
            },
        }
    }
    first = dataset._align_moving_source_positions(
        positions, provenance, waveform_length=32
    )
    second = dataset._align_moving_source_positions(
        positions, provenance, waveform_length=32
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(
        first[:, 0, 0],
        np.asarray([1.1, 1.2, 1.2], dtype=np.float32),
        atol=1e-6,
    )


def test_moving_scene_renderer_keeps_same_seed_trajectory_and_rir_identity(monkeypatch):
    array = resolve_microphone_array(
        custom_geometry={"positions_m": [[2.0, 2.0, 1.0], [2.08, 2.0, 1.0]]},
        reference_channel=0,
    )
    timebase = TrajectoryTimebase(8000, 32, 16, 2)
    actors = []
    for index, uid in enumerate(("source_0", "source_1")):
        start = (3.0, 2.0 + index, 1.0)
        trajectory = SourceTrajectory(
            np.asarray(
                [[start], [(3.2, 2.0 + index, 1.0)]], dtype=np.float32
            ),
            timebase,
        )
        actors.append(AcousticActorPlan(uid, "target", index, start, trajectory))
    plan = AcousticScenePlan(
        8000,
        32,
        False,
        RIRPlan.disabled(),
        SourceGainPlan.disabled(),
        NoisePlan.disabled(),
        identity={"scene_uid": "ssl-moving-provenance-fixture"},
        room={"room_uid": "ssl-room", "room_dim_m": [6, 6, 3]},
        array=array,
        actors=tuple(actors),
        timebase=timebase,
    )

    class Provider:
        def sample_moving_scene_tracks(self, request, **kwargs):
            del kwargs
            trajectory = np.concatenate(
                [actor.positions_m for actor in request.actor_trajectories.values()],
                axis=1,
            )
            full = np.zeros((2, 2, 2, 4), dtype=np.float32)
            full[..., 0] = 1.0
            full[..., 3] = 0.2
            direct = full.copy()
            direct[..., 3] = 0.0
            moving_plan = {
                "trajectory": trajectory.tolist(),
                "trajectory_timebase": request.timebase.to_dict(),
                "source_loc": trajectory[0].tolist(),
                "sample_rate": 8000,
                "room_dim": [6, 6, 3],
                "source_ids": ["source_0", "source_1"],
                "group_uid": "ssl-moving-group",
                "num_samples": 4,
                "mic_loc": np.asarray(plan.array.positions_m).tolist(),
                "geometry": {"positions_m": np.asarray(plan.array.positions_m).tolist()},
            }
            return MovingSpatialRIRTrackSet(
                full,
                direct,
                moving_plan,
                {"generator_version": "moving-rir-v3", "rir_sha256": "a" * 64},
            )

    dataset = SimpleNamespace(
        sim_cfg=SimpleNamespace(rir_backend="online", rir_normalize=False),
        _simulation_view=SimpleNamespace(
            shared=SimpleNamespace(spatial=object())
        ),
    )
    context = SceneRenderContext(DatasetAcousticRuntimeAdapter(dataset))
    monkeypatch.setattr(
        spatial_adapter.scene_sampler,
        "prepare_scene_plan",
        lambda **kwargs: plan,
    )
    request = spatial_adapter.SpatialSceneRenderRequest(
        context=context,
        rng=SimulationRNG.for_task(
            dataset,
            3,
            "ssl",
            rng_factory=lambda *args: np.random.default_rng(17),
        ),
        background_enabled=False,
        sources=(np.ones(32, dtype=np.float32), np.ones(32, dtype=np.float32)),
        source_uids=("source_0", "source_1"),
        index=3,
        segment_length=32,
        sample_rate=8000,
        provider=Provider(),
        moving=True,
        trajectory_points=2,
        max_displacement_m=0.25,
        reference_channel=0,
        early_boundary=1,
        timeline=(),
    )
    first = spatial_adapter.SpatialSceneRenderer().render(request)
    second = spatial_adapter.SpatialSceneRenderer().render(request)
    for key in (
        "scene_uid",
        "geometry_identity",
        "track_identity",
        "track_facts",
        "rir",
    ):
        assert first.provenance[key] == second.provenance[key]
    np.testing.assert_array_equal(first.source_positions, second.source_positions)


def test_ssl_profile_marker_does_not_convert_dataset_configuration():
    from core.utils.config import load_merged_config
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    root = Path(__file__).resolve().parents[1]
    cfg = load_merged_config(root / "configs/ssl/fnssl_dynamic.yaml")
    current = SSLTrainingEntry._dataset_config(cfg, "train")
    legacy_marker = deepcopy(cfg)
    legacy_marker["meta"]["ssl_profile"] = "legacy_fixed_array_v1"
    legacy = SSLTrainingEntry._dataset_config(legacy_marker, "train")
    for key in (
        "input_representation",
        "target_representation",
        "output_stride",
        "coordinate_frame",
        "simulation_cfg",
    ):
        assert legacy[key] == current[key]
