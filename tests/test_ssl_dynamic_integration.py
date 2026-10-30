"""SSL typed offline scene 到完整 Dataset 投影的集成测试。"""

import json

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from core.utils.dataloader_worker import dataloader_worker_init_fn
from tasks.ssl.data.collate import ssl_collate_fn
from tasks.ssl.data.dynamic_dataset import SSLDynamicMixingDataset
from core.data.simulation.scene_manifest import build_track_identity


pytestmark = pytest.mark.unit


def _spawn_worker_take_first(batch):
    """保持在模块顶层以满足 spawn worker 的函数 pickle 合同。"""

    return batch[0]


def test_ssl_dynamic_supplied_rir_runs_shared_layers_into_projection(tmp_path):
    sr = 8000
    length = 128
    time = np.arange(length, dtype=np.float32) / sr
    speech_paths = []
    for index, frequency in enumerate((240.0, 420.0)):
        path = tmp_path / f"speech_{index}.wav"
        sf.write(path, np.sin(2 * np.pi * frequency * time).astype(np.float32), sr, subtype="FLOAT")
        speech_paths.append(path)
    speech_scp = tmp_path / "speech.scp"
    speech_scp.write_text(
        "\n".join(f"source_{i} {sr} {length} {path}" for i, path in enumerate(speech_paths)) + "\n",
        encoding="utf-8",
    )

    rir_paths = []
    direct_paths = []
    for index, delay in enumerate((0, 1)):
        rir = np.zeros((2, 8), dtype=np.float32)
        rir[0, delay] = 1.0
        rir[1, delay + 1] = 0.8
        path = tmp_path / f"rir_{index}.npy"
        np.save(path, rir)
        rir_paths.append(path)
        direct = rir.copy()
        direct[0, delay] *= 0.9
        direct_path = tmp_path / f"direct_{index}.npy"
        np.save(direct_path, direct)
        direct_paths.append(direct_path)
    rir_manifest = tmp_path / "spatial_rir.json"
    rir_manifest.write_text(
        json.dumps(
            {
                "entries": {
                    "source_0": {'full': str(rir_paths[0]), 'direct': str(direct_paths[0]), 'early': str(direct_paths[0])},
                    "source_1": {'full': str(rir_paths[1]), 'direct': str(direct_paths[1]), 'early': str(direct_paths[1])},
                },
                "metadata": {
                    "schema_version": 1,
                    "sample_rate": sr,
                    "scene_uid": "ssl-offline-scene",
                    "actor_order": ["source_0", "source_1"],
                    "room": {"room_uid": "ssl-room", "room_dim_m": [6, 6, 3]},
                    "actor_positions_m": {"source_0": [1, 1, 1.5], "source_1": [2, 2, 1.5]},
                    "generator_version": "ssl-fixture-v1",
                    "track_identities": {
                        f"source_{i}": {
                            "full": build_track_identity(rir_paths[i]),
                            "direct": build_track_identity(direct_paths[i]),
                            "early": build_track_identity(direct_paths[i]),
                        } for i in range(2)
                    },
                    "geometry": {
                        "positions_m": [[2.0, 2.0, 1.0], [2.08, 2.0, 1.0]],
                        "reference_channel": 0,
                        "array_profile": "linear_2ch_80mm",
                        "source_positions": [[1.0, 1.0, 1.5], [2.0, 2.0, 1.5]],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    dataset = SSLDynamicMixingDataset(
        str(speech_scp),
        tracks=2,
        sample_rate=sr,
        segment_seconds=length / sr,
        rir_scp=str(rir_manifest),
        simulation_cfg={
            "spatial_enabled": True,
            "reference_channel": 0,
            "prob_reverberation": 1.0,
            "prob_channel_enabled": False,
            "noisy_post_mix_degradations_enabled": False,
            "peak_normalization_mode": "off",
        },
        custom_geometry={"positions_m": [[2.0, 2.0, 1.0], [2.08, 2.0, 1.0]]},
        n_fft=16,
        hop_length=8,
        win_length=16,
        window="hamming",
        center=False,
    )

    item = dataset[0]

    assert item["features"].shape == (15, 9, 4)
    assert item["target_dpipd"].shape == (15, 9, 2, 1, 2)
    assert item["doa"].shape == (15, 2, 2)
    assert item["vad"].shape == (15, 2)
    assert item["provenance"]["direct_path_exact"] is True
    assert item["provenance"]["direct_path_contract"] == "core_spatial_direct_rir"
    assert item["provenance"]["backend"] == "offline"
    assert "noise_image_provenance" in item["provenance"]
    assert item["provenance"]["scene_render"]["noise_image_provenance"] == item[
        "provenance"
    ]["noise_image_provenance"]
    assert item["provenance"]["post_mix_stage_order"] == [
        "pre-finite", "D-normalize", "D-guard", "E", "F",
        "task-post-EF", "final-guard", "post-finite",
    ]
    assert item["provenance"]["decomposable_final"] is True
    assert item["provenance"]["layer_d_facts"]["guard"]["preserves_decomposition"] is True
    scene = item["provenance"]["scene"]
    assert scene["post_mix_stage_order"] == item["provenance"]["post_mix_stage_order"]
    assert scene["decomposable_final"] is True
    assert scene["layer_d_facts"] == item["provenance"]["layer_d_facts"]
    assert scene["layer_e"] == item["provenance"]["layer_e"]
    assert scene["layer_f"] == item["provenance"]["layer_f"]
    assert np.isfinite(item["features"].numpy()).all()
    assert np.isfinite(item["target_dpipd"].numpy()).all()

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        multiprocessing_context="spawn",
        collate_fn=_spawn_worker_take_first,
        worker_init_fn=dataloader_worker_init_fn,
        generator=torch.Generator().manual_seed(1234),
    )
    spawned_item = next(iter(loader))
    assert spawned_item["provenance"]["backend"] == "offline"
    assert spawned_item["features"].shape == item["features"].shape
    assert torch.isfinite(spawned_item["features"]).all()


def test_ssl_dynamic_dataset_rejects_moving_rir(tmp_path):
    with pytest.raises(ValueError, match="moving RIR"):
        SSLDynamicMixingDataset(
            str(tmp_path / "speech.scp"),
            tracks=2,
            sample_rate=8000,
            simulation_cfg={
                "rir_backend": "online",
                "spatial_enabled": True,
                                "array_profile": "linear_2ch_80mm",
                "prob_moving_rir": 1.0,
            },
        )


def _scene_item(frames, *, targets_frames=None):
    return {
        "features": torch.zeros(frames, 3, 4),
        "target_dpipd": torch.zeros(targets_frames or frames, 3, 1, 1, 2),
        "doa": torch.zeros(targets_frames or frames, 1, 2),
        "vad": torch.zeros(targets_frames or frames, 1),
        "pair_index": torch.tensor([[0, 1]], dtype=torch.int64),
        "pair_mask": torch.ones(1, dtype=torch.bool),
        "geometry": torch.tensor([[2.0, 2.0, 1.0], [2.08, 2.0, 1.0]]),
        "input_lengths": frames,
        "output_lengths": targets_frames or frames,
        "frame_origin": 0,
        "time_stride": 1,
        "input_representation": "multichannel_stft_real_imag",
        "target_representation": "dpipd_far_field",
        "coordinate_frame": "room_xyz_m",
        "sample_rate": 16000,
        "provenance": {"backend": "fixture"},
    }


def test_ssl_collate_pads_across_items_and_keeps_per_item_frames_aligned():
    batch = ssl_collate_fn([_scene_item(4), _scene_item(6)])
    assert batch.features.shape[:2] == (2, 6)
    assert batch.target_dpipd.shape[0:2] == (2, 6)
    assert torch.equal(batch.input_lengths, torch.tensor([4, 6], dtype=torch.int64))
    assert torch.equal(batch.output_lengths, torch.tensor([4, 6], dtype=torch.int64))


def test_ssl_collate_rejects_item_with_mismatched_frame_axes():
    item = _scene_item(4, targets_frames=5)
    item["output_lengths"] = 4
    with pytest.raises(ValueError, match="lengths|frame"):
        ssl_collate_fn([item])
