"""SSL canonical 声源数与 VAD 投影配置贯通验收。"""

import json
import math

import numpy as np
import pytest
import soundfile as sf
import torch


def _ssl_config(*, num_tracks: int | None = 3, loss_max_tracks=4):
    ssl = {"input_representation": "reference_ipd"}
    if num_tracks is not None:
        ssl["num_tracks"] = num_tracks
    loss_args = {} if loss_max_tracks is ... else {"max_tracks": loss_max_tracks}
    return {
        "meta": {"seed": 7},
        "ssl": ssl,
        "model": {"type": "FixedArrayIPDNet", "args": {"input_size": 1}},
        "data": {
            "train": {"tracks": 3},
            "validation": {"max_track": 3},
        },
        "loss": {"type": "SSLDPIPDPITLoss", "args": loss_args},
    }


def _official_ssl_geometry_case(tmp_path, *, model_type, channels):
    """从已注册官方模板构造可运行的小型离线阵列 fixture。"""

    from core.data.simulation.scene_manifest import build_track_identity
    from core.utils.config import load_merged_config

    config_path = {
        "IPDNetFixed": "configs/ssl/ipdnet_fixed_dynamic.yaml",
        "IPDNetVariable": "configs/ssl/ipdnet_variable_dynamic.yaml",
    }[model_type]
    config = load_merged_config(config_path)
    config["meta"].pop("ssl_profile", None)
    config["meta"]["seed"] = 7
    config["ssl"]["num_tracks"] = 2
    config["data"]["sample_rate"] = 8000
    config["stft"] = {
        "n_fft": 16,
        "hop_length": 8,
        "win_length": 16,
        "window": "hamming",
        "center": False,
    }
    model_args = config["model"]["args"]
    model_args.update(
        {
            "input_size": 4,
            "max_track": 2,
            "n_fft": 16,
            "hidden_size": 4,
            "cnn_hidden_dim": 4,
            "dropout": 0.0,
            "stride": 12,
        }
    )

    sample_rate = 8000
    sample_count = 200  # center=False 时产生 24 帧，可被官方 stride=12 整除。
    sample_index = np.arange(sample_count, dtype=np.float32) / sample_rate
    speech_paths = []
    for index, frequency in enumerate((240.0, 420.0)):
        path = tmp_path / f"speech_{index}.wav"
        waveform = np.sin(2.0 * np.pi * frequency * sample_index).astype(np.float32)
        sf.write(path, waveform, sample_rate, subtype="FLOAT")
        speech_paths.append(path)
    speech_scp = tmp_path / "speech.scp"
    speech_scp.write_text(
        "\n".join(
            f"source_{index} {sample_rate} {sample_count} {path}"
            for index, path in enumerate(speech_paths)
        )
        + "\n",
        encoding="utf-8",
    )

    positions = [[2.0 + 0.04 * index, 2.0, 1.0] for index in range(channels)]
    rir_paths = []
    direct_paths = []
    for source_index in range(2):
        rir = np.zeros((channels, 8), dtype=np.float32)
        for channel in range(channels):
            rir[channel, min(7, source_index + channel)] = 1.0 - 0.1 * channel
        rir_path = tmp_path / f"rir_{source_index}.npy"
        np.save(rir_path, rir)
        rir_paths.append(rir_path)
        direct_path = tmp_path / f"direct_{source_index}.npy"
        np.save(direct_path, rir.copy())
        direct_paths.append(direct_path)
    rir_manifest = tmp_path / "rir_manifest.json"
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
                    "scene_uid": f"ssl-{model_type}-{channels}ch",
                    "actor_order": ["source_0", "source_1"],
                    "room": {"room_uid": "ssl-room", "room_dim_m": [6, 6, 3]},
                    "actor_positions_m": {
                        "source_0": [1.0, 1.0, 1.5],
                        "source_1": [2.0, 2.0, 1.5],
                    },
                    "generator_version": "ssl-config-completion-fixture-v1",
                    "track_identities": {
                        f"source_{index}": {
                            "full": build_track_identity(rir_paths[index]),
                            "direct": build_track_identity(direct_paths[index]),
                            "early": build_track_identity(direct_paths[index]),
                        }
                        for index in range(2)
                    },
                    "geometry": {
                        "positions_m": positions,
                        "reference_channel": 0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    simulation_cfg = {
        "rir_backend": "offline",
        "spatial_enabled": True,
        "reference_channel": 0,
        "custom_geometry": {"positions_m": positions},
        "prob_reverberation": 1.0,
        "prob_moving_rir": 0.0,
        "prob_channel_enabled": False,
        "noisy_post_mix_degradations_enabled": False,
        "peak_normalization_mode": "off",
    }
    for split in ("train", "validation"):
        section = config["data"][split]
        section.update(
            {
                "speech_scp": str(speech_scp),
                "noise_scp": None,
                "rir_scp": str(rir_manifest),
                "tracks": 2,
                "segment_seconds": sample_count / sample_rate,
                "freqs": 9,
                "trajectory_points": 1,
                "simulation_cfg": dict(simulation_cfg),
            }
        )
    return config


def _official_entry_datasets(tmp_path, *, model_type, channels):
    from tasks.ssl.data.dynamic_dataset import SSLDynamicMixingDataset
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    config = _official_ssl_geometry_case(
        tmp_path, model_type=model_type, channels=channels
    )
    entry = SSLTrainingEntry("ssl_geometry_contract")
    train_cfg = entry._dataset_config(config, "train")
    validation_cfg = entry._dataset_config(config, "validation")
    validation_cfg["simulation_split"] = "validation"
    train_dataset = SSLDynamicMixingDataset(**train_cfg)
    validation_dataset = SSLDynamicMixingDataset(**validation_cfg)
    return config, entry, train_dataset, validation_dataset


@pytest.mark.unit
def test_ssl_num_tracks_canonical_and_legacy_derivation():
    from tasks.ssl.config import resolve_ssl_num_tracks

    canonical = _ssl_config()
    assert resolve_ssl_num_tracks(canonical) == 3

    legacy = _ssl_config(num_tracks=None, loss_max_tracks=4)
    legacy["model"]["args"]["max_track"] = 3
    assert resolve_ssl_num_tracks(legacy) == 3


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("canonical", True),
        ("canonical", None),
        ("canonical", 0),
        ("model", True),
        ("model", None),
        ("model", 0),
        ("train", True),
        ("train", None),
        ("train", 0),
        ("validation", True),
        ("validation", None),
        ("validation", 0),
    ],
)
@pytest.mark.unit
def test_ssl_num_tracks_rejects_bool_null_and_zero(field, bad_value):
    from tasks.ssl.config import resolve_ssl_num_tracks

    config = _ssl_config()
    if field == "canonical":
        config["ssl"]["num_tracks"] = bad_value
        expected = "ssl.num_tracks"
    elif field == "model":
        config["model"]["args"]["max_track"] = bad_value
        expected = "model.args.max_track"
    elif field == "train":
        config["data"]["train"]["tracks"] = bad_value
        expected = "data.train.tracks"
    else:
        config["data"]["validation"]["max_track"] = bad_value
        expected = "data.validation.max_track"

    with pytest.raises(ValueError, match=expected.replace(".", r"\.")):
        resolve_ssl_num_tracks(config)


@pytest.mark.parametrize(
    "conflict",
    ["model", "train", "validation", "split_fields", "legacy_fields"],
)
@pytest.mark.unit
def test_ssl_num_tracks_rejects_conflicting_declarations(conflict):
    from tasks.ssl.config import resolve_ssl_num_tracks

    config = _ssl_config()
    if conflict == "model":
        config["model"]["args"]["max_track"] = 2
    elif conflict == "train":
        config["data"]["train"]["tracks"] = 2
    elif conflict == "validation":
        config["data"]["validation"]["max_track"] = 2
    elif conflict == "split_fields":
        config["data"]["train"]["max_track"] = 2
    else:
        config["ssl"].pop("num_tracks")
        config["model"]["args"]["max_track"] = 2

    with pytest.raises(
        ValueError,
        match="source-count|source count|must match|must be equal",
    ):
        resolve_ssl_num_tracks(config)


@pytest.mark.unit
def test_ssl_num_tracks_uses_pit_max_tracks_as_capacity():
    from tasks.ssl.config import resolve_ssl_num_tracks

    default_capacity = _ssl_config(num_tracks=8, loss_max_tracks=...)
    default_capacity["model"]["args"]["max_track"] = 8
    default_capacity["data"]["train"]["tracks"] = 8
    default_capacity["data"]["validation"]["max_track"] = 8
    assert resolve_ssl_num_tracks(default_capacity) == 8

    explicit_capacity = _ssl_config(num_tracks=3, loss_max_tracks=5)
    assert resolve_ssl_num_tracks(explicit_capacity) == 3

    too_small = _ssl_config(num_tracks=5, loss_max_tracks=4)
    too_small["model"]["args"]["max_track"] = 5
    too_small["data"]["train"]["tracks"] = 5
    too_small["data"]["validation"]["max_track"] = 5
    with pytest.raises(ValueError, match="capacity must be >= canonical source count"):
        resolve_ssl_num_tracks(too_small)

    over_default = _ssl_config(num_tracks=9, loss_max_tracks=...)
    over_default["model"]["args"]["max_track"] = 9
    over_default["data"]["train"]["tracks"] = 9
    over_default["data"]["validation"]["max_track"] = 9
    with pytest.raises(ValueError, match="capacity must be >= canonical source count"):
        resolve_ssl_num_tracks(over_default)


@pytest.mark.parametrize("bad_capacity", [True, None, 0])
@pytest.mark.unit
def test_ssl_pit_capacity_rejects_bool_null_and_zero(bad_capacity):
    from tasks.ssl.config import resolve_ssl_num_tracks

    config = _ssl_config(loss_max_tracks=bad_capacity)
    with pytest.raises(ValueError, match=r"loss\.args\.max_tracks"):
        resolve_ssl_num_tracks(config)


@pytest.mark.unit
def test_train_validation_dataset_and_model_share_canonical_track_count(monkeypatch):
    from core.data import config_flags, dataset_build, resource_paths
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    config = _ssl_config()
    simulation_cfg = {
        "rir_backend": "offline",
        "spatial_enabled": True,
        "reference_channel": 0,
        "custom_geometry": {"positions_m": [[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]]},
    }
    for section in config["data"].values():
        section.clear()
    config["data"]["train"].update({"tracks": 3, "simulation_cfg": simulation_cfg})
    config["data"]["validation"].update(
        {"max_track": 3, "simulation_cfg": simulation_cfg}
    )
    monkeypatch.setattr(config_flags, "resolve_strict_load", lambda _: True)
    monkeypatch.setattr(config_flags, "resolve_simulation_cfg_strict", lambda _: True)
    monkeypatch.setattr(config_flags, "resolve_placeholder_num_samples", lambda _: 1)
    monkeypatch.setattr(dataset_build, "validate_layer_a_runtime_config", lambda *_a, **_k: None)
    monkeypatch.setattr(resource_paths, "resolve_data_section_resource_paths", lambda _section: None)

    entry = SSLTrainingEntry("ssl_config_completion")
    train_cfg = entry._dataset_config(config, "train")
    val_cfg = entry._dataset_config(config, "validation")
    assert (train_cfg["tracks"], train_cfg["max_track"]) == (3, 3)
    assert (val_cfg["tracks"], val_cfg["max_track"]) == (3, 3)
    assert train_cfg["vad_mode"] == val_cfg["vad_mode"] == "direct_path"

    model = entry.build_model(config, rank=0)
    assert model.max_track == 3
    assert "max_track" not in config["model"]["args"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_type", "channels", "expected_dataset_size"),
    [
        ("IPDNetFixed", 2, 4),
        ("IPDNetVariable", 2, 4),
        ("IPDNetVariable", 3, 6),
    ],
)
def test_official_entry_builds_train_and_validation_datasets_from_array_geometry(
    tmp_path, model_type, channels, expected_dataset_size
):
    config, entry, train_dataset, validation_dataset = _official_entry_datasets(
        tmp_path, model_type=model_type, channels=channels
    )

    assert train_dataset.input_size == expected_dataset_size
    assert validation_dataset.input_size == expected_dataset_size
    assert train_dataset.tracks == validation_dataset.tracks == 2
    assert config["model"]["args"]["input_size"] == 4

    model = entry.build_model(config, rank=0)
    assert model.input_size == 4
    assert model.max_track == 2


@pytest.mark.unit
def test_official_entry_rejects_split_feature_size_that_conflicts_with_geometry(tmp_path):
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    config = _official_ssl_geometry_case(
        tmp_path, model_type="IPDNetFixed", channels=2
    )
    config["data"]["train"]["input_size"] = 10

    with pytest.raises(
        ValueError,
        match=r"data\.train\.input_size must equal feature dimension 4",
    ):
        SSLTrainingEntry("ssl_geometry_contract")._dataset_config(config, "train")


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_path",
    [
        "configs/ssl/fnssl_dynamic.yaml",
        "configs/ssl/ipdnet_fixed_dynamic.yaml",
        "configs/ssl/ipdnet_variable_dynamic.yaml",
        "configs/ssl/ipdnet2_dynamic.yaml",
    ],
)
def test_official_ssl_templates_build_dpipd_loss_from_merged_config(config_path):
    import tasks.ssl.losses  # noqa: F401 -- register SSL losses
    from core.losses.build import build_loss
    from core.utils.config import load_merged_config

    config = load_merged_config(config_path)
    loss = build_loss(
        config["loss"],
        stft_cfg=config.get("stft"),
        sample_rate=config["data"]["sample_rate"],
    )

    assert config["loss"]["args"] is None
    assert loss.max_tracks == 8
    assert loss.doa_loss_weight == 0.0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("model_type", "channels"),
    [("IPDNetFixed", 2), ("IPDNetVariable", 3)],
)
def test_official_entry_geometry_reaches_cuda_trainer_forward_and_backward(
    tmp_path, model_type, channels
):
    from torch.utils.data import DataLoader

    import tasks.ssl.losses  # noqa: F401 -- register SSL losses
    from core.losses.build import build_loss
    from tasks.ssl.data import ssl_collate_fn
    from tasks.ssl.trainer.ssl_trainer import SSLTrainer

    config, entry, train_dataset, _ = _official_entry_datasets(
        tmp_path, model_type=model_type, channels=channels
    )
    loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=ssl_collate_fn,
    )
    model = entry.build_model(config, rank=0).to("cuda")
    loss_fn = build_loss(
        config["loss"],
        stft_cfg=config["stft"],
        sample_rate=config["data"]["sample_rate"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer_config = {
        "meta": {"allow_no_stft": True, "seed": config["meta"]["seed"]},
        "model": config["model"],
        "data": {"sample_rate": config["data"]["sample_rate"]},
        "loss": config["loss"],
        "distributed": {"find_unused_parameters": False},
        "trainer": {
            "epochs": 1,
            "exp_path": str(tmp_path / "cuda_experiment"),
            "strict_checks": True,
            "val_metrics": {"enabled": False},
        },
    }
    trainer = SSLTrainer(
        train_loader=loader,
        val_loader=loader,
        sample_rate=config["data"]["sample_rate"],
        stft_module=None,
        rank=0,
        world_size=1,
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
        loss_fn=loss_fn,
        config=trainer_config,
    )

    batch = next(iter(loader)).to("cuda")
    loss, finite, _ = trainer._forward_and_compute_loss(batch, 0, 0, 8000)
    assert finite
    assert loss.device.type == "cuda"
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)


@pytest.mark.unit
def test_direct_path_vad_is_consumed_and_recorded_in_provenance():
    from tasks.ssl.data.projection import (
        SSLProjectionConfig,
        SSLSceneInput,
        project_ssl_scene,
    )

    sample_rate = 16000
    sample_index = torch.arange(64, dtype=torch.float32)
    waveform = torch.sin(2.0 * math.pi * 1000.0 * sample_index / sample_rate)
    direct_tracks = torch.zeros(2, 2, sample_index.numel(), dtype=torch.float32)
    direct_tracks[0, 0] = waveform
    direct_tracks[0, 1] = 0.7 * waveform
    scene = SSLSceneInput(
        observation=direct_tracks.sum(dim=0),
        direct_tracks=direct_tracks,
        mic_positions=torch.tensor([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]]),
        source_positions=torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        sample_rate=sample_rate,
        provenance={"backend": "unit_fixture"},
        coordinate_frame="array_local",
    )
    projection = SSLProjectionConfig(
        n_fft=16,
        hop_length=8,
        win_length=16,
        window="hann",
        center=False,
        input_representation="reference_ipd",
        vad_threshold_ratio=0.5,
        vad_mode="direct_path",
    )

    item = project_ssl_scene(scene, projection)
    assert bool(torch.all(item["vad"][:, 0] == 1.0))
    assert bool(torch.all(item["vad"][:, 1] == 0.0))
    assert item["provenance"]["vad_supervision"] == {
        "mode": "direct_path",
        "activity_source": "direct_path_spectra",
        "threshold_ratio": 0.5,
        "aggregation": "framewise",
    }


@pytest.mark.unit
def test_projection_rejects_unimplemented_vad_mode():
    from tasks.ssl.data.projection import SSLProjectionConfig

    with pytest.raises(ValueError, match="vad_mode"):
        SSLProjectionConfig(vad_mode="mixture_energy")
