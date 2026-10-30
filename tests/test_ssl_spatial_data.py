"""SSL dynamic_ssl 入口校验与模型 STFT 几何合同。"""

import pytest


pytestmark = pytest.mark.unit


def test_dynamic_ssl_requires_speech_noise_and_simulation_config():
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    section = {
        "provider": "online",
        "speech_scp": "speech.jsonl",
        "noise_scp": "noise.jsonl",
        "simulation_cfg": {"rir_backend": "online", "prob_moving_rir": 0.0},
    }
    resolved = SSLTrainingEntry._resolve_simulation_cfg(section, "train")
    assert resolved.rir_backend == "online"
    for missing in ("speech_scp", "noise_scp"):
        invalid = dict(section)
        invalid.pop(missing)
        with pytest.raises(ValueError, match=missing):
            SSLTrainingEntry._validate_split_resources("dynamic_ssl", invalid, "train")
    SSLTrainingEntry._validate_split_resources("dynamic_ssl", section, "train")
    with pytest.raises(ValueError, match="not consumed"):
        SSLTrainingEntry._validate_split_resources(
            "dynamic_ssl", {**section, "rir_scp": "unused.jsonl"}, "train"
        )
    with pytest.raises(ValueError, match="moving SSL requires trajectory_points >= 2"):
        SSLTrainingEntry._resolve_simulation_cfg(
            {"simulation_cfg": {"rir_backend": "online", "prob_moving_rir": 0.1}},
            "train",
        )
    accepted = SSLTrainingEntry._resolve_simulation_cfg(
        {
            "simulation_cfg": {
                "rir_backend": "online",
                "prob_moving_rir": 0.1,
                "spatial_enabled": True,
                "array_profile": "linear_2ch_80mm",
                "reference_channel": 0,
            },
            "trajectory_points": 4,
        },
        "train",
    )
    assert accepted.prob_moving_rir == 0.1


def test_ssl_rejects_non_dynamic_ssl_mode():
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    with pytest.raises(ValueError, match="only data.mode=dynamic_ssl"):
        SSLTrainingEntry._validate_split_resources("offline_ssl", {}, "train")


def test_dynamic_ssl_requires_simulation_cfg():
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    with pytest.raises(ValueError, match="required for formal dynamic SSL"):
        SSLTrainingEntry._resolve_simulation_cfg({"provider": "online"}, "train")


def test_ssl_entry_does_not_fallback_validation_to_legacy_val_section():
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    cfg = {
        "data": {
            "mode": "dynamic_ssl",
            "val": {"speech_scp": "legacy.jsonl"},
        },
        "model": {"args": {}},
    }
    with pytest.raises(ValueError, match="data.validation must be a non-empty mapping"):
        SSLTrainingEntry._dataset_config(cfg, "validation")


def test_model_consumes_and_validates_stft_geometry():
    from tasks.ssl.models import build_model

    model = build_model(
        {"type": "FixedArrayIPDNet", "args": {"input_size": 2, "max_track": 2}},
        stft_cfg={
            "n_fft": 32,
            "hop_length": 8,
            "win_length": 32,
            "window": "hamming",
            "center": False,
        },
    )
    assert (model.n_fft, model.hop_length, model.win_length, model.center) == (32, 8, 32, False)
    with pytest.raises(ValueError, match="context conflict"):
        build_model(
            {"type": "FixedArrayIPDNet", "args": {"input_size": 2, "max_track": 2, "n_fft": 64}},
            stft_cfg={
                "n_fft": 32,
                "hop_length": 8,
                "win_length": 32,
                "window": "hamming",
                "center": False,
            },
        )


def test_ssl_dataset_config_propagates_seed_and_shared_lifecycle_flags():
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    cfg = {
        "meta": {"seed": 17},
        "data": {
            "mode": "dynamic_ssl",
            "strict_load": True,
            "placeholder_num_samples": 33,
            "train": {
                "speech_scp": "speech.scp",
                "noise_scp": "noise.scp",
                "simulation_cfg": {
                    "spatial_enabled": True,
                    "array_profile": "linear_2ch_80mm",
                    "reference_channel": 0,
                },
            },
        },
        "model": {"args": {"input_size": 1, "max_track": 2}},
    }
    section = SSLTrainingEntry._dataset_config(cfg, "train")
    assert section["seed"] == 17
    assert section["strict_load"] is True
    assert section["placeholder_num_samples"] == 33

    cfg["data"]["strict_load"] = False
    with pytest.raises(ValueError, match="strict_load=True"):
        SSLTrainingEntry._dataset_config(cfg, "train")


def test_ssl_dataset_config_rejects_unconsumed_builder_only_split_keys():
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    cfg = {
        "meta": {"seed": 17},
        "data": {
            "mode": "dynamic_ssl",
            "strict_load": True,
            "train": {
                "speech_scp": "speech.scp",
                "noise_scp": "noise.scp",
                "rir_scp": "rir.scp",
                "simulation_cfg": {},
                "sampling": {"speech": {}},
            },
        },
        "model": {"args": {"input_size": 1, "max_track": 2}},
    }
    with pytest.raises(ValueError, match="builder-only keys unsupported by dynamic_ssl"):
        SSLTrainingEntry._dataset_config(cfg, "train")


def test_ssl_dataset_rejects_lenient_resource_mode_before_loading():
    from tasks.ssl.data.dynamic_dataset import SSLDynamicMixingDataset

    with pytest.raises(ValueError, match="strict_load=True"):
        SSLDynamicMixingDataset(
            "missing.speech.scp",
            tracks=1,
            sample_rate=8000,
            strict_load=False,
        )
