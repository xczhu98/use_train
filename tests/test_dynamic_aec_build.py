"""Tests for tasks.aec.data.dynamic_aec_build defaulting state-detector STFT params."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch

from tasks.aec.data.dynamic_aec_build import (
    build_dynamic_aec_dataset,
    resolve_aec_dataset_kwargs,
    validate_aec_dynamic_aec_config,
)


def _base_cfg():
    return {
        "data": {
            "mode": "dynamic_aec",
            "sample_rate": 16000,
            "placeholder_num_samples": 1,
            "train": {
                "real_farend_echo_pair_json": "dummy.json",
                "simulation_cfg": {},
            },
            "dataloader": {"batch_size": 1, "num_workers": 0},
        },
        "meta": {"seed": 0},
    }


@pytest.mark.parametrize("mode", ["dynamic", "offline"])
@pytest.mark.parametrize(
    "entrypoint",
    [
        resolve_aec_dataset_kwargs,
        build_dynamic_aec_dataset,
        validate_aec_dynamic_aec_config,
    ],
)
def test_specialized_aec_builder_rejects_generic_modes_before_pair_lookup(
    mode, entrypoint
):
    cfg = {
        "data": {
            "mode": mode,
            "train": {"simulation_cfg": {}},
        }
    }

    with pytest.raises(
        ValueError,
        match=rf"data\.mode {mode!r} belongs to task 'se', not 'aec'",
    ):
        if entrypoint is validate_aec_dynamic_aec_config:
            entrypoint(cfg)
        else:
            entrypoint(cfg, "train")


def test_stft_cfg_defaults_state_detector_frame_params():
    # state label 按需计算：仅当显式配置 state_detector_label_type 时
    # 才从 stft_cfg 注入 hop/win（否则逐样本白算 state label，见审计 WP-E Fix 5）
    cfg = _base_cfg()
    cfg["data"]["train"]["state_detector_label_type"] = "scenario"
    stft_cfg = {"n_fft": 512, "hop_length": 256, "win_length": 512}

    kwargs = resolve_aec_dataset_kwargs(cfg, "train", stft_cfg=stft_cfg)

    assert kwargs["state_detector_hop_length"] == 256
    assert kwargs["state_detector_win_length"] == 512


def test_state_detector_rejects_non_centered_stft():
    """现有 state label 算法只支持 centered STFT 帧定义。"""
    cfg = _base_cfg()
    cfg["data"]["train"]["state_detector_label_type"] = "scenario"
    stft_cfg = {
        "n_fft": 512,
        "hop_length": 256,
        "win_length": 512,
        "center": False,
    }

    with pytest.raises(ValueError, match=r"state detector.*stft\.center=false"):
        resolve_aec_dataset_kwargs(cfg, "train", stft_cfg=stft_cfg)


def test_no_label_type_skips_state_detector_injection():
    # 未配置 label_type：center=false 也不触发 state detector 专属门禁。
    cfg = _base_cfg()
    stft_cfg = {
        "n_fft": 512,
        "hop_length": 256,
        "win_length": 512,
        "center": False,
    }

    kwargs = resolve_aec_dataset_kwargs(cfg, "train", stft_cfg=stft_cfg)

    assert kwargs.get("state_detector_hop_length") is None
    assert kwargs.get("state_detector_win_length") is None


def test_explicit_state_detector_params_override_stft_cfg():
    cfg = _base_cfg()
    cfg["data"]["train"]["state_detector_hop_length"] = 123
    cfg["data"]["train"]["state_detector_win_length"] = 456
    stft_cfg = {"n_fft": 512, "hop_length": 256, "win_length": 512}

    kwargs = resolve_aec_dataset_kwargs(cfg, "train", stft_cfg=stft_cfg)

    assert kwargs["state_detector_hop_length"] == 123
    assert kwargs["state_detector_win_length"] == 456


def test_state_detector_hop_must_match_stft_hop():
    """标签帧率必须等于模型 STFT 帧率，错配在配置边界报错。"""
    cfg = _base_cfg()
    cfg["data"]["train"]["state_detector_label_type"] = "scenario"
    cfg["data"]["train"]["state_detector_hop_length"] = 123
    stft_cfg = {"n_fft": 512, "hop_length": 256, "win_length": 512}

    with pytest.raises(
        ValueError,
        match=r"state_detector_hop_length=123 must equal stft\.hop_length=256",
    ):
        resolve_aec_dataset_kwargs(cfg, "train", stft_cfg=stft_cfg)


def test_no_stft_cfg_leaves_params_unset():
    kwargs = resolve_aec_dataset_kwargs(_base_cfg(), "train", stft_cfg=None)

    assert "state_detector_hop_length" not in kwargs
    assert "state_detector_win_length" not in kwargs


@pytest.mark.parametrize(
    "key", ["aec_farend_echo_dual_file", "real_farend_echo_pair_scp"]
)
@pytest.mark.parametrize("value", [None, False, True, "legacy.json"])
def test_dynamic_aec_builder_rejects_removed_echo_pair_keys(key, value):
    cfg = _base_cfg()
    cfg["data"]["train"][key] = value

    with pytest.raises(ValueError, match=rf"{key}.*removed"):
        resolve_aec_dataset_kwargs(cfg, "train")


@pytest.mark.parametrize("key", ["aec_stream_align", "linear_aec"])
@pytest.mark.parametrize("location", ["section", "simulation_cfg"])
def test_dynamic_aec_builder_rejects_retired_frontend_dataset_keys(key, location):
    cfg = _base_cfg()
    if location == "section":
        cfg["data"]["train"][key] = {"enabled": True}
    else:
        cfg["data"]["train"]["simulation_cfg"][key] = {"enabled": True}

    with pytest.raises(ValueError, match=rf"{key}.*aec_frontend"):
        resolve_aec_dataset_kwargs(cfg, "train")

    source = (
        cfg["data"]["train"]
        if location == "section"
        else cfg["data"]["train"]["simulation_cfg"]
    )
    assert source[key] == {"enabled": True}


@pytest.mark.parametrize("key", ["aec_stream_align", "linear_aec"])
def test_dynamic_aec_builder_rejects_data_level_frontend_keys(key):
    cfg = _base_cfg()
    cfg["data"][key] = {"enabled": True}

    with pytest.raises(ValueError, match=rf"{key}.*aec_frontend"):
        resolve_aec_dataset_kwargs(cfg, "train")


def _write_scp(path: Path, uid: str, audio_path: Path, sr: int, length: int) -> None:
    path.write_text(f"{uid} {sr} {length} {audio_path}\n", encoding="utf-8")


def test_dynamic_aec_builder_constructs_and_samples_real_small_manifests(tmp_path):
    sr = 16000
    length = sr
    t = np.arange(length, dtype=np.float32) / sr
    speech = 0.1 * np.sin(2 * np.pi * 300 * t)
    noise = 0.01 * np.sin(2 * np.pi * 700 * t)
    rir = np.zeros(length, dtype=np.float32)
    rir[0] = 1.0
    echo = 0.05 * np.sin(2 * np.pi * 300 * t)
    farend = 0.1 * np.sin(2 * np.pi * 300 * t)
    speech_wav, noise_wav, rir_wav = (
        tmp_path / "speech.wav",
        tmp_path / "noise.wav",
        tmp_path / "rir.wav",
    )
    stereo_wav = tmp_path / "pair.wav"
    soundfile.write(speech_wav, speech, sr, subtype="FLOAT")
    soundfile.write(noise_wav, noise, sr, subtype="FLOAT")
    soundfile.write(rir_wav, rir, sr, subtype="FLOAT")
    soundfile.write(stereo_wav, np.stack([echo, farend], axis=1), sr, subtype="FLOAT")
    speech_scp, noise_scp, rir_scp = (
        tmp_path / "speech.scp",
        tmp_path / "noise.scp",
        tmp_path / "rir.scp",
    )
    _write_scp(speech_scp, "speech", speech_wav, sr, length)
    _write_scp(noise_scp, "noise", noise_wav, sr, length)
    _write_scp(rir_scp, "rir", rir_wav, sr, length)
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps({"pairs": [str(stereo_wav)]}), encoding="utf-8")
    section = {
        "speech_scp": str(speech_scp),
        "noise_scp": str(noise_scp),
        "rir_scp": str(rir_scp),
        "real_farend_echo_pair_json": str(pairs),
        "max_duration_seconds": 0.5,
        "simulation_cfg": {
            "prob_target_only": 0.0,
            "prob_noise_only": 0.0,
            "prob_echo_only": 0.0,
            "prob_noise_echo_only": 0.0,
            "prob_target_noise": 0.0,
            "prob_target_echo": 1.0,
            "prob_target_noise_echo": 0.0,
            "prob_reverberation": 0.0,
            "prob_wind_noise_sidechain": 0.0,
            "prob_channel_enabled": False,
            "noisy_post_mix_degradations_enabled": False,
            "aec_ref_echo_random_delay_prob": 0.0,
        },
    }
    cfg = {
        "data": {
            "mode": "dynamic_aec",
            "sample_rate": sr,
            "train": section,
            "validation": {
                "inherit_simulation": True,
                "speech_scp": str(speech_scp),
                "noise_scp": str(noise_scp),
                "rir_scp": str(rir_scp),
                "real_farend_echo_pair_json": str(pairs),
            },
        },
        "meta": {"seed": 7},
    }

    train_ds = build_dynamic_aec_dataset(cfg, "train")
    val_ds = build_dynamic_aec_dataset(cfg, "val")
    val_ds.set_epoch(0, base_seed=7)
    traced = train_ds.get_sample_with_trace(0, force_branch="prob_target_echo")
    val_sample = val_ds[0]

    assert len(traced.sample) == len(val_sample) == 11
    assert traced.sample[3] == val_sample[3] == sr
    assert int(traced.sample[4]) == int(val_sample[4]) == sr // 2
    assert traced.trace.metadata["branch"] == "prob_target_echo"
    assert traced.trace.sources["scene"].metadata["branch"] == "prob_target_echo"
    assert torch.isfinite(traced.sample[0]).all()
    assert torch.isfinite(val_sample[0]).all()
    assert val_ds.max_duration_seconds == 0.5


def test_val_inherits_aec_ref_echo_settings_after_train_resolution():
    """Regression: resolving train must not mutate cfg before val inheritance."""
    cfg = _base_cfg()
    cfg["data"]["train"]["simulation_cfg"]["aec_ref_echo_gcc_farend_pad_policy"] = "apply"
    cfg["data"]["train"]["simulation_cfg"]["aec_ref_echo_random_delay_prob"] = 0.0
    cfg["data"]["validation"] = {
        "inherit_simulation": True,
        "speech_scp": "dummy_val.scp",
    }

    resolve_aec_dataset_kwargs(cfg, "train", stft_cfg=None)
    val_kwargs = resolve_aec_dataset_kwargs(cfg, "val", stft_cfg=None)

    assert val_kwargs["aec_ref_echo_gcc_farend_pad_policy"] == "apply"
    # Original cfg must remain intact so that subsequent builds see the same config.
    assert cfg["data"]["train"]["simulation_cfg"]["aec_ref_echo_gcc_farend_pad_policy"] == "apply"

pytestmark = [pytest.mark.unit]
