"""Integration tests: aec_echo_device_eq wired end-to-end through DynamicMixingDataset._run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile

from tasks.aec.data.dynamic_dataset import DynamicMixingDataset as AECDynamicMixingDataset


def _write_scp(path: Path, entries: dict[str, Path]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for uid, audio_path in entries.items():
            info = soundfile.info(str(audio_path))
            f.write(f"{uid} {info.samplerate} {info.frames} {audio_path}\n")


def _make_wav(path: Path, duration: float, sr: int, *, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    signal = np.sin(2.0 * np.pi * 300.0 * t).astype(np.float32)
    signal += 0.01 * rng.standard_normal(signal.shape).astype(np.float32)
    signal = np.clip(signal, -0.99, 0.99)
    soundfile.write(path, signal, sr)


def _make_impulse_rir(path: Path, sr: int) -> None:
    rir = np.zeros(sr, dtype=np.float32)
    rir[0] = 1.0
    soundfile.write(path, rir, sr)


@pytest.fixture
def aec_paths(tmp_path: Path):
    sr = 16000
    duration = 2.0

    speech_dir = tmp_path / "speech"
    noise_dir = tmp_path / "noise"
    rir_dir = tmp_path / "rir"
    wind_dir = tmp_path / "wind"
    aec_dir = tmp_path / "aec"
    for d in (speech_dir, noise_dir, rir_dir, wind_dir, aec_dir):
        d.mkdir()

    speech_wav = speech_dir / "spk1_001.wav"
    noise_wav = noise_dir / "noise_001.wav"
    rir_wav = rir_dir / "rir_001.wav"
    wind_wav = wind_dir / "wind_001.wav"
    farend_wav = aec_dir / "farend_001.wav"
    echo_wav = aec_dir / "echo_001.wav"

    _make_wav(speech_wav, duration, sr, seed=10)
    _make_wav(noise_wav, duration, sr, seed=20)
    _make_impulse_rir(rir_wav, sr)
    _make_wav(wind_wav, duration, sr, seed=30)
    _make_wav(farend_wav, duration, sr, seed=40)
    farend, _ = soundfile.read(farend_wav, dtype="float32")
    echo = np.zeros_like(farend)
    delay = int(0.05 * sr)
    echo[delay:] = farend[:-delay] * 0.3
    soundfile.write(echo_wav, echo, sr)
    stereo_wav = aec_dir / "stereo_001.wav"
    soundfile.write(stereo_wav, np.stack([echo, farend], axis=1), sr)

    speech_scp = tmp_path / "speech.scp"
    noise_scp = tmp_path / "noise.scp"
    rir_scp = tmp_path / "rir.scp"
    wind_scp = tmp_path / "wind.scp"
    pair_json = tmp_path / "farend_echo.json"
    _write_scp(speech_scp, {"spk1_001": speech_wav})
    _write_scp(noise_scp, {"noise_001": noise_wav})
    _write_scp(rir_scp, {"rir_001": rir_wav})
    _write_scp(wind_scp, {"wind_001": wind_wav})
    with pair_json.open("w", encoding="utf-8") as f:
        json.dump({"pairs": [str(stereo_wav)]}, f)

    return {
        "speech_scp": str(speech_scp),
        "noise_scp": str(noise_scp),
        "rir_scp": str(rir_scp),
        "windnoise_scp": str(wind_scp),
        "real_farend_echo_pair_json": str(pair_json),
        "aec_ref_echo_random_delay_prob": 0.0,
        "max_duration_seconds": 1.0,
        "target_sr": sr,
        "task": "joint",
        "strict_load": True,
    }


def _simulation_cfg(**overrides):
    cfg = {
        "prob_target_only": 0.0,
        "prob_noise_only": 0.0,
        "prob_echo_only": 0.0,
        "prob_noise_echo_only": 0.0,
        "prob_target_noise": 0.0,
        "prob_target_echo": 1.0,
        "prob_target_noise_echo": 0.0,
        "snr_low": 5.0,
        "snr_high": 20.0,
        "prob_channel_enabled": False,
        "noisy_post_mix_degradations_enabled": False,
        "prob_reverberation": 0.0,
        "ser_range": [-10.0, 10.0],
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def echo_eq_dataset(aec_paths):
    ds = AECDynamicMixingDataset(
        **aec_paths,
        simulation_cfg=_simulation_cfg(),
        aec_echo_device_eq={
            "enabled": True,
            "prob": 1.0,
            "variant_probs": {"random_eq": 1.0},
        },
    )
    ds.set_epoch(0, base_seed=1)
    ds.export_simulation_stems = True
    return ds


def test_construction_rejects_unknown_variant_probs_key(aec_paths):
    with pytest.raises(ValueError, match="variant_probs has unknown variant"):
        AECDynamicMixingDataset(
            **aec_paths,
            simulation_cfg=_simulation_cfg(),
            aec_echo_device_eq={"enabled": True, "variant_probs": {"sigmoid": 1.0}},
        )


def test_disabled_by_default_no_audit_change(aec_paths):
    ds = AECDynamicMixingDataset(**aec_paths, simulation_cfg=_simulation_cfg())
    ds.set_epoch(0, base_seed=1)
    ds.export_simulation_stems = True
    _ = ds[0]
    audit = ds._last_simulation_stems["simulation_by_layer"]["C"]["c3_echo_device_eq"]
    assert audit == {"applied": False, "variant": None}


def test_random_eq_variant_runs_and_reports_itself(echo_eq_dataset):
    _ = echo_eq_dataset[0]
    audit = echo_eq_dataset._last_simulation_stems["simulation_by_layer"]["C"]["c3_echo_device_eq"]
    assert audit["applied"] is True
    assert audit["variant"] == "random_eq"


def test_bass_boost_variant_runs_and_reports_itself(aec_paths):
    ds = AECDynamicMixingDataset(
        **aec_paths,
        simulation_cfg=_simulation_cfg(),
        aec_echo_device_eq={
            "enabled": True,
            "prob": 1.0,
            "variant_probs": {"bass_boost": 1.0},
        },
    )
    ds.set_epoch(0, base_seed=1)
    ds.export_simulation_stems = True
    _ = ds[0]
    audit = ds._last_simulation_stems["simulation_by_layer"]["C"]["c3_echo_device_eq"]
    assert audit["applied"] is True
    assert audit["variant"] == "bass_boost"


def test_device_eq_and_echo_clipping_compose_in_one_sample(aec_paths):
    """Both features can trigger on the same sample -- EQ (linear) runs before
    clipping (non-linear) in _run, and neither should crash or skip the other.
    """
    ds = AECDynamicMixingDataset(
        **aec_paths,
        simulation_cfg=_simulation_cfg(),
        aec_echo_device_eq={"enabled": True, "prob": 1.0, "variant_probs": {"random_eq": 1.0}},
        aec_echo_clipping={
            "enabled": True,
            "prob": 1.0,
            "path_probs": {"post_ser_distortion": 1.0},
            "paths": {
                "post_ser_distortion": {
                    "variant_probs": {"hard": 1.0},
                },
            },
        },
    )
    ds.set_epoch(0, base_seed=1)
    ds.export_simulation_stems = True
    _ = ds[0]
    layer_c = ds._last_simulation_stems["simulation_by_layer"]["C"]
    assert layer_c["c3_echo_device_eq"]["applied"] is True
    assert layer_c["c3_echo_clipping"]["applied"] is True


def test_construction_rejects_zero_variant_probs_weights(aec_paths):
    with pytest.raises(ValueError, match="variant_probs weights must sum to > 0"):
        AECDynamicMixingDataset(
            **aec_paths,
            simulation_cfg=_simulation_cfg(),
            aec_echo_device_eq={"enabled": True, "variant_probs": {"random_eq": 0.0}},
        )


def test_construction_rejects_inverted_random_eq_range(aec_paths):
    with pytest.raises(ValueError, match="random_eq_db_range must satisfy lo <= hi"):
        AECDynamicMixingDataset(
            **aec_paths,
            simulation_cfg=_simulation_cfg(),
            aec_echo_device_eq={"enabled": True, "random_eq_db_range": [6.0, -6.0]},
        )


def test_construction_rejects_non_positive_bass_boost_cutoff(aec_paths):
    with pytest.raises(ValueError, match="bass_boost_highpass_cutoff_range lower bound must be positive"):
        AECDynamicMixingDataset(
            **aec_paths,
            simulation_cfg=_simulation_cfg(),
            aec_echo_device_eq={"enabled": True, "bass_boost_highpass_cutoff_range": [-100.0, 200.0]},
        )


def test_construction_rejects_positive_bass_boost_attenuation(aec_paths):
    with pytest.raises(ValueError, match="bass_boost_attenuation_min_db must be <= 0 dB"):
        AECDynamicMixingDataset(
            **aec_paths,
            simulation_cfg=_simulation_cfg(),
            aec_echo_device_eq={"enabled": True, "bass_boost_attenuation_min_db": 6.0},
        )

pytestmark = [pytest.mark.integration]
