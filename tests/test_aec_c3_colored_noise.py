"""Tests: Layer C-3 colored noise（自原 Layer F-5 迁入声学混合层）的分支语义。

覆盖 tasks/aec/data/dynamic_dataset.py + core/data/dynamic_dataset_base.py：
1. need_c12 分支（prob_target_noise）prob_colored_noise=1.0：colored noise 进入
   mic 与 noise stem，实测 SNR（相对 clean target）落在 snr_range 内，
   mic == clean + noise stem（声学加性、可线性分解）。
2. 非 need_c12 分支（echo_only）prob_colored_noise=1.0：C-3 不施加——
   noise stem 全零、mic 为纯回声。这是迁移后有意的语义修复：物理上
   colored noise 是声学环境噪声，无近端信号的场景不应凭空出现。
3. prob_colored_noise=0.0：noise stem 仅剩基线 C-1 噪声（120 dB SNR），
   无 colored noise 混入。
"""

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


def _make_impulse_rir(path: Path, sr: int) -> None:
    rir = np.zeros(sr, dtype=np.float32)
    rir[0] = 1.0
    soundfile.write(path, rir, sr)


@pytest.fixture
def make_paths(tmp_path: Path):
    """构造 near / echo 同相正弦样本（风格同 test_aec_decomposable_metadata）。

    默认分支 prob_target_noise（need_c12=True、need_c3=False），snr=120 dB
    把基线 C-1 噪声压到可忽略量级，便于隔离 C-3 colored noise 的贡献。
    """

    def _build(sim_overrides: dict | None = None, **dataset_kwargs):
        sr = 16000
        duration = 1.0

        speech_dir = tmp_path / "speech"
        noise_dir = tmp_path / "noise"
        rir_dir = tmp_path / "rir"
        wind_dir = tmp_path / "wind"
        aec_dir = tmp_path / "aec"
        for d in (speech_dir, noise_dir, rir_dir, wind_dir, aec_dir):
            d.mkdir(parents=True, exist_ok=True)

        t = np.arange(int(duration * sr)) / sr
        speech = (0.4 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)
        noise = (0.001 * np.sin(2.0 * np.pi * 700.0 * t)).astype(np.float32)

        speech_wav = speech_dir / "spk1_001.wav"
        noise_wav = noise_dir / "noise_001.wav"
        rir_wav = rir_dir / "rir_001.wav"
        wind_wav = wind_dir / "wind_001.wav"
        farend_wav = aec_dir / "farend_001.wav"
        echo_wav = aec_dir / "echo_001.wav"
        soundfile.write(speech_wav, speech, sr, subtype="FLOAT")
        soundfile.write(noise_wav, noise, sr)
        _make_impulse_rir(rir_wav, sr)
        soundfile.write(wind_wav, noise, sr)
        soundfile.write(farend_wav, speech, sr, subtype="FLOAT")
        soundfile.write(echo_wav, speech, sr, subtype="FLOAT")
        stereo_wav = aec_dir / "stereo_001.wav"
        soundfile.write(stereo_wav, np.stack([speech, speech], axis=1), sr, subtype="FLOAT")

        speech_scp = tmp_path / "speech.scp"
        noise_scp = tmp_path / "noise.scp"
        rir_scp = tmp_path / "rir.scp"
        wind_scp = tmp_path / "wind.scp"
        pair_json = tmp_path / "pair.json"
        _write_scp(speech_scp, {"spk1_001": speech_wav})
        _write_scp(noise_scp, {"noise_001": noise_wav})
        _write_scp(rir_scp, {"rir_001": rir_wav})
        _write_scp(wind_scp, {"wind_001": wind_wav})
        with pair_json.open("w", encoding="utf-8") as f:
            json.dump({"pairs": [str(stereo_wav)]}, f)

        simulation_cfg = {
            "prob_target_only": 0.0,
            "prob_noise_only": 0.0,
            "prob_echo_only": 0.0,
            "prob_noise_echo_only": 0.0,
            "prob_target_noise": 1.0,
            "prob_target_echo": 0.0,
            "prob_target_noise_echo": 0.0,
            "snr_low": 120.0,
            "snr_high": 120.0,
            "prob_channel_enabled": False,
            "noisy_post_mix_degradations_enabled": False,
            "prob_reverberation": 0.0,
            "use_high_pass": False,
            "ser_range": [6.0, 6.0],
            # colored-noise 职责不约束 ERL；覆盖 physical range 以保持旧抽样。
            "erl_range": [-60.0, 60.0],
        }
        if sim_overrides:
            simulation_cfg.update(sim_overrides)

        paths = {
            "speech_scp": str(speech_scp),
            "noise_scp": str(noise_scp),
            "rir_scp": str(rir_scp),
            "windnoise_scp": str(wind_scp),
            "simulation_cfg": simulation_cfg,
            "real_farend_echo_pair_json": str(pair_json),
            "aec_ref_echo_random_delay_prob": 0.0,
            "max_duration_seconds": 1.0,
            "target_sr": sr,
            "task": "joint",
            "strict_load": True,
        }
        paths.update(dataset_kwargs)
        return paths

    return _build


def _run_one(paths):
    ds = AECDynamicMixingDataset(**paths)
    ds.set_epoch(0, base_seed=123)
    ds.export_simulation_stems = True
    sample = ds[0]
    return ds, sample


def _snr_db(ref: np.ndarray, noise: np.ndarray) -> float:
    L = min(ref.size, noise.size)
    p_ref = float(np.mean(ref[:L] ** 2))
    p_noise = float(np.mean(noise[:L] ** 2))
    return 10.0 * float(np.log10(max(p_ref, 1e-20) / max(p_noise, 1e-20)))


def _active_snr_db(ref: np.ndarray, noise: np.ndarray) -> float:
    from core.data.simulation.urgent_impl import detect_non_silence

    ref_2d = np.asarray(ref, dtype=np.float64).reshape(1, -1)
    noise_2d = np.asarray(noise, dtype=np.float64).reshape(1, -1)
    ref_active = ref_2d[:, detect_non_silence(ref_2d)]
    noise_active = noise_2d[:, detect_non_silence(noise_2d)]
    p_ref = float(np.mean(ref_active ** 2))
    p_noise = float(np.mean(noise_active ** 2))
    return 10.0 * float(np.log10(max(p_ref, 1e-20) / max(p_noise, 1e-20)))


@pytest.mark.parametrize("color", ["white", "pink", "brown", "equalized"])
def test_c3_colored_noise_all_types_use_active_power_calibration(make_paths, color):
    """四种 colored noise 均复用 ordinary mix_noise 的 active-power 标定。"""
    paths = make_paths(
        sim_overrides={
            "prob_colored_noise": 1.0,
            "colored_noise_snr_range": [10.0, 10.0],
            "colored_noise_types": [color],
        }
    )
    ds, sample = _run_one(paths)

    clean = sample[0].numpy()
    mic = sample[1].numpy()
    noise = np.asarray(ds._last_noise_additive_1d)
    L = min(clean.size, mic.size, noise.size)

    # 基线 C-1 噪声为 120 dB，可忽略；nominal 与 calibration-realized 应一致。
    assert _active_snr_db(clean[:L], noise[:L]) == pytest.approx(10.0, abs=0.15)
    rms = float(np.sqrt(np.mean(noise[:L] ** 2)))
    assert float(np.std(noise[:L])) > 0.5 * rms
    # 声学加性：mic == clean + noise stem（Layer D/E/F 均未触发）
    np.testing.assert_allclose(mic[:L], clean[:L] + noise[:L], atol=1e-6)
    # 属 Layer C 声学混合，不破坏线性分解
    assert ds._last_layer_ef_decomposable is True


def _log_psd_slope(signal: np.ndarray, sample_rate: int) -> float:
    """估计 80 Hz 至公共 7.5 kHz 频带内的 log(PSD)-log(f) 斜率。"""
    centered = np.asarray(signal, dtype=np.float64) - float(np.mean(signal))
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate)
    psd = np.abs(spectrum) ** 2
    band = (freqs >= 80.0) & (freqs <= 7500.0) & (psd > 0.0)
    return float(np.polyfit(np.log10(freqs[band]), np.log10(psd[band]), 1)[0])


@pytest.mark.parametrize(("generator_name", "expected_slope"), [("_gen_pink", -1.0), ("_gen_brown", -2.0)])
def test_colored_noise_psd_uses_physical_frequency_axis(generator_name, expected_slope):
    """Pink/Brown 在 16/32 kHz 的公共物理频带应保持目标 PSD 斜率。"""
    from core.data import post_augment

    slopes = []
    for sample_rate in (16000, 32000):
        generator = getattr(post_augment, generator_name)
        noise = generator(8 * sample_rate, sample_rate, rng=np.random.default_rng(2026))
        assert noise.shape == (8 * sample_rate,)
        assert np.isfinite(noise).all()
        assert float(np.std(noise)) > 0.01
        slopes.append(_log_psd_slope(noise, sample_rate))

    assert slopes[0] == pytest.approx(expected_slope, abs=0.12)
    assert slopes[1] == pytest.approx(expected_slope, abs=0.12)
    assert slopes[1] == pytest.approx(slopes[0], abs=0.08)


def test_colored_noise_audit_is_optional_and_rng_neutral():
    from core.data.post_augment import colored_noise

    t = np.arange(16000, dtype=np.float32) / 16000.0
    clean = (0.4 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)
    rng_plain = np.random.default_rng(77)
    rng_audit = np.random.default_rng(77)
    audit = {}

    plain = colored_noise(
        clean, 16000, 10.0, 10.0, ["pink"], rng=rng_plain
    )
    audited = colored_noise(
        clean, 16000, 10.0, 10.0, ["pink"], rng=rng_audit,
        audit_out=audit,
    )

    np.testing.assert_array_equal(audited, plain)
    assert rng_audit.random() == rng_plain.random()
    assert set(audit) == {
        "color",
        "nominal_snr_db",
        "reference_active_power",
        "noise_active_power_before_scale",
        "noise_scale",
        "calibration_realized_snr_db",
        "level_audit",
    }
    assert audit["color"] == "pink"
    assert audit["nominal_snr_db"] == pytest.approx(10.0)
    assert audit["calibration_realized_snr_db"] == pytest.approx(10.0, abs=1e-6)
    assert audit["reference_active_power"] > 0.0
    assert audit["noise_active_power_before_scale"] > 0.0
    assert audit["noise_scale"] > 0.0


def test_c3_colored_noise_skipped_on_echo_only(make_paths):
    """echo_only + prob=1.0：C-3 不施加（need_c12=False）——noise stem
    全零、mic 为纯回声。迁移后有意的语义修复。"""
    paths = make_paths(
        sim_overrides={
            "prob_target_noise": 0.0,
            "prob_echo_only": 1.0,
            "prob_colored_noise": 1.0,
        }
    )
    ds, sample = _run_one(paths)

    clean = sample[0].numpy()
    mic = sample[1].numpy()
    noise = np.asarray(ds._last_noise_additive_1d)

    assert float(np.max(np.abs(noise))) == 0.0
    assert float(np.max(np.abs(clean))) == 0.0
    # mic 纯回声（与 final_echo stem 完全一致，无近端、无噪声）
    final_echo = np.asarray(ds._last_simulation_stems["final_echo"])
    L = min(mic.size, final_echo.size)
    np.testing.assert_allclose(mic[:L], final_echo[:L], atol=1e-6)
    assert float(np.max(np.abs(mic))) > 0.01  # 回声确实存在


def test_c3_colored_noise_prob_zero_noop(make_paths):
    """prob_target_noise + prob=0.0：noise stem 仅剩 120 dB 基线噪声，
    无 colored noise 混入。"""
    paths = make_paths(sim_overrides={"prob_colored_noise": 0.0})
    ds, sample = _run_one(paths)

    clean = sample[0].numpy()
    noise = np.asarray(ds._last_noise_additive_1d)
    assert _snr_db(clean, noise) > 100.0

pytestmark = [pytest.mark.unit]
