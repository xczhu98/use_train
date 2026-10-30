"""AEC deterministic peak guard 合同测试。

覆盖：
1. 任一相关 stem 超幅都会触发共同同比缩放；
2. 相同输入始终得到相同结果；
3. 缩放后保持 ``mic = near + noise + echo`` 分解关系；
4. 相消场景中 mic 虽低幅，超幅 echo 仍由 deterministic guard 保护。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile

from tasks.aec.data.dynamic_dataset import DynamicMixingDataset as AECDynamicMixingDataset
from core.data.dynamic_dataset_base import apply_deterministic_peak_guard
from core.data.post_augment import speaker_distortion


def _staggered_stems(mic_peak: float, echo_peak: float, n: int = 16000):
    """相消构造：near ≈ −echo + mic 残差，echo stem 与 mic 峰值独立可控，
    且 noisy = clean + noise + echo 线性关系严格成立。"""
    t = np.arange(n) / 16000.0
    echo = (echo_peak * np.cos(2.0 * np.pi * 300.0 * t)).astype(np.float32)
    noise = (0.001 * np.sin(2.0 * np.pi * 700.0 * t)).astype(np.float32)
    clean = (-echo + mic_peak * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)
    noisy = (clean + noise + echo).astype(np.float32)
    return {"noisy": noisy, "clean": clean, "noise": noise, "echo": echo}


def _all_stem_peak(stems: dict[str, np.ndarray]) -> float:
    return float(max(float(np.max(np.abs(a))) for a in stems.values()))


def test_echo_above_threshold_triggers_deterministic_guard():
    """mic 峰值较低但 echo 超幅时，deterministic guard 仍共同缩放。"""
    source = _staggered_stems(0.5, 1.5)
    peak_before = _all_stem_peak(source)
    threshold = float(np.float32(0.99))

    result = apply_deterministic_peak_guard(source)

    assert result.facts.to_metadata() == {
        "roles": ["noisy", "clean", "noise", "echo"],
        "threshold": threshold,
        "peak": pytest.approx(peak_before, rel=1e-5),
        "scale": pytest.approx(threshold / peak_before, rel=1e-4),
        "applied": True,
    }
    assert float(np.max(np.abs(result.waveforms["echo"]))) <= threshold + 1e-6
    assert np.allclose(
        result.waveforms["noisy"],
        result.waveforms["clean"]
        + result.waveforms["noise"]
        + result.waveforms["echo"],
        atol=1e-5,
    )


def test_deterministic_guard_repeats_and_preserves_decomposition():
    """同一输入重复执行 deterministic guard 得到相同共同缩放。"""
    source = _staggered_stems(1.2, 1.5)
    first = apply_deterministic_peak_guard(
        {key: value.copy() for key, value in source.items()}
    )
    second = apply_deterministic_peak_guard(
        {key: value.copy() for key, value in source.items()}
    )

    peak_before = _all_stem_peak(source)
    threshold = float(np.float32(0.99))
    assert first.facts.to_metadata() == {
        "roles": ["noisy", "clean", "noise", "echo"],
        "threshold": threshold,
        "peak": pytest.approx(peak_before, rel=1e-5),
        "scale": pytest.approx(threshold / peak_before, rel=1e-4),
        "applied": True,
    }
    assert first.facts == second.facts
    for key, waveform in first.waveforms.items():
        assert float(np.max(np.abs(waveform))) <= threshold + 1e-6, key
        np.testing.assert_array_equal(waveform, second.waveforms[key])
    assert np.allclose(
        first.waveforms["noisy"],
        first.waveforms["clean"]
        + first.waveforms["noise"]
        + first.waveforms["echo"],
        atol=1e-5,
    )


def test_deterministic_guard_below_threshold_is_identity():
    source = _staggered_stems(0.5, 0.3)
    expected = {key: value.copy() for key, value in source.items()}
    peak_before = _all_stem_peak(source)

    result = apply_deterministic_peak_guard(source)

    assert result.facts.to_metadata() == {
        "roles": ["noisy", "clean", "noise", "echo"],
        "threshold": float(np.float32(0.99)),
        "peak": pytest.approx(peak_before, rel=1e-5),
        "scale": 1.0,
        "applied": False,
    }
    for key, waveform in result.waveforms.items():
        np.testing.assert_array_equal(waveform, expected[key])


def test_layer_f_fixed_rail_pre_drive_is_observation_only():
    """fixed_rail 先 pre-drive 再裁到固定 ±1 rail，target 不参与该 API。"""
    mic = np.array([-0.6, -0.2, 0.2, 0.6], dtype=np.float32)
    target = mic.copy()
    facts = {}
    cfg = {
        "variants": ("fixed_rail",),
        "variant_probs": {"fixed_rail": 1.0},
        "pre_drive_db_min": 6.020599913279624,
        "pre_drive_db_max": 6.020599913279624,
    }

    observed = speaker_distortion(
        mic,
        16000,
        cfg,
        rng=np.random.default_rng(7),
        facts_out=facts,
    )

    np.testing.assert_allclose(observed, [-1.0, -0.4, 0.4, 1.0], atol=1e-6)
    np.testing.assert_array_equal(target, mic)
    assert facts["variant"] == "fixed_rail"
    assert facts["pre_drive_db"] == pytest.approx(6.020599913279624)
    assert facts["rail"] == 1.0
    assert facts["pre_peak"] == pytest.approx(0.6)
    assert facts["post_peak"] == 1.0
    assert facts["changed_sample_ratio"] == 1.0
    assert facts["applied"] is True


# ---------------------------------------------------------------------------
# 集成：相消场景（near ≈ −echo + 低噪），mic 低幅而 echo stem 超 0.99
# ---------------------------------------------------------------------------


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
def cancel_paths(tmp_path: Path):
    """构造 near = −echo 的相消样本：echo 峰值 1.2 > 0.99，mic ≈ 低噪。"""
    sr = 16000
    duration = 1.0

    speech_dir = tmp_path / "speech"
    noise_dir = tmp_path / "noise"
    rir_dir = tmp_path / "rir"
    wind_dir = tmp_path / "wind"
    aec_dir = tmp_path / "aec"
    for d in (speech_dir, noise_dir, rir_dir, wind_dir, aec_dir):
        d.mkdir()

    t = np.arange(int(duration * sr)) / sr
    farend = (1.2 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)
    # echo 与 farend 同内容（GCC 对齐后 delay=0）；near = −farend → mic ≈ 0
    echo = farend.copy()
    speech = (-farend).copy()
    noise = (0.001 * np.sin(2.0 * np.pi * 700.0 * t)).astype(np.float32)

    speech_wav = speech_dir / "spk1_001.wav"
    noise_wav = noise_dir / "noise_001.wav"
    rir_wav = rir_dir / "rir_001.wav"
    wind_wav = wind_dir / "wind_001.wav"
    farend_wav = aec_dir / "farend_001.wav"
    echo_wav = aec_dir / "echo_001.wav"
    soundfile.write(speech_wav, speech, sr)
    soundfile.write(noise_wav, noise, sr)
    _make_impulse_rir(rir_wav, sr)
    soundfile.write(wind_wav, noise, sr)
    soundfile.write(farend_wav, farend, sr)
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

    simulation_cfg = {
        "prob_target_only": 0.0,
        "prob_noise_only": 0.0,
        "prob_echo_only": 0.0,
        "prob_noise_echo_only": 0.0,
        "prob_target_noise": 0.0,
        "prob_target_echo": 1.0,
        "prob_target_noise_echo": 0.0,
        "snr_low": 60.0,
        "snr_high": 60.0,
        "prob_channel_enabled": False,
        "noisy_post_mix_degradations_enabled": False,
        "prob_reverberation": 0.0,
        "use_high_pass": False,
        "ser_range": [0.0, 0.0],
    }

    return {
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


def test_cancellation_scene_uses_deterministic_guard_peak(cancel_paths):
    """相消后 mic 较低，但超幅 echo 仍触发 deterministic guard 共同缩放。"""
    ds = AECDynamicMixingDataset(**cancel_paths)
    ds.set_epoch(0, base_seed=123)
    ds.export_simulation_stems = True

    _ = ds[0]

    stems = ds._last_simulation_stems
    by_layer = stems["simulation_by_layer"]
    layer_d = by_layer["D"]
    guard_keys = {"roles", "threshold", "peak", "scale", "applied"}
    assert layer_d["stage_order"] == ("D-normalize", "D-guard")
    assert set(layer_d["normalize"]) == {
        "mode",
        "target_peak",
        "scale",
        "preserves_decomposition",
    }
    mic_guard = layer_d["guard"]["mic_related"]
    assert set(mic_guard) == guard_keys
    assert mic_guard["roles"] == ["mic", "nearend", "echo", "noise"]
    assert mic_guard["applied"] is True
    farend_guard = layer_d["guard"]["farend"]
    assert set(farend_guard) == guard_keys
    assert farend_guard["roles"] == ["farend"]
    assert "D-normalize" not in by_layer
    assert "aec_context" not in by_layer
    trace_metadata = ds._last_simulation_trace.metadata
    canonical_layer_d = trace_metadata["layer_d_facts"]
    assert canonical_layer_d["stage_order"] == ["D-normalize", "D-guard"]
    assert canonical_layer_d["normalize"] == layer_d["normalize"]
    assert canonical_layer_d["guard"] == layer_d["guard"]
    json.dumps(canonical_layer_d)
    assert "layer_d_normalize" not in trace_metadata
    assert "layer_d" not in trace_metadata
    assert "farend_guard" not in trace_metadata["aec_context"]
    assert float(np.max(np.abs(stems["layer_d_echo"]))) <= 0.99 + 1e-6
    np.testing.assert_allclose(
        stems["layer_d_noisy"],
        stems["layer_d_clean"] + stems["layer_d_noise"] + stems["layer_d_echo"],
        atol=2e-5,
    )

pytestmark = [pytest.mark.unit]
