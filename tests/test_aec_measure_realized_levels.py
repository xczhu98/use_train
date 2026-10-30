"""Regression tests for ``tasks.aec.data.dynamic_dataset._measure_realized_levels``."""

import numpy as np
import pytest

from tasks.aec.data.dynamic_dataset import _measure_realized_levels


def _sine(sr: int, duration: float, freq: float, amplitude: float) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return amplitude * np.sin(2.0 * np.pi * freq * t)


def test_measure_realized_levels_returns_finite_values_with_echo():
    sr = 16000
    clean = _sine(sr, 1.0, 200.0, 0.1)
    echo = _sine(sr, 1.0, 200.0, 0.05)
    farend = _sine(sr, 1.0, 200.0, 0.2)
    noise = _sine(sr, 1.0, 300.0, 0.01)
    noisy = clean + echo + noise

    out = _measure_realized_levels(
        clean=clean,
        noisy=noisy,
        echo=echo,
        farend=farend,
        noise=noise,
        measure_point="final",
    )

    assert np.isfinite(out["clean_dbfs_rms"])
    assert np.isfinite(out["mic_dbfs_rms"])
    assert np.isfinite(out["noise_dbfs_rms"])
    assert np.isfinite(out["echo_dbfs_rms"])
    assert np.isfinite(out["far_dbfs_rms"])
    assert np.isfinite(out["ser_db"])
    assert np.isfinite(out["erl_db"])
    assert 0.0 <= out["mic_clip_frame_ratio"] <= 1.0
    assert 0.0 <= out["mic_near_fullscale_ratio"] <= 1.0
    assert out["measure_point"] == "final"


def test_measure_realized_levels_handles_zero_echo_as_none():
    sr = 16000
    clean = _sine(sr, 1.0, 200.0, 0.1)
    echo = np.zeros(sr)
    farend = _sine(sr, 1.0, 200.0, 0.2)
    noise = _sine(sr, 1.0, 300.0, 0.01)
    noisy = clean + noise

    out = _measure_realized_levels(
        clean=clean,
        noisy=noisy,
        echo=echo,
        farend=farend,
        noise=noise,
    )

    assert out["echo_dbfs_rms"] is None
    assert out["far_dbfs_rms"] is None
    assert out["ser_db"] is None
    assert out["erl_db"] is None


def test_measure_realized_levels_detects_layer_d_guard_platform():
    """A 0.99 plateau from the Layer D deterministic guard counts as clipped."""
    sr = 16000
    clean = np.zeros(sr, dtype=np.float32)
    # 0.99 is the Layer D deterministic peak guard threshold in float32.
    noisy = np.full(sr, 0.99, dtype=np.float32)
    out = _measure_realized_levels(
        clean=clean,
        noisy=noisy,
        echo=None,
        farend=None,
        noise=None,
    )

    assert out["mic_clip_frame_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert out["mic_peak_abs"] == pytest.approx(0.99, abs=1e-6)


def test_measure_realized_levels_empty_inputs():
    out = _measure_realized_levels(
        clean=np.array([], dtype=np.float32),
        noisy=np.array([], dtype=np.float32),
        echo=None,
        farend=None,
        noise=None,
    )

    assert out["mic_dbfs_rms"] == pytest.approx(-120.0)
    assert out["mic_clip_frame_ratio"] == 0.0
    assert out["mic_peak_abs"] == 0.0


def test_pure_echo_ser_is_na_but_erl_remains_available():
    sr = 16000
    echo = _sine(sr, 1.0, 200.0, 0.05)
    farend = _sine(sr, 1.0, 200.0, 0.2)
    out = _measure_realized_levels(
        clean=np.zeros(sr, dtype=np.float32),
        noisy=echo,
        echo=echo,
        farend=farend,
        noise=np.zeros(sr, dtype=np.float32),
        branch="echo_only",
        decomposable=True,
        reconstruction_reference=echo,
    )

    assert out["ser_db"] is None
    assert out["ser_applicable"] is False
    assert out["ser_reason_code"] == "scenario_has_no_nearend"
    assert out["erl_db"] == pytest.approx(12.0412, abs=1e-3)
    assert out["erl_applicable"] is True
    assert out["reconstruction_residual"]["rms"] == pytest.approx(0.0, abs=1e-8)


def test_realized_snr_and_fail_closed_reasons_are_json_safe():
    import json

    sr = 16000
    clean = _sine(sr, 1.0, 200.0, 0.1)
    echo = _sine(sr, 1.0, 400.0, 0.05)
    farend = _sine(sr, 1.0, 400.0, 0.2)
    noise = _sine(sr, 1.0, 700.0, 0.01)
    reconstruction = clean + echo + noise
    out = _measure_realized_levels(
        clean=clean,
        noisy=reconstruction,
        echo=echo,
        farend=farend,
        noise=noise,
        branch="prob_target_noise_echo",
        decomposable=True,
        decomposable_reason_codes=[],
        reconstruction_reference=reconstruction,
    )
    assert out["snr_db"] == pytest.approx(20.0, abs=1e-6)
    assert out["ser_db"] == pytest.approx(6.0206, abs=1e-3)
    assert out["erl_db"] == pytest.approx(12.0412, abs=1e-3)
    assert out["snr_applicable"] is True
    assert out["ser_applicable"] is True
    assert out["erl_applicable"] is True
    audits = out["level_audits"]
    assert set(audits) == {"snr", "ser", "erl"}
    assert audits["snr"]["numerator"] == "training_target"
    assert audits["snr"]["denominator"] == "noise_stem"
    assert audits["ser"]["denominator"] == "echo_stem"
    assert audits["erl"]["numerator"] == "farend_stem"
    assert audits["erl"]["denominator"] == "echo_stem"
    for key, expected in (("snr", 20.0), ("ser", 6.0206), ("erl", 12.0412)):
        assert audits[key]["policy"] == "full_mean_power"
        assert audits[key]["final_realized_db"] == pytest.approx(expected, abs=1e-3)
        assert audits[key]["final_measure_point"] == "post_dataset_post_mix"
        assert audits[key]["applicable"] is True
        assert audits[key]["reason"] is None
    assert out["reconstruction_residual"]["peak_abs"] == pytest.approx(0.0, abs=1e-8)
    json.dumps(out, allow_nan=False)

    blocked = _measure_realized_levels(
        clean=clean,
        noisy=reconstruction,
        echo=echo,
        farend=farend,
        noise=noise,
        branch="prob_target_noise_echo",
        decomposable=False,
        decomposable_reason_codes=["layer_f_nonlinear"],
        reconstruction_reference=reconstruction,
    )
    assert blocked["ser_db"] is None
    assert blocked["snr_db"] is None
    assert blocked["erl_db"] is not None
    assert blocked["decomposable_reason_codes"] == ["layer_f_nonlinear"]
    assert blocked["reconstruction_residual"]["reason_code"] == "mic_not_decomposable"
    assert blocked["level_audits"]["snr"]["final_realized_db"] is None
    assert blocked["level_audits"]["snr"]["applicable"] is False
    assert blocked["level_audits"]["snr"]["reason"] == "mic_not_decomposable"
    assert blocked["level_audits"]["ser"]["final_realized_db"] is None
    assert blocked["level_audits"]["ser"]["reason"] == "mic_not_decomposable"
    assert blocked["level_audits"]["erl"]["applicable"] is True
    json.dumps(blocked, allow_nan=False)


def test_aec_task_ratios_remove_passthrough_noise_from_target():
    sr = 16000
    nearend = _sine(sr, 1.0, 200.0, 0.1)
    echo = _sine(sr, 1.0, 400.0, 0.05)
    farend = _sine(sr, 1.0, 400.0, 0.2)
    noise = _sine(sr, 1.0, 700.0, 0.1)
    target = nearend + noise
    mic = target + echo

    out = _measure_realized_levels(
        clean=target,
        noisy=mic,
        echo=echo,
        farend=farend,
        noise=noise,
        branch="prob_target_noise_echo",
        task="aec",
        decomposable=True,
        reconstruction_reference=mic,
    )

    assert out["ratio_reference"] == "nearend_component"
    assert out["ser_db"] == pytest.approx(6.0206, abs=1e-3)
    assert out["snr_db"] == pytest.approx(0.0, abs=1e-6)


pytestmark = [pytest.mark.unit]
