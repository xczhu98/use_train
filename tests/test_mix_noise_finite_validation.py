"""mix_noise 非有限数值边界回归测试。"""

from __future__ import annotations

import numpy as np
import pytest

from core.data.simulation.urgent_impl import mix_noise


pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize("snr", [np.nan, np.inf, -np.inf])
def test_mix_noise_rejects_nonfinite_snr_before_empty_return(snr):
    empty = np.empty((1, 0), dtype=np.float32)

    with pytest.raises(ValueError, match="snr must be finite"):
        mix_noise(empty, empty, snr=snr)


@pytest.mark.parametrize("bad_input", ["speech", "noise", "reference"])
def test_mix_noise_rejects_nonfinite_input_samples(bad_input):
    speech = np.ones((1, 8), dtype=np.float32)
    noise = np.ones((1, 8), dtype=np.float32)
    reference = np.ones((1, 8), dtype=np.float32)
    target = {"speech": speech, "noise": noise, "reference": reference}[bad_input]
    target[0, 0] = np.nan

    with pytest.raises(ValueError, match="samples must be finite"):
        mix_noise(speech, noise, snr=0.0, snr_reference=reference)


def test_mix_noise_uses_float64_power_accumulator_for_large_float32_inputs():
    speech = np.full((1, 8), 1e20, dtype=np.float32)
    noise = np.full((1, 8), 1e20, dtype=np.float32)

    noisy, scaled_noise = mix_noise(speech, noise, snr=0.0)

    assert np.all(np.isfinite(noisy))
    assert np.all(np.isfinite(scaled_noise))
    np.testing.assert_allclose(scaled_noise, noise, rtol=1e-6, atol=0.0)


def test_mix_noise_finite_output_and_rng_contract_are_stable():
    speech = np.array([[0.5, -0.25, 0.125, -0.0625]], dtype=np.float32)
    noise = np.array([[0.1, -0.2, 0.3, -0.4]], dtype=np.float32)
    rng = np.random.default_rng(123)
    state_before = rng.bit_generator.state
    speech_power = np.mean(np.square(speech.astype(np.float64)), dtype=np.float64)
    noise_power = np.mean(np.square(noise.astype(np.float64)), dtype=np.float64)
    scale = 10.0 ** (-6.0 / 20.0) * np.sqrt(speech_power / noise_power)
    expected_noise = scale * noise

    noisy, scaled_noise = mix_noise(speech, noise, snr=6.0, rng=rng)

    np.testing.assert_allclose(scaled_noise, expected_noise, rtol=1e-7, atol=0.0)
    np.testing.assert_allclose(noisy, speech + expected_noise, rtol=1e-7, atol=0.0)
    assert rng.bit_generator.state == state_before


def test_mix_noise_audit_is_json_safe_and_waveform_neutral():
    import json

    speech = np.array([[0.5, -0.25, 0.125, -0.0625]], dtype=np.float32)
    noise = np.array([[0.1, -0.2, 0.3, -0.4]], dtype=np.float32)
    rng_plain = np.random.default_rng(123)
    rng_audit = np.random.default_rng(123)
    audit = {}

    plain = mix_noise(speech, noise, snr=6.0, rng=rng_plain)
    audited = mix_noise(speech, noise, snr=6.0, rng=rng_audit, audit_out=audit)

    for expected, actual in zip(plain, audited):
        np.testing.assert_array_equal(actual, expected)
    assert rng_audit.bit_generator.state == rng_plain.bit_generator.state
    level = audit["level_audit"]
    assert level["candidate_drawn_db"] == pytest.approx(6.0)
    assert level["selected_nominal_db"] == pytest.approx(6.0)
    assert level["policy"] == "active_mean_power"
    assert level["numerator"] == "snr_reference_active"
    assert level["denominator"] == "scaled_additive_noise_active"
    assert level["final_realized_db"] is None
    assert level["applicable"] is False
    assert level["reason"] == "final_not_measured"
    json.dumps(audit, allow_nan=False)


def test_mix_noise_rejects_nonfinite_active_power():
    speech = np.full((1, 2), np.finfo(np.float64).max, dtype=np.float64)
    noise = np.ones((1, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="active power must be finite"):
        mix_noise(speech, noise, snr=0.0)


def test_mix_noise_rejects_nonfinite_scale_before_clamp():
    speech = np.ones((1, 8), dtype=np.float32)
    noise = np.ones((1, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="scale must be finite"):
        mix_noise(speech, noise, snr=-1e308)
