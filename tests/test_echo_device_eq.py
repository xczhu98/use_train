"""Unit tests for tasks/aec/data/echo_device_eq.py."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as scipy_signal

from tasks.aec.data.echo_device_eq import (
    apply_echo_device_eq,
    select_echo_device_eq_variant,
)
from core.data.post_augment import _design_post_eq_sos, post_eq


def _sine(seconds: float = 1.0, sr: int = 16000, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


def test_select_variant_never_triggers_at_zero_prob():
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert select_echo_device_eq_variant(rng, 0.0, {"random_eq": 1.0}) is None


def test_select_variant_always_triggers_at_prob_one():
    rng = np.random.default_rng(0)
    variant_probs = {"random_eq": 0.6, "bass_boost": 0.4}
    seen = set()
    for _ in range(200):
        variant = select_echo_device_eq_variant(rng, 1.0, variant_probs)
        assert variant in variant_probs
        seen.add(variant)
    assert seen == set(variant_probs)


def test_apply_random_eq_changes_the_signal_and_stays_float32():
    rng = np.random.default_rng(1)
    echo = _sine()
    out = apply_echo_device_eq(echo, 16000, "random_eq", (-6.0, 6.0), (500.0, 2000.0), -25.0, rng)
    assert out.dtype == np.float32
    assert out.shape == echo.shape
    assert not np.allclose(out, echo)


def test_random_eq_same_seed_is_reproducible():
    echo = _sine()
    first = post_eq(echo, 16000, -6.0, 6.0, rng=np.random.default_rng(1234))
    second = post_eq(echo, 16000, -6.0, 6.0, rng=np.random.default_rng(1234))

    np.testing.assert_array_equal(first, second)


def test_random_eq_is_causal_without_pre_ringing():
    delay = 512
    impulse = np.zeros(2048, dtype=np.float32)
    impulse[delay] = 1.0

    out = post_eq(impulse, 16000, -6.0, 6.0, rng=np.random.default_rng(9))

    assert np.count_nonzero(out[:delay]) == 0
    assert out[delay] != 0.0


@pytest.mark.parametrize("sr", [16000, 32000])
def test_random_eq_sos_is_finite_smooth_and_within_cumulative_db_budget(sr):
    db_min, db_max = -6.0, 6.0
    lower_bound, upper_bound = min(0.0, db_min), max(0.0, db_max)

    for seed in range(32):
        sos = _design_post_eq_sos(sr, db_min, db_max, rng=np.random.default_rng(seed))
        assert 1 <= sos.shape[0] <= 5
        assert sos.shape[1] == 6
        np.testing.assert_array_equal(sos[:, 3], np.ones(sos.shape[0]))
        assert np.isfinite(sos).all()
        _, poles, _ = scipy_signal.sos2zpk(sos)
        assert np.all(np.abs(poles) < 1.0)

        _, response = scipy_signal.sosfreqz(sos, worN=16384, fs=sr)
        response_db = 20.0 * np.log10(np.maximum(np.abs(response), np.finfo(float).tiny))
        assert np.isfinite(response_db).all()
        assert response_db.min() >= lower_bound - 1e-8
        assert response_db.max() <= upper_bound + 1e-8
        # Peaking biquad 的稠密频响应连续，禁止矩形频带产生离散跳变。
        assert np.max(np.abs(np.diff(response_db))) < 0.25


def test_random_eq_gain_zero_is_identity():
    sos = _design_post_eq_sos(16000, 0.0, 0.0, rng=np.random.default_rng(5))
    _, response = scipy_signal.sosfreqz(sos, worN=4096)

    np.testing.assert_allclose(response, 1.0, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    ("sr", "db_min", "db_max", "message"),
    [
        (0, -5.0, 5.0, "sample rate"),
        (199, -5.0, 5.0, "sample rate"),
        (16000, np.nan, 5.0, "finite"),
        (16000, -5.0, np.inf, "finite"),
        (16000, 5.0, -5.0, "db_min"),
    ],
)
def test_random_eq_rejects_invalid_design_inputs(sr, db_min, db_max, message):
    with pytest.raises(ValueError, match=message):
        _design_post_eq_sos(sr, db_min, db_max, rng=np.random.default_rng(0))


def test_apply_bass_boost_changes_the_signal_and_stays_float32():
    rng = np.random.default_rng(2)
    echo = _sine()
    out = apply_echo_device_eq(echo, 16000, "bass_boost", (-6.0, 6.0), (500.0, 2000.0), -25.0, rng)
    assert out.dtype == np.float32
    assert out.shape == echo.shape
    assert not np.allclose(out, echo)


def test_apply_bass_boost_preserves_lows_and_attenuates_highs():
    sr = 16000
    seconds = 2.0
    low_hz, high_hz = 100.0, 3000.0
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    low = 0.2 * np.sin(2 * np.pi * low_hz * t)
    high = 0.2 * np.sin(2 * np.pi * high_hz * t)
    echo = (low + high).astype(np.float32)

    out = apply_echo_device_eq(
        echo,
        sr,
        "bass_boost",
        (-6.0, 6.0),
        (1000.0, 1000.0),
        -20.0,
        np.random.default_rng(7),
    )

    # 忽略因果滤波器启动段，再投影到两个恰好整周期的音调。
    steady = slice(sr // 2, None)
    low_ref, high_ref = low[steady], high[steady]
    low_gain = abs(np.vdot(out[steady], low_ref)) / np.vdot(low_ref, low_ref)
    high_gain = abs(np.vdot(out[steady], high_ref)) / np.vdot(high_ref, high_ref)
    input_ratio = np.sqrt(np.mean(low_ref**2)) / np.sqrt(np.mean(high_ref**2))
    output_ratio = low_gain / high_gain * input_ratio

    assert low_gain == pytest.approx(1.0, rel=0.02)
    assert high_gain < 0.7
    assert output_ratio > input_ratio * 1.4


def test_apply_rejects_unknown_variant():
    rng = np.random.default_rng(0)
    echo = _sine()
    with pytest.raises(ValueError, match="unknown variant"):
        apply_echo_device_eq(echo, 16000, "sigmoid", (-6.0, 6.0), (500.0, 2000.0), -25.0, rng)

pytestmark = [pytest.mark.unit]
