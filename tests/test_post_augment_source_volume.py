"""Unit tests for core/data/post_augment.py::source_volume_normalize."""

from __future__ import annotations

import numpy as np
import pytest

from core.data.post_augment import _fast_active_rms, source_volume_normalize


def _sine(seconds: float = 1.0, sr: int = 16000, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def test_uniform_distribution_is_default_and_backward_compatible():
    rng = np.random.default_rng(0)
    signal = _sine()
    out, target_db = source_volume_normalize(signal, -40, 0, use_rms=True, sr=16000, rng=rng)
    assert -40 <= target_db <= 0
    assert out.dtype == np.float32
    assert out.shape == signal.shape


def test_beta_distribution_stays_within_range():
    rng = np.random.default_rng(1)
    signal = _sine()
    for _ in range(50):
        _, target_db = source_volume_normalize(
            signal, -40, 0, use_rms=True, sr=16000, rng=rng,
            distribution="beta", beta_a=2.0, beta_b=2.0,
        )
        assert -40 <= target_db <= 0


def test_beta_2_2_draws_center_near_midpoint():
    rng = np.random.default_rng(2)
    signal = _sine()
    draws = [
        source_volume_normalize(
            signal, -40, 0, use_rms=True, sr=16000, rng=rng,
            distribution="beta", beta_a=2.0, beta_b=2.0,
        )[1]
        for _ in range(2000)
    ]
    assert -23.0 < float(np.mean(draws)) < -17.0  # midpoint is -20


@pytest.mark.parametrize("target_db", [-36.0, -20.0, -6.0])
def test_fixed_target_hits_active_rms(target_db):
    signal = np.concatenate(
        [_sine(seconds=0.04, amp=0.4), np.zeros(640, dtype=np.float32)]
    )

    out, drawn_db = source_volume_normalize(
        signal,
        target_db,
        target_db,
        use_rms=True,
        sr=16000,
        rng=np.random.default_rng(3),
    )

    assert drawn_db == pytest.approx(target_db)
    assert _fast_active_rms(out, sr=16000) == pytest.approx(
        10.0 ** (target_db / 20.0), rel=2e-5, abs=2e-7
    )


@pytest.mark.parametrize("target_db", [-36.0, -20.0, -6.0])
def test_fixed_target_hits_peak(target_db):
    signal = np.array([-0.25, 0.5, -1.0, 0.75], dtype=np.float32)

    out, drawn_db = source_volume_normalize(
        signal,
        target_db,
        target_db,
        use_rms=False,
        sr=16000,
        rng=np.random.default_rng(4),
    )

    assert drawn_db == pytest.approx(target_db)
    assert float(np.max(np.abs(out))) == pytest.approx(
        10.0 ** (target_db / 20.0), rel=2e-5, abs=2e-7
    )


@pytest.mark.parametrize("forced_gain", [3.0, 2000.0])
def test_direct_helper_reuses_forced_gain_without_clipping(forced_gain):
    signal = np.array([1.0, -0.5, 0.25, 0.0], dtype=np.float32)
    calibration = {}

    out, _ = source_volume_normalize(
        signal,
        -20.0,
        -20.0,
        use_rms=False,
        rng=np.random.default_rng(5),
        forced_gain=forced_gain,
        calibration_out=calibration,
    )

    np.testing.assert_array_equal(out, signal * forced_gain)
    assert calibration["gain"] == forced_gain
    assert calibration["gain_clipped"] is False
    assert calibration["reason"] is None


def test_calibration_metadata_normal_applied_is_finite_and_json_safe():
    import json

    calibration = {}
    _, target_db = source_volume_normalize(
        _sine(),
        -24.0,
        -24.0,
        use_rms=True,
        sr=16000,
        rng=np.random.default_rng(5),
        calibration_out=calibration,
    )

    assert target_db == pytest.approx(-24.0)
    assert calibration["target_db"] == pytest.approx(-24.0)
    assert calibration["target_db_requested"] == pytest.approx(-24.0)
    assert calibration["reference_db_after"] == pytest.approx(-24.0, abs=2e-4)
    assert calibration["gain_clipped"] is False
    assert calibration["status"] == "applied"
    assert calibration["reason"] is None
    json.dumps(calibration, allow_nan=False)


@pytest.mark.parametrize(
    ("signal", "expected_status", "expected_reason"),
    [
        (np.zeros(320, dtype=np.float32), "skipped", "reference_all_zero"),
        (np.full(320, 1e-8, dtype=np.float32), "applied", "reference_near_silent"),
    ],
)
def test_zero_and_near_silent_reference_are_auditable_and_json_safe(
    signal, expected_status, expected_reason
):
    import json

    calibration = {}
    out, _ = source_volume_normalize(
        signal,
        -120.0,
        -120.0,
        use_rms=False,
        rng=np.random.default_rng(6),
        calibration_out=calibration,
    )

    assert np.isfinite(out).all()
    assert calibration["status"] == expected_status
    assert calibration["reason"] == expected_reason
    assert np.isfinite(calibration["reference_db_before"])
    assert np.isfinite(calibration["reference_db_after"])
    json.dumps(calibration, allow_nan=False)


@pytest.mark.parametrize("target_db", [1e308, -1e308])
def test_extreme_finite_target_db_is_safe_clamped_and_json_serializable(target_db):
    import json

    calibration = {}
    out, drawn_db = source_volume_normalize(
        _sine(),
        target_db,
        target_db,
        use_rms=True,
        sr=16000,
        rng=np.random.default_rng(7),
        calibration_out=calibration,
    )

    assert drawn_db == target_db
    assert np.isfinite(out).all()
    assert calibration["target_db"] == target_db
    assert calibration["target_db_requested"] == target_db
    assert calibration["gain"] in (1e-3, 1e3)
    assert calibration["gain_clipped"] is True
    assert calibration["status"] == "applied"
    assert calibration["reason"] == "gain_clamped"
    assert np.isfinite(calibration["reference_db_after"])
    json.dumps(calibration, allow_nan=False)


@pytest.mark.parametrize("distribution", ["uniform", "beta"])
def test_extreme_finite_cross_range_draw_stays_finite_and_json_safe(distribution):
    import json

    calibration = {}
    out, target_db = source_volume_normalize(
        _sine(),
        -1e308,
        1e308,
        rng=np.random.default_rng(9),
        distribution=distribution,
        beta_a=2.0,
        beta_b=3.0,
        calibration_out=calibration,
    )

    assert np.isfinite(target_db)
    assert np.isfinite(out).all()
    assert np.isfinite(calibration["target_db_requested"])
    assert calibration["gain_clipped"] is True
    json.dumps(calibration, allow_nan=False)


@pytest.mark.parametrize("target_db", [np.inf, -np.inf, np.nan])
def test_nonfinite_target_db_is_rejected_before_exponentiation(target_db):
    with pytest.raises(ValueError, match="target dB must be finite"):
        source_volume_normalize(
            _sine(),
            target_db,
            target_db,
            rng=np.random.default_rng(8),
        )


def test_rejects_unknown_distribution():
    rng = np.random.default_rng(0)
    signal = _sine()
    with pytest.raises(ValueError, match="unknown distribution"):
        source_volume_normalize(signal, -40, 0, use_rms=True, sr=16000, rng=rng, distribution="gaussian")


@pytest.mark.parametrize("field", ["beta_a", "beta_b"])
@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf, True, "2.0"])
def test_beta_distribution_rejects_invalid_shape_at_direct_api(field, invalid):
    kwargs = {"beta_a": 2.0, "beta_b": 2.0, field: invalid}
    with pytest.raises(ValueError, match=rf"{field}.*finite positive number"):
        source_volume_normalize(
            _sine(),
            -40,
            0,
            rng=np.random.default_rng(10),
            distribution="beta",
            **kwargs,
        )


pytestmark = [pytest.mark.unit]
