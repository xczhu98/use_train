"""CPU Layer-F clipping RNG 与低采样率频率边界回归测试。"""

from __future__ import annotations

import numpy as np
import pytest

from core.data.post_augment import band_reject, lowpass_filter, speaker_distortion

pytestmark = pytest.mark.unit


def _clip_cfg() -> dict:
    return {
        "variants": ("fixed_rail", "hard", "soft", "sigmoid", "sox"),
        "variant_probs": {
            "fixed_rail": 0.2,
            "hard": 0.2,
            "soft": 0.2,
            "sigmoid": 0.2,
            "sox": 0.2,
        },
        "pre_drive_db_min": 3.0,
        "pre_drive_db_max": 6.0,
        "hard_clip_portion": 0.5,
        "hard_clip_on_rate": True,
        "clip_rate_min": 0.1,
        "clip_rate_max": 0.2,
        "threshold_db_min": -12.0,
        "threshold_db_max": -6.0,
        "soft_slope_min": 2.0,
        "soft_slope_max": 3.0,
        "sigmoid_slope_min": 2.0,
        "sigmoid_slope_max": 3.0,
        "sox_threshold_db_min": -12.0,
        "sox_threshold_db_max": -6.0,
    }


def test_clipping_is_fixed_seed_deterministic_and_reports_observation_facts():
    waveform = np.linspace(-1.5, 1.5, 32 * 17, dtype=np.float32)
    first_facts, second_facts = {}, {}
    first = speaker_distortion(
        waveform, 16_000, _clip_cfg(), rng=np.random.default_rng(137), facts_out=first_facts
    )
    second = speaker_distortion(
        waveform, 16_000, _clip_cfg(), rng=np.random.default_rng(137), facts_out=second_facts
    )

    np.testing.assert_array_equal(first, second)
    assert first_facts == second_facts
    assert first_facts["variant"] in _clip_cfg()["variants"]
    assert first_facts["changed_sample_count"] >= 0
    assert np.isfinite(first).all()


@pytest.mark.parametrize(
    ("operation", "configured_range"),
    [
        ("lowpass", (1100.0, 1500.0)),
        ("band_reject", (1100.0, 1500.0)),
    ],
)
def test_spectral_frequency_range_closes_at_low_sample_rate(operation, configured_range):
    samples = np.arange(2048, dtype=np.float32)
    waveform = np.cos(2 * np.pi * 450.0 * samples / 2000.0).astype(np.float32)
    rng = np.random.default_rng(0)

    if operation == "lowpass":
        output = lowpass_filter(waveform, 2_000, *configured_range, rng=rng)
    else:
        output = band_reject(
            waveform,
            2_000,
            *configured_range,
            100.0,
            100.0,
            rng=rng,
        )

    assert output.shape == waveform.shape
    assert np.isfinite(output).all()
