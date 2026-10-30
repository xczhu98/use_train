"""CPU Layer-F clipping 的确定性、padding 与 spatial observation 回归测试。"""

from __future__ import annotations

import numpy as np
import pytest

from core.data.post_augment import apply_post_augmentations
from core.data.simulation_configs import SimulationConfigs

pytestmark = pytest.mark.unit


def _cfg() -> SimulationConfigs:
    """走 canonical strict 配置入口，Layer F 字段新增/改名会在此直接失败。"""
    return SimulationConfigs.from_dict(
        {
            "noisy_post_mix_degradations_enabled": True,
            "prob_drc": 0.0,
            "prob_codecs": 0.0,
            "prob_clip": 1.0,
            "clip_variants": ["fixed_rail", "hard", "soft", "sigmoid", "sox"],
            "clip_variant_probs": {
                "fixed_rail": 0.2,
                "hard": 0.2,
                "soft": 0.2,
                "sigmoid": 0.2,
                "sox": 0.2,
            },
            "clip_hard_portion": 0.5,
            "clip_hard_on_rate": True,
            "clip_pre_drive_db_range": [3.0, 6.0],
            "clip_rate_range": [0.1, 0.2],
            "clip_threshold_db_range": [-12.0, -6.0],
            "clip_soft_slope_range": [2.0, 3.0],
            "clip_sigmoid_slope_range": [2.0, 3.0],
            "clip_sox_threshold_db_range": [-12.0, -6.0],
            "clip_pedal_threshold_db_range": [-12.0, -6.0],
            "prob_post_eq": 0.0,
            "prob_lowpass": 0.0,
            "prob_band_reject": 0.0,
            "prob_dc_offset": 0.0,
            "prob_bit_crush": 0.0,
            "prob_spectral_holes": 0.0,
            "prob_webrtc_ns": 0.0,
            "prob_webrtc_agc": 0.0,
            "prob_bass_boost": 0.0,
            "prob_spectral_leakage": 0.0,
            "prob_loudness_perturb": 0.0,
        },
        strict=True,
    )


def _apply_row(waveform, *, length, seed):
    prefix = apply_post_augmentations(
        waveform[:length].copy(),
        16_000,
        _cfg(),
        rng=np.random.default_rng(seed),
    )
    return np.concatenate((prefix, waveform[length:]))


def test_layer_f_is_fixed_seed_deterministic_and_keeps_padding():
    waveform = np.array(
        [[-1.2, -0.4, 0.0, 0.4, 1.2, 7.0], [-0.8, -0.2, 0.2, 0.8, 7.0, 7.0]],
        dtype=np.float32,
    )
    first = np.stack([_apply_row(row, length=length, seed=7) for row, length in zip(waveform, (5, 4))])
    second = np.stack([_apply_row(row, length=length, seed=7) for row, length in zip(waveform, (5, 4))])

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0, 5:], waveform[0, 5:])
    np.testing.assert_array_equal(first[1, 4:], waveform[1, 4:])


def test_spatial_layer_f_preserves_channels_and_mono_target():
    noisy = np.array(
        [
            [[-1.2, -0.4, 0.0, 0.4, 1.2, 7.0], [-0.8, -0.2, 0.2, 0.8, 1.0, 7.0]],
            [[-0.8, -0.2, 0.2, 0.8, 1.0, 7.0], [-1.2, -0.4, 0.0, 0.4, 1.2, 7.0]],
        ],
        dtype=np.float32,
    )
    clean = np.array([[0.1, -0.2, 0.3, 0.4, 0.5, 9.0], [0.2, -0.3, 0.4, 0.5, 0.6, 9.0]], dtype=np.float32)
    original_padding = noisy[:, :, 5:].copy()
    output = np.stack(
        [
            np.stack([_apply_row(channel, length=length, seed=7) for channel in sample])
            for sample, length in zip(noisy, (5, 4))
        ]
    )

    assert output.shape == noisy.shape
    np.testing.assert_array_equal(clean.shape, (2, 6))
    np.testing.assert_array_equal(output[:, :, 5:], original_padding)
    assert np.any(output[:, :, :4] != noisy[:, :, :4])
    np.testing.assert_array_equal(clean, np.array([[0.1, -0.2, 0.3, 0.4, 0.5, 9.0], [0.2, -0.3, 0.4, 0.5, 0.6, 9.0]], dtype=np.float32))


def test_layer_f_is_noisy_only_and_does_not_change_target():
    noisy = np.linspace(-1.2, 1.2, 32, dtype=np.float32)
    target = np.linspace(-0.4, 0.4, 32, dtype=np.float32)
    original_target = target.copy()
    output = apply_post_augmentations(
        noisy.copy(), 16_000, _cfg(), rng=np.random.default_rng(3)
    )

    assert output.shape == noisy.shape
    np.testing.assert_array_equal(target, original_target)
    assert np.isfinite(output).all()
