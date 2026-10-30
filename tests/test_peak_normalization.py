"""PeakNormalizationResult 纯 Interface 的数学与 RNG 合同测试。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.data.dynamic_dataset_base import BaseDynamicMixingDataset
from core.data.peak_normalization import (
    PeakNormalizationResult,
    apply_joint_role_peak_normalization,
    apply_peak_normalization,
)


pytestmark = [pytest.mark.unit]


class _CountingRng:
    def __init__(self, value: float = 0.75) -> None:
        self.value = value
        self.calls: list[tuple[float, float]] = []

    def uniform(self, lo: float, hi: float) -> float:
        self.calls.append((lo, hi))
        return self.value


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = np.array([0.25, -0.5, 0.0], dtype=np.float32)
    noise = np.array([0.25, 0.0, 0.0], dtype=np.float32)
    noisy = (clean + noise).astype(np.float32)
    return noisy, clean, noise


def test_generic_joint_roles_use_anchors_only_and_freeze_json_safe_facts():
    rng = _CountingRng(0.5)
    result = apply_joint_role_peak_normalization(
        {
            "mic": np.array([2.0, -2.0], dtype=np.float64),
            "nearend": np.array([1.0, -1.0], dtype=np.float64),
            "farend": np.array([100.0, -100.0], dtype=np.float64),
        },
        anchor_roles=("mic",),
        mode="joint",
        scale_range=(0.5, 0.5),
        rng=rng,
    )

    assert rng.calls == [(0.5, 0.5)]
    assert result.scale_facts == {
        "mode": "joint",
        "target_peak": 0.5,
        "scale": 0.25,
        "anchor_roles": ("mic",),
        "anchor_maxima": (2.0,),
        "shared_max": 2.0,
    }
    with pytest.raises(TypeError, match="immutable"):
        result.scale_facts["scale"] = 1.0
    with pytest.raises(TypeError):
        result.waveforms["mic"] = np.zeros(2, dtype=np.float32)
    assert json.loads(json.dumps(result.scale_facts)) == {
        "mode": "joint",
        "target_peak": 0.5,
        "scale": 0.25,
        "anchor_roles": ["mic"],
        "anchor_maxima": [2.0],
        "shared_max": 2.0,
    }
    for waveform in result.waveforms.values():
        assert waveform.dtype == np.float32
    np.testing.assert_array_equal(result.waveforms["mic"], [0.5, -0.5])
    np.testing.assert_array_equal(result.waveforms["nearend"], [0.25, -0.25])
    np.testing.assert_array_equal(result.waveforms["farend"], [25.0, -25.0])


def test_generic_joint_off_is_zero_draw_and_identity_scale():
    rng = _CountingRng()
    waveform = np.array([2.0, -2.0], dtype=np.float64)

    result = apply_joint_role_peak_normalization(
        {"mic": waveform, "farend": waveform * 10.0},
        anchor_roles=("mic",),
        mode="off",
        scale_range=(0.5, 0.95),
        rng=rng,
    )

    assert rng.calls == []
    assert result.scale_facts["target_peak"] is None
    assert result.scale_facts["scale"] == 1.0
    for actual, expected in zip(result.waveforms.values(), (waveform, waveform * 10.0)):
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, expected)


def test_off_is_zero_draw_float32_passthrough():
    noisy, clean, noise = _inputs()
    rng = _CountingRng()

    result = apply_peak_normalization(
        noisy.astype(np.float64),
        clean.astype(np.float64),
        noise.astype(np.float64),
        mode="off",
        scale_range=(0.5, 0.95),
        rng=rng,
    )

    assert isinstance(result, PeakNormalizationResult)
    assert rng.calls == []
    assert result.mode == "off"
    assert result.target_peak is None
    assert result.noisy_scale == result.clean_scale == result.noise_scale == 1.0
    assert result.preserves_decomposition is True
    for actual, expected in zip(
        (result.noisy, result.clean, result.noise), (noisy, clean, noise)
    ):
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, expected)


def test_joint_draws_once_and_exposes_common_scale():
    noisy, clean, noise = _inputs()
    rng = _CountingRng(0.75)

    result = apply_peak_normalization(
        noisy,
        clean,
        noise,
        mode="joint",
        scale_range=(0.5, 0.95),
        rng=rng,
    )

    assert rng.calls == [(0.5, 0.95)]
    assert result.target_peak == 0.75
    assert result.noisy_scale == result.clean_scale == result.noise_scale == 1.5
    assert result.preserves_decomposition is True
    np.testing.assert_array_equal(result.noisy, noisy * 1.5)
    np.testing.assert_array_equal(result.clean, clean * 1.5)
    np.testing.assert_array_equal(result.noise, noise * 1.5)
    np.testing.assert_array_equal(result.noisy, result.clean + result.noise)






@pytest.mark.parametrize("mode", ["joint"])
def test_silence_uses_identity_scales(mode: str):
    silence = np.zeros(4, dtype=np.float32)
    rng = _CountingRng(0.75)

    result = apply_peak_normalization(
        silence,
        silence,
        silence,
        mode=mode,
        scale_range=(0.5, 0.95),
        rng=rng,
    )

    assert rng.calls == [(0.5, 0.95)]
    assert result.noisy_scale == result.clean_scale == result.noise_scale == 1.0
    np.testing.assert_array_equal(result.noisy, silence)
    np.testing.assert_array_equal(result.clean, silence)
    np.testing.assert_array_equal(result.noise, silence)


def test_joint_tiny_nonzero_keeps_legacy_epsilon_scale():
    tiny = np.array([1e-13], dtype=np.float32)
    silence = np.zeros(1, dtype=np.float32)

    result = apply_peak_normalization(
        tiny,
        silence,
        silence,
        mode="joint",
        scale_range=(0.75, 0.75),
        rng=_CountingRng(0.75),
    )

    assert result.noisy_scale == pytest.approx(0.75 / 1e-12)
    np.testing.assert_allclose(result.noisy, np.array([0.075], dtype=np.float32))


def test_unknown_mode_fails_before_rng_draw():
    noisy, clean, noise = _inputs()
    rng = _CountingRng()

    with pytest.raises(ValueError, match="peak normalization mode"):
        apply_peak_normalization(
            noisy,
            clean,
            noise,
            mode="unknown",
            scale_range=(0.5, 0.95),
            rng=rng,
        )

    assert rng.calls == []


def test_base_dataset_legacy_four_tuple_adapter_is_removed():
    """旧四元组投影 Interface 已移除，Layer D 只保留结构化 result 入口。"""
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)

    assert not hasattr(dataset, "_apply_peak_normalization")
    assert hasattr(dataset, "_apply_peak_normalization_result")


@pytest.mark.parametrize("mode", ["separate", "single"])
def test_shared_layer_d_rejects_pase_specific_modes(mode):
    """独立归一化属于 PASE recipe，公共 Layer D 不得接受。"""
    noisy, clean, noise = _inputs()
    with pytest.raises(ValueError, match="off.*joint"):
        apply_peak_normalization(noisy, clean, noise, mode=mode,
                                 scale_range=(0.5, 0.95), rng=_CountingRng(0.75))
