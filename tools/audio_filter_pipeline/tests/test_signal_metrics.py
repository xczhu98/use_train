import json
import math
from dataclasses import FrozenInstanceError, asdict, fields


import numpy as np
import pytest

import audio_filter_pipeline.processors.signal_metrics as signal_metrics_module
from audio_filter_pipeline.processors.signal_metrics import (
    SignalMetrics,
    compute_signal_metrics,
)

pytestmark = pytest.mark.unit

_OPTIONAL_FACTS = (
    "zero_ratio",
    "peak",
    "clipping_ratio",
    "dc_offset",
    "rms_dbfs",
)


def test_signal_metrics_is_frozen_and_has_exact_fields():
    assert [field.name for field in fields(SignalMetrics)] == [
        "finite",
        "zero_ratio",
        "peak",
        "clipping_ratio",
        "dc_offset",
        "rms_dbfs",
        "num_samples",
    ]

    metrics = SignalMetrics(True, 1.0, 0.0, 0.0, 0.0, None, 0)
    with pytest.raises(FrozenInstanceError):
        metrics.peak = 1.0


def test_empty_input_returns_valid_empty_facts():
    assert compute_signal_metrics(np.array([], dtype=np.float32)) == SignalMetrics(
        finite=True,
        zero_ratio=1.0,
        peak=0.0,
        clipping_ratio=0.0,
        dc_offset=0.0,
        rms_dbfs=None,
        num_samples=0,
    )


def test_all_zero_input_has_no_finite_rms_dbfs():
    assert compute_signal_metrics(np.zeros(4, dtype=np.float64)) == SignalMetrics(
        finite=True,
        zero_ratio=1.0,
        peak=0.0,
        clipping_ratio=0.0,
        dc_offset=0.0,
        rms_dbfs=None,
        num_samples=4,
    )


def test_hand_derived_vector_values():
    metrics = compute_signal_metrics(
        np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
    )

    assert metrics.finite is True
    assert metrics.num_samples == 5
    assert type(metrics.num_samples) is int
    assert metrics.zero_ratio == pytest.approx(0.2)
    assert metrics.peak == pytest.approx(1.0)
    assert metrics.clipping_ratio == pytest.approx(0.4)
    assert metrics.dc_offset == pytest.approx(0.0)
    assert metrics.rms_dbfs == pytest.approx(-3.010299956639812)


def test_clipping_boundary_is_inclusive():
    just_below = np.nextafter(np.float64(1.0), np.float64(0.0))
    metrics = compute_signal_metrics(np.array([-1.0, just_below, 1.0]))

    assert metrics.clipping_ratio == pytest.approx(2.0 / 3.0)


def test_multidimensional_input_is_flattened_across_all_elements():
    audio = np.array([[0.0, 1.0], [-1.0, 0.5]], dtype=np.float64)

    metrics = compute_signal_metrics(audio)

    assert metrics.num_samples == 4
    assert metrics.zero_ratio == pytest.approx(0.25)
    assert metrics.peak == pytest.approx(1.0)
    assert metrics.clipping_ratio == pytest.approx(0.5)
    assert metrics.dc_offset == pytest.approx(0.125)
    assert metrics.rms_dbfs == pytest.approx(-2.498774732165999)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_supported_numeric_dtypes_are_measured_in_float64(dtype):
    metrics = compute_signal_metrics(np.array([-1, 0, 1], dtype=dtype))

    assert metrics == SignalMetrics(
        finite=True,
        zero_ratio=pytest.approx(1.0 / 3.0),
        peak=1.0,
        clipping_ratio=pytest.approx(2.0 / 3.0),
        dc_offset=0.0,
        rms_dbfs=pytest.approx(-1.760912590556813),
        num_samples=3,
    )


def test_large_signed_integer_cancellation_preserves_dc_residual():
    peak = 2**62
    metrics = compute_signal_metrics(
        np.array([peak, 1, -peak], dtype=np.int64)
    )

    assert metrics.peak == float(peak)
    assert metrics.dc_offset == pytest.approx(1.0 / 3.0)
    assert metrics.rms_dbfs == pytest.approx(
        20.0 * math.log10(float(peak) * math.sqrt(2.0 / 3.0))
    )
    assert math.isfinite(metrics.rms_dbfs)


def test_max_float_rms_avoids_square_overflow_and_is_json_safe():
    max_float = np.finfo(np.float64).max
    metrics = compute_signal_metrics(
        np.array([max_float, max_float], dtype=np.float64)
    )
    payload = asdict(metrics)

    assert metrics.peak == max_float
    assert metrics.dc_offset == max_float
    assert metrics.rms_dbfs == pytest.approx(20.0 * math.log10(max_float))
    assert all(
        value is None or not isinstance(value, float) or math.isfinite(value)
        for value in payload.values()
    )
    json.dumps(payload, allow_nan=False)


def test_large_float_cancellation_preserves_small_dc_residual():
    max_float = np.finfo(np.float64).max
    residual = 1e292
    metrics = compute_signal_metrics(
        np.array([max_float, residual, -max_float], dtype=np.float64)
    )

    assert metrics.dc_offset == pytest.approx(residual / 3.0)
    assert math.isfinite(metrics.dc_offset)
    assert math.isfinite(metrics.rms_dbfs)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_suppresses_every_numeric_fact(bad_value):
    metrics = compute_signal_metrics(np.array([0.0, bad_value, 1.0]))

    assert metrics.finite is False
    assert metrics.num_samples == 3
    assert all(getattr(metrics, name) is None for name in _OPTIONAL_FACTS)


@pytest.mark.parametrize(
    "audio",
    [
        [0.0, 1.0],
        "audio",
        np.array([True, False]),
        np.array([1.0 + 2.0j]),
        np.array([1, "2"], dtype=object),
        np.array(["1", "2"]),
    ],
)
def test_unsupported_inputs_and_dtypes_are_rejected(audio):
    with pytest.raises((TypeError, ValueError), match="array|dtype|numeric|real|bool|complex"):
        compute_signal_metrics(audio)


def test_output_is_canonical_json_safe_python_data():
    metrics = compute_signal_metrics(np.array([-1, 0, 1], dtype=np.int16))
    payload = asdict(metrics)

    assert type(payload["finite"]) is bool
    assert type(payload["num_samples"]) is int
    assert all(
        value is None or type(value) is float
        for name, value in payload.items()
        if name not in {"finite", "num_samples"}
    )
    json.dumps(payload, allow_nan=False)


def test_input_is_not_mutated():
    audio = np.array([[0.0, -1.0], [0.25, 1.0]], dtype=np.float32)
    original = audio.copy()

    compute_signal_metrics(audio)

    np.testing.assert_array_equal(audio, original)


@pytest.mark.parametrize("num_samples", [17, 262_145, 786_449])
def test_reductions_use_fixed_size_blocks_for_different_lengths(monkeypatch, num_samples):
    block_limit = signal_metrics_module._SIGNAL_METRICS_BLOCK_SAMPLES
    audio = np.linspace(-1.0, 1.0, num_samples, dtype=np.float32)
    observed_sizes = []
    real_reduce_block = signal_metrics_module._reduce_block

    def recording_reduce_block(block, *args, **kwargs):
        observed_sizes.append(block.size)
        return real_reduce_block(block, *args, **kwargs)

    monkeypatch.setattr(signal_metrics_module, "_reduce_block", recording_reduce_block)

    metrics = compute_signal_metrics(audio)

    assert metrics.num_samples == audio.size
    assert max(observed_sizes) <= block_limit
    assert len(observed_sizes) == 2 * math.ceil(num_samples / block_limit)


def test_extreme_cancellation_is_preserved_across_block_boundaries():
    block_limit = signal_metrics_module._SIGNAL_METRICS_BLOCK_SAMPLES
    peak = np.finfo(np.float64).max
    audio = np.zeros(block_limit + 2, dtype=np.float64)
    audio[block_limit - 1 :] = [peak, 1e292, -peak]

    metrics = compute_signal_metrics(audio)

    assert metrics.dc_offset == pytest.approx(1e292 / audio.size)
    assert math.isfinite(metrics.rms_dbfs)


@pytest.mark.parametrize("crosses_block", [False, True], ids=["single-block", "cross-block"])
@pytest.mark.parametrize("order", ["grouped", "alternating"])
def test_float64_nextafter_cancellation_matches_math_fsum(crosses_block, order):
    block_limit = signal_metrics_module._SIGNAL_METRICS_BLOCK_SAMPLES
    pair_count = block_limit if crosses_block else block_limit // 4
    positive = np.float64(1.0)
    negative = np.nextafter(positive, np.float64(0.0))
    audio = np.empty(pair_count * 2, dtype=np.float64)
    if order == "grouped":
        audio[:pair_count] = positive
        audio[pair_count:] = -negative
    else:
        audio[0::2] = positive
        audio[1::2] = -negative
    expected = math.fsum(map(float, audio)) / audio.size

    metrics = compute_signal_metrics(audio)

    assert metrics.dc_offset == expected


def test_superaccumulator_state_is_fixed_and_normalized_across_blocks(monkeypatch):
    block_limit = signal_metrics_module._SIGNAL_METRICS_BLOCK_SAMPLES
    audio = np.tile(
        np.array([1.0, -np.nextafter(1.0, 0.0)], dtype=np.float64),
        block_limit * 2,
    )
    observed = []
    real_add = signal_metrics_module._add_exact_units

    def recording_add(state, value):
        real_add(state, value)
        observed.append((state.shape, state.dtype, int(state.max())))

    monkeypatch.setattr(signal_metrics_module, "_add_exact_units", recording_add)

    compute_signal_metrics(audio)

    assert len(observed) == 4
    assert {shape for shape, _dtype, _maximum in observed} == {
        (2, signal_metrics_module._SUPERACCUMULATOR_LIMBS)
    }
    assert {dtype for _shape, dtype, _maximum in observed} == {np.dtype(np.int64)}
    assert all(
        maximum <= signal_metrics_module._SUPERACCUMULATOR_LIMB_MASK
        for _shape, _dtype, maximum in observed
    )


@pytest.mark.parametrize("crosses_block", [False, True], ids=["single-block", "cross-block"])
def test_int64_grouped_cancellation_matches_integer_sum(crosses_block):
    block_limit = signal_metrics_module._SIGNAL_METRICS_BLOCK_SAMPLES
    group_size = block_limit // 2 if crosses_block else block_limit // 8
    peak = 2**62
    positive = 3 * 2**60
    negative_magnitude = positive - 512
    audio = np.concatenate(
        (
            np.array([peak, -peak], dtype=np.int64),
            np.full(group_size, positive, dtype=np.int64),
            np.full(group_size, -negative_magnitude, dtype=np.int64),
        )
    )
    expected = (group_size * 512) / audio.size

    metrics = compute_signal_metrics(audio)

    assert metrics.dc_offset == expected
