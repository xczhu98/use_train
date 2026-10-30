"""共享 timebase mechanics 的纯函数合同测试。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from core.data.simulation.timebase_impairment import (
    DELAY_SIGN_CONVENTION,
    DelayTrajectoryPlan,
    LinearTimebaseRatioPlan,
    TimebaseBackendFacts,
    TimebaseImpairmentResult,
    apply_delay_trajectory_plan,
    apply_integer_delay as apply_public_integer_delay,
    apply_linear_timebase_ratio_plan,
)
from core.data.timebase import (
    apply_delay_trajectory,
    apply_integer_delay,
    apply_linear_timebase_ratio,
)


pytestmark = [pytest.mark.unit]


def test_delay_trajectory_is_the_single_source_for_waveform_and_valid_mask():
    waveform = np.arange(8, dtype=np.float32)
    total_delay = np.array([2.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 1.0])

    result = apply_delay_trajectory(
        waveform,
        total_delay_samples=total_delay,
        base_delay_samples=2.0,
        edge_mode="constant",
    )

    np.testing.assert_array_equal(
        result.waveform,
        np.array([0.0, 1.0, 2.0, 2.0, 3.0, 6.0, 7.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result.read_positions,
        np.array([0.0, 1.0, 2.0, 2.0, 3.0, 6.0, 7.0, 8.0]),
    )
    np.testing.assert_array_equal(
        result.valid_mask,
        np.array([True, True, True, True, True, True, True, False]),
    )
    np.testing.assert_array_equal(result.total_delay_samples, total_delay)


def test_linear_timebase_ratio_matches_legacy_numpy_recipe_byte_for_byte():
    waveform = np.array([0.25, -0.5, 0.75, 0.125, -0.25], dtype=np.float32)
    ratio = 1.00008
    positions = np.arange(waveform.size, dtype=np.float64) * ratio
    valid = positions <= waveform.size - 1
    expected = np.zeros(waveform.size, dtype=np.float32)
    expected[valid] = np.interp(
        positions[valid],
        np.arange(waveform.size, dtype=np.float64),
        waveform.astype(np.float64),
    ).astype(np.float32)

    actual = apply_linear_timebase_ratio(waveform, ratio)

    np.testing.assert_array_equal(actual.waveform, expected)
    np.testing.assert_array_equal(actual.read_positions, positions)
    np.testing.assert_array_equal(actual.valid_mask, valid)


@pytest.mark.parametrize(
    ("delay", "expected"),
    [
        (0, [1.0, 2.0, 3.0]),
        (2, [0.0, 0.0, 1.0]),
        (3, [0.0, 0.0, 0.0]),
        (5, [0.0, 0.0, 0.0]),
    ],
)
def test_integer_delay_matches_legacy_zero_padding(delay, expected):
    waveform = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    actual = apply_integer_delay(waveform, delay)

    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.float32))


@pytest.mark.parametrize("delay", [0, 1, 3, 5], ids=["identity", "shift", "length", "oversized"])
def test_public_integer_delay_is_an_immutable_float32_snapshot(delay):
    waveform = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    expected = apply_integer_delay(waveform, delay)

    actual = apply_public_integer_delay(waveform, delay)

    assert actual.dtype == np.float32
    assert not actual.flags.writeable
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual.tobytes(), expected.tobytes())
    _assert_bytes_backed_immutable_snapshot(actual)
    waveform[0] = 99.0
    np.testing.assert_array_equal(actual, expected)


def test_delay_trajectory_rejects_nonfinite_or_wrong_length():
    waveform = np.ones(3, dtype=np.float32)
    with pytest.raises(ValueError, match="length"):
        apply_delay_trajectory(
            waveform,
            total_delay_samples=np.zeros(2),
            base_delay_samples=0.0,
        )
    with pytest.raises(ValueError, match="finite"):
        apply_delay_trajectory(
            waveform,
            total_delay_samples=np.array([0.0, np.nan, 0.0]),
            base_delay_samples=0.0,
        )


def _assert_bytes_backed_immutable_snapshot(array: np.ndarray) -> None:
    assert isinstance(array.base, np.ndarray)
    assert isinstance(array.base.base, bytes)
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        array.setflags(write=True)


def test_public_array_snapshots_preserve_dtype_shape_and_values():
    trajectory = np.array([[1, 2], [3, 4]], dtype=np.int16)
    result_values = {
        "waveform": np.array([[0.25, -0.5]], dtype=np.float32),
        "read_positions": np.array([[0.0, 1.0]], dtype=np.float64),
        "valid_mask": np.array([[True, False]], dtype=np.bool_),
        "total_delay_samples": np.array([[2, 3]], dtype=np.int16),
    }
    plan = DelayTrajectoryPlan(trajectory, base_delay_samples=0.0)
    result = TimebaseImpairmentResult(
        **result_values,
        backend_facts=TimebaseBackendFacts("test", "test", ()),
    )
    expected_result_values = {
        field_name: value.copy() for field_name, value in result_values.items()
    }
    trajectory[0, 0] = 99
    for value in result_values.values():
        value.flat[0] = 0

    assert plan.total_delay_samples.dtype == np.float64
    assert plan.total_delay_samples.shape == (trajectory.size,)
    np.testing.assert_array_equal(
        plan.total_delay_samples, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    )
    _assert_bytes_backed_immutable_snapshot(plan.total_delay_samples)
    for field_name, expected in expected_result_values.items():
        actual = getattr(result, field_name)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
        _assert_bytes_backed_immutable_snapshot(actual)


def test_public_delay_plan_is_immutable_and_delegates_legacy_mechanics():
    waveform = np.arange(8, dtype=np.float32)
    total_delay = np.array([2.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 1.0])
    plan = DelayTrajectoryPlan(
        total_delay_samples=total_delay,
        base_delay_samples=2.0,
    )

    total_delay[0] = 99.0
    with pytest.raises(FrozenInstanceError):
        plan.base_delay_samples = 1.0
    _assert_bytes_backed_immutable_snapshot(plan.total_delay_samples)
    np.testing.assert_array_equal(
        plan.total_delay_samples,
        np.array([2.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 1.0]),
    )

    public = apply_delay_trajectory_plan(waveform, plan)
    legacy = apply_delay_trajectory(
        waveform,
        total_delay_samples=plan.total_delay_samples,
        base_delay_samples=plan.base_delay_samples,
        edge_mode=plan.edge_mode,
    )

    assert plan.sign_convention == DELAY_SIGN_CONVENTION
    assert public.backend_facts.operation == "delay_trajectory"
    assert public.backend_facts.interpolation_backend == "scipy.ndimage.map_coordinates"
    for field_name in (
        "waveform",
        "read_positions",
        "valid_mask",
        "total_delay_samples",
    ):
        _assert_bytes_backed_immutable_snapshot(getattr(public, field_name))
    np.testing.assert_array_equal(public.waveform, legacy.waveform)
    np.testing.assert_array_equal(public.read_positions, legacy.read_positions)
    np.testing.assert_array_equal(public.valid_mask, legacy.valid_mask)
    np.testing.assert_array_equal(public.total_delay_samples, legacy.total_delay_samples)


def test_public_delay_plan_nearest_edge_mode_delegates_waveform_and_invalid_mask():
    waveform = np.arange(4, dtype=np.float32)
    plan = DelayTrajectoryPlan(
        total_delay_samples=np.full(waveform.size, 2.0),
        base_delay_samples=0.0,
        edge_mode="nearest",
    )

    public = apply_delay_trajectory_plan(waveform, plan)
    legacy = apply_delay_trajectory(
        waveform,
        total_delay_samples=plan.total_delay_samples,
        base_delay_samples=plan.base_delay_samples,
        edge_mode="nearest",
    )

    np.testing.assert_array_equal(public.waveform, legacy.waveform)
    np.testing.assert_array_equal(public.valid_mask, legacy.valid_mask)
    np.testing.assert_array_equal(public.waveform, np.array([0.0, 0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(public.valid_mask, [False, False, True, True])


def test_public_delay_plan_rejects_invalid_sign_convention():
    waveform = np.zeros(2, dtype=np.float32)
    plan = DelayTrajectoryPlan(
        total_delay_samples=np.zeros(2),
        base_delay_samples=0.0,
        sign_convention="positive_delay_means_capture_leads_reference",
    )

    with pytest.raises(ValueError, match="sign_convention"):
        apply_delay_trajectory_plan(waveform, plan)


@pytest.mark.parametrize("ratio", [1.00008, 0.99992], ids=["positive_ppm", "negative_ppm"])
def test_public_linear_ratio_plan_matches_legacy_numpy_recipe_byte_for_byte(ratio):
    waveform = np.array([0.25, -0.5, 0.75, 0.125, -0.25], dtype=np.float32)
    positions = np.arange(waveform.size, dtype=np.float64) * ratio
    valid = positions <= waveform.size - 1
    expected = np.zeros(waveform.size, dtype=np.float32)
    expected[valid] = np.interp(
        positions[valid],
        np.arange(waveform.size, dtype=np.float64),
        waveform.astype(np.float64),
    ).astype(np.float32)

    public = apply_linear_timebase_ratio_plan(
        waveform, LinearTimebaseRatioPlan(ratio)
    )
    legacy = apply_linear_timebase_ratio(waveform, ratio)

    assert public.backend_facts.operation == "linear_ratio"
    assert public.backend_facts.interpolation_backend == "numpy.interp"
    assert "scipy.signal.resample" in public.backend_facts.asrc_backends
    np.testing.assert_array_equal(public.waveform, expected)
    np.testing.assert_array_equal(public.read_positions, positions)
    np.testing.assert_array_equal(public.valid_mask, valid)
    np.testing.assert_array_equal(public.waveform, legacy.waveform)
