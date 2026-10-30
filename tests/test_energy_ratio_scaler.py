"""EnergyRatioScaler 纯数学与兼容公式测试。"""

import numpy as np
import pytest

from core.data.simulation.energy_ratio import EnergyRatioScaler

pytestmark = [pytest.mark.unit]


def test_active_mean_power_uses_float64_accumulator():
    samples = np.full((1, 8), 1e20, dtype=np.float32)
    power = EnergyRatioScaler.measure(
        samples,
        policy="active_mean_power",
        active_mask=np.ones(8, dtype=bool),
    )
    assert np.isfinite(power)
    assert power == pytest.approx(1e40, rel=1e-6)


def test_power_ratio_preserves_floor_cap_and_operation_order():
    result = EnergyRatioScaler.power_ratio(
        4.0,
        0.0,
        -200.0,
        policy="active_mean_power",
        denominator_floor=1e-10,
        gain_cap=1e4,
    )
    assert result.denominator_energy == 1e-10
    assert result.gain == 1e4


def test_full_sum_energy_and_tse_floor_placement_match_legacy_formula():
    target = np.array([[0.5, -0.25, 0.125]], dtype=np.float32)
    interferer = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
    e_target = EnergyRatioScaler.measure(target, policy="full_sum_energy")
    e_interf = EnergyRatioScaler.measure(interferer, policy="full_sum_energy")
    result = EnergyRatioScaler.sum_energy_ratio(e_target, e_interf, 6.0)
    expected = np.sqrt(e_target / (e_interf * 10.0 ** 0.6 + 1e-10))
    assert result.gain == pytest.approx(expected, rel=0.0, abs=0.0)


def test_energy_db_ratio_uses_measured_energy_values():
    assert EnergyRatioScaler.energy_db_ratio(4.0, 1.0) == pytest.approx(
        6.020599913279624
    )


def test_measure_full_mean_and_empty_policy():
    samples = np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    assert EnergyRatioScaler.measure(samples, policy="full_mean_power") == 2.5
    assert EnergyRatioScaler.measure(
        np.array([], dtype=np.float32), policy="full_sum_energy"
    ) == 1e-10


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_energy_fails_closed(value):
    with pytest.raises(ValueError, match="energy must be finite"):
        EnergyRatioScaler.measure(np.array([value]), policy="full_mean_power")
