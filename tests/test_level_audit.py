"""LevelAudit 四阶段与 JSON-safe 合同测试。"""

import json

import numpy as np
import pytest

from core.data.simulation.level_audit import LevelAudit, json_safe

pytestmark = [pytest.mark.unit]


def test_level_audit_records_four_stages_and_measure_semantics():
    audit = LevelAudit(
        name="mixture.snr",
        candidate_drawn_db=20.0,
        selected_nominal_db=18.0,
        calibration_realized_db=18.0,
        final_realized_db=17.5,
        numerator="early_rir_nearend",
        denominator="ordinary_noise",
        policy="active_mean_power",
        calibration_measure_point="post_rir_pre_mix",
        final_measure_point="final",
        applicable=True,
        observations={"post_rir": 18.0},
    )
    payload = audit.to_dict()
    assert payload["candidate_drawn_db"] == 20.0
    assert payload["selected_nominal_db"] == 18.0
    assert payload["calibration_realized_db"] == 18.0
    assert payload["final_realized_db"] == 17.5
    assert payload["numerator"] == "early_rir_nearend"
    assert payload["denominator"] == "ordinary_noise"
    assert payload["observations"] == {"post_rir": 18.0}
    json.dumps(payload, allow_nan=False)


def test_non_applicable_requires_reason_and_uses_none_not_pseudo_value():
    audit = LevelAudit(
        name="mixture.sir",
        candidate_drawn_db=0.0,
        selected_nominal_db=0.0,
        calibration_realized_db=0.0,
        final_realized_db=None,
        numerator="target",
        denominator="interference",
        policy="full_sum_energy",
        calibration_measure_point="pre_rir",
        final_measure_point="final",
        applicable=False,
        reason="nonlinear_observation",
    )
    assert audit.to_dict()["final_realized_db"] is None
    assert audit.to_dict()["reason"] == "nonlinear_observation"

    with pytest.raises(ValueError, match="requires a reason"):
        LevelAudit(
            name="bad",
            candidate_drawn_db=None,
            selected_nominal_db=None,
            calibration_realized_db=None,
            final_realized_db=None,
            numerator="a",
            denominator="b",
            policy="full_mean_power",
            calibration_measure_point="none",
            final_measure_point="none",
            applicable=False,
        )


def test_json_safe_converts_numpy_and_nonfinite_values():
    payload = json_safe(
        {"x": np.float32(1.25), "bad": np.inf, "items": np.array([1, 2])}
    )
    assert payload == {"x": 1.25, "bad": None, "items": [1, 2]}
    json.dumps(payload, allow_nan=False)
