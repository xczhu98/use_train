"""AEC ERL configured/physical/effective range 合同测试。"""

from __future__ import annotations

import json

import pytest

from tasks.aec.data.dynamic_dataset import DynamicMixingDataset as AECDynamicMixingDataset
from tasks.aec.data.erl_coupling import (
    ERL_RANGE_EMPTY_INTERSECTION,
    ERLCouplingContractError,
    resolve_erl_coupling_range,
)

pytestmark = [pytest.mark.unit]


def test_resolve_erl_coupling_range_intersects_closed_ranges():
    result = resolve_erl_coupling_range(
        configured_range=[-20.0, 40.0],
        far_db_range=[-40.0, -12.0],
        echo_db=-30.0,
    )

    assert result.configured == (-20.0, 40.0)
    assert result.physical == (-10.0, 18.0)
    assert result.effective == (-10.0, 18.0)
    assert result.to_audit(6.0) == {
        "configured_range_db": [-20.0, 40.0],
        "physical_range_db": [-10.0, 18.0],
        "effective_range_db": [-10.0, 18.0],
        "selected_erl_db": 6.0,
    }


def test_resolve_erl_coupling_range_accepts_single_point_intersection():
    result = resolve_erl_coupling_range(
        configured_range=[10.0, 20.0],
        far_db_range=[-20.0, -20.0],
        echo_db=-40.0,
    )

    assert result.effective == (20.0, 20.0)

    pinned = resolve_erl_coupling_range(
        configured_range=[10.0, 10.0],
        far_db_range=[-20.0, -20.0],
        echo_db=-30.0,
    )
    assert pinned.effective == (10.0, 10.0)
    assert pinned.to_audit(10.0)["selected_erl_db"] == 10.0


def test_empty_intersection_raises_stable_json_safe_contract_error():
    with pytest.raises(ERLCouplingContractError) as caught:
        resolve_erl_coupling_range(
            configured_range=[30.0, 40.0],
            far_db_range=[-40.0, -30.0],
            echo_db=-20.0,
        )

    error = caught.value
    assert error.reason == ERL_RANGE_EMPTY_INTERSECTION
    assert error.metadata == {
        "configured_range_db": [30.0, 40.0],
        "physical_range_db": [-20.0, -10.0],
        "effective_range_db": None,
        "far_db_range_db": [-40.0, -30.0],
        "echo_db": -20.0,
    }
    json.dumps(error.metadata, allow_nan=False)


@pytest.mark.parametrize(
    "configured,far_range,echo_db",
    [
        ([float("nan"), 10.0], [-40.0, -12.0], -30.0),
        ([-20.0, 10.0], [-40.0, float("inf")], -30.0),
        ([-20.0, 10.0], [-40.0, -12.0], float("nan")),
        ([10.0, -20.0], [-40.0, -12.0], -30.0),
    ],
)
def test_invalid_inputs_fail_with_json_safe_metadata(configured, far_range, echo_db):
    with pytest.raises(ERLCouplingContractError) as caught:
        resolve_erl_coupling_range(configured, far_range, echo_db)

    assert caught.value.reason.startswith("erl_")
    json.dumps(caught.value.metadata, allow_nan=False)


def _bare_lenient_dataset(build_error: Exception):
    dataset = object.__new__(AECDynamicMixingDataset)
    dataset.strict_load = False
    dataset.speech_uids = {16000: ["speech"]}
    dataset.source_length = {"speech": 16000}
    dataset._worker_maybe_clear_caches = lambda: None
    dataset._get_from_index = lambda _index: (16000, 0)
    dataset._processing_sr = lambda fs: fs
    dataset._build_sample = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        build_error
    )
    dataset._placeholder_sample = lambda *_args: "placeholder"
    return dataset


def test_lenient_dataset_does_not_swallow_erl_contract_error():
    error = ERLCouplingContractError(
        ERL_RANGE_EMPTY_INTERSECTION,
        {"configured_range_db": [30.0, 40.0], "physical_range_db": [-20.0, -10.0]},
    )
    dataset = _bare_lenient_dataset(error)

    with pytest.raises(ERLCouplingContractError) as caught:
        dataset[0]

    assert caught.value is error


def test_lenient_dataset_keeps_ordinary_error_placeholder_semantics():
    dataset = _bare_lenient_dataset(OSError("temporary read failure"))

    assert dataset[0] == "placeholder"
