"""固定顺序 post-mix pipeline 的纯合同测试。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
import json

import numpy as np
import pytest

from core.data.post_mix_pipeline import (
    POST_MIX_STAGE_ORDER,
    PostMixContext,
    PostMixStage,
    PostMixStageResult,
    PostMixState,
    run_post_mix_orchestration,
    run_post_mix_pipeline,
)
from core.data.simulation_trace import SimulationTrace
from core.data.waveform_bundle import WaveformBundle


pytestmark = [pytest.mark.unit]


def _state(value: float = 0.0, *, length: int = 4) -> PostMixState:
    return PostMixState(
        {
            "observation": np.full(length, value, dtype=np.float32),
            "target": np.full((1, length), value + 1.0, dtype=np.float32),
        }
    )


class _OrderedAdapter:
    def __init__(self) -> None:
        self.calls: list[PostMixStage] = []

    def apply_stage(
        self, stage: PostMixStage, state: PostMixState
    ) -> PostMixStageResult:
        self.calls.append(stage)
        value = float(len(self.calls))
        next_state = PostMixState(
            {
                role: np.full(waveform.shape, value, dtype=np.float32)
                for role, waveform in state.waveforms.items()
            }
        )
        return PostMixStageResult(stage=stage, state=next_state, applied=True)


def test_pipeline_executes_and_records_exact_fixed_stage_order() -> None:
    expected_order: tuple[PostMixStage, ...] = (
        "pre-finite",
        "D-normalize",
        "D-guard",
        "E",
        "F",
        "task-post-EF",
        "final-guard",
        "post-finite",
    )
    adapter = _OrderedAdapter()

    trace = run_post_mix_pipeline(_state(), adapter)

    assert POST_MIX_STAGE_ORDER == expected_order
    assert tuple(adapter.calls) == expected_order
    assert trace.stage_order == expected_order
    assert tuple(result.stage for result in trace.results) == expected_order
    np.testing.assert_array_equal(
        trace.final_state.waveforms["observation"],
        np.full(4, len(expected_order), dtype=np.float32),
    )


def test_compact_pipeline_retains_stage_facts_without_intermediate_waveforms() -> None:
    trace = run_post_mix_pipeline(
        _state(), _OrderedAdapter(), retain_stage_states=False
    )

    assert trace.retains_stage_states is False
    assert trace.stage_order == POST_MIX_STAGE_ORDER
    assert all(result.applied for result in trace.results)
    assert all(result.state is trace.final_state for result in trace.results)


@pytest.mark.parametrize(
    ("waveforms", "error", "message"),
    [
        ({}, ValueError, "non-empty mapping"),
        ({"x": np.ones(4, dtype=np.float64)}, TypeError, "float32 ndarray"),
        ({"x": np.ones(4, dtype=np.float32), "y": np.ones(3, dtype=np.float32)}, ValueError, "time-axis"),
        ({"x": np.array([0.0, np.nan], dtype=np.float32)}, ValueError, "NaN or Inf"),
        ({"x": np.empty((1, 0), dtype=np.float32)}, ValueError, "non-empty final time axis"),
    ],
)
def test_state_rejects_invalid_waveform_contract(waveforms, error, message) -> None:
    with pytest.raises(error, match=message):
        PostMixState(waveforms)


def test_state_owns_readonly_float32_waveforms() -> None:
    source = np.arange(4, dtype=np.float32)
    state = PostMixState({"observation": source})

    source[:] = -1.0
    np.testing.assert_array_equal(
        state.waveforms["observation"], np.arange(4, dtype=np.float32)
    )
    assert state.waveforms["observation"].dtype == np.float32
    with pytest.raises(ValueError, match="read-only"):
        state.waveforms["observation"][0] = 99.0
    with pytest.raises(ValueError):
        state.waveforms["observation"].setflags(write=True)
    with pytest.raises(TypeError):
        state.waveforms["other"] = np.zeros(4, dtype=np.float32)


def test_internal_replacement_shares_unchanged_readonly_tracks() -> None:
    state = _state()
    replacement = np.full(4, 3.0, dtype=np.float32)

    next_state = state._replace_waveforms({"observation": replacement})

    assert next_state is not state
    assert next_state.waveforms["target"] is state.waveforms["target"]
    assert next_state.waveforms["observation"] is not replacement
    replacement[:] = -1.0
    np.testing.assert_array_equal(next_state.waveforms["observation"], 3.0)
    with pytest.raises(ValueError):
        next_state.waveforms["observation"].setflags(write=True)


def test_internal_identity_replacement_returns_same_state() -> None:
    state = _state()

    assert state._replace_waveforms({}) is state
    assert (
        state._replace_waveforms({"observation": state.waveforms["observation"]})
        is state
    )
    assert (
        state._replace_waveforms({"observation": state.waveforms["observation"].copy()})
        is state
    )


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"missing": np.ones(4, dtype=np.float32)}, KeyError),
        ({"observation": np.ones(4, dtype=np.float64)}, TypeError),
        ({"observation": np.ones(3, dtype=np.float32)}, ValueError),
        ({"observation": np.array([0.0, 0.0, 0.0, np.inf], dtype=np.float32)}, ValueError),
    ],
)
def test_internal_replacement_revalidates_changed_tracks(changes, error) -> None:
    with pytest.raises(error):
        _state()._replace_waveforms(changes)


class _WrongStageAdapter:
    def __init__(self) -> None:
        self.calls: list[PostMixStage] = []

    def apply_stage(
        self, stage: PostMixStage, state: PostMixState
    ) -> PostMixStageResult:
        self.calls.append(stage)
        returned_stage: PostMixStage = (
            "D-guard" if stage == "D-normalize" else stage
        )
        return PostMixStageResult(returned_stage, state, applied=False)


def test_wrong_stage_fails_immediately_without_running_later_stages() -> None:
    adapter = _WrongStageAdapter()

    with pytest.raises(
        ValueError,
        match="returned stage 'D-guard'.*expected 'D-normalize'",
    ):
        run_post_mix_pipeline(_state(), adapter)

    assert adapter.calls == ["pre-finite", "D-normalize"]


class _MissingStageAdapter:
    def __init__(self) -> None:
        self.calls: list[PostMixStage] = []

    def apply_stage(
        self, stage: PostMixStage, state: PostMixState
    ) -> PostMixStageResult | None:
        self.calls.append(stage)
        if stage == "E":
            return None
        return PostMixStageResult(stage, state, applied=False)


def test_missing_stage_result_is_rejected_and_stops_pipeline() -> None:
    adapter = _MissingStageAdapter()

    with pytest.raises(TypeError, match="did not return PostMixStageResult for 'E'"):
        run_post_mix_pipeline(_state(), adapter)  # type: ignore[arg-type]

    assert adapter.calls == ["pre-finite", "D-normalize", "D-guard", "E"]


class _InvalidOutputAdapter:
    def __init__(self) -> None:
        self.calls: list[PostMixStage] = []

    def apply_stage(
        self, stage: PostMixStage, state: PostMixState
    ) -> PostMixStageResult:
        self.calls.append(stage)
        if stage == "F":
            PostMixState({"observation": np.array([np.inf], dtype=np.float32)})
        return PostMixStageResult(stage, state, applied=False)


def test_invalid_intermediate_state_fails_at_origin_and_stops() -> None:
    adapter = _InvalidOutputAdapter()

    with pytest.raises(ValueError, match="NaN or Inf"):
        run_post_mix_pipeline(_state(), adapter)

    assert adapter.calls == [
        "pre-finite",
        "D-normalize",
        "D-guard",
        "E",
        "F",
    ]


def test_contract_records_are_frozen() -> None:
    state = _state()
    result = PostMixStageResult("pre-finite", state, applied=False)
    trace = run_post_mix_pipeline(state, _OrderedAdapter())

    with pytest.raises(FrozenInstanceError):
        state.waveforms = {}
    with pytest.raises(FrozenInstanceError):
        result.applied = True
    with pytest.raises(FrozenInstanceError):
        trace.results = ()


def test_typed_orchestration_returns_final_bundle_and_frozen_context() -> None:
    bundle = WaveformBundle(
        {
            "observation": np.zeros(4, dtype=np.float32),
            "target": np.ones(4, dtype=np.float32),
        },
        metadata={"sample_rate": 16_000},
    )
    adapter = _OrderedAdapter()

    result = run_post_mix_orchestration(
        bundle,
        adapter,
        context={"branch": "main", "nested": {"stage_count": 9}},
    )

    assert adapter.calls == list(POST_MIX_STAGE_ORDER)
    assert result.trace.stage_order == POST_MIX_STAGE_ORDER
    assert result.bundle.metadata == bundle.metadata
    assert result.context.facts["nested"]["stage_count"] == 9
    np.testing.assert_array_equal(
        result.bundle.require("observation"),
        np.full(4, len(POST_MIX_STAGE_ORDER), dtype=np.float32),
    )
    with pytest.raises(TypeError):
        result.context.facts["branch"] = "changed"


@pytest.mark.parametrize(
    "facts",
    [
        {"waveform": np.ones(2, dtype=np.float32)},
        {"infinite": float("inf")},
        {"bad": object()},
    ],
)
def test_typed_orchestration_rejects_non_json_context_facts(facts) -> None:
    bundle = WaveformBundle({"observation": np.zeros(2, dtype=np.float32)})

    with pytest.raises((TypeError, ValueError)):
        run_post_mix_orchestration(bundle, _OrderedAdapter(), context=facts)


def test_typed_orchestration_has_no_rng_interface() -> None:
    assert "rng" not in inspect.signature(run_post_mix_orchestration).parameters
    assert isinstance(PostMixContext({}), PostMixContext)


def test_context_to_dict_projects_nested_frozen_facts_to_strict_json() -> None:
    context = PostMixContext(
        {
            "enabled": np.bool_(True),
            "attempts": np.int64(3),
            "gain": np.float32(0.75),
            "nested": {"items": (np.bool_(False), [np.int32(4), "ok"])},
        }
    )

    projected = context.to_dict()

    assert projected == {
        "enabled": True,
        "attempts": 3,
        "gain": pytest.approx(0.75),
        "nested": {"items": [False, [4, "ok"]]},
    }
    assert json.dumps(projected, allow_nan=False)
    assert isinstance(context.facts["nested"], type(context.facts))


@pytest.mark.parametrize("value", [np.float32(np.inf), np.float64(np.nan)])
def test_context_rejects_nonfinite_numpy_scalars(value) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        PostMixContext({"value": value})


def test_context_projection_is_accepted_by_simulation_trace_metadata() -> None:
    trace = SimulationTrace(index=1, uid="utt", task="se", capture_stems=False)
    context = PostMixContext({"nested": {"flags": (np.bool_(True), np.int32(2))}})

    trace.source("scene").record_metadata("post_mix_context", context.to_dict())
    trace.seal()

    assert trace.sources["scene"].metadata["post_mix_context"] == {
        "nested": {"flags": [True, 2]}
    }
    assert json.dumps(trace.metadata_dict(), allow_nan=False)
