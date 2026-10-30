"""SimulationTrace 样本隔离、namespace 与 seal 合同测试。"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tools.verify.simulation_audit_contract import _descriptor_from_stem

from core.data.simulation_trace import SimulationTrace, TracedSample

pytestmark = pytest.mark.unit


def _trace(*, capture_stems: bool = True) -> SimulationTrace:
    return SimulationTrace(
        index=3,
        uid="utt-3",
        task="tse",
        capture_stems=capture_stems,
    )


def test_metadata_and_stems_use_separate_namespaces() -> None:
    trace = _trace()
    target = trace.source("target")
    target.record_metadata("level", {"db": np.float32(-20.0)})
    target.record_stem("clean", np.array([0.1, -0.2], dtype=np.float64))
    trace.seal()

    assert target.metadata == {"level": {"db": -20.0}}
    assert target.stems["clean"].dtype == np.float32
    assert json.dumps(trace.metadata_dict(), allow_nan=False)


def test_metadata_rejects_waveform_arrays() -> None:
    trace = _trace()
    with pytest.raises(TypeError, match="record_stem"):
        trace.source("scene").record_metadata("bad", {"wave": np.zeros(4)})


@pytest.mark.parametrize("method", ["record_stem", "replace_stem"])
def test_seal_freezes_stem_snapshot_and_descriptor_hash(method: str) -> None:
    trace = _trace()
    scene = trace.source("scene")
    waveform = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    getattr(scene, method)("clean", waveform)
    open_alias = scene.stems["clean"]

    assert open_alias.dtype == np.float32
    assert open_alias.shape == (3,)
    assert open_alias.flags.writeable
    waveform[0] = 99.0
    np.testing.assert_array_equal(open_alias, [1.0, -2.0, 3.0])
    open_alias[0] = 4.0

    scene.replace_metadata("final_track", {"sample_rate": 16000, "num_samples": 3})
    trace.seal()
    sealed = trace.sources["scene"].stems["clean"]

    assert not sealed.flags.writeable
    assert isinstance(sealed.base, np.ndarray)
    assert isinstance(sealed.base.base, bytes)
    with pytest.raises(ValueError):
        sealed[0] = 8.0
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        sealed.setflags(write=True)
    with pytest.raises(RuntimeError, match="sealed"):
        scene.replace_stem("late", np.ones(3, dtype=np.float32))

    descriptor = _descriptor_from_stem(
        role="target", stem="final_target", waveform=sealed,
        metadata={"final_track": {"sample_rate": 16000, "num_samples": 3}},
    )
    assert descriptor["shape"] == [3]
    assert descriptor["dtype"] == "float32"
    assert descriptor["sha256"] == hashlib.sha256(
        np.array([4.0, -2.0, 3.0], dtype=np.float32).tobytes()
    ).hexdigest()
    open_alias[0] = -7.0
    assert descriptor["sha256"] == hashlib.sha256(sealed.tobytes()).hexdigest()
    np.testing.assert_array_equal(sealed, [4.0, -2.0, 3.0])


def test_capture_stems_false_keeps_no_array_reference() -> None:
    trace = _trace(capture_stems=False)
    waveform = np.arange(8, dtype=np.float32)
    trace.source("scene").record_stem("noisy", waveform)
    trace.seal()

    assert not trace.sources["scene"].stems
    assert "noisy" not in trace.metadata_dict()["sources"]["scene"]


def test_tse_roles_do_not_overwrite_each_other() -> None:
    trace = _trace()
    roles = (
        "target",
        "interference/0",
        "interference/1",
        "alternate_target/0",
        "alternate_target/1",
        "enrollment/0",
        "enrollment/1",
    )
    for index, role in enumerate(roles):
        source = trace.source(role)
        source.record_metadata("uid", f"speaker-{index}")
        source.record_stem("clean", np.full(3, index, dtype=np.float32))
    trace.seal()

    assert tuple(trace.sources) == roles
    for index, role in enumerate(roles):
        assert trace.sources[role].metadata["uid"] == f"speaker-{index}"
        np.testing.assert_array_equal(trace.sources[role].stems["clean"], index)


def test_identity_and_role_fields_are_immutable() -> None:
    trace = _trace()
    scene = trace.source("scene")

    for name, value in (
        ("index", 4),
        ("uid", "other"),
        ("task", "se"),
        ("capture_stems", False),
        ("placeholder", True),
    ):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(trace, name, value)
    with pytest.raises(AttributeError, match="immutable"):
        scene.role = "other"
    with pytest.raises(AttributeError, match="immutable"):
        scene.capture_stems = False

    trace.seal()
    for name, value in (("index", 5), ("uid", "sealed"), ("task", "aec"), ("capture_stems", False), ("placeholder", True)):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(trace, name, value)
    with pytest.raises(AttributeError, match="immutable"):
        scene.role = "sealed"


def test_seal_blocks_partial_mutation_and_traced_sample_requires_seal() -> None:
    trace = _trace()
    scene = trace.source("scene")
    with pytest.raises(ValueError, match="sealed"):
        TracedSample(sample=(1,), trace=trace)

    trace.seal()
    traced = TracedSample(sample=(1,), trace=trace)
    assert traced.sample == (1,)
    with pytest.raises(RuntimeError, match="sealed"):
        scene.record_metadata("late", True)
    with pytest.raises(RuntimeError, match="sealed"):
        trace.source("late")


def test_legacy_projection_copies_role_metadata_and_stems() -> None:
    trace = _trace()
    scene = trace.source("scene")
    scene.record_metadata("simulation_by_layer", {"B": {"branch": "main"}})
    scene.record_stem("final_noisy", np.ones(2, dtype=np.float32))
    trace.set_legacy_snapshot("_last_run_branch", "main")
    trace.seal()

    projected = trace.legacy_stems()
    assert projected is not None
    assert projected["simulation_by_layer"]["B"]["branch"] == "main"
    np.testing.assert_array_equal(projected["final_noisy"], 1.0)
    projected["final_noisy"][0] = 9.0
    assert trace.sources["scene"].stems["final_noisy"][0] == 1.0
    assert trace.legacy_snapshot() == {"_last_run_branch": "main"}


def test_metadata_and_legacy_snapshots_are_nested_defensive_copies() -> None:
    trace = _trace()
    source = trace.source("scene")
    source.record_metadata("nested", {"items": [{"value": 1}]})
    trace.record_metadata("nested", {"items": [{"value": 2}]})
    legacy = {"items": [{"value": 3}]}
    trace.set_legacy_snapshot("_last_nested", legacy)
    legacy["items"][0]["value"] = 99

    source_view = source.metadata
    trace_view = trace.metadata
    source_view["nested"]["items"][0]["value"] = 10
    trace_view["nested"]["items"][0]["value"] = 20
    trace.seal()

    assert source.metadata["nested"]["items"][0]["value"] == 1
    assert trace.metadata["nested"]["items"][0]["value"] == 2
    first_legacy = trace.legacy_snapshot()
    first_legacy["_last_nested"]["items"][0]["value"] = 30
    assert trace.legacy_snapshot()["_last_nested"]["items"][0]["value"] == 3


def test_simulation_trace_seal_preflights_all_sources_transactionally() -> None:
    trace = _trace()
    first = trace.source("first")
    second = trace.source("second")
    first.record_stem("wave", np.ones(2, dtype=np.float32))
    second.record_stem("wave", np.full(2, 2.0, dtype=np.float32))
    second.seal()

    with pytest.raises(RuntimeError, match="sealed before owner commit"):
        trace.seal()

    assert not trace.sealed
    assert not first.sealed
    assert first.stems["wave"].flags.writeable
    assert second.sealed


def test_simulation_trace_seal_prepare_failure_leaves_every_source_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _trace()
    first = trace.source("first")
    second = trace.source("second")
    first.record_stem("wave", np.ones(2, dtype=np.float32))
    second.record_stem("wave", np.full(2, 2.0, dtype=np.float32))

    def fail_snapshot(_: np.ndarray) -> np.ndarray:
        raise ValueError("injected snapshot failure")

    monkeypatch.setattr(
        "core.data.simulation_trace._immutable_float32_snapshot", fail_snapshot
    )
    with pytest.raises(ValueError, match="injected snapshot failure"):
        trace.seal()

    assert not trace.sealed
    assert not first.sealed and not second.sealed
    assert first.stems["wave"].flags.writeable
    assert second.stems["wave"].flags.writeable
