"""gpuRIR source 自适应分块的回归测试。"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from core.data.simulation import gpurir_backend as backend

pytestmark = pytest.mark.unit


def _plan(sources, *, scene=False):
    plan = {
        "room_dim": [4.0, 4.0, 3.0],
        "source_loc": [[1.0 + index, 1.0, 1.2] for index in range(sources)],
        "mic_loc": [2.0, 2.0, 1.2],
        "num_samples": 8,
        "sample_rate": 8000,
        "rt60": 0.3,
        "diffuse_db": 15.0,
        "sound_speed": 340.0,
        "group_uid": "chunking",
        "source_ids": [f"s{index}" for index in range(sources)],
    }
    if scene:
        plan["scene_uid"] = "chunking-scene"
    return plan


def _fake_group(max_sources, calls):
    """超过 ``max_sources`` 时抛 MemoryError 的 fake backend。"""

    def generate(plan, gpu_rir=None, *, timeout=None, track_kind="full", **kwargs):
        count = len(plan["source_loc"])
        calls.append((track_kind, count, tuple(plan["source_ids"])))
        if count > max_sources:
            raise MemoryError("gpuRIR internal allocation exceeds memory budget")
        wave = (np.arange(count * 8, dtype="<f4").reshape(count, 8) + 1.0)
        facts = {
            **plan,
            "track_kind": track_kind,
            "rir_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
            "provenance": {"requested_shape": list(wave.shape), "elapsed_sec": 0.5},
        }
        if plan.get("scene_uid") is not None:
            facts["actor_order"] = list(plan.get("source_ids", ()))
        return wave, facts

    return generate


def test_within_budget_uses_single_backend_call(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "generate_rir_group", _fake_group(99, calls))
    plan = _plan(4)
    wave, facts = backend.generate_rir_group_budgeted(plan, None, track_kind="full")
    assert [count for _, count, _ in calls] == [4]
    assert wave.shape == (4, 8)
    assert facts["source_loc"] == plan["source_loc"]
    assert facts["source_ids"] == plan["source_ids"]


def test_over_budget_splits_by_source_and_rebuilds_identity(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "generate_rir_group", _fake_group(2, calls))
    plan = _plan(5, scene=True)
    wave, facts = backend.generate_rir_group_budgeted(plan, None, track_kind="full")

    # 5 → (2,3)；3 → (1,2)：整块与中间块各失败一次，叶子块全部成功。
    assert [count for _, count, _ in calls] == [5, 2, 3, 1, 2]
    assert wave.shape == (5, 8)
    # 拼接后必须恢复整块的 source 身份，不能留下子块的切片。
    assert facts["source_loc"] == plan["source_loc"]
    assert facts["source_ids"] == plan["source_ids"]
    assert facts["actor_order"] == plan["source_ids"]
    assert facts["rir_sha256"] == hashlib.sha256(wave.tobytes()).hexdigest()
    assert facts["provenance"]["source_chunks"] == 2
    assert facts["provenance"]["requested_shape"] == [5, 8]
    # 叶子块 0.5s × 3 次 backend 调用，逐层累加。
    assert facts["provenance"]["elapsed_sec"] == pytest.approx(1.5)


def test_single_source_over_budget_stays_fail_closed(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "generate_rir_group", _fake_group(0, calls))
    with pytest.raises(MemoryError, match="memory budget"):
        backend.generate_rir_group_budgeted(_plan(3), None, track_kind="full")
    # 二分到单 source 后仍超预算即立刻抛出，不再尝试剩余 source（不得静默减少）。
    assert [count for _, count, _ in calls] == [3, 1]


def test_rir_tracks_keep_full_and_direct_shapes_aligned(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "generate_rir_group", _fake_group(2, calls))
    plan = _plan(4)
    tracks = backend.generate_rir_tracks(plan, None)
    assert tracks["full"][0].shape == tracks["direct"][0].shape == (4, 8)
    assert tracks["full"][1]["source_loc"] == plan["source_loc"]
    assert tracks["direct"][1]["source_loc"] == plan["source_loc"]
    assert {kind for kind, _, _ in calls} == {"full", "direct"}
