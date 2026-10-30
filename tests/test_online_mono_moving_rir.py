"""在线 mono moving RIR（缺口 1）单元测试。

覆盖：SimulationConfigs 校验放宽、OnlineRIRProvider mono moving 规划、
MovingSpatialRIRTrackSet mono 形状合同、`_mix_acoustic_rir` 在线 mono
moving 接线（分段参数、early 派生、RIRPlan 记录、概率门控）。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import core.data.dynamic_dataset_base as dynamic_base
from core.data.dynamic_dataset_base import BaseDynamicMixingDataset
from core.data.simulation.rir_config import OnlineRIRConfig
from core.data.simulation.rir_provider import (
    MovingSpatialRIRTrackSet,
    OnlineRIRProvider,
    SpatialRIRResult,
)
from core.data.simulation_configs import SimulationConfigs


pytestmark = [pytest.mark.unit]

SAMPLE_RATE = 8000
NUM_SAMPLES = 4000
TRAJECTORY_POINTS = 4
RIR_LENGTH = 64


def _mono_plan() -> dict:
    rng = np.random.default_rng(7)
    positions = np.linspace(
        np.array([[1.0, 1.0, 1.5]]),
        np.array([[1.4, 1.2, 1.5]]),
        TRAJECTORY_POINTS,
        dtype=np.float32,
    )
    hop = NUM_SAMPLES // TRAJECTORY_POINTS
    return {
        "group_uid": "moving-test-group",
        "sample_rate": SAMPLE_RATE,
        "num_samples": RIR_LENGTH,
        "room_dim": [4.0, 4.0, 3.0],
        "rt60": 0.3,
        "diffuse_db": 15.0,
        "sound_speed": 340.0,
        "mic_loc": [2.0, 2.0, 1.5],
        "source_loc": positions[0].tolist(),
        "source_ids": ["speech"],
        "trajectory": positions.tolist(),
        "trajectory_timebase": {
            "sample_rate": SAMPLE_RATE,
            "num_samples": NUM_SAMPLES,
            "hop_samples": hop,
            "frame_count": TRAJECTORY_POINTS,
            "start_sample": 0,
        },
    }


def _mono_track_set() -> MovingSpatialRIRTrackSet:
    rng = np.random.default_rng(11)
    bank = rng.standard_normal((TRAJECTORY_POINTS, 1, RIR_LENGTH)).astype(np.float32)
    direct = rng.standard_normal((TRAJECTORY_POINTS, 1, RIR_LENGTH)).astype(np.float32)
    return MovingSpatialRIRTrackSet(bank, direct, _mono_plan(), {"full": {}, "direct": {}})


def _online_rir_config(prob_moving_rir: float = 1.0) -> OnlineRIRConfig:
    return OnlineRIRConfig.from_simulation(
        SimulationConfigs.from_dict(
            {
                "rir_backend": "online",
                "prob_moving_rir": float(prob_moving_rir),
                "gpurir_rt60_range_sec": [0.1, 0.5],
            }
        )
    )


def _dataset(sim_cfg) -> BaseDynamicMixingDataset:
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = sim_cfg
    dataset._rir_reuse_cache = None
    dataset._active_acoustic_plan_record = {"noise_sources": [], "noise_assets": []}
    dataset.strict_load = True
    return dataset


class _FakeMovingClient:
    """按请求的 plan 回显 group_uid 的 moving_full_direct 假 client。"""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.track_kinds: list[str] = []

    def generate_tracks(self, plan, *, track_kind):
        assert track_kind in {"moving_full", "moving_full_direct"}
        self.requests.append(plan)
        self.track_kinds.append(track_kind)
        rng = np.random.default_rng(3)
        points = plan["trajectory_timebase"]["frame_count"]
        source_count = len(plan["source_ids"])
        shape = (points, source_count, int(plan["num_samples"]))
        tracks = {}
        names = ("full",) if track_kind == "moving_full" else ("full", "direct")
        for name in names:
            wave = rng.standard_normal(shape).astype(np.float32)
            tracks[name] = (
                wave,
                {"group_uid": plan["group_uid"], "track_kind": name},
            )
        return tracks


class _StaticOnlyProvider:
    def __init__(self) -> None:
        self.static_calls = 0
        self.moving_calls = 0

    def sample_scene(self, sr, source_count, rng, *, normalize=True):
        self.static_calls += 1
        raise AssertionError("static path must not be used when moving is drawn")

    def sample_moving_spatial_tracks(
        self, sr, source_count, rng, *, num_samples, normalize=True, track_kinds=("full", "direct")
    ):
        self.moving_calls += 1
        assert num_samples == NUM_SAMPLES
        assert track_kinds == ("full",)
        return _mono_track_set()


def test_moving_spatial_rir_track_set_accepts_legacy_mono_bank():
    track_set = _mono_track_set()
    assert track_set.full.shape == (TRAJECTORY_POINTS, 1, RIR_LENGTH)
    assert track_set.direct.shape == (TRAJECTORY_POINTS, 1, RIR_LENGTH)


def test_moving_spatial_rir_track_set_rejects_mono_shape_mismatch():
    plan = _mono_plan()
    bad = np.zeros((TRAJECTORY_POINTS, 1, 1, RIR_LENGTH), dtype=np.float32)
    with pytest.raises(ValueError, match="moving full RIR must be finite with shape"):
        MovingSpatialRIRTrackSet(bad, bad.copy(), plan, {})


def test_online_provider_mono_moving_planning_uses_legacy_shape():
    client = _FakeMovingClient()
    provider = OnlineRIRProvider(_online_rir_config(), client)
    track_set = provider.sample_moving_spatial_tracks(
        SAMPLE_RATE, 1, np.random.default_rng(5), num_samples=NUM_SAMPLES,
    )
    (request,) = client.requests
    # legacy mono 布局：[P, K, L]，无 microphone 轴；L 由几何采样决定。
    expected = (TRAJECTORY_POINTS, 1, int(request["num_samples"]))
    assert track_set.full.shape == expected
    assert track_set.direct.shape == expected
    assert np.asarray(request["mic_loc"]).shape == (3,)
    assert request["trajectory_timebase"]["hop_samples"] == NUM_SAMPLES // TRAJECTORY_POINTS
    assert request["trajectory_timebase"]["num_samples"] == NUM_SAMPLES
    # full/direct 共享同一 group identity。
    assert track_set.plan["group_uid"].startswith("moving-")


def test_online_provider_mono_moving_can_skip_direct_track():
    client = _FakeMovingClient()
    provider = OnlineRIRProvider(_online_rir_config(), client)
    track_set = provider.sample_moving_spatial_tracks(
        SAMPLE_RATE, 1, np.random.default_rng(5), num_samples=NUM_SAMPLES,
        track_kinds=("full",),
    )
    assert client.track_kinds == ["moving_full"]
    assert track_set.direct is None


def test_mix_acoustic_rir_online_mono_moving_contract(monkeypatch):
    sim_cfg = SimpleNamespace(
        rir_backend="online",
        spatial_enabled=False,
        prob_moving_rir=1.0,
        rir_normalize=True,
        early_rir_sec=0.05,
        target_type="early",
        reuse_rir=False,
    )
    dataset = _dataset(sim_cfg)
    provider = _StaticOnlyProvider()
    dataset._online_rir_provider = provider

    calls = []

    def moving_reverb(speech, rir_seq, seg_sec, fs, **kwargs):
        calls.append((np.asarray(rir_seq).copy(), seg_sec, fs, kwargs))
        return speech.copy()

    monkeypatch.setattr(dynamic_base, "add_moving_reverberation", moving_reverb)
    monkeypatch.setattr(
        dynamic_base,
        "add_paired_reverberation",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("paired path used")),
    )

    speech = np.ones((1, NUM_SAMPLES), dtype=np.float32)
    noisy, reference, target = dataset._mix_acoustic_rir(
        speech, SAMPLE_RATE, np.random.default_rng(0), True
    )

    assert provider.moving_calls == 1
    assert len(calls) == 2  # full + early 各一次
    track_set = _mono_track_set()
    expected_seq = track_set.full[:, 0, :]
    expected_seg_sec = (NUM_SAMPLES // TRAJECTORY_POINTS) / SAMPLE_RATE
    for rir_seq, seg_sec, fs, kwargs in calls:
        np.testing.assert_allclose(rir_seq, expected_seq, rtol=1e-6, atol=1e-8)
        assert seg_sec == pytest.approx(expected_seg_sec)
        assert fs == SAMPLE_RATE
        assert kwargs["normalization_peak"] == pytest.approx(float(np.max(np.abs(expected_seq))))
    np.testing.assert_array_equal(noisy, speech)
    np.testing.assert_array_equal(reference, speech)
    np.testing.assert_array_equal(target, reference)  # target_type="early"

    plan = dataset._active_acoustic_plan_record["rir_plan"]
    assert plan.moving is True
    assert plan.backend == "online"
    assert plan.path is None
    assert plan.segment_sec == pytest.approx(expected_seg_sec)
    assert plan.uid.endswith("/moving")
    assert "online_rir_scene" in dataset._active_acoustic_plan_record


def test_mix_acoustic_rir_online_mono_moving_dry_target(monkeypatch):
    sim_cfg = SimpleNamespace(
        rir_backend="online",
        spatial_enabled=False,
        prob_moving_rir=1.0,
        rir_normalize=True,
        early_rir_sec=0.05,
        target_type="dry",
        reuse_rir=False,
    )
    dataset = _dataset(sim_cfg)
    dataset._online_rir_provider = _StaticOnlyProvider()
    monkeypatch.setattr(
        dynamic_base, "add_moving_reverberation", lambda speech, *a, **k: speech.copy()
    )
    speech = np.ones((1, NUM_SAMPLES), dtype=np.float32)
    _, _, target = dataset._mix_acoustic_rir(
        speech, SAMPLE_RATE, np.random.default_rng(0), True
    )
    np.testing.assert_array_equal(target, speech)


def test_mix_acoustic_rir_online_mono_static_when_moving_not_drawn():
    sim_cfg = SimpleNamespace(
        rir_backend="online",
        spatial_enabled=False,
        prob_moving_rir=0.0,
        rir_normalize=True,
        early_rir_sec=0.05,
        target_type="early",
        reuse_rir=False,
    )
    dataset = _dataset(sim_cfg)

    class _Provider:
        def __init__(self) -> None:
            self.moving_calls = 0

        def sample_scene(self, sr, source_count, rng, *, normalize=True):
            return SpatialRIRResult(
                np.ones((1, 8), dtype=np.float32), {}, {}, "test-v1",
                details={"group_uid": "group-static-1"},
            )

        def sample_moving_spatial_tracks(self, *args, **kwargs):
            self.moving_calls += 1
            raise AssertionError("moving must not be requested when prob=0")

    provider = _Provider()
    dataset._online_rir_provider = provider
    speech = np.ones((1, NUM_SAMPLES), dtype=np.float32)
    dataset._mix_acoustic_rir(speech, SAMPLE_RATE, np.random.default_rng(0), True)
    assert provider.moving_calls == 0
    plan = dataset._active_acoustic_plan_record["rir_plan"]
    assert plan.moving is False


def test_mix_acoustic_rir_online_mono_moving_rejects_spatial_bank(monkeypatch):
    sim_cfg = SimpleNamespace(
        rir_backend="online",
        spatial_enabled=False,
        prob_moving_rir=1.0,
        rir_normalize=True,
        early_rir_sec=0.05,
        target_type="early",
        reuse_rir=False,
    )
    dataset = _dataset(sim_cfg)

    class _BadProvider:
        def sample_moving_spatial_tracks(self, *args, **kwargs):
            bad_plan = _mono_plan()
            bad_plan["mic_loc"] = [[2.0, 2.0, 1.5], [2.1, 2.0, 1.5]]
            bank = np.zeros((TRAJECTORY_POINTS, 1, 2, RIR_LENGTH), dtype=np.float32)
            return MovingSpatialRIRTrackSet(bank, bank.copy(), bad_plan, {})

    dataset._online_rir_provider = _BadProvider()
    monkeypatch.setattr(
        dynamic_base, "add_moving_reverberation", lambda speech, *a, **k: speech.copy()
    )
    with pytest.raises(ValueError, match="online mono moving RIR bank"):
        dataset._mix_acoustic_rir(
            np.ones((1, NUM_SAMPLES), dtype=np.float32),
            SAMPLE_RATE,
            np.random.default_rng(0),
            True,
        )


def test_request_shape_accepts_legacy_mono_moving_plan():
    """service 侧 _request_shape 必须接受 mono mic_loc [3] 并固化 [P,K,L]。"""
    from core.data.simulation.rir_service import _request_shape

    shape = _request_shape(_mono_plan(), "moving_full_direct")
    assert shape == (TRAJECTORY_POINTS, 1, RIR_LENGTH)
    bad = dict(_mono_plan())
    bad["mic_loc"] = [[1.0, 2.0]]
    with pytest.raises(ValueError, match="mic_loc must have shape"):
        _request_shape(bad, "moving_full_direct")


def test_mix_online_moving_mono_record_strips_frame_facts(monkeypatch):
    """trace 记录只保留整轨身份，不带逐帧 frame_facts。"""
    sim_cfg = SimpleNamespace(
        rir_backend="online",
        spatial_enabled=False,
        prob_moving_rir=1.0,
        rir_normalize=True,
        early_rir_sec=0.05,
        target_type="early",
        reuse_rir=False,
    )
    dataset = _dataset(sim_cfg)

    class _Provider:
        def sample_moving_spatial_tracks(self, *args, **kwargs):
            track_set = _mono_track_set()
            facts = {
                "full": {
                    "group_uid": "g-full",
                    "cache_hit": False,
                    "frame_facts": [{"elapsed_sec": 0.1}] * TRAJECTORY_POINTS,
                },
                "direct": {
                    "group_uid": "g-full",
                    "cache_hit": False,
                    "frame_facts": [{"elapsed_sec": 0.2}] * TRAJECTORY_POINTS,
                },
            }
            return MovingSpatialRIRTrackSet(
                track_set.full, track_set.direct, track_set.plan, facts
            )

    dataset._online_rir_provider = _Provider()
    monkeypatch.setattr(
        dynamic_base, "add_moving_reverberation", lambda speech, *a, **k: speech.copy()
    )
    dataset._mix_acoustic_rir(
        np.ones((1, NUM_SAMPLES), dtype=np.float32),
        SAMPLE_RATE,
        np.random.default_rng(0),
        True,
    )
    recorded = dataset._active_acoustic_plan_record["online_rir_scene"]
    assert "frame_facts" not in recorded["track_facts"]["full"]
    assert "frame_facts" not in recorded["track_facts"]["direct"]
    assert recorded["track_facts"]["full"]["group_uid"] == "g-full"
