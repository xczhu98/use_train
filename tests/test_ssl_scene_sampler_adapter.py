"""SSL scene sampler 到真实共享 renderer 的任务适配合同。"""
from types import SimpleNamespace

import numpy as np
import pytest

from core.data.acoustic_scene_plan import AcousticActorPlan, AcousticScenePlan, NoisePlan, RIRPlan, SourceGainPlan
from core.data.simulation.array_profiles import resolve_microphone_array
from core.data.simulation.role_policy import BackgroundPolicy
from core.data.simulation.rir_provider import SpatialRIRResult, SpatialRIRTrackSet, MovingSpatialRIRTrackSet
from core.data.simulation.trajectory import TrajectoryTimebase, SourceTrajectory
from core.data.simulation.rng import SimulationRNG
from core.data.simulation.scene_context import DatasetAcousticRuntimeAdapter, SceneRenderContext
from tasks.ssl.data import spatial_adapter as adapter
from tasks.ssl.data.dynamic_dataset import SSLDynamicMixingDataset

pytestmark = pytest.mark.unit


def _plan(moving):
    array = resolve_microphone_array(custom_geometry={"positions_m": [[2, 2, 1], [2.08, 2, 1]]}, reference_channel=0)
    clock = TrajectoryTimebase(8000, 32, 16, 2) if moving else None
    actors = []
    for i, uid in enumerate(("a", "b")):
        position = (3., 2. + i, 1.)
        trajectory = None if clock is None else SourceTrajectory(np.array([[position], [(3.2, 2. + i, 1.)]]), clock)
        actors.append(AcousticActorPlan(uid, "target", i, position, trajectory))
    return AcousticScenePlan(8000, 32, False, RIRPlan.disabled(), SourceGainPlan.disabled(), NoisePlan.disabled(),
                             identity={"scene_uid": "ssl-scene"}, room={"room_uid": "room", "room_dim_m": [6, 6, 3]},
                             array=array, actors=tuple(actors), timebase=clock)


class Provider:
    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    def sample_scene_tracks(self, plan, **kwargs):
        assert plan is self.plan
        self.calls.append("static")
        full = np.zeros((2, 2, 4), np.float32)
        full[..., 0] = 1
        full[..., 3] = .2
        direct = full.copy()
        direct[..., 3] = 0
        geometry = {"positions_m": np.asarray(plan.array.positions_m).tolist()}
        provenance = {"scene_uid": "ssl-scene", "room": dict(plan.room), "source_order": ["a", "b"]}
        return SpatialRIRTrackSet(
            SpatialRIRResult(full, geometry, provenance, "fixture"),
            SpatialRIRResult(direct, geometry, provenance, "fixture"),
        )

    def sample_moving_scene_tracks(self, request, **kwargs):
        self.calls.append("moving")
        assert request.source_uids == ("a", "b")
        trajectory = np.concatenate([a.positions_m for a in request.actor_trajectories.values()], axis=1)
        full = np.zeros((2, 2, 2, 4), np.float32)
        full[..., 0] = 1
        full[..., 3] = .2
        direct = full.copy()
        direct[..., 3] = 0
        plan = {"trajectory": trajectory.tolist(), "trajectory_timebase": request.timebase.to_dict(),
                "source_loc": trajectory[0].tolist(), "sample_rate": 8000, "room_dim": [6, 6, 3],
                "source_ids": ["a", "b"], "group_uid": "group", "num_samples": 4,
                "mic_loc": np.asarray(self.plan.array.positions_m).tolist(),
                "geometry": {"positions_m": np.asarray(self.plan.array.positions_m).tolist()}}
        return MovingSpatialRIRTrackSet(full, direct, plan, {"group_uid": "group"})

    def sample_spatial_tracks(self, *args, **kwargs):
        pytest.fail("legacy static provider called")

    def sample_moving_spatial_tracks(self, *args, **kwargs):
        pytest.fail("legacy moving provider called")


@pytest.mark.parametrize("moving,backend", [(False, "online"), (True, "online"), (False, "offline")])
def test_ssl_uses_scene_sampler_and_real_renderer(monkeypatch, moving, backend):
    plan = _plan(moving)
    provider = Provider(plan)
    view = object()
    dataset = SimpleNamespace(sim_cfg=SimpleNamespace(rir_backend=backend, rir_normalize=False),
                              _simulation_view=SimpleNamespace(shared=SimpleNamespace(spatial=view)))
    context = SceneRenderContext(DatasetAcousticRuntimeAdapter(dataset))
    def prepare(**kwargs):
        assert kwargs["provider"] is provider
        assert kwargs["spatial_view"] is view
        assert kwargs["task"] == "ssl"
        assert kwargs["roles"] == ("target", "target")
        assert kwargs["moving"] == moving
        return plan
    monkeypatch.setattr(adapter.scene_sampler, "prepare_scene_plan", prepare, raising=False)
    request = adapter.SpatialSceneRenderRequest(
        context=context,
        rng=SimulationRNG.for_task(
            dataset,
            0,
            "ssl",
            rng_factory=lambda *args: np.random.default_rng(0),
        ),
        background_enabled=False,
        sources=(np.ones(32), np.ones(32)),
        source_uids=("a", "b"),
        index=0,
        segment_length=32,
        sample_rate=8000,
        provider=provider,
        moving=moving,
        trajectory_points=2,
        max_displacement_m=.25,
        reference_channel=0,
        early_boundary=1,
        timeline={},
    )
    result = adapter.SpatialSceneRenderer().render(request)
    assert provider.calls == (["moving"] if moving else ["static"])
    np.testing.assert_allclose(result.mic_positions, plan.array.positions_m)
    assert result.direct_tracks.shape == (2, 2, 32)
    assert result.provenance["scene_uid"] == "ssl-scene"
    assert not np.array_equal(result.pre_mix_scene.full_tracks, result.direct_tracks)
    assert "track_identity" in result.provenance
    if moving:
        assert result.provenance["rir"]["trajectory_timebase"] == plan.timebase.to_dict()
        np.testing.assert_allclose(result.source_positions[:, 0], plan.actors[0].trajectory.positions_m[:, 0])
    else:
        np.testing.assert_allclose(result.source_positions, [a.position_m for a in plan.actors])


def test_ssl_moving_projection_retains_frame_count_and_validates_clock():
    dataset = object.__new__(SSLDynamicMixingDataset)
    dataset.tracks, dataset.sample_rate = 2, 8000
    dataset.hop_length, dataset.n_fft, dataset.center = 8, 16, False
    plan = _plan(True)
    positions = np.concatenate([a.trajectory.positions_m for a in plan.actors], axis=1)
    provenance = {"rir": {"trajectory": positions.tolist(), "trajectory_timebase": plan.timebase.to_dict()}}
    aligned = dataset._align_moving_source_positions(positions, provenance, waveform_length=32)
    assert aligned.shape == (3, 2, 3)
    # center=False 时第一帧标签对齐窗中心 sample=8，而不是 RIR 首帧起点 sample=0。
    np.testing.assert_allclose(aligned[0], (positions[0] + positions[1]) / 2.0)
    np.testing.assert_allclose(aligned[-1], positions[-1])
    provenance["rir"]["trajectory_timebase"]["sample_rate"] = 16000
    with pytest.raises(ValueError, match="timebase"):
        dataset._align_moving_source_positions(positions, provenance, waveform_length=32)


def test_ssl_online_source_budget_is_checked_before_provider_use():
    dataset = object.__new__(SSLDynamicMixingDataset)
    dataset.tracks = 2
    dataset.trajectory_points = 1
    with pytest.raises(ValueError, match="tracks exceeds gpurir_max_sources"):
        dataset._validate_simulation_cfg(
            SimpleNamespace(
                prob_moving_rir=0.0,
                rir_backend="online",
                gpurir_max_sources=1,
            )
        )


def test_ssl_observation_renderer_resolves_and_forwards_background_spec(monkeypatch):
    """SSL 必须把空间背景策略注入 speech scene，而不是依赖 renderer 默认值。"""
    plan = _plan(False)
    dataset = SimpleNamespace(
        noise_flt={"noise": "fixture"},
        _worker_base_seed=[0],
        _worker_epoch=[0],
        sim_cfg=SimpleNamespace(
            rir_backend="online",
            spatial_enabled=True,
            array_profile="linear_2ch_80mm",
            custom_geometry=None,
            prob_point_source_background=1.0,
        ),
    )
    context = SceneRenderContext(DatasetAcousticRuntimeAdapter(dataset))
    request = adapter.SpatialSceneRenderRequest(
        context=context,
        rng=SimulationRNG.for_task(
            dataset,
            0,
            "ssl",
            rng_factory=lambda *args: np.random.default_rng(0),
        ),
        background_enabled=True,
        sources=(np.ones(32, dtype=np.float32), np.ones(32, dtype=np.float32)),
        source_uids=("a", "b"),
        index=0,
        segment_length=32,
        sample_rate=8000,
        provider=object(),
        moving=False,
        trajectory_points=1,
        max_displacement_m=0.25,
        reference_channel=0,
        early_boundary=1,
        timeline=(),
    )
    rendered = adapter.SpatialSceneRenderResult(
        pre_mix_scene=SimpleNamespace(full_tracks=np.zeros((2, 2, 32), dtype=np.float32)),
        direct_tracks=np.zeros((2, 2, 32), dtype=np.float32),
        mic_positions=np.asarray(plan.array.positions_m),
        source_positions=np.asarray([actor.position_m for actor in plan.actors]),
        provenance={},
        backend="online",
        scene_plan=plan,
    )

    class SentinelBackgroundSpec:
        policy = BackgroundPolicy.POINT_SOURCE_RIR

    sentinel = SentinelBackgroundSpec()
    resolver_calls = {}

    def resolve(**kwargs):
        resolver_calls.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "core.data.simulation.background_resolver.resolve_background_spec",
        resolve,
    )

    class Bundle:
        def __init__(self, tracks):
            self.tracks = tracks

        def require(self, role):
            return self.tracks[role]

        def replace(self, **changes):
            return Bundle(changes.get("tracks", self.tracks))

    captured = {}
    tracks = {
        "mixture": np.zeros((2, 32), dtype=np.float32),
        "target/0": np.zeros(32, dtype=np.float32),
        "target/1": np.zeros(32, dtype=np.float32),
        "source_full_rir/0": np.zeros((2, 32), dtype=np.float32),
        "source_full_rir/1": np.zeros((2, 32), dtype=np.float32),
    }

    def render_scene(*args, **kwargs):
        captured["background_spec"] = kwargs["background_spec"]
        captured["array_profile"] = args[0].sim_cfg.array_profile
        captured["custom_geometry"] = args[0].sim_cfg.custom_geometry
        captured["observation_geometry"] = kwargs["observation_geometry"]
        return SimpleNamespace(
            bundle=Bundle(tracks),
            facts={"source_volume_calibration": None},
        )

    monkeypatch.setattr(adapter, "render_speech_scene", render_scene)
    monkeypatch.setattr(adapter, "ObservationPostMixRequest", SimpleNamespace)

    class FakePostMixRenderer:
        def apply(self, post_mix_request):
            return SimpleNamespace(
                bundle=post_mix_request.scene_bundle,
                facts={
                    "layer_d_facts": {},
                    "layer_e": {},
                    "layer_f": {},
                    "final_peak_guard": {},
                    "decomposable_final": True,
                },
                trace=SimpleNamespace(stage_order=("D", "E", "F")),
            )

    monkeypatch.setattr(adapter, "CoreObservationPostMixRenderer", FakePostMixRenderer)
    adapter.ObservationPostMixRenderer().apply(request, rendered)

    assert captured["background_spec"] is sentinel
    assert captured["observation_geometry"] == {
        "positions_m": [list(position) for position in plan.array.positions_m],
    }
    assert captured["array_profile"] == "linear_2ch_80mm"
    assert captured["custom_geometry"] is None
    assert dataset.sim_cfg.array_profile == "linear_2ch_80mm"
    assert dataset.sim_cfg.custom_geometry is None
    assert resolver_calls["cfg"] is dataset.sim_cfg
    assert resolver_calls["scene_plan"] is plan
    assert resolver_calls["provider"] is request.provider
    assert resolver_calls["sample_rate"] == 8000
    assert resolver_calls["image_geometry"] == plan.array.positions_m
    assert resolver_calls["shared_scene"] is True
    assert resolver_calls["enabled"] is True
    assert isinstance(resolver_calls["rng"], np.random.Generator)


@pytest.mark.parametrize(
    ("noise", "wind", "sidechain_prob", "background_prob", "expected"),
    [
        ({}, {"wind": "fixture"}, 1.0, 0.0, True),
        ({}, {"wind": "fixture"}, 0.0, 1.0, True),
        ({}, {"wind": "fixture"}, 0.0, 0.0, False),
        ({"noise": "fixture"}, {}, 0.0, 0.0, True),
    ],
)
def test_ssl_background_gate_accepts_wind_only_resources(
    noise, wind, sidechain_prob, background_prob, expected
):
    config = SimpleNamespace(
        prob_wind_noise_sidechain=sidechain_prob,
        prob_wind_noise_background=background_prob,
    )
    assert adapter._has_spatial_background_resource(
        noise_available=bool(noise),
        wind_available=bool(wind),
        config=config,
    ) is expected
