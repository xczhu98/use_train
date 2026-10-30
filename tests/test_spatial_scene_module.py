"""SpatialSceneRenderer deep Module 的静态、移动和 fail-fast 合同。"""

import numpy as np
import pytest
from dataclasses import replace

from core.data.simulation import (
    MovingSpatialRIRTrackSet,
    SpatialSceneExecutor,
    SpatialRIRResult,
    SpatialRIRTrackSet,
    SpatialSceneRenderer,
    SpatialSceneRequest,
)
from core.data.simulation.trajectory import TrajectoryTimebase, plan_linear_source_trajectory
from core.data.acoustic_scene_plan import AcousticActorPlan, AcousticScenePlan, NoisePlan, RIRPlan, SourceGainPlan


class StaticProvider:
    def __init__(self, tracks):
        self.tracks = tracks

    def sample_scene_tracks(self, scene_plan, **kwargs):
        assert tuple(actor.actor_uid for actor in scene_plan.actors)
        return self.tracks


class MovingProvider:
    def __init__(self, tracks):
        self.tracks = tracks

    def sample_moving_scene_tracks(self, request, **kwargs):
        assert request.source_uids == ("a", "b")
        return self.tracks


class RecordingRenderer:
    def __init__(self, result):
        self.result = result
        self.requests = []

    def render(self, request):
        self.requests.append(request)
        return self.result


def _static_tracks():
    geometry = {
        "array_profile": "fixture", "num_channels": 2,
        "positions_m": [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]],
    }
    provenance = {"scene_uid": "scene-fixture", "source_order": ["a", "b"], "room": {"room_uid": "fixture-room"}}
    full = SpatialRIRResult(np.ones((2, 2, 2), dtype=np.float32), geometry, provenance, "fixture")
    direct = SpatialRIRResult(np.array([[[0, 1], [0, 1]], [[0, 2], [0, 2]]], dtype=np.float32), geometry, provenance, "fixture")
    early = SpatialRIRResult(np.ones((2, 2, 1), dtype=np.float32), geometry, provenance, "fixture")
    return SpatialRIRTrackSet(full=full, direct=direct, early=early)


def _scene_plan(*, actor_order=("a", "b"), positions=((-0.04, 0.0, 0.0), (0.04, 0.0, 0.0))):
    from core.data.simulation.array_profiles import ArrayTopology, MicrophoneArrayProfile, ResolvedMicrophoneArray
    array = ResolvedMicrophoneArray(MicrophoneArrayProfile("fixture", ArrayTopology(positions)), 0)
    actors = tuple(
        AcousticActorPlan(uid, "target" if index == 0 else "interference", supervision_index=index,
                          position_m=(0.0, float(index + 1), 1.0))
        for index, uid in enumerate(actor_order)
    )
    return AcousticScenePlan(
        sample_rate=16000, num_samples=4, need_noise=False,
            rir=RIRPlan.disabled(), source_gain=SourceGainPlan.disabled(),
        noise=NoisePlan.disabled(), identity={"scene_uid": "scene-fixture"},
        array=array, actors=actors, room={"room_uid": "fixture-room"},
    )


@pytest.mark.unit
def test_static_renderer_hides_track_selection_and_returns_observation_contract():
    sources = np.ones((2, 4), dtype=np.float32)
    background = np.full((2, 4), 0.25, dtype=np.float32)
    request = SpatialSceneRequest(
        sources=sources, sample_rate=16000, provider=StaticProvider(_static_tracks()),
        source_uids=("a", "b"), track_kinds=("full", "direct", "early"),
        early_boundary=1, background=background, scene_plan=_scene_plan(),
    )
    result = SpatialSceneRenderer().render(request)
    assert result.observation.shape == result.mixture.shape == (2, 4)
    assert result.full_tracks.shape == result.direct_tracks.shape == result.early_tracks.shape == (2, 2, 4)
    np.testing.assert_allclose(result.observation, result.mixture + background)
    assert result.geometry["array_profile"] == "fixture"
    assert result.source_uids == ("a", "b")
    assert result.provenance["source_order"] == ["a", "b"]
    assert result.provenance["track_identity"]["track_set"]["direct"]


@pytest.mark.unit
def test_spatial_scene_executor_delegates_typed_request_without_copying_result():
    request = SpatialSceneRequest(
        sources=np.ones((2, 4), dtype=np.float32), sample_rate=16000,
        provider=StaticProvider(_static_tracks()), source_uids=("a", "b"),
        scene_plan=_scene_plan(),
    )
    expected = SpatialSceneRenderer().render(request)
    renderer = RecordingRenderer(expected)
    result = SpatialSceneExecutor(renderer=renderer).execute(request)
    assert len(renderer.requests) == 1
    assert renderer.requests[0] is request
    assert result is expected


@pytest.mark.unit
def test_spatial_scene_executor_rejects_invalid_request_and_renderer_result():
    with pytest.raises(TypeError, match="SpatialSceneRequest"):
        SpatialSceneExecutor().execute(object())
    request = SpatialSceneRequest(
        sources=np.ones((2, 4), dtype=np.float32), sample_rate=16000,
        provider=StaticProvider(_static_tracks()), source_uids=("a", "b"),
        scene_plan=_scene_plan(),
    )
    with pytest.raises(TypeError, match="SpatialSceneResult"):
        SpatialSceneExecutor(renderer=RecordingRenderer(object())).execute(request)


@pytest.mark.unit
def test_renderer_records_scene_plan_identity_and_actor_order():
    request = SpatialSceneRequest(
        sources=np.ones((2, 4), dtype=np.float32), sample_rate=16000,
        provider=StaticProvider(_static_tracks()), source_uids=("a", "b"),
        scene_plan=_scene_plan(),
    )
    result = SpatialSceneRenderer().render(request)
    assert result.provenance["scene_uid"] == "scene-fixture"
    assert result.provenance["actor_order"] == ["a", "b"]
    assert result.provenance["geometry_identity"]["topology"]["positions_m"] == [
        [-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]
    ]


@pytest.mark.unit
def test_renderer_rejects_scene_plan_actor_order_mismatch():
    with pytest.raises(ValueError, match="actor order"):
        SpatialSceneRequest(
            sources=np.ones((2, 4), dtype=np.float32), sample_rate=16000,
            provider=StaticProvider(_static_tracks()), source_uids=("a", "b"),
            scene_plan=_scene_plan(actor_order=("b", "a")),
        )


@pytest.mark.unit
def test_renderer_rejects_scene_plan_geometry_mismatch():
    plan = _scene_plan(positions=((-0.2, 0.0, 0.0), (0.2, 0.0, 0.0)))
    with pytest.raises(ValueError, match="geometry identity"):
        SpatialSceneRenderer().render(SpatialSceneRequest(
            sources=np.ones((2, 4), dtype=np.float32), sample_rate=16000,
            provider=StaticProvider(_static_tracks()), source_uids=("a", "b"),
            scene_plan=plan,
        ))


@pytest.mark.unit
def test_request_requires_scene_identity_and_resolved_array():
    base = _scene_plan()
    with pytest.raises(ValueError, match="scene_uid"):
        SpatialSceneRequest(
            sources=np.ones((2, 4), dtype=np.float32), sample_rate=16000,
            provider=StaticProvider(_static_tracks()), source_uids=("a", "b"),
            scene_plan=AcousticScenePlan(
                sample_rate=16000, num_samples=4, need_noise=False,
                rir=RIRPlan.disabled(), source_gain=SourceGainPlan.disabled(),
                noise=NoisePlan.disabled(), identity={}, array=base.array,
                actors=base.actors,
            ),
        )
    with pytest.raises(ValueError, match="resolved microphone array"):
        SpatialSceneRequest(
            sources=np.ones((2, 4), dtype=np.float32), sample_rate=16000,
            provider=StaticProvider(_static_tracks()), source_uids=("a", "b"),
            scene_plan=AcousticScenePlan(
                sample_rate=16000, num_samples=4, need_noise=False,
                rir=RIRPlan.disabled(), source_gain=SourceGainPlan.disabled(),
                noise=NoisePlan.disabled(), identity={"scene_uid": "scene-fixture"},
                actors=base.actors,
            ),
        )


@pytest.mark.unit
def test_renderer_rejects_provider_scene_identity_mismatch():
    tracks = _static_tracks()
    tracks.full.provenance["scene_uid"] = "other-scene"
    with pytest.raises(ValueError, match="scene identity"):
        SpatialSceneRenderer().render(SpatialSceneRequest(
            sources=np.ones((2, 4), dtype=np.float32), sample_rate=16000,
            provider=StaticProvider(tracks), source_uids=("a", "b"),
            scene_plan=_scene_plan(),
        ))


@pytest.mark.unit
def test_static_renderer_fails_when_requested_direct_track_is_missing():
    full = SpatialRIRResult(np.ones((1, 1, 1), dtype=np.float32), {"array": "x"}, {"scene": "x"}, "fixture")
    request = SpatialSceneRequest(
        sources=np.ones((1, 2), dtype=np.float32), sample_rate=16000,
        provider=StaticProvider(SpatialRIRTrackSet(full=full)), source_uids=("a",),
        track_kinds=("full", "direct"),
        scene_plan=replace(_scene_plan(actor_order=("a",), positions=((0., 0., 0.),)), num_samples=2),
    )
    with pytest.raises(ValueError, match="direct track"):
        SpatialSceneRenderer().render(request)


@pytest.mark.unit
def test_moving_renderer_uses_provider_timebase_and_keeps_direct_track():
    timebase = TrajectoryTimebase(16000, 8, 4, 2)
    trajectory = plan_linear_source_trajectory(
        np.array([[0.2, 0.2, 1.0], [0.3, 0.2, 1.0]], dtype=np.float32),
        np.array([[0.3, 0.2, 1.0], [0.4, 0.2, 1.0]], dtype=np.float32), timebase,
    )
    plan = {
        "trajectory": trajectory.positions_m.tolist(),
        "trajectory_timebase": timebase.to_dict(),
        "geometry": {"array_profile": "fixture", "positions_m": [[0., 0., 0.], [0.1, 0., 0.]]},
        "room": {"room_uid": "fixture-room"},
        "scene_uid": "scene-fixture",
        "room_dim": [4.0, 4.0, 3.0],
        "mic_loc": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        "source_loc": trajectory.positions_m[0].tolist(),
        "sample_rate": 16000,
        "num_samples": 2,
    }
    tracks = MovingSpatialRIRTrackSet(
        full=np.ones((2, 2, 2, 2), dtype=np.float32),
        direct=np.concatenate([np.zeros((2, 2, 2, 1), dtype=np.float32), np.ones((2, 2, 2, 1), dtype=np.float32)], axis=-1),
        plan=plan, track_facts={"group_uid": "moving-1"},
    )
    scene = _scene_plan(positions=((0., 0., 0.), (0.1, 0., 0.)))
    from core.data.simulation.trajectory import SourceTrajectory
    scene = replace(scene, num_samples=8, timebase=timebase, actors=tuple(
        replace(actor, position_m=tuple(trajectory.positions_m[0, i]),
                trajectory=SourceTrajectory(trajectory.positions_m[:, i:i+1], timebase))
        for i, actor in enumerate(scene.actors)
    ))
    request = SpatialSceneRequest(
        sources=np.ones((2, 8), dtype=np.float32), sample_rate=16000,
        provider=MovingProvider(tracks), source_uids=("a", "b"), mode="moving",
        trajectory=trajectory, track_kinds=("full", "direct", "early"),
        early_boundary=1, num_samples=8, scene_plan=scene,
    )
    result = SpatialSceneRenderer().render(request)
    assert result.trajectory == trajectory
    assert result.timebase == timebase
    assert result.full_tracks.shape == (2, 2, 8)
    assert result.direct_tracks is not None


@pytest.mark.unit
def test_request_rejects_static_trajectory_and_duplicate_source_order():
    timebase = TrajectoryTimebase(16000, 4, 2, 2)
    trajectory = plan_linear_source_trajectory(
        np.zeros((1, 3), dtype=np.float32), np.ones((1, 3), dtype=np.float32), timebase,
    )
    with pytest.raises(ValueError, match="static scene"):
        SpatialSceneRequest(np.ones((1, 4)), 16000, StaticProvider(_static_tracks()), source_uids=("a",), trajectory=trajectory)
    with pytest.raises(ValueError, match="unique"):
        SpatialSceneRequest(np.ones((2, 4)), 16000, StaticProvider(_static_tracks()), source_uids=("a", "a"))


@pytest.mark.unit
def test_renderer_rejects_legacy_callable_instead_of_sampling_new_scene():
    def legacy(*args, **kwargs):
        raise AssertionError("must not call legacy provider")
    with pytest.raises(TypeError, match="sample_scene_tracks"):
        SpatialSceneRenderer().render(SpatialSceneRequest(
            np.ones((2, 4)), 16000, legacy, source_uids=("a", "b"), scene_plan=_scene_plan(),
        ))
