"""验证 scene 在生成 RIR 前具有真实房间几何，并分离 L 与 T。"""
from types import SimpleNamespace

import numpy as np
import pytest

from core.data.simulation import scene_sampler
from core.data.simulation.scene_sampler import prepare_scene_plan
from core.data.simulation_configs import SimulationConfigs
from core.data.simulation_views import project_simulation_view

pytestmark = pytest.mark.unit


def _request(**overrides):
    cfg = SimulationConfigs.from_dict({
        "spatial_enabled": True, "array_profile": "linear_2ch_80mm",
        "reference_channel": 0, "rir_backend": "online", "prob_reverberation": 1.0,
    })
    values = dict(
        provider=object(), task="aec", sample_index=0, sample_rate=8000,
        num_samples=128, source_uids=("speech", "echo"),
        spatial_view=project_simulation_view(cfg, "aec").shared.spatial,
        simulation_cfg=cfg, rng=np.random.default_rng(4),
        roles=("target", "loudspeaker"),
    )
    values.update(overrides)
    return values


def test_sampler_preserves_topology_inside_room_and_waveform_length():
    request = _request()
    plan = prepare_scene_plan(**request)
    positions = np.asarray(plan.array.positions_m)
    room = np.asarray(plan.room["room_dim_m"])
    assert np.all(positions > 0) and np.all(positions < room)
    np.testing.assert_allclose(positions[1] - positions[0], [.08, 0, 0], atol=1e-6)
    assert plan.num_samples == 128
    assert plan.room["rir_num_samples"] > 128
    actors = np.asarray([actor.position_m for actor in plan.actors])
    assert np.all(actors > 0) and np.all(actors < room)
    assert not np.array_equal(actors[0], actors[1])
    assert plan.identity["geometry_scene_uid"] == plan.identity["scene_uid"]
    reordered = prepare_scene_plan(
        **_request(source_uids=("echo", "speech"), roles=("loudspeaker", "target"))
    )
    assert reordered.identity["geometry_scene_uid"] == plan.identity["geometry_scene_uid"]
    assert reordered.identity["scene_instance_uid"] != plan.identity["scene_instance_uid"]


def test_sampler_accepts_absolute_room_microphone_positions_above_topology_budget(monkeypatch):
    """房间内绝对坐标不能被误当成以阵列中心为原点的拓扑坐标。"""
    request = _request(
        sample_rate=8000,
        num_samples=128,
        source_uids=("speech",),
        roles=("target",),
        rng=np.random.default_rng(4),
    )
    request["simulation_cfg"].gpurir_room_dim_range_m = [[8.0, 8.0], [8.0, 8.0], [3.0, 3.0]]
    monkeypatch.setattr(
        scene_sampler,
        "sample_rir_scene",
        lambda *args, **kwargs: {
            "source_loc": [[4.0, 4.0, 1.5]],
            "room_dim": [8.0, 8.0, 3.0],
            "mic_loc": [[7.82, 7.9, 1.5], [7.9, 7.9, 1.5]],
            "mic_center": [7.86, 7.9, 1.5],
            "microphone_offsets": [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]],
            "array_topology": {
                "positions_m": [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]],
                "spacing_m": 0.08,
                "coordinate_system": "cartesian_xyz_m",
            },
            "coordinate_system": "cartesian_xyz_m",
            "reference_channel": 0,
            "rt60": 0.2,
            "sound_speed": 340.0,
            "diffuse_db": 15.0,
            "num_samples": 4096,
            "group_uid": "fixture-group",
        },
    )

    plan = prepare_scene_plan(**request)

    np.testing.assert_allclose(
        plan.array.positions_m,
        [[7.82, 7.9, 1.5], [7.9, 7.9, 1.5]],
    )


def test_sampler_rejects_online_actor_positions_outside_room(monkeypatch):
    request = _request(
        source_uids=("speech",),
        roles=("target",),
        rng=np.random.default_rng(4),
    )
    monkeypatch.setattr(
        scene_sampler,
        "sample_rir_scene",
        lambda *args, **kwargs: {
            "source_loc": [[8.1, 4.0, 1.5]],
            "room_dim": [8.0, 8.0, 3.0],
            "mic_loc": [[4.0, 4.0, 1.5], [4.08, 4.0, 1.5]],
            "spacing_m": 0.08,
            "coordinate_system": "cartesian_xyz_m",
            "rt60": 0.2,
            "sound_speed": 340.0,
            "diffuse_db": 15.0,
            "num_samples": 4096,
            "group_uid": "fixture-group",
        },
    )
    with pytest.raises(ValueError, match="scene actor positions"):
        prepare_scene_plan(**request)


def test_moving_actor_and_stationary_loudspeaker_share_clock():
    plan = prepare_scene_plan(**_request(moving=True, stationary_actor_uids=("echo",)))
    speech, echo = plan.actors
    assert speech.trajectory.timebase == echo.trajectory.timebase == plan.timebase
    assert plan.timebase.num_samples == plan.num_samples == 128
    assert not np.array_equal(speech.trajectory.positions_m[0], speech.trajectory.positions_m[-1])
    np.testing.assert_array_equal(echo.trajectory.positions_m[0], echo.trajectory.positions_m[-1])
    repeated = prepare_scene_plan(**_request(moving=True, stationary_actor_uids=("echo",)))
    assert repeated.to_dict() == plan.to_dict()


def test_se_moving_plan_uses_configured_physical_speed_and_binds_it_to_identity():
    request = _request(
        task="se",
        sample_rate=8000,
        num_samples=32000,
        source_uids=("speech",),
        roles=("target",),
    )
    cfg = request["simulation_cfg"]
    cfg.gpurir_room_dim_range_m = [[8.0, 8.0], [8.0, 8.0], [3.0, 3.0]]
    cfg.gpurir_rt60_range_sec = [0.2, 0.2]
    cfg.gpurir_source_distance_range_m = [0.5, 1.0]
    cfg.moving_source_speed_range_mps = [0.2, 0.2]

    plan = prepare_scene_plan(**request, moving=True)
    replay = prepare_scene_plan(**_request(
        task="se",
        sample_rate=8000,
        num_samples=32000,
        source_uids=("speech",),
        roles=("target",),
        simulation_cfg=cfg,
    ), moving=True)
    trajectory = plan.actors[0].trajectory
    duration = float(plan.timebase.frame_times[-1] - plan.timebase.frame_times[0])
    realized_speed = np.linalg.norm(
        trajectory.positions_m[-1, 0] - trajectory.positions_m[0, 0]
    ) / duration

    assert realized_speed == pytest.approx(0.2, rel=1e-6)
    assert plan.identity["motion_speeds_mps"] == {"speech": 0.2}
    assert plan.identity["moving_duration_sec"] == duration
    assert replay.to_dict() == plan.to_dict()


def test_offline_does_not_sample_room_or_positions():
    request = _request()
    online_plan = prepare_scene_plan(**request)
    cfg = request["simulation_cfg"]
    cfg.rir_backend = "offline"
    manifest = SimpleNamespace(
        scene_uid="measured-scene", actor_order=("speech", "echo"),
        room=online_plan.to_dict()["room"],
        geometry={"positions_m": np.asarray(online_plan.array.positions_m).tolist(), "reference_channel": 0},
        metadata={"sample_rate": 8000, "actor_positions_m": {
            actor.actor_uid: list(actor.position_m) for actor in online_plan.actors
        }},
    )
    class NoRandom:
        def __getattr__(self, name):
            raise AssertionError(f"offline must not sample {name}")
    request.update(provider=SimpleNamespace(scene_manifest=manifest), rng=NoRandom())
    plan = prepare_scene_plan(**request)
    assert plan.identity["scene_uid"] == "measured-scene"
    assert plan.to_dict()["room"] == online_plan.to_dict()["room"]
    assert [a.position_m for a in plan.actors] == [a.position_m for a in online_plan.actors]
    manifest.metadata["sample_rate"] = 16000
    with pytest.raises(ValueError, match="sample_rate"):
        prepare_scene_plan(**request)


def test_offline_rejects_microphone_outside_manifest_room():
    request = _request()
    online_plan = prepare_scene_plan(**request)
    request["simulation_cfg"].rir_backend = "offline"
    manifest = SimpleNamespace(
        scene_uid="out-of-room-scene", actor_order=("speech", "echo"),
        room=online_plan.to_dict()["room"],
        geometry={
            "positions_m": [[12.0, 12.0, 1.0], [12.08, 12.0, 1.0]],
            "reference_channel": 0,
        },
        metadata={"sample_rate": 8000, "actor_positions_m": {
            actor.actor_uid: list(actor.position_m) for actor in online_plan.actors
        }},
    )
    request["provider"] = SimpleNamespace(scene_manifest=manifest)
    with pytest.raises(ValueError, match="inside room_dim_m"):
        prepare_scene_plan(**request)


def test_offline_rejects_actor_outside_manifest_room():
    request = _request()
    online_plan = prepare_scene_plan(**request)
    request["simulation_cfg"].rir_backend = "offline"
    actor_positions = {
        actor.actor_uid: list(actor.position_m) for actor in online_plan.actors
    }
    actor_positions["speech"] = [12.0, 12.0, 1.0]
    manifest = SimpleNamespace(
        scene_uid="out-of-room-actor-scene", actor_order=("speech", "echo"),
        room=online_plan.to_dict()["room"],
        geometry={
            "positions_m": np.asarray(online_plan.array.positions_m).tolist(),
            "reference_channel": 0,
        },
        metadata={"sample_rate": 8000, "actor_positions_m": actor_positions},
    )
    request["provider"] = SimpleNamespace(scene_manifest=manifest)
    with pytest.raises(ValueError, match="offline scene actor positions"):
        prepare_scene_plan(**request)


def test_offline_rejects_missing_real_actor_positions():
    request = _request()
    request["simulation_cfg"].rir_backend = "offline"
    request["provider"] = SimpleNamespace(scene_manifest=SimpleNamespace(
        actor_order=("speech", "echo"), metadata={"sample_rate": 8000},
    ))
    with pytest.raises(ValueError, match="actor_positions_m"):
        prepare_scene_plan(**request)
