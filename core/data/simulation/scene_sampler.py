"""在 provider 之前固定 room、array 和 actor 几何的 scene sampler。"""
from __future__ import annotations

from typing import Any
from dataclasses import replace
import hashlib
import json
import numpy as np

from core.data.acoustic_scene_plan import (
    AcousticActorPlan, AcousticScenePlan, NoisePlan, RIRPlan, SourceGainPlan,
)
from core.data.simulation.rir_geometry import sample_rir_scene
from core.data.simulation.trajectory import (
    SourceTrajectory,
    TrajectoryTimebase,
    sample_source_trajectories_by_speed,
)
from core.data.simulation.rir_config import OnlineRIRConfig


def _validate_positions(
    positions_m: Any,
    *,
    name: str,
) -> np.ndarray:
    """校验静态位置或轨迹点的形状与有限性。"""
    try:
        positions = np.asarray(positions_m, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape [..., 3]") from exc
    if positions.ndim < 2 or positions.shape[-1] != 3 or positions.shape[-2] < 1:
        raise ValueError(f"{name} must have shape [..., 3]")
    if not np.isfinite(positions).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return positions


def _validate_room_positions(
    positions_m: Any,
    room_dim_m: Any,
    *,
    name: str,
) -> np.ndarray:
    """校验 actor 的静态位置或轨迹点严格位于房间内部。"""
    try:
        room = np.asarray(room_dim_m, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("room_dim_m must have shape [3]") from exc
    if room.shape != (3,) or not np.isfinite(room).all() or np.any(room <= 0.0):
        raise ValueError("room_dim_m must have three positive finite dimensions")
    positions = _validate_positions(positions_m, name=name)
    if np.any(positions <= 0.0) or np.any(positions >= room):
        raise ValueError(f"{name} must lie strictly inside room_dim_m")
    return positions


def _scene_instance_uid(
    geometry_scene_uid: str,
    source_uids: tuple[str, ...],
    roles: tuple[str, ...],
    positions_m: np.ndarray,
) -> str:
    payload = {
        "geometry_scene_uid": str(geometry_scene_uid),
        "actors": [
            {
                "source_uid": uid,
                "role": role,
                "position_m": [float(value) for value in position],
            }
            for uid, role, position in zip(source_uids, roles, positions_m, strict=True)
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()[:24]
    return f"{geometry_scene_uid}/instance/{digest}"


def sample_static_scene_plan(
    *,
    task: str,
    sample_index: int,
    sample_rate: int,
    num_samples: int,
    source_uids: tuple[str, ...],
    spatial_view: Any,
    online_config: Any,
    rng: np.random.Generator,
    roles: tuple[str, ...] | None = None,
    scene_uid: str | None = None,
) -> tuple[AcousticScenePlan, dict[str, Any]]:
    """复用既有几何采样器，并把结果冻结为 scene plan。"""
    source_uids = tuple(source_uids)
    if not source_uids or len(set(source_uids)) != len(source_uids):
        raise ValueError("source_uids must be non-empty and unique")
    if roles is None:
        roles = tuple("target" if i == 0 else "interference" for i in range(len(source_uids)))
    if len(roles) != len(source_uids):
        raise ValueError("roles must match source_uids")
    rir_cfg = (
        online_config
        if isinstance(online_config, OnlineRIRConfig)
        else OnlineRIRConfig.from_simulation(online_config)
    )
    raw = sample_rir_scene(
        rir_cfg,
        sample_rate,
        len(source_uids),
        rng,
        use_config_microphone_array=True,
    )
    positions = _validate_room_positions(
        raw["source_loc"], raw["room_dim"], name="scene actor positions"
    )
    if positions.shape != (len(source_uids), 3):
        raise ValueError("scene sampler returned invalid actor positions")
    from core.data.simulation.array_profiles import resolve_scene_microphone_array

    resolved_positions = getattr(spatial_view, "microphone_positions", None)
    if resolved_positions is None:
        raise ValueError("scene sampler requires resolved microphone positions")
    # topology 是相对坐标；scene 中保存同一拓扑在房间内的真实位姿。
    array = resolve_scene_microphone_array(
        positions_m=raw["mic_loc"],
        room_dim_m=raw["room_dim"],
        spacing_m=raw.get("spacing_m"),
        coordinate_system=raw.get("coordinate_system", "cartesian_xyz_m"),
        reference_channel=int(spatial_view.reference_channel),
    )
    actors = tuple(
        AcousticActorPlan(
            actor_uid=uid,
            role=role,
            supervision_index=i,
            position_m=tuple(float(v) for v in positions[i]),
        )
        for i, (uid, role) in enumerate(zip(source_uids, roles))
    )
    geometry_scene_uid = scene_uid or f"{task}/sample/{int(sample_index)}/group/{raw['group_uid']}"
    scene_instance_uid = _scene_instance_uid(
        geometry_scene_uid, source_uids, roles, positions,
    )
    room = {
        "room_dim_m": list(raw["room_dim"]),
        "rt60": float(raw["rt60"]),
        "sound_speed": float(raw["sound_speed"]),
        "coordinate_system": str(raw["coordinate_system"]),
        "diffuse_db": float(raw["diffuse_db"]),
        "rir_num_samples": int(raw["num_samples"]),
    }
    room["room_uid"] = "room-" + hashlib.sha256(
        json.dumps(room, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()[:24]
    plan = AcousticScenePlan(
        sample_rate=int(sample_rate),
        num_samples=int(num_samples),
        need_noise=False,
        rir=RIRPlan.disabled(),
        source_gain=SourceGainPlan.disabled(),
        noise=NoisePlan.disabled(),
        identity={
            "scene_uid": geometry_scene_uid,
            "geometry_scene_uid": geometry_scene_uid,
            "scene_instance_uid": scene_instance_uid,
            "task": task,
            "sample_index": int(sample_index),
        },
        room=room,
        array=array,
        actors=actors,
    )
    raw = dict(raw)
    raw["reference_channel"] = int(spatial_view.reference_channel)
    raw["scene_uid"] = geometry_scene_uid
    raw["geometry_scene_uid"] = geometry_scene_uid
    raw["scene_instance_uid"] = scene_instance_uid
    raw["source_ids"] = list(source_uids)
    return plan, raw


def prepare_scene_plan(
    *, provider, task, sample_index, sample_rate, num_samples, source_uids,
    spatial_view, simulation_cfg, rng, moving=False, trajectory_points=4,
    max_displacement_m=0.25, roles=None, stationary_actor_uids=(),
) -> AcousticScenePlan:
    """online 先采样完整场景；offline 只读取已有场景，绝不随机覆盖资产几何。

    offline 几何完全物化在 manifest 中：``moving`` 请求参数不参与抽样，
    scene 是否为 moving 由 manifest 的 ``metadata.moving`` 决定；moving
    manifest 的逐 actor 轨迹原样进入 plan，不做任何运行时重采样。
    """
    source_uids = tuple(source_uids)
    # scene-level providers supply explicit full/direct/early assets; the
    # provider scene is authoritative and must not be re-gated by the legacy
    # random reverberation probability.
    if simulation_cfg.rir_backend == "offline":
        manifest = getattr(provider, "scene_manifest", None)
        if manifest is None:
            raise ValueError("offline scene sampling requires typed scene manifest")
        resolver = getattr(provider, "resolve_scene_manifest", None)
        if callable(resolver):
            manifest = resolver(
                source_uids=source_uids, sample_index=sample_index,
                sample_rate=sample_rate,
                geometry={"positions_m": spatial_view.microphone_positions},
                roles=roles,
            )
        metadata = manifest.metadata
        if metadata.get("sample_rate") != sample_rate:
            raise ValueError("offline scene manifest sample_rate must match request")
        if not source_uids or len(set(source_uids)) != len(source_uids) or not set(source_uids).issubset(manifest.actor_order):
            raise ValueError("offline scene source_uids must select unique manifest actors")
        actor_positions = metadata.get("actor_positions_m")
        if not isinstance(actor_positions, dict) or not set(source_uids).issubset(actor_positions):
            raise ValueError("offline scene manifest must declare actor_positions_m for all actors")
        room_dim = manifest.room.get("room_dim_m")
        from core.data.simulation.array_profiles import (
            resolve_microphone_array,
            resolve_scene_microphone_array,
        )
        topology = manifest.geometry.get("topology", {})
        if room_dim is None:
            # schema v1 的 geometry 是阵列中心相对拓扑，按 10 m geometry budget
            # 解析；不能把缺失 room_dim_m 的相对坐标误当作房间绝对坐标。
            array = resolve_microphone_array(
                custom_geometry={
                    "positions_m": manifest.geometry["positions_m"],
                    "spacing_m": manifest.geometry.get("spacing_m", topology.get("spacing_m")),
                    "coordinate_system": manifest.geometry.get(
                        "coordinate_system", topology.get("coordinate_system", "cartesian_xyz_m")
                    ),
                },
                reference_channel=manifest.geometry.get("reference_channel", spatial_view.reference_channel),
            )
        else:
            array = resolve_scene_microphone_array(
                positions_m=manifest.geometry["positions_m"],
                room_dim_m=room_dim,
                spacing_m=manifest.geometry.get("spacing_m", topology.get("spacing_m")),
                coordinate_system=manifest.geometry.get(
                    "coordinate_system", topology.get("coordinate_system", "cartesian_xyz_m")
                ),
                reference_channel=manifest.geometry.get("reference_channel", spatial_view.reference_channel),
            )
        desired = np.asarray(spatial_view.microphone_positions, dtype=np.float64)
        actual = np.asarray(array.positions_m, dtype=np.float64)
        if actual.shape != desired.shape or not np.allclose(
            actual - actual[0], desired - desired[0], rtol=0, atol=1e-6
        ) or array.reference_channel != spatial_view.reference_channel:
            raise ValueError("offline scene microphone topology/reference mismatch")
        roles = roles or tuple("target" if i == 0 else "interference" for i in range(len(source_uids)))
        if len(roles) != len(source_uids):
            raise ValueError("roles must match source_uids")
        geometry_scene_uid = str(metadata.get("geometry_scene_uid", manifest.scene_uid))
        ordered_positions = _validate_positions(
            [actor_positions[uid] for uid in source_uids],
            name="offline scene actor positions",
        )
        if room_dim is not None:
            ordered_positions = _validate_room_positions(
                ordered_positions, room_dim, name="offline scene actor positions"
            )
        scene_instance_uid = _scene_instance_uid(
            geometry_scene_uid, source_uids, tuple(roles), ordered_positions,
        )
        plan = AcousticScenePlan(
            sample_rate=sample_rate, num_samples=num_samples, need_noise=False,
            rir=RIRPlan.disabled(), source_gain=SourceGainPlan.disabled(), noise=NoisePlan.disabled(),
            identity={
                "scene_uid": manifest.scene_uid,
                "geometry_scene_uid": geometry_scene_uid,
                "scene_instance_uid": scene_instance_uid,
                "task": task,
                "sample_index": sample_index,
            },
            room=manifest.room, array=array,
            actors=tuple(AcousticActorPlan(
                actor_uid=uid, role=roles[i], position_m=tuple(float(v) for v in ordered_positions[i]),
                supervision_index=None if roles[i] in {"loudspeaker", "latent_target"} else i,
            ) for i, uid in enumerate(source_uids)),
        )
        if metadata.get("moving") is True:
            plan = _apply_manifest_trajectories(plan, metadata, num_samples)
        return plan
    if simulation_cfg.rir_backend != "online":
        raise ValueError("scene sampling requires offline or online RIR backend")
    plan, raw = sample_static_scene_plan(
        task=task, sample_index=sample_index, sample_rate=sample_rate, num_samples=num_samples,
        source_uids=source_uids, spatial_view=spatial_view, online_config=simulation_cfg,
        rng=rng, roles=roles,
    )
    if not moving:
        return plan
    if type(trajectory_points) is not int or not 2 <= trajectory_points <= min(128, num_samples):
        raise ValueError("trajectory_points must be in [2,min(128,num_samples)]")
    if not set(stationary_actor_uids).issubset(source_uids):
        raise ValueError("stationary actors must belong to scene")
    cfg = OnlineRIRConfig.from_simulation(simulation_cfg)
    clock = TrajectoryTimebase(sample_rate, num_samples, num_samples // trajectory_points, trajectory_points)
    start = np.asarray(raw["source_loc"], dtype=np.float64)
    motion_facts = {}
    if task == "se":
        stationary_indices = tuple(
            index for index, uid in enumerate(source_uids) if uid in stationary_actor_uids
        )
        trajectory, sampled_speeds = sample_source_trajectories_by_speed(
            start,
            room_dim_m=np.asarray(raw["room_dim"], dtype=np.float64),
            timebase=clock,
            speed_range_mps=cfg.moving_source_speed_range_mps,
            rng=rng,
            wall_margin_m=cfg.margin,
            min_separation_m=cfg.min_separation,
            source_height_range_m=cfg.source_height,
            stationary_indices=stationary_indices,
        )
        positions = _validate_room_positions(
            trajectory.positions_m,
            raw["room_dim"],
            name="scene actor trajectory positions",
        )
        motion_facts = {
            "motion_speeds_mps": {
                uid: speed for uid, speed in zip(source_uids, sampled_speeds, strict=True)
            },
            "moving_duration_sec": float(clock.frame_times[-1] - clock.frame_times[0]),
        }
        digest_payload = {
            "positions_m": positions.tolist(),
            "timebase": clock.to_dict(),
            **motion_facts,
        }
        digest = hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()[:24]
    else:
        if isinstance(max_displacement_m, bool) or not np.isfinite(max_displacement_m) or max_displacement_m <= 0:
            raise ValueError("max_displacement_m must be positive and finite")
        room = np.asarray(raw["room_dim"], dtype=np.float64)
        alpha = np.linspace(0, 1, trajectory_points)[:, None, None]
        for _ in range(100):
            end = start.copy()
            for index, uid in enumerate(source_uids):
                if uid in stationary_actor_uids:
                    continue
                lower = np.maximum(cfg.margin - start[index], -max_displacement_m)
                upper = np.minimum(room - cfg.margin - start[index], max_displacement_m)
                lower[2] = max(lower[2], cfg.source_height[0] - start[index, 2])
                upper[2] = min(upper[2], cfg.source_height[1] - start[index, 2])
                end[index] += rng.uniform(lower, upper)
            delta = end - start
            valid = True
            for left in range(len(start)):
                for right in range(left):
                    separation = start[left] - start[right]
                    velocity = delta[left] - delta[right]
                    closest = np.clip(
                        -np.dot(separation, velocity)
                        / max(np.dot(velocity, velocity), 1e-20),
                        0,
                        1,
                    )
                    if np.linalg.norm(separation + closest * velocity) < cfg.min_separation:
                        valid = False
            if valid:
                break
        else:
            raise ValueError("cannot sample independent trajectories with minimum source separation")
        positions = _validate_room_positions(
            start[None] * (1 - alpha) + end[None] * alpha,
            room,
            name="scene actor trajectory positions",
        )
        digest = hashlib.sha256(np.asarray(positions, dtype="<f4").tobytes()).hexdigest()[:24]
    actors = tuple(
        replace(actor, trajectory=SourceTrajectory(positions[:, index : index + 1], clock))
        for index, actor in enumerate(plan.actors)
    )
    return replace(
        plan,
        actors=actors,
        timebase=clock,
        identity={
            **plan.identity,
            **motion_facts,
            "scene_uid": f"{plan.identity['scene_uid']}/moving/{digest}",
            "scene_instance_uid": f"{plan.identity.get('scene_instance_uid', plan.identity['scene_uid'])}/moving/{digest}",
        },
    )


def _apply_manifest_trajectories(plan, metadata, num_samples):
    """把 moving manifest 的逐 actor 轨迹原样冻结进 offline scene plan。

    manifest load 时已校验轨迹形状、房间边界与内容哈希；这里只复核与当前
    请求的采样率/长度兼容性，不做任何重采样。scene_uid 保持 manifest 值，
    以便 provider 按 scene_uid 解析资产；仅 scene_instance_uid 追加
    moving digest。
    """
    timebase_raw = metadata["trajectory_timebase"]
    clock = TrajectoryTimebase(
        int(timebase_raw["sample_rate"]), int(timebase_raw["num_samples"]),
        int(timebase_raw["hop_samples"]), int(timebase_raw["frame_count"]),
        int(timebase_raw.get("start_sample", 0)),
    )
    if clock.sample_rate != plan.sample_rate:
        raise ValueError(
            "offline moving scene trajectory sample_rate does not match request"
        )
    if clock.start_sample + (clock.frame_count - 1) * clock.hop_samples >= num_samples:
        raise ValueError(
            "offline moving scene trajectory exceeds requested waveform length"
        )
    trajectories = metadata["actor_trajectories"]
    actors = tuple(
        replace(
            actor,
            trajectory=SourceTrajectory(
                np.asarray(trajectories[actor.actor_uid], dtype=np.float32), clock
            ),
        )
        for actor in plan.actors
    )
    positions = _validate_room_positions(
        np.concatenate(
            [actor.trajectory.positions_m for actor in actors], axis=1
        ),
        plan.room.get("room_dim_m"),
        name="offline scene actor trajectory positions",
    )
    digest = hashlib.sha256(np.asarray(positions, dtype="<f4").tobytes()).hexdigest()[:24]
    return replace(
        plan,
        actors=actors,
        timebase=clock,
        identity={
            **plan.identity,
            "scene_instance_uid": f"{plan.identity.get('scene_instance_uid', plan.identity['scene_uid'])}/moving/{digest}",
        },
    )
