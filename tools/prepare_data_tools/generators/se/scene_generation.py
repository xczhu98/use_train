"""多麦克风 scene-level offline RIR 规划与生成辅助。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from core.data.simulation.array_profiles import resolve_microphone_array
from core.data.simulation.gpurir_backend import (
    generate_moving_scene_rir_tracks,
    generate_rir_tracks,
)
from core.data.simulation.trajectory import TrajectoryTimebase
from tools.prepare_data_tools.generators.se.rir_config import MIN_SOURCE_MICROPHONE_DISTANCE, RIRConfig
from tools.prepare_data_tools.generators.se.rir_scene import (
    advance_with_wall_reflections,
    sample_room_rt60,
    resolve_scene_seed,
)


@dataclass(frozen=True)
class SpatialScenePlan:
    """生成器侧不可变 scene 计划；C 由 resolved array 推导。"""

    scene_uid: str
    sample_rate: int
    rir_num_samples: int
    room: dict[str, Any]
    array: dict[str, Any]
    actors: tuple[dict[str, Any], ...]
    generator_version: str
    moving: bool = False
    trajectory_timebase: dict[str, Any] | None = None

    @property
    def actor_order(self) -> tuple[str, ...]:
        return tuple(actor["actor_uid"] for actor in self.actors)

    @property
    def channels(self) -> int:
        return len(self.array["positions_m"])

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "scene_uid": self.scene_uid,
            "sample_rate": self.sample_rate,
            "rir_num_samples": self.rir_num_samples,
            "room": self.room,
            "array": self.array,
            "actors": list(self.actors),
            "generator_version": self.generator_version,
        }
        if self.moving:
            payload["moving"] = True
            payload["trajectory_timebase"] = dict(self.trajectory_timebase or {})
        return payload


def resolve_spatial_array(config: RIRConfig):
    if config.array_profile is None and config.custom_geometry is None:
        return None
    return resolve_microphone_array(
        array_profile=config.array_profile,
        custom_geometry=config.custom_geometry,
        reference_channel=config.reference_channel,
    )


def _sample_actor_positions(config: RIRConfig, room: np.ndarray, center: np.ndarray, offsets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positions: list[np.ndarray] = []
    lower = np.full(3, config.source_wall_margin_m, dtype=np.float64)
    upper = room - config.source_wall_margin_m
    lower[2] = max(lower[2], (config.source_height_range_m or (0.2, room[2] - config.source_wall_margin_m))[0])
    upper[2] = min(upper[2], (config.source_height_range_m or (0.2, room[2] - config.source_wall_margin_m))[1])
    distance = config.dist_range
    for _ in range(len(config.actor_roles)):
        for _attempt in range(500):
            direction = rng.normal(size=3)
            direction /= max(np.linalg.norm(direction), 1e-12)
            position = center + direction * rng.uniform(*distance)
            if np.any(position <= lower) or np.any(position >= upper):
                continue
            if all(np.linalg.norm(position - previous) >= config.source_source_min_separation_m for previous in positions):
                positions.append(position)
                break
        else:
            raise ValueError("cannot place all scene actors under configured room/distance constraints")
    return np.asarray(positions, dtype=np.float64)


def sample_spatial_scene(config: RIRConfig, index: int, *, generator_version: str) -> SpatialScenePlan:
    """一次采样 room、absolute array geometry 和所有 actor positions。"""
    array = resolve_spatial_array(config)
    if array is None or array.channels <= 1:
        raise ValueError("sample_spatial_scene requires resolved microphone array with C>1")
    if config.moving:
        raise ValueError("offline scene generator currently emits static spatial scenes; use online moving RIR")
    rng = np.random.default_rng(resolve_scene_seed(config, index))
    room_list, rt60 = sample_room_rt60(config, rng)
    room = np.asarray(room_list, dtype=np.float64)
    offsets = np.asarray(array.positions_m, dtype=np.float64)
    low = np.full(3, config.microphone_wall_margin_m) - offsets.min(axis=0)
    high = room - config.microphone_wall_margin_m - offsets.max(axis=0)
    if np.any(low >= high):
        raise ValueError("microphone array cannot fit inside sampled room")
    center = np.asarray([rng.uniform(low[i], high[i]) for i in range(3)], dtype=np.float64)
    mic_positions = center[None, :] + offsets
    actor_positions = _sample_actor_positions(config, room, center, offsets, rng)
    room_payload = {
        "room_uid": "room-" + hashlib.sha256(json.dumps({"room_dim_m": room_list, "rt60": rt60}, sort_keys=True).encode()).hexdigest()[:24],
        "room_dim_m": [float(value) for value in room_list],
        "rt60": float(rt60),
        "sound_speed": float(config.c),
        "diffuse_db": float(config.backend_options.gpurir.diffuse_transition_db),
        "rir_num_samples": int(config.n_samples),
        "early_rir_sec": float(config.early_rir_sec),
    }
    array_payload = {
        "array_profile": config.array_profile,
        "positions_m": mic_positions.tolist(),
        "reference_channel": int(config.reference_channel),
        "channel_order": list(config.channel_order or tuple(f"mic_{i}" for i in range(array.channels))),
        "coordinate_system": array.profile.topology.coordinate_system,
    }
    if len(array_payload["channel_order"]) != array.channels:
        raise ValueError("channel_order length must match resolved microphone channels")
    actors = tuple({
        "actor_uid": role,
        "role": role,
        "position_m": actor_positions[i].tolist(),
    } for i, role in enumerate(config.actor_roles))
    identity_payload = {
        "sample_rate": config.fs, "room": room_payload, "array": array_payload,
        "actors": actors, "generator_version": generator_version,
    }
    scene_uid = "scene-" + hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()[:32]
    return SpatialScenePlan(
        scene_uid=scene_uid, sample_rate=int(config.fs), rir_num_samples=int(config.n_samples),
        room=room_payload, array=array_payload, actors=actors,
        generator_version=generator_version,
    )


def build_gpurir_plan(scene: SpatialScenePlan) -> dict[str, Any]:
    """把 scene 计划投影为 core gpuRIR backend 的 canonical plan。"""
    return {
        "version": 3,
        "backend": "gpurir",
        "group_uid": scene.scene_uid,
        "scene_uid": scene.scene_uid,
        "sample_rate": scene.sample_rate,
        "num_samples": scene.rir_num_samples,
        "room_dim": scene.room["room_dim_m"],
        "rt60": scene.room["rt60"],
        "diffuse_db": scene.room["diffuse_db"],
        "sound_speed": scene.room["sound_speed"],
        "mic_loc": scene.array["positions_m"],
        "source_loc": [actor["position_m"] for actor in scene.actors],
        "source_ids": list(scene.actor_order),
        "actor_roles": [actor["role"] for actor in scene.actors],
        "array": scene.array,
        "room": scene.room,
        "reference_channel": scene.array["reference_channel"],
        "num_channels": scene.channels,
        "speaker_pattern": "omni",
        "microphone_pattern": "omni",
    }


def generate_gpurir_scene(scene: SpatialScenePlan, gpu_rir: Any) -> dict[str, Any]:
    """生成同一 scene 的 full/direct/early [A,C,L] track bank。"""
    plan = build_gpurir_plan(scene)
    generated = generate_rir_tracks(plan, gpu_rir)
    full = np.asarray(generated["full"][0], dtype="<f4")
    direct = np.asarray(generated["direct"][0], dtype="<f4")
    # Early track 以每个 source×microphone 的 direct-path peak 为锚点，
    # 避免固定绝对样本窗截掉远距离 actor 的直达声。
    early = np.zeros_like(full)
    window_samples = max(1, int(round(scene.room.get("early_rir_sec", 0.05) * scene.sample_rate)))
    direct_peak = np.argmax(np.abs(direct), axis=-1)
    for actor_index in range(full.shape[0]):
        for channel_index in range(full.shape[1]):
            start = int(direct_peak[actor_index, channel_index])
            stop = min(full.shape[-1], start + window_samples)
            early[actor_index, channel_index, start:stop] = full[actor_index, channel_index, start:stop]
    return {
        "scene": scene.to_dict(),
        "full": full,
        "direct": direct,
        "early": early,
        "facts": {
            "scene_uid": scene.scene_uid,
            "actor_order": list(scene.actor_order),
            "room": scene.room,
            "array": scene.array,
            "sample_rate": scene.sample_rate,
            "generator_version": scene.generator_version,
            "track_kind": "scene_static",
            "early_policy": "direct_peak_plus_duration",
            "early_window_samples": window_samples,
        },
    }


def sample_spatial_moving_scene(config: RIRConfig, index: int, *, generator_version: str) -> SpatialScenePlan:
    """采样 moving scene：room、array、actor 起点与逐 actor 独立轨迹。

    所有 actor 共享同一 trajectory timebase（帧数与 hop 一致），与在线
    moving scene 合同对齐；轨迹逐 actor 独立采样，使用与 legacy mono moving
    相同的墙反射步进语义。endpoint 不额外约束，移动距离由 speed_range 与
    总时长决定。
    """
    array = resolve_spatial_array(config)
    if array is None or array.channels <= 1:
        raise ValueError("moving scene generator requires array_profile/custom_geometry with C>1")
    rng = np.random.default_rng(resolve_scene_seed(config, index))
    room_list, rt60 = sample_room_rt60(config, rng)
    room = np.asarray(room_list, dtype=np.float64)
    offsets = np.asarray(array.positions_m, dtype=np.float64)
    low = np.full(3, config.microphone_wall_margin_m) - offsets.min(axis=0)
    high = room - config.microphone_wall_margin_m - offsets.max(axis=0)
    if np.any(low >= high):
        raise ValueError("microphone array cannot fit inside sampled room")
    center = np.asarray([rng.uniform(low[i], high[i]) for i in range(3)], dtype=np.float64)
    mic_positions = center[None, :] + offsets
    actor_starts = _sample_actor_positions(config, room, center, offsets, rng)

    segment_lo, segment_hi = config.n_segments_range
    frame_count = int(rng.integers(int(segment_lo), int(segment_hi) + 1))
    hop = max(1, int(round(float(config.segment_sec) * int(config.fs))))
    clock = TrajectoryTimebase(int(config.fs), frame_count * hop, hop, frame_count)

    actors: list[dict[str, Any]] = []
    trajectory_payloads: dict[str, Any] = {}
    for i, role in enumerate(config.actor_roles):
        speed = float(rng.uniform(config.speed_range[0], config.speed_range[1]))
        step_len = speed * float(config.segment_sec)
        traj_type = config.traj_type
        if traj_type not in ("linear", "waypoint"):
            traj_type = "linear" if rng.random() < 0.5 else "waypoint"
        az = float(rng.uniform(0.0, 2.0 * math.pi))
        direction = np.array([math.cos(az), math.sin(az), 0.0])
        waypoint_every = 3
        traj = np.zeros((frame_count, 3), dtype=np.float32)
        traj[0] = actor_starts[i]
        for t in range(1, frame_count):
            if traj_type == "waypoint" and t % waypoint_every == 0:
                az += float(rng.uniform(-math.pi / 3.0, math.pi / 3.0))
                direction = np.array([math.cos(az), math.sin(az), 0.0])
            next_position, direction = advance_with_wall_reflections(
                traj[t - 1], direction, step_len, room, config.source_wall_margin_m
            )
            traj[t] = next_position
        actor_uid = role
        trajectory_payloads[actor_uid] = traj[:, None, :].tolist()
        actors.append({
            "actor_uid": actor_uid,
            "role": role,
            "position_m": actor_starts[i].tolist(),
            "trajectory": trajectory_payloads[actor_uid],
        })

    room_payload = {
        "room_uid": "room-" + hashlib.sha256(json.dumps({"room_dim_m": room_list, "rt60": rt60}, sort_keys=True).encode()).hexdigest()[:24],
        "room_dim_m": [float(value) for value in room_list],
        "rt60": float(rt60),
        "sound_speed": float(config.c),
        "diffuse_db": float(config.backend_options.gpurir.diffuse_transition_db),
        "rir_num_samples": int(config.n_samples),
        "early_rir_sec": float(config.early_rir_sec),
    }
    array_payload = {
        "array_profile": config.array_profile,
        "positions_m": mic_positions.tolist(),
        "reference_channel": int(config.reference_channel),
        "channel_order": list(config.channel_order or tuple(f"mic_{i}" for i in range(array.channels))),
        "coordinate_system": array.profile.topology.coordinate_system,
    }
    if len(array_payload["channel_order"]) != array.channels:
        raise ValueError("channel_order length must match resolved microphone channels")
    identity_payload = {
        "sample_rate": config.fs, "room": room_payload, "array": array_payload,
        "actors": actors, "generator_version": generator_version,
    }
    scene_uid = "scene-" + hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()[:32]
    return SpatialScenePlan(
        scene_uid=scene_uid, sample_rate=int(config.fs), rir_num_samples=int(config.n_samples),
        room=room_payload, array=array_payload, actors=tuple(actors),
        generator_version=generator_version, moving=True,
        trajectory_timebase=clock.to_dict(),
    )


def build_moving_gpurir_plan(scene: SpatialScenePlan) -> dict[str, Any]:
    """把 moving scene 计划投影为 core moving scene backend 的 canonical plan。"""
    plan = build_gpurir_plan(scene)
    plan["actor_trajectories"] = {
        actor["actor_uid"]: actor["trajectory"] for actor in scene.actors
    }
    plan["trajectory_timebase"] = dict(scene.trajectory_timebase or {})
    return plan


def generate_gpurir_moving_scene(scene: SpatialScenePlan, gpu_rir: Any) -> dict[str, Any]:
    """生成同一 moving scene 的 full/direct/early ``[P,A,C,L]`` track bank。"""
    if not scene.moving or scene.trajectory_timebase is None:
        raise ValueError("generate_gpurir_moving_scene requires a moving scene plan")
    plan = build_moving_gpurir_plan(scene)
    generated = generate_moving_scene_rir_tracks(plan, gpu_rir)
    full = np.asarray(generated["full"][0], dtype="<f4")
    direct = np.asarray(generated["direct"][0], dtype="<f4")
    if full.ndim != 4 or direct.shape != full.shape:
        raise ValueError(f"moving scene banks must be [P,A,C,L], got {full.shape}")
    # Early track 逐帧沿用静态 scene 的 direct-peak 锚定策略。
    early = np.zeros_like(full)
    window_samples = max(1, int(round(scene.room.get("early_rir_sec", 0.05) * scene.sample_rate)))
    direct_peak = np.argmax(np.abs(direct), axis=-1)
    for frame in range(full.shape[0]):
        for actor_index in range(full.shape[1]):
            for channel_index in range(full.shape[2]):
                start = int(direct_peak[frame, actor_index, channel_index])
                stop = min(full.shape[-1], start + window_samples)
                early[frame, actor_index, channel_index, start:stop] = full[
                    frame, actor_index, channel_index, start:stop
                ]
    return {
        "scene": scene.to_dict(),
        "full": full,
        "direct": direct,
        "early": early,
        "facts": {
            "scene_uid": scene.scene_uid,
            "actor_order": list(scene.actor_order),
            "room": scene.room,
            "array": scene.array,
            "sample_rate": scene.sample_rate,
            "generator_version": scene.generator_version,
            "track_kind": "scene_moving",
            "early_policy": "direct_peak_plus_duration",
            "early_window_samples": window_samples,
        },
    }


__all__ = [
    "SpatialScenePlan",
    "resolve_spatial_array",
    "sample_spatial_scene",
    "sample_spatial_moving_scene",
    "build_gpurir_plan",
    "build_moving_gpurir_plan",
    "generate_gpurir_scene",
    "generate_gpurir_moving_scene",
]
