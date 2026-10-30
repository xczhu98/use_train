"""RIR 场景几何采样。

该模块只负责房间、RT60、源麦位置和移动轨迹，不依赖具体 RIR backend。
"""

from __future__ import annotations

import math

import numpy as np

from tools.prepare_data_tools.generators.se.rir_config import (
    MIN_SOURCE_MICROPHONE_DISTANCE,
    RIRConfig,
)


def resolve_scene_seed(cfg: RIRConfig, idx: int) -> int:
    """返回与 GPU、worker 数量和任务调度无关的非负场景种子。"""
    seed = cfg.scene_seed if cfg.scene_seed is not None else cfg.seed + idx
    return int(seed) % (1 << 64)


def validate_distance_range(dist_range: tuple[float, float]) -> None:
    """校验声源与麦克风距离范围，避免重合或反向区间。"""
    lo, hi = dist_range
    if not all(math.isfinite(value) for value in (lo, hi)) or not (
        MIN_SOURCE_MICROPHONE_DISTANCE <= lo <= hi
    ):
        raise ValueError(
            "source-microphone distance range must be finite, ordered, and "
            f"strictly positive (minimum {MIN_SOURCE_MICROPHONE_DISTANCE} m), got {dist_range}"
        )


def sample_uniform_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """在三维单位球面上均匀采样一个方向。"""
    azimuth = float(rng.uniform(0.0, 2.0 * math.pi))
    z = float(rng.uniform(-1.0, 1.0))
    radius_xy = math.sqrt(max(0.0, 1.0 - z * z))
    return np.array(
        [radius_xy * math.cos(azimuth), radius_xy * math.sin(azimuth), z]
    )


def advance_with_wall_reflections(
    position: np.ndarray,
    direction: np.ndarray,
    step_len: float,
    room_dim: list[float],
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """将一步位移折叠到房间内，并保留多次墙面反射后的行进方向。"""
    next_position = np.empty(3, dtype=np.float64)
    next_direction = np.asarray(direction, dtype=np.float64).copy()
    for axis in range(3):
        lo = margin
        hi = float(room_dim[axis]) - margin
        span = hi - lo
        unfolded = float(position[axis]) - lo + next_direction[axis] * step_len
        phase = unfolded % (2.0 * span)
        if phase <= span:
            next_position[axis] = lo + phase
        else:
            next_position[axis] = hi - (phase - span)
            next_direction[axis] = -next_direction[axis]
    return next_position, next_direction


def sample_room_rt60(cfg: RIRConfig, rng: np.random.Generator) -> tuple[list[float], float]:
    """采样房间尺寸与物理自洽的 RT60（Sabine 校验 alpha < 1）。"""
    room_dim = [
        float(rng.uniform(cfg.room_length_range[0], cfg.room_length_range[1])),
        float(rng.uniform(cfg.room_width_range[0], cfg.room_width_range[1])),
        float(rng.uniform(cfg.room_height_range[0], cfg.room_height_range[1])),
    ]
    L = np.array(room_dim)
    V = float(np.prod(L))
    S = 2.0 * float(L[0] * L[1] + L[0] * L[2] + L[1] * L[2])
    for _ in range(100):
        rt60_tgt = float(rng.uniform(cfg.rt60_range[0], cfg.rt60_range[1]))
        alpha = 24.0 * math.log(10.0) * V / (cfg.c * S * rt60_tgt)
        if alpha < 1.0:
            return room_dim, rt60_tgt
    raise ValueError(
        "could not sample an RT60 with Sabine absorption coefficient alpha < 1 "
        f"for room_dim={room_dim} and rt60_range={cfg.rt60_range}"
    )


def sample_source_start(
    rng: np.random.Generator,
    room_dim: list[float],
    wall_margin_m: float = 0.5,
    height_range_m: tuple[float, float] | None = None,
) -> list[float]:
    """采样声源起点，并应用可配置的墙距和高度约束。"""
    if len(room_dim) != 3 or not all(
        math.isfinite(value) and value >= 1.0 for value in room_dim
    ):
        raise ValueError("room dimensions must contain three finite values >= 1.0 m")
    if not 0.0 <= wall_margin_m < min(room_dim) / 2.0:
        raise ValueError(f"source wall margin is incompatible with room dimensions: {wall_margin_m}")
    if height_range_m is None:
        z_hi = max(0.2, min(2.3, room_dim[2] - wall_margin_m))
        z_lo = min(1.0, z_hi)
    else:
        z_lo, z_hi = height_range_m
        z_lo = max(z_lo, wall_margin_m)
        z_hi = min(z_hi, room_dim[2] - wall_margin_m)
        if z_lo >= z_hi:
            raise ValueError(
                f"source height range {height_range_m} is incompatible with room height {room_dim[2]}"
            )
    return [
        float(rng.uniform(wall_margin_m, room_dim[0] - wall_margin_m)),
        float(rng.uniform(wall_margin_m, room_dim[1] - wall_margin_m)),
        float(rng.uniform(z_lo, z_hi)),
    ]


def sample_room_and_positions(
    cfg: RIRConfig,
    idx: int,
    gpu_id: int = 0,
) -> tuple[list[float], list[float], list[float], float, float]:
    """为三种 backend 统一采样房间、RT60、声源和麦克风位置。"""
    validate_distance_range(cfg.dist_range)
    rng = np.random.default_rng(resolve_scene_seed(cfg, idx))
    room_dim, rt60_tgt = sample_room_rt60(cfg, rng)
    source_loc = sample_source_start(
        rng,
        room_dim,
        wall_margin_m=cfg.source_wall_margin_m,
        height_range_m=cfg.source_height_range_m,
    )

    mic_loc = None
    near_mic_dist = 0.0
    src_arr = np.array(source_loc)
    margin = cfg.microphone_wall_margin_m
    for _ in range(500):
        near_mic_dist = float(rng.uniform(cfg.dist_range[0], cfg.dist_range[1]))
        direction = sample_uniform_unit_vector(rng)
        candidate = src_arr + near_mic_dist * direction
        if all(margin < candidate[i] < room_dim[i] - margin for i in range(3)) and (
            cfg.microphone_height_range_m is None
            or cfg.microphone_height_range_m[0]
            <= candidate[2]
            <= cfg.microphone_height_range_m[1]
        ):
            mic_loc = candidate.tolist()
            break

    if mic_loc is None:
        near_mic_dist = float(rng.uniform(cfg.dist_range[0], cfg.dist_range[1]))
        for _ in range(100):
            theta = float(rng.uniform(0.0, 2.0 * math.pi))
            direction = np.array([math.cos(theta), math.sin(theta), 0.0])
            candidate = src_arr + near_mic_dist * direction
            if all(margin < candidate[i] < room_dim[i] - margin for i in range(3)) and (
                cfg.microphone_height_range_m is None
                or cfg.microphone_height_range_m[0]
                <= candidate[2]
                <= cfg.microphone_height_range_m[1]
            ):
                mic_loc = candidate.tolist()
                break

    if mic_loc is None:
        raise ValueError(
            "could not place microphone within the requested distance range "
            f"{cfg.dist_range} for room_dim={room_dim}"
        )
    return room_dim, source_loc, mic_loc, rt60_tgt, near_mic_dist


def sample_moving_scene(
    cfg: RIRConfig,
    idx: int,
    gpu_id: int = 0,
) -> tuple[list[float], np.ndarray, list[float], float, float, float]:
    """采样移动声源场景：房间、轨迹、固定麦克风和 RT60。"""
    validate_distance_range(cfg.dist_range)
    rng = np.random.default_rng(resolve_scene_seed(cfg, idx))
    room_dim, rt60_tgt = sample_room_rt60(cfg, rng)
    source_start = sample_source_start(
        rng,
        room_dim,
        wall_margin_m=cfg.source_wall_margin_m,
        height_range_m=cfg.source_height_range_m,
    )
    margin = cfg.microphone_wall_margin_m

    mic_loc = None
    near_mic_dist = 0.0
    src_arr = np.array(source_start)
    for _ in range(500):
        near_mic_dist = float(rng.uniform(cfg.dist_range[0], cfg.dist_range[1]))
        direction = sample_uniform_unit_vector(rng)
        candidate = src_arr + near_mic_dist * direction
        if all(margin < candidate[i] < room_dim[i] - margin for i in range(3)) and (
            cfg.microphone_height_range_m is None
            or cfg.microphone_height_range_m[0]
            <= candidate[2]
            <= cfg.microphone_height_range_m[1]
        ):
            mic_loc = candidate.tolist()
            break
    if mic_loc is None:
        raise ValueError(
            "could not place microphone within the requested distance range "
            f"{cfg.dist_range} for room_dim={room_dim}"
        )

    segment_lo, segment_hi = cfg.n_segments_range
    T = int(rng.integers(segment_lo, segment_hi + 1))
    speed = float(rng.uniform(cfg.speed_range[0], cfg.speed_range[1]))
    step_len = speed * cfg.segment_sec
    if cfg.traj_type in ("linear", "waypoint"):
        traj_type = cfg.traj_type
    else:
        traj_type = "linear" if rng.random() < 0.5 else "waypoint"

    az = float(rng.uniform(0.0, 2.0 * math.pi))
    direction = np.array([math.cos(az), math.sin(az), 0.0])
    waypoint_every = 3
    traj = np.zeros((T, 3), dtype=np.float32)
    traj[0] = source_start
    for t in range(1, T):
        if traj_type == "waypoint" and t % waypoint_every == 0:
            az += float(rng.uniform(-math.pi / 3.0, math.pi / 3.0))
            direction = np.array([math.cos(az), math.sin(az), 0.0])
        next_position, direction = advance_with_wall_reflections(
            traj[t - 1], direction, step_len, room_dim, cfg.source_wall_margin_m
        )
        traj[t] = next_position
    return room_dim, traj, mic_loc, rt60_tgt, near_mic_dist, speed


__all__ = [
    "advance_with_wall_reflections",
    "resolve_scene_seed",
    "sample_moving_scene",
    "sample_room_and_positions",
    "sample_room_rt60",
    "sample_source_start",
    "sample_uniform_unit_vector",
    "validate_distance_range",
]
