"""可复现的离线声源轨迹与 frame timebase 合同。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# 自适应缩速的重试衰减因子：第 n 次失败后将速度区间整体乘以 0.9**n。
_SPEED_RETRY_SHRINK: float = 0.9


@dataclass(frozen=True)
class TrajectoryTimebase:
    """把 waveform sample 轴和 trajectory frame 轴绑定起来。"""

    sample_rate: int
    num_samples: int
    hop_samples: int
    frame_count: int
    start_sample: int = 0

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("sample_rate", self.sample_rate, 1),
            ("num_samples", self.num_samples, 1),
            ("hop_samples", self.hop_samples, 1),
            ("frame_count", self.frame_count, 1),
            ("start_sample", self.start_sample, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an exact integer >= {minimum}")
        last = self.start_sample + (self.frame_count - 1) * self.hop_samples
        if last >= self.num_samples:
            raise ValueError("trajectory frame samples exceed num_samples waveform axis")

    @property
    def frame_samples(self) -> np.ndarray:
        return self.start_sample + np.arange(self.frame_count, dtype=np.int64) * self.hop_samples

    @property
    def frame_times(self) -> np.ndarray:
        return self.frame_samples.astype(np.float64) / float(self.sample_rate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "num_samples": self.num_samples,
            "hop_samples": self.hop_samples,
            "frame_count": self.frame_count,
            "start_sample": self.start_sample,
            "frame_samples": self.frame_samples.tolist(),
            "frame_times_sec": self.frame_times.tolist(),
        }


@dataclass(frozen=True)
class SourceTrajectory:
    """离线移动源轨迹，位置布局固定为 ``[T,K,3]``。"""

    positions_m: np.ndarray
    timebase: TrajectoryTimebase

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_m, dtype=np.float32)
        if positions.ndim != 3 or positions.shape[0] != self.timebase.frame_count or positions.shape[2] != 3:
            raise ValueError("positions_m must have shape [T,K,3] matching timebase")
        if positions.shape[1] < 1 or not np.isfinite(positions).all():
            raise ValueError("positions_m must be non-empty and finite")
        positions = np.ascontiguousarray(positions, dtype="<f4")
        positions.setflags(write=False)
        object.__setattr__(self, "positions_m", positions)

    @property
    def frame_count(self) -> int:
        return self.positions_m.shape[0]

    @property
    def source_count(self) -> int:
        return self.positions_m.shape[1]

    def to_dict(self) -> dict[str, Any]:
        return {"positions_m": self.positions_m.tolist(), "timebase": self.timebase.to_dict()}


def plan_linear_source_trajectory(
    start_positions_m: np.ndarray,
    end_positions_m: np.ndarray,
    timebase: TrajectoryTimebase,
) -> SourceTrajectory:
    """按固定 timebase 线性规划轨迹，不消费随机数。"""
    start = np.asarray(start_positions_m, dtype=np.float32)
    end = np.asarray(end_positions_m, dtype=np.float32)
    if start.shape != end.shape or start.ndim != 2 or start.shape[1] != 3:
        raise ValueError("start_positions_m and end_positions_m must both have shape [K,3]")
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        raise ValueError("trajectory endpoints must be finite")
    alpha = np.linspace(0.0, 1.0, timebase.frame_count, dtype=np.float32)[:, None, None]
    return SourceTrajectory(start[None] * (1.0 - alpha) + end[None] * alpha, timebase)


def sample_source_trajectories_by_speed(
    start_positions_m: np.ndarray,
    *,
    room_dim_m: np.ndarray,
    timebase: TrajectoryTimebase,
    speed_range_mps: tuple[float, float],
    rng: np.random.Generator,
    wall_margin_m: float,
    min_separation_m: float = 0.0,
    source_height_range_m: tuple[float, float] | None = None,
    stationary_indices: tuple[int, ...] = (),
    max_attempts: int = 100,
) -> tuple[SourceTrajectory, tuple[float, ...]]:
    """在房间边界内抽样水平线性轨迹，并保持指定的物理速度。

    自适应缩速：首次尝试严格按 ``speed_range_mps`` 抽样；若位移无法落入
    房间（速度×时长超过房间可达位移），后续尝试按固定衰减因子逐级缩小
    速度区间重试，并把越界端点沿位移方向投影回房间内（轨迹保持线性、
    单源速度保持恒定，仅缩短移动距离）。因此返回的实际速度可能低于配置
    下限；首尝试即成功的样本与严格抽样逐 bit 一致，rng 消耗序列不变。
    """
    starts = np.asarray(start_positions_m, dtype=np.float64)
    room = np.asarray(room_dim_m, dtype=np.float64)
    if starts.ndim != 2 or starts.shape[1] != 3 or starts.shape[0] < 1:
        raise ValueError("start_positions_m must have shape [K,3]")
    if room.shape != (3,) or not np.isfinite(room).all() or np.any(room <= 0.0):
        raise ValueError("room_dim_m must contain three finite positive dimensions")
    if not np.isfinite(starts).all():
        raise ValueError("start_positions_m must be finite")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    if (
        not isinstance(speed_range_mps, (tuple, list))
        or len(speed_range_mps) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
            or value <= 0.0
            for value in speed_range_mps
        )
        or speed_range_mps[0] > speed_range_mps[1]
    ):
        raise ValueError("speed_range_mps must be an ordered positive finite range")
    for name, value, allow_zero in (
        ("wall_margin_m", wall_margin_m, True),
        ("min_separation_m", min_separation_m, True),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
            or (value < 0.0 if allow_zero else value <= 0.0)
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive exact int")
    stationary = set(stationary_indices)
    if any(type(index) is not int or not 0 <= index < starts.shape[0] for index in stationary):
        raise ValueError("stationary_indices must be valid source indices")
    if source_height_range_m is not None:
        height = np.asarray(source_height_range_m, dtype=np.float64)
        if height.shape != (2,) or not np.isfinite(height).all() or height[0] > height[1]:
            raise ValueError("source_height_range_m must be an ordered finite range")
        if np.any(starts[:, 2] < height[0]) or np.any(starts[:, 2] > height[1]):
            raise ValueError("source start positions must satisfy source_height_range_m")

    duration_sec = float(timebase.frame_times[-1] - timebase.frame_times[0])
    if duration_sec <= 0.0:
        raise ValueError("trajectory timebase must span a positive duration")
    for left in range(starts.shape[0]):
        for right in range(left):
            if float(np.linalg.norm(starts[left] - starts[right])) < min_separation_m:
                raise ValueError(
                    "start positions violate min_separation_m; cannot sample source trajectories"
                )
    bounds = np.asarray([wall_margin_m, wall_margin_m, wall_margin_m], dtype=np.float64)
    upper = room - bounds
    for attempt in range(max_attempts):
        # 首次尝试 shrink=1.0，与严格抽样完全一致；仅失败后才逐级缩速。
        shrink = _SPEED_RETRY_SHRINK ** attempt
        ends = starts.copy()
        speeds = np.zeros(starts.shape[0], dtype=np.float64)
        for index in range(starts.shape[0]):
            if index in stationary:
                continue
            speed = float(rng.uniform(speed_range_mps[0] * shrink, speed_range_mps[1] * shrink))
            azimuth = float(rng.uniform(-np.pi, np.pi))
            displacement = speed * duration_sec * np.asarray(
                [np.cos(azimuth), np.sin(azimuth), 0.0], dtype=np.float64
            )
            # 端点投影：按最紧轴向把越界端点沿位移方向收回房间内。
            alpha = 1.0
            for axis in range(3):
                component = float(displacement[axis])
                if component == 0.0:
                    continue
                headroom = (
                    upper[axis] - starts[index, axis]
                    if component > 0.0
                    else starts[index, axis] - bounds[axis]
                )
                alpha = min(alpha, max(headroom, 0.0) / abs(component))
            ends[index] = starts[index] + alpha * displacement
            speeds[index] = speed * alpha
        valid = True
        for left in range(starts.shape[0]):
            for right in range(left):
                separation = starts[left] - starts[right]
                relative_velocity = (ends[left] - starts[left]) - (ends[right] - starts[right])
                denominator = float(np.dot(relative_velocity, relative_velocity))
                closest = np.clip(
                    -float(np.dot(separation, relative_velocity))
                    / max(denominator, 1e-20),
                    0.0,
                    1.0,
                )
                if np.linalg.norm(separation + closest * relative_velocity) < min_separation_m:
                    valid = False
        if valid:
            trajectory = plan_linear_source_trajectory(starts, ends, timebase)
            return trajectory, tuple(float(speed) for speed in speeds)
    raise ValueError(
        "cannot sample source trajectories even after adaptive speed shrink; "
        "check room, height, and separation constraints"
    )


def validate_trajectory_timebase(trajectory: SourceTrajectory, *, sample_rate: int, num_samples: int, hop_samples: int, frame_count: int) -> None:
    """拒绝 trajectory 与实际 waveform/frame 轴不一致的离线 scene。"""
    expected = TrajectoryTimebase(sample_rate, num_samples, hop_samples, frame_count, trajectory.timebase.start_sample)
    if trajectory.timebase != expected:
        raise ValueError("trajectory timebase does not match scene waveform/frame contract")
