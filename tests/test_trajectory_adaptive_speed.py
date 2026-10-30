"""声源轨迹自适应缩速采样的合同测试。"""

from __future__ import annotations

import numpy as np
import pytest

from core.data.simulation.trajectory import (
    TrajectoryTimebase,
    plan_linear_source_trajectory,
    sample_source_trajectories_by_speed,
)


pytestmark = [pytest.mark.unit]


def _timebase(seconds: float = 32.0, sample_rate: int = 16000) -> TrajectoryTimebase:
    num_samples = int(seconds * sample_rate)
    return TrajectoryTimebase(sample_rate, num_samples, num_samples // 128, 128)


def _two_sources(rng: np.random.Generator, room=(3.0, 3.0, 2.5), margin: float = 0.1):
    room_arr = np.asarray(room, dtype=np.float64)
    while True:
        starts = np.column_stack(
            [
                rng.uniform(margin, room_arr[0] - margin, 2),
                rng.uniform(margin, room_arr[1] - margin, 2),
                rng.uniform(1.1, 1.8, 2),
            ]
        )
        if np.linalg.norm(starts[0] - starts[1]) >= 0.3:
            return starts, room_arr


def test_infeasible_speed_range_now_succeeds_with_reduced_speed():
    """32 s 画布 + 4 s 基准速度（位移 3.8–12.7 m）在 3 m 房间原本必然失败。"""
    rng = np.random.default_rng(1)
    starts, room = _two_sources(rng)
    trajectory, speeds = sample_source_trajectories_by_speed(
        starts,
        room_dim_m=room,
        timebase=_timebase(32.0),
        speed_range_mps=(0.12, 0.4),
        rng=rng,
        wall_margin_m=0.1,
        min_separation_m=0.3,
        source_height_range_m=(1.1, 1.8),
    )
    positions = np.asarray(trajectory.positions_m, dtype=np.float64)
    # 全部轨迹帧都落在房间边界内。
    assert np.all(positions >= 0.1 - 1e-6)
    assert np.all(positions <= room - 0.1 + 1e-6)
    # 实际速度非负且不超过配置上限；缩速后允许低于配置下限。
    speed_arr = np.asarray(speeds, dtype=np.float64)
    assert np.all(speed_arr >= 0.0)
    assert np.all(speed_arr <= 0.4 + 1e-9)


def test_trajectory_stays_linear_under_projection():
    """投影只缩短移动距离，轨迹仍是固定 timebase 上的线性插值。"""
    rng = np.random.default_rng(2)
    starts, room = _two_sources(rng)
    trajectory, speeds = sample_source_trajectories_by_speed(
        starts,
        room_dim_m=room,
        timebase=_timebase(32.0),
        speed_range_mps=(0.12, 0.4),
        rng=rng,
        wall_margin_m=0.1,
        min_separation_m=0.3,
        source_height_range_m=(1.1, 1.8),
    )
    ends = np.asarray(trajectory.positions_m, dtype=np.float64)[-1]
    expected = plan_linear_source_trajectory(starts, ends, trajectory.timebase)
    np.testing.assert_allclose(
        np.asarray(trajectory.positions_m), np.asarray(expected.positions_m), atol=1e-6
    )


def test_feasible_config_keeps_strict_sampling_semantics():
    """可行配置下实际速度保持在配置区间内（首尝试成功路径未缩速）。"""
    rng = np.random.default_rng(3)
    starts, room = _two_sources(rng, room=(8.0, 8.0, 3.5))
    trajectory, speeds = sample_source_trajectories_by_speed(
        starts,
        room_dim_m=room,
        timebase=_timebase(4.0),
        speed_range_mps=(0.12, 0.4),
        rng=rng,
        wall_margin_m=0.1,
        min_separation_m=0.3,
        source_height_range_m=(1.1, 1.8),
    )
    speed_arr = np.asarray(speeds, dtype=np.float64)
    # 4 s 画布位移 0.45–1.5 m，8 m 房间首尝试几乎必然成功；一旦成功不得缩速。
    assert np.all(speed_arr >= 0.12 - 1e-9)
    assert np.all(speed_arr <= 0.4 + 1e-9)
    assert np.all(np.asarray(trajectory.positions_m) >= 0.1 - 1e-6)


def test_stationary_source_keeps_zero_speed():
    rng = np.random.default_rng(4)
    starts, room = _two_sources(rng)
    trajectory, speeds = sample_source_trajectories_by_speed(
        starts,
        room_dim_m=room,
        timebase=_timebase(32.0),
        speed_range_mps=(0.12, 0.4),
        rng=rng,
        wall_margin_m=0.1,
        min_separation_m=0.3,
        source_height_range_m=(1.1, 1.8),
        stationary_indices=(1,),
    )
    assert speeds[1] == 0.0
    stationary_track = np.asarray(trajectory.positions_m)[:, 1, :]
    np.testing.assert_allclose(
        stationary_track,
        np.broadcast_to(starts[1], stationary_track.shape),
        atol=1e-6,
    )


def test_start_positions_violate_min_separation_fail_fast():
    rng = np.random.default_rng(5)
    starts = np.array([[1.0, 1.5, 1.4], [1.1, 1.5, 1.4]], dtype=np.float64)
    with pytest.raises(ValueError, match="start positions violate min_separation_m"):
        sample_source_trajectories_by_speed(
            starts,
            room_dim_m=np.array([3.0, 3.0, 2.5]),
            timebase=_timebase(4.0),
            speed_range_mps=(0.12, 0.4),
            rng=rng,
            wall_margin_m=0.1,
            min_separation_m=0.3,
            source_height_range_m=(1.1, 1.8),
        )


def test_sampling_is_deterministic_for_fixed_seed():
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    starts_a, room_a = _two_sources(rng_a)
    starts_b, room_b = _two_sources(rng_b)
    traj_a, speeds_a = sample_source_trajectories_by_speed(
        starts_a,
        room_dim_m=room_a,
        timebase=_timebase(32.0),
        speed_range_mps=(0.12, 0.4),
        rng=rng_a,
        wall_margin_m=0.1,
        min_separation_m=0.3,
        source_height_range_m=(1.1, 1.8),
    )
    traj_b, speeds_b = sample_source_trajectories_by_speed(
        starts_b,
        room_dim_m=room_b,
        timebase=_timebase(32.0),
        speed_range_mps=(0.12, 0.4),
        rng=rng_b,
        wall_margin_m=0.1,
        min_separation_m=0.3,
        source_height_range_m=(1.1, 1.8),
    )
    np.testing.assert_allclose(
        np.asarray(traj_a.positions_m), np.asarray(traj_b.positions_m), atol=0.0
    )
    assert speeds_a == speeds_b
