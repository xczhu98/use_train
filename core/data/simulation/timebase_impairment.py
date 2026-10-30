"""共享 timebase impairment 的 immutable plan/result facade。

本模块不抽样、不消费 RNG，也不复制 ``core.data.timebase`` 的数值实现。它只
把既定的 timebase plan 委托给既有 mechanics，并为调用方提供明确的符号约定、
valid mask 与后端事实。

时延符号约定：正 ``total_delay_samples`` 表示 capture/echo 相对 reference
滞后；对于已含 ``base_delay_samples`` 的 waveform，正相对时延读取更早样点。
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np

from core.data import timebase as _timebase

DELAY_SIGN_CONVENTION = "positive_delay_means_capture_or_echo_lags_reference"


def _immutable_array_snapshot(value: np.ndarray) -> np.ndarray:
    """返回保留 dtype/shape/values 且无法重新启用写入的 bytes-backed 快照。"""

    array = np.asarray(value)
    return np.frombuffer(
        array.tobytes(order="C"),
        dtype=array.dtype,
        count=array.size,
    ).reshape(array.shape)


@dataclass(frozen=True)
class TimebaseBackendFacts:
    """不改变数值路径的可审计后端事实。"""

    operation: str
    interpolation_backend: str
    asrc_backends: tuple[str, ...]


@dataclass(frozen=True)
class DelayTrajectoryPlan:
    """一次总时延 trajectory 应用计划。"""

    total_delay_samples: np.ndarray
    base_delay_samples: float
    edge_mode: str = "constant"
    sign_convention: str = DELAY_SIGN_CONVENTION

    def __post_init__(self) -> None:
        trajectory = _immutable_array_snapshot(
            np.asarray(self.total_delay_samples, dtype=np.float64).ravel()
        )
        object.__setattr__(self, "total_delay_samples", trajectory)


@dataclass(frozen=True)
class LinearTimebaseRatioPlan:
    """线性 ratio 应用计划，语义为 ``output[i] = input[i * ratio]``。"""

    ratio: float


@dataclass(frozen=True)
class TimebaseImpairmentResult:
    """由既有 mechanics 计算的不可变 waveform、positions、valid mask 和事实。"""

    waveform: np.ndarray
    read_positions: np.ndarray
    valid_mask: np.ndarray
    total_delay_samples: np.ndarray | None
    backend_facts: TimebaseBackendFacts

    def __post_init__(self) -> None:
        for field_name in (
            "waveform",
            "read_positions",
            "valid_mask",
            "total_delay_samples",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self, field_name, _immutable_array_snapshot(value)
                )


def available_backend_facts(
    *,
    operation: str,
    interpolation_backend: str,
) -> TimebaseBackendFacts:
    """返回当前环境可发现的 ASRC 后端；不导入或执行这些可选依赖。"""

    asrc_backends = tuple(
        backend
        for backend, module in (
            ("soxr", "soxr"),
            ("librosa", "librosa"),
            ("scipy.signal.resample", "scipy"),
        )
        if importlib.util.find_spec(module) is not None
    )
    return TimebaseBackendFacts(
        operation=operation,
        interpolation_backend=interpolation_backend,
        asrc_backends=asrc_backends,
    )


def _result_from_legacy(
    result: _timebase.TimebaseResult,
    *,
    operation: str,
    interpolation_backend: str,
) -> TimebaseImpairmentResult:
    return TimebaseImpairmentResult(
        waveform=result.waveform,
        read_positions=result.read_positions,
        valid_mask=result.valid_mask,
        total_delay_samples=result.total_delay_samples,
        backend_facts=available_backend_facts(
            operation=operation,
            interpolation_backend=interpolation_backend,
        ),
    )


def apply_delay_trajectory_plan(
    waveform: np.ndarray,
    plan: DelayTrajectoryPlan,
) -> TimebaseImpairmentResult:
    """应用既定 trajectory；waveform 与 valid mask 委托同一个旧实现。"""

    if plan.sign_convention != DELAY_SIGN_CONVENTION:
        raise ValueError(
            "DelayTrajectoryPlan sign_convention must be "
            f"{DELAY_SIGN_CONVENTION!r}, got {plan.sign_convention!r}"
        )
    return _result_from_legacy(
        _timebase.apply_delay_trajectory(
            waveform,
            total_delay_samples=plan.total_delay_samples,
            base_delay_samples=plan.base_delay_samples,
            edge_mode=plan.edge_mode,
        ),
        operation="delay_trajectory",
        interpolation_backend="scipy.ndimage.map_coordinates",
    )


def apply_linear_timebase_ratio_plan(
    waveform: np.ndarray,
    plan: LinearTimebaseRatioPlan,
) -> TimebaseImpairmentResult:
    """应用既有 ``np.interp`` linear-ratio mechanics，保留原数值路径。"""

    return _result_from_legacy(
        _timebase.apply_linear_timebase_ratio(waveform, plan.ratio),
        operation="linear_ratio",
        interpolation_backend="numpy.interp",
    )


def apply_linear_timebase_ratio(
    waveform: np.ndarray,
    ratio: float,
) -> TimebaseImpairmentResult:
    """兼容 convenience facade，等价于 ``LinearTimebaseRatioPlan(ratio)``。"""

    return apply_linear_timebase_ratio_plan(waveform, LinearTimebaseRatioPlan(ratio))


def apply_integer_delay(waveform: np.ndarray, delay_samples: int) -> np.ndarray:
    """返回既有零填充整数时延 mechanics 的不可变 float32 快照。"""

    return _immutable_array_snapshot(
        _timebase.apply_integer_delay(waveform, delay_samples)
    )


__all__ = [
    "DELAY_SIGN_CONVENTION",
    "DelayTrajectoryPlan",
    "LinearTimebaseRatioPlan",
    "TimebaseBackendFacts",
    "TimebaseImpairmentResult",
    "apply_delay_trajectory_plan",
    "apply_integer_delay",
    "apply_linear_timebase_ratio",
    "apply_linear_timebase_ratio_plan",
    "available_backend_facts",
]
