"""共享 waveform timebase mechanics；纯函数，不抽样也不消费 RNG。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class TimebaseResult:
    """一次 timebase 变换的 waveform、读位置与有效区间。"""

    waveform: np.ndarray
    read_positions: np.ndarray
    valid_mask: np.ndarray
    total_delay_samples: np.ndarray | None = None


def _waveform_f32_1d(waveform: np.ndarray) -> np.ndarray:
    value = np.asarray(waveform, dtype=np.float32).ravel()
    if not np.isfinite(value).all():
        raise ValueError("waveform must contain only finite samples")
    return value


def _finite_positions(read_positions: np.ndarray, *, length: int) -> np.ndarray:
    positions = np.asarray(read_positions, dtype=np.float64).ravel()
    if positions.size != length:
        raise ValueError(
            "read_positions length must match waveform length: "
            f"actual={positions.size}, expected={length}"
        )
    if not np.isfinite(positions).all():
        raise ValueError("read_positions must contain only finite values")
    return positions


def _retime_at_positions(
    waveform: np.ndarray,
    read_positions: np.ndarray,
    *,
    edge_mode: str,
) -> TimebaseResult:
    source = _waveform_f32_1d(waveform)
    positions = _finite_positions(read_positions, length=source.size)
    mode = str(edge_mode).strip().lower()
    if mode not in ("constant", "nearest"):
        raise ValueError(
            f"edge_mode must be 'constant' or 'nearest', got {edge_mode!r}"
        )
    output = map_coordinates(
        source,
        positions.reshape(1, -1),
        order=1,
        mode=mode,
        cval=0.0,
        output=np.float32,
        prefilter=False,
    ).reshape(-1)
    valid = (positions >= 0.0) & (positions <= float(source.size - 1))
    return TimebaseResult(
        waveform=output,
        read_positions=positions,
        valid_mask=valid,
    )


def apply_delay_trajectory(
    waveform: np.ndarray,
    *,
    total_delay_samples: np.ndarray,
    base_delay_samples: float,
    edge_mode: str = "constant",
) -> TimebaseResult:
    """按权威 total-delay trajectory 重定时 waveform。

    输入 waveform 已包含 ``base_delay_samples``；因此只施加 trajectory 相对
    baseline 的增量，正增量读取更早样点，表示 echo/capture 进一步滞后。
    """

    source = _waveform_f32_1d(waveform)
    delays = np.asarray(total_delay_samples, dtype=np.float64).ravel()
    if delays.size != source.size:
        raise ValueError(
            "total_delay_samples length must match waveform length: "
            f"actual={delays.size}, expected={source.size}"
        )
    if not np.isfinite(delays).all() or not np.isfinite(base_delay_samples):
        raise ValueError("delay trajectory and base_delay_samples must be finite")
    positions = np.arange(source.size, dtype=np.float64) - (
        delays - float(base_delay_samples)
    )
    result = _retime_at_positions(source, positions, edge_mode=edge_mode)
    return TimebaseResult(
        waveform=result.waveform,
        read_positions=result.read_positions,
        valid_mask=result.valid_mask,
        total_delay_samples=delays.copy(),
    )


def apply_linear_timebase_ratio(
    waveform: np.ndarray,
    ratio: float,
) -> TimebaseResult:
    """以 ``output[i] = input[i * ratio]`` 施加线性 timebase ratio。

    实现保留 OOD generator 原有 ``np.interp`` 数值路径，便于无版本变化迁移。
    """

    source = _waveform_f32_1d(waveform)
    parsed_ratio = float(ratio)
    if not np.isfinite(parsed_ratio) or parsed_ratio <= 0.0:
        raise ValueError(f"ratio must be positive and finite, got {ratio!r}")
    positions = np.arange(source.size, dtype=np.float64) * parsed_ratio
    valid = positions <= float(source.size - 1)
    output = np.zeros(source.size, dtype=np.float32)
    output[valid] = np.interp(
        positions[valid],
        np.arange(source.size, dtype=np.float64),
        source.astype(np.float64),
    ).astype(np.float32)
    return TimebaseResult(
        waveform=output,
        read_positions=positions,
        valid_mask=valid,
    )


def apply_integer_delay(waveform: np.ndarray, delay_samples: int) -> np.ndarray:
    """前补零并裁回原长；非正 delay 保持 identity-copy。"""

    source = _waveform_f32_1d(waveform)
    if isinstance(delay_samples, (bool, np.bool_)) or not isinstance(
        delay_samples, (int, np.integer)
    ):
        raise TypeError("delay_samples must be an integer")
    delay = int(delay_samples)
    if delay <= 0:
        return source.copy()
    output = np.zeros_like(source)
    if delay < source.size:
        output[delay:] = source[: source.size - delay]
    return output


__all__ = [
    "TimebaseResult",
    "apply_delay_trajectory",
    "apply_integer_delay",
    "apply_linear_timebase_ratio",
]
