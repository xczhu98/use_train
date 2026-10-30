"""共享的时间布局配置值对象。

公开 YAML 仍由各任务的 split-level Dataset 字段承载画布生命周期；本模块只
在 core 内把已经校验过的字段归一成不可变 ``CanvasSpec``，避免不同任务重复
解释 ``max_duration_seconds``、sample cap 和 tile gap。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real


@dataclass(frozen=True, slots=True)
class CanvasSpec:
    """一条 split-level waveform canvas 的运行时规范。"""

    max_duration_seconds: float | None
    max_duration_samples: int
    repeat_to_max_duration: bool
    repeat_tile_silence_gap: bool
    repeat_tile_silence_gap_range: tuple[float, float]

    def __post_init__(self) -> None:
        if self.max_duration_seconds is not None and (
            isinstance(self.max_duration_seconds, bool)
            or not math.isfinite(float(self.max_duration_seconds))
            or float(self.max_duration_seconds) <= 0.0
        ):
            raise ValueError(
                "CanvasSpec.max_duration_seconds must be finite and positive"
            )
        if type(self.max_duration_samples) is not int or self.max_duration_samples < 0:
            raise ValueError(
                "CanvasSpec.max_duration_samples must be a non-negative exact int"
            )
        if type(self.repeat_to_max_duration) is not bool:
            raise TypeError("CanvasSpec.repeat_to_max_duration must be bool")
        if type(self.repeat_tile_silence_gap) is not bool:
            raise TypeError("CanvasSpec.repeat_tile_silence_gap must be bool")
        gap = self.repeat_tile_silence_gap_range
        if (
            not isinstance(gap, tuple)
            or len(gap) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in gap
            )
            or gap[0] < 0.0
            or gap[0] > gap[1]
        ):
            raise ValueError(
                "CanvasSpec.repeat_tile_silence_gap_range must be an ordered "
                "finite non-negative tuple"
            )

    @property
    def has_duration_cap(self) -> bool:
        """是否存在 split-level 的有效输出长度 cap。"""
        return self.max_duration_seconds is not None or self.max_duration_samples > 0

    def samples_for_sr(self, sample_rate: int) -> int:
        """按当前 sample rate 解析最终 canvas/cap 长度。"""
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive exact int")
        if self.max_duration_seconds is not None:
            return int(self.max_duration_seconds * sample_rate)
        return self.max_duration_samples


__all__ = ["CanvasSpec"]
