"""动态仿真命名 RNG seam。

该模块只负责把 task-local 的语义名称映射到既有 ``scene_rng`` stream id，
不改变 seed 派生算法，也不在关闭分支额外消费随机数。新的随机流必须先
登记语义名称，调用方不能直接传入裸 stream id。
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

from core.data.speech_scene import scene_rng


_STREAMS: dict[str, dict[str, int]] = {
    "ssl": {
        "source_selection": 10,
        "speech_scene": 11,
        "background_resolution": 12,
        "post_mix": 300,
        "speech_preparation": 100,
        "timeline": 150,
        "moving_gate": 175,
        "scene_plan": 200,
    },
    "ss": {
        "scenario": 0,
        "speech_scene": 1,
        "post_mix": 2,
        "background_mix": 301,
        "snr": 302,
        "timeline": 303,
        "source_volume": 304,
        "background_resolution": 305,
        "speech_preparation": 100,
        "latent_reference": 200,
        "scene_plan": 300,
    },
    "tse": {
        "scenario": 0,
        "speech_scene": 1,
        "post_mix": 2,
        "timeline": 303,
        "moving_gate": 301,
        "scene_plan": 302,
        "rir": 300,
        "background_resolution": 305,
        "speech_preparation": 100,
        "latent_reference": 200,
        "enrollment": 10000,
    },
}


@dataclass(frozen=True, slots=True)
class SimulationRNG:
    """按 task 和 index 解析命名随机流的轻量 facade。"""

    dataset: object
    index: int
    task: str
    rng_factory: Callable[[object, int, int], np.random.Generator] = scene_rng

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("SimulationRNG.index must be a non-negative exact int")
        if self.task not in _STREAMS:
            raise ValueError(f"SimulationRNG has no registered task stream map: {self.task!r}")

    @classmethod
    def for_task(
        cls,
        dataset: object,
        index: int,
        task: str,
        *,
        rng_factory: Callable[[object, int, int], np.random.Generator] | None = None,
    ) -> "SimulationRNG":
        return cls(
            dataset=dataset,
            index=index,
            task=task,
            rng_factory=scene_rng if rng_factory is None else rng_factory,
        )

    def stream(self, name: str, *, offset: int = 0) -> np.random.Generator:
        """返回命名 stream；offset 只用于既有的 source-position 子流。"""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("SimulationRNG stream name must be a non-empty string")
        if type(offset) is not int or offset < 0:
            raise ValueError("SimulationRNG stream offset must be a non-negative exact int")
        try:
            stream_id = _STREAMS[self.task][name] + offset
        except KeyError as exc:
            raise KeyError(
                f"unknown {self.task} simulation RNG stream {name!r}; "
                "register the semantic mapping before using it"
            ) from exc
        return self.rng_factory(self.dataset, self.index, stream_id)

    def stream_facts(self) -> dict[str, object]:
        """返回可写入 trace 的静态映射，不包含 RNG 状态。"""
        return {
            "task": self.task,
            "streams": dict(_STREAMS[self.task]),
        }


__all__ = ["SimulationRNG"]
