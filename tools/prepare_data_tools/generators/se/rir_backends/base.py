"""RIR backend 公共接口。"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np

from tools.prepare_data_tools.generators.se.rir_config import RIRConfig


def trim_or_pad(rir: np.ndarray, n_samples: int) -> np.ndarray:
    """将静态或移动 RIR 截断/补零到目标长度。"""
    rir = np.asarray(rir, dtype=np.float32)
    if rir.ndim > 1:
        out = np.zeros((rir.shape[0], n_samples), dtype=np.float32)
        take = min(rir.shape[1], n_samples)
        out[:, :take] = rir[:, :take]
        return out
    rir = rir.ravel()
    if rir.shape[0] > n_samples:
        return rir[:n_samples]
    if rir.shape[0] < n_samples:
        return np.pad(rir, (0, n_samples - rir.shape[0]), mode="constant")
    return rir


class BaseRIRBackend(abc.ABC):
    """RIR backend 抽象接口。"""

    name: str = ""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """当前环境是否具备依赖。"""

    @abc.abstractmethod
    def generate_one(
        self, index: int, config: RIRConfig, gpu_id: int = 0
    ) -> dict[str, Any]:
        """生成一条静态或移动 RIR。"""

    def availability_error(self) -> str:
        return f"backend '{self.name}' requires extra dependencies."


__all__ = ["BaseRIRBackend", "trim_or_pad"]
