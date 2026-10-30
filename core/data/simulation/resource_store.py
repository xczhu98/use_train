"""动态仿真的音频资源读取 seam。

``ResourceStore`` 只负责根据已解析的路径读取 speech/noise waveform；UID 选择、
noise policy、缓存和 scene provenance 仍由 Dataset/任务层管理。测试或任务专用
读取规则必须通过 dataclass 的显式 reader 注入，不能依赖 Dataset 模块级 monkeypatch。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from core.data.simulation_audio_utils import (
    _load_audio_uncached,
    _read_cached_2d,
    _read_spatial_noise_2d,
)


class SpeechReader(Protocol):
    """clean speech reader 的显式调用合同。"""

    def __call__(self, path: str, sample_rate: int, strict: bool) -> np.ndarray: ...


class NoiseReader(Protocol):
    """mono/spatial noise reader 的显式调用合同。"""

    def __call__(
        self,
        path: str,
        sample_rate: int,
        max_duration: int,
        *,
        strict: bool,
        rng: np.random.Generator,
        crop_record: dict,
    ) -> np.ndarray: ...

@dataclass(frozen=True)
class ResourceStore:
    """读取动态仿真所需的 speech、mono noise 和 spatial noise 资源。"""

    strict_load: bool = True
    # 读取器可显式注入，测试和任务扩展不需要改动 Dataset 内部状态。
    speech_reader: SpeechReader | None = None
    noise_reader: NoiseReader | None = None
    spatial_noise_reader: NoiseReader | None = None

    def read_speech(self, path: str, sample_rate: int) -> np.ndarray:
        """读取一条 clean speech，不占用共享 audio cache。"""
        reader = _load_audio_uncached if self.speech_reader is None else self.speech_reader
        return reader(path, int(sample_rate), bool(self.strict_load))

    def read_noise(
        self,
        path: str,
        sample_rate: int,
        max_duration: int,
        *,
        spatial: bool,
        strict: bool | None = None,
        rng: np.random.Generator,
        crop_record: dict,
    ) -> np.ndarray:
        """读取一段 mono 或 spatial noise；选择和裁剪策略由统一 I/O seam 执行。"""
        if spatial:
            reader = (
                _read_spatial_noise_2d
                if self.spatial_noise_reader is None
                else self.spatial_noise_reader
            )
        else:
            reader = _read_cached_2d if self.noise_reader is None else self.noise_reader
        return reader(
            path,
            int(sample_rate),
            int(max_duration),
            strict=bool(self.strict_load if strict is None else strict),
            rng=rng,
            crop_record=crop_record,
        )


__all__ = ["NoiseReader", "ResourceStore", "SpeechReader"]
