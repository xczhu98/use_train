"""共享 noise source 的选择与 I/O provenance 合同。

本模块只描述一次 source read 的 family、分类元数据、应用位置和缓存事实。
它不承载 waveform，也不替代 Layer C 的 ``NoiseSourcePlan`` 或 post-mix state。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NoiseSourceFamily(str, Enum):
    """可审计的 source family。"""

    ORDINARY = "ordinary"
    MUSIC = "music"
    WIND = "wind"
    REUSE_CACHE = "reuse_cache"


class NoiseApplicationRole(str, Enum):
    """source 被施加的位置；避免把 task-local 后处理伪装为 Layer C。"""

    LAYER_C_ADDITIVE = "layer_c_additive"
    TSE_ENROLLMENT_ADDITIVE = "tse_enrollment_additive"
    AEC_POST_EF_MIC_EXTRA = "aec_post_ef_mic_extra"


class NoiseCacheProvenance(str, Enum):
    """解码缓存的可观测 provenance。

    ``_read_cached_2d`` / ``_load_audio_cached`` 当前不公开 hit/miss；调用方只能
    诚实标记 ``SHARED_AUDIO_CACHE_UNKNOWN``，不能臆造 cache hit。
    """

    DATASET_REUSE_CACHE = "dataset_reuse_cache"
    SHARED_AUDIO_CACHE_UNKNOWN = "shared_audio_cache_unknown"


@dataclass(frozen=True)
class NoiseCategory:
    """可选的 manifest 分类 identity。"""

    category_id: int | None
    category_name: str | None

    def __post_init__(self) -> None:
        if self.category_id is not None and (
            type(self.category_id) is not int or self.category_id < 0
        ):
            raise ValueError("category_id must be a non-negative int or None")
        if self.category_name is not None and (
            not isinstance(self.category_name, str)
            or not self.category_name
            or self.category_name != self.category_name.strip()
        ):
            raise ValueError("category_name must be a non-empty stripped string or None")

    def to_dict(self) -> dict[str, int | str | None]:
        return {"id": self.category_id, "name": self.category_name}


@dataclass(frozen=True)
class NoiseSourcePolicy:
    """一次 noise source selection 与解码路径的 typed、无 waveform 事实。"""

    family: NoiseSourceFamily
    category: NoiseCategory
    application_role: NoiseApplicationRole
    cache_provenance: NoiseCacheProvenance
    uid: str | None
    path: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.family, NoiseSourceFamily):
            raise TypeError("family must be a NoiseSourceFamily")
        if not isinstance(self.category, NoiseCategory):
            raise TypeError("category must be a NoiseCategory")
        if not isinstance(self.application_role, NoiseApplicationRole):
            raise TypeError("application_role must be a NoiseApplicationRole")
        if not isinstance(self.cache_provenance, NoiseCacheProvenance):
            raise TypeError("cache_provenance must be a NoiseCacheProvenance")
        if self.uid is not None and (not isinstance(self.uid, str) or not self.uid):
            raise ValueError("uid must be a non-empty string or None")
        if self.path is not None and (not isinstance(self.path, str) or not self.path):
            raise ValueError("path must be a non-empty string or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family.value,
            "category": self.category.to_dict(),
            "application_role": self.application_role.value,
            "cache_provenance": self.cache_provenance.value,
            "uid": self.uid,
            "path": self.path,
        }


__all__ = [
    "NoiseApplicationRole",
    "NoiseCacheProvenance",
    "NoiseCategory",
    "NoiseSourceFamily",
    "NoiseSourcePolicy",
]
