"""样本级仿真事实与可选 waveform stems 的原子返回 Module。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Generic, Mapping, TypeVar

import numpy as np

from core.data.simulation.level_audit import json_safe


SampleT = TypeVar("SampleT")


def _reject_metadata_arrays(value: Any, path: str) -> None:
    """拒绝把 waveform ndarray 混入 JSON metadata namespace。"""
    if isinstance(value, np.ndarray):
        raise TypeError(f"trace metadata must not contain ndarray at {path}; use record_stem()")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_metadata_arrays(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_metadata_arrays(item, f"{path}[{index}]")


def _immutable_float32_snapshot(waveform: np.ndarray) -> np.ndarray:
    """返回独立 bytes-backed float32 快照，禁止恢复 writeable 标志。"""
    contiguous = np.ascontiguousarray(waveform, dtype=np.float32)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float32).reshape(contiguous.shape)


class SourceSimulationTrace:
    """一个 source role 的 JSON metadata 与可选 waveform stems。"""

    __slots__ = ("role", "capture_stems", "_metadata", "_stems", "_sealed")
    _IMMUTABLE_FIELDS = frozenset(("role", "capture_stems"))

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._IMMUTABLE_FIELDS and hasattr(self, name):
            raise AttributeError(f"simulation trace field {name!r} is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, role: str, *, capture_stems: bool) -> None:
        role = str(role).strip("/")
        if not role or "//" in role:
            raise ValueError(f"invalid simulation trace role: {role!r}")
        self.role = role
        self.capture_stems = bool(capture_stems)
        self._metadata: dict[str, Any] = {}
        self._stems: dict[str, np.ndarray] = {}
        self._sealed = False

    @property
    def metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(deepcopy(self._metadata))

    @property
    def stems(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType(self._stems)

    @property
    def sealed(self) -> bool:
        return self._sealed

    def _require_open(self) -> None:
        if self._sealed:
            raise RuntimeError(f"simulation trace role {self.role!r} is sealed")

    def record_metadata(self, name: str, value: Any) -> None:
        """写入一个 JSON-safe metadata 事实；同名事实禁止静默覆盖。"""
        self._require_open()
        key = str(name)
        if key in self._metadata:
            raise KeyError(f"trace metadata already recorded: {self.role}.{key}")
        _reject_metadata_arrays(value, f"{self.role}.{key}")
        self._metadata[key] = json_safe(value)

    def replace_metadata(self, name: str, value: Any) -> None:
        """显式替换一个分阶段补齐的 metadata 事实。"""
        self._require_open()
        key = str(name)
        _reject_metadata_arrays(value, f"{self.role}.{key}")
        self._metadata[key] = json_safe(value)

    def record_stem(self, name: str, waveform: Any) -> None:
        """按配置捕获独立 float32 waveform；关闭时不保留输入引用。"""
        self._require_open()
        if not self.capture_stems:
            return
        key = str(name)
        if key in self._stems:
            raise KeyError(f"trace stem already recorded: {self.role}.{key}")
        self._stems[key] = np.asarray(waveform, dtype=np.float32).copy()

    def replace_stem(self, name: str, waveform: Any) -> None:
        """显式替换经过裁剪、重采样或最终对齐的 waveform stem。"""
        self._require_open()
        if not self.capture_stems:
            return
        self._stems[str(name)] = np.asarray(waveform, dtype=np.float32).copy()

    def ingest_legacy_mapping(self, values: Mapping[str, Any] | None) -> None:
        """把旧顶层 sidecar dict 分离导入 metadata/stems namespace。"""
        self._require_open()
        for name, value in (values or {}).items():
            if isinstance(value, np.ndarray):
                self.replace_stem(str(name), value)
            else:
                self.replace_metadata(str(name), value)

    def _prepare_seal(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        self._require_open()
        metadata = deepcopy(self._metadata)
        stems = {
            name: _immutable_float32_snapshot(waveform)
            for name, waveform in self._stems.items()
        }
        return metadata, stems

    def _commit_seal(
        self, metadata: dict[str, Any], stems: dict[str, np.ndarray]
    ) -> None:
        self._metadata = metadata
        self._stems = stems
        self._sealed = True

    def seal(self) -> None:
        metadata, stems = self._prepare_seal()
        self._commit_seal(metadata, stems)

    def metadata_dict(self) -> dict[str, Any]:
        return deepcopy(self._metadata)

    def stems_dict(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._stems.items()}


class SimulationTrace:
    """一次 Dataset 调用独占的仿真 trace，成功完成后统一 seal。"""

    __slots__ = (
        "index",
        "uid",
        "task",
        "capture_stems",
        "placeholder",
        "_metadata",
        "_sources",
        "_legacy",
        "_sealed",
    )
    _IMMUTABLE_FIELDS = frozenset(("index", "uid", "task", "capture_stems"))

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._IMMUTABLE_FIELDS and hasattr(self, name):
            raise AttributeError(f"simulation trace field {name!r} is immutable")
        if name == "placeholder" and hasattr(self, "placeholder"):
            raise AttributeError("simulation trace field 'placeholder' is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        index: int,
        uid: str,
        task: str,
        capture_stems: bool,
    ) -> None:
        self.index = int(index)
        self.uid = str(uid)
        self.task = str(task)
        self.capture_stems = bool(capture_stems)
        self.placeholder = False
        self._metadata: dict[str, Any] = {}
        self._sources: dict[str, SourceSimulationTrace] = {}
        self._legacy: dict[str, Any] = {}
        self._sealed = False

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(deepcopy(self._metadata))

    @property
    def sources(self) -> Mapping[str, SourceSimulationTrace]:
        return MappingProxyType(self._sources)

    def _require_open(self) -> None:
        if self._sealed:
            raise RuntimeError("simulation trace is sealed")

    def source(self, role: str) -> SourceSimulationTrace:
        """返回 role 独占 trace；首次访问时创建。"""
        self._require_open()
        normalized = str(role).strip("/")
        source = self._sources.get(normalized)
        if source is None:
            source = SourceSimulationTrace(
                normalized, capture_stems=self.capture_stems
            )
            self._sources[normalized] = source
        return source

    def get_source(self, role: str) -> SourceSimulationTrace | None:
        return self._sources.get(str(role).strip("/"))

    def record_metadata(self, name: str, value: Any) -> None:
        self._require_open()
        key = str(name)
        if key in self._metadata:
            raise KeyError(f"trace metadata already recorded: {key}")
        _reject_metadata_arrays(value, key)
        self._metadata[key] = json_safe(value)

    def replace_metadata(self, name: str, value: Any) -> None:
        self._require_open()
        key = str(name)
        _reject_metadata_arrays(value, key)
        self._metadata[key] = json_safe(value)

    def mark_placeholder(self, *, reason: str) -> None:
        self._require_open()
        object.__setattr__(self, "placeholder", True)
        self.replace_metadata("placeholder", {"reason": str(reason)})

    def set_legacy_snapshot(self, name: str, value: Any) -> None:
        """记录成功发布时才投影到 Dataset 的兼容快照。"""
        self._require_open()
        self._legacy[str(name)] = deepcopy(value)

    def legacy_snapshot(self) -> dict[str, Any]:
        return deepcopy(self._legacy)

    def legacy_stems(self, role: str = "scene") -> dict[str, Any] | None:
        """把单个 role 投影为旧 metadata+stems dict Adapter。"""
        source = self.get_source(role)
        if source is None:
            return None
        output = source.metadata_dict()
        output.update(source.stems_dict())
        return output or None

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "uid": self.uid,
            "task": self.task,
            "placeholder": self.placeholder,
            **deepcopy(self._metadata),
            "sources": {
                role: source.metadata_dict()
                for role, source in self._sources.items()
            },
        }

    def seal(self) -> None:
        self._require_open()
        sources = tuple(self._sources.values())
        sealed_roles = tuple(source.role for source in sources if source.sealed)
        if sealed_roles:
            raise RuntimeError(
                "simulation trace contains source roles sealed before owner commit: "
                f"{sealed_roles!r}"
            )

        metadata = deepcopy(self._metadata)
        legacy = deepcopy(self._legacy)
        prepared_sources = tuple(
            (source, *source._prepare_seal()) for source in sources
        )

        self._metadata = metadata
        self._legacy = legacy
        for source, source_metadata, source_stems in prepared_sources:
            source._commit_seal(source_metadata, source_stems)
        self._sealed = True


@dataclass(frozen=True)
class TracedSample(Generic[SampleT]):
    """原始 sample 与其唯一 trace 的原子返回值。"""

    sample: SampleT
    trace: SimulationTrace

    def __post_init__(self) -> None:
        if not self.trace.sealed:
            raise ValueError("TracedSample requires a sealed SimulationTrace")


__all__ = ["SimulationTrace", "SourceSimulationTrace", "TracedSample"]
