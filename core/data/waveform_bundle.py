"""同一时间轴上命名 waveform 的不可变纯数据 Module。"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType

import numpy as np


class _FrozenArray(np.ndarray):
    """由 ``WaveformBundle`` 创建的 bytes-backed 只读数组标记。

    该类型只用于内部识别已经完成不可变快照的数组。普通 ndarray 仍然按原
    语义复制，避免调用方持有的可写 base array 通过别的 view 修改 bundle。
    """


def _name(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value

def _freeze_json(value: object, path: str) -> object:
    """冻结 JSON-safe metadata；ndarray 与非有限数 fail-closed。"""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"metadata number at {path} must be finite")
        return number
    if isinstance(value, np.ndarray):
        raise TypeError(f"metadata must not contain ndarray at {path}")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"metadata keys at {path} must be strings")
        return MappingProxyType(
            {key: _freeze_json(item, f"{path}.{key}") for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[{i}]") for i, item in enumerate(value))
    raise TypeError(f"metadata value at {path} is not JSON-safe: {type(value).__name__}")

def _freeze_tracks(tracks: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    if not isinstance(tracks, Mapping):
        raise TypeError("tracks must be a mapping")
    if not tracks:
        raise ValueError("tracks must contain at least one waveform")
    frozen: dict[str, np.ndarray] = {}
    timeline: int | None = None
    for raw_name, waveform in tracks.items():
        name = _name(raw_name, "track name")
        if not isinstance(waveform, np.ndarray):
            raise TypeError(f"track {name!r} must be a numpy.ndarray")
        if waveform.dtype != np.float32:
            raise TypeError(f"track {name!r} must have dtype float32")
        if waveform.ndim < 1:
            raise ValueError(f"track {name!r} must have a final time axis")
        if timeline is not None and waveform.shape[-1] != timeline:
            raise ValueError(f"track {name!r} does not share timeline length {timeline}")
        if not np.isfinite(waveform).all():
            raise ValueError(f"track {name!r} must contain only finite samples")
        timeline = waveform.shape[-1]
        frozen[name] = freeze_waveform_snapshot(waveform)
    return MappingProxyType(frozen)


def freeze_waveform_snapshot(waveform: np.ndarray) -> np.ndarray:
    """返回不可重新打开 writeable flag 的 waveform 快照。

    内部已经冻结的数组直接复用；普通数组仍通过 immutable bytes 复制，保证
    调用方持有的可写 base array 不会改变 bundle 内容。
    """
    if not isinstance(waveform, np.ndarray) or waveform.dtype != np.float32:
        raise TypeError("waveform snapshot requires a float32 ndarray")
    if isinstance(waveform, _FrozenArray) and not waveform.flags.writeable:
        return waveform
    contiguous = np.ascontiguousarray(waveform)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float32).reshape(
        contiguous.shape
    ).view(_FrozenArray)

@dataclass(frozen=True)
class WaveformBundle:
    """一条 timeline 上的命名 finite float32 waveform 集合。

    track 最后一维共享时间长度；所有操作返回独立只读快照。metadata 仅含
    JSON-safe 值且不保存 ndarray。
    """

    tracks: Mapping[str, np.ndarray]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "tracks", _freeze_tracks(self.tracks))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))

    @property
    def num_samples(self) -> int:
        return next(iter(self.tracks.values())).shape[-1]

    def require(self, role: str) -> np.ndarray:
        """读取必需 role；缺失时明确失败。"""
        role = _name(role, "role")
        if role not in self.tracks:
            raise KeyError(f"required waveform role {role!r} is missing")
        return self.tracks[role]

    def replace(self, **changes: object) -> "WaveformBundle":
        """不可变式替换完整 tracks 或 metadata，并重验全部合同。"""
        unknown = set(changes) - {"tracks", "metadata"}
        if unknown:
            raise TypeError(f"unknown replacement fields: {sorted(unknown)}")
        return type(self)(changes.get("tracks", self.tracks), changes.get("metadata", self.metadata))

    def map_tracks(
        self, transform: Callable[[str, np.ndarray], np.ndarray]
    ) -> "WaveformBundle":
        """以外部 plan 提供的纯变换同步映射所有 track。"""
        if not callable(transform):
            raise TypeError("transform must be callable")
        mapped = {}
        for name, waveform in self.tracks.items():
            result = transform(name, waveform)
            if not isinstance(result, np.ndarray):
                raise TypeError(f"transform result for {name!r} must be a numpy.ndarray")
            if result.shape != waveform.shape:
                raise ValueError(f"transform changed shape for {name!r}; broadcast/crop forbidden")
            mapped[name] = result
        return type(self)(mapped, self.metadata)

    def apply_gain(self, gain: float) -> "WaveformBundle":
        """对所有 track 应用同一有限实数 gain。"""
        if isinstance(gain, (bool, np.bool_)) or not isinstance(gain, Real):
            raise TypeError("gain must be a real scalar")
        if not np.isfinite(float(gain)):
            raise ValueError("gain must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            return self.map_tracks(lambda _, wav: np.multiply(wav, gain, dtype=np.float32))

__all__ = ["WaveformBundle", "freeze_waveform_snapshot"]
