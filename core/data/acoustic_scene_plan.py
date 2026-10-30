"""任务无关的声学场景决策与 typed replay Module。

Plan 只保存 JSON-safe 的 identity、抽样决策和 crop；大 waveform 仅存在于
``ResolvedAssets`` 或 replay cache snapshot，不进入 trace/manifest。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Generic, Mapping, TypeVar

import numpy as np

from core.data.acoustic_scene_replay import AssetForcedReplayError
from core.data.simulation.array_profiles import ResolvedMicrophoneArray
from core.data.simulation.trajectory import SourceTrajectory, TrajectoryTimebase


T = TypeVar("T")


# 保留 C-0 旧公开名；新的任务无关 helper 使用 AssetForcedReplayError。
C0AssetReplayError = AssetForcedReplayError


def _freeze_json(value: Any, *, path: str = "value") -> Any:
    """复制并冻结 JSON-safe 值；waveform/ndarray 与非有限数 fail-fast。"""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"AcousticScenePlan JSON-safe value must be finite at {path}")
        return result
    if isinstance(value, np.ndarray):
        raise TypeError(
            f"AcousticScenePlan JSON-safe metadata must not contain ndarray at {path}"
        )
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_json(item, path=f"{path}.{key}")
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"AcousticScenePlan JSON-safe metadata has unsupported value at {path}: "
        f"{type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _validate_nonnegative(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class AudioCropPlan:
    """一个已解析资源窗口；identity 不等同于 waveform。"""

    identity: str
    source_sample_rate: int
    source_num_samples: int
    start_sample: int
    num_samples: int

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("AudioCropPlan.identity must be non-empty")
        for name in (
            "source_sample_rate",
            "source_num_samples",
            "start_sample",
            "num_samples",
        ):
            _validate_nonnegative(getattr(self, name), name)
        if self.source_sample_rate <= 0:
            raise ValueError("source_sample_rate must be positive")
        if self.start_sample + self.num_samples > self.source_num_samples:
            raise ValueError("audio crop exceeds source_num_samples")

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "source_sample_rate": self.source_sample_rate,
            "source_num_samples": self.source_num_samples,
            "start_sample": self.start_sample,
            "num_samples": self.num_samples,
        }


@dataclass(frozen=True)
class RIRPlan:
    """RIR identity、形态与既有 runtime 决策。"""

    enabled: bool
    uid: str | None
    path: str | None
    moving: bool
    reuse_enabled: bool
    cache_hit: bool
    segment_sec: float | None
    normalize: bool
    early_rir_sec: float
    target_type: str
    crop: AudioCropPlan | None = None
    backend: str = "offline"

    def __post_init__(self) -> None:
        if self.enabled and not self.uid:
            raise ValueError("enabled RIRPlan requires uid")
        if self.backend not in {"offline", "online"}:
            raise ValueError("unsupported RIRPlan backend")
        if self.backend == "offline" and self.enabled and not self.cache_hit and not self.path:
            raise ValueError("cold enabled RIRPlan requires path")
        for name in ("segment_sec", "early_rir_sec"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")

    @classmethod
    def disabled(
        cls, *, early_rir_sec: float = 0.0, target_type: str = "early"
    ) -> "RIRPlan":
        return cls(
            enabled=False,
            uid=None,
            path=None,
            moving=False,
            reuse_enabled=False,
            cache_hit=False,
            segment_sec=None,
            normalize=False,
            early_rir_sec=float(early_rir_sec),
            target_type=str(target_type),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            **({"backend": self.backend} if self.backend != "offline" else {}),
            "uid": self.uid,
            "path": self.path,
            "moving": self.moving,
            "reuse_enabled": self.reuse_enabled,
            "cache_hit": self.cache_hit,
            "segment_sec": self.segment_sec,
            "normalize": self.normalize,
            "early_rir_sec": self.early_rir_sec,
            "target_type": self.target_type,
            "crop": self.crop.to_dict() if self.crop is not None else None,
        }


@dataclass(frozen=True)
class SourceGainPlan:
    """post-RIR source gain 决策。"""

    enabled: bool
    target_db: float | None
    gain: float
    forced: bool
    reference: str | None

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.gain)) or self.gain <= 0.0:
            raise ValueError("SourceGainPlan.gain must be finite and positive")
        if self.target_db is not None and not np.isfinite(float(self.target_db)):
            raise ValueError("SourceGainPlan.target_db must be finite or None")

    @classmethod
    def disabled(cls) -> "SourceGainPlan":
        return cls(
            enabled=False,
            target_db=None,
            gain=1.0,
            forced=False,
            reference=None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "target_db": self.target_db,
            "gain": self.gain,
            "forced": self.forced,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class NoiseSourcePlan:
    """一个实际选择的 noise/music/wind source 决策。"""

    family: str
    uid: str | None
    path: str | None
    category_id: int | None = None
    category_name: str | None = None
    crop: AudioCropPlan | None = None
    snr_db: float | None = None
    applied: bool = True
    cache_hit: bool = False

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("NoiseSourcePlan.family must be non-empty")
        if self.applied and (not self.uid or (not self.path and not self.cache_hit)):
            raise ValueError(
                "applied NoiseSourcePlan requires uid and path unless cache_hit"
            )
        if self.snr_db is not None and not np.isfinite(float(self.snr_db)):
            raise ValueError("NoiseSourcePlan.snr_db must be finite or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "uid": self.uid,
            "path": self.path,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "crop": self.crop.to_dict() if self.crop is not None else None,
            "snr_db": self.snr_db,
            "applied": self.applied,
            "cache_hit": self.cache_hit,
        }


@dataclass(frozen=True)
class NoisePlan:
    """一次 Layer C noise 决策聚合。"""

    enabled: bool
    mode: str
    mix_ratio: float | None
    sources: tuple[NoiseSourcePlan, ...]
    colored_enabled: bool
    colored_type: str | None
    colored_snr_db: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))
        for name in ("mix_ratio", "colored_snr_db"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(f"NoisePlan.{name} must be finite or None")

    @classmethod
    def disabled(cls) -> "NoisePlan":
        return cls(
            enabled=False,
            mode="none",
            mix_ratio=None,
            sources=(),
            colored_enabled=False,
            colored_type=None,
            colored_snr_db=None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "mix_ratio": self.mix_ratio,
            "sources": [source.to_dict() for source in self.sources],
            "colored_enabled": self.colored_enabled,
            "colored_type": self.colored_type,
            "colored_snr_db": self.colored_snr_db,
        }


@dataclass(frozen=True, slots=True)
class AcousticActorPlan:
    """一个具有稳定身份的 sample-level 空间声学 actor。

    该对象只描述 room/array 共享场景中的 actor 身份和几何；waveform、RIR
    以及任务监督对象仍由后续 renderer/task 层管理。
    """

    actor_uid: str
    role: str
    supervision_index: int | None = None
    position_m: tuple[float, float, float] | None = None
    trajectory: SourceTrajectory | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.actor_uid, str) or not self.actor_uid.strip():
            raise ValueError("actor_uid must be a non-empty string")
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("actor role must be a non-empty string")
        object.__setattr__(self, "actor_uid", self.actor_uid.strip())
        object.__setattr__(self, "role", self.role.strip())
        if self.supervision_index is not None:
            if type(self.supervision_index) is not int or self.supervision_index < 0:
                raise ValueError("supervision_index must be a non-negative integer or None")
        if self.position_m is not None:
            position = tuple(float(value) for value in self.position_m)
            if len(position) != 3 or not np.isfinite(position).all():
                raise ValueError("position_m must contain three finite values")
            object.__setattr__(self, "position_m", position)
        if self.position_m is None and self.trajectory is None:
            raise ValueError("actor requires position_m or trajectory")

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_uid": self.actor_uid,
            "role": self.role,
            "supervision_index": self.supervision_index,
            "position_m": list(self.position_m) if self.position_m is not None else None,
            "trajectory": self.trajectory.to_dict() if self.trajectory is not None else None,
        }


@dataclass(frozen=True)
class AcousticScenePlan:
    """共享 C-0~C-3 的最小不可变 Interface。"""

    sample_rate: int
    num_samples: int
    need_noise: bool
    rir: RIRPlan
    source_gain: SourceGainPlan
    noise: NoisePlan
    identity: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    room: Mapping[str, Any] | None = None
    array: ResolvedMicrophoneArray | None = None
    timebase: TrajectoryTimebase | None = None
    actors: tuple[AcousticActorPlan, ...] = ()

    def __post_init__(self) -> None:
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        _validate_nonnegative(self.num_samples, "num_samples")
        if self.schema_version != 1:
            raise ValueError("unsupported AcousticScenePlan schema_version")
        object.__setattr__(self, "identity", _freeze_json(self.identity, path="identity"))
        if self.room is not None:
            object.__setattr__(self, "room", _freeze_json(self.room, path="room"))
        object.__setattr__(self, "actors", tuple(self.actors))
        for actor in self.actors:
            if not isinstance(actor, AcousticActorPlan):
                raise TypeError("AcousticScenePlan.actors must contain AcousticActorPlan values")
        actor_uids = [actor.actor_uid for actor in self.actors]
        if len(actor_uids) != len(set(actor_uids)):
            raise ValueError("AcousticScenePlan actor_uid values must be unique")
        for actor in self.actors:
            if actor.trajectory is not None:
                if self.timebase is None or actor.trajectory.timebase != self.timebase:
                    raise ValueError("actor trajectory must share the scene timebase")
        if self.timebase is not None:
            if self.timebase.sample_rate != self.sample_rate or self.timebase.num_samples != self.num_samples:
                raise ValueError("scene timebase must match sample_rate and num_samples")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "sample_rate": self.sample_rate,
            "num_samples": self.num_samples,
            "need_noise": self.need_noise,
            "identity": _thaw_json(self.identity),
            "rir": self.rir.to_dict(),
            "source_gain": self.source_gain.to_dict(),
            "noise": self.noise.to_dict(),
        }
        if self.room is not None:
            payload["room"] = _thaw_json(self.room)
        if self.array is not None:
            payload["array"] = self.array.to_dict()
        if self.timebase is not None:
            payload["timebase"] = self.timebase.to_dict()
        if self.actors:
            payload["actors"] = [actor.to_dict() for actor in self.actors]
        # 统一走 strict validator，防止后续字段扩展把 waveform 混入 plan。
        return _thaw_json(_freeze_json(payload, path="scene"))


@dataclass(frozen=True)
class ResolvedAssets:
    """runtime 大资产的独立快照发布边界；复制为连续只读数组，永不序列化。"""

    rir: np.ndarray | None = None
    noises: tuple[np.ndarray, ...] = ()
    role: str | None = None
    sample_rate: int | None = None
    num_samples: int | None = None

    def __post_init__(self) -> None:
        if self.role is not None and (not isinstance(self.role, str) or not self.role):
            raise ValueError("ResolvedAssets.role must be a non-empty string or None")
        for name in ("sample_rate", "num_samples"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"ResolvedAssets.{name} must be a non-negative integer or None")
        if self.sample_rate == 0:
            raise ValueError("ResolvedAssets.sample_rate must be positive when provided")
        rir = (
            None if self.rir is None
            else np.array(self.rir, dtype=np.float32, order="C", copy=True)
        )
        if rir is not None:
            rir.setflags(write=False)
        noises = tuple(
            np.array(item, dtype=np.float32, order="C", copy=True) for item in self.noises
        )
        for item in noises:
            item.setflags(write=False)
        object.__setattr__(self, "rir", rir)
        object.__setattr__(self, "noises", noises)


@dataclass(frozen=True)
class RIRCacheSnapshot:
    """兼容 legacy 2/3/4 元组的 typed RIR cache。"""

    sample_rate: int
    waveform: np.ndarray
    tail: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        waveform = np.asarray(self.waveform, dtype=np.float32).copy()
        waveform.setflags(write=False)
        object.__setattr__(self, "waveform", waveform)
        object.__setattr__(self, "tail", deepcopy(tuple(self.tail)))

    @classmethod
    def from_legacy(cls, value: Any) -> "RIRCacheSnapshot | None":
        if value is None:
            return None
        if not isinstance(value, tuple) or len(value) < 2:
            raise TypeError("_rir_reuse_cache must be None or a tuple with at least 2 items")
        return cls(int(value[0]), value[1], tuple(value[2:]))

    def to_legacy(self) -> tuple[Any, ...]:
        return (self.sample_rate, self.waveform.copy(), *deepcopy(self.tail))


@dataclass(frozen=True)
class AcousticReplaySnapshot:
    """一个可恢复的 RNG cursor + RIR cache typed snapshot。"""

    rng_state: Mapping[str, Any]
    rir_cache: RIRCacheSnapshot | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rng_state", deepcopy(dict(self.rng_state)))


@dataclass(frozen=True)
class AcousticReplayResult(Generic[T]):
    value: T
    completion: AcousticReplaySnapshot


class AcousticReplayAdapter:
    """封装 NumPy RNG 与 Dataset RIR cache 的 capture/restore/replay。"""

    __slots__ = ("_rng", "_owner", "_cache_attribute")

    def __init__(
        self,
        rng: np.random.Generator,
        owner: Any,
        *,
        cache_attribute: str = "_rir_reuse_cache",
    ) -> None:
        self._rng = rng
        self._owner = owner
        self._cache_attribute = str(cache_attribute)

    def capture(self) -> AcousticReplaySnapshot:
        return AcousticReplaySnapshot(
            rng_state=deepcopy(self._rng.bit_generator.state),
            rir_cache=RIRCacheSnapshot.from_legacy(
                getattr(self._owner, self._cache_attribute, None)
            ),
        )

    def restore(self, snapshot: AcousticReplaySnapshot) -> None:
        self._rng.bit_generator.state = deepcopy(dict(snapshot.rng_state))
        cache = snapshot.rir_cache
        setattr(
            self._owner,
            self._cache_attribute,
            None if cache is None else cache.to_legacy(),
        )

    def replay(
        self,
        snapshot: AcousticReplaySnapshot,
        operation: Callable[[], T],
    ) -> AcousticReplayResult[T]:
        self.restore(snapshot)
        value = operation()
        return AcousticReplayResult(value=value, completion=self.capture())


__all__ = [
    "AcousticReplayAdapter",
    "C0AssetReplayError",
    "AcousticReplayResult",
    "AcousticReplaySnapshot",
    "AcousticScenePlan",
    "AcousticActorPlan",
    "AudioCropPlan",
    "NoisePlan",
    "NoiseSourcePlan",
    "RIRCacheSnapshot",
    "RIRPlan",
    "ResolvedAssets",
    "SourceGainPlan",
]
