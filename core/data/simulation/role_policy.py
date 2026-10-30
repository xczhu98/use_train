"""声学源角色、RIR 与背景策略的共享合同。

``AcousticActorPlan`` 继续只描述 room/array 几何；这里的 ``AcousticActor``
是把已经准备好的 mono waveform 绑定到 renderer 的 runtime seam。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np

from core.data.simulation.array_profiles import resolve_geometry_positions


class AcousticRole(str, Enum):
    TARGET_SPEECH = "target_speech"
    INTERFERENCE_SPEECH = "interference_speech"
    SOURCE_SPEECH = "source_speech"
    NEAREND = "nearend"
    LOUDSPEAKER_ECHO = "loudspeaker_echo"
    ENROLLMENT = "enrollment"
    DIGITAL_REFERENCE = "digital_reference"


class RIRPolicy(str, Enum):
    NONE = "none"
    MONO_SOURCE = "mono_source"
    SPATIAL_SOURCE = "spatial_source"
    NEARFIELD_INDEPENDENT = "nearfield_independent"


class BackgroundPolicy(str, Enum):
    NONE = "none"
    ADDITIVE_MONO = "additive_mono"
    DIFFUSE_COHERENCE = "diffuse_coherence"
    POINT_SOURCE_RIR = "point_source_rir"
    DIRECT_MULTICHANNEL = "direct_multichannel"
    DEVICE_SIDECHAIN = "device_sidechain"


_ACTOR_POLICY_MATRIX = {
    AcousticRole.TARGET_SPEECH: frozenset(
        {RIRPolicy.MONO_SOURCE, RIRPolicy.SPATIAL_SOURCE}
    ),
    AcousticRole.INTERFERENCE_SPEECH: frozenset(
        {RIRPolicy.MONO_SOURCE, RIRPolicy.SPATIAL_SOURCE}
    ),
    AcousticRole.SOURCE_SPEECH: frozenset(
        {RIRPolicy.MONO_SOURCE, RIRPolicy.SPATIAL_SOURCE}
    ),
    AcousticRole.NEAREND: frozenset(
        {RIRPolicy.MONO_SOURCE, RIRPolicy.SPATIAL_SOURCE}
    ),
    AcousticRole.LOUDSPEAKER_ECHO: frozenset(
        {RIRPolicy.MONO_SOURCE, RIRPolicy.SPATIAL_SOURCE}
    ),
    AcousticRole.ENROLLMENT: frozenset({RIRPolicy.NONE, RIRPolicy.NEARFIELD_INDEPENDENT}),
    AcousticRole.DIGITAL_REFERENCE: frozenset({RIRPolicy.NONE}),
}


def _enum_value(enum_type, value):
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"invalid {enum_type.__name__}={value!r}; choices: {choices}") from exc


def validate_actor_policy(role: AcousticRole | str, rir_policy: RIRPolicy | str) -> tuple[AcousticRole, RIRPolicy]:
    """校验角色与 RIR 策略的组合，并返回规范化枚举。"""

    role = _enum_value(AcousticRole, role)
    rir_policy = _enum_value(RIRPolicy, rir_policy)
    if rir_policy not in _ACTOR_POLICY_MATRIX[role]:
        raise ValueError(f"RIR policy {rir_policy.value!r} is invalid for role {role.value!r}")
    return role, rir_policy


@dataclass(frozen=True)
class AcousticActor:
    uid: str
    role: AcousticRole | str
    rir_policy: RIRPolicy | str
    waveform: np.ndarray
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self):
        uid = str(self.uid).strip()
        if not uid:
            raise ValueError("AcousticActor.uid must be a non-empty string")
        waveform = np.asarray(self.waveform, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("AcousticActor.waveform must be a non-empty mono [T] array")
        if not np.isfinite(waveform).all():
            raise ValueError("AcousticActor.waveform must contain only finite values")
        role, rir_policy = validate_actor_policy(self.role, self.rir_policy)
        object.__setattr__(self, "uid", uid)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "rir_policy", rir_policy)
        object.__setattr__(self, "waveform", waveform)
        if self.metadata is not None:
            object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """返回可写入 provenance 的语义字段，不复制波形。"""

        return {
            "uid": self.uid,
            "role": self.role.value,
            "rir_policy": self.rir_policy.value,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class BackgroundSpec:
    policy: BackgroundPolicy | str
    uid: str | None = None
    geometry_bound: bool = False
    shared_scene: bool = False
    point_source_provider: Any | None = None

    def __post_init__(self):
        policy = _enum_value(BackgroundPolicy, self.policy)
        uid = None if self.uid is None else str(self.uid).strip()
        if self.uid is not None and not uid:
            raise ValueError("BackgroundSpec.uid must be non-empty when provided")
        if policy in {
            BackgroundPolicy.DIFFUSE_COHERENCE,
            BackgroundPolicy.DIRECT_MULTICHANNEL,
            BackgroundPolicy.POINT_SOURCE_RIR,
        } and not self.geometry_bound:
            raise ValueError(f"{policy.value} background requires geometry_bound=True")
        if policy == BackgroundPolicy.POINT_SOURCE_RIR:
            if not callable(getattr(self.point_source_provider, "render", None)):
                raise TypeError(
                    "point_source_rir requires a provider with render(waveform, sample_rate=...)"
                )
            provider_geometry = getattr(self.point_source_provider, "geometry", None)
            try:
                resolve_geometry_positions(
                    provider_geometry,
                    name="point_source_rir provider geometry",
                    max_norm_m=None,
                )
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "point_source_rir provider must expose finite microphone geometry"
                ) from exc
        elif self.point_source_provider is not None:
            raise ValueError("point_source_provider is only valid for point_source_rir")
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "uid", uid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "uid": self.uid,
            "geometry_bound": self.geometry_bound,
            "shared_scene": self.shared_scene,
            "point_source_provider": (
                type(self.point_source_provider).__name__
                if self.point_source_provider is not None else None
            ),
        }


def build_background_spec(
    *,
    image_geometry: Any | None,
    shared_scene: bool = False,
    point_source_provider: Any | None = None,
) -> BackgroundSpec:
    """按 observation geometry 选择统一背景策略。

    ``point_source_provider`` 是显式 runtime 注入 seam；它不能从 geometry
    猜测 source 位置。未注入 provider 时，空间 observation 使用
    ``diffuse_coherence``，单通道 observation 使用 ``additive_mono``。
    """

    if point_source_provider is not None:
        if image_geometry is None:
            raise ValueError("point_source_rir requires observation geometry")
        return BackgroundSpec(
            BackgroundPolicy.POINT_SOURCE_RIR,
            geometry_bound=True,
            shared_scene=shared_scene,
            point_source_provider=point_source_provider,
        )
    if image_geometry is None:
        return BackgroundSpec(BackgroundPolicy.ADDITIVE_MONO)
    return BackgroundSpec(
        BackgroundPolicy.DIFFUSE_COHERENCE,
        geometry_bound=True,
        shared_scene=shared_scene,
    )


def validate_scene_actor_render_mode(
    sources, *, spatial_enabled: bool
) -> None:
    """校验 runtime actor 的 RIR policy 与观测通道模式一致。

    ``NONE`` 只适用于不进入 speech scene 的 cue/reference actor；一旦进入
    renderer，必须显式声明 mono 或 spatial source policy，避免事实字段与实际
    渲染步骤不一致。
    """

    actors = tuple(sources)
    if not actors:
        return
    if not all(isinstance(item, AcousticActor) for item in actors):
        raise TypeError("scene actor validation requires AcousticActor entries")
    expected = RIRPolicy.SPATIAL_SOURCE if spatial_enabled else RIRPolicy.MONO_SOURCE
    incompatible = [
        (actor.uid, actor.rir_policy.value)
        for actor in actors
        if actor.rir_policy != expected
    ]
    if incompatible:
        mode = "spatial" if spatial_enabled else "mono"
        raise ValueError(
            f"AcousticActor RIR policy is incompatible with {mode} scene: {incompatible!r}"
        )


__all__ = [
    "AcousticActor",
    "AcousticRole",
    "BackgroundPolicy",
    "BackgroundSpec",
    "build_background_spec",
    "RIRPolicy",
    "validate_scene_actor_render_mode",
    "validate_actor_policy",
]
