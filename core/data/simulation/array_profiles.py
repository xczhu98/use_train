"""任务无关的物理麦克风阵列 profile 与 geometry provenance。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any


_COORDINATE_SYSTEM = "cartesian_xyz_m"
_ALLOWED_TOPOLOGY_KEYS = frozenset({"positions_m", "spacing_m", "coordinate_system"})
_ALLOWED_PROVENANCE_KEYS = frozenset(
    {"source", "external_repository", "external_path", "external_commit", "notes"}
)


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _normalize_positions(
    value: Any,
    *,
    name: str = "positions_m",
    max_norm_m: float | None = 10.0,
) -> tuple[tuple[float, float, float], ...]:
    if isinstance(value, (str, bytes)) or value is None:
        raise ValueError(f"{name} must have shape [C, 3]")
    try:
        rows = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must have shape [C, 3]") from exc
    if not rows:
        raise ValueError(f"{name} must contain at least one microphone")
    normalized = []
    for index, row in enumerate(rows):
        if isinstance(row, (str, bytes)):
            raise ValueError(f"{name} must have shape [C, 3], invalid row {index}")
        try:
            row_values = tuple(row)
        except TypeError as exc:
            raise ValueError(f"{name} must have shape [C, 3], invalid row {index}") from exc
        if len(row_values) != 3:
            raise ValueError(f"{name} must have shape [C, 3], invalid row {index}")
        normalized.append(tuple(_finite_real(item, name=f"{name}[{index}][{axis}]") for axis, item in enumerate(row_values)))
    if len(normalized) > 1:
        for left in range(len(normalized)):
            for right in range(left):
                distance = math.sqrt(sum((normalized[left][axis] - normalized[right][axis]) ** 2 for axis in range(3)))
                if distance < 1e-5:
                    raise ValueError(f"{name} must not contain duplicate microphones")
    if max_norm_m is not None and max(
        math.sqrt(sum(item * item for item in position)) for position in normalized
    ) > max_norm_m:
        raise ValueError(f"{name} exceeds the 10 m geometry budget")
    return tuple(normalized)


def _normalize_coordinate_system(value: Any) -> str:
    if not isinstance(value, str) or value not in {_COORDINATE_SYSTEM, "cartesian"}:
        raise ValueError(f"coordinate_system must be {_COORDINATE_SYSTEM!r}")
    return _COORDINATE_SYSTEM


@dataclass(frozen=True, slots=True)
class ArrayTopology:
    """纯物理阵列拓扑，不包含任务、模型或数据集名称。"""

    positions_m: tuple[tuple[float, float, float], ...]
    spacing_m: float | None = None
    coordinate_system: str = _COORDINATE_SYSTEM
    _max_norm_m: float | None = field(default=10.0, repr=False, compare=False, kw_only=True)

    def __post_init__(self) -> None:
        if self._max_norm_m is not None and (
            isinstance(self._max_norm_m, bool)
            or not isinstance(self._max_norm_m, Real)
            or not math.isfinite(float(self._max_norm_m))
            or float(self._max_norm_m) <= 0.0
        ):
            raise ValueError("_max_norm_m must be positive and finite or None")
        positions = _normalize_positions(self.positions_m, max_norm_m=self._max_norm_m)
        coordinate_system = _normalize_coordinate_system(self.coordinate_system)
        spacing = None if self.spacing_m is None else _finite_real(self.spacing_m, name="spacing_m")
        if spacing is not None and spacing <= 0.0:
            raise ValueError("spacing_m must be positive")
        if spacing is not None and len(positions) > 1:
            adjacent = tuple(
                math.sqrt(sum((positions[index + 1][axis] - positions[index][axis]) ** 2 for axis in range(3)))
                for index in range(len(positions) - 1)
            )
            if any(not math.isclose(value, spacing, rel_tol=1e-6, abs_tol=1e-7) for value in adjacent):
                raise ValueError("spacing_m must match every adjacent microphone distance")
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "spacing_m", spacing)
        object.__setattr__(self, "coordinate_system", coordinate_system)

    @property
    def channels(self) -> int:
        return len(self.positions_m)

    @property
    def num_channels(self) -> int:
        return self.channels

    @property
    def C(self) -> int:
        return self.channels

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArrayTopology":
        if not isinstance(value, Mapping):
            raise TypeError("array topology must be a mapping")
        unknown = set(value) - _ALLOWED_TOPOLOGY_KEYS
        if unknown:
            raise ValueError(f"array topology has unknown keys: {sorted(unknown)!r}")
        if "positions_m" not in value:
            raise ValueError("array topology.positions_m is required")
        return cls(
            positions_m=value["positions_m"],
            spacing_m=value.get("spacing_m"),
            coordinate_system=value.get("coordinate_system", _COORDINATE_SYSTEM),
        )

    @classmethod
    def from_scene_positions(
        cls,
        value: Any,
        *,
        spacing_m: float | None = None,
        coordinate_system: str = _COORDINATE_SYSTEM,
    ) -> "ArrayTopology":
        """构造房间坐标系中的麦克风位置，不套用阵列原点预算。"""
        return cls(
            positions_m=value,
            spacing_m=spacing_m,
            coordinate_system=coordinate_system,
            _max_norm_m=None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "positions_m": [list(position) for position in self.positions_m],
            "spacing_m": self.spacing_m,
            "coordinate_system": self.coordinate_system,
            "channels": self.channels,
        }


@dataclass(frozen=True, slots=True)
class MicrophoneArrayProfile:
    """命名的物理阵列 profile；命名不携带任务或模型语义。"""

    name: str
    topology: ArrayTopology

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("array profile name must be a non-empty string")
        if not isinstance(self.topology, ArrayTopology):
            raise TypeError("array profile topology must be an ArrayTopology")
        object.__setattr__(self, "name", self.name.strip())

    @property
    def positions_m(self) -> tuple[tuple[float, float, float], ...]:
        return self.topology.positions_m

    @property
    def channels(self) -> int:
        return self.topology.channels

    @property
    def num_channels(self) -> int:
        return self.channels

    @property
    def C(self) -> int:
        return self.channels

    @property
    def spacing_m(self) -> float | None:
        return self.topology.spacing_m

    @property
    def coordinate_system(self) -> str:
        return self.topology.coordinate_system

    def resolve(self, reference_channel: int = 0) -> "ResolvedMicrophoneArray":
        return ResolvedMicrophoneArray(self, reference_channel)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "topology": self.topology.to_dict()}


@dataclass(frozen=True, slots=True)
class ResolvedMicrophoneArray:
    """结合 reference channel 后供 runtime/RIR 使用的 geometry。"""

    profile: MicrophoneArrayProfile
    reference_channel: int = 0

    def __post_init__(self) -> None:
        if type(self.reference_channel) is not int or not 0 <= self.reference_channel < self.profile.channels:
            raise ValueError(
                f"reference_channel must be an exact int in [0, {self.profile.channels})"
            )

    @property
    def positions_m(self):
        return self.profile.positions_m

    @property
    def channels(self) -> int:
        return self.profile.channels

    def to_dict(self) -> dict[str, Any]:
        return {
            "array_profile": None if self.profile.name == "custom" else self.profile.name,
            "geometry_kind": "custom" if self.profile.name == "custom" else "profile",
            "reference_channel": self.reference_channel,
            "topology": self.profile.topology.to_dict(),
        }


def _circular_positions(radius_m: float, channels: int) -> tuple[tuple[float, float, float], ...]:
    """等角分布在 xy 平面圆周上的麦克风坐标，原点为 array center。"""
    return tuple(
        (
            radius_m * math.cos(2.0 * math.pi * index / channels),
            radius_m * math.sin(2.0 * math.pi * index / channels),
            0.0,
        )
        for index in range(channels)
    )


_CIRCULAR_8CH_R80MM_POSITIONS = _circular_positions(0.08, 8)
_CIRCULAR_8CH_R80MM_CHORD_M = math.dist(
    _CIRCULAR_8CH_R80MM_POSITIONS[0], _CIRCULAR_8CH_R80MM_POSITIONS[1]
)
_CHIME3_6CH_POSITIONS = (
    (-0.10, 0.095, 0.0),
    (0.0, 0.095, 0.0),
    (0.10, 0.095, 0.0),
    (-0.10, -0.095, 0.0),
    (0.0, -0.095, 0.0),
    (0.10, -0.095, 0.0),
)


_CANONICAL_PROFILES = {
    "linear_2ch_80mm": MicrophoneArrayProfile(
        "linear_2ch_80mm",
        ArrayTopology(((-0.04, 0.0, 0.0), (0.04, 0.0, 0.0)), spacing_m=0.08),
    ),
    "linear_4ch_40mm": MicrophoneArrayProfile(
        "linear_4ch_40mm",
        ArrayTopology(((-0.06, 0.0, 0.0), (-0.02, 0.0, 0.0), (0.02, 0.0, 0.0), (0.06, 0.0, 0.0)), spacing_m=0.04),
    ),
    "linear_6ch_40mm": MicrophoneArrayProfile(
        "linear_6ch_40mm",
        ArrayTopology(((-0.10, 0.0, 0.0), (-0.06, 0.0, 0.0), (-0.02, 0.0, 0.0), (0.02, 0.0, 0.0), (0.06, 0.0, 0.0), (0.10, 0.0, 0.0)), spacing_m=0.04),
    ),
    "chime3_6ch_rect_100x190mm": MicrophoneArrayProfile(
        "chime3_6ch_rect_100x190mm", ArrayTopology(_CHIME3_6CH_POSITIONS)
    ),
    # 8 通道圆阵：半径 80 mm、等角分布；spacing_m 记录相邻弦长而非半径。
    "circular_8ch_r80mm": MicrophoneArrayProfile(
        "circular_8ch_r80mm",
        ArrayTopology(_CIRCULAR_8CH_R80MM_POSITIONS, spacing_m=_CIRCULAR_8CH_R80MM_CHORD_M),
    ),
}

ARRAY_PROFILES = dict(_CANONICAL_PROFILES)
CANONICAL_ARRAY_PROFILES = ARRAY_PROFILES


def get_array_profile(name: str) -> MicrophoneArrayProfile:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("array_profile must be a non-empty string")
    try:
        return ARRAY_PROFILES[name.strip()]
    except KeyError as exc:
        raise ValueError(
            f"unknown array_profile {name!r}; expected one of {sorted(ARRAY_PROFILES)!r}"
        ) from exc


def normalize_geometry_provenance(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("geometry_provenance must be a mapping or None")
    unknown = set(value) - _ALLOWED_PROVENANCE_KEYS
    if unknown:
        raise ValueError(
            f"geometry_provenance has unknown keys: {sorted(unknown)!r}; "
            f"canonical keys are {sorted(_ALLOWED_PROVENANCE_KEYS)!r} "
            "(external repository/path/commit must use the external_* names)"
        )
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key == "notes":
            if not isinstance(item, str):
                raise ValueError("geometry_provenance.notes must be a string")
        elif not isinstance(item, str) or not item.strip():
            raise ValueError(f"geometry_provenance.{key} must be a non-empty string")
        result[key] = item.strip() if isinstance(item, str) else item
    return result


def _normalize_validation_value(value: Any, *, path: str) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must be JSON-safe")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{path} keys must be non-empty strings")
            normalized_key = key.strip()
            result[normalized_key] = _normalize_validation_value(
                item, path=f"{path}.{normalized_key}"
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _normalize_validation_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{path} must be JSON-safe")


def normalize_validation_recipe(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("validation_recipe must be a mapping or None")
    normalized = _normalize_validation_value(value, path="validation_recipe")
    assert isinstance(normalized, dict)
    return normalized


def validate_reference_channel(reference_channel: int, num_channels: int) -> int:
    if type(num_channels) is not int or num_channels < 1:
        raise ValueError("num_channels must be an exact int >= 1")
    if type(reference_channel) is not int or not 0 <= reference_channel < num_channels:
        raise ValueError(f"reference_channel must be an exact int in [0, {num_channels})")
    return reference_channel


def resolve_microphone_array(
    *,
    array_profile: str | None = None,
    custom_geometry: Mapping[str, Any] | Sequence[Sequence[Real]] | None = None,
    reference_channel: int = 0,
) -> ResolvedMicrophoneArray | None:
    if array_profile is not None and custom_geometry is not None:
        raise ValueError(
            "array_profile and custom microphone geometry are mutually exclusive; configure only one"
        )
    if array_profile is None and custom_geometry is None:
        return None
    if array_profile is not None:
        profile = get_array_profile(array_profile)
    elif isinstance(custom_geometry, Mapping):
        profile = MicrophoneArrayProfile("custom", ArrayTopology.from_mapping(custom_geometry))
    else:
        profile = MicrophoneArrayProfile("custom", ArrayTopology(custom_geometry))
    return profile.resolve(reference_channel)


def resolve_scene_microphone_array(
    *,
    positions_m: Sequence[Sequence[Real]],
    room_dim_m: Sequence[Real] | None,
    spacing_m: float | None = None,
    coordinate_system: str = _COORDINATE_SYSTEM,
    reference_channel: int = 0,
) -> ResolvedMicrophoneArray:
    """解析房间坐标系中的麦克风位置并在有房间尺寸时校验边界。

    ``resolve_microphone_array`` 接收的是以阵列中心为原点的拓扑坐标，
    因而带有 10 m 范围预算；scene plan 则保存 gpuRIR 所需的房间绝对坐标，
    两者必须使用不同的校验入口。
    """
    topology = ArrayTopology.from_scene_positions(
        positions_m,
        spacing_m=spacing_m,
        coordinate_system=coordinate_system,
    )
    if room_dim_m is None:
        return MicrophoneArrayProfile("custom", topology).resolve(reference_channel)
    if isinstance(room_dim_m, (str, bytes)):
        raise ValueError("room_dim_m must have shape [3]")
    try:
        room = tuple(_finite_real(value, name=f"room_dim_m[{index}]") for index, value in enumerate(room_dim_m))
    except TypeError as exc:
        raise ValueError("room_dim_m must have shape [3]") from exc
    if len(room) != 3 or any(value <= 0.0 for value in room):
        raise ValueError("room_dim_m must have shape [3] with positive dimensions")
    if any(
        position[axis] <= 0.0 or position[axis] >= room[axis]
        for position in topology.positions_m
        for axis in range(3)
    ):
        raise ValueError("scene microphone positions must lie strictly inside room_dim_m")
    return MicrophoneArrayProfile("custom", topology).resolve(reference_channel)


def resolve_microphone_positions(
    *,
    array_profile: str | None = None,
    custom_geometry: Mapping[str, Any] | Sequence[Sequence[Real]] | None = None,
    reference_channel: int = 0,
) -> tuple[tuple[float, float, float], ...] | None:
    resolved = resolve_microphone_array(
        array_profile=array_profile,
        custom_geometry=custom_geometry,
        reference_channel=reference_channel,
    )
    return None if resolved is None else resolved.positions_m


def resolve_geometry_positions(
    geometry: Mapping[str, Any] | Sequence[Sequence[Real]],
    *,
    expected_channels: int | None = None,
    name: str = "geometry",
    max_norm_m: float | None = 10.0,
) -> tuple[tuple[float, float, float], ...]:
    """从 RIR/scene geometry 读取唯一的麦克风位置合同。

    runtime provider 可能携带 ``positions_m``、``mic_loc`` 或历史的
    ``microphone_positions_m`` 别名；``topology.positions_m`` 也允许作为
    结构化 geometry 的来源。多个别名同时存在时必须逐点一致，避免同一
    scene 在不同 adapter 中被解释成不同阵列。默认 ``max_norm_m=10``
    校验阵列局部坐标；scene 的房间绝对坐标调用方应显式传入 ``None``。
    """

    candidates: list[tuple[str, Any]] = []
    if isinstance(geometry, Mapping):
        for key in ("positions_m", "mic_loc", "microphone_positions_m"):
            if key in geometry and geometry[key] is not None:
                candidates.append((f"{name}.{key}", geometry[key]))
        topology = geometry.get("topology")
        if isinstance(topology, Mapping) and topology.get("positions_m") is not None:
            candidates.append((f"{name}.topology.positions_m", topology["positions_m"]))
        if not candidates:
            raise ValueError(
                f"{name} must declare positions_m, mic_loc, "
                "microphone_positions_m, or topology.positions_m"
            )
    else:
        candidates.append((name, geometry))

    if max_norm_m is not None and (
        isinstance(max_norm_m, bool)
        or not isinstance(max_norm_m, Real)
        or not math.isfinite(float(max_norm_m))
        or float(max_norm_m) <= 0.0
    ):
        raise ValueError("max_norm_m must be positive and finite or None")
    normalized = [
        (
            candidate_name,
            _normalize_positions(
                value,
                name=candidate_name,
                max_norm_m=max_norm_m,
            ),
        )
        for candidate_name, value in candidates
    ]
    positions = normalized[0][1]
    if expected_channels is not None:
        if type(expected_channels) is not int or expected_channels < 1:
            raise ValueError("expected_channels must be a positive exact int")
        if len(positions) != expected_channels:
            raise ValueError(
                f"{name} has C={len(positions)}, expected C={expected_channels}"
            )
    for candidate_name, candidate in normalized[1:]:
        if len(candidate) != len(positions) or any(
            not math.isclose(
                candidate[row][axis],
                positions[row][axis],
                rel_tol=0.0,
                abs_tol=1e-7,
            )
            for row in range(len(positions))
            for axis in range(3)
        ):
            raise ValueError(
                f"{name} geometry aliases disagree: "
                f"{normalized[0][0]} vs {candidate_name}"
            )
    return positions


def build_geometry_metadata(
    *,
    array_profile: str | None = None,
    custom_geometry: Mapping[str, Any] | Sequence[Sequence[Real]] | None = None,
    reference_channel: int = 0,
    num_channels: int | None = None,
    external: Mapping[str, Any] | None = None,
    validation_recipe: Mapping[str, Any] | None = None,
    scene_geometry: bool = False,
) -> dict[str, Any]:
    """生成可写入 RIR、scene 与 manifest 的 geometry provenance。"""
    if type(scene_geometry) is not bool:
        raise TypeError("scene_geometry must be an exact bool")
    if scene_geometry:
        if array_profile is not None:
            raise ValueError("scene geometry cannot use array_profile")
        topology = (
            custom_geometry.get("topology", {})
            if isinstance(custom_geometry, Mapping)
            else {}
        )
        if not isinstance(topology, Mapping):
            topology = {}
        if isinstance(custom_geometry, Mapping):
            positions = custom_geometry.get("positions_m")
            if positions is None:
                positions = topology.get("positions_m")
            spacing = custom_geometry.get("spacing_m", topology.get("spacing_m"))
            coordinate_system = custom_geometry.get(
                "coordinate_system", topology.get("coordinate_system", _COORDINATE_SYSTEM)
            )
            room_dim = custom_geometry.get("room_dim_m")
            if room_dim is None and isinstance(custom_geometry.get("room"), Mapping):
                room_dim = custom_geometry["room"].get("room_dim_m")
        else:
            positions = custom_geometry
            spacing = None
            coordinate_system = _COORDINATE_SYSTEM
            room_dim = None
        resolved = resolve_scene_microphone_array(
            positions_m=positions,
            room_dim_m=room_dim,
            spacing_m=spacing,
            coordinate_system=coordinate_system,
            reference_channel=reference_channel,
        )
    else:
        resolved = resolve_microphone_array(
            array_profile=array_profile,
            custom_geometry=custom_geometry,
            reference_channel=reference_channel,
        )
    if resolved is None:
        channels = 1 if num_channels is None else num_channels
        return {
            "array_profile": None,
            "reference_channel": validate_reference_channel(reference_channel, channels),
            "num_channels": channels,
            "positions_m": None,
            "coordinate_system": _COORDINATE_SYSTEM,
            "provenance": normalize_geometry_provenance(external),
            "validation_recipe": normalize_validation_recipe(validation_recipe),
        }
    if num_channels is not None and num_channels != resolved.channels:
        raise ValueError(
            f"array geometry has C={resolved.channels}, but num_channels={num_channels}"
        )
    metadata = resolved.to_dict()
    metadata["positions_m"] = [list(position) for position in resolved.positions_m]
    metadata["num_channels"] = resolved.channels
    metadata["provenance"] = normalize_geometry_provenance(external)
    metadata["validation_recipe"] = normalize_validation_recipe(validation_recipe)
    return metadata


__all__ = [
    "ARRAY_PROFILES",
    "CANONICAL_ARRAY_PROFILES",
    "ArrayTopology",
    "MicrophoneArrayProfile",
    "build_geometry_metadata",
    "ResolvedMicrophoneArray",
    "get_array_profile",
    "normalize_geometry_provenance",
    "normalize_validation_recipe",
    "validate_reference_channel",
    "resolve_microphone_array",
    "resolve_scene_microphone_array",
    "resolve_microphone_positions",
    "resolve_geometry_positions",
]
