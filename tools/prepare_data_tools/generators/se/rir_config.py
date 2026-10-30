"""RIR 生成器的 typed 配置与 runtime 投影。

配置文件与生成代码放在同一目录。配置字段使用 ``snake_case``，带单位的
数值在字段名中显式标注单位；CLI 仅负责选择配置文件和覆盖输出目录。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from core.utils.config import load_merged_config


MIN_SOURCE_MICROPHONE_DISTANCE = 0.05


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _allowed(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {path} key(s): {', '.join(unknown)}")


def _pair(value: Any, path: str, *, integer: bool = False) -> tuple[Any, Any]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{path} must contain exactly two values")
    if integer:
        if any(type(item) is not int for item in value):
            raise ValueError(f"{path} must contain integer values")
        result = tuple(value)
    else:
        if any(isinstance(item, bool) for item in value):
            raise ValueError(f"{path} must contain numeric values, not booleans")
        try:
            result = tuple(float(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} must contain numeric values") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{path} must contain finite values")
    return result


def _positive_pair(value: Any, path: str) -> tuple[float, float]:
    result = _pair(value, path)
    if not result[0] <= result[1]:
        raise ValueError(f"{path} must satisfy lo <= hi")
    return result


def _ratio(value: Any, path: str) -> tuple[float, float]:
    result = _pair(value, path)
    if result[0] < 0.0 or result[1] < 0.0:
        raise ValueError(f"{path} must be non-negative")
    if sum(result) <= 0.0:
        raise ValueError(f"{path} must have a positive sum")
    return result


def _optional_pair(value: Any, path: str) -> tuple[float, float] | None:
    if value is None:
        return None
    return _positive_pair(value, path)


def _strict_bool(value: Any, path: str, default: bool) -> bool:
    value = default if value is None else value
    if type(value) is not bool:
        raise ValueError(f"{path} must be a boolean")
    return value


def _strict_int(value: Any, path: str, default: int) -> int:
    value = default if value is None else value
    if type(value) is not int:
        raise ValueError(f"{path} must be an integer")
    return value


def _integer_range(value: Any, path: str, default: tuple[int, int]) -> tuple[int, int]:
    result = default if value is None else _pair(value, path, integer=True)
    if result[0] < 1 or result[0] > result[1]:
        raise ValueError(f"{path} must satisfy 1 <= lo <= hi")
    return int(result[0]), int(result[1])


def _orientation(value: Any, path: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{path} must contain three numeric values")
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must contain numeric values") from exc
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{path} must contain finite values")
    norm = math.sqrt(sum(item * item for item in vector))
    if norm <= 0.0:
        raise ValueError(f"{path} must not be the zero vector")
    return tuple(item / norm for item in vector)


@dataclass(frozen=True)
class FRAMBackendConfig:
    """FRAM-RIR 实现参数。"""

    n_image_range: tuple[int, int] = (512, 2049)
    reflection_perturbation_range: tuple[float, float] = (-2.0, 2.0)
    reflection_tau: float = 0.25
    direct_window_range_ms: tuple[float, float] = (-6.0, 50.0)
    highpass_cutoff_hz: float = 80.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], path: str) -> "FRAMBackendConfig":
        _allowed(
            raw,
            {
                "n_image_range",
                "reflection_perturbation_range",
                "reflection_tau",
                "direct_window_range_ms",
                "highpass_cutoff_hz",
            },
            path,
        )
        n_image_range = _pair(raw.get("n_image_range", [512, 2049]), f"{path}.n_image_range", integer=True)
        if n_image_range[0] < 1 or n_image_range[0] >= n_image_range[1]:
            raise ValueError(f"{path}.n_image_range must satisfy 1 <= lo < hi")
        reflection_tau = float(raw.get("reflection_tau", 0.25))
        highpass_cutoff_hz = float(raw.get("highpass_cutoff_hz", 80.0))
        if not math.isfinite(reflection_tau) or reflection_tau < 0.0:
            raise ValueError(f"{path}.reflection_tau must be finite and >= 0")
        if not math.isfinite(highpass_cutoff_hz) or highpass_cutoff_hz < 0.0:
            raise ValueError(f"{path}.highpass_cutoff_hz must be finite and >= 0")
        return cls(
            n_image_range=(int(n_image_range[0]), int(n_image_range[1])),
            reflection_perturbation_range=_positive_pair(
                raw.get("reflection_perturbation_range", [-2.0, 2.0]),
                f"{path}.reflection_perturbation_range",
            ),
            reflection_tau=reflection_tau,
            direct_window_range_ms=_positive_pair(
                raw.get("direct_window_range_ms", [-6.0, 50.0]),
                f"{path}.direct_window_range_ms",
            ),
            highpass_cutoff_hz=highpass_cutoff_hz,
        )


@dataclass(frozen=True)
class GPURIRBackendConfig:
    """gpuRIR 实现参数。"""

    diffuse_transition_db: float = 15.0
    speaker_pattern: str = "omni"
    microphone_pattern: str = "omni"
    speaker_orientation: tuple[float, float, float] | None = None
    microphone_orientation: tuple[float, float, float] | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], path: str) -> "GPURIRBackendConfig":
        _allowed(
            raw,
            {
                "diffuse_transition_db",
                "speaker_pattern",
                "microphone_pattern",
                "speaker_orientation",
                "microphone_orientation",
            },
            path,
        )
        transition_db = float(raw.get("diffuse_transition_db", 15.0))
        if not math.isfinite(transition_db) or not 0.0 < transition_db <= 60.0:
            raise ValueError(f"{path}.diffuse_transition_db must satisfy 0 < value <= 60")
        patterns = {"omni", "homni", "card", "hypcard", "subcard", "bidir"}
        speaker_pattern = str(raw.get("speaker_pattern", "omni"))
        microphone_pattern = str(raw.get("microphone_pattern", "omni"))
        if speaker_pattern not in patterns or microphone_pattern not in patterns:
            raise ValueError(f"{path} contains an unsupported polar pattern")
        speaker_orientation = _orientation(
            raw.get("speaker_orientation"), f"{path}.speaker_orientation"
        )
        microphone_orientation = _orientation(
            raw.get("microphone_orientation"), f"{path}.microphone_orientation"
        )
        if speaker_pattern != "omni" and speaker_orientation is None:
            raise ValueError(
                f"{path}.speaker_orientation is required for {speaker_pattern}"
            )
        if microphone_pattern != "omni" and microphone_orientation is None:
            raise ValueError(
                f"{path}.microphone_orientation is required for {microphone_pattern}"
            )
        return cls(
            transition_db,
            speaker_pattern,
            microphone_pattern,
            speaker_orientation,
            microphone_orientation,
        )


@dataclass(frozen=True)
class RIRGeneratorBackendConfig:
    """rir_generator 实现参数。"""

    reflection_order: int = -1
    dimension: int = 3
    highpass_filter: bool = True

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], path: str
    ) -> "RIRGeneratorBackendConfig":
        _allowed(raw, {"reflection_order", "dimension", "highpass_filter"}, path)
        reflection_order = _strict_int(raw.get("reflection_order"), f"{path}.reflection_order", -1)
        dimension = _strict_int(raw.get("dimension"), f"{path}.dimension", 3)
        highpass_filter = raw.get("highpass_filter", True)
        if reflection_order < -1:
            raise ValueError(f"{path}.reflection_order must be >= -1")
        if dimension not in {2, 3}:
            raise ValueError(f"{path}.dimension must be 2 or 3")
        if not isinstance(highpass_filter, bool):
            raise ValueError(f"{path}.highpass_filter must be a boolean")
        return cls(reflection_order, dimension, highpass_filter)


@dataclass(frozen=True)
class BackendOptions:
    """全部 backend 的 typed 参数集合。"""

    fram: FRAMBackendConfig = field(default_factory=FRAMBackendConfig)
    gpurir: GPURIRBackendConfig = field(default_factory=GPURIRBackendConfig)
    rirgen: RIRGeneratorBackendConfig = field(default_factory=RIRGeneratorBackendConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], path: str) -> "BackendOptions":
        _allowed(raw, {"fram", "gpurir", "rirgen"}, path)
        return cls(
            fram=FRAMBackendConfig.from_mapping(
                _mapping(raw.get("fram", {}), f"{path}.fram"), f"{path}.fram"
            ),
            gpurir=GPURIRBackendConfig.from_mapping(
                _mapping(raw.get("gpurir", {}), f"{path}.gpurir"), f"{path}.gpurir"
            ),
            rirgen=RIRGeneratorBackendConfig.from_mapping(
                _mapping(raw.get("rirgen", {}), f"{path}.rirgen"), f"{path}.rirgen"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """返回可写入 JSON/YAML 的普通 mapping。"""
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class RIRConfig:
    """单个 field/mode/backend 任务使用的 immutable 配置。"""

    fs: int
    c: float
    rt60_range: tuple[float, float]
    room_length_range: tuple[float, float]
    room_width_range: tuple[float, float]
    room_height_range: tuple[float, float]
    dist_range: tuple[float, float]
    n_samples: int
    seed: int
    scene_seed: int | None = None
    moving: bool = False
    n_segments_range: tuple[int, int] = (3, 6)
    segment_sec: float = 0.5
    speed_range: tuple[float, float] = (0.5, 1.5)
    traj_type: str = "both"
    backend_options: BackendOptions = field(default_factory=BackendOptions)
    source_wall_margin_m: float = 0.5
    microphone_wall_margin_m: float = 0.1
    source_source_min_separation_m: float = 0.3
    source_height_range_m: tuple[float, float] | None = None
    microphone_height_range_m: tuple[float, float] | None = None
    # 配置阵列后才启用 scene-level spatial 生成；C 由 geometry 推导。
    array_profile: str | None = None
    custom_geometry: dict[str, Any] | None = None
    reference_channel: int = 0
    channel_order: tuple[str, ...] | None = None
    actor_roles: tuple[str, ...] = ()
    early_rir_sec: float = 0.05


@dataclass(frozen=True)
class GeneratorConfig:
    """完整 RIR generator 配置；负责 YAML schema 和跨字段校验。"""

    schema_version: int
    profile: str
    backend: str
    field: str
    output_dir: Path | None
    sample_rates_hz: tuple[int, ...]
    total_rirs: int | None
    near_rirs: int | None
    far_rirs: int | None
    near_far_ratio: tuple[float, float]
    static_moving_ratio: tuple[float, float]
    rt60_range_sec: tuple[float, float]
    near_distance_range_m: tuple[float, float]
    far_distance_range_m: tuple[float, float]
    allow_distance_overlap: bool
    room_length_range_m: tuple[float, float]
    room_width_range_m: tuple[float, float]
    room_height_range_m: tuple[float, float]
    min_source_microphone_distance_m: float
    source_wall_margin_m: float
    microphone_wall_margin_m: float
    source_source_min_separation_m: float
    source_height_range_m: tuple[float, float] | None
    microphone_height_range_m: tuple[float, float] | None
    near_rir_length_samples: int | None
    far_rir_length_samples: int | None
    sound_speed_mps: float
    motion_enabled: bool
    n_segments_range: tuple[int, int]
    segment_duration_sec: float
    speed_range_mps: tuple[float, float]
    trajectory_type: str
    workers: int
    shards: int
    gpus: tuple[int | str, ...] | None
    output_format: str
    write_metadata_json: bool
    compute_sha256: bool
    validation_enabled: bool
    validation_peak_threshold: float
    seed: int
    array_profile: str | None
    custom_geometry: dict[str, Any] | None
    reference_channel: int
    channel_order: tuple[str, ...] | None
    actor_roles: tuple[str, ...]
    early_rir_sec: float
    backend_options: BackendOptions

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, source: str = "config") -> "GeneratorConfig":
        raw = _mapping(raw, source)
        _allowed(
            raw,
            {
                "schema_version", "profile", "backend", "field", "output_dir",
                "sample_rates_hz", "dataset", "scene", "acoustics", "motion",
                "execution", "output", "validation", "seed",
            },
            source,
        )
        schema_version = raw.get("schema_version", 1)
        if schema_version != 1:
            raise ValueError(f"{source}.schema_version must be 1")

        backend_raw = raw.get("backend", "gpurir")
        if isinstance(backend_raw, str):
            backend = backend_raw
            backend_options = BackendOptions()
        else:
            backend_map = _mapping(backend_raw, f"{source}.backend")
            _allowed(backend_map, {"name", "gpurir", "fram", "rirgen"}, f"{source}.backend")
            backend = str(backend_map.get("name", "gpurir"))
            backend_params = {
                name: dict(_mapping(value, f"{source}.backend.{name}"))
                for name, value in backend_map.items()
                if name != "name"
            }
            backend_options = BackendOptions.from_mapping(
                backend_params, f"{source}.backend"
            )
        if backend not in {"rirgen", "fram", "gpurir", "all"}:
            raise ValueError(f"{source}.backend has unsupported value {backend!r}")

        dataset = _mapping(raw.get("dataset", {}), f"{source}.dataset")
        _allowed(
            dataset,
            {"total_rirs", "near_rirs", "far_rirs", "near_far_ratio", "static_moving_ratio"},
            f"{source}.dataset",
        )
        total_rirs = dataset.get("total_rirs")
        near_rirs = dataset.get("near_rirs")
        far_rirs = dataset.get("far_rirs")
        for name, value in (("total_rirs", total_rirs), ("near_rirs", near_rirs), ("far_rirs", far_rirs)):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{source}.dataset.{name} must be a non-negative integer or null")
        if total_rirs is not None:
            total_rirs = int(total_rirs)
        if near_rirs is not None:
            near_rirs = int(near_rirs)
        if far_rirs is not None:
            far_rirs = int(far_rirs)

        scene = _mapping(raw.get("scene", {}), f"{source}.scene")
        _allowed(scene, {"room", "geometry", "distance", "actors"}, f"{source}.scene")
        room = _mapping(scene.get("room", {}), f"{source}.scene.room")
        _allowed(room, {"length_range_m", "width_range_m", "height_range_m"}, f"{source}.scene.room")
        geometry = _mapping(scene.get("geometry", {}), f"{source}.scene.geometry")
        _allowed(
            geometry,
            {
                "min_source_microphone_distance_m",
                "source_wall_margin_m",
                "microphone_wall_margin_m",
                "source_source_min_separation_m",
                "source_height_range_m",
                "microphone_height_range_m",
                "array_profile", "custom_geometry", "reference_channel", "channel_order",
            },
            f"{source}.scene.geometry",
        )
        distance = _mapping(scene.get("distance", {}), f"{source}.scene.distance")
        _allowed(
            distance,
            {"near_range_m", "far_range_m", "allow_overlap"},
            f"{source}.scene.distance",
        )
        actors = _mapping(scene.get("actors", {}), f"{source}.scene.actors")
        _allowed(actors, {"count", "roles"}, f"{source}.scene.actors")
        raw_roles = actors.get("roles")
        if raw_roles is None:
            actor_count = int(actors.get("count", 1))
            if actor_count < 1:
                raise ValueError(f"{source}.scene.actors.count must be positive")
            actor_roles = tuple(f"source_{i}" for i in range(actor_count))
        else:
            if not isinstance(raw_roles, (list, tuple)) or not raw_roles:
                raise ValueError(f"{source}.scene.actors.roles must be a non-empty list")
            actor_roles = tuple(str(role).strip() for role in raw_roles)
            if any(not role for role in actor_roles) or len(set(actor_roles)) != len(actor_roles):
                raise ValueError(f"{source}.scene.actors.roles must be unique non-empty strings")

        acoustics = _mapping(raw.get("acoustics", {}), f"{source}.acoustics")
        _allowed(
            acoustics,
            {"rt60_range_sec", "sound_speed_mps", "near_rir_length_samples", "far_rir_length_samples", "early_rir_sec"},
            f"{source}.acoustics",
        )
        motion = _mapping(raw.get("motion", {}), f"{source}.motion")
        _allowed(
            motion,
            {"enabled", "n_segments_range", "segment_duration_sec", "speed_range_mps", "trajectory_type"},
            f"{source}.motion",
        )
        execution = _mapping(raw.get("execution", {}), f"{source}.execution")
        _allowed(execution, {"workers", "shards", "gpus"}, f"{source}.execution")
        output = _mapping(raw.get("output", {}), f"{source}.output")
        _allowed(output, {"format", "write_metadata_json", "compute_sha256"}, f"{source}.output")
        validation = _mapping(raw.get("validation", {}), f"{source}.validation")
        _allowed(validation, {"enabled", "peak_threshold"}, f"{source}.validation")

        raw_sample_rates = raw.get("sample_rates_hz", [16000])
        if not isinstance(raw_sample_rates, (list, tuple)) or any(
            type(value) is not int for value in raw_sample_rates
        ):
            raise ValueError(f"{source}.sample_rates_hz must contain integer values")
        sample_rates = tuple(raw_sample_rates)
        if not sample_rates or any(value <= 0 for value in sample_rates) or len(set(sample_rates)) != len(sample_rates):
            raise ValueError(f"{source}.sample_rates_hz must contain unique positive integers")
        min_distance = float(
            geometry.get("min_source_microphone_distance_m", MIN_SOURCE_MICROPHONE_DISTANCE)
        )
        if not math.isfinite(min_distance) or min_distance <= 0.0:
            raise ValueError(f"{source}.scene.geometry.min_source_microphone_distance_m must be > 0")

        config = cls(
            schema_version=1,
            profile=str(raw.get("profile", "custom")),
            backend=backend,
            field=str(raw.get("field", "both")),
            output_dir=Path(raw["output_dir"]).expanduser() if raw.get("output_dir") else None,
            sample_rates_hz=sample_rates,
            total_rirs=total_rirs,
            near_rirs=near_rirs,
            far_rirs=far_rirs,
            near_far_ratio=_ratio(dataset.get("near_far_ratio", [5.0, 5.0]), f"{source}.dataset.near_far_ratio"),
            static_moving_ratio=_ratio(dataset.get("static_moving_ratio", [1.0, 0.0]), f"{source}.dataset.static_moving_ratio"),
            rt60_range_sec=_positive_pair(acoustics.get("rt60_range_sec", [0.1, 1.0]), f"{source}.acoustics.rt60_range_sec"),
            near_distance_range_m=_positive_pair(distance.get("near_range_m", [min_distance, 0.5]), f"{source}.scene.distance.near_range_m"),
            far_distance_range_m=_positive_pair(distance.get("far_range_m", [0.8, 3.0]), f"{source}.scene.distance.far_range_m"),
            allow_distance_overlap=_strict_bool(
                distance.get("allow_overlap"),
                f"{source}.scene.distance.allow_overlap",
                False,
            ),
            room_length_range_m=_positive_pair(room.get("length_range_m", [3.0, 10.0]), f"{source}.scene.room.length_range_m"),
            room_width_range_m=_positive_pair(room.get("width_range_m", [3.0, 10.0]), f"{source}.scene.room.width_range_m"),
            room_height_range_m=_positive_pair(room.get("height_range_m", [2.5, 4.5]), f"{source}.scene.room.height_range_m"),
            min_source_microphone_distance_m=min_distance,
            source_wall_margin_m=float(geometry.get("source_wall_margin_m", 0.5)),
            microphone_wall_margin_m=float(geometry.get("microphone_wall_margin_m", 0.1)),
            source_source_min_separation_m=float(geometry.get("source_source_min_separation_m", 0.3)),
            source_height_range_m=_optional_pair(
                geometry.get("source_height_range_m"),
                f"{source}.scene.geometry.source_height_range_m",
            ),
            microphone_height_range_m=_optional_pair(
                geometry.get("microphone_height_range_m"),
                f"{source}.scene.geometry.microphone_height_range_m",
            ),
            array_profile=(str(geometry["array_profile"]) if geometry.get("array_profile") is not None else None),
            custom_geometry=(dict(geometry["custom_geometry"]) if geometry.get("custom_geometry") is not None else None),
            reference_channel=int(geometry.get("reference_channel", 0)),
            channel_order=(tuple(str(item) for item in geometry["channel_order"]) if geometry.get("channel_order") is not None else None),
            actor_roles=actor_roles,
            early_rir_sec=float(acoustics.get("early_rir_sec", 0.05)),
            near_rir_length_samples=_optional_positive_int(acoustics.get("near_rir_length_samples"), f"{source}.acoustics.near_rir_length_samples"),
            far_rir_length_samples=_optional_positive_int(acoustics.get("far_rir_length_samples"), f"{source}.acoustics.far_rir_length_samples"),
            sound_speed_mps=float(acoustics.get("sound_speed_mps", 340.0)),
            motion_enabled=_strict_bool(
                motion.get("enabled"), f"{source}.motion.enabled", False
            ),
            n_segments_range=_integer_range(
                motion.get("n_segments_range"),
                f"{source}.motion.n_segments_range",
                (3, 6),
            ),
            segment_duration_sec=float(motion.get("segment_duration_sec", 0.5)),
            speed_range_mps=_positive_pair(motion.get("speed_range_mps", [0.5, 1.5]), f"{source}.motion.speed_range_mps"),
            trajectory_type=str(motion.get("trajectory_type", "both")),
            workers=_strict_int(execution.get("workers"), f"{source}.execution.workers", 16),
            shards=_strict_int(execution.get("shards"), f"{source}.execution.shards", 10),
            gpus=_parse_gpus(execution.get("gpus"), f"{source}.execution.gpus"),
            output_format=str(output.get("format", "pkl")),
            write_metadata_json=_strict_bool(
                output.get("write_metadata_json"), f"{source}.output.write_metadata_json", False
            ),
            compute_sha256=_strict_bool(
                output.get("compute_sha256"), f"{source}.output.compute_sha256", False
            ),
            validation_enabled=_strict_bool(
                validation.get("enabled"), f"{source}.validation.enabled", True
            ),
            validation_peak_threshold=float(validation.get("peak_threshold", 1e-6)),
            seed=int(raw.get("seed", 42)),
            backend_options=backend_options,
        )
        config.validate(source)
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GeneratorConfig":
        path = Path(path).expanduser().resolve()
        raw = load_merged_config(path)
        return cls.from_mapping(raw, source=str(path))

    def validate(self, source: str = "config") -> None:
        if self.field not in {"near", "far", "both"}:
            raise ValueError(f"{source}.field must be near, far, or both")
        if self.sound_speed_mps <= 0.0 or not math.isfinite(self.sound_speed_mps):
            raise ValueError(f"{source}.acoustics.sound_speed_mps must be finite and > 0")
        if self.workers <= 0 or self.shards <= 0:
            raise ValueError(f"{source}.execution workers and shards must be > 0")
        if self.output_format not in {"auto", "wav", "pkl", "flac"}:
            raise ValueError(f"{source}.output.format is unsupported: {self.output_format!r}")
        if self.array_profile is not None and self.custom_geometry is not None:
            raise ValueError(f"{source}.scene.geometry array_profile and custom_geometry are mutually exclusive")
        if not isinstance(self.reference_channel, int) or self.reference_channel < 0:
            raise ValueError(f"{source}.scene.geometry.reference_channel must be non-negative")
        if not math.isfinite(self.early_rir_sec) or self.early_rir_sec < 0:
            raise ValueError(f"{source}.acoustics.early_rir_sec must be finite and >= 0")
        if not math.isfinite(self.source_source_min_separation_m) or self.source_source_min_separation_m <= 0:
            raise ValueError(f"{source}.scene.geometry.source_source_min_separation_m must be > 0")
        if self.room_length_range_m[0] < 1.0 or self.room_width_range_m[0] < 1.0:
            raise ValueError(f"{source}.scene.room horizontal dimensions must be >= 1 m")
        if self.room_height_range_m[0] < 1.0:
            raise ValueError(f"{source}.scene.room.height_range_m must be >= 1 m")
        for name, distance_range in (
            ("near_range_m", self.near_distance_range_m),
            ("far_range_m", self.far_distance_range_m),
        ):
            if distance_range[0] < self.min_source_microphone_distance_m:
                raise ValueError(
                    f"{source}.scene.distance.{name} lower bound must be >= "
                    "scene.geometry.min_source_microphone_distance_m"
                )
        if not self.allow_distance_overlap:
            near_lo, near_hi = self.near_distance_range_m
            far_lo, far_hi = self.far_distance_range_m
            if near_lo < far_hi and far_lo < near_hi:
                raise ValueError(
                    f"{source}.scene.distance near/far ranges overlap; "
                    "set allow_overlap=true to permit overlap"
                )
        for name, margin in (
            ("source_wall_margin_m", self.source_wall_margin_m),
            ("microphone_wall_margin_m", self.microphone_wall_margin_m),
        ):
            if not math.isfinite(margin) or margin < 0.0:
                raise ValueError(f"{source}.scene.geometry.{name} must be finite and >= 0")
        if not math.isfinite(self.segment_duration_sec) or not all(
            math.isfinite(value) for value in self.speed_range_mps
        ):
            raise ValueError(f"{source}.motion values must be finite")
        if self.motion_enabled and (
            self.n_segments_range[0] < 2
            or self.segment_duration_sec <= 0.0
            or not (0.0 < self.speed_range_mps[0] < self.speed_range_mps[1])
        ):
            raise ValueError(
                f"{source}.motion requires n_segments_range >= 2, positive segment duration, "
                "and 0 < speed lo < speed hi"
            )
        if self.trajectory_type not in {"linear", "waypoint", "both"}:
            raise ValueError(f"{source}.motion.trajectory_type is unsupported: {self.trajectory_type!r}")
        if not math.isfinite(self.validation_peak_threshold) or self.validation_peak_threshold < 0.0:
            raise ValueError(f"{source}.validation.peak_threshold must be finite and >= 0")
        for name, height_range in (
            ("source_height_range_m", self.source_height_range_m),
            ("microphone_height_range_m", self.microphone_height_range_m),
        ):
            if height_range is not None and height_range[0] <= 0.0:
                raise ValueError(f"{source}.scene.geometry.{name} must be strictly positive")

    def to_runtime_namespace(self, output_dir: Path | None = None) -> dict[str, Any]:
        """将 canonical 配置投影为内部执行计划所需的扁平字段。"""
        static_moving_ratio = (
            self.static_moving_ratio
            if self.motion_enabled
            else (1.0, 0.0)
        )
        return {
            "method": self.backend,
            "field": self.field,
            "out_dir": output_dir if output_dir is not None else self.output_dir,
            "fs": list(self.sample_rates_hz),
            "n_rirs_total": self.total_rirs,
            "n_rirs_near": self.near_rirs,
            "n_rirs_far": self.far_rirs,
            "near_far_ratio": self.near_far_ratio,
            "static_moving_ratio": static_moving_ratio,
            "rt60_range": self.rt60_range_sec,
            "dist_range_near": self.near_distance_range_m,
            "dist_range_far": self.far_distance_range_m,
            "allow_distance_overlap": self.allow_distance_overlap,
            "room_length_range": self.room_length_range_m,
            "room_width_range": self.room_width_range_m,
            "room_height_range": self.room_height_range_m,
            "n_samples_near": self.near_rir_length_samples,
            "n_samples_far": self.far_rir_length_samples,
            "c": self.sound_speed_mps,
            "workers": self.workers,
            "shards": self.shards,
            "gpus": list(self.gpus) if self.gpus is not None else None,
            "format": self.output_format,
            "write_meta_json": self.write_metadata_json,
            "moving": False,
            "segments": None,
            "segments_range": self.n_segments_range,
            "segment_sec": self.segment_duration_sec,
            "speed_range": self.speed_range_mps,
            "traj_type": self.trajectory_type,
            "array_profile": self.array_profile,
            "custom_geometry": self.custom_geometry,
            "reference_channel": self.reference_channel,
            "channel_order": list(self.channel_order) if self.channel_order is not None else None,
            "actor_roles": list(self.actor_roles),
            "early_rir_sec": self.early_rir_sec,
            "source_wall_margin_m": self.source_wall_margin_m,
            "microphone_wall_margin_m": self.microphone_wall_margin_m,
            "source_source_min_separation_m": self.source_source_min_separation_m,
            "source_height_range_m": self.source_height_range_m,
            "microphone_height_range_m": self.microphone_height_range_m,
            "validate": self.validation_enabled,
            "inventory": self.compute_sha256,
            "abnormal_threshold": self.validation_peak_threshold,
            "seed": self.seed,
            "backend_options": self.backend_options,
        }

    def with_runtime_overrides(
        self,
        args: Any,
        output_dir: Path | None = None,
    ) -> "GeneratorConfig":
        """将 canonical CLI override 合并回 typed config。"""
        resolved = replace(
            self,
            backend=args.method,
            field=args.field,
            output_dir=output_dir if output_dir is not None else args.out_dir,
            sample_rates_hz=tuple(args.fs),
            total_rirs=args.n_rirs_total,
            near_rirs=args.n_rirs_near,
            far_rirs=args.n_rirs_far,
            near_far_ratio=tuple(args.near_far_ratio),
            static_moving_ratio=tuple(args.static_moving_ratio),
            rt60_range_sec=tuple(args.rt60_range),
            near_distance_range_m=tuple(args.dist_range_near),
            far_distance_range_m=tuple(args.dist_range_far),
            allow_distance_overlap=bool(args.allow_distance_overlap),
            room_length_range_m=tuple(args.room_length_range),
            room_width_range_m=tuple(args.room_width_range),
            room_height_range_m=tuple(args.room_height_range),
            source_wall_margin_m=float(args.source_wall_margin_m),
            microphone_wall_margin_m=float(args.microphone_wall_margin_m),
            source_height_range_m=args.source_height_range_m,
            microphone_height_range_m=args.microphone_height_range_m,
            near_rir_length_samples=args.n_samples_near,
            far_rir_length_samples=args.n_samples_far,
            sound_speed_mps=float(args.c),
            motion_enabled=bool(
                args.moving
                or (
                    args.method == "gpurir"
                    and args.static_moving_ratio[1] > 0.0
                )
            ),
            n_segments_range=tuple(args.segments_range),
            segment_duration_sec=float(args.segment_sec),
            speed_range_mps=tuple(args.speed_range),
            trajectory_type=args.traj_type,
            workers=int(args.workers),
            shards=int(args.shards),
            gpus=tuple(args.gpus) if args.gpus is not None else None,
            output_format=args.format,
            write_metadata_json=bool(args.write_meta_json),
            compute_sha256=bool(args.inventory),
            validation_enabled=bool(args.validate),
            validation_peak_threshold=float(args.abnormal_threshold),
            seed=int(args.seed),
            backend_options=args.backend_options,
        )
        resolved.validate("effective config")
        return resolved

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _optional_positive_int(value: Any, path: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{path} must be a positive integer or null")
    return int(value)


def _parse_gpus(value: Any, path: str) -> tuple[int | str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{path} must be null or a non-empty list")
    result: list[int | str] = []
    for token in value:
        if isinstance(token, bool):
            raise ValueError(f"{path} contains an invalid GPU token")
        if isinstance(token, int):
            if token < 0:
                raise ValueError(f"{path} GPU ids must be non-negative")
            result.append(token)
        else:
            token = str(token).strip()
            if not token:
                raise ValueError(f"{path} contains an empty GPU token")
            result.append(token)
    if len({str(token) for token in result}) != len(result):
        raise ValueError(f"{path} must not contain duplicate GPU tokens")
    return tuple(result)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


__all__ = [
    "BackendOptions",
    "FRAMBackendConfig",
    "GeneratorConfig",
    "GPURIRBackendConfig",
    "MIN_SOURCE_MICROPHONE_DISTANCE",
    "RIRConfig",
    "RIRGeneratorBackendConfig",
]
