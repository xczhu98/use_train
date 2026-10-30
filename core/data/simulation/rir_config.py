"""core 在线 RIR 配置；公开 YAML 字段保持 Layer C 的平面前缀风格。"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from numbers import Real

from core.data.simulation.array_profiles import (
    normalize_geometry_provenance,
    normalize_validation_recipe,
    resolve_microphone_array,
)


class OnlineRIRFields:
    # 顺序固定为 [length, width, height]；每个元素都是 [lower, upper]。
    gpurir_room_dim_range_m: tuple = ((3.0, 8.0), (3.0, 8.0), (2.5, 3.5))
    gpurir_rt60_range_sec: tuple = (0.1, 1.0)
    gpurir_source_distance_range_m: tuple = (0.2, 3.0)
    # 仅用于 enrollment 在线 RIR 生成的声源距离（注册音是近场录制，物理上 5-20cm）；
    # 通过 OnlineRIRProvider.sample_scene(distance_range=...) 覆盖，其余几何参数
    #（room/rt60/mic 高度/separation）与 mixture 共用上方 gpurir_* 字段。
    gpurir_enrollment_distance_range_m: tuple = (0.05, 0.2)
    gpurir_source_height_range_m: tuple = (1.1, 1.8)
    gpurir_mic_height_range_m: tuple = (1.0, 1.8)
    gpurir_source_min_separation_m: float = 0.3
    gpurir_wall_margin_m: float = 0.1
    gpurir_sound_speed_mps: float = 343.0
    gpurir_diffuse_transition_db: float = 15.0
    gpurir_max_sources: int = 16
    gpurir_timeout_sec: float = 60.0
    gpurir_queue_size: int = 8
    gpurir_cache_size: int = 8
    gpurir_device: int | None = None
ONLINE_RANGE_FIELDS = frozenset(
    key
    for key in OnlineRIRFields.__annotations__
    if '_range_' in key and key != 'gpurir_room_dim_range_m'
)

@dataclass(frozen=True)
class OnlineRIRConfig:
    room_ranges: tuple
    rt60_range: tuple
    distance_range: tuple
    source_height: tuple
    mic_height: tuple
    min_separation: float
    margin: float
    sound_speed: float
    diffuse_db: float
    max_sources: int
    timeout: float
    queue_size: int
    cache_size: int
    device: int | None
    microphone_positions: tuple[tuple[float, float, float], ...] | None = None
    array_profile: str | None = None
    geometry_provenance: dict | None = None
    validation_recipe: dict | None = None
    reference_channel: int = 0
    prob_array_rotation: float = 0.0
    # 内部统一使用 rad；配置侧 SimulationConfigs.array_rotation_z_deg_range 在
    # from_simulation 边界完成 degree → rad 转换。
    array_rotation_z_rad_range: tuple = (0.0, 2 * math.pi)
    prob_microphone_position_jitter: float = 0.0
    microphone_position_jitter_m_range: tuple = (0.0, 0.005)
    moving_source_speed_range_mps: tuple = (0.12, 0.4)

    @classmethod
    def from_simulation(cls, cfg):
        room_ranges = getattr(cfg, 'gpurir_room_dim_range_m')
        if (
            not isinstance(room_ranges, (list, tuple))
            or len(room_ranges) != 3
            or any(
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or any(
                    isinstance(endpoint, bool)
                    or not isinstance(endpoint, Real)
                    or not math.isfinite(endpoint)
                    or endpoint <= 0
                    for endpoint in value
                )
                or value[0] > value[1]
                for value in room_ranges
            )
        ):
            raise ValueError(
                'gpurir_room_dim_range_m must be '
                '[[length_lo, length_hi], [width_lo, width_hi], '
                '[height_lo, height_hi]]'
            )
        for key in ONLINE_RANGE_FIELDS:
            value = getattr(cfg, key)
            if not isinstance(value, (list, tuple)) or len(value) != 2 or any((isinstance(v, bool) or not isinstance(v, Real) or (not math.isfinite(v)) or (v <= 0) for v in value)) or (value[0] > value[1]):
                raise ValueError(f'{key} must be an ordered positive finite range')
        for key in ('gpurir_source_min_separation_m', 'gpurir_wall_margin_m', 'gpurir_sound_speed_mps', 'gpurir_diffuse_transition_db', 'gpurir_timeout_sec'):
            value = getattr(cfg, key)
            if isinstance(value, bool) or not isinstance(value, Real) or (not math.isfinite(value)) or (value <= 0):
                raise ValueError(f'{key} must be positive and finite')
        for key in ('gpurir_max_sources', 'gpurir_queue_size', 'gpurir_cache_size'):
            value = getattr(cfg, key)
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive exact int')
        if cfg.gpurir_max_sources > 16 or cfg.gpurir_rt60_range_sec[1] > 2.0:
            raise ValueError('online RIR budget supports at most 16 sources and RT60 <= 2 seconds')
        if cfg.gpurir_diffuse_transition_db >= 60:
            raise ValueError('gpurir_diffuse_transition_db must be below 60 dB')
        if cfg.gpurir_device is not None and (type(cfg.gpurir_device) is not int or cfg.gpurir_device < 0):
            raise ValueError('gpurir_device must be a non-negative visible-device index or null')
        room = tuple(tuple(value) for value in room_ranges)
        margin = cfg.gpurir_wall_margin_m
        if min((r[0] for r in room)) <= 2 * margin:
            raise ValueError('room must be larger than both wall margins')
        for key in ('gpurir_source_height_range_m', 'gpurir_mic_height_range_m'):
            height = getattr(cfg, key)
            if height[0] < margin or height[1] > room[2][0] - margin:
                raise ValueError(f'{key} must fit the smallest configured room')
        custom_geometry = getattr(cfg, "custom_geometry", None)
        resolved = resolve_microphone_array(
            array_profile=getattr(cfg, "array_profile", None),
            custom_geometry=custom_geometry,
            reference_channel=getattr(cfg, "reference_channel", 0),
        )
        geometry_provenance = normalize_geometry_provenance(
            getattr(cfg, "geometry_provenance", None)
        )
        validation_recipe = normalize_validation_recipe(
            getattr(cfg, "validation_recipe", None)
        )
        new_probabilities = (
            "prob_array_rotation",
            "prob_microphone_position_jitter",
        )
        for key in new_probabilities:
            value = getattr(cfg, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{key} must be a finite probability in [0, 1]")
        for key, allow_zero in (
            # 配置侧单位是 degree；这里只做结构校验，数值在下方统一转为 rad。
            ("array_rotation_z_deg_range", True),
            ("microphone_position_jitter_m_range", True),
            ("moving_source_speed_range_mps", False),
        ):
            value = getattr(cfg, key)
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, Real)
                    or not math.isfinite(float(item))
                    for item in value
                )
                or value[0] > value[1]
            ):
                raise ValueError(f"{key} must be an ordered finite range")
            if value[0] < 0.0 or (not allow_zero and value[0] == 0.0):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{key} lower bound must be {qualifier}")
        reference_channel = getattr(cfg, "reference_channel", 0)
        if reference_channel == "auto":
            reference_channel = 0
        return cls(
            room,
            tuple(cfg.gpurir_rt60_range_sec),
            tuple(cfg.gpurir_source_distance_range_m),
            tuple(cfg.gpurir_source_height_range_m),
            tuple(cfg.gpurir_mic_height_range_m),
            cfg.gpurir_source_min_separation_m,
            margin,
            cfg.gpurir_sound_speed_mps,
            cfg.gpurir_diffuse_transition_db,
            cfg.gpurir_max_sources,
            cfg.gpurir_timeout_sec,
            cfg.gpurir_queue_size,
            cfg.gpurir_cache_size,
            cfg.gpurir_device,
            None if resolved is None else resolved.positions_m,
            getattr(cfg, "array_profile", None),
            geometry_provenance,
            validation_recipe,
            reference_channel,
            float(cfg.prob_array_rotation),
            # 配置侧单位 degree → rad；默认 [0, 360] 即整圈。
            tuple(math.radians(v) for v in cfg.array_rotation_z_deg_range),
            float(cfg.prob_microphone_position_jitter),
            tuple(cfg.microphone_position_jitter_m_range),
            tuple(cfg.moving_source_speed_range_mps),
        )

    def to_dict(self):
        return asdict(self)
