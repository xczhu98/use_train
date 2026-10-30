"""纯 CPU 几何规划：一个房间、一个麦克风、多个不同声源位置。"""
from __future__ import annotations
import hashlib
import json
import math
from numbers import Real

import numpy as np
from core.data.simulation.rir_config import OnlineRIRConfig
from core.data.simulation.array_profiles import resolve_microphone_array


def sample_rir_scene(
    cfg: OnlineRIRConfig,
    sample_rate: int,
    source_count: int,
    rng,
    *,
    distance_range=None,
    microphone_array=None,
    use_config_microphone_array=True,
):
    if distance_range is None:
        distance_range = cfg.distance_range
    else:
        distance_range = tuple(distance_range)
        if (len(distance_range) != 2 or any((isinstance(v, bool) or not isinstance(v, Real) or (not math.isfinite(v)) or (v <= 0) for v in distance_range)) or (distance_range[0] > distance_range[1])):
            raise ValueError('distance_range override must be an ordered positive finite range')
    if type(source_count) is not int or not 1 <= source_count <= cfg.max_sources:
        raise ValueError('online source_count is outside the configured budget')
    if type(sample_rate) is not int or not 1000 <= sample_rate <= 192000:
        raise ValueError('invalid online RIR sample rate')
    configured_profile = getattr(cfg, "array_profile", None) if use_config_microphone_array else None
    configured_geometry = getattr(cfg, "custom_geometry", None) if use_config_microphone_array else None
    if configured_geometry is None and use_config_microphone_array:
        configured_geometry = getattr(cfg, "microphone_positions", None)
    if configured_profile is not None and microphone_array is None:
        configured_geometry = None
    if microphone_array is not None:
        configured_profile = None
        configured_geometry = microphone_array
    resolved = resolve_microphone_array(
        array_profile=configured_profile,
        custom_geometry=configured_geometry,
        reference_channel=getattr(cfg, "reference_channel", 0),
    )
    offsets = resolved.positions_m if resolved is not None else None
    offset_array = np.asarray(offsets, dtype=np.float64) if offsets is not None else np.zeros((1, 3), dtype=np.float64)
    if offset_array.ndim != 2 or offset_array.shape[1] != 3:
        raise ValueError("microphone array must have shape [C, 3]")
    array_augmentation = {
        "rotation_z_rad": None,
        "microphone_jitter_max_m": 0.0,
    }
    if offsets is None and (
        cfg.prob_array_rotation > 0.0 or cfg.prob_microphone_position_jitter > 0.0
    ):
        raise ValueError("array augmentation requires a configured microphone array")
    if offsets is not None and cfg.prob_array_rotation > 0.0 and rng.random() < cfg.prob_array_rotation:
        rotation_z = float(rng.uniform(*cfg.array_rotation_z_rad_range))
        cosine, sine = math.cos(rotation_z), math.sin(rotation_z)
        rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        offset_array = offset_array @ rotation.T
        array_augmentation["rotation_z_rad"] = rotation_z
    if (
        offsets is not None
        and cfg.prob_microphone_position_jitter > 0.0
        and rng.random() < cfg.prob_microphone_position_jitter
    ):
        jitter_max = float(rng.uniform(*cfg.microphone_position_jitter_m_range))
        if jitter_max > 0.0:
            offset_array += rng.uniform(
                -jitter_max, jitter_max, size=offset_array.shape
            )
            array_augmentation["microphone_jitter_max_m"] = jitter_max
    for _ in range(100):
        room = np.array([rng.uniform(*r) for r in cfg.room_ranges], dtype=np.float64)
        surface = 2 * (room[0] * room[1] + room[0] * room[2] + room[1] * room[2])
        rt60 = float(rng.uniform(*cfg.rt60_range))
        if 24 * math.log(10) * float(np.prod(room)) / (cfg.sound_speed * surface * rt60) >= 1:
            continue
        lower = np.full(3, cfg.margin, dtype=np.float64) - offset_array.min(axis=0)
        upper = room - cfg.margin - offset_array.max(axis=0)
        lower[2] = max(lower[2], cfg.mic_height[0] - offset_array[:, 2].min())
        upper[2] = min(upper[2], cfg.mic_height[1] - offset_array[:, 2].max())
        if np.any(lower > upper):
            continue
        center = np.array([rng.uniform(lower[i], upper[i]) for i in range(3)])
        mic_positions = center[None, :] + offset_array
        positions = []
        for _source in range(source_count):
            for _attempt in range(500):
                direction = rng.normal(size=3)
                direction /= max(np.linalg.norm(direction), 1e-12)
                position = center + direction * rng.uniform(*distance_range)
                if np.all(position > cfg.margin) and np.all(position < room - cfg.margin) and (cfg.source_height[0] <= position[2] <= cfg.source_height[1]) and all((np.linalg.norm(position - other) >= cfg.min_separation for other in positions)):
                    positions.append(position)
                    break
            else:
                break
        if len(positions) == source_count:
            break
    else:
        raise ValueError('cannot place requested sources with the configured room/distance constraints')
    distances = [float(np.linalg.norm(p - center)) for p in positions]
    mic_loc = mic_positions.tolist() if offsets is not None else center.tolist()
    coordinate_system = (
        "cartesian_xyz_m" if resolved is None else resolved.profile.topology.coordinate_system
    )
    spacing_m = (
        None
        if resolved is None or array_augmentation["microphone_jitter_max_m"] > 0.0
        else resolved.profile.topology.spacing_m
    )
    array_topology = None if resolved is None else {
        "positions_m": offset_array.tolist(),
        "spacing_m": spacing_m,
        "coordinate_system": coordinate_system,
        "channels": int(offset_array.shape[0]),
    }
    plan = {
        'version': 3 if offsets is not None else 1,
        'backend': 'online',
        'sample_rate': sample_rate,
        'room_dim': room.tolist(),
        'mic_loc': mic_loc,
        'mic_center': center.tolist(),
        'microphone_offsets': offset_array.tolist() if offsets is not None else None,
        'num_channels': int(offset_array.shape[0]),
        'array_profile': getattr(cfg, 'array_profile', None),
        'array_topology': array_topology,
        'array_augmentation': array_augmentation,
        'coordinate_system': coordinate_system,
        'spacing_m': spacing_m,
        'reference_channel': int(getattr(cfg, 'reference_channel', 0)),
        'geometry_provenance': getattr(cfg, 'geometry_provenance', None),
        'validation_recipe': getattr(cfg, 'validation_recipe', None),
        'source_loc': [p.tolist() for p in positions],
        'dist': distances,
        'rt60': rt60,
        'sound_speed': cfg.sound_speed,
        'diffuse_db': cfg.diffuse_db,
        'num_samples': math.ceil((rt60 + (max(distances) + float(np.max(np.linalg.norm(offset_array, axis=1)))) / cfg.sound_speed + 0.05) * sample_rate),
        'speaker_pattern': 'omni',
        'microphone_pattern': 'omni',
        'source_ids': [f'source/{i}' for i in range(source_count)],
    }
    payload = json.dumps(plan, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    plan['group_uid'] = 'online-' + hashlib.sha256(payload).hexdigest()[:24]
    return plan
