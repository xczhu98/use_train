"""RIR 生成计划的纯逻辑。"""

from __future__ import annotations

import argparse
import hashlib
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tools.prepare_data_tools.generators.se.rir_config import BackendOptions, RIRConfig


DEFAULT_RT60_RANGE_SEC = (0.1, 1.0)
DEFAULT_TOTAL_RIRS = 100000


def derive_scene_seed(
    base_seed: int,
    backend: str,
    sample_rate_hz: int,
    field: str,
    mode: str,
    index: int,
) -> int:
    """按任务域派生稳定场景种子，不受 GPU 或 worker 分配影响。"""
    payload = "\x00".join(
        (
            str(int(base_seed)),
            backend,
            str(int(sample_rate_hz)),
            field,
            mode,
            str(int(index)),
        )
    ).encode("utf-8")
    digest = hashlib.blake2b(
        payload,
        digest_size=8,
        person=b"rir-scene-v1",
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def split_count(total: int, ratio: tuple[float, float]) -> tuple[int, int]:
    """按最大余数法确定性切分数量，并保证总数不变。"""
    if total < 0:
        raise ValueError("total must be >= 0")
    ratio_sum = sum(ratio)
    if ratio_sum <= 0 or not all(
        math.isfinite(value) and value >= 0 for value in ratio
    ):
        raise ValueError(
            "ratio must contain finite non-negative values with a positive sum"
        )
    quotas = [total * value / ratio_sum for value in ratio]
    counts = [math.floor(quota) for quota in quotas]
    remainder = total - sum(counts)
    order = sorted(
        range(len(counts)),
        key=lambda index: (quotas[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts[0], counts[1]


def normalize_counts(args: argparse.Namespace) -> bool:
    """按 field 和比例规范化数量；返回是否显式指定了 near/far 数量。"""
    explicit_field_counts = args.n_rirs_near is not None or args.n_rirs_far is not None
    if explicit_field_counts:
        near_rirs = args.n_rirs_near if args.n_rirs_near is not None else 0
        far_rirs = args.n_rirs_far if args.n_rirs_far is not None else 0
    else:
        total_rirs = args.n_rirs_total if args.n_rirs_total is not None else DEFAULT_TOTAL_RIRS
        if args.field == "both":
            near_rirs, far_rirs = split_count(total_rirs, args.near_far_ratio)
        elif args.field == "near":
            near_rirs, far_rirs = total_rirs, 0
        else:
            near_rirs, far_rirs = 0, total_rirs
    args.n_rirs_near = near_rirs if args.field in ("near", "both") else 0
    args.n_rirs_far = far_rirs if args.field in ("far", "both") else 0
    args.n_rirs_total = args.n_rirs_near + args.n_rirs_far
    return explicit_field_counts


def normalize_segment_range(args: argparse.Namespace) -> tuple[int, int]:
    """规范化固定段数和段数范围，只允许提供一种形式。"""
    if args.segments is not None:
        if args.segments < 1:
            raise ValueError("--n-segments must be >= 1")
        segment_range = (args.segments, args.segments)
    elif args.segments_range is not None:
        segment_range = tuple(args.segments_range)
    else:
        segment_range = (3, 6)
    args.segments_range = segment_range
    args.segments = segment_range[1]
    return segment_range


def recommended_rir_length_samples(
    sample_rate_hz: int,
    rt60_range_sec: tuple[float, float],
    distance_range_m: tuple[float, float],
    sound_speed_mps: float,
) -> int:
    """计算容纳 RT60、直达传播和安全余量的 256 对齐长度。"""
    required = sample_rate_hz * (
        rt60_range_sec[1] + distance_range_m[1] / sound_speed_mps + 0.05
    )
    return max(256, math.ceil(required / 256.0) * 256)


def effective_mode(cfg: RIRConfig, mode: str | None) -> str:
    """解析 static/moving 模式。"""
    if mode is None:
        mode = "moving" if cfg.moving else "static"
    if mode not in {"static", "moving"}:
        raise ValueError(f"Unknown mode: {mode}")
    return mode


def build_field_config(
    args: argparse.Namespace,
    field: str,
    sample_rate_hz: int,
    mode: str | None = None,
) -> RIRConfig:
    """根据 field、采样率和模式构造单任务配置。"""
    if mode is None:
        mode = "moving" if getattr(args, "moving", False) else "static"
    if mode not in {"static", "moving"}:
        raise ValueError(f"Unknown mode: {mode}")
    rt60_range = getattr(args, "rt60_range", None) or DEFAULT_RT60_RANGE_SEC
    if field == "near":
        distance_range = args.dist_range_near
        explicit_samples = args.n_samples_near
    elif field == "far":
        distance_range = args.dist_range_far
        explicit_samples = args.n_samples_far
    else:
        raise ValueError(f"Unknown field: {field}")
    rir_length_samples = explicit_samples
    if rir_length_samples is None:
        rir_length_samples = recommended_rir_length_samples(
            sample_rate_hz,
            rt60_range,
            distance_range,
            args.c,
        )
    segment_range = getattr(args, "segments_range", None)
    if segment_range is None:
        fixed_segments = getattr(args, "segments", None)
        segment_range = (
            (fixed_segments, fixed_segments)
            if fixed_segments is not None
            else (3, 6)
        )
    return RIRConfig(
        fs=sample_rate_hz,
        c=args.c,
        rt60_range=rt60_range,
        room_length_range=args.room_length_range,
        room_width_range=args.room_width_range,
        room_height_range=args.room_height_range,
        source_wall_margin_m=getattr(args, "source_wall_margin_m", 0.5),
        microphone_wall_margin_m=getattr(args, "microphone_wall_margin_m", 0.1),
        source_source_min_separation_m=getattr(args, "source_source_min_separation_m", 0.3),
        source_height_range_m=getattr(args, "source_height_range_m", None),
        microphone_height_range_m=getattr(args, "microphone_height_range_m", None),
        dist_range=distance_range,
        n_samples=rir_length_samples,
        seed=args.seed,
        backend_options=getattr(args, "backend_options", BackendOptions()),
        moving=mode == "moving",
        n_segments_range=tuple(segment_range),
        segment_sec=args.segment_sec,
        speed_range=args.speed_range,
        traj_type=args.traj_type,
        array_profile=getattr(args, "array_profile", None),
        custom_geometry=getattr(args, "custom_geometry", None),
        reference_channel=getattr(args, "reference_channel", 0),
        channel_order=getattr(args, "channel_order", None),
        actor_roles=getattr(args, "actor_roles", ()),
        early_rir_sec=getattr(args, "early_rir_sec", 0.05),
    )


def iter_tasks(
    backend: str,
    field: str,
    n_rirs: int,
    cfg: RIRConfig,
    output_dir: Path,
    output_format: str,
    write_metadata_json: bool,
    shards: int,
    start_index: int = 0,
    *,
    ordinal_start: int = 0,
    ordinal_step: int = 1,
    mode: str | None = None,
) -> Iterator[dict[str, Any]]:
    """惰性构造生成任务，避免保留完整任务列表。"""
    mode = effective_mode(cfg, mode)
    if ordinal_step < 1:
        raise ValueError("ordinal_step must be >= 1")
    cfg_dict = {
        "fs": cfg.fs,
        "c": cfg.c,
        "rt60_range": cfg.rt60_range,
        "room_length_range": cfg.room_length_range,
        "room_width_range": cfg.room_width_range,
        "room_height_range": cfg.room_height_range,
        "source_wall_margin_m": cfg.source_wall_margin_m,
        "microphone_wall_margin_m": cfg.microphone_wall_margin_m,
        "source_source_min_separation_m": cfg.source_source_min_separation_m,
        "source_height_range_m": cfg.source_height_range_m,
        "microphone_height_range_m": cfg.microphone_height_range_m,
        "dist_range": cfg.dist_range,
        "n_samples": cfg.n_samples,
        "seed": cfg.seed,
        "scene_seed": cfg.scene_seed,
        "moving": mode == "moving",
        "n_segments_range": cfg.n_segments_range,
        "segment_sec": cfg.segment_sec,
        "speed_range": cfg.speed_range,
        "traj_type": cfg.traj_type,
        "array_profile": cfg.array_profile,
        "custom_geometry": cfg.custom_geometry,
        "reference_channel": cfg.reference_channel,
        "channel_order": cfg.channel_order,
        "actor_roles": cfg.actor_roles,
        "early_rir_sec": cfg.early_rir_sec,
        "backend_options": cfg.backend_options,
    }
    run_mode = f"{field}_moving" if mode == "moving" else field
    for ordinal in range(ordinal_start, n_rirs, ordinal_step):
        index = start_index + ordinal
        yield {
            "backend": backend,
            "field": field,
            "run_mode": run_mode,
            "mode": mode,
            "idx": index,
            "cfg": {
                **cfg_dict,
                "scene_seed": derive_scene_seed(
                    cfg.seed,
                    backend,
                    cfg.fs,
                    run_mode,
                    mode,
                    index,
                ),
            },
            "out_dir": str(output_dir),
            "format": output_format,
            "write_meta_json": write_metadata_json,
            "shards": shards,
        }


def iter_gpu_tasks(
    field: str,
    n_rirs: int,
    cfg: RIRConfig,
    output_dir: Path,
    output_format: str,
    write_metadata_json: bool,
    gpu_ids: list[int | str],
    gpu_slot: int,
    shards: int,
    start_index: int = 0,
    *,
    mode: str | None = None,
) -> Iterator[dict[str, Any]]:
    """生成指定 GPU slot 的 round-robin 任务。"""
    mode = effective_mode(cfg, mode)
    if not 0 <= gpu_slot < len(gpu_ids):
        raise ValueError("gpu_slot is out of range")
    for task in iter_tasks(
        "gpurir",
        field,
        n_rirs,
        cfg,
        output_dir,
        output_format,
        write_metadata_json,
        shards,
        start_index,
        ordinal_start=gpu_slot,
        ordinal_step=len(gpu_ids),
        mode=mode,
    ):
        task["gpu_id"] = gpu_ids[gpu_slot]
        yield task


__all__ = [
    "build_field_config",
    "derive_scene_seed",
    "effective_mode",
    "iter_gpu_tasks",
    "iter_tasks",
    "normalize_counts",
    "normalize_segment_range",
    "recommended_rir_length_samples",
    "split_count",
]
