"""RIR manifest summary、inventory 与事务发布。"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
from pathlib import Path
from typing import Any

from core.utils.file_transaction import write_text_files_transactionally
from tools.prepare_data_tools.generators.se.rir_config import BackendOptions
from tools.prepare_data_tools.generators.se.rir_output import sha256_file
from tools.shared.scp_io import serialize_manifest_relative_path


logger = logging.getLogger("generate_simulated_rirs")


def entry_dimensions(
    entry: tuple[str, int, int, str],
    backends: list[str],
    output_dir: Path,
) -> tuple[str, str, str]:
    """从输出目录或 UID 推导 backend、mode 和 field。"""
    uid, _, _, path = entry
    try:
        relative_parts = Path(path).relative_to(output_dir).parts
        if len(relative_parts) >= 4 and relative_parts[2] in {"static", "moving"}:
            return relative_parts[0], relative_parts[2], relative_parts[3]
    except ValueError:
        pass
    for backend in backends:
        prefix = f"{backend}_fs"
        if not uid.startswith(prefix):
            continue
        sample_rate, separator, remainder = uid[len(prefix) :].partition("_")
        if not separator or not sample_rate.isdigit():
            continue
        for field in ("near", "far"):
            if remainder.startswith(f"{field}_moving_"):
                return backend, "moving", field
            if remainder.startswith(f"{field}_"):
                return backend, "static", field
    return "unknown", "unknown", "unknown"


def ratio_counts(counts: dict[str, int], first: str, second: str) -> dict[str, float]:
    """计算两个计数维度的归一化比例。"""
    denominator = counts[first] + counts[second]
    if denominator == 0:
        return {first: 0.0, second: 0.0}
    return {
        first: counts[first] / denominator,
        second: counts[second] / denominator,
    }


def build_summary(
    output_dir: Path,
    args: argparse.Namespace,
    backends: list[str],
    fields: list[str],
    gpu_ids: list[int | str],
    unique_entries: list[tuple[str, int, int, str]],
    planned_specs: list[dict[str, Any]],
    failed: int,
    *,
    generator_id: str,
    generator_version: str,
    generator_version_history: tuple[str, ...],
) -> dict[str, Any]:
    """构建绑定 generator、有效配置和输出 inventory 的 summary。"""
    parameter_names = {
        "method": "backend",
        "fs": "sample_rates_hz",
        "n_rirs_total": "total_rirs",
        "n_rirs_near": "near_rirs",
        "n_rirs_far": "far_rirs",
        "rt60_range": "rt60_range_sec",
        "dist_range_near": "near_distance_range_m",
        "dist_range_far": "far_distance_range_m",
        "room_length_range": "room_length_range_m",
        "room_width_range": "room_width_range_m",
        "room_height_range": "room_height_range_m",
        "n_samples_near": "near_rir_length_samples",
        "n_samples_far": "far_rir_length_samples",
        "c": "sound_speed_mps",
        "segments": "n_segments",
        "segment_sec": "segment_duration_sec",
        "speed_range": "speed_range_mps",
        "traj_type": "trajectory_type",
        "moving": "motion_enabled",
        "write_meta_json": "write_metadata_json",
        "format": "output_format",
        "abnormal_threshold": "validation_peak_threshold",
        "inventory": "compute_sha256",
    }
    excluded = {
        "out_dir",
        "validate",
        "rir_config_path",
        "effective_config",
        "dump_effective_config",
    }
    normalized_params = {
        parameter_names.get(key, key): (
            value.to_dict() if isinstance(value, BackendOptions) else value
        )
        for key, value in sorted(vars(args).items())
        if key not in excluded
    }
    effective_config = getattr(args, "effective_config", None)
    dimensions = {
        entry: entry_dimensions(entry, backends, output_dir)
        for entry in unique_entries
    }

    compute_sha256 = bool(getattr(args, "inventory", False))
    if not compute_sha256:
        logger.info("Skipping SHA-256 fingerprints (--inventory to enable)")
    output_inventory = []
    for entry in unique_entries:
        uid, sample_rate_hz, frames, path = entry
        backend, mode, field = dimensions[entry]
        output_inventory.append(
            {
                "uid": uid,
                "backend": backend,
                "mode": mode,
                "field": field,
                "sample_rate_hz": sample_rate_hz,
                "n_samples": frames,
                "path": path,
                "sha256": sha256_file(Path(path)) if compute_sha256 else None,
            }
        )

    requested_counts = {"total": 0, "near": 0, "far": 0, "static": 0, "moving": 0}
    requested_by_backend: dict[str, int] = {}
    requested_by_backend_mode_field: dict[str, dict[str, dict[str, int]]] = {}
    for spec in planned_specs:
        count = int(spec["requested"])
        backend = str(spec["backend"])
        mode = str(spec["mode"])
        field = str(spec["field"])
        requested_counts["total"] += count
        requested_counts[field] += count
        requested_counts[mode] += count
        requested_by_backend[backend] = requested_by_backend.get(backend, 0) + count
        by_mode = requested_by_backend_mode_field.setdefault(backend, {})
        by_field = by_mode.setdefault(mode, {})
        by_field[field] = by_field.get(field, 0) + count

    valid_counts = {"total": len(unique_entries), "near": 0, "far": 0, "static": 0, "moving": 0}
    valid_by_backend: dict[str, int] = {}
    valid_by_mode: dict[str, int] = {"static": 0, "moving": 0}
    valid_by_field: dict[str, int] = {"near": 0, "far": 0}
    valid_by_backend_mode_field: dict[str, dict[str, dict[str, int]]] = {}
    for entry in unique_entries:
        backend, mode, field = dimensions[entry]
        if mode in valid_by_mode:
            valid_counts[mode] += 1
            valid_by_mode[mode] += 1
        if field in valid_by_field:
            valid_counts[field] += 1
            valid_by_field[field] += 1
        valid_by_backend[backend] = valid_by_backend.get(backend, 0) + 1
        by_mode = valid_by_backend_mode_field.setdefault(backend, {})
        by_field = by_mode.setdefault(mode, {})
        by_field[field] = by_field.get(field, 0) + 1

    requested_ratios = {
        "near_far": ratio_counts(requested_counts, "near", "far"),
        "static_moving": ratio_counts(requested_counts, "static", "moving"),
    }
    valid_ratios = {
        "near_far": ratio_counts(valid_counts, "near", "far"),
        "static_moving": ratio_counts(valid_counts, "static", "moving"),
    }
    effective_formats = {
        "static": args.format if args.format != "auto" else "pkl",
        "moving": "pkl",
    }
    summary: dict[str, Any] = {
        "generator_id": generator_id,
        "generator_version": generator_version,
        "generator_version_history": list(generator_version_history),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "parameters": normalized_params,
        "config": {
            "source": str(getattr(args, "rir_config_path", "")) or None,
            "effective": effective_config,
        },
        "output_inventory": output_inventory,
        "sample_rates_hz": args.fs,
        "backend": args.method,
        "field": args.field,
        "output_format": args.format,
        "effective_formats": effective_formats,
        "layout_version": 3,
        "rt60_range_sec": list(args.rt60_range),
        "shared_rt60_range_sec": list(args.rt60_range),
        "ratio_weights": {
            "near_far": list(args.near_far_ratio),
            "static_moving": list(args.static_moving_ratio),
        },
        "effective_gpu_ids": list(gpu_ids) if "gpurir" in backends else [],
        "total_rirs": len(unique_entries),
        "failed": failed,
        "compute_sha256": compute_sha256,
        "counts": {
            "requested": requested_counts,
            "valid": valid_counts,
            "failed": failed,
            "requested_by_backend": requested_by_backend,
            "requested_by_backend_mode_field": requested_by_backend_mode_field,
        },
        "requested_counts": requested_counts,
        "effective_counts": requested_counts,
        "valid_counts": valid_counts,
        "ratios": {
            "requested": requested_ratios,
            "valid": valid_ratios,
            "near_far": valid_ratios["near_far"],
            "static_moving": valid_ratios["static_moving"],
        },
        "statistics": {
            "by_backend": valid_by_backend,
            "by_mode": valid_by_mode,
            "by_field": valid_by_field,
            "by_backend_mode_field": valid_by_backend_mode_field,
        },
        "backends": {},
    }
    for backend in backends:
        for field in fields:
            key = f"{backend}_{field}_moving" if args.moving else f"{backend}_{field}"
            summary["backends"][key] = sum(
                1
                for entry in unique_entries
                if dimensions[entry][0] == backend and dimensions[entry][2] == field
            )
    for backend, modes in valid_by_backend_mode_field.items():
        for mode, field_counts in modes.items():
            for field, count in field_counts.items():
                summary["backends"][f"{backend}_{mode}_{field}"] = count
    return summary


def publish_manifest(
    output_dir: Path,
    unique_entries: list[tuple[str, int, int, str]],
    summary: dict[str, Any],
) -> None:
    """事务写入混合 SCP、逐采样率 SCP 和 summary。"""
    scp_path = output_dir / "rirs.scp"

    def scp_lines(entries: list[tuple[str, int, int, str]]) -> str:
        payload = "\n".join(
            f"{shlex.quote(uid)} {sample_rate_hz} {frames} {shlex.quote(relative_path)}"
            for uid, sample_rate_hz, frames, path in entries
            for relative_path in [serialize_manifest_relative_path(path, scp_path)]
        )
        return payload + ("\n" if payload else "")

    fs_groups: dict[int, list[tuple[str, int, int, str]]] = {}
    for entry in unique_entries:
        fs_groups.setdefault(entry[1], []).append(entry)
    files_to_write = [(scp_path, lambda handle: handle.write(scp_lines(unique_entries)))]
    for sample_rate_hz in sorted(fs_groups):
        fs_scp_path = output_dir / f"rirs_fs{sample_rate_hz}.scp"
        files_to_write.append(
            (
                fs_scp_path,
                lambda handle, entries=fs_groups[sample_rate_hz]: handle.write(
                    scp_lines(entries)
                ),
            )
        )
    summary_path = output_dir / "rirs_meta.json"
    summary_payload = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    files_to_write.append((summary_path, lambda handle: handle.write(summary_payload)))
    write_text_files_transactionally(files_to_write, overwrite=True)
    logger.info("Wrote %s with %d entries", scp_path, len(unique_entries))
    for sample_rate_hz in sorted(fs_groups):
        logger.info(
            "Wrote rirs_fs%s.scp with %d entries",
            sample_rate_hz,
            len(fs_groups[sample_rate_hz]),
        )
    logger.info("Wrote summary: %s", summary_path)


__all__ = ["build_summary", "entry_dimensions", "publish_manifest", "ratio_counts"]
