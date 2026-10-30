"""RIR artifact 写入与 manifest record 辅助。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from tools.shared.pickle_io import save_pkl_file


def sha256_file(path: Path) -> str:
    """返回文件内容的 SHA-256 身份。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_uid(
    method: str,
    fs: int,
    field: str,
    rt60: float,
    dist: float,
    room_dim: list[float],
    idx: int,
) -> str:
    """构造包含关键元信息的 uid / 文件名。"""
    return (
        f"{method}_fs{fs}_{field}_rt{rt60:.2f}_dist{dist:.2f}_"
        f"room{room_dim[0]:.1f}x{room_dim[1]:.1f}x{room_dim[2]:.1f}_{idx:08d}"
    )


def save_result(
    result: dict[str, Any],
    task: dict[str, Any],
) -> tuple[str, int, int, str] | None:
    """保存单条 RIR，返回四列 SCP record。"""
    out_dir = Path(task["out_dir"])
    method = task["backend"]
    field = str(task["field"])
    requested_mode = task.get("mode")
    if result.get("moving", False):
        mode = "moving"
    elif requested_mode in {"static", "moving"}:
        mode = requested_mode
    else:
        mode = "static"
    moving = mode == "moving"
    fmt = task["format"]
    if fmt == "auto":
        fmt = "pkl"
    idx = task["idx"]
    write_meta = task.get("write_meta_json", False)
    shards = max(1, int(task.get("shards", 1)))

    rir = np.asarray(result["rir"], dtype=np.float32)
    if rir.ndim not in {1, 2} or rir.size == 0 or rir.shape[-1] == 0:
        raise ValueError(f"RIR must be a non-empty 1D or 2D array, got shape {rir.shape}")
    if not np.isfinite(rir).all():
        raise ValueError("RIR contains non-finite samples")
    fs = int(result["fs"])
    n_samples = rir.shape[-1]

    uid = make_uid(
        method=method,
        fs=fs,
        field=f"{field}_moving" if moving else field,
        rt60=float(result["rt60"]),
        dist=float(result["dist"]),
        room_dim=[float(x) for x in result["room_dim"]],
        idx=idx,
    )

    shard_id = idx % shards
    out_path = (
        out_dir
        / method
        / f"fs{fs}"
        / mode
        / field
        / f"shard_{shard_id:04d}"
        / f"{uid}.{fmt}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt in {"wav", "flac"}:
        if moving or rir.ndim > 1:
            raise ValueError(
                f"Moving RIR ({rir.shape}) cannot be saved as {fmt}; use --format pkl"
            )
        if fmt == "wav":
            sf.write(str(out_path), rir, samplerate=fs, subtype="FLOAT")
        else:
            peak = float(np.max(np.abs(rir)))
            if peak > 1.0:
                raise ValueError(
                    "FLAC integer PCM cannot preserve RIR samples outside [-1, 1] "
                    f"without scaling (peak={peak:.6g}); use --format pkl or wav"
                )
            sf.write(str(out_path), rir, samplerate=fs, format="FLAC", subtype="PCM_24")
    elif fmt == "pkl":
        payload = {
            "rir": rir,
            "fs": fs,
            "rt60": float(result["rt60"]),
            "dist": float(result["dist"]),
            "room_dim": [float(x) for x in result["room_dim"]],
            "source_loc": [float(x) for x in result["source_loc"]],
            "mic_loc": [float(x) for x in result["mic_loc"]],
            "method": method,
            "mode": mode,
            "field": field,
            "idx": idx,
            "uid": uid,
        }
        if moving:
            payload["moving"] = True
            payload["n_segments"] = int(result.get("n_segments", rir.shape[0]))
            payload["segment_sec"] = float(result.get("segment_sec", 0.5))
            payload["trajectory"] = np.asarray(result["trajectory"], dtype=np.float32)
            payload["speed"] = float(result.get("speed", 0.0))
            payload["dist_min"] = float(result.get("dist_min", float(result["dist"])))
            payload["dist_max"] = float(result.get("dist_max", float(result["dist"])))
        save_pkl_file(out_path, payload)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    if write_meta:
        meta_path = out_path.with_suffix(".json")
        meta = {
            "uid": uid,
            "method": method,
            "mode": mode,
            "field": field,
            "fs": fs,
            "n_samples": n_samples,
            "n_segments": rir.shape[0] if rir.ndim > 1 else 1,
            "moving": moving,
            "rt60": float(result["rt60"]),
            "dist": float(result["dist"]),
            "room_dim": [float(x) for x in result["room_dim"]],
            "source_loc": [float(x) for x in result["source_loc"]],
            "mic_loc": [float(x) for x in result["mic_loc"]],
        }
        if moving:
            meta["dist_min"] = float(result.get("dist_min", float(result["dist"])))
            meta["dist_max"] = float(result.get("dist_max", float(result["dist"])))
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return (uid, fs, n_samples, str(out_path.resolve()))


def save_scene_result(result: dict[str, Any], *, output_dir: Path, index: int, shards: int = 1) -> dict[str, Any]:
    """保存多通道 scene bundle，并返回可写入 scenes.jsonl 的 typed record。"""
    scene = result["scene"]
    scene_uid = str(scene["scene_uid"])
    actors = list(scene["actors"])
    full = np.asarray(result["full"], dtype="<f4")
    direct = np.asarray(result["direct"], dtype="<f4")
    early = np.asarray(result["early"], dtype="<f4")
    expected = (len(actors), len(scene["array"]["positions_m"]), int(scene["rir_num_samples"]))
    for name, value in (("full", full), ("direct", direct), ("early", early)):
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f"scene {name} must have finite shape {expected}, got {value.shape}")
    scene_root = output_dir / "scene_bundles" / f"shard_{index % max(1, int(shards)):04d}"
    scene_root.mkdir(parents=True, exist_ok=True)
    bundle_path = scene_root / f"{scene_uid}.npz"
    np.savez_compressed(bundle_path, full=full, direct=direct, early=early)
    manifest_dir = output_dir / "scene_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    actor_entries: dict[str, dict[str, str]] = {}
    identities: dict[str, dict[str, str]] = {}
    for actor_index, actor in enumerate(actors):
        actor_uid = actor["actor_uid"]
        actor_entries[actor_uid] = {}
        identities[actor_uid] = {}
        for track, bank in (("full", full), ("direct", direct), ("early", early)):
            digest = hashlib.sha256(np.ascontiguousarray(bank[actor_index], dtype="<f4").tobytes()).hexdigest()
            identities[actor_uid][track] = digest
        actor_entries[actor_uid] = {
            # scene manifest 内路径相对 manifest 所在目录，兼容 loader 的路径解析。
            "bundle_path": str(Path("..").joinpath(bundle_path.relative_to(output_dir))),
            "actor_index": actor_index,
            "full": "full",
            "direct": "direct",
            "early": "early",
        }
    metadata = {
        "schema_version": 2,
        "scene_uid": scene_uid,
        "sample_rate": int(scene["sample_rate"]),
        "actor_order": [actor["actor_uid"] for actor in actors],
        "actor_roles": [actor["role"] for actor in actors],
        "actor_positions_m": {actor["actor_uid"]: actor["position_m"] for actor in actors},
        "room": dict(scene["room"]),
        "geometry": dict(scene["array"]),
        "generator_version": scene["generator_version"],
        "track_identities": identities,
        "track_shapes": {track: list(expected) for track in ("full", "direct", "early")},
    }
    record = {"entries": actor_entries, "metadata": metadata}
    manifest_path = manifest_dir / f"{scene_uid}.json"
    manifest_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    record["scene_manifest_path"] = str(manifest_path.relative_to(output_dir))
    return record


def save_moving_scene_result(result: dict[str, Any], *, output_dir: Path, index: int, shards: int = 1) -> dict[str, Any]:
    """保存 moving scene bundle（``[P,A,C,L]``）并返回 schema v3 typed record。"""
    from core.data.simulation.scene_manifest import trajectory_digest

    scene = result["scene"]
    scene_uid = str(scene["scene_uid"])
    actors = list(scene["actors"])
    full = np.asarray(result["full"], dtype="<f4")
    direct = np.asarray(result["direct"], dtype="<f4")
    early = np.asarray(result["early"], dtype="<f4")
    expected = (
        int(scene["trajectory_timebase"]["frame_count"]),
        len(actors),
        len(scene["array"]["positions_m"]),
        int(scene["rir_num_samples"]),
    )
    for name, value in (("full", full), ("direct", direct), ("early", early)):
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f"moving scene {name} must have finite shape {expected}, got {value.shape}")
    scene_root = output_dir / "scene_bundles" / f"shard_{index % max(1, int(shards)):04d}"
    scene_root.mkdir(parents=True, exist_ok=True)
    bundle_path = scene_root / f"{scene_uid}.npz"
    np.savez_compressed(bundle_path, full=full, direct=direct, early=early)
    manifest_dir = output_dir / "scene_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    actor_entries: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, str]] = {}
    for actor_index, actor in enumerate(actors):
        actor_uid = actor["actor_uid"]
        identities[actor_uid] = {}
        for track, bank in (("full", full), ("direct", direct), ("early", early)):
            digest = hashlib.sha256(
                np.ascontiguousarray(bank[:, actor_index], dtype="<f4").tobytes()
            ).hexdigest()
            identities[actor_uid][track] = digest
        actor_entries[actor_uid] = {
            "bundle_path": str(Path("..").joinpath(bundle_path.relative_to(output_dir))),
            "actor_index": actor_index,
            "full": "full",
            "direct": "direct",
            "early": "early",
        }
    actor_trajectories = {actor["actor_uid"]: actor["trajectory"] for actor in actors}
    trajectory_timebase = dict(scene["trajectory_timebase"])
    metadata = {
        "schema_version": 3,
        "moving": True,
        "scene_uid": scene_uid,
        "sample_rate": int(scene["sample_rate"]),
        "actor_order": [actor["actor_uid"] for actor in actors],
        "actor_roles": [actor["role"] for actor in actors],
        "actor_positions_m": {actor["actor_uid"]: actor["position_m"] for actor in actors},
        "actor_trajectories": actor_trajectories,
        "trajectory_timebase": trajectory_timebase,
        "trajectory_sha256": trajectory_digest(actor_trajectories, trajectory_timebase),
        "room": dict(scene["room"]),
        "geometry": dict(scene["array"]),
        "generator_version": scene["generator_version"],
        "track_identities": identities,
        "track_shapes": {track: list(expected) for track in ("full", "direct", "early")},
    }
    record = {"entries": actor_entries, "metadata": metadata}
    manifest_path = manifest_dir / f"{scene_uid}.json"
    manifest_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    record["scene_manifest_path"] = str(manifest_path.relative_to(output_dir))
    return record


__all__ = ["make_uid", "save_result", "save_scene_result", "save_moving_scene_result", "sha256_file"]
