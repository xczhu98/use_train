"""typed offline spatial RIR scene manifest 合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SCENE_MANIFEST_SCHEMA_VERSION = 3
_REQUIRED_IDENTITY = (
    "scene_uid", "actor_order", "room", "geometry", "generator_version", "track_identities"
)
_REQUIRED_SCENE_V2 = ("sample_rate", "actor_positions_m")
_REQUIRED_TRACKS = ("full", "direct", "early")
_REQUIRED_SCENE_MOVING = ("moving", "actor_trajectories", "trajectory_timebase", "trajectory_sha256")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trajectory_digest(actor_trajectories: Any, trajectory_timebase: Any) -> str:
    """moving scene 轨迹的 canonical 内容身份（独立于 track 文件哈希）。"""
    payload = {
        "actor_trajectories": actor_trajectories,
        "trajectory_timebase": trajectory_timebase,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _resolve_index_entry(root: Path, manifest_path: str, line_number: int) -> Path:
    """解析 scene index 行内 manifest 路径；越出索引目录一律 fail-closed。"""
    resolved = (root / Path(manifest_path).expanduser()).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"scene index line {line_number} scene_manifest_path escapes the index directory"
        )
    return resolved


def _validate_moving_metadata(metadata: Mapping[str, Any]) -> None:
    """校验 schema v3 moving block：timebase、逐 actor 轨迹与内容哈希。

    静态 manifest（无 ``moving: true``）不进入本校验；moving manifest 必须覆盖
    全部 actor、共享一个 timebase，且轨迹首帧与 ``actor_positions_m`` 一致。
    """
    import numpy as np

    from core.data.simulation.trajectory import TrajectoryTimebase

    moving_value = metadata.get("moving", False)
    if not isinstance(moving_value, bool):
        raise ValueError("offline scene manifest metadata.moving must be a bool when present")
    if moving_value is not True:
        return
    missing = [key for key in _REQUIRED_SCENE_MOVING if key not in metadata]
    if missing:
        raise ValueError(f"offline moving scene manifest missing fields: {missing!r}")
    timebase_raw = metadata["trajectory_timebase"]
    if not isinstance(timebase_raw, Mapping):
        raise ValueError("offline moving scene trajectory_timebase must be a mapping")
    clock_fields = {}
    for key in ("sample_rate", "num_samples", "hop_samples", "frame_count", "start_sample"):
        value = timebase_raw.get(key, 0 if key == "start_sample" else None)
        if value is None or type(value) is not int:
            raise ValueError(f"moving trajectory_timebase.{key} must be an exact int")
        clock_fields[key] = value
    clock = TrajectoryTimebase(**clock_fields)
    if clock.sample_rate != metadata["sample_rate"]:
        raise ValueError("moving trajectory_timebase sample_rate must match scene sample_rate")
    trajectories = metadata["actor_trajectories"]
    actor_order = metadata["actor_order"]
    positions = metadata["actor_positions_m"]
    if not isinstance(trajectories, Mapping) or set(trajectories) != set(actor_order):
        raise ValueError("moving actor_trajectories must cover exactly actor_order")
    room = metadata.get("room")
    room_dim = room.get("room_dim_m") if isinstance(room, Mapping) else None
    if room_dim is None:
        raise ValueError("moving scene manifest room must declare room_dim_m")
    room_array = np.asarray(room_dim, dtype=np.float64)
    if room_array.shape != (3,) or not np.isfinite(room_array).all() or np.any(room_array <= 0):
        raise ValueError("moving scene manifest room_dim_m must be positive finite [3]")
    for uid in actor_order:
        points = np.asarray(trajectories[uid], dtype=np.float64)
        if points.shape != (clock.frame_count, 1, 3) or not np.isfinite(points).all():
            raise ValueError(
                f"moving trajectory for {uid!r} must be finite with shape [P,1,3], got {points.shape}"
            )
        if np.any(points <= 0) or np.any(points >= room_array):
            raise ValueError(f"moving trajectory for {uid!r} leaves the scene room")
        start = np.asarray(positions[uid], dtype=np.float64)
        if not np.allclose(points[0, 0], start, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"moving trajectory for {uid!r} must start at actor_positions_m"
            )
    digest = _trajectory_digest(trajectories, timebase_raw)
    if digest != metadata["trajectory_sha256"]:
        raise ValueError("moving trajectory_sha256 does not match trajectory content")


@dataclass(frozen=True)
class OfflineSceneManifest:
    """一个 scene 级 offline RIR manifest。

    ``entries`` 保留 source actor 顺序；每个 actor 必须绑定 full/direct/early
    文件和对应 content hash。JSON/SCP 格式由底层 loader 读取，但不被这个
    typed spatial 合同接受。
    """

    path: Path
    scene_uid: str
    actor_order: tuple[str, ...]
    room: Mapping[str, Any]
    geometry: Mapping[str, Any]
    generator_version: str
    track_identities: Mapping[str, Mapping[str, str]]
    metadata: Mapping[str, Any]
    typed: bool = True

    @classmethod
    def load(cls, manifest: str | Path, *, require_typed: bool = False) -> "OfflineSceneManifest | OfflineSceneIndex | None":
        path = Path(manifest).expanduser()
        if not path.is_file():
            raise ValueError(f"offline spatial RIR manifest does not exist: {path}")
        if path.suffix.lower() == ".jsonl":
            return OfflineSceneIndex.load(path, require_typed=require_typed)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if require_typed:
                raise ValueError("typed offline spatial scene manifest must be JSON")
            return None
        if not isinstance(raw, dict) or "entries" not in raw:
            if require_typed:
                raise ValueError("offline spatial manifest must declare typed 'entries' and identity")
            return None
        metadata = raw.get("metadata", raw)
        if not isinstance(metadata, Mapping):
            raise ValueError("offline spatial scene metadata must be a mapping")
        missing = [key for key in _REQUIRED_IDENTITY if key not in metadata]
        schema_version = metadata.get("schema_version", 1)
        if schema_version >= 2:
            missing.extend(key for key in _REQUIRED_SCENE_V2 if key not in metadata)
        if missing:
            if require_typed:
                raise ValueError(f"offline spatial scene manifest missing identity fields: {missing!r}")
            return None
        scene_uid = metadata["scene_uid"]
        actor_order = metadata["actor_order"]
        room = metadata["room"]
        geometry = metadata["geometry"]
        generator = metadata["generator_version"]
        identities = metadata["track_identities"]
        if not isinstance(scene_uid, str) or not scene_uid.strip():
            raise ValueError("offline spatial scene scene_uid must be a non-empty string")
        if not isinstance(actor_order, list) or not actor_order or any(not isinstance(x, str) or not x.strip() for x in actor_order):
            raise ValueError("offline spatial scene actor_order must be a non-empty string list")
        if len(set(actor_order)) != len(actor_order):
            raise ValueError("offline spatial scene actor_order must be unique")
        if not isinstance(room, Mapping) or not room:
            raise ValueError("offline spatial scene room must be a non-empty mapping")
        if not isinstance(geometry, Mapping) or not geometry:
            raise ValueError("offline spatial scene geometry must be a non-empty mapping")
        if not isinstance(generator, str) or not generator.strip():
            raise ValueError("offline spatial scene generator_version must be non-empty")
        if schema_version >= 2 and (type(metadata["sample_rate"]) is not int or metadata["sample_rate"] <= 0):
            raise ValueError("offline spatial scene sample_rate must be a positive integer")
        if schema_version >= 2 and not isinstance(metadata["actor_positions_m"], Mapping):
            raise ValueError("offline spatial scene actor_positions_m must be a mapping")
        if schema_version >= 2 and set(metadata["actor_positions_m"]) != set(actor_order):
            raise ValueError("offline spatial scene actor_positions_m must cover actor_order")
        if "positions_m" not in geometry:
            raise ValueError("offline spatial scene geometry requires positions_m")
        if schema_version >= 2 and type(geometry.get("reference_channel")) is not int:
            raise ValueError("offline spatial scene geometry requires reference_channel")
        if schema_version >= 2:
            order = geometry.get("channel_order")
            if not isinstance(order, list) or len(order) != len(geometry["positions_m"]):
                raise ValueError("offline spatial scene geometry channel_order must match positions_m")
        _validate_moving_metadata(metadata)
        if not isinstance(identities, Mapping):
            raise ValueError("offline spatial scene track_identities must be a mapping")
        normalized: dict[str, dict[str, str]] = {}
        for actor in actor_order:
            value = identities.get(actor)
            if not isinstance(value, Mapping):
                raise ValueError(f"offline spatial scene missing track identities for actor {actor!r}")
            if any(not isinstance(value.get(track), str) or not value[track].strip() for track in _REQUIRED_TRACKS):
                raise ValueError(f"offline spatial scene actor {actor!r} must identify full/direct/early tracks")
            normalized[actor] = {track: value[track] for track in _REQUIRED_TRACKS}
        return cls(path, scene_uid, tuple(actor_order), dict(room), dict(geometry), generator, normalized, dict(metadata))

    def validate_request(self, source_keys: tuple[str, ...], geometry: Mapping[str, Any]) -> None:
        if tuple(source_keys) != self.actor_order:
            raise ValueError("offline spatial scene actor_order does not match requested source_uids")
        expected = self.geometry.get("positions_m")
        actual = geometry.get("positions_m")
        if expected is not None and actual is not None and expected != actual:
            raise ValueError("offline spatial scene geometry positions_m does not match resolved geometry")

    def track_hashes(self, source_keys: tuple[str, ...], track: str) -> dict[str, str]:
        if track not in _REQUIRED_TRACKS:
            raise ValueError(f"unsupported offline RIR track: {track}")
        return {key: self.track_identities[key][track] for key in source_keys}


@dataclass(frozen=True)
class OfflineSceneIndex:
    """split-level JSONL scene index; each line points to one typed scene JSON."""

    path: Path
    records: Mapping[str, Mapping[str, Any]]

    @classmethod
    def load(cls, path: str | Path, *, require_typed: bool = True) -> "OfflineSceneIndex":
        path = Path(path).expanduser().resolve()
        records: dict[str, Mapping[str, Any]] = {}
        # 逐行流式读取：split index 可能很大，不整文件装载。
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"scene index line {line_number} is not JSON") from exc
                metadata = record.get("metadata") if isinstance(record, Mapping) else None
                scene_uid = metadata.get("scene_uid") if isinstance(metadata, Mapping) else None
                manifest_path = record.get("scene_manifest_path") if isinstance(record, Mapping) else None
                if not isinstance(scene_uid, str) or not scene_uid.strip() or not isinstance(manifest_path, str):
                    if require_typed:
                        raise ValueError(f"scene index line {line_number} requires scene_uid and scene_manifest_path")
                    continue
                actor_order = list(metadata.get("actor_order", []))
                actor_roles = list(metadata.get("actor_roles", []))
                if actor_roles and len(actor_roles) != len(actor_order):
                    raise ValueError(
                        f"scene index line {line_number} actor_roles must match actor_order"
                    )
                records[scene_uid] = {
                    "scene_uid": scene_uid,
                    "actor_order": actor_order,
                    "actor_roles": actor_roles,
                    "sample_rate": metadata.get("sample_rate"),
                    "geometry": dict(metadata.get("geometry", {})),
                    "scene_manifest_path": str(_resolve_index_entry(path.parent, manifest_path, line_number)),
                }
        if not records and require_typed:
            raise ValueError(f"typed scene index is empty: {path}")
        return cls(path, records)

    def get(self, scene_uid: str) -> OfflineSceneManifest:
        record = self.records.get(str(scene_uid))
        if record is None:
            raise ValueError(f"offline scene index does not contain scene_uid={scene_uid!r}")
        manifest = OfflineSceneManifest.load(record["scene_manifest_path"], require_typed=True)
        if isinstance(manifest, OfflineSceneIndex) or manifest is None:
            raise ValueError("scene index entry must resolve to one typed scene manifest")
        return manifest

    def select(
        self,
        source_keys: tuple[str, ...],
        sample_index: int,
        *,
        sample_rate: int | None = None,
        geometry: Mapping[str, Any] | None = None,
        roles: tuple[str, ...] | None = None,
    ) -> OfflineSceneManifest:
        expected_positions = None if geometry is None else geometry.get("positions_m")

        def relative_positions(value):
            if value is None:
                return None
            rows = [list(row) for row in value]
            if not rows:
                return rows
            origin = rows[0]
            return [[float(item) - float(base) for item, base in zip(row, origin)] for row in rows]

        expected_relative = relative_positions(expected_positions)
        candidates = [
            uid for uid, record in self.records.items()
            if set(source_keys).issubset(set(record.get("actor_order", ())))
            and (sample_rate is None or record.get("sample_rate") == sample_rate)
            and (
                roles is None
                or tuple(
                    record.get("actor_roles", [])[record.get("actor_order", []).index(key)]
                    for key in source_keys
                ) == tuple(roles)
            )
            and (
                expected_relative is None
                or relative_positions(record.get("geometry", {}).get("positions_m")) == expected_relative
            )
        ]
        if not candidates:
            raise ValueError("offline scene index has no scene containing requested actor_uids")
        return self.get(sorted(candidates)[int(sample_index) % len(candidates)])


def _first_bundle_array(bundle: Any, path: Path) -> Any:
    files = list(bundle.files)
    if not files:
        raise ValueError(f"offline RIR bundle has no arrays: {path}")
    return bundle[files[0]]


def build_track_identity(path: str | Path) -> str:
    """计算 canonical RIR waveform bytes 的 hash，供 generator/manifest 写入。

    ``.npz`` bundle 取首个数组，空 bundle 显式报错；``.npy`` 只读取一次，
    不进入 npz 上下文管理分支。
    """
    import numpy as np

    path = Path(path).expanduser()
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as loaded:
            array = _first_bundle_array(loaded, path)
    else:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.ndarray):
            array = loaded
        else:
            # 无后缀的 npz bundle：仍按 bundle 语义读取，且只读一次。
            with loaded as bundle:
                array = _first_bundle_array(bundle, path)
    value = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
    return hashlib.sha256(value.tobytes()).hexdigest()


__all__ = [
    "SCENE_MANIFEST_SCHEMA_VERSION",
    "OfflineSceneManifest",
    "OfflineSceneIndex",
    "build_track_identity",
    "trajectory_digest",
]


def trajectory_digest(actor_trajectories: Any, trajectory_timebase: Any) -> str:
    """moving scene 轨迹内容哈希；generator 写入与 manifest 复核共用。"""
    return _trajectory_digest(actor_trajectories, trajectory_timebase)
