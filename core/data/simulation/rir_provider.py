"""共享 spatial RIR provider：offline 与 online backend 使用同一 typed seam。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from core.data.simulation.rir_config import OnlineRIRConfig
from core.data.simulation.rir_geometry import sample_rir_scene


@dataclass(frozen=True)
class SpatialRIRResult:
    """一个 source×microphone RIR bank 及其可审计身份。

    ``rirs`` 接受 canonical ``[K,C,L]``，也接受历史 mono ``[K,L]``；后者只在
    ``legacy_rirs`` 保持二维布局，``canonical_rirs`` 始终返回 ``[K,1,L]``。
    """

    rirs: np.ndarray
    geometry: Mapping[str, Any]
    provenance: Mapping[str, Any]
    generator_version: str
    content_hash: str | None = None
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        value = np.asarray(self.rirs, dtype=np.float32)
        if value.ndim not in (2, 3) or value.shape[0] < 1 or value.shape[-1] < 1:
            raise ValueError(f"spatial RIRs must have shape [K,C,L] or legacy [K,L], got {value.shape}")
        if value.ndim == 3 and value.shape[1] < 1:
            raise ValueError("spatial RIRs must have a positive microphone axis")
        if not np.isfinite(value).all():
            raise ValueError("spatial RIRs must be finite")
        if not isinstance(self.geometry, Mapping) or not isinstance(self.provenance, Mapping):
            raise TypeError("geometry and provenance must be mappings")
        if not isinstance(self.generator_version, str) or not self.generator_version.strip():
            raise ValueError("generator_version must be a non-empty string")
        value = np.ascontiguousarray(value, dtype="<f4")
        value.setflags(write=False)
        digest = hashlib.sha256(value.tobytes()).hexdigest()
        if self.content_hash is not None and self.content_hash != digest:
            raise ValueError("spatial RIR content_hash does not match waveform bytes")
        object.__setattr__(self, "rirs", value)
        object.__setattr__(self, "geometry", dict(self.geometry))
        object.__setattr__(self, "provenance", dict(self.provenance))
        if self.details is not None and not isinstance(self.details, Mapping):
            raise TypeError("details must be a mapping or None")
        object.__setattr__(self, "details", dict(self.details or {}))
        object.__setattr__(self, "content_hash", digest)

    @property
    def canonical_rirs(self) -> np.ndarray:
        return self.rirs[:, None, :] if self.rirs.ndim == 2 else self.rirs

    @property
    def legacy_rirs(self) -> np.ndarray:
        if self.rirs.ndim == 2:
            return self.rirs
        if self.rirs.shape[1] != 1:
            raise ValueError("legacy [K,L] projection requires C=1")
        return self.rirs[:, 0, :]

    @property
    def rir_sha256(self) -> str:
        return self.content_hash

    @property
    def rir_hash(self) -> str:
        return self.content_hash

    @property
    def hash(self) -> str:
        return self.content_hash

    @property
    def facts(self) -> dict[str, Any]:
        return {
            **self.details,
            "geometry": dict(self.geometry),
            "provenance": dict(self.provenance),
            "generator_version": self.generator_version,
            "content_hash": self.content_hash,
            "rir_sha256": self.content_hash,
            "shape": list(self.rirs.shape),
            "provider_contract": "[K,C,L]" if self.rirs.ndim == 3 else "[K,L]",
        }

    def __iter__(self):
        # 兼容已有 ``wave, facts = provider.sample_scene(...)`` 调用方。
        yield self.rirs
        yield self.facts

    def __array__(self, dtype=None):
        return np.asarray(self.rirs, dtype=dtype)

    def __getitem__(self, item):
        return self.rirs[item]

    @property
    def shape(self):
        return self.rirs.shape

    @property
    def ndim(self):
        return self.rirs.ndim


@dataclass(frozen=True)
class SpatialRIRTrackSet:
    """同一空间场景的 full/direct/early RIR 轨道合同。

    ``full`` 必须存在；``direct`` 与 ``early`` 若存在，必须共享同一 ``[K,C,L]``
    source/channel 布局。该对象只描述 RIR，不执行 waveform rendering。
    """

    full: SpatialRIRResult
    direct: SpatialRIRResult | None = None
    early: SpatialRIRResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.full, SpatialRIRResult):
            raise TypeError("full must be a SpatialRIRResult")
        full = self.full.canonical_rirs
        for name, value in (("direct", self.direct), ("early", self.early)):
            if value is not None:
                if not isinstance(value, SpatialRIRResult):
                    raise TypeError(f"{name} must be a SpatialRIRResult or None")
                if value.canonical_rirs.shape[:2] != full.shape[:2]:
                    raise ValueError(f"{name} RIR must match full source/channel axes")
                if value.geometry != self.full.geometry:
                    raise ValueError(f"{name} RIR geometry must match full geometry")
                if value.provenance != self.full.provenance:
                    raise ValueError(f"{name} RIR provenance must match full provenance")

    @property
    def geometry(self) -> Mapping[str, Any]:
        return self.full.geometry

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self.full.provenance

    @property
    def generator_version(self) -> str:
        return self.full.generator_version

    @property
    def content_hash(self) -> str:
        return self.full.content_hash

    @property
    def rir_sha256(self) -> str:
        return self.full.rir_sha256

    @property
    def rirs(self) -> np.ndarray:
        """兼容 full-only provider 的主 RIR 访问语义。"""
        return self.full.rirs

    @property
    def canonical_rirs(self) -> np.ndarray:
        return self.full.canonical_rirs

    @property
    def facts(self) -> dict[str, Any]:
        return {
            **self.full.facts,
            "track_set": {
                "full": self.full.content_hash,
                "direct": None if self.direct is None else self.direct.content_hash,
                "early": None if self.early is None else self.early.content_hash,
            },
            "direct_path_exact": self.direct is not None,
        }

    @property
    def shape(self):
        return self.full.shape

    @property
    def ndim(self):
        return self.full.ndim

    def __iter__(self):
        yield self.rirs
        yield self.facts

    def __array__(self, dtype=None):
        return np.asarray(self.rirs, dtype=dtype)

    def __getitem__(self, item):
        return self.rirs[item]


@dataclass(frozen=True)
class MovingSpatialRIRTrackSet:
    """同一 room/trajectory 的 full 及可选 direct bank，布局 ``[P,K,C,L]``。

    legacy mono plan（``mic_loc`` 为 ``[3]``）的 bank 布局为 ``[P,K,L]``；
    ``plan.num_samples`` 是 RIR kernel 长度；moving waveform 长度必须另由
    ``plan.waveform_num_samples`` 声明。``validate_moving_rir_plan`` 与 timebase
    校验对两种布局一致。
    """

    full: np.ndarray
    direct: np.ndarray | None
    plan: Mapping[str, Any]
    track_facts: Mapping[str, Any]

    def __post_init__(self):
        from core.data.simulation.gpurir_backend import validate_moving_rir_plan

        trajectory = validate_moving_rir_plan(
            self.plan, track_count=1 if self.direct is None else 2
        )
        microphones = np.asarray(self.plan['mic_loc'])
        legacy_mono = microphones.shape == (3,)
        if legacy_mono:
            expected = (len(trajectory), trajectory.shape[1], self.plan['num_samples'])
        else:
            expected = (len(trajectory), trajectory.shape[1], len(microphones), self.plan['num_samples'])
        for name in ('full',):
            wave = np.array(getattr(self, name), dtype='<f4', copy=True)
            if wave.shape != expected or not np.isfinite(wave).all():
                raise ValueError(f'moving {name} RIR must be finite with shape {expected}')
            wave.setflags(write=False)
            object.__setattr__(self, name, wave)
        if self.direct is not None:
            wave = np.array(self.direct, dtype='<f4', copy=True)
            if wave.shape != expected or not np.isfinite(wave).all():
                raise ValueError(f'moving direct RIR must be finite with shape {expected}')
            wave.setflags(write=False)
            object.__setattr__(self, 'direct', wave)

    @property
    def facts(self):
        tracks = {'full': hashlib.sha256(self.full.tobytes()).hexdigest()}
        if self.direct is not None:
            tracks['direct'] = hashlib.sha256(self.direct.tobytes()).hexdigest()
        return {**self.plan, 'moving': True, 'track_facts': dict(self.track_facts),
                'track_set': tracks, 'direct_path_exact': self.direct is not None}


class MovingRIRCapabilityError(RuntimeError):
    """当前 moving RIR backend 不具备所请求的多 actor 能力。"""


@dataclass(frozen=True)
class MovingSceneRIRRequest:
    """多 actor moving RIR 的最小 typed 请求合同。

    请求只描述共享场景与各 actor 的轨迹，不生成或伪造任何 RIR。所有 actor
    必须共享 room、阵列和 trajectory timebase；actor 仍可拥有不同位置轨迹。
    ``track_kinds=('full',)`` 可跳过未消费的 direct 轨道。
    """

    sample_rate: int
    num_samples: int
    actor_trajectories: Mapping[str, Any]
    room: Mapping[str, Any]
    array: Mapping[str, Any]
    timebase: Any
    source_uids: tuple[str, ...] = ()
    scene_uid: str | None = None
    actor_roles: tuple[str, ...] = ()
    track_kinds: tuple[str, ...] = ('full', 'direct')

    def __post_init__(self) -> None:
        from core.data.simulation.trajectory import SourceTrajectory, TrajectoryTimebase

        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise ValueError("moving scene sample_rate must be a positive exact int")
        if type(self.num_samples) is not int or self.num_samples <= 0:
            raise ValueError("moving scene num_samples must be a positive exact int")
        if not isinstance(self.actor_trajectories, Mapping) or not self.actor_trajectories:
            raise ValueError("actor_trajectories must be a non-empty mapping")
        if not isinstance(self.room, Mapping) or not self.room:
            raise ValueError("moving scene room must be a non-empty mapping")
        if not isinstance(self.array, Mapping) or not self.array:
            raise ValueError("moving scene array must be a non-empty mapping")
        if not isinstance(self.timebase, TrajectoryTimebase):
            raise TypeError("moving scene timebase must be a TrajectoryTimebase")
        if self.timebase.sample_rate != self.sample_rate or self.timebase.num_samples != self.num_samples:
            raise ValueError("all moving actors must share the request sample/timebase contract")
        keys = tuple(self.actor_trajectories)
        if any(not isinstance(key, str) or not key.strip() for key in keys):
            raise ValueError("actor_trajectories keys must be non-empty strings")
        if len(set(keys)) != len(keys):
            raise ValueError("actor_trajectories keys must be unique")
        for actor_uid, trajectory in self.actor_trajectories.items():
            if not isinstance(trajectory, SourceTrajectory):
                raise TypeError(f"actor trajectory for {actor_uid!r} must be a SourceTrajectory")
            if trajectory.timebase != self.timebase:
                raise ValueError("all actor trajectories must share one trajectory timebase")
            if trajectory.source_count != 1:
                raise ValueError("each actor trajectory must describe exactly one source")
        if self.actor_roles and len(self.actor_roles) != len(keys):
            raise ValueError("actor_roles must match actor order")
        if self.source_uids and tuple(self.source_uids) != keys:
            raise ValueError("source_uids must match actor_trajectories insertion order")
        track_kinds = tuple(self.track_kinds)
        if not track_kinds or 'full' not in track_kinds or any(
            kind not in {'full', 'direct'} for kind in track_kinds
        ):
            raise ValueError("moving scene track_kinds require full and support full/direct")
        object.__setattr__(self, "track_kinds", track_kinds)
        object.__setattr__(self, "actor_trajectories", dict(self.actor_trajectories))
        object.__setattr__(self, "room", dict(self.room))
        object.__setattr__(self, "array", dict(self.array))
        object.__setattr__(self, "source_uids", keys if not self.source_uids else tuple(self.source_uids))

    @property
    def actor_count(self) -> int:
        return len(self.actor_trajectories)

    @property
    def microphone_count(self) -> int:
        positions = self.array.get("positions_m", self.array.get("mic_loc"))
        if positions is None:
            raise ValueError("moving scene array must declare positions_m or mic_loc")
        positions = np.asarray(positions)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 1:
            raise ValueError("moving scene array positions must have shape [C,3]")
        return int(positions.shape[0])


def validate_moving_scene_rir_request(request: MovingSceneRIRRequest) -> MovingSceneRIRRequest:
    """在 provider 边界校验并返回多 actor moving 请求。"""
    if not isinstance(request, MovingSceneRIRRequest):
        raise TypeError("moving scene RIR request must be a MovingSceneRIRRequest")
    return request


class SpatialRIRProvider:
    """所有 spatial RIR adapter 必须实现的最小 interface。"""

    def sample_moving_scene_tracks(self, request: MovingSceneRIRRequest, *, normalize=True):
        """请求共享 scene 下各 actor 独立轨迹的 moving RIR。

        旧的 ``sample_moving_spatial_tracks`` 只表达单一 source-group trajectory；
        默认实现明确拒绝该新能力，禁止静默退化或复制轨迹。
        """
        validate_moving_scene_rir_request(request)
        raise MovingRIRCapabilityError(
            "moving RIR backend does not support independent actor trajectories "
            "with shared room/array/timebase"
        )

    def get_rirs(self, sample_rate, source_count, rng, *, source_uids=(), normalize=True, distance_range=None):
        raise NotImplementedError

    def sample_scene_tracks(self, scene_plan, *, normalize=True, track_kinds=("full",)):
        """按已解析的 AcousticScenePlan 生成静态 actor RIR。

        生产 spatial 路径必须从完整 scene plan 取 room、array 和 actor positions；
        旧的 ``sample_spatial_scene`` 仅保留给 legacy 调用方。
        """
        from core.data.acoustic_scene_plan import AcousticScenePlan

        if not isinstance(scene_plan, AcousticScenePlan):
            raise TypeError("scene_plan must be an AcousticScenePlan")
        if scene_plan.room is None or scene_plan.array is None:
            raise ValueError("scene_plan must resolve room and microphone array")
        if not scene_plan.actors or any(actor.position_m is None for actor in scene_plan.actors):
            raise ValueError("static scene_plan must resolve every actor position")
        raise NotImplementedError("scene-level static RIR is not implemented by this provider")

    def sample_spatial_scene(
        self,
        sample_rate,
        source_count,
        rng,
        *,
        source_uids=(),
        normalize=True,
        distance_range=None,
    ):
        """统一 spatial ABI；``source_uids`` 始终作为显式 keyword 传递。"""
        return self.get_rirs(
            sample_rate,
            source_count,
            rng,
            source_uids=tuple(source_uids),
            normalize=normalize,
            distance_range=distance_range,
        )

    def __call__(
        self,
        sample_rate,
        source_count,
        rng,
        *,
        source_uids=(),
        normalize=True,
        distance_range=None,
    ):
        return self.get_rirs(
            sample_rate,
            source_count,
            rng,
            source_uids=tuple(source_uids),
            normalize=normalize,
            distance_range=distance_range,
        )


class OfflineRIRProvider(SpatialRIRProvider):
    """从显式 source-keyed manifest 读取 RIR，不把普通 audio SCP 当作 geometry。"""

    def __init__(self, manifest, *, geometry: Mapping[str, Any] | None = None, require_scene_identity: bool = False):
        self.manifest = manifest
        self.geometry = dict(geometry or {})
        from core.data.simulation.scene_manifest import OfflineSceneManifest
        self.scene_manifest = OfflineSceneManifest.load(manifest, require_typed=require_scene_identity)
        self.require_scene_identity = bool(require_scene_identity)
        from core.data.spatial import channel_count_from_geometry

        self.channels = channel_count_from_geometry(self.geometry)

    def resolve_scene_manifest(self, source_uids=(), sample_index=0, scene_uid=None, sample_rate=None, geometry=None, roles=None):
        """从单 scene manifest 或 split JSONL index 解析一个具体 scene。"""
        from core.data.simulation.scene_manifest import OfflineSceneIndex, OfflineSceneManifest

        if isinstance(self.scene_manifest, OfflineSceneIndex):
            if scene_uid:
                return self.scene_manifest.get(scene_uid)
            return self.scene_manifest.select(
                tuple(source_uids), int(sample_index), sample_rate=sample_rate,
                geometry=geometry, roles=roles,
            )
        if isinstance(self.scene_manifest, OfflineSceneManifest):
            return self.scene_manifest
        raise ValueError("offline scene provider requires a typed scene manifest/index")

    def sample_scene_tracks(self, scene_plan, *, normalize=True, track_kinds=("full",)):
        """从 typed scene manifest 按 plan identity 读取静态 actor RIR。"""
        from core.data.acoustic_scene_plan import AcousticScenePlan, _thaw_json
        if not isinstance(scene_plan, AcousticScenePlan):
            raise TypeError("scene_plan must be an AcousticScenePlan")
        if self.scene_manifest is None:
            raise ValueError("scene-level offline RIR requires a typed scene manifest")
        if scene_plan.room is None or scene_plan.array is None:
            raise ValueError("scene_plan must contain room and microphone array")
        requested_tracks = tuple(dict.fromkeys(track_kinds))
        if not requested_tracks or requested_tracks[0] != "full" or any(
            track not in {"full", "direct", "early"} for track in requested_tracks
        ):
            raise ValueError("offline scene-level provider requires full first and supports full/direct/early")
        scene_uid = scene_plan.identity.get("scene_uid")
        manifest = self.resolve_scene_manifest(
            source_uids=tuple(actor.actor_uid for actor in scene_plan.actors),
            sample_index=int(scene_plan.identity.get("sample_index", 0)),
            scene_uid=scene_uid,
        )
        if str(scene_uid) != manifest.scene_uid:
            raise ValueError("offline scene manifest scene_uid does not match scene_plan")
        source_uids = tuple(actor.actor_uid for actor in scene_plan.actors)
        if manifest.metadata.get("sample_rate") != scene_plan.sample_rate:
            raise ValueError("offline scene manifest sample_rate mismatch")
        if _thaw_json(scene_plan.room) != _thaw_json(manifest.room):
            raise ValueError("offline scene room does not match manifest")
        positions = manifest.metadata.get("actor_positions_m", {})
        for actor in scene_plan.actors:
            if actor.actor_uid not in positions or actor.position_m is None or not np.allclose(
                actor.position_m, positions[actor.actor_uid], rtol=0, atol=1e-6
            ) or actor.trajectory is not None:
                raise ValueError("offline actor position does not match static manifest")
        manifest.validate_request(
            manifest.actor_order,
            {"positions_m": np.asarray(scene_plan.array.positions_m, dtype=np.float64).tolist()},
        )
        if len(set(source_uids)) != len(source_uids) or not set(source_uids).issubset(manifest.actor_order):
            raise ValueError("requested actor order must select unique manifest actors")
        from core.data.spatial import load_offline_rir_bank_with_metadata

        tracks = {}
        geometry = dict(manifest.geometry)
        provenance = {
            "scene_uid": scene_uid, "actor_order": list(source_uids), "source_order": list(source_uids),
            "room": _thaw_json(scene_plan.room), "sample_rate": scene_plan.sample_rate,
            "actor_positions_m": positions,
        }
        for track in requested_tracks:
            bank, metadata = load_offline_rir_bank_with_metadata(
                manifest.path, source_uids, geometry=geometry, track=track,
                include_internal_metadata=True,
            )
            hashes = manifest.track_hashes(source_uids, track)
            for i, uid in enumerate(source_uids):
                actual_hash = hashlib.sha256(np.ascontiguousarray(bank[i], dtype='<f4').tobytes()).hexdigest()
                if actual_hash != hashes[uid]:
                    raise ValueError(f"offline {track} content hash mismatch for actor {uid}")
            tracks[track] = SpatialRIRResult(
                np.asarray(bank), geometry, provenance, manifest.generator_version,
            )
        return SpatialRIRTrackSet(
            full=tracks["full"], direct=tracks.get("direct"), early=tracks.get("early")
        )

    def sample_moving_scene_tracks(self, request, *, normalize=True):
        """从 typed moving scene manifest 读取逐 actor 的 ``[P,A,C,L]`` bank。

        manifest 的 ``actor_order`` 必须与请求的 ``source_uids`` 完全一致；
        轨迹与 timebase 以 manifest 为权威，并与请求逐点复核，禁止静默接受
        漂移。actor hash 按 ``bank[:, i]``（``[P,C,L]`` 字节）复核。
        """
        from core.data.spatial import load_offline_moving_rir_bank_with_metadata
        from core.data.simulation.scene_manifest import OfflineSceneManifest
        from core.data.simulation.trajectory import TrajectoryTimebase

        request = validate_moving_scene_rir_request(request)
        if self.scene_manifest is None:
            raise ValueError("offline moving scene RIR requires a typed scene manifest")
        manifest = self.resolve_scene_manifest(
            source_uids=tuple(request.source_uids), scene_uid=request.scene_uid,
        )
        if not isinstance(manifest, OfflineSceneManifest):
            raise ValueError("offline moving scene request must resolve one typed manifest")
        if manifest.metadata.get("moving") is not True:
            raise ValueError("offline scene manifest is not a moving scene")
        source_uids = tuple(request.source_uids)
        if len(set(source_uids)) != len(source_uids) or not set(source_uids).issubset(
            manifest.actor_order
        ):
            raise ValueError(
                "offline moving scene requested actors must be unique manifest actors"
            )
        metadata = manifest.metadata
        if metadata.get("sample_rate") != request.sample_rate:
            raise ValueError("offline moving scene sample_rate mismatch")
        timebase_raw = metadata["trajectory_timebase"]
        clock = TrajectoryTimebase(
            int(timebase_raw["sample_rate"]), int(timebase_raw["num_samples"]),
            int(timebase_raw["hop_samples"]), int(timebase_raw["frame_count"]),
            int(timebase_raw.get("start_sample", 0)),
        )
        if clock != request.timebase:
            raise ValueError("offline moving scene timebase does not match request")
        request_positions = np.asarray(request.array.get("positions_m"), dtype=np.float64)
        manifest_positions = np.asarray(manifest.geometry["positions_m"], dtype=np.float64)
        if request_positions.shape != manifest_positions.shape or not np.allclose(
            request_positions, manifest_positions, rtol=0.0, atol=1e-6
        ):
            raise ValueError("offline moving scene microphone geometry does not match request")
        declared_trajectories = metadata["actor_trajectories"]
        for uid in source_uids:
            requested = request.actor_trajectories[uid].positions_m
            declared = np.asarray(declared_trajectories[uid], dtype=np.float32)
            if not np.array_equal(requested, declared):
                raise ValueError(f"offline moving trajectory mismatch for actor {uid}")
        tracks: dict[str, np.ndarray] = {}
        requested_tracks = tuple(request.track_kinds)
        for name in requested_tracks:
            bank, _ = load_offline_moving_rir_bank_with_metadata(
                manifest.path, source_uids, geometry=dict(manifest.geometry), track=name,
            )
            hashes = manifest.track_hashes(source_uids, name)
            for i, uid in enumerate(source_uids):
                actual = hashlib.sha256(
                    np.ascontiguousarray(bank[:, i], dtype="<f4").tobytes()
                ).hexdigest()
                if actual != hashes[uid]:
                    raise ValueError(
                        f"offline moving {name} content hash mismatch for actor {uid}"
                    )
            tracks[name] = bank
        peak = max(float(np.max(np.abs(bank))) for bank in tracks.values())
        scale = 1.0 / peak if normalize and peak > 0 else 1.0
        if scale != 1.0:
            tracks = {name: (bank * scale).astype(np.float32) for name, bank in tracks.items()}
        positions = np.concatenate(
            [request.actor_trajectories[uid].positions_m for uid in source_uids], axis=1
        )
        plan = {
            "group_uid": manifest.scene_uid,
            "scene_uid": manifest.scene_uid,
            "sample_rate": int(metadata["sample_rate"]),
            "num_samples": int(tracks["full"].shape[-1]),
            "waveform_num_samples": int(request.num_samples),
            "room_dim": list(manifest.room["room_dim_m"]),
            "rt60": manifest.room.get("rt60"),
            "diffuse_db": float(manifest.room.get("diffuse_db", 15.0)),
            "sound_speed": float(manifest.room.get("sound_speed", 343.0)),
            "mic_loc": [list(map(float, row)) for row in manifest_positions],
            "source_loc": positions[0].tolist(),
            "source_ids": list(source_uids),
            "trajectory": positions.tolist(),
            "trajectory_timebase": dict(timebase_raw),
            "array": dict(manifest.geometry),
            "geometry": dict(manifest.geometry),
            "room": dict(manifest.room),
            "reference_channel": int(manifest.geometry.get("reference_channel", 0)),
        }
        track_facts = {
            name: {
                "scene_uid": manifest.scene_uid,
                "actor_order": list(source_uids),
                "track_kind": name,
                "generator_version": manifest.generator_version,
                "group_normalization_scale": scale,
                "provenance": {"backend": "offline", "manifest": str(manifest.path)},
            }
            for name in requested_tracks
        }
        return MovingSpatialRIRTrackSet(
            tracks["full"], tracks.get("direct"), plan, track_facts
        )

    def get_rirs(self, sample_rate, source_count, rng, *, source_uids=(), normalize=True, distance_range=None):
        del sample_rate, rng, normalize, distance_range
        from core.data.simulation.scene_manifest import OfflineSceneIndex
        if isinstance(self.scene_manifest, OfflineSceneIndex):
            raise ValueError("split scene index requires scene-level sample_scene_tracks(scene_plan)")
        if type(source_count) is not int or source_count < 1:
            raise ValueError("supplied spatial provider source_count must be a positive exact int")
        keys = tuple(source_uids)
        if len(keys) != int(source_count):
            raise ValueError("supplied spatial provider requires one source_uids entry per source")
        from core.data.spatial import load_offline_rir_bank_with_metadata

        bank, metadata = load_offline_rir_bank_with_metadata(
            self.manifest,
            keys,
            geometry=self.geometry,
            track="full",
            include_internal_metadata=True,
        )
        if self.scene_manifest is not None:
            self.scene_manifest.validate_request(keys, self.geometry)
        elif self.require_scene_identity:
            raise ValueError("offline spatial RIR requires a typed scene manifest with identity")
        metadata_provenance = metadata.get("provenance", {})
        if not isinstance(metadata_provenance, Mapping):
            raise ValueError("offline spatial RIR provenance must be a mapping")
        declared_hash = metadata.get(
            "rir_sha256",
            metadata.get("content_hash", metadata_provenance.get("rir_sha256")),
        )
        computed_hash = hashlib.sha256(np.ascontiguousarray(bank, dtype="<f4").tobytes()).hexdigest()
        if declared_hash is not None and declared_hash != computed_hash:
            raise ValueError("offline spatial RIR metadata hash does not match waveform content")
        metadata_geometry = metadata.get("geometry")
        if metadata_geometry is None:
            geometry_keys = {
                "array_profile", "geometry_kind", "topology", "positions_m",
                "coordinate_system", "spacing_m", "reference_channel", "num_channels",
            }
            metadata_geometry = {key: metadata[key] for key in geometry_keys if key in metadata}
        if not isinstance(metadata_geometry, Mapping):
            raise ValueError("offline spatial RIR geometry must be a mapping")
        merged_geometry = {**metadata_geometry, **self.geometry, "num_channels": self.channels}
        generator_version = str(
            metadata.get(
                "generator_version", metadata_provenance.get("generator_version", "spatial-rir-contract-v1")
            )
        )
        provenance = dict(metadata_provenance)
        if self.scene_manifest is not None:
            if generator_version != self.scene_manifest.generator_version:
                raise ValueError("offline spatial RIR generator_version does not match scene manifest")
            expected_hashes = self.scene_manifest.track_hashes(keys, "full")
            for key, value in zip(keys, bank, strict=True):
                actual = hashlib.sha256(np.ascontiguousarray(value, dtype="<f4").tobytes()).hexdigest()
                if expected_hashes[key] != actual:
                    raise ValueError(f"offline spatial RIR full content_hash mismatch for actor {key!r}")
            provenance = {**provenance, "scene_uid": self.scene_manifest.scene_uid,
                          "actor_order": list(self.scene_manifest.actor_order),
                          "room": dict(self.scene_manifest.room),
                          "geometry": dict(self.scene_manifest.geometry)}
        declared_order = metadata.get("source_order", provenance.get("source_order"))
        if declared_order is not None and tuple(declared_order) != keys:
            raise ValueError("offline spatial RIR source_order does not match requested source_uids")
        full = SpatialRIRResult(bank, merged_geometry, provenance, generator_version)
        entry_modes = metadata.get("_entry_modes", {})
        if any(mode == "tracks" for mode in entry_modes.values()):
            if not all(entry_modes.get(key) == "tracks" for key in keys):
                raise ValueError("offline spatial RIR manifest cannot mix full-only and track-set entries")
            direct_bank, direct_metadata = load_offline_rir_bank_with_metadata(
                self.manifest,
                keys,
                geometry=self.geometry,
                track="direct",
                include_internal_metadata=True,
            )
            direct_hash = (
                direct_metadata.get("direct_rir_sha256")
                or direct_metadata.get("track_hashes", {}).get("direct")
            )
            if self.scene_manifest is not None:
                direct_hash = None
            direct_result = SpatialRIRResult(
                direct_bank, merged_geometry, provenance, generator_version,
                content_hash=direct_hash,
            )
            if self.scene_manifest is not None:
                expected_direct = self.scene_manifest.track_hashes(keys, "direct")
                for key, value in zip(keys, direct_bank, strict=True):
                    actual = hashlib.sha256(np.ascontiguousarray(value, dtype="<f4").tobytes()).hexdigest()
                    if expected_direct[key] != actual:
                        raise ValueError(f"offline spatial RIR direct content_hash mismatch for actor {key!r}")
            early_result = None
            if all(metadata.get("_entry_tracks", {}).get(key, {}).get("early") for key in keys):
                early_bank, early_metadata = load_offline_rir_bank_with_metadata(
                    self.manifest,
                    keys,
                    geometry=self.geometry,
                    track="early",
                    include_internal_metadata=True,
                )
                early_result = SpatialRIRResult(
                    early_bank, merged_geometry, provenance, generator_version,
                    content_hash=(
                        early_metadata.get("early_rir_sha256")
                        or early_metadata.get("track_hashes", {}).get("early")
                    ),
                )
            return SpatialRIRTrackSet(full=full, direct=direct_result, early=early_result)
        return full

    load = get_rirs
    sample_scene = get_rirs


class OnlineRIRProvider(SpatialRIRProvider):

    def __init__(self, config, client):
        self.config = config
        self.client = client

    @staticmethod
    def _scene_payload(scene_plan):
        """把不可变 domain plan 投影成 JSON，不复制采样行为。"""
        from core.data.acoustic_scene_plan import AcousticScenePlan
        if not isinstance(scene_plan, AcousticScenePlan):
            raise TypeError("scene_plan must be an AcousticScenePlan")
        payload = scene_plan.to_dict()
        if scene_plan.room is None or scene_plan.array is None or not scene_plan.actors:
            raise ValueError("scene plan must resolve room, array and actors")
        payload["scene_uid"] = scene_plan.identity.get("scene_uid")
        payload["array"] = {
            "positions_m": np.asarray(scene_plan.array.positions_m).tolist(),
            "reference_channel": scene_plan.array.reference_channel,
        }
        payload["actors"] = [{
            "actor_uid": actor.actor_uid, "role": actor.role,
            "position_m": list(actor.position_m) if actor.position_m is not None else None,
            **({"trajectory": actor.trajectory.positions_m.tolist()} if actor.trajectory is not None else {}),
        } for actor in scene_plan.actors]
        return payload

    def _generate_resolved_scene(self, scene, *, moving, normalize, track_kind=None):
        """经真实 scene IPC 请求整组轨道，统一校验与整组归一化。"""
        from core.data.simulation.rir_service import _canonical_scene_request, _request_shape, OnlineRIRError
        if type(normalize) is not bool:
            raise TypeError("normalize must be a bool")
        if not callable(getattr(self.client, "generate_scene", None)):
            raise MovingRIRCapabilityError("RIR client must implement generate_scene for resolved scenes")
        # domain plan 冻结 tuple，wire 使用 JSON array；双方基于同一规范表示比较。
        scene = json.loads(json.dumps(scene, allow_nan=False))
        request_kind = "scene_moving" if moving else "scene_static"
        canonical = _canonical_scene_request(request_kind, scene, normalize=False)
        raw = canonical["plan"]
        track_kind = track_kind or ("scene_moving_full_direct" if moving else "full_direct")
        expected = _request_shape(raw, track_kind)
        names = ("full",) if track_kind in {"full", "scene_moving_full"} else ("full", "direct")
        generate_scene = self.client.generate_scene
        generated = generate_scene(
            scene, moving=moving, normalize=False, track_kind=track_kind,
        )
        if not isinstance(generated, Mapping) or set(generated) != set(names):
            raise OnlineRIRError("scene response requires full/direct tracks")
        arrays, facts = {}, {}
        for name in names:
            wave, metadata = generated[name]
            wave = np.asarray(wave, dtype='<f4')
            if wave.shape != expected or not np.isfinite(wave).all():
                raise OnlineRIRError(f"scene {name} shape/value mismatch: {wave.shape}, expected {expected}")
            metadata = dict(metadata)
            if metadata.get("scene_uid") != scene["scene_uid"] or metadata.get("actor_order") != raw["source_ids"]:
                raise OnlineRIRError(f"scene {name} actor/scene identity mismatch")
            if metadata.get("track_kind") != name:
                raise OnlineRIRError(f"scene {name} track kind mismatch")
            if hashlib.sha256(wave.tobytes()).hexdigest() != metadata.get("rir_sha256"):
                raise OnlineRIRError(f"scene {name} content identity mismatch")
            for key in ("room_dim", "mic_loc", "source_loc", "sample_rate"):
                if metadata.get(key) != raw[key]:
                    raise OnlineRIRError(f"scene {name} geometry mismatch: {key}")
            if moving and metadata.get("actor_trajectories") != raw["actor_trajectories"]:
                raise OnlineRIRError(f"scene {name} actor trajectory mismatch")
            arrays[name], facts[name] = wave, metadata
        peak = max(float(np.max(np.abs(wave))) for wave in arrays.values())
        scale = 1.0 / peak if normalize and peak > 0 else 1.0
        for name in arrays:
            arrays[name] = (arrays[name] * scale).astype('<f4')
            facts[name].update(
                raw_rir_sha256=facts[name]["rir_sha256"],
                group_normalization_scale=scale,
                rir_sha256=hashlib.sha256(arrays[name].tobytes()).hexdigest(),
            )
        return arrays, facts, raw

    def sample_scene_tracks(self, scene_plan, *, normalize=True, track_kinds=("full",)):
        """使用已采样的静态 scene；early 在 renderer 依据显式窗口派生。

        未请求 ``direct`` 时，online backend 只生成 full 轨道，避免为
        full+early 训练路径重复执行一次 direct ``simulateRIR``。
        """
        if not track_kinds or "full" not in track_kinds or any(k not in {"full", "direct", "early"} for k in track_kinds):
            raise ValueError("scene tracks require full and only support full/direct/early")
        if scene_plan.timebase is not None or any(a.trajectory is not None for a in scene_plan.actors):
            raise ValueError("static scene cannot carry actor trajectories")
        scene = self._scene_payload(scene_plan)
        requested_track_kind = "full_direct" if "direct" in track_kinds else "full"
        arrays, facts, raw = self._generate_resolved_scene(
            scene, moving=False, normalize=normalize, track_kind=requested_track_kind,
        )
        provenance = {
            "scene_uid": scene["scene_uid"], "source_order": raw["source_ids"],
            "actor_order": raw["source_ids"], "room": scene["room"],
            "actor_positions_m": {a["actor_uid"]: a["position_m"] for a in scene["actors"]},
            "sample_rate": scene["sample_rate"],
        }
        tracks = {name: SpatialRIRResult(
            arrays[name], scene["array"], provenance, facts[name]["generator_version"], details=facts[name]
        ) for name in arrays}
        return SpatialRIRTrackSet(full=tracks["full"], direct=tracks.get("direct"))

    def sample_moving_scene_tracks(self, request: MovingSceneRIRRequest, *, normalize=True):
        """独立 actor 轨迹共用 room/array/timebase，实际经过 scene IPC。"""
        validate_moving_scene_rir_request(request)
        scene_uid = request.scene_uid or request.room.get("scene_uid")
        if not scene_uid:
            raise ValueError("moving scene request requires scene_uid")
        scene = {
            "scene_uid": scene_uid, "sample_rate": request.sample_rate,
            "num_samples": request.num_samples, "room": dict(request.room),
            "array": dict(request.array), "timebase": request.timebase.to_dict(),
            "actors": [{
                "actor_uid": uid, "role": request.actor_roles[i] if request.actor_roles else "source",
                "position_m": trajectory.positions_m[0, 0].tolist(),
                "trajectory": trajectory.positions_m.tolist(),
            } for i, (uid, trajectory) in enumerate(request.actor_trajectories.items())],
        }
        requested_track_kind = (
            "scene_moving_full_direct" if "direct" in request.track_kinds
            else "scene_moving_full"
        )
        arrays, facts, raw = self._generate_resolved_scene(
            scene, moving=True, normalize=normalize, track_kind=requested_track_kind,
        )
        # [P,A,3] 是独立 actor 轨迹按固定顺序堆叠的无损表示，不意味着同步平移。
        trajectories = tuple(request.actor_trajectories.items())
        trajectory = np.concatenate([item.positions_m for _, item in trajectories], axis=1)
        duration_sec = float(
            request.timebase.frame_times[-1] - request.timebase.frame_times[0]
        )
        motion_speeds = {
            uid: float(
                np.linalg.norm(value.positions_m[-1, 0] - value.positions_m[0, 0])
                / duration_sec
            )
            for uid, value in trajectories
        }
        raw.update(
            trajectory=trajectory.tolist(), source_loc=trajectory[0].tolist(),
            geometry=dict(request.array), room=dict(request.room),
            motion_speeds_mps=motion_speeds, moving_duration_sec=duration_sec,
        )
        return MovingSpatialRIRTrackSet(arrays["full"], arrays.get("direct"), raw, facts)

    @classmethod
    def from_simulation(cls, cfg):
        from core.data.simulation.rir_service import acquire_rir_client
        config = OnlineRIRConfig.from_simulation(cfg)
        return cls(config, acquire_rir_client(config))

    @staticmethod
    def _resolve_source_uids(source_count, source_uids):
        keys = tuple(source_uids)
        if not keys:
            return tuple(f"source/{index}" for index in range(int(source_count)))
        if len(keys) != int(source_count):
            raise ValueError(
                "online spatial RIR provider requires one source_uids entry per source"
            )
        if any(not isinstance(key, str) or not key.strip() for key in keys):
            raise ValueError("online spatial RIR provider source_uids must be non-empty strings")
        return keys

    def _sample(
        self,
        sample_rate,
        source_count,
        rng,
        *,
        source_uids=(),
        normalize,
        distance_range=None,
        spatial=False,
    ):
        from core.data.simulation.rir_service import OnlineRIRError
        try:
            keys = self._resolve_source_uids(source_count, source_uids)
            plan = sample_rir_scene(
                self.config,
                sample_rate,
                source_count,
                rng,
                distance_range=distance_range,
                microphone_array=None,
                use_config_microphone_array=spatial,
            )
            plan["source_ids"] = list(keys)
            payload = json.dumps(
                plan, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            plan["group_uid"] = "online-" + hashlib.sha256(payload).hexdigest()[:24]
            wave, facts = self.client.generate(plan)
        except OnlineRIRError:
            raise
        except Exception as exc:
            raise OnlineRIRError(f'online RIR scene failed: {exc}') from exc
        wave = np.asarray(wave, dtype=np.float32)
        expected = (int(source_count), len(self.config.microphone_positions)) if spatial else (int(source_count),)
        if spatial:
            if wave.ndim != 3 or wave.shape[:2] != expected:
                raise OnlineRIRError(f'spatial RIR provider returned {wave.shape}, expected [K,C,L] with K,C={expected}')
        elif wave.ndim != 2 or wave.shape[0] != source_count:
            raise OnlineRIRError(f'RIR provider returned {wave.shape}, expected [K,L]')
        if wave.shape[-1] < 1 or not np.isfinite(wave).all():
            raise OnlineRIRError('RIR provider returned non-finite or empty samples')
        peak = float(np.max(np.abs(wave)))
        scale = 1.0 / peak if normalize and peak > 0 else 1.0
        normalized = (wave * scale).astype(np.float32)
        facts = dict(facts) if isinstance(facts, Mapping) else {}
        provider_geometry = {
            key: facts[key]
            for key in (
                "array_profile", "array_topology", "coordinate_system", "spacing_m",
                "reference_channel", "num_channels", "mic_loc", "microphone_offsets",
            )
            if key in facts
        }
        provider_provenance = {
            **facts.get("provenance", {}),
            "provider_contract": "[K,C,L]" if spatial else "[K,L]",
        }
        result = SpatialRIRResult(
            normalized,
            provider_geometry,
            provider_provenance,
            str(facts.get("generator_version", "gpuRIR-online-v2")),
            details={
                **facts,
                "group_normalization_scale": scale,
                "raw_rir_sha256": facts.get("rir_sha256"),
                "rir_sha256": hashlib.sha256(normalized.tobytes()).hexdigest(),
            },
        )
        return result

    def sample_spatial_tracks(
        self,
        sample_rate,
        source_count,
        rng,
        *,
        source_uids=(),
        normalize=True,
        distance_range=None,
    ):
        """Request exact full/direct tracks from one online scene plan.

        Older clients/backends fail closed; no early/full approximation is
        fabricated here.
        """
        from core.data.simulation.rir_service import OnlineRIRError
        if self.config.microphone_positions is None:
            raise ValueError("full_direct online RIR requires a microphone array")
        try:
            keys = self._resolve_source_uids(source_count, source_uids)
            plan = sample_rir_scene(
                self.config, sample_rate, source_count, rng,
                distance_range=distance_range, microphone_array=None,
                use_config_microphone_array=True,
            )
            plan["source_ids"] = list(keys)
            payload = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            plan["group_uid"] = "online-" + hashlib.sha256(payload).hexdigest()[:24]
            if not hasattr(self.client, "generate_tracks"):
                raise OnlineRIRError("online client does not support explicit full_direct track requests")
            generated = self.client.generate_tracks(plan, track_kind="full_direct")
            if not isinstance(generated, Mapping) or set(generated) != {"full", "direct"}:
                raise OnlineRIRError("online full_direct response must contain exactly full and direct tracks")
            results = {}
            # 逐轨耗时不是 scene identity；保留在各自 details，公共 provenance
            # 只绑定共同请求，不能以 elapsed_sec 不同拒绝合法双轨。
            common_provenance = {
                'backend': 'gpuRIR', 'group_uid': plan['group_uid'],
                'source_order': list(keys), 'provider_contract': '[K,C,L]',
            }
            common_scale = 1.0
            if normalize:
                peak = max(float(np.max(np.abs(np.asarray(generated[name][0])))) for name in ("full", "direct"))
                common_scale = 1.0 / peak if peak > 0 else 1.0
            for name in ("full", "direct"):
                wave, facts = generated[name]
                wave = np.asarray(wave, dtype=np.float32)
                expected = (int(source_count), len(self.config.microphone_positions), wave.shape[-1])
                if wave.ndim != 3 or wave.shape[:2] != expected[:2] or wave.shape[-1] < 1 or not np.isfinite(wave).all():
                    raise OnlineRIRError(f"online {name} track shape/value contract failed: {wave.shape}")
                facts = dict(facts)
                facts["group_normalization_scale"] = common_scale
                facts["raw_rir_sha256"] = facts.get("rir_sha256")
                facts["rir_sha256"] = hashlib.sha256((wave * common_scale).astype("<f4").tobytes()).hexdigest()
                results[name] = SpatialRIRResult(
                    (wave * common_scale).astype(np.float32),
                    {key: facts[key] for key in ("array_profile", "coordinate_system", "reference_channel", "num_channels", "mic_loc") if key in facts},
                    common_provenance,
                    str(facts.get("generator_version", "gpuRIR-online-v2")), details=facts,
                )
            return SpatialRIRTrackSet(full=results["full"], direct=results["direct"])
        except OnlineRIRError:
            raise
        except Exception as exc:
            raise OnlineRIRError(f"online full_direct scene failed: {exc}") from exc

    def sample_moving_spatial_tracks(
        self, sample_rate, source_count, rng, *, num_samples,
        trajectory_points=4, source_uids=(), normalize=True,
        track_kinds=("full", "direct"),
    ):
        """按 ``OnlineRIRConfig.moving_source_speed_range_mps`` 规划运动轨迹并采样 RIR。

        支持 microphone array ``[P,K,C,L]`` 与 mono ``[P,K,L]`` bank。轨迹按
        source 生成独立水平速度方向，保留 room/timebase/separation fail-fast 合同。
        未请求 ``direct`` 时只生成 full 轨道，避免 mono moving 路径为未消费的
        direct 轨道重复执行一次 gpuRIR。
        """
        from core.data.simulation.rir_service import OnlineRIRError
        from core.data.simulation.gpurir_backend import validate_moving_rir_plan
        from core.data.simulation.trajectory import (
            TrajectoryTimebase,
            sample_source_trajectories_by_speed,
        )

        requested_tracks = tuple(track_kinds)
        if (
            not requested_tracks
            or "full" not in requested_tracks
            or any(name not in {"full", "direct", "early"} for name in requested_tracks)
        ):
            raise ValueError("moving tracks require full and only support full/direct/early")
        spatial = self.config.microphone_positions is not None
        if type(num_samples) is not int or num_samples < 1:
            raise ValueError('num_samples must be a positive exact integer')
        if type(trajectory_points) is not int or not 2 <= trajectory_points <= min(128, num_samples):
            raise ValueError('trajectory_points must be an integer in [2,min(128,num_samples)]')
        keys = self._resolve_source_uids(source_count, source_uids)
        plan = sample_rir_scene(
            self.config, sample_rate, source_count, rng,
            use_config_microphone_array=spatial,
        )
        start = np.asarray(plan['source_loc'], dtype=np.float64)
        clock = TrajectoryTimebase(sample_rate, num_samples, num_samples // trajectory_points, trajectory_points)
        trajectory, sampled_speeds = sample_source_trajectories_by_speed(
            start,
            room_dim_m=np.asarray(plan['room_dim'], dtype=np.float64),
            timebase=clock,
            speed_range_mps=self.config.moving_source_speed_range_mps,
            rng=rng,
            wall_margin_m=self.config.margin,
            min_separation_m=self.config.min_separation,
            source_height_range_m=self.config.source_height,
        )
        motion_duration_sec = float(clock.frame_times[-1] - clock.frame_times[0])
        plan.update(
            source_ids=list(keys),
            source_loc=trajectory.positions_m[0].tolist(),
            trajectory=trajectory.positions_m.tolist(),
            trajectory_timebase=clock.to_dict(),
            motion_speeds_mps=dict(zip(keys, sampled_speeds, strict=True)),
            moving_duration_sec=motion_duration_sec,
        )
        plan['group_uid'] = 'moving-' + hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()[:24]
        requested_track_kind = (
            'moving_full_direct' if 'direct' in requested_tracks else 'moving_full'
        )
        validate_moving_rir_plan(
            plan, track_count=2 if requested_track_kind == 'moving_full_direct' else 1
        )
        generated = self.client.generate_tracks(plan, track_kind=requested_track_kind)
        names = ('full',) if requested_track_kind == 'moving_full' else ('full', 'direct')
        if not isinstance(generated, Mapping) or set(generated) != set(names):
            raise OnlineRIRError(
                f'moving response must contain exactly {", ".join(names)}'
            )
        arrays, facts = {}, {}
        expected = (
            (trajectory_points, source_count, plan['num_samples'])
            if not spatial else
            (trajectory_points, source_count, len(self.config.microphone_positions), plan['num_samples'])
        )
        for name in names:
            wave, metadata = generated[name]
            wave = np.asarray(wave, dtype=np.float32)
            if wave.shape != expected or not np.isfinite(wave).all():
                raise OnlineRIRError(f'invalid moving {name} RIR shape/values')
            if metadata.get('group_uid') != plan['group_uid'] or metadata.get('track_kind') != name:
                raise OnlineRIRError(f'invalid moving {name} identity')
            arrays[name], facts[name] = wave, dict(metadata)
        peak = max(float(np.max(np.abs(wave))) for wave in arrays.values())
        scale = 1.0 / peak if normalize and peak > 0 else 1.0
        for name in arrays:
            arrays[name] = (arrays[name] * scale).astype(np.float32)
            facts[name]['group_normalization_scale'] = scale
        return MovingSpatialRIRTrackSet(
            arrays['full'], arrays.get('direct'), plan, facts
        )

    def sample_scene(
        self,
        sample_rate,
        source_count,
        rng,
        *,
        source_uids=(),
        normalize=True,
        distance_range=None,
    ):
        """Legacy mono scene contract; C=1 remains ``[K,L]``."""
        return self._sample(
            sample_rate,
            source_count,
            rng,
            source_uids=source_uids,
            normalize=normalize,
            distance_range=distance_range,
        )

    def sample_spatial_scene(
        self,
        sample_rate,
        source_count,
        rng,
        *,
        source_uids=(),
        normalize=True,
        distance_range=None,
    ):
        """Generate a fixed-array source×microphone bank with shape ``[K,C,L]``."""
        if self.config.microphone_positions is None:
            raise ValueError('spatial RIR generation requires resolved microphone geometry')
        return self._sample(
            sample_rate,
            source_count,
            rng,
            source_uids=source_uids,
            normalize=normalize,
            distance_range=distance_range,
            spatial=True,
        )

    def get_rirs(
        self,
        sample_rate,
        source_count,
        rng,
        *,
        source_uids=(),
        normalize=True,
        distance_range=None,
    ):
        if self.config.microphone_positions is None:
            return self.sample_scene(
                sample_rate,
                source_count,
                rng,
                source_uids=source_uids,
                normalize=normalize,
                distance_range=distance_range,
            )
        return self.sample_spatial_scene(
            sample_rate,
            source_count,
            rng,
            source_uids=source_uids,
            normalize=normalize,
            distance_range=distance_range,
        )


def validate_spatial_rir_provider(provider):
    """校验显式 provider seam，不从配置资源推导或替换 provider。"""
    if provider is None:
        raise ValueError("explicit spatial RIR provider must not be None")
    if not any(
        callable(getattr(provider, name, None))
        for name in (
            "sample_scene_tracks",
            "sample_moving_scene_tracks",
        )
    ):
        raise TypeError(
            "spatial_rir_provider must implement sample_scene_tracks or sample_moving_scene_tracks; "
            "legacy source-count/callable providers are not supported"
        )
    return provider


def build_spatial_rir_provider(
    cfg,
    *,
    client=None,
    provider=None,
    rir_scp=None,
    geometry=None,
):
    """创建统一 spatial provider；offline backend 使用 split-level ``rir_scp``。

    Dataset owner 只负责把 split 解析后的资源放入 ``cfg``，provider 的 policy、
    geometry 和 channel 合同均在这里收口。显式 provider 由 Base dataset 的
    provider seam 校验；offline backend 必须满足 split-level RIR manifest 合同。
    """
    if not getattr(cfg, "spatial_enabled", False):
        raise ValueError("spatial RIR provider requires spatial_enabled=True")

    backend = getattr(cfg, "rir_backend", "offline")
    if backend == "offline":
        if not rir_scp:
            raise ValueError("offline spatial RIR provider requires split-level rir_scp")
        if provider is not None:
            return validate_spatial_rir_provider(provider)
        from core.data.simulation.array_profiles import resolve_microphone_array

        resolved = resolve_microphone_array(
            array_profile=getattr(cfg, "array_profile", None),
            custom_geometry=getattr(cfg, "custom_geometry", None),
            reference_channel=getattr(cfg, "reference_channel", 0),
        )
        if resolved is None:
            raise ValueError(
                "offline spatial RIR provider requires array_profile or custom_geometry"
            )
        resolved_geometry = {
            "positions_m": [list(position) for position in resolved.positions_m],
            "reference_channel": resolved.reference_channel,
            "array_profile": None if resolved.profile.name == "custom" else resolved.profile.name,
            "spacing_m": resolved.profile.spacing_m,
            "coordinate_system": resolved.profile.coordinate_system,
        }
        resolved_geometry.update(dict(geometry or {}))
        return OfflineRIRProvider(rir_scp, geometry=resolved_geometry, require_scene_identity=True)
    if provider is not None:
        return validate_spatial_rir_provider(provider)
    if backend == "online":
        config = OnlineRIRConfig.from_simulation(cfg)
        if client is None:
            from core.data.simulation.rir_service import acquire_rir_client
            client = acquire_rir_client(config)
        return OnlineRIRProvider(config, client)
    raise ValueError(f"unsupported rir_backend: {backend!r}")


__all__ = [
    "OnlineRIRProvider",
    "SpatialRIRProvider",
    "SpatialRIRResult",
    "SpatialRIRTrackSet",
    "OfflineRIRProvider",
    "validate_spatial_rir_provider",
    "build_spatial_rir_provider",
]
