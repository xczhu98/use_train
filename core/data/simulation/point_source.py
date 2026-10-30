"""点源背景的 source×microphone RIR 渲染器。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import convolve

from core.data.simulation.noise_image import NoiseImage
from core.data.simulation.array_profiles import resolve_geometry_positions
from core.data.simulation.rir_provider import (
    MovingSceneRIRRequest,
    MovingSpatialRIRTrackSet,
    SpatialRIRResult,
    SpatialRIRTrackSet,
)
from core.data.simulation.moving_scene import render_moving_spatial_scene
from core.data.simulation.trajectory import TrajectoryTimebase


def _finite_mono(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite mono waveform")
    return np.ascontiguousarray(array, dtype=np.float32)


def _geometry_positions(geometry: Mapping[str, Any], *, channels: int) -> np.ndarray:
    positions = np.asarray(
        resolve_geometry_positions(
            geometry,
            expected_channels=channels,
            name="point source geometry",
            max_norm_m=None,
        ),
        dtype=np.float64,
    )
    return positions


def _normalize_rir_bank(rirs: np.ndarray, *, normalize: bool) -> tuple[np.ndarray, float | None]:
    value = np.asarray(rirs, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("point source RIR must be finite")
    if not normalize:
        return value, None
    peak = float(np.max(np.abs(value)))
    if peak <= 1e-8:
        return value, peak
    return value / np.float32(peak), peak


def _scene_uid(scene_plan: Any) -> str:
    identity = getattr(scene_plan, "identity", None)
    if not isinstance(identity, Mapping):
        raise ValueError("scene plan identity must be a mapping")
    for key in ("scene_uid", "scene_id"):
        value = identity.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("scene plan identity must contain scene_uid")


def _thaw_scene_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_scene_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw_scene_value(item) for item in value]
    return value


def _assert_positions_match(left: Any, right: Any, *, message: str) -> None:
    left_positions = np.asarray(
        resolve_geometry_positions(
            left,
            name="left geometry",
            max_norm_m=None,
        ), dtype=np.float64
    )
    right_positions = np.asarray(
        resolve_geometry_positions(
            right,
            name="right geometry",
            max_norm_m=None,
        ), dtype=np.float64
    )
    if left_positions.shape != right_positions.shape or not np.allclose(
        left_positions, right_positions, rtol=0.0, atol=1e-6
    ):
        raise ValueError(message)


def _validate_static_result_against_scene(
    result: SpatialRIRResult,
    scene_plan: Any,
    source_uids: tuple[str, ...],
) -> None:
    """把 static provider 返回值与请求 scene 的全部身份字段逐项对齐。"""

    canonical = result.canonical_rirs
    if canonical.shape[0] != len(source_uids):
        raise ValueError(
            "point source static RIR source axis does not match scene actor count"
        )
    expected_geometry = {
        "positions_m": np.asarray(scene_plan.array.positions_m, dtype=np.float64).tolist()
    }
    _assert_positions_match(
        result.geometry,
        expected_geometry,
        message="point source static RIR geometry does not match scene plan",
    )
    provenance = result.provenance
    if not isinstance(provenance, Mapping):
        raise TypeError("point source static RIR provenance must be a mapping")
    for key in ("actor_order", "source_order"):
        if key in provenance and tuple(provenance[key]) != source_uids:
            raise ValueError(
                f"point source static RIR {key} does not match scene actor order"
            )
    declared_order = provenance.get("actor_order", provenance.get("source_order"))
    if declared_order is None:
        raise ValueError(
            "point source static RIR provenance must declare actor_order or source_order"
        )
    declared_scene_uid = provenance.get("scene_uid", provenance.get("scene_id"))
    if declared_scene_uid is None or str(declared_scene_uid).strip() != _scene_uid(scene_plan):
        raise ValueError("point source static RIR scene identity does not match scene plan")
    scene_payload = scene_plan.to_dict()
    declared_room = provenance.get("room")
    if declared_room is None or _thaw_scene_value(declared_room) != _thaw_scene_value(scene_payload["room"]):
        raise ValueError("point source static RIR room does not match scene plan")


def _timebase_from_mapping(raw: Mapping[str, Any], *, name: str) -> TrajectoryTimebase:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} must be a mapping")
    fields = ("sample_rate", "num_samples", "hop_samples", "frame_count")
    missing = [field for field in fields if field not in raw]
    if missing:
        raise ValueError(f"{name} is missing fields: {missing!r}")
    invalid = [field for field in fields if type(raw[field]) is not int]
    if "start_sample" in raw and type(raw["start_sample"]) is not int:
        invalid.append("start_sample")
    if invalid:
        raise ValueError(f"{name} fields must be exact integers: {invalid!r}")
    return TrajectoryTimebase(
        raw["sample_rate"],
        raw["num_samples"],
        raw["hop_samples"],
        raw["frame_count"],
        raw.get("start_sample", 0),
    )


def _moving_source_count(full: np.ndarray) -> int:
    if full.ndim not in (3, 4):
        raise ValueError("moving track set full RIR must have shape [P,K,L] or [P,K,C,L]")
    return int(full.shape[1])


def _validate_moving_track_set_against_scene(
    track_set: MovingSpatialRIRTrackSet,
    scene_plan: Any,
    source_uids: tuple[str, ...],
) -> None:
    """校验 moving provider 没有返回另一条轨迹或另一间房的结果。"""

    full = np.asarray(track_set.full, dtype=np.float32)
    if _moving_source_count(full) != len(source_uids):
        raise ValueError(
            "moving point source RIR source axis does not match scene actor count"
        )
    plan = track_set.plan
    if not isinstance(plan, Mapping):
        raise TypeError("moving track set plan must be a mapping")
    declared_source_ids = plan.get("source_ids")
    if not isinstance(declared_source_ids, (list, tuple)):
        raise ValueError("moving track set plan must declare source_ids")
    if tuple(str(uid) for uid in declared_source_ids) != source_uids:
        raise ValueError("moving track set source_ids do not match scene actor order")
    expected_timebase = scene_plan.timebase
    actual_timebase = _timebase_from_mapping(
        plan.get("trajectory_timebase"),
        name="moving track set trajectory_timebase",
    )
    if actual_timebase != expected_timebase:
        raise ValueError("moving track set timebase does not match scene plan")
    waveform_num_samples = plan.get("waveform_num_samples")
    if (
        type(waveform_num_samples) is not int
        or waveform_num_samples < 1
        or waveform_num_samples != int(scene_plan.num_samples)
    ):
        raise ValueError("moving track set waveform length does not match scene plan")
    expected_geometry = {
        "positions_m": np.asarray(scene_plan.array.positions_m, dtype=np.float64).tolist()
    }
    _assert_positions_match(
        plan.get("array", plan.get("geometry", plan.get("mic_loc"))),
        expected_geometry,
        message="moving point source RIR geometry does not match scene plan",
    )
    scene_payload = scene_plan.to_dict()
    declared_room = plan.get("room")
    if declared_room is None or _thaw_scene_value(declared_room) != _thaw_scene_value(scene_payload["room"]):
        raise ValueError("moving point source RIR room does not match scene plan")
    declared_scene_uid = plan.get("scene_uid", plan.get("group_uid"))
    if declared_scene_uid is None or str(declared_scene_uid).strip() != _scene_uid(scene_plan):
        raise ValueError("moving point source RIR scene identity does not match scene plan")

    expected_trajectories = {
        actor.actor_uid: np.asarray(actor.trajectory.positions_m, dtype=np.float32)
        for actor in scene_plan.actors
    }
    declared_trajectories = plan.get("actor_trajectories")
    if isinstance(declared_trajectories, Mapping):
        if tuple(declared_trajectories) != source_uids:
            raise ValueError("moving track actor_trajectories order does not match scene")
        for uid in source_uids:
            if not np.array_equal(
                np.asarray(declared_trajectories[uid], dtype=np.float32),
                expected_trajectories[uid],
            ):
                raise ValueError(
                    f"moving track trajectory does not match scene actor {uid!r}"
                )
    elif "trajectory" in plan:
        expected_trajectory = np.concatenate(
            [expected_trajectories[uid] for uid in source_uids], axis=1
        )
        if not np.array_equal(
            np.asarray(plan["trajectory"], dtype=np.float32), expected_trajectory
        ):
            raise ValueError("moving track trajectory does not match scene plan")
    else:
        raise ValueError(
            "moving track set must declare actor_trajectories or trajectory"
        )


@dataclass(frozen=True, slots=True)
class PointSourceRIRProvider:
    """使用一组显式 source×microphone RIR 渲染点源背景。

    本 provider 只支持静态场景；moving 点源由同文件的
    ``MovingPointSourceRIRProvider`` 按共享 ``TrajectoryTimebase`` 处理。它不负责选择
    随机噪声，也不从 geometry 猜 RIR。调用方必须先用
    offline typed scene provider 或 online scene provider 解析出单个 actor 的
    RIR，再通过 ``from_scene_plan`` 或 ``from_result`` 构造本对象。
    """

    rirs: np.ndarray
    geometry: Mapping[str, Any]
    sample_rate: int
    source_uid: str
    generator_version: str = "point-source-rir-v1"
    provenance: Mapping[str, Any] | None = None
    normalize_rir: bool = True

    def __post_init__(self) -> None:
        value = np.asarray(self.rirs, dtype=np.float32)
        if value.ndim == 3:
            if value.shape[0] != 1:
                raise ValueError("point source RIR must contain exactly one source")
            value = value[0]
        if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
            raise ValueError("point source RIR must have shape [C,L]")
        if not np.isfinite(value).all():
            raise ValueError("point source RIR must be finite")
        positions = _geometry_positions(self.geometry, channels=value.shape[0])
        if type(self.sample_rate) is not int or self.sample_rate < 1:
            raise ValueError("point source sample_rate must be a positive exact int")
        source_uid = str(self.source_uid).strip()
        if not source_uid:
            raise ValueError("point source source_uid must be non-empty")
        if type(self.normalize_rir) is not bool:
            raise TypeError("normalize_rir must be bool")
        if not isinstance(self.generator_version, str) or not self.generator_version.strip():
            raise ValueError("generator_version must be non-empty")
        value = np.ascontiguousarray(value, dtype="<f4")
        value.setflags(write=False)
        object.__setattr__(self, "rirs", value)
        geometry = dict(self.geometry)
        geometry["positions_m"] = positions.tolist()
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "source_uid", source_uid)
        object.__setattr__(self, "provenance", dict(self.provenance or {}))

    @classmethod
    def from_result(
        cls,
        result: SpatialRIRResult,
        *,
        source_uid: str,
        sample_rate: int,
        normalize_rir: bool = True,
    ) -> "PointSourceRIRProvider":
        """从单 source ``SpatialRIRResult`` 构造点源 provider。"""

        if not isinstance(result, SpatialRIRResult):
            raise TypeError("result must be a SpatialRIRResult")
        value = result.canonical_rirs
        if value.shape[0] != 1:
            raise ValueError("point source result must contain exactly one source")
        return cls(
            value,
            result.geometry,
            sample_rate,
            source_uid,
            generator_version=result.generator_version,
            provenance={**result.facts, "source_uid": str(source_uid).strip()},
            normalize_rir=normalize_rir,
        )

    def validate_geometry(self, geometry: Any) -> None:
        """校验 provider 与当前 observation scene 使用同一麦克风阵列。"""

        positions = np.asarray(
            resolve_geometry_positions(
                geometry,
                expected_channels=self.rirs.shape[0],
                name="observation scene geometry",
                max_norm_m=None,
            ),
            dtype=np.float64,
        )
        expected = np.asarray(self.geometry["positions_m"], dtype=np.float64)
        if positions.shape != expected.shape or not np.allclose(
            positions, expected, rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                "point source RIR geometry does not match observation scene array"
            )

    @classmethod
    def from_scene_plan(
        cls,
        provider: Any,
        scene_plan: Any,
        *,
        source_uid: str,
        sample_rate: int,
        normalize_rir: bool = True,
    ) -> "PointSourceRIRProvider":
        """从完整静态 scene plan 解析指定 actor 的 RIR。"""

        if not callable(getattr(provider, "sample_scene_tracks", None)):
            raise TypeError("point source provider requires sample_scene_tracks")
        source_uids = tuple(actor.actor_uid for actor in scene_plan.actors)
        if source_uid not in source_uids:
            raise ValueError(f"point source actor {source_uid!r} is absent from scene plan")
        if scene_plan.timebase is not None or any(
            actor.trajectory is not None for actor in scene_plan.actors
        ):
            raise ValueError("point source provider currently requires a static scene plan")
        if scene_plan.room is None or scene_plan.array is None:
            raise ValueError("point source static scene plan must resolve room and array")
        tracks = provider.sample_scene_tracks(
            scene_plan, normalize=False, track_kinds=("full",)
        )
        if isinstance(tracks, SpatialRIRTrackSet):
            result = tracks.full
        elif isinstance(tracks, SpatialRIRResult):
            result = tracks
        else:
            raise TypeError("point source scene provider must return SpatialRIRResult or SpatialRIRTrackSet")
        _validate_static_result_against_scene(result, scene_plan, source_uids)
        index = source_uids.index(source_uid)
        provenance = dict(result.provenance)
        for key in ("actor_order", "source_order"):
            if key in provenance:
                provenance[key] = [source_uid]
        actor_positions = provenance.get("actor_positions_m")
        if isinstance(actor_positions, Mapping) and source_uid in actor_positions:
            provenance["actor_positions_m"] = {
                source_uid: actor_positions[source_uid]
            }
        provenance.update({"source_uid": source_uid, "source_index": index})
        selected = SpatialRIRResult(
            result.canonical_rirs[index:index + 1],
            result.geometry,
            provenance,
            result.generator_version,
            details=result.details,
        )
        return cls.from_result(
            selected,
            source_uid=source_uid,
            sample_rate=sample_rate,
            normalize_rir=normalize_rir,
        )

    def render(self, waveform: np.ndarray, *, sample_rate: int) -> NoiseImage:
        """将 mono 点源卷积成与 RIR 通道数一致的 observation image。"""

        if type(sample_rate) is not int or sample_rate != self.sample_rate:
            raise ValueError(
                f"point source sample rate mismatch: provider={self.sample_rate}, input={sample_rate}"
            )
        source = _finite_mono(waveform, name="point source waveform")
        rir, peak = _normalize_rir_bank(
            self.rirs, normalize=self.normalize_rir
        )
        image = np.stack(
            [
                convolve(source, channel, mode="full")[: source.size]
                for channel in rir
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        facts = {
            **dict(self.provenance or {}),
            "projection": "point_source_rir",
            "rir_applied": True,
            "source_uid": self.source_uid,
            "sample_rate": self.sample_rate,
            "geometry": dict(self.geometry),
            "generator_version": self.generator_version,
            "normalization_peak": peak,
            "input_shape": [int(source.size)],
            "output_shape": [int(image.shape[0]), int(image.shape[1])],
        }
        return NoiseImage(image, facts)


@dataclass(frozen=True, slots=True)
class MovingPointSourceRIRProvider:
    """按共享 ``TrajectoryTimebase`` 渲染时变点源背景。

    ``rirs`` 的 canonical 形状为 ``[P,C,L]``，其中 P 与
    ``timebase.frame_count`` 一致。每个轨道从对应 frame sample 开始作用，
    跨 frame 的卷积尾部由 moving scene overlap-add renderer 保留。
    """

    rirs: np.ndarray
    geometry: Mapping[str, Any]
    timebase: TrajectoryTimebase
    sample_rate: int
    source_uid: str
    generator_version: str = "point-source-moving-rir-v1"
    provenance: Mapping[str, Any] | None = None
    normalize_rir: bool = True

    def __post_init__(self) -> None:
        value = np.asarray(self.rirs, dtype=np.float32)
        if value.ndim == 4:
            if value.shape[1] != 1:
                raise ValueError(
                    "moving point source RIR must contain exactly one source"
                )
            value = value[:, 0]
        elif value.ndim == 3:
            pass
        elif value.ndim == 2:
            value = value[:, None, :]
        else:
            raise ValueError(
                "moving point source RIR must have shape [P,C,L], [P,1,C,L], or [P,L]"
            )
        if value.shape[0] < 1 or value.shape[1] < 1 or value.shape[2] < 1:
            raise ValueError("moving point source RIR must have positive [P,C,L] axes")
        if not np.isfinite(value).all():
            raise ValueError("moving point source RIR must be finite")
        if not isinstance(self.timebase, TrajectoryTimebase):
            raise TypeError("moving point source timebase must be a TrajectoryTimebase")
        if self.timebase.sample_rate != self.sample_rate:
            raise ValueError(
                "moving point source timebase sample_rate must match provider sample_rate"
            )
        if self.timebase.frame_count != value.shape[0]:
            raise ValueError(
                "moving point source RIR frame count must match timebase"
            )
        if self.timebase.frame_count < 2:
            raise ValueError(
                "moving point source provider requires at least two trajectory frames"
            )
        if self.timebase.start_sample != 0:
            raise ValueError(
                "moving point source provider requires timebase.start_sample=0"
            )
        positions = _geometry_positions(self.geometry, channels=value.shape[1])
        if type(self.sample_rate) is not int or self.sample_rate < 1:
            raise ValueError("moving point source sample_rate must be a positive exact int")
        source_uid = str(self.source_uid).strip()
        if not source_uid:
            raise ValueError("moving point source source_uid must be non-empty")
        if type(self.normalize_rir) is not bool:
            raise TypeError("normalize_rir must be bool")
        if not isinstance(self.generator_version, str) or not self.generator_version.strip():
            raise ValueError("generator_version must be non-empty")
        value = np.ascontiguousarray(value, dtype="<f4")
        value.setflags(write=False)
        object.__setattr__(self, "rirs", value)
        geometry = dict(self.geometry)
        geometry["positions_m"] = positions.tolist()
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "source_uid", source_uid)
        object.__setattr__(self, "provenance", dict(self.provenance or {}))

    @classmethod
    def from_track_set(
        cls,
        track_set: MovingSpatialRIRTrackSet,
        *,
        source_uid: str,
        source_index: int,
        sample_rate: int,
        normalize_rir: bool = True,
    ) -> "MovingPointSourceRIRProvider":
        """从共享 moving scene track set 选择一个 actor 的 full RIR。

        ``plan.num_samples`` 是每帧 RIR kernel 长度；波形时间轴必须由
        ``plan.waveform_num_samples`` 显式声明并与 ``trajectory_timebase`` 一致。
        """

        if not isinstance(track_set, MovingSpatialRIRTrackSet):
            raise TypeError("track_set must be a MovingSpatialRIRTrackSet")
        if type(source_index) is not int or source_index < 0:
            raise ValueError("source_index must be a non-negative exact int")
        full = np.asarray(track_set.full, dtype=np.float32)
        if full.ndim == 4:
            if source_index >= full.shape[1]:
                raise IndexError("source_index exceeds moving RIR source axis")
            selected = full[:, source_index]
        elif full.ndim == 3:
            if source_index >= full.shape[1]:
                raise IndexError("source_index exceeds moving mono RIR source axis")
            selected = full[:, source_index, :]
        else:
            raise ValueError("moving track set full RIR has unsupported shape")
        raw_timebase = track_set.plan.get("trajectory_timebase")
        timebase = _timebase_from_mapping(raw_timebase, name="moving track set trajectory_timebase")
        plan_sample_rate = track_set.plan.get("sample_rate")
        if plan_sample_rate is not None and int(plan_sample_rate) != int(sample_rate):
            raise ValueError("moving track set sample_rate does not match requested sample_rate")
        rir_num_samples = track_set.plan.get("num_samples")
        if type(rir_num_samples) is not int or rir_num_samples < 1:
            raise ValueError("moving track set plan must contain positive RIR num_samples")
        if full.shape[-1] != rir_num_samples:
            raise ValueError(
                "moving track set RIR length does not match plan.num_samples"
            )
        waveform_num_samples = track_set.plan.get("waveform_num_samples")
        if (
            type(waveform_num_samples) is not int
            or waveform_num_samples < 1
            or waveform_num_samples != timebase.num_samples
        ):
            raise ValueError(
                "moving track set plan must declare waveform_num_samples matching "
                "trajectory timebase"
            )
        raw_geometry = track_set.plan.get("array")
        if isinstance(raw_geometry, Mapping):
            geometry = dict(raw_geometry)
        else:
            mic_loc = np.asarray(track_set.plan.get("mic_loc"), dtype=np.float64)
            if mic_loc.shape == (3,):
                mic_loc = mic_loc[None, :]
            geometry = {"positions_m": mic_loc.tolist()}
        source_ids = track_set.plan.get("source_ids")
        if not isinstance(source_ids, (list, tuple)) or len(source_ids) != _moving_source_count(full):
            raise ValueError(
                "moving track set plan must declare source_ids for every source"
            )
        source_ids = tuple(str(uid).strip() for uid in source_ids)
        if any(not uid for uid in source_ids) or len(set(source_ids)) != len(source_ids):
            raise ValueError("moving track set source_ids must be unique non-empty strings")
        if source_index >= len(source_ids):
            raise IndexError("source_index exceeds moving source_ids")
        declared_uid = source_ids[source_index]
        if declared_uid != str(source_uid).strip():
            raise ValueError(
                "moving track source_uid does not match requested source_index"
            )
        provenance = {
            **track_set.facts,
            "source_uid": str(source_uid).strip(),
            "source_index": source_index,
        }
        if not isinstance(track_set.track_facts, Mapping):
            raise TypeError("moving track_facts must be a mapping")
        full_facts = track_set.track_facts.get("full", {})
        if not isinstance(full_facts, Mapping):
            raise TypeError("moving track_facts['full'] must be a mapping")
        return cls(
            selected,
            geometry,
            timebase,
            sample_rate,
            source_uid,
            generator_version=str(
                full_facts.get("generator_version", "point-source-moving-rir-v1")
            ),
            provenance=provenance,
            normalize_rir=normalize_rir,
        )

    @classmethod
    def from_scene_plan(
        cls,
        provider: Any,
        scene_plan: Any,
        *,
        source_uid: str,
        sample_rate: int,
        normalize_rir: bool = True,
    ) -> "MovingPointSourceRIRProvider":
        """从 shared moving scene plan 解析指定 actor 的 moving RIR。"""

        if scene_plan.timebase is None:
            raise ValueError("moving point source provider requires a moving scene plan")
        source_uids = tuple(actor.actor_uid for actor in scene_plan.actors)
        if source_uid not in source_uids:
            raise ValueError(f"moving point source actor {source_uid!r} is absent")
        if any(actor.trajectory is None for actor in scene_plan.actors):
            raise ValueError("moving scene plan must provide every actor trajectory")
        scene_payload = scene_plan.to_dict()
        room = scene_payload.get("room")
        if not isinstance(room, Mapping) or not room:
            raise ValueError("moving scene plan must contain room identity")
        identity = scene_payload.get("identity", {})
        scene_uid = next(
            (
                str(identity[key]).strip()
                for key in ("scene_uid", "scene_id")
                if isinstance(identity, Mapping)
                and isinstance(identity.get(key), str)
                and identity[key].strip()
            ),
            None,
        )
        if scene_uid is None:
            raise ValueError("moving scene plan identity must contain scene_uid")
        request = MovingSceneRIRRequest(
            sample_rate=int(sample_rate),
            num_samples=int(scene_plan.num_samples),
            actor_trajectories={
                actor.actor_uid: actor.trajectory for actor in scene_plan.actors
            },
            room=dict(room),
            array={
                "positions_m": np.asarray(
                    scene_plan.array.positions_m, dtype=np.float64
                ).tolist(),
                "reference_channel": int(scene_plan.array.reference_channel),
            },
            timebase=scene_plan.timebase,
            source_uids=source_uids,
            scene_uid=scene_uid,
            actor_roles=tuple(actor.role for actor in scene_plan.actors),
        )
        sampler = getattr(provider, "sample_moving_scene_tracks", None)
        if not callable(sampler):
            raise TypeError(
                "moving point source provider requires sample_moving_scene_tracks"
            )
        tracks = sampler(request, normalize=False)
        _validate_moving_track_set_against_scene(tracks, scene_plan, source_uids)
        return cls.from_track_set(
            tracks,
            source_uid=source_uid,
            source_index=source_uids.index(source_uid),
            sample_rate=sample_rate,
            normalize_rir=normalize_rir,
        )

    def validate_geometry(self, geometry: Any) -> None:
        positions = np.asarray(
            resolve_geometry_positions(
                geometry,
                expected_channels=self.rirs.shape[1],
                name="observation scene geometry",
                max_norm_m=None,
            ),
            dtype=np.float64,
        )
        expected = np.asarray(self.geometry["positions_m"], dtype=np.float64)
        if positions.shape != expected.shape or not np.allclose(
            positions, expected, rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                "moving point source RIR geometry does not match observation scene array"
            )

    def render(self, waveform: np.ndarray, *, sample_rate: int) -> NoiseImage:
        """按逐帧 RIR 与共享 timebase 生成 ``[C,T]`` 点源 image。"""

        if type(sample_rate) is not int or sample_rate != self.sample_rate:
            raise ValueError(
                "moving point source sample rate does not match provider"
            )
        source = _finite_mono(waveform, name="moving point source waveform")
        if source.size != self.timebase.num_samples:
            raise ValueError(
                "moving point source waveform length must match timebase.num_samples"
            )
        rir, peak = _normalize_rir_bank(
            self.rirs, normalize=self.normalize_rir
        )
        rendered = render_moving_spatial_scene(
            source[None, :],
            rir[:, None, :, :],
            direct_rirs=rir[:, None, :, :],
            frame_samples=self.timebase.frame_samples,
            early_boundary=rir.shape[-1],
            reference_channel=0,
            num_samples=self.timebase.num_samples,
        )
        image = np.asarray(rendered.mixture, dtype=np.float32)
        facts = {
            **dict(self.provenance or {}),
            "projection": "point_source_moving_rir",
            "rir_applied": True,
            "moving": True,
            "source_uid": self.source_uid,
            "sample_rate": self.sample_rate,
            "geometry": dict(self.geometry),
            "timebase": self.timebase.to_dict(),
            "generator_version": self.generator_version,
            "normalization_peak": peak,
            "input_shape": [int(source.size)],
            "output_shape": [int(image.shape[0]), int(image.shape[1])],
        }
        return NoiseImage(image, facts)


__all__ = ["MovingPointSourceRIRProvider", "PointSourceRIRProvider"]
