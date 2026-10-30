"""统一的 spatial scene deep Module。

调用方只需要提交 ``SpatialSceneRequest``，本模块隐藏 static/moving provider、
full/direct/early track 选择、trajectory timebase、背景合并和 waveform renderer。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from core.data.simulation.moving_scene import render_moving_spatial_scene
from core.data.simulation.rir_provider import (
    MovingSceneRIRRequest,
    MovingSpatialRIRTrackSet,
    SpatialRIRResult,
    SpatialRIRTrackSet,
)
from core.data.simulation.trajectory import SourceTrajectory, TrajectoryTimebase
from core.data.spatial import (
    SpatialRenderResult,
    derive_rts_rir_bank,
    render_spatial_scene,
)
from core.data.acoustic_scene_plan import AcousticScenePlan, _thaw_json

TrackKind = Literal["full", "direct", "early"]


def _finite_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {array.shape}")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return np.ascontiguousarray(array, dtype=np.float32)


def _declared_track_kind(track: SpatialRIRResult, fallback: TrackKind) -> str:
    value = track.provenance.get("track_kind")
    if value is None:
        value = track.details.get("track_kind")
    return fallback if value is None else str(value)


def _track_set_identity(track_set: SpatialRIRTrackSet | MovingSpatialRIRTrackSet) -> dict[str, Any]:
    if isinstance(track_set, SpatialRIRTrackSet):
        tracks = {}
        for track_kind, track in (
            ("full", track_set.full),
            ("direct", track_set.direct),
            ("early", track_set.early),
        ):
            if track is None:
                tracks[track_kind] = None
                continue
            # content_hash 仅描述该轨道的波形内容；full/direct/early 的
            # 合法关系由 track_kind 和共同 scene provenance 决定，不能要求 hash 相同。
            tracks[track_kind] = {
                "track_kind": _declared_track_kind(track, track_kind),
                "content_hash": track.content_hash,
                "provenance": dict(track.provenance),
            }
        return {"track_set": tracks, "facts": track_set.facts}
    return {"facts": track_set.facts, "track_set": track_set.facts.get("track_set", {})}


@dataclass(frozen=True, slots=True)
class SpatialSceneRequest:
    """SpatialSceneRenderer 的唯一输入合同。

    ``sources`` 为 ``[K,N]``，``provider`` 必须提供静态或移动 spatial track
    adapter。``trajectory`` 只用于 moving 请求，并且必须与 provider 返回的
    trajectory/timebase 一致。``background`` 若提供，必须是最终 observation
    侧的 ``[C,T]`` 数组；它不会改变 full/direct/early source tracks。
    """

    sources: np.ndarray
    sample_rate: int
    provider: Any
    source_uids: tuple[str, ...] = ()
    mode: Literal["static", "moving"] = "static"
    track_kinds: tuple[TrackKind, ...] = ("full", "direct", "early")
    early_boundary: int = 1
    rts_target_t60_sec: float | None = None
    rts_time_after_max_sec: float = 0.002
    reference_channel: int = 0
    rng: Any = None
    num_samples: int | None = None
    trajectory: SourceTrajectory | None = None
    trajectory_points: int = 4
    max_displacement_m: float = 0.25
    background: np.ndarray | None = None
    normalize: bool = True
    scene_plan: AcousticScenePlan | None = None

    def __post_init__(self) -> None:
        sources = _finite_array(self.sources, name="sources", ndim=2)
        if type(self.sample_rate) is not int or self.sample_rate < 1:
            raise ValueError("sample_rate must be a positive exact int")
        if self.mode not in {"static", "moving"}:
            raise ValueError("mode must be 'static' or 'moving'")
        if not self.source_uids:
            source_uids = tuple(f"source/{index}" for index in range(sources.shape[0]))
        else:
            source_uids = tuple(self.source_uids)
        if len(source_uids) != sources.shape[0] or len(set(source_uids)) != len(source_uids):
            raise ValueError("source_uids must be unique and match source count")
        if any(not isinstance(uid, str) or not uid.strip() for uid in source_uids):
            raise ValueError("source_uids must contain non-empty strings")
        if not isinstance(self.track_kinds, tuple) or not self.track_kinds:
            raise ValueError("track_kinds must be a non-empty tuple")
        if any(kind not in {"full", "direct", "early"} for kind in self.track_kinds):
            raise ValueError("track_kinds must contain only full, direct, early")
        if len(set(self.track_kinds)) != len(self.track_kinds):
            raise ValueError("track_kinds must not contain duplicates")
        if type(self.early_boundary) is not int or self.early_boundary < 0:
            raise ValueError("early_boundary must be a non-negative exact int")
        if self.rts_target_t60_sec is not None and (
            isinstance(self.rts_target_t60_sec, bool)
            or not isinstance(self.rts_target_t60_sec, (int, float, np.number))
            or not np.isfinite(self.rts_target_t60_sec)
            or self.rts_target_t60_sec <= 0.0
        ):
            raise ValueError("rts_target_t60_sec must be a positive finite number or None")
        if (
            isinstance(self.rts_time_after_max_sec, bool)
            or not isinstance(self.rts_time_after_max_sec, (int, float, np.number))
            or not np.isfinite(self.rts_time_after_max_sec)
            or self.rts_time_after_max_sec < 0.0
        ):
            raise ValueError("rts_time_after_max_sec must be finite and non-negative")
        if type(self.reference_channel) is not int or self.reference_channel < 0:
            raise ValueError("reference_channel must be a non-negative exact int")
        if self.num_samples is not None and (type(self.num_samples) is not int or self.num_samples < sources.shape[1]):
            raise ValueError("num_samples must be an exact int >= source length")
        if self.mode == "static" and self.trajectory is not None:
            raise ValueError("static scene must not provide a trajectory")
        if self.mode == "moving":
            if self.trajectory is not None and self.trajectory.source_count != sources.shape[0]:
                raise ValueError("trajectory source count must match sources")
            if type(self.trajectory_points) is not int or self.trajectory_points < 2:
                raise ValueError("trajectory_points must be an exact int >= 2")
            if not np.isfinite(self.max_displacement_m) or self.max_displacement_m <= 0:
                raise ValueError("max_displacement_m must be positive and finite")
        background = None if self.background is None else _finite_array(self.background, name="background", ndim=2)
        if self.scene_plan is None:
            raise ValueError("SpatialSceneRequest requires a complete scene_plan")
        if not isinstance(self.scene_plan, AcousticScenePlan):
            raise TypeError("scene_plan must be an AcousticScenePlan or None")
        if self.scene_plan.array is None:
            raise ValueError("scene_plan must contain resolved microphone array")
        if not any(
            isinstance(self.scene_plan.identity.get(key), str)
            and self.scene_plan.identity.get(key).strip()
            for key in ("scene_uid", "scene_id")
        ):
            raise ValueError("scene_plan identity must contain non-empty scene_uid")
        if self.scene_plan.sample_rate != self.sample_rate:
            raise ValueError("scene_plan sample_rate must match request")
        if self.scene_plan.room is None:
            raise ValueError("scene_plan must contain room identity")
        if self.scene_plan.num_samples != (self.num_samples or sources.shape[1]):
            raise ValueError("scene_plan num_samples must match output waveform length")
        if self.scene_plan.array.reference_channel != self.reference_channel:
            raise ValueError("scene_plan reference channel must match request")
        actor_order = tuple(actor.actor_uid for actor in self.scene_plan.actors)
        if actor_order != source_uids:
            raise ValueError("scene_plan actor order must match request source_uids")
        if self.mode == "moving":
            if self.scene_plan.timebase is None:
                raise ValueError("moving scene_plan must contain a shared timebase")
            actor_trajectories = tuple(actor.trajectory for actor in self.scene_plan.actors)
            if any(trajectory is None for trajectory in actor_trajectories):
                raise ValueError("moving scene_plan actors must all contain trajectories")
            if self.trajectory is not None and self.scene_plan.timebase != self.trajectory.timebase:
                raise ValueError("scene_plan timebase must match request trajectory")
        elif any(actor.position_m is None or actor.trajectory is not None for actor in self.scene_plan.actors):
            raise ValueError("static scene_plan must contain static actor positions")
        if background is not None and background.shape[1] < sources.shape[1]:
            raise ValueError("background must cover the source waveform")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "source_uids", source_uids)
        object.__setattr__(self, "background", background)


@dataclass(frozen=True, slots=True)
class SpatialSceneResult:
    """统一 static/moving scene 输出合同，所有数组均为不可写 float32。"""

    observation: np.ndarray
    mixture: np.ndarray
    full_tracks: np.ndarray
    direct_tracks: np.ndarray | None
    early_tracks: np.ndarray
    background: np.ndarray
    geometry: Mapping[str, Any]
    provenance: Mapping[str, Any]
    source_uids: tuple[str, ...]
    trajectory: SourceTrajectory | None = None
    timebase: TrajectoryTimebase | None = None
    rts_tracks: np.ndarray | None = None

    def __post_init__(self) -> None:
        observation = _finite_array(self.observation, name="observation", ndim=2)
        mixture = _finite_array(self.mixture, name="mixture", ndim=2)
        full = _finite_array(self.full_tracks, name="full_tracks", ndim=3)
        early = _finite_array(self.early_tracks, name="early_tracks", ndim=3)
        direct = None if self.direct_tracks is None else _finite_array(self.direct_tracks, name="direct_tracks", ndim=3)
        rts = None if self.rts_tracks is None else _finite_array(self.rts_tracks, name="rts_tracks", ndim=3)
        background = _finite_array(self.background, name="background", ndim=2)
        if full.shape != early.shape or (direct is not None and direct.shape != full.shape):
            raise ValueError("full/direct/early tracks must share [K,C,T] shape")
        if rts is not None and rts.shape != full.shape:
            raise ValueError("rts_tracks must match full_tracks [K,C,T] shape")
        expected = (full.shape[1], full.shape[2])
        if mixture.shape != expected or observation.shape != expected or background.shape != expected:
            raise ValueError("observation, mixture, and background must have shape [C,T]")
        if not isinstance(self.geometry, Mapping) or not isinstance(self.provenance, Mapping):
            raise TypeError("geometry and provenance must be mappings")
        positions = self.geometry.get("positions_m")
        if positions is not None:
            positions = np.asarray(positions)
            if positions.ndim != 2 or positions.shape != (full.shape[1], 3):
                raise ValueError("geometry positions_m must have shape [C,3] matching tracks")
        source_uids = tuple(self.source_uids)
        if len(source_uids) != full.shape[0] or len(set(source_uids)) != len(source_uids):
            raise ValueError("source_uids must be unique and match source axis")
        if any(not isinstance(uid, str) or not uid.strip() for uid in source_uids):
            raise ValueError("source_uids must contain non-empty strings")
        source_order = self.provenance.get("source_order")
        if source_order is not None:
            if tuple(source_order) != source_uids:
                raise ValueError("provenance source_order must match source_uids")
        if self.trajectory is not None:
            if self.timebase is None or self.trajectory.timebase != self.timebase:
                raise ValueError("trajectory and timebase must share the same contract")
            frame_samples = self.provenance.get("frame_samples")
            if frame_samples is not None and list(frame_samples) != self.timebase.frame_samples.tolist():
                raise ValueError("provenance frame_samples must match trajectory timebase")
        elif self.timebase is not None:
            raise ValueError("timebase requires a trajectory")
        for name, value in (("observation", observation), ("mixture", mixture), ("full_tracks", full),
                            ("direct_tracks", direct), ("early_tracks", early),
                            ("background", background), ("rts_tracks", rts)):
            if value is not None:
                value = np.ascontiguousarray(value, dtype=np.float32)
                value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source_uids", source_uids)
        object.__setattr__(self, "geometry", dict(self.geometry))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def has_direct_path(self) -> bool:
        return self.direct_tracks is not None


class SpatialSceneRenderer:
    """隐藏 provider、RIR track、trajectory 和 waveform renderer 的 deep Module。"""

    def render(self, request: SpatialSceneRequest) -> SpatialSceneResult:
        if not isinstance(request, SpatialSceneRequest):
            raise TypeError("render expects a SpatialSceneRequest")
        if request.mode == "moving":
            track_set = self._sample_moving(request)
            rendered, trajectory, timebase = self._render_moving(request, track_set)
        else:
            track_set = self._sample_static(request)
            rendered, trajectory, timebase = self._render_static(request, track_set)
        self._validate_rendered(request, track_set, rendered)
        background = self._resolve_background(request, rendered.mixture)
        observation = rendered.mixture + background
        provenance = {
            **rendered.provenance,
            "source_order": list(request.source_uids),
            "sample_rate": request.sample_rate,
            "mode": request.mode,
            "track_identity": _track_set_identity(track_set),
        }
        if request.scene_plan is not None:
            self._validate_scene_plan_geometry(request.scene_plan, rendered.geometry)
            self._validate_scene_plan_provenance(request.scene_plan, rendered.provenance)
            provenance["scene_plan"] = request.scene_plan.to_dict()
            provenance["scene_uid"] = self._scene_uid(request.scene_plan)
            provenance["geometry_scene_uid"] = request.scene_plan.identity.get(
                "geometry_scene_uid", provenance["scene_uid"]
            )
            provenance["scene_instance_uid"] = request.scene_plan.identity.get(
                "scene_instance_uid", provenance["scene_uid"]
            )
            provenance["actor_order"] = list(request.source_uids)
            provenance["geometry_identity"] = self._geometry_identity(request.scene_plan)
        if isinstance(track_set, (SpatialRIRTrackSet, MovingSpatialRIRTrackSet)):
            provenance.update({"geometry": dict(rendered.geometry), "track_facts": track_set.facts})
        if request.rts_target_t60_sec is not None:
            provenance["target_derivation"] = {
                "type": "rts",
                "original_t60_sec": request.scene_plan.room.get("rt60"),
                "target_t60_sec": float(request.rts_target_t60_sec),
                "time_after_max_sec": float(request.rts_time_after_max_sec),
                "reference_channel": int(request.reference_channel),
            }
        return SpatialSceneResult(
            observation=observation,
            mixture=rendered.mixture,
            full_tracks=rendered.full_tracks,
            direct_tracks=rendered.direct_tracks,
            early_tracks=rendered.early_tracks,
            background=background,
            geometry=rendered.geometry,
            provenance=provenance,
            source_uids=tuple(request.source_uids),
            trajectory=trajectory,
            timebase=timebase,
            rts_tracks=rendered.rts_tracks,
        )

    @staticmethod
    def _scene_uid(plan: AcousticScenePlan) -> str:
        identity = plan.identity
        for key in ("scene_uid", "scene_id"):
            value = identity.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise ValueError("scene_plan identity must contain non-empty scene_uid")

    @staticmethod
    def _geometry_identity(plan: AcousticScenePlan) -> dict[str, Any]:
        if plan.array is None:
            raise ValueError("scene_plan must contain resolved microphone array")
        return plan.array.to_dict()

    @classmethod
    def _validate_scene_plan_geometry(cls, plan: AcousticScenePlan, geometry: Mapping[str, Any]) -> None:
        expected = cls._geometry_identity(plan)
        actual_positions = geometry.get("positions_m")
        if actual_positions is None:
            actual_positions = geometry.get("topology", {}).get("positions_m")
        expected_positions = expected["topology"]["positions_m"]
        if actual_positions is None or not np.allclose(actual_positions, expected_positions):
            raise ValueError("scene_plan geometry identity does not match rendered geometry")

    @classmethod
    def _validate_scene_plan_provenance(
        cls, plan: AcousticScenePlan, provenance: Mapping[str, Any]
    ) -> None:
        """校验 provider 返回的 scene identity 与请求计划属于同一场景。"""
        identity = plan.identity
        expected_uid = next(
            identity[key] for key in ("scene_uid", "scene_id")
            if isinstance(identity.get(key), str) and identity[key].strip()
        )
        actual_uid = provenance.get("scene_uid", provenance.get("scene_id"))
        if actual_uid is not None and str(actual_uid) != str(expected_uid):
            raise ValueError("provider scene identity does not match scene_plan scene_uid")
        if plan.room is not None:
            actual_room = provenance.get("room")
            if actual_room is not None and _thaw_json(actual_room) != plan.to_dict()["room"]:
                raise ValueError("provider room identity does not match scene_plan room")

    def _sample_static(self, request: SpatialSceneRequest) -> SpatialRIRTrackSet:
        sampler = getattr(request.provider, "sample_scene_tracks", None)
        if not callable(sampler):
            raise TypeError("static spatial provider must implement sample_scene_tracks")
        result = sampler(request.scene_plan, normalize=request.normalize, track_kinds=request.track_kinds)
        if not isinstance(result, SpatialRIRTrackSet):
            raise TypeError("scene-level static provider must return SpatialRIRTrackSet")
        return result

    def _sample_moving(self, request: SpatialSceneRequest) -> MovingSpatialRIRTrackSet:
        sampler = getattr(request.provider, "sample_moving_scene_tracks", None)
        if not callable(sampler):
            raise TypeError("moving spatial provider must implement sample_moving_scene_tracks")
        plan = request.scene_plan
        result = sampler(MovingSceneRIRRequest(
            sample_rate=request.sample_rate,
            num_samples=request.num_samples or request.sources.shape[1],
            actor_trajectories={actor.actor_uid: actor.trajectory for actor in plan.actors},
            room=plan.to_dict()["room"],
            array={"positions_m": np.asarray(plan.array.positions_m).tolist(),
                   "reference_channel": plan.array.reference_channel},
            timebase=plan.timebase, source_uids=tuple(request.source_uids),
            scene_uid=self._scene_uid(plan), actor_roles=tuple(actor.role for actor in plan.actors),
            track_kinds=tuple(kind for kind in request.track_kinds if kind in {'full', 'direct'}),
        ), normalize=request.normalize)
        if not isinstance(result, MovingSpatialRIRTrackSet):
            raise TypeError("scene-level moving provider must return MovingSpatialRIRTrackSet")
        return result

    @staticmethod
    def _derive_rts_rirs(request: SpatialSceneRequest, full_rirs: np.ndarray):
        if request.rts_target_t60_sec is None:
            return None
        room = request.scene_plan.room
        original_t60 = None if room is None else room.get("rt60")
        if original_t60 is None:
            raise ValueError("RTS target requires scene_plan.room.rt60")
        return derive_rts_rir_bank(
            full_rirs,
            reference_channel=request.reference_channel,
            original_t60_sec=original_t60,
            target_t60_sec=request.rts_target_t60_sec,
            sample_rate=request.sample_rate,
            time_after_max_sec=request.rts_time_after_max_sec,
        )

    def _render_static(self, request: SpatialSceneRequest, tracks: SpatialRIRTrackSet):
        needed = set(request.track_kinds)
        if "direct" in needed and tracks.direct is None:
            raise ValueError("requested direct track is unavailable")
        if "early" in needed:
            early_boundary = request.early_boundary
        else:
            early_boundary = 0
        full_rirs = tracks.full.canonical_rirs
        rts_rirs = self._derive_rts_rirs(request, full_rirs)
        rendered = render_spatial_scene(
            request.sources, full_rirs,
            early_boundary=early_boundary, reference_channel=request.reference_channel,
            num_samples=request.num_samples,
            direct_rirs=(
                tracks.direct.canonical_rirs
                if "direct" in needed and tracks.direct is not None else None
            ),
            early_rirs=(tracks.early.canonical_rirs if "early" in needed and tracks.early is not None else None),
            rts_rirs=rts_rirs,
            geometry=tracks.geometry, provenance=tracks.provenance,
        )
        return rendered, None, None

    def _render_moving(self, request: SpatialSceneRequest, tracks: MovingSpatialRIRTrackSet):
        if "direct" in request.track_kinds and tracks.direct is None:
            raise ValueError("requested direct track is unavailable")
        plan = tracks.plan
        raw_timebase = plan.get("trajectory_timebase")
        if not isinstance(raw_timebase, Mapping):
            raise ValueError("moving provider plan must contain trajectory_timebase")
        timebase = TrajectoryTimebase(
            int(raw_timebase["sample_rate"]), int(raw_timebase["num_samples"]),
            int(raw_timebase["hop_samples"]), int(raw_timebase["frame_count"]),
            int(raw_timebase.get("start_sample", 0)),
        )
        positions = np.asarray(plan.get("trajectory"), dtype=np.float32)
        trajectory = request.trajectory or SourceTrajectory(positions, timebase)
        if trajectory.timebase != timebase or trajectory.source_count != request.sources.shape[0]:
            raise ValueError("moving provider trajectory does not match request timebase/source order")
        if not np.array_equal(trajectory.positions_m, positions):
            raise ValueError("moving provider positions do not match requested trajectory")
        if request.scene_plan is not None:
            planned = np.concatenate([actor.trajectory.positions_m for actor in request.scene_plan.actors], axis=1)
            if timebase != request.scene_plan.timebase or not np.array_equal(positions, planned):
                raise ValueError("moving provider trajectory does not match scene plan actors")
        rts_rirs = self._derive_rts_rirs(request, tracks.full)
        rendered = render_moving_spatial_scene(
            request.sources,
            tracks.full,
            direct_rirs=tracks.direct,
            frame_samples=timebase.frame_samples,
            early_boundary=request.early_boundary,
            reference_channel=request.reference_channel, num_samples=request.num_samples,
            rts_rirs=rts_rirs,
        )
        rendered = SpatialRenderResult(
            rendered.full_tracks, rendered.early_tracks, rendered.reference_tracks, rendered.mixture,
            direct_tracks=(rendered.direct_tracks if "direct" in request.track_kinds else None),
            geometry=tracks.plan.get("geometry", {}), provenance=tracks.facts,
            rts_tracks=rendered.rts_tracks,
        )
        return rendered, trajectory, timebase

    @staticmethod
    def _resolve_background(request: SpatialSceneRequest, mixture: np.ndarray) -> np.ndarray:
        if request.background is None:
            return np.zeros_like(mixture, dtype=np.float32)
        background = request.background
        if background.shape != mixture.shape:
            raise ValueError(f"background shape {background.shape} must match rendered mixture {mixture.shape}")
        return np.ascontiguousarray(background, dtype=np.float32)

    @staticmethod
    def _validate_rendered(request, tracks, rendered: SpatialRenderResult) -> None:
        if rendered.full_tracks.shape[0] != request.sources.shape[0]:
            raise ValueError("rendered source order/count does not match request")
        if rendered.full_tracks.shape[1] <= request.reference_channel:
            raise ValueError("reference_channel exceeds rendered microphone channels")
        if isinstance(tracks, SpatialRIRTrackSet):
            if tracks.full.canonical_rirs.shape[:2] != rendered.full_tracks.shape[:2]:
                raise ValueError("static RIR source/channel axes do not match rendered tracks")
            declared_order = tracks.provenance.get("source_order")
            if declared_order is not None and tuple(declared_order) != tuple(request.source_uids):
                raise ValueError("static RIR source_order does not match request")
            shared_provenance = tracks.full.provenance
            for track_name, track in (
                ("full", tracks.full),
                ("direct", tracks.direct),
                ("early", tracks.early),
            ):
                if track is None:
                    continue
                declared_kind = _declared_track_kind(track, track_name)
                if declared_kind != track_name:
                    raise ValueError(
                        f"static {track_name} track provenance kind mismatch: {declared_kind!r}"
                    )
                if track is tracks.full:
                    continue
                for identity_key in (
                    "geometry_scene_uid", "scene_instance_uid", "scene_uid",
                    "group_uid", "source_order", "actor_order", "room", "sample_rate",
                ):
                    if identity_key not in shared_provenance and identity_key not in track.provenance:
                        continue
                    if _thaw_json(shared_provenance.get(identity_key)) != _thaw_json(
                        track.provenance.get(identity_key)
                    ):
                        raise ValueError(
                            f"static {track_name} scene identity provenance mismatch: {identity_key}"
                        )
        else:
            expected = tracks.full.shape[:3]
            if rendered.full_tracks.shape[0] != expected[1] or rendered.full_tracks.shape[1] != expected[2]:
                raise ValueError("moving RIR source/channel axes do not match rendered tracks")
            plan_order = tracks.plan.get("source_ids")
            if plan_order is not None and tuple(plan_order) != tuple(request.source_uids):
                raise ValueError("moving RIR source_order does not match request")
            if tracks.track_facts.get("group_uid") is None and tracks.plan.get("group_uid") is None:
                raise ValueError("moving RIR provenance must contain group_uid")


__all__ = ["SpatialSceneRequest", "SpatialSceneResult", "SpatialSceneRenderer"]
