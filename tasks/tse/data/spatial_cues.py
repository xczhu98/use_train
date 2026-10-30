"""TSE 空间 cue 的几何构造与 STFT frame 对齐纯函数。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal

import torch

from core.data.acoustic_scene_plan import AcousticScenePlan
from core.data.simulation.array_profiles import ResolvedMicrophoneArray
from tasks.tse.cue_contract import TSECue


SpatialRepresentation = Literal["static", "track"]
TrackAlignment = Literal["renderer_segment"]


def _finite_vector3(value: Sequence[Real], name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain three finite metre coordinates")
    values = tuple(value)
    if len(values) != 3:
        raise ValueError(f"{name} must contain three finite metre coordinates")
    result = tuple(float(item) for item in values)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain three finite metre coordinates")
    return result  # type: ignore[return-value]


def _positive_finite(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def azimuth_elevation_from_positions(
    actor_position_m: Sequence[Real],
    array_origin_m: Sequence[Real],
) -> torch.Tensor:
    """返回 ``[azimuth, elevation]``，原点为阵列 origin，单位为弧度。

    这是 TSE spatial cue 方向角的唯一定义（训练、offline 解码、模型前端
    与文件推理共用同一口径）：

    - ``azimuth = atan2(dy, dx)``：数学惯例，从 +X 轴向 +Y 轴；
    - ``elevation = atan2(dz, hypot(dx, dy))``：从水平面（XY）起算，向上为正。

    注意与 SSL 任务约定的区别：SSL（``tasks/ssl/data/projection.py::_doa``
    与 ``tasks/ssl/models/ipdnet.py::DPIPD`` 候选网格）使用物理球坐标，
    其 "elevation" 是从 +Z 轴起算的极角（0..π）。两者换算关系为
    ``el_本文 = π/2 - 极角_SSL``；跨任务消费方向角时必须显式换算，
    禁止直接互换。
    """
    actor = _finite_vector3(actor_position_m, "actor_position_m")
    origin = _finite_vector3(array_origin_m, "array_origin_m")
    dx, dy, dz = (actor[index] - origin[index] for index in range(3))
    horizontal = math.hypot(dx, dy)
    if horizontal == 0.0 and dz == 0.0:
        raise ValueError("actor position must differ from the array origin")
    return torch.tensor(
        [math.atan2(dy, dx), math.atan2(dz, horizontal)],
        dtype=torch.float32,
    )


def _array_origin(
    array: ResolvedMicrophoneArray,
    explicit_origin_m: Sequence[Real] | None,
) -> tuple[float, float, float]:
    if explicit_origin_m is not None:
        return _finite_vector3(explicit_origin_m, "array_origin_m")
    positions = tuple(
        _finite_vector3(position, f"array.positions_m[{index}]")
        for index, position in enumerate(array.positions_m)
    )
    count = float(len(positions))
    return tuple(sum(position[axis] for position in positions) / count for axis in range(3))  # type: ignore[return-value]


def _scene_geometry_facts(
    scene_plan: AcousticScenePlan,
) -> tuple[ResolvedMicrophoneArray, str, str, float]:
    if not isinstance(scene_plan, AcousticScenePlan):
        raise TypeError("scene_plan must be an AcousticScenePlan")
    if scene_plan.array is None:
        raise ValueError("scene_plan must contain resolved microphone geometry")
    scene_uid = next(
        (
            str(scene_plan.identity[key]).strip()
            for key in ("scene_uid", "scene_id")
            if isinstance(scene_plan.identity.get(key), str)
            and str(scene_plan.identity[key]).strip()
        ),
        None,
    )
    if scene_uid is None:
        raise ValueError("scene_plan identity must contain scene_uid or scene_id")
    if scene_plan.room is None:
        raise ValueError("scene_plan must contain room metadata with actual sound_speed")
    unsupported_rotation = sorted(
        key
        for key in (
            "array_rotation",
            "array_rotation_rad",
            "array_orientation",
            "microphone_orientation",
            "rotation_matrix",
        )
        if key in scene_plan.room
    )
    if unsupported_rotation:
        raise ValueError(
            "room-array rotation is not modeled; unsupported room keys: "
            f"{unsupported_rotation}"
        )
    speed = scene_plan.room.get("sound_speed")
    if speed is None:
        raise ValueError("scene_plan.room.sound_speed is required; no default is assumed")
    coordinate_frame = scene_plan.room.get("coordinate_system")
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise ValueError("scene_plan.room.coordinate_system is required")
    if coordinate_frame.strip() != scene_plan.array.profile.coordinate_system:
        raise ValueError("room and microphone array must use the same unrotated coordinate frame")
    return (
        scene_plan.array,
        scene_uid,
        coordinate_frame.strip(),
        _positive_finite(speed, "scene_plan.room.sound_speed"),
    )


@dataclass(frozen=True)
class StaticSpatialCue:
    """带场景与 actor 身份的静态方向；``value`` 为 ``[2]`` 弧度。"""

    value: torch.Tensor
    actor_uid: str
    scene_uid: str
    coordinate_frame: str
    sound_speed_mps: float
    array_origin_m: tuple[float, float, float]
    reference_channel: int

    def to_tse_cue(self) -> TSECue:
        return TSECue("static", self.value, True, None, None)


@dataclass(frozen=True)
class TrackSpatialCue:
    """与 moving renderer 分段边界一致的 actor 方向轨迹及来源身份。"""

    cue: TSECue
    actor_uid: str
    scene_uid: str
    coordinate_frame: str
    sound_speed_mps: float
    array_origin_m: tuple[float, float, float]
    reference_channel: int
    source_sample_anchors: tuple[int, ...]
    output_start_sample: int
    output_num_samples: int
    renderer: str = "moving_segment_overlap_add"

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "actor_uid": self.actor_uid,
            "scene_uid": self.scene_uid,
            "coordinate_frame": self.coordinate_frame,
            "sound_speed_mps": self.sound_speed_mps,
            "array_origin_m": list(self.array_origin_m),
            "reference_channel": self.reference_channel,
            "source_sample_anchors": list(self.source_sample_anchors),
            "output_start_sample": self.output_start_sample,
            "output_num_samples": self.output_num_samples,
            "renderer": self.renderer,
        }

    def to_tse_cue(self) -> TSECue:
        return self.cue


def build_static_spatial_cue(
    *,
    actor_positions_m: Mapping[str, Sequence[Real]],
    actor_uid: str,
    array: ResolvedMicrophoneArray,
    scene_uid: str,
    coordinate_frame: str,
    sound_speed_mps: Real,
    array_origin_m: Sequence[Real] | None = None,
) -> StaticSpatialCue:
    """从同一显式坐标系的 actor 与阵列几何中心构造静态 DOA。"""
    if not isinstance(actor_positions_m, Mapping):
        raise TypeError("actor_positions_m must be a mapping keyed by actor_uid")
    if not isinstance(actor_uid, str) or not actor_uid.strip():
        raise ValueError("actor_uid must be a non-empty string")
    if actor_uid not in actor_positions_m:
        raise KeyError(f"actor_positions_m does not contain actor {actor_uid!r}")
    if not isinstance(array, ResolvedMicrophoneArray):
        raise TypeError("array must be a ResolvedMicrophoneArray")
    if not isinstance(scene_uid, str) or not scene_uid.strip():
        raise ValueError("scene_uid must be a non-empty string")
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise ValueError("coordinate_frame must explicitly identify the shared coordinate frame")
    if coordinate_frame.strip() != array.profile.coordinate_system:
        raise ValueError("actor and microphone array must use the same coordinate frame")
    speed = _positive_finite(sound_speed_mps, "sound_speed_mps")
    origin = _array_origin(array, array_origin_m)
    value = azimuth_elevation_from_positions(actor_positions_m[actor_uid], origin)
    return StaticSpatialCue(
        value=value,
        actor_uid=actor_uid.strip(),
        scene_uid=scene_uid.strip(),
        coordinate_frame=coordinate_frame.strip(),
        sound_speed_mps=speed,
        array_origin_m=origin,
        reference_channel=array.reference_channel,
    )


def build_static_spatial_cue_from_scene(
    scene_plan: AcousticScenePlan,
    *,
    actor_uid: str,
    array_origin_m: Sequence[Real] | None = None,
) -> StaticSpatialCue:
    """按 ``actor_uid`` 从静态 scene 构造方向；reference channel 不参与 DOA。"""
    array, scene_uid, coordinate_frame, speed = _scene_geometry_facts(scene_plan)
    actors = {actor.actor_uid: actor for actor in scene_plan.actors}
    if actor_uid not in actors:
        raise KeyError(f"scene_plan does not contain actor {actor_uid!r}")
    actor = actors[actor_uid]
    if actor.trajectory is not None:
        raise ValueError("static spatial cue construction rejects moving actor trajectories")
    if actor.position_m is None:
        raise ValueError("static actor must contain position_m")
    return build_static_spatial_cue(
        actor_positions_m={actor_uid: actor.position_m},
        actor_uid=actor_uid,
        array=array,
        scene_uid=scene_uid,
        coordinate_frame=coordinate_frame,
        sound_speed_mps=speed,
        array_origin_m=array_origin_m,
    )


def build_track_spatial_cue_from_scene(
    scene_plan: AcousticScenePlan,
    *,
    actor_uid: str,
    output_start_sample: int = 0,
    output_num_samples: int | None = None,
    array_origin_m: Sequence[Real] | None = None,
) -> TrackSpatialCue:
    """按 actor identity 和 moving renderer 分段起点构造 canonical track cue。"""
    array, scene_uid, coordinate_frame, speed = _scene_geometry_facts(scene_plan)
    if scene_plan.timebase is None:
        raise ValueError("track spatial cue requires scene_plan.timebase")
    actors = {actor.actor_uid: actor for actor in scene_plan.actors}
    if actor_uid not in actors:
        raise KeyError(f"scene_plan does not contain actor {actor_uid!r}")
    actor = actors[actor_uid]
    trajectory = actor.trajectory
    if trajectory is None:
        raise ValueError("track spatial cue requires an actor trajectory")
    if trajectory.timebase != scene_plan.timebase:
        raise ValueError("actor trajectory must share scene_plan.timebase")
    if trajectory.source_count != 1:
        raise ValueError(
            "actor trajectory must contain exactly one identity row; source-axis inference is forbidden"
        )
    anchors = tuple(int(value) for value in scene_plan.timebase.frame_samples.tolist())
    if not anchors or anchors[0] != 0:
        raise ValueError("moving renderer trajectory sample anchors must start at 0")
    if type(output_start_sample) is not int or not 0 <= output_start_sample < scene_plan.num_samples:
        raise ValueError("output_start_sample must be an exact int in [0, scene length)")
    if output_num_samples is None:
        output_num_samples = scene_plan.num_samples - output_start_sample
    if type(output_num_samples) is not int or output_num_samples <= 0:
        raise ValueError("output_num_samples must be a positive exact int")
    output_stop = output_start_sample + output_num_samples
    if output_stop > scene_plan.num_samples:
        raise ValueError("output crop must not exceed scene_plan.num_samples")

    active_index = max(
        index for index, sample in enumerate(anchors) if sample <= output_start_sample
    )
    selected_indices = [active_index]
    selected_indices.extend(
        index
        for index, sample in enumerate(anchors)
        if output_start_sample < sample < output_stop
    )
    selected_indices = list(dict.fromkeys(selected_indices))
    source_anchors = tuple(anchors[index] for index in selected_indices)
    relative_anchors = (0,) + tuple(
        anchor - output_start_sample for anchor in source_anchors[1:]
    )
    origin = _array_origin(array, array_origin_m)
    directions = torch.stack(
        [
            azimuth_elevation_from_positions(
                trajectory.positions_m[index, 0],
                origin,
            )
            for index in selected_indices
        ],
        dim=1,
    )
    cue = TSECue(
        "track",
        directions,
        True,
        len(selected_indices),
        torch.tensor(relative_anchors, dtype=torch.int64),
    )
    return TrackSpatialCue(
        cue=cue,
        actor_uid=actor_uid,
        scene_uid=scene_uid,
        coordinate_frame=coordinate_frame,
        sound_speed_mps=speed,
        array_origin_m=origin,
        reference_channel=array.reference_channel,
        source_sample_anchors=source_anchors,
        output_start_sample=output_start_sample,
        output_num_samples=output_num_samples,
    )


def _frame_samples(value: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    frames = torch.as_tensor(value, device=device)
    if frames.ndim != 1 or frames.numel() < 1:
        raise ValueError("frame_sample_indices must have shape [T] with T >= 1")
    if frames.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("frame_sample_indices must use an integer dtype")
    frames = frames.to(dtype=torch.int64)
    if bool((frames < 0).any().item()):
        raise ValueError("frame_sample_indices must be non-negative")
    if frames.numel() > 1 and bool((frames[1:] < frames[:-1]).any().item()):
        raise ValueError("frame_sample_indices must be monotonic non-decreasing")
    return frames


def align_spatial_cue_to_frames(
    cue_value: torch.Tensor | None,
    *,
    representation: SpatialRepresentation,
    present: torch.Tensor,
    frame_sample_indices: torch.Tensor,
    sample_anchors: torch.Tensor | None = None,
    lengths: torch.Tensor | None = None,
    track_alignment: TrackAlignment = "renderer_segment",
) -> torch.Tensor:
    """对齐到 ``[B,2,T]``；track 严格按 renderer 的分段起点语义取值。"""
    if representation not in {"static", "track"}:
        raise ValueError("representation must be 'static' or 'track'")
    if not isinstance(present, torch.Tensor) or present.ndim != 1 or present.dtype != torch.bool:
        raise TypeError("present must be torch.bool with shape [B]")
    batch = int(present.shape[0])
    device = present.device
    frames = _frame_samples(frame_sample_indices, device=device)
    frame_count = int(frames.numel())

    if cue_value is None:
        if bool(present.any().item()):
            raise ValueError("present spatial rows require cue_value")
        return torch.zeros((batch, 2, frame_count), dtype=torch.float32, device=device)
    if not isinstance(cue_value, torch.Tensor) or not cue_value.dtype.is_floating_point:
        raise TypeError("cue_value must be a floating-point tensor")
    if cue_value.device != device:
        raise ValueError("cue_value and present must use the same device")
    if not bool(torch.isfinite(cue_value).all().item()):
        raise ValueError("cue_value must contain only finite radians")

    if representation == "static":
        if tuple(cue_value.shape) != (batch, 2):
            raise ValueError(f"static cue_value must have shape [{batch},2]")
        if sample_anchors is not None or lengths is not None:
            raise ValueError("static spatial cues must not provide sample_anchors or lengths")
        aligned = cue_value.unsqueeze(-1).expand(-1, -1, frame_count).clone()
        aligned[~present] = 0
        return aligned

    if track_alignment != "renderer_segment":
        raise ValueError(
            "track_alignment must be 'renderer_segment'; point-count resizing is unsupported"
        )
    if cue_value.ndim != 3 or cue_value.shape[:2] != (batch, 2):
        raise ValueError(f"track cue_value must have shape [{batch},2,P]")
    capacity = int(cue_value.shape[2])
    if not isinstance(sample_anchors, torch.Tensor) or tuple(sample_anchors.shape) != (
        batch,
        capacity,
    ):
        raise ValueError(f"sample_anchors must have shape [{batch},{capacity}]")
    if sample_anchors.device != device or sample_anchors.dtype != torch.int64:
        raise TypeError("sample_anchors must be torch.int64 on the cue device")
    if not isinstance(lengths, torch.Tensor) or tuple(lengths.shape) != (batch,):
        raise ValueError(f"lengths must have shape [{batch}]")
    if lengths.device != device or lengths.dtype != torch.int32:
        raise TypeError("lengths must be torch.int32 on the cue device")

    aligned = cue_value.new_zeros((batch, 2, frame_count))
    for row in range(batch):
        count = int(lengths[row].item())
        if not bool(present[row].item()):
            if count != 0:
                raise ValueError("missing track rows must use length=0")
            continue
        if not 1 <= count <= capacity:
            raise ValueError("present track lengths must be within [1,P]")
        anchors = sample_anchors[row, :count]
        if int(anchors[0].item()) != 0:
            raise ValueError(
                "renderer_segment track anchors must start at sample 0 of the returned mixture"
            )
        if count > 1 and bool((anchors[1:] <= anchors[:-1]).any().item()):
            raise ValueError("valid sample_anchors must be strictly increasing")
        indices = torch.searchsorted(anchors, frames, right=True) - 1
        aligned[row] = cue_value[row, :, indices]
    return aligned


__all__ = [
    "SpatialRepresentation",
    "StaticSpatialCue",
    "TrackAlignment",
    "TrackSpatialCue",
    "align_spatial_cue_to_frames",
    "azimuth_elevation_from_positions",
    "build_static_spatial_cue",
    "build_static_spatial_cue_from_scene",
    "build_track_spatial_cue_from_scene",
]
