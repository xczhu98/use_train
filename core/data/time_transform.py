"""可重放的 waveform 时间变换 plan 与纯 apply Module。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeAlias

import numpy as np

from core.data.waveform_bundle import WaveformBundle


POLICY_LAYER_A_RANDOM_CROP = "layer_a_random_crop"
POLICY_FINAL_CENTER_CROP = "final_center_crop"
POLICY_OFFLINE_RANDOM_CROP = "offline_random_crop"


def _nonnegative_int(value: int, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: int, *, field_name: str) -> int:
    value = _nonnegative_int(value, field_name=field_name)
    if value == 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True)
class CropOp:
    """从当前时间轴截取 ``[start, start + output_length)``。"""

    start: int
    output_length: int

    def __post_init__(self) -> None:
        _nonnegative_int(self.start, field_name="CropOp.start")
        _positive_int(self.output_length, field_name="CropOp.output_length")


@dataclass(frozen=True)
class TileOp:
    """按 ``waveform + zero gap`` 周期平铺后，从 ``start`` 截取。"""

    output_length: int
    start: int = 0
    gap_length: int = 0

    def __post_init__(self) -> None:
        _positive_int(self.output_length, field_name="TileOp.output_length")
        _nonnegative_int(self.start, field_name="TileOp.start")
        _nonnegative_int(self.gap_length, field_name="TileOp.gap_length")


@dataclass(frozen=True)
class CanvasOp:
    """把当前 waveform 单次写入固定零画布的 ``offset`` 位置。"""

    output_length: int
    offset: int = 0

    def __post_init__(self) -> None:
        _positive_int(self.output_length, field_name="CanvasOp.output_length")
        _nonnegative_int(self.offset, field_name="CanvasOp.offset")


@dataclass(frozen=True)
class ResampleOp:
    """声明一次由调用者 Adapter 实现的确定性重采样。"""

    source_sample_rate: int
    target_sample_rate: int
    output_length: int

    def __post_init__(self) -> None:
        _positive_int(
            self.source_sample_rate,
            field_name="ResampleOp.source_sample_rate",
        )
        _positive_int(
            self.target_sample_rate,
            field_name="ResampleOp.target_sample_rate",
        )
        _positive_int(self.output_length, field_name="ResampleOp.output_length")


TimeTransformOp: TypeAlias = CropOp | TileOp | CanvasOp | ResampleOp
ResampleAdapter: TypeAlias = Callable[[np.ndarray, int, int], np.ndarray]


def _next_length(operation: TimeTransformOp, input_length: int) -> int:
    if isinstance(operation, CropOp):
        if operation.start + operation.output_length > input_length:
            raise ValueError(
                "CropOp exceeds current time axis: "
                f"start={operation.start}, output_length={operation.output_length}, "
                f"input_length={input_length}"
            )
    elif isinstance(operation, TileOp):
        if input_length == 0:
            raise ValueError("TileOp cannot tile an empty time axis")
        period_length = input_length + operation.gap_length
        if operation.start >= period_length:
            raise ValueError(
                "TileOp.start must be within one tile period: "
                f"start={operation.start}, period_length={period_length}"
            )
    elif isinstance(operation, CanvasOp):
        if operation.offset + input_length > operation.output_length:
            raise ValueError(
                "CanvasOp cannot place the current time axis on its canvas: "
                f"offset={operation.offset}, input_length={input_length}, "
                f"output_length={operation.output_length}"
            )
    elif isinstance(operation, ResampleOp):
        # 长度舍入属于注入 Adapter 的合同；plan 只固化并校验声明结果。
        pass
    else:
        raise TypeError(
            "unsupported time transform operation: "
            f"{type(operation).__name__}"
        )
    return operation.output_length


@dataclass(frozen=True)
class TimeTransformPlan:
    """冻结已抽样参数；构造时完整验证时间轴几何关系。"""

    input_length: int
    operations: tuple[TimeTransformOp, ...]
    policy: str
    required_tracks: tuple[str, ...] = ()
    output_length: int = field(init=False)

    def __post_init__(self) -> None:
        _nonnegative_int(self.input_length, field_name="TimeTransformPlan.input_length")
        if not isinstance(self.policy, str) or not self.policy.strip():
            raise ValueError("TimeTransformPlan.policy must be a non-empty string")
        required_tracks = tuple(self.required_tracks)
        if any(not isinstance(name, str) or not name for name in required_tracks):
            raise ValueError("required track names must be non-empty strings")
        if len(set(required_tracks)) != len(required_tracks):
            raise ValueError("required track names must be unique")
        operations = tuple(self.operations)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "required_tracks", required_tracks)
        current_length = self.input_length
        for operation in operations:
            current_length = _next_length(operation, current_length)
        object.__setattr__(self, "output_length", current_length)


def plan_time_transform(
    *,
    input_length: int,
    operations: tuple[TimeTransformOp, ...],
    policy: str,
    required_tracks: tuple[str, ...] = (),
) -> TimeTransformPlan:
    """用调用者已抽样的参数构造 plan；此 planner 不接收 RNG。"""

    return TimeTransformPlan(
        input_length=input_length,
        operations=operations,
        policy=policy,
        required_tracks=required_tracks,
    )


def _owned_waveform(value: np.ndarray, *, track_name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"track {track_name!r} must be a numpy.ndarray")
    if value.dtype != np.float32:
        raise TypeError(f"track {track_name!r} must have dtype float32")
    if value.ndim < 1:
        raise ValueError(f"track {track_name!r} must have a final time axis")
    if not np.isfinite(value).all():
        raise ValueError(f"track {track_name!r} contains non-finite samples")
    return np.ascontiguousarray(value)


def _tile_waveform(waveform: np.ndarray, operation: TileOp) -> np.ndarray:
    input_length = int(waveform.shape[-1])
    period_length = input_length + operation.gap_length
    shape = waveform.shape[:-1] + (period_length,)
    period = np.zeros(shape, dtype=np.float32)
    period[..., :input_length] = waveform
    needed = operation.start + operation.output_length
    repeats = (needed + period_length - 1) // period_length
    tiled = np.concatenate([period] * repeats, axis=-1)
    return tiled[..., operation.start:needed]


def _apply_operation(
    waveform: np.ndarray,
    operation: TimeTransformOp,
    *,
    track_name: str,
    resample_adapter: ResampleAdapter | None,
) -> np.ndarray:
    if isinstance(operation, CropOp):
        result = waveform[..., operation.start : operation.start + operation.output_length]
    elif isinstance(operation, TileOp):
        result = _tile_waveform(waveform, operation)
    elif isinstance(operation, CanvasOp):
        shape = waveform.shape[:-1] + (operation.output_length,)
        result = np.zeros(shape, dtype=np.float32)
        result[..., operation.offset : operation.offset + waveform.shape[-1]] = waveform
    elif isinstance(operation, ResampleOp):
        if resample_adapter is None:
            raise ValueError("ResampleOp requires an injected resample_adapter")
        result = resample_adapter(
            waveform.copy(),
            operation.source_sample_rate,
            operation.target_sample_rate,
        )
    else:  # pragma: no cover - plan 构造已拒绝未知 op
        raise TypeError(f"unsupported time transform operation: {type(operation).__name__}")

    result = _owned_waveform(result, track_name=track_name)
    if result.shape[:-1] != waveform.shape[:-1]:
        raise ValueError(
            f"track {track_name!r} operation changed non-time axes: "
            f"actual={result.shape[:-1]}, expected={waveform.shape[:-1]}"
        )
    if result.shape[-1] != operation.output_length:
        raise ValueError(
            f"track {track_name!r} operation output length mismatch: "
            f"actual={result.shape[-1]}, expected={operation.output_length}"
        )
    return result


def apply_time_transform(
    plan: TimeTransformPlan,
    bundle: WaveformBundle,
    *,
    resample_adapter: ResampleAdapter | None = None,
) -> WaveformBundle:
    """确定性应用同一 plan；不接收、创建或消费 RNG。"""

    if not isinstance(plan, TimeTransformPlan):
        raise TypeError("plan must be a TimeTransformPlan")
    if not isinstance(bundle, WaveformBundle):
        raise TypeError("bundle must be a WaveformBundle")
    missing = tuple(name for name in plan.required_tracks if name not in bundle.tracks)
    if missing:
        raise ValueError(f"required tracks are missing: {missing}")
    for name, waveform in bundle.tracks.items():
        if waveform.shape[-1] != plan.input_length:
            raise ValueError(
                f"track {name!r} input length mismatch: "
                f"actual={waveform.shape[-1]}, expected={plan.input_length}"
            )

    transformed: dict[str, np.ndarray] = {}
    for name, waveform in bundle.tracks.items():
        current = waveform
        for operation in plan.operations:
            current = _apply_operation(
                current,
                operation,
                track_name=name,
                resample_adapter=resample_adapter,
            )
        if current.shape[-1] != plan.output_length:
            raise ValueError(
                f"track {name!r} final output length mismatch: "
                f"actual={current.shape[-1]}, expected={plan.output_length}"
            )
        transformed[name] = current
    return WaveformBundle(transformed, metadata=bundle.metadata)


@dataclass(frozen=True)
class SceneTimeTransformResult:
    """一次 scene-wide target-SR 与最终时间 plan 的 typed 结果。"""

    bundle: WaveformBundle
    plan: TimeTransformPlan
    sample_rate: int
    fixed_silence_canvas_active: bool
    source_sample_rate: int

    def __post_init__(self) -> None:
        _positive_int(self.sample_rate, field_name="SceneTimeTransformResult.sample_rate")
        _positive_int(
            self.source_sample_rate,
            field_name="SceneTimeTransformResult.source_sample_rate",
        )
        if self.bundle.num_samples != self.plan.output_length:
            raise ValueError("scene bundle length must match time transform plan output")

    def to_metadata(self) -> dict[str, object]:
        """返回 trace 可安全记录的计划事实，不暴露 waveform。"""
        return {
            "policy": self.plan.policy,
            "source_sample_rate": int(self.source_sample_rate),
            "target_sample_rate": int(self.sample_rate),
            "input_length": self.plan.input_length,
            "output_length": self.plan.output_length,
            "operations": [type(operation).__name__ for operation in self.plan.operations],
            "required_tracks": list(self.plan.required_tracks),
            "fixed_silence_canvas_active": self.fixed_silence_canvas_active,
        }


def apply_scene_time_transform(
    *,
    bundle: WaveformBundle,
    source_sample_rate: int,
    target_sample_rate: int | None,
    rng: np.random.Generator,
    fixed_silence_canvas_active: bool,
    build_final_time_plan: Callable[..., Any],
    resample_adapter: ResampleAdapter,
    policy_prefix: str | None = None,
) -> SceneTimeTransformResult:
    """在既有 planner 的 RNG 决策点构建单一 plan 并同步变换所有 scene roles.

    ``build_final_time_plan`` 是任务 Adapter 注入的决策实现；本 Module
    不解释 role、branch 或 Dataset 状态。重采样的实际长度由注入 backend 的零波形
    probe 固化，避免复制其 rounding 规则。
    """
    if not isinstance(bundle, WaveformBundle):
        raise TypeError("bundle must be a WaveformBundle")
    source_sample_rate = _positive_int(
        int(source_sample_rate), field_name="source_sample_rate"
    )
    resolved_target_sample_rate = (
        source_sample_rate
        if target_sample_rate is None
        else _positive_int(int(target_sample_rate), field_name="target_sample_rate")
    )
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    if not callable(build_final_time_plan):
        raise TypeError("build_final_time_plan must be callable")
    if not callable(resample_adapter):
        raise TypeError("resample_adapter must be callable")
    if policy_prefix is not None and (
        not isinstance(policy_prefix, str) or not policy_prefix.strip()
    ):
        raise ValueError("policy_prefix must be a non-empty string when provided")

    operations: list[TimeTransformOp] = []
    planning_length = bundle.num_samples
    if resolved_target_sample_rate != source_sample_rate:
        probe = resample_adapter(
            np.zeros(planning_length, dtype=np.float32),
            source_sample_rate,
            resolved_target_sample_rate,
        )
        planning_length = int(np.asarray(probe).shape[-1])
        operations.append(
            ResampleOp(
                source_sample_rate=source_sample_rate,
                target_sample_rate=resolved_target_sample_rate,
                output_length=planning_length,
            )
        )

    planning_waveform = np.zeros(planning_length, dtype=np.float32)
    final_plan = build_final_time_plan(
        planning_waveform,
        planning_waveform,
        resolved_target_sample_rate,
        rng,
        fixed_silence_canvas_active=bool(fixed_silence_canvas_active),
    )
    transform = getattr(final_plan, "transform", None)
    output_length = getattr(final_plan, "output_length", None)
    if transform is not None:
        if not isinstance(transform, TimeTransformPlan):
            raise TypeError("final planner transform must be a TimeTransformPlan")
        operations.extend(transform.operations)
    if type(output_length) is not int or output_length < 0:
        raise ValueError("final planner must expose a non-negative integer output_length")
    final_policy = "identity" if transform is None else transform.policy
    policy = (
        final_policy
        if resolved_target_sample_rate == source_sample_rate
        else (
            f"{policy_prefix}+{final_policy}"
            if policy_prefix is not None
            else f"target_sr+{final_policy}"
        )
    )
    plan = plan_time_transform(
        input_length=bundle.num_samples,
        operations=tuple(operations),
        policy=policy,
        required_tracks=tuple(bundle.tracks),
    )
    if plan.output_length != output_length:
        raise RuntimeError(
            "scene time plan output disagrees with final alignment plan: "
            f"scene={plan.output_length}, final={output_length}"
        )
    transformed = apply_time_transform(
        plan,
        bundle,
        resample_adapter=(
            resample_adapter
            if resolved_target_sample_rate != source_sample_rate
            else None
        ),
    )
    return SceneTimeTransformResult(
        bundle=transformed,
        plan=plan,
        sample_rate=resolved_target_sample_rate,
        fixed_silence_canvas_active=bool(fixed_silence_canvas_active),
        source_sample_rate=source_sample_rate,
    )


__all__ = [
    "CanvasOp",
    "CropOp",
    "POLICY_FINAL_CENTER_CROP",
    "POLICY_LAYER_A_RANDOM_CROP",
    "POLICY_OFFLINE_RANDOM_CROP",
    "ResampleAdapter",
    "ResampleOp",
    "SceneTimeTransformResult",
    "TileOp",
    "TimeTransformPlan",
    "WaveformBundle",
    "apply_scene_time_transform",
    "apply_time_transform",
    "plan_time_transform",
]
