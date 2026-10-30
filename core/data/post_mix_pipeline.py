"""任务无关的固定顺序 post-mix pipeline 合同。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from core.data.waveform_bundle import WaveformBundle, freeze_waveform_snapshot


PostMixStage = Literal[
    "pre-finite",
    "D-normalize",
    "D-guard",
    "E",
    "F",
    "task-post-EF",
    "final-guard",
    "post-finite",
]
POST_MIX_STAGE_ORDER: tuple[PostMixStage, ...] = (
    "pre-finite",
    "D-normalize",
    "D-guard",
    "E",
    "F",
    "task-post-EF",
    "final-guard",
    "post-finite",
)
_STAGE_SET = frozenset(POST_MIX_STAGE_ORDER)


def _validate_waveform(
    role: object, waveform: object, *, time_length: int | None
) -> np.ndarray:
    if not isinstance(role, str) or not role:
        raise ValueError("waveform keys must be non-empty strings")
    if not isinstance(waveform, np.ndarray) or waveform.dtype != np.float32:
        raise TypeError(f"waveform {role!r} must be a float32 ndarray")
    if waveform.ndim < 1 or waveform.shape[-1] <= 0:
        raise ValueError(f"waveform {role!r} must have a non-empty final time axis")
    if time_length is not None and waveform.shape[-1] != time_length:
        raise ValueError("all waveforms must share the final time-axis length")
    if not np.isfinite(waveform).all():
        raise ValueError(f"waveform {role!r} contains NaN or Inf")
    return waveform


def _freeze_waveform(waveform: np.ndarray) -> np.ndarray:
    """建立或复用 bytes-backed readonly 快照。"""
    return freeze_waveform_snapshot(waveform)


@dataclass(frozen=True, eq=False)
class PostMixState:
    """只携带任务无关波形；stem 含义完全由 Adapter 解释。"""

    waveforms: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if not isinstance(self.waveforms, Mapping) or not self.waveforms:
            raise ValueError("waveforms must be a non-empty mapping")
        frozen: dict[str, np.ndarray] = {}
        time_length: int | None = None
        for role, waveform in self.waveforms.items():
            validated = _validate_waveform(role, waveform, time_length=time_length)
            if time_length is None:
                time_length = validated.shape[-1]
            frozen[role] = _freeze_waveform(validated)
        object.__setattr__(self, "waveforms", MappingProxyType(frozen))

    @classmethod
    def _from_frozen_waveforms(
        cls, waveforms: Mapping[str, np.ndarray]
    ) -> "PostMixState":
        """复用框架已拥有的只读 waveform，避免再次复制完整 scene。"""
        if not isinstance(waveforms, Mapping) or not waveforms:
            raise ValueError("frozen waveforms must be a non-empty mapping")
        time_length: int | None = None
        shared: dict[str, np.ndarray] = {}
        for role, waveform in waveforms.items():
            validated = _validate_waveform(role, waveform, time_length=time_length)
            if validated.flags.writeable:
                raise ValueError("shared frozen waveforms must be read-only")
            if time_length is None:
                time_length = validated.shape[-1]
            shared[role] = validated
        state = object.__new__(cls)
        object.__setattr__(state, "waveforms", MappingProxyType(shared))
        return state

    def _replace_waveforms(
        self, changes: Mapping[str, np.ndarray]
    ) -> "PostMixState":
        """受控替换已有轨道；未变化轨道复用当前只读快照。"""
        if not isinstance(changes, Mapping):
            raise TypeError("waveform replacements must be a mapping")
        unknown = tuple(role for role in changes if role not in self.waveforms)
        if unknown:
            raise KeyError(f"unknown waveform replacement roles: {unknown}")
        time_length = next(iter(self.waveforms.values())).shape[-1]
        effective: dict[str, np.ndarray] = {}
        for role, waveform in changes.items():
            validated = _validate_waveform(role, waveform, time_length=time_length)
            current = self.waveforms[role]
            if validated is current or np.array_equal(validated, current):
                continue
            effective[role] = validated
        if not effective:
            return self

        waveforms = dict(self.waveforms)
        for role, waveform in effective.items():
            waveforms[role] = _freeze_waveform(waveform)

        replacement = object.__new__(type(self))
        object.__setattr__(replacement, "waveforms", MappingProxyType(waveforms))
        return replacement


@dataclass(frozen=True, eq=False)
class PostMixStageResult:
    """单个固定 stage 的输出；``applied=False`` 声明波形 no-op。"""

    stage: PostMixStage
    state: PostMixState
    applied: bool

    def __post_init__(self) -> None:
        if self.stage not in _STAGE_SET:
            raise ValueError(f"unknown post-mix stage: {self.stage!r}")
        if not isinstance(self.state, PostMixState):
            raise TypeError("state must be PostMixState")
        if type(self.applied) is not bool:
            raise TypeError("applied must be bool")


@dataclass(frozen=True, eq=False)
class PostMixTrace:
    """完整成功执行的有序结果；失败执行不会产生 partial trace。"""

    initial_state: PostMixState
    results: tuple[PostMixStageResult, ...]
    retains_stage_states: bool = True

    def __post_init__(self) -> None:
        results = tuple(self.results)
        if tuple(result.stage for result in results) != POST_MIX_STAGE_ORDER:
            raise ValueError("trace must contain every post-mix stage in fixed order")
        if type(self.retains_stage_states) is not bool:
            raise TypeError("retains_stage_states must be bool")
        if not self.retains_stage_states:
            final_state = results[-1].state
            if any(result.state is not final_state for result in results):
                raise ValueError("compact trace results must share the final state")
        object.__setattr__(self, "results", results)

    @property
    def stage_order(self) -> tuple[PostMixStage, ...]:
        return tuple(result.stage for result in self.results)

    @property
    def final_state(self) -> PostMixState:
        return self.results[-1].state


def _freeze_context(value: object, *, path: str) -> object:
    """冻结 task-neutral JSON-safe facts；waveform 必须保留在 bundle。"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"context number at {path} must be finite")
        return number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"context keys at {path} must be strings")
        return MappingProxyType(
            {
                key: _freeze_context(item, path=f"{path}.{key}")
                for key, item in value.items()
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_context(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, np.ndarray):
        raise TypeError(f"context must not contain ndarray at {path}")
    raise TypeError(f"context value at {path} is not JSON-safe: {type(value).__name__}")


def _context_to_dict(value: object, *, path: str) -> object:
    """将冻结 context 投影为严格 JSON-safe 的可变 Python 容器。"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"context number at {path} must be finite")
        return number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"context keys at {path} must be strings")
        return {
            key: _context_to_dict(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            _context_to_dict(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.ndarray):
        raise TypeError(f"context must not contain ndarray at {path}")
    raise TypeError(f"context value at {path} is not JSON-safe: {type(value).__name__}")


@dataclass(frozen=True, eq=False)
class PostMixContext:
    """post-mix orchestration 的只读任务 facts；不承载 waveform。"""

    facts: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.facts, Mapping):
            raise TypeError("facts must be a mapping")
        object.__setattr__(self, "facts", _freeze_context(self.facts, path="context"))

    def to_dict(self) -> dict[str, object]:
        """投影为可直接传给 trace 或 ``json.dumps`` 的严格 JSON-safe facts。"""
        projected = _context_to_dict(self.facts, path="context")
        if not isinstance(projected, dict):
            raise TypeError("post-mix context projection must be a mapping")
        return projected


@dataclass(frozen=True, eq=False)
class PostMixResult:
    """一次完整固定 stage 执行的 typed runtime 结果。"""

    initial_bundle: WaveformBundle
    bundle: WaveformBundle
    context: PostMixContext
    trace: PostMixTrace

    def __post_init__(self) -> None:
        if not isinstance(self.initial_bundle, WaveformBundle):
            raise TypeError("initial_bundle must be a WaveformBundle")
        if not isinstance(self.bundle, WaveformBundle):
            raise TypeError("bundle must be a WaveformBundle")
        if not isinstance(self.context, PostMixContext):
            raise TypeError("context must be a PostMixContext")
        if not isinstance(self.trace, PostMixTrace):
            raise TypeError("trace must be a PostMixTrace")
        if self.bundle.tracks.keys() != self.trace.final_state.waveforms.keys():
            raise ValueError("result bundle roles must match final post-mix state")
        for role, waveform in self.trace.final_state.waveforms.items():
            if not np.array_equal(self.bundle.require(role), waveform):
                raise ValueError(
                    f"result bundle waveform for role {role!r} disagrees with final state"
                )


@runtime_checkable
class TaskPostMixAdapter(Protocol):
    """任务侧逐 stage Adapter；no-op stage 不得消费 RNG。"""

    def apply_stage(
        self, stage: PostMixStage, state: PostMixState
    ) -> PostMixStageResult: ...


def _same_waveforms(left: PostMixState, right: PostMixState) -> bool:
    return left is right or (
        left.waveforms.keys() == right.waveforms.keys()
        and all(
            np.array_equal(left.waveforms[key], right.waveforms[key])
            for key in left.waveforms
        )
    )


class PostMixEngine:
    """执行固定 D/E/F 生命周期的无状态编排器。

    任务侧 Adapter 是唯一的策略扩展点：它负责每个 stage 的 DSP 和
    task-specific facts，Engine 只负责统一的 stage 顺序、no-op 合同和
    typed bundle 投影。SE、AEC 与 observation renderer 都通过这个 seam
    执行，避免各自复制一份循环逻辑。
    """

    __slots__ = ()

    def run(
        self,
        initial_state: PostMixState,
        adapter: TaskPostMixAdapter,
        *,
        retain_stage_states: bool = True,
    ) -> PostMixTrace:
        """运行固定 stage pipeline 并返回完整 trace。"""
        return _run_post_mix_pipeline_impl(
            initial_state,
            adapter,
            retain_stage_states=retain_stage_states,
        )

    def run_bundle(
        self,
        bundle: WaveformBundle,
        adapter: TaskPostMixAdapter,
        *,
        context: PostMixContext | Mapping[str, object] | None = None,
        retain_stage_states: bool = True,
    ) -> PostMixResult:
        """运行固定 stage pipeline，并将最终 state 投影回 typed bundle。"""
        return _run_post_mix_orchestration_impl(
            bundle,
            adapter,
            context=context,
            retain_stage_states=retain_stage_states,
        )


_POST_MIX_ENGINE = PostMixEngine()


def _run_post_mix_pipeline_impl(
    initial_state: PostMixState,
    adapter: TaskPostMixAdapter,
    *,
    retain_stage_states: bool = True,
) -> PostMixTrace:
    """按唯一模板执行；Adapter 失败、漏 stage 或乱序时立即 fail-closed。"""
    if not isinstance(initial_state, PostMixState):
        raise TypeError("initial_state must be PostMixState")
    if type(retain_stage_states) is not bool:
        raise TypeError("retain_stage_states must be bool")
    apply_stage = getattr(adapter, "apply_stage", None)
    if not callable(apply_stage):
        raise TypeError("adapter must provide callable apply_stage")

    current = initial_state
    results: list[PostMixStageResult] = []
    stage_facts: list[tuple[PostMixStage, bool]] = []
    for expected_stage in POST_MIX_STAGE_ORDER:
        result = apply_stage(expected_stage, current)
        if not isinstance(result, PostMixStageResult):
            raise TypeError(f"adapter did not return PostMixStageResult for {expected_stage!r}")
        if result.stage != expected_stage:
            raise ValueError(
                f"adapter returned stage {result.stage!r}, expected {expected_stage!r}"
            )
        if not result.applied and not _same_waveforms(current, result.state):
            raise ValueError(f"no-op stage {expected_stage!r} changed waveforms")
        if retain_stage_states:
            results.append(result)
        else:
            stage_facts.append((result.stage, result.applied))
        current = result.state
    if not retain_stage_states:
        results = [
            PostMixStageResult(stage, current, applied)
            for stage, applied in stage_facts
        ]
    return PostMixTrace(
        initial_state=initial_state,
        results=tuple(results),
        retains_stage_states=retain_stage_states,
    )


def run_post_mix_pipeline(
    initial_state: PostMixState,
    adapter: TaskPostMixAdapter,
    *,
    retain_stage_states: bool = True,
) -> PostMixTrace:
    """兼容入口；统一转发到 :class:`PostMixEngine`。"""
    return _POST_MIX_ENGINE.run(
        initial_state,
        adapter,
        retain_stage_states=retain_stage_states,
    )


def _run_post_mix_orchestration_impl(
    bundle: WaveformBundle,
    adapter: TaskPostMixAdapter,
    *,
    context: PostMixContext | Mapping[str, object] | None = None,
    retain_stage_states: bool = True,
) -> PostMixResult:
    """以 typed bundle 执行固定 pipeline，并返回完整不可变 runtime 结果。

    Adapter 继续解释 waveform role 并持有 task-specific DSP；该 Module 只负责
    建立 state、固定顺序执行和将最终 state 投影为可审计的 bundle。它不创建、
    接收或消费 RNG。
    """
    if not isinstance(bundle, WaveformBundle):
        raise TypeError("bundle must be a WaveformBundle")
    runtime_context = (
        PostMixContext({})
        if context is None
        else context
        if isinstance(context, PostMixContext)
        else PostMixContext(context)
    )
    initial_state = PostMixState._from_frozen_waveforms(bundle.tracks)
    trace = _run_post_mix_pipeline_impl(
        initial_state,
        adapter,
        retain_stage_states=retain_stage_states,
    )
    final_bundle = WaveformBundle(trace.final_state.waveforms, metadata=bundle.metadata)
    return PostMixResult(
        initial_bundle=bundle,
        bundle=final_bundle,
        context=runtime_context,
        trace=trace,
    )


def run_post_mix_orchestration(
    bundle: WaveformBundle,
    adapter: TaskPostMixAdapter,
    *,
    context: PostMixContext | Mapping[str, object] | None = None,
    retain_stage_states: bool = True,
) -> PostMixResult:
    """兼容入口；统一转发到 :class:`PostMixEngine`。"""
    return _POST_MIX_ENGINE.run_bundle(
        bundle,
        adapter,
        context=context,
        retain_stage_states=retain_stage_states,
    )


__all__ = [
    "POST_MIX_STAGE_ORDER",
    "PostMixEngine",
    "PostMixContext",
    "PostMixResult",
    "PostMixStage",
    "PostMixStageResult",
    "PostMixState",
    "PostMixTrace",
    "TaskPostMixAdapter",
    "run_post_mix_orchestration",
    "run_post_mix_pipeline",
]
