"""空间 observation channel dispatcher for shared and per-channel effects."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

import numpy as np

from core.data.post_augment import apply_post_augmentations
from core.data.simulation.contracts import LayerEOperationResult, LayerEResult


@dataclass(frozen=True)
class ChannelDispatchResult:
    """逐 channel 执行结果与保留给任务侧的事实。"""

    waveform: np.ndarray
    channel_results: tuple[Any, ...]
    channel_facts: tuple[dict[str, Any], ...]
    policy: str


@dataclass(frozen=True)
class ChannelPostAugmentationResult:
    """空间 Layer F 的波形、逐 channel 操作和聚合 clipping facts。"""

    waveform: np.ndarray
    channel_facts: tuple[dict[str, Any], ...]
    applied_ops: tuple[str, ...]
    clipping_facts: dict[str, Any]


def _state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    return left == right


def _clone_rng(rng: np.random.Generator) -> np.random.Generator:
    bit_generator = type(rng.bit_generator)()
    bit_generator.state = copy.deepcopy(rng.bit_generator.state)
    return np.random.Generator(bit_generator)


def _channel_waveform(result):
    """Layer E 返回 typed result，Layer F 返回显式 waveform/facts 对。"""
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return np.asarray(result[0])
    return np.asarray(getattr(result, 'waveform', result))


def _channel_facts(result):
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return dict(result[1])
    return result.to_metadata() if hasattr(result, 'to_metadata') else {}


def dispatch_observation_channels(
    observation: np.ndarray,
    *,
    policy: str,
    rng: np.random.Generator,
    apply_channel: Callable[[np.ndarray, np.random.Generator, int], Any],
) -> ChannelDispatchResult:
    """以显式随机计划对 ``[C,T]`` observation 执行逐 channel callback。

    ``shared`` 为每路 callback 提供同一初始 RNG 状态，并要求所有 callback
    消费相同数量的随机数；调用方 RNG 只前进一次。``per_channel`` 使用调用方
    RNG 顺序消费，每路可以拥有独立参数。失败不会把 shared 的局部状态泄漏到
    调用方，也不会吞掉 callback 异常。
    """
    value = np.asarray(observation)
    if value.ndim not in (1, 2) or value.shape[-1] < 1:
        raise ValueError(f"observation must be [T] or [C,T], got {value.shape}")
    if policy not in {"shared", "per_channel"}:
        raise ValueError("channel policy must be 'shared' or 'per_channel'")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")

    channels = value.reshape(1, -1) if value.ndim == 1 else value
    results: list[Any] = []
    terminal_states: list[Any] = []
    for index, channel in enumerate(channels):
        local_rng = rng
        if policy == "shared":
            local_rng = _clone_rng(rng)
        result = apply_channel(channel, local_rng, index)
        waveform = _channel_waveform(result)
        if waveform.ndim != 1 or waveform.shape != channel.shape:
            raise ValueError(
                f"channel callback must preserve [T] shape, got {waveform.shape}"
            )
        results.append(result)
        if policy == "shared":
            terminal_states.append(copy.deepcopy(local_rng.bit_generator.state))

    if policy == "shared":
        if terminal_states and not all(
            _state_equal(terminal_states[0], state) for state in terminal_states[1:]
        ):
            raise RuntimeError(
                "shared channel effect consumed different RNG counts across channels; "
                "use an explicit parameter plan or per_channel policy"
            )
        if terminal_states:
            rng.bit_generator.state = terminal_states[0]

    outputs = np.stack(
        [_channel_waveform(result).astype(np.float32, copy=False) for result in results],
        axis=0,
    )
    facts = tuple(
        _channel_facts(result)
        for result in results
    )
    return ChannelDispatchResult(
        waveform=outputs[0] if value.ndim == 1 else outputs,
        channel_results=tuple(results),
        channel_facts=facts,
        policy=policy,
    )


def aggregate_layer_e_results(
    dispatch: ChannelDispatchResult,
) -> LayerEResult:
    """将逐 channel Layer E 结果合并为兼容的 ``LayerEResult``。

    聚合字段表示任一路实际发生的 operation；精确的逐 channel 事实由
    ``dispatch.channel_facts`` 保留，供 Layer F 的 skip gate 消费。
    """
    results = dispatch.channel_results
    if not results or not all(isinstance(result, LayerEResult) for result in results):
        raise TypeError("Layer E channel results must contain LayerEResult values")
    first = results[0]
    names = tuple(
        dict.fromkeys(
            operation.name
            for result in results
            for operation in result.operations
        )
    )
    def operation_reason(name: str) -> str | None:
        entries = [
            operation
            for result in results
            for operation in result.operations
            if operation.name == name
        ]
        if len({operation.applied for operation in entries}) > 1:
            return "channel_dependent"
        return next(
            (operation.reason for operation in entries if operation.reason is not None),
            None,
        )

    operations = tuple(
        LayerEOperationResult(
            name=name,
            applied=any(
                operation.name == name and operation.applied
                for result in results
                for operation in result.operations
            ),
            observation_changed=any(
                operation.name == name and operation.observation_changed
                for result in results
                for operation in result.operations
            ),
            reason=operation_reason(name),
        )
        for name in names
    )
    skip_reasons = {result.skip_reason for result in results}
    return replace(
        first,
        waveform=dispatch.waveform,
        enabled=any(result.enabled for result in results),
        num_augmentations_drawn=max(result.num_augmentations_drawn for result in results),
        selected=tuple(dict.fromkeys(name for result in results for name in result.selected)),
        operations=operations,
        observation_changed=any(result.observation_changed for result in results),
        skip_reason=next(iter(skip_reasons)) if len(skip_reasons) == 1 else None,
    )


def apply_post_augmentations_channels(
    observation: np.ndarray,
    sr: int,
    cfg: Any,
    *,
    policy: str,
    rng: np.random.Generator,
    layer_e_results: Sequence[LayerEResult],
    sanitize_nonfinite: bool = False,
    apply_fn: Callable[..., np.ndarray] = apply_post_augmentations,
) -> ChannelPostAugmentationResult:
    """执行空间 Layer F，并按 channel 消费 Layer E skip gates。

    ``sanitize_nonfinite`` 仅由 lenient dataset contract 显式开启；strict
    caller 保留非有限输出，交给 post-finite/state contract fail-fast。

    ``shared`` 下任一路 Layer E 命中 clipping/codec 都会让该 skip gate 作用于
    全部 channel，避免某一路少消费一项随机参数而破坏共享随机计划；需要独立
    gate 时使用 ``per_channel``。
    """
    value = np.asarray(observation)
    channels = 1 if value.ndim == 1 else value.shape[0]
    shared_skip_clip = policy == "shared" and any(
        result.operation_applied("clipping") for result in layer_e_results
    )
    shared_skip_codec = policy == "shared" and any(
        result.operation_applied("codec") for result in layer_e_results
    )
    if len(layer_e_results) != channels:
        raise ValueError(
            f"Layer E channel result count {len(layer_e_results)} does not match {channels}"
        )

    def apply_channel(channel, channel_rng, index):
        layer_e = layer_e_results[index]
        skip_clip = shared_skip_clip or layer_e.operation_applied("clipping")
        skip_codec = shared_skip_codec or layer_e.operation_applied("codec")
        operations: list[str] = []
        clipping: dict[str, Any] = {}
        waveform = apply_fn(
            channel,
            sr,
            cfg,
            skip_clip=skip_clip,
            skip_codec=skip_codec,
            rng=channel_rng,
            applied_ops_out=operations,
            clipping_facts_out=clipping,
            sanitize_nonfinite=sanitize_nonfinite,
        )
        if skip_clip or clipping:
            clipping.update(
                {
                    "event": False if skip_clip else True,
                    "applied": False if skip_clip else bool(clipping.get("applied", False)),
                    "actual_changed": False if skip_clip else bool(clipping.get("applied", False)),
                    "skipped": skip_clip,
                    "skip_reason": "layer_e_clipping_applied" if skip_clip else clipping.get("skip_reason"),
                }
            )
        return np.asarray(waveform, dtype=np.float32), {
            "operations": list(operations),
            "clipping": clipping,
            "skip_clip": skip_clip,
            "skip_codec": skip_codec,
        }

    dispatch = dispatch_observation_channels(
        value,
        policy=policy,
        rng=rng,
        apply_channel=apply_channel,
    )
    channel_facts = dispatch.channel_facts
    if value.ndim == 1:
        clipping = dict(channel_facts[0].get("clipping", {}))
    else:
        clipping = {
            f"channel_{index}": dict(facts.get("clipping", {}))
            for index, facts in enumerate(channel_facts)
        }
        clipping.update(
            {
                "event": any(bool(facts.get("clipping", {}).get("event", False)) for facts in channel_facts),
                "applied": any(bool(facts.get("clipping", {}).get("applied", False)) for facts in channel_facts),
                "actual_changed": any(bool(facts.get("clipping", {}).get("actual_changed", False)) for facts in channel_facts),
                "skipped": all(bool(facts.get("skip_clip", False)) for facts in channel_facts),
                "skip_reason": "layer_e_clipping_applied" if all(bool(facts.get("skip_clip", False)) for facts in channel_facts) else None,
            }
        )
    applied_ops = tuple(
        dict.fromkeys(
            operation
            for facts in channel_facts
            for operation in facts.get("operations", ())
        )
    )
    return ChannelPostAugmentationResult(
        waveform=dispatch.waveform,
        channel_facts=channel_facts,
        applied_ops=applied_ops,
        clipping_facts=clipping,
    )


__all__ = [
    "ChannelDispatchResult",
    "ChannelPostAugmentationResult",
    "aggregate_layer_e_results",
    "apply_post_augmentations_channels",
    "dispatch_observation_channels",
]
