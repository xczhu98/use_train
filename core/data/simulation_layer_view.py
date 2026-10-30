"""``simulation_by_layer`` 的规范化视图：A–F 层与聚合事实的单一事实来源。

生产端（SE/AEC dynamic dataset 的 ``simulation_by_layer`` 构造）在同一个 dict 中
同时写入 A–F 分层事实与两个**聚合事实**：

- ``final_peak_normalize``：final-guard 事实；
- ``realized``：post-Layer-F 实测电平/峰值/削波比例。

verify 工具等消费者按「``simulation_by_layer`` 只含 A–F、聚合事实在 stems 顶层」
的约定读取。两侧曾在没有耦合测试的情况下漂移；本模块提供统一的拆分与提升语义，
producer 与 consumer 都必须经由这里解释 ``simulation_by_layer``。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LAYER_KEYS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
AGGREGATE_FACT_KEYS: tuple[str, ...] = ("final_peak_normalize", "realized")


def split_simulation_by_layer(
    by_layer: Mapping[str, Any],
    *,
    context: str = "simulation_by_layer",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """拆分为 ``(A–F 层视图, 聚合事实视图)``；未知键 fail-closed。

    A–F 与两个聚合键之外的任何键都视为 producer/consumer 键集合漂移，直接报错，
    不允许静默忽略。
    """
    if not isinstance(by_layer, Mapping):
        raise RuntimeError(f"{context} must be a mapping")
    layers: dict[str, Any] = {}
    aggregates: dict[str, Any] = {}
    for key, value in by_layer.items():
        if key in LAYER_KEYS:
            layers[key] = value
        elif key in AGGREGATE_FACT_KEYS:
            aggregates[key] = value
        else:
            raise RuntimeError(
                f"{context} must contain only Layers A-F plus "
                f"{list(AGGREGATE_FACT_KEYS)}; unexpected={key!r}"
            )
    return layers, aggregates


def resolve_aggregate_facts(
    aggregates: Mapping[str, Any],
    stems: Mapping[str, Any] | None = None,
    *,
    context: str = "simulation_by_layer",
) -> dict[str, Any]:
    """合并聚合事实：``by_layer`` 内嵌键优先，回落 ``stems`` 顶层（兼容历史产物）。

    非 Mapping 的取值一律报错（fail-closed），避免下游以错误类型继续渲染。
    """
    resolved: dict[str, Any] = {}
    for key in AGGREGATE_FACT_KEYS:
        value = aggregates.get(key)
        if value is None and stems is not None:
            value = stems.get(key)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{context}.{key} must be a mapping")
        resolved[key] = value
    return resolved


def canonical_simulation_view(
    stems: Mapping[str, Any],
    *,
    context: str = "simulation_by_layer",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """返回 ``(A–F 层视图, 聚合事实视图, canonical stems)``。

    ``canonical stems`` 是 ``stems`` 的浅拷贝并把聚合事实提升到顶层，供按顶层键
    读取的既有消费方直接使用；``simulation_by_layer`` 原样保留在副本中（consumer
    应使用返回的层视图做 A–F 校验）。
    """
    raw = stems.get(context)
    layers, nested = split_simulation_by_layer(raw, context=context)
    aggregates = resolve_aggregate_facts(nested, stems, context=context)
    canonical = dict(stems)
    canonical.update(aggregates)
    return layers, aggregates, canonical
