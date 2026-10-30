"""共享推理输入 inventory 接口。"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from core.utils.inference_artifacts import (
    build_input_inventory as _build_input_inventory,
    canonical_json_value,
    fingerprint_json,
    stat_audio_identity,
)


def build_inventory(items: Iterable[Any]) -> dict[str, Any]:
    """建立带严格 JSON attributes 的 ``stat-v1`` 输入身份 inventory。"""
    return _build_input_inventory(items)


def inventory_items(inventory: Mapping[str, Any]) -> tuple[str, ...]:
    """返回 inventory 中保持原顺序的 UID。"""
    items = inventory.get("items")
    if not isinstance(items, list):
        raise ValueError("input inventory.items 必须是 list")
    return tuple(item["uid"] for item in items)


def verify_inventory(inventory: Mapping[str, Any]) -> None:
    """重新 stat 所有输入，发现输入在运行期间变化时失败。"""
    if inventory.get("algorithm") != "stat-v1" or not isinstance(inventory.get("items"), list):
        raise ValueError("input inventory 结构非法")
    payload = {"algorithm": inventory["algorithm"], "items": inventory["items"]}
    if inventory.get("digest") != fingerprint_json(payload):
        raise ValueError("input inventory digest 与内容不一致")
    for item in inventory["items"]:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"uid", "inputs", "attributes"}
            or not isinstance(item.get("inputs"), Mapping)
            or not isinstance(item.get("attributes"), Mapping)
        ):
            raise ValueError("input inventory 条目结构非法")
        try:
            if canonical_json_value(item["attributes"]) != item["attributes"]:
                raise ValueError("attributes 非 canonical JSON")
        except (TypeError, ValueError) as exc:
            raise ValueError("input inventory attributes 非法") from exc
        for expected in item["inputs"].values():
            if expected is not None and stat_audio_identity(expected["path"]) != dict(expected):
                raise RuntimeError(f"输入在推理期间发生变化: {expected['path']}")
