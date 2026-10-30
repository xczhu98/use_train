"""任务无关的 Layer B 场景描述。

该模块只描述场景合同，不负责抽样、渲染或改变任何现有任务行为。场景
名称沿用配置中的 canonical ``prob_*`` 名称；``source_count`` 表示实际
进入该场景的 speech source 数量，不包含 latent reference、enrollment 或
echo。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, Mapping


_KNOWN_TASKS = frozenset({"se", "aec", "tse", "ss"})
_REQUIRED_FIELDS = frozenset(
    {
        "scene_name",
        "target_present",
        "interference_present",
        "background_present",
        "echo_present",
        "source_count",
    }
)
_ALL_FIELDS = _REQUIRED_FIELDS | {"task"}


@dataclass(frozen=True, slots=True)
class SceneSpec:
    """一个可序列化、可校验的 Layer B 场景合同。

    ``target_present`` 表示任务定义的 designated target 或监督 source 集合是否
    实际出现在场景中；它是 Layer B 存在性字段，不是 waveform track 名。在
    TSE 中它指注册目标，在 AEC 中指 near-end，在 SS 中指 K 路 speech
    source 集合。``interference_present`` 表示额外的非 target speech
    source，``background_present`` 表示 additive background，``echo_present``
    表示 AEC echo。``source_count`` 只计算实际 speech source。
    """

    scene_name: str
    target_present: bool
    interference_present: bool
    background_present: bool
    echo_present: bool
    source_count: int
    task: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scene_name, str) or not self.scene_name.strip():
            raise ValueError("scene_name must be a non-empty string")
        for field_name in (
            "target_present",
            "interference_present",
            "background_present",
            "echo_present",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise TypeError(f"{field_name} must be bool")
        if isinstance(self.source_count, bool) or not isinstance(self.source_count, Integral):
            raise TypeError("source_count must be a non-negative integer")
        if int(self.source_count) < 0:
            raise ValueError("source_count must be non-negative")
        if self.task is not None:
            if not isinstance(self.task, str) or self.task not in _KNOWN_TASKS:
                raise ValueError(f"task must be one of {sorted(_KNOWN_TASKS)} or None")
            from core.data.simulation.scene_names import validate_scene_name

            validate_scene_name(self.task, self.scene_name)
        if (self.target_present or self.interference_present) and self.source_count == 0:
            raise ValueError(
                "target_present/interference_present requires at least one speech source"
            )

    def to_dict(self) -> dict[str, Any]:
        """返回只含 JSON 原生标量的稳定字典。"""
        result = asdict(self)
        # dataclass 已经过严格类型校验；这里显式转换 Integral，避免 numpy
        # 整数在边界对象中泄漏，保证 json.dumps 可以直接处理。
        result["source_count"] = int(result["source_count"])
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SceneSpec":
        """从字典恢复并重新执行完整场景合同校验。"""
        if not isinstance(value, Mapping):
            raise TypeError("SceneSpec value must be a mapping")
        keys = set(value)
        missing = _REQUIRED_FIELDS - keys
        unknown = keys - _ALL_FIELDS
        if missing:
            raise ValueError(f"SceneSpec is missing fields: {sorted(missing)}")
        if unknown:
            raise ValueError(f"SceneSpec has unknown fields: {sorted(unknown)}")
        return cls(
            scene_name=value["scene_name"],
            target_present=value["target_present"],
            interference_present=value["interference_present"],
            background_present=value["background_present"],
            echo_present=value["echo_present"],
            source_count=value["source_count"],
            task=value.get("task"),
        )

    def to_json(self) -> str:
        """返回稳定排序的 JSON 文本，供 trace/manifest 边界使用。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "SceneSpec":
        """从 JSON 文本恢复场景，并拒绝非对象或非法字段。"""
        if not isinstance(value, str):
            raise TypeError("SceneSpec JSON value must be a string")
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("SceneSpec JSON is invalid") from exc
        return cls.from_dict(decoded)


__all__ = ["SceneSpec"]
