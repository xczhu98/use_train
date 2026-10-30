"""共享推理接口的数据合同。

任务 adapter 只需要实现 :class:`InferenceAdapter` 所需的生命周期方法；本模块不
依赖任何具体任务、模型或音频库。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from core.utils.artifact_identity import SourceBinding
from core.utils.inference_artifacts import canonical_json_value

JSONValue = Any


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class ArtifactBinding:
    """调用方声明会影响推理结果的文件和 JSON 参数。"""

    files: Mapping[str, str | Path | None]
    parameters: Mapping[str, JSONValue] = field(default_factory=dict)
    _source_binding: SourceBinding | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.files, Mapping) or not self.files:
            raise ValueError("artifact files must be a non-empty mapping")
        if any(not isinstance(name, str) or not name for name in self.files):
            raise ValueError("artifact names must be non-empty strings")
        object.__setattr__(self, "files", _freeze_mapping(self.files))
        object.__setattr__(self, "parameters", _freeze_json(canonical_json_value(self.parameters)))

    @classmethod
    def from_fingerprint(cls, fingerprint: Mapping[str, Any]) -> "ArtifactBinding":
        """从已加载/已校验的 identity 构造，绝不按当前路径重新 fingerprint。"""
        source_binding = SourceBinding.from_fingerprint(fingerprint)
        value = source_binding.as_dict()
        instance = cls(
            files={
                name: entry["path"] if entry is not None else None
                for name, entry in value["files"].items()
            },
            parameters=value["parameters"],
        )
        object.__setattr__(instance, "_source_binding", source_binding)
        return instance

    def source_binding(self) -> SourceBinding:
        binding = self._source_binding
        if binding is None:
            binding = SourceBinding.bind(dict(self.files), parameters=self.parameters)
            object.__setattr__(self, "_source_binding", binding)
        return binding


@dataclass(frozen=True)
class ReusePolicy:
    """公共 runner 的复用策略；旧 CLI 的 skip-existing 仍可映射到 enabled。

    ``on_rejection`` 决定复用被拒（manifest/指纹/输入不匹配）时的行为：
    ``rerun`` 记录原因并全部重推（默认），``fail`` 立即报错（fail-closed）。
    """

    enabled: bool = False
    on_rejection: str = "rerun"

    def __post_init__(self) -> None:
        if self.on_rejection not in {"rerun", "fail"}:
            raise ValueError("reuse rejection policy must be rerun or fail")


@dataclass(frozen=True)
class FileInferenceItem:
    """新文件推理合同：输入和发布目标分离。"""

    uid: str
    inputs: Mapping[str, str | Path | None]
    output: str | Path
    attributes: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.uid, str) or not self.uid:
            raise ValueError("inference item uid must be a non-empty string")
        if not isinstance(self.inputs, Mapping) or not self.inputs:
            raise ValueError("inference item inputs must be a non-empty mapping")
        if any(not isinstance(role, str) or not role for role in self.inputs):
            raise ValueError("inference input roles must be non-empty strings")
        normalized = canonical_json_value(self.attributes)
        object.__setattr__(self, "inputs", _freeze_mapping(self.inputs))
        object.__setattr__(self, "attributes", _freeze_json(normalized))


@dataclass(frozen=True)
class InferenceSpec:
    """面向调用方的不可变文件推理计划。"""

    config: str | Path
    output_dir: str | Path
    items: Sequence[FileInferenceItem]
    checkpoint: str | Path | None = None
    artifacts: ArtifactBinding | None = None
    reuse: ReusePolicy = field(default_factory=ReusePolicy)
    allow_partial: bool = False
    run_metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        items = tuple(self.items)
        if any(not isinstance(item, FileInferenceItem) for item in items):
            raise TypeError("InferenceSpec.items must contain FileInferenceItem values")
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "run_metadata", _freeze_json(canonical_json_value(self.run_metadata)))

@dataclass(frozen=True)
class PreparedOutput:
    """新 adapter 输出合同；发布路径由 FileInferenceItem.output 决定。"""

    uid: str
    writer: Callable[[Path], None]
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_json(canonical_json_value(self.metadata)))


class InferenceAdapter(Protocol):
    """任务 Adapter 只消费文件输入并返回待发布内容。"""

    def infer(self, item: FileInferenceItem) -> PreparedOutput:
        ...


@dataclass(frozen=True)
class InferenceReport:
    """单次文件推理运行报告。"""

    status: str
    results: tuple[Any, ...]
    manifest_path: Path
    scp_path: Path
