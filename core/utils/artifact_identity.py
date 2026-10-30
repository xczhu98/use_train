"""共享的 artifact identity 与 source binding 深模块。

本模块定义跨推理和 ONNX 发布复用的底层身份语义。上层 manifest 可以拥有不同
schema，但 source binding 的 canonical fingerprint、文件复核和失败语义必须一致。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.utils.inference_artifacts import (
    _validate_content_fingerprint,
    build_content_fingerprint,
    fingerprint_file,
)


def _json_snapshot(value: Any, active: frozenset[int] = frozenset()) -> Any:
    if value is None or type(value) in (str, int, float, bool):
        return value
    if id(value) in active:
        raise ValueError("source binding JSON must not contain cycles")
    active = active | {id(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("source binding JSON keys must be strings")
        return {key: _json_snapshot(item, active) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_snapshot(item, active) for item in value]
    raise ValueError(f"source binding contains non-JSON value: {type(value).__name__}")


@dataclass(frozen=True)
class ArtifactIdentity:
    """一个已冻结的普通文件内容身份。"""

    path: str
    size: int
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> "ArtifactIdentity":
        value = fingerprint_file(path)
        return cls(path=str(value["path"]), size=int(value["size"]), sha256=str(value["sha256"]))

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True, init=False)
class SourceBinding:
    """具名来源的不可变 JSON 快照；构造时校验原聚合 SHA，不重新绑定文件。

    ``value`` 和 ``as_dict()`` 每次返回深隔离的普通 JSON 对象。文件当前内容
    由 ``verify()`` 显式复核，预计算身份因此可以保留实际 loader 读取的旧内容。
    """

    _value_json: bytes

    def __init__(self, value: Mapping[str, Any]) -> None:
        snapshot = _json_snapshot(value)
        _validate_content_fingerprint(snapshot, "source binding")
        serialized = json.dumps(
            snapshot, allow_nan=False, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _validate_content_fingerprint(json.loads(serialized), "source binding")
        object.__setattr__(self, "_value_json", serialized)

    @property
    def value(self) -> dict[str, Any]:
        return self.as_dict()

    @classmethod
    def from_fingerprint(cls, value: Mapping[str, Any]) -> "SourceBinding":
        """校验并冻结已有完整 fingerprint，保留原始 SHA 与 None 文件条目。"""
        return cls(value)

    @classmethod
    def bind(
        cls,
        files: Mapping[str, str | Path | None],
        *,
        parameters: Mapping[str, Any] | None = None,
        precomputed: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> "SourceBinding":
        if parameters is not None and not isinstance(parameters, Mapping):
            raise ValueError("source binding parameters must be a mapping")
        value = build_content_fingerprint(
            files, parameters=_json_snapshot(parameters) if parameters is not None else None,
            precomputed=precomputed,
        )
        for name in precomputed or ():
            declared = files[name]
            if declared is None or str(Path(declared).expanduser().resolve()) != value["files"][name]["path"]:
                raise ValueError(f"precomputed identity does not match declared source: {name!r}")
        return cls(value)

    def as_dict(self) -> dict[str, Any]:
        return json.loads(self._value_json)

    def verify(self, *, names: tuple[str, ...] | None = None) -> None:
        snapshot = self.as_dict()
        _validate_content_fingerprint(snapshot, "source binding")
        files = snapshot["files"]
        selected = names or tuple(files)
        for name in selected:
            if name not in files:
                raise ValueError(f"unknown source artifact: {name!r}")
            expected = files[name]
            if expected is None:
                continue
            current = ArtifactIdentity.from_path(expected["path"]).as_dict()
            if current != expected:
                raise RuntimeError(f"source artifact changed: {name}")


def bind_source_files(
    files: Mapping[str, str | Path | None],
    *,
    parameters: Mapping[str, Any] | None = None,
    precomputed: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """绑定 source 文件并返回历史 manifest 使用的 mapping。"""
    return SourceBinding.bind(files, parameters=parameters, precomputed=precomputed).as_dict()


def verify_source_binding(
    binding: Mapping[str, Any], *, names: tuple[str, ...] | None = None
) -> None:
    """复核 source binding，文件缺失、变化或结构错误均 fail-closed。"""
    SourceBinding(binding).verify(names=names)
