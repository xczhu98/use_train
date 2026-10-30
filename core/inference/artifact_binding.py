"""共享推理 artifact、配置与 checkpoint 身份绑定接口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.utils.artifact_identity import (
    SourceBinding,
    bind_source_files,
    verify_source_binding,
)
from core.utils.inference_artifacts import fingerprint_json


def bind_artifacts(
    files: Mapping[str, str | Path | None],
    *,
    parameters: Mapping[str, Any] | None = None,
    precomputed: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """绑定具名文件和可序列化运行参数的内容 fingerprint。"""
    return bind_source_files(files, parameters=parameters, precomputed=precomputed)


def bind_config(config: Mapping[str, Any]) -> dict[str, str]:
    """为已加载的 effective config 生成稳定 identity。"""
    if not isinstance(config, Mapping):
        raise TypeError("effective config 必须是 mapping")
    return {
        "schema": "effective-merged-config-v1",
        "algorithm": "sha256-json-v1",
        "sha256": fingerprint_json(config),
    }


def verify_file_bindings(
    bindings: Mapping[str, Any] | SourceBinding, *, names: tuple[str, ...] | None = None
) -> None:
    """复核已冻结 fingerprint 中指定文件仍保持原内容。"""
    if isinstance(bindings, SourceBinding):
        bindings.verify(names=names)
    else:
        verify_source_binding(bindings, names=names)
