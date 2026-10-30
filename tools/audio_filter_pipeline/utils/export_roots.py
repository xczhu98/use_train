"""配置与导出器共用的 export-root 纯路径边界函数。"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Optional


def canonical_boundary_path(value: str, field_name: str) -> str:
    """返回不要求路径存在的 canonical absolute path。"""
    try:
        if type(value) is not str or not value.strip():
            raise ValueError
        return os.path.realpath(os.path.abspath(value))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} path cannot be canonicalized") from exc


def boundary_paths_overlap(left: str, right: str) -> bool:
    """判断 canonical 路径相同或一方为另一方祖先。"""
    try:
        common = os.path.commonpath((left, right))
    except (OSError, ValueError) as exc:
        raise ValueError("configured paths cannot be compared safely") from exc
    return common == left or common == right


def resolve_export_root(output_root: str, export_root: Optional[str]) -> Path:
    """解析 exporter 实际根；不创建目录、不读取文件内容。"""
    canonical_output = canonical_boundary_path(output_root, "output_root")
    if export_root is None:
        return Path(canonical_output) / "audio_exports"
    canonical_export = canonical_boundary_path(export_root, "export_root")
    if boundary_paths_overlap(canonical_output, canonical_export):
        raise ValueError(
            "explicit export_root must not overlap output_root; use an isolated sibling root"
        )
    return Path(canonical_export)


def validate_export_root_boundaries(
    output_root: Optional[str],
    dataset_input_roots: Iterable[tuple[str, str]],
    export_roots: Mapping[str, Optional[str]],
) -> None:
    """拒绝 artifact、source 与导出根之间的危险路径布局。"""
    canonical_output = (
        canonical_boundary_path(output_root, "output_root")
        if output_root is not None else None
    )
    canonical_inputs = [
        (name, canonical_boundary_path(root, name))
        for name, root in dataset_input_roots
    ]
    if canonical_output is not None:
        for input_name, canonical_input in canonical_inputs:
            if boundary_paths_overlap(canonical_output, canonical_input):
                raise ValueError(
                    f"output_root conflicts with {input_name}: paths must not overlap"
                )

    canonical_exports = [
        (name, canonical_boundary_path(root, name))
        for name, root in export_roots.items()
        if root is not None
    ]
    protected = list(canonical_inputs)
    if canonical_output is not None:
        protected.insert(0, ("output_root", canonical_output))
    for export_name, canonical_export in canonical_exports:
        for protected_name, canonical_protected in protected:
            if boundary_paths_overlap(canonical_export, canonical_protected):
                raise ValueError(
                    f"{export_name} conflicts with {protected_name}: "
                    "paths must not be equal or ancestor/descendant"
                )

    if canonical_output is not None:
        default_root = str(Path(canonical_output) / "audio_exports")
        for input_name, canonical_input in canonical_inputs:
            if boundary_paths_overlap(default_root, canonical_input):
                raise ValueError(
                    "default export root <output_root>/audio_exports conflicts with "
                    f"{input_name}: paths must not overlap"
                )

    for index, (left_name, left_root) in enumerate(canonical_exports):
        for right_name, right_root in canonical_exports[index + 1:]:
            if boundary_paths_overlap(left_root, right_root):
                raise ValueError(
                    f"{left_name} conflicts with {right_name}: "
                    "paths must not be equal or ancestor/descendant"
                )
