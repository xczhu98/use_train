"""SE multi-root offline index generation helpers.

本模块只构造 schema v2 GenerationManifest 与 root markers；目录事务由
``offline_publication`` 统一执行。
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Iterable, Mapping

from tools.prepare_data_tools.generation_manifest import (
    GENERATION_ROOT_MARKER,
    MANIFEST_NAME,
    GenerationManifest,
    GenerationManifestError,
    sha256_file,
    write_generation_root_marker,
)
from tools.prepare_data_tools.offline_publication import OWNER_FILE

INDEX_ROOT_ALIAS = "indexes"
WAVEFORM_ROOT_ALIAS = "waveforms"


def alias_relative_waveform_path(waveform_root: Path, path: Path) -> str:
    """把 waveform 路径编码为受控 alias-relative SCP 字段。"""
    root = waveform_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"waveform path escapes --waveform-root: {resolved}") from exc
    return f"{WAVEFORM_ROOT_ALIAS}:{relative}"


def read_existing_generation_identity(root: Path) -> tuple[str, str]:
    """只读取精确 shared ownership/generation markers 的 owner 与 generation id。"""
    owner = _read_regular_json(root / OWNER_FILE, "ownership marker")
    if set(owner) != {"schema_version", "owner_id"} or owner.get("schema_version") != 1:
        raise GenerationManifestError("shared ownership marker schema 非法")
    owner_id = owner.get("owner_id")
    if not isinstance(owner_id, str) or not owner_id:
        raise GenerationManifestError("shared ownership marker 缺少 owner_id")

    generation = _read_regular_json(root / GENERATION_ROOT_MARKER, "generation marker")
    expected_fields = {
        "schema_version",
        "owner_id",
        "generation_id",
        "manifest_digest",
        "root_alias",
        "resource_digest",
    }
    if set(generation) != expected_fields or generation.get("schema_version") != 1:
        raise GenerationManifestError("generation marker schema 非法")
    if generation.get("owner_id") != owner_id:
        raise GenerationManifestError("ownership/generation marker owner_id 不一致")
    if generation.get("root_alias") != WAVEFORM_ROOT_ALIAS:
        raise GenerationManifestError("generation marker root_alias 不是 waveforms")
    generation_id = generation.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise GenerationManifestError("generation marker 缺少 generation_id")
    return owner_id, generation_id


def build_se_index_manifest(
    *,
    index_root: Path,
    waveform_root: Path,
    generation_id: str,
    generator_id: str,
    generator_version: str,
    split_scps: Mapping[str, tuple[str, str]],
    waveform_paths: Iterable[Path],
    parameters: Mapping[str, object] | None = None,
) -> GenerationManifest:
    """按实际 SCP/waveform 文件构造稳定 schema v2 manifest。"""
    index_root = index_root.resolve(strict=True)
    waveform_root = waveform_root.resolve(strict=True)
    inventory: list[dict[str, object]] = []
    required_gates: list[dict[str, object]] = []
    published = 0

    for split, (clean_relative, noisy_relative) in sorted(split_scps.items()):
        clean_path = index_root / clean_relative
        noisy_path = index_root / noisy_relative
        clean_count = _count_scp_records(clean_path)
        noisy_count = _count_scp_records(noisy_path)
        if clean_count <= 0 or clean_count != noisy_count:
            raise ValueError(
                f"split {split!r} SCP count mismatch: clean={clean_count}, noisy={noisy_count}"
            )
        published += clean_count
        inventory.extend(
            (
                _inventory_entry(INDEX_ROOT_ALIAS, index_root, clean_relative, "metadata"),
                _inventory_entry(INDEX_ROOT_ALIAS, index_root, noisy_relative, "metadata"),
            )
        )
        required_gates.append(
            {
                "name": f"{split}-reload",
                "adapter": "se-offline-scp-v1",
                "split": split,
                "inputs": {
                    "clean_scp": {
                        "root_alias": INDEX_ROOT_ALIAS,
                        "path": clean_relative,
                    },
                    "noisy_scp": {
                        "root_alias": INDEX_ROOT_ALIAS,
                        "path": noisy_relative,
                    },
                },
                "options": {"strict_load": True},
            }
        )

    relative_waveforms: set[str] = set()
    for path in waveform_paths:
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(waveform_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"waveform path escapes waveform root: {resolved}") from exc
        relative_waveforms.add(relative)
    if not relative_waveforms:
        raise ValueError("waveform inventory must not be empty")
    inventory.extend(
        _inventory_entry(WAVEFORM_ROOT_ALIAS, waveform_root, relative, "waveform")
        for relative in sorted(relative_waveforms)
    )

    return GenerationManifest.from_dict(
        {
            "schema_version": 2,
            "generation_id": generation_id,
            "status": "complete",
            "artifact_kind": "se-multi-root-offline",
            "task": "se",
            "generator": {"id": generator_id, "version": generator_version},
            "root_bindings": [
                {"alias": INDEX_ROOT_ALIAS, "role": "manifest"},
                {"alias": WAVEFORM_ROOT_ALIAS, "role": "resource"},
            ],
            "splits": sorted(split_scps),
            "parameters": dict(parameters or {}),
            "source_identities": [],
            "inventory": inventory,
            "counts": {
                "requested": published,
                "published": published,
                "failed": 0,
                "dropped": 0,
            },
            "required_gates": required_gates,
        }
    )


def write_manifest_and_root_markers(
    *,
    index_root: Path,
    waveform_root: Path,
    manifest: GenerationManifest,
    owner_id: str,
    mark_waveform_root: bool,
) -> None:
    """写 manifest 与 indexes marker；完整生成时同时标记 waveform staging。"""
    (index_root / MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
    write_generation_root_marker(
        index_root,
        manifest=manifest,
        root_alias=INDEX_ROOT_ALIAS,
        owner_id=owner_id,
    )
    if mark_waveform_root:
        write_generation_root_marker(
            waveform_root,
            manifest=manifest,
            root_alias=WAVEFORM_ROOT_ALIAS,
            owner_id=owner_id,
        )


def _inventory_entry(alias: str, root: Path, relative: str, kind: str) -> dict[str, object]:
    path = root / relative
    return {
        "root_alias": alias,
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "kind": kind,
    }


def _count_scp_records(path: Path) -> int:
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _read_regular_json(path: Path, location: str) -> dict[str, object]:
    try:
        path_stat = path.lstat()
        if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink != 1:
            raise GenerationManifestError(f"{location} 必须是单链接普通文件: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except GenerationManifestError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GenerationManifestError(f"无法读取 {location}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GenerationManifestError(f"{location} 必须是 JSON object")
    return payload


__all__ = [
    "INDEX_ROOT_ALIAS",
    "WAVEFORM_ROOT_ALIAS",
    "alias_relative_waveform_path",
    "build_se_index_manifest",
    "read_existing_generation_identity",
    "write_manifest_and_root_markers",
]
