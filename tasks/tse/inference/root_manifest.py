"""顶层 TSE 文件推理 manifest。

根入口使用一个严格 JSONL manifest 同时描述 mixture、模型声明的文件 cue 和
inline cue。输出路径仍由公共 ``inference.py``/``InferenceSpec`` 生成，避免把
manifest 解析、输出锁或复用逻辑塞进任务 adapter。

每行格式为：

``{"uid": "utt", "mixture": "mix.wav", "speaker_waveform": "enroll.wav", "attributes": {}}``

相对路径按 manifest 所在目录解析；``attributes.cues`` 用于 inline
``speaker_embedding``/``spatial``/``visual``/``textual`` cue。文件 cue 必须使用
模型 IO 合同中的 canonical role，``enrollment``/``enroll`` 不再接受。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


_FILE_CUE_ROLES = frozenset(
    {"speaker_waveform", "speaker_embedding", "spatial", "visual", "textual"}
)
_ALLOWED_FIELDS = frozenset({"uid", "mixture", "attributes", *_FILE_CUE_ROLES})


@dataclass(frozen=True)
class TSERootManifestItem:
    """一个已完成路径解析的根推理输入条目。"""

    uid: str
    inputs: Mapping[str, Path]
    attributes: Mapping[str, Any]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"TSE root manifest object contains duplicate key: {key!r}")
        value[key] = item
    return value


def _validate_uid(uid: Any, *, source: str) -> None:
    if (
        not isinstance(uid, str)
        or not uid
        or uid != uid.strip()
        or any(char.isspace() for char in uid)
        or uid in {".", ".."}
        or "\x00" in uid
        or "/" in uid
        or "\\" in uid
        or Path(uid).name != uid
    ):
        raise ValueError(f"{source} contains an invalid uid: {uid!r}")


def _resolve_input_path(
    raw: Any,
    *,
    manifest_path: Path,
    root_dir: Path | None,
    source: str,
) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{source} must be a non-empty path string")
    raw_path = Path(raw).expanduser()
    candidate = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    if candidate.is_symlink():
        raise ValueError(f"{source} must not be a symbolic link: {candidate}")
    resolved = candidate.resolve(strict=False)
    if root_dir is not None:
        try:
            resolved.relative_to(root_dir)
        except ValueError as exc:
            raise ValueError(
                f"{source} escapes --scp_root_dir {root_dir}: {candidate}"
            ) from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"{source} does not exist or is not a regular file: {candidate}")
    return candidate


def read_tse_root_manifest(
    path: str | Path,
    *,
    root_dir: str | Path | None = None,
) -> tuple[TSERootManifestItem, ...]:
    """读取并严格校验顶层 TSE JSONL manifest。"""

    manifest_path = Path(path).expanduser().resolve(strict=True)
    if not manifest_path.is_file():
        raise ValueError(f"TSE root manifest is not a regular file: {manifest_path}")
    root = Path(root_dir).expanduser().resolve(strict=True) if root_dir else None
    if root is not None and not root.is_dir():
        raise NotADirectoryError(f"--scp_root_dir is not a directory: {root}")

    entries: list[TSERootManifestItem] = []
    seen: set[str] = set()
    with manifest_path.open(encoding="utf-8") as stream:
        for lineno, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            source = f"TSE root manifest {manifest_path}:{lineno}"
            try:
                row = json.loads(line, object_pairs_hook=_strict_object)
            except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{source} is not strict JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{source} must be a JSON object")
            unknown = sorted(set(row) - _ALLOWED_FIELDS)
            if unknown:
                raise ValueError(f"{source} has unknown fields: {unknown}")
            if "uid" not in row or "mixture" not in row:
                raise ValueError(f"{source} requires uid and mixture")
            uid = row["uid"]
            _validate_uid(uid, source=source)
            if uid in seen:
                raise ValueError(f"{source} repeats uid: {uid!r}")
            seen.add(uid)
            inputs: dict[str, Path] = {
                "mixture": _resolve_input_path(
                    row["mixture"],
                    manifest_path=manifest_path,
                    root_dir=root,
                    source=f"{source}.mixture",
                )
            }
            for role in sorted(_FILE_CUE_ROLES):
                if role in row:
                    inputs[role] = _resolve_input_path(
                        row[role],
                        manifest_path=manifest_path,
                        root_dir=root,
                        source=f"{source}.{role}",
                    )
            attributes = row.get("attributes", {})
            if not isinstance(attributes, Mapping):
                raise TypeError(f"{source}.attributes must be a JSON object")
            entries.append(
                TSERootManifestItem(
                    uid=uid,
                    inputs=dict(inputs),
                    attributes=dict(attributes),
                )
            )
    if not entries:
        raise ValueError(f"TSE root manifest contains no input items: {manifest_path}")
    return tuple(entries)


__all__ = ["TSERootManifestItem", "read_tse_root_manifest"]
