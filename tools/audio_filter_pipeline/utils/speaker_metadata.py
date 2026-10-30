"""独立、不可变的 speaker metadata 解析器。"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Pattern, Tuple

from ..config import SpeakerMetadataSpec


_MAPPING_MAX_BYTES = 64 * 1024 * 1024
_MAPPING_FIELDS = frozenset({"source_relative_path", "speaker_id"})


def _absolute_path(value: Any, field_name: str) -> str:
    try:
        path = os.fspath(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a non-empty path string") from exc
    if type(path) is not str or not path.strip():
        raise ValueError(f"{field_name} must be a non-empty path string")
    return os.path.abspath(path)


def validate_source_relative_path(value: Any) -> str:
    """验证并返回规范化的 POSIX source-relative path。"""
    if type(value) is not str or not value:
        raise ValueError("source_relative_path must be a non-empty string")
    if "\x00" in value or "\\" in value or value.startswith("/"):
        raise ValueError(
            "source_relative_path must be a normalized relative POSIX path"
        )
    components = value.split("/")
    if any(component in ("", ".", "..") for component in components):
        raise ValueError(
            "source_relative_path must not contain empty, dot, or parent components"
        )
    return value


def _validate_speaker_id(value: Any) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(
            "speaker_id must be a non-empty string without leading or trailing whitespace"
        )
    return value


def _reject_json_constant(value: str):
    raise ValueError(f"invalid JSON constant: {value}")


def _reject_duplicate_json_fields(pairs):
    value = {}
    for key, nested in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key!r}")
        value[key] = nested
    return value


def _read_bounded_snapshot(fd: int) -> bytes:
    chunks = []
    total = 0
    while total <= _MAPPING_MAX_BYTES:
        chunk = os.read(fd, min(1024 * 1024, _MAPPING_MAX_BYTES + 1 - total))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
    raise ValueError(
        f"speaker mapping file exceeds {_MAPPING_MAX_BYTES} byte limit"
    )


def _mapping_lines(snapshot: bytes):
    if not snapshot:
        return ()
    lines = snapshot.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    return lines


def _parse_mapping_snapshot(snapshot: bytes) -> Mapping[str, str]:
    try:
        text_lines = [line.decode("utf-8") for line in _mapping_lines(snapshot)]
    except UnicodeDecodeError as exc:
        raise ValueError(f"speaker mapping file is not valid UTF-8: {exc}") from exc

    mapping = {}
    for line_number, line in enumerate(text_lines, 1):
        if not line:
            raise ValueError(
                f"speaker mapping file contains blank line {line_number}"
            )
        try:
            row = json.loads(
                line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_fields,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"invalid speaker mapping JSON on line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"speaker mapping line {line_number} must be a JSON object"
            )
        if set(row) != _MAPPING_FIELDS:
            raise ValueError(
                f"speaker mapping line {line_number} must contain exactly fields "
                f"{sorted(_MAPPING_FIELDS)!r}"
            )
        source_path = validate_source_relative_path(row["source_relative_path"])
        speaker_id = _validate_speaker_id(row["speaker_id"])
        if source_path in mapping:
            raise ValueError(
                f"duplicate source_relative_path in speaker mapping: {source_path!r}"
            )
        mapping[source_path] = speaker_id
    return MappingProxyType(mapping)


def _verify_opened_mapping_identity(
    fd: int,
    mapping_path: str,
    canonical_path: str,
    mapping_base_dir: str,
) -> os.stat_result:
    """确认 descriptor、当前路径与初始 canonical path 仍指向同一普通文件。"""
    opened = os.fstat(fd)
    try:
        current = os.stat(mapping_path, follow_symlinks=False)
    except OSError as exc:
        raise ValueError("speaker mapping path changed while opening") from exc
    if stat.S_ISLNK(current.st_mode):
        raise ValueError("speaker mapping file final target must not be a symlink")
    if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
        raise ValueError("speaker mapping path changed while opening")
    current_canonical_path = os.path.realpath(mapping_path)
    if os.path.commonpath((mapping_base_dir, current_canonical_path)) != mapping_base_dir:
        raise ValueError("opened speaker mapping path would escape mapping_base_dir")
    if current_canonical_path != canonical_path:
        raise ValueError("speaker mapping path changed while opening")

    descriptor_path = f"/proc/self/fd/{fd}"
    if os.path.exists(descriptor_path):
        opened_path = os.path.realpath(descriptor_path)
        if opened_path != canonical_path:
            raise ValueError(
                "opened speaker mapping path would escape or change within "
                "mapping_base_dir"
            )
    return opened


def _mapping_fstat_signature(metadata: os.stat_result) -> Tuple[int, ...]:
    """返回读取期间不得变化的完整 mapping 文件元数据。"""
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_blksize,
        metadata.st_blocks,
    )


def _open_mapping_snapshot(
    mapping_path: str, canonical_path: str, mapping_base_dir: str,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    flags |= nofollow
    try:
        fd = os.open(mapping_path, flags)
    except OSError as exc:
        if nofollow and exc.errno == errno.ELOOP:
            raise ValueError("speaker mapping file final target must not be a symlink") from exc
        raise ValueError(f"cannot open speaker mapping file {mapping_path!r}: {exc}") from exc

    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("speaker mapping file must be a regular file")
        if not nofollow and os.path.islink(mapping_path):
            raise ValueError("speaker mapping file final target must not be a symlink")
        before = _verify_opened_mapping_identity(
            fd, mapping_path, canonical_path, mapping_base_dir,
        )
        snapshot = _read_bounded_snapshot(fd)
        after = _verify_opened_mapping_identity(
            fd, mapping_path, canonical_path, mapping_base_dir,
        )
        if (_mapping_fstat_signature(before) != _mapping_fstat_signature(after)
                or len(snapshot) != before.st_size
                or len(snapshot) != after.st_size):
            raise ValueError("speaker mapping file changed while reading")
        return snapshot
    finally:
        os.close(fd)


def _resolve_mapping_path(
    mapping_path: Any, mapping_base_dir: Any,
) -> Tuple[str, str, str]:
    base = os.path.realpath(_absolute_path(mapping_base_dir, "mapping_base_dir"))
    try:
        raw_mapping_path = os.fspath(mapping_path)
    except TypeError as exc:
        raise ValueError("mapping_path must be a non-empty path string") from exc
    if type(raw_mapping_path) is not str or not raw_mapping_path.strip():
        raise ValueError("mapping_path must be a non-empty path string")

    candidate = (
        raw_mapping_path
        if os.path.isabs(raw_mapping_path)
        else os.path.join(base, raw_mapping_path)
    )
    open_path = os.path.abspath(candidate)
    canonical = os.path.realpath(open_path)
    try:
        contained = os.path.commonpath((base, canonical)) == base
    except ValueError:
        contained = False
    if not contained:
        raise ValueError("speaker mapping path would escape mapping_base_dir")
    return base, open_path, canonical


def _compile_regex(spec: SpeakerMetadataSpec) -> Pattern[str]:
    if type(spec.regex) is not str or not spec.regex.strip():
        raise ValueError("speaker regex must be a non-empty string")
    try:
        compiled = re.compile(spec.regex)
    except re.error as exc:
        raise ValueError(f"speaker regex must be compilable: {exc}") from exc
    if type(spec.regex_group) is not str or spec.regex_group not in compiled.groupindex:
        raise ValueError("speaker regex_group must name a group in regex")
    return compiled


@dataclass(frozen=True)
class SpeakerMetadataResolver:
    """基于冻结配置与 mapping snapshot 解析 source-relative speaker ID。"""

    spec: SpeakerMetadataSpec
    input_root: str
    mapping_base_dir: Optional[str] = None
    _compiled_regex: Optional[Pattern[str]] = field(
        default=None, init=False, repr=False, compare=False,
    )
    _mapping: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}),
        init=False,
        repr=False,
        compare=False,
    )
    _payload_items: Tuple[Tuple[str, Any], ...] = field(
        default=(), init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.spec, SpeakerMetadataSpec):
            raise TypeError("spec must be a SpeakerMetadataSpec")
        object.__setattr__(
            self, "input_root", _absolute_path(self.input_root, "input_root")
        )
        if type(self.spec.required) is not bool:
            raise ValueError("speaker required must be bool")

        source = self.spec.source
        if source == "none":
            if self.spec.required:
                raise ValueError("speaker source 'none' requires required=False")
            payload = {"source": source, "required": False}
        elif source == "path_component":
            index = self.spec.path_component
            if type(index) is not int or index < 0:
                raise ValueError("speaker path_component must be a non-negative integer")
            payload = {
                "source": source,
                "path_component": index,
                "required": self.spec.required,
            }
        elif source == "regex":
            compiled = _compile_regex(self.spec)
            object.__setattr__(self, "_compiled_regex", compiled)
            payload = {
                "source": source,
                "regex": self.spec.regex,
                "regex_group": self.spec.regex_group,
                "required": self.spec.required,
            }
        elif source == "mapping_file":
            base, open_path, canonical_path = _resolve_mapping_path(
                self.spec.mapping_path, self.mapping_base_dir,
            )
            snapshot = _open_mapping_snapshot(open_path, canonical_path, base)
            mapping = _parse_mapping_snapshot(snapshot)
            object.__setattr__(self, "mapping_base_dir", base)
            object.__setattr__(self, "_mapping", mapping)
            payload = {
                "source": source,
                "mapping_path": canonical_path,
                "mapping_sha256": hashlib.sha256(snapshot).hexdigest(),
                "required": self.spec.required,
            }
        else:
            raise ValueError(f"unsupported speaker metadata source: {source!r}")

        object.__setattr__(self, "_payload_items", tuple(sorted(payload.items())))

    def resolve(self, source_relative_path: str) -> Optional[str]:
        """解析单个规范化 POSIX source-relative path。"""
        source_path = validate_source_relative_path(source_relative_path)
        source = self.spec.source
        if source == "none":
            return None
        if source == "path_component":
            parents = source_path.split("/")[:-1]
            index = self.spec.path_component
            label = parents[index] if index is not None and index < len(parents) else None
        elif source == "regex":
            match = self._compiled_regex.search(source_path) if self._compiled_regex else None
            label = match.group(self.spec.regex_group) if match else None
        else:
            label = self._mapping.get(source_path)

        if label is None or label == "":
            if self.spec.required:
                raise ValueError(
                    f"required speaker label is missing for {source_path!r}"
                )
            return None
        return _validate_speaker_id(label)

    def config_payload(self) -> Dict[str, Any]:
        """返回用于 Step0 config hashing 的 fresh canonical JSON-safe metadata。"""
        return dict(self._payload_items)
