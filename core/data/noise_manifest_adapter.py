"""训练侧噪声 source manifest 的严格解析与输入快照工具。"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

NOISE_SOURCE_PARSER_VERSION = 10
_COMMON_SAMPLE_RATES = frozenset(
    {8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 96000}
)
_MAX_SAMPLE_RATE = 768_000
_MAX_AUDIO_FRAMES = 2**53 - 1


@dataclass(frozen=True)
class FileIdentity:
    """同一打开文件描述符上的稳定文件身份。"""

    size: int
    mtime_ns: int
    ctime_ns: int
    inode: int
    device: int


@dataclass(frozen=True)
class FileBytesSnapshot:
    """manifest 的 bytes、摘要与读取时身份。"""

    data: bytes
    sha256: str
    identity: FileIdentity


@dataclass(frozen=True)
class NoiseManifestCapabilities:
    """由实际 metadata 完整度派生的训练能力。"""

    has_metadata: bool
    category_complete: bool
    strict_fields_complete: bool


@dataclass(frozen=True)
class ParsedNoiseManifest:
    """训练噪声 parser 的稳定返回对象。"""

    by_fs: dict[int, dict[str, str]]
    uids: dict[int, list[str]]
    flat: dict[str, str]
    lengths: dict[str, int] | None
    metadata: dict[str, dict[str, Any]]
    capabilities: NoiseManifestCapabilities
    parser_version: int = NOISE_SOURCE_PARSER_VERSION



def _identity_from_stat(value: os.stat_result) -> FileIdentity:
    return FileIdentity(
        size=value.st_size,
        mtime_ns=value.st_mtime_ns,
        ctime_ns=value.st_ctime_ns,
        inode=value.st_ino,
        device=value.st_dev,
    )


def read_file_bytes_snapshot(path: str | Path) -> FileBytesSnapshot:
    """在同一 FD 上读取 bytes，并绑定当前非 symlink leaf 身份。"""
    candidate = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags)
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            before_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(before_stat.st_mode):
                raise ValueError(f"manifest path must be a regular file: {candidate}")
            before = _identity_from_stat(before_stat)
            data = handle.read()
            after = _identity_from_stat(os.fstat(handle.fileno()))
        path_after = _identity_from_stat(candidate.stat(follow_symlinks=False))
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if before != after or before != path_after or len(data) != before.size:
        raise ValueError(f"{candidate}: file changed while reading")
    return FileBytesSnapshot(
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        identity=before,
    )


def assert_file_identity(path: str | Path, expected: FileIdentity) -> None:
    """复验 path 当前 regular leaf 身份与先前 snapshot 一致。"""
    candidate = Path(path)
    value = candidate.stat(follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode) or _identity_from_stat(value) != expected:
        raise ValueError(f"{path}: file identity changed")


def _nonempty_string(value: Any, *, field: str, location: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{location}: {field} must be a non-empty trimmed string")
    return value


def _positive_int(
    value: Any,
    *,
    field: str,
    location: str,
    maximum: int,
) -> int:
    if type(value) is not int or value <= 0 or value > maximum:
        raise ValueError(
            f"{location}: {field} must be an exact integer > 0 and <= {maximum}"
        )
    return value


def _base_record(
    item: Any,
    *,
    location: str,
    require_length: bool,
) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise ValueError(f"{location}: record must be an object")
    record = dict(item)
    _nonempty_string(record.get("uid"), field="uid", location=location)
    _nonempty_string(record.get("path"), field="path", location=location)
    _positive_int(
        record.get("fs"),
        field="fs",
        location=location,
        maximum=_MAX_SAMPLE_RATE,
    )
    if require_length or "length" in record:
        _positive_int(
            record.get("length"),
            field="length",
            location=location,
            maximum=_MAX_AUDIO_FRAMES,
        )
    return record


def _add_record(
    record: dict[str, Any],
    *,
    location: str,
    by_fs: dict[int, dict[str, str]],
    flat: dict[str, str],
    lengths: dict[str, int],
    metadata: dict[str, dict[str, Any]],
    keep_metadata: bool,
) -> None:
    uid = record["uid"]
    if uid in flat:
        raise ValueError(f"{location}: duplicate uid {uid!r}")
    fs = record["fs"]
    path = record["path"]
    by_fs[fs][uid] = path
    flat[uid] = path
    if "length" in record:
        lengths[uid] = record["length"]
    if keep_metadata:
        metadata[uid] = record


def _strict_json_loads(text: str, *, location: str) -> Any:
    """解析 JSON，并对任意层级重复 object key fail-closed。"""

    class _ObjectPairs(list[tuple[str, Any]]):
        pass

    try:
        raw = json.loads(text, object_pairs_hook=_ObjectPairs)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{location}: invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}"
        ) from exc

    def _materialize(value: Any, *, object_path: str) -> Any:
        if isinstance(value, _ObjectPairs):
            result: dict[str, Any] = {}
            for key, child in value:
                if key in result:
                    raise ValueError(
                        f"{location}: duplicate JSON object key {key!r} at {object_path}"
                    )
                result[key] = _materialize(
                    child,
                    object_path=f"{object_path}[{key!r}]",
                )
            return result
        if isinstance(value, list):
            return [
                _materialize(child, object_path=f"{object_path}[{index}]")
                for index, child in enumerate(value)
            ]
        return value

    return _materialize(raw, object_path="$")


def _iter_stable_text_lines(path: Path, *, format_name: str) -> Iterator[tuple[int, str]]:
    """逐行读取 regular file，并在 EOF 后复验 FD 与路径身份。"""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before_stat = os.fstat(descriptor)
        if not stat.S_ISREG(before_stat.st_mode):
            raise ValueError(f"manifest path must be a regular file: {path}")
        before = _identity_from_stat(before_stat)
        with os.fdopen(descriptor, "r", encoding="utf-8", newline="") as handle:
            descriptor = -1
            try:
                for line_number, raw_line in enumerate(handle, start=1):
                    yield line_number, raw_line.rstrip("\r\n")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path}: {format_name} must be UTF-8") from exc
            after = _identity_from_stat(os.fstat(handle.fileno()))
        path_after = _identity_from_stat(path.stat(follow_symlinks=False))
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if before != after or before != path_after:
        raise ValueError(f"{path}: file changed while reading")


def _jsonl_records(path: Path) -> Iterator[tuple[str, Any]]:
    for line_number, raw_line in _iter_stable_text_lines(path, format_name="JSONL"):
        location = f"{path}: line {line_number}"
        if not raw_line.strip():
            raise ValueError(f"{location}: blank JSONL line")
        yield location, _strict_json_loads(raw_line, location=location)


@dataclass(frozen=True)
class _ScpField:
    """SCP shell field plus whether quoting disambiguated its boundaries."""

    value: str
    quoted: bool


def _split_scp_fields(raw_line: str, *, location: str) -> list[_ScpField]:
    """按 POSIX shell word 规则切字段，并保留显式 quoting provenance。"""
    fields: list[_ScpField] = []
    index = 0
    length = len(raw_line)
    while index < length:
        while index < length and raw_line[index].isspace():
            index += 1
        if index == length:
            return fields
        chars: list[str] = []
        quoted = False
        quote: str | None = None
        while index < length:
            char = raw_line[index]
            if quote is not None:
                quoted = True
                if char == quote:
                    quote = None
                elif char == "\\" and quote == '"':
                    if index + 1 < length and raw_line[index + 1] in '$`"\\':
                        index += 1
                        chars.append(raw_line[index])
                    else:
                        # POSIX 双引号内，反斜杠仅转义 $、`、"、\\ 与换行；
                        # 其余字符前的反斜杠必须原样保留。
                        chars.append(char)
                else:
                    chars.append(char)
                index += 1
                continue
            if char in "'\"":
                quoted = True
                quote = char
                index += 1
            elif char == "\\":
                if index + 1 >= length:
                    raise ValueError(f"{location}: invalid shell quoting")
                index += 1
                chars.append(raw_line[index])
                index += 1
            elif char.isspace():
                break
            else:
                chars.append(char)
                index += 1
        if quote is not None:
            raise ValueError(f"{location}: invalid shell quoting")
        fields.append(_ScpField("".join(chars), quoted))
    return fields


def _parse_scp_line(raw_line: str, *, location: str) -> dict[str, Any]:
    """严格解析 canonical ``uid fs length path`` SCP record。"""
    fields = _split_scp_fields(raw_line, location=location)
    if len(fields) != 4:
        raise ValueError(f"{location}: expected exactly four fields: uid fs length path")
    uid, fs_text, length_text, path = (field.value for field in fields)
    try:
        fs = int(fs_text)
        length = int(length_text)
    except ValueError as exc:
        raise ValueError(
            f"{location}: fs and length must be exact positive integers"
        ) from exc
    return _base_record(
        {"uid": uid, "fs": fs, "length": length, "path": path},
        location=location,
        require_length=True,
    )


def _scp_records(path: Path) -> Iterator[tuple[str, Any]]:
    for line_number, raw_line in _iter_stable_text_lines(path, format_name="SCP"):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        location = f"{path}: line {line_number}"
        yield location, _parse_scp_line(stripped, location=location)


def _category_metadata_complete(record: Mapping[str, Any]) -> bool:
    category_id = record.get("category_id")
    category_name = record.get("balance_category", record.get("category_name"))
    return (
        type(category_id) is int
        and category_id >= 0
        and isinstance(category_name, str)
        and bool(category_name)
        and category_name == category_name.strip()
    )


def _resolve_record_path(raw_path: str, *, manifest_dir: Path) -> str:
    """相对音频路径锚定到 manifest 所在目录；绝对路径与 ``~`` 开头保持不变。

    旧行为把相对路径留给消费端按进程 cwd 解析，跨目录启动训练会静默失效；
    manifest 与其音频通常是同树分发的，锚定到 manifest 目录才是可移植语义。
    """
    if os.path.isabs(raw_path) or raw_path.startswith("~"):
        return raw_path
    return os.path.normpath(str(manifest_dir / raw_path))


def read_noise_source_manifest(
    path: str | Path,
    *,
    production_admission: bool = False,
) -> ParsedNoiseManifest:
    """读取严格四列 SCP 或 JSONL source，并校验基础 record schema。

    默认路径只消费 ``uid/path/fs/length`` 与 JSONL 可选 category metadata，
    不加载或重放 Noise Filter policy/admission/fingerprint。只有调用方显式传入
    ``production_admission=True`` 时才执行 fail-closed schema v6 准入。
    相对音频路径一律相对 manifest 文件所在目录解析（见 ``_resolve_record_path``）。
    """
    candidate = Path(path)
    manifest_dir = Path(os.path.abspath(candidate)).parent
    suffix = candidate.suffix.lower()
    if suffix == ".jsonl":
        located_records = _jsonl_records(candidate)
        keep_metadata = True
    elif suffix == ".scp":
        located_records = _scp_records(candidate)
        keep_metadata = False
    else:
        raise ValueError(
            f"{candidate}: source manifest must use .scp (uid fs length path) or .jsonl"
        )

    by_fs: dict[int, dict[str, str]] = defaultdict(dict)
    flat: dict[str, str] = {}
    lengths: dict[str, int] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for location, raw_record in located_records:
        record = (
            raw_record
            if not keep_metadata
            else _base_record(raw_record, location=location, require_length=True)
        )
        record["path"] = _resolve_record_path(record["path"], manifest_dir=manifest_dir)
        _add_record(
            record,
            location=location,
            by_fs=by_fs,
            flat=flat,
            lengths=lengths,
            metadata=metadata,
            keep_metadata=keep_metadata,
        )
    if production_admission:
        if not keep_metadata:
            raise ValueError("production admission requires JSON record metadata")
        from core.data.noise_admission import validate_noise_manifest_records

        validate_noise_manifest_records(metadata, source_path=candidate)

    categories_complete = bool(flat) and keep_metadata and all(
        _category_metadata_complete(metadata[uid]) for uid in flat
    )
    return ParsedNoiseManifest(
        by_fs=dict(by_fs),
        uids={fs: list(records) for fs, records in by_fs.items()},
        flat=flat,
        lengths=(lengths if len(lengths) == len(flat) and lengths else None),
        metadata=metadata,
        capabilities=NoiseManifestCapabilities(
            has_metadata=keep_metadata,
            category_complete=categories_complete,
            strict_fields_complete=bool(metadata) and production_admission,
        ),
    )


def canonical_json_bytes(value: Any) -> bytes:
    """生成拒绝 NaN/Inf 的 canonical JSON bytes。"""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
