"""v3 不可变 artifact 的原子发布与严格读取接口。"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import secrets
import stat
import tempfile
import threading
import warnings
import weakref
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .path_utils import PRODUCTION_LAYOUT_VERSION, v3_artifact_dir


ARTIFACT_SCHEMA_VERSION = 2
_CONTRACT_FILES = frozenset({"rows.jsonl", "index.json", "_SUCCESS"})
_INDEX_FIELDS = frozenset({
    "schema_version",
    "layout_version",
    "step",
    "dataset_id",
    "config_digest",
    "lineage_digest",
    "row_count",
    "rows_sha256",
    "artifact_digest",
})
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_INDEX_MAX_BYTES = 64 * 1024
_MARKER_BYTES = 65


class PipelineStep(str, Enum):
    """v3 pipeline 的固定步骤域。"""

    STEP0 = "step0"
    STEP1 = "step1"
    STEP2 = "step2"
    STEP3 = "step3"
    STEP4 = "step4"
    STEP5 = "step5"
    STEP6 = "step6"


@dataclass(frozen=True)
class ArtifactIndex:
    """artifact 的不可变索引；路径属性与 opener capability 不参与序列化。

    只有 ``open_artifact_for_read()`` 与 ``inspect_artifact_unbound()`` 返回的
    实例可传给 ``iter_artifact_rows()``；前者另具 production-bound capability，
    后者仅供诊断。手工构造或复制的实例不具备任何 opener capability。
    """

    schema_version: int
    layout_version: int
    step: str
    dataset_id: str
    config_digest: str
    lineage_digest: str
    row_count: int
    rows_sha256: str
    artifact_digest: str
    artifact_dir: str
    output_root: str

    @property
    def rows_path(self) -> str:
        return os.path.join(self.artifact_dir, "rows.jsonl")

    @property
    def index_path(self) -> str:
        return os.path.join(self.artifact_dir, "index.json")

    @property
    def success_path(self) -> str:
        return os.path.join(self.artifact_dir, "_SUCCESS")

    def serialized(self, *, include_artifact_digest: bool = True) -> dict:
        payload = {
            "schema_version": self.schema_version,
            "layout_version": self.layout_version,
            "step": self.step,
            "dataset_id": self.dataset_id,
            "config_digest": self.config_digest,
            "lineage_digest": self.lineage_digest,
            "row_count": self.row_count,
            "rows_sha256": self.rows_sha256,
        }
        if include_artifact_digest:
            payload["artifact_digest"] = self.artifact_digest
        return payload


_VERIFIED_INDEX_REFS: dict[
    int, tuple[weakref.ReferenceType[ArtifactIndex], tuple[Any, ...], bool],
] = {}
_VERIFIED_INDEX_LOCK = threading.Lock()


def _index_fingerprint(index: ArtifactIndex) -> tuple[Any, ...]:
    return (
        index.schema_version,
        index.layout_version,
        index.step,
        index.dataset_id,
        index.config_digest,
        index.lineage_digest,
        index.row_count,
        index.rows_sha256,
        index.artifact_digest,
        index.artifact_dir,
        index.output_root,
    )


def _mark_verified_index(
    index: ArtifactIndex, *, production_bound: bool,
) -> ArtifactIndex:
    """登记 opener 实例的 identity、快照与 production-bound 资格。"""
    identity = id(index)

    def discard(reference) -> None:
        with _VERIFIED_INDEX_LOCK:
            registered = _VERIFIED_INDEX_REFS.get(identity)
            if registered is not None and registered[0] is reference:
                _VERIFIED_INDEX_REFS.pop(identity, None)

    reference = weakref.ref(index, discard)
    registered = (reference, _index_fingerprint(index), production_bound)
    with _VERIFIED_INDEX_LOCK:
        _VERIFIED_INDEX_REFS[identity] = registered
    return index


def _verified_index_registration(index: ArtifactIndex):
    if not isinstance(index, ArtifactIndex):
        return None
    with _VERIFIED_INDEX_LOCK:
        registered = _VERIFIED_INDEX_REFS.get(id(index))
    if (
        registered is None
        or registered[0]() is not index
        or registered[1] != _index_fingerprint(index)
    ):
        return None
    return registered


def is_production_bound_artifact_index(index: ArtifactIndex) -> bool:
    """判定 index 是否来自 expectation-bound 生产 opener，且身份快照未变。"""
    registered = _verified_index_registration(index)
    return registered is not None and registered[2]


def _parse_step(step) -> PipelineStep:
    try:
        return step if isinstance(step, PipelineStep) else PipelineStep(step)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid step: {step!r}") from exc


def _require_digest(value, field: str) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"invalid {field}: expected lowercase SHA-256 hex")
    return value


def _require_dataset_id(value) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(
            f"dataset_id must be a non-empty string, got {value!r}"
        )
    return value


def _require_string_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object key must be a string")
            _require_string_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _require_string_keys(nested)


def _canonical_json_bytes(value: Any) -> bytes:
    _require_string_keys(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number: {value}")
    return parsed


def _reject_duplicate_keys(pairs):
    value = {}
    for key, nested in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        value[key] = nested
    return value


def _load_json_object(raw: bytes, label: str) -> dict:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_float,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}: expected JSON object")
    return value


def _absolute(path) -> str:
    return os.path.abspath(os.fspath(path))


def _add_exception_note(error: BaseException, note: str) -> None:
    """将 cleanup 诊断附到原异常，并兼容 Python 3.10。"""
    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(note)
        return
    notes = list(getattr(error, "__notes__", ()))
    notes.append(note)
    error.__notes__ = notes


_DIRECTORY_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_FLAGS = (
    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


def _open_directory_at(
    parent_fd: int, name: str, *, create: bool, sync_parent: bool = False,
) -> int:
    created = False
    if create:
        try:
            os.mkdir(name, mode=0o755, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            pass
        if created and sync_parent:
            os.fsync(parent_fd)
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ValueError(f"artifact path contains symlink or non-directory: {name!r}") from exc
        raise
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"artifact path is not a directory: {name!r}")
        return descriptor
    except BaseException as exc:
        _close_preserving(descriptor, exc, "directory fd")
        raise


@dataclass
class _DirectoryChain:
    """持有绝对目录链的全部 fd，以便复验每层 name binding。"""

    descriptors: list[int]
    names: list[str]

    @property
    def leaf_fd(self) -> int:
        return self.descriptors[-1]

    def verify(self) -> None:
        for parent_fd, name, child_fd in zip(
            self.descriptors, self.names, self.descriptors[1:],
        ):
            _require_named_inode(parent_fd, name, child_fd)

    def close(self) -> None:
        descriptors = self.descriptors
        self.descriptors = []
        first_error = None
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    _add_exception_note(
                        first_error, f"additional directory fd close failed: {exc!r}",
                    )
        if first_error is not None:
            raise first_error


def _open_artifact_parent(
    output_root: str, step: str, *, create: bool,
) -> _DirectoryChain:
    absolute = _absolute(output_root)
    descriptors = [os.open(os.sep, _DIRECTORY_FLAGS)]
    names = []
    try:
        components = [part for part in absolute.split(os.sep) if part]
        # 无中间 artifacts 层：output_root 下直接是 step。
        components.append(step)
        for component in components:
            child = _open_directory_at(
                descriptors[-1], component, create=create, sync_parent=create,
            )
            names.append(component)
            descriptors.append(child)
        return _DirectoryChain(descriptors, names)
    except BaseException as exc:
        chain = _DirectoryChain(descriptors, names)
        try:
            chain.close()
        except BaseException as close_exc:
            _add_exception_note(exc, f"directory chain close failed: {close_exc!r}")
        raise


def _open_artifact_directory(
    output_root: str, step: str, dataset_name: str,
) -> tuple[_DirectoryChain, int]:
    parent_chain = _open_artifact_parent(output_root, step, create=False)
    artifact_fd = None
    try:
        artifact_fd = _open_directory_at(
            parent_chain.leaf_fd, dataset_name, create=False,
        )
        return parent_chain, artifact_fd
    except BaseException as exc:
        if artifact_fd is not None:
            try:
                os.close(artifact_fd)
            except BaseException as close_exc:
                _add_exception_note(exc, f"artifact fd close failed: {close_exc!r}")
        try:
            parent_chain.close()
        except BaseException as close_exc:
            _add_exception_note(exc, f"directory chain close failed: {close_exc!r}")
        raise


def _same_inode(left, right) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _same_file_snapshot(left, right) -> bool:
    return (
        _same_inode(left, right)
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _require_same_file_snapshot(
    before: os.stat_result, descriptor: int, message: str,
) -> None:
    if not _same_file_snapshot(before, os.fstat(descriptor)):
        raise RuntimeError(message)


def _close_preserving(
    descriptor: int, primary: BaseException, label: str,
) -> None:
    try:
        os.close(descriptor)
    except BaseException as close_exc:
        _add_exception_note(primary, f"{label} close failed: {close_exc!r}")


def _open_owned_duplicate_stream(descriptor: int, mode: str):
    """创建不接管 duplicated fd 的 stream，由 caller 显式收敛两层资源。"""
    duplicate = os.dup(descriptor)
    try:
        stream = os.fdopen(duplicate, mode, closefd=False)
    except BaseException as exc:
        _close_preserving(duplicate, exc, "duplicated fd")
        raise
    return stream, duplicate


def _close_owned_duplicate_stream(stream, descriptor: int) -> Optional[BaseException]:
    """尽力关闭 stream 及独立 raw fd，并返回首个资源错误。"""
    first_error = None
    try:
        stream.close()
    except BaseException as exc:
        first_error = exc
        try:
            stream.close()
        except BaseException as retry_exc:
            _add_exception_note(
                exc, f"duplicated stream close retry failed: {retry_exc!r}",
            )
    try:
        os.close(descriptor)
    except BaseException as close_exc:
        if first_error is None:
            first_error = close_exc
        else:
            _add_exception_note(
                first_error, f"duplicated raw fd close failed: {close_exc!r}",
            )
    return first_error


def _require_named_inode(parent_fd: int, name: str, held_fd: int) -> None:
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"artifact directory entry changed: {name!r}") from exc
    if not _same_inode(named, os.fstat(held_fd)):
        raise RuntimeError(f"artifact directory entry changed: {name!r}")


def _write_fsynced_at(directory_fd: int, name: str, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


_RENAME_OUTCOME = threading.local()


def _renameat2_noreplace(
    source_fd: int, source_name: str, target_fd: int, target_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is required for atomic no-replace publication")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_fd,
        os.fsencode(source_name),
        target_fd,
        os.fsencode(target_name),
        1,
    )
    if result == 0:
        token = getattr(_RENAME_OUTCOME, "token", None)
        if token is not None:
            token["renamed"] = True
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), target_name)
    raise OSError(error_number, os.strerror(error_number), target_name)


def _atomic_rename_noreplace_between(
    source_fd: int, source_name: str, target_fd: int, target_name: str,
) -> None:
    """在固定 dirfd 间以 renameat2(RENAME_NOREPLACE) 移动 entry。"""
    _renameat2_noreplace(source_fd, source_name, target_fd, target_name)


def _quarantine_wrong_publication(
    parent_fd: int, final_name: str, verified_fd: int,
) -> None:
    """仅将错误 publication 安全移出 canonical name，并保留诊断数据。"""
    try:
        published_stat = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if _same_inode(published_stat, os.fstat(verified_fd)):
        return
    quarantine_name = f".{final_name}.quarantine-{secrets.token_hex(16)}"
    rename_error = None
    try:
        _atomic_rename_noreplace_between(
            parent_fd, final_name, parent_fd, quarantine_name,
        )
    except BaseException as exc:
        rename_error = exc

    try:
        final_stat = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        final_stat = None
    try:
        quarantine_stat = os.stat(
            quarantine_name, dir_fd=parent_fd, follow_symlinks=False,
        )
    except FileNotFoundError:
        quarantine_stat = None

    final_is_published = (
        final_stat is not None and _same_inode(final_stat, published_stat)
    )
    quarantine_is_published = (
        quarantine_stat is not None and _same_inode(quarantine_stat, published_stat)
    )
    if final_stat is None and quarantine_is_published:
        os.fsync(parent_fd)
        return
    if final_is_published and quarantine_stat is None and rename_error is not None:
        raise rename_error
    raise RuntimeError("unverified publication quarantine outcome is uncertain")


def _rmdir_owned_and_confirm_absent(
    container_fd: int,
    name: str,
    held_fd: Optional[int] = None,
    expected_stat: Optional[os.stat_result] = None,
) -> None:
    if held_fd is not None:
        _require_named_inode(container_fd, name, held_fd)
    else:
        entry = os.stat(name, dir_fd=container_fd, follow_symlinks=False)
        if not stat.S_ISDIR(entry.st_mode):
            raise ValueError(f"owned artifact entry is not a directory: {name!r}")
        if expected_stat is None or not _same_inode(entry, expected_stat):
            raise RuntimeError(f"owned artifact directory entry changed: {name!r}")
    try:
        os.rmdir(name, dir_fd=container_fd)
    except BaseException:
        try:
            os.stat(name, dir_fd=container_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise
    try:
        os.stat(name, dir_fd=container_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise RuntimeError(f"owned artifact directory still exists: {name!r}")


def _remove_owned_directory(container_fd: int, name: str, held_fd: int) -> None:
    """只清理预期普通 contract 文件，不递归处理未知 entry。"""
    _require_named_inode(container_fd, name, held_fd)
    names = []
    with os.scandir(held_fd) as entries:
        for entry in entries:
            if entry.name not in _CONTRACT_FILES:
                raise ValueError(f"unexpected artifact file: {entry.name!r}")
            names.append(entry.name)
    for entry_name in names:
        try:
            entry = os.stat(entry_name, dir_fd=held_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(entry.st_mode):
            raise ValueError(f"artifact contract file is not regular: {entry_name!r}")
        os.unlink(entry_name, dir_fd=held_fd)
    _rmdir_owned_and_confirm_absent(container_fd, name, held_fd)


def _open_contract_file(artifact_fd: int, name: str) -> int:
    try:
        descriptor = os.open(name, _FILE_FLAGS, dir_fd=artifact_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"artifact contract file is a symlink: {name!r}") from exc
        raise ValueError(f"missing artifact contract file: {name!r}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"artifact contract file is not regular: {name!r}")
        return descriptor
    except BaseException as exc:
        _close_preserving(descriptor, exc, "contract fd")
        raise


def _read_bounded_fd(descriptor: int, *, limit: int, label: str) -> bytes:
    expected_size = os.fstat(descriptor).st_size
    if expected_size > limit:
        raise ValueError(f"{label} file is too large")
    os.lseek(descriptor, 0, os.SEEK_SET)
    data = bytearray()
    while len(data) <= limit:
        chunk = os.read(descriptor, min(8192, limit + 1 - len(data)))
        if not chunk:
            break
        data.extend(chunk)
    if len(data) > limit or os.read(descriptor, 1):
        raise ValueError(f"{label} file is too large")
    if os.fstat(descriptor).st_size != expected_size or len(data) != expected_size:
        raise RuntimeError(f"{label} file changed during read")
    return bytes(data)


def _validate_contract_names(artifact_fd: int) -> None:
    seen = set()
    with os.scandir(artifact_fd) as entries:
        for entry in entries:
            name = entry.name
            if name not in _CONTRACT_FILES:
                raise ValueError(f"unexpected artifact file: {name!r}")
            seen.add(name)
    missing = _CONTRACT_FILES - seen
    if missing:
        raise ValueError(f"missing artifact contract file(s): {sorted(missing)!r}")


def _contract_named_stats(artifact_fd: int) -> dict[str, os.stat_result]:
    stats = {}
    for name in _CONTRACT_FILES:
        entry = os.stat(name, dir_fd=artifact_fd, follow_symlinks=False)
        if not stat.S_ISREG(entry.st_mode):
            raise ValueError(f"artifact contract file is not regular: {name!r}")
        stats[name] = entry
    return stats


def _require_contract_named_stats(
    artifact_fd: int, expected: Mapping[str, os.stat_result],
) -> None:
    current = _contract_named_stats(artifact_fd)
    for name, before in expected.items():
        after = current[name]
        if not _same_file_snapshot(before, after):
            raise RuntimeError(f"artifact contract entry changed: {name!r}")


def _merge_resource_error(
    first_error: Optional[BaseException], error: Optional[BaseException], label: str,
) -> Optional[BaseException]:
    """保留首个资源错误，并把后续独立资源错误附为诊断。"""
    if error is None:
        return first_error
    if first_error is None:
        _add_exception_note(error, f"{label}: {error!r}")
        return error
    _add_exception_note(first_error, f"{label}: {error!r}")
    for note in getattr(error, "__notes__", ()):
        _add_exception_note(first_error, note)
    return first_error


def _attach_resource_error(
    primary: BaseException, resource_error: Optional[BaseException], label: str,
) -> None:
    if resource_error is None:
        return
    _add_exception_note(primary, f"{label}: {resource_error!r}")
    for note in getattr(resource_error, "__notes__", ()):
        _add_exception_note(primary, note)


def _scan_rows_fd(descriptor: int) -> tuple[str, int, Optional[BaseException]]:
    """验证 canonical JSONL，并显式返回 verifier stream 的资源错误。"""
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    count = 0
    stream, duplicate = _open_owned_duplicate_stream(descriptor, "rb")
    primary_error = None
    try:
        for raw_line in stream:
            count += 1
            digest.update(raw_line)
            if not raw_line.endswith(b"\n"):
                raise ValueError(f"invalid artifact row {count}: missing LF newline")
            try:
                row = _load_json_object(raw_line[:-1], f"artifact row {count}")
            except ValueError as exc:
                raise ValueError(f"invalid artifact row {count}: {exc}") from exc
            if raw_line != _canonical_json_bytes(row) + b"\n":
                raise ValueError(f"invalid artifact row {count}: non-canonical bytes")
    except BaseException as exc:
        primary_error = exc
    close_error = _close_owned_duplicate_stream(stream, duplicate)
    if primary_error is not None:
        _attach_resource_error(
            primary_error, close_error, "rows verifier resource close failed",
        )
        raise primary_error
    return digest.hexdigest(), count, close_error


_ROWS_SCAN_OUTCOME = threading.local()


def _rows_hash_and_count_fd(descriptor: int) -> tuple[str, int]:
    """正常 snapshot 的兼容 hook；资源错误由 caller 延后裁定。"""
    _ROWS_SCAN_OUTCOME.resource_error = None
    digest, count, resource_error = _scan_rows_fd(descriptor)
    _ROWS_SCAN_OUTCOME.resource_error = resource_error
    return digest, count


def _take_rows_scan_resource_error() -> Optional[BaseException]:
    resource_error = getattr(_ROWS_SCAN_OUTCOME, "resource_error", None)
    _ROWS_SCAN_OUTCOME.resource_error = None
    return resource_error


class ArtifactWriter:
    """单次使用的 v3 artifact immutable-directory transaction。"""

    def __init__(self, output_root, step, dataset_id, config_digest, lineage_digest):
        self._output_root = _absolute(output_root)
        self._step = _parse_step(step)
        self._dataset_id = _require_dataset_id(dataset_id)
        self._config_digest = _require_digest(config_digest, "config_digest")
        self._lineage_digest = _require_digest(lineage_digest, "lineage_digest")
        self._final_dir = _absolute(
            v3_artifact_dir(self._output_root, self._step.value, self._dataset_id)
        )
        self._final_name = os.path.basename(self._final_dir)
        self._state = "active"
        self._cleanup_error = None
        self._row_count = 0
        self._rows_digest = hashlib.sha256()
        self._rows = None
        self._parent_chain = None
        self._parent_fd = None
        self._staging_name = None
        self._staging_stat = None
        self._staging_fd = None
        self._temp_name = None
        self._temp_stat = None
        self._temp_fd = None

        rows_fd = None
        try:
            self._parent_chain = _open_artifact_parent(
                self._output_root, self._step.value, create=True,
            )
            self._parent_fd = self._parent_chain.leaf_fd
            try:
                existing_fd = _open_directory_at(
                    self._parent_fd, self._final_name, create=False,
                )
            except FileNotFoundError:
                existing_fd = None
            if existing_fd is not None:
                os.close(existing_fd)
                raise FileExistsError(f"artifact target already exists: {self._final_dir!r}")
            for _attempt in range(128):
                candidate = f".{self._final_name}.stage-{secrets.token_hex(16)}"
                try:
                    os.mkdir(candidate, mode=0o700, dir_fd=self._parent_fd)
                except FileExistsError:
                    continue
                self._staging_name = candidate
                self._staging_stat = os.stat(
                    candidate, dir_fd=self._parent_fd, follow_symlinks=False,
                )
                self._staging_fd = _open_directory_at(
                    self._parent_fd, candidate, create=False,
                )
                break
            else:  # pragma: no cover - 128 次随机碰撞不可现实触发
                raise FileExistsError("cannot allocate private artifact staging directory")
            self._temp_name = "artifact"
            os.mkdir(self._temp_name, mode=0o700, dir_fd=self._staging_fd)
            self._temp_stat = os.stat(
                self._temp_name, dir_fd=self._staging_fd, follow_symlinks=False,
            )
            self._temp_fd = _open_directory_at(
                self._staging_fd, self._temp_name, create=False,
            )
            rows_fd = os.open(
                "rows.jsonl",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=self._temp_fd,
            )
            self._rows = os.fdopen(rows_fd, "wb")
            rows_fd = None
        except BaseException as exc:
            if rows_fd is not None:
                try:
                    os.close(rows_fd)
                except BaseException:
                    pass
            try:
                self._cleanup_unpublished()
            except BaseException as cleanup_exc:
                self._state = "cleanup_failed"
                self._cleanup_error = cleanup_exc
                _add_exception_note(exc, f"artifact cleanup failed: {cleanup_exc!r}")
                try:
                    self._close_descriptors()
                except BaseException as close_exc:
                    _add_exception_note(
                        exc, f"artifact descriptor close failed: {close_exc!r}",
                    )
            else:
                self._state = "aborted"
            raise

    @property
    def state(self) -> str:
        """返回 writer 的终态或当前活动状态。"""
        return self._state

    @property
    def cleanup_error(self):
        """返回最近一次清理失败；无失败时为 ``None``。"""
        return self._cleanup_error

    def _require_active(self) -> None:
        if self._state != "active":
            raise RuntimeError(f"artifact writer is already {self._state}")

    def preview_index(self) -> ArtifactIndex:
        """返回当前已 append rows 的最终 index 身份，不写盘且不改变 writer 状态。"""
        self._require_active()
        without_digest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "layout_version": PRODUCTION_LAYOUT_VERSION,
            "step": self._step.value,
            "dataset_id": self._dataset_id,
            "config_digest": self._config_digest,
            "lineage_digest": self._lineage_digest,
            "row_count": self._row_count,
            "rows_sha256": self._rows_digest.hexdigest(),
        }
        artifact_digest = hashlib.sha256(
            _canonical_json_bytes(without_digest)
        ).hexdigest()
        return ArtifactIndex(
            **without_digest,
            artifact_digest=artifact_digest,
            artifact_dir=self._final_dir,
            output_root=self._output_root,
        )

    def _close_one(self, attribute: str) -> None:
        descriptor = getattr(self, attribute)
        setattr(self, attribute, None)
        if descriptor is not None:
            os.close(descriptor)

    def _close_descriptors(self) -> None:
        first_error = None
        for attribute in ("_temp_fd", "_staging_fd"):
            try:
                self._close_one(attribute)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        parent_chain = self._parent_chain
        self._parent_chain = None
        self._parent_fd = None
        if parent_chain is not None:
            try:
                parent_chain.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _cleanup_unpublished(self) -> None:
        first_error = None
        if self._rows is not None:
            try:
                self._rows.close()
            except BaseException as exc:
                first_error = exc
            finally:
                self._rows = None
        try:
            if self._temp_name:
                if self._temp_fd is not None:
                    _remove_owned_directory(
                        self._staging_fd, self._temp_name, self._temp_fd,
                    )
                elif self._staging_fd is not None:
                    _rmdir_owned_and_confirm_absent(
                        self._staging_fd,
                        self._temp_name,
                        expected_stat=self._temp_stat,
                    )
                self._temp_name = None
                self._temp_stat = None
            if self._staging_name:
                if self._staging_fd is not None:
                    _require_named_inode(
                        self._parent_fd, self._staging_name, self._staging_fd,
                    )
                    with os.scandir(self._staging_fd) as entries:
                        for _entry in entries:
                            raise ValueError(
                                "unexpected entry in private staging directory"
                            )
                    _rmdir_owned_and_confirm_absent(
                        self._parent_fd, self._staging_name, self._staging_fd,
                    )
                elif self._parent_fd is not None:
                    _rmdir_owned_and_confirm_absent(
                        self._parent_fd,
                        self._staging_name,
                        expected_stat=self._staging_stat,
                    )
                self._staging_name = None
                self._staging_stat = None
        except BaseException as exc:
            self._state = "cleanup_failed"
            self._cleanup_error = exc
            if first_error is not None:
                _add_exception_note(exc, f"rows close also failed: {first_error!r}")
            raise
        try:
            self._close_descriptors()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        if first_error is not None:
            raise first_error

    def _fail_and_cleanup(self, original: BaseException) -> None:
        try:
            self._cleanup_unpublished()
        except BaseException as cleanup_exc:
            self._state = "cleanup_failed"
            self._cleanup_error = cleanup_exc
            _add_exception_note(original, f"artifact cleanup failed: {cleanup_exc!r}")
        else:
            self._state = "aborted"

    def append(self, row: Mapping[str, Any]) -> None:
        self._require_active()
        if not isinstance(row, Mapping):
            raise TypeError("artifact row must be a mapping")
        try:
            encoded = _canonical_json_bytes(dict(row)) + b"\n"
        except ValueError as exc:
            raise ValueError("artifact row must contain finite JSON values") from exc
        except (TypeError, OverflowError) as exc:
            raise TypeError(
                "artifact row must contain JSON-serializable values with string object keys"
            ) from exc
        try:
            written = self._rows.write(encoded)
            if written is not None and written != len(encoded):
                raise OSError("short artifact row write")
        except BaseException as exc:
            self._fail_and_cleanup(exc)
            raise
        self._rows_digest.update(encoded)
        self._row_count += 1

    def finalize(self) -> ArtifactIndex:
        self._require_active()
        published = False
        publication_token = {"renamed": False}
        serialized = None
        try:
            self._parent_chain.verify()
            self._rows.flush()
            os.fsync(self._rows.fileno())
            self._rows.close()
            self._rows = None

            without_digest = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "layout_version": PRODUCTION_LAYOUT_VERSION,
                "step": self._step.value,
                "dataset_id": self._dataset_id,
                "config_digest": self._config_digest,
                "lineage_digest": self._lineage_digest,
                "row_count": self._row_count,
                "rows_sha256": self._rows_digest.hexdigest(),
            }
            artifact_digest = hashlib.sha256(
                _canonical_json_bytes(without_digest)
            ).hexdigest()
            serialized = dict(without_digest, artifact_digest=artifact_digest)
            _write_fsynced_at(
                self._temp_fd, "index.json", _canonical_json_bytes(serialized) + b"\n",
            )
            _write_fsynced_at(
                self._temp_fd, "_SUCCESS", (artifact_digest + "\n").encode("ascii"),
            )
            _validate_contract_names(self._temp_fd)
            contract_stats = _contract_named_stats(self._temp_fd)
            os.fsync(self._temp_fd)
            _require_named_inode(self._staging_fd, self._temp_name, self._temp_fd)
            _require_contract_named_stats(self._temp_fd, contract_stats)
            sealed_name = f".artifact.sealed-{secrets.token_hex(16)}"
            os.rename(
                self._temp_name, sealed_name,
                src_dir_fd=self._staging_fd, dst_dir_fd=self._staging_fd,
            )
            self._temp_name = sealed_name
            _require_named_inode(self._staging_fd, self._temp_name, self._temp_fd)
            _require_contract_named_stats(self._temp_fd, contract_stats)
            self._parent_chain.verify()
            previous_token = getattr(_RENAME_OUTCOME, "token", None)
            _RENAME_OUTCOME.token = publication_token
            try:
                _atomic_rename_noreplace_between(
                    self._staging_fd, self._temp_name,
                    self._parent_fd, self._final_name,
                )
            finally:
                published = publication_token["renamed"]
                if previous_token is None:
                    del _RENAME_OUTCOME.token
                else:
                    _RENAME_OUTCOME.token = previous_token
            self._temp_name = None
            _require_named_inode(self._parent_fd, self._final_name, self._temp_fd)
            _validate_contract_names(self._temp_fd)
            _require_contract_named_stats(self._temp_fd, contract_stats)
            _require_named_inode(
                self._parent_fd, self._staging_name, self._staging_fd,
            )
            self._parent_chain.verify()
            os.fsync(self._parent_fd)
            self._parent_chain.verify()
            _rmdir_owned_and_confirm_absent(
                self._parent_fd, self._staging_name, self._staging_fd,
            )
            self._staging_name = None
            self._state = "finalized"
            self._close_descriptors()
            return ArtifactIndex(
                **serialized,
                artifact_dir=self._final_dir,
                output_root=self._output_root,
            )
        except BaseException as exc:
            if published:
                self._state = "publication_failed"
                self._temp_name = None
                self._rows = None
                cleanup_error = None
                try:
                    _quarantine_wrong_publication(
                        self._parent_fd, self._final_name, self._temp_fd,
                    )
                    if self._staging_name and self._staging_fd is not None:
                        _rmdir_owned_and_confirm_absent(
                            self._parent_fd, self._staging_name, self._staging_fd,
                        )
                        self._staging_name = None
                except BaseException as cleanup_exc:
                    cleanup_error = cleanup_exc
                try:
                    self._close_descriptors()
                except BaseException as close_exc:
                    if cleanup_error is None:
                        cleanup_error = close_exc
                if cleanup_error is not None:
                    _add_exception_note(
                        exc, f"post-publication cleanup failed: {cleanup_error!r}",
                    )
            else:
                self._fail_and_cleanup(exc)
            raise

    def abort(self) -> None:
        if self._state not in ("active", "cleanup_failed"):
            raise RuntimeError(f"artifact writer is already {self._state}")
        self._cleanup_error = None
        try:
            self._cleanup_unpublished()
        except BaseException as exc:
            self._state = "cleanup_failed"
            self._cleanup_error = exc
            raise
        self._state = "aborted"

    def __enter__(self) -> "ArtifactWriter":
        self._require_active()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._state != "active":
            return False
        if exc_type is None:
            self.finalize()
        else:
            try:
                self.abort()
            except BaseException as cleanup_exc:
                _add_exception_note(
                    exc_value, f"artifact cleanup failed: {cleanup_exc!r}",
                )
        return False

    def __del__(self):  # pragma: no cover - 行为由 gc 回归测试覆盖
        if getattr(self, "_state", None) not in ("active", "cleanup_failed"):
            return
        try:
            warnings.warn(
                "unclosed ArtifactWriter; aborting unpublished artifact",
                ResourceWarning,
                source=self,
            )
        except BaseException:
            pass
        try:
            self._cleanup_unpublished()
        except BaseException as exc:
            self._state = "cleanup_failed"
            self._cleanup_error = exc
            # 清理可能在部分成功后报错；再试一次可收敛剩余 staging。
            try:
                self._cleanup_unpublished()
            except BaseException:
                pass
        finally:
            try:
                self._close_descriptors()
            except BaseException:
                pass
        if not self._staging_name and not self._temp_name:
            self._state = "aborted"


def _validate_index_payload(payload: dict, step: str, dataset_id: str) -> None:
    if set(payload) != _INDEX_FIELDS:
        raise ValueError("invalid index fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("schema version mismatch")
    if type(payload["layout_version"]) is not int or payload["layout_version"] != PRODUCTION_LAYOUT_VERSION:
        raise ValueError("layout version mismatch")
    if payload["step"] != step:
        raise ValueError("step mismatch")
    if payload["dataset_id"] != dataset_id:
        raise ValueError("dataset mismatch")
    for field in ("config_digest", "lineage_digest", "rows_sha256", "artifact_digest"):
        _require_digest(payload[field], field)
    row_count = payload["row_count"]
    if type(row_count) is not int or row_count < 0:
        raise ValueError("invalid row_count")
    without_digest = {key: value for key, value in payload.items() if key != "artifact_digest"}
    computed = hashlib.sha256(_canonical_json_bytes(without_digest)).hexdigest()
    if computed != payload["artifact_digest"]:
        raise ValueError("artifact digest mismatch")


def _record_verification_error(
    first_error: Optional[BaseException], error: BaseException, label: str,
) -> BaseException:
    """保留首个 integrity 错误，并记录其余独立校验结果。"""
    if first_error is None:
        return error
    _add_exception_note(first_error, f"{label}: {error!r}")
    for note in getattr(error, "__notes__", ()):
        _add_exception_note(first_error, note)
    return first_error


def _verify_artifact_snapshot(
    artifact_fd: int,
    step: str,
    dataset_id: str,
    *,
    expected_config_digest: Optional[str] = None,
    expected_lineage_digest: Optional[str] = None,
    expected_serialized: Optional[dict] = None,
) -> dict:
    """同时 pin 三个 contract fd，并在返回前复验全部 name binding。"""
    _validate_contract_names(artifact_fd)
    descriptors = {}
    stats = {}
    primary_error = None
    rows_resource_error = None
    try:
        for name in ("index.json", "_SUCCESS", "rows.jsonl"):
            descriptor = _open_contract_file(artifact_fd, name)
            descriptors[name] = descriptor
            stats[name] = os.fstat(descriptor)
        index_raw = _read_bounded_fd(
            descriptors["index.json"], limit=_INDEX_MAX_BYTES, label="index",
        )
        payload = _load_json_object(index_raw, "index")
        if index_raw != _canonical_json_bytes(payload) + b"\n":
            raise ValueError("non-canonical index bytes")
        _validate_index_payload(payload, step, dataset_id)
        if expected_serialized is not None and payload != expected_serialized:
            raise ValueError("artifact index changed after open")
        if expected_config_digest is not None:
            expected = _require_digest(
                expected_config_digest, "expected_config_digest",
            )
            if payload["config_digest"] != expected:
                raise ValueError("expected config digest mismatch")
        if expected_lineage_digest is not None:
            expected = _require_digest(
                expected_lineage_digest, "expected_lineage_digest",
            )
            if payload["lineage_digest"] != expected:
                raise ValueError("expected lineage digest mismatch")
        if os.fstat(descriptors["_SUCCESS"]).st_size != _MARKER_BYTES:
            raise ValueError("marker size mismatch")
        marker = _read_bounded_fd(
            descriptors["_SUCCESS"], limit=_MARKER_BYTES, label="marker",
        )
        if marker != (payload["artifact_digest"] + "\n").encode("ascii"):
            raise ValueError("marker digest mismatch or marker changed")
        rows_digest, row_count = _rows_hash_and_count_fd(
            descriptors["rows.jsonl"],
        )
        rows_resource_error = _take_rows_scan_resource_error()
        if rows_digest != payload["rows_sha256"]:
            raise ValueError("rows hash mismatch or rows changed")
        if row_count != payload["row_count"]:
            raise ValueError("row count mismatch or rows changed")
        for name, descriptor in descriptors.items():
            current = os.fstat(descriptor)
            before = stats[name]
            if not _same_file_snapshot(before, current):
                raise RuntimeError(f"artifact contract fd changed: {name!r}")
        _validate_contract_names(artifact_fd)
        _require_contract_named_stats(artifact_fd, stats)
        if rows_resource_error is not None:
            raise rows_resource_error
        return payload
    except BaseException as exc:
        primary_error = exc
        if rows_resource_error is not None and exc is not rows_resource_error:
            _attach_resource_error(
                exc, rows_resource_error, "rows verifier resource close failed",
            )
        raise
    finally:
        first_error = None
        for descriptor in descriptors.values():
            try:
                os.close(descriptor)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            if primary_error is not None:
                _add_exception_note(
                    primary_error, f"contract fd close failed: {first_error!r}",
                )
            else:
                raise first_error


def _open_artifact_index(
    output_root,
    step,
    dataset_id,
    *,
    expected_config_digest: Optional[str],
    expected_lineage_digest: Optional[str],
) -> ArtifactIndex:
    root = _absolute(output_root)
    parsed_step = _parse_step(step)
    original_dataset_id = _require_dataset_id(dataset_id)
    directory = _absolute(v3_artifact_dir(root, parsed_step.value, original_dataset_id))
    dataset_name = os.path.basename(directory)
    try:
        parent_chain, artifact_fd = _open_artifact_directory(
            root, parsed_step.value, dataset_name,
        )
    except FileNotFoundError as exc:
        raise ValueError(f"missing artifact directory: {directory!r}") from exc
    parent_fd = parent_chain.leaf_fd
    primary_error = None
    try:
        parent_chain.verify()
        _require_named_inode(parent_fd, dataset_name, artifact_fd)
        payload = _verify_artifact_snapshot(
            artifact_fd,
            parsed_step.value,
            original_dataset_id,
            expected_config_digest=expected_config_digest,
            expected_lineage_digest=expected_lineage_digest,
        )
        _require_named_inode(parent_fd, dataset_name, artifact_fd)
        parent_chain.verify()
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        first_error = None
        try:
            os.close(artifact_fd)
        except BaseException as exc:
            first_error = exc
        try:
            parent_chain.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        if first_error is not None:
            if primary_error is not None:
                _add_exception_note(
                    primary_error, f"artifact directory close failed: {first_error!r}",
                )
            else:
                raise first_error

    return _mark_verified_index(
        ArtifactIndex(
            **payload,
            artifact_dir=directory,
            output_root=root,
        ),
        production_bound=(
            expected_config_digest is not None
            and expected_lineage_digest is not None
        ),
    )


def open_artifact_for_read(
    output_root,
    step,
    dataset_id,
    *,
    expected_config_digest,
    expected_lineage_digest,
) -> ArtifactIndex:
    """读取生产 artifact，并强制绑定预期 config 与 lineage digest。"""
    return _open_artifact_index(
        output_root,
        step,
        dataset_id,
        expected_config_digest=_require_digest(
            expected_config_digest, "expected_config_digest",
        ),
        expected_lineage_digest=_require_digest(
            expected_lineage_digest, "expected_lineage_digest",
        ),
    )


def inspect_artifact_unbound(output_root, step, dataset_id) -> ArtifactIndex:
    """仅供诊断的无 expectation 检查；仍执行全部内部完整性校验。

    生产 stage 不得使用此入口替代 ``open_artifact_for_read()`` 的 provenance
    绑定。
    """
    return _open_artifact_index(
        output_root,
        step,
        dataset_id,
        expected_config_digest=None,
        expected_lineage_digest=None,
    )


def drop_artifact(output_root, step, dataset_id) -> None:
    """幂等删除一个已发布 artifact；目录不存在时静默成功。

    复用 ``_remove_owned_directory`` 的 fail-closed 校验：目录内出现任何
    非 contract 文件/非普通文件即 ``ValueError`` 拒绝删除，绝不递归清理
    未知 entry。用于配置变更（如 MOS 阈值调整）后失效 artifact 的显式重建。
    """
    root = _absolute(output_root)
    parsed_step = _parse_step(step)
    dataset_id = _require_dataset_id(dataset_id)
    dataset_name = os.path.basename(
        v3_artifact_dir(root, parsed_step.value, dataset_id)
    )
    try:
        parent_chain, artifact_fd = _open_artifact_directory(
            root, parsed_step.value, dataset_name,
        )
    except FileNotFoundError:
        return
    first_error = None
    try:
        _remove_owned_directory(parent_chain.leaf_fd, dataset_name, artifact_fd)
    except BaseException as exc:
        first_error = exc

    descriptor = artifact_fd
    artifact_fd = None
    try:
        os.close(descriptor)
    except BaseException as close_exc:
        first_error = _merge_resource_error(
            first_error, close_exc, "artifact fd close failed",
        )
    try:
        parent_chain.close()
    except BaseException as close_exc:
        first_error = _merge_resource_error(
            first_error, close_exc, "directory chain close failed",
        )
    if first_error is not None:
        raise first_error


def _run_integrity_check(
    first_error: Optional[BaseException], check, label: str,
) -> Optional[BaseException]:
    try:
        check()
    except BaseException as exc:
        return _record_verification_error(first_error, exc, label)
    return first_error


def _scan_artifact_rows(stream) -> Iterator[tuple[bytes, Mapping[str, Any]]]:
    """单遍解析 canonical JSONL；digest 与 count 由 reader 同步累计。"""
    count = 0
    for raw_line in stream:
        count += 1
        if not raw_line.endswith(b"\n"):
            raise ValueError(f"invalid artifact row {count}: missing LF newline")
        try:
            row = _load_json_object(raw_line[:-1], f"artifact row {count}")
        except ValueError as exc:
            raise ValueError(f"invalid artifact row {count}: {exc}") from exc
        if raw_line != _canonical_json_bytes(row) + b"\n":
            raise ValueError(f"invalid artifact row {count}: non-canonical bytes")
        yield raw_line, row


def _iter_spooled_rows(stream) -> Iterator[Mapping[str, Any]]:
    """解析已验证的私有 canonical spool，不再访问 artifact rows 源。"""
    for count, raw_line in enumerate(stream, 1):
        yield _load_json_object(raw_line[:-1], f"artifact spool row {count}")


def _spool_validated_rows(
    descriptor: int, *, expected_digest: str, expected_count: int,
):
    """单次扫描源 rows，并返回 positioned-at-start 的匿名 canonical spool。"""
    spool = tempfile.TemporaryFile(mode="w+b", prefix="audio-filter-rows-")
    source_stream = None
    source_fd = None
    primary_error = None
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        source_stream, source_fd = _open_owned_duplicate_stream(descriptor, "rb")
        digest = hashlib.sha256()
        count = 0
        for raw_line, _row in _scan_artifact_rows(source_stream):
            written = spool.write(raw_line)
            if written is not None and written != len(raw_line):
                raise OSError("short artifact spool write")
            digest.update(raw_line)
            count += 1
        if digest.hexdigest() != expected_digest:
            raise ValueError("rows hash mismatch or rows changed")
        if count != expected_count:
            raise ValueError("row count mismatch or rows changed")
        spool.flush()
        spool.seek(0)
    except BaseException as exc:
        primary_error = exc

    close_error = None
    if source_stream is not None:
        close_error = _close_owned_duplicate_stream(source_stream, source_fd)
    if primary_error is not None:
        _attach_resource_error(
            primary_error, close_error, "rows source resource close failed",
        )
        try:
            spool.close()
        except BaseException as close_exc:
            _add_exception_note(
                primary_error, f"rows spool close failed: {close_exc!r}",
            )
        raise primary_error
    if close_error is not None:
        try:
            spool.close()
        except BaseException as spool_close_error:
            _add_exception_note(
                close_error, f"rows spool close failed: {spool_close_error!r}",
            )
        raise close_error
    return spool


class ArtifactReader(Iterator[Mapping[str, Any]]):
    """绑定 artifact 描述符并持有已验证 rows 快照的单遍 reader。

    reader 在构造时固定 ancestor/artifact/contract fd，单次扫描 rows 源并写入
    私有 canonical 临时 spool；只有 digest/count、held fd、name、index 与 marker
    全部通过后才返回。业务迭代仅读取 spool，正常耗尽或提前 ``close()`` 仍会
    复核 held source capability，且不会再次读取 rows 源。
    """

    def __init__(
        self,
        output_root,
        step,
        dataset_id,
        *,
        expected_config_digest,
        expected_lineage_digest,
        _expected_serialized: Optional[dict] = None,
        _production_bound: bool = True,
    ):
        self._root = _absolute(output_root)
        self._step = _parse_step(step)
        self._dataset_id = _require_dataset_id(dataset_id)
        self._expected_config_digest = _require_digest(
            expected_config_digest, "expected_config_digest",
        )
        self._expected_lineage_digest = _require_digest(
            expected_lineage_digest, "expected_lineage_digest",
        )
        self._production_bound = _production_bound
        self._directory = _absolute(
            v3_artifact_dir(self._root, self._step.value, self._dataset_id)
        )
        self._dataset_name = os.path.basename(self._directory)
        self._parent_chain = None
        self._parent_fd = None
        self._artifact_fd = None
        self._contract_fds: dict[str, int] = {}
        self._contract_stats: dict[str, os.stat_result] = {}
        self._spool = None
        self._scanner = None
        self._closed = False
        self._index = None

        primary_error = None
        try:
            try:
                self._parent_chain, self._artifact_fd = _open_artifact_directory(
                    self._root, self._step.value, self._dataset_name,
                )
            except (FileNotFoundError, OSError) as exc:
                raise ValueError(
                    f"missing or changed artifact directory: {self._directory!r}"
                ) from exc
            self._parent_fd = self._parent_chain.leaf_fd
            self._parent_chain.verify()
            _require_named_inode(
                self._parent_fd, self._dataset_name, self._artifact_fd,
            )
            _validate_contract_names(self._artifact_fd)
            for name in ("index.json", "_SUCCESS", "rows.jsonl"):
                descriptor = _open_contract_file(self._artifact_fd, name)
                self._contract_fds[name] = descriptor
                self._contract_stats[name] = os.fstat(descriptor)

            index_raw = _read_bounded_fd(
                self._contract_fds["index.json"],
                limit=_INDEX_MAX_BYTES,
                label="index",
            )
            payload = _load_json_object(index_raw, "index")
            if index_raw != _canonical_json_bytes(payload) + b"\n":
                raise ValueError("non-canonical index bytes")
            _validate_index_payload(payload, self._step.value, self._dataset_id)
            if _expected_serialized is not None and payload != _expected_serialized:
                raise ValueError("artifact index changed after open")
            if payload["config_digest"] != self._expected_config_digest:
                raise ValueError("expected config digest mismatch")
            if payload["lineage_digest"] != self._expected_lineage_digest:
                raise ValueError("expected lineage digest mismatch")

            marker_fd = self._contract_fds["_SUCCESS"]
            if os.fstat(marker_fd).st_size != _MARKER_BYTES:
                raise ValueError("marker size mismatch")
            marker = _read_bounded_fd(
                marker_fd, limit=_MARKER_BYTES, label="marker",
            )
            if marker != (payload["artifact_digest"] + "\n").encode("ascii"):
                raise ValueError("marker digest mismatch or marker changed")

            self._index = ArtifactIndex(
                **payload,
                artifact_dir=self._directory,
                output_root=self._root,
            )
            self._spool = _spool_validated_rows(
                self._contract_fds["rows.jsonl"],
                expected_digest=payload["rows_sha256"],
                expected_count=payload["row_count"],
            )
            held_error = self._verify_held_contract()
            if held_error is not None:
                raise held_error
            _mark_verified_index(
                self._index, production_bound=self._production_bound,
            )
            self._scanner = _iter_spooled_rows(self._spool)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            if primary_error is not None:
                resource_error = self._release_resources()
                _attach_resource_error(
                    primary_error, resource_error, "artifact reader cleanup failed",
                )
                self._closed = True

    @property
    def index(self) -> ArtifactIndex:
        """返回当前 reader 绑定的 immutable metadata。"""
        return self._index

    @property
    def closed(self) -> bool:
        return self._closed

    def __iter__(self) -> "ArtifactReader":
        return self

    def _ensure_scanner(self) -> None:
        if self._scanner is None:
            raise RuntimeError("artifact reader spool is unavailable")

    def _verify_held_contract(self) -> Optional[BaseException]:
        """复核固定 fd 与公开 name；不再次读取 rows 内容。"""
        error = None
        expected = self._index.serialized()
        rows_fd = self._contract_fds["rows.jsonl"]
        error = _run_integrity_check(
            error,
            lambda: _require_same_file_snapshot(
                self._contract_stats["rows.jsonl"],
                rows_fd,
                "artifact rows fd changed during iteration",
            ),
            "held rows snapshot verification failed",
        )
        for name, descriptor in self._contract_fds.items():
            error = _run_integrity_check(
                error,
                lambda name=name, descriptor=descriptor: _require_named_inode(
                    self._artifact_fd, name, descriptor,
                ),
                f"held contract binding verification failed for {name!r}",
            )

        def verify_index() -> None:
            index_fd = self._contract_fds["index.json"]
            raw = _read_bounded_fd(
                index_fd, limit=_INDEX_MAX_BYTES, label="index",
            )
            payload = _load_json_object(raw, "index")
            if raw != _canonical_json_bytes(payload) + b"\n":
                raise ValueError("non-canonical index bytes")
            _validate_index_payload(payload, self._step.value, self._dataset_id)
            if payload != expected:
                raise ValueError("artifact index changed after open")

        error = _run_integrity_check(
            error, verify_index, "held index verification failed",
        )

        def verify_marker() -> None:
            marker_fd = self._contract_fds["_SUCCESS"]
            if os.fstat(marker_fd).st_size != _MARKER_BYTES:
                raise ValueError("marker size mismatch")
            marker = _read_bounded_fd(
                marker_fd, limit=_MARKER_BYTES, label="marker",
            )
            if marker != (expected["artifact_digest"] + "\n").encode("ascii"):
                raise ValueError("marker digest mismatch or marker changed")

        error = _run_integrity_check(
            error, verify_marker, "held marker verification failed",
        )
        for name, descriptor in self._contract_fds.items():
            error = _run_integrity_check(
                error,
                lambda name=name, descriptor=descriptor: _require_same_file_snapshot(
                    self._contract_stats[name],
                    descriptor,
                    f"artifact contract fd changed: {name!r}",
                ),
                f"held contract snapshot verification failed for {name!r}",
            )
        error = _run_integrity_check(
            error,
            lambda: _validate_contract_names(self._artifact_fd),
            "artifact contract enumeration failed",
        )
        error = _run_integrity_check(
            error,
            lambda: _require_contract_named_stats(
                self._artifact_fd, self._contract_stats,
            ),
            "artifact contract binding verification failed",
        )
        error = _run_integrity_check(
            error,
            lambda: _require_named_inode(
                self._parent_fd, self._dataset_name, self._artifact_fd,
            ),
            "artifact directory binding verification failed",
        )
        error = _run_integrity_check(
            error,
            self._parent_chain.verify,
            "artifact ancestor binding verification failed",
        )
        return error

    def _release_resources(self) -> Optional[BaseException]:
        error = None
        self._scanner = None
        spool = self._spool
        self._spool = None
        if spool is not None:
            try:
                spool.close()
            except BaseException as exc:
                error = _merge_resource_error(
                    error, exc, "rows spool close failed",
                )
        descriptors = self._contract_fds
        self._contract_fds = {}
        for name, descriptor in descriptors.items():
            try:
                os.close(descriptor)
            except BaseException as exc:
                error = _merge_resource_error(
                    error, exc, f"contract fd close failed for {name!r}",
                )
        artifact_fd = self._artifact_fd
        self._artifact_fd = None
        if artifact_fd is not None:
            try:
                os.close(artifact_fd)
            except BaseException as exc:
                error = _merge_resource_error(
                    error, exc, "artifact fd close failed",
                )
        parent_chain = self._parent_chain
        self._parent_chain = None
        self._parent_fd = None
        if parent_chain is not None:
            try:
                parent_chain.close()
            except BaseException as exc:
                error = _merge_resource_error(
                    error, exc, "reader parent close failed",
                )
        return error

    def _terminate(self, primary: BaseException) -> None:
        integrity_error = self._verify_held_contract()
        if integrity_error is not None:
            _add_exception_note(
                primary, f"artifact close-time verification failed: {integrity_error!r}",
            )
            for note in getattr(integrity_error, "__notes__", ()):
                _add_exception_note(primary, note)
        resource_error = self._release_resources()
        _attach_resource_error(
            primary, resource_error, "artifact reader resource close failed",
        )
        self._closed = True

    def _finish(self) -> None:
        integrity_error = self._verify_held_contract()
        resource_error = self._release_resources()
        self._closed = True
        if integrity_error is not None:
            _attach_resource_error(
                integrity_error,
                resource_error,
                "artifact reader resource close failed",
            )
            raise integrity_error
        if resource_error is not None:
            raise resource_error
        _mark_verified_index(
            self._index, production_bound=self._production_bound,
        )

    def __next__(self) -> Mapping[str, Any]:
        if self._closed:
            raise StopIteration
        try:
            self._ensure_scanner()
            return next(self._scanner)
        except StopIteration:
            self._finish()
            raise
        except BaseException as exc:
            self._terminate(exc)
            raise

    def close(self) -> None:
        """提前关闭时复核 held source capability，并释放 spool 与全部 fd。"""
        if self._closed:
            return
        try:
            self._finish()
        except BaseException as exc:
            if not self._closed:
                self._terminate(exc)
            raise

    def __enter__(self) -> "ArtifactReader":
        if self._closed:
            raise RuntimeError("artifact reader is already closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            self.close()
        except BaseException as close_exc:
            if exc_value is None:
                raise
            _add_exception_note(
                exc_value, f"artifact reader close failed: {close_exc!r}",
            )
        return False

    def __del__(self):  # pragma: no cover - 仅作为遗忘显式 close 的资源兜底
        if not getattr(self, "_closed", True):
            try:
                self._release_resources()
            except BaseException:
                pass
            self._closed = True


def open_artifact_reader(
    output_root,
    step,
    dataset_id,
    *,
    expected_config_digest,
    expected_lineage_digest,
) -> ArtifactReader:
    """打开 descriptor-bound 单遍生产 reader。"""
    return ArtifactReader(
        output_root,
        step,
        dataset_id,
        expected_config_digest=expected_config_digest,
        expected_lineage_digest=expected_lineage_digest,
    )


def iter_artifact_rows(index: ArtifactIndex) -> Iterator[Mapping[str, Any]]:
    """从 verified index 打开 reader，并保留原 opener 的 capability 等级。"""
    if not isinstance(index, ArtifactIndex):
        raise TypeError("index must be an ArtifactIndex")
    registration = _verified_index_registration(index)
    if registration is None:
        raise ValueError("index must come from a verified public opener")
    parsed_step = _parse_step(index.step)
    root = _absolute(index.output_root)
    canonical_dir = _absolute(
        v3_artifact_dir(root, parsed_step.value, index.dataset_id)
    )
    if index.output_root != root or index.artifact_dir != canonical_dir:
        raise ValueError("artifact runtime path is not canonical")
    return ArtifactReader(
        root,
        parsed_step,
        index.dataset_id,
        expected_config_digest=index.config_digest,
        expected_lineage_digest=index.lineage_digest,
        _expected_serialized=index.serialized(),
        _production_bound=registration[2],
    )
