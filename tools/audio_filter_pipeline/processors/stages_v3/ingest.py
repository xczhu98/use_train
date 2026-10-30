"""Artifact-backed Step0 source ingest stage。

Step0 行是不可变历史源快照；后续 Step1 必须对实际解码的 descriptor 复验
``source_size_bytes`` 与 ``source_sha256``，源删除或不匹配时 fail closed。快照发布后
新增文件不属于该快照，恢复有效 Step0 artifact 时也不会重新扫描 live input。
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import math
import os
import sqlite3
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import soundfile

from ...config import DatasetSpec, PipelineConfig
from ...utils.artifacts import ArtifactIndex, PipelineStep
from ...utils.path_utils import AUDIO_EXTENSIONS, make_source_identity
from ...utils.speaker_metadata import (
    SpeakerMetadataResolver,
    validate_source_relative_path,
)
from . import aggregate_source
from .aggregate_source import (
    AGGREGATE_SOURCE_LAYOUT,
    AggregateDescriptor,
    AggregateMember,
    AggregateMemberEmptyError,
    scan_aggregate_directory,
    verify_aggregate_identity,
)
from .base import ArtifactStage, StageRunContext


logger = logging.getLogger(__name__)
_EXCEPTION_SHUTDOWN_TIMEOUT_SECONDS = 1.0


def _add_exception_note(error: BaseException, note: str) -> None:
    """将 executor 清理诊断附加到 primary exception，绝不替换它。"""
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)
        return
    notes = list(getattr(error, "__notes__", ()))
    notes.append(note)
    error.__notes__ = notes


def _executor_members(executor: Any, attribute: str) -> tuple[Any, ...]:
    """读取 concurrent.futures 私有成员；不存在时只返回空快照。"""
    try:
        members = getattr(executor, attribute, ())
        if isinstance(members, Mapping):
            members = members.values()
        members = tuple(members or ())
    except BaseException:
        return ()
    unique = {}
    for member in members:
        unique.setdefault(id(member), member)
    return tuple(unique.values())


def _process_is_alive(process: Any) -> bool:
    poll = getattr(process, "poll", None)
    if callable(poll):
        try:
            return poll() is None
        except BaseException:
            return True
    is_alive = getattr(process, "is_alive", None)
    if callable(is_alive):
        try:
            return bool(is_alive())
        except BaseException:
            return True
    return True


def _bounded_wait_process(process: Any, deadline: float, diagnostics: list[str]) -> None:
    """只用带 timeout 的 wait/join，绝不回退到无界等待。"""
    waiter = getattr(process, "join", None) or getattr(process, "wait", None)
    if not callable(waiter):
        diagnostics.append("child has no bounded wait/join API")
        return
    remaining = max(0.0, deadline - time.monotonic())
    try:
        waiter(timeout=remaining)
    except TypeError:
        diagnostics.append("child bounded wait API rejected timeout")
    except BaseException as exc:
        diagnostics.append(f"child bounded wait failed: {exc!r}")


def _terminate_executor_children(
    processes: tuple[Any, ...],
    deadline: float,
    diagnostics: list[str],
) -> None:
    """best-effort 终止 executor 可见的 child processes，所有等待均有界。"""
    for process in processes:
        if not _process_is_alive(process):
            continue
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            try:
                terminate()
            except BaseException as exc:
                diagnostics.append(f"child terminate failed: {exc!r}")
        _bounded_wait_process(process, deadline, diagnostics)
        if not _process_is_alive(process):
            continue
        kill = getattr(process, "kill", None)
        if callable(kill):
            try:
                kill()
            except BaseException as exc:
                diagnostics.append(f"child kill failed: {exc!r}")
        _bounded_wait_process(process, deadline, diagnostics)
        if _process_is_alive(process):
            diagnostics.append("child remained alive after bounded terminate/kill")


def _shutdown_executor_bounded(
    executor: Any,
    *,
    pending_futures: Iterable[Any],
    stage: str,
    workers: int,
    primary_error: BaseException | None = None,
    timeout_seconds: float = _EXCEPTION_SHUTDOWN_TIMEOUT_SECONDS,
) -> None:
    """异常路径非阻塞关闭 executor，并有界清理可访问 worker。

    ``ProcessPoolExecutor._processes`` 与 ``ThreadPoolExecutor._threads`` 是私有
    API，只作为 best-effort 诊断/清理快照；轻量测试 executor 缺少这些字段时仍
    只调用其公开 ``shutdown(wait=False)``，不会再次执行无界 wait。
    """
    diagnostics: list[str] = []
    pending = tuple(pending_futures)
    cancelled = 0
    for future in pending:
        try:
            cancelled += int(bool(future.cancel()))
        except BaseException as exc:
            diagnostics.append(f"future cancel failed: {exc!r}")

    # 在 shutdown 触发 manager thread 清空私有集合前先做快照；私有 API
    # 只用于 best-effort child 终止，缺失时仍不影响轻量测试 executor。
    processes = _executor_members(executor, "_processes")
    threads = _executor_members(executor, "_threads")
    try:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # Python 3.8 或轻量测试 executor 可能没有 cancel_futures 参数。
            executor.shutdown(wait=False)
    except BaseException as exc:
        diagnostics.append(f"executor shutdown failed: {exc!r}")

    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    _terminate_executor_children(processes, deadline, diagnostics)
    for thread in threads:
        if not getattr(thread, "is_alive", lambda: False)():
            continue
        remaining = max(0.0, deadline - time.monotonic())
        try:
            thread.join(timeout=remaining)
        except BaseException as exc:
            diagnostics.append(f"worker thread join failed: {exc!r}")
        if getattr(thread, "is_alive", lambda: False)():
            diagnostics.append("worker thread remained alive after bounded join")

    summary = (
        f"stage={stage} workers={workers} pending={len(pending)} "
        f"cancelled={cancelled} timeout={float(timeout_seconds):.3f}s "
        f"child_processes={len(processes)} worker_threads={len(threads)}"
    )
    if diagnostics:
        summary += " diagnostics=" + "; ".join(diagnostics)
    logger.warning("bounded executor shutdown: %s", summary)
    if primary_error is not None:
        _add_exception_note(primary_error, f"bounded executor shutdown: {summary}")


STEP0_ROW_FIELDS = (
    "source_key",
    "dataset_id",
    "language",
    "category",
    "quality_profile",
    "input_root",
    "original_path",
    "original_relative_path",
    "source_relative_path",
    "source_size_bytes",
    "source_sha256",
    "sample_rate",
    "frames",
    "channels",
    "duration_sec",
    "format",
    "subtype",
    "speaker_id",
    "speaker_scope",
)


def _step0_row(**values: Any) -> dict[str, Any]:
    if set(values) != set(STEP0_ROW_FIELDS):
        raise RuntimeError("Step0 row fields do not match the declared row contract")
    return {field: values[field] for field in STEP0_ROW_FIELDS}


def _canonical_dataset(spec: DatasetSpec) -> DatasetSpec:
    if not isinstance(spec, DatasetSpec):
        raise TypeError("dataset must be a DatasetSpec")
    if type(spec.input_root) is not str or not spec.input_root.strip():
        raise ValueError("dataset input_root must be a non-empty string")
    return replace(spec, input_root=os.path.realpath(os.path.abspath(spec.input_root)))


def _source_selector_payload(spec: DatasetSpec) -> dict[str, list[Any]]:
    """返回 selector 的 canonical JSON-safe payload；未配置时返回空 mapping。"""
    selector = spec.source_selector
    if (
        not selector.sample_rates_hz
        and not selector.include_speaker_ids
        and not selector.exclude_relative_paths
    ):
        return {}
    payload: dict[str, list[Any]] = {}
    if selector.sample_rates_hz:
        payload["sample_rates_hz"] = list(selector.sample_rates_hz)
    if selector.include_speaker_ids:
        payload["include_speaker_ids"] = list(selector.include_speaker_ids)
    if selector.exclude_relative_paths:
        payload["exclude_relative_paths"] = list(selector.exclude_relative_paths)
    return payload


def _inode_identity(snapshot: os.stat_result) -> tuple[int, int, int]:
    return snapshot.st_dev, snapshot.st_ino, stat.S_IFMT(snapshot.st_mode)


def _file_snapshot(snapshot: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        snapshot.st_dev,
        snapshot.st_ino,
        snapshot.st_mode,
        snapshot.st_size,
        snapshot.st_mtime_ns,
        snapshot.st_ctime_ns,
        snapshot.st_nlink,
    )


def _require_named_inode(
    parent_descriptor: int,
    name: str,
    held_descriptor: int,
    *,
    label: str,
) -> os.stat_result:
    try:
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} pathname binding changed: {name!r}") from exc
    if _inode_identity(named) != _inode_identity(os.fstat(held_descriptor)):
        raise ValueError(f"{label} pathname binding changed: {name!r}")
    return named


def _close_descriptors(
    descriptors: Iterable[int],
    *,
    primary_error: BaseException | None = None,
    label: str,
) -> None:
    first_error: BaseException | None = None
    for descriptor in reversed(tuple(descriptors)):
        try:
            os.close(descriptor)
        except BaseException as exc:
            if first_error is None:
                first_error = exc
            elif primary_error is not None:
                _add_exception_note(primary_error, f"{label} close failed: {exc!r}")
    if first_error is None:
        return
    if primary_error is not None:
        _add_exception_note(primary_error, f"{label} close failed: {first_error!r}")
        return
    raise first_error


def _close_iterator(
    iterator: Any,
    *,
    primary_error: BaseException | None = None,
    label: str,
) -> None:
    closer = getattr(iterator, "close", None)
    if not callable(closer):
        return
    try:
        closer()
    except BaseException as exc:
        if primary_error is not None:
            _add_exception_note(primary_error, f"{label} close failed: {exc!r}")
            return
        raise


def _hash_source_descriptor(descriptor: int, expected_size: int) -> str:
    """按首次 held snapshot 大小 hash exact bytes，且不改变共享 offset。"""
    digest = hashlib.sha256()
    offset = 0
    while offset < expected_size:
        remaining = expected_size - offset
        chunk = os.pread(descriptor, min(1024 * 1024, remaining), offset)
        if not chunk:
            raise ValueError("source descriptor hash reached EOF before expected size")
        digest.update(chunk)
        offset += len(chunk)

    if os.pread(descriptor, 1, expected_size):
        raise ValueError("source descriptor hash has extra data beyond expected size")
    return digest.hexdigest()


def _read_audio_header(descriptor: int):
    """通过 owned duplicate 投影 header，并恢复 held descriptor 的 offset。"""
    initial_offset = os.lseek(descriptor, 0, os.SEEK_CUR)
    duplicate = os.dup(descriptor)
    try:
        try:
            duplicate_stream = os.fdopen(duplicate, "rb")
        except BaseException:
            os.close(duplicate)
            raise
        with duplicate_stream:
            return soundfile.info(duplicate_stream)
    finally:
        os.lseek(descriptor, initial_offset, os.SEEK_SET)


def _descriptor_path(descriptor: int, label: str) -> str:
    proc_path = f"/proc/self/fd/{descriptor}"
    if not os.path.exists(proc_path):
        raise ValueError(f"cannot verify {label} descriptor canonical path")
    target = os.path.realpath(proc_path)
    if target.endswith(" (deleted)"):
        raise ValueError(f"{label} descriptor target changed")
    return target


def _contained(path: str, root: str, *, single_file: bool) -> bool:
    if single_file:
        return path == root
    try:
        return os.path.commonpath((root, path)) == root and path != root
    except (OSError, ValueError):
        return False


def _positive_header_integer(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"audio {field_name} must be a positive integer")
    return value


def _header_string(value: Any, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"audio {field_name} must be a non-empty string")
    return value


def _duration_sec(frames: int, samplerate: int) -> float:
    return frames / samplerate


@dataclass(frozen=True)
class _ScannedSource:
    """descriptor walker 冻结的 source pathname 与 inode snapshot。"""

    path: str
    snapshot: tuple[int, int, int, int, int, int, int]
    aggregate: AggregateDescriptor | None = None


def _aggregate_to_payload(aggregate: AggregateDescriptor | None) -> str | None:
    if aggregate is None:
        return None
    return json.dumps(
        {
            "input_root": aggregate.input_root,
            "source_relative_path": aggregate.source_relative_path,
            "directory_snapshot": aggregate.directory_snapshot,
            "members": [
                {
                    "index": member.index,
                    "name": member.name,
                    "size_bytes": member.size_bytes,
                    "raw_sha256": member.raw_sha256,
                    "sample_rate": member.sample_rate,
                    "frames": member.frames,
                    "channels": member.channels,
                    "format": member.format,
                    "subtype": member.subtype,
                    "snapshot": member.snapshot,
                }
                for member in aggregate.members
            ],
            "source_size_bytes": aggregate.source_size_bytes,
            "frames": aggregate.frames,
            "sample_rate": aggregate.sample_rate,
            "channels": aggregate.channels,
            "format": aggregate.format,
            "subtype": aggregate.subtype,
            "manifest_sha256": aggregate.manifest_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _aggregate_from_payload(payload: str | None) -> AggregateDescriptor | None:
    if payload is None:
        return None
    try:
        value = json.loads(payload)
        if not isinstance(value, Mapping):
            raise TypeError("aggregate payload is not a mapping")
        members = tuple(
            AggregateMember(
                index=member["index"],
                name=member["name"],
                size_bytes=member["size_bytes"],
                raw_sha256=member["raw_sha256"],
                sample_rate=member["sample_rate"],
                frames=member["frames"],
                channels=member["channels"],
                format=member["format"],
                subtype=member["subtype"],
                snapshot=tuple(member["snapshot"]),
            )
            for member in value["members"]
        )
        return AggregateDescriptor(
            input_root=value["input_root"],
            source_relative_path=value["source_relative_path"],
            directory_snapshot=tuple(value["directory_snapshot"]),
            members=members,
            source_size_bytes=value["source_size_bytes"],
            frames=value["frames"],
            sample_rate=value["sample_rate"],
            channels=value["channels"],
            format=value["format"],
            subtype=value["subtype"],
            manifest_sha256=value["manifest_sha256"],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("source scan spool aggregate payload is invalid") from exc


class _SourceSpool:
    """用 SQLite 临时排序 descriptor scan 结果，避免累积全量 source list。"""

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(
            prefix="audio-filter-step0-sources-",
        )
        self._connection: sqlite3.Connection | None = None
        self._closed = False
        try:
            self._connection = sqlite3.connect(
                os.path.join(self._temporary.name, "sources.sqlite3")
            )
            self._connection.execute("PRAGMA temp_store=FILE")
            self._connection.execute(
                "CREATE TABLE sources (path TEXT PRIMARY KEY, snapshot TEXT NOT NULL, "
                "aggregate TEXT)"
            )
        except BaseException:
            if self._connection is not None:
                self._connection.close()
            self._temporary.cleanup()
            self._closed = True
            raise

    def _require_connection(self) -> sqlite3.Connection:
        if self._closed or self._connection is None:
            raise RuntimeError("source scan spool is closed")
        return self._connection

    def append(self, source: _ScannedSource) -> None:
        connection = self._require_connection()
        try:
            connection.execute(
                "INSERT INTO sources(path, snapshot, aggregate) VALUES (?, ?, ?)",
                (
                    source.path,
                    json.dumps(source.snapshot, separators=(",", ":")),
                    _aggregate_to_payload(source.aggregate),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"duplicate source pathname in Step0 scan: {source.path!r}") from exc

    def iter_sources(self) -> Iterator[_ScannedSource]:
        connection = self._require_connection()
        connection.commit()
        cursor = connection.execute(
            "SELECT path, snapshot, aggregate FROM sources ORDER BY path"
        )
        try:
            for path, snapshot, aggregate in cursor:
                try:
                    parsed_snapshot = tuple(json.loads(snapshot))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError("source scan spool snapshot is invalid") from exc
                if len(parsed_snapshot) != 7:
                    raise ValueError("source scan spool snapshot has invalid length")
                yield _ScannedSource(
                    path=path,
                    snapshot=parsed_snapshot,
                    aggregate=_aggregate_from_payload(aggregate),
                )
        finally:
            cursor.close()

    def close(self, primary_error: BaseException | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        cleanup_error: BaseException | None = None
        try:
            if self._connection is not None:
                self._connection.close()
        except BaseException as exc:
            cleanup_error = exc
        try:
            self._temporary.cleanup()
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
        if cleanup_error is None:
            return
        if primary_error is not None:
            _add_exception_note(primary_error, f"source scan spool cleanup failed: {cleanup_error!r}")
            return
        raise cleanup_error


@dataclass(frozen=True)
class _ProcessedSource:
    """线程返回的最终行；被 selector 或采样率门禁过滤时直接返回 ``None``。"""

    row: dict[str, Any]


@dataclass
class _HeldSource:
    """持有 root 到 terminal parent 的目录链及 source descriptor。"""

    directory_descriptors: list[int]
    directory_names: tuple[str, ...]
    source_name: str
    descriptor: int
    snapshot: tuple[int, int, int, int, int, int, int]
    original_path: str
    aggregate: AggregateDescriptor | None = None

    def verify_chain(
        self,
        *,
        verify_aggregate: bool = True,
        aggregate_full_scan: bool = True,
    ) -> None:
        for parent, name, child in zip(
            self.directory_descriptors,
            self.directory_names,
            self.directory_descriptors[1:],
        ):
            _require_named_inode(parent, name, child, label="source directory component")
        named = os.stat(
            self.source_name,
            dir_fd=self.directory_descriptors[-1],
            follow_symlinks=False,
        )
        if self.aggregate is None:
            valid = stat.S_ISREG(named.st_mode) and _file_snapshot(named) == self.snapshot
        else:
            valid = stat.S_ISDIR(named.st_mode) and _file_snapshot(named) == self.snapshot
        if not valid:
            raise ValueError(f"source pathname binding changed: {self.source_name!r}")
        if verify_aggregate and self.aggregate is not None:
            verify_aggregate_identity(
                self.descriptor,
                self.aggregate,
                full_scan=aggregate_full_scan,
            )

    def close(self, primary_error: BaseException | None = None) -> None:
        descriptors = [self.descriptor, *self.directory_descriptors]
        self.descriptor = -1
        self.directory_descriptors = []
        _close_descriptors(
            descriptors,
            primary_error=primary_error,
            label="held source descriptor",
        )


class _SubmittedRootDescriptor:
    """提交方持有 duplicated root fd；worker 开始时原子接管。"""

    def __init__(self, descriptor: int):
        self._descriptor = descriptor
        self._cleanup_error: BaseException | None = None
        self._lock = threading.Lock()

    def take(self) -> int:
        with self._lock:
            if self._descriptor is None:
                raise RuntimeError("submitted root descriptor is no longer owned")
            descriptor = self._descriptor
            self._descriptor = None
            return descriptor

    def close_if_owned(self, primary_error: BaseException | None = None) -> None:
        with self._lock:
            descriptor = self._descriptor
            self._descriptor = None
            cleanup_error = self._cleanup_error
            self._cleanup_error = None
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
        if cleanup_error is None:
            return
        if primary_error is not None:
            _add_exception_note(
                primary_error,
                f"submitted root descriptor close failed: {cleanup_error!r}",
            )
            return
        raise cleanup_error

    def close_from_cancel_callback(self) -> None:
        """Future callback 无 primary；保存 cleanup error 供池化 finally 归并。"""
        with self._lock:
            descriptor = self._descriptor
            self._descriptor = None
        if descriptor is None:
            return
        try:
            os.close(descriptor)
        except BaseException as exc:
            with self._lock:
                self._cleanup_error = self._cleanup_error or exc


@dataclass(frozen=True)
class Step0IngestStage(ArtifactStage):
    """Scan one immutable dataset and publish canonical Step0 source rows."""

    dataset: DatasetSpec
    mapping_base_dir: Optional[str] = None
    recursive: bool = True
    # 并发哈希/读头线程数：os.pread 与 hashlib 在 C 层释放 GIL，线程足够即可
    # 并行 IO；线程共享同一 root descriptor，descriptor-bound fail-closed 校验
    # 原样保留。workers<=1 时退化为原串行行为。
    workers: int = 1
    _root_identity: Optional[tuple[int, int, int]] = field(
        init=False, repr=False, compare=False,
    )
    _resolver: SpeakerMetadataResolver = field(init=False, repr=False, compare=False)

    step = PipelineStep.STEP0
    upstream_step = None

    def __post_init__(self) -> None:
        canonical = _canonical_dataset(self.dataset)
        if type(self.recursive) is not bool:
            raise ValueError("recursive must be bool")
        root_identity = None
        try:
            root_identity = _inode_identity(os.stat(canonical.input_root, follow_symlinks=False))
        except FileNotFoundError:
            pass
        resolver = SpeakerMetadataResolver(
            canonical.speaker,
            input_root=canonical.input_root,
            mapping_base_dir=self.mapping_base_dir,
        )
        object.__setattr__(self, "dataset", canonical)
        object.__setattr__(self, "_root_identity", root_identity)
        object.__setattr__(self, "_resolver", resolver)
        if canonical.speaker.source == "mapping_file":
            object.__setattr__(self, "mapping_base_dir", resolver.mapping_base_dir)

    def _bound_config_dataset(self, config: PipelineConfig) -> DatasetSpec:
        configured = config.dataset_spec(self.dataset.dataset_id)
        if not os.path.isabs(configured.input_root):
            raise ValueError("config dataset mismatch: input_root must be absolute")
        if _canonical_dataset(configured) != self.dataset:
            raise ValueError("config dataset mismatch with stage-bound dataset")
        return self.dataset

    def _min_source_sample_rate_hz(self, config: PipelineConfig) -> Optional[int]:
        value = getattr(config, "min_source_sample_rate_hz", None)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError("min_source_sample_rate_hz must be a positive integer or None")
        return value

    def config_payload(self, config: PipelineConfig) -> Mapping[str, Any]:
        dataset = self._bound_config_dataset(config)
        dataset_payload = {
            "dataset_id": dataset.dataset_id,
            "input_root": dataset.input_root,
            "language": dataset.language,
            "category": dataset.category,
            "quality_profile": dataset.quality_profile,
        }
        selector_payload = _source_selector_payload(dataset)
        if selector_payload:
            dataset_payload["source_selector"] = selector_payload
        if dataset.source_layout != "regular_file":
            dataset_payload["source_layout"] = dataset.source_layout
        scanner_payload = {
            "version": 3 if dataset.source_layout == "regular_file" else 4,
            "recursive": self.recursive,
            "audio_extensions": sorted(AUDIO_EXTENSIONS),
        }
        if dataset.source_layout != "regular_file":
            scanner_payload["source_layout"] = dataset.source_layout
            scanner_payload["manifest"] = {
                "domain": aggregate_source.AGGREGATE_MANIFEST_DOMAIN,
                "version": aggregate_source.AGGREGATE_MANIFEST_VERSION,
            }
            scanner_payload["policy"] = {
                "scanner_version": aggregate_source.AGGREGATE_SCANNER_VERSION,
                "policy_version": aggregate_source.AGGREGATE_POLICY_VERSION,
                "read_chunk_frames": aggregate_source.AGGREGATE_READ_CHUNK_FRAMES,
                "max_member_frames": aggregate_source.AGGREGATE_MAX_MEMBER_FRAMES,
                "max_members": aggregate_source.AGGREGATE_MAX_MEMBERS,
            }
        min_source_sample_rate_hz = self._min_source_sample_rate_hz(config)
        if min_source_sample_rate_hz is not None:
            scanner_payload["min_source_sample_rate_hz"] = min_source_sample_rate_hz
        return {
            "dataset": dataset_payload,
            "speaker": self._resolver.config_payload(),
            "scanner": scanner_payload,
            "row_contract": {
                "version": 5,
                "fields": list(STEP0_ROW_FIELDS),
            },
        }

    def _validate_context_dataset(self, context: StageRunContext) -> None:
        if context.dataset_id != self.dataset.dataset_id:
            raise ValueError("context dataset mismatch with stage-bound dataset")
        self._bound_config_dataset(context.config)

    def _validate_contract(self, context: StageRunContext) -> None:
        super()._validate_contract(context)
        self._validate_context_dataset(context)

    def _open_root_descriptor(self) -> tuple[int, bool]:
        input_root = self.dataset.input_root
        if self._root_identity is None:
            raise ValueError(f"input_root did not exist at stage construction: {input_root!r}")
        root_type = self._root_identity[2]
        single_file = root_type == stat.S_IFREG
        if not single_file and root_type != stat.S_IFDIR:
            raise ValueError("input_root must be a regular audio file or directory")
        if single_file and Path(input_root).suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(
                f"input_root regular file has unsupported audio extension: {input_root!r}"
            )
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
        if not single_file:
            flags |= getattr(os, "O_DIRECTORY", 0)
        try:
            descriptor = os.open(input_root, flags)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValueError("input_root must not be a symlink") from exc
            raise ValueError(f"cannot open input_root {input_root!r}: {exc}") from exc
        try:
            self._verify_root_descriptor(descriptor, single_file)
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor, single_file

    def _verify_root_descriptor(self, descriptor: int, single_file: bool) -> None:
        current = os.fstat(descriptor)
        if _inode_identity(current) != self._root_identity:
            raise ValueError("input_root identity changed")
        if _descriptor_path(descriptor, "input_root") != self.dataset.input_root:
            raise ValueError("input_root pathname binding changed")
        try:
            named = os.stat(self.dataset.input_root, follow_symlinks=False)
        except OSError as exc:
            raise ValueError("input_root pathname binding changed") from exc
        if _inode_identity(named) != self._root_identity:
            raise ValueError("input_root identity changed")
        expected_type = stat.S_IFREG if single_file else stat.S_IFDIR
        if stat.S_IFMT(current.st_mode) != expected_type:
            raise ValueError("input_root type changed")

    def _verify_empty_aggregate_bindings(
        self,
        directory_descriptor: int,
        name: str,
        child_descriptor: int,
        relative: tuple[str, ...],
        child_snapshot: tuple[int, int, int, int, int, int, int],
        directory_snapshot: tuple[int, int, int, int, int, int, int],
        root_descriptor: int | None,
    ) -> None:
        _require_named_inode(
            directory_descriptor,
            name,
            child_descriptor,
            label="aggregate directory",
        )
        if (
            _file_snapshot(os.fstat(child_descriptor)) != child_snapshot
            or not stat.S_ISDIR(os.fstat(child_descriptor).st_mode)
        ):
            raise ValueError(f"aggregate directory changed while scanning: {name!r}")
        expected_path = os.path.join(self.dataset.input_root, *relative)
        if _descriptor_path(child_descriptor, "aggregate directory") != expected_path:
            raise ValueError("aggregate directory pathname binding changed")
        if _file_snapshot(os.fstat(directory_descriptor)) != directory_snapshot:
            raise ValueError("source directory changed while scanning")
        if root_descriptor is not None:
            self._verify_root_descriptor(root_descriptor, False)

    def _scan_directory_descriptor(
        self,
        directory_descriptor: int,
        components: tuple[str, ...],
        append_source: Callable[[_ScannedSource], None],
        *,
        root_descriptor: int | None = None,
    ) -> None:
        """只从 held directory fd 枚举，并复验扫描期间目录未被换绑。"""
        directory_snapshot = _file_snapshot(os.fstat(directory_descriptor))
        with os.scandir(directory_descriptor) as entries:
            entry_snapshots = sorted(
                (entry.name, entry.stat(follow_symlinks=False)) for entry in entries
            )
        if _file_snapshot(os.fstat(directory_descriptor)) != directory_snapshot:
            raise ValueError("source directory changed while scanning")

        for name, entry_snapshot in entry_snapshots:
            if stat.S_ISLNK(entry_snapshot.st_mode):
                raise ValueError(
                    f"source directory component or source is a symlink: {name!r}"
                )
            if stat.S_ISREG(entry_snapshot.st_mode):
                if Path(name).suffix.lower() in AUDIO_EXTENSIONS:
                    source_flags = (
                        os.O_RDONLY
                        | getattr(os, "O_NONBLOCK", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                    )
                    source_descriptor = os.open(
                        name, source_flags, dir_fd=directory_descriptor,
                    )
                    try:
                        named = _require_named_inode(
                            directory_descriptor,
                            name,
                            source_descriptor,
                            label="source",
                        )
                        opened_snapshot = _file_snapshot(os.fstat(source_descriptor))
                        if (
                            _file_snapshot(named) != opened_snapshot
                            or opened_snapshot != _file_snapshot(entry_snapshot)
                        ):
                            raise ValueError(f"source changed while scanning: {name!r}")
                    finally:
                        os.close(source_descriptor)
                    relative = (*components, name)
                    append_source(_ScannedSource(
                        path=os.path.join(self.dataset.input_root, *relative),
                        snapshot=opened_snapshot,
                    ))
                continue
            if not stat.S_ISDIR(entry_snapshot.st_mode):
                continue
            relative = (*components, name)
            is_aggregate = (
                self.dataset.source_layout == AGGREGATE_SOURCE_LAYOUT
                and name.endswith("_active")
            )
            if is_aggregate:
                directory_flags = (
                    os.O_RDONLY
                    | getattr(os, "O_NONBLOCK", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_DIRECTORY", 0)
                )
                try:
                    child = os.open(name, directory_flags, dir_fd=directory_descriptor)
                except OSError as exc:
                    raise ValueError(
                        f"aggregate directory changed while scanning: {name!r}"
                    ) from exc
                primary_error: BaseException | None = None
                try:
                    named = _require_named_inode(
                        directory_descriptor, name, child, label="aggregate directory",
                    )
                    child_snapshot = _file_snapshot(os.fstat(child))
                    if (
                        _file_snapshot(named) != child_snapshot
                        or child_snapshot != _file_snapshot(entry_snapshot)
                    ):
                        raise ValueError(
                            f"aggregate directory changed while scanning: {name!r}"
                        )
                    aggregate = scan_aggregate_directory(
                        child,
                        self.dataset.input_root,
                        "/".join(relative),
                    )
                    if aggregate.directory_snapshot != child_snapshot:
                        raise ValueError(
                            f"aggregate directory changed while scanning: {name!r}"
                        )
                    _require_named_inode(
                        directory_descriptor, name, child, label="aggregate directory",
                    )
                    if _file_snapshot(os.fstat(child)) != child_snapshot:
                        raise ValueError(
                            f"aggregate directory changed while scanning: {name!r}"
                        )
                    append_source(_ScannedSource(
                        path=os.path.join(self.dataset.input_root, *relative),
                        snapshot=child_snapshot,
                        aggregate=aggregate,
                    ))
                except AggregateMemberEmptyError as exc:
                    self._verify_empty_aggregate_bindings(
                        directory_descriptor,
                        name,
                        child,
                        relative,
                        child_snapshot,
                        directory_snapshot,
                        root_descriptor,
                    )
                    if exc.aggregate_relative_path is None:
                        exc.aggregate_relative_path = "/".join(relative)
                    primary_error = None
                    logger.warning(
                        "excluding empty streaming FLAC aggregate: "
                        "dataset_id=%s source_relative_path=%s member=%s reason=%s",
                        self.dataset.dataset_id,
                        exc.aggregate_relative_path,
                        exc.member_name,
                        str(exc),
                    )
                except BaseException as exc:
                    primary_error = exc
                    raise
                finally:
                    _close_descriptors(
                        (child,),
                        primary_error=primary_error,
                        label="scan aggregate directory descriptor",
                    )
                continue
            if not self.recursive:
                continue
            flags = (
                os.O_RDONLY
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                child = os.open(name, flags, dir_fd=directory_descriptor)
            except OSError as exc:
                raise ValueError(
                    f"source directory component changed while scanning: {name!r}"
                ) from exc
            primary_error: BaseException | None = None
            try:
                named = _require_named_inode(
                    directory_descriptor,
                    name,
                    child,
                    label="source directory component",
                )
                child_snapshot = _file_snapshot(os.fstat(child))
                if (
                    _file_snapshot(named) != child_snapshot
                    or child_snapshot != _file_snapshot(entry_snapshot)
                ):
                    raise ValueError(
                        f"source directory component changed while scanning: {name!r}"
                    )
                self._scan_directory_descriptor(
                    child,
                    relative,
                    append_source,
                    root_descriptor=root_descriptor,
                )
                named = _require_named_inode(
                    directory_descriptor,
                    name,
                    child,
                    label="source directory component",
                )
                if (
                    _file_snapshot(named) != child_snapshot
                    or _file_snapshot(os.fstat(child)) != child_snapshot
                ):
                    raise ValueError(
                        f"source directory component changed while scanning: {name!r}"
                    )
                if _file_snapshot(os.fstat(directory_descriptor)) != directory_snapshot:
                    raise ValueError("source directory changed while scanning")
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                _close_descriptors(
                    (child,),
                    primary_error=primary_error,
                    label="scan directory descriptor",
                )

        if _file_snapshot(os.fstat(directory_descriptor)) != directory_snapshot:
            raise ValueError("source directory changed while scanning")

    def _scan_sources(
        self, root_descriptor: int, single_file: bool,
    ) -> Iterator[_ScannedSource]:
        """磁盘排序并流式产生 descriptor 扫描的 source 快照。"""
        self._verify_root_descriptor(root_descriptor, single_file)
        excluded_paths = frozenset(
            self.dataset.source_selector.exclude_relative_paths
        )

        def append_selected(source: _ScannedSource) -> None:
            if source.aggregate is not None:
                relative_path = source.aggregate.source_relative_path
            else:
                relative_path = os.path.relpath(
                    source.path, self.dataset.input_root,
                ).replace(os.sep, "/")
            if relative_path in excluded_paths:
                logger.info(
                    "Step0 dataset=%s excluded source_relative_path=%s",
                    self.dataset.dataset_id,
                    relative_path,
                )
                return
            spool.append(source)

        if single_file:
            source = _ScannedSource(
                path=self.dataset.input_root,
                snapshot=_file_snapshot(os.fstat(root_descriptor)),
            )
            relative_path = self.dataset.input_root.replace(os.sep, "/").lstrip("/")
            if relative_path in excluded_paths:
                logger.info(
                    "Step0 dataset=%s excluded source_relative_path=%s",
                    self.dataset.dataset_id,
                    relative_path,
                )
            else:
                yield source
            return

        spool = _SourceSpool()
        primary_error: BaseException | None = None
        try:
            parent_path = os.path.dirname(self.dataset.input_root)
            root_name = os.path.basename(self.dataset.input_root)
            parent_flags = (
                os.O_RDONLY
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_DIRECTORY", 0)
            )
            parent_descriptor = os.open(parent_path, parent_flags)
            scan_error: BaseException | None = None
            try:
                parent_snapshot = _file_snapshot(os.fstat(parent_descriptor))
                _require_named_inode(
                    parent_descriptor, root_name, root_descriptor, label="input_root",
                )
                self._scan_directory_descriptor(
                    root_descriptor,
                    (),
                    append_selected,
                    root_descriptor=root_descriptor,
                )
                if _file_snapshot(os.fstat(parent_descriptor)) != parent_snapshot:
                    raise ValueError("input_root parent changed while scanning")
                _require_named_inode(
                    parent_descriptor, root_name, root_descriptor, label="input_root",
                )
                self._verify_root_descriptor(root_descriptor, single_file)
            except BaseException as exc:
                scan_error = exc
                raise
            finally:
                _close_descriptors(
                    (parent_descriptor,),
                    primary_error=scan_error,
                    label="input_root parent descriptor",
                )
            yield from spool.iter_sources()
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            spool.close(primary_error)

    def _source_path_components(
        self, source_path: str, source_relative_path: str,
    ) -> tuple[str, ...]:
        normalized = validate_source_relative_path(source_relative_path)
        lexical_relative = os.path.relpath(
            os.path.abspath(source_path), self.dataset.input_root,
        ).replace(os.sep, "/")
        if lexical_relative != normalized:
            raise ValueError("source_relative_path does not match enumerated source path")
        return tuple(normalized.split("/"))

    def _open_directory_source_descriptor(
        self,
        root_descriptor: int,
        source_path: str,
        components: tuple[str, ...],
        scanned_snapshot: tuple[int, int, int, int, int, int, int],
        aggregate: AggregateDescriptor | None = None,
    ) -> _HeldSource:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_DIRECTORY", 0)
        )
        source_flags = (
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        if aggregate is not None:
            source_flags |= getattr(os, "O_DIRECTORY", 0)
        directories = [os.dup(root_descriptor)]
        descriptor: int | None = None
        try:
            for component in components[:-1]:
                try:
                    child = os.open(component, directory_flags, dir_fd=directories[-1])
                except OSError as exc:
                    raise ValueError(
                        f"source directory component is not a real directory: {component!r}"
                    ) from exc
                try:
                    _require_named_inode(
                        directories[-1], component, child,
                        label="source directory component",
                    )
                except BaseException as exc:
                    _close_descriptors(
                        (child,), primary_error=exc, label="source directory component",
                    )
                    raise
                directories.append(child)
            for parent, name, child in zip(
                directories, components[:-1], directories[1:],
            ):
                _require_named_inode(
                    parent, name, child, label="source directory component",
                )
            try:
                descriptor = os.open(
                    components[-1], source_flags, dir_fd=directories[-1],
                )
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise ValueError(
                        f"source symlink changed or is unsupported: {source_path!r}"
                    ) from exc
                raise ValueError(f"cannot open source {source_path!r}: {exc}") from exc
            opened = os.fstat(descriptor)
            expected_directory = aggregate is not None
            if expected_directory and not stat.S_ISDIR(opened.st_mode):
                raise ValueError(f"aggregate source must be a directory: {source_path!r}")
            if not expected_directory and not stat.S_ISREG(opened.st_mode):
                raise ValueError(f"source must be a regular file: {source_path!r}")
            if _file_snapshot(opened) != scanned_snapshot:
                raise ValueError(f"source changed after descriptor scan: {source_path!r}")
            _require_named_inode(
                directories[-1], components[-1], descriptor, label="source",
            )
            original_path = _descriptor_path(descriptor, "source")
            if not _contained(original_path, self.dataset.input_root, single_file=False):
                raise ValueError(f"source descriptor escapes frozen input_root: {source_path!r}")
            held = _HeldSource(
                directory_descriptors=directories,
                directory_names=components[:-1],
                source_name=components[-1],
                descriptor=descriptor,
                snapshot=_file_snapshot(opened),
                original_path=original_path,
                aggregate=aggregate,
            )
            held.verify_chain(verify_aggregate=aggregate is None)
            return held
        except BaseException as exc:
            descriptors = directories
            if descriptor is not None:
                descriptors = [descriptor, *descriptors]
            _close_descriptors(
                descriptors, primary_error=exc, label="held source descriptor",
            )
            raise

    def _open_source_descriptor(
        self,
        root_descriptor: int,
        source_path: str,
        source_relative_path: str,
        single_file: bool,
        scanned_snapshot: tuple[int, int, int, int, int, int, int],
        aggregate: AggregateDescriptor | None = None,
    ) -> _HeldSource:
        if not single_file:
            components = self._source_path_components(source_path, source_relative_path)
            return self._open_directory_source_descriptor(
                root_descriptor,
                source_path,
                components,
                scanned_snapshot,
                aggregate,
            )
        if aggregate is not None or self.dataset.source_layout == AGGREGATE_SOURCE_LAYOUT:
            raise ValueError("streaming FLAC aggregate source must use a directory input_root")
        if os.path.abspath(source_path) != self.dataset.input_root:
            raise ValueError("single-file source does not match frozen input_root")
        descriptor = os.dup(root_descriptor)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError(f"source must be a regular file: {source_path!r}")
            if _inode_identity(opened) != self._root_identity:
                raise ValueError("single-file source identity changed")
            if _file_snapshot(opened) != scanned_snapshot:
                raise ValueError("single-file source changed after descriptor scan")
            original_path = _descriptor_path(descriptor, "source")
            return _HeldSource(
                directory_descriptors=[],
                directory_names=(),
                source_name="",
                descriptor=descriptor,
                snapshot=_file_snapshot(opened),
                original_path=original_path,
            )
        except BaseException as exc:
            _close_descriptors(
                (descriptor,), primary_error=exc, label="held source descriptor",
            )
            raise

    def _verify_source_descriptor(
        self,
        descriptor: int,
        source_path: str,
        snapshot: tuple[int, int, int, int, int, int, int],
        original_path: str,
        root_descriptor: int,
        single_file: bool,
        held_source: _HeldSource | None = None,
        verify_aggregate: bool = True,
        aggregate_full_scan: bool = True,
    ) -> None:
        if _file_snapshot(os.fstat(descriptor)) != snapshot:
            raise ValueError(f"source descriptor changed while reading: {source_path!r}")
        if _descriptor_path(descriptor, "source") != original_path:
            raise ValueError(f"source descriptor pathname changed: {source_path!r}")
        if held_source is not None and not single_file:
            held_source.verify_chain(
                verify_aggregate=verify_aggregate,
                aggregate_full_scan=aggregate_full_scan,
            )
        else:
            try:
                named = os.stat(source_path, follow_symlinks=False)
            except OSError as exc:
                raise ValueError(f"source pathname binding changed: {source_path!r}") from exc
            if not stat.S_ISREG(named.st_mode) or _file_snapshot(named) != snapshot:
                raise ValueError(f"source pathname binding changed: {source_path!r}")
        if not _contained(
            original_path, self.dataset.input_root, single_file=single_file,
        ):
            raise ValueError(f"source descriptor escapes frozen input_root: {source_path!r}")
        self._verify_root_descriptor(root_descriptor, single_file)

    def _row_for_source(
        self,
        source_path: str,
        root_descriptor: int,
        single_file: bool,
        min_source_sample_rate_hz: Optional[int] = None,
        scanned_snapshot: tuple[int, int, int, int, int, int, int] | None = None,
        aggregate: AggregateDescriptor | None = None,
    ) -> Optional[_ProcessedSource]:
        dataset = self.dataset
        selector = dataset.source_selector
        selected_speakers = frozenset(selector.include_speaker_ids)
        selected_sample_rates = frozenset(selector.sample_rates_hz)
        identity = make_source_identity(
            dataset.input_root,
            source_path,
            dataset.dataset_id,
            dataset.language,
        )
        if scanned_snapshot is None:
            raise ValueError("source is missing descriptor-scan snapshot")
        held_source = self._open_source_descriptor(
            root_descriptor,
            source_path,
            identity.source_relative_path,
            single_file,
            scanned_snapshot,
            aggregate,
        )
        descriptor = held_source.descriptor
        snapshot = held_source.snapshot
        original_path = held_source.original_path
        aggregate = held_source.aggregate
        primary_error: BaseException | None = None
        try:
            if aggregate is None:
                source_size_bytes = snapshot[3]
                if type(source_size_bytes) is not int or source_size_bytes < 0:
                    raise ValueError("source size must be a non-negative integer")
            else:
                source_size_bytes = aggregate.source_size_bytes

            metadata = None
            speaker_id = None
            rate_gate_enabled = bool(
                selected_sample_rates or min_source_sample_rate_hz is not None
            )
            speaker_only_fast_path = bool(selected_speakers and not rate_gate_enabled)
            if speaker_only_fast_path:
                speaker_id = self._resolver.resolve(identity.source_relative_path)
                if speaker_id not in selected_speakers:
                    self._verify_source_descriptor(
                        descriptor, source_path, snapshot, original_path,
                        root_descriptor, single_file, held_source,
                        verify_aggregate=True,
                        aggregate_full_scan=False,
                    )
                    return None

            if aggregate is None:
                metadata = _read_audio_header(descriptor)
                sample_rate = _positive_header_integer(
                    metadata.samplerate, "sample rate",
                )
                frames = _positive_header_integer(metadata.frames, "frames")
                channels = _positive_header_integer(metadata.channels, "channels")
                audio_format = _header_string(metadata.format, "format")
                subtype = _header_string(metadata.subtype, "subtype")
            else:
                sample_rate = aggregate.sample_rate
                frames = aggregate.frames
                channels = aggregate.channels
                audio_format = aggregate.format
                subtype = aggregate.subtype
            duration_sec = _duration_sec(frames, sample_rate)
            if not isinstance(duration_sec, (int, float)) or isinstance(duration_sec, bool):
                raise ValueError("audio duration must be a finite positive number")
            duration_sec = float(duration_sec)
            if not math.isfinite(duration_sec) or duration_sec <= 0:
                raise ValueError("audio duration must be finite and positive")
            self._verify_source_descriptor(
                descriptor, source_path, snapshot, original_path,
                root_descriptor, single_file, held_source,
                verify_aggregate=False,
            )
            if (
                min_source_sample_rate_hz is not None
                and sample_rate < min_source_sample_rate_hz
            ):
                self._verify_source_descriptor(
                    descriptor, source_path, snapshot, original_path,
                    root_descriptor, single_file, held_source,
                    verify_aggregate=True,
                    aggregate_full_scan=False,
                )
                return None
            if selected_sample_rates and sample_rate not in selected_sample_rates:
                self._verify_source_descriptor(
                    descriptor, source_path, snapshot, original_path,
                    root_descriptor, single_file, held_source,
                    verify_aggregate=True,
                    aggregate_full_scan=False,
                )
                return None

            if not speaker_only_fast_path:
                speaker_id = self._resolver.resolve(identity.source_relative_path)
                if selected_speakers and speaker_id not in selected_speakers:
                    self._verify_source_descriptor(
                        descriptor, source_path, snapshot, original_path,
                        root_descriptor, single_file, held_source,
                        verify_aggregate=True,
                        aggregate_full_scan=False,
                    )
                    return None

            source_sha256 = (
                _hash_source_descriptor(descriptor, source_size_bytes)
                if aggregate is None else aggregate.manifest_sha256
            )
            speaker_scope = dataset.dataset_id if speaker_id is not None else None
            self._verify_source_descriptor(
                descriptor,
                source_path,
                snapshot,
                original_path,
                root_descriptor,
                single_file,
                held_source,
                verify_aggregate=True,
                aggregate_full_scan=False,
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            held_source.close(primary_error)

        row = _step0_row(
            source_key=identity.source_key,
            dataset_id=dataset.dataset_id,
            language=dataset.language,
            category=dataset.category,
            quality_profile=dataset.quality_profile,
            input_root=dataset.input_root,
            original_path=original_path,
            original_relative_path=identity.original_relative_path,
            source_relative_path=identity.source_relative_path,
            source_size_bytes=source_size_bytes,
            source_sha256=source_sha256,
            sample_rate=sample_rate,
            frames=frames,
            channels=channels,
            duration_sec=duration_sec,
            format=audio_format,
            subtype=subtype,
            speaker_id=speaker_id,
            speaker_scope=speaker_scope,
        )
        return _ProcessedSource(row=row)

    def produce_rows(
        self,
        context: StageRunContext,
        upstream_index: Optional[ArtifactIndex],
        upstream_rows: Iterable[Mapping[str, Any]],
    ) -> Iterable[Mapping[str, Any]]:
        self._validate_context_dataset(context)
        if upstream_index is not None:
            raise ValueError("Step0 source stage must not receive upstream artifact")

        min_source_sample_rate_hz = self._min_source_sample_rate_hz(context.config)
        root_descriptor, single_file = self._open_root_descriptor()
        primary_error: BaseException | None = None
        try:
            source_keys: set[str] = set()
            matched_speaker_ids: set[str] = set()
            matched_sample_rates: set[int] = set()
            if self.workers <= 1:
                sources = self._scan_sources(root_descriptor, single_file)
                try:
                    for source in sources:
                        row = self._consume_processed_source(
                            self._row_for_source(
                                source.path,
                                root_descriptor,
                                single_file,
                                min_source_sample_rate_hz,
                                source.snapshot,
                                source.aggregate,
                            ),
                            source_keys,
                            matched_speaker_ids,
                            matched_sample_rates,
                        )
                        if row is not None:
                            yield row
                except BaseException as exc:
                    _close_iterator(
                        sources,
                        primary_error=exc,
                        label="Step0 source scan iterator",
                    )
                    raise
                else:
                    _close_iterator(sources, label="Step0 source scan iterator")
            else:
                yield from self._pooled_rows(
                    root_descriptor,
                    single_file,
                    source_keys,
                    matched_speaker_ids,
                    matched_sample_rates,
                    min_source_sample_rate_hz,
                )
            self._verify_root_descriptor(root_descriptor, single_file)
            self._validate_selector_matches(
                matched_speaker_ids, matched_sample_rates,
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            _close_descriptors(
                (root_descriptor,),
                primary_error=primary_error,
                label="root descriptor",
            )

    def _consume_processed_source(
        self,
        processed: Optional[_ProcessedSource],
        source_keys: set[str],
        matched_speaker_ids: set[str],
        matched_sample_rates: set[int],
    ) -> Optional[Mapping[str, Any]]:
        """在主线程仅从最终产出的 row 归并 selector 命中统计。"""
        if processed is None:
            return None
        row = processed.row
        selector = self.dataset.source_selector
        if selector.include_speaker_ids:
            matched_speaker_ids.add(row["speaker_id"])
        if selector.sample_rates_hz:
            matched_sample_rates.add(row["sample_rate"])
        return self._dedup_source_key(row, source_keys)

    def _validate_selector_matches(
        self,
        matched_speaker_ids: set[str],
        matched_sample_rates: set[int],
    ) -> None:
        selector = self.dataset.source_selector
        missing_speakers = set(selector.include_speaker_ids) - matched_speaker_ids
        if missing_speakers:
            raise ValueError(
                "source_selector include_speaker_ids matched no source for: "
                f"{sorted(missing_speakers)!r}"
            )
        missing_sample_rates = set(selector.sample_rates_hz) - matched_sample_rates
        if missing_sample_rates:
            raise ValueError(
                "source_selector sample_rates_hz matched no row for: "
                f"{sorted(missing_sample_rates)!r}"
            )

    def _dedup_source_key(
        self, row: Mapping[str, Any], source_keys: set[str],
    ) -> Mapping[str, Any]:
        source_key = row["source_key"]
        if source_key in source_keys:
            raise ValueError(f"duplicate source_key in Step0 scan: {source_key!r}")
        source_keys.add(source_key)
        return row

    def _row_for_submitted_source(
        self,
        source: _ScannedSource,
        root_ownership: _SubmittedRootDescriptor,
        single_file: bool,
        min_source_sample_rate_hz: Optional[int],
    ) -> Optional[_ProcessedSource]:
        """worker 原子接管独立 root fd，并在完成或异常时关闭。"""
        root_descriptor = root_ownership.take()
        primary_error: BaseException | None = None
        try:
            return self._row_for_source(
                source.path,
                root_descriptor,
                single_file,
                min_source_sample_rate_hz,
                source.snapshot,
                source.aggregate,
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                os.close(root_descriptor)
            except BaseException as exc:
                if primary_error is None:
                    raise
                _add_exception_note(
                    primary_error, f"worker root descriptor close failed: {exc!r}",
                )

    def _pooled_rows(
        self,
        root_descriptor: int,
        single_file: bool,
        source_keys: set[str],
        matched_speaker_ids: Optional[set[str]] = None,
        matched_sample_rates: Optional[set[int]] = None,
        min_source_sample_rate_hz: Optional[int] = None,
    ) -> Iterable[Mapping[str, Any]]:
        """线程池每任务持有独立 root fd，输出仍按扫描顺序排列。"""
        sources = iter(self._scan_sources(root_descriptor, single_file))
        pending: dict[
            int, tuple[Future[Optional[_ProcessedSource]], _SubmittedRootDescriptor]
        ] = {}
        matched_speaker_ids = (
            matched_speaker_ids if matched_speaker_ids is not None else set()
        )
        matched_sample_rates = (
            matched_sample_rates if matched_sample_rates is not None else set()
        )
        max_pending = self.workers * 2
        next_submit = 0
        next_yield = 0
        executor = ThreadPoolExecutor(max_workers=self.workers)
        completed_normally = False
        primary_error: BaseException | None = None
        try:
            exhausted = False
            while True:
                while not exhausted and len(pending) < max_pending:
                    try:
                        source = next(sources)
                    except StopIteration:
                        exhausted = True
                        break
                    duplicate = os.dup(root_descriptor)
                    try:
                        ownership = _SubmittedRootDescriptor(duplicate)
                    except BaseException as exc:
                        _close_descriptors(
                            (duplicate,),
                            primary_error=exc,
                            label="submitted root descriptor",
                        )
                        raise
                    try:
                        future = executor.submit(
                            self._row_for_submitted_source,
                            source,
                            ownership,
                            single_file,
                            min_source_sample_rate_hz,
                        )
                    except BaseException as exc:
                        ownership.close_if_owned(exc)
                        raise
                    pending[next_submit] = (future, ownership)
                    add_done_callback = getattr(future, "add_done_callback", None)
                    if callable(add_done_callback):
                        add_done_callback(
                            lambda completed, owned=ownership: (
                                owned.close_from_cancel_callback()
                                if completed.cancelled() else None
                            )
                        )
                    next_submit += 1
                if not pending:
                    completed_normally = True
                    return
                future, ownership = pending.pop(next_yield)
                try:
                    processed = future.result()
                except BaseException as exc:
                    ownership.close_if_owned(exc)
                    raise
                ownership.close_if_owned()
                next_yield += 1
                if isinstance(processed, Mapping):
                    # 兼容轻量 executor 测试替身的旧 row 返回形态。
                    processed = _ProcessedSource(row=dict(processed))
                row = self._consume_processed_source(
                    processed,
                    source_keys,
                    matched_speaker_ids,
                    matched_sample_rates,
                )
                if row is not None:
                    yield row
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            cleanup_error: BaseException | None = None
            if completed_normally:
                try:
                    executor.shutdown(wait=True, cancel_futures=True)
                except TypeError:
                    try:
                        executor.shutdown(wait=True)
                    except BaseException as exc:
                        cleanup_error = exc
                except BaseException as exc:
                    cleanup_error = exc
            else:
                _shutdown_executor_bounded(
                    executor,
                    pending_futures=(future for future, _ in pending.values()),
                    stage=f"step0-ingest:{self.dataset.dataset_id}",
                    workers=self.workers,
                    primary_error=primary_error,
                )
            cleanup_primary = primary_error or cleanup_error
            for _, ownership in pending.values():
                try:
                    ownership.close_if_owned(cleanup_primary)
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
                    cleanup_primary = primary_error or cleanup_error
            try:
                _close_iterator(
                    sources,
                    primary_error=primary_error or cleanup_error,
                    label="Step0 source scan iterator",
                )
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
            if cleanup_error is not None:
                if primary_error is not None:
                    _add_exception_note(
                        primary_error,
                        f"step0 thread executor cleanup failed: {cleanup_error!r}",
                    )
                else:
                    raise cleanup_error
