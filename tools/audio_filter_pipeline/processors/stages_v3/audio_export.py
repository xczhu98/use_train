"""Generation-3 Step4/Step6 artifact 成功后的显式音频导出器。"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import logging
import math
import os
import secrets
import stat
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - 非 POSIX 平台无 RLIMIT 语义
    resource = None

import numpy as np
import soundfile as sf
from tqdm import tqdm

from ...utils.artifacts import (
    ArtifactIndex,
    PipelineStep,
    is_production_bound_artifact_index,
    iter_artifact_rows,
)
from ...utils.export_roots import resolve_export_root
from ...utils.path_utils import (
    SafeOutputPath,
    open_safe_output_path,
    roots_overlap,
    validate_safe_output_path,
)
from ...utils.resample import resample_audio
from .audio_export_scheduler import (
    ExportIOCounters,
    VerifiedSourceSessionPool,
    source_session_identity,
)
from .mos_filter import STEP4_ROW_FIELDS
from .export import (
    EXPORT_LAYOUT_TEMPLATE,
    EXPORT_LAYOUT_VERSION,
    EXPORT_MANIFEST_FILENAME,
    EXPORT_MANIFEST_SCHEMA,
    EXPORT_ROUTING_VERSION,
    EXPORT_TRANSACTION_SCHEMA,
    STEP6_ROW_FIELDS,
    _FINAL_SELECTION_VERSION,
    _effective_bandwidth_hz,
    _finite_nonnegative_real,
    target_sample_rate_for_effective_bandwidth,
)
from .source_io import read_verified_source_interval


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportedAudio:
    """一个已发布 Step6 音频文件的不可变定位结果。"""

    segment_id: str
    target_sample_rate: int
    output_path: str
    output_relative_path: str


def _exact_positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive Python integer")
    return value


def _count_open_fds() -> int:
    """返回当前进程已占用的 fd 数；无 procfs 时退化为零。"""
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return 0


_EXPORT_FDS_PER_WORKER = 256
_EXPORT_FIXED_FD_RESERVE = 1024


def preflight_export_nofile(workers: int, *, max_pending: int) -> None:
    """导出前校验并提升 RLIMIT_NOFILE，覆盖安全路径与事务的真实峰值。

    一个 active worker 会同时持有 destination lock、多个
    ``SafeOutputPath`` 目录链、source session、写入/校验 descriptor 和
    transaction marker descriptor。目录链深度随输出根和源相对路径变化，
    因此不能再用只够覆盖浅路径的 ``workers * 64`` 估算。
    """
    _exact_positive_int(workers, "workers")
    _exact_positive_int(max_pending, "max_pending")
    if resource is None:
        return
    # 256/worker 是对当前导出协议中同时存活的多条安全目录链、源会话、
    # transaction 与验证 descriptor 的保守预算；固定余量覆盖主线程、manifest
    # lock、executor 和运行时在导出前已经持有的少量 descriptor。
    baseline = _count_open_fds()
    required = (
        baseline
        + workers * _EXPORT_FDS_PER_WORKER
        + max_pending
        + _EXPORT_FIXED_FD_RESERVE
    )
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        return
    if soft >= required:
        return
    if hard != resource.RLIM_INFINITY and hard < required:
        raise ValueError(
            f"RLIMIT_NOFILE hard limit {hard} is below the estimated export "
            f"peak of {required} file descriptors (workers={workers}, "
            f"max_pending={max_pending}); raise the hard limit first "
            "(e.g. ulimit -Hn 1048576) and retry"
        )
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"RLIMIT_NOFILE soft limit {soft} is below the estimated export "
            f"peak of {required} file descriptors and could not be raised "
            f"automatically: {exc}; run with a larger limit "
            "(e.g. ulimit -n 1048576) and retry"
        ) from exc
    logger.warning(
        "raised RLIMIT_NOFILE soft limit from %d to %d for export concurrency",
        soft, required,
    )


def _safe_relative_path(value: Any) -> PurePosixPath:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError("source_relative_path must be a nonblank exact Python string")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("source_relative_path must be a safe relative POSIX path")
    return path


def _safe_path_component(value: Any, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be a nonblank exact Python string")
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"{field} must be a safe single path component")
    return value


def _default_decoder(row: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    """以 Step0 descriptor binding 安全复验后读取精确区间。"""
    try:
        return read_verified_source_interval(row)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"source segment decode failed: {exc}") from exc


def _default_writer(path: str, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(path, audio, sample_rate, format="FLAC", subtype="PCM_24")


def _close_iterator(iterator: Any, label: str) -> None:
    close = getattr(iterator, "close", None)
    if not callable(close):
        return
    try:
        close()
    except BaseException:
        # 迭代器关闭失败不能覆盖导出主异常；保留日志供调用方排查。
        logger.warning("failed to close %s iterator", label, exc_info=True)


def _add_exception_note(error: BaseException, note: str) -> None:
    """尽力附加清理诊断，且绝不让诊断失败覆盖原异常。"""
    try:
        add_note = getattr(error, "add_note", None)
        if callable(add_note):
            add_note(note)
            return
        notes = list(getattr(error, "__notes__", ()))
        notes.append(note)
        error.__notes__ = notes
    except BaseException:
        # Python 3.10 没有 BaseException.add_note；异常对象也可能拒绝
        # 动态属性，清理诊断失败时只能记录日志，不能替换主异常。
        logger.warning("failed to attach exception note", exc_info=True)


_OUTPUT_FORMAT = "FLAC"
_OUTPUT_SUBTYPE = "PCM_24"
_MANIFEST_SCHEMA = EXPORT_MANIFEST_SCHEMA
_MANIFEST_FILENAME = EXPORT_MANIFEST_FILENAME
_MANIFEST_MAX_LINE_BYTES = 1024 * 1024
_MANIFEST_MAX_BYTES = 256 * 1024 * 1024
_MANIFEST_MAX_RECORDS = 1_000_000
_AUDIO_VERIFY_BLOCK_FRAMES = 65_536
_AUDIO_HASH_BLOCK_BYTES = 1024 * 1024
_AUDIO_VERIFY_MARGIN_BYTES = 2 * 1024 * 1024
# 小于该阈值的已发布音频一次性读入内存，decode 与 sha256 共用同一份字节
# 快照，磁盘只读一遍；超过阈值仍走流式 decode + 分块 hash 的两遍路径。
_AUDIO_VERIFY_BUFFERED_MAX_BYTES = 64 * 1024 * 1024
# 同一 aggregate source 被门控时允许的有界前瞻候选数，避免一个 source
# 占满整个 pending window 后阻塞后续 source；不会物化完整 rows。
_MAX_DEFERRED_CANDIDATES = 1024
_MAX_SINGLE_TASK_INFLIGHT_BYTES = 512 * 1024 * 1024
_MAX_DERIVED_FRAMES = _MAX_SINGLE_TASK_INFLIGHT_BYTES // np.dtype(np.float32).itemsize
_MAX_SAMPLE_RATE = (1 << 63) - 1
# 导出恢复 journal 自行演进；routing 的共享 schema 仍只描述生产 artifact 合约。
_TRANSACTION_SCHEMA = "audio_filter_pipeline.audio_export_transaction.v3"
_TRANSACTION_MAX_BYTES = 2 * _MANIFEST_MAX_LINE_BYTES
_TRANSACTION_PREFIX = ".audio-export-transaction-"
_TRANSACTION_STAGING_PREFIX = ".audio-export-staging-"
_TRANSACTION_LOCK_PREFIX = ".audio-export-lock-"
_STAGING_OWNER_SCHEMA = "audio_filter_pipeline.audio_export_staging_owner.v1"
_STAGING_OWNER_PREFIX = ".owner-"
_SOURCE_IDENTITY_FIELDS = (
    "source_key", "input_root", "original_path", "source_relative_path",
    "source_size_bytes", "source_sha256", "format", "subtype",
    "sample_rate", "frames", "channels", "start_sample", "end_sample",
    "target_sample_rate",
)


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
    )


def _snapshot_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _file_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def _require_single_link(value: os.stat_result, label: str) -> None:
    if value.st_nlink != 1:
        raise ValueError(f"{label} is hard-linked; refusing to mutate it")


def _hash_descriptor_region(descriptor: int, size: int) -> str:
    """在 held fd 上分块 hash 指定前缀，不把大文件读入内存。"""
    if size < 0:
        raise ValueError("descriptor region size must be nonnegative")
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(
            descriptor, min(_AUDIO_HASH_BLOCK_BYTES, size - offset), offset,
        )
        if not chunk:
            raise ValueError("descriptor ended before the requested region")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _descriptor_region_matches(
    descriptor: int, offset: int, expected: bytes,
) -> bool:
    """分块比较 held fd 区间，避免整体读取目标文件。"""
    if offset < 0:
        raise ValueError("descriptor region offset must be nonnegative")
    position = 0
    while position < len(expected):
        size = min(_AUDIO_HASH_BLOCK_BYTES, len(expected) - position)
        chunk = os.pread(descriptor, size, offset + position)
        if len(chunk) != size or chunk != expected[position:position + size]:
            return False
        position += size
    return True


def _close_resources(
    resources: Iterable[tuple[Any, str]], primary: BaseException | None,
) -> BaseException | None:
    """逐个关闭资源；主异常存在时只记录 cleanup 异常。"""
    cleanup_error: BaseException | None = None
    for resource, label in resources:
        if resource is None:
            continue
        try:
            closer = resource if callable(resource) else resource.close
            closer()
        except BaseException as exc:
            if primary is not None:
                _add_exception_note(primary, f"{label} close failed: {exc!r}")
            elif cleanup_error is None:
                cleanup_error = exc
            else:
                _add_exception_note(cleanup_error, f"{label} close failed: {exc!r}")
    return cleanup_error


def _unlock_and_close(
    descriptor: int, label: str, primary: BaseException | None,
) -> BaseException | None:
    """尽力解锁并关闭 descriptor；任一步失败都不能跳过下一步。"""
    return _close_resources(
        (
            (
                lambda fd=descriptor: fcntl.flock(fd, fcntl.LOCK_UN),
                f"{label} unlock",
            ),
            (
                lambda fd=descriptor: os.close(fd),
                f"{label} descriptor",
            ),
        ),
        primary,
    )


def _open_bound_target(
    safe: SafeOutputPath,
    expected_identity: tuple[int, int],
    flags: int,
    label: str,
    *,
    require_single_link: bool = True,
) -> int:
    descriptor = None
    try:
        descriptor = safe.open_target(flags)
        current = os.fstat(descriptor)
        if _snapshot_identity(current) != expected_identity:
            raise ValueError(f"{label} inode changed while opening")
        if require_single_link:
            _require_single_link(current, label)
        return descriptor
    except BaseException as exc:
        _close_resources(
            (
                (
                    (lambda fd=descriptor: os.close(fd))
                    if descriptor is not None else None,
                    f"{label} descriptor",
                ),
            ),
            exc,
        )
        raise


def _hash_audio_descriptor(descriptor: int, expected: os.stat_result) -> str:
    """在 held fd 上分块 hash，并拒绝 inode/长度/写入时间变化。"""
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(descriptor, _AUDIO_HASH_BLOCK_BYTES, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    current = os.fstat(descriptor)
    if (
        not _same_identity(expected, current)
        or current.st_size != expected.st_size
        or current.st_mtime_ns != expected.st_mtime_ns
        or current.st_ctime_ns != expected.st_ctime_ns
        or offset != expected.st_size
    ):
        raise ValueError("audio file changed while hashing")
    return digest.hexdigest()


def _decode_audio_stream(
    source: Any, expected: Mapping[str, Any], label: str,
) -> dict[str, Any]:
    """从可读对象分块解码并复验元数据/帧数/幅值，返回规范化 metadata。"""
    sound = None
    try:
        sound = sf.SoundFile(source, mode="r")
        metadata = {
            "format": sound.format,
            "subtype": sound.subtype,
            "sample_rate": int(sound.samplerate),
            "frames": int(sound.frames),
            "channels": int(sound.channels),
        }
        if metadata != dict(expected):
            raise ValueError(
                f"published audio metadata mismatch for {label}: "
                f"expected {dict(expected)!r}, got {metadata!r}"
            )
        frames = 0
        while True:
            block = sound.read(
                frames=_AUDIO_VERIFY_BLOCK_FRAMES,
                dtype="float32",
                always_2d=True,
            )
            if block.size == 0:
                break
            if block.ndim != 2 or block.shape[1] != expected["channels"]:
                raise ValueError(f"published audio shape mismatch for {label}")
            if not np.isfinite(block).all():
                raise ValueError(f"published audio contains non-finite samples: {label}")
            if float(np.max(np.abs(block))) > 1.0:
                raise ValueError(f"published audio amplitude exceeds [-1, 1]: {label}")
            frames += int(block.shape[0])
            if frames > expected["frames"]:
                raise ValueError(f"published audio has too many frames: {label}")
        if frames != expected["frames"]:
            raise ValueError(
                f"published audio frame count mismatch for {label}: "
                f"expected {expected['frames']}, got {frames}"
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"published audio cannot be read: {label}") from exc
    finally:
        primary = sys.exc_info()[1]
        cleanup_error = _close_resources(
            ((sound, f"{label} sound stream"),),
            primary,
        )
        if primary is None and cleanup_error is not None:
            raise cleanup_error
    return metadata


def _read_descriptor_snapshot(
    descriptor: int, expected: os.stat_result, label: str,
) -> bytes:
    """在 held fd 上分块读入有界小文件；读后复核 stat 拒绝任何并发变化。"""
    chunks: list[bytes] = []
    offset = 0
    while offset < expected.st_size:
        chunk = os.pread(
            descriptor,
            min(_AUDIO_HASH_BLOCK_BYTES, expected.st_size - offset),
            offset,
        )
        if not chunk:
            raise ValueError(f"published audio changed while being read: {label}")
        chunks.append(chunk)
        offset += len(chunk)
    current = os.fstat(descriptor)
    if (
        not _same_identity(expected, current)
        or current.st_size != expected.st_size
        or current.st_mtime_ns != expected.st_mtime_ns
        or current.st_ctime_ns != expected.st_ctime_ns
    ):
        raise ValueError(f"published audio changed while being read: {label}")
    return b"".join(chunks)


def _verify_audio_descriptor(
    descriptor: int,
    expected: Mapping[str, Any],
    label: str,
    *,
    counters: ExportIOCounters | None = None,
    phase: str = "unspecified",
) -> tuple[dict[str, Any], str]:
    """通过独立 SoundFile stream 分块验证 held regular-file fd。

    小文件走单遍快照路径：一次读入字节后 hash 与 decode 共用该快照，磁盘
    只读一遍；超过 ``_AUDIO_VERIFY_BUFFERED_MAX_BYTES`` 的文件保持流式
    decode + 分块 hash 的两遍路径，避免无界内存占用。
    """
    try:
        initial = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError(f"published audio cannot be inspected: {label}") from exc
    if not stat.S_ISREG(initial.st_mode):
        raise ValueError(f"published audio is not a regular file: {label}")

    if initial.st_size <= _AUDIO_VERIFY_BUFFERED_MAX_BYTES:
        payload = _read_descriptor_snapshot(descriptor, initial, label)
        digest = hashlib.sha256(payload).hexdigest()
        metadata = _decode_audio_stream(io.BytesIO(payload), expected, label)
        if counters is not None:
            counters.record_output_verify(phase, initial.st_size, read_passes=1)
        return metadata, digest

    stream = None
    try:
        duplicate = os.dup(descriptor)
        try:
            stream = os.fdopen(duplicate, "rb", closefd=True)
        except BaseException as exc:
            _close_resources(
                ((lambda fd=duplicate: os.close(fd), "audio verification stream fd"),),
                exc,
            )
            raise
        metadata = _decode_audio_stream(stream, expected, label)
    finally:
        primary = sys.exc_info()[1]
        cleanup_error = _close_resources(
            ((stream, f"{label} byte stream"),),
            primary,
        )
        if primary is None and cleanup_error is not None:
            raise cleanup_error

    current = os.fstat(descriptor)
    if (
        not _same_identity(initial, current)
        or current.st_size != initial.st_size
        or current.st_mtime_ns != initial.st_mtime_ns
        or current.st_ctime_ns != initial.st_ctime_ns
    ):
        raise ValueError(f"published audio changed while being verified: {label}")
    digest = _hash_audio_descriptor(descriptor, initial)
    if counters is not None:
        counters.record_output_verify(phase, initial.st_size)
    return metadata, digest


def _strict_json_object(pairs: list[tuple[Any, Any]]) -> dict[Any, Any]:
    value: dict[Any, Any] = {}
    for key, nested in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        value[key] = nested
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is forbidden: {value}")
    return parsed


def _json_tail_is_incomplete(raw: bytes) -> bool:
    """仅将 EOF 截断的 JSON 视为可恢复短写。"""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    try:
        json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_float,
        )
    except json.JSONDecodeError as exc:
        stripped = text.rstrip()
        if not stripped:
            return False
        if exc.msg.startswith("Unterminated string"):
            return True
        if exc.pos >= len(stripped):
            return True
        if (
            exc.msg == "Invalid \\escape"
            and exc.pos == len(stripped) - 1
            and stripped.endswith("\\")
        ):
            return True
        return False
    except (UnicodeDecodeError, ValueError):
        return False
    else:
        return False


def _canonical_json_line(value: Mapping[str, Any]) -> bytes:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"manifest record is not canonical JSON: {exc}") from exc
    if len(payload) > _MANIFEST_MAX_LINE_BYTES:
        raise ValueError("manifest record exceeds bounded line size")
    return payload


def _canonical_source_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    identity = {field: row[field] for field in _SOURCE_IDENTITY_FIELDS}
    identity["input_root"] = os.path.realpath(os.path.abspath(identity["input_root"]))
    identity["original_path"] = os.path.realpath(os.path.abspath(identity["original_path"]))
    return identity


def _producer_identity(index: ArtifactIndex) -> dict[str, Any]:
    return {
        "step": index.step,
        "dataset_id": index.dataset_id,
        "config_digest": index.config_digest,
        "lineage_digest": index.lineage_digest,
        "rows_sha256": index.rows_sha256,
        "artifact_digest": index.artifact_digest,
    }


def _json_safe_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    """在候选预取前冻结深层 JSON 值，拒绝非有限或非 JSON-safe 对象。"""
    try:
        encoded = json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
        value = json.loads(
            encoded,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_float,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"export row is not JSON-safe: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("export row snapshot must be a JSON object")
    return value


def _ceil_scaled_frames(source_frames: Any, target_rate: Any, source_rate: Any) -> int:
    """使用整数运算计算重采样后的帧数，并限制派生结果规模。"""
    if (
        type(source_frames) is not int or source_frames <= 0
        or type(target_rate) is not int or not 0 < target_rate <= _MAX_SAMPLE_RATE
        or type(source_rate) is not int or not 0 < source_rate <= _MAX_SAMPLE_RATE
    ):
        raise ValueError("audio frame scaling inputs must be bounded positive integers")
    # 先比较上界，避免对攻击性超大 Python int 执行无界乘法。
    max_source_frames = (_MAX_DERIVED_FRAMES * source_rate) // target_rate
    if source_frames > max_source_frames:
        raise ValueError("derived output frame count exceeds bounded limit")
    numerator = source_frames * target_rate
    frames = (numerator + source_rate - 1) // source_rate
    if frames <= 0 or frames > _MAX_DERIVED_FRAMES:
        raise ValueError("derived output frame count exceeds bounded limit")
    return frames


def _expected_output_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    source_frames = row["end_sample"] - row["start_sample"]
    return {
        "format": _OUTPUT_FORMAT,
        "subtype": _OUTPUT_SUBTYPE,
        "sample_rate": row["target_sample_rate"],
        "frames": _ceil_scaled_frames(
            source_frames, row["target_sample_rate"], row["sample_rate"],
        ),
        # 导出固定为仅第一通道（mono）；source 侧多通道信息仍由
        # ``_canonical_source_identity`` 在 manifest 中如实记录。
        "channels": 1,
    }


@dataclass(frozen=True)
class _ExportTaskResult:
    exported: ExportedAudio
    manifest_record: dict[str, Any]


@dataclass
class _ExportTransaction:
    safe: SafeOutputPath
    descriptor: int
    identity: tuple[int, int]
    signature: tuple[int, int, int, int, int]
    value: dict[str, Any]


class _ManifestState:
    """单个 producer artifact 与可选语种 namespace 的有界 JSONL 状态索引。"""

    def __init__(
        self,
        export_root: Path,
        step: str,
        dataset_id: str,
        producer: Mapping[str, Any],
        *,
        language: str | None = None,
    ) -> None:
        self._export_root = export_root
        self._step = step
        self._dataset_id = dataset_id
        self._language = (
            None if language is None else _safe_path_component(language, "language")
        )
        self._producer = dict(producer)
        manifest_root = export_root / step
        if self._language is not None:
            manifest_root /= self._language
        self._path = (
            manifest_root / dataset_id / self._producer["artifact_digest"]
            / _MANIFEST_FILENAME
        )
        self._lock_path = self._path.with_name(f"{self._path.name}.lock")
        self.records: dict[str, dict[str, Any]] = {}
        self._record_count = 0
        self._manifest_bytes = 0
        self._loaded = False
        self._signature: tuple[int, int, int, int, int] | None = None
        self._manifest_digest: Any | None = None
        # 成功获取 manifest lock 后保留 SafeOutputPath；parent_fd 是
        # 目录锁锚点，不能在返回 descriptor 前关闭或丢失。
        self._lock_resources: dict[
            int, tuple[Any, int | None, tuple[int, int] | None, tuple[int, int]]
        ] = {}
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    def _reset_cache(self) -> None:
        self.records = {}
        self._record_count = 0
        self._manifest_bytes = 0
        self._loaded = False
        self._signature = None
        self._manifest_digest = None

    def _manifest_signature(self) -> tuple[int, int, int, int, int] | None:
        try:
            snapshot = os.stat(self._path, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(snapshot.st_mode) or not stat.S_ISREG(snapshot.st_mode):
            raise ValueError(f"manifest path is not a regular file: {self._path}")
        _require_single_link(snapshot, f"manifest path {self._path}")
        return (
            snapshot.st_dev, snapshot.st_ino, snapshot.st_size,
            snapshot.st_mtime_ns, snapshot.st_ctime_ns,
        )

    def _exclusive_file_lock(self):
        descriptor = None
        safe = None
        anchor_fd = None
        anchor_identity = None
        anchor_locked = False
        file_identity = None
        try:
            safe = open_safe_output_path(
                str(self._export_root), str(self._lock_path), create=True,
            )
            # SafeOutputPath 在 FakeSafe 测试替身中可能没有 parent_fd；真实
            # 实现则用独立 dup 作为稳定的父目录锁锚点，避免依赖可替换 pathname。
            try:
                parent_fd = safe.parent_fd
            except AttributeError:
                parent_fd = None
            if parent_fd is not None:
                parent_snapshot = os.fstat(parent_fd)
                if not stat.S_ISDIR(parent_snapshot.st_mode):
                    raise ValueError(
                        f"manifest lock parent is not a directory: {self._lock_path}"
                    )
                anchor_fd = os.dup(parent_fd)
                anchor_snapshot = os.fstat(anchor_fd)
                if (
                    not stat.S_ISDIR(anchor_snapshot.st_mode)
                    or (anchor_snapshot.st_dev, anchor_snapshot.st_ino)
                    != (parent_snapshot.st_dev, parent_snapshot.st_ino)
                ):
                    raise ValueError(f"manifest lock parent changed while opening: {self._lock_path}")
                anchor_identity = (anchor_snapshot.st_dev, anchor_snapshot.st_ino)
                try:
                    # 同一进程内多个 exporter 也会共享该目录；这里阻塞等待，
                    # 不能把正常的 manifest append 串行化误报为冲突。
                    fcntl.flock(anchor_fd, fcntl.LOCK_EX)
                except OSError as exc:
                    raise ValueError("manifest lock parent cannot be locked safely") from exc
                anchor_locked = True

            named_before = safe.target_stat()
            descriptor = safe.open_target(os.O_RDWR | os.O_CREAT, 0o600)
            snapshot = os.fstat(descriptor)
            if not stat.S_ISREG(snapshot.st_mode):
                raise ValueError(f"manifest lock is not a regular file: {self._lock_path}")
            _require_single_link(snapshot, f"manifest lock {self._lock_path}")
            file_identity = _snapshot_identity(snapshot)
            if named_before is not None:
                _require_single_link(named_before, f"manifest lock {self._lock_path}")
                if _file_signature(snapshot) != _file_signature(named_before):
                    raise ValueError(f"manifest lock changed while opening: {self._lock_path}")
            named_after = safe.target_stat()
            if named_after is None or _file_signature(named_after) != _file_signature(snapshot):
                raise ValueError(f"manifest lock changed while opening: {self._lock_path}")
            _require_single_link(named_after, f"manifest lock {self._lock_path}")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            # flock 成功后只复验 held descriptor；不能再次通过 lock pathname
            # 查询，因为 pathname 可能已被替换为另一个 inode。
            locked_snapshot = os.fstat(descriptor)
            if (
                not stat.S_ISREG(locked_snapshot.st_mode)
                or _snapshot_identity(locked_snapshot) != file_identity
            ):
                raise ValueError(f"manifest lock changed while acquiring: {self._lock_path}")
            _require_single_link(locked_snapshot, f"manifest lock {self._lock_path}")
            self._lock_resources[descriptor] = (
                safe, anchor_fd, anchor_identity, file_identity,
            )
            safe = None
            anchor_fd = None
            return descriptor
        except BaseException as exc:
            if descriptor is not None:
                _unlock_and_close(descriptor, "manifest lock", exc)
            if anchor_fd is not None:
                if anchor_locked:
                    _unlock_and_close(anchor_fd, "manifest lock parent", exc)
                else:
                    _close_resources(
                        ((lambda fd=anchor_fd: os.close(fd), "manifest lock parent descriptor"),),
                        exc,
                    )
            if safe is not None:
                _close_resources(((safe, "manifest lock safe path"),), exc)
            raise

    def _release_exclusive_file_lock(
        self, descriptor: int, primary: BaseException | None,
    ) -> BaseException | None:
        resource = self._lock_resources.pop(descriptor, None)
        if resource is None:
            return _unlock_and_close(descriptor, "manifest lock", primary)

        safe, anchor_fd, anchor_identity, file_identity = resource
        cleanup_error: BaseException | None = None

        try:
            current = os.fstat(descriptor)
            if (
                not stat.S_ISREG(current.st_mode)
                or _snapshot_identity(current) != file_identity
            ):
                raise ValueError("manifest lock descriptor identity changed before release")
        except BaseException as exc:
            cleanup_error = exc
            if primary is not None:
                _add_exception_note(primary, f"manifest lock validation failed: {exc!r}")

        file_cleanup = _unlock_and_close(descriptor, "manifest lock", primary)
        if primary is None and file_cleanup is not None:
            cleanup_error = cleanup_error or file_cleanup

        if anchor_fd is not None:
            try:
                current_anchor = os.fstat(anchor_fd)
                if (
                    not stat.S_ISDIR(current_anchor.st_mode)
                    or anchor_identity is None
                    or (current_anchor.st_dev, current_anchor.st_ino) != anchor_identity
                ):
                    raise ValueError("manifest lock parent identity changed before release")
            except BaseException as exc:
                if primary is not None:
                    _add_exception_note(primary, f"manifest lock parent validation failed: {exc!r}")
                elif cleanup_error is None:
                    cleanup_error = exc
            anchor_cleanup = _unlock_and_close(anchor_fd, "manifest lock parent", primary)
            if primary is None and anchor_cleanup is not None:
                cleanup_error = cleanup_error or anchor_cleanup

        safe_cleanup = _close_resources(
            ((safe, "manifest lock safe path"),), primary,
        )
        if primary is None and safe_cleanup is not None:
            cleanup_error = cleanup_error or safe_cleanup
        return cleanup_error

    def _load_unlocked(self) -> None:
        try:
            signature = self._manifest_signature()
        except BaseException:
            self._reset_cache()
            raise
        if (
            self._loaded
            and signature == self._signature
            and self._manifest_digest is not None
        ):
            return
        self._reset_cache()
        if signature is None:
            self._manifest_digest = hashlib.sha256()
            self._loaded = True
            return
        if signature[2] > _MANIFEST_MAX_BYTES:
            raise ValueError("export manifest exceeds bounded total size")

        safe = None
        descriptor = None
        stream_descriptor = None
        stream = None
        incomplete_offset: int | None = None
        valid_bytes = 0
        manifest_digest = hashlib.sha256()
        try:
            safe = open_safe_output_path(
                str(self._export_root), str(self._path), create=False,
            )
            bound_stat = safe.target_stat()
            if bound_stat is None:
                raise ValueError("export manifest disappeared while it was being opened")
            _require_single_link(bound_stat, f"manifest path {self._path}")
            descriptor = _open_bound_target(
                safe, _snapshot_identity(bound_stat), os.O_RDWR, str(self._path),
            )
            opened = os.fstat(descriptor)
            _require_single_link(opened, f"manifest path {self._path}")
            if _file_signature(opened) != signature:
                raise ValueError("export manifest changed while it was being opened")
            stream_descriptor = os.dup(descriptor)
            try:
                stream = os.fdopen(stream_descriptor, "rb", closefd=True)
            except BaseException as exc:
                _close_resources(
                    (
                        (
                            lambda fd=stream_descriptor: os.close(fd),
                            "manifest stream descriptor",
                        ),
                    ),
                    exc,
                )
                stream_descriptor = None
                raise
            stream_descriptor = None
            try:
                line_number = 0
                while True:
                    line = stream.readline(_MANIFEST_MAX_LINE_BYTES + 1)
                    if not line:
                        break
                    line_number += 1
                    if len(line) > _MANIFEST_MAX_LINE_BYTES:
                        raise ValueError("manifest line exceeds bounded size")
                    if not line.endswith(b"\n"):
                        # 完整 JSON 缺少终止 LF 不是短写，必须拒绝；解析不完整的
                        # EOF 尾部才沿用既有安全截断恢复。
                        extra = stream.read(1)
                        if extra:
                            raise ValueError("manifest line exceeds bounded size")
                        if not _json_tail_is_incomplete(line):
                            raise ValueError(
                                "manifest complete JSON tail is missing LF or is invalid"
                            )
                        incomplete_offset = valid_bytes
                        break
                    valid_bytes += len(line)
                    if valid_bytes > _MANIFEST_MAX_BYTES:
                        raise ValueError("export manifest exceeds bounded total size")
                    if self._record_count >= _MANIFEST_MAX_RECORDS:
                        raise ValueError("export manifest exceeds bounded record count")
                    try:
                        record = json.loads(
                            line[:-1].decode("utf-8"),
                            object_pairs_hook=_strict_json_object,
                            parse_constant=_reject_json_constant,
                            parse_float=_parse_finite_float,
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                        raise ValueError(
                            f"manifest line {line_number} is invalid strict JSON"
                        ) from exc
                    self._validate_record(record, line_number)
                    relative = record["output_relative_path"]
                    previous = self.records.get(relative)
                    if previous is not None and previous != record:
                        raise ValueError(f"manifest has conflicting records for {relative!r}")
                    self.records[relative] = record
                    self._record_count += 1
                    manifest_digest.update(line)
            finally:
                stream_cleanup_error = _close_resources(
                    ((stream, "manifest byte stream"),),
                    sys.exc_info()[1],
                )
                stream = None
                if stream_cleanup_error is not None:
                    raise stream_cleanup_error

            current = os.fstat(descriptor)
            current_named = safe.target_stat()
            if (
                current_named is None
                or _file_signature(current) != signature
                or _file_signature(current_named) != signature
            ):
                raise ValueError("export manifest changed while it was being read")
            _require_single_link(current, f"manifest path {self._path}")
            _require_single_link(current_named, f"manifest path {self._path}")
            if incomplete_offset is not None:
                if incomplete_offset >= current.st_size:
                    raise ValueError("manifest incomplete tail offset is invalid")
                _require_single_link(current, f"manifest path {self._path}")
                _require_single_link(current_named, f"manifest path {self._path}")
                os.ftruncate(descriptor, incomplete_offset)
                os.fsync(descriptor)
                safe.fsync_parent()
                current = os.fstat(descriptor)
                current_named = safe.target_stat()
                if current_named is not None:
                    _require_single_link(current_named, f"manifest path {self._path}")
                if (
                    current.st_size != incomplete_offset
                    or current_named is None
                    or _file_signature(current_named) != _file_signature(current)
                ):
                    raise ValueError("manifest incomplete tail truncation failed")
            if valid_bytes != current.st_size:
                raise ValueError("export manifest digest coverage is inconsistent")
            self._manifest_bytes = current.st_size
            self._signature = _file_signature(current)
            self._manifest_digest = manifest_digest
            self._loaded = True
        except OSError as exc:
            raise ValueError(f"cannot read export manifest safely: {self._path}: {exc}") from exc
        finally:
            primary = sys.exc_info()[1]
            if primary is not None:
                self._reset_cache()
            cleanup_error = _close_resources(
                (
                    (
                        (lambda fd=descriptor: os.close(fd))
                        if descriptor is not None else None,
                        f"manifest descriptor {self._path}",
                    ),
                    (safe, f"manifest safe path {self._path}"),
                ),
                primary,
            )
            if cleanup_error is not None:
                self._reset_cache()
            if primary is None and cleanup_error is not None:
                raise cleanup_error

    def load(self) -> None:
        with self._lock:
            # invocation 内缓存未变化时不重复获取跨进程 flock；外部替换、追加或
            # truncate 会改变签名并仍走锁内 fail-closed 重载。
            try:
                signature = self._manifest_signature()
            except BaseException:
                self._reset_cache()
                raise
            if (
                self._loaded
                and signature == self._signature
                and self._manifest_digest is not None
            ):
                return
            try:
                descriptor = self._exclusive_file_lock()
            except BaseException:
                self._reset_cache()
                raise
            try:
                self._load_unlocked()
            finally:
                primary = sys.exc_info()[1]
                cleanup_error = self._release_exclusive_file_lock(descriptor, primary)
                if cleanup_error is not None:
                    self._reset_cache()
                if primary is None and cleanup_error is not None:
                    raise cleanup_error

    def _validate_record(self, record: Any, line_number: int) -> None:
        if not isinstance(record, Mapping):
            raise ValueError(f"manifest line {line_number} must be an object")
        expected_fields = {
            "schema", "status", "producer", "step", "dataset_id", "segment_id",
            "output_relative_path", "source", "audio", "output_sha256",
        }
        if set(record) != expected_fields:
            raise ValueError(f"manifest line {line_number} has invalid fields")
        if record.get("schema") != _MANIFEST_SCHEMA or record.get("status") != "published":
            raise ValueError(f"manifest line {line_number} has unsupported schema or status")
        if record.get("step") != self._step or record.get("dataset_id") != self._dataset_id:
            raise ValueError(f"manifest line {line_number} has wrong step or dataset")
        if record.get("producer") != self._producer:
            raise ValueError(f"manifest line {line_number} has wrong producer identity")
        relative = record.get("output_relative_path")
        if type(relative) is not str or not relative.strip() or "\x00" in relative:
            raise ValueError(f"manifest line {line_number} has invalid output path")
        segment_id = record.get("segment_id")
        if (
            type(segment_id) is not str or not segment_id.strip()
            or "\x00" in segment_id or "/" in segment_id or "\\" in segment_id
        ):
            raise ValueError(f"manifest line {line_number} has invalid segment_id")
        path = PurePosixPath(relative)
        expected_prefix = PurePosixPath(self._step)
        if self._language is not None:
            expected_prefix /= self._language
        expected_prefix /= PurePosixPath(self._dataset_id) / self._producer["artifact_digest"]
        if (
            path.is_absolute()
            or len(path.parts) <= len(expected_prefix.parts)
            or any(part in ("", ".", "..") for part in path.parts)
            or path.parts[: len(expected_prefix.parts)] != expected_prefix.parts
        ):
            raise ValueError(f"manifest line {line_number} has unsafe output path")
        source = record.get("source")
        audio = record.get("audio")
        output_sha256 = record.get("output_sha256")
        if not isinstance(source, Mapping) or not isinstance(audio, Mapping):
            raise ValueError(f"manifest line {line_number} has incomplete metadata")
        if set(source) != set(_SOURCE_IDENTITY_FIELDS):
            raise ValueError(f"manifest line {line_number} has invalid source identity fields")
        for field in (
            "source_key", "input_root", "original_path", "source_relative_path",
            "format", "subtype",
        ):
            value = source[field]
            if type(value) is not str or not value.strip() or "\x00" in value:
                raise ValueError(
                    f"manifest line {line_number} has invalid source {field}"
                )
        for field in ("input_root", "original_path"):
            value = source[field]
            if not os.path.isabs(value) or os.path.realpath(os.path.abspath(value)) != value:
                raise ValueError(f"manifest line {line_number} has noncanonical source {field}")
        try:
            _safe_relative_path(source["source_relative_path"])
        except ValueError as exc:
            raise ValueError(
                f"manifest line {line_number} has unsafe source_relative_path"
            ) from exc
        if type(source["source_size_bytes"]) is not int or source["source_size_bytes"] < 0:
            raise ValueError(f"manifest line {line_number} has invalid source size")
        for field in ("sample_rate", "frames", "channels", "target_sample_rate"):
            if type(source[field]) is not int or source[field] <= 0:
                raise ValueError(f"manifest line {line_number} has invalid source {field}")
        if (
            type(source["start_sample"]) is not int
            or type(source["end_sample"]) is not int
            or not 0 <= source["start_sample"] < source["end_sample"] <= source["frames"]
        ):
            raise ValueError(f"manifest line {line_number} has invalid source interval")
        if (
            type(source.get("source_sha256")) is not str
            or len(source["source_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in source["source_sha256"])
        ):
            raise ValueError(f"manifest line {line_number} has invalid source SHA-256")
        if set(audio) != {"format", "subtype", "sample_rate", "frames", "channels"}:
            raise ValueError(f"manifest line {line_number} has invalid audio fields")
        if (
            type(output_sha256) is not str
            or len(output_sha256) != 64
            or any(char not in "0123456789abcdef" for char in output_sha256)
        ):
            raise ValueError(f"manifest line {line_number} has invalid output SHA-256")
        for field in ("sample_rate", "frames", "channels"):
            if type(audio.get(field)) is not int or audio[field] <= 0:
                raise ValueError(f"manifest line {line_number} has invalid audio {field}")
        if audio.get("format") != _OUTPUT_FORMAT or audio.get("subtype") != _OUTPUT_SUBTYPE:
            raise ValueError(f"manifest line {line_number} has unsupported audio format")

    def get(self, relative: str) -> dict[str, Any] | None:
        self.load()
        return self.get_cached(relative)

    def get_cached(self, relative: str) -> dict[str, Any] | None:
        """读取已初始化缓存；调用方负责先执行一次 ``load()``。"""
        with self._lock:
            if not self._loaded:
                raise RuntimeError("export manifest cache has not been loaded")
            return self.records.get(relative)

    def append(self, record: dict[str, Any]) -> None:
        self._validate_record(record, 0)
        payload = _canonical_json_line(record)
        with self._lock:
            try:
                lock_descriptor = self._exclusive_file_lock()
            except BaseException:
                self._reset_cache()
                raise
            cache_committed = False
            try:
                # 锁必须覆盖 reload 与 append，避免两个 exporter 同时基于旧索引
                # 通过相同的 destination 检查并写入冲突记录。
                self._load_unlocked()
                relative = record["output_relative_path"]
                previous = self.records.get(relative)
                if previous is not None:
                    if previous != record:
                        raise ValueError(f"manifest record conflicts for {relative!r}")
                    cache_committed = True
                    return
                if self._record_count >= _MANIFEST_MAX_RECORDS:
                    raise ValueError("export manifest exceeds bounded record count")
                if self._manifest_bytes + len(payload) > _MANIFEST_MAX_BYTES:
                    raise ValueError("export manifest exceeds bounded total size")

                expected_signature = self._signature
                safe = None
                descriptor = None
                original_size = 0
                original_prefix_hash = None
                final_signature = None
                try:
                    safe = open_safe_output_path(
                        str(self._export_root), str(self._path), create=True,
                    )
                    named_before = safe.target_stat()
                    if expected_signature is None:
                        if named_before is not None:
                            raise ValueError("export manifest changed before append")
                        open_flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_EXCL
                        expected_identity = None
                    else:
                        if (
                            named_before is None
                            or _file_signature(named_before) != expected_signature
                        ):
                            raise ValueError("export manifest changed before append")
                        _require_single_link(named_before, f"manifest path {self._path}")
                        open_flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
                        expected_identity = _snapshot_identity(named_before)
                    descriptor = safe.open_target(open_flags, 0o644)
                    snapshot = os.fstat(descriptor)
                    if not stat.S_ISREG(snapshot.st_mode):
                        raise ValueError(f"manifest path is not a regular file: {self._path}")
                    _require_single_link(snapshot, f"manifest path {self._path}")
                    if (
                        expected_identity is not None
                        and _snapshot_identity(snapshot) != expected_identity
                    ):
                        raise ValueError("export manifest inode changed while opening")
                    if (
                        expected_signature is not None
                        and _file_signature(snapshot) != expected_signature
                    ):
                        raise ValueError("export manifest changed while opening")
                    named_after = safe.target_stat()
                    if (
                        named_after is None
                        or _file_signature(named_after) != _file_signature(snapshot)
                    ):
                        raise ValueError("export manifest changed while opening")
                    _require_single_link(named_after, f"manifest path {self._path}")
                    original_size = snapshot.st_size
                    if not 0 <= original_size <= _MANIFEST_MAX_BYTES:
                        raise ValueError("export manifest original size is out of range")
                    if original_size + len(payload) > _MANIFEST_MAX_BYTES:
                        raise ValueError("export manifest exceeds bounded total size")
                    if (
                        not self._loaded
                        or self._signature != expected_signature
                        or self._manifest_bytes != original_size
                        or self._manifest_digest is None
                    ):
                        raise ValueError("export manifest digest cache is not synchronized")
                    original_prefix_digest = self._manifest_digest.copy()
                    original_prefix_hash = original_prefix_digest.hexdigest()
                    final_digest = original_prefix_digest.copy()
                    final_digest.update(payload)
                    offset = 0
                    try:
                        while offset < len(payload):
                            written = os.write(descriptor, payload[offset:])
                            if written <= 0:
                                raise OSError("manifest append made no progress")
                            offset += written
                        os.fsync(descriptor)
                        safe.fsync_parent()
                        final = os.fstat(descriptor)
                        _require_single_link(final, f"manifest path {self._path}")
                        final_named = safe.target_stat()
                        if final_named is not None:
                            _require_single_link(final_named, f"manifest path {self._path}")
                        if (
                            final_named is None
                            or _file_signature(final_named) != _file_signature(final)
                            or final.st_size != original_size + len(payload)
                        ):
                            raise ValueError("export manifest changed after append")
                        final_signature = _file_signature(final)
                    except BaseException as exc:
                        # 短写、fsync 失败或进程异常都不能留下半条 JSONL；截断前
                        # 必须确认原始前缀未变，且新增后缀仍只是本次 payload 的前缀。
                        try:
                            rollback_named = safe.target_stat()
                            current = os.fstat(descriptor)
                            _require_single_link(current, f"manifest path {self._path}")
                            if rollback_named is not None:
                                _require_single_link(rollback_named, f"manifest path {self._path}")
                            current_size = current.st_size
                            if (
                                rollback_named is None
                                or _snapshot_identity(current)
                                != _snapshot_identity(snapshot)
                                or _file_signature(rollback_named) != _file_signature(current)
                                or not 0 <= original_size <= current_size <= original_size + len(payload)
                                or original_prefix_hash is None
                                or _hash_descriptor_region(descriptor, original_size)
                                != original_prefix_hash
                                or not _descriptor_region_matches(
                                    descriptor, original_size,
                                    payload[:current_size - original_size],
                                )
                            ):
                                raise ValueError("export manifest changed during append rollback")
                            os.ftruncate(descriptor, original_size)
                            os.fsync(descriptor)
                            safe.fsync_parent()
                            rolled_back = os.fstat(descriptor)
                            _require_single_link(rolled_back, f"manifest path {self._path}")
                            rollback_named = safe.target_stat()
                            if rollback_named is not None:
                                _require_single_link(
                                    rollback_named, f"manifest path {self._path}",
                                )
                            if (
                                rollback_named is None
                                or _file_signature(rollback_named) != _file_signature(rolled_back)
                                or rolled_back.st_size != original_size
                                or _hash_descriptor_region(descriptor, original_size)
                                != original_prefix_hash
                            ):
                                raise ValueError("export manifest rollback verification failed")
                        except BaseException as rollback:
                            _add_exception_note(exc, f"manifest rollback failed: {rollback!r}")
                        raise
                except ValueError:
                    raise
                except OSError as exc:
                    raise ValueError(f"cannot atomically append export manifest: {exc}") from exc
                finally:
                    primary = sys.exc_info()[1]
                    cleanup_error = _close_resources(
                        (
                            (
                                (lambda fd=descriptor: os.close(fd))
                                if descriptor is not None else None,
                                f"manifest append descriptor {self._path}",
                            ),
                            (safe, f"manifest append safe path {self._path}"),
                        ),
                        primary,
                    )
                    if primary is None and cleanup_error is not None:
                        raise cleanup_error
                if final_signature is None:
                    raise ValueError("export manifest append has no verified final signature")
                self.records[relative] = record
                self._record_count += 1
                self._manifest_bytes = original_size + len(payload)
                self._signature = final_signature
                self._manifest_digest = final_digest
                self._loaded = True
                cache_committed = True
            finally:
                primary = sys.exc_info()[1]
                if not cache_committed:
                    self._reset_cache()
                cleanup_error = self._release_exclusive_file_lock(lock_descriptor, primary)
                if cleanup_error is not None:
                    self._reset_cache()
                if primary is None and cleanup_error is not None:
                    raise cleanup_error


class Step6AudioExporter:
    """显式消费已成功发布的 Step6 routing rows；ArtifactStage 不调用此类。"""

    _EXPORT_STEP = "step6"
    _ROW_FIELDS = frozenset(STEP6_ROW_FIELDS)
    _INCLUDE_LANGUAGE_IN_LAYOUT = True

    def __init__(
        self,
        output_root: str,
        *,
        export_root: str | None = None,
        decoder: Callable[[Mapping[str, Any]], tuple[np.ndarray, int]] = _default_decoder,
        resampler: Callable[..., np.ndarray] = resample_audio,
        writer: Callable[[str, np.ndarray, int], None] = _default_writer,
        workers: int = 32,
        max_pending: int = 64,
        max_inflight_bytes: int = 4 * 1024 * 1024 * 1024,
        output_sample_rate_hz: int | None = None,
    ) -> None:
        if type(output_root) is not str or not output_root.strip():
            raise ValueError("output_root must be a nonblank exact Python string")
        if export_root is not None and (type(export_root) is not str or not export_root.strip()):
            raise ValueError("export_root must be a nonblank exact Python string or None")
        if not callable(decoder) or not callable(resampler) or not callable(writer):
            raise TypeError("decoder, resampler, and writer must be callable")
        _exact_positive_int(workers, "workers")
        _exact_positive_int(max_pending, "max_pending")
        if max_pending < workers:
            raise ValueError("max_pending must be >= workers")
        _exact_positive_int(max_inflight_bytes, "max_inflight_bytes")
        if output_sample_rate_hz is not None and (
            type(output_sample_rate_hz) is not int
            or output_sample_rate_hz not in {16_000, 32_000}
        ):
            raise ValueError("output_sample_rate_hz must be 16000, 32000, or None")
        self._output_sample_rate_hz = output_sample_rate_hz
        # 配置层与 direct exporter 入口共享同一纯函数边界语义。
        self._export_root = resolve_export_root(output_root, export_root)
        self._decoder = decoder
        self._uses_default_decoder = decoder is _default_decoder
        self._resampler = resampler
        self._writer = writer
        self._workers = workers
        self._max_pending = max_pending
        self._max_inflight_bytes = max_inflight_bytes
        self._manifests: dict[tuple[str | None, str], _ManifestState] = {}
        self._producer: dict[str, Any] | None = None
        self._last_io_counts: dict[str, Any] = {}

    @property
    def last_io_counts(self) -> Mapping[str, Any]:
        """最近一次 export invocation 的只读计数快照。"""
        return dict(self._last_io_counts)

    def _manifest_for(
        self, dataset_id: str, language: str | None = None,
    ) -> _ManifestState:
        if self._producer is None:
            raise RuntimeError("producer artifact identity is not bound")
        if self._INCLUDE_LANGUAGE_IN_LAYOUT:
            if language is None:
                raise ValueError("Step6 manifest requires language")
            language = _safe_path_component(language, "language")
        else:
            language = None
        key = (language, dataset_id)
        manifest = self._manifests.get(key)
        if manifest is None:
            manifest = _ManifestState(
                self._export_root, self._EXPORT_STEP, dataset_id, self._producer,
                language=language,
            )
            self._manifests[key] = manifest
        return manifest

    def _validate_input_root_isolated(self, row: Mapping[str, Any]) -> None:
        # 无论导出根来自显式配置还是缺省 audio_exports 布局，都必须与每个
        # row 声明的输入根隔离；不能假设调用方一定经过 PipelineConfig 校验。
        if roots_overlap(str(row["input_root"]), str(self._export_root)):
            raise ValueError("input_root overlaps export_root")

    @staticmethod
    def _exported_from_row(row: Mapping[str, Any], destination: Path, relative: str) -> ExportedAudio:
        return ExportedAudio(
            segment_id=row["segment_id"],
            target_sample_rate=row["target_sample_rate"],
            output_path=str(destination),
            output_relative_path=relative,
        )

    def _resume_existing(
        self,
        row: Mapping[str, Any],
        destination: Path,
        relative: str,
        manifest: _ManifestState,
        counters: ExportIOCounters,
    ) -> ExportedAudio | None:
        record = manifest.get_cached(relative)
        try:
            safe = open_safe_output_path(
                str(self._export_root), str(destination), create=False,
            )
        except FileNotFoundError:
            if record is None:
                return None
            raise FileNotFoundError(
                f"manifest records published output but destination is missing: {destination}"
            ) from None
        descriptor = None
        try:
            target_stat = safe.target_stat()
            if record is None:
                if target_stat is not None:
                    raise FileExistsError(
                        f"refusing to overwrite existing destination: {destination}"
                    )
                return None
            if target_stat is None:
                raise FileNotFoundError(
                    f"manifest records published output but destination is missing: {destination}"
                )
            _require_single_link(target_stat, f"published destination {destination}")
            expected_source = _canonical_source_identity(row)
            expected_audio = _expected_output_metadata(row)
            if (
                record.get("segment_id") != row["segment_id"]
                or record.get("source") != expected_source
                or record.get("audio") != expected_audio
            ):
                raise ValueError(f"manifest/source metadata mismatch for {destination}")
            descriptor = _open_bound_target(
                safe, _snapshot_identity(target_stat), os.O_RDONLY, str(destination),
            )
            try:
                actual_audio, actual_sha256 = _verify_audio_descriptor(
                    descriptor,
                    expected_audio,
                    str(destination),
                    counters=counters,
                    phase="resume",
                )
            finally:
                primary = sys.exc_info()[1]
                cleanup_error = _close_resources(
                    (
                        (
                            lambda fd=descriptor: os.close(fd),
                            f"resume descriptor {destination}",
                        ),
                    ),
                    primary,
                )
                if primary is None and cleanup_error is not None:
                    raise cleanup_error
                descriptor = None
            if actual_audio != expected_audio or actual_sha256 != record.get("output_sha256"):
                raise ValueError(f"existing output hash or metadata mismatch for {destination}")
            return self._exported_from_row(row, destination, relative)
        finally:
            primary = sys.exc_info()[1]
            cleanup_error = _close_resources(
                ((safe, f"resume safe path {destination}"),),
                primary,
            )
            if primary is None and cleanup_error is not None:
                raise cleanup_error

    @staticmethod
    def _estimated_inflight_bytes(row: Mapping[str, Any]) -> int:
        """严格估算单任务 waveform/verification bytes；不代表总 RSS。"""
        source_frames = row["end_sample"] - row["start_sample"]
        target_frames = _ceil_scaled_frames(
            source_frames, row["target_sample_rate"], row["sample_rate"],
        )
        channels = row["channels"]
        if type(channels) is not int or channels <= 0:
            raise ValueError("audio channels must be a positive Python integer")
        itemsize = np.dtype(np.float32).itemsize
        max_samples = _MAX_SINGLE_TASK_INFLIGHT_BYTES // itemsize

        def buffer_bytes(frames: int) -> int:
            if frames > max_samples or channels > max_samples // frames:
                # 返回一个有界的超预算值，保留调用方现有的 hard-limit 错误契约。
                return _MAX_SINGLE_TASK_INFLIGHT_BYTES + 1
            return frames * channels * itemsize

        source_bytes = buffer_bytes(source_frames)
        resampled_bytes = buffer_bytes(target_frames)
        verify_block_frames = min(_AUDIO_VERIFY_BLOCK_FRAMES, target_frames)
        verify_block_bytes = buffer_bytes(verify_block_frames)
        buffered_snapshot_bytes = _AUDIO_VERIFY_BUFFERED_MAX_BYTES
        if (
            type(buffered_snapshot_bytes) is not int
            or buffered_snapshot_bytes < 0
        ):
            raise ValueError("audio verification snapshot budget must be non-negative")
        return (
            source_bytes
            + resampled_bytes
            + _AUDIO_VERIFY_MARGIN_BYTES
            + verify_block_bytes
            + buffered_snapshot_bytes
        )

    def _export_eligible(self, row: Mapping[str, Any]) -> bool:
        final_selection = row.get("final_selection")
        if not isinstance(final_selection, Mapping):
            raise ValueError("final_selection must be a mapping")
        if set(final_selection) != {"version", "export_eligible", "reason"}:
            raise ValueError("final_selection fields do not match the exact contract")
        if final_selection.get("version") != _FINAL_SELECTION_VERSION:
            raise ValueError(
                f"final_selection version must be exactly {_FINAL_SELECTION_VERSION}"
            )
        eligible = final_selection.get("export_eligible")
        if type(eligible) is not bool:
            raise ValueError("final_selection export_eligible must be bool")
        reason = final_selection.get("reason")
        if type(reason) is not str or not reason:
            raise ValueError("final_selection reason must be a nonblank exact Python string")
        return eligible

    def _validate_row(self, row: Mapping[str, Any]) -> None:
        actual_fields = set(row)
        if actual_fields != self._ROW_FIELDS:
            missing = sorted(self._ROW_FIELDS - actual_fields)
            extra = sorted(actual_fields - self._ROW_FIELDS)
            raise ValueError(
                f"{self._EXPORT_STEP} row fields do not match the exact contract: "
                f"missing={missing!r}, extra={extra!r}"
            )
        _safe_path_component(row["dataset_id"], "dataset_id")
        if self._INCLUDE_LANGUAGE_IN_LAYOUT:
            _safe_path_component(row["language"], "language")
        _safe_path_component(row["segment_id"], "segment_id")
        for field in ("source_key", "input_root", "original_path", "format", "subtype"):
            if type(row[field]) is not str or not row[field].strip():
                raise ValueError(f"{field} must be a nonblank exact Python string")
        _safe_relative_path(row["source_relative_path"])
        if type(row["source_size_bytes"]) is not int or row["source_size_bytes"] < 0:
            raise ValueError("source_size_bytes must be a nonnegative Python integer")
        if (type(row["source_sha256"]) is not str or len(row["source_sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in row["source_sha256"])):
            raise ValueError("source_sha256 must be lowercase SHA-256 hex")
        sample_rate = _exact_positive_int(row["sample_rate"], "sample_rate")
        frames = _exact_positive_int(row["frames"], "frames")
        _exact_positive_int(row["channels"], "channels")
        if type(row["source_sample_rate"]) is not int or row["source_sample_rate"] != sample_rate:
            raise ValueError("source_sample_rate must exactly match sample_rate")
        if type(row["segment_sample_rate"]) is not int or row["segment_sample_rate"] != sample_rate:
            raise ValueError("segment_sample_rate must exactly match sample_rate")
        bandwidth_hz = _finite_nonnegative_real(row["bandwidth_hz"], "bandwidth_hz")
        if bandwidth_hz > sample_rate / 2:
            raise ValueError("bandwidth_hz must be within source Nyquist bounds")
        estimated_sr = row["estimated_sr"]
        if bandwidth_hz == 0.0:
            if estimated_sr is not None:
                raise ValueError("estimated_sr must be None when bandwidth_hz is zero")
        elif type(estimated_sr) is not int or not 0 < estimated_sr <= sample_rate:
            raise ValueError(
                "estimated_sr must be a positive Python integer when bandwidth_hz is nonzero"
            )
        expected_effective_bandwidth_hz = _effective_bandwidth_hz(row)
        if row["effective_bandwidth_hz"] != expected_effective_bandwidth_hz:
            raise ValueError(
                "effective_bandwidth_hz must exactly match bandwidth analysis facts"
            )
        target_sample_rate = _exact_positive_int(
            row["target_sample_rate"], "target_sample_rate"
        )
        start = row["start_sample"]
        end = row["end_sample"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= frames:
            raise ValueError("segment interval must be nonempty, half-open, and within source frames")
        routing = row["export_routing"]
        if not isinstance(routing, Mapping):
            raise ValueError("export_routing must be a mapping")
        routing_fields = {
            "version", "layout_version", "layout_template", "manifest_schema",
            "manifest_filename", "transaction_schema", "mode", "audio_write",
            "minimum_target_sample_rate_hz", "source_below_minimum_rule",
            "upsampling_allowed", "source_path", "segment_interval_samples",
        }
        if "configured_target_sample_rate_hz" in routing:
            routing_fields.add("configured_target_sample_rate_hz")
        if set(routing) != routing_fields:
            raise ValueError("export_routing fields do not match the exact contract")
        if routing.get("version") != EXPORT_ROUTING_VERSION or routing.get("mode") != "metadata_only":
            raise ValueError(
                f"export_routing must be the metadata-only version {EXPORT_ROUTING_VERSION} contract"
            )
        expected_contract = {
            "layout_version": EXPORT_LAYOUT_VERSION,
            "layout_template": EXPORT_LAYOUT_TEMPLATE,
            "manifest_schema": EXPORT_MANIFEST_SCHEMA,
            "manifest_filename": EXPORT_MANIFEST_FILENAME,
            "transaction_schema": EXPORT_TRANSACTION_SCHEMA,
        }
        if any(routing.get(key) != value for key, value in expected_contract.items()):
            raise ValueError("export_routing exporter contract does not match current constants")
        if routing.get("audio_write") is not False:
            raise ValueError("export_routing audio_write must be exactly False")
        if routing.get("minimum_target_sample_rate_hz") != 16_000:
            raise ValueError("export_routing minimum target sample rate must be 16000 Hz")
        if routing.get("source_below_minimum_rule") != "native":
            raise ValueError("export_routing low-rate source rule must be native")
        if routing.get("upsampling_allowed") is not False:
            raise ValueError("export_routing upsampling_allowed must be exactly False")
        if routing.get("source_path") != row["original_path"]:
            raise ValueError("export_routing source_path must exactly match original_path")
        if routing.get("segment_interval_samples") != [start, end]:
            raise ValueError("export_routing interval must exactly match segment interval")
        configured_target = routing.get("configured_target_sample_rate_hz")
        if configured_target is not None:
            if (
                type(configured_target) is not int
                or configured_target not in {16_000, 32_000}
                or configured_target > sample_rate
            ):
                raise ValueError("configured target sample rate is invalid for source")
            if self._output_sample_rate_hz != configured_target:
                raise ValueError("exporter output_sample_rate_hz does not match routing")
            expected_target_sample_rate = configured_target
        else:
            if self._output_sample_rate_hz is not None:
                raise ValueError("exporter fixed sample rate is missing from routing")
            expected_target_sample_rate = target_sample_rate_for_effective_bandwidth(
                row.get("effective_bandwidth_hz"), sample_rate,
            )
        if target_sample_rate != expected_target_sample_rate:
            raise ValueError(
                f"target_sample_rate does not match export_routing version {EXPORT_ROUTING_VERSION} policy"
            )

    def _prepare_candidate(self, row: Mapping[str, Any]) -> dict[str, Any] | None:
        """单一 seam：深快照后先验证 exact schema，再执行 selection。"""
        snapshot = _json_safe_snapshot(row)
        self._validate_row(snapshot)
        if not self._export_eligible(snapshot):
            return None
        return snapshot

    def _destination(self, row: Mapping[str, Any]) -> tuple[Path, str]:
        if self._producer is None:
            raise RuntimeError("producer artifact identity is not bound")
        relative = _safe_relative_path(row["source_relative_path"])
        source_name = relative.name
        source_suffix = Path(source_name).suffix
        stem = source_name[:-len(source_suffix)] if source_suffix else source_name
        filename = f"{stem}--{row['segment_id']}.flac"
        destination_relative = PurePosixPath(self._EXPORT_STEP)
        if self._INCLUDE_LANGUAGE_IN_LAYOUT:
            destination_relative /= _safe_path_component(row["language"], "language")
        destination_relative /= (
            PurePosixPath(row["dataset_id"])
            / self._producer["artifact_digest"] / f"{row['target_sample_rate']}hz"
            / relative.parent / filename
        )
        return self._export_root.joinpath(*destination_relative.parts), (
            destination_relative.as_posix()
        )

    @staticmethod
    def _transaction_digest(destination: Path) -> str:
        return hashlib.sha256(str(destination).encode("utf-8")).hexdigest()

    @classmethod
    def _staging_dir_name(cls, destination: Path) -> str:
        return f"{_TRANSACTION_STAGING_PREFIX}{cls._transaction_digest(destination)}"

    @classmethod
    def _transaction_marker_name(cls, destination: Path) -> str:
        return f"{_TRANSACTION_PREFIX}{cls._transaction_digest(destination)}.json"

    def _staging_dir(self, destination: Path) -> Path:
        return destination.parent / self._staging_dir_name(destination)

    def _transaction_marker_path(self, destination: Path) -> Path:
        return self._staging_dir(destination) / "marker.json"

    def _legacy_transaction_marker_path(self, destination: Path) -> Path:
        """已退役的 v2 marker 路径；仅用于检测残留并 fail-fast。"""
        return destination.parent / self._transaction_marker_name(destination)

    def _transaction_temp_path(self, destination: Path, value: Mapping[str, Any]) -> Path:
        return self._staging_dir(destination) / value["temp_name"]

    def _staging_owner_path(self, destination: Path, token: str) -> Path:
        return self._staging_dir(destination) / f"{_STAGING_OWNER_PREFIX}{token}.json"

    def _create_staging_owner(self, destination: Path) -> str:
        """持久化随机 capability；无此记录的 staging 内容绝不自动清理。"""
        token = secrets.token_hex(32)
        owner_path = self._staging_owner_path(destination, token)
        value = {
            "schema": _STAGING_OWNER_SCHEMA,
            "target_path": str(destination),
            "token": token,
        }
        payload = _canonical_json_line(value)
        safe = open_safe_output_path(str(self._export_root), str(owner_path), create=True)
        descriptor = None
        temporary_safe = None
        try:
            os.fchmod(safe.parent_fd, 0o700)
            if safe.target_stat() is not None:
                raise FileExistsError(f"export staging owner already exists: {owner_path}")
            names = os.listdir(safe.parent_fd)
            # owner 发布采用临时文件 + rename；进程若在 rename 前退出，会留下
            # 只有 .owner-<32hex>.tmp 的孤儿 staging。当前 destination lock
            # 已排除并发创建者，此类中间文件可安全回收；其他未知 entry 仍
            # fail-closed，避免误删外部或不完整的导出事务。
            orphan_owner_temps = [
                name for name in names
                if (
                    name.startswith(_STAGING_OWNER_PREFIX)
                    and name.endswith(".tmp")
                    and len(name) == len(_STAGING_OWNER_PREFIX) + 32 + len(".tmp")
                    and all(char in "0123456789abcdef" for char in name[
                        len(_STAGING_OWNER_PREFIX):-len(".tmp")
                    ])
                )
            ]
            if orphan_owner_temps and len(orphan_owner_temps) == len(names):
                for name in orphan_owner_temps:
                    orphan_path = owner_path.parent / name
                    orphan_safe = open_safe_output_path(
                        str(self._export_root), str(orphan_path), create=False,
                    )
                    try:
                        orphan = orphan_safe.target_stat()
                        if orphan is not None:
                            _require_single_link(orphan, "orphan staging owner temporary file")
                            orphan_safe.unlink_target(_snapshot_identity(orphan))
                            orphan_safe.fsync_parent()
                    finally:
                        _close_resources(
                            ((orphan_safe, "orphan staging owner temporary safe path"),),
                            sys.exc_info()[1],
                        )
                names = os.listdir(safe.parent_fd)
            if names:
                raise ValueError("export staging directory is not empty before owner creation")
            descriptor, temporary_name = safe.create_temp_file(".owner-", suffix=".tmp")
            try:
                self._write_all(descriptor, payload)
                os.fsync(descriptor)
            finally:
                _close_resources(
                    ((lambda fd=descriptor: os.close(fd), "staging owner descriptor"),),
                    sys.exc_info()[1],
                )
                descriptor = None
            temporary_safe = open_safe_output_path(
                str(self._export_root), str(owner_path.parent / temporary_name), create=False,
            )
            temporary = temporary_safe.target_stat()
            if temporary is None:
                raise ValueError("staging owner temporary file disappeared")
            _require_single_link(temporary, "staging owner temporary file")
            safe.rename_from_noreplace(temporary_safe)
            safe.fsync_parent()
            published = safe.target_stat()
            if published is None or _snapshot_identity(published) != _snapshot_identity(temporary):
                raise ValueError("staging owner changed after publication")
            return token
        finally:
            _close_resources(
                (
                    ((lambda fd=descriptor: os.close(fd)) if descriptor is not None else None,
                     "staging owner descriptor"),
                    (temporary_safe, "staging owner temporary safe path"),
                    (safe, "staging owner safe path"),
                ),
                sys.exc_info()[1],
            )

    def _read_staging_owner(self, destination: Path) -> str | None:
        staging_path = self._staging_dir(destination)
        staging_safe = None
        safe = None
        descriptor = None
        try:
            staging_safe = open_safe_output_path(
                str(self._export_root), str(staging_path / "probe"), create=False,
            )
            names = os.listdir(staging_safe.parent_fd)
            owners = [
                name for name in names
                if (
                    name.startswith(_STAGING_OWNER_PREFIX)
                    and name.endswith(".json")
                    and len(name) == len(_STAGING_OWNER_PREFIX) + 64 + len(".json")
                    and all(char in "0123456789abcdef" for char in name[
                        len(_STAGING_OWNER_PREFIX):-len(".json")
                    ])
                )
            ]
            if not owners:
                return None
            if len(owners) != 1:
                raise ValueError("export staging owner capability is ambiguous")
            token = owners[0][len(_STAGING_OWNER_PREFIX):-len(".json")]
            owner_path = self._staging_owner_path(destination, token)
            safe = open_safe_output_path(str(self._export_root), str(owner_path), create=False)
            current = safe.target_stat()
            if current is None:
                raise ValueError("export staging owner disappeared")
            _require_single_link(current, "staging owner")
            descriptor = _open_bound_target(
                safe, _snapshot_identity(current), os.O_RDONLY, str(owner_path),
            )
            value = self._read_transaction_value(descriptor)
            if (
                set(value) != {"schema", "target_path", "token"}
                or value["schema"] != _STAGING_OWNER_SCHEMA
                or value["target_path"] != str(destination)
                or value["token"] != token
            ):
                raise ValueError("export staging owner is invalid")
            return token
        except FileNotFoundError:
            return None
        finally:
            _close_resources(
                (
                    ((lambda fd=descriptor: os.close(fd)) if descriptor is not None else None,
                     "staging owner descriptor"),
                    (safe, "staging owner safe path"),
                    (staging_safe, "staging directory safe path"),
                ),
                sys.exc_info()[1],
            )

    @contextmanager
    def _destination_lock(self, destination: Path):
        """锁定 0700 destination 专属目录 fd；anchor 替换不会改变锁对象。"""
        lock_path = destination.parent / (
            f"{_TRANSACTION_LOCK_PREFIX}{self._transaction_digest(destination)}"
        ) / "anchor"
        safe = None
        anchor_descriptor = None
        domain_fd = None
        anchor_identity = None
        domain_identity = None
        try:
            safe = open_safe_output_path(str(self._export_root), str(lock_path), create=True)
            domain_fd = os.dup(safe.parent_fd)
            domain_stat = os.fstat(domain_fd)
            if not stat.S_ISDIR(domain_stat.st_mode):
                raise ValueError("destination lock domain is not a directory")
            os.fchmod(domain_fd, 0o700)
            domain_identity = _snapshot_identity(domain_stat)
            named_before = safe.target_stat()
            anchor_descriptor = safe.open_target(os.O_RDWR | os.O_CREAT, 0o600)
            anchor_stat = os.fstat(anchor_descriptor)
            if not stat.S_ISREG(anchor_stat.st_mode):
                raise ValueError("destination lock anchor is not a regular file")
            _require_single_link(anchor_stat, "destination lock anchor")
            anchor_identity = _snapshot_identity(anchor_stat)
            if named_before is not None and _file_signature(named_before) != _file_signature(anchor_stat):
                raise ValueError("destination lock anchor changed while opening")
            named_after = safe.target_stat()
            if named_after is None or _file_signature(named_after) != _file_signature(anchor_stat):
                raise ValueError("destination lock anchor changed while opening")
            fcntl.flock(domain_fd, fcntl.LOCK_EX)
            locked_domain = os.fstat(domain_fd)
            if _snapshot_identity(locked_domain) != domain_identity:
                raise ValueError("destination lock domain changed while acquiring")
            yield
        finally:
            primary = sys.exc_info()[1]
            cleanup_error = _unlock_and_close(
                domain_fd, "destination lock domain", primary,
            ) if domain_fd is not None else None
            anchor_error = _close_resources(
                (
                    (
                        (lambda fd=anchor_descriptor: os.close(fd))
                        if anchor_descriptor is not None else None,
                        "destination lock anchor descriptor",
                    ),
                ),
                primary,
            )
            if cleanup_error is None:
                cleanup_error = anchor_error
            safe_error = _close_resources(((safe, "destination lock safe path"),), primary)
            if primary is None:
                if cleanup_error is not None:
                    raise cleanup_error
                if safe_error is not None:
                    raise safe_error

    @staticmethod
    def _write_all(descriptor: int, payload: bytes) -> None:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("transaction marker write made no progress")
            offset += written

    def _create_transaction(
        self,
        destination: Path,
        temporary_name: str,
        temporary_identity: tuple[int, int],
        staging_token: str,
        record: dict[str, Any],
    ) -> _ExportTransaction:
        marker_path = self._transaction_marker_path(destination)
        value = {
            "schema": _TRANSACTION_SCHEMA,
            "producer": dict(record["producer"]),
            "target_path": str(destination),
            "temp_name": temporary_name,
            "temp_dev": temporary_identity[0],
            "temp_ino": temporary_identity[1],
            "staging_token": staging_token,
            "record": record,
        }
        payload = _canonical_json_line(value)
        safe = open_safe_output_path(
            str(self._export_root), str(marker_path), create=True,
        )
        temporary_safe = None
        descriptor = None
        identity = None
        try:
            if safe.target_stat() is not None:
                raise FileExistsError(
                    f"export transaction marker already exists: {marker_path}"
                )
            temporary_descriptor, temporary_name = safe.create_temp_file(
                f".marker-{staging_token}.", suffix=".tmp", mode=0o600,
            )
            try:
                self._write_all(temporary_descriptor, payload)
                os.fsync(temporary_descriptor)
            finally:
                _close_resources(
                    ((lambda fd=temporary_descriptor: os.close(fd), "transaction marker staging descriptor"),),
                    sys.exc_info()[1],
                )
            temporary_safe = open_safe_output_path(
                str(self._export_root), str(marker_path.parent / temporary_name), create=False,
            )
            staged = temporary_safe.target_stat()
            if staged is None:
                raise ValueError("transaction marker staging file disappeared")
            _require_single_link(staged, "transaction marker staging file")
            safe.rename_from_noreplace(temporary_safe)
            safe.fsync_parent()
            final_named = safe.target_stat()
            if final_named is None or _snapshot_identity(final_named) != _snapshot_identity(staged):
                raise ValueError("export transaction marker changed after publication")
            descriptor = _open_bound_target(
                safe, _snapshot_identity(final_named), os.O_RDWR, str(marker_path),
            )
            final = os.fstat(descriptor)
            _require_single_link(final, "created export transaction marker")
            identity = _snapshot_identity(final)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return _ExportTransaction(
                safe, descriptor, identity, _file_signature(final), value,
            )
        except BaseException as exc:
            if descriptor is not None:
                _unlock_and_close(descriptor, "transaction marker", exc)
            if identity is not None:
                try:
                    safe.unlink_target(identity)
                    safe.fsync_parent()
                except BaseException as cleanup:
                    _add_exception_note(exc, f"transaction marker rollback failed: {cleanup!r}")
            if temporary_safe is not None:
                _close_resources(((temporary_safe, "transaction marker staging safe path"),), exc)
                temporary_safe = None
            _close_resources(((safe, "transaction marker safe path"),), exc)
            raise
        finally:
            if temporary_safe is not None:
                _close_resources(
                    ((temporary_safe, "transaction marker staging safe path"),),
                    sys.exc_info()[1],
                )

    @staticmethod
    def _read_transaction_value(descriptor: int) -> dict[str, Any]:
        size = os.fstat(descriptor).st_size
        if size <= 0 or size > _TRANSACTION_MAX_BYTES:
            raise ValueError("export transaction marker has invalid size")
        chunks = []
        offset = 0
        while offset < size:
            chunk = os.pread(descriptor, min(64 * 1024, size - offset), offset)
            if not chunk:
                raise ValueError("export transaction marker was truncated")
            chunks.append(chunk)
            offset += len(chunk)
        raw = b"".join(chunks)
        if raw.count(b"\n") != 1 or not raw.endswith(b"\n"):
            raise ValueError(
                "export transaction marker must be one canonical JSON line"
            )
        try:
            value = json.loads(
                raw[:-1].decode("utf-8"),
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
                parse_float=_parse_finite_float,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("export transaction marker is invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("export transaction marker must be an object")
        try:
            canonical = _canonical_json_line(value)
        except ValueError as exc:
            raise ValueError("export transaction marker is not canonical JSON") from exc
        if canonical != raw:
            raise ValueError("export transaction marker is not canonical JSON")
        return value

    def _open_transaction_at(self, marker_path: Path) -> _ExportTransaction | None:
        safe = None
        descriptor = None
        try:
            safe = open_safe_output_path(
                str(self._export_root), str(marker_path), create=False,
            )
            marker_stat = safe.target_stat()
            if marker_stat is None:
                missing_safe = safe
                safe = None
                cleanup_error = _close_resources(
                    ((missing_safe, "export transaction marker safe path"),),
                    None,
                )
                if cleanup_error is not None:
                    raise cleanup_error
                return None
            _require_single_link(marker_stat, f"export transaction marker {marker_path}")
            descriptor = _open_bound_target(
                safe, _snapshot_identity(marker_stat), os.O_RDWR, str(marker_path),
            )
            opened = os.fstat(descriptor)
            _require_single_link(opened, f"export transaction marker {marker_path}")
            if _file_signature(opened) != _file_signature(marker_stat):
                raise ValueError("export transaction marker changed while opening")
            identity = _snapshot_identity(opened)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise FileExistsError("export transaction is active") from exc
            current = safe.target_stat()
            if current is None or _file_signature(current) != _file_signature(opened):
                raise ValueError("export transaction marker changed while opening")
            value = self._read_transaction_value(descriptor)
            after_read = os.fstat(descriptor)
            current = safe.target_stat()
            if current is None or _file_signature(current) != _file_signature(after_read):
                raise ValueError("export transaction marker changed while reading")
            return _ExportTransaction(
                safe, descriptor, identity, _file_signature(after_read), value,
            )
        except FileNotFoundError as exc:
            if descriptor is not None:
                _unlock_and_close(descriptor, "export transaction marker", exc)
                descriptor = None
            _close_resources(
                ((safe, "export transaction marker safe path"),),
                exc,
            )
            return None
        except BaseException as exc:
            if descriptor is not None:
                _unlock_and_close(descriptor, "export transaction marker", exc)
                descriptor = None
            _close_resources(
                ((safe, "export transaction marker safe path"),),
                exc,
            )
            raise

    def _open_transaction(self, destination: Path) -> _ExportTransaction | None:
        # 已退役的 v2 marker（destination 同级的 .audio-export-transaction-*.json）
        # 不再读取：检测到残留即 fail-fast，由人工核对目标文件后删除再重跑。
        legacy_path = self._legacy_transaction_marker_path(destination)
        if os.path.lexists(legacy_path):
            raise ValueError(
                f"legacy v2 export transaction marker is not supported: {legacy_path}; "
                "verify the destination audio and remove this marker before re-running"
            )
        marker_path = self._transaction_marker_path(destination)
        try:
            return self._open_transaction_at(marker_path)
        except ValueError:
            if (
                not self._private_marker_is_incomplete(destination)
                or not self._cleanup_incomplete_staging(destination, remove_marker=True)
            ):
                raise
            return None

    def _private_marker_is_incomplete(self, destination: Path) -> bool:
        try:
            if self._read_staging_owner(destination) is None:
                return False
        except (FileNotFoundError, ValueError):
            return False
        marker_path = self._transaction_marker_path(destination)
        safe = None
        descriptor = None
        try:
            safe = open_safe_output_path(
                str(self._export_root), str(marker_path), create=False,
            )
            marker = safe.target_stat()
            if marker is None or marker.st_size > _TRANSACTION_MAX_BYTES:
                return False
            descriptor = _open_bound_target(
                safe, _snapshot_identity(marker), os.O_RDONLY, str(marker_path),
            )
            raw = os.pread(descriptor, marker.st_size, 0)
            current = os.fstat(descriptor)
            named = safe.target_stat()
            if (
                len(raw) != marker.st_size
                or _file_signature(current) != _file_signature(marker)
                or named is None
                or _file_signature(named) != _file_signature(current)
            ):
                return False
            return not raw or (not raw.endswith(b"\n") and _json_tail_is_incomplete(raw))
        except FileNotFoundError:
            return False
        finally:
            _close_resources(
                (
                    ((lambda fd=descriptor: os.close(fd)) if descriptor is not None else None,
                     "private marker descriptor"),
                    (safe, "private marker safe path"),
                ),
                sys.exc_info()[1],
            )

    def _cleanup_incomplete_staging(
        self, destination: Path, *, remove_marker: bool = False,
    ) -> bool:
        """仅清理持久化随机 owner capability 能证明属于 exporter 的 staging。"""
        try:
            token = self._read_staging_owner(destination)
        except FileNotFoundError:
            return False
        if token is None:
            return False
        owner_path = self._staging_owner_path(destination, token)
        owner_safe = open_safe_output_path(
            str(self._export_root), str(owner_path), create=False,
        )
        try:
            owner = owner_safe.target_stat()
            if owner is None:
                return False
            _require_single_link(owner, "export staging owner")
            marker_path = self._transaction_marker_path(destination)
            marker_safe = open_safe_output_path(
                str(self._export_root), str(marker_path), create=False,
            )
            try:
                marker = marker_safe.target_stat()
                if marker is not None:
                    if not remove_marker:
                        return False
                    _require_single_link(marker, "incomplete private export marker")
                    marker_safe.unlink_target(_snapshot_identity(marker))
                    marker_safe.fsync_parent()
            finally:
                _close_resources(
                    ((marker_safe, "private staging marker safe path"),), sys.exc_info()[1],
                )
            marker_prefix = f".marker-{token}."
            audio_prefix = f".{destination.stem}.{token}."
            for name in os.listdir(owner_safe.parent_fd):
                if name == owner_path.name:
                    continue
                is_marker_temp = (
                    name.startswith(marker_prefix)
                    and name.endswith(".tmp")
                    and len(name) == len(marker_prefix) + 32 + len(".tmp")
                )
                is_audio_temp = (
                    name.startswith(audio_prefix)
                    and name.endswith(destination.suffix)
                    and len(name) == len(audio_prefix) + 32 + len(destination.suffix)
                )
                # safe unlink 在异常路径会先把原 entry 改名为
                # ``.<original>.cleanup-<32hex>``。该隔离文件仍由本 exporter
                # 的 owner capability 绑定，恢复时必须识别并清理，否则每次
                # resume 都会把自己的清理产物误判为外部 entry。
                audio_cleanup_prefix = f".{audio_prefix}"
                is_audio_cleanup = (
                    name.startswith(audio_cleanup_prefix)
                    and name.endswith(f"{destination.suffix}.cleanup-" + name[-32:])
                    and len(name) == len(audio_cleanup_prefix) + 32 + len(destination.suffix) + len(".cleanup-") + 32
                    and name[-(len(".cleanup-") + 32):-32] == ".cleanup-"
                )
                marker_cleanup_prefix = f".{marker_prefix}"
                is_marker_cleanup = (
                    name.startswith(marker_cleanup_prefix)
                    and name.endswith(".tmp.cleanup-" + name[-32:])
                    and len(name) == len(marker_cleanup_prefix) + 32 + len(".tmp.cleanup-") + 32
                    and name[-(len(".cleanup-") + 32):-32] == ".cleanup-"
                )
                if is_audio_cleanup:
                    cleanup_token = name[-32:]
                    is_audio_cleanup = all(char in "0123456789abcdef" for char in cleanup_token)
                if is_marker_cleanup:
                    cleanup_token = name[-32:]
                    is_marker_cleanup = all(char in "0123456789abcdef" for char in cleanup_token)
                if not (is_marker_temp or is_audio_temp or is_audio_cleanup or is_marker_cleanup):
                    raise ValueError("private export staging contains an unrecognized entry")
                entry_path = owner_path.parent / name
                named = os.stat(name, dir_fd=owner_safe.parent_fd, follow_symlinks=False)
                if stat.S_ISDIR(named.st_mode):
                    # path_utils 的 cleanup quarantine 是空目录，不是普通文件。
                    # 绑定父 fd、确认目录 inode 与内容为空后再 rmdir。
                    directory_fd = os.open(
                        name,
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_CLOEXEC", 0),
                        dir_fd=owner_safe.parent_fd,
                    )
                    try:
                        directory_stat = os.fstat(directory_fd)
                        if (
                            not stat.S_ISDIR(directory_stat.st_mode)
                            or _snapshot_identity(directory_stat) != _snapshot_identity(named)
                            or os.listdir(directory_fd)
                        ):
                            raise ValueError("private export staging cleanup quarantine is not empty")
                        latest = os.stat(name, dir_fd=owner_safe.parent_fd, follow_symlinks=False)
                        if _snapshot_identity(latest) != _snapshot_identity(directory_stat):
                            raise ValueError("private export staging cleanup quarantine changed")
                        os.rmdir(name, dir_fd=owner_safe.parent_fd)
                        os.fsync(owner_safe.parent_fd)
                    finally:
                        os.close(directory_fd)
                    continue
                safe = open_safe_output_path(
                    str(self._export_root), str(entry_path), create=False,
                )
                try:
                    entry = safe.target_stat()
                    if entry is None:
                        continue
                    _require_single_link(entry, "private export staging entry")
                    safe.unlink_target(_snapshot_identity(entry))
                    safe.fsync_parent()
                finally:
                    _close_resources(
                        ((safe, "private staging entry safe path"),), sys.exc_info()[1],
                    )
            owner_safe.unlink_target(_snapshot_identity(owner))
            owner_safe.fsync_parent()
            owner_safe.remove_empty_parent()
            return True
        finally:
            _close_resources(
                ((owner_safe, "private staging owner safe path"),), sys.exc_info()[1],
            )

    def _close_transaction(self, transaction: _ExportTransaction, *, remove: bool) -> None:
        primary: BaseException | None = None

        def capture(exc: BaseException) -> None:
            nonlocal primary
            if primary is None:
                primary = exc
            else:
                _add_exception_note(primary, f"additional transaction cleanup failure: {exc!r}")

        try:
            current = transaction.safe.target_stat()
            if (
                current is None
                or _file_signature(current) != transaction.signature
            ):
                raise ValueError("export transaction marker changed before closing")
            _require_single_link(current, "export transaction marker")
            if remove:
                transaction.safe.unlink_target(transaction.identity)
                transaction.safe.fsync_parent()
                self._cleanup_incomplete_staging(Path(transaction.value["target_path"]))
        except BaseException as exc:
            capture(exc)
        try:
            fcntl.flock(transaction.descriptor, fcntl.LOCK_UN)
        except BaseException as exc:
            capture(exc)
        try:
            os.close(transaction.descriptor)
        except BaseException as exc:
            capture(exc)
        try:
            transaction.safe.close()
        except BaseException as exc:
            capture(exc)
        if primary is not None:
            raise primary

    @staticmethod
    def _unlink_bound_entry(safe: SafeOutputPath, identity: tuple[int, int], label: str) -> None:
        current = safe.target_stat()
        if current is None:
            return
        if _snapshot_identity(current) != identity:
            raise RuntimeError(f"foreign inode preserved during cleanup: {label}")
        safe.unlink_target(identity)
        safe.fsync_parent()

    def _recover_transaction(
        self,
        row: Mapping[str, Any],
        destination: Path,
        relative: str,
        manifest: _ManifestState,
        counters: ExportIOCounters,
    ) -> None:
        transaction = self._open_transaction(destination)
        if transaction is None:
            self._cleanup_incomplete_staging(destination)
            return
        temporary_safe = None
        target_safe = None
        remove_marker = False
        try:
            value = transaction.value
            expected_fields = {
                "schema", "producer", "target_path", "temp_name", "temp_dev",
                "temp_ino", "record", "staging_token",
            }
            if set(value) != expected_fields:
                raise ValueError("export transaction marker fields are invalid")
            if (
                value["schema"] != _TRANSACTION_SCHEMA
                or value["producer"] != self._producer
                or value["target_path"] != str(destination)
            ):
                raise ValueError("export transaction marker producer or target is invalid")
            staging_token = value["staging_token"]
            if (
                type(staging_token) is not str
                or len(staging_token) != 64
                or any(char not in "0123456789abcdef" for char in staging_token)
                or self._read_staging_owner(destination) != staging_token
            ):
                raise ValueError("v3 export transaction staging owner is invalid")
            temporary_name = value["temp_name"]
            temporary_prefix = f".{destination.stem}.{staging_token}."
            temporary_suffix = destination.suffix
            if type(temporary_name) is not str:
                raise ValueError("export transaction temporary name is invalid")
            temporary_token = ""
            if (
                temporary_name.startswith(temporary_prefix)
                and temporary_name.endswith(temporary_suffix)
            ):
                suffix_start = len(temporary_name) - len(temporary_suffix)
                temporary_token = temporary_name[len(temporary_prefix):suffix_start]
            if (
                not temporary_name
                or temporary_name == destination.name
                or "\x00" in temporary_name or temporary_name in {".", ".."}
                or "/" in temporary_name or "\\" in temporary_name
                or not temporary_name.startswith(temporary_prefix)
                or not temporary_name.endswith(temporary_suffix)
                or len(temporary_token) != 32
                or any(char not in "0123456789abcdef" for char in temporary_token)
            ):
                raise ValueError("export transaction temporary name is invalid")
            if (
                type(value["temp_dev"]) is not int or value["temp_dev"] < 0
                or type(value["temp_ino"]) is not int or value["temp_ino"] <= 0
            ):
                raise ValueError("export transaction temporary identity is invalid")
            record = value["record"]
            manifest._validate_record(record, 0)
            expected_source = _canonical_source_identity(row)
            expected_audio = _expected_output_metadata(row)
            if (
                record["step"] != self._EXPORT_STEP
                or record["dataset_id"] != row["dataset_id"]
                or record["segment_id"] != row["segment_id"]
                or record["output_relative_path"] != relative
                or record["source"] != expected_source
                or record["audio"] != expected_audio
            ):
                raise ValueError("export transaction record does not match current row")
            owned_identity = (value["temp_dev"], value["temp_ino"])

            target_safe = open_safe_output_path(
                str(self._export_root), str(destination), create=False,
            )
            target_stat = target_safe.target_stat()
            temporary_path = self._transaction_temp_path(destination, value)
            try:
                temporary_safe = open_safe_output_path(
                    str(self._export_root), str(temporary_path), create=False,
                )
            except FileNotFoundError:
                # 正常发布会在目标验证后删除 temporary directory entry；marker
                # 仍指向同一 inode 身份，目标存在时缺少 temp 是可恢复状态。
                temporary_safe = None
                temporary_stat = None
            else:
                temporary_stat = temporary_safe.target_stat()
            if target_stat is not None and _snapshot_identity(target_stat) != owned_identity:
                raise ValueError("export transaction found foreign destination inode")
            if temporary_stat is not None and _snapshot_identity(temporary_stat) != owned_identity:
                raise ValueError("export transaction found foreign temporary inode")
            if target_stat is None and temporary_stat is None:
                raise ValueError("export transaction lost both destination and temporary inode")
            if target_stat is not None and temporary_stat is not None:
                raise ValueError("v3 export transaction has both destination and temporary entries")
            if target_stat is not None:
                _require_single_link(target_stat, "v3 export transaction destination")
            if temporary_stat is not None:
                _require_single_link(temporary_stat, "v3 export transaction temporary")

            if target_stat is not None:
                descriptor = _open_bound_target(
                    target_safe, _snapshot_identity(target_stat),
                    os.O_RDONLY, str(destination),
                    require_single_link=True,
                )
                try:
                    actual_metadata, actual_hash = _verify_audio_descriptor(
                        descriptor,
                        record["audio"],
                        str(destination),
                        counters=counters,
                        phase="transaction_destination",
                    )
                finally:
                    primary = sys.exc_info()[1]
                    cleanup_error = _close_resources(
                        (
                            (
                                lambda fd=descriptor: os.close(fd),
                                "transaction destination descriptor",
                            ),
                        ),
                        primary,
                    )
                    if primary is None and cleanup_error is not None:
                        raise cleanup_error
                if actual_metadata != record["audio"] or actual_hash != record["output_sha256"]:
                    raise ValueError("export transaction destination verification failed")
                if temporary_stat is not None:
                    temporary_descriptor = _open_bound_target(
                        temporary_safe, _snapshot_identity(temporary_stat),
                        os.O_RDONLY, str(temporary_path),
                        require_single_link=True,
                    )
                    try:
                        temporary_metadata, temporary_hash = _verify_audio_descriptor(
                            temporary_descriptor,
                            record["audio"],
                            str(temporary_path),
                            counters=counters,
                            phase="transaction_temporary",
                        )
                    finally:
                        primary = sys.exc_info()[1]
                        cleanup_error = _close_resources(
                            (
                                (
                                    lambda fd=temporary_descriptor: os.close(fd),
                                    "transaction temporary descriptor",
                                ),
                            ),
                            primary,
                        )
                        if primary is None and cleanup_error is not None:
                            raise cleanup_error
                    if (
                        temporary_metadata != record["audio"]
                        or temporary_hash != record["output_sha256"]
                        or temporary_metadata != actual_metadata
                        or temporary_hash != actual_hash
                    ):
                        raise ValueError("export transaction temporary verification failed")
                existing = manifest.records.get(record["output_relative_path"])
                if existing is not None and existing != record:
                    raise ValueError("export transaction conflicts with manifest")
                if existing is None:
                    manifest.append(record)
                if temporary_stat is not None:
                    self._unlink_bound_entry(
                        temporary_safe, owned_identity,
                        str(temporary_path),
                    )
                remove_marker = True
            else:
                # 目标缺失但 marker 与 temporary inode 仍完整时，先 descriptor-bound
                # 复验内容，再以 no-overwrite rename 安全重做发布；任何竞争或身份
                # 不一致都保留现场供后续诊断。
                temporary_descriptor = _open_bound_target(
                    temporary_safe, _snapshot_identity(temporary_stat),
                    os.O_RDONLY, str(temporary_path),
                    require_single_link=True,
                )
                try:
                    temporary_metadata, temporary_hash = _verify_audio_descriptor(
                        temporary_descriptor,
                        record["audio"],
                        str(temporary_path),
                        counters=counters,
                        phase="transaction_temporary",
                    )
                finally:
                    primary = sys.exc_info()[1]
                    cleanup_error = _close_resources(
                        (
                            (
                                lambda fd=temporary_descriptor: os.close(fd),
                                "transaction temporary descriptor",
                            ),
                        ),
                        primary,
                    )
                    if primary is None and cleanup_error is not None:
                        raise cleanup_error
                if (
                    temporary_metadata != record["audio"]
                    or temporary_hash != record["output_sha256"]
                ):
                    raise ValueError("export transaction temporary verification failed")

                target_safe.rename_from_noreplace(temporary_safe)
                published_stat = target_safe.target_stat()
                if published_stat is None or _snapshot_identity(published_stat) != owned_identity:
                    raise ValueError("export transaction recovery found foreign destination inode")
                target_safe.fsync_parent()
                destination_descriptor = _open_bound_target(
                    target_safe, owned_identity, os.O_RDONLY, str(destination),
                    require_single_link=True,
                )
                try:
                    actual_metadata, actual_hash = _verify_audio_descriptor(
                        destination_descriptor,
                        record["audio"],
                        str(destination),
                        counters=counters,
                        phase="transaction_recovered_destination",
                    )
                finally:
                    primary = sys.exc_info()[1]
                    cleanup_error = _close_resources(
                        (
                            (
                                lambda fd=destination_descriptor: os.close(fd),
                                "transaction recovered destination descriptor",
                            ),
                        ),
                        primary,
                    )
                    if primary is None and cleanup_error is not None:
                        raise cleanup_error
                if actual_metadata != record["audio"] or actual_hash != record["output_sha256"]:
                    raise ValueError("export transaction destination verification failed")
                existing = manifest.records.get(record["output_relative_path"])
                if existing is not None and existing != record:
                    raise ValueError("export transaction conflicts with manifest")
                if existing is None:
                    manifest.append(record)
                remove_marker = True
        finally:
            primary = sys.exc_info()[1]
            cleanup_error = _close_resources(
                (
                    (temporary_safe, "temporary safe path"),
                    (target_safe, "target safe path"),
                ),
                primary,
            )
            try:
                self._close_transaction(
                    transaction,
                    remove=remove_marker and primary is None and cleanup_error is None,
                )
            except BaseException as exc:
                if primary is not None:
                    _add_exception_note(primary, f"transaction cleanup failed: {exc!r}")
                elif cleanup_error is None:
                    cleanup_error = exc
                else:
                    _add_exception_note(cleanup_error, f"transaction cleanup failed: {exc!r}")
            if primary is None and cleanup_error is not None:
                raise cleanup_error

    def _publish_atomic(
        self,
        destination: Path,
        audio: np.ndarray,
        sample_rate: int,
        record_context: Mapping[str, Any],
        counters: ExportIOCounters,
    ) -> tuple[dict[str, Any], str, tuple[int, int], dict[str, Any], _ExportTransaction]:
        if not isinstance(audio, np.ndarray) or audio.ndim != 2 or audio.shape[0] <= 0:
            raise ValueError("audio must be a nonempty two-dimensional array")
        if audio.dtype != np.dtype(np.float32):
            raise ValueError("audio must use float32 dtype")
        expected = {
            "format": _OUTPUT_FORMAT,
            "subtype": _OUTPUT_SUBTYPE,
            "sample_rate": sample_rate,
            "frames": int(audio.shape[0]),
            "channels": int(audio.shape[1]),
        }
        if not np.isfinite(audio).all():
            raise ValueError("audio contains non-finite samples")
        if audio.size and float(np.max(np.abs(audio))) > 1.0:
            raise ValueError("audio amplitude exceeds [-1, 1]")

        destination_safe = None
        staging_safe = None
        temporary_safe = None
        write_descriptor = None
        transaction = None
        temporary_name = None
        temporary_identity = None
        staging_token = None
        linked = False
        cleanup_uncertain = False
        transaction_closed = False
        primary_error: BaseException | None = None
        try:
            destination_safe = open_safe_output_path(
                str(self._export_root), str(destination), create=True,
            )
            if destination_safe.target_stat() is not None:
                raise FileExistsError(f"refusing to overwrite existing destination: {destination}")
            staging_safe = open_safe_output_path(
                str(self._export_root), str(self._transaction_marker_path(destination)), create=True,
            )
            if staging_safe.target_stat() is not None:
                raise FileExistsError("export transaction marker already exists")
            staging_token = self._create_staging_owner(destination)
            write_descriptor, temporary_name = staging_safe.create_temp_file(
                prefix=f".{destination.stem}.{staging_token}.", suffix=destination.suffix,
            )
            temporary_stat = os.fstat(write_descriptor)
            temporary_identity = _snapshot_identity(temporary_stat)
            temporary_path = self._staging_dir(destination) / temporary_name
            writer_path = f"/proc/self/fd/{write_descriptor}"
            if not os.path.exists(writer_path):
                writer_path = str(temporary_path)
            try:
                self._writer(writer_path, audio, sample_rate)
                os.fsync(write_descriptor)
            finally:
                writer_close_error = _close_resources(
                    (
                        (
                            lambda fd=write_descriptor: os.close(fd),
                            "temporary writer fd",
                        ),
                    ),
                    sys.exc_info()[1],
                )
                write_descriptor = None
                if writer_close_error is not None:
                    raise writer_close_error

            temporary_safe = open_safe_output_path(
                str(self._export_root), str(temporary_path), create=False,
            )
            current_temp = temporary_safe.target_stat()
            if current_temp is None or _snapshot_identity(current_temp) != temporary_identity:
                raise ValueError("temporary audio inode changed during writing")
            verify_descriptor = _open_bound_target(
                temporary_safe, _snapshot_identity(current_temp),
                os.O_RDONLY, str(temporary_path),
            )
            try:
                metadata, output_sha256 = _verify_audio_descriptor(
                    verify_descriptor,
                    expected,
                    str(temporary_path),
                    counters=counters,
                    phase="temporary_publish",
                )
            finally:
                primary = sys.exc_info()[1]
                cleanup_error = _close_resources(
                    (
                        (
                            lambda fd=verify_descriptor: os.close(fd),
                            "temporary verification descriptor",
                        ),
                    ),
                    primary,
                )
                if primary is None and cleanup_error is not None:
                    raise cleanup_error
            record = {
                "schema": _MANIFEST_SCHEMA,
                "status": "published",
                "producer": dict(record_context["producer"]),
                "step": record_context["step"],
                "dataset_id": record_context["dataset_id"],
                "segment_id": record_context["segment_id"],
                "output_relative_path": record_context["output_relative_path"],
                "source": dict(record_context["source"]),
                "audio": metadata,
                "output_sha256": output_sha256,
            }
            if staging_token is None:
                raise RuntimeError("staging owner token was not initialized")
            transaction = self._create_transaction(
                destination, temporary_name, temporary_identity, staging_token, record,
            )
            # 先标记为可能已发布：即便中断发生在 rename syscall 返回前，
            # resume 也应保留 marker，由 descriptor-bound recovery 决定清理或补写。
            linked = True
            destination_safe.rename_from_noreplace(temporary_safe)
            published_stat = destination_safe.target_stat()
            if published_stat is None or _snapshot_identity(published_stat) != temporary_identity:
                raise ValueError("published destination inode changed")
            _require_single_link(published_stat, "published destination")
            published_descriptor = _open_bound_target(
                destination_safe, _snapshot_identity(published_stat),
                os.O_RDONLY, str(destination),
            )
            try:
                published_metadata, published_hash = _verify_audio_descriptor(
                    published_descriptor,
                    expected,
                    str(destination),
                    counters=counters,
                    phase="published_destination",
                )
            finally:
                primary = sys.exc_info()[1]
                cleanup_error = _close_resources(
                    (
                        (
                            lambda fd=published_descriptor: os.close(fd),
                            "published verification descriptor",
                        ),
                    ),
                    primary,
                )
                if primary is None and cleanup_error is not None:
                    raise cleanup_error
            if published_metadata != metadata or published_hash != output_sha256:
                raise ValueError(f"published audio changed during atomic publish: {destination}")
            destination_safe.fsync_parent()
            # rename 已移走临时 entry；此处复验 staging 名称未被竞争方替换。
            # marker 仍保留，因此 manifest append 被中断时 resume 可补写记录。
            temporary_safe.unlink_target(temporary_identity)
            temporary_safe.fsync_parent()
            return metadata, output_sha256, temporary_identity, record, transaction
        except BaseException as exc:
            # 普通异常可以回滚本次拥有的 inode；进程级中断在 link 完成后必须
            # 保留 destination 与 marker，交给下一次 resume 做 descriptor-bound 恢复。
            primary_error = exc
            preserve_for_recovery = (
                not isinstance(exc, Exception) and linked and transaction is not None
            )
            if not preserve_for_recovery:
                if linked and destination_safe is not None and temporary_identity is not None:
                    try:
                        self._unlink_bound_entry(
                            destination_safe, temporary_identity, str(destination),
                        )
                    except BaseException as cleanup:
                        cleanup_uncertain = True
                        _add_exception_note(exc, f"published destination cleanup failed: {cleanup!r}")
                if not cleanup_uncertain and temporary_identity is not None and temporary_name is not None:
                    cleanup_safe = temporary_safe
                    opened_cleanup_safe = False
                    if cleanup_safe is None:
                        try:
                            cleanup_safe = open_safe_output_path(
                                str(self._export_root),
                                str(self._staging_dir(destination) / temporary_name),
                                create=False,
                            )
                            opened_cleanup_safe = True
                        except FileNotFoundError:
                            cleanup_safe = None
                        except BaseException as cleanup:
                            cleanup_safe = None
                            cleanup_uncertain = True
                            _add_exception_note(exc, f"temporary cleanup path could not be opened: {cleanup!r}")
                    if cleanup_safe is not None:
                        try:
                            self._unlink_bound_entry(
                                cleanup_safe, temporary_identity,
                                str(self._staging_dir(destination) / temporary_name),
                            )
                        except BaseException as cleanup:
                            cleanup_uncertain = True
                            _add_exception_note(exc, f"temporary cleanup failed: {cleanup!r}")
                        finally:
                            if opened_cleanup_safe:
                                close_error = _close_resources(
                                    ((cleanup_safe, "temporary cleanup safe path"),),
                                    exc,
                                )
                                if close_error is not None:
                                    cleanup_uncertain = True
                if transaction is not None:
                    try:
                        self._close_transaction(
                            transaction, remove=not cleanup_uncertain,
                        )
                    except BaseException as cleanup:
                        _add_exception_note(exc, f"export transaction cleanup failed: {cleanup!r}")
                    transaction_closed = True
            else:
                try:
                    self._close_transaction(transaction, remove=False)
                except BaseException as cleanup:
                    _add_exception_note(exc, f"export transaction close after interruption failed: {cleanup!r}")
                transaction_closed = True
            raise
        finally:
            cleanup_resources: list[tuple[Any, str]] = []
            if write_descriptor is not None:
                cleanup_resources.append(
                    (lambda fd=write_descriptor: os.close(fd), "temporary writer fd")
                )
            if destination_safe is not None:
                cleanup_resources.append((destination_safe, "destination safe path"))
            if temporary_safe is not None:
                cleanup_resources.append((temporary_safe, "temporary safe path"))
            if staging_safe is not None:
                cleanup_resources.append((staging_safe, "staging safe path"))
            cleanup_error = _close_resources(cleanup_resources, primary_error)
            if transaction is not None and not transaction_closed:
                if primary_error is not None:
                    try:
                        self._close_transaction(transaction, remove=False)
                    except BaseException as cleanup:
                        _add_exception_note(primary_error, f"export transaction fallback close failed: {cleanup!r}")
                    transaction_closed = True
                elif cleanup_error is not None:
                    try:
                        self._close_transaction(transaction, remove=False)
                    except BaseException as cleanup:
                        _add_exception_note(cleanup_error, f"export transaction fallback close failed: {cleanup!r}")
                    transaction_closed = True
            if primary_error is None and cleanup_error is not None:
                raise cleanup_error

    def export_artifact(self, index: ArtifactIndex) -> tuple[ExportedAudio, ...]:
        """仅消费 expectation-bound 生产 opener 返回的 producer artifact。"""
        if not isinstance(index, ArtifactIndex):
            raise TypeError("index must be an ArtifactIndex")
        if not is_production_bound_artifact_index(index):
            raise ValueError("audio exporter requires a production-bound ArtifactIndex")
        if index.step != self._EXPORT_STEP:
            raise ValueError(
                f"{self._EXPORT_STEP} exporter requires a {self._EXPORT_STEP} ArtifactIndex"
            )
        _safe_path_component(index.dataset_id, "dataset_id")
        producer = _producer_identity(index)
        self._producer = producer
        self._manifests = {}
        rows = iter_artifact_rows(index)
        seen_destinations: set[Path] = set()
        completed: dict[int, ExportedAudio] = {}
        candidate_index = 0
        source_exhausted = False
        io_counters = ExportIOCounters()
        session_pool = (
            VerifiedSourceSessionPool(min(self._workers, self._max_pending), io_counters)
            if self._uses_default_decoder else None
        )
        candidate_buffer: deque[
            tuple[int, Mapping[str, Any], Path, str, _ManifestState, tuple[Any, ...] | None]
        ] = deque()

        def candidates() -> Iterable[
            tuple[int, Mapping[str, Any], Path, str, _ManifestState, tuple[Any, ...] | None]
        ]:
            nonlocal candidate_index
            row_iterator = iter(rows)
            try:
                for source_row in row_iterator:
                    if not isinstance(source_row, Mapping):
                        raise ValueError("export row must be a mapping")
                    row = self._prepare_candidate(source_row)
                    if row is None:
                        continue
                    if row["dataset_id"] != producer["dataset_id"]:
                        raise ValueError("export row dataset_id does not match producer artifact")
                    self._validate_input_root_isolated(row)
                    estimated_bytes = self._estimated_inflight_bytes(row)
                    if estimated_bytes > _MAX_SINGLE_TASK_INFLIGHT_BYTES:
                        raise ValueError(
                            "single export task exceeds hard estimated waveform/verification "
                            "byte limit (not total RSS): "
                            f"estimated={estimated_bytes}, limit={_MAX_SINGLE_TASK_INFLIGHT_BYTES}"
                        )
                    if estimated_bytes > self._max_inflight_bytes:
                        raise ValueError(
                            "single export task exceeds configured estimated "
                            "waveform/verification byte limit (not total RSS): "
                            f"estimated={estimated_bytes}, limit={self._max_inflight_bytes}"
                        )
                    destination, relative = self._destination(row)
                    if destination in seen_destinations:
                        raise ValueError(
                            f"duplicate destination in {self._EXPORT_STEP} export batch: {destination}"
                        )
                    seen_destinations.add(destination)
                    # 只预检路径，不创建目录，避免空批次改变既有输出布局。
                    validate_safe_output_path(str(self._export_root), str(destination))
                    manifest_language = (
                        row["language"] if self._INCLUDE_LANGUAGE_IN_LAYOUT else None
                    )
                    manifest = self._manifest_for(row["dataset_id"], manifest_language)
                    manifest.load()
                    # marker 恢复必须先于候选分发在 producer 侧完成：恢复可能补写
                    # manifest 或清理残留 inode，worker 随后按 resume 契约复验。
                    with self._destination_lock(destination):
                        self._recover_transaction(
                            row, destination, relative, manifest, io_counters,
                        )
                    source_identity = None
                    if session_pool is not None and manifest.get_cached(relative) is None:
                        source_identity = source_session_identity(row)
                    index = candidate_index
                    candidate_index += 1
                    yield index, row, destination, relative, manifest, source_identity
            finally:
                _close_iterator(row_iterator, "rows")

        def export_one(
            row: Mapping[str, Any], destination: Path, relative: str,
            manifest: _ManifestState,
        ) -> _ExportTaskResult:
            session_context = (
                session_pool.acquire(row)
                if session_pool is not None else nullcontext(None)
            )
            with session_context as session:
                if session is None:
                    audio, decoded_sample_rate = self._decoder(row)
                else:
                    audio, decoded_sample_rate = session.decode_interval(
                        row["start_sample"], row["end_sample"],
                    )
                if decoded_sample_rate != row["sample_rate"]:
                    raise ValueError("decoder sample_rate does not match export row")
                expected_shape = (row["end_sample"] - row["start_sample"], row["channels"])
                if not isinstance(audio, np.ndarray) or audio.shape != expected_shape:
                    raise ValueError("decoder did not return the exact export segment shape")
                if audio.dtype != np.dtype(np.float32):
                    raise ValueError("decoder must return float32 audio")
                if not np.isfinite(audio).all():
                    raise ValueError("decoder returned non-finite audio")
                if audio.size and float(np.max(np.abs(audio))) > 1.0:
                    raise ValueError("decoder returned audio amplitude exceeding [-1, 1]")
                # 全通道解码完成后仅保留第一通道（mono）；分析与打分同源，
                # 导出的 FLAC 恒为 1 通道（manifest source.channels 仍记录源通道）。
                audio = audio[:, :1].copy()
                resampled = self._resampler(audio, decoded_sample_rate, row["target_sample_rate"])
                if not isinstance(resampled, np.ndarray) or resampled.ndim != 2:
                    raise ValueError("resampler returned malformed audio")
                if resampled.dtype != np.dtype(np.float32):
                    raise ValueError("resampler must return float32 audio")
                expected_audio = _expected_output_metadata(row)
                if resampled.shape != (expected_audio["frames"], expected_audio["channels"]):
                    raise ValueError("resampler returned unexpected frame or channel count")
                if not np.isfinite(resampled).all():
                    raise ValueError("resampler returned non-finite audio")
                # source waveform 已经读入自有内存；在原子发布前复验 held
                # descriptor 与 pathname binding，避免把失效 source 当成成功导出。
                if session is not None:
                    session.verify_bindings()
                _metadata, _output_sha256, _published_identity, record, transaction = (
                    self._publish_atomic(
                        destination,
                        resampled,
                        row["target_sample_rate"],
                        {
                            "producer": self._producer,
                            "step": self._EXPORT_STEP,
                            "dataset_id": row["dataset_id"],
                            "segment_id": row["segment_id"],
                            "output_relative_path": relative,
                            "source": _canonical_source_identity(row),
                        },
                        io_counters,
                    )
                )
                try:
                    manifest.append(record)
                except Exception as exc:
                    # append 在 fsync 成功后 close/unlock 失败时，record 可能已经可见；
                    # 绝不能据此删除已发布音频。保留完整 marker 由恢复流锁内核验记录。
                    try:
                        self._close_transaction(transaction, remove=False)
                    except BaseException as cleanup:
                        _add_exception_note(exc, f"export transaction close failed: {cleanup!r}")
                    raise
                except BaseException as exc:
                    # KeyboardInterrupt/SystemExit 等进程级中断不能贸然删除已发布
                    # inode；marker 留在磁盘，下一次 resume 会验证并补写/清理。
                    try:
                        self._close_transaction(transaction, remove=False)
                    except BaseException:
                        logger.warning(
                            "failed to close export transaction after manifest interruption: %s",
                            destination, exc_info=True,
                        )
                    raise exc
                else:
                    self._close_transaction(transaction, remove=True)
                return _ExportTaskResult(
                    exported=self._exported_from_row(row, destination, relative),
                    manifest_record=record,
                )

        progress: tqdm | None = None
        resume_progress: tqdm | None = None

        def report_resume_check() -> None:
            if resume_progress is None:
                raise RuntimeError("resume-check progress is not initialized")
            resume_progress.update(1)

        def resume_or_export(
            row: Mapping[str, Any], destination: Path, relative: str,
            manifest: _ManifestState,
        ) -> tuple[ExportedAudio, bool]:
            # 同一 destination 的 resume 判断与新发布必须在稳定锁内连续执行；
            # 不同 destination 仍可并行，且不会以 parent directory 锁串行化。
            with self._destination_lock(destination):
                manifest.load()
                existing = self._resume_existing(
                    row, destination, relative, manifest, io_counters,
                )
                report_resume_check()
                if existing is not None:
                    return existing, False
                return export_one(row, destination, relative, manifest).exported, True

        logger.info(
            "Export step=%s start exports=streaming workers=%d max_pending=%d "
            "max_inflight_gib=%.2f",
            self._EXPORT_STEP, self._workers, self._max_pending,
            self._max_inflight_bytes / 1024 ** 3,
        )
        started_at = time.perf_counter()
        pending: dict[
            Future[tuple[ExportedAudio, bool]],
            tuple[int, int, tuple[Any, ...] | None],
        ] = {}
        active_sources: set[tuple[Any, ...]] = set()
        pending_bytes = 0
        new_exports = 0
        resumed = 0
        failure: BaseException | None = None
        iter_candidates = iter(candidates())

        def fill_candidate_buffer() -> None:
            nonlocal source_exhausted
            buffer_limit = self._max_pending + _MAX_DEFERRED_CANDIDATES
            while not source_exhausted and len(candidate_buffer) + len(pending) < buffer_limit:
                try:
                    candidate_buffer.append(next(iter_candidates))
                except StopIteration:
                    source_exhausted = True

        try:
            progress = tqdm(
                total=None,
                unit="audio",
                desc=f"export:{self._EXPORT_STEP}",
                file=sys.stdout,
                disable=None,
            )
            resume_progress = tqdm(
                total=None,
                unit="audio",
                desc=f"resume-check:{self._EXPORT_STEP}",
                file=sys.stdout,
                disable=None,
            )
            with ThreadPoolExecutor(max_workers=self._workers) as executor:
                while True:
                    # 仅预取一个 bounded window；不会把整个 rows 物化后再提交。
                    fill_candidate_buffer()
                    # 同一 source 只允许一个需要真实导出的任务进入 pending；否则
                    # aggregate reader 会因 worker 完成顺序不同而反复回退到 member 起点。
                    # 已有发布记录的 resume-only 任务不占用该 gate，保持复验并行度。
                    for _ in range(len(candidate_buffer)):
                        (
                            index, row, destination, relative, manifest,
                            source_identity,
                        ) = candidate_buffer.popleft()
                        if (
                            source_identity is not None
                            and source_identity in active_sources
                        ):
                            candidate_buffer.append(
                                (
                                    index, row, destination, relative, manifest,
                                    source_identity,
                                )
                            )
                            continue
                        estimated_bytes = self._estimated_inflight_bytes(row)
                        if pending_bytes + estimated_bytes > self._max_inflight_bytes:
                            candidate_buffer.append(
                                (
                                    index, row, destination, relative, manifest,
                                    source_identity,
                                )
                            )
                            continue
                        future = executor.submit(
                            resume_or_export, row, destination, relative, manifest,
                        )
                        pending[future] = (index, estimated_bytes, source_identity)
                        if source_identity is not None:
                            active_sources.add(source_identity)
                        pending_bytes += estimated_bytes
                        if pending_bytes > self._max_inflight_bytes:
                            raise AssertionError("pending estimated byte gate was exceeded")
                        if len(pending) >= self._max_pending:
                            break
                    if pending:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            index, estimated_bytes, source_identity = pending.pop(future)
                            pending_bytes -= estimated_bytes
                            if source_identity is not None:
                                active_sources.discard(source_identity)
                            try:
                                exported, newly_exported = future.result()
                            except BaseException as exc:
                                if failure is None:
                                    failure = exc
                            else:
                                completed[index] = exported
                                if newly_exported:
                                    new_exports += 1
                                    assert progress is not None
                                    progress.update(1)
                                else:
                                    resumed += 1
                                assert resume_progress is not None
                                resume_progress.set_postfix_str(
                                    f"checked={len(completed)} new={new_exports} resumed={resumed}",
                                    refresh=False,
                                )
                                assert progress is not None
                                progress.set_postfix_str(
                                    f"new={new_exports} resumed={resumed}",
                                    refresh=False,
                                )
                        if failure is not None:
                            for future in pending:
                                future.cancel()
                            raise failure
                        continue
                    if source_exhausted and not candidate_buffer:
                        break
                    if candidate_buffer:
                        raise AssertionError(
                            "validated export candidate could not pass the strict estimated byte gate"
                        )
        finally:
            primary = sys.exc_info()[1]
            _close_iterator(iter_candidates, "candidate")
            cleanup_error = _close_resources(
                (
                    (progress, "export progress"),
                    (resume_progress, "resume-check progress"),
                ),
                primary,
            )
            if session_pool is not None:
                try:
                    session_pool.close(primary or cleanup_error)
                except BaseException as exc:
                    if primary is not None:
                        _add_exception_note(primary, f"source session pool cleanup failed: {exc!r}")
                    elif cleanup_error is not None:
                        _add_exception_note(cleanup_error, f"source session pool cleanup failed: {exc!r}")
                    else:
                        cleanup_error = exc
            self._last_io_counts = io_counters.snapshot()
            logger.info(
                "Export step=%s I/O counts=%s", self._EXPORT_STEP, self._last_io_counts,
            )
            if primary is None and cleanup_error is not None:
                raise cleanup_error
        elapsed = time.perf_counter() - started_at
        logger.info(
            "Export step=%s done exports=%d (new=%d resumed=%d) "
            "elapsed=%.1fs (%.1f audio/s)",
            self._EXPORT_STEP, len(completed), new_exports, resumed, elapsed,
            len(completed) / elapsed if elapsed > 0 else 0.0,
        )
        return tuple(completed[index] for index in range(candidate_index))


class Step4AudioExporter(Step6AudioExporter):
    """Step4 MOS 筛选后的 SE 干净数据集导出器。

    显式消费已成功发布的 Step4 rows，导出 ``mos_selection.decision == 'keep'``
    的 segment；采样率路由与 Step6 完全一致（宽带源最低 16 kHz，低采样率源
    保持 native 且不升采样），输出到 ``audio_exports/step4/`` 下。
    """

    _EXPORT_STEP = "step4"
    _ROW_FIELDS = frozenset(STEP4_ROW_FIELDS)
    _INCLUDE_LANGUAGE_IN_LAYOUT = False

    def _export_eligible(self, row: Mapping[str, Any]) -> bool:
        mos_selection = row.get("mos_selection")
        if not isinstance(mos_selection, Mapping):
            raise ValueError("mos_selection must be a mapping")
        if set(mos_selection) != {
            "decision", "passed_models", "required_models", "policy_id", "policy_hash",
        }:
            raise ValueError("mos_selection fields do not match the exact contract")
        decision = mos_selection.get("decision")
        if decision not in {"keep", "reject"}:
            raise ValueError("mos_selection decision must be exactly 'keep' or 'reject'")
        return decision == "keep"

    def _validate_row(self, row: Mapping[str, Any]) -> None:
        actual_fields = set(row)
        if actual_fields != self._ROW_FIELDS:
            missing = sorted(self._ROW_FIELDS - actual_fields)
            extra = sorted(actual_fields - self._ROW_FIELDS)
            raise ValueError(
                f"{self._EXPORT_STEP} row fields do not match the exact contract: "
                f"missing={missing!r}, extra={extra!r}"
            )
        _safe_path_component(row["dataset_id"], "dataset_id")
        if self._INCLUDE_LANGUAGE_IN_LAYOUT:
            _safe_path_component(row["language"], "language")
        _safe_path_component(row["segment_id"], "segment_id")
        for field in ("source_key", "input_root", "original_path", "format", "subtype"):
            if type(row[field]) is not str or not row[field].strip():
                raise ValueError(f"{field} must be a nonblank exact Python string")
        _safe_relative_path(row["source_relative_path"])
        if type(row["source_size_bytes"]) is not int or row["source_size_bytes"] < 0:
            raise ValueError("source_size_bytes must be a nonnegative Python integer")
        if (type(row["source_sha256"]) is not str or len(row["source_sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in row["source_sha256"])):
            raise ValueError("source_sha256 must be lowercase SHA-256 hex")
        sample_rate = _exact_positive_int(row["sample_rate"], "sample_rate")
        frames = _exact_positive_int(row["frames"], "frames")
        _exact_positive_int(row["channels"], "channels")
        start = row["start_sample"]
        end = row["end_sample"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= frames:
            raise ValueError("segment interval must be nonempty, half-open, and within source frames")
        bandwidth_hz = _finite_nonnegative_real(row["bandwidth_hz"], "bandwidth_hz")
        if bandwidth_hz > sample_rate / 2:
            raise ValueError("bandwidth_hz must be within source Nyquist bounds")
        estimated_sr = row["estimated_sr"]
        if bandwidth_hz == 0.0:
            if estimated_sr is not None:
                raise ValueError("estimated_sr must be None when bandwidth_hz is zero")
        elif type(estimated_sr) is not int or not 0 < estimated_sr <= sample_rate:
            raise ValueError(
                "estimated_sr must be a positive Python integer when bandwidth_hz is nonzero"
            )

    def _routed_sample_rate(self, row: Mapping[str, Any]) -> int:
        """按固定配置采样率或默认有效带宽策略路由，且禁止升采样。"""
        if self._output_sample_rate_hz is not None:
            if self._output_sample_rate_hz > row["sample_rate"]:
                raise ValueError("output_sample_rate_hz must not upsample any selected dataset")
            return self._output_sample_rate_hz
        return target_sample_rate_for_effective_bandwidth(
            _effective_bandwidth_hz(row), row["sample_rate"],
        )

    def _prepare_candidate(self, row: Mapping[str, Any]) -> dict[str, Any] | None:
        """单一 seam：先验证 Step4 exact schema，再执行 selection 与 route。"""
        snapshot = _json_safe_snapshot(row)
        self._validate_row(snapshot)
        if not self._export_eligible(snapshot):
            return None
        snapshot["target_sample_rate"] = self._routed_sample_rate(snapshot)
        return snapshot
