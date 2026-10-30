"""MOS 主进程与 worker 共用的严格 JSONL 和资源清理协议（仅 stdlib）。"""

from __future__ import annotations

import errno
import json
import math
import os
import stat
import sys
from collections.abc import Iterable, Iterator, Mapping
from typing import Any


def add_cleanup_note(primary: BaseException, label: str, error: BaseException) -> None:
    """附加清理诊断，且绝不覆盖原始业务异常。"""
    try:
        note = f"cleanup failed for {label}: {error!r}"
        add_note = getattr(primary, "add_note", None)
        if callable(add_note):
            add_note(note)
        else:
            notes = list(getattr(primary, "__notes__", ()))
            notes.append(note)
            primary.__notes__ = notes
    except BaseException:
        pass


def record_cleanup_error(
    first_error: BaseException | None,
    error: BaseException,
    label: str,
) -> BaseException:
    if first_error is not None:
        add_cleanup_note(first_error, label, error)
        return first_error
    return error


def finalize_streams(
    streams: Iterable[Any],
    primary: BaseException | None = None,
    *,
    sync: bool,
) -> None:
    """按 flush → fsync → close 顺序穷尽清理所有 stream。"""
    first_error: BaseException | None = None
    for index, stream in enumerate(streams):
        if sync:
            try:
                stream.flush()
            except BaseException as exc:
                first_error = record_cleanup_error(
                    first_error, exc, f"stream {index} flush",
                )
            try:
                os.fsync(stream.fileno())
            except BaseException as exc:
                first_error = record_cleanup_error(
                    first_error, exc, f"stream {index} fsync",
                )
        try:
            stream.close()
        except BaseException as exc:
            first_error = record_cleanup_error(
                first_error, exc, f"stream {index} close",
            )
    if first_error is not None:
        if primary is not None:
            add_cleanup_note(primary, "stream finalization", first_error)
        else:
            raise first_error


def close_fd_preserving(
    descriptor: int,
    primary: BaseException,
    label: str,
) -> None:
    try:
        os.close(descriptor)
    except BaseException as exc:
        add_cleanup_note(primary, label, exc)


def open_regular_stream(path: str, label: str) -> tuple[Any, os.stat_result]:
    descriptor = None
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise ValueError(f"{label} must not be a symlink or special file") from exc
            raise
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError(f"{label} must be a regular file")
        stream = os.fdopen(descriptor, "rb", closefd=True)
        descriptor = None
        return stream, initial
    except BaseException as exc:
        if descriptor is not None:
            close_fd_preserving(descriptor, exc, f"{label} descriptor")
        raise


def open_new_regular_stream(path: str, label: str) -> Any:
    descriptor = None
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    )
    try:
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise ValueError(f"{label} must be a new regular file") from exc
            if exc.errno == errno.EEXIST:
                try:
                    existing = os.lstat(path)
                except FileNotFoundError:
                    raise
                if stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode):
                    raise ValueError(f"{label} must be a new regular file") from exc
            raise
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be a regular file")
        stream = os.fdopen(descriptor, "wb", closefd=True)
        descriptor = None
        return stream
    except BaseException as exc:
        if descriptor is not None:
            close_fd_preserving(descriptor, exc, f"{label} descriptor")
        raise


def file_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON constant is not allowed: {value}")


def parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is forbidden: {value}")
    return parsed


def encode_json_line(value: Mapping[str, Any], label: str, *, max_line_bytes: int) -> bytes:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not serializable JSON") from exc
    if len(payload) > max_line_bytes:
        raise ValueError(f"{label} exceeds JSONL line limit {max_line_bytes} bytes")
    return payload


def decode_json_line(payload: bytes, label: str, *, max_line_bytes: int) -> Any:
    if len(payload) > max_line_bytes:
        raise ValueError(f"{label} exceeds JSONL line limit {max_line_bytes} bytes")
    if not payload.endswith(b"\n"):
        raise ValueError(f"{label} is incomplete or missing LF newline")
    try:
        return json.loads(
            payload[:-1].decode("utf-8", "strict"),
            object_pairs_hook=strict_json_object,
            parse_constant=reject_json_constant,
            parse_float=parse_finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} contains invalid JSON") from exc


def iter_bounded_jsonl(
    source: Any,
    label: str,
    *,
    max_line_bytes: int,
    max_file_bytes: int,
) -> Iterator[bytes]:
    """读取绑定 regular file；支持 caller-owned binary stream。"""
    owns_stream = isinstance(source, (str, bytes, os.PathLike))
    if owns_stream:
        try:
            stream, initial = open_regular_stream(os.fspath(source), label)
        except FileNotFoundError:
            raise RuntimeError(f"{label} is missing") from None
    else:
        stream = source
        initial = os.fstat(stream.fileno())
    try:
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if initial.st_size > max_file_bytes:
            raise ValueError(f"{label} exceeds JSONL file limit {max_file_bytes} bytes")
        total = 0
        while True:
            payload = stream.readline(max_line_bytes + 1)
            if not payload:
                break
            if len(payload) > max_line_bytes:
                raise ValueError(f"{label} contains a line exceeding {max_line_bytes} bytes")
            total += len(payload)
            if total > max_file_bytes:
                raise ValueError(f"{label} exceeds JSONL file limit {max_file_bytes} bytes")
            if not payload.endswith(b"\n"):
                raise ValueError(f"{label} is incomplete or missing LF newline")
            yield payload
        final = os.fstat(stream.fileno())
        if file_signature(final) != file_signature(initial) or total != final.st_size:
            raise ValueError(f"{label} changed while being read")
    finally:
        if owns_stream:
            finalize_streams((stream,), sys.exc_info()[1], sync=False)
