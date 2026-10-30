"""Durable two-file publication primitives for manifest materialization."""

from __future__ import annotations

import hashlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable

from core.utils.file_transaction import publish_staged_files, unlink_file_if_exists


@contextmanager
def exclusive_output_set_lock(targets: Iterable[Path]):
    """按规范化目标路径集合获取稳定的跨进程排他锁。

    锁依赖 POSIX ``fcntl.flock``（惰性导入，避免 Windows 上 import 期失败）；
    非 POSIX 平台显式报错，而不是静默退化为无锁发布。
    """
    if os.name != "posix":  # pragma: no cover - Windows 无 fcntl.flock
        raise RuntimeError(
            "exclusive_output_set_lock requires POSIX fcntl.flock; "
            f"platform {os.name!r} is not supported"
        )
    import fcntl

    normalized_targets = sorted(
        str(Path(target).resolve(strict=False)) for target in targets
    )
    if not normalized_targets:
        raise ValueError("at least one output target is required")
    digest = hashlib.sha256("\0".join(normalized_targets).encode("utf-8")).hexdigest()
    lock_directory = Path(normalized_targets[0]).parent
    lock_directory.mkdir(parents=True, exist_ok=True)
    lock_path = lock_directory / f".materialize-{digest}.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def validate_distinct_output_paths(output: Path, vocab_output: Path) -> None:
    resolved_output = output.resolve(strict=False)
    resolved_vocab = vocab_output.resolve(strict=False)
    if resolved_output == resolved_vocab:
        raise ValueError(
            "output_path and category_vocab_path must refer to different physical files"
        )
    if output.exists() and vocab_output.exists() and os.path.samefile(output, vocab_output):
        raise ValueError(
            "output_path and category_vocab_path must refer to different physical files"
        )


def unlink_if_exists(path: str | Path) -> None:
    """通过共享事务 helper 的稳定父目录 dirfd 清理文件。"""
    unlink_file_if_exists(path)


def backup_existing_target(
    target: Path,
    *,
    fsync: Callable[[int], None] = os.fsync,
    unlink: Callable[[str | Path], None] = unlink_if_exists,
) -> Path | None:
    if not target.exists():
        return None
    backup_file = tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=".", suffix=".bak", delete=False
    )
    backup = Path(backup_file.name)
    try:
        with target.open("rb") as source, backup_file:
            while chunk := source.read(1024 * 1024):
                backup_file.write(chunk)
            backup_file.flush()
            fsync(backup_file.fileno())
    except Exception:
        unlink(backup)
        raise
    return backup


def open_atomic_temp(target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=".",
        suffix=".tmp",
        delete=False,
    )
    return temp_file, Path(temp_file.name)


def publish_output_pair(
    strict_temp_path: Path,
    output: Path,
    vocab_temp_path: Path,
    vocab_output: Path,
    *,
    replace: Callable[[Path, Path], None] | None = None,
    backup_existing: Callable[[Path], Path | None] | None = None,
    unlink: Callable[[str | Path], None] | None = None,
    fsync_directories: Callable[[Iterable[Path]], None] | None = None,
) -> None:
    """默认以稳定 dirfd 发布；显式 callback 仅保留给故障注入测试。"""
    staged = (strict_temp_path, vocab_temp_path)
    targets = (output, vocab_output)
    callbacks = {}
    if replace is not None:
        callbacks["replace"] = replace
    if backup_existing is not None:
        callbacks["backup_existing"] = backup_existing
    if unlink is not None:
        callbacks["unlink_if_exists"] = unlink
    if fsync_directories is not None:
        callbacks["fsync_parent_directories"] = fsync_directories
    cleanup = unlink or unlink_file_if_exists
    try:
        publish_staged_files(staged, targets, **callbacks)
    finally:
        for temporary_path in staged:
            cleanup(temporary_path)


__all__ = [
    "backup_existing_target",
    "exclusive_output_set_lock",
    "open_atomic_temp",
    "publish_output_pair",
    "unlink_if_exists",
    "validate_distinct_output_paths",
]
