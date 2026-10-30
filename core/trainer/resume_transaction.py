"""Resume checkpoint snapshot transaction 与本地 lease 协议。"""

from __future__ import annotations

import atexit
import fcntl
import functools
import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

from core.trainer.checkpoint_state import checkpoint_content_identity


def snapshot_lock_path(snapshot: Path) -> Path:
    """返回 rank-local snapshot 的独立 consumer lease 文件。"""
    return snapshot.with_name(f"{snapshot.name}.lock")


def snapshot_guard_path(snapshot: Path) -> Path:
    """从唯一 snapshot 名恢复同 checkpoint basename 的转换 guard。"""
    name = snapshot.name
    if not name.startswith(".") or ".resume-" not in name:
        raise ValueError(f"invalid resume snapshot name: {name}")
    canonical_name = name[1:].rsplit(".resume-", 1)[0]
    return snapshot.with_name(f".{canonical_name}.resume.guard.lock")


def matches_snapshot_name(name: str, canonical_name: str, suffix: str) -> bool:
    """按字面 basename 严格匹配本协议的 ``pid-uuid`` snapshot 名。"""
    prefix = f".{canonical_name}.resume-"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return False
    token = name[len(prefix) : -len(suffix)]
    pid, separator, nonce = token.partition("-")
    return bool(
        separator
        and pid.isdecimal()
        and len(nonce) == 32
        and all(char in "0123456789abcdef" for char in nonce)
    )


def reclaim_stale_snapshots(canonical: Path) -> None:
    """仅回收同 basename 且未被独占 lease 保护的 snapshot。"""
    for candidate in canonical.parent.iterdir():
        if not matches_snapshot_name(
            candidate.name, canonical.name, ".snapshot"
        ):
            continue
        lock_path = snapshot_lock_path(candidate)
        lock_fd = None
        try:
            lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            candidate.unlink(missing_ok=True)
            lock_path.unlink(missing_ok=True)
        except FileNotFoundError:
            continue
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(lock_fd)


def reclaim_stale_snapshot_temps(canonical: Path) -> None:
    """持有 basename guard 时回收严格匹配本协议的遗留 copy temp。"""
    for candidate in canonical.parent.iterdir():
        if matches_snapshot_name(
            candidate.name, canonical.name, ".snapshot.tmp"
        ):
            candidate.unlink(missing_ok=True)


class ResumeSnapshotTransaction:
    """封装 immutable snapshot 的创建与 owner lease 发布。

    ``owner`` 保持兼容 facade 所需的 lease 字典与 atexit callback；本组件不调用
    任何 distributed collective，跨 rank 汇聚仍由 ``UnifiedTrainingEntry`` 管理。
    """

    @staticmethod
    def freeze(
        owner: Any,
        checkpoint_path: Path,
        *,
        staging_dir: Path | None = None,
    ) -> tuple[Path, tuple[int, str]]:
        canonical = checkpoint_path.resolve(strict=True)
        staging = (
            canonical.parent if staging_dir is None else Path(staging_dir).resolve()
        )
        staging.mkdir(parents=True, exist_ok=True)
        snapshot = staging / (
            f".{canonical.name}.resume-{os.getpid()}-{uuid.uuid4().hex}.snapshot"
        )
        temporary = snapshot.with_suffix(f"{snapshot.suffix}.tmp")
        lock_path = snapshot_lock_path(snapshot)
        guard_path = staging / f".{canonical.name}.resume.guard.lock"
        guard_fd = None
        lease_fd = None
        exit_callback = None
        published = False
        try:
            guard_fd = os.open(guard_path, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(guard_fd, fcntl.LOCK_EX)
            staging_canonical = staging / canonical.name
            reclaim_stale_snapshots(staging_canonical)
            reclaim_stale_snapshot_temps(staging_canonical)
            digest = hashlib.sha256()
            copied_size = 0
            with canonical.open("rb") as source, temporary.open("xb") as target:
                before = os.fstat(source.fileno())
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
                    digest.update(chunk)
                    copied_size += len(chunk)
                target.flush()
                os.fsync(target.fileno())
                after = os.fstat(source.fileno())
            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or copied_size != after.st_size
            ):
                raise RuntimeError("checkpoint changed while creating resume snapshot")
            os.chmod(temporary, 0o400)
            os.replace(temporary, snapshot)
            published = True
            identity = (copied_size, digest.hexdigest())
            lease_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_stat = os.fstat(lease_fd)
            lock_identity = (lock_stat.st_dev, lock_stat.st_ino)
            exit_callback = functools.partial(
                owner._atexit_release_resume_checkpoint_owner,
                snapshot,
            )
            atexit.register(exit_callback)
            leases = getattr(owner, "_resume_snapshot_leases", None)
            if leases is None:
                leases = {}
                owner._resume_snapshot_leases = leases
            leases[snapshot] = (
                lease_fd,
                lock_path,
                exit_callback,
                identity,
                lock_identity,
            )
            lease_fd = None
            exit_callback = None
            return snapshot, identity
        except BaseException:
            if exit_callback is not None:
                atexit.unregister(exit_callback)
            if lease_fd is not None:
                try:
                    fcntl.flock(lease_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(lease_fd)
            if published:
                try:
                    snapshot.unlink(missing_ok=True)
                except OSError:
                    pass
            temporary.unlink(missing_ok=True)
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        finally:
            if guard_fd is not None:
                try:
                    fcntl.flock(guard_fd, fcntl.LOCK_UN)
                finally:
                    os.close(guard_fd)

    @staticmethod
    def best_effort_release(
        snapshot: Path,
        lock_path: Path,
        lease_fd: int,
        snapshot_identity: tuple[int, str],
        lock_identity: tuple[int, int],
    ) -> None:
        """释放本地 lease；仅最后一个 consumer 校验 identity 后删除目录项。"""
        guard_fd = None
        cleanup_fd = None
        try:
            guard_path = snapshot_guard_path(snapshot)
            guard_fd = os.open(guard_path, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(guard_fd, fcntl.LOCK_EX)
            try:
                fcntl.flock(lease_fd, fcntl.LOCK_UN)
            finally:
                os.close(lease_fd)
            lease_fd = None
            cleanup_fd = os.open(lock_path, os.O_RDWR)
            lock_stat = os.fstat(cleanup_fd)
            if (lock_stat.st_dev, lock_stat.st_ino) != lock_identity:
                return
            try:
                fcntl.flock(cleanup_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            try:
                actual_identity = checkpoint_content_identity(snapshot)
            except (OSError, RuntimeError):
                return
            if actual_identity != snapshot_identity:
                return
            # 目录项复核与删除锚定同一 dirfd 且不跟随符号链接：stat 与 unlink 之间
            # 路径不再被重新解析；协议内的并发创建已由独占 guard flock 排除。
            dir_fd = os.open(lock_path.parent, os.O_RDONLY)
            try:
                path_stat = os.stat(
                    lock_path.name, dir_fd=dir_fd, follow_symlinks=False
                )
                if (path_stat.st_dev, path_stat.st_ino) != lock_identity:
                    return
                snapshot.unlink(missing_ok=True)
                os.unlink(lock_path.name, dir_fd=dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, ValueError):
            pass
        finally:
            if lease_fd is not None:
                try:
                    fcntl.flock(lease_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(lease_fd)
            if cleanup_fd is not None:
                try:
                    fcntl.flock(cleanup_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(cleanup_fd)
            if guard_fd is not None:
                try:
                    fcntl.flock(guard_fd, fcntl.LOCK_UN)
                finally:
                    os.close(guard_fd)
