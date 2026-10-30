"""Generation-3 audio pipeline output/dataset run lock."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import socket
import stat
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .path_utils import open_safe_directory


class RunLockError(RuntimeError):
    """运行锁抢占、冲突或安全释放失败。"""


class RunLock:
    """使用持久 lock inode 与 ``flock`` 保护 output-root/dataset。"""

    _LOCK_DIRECTORY = ".audio_filter_pipeline_locks"
    _MAX_OWNER_BYTES = 1024 * 1024
    _READ_CHUNK_BYTES = 64 * 1024

    def __init__(self, output_root: str, dataset_id: str) -> None:
        if type(output_root) is not str or not output_root.strip():
            raise ValueError("run lock output_root must be a nonblank string")
        if type(dataset_id) is not str or not dataset_id.strip():
            raise ValueError("run lock dataset_id must be a nonblank string")
        if dataset_id in {".", ".."} or "/" in dataset_id or "\\" in dataset_id:
            raise ValueError("run lock dataset_id must be a single path component")
        raw_output_root = os.path.abspath(output_root)
        # 保留旧 owner JSON/lock_path 的 canonical output_root 语义；实际打开时
        # 仍使用未解析的路径逐层 O_NOFOLLOW，因而 output_root 的 symlink 会拒绝。
        self.output_root = os.path.realpath(raw_output_root)
        self._lock_directory_path = str(Path(raw_output_root) / self._LOCK_DIRECTORY)
        self.dataset_id = dataset_id
        lock_dir = Path(self.output_root) / self._LOCK_DIRECTORY
        # 锁目录自身已限定 output-root 身份；文件名只绑定 dataset，避免同一目录
        # 经 bind mount 等不同 pathname 访问时分裂成两把锁。
        lock_key = f"audio_filter_pipeline.run_lock\0{dataset_id}"
        self.lock_path = str(
            lock_dir / f"{uuid.uuid5(uuid.NAMESPACE_URL, lock_key).hex}.lock"
        )
        self._lock_name = os.path.basename(self.lock_path)
        self._protocol_anchor_name = f".{self._lock_name}.anchor"
        self._fd: int | None = None
        self._anchor_fd: int | None = None
        self._anchor_identity: tuple[int, int] | None = None
        self._token: str | None = None
        self._identity: tuple[int, int] | None = None
        self._owner_bytes: bytes | None = None
        self._acquired = False

    @property
    def token(self) -> str | None:
        return self._token

    def _owner_payload(self) -> dict[str, Any]:
        if self._token is None:
            raise RunLockError("run lock token is not initialized")
        return {
            "schema": "audio_filter_pipeline.run_lock.v1",
            "token": self._token,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "argv": list(sys.argv),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_root": self.output_root,
            "dataset_id": self.dataset_id,
        }

    @classmethod
    def _read_owner_bytes(cls, fd: int) -> bytes:
        chunks: list[bytes] = []
        total = 0
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            while True:
                chunk = os.read(fd, cls._READ_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > cls._MAX_OWNER_BYTES:
                    raise RunLockError("run lock owner metadata is too large")
                chunks.append(chunk)
        except RunLockError:
            raise
        except OSError as exc:
            raise RunLockError("run lock owner metadata is unreadable") from exc
        return b"".join(chunks)

    @staticmethod
    def _decode_owner(raw: bytes) -> dict[str, Any]:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RunLockError("run lock owner metadata is unreadable") from exc
        if not isinstance(value, dict) or type(value.get("token")) is not str:
            raise RunLockError("run lock owner metadata is invalid")
        return value

    @classmethod
    def _read_owner(cls, fd: int) -> dict[str, Any]:
        """读取 owner JSON；保留旧测试和调用方使用的 helper 契约。"""
        return cls._decode_owner(cls._read_owner_bytes(fd))

    @classmethod
    def _write_owner(cls, fd: int, payload: bytes) -> None:
        owner_bytes = payload + b"\n"
        if len(owner_bytes) > cls._MAX_OWNER_BYTES:
            raise RunLockError("run lock owner metadata is too large")
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        remaining = memoryview(owner_bytes)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError(errno.EIO, "run lock owner metadata write made no progress")
            remaining = remaining[written:]
        os.fsync(fd)

    @staticmethod
    def _close_quietly(fd: int | None) -> None:
        if fd is None:
            return
        try:
            os.close(fd)
        except OSError:
            pass

    @staticmethod
    def _validate_lock_inode(
        identity: os.stat_result,
        label: str,
        *,
        expected_nlink: int | None = None,
    ) -> None:
        if not stat.S_ISREG(identity.st_mode):
            raise RunLockError(f"{label} must be a regular file")
        if identity.st_uid != os.geteuid():
            raise RunLockError(f"{label} owner UID does not match the current user")
        if expected_nlink is not None and identity.st_nlink != expected_nlink:
            raise RunLockError(
                f"{label} has unsafe link topology: expected "
                f"{expected_nlink} links, found {identity.st_nlink}"
            )

    def _open_lock_directory(self) -> int:
        descriptor: int | None = None
        try:
            descriptor = open_safe_directory(self._lock_directory_path, create=True)
            directory_stat = os.fstat(descriptor)
            if directory_stat.st_uid != os.geteuid():
                raise RunLockError(
                    "run lock directory owner UID does not match the current user"
                )
            os.fchmod(descriptor, 0o700)
            return descriptor
        except RunLockError:
            self._close_quietly(descriptor)
            raise
        except (OSError, ValueError) as exc:
            self._close_quietly(descriptor)
            raise RunLockError(
                f"cannot safely open run lock directory {self._lock_directory_path!r}"
            ) from exc

    def _open_lock_file(self, directory_fd: int) -> tuple[int, tuple[int, int]]:
        lock_name = os.path.basename(self.lock_path)
        try:
            before = os.stat(lock_name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            before = None
        except OSError as exc:
            raise RunLockError("run lock file cannot be inspected safely") from exc
        if before is not None:
            if stat.S_ISLNK(before.st_mode):
                raise RunLockError("run lock file must not be a symlink")
            self._validate_lock_inode(before, "run lock file")

        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            fd = os.open(lock_name, flags, 0o600, dir_fd=directory_fd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise RunLockError("run lock file must not be a symlink") from exc
            raise RunLockError(f"cannot open run lock file {self.lock_path!r}") from exc

        try:
            try:
                identity = os.fstat(fd)
                self._validate_lock_inode(identity, "run lock file")
                if before is not None and (
                    identity.st_dev,
                    identity.st_ino,
                ) != (before.st_dev, before.st_ino):
                    raise RunLockError("run lock file was replaced while opening")
                current = os.stat(lock_name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISLNK(current.st_mode):
                    raise RunLockError("run lock file must not be a symlink")
                self._validate_lock_inode(current, "run lock file")
                if (current.st_dev, current.st_ino) != (
                    identity.st_dev,
                    identity.st_ino,
                ):
                    raise RunLockError("run lock file was replaced while opening")
                return fd, (identity.st_dev, identity.st_ino)
            except FileNotFoundError as exc:
                raise RunLockError("run lock file disappeared while opening") from exc
            except OSError as exc:
                raise RunLockError("run lock file cannot be verified safely") from exc
        except BaseException:
            self._close_quietly(fd)
            raise

    def _open_protocol_anchor(
        self, directory_fd: int,
    ) -> tuple[int, tuple[int, int]]:
        """打开持久协议锚点；调用方必须在加锁后校验 pathname 绑定。"""
        try:
            try:
                anchor_stat = os.stat(
                    self._protocol_anchor_name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                lock_stat = os.stat(
                    self._lock_name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                self._validate_lock_inode(
                    lock_stat, "run lock file", expected_nlink=1,
                )
                os.link(
                    self._lock_name,
                    self._protocol_anchor_name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                anchor_stat = os.stat(
                    self._protocol_anchor_name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
        except RunLockError:
            raise
        except OSError as exc:
            raise RunLockError("run lock protocol anchor cannot be prepared safely") from exc

        if stat.S_ISLNK(anchor_stat.st_mode):
            raise RunLockError("run lock protocol anchor must not be a symlink")
        self._validate_lock_inode(
            anchor_stat, "run lock protocol anchor", expected_nlink=2,
        )

        flags = (
            os.O_RDWR
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            fd = os.open(self._protocol_anchor_name, flags, dir_fd=directory_fd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise RunLockError("run lock protocol anchor must not be a symlink") from exc
            raise RunLockError("cannot open run lock protocol anchor") from exc

        try:
            identity = os.fstat(fd)
            self._validate_lock_inode(
                identity, "run lock protocol anchor", expected_nlink=2,
            )
            if (identity.st_dev, identity.st_ino) != (
                anchor_stat.st_dev,
                anchor_stat.st_ino,
            ):
                raise RunLockError("run lock protocol anchor was replaced while opening")
            return fd, (identity.st_dev, identity.st_ino)
        except BaseException:
            self._close_quietly(fd)
            raise

    def _validate_path_binding(
        self, directory_fd: int, identity: tuple[int, int], *, phase: str,
    ) -> None:
        for name, label in (
            (self._lock_name, "run lock file"),
            (self._protocol_anchor_name, "run lock protocol anchor"),
        ):
            try:
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError as exc:
                raise RunLockError(f"{label} disappeared {phase}") from exc
            except OSError as exc:
                raise RunLockError(f"{label} cannot be verified {phase}") from exc
            if (
                stat.S_ISLNK(current.st_mode)
                or (current.st_dev, current.st_ino) != identity
            ):
                raise RunLockError(f"{label} was replaced {phase}")
            self._validate_lock_inode(current, label, expected_nlink=2)

    def _release_acquire_failure(
        self,
        fd: int | None,
        locked: bool,
        anchor_fd: int | None = None,
        anchor_locked: bool = False,
    ) -> None:
        # 失败路径也必须逐项尝试清理；例如文件解锁失败不能跳过父目录
        # 锚点的解锁和关闭。
        if fd is not None:
            if locked:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except BaseException:
                    pass
            self._close_quietly(fd)
        if anchor_fd is not None:
            if anchor_locked:
                try:
                    fcntl.flock(anchor_fd, fcntl.LOCK_UN)
                except BaseException:
                    pass
            self._close_quietly(anchor_fd)

    def acquire(self) -> "RunLock":
        if self._acquired or self._fd is not None or self._anchor_fd is not None:
            raise RunLockError("run lock is already acquired")

        directory_fd: int | None = None
        anchor_fd: int | None = None
        fd: int | None = None
        locked = False
        anchor_locked = False
        self._token = uuid.uuid4().hex
        try:
            directory_fd = self._open_lock_directory()
            anchor_fd = directory_fd
            directory_fd = None
            anchor_stat = os.fstat(anchor_fd)
            if not stat.S_ISDIR(anchor_stat.st_mode):
                raise RunLockError("run lock directory anchor is not a directory")
            self._anchor_identity = (anchor_stat.st_dev, anchor_stat.st_ino)
            try:
                fcntl.flock(anchor_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise RunLockError(
                        f"run lock is already held for output_root={self.output_root!r}, "
                        f"dataset_id={self.dataset_id!r}: {self.lock_path}"
                    ) from exc
                raise RunLockError("cannot lock run lock directory") from exc
            anchor_locked = True

            # 目录 flock 只串行化 pathname/协议锚点的准备；运行期互斥由
            # dataset-specific 协议 inode 提供，不能阻塞其他 dataset。
            fd, public_identity = self._open_lock_file(anchor_fd)
            protocol_fd, identity = self._open_protocol_anchor(anchor_fd)
            self._close_quietly(fd)
            fd = protocol_fd
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise RunLockError(
                        f"run lock is already held for output_root={self.output_root!r}, "
                        f"dataset_id={self.dataset_id!r}: {self.lock_path}"
                    ) from exc
                raise RunLockError(f"cannot lock run lock {self.lock_path!r}") from exc
            locked = True
            if public_identity != identity:
                raise RunLockError("run lock pathname drifted from protocol anchor")
            self._validate_path_binding(anchor_fd, identity, phase="after locking")

            # 拿到协议 inode flock 后才修改 owner；写入后再次确认 namespace
            # 仍绑定该 inode，避免把已漂移 pathname 当作已成功 acquisition。
            os.fchmod(fd, 0o600)
            payload = json.dumps(
                self._owner_payload(),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            self._write_owner(fd, payload)
            self._validate_path_binding(anchor_fd, identity, phase="after owner write")
            try:
                fcntl.flock(anchor_fd, fcntl.LOCK_UN)
            except OSError as exc:
                raise RunLockError("cannot unlock run lock directory") from exc
            anchor_locked = False

            self._fd = fd
            self._identity = identity
            self._owner_bytes = payload + b"\n"
            self._anchor_fd = anchor_fd
            self._acquired = True
            fd = None
            anchor_fd = None
            return self
        except BaseException as exc:
            self._release_acquire_failure(fd, locked, anchor_fd, anchor_locked)
            self._token = None
            self._identity = None
            self._owner_bytes = None
            self._fd = None
            self._anchor_fd = None
            self._anchor_identity = None
            self._acquired = False
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                # 中断/退出类异常保持原语义向上传播，不得吞成 RunLockError。
                raise
            if isinstance(exc, RunLockError):
                raise
            raise RunLockError(f"cannot acquire run lock {self.lock_path!r}") from exc
        finally:
            self._close_quietly(directory_fd)

    def _validate_owned_file(self) -> None:
        fd = self._fd
        if (
            fd is None
            or self._token is None
            or self._identity is None
            or self._owner_bytes is None
        ):
            raise RunLockError("run lock ownership state is incomplete")

        try:
            identity = os.fstat(fd)
        except OSError as exc:
            raise RunLockError("run lock descriptor is unreadable before release") from exc
        if not stat.S_ISREG(identity.st_mode):
            raise RunLockError("run lock descriptor is no longer a regular file")
        if (identity.st_dev, identity.st_ino) != self._identity:
            raise RunLockError("run lock descriptor identity changed before release")

        raw = self._read_owner_bytes(fd)
        owner = self._decode_owner(raw)
        if owner.get("token") != self._token:
            raise RunLockError("run lock owner token changed before release")
        if raw != self._owner_bytes:
            raise RunLockError("run lock owner metadata changed before release")

    def _validate_anchor(self) -> None:
        fd = self._anchor_fd
        if fd is None or self._anchor_identity is None:
            raise RunLockError("run lock directory anchor state is incomplete")
        try:
            identity = os.fstat(fd)
        except OSError as exc:
            raise RunLockError("run lock directory anchor is unreadable before release") from exc
        if not stat.S_ISDIR(identity.st_mode):
            raise RunLockError("run lock directory anchor is no longer a directory")
        if (identity.st_dev, identity.st_ino) != self._anchor_identity:
            raise RunLockError("run lock directory anchor identity changed before release")

    @staticmethod
    def _close_locked_descriptor(
        fd: int, label: str,
    ) -> tuple[BaseException | None, bool]:
        """解锁并关闭一个 held fd；任一步失败都继续执行下一步。"""
        first_error: BaseException | None = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except BaseException as exc:
            first_error = RunLockError(f"{label} cannot be unlocked safely")
            first_error.__cause__ = exc
        closed = False
        try:
            os.close(fd)
            closed = True
        except BaseException as exc:
            error = RunLockError(f"{label} descriptor cannot be closed safely")
            error.__cause__ = exc
            if first_error is None:
                first_error = error
            else:
                RunLock._add_exception_note(first_error, str(error))
        return first_error, closed

    def _release_owned_file(self) -> None:
        self._validate_owned_file()
        fd = self._fd
        assert fd is not None
        cleanup_error, closed = self._close_locked_descriptor(fd, "run lock")
        if closed:
            self._fd = None
        if cleanup_error is not None:
            raise cleanup_error

    @staticmethod
    def _add_exception_note(error: BaseException, note: str) -> None:
        try:
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note(note)
                return
            notes = list(getattr(error, "__notes__", ()))
            notes.append(note)
            error.__notes__ = notes
        except BaseException:
            pass

    def release(self) -> None:
        if not self._acquired:
            return
        # 所有权、parent anchor 与 pathname 绑定任一校验失败时保留现场，
        # 允许调用方修复或诊断后再次 release。
        if self._fd is None or self._identity is None or self._anchor_fd is None:
            raise RunLockError("run lock ownership state is incomplete")
        self._validate_anchor()
        try:
            fcntl.flock(self._anchor_fd, fcntl.LOCK_EX)
        except OSError as exc:
            raise RunLockError("cannot lock run lock directory before release") from exc
        try:
            self._validate_path_binding(
                self._anchor_fd, self._identity, phase="before release",
            )
            self._validate_owned_file()
        except BaseException:
            try:
                fcntl.flock(self._anchor_fd, fcntl.LOCK_UN)
            except BaseException:
                pass
            raise

        cleanup_error: BaseException | None = None
        if self._fd is not None:
            try:
                self._release_owned_file()
            except BaseException as exc:
                cleanup_error = exc
        if self._anchor_fd is not None:
            anchor_fd = self._anchor_fd
            anchor_error, closed = self._close_locked_descriptor(
                anchor_fd, "run lock directory anchor",
            )
            if closed:
                self._anchor_fd = None
            if anchor_error is not None:
                if cleanup_error is None:
                    cleanup_error = anchor_error
                else:
                    self._add_exception_note(cleanup_error, str(anchor_error))

        if self._fd is None and self._anchor_fd is None:
            self._acquired = False
            self._token = None
            self._identity = None
            self._owner_bytes = None
            self._anchor_identity = None
        if cleanup_error is not None:
            raise cleanup_error

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        business_error = exc_value if isinstance(exc_value, BaseException) else None
        try:
            self.release()
        except BaseException as release_error:
            if business_error is None:
                raise
            try:
                self._add_exception_note(
                    business_error,
                    f"RunLock release failed: {release_error!r}",
                )
            except BaseException:
                # 释放诊断绝不能覆盖业务异常。
                pass
        return False
