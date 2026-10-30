"""Stage D writer、promote 与 cleanup 共用的安全文件锁。"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import stat
import time
from contextlib import contextmanager
from typing import Iterator

_RUN_LOCK_PREFIX = "pipeline_stage_d_run_"
_STAGE_C_NAMESPACE_LOCK_PREFIX = "pipeline_stage_c_namespace_"
_STAGE_C_RUN_LOCK_PREFIX = "pipeline_stage_c_run_"
_SUBSTAGE_WRITER_LOCK_PREFIX = "pipeline_substage_writer_"
_PROMOTE_LOCK_PREFIX = "pipeline_promote_"
_CLEANUP_LOCK_PREFIX = "pipeline_cleanup_"
_REGISTRY_ENV = "USE_TRAIN_LOCK_REGISTRY"
_REGISTRY_MODE = 0o700
_LOCK_MODE = 0o600


class _LockFD(int):
    """携带稳定 registry dirfd 的锁文件描述符。"""

    registry_fd: int
    registry_path: str
    lock_name: str

    def __new__(
        cls,
        fd: int,
        *,
        registry_fd: int,
        registry_path: str,
        lock_name: str,
    ) -> "_LockFD":
        value = int.__new__(cls, fd)
        value.registry_fd = registry_fd
        value.registry_path = registry_path
        value.lock_name = lock_name
        return value


def _lock_registry_dir() -> str:
    """返回用户私有且持久保留的锁 registry 目录。"""
    configured = os.environ.get(_REGISTRY_ENV)
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return os.path.abspath(os.path.expanduser("~/.cache/use_train/pipeline-locks"))


def _identity_tuple(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _validate_owner_mode(info: os.stat_result, *, mode: int, purpose: str, path: str) -> None:
    """校验节点 owner 与精确权限，拒绝可被其他用户操纵的节点。"""
    uid = getattr(os, "getuid", lambda: info.st_uid)()
    if info.st_uid != uid:
        raise RuntimeError(f"{purpose} owner 不属于当前用户: {path}")
    if stat.S_IMODE(info.st_mode) != mode:
        raise RuntimeError(f"{purpose} mode 必须是 {mode:04o}: {path}")


def _ensure_lock_registry() -> str:
    registry = _lock_registry_dir()
    try:
        os.makedirs(registry, mode=_REGISTRY_MODE, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"无法创建锁 registry {registry}: {exc}") from exc
    try:
        info = os.lstat(registry)
    except OSError as exc:
        raise RuntimeError(f"无法检查锁 registry {registry}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"锁 registry 必须是目录且不得是符号链接: {registry}")
    _validate_owner_mode(info, mode=_REGISTRY_MODE, purpose="锁 registry", path=registry)
    return registry


def _lock_path_for_identity(identity: str, prefix: str) -> str:
    digest = hashlib.md5(identity.encode("utf-8")).hexdigest()[:16]
    return os.path.join(_ensure_lock_registry(), f"{prefix}{digest}.lock")


def _identity_lock_path(path: str, prefix: str) -> str:
    return _lock_path_for_identity(os.path.realpath(path), prefix)


def run_lock_path(stage_d_dir: str) -> str:
    """按 ``stage_d_dir`` 的 realpath 返回稳定锁文件路径。"""
    return _identity_lock_path(stage_d_dir, _RUN_LOCK_PREFIX)


def _physical_identity_lock_path(info: os.stat_result, prefix: str) -> str:
    """按已打开节点的物理身份返回稳定锁文件路径。"""
    return _lock_path_for_identity(f"{info.st_dev}:{info.st_ino}", prefix)


def writer_lock_path(output_dir: str) -> str:
    """按已存在输出目录的物理身份返回稳定 writer 锁路径。"""
    try:
        info = os.stat(output_dir)
    except OSError as exc:
        raise RuntimeError(f"无法检查子阶段输出目录 {output_dir}: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"子阶段输出路径必须是目录: {output_dir}")
    return _physical_identity_lock_path(info, _SUBSTAGE_WRITER_LOCK_PREFIX)


_PathIdentity = tuple[int, int, int, str | None]
_PathSnapshot = list[tuple[str, _PathIdentity | None]]


def _path_prefixes(path: str) -> list[str]:
    prefixes = [os.path.sep]
    current = os.path.sep
    for component in os.path.abspath(path).split(os.path.sep):
        if component:
            current = os.path.join(current, component)
            prefixes.append(current)
    return prefixes


def _path_identity(path: str) -> _PathIdentity:
    info = os.lstat(path)
    target = os.readlink(path) if stat.S_ISLNK(info.st_mode) else None
    return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode), target


def _snapshot_path_chain(path: str) -> _PathSnapshot:
    """快照现有 ancestor；首个缺失节点之后以 ``None`` 占位。"""
    snapshot: _PathSnapshot = []
    missing = False
    for prefix in _path_prefixes(path):
        if missing:
            snapshot.append((prefix, None))
            continue
        try:
            identity = _path_identity(prefix)
        except FileNotFoundError:
            missing = True
            identity = None
        except OSError as exc:
            raise RuntimeError(f"无法检查输出目录或 ancestor {prefix}: {exc}") from exc
        snapshot.append((prefix, identity))
    return snapshot


def _open_directory_chain(
    path: str,
    *,
    create_missing: bool,
) -> tuple[int, _PathSnapshot]:
    """逐级 ``openat`` 无 symlink 路径，并记录实际打开的 ancestor identity。"""
    flags = (
        getattr(os, "O_PATH", os.O_RDONLY)
        | os.O_DIRECTORY
        | os.O_CLOEXEC
        | os.O_NOFOLLOW
    )
    fd = os.open(os.path.sep, flags)
    root_info = os.fstat(fd)
    snapshot: _PathSnapshot = [
        (os.path.sep, (root_info.st_dev, root_info.st_ino, stat.S_IFDIR, None))
    ]
    current = os.path.sep
    try:
        for component in os.path.abspath(path).split(os.path.sep):
            if not component:
                continue
            current = os.path.join(current, component)
            try:
                child_fd = os.open(component, flags, dir_fd=fd)
            except FileNotFoundError as exc:
                if not create_missing:
                    raise RuntimeError(
                        f"带 symlink 的输出路径 target 必须已完整存在: {current}"
                    ) from exc
                try:
                    os.mkdir(component, mode=0o777, dir_fd=fd)
                except FileExistsError:
                    pass
                except OSError as mkdir_exc:
                    raise RuntimeError(
                        f"无法安全创建输出目录 {current}: {mkdir_exc}"
                    ) from mkdir_exc
                try:
                    child_fd = os.open(component, flags, dir_fd=fd)
                except OSError as open_exc:
                    raise RuntimeError(
                        f"无法安全打开输出目录 {current}: {open_exc}"
                    ) from open_exc
            except OSError as exc:
                raise RuntimeError(
                    f"输出目录 ancestor 必须是无 symlink 的实体目录: {current}: {exc}"
                ) from exc
            child_info = os.fstat(child_fd)
            snapshot.append(
                (current, (child_info.st_dev, child_info.st_ino, stat.S_IFDIR, None))
            )
            os.close(fd)
            fd = child_fd
        return fd, snapshot
    except BaseException:
        os.close(fd)
        raise


def _validate_path_snapshot(snapshot: _PathSnapshot) -> None:
    for path, expected in snapshot:
        try:
            current = _path_identity(path)
        except OSError as exc:
            raise RuntimeError(f"输出目录或 ancestor 被替换: {path}: {exc}") from exc
        if expected is None:
            if current[2] != stat.S_IFDIR:
                raise RuntimeError(f"输出目录或 ancestor 被替换: {path}")
        elif current != expected:
            raise RuntimeError(f"输出目录或 ancestor 被替换: {path}")


def _validate_output_directory(
    output_fd: int,
    requested_path: str,
    canonical_path: str,
    requested_snapshot: _PathSnapshot,
    canonical_snapshot: _PathSnapshot,
    output_identity: tuple[int, int],
) -> None:
    """确认业务目录与 requested/canonical ancestor 仍绑定原 identity。"""
    _validate_path_snapshot(requested_snapshot)
    if _snapshot_path_chain(canonical_path) != canonical_snapshot:
        raise RuntimeError(f"输出目录 canonical ancestor 被替换: {canonical_path}")
    try:
        opened = os.fstat(output_fd)
        visible = os.stat(requested_path)
    except OSError as exc:
        raise RuntimeError(f"输出目录被替换或消失: {requested_path}: {exc}") from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(visible.st_mode)
        or _identity_tuple(opened) != output_identity
        or _identity_tuple(visible) != output_identity
        or os.path.realpath(requested_path) != canonical_path
    ):
        raise RuntimeError(f"输出目录被替换: {requested_path}")


def _prepare_output_directory(
    output_dir: str,
) -> tuple[str, str, int, _PathSnapshot, _PathSnapshot, os.stat_result]:
    """安全创建并打开输出目录，锁 identity 只从持有的目录 fd 派生。"""
    requested_path = os.path.abspath(output_dir)
    requested_snapshot = _snapshot_path_chain(requested_path)
    has_symlink = any(
        identity is not None and identity[2] == stat.S_IFLNK
        for _, identity in requested_snapshot
    )
    if has_symlink:
        try:
            existing = os.stat(requested_path)
        except OSError as exc:
            raise RuntimeError(
                f"带 symlink 的输出路径必须已指向现有目录: {requested_path}: {exc}"
            ) from exc
        if not stat.S_ISDIR(existing.st_mode):
            raise RuntimeError(f"带 symlink 的输出路径必须指向目录: {requested_path}")
    canonical_path = os.path.realpath(requested_path)
    physical_path = canonical_path if has_symlink else requested_path
    output_fd, canonical_snapshot = _open_directory_chain(
        physical_path,
        create_missing=not has_symlink,
    )
    try:
        output_info = os.fstat(output_fd)
        canonical_path = os.path.realpath(requested_path)
        _validate_output_directory(
            output_fd,
            requested_path,
            canonical_path,
            requested_snapshot,
            canonical_snapshot,
            _identity_tuple(output_info),
        )
        return (
            requested_path,
            canonical_path,
            output_fd,
            requested_snapshot,
            canonical_snapshot,
            output_info,
        )
    except BaseException:
        os.close(output_fd)
        raise


def promote_lock_path(final_root: str) -> str:
    """按 final-root realpath 返回稳定共享发布锁路径。"""
    return _identity_lock_path(final_root, _PROMOTE_LOCK_PREFIX)


def cleanup_lock_path(output_root: str) -> str:
    """按 output-root realpath 返回稳定 cleanup 锁路径。"""
    return _identity_lock_path(output_root, _CLEANUP_LOCK_PREFIX)


def _validate_registry_fd(registry_fd: int, registry_path: str) -> None:
    info = os.fstat(registry_fd)
    if not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"锁 registry 必须是目录: {registry_path}")
    _validate_owner_mode(
        info,
        mode=_REGISTRY_MODE,
        purpose="锁 registry",
        path=registry_path,
    )


def _validate_lock_identity(
    fd: _LockFD,
    lock_path: str,
    before: os.stat_result | None,
    *,
    purpose: str,
) -> os.stat_result:
    identity = os.fstat(fd)
    if stat.S_ISLNK(identity.st_mode) or not stat.S_ISREG(identity.st_mode):
        raise RuntimeError(f"{purpose}必须是 regular file: {lock_path}")
    if identity.st_nlink != 1:
        raise RuntimeError(f"{purpose}不得是 hardlink: {lock_path}")
    _validate_owner_mode(identity, mode=_LOCK_MODE, purpose=purpose, path=lock_path)
    if before is not None and _identity_tuple(identity) != _identity_tuple(before):
        raise RuntimeError(f"{purpose}在打开期间被替换: {lock_path}")
    _validate_registry_fd(fd.registry_fd, fd.registry_path)
    try:
        current = os.stat(
            fd.lock_name,
            dir_fd=fd.registry_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"{purpose}在打开期间消失: {lock_path}") from exc
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
        raise RuntimeError(f"{purpose}路径类型已变化: {lock_path}")
    if _identity_tuple(current) != _identity_tuple(identity):
        raise RuntimeError(f"{purpose}在打开期间被替换: {lock_path}")
    _validate_owner_mode(current, mode=_LOCK_MODE, purpose=purpose, path=lock_path)
    return identity


def _close_lock_fd(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass
    registry_fd = getattr(fd, "registry_fd", None)
    if registry_fd is not None:
        try:
            os.close(registry_fd)
        except OSError:
            pass


def _open_lock_file(lock_path: str, *, purpose: str) -> int:
    """通过稳定 registry dirfd 打开锁文件并校验 owner/mode/identity。"""
    registry_path = os.path.dirname(os.path.abspath(lock_path))
    lock_name = os.path.basename(lock_path)
    registry_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        registry_fd = os.open(registry_path, registry_flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise RuntimeError(
                f"锁 registry 必须是目录且不得是符号链接: {registry_path}"
            ) from exc
        raise RuntimeError(f"无法打开锁 registry {registry_path}: {exc}") from exc

    raw_fd = None
    fd = None
    try:
        _validate_registry_fd(registry_fd, registry_path)
        try:
            before = os.stat(lock_name, dir_fd=registry_fd, follow_symlinks=False)
        except FileNotFoundError:
            before = None
        except OSError as exc:
            raise RuntimeError(f"无法安全检查{purpose} {lock_path}: {exc}") from exc

        if before is not None:
            if stat.S_ISLNK(before.st_mode):
                raise RuntimeError(f"{purpose}不得是符号链接: {lock_path}")
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError(f"{purpose}必须是 regular file: {lock_path}")
            if before.st_nlink != 1:
                raise RuntimeError(f"{purpose}不得是 hardlink: {lock_path}")
            _validate_owner_mode(before, mode=_LOCK_MODE, purpose=purpose, path=lock_path)

        flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        try:
            raw_fd = os.open(lock_name, flags, _LOCK_MODE, dir_fd=registry_fd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise RuntimeError(f"{purpose}不得是符号链接: {lock_path}") from exc
            raise RuntimeError(f"无法打开{purpose} {lock_path}: {exc}") from exc

        fd = _LockFD(
            raw_fd,
            registry_fd=registry_fd,
            registry_path=registry_path,
            lock_name=lock_name,
        )
        raw_fd = None
        os.fchmod(fd, _LOCK_MODE)
        _validate_lock_identity(fd, lock_path, before, purpose=purpose)
        return fd
    except BaseException as exc:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        elif raw_fd is not None:
            try:
                os.close(raw_fd)
            except OSError:
                pass
        try:
            os.close(registry_fd)
        except OSError:
            pass
        if isinstance(exc, RuntimeError):
            raise
        if isinstance(exc, OSError):
            raise RuntimeError(f"无法安全校验{purpose} {lock_path}: {exc}") from exc
        raise


def acquire_lock(lock_path: str, *, purpose: str, non_blocking: bool = True) -> int:
    """获取已打开锁文件；flock 后通过稳定 dirfd 再次确认 inode。"""
    fd = _open_lock_file(lock_path, purpose=purpose)
    flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if non_blocking else 0)
    try:
        fcntl.flock(fd, flags)
        os.fchmod(fd, _LOCK_MODE)
        _validate_lock_identity(fd, lock_path, None, purpose=purpose)
        return fd
    except (IOError, OSError, RuntimeError) as exc:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        _close_lock_fd(fd)
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"无法获取{purpose} {lock_path}: {exc}") from exc


def release_lock(fd: int) -> None:
    """释放锁及稳定 registry dirfd；持久保留 registry 与锁文件。"""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except Exception:
        pass
    _close_lock_fd(fd)


def acquire_run_lock(stage_d_dir: str) -> int:
    """非阻塞获取 Stage D run 排它锁；冲突时抛出 ``RuntimeError``。"""
    path = run_lock_path(stage_d_dir)
    try:
        return acquire_lock(path, purpose="run 锁")
    except RuntimeError as exc:
        raise RuntimeError(
            f"无法获取 run 锁 {path}，可能有 Stage D writer、promote 或 cleanup 在运行: {exc}"
        ) from exc


def release_run_lock(fd: int) -> None:
    """释放并关闭 ``acquire_run_lock`` 返回的文件描述符。"""
    release_lock(fd)


@contextmanager
def stage_d_run_lock(stage_d_dir: str) -> Iterator[None]:
    """在上下文期间持有指定 Stage D run 的排它锁。"""
    fd = acquire_run_lock(stage_d_dir)
    try:
        yield
    finally:
        release_run_lock(fd)


def _normalize_writer_lock_timeout(timeout: float) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0:
        raise ValueError("writer lock timeout 必须是非负有限数值")
    normalized = float(timeout)
    if not normalized < float("inf"):
        raise ValueError("writer lock timeout 必须是非负有限数值")
    return normalized


def _acquire_lock_with_timeout(
    lock_path: str,
    *,
    purpose: str,
    timeout: float,
) -> int:
    """轮询获取排它锁；超时错误包含锁路径与等待时长。"""
    timeout = _normalize_writer_lock_timeout(timeout)
    fd = _open_lock_file(lock_path, purpose=purpose)
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.fchmod(fd, _LOCK_MODE)
                _validate_lock_identity(fd, lock_path, None, purpose=purpose)
                return fd
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    waited = time.monotonic() - started
                    raise RuntimeError(
                        f"获取{purpose}超时: lock_path={lock_path}, waited={waited:.3f}s"
                    )
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    except BaseException:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        _close_lock_fd(fd)
        raise


@contextmanager
def stage_c_run_lock(
    output_dir: str,
    *,
    timeout: float = 30.0,
) -> Iterator[None]:
    """按 namespace→inode 顺序锁定 Stage C 输出根并覆盖整次发布。"""
    timeout = _normalize_writer_lock_timeout(timeout)
    (
        requested,
        canonical,
        output_fd,
        requested_snapshot,
        canonical_snapshot,
        output_info,
    ) = _prepare_output_directory(output_dir)
    namespace_lock_fd = None
    physical_lock_fd = None
    output_identity = _identity_tuple(output_info)
    deadline = time.monotonic() + timeout
    try:
        namespace_path = _lock_path_for_identity(
            canonical,
            _STAGE_C_NAMESPACE_LOCK_PREFIX,
        )
        namespace_lock_fd = _acquire_lock_with_timeout(
            namespace_path,
            purpose=f"Stage C run namespace 锁（output={canonical}）",
            timeout=timeout,
        )
        physical_path = _physical_identity_lock_path(
            output_info,
            _STAGE_C_RUN_LOCK_PREFIX,
        )
        physical_lock_fd = _acquire_lock_with_timeout(
            physical_path,
            purpose=f"Stage C run inode 锁（output={canonical}）",
            timeout=max(0.0, deadline - time.monotonic()),
        )
        _validate_output_directory(
            output_fd,
            requested,
            canonical,
            requested_snapshot,
            canonical_snapshot,
            output_identity,
        )
        try:
            yield
        finally:
            _validate_output_directory(
                output_fd,
                requested,
                canonical,
                requested_snapshot,
                canonical_snapshot,
                output_identity,
            )
    finally:
        if physical_lock_fd is not None:
            release_lock(physical_lock_fd)
        if namespace_lock_fd is not None:
            release_lock(namespace_lock_fd)
        os.close(output_fd)


@contextmanager
def substage_writer_lock(
    output_dir: str,
    *,
    timeout: float = 30.0,
) -> Iterator[None]:
    """绑定实际输出目录持排它锁，并拒绝锁期间的路径替换。"""
    timeout = _normalize_writer_lock_timeout(timeout)
    (
        requested,
        canonical,
        output_fd,
        requested_snapshot,
        canonical_snapshot,
        output_info,
    ) = _prepare_output_directory(output_dir)
    lock_fd = None
    output_identity = _identity_tuple(output_info)
    try:
        path = _physical_identity_lock_path(
            output_info, _SUBSTAGE_WRITER_LOCK_PREFIX
        )
        lock_fd = _acquire_lock_with_timeout(
            path,
            purpose=f"子阶段 writer 锁（output={canonical}）",
            timeout=timeout,
        )
        _validate_output_directory(
            output_fd,
            requested,
            canonical,
            requested_snapshot,
            canonical_snapshot,
            output_identity,
        )
        try:
            yield
        finally:
            _validate_output_directory(
                output_fd,
                requested,
                canonical,
                requested_snapshot,
                canonical_snapshot,
                output_identity,
            )
    finally:
        if lock_fd is not None:
            release_lock(lock_fd)
        os.close(output_fd)
