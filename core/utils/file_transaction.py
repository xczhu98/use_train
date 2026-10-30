"""同一发布集合内多个普通文件的异常安全事务 helper。

该事务保证普通异常路径会恢复完整旧文件集合，并序列化使用本 helper 的并发
writer；它不提供跨路径瞬时可见性，也不承诺进程崩溃或断电后的自动恢复。
"""

from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Iterator, TextIO

try:
    import fcntl
except ImportError:  # pragma: no cover - 项目部署目标为 Linux。
    fcntl = None

TextWriter = Callable[[TextIO], None]
DirectoryFds = dict[Path, int]


def write_text_files_transactionally(
    outputs: Iterable[tuple[Path, TextWriter]],
    *,
    overwrite: bool,
) -> None:
    """先完整落盘 staging 文件，再以稳定父目录 FD 发布整个文件集合。"""
    entries = [(_absolute_path(target), writer) for target, writer in outputs]
    if not entries:
        raise ValueError("at least one output file is required")
    targets = [target for target, _ in entries]

    with _lock_parent_directories(targets, create_missing=True) as directory_fds:
        _validate_target_leaves(targets, directory_fds)
        _validate_distinct_targets(targets, directory_fds)
        if not overwrite:
            existing = [str(target) for target in targets if _target_exists(target, directory_fds)]
            if existing:
                raise FileExistsError(
                    "output already exists; pass --overwrite to replace: " + ", ".join(existing)
                )

        staged: list[Path] = []
        try:
            for target, writer in entries:
                descriptor, temporary_path = _create_temporary_file(
                    directory_fds[_parent_key(target)],
                    target.parent,
                    prefix=f".{target.name}.",
                    suffix=".tmp",
                )
                staged.append(temporary_path)
                os.fchmod(descriptor, 0o644)
                with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
                    writer(output_file)
                    output_file.flush()
                    os.fsync(output_file.fileno())
            _publish_staged_files(staged, targets, directory_fds=directory_fds)
        except BaseException:
            for temporary_path in staged:
                _unlink_at(temporary_path, directory_fds)
            raise


def publish_staged_files(
    staged: Iterable[Path],
    targets: Iterable[Path],
    *,
    replace: Callable[[Path, Path], None] | None = None,
    backup_existing: Callable[[Path], Path | None] | None = None,
    unlink_if_exists: Callable[[Path], None] | None = None,
    fsync_parent_directories: Callable[[Iterable[Path]], None] | None = None,
) -> None:
    """锁定相关父目录并以稳定 dirfd 发布已 fsync 的 staging 文件。"""
    staged_paths = [_absolute_path(path) for path in staged]
    target_paths = [_absolute_path(path) for path in targets]
    if len(staged_paths) != len(target_paths) or not target_paths:
        raise ValueError("staged files and targets must have the same non-zero length")
    with _lock_parent_directories([*staged_paths, *target_paths]) as directory_fds:
        _publish_staged_files(
            staged_paths,
            target_paths,
            directory_fds=directory_fds,
            replace=replace,
            backup_existing=backup_existing,
            unlink_if_exists=unlink_if_exists,
            fsync_parent_directories=fsync_parent_directories,
        )


def _publish_staged_files(
    staged_paths: list[Path],
    target_paths: list[Path],
    *,
    directory_fds: DirectoryFds | None = None,
    replace: Callable[[Path, Path], None] | None = None,
    backup_existing: Callable[[Path], Path | None] | None = None,
    unlink_if_exists: Callable[[Path], None] | None = None,
    fsync_parent_directories: Callable[[Iterable[Path]], None] | None = None,
) -> None:
    """锁内提交 staging 文件；失败时恢复整个目标集合。"""
    if directory_fds is None:
        directory_fds = {}
    replace_fn = replace or (
        lambda source, target: _replace_at(source, target, directory_fds)
    )
    backup_fn = backup_existing or (
        lambda target: _backup_existing_target_at(target, directory_fds)
    )
    unlink_fn = unlink_if_exists or (
        lambda path: _unlink_at(path, directory_fds)
    )
    fsync_dirs_fn = fsync_parent_directories or (
        lambda paths: _fsync_parent_directories_at(paths, directory_fds)
    )
    exists_fn = lambda path: _path_exists(path, directory_fds)
    staged_identities = _validate_publish_paths(
        staged_paths, target_paths, directory_fds
    )
    target_modes = [
        stat.S_IMODE(target_stat.st_mode)
        if (target_stat := _target_lstat(target, directory_fds)) is not None
        else 0o644
        for target in target_paths
    ]
    backups: list[Path | None] = []
    published = [False] * len(target_paths)
    retained_backups: set[Path] = set()
    try:
        for target in target_paths:
            backups.append(backup_fn(target))
        _verify_staging_identities(
            staged_paths, staged_identities, directory_fds
        )
        for temporary_path, identity, mode in zip(
            staged_paths, staged_identities, target_modes
        ):
            _verify_staging_identity(temporary_path, identity, directory_fds)
            _chmod_at(temporary_path, mode, directory_fds)
            _verify_staging_identity(temporary_path, identity, directory_fds)
        for index, (temporary_path, target, identity) in enumerate(
            zip(staged_paths, target_paths, staged_identities)
        ):
            _verify_staging_identity(temporary_path, identity, directory_fds)
            replace_fn(temporary_path, target)
            published[index] = True
            _verify_published_identity(target, identity, directory_fds)
        fsync_dirs_fn(target_paths)
    except BaseException as publish_error:
        rollback_errors: list[BaseException] = []
        for index, (target, backup) in enumerate(zip(target_paths, backups)):
            if not published[index]:
                continue
            try:
                if backup is None:
                    unlink_fn(target)
                else:
                    replace_fn(backup, target)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
                if backup is not None and exists_fn(backup):
                    retained_backups.add(backup)
        try:
            fsync_dirs_fn(target_paths)
        except BaseException as rollback_fsync_error:
            rollback_errors.append(rollback_fsync_error)
            retained_backups.update(
                backup for backup in backups if backup is not None and exists_fn(backup)
            )
        cleanup_errors = _cleanup_backups(backups, retained_backups, unlink_fn, exists_fn)
        rollback_errors.extend(cleanup_errors)
        if rollback_errors:
            retained_backups.update(
                backup for backup in backups if backup is not None and exists_fn(backup)
            )
            retained = ", ".join(str(path) for path in sorted(retained_backups))
            raise RuntimeError(
                f"output publish failed ({publish_error}); rollback also failed: "
                f"{rollback_errors!r}; backup retained at: {retained}"
            ) from publish_error
        raise

    cleanup_errors = _cleanup_backups(backups, retained_backups, unlink_fn, exists_fn)
    if cleanup_errors:
        retained_backups.update(
            backup for backup in backups if backup is not None and exists_fn(backup)
        )
        retained = ", ".join(str(path) for path in sorted(retained_backups))
        raise RuntimeError(
            "output publish committed, but backup cleanup failed: "
            f"{cleanup_errors!r}; backup retained at: {retained}"
        ) from cleanup_errors[0]


def _validate_publish_paths(
    staged_paths: list[Path],
    target_paths: list[Path],
    directory_fds: DirectoryFds,
) -> list[tuple[int, int]]:
    """校验发布叶节点，并返回每个 staging 文件绑定的物理身份。"""
    if len(staged_paths) != len(target_paths) or not target_paths:
        raise ValueError("staged files and targets must have the same non-zero length")
    normalized_staged = [_absolute_path(path) for path in staged_paths]
    normalized_targets = [_absolute_path(path) for path in target_paths]
    if len(set(normalized_staged)) != len(normalized_staged):
        raise ValueError("staging files must be different")
    if set(normalized_staged) & set(normalized_targets):
        raise ValueError("staging files and targets must be different")

    _validate_target_leaves(normalized_targets, directory_fds)
    _validate_distinct_targets(normalized_targets, directory_fds)
    staged_stats = [
        _target_lstat(path, directory_fds) for path in normalized_staged
    ]
    target_stats = [
        _target_lstat(path, directory_fds) for path in normalized_targets
    ]
    for index, first_stat in enumerate(staged_stats):
        if first_stat is None:
            continue
        for second_stat in staged_stats[index + 1 :]:
            if second_stat is not None and os.path.samestat(first_stat, second_stat):
                raise ValueError(
                    "staging files must refer to different physical files"
                )
        for target_stat in target_stats:
            if target_stat is not None and os.path.samestat(first_stat, target_stat):
                raise ValueError(
                    "staging files and targets must refer to different physical files"
                )

    invalid_staged = [
        str(path)
        for path, path_stat in zip(normalized_staged, staged_stats)
        if path_stat is None
        or not stat.S_ISREG(path_stat.st_mode)
        or path_stat.st_nlink != 1
    ]
    if invalid_staged:
        raise RuntimeError(
            "transaction staging files must be regular files with exactly one hard link: "
            + ", ".join(invalid_staged)
        )
    return [(path_stat.st_dev, path_stat.st_ino) for path_stat in staged_stats]


def _verify_staging_identity(
    path: Path,
    expected: tuple[int, int],
    directory_fds: DirectoryFds,
) -> None:
    path_stat = _target_lstat(path, directory_fds)
    if (
        path_stat is None
        or not stat.S_ISREG(path_stat.st_mode)
        or path_stat.st_nlink != 1
        or (path_stat.st_dev, path_stat.st_ino) != expected
    ):
        raise RuntimeError(
            f"transaction staging file identity changed before publish: {path}"
        )


def _verify_staging_identities(
    paths: Iterable[Path],
    expected: Iterable[tuple[int, int]],
    directory_fds: DirectoryFds,
) -> None:
    for path, identity in zip(paths, expected):
        _verify_staging_identity(path, identity, directory_fds)


def _verify_published_identity(
    target: Path,
    expected: tuple[int, int],
    directory_fds: DirectoryFds,
) -> None:
    target_stat = _target_lstat(target, directory_fds)
    if target_stat is None or (target_stat.st_dev, target_stat.st_ino) != expected:
        raise RuntimeError(
            f"published target does not match bound staging identity: {target}"
        )


def _validate_distinct_targets(targets: list[Path], directory_fds: DirectoryFds) -> None:
    normalized = [_absolute_path(target) for target in targets]
    if len(set(normalized)) != len(normalized):
        raise ValueError("output files must be different")
    target_stats = [_target_lstat(target, directory_fds) for target in normalized]
    for index, first_stat in enumerate(target_stats):
        if first_stat is None:
            continue
        for second_stat in target_stats[index + 1 :]:
            if second_stat is not None and os.path.samestat(first_stat, second_stat):
                raise ValueError("output files must refer to different physical files")


def _validate_target_leaves(targets: Iterable[Path], directory_fds: DirectoryFds) -> None:
    target_stats = [
        (target, _target_lstat(target, directory_fds)) for target in targets
    ]
    symlinks = [
        str(target)
        for target, target_stat in target_stats
        if target_stat is not None and stat.S_ISLNK(target_stat.st_mode)
    ]
    if symlinks:
        raise RuntimeError(
            "transaction targets must not be symbolic links: " + ", ".join(symlinks)
        )
    special_nodes = [
        str(target)
        for target, target_stat in target_stats
        if target_stat is not None and not stat.S_ISREG(target_stat.st_mode)
    ]
    if special_nodes:
        raise RuntimeError(
            "transaction target is not a regular file: " + ", ".join(special_nodes)
        )


def _cleanup_backups(
    backups: Iterable[Path | None],
    retained_backups: set[Path],
    unlink_fn: Callable[[Path], None],
    exists_fn: Callable[[Path], bool] | None = None,
) -> list[BaseException]:
    exists = exists_fn or Path.exists
    errors: list[BaseException] = []
    for backup in backups:
        if backup is None or backup in retained_backups or not exists(backup):
            continue
        try:
            unlink_fn(backup)
        except BaseException as cleanup_error:
            errors.append(cleanup_error)
            if exists(backup):
                retained_backups.add(backup)
    return errors


@contextmanager
def _lock_parent_directories(
    paths: Iterable[Path],
    *,
    create_missing: bool = False,
) -> Iterator[DirectoryFds]:
    """拒绝 symlink/非目录 ancestor，并锁住稳定的父目录 FD。"""
    if fcntl is None:
        raise RuntimeError("transactional file publishing requires fcntl")
    directories = sorted(
        {_absolute_path(path).parent for path in paths}, key=os.fspath
    )
    directory_fds: DirectoryFds = {}
    locked: list[int] = []
    try:
        for directory in directories:
            descriptor = _open_directory(directory, create_missing=create_missing)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            except BaseException:
                os.close(descriptor)
                raise
            directory_fds[_absolute_path(directory)] = descriptor
            locked.append(descriptor)
        yield directory_fds
    finally:
        for descriptor in reversed(locked):
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _open_directory(directory: Path, *, create_missing: bool) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.sep, flags)
    try:
        for component in directory.parts[1:]:
            while True:
                try:
                    component_stat = os.lstat(component, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create_missing:
                        raise
                    try:
                        os.mkdir(component, 0o755, dir_fd=descriptor)
                    except FileExistsError:
                        continue
                    component_stat = os.lstat(component, dir_fd=descriptor)
                if stat.S_ISLNK(component_stat.st_mode):
                    raise RuntimeError(
                        f"transaction paths must not contain symbolic-link ancestors: {directory}"
                    )
                if not stat.S_ISDIR(component_stat.st_mode):
                    raise NotADirectoryError(
                        f"transaction path ancestor is not a directory: {directory}"
                    )
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
                break
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _absolute_path(path: Path) -> Path:
    """只做词法绝对化，不跟随任何 symlink。"""
    return Path(os.path.abspath(os.fspath(path)))


def _parent_key(path: Path) -> Path:
    return _absolute_path(path).parent


def _target_lstat(target: Path, directory_fds: DirectoryFds) -> os.stat_result | None:
    try:
        return os.lstat(target.name, dir_fd=directory_fds[_parent_key(target)])
    except FileNotFoundError:
        return None


def _target_exists(target: Path, directory_fds: DirectoryFds) -> bool:
    return _target_lstat(target, directory_fds) is not None


def _path_exists(path: Path, directory_fds: DirectoryFds) -> bool:
    return _target_lstat(path, directory_fds) is not None


def _candidate_names() -> Iterator[str]:
    """生成临时文件候选名；不依赖 stdlib 的私有 API。"""
    while True:
        yield f"tmp{os.urandom(8).hex()}"


def _create_temporary_file(
    directory_fd: int,
    directory: Path,
    *,
    prefix: str,
    suffix: str,
) -> tuple[int, Path]:
    candidates = _candidate_names()
    for _ in range(100):
        candidate = next(candidates)
        name = f"{prefix}{candidate}{suffix}"
        try:
            descriptor = os.open(
                name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            continue
        return descriptor, directory / name
    raise FileExistsError(f"could not create temporary file in {directory}")


def _replace_at(source: Path, target: Path, directory_fds: DirectoryFds) -> None:
    os.replace(
        source.name,
        target.name,
        src_dir_fd=directory_fds[_parent_key(source)],
        dst_dir_fd=directory_fds[_parent_key(target)],
    )


def _chmod_at(path: Path, mode: int, directory_fds: DirectoryFds) -> None:
    os.chmod(
        path.name,
        mode,
        dir_fd=directory_fds[_parent_key(path)],
        follow_symlinks=False,
    )


def _unlink_at(path: Path, directory_fds: DirectoryFds) -> None:
    try:
        os.unlink(path.name, dir_fd=directory_fds[_parent_key(path)])
    except FileNotFoundError:
        pass


def _backup_existing_target_at(target: Path, directory_fds: DirectoryFds) -> Path | None:
    target_fd = directory_fds[_parent_key(target)]
    target_stat = _target_lstat(target, directory_fds)
    if target_stat is None:
        return None
    if stat.S_ISLNK(target_stat.st_mode):
        raise RuntimeError(f"transaction targets must not be symbolic links: {target}")
    source_descriptor = os.open(
        target.name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=target_fd,
    )
    try:
        backup_descriptor, backup = _create_temporary_file(
            target_fd,
            target.parent,
            prefix=f".{target.name}.",
            suffix=".bak",
        )
    except BaseException:
        os.close(source_descriptor)
        raise
    try:
        with os.fdopen(source_descriptor, "rb") as source, os.fdopen(
            backup_descriptor, "wb"
        ) as backup_file:
            source_stat = os.fstat(source.fileno())
            if not stat.S_ISREG(source_stat.st_mode):
                raise RuntimeError(f"transaction target is not a regular file: {target}")
            os.fchmod(backup_file.fileno(), stat.S_IMODE(source_stat.st_mode))
            while chunk := source.read(1024 * 1024):
                backup_file.write(chunk)
            backup_file.flush()
            os.fsync(backup_file.fileno())
    except BaseException:
        _unlink_at(backup, directory_fds)
        raise
    return backup


def _fsync_parent_directories_at(
    targets: Iterable[Path], directory_fds: DirectoryFds
) -> None:
    if os.name == "nt":
        return
    for parent in {_parent_key(target) for target in targets}:
        os.fsync(directory_fds[parent])


def unlink_file_if_exists(path: str | Path) -> None:
    """通过独立稳定父目录 dirfd 删除事务残留；路径不存在时忽略。"""
    absolute_path = _absolute_path(Path(path))
    try:
        directory_fd = _open_directory(absolute_path.parent, create_missing=False)
    except FileNotFoundError:
        return
    try:
        try:
            os.unlink(absolute_path.name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
    finally:
        os.close(directory_fd)
