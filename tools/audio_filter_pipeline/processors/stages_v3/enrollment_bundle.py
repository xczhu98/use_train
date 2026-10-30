"""从已验证的 Step3 artifact 离线生成冻结 enrollment bundle。"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import math
import os
import stat
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ...config import validation_context_for_step
from ...utils.artifacts import (
    ArtifactIndex,
    PipelineStep,
    _atomic_rename_noreplace_between,
    is_production_bound_artifact_index,
    iter_artifact_rows,
    open_artifact_for_read,
)
from ...utils.path_utils import SafeOutputPath, open_safe_output_path
from .base import EMPTY_LINEAGE_DIGEST, ArtifactStage
from .select import STEP3_ROW_FIELDS
from .wespeaker_runtime import WeSpeakerEmbeddingRuntime, WeSpeakerRuntimeOptions

_BUNDLE_FIELDS = {"dataset_id", "speaker_scope", "speaker_id", "embedding", "provenance"}


def _require_production_step3_index(index: ArtifactIndex) -> None:
    if not isinstance(index, ArtifactIndex):
        raise TypeError("enrollment bundle requires an ArtifactIndex")
    if not is_production_bound_artifact_index(index):
        raise ValueError(
            "enrollment bundle requires a production-bound Step3 ArtifactIndex "
            "opened from current producer identity"
        )


@dataclass
class _PreparedTarget:
    raw: Path
    canonical: Path
    safe: SafeOutputPath


@dataclass
class _MarkerHandle:
    descriptor: int
    marker_path: Path
    identity: tuple[int, int]
    value: dict[str, Any]
    parent_target: _PreparedTarget


_MANIFEST_SCHEMA = "audio_filter_pipeline.enrollment_exclusion.v2"
_LINEAR_SUM_SELECTION_ALGORITHM = "mean_cosine_linear_sum.v1"
_MANIFEST_FIELDS = {"schema", "algorithm", "bundle_sha256", "step3", "provenance", "enrollments"}
_HEX = "0123456789abcdef"


def enrollment_exclusion_manifest_algorithm(manifest: Mapping[str, Any]) -> str:
    """验证当前 enrollment exclusion manifest，并返回选择算法身份。"""
    if not isinstance(manifest, Mapping):
        raise ValueError("enrollment exclusion manifest must be a mapping")
    if manifest.get("schema") != _MANIFEST_SCHEMA:
        raise ValueError("enrollment exclusion manifest schema is unsupported")
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("enrollment exclusion manifest v2 schema is invalid")
    if manifest["algorithm"] != _LINEAR_SUM_SELECTION_ALGORITHM:
        raise ValueError("enrollment exclusion manifest selection algorithm is invalid")
    return _LINEAR_SUM_SELECTION_ALGORITHM


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(value: Any, field: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(f"{field} must be lowercase SHA-256 hex")
    return value


def open_current_step3_for_enrollment(
    output_root: str,
    dataset_id: str,
    config: Any,
    stages: Mapping[PipelineStep, ArtifactStage],
) -> ArtifactIndex:
    """按当前 stage 配置身份和实际前代 artifact digest 逐级绑定 Step3。"""
    lineage_digest = EMPTY_LINEAGE_DIGEST
    current = None
    for step in (
        PipelineStep.STEP0,
        PipelineStep.STEP1,
        PipelineStep.STEP2,
        PipelineStep.STEP3,
    ):
        stage = stages.get(step)
        if not isinstance(stage, ArtifactStage) or stage.step is not step:
            raise ValueError(f"current producer identity is missing valid {step.value} stage")
        current = open_artifact_for_read(
            output_root,
            step,
            dataset_id,
            expected_config_digest=stage.config_digest(config),
            expected_lineage_digest=lineage_digest,
        )
        lineage_digest = current.artifact_digest
    assert current is not None
    return current


_TRANSACTION_SCHEMA = "audio_filter_pipeline.enrollment_bundle_transaction.v1"
_MARKER_PREFIX = ".enrollment_bundle_transaction-"
_MARKER_SUFFIX = ".json"
_MARKER_MAX_BYTES = 1024 * 1024


def _canonical_marker_path(value: Any, field: str) -> str:
    if type(value) is not str or not os.path.isabs(value):
        raise ValueError(f"{field} must be a canonical absolute path")
    canonical = os.path.realpath(os.path.abspath(value))
    if value != canonical:
        raise ValueError(f"{field} must be a canonical absolute path")
    return value


def _prepare_targets(
    bundle_path: str, exclusion_path: str,
) -> tuple[_PreparedTarget, _PreparedTarget]:
    """校验并绑定 enrollment target 的真实父目录。

    配置路径必须已经是 canonical absolute path；不接受通过父目录 symlink、
    ``..`` 或冗余路径组件将配置位置与实际发布位置分离。返回的
    ``_PreparedTarget`` 持有 descriptor-bound ``SafeOutputPath``，调用方取得
    所有权后必须调用 ``_close_prepared_targets``；构造中途失败会立即关闭
    已打开的全部目录 fd。
    """
    prepared: list[_PreparedTarget] = []
    try:
        for value, field in ((bundle_path, "bundle"), (exclusion_path, "exclusion")):
            try:
                raw_text = os.fspath(value)
                raw = Path(raw_text)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} path must be an absolute .jsonl path") from exc
            if type(raw_text) is not str or not raw.is_absolute() or raw.suffix != ".jsonl":
                raise ValueError("bundle and exclusion paths must be absolute .jsonl paths")
            try:
                raw_absolute = os.path.abspath(raw_text)
                canonical = Path(os.path.realpath(raw_absolute))
            except (OSError, ValueError) as exc:
                raise ValueError(f"{field} path must be a canonical absolute path") from exc
            if canonical.suffix != ".jsonl":
                raise ValueError("bundle and exclusion paths must be absolute .jsonl paths")
            if raw_text != str(canonical):
                raise ValueError(f"{field} path must be a canonical absolute path")
            safe_target = None
            try:
                safe_target = open_safe_output_path(
                    str(canonical.parent), str(canonical), create=True,
                )
                safe_target.target_stat()
            except BaseException as exc:
                if safe_target is not None:
                    try:
                        safe_target.close()
                    except OSError:
                        pass
                if isinstance(exc, ValueError):
                    raise
                if isinstance(exc, OSError):
                    raise ValueError(f"{field} target cannot be opened safely") from exc
                raise
            prepared.append(_PreparedTarget(raw, canonical, safe_target))
        if prepared[0].canonical == prepared[1].canonical:
            raise ValueError("bundle and exclusion paths must differ")
        return prepared[0], prepared[1]
    except BaseException:
        for target in reversed(prepared):
            try:
                target.safe.close()
            except OSError:
                pass
        raise


def _close_prepared_targets(targets: Sequence[_PreparedTarget]) -> None:
    first_error: OSError | None = None
    for target in reversed(tuple(targets)):
        try:
            target.safe.close()
        except OSError as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _transaction_marker_path(targets: Sequence[Path]) -> Path:
    ordered = tuple(sorted(str(target) for target in targets))
    digest = hashlib.sha256((ordered[0] + chr(0) + ordered[1]).encode("utf-8")).hexdigest()
    return Path(ordered[0]).parent / f"{_MARKER_PREFIX}{digest}{_MARKER_SUFFIX}"


def _strict_json_object(pairs: list[tuple[Any, Any]]) -> dict[Any, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("transaction marker contains duplicate fields")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"transaction marker contains non-finite JSON constant: {value}")


def _read_marker(descriptor: int) -> dict[str, Any]:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MARKER_MAX_BYTES:
                raise ValueError("transaction marker is too large")
            chunks.append(chunk)
        value = json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("transaction marker is unreadable") from exc
    except ValueError:
        raise
    if not isinstance(value, dict):
        raise ValueError("transaction marker root must be an object")
    return value


def _stat_at(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"cannot inspect transaction path safely: {name}") from exc


def _link_target(
    source_fd: int, source_name: str, target_fd: int, target_name: str,
) -> None:
    """在固定源/目标目录 fd 间创建不覆盖 hard link。"""
    for value, field in ((source_name, "source_name"), (target_name, "target_name")):
        if (
            type(value) is not str
            or not value
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
            or "\x00" in value
        ):
            raise ValueError(f"{field} must be a single basename")
    os.link(
        source_name,
        target_name,
        src_dir_fd=source_fd,
        dst_dir_fd=target_fd,
        follow_symlinks=False,
    )


def _snapshot_file_at(
    parent_fd: int,
    name: str,
    expected_identity: tuple[int, int] | None = None,
    expected_hash: str | None = None,
) -> tuple[tuple[int, int], str]:
    """在固定父目录 fd 下读取并校验 regular file，避免 FIFO 阻塞。"""
    before = _stat_at(parent_fd, name)
    if before is None:
        raise ValueError(f"file disappeared while validating transaction entry: {name}")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"transaction entry is not a regular file: {name}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError as exc:
        raise ValueError(f"file disappeared while validating transaction entry: {name}") from exc
    except OSError as exc:
        raise ValueError(f"cannot safely open transaction entry: {name}") from exc
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError(f"transaction entry is not a regular file: {name}")
        identity = (initial.st_dev, initial.st_ino)
        if expected_identity is not None and identity != expected_identity:
            raise ValueError(f"transaction entry identity changed: {name}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        current = _stat_at(parent_fd, name)
        if current is None or not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != identity:
            raise ValueError(f"transaction entry changed while hashing: {name}")
        actual_hash = digest.hexdigest()
        if expected_hash is not None and actual_hash != expected_hash:
            raise ValueError(f"transaction entry content hash changed: {name}")
        return identity, actual_hash
    except OSError as exc:
        raise ValueError(f"cannot safely read transaction entry: {name}") from exc
    finally:
        os.close(descriptor)


def _open_quarantine_directory(
    parent_fd: int, path_name: str,
) -> tuple[str, int, tuple[int, int]]:
    flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    for _ in range(128):
        quarantine_name = f".{path_name}.cleanup-{uuid.uuid4().hex}"
        try:
            os.mkdir(quarantine_name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        descriptor = None
        try:
            descriptor = os.open(quarantine_name, flags, dir_fd=parent_fd)
            current = os.fstat(descriptor)
            if not stat.S_ISDIR(current.st_mode):
                raise ValueError("transaction quarantine is not a directory")
            if os.listdir(descriptor):
                raise ValueError("transaction quarantine is not empty")
            return quarantine_name, descriptor, (current.st_dev, current.st_ino)
        except BaseException:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            try:
                os.rmdir(quarantine_name, dir_fd=parent_fd)
            except OSError:
                pass
            raise
    raise OSError("cannot allocate transaction quarantine directory")


def _restore_quarantined(
    parent_fd: int, quarantine_fd: int, quarantine_name: str, original_name: str,
) -> bool:
    try:
        _atomic_rename_noreplace_between(
            quarantine_fd, "entry", parent_fd, original_name,
        )
    except (FileExistsError, OSError):
        return False
    try:
        return os.stat("entry", dir_fd=quarantine_fd, follow_symlinks=False)
    except FileNotFoundError:
        try:
            restored = _stat_at(parent_fd, original_name)
        except ValueError:
            return False
        return restored is not None
    except OSError:
        return False


def _cleanup_quarantine_directory(
    parent_fd: int,
    quarantine_fd: int,
    quarantine_name: str,
    quarantine_identity: tuple[int, int],
) -> bool:
    try:
        if os.listdir(quarantine_fd):
            return False
        current = _stat_at(parent_fd, quarantine_name)
    except (OSError, ValueError):
        return False
    if current is None:
        return True
    if (current.st_dev, current.st_ino) != quarantine_identity:
        return False
    try:
        os.rmdir(quarantine_name, dir_fd=parent_fd)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    try:
        return _stat_at(parent_fd, quarantine_name) is None
    except ValueError:
        return False


def _unlink_matching_at(
    parent_fd: int,
    name: str,
    expected_identity: tuple[int, int],
    expected_hash: str | None,
) -> bool:
    """在固定父目录 fd 下隔离、复验并删除匹配文件。"""
    quarantine_fd = None
    quarantine_name = None
    quarantine_identity = None
    quarantine_removed = False
    try:
        current = _stat_at(parent_fd, name)
        if current is None:
            return True
        if not stat.S_ISREG(current.st_mode):
            return False
        try:
            _snapshot_file_at(parent_fd, name, expected_identity, expected_hash)
        except ValueError:
            try:
                return _stat_at(parent_fd, name) is None
            except ValueError:
                return False
        quarantine_name, quarantine_fd, quarantine_identity = _open_quarantine_directory(
            parent_fd, name,
        )
        try:
            _atomic_rename_noreplace_between(
                parent_fd, name, quarantine_fd, "entry",
            )
        except FileNotFoundError:
            return False
        except OSError:
            return False
        try:
            _snapshot_file_at(
                quarantine_fd, "entry", expected_identity, expected_hash,
            )
        except ValueError:
            restored = _restore_quarantined(
                parent_fd, quarantine_fd, quarantine_name, name,
            )
            if not restored:
                return False
            quarantine_removed = _cleanup_quarantine_directory(
                parent_fd, quarantine_fd, quarantine_name, quarantine_identity,
            )
            return quarantine_removed
        try:
            os.unlink("entry", dir_fd=quarantine_fd)
        except (FileNotFoundError, OSError):
            _restore_quarantined(
                parent_fd, quarantine_fd, quarantine_name, name,
            )
            return False
        quarantine_removed = _cleanup_quarantine_directory(
            parent_fd, quarantine_fd, quarantine_name, quarantine_identity,
        )
        return quarantine_removed
    except (OSError, ValueError):
        return False
    finally:
        if quarantine_fd is not None:
            if (
                not quarantine_removed
                and quarantine_name is not None
                and quarantine_identity is not None
            ):
                _cleanup_quarantine_directory(
                    parent_fd, quarantine_fd, quarantine_name, quarantine_identity,
                )
            try:
                os.close(quarantine_fd)
            except OSError:
                pass


def _validate_marker(
    value: Mapping[str, Any], expected_targets: Sequence[Path],
) -> tuple[str, tuple[tuple[Path, Path, tuple[int, int], str], ...]]:
    if set(value) != {"schema", "token", "targets"}:
        raise ValueError("transaction marker fields are not exact")
    if value["schema"] != _TRANSACTION_SCHEMA:
        raise ValueError("transaction marker schema is invalid")
    token = value["token"]
    if type(token) is not str or len(token) != 32 or any(char not in _HEX for char in token):
        raise ValueError("transaction marker token is invalid")
    marker_targets = value["targets"]
    if type(marker_targets) is not list or len(marker_targets) != 2:
        raise ValueError("transaction marker targets are invalid")
    entries = []
    for index, (entry, expected_target) in enumerate(zip(marker_targets, expected_targets)):
        if not isinstance(entry, Mapping) or set(entry) != {
            "target_path", "temp_path", "temp_dev", "temp_ino", "content_sha256",
        }:
            raise ValueError(f"transaction marker target {index} fields are not exact")
        target_path = _canonical_marker_path(entry["target_path"], f"marker target {index} path")
        if target_path != str(expected_target) or Path(target_path).suffix != ".jsonl":
            raise ValueError("transaction marker target paths do not match")
        temp_path = _canonical_marker_path(entry["temp_path"], f"marker temp {index} path")
        if Path(temp_path).parent != expected_target.parent:
            raise ValueError("transaction marker temp parent does not match target")
        prefix = f".{expected_target.name}.{token}."
        if not Path(temp_path).name.startswith(prefix):
            raise ValueError("transaction marker temp path is not token-bound")
        temp_dev, temp_ino = entry["temp_dev"], entry["temp_ino"]
        if type(temp_dev) is not int or type(temp_ino) is not int or temp_dev < 0 or temp_ino <= 0:
            raise ValueError("transaction marker temp identity is invalid")
        temporary = Path(temp_path)
        if temporary == expected_target:
            raise ValueError("transaction marker temp path equals target path")
        if any(existing[1] == temporary for existing in entries):
            raise ValueError("transaction marker temp paths are not distinct")
        content_sha256 = _sha256(entry["content_sha256"], f"marker target {index} content_sha256")
        entries.append((expected_target, temporary, (temp_dev, temp_ino), content_sha256))
    return token, tuple(entries)


def _open_existing_marker(
    marker_path: Path, parent_target: _PreparedTarget,
) -> _MarkerHandle | None:
    """在已绑定的 target 父目录 fd 下打开并锁定事务 marker。"""
    parent_target.safe.verify()
    parent_fd = parent_target.safe.parent_fd
    marker_name = marker_path.name
    before = _stat_at(parent_fd, marker_name)
    if before is None:
        return None
    if stat.S_ISLNK(before.st_mode):
        raise ValueError("transaction marker must not be a symlink")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("transaction marker must be a regular file")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(marker_name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError("transaction marker cannot be opened safely") from exc
    try:
        current = os.fstat(descriptor)
        identity = (current.st_dev, current.st_ino)
        if not stat.S_ISREG(current.st_mode) or identity != (before.st_dev, before.st_ino):
            raise ValueError("transaction marker was replaced while opening")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise FileExistsError("enrollment bundle transaction is active") from exc
            raise ValueError("transaction marker cannot be locked") from exc
        current = _stat_at(parent_fd, marker_name)
        if current is None or (current.st_dev, current.st_ino) != identity:
            raise ValueError("transaction marker was replaced while locking")
        return _MarkerHandle(
            descriptor, marker_path, identity, _read_marker(descriptor), parent_target,
        )
    except BaseException:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)
        raise


def _close_marker(handle: _MarkerHandle | None) -> None:
    if handle is None:
        return
    try:
        fcntl.flock(handle.descriptor, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(handle.descriptor)
    except OSError:
        pass


def _remove_marker(handle: _MarkerHandle, strict: bool) -> bool:
    """在 marker 所属父目录 fd 下隔离并删除原 inode，避免路径重解析竞态。"""
    try:
        if _read_marker(handle.descriptor) != handle.value:
            raise ValueError("transaction marker content changed before removal")
        handle.parent_target.safe.verify()
        parent_fd = handle.parent_target.safe.parent_fd
        marker_name = handle.marker_path.name
        current = _stat_at(parent_fd, marker_name)
        if current is None:
            return True
        if (stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode)
                or (current.st_dev, current.st_ino) != handle.identity):
            raise ValueError("transaction marker changed before removal")
        if not _unlink_matching_at(parent_fd, marker_name, handle.identity, None):
            raise ValueError("transaction marker could not be removed safely")
        handle.parent_target.safe.fsync_parent()
        return True
    except (OSError, ValueError):
        if strict:
            raise
        return False


def _create_marker(
    marker_path: Path,
    value: Mapping[str, Any],
    parent_target: _PreparedTarget,
) -> _MarkerHandle:
    flags = (
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = None
    identity = None
    parent_fd = None
    marker_name = marker_path.name
    try:
        parent_target.safe.verify()
        parent_fd = parent_target.safe.parent_fd
        descriptor = os.open(marker_name, flags, 0o600, dir_fd=parent_fd)
        created = os.fstat(descriptor)
        if not stat.S_ISREG(created.st_mode):
            raise ValueError("created transaction marker is not a regular file")
        identity = (created.st_dev, created.st_ino)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        payload = _canonical_bytes(value) + b"\n"
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
        os.fsync(descriptor)
        parent_target.safe.fsync_parent()
        return _MarkerHandle(
            descriptor, marker_path, identity, dict(value), parent_target,
        )
    except FileExistsError as exc:
        if parent_fd is not None:
            current = _stat_at(parent_fd, marker_name)
            if current is not None and stat.S_ISLNK(current.st_mode):
                raise ValueError("transaction marker must not be a symlink") from exc
        raise
    except BaseException:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(descriptor)
            except OSError:
                pass
        if parent_fd is not None and identity is not None:
            _unlink_matching_at(parent_fd, marker_name, identity, None)
            try:
                parent_target.safe.fsync_parent()
            except (OSError, ValueError):
                pass
        raise


def _recover_pending_transaction(
    marker_path: Path,
    expected_targets: Sequence[Path],
    prepared_targets: Sequence[_PreparedTarget],
) -> None:
    targets_by_path = {str(target.canonical): target for target in prepared_targets}
    marker_parent = targets_by_path[str(expected_targets[0])]
    handle = _open_existing_marker(marker_path, marker_parent)
    if handle is None:
        return
    try:
        _token, entries = _validate_marker(handle.value, expected_targets)
        for target, temporary, identity, content_sha256 in entries:
            prepared = targets_by_path.get(str(target))
            if prepared is None:
                raise ValueError("transaction recovery target is not prepared")
            prepared.safe.verify()
            parent_fd = prepared.safe.parent_fd
            temporary_stat = _stat_at(parent_fd, temporary.name)
            target_stat = prepared.safe.target_stat()
            if temporary_stat is None and target_stat is None:
                raise ValueError("transaction recovery lost both owned entries")
            if temporary_stat is not None:
                _snapshot_file_at(parent_fd, temporary.name, identity, content_sha256)
            if target_stat is not None:
                _snapshot_file_at(
                    parent_fd, prepared.safe.target_name, identity, content_sha256,
                )
        cleanup_ok = True
        for target, _temporary, identity, content_sha256 in entries:
            prepared = targets_by_path[str(target)]
            cleanup_ok = _unlink_matching_at(
                prepared.safe.parent_fd,
                prepared.safe.target_name,
                identity,
                content_sha256,
            ) and cleanup_ok
        for target, temporary, identity, content_sha256 in entries:
            prepared = targets_by_path[str(target)]
            cleanup_ok = _unlink_matching_at(
                prepared.safe.parent_fd,
                temporary.name,
                identity,
                content_sha256,
            ) and cleanup_ok
        if not cleanup_ok:
            raise ValueError("transaction recovery found a changed owned file")
        if not _remove_marker(handle, strict=True):
            raise ValueError("transaction marker could not be removed")
        for prepared in prepared_targets:
            prepared.safe.fsync_parent()
    finally:
        _close_marker(handle)


def _rollback_transaction(
    handle: _MarkerHandle | None,
    published: Sequence[tuple[_PreparedTarget, tuple[int, int], str]],
    temporary_records: Sequence[
        tuple[_PreparedTarget, str, tuple[int, int], str | None]
    ],
) -> None:
    cleanup_ok = True
    for target, identity, content_sha256 in reversed(published):
        cleanup_ok = _unlink_matching_at(
            target.safe.parent_fd,
            target.safe.target_name,
            identity,
            content_sha256,
        ) and cleanup_ok
    for target, temporary_name, identity, content_sha256 in temporary_records:
        cleanup_ok = _unlink_matching_at(
            target.safe.parent_fd,
            temporary_name,
            identity,
            content_sha256,
        ) and cleanup_ok
    if handle is not None and cleanup_ok:
        _remove_marker(handle, strict=False)


def _identity(row: Mapping[str, Any], dataset_id: str) -> tuple[str, str, str] | None:
    """仅接受与已发布 Step3 dataset 精确绑定的显式 speaker identity。"""
    if row.get("dataset_id") != dataset_id:
        raise ValueError("Step3 row dataset_id does not match requested dataset")
    speaker_id, speaker_scope = row.get("speaker_id"), row.get("speaker_scope")
    if speaker_id is None and speaker_scope is None:
        return None
    if (type(speaker_id) is not str or not speaker_id.strip() or speaker_id != speaker_id.strip()
            or type(speaker_scope) is not str or speaker_scope != dataset_id):
        raise ValueError("Step3 row speaker identity is invalid")
    return dataset_id, speaker_scope, speaker_id


def _candidate_rows(rows: Iterable[Mapping[str, Any]], dataset_id: str) -> dict[tuple[str, str, str], list[Mapping[str, Any]]]:
    candidates: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    seen_segments: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Step3 row must be a mapping")
        if set(row) != set(STEP3_ROW_FIELDS):
            raise ValueError("Step3 row fields do not match the exact current contract")
        for field in ("segment_id", "source_key"):
            value = row[field]
            if type(value) is not str or not value.strip() or value != value.strip():
                raise ValueError(f"Step3 row {field} must be a nonblank exact string")
        segment_id = row["segment_id"]
        if segment_id in seen_segments:
            raise ValueError("Step3 artifact contains duplicate segment_id")
        seen_segments.add(segment_id)
        _sha256(row["source_sha256"], "Step3 row source_sha256")
        for field in ("source_size_bytes", "sample_rate", "frames", "channels", "start_sample", "end_sample"):
            if type(row[field]) is not int:
                raise ValueError(f"Step3 row {field} must be an exact integer")
        if (row["source_size_bytes"] < 0 or row["sample_rate"] <= 0 or row["frames"] <= 0
                or row["channels"] <= 0
                or not 0 <= row["start_sample"] < row["end_sample"] <= row["frames"]):
            raise ValueError("Step3 row source descriptor or segment bounds are invalid")
        identity = _identity(row, dataset_id)
        if identity is not None:
            candidates[identity].append(row)
    return {key: sorted(value, key=lambda row: row["segment_id"]) for key, value in sorted(candidates.items())}


def _normalize(values: Sequence[Real]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("WeSpeaker embedding must be a numeric sequence")
    vector = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError("WeSpeaker embedding must contain finite non-bool values")
        vector.append(float(value))
    if not vector:
        raise ValueError("WeSpeaker embedding must be nonzero")
    scale = max(abs(value) for value in vector)
    if scale == 0.0:
        raise ValueError("WeSpeaker embedding must be nonzero")
    scaled = tuple(value / scale for value in vector)
    norm = math.sqrt(math.fsum(value * value for value in scaled))
    return tuple(value / norm for value in scaled)


def _choose_enrollment(rows: Sequence[Mapping[str, Any]], embeddings: Sequence[Sequence[Real]], top_k: int) -> tuple[tuple[float, ...], tuple[str, ...], dict[str, float]]:
    """按平均 pairwise cosine 选稳定 top-K，再以相同规则选确定性 medoid。"""
    if len(rows) != len(embeddings) or not rows:
        raise ValueError("candidate rows and embeddings must be nonempty and aligned")
    vectors = tuple(_normalize(value) for value in embeddings)
    if len({len(vector) for vector in vectors}) != 1:
        raise ValueError("WeSpeaker embeddings for one speaker must share a dimension")
    if len(vectors) == 1:
        scores = [1.0]
    else:
        vector_sum = tuple(math.fsum(vector[dimension] for vector in vectors) for dimension in range(len(vectors[0])))
        scores = [
            (
                math.fsum(left * right for left, right in zip(vector, vector_sum))
                - math.fsum(value * value for value in vector)
            ) / (len(vectors) - 1)
            for vector in vectors
        ]
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("WeSpeaker enrollment stability scores must be finite")
    selected = sorted(range(len(rows)), key=lambda index: (-scores[index], rows[index]["segment_id"]))[:min(top_k, len(rows))]
    def medoid_score(index: int) -> float:
        similarities = [
            math.fsum(left * right for left, right in zip(vectors[index], vectors[other]))
            for other in selected if other != index
        ]
        return math.fsum(similarities) / len(similarities) if similarities else 1.0
    medoid = min(selected, key=lambda index: (-medoid_score(index), rows[index]["segment_id"]))
    return vectors[medoid], tuple(rows[index]["segment_id"] for index in selected), {rows[index]["segment_id"]: scores[index] for index in range(len(rows))}


def build_enrollment_bundle(index: ArtifactIndex, dataset_id: str, runtime: WeSpeakerEmbeddingRuntime, top_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """读取已验证 Step3 分数 artifact，返回 bundle 行与 SHA 绑定 exclusion manifest。

    enrollment 基于 Step3 全量 segment（与 Step4 筛选解耦）：改 MOS 阈值
    不影响 enrollment 绑定。
    """
    _require_production_step3_index(index)
    if index.step != PipelineStep.STEP3.value or index.dataset_id != dataset_id:
        raise ValueError("enrollment bundle requires the requested Step3 artifact")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    candidates = _candidate_rows(iter_artifact_rows(index), dataset_id)
    if not candidates:
        raise ValueError("verified Step3 artifact contains no enrollment candidates")
    provenance = dict(runtime.provenance)
    expected = {field: provenance[field] for field in ("runtime", "model_version", "model_sha256", "runtime_version", "target_sample_rate")}
    bundle, enrollment_groups = [], []
    for identity, rows in candidates.items():
        embeddings = runtime.extract_batch(rows)
        embedding, _selected_segment_ids, stability = _choose_enrollment(rows, embeddings, top_k)
        bundle.append({"dataset_id": identity[0], "speaker_scope": identity[1], "speaker_id": identity[2], "embedding": list(embedding), "provenance": expected})
        enrollment_groups.append({"dataset_id": identity[0], "speaker_scope": identity[1], "speaker_id": identity[2], "segment_ids": [row["segment_id"] for row in rows], "candidate_count": len(rows), "stability": stability})
    bundle_bytes = b"".join(_canonical_bytes(row) + b"\n" for row in bundle)
    bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
    manifest = {"schema": _MANIFEST_SCHEMA, "algorithm": _LINEAR_SUM_SELECTION_ALGORITHM, "bundle_sha256": bundle_sha256, "step3": {"dataset_id": dataset_id, "config_digest": index.config_digest, "lineage_digest": index.lineage_digest, "artifact_digest": index.artifact_digest}, "provenance": expected, "enrollments": enrollment_groups}
    return bundle, manifest


def _write_enrollment_bundle_prepared(
    bundle: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    prepared: tuple[_PreparedTarget, _PreparedTarget],
) -> tuple[str, str]:
    bundle_target, exclusion_target = prepared
    marker_targets = tuple(sorted(
        (bundle_target.canonical, exclusion_target.canonical),
        key=lambda path: str(path),
    ))
    _recover_pending_transaction(marker_path := _transaction_marker_path(marker_targets), marker_targets, prepared)

    payload = b"".join(_canonical_bytes(dict(row)) + b"\n" for row in bundle)
    if not payload:
        raise ValueError("enrollment bundle must be nonempty")
    bundle_sha256 = hashlib.sha256(payload).hexdigest()
    if manifest.get("bundle_sha256") != bundle_sha256:
        raise ValueError("exclusion manifest bundle SHA-256 binding is invalid")
    exclusion_payload = _canonical_bytes(dict(manifest)) + b"\n"
    targets = ((bundle_target, payload), (exclusion_target, exclusion_payload))
    for target, _content in targets:
        if target.safe.target_stat() is not None:
            raise FileExistsError(
                f"refusing to overwrite existing enrollment output: {target.canonical}"
            )

    token = uuid.uuid4().hex
    temporary_records: list[
        tuple[_PreparedTarget, str, tuple[int, int], str | None]
    ] = []
    records_by_target: dict[Path, tuple[str, tuple[int, int], str]] = {}
    published: list[tuple[_PreparedTarget, tuple[int, int], str]] = []
    marker_handle: _MarkerHandle | None = None
    try:
        for target, content in targets:
            descriptor, temporary_name = target.safe.create_temp_file(
                prefix=f".{target.canonical.name}.{token}.",
            )
            temporary_stat = os.fstat(descriptor)
            temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
            temporary_records.append((target, temporary_name, temporary_identity, None))
            try:
                stream = os.fdopen(descriptor, "wb")
            except BaseException:
                os.close(descriptor)
                raise
            try:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            finally:
                stream.close()
            content_sha256 = hashlib.sha256(content).hexdigest()
            target.safe.verify()
            _snapshot_file_at(
                target.safe.parent_fd,
                temporary_name,
                temporary_identity,
                content_sha256,
            )
            temporary_records[-1] = (target, temporary_name, temporary_identity, content_sha256)
            records_by_target[target.canonical] = (
                temporary_name, temporary_identity, content_sha256,
            )

        marker_value = {
            "schema": _TRANSACTION_SCHEMA,
            "token": token,
            "targets": [
                {
                    "target_path": str(target),
                    "temp_path": str(
                        Path(marker_targets[index]).parent / records_by_target[target][0]
                    ),
                    "temp_dev": records_by_target[target][1][0],
                    "temp_ino": records_by_target[target][1][1],
                    "content_sha256": records_by_target[target][2],
                }
                for index, target in enumerate(marker_targets)
            ],
        }
        target_by_path = {target.canonical: target for target, _content in targets}
        marker_handle = _create_marker(
            marker_path, marker_value, target_by_path[marker_targets[0]],
        )

        for target_path in marker_targets:
            target = target_by_path[target_path]
            temporary_name, temporary_identity, content_sha256 = records_by_target[target_path]
            target.safe.verify()
            _snapshot_file_at(
                target.safe.parent_fd,
                temporary_name,
                temporary_identity,
                content_sha256,
            )
            try:
                _link_target(
                    target.safe.parent_fd,
                    temporary_name,
                    target.safe.parent_fd,
                    target.safe.target_name,
                )
            except BaseException:
                current = target.safe.target_stat()
                if current is not None and (current.st_dev, current.st_ino) == temporary_identity:
                    published.append((target, temporary_identity, content_sha256))
                raise
            published.append((target, temporary_identity, content_sha256))
            _snapshot_file_at(
                target.safe.parent_fd,
                target.safe.target_name,
                temporary_identity,
                content_sha256,
            )
            target.safe.fsync_parent()

        cleanup_ok = True
        for target, _content in targets:
            temporary_name, temporary_identity, content_sha256 = records_by_target[target.canonical]
            cleanup_ok = _unlink_matching_at(
                target.safe.parent_fd,
                temporary_name,
                temporary_identity,
                content_sha256,
            ) and cleanup_ok
        if not cleanup_ok:
            raise ValueError("transaction temporary cleanup found a changed owned file")
        _remove_marker(marker_handle, strict=True)
    except Exception:
        _rollback_transaction(marker_handle, published, temporary_records)
        raise
    except BaseException:
        # marker 已落盘时保留现场，下一次调用负责按 marker 恢复；此前只清理本次 temp。
        if marker_handle is None:
            _rollback_transaction(None, published, temporary_records)
        raise
    finally:
        _close_marker(marker_handle)
    return bundle_sha256, hashlib.sha256(exclusion_payload).hexdigest()


def write_enrollment_bundle(
    bundle: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    bundle_path: str,
    exclusion_path: str,
) -> tuple[str, str]:
    """写出 canonical JSONL bundle 与绑定 exclusion manifest，并返回两个摘要。"""
    prepared = _prepare_targets(bundle_path, exclusion_path)
    try:
        return _write_enrollment_bundle_prepared(bundle, manifest, prepared)
    finally:
        _close_prepared_targets(prepared)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从当前配置绑定的 Step3 artifact 生成冻结 WeSpeaker enrollment bundle")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--runtime-version", required=True)
    parser.add_argument("--target-sample-rate", required=True, type=int)
    parser.add_argument("--worker-script", required=True)
    parser.add_argument("--worker-script-sha256", required=True)
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--bundle-path", required=True)
    parser.add_argument("--exclusion-path", required=True)
    return parser


def _current_step3_from_config(config_path: str, dataset_id: str) -> ArtifactIndex:
    """加载权威配置并重建当前 Step0–Step3 producer identity。"""
    from ...cli.main import load_config
    from . import cli as v3_cli

    canonical_config = os.path.realpath(os.path.abspath(config_path))
    if config_path != canonical_config or not os.path.isfile(canonical_config):
        raise ValueError("config must be a canonical absolute existing file")
    raw_config = load_config(canonical_config)
    config_args = SimpleNamespace(
        output_root=None,
        output=None,
        step0_workers=None,
        step3_gpu_ids=None,
        use_gpu=None,
        no_use_gpu=False,
    )
    config = v3_cli._build_config(config_args, raw_config, PipelineStep.STEP3)
    config.validate(
        context=validation_context_for_step(PipelineStep.STEP3.value, 3),
        stage_names=[],
    )
    use_gpu = v3_cli._use_gpu(config_args, raw_config)
    plan = v3_cli.MosExecutionPlan.from_config(config, use_gpu=use_gpu)
    verified = v3_cli.verify_mos_assets(plan)
    factories = v3_cli._factories(
        config_args,
        raw_config,
        config,
        PipelineStep.STEP3,
        plan,
        verified,
        config_base_dir=os.path.dirname(canonical_config),
    )
    stages = {
        step: factories[step](config, dataset_id)
        for step in (
            PipelineStep.STEP0,
            PipelineStep.STEP1,
            PipelineStep.STEP2,
            PipelineStep.STEP3,
        )
    }
    return open_current_step3_for_enrollment(
        config.output_root, dataset_id, config, stages,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    index = _current_step3_from_config(args.config, args.dataset_id)
    runtime = WeSpeakerEmbeddingRuntime(WeSpeakerRuntimeOptions(model_path=args.model_path, model_version=args.model_version, model_sha256=args.model_sha256, runtime_version=args.runtime_version, target_sample_rate=args.target_sample_rate, worker_script=args.worker_script, worker_script_sha256=args.worker_script_sha256))
    bundle, manifest = build_enrollment_bundle(index, args.dataset_id, runtime, args.top_k)
    bundle_sha256, exclusion_sha256 = write_enrollment_bundle(bundle, manifest, args.bundle_path, args.exclusion_path)
    print(json.dumps({"bundle_path": args.bundle_path, "bundle_sha256": bundle_sha256, "exclusion_path": args.exclusion_path, "exclusion_sha256": exclusion_sha256}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
