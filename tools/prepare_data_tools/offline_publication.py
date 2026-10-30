"""GenerationManifest schema v2 multi-root owned directory publication transaction。

所有路径只能是其 allowed parent 直接子目录；不使用 ``rm -rf`` 或
``shutil.rmtree``。旧单 root publication 已下线并 fail-closed。
"""

from __future__ import annotations

import json
import os
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from tools.prepare_data_tools.generation_manifest import (
    GENERATION_ROOT_MARKER,
    MANIFEST_NAME,
    GenerationManifest,
    GenerationManifestError,
    bind_generation_roots,
    generation_root_resource_digest,
    load_generation_manifest,
    sha256_file,
    validate_generation_root_marker,
)
from tools.prepare_data_tools.offline_reload_gate import (
    OfflineReloadError,
    ReloadReport,
    require_publish_eligible,
    run_offline_reload_gates,
)

OWNER_FILE = ".offline-publication-owner.json"
OWNER_SCHEMA_VERSION = 1
RECEIPT_FILE = "generation_gate_receipt.json"
RECEIPT_SCHEMA_VERSION = 2


class OfflinePublicationError(RuntimeError):
    """发布未提交，或已完成安全回滚。"""


class PublicationCommittedError(OfflinePublicationError):
    """新 target 已带 receipt 提交，但旧 backup 清理失败。"""


@dataclass(frozen=True)
class OwnedPublicationPaths:
    allowed_parent: Path
    staging: Path
    target: Path
    backup: Path
    owner_id: str


@dataclass(frozen=True)
class MultiRootPublicationTarget:
    alias: str
    paths: OwnedPublicationPaths


@dataclass(frozen=True)
class ExistingOwnedRoot:
    alias: str
    allowed_parent: Path
    root: Path
    owner_id: str


@dataclass(frozen=True)
class MultiRootPublicationReceipt:
    schema_version: int
    status: str
    owner_id: str
    generation_id: str
    manifest_sha256: str
    root_digests: Mapping[str, str]
    completed_at: str
    reload_report: ReloadReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "owner_id": self.owner_id,
            "generation_id": self.generation_id,
            "manifest_sha256": self.manifest_sha256,
            "root_digests": dict(self.root_digests),
            "completed_at": self.completed_at,
            "reload_report": self.reload_report.to_dict(),
        }


def write_ownership_marker(root: str | Path, *, owner_id: str) -> Path:
    """为调用者刚创建的空/owned 目录写入精确 ownership marker。"""
    owner_id = _validate_owner_id(owner_id)
    candidate = Path(root)
    root_stat = _ordinary_directory_stat(candidate, location="ownership root")
    marker = candidate / OWNER_FILE
    if marker.exists() or marker.is_symlink():
        raise OfflinePublicationError(f"ownership marker 已存在，拒绝覆盖: {marker}")
    if any(candidate.iterdir()):
        raise OfflinePublicationError(
            f"ownership marker 只能写入调用者新建的空目录，拒绝接管已有资产: {candidate}"
        )
    payload = {
        "schema_version": OWNER_SCHEMA_VERSION,
        "owner_id": owner_id,
    }
    descriptor = os.open(
        marker,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (candidate.lstat().st_dev, candidate.lstat().st_ino) != (
        root_stat.st_dev,
        root_stat.st_ino,
    ):
        raise OfflinePublicationError(f"ownership root 在写 marker 时被替换: {candidate}")
    return marker


def publish_owned_offline_dataset_roots(
    targets: Sequence[MultiRootPublicationTarget],
    *,
    manifest_root_alias: str,
    existing_roots: Sequence[ExistingOwnedRoot] = (),
    manifest_path: str | Path = MANIFEST_NAME,
) -> MultiRootPublicationReceipt:
    """联合发布多个 owned roots；所有 rename 前完成全量预检。"""
    if not targets:
        raise OfflinePublicationError("multi-root transaction 至少需要一个 target")
    ordered_targets = sorted(targets, key=lambda item: item.alias)
    ordered_existing = sorted(existing_roots, key=lambda item: item.alias)
    aliases = [item.alias for item in ordered_targets] + [
        item.alias for item in ordered_existing
    ]
    if len(set(aliases)) != len(aliases):
        raise OfflinePublicationError(f"multi-root alias 重复: {aliases}")
    target_by_alias = {item.alias: item for item in ordered_targets}
    if manifest_root_alias not in target_by_alias:
        raise OfflinePublicationError("manifest_root_alias 必须是本事务的 target")
    owner_ids = {
        item.paths.owner_id for item in ordered_targets
    } | {item.owner_id for item in ordered_existing}
    if len(owner_ids) != 1:
        raise OfflinePublicationError("所有 roots 必须使用同一 owner_id")
    owner_id = _validate_owner_id(next(iter(owner_ids)))

    normalized: dict[str, tuple[Path, Path, Path, Path]] = {}
    had_target: dict[str, bool] = {}
    old_generation_ids: set[str] = set()
    for item in ordered_targets:
        parent, staging, target, backup = _validate_publication_paths(item.paths)
        normalized[item.alias] = (parent, staging, target, backup)
        _require_owned_directory(staging, owner_id, location=f"staging {item.alias!r}")
        _preflight_declared_owned_tree(
            staging, allowed_parent=parent, owner_id=owner_id
        )
        if backup.exists() or backup.is_symlink():
            raise OfflinePublicationError(
                f"backup 必须在事务开始时不存在: {backup}"
            )
        exists = target.exists() or target.is_symlink()
        had_target[item.alias] = exists
        if exists:
            _require_owned_directory(target, owner_id, location=f"target {item.alias!r}")
            old_generation_ids.add(_require_generation_marker_owner(
                target,
                owner_id=owner_id,
                root_alias=item.alias,
                location=f"target {item.alias!r}",
            ))
            _preflight_declared_owned_tree(
                target, allowed_parent=parent, owner_id=owner_id
            )
        receipt_path = staging / RECEIPT_FILE
        if receipt_path.exists() or receipt_path.is_symlink():
            raise OfflinePublicationError(
                f"staging 不允许预存 publication receipt: {receipt_path}"
            )
    if len(old_generation_ids) > 1:
        raise OfflinePublicationError(
            f"旧 targets generation identity 不一致: {sorted(old_generation_ids)}"
        )

    existing_paths: dict[str, Path] = {}
    for item in ordered_existing:
        raw_parent = Path(os.path.abspath(os.fspath(item.allowed_parent)))
        _ordinary_directory_stat(raw_parent, location=f"existing parent {item.alias!r}")
        parent = raw_parent.resolve(strict=True)
        root = Path(os.path.abspath(os.fspath(item.root)))
        if root.parent != parent:
            raise OfflinePublicationError(
                f"existing root 必须是 allowed parent 的直接子路径: {root}"
            )
        _require_owned_directory(root, owner_id, location=f"existing root {item.alias!r}")
        _preflight_declared_owned_tree(
            root, allowed_parent=parent, owner_id=owner_id
        )
        existing_paths[item.alias] = root.resolve(strict=True)

    manifest_staging = normalized[manifest_root_alias][1]
    staging_roots = {
        alias: values[1] for alias, values in normalized.items()
    } | existing_paths
    external_staging = {
        alias: path
        for alias, path in staging_roots.items()
        if alias != manifest_root_alias
    }
    try:
        manifest = load_generation_manifest(
            manifest_staging,
            manifest_path,
            root_bindings=external_staging,
        )
    except (GenerationManifestError, OSError) as exc:
        raise OfflinePublicationError(f"multi-root staging 静态校验失败: {exc}") from exc
    if manifest.schema_version != 2 or manifest.generation_id is None:
        raise OfflinePublicationError("multi-root transaction 只接受 schema v2 manifest")
    declared_aliases = {binding.alias for binding in manifest.root_bindings}
    if declared_aliases != set(aliases):
        raise OfflinePublicationError(
            f"manifest roots 与 transaction roots 不一致: "
            f"declared={sorted(declared_aliases)}, actual={sorted(aliases)}"
        )
    _require_complete_declared_trees(
        staging_roots,
        manifest=manifest,
        manifest_root_alias=manifest_root_alias,
        manifest_path=Path(manifest_path).as_posix(),
    )
    for alias, root in staging_roots.items():
        try:
            validate_generation_root_marker(
                root,
                manifest=manifest,
                root_alias=alias,
                owner_id=owner_id,
            )
        except GenerationManifestError as exc:
            raise OfflinePublicationError(
                f"root {alias!r} generation identity 预检失败: {exc}"
            ) from exc

    moved_old: set[str] = set()
    published_new: set[str] = set()
    try:
        for item in ordered_targets:
            parent, staging, target, backup = normalized[item.alias]
            if had_target[item.alias]:
                os.replace(target, backup)
                moved_old.add(item.alias)
            os.replace(staging, target)
            published_new.add(item.alias)
            _fsync_directory(parent)
    except BaseException as rename_error:
        _rollback_multi_root_transaction(
            ordered_targets,
            normalized=normalized,
            owner_id=owner_id,
            manifest=manifest,
            moved_old=moved_old,
            published_new=published_new,
            cause=rename_error,
            phase="rename",
        )
        raise AssertionError("unreachable")

    final_roots = {
        alias: values[2] for alias, values in normalized.items()
    } | existing_paths
    manifest_target = final_roots[manifest_root_alias]
    receipt: MultiRootPublicationReceipt | None = None
    try:
        external_final = {
            alias: path
            for alias, path in final_roots.items()
            if alias != manifest_root_alias
        }
        manifest = load_generation_manifest(
            manifest_target,
            manifest_path,
            root_bindings=external_final,
        )
        report = run_offline_reload_gates(
            manifest_target,
            manifest=manifest,
            root_bindings=external_final,
            probe=None,
        )
        require_publish_eligible(report)
        receipt = MultiRootPublicationReceipt(
            schema_version=2,
            status="complete",
            owner_id=owner_id,
            generation_id=manifest.generation_id or "",
            manifest_sha256=sha256_file(manifest_target / Path(manifest_path)),
            root_digests=MappingProxyType({
                alias: generation_root_resource_digest(manifest, alias)
                for alias in sorted(final_roots)
            }),
            completed_at=datetime.now(timezone.utc).isoformat(),
            reload_report=report,
        )
        _write_receipt_atomically(manifest_target, receipt)
    except BaseException as final_error:
        _remove_uncommitted_receipt(manifest_target)
        _rollback_multi_root_transaction(
            ordered_targets,
            normalized=normalized,
            owner_id=owner_id,
            manifest=manifest,
            moved_old=moved_old,
            published_new=published_new,
            cause=final_error,
            phase="full reload/receipt",
        )
        raise AssertionError("unreachable")
    if receipt is None:
        raise AssertionError("multi-root receipt missing")

    cleaned_backups: list[str] = []
    try:
        for item in ordered_targets:
            if not had_target[item.alias]:
                continue
            parent, _staging, _target, backup = normalized[item.alias]
            _remove_declared_owned_tree(
                backup,
                allowed_parent=parent,
                owner_id=owner_id,
            )
            cleaned_backups.append(item.alias)
            _fsync_directory(parent)
    except BaseException as cleanup_error:
        if not cleaned_backups:
            _remove_uncommitted_receipt(manifest_target)
            _rollback_multi_root_transaction(
                ordered_targets,
                normalized=normalized,
                owner_id=owner_id,
                manifest=manifest,
                moved_old=moved_old,
                published_new=published_new,
                cause=cleanup_error,
                phase="cleanup",
            )
            raise AssertionError("unreachable")
        raise PublicationCommittedError(
            "multi-root 发布已提交且 receipt 保留；部分旧 backups 已安全清理，"
            f"后续 cleanup 失败，不能伪称可完整回滚: cleaned={cleaned_backups}: "
            f"{cleanup_error}"
        ) from cleanup_error
    return receipt


def _rollback_multi_root_transaction(
    targets: Sequence[MultiRootPublicationTarget],
    *,
    normalized: Mapping[str, tuple[Path, Path, Path, Path]],
    owner_id: str,
    manifest: GenerationManifest,
    moved_old: set[str],
    published_new: set[str],
    cause: BaseException,
    phase: str,
) -> None:
    errors: list[BaseException] = []
    for item in reversed(list(targets)):
        parent, staging, target, backup = normalized[item.alias]
        if item.alias in published_new:
            try:
                if staging.exists() or staging.is_symlink():
                    raise OfflinePublicationError(
                        f"rollback staging 被占用: {staging}"
                    )
                _require_owned_directory(
                    target, owner_id, location=f"rollback target {item.alias!r}"
                )
                validate_generation_root_marker(
                    target,
                    manifest=manifest,
                    root_alias=item.alias,
                    owner_id=owner_id,
                )
                os.replace(target, staging)
            except BaseException as exc:
                errors.append(exc)
        if item.alias in moved_old:
            try:
                if target.exists() or target.is_symlink():
                    raise OfflinePublicationError(
                        f"rollback target 被占用: {target}"
                    )
                _require_owned_directory(
                    backup, owner_id, location=f"rollback backup {item.alias!r}"
                )
                os.replace(backup, target)
            except BaseException as exc:
                errors.append(exc)
        try:
            _fsync_directory(parent)
        except BaseException as exc:
            errors.append(exc)
    if errors:
        raise OfflinePublicationError(
            f"multi-root {phase} 失败 ({cause})；回滚不完整: {errors!r}"
        ) from cause
    raise OfflinePublicationError(
        f"multi-root {phase} 失败，全部 roots 已回滚: {cause}"
    ) from cause


def _require_generation_marker_owner(
    root: Path, *, owner_id: str, root_alias: str, location: str
) -> str:
    marker = root / GENERATION_ROOT_MARKER
    try:
        marker_stat = marker.lstat()
        if not stat.S_ISREG(marker_stat.st_mode) or marker_stat.st_nlink != 1:
            raise OfflinePublicationError(
                f"{location} generation marker 必须是单链接普通文件"
            )
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except OfflinePublicationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OfflinePublicationError(
            f"{location} 缺少或无法读取 generation marker: {exc}"
        ) from exc
    expected_fields = {
        "schema_version",
        "owner_id",
        "generation_id",
        "manifest_digest",
        "root_alias",
        "resource_digest",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise OfflinePublicationError(
            f"{location} generation marker schema 不精确"
        )
    if payload.get("schema_version") != 1 or payload.get("owner_id") != owner_id:
        raise OfflinePublicationError(
            f"{location} generation marker owner/schema 不匹配"
        )
    for field in ("manifest_digest", "resource_digest"):
        digest = payload.get(field)
        if not isinstance(digest, str) or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise OfflinePublicationError(
                f"{location} generation marker {field} 非法"
            )
    if payload.get("root_alias") != root_alias:
        raise OfflinePublicationError(
            f"{location} generation marker root_alias 不匹配"
        )
    generation_id = payload.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise OfflinePublicationError(
            f"{location} generation marker 缺少 generation_id"
        )
    return generation_id


def _remove_uncommitted_receipt(manifest_root: Path) -> None:
    receipt = manifest_root / RECEIPT_FILE
    if not receipt.exists() and not receipt.is_symlink():
        return
    receipt_stat = receipt.lstat()
    if not stat.S_ISREG(receipt_stat.st_mode) or receipt_stat.st_nlink != 1:
        raise OfflinePublicationError("失败回滚拒绝删除非法 receipt")
    receipt.unlink()
    _fsync_directory(manifest_root)


def _rollback_after_final_gate_failure(
    *,
    staging: Path,
    target: Path,
    backup: Path,
    parent: Path,
    owner_id: str,
    had_target: bool,
    gate_error: BaseException,
) -> None:
    rollback_errors: list[BaseException] = []
    try:
        _require_owned_directory(target, owner_id, location="failed new target")
        if staging.exists() or staging.is_symlink():
            raise OfflinePublicationError(
                f"回滚 staging 路径意外存在，拒绝覆盖: {staging}"
            )
        os.replace(target, staging)
    except BaseException as error:
        rollback_errors.append(error)
    if not rollback_errors and had_target:
        try:
            _require_owned_directory(backup, owner_id, location="old backup")
            os.replace(backup, target)
        except BaseException as error:
            rollback_errors.append(error)
    try:
        _fsync_directory(parent)
    except BaseException as error:
        rollback_errors.append(error)
    if rollback_errors:
        raise OfflinePublicationError(
            f"最终路径 full reload/receipt 失败 ({gate_error})；回滚也失败: "
            f"{rollback_errors!r}；new staging={staging}；old backup={backup}"
        ) from gate_error
    restored = "旧 target 已恢复" if had_target else "target 已恢复为不存在"
    raise OfflinePublicationError(
        f"最终路径 full reload/receipt 失败，{restored}；失败新集合保留在 staging: "
        f"{staging}: {gate_error}"
    ) from gate_error


def _validate_publication_paths(
    paths: OwnedPublicationPaths,
) -> tuple[Path, Path, Path, Path]:
    owner_id = _validate_owner_id(paths.owner_id)
    if owner_id != paths.owner_id:
        raise OfflinePublicationError("owner_id 规范化失败")
    parent = Path(os.path.abspath(os.fspath(paths.allowed_parent)))
    _ordinary_directory_stat(parent, location="allowed parent")
    parent = parent.resolve(strict=True)
    children = tuple(
        Path(os.path.abspath(os.fspath(path)))
        for path in (paths.staging, paths.target, paths.backup)
    )
    if len(set(children)) != 3:
        raise OfflinePublicationError("staging/target/backup 必须是三个不同路径")
    for label, child in zip(("staging", "target", "backup"), children):
        if child.parent != parent:
            raise OfflinePublicationError(
                f"{label} 必须是 allowed parent 的直接子路径: {child}"
            )
        if child.name in {"", ".", ".."}:
            raise OfflinePublicationError(f"{label} basename 非法: {child}")
        if child.is_symlink():
            raise OfflinePublicationError(f"{label} 不允许是 symlink: {child}")
        if child.exists():
            _ordinary_directory_stat(child, location=label)
    return parent, children[0], children[1], children[2]


def _require_owned_directory(root: Path, owner_id: str, *, location: str) -> None:
    _ordinary_directory_stat(root, location=location)
    marker = root / OWNER_FILE
    try:
        marker_stat = marker.lstat()
    except OSError as exc:
        raise OfflinePublicationError(f"{location} 缺少 ownership marker: {marker}") from exc
    if not stat.S_ISREG(marker_stat.st_mode) or marker_stat.st_nlink != 1:
        raise OfflinePublicationError(f"{location} ownership marker 必须是单链接普通文件")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OfflinePublicationError(f"{location} ownership marker 不可读: {exc}") from exc
    expected = {"schema_version": OWNER_SCHEMA_VERSION, "owner_id": owner_id}
    if payload != expected:
        raise OfflinePublicationError(
            f"{location} ownership marker 不匹配: expected={expected!r} actual={payload!r}"
        )


def _ordinary_directory_stat(path: Path, *, location: str) -> os.stat_result:
    try:
        path_stat = path.lstat()
    except OSError as exc:
        raise OfflinePublicationError(f"{location} 目录不可访问: {path}: {exc}") from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
        raise OfflinePublicationError(f"{location} 必须是普通目录且不能是 symlink: {path}")
    return path_stat


def _validate_owner_id(owner_id: object) -> str:
    if (
        not isinstance(owner_id, str)
        or not owner_id.strip()
        or owner_id != owner_id.strip()
        or any(char in owner_id for char in ("\x00", "\r", "\n"))
    ):
        raise OfflinePublicationError("owner_id 必须是无首尾空白的非空字符串")
    return owner_id


def _write_receipt_atomically(target: Path, receipt: MultiRootPublicationReceipt) -> None:
    final = target / RECEIPT_FILE
    if final.is_symlink() or final.exists():
        raise OfflinePublicationError(f"receipt 路径必须不存在且不能是 symlink: {final}")
    temporary = target / f".{RECEIPT_FILE}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(
                receipt.to_dict(),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, final)
        _fsync_directory(target)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _preflight_declared_owned_tree(
    root: Path, *, allowed_parent: Path, owner_id: str
) -> tuple[tuple[Path, os.stat_result], ...]:
    """在删除任何节点前绑定全树 identity 并拒绝非普通节点。"""
    if root.parent != allowed_parent:
        raise OfflinePublicationError(f"拒绝清理 allowed parent 外路径: {root}")
    _ordinary_directory_stat(root, location="backup cleanup preflight")
    _require_owned_directory(root, owner_id, location="backup cleanup preflight")
    path_stats: list[tuple[Path, os.stat_result]] = []
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path_stat = path.lstat()
        if stat.S_ISLNK(path_stat.st_mode):
            raise OfflinePublicationError(f"backup cleanup 拒绝 symlink: {path}")
        if not stat.S_ISDIR(path_stat.st_mode) and not (
            stat.S_ISREG(path_stat.st_mode) and path_stat.st_nlink == 1
        ):
            raise OfflinePublicationError(
                f"backup cleanup 拒绝 hardlink/special node: {path}"
            )
        path_stats.append((path, path_stat))
    return tuple(path_stats)


def _require_complete_declared_trees(
    roots: Mapping[str, Path],
    *,
    manifest: GenerationManifest,
    manifest_root_alias: str,
    manifest_path: str,
) -> None:
    """拒绝未被 manifest/source identity 或 publication 控制面声明的文件。"""
    declared_by_alias: dict[str, set[str]] = {alias: set() for alias in roots}
    for source in manifest.source_identities:
        declared_by_alias[source.root_alias].add(source.path)
    for entry in manifest.inventory:
        declared_by_alias[entry.root_alias].add(entry.path)
    for alias, root in roots.items():
        controls = {OWNER_FILE, GENERATION_ROOT_MARKER, RECEIPT_FILE}
        if alias == manifest_root_alias:
            controls.add(manifest_path)
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        extra = sorted(actual - declared_by_alias[alias] - controls)
        if extra:
            raise OfflinePublicationError(
                f"root {alias!r} 含未声明 publication 文件: {extra[:10]}"
            )


def _remove_declared_owned_tree(
    root: Path, *, allowed_parent: Path, owner_id: str
) -> None:
    """只清理已验证 owned backup；不使用递归删除捷径。"""
    root_stat = _ordinary_directory_stat(root, location="backup cleanup")
    root_identity = (root_stat.st_dev, root_stat.st_ino)
    path_stats = _preflight_declared_owned_tree(
        root, allowed_parent=allowed_parent, owner_id=owner_id
    )
    for path, expected in path_stats:
        current = path.lstat()
        if (current.st_dev, current.st_ino, current.st_mode) != (
            expected.st_dev, expected.st_ino, expected.st_mode
        ):
            raise OfflinePublicationError(f"backup cleanup 前节点被替换: {path}")
    for path, path_stat in path_stats:
        path.rmdir() if stat.S_ISDIR(path_stat.st_mode) else path.unlink()
    current = root.lstat()
    if (current.st_dev, current.st_ino) != root_identity:
        raise OfflinePublicationError(f"backup cleanup 期间 root 被替换: {root}")
    root.rmdir()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "OWNER_FILE",
    "RECEIPT_FILE",
    "ExistingOwnedRoot",
    "MultiRootPublicationReceipt",
    "MultiRootPublicationTarget",
    "OfflinePublicationError",
    "OwnedPublicationPaths",
    "PublicationCommittedError",
    "publish_owned_offline_dataset_roots",
    "write_ownership_marker",
]
