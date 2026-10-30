"""GenerationManifest v2：离线生成集合的唯一发布契约。

所有资源都使用显式 root alias 绑定到受控 roots；旧 schema v1 publication
不再可读取或发布，必须使用当前生成器重新生成。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

SCHEMA_VERSION = 2
MANIFEST_NAME = "generation_manifest.json"
GENERATION_ROOT_MARKER = ".generation-root.json"
GENERATION_ROOT_MARKER_SCHEMA_VERSION = 1
PUBLICATION_OWNER_MARKER = ".offline-publication-owner.json"
PUBLICATION_OWNER_SCHEMA_VERSION = 1
STATUS_COMPLETE = "complete"
ADAPTER_SE = "se-offline-scp-v1"
ADAPTER_AEC = "aec-offline-scp-v1"
ADAPTER_TSE = "tse-wesep-offline-v1"
ADAPTER_NONE = "none"
SUPPORTED_ADAPTERS = frozenset({ADAPTER_SE, ADAPTER_AEC, ADAPTER_TSE, ADAPTER_NONE})
_TASK_ADAPTERS = {
    "se": frozenset({ADAPTER_SE, ADAPTER_NONE}),
    # AEC reload adapter 仍下线；TSE 仅恢复 wesep offline 生产重载契约。
    "aec": frozenset({ADAPTER_NONE}),
    "tse": frozenset({ADAPTER_TSE}),
}
_SHA256_LENGTH = 64
_SECRET_BASENAMES = frozenset(
    {
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "known_hosts",
        "netrc",
    }
)
_SECRET_SUFFIXES = frozenset({".key", ".p12", ".pem", ".pfx"})


class GenerationManifestError(ValueError):
    """GenerationManifest 缺失、非法或与 generation root 内容不一致。"""


@dataclass(frozen=True)
class GeneratorIdentity:
    id: str
    version: str


@dataclass(frozen=True)
class GenerationCounts:
    requested: int
    published: int
    failed: int
    dropped: int


@dataclass(frozen=True)
class RootBinding:
    alias: str
    role: str


@dataclass(frozen=True)
class ResourceReference:
    root_alias: str
    path: str

    def to_dict(self) -> dict[str, str]:
        return {"root_alias": self.root_alias, "path": self.path}


@dataclass(frozen=True)
class ResourceIdentity:
    name: str
    path: str
    size: int
    sha256: str
    root_alias: str


@dataclass(frozen=True)
class InventoryEntry:
    path: str
    size: int
    sha256: str
    kind: str
    root_alias: str


@dataclass(frozen=True)
class RequiredGate:
    name: str
    adapter: str
    split: str
    inputs: Mapping[str, ResourceReference]
    options: Mapping[str, Any]


@dataclass(frozen=True)
class GenerationManifest:
    schema_version: int
    status: str
    artifact_kind: str
    task: str
    generator: GeneratorIdentity
    splits: tuple[str, ...]
    parameters: Mapping[str, Any]
    source_identities: tuple[ResourceIdentity, ...]
    inventory: tuple[InventoryEntry, ...]
    counts: GenerationCounts
    required_gates: tuple[RequiredGate, ...]
    generation_id: str
    root_bindings: tuple[RootBinding, ...]

    @classmethod
    def from_dict(cls, payload: object) -> "GenerationManifest":
        """严格解析唯一支持的 schema v2 publication envelope。"""
        raw = _mapping(payload, "GenerationManifest")
        if "schema_version" not in raw:
            raise GenerationManifestError("GenerationManifest: 缺少字段 ['schema_version']")
        schema_version = _exact_int(raw["schema_version"], "schema_version")
        if schema_version != SCHEMA_VERSION:
            raise GenerationManifestError(
                f"schema_version 必须是当前 v{SCHEMA_VERSION}；旧 v1 publication 必须重新生成，"
                f"得到 {schema_version!r}"
            )
        common_fields = {
            "schema_version", "status", "artifact_kind", "task", "generator",
            "splits", "parameters", "source_identities", "inventory", "counts",
            "required_gates", "generation_id", "root_bindings",
        }
        data = _exact_mapping(payload, location="GenerationManifest", fields=common_fields)
        generation_id = _nonempty_string(data["generation_id"], "generation_id")
        root_bindings = _parse_root_bindings(data["root_bindings"])

        status = _nonempty_string(data["status"], "status")
        if status != STATUS_COMPLETE:
            raise GenerationManifestError(
                f"status 必须是 {STATUS_COMPLETE!r}，得到 {status!r}"
            )
        artifact_kind = _nonempty_string(data["artifact_kind"], "artifact_kind")
        task = _nonempty_string(data["task"], "task")
        if task not in _TASK_ADAPTERS:
            raise GenerationManifestError(
                f"task 必须是 {sorted(_TASK_ADAPTERS)} 之一，得到 {task!r}"
            )
        generator_data = _exact_mapping(
            data["generator"], location="generator", fields={"id", "version"}
        )
        generator = GeneratorIdentity(
            id=_nonempty_string(generator_data["id"], "generator.id"),
            version=_nonempty_string(generator_data["version"], "generator.version"),
        )
        splits = _string_tuple(data["splits"], "splits")
        if len(set(splits)) != len(splits):
            raise GenerationManifestError("splits 不允许重复")
        parameters = _freeze_json_mapping(data["parameters"], "parameters")
        aliases = frozenset(binding.alias for binding in root_bindings)
        source_identities = _parse_source_identities(
            data["source_identities"], root_aliases=aliases
        )
        inventory = _parse_inventory(data["inventory"], root_aliases=aliases)
        counts_data = _exact_mapping(
            data["counts"], location="counts",
            fields={"requested", "published", "failed", "dropped"},
        )
        counts = GenerationCounts(
            requested=_nonnegative_int(counts_data["requested"], "counts.requested"),
            published=_nonnegative_int(counts_data["published"], "counts.published"),
            failed=_nonnegative_int(counts_data["failed"], "counts.failed"),
            dropped=_nonnegative_int(counts_data["dropped"], "counts.dropped"),
        )
        if counts.requested != counts.published + counts.failed + counts.dropped:
            raise GenerationManifestError(
                "counts 算术不一致：requested 必须等于 published + failed + dropped"
            )
        required_gates = _parse_required_gates(
            data["required_gates"], task, splits, root_aliases=aliases
        )
        if not required_gates:
            raise GenerationManifestError("required_gates 至少需要一项")
        if task == "tse":
            gate_splits = [gate.split for gate in required_gates]
            if sorted(gate_splits) != sorted(splits):
                raise GenerationManifestError(
                    "TSE publication 要求每个 split 恰有一个 production reload gate"
                )
        manifest = cls(
            schema_version=schema_version,
            status=status,
            artifact_kind=artifact_kind,
            task=task,
            generator=generator,
            splits=splits,
            parameters=parameters,
            source_identities=source_identities,
            inventory=inventory,
            counts=counts,
            required_gates=required_gates,
            generation_id=generation_id,
            root_bindings=root_bindings,
        )
        _validate_declared_resource_links(manifest)
        return manifest

    @property
    def manifest_root_alias(self) -> str | None:
        for binding in self.root_bindings:
            if binding.role == "manifest":
                return binding.alias
        return None

    def to_dict(self) -> dict[str, Any]:
        """返回只含 JSON-safe 基础类型的稳定 payload。"""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "status": self.status,
            "artifact_kind": self.artifact_kind,
            "task": self.task,
            "generator": {"id": self.generator.id, "version": self.generator.version},
            "splits": list(self.splits),
            "parameters": _thaw_json(self.parameters),
            "source_identities": [_resource_identity_dict(item) for item in self.source_identities],
            "inventory": [_inventory_entry_dict(item) for item in self.inventory],
            "counts": {
                "requested": self.counts.requested,
                "published": self.counts.published,
                "failed": self.counts.failed,
                "dropped": self.counts.dropped,
            },
            "required_gates": [
                {
                    "name": gate.name,
                    "adapter": gate.adapter,
                    "split": gate.split,
                    "inputs": {
                        name: value.to_dict() for name, value in gate.inputs.items()
                    },
                    "options": _thaw_json(gate.options),
                }
                for gate in self.required_gates
            ],
        }
        payload["generation_id"] = self.generation_id
        payload["root_bindings"] = [
            {"alias": binding.alias, "role": binding.role}
            for binding in self.root_bindings
        ]
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent,
            allow_nan=False,
        ) + "\n"

    def inventory_by_resource(self) -> Mapping[tuple[str, str], InventoryEntry]:
        """按显式 root alias 查找 inventory；不提供单 root 投影。"""
        return MappingProxyType(
            {(item.root_alias, item.path): item for item in self.inventory}
        )


def load_generation_manifest(
    generation_root: str | Path,
    manifest_path: str | Path = MANIFEST_NAME,
    *,
    validate_files: bool = True,
    root_bindings: Mapping[str, str | Path] | None = None,
) -> GenerationManifest:
    """加载严格 schema v2 manifest；外部 roots 必须由 alias 显式绑定。"""
    root = validate_generation_root(generation_root)
    relative_manifest = validate_relative_resource_path(
        os.fspath(manifest_path), location="manifest_path"
    )
    path = resolve_generation_resource(root, relative_manifest, location="manifest_path")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GenerationManifestError(f"无法读取 GenerationManifest {path}: {exc}") from exc
    manifest = GenerationManifest.from_dict(payload)
    bound_roots = bind_generation_roots(root, manifest, root_bindings=root_bindings)
    if validate_files:
        validate_manifest_files(root, manifest, root_bindings=bound_roots)
    return manifest


def bind_generation_roots(
    manifest_root: str | Path,
    manifest: GenerationManifest,
    *,
    root_bindings: Mapping[str, str | Path] | None = None,
    validate_markers: bool = True,
) -> Mapping[str, Path]:
    """把 schema v2 alias 绑定到调用者显式声明的普通 roots。"""
    if manifest.schema_version != SCHEMA_VERSION:
        raise GenerationManifestError(f"仅支持 schema v{SCHEMA_VERSION} manifest")
    manifest_alias = manifest.manifest_root_alias
    if manifest_alias is None:
        raise GenerationManifestError("schema v2 缺少 manifest root alias")
    supplied = dict(root_bindings or {})
    if not all(isinstance(alias, str) for alias in supplied):
        raise GenerationManifestError("root_bindings alias 必须是字符串")
    if manifest_alias in supplied:
        raise GenerationManifestError(
            f"root_bindings 不允许覆盖 manifest root alias {manifest_alias!r}"
        )
    declared = {binding.alias for binding in manifest.root_bindings}
    expected_external = declared - {manifest_alias}
    missing = sorted(expected_external - set(supplied))
    unknown = sorted(set(supplied) - expected_external)
    if missing or unknown:
        raise GenerationManifestError(
            f"root_bindings 非法: 缺少={missing}, 未知={unknown}"
        )
    roots = {manifest_alias: validate_generation_root(manifest_root)}
    for alias in sorted(expected_external):
        roots[alias] = validate_generation_root(supplied[alias])
    identities: dict[tuple[int, int], str] = {}
    for alias, bound_root in roots.items():
        root_stat = bound_root.lstat()
        identity = (root_stat.st_dev, root_stat.st_ino)
        if identity in identities:
            raise GenerationManifestError(
                f"root aliases {identities[identity]!r}/{alias!r} 绑定到同一目录"
            )
        identities[identity] = alias
        if validate_markers:
            validate_generation_root_marker(bound_root, manifest=manifest, root_alias=alias)
    return MappingProxyType(roots)


def validate_generation_root(generation_root: str | Path) -> Path:
    """绑定普通 generation root；root 自身不得是 symlink。"""
    root = Path(generation_root).expanduser()
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise GenerationManifestError(f"generation root 不可访问: {root}: {exc}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise GenerationManifestError(f"generation root 必须是普通目录且不能是 symlink: {root}")
    return root.resolve(strict=True)


def validate_relative_resource_path(value: object, *, location: str) -> str:
    """规范化 generation-root 相对 POSIX 路径。"""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise GenerationManifestError(f"{location} 必须是非空且无首尾空白的相对路径")
    if any(char in value for char in ("\x00", "\r", "\n")):
        raise GenerationManifestError(f"{location} 包含非法控制字符")
    if "\\" in value:
        raise GenerationManifestError(f"{location} 必须使用 POSIX '/' 分隔符")
    pure = PurePosixPath(value)
    if pure.is_absolute() or value.startswith("/"):
        raise GenerationManifestError(f"{location} 不允许绝对路径: {value!r}")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise GenerationManifestError(f"{location} 不允许 '.'、'..' 或空路径段: {value!r}")
    normalized = pure.as_posix()
    _reject_secret_path(normalized, location=location)
    return normalized


def resolve_generation_resource(root: Path, relative: str, *, location: str) -> Path:
    """不跟随 root 内 symlink，解析显式资源路径。"""
    relative = validate_relative_resource_path(relative, location=location)
    candidate = root
    parts = PurePosixPath(relative).parts
    for index, part in enumerate(parts):
        candidate = candidate / part
        try:
            path_stat = candidate.lstat()
        except OSError as exc:
            raise GenerationManifestError(f"{location} 资源不可访问: {relative}: {exc}") from exc
        if stat.S_ISLNK(path_stat.st_mode):
            raise GenerationManifestError(f"{location} 拒绝 symlink 或 symlink escape: {relative}")
        if index < len(parts) - 1 and not stat.S_ISDIR(path_stat.st_mode):
            raise GenerationManifestError(f"{location} 的祖先不是目录: {relative}")
    try:
        candidate.relative_to(root)
    except ValueError as exc:  # 词法校验后的纵深防御。
        raise GenerationManifestError(f"{location} 逃逸 generation root: {relative}") from exc
    return candidate


def validate_manifest_files(
    root: Path,
    manifest: GenerationManifest,
    *,
    root_bindings: Mapping[str, Path] | None = None,
) -> None:
    """只读取 manifest 显式声明的 alias-qualified source/inventory 文件。"""
    roots = root_bindings or bind_generation_roots(root, manifest)
    seen: set[tuple[str, str]] = set()
    for source in manifest.source_identities:
        key = (source.root_alias, source.path)
        if key in seen:
            raise GenerationManifestError(f"资源路径重复声明: {key}")
        seen.add(key)
        _validate_file_identity(
            roots[key[0]], source.path, source.size, source.sha256,
            location=f"source {source.name!r}",
        )
    for entry in manifest.inventory:
        key = (entry.root_alias, entry.path)
        if key in seen:
            raise GenerationManifestError(f"资源路径重复声明: {key}")
        seen.add(key)
        _validate_file_identity(
            roots[key[0]], entry.path, entry.size, entry.sha256,
            location=f"inventory root_alias={key[0]!r}",
        )


def generation_root_resource_digest(
    manifest: GenerationManifest, root_alias: str
) -> str:
    """稳定绑定某 alias 的全部 source/inventory 声明。"""
    _require_declared_root_alias(manifest, root_alias, location="root marker")
    records: list[dict[str, object]] = []
    for source in manifest.source_identities:
        if source.root_alias == root_alias:
            records.append({
                "type": "source", "name": source.name, "path": source.path,
                "size": source.size, "sha256": source.sha256,
            })
    for entry in manifest.inventory:
        if entry.root_alias == root_alias:
            records.append({
                "type": "inventory", "path": entry.path, "size": entry.size,
                "sha256": entry.sha256, "kind": entry.kind,
            })
    encoded = json.dumps(
        sorted(records, key=lambda item: json.dumps(item, sort_keys=True)),
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _generation_root_marker_payload(
    manifest: GenerationManifest, *, root_alias: str, owner_id: str
) -> dict[str, object]:
    if manifest.schema_version != SCHEMA_VERSION:
        raise GenerationManifestError(f"generation root marker 仅适用于 schema v{SCHEMA_VERSION}")
    return {
        "schema_version": GENERATION_ROOT_MARKER_SCHEMA_VERSION,
        "owner_id": _nonempty_string(owner_id, "owner_id"),
        "generation_id": manifest.generation_id,
        "manifest_digest": hashlib.sha256(
            manifest.to_json(indent=None).encode("utf-8")
        ).hexdigest(),
        "root_alias": _require_declared_root_alias(
            manifest, root_alias, location="root marker"
        ),
        "resource_digest": generation_root_resource_digest(manifest, root_alias),
    }


def write_generation_root_marker(
    root: str | Path,
    *,
    manifest: GenerationManifest,
    root_alias: str,
    owner_id: str,
) -> Path:
    """为调用者已拥有的 v2 root 写 generation identity marker。"""
    candidate = validate_generation_root(root)
    validated_owner = _nonempty_string(owner_id, "owner_id")
    owner_marker = candidate / PUBLICATION_OWNER_MARKER
    try:
        owner_stat = owner_marker.lstat()
        if not stat.S_ISREG(owner_stat.st_mode) or owner_stat.st_nlink != 1:
            raise GenerationManifestError("shared ownership marker 必须是单链接普通文件")
        owner_payload = json.loads(owner_marker.read_text(encoding="utf-8"))
    except GenerationManifestError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GenerationManifestError(
            f"写 generation marker 前缺少合法 shared ownership marker: {exc}"
        ) from exc
    expected_owner = {
        "schema_version": PUBLICATION_OWNER_SCHEMA_VERSION,
        "owner_id": validated_owner,
    }
    if owner_payload != expected_owner:
        raise GenerationManifestError(
            f"shared ownership marker 不匹配: expected={expected_owner!r} "
            f"actual={owner_payload!r}"
        )
    marker = candidate / GENERATION_ROOT_MARKER
    if marker.exists() or marker.is_symlink():
        raise GenerationManifestError(f"generation root marker 已存在: {marker}")
    payload = _generation_root_marker_payload(
        manifest, root_alias=root_alias, owner_id=validated_owner
    )
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
    return marker


def validate_generation_root_marker(
    root: str | Path,
    *,
    manifest: GenerationManifest,
    root_alias: str,
    owner_id: str | None = None,
) -> Mapping[str, object]:
    """复核 root marker 与 manifest 联合身份；可额外固定 owner。"""
    candidate = validate_generation_root(root)
    marker = candidate / GENERATION_ROOT_MARKER
    try:
        marker_stat = marker.lstat()
        if not stat.S_ISREG(marker_stat.st_mode) or marker_stat.st_nlink != 1:
            raise GenerationManifestError("generation root marker 必须是单链接普通文件")
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except GenerationManifestError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GenerationManifestError(
            f"generation root marker 不可读: {marker}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise GenerationManifestError("generation root marker 必须是 JSON object")
    marker_owner = payload.get("owner_id")
    expected = _generation_root_marker_payload(
        manifest,
        root_alias=root_alias,
        owner_id=owner_id if owner_id is not None else marker_owner,
    )
    if payload != expected:
        raise GenerationManifestError(
            f"generation root marker 不匹配: expected={expected!r} actual={payload!r}"
        )
    owner_marker = candidate / PUBLICATION_OWNER_MARKER
    try:
        owner_stat = owner_marker.lstat()
        if not stat.S_ISREG(owner_stat.st_mode) or owner_stat.st_nlink != 1:
            raise GenerationManifestError("shared ownership marker 必须是单链接普通文件")
        owner_payload = json.loads(owner_marker.read_text(encoding="utf-8"))
    except GenerationManifestError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GenerationManifestError(
            f"shared ownership marker 不可读: {owner_marker}: {exc}"
        ) from exc
    expected_owner = {
        "schema_version": PUBLICATION_OWNER_SCHEMA_VERSION,
        "owner_id": marker_owner,
    }
    if owner_payload != expected_owner:
        raise GenerationManifestError(
            f"shared ownership marker 不匹配: expected={expected_owner!r} "
            f"actual={owner_payload!r}"
        )
    return MappingProxyType(payload)


def sha256_file(path: str | Path) -> str:
    """计算普通非秘密文件 SHA-256；调用者负责 generation-root scope。"""
    candidate = Path(path)
    _reject_secret_path(candidate.name, location="sha256_file")
    digest, _ = _stable_file_digest(candidate, location="sha256_file")
    return digest


def _validate_file_identity(
    root: Path, relative: str, expected_size: int, expected_sha256: str, *, location: str
) -> None:
    path = resolve_generation_resource(root, relative, location=location)
    digest, size = _stable_file_digest(path, location=f"{location} {relative!r}")
    if size != expected_size:
        raise GenerationManifestError(
            f"{location} size 不匹配: {relative}: declared={expected_size} actual={size}"
        )
    if digest != expected_sha256:
        raise GenerationManifestError(f"{location} SHA-256 不匹配: {relative}")


def _stable_file_digest(path: Path, *, location: str) -> tuple[str, int]:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise GenerationManifestError(
                f"{location} 必须是单链接普通文件: {path}"
            )
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise GenerationManifestError(f"{location} 在打开前被替换: {path}")
            digest = hashlib.sha256()
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                after_open = os.fstat(handle.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        after = path.lstat()
    except GenerationManifestError:
        raise
    except OSError as exc:
        raise GenerationManifestError(f"{location} 无法安全读取: {path}: {exc}") from exc
    identities = {
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        for item in (before, opened, after_open, after)
    }
    if len(identities) != 1:
        raise GenerationManifestError(f"{location} 在读取期间发生变化: {path}")
    return digest.hexdigest(), after_open.st_size


def _validate_root_alias(value: object, *, location: str) -> str:
    alias = _nonempty_string(value, location)
    if not alias[0].isalnum() or any(
        not (char.isalnum() or char in {"_", "-"}) for char in alias
    ):
        raise GenerationManifestError(
            f"{location} 必须是字母数字开头的安全 alias: {alias!r}"
        )
    return alias


def _parse_root_bindings(value: object) -> tuple[RootBinding, ...]:
    items = _list(value, "root_bindings")
    if not items:
        raise GenerationManifestError("root_bindings 至少需要一项")
    parsed: list[RootBinding] = []
    aliases: set[str] = set()
    manifest_count = 0
    for index, item in enumerate(items):
        location = f"root_bindings[{index}]"
        data = _exact_mapping(item, location=location, fields={"alias", "role"})
        alias = _validate_root_alias(data["alias"], location=f"{location}.alias")
        role = _nonempty_string(data["role"], f"{location}.role")
        if role not in {"manifest", "resource"}:
            raise GenerationManifestError(f"{location}.role 未知: {role!r}")
        if alias in aliases:
            raise GenerationManifestError(f"root_bindings alias 重复: {alias!r}")
        aliases.add(alias)
        manifest_count += role == "manifest"
        parsed.append(RootBinding(alias=alias, role=role))
    if manifest_count != 1:
        raise GenerationManifestError("root_bindings 必须恰有一个 manifest root")
    return tuple(parsed)


def _parse_resource_reference(
    value: object, *, location: str, root_aliases: frozenset[str]
) -> ResourceReference:
    data = _exact_mapping(value, location=location, fields={"root_alias", "path"})
    alias = _validate_root_alias(data["root_alias"], location=f"{location}.root_alias")
    if alias not in root_aliases:
        raise GenerationManifestError(f"{location} 使用未知 root_alias: {alias!r}")
    return ResourceReference(
        root_alias=alias,
        path=validate_relative_resource_path(data["path"], location=f"{location}.path"),
    )


def _parse_source_identities(
    value: object, *, root_aliases: frozenset[str]
) -> tuple[ResourceIdentity, ...]:
    items = _list(value, "source_identities")
    parsed = []
    names: set[str] = set()
    paths: set[tuple[str, str]] = set()
    fields = {"name", "path", "size", "sha256", "root_alias"}
    for index, item in enumerate(items):
        location = f"source_identities[{index}]"
        data = _exact_mapping(item, location=location, fields=fields)
        alias = _validate_root_alias(
            data["root_alias"], location=f"{location}.root_alias"
        )
        if alias not in root_aliases:
            raise GenerationManifestError(f"{location} 使用未知 root_alias: {alias!r}")
        identity = ResourceIdentity(
            name=_nonempty_string(data["name"], f"{location}.name"),
            path=validate_relative_resource_path(data["path"], location=f"{location}.path"),
            size=_nonnegative_int(data["size"], f"{location}.size"),
            sha256=_sha256(data["sha256"], f"{location}.sha256"),
            root_alias=alias,
        )
        key = (alias, identity.path)
        if identity.name in names or key in paths:
            raise GenerationManifestError(f"{location} name/resource 不允许重复")
        names.add(identity.name)
        paths.add(key)
        parsed.append(identity)
    return tuple(parsed)


def _parse_inventory(
    value: object, *, root_aliases: frozenset[str]
) -> tuple[InventoryEntry, ...]:
    items = _list(value, "inventory")
    if not items:
        raise GenerationManifestError("inventory 至少需要一项")
    parsed = []
    paths: set[tuple[str, str]] = set()
    fields = {"path", "size", "sha256", "kind", "root_alias"}
    for index, item in enumerate(items):
        location = f"inventory[{index}]"
        data = _exact_mapping(item, location=location, fields=fields)
        alias = _validate_root_alias(
            data["root_alias"], location=f"{location}.root_alias"
        )
        if alias not in root_aliases:
            raise GenerationManifestError(f"{location} 使用未知 root_alias: {alias!r}")
        entry = InventoryEntry(
            path=validate_relative_resource_path(data["path"], location=f"{location}.path"),
            size=_nonnegative_int(data["size"], f"{location}.size"),
            sha256=_sha256(data["sha256"], f"{location}.sha256"),
            kind=_nonempty_string(data["kind"], f"{location}.kind"),
            root_alias=alias,
        )
        key = (alias, entry.path)
        if key in paths:
            raise GenerationManifestError(f"inventory resource 重复: {key}")
        paths.add(key)
        parsed.append(entry)
    return tuple(parsed)


def _parse_required_gates(
    value: object,
    task: str,
    splits: tuple[str, ...],
    *,
    root_aliases: frozenset[str],
) -> tuple[RequiredGate, ...]:
    items = _list(value, "required_gates")
    parsed = []
    names: set[str] = set()
    for index, item in enumerate(items):
        location = f"required_gates[{index}]"
        data = _exact_mapping(
            item, location=location,
            fields={"name", "adapter", "split", "inputs", "options"},
        )
        name = _nonempty_string(data["name"], f"{location}.name")
        adapter = _nonempty_string(data["adapter"], f"{location}.adapter")
        split = _nonempty_string(data["split"], f"{location}.split")
        if adapter not in SUPPORTED_ADAPTERS:
            raise GenerationManifestError(f"{location}.adapter 未知: {adapter!r}")
        if adapter not in _TASK_ADAPTERS[task]:
            raise GenerationManifestError(
                f"task {task!r} 与 adapter {adapter!r} 不匹配"
            )
        if split not in splits:
            raise GenerationManifestError(f"{location}.split 未在 splits 中声明: {split!r}")
        inputs_data = _mapping(data["inputs"], f"{location}.inputs")
        inputs: dict[str, ResourceReference] = {}
        for key, raw_reference in inputs_data.items():
            input_name = _nonempty_string(key, f"{location}.inputs key")
            inputs[input_name] = _parse_resource_reference(
                raw_reference,
                location=f"{location}.inputs.{input_name}",
                root_aliases=root_aliases,
            )
        if adapter == ADAPTER_NONE and inputs:
            raise GenerationManifestError("none adapter 不允许声明 inputs")
        options = _freeze_json_mapping(data["options"], f"{location}.options")
        if name in names:
            raise GenerationManifestError(f"required gate name 重复: {name!r}")
        names.add(name)
        parsed.append(RequiredGate(
            name=name, adapter=adapter, split=split,
            inputs=MappingProxyType(inputs), options=options,
        ))
    return tuple(parsed)


def _validate_declared_resource_links(manifest: GenerationManifest) -> None:
    inventory = {(item.root_alias, item.path) for item in manifest.inventory}
    source_paths = {(item.root_alias, item.path) for item in manifest.source_identities}
    if inventory & source_paths:
        raise GenerationManifestError("source_identities 与 inventory 路径必须互不重复")
    for gate in manifest.required_gates:
        references = {(value.root_alias, value.path) for value in gate.inputs.values()}
        missing = sorted(references - inventory)
        if missing:
            raise GenerationManifestError(
                f"required gate {gate.name!r} inputs 未绑定 inventory: {missing}"
            )


def _require_declared_root_alias(
    manifest: GenerationManifest, root_alias: object, *, location: str
) -> str:
    alias = _validate_root_alias(root_alias, location=f"{location}.root_alias")
    if alias not in {binding.alias for binding in manifest.root_bindings}:
        raise GenerationManifestError(f"{location} 使用未知 root_alias: {alias!r}")
    return alias


def _resource_identity_dict(item: ResourceIdentity) -> dict[str, object]:
    return {
        "name": item.name,
        "path": item.path,
        "size": item.size,
        "sha256": item.sha256,
        "root_alias": item.root_alias,
    }


def _inventory_entry_dict(item: InventoryEntry) -> dict[str, object]:
    return {
        "path": item.path,
        "size": item.size,
        "sha256": item.sha256,
        "kind": item.kind,
        "root_alias": item.root_alias,
    }


def _exact_mapping(value: object, *, location: str, fields: set[str]) -> Mapping[str, Any]:
    data = _mapping(value, location)
    actual = set(data)
    missing = fields - actual
    unknown = actual - fields
    if missing or unknown:
        details = []
        if missing:
            details.append(f"缺少字段 {sorted(missing)}")
        if unknown:
            details.append(f"未知字段 {sorted(unknown)}")
        raise GenerationManifestError(f"{location}: " + "；".join(details))
    return data


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise GenerationManifestError(f"{location} 必须是 string-key JSON object")
    return value


def _list(value: object, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise GenerationManifestError(f"{location} 必须是 JSON array")
    return value


def _exact_int(value: object, location: str) -> int:
    if type(value) is not int:
        raise GenerationManifestError(f"{location} 必须是 exact int，得到 {value!r}")
    return value


def _nonnegative_int(value: object, location: str) -> int:
    parsed = _exact_int(value, location)
    if parsed < 0:
        raise GenerationManifestError(f"{location} 必须是非负整数")
    return parsed


def _nonempty_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise GenerationManifestError(f"{location} 必须是无首尾空白的非空字符串")
    if any(char in value for char in ("\x00", "\r", "\n")):
        raise GenerationManifestError(f"{location} 包含非法控制字符")
    return value


def _string_tuple(value: object, location: str) -> tuple[str, ...]:
    items = _list(value, location)
    if not items:
        raise GenerationManifestError(f"{location} 至少需要一项")
    return tuple(
        _nonempty_string(item, f"{location}[{index}]")
        for index, item in enumerate(items)
    )


def _sha256(value: object, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise GenerationManifestError(f"{location} 必须是 64 位小写 SHA-256")
    return value


def _freeze_json_mapping(value: object, location: str) -> Mapping[str, Any]:
    data = _mapping(value, location)
    return MappingProxyType(
        {key: _freeze_json(child, f"{location}.{key}") for key, child in data.items()}
    )


def _freeze_json(value: object, location: str) -> Any:
    if value is None or isinstance(value, (str, bool)) or type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise GenerationManifestError(f"{location} 不允许 NaN/Inf")
        return value
    if isinstance(value, list):
        return tuple(_freeze_json(child, f"{location}[]") for child in value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise GenerationManifestError(f"{location} JSON object key 必须是字符串")
        return MappingProxyType(
            {key: _freeze_json(child, f"{location}.{key}") for key, child in value.items()}
        )
    raise GenerationManifestError(
        f"{location} 不是 JSON-safe 类型: {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


def _reject_secret_path(relative: str, *, location: str) -> None:
    for raw_part in PurePosixPath(relative).parts:
        part = raw_part.lower()
        if (
            part == ".env"
            or part.startswith(".env.")
            or part in _SECRET_BASENAMES
            or any(part.endswith(suffix) for suffix in _SECRET_SUFFIXES)
            or "credential" in part
            or "private_key" in part
        ):
            raise GenerationManifestError(
                f"{location} 拒绝读取或散列疑似秘密文件: {relative!r}"
            )


__all__ = [
    "ADAPTER_AEC",
    "ADAPTER_NONE",
    "ADAPTER_SE",
    "ADAPTER_TSE",
    "GENERATION_ROOT_MARKER",
    "MANIFEST_NAME",
    "SCHEMA_VERSION",
    "GenerationCounts",
    "GenerationManifest",
    "GenerationManifestError",
    "GeneratorIdentity",
    "InventoryEntry",
    "RequiredGate",
    "ResourceIdentity",
    "ResourceReference",
    "RootBinding",
    "bind_generation_roots",
    "generation_root_resource_digest",
    "load_generation_manifest",
    "resolve_generation_resource",
    "sha256_file",
    "validate_generation_root",
    "validate_generation_root_marker",
    "validate_manifest_files",
    "validate_relative_resource_path",
    "write_generation_root_marker",
]
