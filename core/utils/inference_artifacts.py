"""推理产物的指纹、清单与原子写入工具。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Iterable, Iterator, Mapping

from core.utils.file_transaction import (
    _create_temporary_file,
    _lock_parent_directories,
    _publish_staged_files,
    _unlink_at,
    write_text_files_transactionally,
)
from tools.shared.scp_provenance import build_scp_provenance

try:
    import fcntl
except ImportError:  # pragma: no cover - 项目部署目标为 Linux。
    fcntl = None


MANIFEST_SCHEMA_VERSION = 3
AUDIO_IDENTITY_ALGORITHM = "stat-v1"
_RESULT_STATUSES = {"ok", "skipped", "failed"}
_OUTPUT_LOCK_FILENAME = ".inference.lock"


class ManifestValidationError(ValueError):
    """推理 manifest 不符合严格 schema 或与当前运行不匹配。"""


def canonical_json_value(
    value: Any,
    *,
    strict_keys: bool = True,
    _active: frozenset[int] = frozenset(),
) -> Any:
    """将 JSON-like 值规范化为只含普通 dict/list 的隔离副本。

    推理合同内部可以使用 mappingproxy/tuple 做深冻结，但发布和 fingerprint
    必须只接收标准 JSON 容器，避免把 mappingproxy 直接交给 ``json``。
    """
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("value 必须可被标准 JSON 序列化")
        return value
    value_id = id(value)
    if value_id in _active:
        raise ValueError("value must not contain cycles")
    active = _active | {value_id}
    if isinstance(value, Mapping):
        if strict_keys and any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {
            key: canonical_json_value(item, strict_keys=strict_keys, _active=active)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            canonical_json_value(item, strict_keys=strict_keys, _active=active)
            for item in value
        ]
    raise ValueError("value 必须可被标准 JSON 序列化")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        canonical_json_value(value, strict_keys=False),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def fingerprint_file(path: str | Path) -> dict[str, Any]:
    """按文件内容生成稳定 SHA-256 指纹，并绑定规范化绝对路径与大小。

    读取前后的身份取自被读取的 FD，并在末尾复核路径仍指向同一 inode：
    同名替换不会让指纹绑定到未读取的内容。
    """
    resolved = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with os.fdopen(os.open(resolved, os.O_RDONLY), "rb") as f:
        before = os.fstat(f.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"不是普通文件: {resolved}")
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(f.fileno())

    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"计算指纹时文件发生变化: {resolved}")
    try:
        current = resolved.stat()
    except OSError as exc:
        raise RuntimeError(f"计算指纹期间文件被移除或替换: {resolved}") from exc
    if (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino):
        raise RuntimeError(f"计算指纹期间文件被移除或替换: {resolved}")
    return {
        "path": str(resolved),
        "size": int(after.st_size),
        "sha256": digest.hexdigest(),
    }


def stat_audio_identity(path: str | Path) -> dict[str, Any]:
    """仅用 ``stat`` 建立音频结构身份，绝不打开或读取 payload。"""
    resolved = Path(path).expanduser().resolve(strict=True)
    file_stat = resolved.stat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"不是普通音频文件: {resolved}")
    return {
        "algorithm": AUDIO_IDENTITY_ALGORITHM,
        "path": str(resolved),
        "st_dev": int(file_stat.st_dev),
        "st_ino": int(file_stat.st_ino),
        "st_size": int(file_stat.st_size),
        "st_mtime_ns": int(file_stat.st_mtime_ns),
    }


def open_parent_directory_no_symlinks(
    path: str | Path, *, create_missing: bool = False
) -> tuple[Path, int]:
    """打开目标 parent directory，拒绝祖先 symlink 并返回稳定 dirfd。"""
    absolute = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.sep, flags)
    try:
        for component in absolute.parent.parts[1:]:
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
                    raise ValueError(
                        f"path contains symbolic link ancestor: {absolute}"
                    )
                if not stat.S_ISDIR(component_stat.st_mode):
                    raise NotADirectoryError(
                        f"path ancestor is not a directory: {absolute}"
                    )
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
                break
        return absolute, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def audio_identity_from_stat(path: str | Path, file_stat: os.stat_result) -> dict[str, Any]:
    """从已打开文件的 stat 建立统一 ``stat-v1`` 音频身份。"""
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"不是普通音频文件: {path}")
    return {
        "algorithm": AUDIO_IDENTITY_ALGORITHM,
        "path": str(path),
        "st_dev": int(file_stat.st_dev),
        "st_ino": int(file_stat.st_ino),
        "st_size": int(file_stat.st_size),
        "st_mtime_ns": int(file_stat.st_mtime_ns),
    }


def stat_audio_identity_no_symlinks(path: str | Path) -> dict[str, Any]:
    """建立音频身份并拒绝 leaf/ancestor symlink。"""
    absolute, parent_fd = open_parent_directory_no_symlinks(path)
    try:
        file_stat = os.lstat(absolute.name, dir_fd=parent_fd)
        if stat.S_ISLNK(file_stat.st_mode):
            raise ValueError(
                f"音频 artifact 不允许符号链接 (symbolic link): {absolute}"
            )
        return audio_identity_from_stat(absolute, file_stat)
    finally:
        os.close(parent_fd)


def open_regular_file_no_symlinks(
    path: str | Path, *, label: str
) -> tuple[Path, int]:
    """以稳定 FD 打开普通文件，拒绝 leaf/ancestor symlink。"""
    absolute, parent_fd = open_parent_directory_no_symlinks(path)
    try:
        leaf_stat = os.lstat(absolute.name, dir_fd=parent_fd)
        if stat.S_ISLNK(leaf_stat.st_mode):
            raise ValueError(f"{label} must not be a symbolic link: {absolute}")
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError(f"{label} must be a regular file: {absolute}")
    finally:
        os.close(parent_fd)
    return absolute, descriptor


def ensure_directory_no_symlinks(path: str | Path) -> Path:
    """创建目录并拒绝其祖先或 leaf 为 symlink。"""
    absolute_marker, parent_fd = open_parent_directory_no_symlinks(
        Path(os.path.abspath(os.fspath(Path(path).expanduser()))) / ".directory-check",
        create_missing=True,
    )
    os.close(parent_fd)
    return absolute_marker.parent


def fingerprint_json(value: Any) -> str:
    """对可标准 JSON 序列化的值生成稳定 SHA-256。"""
    try:
        return _sha256_json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("value 必须可被标准 JSON 序列化") from exc


def fingerprint_path(path: str | Path) -> dict[str, Any]:
    """按内容指纹化文件或目录；目录指纹绑定相对路径与每个文件的内容。"""
    resolved = Path(path).resolve(strict=True)
    if resolved.is_file():
        return {**fingerprint_file(resolved), "kind": "file"}
    if not resolved.is_dir():
        raise ValueError(f"不是普通文件或目录: {resolved}")

    files = sorted(child for child in resolved.rglob("*") if child.is_file())
    entries = []
    for child in files:
        file_fingerprint = fingerprint_file(child)
        entries.append(
            {
                "relative_path": child.relative_to(resolved).as_posix(),
                "size": file_fingerprint["size"],
                "sha256": file_fingerprint["sha256"],
            }
        )
    return {
        "path": str(resolved),
        "kind": "directory",
        "files": len(entries),
        "sha256": fingerprint_json(entries),
    }


_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py"}


def collect_source_tree_files(
    project_root: str | Path,
    paths: Iterable[str | Path],
    *,
    name_prefix: str = "source",
) -> dict[str, Path]:
    """收集显式源码文件/目录中的实现文件，返回稳定的具名文件映射。

    仅遍历调用方列出的边界，目录内跳过隐藏路径与 ``__pycache__``，避免把数据、
    第三方代码或缓存误纳入 artifact；TSE 等入口可传入自己的源码边界复用。
    """
    root = Path(project_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"project_root 不是目录: {root}")

    collected: dict[str, Path] = {}
    for value in paths:
        candidate = (root / value).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"源码路径超出 project_root: {candidate}") from exc
        if candidate.is_file():
            source_files = [candidate] if candidate.suffix.lower() in _SOURCE_SUFFIXES else []
        elif candidate.is_dir():
            source_files = candidate.rglob("*")
        else:
            raise ValueError(f"源码路径不是文件或目录: {candidate}")
        for source_file in source_files:
            relative = source_file.relative_to(root)
            if (
                not source_file.is_file()
                or source_file.suffix.lower() not in _SOURCE_SUFFIXES
                or any(part == "__pycache__" or part.startswith(".") for part in relative.parts)
            ):
                continue
            collected[f"{name_prefix}:{relative.as_posix()}"] = source_file
    if not collected:
        raise ValueError("源码树未收集到任何实现文件")
    return {name: collected[name] for name in sorted(collected)}


def _validate_precomputed_fingerprint(
    fingerprint: Mapping[str, Any], name: str
) -> dict[str, Any]:
    """校验预计算指纹与 ``fingerprint_file`` 输出格式完全一致。"""
    if not isinstance(fingerprint, Mapping) or set(fingerprint) != {
        "path",
        "size",
        "sha256",
    }:
        raise ValueError(f"artifact {name!r} 的预计算指纹必须是 path/size/sha256 mapping")
    path = fingerprint["path"]
    size = fingerprint["size"]
    sha256 = fingerprint["sha256"]
    if not isinstance(path, str) or not path:
        raise ValueError(f"artifact {name!r} 的预计算指纹 path 非法")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"artifact {name!r} 的预计算指纹 size 非法")
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError(f"artifact {name!r} 的预计算指纹 sha256 非法")
    return {"path": path, "size": size, "sha256": sha256}


def build_content_fingerprint(
    files: Mapping[str, str | Path | None],
    *,
    parameters: Mapping[str, Any] | None = None,
    precomputed: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """为具名 artifact 内容与 JSON 运行参数生成稳定 SHA-256。

    ``precomputed`` 允许调用方按名称传入已由受信 loader 验证过的内容指纹
    （格式与 ``fingerprint_file`` 返回值相同），避免对同一文件重复全量哈希；
    名称必须属于 ``files`` 且格式非法时 fail-closed。
    """
    if not isinstance(files, Mapping) or not files:
        raise ValueError("files 必须是非空 mapping")
    precomputed = precomputed or {}
    unknown = sorted(set(precomputed) - set(files))
    if unknown:
        raise ValueError(f"precomputed 指纹包含未知 artifact: {unknown[:5]}")
    fingerprints: dict[str, dict[str, Any] | None] = {}
    for name in sorted(files):
        if not isinstance(name, str) or not name:
            raise ValueError("artifact 名称必须是非空字符串")
        value = files[name]
        if value is None:
            fingerprints[name] = None
        elif name in precomputed:
            fingerprints[name] = _validate_precomputed_fingerprint(
                precomputed[name], name
            )
        else:
            fingerprints[name] = fingerprint_file(value)
    try:
        parameter_values = json.loads(_canonical_json_bytes(parameters or {}))
    except (TypeError, ValueError) as exc:
        raise ValueError("parameters 必须可被标准 JSON 序列化") from exc
    payload = {
        "algorithm": "sha256",
        "files": fingerprints,
        "parameters": parameter_values,
    }
    return {**payload, "sha256": _sha256_json(payload)}


def build_input_inventory(
    items: Iterable[Any],
) -> dict[str, Any]:
    """为按 UID 组织的音频输入建立带 attributes 的 ``stat-v1`` inventory。"""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in items:
        attributes: Any = {}
        if hasattr(entry, "uid") and hasattr(entry, "inputs"):
            uid = entry.uid
            inputs = entry.inputs
            attributes = getattr(entry, "attributes", {})
        else:
            try:
                values = tuple(entry)
            except TypeError as exc:
                raise ValueError("input inventory item 必须是 item 或 tuple") from exc
            if len(values) not in {2, 3}:
                raise ValueError("input inventory tuple 必须是 (uid, inputs) 或 (uid, inputs, attributes)")
            uid, inputs = values[:2]
            if len(values) == 3:
                attributes = values[2]
        if not isinstance(uid, str) or not uid:
            raise ValueError("input uid 必须是非空字符串")
        if uid in seen:
            raise ValueError(f"输入 inventory 存在重复 uid: {uid}")
        seen.add(uid)
        if not isinstance(inputs, Mapping) or not inputs:
            raise ValueError(f"uid={uid!r} 的 inputs 必须是非空 mapping")
        try:
            normalized_attributes = canonical_json_value(attributes)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"uid={uid!r} 的 attributes 必须是严格 JSON") from exc
        input_identities: dict[str, dict[str, Any] | None] = {}
        for role in sorted(inputs):
            if not isinstance(role, str) or not role:
                raise ValueError("input role 必须是非空字符串")
            value = inputs[role]
            input_identities[role] = (
                None if value is None else stat_audio_identity(value)
            )
        records.append(
            {"uid": uid, "inputs": input_identities, "attributes": normalized_attributes}
        )
    payload = {"algorithm": AUDIO_IDENTITY_ALGORITHM, "items": records}
    return {**payload, "digest": _sha256_json(payload)}


@contextmanager
def inference_output_lock(output_dir: str | Path) -> Iterator[Path]:
    """持有 output-dir 持久锁；锁文件不删除，避免 inode 切换破坏互斥。

    非阻塞获取（LOCK_NB）：并发 writer 进入前 fail-fast 报错，与 TSE
    ``exclusive_output_run`` 语义一致，不做无限期等待。
    """
    if fcntl is None:
        raise RuntimeError("inference output locking requires fcntl")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / _OUTPUT_LOCK_FILENAME
    try:
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(
                f"inference output lock 不得是 symlink: {lock_path}"
            ) from exc
        raise
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"inference output lock 必须是普通文件: {lock_path}")
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"inference output_dir 已有活跃 writer: {output}") from exc
        yield output
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def atomic_write_audio(path: str | Path, writer: Callable[[Path], None]) -> None:
    """安全地 staging 音频后原子发布，拒绝任意 ancestor/叶 symlink。"""
    target = Path(path).expanduser()
    with _lock_parent_directories([target], create_missing=True) as directory_fds:
        directory_fd = directory_fds[Path(os.path.abspath(os.fspath(target))).parent]
        descriptor, temporary = _create_temporary_file(
            directory_fd,
            target.parent,
            prefix=f".{target.stem}.",
            suffix=f".tmp{target.suffix}",
        )
        os.close(descriptor)
        # writer 仅能经稳定 dirfd 访问 staging，祖先目录替换不会重定向写入位置。
        writer_path = Path("/proc/self/fd") / str(directory_fd) / temporary.name
        try:
            writer(writer_path)
            staging_fd = os.open(
                temporary.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                os.fsync(staging_fd)
            finally:
                os.close(staging_fd)
            _publish_staged_files([temporary], [target], directory_fds=directory_fds)
        except BaseException:
            _unlink_at(temporary, directory_fds)
            raise


def publish_inference_metadata(
    output_dir: str | Path,
    *,
    scp_content: str,
    manifest: Mapping[str, Any],
) -> None:
    """以一个可回滚集合事务发布 SCP、provenance sidecar 与 manifest。"""
    output = Path(output_dir).expanduser()
    scp_path = output / "inf.scp"
    manifest_content = (
        json.dumps(dict(manifest), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    scp_provenance = build_scp_provenance(
        scp_content,
        kind="generated",
        source=str(output.resolve()),
        pattern="inference_outputs",
        recursive=False,
        scp_path=scp_path,
    )
    sidecar_content = (
        json.dumps(scp_provenance, indent=2, ensure_ascii=False) + "\n"
    )

    def write_content(content: str) -> Callable[[Any], None]:
        return lambda stream: stream.write(content)

    write_text_files_transactionally(
        [
            (scp_path, write_content(scp_content)),
            (Path(f"{scp_path}.provenance.json"), write_content(sidecar_content)),
            (output / "run_manifest.json", write_content(manifest_content)),
        ],
        overwrite=True,
    )


def atomic_write_text(path: str | Path, content: str) -> None:
    """安全地原子发布 UTF-8 文本，拒绝任意 ancestor/叶 symlink。"""
    target = Path(path).expanduser()

    def write_text(stream: Any) -> None:
        stream.write(content)

    write_text_files_transactionally([(target, write_text)], overwrite=True)


def atomic_write_json(path: str | Path, value: Any) -> None:
    """以 UTF-8、稳定缩进格式原子写 JSON。"""
    atomic_write_text(
        path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], where: str) -> None:
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ManifestValidationError(
            f"{where} 字段不匹配: missing={missing}, extra={extra}"
        )


def _validate_file_fingerprint(value: Any, where: str) -> None:
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{where} 必须是 object")
    _require_exact_keys(value, {"path", "size", "sha256"}, where)
    if not isinstance(value["path"], str) or not Path(value["path"]).is_absolute():
        raise ManifestValidationError(f"{where}.path 必须是绝对路径字符串")
    if isinstance(value["size"], bool) or not isinstance(value["size"], int) or value["size"] < 0:
        raise ManifestValidationError(f"{where}.size 必须是非负整数")
    sha256 = value["sha256"]
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ManifestValidationError(f"{where}.sha256 必须是 64 位字符串")
    try:
        int(sha256, 16)
    except ValueError as exc:
        raise ManifestValidationError(f"{where}.sha256 不是十六进制") from exc


def _validate_content_fingerprint(value: Any, where: str) -> None:
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{where} 必须是 object")
    _require_exact_keys(
        value, {"algorithm", "files", "parameters", "sha256"}, where
    )
    if value["algorithm"] != "sha256" or not isinstance(value["files"], dict):
        raise ManifestValidationError(f"{where} algorithm/files 非法")
    if not isinstance(value["parameters"], dict):
        raise ManifestValidationError(f"{where}.parameters 必须是 object")
    if not value["files"]:
        raise ManifestValidationError(f"{where}.files 不能为空")
    for name, fingerprint in value["files"].items():
        if not isinstance(name, str) or not name:
            raise ManifestValidationError(f"{where}.files 名称非法")
        if fingerprint is not None:
            _validate_file_fingerprint(fingerprint, f"{where}.files[{name!r}]")
    payload = {
        "algorithm": value["algorithm"],
        "files": value["files"],
        "parameters": value["parameters"],
    }
    try:
        expected_sha256 = _sha256_json(payload)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"{where}.parameters 不能序列化为 JSON") from exc
    if value["sha256"] != expected_sha256:
        raise ManifestValidationError(f"{where}.sha256 与内容不一致")


def _validate_audio_identity(value: Any, where: str) -> None:
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{where} 必须是 object")
    keys = {"algorithm", "path", "st_dev", "st_ino", "st_size", "st_mtime_ns"}
    _require_exact_keys(value, keys, where)
    if value["algorithm"] != AUDIO_IDENTITY_ALGORITHM:
        raise ManifestValidationError(f"{where}.algorithm 非法")
    if not isinstance(value["path"], str) or not Path(value["path"]).is_absolute():
        raise ManifestValidationError(f"{where}.path 必须是绝对路径字符串")
    for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns"):
        field = value[name]
        if isinstance(field, bool) or not isinstance(field, int) or field < 0:
            raise ManifestValidationError(f"{where}.{name} 必须是非负整数")


def _validate_input_inventory(value: Any, where: str = "input_inventory") -> None:
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{where} 必须是 object")
    _require_exact_keys(value, {"algorithm", "items", "digest"}, where)
    if value["algorithm"] != AUDIO_IDENTITY_ALGORITHM or not isinstance(value["items"], list):
        raise ManifestValidationError(f"{where} algorithm/items 非法")
    seen: set[str] = set()
    for index, item in enumerate(value["items"]):
        item_where = f"{where}.items[{index}]"
        if not isinstance(item, dict):
            raise ManifestValidationError(f"{item_where} 必须是 object")
        _require_exact_keys(item, {"uid", "inputs", "attributes"}, item_where)
        uid = item["uid"]
        if not isinstance(uid, str) or not uid or uid in seen:
            raise ManifestValidationError(f"{item_where}.uid 为空或重复")
        seen.add(uid)
        inputs = item["inputs"]
        if not isinstance(inputs, dict) or not inputs:
            raise ManifestValidationError(f"{item_where}.inputs 必须是非空 object")
        for role, identity in inputs.items():
            if not isinstance(role, str) or not role:
                raise ManifestValidationError(f"{item_where}.inputs role 非法")
            if identity is not None:
                _validate_audio_identity(identity, f"{item_where}.inputs[{role!r}]")
        try:
            normalized_attributes = canonical_json_value(item["attributes"])
        except (TypeError, ValueError) as exc:
            raise ManifestValidationError(
                f"{item_where}.attributes 必须是严格 JSON"
            ) from exc
        if normalized_attributes != item["attributes"]:
            raise ManifestValidationError(
                f"{item_where}.attributes 必须是 canonical JSON"
            )
    payload = {"algorithm": value["algorithm"], "items": value["items"]}
    if value["digest"] != _sha256_json(payload):
        raise ManifestValidationError(f"{where}.digest 与内容不一致")


@dataclass(frozen=True)
class InferenceResult:
    """单个 UID 的推理结果；成功/复用结果必须绑定输出结构身份。"""

    uid: str
    status: str
    output_path: str | None = None
    output_identity: dict[str, Any] | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.uid, str) or not self.uid:
            raise ManifestValidationError("InferenceResult.uid 必须是非空字符串")
        if self.status not in _RESULT_STATUSES:
            raise ManifestValidationError(f"InferenceResult.status 非法: {self.status!r}")
        if self.status == "failed":
            if self.output_path is not None or self.output_identity is not None:
                raise ManifestValidationError("failed result 不得包含输出")
            if not isinstance(self.error, str) or not self.error:
                raise ManifestValidationError("failed result 必须包含非空 error")
        else:
            if not isinstance(self.output_path, str) or not Path(self.output_path).is_absolute():
                raise ManifestValidationError("ok/skipped result 必须包含绝对 output_path")
            _validate_audio_identity(
                self.output_identity, "InferenceResult.output_identity"
            )
            if Path(self.output_identity["path"]).resolve() != Path(self.output_path).resolve():
                raise ManifestValidationError(
                    "output_identity.path 必须与 output_path 指向同一文件"
                )
            if self.error is not None:
                raise ManifestValidationError("ok/skipped result 不得包含 error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "status": self.status,
            "output_path": self.output_path,
            "output_identity": self.output_identity,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "InferenceResult":
        if not isinstance(value, dict):
            raise ManifestValidationError("result 必须是 object")
        _require_exact_keys(
            value,
            {"uid", "status", "output_path", "output_identity", "error"},
            "result",
        )
        return cls(**value)


def build_run_manifest(
    *,
    config: str | Path,
    checkpoint: str | Path | None,
    artifact_fingerprint: Mapping[str, Any],
    input_inventory: Mapping[str, Any],
    results: Iterable[InferenceResult],
    allow_partial: bool,
    created_at: str,
) -> dict[str, Any]:
    """构建并自校验标准推理 manifest。"""
    result_list = list(results)
    expected_uids = [item["uid"] for item in input_inventory.get("items", [])]
    by_status = {
        status: [result.uid for result in result_list if result.status == status]
        for status in _RESULT_STATUSES
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "config": str(Path(config).resolve()),
        "checkpoint": str(Path(checkpoint).resolve()) if checkpoint is not None else None,
        "allow_partial": bool(allow_partial),
        "created_at": created_at,
        "artifact_fingerprint": dict(artifact_fingerprint),
        "input_inventory": dict(input_inventory),
        "expected_uids": expected_uids,
        "ok_uids": by_status["ok"],
        "skipped_uids": by_status["skipped"],
        "failed_uids": by_status["failed"],
        "results": [result.to_dict() for result in result_list],
    }
    validate_run_manifest(manifest)
    return manifest


def validate_run_manifest(
    manifest: Any,
    *,
    expected_artifact_fingerprint: Mapping[str, Any] | None = None,
    expected_input_inventory: Mapping[str, Any] | None = None,
) -> list[InferenceResult]:
    """严格校验 manifest schema、派生 UID 集合及可选的当前运行指纹。"""
    if not isinstance(manifest, dict):
        raise ManifestValidationError("manifest 必须是 object")
    keys = {
        "schema_version", "config", "checkpoint", "allow_partial", "created_at",
        "artifact_fingerprint", "input_inventory", "expected_uids", "ok_uids",
        "skipped_uids", "failed_uids", "results",
    }
    _require_exact_keys(manifest, keys, "manifest")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"不支持的 manifest schema_version: {manifest['schema_version']!r}"
        )
    if not isinstance(manifest["config"], str) or not Path(manifest["config"]).is_absolute():
        raise ManifestValidationError("manifest.config 必须是绝对路径")
    checkpoint = manifest["checkpoint"]
    if checkpoint is not None and (
        not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute()
    ):
        raise ManifestValidationError("manifest.checkpoint 必须为 null 或绝对路径")
    if not isinstance(manifest["allow_partial"], bool):
        raise ManifestValidationError("manifest.allow_partial 必须是 bool")
    if not isinstance(manifest["created_at"], str) or not manifest["created_at"]:
        raise ManifestValidationError("manifest.created_at 必须是非空字符串")

    _validate_content_fingerprint(manifest["artifact_fingerprint"], "artifact_fingerprint")
    artifact_files = manifest["artifact_fingerprint"]["files"]
    config_fingerprint = artifact_files.get("config")
    if config_fingerprint is None or Path(config_fingerprint["path"]).resolve() != Path(manifest["config"]).resolve():
        raise ManifestValidationError("manifest.config 与 artifact fingerprint 不一致")
    checkpoint_fingerprint = artifact_files.get("checkpoint")
    if checkpoint is None:
        if checkpoint_fingerprint is not None:
            raise ManifestValidationError("manifest.checkpoint 与 artifact fingerprint 不一致")
    elif checkpoint_fingerprint is None or Path(checkpoint_fingerprint["path"]).resolve() != Path(checkpoint).resolve():
        raise ManifestValidationError("manifest.checkpoint 与 artifact fingerprint 不一致")

    _validate_input_inventory(manifest["input_inventory"])
    if expected_artifact_fingerprint is not None:
        # JSON 往返会把参数 mapping 的整数键规范化为字符串；比较
        # canonical JSON identity，避免语义相同的 manifest 被错误拒绝。
        if fingerprint_json(manifest["artifact_fingerprint"]) != fingerprint_json(
            dict(expected_artifact_fingerprint)
        ):
            raise ManifestValidationError("artifact fingerprint 与当前运行不一致")
    if expected_input_inventory is not None and manifest["input_inventory"] != dict(expected_input_inventory):
        raise ManifestValidationError("input inventory 与当前运行不一致")

    list_fields = ("expected_uids", "ok_uids", "skipped_uids", "failed_uids")
    for field in list_fields:
        value = manifest[field]
        if not isinstance(value, list) or any(not isinstance(uid, str) or not uid for uid in value):
            raise ManifestValidationError(f"manifest.{field} 必须是非空 UID 构成的 list")
        if len(value) != len(set(value)):
            raise ManifestValidationError(f"manifest.{field} 存在重复 UID")

    inventory_uids = [item["uid"] for item in manifest["input_inventory"]["items"]]
    if manifest["expected_uids"] != inventory_uids:
        raise ManifestValidationError("expected_uids 与 input inventory 顺序不一致")
    if not isinstance(manifest["results"], list):
        raise ManifestValidationError("manifest.results 必须是 list")
    results = [InferenceResult.from_dict(value) for value in manifest["results"]]
    if [result.uid for result in results] != inventory_uids:
        raise ManifestValidationError("results 必须按 expected_uids 顺序完整覆盖")
    for status, field in (("ok", "ok_uids"), ("skipped", "skipped_uids"), ("failed", "failed_uids")):
        actual = [result.uid for result in results if result.status == status]
        if manifest[field] != actual:
            raise ManifestValidationError(f"{field} 与 results 不一致")
    return results


def load_run_manifest(
    path: str | Path,
    *,
    expected_artifact_fingerprint: Mapping[str, Any] | None = None,
    expected_input_inventory: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[InferenceResult]]:
    """读取 JSON 并执行严格 manifest 校验。"""
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"无法读取 manifest {manifest_path}: {exc}") from exc
    results = validate_run_manifest(
        manifest,
        expected_artifact_fingerprint=expected_artifact_fingerprint,
        expected_input_inventory=expected_input_inventory,
    )
    return manifest, results


def reusable_uids_from_manifest(
    path: str | Path,
    *,
    expected_artifact_fingerprint: Mapping[str, Any],
    expected_input_inventory: Mapping[str, Any],
    expected_output_paths: Mapping[str, str | Path],
) -> set[str]:
    """仅返回 manifest 与当前 ``stat-v1`` 输出身份都匹配的成功 UID。"""
    _, results = load_run_manifest(
        path,
        expected_artifact_fingerprint=expected_artifact_fingerprint,
        expected_input_inventory=expected_input_inventory,
    )
    reusable: set[str] = set()
    for result in results:
        if result.status not in {"ok", "skipped"}:
            continue
        expected_path = Path(expected_output_paths[result.uid]).resolve()
        if Path(result.output_path).resolve() != expected_path:
            continue
        try:
            current = stat_audio_identity(expected_path)
        except (OSError, ValueError, RuntimeError):
            continue
        if current == result.output_identity:
            reusable.add(result.uid)
    return reusable
