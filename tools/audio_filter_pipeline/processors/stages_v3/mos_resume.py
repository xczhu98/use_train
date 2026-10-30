"""Step3 MOS 持久分片恢复存储；评分/最终 artifact 发布由调用方集成。"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import logging
import math
import os
import re
import secrets
import stat
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ...utils.artifacts import open_artifact_for_read
from ...utils.strict_jsonl import decode_json_line, encode_json_line
from .runner import DatasetRunLockCapability


__all__ = [
    "ResumeStoreError",
    "SealedShard",
    "Step3ResumeAttempt",
    "Step3ResumeIdentity",
    "Step3ResumeStore",
    "source_boundary_fingerprint",
    "step3_task_identity",
]


_IDENTITY_SCHEMA = "audio_filter_pipeline.step3_resume_identity.v1"
_BOUNDARY_SCHEMA = "audio_filter_pipeline.step3_resume_source_boundary.v1"
_SHARD_ATTESTATION_SCHEMA = "audio_filter_pipeline.step3_resume_shard_attestation.v1"
_SHARD_SEAL_SCHEMA = "audio_filter_pipeline.step3_resume_shard_seal.v1"
_ATTEMPT_SCHEMA = "audio_filter_pipeline.step3_resume_attempt.v1"
_ASSEMBLY_SCHEMA = "audio_filter_pipeline.step3_resume_assembly.v1"
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_SEALED_SHARDS = 4_096
_MAX_SHARD_ROWS = 1_000_000
_MAX_SEALED_BYTES = 16 * 1024 * 1024 * 1024
_MAX_SHARD_BYTES = 512 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_SHARD_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")

_DEFAULT_BOUNDARY_FILES = (
    "processors/stages_v3/mos_gpu_job.py",
    "processors/workers/step3_mos_worker.py",
    "processors/stages_v3/mos_pool.py",
    "processors/stages_v3/mos_runtime.py",
    "processors/stages_v3/source_io.py",
    "processors/stages_v3/select.py",
    "processors/stages_v3/runner.py",
    "processors/stages_v3/base.py",
    "processors/stages_v3/mos_resume.py",
    "utils/artifacts.py",
    "processors/mos.py",
    "processors/mos_contract.py",
    "utils/strict_jsonl.py",
)
_CORE_BOUNDARY_FILE = "core/metrics/mos_models.py"
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
logger = logging.getLogger(__name__)


class ResumeStoreError(RuntimeError):
    """Step3 resume 状态不完整、遭篡改或不具备持久写入权限。"""


def _canonical_json(value: Any, label: str) -> bytes:
    _validate_json(value, label)
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if len(payload) > _MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds the JSON size limit")
    return payload


def _strict_json_copy(value: Any, label: str) -> Any:
    return json.loads(_canonical_json(value, label).decode("utf-8"))


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _validate_json(value: Any, label: str) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite JSON number")
        return
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError(f"{label} object keys must be strings")
        for key, item in value.items():
            _validate_json(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json(item, f"{label}[{index}]")
        return
    raise TypeError(f"{label} must contain only JSON values")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_digest(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256")
    return value


def _require_dataset_id(value: Any, label: str = "dataset_id") -> str:
    if (
        type(value) is not str
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be one nonblank path component")
    return value


def _require_component(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or not value
        or not (value.startswith(".") or _COMPONENT.fullmatch(value) is not None)
    ):
        raise ValueError(f"{label} is not a safe path component")
    return value


def _fsync_directory(path: str) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _renameat2_noreplace(source: str, destination: str) -> int:
    """调用 ``renameat2(RENAME_NOREPLACE)``，返回 0 或 -1 并设置 errno。

    优先 ``os.SYS_renameat2`` 直连 syscall，其次 glibc 的 ``renameat2``
    包装；两者都不可用时 fail-closed，禁止回退到硬编码系统调用号（不同
    架构的 syscall number 不同，猜错会打到无关 syscall）。
    """
    libc = ctypes.CDLL(None, use_errno=True)
    syscall_number = getattr(os, "SYS_renameat2", None)
    if type(syscall_number) is int:
        return libc.syscall(
            syscall_number,
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
        )
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ResumeStoreError(
            "resume sealed publishing requires renameat2 no-replace support"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    return renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )


def _publish_directory_noreplace(source: str, destination: str) -> None:
    """Linux ``renameat2(RENAME_NOREPLACE)``，拒绝会覆盖 sealed shard 的发布。"""
    if os.name != "posix":
        raise ResumeStoreError("resume sealed publishing requires POSIX renameat2")
    result = _renameat2_noreplace(source, destination)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(destination)
    if error in {errno.ENOSYS, errno.EINVAL}:
        raise ResumeStoreError("resume sealed publishing requires renameat2 no-replace support")
    raise OSError(error, os.strerror(error), destination)


def _write_new_file(path: str, payload: bytes, label: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        remaining = memoryview(payload)
        while remaining:
            count = os.write(descriptor, remaining)
            if count <= 0:
                raise OSError("write made no progress")
            remaining = remaining[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(os.path.dirname(path))


def _read_regular(path: str, label: str, maximum: int = _MAX_SHARD_BYTES) -> bytes:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ResumeStoreError(f"{label} cannot be safely opened") from exc
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode) or initial.st_size > maximum:
            raise ResumeStoreError(f"{label} is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = initial.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ResumeStoreError(f"{label} ended before its recorded size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ResumeStoreError(f"{label} grew while being read")
        final = os.fstat(descriptor)
        if (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns, final.st_ctime_ns) != (
            initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns, initial.st_ctime_ns,
        ):
            raise ResumeStoreError(f"{label} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _decode_single_json(path: str, label: str, *, maximum: int = _MAX_JSON_BYTES) -> Any:
    raw = _read_regular(path, label, maximum)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
            parse_float=_parse_finite_float,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResumeStoreError(f"{label} contains invalid strict JSON") from exc
    if _canonical_json(value, label) != raw:
        raise ResumeStoreError(f"{label} is not canonical strict JSON")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON constant is forbidden: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _ensure_directory(path: str, label: str) -> str:
    """逐级创建/验证 0700 非链接目录，返回 canonical directory。"""
    canonical = os.path.realpath(os.path.abspath(path))
    if not os.path.isdir(canonical):
        raise ResumeStoreError(f"{label} is not an existing directory")
    if os.path.islink(path):
        raise ResumeStoreError(f"{label} must not be a symlink")
    return canonical


def _ensure_child_directory(parent: str, name: str, label: str) -> str:
    _require_component(name, f"{label} name")
    path = os.path.join(parent, name)
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    current = os.lstat(path)
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
        raise ResumeStoreError(f"{label} must be a directory, not a link or special file")
    if current.st_uid != os.geteuid():
        raise ResumeStoreError(f"{label} owner UID does not match the current user")
    os.chmod(path, 0o700)
    if os.path.realpath(path) != os.path.join(os.path.realpath(parent), name):
        raise ResumeStoreError(f"{label} path binding changed")
    return path


def _pipeline_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def source_boundary_fingerprint() -> dict[str, Any]:
    """散列 Step3 实际评分、输入、协议及模型实现的固定源码集。"""
    pipeline_root = _pipeline_root()
    repository_root = _repository_root()
    paths = [
        (relative, pipeline_root / relative)
        for relative in _DEFAULT_BOUNDARY_FILES
    ]
    paths.append((_CORE_BOUNDARY_FILE, repository_root / _CORE_BOUNDARY_FILE))
    files: dict[str, str] = {}
    for relative, path in paths:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ResumeStoreError(
                f"resume source boundary file is unreadable: {relative}"
            ) from exc
        files[relative] = _sha256(data)
    payload = {"schema": _BOUNDARY_SCHEMA, "files": files}
    return {**payload, "sha256": _sha256(_canonical_json(payload, "source boundary"))}


def step3_task_identity(
    ordinal: int,
    row: Mapping[str, Any],
    models: Iterable[str],
    scoring_policy_hash: str,
) -> str:
    """返回 select、GPU job 与 sealed shard 共用的严格任务身份。"""
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("Step3 task ordinal is invalid")
    if not isinstance(row, Mapping):
        raise TypeError("Step3 task row must be a mapping")
    _require_digest(scoring_policy_hash, "Step3 task scoring_policy_hash")
    normalized_models = list(models)
    if (
        not normalized_models
        or any(type(model) is not str or not model for model in normalized_models)
        or len(set(normalized_models)) != len(normalized_models)
    ):
        raise ValueError("Step3 task models are invalid")
    payload = encode_json_line(
        {
            "ordinal": ordinal,
            "row": dict(row),
            "models": normalized_models,
            "scoring_policy_hash": scoring_policy_hash,
        },
        "Step3 task identity payload",
        max_line_bytes=_MAX_JSON_BYTES,
    )
    return _sha256(payload)


@dataclass(frozen=True)
class Step3ResumeIdentity:
    """完整 Step3 恢复身份；任一输入、执行或源码边界变化都会分裂 run key。"""

    payload: Mapping[str, Any]
    run_key: str

    @classmethod
    def create(cls, value: Mapping[str, Any]) -> "Step3ResumeIdentity":
        if not isinstance(value, Mapping) or set(value) != {
            "schema", "dataset_id", "step2", "step3_config", "model_provider",
            "execution_policy", "source_boundary",
        }:
            raise ValueError("Step3 resume identity fields are invalid")
        if value["schema"] != _IDENTITY_SCHEMA:
            raise ValueError("Step3 resume identity schema is invalid")
        _require_dataset_id(value["dataset_id"])
        step2 = value["step2"]
        if not isinstance(step2, Mapping) or set(step2) != {
            "artifact_digest", "rows_sha256", "row_count", "lineage_digest",
        }:
            raise ValueError("Step3 resume Step2 identity fields are invalid")
        for field in ("artifact_digest", "rows_sha256", "lineage_digest"):
            _require_digest(step2[field], f"Step3 resume Step2 {field}")
        if type(step2["row_count"]) is not int or not 0 <= step2["row_count"] <= _MAX_SHARD_ROWS * _MAX_SEALED_SHARDS:
            raise ValueError("Step3 resume Step2 row_count is outside bounds")
        for field in ("step3_config", "model_provider", "execution_policy"):
            if not isinstance(value[field], Mapping):
                raise TypeError(f"Step3 resume {field} must be a JSON object")
            _validate_json(value[field], f"Step3 resume {field}")
        observed_boundary = value["source_boundary"]
        if not isinstance(observed_boundary, Mapping):
            raise TypeError("Step3 resume source_boundary must be a mapping")
        expected_boundary = source_boundary_fingerprint()
        if _strict_json_copy(observed_boundary, "Step3 resume source_boundary") != expected_boundary:
            raise ResumeStoreError("Step3 resume source boundary fingerprint mismatch")
        payload = _strict_json_copy(value, "Step3 resume identity")
        return cls(
            payload=_freeze_json(payload),
            run_key=_sha256(_canonical_json(payload, "Step3 resume identity")),
        )

    def verify_source_boundary(self) -> None:
        if _plain_json(self.payload["source_boundary"]) != source_boundary_fingerprint():
            raise ResumeStoreError("Step3 resume source boundary changed after identity creation")


@dataclass(frozen=True)
class SealedShard:
    shard_id: str
    descriptor: Mapping[str, Any]
    runtime_attestation: Mapping[str, Any] | None
    rows_sha256: str
    rows_count: int
    rows_bytes: int
    attestation_sha256: str
    seal_sha256: str
    path: str


class Step3ResumeAttempt:
    """一个随机 staging 目录的唯一写者；close/abort 仅清理该 attempt。"""

    def __init__(self, store: "Step3ResumeStore", capability: DatasetRunLockCapability, token: str, path: str) -> None:
        self._store = store
        self._capability = capability
        self._token = token
        self._path = path
        self._closed = False

    @property
    def path(self) -> str:
        return self._path

    def seal_shard(
        self,
        descriptor: Mapping[str, Any],
        rows: Iterable[Mapping[str, Any]],
        runtime_attestation: Mapping[str, Any] | None = None,
    ) -> SealedShard:
        if self._closed:
            raise ResumeStoreError("resume attempt is closed")
        return self._store._seal_shard(
            self, descriptor, rows, runtime_attestation,
        )
    def abort(self) -> None:
        if self._closed:
            return
        self._store._abort_attempt(self)
        self._closed = True

    close = abort

    def __enter__(self) -> "Step3ResumeAttempt":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.abort()
        return False


class Step3ResumeStore:
    """受 dataset lock capability 约束的 Step3 resume storage API。"""

    def __init__(
        self,
        output_root: str,
        dataset_id: str,
        identity: Step3ResumeIdentity,
    ) -> None:
        if not isinstance(identity, Step3ResumeIdentity):
            raise TypeError("identity must be a Step3ResumeIdentity")
        self.output_root = _ensure_directory(output_root, "resume output_root")
        self.dataset_id = _require_dataset_id(dataset_id)
        if identity.payload["dataset_id"] != self.dataset_id:
            raise ValueError("resume identity dataset_id does not match store")
        identity.verify_source_boundary()
        self.identity = identity
        self._base_dir = os.path.join(self.output_root, ".audio_filter_pipeline_resume")
        self._step3_dir = os.path.join(self._base_dir, "step3")
        encoded = "d-" + self.dataset_id.encode("utf-8").hex()
        self._dataset_dir = os.path.join(self._step3_dir, encoded)
        self.run_dir = os.path.join(self._dataset_dir, identity.run_key)
        self.staging_dir = os.path.join(self.run_dir, "staging")
        self.sealed_dir = os.path.join(self.run_dir, "sealed")
        self._validate_existing_layout()

    def _validate_existing_layout(self) -> None:
        for parent, path, name, label in (
            (self.output_root, self._base_dir, ".audio_filter_pipeline_resume", "resume root"),
            (self._base_dir, self._step3_dir, "step3", "resume Step3 root"),
            (self._step3_dir, self._dataset_dir, os.path.basename(self._dataset_dir), "resume dataset root"),
            (self._dataset_dir, self.run_dir, self.identity.run_key, "resume run root"),
            (self.run_dir, self.staging_dir, "staging", "resume staging root"),
            (self.run_dir, self.sealed_dir, "sealed", "resume sealed root"),
        ):
            _require_component(name, f"{label} name")
            if os.path.lexists(path):
                entry = os.lstat(path)
                if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
                    raise ResumeStoreError(f"{label} must be a directory, not a link or special file")
                if entry.st_uid != os.geteuid():
                    raise ResumeStoreError(f"{label} owner UID does not match the current user")
                if os.path.realpath(path) != os.path.join(os.path.realpath(parent), name):
                    raise ResumeStoreError(f"{label} path binding changed")

    def _initialize_layout(self, capability: DatasetRunLockCapability) -> None:
        self._validate_capability(capability)
        base = _ensure_child_directory(self.output_root, ".audio_filter_pipeline_resume", "resume root")
        step3 = _ensure_child_directory(base, "step3", "resume Step3 root")
        dataset = _ensure_child_directory(step3, os.path.basename(self._dataset_dir), "resume dataset root")
        run = _ensure_child_directory(dataset, self.identity.run_key, "resume run root")
        staging = _ensure_child_directory(run, "staging", "resume staging root")
        sealed = _ensure_child_directory(run, "sealed", "resume sealed root")
        if (run, staging, sealed) != (self.run_dir, self.staging_dir, self.sealed_dir):
            raise ResumeStoreError("resume layout binding changed")

    @staticmethod
    def identity_payload(
        *,
        dataset_id: str,
        step2_artifact_digest: str,
        step2_rows_sha256: str,
        step2_row_count: int,
        step2_lineage_digest: str,
        step3_config: Mapping[str, Any],
        model_provider: Mapping[str, Any],
        execution_policy: Mapping[str, Any],
    ) -> dict[str, Any]:
        """构造供 ``Step3ResumeIdentity.create`` 使用的完整严格身份。"""
        return {
            "schema": _IDENTITY_SCHEMA,
            "dataset_id": dataset_id,
            "step2": {
                "artifact_digest": step2_artifact_digest,
                "rows_sha256": step2_rows_sha256,
                "row_count": step2_row_count,
                "lineage_digest": step2_lineage_digest,
            },
            "step3_config": dict(step3_config),
            "model_provider": dict(model_provider),
            "execution_policy": dict(execution_policy),
            "source_boundary": source_boundary_fingerprint(),
        }

    def _validate_capability(self, capability: DatasetRunLockCapability) -> None:
        if not isinstance(capability, DatasetRunLockCapability):
            raise TypeError("persistent resume writes require DatasetRunLockCapability")
        self.identity.verify_source_boundary()
        capability.validate(self.output_root, self.dataset_id)

    def begin_attempt(self, capability: DatasetRunLockCapability) -> Step3ResumeAttempt:
        self._initialize_layout(capability)
        for _ in range(32):
            token = secrets.token_hex(24)
            path = os.path.join(self.staging_dir, f"attempt-{token}")
            try:
                os.mkdir(path, 0o700)
            except FileExistsError:
                continue
            marker = {
                "schema": _ATTEMPT_SCHEMA,
                "run_key": self.identity.run_key,
                "token": token,
            }
            try:
                _write_new_file(
                    os.path.join(path, "attempt.json"),
                    _canonical_json(marker, "resume attempt marker"),
                    "resume attempt marker",
                )
                _fsync_directory(self.staging_dir)
            except BaseException:
                self._remove_owned_attempt(path, token)
                raise
            return Step3ResumeAttempt(self, capability, token, path)
        raise ResumeStoreError("cannot create a unique resume attempt")

    def _validate_attempt(self, attempt: Step3ResumeAttempt) -> None:
        if not isinstance(attempt, Step3ResumeAttempt) or attempt._store is not self:
            raise TypeError("resume attempt belongs to another store")
        self._validate_capability(attempt._capability)
        if os.path.realpath(attempt._path) != os.path.join(self.staging_dir, os.path.basename(attempt._path)):
            raise ResumeStoreError("resume attempt path binding changed")
        marker = _decode_single_json(
            os.path.join(attempt._path, "attempt.json"), "resume attempt marker",
        )
        if marker != {
            "schema": _ATTEMPT_SCHEMA,
            "run_key": self.identity.run_key,
            "token": attempt._token,
        }:
            raise ResumeStoreError("resume attempt ownership marker mismatch")

    @staticmethod
    def _validated_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
        required_fields = {
            "shard_id", "ordinals", "task_identities", "models",
            "scoring_policy_hash", "task_bindings",
        }
        if not isinstance(value, Mapping) or set(value) != required_fields:
            # 缺 task_bindings 的旧 descriptor 不再兼容读取：直接拒绝，要求
            # 删除 staging 下的旧 resume 状态后重跑，而不是静默按旧语义恢复。
            raise ValueError("resume shard descriptor fields are invalid")
        shard_id = value["shard_id"]
        if type(shard_id) is not str or _SHARD_NAME.fullmatch(shard_id) is None:
            raise ValueError("resume shard_id is invalid")
        ordinals = value["ordinals"]
        identities = value["task_identities"]
        if (
            not isinstance(ordinals, list)
            or not isinstance(identities, list)
            or not ordinals
            or len(ordinals) != len(identities)
            or len(ordinals) > _MAX_SHARD_ROWS
            or any(type(item) is not int or item < 0 for item in ordinals)
            or any(left >= right for left, right in zip(ordinals, ordinals[1:]))
        ):
            raise ValueError("resume shard ordinals are invalid")
        for identity in identities:
            _require_digest(identity, "resume shard task identity")
        models = value["models"]
        if (
            not isinstance(models, list)
            or not models
            or any(type(model) is not str or not model for model in models)
            or len(set(models)) != len(models)
        ):
            raise ValueError("resume shard models are invalid")
        _require_digest(value["scoring_policy_hash"], "resume shard scoring_policy_hash")
        bindings = value["task_bindings"]
        if not isinstance(bindings, list) or len(bindings) != len(ordinals):
            raise ValueError("resume shard task_bindings are invalid")
        for binding in bindings:
            if not isinstance(binding, Mapping) or set(binding) != {
                "models", "scoring_policy_hash",
            }:
                raise ValueError("resume shard task binding fields are invalid")
            binding_models = binding["models"]
            if (
                not isinstance(binding_models, list)
                or not binding_models
                or any(type(model) is not str or not model for model in binding_models)
                or len(set(binding_models)) != len(binding_models)
            ):
                raise ValueError("resume shard task binding models are invalid")
            _require_digest(
                binding["scoring_policy_hash"],
                "resume shard task binding scoring_policy_hash",
            )
        return _strict_json_copy(value, "resume shard descriptor")

    @staticmethod
    def _validated_row(value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping) or set(value) != {"ordinal", "task_identity", "payload"}:
            raise ValueError("resume shard row fields are invalid")
        if type(value["ordinal"]) is not int or value["ordinal"] < 0:
            raise ValueError("resume shard row ordinal is invalid")
        _require_digest(value["task_identity"], "resume shard row task_identity")
        _validate_json(value["payload"], "resume shard row payload")
        return _strict_json_copy(value, "resume shard row")

    def _sealed_name(self, descriptor: Mapping[str, Any]) -> str:
        descriptor_hash = _sha256(_canonical_json(descriptor, "resume shard descriptor"))
        return f"{descriptor['shard_id']}--{descriptor_hash}"

    def _seal_shard(
        self,
        attempt: Step3ResumeAttempt,
        descriptor_value: Mapping[str, Any],
        rows: Iterable[Mapping[str, Any]],
        runtime_attestation: Mapping[str, Any] | None,
    ) -> SealedShard:
        self._validate_attempt(attempt)
        descriptor = self._validated_descriptor(descriptor_value)
        if runtime_attestation is not None:
            if not isinstance(runtime_attestation, Mapping) or not runtime_attestation:
                raise ValueError("resume shard runtime_attestation is invalid")
            _validate_json(runtime_attestation, "resume shard runtime_attestation")
        if not isinstance(rows, Iterable):
            raise TypeError("resume shard rows must be iterable")
        expected_ordinals = descriptor["ordinals"]
        expected_identities = descriptor["task_identities"]
        encoded_rows: list[bytes] = []
        for index, value in enumerate(rows):
            if index >= len(expected_ordinals):
                raise ResumeStoreError("resume shard contains too many rows")
            row = self._validated_row(value)
            if (
                row["ordinal"] != expected_ordinals[index]
                or row["task_identity"] != expected_identities[index]
            ):
                raise ResumeStoreError("resume shard row is detached from task identity")
            encoded = encode_json_line(row, "resume shard row", max_line_bytes=_MAX_JSON_BYTES)
            if len(encoded) > _MAX_JSON_BYTES:
                raise ResumeStoreError("resume shard row exceeds byte limit")
            encoded_rows.append(encoded)
        if len(encoded_rows) != len(expected_ordinals):
            raise ResumeStoreError("resume shard row count is incomplete")
        rows_payload = b"".join(encoded_rows)
        if len(rows_payload) > _MAX_SHARD_BYTES:
            raise ResumeStoreError("resume shard exceeds byte limit")
        rows_hash = _sha256(rows_payload)
        name = self._sealed_name(descriptor)
        existing = os.path.join(self.sealed_dir, name)
        if os.path.lexists(existing):
            sealed = self._read_sealed(existing)
            if (
                sealed.descriptor == descriptor
                and sealed.rows_sha256 == rows_hash
                and sealed.rows_count == len(encoded_rows)
                and sealed.rows_bytes == len(rows_payload)
            ):
                return sealed
            raise ResumeStoreError("resume sealed shard name collides with different content")

        staging_name = f"shard-{name}-{secrets.token_hex(12)}"
        staging = os.path.join(attempt._path, staging_name)
        os.mkdir(staging, 0o700)
        try:
            attestation = {
                "schema": _SHARD_ATTESTATION_SCHEMA,
                "identity_sha256": self.identity.run_key,
                "descriptor": descriptor,
                "rows_sha256": rows_hash,
                "rows_count": len(encoded_rows),
                "rows_bytes": len(rows_payload),
                "runtime_attestation": (
                    _strict_json_copy(runtime_attestation, "resume shard runtime_attestation")
                    if runtime_attestation is not None else None
                ),
            }
            attestation_payload = _canonical_json(attestation, "resume shard attestation")
            seal = {
                "schema": _SHARD_SEAL_SCHEMA,
                "identity_sha256": self.identity.run_key,
                "rows_sha256": rows_hash,
                "attestation_sha256": _sha256(attestation_payload),
            }
            _write_new_file(os.path.join(staging, "rows.jsonl"), rows_payload, "resume shard rows")
            _write_new_file(os.path.join(staging, "attestation.json"), attestation_payload, "resume shard attestation")
            _write_new_file(
                os.path.join(staging, "seal.json"),
                _canonical_json(seal, "resume shard seal"),
                "resume shard seal",
            )
            _fsync_directory(staging)
            try:
                _publish_directory_noreplace(staging, existing)
            except FileExistsError:
                sealed = self._read_sealed(existing)
                if (
                    sealed.descriptor == descriptor
                    and sealed.rows_sha256 == rows_hash
                    and sealed.rows_count == len(encoded_rows)
                    and sealed.rows_bytes == len(rows_payload)
                ):
                    return sealed
                raise ResumeStoreError("resume sealed shard publish collision")
            _fsync_directory(self.sealed_dir)
            return self._read_sealed(existing)
        finally:
            if os.path.lexists(staging):
                self._remove_owned_tree(staging)

    def _read_sealed(self, path: str) -> SealedShard:
        self.identity.verify_source_boundary()
        entry = os.lstat(path)
        if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
            raise ResumeStoreError("resume sealed shard is not a directory")
        if entry.st_uid != os.geteuid():
            raise ResumeStoreError("resume sealed shard owner UID mismatch")
        names = set(os.listdir(path))
        if names != {"rows.jsonl", "attestation.json", "seal.json"}:
            raise ResumeStoreError("resume sealed shard files are incomplete")
        attestation_raw = _read_regular(os.path.join(path, "attestation.json"), "resume shard attestation")
        attestation = _decode_single_json(os.path.join(path, "attestation.json"), "resume shard attestation")
        if not isinstance(attestation, Mapping) or set(attestation) != {
            "schema", "identity_sha256", "descriptor", "rows_sha256", "rows_count", "rows_bytes", "runtime_attestation",
        } or attestation["schema"] != _SHARD_ATTESTATION_SCHEMA:
            raise ResumeStoreError("resume shard attestation schema is invalid")
        if attestation["identity_sha256"] != self.identity.run_key:
            raise ResumeStoreError("resume shard identity mismatch")
        descriptor = self._validated_descriptor(attestation["descriptor"])
        runtime_attestation = attestation["runtime_attestation"]
        if runtime_attestation is not None:
            if not isinstance(runtime_attestation, Mapping) or not runtime_attestation:
                raise ResumeStoreError("resume shard runtime attestation is invalid")
            _validate_json(runtime_attestation, "resume shard runtime attestation")
        rows_sha = _require_digest(attestation["rows_sha256"], "resume shard rows_sha256")
        count, byte_count, parsed_rows, rows_payload = self._verify_rows(
            os.path.join(path, "rows.jsonl"), descriptor,
        )
        if (
            attestation["rows_count"] != count
            or attestation["rows_bytes"] != byte_count
            or rows_sha != _sha256(rows_payload)
            or parsed_rows != count
        ):
            raise ResumeStoreError("resume shard rows attestation mismatch")
        seal_raw = _read_regular(os.path.join(path, "seal.json"), "resume shard seal", _MAX_JSON_BYTES)
        seal = _decode_single_json(os.path.join(path, "seal.json"), "resume shard seal")
        if seal != {
            "schema": _SHARD_SEAL_SCHEMA,
            "identity_sha256": self.identity.run_key,
            "rows_sha256": rows_sha,
            "attestation_sha256": _sha256(attestation_raw),
        }:
            raise ResumeStoreError("resume shard seal mismatch")
        expected_name = self._sealed_name(descriptor)
        if os.path.basename(path) != expected_name:
            raise ResumeStoreError("resume sealed shard pathname does not bind its descriptor")
        return SealedShard(
            shard_id=descriptor["shard_id"],
            descriptor=descriptor,
            runtime_attestation=(
                _freeze_json(runtime_attestation)
                if runtime_attestation is not None else None
            ),
            rows_sha256=rows_sha,
            rows_count=count,
            rows_bytes=byte_count,
            attestation_sha256=_sha256(attestation_raw),
            seal_sha256=_sha256(seal_raw),
            path=path,
        )

    def _verify_rows(
        self, path: str, descriptor: Mapping[str, Any],
    ) -> tuple[int, int, int, bytes]:
        payload = _read_regular(path, "resume shard rows")
        expected_ordinals = descriptor["ordinals"]
        expected_identities = descriptor["task_identities"]
        parsed = 0
        for index, line in enumerate(payload.splitlines(keepends=True)):
            if index >= len(expected_ordinals):
                raise ResumeStoreError("resume sealed shard has extra rows")
            try:
                value = decode_json_line(line, "resume shard row", max_line_bytes=_MAX_JSON_BYTES)
            except ValueError as exc:
                raise ResumeStoreError("resume sealed shard row is invalid") from exc
            try:
                row = self._validated_row(value)
            except (TypeError, ValueError) as exc:
                raise ResumeStoreError("resume sealed shard row schema is invalid") from exc
            if row["ordinal"] != expected_ordinals[index] or row["task_identity"] != expected_identities[index]:
                raise ResumeStoreError("resume sealed shard row identity mismatch")
            if encode_json_line(row, "resume shard row", max_line_bytes=_MAX_JSON_BYTES) != line:
                raise ResumeStoreError("resume sealed shard row is not canonical")
            parsed += 1
        if parsed != len(expected_ordinals):
            raise ResumeStoreError("resume sealed shard rows are incomplete")
        return parsed, len(payload), parsed, payload

    def sealed_shards(self) -> tuple[SealedShard, ...]:
        self.identity.verify_source_boundary()
        if not os.path.lexists(self.sealed_dir):
            return ()
        names = sorted(os.listdir(self.sealed_dir))
        if len(names) > _MAX_SEALED_SHARDS:
            raise ResumeStoreError("resume sealed shard count exceeds limit")
        shards = [self._read_sealed(os.path.join(self.sealed_dir, name)) for name in names]
        shards.sort(key=lambda shard: shard.descriptor["ordinals"][0])
        return tuple(shards)

    def iter_sealed_rows(self) -> Iterator[tuple[SealedShard, dict[str, Any]]]:
        """读取全部已 seal 的行；调用方必须与当前任务身份逐行绑定。"""
        for shard in self.sealed_shards():
            payload = _read_regular(
                os.path.join(shard.path, "rows.jsonl"), "resume shard rows",
            )
            for line in payload.splitlines(keepends=True):
                yield shard, self._validated_row(decode_json_line(
                    line, "resume shard row", max_line_bytes=_MAX_JSON_BYTES,
                ))

    def complete_shards(self) -> tuple[SealedShard, ...]:
        # LPT 调度器按权重把任务交错分配到各 worker，sealed shard 的 ordinals
        # 因此不是连续块；覆盖率必须按多重集校验，不能依赖 shard 拼接顺序。
        shards = self.sealed_shards()
        observed: list[int] = []
        total_bytes = 0
        for shard in shards:
            observed.extend(shard.descriptor["ordinals"])
            total_bytes += shard.rows_bytes
        expected_count = self.identity.payload["step2"]["row_count"]
        if total_bytes > _MAX_SEALED_BYTES:
            raise ResumeStoreError("resume sealed byte total exceeds limit")
        if len(observed) != expected_count:
            raise ResumeStoreError("resume sealed shards do not provide complete global ordinal coverage")
        observed.sort()
        if any(index != ordinal for index, ordinal in enumerate(observed)):
            raise ResumeStoreError("resume sealed shards do not provide complete global ordinal coverage")
        return shards

    def iter_complete_rows(self) -> Iterator[dict[str, Any]]:
        for shard in self.complete_shards():
            payload = _read_regular(os.path.join(shard.path, "rows.jsonl"), "resume shard rows")
            for line in payload.splitlines(keepends=True):
                yield self._validated_row(decode_json_line(
                    line, "resume shard row", max_line_bytes=_MAX_JSON_BYTES,
                ))

    def sealed_manifest_digest(self) -> str:
        """以逐 shard canonical JSON 行的流式 digest 绑定完整 sealed manifest。

        单 shard descriptor 可接近 `_MAX_JSON_BYTES`，把所有 shard 的完整
        descriptor 塞进单个 JSON 对象会超出 8 MiB 上限；逐行散列不受 shard
        数量与 descriptor 大小限制，且对同一 sealed 集合保持确定性。
        """
        digest = hashlib.sha256()
        for shard in self.complete_shards():
            record = {
                "shard_id": shard.shard_id,
                "descriptor": shard.descriptor,
                "rows_sha256": shard.rows_sha256,
                "rows_count": shard.rows_count,
                "rows_bytes": shard.rows_bytes,
                "attestation_sha256": shard.attestation_sha256,
                "seal_sha256": shard.seal_sha256,
            }
            digest.update(_canonical_json(
                record, "resume sealed shard manifest record",
            ))
            digest.update(b"\n")
        return digest.hexdigest()

    @staticmethod
    def _validated_assembly(value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping) or set(value) != {
            "artifact_hash", "artifact_path", "config_digest", "lineage_digest", "scorer_identity_sha256", "sidecar_path", "proof",
        }:
            raise ValueError("resume assembly fields are invalid")
        for field in ("artifact_hash", "config_digest", "lineage_digest", "scorer_identity_sha256"):
            _require_digest(value[field], f"resume assembly {field}")
        if type(value["artifact_path"]) is not str or not os.path.isabs(value["artifact_path"]):
            raise ValueError("resume assembly artifact_path must be absolute")
        if type(value["sidecar_path"]) is not str or not os.path.isabs(value["sidecar_path"]):
            raise ValueError("resume assembly sidecar_path must be absolute")
        if value["proof"] is not None:
            _validate_json(value["proof"], "resume assembly proof")
        return _strict_json_copy(value, "resume assembly")

    def _assembly_path(self) -> str:
        return os.path.join(self.run_dir, "assembly.json")

    def record_assembly_attestation(
        self, capability: DatasetRunLockCapability, assembly_value: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """在 artifact rename 前持久化、与完整 sealed manifest 绑定的 proof。"""
        self._initialize_layout(capability)
        assembly = self._validated_assembly(assembly_value)
        expected_path = os.path.join(self.output_root, "step3", self.dataset_id)
        if os.path.realpath(assembly["artifact_path"]) != os.path.realpath(expected_path):
            raise ResumeStoreError("resume assembly artifact path is outside the Step3 target")
        expected_sidecar = (
            f"{expected_path}.{assembly['artifact_hash']}.execution-attestation.json"
        )
        if os.path.realpath(assembly["sidecar_path"]) != os.path.realpath(expected_sidecar):
            raise ResumeStoreError("resume assembly sidecar path is outside the Step3 target")
        record = {
            "schema": _ASSEMBLY_SCHEMA,
            "identity_sha256": self.identity.run_key,
            "source_boundary": _plain_json(self.identity.payload["source_boundary"]),
            "sealed_manifest_sha256": self.sealed_manifest_digest(),
            "assembly": assembly,
        }
        path = self._assembly_path()
        payload = _canonical_json(record, "resume assembly record")
        try:
            _write_new_file(path, payload, "resume assembly record")
        except FileExistsError:
            existing = _decode_single_json(path, "resume assembly record")
            if existing != record:
                raise ResumeStoreError("resume assembly record collision")
        return record

    def _verify_published_artifact(self, assembly: Mapping[str, Any]) -> None:
        expected_path = os.path.join(self.output_root, "step3", self.dataset_id)
        if os.path.realpath(assembly["artifact_path"]) != os.path.realpath(expected_path):
            raise ResumeStoreError("resume assembly artifact path is outside the Step3 target")
        try:
            index = open_artifact_for_read(
                self.output_root,
                "step3",
                self.dataset_id,
                expected_config_digest=assembly["config_digest"],
                expected_lineage_digest=assembly["lineage_digest"],
            )
        except BaseException as exc:
            raise ResumeStoreError("resume assembly final artifact cannot be verified") from exc
        if index.artifact_digest != assembly["artifact_hash"]:
            raise ResumeStoreError("resume assembly final artifact hash mismatch")

    def recover_assembly_attestation(self, assembly_value: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """仅验证并返回可补发 sidecar 的 proof；本模块从不发布 artifact/sidecar。"""
        self.identity.verify_source_boundary()
        path = self._assembly_path()
        if not os.path.lexists(path):
            return None
        record = _decode_single_json(path, "resume assembly record")
        requested = self._validated_assembly(assembly_value)
        if not isinstance(record, Mapping) or set(record) != {
            "schema", "identity_sha256", "source_boundary", "sealed_manifest_sha256", "assembly",
        }:
            raise ResumeStoreError("resume assembly record schema is invalid")
        assembly = self._validated_assembly(record["assembly"])
        if any(
            requested[field] != assembly[field]
            for field in assembly if field != "proof"
        ) or (requested["proof"] is not None and requested["proof"] != assembly["proof"]):
            raise ResumeStoreError("resume assembly record binding mismatch")
        expected = {
            "schema": _ASSEMBLY_SCHEMA,
            "identity_sha256": self.identity.run_key,
            "source_boundary": _plain_json(self.identity.payload["source_boundary"]),
            "sealed_manifest_sha256": self.sealed_manifest_digest(),
            "assembly": assembly,
        }
        if record != expected:
            raise ResumeStoreError("resume assembly record binding mismatch")
        self._verify_published_artifact(assembly)
        if os.path.lexists(assembly["sidecar_path"]):
            return None
        return {
            "proof": assembly["proof"],
            "sealed_manifest_sha256": expected["sealed_manifest_sha256"],
            "identity_sha256": self.identity.run_key,
        }

    def consume_assembly_attestation(
        self, capability: DatasetRunLockCapability, assembly_value: Mapping[str, Any],
    ) -> None:
        self._validate_capability(capability)
        if self.recover_assembly_attestation(assembly_value) is not None:
            raise ResumeStoreError("assembly sidecar is still missing; record cannot be consumed")
        path = self._assembly_path()
        if os.path.lexists(path):
            current = _decode_single_json(path, "resume assembly record")
            if current.get("identity_sha256") != self.identity.run_key:
                raise ResumeStoreError("resume assembly record ownership mismatch")
            os.unlink(path)
            _fsync_directory(self.run_dir)

    def _abort_attempt(self, attempt: Step3ResumeAttempt) -> None:
        self._validate_attempt(attempt)
        self._remove_owned_attempt(attempt._path, attempt._token)
        _fsync_directory(self.staging_dir)

    def _remove_owned_attempt(self, path: str, token: str) -> None:
        if not os.path.lexists(path):
            return
        if os.path.realpath(path) != os.path.join(self.staging_dir, os.path.basename(path)):
            raise ResumeStoreError("refusing to remove an attempt outside the staging root")
        marker = _decode_single_json(os.path.join(path, "attempt.json"), "resume attempt marker")
        if marker != {"schema": _ATTEMPT_SCHEMA, "run_key": self.identity.run_key, "token": token}:
            raise ResumeStoreError("refusing to remove a non-owned resume attempt")
        self._remove_owned_tree(path)

    @staticmethod
    def _remove_owned_tree(path: str) -> None:
        entry = os.lstat(path)
        if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
            raise ResumeStoreError("refusing to remove a linked or non-directory owned path")
        for name in os.listdir(path):
            child = os.path.join(path, name)
            child_stat = os.lstat(child)
            if stat.S_ISLNK(child_stat.st_mode):
                raise ResumeStoreError("refusing to remove an owned tree containing a symlink")
            if stat.S_ISDIR(child_stat.st_mode):
                Step3ResumeStore._remove_owned_tree(child)
            elif stat.S_ISREG(child_stat.st_mode):
                os.unlink(child)
            else:
                raise ResumeStoreError("refusing to remove an owned tree containing a special file")
        os.rmdir(path)
