"""三路验证脚本共享小工具（JAE-M05 / JAE-M10）。

- ``load_checkpoint_strict``：checkpoint 完整性校验，直接复用
  ``onnx.export.load_checkpoint`` 的既有实现（不在此处复制逻辑），防止
  "checkpoint 部分权重随机初始化也能三路一致（三路同错）"的假 PASS。
- ``load_checkpoint_training_config`` / ``require_checkpoint_config_provenance``：
  绑定 checkpoint 同目录 sidecar 或内嵌 resolved config，拒绝 verifier 配置漂移。
- ``sha256_file`` / ``print_kept_onnx_artifact``：显式保留或已发布的 ONNX
  输出路径 + SHA256 供审计复核；lifecycle 所有的临时 artifact 在退出时清理。
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import io
import json
import logging
import math
import os
import secrets
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Mapping

import torch


logger = logging.getLogger(__name__)

_ONNX_RELEASE_ROOT = Path(__file__).resolve().parents[1] / "onnx_model"


@dataclass(frozen=True)
class ThreeWayComparison:
    """单个三路比较项的结构化结果。"""

    name: str
    mse: float
    max_abs: float
    compared_count: int
    threshold: float
    status: str

    @property
    def passed(self) -> bool:
        count_valid = (
            isinstance(self.compared_count, int)
            and not isinstance(self.compared_count, bool)
            and self.compared_count > 0
        )
        threshold_valid = (
            isinstance(self.threshold, (int, float))
            and not isinstance(self.threshold, bool)
            and math.isfinite(float(self.threshold))
            and float(self.threshold) > 0
        )
        metrics_valid = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in (self.mse, self.max_abs)
        )
        return (
            self.status == "PASS"
            and count_valid
            and threshold_valid
            and metrics_valid
            and float(self.mse) < float(self.threshold)
        )


@dataclass(frozen=True)
class ThreeWayResult:
    """A/B/C 比较的统一发布 gate 输入。"""

    offline_vs_streaming_pt: ThreeWayComparison
    streaming_pt_vs_ort: ThreeWayComparison
    offline_vs_ort: ThreeWayComparison
    extra_gates: Mapping[str, bool] | None = None

    @property
    def comparisons(self) -> tuple[ThreeWayComparison, ...]:
        return (
            self.offline_vs_streaming_pt,
            self.streaming_pt_vs_ort,
            self.offline_vs_ort,
        )

    @property
    def all_pass(self) -> bool:
        return all(item.passed for item in self.comparisons) and all(
            bool(value) for value in (self.extra_gates or {}).values()
        )


class OwnedOnnxLifecycle:
    """以稳定 parent/file FD 跟踪 verifier 临时 artifact ownership。

    默认在 context 退出时清理 owned 临时文件；仅调用 ``keep()`` 的 artifact
    由调用方保留，正式发布副本由发布事务单独管理。
    """

    def __init__(self) -> None:
        self.path: Path | None = None
        self.kept = False
        self._file_fd: int | None = None
        self._parent_fd: int | None = None
        self._name: str | None = None

    def own(self, path) -> Path:
        if self.path is not None:
            raise RuntimeError("ONNX lifecycle already owns an artifact")
        artifact = Path(path)
        file_fd = _open_regular_file(artifact, label="owned ONNX artifact")
        parent_flags = (
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        parent_fd: int | None = None
        try:
            parent_fd = os.open(artifact.parent, parent_flags)
            _require_anchored_entry(
                parent_fd,
                artifact.name,
                os.fstat(file_fd),
                label="owned ONNX artifact",
            )
        except Exception:
            if parent_fd is not None:
                os.close(parent_fd)
            os.close(file_fd)
            raise
        self.path = artifact
        self._file_fd = file_fd
        self._parent_fd = parent_fd
        self._name = artifact.name
        return artifact

    def keep(self) -> None:
        self.kept = True

    def cleanup(self) -> None:
        try:
            if (
                not self.kept
                and self._file_fd is not None
                and self._parent_fd is not None
                and self._name is not None
            ):
                quarantine = _random_artifact_name("onnx-cleanup")
                try:
                    os.replace(
                        self._name,
                        quarantine,
                        src_dir_fd=self._parent_fd,
                        dst_dir_fd=self._parent_fd,
                    )
                except FileNotFoundError:
                    pass
                else:
                    quarantined = os.stat(
                        quarantine,
                        dir_fd=self._parent_fd,
                        follow_symlinks=False,
                    )
                    if _same_file_identity(quarantined, os.fstat(self._file_fd)):
                        os.unlink(quarantine, dir_fd=self._parent_fd)
                    else:
                        # 非 lifecycle 所有的 ABA 替换项必须恢复原名，不得删除。
                        os.replace(
                            quarantine,
                            self._name,
                            src_dir_fd=self._parent_fd,
                            dst_dir_fd=self._parent_fd,
                        )
        finally:
            if self._file_fd is not None:
                os.close(self._file_fd)
                self._file_fd = None
            if self._parent_fd is not None:
                os.close(self._parent_fd)
                self._parent_fd = None


@contextmanager
def owned_onnx_cleanup():
    lifecycle = OwnedOnnxLifecycle()
    try:
        yield lifecycle
    finally:
        lifecycle.cleanup()


@dataclass(frozen=True)
class OnnxArtifactFingerprint:
    """绑定 ONNX 字节与持有文件描述符身份的不可变指纹。"""

    size_bytes: int
    sha256: str
    device: int
    inode: int


def _require_regular_file(file_stat, *, label: str, allow_unlinked: bool = False) -> None:
    if not stat.S_ISREG(file_stat.st_mode):
        raise RuntimeError(f"{label} must be a regular file")
    expected_links = {0, 1} if allow_unlinked else {1}
    if file_stat.st_nlink not in expected_links:
        raise RuntimeError(f"{label} must not be a hard link (nlink={file_stat.st_nlink})")


def _fingerprint_fd(fd: int, *, label: str, chunk_size: int = 1 << 20) -> OnnxArtifactFingerprint:
    """从同一持有 FD 计算摘要，并拒绝读取期间的 inode 内容漂移。"""
    before = os.fstat(fd)
    _require_regular_file(before, label=label, allow_unlinked=True)
    digest = hashlib.sha256()
    bytes_read = 0
    while bytes_read < before.st_size:
        chunk = os.pread(fd, min(chunk_size, before.st_size - bytes_read), bytes_read)
        if not chunk:
            break
        digest.update(chunk)
        bytes_read += len(chunk)
    after = os.fstat(fd)
    stable_fields = (
        "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink",
    )
    if (
        any(getattr(before, field) != getattr(after, field) for field in stable_fields)
        or bytes_read != before.st_size
    ):
        raise RuntimeError(f"{label} changed while fingerprinting")
    return OnnxArtifactFingerprint(
        size_bytes=bytes_read,
        sha256=digest.hexdigest(),
        device=after.st_dev,
        inode=after.st_ino,
    )


def _read_fd_bytes(fd: int, size_bytes: int, chunk_size: int = 1 << 20) -> bytes:
    chunks = []
    offset = 0
    while offset < size_bytes:
        chunk = os.pread(fd, min(chunk_size, size_bytes - offset), offset)
        if not chunk:
            raise RuntimeError("verified ONNX snapshot became truncated")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _open_regular_file(path, *, label: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"failed to open {label} without following symlinks: {path}") from exc
    try:
        _require_regular_file(os.fstat(fd), label=label)
    except Exception:
        os.close(fd)
        raise
    return fd


def _copy_fd(source_fd: int, target_fd: int, chunk_size: int = 1 << 20) -> None:
    offset = 0
    while True:
        chunk = os.pread(source_fd, chunk_size, offset)
        if not chunk:
            break
        view = memoryview(chunk)
        while view:
            written = os.write(target_fd, view)
            if written <= 0:
                raise RuntimeError("failed to write complete ONNX snapshot")
            view = view[written:]
        offset += len(chunk)


def _require_expected_fingerprint(
    actual: OnnxArtifactFingerprint,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    label: str,
) -> None:
    """按 size/SHA256 校验 artifact，任一不一致立即 fail-closed。"""
    if actual.size_bytes != expected_size_bytes or actual.sha256 != expected_sha256:
        raise RuntimeError(
            f"{label} fingerprint mismatch: expected size={expected_size_bytes}, "
            f"sha256={expected_sha256}; got size={actual.size_bytes}, "
            f"sha256={actual.sha256}"
        )


def fingerprint_fd(fd: int, *, label: str, chunk_size: int = 1 << 20) -> OnnxArtifactFingerprint:
    """公共入口：从持有 FD 计算 artifact 摘要（校验期间内容须稳定）。"""
    return _fingerprint_fd(fd, label=label, chunk_size=chunk_size)


def open_regular_file(path, *, label: str) -> int:
    """公共入口：以 O_NOFOLLOW 打开 regular file 并返回持有 FD。"""
    return _open_regular_file(path, label=label)


def require_expected_fingerprint(
    actual: OnnxArtifactFingerprint,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    label: str,
) -> None:
    """公共入口：按 size/SHA256 校验 artifact，不一致即 fail-closed。"""
    return _require_expected_fingerprint(
        actual,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        label=label,
    )


def onnx_release_root() -> Path:
    """公共入口：返回 ONNX release 根；读取当前模块级真源（支持测试替换）。"""
    return _ONNX_RELEASE_ROOT


class VerifiedOnnxArtifact:
    """让 ORT 实际加载字节与后续发布 snapshot 绑定到同一持有 FD。"""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self._verified_fingerprint: OnnxArtifactFingerprint | None = None
        self._snapshot_fd: int | None = None
        self._snapshot_path: Path | None = None
        self._snapshot_dir: Path | None = None
        self._source_stat = None

    @property
    def verified_fingerprint(self) -> OnnxArtifactFingerprint:
        if self._verified_fingerprint is None:
            raise RuntimeError("ONNX artifact is not bound to a verification session")
        return self._verified_fingerprint

    def _cleanup_snapshot(self) -> None:
        if self._snapshot_fd is not None:
            os.close(self._snapshot_fd)
            self._snapshot_fd = None
        if self._snapshot_path is not None:
            self._snapshot_path.unlink(missing_ok=True)
            self._snapshot_path = None
        if self._snapshot_dir is not None:
            try:
                self._snapshot_dir.rmdir()
            except FileNotFoundError:
                pass
            self._snapshot_dir = None

    def close(self) -> None:
        """释放 verifier-owned snapshot；不删除调用方 source。"""
        self._cleanup_snapshot()

    def create_session(self, session_factory=None):
        if self._verified_fingerprint is not None:
            raise RuntimeError("ONNX artifact is already bound to a verification session")
        source_fd: int | None = None
        writable_snapshot_fd: int | None = None
        try:
            source_fd = _open_regular_file(self.path, label="ONNX verification source")
            snapshot_dir = Path(tempfile.mkdtemp(prefix=".onnx_verify_"))
            self._snapshot_dir = snapshot_dir
            snapshot_path = snapshot_dir / "snapshot.onnx"
            self._snapshot_path = snapshot_path
            snapshot_flags = (
                os.O_RDWR | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            writable_snapshot_fd = os.open(snapshot_path, snapshot_flags, 0o600)
            source_before = os.fstat(source_fd)
            _copy_fd(source_fd, writable_snapshot_fd)
            source_after = os.fstat(source_fd)
            self._source_stat = source_after
            stable_fields = (
                "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink",
            )
            if any(
                getattr(source_before, field) != getattr(source_after, field)
                for field in stable_fields
            ):
                raise RuntimeError("ONNX verification source changed while snapshotting")
            os.fsync(writable_snapshot_fd)
            os.fchmod(writable_snapshot_fd, 0o400)
            writable_stat = os.fstat(writable_snapshot_fd)
            snapshot_fd = _open_regular_file(
                snapshot_path, label="ONNX verification snapshot",
            )
            # 立即转移给实例，后续异常由 ``_cleanup_snapshot`` 关闭。
            self._snapshot_fd = snapshot_fd
            readonly_stat = os.fstat(snapshot_fd)
            if (
                writable_stat.st_dev != readonly_stat.st_dev
                or writable_stat.st_ino != readonly_stat.st_ino
            ):
                raise RuntimeError("ONNX verification snapshot path identity changed")
            os.close(writable_snapshot_fd)
            writable_snapshot_fd = None
            fingerprint = _fingerprint_fd(snapshot_fd, label="ONNX verification snapshot")
            model_bytes = _read_fd_bytes(snapshot_fd, fingerprint.size_bytes)
            if hashlib.sha256(model_bytes).hexdigest() != fingerprint.sha256:
                raise RuntimeError("ORT input bytes do not match ONNX verification snapshot")
            if session_factory is None:
                from onnx.utils import require_onnxruntime

                session = require_onnxruntime().InferenceSession(
                    model_bytes, providers=["CPUExecutionProvider"],
                )
            else:
                session = session_factory(model_bytes, providers=["CPUExecutionProvider"])
            current = _fingerprint_fd(snapshot_fd, label="ONNX verification snapshot")
            _require_expected_fingerprint(
                current,
                expected_sha256=fingerprint.sha256,
                expected_size_bytes=fingerprint.size_bytes,
                label="ONNX verification snapshot",
            )
            if current.device != fingerprint.device or current.inode != fingerprint.inode:
                raise RuntimeError("ONNX verification snapshot identity changed")
            self._verified_fingerprint = fingerprint
            return session
        except Exception:
            self._cleanup_snapshot()
            raise
        finally:
            if writable_snapshot_fd is not None:
                os.close(writable_snapshot_fd)
            if source_fd is not None:
                os.close(source_fd)

    def publish(self, target, *, all_pass: bool, publisher=None, manifest=None, require_manifest: bool = False):
        """发布已由 ORT 验证的 snapshot，并按需提交 provenance manifest。

        ``require_manifest=True`` 要求 ``manifest`` 可构建且 publisher 显式接受
        ``manifest=``；不接受该关键字的自定义 publisher 会在执行前直接失败，
        不会降级为只发布 ONNX。调用方只要传入 ``manifest``，即使没有设置
        ``require_manifest``，也按正式 manifest 发布处理，避免 manifest 被静默丢弃。
        """
        fingerprint = self.verified_fingerprint
        if self._snapshot_fd is None or self._snapshot_path is None:
            raise RuntimeError("verified ONNX snapshot is no longer available")
        current = _fingerprint_fd(self._snapshot_fd, label="ONNX publication snapshot")
        _require_expected_fingerprint(
            current,
            expected_sha256=fingerprint.sha256,
            expected_size_bytes=fingerprint.size_bytes,
            label="ONNX publication snapshot",
        )
        if current.device != fingerprint.device or current.inode != fingerprint.inode:
            raise RuntimeError("ONNX publication snapshot identity changed")
        if publisher is None:
            publisher = publish_verified_onnx
        manifest_spec = None
        if require_manifest or manifest is not None:
            if not isinstance(manifest, Mapping):
                raise RuntimeError(
                    "formal ONNX publication requires checkpoint/config/runtime manifest binding"
                )
            from onnx.verify.manifest import validate_onnx_manifest_spec

            manifest_spec = dict(manifest)
            source_files = manifest_spec.get("source_files")
            source_binding = manifest_spec.get("source_binding")
            runtime_contract = manifest_spec.get("runtime_contract")
            model_id = manifest_spec.get("model_id")
            if not model_id or (not source_files and not source_binding) or not runtime_contract:
                raise RuntimeError(
                    "ONNX publication manifest must bind model_id, source_files/source_binding, "
                    "and runtime_contract"
                )
            validate_onnx_manifest_spec(
                model_id=str(model_id),
                source_files=source_files,
                source_binding=source_binding,
                runtime_contract=runtime_contract,
            )
        try:
            publisher_kwargs = dict(
                all_pass=all_pass,
                expected_sha256=fingerprint.sha256,
                expected_size_bytes=fingerprint.size_bytes,
                expected_device=fingerprint.device,
                expected_inode=fingerprint.inode,
                source_fd=self._snapshot_fd,
            )
            if manifest_spec is not None:
                publisher_kwargs["manifest"] = manifest_spec
            result = publisher(self._snapshot_path, target, **publisher_kwargs)
        finally:
            self._cleanup_snapshot()
        if result is not None or not all_pass:
            _unlink_source_if_owned(self.path, self._source_stat)
        return result


def require_strict_runtime_spec(session, model_id: str):
    """正式 verifier 统一执行 metadata + graph shape + exact I/O gate。"""
    from onnx.runtime_spec import runtime_spec_from_session

    return runtime_spec_from_session(session, model_id)


def require_release_preflight(
    *,
    save_onnx,
    checkpoint,
    config=None,
    require_config: bool = False,
    inference_dtype: str = "float32",
    thresholds: dict[str, tuple[float, float]],
) -> None:
    """发布请求必须满足 checkpoint/config/fp32/预注册阈值合同。"""
    if save_onnx is None:
        return
    if checkpoint is None:
        raise ValueError("release --save-onnx requires --checkpoint")
    if require_config and config is None:
        raise ValueError("release --save-onnx requires --config")
    if inference_dtype != "float32":
        raise ValueError("release --save-onnx only accepts --inference-dtype float32")
    if not thresholds:
        raise ValueError("release threshold mapping must not be empty")
    for name, (actual, preregistered) in thresholds.items():
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise ValueError(f"{name} must be a finite positive number")
        actual = float(actual)
        preregistered = float(preregistered)
        if not math.isfinite(actual) or actual <= 0:
            raise ValueError(f"{name} must be finite and > 0, got {actual!r}")
        if not math.isfinite(preregistered) or preregistered <= 0:
            raise ValueError(f"invalid preregistered {name}: {preregistered!r}")
        if actual > preregistered:
            raise ValueError(
                f"release {name}={actual:g} cannot exceed preregistered {preregistered:g}"
            )


def _checkpoint_embedded_training_config(
    checkpoint_data: Mapping[str, object],
) -> dict[str, object] | None:
    """校验 checkpoint 的 ``resolved_config``/``config`` 兼容键，并返回唯一 embedded config。"""
    from core.trainer.checkpoint_state import resolved_config_identity

    embedded: dict[str, dict[str, object]] = {}
    for key in ("resolved_config", "config"):
        value = checkpoint_data.get(key)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise RuntimeError(f"checkpoint {key} must be a mapping")
        embedded[key] = dict(value)
    if len(embedded) == 2 and resolved_config_identity(
        embedded["resolved_config"]
    ) != resolved_config_identity(embedded["config"]):
        raise RuntimeError("checkpoint embedded config aliases conflict")
    return embedded.get("resolved_config") or embedded.get("config")


def checkpoint_embedded_training_config(
    checkpoint_data: Mapping[str, object],
) -> dict[str, object] | None:
    """公共入口：校验 checkpoint 的 ``resolved_config``/``config`` 兼容键并返回唯一 embedded config。"""
    return _checkpoint_embedded_training_config(checkpoint_data)


def _load_checkpoint_config_sidecars(checkpoint_path: Path) -> list[tuple[Path, dict]]:
    """枚举 checkpoint 同目录的受限 resolved-config sidecar。"""
    from core.utils.config import load_merged_config

    parent = checkpoint_path.parent
    paths = sorted(
        {
            *parent.glob("resolved_config*.yaml"),
            *parent.glob("resolved_config*.yml"),
            *parent.glob("resolved_config*.json"),
        }
    )
    return [(path, load_merged_config(path)) for path in paths if path.is_file()]


def load_checkpoint_training_config(checkpoint, model_id: str) -> dict[str, object]:
    """按 checkpoint 可验证绑定读取训练 resolved config；歧义时 fail-closed。"""
    from core.trainer.checkpoint_state import (
        parse_resolved_config_provenance,
        resolved_config_identity,
    )
    from core.utils.config import load_merged_config
    from core.utils.safe_load import safe_torch_load

    checkpoint_path = Path(checkpoint)
    checkpoint_data = safe_torch_load(
        checkpoint_path,
        purpose="verify ONNX checkpoint config provenance",
    )
    if not isinstance(checkpoint_data, Mapping):
        raise RuntimeError("checkpoint must contain a mapping")
    embedded_config = _checkpoint_embedded_training_config(checkpoint_data)
    provenance_value = checkpoint_data.get("resolved_config_provenance")

    if provenance_value is not None:
        provenance = parse_resolved_config_provenance(provenance_value)
        source_path = checkpoint_path.parent / str(provenance["source"])
        if not source_path.is_file():
            raise RuntimeError(
                f"checkpoint resolved config provenance source is missing: {source_path}"
            )
        sidecar_config = load_merged_config(source_path)
        expected_identity = str(provenance["sha256"])
        if resolved_config_identity(sidecar_config) != expected_identity:
            raise RuntimeError("checkpoint resolved config sidecar identity mismatch")
        for key in ("resolved_config", "config"):
            value = checkpoint_data.get(key)
            if not isinstance(value, Mapping):
                raise RuntimeError(
                    f"bound checkpoint must embed mapping '{key}'"
                )
            if resolved_config_identity(value) != expected_identity:
                raise RuntimeError(
                    f"checkpoint embedded {key} identity does not match provenance"
                )
        return dict(sidecar_config)

    sidecars = _load_checkpoint_config_sidecars(checkpoint_path)
    if embedded_config is not None:
        embedded_identity = resolved_config_identity(embedded_config)
        if not sidecars:
            return embedded_config
        matching = [
            config
            for _path, config in sidecars
            if resolved_config_identity(config) == embedded_identity
        ]
        if not matching:
            raise RuntimeError(
                "checkpoint resolved_config sidecar does not match embedded training config"
            )
        return embedded_config

    if not sidecars:
        raise RuntimeError(
            "checkpoint config provenance is missing: expected a resolved_config sidecar "
            "or embedded resolved_config/config"
        )
    by_identity: dict[str, dict] = {}
    for _path, config in sidecars:
        by_identity.setdefault(resolved_config_identity(config), config)
    if len(by_identity) != 1:
        raise RuntimeError(
            "legacy checkpoint config provenance is ambiguous across resolved_config sidecars"
        )
    return dict(next(iter(by_identity.values())))


def require_checkpoint_config_provenance(
    checkpoint,
    config: Mapping[str, object],
    model_id: str,
) -> dict[str, object]:
    """要求 verifier 配置与 checkpoint 的 resolved training config 完整一致。"""
    from onnx.runtime_spec import deployment_config_binding

    if not isinstance(config, Mapping):
        raise ValueError("verifier config must be a mapping")
    training_config = load_checkpoint_training_config(checkpoint, model_id)
    training_binding = deployment_config_binding(training_config, model_id)
    verifier_binding = deployment_config_binding(config, model_id)
    if verifier_binding["resolved_config"] != training_binding["resolved_config"]:
        raise RuntimeError(
            f"{model_id} verifier config fingerprint does not match checkpoint training "
            "resolved_config"
        )
    if verifier_binding["config"] != training_binding["config"]:
        raise RuntimeError(
            f"{model_id} verifier deployment semantics do not match checkpoint training config"
        )
    return training_binding


def release_results_pass(
    statuses,
    *,
    compared_counts,
    metrics,
    thresholds,
) -> bool:
    """仅三个完整、有限且重新满足阈值的 PASS 比较允许发布。"""
    if list(statuses) != ["PASS", "PASS", "PASS"]:
        return False
    counts = list(compared_counts)
    values = list(metrics)
    limits = list(thresholds)
    if len(counts) != 3 or len(values) != 3 or len(limits) != 3:
        return False
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in counts
    ):
        return False
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        return False
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in limits
    ):
        return False
    return all(float(metric) < float(limit) for metric, limit in zip(values, limits))


def _freeze_source_file(path: Path, *, label: str):
    """从一个持有 FD 固化 bytes 与内容身份。"""
    descriptor = _open_regular_file(path, label=label)
    try:
        fingerprint = _fingerprint_fd(descriptor, label=label)
        content = _read_fd_bytes(descriptor, fingerprint.size_bytes)
        if hashlib.sha256(content).hexdigest() != fingerprint.sha256:
            raise RuntimeError(f"{label} frozen bytes digest mismatch")
        return fingerprint, content
    finally:
        os.close(descriptor)


def _source_identity(path: Path, fingerprint: OnnxArtifactFingerprint) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size": fingerprint.size_bytes,
        "sha256": fingerprint.sha256,
    }


def load_checkpoint_strict(
    model: torch.nn.Module,
    ckpt_path,
    *,
    return_source_binding: bool = False,
    source_files: Mapping[str, PathLike | None] | None = None,
    model_id: str | None = None,
):
    """从一次冻结的 loader snapshot 严格加载 checkpoint。"""
    from core.utils.artifact_identity import SourceBinding
    from onnx.export import load_checkpoint

    checkpoint_path = Path(ckpt_path).expanduser().resolve(strict=True)
    checkpoint_fingerprint, checkpoint_content = _freeze_source_file(
        checkpoint_path, label="ONNX checkpoint source"
    )
    stream = io.BytesIO(checkpoint_content)
    stream.name = f"{checkpoint_path} (immutable ONNX loader snapshot)"
    load_checkpoint(model, stream)
    if not return_source_binding:
        return None

    files = {"checkpoint": checkpoint_path}
    precomputed = {"checkpoint": _source_identity(checkpoint_path, checkpoint_fingerprint)}
    for name, value in (source_files or {}).items():
        if value is None or name == "checkpoint":
            continue
        path = Path(value).expanduser().resolve(strict=True)
        fingerprint, _content = _freeze_source_file(
            path, label=f"ONNX source artifact {name}"
        )
        files[name] = path
        precomputed[name] = _source_identity(path, fingerprint)
    return SourceBinding.bind(
        files,
        precomputed=precomputed,
        parameters={"model_id": str(model_id or "unknown")},
    ).as_dict()


def freeze_release_source_binding(
    *,
    model_id,
    checkpoint,
    config=None,
    checkpoint_binding: Mapping[str, object] | None = None,
):
    """冻结正式发布 source；checkpoint 可直接复用实际 loader snapshot。"""
    from core.utils.artifact_identity import SourceBinding

    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    files = {"checkpoint": checkpoint_path}
    precomputed: dict[str, Mapping[str, object]] = {}
    if checkpoint_binding is not None:
        if set(checkpoint_binding) == {"path", "size", "sha256"}:
            precomputed["checkpoint"] = dict(checkpoint_binding)
        else:
            binding = SourceBinding.from_fingerprint(checkpoint_binding)
            identity = binding.as_dict()["files"].get("checkpoint")
            if identity is None:
                raise ValueError("checkpoint binding does not contain checkpoint identity")
            precomputed["checkpoint"] = identity
    else:
        fingerprint, _content = _freeze_source_file(
            checkpoint_path, label="ONNX publication checkpoint source"
        )
        precomputed["checkpoint"] = _source_identity(checkpoint_path, fingerprint)
    if config is not None:
        config_path = Path(config).expanduser().resolve(strict=True)
        fingerprint, _content = _freeze_source_file(
            config_path, label="ONNX publication config source"
        )
        files["config"] = config_path
        precomputed["config"] = _source_identity(config_path, fingerprint)
    binding = SourceBinding.bind(
        files,
        precomputed=precomputed,
        parameters={"model_id": model_id},
    )
    return binding.as_dict()


def require_exact_onnx_io(
    expected_inputs,
    expected_outputs,
    actual_inputs,
    actual_outputs,
) -> None:
    """拒绝名称缺失、额外或顺序变化的 ONNX I/O ABI。"""
    expected_inputs = list(expected_inputs)
    expected_outputs = list(expected_outputs)
    actual_inputs = list(actual_inputs)
    actual_outputs = list(actual_outputs)
    if actual_inputs != expected_inputs or actual_outputs != expected_outputs:
        raise RuntimeError(
            "ONNX I/O ABI mismatch: "
            f"expected inputs={expected_inputs}, outputs={expected_outputs}; "
            f"got inputs={actual_inputs}, outputs={actual_outputs}"
        )


def require_cache_metadata_abi(
    metadata: dict[str, str],
    metadata_key: str,
    expected_cache: dict[str, list[int]],
    input_infos,
) -> None:
    """严格比对 metadata cache 名称/shape 与 ONNX 输入静态 ABI。"""
    from onnx.utils import materialize_cache_shapes, parse_onnx_cache_config

    parsed = parse_onnx_cache_config(metadata, metadata_key)
    if parsed != expected_cache:
        raise RuntimeError(
            f"ONNX {metadata_key} ABI mismatch: expected {expected_cache}, got {parsed}"
        )
    materialized = materialize_cache_shapes(parsed, input_infos, batch_size=1)
    if materialized != expected_cache:
        raise RuntimeError(
            f"ONNX {metadata_key} graph shape mismatch: expected {expected_cache}, "
            f"got {materialized}"
        )


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    """流式计算文件 SHA256（避免一次性读入大文件）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _unlink_source_if_owned(source: Path, source_stat=None) -> None:
    """将路径原子移入 quarantine，只删除与 owned inode 一致的目录项。"""
    parent_flags = (
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_fd = os.open(source.parent, parent_flags)
    quarantine = _random_artifact_name("onnx-source-cleanup")
    try:
        try:
            os.replace(
                source.name,
                quarantine,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        except FileNotFoundError:
            return
        quarantined = os.stat(quarantine, dir_fd=parent_fd, follow_symlinks=False)
        if source_stat is None or _same_file_identity(quarantined, source_stat):
            os.unlink(quarantine, dir_fd=parent_fd)
        else:
            os.replace(
                quarantine,
                source.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
    finally:
        os.close(parent_fd)


def _release_relative_target(target: Path) -> tuple[Path, tuple[str, ...]]:
    root = Path(os.path.abspath(_ONNX_RELEASE_ROOT))
    absolute_target = Path(os.path.abspath(target))
    try:
        relative = absolute_target.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"ONNX publication target must stay under {_ONNX_RELEASE_ROOT}: {target}"
        ) from exc
    parts = relative.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"invalid ONNX publication target: {target}")
    return root, parts


def _open_release_parent(root: Path, parent_parts: tuple[str, ...]) -> tuple[int, int]:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    nofollow_flags = flags | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(root, nofollow_flags)
    except OSError as exc:
        raise ValueError(f"invalid ONNX release root: {root}") from exc
    parent_fd = os.dup(root_fd)
    try:
        for component in parent_parts:
            try:
                next_fd = os.open(component, nofollow_flags, dir_fd=parent_fd)
            except FileNotFoundError:
                os.mkdir(component, 0o755, dir_fd=parent_fd)
                os.fsync(parent_fd)
                next_fd = os.open(component, nofollow_flags, dir_fd=parent_fd)
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ValueError(
                        "ONNX publication target parent must not contain symlinks"
                    ) from exc
                raise
            os.close(parent_fd)
            parent_fd = next_fd
        return root_fd, parent_fd
    except Exception:
        os.close(parent_fd)
        os.close(root_fd)
        raise


def _require_parent_anchor(
    root: Path,
    parent_parts: tuple[str, ...],
    parent_fd: int,
) -> None:
    check_root_fd, check_parent_fd = _open_release_parent(root, parent_parts)
    try:
        expected = os.fstat(parent_fd)
        actual = os.fstat(check_parent_fd)
        if expected.st_dev != actual.st_dev or expected.st_ino != actual.st_ino:
            raise RuntimeError("ONNX publication target parent changed during publication")
    finally:
        os.close(check_parent_fd)
        os.close(check_root_fd)


def _random_artifact_name(prefix: str) -> str:
    return f".{prefix}-{secrets.token_hex(16)}.onnx"


def _create_anchored_file(parent_fd: int, name: str) -> int:
    flags = (
        os.O_RDWR | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    return os.open(name, flags, 0o600, dir_fd=parent_fd)


def _renameat2(parent_fd: int, left_name: str, right_name: str, flags: int) -> None:
    """调用 Linux ``renameat2``；不支持时 fail-closed。"""
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise RuntimeError("atomic ONNX replacement requires renameat2") from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(
        parent_fd,
        os.fsencode(left_name),
        parent_fd,
        os.fsencode(right_name),
        flags,
    ) != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            f"atomic ONNX rename failed: {os.strerror(error_number)}",
        )


def _rename_exchange(parent_fd: int, left_name: str, right_name: str) -> None:
    """原子交换 existing target 与 staging，杜绝正式路径缺失窗口。"""
    _renameat2(parent_fd, left_name, right_name, 2)  # RENAME_EXCHANGE


def _rename_noreplace(parent_fd: int, left_name: str, right_name: str) -> None:
    """仅在 target 仍不存在时提交 staging。"""
    _renameat2(parent_fd, left_name, right_name, 1)  # RENAME_NOREPLACE


def _acquire_target_lock(parent_fd: int, target_name: str) -> tuple[int, str, os.stat_result]:
    """获取稳定的 per-target 进程锁；lock entry 不随单次发布删除。"""
    target_digest = hashlib.sha256(target_name.encode("utf-8")).hexdigest()
    lock_name = f".onnx-publish-{target_digest}.lock"
    flags = (
        os.O_RDWR | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    lock_fd = os.open(lock_name, flags, 0o600, dir_fd=parent_fd)
    try:
        lock_stat = os.fstat(lock_fd)
        _require_regular_file(lock_stat, label="ONNX publication lock")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        _require_anchored_entry(
            parent_fd, lock_name, lock_stat, label="ONNX publication lock",
        )
        return lock_fd, lock_name, lock_stat
    except Exception:
        os.close(lock_fd)
        raise


def _require_anchored_entry(parent_fd: int, name: str, held_stat, *, label: str) -> None:
    entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    _require_regular_file(entry, label=label)
    if entry.st_dev != held_stat.st_dev or entry.st_ino != held_stat.st_ino:
        raise RuntimeError(f"{label} path no longer names the held inode")


def _same_file_identity(left, right) -> bool:
    left_device = getattr(left, "st_dev", getattr(left, "device", None))
    left_inode = getattr(left, "st_ino", getattr(left, "inode", None))
    right_device = getattr(right, "st_dev", getattr(right, "device", None))
    right_inode = getattr(right, "st_ino", getattr(right, "inode", None))
    return (
        left_device is not None
        and left_inode is not None
        and left_device == right_device
        and left_inode == right_inode
    )


def _paths_alias(source: Path, target: Path) -> bool:
    """判断 source/target 是否指向同一目录项或同一物理文件。"""
    try:
        if source.resolve(strict=False) == target.resolve(strict=False):
            return True
    except (OSError, RuntimeError):
        pass
    try:
        return source.exists() and target.exists() and os.path.samefile(source, target)
    except OSError:
        return False


def _cleanup_staging_entry(
    parent_fd: int,
    staging_name: str,
    staged_stat,
) -> str | None:
    """清理本事务 staging；失败返回可直接暴露给 CLI 的状态。"""
    quarantine = _random_artifact_name("onnx-stage-cleanup")
    try:
        os.replace(
            staging_name,
            quarantine,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
    except FileNotFoundError:
        return None
    except OSError:
        return "retained"
    try:
        entry = os.stat(quarantine, dir_fd=parent_fd, follow_symlinks=False)
        if staged_stat is None or not _same_file_identity(entry, staged_stat):
            return "identity-unconfirmed"
        os.unlink(quarantine, dir_fd=parent_fd)
        return None
    except OSError:
        return "retained"


def _rollback_publication(
    parent_fd: int,
    target_name: str,
    backup_name: str | None,
    *,
    staged_stat,
    backup_stat=None,
) -> None:
    """只撤销本事务创建的 inode，并验证恢复后的旧 target 身份。"""
    try:
        target_stat = os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        target_stat = None
    if backup_name is not None:
        if backup_stat is None:
            raise RuntimeError("rollback backup identity is missing")
        backup_entry = os.stat(
            backup_name, dir_fd=parent_fd, follow_symlinks=False,
        )
        _require_regular_file(backup_entry, label="rollback ONNX backup")
        if not _same_file_identity(backup_entry, backup_stat):
            raise RuntimeError("rollback ONNX backup identity changed")
        if target_stat is not None:
            _require_regular_file(target_stat, label="rollback ONNX target")
            if not _same_file_identity(target_stat, staged_stat):
                raise RuntimeError(
                    "rollback CAS refused to overwrite a concurrently published target"
                )
        os.replace(
            backup_name,
            target_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        restored = os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
        _require_regular_file(restored, label="restored ONNX target")
        if not _same_file_identity(restored, backup_stat):
            raise RuntimeError("restored ONNX target identity mismatch")
    elif target_stat is not None:
        _require_regular_file(target_stat, label="rollback ONNX target")
        if not _same_file_identity(target_stat, staged_stat):
            raise RuntimeError("rollback refused to remove an unknown ONNX target inode")
        rollback_quarantine = _random_artifact_name("onnx-rollback")
        os.replace(
            target_name,
            rollback_quarantine,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        quarantined = os.stat(
            rollback_quarantine, dir_fd=parent_fd, follow_symlinks=False,
        )
        _require_regular_file(quarantined, label="rollback ONNX target")
        if not _same_file_identity(quarantined, staged_stat):
            os.replace(
                rollback_quarantine,
                target_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            raise RuntimeError("rollback refused to remove an unknown ONNX target inode")
        os.unlink(rollback_quarantine, dir_fd=parent_fd)

    os.fsync(parent_fd)
    if backup_name is not None:
        restored = os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
        _require_regular_file(restored, label="restored ONNX target after fsync")
        if not _same_file_identity(restored, backup_stat):
            raise RuntimeError("restored ONNX target identity changed after fsync")
    else:
        try:
            os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("rollback failed to remove new ONNX target")


def _publish_verified_onnx_with_manifest(
    temp_path: PathLike,
    target_path: PathLike,
    *,
    all_pass: bool,
    expected_sha256: str,
    expected_size_bytes: int,
    expected_device: int | None,
    expected_inode: int | None,
    source_fd: int | None,
    own_source: bool | None,
    manifest: Mapping,
) -> dict | None:
    """在同一 target lock 下提交 ONNX 与 manifest 两个文件。"""
    source = Path(temp_path)
    target = Path(target_path)
    if _paths_alias(source, target):
        raise ValueError("temp_path and target_path must not alias")
    if not all_pass:
        if source_fd is None and own_source is not False:
            cleanup_fd = _open_regular_file(source, label="failed ONNX publication source")
            try:
                _unlink_source_if_owned(source, os.fstat(cleanup_fd))
            finally:
                os.close(cleanup_fd)
        return None

    publication_fd = os.dup(source_fd) if source_fd is not None else _open_regular_file(
        source, label="ONNX publication source"
    )
    root_fd = parent_fd = lock_fd = None
    staging_fds: list[int] = []
    staging_names: list[str] = []
    states: list[dict[str, object]] = []
    committed = False
    cleanup_status: list[str] = []
    owned_source_fd = source_fd is None
    try:
        source_fingerprint = _fingerprint_fd(publication_fd, label="ONNX publication source")
        _require_expected_fingerprint(
            source_fingerprint,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
            label="ONNX publication source",
        )
        if expected_device is not None and source_fingerprint.device != expected_device:
            raise RuntimeError("ONNX publication source device mismatch")
        if expected_inode is not None and source_fingerprint.inode != expected_inode:
            raise RuntimeError("ONNX publication source inode mismatch")
        root, target_parts = _release_relative_target(target)
        root_fd, parent_fd = _open_release_parent(root, target_parts[:-1])
        target_name = target_parts[-1]
        lock_fd, lock_name, lock_stat = _acquire_target_lock(parent_fd, target_name)
        _require_anchored_entry(parent_fd, lock_name, lock_stat, label="ONNX publication lock")

        def create_stage(suffix: str) -> tuple[int, str]:
            for _ in range(128):
                name = _random_artifact_name(suffix)
                try:
                    fd = _create_anchored_file(parent_fd, name)
                except FileExistsError:
                    continue
                staging_fds.append(fd)
                staging_names.append(name)
                return fd, name
            raise RuntimeError("Unable to allocate ONNX publication staging file")

        onnx_fd, onnx_stage = create_stage("onnx-stage")
        _copy_fd(publication_fd, onnx_fd)
        os.fsync(onnx_fd)
        os.fchmod(onnx_fd, 0o400)
        onnx_stat = os.fstat(onnx_fd)
        onnx_digest = _fingerprint_fd(onnx_fd, label="staged ONNX artifact")
        _require_expected_fingerprint(
            onnx_digest,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
            label="staged ONNX artifact",
        )

        from onnx.verify.manifest import (
            build_onnx_manifest,
            manifest_path,
            verify_onnx_manifest_sources,
        )

        manifest_spec = dict(manifest)
        model_id = manifest_spec.get("model_id")
        source_files = manifest_spec.get("source_files")
        source_binding = manifest_spec.get("source_binding")
        runtime_contract = manifest_spec.get("runtime_contract")
        if not model_id or (not source_files and not source_binding) or not runtime_contract:
            raise RuntimeError(
                "formal ONNX publication requires model_id, source_files/source_binding, "
                "and runtime_contract"
            )
        artifact = {
            "path": str(target.resolve()),
            "size": onnx_digest.size_bytes,
            "sha256": onnx_digest.sha256,
        }
        manifest_value = build_onnx_manifest(
            target=target,
            model_id=str(model_id),
            source_files=source_files,
            source_binding=source_binding,
            runtime_contract=runtime_contract,
            verification=manifest_spec.get("verification"),
            artifact=artifact,
        )
        verify_onnx_manifest_sources(manifest_value)
        manifest_name = manifest_path(target).name
        manifest_fd, manifest_stage = create_stage("manifest-stage")
        payload = (json.dumps(manifest_value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        offset = 0
        while offset < len(payload):
            written = os.write(manifest_fd, payload[offset:])
            if written <= 0:
                raise RuntimeError("failed to write ONNX manifest staging")
            offset += written
        os.fsync(manifest_fd)
        os.fchmod(manifest_fd, 0o644)
        manifest_stat = os.fstat(manifest_fd)

        def commit_entry(stage_name: str, target_name_value: str, stage_stat):
            try:
                existing = os.stat(target_name_value, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                _require_regular_file(existing, label="existing ONNX publication target")
                _rename_exchange(parent_fd, stage_name, target_name_value)
                exchanged_entry = os.stat(
                    stage_name, dir_fd=parent_fd, follow_symlinks=False
                )
                if not _same_file_identity(exchanged_entry, existing):
                    _rename_exchange(parent_fd, stage_name, target_name_value)
                    raise RuntimeError("ONNX publication exchange CAS conflict preserved concurrent target")
                backup_name = _random_artifact_name("onnx-backup")
                os.replace(
                    stage_name,
                    backup_name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                backup_stat_value = os.stat(
                    backup_name, dir_fd=parent_fd, follow_symlinks=False
                )
                if not _same_file_identity(backup_stat_value, existing):
                    raise RuntimeError("ONNX publication backup identity mismatch")
                return {
                    "target": target_name_value,
                    "stage": stage_stat,
                    "backup": backup_name,
                    "backup_stat": backup_stat_value,
                }
            _rename_noreplace(parent_fd, stage_name, target_name_value)
            return {"target": target_name_value, "stage": stage_stat, "backup": None, "backup_stat": None}

        states.append(commit_entry(onnx_stage, target_name, onnx_stat))
        states.append(commit_entry(manifest_stage, manifest_name, manifest_stat))
        for state in states:
            _require_anchored_entry(
                parent_fd,
                state["target"],
                state["stage"],
                label="published ONNX publication file",
            )
        os.fsync(parent_fd)
        committed = True
        for state in states:
            backup_name = state["backup"]
            if backup_name is not None:
                try:
                    backup_stat_value = os.stat(
                        backup_name, dir_fd=parent_fd, follow_symlinks=False
                    )
                    if not _same_file_identity(backup_stat_value, state["backup_stat"]):
                        raise RuntimeError("ONNX publication backup identity changed")
                    os.unlink(backup_name, dir_fd=parent_fd)
                except OSError:
                    cleanup_status.append("backup_cleanup=retained")
        if owned_source_fd:
            try:
                _unlink_source_if_owned(source, os.fstat(publication_fd))
            except OSError:
                cleanup_status.append("source_cleanup=retained")
        for state in states:
            _require_anchored_entry(
                parent_fd,
                state["target"],
                state["stage"],
                label="published ONNX publication file after fsync",
            )
        result = {
            "path": str(target),
            "sha256": onnx_digest.sha256,
            "size_bytes": onnx_digest.size_bytes,
            "manifest": str(manifest_path(target)),
        }
        for status in cleanup_status:
            key, value = status.split("=", 1)
            result[key] = value
        return result
    except Exception as publication_error:
        if committed:
            suffix = f"; {'; '.join(cleanup_status)}" if cleanup_status else ""
            raise RuntimeError(
                "ONNX publication committed; post-fsync processing failed: "
                f"{publication_error}{suffix}"
            ) from publication_error
        if parent_fd is not None:
            for state in reversed(states):
                try:
                    _rollback_publication(
                        parent_fd,
                        state["target"],
                        state["backup"],
                        staged_stat=state["stage"],
                        backup_stat=state["backup_stat"],
                    )
                except FileNotFoundError:
                    pass
                except Exception as rollback_error:
                    raise RuntimeError(
                        "ONNX publication failed; rollback durability or identity is unconfirmed"
                    ) from rollback_error
            for name, fd in zip(staging_names, staging_fds):
                try:
                    _cleanup_staging_entry(parent_fd, name, os.fstat(fd))
                except OSError:
                    pass
        raise publication_error
    finally:
        for fd in staging_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        if parent_fd is not None:
            os.close(parent_fd)
        if root_fd is not None:
            os.close(root_fd)
        os.close(publication_fd)


def publish_verified_onnx(
    temp_path: PathLike,
    target_path: PathLike,
    *,
    all_pass: bool,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
    expected_device: int | None = None,
    expected_inode: int | None = None,
    source_fd: int | None = None,
    own_source: bool | None = None,
    manifest: Mapping | None = None,
) -> dict | None:
    """从绑定 snapshot FD 发布；commit 前失败只执行一次身份校验 rollback。

    FAIL 分支会清理 ``temp_path``（按 inode 归属校验后删除），所有权由
    ``own_source`` 显式声明：清理仅发生在 ``source_fd is None`` 且
    ``own_source is not False`` 时；显式 ``own_source=False`` 或由调用方传入
    ``source_fd``（表示调用方自行管理源文件生命周期）时一律保留该文件。
    """
    source = Path(temp_path)
    target = Path(target_path)

    if _paths_alias(source, target):
        raise ValueError("temp_path and target_path must not alias")
    if manifest is not None:
        if not expected_sha256 or expected_size_bytes is None:
            raise RuntimeError(
                "verified ONNX publication requires expected_sha256 and expected_size_bytes"
            )
        return _publish_verified_onnx_with_manifest(
            source,
            target,
            all_pass=all_pass,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
            expected_device=expected_device,
            expected_inode=expected_inode,
            source_fd=source_fd,
            own_source=own_source,
            manifest=manifest,
        )
    if not all_pass:
        if source_fd is None and own_source is not False:
            cleanup_fd = _open_regular_file(source, label="failed ONNX publication source")
            try:
                _unlink_source_if_owned(source, os.fstat(cleanup_fd))
            finally:
                os.close(cleanup_fd)
        return None
    if not expected_sha256 or expected_size_bytes is None:
        raise RuntimeError(
            "verified ONNX publication requires expected_sha256 and expected_size_bytes"
        )
    if (
        isinstance(expected_size_bytes, bool)
        or not isinstance(expected_size_bytes, int)
        or expected_size_bytes < 0
    ):
        raise ValueError("expected_size_bytes must be a non-negative integer")
    owned_source_fd = source_fd is None
    publication_fd = (
        _open_regular_file(source, label="ONNX publication source")
        if owned_source_fd else os.dup(source_fd)
    )
    source_stat = None
    root_fd: int | None = None
    parent_fd: int | None = None
    lock_fd: int | None = None
    lock_name: str | None = None
    lock_stat = None
    staging_fd: int | None = None
    backup_fd: int | None = None
    staging_name: str | None = None
    backup_name: str | None = None
    staged_stat = None
    backup_stat = None
    backup_moved = False
    target_replaced = False
    rollback_attempted = False
    staging_cleanup_attempted = False
    committed = False
    backup_cleanup = None
    source_cleanup = None
    try:
        source_stat = os.fstat(publication_fd)
        source_before = _fingerprint_fd(publication_fd, label="ONNX publication source")
        _require_expected_fingerprint(
            source_before,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
            label="ONNX publication source",
        )
        if expected_device is not None and source_before.device != expected_device:
            raise RuntimeError("ONNX publication source device mismatch")
        if expected_inode is not None and source_before.inode != expected_inode:
            raise RuntimeError("ONNX publication source inode mismatch")

        try:
            root, target_parts = _release_relative_target(target)
            root_fd, parent_fd = _open_release_parent(root, target_parts[:-1])
        except Exception:
            if owned_source_fd:
                _unlink_source_if_owned(source, source_stat)
            raise
        target_name = target_parts[-1]
        lock_fd, lock_name, lock_stat = _acquire_target_lock(parent_fd, target_name)
        staging_name = _random_artifact_name("onnx-stage")
        staging_fd = _create_anchored_file(parent_fd, staging_name)
        staged_stat = os.fstat(staging_fd)

        _copy_fd(publication_fd, staging_fd)
        os.fsync(staging_fd)
        os.fchmod(staging_fd, 0o400)
        writable_stat = os.fstat(staging_fd)
        readonly_flags = (
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        readonly_fd = os.open(staging_name, readonly_flags, dir_fd=parent_fd)
        try:
            readonly_stat = os.fstat(readonly_fd)
            if not _same_file_identity(writable_stat, readonly_stat):
                raise RuntimeError("staged ONNX artifact identity changed while reopening")
        except Exception:
            os.close(readonly_fd)
            raise
        os.close(staging_fd)
        staging_fd = readonly_fd
        staged_stat = readonly_stat

        source_after = _fingerprint_fd(publication_fd, label="ONNX publication source")
        _require_expected_fingerprint(
            source_after,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
            label="ONNX publication source after copy",
        )
        if not _same_file_identity(source_after, source_before):
            raise RuntimeError("ONNX publication source identity changed")

        _require_anchored_entry(
            parent_fd, staging_name, staged_stat, label="staged ONNX artifact",
        )
        final_fingerprint = _fingerprint_fd(
            staging_fd, label="staged ONNX artifact before replace",
        )
        _require_expected_fingerprint(
            final_fingerprint,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
            label="staged ONNX artifact before replace",
        )
        _require_parent_anchor(root, target_parts[:-1], parent_fd)
        commit_fingerprint = _fingerprint_fd(
            staging_fd, label="staged ONNX artifact at commit",
        )
        _require_expected_fingerprint(
            commit_fingerprint,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
            label="staged ONNX artifact at commit",
        )
        _require_anchored_entry(
            parent_fd, staging_name, staged_stat, label="staged ONNX artifact",
        )
        _require_anchored_entry(
            parent_fd, lock_name, lock_stat, label="ONNX publication lock",
        )

        try:
            existing = os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            _require_regular_file(existing, label="existing ONNX publication target")
            _rename_exchange(parent_fd, staging_name, target_name)
            target_replaced = True
            exchanged_entry = os.stat(
                staging_name, dir_fd=parent_fd, follow_symlinks=False,
            )
            _require_regular_file(exchanged_entry, label="exchanged ONNX backup")
            backup_stat = exchanged_entry
            backup_name = staging_name
            backup_moved = True
            if not _same_file_identity(exchanged_entry, existing):
                # CAS 后 target 被替换时，必须原子交换回去，不得吞掉并发产物。
                try:
                    _rename_exchange(parent_fd, staging_name, target_name)
                except Exception as exchange_error:
                    raise RuntimeError(
                        "ONNX publication exchange CAS conflict requires rollback"
                    ) from exchange_error
                target_replaced = False
                backup_moved = False
                backup_name = None
                raise RuntimeError(
                    "ONNX publication exchange CAS conflict preserved concurrent target"
                )
            durable_backup_name = _random_artifact_name("onnx-backup")
            os.replace(
                backup_name,
                durable_backup_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            backup_name = durable_backup_name
            backup_entry = os.stat(
                backup_name, dir_fd=parent_fd, follow_symlinks=False,
            )
            _require_regular_file(backup_entry, label="ONNX publication backup")
            if not _same_file_identity(backup_entry, backup_stat):
                raise RuntimeError("ONNX publication backup identity mismatch")
            backup_fd = os.open(backup_name, readonly_flags, dir_fd=parent_fd)
            if not _same_file_identity(os.fstat(backup_fd), backup_stat):
                raise RuntimeError("ONNX publication backup FD identity mismatch")
        else:
            _require_anchored_entry(
                parent_fd, staging_name, staged_stat, label="staged ONNX artifact",
            )
            try:
                _rename_noreplace(parent_fd, staging_name, target_name)
            except FileExistsError as exc:
                raise RuntimeError(
                    "ONNX publication create CAS conflict preserved concurrent target"
                ) from exc
            target_replaced = True

        _require_anchored_entry(
            parent_fd, target_name, staged_stat, label="published ONNX artifact",
        )
        post_replace = _fingerprint_fd(staging_fd, label="published ONNX artifact")
        _require_expected_fingerprint(
            post_replace,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
            label="published ONNX artifact",
        )
        _require_parent_anchor(root, target_parts[:-1], parent_fd)
        os.fsync(parent_fd)
        # 目录 fsync 成功即已提交；后续复核失败不得回滚。
        committed = True
        _require_anchored_entry(
            parent_fd, target_name, staged_stat, label="published ONNX artifact after fsync",
        )
        final_target = _fingerprint_fd(
            staging_fd, label="published ONNX artifact after fsync",
        )
        _require_expected_fingerprint(
            final_target,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
            label="published ONNX artifact after fsync",
        )
        _require_parent_anchor(root, target_parts[:-1], parent_fd)

        if backup_moved and backup_name is not None:
            backup_quarantine = _random_artifact_name("onnx-backup-cleanup")
            try:
                os.replace(
                    backup_name,
                    backup_quarantine,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                quarantined = os.stat(
                    backup_quarantine, dir_fd=parent_fd, follow_symlinks=False,
                )
                _require_regular_file(quarantined, label="ONNX publication backup")
                if not _same_file_identity(quarantined, backup_stat):
                    raise RuntimeError("ONNX publication backup identity changed")
                os.unlink(backup_quarantine, dir_fd=parent_fd)
                backup_moved = False
                try:
                    os.fsync(parent_fd)
                except OSError:
                    # target 已提交；backup 删除持久性未知。
                    backup_cleanup = "durability-unconfirmed"
            except RuntimeError:
                # quarantine inode 不符时不得删除未知文件。
                backup_cleanup = "identity-unconfirmed"
            except OSError:
                # rename/unlink 可能已生效；任一候选名保留已知 inode 即为 retained。
                retained = False
                for candidate in (backup_name, backup_quarantine):
                    try:
                        backup_entry = os.stat(
                            candidate, dir_fd=parent_fd, follow_symlinks=False,
                        )
                    except OSError:
                        continue
                    if _same_file_identity(backup_entry, backup_stat):
                        retained = True
                        break
                backup_cleanup = "retained" if retained else "durability-unconfirmed"

        if owned_source_fd:
            try:
                _unlink_source_if_owned(source, source_stat)
            except OSError:
                # source 清理不属于 target commit；失败时保留并报告。
                source_cleanup = "retained"

        # backup/source 清理必须在最终 inode/digest/parent 复核前完成。
        _require_parent_anchor(root, target_parts[:-1], parent_fd)
        final_return = _fingerprint_fd(
            staging_fd, label="final published ONNX artifact",
        )
        _require_expected_fingerprint(
            final_return,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
            label="final published ONNX artifact",
        )
        _require_anchored_entry(
            parent_fd, target_name, staged_stat, label="final published ONNX artifact",
        )
        result = {
            "path": str(target),
            "sha256": final_return.sha256,
            "size_bytes": final_return.size_bytes,
        }
        if backup_cleanup is not None:
            result["backup_cleanup"] = backup_cleanup
        if source_cleanup is not None:
            result["source_cleanup"] = source_cleanup
        return result
    except Exception as publication_error:
        if committed:
            cleanup_states = []
            if backup_moved:
                cleanup_states.append("backup_cleanup=retained")
            if owned_source_fd:
                cleanup_states.append("source_cleanup=retained")
            suffix = f"; {'; '.join(cleanup_states)}" if cleanup_states else ""
            raise RuntimeError(
                "ONNX publication committed; post-fsync processing failed: "
                f"{publication_error}{suffix}"
            ) from publication_error
        if (
            parent_fd is not None
            and staged_stat is not None
            and (target_replaced or backup_moved)
        ):
            rollback_attempted = True
            try:
                _rollback_publication(
                    parent_fd,
                    target_parts[-1],
                    backup_name if backup_moved else None,
                    staged_stat=staged_stat,
                    backup_stat=backup_stat,
                )
                backup_moved = False
                target_replaced = False
            except Exception as rollback_error:
                if "rollback CAS" in str(rollback_error):
                    raise RuntimeError(
                        "ONNX publication failed; rollback CAS conflict preserved "
                        "concurrent target"
                    ) from rollback_error
                raise RuntimeError(
                    "ONNX publication failed; rollback durability or identity is unconfirmed"
                ) from rollback_error
            raise RuntimeError(
                "ONNX publication commit failed; previous target restored"
            ) from publication_error
        if parent_fd is not None and staging_name is not None:
            staging_cleanup_attempted = True
            staging_cleanup = _cleanup_staging_entry(
                parent_fd, staging_name, staged_stat,
            )
            if staging_cleanup is not None:
                raise RuntimeError(
                    f"{publication_error}; staging_cleanup={staging_cleanup}"
                ) from publication_error
        raise
    finally:
        if backup_fd is not None:
            os.close(backup_fd)
        if staging_fd is not None:
            os.close(staging_fd)
        if parent_fd is not None:
            if (
                staging_name is not None
                and not target_replaced
                and not staging_cleanup_attempted
            ):
                _cleanup_staging_entry(parent_fd, staging_name, staged_stat)
            # ``rollback_attempted`` 防止 finally 重复删除已恢复 target。
            if backup_moved and not committed and not rollback_attempted:
                try:
                    _rollback_publication(
                        parent_fd,
                        target_parts[-1],
                        backup_name,
                        staged_stat=staged_stat,
                        backup_stat=backup_stat,
                    )
                except Exception:
                    # 兜底回滚失败必须留痕：backup/target 可能停在半状态，
                    # 静默 pass 会让调用方误判目录仍然一致。
                    logger.error(
                        "ONNX publication rollback failed in cleanup path; "
                        "backup/target state may be inconsistent",
                        exc_info=True,
                    )
            if lock_fd is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
                lock_fd = None
            os.close(parent_fd)
        if root_fd is not None:
            os.close(root_fd)
        os.close(publication_fd)


def print_kept_onnx_artifact(onnx_path) -> None:
    """打印已显式保留或已发布 ONNX 的路径与 SHA256。

    lifecycle 临时 artifact 默认在退出时清理；调用此函数前必须保证该路径的
    所有权已转移为发布产物，或 lifecycle 已通过 ``keep()`` 明确保留。
    """
    p = Path(onnx_path)
    digest = sha256_file(p)
    size_mb = p.stat().st_size / 1024 / 1024
    print(f"  ONNX artifact kept: {p} ({size_mb:.1f} MB, sha256={digest})")
