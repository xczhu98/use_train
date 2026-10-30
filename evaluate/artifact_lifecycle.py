"""评估产物 envelope 的稳定 schema 与原子发布。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import socket
import stat
from typing import Any, Callable, Mapping
from uuid import uuid4

from core.utils.file_transaction import write_text_files_transactionally
from core.utils.inference_artifacts import atomic_write_json, inference_output_lock

ARTIFACT_SCHEMA = "evaluation-artifact-v1"
ARTIFACT_FILENAME = "evaluation_artifact.json"
CURRENT_SCHEMA = "evaluation-current-v1"
CURRENT_FILENAME = "evaluation_current.json"
RESERVATION_SCHEMA = "evaluation-reservation-v1"
RESERVATION_FILENAME = "evaluation_reservation.json"
EVALUATION_RESERVATION_TOKEN_ENV = "USE_TRAIN_EVAL_RESERVATION_TOKEN"
_RESULT_MANIFEST_SCHEMA = "evaluation-results-v1"
_RESULT_MANIFEST_FILENAME = "result_manifest.json"
_OUTPUT_LOCK_FILENAME = ".inference.lock"


@dataclass(frozen=True)
class CommittedArtifactSnapshot:
    output_dir: Path
    files: dict[str, str]


@dataclass(frozen=True)
class ArtifactRunReservation:
    output_dir: Path
    run_id: str
    token: str
    ownership: dict[str, Any]
    owner: dict[str, Any]


@dataclass(frozen=True)
class ArtifactRunStart:
    output_dir: Path
    run_id: str
    previous_commit: CommittedArtifactSnapshot | None


def normalize_ownership_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """将 ownership binding 收敛为可稳定比较的 JSON object。"""
    try:
        encoded = json.dumps(
            dict(binding), ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation ownership binding must be finite JSON") from exc
    if not isinstance(normalized, dict):
        raise ValueError("evaluation ownership binding must be a JSON object")
    return normalized


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _binding_matches(expected: Any, actual: Any) -> bool:
    """比较已规范化的 ownership binding，拒绝未授权的额外字段。"""
    return expected == actual


def _has_in_progress_run(output: Path) -> bool:
    current_path = output / CURRENT_FILENAME
    if not current_path.exists():
        return False
    current = _load_json_object(current_path, label="evaluation CURRENT")
    if current.get("status") != "in_progress":
        return False
    owner = current.get("owner")
    if isinstance(owner, Mapping) and _reservation_owner_is_dead(owner):
        return False
    return True


def _linux_process_start_ticks(pid: int) -> int:
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat_text.rsplit(")", 1)[1].split()
        return int(fields[19])
    except (IndexError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"cannot establish evaluation reservation owner identity: pid={pid}"
        ) from exc


def _reservation_owner() -> dict[str, Any]:
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except OSError as exc:
        raise RuntimeError("cannot establish evaluation reservation boot identity") from exc
    owner = {
        "hostname": socket.gethostname(),
        "boot_id": boot_id,
        "pid": os.getpid(),
        "process_start_ticks": _linux_process_start_ticks(os.getpid()),
    }
    if not owner["hostname"] or not owner["boot_id"]:
        raise RuntimeError("cannot establish evaluation reservation owner identity")
    return owner


def _reservation_owner_is_dead(owner: Mapping[str, Any]) -> bool:
    local = _reservation_owner()
    if (
        owner.get("hostname") != local["hostname"]
        or owner.get("boot_id") != local["boot_id"]
        or not isinstance(owner.get("pid"), int)
        or not isinstance(owner.get("process_start_ticks"), int)
    ):
        return False
    pid = owner["pid"]
    if pid <= 0:
        return False
    try:
        return _linux_process_start_ticks(pid) != owner["process_start_ticks"]
    except RuntimeError:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except (PermissionError, OSError):
            return False
        return False


def _reservation_payload(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise RuntimeError(f"evaluation reservation must not be a symlink: {path}")
    payload = _load_json_object(path, label="evaluation reservation")
    if (
        payload.get("schema") != RESERVATION_SCHEMA
        or payload.get("status") != "reserved"
        or not isinstance(payload.get("run_id"), str)
        or not isinstance(payload.get("token_sha256"), str)
        or not isinstance(payload.get("ownership"), dict)
        or not isinstance(payload.get("owner"), dict)
    ):
        raise RuntimeError(f"invalid evaluation reservation: {path}")
    return payload


def reserve_artifact_run(
    output_dir: str | Path,
    ownership: Mapping[str, Any],
    *,
    preflight: Callable[[], None] | None = None,
) -> ArtifactRunReservation:
    """在 output-dir 锁内持久化可验证 owner 对一次评估启动的占有权。"""
    configured = Path(output_dir).expanduser()
    if configured.is_symlink():
        raise ValueError("evaluation reservation output directory must not be a symlink")
    normalized = normalize_ownership_binding(ownership)
    owner = _reservation_owner()
    run_id = new_run_id()
    token = secrets.token_urlsafe(32)
    with inference_output_lock(configured) as output:
        path = output / RESERVATION_FILENAME
        if path.exists() or path.is_symlink():
            existing = _reservation_payload(path)
            if not _reservation_owner_is_dead(existing["owner"]):
                raise RuntimeError("evaluation output has an active reservation")
            path.unlink()
        if _has_in_progress_run(output):
            raise RuntimeError("evaluation output has an active in-progress run")
        if preflight is not None:
            preflight()
        atomic_write_json(
            path,
            {
                "schema": RESERVATION_SCHEMA,
                "status": "reserved",
                "run_id": run_id,
                "token_sha256": _token_digest(token),
                "ownership": normalized,
                "owner": owner,
            },
        )
    return ArtifactRunReservation(output, run_id, token, normalized, owner)


def claim_artifact_run(
    output_dir: str | Path, token: str, ownership: Mapping[str, Any]
) -> ArtifactRunStart:
    """child 在锁内校验 reservation、捕获 predecessor 并发布 CURRENT。"""
    if not token:
        raise RuntimeError("evaluation reservation token is required")
    normalized = normalize_ownership_binding(ownership)
    with inference_output_lock(output_dir) as output:
        path = output / RESERVATION_FILENAME
        if not path.exists() and not path.is_symlink():
            raise RuntimeError("evaluation reservation does not exist")
        payload = _reservation_payload(path)
        if not hmac.compare_digest(payload["token_sha256"], _token_digest(token)):
            raise RuntimeError("evaluation reservation token mismatch")
        if not _binding_matches(payload["ownership"], normalized):
            raise RuntimeError("evaluation reservation binding mismatch")
        run_id = payload["run_id"]
        previous = _capture_committed_artifact_unlocked(output)
        path.unlink()
        begin_artifact_run(output, run_id)
        return ArtifactRunStart(output, run_id, previous)


def begin_unreserved_artifact_run(
    output_dir: str | Path,
    run_id: str,
    *,
    preflight: Callable[[], None] | None = None,
) -> ArtifactRunStart:
    """在锁内完成 preflight、predecessor 捕获与 direct run 发布。"""
    with inference_output_lock(output_dir) as output:
        reservation_path = output / RESERVATION_FILENAME
        if reservation_path.exists() or reservation_path.is_symlink():
            raise RuntimeError("evaluation output has an active reservation")
        if _has_in_progress_run(output):
            raise RuntimeError("evaluation output has an active in-progress run")
        if preflight is not None:
            preflight()
        previous = _capture_committed_artifact_unlocked(output)
        begin_artifact_run(output, run_id)
        return ArtifactRunStart(output, run_id, previous)


def release_artifact_reservation(reservation: ArtifactRunReservation) -> bool:
    """仅释放调用方自己的未认领 reservation；已 claim 或已换代时不回滚。"""
    with inference_output_lock(reservation.output_dir) as output:
        path = output / RESERVATION_FILENAME
        if not path.exists() and not path.is_symlink():
            return False
        payload = _reservation_payload(path)
        if (
            payload["run_id"] != reservation.run_id
            or payload["ownership"] != reservation.ownership
            or payload["owner"] != reservation.owner
            or not hmac.compare_digest(
                payload["token_sha256"], _token_digest(reservation.token)
            )
        ):
            return False
        current_path = output / CURRENT_FILENAME
        if current_path.exists():
            current = _load_json_object(current_path, label="evaluation CURRENT")
            if (
                current.get("status") == "in_progress"
                and current.get("run_id") == reservation.run_id
            ):
                return False
        path.unlink()
        return True


def new_run_id() -> str:
    """生成一次评估生命周期内共享的不可碰撞 run id。"""
    return uuid4().hex


def attach_run_id(summary: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    """复制 summary 并绑定本次生命周期标识。"""
    payload = dict(summary)
    existing = payload.get("run_id")
    if existing is not None and existing != run_id:
        raise ValueError(
            f"summary run_id mismatch: expected={run_id!r}, actual={existing!r}"
        )
    payload["run_id"] = run_id
    return payload


def build_artifact_envelope(
    *,
    task: str,
    pipeline: str,
    step: str,
    status: str,
    provenance: Mapping[str, Any],
    inference_coverage: Any,
    metric_coverage: Any,
    summary: Mapping[str, Any],
    run_id: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """构建跨任务稳定、指标载荷保持任务专属的评估 envelope。"""
    resolved_run_id = run_id or str(summary.get("run_id") or new_run_id())
    bound_summary = attach_run_id(summary, resolved_run_id)
    envelope: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "run_id": resolved_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "pipeline": pipeline,
        "step": step,
        "status": status,
        "provenance": dict(provenance),
        "inference_coverage": inference_coverage,
        "metric_coverage": metric_coverage,
        "summary": bound_summary,
    }
    if error is not None:
        envelope["error"] = error
    return envelope


def publish_artifact_envelope(
    output_dir: str | Path,
    envelope: Mapping[str, Any],
) -> Path:
    """原子写入固定文件名，不替代各任务已有 summary/sidecar。"""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / ARTIFACT_FILENAME
    atomic_write_json(path, dict(envelope))
    return path


def begin_artifact_run(output_dir: str | Path, run_id: str) -> Path:
    """在 run 开始时失效旧提交，避免旧 envelope 被当作本次结果消费。"""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / CURRENT_FILENAME
    atomic_write_json(
        path,
        {
            "schema": CURRENT_SCHEMA,
            "status": "in_progress",
            "run_id": run_id,
            "owner": _reservation_owner(),
        },
    )
    return path


def _read_descriptor_snapshot(
    descriptor: int, *, path: Path, label: str
) -> tuple[str, int, str]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"committed {label} is not a regular file: {path}")
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    after = os.fstat(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise RuntimeError(f"committed {label} changed while reading: {path}")
    content = b"".join(chunks)
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"committed {label} must be UTF-8: {path}") from exc
    return text, len(content), hashlib.sha256(content).hexdigest()


def _read_text_snapshot(path: Path, *, label: str) -> tuple[str, int, str]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise RuntimeError(f"cannot read committed {label}: {path}") from exc
    try:
        return _read_descriptor_snapshot(descriptor, path=path, label=label)
    finally:
        os.close(descriptor)


def _validated_relative_parts(relative: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise RuntimeError(f"invalid committed {label} path: {relative!r}")
    relative_path = Path(relative)
    raw_parts = relative.split("/")
    if (
        relative_path.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_parts)
        or tuple(raw_parts) != relative_path.parts
    ):
        raise RuntimeError(f"invalid committed {label} path: {relative!r}")
    return relative_path.parts


def _read_manifested_text_snapshot(
    output: Path, relative: Any, *, label: str
) -> tuple[Path, str, int, str]:
    parts = _validated_relative_parts(relative, label=label)
    path = output.joinpath(*parts)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptors: list[int] = []
    try:
        descriptors.append(os.open(output, directory_flags))
        for part in parts[:-1]:
            descriptors.append(os.open(part, directory_flags, dir_fd=descriptors[-1]))
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=descriptors[-1],
        )
        descriptors.append(descriptor)
        text, size, digest = _read_descriptor_snapshot(
            descriptor, path=path, label=label
        )
    except OSError as exc:
        raise RuntimeError(f"cannot read committed {label}: {path}") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    return path, text, size, digest


def _parse_json_snapshot(
    text: str, *, path: Path, label: str, size: int, digest: str
) -> tuple[dict[str, Any], str, int, str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid {label}: expected JSON object: {path}")
    return payload, text, size, digest


def _load_json_snapshot(
    path: Path, *, label: str
) -> tuple[dict[str, Any], str, int, str]:
    text, size, digest = _read_text_snapshot(path, label=label)
    return _parse_json_snapshot(
        text, path=path, label=label, size=size, digest=digest
    )


def _load_manifested_json_snapshot(
    output: Path, relative: Any, *, label: str
) -> tuple[Path, dict[str, Any], str, int, str]:
    path, text, size, digest = _read_manifested_text_snapshot(
        output, relative, label=label
    )
    payload, text, size, digest = _parse_json_snapshot(
        text, path=path, label=label, size=size, digest=digest
    )
    return path, payload, text, size, digest


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    return _load_json_snapshot(path, label=label)[0]


def validate_artifact_summary_filename(filename: Any) -> str:
    """校验 summary 为 output 根目录下且不与 lifecycle 文件重叠的 basename。"""
    reserved = {
        CURRENT_FILENAME,
        ARTIFACT_FILENAME,
        RESERVATION_FILENAME,
        _RESULT_MANIFEST_FILENAME,
        _OUTPUT_LOCK_FILENAME,
    }
    if (
        not isinstance(filename, str)
        or not filename
        or "/" in filename
        or "\\" in filename
        or filename in {".", ".."}
        or filename in reserved
    ):
        raise ValueError(f"invalid evaluation summary filename: {filename!r}")
    return filename


def _committed_filename(filename: Any, *, label: str) -> str:
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise RuntimeError(f"invalid committed {label} filename: {filename!r}")
    return filename


def _committed_reference_filename(filename: Any, *, label: str) -> str:
    relative = _committed_filename(filename, label=label)
    if label == "summary":
        try:
            return validate_artifact_summary_filename(relative)
        except ValueError as exc:
            raise RuntimeError(f"invalid committed summary filename: {relative!r}") from exc
    if label == "envelope" and relative != ARTIFACT_FILENAME:
        raise RuntimeError(f"invalid committed envelope filename: {relative!r}")
    return relative


def _validate_result_manifest(
    output: Path,
    summary: Mapping[str, Any],
    *,
    expected_run_id: str,
    summary_filename: str,
) -> tuple[Path, dict[str, Any], str, dict[str, str]]:
    reference = summary.get("result_manifest")
    if not isinstance(reference, Mapping):
        raise RuntimeError("committed evaluation summary has no result manifest reference")
    if reference.get("filename") != _RESULT_MANIFEST_FILENAME:
        raise RuntimeError("committed evaluation result manifest filename is invalid")
    manifest_path, manifest, manifest_text, manifest_size, manifest_digest = (
        _load_manifested_json_snapshot(
            output,
            _committed_filename(
                reference.get("filename"), label="result manifest"
            ),
            label="evaluation result manifest",
        )
    )
    if (
        reference.get("sha256") != manifest_digest
        or reference.get("size") != manifest_size
    ):
        raise RuntimeError("committed evaluation result manifest digest mismatch")
    if (
        manifest.get("schema") != _RESULT_MANIFEST_SCHEMA
        or manifest.get("run_id") != expected_run_id
        or not isinstance(manifest.get("files"), dict)
    ):
        raise RuntimeError("committed evaluation result manifest binding is invalid")
    reserved = {
        CURRENT_FILENAME,
        ARTIFACT_FILENAME,
        RESERVATION_FILENAME,
        _RESULT_MANIFEST_FILENAME,
        _OUTPUT_LOCK_FILENAME,
        summary_filename,
    }
    artifacts: dict[str, str] = {}
    for relative, expected in manifest["files"].items():
        if not isinstance(expected, Mapping) or relative in reserved:
            raise RuntimeError(f"invalid committed metric artifact entry: {relative!r}")
        _path, content, size, digest = _read_manifested_text_snapshot(
            output, relative, label="metric artifact"
        )
        if expected.get("sha256") != digest or expected.get("size") != size:
            raise RuntimeError(f"committed metric artifact digest mismatch: {relative!r}")
        artifacts[relative] = content
    return manifest_path, manifest, manifest_text, artifacts


def _capture_committed_artifact_unlocked(output: Path) -> CommittedArtifactSnapshot | None:
    current_path = output / CURRENT_FILENAME
    if not current_path.is_file():
        return None
    current, current_text, _current_size, _current_digest = _load_json_snapshot(
        current_path, label="evaluation CURRENT"
    )
    if current.get("schema") != CURRENT_SCHEMA or current.get("status") != "committed":
        return None
    run_id = current.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("committed evaluation CURRENT has no run_id")
    files = {CURRENT_FILENAME: current_text}
    loaded: dict[str, dict[str, Any]] = {}
    summary_filename = ""
    for label in ("summary", "envelope"):
        reference = current.get(label)
        if not isinstance(reference, Mapping):
            raise RuntimeError(f"committed evaluation CURRENT has no {label} reference")
        relative = _committed_reference_filename(
            reference.get("filename"), label=label
        )
        path, payload, text, _size, digest = _load_manifested_json_snapshot(
            output, relative, label=f"evaluation {label}"
        )
        if digest != reference.get("sha256"):
            raise RuntimeError(f"committed evaluation {label} digest mismatch: {path}")
        if payload.get("run_id") != run_id:
            raise RuntimeError(
                f"committed evaluation {label} run_id mismatch: "
                f"expected={run_id!r}, actual={payload.get('run_id')!r}"
            )
        files[relative] = text
        loaded[label] = payload
        if label == "summary":
            summary_filename = relative
    committed_summary = loaded["summary"].get("summary", loaded["summary"])
    if (
        not isinstance(committed_summary, Mapping)
        or committed_summary != loaded["envelope"].get("summary")
    ):
        raise RuntimeError("committed evaluation summary/envelope mismatch")
    result_reference = committed_summary.get("result_manifest")
    if result_reference is not None:
        result_path, _manifest, manifest_text, artifacts = _validate_result_manifest(
            output,
            committed_summary,
            expected_run_id=run_id,
            summary_filename=summary_filename,
        )
        files[result_path.relative_to(output).as_posix()] = manifest_text
        files.update(artifacts)
    return CommittedArtifactSnapshot(output, files)


def capture_committed_artifact(
    output_dir: str | Path,
) -> CommittedArtifactSnapshot | None:
    """在 output lock 内捕获当前完整 commit。"""
    with inference_output_lock(output_dir) as output:
        return _capture_committed_artifact_unlocked(output)


def _restore_committed_artifact_unlocked(
    output: Path, snapshot: CommittedArtifactSnapshot
) -> None:
    if snapshot.output_dir != output or CURRENT_FILENAME not in snapshot.files:
        raise RuntimeError("evaluation predecessor snapshot binding is invalid")
    artifact_files: list[tuple[tuple[str, ...], str]] = []
    for relative, content in snapshot.files.items():
        if not isinstance(content, str):
            raise RuntimeError("evaluation predecessor snapshot content is invalid")
        parts = _validated_relative_parts(relative, label="snapshot artifact")
        if relative in {RESERVATION_FILENAME, _OUTPUT_LOCK_FILENAME}:
            raise RuntimeError("evaluation predecessor snapshot path is reserved")
        if relative != CURRENT_FILENAME:
            artifact_files.append((parts, content))
    artifact_files.append(((CURRENT_FILENAME,), snapshot.files[CURRENT_FILENAME]))
    write_text_files_transactionally(
        [
            (
                output.joinpath(*parts),
                lambda stream, content=content: stream.write(content),
            )
            for parts, content in artifact_files
        ],
        overwrite=True,
    )


def abort_artifact_run(
    output_dir: str | Path,
    *,
    run_id: str,
    previous_commit: CommittedArtifactSnapshot | None,
    reason: str,
) -> bool:
    """以 run-id CAS 终止失败 run；有 predecessor 时原子恢复完整提交集合。"""
    with inference_output_lock(output_dir) as output:
        current_path = output / CURRENT_FILENAME
        if not current_path.is_file():
            return False
        current = _load_json_object(current_path, label="evaluation CURRENT")
        if current.get("run_id") != run_id or current.get("status") != "in_progress":
            return False
        if previous_commit is not None:
            if previous_commit.output_dir != output:
                raise RuntimeError("evaluation predecessor snapshot output mismatch")
            _restore_committed_artifact_unlocked(output, previous_commit)
        else:
            atomic_write_json(
                current_path,
                {
                    "schema": CURRENT_SCHEMA,
                    "status": "aborted",
                    "run_id": run_id,
                    "reason": reason,
                },
            )
    return True


def _artifact_output_lock(
    output_dir: str | Path, *, lock_held: bool
):
    return nullcontext(Path(output_dir).expanduser().resolve()) if lock_held else inference_output_lock(output_dir)


def publish_artifact_commit(
    output_dir: str | Path,
    *,
    run_id: str,
    summary_filename: str,
    precommit: Callable[[], None] | None = None,
    require_result_manifest: bool = False,
    lock_held: bool = False,
) -> Path:
    """在 output lock 内复核任务资产与整组发布文件，最后提交 CURRENT。"""
    with _artifact_output_lock(output_dir, lock_held=lock_held) as output:
        current_path = output / CURRENT_FILENAME
        current = _load_json_object(current_path, label="evaluation CURRENT")
        if (
            current.get("schema") != CURRENT_SCHEMA
            or current.get("status") != "in_progress"
            or current.get("run_id") != run_id
        ):
            raise RuntimeError(f"evaluation run is not active: run_id={run_id!r}")
        summary_filename = _committed_reference_filename(
            summary_filename, label="summary"
        )
        summary_path, summary, _summary_text, _summary_size, summary_digest = (
            _load_manifested_json_snapshot(
                output, summary_filename, label="evaluation summary"
            )
        )
        envelope_path, envelope, _envelope_text, _envelope_size, envelope_digest = (
            _load_manifested_json_snapshot(
                output, ARTIFACT_FILENAME, label="evaluation envelope"
            )
        )
        if summary.get("run_id") != run_id:
            raise RuntimeError(
                f"summary run_id mismatch: expected={run_id!r}, "
                f"actual={summary.get('run_id')!r}"
            )
        if envelope.get("run_id") != run_id:
            raise RuntimeError(
                f"envelope run_id mismatch: expected={run_id!r}, "
                f"actual={envelope.get('run_id')!r}"
            )
        embedded_summary = envelope.get("summary")
        persisted_summary = summary.get("summary", summary)
        if (
            not isinstance(persisted_summary, Mapping)
            or not isinstance(embedded_summary, Mapping)
            or dict(embedded_summary) != dict(persisted_summary)
        ):
            raise RuntimeError("evaluation summary and envelope embedded summary mismatch")
        summary_provenance = persisted_summary.get("provenance")
        envelope_provenance = envelope.get("provenance")
        if summary_provenance is not None:
            if not isinstance(summary_provenance, Mapping) or not isinstance(
                envelope_provenance, Mapping
            ):
                raise RuntimeError("evaluation summary/envelope provenance is invalid")
            if dict(summary_provenance) != dict(envelope_provenance):
                raise RuntimeError("evaluation summary and envelope provenance mismatch")
        effective_identity = persisted_summary.get("effective_config_identity")
        if effective_identity is not None:
            if not isinstance(effective_identity, Mapping) or not isinstance(
                envelope_provenance, Mapping
            ):
                raise RuntimeError("evaluation effective config provenance is invalid")
            if dict(effective_identity) != envelope_provenance.get(
                "resolved_config_fingerprint"
            ):
                raise RuntimeError("evaluation effective config provenance mismatch")
        if require_result_manifest or persisted_summary.get("result_manifest") is not None:
            _validate_result_manifest(
                output,
                persisted_summary,
                expected_run_id=run_id,
                summary_filename=summary_filename,
            )
        if precommit is not None:
            precommit()
        # precommit 是外部 seam；返回后必须对磁盘上的最终集合再验证一次，
        # 防止回调修改 metric artifact 后仍提交不可读的 committed CURRENT。
        _, final_summary, _, _, final_summary_digest = _load_manifested_json_snapshot(
            output, summary_filename, label="evaluation summary"
        )
        _, final_envelope, _, _, final_envelope_digest = _load_manifested_json_snapshot(
            output, ARTIFACT_FILENAME, label="evaluation envelope"
        )
        if final_summary.get("run_id") != run_id or final_envelope.get("run_id") != run_id:
            raise RuntimeError("evaluation final snapshot run_id mismatch")
        if final_envelope.get("summary") != final_summary.get("summary", final_summary):
            raise RuntimeError("evaluation final summary/envelope mismatch")
        final_persisted_summary = final_summary.get("summary", final_summary)
        if require_result_manifest or final_persisted_summary.get("result_manifest") is not None:
            _validate_result_manifest(
                output,
                final_persisted_summary,
                expected_run_id=run_id,
                summary_filename=summary_filename,
            )
        summary_digest = final_summary_digest
        envelope_digest = final_envelope_digest

        atomic_write_json(
            current_path,
            {
                "schema": CURRENT_SCHEMA,
                "status": "committed",
                "run_id": run_id,
                "summary": {
                    "filename": summary_filename,
                    "sha256": summary_digest,
                },
                "envelope": {
                    "filename": ARTIFACT_FILENAME,
                    "sha256": envelope_digest,
                },
            },
        )
        return current_path


def load_committed_artifact(output_dir: str | Path) -> dict[str, Any]:
    """只读取 CURRENT 最终提交且摘要、run_id 全部一致的 envelope。"""
    configured = Path(output_dir).expanduser()
    if configured.is_symlink():
        raise RuntimeError("evaluation output directory must not be a symlink")
    output = configured.resolve()
    _current_path, current, _text, _size, current_digest = (
        _load_manifested_json_snapshot(
            output, CURRENT_FILENAME, label="evaluation CURRENT"
        )
    )
    if current.get("schema") != CURRENT_SCHEMA:
        raise RuntimeError(
            f"unsupported evaluation CURRENT schema: {current.get('schema')!r}"
        )
    if current.get("status") != "committed":
        raise RuntimeError(
            f"evaluation run is not committed: status={current.get('status')!r}"
        )
    run_id = current.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("committed evaluation CURRENT has no run_id")

    loaded: dict[str, dict[str, Any]] = {}
    summary_filename = ""
    for label in ("summary", "envelope"):
        reference = current.get(label)
        if not isinstance(reference, Mapping):
            raise RuntimeError(f"committed evaluation CURRENT has no {label} reference")
        relative = _committed_reference_filename(
            reference.get("filename"), label=label
        )
        path, payload, _text, _size, digest = _load_manifested_json_snapshot(
            output, relative, label=f"evaluation {label}"
        )
        if not isinstance(reference.get("sha256"), str) or digest != reference.get(
            "sha256"
        ):
            raise RuntimeError(f"committed evaluation {label} digest mismatch: {path}")
        if payload.get("run_id") != run_id:
            raise RuntimeError(
                f"committed evaluation {label} run_id mismatch: "
                f"expected={run_id!r}, actual={payload.get('run_id')!r}"
            )
        loaded[label] = payload
        if label == "summary":
            summary_filename = relative
    persisted_summary = loaded["summary"].get("summary", loaded["summary"])
    if (
        not isinstance(persisted_summary, Mapping)
        or persisted_summary != loaded["envelope"].get("summary")
    ):
        raise RuntimeError("committed evaluation summary/envelope mismatch")
    if persisted_summary.get("result_manifest") is not None:
        _validate_result_manifest(
            output,
            persisted_summary,
            expected_run_id=run_id,
            summary_filename=summary_filename,
        )
    _path, final_current, _text, _size, final_digest = (
        _load_manifested_json_snapshot(
            output, CURRENT_FILENAME, label="evaluation CURRENT"
        )
    )
    if final_digest != current_digest or final_current != current:
        raise RuntimeError("evaluation CURRENT changed while loading committed artifact")
    return loaded["envelope"]
