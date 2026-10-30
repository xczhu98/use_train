"""Generation-3 Step3 单次 GPU 评分事务的 owned 深 Module。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import selectors
import stat
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..mos import normalize_selectable_mos_models, verify_urgent_pk_trusted_assets
from ..mos_contract import (
    MOS_ASSET_LOAD_BINDING,
    MOS_ASSET_PROVENANCE_FIELDS,
    MOS_MODEL_REQUEST_OPTION_FIELDS,
    MOS_PROVIDER_ATTESTATION_FIELDS,
    MOS_SCORING_EXECUTION_POLICY,
    MOS_SHIPPED_MODELS,
    MOS_WORKER_ATTESTATION_FIELDS,
    MOS_WORKER_ORDINAL_SIDECAR_FIELDS,
    MOS_WORKER_MODEL_TUPLE_ATTESTATION_FIELDS,
    MOS_WORKER_PROTOCOL,
    MOS_WORKER_TUPLE_PROVIDER_ATTESTATION_FIELDS,
    is_mos_device_attestation,
    validate_mos_provider_attestation,
)
from .mos_assets import selected_mos_asset_provenance
from .mos_lpt import (
    MAX_WORKER_INPUT_SUBSHARDS,
    STEP3_JSONL_MAX_FILE_BYTES,
    _DiskBackedLptPlanner,
    _DiskBackedResultStore,
    _LPT_BATCH_MAX_BYTES,
    _LPT_BATCH_MAX_ROWS,
    _MAX_UINT64,
)
from .mos_resume import Step3ResumeStore, step3_task_identity
from .mos_runtime import Step3MosRuntime
from .select import Step3MosSelectionStage, TUPLE_SPOOL_MAX_FILE_BYTES
from .wespeaker_runtime import (
    BoundedSubprocessError,
    BoundedSubprocessOutputLimitError,
    BoundedSubprocessTimeoutError,
    run_bounded_subprocess,
    terminate_process_group,
)

__all__ = ["_GpuScoringJobSettings", "_OwnedGpuScoringJob"]

_PROTOCOL = MOS_WORKER_PROTOCOL
_MODELS = MOS_SHIPPED_MODELS
ASSET_LOAD_BINDING = MOS_ASSET_LOAD_BINDING
_JSONL_MAX_LINE_BYTES = 8 * 1024 * 1024
_JSONL_MAX_FILE_BYTES = STEP3_JSONL_MAX_FILE_BYTES
_RUNTIME_ATTESTATION_MAX_BYTES = 1024 * 1024
# 单个 request（即一个 sealed shard）的 descriptor 以单行 canonical JSON 参与
# resume key、sealed attestation（_MAX_JSON_BYTES）与 shard_id 派生，必须显著低于
# 8 MiB 单行上限；按精确字节预算把过大的 request 切成多个 part。
_REQUEST_DESCRIPTOR_BUDGET_BYTES = 4 * 1024 * 1024
_CHECKPOINT_ATTESTATION_MODELS = frozenset({"nisqa", "utmos"})
_ORDINAL_SIDECAR_FIELDS = MOS_WORKER_ORDINAL_SIDECAR_FIELDS


from ...utils.strict_jsonl import (
    add_cleanup_note as _add_cleanup_note,
    decode_json_line as _shared_decode_json_line,
    encode_json_line as _shared_encode_json_line,
    file_signature as _file_signature,
    finalize_streams as _finalize_streams,
    iter_bounded_jsonl as _shared_iter_bounded_jsonl,
    open_new_regular_stream as _open_new_regular_stream,
    open_regular_stream as _open_regular_stream,
    record_cleanup_error as _record_cleanup_error,
)


class _FrozenList(tuple):
    """保留 JSON array 相等语义的不可变序列。"""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return tuple(self) == tuple(other)
        return False

    __hash__ = tuple.__hash__


def _freeze_json(value: Any, label: str) -> Any:
    """严格复制并冻结注入的 JSON provenance。"""
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError(f"{label} object keys must be strings")
        return MappingProxyType({
            key: _freeze_json(item, f"{label}.{key}")
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return _FrozenList(
            _freeze_json(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{label} must contain only JSON values")


def _plain_json(value: Any) -> Any:
    """返回与内部冻结快照隔离的 JSON-safe 副本。"""
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _tree_sha256(value: Any, label: str) -> str:
    """校验真实 loader 所消费的模型树内容身份。"""
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be 64 lowercase hex characters")
    return value


def _validated_urgent_pk_assets(value: Any, label: str) -> dict[str, str]:
    """校验 URGENT-PK checkpoint、commit 与 pinned package 身份。"""
    expected_fields = MOS_ASSET_PROVENANCE_FIELDS["urgent_pk"]
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected_fields
    ):
        raise ValueError(f"{label} fields are invalid")
    checkpoint_sha256 = _tree_sha256(
        value["checkpoint_sha256"], f"{label} checkpoint_sha256",
    )
    source_package_sha256 = _tree_sha256(
        value["source_package_sha256"], f"{label} source_package_sha256",
    )
    source_commit = value["source_commit"]
    if (
        type(source_commit) is not str
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError(f"{label} source_commit is invalid")
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "source_commit": source_commit,
        "source_package_sha256": source_package_sha256,
    }


def _close_process_streams(proc: Any) -> None:
    first_error = None
    for name in ("stdin", "stdout", "stderr"):
        try:
            stream = getattr(proc, name, None)
        except BaseException as exc:
            first_error = _record_cleanup_error(
                first_error, exc, f"process {name} lookup",
            )
            continue
        if stream is None:
            continue
        try:
            stream.close()
        except BaseException as exc:
            first_error = _record_cleanup_error(
                first_error, exc, f"process {name} close",
            )
    if first_error is not None:
        raise first_error


def _close_iterator(iterator: object) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def _finish_iterator(
    iterator: object,
    label: str,
    primary: BaseException | None,
) -> None:
    try:
        _close_iterator(iterator)
    except BaseException as cleanup_error:
        if isinstance(primary, GeneratorExit):
            raise cleanup_error
        if primary is not None:
            _add_cleanup_note(primary, label, cleanup_error)
        else:
            raise


def _encode_json_line(value: Mapping[str, Any], label: str) -> bytes:
    return _shared_encode_json_line(
        value, label, max_line_bytes=_JSONL_MAX_LINE_BYTES,
    )


def _decode_json_line(payload: bytes, label: str) -> Any:
    return _shared_decode_json_line(
        payload, label, max_line_bytes=_JSONL_MAX_LINE_BYTES,
    )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _descriptor_skeleton_bytes(models: Sequence[str]) -> int:
    """request descriptor 固定骨架的 canonical JSON 字节数。

    shard_id 恒为 "r-" + 62 hex、scoring_policy_hash 恒为 64 hex，占位长度与
    真实值一致；配合逐行 ``len + 1`` 的元素增量记账，整体是对真实 descriptor
    编码长度的保守上界（每 request 固定高估 3 字节）。
    """
    skeleton = {
        "models": list(models),
        "ordinals": [],
        "scoring_policy_hash": "x" * 64,
        "shard_id": "x" * 64,
        "task_bindings": [],
        "task_identities": [],
    }
    return len(_canonical_json_bytes(skeleton))


def _validate_runtime_provider_entry(
    model: str,
    value: Any,
    expected_checkpoint_sha256: str | None = None,
) -> None:
    """按模型校验 worker 报告的 provider exact schema 与 checkpoint 身份。"""
    if model in {"scoreq", "dnsmos"}:
        validate_mos_provider_attestation(
            model,
            value,
            requested="cuda",
            schema_label=f"MOS worker {model} provider schema",
        )
    else:
        expected_fields = MOS_PROVIDER_ATTESTATION_FIELDS[model]
        if (
            type(value) is not dict
            or any(type(key) is not str for key in value)
            or set(value) != expected_fields
        ):
            raise ValueError(f"MOS worker {model} provider schema is invalid")
        if (
            not is_mos_device_attestation(
                value["device"], requested="cuda",
            )
            or not is_mos_device_attestation(
                value["parameters"], requested="cuda",
            )
        ):
            raise ValueError(f"MOS worker {model} provider values are invalid")
        if model in _CHECKPOINT_ATTESTATION_MODELS:
            checkpoint_sha256 = _tree_sha256(
                value["checkpoint_sha256"],
                f"MOS worker {model} checkpoint_sha256",
            )
            expected_checkpoint_sha256 = _tree_sha256(
                expected_checkpoint_sha256,
                f"MOS parent expected {model} checkpoint_sha256",
            )
            if checkpoint_sha256 != expected_checkpoint_sha256:
                raise ValueError(
                    f"MOS worker {model} checkpoint digest mismatch: "
                    f"expected={expected_checkpoint_sha256}, "
                    f"actual={checkpoint_sha256}"
                )


def _read_runtime_attestation(
    path: str,
    expected_models: tuple[str, ...],
    expected_checkpoint_sha256: Mapping[str, str] | None = None,
    expected_urgent_pk_assets: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """读取 bounded worker sidecar，并绑定 parent request 资产身份。"""
    expected_checkpoint_sha256 = expected_checkpoint_sha256 or {}
    required_checkpoint_models = set(expected_models) & _CHECKPOINT_ATTESTATION_MODELS
    if (
        not isinstance(expected_checkpoint_sha256, Mapping)
        or any(type(key) is not str for key in expected_checkpoint_sha256)
        or set(expected_checkpoint_sha256) != required_checkpoint_models
    ):
        raise ValueError(
            "MOS parent expected checkpoint attestation models are invalid"
        )
    for model in required_checkpoint_models:
        _tree_sha256(
            expected_checkpoint_sha256[model],
            f"MOS parent expected {model} checkpoint_sha256",
        )
    urgent_pk_required = "urgent_pk" in expected_models
    if urgent_pk_required:
        expected_urgent_pk_assets = _validated_urgent_pk_assets(
            expected_urgent_pk_assets, "MOS parent expected urgent_pk assets",
        )
    elif expected_urgent_pk_assets is not None:
        raise ValueError("MOS parent expected urgent_pk assets are invalid")
    payloads = list(_shared_iter_bounded_jsonl(
        path,
        "MOS worker runtime attestation",
        max_line_bytes=_RUNTIME_ATTESTATION_MAX_BYTES,
        max_file_bytes=_RUNTIME_ATTESTATION_MAX_BYTES,
    ))
    if len(payloads) != 1:
        raise ValueError("MOS worker runtime attestation must contain exactly one row")
    value = _shared_decode_json_line(
        payloads[0],
        "MOS worker runtime attestation",
        max_line_bytes=_RUNTIME_ATTESTATION_MAX_BYTES,
    )
    if (
        not isinstance(value, dict)
        or set(value) != MOS_WORKER_ATTESTATION_FIELDS
    ):
        raise ValueError("MOS worker runtime attestation schema is invalid")
    if value["protocol"] != _PROTOCOL:
        raise ValueError("MOS worker runtime attestation protocol is invalid")
    if value["asset_load_binding"] != _plain_json(ASSET_LOAD_BINDING):
        raise ValueError("MOS worker runtime attestation binding is invalid")
    if value["scoring_execution_policy"] != _plain_json(
        MOS_SCORING_EXECUTION_POLICY,
    ):
        raise ValueError("MOS worker runtime attestation scoring policy is invalid")
    if value["worker_models"] != list(expected_models):
        raise ValueError("MOS worker runtime attestation models are invalid")
    items = value["model_tuple_attestations"]
    if not isinstance(items, list) or not items:
        raise ValueError("MOS worker runtime attestation tuples are invalid")
    observed_models = []
    seen = set()
    for item in items:
        if (
            not isinstance(item, dict)
            or set(item) != MOS_WORKER_MODEL_TUPLE_ATTESTATION_FIELDS
        ):
            raise ValueError("MOS worker model tuple attestation schema is invalid")
        models = normalize_selectable_mos_models(item["models"])
        if models in seen or not set(models).issubset(expected_models):
            raise ValueError("MOS worker model tuple attestation models are invalid")
        attestation = item["attestation"]
        expected_fields = set(MOS_WORKER_TUPLE_PROVIDER_ATTESTATION_FIELDS)
        if "urgent_pk" in models:
            expected_fields.add("urgent_pk_assets")
        if not isinstance(attestation, dict) or set(attestation) != expected_fields:
            raise ValueError("MOS worker provider attestation schema is invalid")
        if attestation["scoring_execution_policy"] != _plain_json(
            MOS_SCORING_EXECUTION_POLICY,
        ):
            raise ValueError(
                "MOS worker provider attestation scoring policy is invalid"
            )
        if (
            attestation["requested_device"] != "cuda"
            or attestation["fallback"] is not False
            or attestation["selected_models"] != list(models)
            or attestation["asset_load_binding"] != _plain_json(ASSET_LOAD_BINDING)
            or not isinstance(attestation["actual_providers"], dict)
            or set(attestation["actual_providers"]) != set(models)
        ):
            raise ValueError("MOS worker provider attestation values are invalid")
        for model in models:
            _validate_runtime_provider_entry(
                model,
                attestation["actual_providers"][model],
                expected_checkpoint_sha256.get(model),
            )
        if "urgent_pk" in models:
            assets = _validated_urgent_pk_assets(
                attestation["urgent_pk_assets"], "MOS worker urgent_pk assets",
            )
            for field in MOS_ASSET_PROVENANCE_FIELDS["urgent_pk"]:
                if assets[field] != expected_urgent_pk_assets[field]:
                    raise ValueError(f"MOS worker urgent_pk {field} mismatch")
        seen.add(models)
        for model in models:
            if model not in observed_models:
                observed_models.append(model)
    if tuple(observed_models) != expected_models:
        raise ValueError("MOS worker runtime attestation coverage is invalid")
    return value


def _iter_bounded_jsonl(path: str, label: str) -> Iterator[bytes]:
    return _shared_iter_bounded_jsonl(
        path,
        label,
        max_line_bytes=_JSONL_MAX_LINE_BYTES,
        max_file_bytes=_JSONL_MAX_FILE_BYTES,
    )


def _iter_tuple_task_spool(path: str) -> Iterator[bytes]:
    return _shared_iter_bounded_jsonl(
        path,
        "Step3 tuple task spool",
        max_line_bytes=_JSONL_MAX_LINE_BYTES,
        max_file_bytes=TUPLE_SPOOL_MAX_FILE_BYTES,
    )


def _pathname_signature(path: str, label: str) -> tuple[int, int, int, int, int]:
    try:
        value = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        raise RuntimeError(f"{label} is missing") from None
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return _file_signature(value)


_HEARTBEAT_PROTOCOL = b"audio_filter_pipeline.step3_mos_worker.heartbeat.v1\n"
_PROGRESS_PREFIX = b"audio_filter_pipeline.step3_mos_worker.progress.v1:"
_HEARTBEAT_STDOUT_LIMIT_BYTES = 4 * 1024 * 1024


def _heartbeat_launcher_payload() -> bytes:
    """返回显式启动 owned worker 的稳定 launcher。"""
    resolved_path = Path(__file__).resolve()
    package_parent = str(resolved_path.parents[3])
    project_root = str(resolved_path.parents[4])
    source = f'''\
import sys

sys.path.insert(0, {project_root!r})
sys.path.insert(0, {package_parent!r})
from audio_filter_pipeline.processors.workers import step3_mos_worker

raise SystemExit(step3_mos_worker.main(sys.argv[1:]))
'''
    return source.encode("utf-8")


def _write_heartbeat_launcher(directory: str) -> str:
    """在 owned 私有目录写入 heartbeat worker launcher。"""
    path = os.path.join(directory, "heartbeat_launcher.py")
    stream = _open_new_regular_stream(path, "MOS worker heartbeat launcher")
    try:
        stream.write(_heartbeat_launcher_payload())
    finally:
        _finalize_streams((stream,), sys.exc_info()[1], sync=True)
    return path


@dataclass(frozen=True)
class _GpuScoringJobSettings:
    """单次 owned GPU 评分事务所需的不可变设置。"""

    gpu_ids: tuple[int, ...]
    dataset_id: str
    verified_models: tuple[str, ...]
    options_by_model: Mapping[str, Mapping[str, Any] | None]
    asset_provenance: Mapping[str, Any] | None
    worker_idle_timeout_seconds: float
    worker_hard_timeout_seconds: float | None
    worker_stderr_tail_bytes: int
    popen: Callable[..., Any]
    progress_callback: Callable[[int], None]
    request_abort: Callable[[BaseException], None] | None = None
    join_aborted: Callable[[BaseException], None] | None = None
    resume_store: Any = None
    resume_capability: Any = None


class _OwnedGpuScoringJob:
    """拥有一次 GPU 评分的临时文件、子进程、调度状态与结果恢复。"""

    def __init__(
        self,
        settings: _GpuScoringJobSettings,
        request_runner: Callable[[int, str, str], None] | None = None,
    ) -> None:
        if type(settings) is not _GpuScoringJobSettings:
            raise TypeError("settings must be _GpuScoringJobSettings")
        if request_runner is not None and not callable(request_runner):
            raise TypeError("request_runner must be callable or None")
        self._gpu_ids = settings.gpu_ids
        self._dataset_id = settings.dataset_id
        self._verified_models = settings.verified_models
        self._asset_provenance = settings.asset_provenance
        for model, field in MOS_MODEL_REQUEST_OPTION_FIELDS.items():
            setattr(self, f"_{field}", settings.options_by_model[model])
        self._worker_idle_timeout_seconds = settings.worker_idle_timeout_seconds
        self._worker_hard_timeout_seconds = settings.worker_hard_timeout_seconds
        self._worker_stderr_tail_bytes = settings.worker_stderr_tail_bytes
        self._popen = settings.popen
        self._request_runner = request_runner
        self._request_abort = settings.request_abort
        self._join_aborted = settings.join_aborted
        self._progress_callback = settings.progress_callback
        self._resume_store = settings.resume_store
        self._resume_capability = settings.resume_capability
        self._processes: set[Any] = set()
        self._processes_lock = threading.Lock()
        self._cancelled = threading.Event()
        self._cancel_lock = threading.Lock()
        self._cancel_error: BaseException | None = None
        self._abort_joined = False

    def _cancel(self, error: BaseException) -> None:
        with self._cancel_lock:
            if self._cancel_error is not None:
                return
            self._cancel_error = error
            self._cancelled.set()
        if self._request_abort is not None:
            try:
                self._request_abort(error)
            except BaseException as cleanup_error:
                _add_cleanup_note(error, "MOS request abort", cleanup_error)

    def _raise_if_cancelled(self) -> None:
        if not self._cancelled.is_set():
            return
        with self._cancel_lock:
            error = self._cancel_error
        if error is not None:
            raise error
        raise RuntimeError("MOS GPU scoring request was cancelled")

    def _join_after_abort(self, error: BaseException) -> None:
        if self._join_aborted is None:
            return
        with self._cancel_lock:
            if self._abort_joined:
                return
            self._abort_joined = True
        try:
            self._join_aborted(error)
        except BaseException as cleanup_error:
            _add_cleanup_note(error, "MOS worker abort join", cleanup_error)

    def _notify_progress(self, completed_rows: int) -> None:
        try:
            self._progress_callback(completed_rows)
        except BaseException:
            pass

    @staticmethod
    def _terminate_process_group(proc: Any) -> None:
        terminate_process_group(proc, killpg=os.killpg)

    def _terminate_active_children(
        self, primary: BaseException | None = None,
    ) -> None:
        with self._processes_lock:
            processes = tuple(self._processes)
        first_error = None
        for index, proc in enumerate(processes):
            try:
                self._terminate_process_group(proc)
            except BaseException as exc:
                first_error = _record_cleanup_error(
                    first_error, exc, f"MOS active worker {index} termination",
                )
        if first_error is None:
            return
        if isinstance(primary, GeneratorExit):
            raise first_error
        if primary is not None:
            _add_cleanup_note(primary, "MOS active worker cleanup", first_error)
            return
        raise first_error

    def _cleanup_worker_process(
        self,
        proc: Any,
        primary: BaseException | None,
    ) -> None:
        cleanup_error = None
        try:
            self._terminate_process_group(proc)
        except BaseException as exc:
            cleanup_error = _record_cleanup_error(
                cleanup_error, exc, "MOS worker termination",
            )

        wait_succeeded = False
        try:
            proc.wait(timeout=0.5)
            wait_succeeded = True
        except BaseException as exc:
            # wait 失败时必须保留 registry；后续协调清理仍需看到这个 child。
            cleanup_error = _record_cleanup_error(
                cleanup_error, exc, "MOS worker wait",
            )

        try:
            _close_process_streams(proc)
        except BaseException as exc:
            cleanup_error = _record_cleanup_error(
                cleanup_error, exc, "MOS worker pipe close",
            )

        if wait_succeeded:
            try:
                returncode = proc.poll()
            except BaseException as exc:
                cleanup_error = _record_cleanup_error(
                    cleanup_error, exc, "MOS worker poll",
                )
            else:
                if returncode is None:
                    cleanup_error = _record_cleanup_error(
                        cleanup_error,
                        RuntimeError("MOS worker remains active after successful wait"),
                        "MOS worker registry cleanup",
                    )
                else:
                    try:
                        with self._processes_lock:
                            self._processes.discard(proc)
                    except BaseException as exc:
                        cleanup_error = _record_cleanup_error(
                            cleanup_error, exc, "MOS worker registry cleanup",
                        )

        if cleanup_error is None:
            return
        if isinstance(primary, GeneratorExit):
            raise cleanup_error
        if primary is not None:
            _add_cleanup_note(primary, "MOS worker cleanup", cleanup_error)
            return
        raise cleanup_error

    def _stderr_tail(self, stderr: bytes | None) -> str:
        return (stderr or b"")[-self._worker_stderr_tail_bytes:].decode("utf-8", "replace").strip()

    def _run_worker_process(self, proc: Any) -> tuple[bytes, bytes]:
        """有界收集 child 输出；仅合法 heartbeat 可刷新 idle deadline。"""
        stdout_stream = getattr(proc, "stdout", None)
        stderr_stream = getattr(proc, "stderr", None)
        try:
            stdout_fd = stdout_stream.fileno()
            stderr_fd = stderr_stream.fileno()
        except (AttributeError, OSError, ValueError):
            fallback_timeout = self._worker_idle_timeout_seconds
            timeout_kind = "idle"
            if (
                self._worker_hard_timeout_seconds is not None
                and self._worker_hard_timeout_seconds < fallback_timeout
            ):
                fallback_timeout = self._worker_hard_timeout_seconds
                timeout_kind = "hard"
            try:
                return run_bounded_subprocess(
                    proc,
                    timeout_seconds=fallback_timeout,
                    stderr_tail_bytes=self._worker_stderr_tail_bytes,
                    killpg=os.killpg,
                )
            except BoundedSubprocessTimeoutError as exc:
                exc.timeout_kind = timeout_kind
                raise

        selector = selectors.DefaultSelector()
        stdout_buffer = bytearray()
        stdout_pending = bytearray()
        stderr_buffer = bytearray()
        streams = {stdout_fd: "stdout", stderr_fd: "stderr"}
        start = time.monotonic()
        idle_deadline = start + self._worker_idle_timeout_seconds
        hard_deadline = (
            None if self._worker_hard_timeout_seconds is None
            else start + self._worker_hard_timeout_seconds
        )
        timeout_kind = None
        try:
            for descriptor, name in streams.items():
                os.set_blocking(descriptor, False)
                selector.register(descriptor, selectors.EVENT_READ, name)
            while streams:
                now = time.monotonic()
                if hard_deadline is not None and now >= hard_deadline:
                    timeout_kind = "hard"
                    break
                if now >= idle_deadline:
                    timeout_kind = "idle"
                    break
                deadline = idle_deadline
                if hard_deadline is not None:
                    deadline = min(deadline, hard_deadline)
                events = selector.select(min(0.1, max(0.0, deadline - now)))
                if not events:
                    continue
                for key, _mask in events:
                    descriptor = key.fd
                    name = streams.get(descriptor)
                    if name is None:
                        continue
                    try:
                        chunk = os.read(descriptor, 64 * 1024)
                    except (BlockingIOError, InterruptedError):
                        continue
                    if not chunk:
                        selector.unregister(descriptor)
                        streams.pop(descriptor)
                        continue
                    if name == "stderr":
                        stderr_buffer.extend(chunk)
                        if len(stderr_buffer) > self._worker_stderr_tail_bytes:
                            del stderr_buffer[:-self._worker_stderr_tail_bytes]
                        continue
                    stdout_pending.extend(chunk)
                    if (
                        len(stdout_buffer) + len(stdout_pending)
                        > _HEARTBEAT_STDOUT_LIMIT_BYTES
                    ):
                        raise BoundedSubprocessOutputLimitError(
                            "stdout",
                            stdout=bytes(stdout_buffer + stdout_pending),
                            stderr=bytes(stderr_buffer),
                        )
                    while b"\n" in stdout_pending:
                        line, _, remainder = stdout_pending.partition(b"\n")
                        stdout_pending = bytearray(remainder)
                        framed = line + b"\n"
                        if framed == _HEARTBEAT_PROTOCOL:
                            idle_deadline = time.monotonic() + self._worker_idle_timeout_seconds
                            continue
                        if framed.startswith(_PROGRESS_PREFIX):
                            value = framed[len(_PROGRESS_PREFIX):-1]
                            try:
                                completed_rows = int(value.decode("ascii"))
                            except (UnicodeDecodeError, ValueError) as exc:
                                raise RuntimeError("MOS worker progress frame is invalid") from exc
                            if completed_rows <= 0:
                                raise RuntimeError("MOS worker progress frame is invalid")
                            self._notify_progress(completed_rows)
                            continue
                        stdout_buffer.extend(framed)
                        if len(stdout_buffer) > _HEARTBEAT_STDOUT_LIMIT_BYTES:
                            raise BoundedSubprocessOutputLimitError(
                                "stdout",
                                stdout=bytes(stdout_buffer),
                                stderr=bytes(stderr_buffer),
                            )
            if timeout_kind is not None:
                error = BoundedSubprocessTimeoutError(
                    f"MOS worker {timeout_kind} timeout",
                    stdout=bytes(stdout_buffer),
                    stderr=bytes(stderr_buffer),
                    returncode=getattr(proc, "returncode", None),
                )
                error.timeout_kind = timeout_kind
                raise error
            stdout_buffer.extend(stdout_pending)
            if len(stdout_buffer) > _HEARTBEAT_STDOUT_LIMIT_BYTES:
                raise BoundedSubprocessOutputLimitError(
                    "stdout",
                    stdout=bytes(stdout_buffer),
                    stderr=bytes(stderr_buffer),
                )
            proc.wait(timeout=0.5)
            return bytes(stdout_buffer), bytes(stderr_buffer)
        except BaseException:
            self._terminate_process_group(proc)
            raise
        finally:
            selector.close()

    def _expected_urgent_pk_assets(self) -> dict[str, str]:
        """返回 worker attestation 必须匹配的 verified URGENT-PK 身份。"""
        if self._asset_provenance is not None:
            assets = self._asset_provenance["models"].get("urgent_pk")
            if assets is None:
                raise ValueError("urgent_pk worker request lacks verified provenance")
            return _validated_urgent_pk_assets(
                assets, "preverified urgent_pk assets",
            )
        options_by_model = {
            model: getattr(self, f"_{field}")
            for model, field in MOS_MODEL_REQUEST_OPTION_FIELDS.items()
        }
        assets = selected_mos_asset_provenance(
            ("urgent_pk",), options_by_model,
            urgent_verifier=verify_urgent_pk_trusted_assets,
        )["models"]["urgent_pk"]
        return _validated_urgent_pk_assets(assets, "verified urgent_pk assets")

    @staticmethod
    def _write_json(path: str, value: Mapping[str, Any]) -> None:
        payload = _encode_json_line(value, "MOS worker request")
        stream = _open_new_regular_stream(path, "MOS worker request")
        try:
            stream.write(payload)
        finally:
            _finalize_streams((stream,), sys.exc_info()[1], sync=True)

    @staticmethod
    def _iter_ordinal_sidecar(
        sidecar: Mapping[str, Any], start: int, stop: int, row_count: int | None,
        *, private_dir: str | None = None,
    ) -> Iterator[int]:
        if (
            not isinstance(sidecar, Mapping)
            or set(sidecar) != _ORDINAL_SIDECAR_FIELDS
            or row_count is not None and (type(row_count) is not int or row_count <= 0)
        ):
            raise ValueError("MOS sparse ordinal sidecar schema is invalid")
        path = sidecar["ordinal_sidecar_path"]
        digest = sidecar["ordinal_sidecar_sha256"]
        size_bytes = sidecar["ordinal_sidecar_size_bytes"]
        if (
            type(path) is not str or not os.path.isabs(path)
            or type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or type(size_bytes) is not int or not 0 < size_bytes <= _JSONL_MAX_FILE_BYTES
        ):
            raise ValueError("MOS sparse ordinal sidecar values are invalid")
        if private_dir is not None and (
            os.path.dirname(os.path.realpath(path)) != os.path.realpath(private_dir)
        ):
            raise ValueError("MOS sparse ordinal sidecar must remain in the private request directory")
        stream, initial = _open_regular_stream(path, "MOS sparse ordinal sidecar")
        if initial.st_size != size_bytes:
            primary = ValueError("MOS sparse ordinal sidecar size mismatch")
            _finalize_streams((stream,), primary, sync=False)
            raise primary
        payloads = None
        observed_digest = hashlib.sha256()
        previous = -1
        count = 0
        try:
            payloads = iter(_shared_iter_bounded_jsonl(
                stream,
                "MOS sparse ordinal sidecar",
                max_line_bytes=_JSONL_MAX_LINE_BYTES,
                max_file_bytes=_JSONL_MAX_FILE_BYTES,
            ))
            for payload in payloads:
                observed_digest.update(payload)
                item = _decode_json_line(payload, "MOS sparse ordinal sidecar row")
                ordinal = item.get("ordinal") if isinstance(item, dict) and set(item) == {"ordinal"} else None
                if (
                    type(ordinal) is not int or ordinal <= previous
                    or ordinal < start or ordinal >= stop
                    or count == 0 and ordinal != start
                ):
                    raise ValueError("MOS sparse ordinal sidecar ordinals are invalid")
                previous = ordinal
                count += 1
                if row_count is not None and count > row_count:
                    raise ValueError("MOS sparse ordinal sidecar row count is invalid")
                yield ordinal
            if row_count is not None and count != row_count:
                raise ValueError("MOS sparse ordinal sidecar row count is invalid")
            if observed_digest.hexdigest() != digest:
                raise ValueError("MOS sparse ordinal sidecar digest mismatch")
            if _file_signature(os.fstat(stream.fileno())) != _file_signature(initial):
                raise ValueError("MOS sparse ordinal sidecar changed during validation")
        finally:
            primary = sys.exc_info()[1]
            closing_generator = isinstance(primary, GeneratorExit)
            cleanup_error = None
            close = getattr(payloads, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:
                    cleanup_error = _record_cleanup_error(
                        cleanup_error, exc, "MOS sparse ordinal payload iterator",
                    )
            try:
                _finalize_streams(
                    (stream,),
                    None if closing_generator else primary or cleanup_error,
                    sync=False,
                )
            except BaseException as exc:
                cleanup_error = _record_cleanup_error(
                    cleanup_error, exc, "MOS sparse ordinal sidecar stream",
                )
            if closing_generator and cleanup_error is not None:
                raise cleanup_error
            if primary is not None:
                if cleanup_error is not None:
                    _add_cleanup_note(primary, "MOS sparse ordinal sidecar cleanup", cleanup_error)
            elif cleanup_error is not None:
                raise cleanup_error

    @staticmethod
    def _read_result(
        path: str,
        input_path: str,
        input_sha256: str,
        start: int,
        stop: int,
        ordinals: Sequence[int] | Mapping[str, Any] | None = None,
        models: Sequence[str] = _MODELS,
    ) -> Iterator[dict[str, Any]]:
        if type(start) is not int or type(stop) is not int or start < 0 or stop <= start:
            raise ValueError("MOS worker request ordinal range is invalid")
        if ordinals is None:
            expected_ordinals: Iterator[int] = iter(range(start, stop))
        elif isinstance(ordinals, Mapping):
            expected_ordinals = _OwnedGpuScoringJob._iter_ordinal_sidecar(
                ordinals, start, stop, None,
            )
        else:
            if (not ordinals or ordinals[0] != start
                    or ordinals[-1] >= stop
                    or any(left >= right for left, right in zip(ordinals, ordinals[1:]))):
                raise ValueError("MOS worker request ordinal list is invalid")
            expected_ordinals = iter(ordinals)
        selected_models = normalize_selectable_mos_models(models)
        source_stream, source_initial = _open_regular_stream(
            input_path, "MOS worker input shard",
        )
        result_stream = None
        results = None
        sources = None
        try:
            result_stream, _result_initial = _open_regular_stream(
                path, "MOS worker result output",
            )
            results = iter(_shared_iter_bounded_jsonl(
                result_stream,
                "MOS worker result output",
                max_line_bytes=_JSONL_MAX_LINE_BYTES,
                max_file_bytes=_JSONL_MAX_FILE_BYTES,
            ))
            input_digest = hashlib.sha256()
            for payload in _shared_iter_bounded_jsonl(
                source_stream,
                "MOS worker input shard",
                max_line_bytes=_JSONL_MAX_LINE_BYTES,
                max_file_bytes=_JSONL_MAX_FILE_BYTES,
            ):
                input_digest.update(payload)
            if input_digest.hexdigest() != input_sha256:
                raise ValueError("MOS worker input shard digest mismatch")
            if _file_signature(os.fstat(source_stream.fileno())) != _file_signature(source_initial):
                raise ValueError("MOS worker input shard changed after digest validation")
            source_stream.seek(0)
            sources = iter(_shared_iter_bounded_jsonl(
                source_stream,
                "MOS worker input shard",
                max_line_bytes=_JSONL_MAX_LINE_BYTES,
                max_file_bytes=_JSONL_MAX_FILE_BYTES,
            ))
            for expected_ordinal in expected_ordinals:
                try:
                    source_payload = next(sources)
                except StopIteration:
                    raise ValueError("MOS worker input shard row count is invalid") from None
                source = _decode_json_line(source_payload, "MOS worker input row")
                if (
                    not isinstance(source, dict)
                    or set(source) != {"ordinal", "row"}
                    or type(source["ordinal"]) is not int
                    or source["ordinal"] != expected_ordinal
                    or not isinstance(source["row"], Mapping)
                ):
                    raise ValueError("MOS worker input shard ordinal or row is invalid")
                try:
                    result_payload = next(results)
                except StopIteration:
                    raise ValueError("MOS worker result row count is invalid") from None
                item = _decode_json_line(result_payload, "MOS worker result line")
                if not isinstance(item, dict) or set(item) != {"ordinal", "scores"}:
                    raise ValueError("MOS worker result schema is invalid")
                if type(item["ordinal"]) is not int or item["ordinal"] != expected_ordinal:
                    raise ValueError("MOS worker result ordinal is invalid")
                scores = Step3MosSelectionStage._validate_scores(
                    item["scores"], selected_models,
                )
                yield {
                    "ordinal": expected_ordinal,
                    "row": source["row"],
                    "scores": scores,
                }
            try:
                next(sources)
            except StopIteration:
                pass
            else:
                raise ValueError("MOS worker input shard row count is invalid")
            try:
                next(results)
            except StopIteration:
                pass
            else:
                raise ValueError(
                    "MOS worker result row count is invalid: extra or duplicate ordinal"
                )
        finally:
            primary = sys.exc_info()[1]
            cleanup_error = None
            for label, iterator in (
                ("MOS expected ordinal iterator", expected_ordinals),
                ("MOS input shard iterator", sources),
                ("MOS result iterator", results),
            ):
                close_iterator = getattr(iterator, "close", None)
                if callable(close_iterator):
                    try:
                        close_iterator()
                    except BaseException as exc:
                        cleanup_error = _record_cleanup_error(
                            cleanup_error, exc, label,
                        )
            closing_generator = isinstance(primary, GeneratorExit)
            streams = tuple(
                stream for stream in (source_stream, result_stream)
                if stream is not None
            )
            try:
                _finalize_streams(
                    streams,
                    None if closing_generator else primary or cleanup_error,
                    sync=False,
                )
            except BaseException as exc:
                cleanup_error = _record_cleanup_error(
                    cleanup_error, exc, "MOS worker input/result streams",
                )
            if closing_generator and cleanup_error is not None:
                raise cleanup_error
            if primary is not None:
                if cleanup_error is not None:
                    _add_cleanup_note(primary, "MOS result cleanup", cleanup_error)
            elif cleanup_error is not None:
                raise cleanup_error

    def run(
        self,
        groups: Sequence[Mapping[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        """按 ``(models, source_identity)`` 全局 LPT 调度 tuple task spools。"""
        if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
            raise TypeError("Step3 MOS task groups must be a sequence")

        descriptors: list[dict[str, Any]] = []
        seen_models: set[tuple[str, ...]] = set()
        seen_paths: set[str] = set()
        previous_models: tuple[str, ...] | None = None
        total_rows = 0
        for group in groups:
            if not isinstance(group, Mapping) or set(group) != {"models", "path", "row_count"}:
                raise ValueError("Step3 MOS task group schema is invalid")
            models = normalize_selectable_mos_models(group["models"])
            path, row_count = group["path"], group["row_count"]
            if not set(models).issubset(self._verified_models):
                raise ValueError(
                    "requested MOS models must be a subset of parent-verified models"
                )
            if (
                type(path) is not str
                or not os.path.isabs(path)
                or type(row_count) is not int
                or row_count <= 0
            ):
                raise ValueError("Step3 MOS task group values are invalid")
            canonical_path = os.path.realpath(path)
            if canonical_path in seen_paths:
                raise ValueError("Step3 MOS task group path is duplicate")
            if models != previous_models:
                if models in seen_models:
                    raise ValueError("Step3 MOS model tuple parts must be contiguous")
                seen_models.add(models)
                previous_models = models
            seen_paths.add(canonical_path)
            descriptors.append({
                "models": models,
                "path": path,
                "row_count": row_count,
                "signature": None,
            })
            total_rows += row_count
        if total_rows == 0:
            return {
                "provider_class": "cuda_subprocess",
                "device_class": "gpu",
                "gpu_ids": [],
                "sharding": "weighted_lpt_ordinals",
                "worker_idle_timeout_seconds": self._worker_idle_timeout_seconds,
                "worker_hard_timeout_seconds": self._worker_hard_timeout_seconds,
                "worker_heartbeat_protocol": _HEARTBEAT_PROTOCOL.decode(
                    "ascii"
                ).strip(),
                "worker_heartbeat_launcher_sha256": hashlib.sha256(
                    _heartbeat_launcher_payload()
                ).hexdigest(),
                "worker_stderr_tail_bytes": self._worker_stderr_tail_bytes,
                "runtime": {"execution_state": "no_tasks"},
            }

        workers_limit = min(len(self._gpu_ids), total_rows)

        def iter_tasks(
            descriptor: dict[str, Any],
            previous_ordinal: int,
        ) -> Iterator[tuple[dict[str, Any], bytes]]:
            models = descriptor["models"]
            path = descriptor["path"]
            row_count = descriptor["row_count"]
            stream, initial = _open_regular_stream(path, "Step3 tuple task spool")
            initial_signature = _file_signature(initial)
            expected_signature = descriptor["signature"]
            payloads = None
            try:
                if _pathname_signature(path, "Step3 tuple task spool") != initial_signature:
                    raise ValueError("Step3 tuple task spool pathname changed before scan")
                if expected_signature is None:
                    descriptor["signature"] = initial_signature
                elif initial_signature != expected_signature:
                    raise ValueError("Step3 tuple task spool changed between LPT scans")
                count = 0
                previous = previous_ordinal
                payloads = iter(_shared_iter_bounded_jsonl(
                    stream,
                    "Step3 tuple task spool",
                    max_line_bytes=_JSONL_MAX_LINE_BYTES,
                    max_file_bytes=TUPLE_SPOOL_MAX_FILE_BYTES,
                ))
                for payload in payloads:
                    task = _decode_json_line(payload, "Step3 tuple task row")
                    if (not isinstance(task, dict)
                            or set(task) != {
                                "ordinal", "row", "models", "scoring_policy_hash",
                            }
                            or task["models"] != list(models)
                            or type(task["ordinal"]) is not int
                            or not isinstance(task["row"], Mapping)):
                        raise ValueError("Step3 tuple task row is invalid")
                    if task["ordinal"] <= previous:
                        raise ValueError(
                            "Step3 tuple part ordinals must be strictly increasing"
                        )
                    previous = task["ordinal"]
                    count += 1
                    if count > row_count:
                        raise ValueError("Step3 tuple task row count exceeds descriptor")
                    yield task, payload
                if count != row_count:
                    raise ValueError("Step3 tuple task row count differs from descriptor")
                if _file_signature(os.fstat(stream.fileno())) != initial_signature:
                    raise ValueError("Step3 tuple task spool changed during scan")
                if _pathname_signature(path, "Step3 tuple task spool") != initial_signature:
                    raise ValueError("Step3 tuple task spool pathname changed during scan")
            finally:
                primary = sys.exc_info()[1]
                closing_generator = isinstance(primary, GeneratorExit)
                cleanup_error = None
                if payloads is not None:
                    try:
                        _close_iterator(payloads)
                    except BaseException as exc:
                        cleanup_error = _record_cleanup_error(
                            cleanup_error, exc, "Step3 tuple task payload iterator",
                        )
                try:
                    _finalize_streams(
                        (stream,),
                        None if closing_generator else primary or cleanup_error,
                        sync=False,
                    )
                except BaseException as exc:
                    cleanup_error = _record_cleanup_error(
                        cleanup_error, exc, "Step3 tuple task spool stream",
                    )
                if closing_generator and cleanup_error is not None:
                    raise cleanup_error
                if primary is not None:
                    if cleanup_error is not None:
                        _add_cleanup_note(
                            primary, "Step3 tuple task spool cleanup", cleanup_error,
                        )
                elif cleanup_error is not None:
                    raise cleanup_error

        with tempfile.TemporaryDirectory(prefix="audio-filter-step3-mos-groups-") as directory:
            os.chmod(directory, stat.S_IRWXU)
            planner = _DiskBackedLptPlanner(
                os.path.join(directory, "lpt.sqlite3"), total_rows, workers_limit,
            )
            try:
                current_models: tuple[str, ...] | None = None
                previous_ordinal = -1
                for descriptor in descriptors:
                    models = descriptor["models"]
                    if models != current_models:
                        current_models = models
                        previous_ordinal = -1
                    tasks = iter(iter_tasks(descriptor, previous_ordinal))
                    try:
                        for task, payload in tasks:
                            ordinal, row = task["ordinal"], task["row"]
                            previous_ordinal = ordinal
                            Step3MosSelectionStage._validate_upstream_row(
                                row, self._dataset_id,
                            )
                            identity = _DiskBackedLptPlanner.identity_bytes(
                                models, Step3MosRuntime.source_identity(row),
                            )
                            weight = row["end_sample"] - row["start_sample"]
                            if type(weight) is not int or not 0 < weight <= _MAX_UINT64:
                                raise ValueError(
                                    "Step2 upstream row interval weight must be a positive uint64"
                                )
                            planner.add(
                                ordinal=ordinal,
                                identity=identity,
                                weight=weight,
                                payload=payload,
                            )
                    finally:
                        _finish_iterator(
                            tasks, "Step3 first LPT scan iterator", sys.exc_info()[1],
                        )
                workers = planner.finalize()
                expected_urgent_pk_assets = (
                    self._expected_urgent_pk_assets()
                    if any(
                        "urgent_pk" in descriptor["models"]
                        for descriptor in descriptors
                    )
                    else None
                )
                heartbeat_launcher = _write_heartbeat_launcher(directory)
                subshards_by_worker: list[list[dict[str, Any]]] = [
                    [] for _ in range(workers)
                ]
                records_by_worker: list[list[tuple[Any, ...]]] = [
                    [] for _ in range(workers)
                ]
                worker_models_by_worker: list[list[str]] = [
                    [] for _ in range(workers)
                ]
                active: dict[int, dict[str, Any]] = {}
                next_part = 0

                def open_part(
                    worker: int,
                    models: tuple[str, ...],
                    scoring_policy_hash: str,
                ) -> dict[str, Any]:
                    nonlocal next_part
                    part = next_part
                    next_part += 1
                    input_path = os.path.join(
                        directory, f"input-{worker}-{part}.jsonl",
                    )
                    sidecar_path = os.path.join(
                        directory, f"ordinals-{worker}-{part}.jsonl",
                    )
                    input_stream = _open_new_regular_stream(
                        input_path, "MOS worker input shard",
                    )
                    try:
                        sidecar_stream = _open_new_regular_stream(
                            sidecar_path, "MOS sparse ordinal sidecar",
                        )
                    except BaseException:
                        _finalize_streams(
                            (input_stream,), sys.exc_info()[1], sync=False,
                        )
                        raise
                    return {
                        "worker": worker,
                        "part": part,
                        "models": models,
                        "scoring_policy_hash": scoring_policy_hash,
                        "binding_bytes": len(_canonical_json_bytes({
                            "models": list(models),
                            "scoring_policy_hash": scoring_policy_hash,
                        })),
                        "descriptor_bytes": _descriptor_skeleton_bytes(models),
                        "task_identities": [],
                        "task_ordinals": [],
                        "input_path": input_path,
                        "input_stream": input_stream,
                        "input_size": 0,
                        "input_digest": hashlib.sha256(),
                        "sidecar_path": sidecar_path,
                        "sidecar_stream": sidecar_stream,
                        "sidecar_size": 0,
                        "sidecar_digest": hashlib.sha256(),
                        "rows": 0,
                        "first": None,
                        "last": None,
                        "contiguous": True,
                    }

                def finalize_part(state: dict[str, Any]) -> None:
                    input_stream = state.pop("input_stream")
                    sidecar_stream = state.pop("sidecar_stream")
                    _finalize_streams(
                        (input_stream, sidecar_stream), None, sync=True,
                    )
                    worker = state["worker"]
                    models = state["models"]
                    start = state["first"]
                    stop = state["last"] + 1
                    output_path = os.path.join(
                        directory,
                        f"result-{worker}-{state['part']}-{start}-{secrets.token_hex(8)}.jsonl",
                    )
                    subshard: dict[str, Any] = {
                        "models": list(models),
                        "input_path": state["input_path"],
                        "output_path": output_path,
                        "row_count": state["rows"],
                        "input_sha256": state["input_digest"].hexdigest(),
                        "start_ordinal": start,
                        "end_ordinal": stop,
                    }
                    request_ordinals: Mapping[str, Any] | None = None
                    if state["contiguous"] and state["rows"] == stop - start:
                        os.unlink(state["sidecar_path"])
                    else:
                        request_ordinals = {
                            "ordinal_sidecar_path": state["sidecar_path"],
                            "ordinal_sidecar_sha256": state["sidecar_digest"].hexdigest(),
                            "ordinal_sidecar_size_bytes": state["sidecar_size"],
                        }
                        subshard.update(request_ordinals)
                    subshards_by_worker[worker].append(subshard)
                    records_by_worker[worker].append({
                        "input_path": state["input_path"],
                        "output_path": output_path,
                        "input_sha256": subshard["input_sha256"],
                        "start": start,
                        "stop": stop,
                        "ordinals": request_ordinals,
                        "models": models,
                        "scoring_policy_hash": state["scoring_policy_hash"],
                        "task_identities": tuple(state["task_identities"]),
                        "task_ordinals": tuple(state["task_ordinals"]),
                        "descriptor_bytes": state["descriptor_bytes"],
                    })
                    for model in models:
                        if model not in worker_models_by_worker[worker]:
                            worker_models_by_worker[worker].append(model)

                def close_active(primary: BaseException | None) -> None:
                    streams = []
                    for state in active.values():
                        streams.extend((
                            state.pop("input_stream", None),
                            state.pop("sidecar_stream", None),
                        ))
                    active.clear()
                    _finalize_streams(
                        tuple(stream for stream in streams if stream is not None),
                        primary,
                        sync=primary is None,
                    )

                current_models: tuple[str, ...] | None = None
                lookup_batch: list[tuple[dict[str, Any], bytes, bytes]] = []
                lookup_bytes = 0

                def flush_lookup_batch() -> None:
                    nonlocal lookup_bytes
                    if not lookup_batch:
                        return
                    assignments = planner.assignments(
                        (task["ordinal"], identity, payload)
                        for task, payload, identity in lookup_batch
                    )
                    for (task, _task_payload, _identity), worker in zip(
                        lookup_batch, assignments, strict=True,
                    ):
                        models = tuple(task["models"])
                        ordinal = task["ordinal"]
                        input_payload = _encode_json_line(
                            {"ordinal": ordinal, "row": dict(task["row"])},
                            "MOS worker input row",
                        )
                        sidecar_payload = _encode_json_line(
                            {"ordinal": ordinal}, "MOS sparse ordinal sidecar row",
                        )
                        if (
                            len(input_payload) > _JSONL_MAX_FILE_BYTES
                            or len(sidecar_payload) > _JSONL_MAX_FILE_BYTES
                        ):
                            raise ValueError("MOS worker row exceeds physical shard limit")
                        scoring_policy_hash = task["scoring_policy_hash"]
                        state = active.get(worker)
                        if state is None:
                            state = open_part(worker, models, scoring_policy_hash)
                            active[worker] = state
                        if (
                            state["models"] != models
                            or state["scoring_policy_hash"] != scoring_policy_hash
                        ):
                            finalize_part(state)
                            state = open_part(worker, models, scoring_policy_hash)
                            active[worker] = state
                        descriptor_increment = (
                            len(str(ordinal)) + state["binding_bytes"] + 69
                        )
                        descriptor_over_budget = (
                            state["descriptor_bytes"] + descriptor_increment
                            > _REQUEST_DESCRIPTOR_BUDGET_BYTES
                        )
                        if state["rows"] and (
                            state["input_size"] + len(input_payload) > _JSONL_MAX_FILE_BYTES
                            or state["sidecar_size"] + len(sidecar_payload) > _JSONL_MAX_FILE_BYTES
                            or descriptor_over_budget
                        ):
                            finalize_part(state)
                            state = open_part(worker, models, scoring_policy_hash)
                            active[worker] = state
                            descriptor_increment = (
                                len(str(ordinal)) + state["binding_bytes"] + 69
                            )
                            descriptor_over_budget = (
                                state["descriptor_bytes"] + descriptor_increment
                                > _REQUEST_DESCRIPTOR_BUDGET_BYTES
                            )
                        if descriptor_over_budget:
                            raise ValueError(
                                "MOS worker task descriptor exceeds request byte budget"
                            )
                        if state["last"] is not None and ordinal != state["last"] + 1:
                            state["contiguous"] = False
                        state["input_stream"].write(input_payload)
                        state["input_digest"].update(input_payload)
                        state["input_size"] += len(input_payload)
                        state["sidecar_stream"].write(sidecar_payload)
                        state["sidecar_digest"].update(sidecar_payload)
                        state["sidecar_size"] += len(sidecar_payload)
                        state["task_identities"].append(step3_task_identity(
                            ordinal, task["row"], models, scoring_policy_hash,
                        ))
                        state["task_ordinals"].append(ordinal)
                        state["rows"] += 1
                        state["descriptor_bytes"] += descriptor_increment
                        state["first"] = ordinal if state["first"] is None else state["first"]
                        state["last"] = ordinal
                    lookup_batch.clear()
                    lookup_bytes = 0

                try:
                    previous_ordinal = -1
                    for descriptor in descriptors:
                        models = descriptor["models"]
                        if current_models is not None and models != current_models:
                            flush_lookup_batch()
                            for state in tuple(active.values()):
                                finalize_part(state)
                            active.clear()
                            previous_ordinal = -1
                        current_models = models
                        tasks = iter(iter_tasks(descriptor, previous_ordinal))
                        try:
                            for task, payload in tasks:
                                previous_ordinal = task["ordinal"]
                                identity = _DiskBackedLptPlanner.identity_bytes(
                                    models, Step3MosRuntime.source_identity(task["row"]),
                                )
                                item_bytes = len(payload) + len(identity) + 64
                                if lookup_batch and (
                                    len(lookup_batch) >= _LPT_BATCH_MAX_ROWS
                                    or lookup_bytes + item_bytes > _LPT_BATCH_MAX_BYTES
                                ):
                                    flush_lookup_batch()
                                lookup_batch.append((task, payload, identity))
                                lookup_bytes += item_bytes
                        finally:
                            _finish_iterator(
                                tasks, "Step3 second LPT scan iterator",
                                sys.exc_info()[1],
                            )
                    flush_lookup_batch()
                    for state in tuple(active.values()):
                        finalize_part(state)
                    active.clear()
                except BaseException:
                    close_active(sys.exc_info()[1])
                    raise
            finally:
                primary = sys.exc_info()[1]
                try:
                    planner.close()
                except BaseException as cleanup_error:
                    if isinstance(primary, GeneratorExit):
                        raise cleanup_error
                    if primary is not None:
                        _add_cleanup_note(primary, "MOS LPT planner cleanup", cleanup_error)
                    else:
                        raise

            if self._resume_store is not None and self._resume_capability is None:
                raise RuntimeError("MOS resume store requires a persistent write capability")

            all_options = {
                model: getattr(self, f"_{field}")
                for model, field in MOS_MODEL_REQUEST_OPTION_FIELDS.items()
            }
            option_fields = MOS_MODEL_REQUEST_OPTION_FIELDS
            requests: list[dict[str, Any]] = []
            for worker in range(workers):
                subshards = subshards_by_worker[worker]
                records = records_by_worker[worker]
                if len(subshards) != len(records):
                    raise RuntimeError("MOS worker physical shard plan is detached")
                chunks: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
                chunk_subshards: list[dict[str, Any]] = []
                chunk_records: list[dict[str, Any]] = []
                # 这里只累计每个 part 的 descriptor payload；固定 skeleton
                # 根据当前 request 的模型并集每轮重新计算一次。
                chunk_descriptor_bytes = 0
                for subshard, record in zip(subshards, records, strict=True):
                    record_skeleton_bytes = _descriptor_skeleton_bytes(
                        tuple(record["models"])
                    )
                    if record["descriptor_bytes"] < record_skeleton_bytes:
                        raise RuntimeError("MOS physical shard descriptor accounting is invalid")
                    record_payload_bytes = (
                        record["descriptor_bytes"] - record_skeleton_bytes
                    )
                    candidate_models = tuple(dict.fromkeys(
                        model
                        for item in (*chunk_records, record)
                        for model in item["models"]
                    ))
                    candidate_bytes = (
                        _descriptor_skeleton_bytes(candidate_models)
                        + chunk_descriptor_bytes
                        + record_payload_bytes
                    )
                    if chunk_records and (
                        len(chunk_records) >= MAX_WORKER_INPUT_SUBSHARDS
                        or candidate_bytes > _REQUEST_DESCRIPTOR_BUDGET_BYTES
                    ):
                        chunks.append((chunk_subshards, chunk_records))
                        chunk_subshards = []
                        chunk_records = []
                        chunk_descriptor_bytes = 0
                        candidate_bytes = (
                            _descriptor_skeleton_bytes(tuple(record["models"]))
                            + record_payload_bytes
                        )
                    chunk_subshards.append(subshard)
                    chunk_records.append(record)
                    chunk_descriptor_bytes += record_payload_bytes
                if chunk_records:
                    chunks.append((chunk_subshards, chunk_records))
                for part, (request_subshards, request_records) in enumerate(chunks):
                    request_path = os.path.join(directory, f"request-{worker}-{part}.json")
                    attestation_path = os.path.join(
                        directory, f"attestation-{worker}-{part}.json",
                    )
                    worker_models = tuple(dict.fromkeys(
                        model
                        for record in request_records
                        for model in record["models"]
                    ))
                    request = {
                        "protocol": _PROTOCOL,
                        "asset_load_binding": _plain_json(ASSET_LOAD_BINDING),
                        "scoring_execution_policy": _plain_json(
                            MOS_SCORING_EXECUTION_POLICY,
                        ),
                        "private_dir": directory,
                        "attestation_path": attestation_path,
                        "subshards": request_subshards,
                    }
                    request.update({
                        field: (
                            _plain_json(all_options[model])
                            if model in worker_models else None
                        )
                        for model, field in option_fields.items()
                    })
                    self._write_json(request_path, request)
                    requests.append({
                        "path": request_path,
                        "attestation_path": attestation_path,
                        "worker_models": worker_models,
                        "expected_checkpoint_sha256": {
                            model: request[option_fields[model]]["checkpoint_sha256"]
                            for model in worker_models
                            if model in _CHECKPOINT_ATTESTATION_MODELS
                        },
                        "expected_urgent_pk_assets": (
                            dict(expected_urgent_pk_assets)
                            if "urgent_pk" in worker_models else None
                        ),
                        "gpu_id": self._gpu_ids[worker],
                        "records": request_records,
                        "worker": worker,
                        "part": part,
                    })

            def run_worker(index: int) -> None:
                self._raise_if_cancelled()
                if self._request_runner is not None:
                    self._request_runner(
                        requests[index]["gpu_id"], requests[index]["path"], directory,
                    )
                    return
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(requests[index]["gpu_id"])
                environment["AUDIO_FILTER_STEP3_HEARTBEAT_SECONDS"] = str(
                    max(0.01, min(30.0, self._worker_idle_timeout_seconds / 3.0))
                )
                proc = None
                try:
                    proc = self._popen(
                        [sys.executable, heartbeat_launcher, requests[index]["path"]],
                        shell=False, env=environment, cwd=directory,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                    )
                    with self._processes_lock:
                        self._processes.add(proc)
                    self._raise_if_cancelled()
                    _stdout, stderr = self._run_worker_process(proc)
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"MOS worker {index} failed with return code {proc.returncode}; "
                            f"stderr_tail={self._stderr_tail(stderr)!r}"
                        )
                except BoundedSubprocessTimeoutError as exc:
                    timeout_kind = getattr(exc, "timeout_kind", "idle")
                    timeout_seconds = (
                        self._worker_hard_timeout_seconds
                        if timeout_kind == "hard"
                        else self._worker_idle_timeout_seconds
                    )
                    raise RuntimeError(
                        f"MOS worker {index} timed out ({timeout_kind} timeout) after "
                        f"{timeout_seconds:g}s; "
                        f"stderr_tail={self._stderr_tail(exc.stderr)!r}"
                    ) from exc
                except BoundedSubprocessOutputLimitError as exc:
                    raise RuntimeError(
                        f"MOS worker {index} exceeded bounded {exc.stream} output limit; "
                        f"stderr_tail={self._stderr_tail(exc.stderr)!r}"
                    ) from exc
                except BoundedSubprocessError as exc:
                    raise RuntimeError(
                        f"MOS worker {index} failed during bounded I/O; "
                        f"stderr_tail={self._stderr_tail(exc.stderr)!r}"
                    ) from exc
                finally:
                    if proc is not None:
                        self._cleanup_worker_process(proc, sys.exc_info()[1])

            result_store = _DiskBackedResultStore(
                os.path.join(directory, "results.sqlite3"), total_rows,
            )
            result_payloads = None
            attempt = None
            worker_attestations: list[Mapping[str, Any]] = []
            try:
                descriptors_by_request: dict[int, dict[str, Any]] = {}
                for index, request in enumerate(requests):
                    request_records = request["records"]
                    task_bindings = sorted(
                        (
                            ordinal,
                            identity,
                            list(record["models"]),
                            record["scoring_policy_hash"],
                        )
                        for record in request_records
                        for ordinal, identity in zip(
                            record["task_ordinals"], record["task_identities"], strict=True,
                        )
                    )
                    bindings = [
                        {"models": item[2], "scoring_policy_hash": item[3]}
                        for item in task_bindings
                    ]
                    scoring_policy_hash = hashlib.sha256(_encode_json_line(
                        {"bindings": bindings}, "MOS request scoring bindings",
                    )).hexdigest()
                    descriptor_content = {
                        "ordinals": [item[0] for item in task_bindings],
                        "task_identities": [item[1] for item in task_bindings],
                        "models": list(request["worker_models"]),
                        "scoring_policy_hash": scoring_policy_hash,
                        "task_bindings": bindings,
                    }
                    shard_id = "r-" + hashlib.sha256(_encode_json_line(
                        descriptor_content, "MOS stable request descriptor",
                    )).hexdigest()[:62]
                    descriptors_by_request[index] = {
                        "shard_id": shard_id,
                        **descriptor_content,
                    }
                pending = set(range(len(requests)))
                if self._resume_store is not None:
                    sealed_by_descriptor = {
                        _encode_json_line(shard.descriptor, "resume sealed descriptor"): shard
                        for shard in self._resume_store.sealed_shards()
                    }
                    cached_rows: dict[str, list[Mapping[str, Any]]] = {}
                    for shard, cached in self._resume_store.iter_sealed_rows():
                        key = _encode_json_line(
                            shard.descriptor, "resume sealed descriptor",
                        ).decode("utf-8")
                        cached_rows.setdefault(key, []).append(cached)
                    for index, descriptor in descriptors_by_request.items():
                        descriptor_key = _encode_json_line(
                            descriptor, "resume sealed descriptor",
                        )
                        shard = sealed_by_descriptor.pop(descriptor_key, None)
                        if shard is None:
                            continue
                        if shard.runtime_attestation is None:
                            raise RuntimeError("resume sealed shard lacks runtime attestation")
                        for cached in cached_rows.pop(descriptor_key.decode("utf-8"), ()):
                            if (
                                not isinstance(cached, Mapping)
                                or set(cached) != {"ordinal", "task_identity", "payload"}
                            ):
                                raise RuntimeError("resume cached row is invalid")
                            result_store.add(
                                cached["ordinal"],
                                _encode_json_line(
                                    cached["payload"], "MOS cached result row",
                                ),
                            )
                        worker_attestations.append(_plain_json(shard.runtime_attestation))
                        pending.remove(index)
                    if cached_rows:
                        raise RuntimeError("resume cached rows do not match sealed descriptors")
                    if sealed_by_descriptor:
                        raise RuntimeError("resume sealed shard does not match current request plan")
                    if pending:
                        attempt = self._resume_store.begin_attempt(self._resume_capability)

                def consume_request(index: int) -> None:
                    request = requests[index]
                    runtime = {
                        "gpu_id": request["gpu_id"],
                        **_read_runtime_attestation(
                            request["attestation_path"],
                            request["worker_models"],
                            request["expected_checkpoint_sha256"],
                            request["expected_urgent_pk_assets"],
                        ),
                    }
                    rows = []
                    for record in request["records"]:
                        results = iter(self._read_result(
                            record["output_path"], record["input_path"],
                            record["input_sha256"], record["start"], record["stop"],
                            record["ordinals"], record["models"],
                        ))
                        try:
                            for expected_ordinal, expected_identity, item in zip(
                                record["task_ordinals"], record["task_identities"], results,
                                strict=True,
                            ):
                                if item["ordinal"] != expected_ordinal:
                                    raise ValueError("MOS request result ordinal is invalid")
                                stored = {**item, "models": list(record["models"])}
                                rows.append({
                                    "ordinal": expected_ordinal,
                                    "task_identity": expected_identity,
                                    "payload": stored,
                                })
                        finally:
                            _finish_iterator(
                                results, "MOS physical result iterator", sys.exc_info()[1],
                            )
                    if len(rows) != len(descriptors_by_request[index]["ordinals"]):
                        raise ValueError("MOS request result count is invalid")
                    rows.sort(key=lambda row: row["ordinal"])
                    if [row["ordinal"] for row in rows] != descriptors_by_request[index]["ordinals"]:
                        raise ValueError("MOS request result order is invalid")
                    if [row["task_identity"] for row in rows] != descriptors_by_request[index]["task_identities"]:
                        raise ValueError("MOS request result task identity is invalid")
                    if attempt is not None:
                        attempt.seal_shard(
                            descriptors_by_request[index], rows, runtime,
                        )
                    for row in rows:
                        result_store.add(
                            row["ordinal"],
                            _encode_json_line(row["payload"], "MOS validated result row"),
                        )
                    worker_attestations.append(runtime)
                    self._progress_callback(len(rows))

                executor = ThreadPoolExecutor(max_workers=workers)
                futures: dict[Any, tuple[int, int]] = {}
                queues = {
                    worker: iter(tuple(
                        index for index in sorted(pending)
                        if requests[index]["worker"] == worker
                    ))
                    for worker in range(workers)
                }

                def submit_next(worker: int) -> None:
                    index = next(queues[worker], None)
                    if index is not None:
                        futures[executor.submit(run_worker, index)] = (worker, index)

                try:
                    for worker in range(workers):
                        submit_next(worker)
                    while futures:
                        future = next(as_completed(tuple(futures)))
                        worker, request_index = futures.pop(future)
                        future.result()
                        consume_request(request_index)
                        submit_next(worker)
                except BaseException as exc:
                    self._cancel(exc)
                    for future in futures:
                        future.cancel()
                    self._terminate_active_children(exc)
                    try:
                        executor.shutdown(wait=True, cancel_futures=True)
                    except BaseException as cleanup_error:
                        _add_cleanup_note(exc, "MOS worker executor shutdown", cleanup_error)
                    self._join_after_abort(exc)
                    raise
                else:
                    executor.shutdown(wait=True)
                finally:
                    if attempt is not None:
                        attempt.abort()

                runtime_attestation = _freeze_json({
                    "protocol": _PROTOCOL,
                    "asset_load_binding": _plain_json(ASSET_LOAD_BINDING),
                    "scoring_execution_policy": _plain_json(
                        MOS_SCORING_EXECUTION_POLICY,
                    ),
                    "workers": sorted(
                        worker_attestations, key=lambda item: item["gpu_id"],
                    ),
                }, "MOS pool runtime attestation")
                try:
                    result_store.finalize()
                    result_payloads = iter(result_store.iter_payloads())
                    for expected, (ordinal, payload) in enumerate(result_payloads):
                        if ordinal != expected:
                            raise ValueError(
                                "MOS worker result global ordinal ordering is invalid"
                            )
                        item = _decode_json_line(payload, "MOS stored result row")
                        if (
                            not isinstance(item, dict)
                            or set(item) != {"ordinal", "row", "scores", "models"}
                            or item["ordinal"] != ordinal
                        ):
                            raise ValueError("MOS stored result row is invalid")
                        yield item
                    execution_attestation = _freeze_json({
                        "provider_class": "cuda_subprocess",
                        "device_class": "gpu",
                        "gpu_ids": list(self._gpu_ids[:workers]),
                        "sharding": "weighted_lpt_ordinals",
                        "worker_idle_timeout_seconds": self._worker_idle_timeout_seconds,
                        "worker_hard_timeout_seconds": self._worker_hard_timeout_seconds,
                        "worker_heartbeat_protocol": _HEARTBEAT_PROTOCOL.decode(
                            "ascii"
                        ).strip(),
                        "worker_heartbeat_launcher_sha256": hashlib.sha256(
                            _heartbeat_launcher_payload()
                        ).hexdigest(),
                        "worker_stderr_tail_bytes": self._worker_stderr_tail_bytes,
                        "runtime": _plain_json(runtime_attestation),
                    }, "MOS pool execution attestation")
                    return _plain_json(execution_attestation)
                finally:
                    primary = sys.exc_info()[1]
                    closing_generator = isinstance(primary, GeneratorExit)
                    cleanup_error = None
                    if result_payloads is not None:
                        try:
                            _close_iterator(result_payloads)
                        except BaseException as exc:
                            cleanup_error = _record_cleanup_error(
                                cleanup_error, exc, "MOS result SQLite iterator",
                            )
                    try:
                        result_store.close()
                    except BaseException as exc:
                        cleanup_error = _record_cleanup_error(
                            cleanup_error, exc, "MOS result SQLite store",
                        )
                    if closing_generator and cleanup_error is not None:
                        raise cleanup_error
                    if primary is not None:
                        if cleanup_error is not None:
                            _add_cleanup_note(
                                primary, "MOS result store cleanup", cleanup_error,
                            )
                    elif cleanup_error is not None:
                        raise cleanup_error
            finally:
                primary = sys.exc_info()[1]
                if primary is not None:
                    self._cancel(primary)
                cleanup_error = None
                try:
                    self._terminate_active_children(primary)
                except BaseException as exc:
                    cleanup_error = exc
                if primary is not None:
                    self._join_after_abort(primary)
                if cleanup_error is not None:
                    if isinstance(primary, GeneratorExit):
                        raise cleanup_error
                    if primary is not None:
                        _add_cleanup_note(
                            primary, "MOS active worker final cleanup", cleanup_error,
                        )
                    else:
                        raise cleanup_error
