"""Dependency-isolated parent adapter for local WeSpeaker embeddings."""

from __future__ import annotations

import hashlib
import math
import os
import re
import selectors
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any

from ..optional_runtime import (
    AssetDigestMismatchError,
    AssetScanLimits,
    LocalAssetError,
    LocalModelAsset,
    OptionalWorkerError,
    OptionalWorkerSession,
    OptionalWorkerSpec,
    WorkerLimits,
    WorkerProtocolError,
    WorkerRemoteAssetError,
    WorkerRemoteError,
    WorkerRuntimeVersionMismatchError,
    WorkerUnavailableError,
    run_optional_worker,
)

__all__ = [
    "WeSpeakerRuntimeError", "WeSpeakerRuntimeUnavailableError",
    "WeSpeakerRuntimeAssetError", "WeSpeakerRuntimeVersionError",
    "WeSpeakerRuntimeRemoteError", "WeSpeakerRuntimeOptions", "WeSpeakerEmbeddingRuntime",
    "WeSpeakerEmbeddingSession", "BoundedSubprocessError", "BoundedSubprocessTimeoutError",
    "BoundedSubprocessOutputLimitError", "run_bounded_subprocess",
    "runtime_options_payload", "terminate_process_group", "communicate_after_timeout",
]

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_WORKER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workers", "wespeaker_worker.py"))
_TRUSTED_ROOT = os.path.dirname(os.path.dirname(_WORKER_PATH))
_DEFAULT_LIMITS = WorkerLimits(30_000, 500, 500, 60, 256, 128 * 1024 * 1024, 32 * 1024**3)
_DEFAULT_SUBPROCESS_STDOUT_LIMIT_BYTES = 4 * 1024 * 1024
_DEFAULT_SUBPROCESS_STDERR_TAIL_BYTES = 256 * 1024
_SUBPROCESS_IO_CHUNK_BYTES = 64 * 1024
_SUBPROCESS_CLEANUP_TIMEOUT_SECONDS = 0.05


class WeSpeakerRuntimeError(RuntimeError):
    pass


class WeSpeakerRuntimeUnavailableError(WeSpeakerRuntimeError):
    pass


class WeSpeakerRuntimeAssetError(WeSpeakerRuntimeError):
    pass


class WeSpeakerRuntimeVersionError(WeSpeakerRuntimeError):
    pass


class WeSpeakerRuntimeRemoteError(WeSpeakerRuntimeError):
    pass


@dataclass(frozen=True)
class WeSpeakerRuntimeOptions:
    """可由未来配置注入的本地 bundle、worker 与 runtime 绑定。"""

    model_path: str
    model_version: str
    model_sha256: str
    runtime_version: str
    target_sample_rate: int = 16_000
    python_executable: str = field(default_factory=lambda: os.path.realpath(sys.executable))
    trusted_code_root: str = _TRUSTED_ROOT
    worker_script: str = _WORKER_PATH
    worker_script_sha256: str = ""
    worker_limits: WorkerLimits = field(default=_DEFAULT_LIMITS)
    asset_limits: AssetScanLimits = field(default_factory=AssetScanLimits)
    cuda_visible_devices: str | None = None

    def __post_init__(self) -> None:
        for name in ("model_path", "python_executable", "trusted_code_root", "worker_script"):
            value = getattr(self, name)
            if type(value) is not str or not value or not os.path.isabs(value) or value != os.path.normpath(os.path.abspath(value)):
                raise ValueError(f"{name} must be a canonical absolute path")
        for name in ("model_version", "runtime_version"):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be a nonblank exact string")
        for name in ("model_sha256", "worker_script_sha256"):
            value = getattr(self, name)
            if type(value) is not str or _HEX.fullmatch(value) is None:
                raise ValueError(f"{name} must be lowercase SHA-256 hex")
        if type(self.target_sample_rate) is not int or self.target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be a positive Python integer")
        if not isinstance(self.worker_limits, WorkerLimits) or not isinstance(self.asset_limits, AssetScanLimits):
            raise TypeError("worker and asset limits must use optional-runtime contracts")
        if self.cuda_visible_devices is not None:
            value = self.cuda_visible_devices
            if type(value) is not str or not value or value != value.strip() or not value.isdecimal():
                raise ValueError("cuda_visible_devices must be one physical GPU id")

    def to_payload(self) -> dict[str, object]:
        """将运行时选项转换成可直接 JSON 编码的普通字典。"""
        return {
            "model_path": self.model_path,
            "model_version": self.model_version,
            "model_sha256": self.model_sha256,
            "runtime_version": self.runtime_version,
            "target_sample_rate": self.target_sample_rate,
            "python_executable": self.python_executable,
            "trusted_code_root": self.trusted_code_root,
            "worker_script": self.worker_script,
            "worker_script_sha256": self.worker_script_sha256,
            "worker_limits": {
                name: getattr(self.worker_limits, name)
                for name in (
                    "timeout_ms", "terminate_grace_ms", "kill_wait_ms",
                    "rlimit_cpu_seconds", "rlimit_nofile", "rlimit_fsize_bytes",
                    "rlimit_as_bytes", "max_request_bytes", "max_stdout_bytes",
                    "max_stderr_bytes", "max_worker_script_bytes", "max_json_depth",
                    "io_chunk_bytes",
                )
            },
            "asset_limits": {
                name: getattr(self.asset_limits, name)
                for name in (
                    "max_entries", "max_total_bytes", "max_path_bytes",
                    "read_chunk_bytes", "max_depth",
                )
            },
            "cuda_visible_devices": self.cuda_visible_devices,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "WeSpeakerRuntimeOptions":
        """从 JSON 解码后的字典恢复选项，并重建嵌套 limits 对象。"""
        if not isinstance(payload, Mapping):
            raise TypeError("WeSpeakerRuntimeOptions payload must be a mapping")
        values = dict(payload)
        worker_limits = values.get("worker_limits", _DEFAULT_LIMITS)
        asset_limits = values.get("asset_limits", AssetScanLimits())
        if isinstance(worker_limits, Mapping):
            worker_limits = WorkerLimits(**dict(worker_limits))
        elif not isinstance(worker_limits, WorkerLimits):
            raise TypeError("worker_limits must be a mapping or WorkerLimits")
        if isinstance(asset_limits, Mapping):
            asset_limits = AssetScanLimits(**dict(asset_limits))
        elif not isinstance(asset_limits, AssetScanLimits):
            raise TypeError("asset_limits must be a mapping or AssetScanLimits")
        values["worker_limits"] = worker_limits
        values["asset_limits"] = asset_limits
        return cls(**values)


def runtime_options_payload(options: Mapping[str, Any] | WeSpeakerRuntimeOptions) -> dict[str, object]:
    """在保留测试/调用方精简 mapping 的同时规范化嵌套 limits。"""
    if isinstance(options, WeSpeakerRuntimeOptions):
        return options.to_payload()
    if not isinstance(options, Mapping):
        raise TypeError("WeSpeaker runtime options must be a mapping")
    payload = dict(options)
    for name, limit_type, fields in (
        ("worker_limits", WorkerLimits, (
            "timeout_ms", "terminate_grace_ms", "kill_wait_ms", "rlimit_cpu_seconds",
            "rlimit_nofile", "rlimit_fsize_bytes", "rlimit_as_bytes", "max_request_bytes",
            "max_stdout_bytes", "max_stderr_bytes", "max_worker_script_bytes", "max_json_depth",
            "io_chunk_bytes",
        )),
        ("asset_limits", AssetScanLimits, (
            "max_entries", "max_total_bytes", "max_path_bytes", "read_chunk_bytes", "max_depth",
        )),
    ):
        value = payload.get(name)
        if isinstance(value, limit_type):
            payload[name] = {field: getattr(value, field) for field in fields}
        elif value is not None and not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping or {limit_type.__name__}")
    return payload


def terminate_process_group(
    proc: Any,
    *,
    killpg: Any = os.killpg,
    terminate_wait_seconds: float = 0.5,
    kill_wait_seconds: float = 0.5,
    force: bool = False,
) -> None:
    """以有界 TERM/KILL 顺序终止并回收进程组，未退出时明确失败。"""
    pid = getattr(proc, "pid", None)
    if not force and proc.poll() is not None:
        return
    if type(pid) is not int or pid <= 0:
        try:
            proc.wait(timeout=kill_wait_seconds)
        except subprocess.TimeoutExpired:
            raise
        except (ChildProcessError, ProcessLookupError):
            return
        if proc.poll() is None:
            raise subprocess.TimeoutExpired("process cleanup", kill_wait_seconds)
        return
    for sig, wait_seconds in (
        (signal.SIGTERM, terminate_wait_seconds),
        (signal.SIGKILL, kill_wait_seconds),
    ):
        try:
            killpg(pid, sig)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            continue
        except (ChildProcessError, ProcessLookupError):
            return
        if proc.poll() is not None:
            return
    try:
        proc.wait(timeout=kill_wait_seconds)
    except subprocess.TimeoutExpired:
        raise
    except (ChildProcessError, ProcessLookupError):
        return
    if proc.poll() is None:
        raise subprocess.TimeoutExpired("process group cleanup", kill_wait_seconds)


def communicate_after_timeout(
    proc: Any,
    *,
    timeout_seconds: float,
    killpg: Any = os.killpg,
    stdout_limit_bytes: int = _DEFAULT_SUBPROCESS_STDOUT_LIMIT_BYTES,
    stderr_tail_bytes: int = _DEFAULT_SUBPROCESS_STDERR_TAIL_BYTES,
) -> tuple[bytes, bytes]:
    """兼容旧调用方，但清理阶段也使用有界 selector/fallback I/O。"""
    terminate_process_group(proc, killpg=killpg, force=True)
    try:
        return run_bounded_subprocess(
            proc,
            timeout_seconds=timeout_seconds,
            stdout_limit_bytes=stdout_limit_bytes,
            stderr_tail_bytes=stderr_tail_bytes,
            killpg=killpg,
        )
    except BoundedSubprocessTimeoutError as exc:
        raise subprocess.TimeoutExpired(
            "bounded subprocess cleanup", timeout_seconds,
            output=exc.stdout, stderr=exc.stderr,
        ) from exc


class BoundedSubprocessError(RuntimeError):
    """有界子进程 I/O 失败，并携带已保留的诊断输出。"""

    def __init__(
        self,
        message: str,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int | None = None,
    ) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class BoundedSubprocessTimeoutError(BoundedSubprocessError):
    """子进程超时后的有界终止错误。"""


class BoundedSubprocessOutputLimitError(BoundedSubprocessError):
    """子进程 stdout 超过硬上限。"""

    def __init__(self, stream: str, **kwargs: Any) -> None:
        self.stream = stream
        super().__init__(f"bounded subprocess {stream} exceeds output limit", **kwargs)


def _bytes_or_empty(value: object) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        return value.encode("utf-8", "replace")
    try:
        return bytes(value)
    except (TypeError, ValueError):
        return b""


def _stderr_tail_bytes(value: object, limit: int) -> bytes:
    return _bytes_or_empty(value)[-limit:]


def _bounded_outputs(
    stdout: object,
    stderr: object,
    *,
    stdout_limit_bytes: int,
    stderr_tail_bytes: int,
) -> tuple[bytes, bytes]:
    bounded_stdout = _bytes_or_empty(stdout)
    bounded_stderr = _stderr_tail_bytes(stderr, stderr_tail_bytes)
    if len(bounded_stdout) > stdout_limit_bytes:
        raise BoundedSubprocessOutputLimitError(
            "stdout", stdout=bounded_stdout[:stdout_limit_bytes + 1], stderr=bounded_stderr,
        )
    return bounded_stdout, bounded_stderr


def _process_streams(proc: Any) -> tuple[Any, Any, Any]:
    return (
        getattr(proc, "stdin", None),
        getattr(proc, "stdout", None),
        getattr(proc, "stderr", None),
    )


def _close_process_streams(proc: Any) -> None:
    for stream in _process_streams(proc):
        if stream is None:
            continue
        try:
            stream.close()
        except (OSError, ValueError):
            pass


def _fallback_bounded_subprocess(
    proc: Any,
    input_data: bytes | None,
    *,
    timeout_seconds: float,
    stdout_limit_bytes: int,
    stderr_tail_bytes: int,
    killpg: Any,
    cleanup_timeout_seconds: float,
) -> tuple[bytes, bytes]:
    """兼容无真实 pipe/fd 的 fake process；真实 Popen 不走此路径。"""
    initial_stdout = initial_stderr = b""
    try:
        if input_data is None:
            result = proc.communicate(timeout=timeout_seconds)
        else:
            result = proc.communicate(input_data, timeout=timeout_seconds)
        stdout, stderr = result if isinstance(result, tuple) and len(result) == 2 else (b"", b"")
        stdout, stderr = _bounded_outputs(
            stdout, stderr, stdout_limit_bytes=stdout_limit_bytes,
            stderr_tail_bytes=stderr_tail_bytes,
        )
    except subprocess.TimeoutExpired as exc:
        initial_stdout = _bytes_or_empty(
            getattr(exc, "stdout", None) or getattr(exc, "output", None)
        )
        initial_stderr = _bytes_or_empty(getattr(exc, "stderr", None))
        terminate_process_group(proc, killpg=killpg, force=True)
        try:
            result = proc.communicate(timeout=cleanup_timeout_seconds)
            cleanup_stdout, cleanup_stderr = (
                result if isinstance(result, tuple) and len(result) == 2 else (b"", b"")
            )
        except subprocess.TimeoutExpired as cleanup_exc:
            cleanup_stdout = _bytes_or_empty(
                getattr(cleanup_exc, "stdout", None) or getattr(cleanup_exc, "output", None)
            )
            cleanup_stderr = _bytes_or_empty(getattr(cleanup_exc, "stderr", None))
        stdout = _bytes_or_empty(cleanup_stdout) or initial_stdout or _bytes_or_empty(
            getattr(proc, "_stdout", None)
        )
        stderr = _bytes_or_empty(cleanup_stderr) or initial_stderr or _bytes_or_empty(
            getattr(proc, "_stderr", None)
        )
        try:
            stdout, stderr = _bounded_outputs(
                stdout, stderr, stdout_limit_bytes=stdout_limit_bytes,
                stderr_tail_bytes=stderr_tail_bytes,
            )
        except BoundedSubprocessOutputLimitError as exc:
            raise exc from None
        raise BoundedSubprocessTimeoutError(
            f"bounded subprocess timed out after {timeout_seconds:g}s",
            stdout=stdout, stderr=stderr, returncode=getattr(proc, "returncode", None),
        ) from exc
    except BaseException:
        terminate_process_group(proc, killpg=killpg, force=True)
        raise
    finally:
        _close_process_streams(proc)
    return stdout, stderr


def _drain_bounded_subprocess(
    proc: Any,
    input_data: bytes | None,
    *,
    timeout_seconds: float,
    stdout_limit_bytes: int,
    stderr_tail_bytes: int,
    killpg: Any,
    cleanup_timeout_seconds: float,
) -> tuple[bytes, bytes]:
    stdin, stdout_stream, stderr_stream = _process_streams(proc)
    streams: dict[str, Any] = {}
    for name, stream in (("stdin", stdin), ("stdout", stdout_stream), ("stderr", stderr_stream)):
        if stream is None:
            continue
        try:
            stream.fileno()
        except (AttributeError, OSError, ValueError):
            raise TypeError("subprocess streams do not expose usable file descriptors")
        streams[name] = stream

    selector = selectors.DefaultSelector()
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    offset = 0
    primary: BaseException | None = None
    terminated = False
    deadline = time.monotonic() + timeout_seconds

    def close_stream(name: str) -> None:
        stream = streams.pop(name, None)
        if stream is None:
            return
        try:
            selector.unregister(stream)
        except (KeyError, OSError, ValueError):
            pass
        try:
            stream.close()
        except (OSError, ValueError):
            pass

    def signal_and_reap() -> None:
        nonlocal terminated
        if terminated:
            return
        terminated = True
        terminate_process_group(
            proc, killpg=killpg,
            terminate_wait_seconds=cleanup_timeout_seconds,
            kill_wait_seconds=cleanup_timeout_seconds,
            force=True,
        )

    try:
        for name, stream in tuple(streams.items()):
            os.set_blocking(stream.fileno(), False)
            if name == "stdin":
                if input_data is None or not input_data:
                    close_stream(name)
                else:
                    selector.register(stream, selectors.EVENT_WRITE, name)
            else:
                selector.register(stream, selectors.EVENT_READ, name)

        while True:
            if time.monotonic() >= deadline:
                primary = BoundedSubprocessTimeoutError(
                    f"bounded subprocess timed out after {timeout_seconds:g}s",
                    stdout=bytes(stdout_buffer), stderr=bytes(stderr_buffer),
                    returncode=getattr(proc, "returncode", None),
                )
                signal_and_reap()
                break
            if proc.poll() is not None and not any(
                name in streams for name in ("stdout", "stderr")
            ):
                break
            wait_seconds = min(max(0.0, deadline - time.monotonic()), 0.1)
            events = selector.select(wait_seconds) if streams else []
            if not events:
                if proc.poll() is not None and not any(
                    name in streams for name in ("stdout", "stderr")
                ):
                    break
                continue
            for key, _ in events:
                name = key.data
                stream = streams.get(name)
                if stream is None:
                    continue
                try:
                    if name == "stdin":
                        written = os.write(
                            stream.fileno(), input_data[offset:offset + _SUBPROCESS_IO_CHUNK_BYTES],
                        )
                        if written <= 0:
                            raise OSError("bounded subprocess stdin made no progress")
                        offset += written
                        if offset >= len(input_data or b""):
                            close_stream(name)
                        continue
                    if name == "stdout":
                        remaining = max(1, stdout_limit_bytes + 1 - len(stdout_buffer))
                        chunk = os.read(
                            stream.fileno(), min(_SUBPROCESS_IO_CHUNK_BYTES, remaining),
                        )
                        if not chunk:
                            close_stream(name)
                            continue
                        stdout_buffer.extend(chunk)
                        if len(stdout_buffer) > stdout_limit_bytes:
                            primary = BoundedSubprocessOutputLimitError(
                                "stdout", stdout=bytes(stdout_buffer),
                                stderr=bytes(stderr_buffer),
                            )
                            signal_and_reap()
                            break
                    else:
                        chunk = os.read(
                            stream.fileno(),
                            min(_SUBPROCESS_IO_CHUNK_BYTES, stderr_tail_bytes + 1),
                        )
                        if not chunk:
                            close_stream(name)
                            continue
                        stderr_buffer.extend(chunk)
                        if len(stderr_buffer) > stderr_tail_bytes:
                            del stderr_buffer[:-stderr_tail_bytes]
                except (BlockingIOError, InterruptedError):
                    continue
                except BrokenPipeError:
                    close_stream(name)
                except OSError as exc:
                    primary = BoundedSubprocessError(
                        "bounded subprocess pipe I/O failed",
                        stdout=bytes(stdout_buffer), stderr=bytes(stderr_buffer),
                        returncode=getattr(proc, "returncode", None),
                    )
                    primary.__cause__ = exc
                    signal_and_reap()
                    break
            if primary is not None:
                break

        if primary is not None and not terminated:
            signal_and_reap()
        if primary is not None:
            try:
                proc.wait(timeout=cleanup_timeout_seconds)
            except (subprocess.TimeoutExpired, ChildProcessError, ProcessLookupError):
                pass
        else:
            try:
                proc.wait(timeout=cleanup_timeout_seconds)
            except subprocess.TimeoutExpired:
                signal_and_reap()
                try:
                    proc.wait(timeout=cleanup_timeout_seconds)
                except (subprocess.TimeoutExpired, ChildProcessError, ProcessLookupError):
                    pass
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit, BoundedSubprocessError)):
            primary = exc
        else:
            primary = BoundedSubprocessError(
                "bounded subprocess failed during selector I/O",
                stdout=bytes(stdout_buffer), stderr=bytes(stderr_buffer),
                returncode=getattr(proc, "returncode", None),
            )
            primary.__cause__ = exc
        signal_and_reap()
        try:
            proc.wait(timeout=cleanup_timeout_seconds)
        except (subprocess.TimeoutExpired, ChildProcessError, ProcessLookupError):
            pass
    finally:
        for name in tuple(streams):
            close_stream(name)
        try:
            selector.close()
        except OSError:
            pass

    stdout = bytes(stdout_buffer)
    stderr = bytes(stderr_buffer)
    if primary is not None:
        if isinstance(primary, BoundedSubprocessError):
            primary.stdout = stdout
            primary.stderr = stderr
            primary.returncode = getattr(proc, "returncode", None)
        raise primary
    return stdout, stderr


def run_bounded_subprocess(
    proc: Any,
    input_data: bytes | None = None,
    *,
    timeout_seconds: float,
    stdout_limit_bytes: int = _DEFAULT_SUBPROCESS_STDOUT_LIMIT_BYTES,
    stderr_tail_bytes: int = _DEFAULT_SUBPROCESS_STDERR_TAIL_BYTES,
    killpg: Any = os.killpg,
    cleanup_timeout_seconds: float = _SUBPROCESS_CLEANUP_TIMEOUT_SECONDS,
) -> tuple[bytes, bytes]:
    """通过 selector 有界收集 stdout/stderr，并在异常时清理独立 process group。"""
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be positive and finite")
    if type(stdout_limit_bytes) is not int or stdout_limit_bytes <= 0:
        raise ValueError("stdout_limit_bytes must be a positive integer")
    if type(stderr_tail_bytes) is not int or stderr_tail_bytes <= 0:
        raise ValueError("stderr_tail_bytes must be a positive integer")
    if (isinstance(cleanup_timeout_seconds, bool)
            or not isinstance(cleanup_timeout_seconds, (int, float))
            or not math.isfinite(float(cleanup_timeout_seconds))
            or cleanup_timeout_seconds <= 0):
        raise ValueError("cleanup_timeout_seconds must be positive and finite")
    try:
        streams = _process_streams(proc)
        if not any(stream is not None for stream in streams):
            return _fallback_bounded_subprocess(
                proc, input_data, timeout_seconds=float(timeout_seconds),
                stdout_limit_bytes=stdout_limit_bytes, stderr_tail_bytes=stderr_tail_bytes,
                killpg=killpg, cleanup_timeout_seconds=float(cleanup_timeout_seconds),
            )
        for stream in streams:
            if stream is not None:
                stream.fileno()
    except (AttributeError, OSError, TypeError, ValueError):
        return _fallback_bounded_subprocess(
            proc, input_data, timeout_seconds=float(timeout_seconds),
            stdout_limit_bytes=stdout_limit_bytes, stderr_tail_bytes=stderr_tail_bytes,
            killpg=killpg, cleanup_timeout_seconds=float(cleanup_timeout_seconds),
        )
    return _drain_bounded_subprocess(
        proc, input_data, timeout_seconds=float(timeout_seconds),
        stdout_limit_bytes=stdout_limit_bytes, stderr_tail_bytes=stderr_tail_bytes,
        killpg=killpg, cleanup_timeout_seconds=float(cleanup_timeout_seconds),
    )


def _source_payload(row: Mapping[str, Any]) -> dict[str, object]:
    if not isinstance(row, Mapping):
        raise ValueError("embedding row must be a mapping")
    path = row.get("original_path")
    if type(path) is not str or not path or not os.path.isabs(path) or path != os.path.normpath(os.path.abspath(path)):
        raise ValueError("embedding row original_path must be a canonical absolute path")
    sha256 = row.get("source_sha256")
    if type(sha256) is not str or _HEX.fullmatch(sha256) is None:
        raise ValueError("embedding row source_sha256 must be lowercase SHA-256 hex")
    fields = ("source_size_bytes", "sample_rate", "frames", "channels", "start_sample", "end_sample")
    values = {name: row.get(name) for name in fields}
    if any(type(value) is not int for value in values.values()):
        raise ValueError("embedding row source header and bounds must be exact integers")
    if (values["source_size_bytes"] < 0 or values["sample_rate"] <= 0 or values["frames"] <= 0
            or values["channels"] <= 0
            or not 0 <= values["start_sample"] < values["end_sample"] <= values["frames"]):
        raise ValueError("embedding row source header or bounds are invalid")
    return {"local_path": path, "sha256": sha256, "size_bytes": values["source_size_bytes"],
            "sample_rate": values["sample_rate"], "frames": values["frames"],
            "channels": values["channels"], "start_sample": values["start_sample"],
            "end_sample": values["end_sample"]}


def _embedding(result: object) -> tuple[float, ...]:
    if not isinstance(result, Mapping) or set(result) != {"embedding", "schema"}:
        raise WeSpeakerRuntimeRemoteError("WeSpeaker result schema is invalid")
    if result["schema"] != "audio_filter_pipeline.wespeaker_embedding.v1":
        raise WeSpeakerRuntimeRemoteError("WeSpeaker result schema is unsupported")
    values = result["embedding"]
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise WeSpeakerRuntimeRemoteError("WeSpeaker embedding is not a sequence")
    output = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise WeSpeakerRuntimeRemoteError("WeSpeaker embedding contains a non-finite value")
        output.append(float(value))
    if not output or not any(value != 0.0 for value in output):
        raise WeSpeakerRuntimeRemoteError("WeSpeaker embedding is empty or zero")
    return tuple(output)


def _embeddings(result: object, expected_count: int) -> tuple[tuple[float, ...], ...]:
    if (not isinstance(result, Mapping) or set(result) != {"embeddings", "schema"}
            or result["schema"] != "audio_filter_pipeline.wespeaker_embeddings.v1"):
        raise WeSpeakerRuntimeRemoteError("WeSpeaker batch result schema is invalid")
    values = result["embeddings"]
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != expected_count:
        raise WeSpeakerRuntimeRemoteError("WeSpeaker batch result count is invalid")
    return tuple(_embedding({"embedding": value, "schema": "audio_filter_pipeline.wespeaker_embedding.v1"})
                 for value in values)


class WeSpeakerEmbeddingSession:
    """单 shard 内复用模型的受限 WeSpeaker optional-worker 会话。"""

    def __init__(self, runtime: "WeSpeakerEmbeddingRuntime") -> None:
        self._runtime = runtime
        self._session = OptionalWorkerSession(
            runtime._spec, runtime._asset, limits=runtime._options.worker_limits,
            asset_limits=runtime._options.asset_limits,
        )

    def __enter__(self) -> "WeSpeakerEmbeddingSession":
        try:
            self._session.__enter__()
        except (OptionalWorkerError, LocalAssetError) as exc:
            self._runtime._raise_optional_worker_error(exc)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        try:
            self._session.__exit__(exc_type, exc, traceback)
        except (OptionalWorkerError, LocalAssetError) as error:
            if exc is not None:
                add_note = getattr(exc, "add_note", None)
                note = "WeSpeaker session cleanup: " + repr(error)
                if callable(add_note):
                    add_note(note)
                else:
                    exc.__notes__ = [*getattr(exc, "__notes__", ()), note]
                return False
            self._runtime._raise_optional_worker_error(error)
        return False

    def extract_batch(self, rows: Sequence[Mapping[str, Any]]) -> tuple[tuple[float, ...], ...]:
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
            raise ValueError("WeSpeaker batch rows must be a nonempty sequence")
        sources = [_source_payload(row) for row in rows]
        if len(sources) > 32:
            raise ValueError("WeSpeaker session batch cannot exceed 32 rows")
        try:
            result = self._session.call(
                "extract_embeddings_pcm_v1",
                {"sources": sources, "target_sample_rate": self._runtime._options.target_sample_rate},
            )
        except OptionalWorkerError as exc:
            self._runtime._raise_optional_worker_error(exc)
        return _embeddings(result, len(sources))


class WeSpeakerEmbeddingRuntime:
    """将 Step3 行映射为严格绑定的 WeSpeaker optional-worker 请求。"""

    def __init__(self, options: WeSpeakerRuntimeOptions) -> None:
        if not isinstance(options, WeSpeakerRuntimeOptions):
            raise TypeError("options must be WeSpeakerRuntimeOptions")
        self._options = options
        self._asset = LocalModelAsset(options.model_path, options.model_version, options.model_sha256)
        env_overrides = ()
        if options.cuda_visible_devices is not None:
            env_overrides = (("CUDA_VISIBLE_DEVICES", options.cuda_visible_devices),)
        self._spec = OptionalWorkerSpec(
            options.python_executable, options.trusted_code_root, options.worker_script,
            options.worker_script_sha256, "wespeaker", options.runtime_version,
            env_overrides=env_overrides,
            asset_env_paths=(("NLTK_DATA", "nltk_data"),),
        )
        provenance = {
            "capability": "wespeaker_optional_worker", "runtime": "wespeaker",
            "runtime_version": options.runtime_version, "model_version": options.model_version,
            "model_sha256": options.model_sha256, "target_sample_rate": options.target_sample_rate,
        }
        if options.cuda_visible_devices is not None:
            provenance["cuda_visible_devices"] = options.cuda_visible_devices
        self._provenance = MappingProxyType(provenance)

    @property
    def provenance(self) -> Mapping[str, object]:
        return self._provenance

    def _raise_optional_worker_error(self, exc: OptionalWorkerError) -> None:
        if isinstance(exc, (AssetDigestMismatchError, WorkerRemoteAssetError)):
            raise WeSpeakerRuntimeAssetError("WeSpeaker model or source asset verification failed") from exc
        if isinstance(exc, (WorkerUnavailableError, LocalAssetError)):
            raise WeSpeakerRuntimeUnavailableError("WeSpeaker optional runtime is unavailable") from exc
        if isinstance(exc, WorkerRuntimeVersionMismatchError):
            raise WeSpeakerRuntimeVersionError("WeSpeaker runtime version does not exactly match configuration") from exc
        if isinstance(exc, WorkerProtocolError):
            raise WeSpeakerRuntimeRemoteError("WeSpeaker worker protocol failed") from exc
        if isinstance(exc, WorkerRemoteError):
            raise WeSpeakerRuntimeRemoteError(f"WeSpeaker worker failed: {exc.code}") from exc
        raise WeSpeakerRuntimeUnavailableError(
            "WeSpeaker optional runtime failed before producing a result"
        ) from exc

    def _invoke(self, operation: str, payload: dict[str, object]) -> object:
        try:
            return run_optional_worker(
                self._spec, self._asset, operation, payload,
                limits=self._options.worker_limits, asset_limits=self._options.asset_limits,
            )
        except (OptionalWorkerError, LocalAssetError) as exc:
            self._raise_optional_worker_error(exc)

    def session(self) -> WeSpeakerEmbeddingSession:
        return WeSpeakerEmbeddingSession(self)

    def __call__(self, row: Mapping[str, Any]) -> tuple[float, ...]:
        return _embedding(self._invoke(
            "extract_embedding_pcm_v1",
            {"source": _source_payload(row), "target_sample_rate": self._options.target_sample_rate},
        ))

    def extract_batch(self, rows: Sequence[Mapping[str, Any]]) -> tuple[tuple[float, ...], ...]:
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
            raise ValueError("WeSpeaker batch rows must be a nonempty sequence")
        sources = [_source_payload(row) for row in rows]
        embeddings = []
        for start in range(0, len(sources), 32):
            batch_sources = sources[start:start + 32]
            embeddings.extend(_embeddings(self._invoke(
                "extract_embeddings_pcm_v1",
                {"sources": batch_sources, "target_sample_rate": self._options.target_sample_rate},
            ), len(batch_sources)))
        return tuple(embeddings)
