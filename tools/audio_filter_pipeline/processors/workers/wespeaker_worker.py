"""Isolated local WeSpeaker embedding optional-runtime worker."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import io
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from numbers import Real
from types import MappingProxyType
from urllib.parse import urlsplit

__all__ = ["handle_request", "main"]

_PROTOCOL = "audio_filter_pipeline.optional_worker.v1"
_OPERATION = "extract_embedding_pcm_v1"
_BATCH_OPERATION = "extract_embeddings_pcm_v1"
_RUNTIME_NAME = "wespeaker"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_MAX_STDIN_BYTES = 1024 * 1024
_MAX_BATCH_SOURCES = 32
_WORKER_ERROR_MAX_CHARS = 4_096
_ROOT_KEYS = {"asset", "nonce", "operation", "payload", "protocol", "request_id",
              "runtime", "worker_script_sha256"}


class _InvalidRequest(Exception):
    pass


class _SourceMissing(Exception):
    pass


class _SourceMismatch(Exception):
    pass


class _NltkDataMissing(Exception):
    pass


@contextlib.contextmanager
def _discard_third_party_stdout() -> object:
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _text(value: object, maximum: int) -> bool:
    if type(value) is not str or not value or value != value.strip():
        return False
    try:
        return len(value.encode("utf-8", "strict")) <= maximum and not any(
            ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)
    except UnicodeEncodeError:
        return False


def _path(value: object) -> bool:
    if not _text(value, 4096) or "://" in value:
        return False
    try:
        parsed = urlsplit(value)
        return (not parsed.scheme and not parsed.netloc and os.path.isabs(value)
                and value == os.path.normpath(os.path.abspath(value)))
    except (TypeError, ValueError, OSError):
        return False


def _keys(value: object, expected: set[str]) -> bool:
    return type(value) is dict and set(value) == expected and all(type(key) is str for key in value)


def _bindings(request: object) -> dict[str, object] | None:
    if not _keys(request, _ROOT_KEYS):
        return None
    asset, runtime = request.get("asset"), request.get("runtime")
    if (not _keys(asset, {"local_path", "sha256", "version"})
            or not _keys(runtime, {"name", "version"})
            or request.get("protocol") != _PROTOCOL
            or not _path(asset.get("local_path"))
            or not _text(asset.get("version"), 256)
            or type(asset.get("sha256")) is not str or _HEX.fullmatch(asset["sha256"]) is None
            or not _text(runtime.get("name"), 128) or not _text(runtime.get("version"), 256)
            or any(type(request.get(name)) is not str or _HEX.fullmatch(request[name]) is None
                   for name in ("nonce", "request_id", "worker_script_sha256"))):
        return None
    return {"asset": dict(asset), "nonce": request["nonce"], "protocol": _PROTOCOL,
            "request_id": request["request_id"], "runtime_name": runtime["name"],
            "runtime_version": runtime["version"],
            "worker_script_sha256": request["worker_script_sha256"]}


def _error(bindings: dict[str, object] | None, code: str, message: str,
           *, version: str | None = None) -> dict[str, object]:
    if bindings is None:
        return {"error": {"code": code, "message": message}, "status": "error"}
    return {"asset": bindings["asset"], "error": {"code": code, "message": message},
            "nonce": bindings["nonce"], "protocol": bindings["protocol"],
            "request_id": bindings["request_id"],
            "runtime": {"name": bindings["runtime_name"], "version":
                        bindings["runtime_version"] if version is None else version},
            "status": "error", "worker_script_sha256": bindings["worker_script_sha256"]}


def _validated_source(value: object) -> Mapping[str, object]:
    source_keys = {"local_path", "sha256", "size_bytes", "sample_rate", "frames", "channels",
                   "start_sample", "end_sample"}
    if not _keys(value, source_keys):
        raise _InvalidRequest
    integers = ("size_bytes", "sample_rate", "frames", "channels", "start_sample", "end_sample")
    if (not _path(value["local_path"]) or type(value["sha256"]) is not str
            or _HEX.fullmatch(value["sha256"]) is None
            or any(type(value[name]) is not int for name in integers)
            or value["size_bytes"] < 0 or value["sample_rate"] <= 0 or value["frames"] <= 0
            or value["channels"] <= 0
            or not 0 <= value["start_sample"] < value["end_sample"] <= value["frames"]):
        raise _InvalidRequest
    return MappingProxyType(dict(value))


def _validate_request(request: object) -> tuple[dict[str, object], tuple[Mapping[str, object], ...], int, str]:
    bindings = _bindings(request)
    if bindings is None or not isinstance(request, dict):
        raise _InvalidRequest
    operation = request.get("operation")
    if bindings["runtime_name"] != _RUNTIME_NAME or operation not in {_OPERATION, _BATCH_OPERATION}:
        raise _InvalidRequest
    payload = request.get("payload")
    if type(payload) is not dict or set(payload) != ({"source", "target_sample_rate"} if operation == _OPERATION else {"sources", "target_sample_rate"}):
        raise _InvalidRequest
    target_rate = payload.get("target_sample_rate")
    if type(target_rate) is not int or target_rate <= 0:
        raise _InvalidRequest
    if operation == _OPERATION:
        sources = (_validated_source(payload["source"]),)
    else:
        values = payload["sources"]
        if type(values) is not list or not 1 <= len(values) <= _MAX_BATCH_SOURCES:
            raise _InvalidRequest
        sources = tuple(_validated_source(source) for source in values)
    return bindings, sources, target_rate, operation


def _snapshot(value: os.stat_result) -> tuple[object, ...]:
    return (value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode), value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _digest_fd(fd: int, size: int) -> str:
    os.lseek(fd, 0, os.SEEK_SET)
    digest, remaining = hashlib.sha256(), size
    while remaining:
        chunk = os.read(fd, min(1024 * 1024, remaining))
        if not chunk:
            raise _SourceMismatch
        digest.update(chunk)
        remaining -= len(chunk)
    if os.read(fd, 1):
        raise _SourceMismatch
    return digest.hexdigest()


def _open_verified_source(source: Mapping[str, object]) -> tuple[int, tuple[object, ...]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(source["local_path"], flags)
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_size != source["size_bytes"]
                or _digest_fd(fd, source["size_bytes"]) != source["sha256"]):
            raise _SourceMismatch
        return fd, _snapshot(before)
    except FileNotFoundError:
        raise _SourceMissing from None
    except OSError:
        raise _SourceMissing from None
    except BaseException:
        if "fd" in locals():
            os.close(fd)
        raise


def _decode_exact_segment(fd: int, source: Mapping[str, object]) -> object:
    import soundfile  # Optional worker dependency; never imported by the parent.

    os.lseek(fd, 0, os.SEEK_SET)
    with os.fdopen(os.dup(fd), "rb", closefd=True) as stream:
        with soundfile.SoundFile(stream) as audio:
            if (audio.samplerate != source["sample_rate"] or audio.frames != source["frames"]
                    or audio.channels != source["channels"]):
                raise _SourceMismatch
            audio.seek(source["start_sample"])
            segment = audio.read(source["end_sample"] - source["start_sample"],
                                 dtype="float32", always_2d=True)
    if getattr(segment, "shape", None) != (source["end_sample"] - source["start_sample"], source["channels"]):
        raise _SourceMismatch
    return segment


def _mono_and_resample(segment: object, source_rate: int, target_rate: int) -> object:
    import numpy as np  # Optional worker dependency; never imported by the parent.

    values = np.asarray(segment, dtype=np.float32)
    # Channel axis is averaged with equal weights; no channel is privileged.
    mono = np.mean(values, axis=1, dtype=np.float32)
    if target_rate == source_rate:
        return np.ascontiguousarray(mono, dtype=np.float32)
    import torchaudio
    import torch
    waveform = torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0)
    resampled = torchaudio.functional.resample(waveform, source_rate, target_rate).squeeze(0)
    return np.ascontiguousarray(resampled.detach().cpu().numpy(), dtype=np.float32)


def _extract_embedding(model: object, pcm: object, sample_rate: int) -> list[float]:
    import soundfile  # Optional worker dependency; never imported by the parent.

    directory = tempfile.mkdtemp(prefix="wespeaker-")
    wav_path = os.path.join(directory, "segment.wav")
    try:
        soundfile.write(wav_path, pcm, sample_rate, format="WAV", subtype="FLOAT")
        with _discard_third_party_stdout():
            embedding = model.extract_embedding(wav_path)
        return _validate_embedding(embedding)
    finally:
        shutil.rmtree(directory)


def _require_nltk_data(asset_path: str) -> None:
    expected = os.path.join(asset_path, "nltk_data")
    configured = os.environ.get("NLTK_DATA")
    try:
        info = os.lstat(expected)
    except OSError:
        raise _NltkDataMissing from None
    if (configured != expected or not os.path.isabs(configured)
            or configured != os.path.normpath(os.path.abspath(configured))
            or stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)):
        raise _NltkDataMissing
    try:
        import nltk  # Optional worker dependency; never imported by the parent.

        nltk.data.path[:] = [expected]
        # g2p_en resolves these archive resources before its first inference.
        nltk.data.find("taggers/averaged_perceptron_tagger.zip")
        nltk.data.find("corpora/cmudict.zip")
    except Exception:
        raise _NltkDataMissing from None


def _load_model(asset_path: str) -> object:
    _require_nltk_data(asset_path)
    with _discard_third_party_stdout():
        import wespeaker  # Optional worker dependency; never imported by the parent.

        return wespeaker.load_model(asset_path)


def _embedding(asset_path: str, pcm: object, sample_rate: int) -> list[float]:
    return _extract_embedding(_load_model(asset_path), pcm, sample_rate)


def _validate_embedding(value: object) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        try:
            value = value.tolist()
        except AttributeError:
            raise ValueError from None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real) or not math.isfinite(float(item)):
            raise ValueError
        result.append(float(item))
    if not result or not any(item != 0.0 for item in result):
        raise ValueError
    return result


def _verify_unchanged(fd: int, baseline: tuple[object, ...], source: Mapping[str, object]) -> None:
    if (_snapshot(os.fstat(fd)) != baseline
            or _digest_fd(fd, source["size_bytes"]) != source["sha256"]
            or _snapshot(os.fstat(fd)) != baseline):
        raise _SourceMismatch


def _success(bindings: Mapping[str, object], result: Mapping[str, object]) -> dict[str, object]:
    return {"asset": bindings["asset"], "nonce": bindings["nonce"], "protocol": bindings["protocol"],
            "request_id": bindings["request_id"], "result": dict(result),
            "runtime": {"name": bindings["runtime_name"], "version": bindings["runtime_version"]},
            "status": "ok", "worker_script_sha256": bindings["worker_script_sha256"]}


def _handle_request(
    request: dict[str, object], *, installed_version: str | None = None,
    model: object | None = None, session_binding: Mapping[str, object] | None = None,
    persistent: bool = False,
) -> tuple[dict[str, object], object | None, Mapping[str, object] | None]:
    bindings = _bindings(request)
    try:
        bindings, sources, target_rate, operation = _validate_request(request)
    except _InvalidRequest:
        return _error(bindings, "invalid_request", "Invalid request."), model, session_binding
    try:
        observed = installed_version if installed_version is not None else importlib.metadata.version(_RUNTIME_NAME)
        if not _text(observed, 256): raise ValueError
    except Exception:
        return _error(bindings, "runtime_error", "WeSpeaker runtime unavailable."), model, session_binding
    if observed != bindings["runtime_version"]:
        return _error(bindings, "runtime_version_mismatch", "Runtime version mismatch.", version=observed), model, session_binding
    binding = {"asset": bindings["asset"], "runtime_name": bindings["runtime_name"],
               "runtime_version": bindings["runtime_version"], "worker_script_sha256": bindings["worker_script_sha256"]}
    if session_binding is not None and binding != session_binding:
        return _error(bindings, "invalid_request", "Session binding mismatch."), model, session_binding
    try:
        use_one_shot_embedding = (not persistent and model is None and session_binding is None
                                  and operation == _OPERATION)
        if model is None and not use_one_shot_embedding:
            model = _load_model(bindings["asset"]["local_path"])
        embeddings = []
        for source in sources:
            fd = None
            try:
                fd, baseline = _open_verified_source(source)
                pcm = _mono_and_resample(_decode_exact_segment(fd, source), source["sample_rate"], target_rate)
                embedding = (_embedding(bindings["asset"]["local_path"], pcm, target_rate)
                             if use_one_shot_embedding else _extract_embedding(model, pcm, target_rate))
                _verify_unchanged(fd, baseline, source)
                embeddings.append(embedding)
            finally:
                if fd is not None: os.close(fd)
        result = ({"embedding": embeddings[0], "schema": "audio_filter_pipeline.wespeaker_embedding.v1"}
                  if operation == _OPERATION else
                  {"embeddings": embeddings, "schema": "audio_filter_pipeline.wespeaker_embeddings.v1"})
    except _SourceMissing:
        return _error(bindings, "asset_missing", "Source asset is unavailable."), model, session_binding
    except _SourceMismatch:
        return _error(bindings, "asset_digest_mismatch", "Source asset verification failed."), model, session_binding
    except _NltkDataMissing:
        return _error(bindings, "asset_missing", "Required NLTK model asset is unavailable."), model, session_binding
    except Exception:
        return _error(bindings, "runtime_error", "WeSpeaker runtime failed."), model, session_binding
    return _success(bindings, result), model, (binding if session_binding is None else session_binding)


def handle_request(request: dict[str, object], *, installed_version: str | None = None) -> dict[str, object]:
    response, _model, _binding = _handle_request(request, installed_version=installed_version)
    return response


def _read_input() -> bytes:
    raw = sys.stdin.buffer.readline(_MAX_STDIN_BYTES + 1)
    if not raw or len(raw) > _MAX_STDIN_BYTES or not raw.endswith(b"\n") or b"\n" in raw[:-1] or b"\r" in raw:
        raise ValueError
    return raw


def _write_response(response: Mapping[str, object]) -> None:
    sys.stdout.buffer.write(json.dumps(response, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n")
    sys.stdout.buffer.flush()


def _report_worker_error(exc: BaseException) -> None:
    """把 worker 根因写到 stderr；父进程只保留 stderr 尾部，消息必须有界。"""
    message = f"{type(exc).__name__}: {exc}"[:_WORKER_ERROR_MAX_CHARS]
    try:
        sys.stderr.write(f"worker failed: {message}\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        # stderr 不可写（父进程已关闭管道）时仍以 exit code 2 表达失败。
        pass


def _persistent_main() -> int:
    model, binding = None, None
    while True:
        try:
            raw = _read_input()
        except ValueError:
            return 0
        try:
            request = json.loads(raw[:-1].decode("utf-8", "strict"))
            response, model, binding = _handle_request(
                request, model=model, session_binding=binding, persistent=True,
            )
        except Exception as exc:
            _report_worker_error(exc)
            return 2
        _write_response(response)


def main() -> int:
    if sys.argv[1:] == ["--persistent"]:
        return _persistent_main()
    if sys.argv[1:]:
        return 2
    try:
        request = json.loads(_read_input()[:-1].decode("utf-8", "strict"))
        response = handle_request(request)
    except Exception:
        response = {"error": {"code": "invalid_request", "message": "Invalid JSON request."}, "status": "error"}
    _write_response(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
