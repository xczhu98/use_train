"""Subprocess worker for Generation-3 Step3 GPU MOS scoring.

The parent supplies an absolute JSON request path.  Rows and results remain JSONL so
one CUDA-bound process can score an arbitrarily large contiguous or sparse ordinal shard.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import re
import os
import stat
import sys
import tempfile
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# Running this file directly sets sys.path to ``processors/workers``.  Add the
# directory containing the audio_filter_pipeline package, rather than relying on
# the caller's working directory.
_PROJECT_PARENT = str(Path(__file__).resolve().parents[3])
if _PROJECT_PARENT not in sys.path:
    sys.path.insert(0, _PROJECT_PARENT)

from audio_filter_pipeline.processors.mos_contract import (  # noqa: E402
    MOS_ASSET_LOAD_BINDING,
    MOS_MODEL_REQUEST_OPTION_FIELDS,
    MOS_SCORING_EXECUTION_POLICY,
    MOS_SHIPPED_MODELS,
    MOS_WORKER_ORDINAL_SIDECAR_FIELDS,
    MOS_WORKER_PROTOCOL,
    MOS_WORKER_REQUEST_FIELDS,
    MOS_WORKER_SUBSHARD_FIELDS,
)
from audio_filter_pipeline.processors.stages_v3.mos_runtime import (  # noqa: E402
    MosRuntimeOptions,
    Step3MosRuntime,
    freeze_model_options,
    plain_json,
)
from audio_filter_pipeline.processors.mos import (  # noqa: E402
    clear_model_cache,
    normalize_selectable_mos_models,
)
from audio_filter_pipeline.processors.stages_v3.mos_lpt import (  # noqa: E402
    MAX_WORKER_INPUT_SUBSHARDS,
    STEP3_JSONL_MAX_FILE_BYTES,
    DiskBackedOrdinalRegistry,
)
from audio_filter_pipeline.processors.stages_v3.select import (  # noqa: E402
    Step3MosSelectionStage,
)

_PROTOCOL = MOS_WORKER_PROTOCOL
# 保留的兼容默认 tuple，不代表 shipped 生产 route；实际集合取 request 非空 options 并集。
_MODELS = MOS_SHIPPED_MODELS
_OPTION_FIELDS = MOS_MODEL_REQUEST_OPTION_FIELDS
# 资产 binding、评分执行 policy 与 worker protocol 分别精确校验，不协商旧 artifact。
ASSET_LOAD_BINDING = MOS_ASSET_LOAD_BINDING
SCORING_EXECUTION_POLICY = MOS_SCORING_EXECUTION_POLICY
_JSONL_MAX_LINE_BYTES = 8 * 1024 * 1024
_JSONL_MAX_FILE_BYTES = STEP3_JSONL_MAX_FILE_BYTES
_FILE_READ_FLAGS = (
    os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
)
_FILE_CREATE_FLAGS = (
    os.O_WRONLY | os.O_CREAT | os.O_EXCL
    | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
)
_ORDINAL_SIDECAR_FIELDS = MOS_WORKER_ORDINAL_SIDECAR_FIELDS
_CONTIGUOUS_SUBSHARD_FIELDS = (
    MOS_WORKER_SUBSHARD_FIELDS - MOS_WORKER_ORDINAL_SIDECAR_FIELDS
)


from audio_filter_pipeline.utils.strict_jsonl import (  # noqa: E402
    add_cleanup_note as _add_cleanup_note,
    decode_json_line as _shared_decode_json_line,
    encode_json_line as _shared_encode_json_line,
    file_signature as _file_signature,
    finalize_streams as _finalize_streams,
    iter_bounded_jsonl as _shared_iter_bounded_jsonl,
    open_new_regular_stream as _open_new_regular_stream,
    open_regular_stream as _open_regular_stream,
    parse_finite_float as _parse_finite_float,
    record_cleanup_error as _record_cleanup_error,
    reject_json_constant as _reject_json_constant,
    strict_json_object as _strict_json_object,
)


def _decode_json_line(payload: bytes, label: str) -> Any:
    if len(payload) > _JSONL_MAX_LINE_BYTES or not payload.endswith(b"\n"):
        raise ValueError(f"{label} exceeds JSONL framing limits")
    return _shared_decode_json_line(
        payload, label, max_line_bytes=_JSONL_MAX_LINE_BYTES,
    )


def _iter_bounded_jsonl(source: Any, label: str):
    return _shared_iter_bounded_jsonl(
        source,
        label,
        max_line_bytes=_JSONL_MAX_LINE_BYTES,
        max_file_bytes=_JSONL_MAX_FILE_BYTES,
    )


def _encode_json_line(value: Mapping[str, Any], label: str) -> bytes:
    return _shared_encode_json_line(
        value, label, max_line_bytes=_JSONL_MAX_LINE_BYTES,
    )


def _validate_request_options(request: Mapping[str, Any]) -> tuple[str, ...]:
    """校验五个 canonical option slots，返回本次 request 的实际模型集合。"""
    configured = []
    for model, field in _OPTION_FIELDS.items():
        value = request[field]
        if value is None:
            continue
        freeze_model_options(value, model, label=field)
        configured.append(model)
    return tuple(configured)


def _write_runtime_attestation(
    path: str, worker_models: tuple[str, ...], attestations: list[Mapping[str, Any]],
) -> None:
    """以 O_EXCL sidecar 发布 worker provider 证明。"""
    value = {
        "protocol": _PROTOCOL,
        "asset_load_binding": plain_json(ASSET_LOAD_BINDING),
        "scoring_execution_policy": plain_json(SCORING_EXECUTION_POLICY),
        "worker_models": list(worker_models),
        "model_tuple_attestations": attestations,
    }
    stream = _open_new_regular_stream(path, "MOS worker runtime attestation")
    try:
        stream.write(_encode_json_line(value, "MOS worker runtime attestation"))
    finally:
        _finalize_streams((stream,), sys.exc_info()[1], sync=True)


def _path_signature(path: str, label: str) -> tuple[int, int, int, int, int]:
    try:
        value = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        raise RuntimeError(f"{label} is missing") from None
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return _file_signature(value)


def _iter_ordinal_sidecar(
    sidecar: Mapping[str, Any], start: int, stop: int, row_count: int,
    private_dir: str, *, stream: Any | None = None,
    signature: tuple[int, int, int, int, int] | None = None,
):
    if not isinstance(sidecar, Mapping) or set(sidecar) != _ORDINAL_SIDECAR_FIELDS:
        raise ValueError("MOS sparse ordinal sidecar schema is invalid")
    path = sidecar["ordinal_sidecar_path"]
    digest = sidecar["ordinal_sidecar_sha256"]
    size_bytes = sidecar["ordinal_sidecar_size_bytes"]
    if (
        type(path) is not str or not os.path.isabs(path)
        or type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or type(size_bytes) is not int or not 0 < size_bytes <= _JSONL_MAX_FILE_BYTES
        or os.path.dirname(os.path.realpath(path)) != private_dir
    ):
        raise ValueError("MOS sparse ordinal sidecar values are invalid")
    owns_stream = stream is None
    if stream is None:
        stream, initial = _open_regular_stream(path, "MOS sparse ordinal sidecar")
        if initial.st_size != size_bytes:
            primary = ValueError("MOS sparse ordinal sidecar size mismatch")
            _finalize_streams((stream,), primary, sync=False)
            raise primary
        signature = _file_signature(initial)
    if signature is None:
        raise RuntimeError("MOS sparse ordinal sidecar signature is unavailable")
    if signature[2] != size_bytes:
        primary = ValueError("MOS sparse ordinal sidecar size mismatch")
        if owns_stream:
            _finalize_streams((stream,), primary, sync=False)
        raise primary
    payloads = None
    observed_digest = hashlib.sha256()
    previous = -1
    count = 0
    try:
        if _path_signature(path, "MOS sparse ordinal sidecar") != signature:
            raise ValueError("MOS sparse ordinal sidecar pathname changed before validation")
        payloads = iter(_iter_bounded_jsonl(stream, "MOS sparse ordinal sidecar"))
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
            if count > row_count:
                raise ValueError("MOS sparse ordinal sidecar row count is invalid")
            yield ordinal
        if count != row_count:
            raise ValueError("MOS sparse ordinal sidecar row count is invalid")
        if observed_digest.hexdigest() != digest:
            raise ValueError("MOS sparse ordinal sidecar digest mismatch")
        if _file_signature(os.fstat(stream.fileno())) != signature:
            raise ValueError("MOS sparse ordinal sidecar changed during validation")
        if _path_signature(path, "MOS sparse ordinal sidecar") != signature:
            raise ValueError("MOS sparse ordinal sidecar pathname changed during validation")
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
        if owns_stream:
            try:
                _finalize_streams(
                    (stream,), None if closing_generator else primary or cleanup_error,
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


def _load_request(path_value: str) -> dict[str, Any]:
    """加载精确匹配 protocol、资产 binding 与评分执行 policy 的 request。"""
    if type(path_value) is not str or not os.path.isabs(path_value):
        raise ValueError("worker request path must be absolute")
    stream, initial = _open_regular_stream(path_value, "worker request")
    try:
        if initial.st_size > _JSONL_MAX_LINE_BYTES:
            raise ValueError("worker request exceeds the size limit")
        raw = stream.read(_JSONL_MAX_LINE_BYTES + 1)
        final = os.fstat(stream.fileno())
        if (
            len(raw) > _JSONL_MAX_LINE_BYTES
            or _file_signature(final) != _file_signature(initial)
            or len(raw) != final.st_size
        ):
            raise ValueError("worker request changed or exceeds the size limit")
    finally:
        _finalize_streams((stream,), sys.exc_info()[1], sync=False)
    try:
        request = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("worker request contains invalid JSON") from exc
    if not isinstance(request, dict) or set(request) != MOS_WORKER_REQUEST_FIELDS:
        raise ValueError("worker request schema is invalid")
    if request["protocol"] != _PROTOCOL:
        raise ValueError("worker protocol is invalid")
    if request["asset_load_binding"] != plain_json(ASSET_LOAD_BINDING):
        raise ValueError("worker asset load binding is invalid")
    if request["scoring_execution_policy"] != plain_json(SCORING_EXECUTION_POLICY):
        raise ValueError("worker scoring execution policy is invalid")
    if (not isinstance(request["private_dir"], str)
            or not os.path.isabs(request["private_dir"])
            or not isinstance(request["attestation_path"], str)
            or not os.path.isabs(request["attestation_path"])
            or not isinstance(request["subshards"], list)
            or not request["subshards"]):
        raise ValueError("worker request values are invalid")
    if len(request["subshards"]) > MAX_WORKER_INPUT_SUBSHARDS:
        raise ValueError("worker subshard count exceeds canonical capacity")
    configured_models = set(_validate_request_options(request))
    private_dir = os.path.realpath(request["private_dir"])
    if os.path.dirname(os.path.realpath(path_value)) != private_dir:
        raise ValueError("worker request path must remain in the private request directory")
    if os.path.dirname(os.path.realpath(request["attestation_path"])) != private_dir:
        raise ValueError("worker attestation path must remain in the private request directory")
    requested_models = set()
    request_path = os.path.realpath(path_value)
    attestation_path = os.path.realpath(request["attestation_path"])
    if request_path == attestation_path:
        raise ValueError("worker request and attestation paths must be unique")
    owned_paths = {request_path, attestation_path}
    for subshard in request["subshards"]:
        fields = set(subshard) if isinstance(subshard, dict) else set()
        sparse_fields = fields & _ORDINAL_SIDECAR_FIELDS
        if (
            not isinstance(subshard, dict)
            or fields not in (
                _CONTIGUOUS_SUBSHARD_FIELDS, MOS_WORKER_SUBSHARD_FIELDS,
            )
            or sparse_fields and sparse_fields != _ORDINAL_SIDECAR_FIELDS
        ):
            raise ValueError("worker subshard schema is invalid")
        models = normalize_selectable_mos_models(subshard["models"])
        requested_models.update(models)
        start = subshard["start_ordinal"]
        stop = subshard["end_ordinal"]
        ordinal_sidecar = (
            {field: subshard[field] for field in _ORDINAL_SIDECAR_FIELDS}
            if sparse_fields else None
        )
        option_field = MOS_MODEL_REQUEST_OPTION_FIELDS
        if (not isinstance(subshard["input_path"], str)
                or not os.path.isabs(subshard["input_path"])
                or not isinstance(subshard["output_path"], str)
                or not os.path.isabs(subshard["output_path"])
                or type(subshard["row_count"]) is not int
                or subshard["row_count"] <= 0
                or not isinstance(subshard["input_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", subshard["input_sha256"])
                or type(start) is not int or type(stop) is not int
                or start < 0 or stop <= start
                or ordinal_sidecar is None and stop - start != subshard["row_count"]
                or ordinal_sidecar is not None and (
                    type(ordinal_sidecar["ordinal_sidecar_path"]) is not str
                    or not os.path.isabs(ordinal_sidecar["ordinal_sidecar_path"])
                    or type(ordinal_sidecar["ordinal_sidecar_sha256"]) is not str
                    or not re.fullmatch(
                        r"[0-9a-f]{64}", ordinal_sidecar["ordinal_sidecar_sha256"],
                    )
                    or type(ordinal_sidecar["ordinal_sidecar_size_bytes"]) is not int
                    or not 0 < ordinal_sidecar["ordinal_sidecar_size_bytes"] <= _JSONL_MAX_FILE_BYTES
                )
                or any(request[option_field[model]] is None for model in models)):
            raise ValueError("worker subshard values are invalid")
        if any(os.path.dirname(os.path.realpath(subshard[field])) != private_dir
               for field in ("input_path", "output_path")):
            raise ValueError("worker subshard paths must remain in the private request directory")
        subshard_paths = [
            os.path.realpath(subshard["input_path"]),
            os.path.realpath(subshard["output_path"]),
        ]
        if ordinal_sidecar is not None:
            subshard_paths.append(os.path.realpath(
                ordinal_sidecar["ordinal_sidecar_path"],
            ))
        if len(set(subshard_paths)) != len(subshard_paths) or any(
            path in owned_paths for path in subshard_paths
        ):
            raise ValueError("worker input, output, and sidecar paths must be globally unique")
        owned_paths.update(subshard_paths)
        subshard["models"] = list(models)
    if requested_models != configured_models:
        raise ValueError(
            "worker options must exactly match the requested model union"
        )
    return request


def _input_path_signature(path: str) -> tuple[int, int, int, int, int]:
    return _path_signature(path, "MOS worker input shard")


def _expected_ordinal_iterator(
    subshard: Mapping[str, Any], *, sidecar_stream: Any | None = None,
    sidecar_signature: tuple[int, int, int, int, int] | None = None,
):
    start = subshard["start_ordinal"]
    stop = subshard["end_ordinal"]
    if _ORDINAL_SIDECAR_FIELDS <= set(subshard):
        sidecar = {field: subshard[field] for field in _ORDINAL_SIDECAR_FIELDS}
        return _iter_ordinal_sidecar(
            sidecar, start, stop, subshard["row_count"],
            os.path.realpath(os.path.dirname(subshard["input_path"])),
            stream=sidecar_stream, signature=sidecar_signature,
        )
    return iter(range(start, stop))


def _validate_input_row(
    payload: bytes,
    expected_ordinal: int,
    expected_dataset_id: str | None,
) -> Mapping[str, Any]:
    item = _decode_json_line(payload, "MOS worker input row")
    if not isinstance(item, dict) or set(item) != {"ordinal", "row"}:
        raise ValueError("input shard row schema is invalid")
    ordinal, row = item["ordinal"], item["row"]
    if type(ordinal) is not int or not isinstance(row, Mapping):
        raise ValueError("input shard row values are invalid")
    if ordinal != expected_ordinal:
        raise ValueError("input shard ordinals are not exactly the assigned ordinal assignment")
    dataset_id = row.get("dataset_id")
    Step3MosSelectionStage._validate_upstream_row(row, dataset_id)
    if expected_dataset_id is not None and dataset_id != expected_dataset_id:
        raise ValueError("worker request row dataset binding is invalid")
    if type(row["channels"]) is not int or row["channels"] <= 0:
        raise ValueError("Step2 upstream row channels must be a positive Python integer")
    return row


def _prevalidate_subshard_input(
    subshard: Mapping[str, Any],
    expected_dataset_id: str | None,
    ordinal_registry: DiskBackedOrdinalRegistry | None = None,
) -> tuple[
    Any, tuple[int, int, int, int, int],
    Any | None, tuple[int, int, int, int, int] | None,
]:
    path = subshard["input_path"]
    input_stream, initial = _open_regular_stream(path, "MOS worker input shard")
    input_signature = _file_signature(initial)
    sidecar_stream = None
    sidecar_signature = None
    payloads = None
    expected_ordinals = None
    keep_streams = False
    try:
        if _input_path_signature(path) != input_signature:
            raise ValueError("MOS worker input shard pathname changed before validation")
        if _ORDINAL_SIDECAR_FIELDS <= set(subshard):
            sidecar_stream, sidecar_initial = _open_regular_stream(
                subshard["ordinal_sidecar_path"], "MOS sparse ordinal sidecar",
            )
            sidecar_signature = _file_signature(sidecar_initial)
        payloads = iter(_iter_bounded_jsonl(input_stream, "MOS worker input shard"))
        expected_ordinals = _expected_ordinal_iterator(
            subshard, sidecar_stream=sidecar_stream,
            sidecar_signature=sidecar_signature,
        )
        digest = hashlib.sha256()
        count = 0
        for payload in payloads:
            digest.update(payload)
            try:
                expected_ordinal = next(expected_ordinals)
            except StopIteration:
                raise ValueError(
                    "input shard contains more rows than the assigned ordinal assignment"
                ) from None
            _validate_input_row(payload, expected_ordinal, expected_dataset_id)
            if ordinal_registry is not None:
                ordinal_registry.add(expected_ordinal)
            count += 1
        try:
            next(expected_ordinals)
        except StopIteration:
            pass
        else:
            raise ValueError("input shard is missing rows from the assigned ordinal assignment")
        if count != subshard["row_count"]:
            raise ValueError(
                "input shard row count does not match the assigned ordinal assignment"
            )
        if digest.hexdigest() != subshard["input_sha256"]:
            raise ValueError("MOS worker input shard digest mismatch")
        if _file_signature(os.fstat(input_stream.fileno())) != input_signature:
            raise ValueError("MOS worker input shard changed during validation")
        if _input_path_signature(path) != input_signature:
            raise ValueError("MOS worker input shard pathname changed during validation")
        input_stream.seek(0)
        if sidecar_stream is not None:
            if _file_signature(os.fstat(sidecar_stream.fileno())) != sidecar_signature:
                raise ValueError("MOS sparse ordinal sidecar changed during validation")
            if _path_signature(
                subshard["ordinal_sidecar_path"], "MOS sparse ordinal sidecar",
            ) != sidecar_signature:
                raise ValueError(
                    "MOS sparse ordinal sidecar pathname changed during validation"
                )
            sidecar_stream.seek(0)
        keep_streams = True
        return input_stream, input_signature, sidecar_stream, sidecar_signature
    finally:
        primary = sys.exc_info()[1]
        cleanup_error = None
        for iterator, label in (
            (expected_ordinals, "MOS input ordinal iterator"),
            (payloads, "MOS input validation iterator"),
        ):
            close = getattr(iterator, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:
                    cleanup_error = _record_cleanup_error(cleanup_error, exc, label)
        if not keep_streams or cleanup_error is not None:
            streams = tuple(
                stream for stream in (input_stream, sidecar_stream) if stream is not None
            )
            try:
                _finalize_streams(
                    streams,
                    None if isinstance(primary, GeneratorExit) else primary or cleanup_error,
                    sync=False,
                )
            except BaseException as exc:
                cleanup_error = _record_cleanup_error(
                    cleanup_error, exc, "MOS worker prevalidation streams",
                )
        if isinstance(primary, GeneratorExit) and cleanup_error is not None:
            raise cleanup_error
        if primary is not None:
            if cleanup_error is not None:
                _add_cleanup_note(primary, "MOS worker input prevalidation", cleanup_error)
        elif cleanup_error is not None:
            raise cleanup_error


def _verify_validated_subshard(
    subshard: Mapping[str, Any],
    validated: tuple[
        Any, tuple[int, int, int, int, int],
        Any | None, tuple[int, int, int, int, int] | None,
    ],
) -> None:
    input_stream, input_signature, sidecar_stream, sidecar_signature = validated
    if _file_signature(os.fstat(input_stream.fileno())) != input_signature:
        raise ValueError("MOS worker input shard changed after validation")
    if _input_path_signature(subshard["input_path"]) != input_signature:
        raise ValueError("MOS worker input shard pathname changed after validation")
    has_sidecar = _ORDINAL_SIDECAR_FIELDS <= set(subshard)
    if has_sidecar != (sidecar_stream is not None and sidecar_signature is not None):
        raise RuntimeError("MOS sparse ordinal sidecar ownership is inconsistent")
    if has_sidecar:
        if _file_signature(os.fstat(sidecar_stream.fileno())) != sidecar_signature:
            raise ValueError("MOS sparse ordinal sidecar changed after validation")
        if _path_signature(
            subshard["ordinal_sidecar_path"], "MOS sparse ordinal sidecar",
        ) != sidecar_signature:
            raise ValueError(
                "MOS sparse ordinal sidecar pathname changed after validation"
            )


def _score_subshard(
    runtime: Any,
    subshard: Mapping[str, Any],
    expected_dataset_id: str | None = None,
    *,
    input_stream: Any | None = None,
    input_signature: tuple[int, int, int, int, int] | None = None,
    sidecar_stream: Any | None = None,
    sidecar_signature: tuple[int, int, int, int, int] | None = None,
) -> None:
    """独立处理一个已验证模型 tuple 子分片，绝不跨 tuple 混批。"""
    owns_inputs = input_stream is None
    models = normalize_selectable_mos_models(subshard["models"])
    expected_ordinals = None
    ordinal_index = 0
    output_bytes = 0
    target = None
    payloads = None
    coordinator = runtime.open_waveform_coordinator(models)

    def write_completed(
        completed: Sequence[tuple[Any, Mapping[str, Any]]], target_stream: Any,
    ) -> None:
        nonlocal output_bytes
        if not completed:
            return
        for ordinal, raw_scores in completed:
            scores = Step3MosSelectionStage._validate_scores(raw_scores, models)
            payload = _encode_json_line(
                {"ordinal": ordinal, "scores": scores}, "MOS worker result row",
            )
            output_bytes += len(payload)
            if output_bytes > _JSONL_MAX_FILE_BYTES:
                raise ValueError(
                    f"MOS worker result exceeds JSONL file limit {_JSONL_MAX_FILE_BYTES} bytes"
                )
            target_stream.write(payload)
        target_stream.flush()
        _emit_progress(len(completed))

    try:
        if input_stream is None:
            (
                input_stream, input_signature,
                sidecar_stream, sidecar_signature,
            ) = _prevalidate_subshard_input(subshard, expected_dataset_id)
        if input_signature is None:
            raise RuntimeError("MOS worker input signature is unavailable")
        validated = (
            input_stream, input_signature, sidecar_stream, sidecar_signature,
        )
        _verify_validated_subshard(subshard, validated)
        expected_ordinals = _expected_ordinal_iterator(
            subshard, sidecar_stream=sidecar_stream,
            sidecar_signature=sidecar_signature,
        )
        target = _open_new_regular_stream(
            subshard["output_path"], "MOS worker result output",
        )
        payloads = iter(_iter_bounded_jsonl(input_stream, "MOS worker input shard"))
        for payload in payloads:
            try:
                expected_ordinal = next(expected_ordinals)
            except StopIteration:
                raise ValueError(
                    "input shard contains more rows than the assigned ordinal assignment"
                ) from None
            row = _validate_input_row(payload, expected_ordinal, expected_dataset_id)
            write_completed(coordinator.push(row, expected_ordinal), target)
            ordinal_index += 1
        try:
            next(expected_ordinals)
        except StopIteration:
            pass
        else:
            raise ValueError("input shard is missing rows from the assigned ordinal assignment")
        write_completed(coordinator.finish(), target)
        if ordinal_index != subshard["row_count"]:
            raise ValueError(
                "input shard row count does not match the assigned ordinal assignment"
            )
        _verify_validated_subshard(subshard, validated)
    finally:
        primary = sys.exc_info()[1]
        cleanup_error = None
        try:
            coordinator.abort(*sys.exc_info())
        except BaseException as exc:
            cleanup_error = _record_cleanup_error(
                cleanup_error, exc, "MOS waveform coordinator",
            )
        for iterator, label in (
            (expected_ordinals, "MOS sparse ordinal sidecar iterator"),
            (payloads, "MOS worker input scoring iterator"),
        ):
            close = getattr(iterator, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:
                    cleanup_error = _record_cleanup_error(cleanup_error, exc, label)
        closing_generator = isinstance(primary, GeneratorExit)
        if target is not None:
            try:
                _finalize_streams(
                    (target,),
                    None if closing_generator else primary or cleanup_error,
                    sync=True,
                )
            except BaseException as exc:
                cleanup_error = _record_cleanup_error(
                    cleanup_error, exc, "MOS worker result output",
                )
        if owns_inputs and input_stream is not None:
            streams = tuple(
                stream for stream in (input_stream, sidecar_stream) if stream is not None
            )
            try:
                _finalize_streams(
                    streams,
                    None if closing_generator else primary or cleanup_error,
                    sync=False,
                )
            except BaseException as exc:
                cleanup_error = _record_cleanup_error(
                    cleanup_error, exc, "MOS worker scoring input streams",
                )
        if closing_generator and cleanup_error is not None:
            raise cleanup_error
        if primary is not None:
            if cleanup_error is not None:
                _add_cleanup_note(primary, "MOS worker cleanup", cleanup_error)
        elif cleanup_error is not None:
            raise cleanup_error


def handle_request(
    request_path: str, *, expected_dataset_id: str | None = None,
) -> None:
    """经禁止 fallback 的 canonical runtime 评分，全部成功后发布证明。

    所选模型均走 ``Step3MosRuntime`` 的 ``allow_fallback=False`` 路径；任一错误
    均阻止 provider attestation sidecar 发布。
    """
    if expected_dataset_id is not None and (
        type(expected_dataset_id) is not str or not expected_dataset_id.strip()
    ):
        raise ValueError("expected_dataset_id must be a nonblank string or None")
    request = _load_request(request_path)
    validated_inputs: list[tuple[
        Any, tuple[int, int, int, int, int],
        Any | None, tuple[int, int, int, int, int] | None,
    ]] = []
    ordinal_registry = None
    ordinal_directory = None
    try:
        ordinal_directory = tempfile.TemporaryDirectory(
            prefix="audio-filter-step3-worker-ordinals-",
            dir=request["private_dir"],
        )
        os.chmod(ordinal_directory.name, 0o700)
        ordinal_registry = DiskBackedOrdinalRegistry(
            os.path.join(ordinal_directory.name, "ordinals.sqlite3")
        )
        expected_ordinal_count = 0
        for subshard in request["subshards"]:
            validated_inputs.append(_prevalidate_subshard_input(
                subshard, expected_dataset_id, ordinal_registry,
            ))
            expected_ordinal_count += subshard["row_count"]
        ordinal_registry.finalize(expected_ordinal_count)
        for subshard, validated in zip(
            request["subshards"], validated_inputs, strict=True,
        ):
            _verify_validated_subshard(subshard, validated)
        runtime = Step3MosRuntime(options=MosRuntimeOptions(
            use_gpu=True,
            urgent_options=request["urgent_options"],
            scoreq_options=request["scoreq_options"],
            dnsmos_options=request["dnsmos_options"],
            nisqa_options=request["nisqa_options"],
            utmos_options=request["utmos_options"],
        ))
        attestations = []
        seen_models = set()
        worker_models = []
        for subshard, validated in zip(
            request["subshards"], validated_inputs, strict=True,
        ):
            input_stream, input_signature, sidecar_stream, sidecar_signature = validated
            models = normalize_selectable_mos_models(subshard["models"])
            for model in models:
                if model not in worker_models:
                    worker_models.append(model)
            if models not in seen_models:
                attestations.append({
                    "models": list(models),
                    "attestation": runtime.runtime_attestation(models),
                })
                seen_models.add(models)
            _score_subshard(
                runtime,
                subshard,
                expected_dataset_id,
                input_stream=input_stream,
                input_signature=input_signature,
                sidecar_stream=sidecar_stream,
                sidecar_signature=sidecar_signature,
            )
        _write_runtime_attestation(
            request["attestation_path"], tuple(worker_models), attestations,
        )
    finally:
        primary = sys.exc_info()[1]
        cleanup_error = None
        streams = tuple(
            stream
            for validated in validated_inputs
            for stream in (validated[0], validated[2])
            if stream is not None
        )
        closing_generator = isinstance(primary, GeneratorExit)
        try:
            _finalize_streams(
                streams,
                None if closing_generator else primary or cleanup_error,
                sync=False,
            )
        except BaseException as exc:
            cleanup_error = _record_cleanup_error(
                cleanup_error, exc, "MOS prevalidated input streams",
            )
        if ordinal_registry is not None:
            try:
                ordinal_registry.close()
            except BaseException as exc:
                cleanup_error = _record_cleanup_error(
                    cleanup_error, exc, "MOS worker ordinal registry",
                )
        if ordinal_directory is not None:
            try:
                ordinal_directory.cleanup()
            except BaseException as exc:
                cleanup_error = _record_cleanup_error(
                    cleanup_error, exc, "MOS worker ordinal directory",
                )
        if closing_generator and cleanup_error is not None:
            raise cleanup_error
        if primary is not None:
            if cleanup_error is not None:
                _add_cleanup_note(primary, "MOS worker input cleanup", cleanup_error)
        elif cleanup_error is not None:
            raise cleanup_error


_HEARTBEAT = b"audio_filter_pipeline.step3_mos_worker.heartbeat.v1\n"
_PROGRESS_PREFIX = b"audio_filter_pipeline.step3_mos_worker.progress.v1:"
_FLEET_READY = b"audio_filter_pipeline.step3_mos_worker.fleet.ready.v1\n"
_FLEET_DONE = b"audio_filter_pipeline.step3_mos_worker.fleet.done.v1\n"
_FLEET_MAX_CONTROL_BYTES = 32 * 1024
_STDOUT_LOCK = threading.Lock()


def _emit_progress(completed_rows: int) -> None:
    if type(completed_rows) is not int or completed_rows <= 0:
        raise ValueError("worker progress rows must be a positive integer")
    with _STDOUT_LOCK:
        sys.stdout.buffer.write(_PROGRESS_PREFIX + str(completed_rows).encode("ascii") + b"\n")
        sys.stdout.buffer.flush()


def _run_with_heartbeat(callback: Callable[[], None]) -> None:
    """仅在一次请求执行期间发 heartbeat，并在返回前彻底停止。"""
    interval = float(os.environ.get("AUDIO_FILTER_STEP3_HEARTBEAT_SECONDS", "30"))
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError("worker heartbeat interval must be a positive finite number")
    output = sys.stdout.buffer
    stopped = threading.Event()

    def heartbeat() -> None:
        while not stopped.wait(interval):
            try:
                with _STDOUT_LOCK:
                    output.write(_HEARTBEAT)
                    output.flush()
            except OSError:
                return

    thread = threading.Thread(
        target=heartbeat, name="step3-mos-worker-heartbeat", daemon=True,
    )
    thread.start()
    try:
        callback()
    finally:
        stopped.set()
        thread.join()


def serve_requests() -> None:
    """顺序处理 parent 请求；任一失败即退出，正常关闭时清空模型 cache。"""
    output = sys.stdout.buffer
    output.write(_FLEET_READY)
    output.flush()
    try:
        while True:
            payload = sys.stdin.buffer.readline(_FLEET_MAX_CONTROL_BYTES + 1)
            if not payload:
                return
            if len(payload) > _FLEET_MAX_CONTROL_BYTES or not payload.endswith(b"\n"):
                raise ValueError("fleet request control exceeds framing limits")
            control = _decode_json_line(payload, "fleet request control")
            if not isinstance(control, dict) or set(control) != {
                "dataset_id", "request_path",
            }:
                raise ValueError("fleet request control schema is invalid")
            dataset_id, request_path = control["dataset_id"], control["request_path"]
            if (
                type(dataset_id) is not str or not dataset_id.strip()
                or type(request_path) is not str or not request_path
                or "\x00" in request_path
            ):
                raise ValueError("fleet request control values are invalid")
            _run_with_heartbeat(lambda: handle_request(
                request_path, expected_dataset_id=dataset_id,
            ))
            output.write(_FLEET_DONE)
            output.flush()
    finally:
        clear_model_cache()


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments == ["--serve"]:
        serve_requests()
        return 0
    if len(arguments) != 1:
        raise ValueError("expected one request path argument or --serve")
    _run_with_heartbeat(lambda: handle_request(arguments[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
