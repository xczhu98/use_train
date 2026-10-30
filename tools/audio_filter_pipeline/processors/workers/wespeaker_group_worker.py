"""单 GPU Step5 WeSpeaker frozen-enrollment child，stdin/stdout 仅使用严格 JSONL。"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
from collections.abc import Mapping, Sequence

from audio_filter_pipeline.processors.stages_v3.wespeaker_runtime import (
    WeSpeakerEmbeddingRuntime,
    WeSpeakerRuntimeError,
    WeSpeakerRuntimeOptions,
)

_REQUEST_SCHEMA = "audio_filter_pipeline.wespeaker_group_request.v3"
_RESULT_SCHEMA = "audio_filter_pipeline.wespeaker_group_result.v3"
_ALGORITHM = "l2_normalized_segment_embedding_to_frozen_enrollment_cosine_v1"
_REQUEST_MAX_BYTES = 32 * 1024 * 1024
_WORKER_ERROR_MAX_CHARS = 4_096


@contextlib.contextmanager
def _discard_stdout():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _report_worker_error(exc: BaseException) -> None:
    """把 worker 根因写到 stderr；父进程只保留 stderr 尾部，消息必须有界。"""
    message = f"{type(exc).__name__}: {exc}"[:_WORKER_ERROR_MAX_CHARS]
    try:
        sys.stderr.write(f"worker failed: {message}\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        # stderr 不可写（父进程已关闭管道）时仍以 exit code 2 表达失败。
        pass


def _normalize(values: Sequence[object]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("embedding must be a sequence")
    vector = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("embedding must be finite")
        vector.append(float(value))
    if not vector:
        raise ValueError("embedding must be nonzero")
    scale = max(abs(value) for value in vector)
    if scale == 0.0:
        raise ValueError("embedding must be nonzero")
    scaled = tuple(value / scale for value in vector)
    norm = math.sqrt(math.fsum(value * value for value in scaled))
    return tuple(value / norm for value in scaled)


def _result(group_key: list[str], segment_id: str, bundle_sha256: str, shadow: Mapping[str, object]) -> dict[str, object]:
    return {"schema": _RESULT_SCHEMA, "group_key": group_key, "segment_id": segment_id, "enrollment_bundle_sha256": bundle_sha256, "shadow": dict(shadow)}


def _handle(request: object) -> list[dict[str, object]]:
    if not isinstance(request, Mapping) or set(request) != {"schema", "options", "groups", "enrollment_bundle_sha256"}:
        raise ValueError("invalid request schema")
    bundle_sha256 = request["enrollment_bundle_sha256"]
    if (request["schema"] != _REQUEST_SCHEMA or not isinstance(request["options"], Mapping)
            or type(bundle_sha256) is not str or len(bundle_sha256) != 64
            or any(char not in "0123456789abcdef" for char in bundle_sha256)):
        raise ValueError("invalid request binding")
    options, groups = WeSpeakerRuntimeOptions.from_payload(request["options"]), request["groups"]
    if not isinstance(groups, list):
        raise ValueError("groups must be a list")
    parsed_groups = []
    seen_segment_ids = set()
    for group in groups:
        if not isinstance(group, Mapping) or set(group) != {"key", "segments", "enrollment_embedding", "enrollment_bundle_sha256"}:
            raise ValueError("invalid group schema")
        key, segments = group["key"], group["segments"]
        if (group["enrollment_bundle_sha256"] != bundle_sha256 or not isinstance(key, list) or len(key) != 3
                or any(type(value) is not str for value in key) or not isinstance(segments, list) or not segments):
            raise ValueError("invalid group values")
        enrollment = _normalize(group["enrollment_embedding"])
        segment_ids = [row.get("segment_id") for row in segments if isinstance(row, Mapping)]
        if len(segment_ids) != len(segments) or any(type(value) is not str or not value for value in segment_ids):
            raise ValueError("invalid segment identity")
        if (len(set(segment_ids)) != len(segment_ids)
                or any(segment_id in seen_segment_ids for segment_id in segment_ids)):
            raise ValueError("duplicate segment identity in full request")
        seen_segment_ids.update(segment_ids)
        parsed_groups.append((key, segments, enrollment, segment_ids))
    runtime, output = WeSpeakerEmbeddingRuntime(options), []
    provenance = {**runtime.provenance, "inference_performed": True, "algorithm": _ALGORITHM, "enrollment_bundle_sha256": bundle_sha256}
    try:
        # 每个 shard 只启动一个 child；child 内持久复用同一受限模型会话，
        # 每次最多 32 条，避免物化全部 embedding。
        with runtime.session() as session:
            for key, segments, enrollment, segment_ids in parsed_groups:
                for start in range(0, len(segments), 32):
                    segment_chunk = segments[start:start + 32]
                    with _discard_stdout():
                        embeddings = session.extract_batch(segment_chunk)
                    if len(embeddings) != len(segment_chunk):
                        raise ValueError("invalid embedding result count")
                    vectors = [_normalize(embedding) for embedding in embeddings]
                    if any(len(vector) != len(enrollment) for vector in vectors):
                        raise ValueError("invalid embedding result dimension")
                    for segment_id, vector in zip(
                        segment_ids[start:start + 32], vectors, strict=True,
                    ):
                        output.append(_result(key, segment_id, bundle_sha256, {"status": "computed", "cosine_similarity_to_enrollment": math.fsum(left * right for left, right in zip(vector, enrollment)), "provenance": provenance}))
    except WeSpeakerRuntimeError as exc:
        # shard 采用全有或全无语义：后段失败时丢弃此前 computed 结果，
        # 再为每个 segment 生成唯一 unavailable 结果。
        output.clear()
        unavailable = {"reason": "embedding_runtime_unavailable", "inference_performed": False, "error_type": type(exc).__name__, "enrollment_bundle_sha256": bundle_sha256}
        for key, _segments, _enrollment, segment_ids in parsed_groups:
            for segment_id in segment_ids:
                output.append(_result(key, segment_id, bundle_sha256, {"status": "unavailable", "provenance": unavailable}))
    return output


def main() -> int:
    try:
        raw = sys.stdin.buffer.readline(_REQUEST_MAX_BYTES + 1)
        if not raw or len(raw) > _REQUEST_MAX_BYTES or not raw.endswith(b"\n") or b"\n" in raw[:-1]:
            raise ValueError("invalid JSONL framing")
        for result in _handle(json.loads(raw[:-1].decode("utf-8", "strict"))):
            sys.stdout.buffer.write(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n")
        return 0
    except Exception as exc:
        _report_worker_error(exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
