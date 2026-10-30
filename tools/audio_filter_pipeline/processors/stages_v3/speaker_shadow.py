"""Generation-3 Step5 frozen-enrollment speaker verification."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from collections.abc import Callable, Iterable, Mapping, Sequence
from numbers import Real
from typing import Any, Optional

from tqdm import tqdm

from ...config import PipelineConfig
from ..mos import normalize_selectable_mos_models
from ...utils.artifacts import ArtifactIndex, PipelineStep
from ...utils.strict_jsonl import decode_json_line
from .base import ArtifactStage, StageRunContext
from .enrollment_bundle import enrollment_exclusion_manifest_algorithm
from .export import _validate_dataset_and_source_identity
from .segment import STEP2_ROW_FIELDS
from .select import STEP3_ROW_FIELDS, Step3MosSelectionStage
from .wespeaker_runtime import (
    _DEFAULT_LIMITS,
    BoundedSubprocessError,
    BoundedSubprocessOutputLimitError,
    BoundedSubprocessTimeoutError,
    WeSpeakerRuntimeError,
    run_bounded_subprocess,
    runtime_options_payload,
    terminate_process_group,
)


EmbeddingExtractor = Callable[[Mapping[str, Any]], Sequence[Real]]
logger = logging.getLogger(__name__)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ALGORITHM = "l2_normalized_segment_embedding_to_frozen_enrollment_cosine_v1"
_GROUP_REQUEST_SCHEMA = "audio_filter_pipeline.wespeaker_group_request.v3"
_GROUP_RESULT_SCHEMA = "audio_filter_pipeline.wespeaker_group_result.v3"
_GROUP_WORKER_PROTOCOL = "audio_filter_pipeline.wespeaker_group_jsonl.v3"
STEP5_ROW_FIELDS = STEP3_ROW_FIELDS + ("speaker_shadow",)
_BUNDLE_MAX_BYTES = 32 * 1024 * 1024
_GROUP_REQUEST_MAX_BYTES = 24 * 1024 * 1024
_RUNTIME_OPTION_FIELDS = frozenset({
    "model_path", "model_version", "model_sha256", "runtime_version",
    "target_sample_rate", "python_executable", "trusted_code_root",
    "worker_script", "worker_script_sha256", "worker_limits", "asset_limits",
    "cuda_visible_devices",
})
_POOL_PROVENANCE_OPTION_FIELDS = (
    "model_path", "model_version", "model_sha256", "runtime_version",
    "target_sample_rate", "worker_script", "worker_script_sha256",
)


def _extractor_provenance(extractor: object) -> Mapping[str, Any]:
    """复制可序列化的稳定 extractor 身份；缺失时拒绝复用缓存。"""
    provenance = getattr(extractor, "provenance", None)
    if not isinstance(provenance, Mapping):
        raise ValueError("embedding extractor requires stable provenance")
    stable = dict(provenance)
    capability = stable.get("capability")
    if type(capability) is not str or not capability or capability != capability.strip():
        raise ValueError("embedding extractor provenance capability is invalid")
    try:
        encoded = json.dumps(
            stable, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8") + b"\n"
        decoded = decode_json_line(
            encoded, "embedding extractor provenance", max_line_bytes=1_048_576,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding extractor provenance must be stable JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("embedding extractor provenance must be a JSON object")
    return decoded


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError("cannot hash frozen WeSpeaker source") from exc
    return digest.hexdigest()


def _frozen_wespeaker_sources(runtime_script: str) -> Mapping[str, Mapping[str, str]]:
    stages_dir = os.path.dirname(__file__)
    sources = {
        "group_script": os.path.abspath(os.path.join(
            stages_dir, "..", "workers", "wespeaker_group_worker.py",
        )),
        "runtime_script": runtime_script,
        "runtime_module": os.path.abspath(os.path.join(stages_dir, "wespeaker_runtime.py")),
        "optional_runtime_module": os.path.abspath(os.path.join(
            stages_dir, "..", "optional_runtime.py",
        )),
    }
    return {
        name: {"path": path, "sha256": _file_sha256(path)}
        for name, path in sources.items()
    }


def _validate_step3_row(
    row: Mapping[str, Any],
    dataset_id: str,
    config: PipelineConfig | None = None,
) -> None:
    """校验 Step3 的完整 immutable row contract 后再进入声纹分组。"""
    if not isinstance(row, Mapping):
        raise ValueError("Step3 upstream row must be a mapping")
    if set(row) != set(STEP3_ROW_FIELDS):
        raise ValueError("Step3 upstream row fields do not match the exact contract")

    step2_row = {field: row[field] for field in STEP2_ROW_FIELDS}
    Step3MosSelectionStage._validate_upstream_row(step2_row, dataset_id)
    _validate_dataset_and_source_identity(
        row,
        dataset_id,
        config,
        label="Step3 upstream row",
        require_original_relative_path=True,
    )
    models = normalize_selectable_mos_models(row["mos_models"])
    Step3MosSelectionStage._validate_scores(row["mos_scores"], models)
    if config is not None:
        policy = config.resolve_selection_policy(
            row["dataset_id"],
            row["language"],
            source_relative_path=row["source_relative_path"],
        )
        projection = config.mos_scoring_projection(
            row["dataset_id"],
            row["language"],
            source_relative_path=row["source_relative_path"],
        )
        if (
            models != policy.mos_models
            or row["mos_scoring_policy_hash"] != projection["scoring_policy_hash"]
        ):
            raise ValueError("Step3 upstream row MOS scoring projection is stale")

    speaker_id, speaker_scope = row["speaker_id"], row["speaker_scope"]
    unknown_speaker = speaker_id is None and speaker_scope is None
    explicit_speaker = (
        type(speaker_id) is str
        and bool(speaker_id.strip())
        and speaker_id == speaker_id.strip()
        and type(speaker_scope) is str
        and speaker_scope == dataset_id
    )
    if not (unknown_speaker or explicit_speaker):
        raise ValueError("Step3 upstream row speaker identity is invalid")


def _embedding_vector(values: Sequence[Real], *, field: str = "embedding") -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{field} must be a numeric sequence")
    vector = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field} must contain finite non-bool real values")
        try:
            converted = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{field} must contain finite non-bool real values") from exc
        if not math.isfinite(converted):
            raise ValueError(f"{field} must contain finite non-bool real values")
        vector.append(converted)
    if not vector or not any(value != 0.0 for value in vector):
        raise ValueError(f"{field} must be nonzero")
    return tuple(vector)


def _l2_normalize(values: Sequence[float]) -> tuple[float, ...]:
    """使用缩放后的平方和，避免有限大值在平方时溢出。"""
    if not values:
        raise ValueError("speaker consistency vector must be nonzero")
    scale = max(abs(value) for value in values)
    if not math.isfinite(scale) or scale == 0.0:
        raise ValueError("speaker consistency vector must be nonzero")
    scaled = tuple(value / scale for value in values)
    scaled_norm = math.sqrt(math.fsum(value * value for value in scaled))
    if not math.isfinite(scaled_norm) or scaled_norm == 0.0:
        raise ValueError("speaker consistency vector norm must be finite and nonzero")
    normalized = tuple(value / scaled_norm for value in scaled)
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("speaker consistency vector norm must be finite")
    return normalized


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("speaker consistency vectors must have matching dimensions")
    try:
        score = math.fsum(a * b for a, b in zip(left, right))
    except (OverflowError, ValueError) as exc:
        raise ValueError("speaker consistency cosine must be finite") from exc
    if not math.isfinite(score):
        raise ValueError("speaker consistency cosine must be finite")
    return score


def _shadow(status: str, reason: str, **provenance: Any) -> Mapping[str, Any]:
    return {"status": status, "provenance": {"reason": reason, "inference_performed": False, **provenance}}


def _load_enrollment_bundle(policy: Mapping[str, Any]) -> Mapping[tuple[str, str, str], Mapping[str, Any]]:
    """加载并绑定冻结 JSONL enrollment embedding bundle，任何歧义均拒绝。"""
    path, expected_sha256 = policy["enrollment_bundle_path"], policy["enrollment_bundle_sha256"]
    try:
        with open(path, "rb") as stream:
            raw = stream.read(_BUNDLE_MAX_BYTES + 1)
    except OSError as exc:
        raise ValueError("cannot read enrollment bundle") from exc
    if len(raw) > _BUNDLE_MAX_BYTES:
        raise ValueError("enrollment bundle exceeds size limit")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("enrollment bundle SHA-256 does not match configuration")
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("enrollment bundle must be nonempty newline-terminated JSONL")
    expected_provenance = {field: policy[field] for field in (
        "runtime", "model_version", "model_sha256", "runtime_version", "target_sample_rate",
    )}
    bundle = {}
    for number, line in enumerate(raw.splitlines(keepends=True), 1):
        entry = decode_json_line(
            line, f"enrollment bundle line {number}",
            max_line_bytes=_BUNDLE_MAX_BYTES,
        )
        if not isinstance(entry, Mapping) or set(entry) != {"dataset_id", "speaker_scope", "speaker_id", "embedding", "provenance"}:
            raise ValueError(f"enrollment bundle line {number} has invalid schema")
        key = (entry["dataset_id"], entry["speaker_scope"], entry["speaker_id"])
        if any(type(value) is not str or not value or value != value.strip() for value in key):
            raise ValueError(f"enrollment bundle line {number} has invalid identity")
        if key[0] != key[1]:
            raise ValueError(f"enrollment bundle line {number} speaker_scope must equal dataset_id")
        if key in bundle:
            raise ValueError("enrollment bundle contains duplicate speaker identity")
        if entry["provenance"] != expected_provenance:
            raise ValueError(f"enrollment bundle line {number} provenance does not match speaker_consistency")
        bundle[key] = {"embedding": _l2_normalize(_embedding_vector(entry["embedding"], field="enrollment embedding")),
                       "provenance": dict(entry["provenance"])}
    return bundle


def _load_enrollment_exclusions(
    policy: Mapping[str, Any],
    bundle: Mapping[tuple[str, str, str], Mapping[str, Any]],
    upstream_index: ArtifactIndex,
) -> Mapping[str, tuple[str, str, str]]:
    """加载并绑定 bundle 与已发布 Step3 的 enrollment exclusion manifest。"""
    path, expected_sha256 = policy["enrollment_exclusion_path"], policy["enrollment_exclusion_sha256"]
    try:
        with open(path, "rb") as stream:
            raw = stream.read(_BUNDLE_MAX_BYTES + 1)
    except OSError as exc:
        raise ValueError("cannot read enrollment exclusion manifest") from exc
    if len(raw) > _BUNDLE_MAX_BYTES or not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ValueError("enrollment exclusion manifest must be one bounded newline-terminated JSON object")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("enrollment exclusion manifest SHA-256 does not match configuration")
    manifest = decode_json_line(
        raw, "enrollment exclusion manifest", max_line_bytes=_BUNDLE_MAX_BYTES,
    )
    enrollment_exclusion_manifest_algorithm(manifest)
    if manifest["bundle_sha256"] != policy["enrollment_bundle_sha256"]:
        raise ValueError("enrollment exclusion manifest is not bound to enrollment bundle SHA-256")
    if manifest["provenance"] != {field: policy[field] for field in ("runtime", "model_version", "model_sha256", "runtime_version", "target_sample_rate")}:
        raise ValueError("enrollment exclusion manifest provenance does not match speaker_consistency")
    if not isinstance(manifest["step3"], Mapping) or not isinstance(manifest["enrollments"], list):
        raise ValueError("enrollment exclusion manifest values are invalid")
    expected_step3 = {
        "dataset_id": upstream_index.dataset_id,
        "config_digest": upstream_index.config_digest,
        "lineage_digest": upstream_index.lineage_digest,
        "artifact_digest": upstream_index.artifact_digest,
    }
    if manifest["step3"] != expected_step3:
        raise ValueError("enrollment exclusion manifest Step3 artifact binding does not match upstream")
    exclusions = {}
    manifest_keys = set()
    for enrollment in manifest["enrollments"]:
        if not isinstance(enrollment, Mapping) or set(enrollment) != {"dataset_id", "speaker_scope", "speaker_id", "segment_ids", "candidate_count", "stability"}:
            raise ValueError("enrollment exclusion manifest enrollment schema is invalid")
        key = (enrollment["dataset_id"], enrollment["speaker_scope"], enrollment["speaker_id"])
        if (key not in bundle or key in manifest_keys
                or any(type(value) is not str or not value or value != value.strip() for value in key)):
            raise ValueError("enrollment exclusion manifest identity is invalid")
        manifest_keys.add(key)
        segment_ids, candidate_count, stability = (
            enrollment["segment_ids"], enrollment["candidate_count"], enrollment["stability"],
        )
        if (not isinstance(segment_ids, list) or not segment_ids
                or type(candidate_count) is not int or candidate_count <= 0
                or candidate_count != len(segment_ids) or not isinstance(stability, Mapping)
                or set(stability) != set(segment_ids)):
            raise ValueError("enrollment exclusion manifest candidate set is invalid")
        for segment_id in segment_ids:
            score = stability[segment_id]
            if (type(segment_id) is not str or not segment_id or segment_id != segment_id.strip()
                    or segment_id in exclusions or isinstance(score, bool)
                    or not isinstance(score, Real) or not math.isfinite(float(score))):
                raise ValueError("enrollment exclusion manifest contains invalid or duplicate segment ID")
            exclusions[segment_id] = key
    if manifest_keys != set(bundle):
        raise ValueError("enrollment exclusion manifest identities do not exactly match bundle")
    return exclusions


class SpeakerGroupExtractorPool:
    """按 speaker group 总帧数做 LPT 分片，并为每个非空 GPU shard 启动一个 child。"""

    def __init__(
        self,
        options: Mapping[str, Any],
        gpu_ids: Sequence[int],
        *,
        worker_timeout_seconds: float = 900.0,
        worker_stderr_tail_bytes: int = 4_096,
        popen: Callable[..., Any] | None = None,
    ) -> None:
        if (isinstance(gpu_ids, (str, bytes)) or not gpu_ids
                or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in gpu_ids)
                or len(set(gpu_ids)) != len(gpu_ids)):
            raise ValueError("speaker group pool requires distinct nonnegative GPU ids")
        if not isinstance(options, Mapping):
            raise TypeError("speaker group pool options must be a mapping")
        if (isinstance(worker_timeout_seconds, bool)
                or not isinstance(worker_timeout_seconds, (int, float))
                or not math.isfinite(float(worker_timeout_seconds))
                or worker_timeout_seconds <= 0):
            raise ValueError("speaker group pool timeout must be positive and finite")
        if type(worker_stderr_tail_bytes) is not int or worker_stderr_tail_bytes <= 0:
            raise ValueError("speaker group pool stderr tail size must be positive")
        self._options, self._gpu_ids = dict(options), tuple(gpu_ids)
        for field in ("model_path", "worker_script"):
            value = self._options.get(field)
            if (type(value) is not str or not value or not os.path.isabs(value)
                    or value != os.path.normpath(os.path.abspath(value))):
                raise ValueError(f"speaker group pool {field} must be a canonical absolute path")
        for field in ("model_version", "runtime_version"):
            value = self._options.get(field)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"speaker group pool {field} must be a nonblank exact string")
        for field in ("model_sha256", "worker_script_sha256"):
            if _SHA256_RE.fullmatch(self._options.get(field, "")) is None:
                raise ValueError(f"speaker group pool {field} must be lowercase SHA-256 hex")
        target_sample_rate = self._options.get("target_sample_rate")
        if type(target_sample_rate) is not int or target_sample_rate <= 0:
            raise ValueError("speaker group pool target_sample_rate must be positive")
        self._worker_timeout_seconds = float(worker_timeout_seconds)
        self._worker_stderr_tail_bytes = worker_stderr_tail_bytes
        self._popen = subprocess.Popen if popen is None else popen
        if not callable(self._popen):
            raise TypeError("speaker group pool popen must be callable")
        self._processes, self._processes_lock = set(), threading.Lock()
        runtime_identity = {
            field: self._options.get(field)
            for field in _POOL_PROVENANCE_OPTION_FIELDS
        }
        self._frozen_sources = _frozen_wespeaker_sources(runtime_identity["worker_script"])
        self.provenance = {
            "capability": "wespeaker_optional_worker",
            "sharding": "lpt_complete_speaker_group_gpu_child_v3",
            "gpu_ids": list(self._gpu_ids),
            "model": {
                "path": runtime_identity["model_path"],
                "version": runtime_identity["model_version"],
                "sha256": runtime_identity["model_sha256"],
            },
            "runtime": {
                "name": "wespeaker",
                "version": runtime_identity["runtime_version"],
                "target_sample_rate": runtime_identity["target_sample_rate"],
            },
            "worker": {
                "protocol": _GROUP_WORKER_PROTOCOL,
                "request_schema": _GROUP_REQUEST_SCHEMA,
                "result_schema": _GROUP_RESULT_SCHEMA,
                **self._frozen_sources,
                "runtime_script_config_sha256": runtime_identity["worker_script_sha256"],
            },
        }

    def _verify_frozen_sources(self) -> None:
        for name, source in self._frozen_sources.items():
            if _file_sha256(source["path"]) != source["sha256"]:
                raise ValueError(f"frozen WeSpeaker {name} source changed before GPU child spawn")
        if self._frozen_sources["runtime_script"]["sha256"] != self._options["worker_script_sha256"]:
            raise ValueError("frozen WeSpeaker runtime worker source does not match configured SHA-256")

    @staticmethod
    def _terminate_process_group(proc: Any) -> None:
        terminate_process_group(proc, killpg=os.killpg)

    def _terminate_active_children(self) -> None:
        with self._processes_lock:
            processes = tuple(self._processes)
        for proc in processes:
            try:
                self._terminate_process_group(proc)
            except BaseException:
                # 单个 child 清理异常不能阻止其余 child 的终止尝试。
                pass
            try:
                exited = proc.poll() is not None
            except BaseException:
                exited = False
            if exited:
                with self._processes_lock:
                    self._processes.discard(proc)

    def _stderr_tail(self, stderr: bytes | None) -> str:
        tail = (stderr or b"")[-self._worker_stderr_tail_bytes:]
        return tail.decode("utf-8", "replace").strip()

    def _validate_shadow(self, shadow: object, bundle_sha256: str) -> Mapping[str, Any]:
        if not isinstance(shadow, Mapping) or type(shadow.get("status")) is not str:
            raise ValueError("GPU child result shadow is invalid")
        status, provenance = shadow["status"], shadow.get("provenance")
        if not isinstance(provenance, Mapping) or provenance.get("enrollment_bundle_sha256") != bundle_sha256:
            raise ValueError("GPU child result enrollment bundle SHA-256 mismatch")
        if status == "computed":
            score = shadow.get("cosine_similarity_to_enrollment")
            if (set(shadow) != {"status", "cosine_similarity_to_enrollment", "provenance"}
                    or isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(float(score))
                    or provenance.get("inference_performed") is not True
                    or provenance.get("algorithm") != _ALGORITHM):
                raise ValueError("GPU child computed result provenance is invalid")
        elif status == "unavailable":
            if (set(shadow) != {"status", "provenance"}
                    or provenance.get("inference_performed") is not False
                    or type(provenance.get("reason")) is not str):
                raise ValueError("GPU child unavailable result provenance is invalid")
        else:
            raise ValueError("GPU child result status is invalid")
        return dict(shadow)

    def extract_groups(
        self,
        groups: Mapping[tuple[str, str, str], Sequence[tuple[int, Mapping[str, Any]]]],
        enrollments: Mapping[tuple[str, str, str], Mapping[str, Any]],
        enrollment_bundle_sha256: str,
    ) -> Mapping[tuple[str, str, str], Mapping[str, Mapping[str, Any]]]:
        if _SHA256_RE.fullmatch(enrollment_bundle_sha256) is None:
            raise ValueError("enrollment bundle SHA-256 is invalid")
        if not isinstance(groups, Mapping) or not isinstance(enrollments, Mapping):
            raise TypeError("speaker groups and enrollments must be mappings")
        assignments: list[list[tuple[tuple[str, str, str], Sequence[tuple[int, Mapping[str, Any]]]]]] = [
            [] for _ in self._gpu_ids
        ]
        loads = [0] * len(self._gpu_ids)
        seen_segments: set[str] = set()
        weighted_groups = []
        group_order = {}
        expected_group_segments: dict[tuple[str, str, str], set[str]] = {}
        for order, (key, members) in enumerate(groups.items()):
            if (not isinstance(key, tuple) or len(key) != 3
                    or any(type(value) is not str or not value for value in key)):
                raise ValueError("speaker group identity is invalid")
            if key not in enrollments or not isinstance(members, Sequence) or not members:
                raise ValueError("speaker group enrollment or members are invalid")
            weight = 0
            group_segments = set()
            for member in members:
                if (not isinstance(member, tuple) or len(member) != 2
                        or type(member[0]) is not int or not isinstance(member[1], Mapping)):
                    raise ValueError("speaker group member is invalid")
                row = member[1]
                segment_id = row.get("segment_id")
                interval = (row.get("end_sample"), row.get("start_sample"))
                if (type(segment_id) is not str or not segment_id or segment_id in seen_segments
                        or any(type(value) is not int for value in interval)
                        or interval[0] <= interval[1]):
                    raise ValueError("speaker group segment identity or bounds are invalid")
                seen_segments.add(segment_id)
                group_segments.add(segment_id)
                weight += interval[0] - interval[1]
            weighted_groups.append((key, members, weight))
            group_order[key] = order
            expected_group_segments[key] = group_segments
        for key, members, weight in sorted(
            weighted_groups, key=lambda item: (-item[2], item[0]),
        ):
            shard = min(range(len(loads)), key=lambda index: (loads[index], index))
            assignments[shard].append((key, members))
            loads[shard] += weight
        for assigned in assignments:
            assigned.sort(key=lambda item: group_order[item[0]])

        def _json_bytes(value: object) -> bytes:
            return json.dumps(
                value, allow_nan=False, separators=(",", ":"),
            ).encode("utf-8")

        def _build_shard_request(shard: int) -> tuple[dict[str, Any], bytes] | None:
            assigned = assignments[shard]
            if not assigned:
                return None
            gpu_id = self._gpu_ids[shard]
            runtime_values = {
                name: value for name, value in self._options.items()
                if name in _RUNTIME_OPTION_FIELDS
            }
            if "worker_limits" not in runtime_values:
                runtime_values["worker_limits"] = replace(
                    _DEFAULT_LIMITS,
                    timeout_ms=max(1, math.ceil(self._worker_timeout_seconds * 1000.0)),
                    max_stderr_bytes=self._worker_stderr_tail_bytes,
                )
            options = runtime_options_payload({
                **runtime_values, "cuda_visible_devices": str(gpu_id),
            })
            payload = {
                "schema": _GROUP_REQUEST_SCHEMA,
                "enrollment_bundle_sha256": enrollment_bundle_sha256,
                "options": options,
                "groups": [
                    {
                        "key": list(key),
                        "segments": [dict(row) for _, row in members],
                        "enrollment_embedding": enrollments[key]["embedding"],
                        "enrollment_bundle_sha256": enrollment_bundle_sha256,
                    }
                    for key, members in assigned
                ],
            }
            request_bytes = _json_bytes(payload) + b"\n"
            if len(request_bytes) > _GROUP_REQUEST_MAX_BYTES:
                raise ValueError(
                    f"GPU shard {gpu_id} exceeds safe GPU child request payload limit"
                )
            return payload, request_bytes

        shard_requests = [
            _build_shard_request(shard) for shard in range(len(self._gpu_ids))
        ]
        nonempty_shards = [
            index for index, request in enumerate(shard_requests) if request is not None
        ]
        if not nonempty_shards:
            return {}

        def run_chunk(shard: int, payload: Mapping[str, Any], request_bytes: bytes):
            expected = {
                row["segment_id"]: tuple(group["key"])
                for group in payload["groups"]
                for row in group["segments"]
            }
            gpu_id = self._gpu_ids[shard]
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            worker = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "workers", "wespeaker_group_worker.py",
            ))
            project_root = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "..", "..",
            ))
            environment["PYTHONPATH"] = project_root + (
                os.pathsep + environment["PYTHONPATH"]
                if environment.get("PYTHONPATH") else ""
            )
            self._verify_frozen_sources()
            proc = self._popen(
                [sys.executable, worker],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                start_new_session=True,
            )
            with self._processes_lock:
                self._processes.add(proc)
            try:
                try:
                    stdout, stderr = run_bounded_subprocess(
                        proc,
                        request_bytes,
                        timeout_seconds=self._worker_timeout_seconds,
                        stdout_limit_bytes=_GROUP_REQUEST_MAX_BYTES,
                        stderr_tail_bytes=self._worker_stderr_tail_bytes,
                        killpg=os.killpg,
                    )
                except BoundedSubprocessTimeoutError as exc:
                    raise ValueError(
                        f"GPU child timed out after {self._worker_timeout_seconds:g}s; "
                        f"stderr_tail={self._stderr_tail(exc.stderr)!r}"
                    ) from exc
                except BoundedSubprocessOutputLimitError as exc:
                    raise ValueError(
                        f"GPU child exceeded bounded {exc.stream} output limit; "
                        f"stderr_tail={self._stderr_tail(exc.stderr)!r}"
                    ) from exc
                except BoundedSubprocessError as exc:
                    raise ValueError(
                        f"GPU child failed during bounded I/O; "
                        f"stderr_tail={self._stderr_tail(exc.stderr)!r}"
                    ) from exc
                if proc.returncode != 0:
                    raise ValueError(
                        f"GPU child exited unsuccessfully; stderr_tail={self._stderr_tail(stderr)!r}"
                    )
            finally:
                # 只有确认 child 已退出才注销；顽固进程保留给协调清理路径。
                if proc.poll() is not None:
                    with self._processes_lock:
                        self._processes.discard(proc)
            observed: dict[str, Mapping[str, Any]] = {}
            for raw in stdout.splitlines():
                try:
                    result = json.loads(raw.decode("utf-8", "strict"))
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise ValueError("GPU child returned invalid JSON") from exc
                if (not isinstance(result, Mapping)
                        or set(result) != {
                            "schema", "group_key", "segment_id",
                            "enrollment_bundle_sha256", "shadow",
                        }
                        or result["schema"] != "audio_filter_pipeline.wespeaker_group_result.v3"
                        or result["enrollment_bundle_sha256"] != enrollment_bundle_sha256
                        or not isinstance(result["group_key"], list)
                        or len(result["group_key"]) != 3
                        or any(type(value) is not str for value in result["group_key"])
                        or type(result["segment_id"]) is not str):
                    raise ValueError("GPU child returned invalid result schema or bundle binding")
                segment_id, key = result["segment_id"], tuple(result["group_key"])
                if segment_id not in expected or expected[segment_id] != key or segment_id in observed:
                    raise ValueError("GPU child result identity mismatch")
                observed[segment_id] = self._validate_shadow(
                    result["shadow"], enrollment_bundle_sha256,
                )
            if set(observed) != set(expected):
                raise ValueError("GPU child returned incomplete or duplicate segment results")
            return {
                key: {
                    segment_id: observed[segment_id]
                    for segment_id, assigned_key in expected.items()
                    if assigned_key == key
                }
                for key in {tuple(group["key"]) for group in payload["groups"]}
            }

        def run_child(shard: int):
            request = shard_requests[shard]
            assert request is not None
            return run_chunk(shard, *request)

        with ThreadPoolExecutor(max_workers=len(nonempty_shards)) as executor:
            futures = {
                executor.submit(run_child, shard): shard for shard in nonempty_shards
            }
            shard_results = []
            try:
                for future in as_completed(futures):
                    shard_results.append(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                self._terminate_active_children()
                raise
        merged: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = {}
        observed_segments: set[str] = set()
        for shard in shard_results:
            for key, result in shard.items():
                target = merged.setdefault(key, {})
                for segment_id, shadow in result.items():
                    if segment_id in observed_segments:
                        raise ValueError("GPU child returned duplicate segment across shards")
                    observed_segments.add(segment_id)
                    target[segment_id] = shadow
        if set(merged) != set(groups):
            raise ValueError("GPU child returned incomplete speaker groups")
        if observed_segments != seen_segments:
            raise ValueError("GPU child returned incomplete or duplicate segment results")
        ordered: dict[tuple[str, str, str], Mapping[str, Mapping[str, Any]]] = {}
        for key, members in groups.items():
            result = merged[key]
            expected = expected_group_segments[key]
            if set(result) != expected:
                raise ValueError("GPU child returned incomplete speaker group segments")
            ordered[key] = {
                row["segment_id"]: result[row["segment_id"]]
                for _, row in members
            }
        return ordered


class Step5SpeakerShadowStage(ArtifactStage):
    """使用冻结 enrollment 对 Step3 合格单说话人片段做 verification。

    直接消费 Step3 分数行（与 Step4 筛选解耦）：改 MOS 阈值不会触发声纹重算。
    """

    step = PipelineStep.STEP5
    upstream_step = PipelineStep.STEP3
    # 需先全量读取并分组才能批量 embedding，读取阶段不产生输出行；
    # 自绘进度条：读取逐行推进、embedding 阶段提示执行中，外壳不再叠画。
    progress_mode = "self_managed"
    progress_unit = "segment"

    def __init__(self, embedding_extractor: Optional[EmbeddingExtractor] = None):
        self._embedding_extractor = embedding_extractor
        self._embedding_provenance = (
            dict(_extractor_provenance(embedding_extractor))
            if embedding_extractor is not None else None
        )

    def config_payload(self, config: PipelineConfig) -> Mapping[str, Any]:
        policy = config.speaker_consistency
        return {
            "row_contract": {
                "version": 3,
                "fields": list(STEP5_ROW_FIELDS),
            },
            "upstream_row_contract": {
                "version": 3,
                "fields": list(STEP3_ROW_FIELDS),
            },
            "speaker_shadow": {
                "version": 9,
                "eligible_rows": "all_segments_with_valid_identity",
                "group_key": ["dataset_id", "speaker_scope", "speaker_id"],
                "algorithm": _ALGORITHM,
                "enabled": policy["enabled"],
                "enrollment_bundle": {
                    "path": policy.get("enrollment_bundle_path"),
                    "sha256": policy.get("enrollment_bundle_sha256"),
                },
                "enrollment_exclusion": {
                    "path": policy.get("enrollment_exclusion_path"),
                    "sha256": policy.get("enrollment_exclusion_sha256"),
                },
                "embedding_runtime": (
                    dict(self._embedding_provenance)
                    if self._embedding_provenance is not None
                    else {"capability": "unavailable"}
                ),
            },
            "waveform_policy": {"read": False, "write": False, "copy": False},
        }

    def produce_rows(self, context: StageRunContext, upstream_index: Optional[ArtifactIndex], upstream_rows: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
        if upstream_index is None or upstream_index.step != PipelineStep.STEP3.value:
            raise ValueError("Step5 speaker shadow requires a Step3 upstream artifact")
        progress = tqdm(
            total=upstream_index.row_count,
            unit=self.progress_unit,
            desc=f"step5:{context.dataset_id}",
            file=sys.stdout,
            disable=None,
        )
        upstream_iterator: Any = None
        try:
            upstream_iterator = iter(upstream_rows)
            rows, groups, source_segments, segment_ids = [], {}, set(), set()
            rows_by_segment_id: dict[str, Mapping[str, Any]] = {}
            policy = context.config.speaker_consistency
            enrollments = _load_enrollment_bundle(policy) if policy["enabled"] else {}
            exclusions = _load_enrollment_exclusions(policy, enrollments, upstream_index) if policy["enabled"] else {}
            for upstream_row in upstream_iterator:
                _validate_step3_row(upstream_row, context.dataset_id, context.config)
                source_segment = (upstream_row["source_key"], upstream_row["start_sample"], upstream_row["end_sample"])
                if source_segment in source_segments or upstream_row["segment_id"] in segment_ids:
                    raise ValueError("duplicate source segment identity in Step3 artifact" if source_segment in source_segments else "duplicate segment_id in Step3 artifact")
                source_segments.add(source_segment); segment_ids.add(upstream_row["segment_id"])
                row = dict(upstream_row)
                rows.append(row)
                rows_by_segment_id[upstream_row["segment_id"]] = row
                progress.update(1)
            for segment_id, key in exclusions.items():
                row = rows_by_segment_id.get(segment_id)
                if (row is None or (row["dataset_id"], row["speaker_scope"], row["speaker_id"]) != key):
                    raise ValueError("enrollment exclusion segment must occur exactly once in Step3 with matching identity")
            for index, row in enumerate(rows):
                if row["segment_id"] in exclusions:
                    row["speaker_shadow"] = _shadow("not_trial", "enrollment_segment_excluded", enrollment_bundle_sha256=policy["enrollment_bundle_sha256"])
                elif row["speaker_id"] is None: row["speaker_shadow"] = _shadow("disabled", "speaker_identity_unavailable")
                elif not policy["enabled"]: row["speaker_shadow"] = _shadow("disabled", "speaker_consistency_disabled")
                else:
                    key = (row["dataset_id"], row["speaker_scope"], row["speaker_id"])
                    if key not in enrollments: row["speaker_shadow"] = _shadow("not_enrolled", "frozen_enrollment_missing")
                    else: groups.setdefault(key, []).append((index, row))
            if self._embedding_extractor is None:
                for members in groups.values():
                    for index, _ in members: rows[index]["speaker_shadow"] = _shadow("unavailable", "no_embedding_extractor", capability="embedding_extractor")
                yield from rows; return
            # 读取已全部完成（进度条满），进入批量 embedding 计算：desc 提示
            # 执行中，避免进度条长时间看似停滞在 0。
            progress.set_description(f"step5:{context.dataset_id}:embedding")
            try:
                pooled = (self._embedding_extractor.extract_groups(
                    groups, enrollments, policy["enrollment_bundle_sha256"],
                ) if isinstance(self._embedding_extractor, SpeakerGroupExtractorPool) and groups else None)
                for key, members in groups.items():
                    if pooled is not None:
                        shadows = pooled.get(key)
                        if shadows is None or set(shadows) != {row["segment_id"] for _, row in members}: raise ValueError("GPU child result is missing a speaker group segment")
                        for index, row in members: rows[index]["speaker_shadow"] = dict(shadows[row["segment_id"]])
                        continue
                    extract_batch = getattr(self._embedding_extractor, "extract_batch", None)
                    embeddings = extract_batch([row for _, row in members]) if callable(extract_batch) else [self._embedding_extractor(row) for _, row in members]
                    if isinstance(embeddings, (str, bytes)) or not isinstance(embeddings, Sequence) or len(embeddings) != len(members): raise ValueError("embedding batch result must match group size")
                    enrollment = enrollments[key]["embedding"]
                    for (index, _), embedding in zip(members, embeddings):
                        vector = _l2_normalize(_embedding_vector(embedding))
                        rows[index]["speaker_shadow"] = {"status": "computed", "cosine_similarity_to_enrollment": _cosine(vector, enrollment), "provenance": {**self._embedding_provenance, "inference_performed": True, "algorithm": _ALGORITHM, "enrollment_bundle_sha256": policy["enrollment_bundle_sha256"]}}
            except WeSpeakerRuntimeError as exc:
                for members in groups.values():
                    for index, _ in members: rows[index]["speaker_shadow"] = _shadow("unavailable", "embedding_runtime_unavailable", error_type=type(exc).__name__)
            yield from rows
        finally:
            if upstream_iterator is not None:
                close = getattr(upstream_iterator, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException:
                        # 关闭上游迭代器失败不能覆盖主异常，但也不能静默吞掉。
                        logger.warning(
                            "Step5 failed to close Step3 upstream iterator",
                            exc_info=True,
                        )
            progress.close()
