#!/usr/bin/env python3
"""统一 SE/AEC/TSE 仿真审计 envelope；task payload 保持独立。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from core.data.simulation_trace import SimulationTrace  # noqa: E402

CONTRACT_VERSION = "1.4"
SCHEMA_VERSION = 1
AUDIT_FORMAT_LAYERS_MEASUREMENTS = "layers-measurements-v1"
AUDIT_FORMAT_ROLE_LOCAL_TRACE = "role-local-simulation-trace-metadata-v1"
AUDIT_FORMAT_TSE_STEM_DESCRIPTORS = "stem-descriptors-v2"
AUDIT_FORMAT_SOURCE_SCENE = "source-scene-stems-v2"
TSE_STEM_UNAVAILABLE_REASON = "capture_not_requested"
TSE_FINAL_PUBLICATION_MEASURE_POINT = "final_published_waveform"
TSE_LAYER_A_MEASURE_POINT = "layer_a_output"
TSE_SIR_PRE_RIR_MEASURE_POINT = "sir_scaled_pre_rir"
_FLOAT32_BYTES = 4
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INDEXED_TSE_ROLES = re.compile(r"^(interference|alternate_target|enrollment)/(\d+)$")


class SimulationAuditContractError(ValueError):
    """Envelope 非法或 capability 声明不一致。"""


def _object(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SimulationAuditContractError(f"{location} 必须是 string-key JSON object")
    return value


def _exact(value: object, location: str, fields: set[str]) -> dict[str, Any]:
    data = _object(value, location)
    if set(data) != fields:
        raise SimulationAuditContractError(
            f"{location} 字段不匹配：missing={sorted(fields - set(data))}, "
            f"unknown={sorted(set(data) - fields)}"
        )
    return data


def _text(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SimulationAuditContractError(f"{location} 必须是非空字符串")
    return value


def _positive_int(value: object, location: str) -> int:
    if type(value) is not int or value <= 0:
        raise SimulationAuditContractError(f"{location} 必须是正整数")
    return value


def _json_copy(value: object, location: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SimulationAuditContractError(f"{location} 不是有限 JSON-safe 值: {exc}") from exc


def _tse_format_and_capability(
    audit_format: str, available: bool, reason: object, contract_version: str
) -> str | None:
    """验证 TSE 两种互斥的 trace 审计能力声明。"""
    if audit_format == AUDIT_FORMAT_SOURCE_SCENE and contract_version != CONTRACT_VERSION:
        raise SimulationAuditContractError("source-scene stems require the current audit contract")
    if audit_format == AUDIT_FORMAT_ROLE_LOCAL_TRACE:
        if available or reason != TSE_STEM_UNAVAILABLE_REASON:
            raise SimulationAuditContractError(
                "TSE metadata-v1 必须如实标记 unavailable 及对应合同版本的 reason"
            )
        return TSE_STEM_UNAVAILABLE_REASON
    if audit_format in {AUDIT_FORMAT_TSE_STEM_DESCRIPTORS, AUDIT_FORMAT_SOURCE_SCENE}:
        if not available or reason is not None:
            raise SimulationAuditContractError(
                "TSE stem-descriptors-v2 必须如实标记 available 且 reason 为 null"
            )
        return None
    raise SimulationAuditContractError("TSE audit_format 不受支持")


def _tse_measure_point(role: str, stem: str, location: str, *, source_scene=False) -> str:
    """验证完整 TSE role/stem allowlist 并返回唯一测量点。"""
    if role == "scene":
        if source_scene:
            if stem in {"mixture", "background", "final_mixture", "target/absent"} or re.fullmatch(r"(?:source_full_rir|target)/[0-9]+", stem):
                return TSE_FINAL_PUBLICATION_MEASURE_POINT
            raise SimulationAuditContractError(f"{location} scene stem 不在 source-scene schema")
        if stem in {"final_mixture", "final_additive_noise_reference"}:
            return TSE_FINAL_PUBLICATION_MEASURE_POINT
        raise SimulationAuditContractError(f"{location} scene stem 不在 TSE schema")
    if role == "target":
        if stem == "layer_a_output":
            return TSE_LAYER_A_MEASURE_POINT
        if stem == "final_target":
            return TSE_FINAL_PUBLICATION_MEASURE_POINT
        raise SimulationAuditContractError(f"{location} target stem 不在 TSE schema")
    matched = _INDEXED_TSE_ROLES.fullmatch(role)
    if matched is None:
        raise SimulationAuditContractError(f"{location} role 不在 TSE schema")
    role_kind, _ = matched.groups()
    if role_kind == "interference" and stem == "sir_scaled_pre_rir":
        return TSE_SIR_PRE_RIR_MEASURE_POINT
    if role_kind == "alternate_target" and stem == "final_target":
        return TSE_FINAL_PUBLICATION_MEASURE_POINT
    if role_kind == "enrollment" and stem == "final_enrollment":
        return TSE_FINAL_PUBLICATION_MEASURE_POINT
    raise SimulationAuditContractError(f"{location} role/stem 不在 TSE schema")


def _source_scene_measure_point(role: str, stem: str, location: str) -> str:
    """验证 TSE/SS 共用 source-scene stem 命名。"""
    if role == "scene":
        if stem in {
            "mixture",
            "background",
            "final_mixture",
            "final_additive_noise_reference",
            "target/absent",
        } or re.fullmatch(r"(?:source_full_rir|target)/\d+", stem):
            return TSE_FINAL_PUBLICATION_MEASURE_POINT
        raise SimulationAuditContractError(f"{location} scene stem 不在 source-scene schema")
    if role == "target" and stem == "final_target":
        return TSE_FINAL_PUBLICATION_MEASURE_POINT
    if re.fullmatch(r"(?:target|alternate_target|enrollment)/\d+", role) and stem in {
        "final_target",
        "final_enrollment",
    }:
        return TSE_FINAL_PUBLICATION_MEASURE_POINT
    raise SimulationAuditContractError(f"{location} source-scene role/stem 不合法")


def _validate_tse_descriptors(
    value: object,
    location: str,
    *,
    source_scene=False,
    task: str | None = None,
    trace_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise SimulationAuditContractError(f"{location} 必须是非空 list")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    required = {
        ("scene", "final_mixture"),
        ("scene", "final_additive_noise_reference"),
        ("target", "layer_a_output"),
        ("target", "final_target"),
        ("enrollment/0", "final_enrollment"),
    }
    if source_scene:
        if task not in {"tse", "ss"}:
            raise SimulationAuditContractError(
                f"{location} source-scene schema 必须声明 task='tse' 或 task='ss'"
            )
        required = {("scene", "mixture"), ("scene", "background")}
        if task == "tse":
            required.add(("enrollment/0", "final_enrollment"))
    alternate_indices: set[int] = set()
    enrollment_indices: set[int] = set()
    scene_source_indices: set[int] = set()
    scene_target_indices: set[int] = set()
    scene_target_absent = False
    fields = {
        "role",
        "stem",
        "measure_point",
        "sample_rate",
        "num_samples",
        "shape",
        "dtype",
        "nbytes",
        "sha256",
    }
    for index, raw in enumerate(value):
        descriptor = _exact(raw, f"{location}[{index}]", fields)
        role = _text(descriptor["role"], f"{location}[{index}].role")
        stem = _text(descriptor["stem"], f"{location}[{index}].stem")
        expected_measure_point = _tse_measure_point(role, stem, f"{location}[{index}]", source_scene=source_scene)
        if descriptor["measure_point"] != expected_measure_point:
            raise SimulationAuditContractError(
                f"{location}[{index}].measure_point 与 role/stem 测量点不匹配"
            )
        _positive_int(descriptor["sample_rate"], f"{location}[{index}].sample_rate")
        num_samples = _positive_int(
            descriptor["num_samples"], f"{location}[{index}].num_samples"
        )
        shape = descriptor["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 1
            or type(shape[0]) is not int
            or shape[0] <= 0
            or shape[0] != num_samples
        ):
            raise SimulationAuditContractError(
                f"{location}[{index}].shape 必须是 [num_samples]"
            )
        if descriptor["dtype"] != "float32":
            raise SimulationAuditContractError(f"{location}[{index}].dtype 必须为 float32")
        nbytes = _positive_int(descriptor["nbytes"], f"{location}[{index}].nbytes")
        if nbytes != num_samples * _FLOAT32_BYTES:
            raise SimulationAuditContractError(
                f"{location}[{index}].nbytes 必须是 num_samples * 4"
            )
        digest = descriptor["sha256"]
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise SimulationAuditContractError(
                f"{location}[{index}].sha256 必须是 64 位小写十六进制"
            )
        key = (role, stem)
        if key in seen:
            raise SimulationAuditContractError(f"{location} 存在重复 role/stem: {key!r}")
        seen.add(key)
        if source_scene and role == "scene":
            source_match = re.fullmatch(r"source_full_rir/(\d+)", stem)
            target_match = re.fullmatch(r"target/(\d+)", stem)
            if source_match is not None:
                scene_source_indices.add(int(source_match.group(1)))
            elif target_match is not None:
                scene_target_indices.add(int(target_match.group(1)))
            elif stem == "target/absent":
                scene_target_absent = True
        matched = _INDEXED_TSE_ROLES.fullmatch(role)
        if matched is not None:
            role_kind, index_text = matched.groups()
            if role_kind == "alternate_target":
                alternate_indices.add(int(index_text))
            elif role_kind == "enrollment":
                enrollment_indices.add(int(index_text))
        result.append(_json_copy(descriptor, f"{location}[{index}]"))
    missing = sorted(required - seen)
    if missing:
        raise SimulationAuditContractError(f"{location} 缺少固定 capture stem: {missing}")
    if source_scene:
        scene = trace_metadata.get("scene") if isinstance(trace_metadata, Mapping) else None
        if not isinstance(scene, Mapping):
            raise SimulationAuditContractError(
                f"{location} source-scene 缺少 simulation_trace.scene metadata"
            )
        if task == "tse":
            expected = scene.get("num_speakers_in_scene")
            target_present = scene.get("target_present")
            if type(expected) is not int or expected < 0:
                raise SimulationAuditContractError(
                    f"{location} TSE scene.num_speakers_in_scene 必须是非负整数"
                )
            if type(target_present) is not bool:
                raise SimulationAuditContractError(
                    f"{location} TSE scene.target_present 必须是 bool"
                )
            if len(scene_source_indices) != expected:
                raise SimulationAuditContractError(
                    f"{location} source 数量 {len(scene_source_indices)} 与 "
                    f"scene.num_speakers_in_scene={expected} 不一致"
                )
            if target_present and scene_target_absent:
                raise SimulationAuditContractError(
                    f"{location} target_present=true 不得发布 target/absent"
                )
            if not target_present and not scene_target_absent:
                raise SimulationAuditContractError(
                    f"{location} target_present=false 必须发布 target/absent"
                )
            if expected == 0 and scene_source_indices:
                raise SimulationAuditContractError(
                    f"{location} TSE 纯噪声场景不得包含 source_full_rir/target 语音轨道"
                )
        else:
            expected = scene.get("source_count")
            if type(expected) is not int or expected <= 0:
                raise SimulationAuditContractError(
                    f"{location} SS scene.source_count 必须是正整数"
                )
            if scene_target_absent:
                raise SimulationAuditContractError(
                    f"{location} SS 不得发布 TSE 专用 target/absent"
                )
            # 历史手写 SS audit fixture 未包含 speech_present，按有语音 source 解释；
            # 正式 dynamic_ss trace 始终显式发布该字段。
            speech_present = scene.get("speech_present", True)
            if type(speech_present) is not bool:
                raise SimulationAuditContractError(
                    f"{location} SS scene.speech_present 必须是 bool"
                )
            if speech_present:
                if scene_source_indices != scene_target_indices:
                    raise SimulationAuditContractError(
                        f"{location} source_full_rir/N 与 target/N 必须一一配对："
                        f"source={sorted(scene_source_indices)}, target={sorted(scene_target_indices)}"
                    )
                if scene_source_indices != set(range(expected)):
                    raise SimulationAuditContractError(
                        f"{location} scene source_full_rir/target 索引必须从 0 连续"
                    )
            else:
                if scene_source_indices:
                    raise SimulationAuditContractError(
                        f"{location} SS noise-only 不得包含 source_full_rir/N"
                    )
                if scene_target_indices != set(range(expected)):
                    raise SimulationAuditContractError(
                        f"{location} SS noise-only 必须发布 K 路连续 zero target/N"
                    )
            if speech_present and len(scene_source_indices) != expected:
                raise SimulationAuditContractError(
                    f"{location} source 数量 {len(scene_source_indices)} 与 "
                    f"scene.source_count={expected} 不一致"
                )
    if alternate_indices != set(range(len(alternate_indices))):
        raise SimulationAuditContractError(
            f"{location} alternate_target/N 索引必须从 0 连续"
        )
    if enrollment_indices != set(range(len(enrollment_indices))):
        raise SimulationAuditContractError(
            f"{location} enrollment/N 索引必须从 0 连续"
        )
    # 每个 alternate target 均须有实际发布的 enrollment；N 的 offset 固定为 +1。
    if enrollment_indices and enrollment_indices != {0} | {index + 1 for index in alternate_indices}:
        raise SimulationAuditContractError(
            f"{location} alternate_target/N 与 enrollment/N 发布配对不一致"
        )
    return result


def _validate_tse_payload(task: str, audit_format: str, payload: object) -> dict[str, Any]:
    if audit_format == AUDIT_FORMAT_ROLE_LOCAL_TRACE:
        data = _exact(payload, "payload", {"simulation_trace"})
        return _json_copy(data, "payload")
    data = _exact(payload, "payload", {"simulation_trace", "stem_descriptors"})
    copied = _json_copy(data, "payload")
    _validate_tse_descriptors(
        copied["stem_descriptors"],
        "payload.stem_descriptors",
        source_scene=audit_format == AUDIT_FORMAT_SOURCE_SCENE,
        task=task,
        trace_metadata=copied["simulation_trace"],
    )
    return copied


@dataclass(frozen=True)
class SimulationAuditEnvelope:
    task: str
    audit_format: str
    stem_available: bool
    stem_reason: str | None
    payload: Mapping[str, Any]
    schema_version: int = SCHEMA_VERSION
    contract_version: str = CONTRACT_VERSION

    @classmethod
    def from_dict(cls, value: object) -> "SimulationAuditEnvelope":
        data = _object(value, "envelope")
        fields = {
            "schema_version",
            "task",
            "audit_format",
            "contract_version",
            "stem_capability",
            "payload",
        }
        data = _exact(data, "envelope", fields)
        contract_version = _text(data["contract_version"], "contract_version")
        if contract_version != CONTRACT_VERSION:
            raise SimulationAuditContractError("contract_version 不受支持")
        if type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
            raise SimulationAuditContractError("schema_version 必须是 exact int 1")
        task = _text(data["task"], "task")
        if task not in {"se", "aec", "tse", "ss"}:
            raise SimulationAuditContractError(f"未知 task: {task!r}")
        audit_format = _text(data["audit_format"], "audit_format")
        capability = _exact(data["stem_capability"], "stem_capability", {"available", "reason"})
        available, reason = capability["available"], capability["reason"]
        if type(available) is not bool:
            raise SimulationAuditContractError("stem_capability.available 必须是 exact bool")
        if available and reason is not None:
            raise SimulationAuditContractError("stem capability 可用时 reason 必须是 null")
        if not available:
            reason = _text(reason, "stem_capability.reason")
        if task in {"se", "aec"}:
            if audit_format != AUDIT_FORMAT_LAYERS_MEASUREMENTS:
                raise SimulationAuditContractError(f"task {task!r} 与 audit_format 不匹配")
            payload = _json_copy(_object(data["payload"], "payload"), "payload")
        else:
            if task == "ss" and audit_format not in {
                AUDIT_FORMAT_ROLE_LOCAL_TRACE,
                AUDIT_FORMAT_SOURCE_SCENE,
            }:
                raise SimulationAuditContractError(
                    "SS 仅支持 role-local 或 source-scene audit format"
                )
            reason = _tse_format_and_capability(
                audit_format, available, reason, contract_version
            )
            payload = _validate_tse_payload(task, audit_format, data["payload"])
        return cls(
            task,
            audit_format,
            available,
            reason,
            payload,
            contract_version=contract_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task": self.task,
            "audit_format": self.audit_format,
            "stem_capability": {"available": self.stem_available, "reason": self.stem_reason},
            "payload": _json_copy(self.payload, "payload"),
            "contract_version": self.contract_version,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent, allow_nan=False) + "\n"


def build_layers_measurements_envelope(
    *,
    task: str,
    layers: Mapping[str, Any],
    measurements: Mapping[str, Any],
    stems_available: bool,
    stems_unavailable_reason: str | None = None,
) -> SimulationAuditEnvelope:
    if task not in {"se", "aec"}:
        raise SimulationAuditContractError("layers/measurements adapter 只接受 SE/AEC")
    return SimulationAuditEnvelope.from_dict(
        {
            "schema_version": 1,
            "contract_version": CONTRACT_VERSION,
            "task": task,
            "audit_format": AUDIT_FORMAT_LAYERS_MEASUREMENTS,
            "stem_capability": {"available": stems_available, "reason": stems_unavailable_reason},
            "payload": {"layers": layers, "measurements": measurements},
        }
    )


def _descriptor_from_stem(
    *, role: str, stem: str, waveform: object, metadata: Mapping[str, Any], source_scene=False
) -> dict[str, Any]:
    measure_point = (
        _source_scene_measure_point(role, stem, "captured stem")
        if source_scene
        else _tse_measure_point(role, stem, "captured stem")
    )
    waveform = np.asarray(waveform)
    if waveform.dtype != np.float32 or waveform.ndim != 1 or waveform.size <= 0:
        raise SimulationAuditContractError(
            f"captured stem {role}/{stem} 必须是非空一维 float32 waveform"
        )
    waveform = np.ascontiguousarray(waveform)
    if not np.isfinite(waveform).all():
        raise SimulationAuditContractError(f"captured stem {role}/{stem} 含 NaN/Inf")
    if source_scene and role == "scene":
        track = metadata.get("scene", {})
        sample_rate = track.get("sample_rate") if isinstance(track, Mapping) else None
        num_samples = track.get("num_samples") if isinstance(track, Mapping) else None
    elif role == "scene":
        track = metadata.get("final_time_transform")
        sample_rate = track.get("target_sample_rate") if isinstance(track, Mapping) else None
        num_samples = track.get("output_length") if isinstance(track, Mapping) else None
    elif measure_point == TSE_LAYER_A_MEASURE_POINT:
        track = metadata.get("layer_a")
        sample_rate = track.get("sample_rate") if isinstance(track, Mapping) else None
        num_samples = track.get("samples_after_align") if isinstance(track, Mapping) else None
    elif measure_point == TSE_SIR_PRE_RIR_MEASURE_POINT:
        track = metadata.get("sir_scaled_pre_rir")
        sample_rate = track.get("sample_rate") if isinstance(track, Mapping) else None
        num_samples = track.get("num_samples") if isinstance(track, Mapping) else None
    else:
        track = metadata.get("final_track") if stem == "final_target" else metadata.get("enrollment")
        sample_rate = track.get("sample_rate") if isinstance(track, Mapping) else None
        num_samples = track.get("num_samples") if isinstance(track, Mapping) else None
    if type(sample_rate) is not int or type(num_samples) is not int:
        raise SimulationAuditContractError(
            f"captured stem {role}/{stem} 缺少对应测量点 sample_rate/num_samples metadata"
        )
    _positive_int(sample_rate, f"captured stem {role}/{stem} sample_rate")
    _positive_int(num_samples, f"captured stem {role}/{stem} num_samples")
    if len(waveform) != num_samples:
        raise SimulationAuditContractError(
            f"captured stem {role}/{stem} 长度与最终发布 metadata 不一致"
        )
    return {
        "role": role,
        "stem": stem,
        "measure_point": measure_point,
        "sample_rate": sample_rate,
        "num_samples": num_samples,
        "shape": list(waveform.shape),
        "dtype": "float32",
        "nbytes": waveform.nbytes,
        "sha256": hashlib.sha256(waveform.tobytes()).hexdigest(),
    }


def _trace_stem_descriptors(
    trace: SimulationTrace, *, source_scene=False
) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for role, source in trace.sources.items():
        for stem, waveform in source.stems.items():
            descriptors.append(
                _descriptor_from_stem(
                    role=role,
                    stem=stem,
                    waveform=waveform,
                    metadata=source.metadata, source_scene=source_scene,
                )
            )
    descriptors.sort(key=lambda item: (item["role"], item["stem"]))
    return _validate_tse_descriptors(
        descriptors,
        "captured stem descriptors",
        source_scene=source_scene,
        task=trace.task,
        trace_metadata=trace.metadata_dict(),
    )


def validate_source_scene_descriptors(
    descriptors: object,
    *,
    task: str,
    trace_metadata: Mapping[str, Any],
    location: str = "source-scene descriptors",
) -> list[dict[str, Any]]:
    """校验 TSE/SS 共用的 source-scene descriptor 基础合同。"""
    if task not in {"tse", "ss"}:
        raise SimulationAuditContractError(
            f"{location} task must be 'tse' or 'ss', got {task!r}"
        )
    return _validate_tse_descriptors(
        descriptors,
        location,
        source_scene=True,
        task=task,
        trace_metadata=trace_metadata,
    )


def validate_tse_source_scene_policy(
    descriptors: object,
    trace_metadata: Mapping[str, Any],
    *,
    location: str = "TSE source-scene descriptors",
) -> list[dict[str, Any]]:
    """执行 TSE 的 target-present/absent 与 source 数量策略。"""
    return validate_source_scene_descriptors(
        descriptors,
        task="tse",
        trace_metadata=trace_metadata,
        location=location,
    )


def validate_ss_source_scene_policy(
    descriptors: object,
    trace_metadata: Mapping[str, Any],
    *,
    location: str = "SS source-scene descriptors",
) -> list[dict[str, Any]]:
    """执行 SS 固定 K source_count 与 source/target 配对策略。"""
    return validate_source_scene_descriptors(
        descriptors,
        task="ss",
        trace_metadata=trace_metadata,
        location=location,
    )


def build_source_scene_descriptors(trace: SimulationTrace) -> list[dict[str, Any]]:
    """从 sealed TSE/SS trace 构建并校验共用 descriptor 列表。"""
    if trace.task == "tse":
        return validate_tse_source_scene_policy(
            _trace_stem_descriptors(trace, source_scene=True),
            trace.metadata_dict(),
        )
    if trace.task == "ss":
        return validate_ss_source_scene_policy(
            _trace_stem_descriptors(trace, source_scene=True),
            trace.metadata_dict(),
        )
    raise SimulationAuditContractError(
        f"source-scene descriptors only support TSE/SS, got {trace.task!r}"
    )


def build_tse_trace_metadata_envelope(trace: SimulationTrace) -> SimulationAuditEnvelope:
    if not isinstance(trace, SimulationTrace) or trace.task != "tse":
        raise SimulationAuditContractError("TSE adapter 要求 trace.task='tse' 的 SimulationTrace")
    if not trace.sealed:
        raise SimulationAuditContractError("TSE adapter 只接受 sealed SimulationTrace")
    metadata = trace.metadata_dict()
    source_scene = isinstance(metadata.get("scene"), dict)
    if not trace.capture_stems:
        if any(source.stems for source in trace.sources.values()):
            raise SimulationAuditContractError("TSE capture-off trace 不得含 stems")
        return SimulationAuditEnvelope.from_dict(
            {
                "schema_version": 1,
                "contract_version": CONTRACT_VERSION,
                "task": "tse",
                "audit_format": AUDIT_FORMAT_ROLE_LOCAL_TRACE,
                "stem_capability": {
                    "available": False,
                    "reason": TSE_STEM_UNAVAILABLE_REASON,
                },
                "payload": {"simulation_trace": metadata},
            }
        )
    return SimulationAuditEnvelope.from_dict(
        {
            "schema_version": 1,
            "contract_version": CONTRACT_VERSION,
            "task": "tse",
            "audit_format": AUDIT_FORMAT_SOURCE_SCENE if source_scene else AUDIT_FORMAT_TSE_STEM_DESCRIPTORS,
            "stem_capability": {"available": True, "reason": None},
            "payload": {
                "simulation_trace": metadata,
            "stem_descriptors": (
                build_source_scene_descriptors(trace)
                if source_scene
                else _trace_stem_descriptors(trace, source_scene=False)
            ),
            },
        }
    )


def build_ss_trace_metadata_envelope(trace: SimulationTrace) -> SimulationAuditEnvelope:
    """验证 SS 固定 K scene bundle 的 metadata/stem envelope。"""
    if not isinstance(trace, SimulationTrace) or trace.task != "ss":
        raise SimulationAuditContractError("SS adapter 要求 trace.task='ss' 的 SimulationTrace")
    if not trace.sealed:
        raise SimulationAuditContractError("SS adapter 只接受 sealed SimulationTrace")
    metadata = trace.metadata_dict()
    if not trace.capture_stems:
        if any(source.stems for source in trace.sources.values()):
            raise SimulationAuditContractError("SS capture-off trace 不得含 stems")
        return SimulationAuditEnvelope.from_dict({
            "schema_version": 1,
            "contract_version": CONTRACT_VERSION,
            "task": "ss",
            "audit_format": AUDIT_FORMAT_ROLE_LOCAL_TRACE,
            "stem_capability": {"available": False, "reason": TSE_STEM_UNAVAILABLE_REASON},
            "payload": {"simulation_trace": metadata},
        })
    return SimulationAuditEnvelope.from_dict({
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "task": "ss",
        "audit_format": AUDIT_FORMAT_SOURCE_SCENE,
        "stem_capability": {"available": True, "reason": None},
        "payload": {
            "simulation_trace": metadata,
            "stem_descriptors": build_source_scene_descriptors(trace),
        },
    })


def load_simulation_audit_envelope(path: str | Path) -> SimulationAuditEnvelope:
    try:
        return SimulationAuditEnvelope.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SimulationAuditContractError(f"无法读取 envelope: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        envelope = load_simulation_audit_envelope(args.audit_json)
    except SimulationAuditContractError as exc:
        print(f"错误：simulation audit envelope 验证失败：{exc}", file=sys.stderr)
        return 1
    print(
        envelope.to_json(indent=None)
        if args.json
        else f"验证通过：schema=v1，task={envelope.task}。",
        end="" if args.json else "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
