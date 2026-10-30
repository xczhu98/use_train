"""Stage D strict noise manifest 的共享 schema/version 准入契约。"""

from __future__ import annotations

from core.data.noise_admission import (
    NOISE_MANIFEST_SCHEMA_VERSION,
    NoiseAdmissionContext,
    build_validated_noise_record as _core_build_validated_noise_record,
)
from core.data.noise_admission_policy import (
    C2_LABEL_FIELDS,
    C2_STAGE_FIELDS,
    C3_STAGE_FIELDS,
    EVIDENCE_CONTRACT_VERSION,
    LEGACY_C2_LABEL_FIELDS,
    LEGACY_C3_FIELDS,
)

GENERATOR_VERSION = "6.1"
_EXPECTED_MANIFEST_SCHEMA_VERSION = 6
_EXPECTED_ADMISSION_POLICY_VERSION = 4
_EXPECTED_EVIDENCE_CONTRACT_VERSION = 5
_AUDIO_RECORD_FIELDS = {
    "record_id",
    "dataset_id",
    "source_id",
    "chunk_id",
    "split",
    "audio",
    "labels",
    "stages",
    "rejected_reason",
    "provenance",
}


def validate_exact_stage_d_audio_record(record: object, *, location: str) -> dict:
    """对 Stage D AudioRecord 执行 exact-schema 与 legacy-field 门禁。"""
    if not isinstance(record, dict):
        raise ValueError(f"{location}: record must be an object")
    missing = _AUDIO_RECORD_FIELDS - set(record)
    unknown = set(record) - _AUDIO_RECORD_FIELDS
    if missing or unknown:
        raise ValueError(
            f"{location}: invalid AudioRecord fields: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    labels = record.get("labels")
    if not isinstance(labels, dict):
        raise ValueError(f"{location}: labels must be an object")
    legacy_labels = LEGACY_C2_LABEL_FIELDS & set(labels)
    unknown_labels = set(labels) - C2_LABEL_FIELDS
    if legacy_labels or unknown_labels:
        raise ValueError(
            f"{location}: invalid strict-like labels fields: "
            f"legacy={sorted(legacy_labels)}, unknown={sorted(unknown_labels)}"
        )
    stages = record.get("stages")
    if not isinstance(stages, dict):
        raise ValueError(f"{location}: stages must be an object")
    legacy_stages = {"C4", "C", "D"} & set(stages)
    if legacy_stages:
        raise ValueError(
            f"{location}: legacy admission stages are forbidden: {sorted(legacy_stages)}"
        )
    for stage_name, allowed_fields in (("C2", C2_STAGE_FIELDS), ("C3", C3_STAGE_FIELDS)):
        value = stages.get(stage_name)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{location}: stages.{stage_name} must be an object")
        legacy = LEGACY_C3_FIELDS & set(value) if stage_name == "C3" else set()
        if legacy:
            raise ValueError(
                f"{location}: legacy admission fields are forbidden: {sorted(legacy)}"
            )
        unknown_stage_fields = set(value) - allowed_fields
        if unknown_stage_fields:
            raise ValueError(
                f"{location}: invalid stages.{stage_name} fields: "
                f"unknown={sorted(unknown_stage_fields)}"
            )
    return record


def _validate_v6_output(output_record: dict, *, location: str) -> None:
    """确认 core builder 产出当前唯一可发布的 strict 6/4/5 契约。"""
    runtime_versions = {
        "manifest_schema_version": NOISE_MANIFEST_SCHEMA_VERSION,
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
    }
    if runtime_versions != {
        "manifest_schema_version": _EXPECTED_MANIFEST_SCHEMA_VERSION,
        "evidence_contract_version": _EXPECTED_EVIDENCE_CONTRACT_VERSION,
    }:
        raise ValueError(f"runtime publication contract is stale: {runtime_versions!r}")
    expected_versions = {
        "manifest_schema_version": _EXPECTED_MANIFEST_SCHEMA_VERSION,
        "admission_policy_version": _EXPECTED_ADMISSION_POLICY_VERSION,
    }
    for field, expected in expected_versions.items():
        if output_record.get(field) != expected:
            raise ValueError(f"{location}: {field} must be exact int {expected}")
    if output_record.get("admission_decision") not in {
        "accept_noise",
        "accept_babble",
    }:
        raise ValueError(f"{location}: admission_decision is not publishable")
    for field in (
        "balance_category_mid",
        "balance_category",
        "balance_category_source",
    ):
        value = output_record.get(field)
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{location}: {field} must be a non-empty trimmed string")
    verification = output_record.get("verification")
    if not isinstance(verification, dict):
        raise ValueError(f"{location}: verification must be an object")
    if verification.get("evidence_contract_version") != _EXPECTED_EVIDENCE_CONTRACT_VERSION:
        raise ValueError(
            f"{location}: verification.evidence_contract_version must be exact int "
            f"{_EXPECTED_EVIDENCE_CONTRACT_VERSION}"
        )
    top5_hits = verification.get("top5_hits")
    if not isinstance(top5_hits, list) or len(top5_hits) != 5:
        raise ValueError(f"{location}: verification.top5_hits must contain exactly 5 items")


def build_validated_noise_record(
    record: dict,
    *,
    location: str,
    context: NoiseAdmissionContext | None = None,
) -> dict:
    """拒绝 legacy 字段，重算准入并返回经过 strict v6 门禁的记录。"""
    labels = record.get("labels") if isinstance(record, dict) else None
    legacy_labels = {"audioset_scores", "audioset_top10"} & set(
        labels if isinstance(labels, dict) else {}
    )
    stages = record.get("stages") if isinstance(record, dict) else None
    c3 = stages.get("C3") if isinstance(stages, dict) else None
    legacy_c3 = {
        "admission_calibration_version",
        "admission_calibration_sha256",
        "calibration_approved",
        "attestation",
    } & set(c3 if isinstance(c3, dict) else {})
    if legacy_labels or legacy_c3:
        raise ValueError(
            f"{location}: legacy admission fields are forbidden: "
            f"labels={sorted(legacy_labels)}, C3={sorted(legacy_c3)}"
        )
    output_record = _core_build_validated_noise_record(
        record,
        location=location,
        context=context,
    )
    _validate_v6_output(output_record, location=location)
    return output_record
