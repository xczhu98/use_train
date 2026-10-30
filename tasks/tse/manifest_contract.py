"""TSE manifest 身份合同，供 Dataset 与资源生成器共用。

本模块只负责 speaker/recording 身份、enrollment schema 与路径覆盖校验，
不导入 Dataset、采样器或训练运行时；错误语义与数据格式保持统一。
"""

from __future__ import annotations

import json
from pathlib import Path

from core.data.resource_paths import resolve_manifest_resource_path

# canonical speaker identity：``(dataset_id, speaker_id)`` 二元组。
SpeakerIdentity = tuple[str, str]

__all__ = [
    "SpeakerIdentity",
    "build_speaker_identities",
    "parse_enrollment_manifest",
    "build_recording_ids",
]


def _validate_identity_field(value: object, *, field: str, location: str) -> str:
    """identity 字段校验：非空、去除首尾空白后不变化；speaker 不得为 unknown。"""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{location}: {field} must be a non-empty trimmed string")
    if field == "speaker_id" and value.casefold() == "unknown":
        raise ValueError(f"{location}: speaker_id must not be unknown")
    return value


def build_speaker_identities(
    metadata: dict, speech_scp: str
) -> dict[str, SpeakerIdentity]:
    """strict JSONL 主事实：从**完整** metadata pool 构建 uid → canonical identity。

    校验作用于整个 speech pool（含 duration-ineligible 记录）——identity
    配对与 interference 会使用完整 pool，不只是 sampling eligible 子集。
    """
    identity_map: dict[str, SpeakerIdentity] = {}
    for uid, record in metadata.items():
        location = f"{speech_scp} (uid={uid!r})"
        dataset_id = _validate_identity_field(
            record.get("dataset_id"), field="dataset_id", location=location
        )
        speaker_id = _validate_identity_field(
            record.get("speaker_id"), field="speaker_id", location=location
        )
        identity_map[uid] = (dataset_id, speaker_id)
    return identity_map


def parse_enrollment_manifest(
    path: str, known_identities: set[SpeakerIdentity]
) -> dict[SpeakerIdentity, tuple[str, ...]]:
    """strict nested enrollment：``dataset_id -> speaker_id -> [paths]``。

    归一化为 canonical tuple identity key；flat speaker map 在 strict 模式
    下明确拒绝（避免 namespace 丢失）；未知 dataset/speaker key 报 coverage
    错误；空 path、非 string path、空 list 均拒绝。
    """
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"spk2enroll_json must be a non-empty nested mapping "
            f"dataset_id -> speaker_id -> [enroll paths] in strict JSONL mode, "
            f"got {type(raw)}"
        )
    known_dataset_ids = {dataset_id for dataset_id, _ in known_identities}
    parsed: dict[SpeakerIdentity, tuple[str, ...]] = {}
    for dataset_id, by_speaker in raw.items():
        if not isinstance(by_speaker, dict):
            raise ValueError(
                f"{path}: flat speaker enrollment maps are not supported in "
                f"strict JSONL mode; expected nested dataset_id -> speaker_id "
                f"-> [paths], but {dataset_id!r} maps to {type(by_speaker)}"
            )
        dataset_key = _validate_identity_field(
            dataset_id, field="dataset_id", location=path
        )
        if dataset_key not in known_dataset_ids:
            raise ValueError(
                f"{path}: enrollment dataset {dataset_key!r} has no matching "
                "dataset_id in the strict speech manifest (coverage)"
            )
        if not by_speaker:
            raise ValueError(
                f"{path}: enrollment dataset {dataset_key!r} must map to a "
                "non-empty speaker mapping"
            )
        for speaker_id, paths in by_speaker.items():
            speaker_key = _validate_identity_field(
                speaker_id, field="speaker_id", location=path
            )
            identity = (dataset_key, speaker_key)
            if identity not in known_identities:
                raise ValueError(
                    f"{path}: enrollment key {identity!r} has no matching "
                    "speaker identity in the strict speech manifest (coverage)"
                )
            if (
                not isinstance(paths, list)
                or not paths
                or any(
                    not isinstance(item, str)
                    or not item
                    or item != item.strip()
                    for item in paths
                )
            ):
                raise ValueError(
                    f"{path}: enrollment paths for {identity!r} must be a "
                    "non-empty list of non-empty trimmed strings"
                )
            parsed[identity] = tuple(
                resolve_manifest_resource_path(item, path) for item in paths
            )
    return parsed


def build_recording_ids(metadata: dict, enrollments: dict, manifest: str) -> dict:
    """按 dataset_id 隔离原始录音；启用身份的语料必须完整覆盖 speech/enrollment。"""
    datasets = {row["dataset_id"] for row in metadata.values() if "recording_id" in row}
    result: dict[str, tuple[str, str]] = {}
    for uid, row in metadata.items():
        if row["dataset_id"] not in datasets:
            continue
        recording = _validate_identity_field(
            row.get("recording_id"), field="recording_id", location=f"{manifest}:{uid}"
        )
        path = str(Path(resolve_manifest_resource_path(row["path"], manifest)).resolve())
        identity = (row["dataset_id"], recording)
        if result.setdefault(path, identity) != identity:
            raise ValueError(f"conflicting recording_id for {path}")
    for (dataset, _speaker), paths in enrollments.items():
        if dataset in datasets:
            for path in paths:
                identity = result.get(str(Path(path).resolve()))
                if identity is None or identity[0] != dataset:
                    raise ValueError(f"enrollment recording_id missing from speech manifest: {path}")
    return result
