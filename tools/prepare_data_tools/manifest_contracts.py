"""训练数据 manifest 的只读格式与能力检查契约。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from core.data.noise_manifest_adapter import (
    ParsedNoiseManifest,
    read_noise_source_manifest as read_noise_manifest,
)
from core.data.simulation_utils import read_noise_source_manifest, read_training_noise_manifest

PROFILE_DYNAMIC_SOURCE = "dynamic-source"
PROFILE_DYNAMIC_SPEECH = "dynamic-speech"
PROFILE_CATEGORIZED_NOISE = "categorized-noise"
PROFILE_STRICT_NOISE_PUBLICATION = "strict-noise-publication"
MANIFEST_PROFILES = (
    PROFILE_DYNAMIC_SOURCE,
    PROFILE_DYNAMIC_SPEECH,
    PROFILE_CATEGORIZED_NOISE,
    PROFILE_STRICT_NOISE_PUBLICATION,
)
_NOISE_PUBLICATION_MARKERS = frozenset(
    {
        "manifest_schema_version",
        "admission_allowed",
        "admission_decision",
        "admission_reason_codes",
        "admission_policy_version",
        "admission_policy_sha256",
        "balance_category_mid",
        "balance_category_source",
        "verification",
    }
)


class ManifestInspectionError(ValueError):
    """manifest 无法读取或不满足所选 profile。"""


@dataclass(frozen=True)
class ManifestCapabilities:
    """与序列化格式独立的 manifest 能力。"""

    has_metadata: bool
    length_complete: bool
    category_complete: bool
    strict_fields_complete: bool


@dataclass(frozen=True)
class ManifestInspection:
    """单个 manifest 的稳定检查结果。"""

    manifest: str
    profile: str
    serialization_format: str
    record_count: int
    sample_rates: tuple[int, ...]
    capabilities: ManifestCapabilities
    yaml_hint: str

    def to_dict(self) -> dict[str, object]:
        """转换为适合稳定 JSON 输出的基础类型。"""
        return asdict(self)


def _serialization_format(path: Path) -> str:
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
    return "scp"


def _yaml_hint(path: Path, *, profile: str) -> str:
    if profile == PROFILE_DYNAMIC_SPEECH:
        key = "speech_scp"
    else:
        key = "noise_scp"
    return f"{key}: {json.dumps(str(path), ensure_ascii=False)}"


def _validate_category_mapping(parsed: ParsedNoiseManifest, *, path: Path) -> None:
    """验证训练均衡要求的 category ID/name 双向唯一映射。"""
    name_by_id: dict[int, str] = {}
    id_by_name: dict[str, int] = {}
    for uid, record in parsed.metadata.items():
        category_id = record["category_id"]
        category_name = record.get("balance_category", record.get("category_name"))
        previous_name = name_by_id.get(category_id)
        if previous_name is not None and previous_name != category_name:
            raise ManifestInspectionError(
                f"{path}: category_id {category_id} maps to multiple category names: "
                f"{previous_name!r} vs {category_name!r} (uid={uid!r})"
            )
        previous_id = id_by_name.get(category_name)
        if previous_id is not None and previous_id != category_id:
            raise ManifestInspectionError(
                f"{path}: category name {category_name!r} maps to multiple category_id "
                f"values: {previous_id} vs {category_id} (uid={uid!r})"
            )
        name_by_id[category_id] = category_name
        id_by_name[category_name] = category_id


def _has_noise_publication_markers(parsed: ParsedNoiseManifest) -> bool:
    """判断基础 metadata 是否声明了 production publication 契约。"""
    return any(
        _NOISE_PUBLICATION_MARKERS.intersection(record)
        for record in parsed.metadata.values()
    )


def _inspect_noise_manifest(path: Path, *, profile: str) -> ManifestInspection:
    parsed = read_training_noise_manifest(path)
    if not parsed.flat:
        raise ManifestInspectionError(
            f"{path}: profile {profile!r} requires at least one record"
        )
    if (
        profile == PROFILE_STRICT_NOISE_PUBLICATION
        or _has_noise_publication_markers(parsed)
    ):
        # publication 标记一旦出现，弱 profile 也必须复用统一生产门禁；该门禁只读取
        # manifest 与 canonical policy 等控制文件，不访问记录指向的音频 payload。
        parsed = read_noise_manifest(path, production_admission=True)

    if profile == PROFILE_CATEGORIZED_NOISE:
        if not parsed.capabilities.category_complete:
            raise ManifestInspectionError(
                f"{path}: profile {profile!r} requires category metadata "
                "on every record"
            )
        _validate_category_mapping(parsed, path=path)

    return ManifestInspection(
        manifest=str(path),
        profile=profile,
        serialization_format=_serialization_format(path),
        record_count=len(parsed.flat),
        sample_rates=tuple(sorted(parsed.by_fs)),
        capabilities=_noise_capabilities(parsed),
        yaml_hint=_yaml_hint(path, profile=profile),
    )


def _noise_capabilities(parsed: ParsedNoiseManifest) -> ManifestCapabilities:
    return ManifestCapabilities(
        has_metadata=parsed.capabilities.has_metadata,
        length_complete=parsed.lengths is not None,
        category_complete=parsed.capabilities.category_complete,
        strict_fields_complete=parsed.capabilities.strict_fields_complete,
    )


def _inspect_dynamic_speech(path: Path) -> ManifestInspection:
    """``dynamic-speech`` 生产发布门禁：要求 canonical 完整 metadata。"""
    parsed = read_noise_source_manifest(path)
    if not parsed.flat:
        raise ManifestInspectionError(
            f"{path}: profile 'dynamic-speech' requires at least one record"
        )
    if parsed.lengths is None:
        raise ManifestInspectionError(
            f"{path}: profile 'dynamic-speech' requires a positive integer "
            "length for every record (four-column SCP or JSONL with "
            "complete metadata)"
        )
    return ManifestInspection(
        manifest=str(path),
        profile=PROFILE_DYNAMIC_SPEECH,
        serialization_format=_serialization_format(path),
        record_count=len(parsed.flat),
        sample_rates=tuple(sorted(parsed.by_fs)),
        capabilities=_noise_capabilities(parsed),
        yaml_hint=_yaml_hint(path, profile=PROFILE_DYNAMIC_SPEECH),
    )


def inspect_manifest(
    path: str | Path,
    *,
    profile: str = PROFILE_DYNAMIC_SOURCE,
) -> ManifestInspection:
    """只读检查 manifest 是否满足指定 profile，不访问记录内资源。"""
    if profile not in MANIFEST_PROFILES:
        choices = ", ".join(MANIFEST_PROFILES)
        raise ManifestInspectionError(
            f"unknown manifest profile {profile!r}; choose one of: {choices}"
        )

    candidate = Path(path)
    try:
        if profile == PROFILE_DYNAMIC_SPEECH:
            return _inspect_dynamic_speech(candidate)
        return _inspect_noise_manifest(candidate, profile=profile)
    except ManifestInspectionError:
        raise
    except (OSError, ValueError) as exc:
        raise ManifestInspectionError(
            f"{candidate}: could not inspect profile {profile!r}: {exc}"
        ) from exc


__all__ = [
    "MANIFEST_PROFILES",
    "PROFILE_CATEGORIZED_NOISE",
    "PROFILE_DYNAMIC_SOURCE",
    "PROFILE_DYNAMIC_SPEECH",
    "PROFILE_STRICT_NOISE_PUBLICATION",
    "ManifestCapabilities",
    "ManifestInspection",
    "ManifestInspectionError",
    "inspect_manifest",
]
