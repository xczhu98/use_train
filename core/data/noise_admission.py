"""噪声准入 strict manifest schema v6 的纯数据契约与内容指纹工具。

schema v6 删除 calibration/attestation/approved 绑定（C3 决策无阈值无校准，
approved 门禁已删除，链路直接可发布），并增加 top-5 hits 审计。
"""
from __future__ import annotations

import hashlib
import math
import stat
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.data.noise_admission_policy import (
    ADMISSION_POLICY_VERSION,
    EVIDENCE_CONTRACT_VERSION,
    LEGACY_C3_FIELDS,
)
from core.data.noise_roles import compute_category_roles

NOISE_MANIFEST_SCHEMA_VERSION = 6
REQUIRED_ARTIFACT_FINGERPRINTS = frozenset(
    {
        "voice_policy_sha256",
        "audiotag_label_dict_sha256",
        "audiotag_checkpoint_sha256",
        "qwen_model_sha256",
        "c1_prompt_sha256",
    }
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def canonical_voice_policy_path() -> Path:
    return repository_root() / "tools/noise_filter_pipeline/config/ontology/audiotag_voice_policy.json"


def sha256_file(path: str | Path) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    """确定性计算文件或目录内容指纹，并对 symlink/非 regular 项 fail-closed。"""
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError(f"fingerprint asset must not be a symlink: {candidate}")
    try:
        candidate_stat = candidate.stat()
    except FileNotFoundError:
        raise FileNotFoundError(f"fingerprint asset does not exist: {candidate}") from None
    if stat.S_ISREG(candidate_stat.st_mode):
        return sha256_file(candidate)
    if not stat.S_ISDIR(candidate_stat.st_mode):
        raise ValueError(f"fingerprint asset must be a regular file or directory: {candidate}")

    resolved = candidate.resolve()
    digest = hashlib.sha256()
    files: list[Path] = []
    for item in resolved.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"fingerprint directory contains symlink: {item}")
        try:
            item_stat = item.stat()
        except FileNotFoundError:
            raise ValueError(f"fingerprint directory contains broken entry: {item}") from None
        if stat.S_ISDIR(item_stat.st_mode):
            continue
        if not stat.S_ISREG(item_stat.st_mode):
            raise ValueError(f"fingerprint directory contains non-regular entry: {item}")
        files.append(item)
    files.sort()
    if not files:
        raise ValueError(f"fingerprint directory contains no files: {resolved}")
    for item in files:
        relative = item.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def load_canonical_voice_policy() -> tuple[Any, str]:
    """读取 canonical policy v4（委托 core 单一事实来源），返回对象与内容 SHA-256。"""
    from core.data.noise_admission_policy import load_admission_policy

    policy = load_admission_policy(canonical_voice_policy_path())
    return policy, policy.sha256


def _nonempty_string(value: Any, *, field: str, location: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{location}: {field} must be a non-empty trimmed string")
    return value


def _sha256(value: Any, *, field: str, location: str) -> str:
    text = _nonempty_string(value, field=field, location=location)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{location}: {field} must be a lowercase SHA-256 hex digest")
    return text


def validate_artifact_fingerprints(value: Any, *, location: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location}: artifact_fingerprints must be an object")
    missing = REQUIRED_ARTIFACT_FINGERPRINTS - set(value)
    unknown = set(value) - REQUIRED_ARTIFACT_FINGERPRINTS
    if missing or unknown:
        raise ValueError(
            f"{location}: invalid artifact_fingerprints fields: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return {
        key: _sha256(value.get(key), field=f"artifact_fingerprints.{key}", location=location)
        for key in sorted(REQUIRED_ARTIFACT_FINGERPRINTS)
    }


def _validate_top5_hits_audit(
    value: Any,
    *,
    policy: Any,
    location: str,
) -> None:
    """校验 strict record 的 top-5 hits 审计项（schema 6 新增）。

    要求恰好 5 项，每项 index/mid/label/kind 与 canonical label space 及
    policy 三分类一致，prob 为 [0,1] 有限值，整体 prob 降序、并列 index 升序；
    发布记录必须无 speech kind（S5 非空即 reject，不会进入发布链）。
    """
    if not isinstance(value, list) or len(value) != 5:
        raise ValueError(f"{location}: verification.top5_hits must contain exactly 5 items")
    previous: tuple[float, int] | None = None
    seen_indices: set[int] = set()
    for position, hit in enumerate(value):
        if not isinstance(hit, Mapping) or set(hit) != {"index", "mid", "label", "kind", "prob"}:
            raise ValueError(
                f"{location}: verification.top5_hits[{position}] must contain "
                "index/mid/label/kind/prob"
            )
        index = hit["index"]
        if type(index) is not int or not 0 <= index < policy.class_count:
            raise ValueError(
                f"{location}: verification.top5_hits[{position}].index is invalid"
            )
        mid = hit["mid"]
        label = hit["label"]
        prob = hit["prob"]
        if isinstance(prob, bool) or not isinstance(prob, (int, float)):
            raise ValueError(
                f"{location}: verification.top5_hits[{position}].prob must be numeric"
            )
        prob = float(prob)
        if not math.isfinite(prob) or not 0.0 <= prob <= 1.0:
            raise ValueError(
                f"{location}: verification.top5_hits[{position}].prob must be "
                "finite and within [0,1]"
            )
        if mid != policy.index_to_mid[index] or label != policy.label_of(mid):
            raise ValueError(
                f"{location}: verification.top5_hits[{position}] does not match "
                "canonical label space"
            )
        if hit["kind"] != policy.class_of(mid):
            raise ValueError(
                f"{location}: verification.top5_hits[{position}].kind does not "
                "match policy MID class"
            )
        if hit["kind"] == "speech":
            raise ValueError(
                f"{location}: verification.top5_hits[{position}] contains speech "
                "MID, which must have been vetoed by C3"
            )
        if index in seen_indices:
            raise ValueError(f"{location}: verification.top5_hits contains duplicate index")
        seen_indices.add(index)
        key = (-prob, index)
        if previous is not None and key < previous:
            raise ValueError(
                f"{location}: verification.top5_hits is not sorted by prob desc / "
                f"index asc at position {position}"
            )
        previous = key


@dataclass(frozen=True)
class NoiseAdmissionContext:
    """一次 production admission operation 复用的 canonical 依赖快照。"""

    policy: Any

    @classmethod
    def load(cls) -> "NoiseAdmissionContext":
        """一次加载 policy 与 label metadata（无 calibration/attestation 依赖）。"""
        from core.data.noise_admission_policy import load_admission_policy

        policy = load_admission_policy(canonical_voice_policy_path())
        return cls(policy=policy)


def validate_noise_manifest_records(
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    source_path: str | Path,
    context: NoiseAdmissionContext | None = None,
) -> None:
    """逐条校验训练噪声 JSONL v6，并校验全文件 policy/artifact 一致性。

    ``metadata`` 必须是对象；每条记录的基础字段类型严格校验，且外层 uid key
    必须是非空 trimmed string，并与记录内 ``uid`` 完全一致。

    Runtime 会重新加载 canonical policy，并把 manifest 内所需 artifact
    fingerprints 与 policy 绑定比较；不会读取或哈希 Qwen/AudioTag
    资产，生成资产变化导致旧 Stage 结果失效由 Stage checkpoint fingerprint 负责。
    此门禁用于发现误配置、陈旧或损坏产物；并发 writer 的发布一致性由 adapter 负责。
    schema 6 不再绑定 calibration/attestation/approved（C3 无阈值决策，approved
    门禁已删除）。schema v5 及更早版本、C4 verification 字段均不兼容。
    """
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{source_path}: metadata must be an object")
    if not metadata:
        raise ValueError(f"{source_path}: production admission manifest must not be empty")

    allowed_fields = {
        "uid",
        "fs",
        "length",
        "path",
        "manifest_schema_version",
        "balance_category_mid",
        "balance_category",
        "balance_category_source",
        "category_name",
        "category_id",
        "admission_allowed",
        "admission_decision",
        "admission_reason_codes",
        "admission_policy_version",
        "admission_policy_sha256",
        "verification",
        "dataset_id",
        "roles",
        "labels",
    }
    validated_records: list[tuple[str, Mapping[str, Any]]] = []
    for uid, item in metadata.items():
        location = f"{source_path} (uid={uid!r})"
        if not isinstance(uid, str) or not uid or uid != uid.strip():
            raise ValueError(
                f"{location}: metadata key must be a non-empty trimmed string"
            )
        if not isinstance(item, Mapping):
            raise ValueError(f"{location}: manifest record must be an object")
        record_uid = _nonempty_string(item.get("uid"), field="uid", location=location)
        if record_uid != uid:
            raise ValueError(f"{location}: uid must equal metadata key")
        fs = item.get("fs")
        if type(fs) is not int or fs <= 0:
            raise ValueError(f"{location}: fs must be an integer > 0")
        length = item.get("length")
        if type(length) is not int or length <= 0:
            raise ValueError(f"{location}: length must be an integer > 0")
        _nonempty_string(item.get("path"), field="path", location=location)
        validated_records.append((uid, item))

    operation = context or NoiseAdmissionContext.load()
    current_policy_sha = operation.policy.sha256
    expected_policy: tuple[int, str] | None = None
    expected_artifacts: dict[str, str] | None = None
    category_name_by_id: dict[int, str] = {}
    category_id_by_name: dict[str, int] = {}
    for uid, item in validated_records:
        location = f"{source_path} (uid={uid!r})"
        unknown_fields = set(item) - allowed_fields
        if unknown_fields:
            raise ValueError(
                f"{location}: invalid manifest fields: unknown={sorted(unknown_fields)}"
            )
        if "labels" in item:
            labels = item["labels"]
            if not isinstance(labels, Mapping) or set(labels) != {"original"}:
                raise ValueError(
                    f"{location}: labels may contain exactly the preserved original field"
                )
        schema_version = item.get("manifest_schema_version")
        if type(schema_version) is not int or schema_version != NOISE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"{location}: manifest_schema_version must be exact int "
                f"{NOISE_MANIFEST_SCHEMA_VERSION}"
            )
        if item.get("admission_allowed") is not True:
            raise ValueError(f"{location}: admission_allowed must be true")
        _nonempty_string(item.get("dataset_id"), field="dataset_id", location=location)
        policy_version = item.get("admission_policy_version")
        if type(policy_version) is not int or policy_version != ADMISSION_POLICY_VERSION:
            raise ValueError(
                f"{location}: admission_policy_version must be exact int "
                f"{ADMISSION_POLICY_VERSION}"
            )
        policy_sha = _sha256(
            item.get("admission_policy_sha256"),
            field="admission_policy_sha256",
            location=location,
        )
        if policy_sha != current_policy_sha:
            raise ValueError(f"{location}: admission policy hash does not match canonical policy")
        policy_identity = (policy_version, policy_sha)
        if expected_policy is None:
            expected_policy = policy_identity
        elif policy_identity != expected_policy:
            raise ValueError(f"{location}: admission policy differs within one manifest")

        decision = item.get("admission_decision")
        if decision not in ("accept_noise", "accept_babble"):
            raise ValueError(
                f"{location}: admission_decision must be accept_noise/accept_babble"
            )
        reason_codes = item.get("admission_reason_codes")
        if (
            not isinstance(reason_codes, list)
            or not reason_codes
            or any(not isinstance(code, str) or not code for code in reason_codes)
        ):
            raise ValueError(f"{location}: admission_reason_codes must be a non-empty string list")

        verification = item.get("verification")
        if not isinstance(verification, Mapping):
            raise ValueError(f"{location}: verification must be an object")
        required_verification = {
            "c1_status",
            "c1_intelligibility",
            "c2_status",
            "c3_status",
            "d2_status",
            "evidence_contract_version",
            "score_contract",
            "class_count",
            "label_space_sha256",
            "artifact_fingerprints",
            "top5_hits",
        }
        missing_verification = required_verification - set(verification)
        unknown_verification = set(verification) - required_verification
        if missing_verification or unknown_verification:
            raise ValueError(
                f"{location}: invalid verification fields: "
                f"missing={sorted(missing_verification)}, unknown={sorted(unknown_verification)}"
            )
        for status_field in ("c1_status", "c2_status", "c3_status", "d2_status"):
            if verification.get(status_field) != "passed":
                raise ValueError(f"{location}: verification.{status_field} must be 'passed'")
        intelligibility = verification.get("c1_intelligibility")
        if intelligibility not in ("noise", "babble"):
            raise ValueError(
                f"{location}: verification.c1_intelligibility must be 'noise' or 'babble'"
            )
        contract_version = verification.get("evidence_contract_version")
        if type(contract_version) is not int or contract_version != EVIDENCE_CONTRACT_VERSION:
            raise ValueError(
                f"{location}: verification.evidence_contract_version must be exact int "
                f"{EVIDENCE_CONTRACT_VERSION}"
            )
        from core.data.noise_admission_policy import CLASSIFIER_LABEL_COUNT, SCORE_CONTRACT

        if verification.get("score_contract") != SCORE_CONTRACT:
            raise ValueError(
                f"{location}: verification.score_contract must be {SCORE_CONTRACT!r}"
            )
        if verification.get("class_count") != CLASSIFIER_LABEL_COUNT:
            raise ValueError(
                f"{location}: verification.class_count must be {CLASSIFIER_LABEL_COUNT}"
            )
        label_space_sha = _sha256(
            verification.get("label_space_sha256"),
            field="verification.label_space_sha256",
            location=location,
        )
        artifacts = validate_artifact_fingerprints(
            verification.get("artifact_fingerprints"), location=location
        )
        if artifacts["voice_policy_sha256"] != policy_sha:
            raise ValueError(
                f"{location}: verification.artifact_fingerprints.voice_policy_sha256 "
                "must equal admission_policy_sha256"
            )
        if label_space_sha != operation.policy.label_space_sha256:
            raise ValueError(
                f"{location}: verification.label_space_sha256 does not match "
                "canonical policy"
            )
        if artifacts["audiotag_label_dict_sha256"] != label_space_sha:
            raise ValueError(
                f"{location}: verification.label_space_sha256 must equal "
                "artifact_fingerprints.audiotag_label_dict_sha256"
            )
        if expected_artifacts is None:
            expected_artifacts = artifacts
        elif artifacts != expected_artifacts:
            raise ValueError(f"{location}: artifact fingerprints differ within one manifest")
        top5_hits = verification.get("top5_hits")
        _validate_top5_hits_audit(
            top5_hits,
            policy=operation.policy,
            location=location,
        )
        noise_hits = [hit for hit in top5_hits if hit["kind"] == "noise"]
        expected_balance_mid = noise_hits[0]["mid"] if noise_hits else "/m/096m7z"
        expected_balance_label = operation.policy.label_of(expected_balance_mid)
        expected_balance_source = (
            "noise_top5_hit" if noise_hits else "generic_noise_fallback"
        )
        balance_category_mid = _nonempty_string(
            item.get("balance_category_mid"),
            field="balance_category_mid",
            location=location,
        )
        balance_category = _nonempty_string(
            item.get("balance_category"), field="balance_category", location=location
        )
        if operation.policy.class_of(balance_category_mid) != "noise":
            raise ValueError(f"{location}: balance_category_mid must map to noise")
        if operation.policy.label_of(balance_category_mid) != balance_category:
            raise ValueError(
                f"{location}: balance_category must be the canonical label of "
                "balance_category_mid"
            )
        balance_source = item.get("balance_category_source")
        actual_balance = (
            balance_category_mid,
            balance_category,
            balance_source,
        )
        expected_balance = (
            expected_balance_mid,
            expected_balance_label,
            expected_balance_source,
        )
        if actual_balance != expected_balance:
            raise ValueError(
                f"{location}: balance category must select the highest-ranked noise "
                f"top-5 hit or canonical fallback; expected={expected_balance!r}, "
                f"actual={actual_balance!r}"
            )
        category_name = _nonempty_string(
            item.get("category_name"), field="category_name", location=location
        )
        if category_name != balance_category:
            raise ValueError(f"{location}: category_name must equal balance_category")
        if "roles" in item:
            canonical_roles = compute_category_roles(balance_category)
            if item.get("roles") != canonical_roles:
                raise ValueError(
                    f"{location}: roles must equal canonical roles {canonical_roles!r} "
                    f"for balance_category {balance_category!r}"
                )
        category_id = item.get("category_id")
        if type(category_id) is not int or category_id < 0:
            raise ValueError(f"{location}: category_id must be a non-negative integer")
        prior_name = category_name_by_id.setdefault(category_id, category_name)
        if prior_name != category_name:
            raise ValueError(f"{location}: category_id maps to multiple category names")
        prior_id = category_id_by_name.setdefault(category_name, category_id)
        if prior_id != category_id:
            raise ValueError(f"{location}: category name maps to multiple category_ids")
        expected_reason_codes = (
            ["admitted_babble"] if decision == "accept_babble" else ["admitted_noise"]
        )
        if reason_codes != expected_reason_codes:
            raise ValueError(
                f"{location}: admission_reason_codes must equal {expected_reason_codes!r}"
            )
        expected_branch = "babble" if decision == "accept_babble" else "noise"
        if intelligibility != expected_branch:
            raise ValueError(
                f"{location}: {decision} requires C1 intelligibility {expected_branch!r}"
            )


def _require_passed_stage(stages: Mapping[str, Any], name: str, location: str) -> Mapping[str, Any]:
    entry = stages.get(name)
    if not isinstance(entry, Mapping) or entry.get("status") != "passed":
        raise ValueError(f"{location}: {name} stage must be passed")
    return entry


def _validate_production_execution(c3: Mapping[str, Any], location: str) -> None:
    execution = c3.get("execution_mode")
    if not isinstance(execution, Mapping) or set(execution) != {"c1", "c2"}:
        raise ValueError(f"{location}: C3 execution_mode must contain exactly c1/c2")
    if execution.get("c1") != "production" or execution.get("c2") != "subprocess_daemon":
        raise ValueError(f"{location}: Stage C execution is not production-publishable")
    if c3.get("publishable") is not True:
        raise ValueError(f"{location}: Stage C execution is not publishable")


def build_validated_noise_record(
    record: Mapping[str, Any],
    *,
    location: str,
    context: NoiseAdmissionContext | None = None,
) -> dict[str, Any]:
    """严格复验生产 C1→C2→C3→D2，并构建 schema v6 record。

    此接口位于 core，供 generator/materializer/promotion 后续稳定 re-export。
    普通训练 source parser 不调用本函数，因此 legacy JSON/SCP 不受标签过滤。
    schema v6 不再绑定 calibration/attestation/approved（C3 无阈值决策，approved
    门禁已删除），并在 verification 中携带 top-5 hits 审计。
    """
    from core.data.noise_admission_policy import validate_admission_contract

    operation = context or NoiseAdmissionContext.load()
    if not isinstance(record, Mapping):
        raise ValueError(f"{location}: AudioRecord must be an object")
    record_id = _nonempty_string(record.get("record_id"), field="record_id", location=location)
    dataset_id = _nonempty_string(record.get("dataset_id"), field="dataset_id", location=location)
    audio = record.get("audio")
    if not isinstance(audio, Mapping):
        raise ValueError(f"{location}: audio must be an object")
    path = _nonempty_string(audio.get("path"), field="audio.path", location=location)
    sample_rate = audio.get("sr")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError(f"{location}: audio.sr must be an exact integer > 0")
    duration = audio.get("duration")
    if (
        isinstance(duration, bool)
        or type(duration) not in (int, float)
        or not math.isfinite(float(duration))
        or duration <= 0
    ):
        raise ValueError(f"{location}: audio.duration must be a finite number > 0")

    stages = record.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError(f"{location}: stages must be an object")
    legacy = {"C4", "C", "D"} & set(stages)
    if legacy:
        raise ValueError(f"{location}: legacy admission stages are forbidden: {sorted(legacy)}")
    c1 = _require_passed_stage(stages, "C1", location)
    c2 = _require_passed_stage(stages, "C2", location)
    c3 = _require_passed_stage(stages, "C3", location)
    _require_passed_stage(stages, "D2", location)
    present_forbidden = LEGACY_C3_FIELDS & set(c3)
    if present_forbidden:
        raise ValueError(
            f"{location}: legacy C3 fields are forbidden: {sorted(present_forbidden)}"
        )
    _validate_production_execution(c3, location)
    intelligibility = c1.get("intelligibility")
    if intelligibility not in ("noise", "babble"):
        raise ValueError(f"{location}: C1 intelligibility must be noise/babble")
    labels = record.get("labels")
    if not isinstance(labels, Mapping):
        raise ValueError(f"{location}: labels must be an object")

    decision = validate_admission_contract(
        c1_intelligibility=intelligibility,
        c2_labels=labels,
        c2_stage=c2,
        admission_stage=c3,
        policy=operation.policy,
        require_accepted=True,
        location=location,
    )
    expected_reason = decision.reason_codes[0]
    if c3.get("reason") != expected_reason:
        raise ValueError(
            f"{location}: stages.C3.reason must equal canonical reason {expected_reason!r}"
        )
    artifacts = validate_artifact_fingerprints(
        c3.get("artifact_fingerprints"), location=location
    )
    from core.data.noise_admission_policy import CLASSIFIER_LABEL_COUNT, SCORE_CONTRACT

    if c2.get("evidence_contract_version") != EVIDENCE_CONTRACT_VERSION:
        raise ValueError(
            f"{location}: stages.C2.evidence_contract_version must be "
            f"{EVIDENCE_CONTRACT_VERSION}"
        )
    if c2.get("score_contract") != SCORE_CONTRACT:
        raise ValueError(f"{location}: stages.C2.score_contract must be {SCORE_CONTRACT!r}")
    if c2.get("class_count") != CLASSIFIER_LABEL_COUNT:
        raise ValueError(
            f"{location}: stages.C2.class_count must be {CLASSIFIER_LABEL_COUNT}"
        )
    label_space_sha = _sha256(
        c2.get("label_space_sha256"),
        field="stages.C2.label_space_sha256",
        location=location,
    )
    if label_space_sha != artifacts["audiotag_label_dict_sha256"]:
        raise ValueError(
            f"{location}: stages.C2.label_space_sha256 must equal "
            "artifact_fingerprints.audiotag_label_dict_sha256"
        )

    length = int(
        (Decimal(str(duration)) * Decimal(sample_rate)).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )
    if length <= 0:
        raise ValueError(f"{location}: derived length must be > 0")
    return {
        "uid": record_id,
        "fs": sample_rate,
        "length": length,
        "path": path,
        "manifest_schema_version": NOISE_MANIFEST_SCHEMA_VERSION,
        "balance_category_mid": decision.balance_category_mid,
        "balance_category": decision.balance_category,
        "balance_category_source": decision.balance_category_source,
        "category_name": decision.balance_category,
        "admission_allowed": True,
        "admission_decision": decision.decision,
        "admission_reason_codes": list(decision.reason_codes),
        "admission_policy_version": operation.policy.version,
        "admission_policy_sha256": operation.policy.sha256,
        "verification": {
            "c1_status": "passed",
            "c1_intelligibility": intelligibility,
            "c2_status": "passed",
            "c3_status": "passed",
            "d2_status": "passed",
            "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
            "score_contract": SCORE_CONTRACT,
            "class_count": CLASSIFIER_LABEL_COUNT,
            "label_space_sha256": label_space_sha,
            "artifact_fingerprints": artifacts,
            "top5_hits": [
                {
                    "index": hit.index,
                    "mid": hit.mid,
                    "label": hit.label,
                    "kind": hit.kind,
                    "prob": hit.prob,
                }
                for hit in decision.top5_hits
            ],
        },
        **(
            {"labels": {"original": labels["original"]}}
            if "original" in labels
            else {}
        ),
        "dataset_id": dataset_id,
    }


def assert_consistent_admission_identity(
    identities: Iterable[tuple[int, str, Mapping[str, str]]],
) -> None:
    """确保一批待发布记录使用相同 policy 与 artifact fingerprints。"""
    expected: tuple[int, str, tuple[tuple[str, str], ...]] | None = None
    for version, policy_sha, artifacts in identities:
        identity = (version, policy_sha, tuple(sorted(artifacts.items())))
        if expected is None:
            expected = identity
        elif identity != expected:
            raise ValueError("noise admission identity differs across output candidates")
