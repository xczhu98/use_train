"""C3 全 527 MID 策略与无阈值 top-5 语义裁决的单一事实源。

C2 只持久化固定 top-5 事实（``audioset_top5``，排序自完整 527 sigmoid
内部排名）；C3 绑定 label space、显式 MID 三分类，然后按冻结语义输出
``reject | accept_noise | accept_babble``：

- top-5 中任一 ``speech`` MID 都触发 ``reject``；
- C1=noise → ``accept_noise``；C1=babble → ``accept_babble``；
- ``babble`` MID 不参与决策；
- ``balance_category`` 取 top-5 中最高排名的 ``noise`` MID，无则回退
  ``/m/096m7z`` / ``Noise``。

决策无阈值、无 calibration、attestation 或 approval bundle 依赖；旧证据和
旧 C3 字段一律 fail-closed。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple

ADMISSION_POLICY_VERSION = 4
EVIDENCE_CONTRACT_VERSION = 5
CLASSIFIER_LABEL_COUNT = 527
AUDIOTAG_TOP_K = 5
SCORE_CONTRACT = "audioset_527_independent_sigmoid_v1"
MID_CLASSES: Tuple[str, ...] = ("speech", "babble", "noise")
GENERIC_NOISE_MID = "/m/096m7z"
LEGACY_C2_LABEL_FIELDS = frozenset(
    {
        "topk",
        "audioset",
        "audioset_mid",
        "audioset_scores",
        "audioset_top10",
        "voice_max_label",
        "voice_max_prob",
        "babble_max_label",
        "babble_max_prob",
        "always_reject_max_label",
        "always_reject_max_prob",
        "background_speech_max_label",
        "background_speech_max_prob",
        "vocal_max_mid",
        "vocal_max_label",
        "vocal_max_prob",
        "foreground_speech_max_mid",
        "foreground_speech_max_label",
        "foreground_speech_max_prob",
        "background_speech_max_mid",
        "auxiliary_reject_max_mid",
        "auxiliary_reject_max_label",
        "auxiliary_reject_max_prob",
        "background_speech_sustain_ratio",
        "background_speech_longest_run",
        "window_count",
    }
)
LEGACY_C3_FIELDS = frozenset(
    {
        "admission_calibration_version",
        "admission_calibration_sha256",
        "calibration_approved",
        "approved",
        "approval_blockers",
        "review_required",
        "approval_bundle",
        "approval_bundle_sha256",
        "attestation",
        "attestation_sha256",
        "calibration_status",
        "content_kind",
        "margin",
        "max_risk",
        "quarantine",
        "reason_codes",
        "risk",
        "sustain",
        "threshold_hits",
    }
)
# labels 白名单：audioset_top5 是 C2 事实字段，original 是 Stage A 来源标签，
# final_category 是 B4 策略过滤写入的类别（noise/rejected 等），三者都是当前
# 合法上游产物。注意：B1 hive 模式还会写 hive_output_label_key（默认
# hive_chunk_source），启用 hive 模式前必须先把该键纳入白名单，否则 C3/D 的
# 严格校验会全量拒绝。
C2_LABEL_FIELDS = frozenset({"audioset_top5", "original", "final_category"})
C2_STAGE_FIELDS = frozenset(
    {
        "status",
        "timestamp",
        # C1 拒绝时 C2 跳过，仅写 status/timestamp/reason（无事实字段）
        "reason",
        "evidence_contract_version",
        "score_contract",
        "class_count",
        "label_space_sha256",
        "audiotag_checkpoint_sha256",
        "top_k",
    }
)
C3_STAGE_FIELDS = frozenset(
    {
        "status",
        "timestamp",
        "admission_decision",
        "admission_reason_codes",
        "reason",
        "admission_allowed",
        "top5_hits",
        "balance_category_mid",
        "balance_category",
        "balance_category_source",
        "admission_policy_version",
        "admission_policy_sha256",
        "evidence_contract_version",
        "score_contract",
        "label_space_sha256",
        "artifact_fingerprints",
        "execution_mode",
        "publishable",
    }
)

AdmissionDecisionType = Literal["reject", "accept_noise", "accept_babble"]
MidClass = Literal["speech", "babble", "noise"]


def _validated_regular_file_path(
    path: str | Path,
    *,
    field: str,
) -> Path:
    """校验本地输入路径为无符号链接组件的 regular file。"""
    candidate = Path(path)
    absolute = Path.cwd() / candidate if not candidate.is_absolute() else candidate
    absolute = Path(os.path.abspath(absolute))
    components = absolute.parts
    current = Path(components[0])
    for component in components[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            raise FileNotFoundError(f"{field} not found: {absolute}") from None
        except OSError as exc:
            raise ValueError(f"{field} path cannot be checked: {current}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise ValueError(f"{field} path must not be a symlink: {current}")
        if current != absolute and not stat.S_ISDIR(mode):
            raise ValueError(f"{field} path component must be a directory: {current}")
    try:
        mode = absolute.lstat().st_mode
    except FileNotFoundError:
        raise FileNotFoundError(f"{field} not found: {absolute}") from None
    if not stat.S_ISREG(mode):
        raise ValueError(f"{field} must be a regular file: {absolute}")
    return absolute


def sha256_file(path: str | Path) -> str:
    """流式计算文件 SHA-256（与 core.data.noise_admission 同语义）。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def default_voice_policy_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "tools/noise_filter_pipeline/config/ontology/audiotag_voice_policy.json"
    )


def default_label_dict_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "third_party_scripts/icefall_audiotaq/data/class_labels_indices.csv"
    )


# ---------------------------------------------------------------------------
# 严格 label metadata（527 类）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LabelMetadata:
    """527 类 classifier label 的严格索引表。"""

    index_to_mid: Mapping[int, str] = field(default_factory=dict)
    mid_to_index: Mapping[str, int] = field(default_factory=dict)
    mid_to_label: Mapping[str, str] = field(default_factory=dict)
    label_to_mid: Mapping[str, str] = field(default_factory=dict)

    @property
    def class_count(self) -> int:
        return len(self.index_to_mid)

    def label_of(self, mid: str) -> str:
        try:
            return self.mid_to_label[mid]
        except KeyError:
            raise ValueError(f"unknown classifier MID {mid!r}") from None

    def index_of(self, mid: str) -> int:
        try:
            return self.mid_to_index[mid]
        except KeyError:
            raise ValueError(f"unknown classifier MID {mid!r}") from None


def load_label_metadata(label_dict_path: str | Path | None = None) -> LabelMetadata:
    """严格读取 AudioSet class_labels_indices.csv。

    要求：
    - header 精确为 index,mid,display_name；
    - 行数为 527，index 连续 0..526，不允许重复/负数/跳跃；
    - mid 与 display_name 均唯一；空值拒绝。
    """
    path = _validated_regular_file_path(
        label_dict_path or default_label_dict_path(), field="label dict"
    )
    rows: list[tuple[int, str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        import csv

        reader = csv.DictReader(handle)
        header = [str(c).strip() for c in (reader.fieldnames or [])]
        if header != ["index", "mid", "display_name"]:
            raise ValueError(
                f"label dict header must be index,mid,display_name, got {header}"
            )
        for line_no, row in enumerate(reader, start=2):
            try:
                index_value = row["index"]
                mid_value = row["mid"]
                label_value = row["display_name"]
                if not isinstance(index_value, str) or not index_value.strip():
                    raise ValueError("index must be a non-empty string")
                if not isinstance(mid_value, str) or not mid_value.strip():
                    raise ValueError("mid must be a non-empty string")
                if not isinstance(label_value, str) or not label_value.strip():
                    raise ValueError("display_name must be a non-empty string")
                index = int(index_value.strip())
                mid = mid_value.strip()
                label = label_value.strip()
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"label dict invalid row {line_no}: {dict(row)!r} ({exc})"
                ) from exc
            rows.append((index, mid, label))
    if len(rows) != CLASSIFIER_LABEL_COUNT:
        raise ValueError(
            f"label dict must contain exactly {CLASSIFIER_LABEL_COUNT} rows, got {len(rows)}"
        )
    indices = [r[0] for r in rows]
    if sorted(indices) != list(range(CLASSIFIER_LABEL_COUNT)):
        raise ValueError(
            f"label dict indices must be contiguous 0..{CLASSIFIER_LABEL_COUNT - 1}, "
            f"got {sorted(indices)}"
        )
    mids = [r[1] for r in rows]
    labels = [r[2] for r in rows]
    if len(set(mids)) != len(mids):
        raise ValueError("label dict contains duplicate mids")
    if len(set(labels)) != len(labels):
        raise ValueError("label dict contains duplicate display names")
    return LabelMetadata(
        index_to_mid={idx: mid for idx, mid, _ in rows},
        mid_to_index={mid: idx for idx, mid, _ in rows},
        mid_to_label={mid: label for _, mid, label in rows},
        label_to_mid={label: mid for _, mid, label in rows},
    )


# ---------------------------------------------------------------------------
# Policy v4
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AdmissionPolicy:
    """显式覆盖 canonical 527 MID 的三分类策略。"""

    version: int
    sha256: str
    label_space_sha256: str
    mid_to_class: Mapping[str, MidClass] = field(default_factory=dict)
    index_to_mid: Mapping[int, str] = field(default_factory=dict)
    mid_to_index: Mapping[str, int] = field(default_factory=dict)
    mid_to_label: Mapping[str, str] = field(default_factory=dict)

    @property
    def class_count(self) -> int:
        return len(self.index_to_mid)

    def class_of(self, mid: str) -> MidClass:
        try:
            return self.mid_to_class[mid]
        except KeyError:
            raise ValueError(f"unknown classifier MID {mid!r}") from None

    def label_of(self, mid: str) -> str:
        try:
            return self.mid_to_label[mid]
        except KeyError:
            raise ValueError(f"unknown classifier MID {mid!r}") from None


def _load_json_object(path: Path, *, name: str) -> Dict[str, Any]:
    validated = _validated_regular_file_path(path, field=name)

    def _unique_object(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = item
        return result

    with validated.open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object: {validated}")
    return value


def load_admission_policy(
    policy_path: str | Path | None = None,
    label_dict_path: str | Path | None = None,
) -> AdmissionPolicy:
    """严格解析 policy v4，并验证 527 MID exact-set 与特殊 MID 约束。"""
    path = _validated_regular_file_path(
        policy_path or default_voice_policy_path(), field="voice policy"
    )
    raw = _load_json_object(path, name="voice policy")
    required = {"admission_policy_version", "label_space_sha256", "mid_classes"}
    missing = required - set(raw)
    unknown = set(raw) - required
    if missing or unknown:
        raise ValueError(
            f"invalid voice policy fields: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    if type(raw["admission_policy_version"]) is not int or raw[
        "admission_policy_version"
    ] != ADMISSION_POLICY_VERSION:
        raise ValueError(
            f"admission_policy_version must be {ADMISSION_POLICY_VERSION}"
        )

    label_path = _validated_regular_file_path(
        label_dict_path or default_label_dict_path(), field="label dict"
    )
    metadata = load_label_metadata(label_path)
    label_space_sha256 = _sha256_digest(
        raw["label_space_sha256"], field="label_space_sha256", location=str(path)
    )
    actual_label_sha256 = sha256_file(label_path)
    if label_space_sha256 != actual_label_sha256:
        raise ValueError(
            "voice policy label_space_sha256 does not match canonical label dict"
        )
    mid_classes = raw["mid_classes"]
    if not isinstance(mid_classes, dict):
        raise ValueError("voice policy mid_classes must be an object")
    known = set(metadata.mid_to_index)
    if set(mid_classes) != known:
        raise ValueError(
            "voice policy mid_classes must cover exactly canonical 527 MID: "
            f"missing={sorted(known - set(mid_classes))}, "
            f"unknown={sorted(set(mid_classes) - known)}"
        )
    invalid = {
        mid: value for mid, value in mid_classes.items() if value not in MID_CLASSES
    }
    if invalid:
        raise ValueError(f"voice policy contains invalid MID classes: {invalid}")

    frozen = {mid: mid_classes[mid] for mid in metadata.mid_to_index}
    required_classes = {
        "/m/0261r1": "speech",
        "/m/07rkbfh": "babble",
        "/m/07qfr4h": "babble",
        "/m/03qtwd": "babble",
        "/m/04rlf": "noise",
        "/m/025td0t": "noise",
        "/t/dd00127": "noise",
        "/m/07c52": "speech",
        "/m/06bz3": "speech",
        "/m/07hvw1": "speech",
    }
    mismatched = {
        mid: (frozen[mid], expected)
        for mid, expected in required_classes.items()
        if frozen[mid] != expected
    }
    if mismatched:
        raise ValueError(f"voice policy violates frozen MID classes: {mismatched}")

    return AdmissionPolicy(
        version=ADMISSION_POLICY_VERSION,
        sha256=sha256_file(path),
        label_space_sha256=label_space_sha256,
        mid_to_class=frozen,
        index_to_mid=dict(metadata.index_to_mid),
        mid_to_index=dict(metadata.mid_to_index),
        mid_to_label=dict(metadata.mid_to_label),
    )


def _finite_prob(value: Any, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{field} must be a finite probability in [0, 1]")
    return float(value)


def _sha256_digest(value: Any, *, field: str, location: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{location}: {field} must be a lowercase SHA-256 hex digest")
    return value


# ---------------------------------------------------------------------------
# C2 evidence v5 解析（固定 top-5）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvidenceTopItem:
    index: int
    mid: str
    label: str
    prob: float


@dataclass(frozen=True)
class EvidenceBundle:
    """固定 top-5 证据：排序在 C2 表单层基于完整 527 分数内部完成。"""

    top5: Tuple[EvidenceTopItem, ...]
    evidence_contract_version: int
    score_contract: str
    class_count: int
    label_space_sha256: str


def _validate_top5_ordered(
    items: Sequence[Tuple[int, float]],
    *,
    location: str,
) -> None:
    """校验 top-5 排序契约：prob 降序、并列时 index 升序。"""
    previous_key: Optional[Tuple[float, int]] = None
    for position, (index, prob) in enumerate(items):
        key = (-prob, index)
        if previous_key is not None and key < previous_key:
            raise ValueError(
                f"{location}[{position}] 违反排序契约 "
                "(prob 降序、并列按 index 升序)"
            )
        previous_key = key


def parse_c2_evidence(
    labels: Mapping[str, Any],
    policy: AdmissionPolicy,
    c2_stage: Optional[Mapping[str, Any]] = None,
) -> EvidenceBundle:
    """严格验证固定 top-5（恰好 5 项）与可选 C2 identity。

    不再解析完整 527 分数数组：排序正确性、label-space 一致性与 prob 范围
    均在 top-5 内部校验；全局 top-5 事实由 C2 表单层基于完整 527 sigmoid
    排序保证。
    """
    if not isinstance(labels, Mapping):
        raise ValueError("C2 labels must be an object")
    legacy_fields = LEGACY_C2_LABEL_FIELDS & set(labels)
    if legacy_fields:
        raise ValueError(
            f"legacy C2 label fields are forbidden: {sorted(legacy_fields)}"
        )
    unknown_label_fields = set(labels) - C2_LABEL_FIELDS
    if unknown_label_fields:
        raise ValueError(
            f"invalid C2 labels fields: unknown={sorted(unknown_label_fields)}"
        )
    top_raw = labels.get("audioset_top5")
    if not isinstance(top_raw, list) or len(top_raw) != AUDIOTAG_TOP_K:
        raise ValueError(
            f"labels.audioset_top5 must contain exactly {AUDIOTAG_TOP_K} items"
        )
    top5: list[EvidenceTopItem] = []
    seen_indices: set[int] = set()
    for position, item in enumerate(top_raw):
        if not isinstance(item, Mapping) or set(item) != {"index", "mid", "label", "prob"}:
            raise ValueError(
                f"labels.audioset_top5[{position}] must contain index/mid/label/prob"
            )
        index = item["index"]
        if type(index) is not int or not 0 <= index < CLASSIFIER_LABEL_COUNT:
            raise ValueError(
                f"labels.audioset_top5[{position}].index must be within 0..526"
            )
        if index in seen_indices:
            raise ValueError(
                f"labels.audioset_top5[{position}].index {index} is duplicated"
            )
        seen_indices.add(index)
        mid = item["mid"]
        label = item["label"]
        prob = _finite_prob(
            item["prob"], field=f"labels.audioset_top5[{position}].prob"
        )
        if mid != policy.index_to_mid[index] or label != policy.label_of(mid):
            raise ValueError(
                f"labels.audioset_top5[{position}] does not match canonical label space"
            )
        top5.append(EvidenceTopItem(index=index, mid=mid, label=label, prob=prob))
    _validate_top5_ordered(
        [(item.index, item.prob) for item in top5],
        location="labels.audioset_top5",
    )

    if not isinstance(c2_stage, Mapping):
        raise ValueError("stages.C2 identity is required and must be an object")
    identity = c2_stage
    unknown_c2_fields = set(identity) - C2_STAGE_FIELDS
    if unknown_c2_fields:
        raise ValueError(
            f"invalid stages.C2 fields: unknown={sorted(unknown_c2_fields)}"
        )
    evidence_version = identity.get("evidence_contract_version")
    score_contract = identity.get("score_contract")
    class_count = identity.get("class_count")
    label_sha = identity.get("label_space_sha256")
    if type(evidence_version) is not int or evidence_version != EVIDENCE_CONTRACT_VERSION:
        raise ValueError(
            f"C2 evidence_contract_version must be {EVIDENCE_CONTRACT_VERSION}"
        )
    if score_contract != SCORE_CONTRACT:
        raise ValueError(f"C2 score_contract must be {SCORE_CONTRACT!r}")
    if type(class_count) is not int or class_count != CLASSIFIER_LABEL_COUNT:
        raise ValueError(f"C2 class_count must be {CLASSIFIER_LABEL_COUNT}")
    if label_sha != policy.label_space_sha256:
        raise ValueError("C2 label_space_sha256 does not match admission policy")
    if "status" in identity and identity["status"] != "passed":
        raise ValueError("C2 status must be 'passed'")
    if "top_k" in identity and (
        type(identity["top_k"]) is not int or identity["top_k"] != AUDIOTAG_TOP_K
    ):
        raise ValueError(f"C2 top_k must be exact int {AUDIOTAG_TOP_K}")
    if "audiotag_checkpoint_sha256" in identity:
        checkpoint_sha = identity["audiotag_checkpoint_sha256"]
        if checkpoint_sha != "unset:audiotag_checkpoint_sha256":
            _sha256_digest(
                checkpoint_sha,
                field="audiotag_checkpoint_sha256",
                location="stages.C2",
            )
    return EvidenceBundle(
        top5=tuple(top5),
        evidence_contract_version=evidence_version,
        score_contract=score_contract,
        class_count=class_count,
        label_space_sha256=label_sha,
    )


# ---------------------------------------------------------------------------
# C3 无阈值 top-5 语义决策与 category selector
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Top5Hit:
    """top-5 中的单条命中审计项（kind 为 policy 语义三分类）。"""

    index: int
    mid: str
    label: str
    kind: MidClass
    prob: float


@dataclass(frozen=True)
class AdmissionDecision:
    decision: AdmissionDecisionType
    reason_codes: Tuple[str, ...]
    admission_allowed: bool
    top5_hits: Tuple[Top5Hit, ...]
    balance_category_mid: Optional[str]
    balance_category: Optional[str]
    balance_category_source: Optional[str]
    policy_version: int
    policy_sha256: str
    evidence_contract_version: int
    score_contract: str
    label_space_sha256: str


def _top5_hits_from_evidence(
    evidence: EvidenceBundle, policy: AdmissionPolicy
) -> Tuple[Top5Hit, ...]:
    return tuple(
        Top5Hit(
            index=item.index,
            mid=item.mid,
            label=item.label,
            kind=policy.class_of(item.mid),
            prob=item.prob,
        )
        for item in evidence.top5
    )


def _validate_evidence_bundle(
    evidence: EvidenceBundle,
    policy: AdmissionPolicy,
) -> None:
    """防御性复验 bundle，避免绕过 ``parse_c2_evidence`` 注入坏 top-5。"""
    if not isinstance(evidence, EvidenceBundle):
        raise ValueError("C2 evidence must be an EvidenceBundle")
    if evidence.evidence_contract_version != EVIDENCE_CONTRACT_VERSION:
        raise ValueError("C2 evidence contract version is stale")
    if evidence.score_contract != SCORE_CONTRACT:
        raise ValueError("C2 score contract is unsupported")
    if type(evidence.class_count) is not int or evidence.class_count != CLASSIFIER_LABEL_COUNT:
        raise ValueError("C2 evidence class count is invalid")
    if evidence.label_space_sha256 != policy.label_space_sha256:
        raise ValueError("C2 evidence does not bind current label space")
    if not isinstance(evidence.top5, tuple) or len(evidence.top5) != AUDIOTAG_TOP_K:
        raise ValueError("C2 evidence must contain exactly five top-ranked items")

    seen_indices: set[int] = set()
    ordered: list[Tuple[int, float]] = []
    for position, item in enumerate(evidence.top5):
        location = f"C2 evidence.top5[{position}]"
        if not isinstance(item, EvidenceTopItem):
            raise ValueError(f"{location} must be an EvidenceTopItem")
        index = item.index
        if type(index) is not int or not 0 <= index < CLASSIFIER_LABEL_COUNT:
            raise ValueError(f"{location}.index must be within 0..526")
        if index in seen_indices:
            raise ValueError(f"{location}.index {index} is duplicated")
        seen_indices.add(index)
        prob = _finite_prob(item.prob, field=f"{location}.prob")
        try:
            expected_mid = policy.index_to_mid[index]
        except KeyError:
            raise ValueError(f"{location}.index is absent from admission policy") from None
        if item.mid != expected_mid or item.label != policy.label_of(expected_mid):
            raise ValueError(f"{location} does not match canonical label space")
        ordered.append((index, prob))
    _validate_top5_ordered(ordered, location="C2 evidence.top5")


def _decision(
    decision: AdmissionDecisionType,
    reason: str,
    *,
    top5_hits: Tuple[Top5Hit, ...],
    policy: AdmissionPolicy,
    balance_hit: Optional[Top5Hit] = None,
    fallback_noise: bool = False,
) -> AdmissionDecision:
    admitted = decision in ("accept_noise", "accept_babble")
    if fallback_noise:
        if policy.class_of(GENERIC_NOISE_MID) != "noise":
            raise ValueError("generic fallback MID must map to noise")
        balance_mid = GENERIC_NOISE_MID
        balance_label = policy.label_of(balance_mid)
        balance_source = "generic_noise_fallback"
    elif balance_hit is not None:
        balance_mid = balance_hit.mid
        balance_label = balance_hit.label
        balance_source = "noise_top5_hit"
    else:
        balance_mid = balance_label = balance_source = None
    return AdmissionDecision(
        decision=decision,
        reason_codes=(reason,),
        admission_allowed=admitted,
        top5_hits=top5_hits,
        balance_category_mid=balance_mid,
        balance_category=balance_label,
        balance_category_source=balance_source,
        policy_version=policy.version,
        policy_sha256=policy.sha256,
        evidence_contract_version=EVIDENCE_CONTRACT_VERSION,
        score_contract=SCORE_CONTRACT,
        label_space_sha256=policy.label_space_sha256,
    )


def evaluate_noise_admission(
    *,
    c1_intelligibility: str,
    evidence: Optional[EvidenceBundle],
    policy: AdmissionPolicy,
) -> AdmissionDecision:
    """按 top-5 speech veto 与 C1 分支计算唯一 C3 结果。"""
    if c1_intelligibility not in ("noise", "babble", "speech"):
        raise ValueError(
            f"c1_intelligibility must be noise/babble/speech, got {c1_intelligibility!r}"
        )

    # C1 speech 是终态；防御性重算不得要求或消费 C2 evidence。
    if c1_intelligibility == "speech":
        return _decision(
            "reject", "c1_speech", top5_hits=(), policy=policy
        )
    if evidence is None:
        raise ValueError("C2 evidence is required for C1 noise/babble")
    _validate_evidence_bundle(evidence, policy)

    top5_hits = _top5_hits_from_evidence(evidence, policy)
    # S5 语义 veto：top-5 中存在任何 speech MID 即 reject，B 不参与决策。
    speech_hits = tuple(hit for hit in top5_hits if hit.kind == "speech")
    if speech_hits:
        return _decision(
            "reject", "speech_mid_in_top5", top5_hits=top5_hits, policy=policy
        )
    noise_hits = tuple(hit for hit in top5_hits if hit.kind == "noise")
    balance_hit = (
        min(noise_hits, key=lambda hit: (-hit.prob, hit.index))
        if noise_hits
        else None
    )

    if c1_intelligibility == "noise":
        return _decision(
            "accept_noise",
            "admitted_noise",
            top5_hits=top5_hits,
            policy=policy,
            balance_hit=balance_hit,
            fallback_noise=balance_hit is None,
        )
    return _decision(
        "accept_babble",
        "admitted_babble",
        top5_hits=top5_hits,
        policy=policy,
        balance_hit=balance_hit,
        fallback_noise=balance_hit is None,
    )


# ---------------------------------------------------------------------------
# C3 admission 序列化与发布链复验
# ---------------------------------------------------------------------------
def decision_to_admission_extra(
    decision: AdmissionDecision,
    *,
    execution_mode: Optional[Mapping[str, str]],
    publishable: bool,
    artifact_fingerprints: Optional[Mapping[str, str]],
    evidence_contract_version: int = EVIDENCE_CONTRACT_VERSION,
) -> Dict[str, Any]:
    """把决策序列化为 C3 admission stage extra（单一事实来源）。"""
    if type(evidence_contract_version) is not int:
        raise ValueError("evidence_contract_version must be an integer")
    if evidence_contract_version != EVIDENCE_CONTRACT_VERSION:
        raise ValueError(
            "evidence_contract_version must be "
            f"{EVIDENCE_CONTRACT_VERSION}"
        )
    if not isinstance(publishable, bool):
        raise ValueError("publishable must be a boolean")
    return {
        "admission_decision": decision.decision,
        "admission_reason_codes": list(decision.reason_codes),
        "reason": decision.reason_codes[0] if decision.reason_codes else None,
        "admission_allowed": decision.admission_allowed,
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
        "balance_category_mid": decision.balance_category_mid,
        "balance_category": decision.balance_category,
        "balance_category_source": decision.balance_category_source,
        "admission_policy_version": decision.policy_version,
        "admission_policy_sha256": decision.policy_sha256,
        "evidence_contract_version": evidence_contract_version,
        "score_contract": decision.score_contract,
        "label_space_sha256": decision.label_space_sha256,
        "artifact_fingerprints": dict(artifact_fingerprints)
        if artifact_fingerprints
        else None,
        "execution_mode": dict(execution_mode) if execution_mode else None,
        "publishable": publishable,
    }


def validate_admission_contract(
    *,
    c1_intelligibility: str,
    c2_labels: Mapping[str, Any],
    admission_stage: Mapping[str, Any],
    policy: AdmissionPolicy,
    require_accepted: bool,
    location: str,
    c2_stage: Optional[Mapping[str, Any]] = None,
) -> AdmissionDecision:
    """复验 C3 序列化字段等于无阈值 top-5 重算结果。"""
    if not isinstance(admission_stage, Mapping):
        raise ValueError(f"{location}: stages.C3 must be an object")
    legacy_fields = LEGACY_C3_FIELDS & set(admission_stage)
    if legacy_fields:
        raise ValueError(
            f"{location}: legacy C3 fields are forbidden: {sorted(legacy_fields)}"
        )
    unknown_c3_fields = set(admission_stage) - C3_STAGE_FIELDS
    if unknown_c3_fields:
        raise ValueError(
            f"{location}: invalid stages.C3 fields: unknown={sorted(unknown_c3_fields)}"
        )
    if "status" in admission_stage and admission_stage["status"] != "passed":
        raise ValueError(f"{location}: stages.C3.status must be 'passed'")
    bundle = (
        None
        if c1_intelligibility == "speech"
        else parse_c2_evidence(c2_labels, policy, c2_stage)
    )
    decision = evaluate_noise_admission(
        c1_intelligibility=c1_intelligibility,
        evidence=bundle,
        policy=policy,
    )
    for key in ("artifact_fingerprints", "execution_mode", "publishable"):
        if key not in admission_stage:
            raise ValueError(f"{location}: stages.C3.{key} is required")
    artifacts = admission_stage["artifact_fingerprints"]
    if artifacts is not None:
        if not isinstance(artifacts, Mapping):
            raise ValueError(f"{location}: stages.C3.artifact_fingerprints must be an object")
        policy_sha = artifacts.get("voice_policy_sha256")
        if policy_sha != policy.sha256:
            raise ValueError(
                f"{location}: stages.C3.artifact_fingerprints.voice_policy_sha256 "
                "must equal current policy.sha256"
            )
        label_sha = artifacts.get("audiotag_label_dict_sha256")
        if label_sha != policy.label_space_sha256:
            raise ValueError(
                f"{location}: stages.C3.artifact_fingerprints."
                "audiotag_label_dict_sha256 must equal current "
                "policy.label_space_sha256"
            )
    execution_mode = admission_stage["execution_mode"]
    if execution_mode is not None and not isinstance(execution_mode, Mapping):
        raise ValueError(f"{location}: stages.C3.execution_mode must be an object or null")
    if not isinstance(admission_stage["publishable"], bool):
        raise ValueError(f"{location}: stages.C3.publishable must be a boolean")
    extra = decision_to_admission_extra(
        decision,
        execution_mode=None,
        publishable=False,
        artifact_fingerprints=None,
    )
    for key in (
        "admission_decision",
        "admission_reason_codes",
        "reason",
        "admission_allowed",
        "top5_hits",
        "balance_category_mid",
        "balance_category",
        "balance_category_source",
        "admission_policy_version",
        "admission_policy_sha256",
        "evidence_contract_version",
        "score_contract",
        "label_space_sha256",
    ):
        if key not in admission_stage:
            raise ValueError(f"{location}: stages.C3.{key} is required")
        written = admission_stage[key]
        if written != extra[key]:
            raise ValueError(
                f"{location}: stages.C3.{key} {written!r} does not match "
                f"recomputed value {extra[key]!r}"
            )
    if require_accepted and decision.decision not in ("accept_noise", "accept_babble"):
        raise ValueError(
            f"{location}: C3 decision {decision.decision!r} is not publishable "
            "(require_accepted=True)"
        )
    return decision
