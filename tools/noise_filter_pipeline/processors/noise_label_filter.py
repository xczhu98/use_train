"""C3 无阈值 top-5 准入的 pipeline 适配与状态写回。

本 Processor 不实现策略矩阵：它只校验 C1/C2/执行身份，调用
``core.data.noise_admission_policy.evaluate_noise_admission``（top-5 Speech
veto + C1 分支语义决策，无阈值/calibration 依赖），再把
``accept_noise|accept_babble`` 映射为 ``passed``，把 ``reject`` 映射为
``rejected``。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

from core.data.noise_admission import validate_artifact_fingerprints
from core.data.noise_admission_policy import (
    AdmissionDecision,
    EVIDENCE_CONTRACT_VERSION,
    SCORE_CONTRACT,
    decision_to_admission_extra,
    evaluate_noise_admission,
    load_admission_policy,
    parse_c2_evidence,
)

from ..core.execution_contract import validate_stage_c_execution_contract
from ..core.manifest_io import AudioRecord, read_manifest_strict
from ..core.manifest_per_dataset import PerDatasetManifestWriter
from ..core.processor_base import Processor

logger = logging.getLogger(__name__)


def _reject_decision(
    reason: str,
    *,
    policy: Any,
) -> AdmissionDecision:
    """构造适配层拒绝决策；拒绝记录不产出 balance category。"""
    return AdmissionDecision(
        decision="reject",
        reason_codes=(reason,),
        admission_allowed=False,
        top5_hits=(),
        balance_category_mid=None,
        balance_category=None,
        balance_category_source=None,
        policy_version=policy.version,
        policy_sha256=policy.sha256,
        evidence_contract_version=EVIDENCE_CONTRACT_VERSION,
        score_contract=SCORE_CONTRACT,
        label_space_sha256=policy.label_space_sha256,
    )


class NoiseAdmissionProcessor(Processor):
    """C3：policy v4 + 无阈值 top-5 语义准入适配层，零模型成本。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, stage_name="C3")
        self.enable = config.get("enable", True)
        if type(self.enable) is not bool:
            raise ValueError("C3 enable 必须是严格 bool")
        # C3 只做无阈值 top-5 语义决策。旧 calibration/threshold 配置即使
        # 值为 null 也不得静默忽略，避免旧运行配置伪装成新契约。
        legacy_contract_keys = {
            "calibration_path",
            "calibration_identity",
            "voice_prob_threshold",
            "mid_thresholds",
            "thresholds",
        }
        configured_legacy_keys = sorted(legacy_contract_keys.intersection(config))
        if configured_legacy_keys:
            raise ValueError(
                "C3 不支持 calibration/threshold 配置: "
                + ", ".join(configured_legacy_keys)
            )

        repo_root = Path(__file__).resolve().parents[3]
        policy_path = config.get("policy_path")
        if policy_path and not os.path.isabs(policy_path):
            policy_path = str(repo_root / policy_path)
        self.policy = load_admission_policy(policy_path)
        self.policy_sha256 = self.policy.sha256

        upstream_fingerprints = config.get("artifact_fingerprints")
        self.upstream_fingerprints = (
            dict(upstream_fingerprints)
            if isinstance(upstream_fingerprints, Mapping)
            else None
        )
        c1_identity = config.get("c1_identity")
        c2_identity = config.get("c2_identity")
        self.c1_identity = dict(c1_identity) if isinstance(c1_identity, Mapping) else None
        self.c2_identity = dict(c2_identity) if isinstance(c2_identity, Mapping) else None
        self.execution_mode = config.get("execution_mode")
        self.publishable = config.get("publishable")
        if self.publishable is not None and type(self.publishable) is not bool:
            raise ValueError("C3 publishable 必须是严格 bool")
        self.evidence_contract_version = config.get(
            "evidence_contract_version", EVIDENCE_CONTRACT_VERSION
        )
        if self.evidence_contract_version != EVIDENCE_CONTRACT_VERSION:
            raise ValueError(
                "C3 config evidence_contract_version must be "
                f"{EVIDENCE_CONTRACT_VERSION}"
            )

    def _execution_contract(self, rec: AudioRecord) -> tuple[Optional[dict], bool, Optional[str]]:
        """核对 orchestrator 契约与 C1/C2 实际写入的 provenance。"""
        try:
            execution_mode, publishable = validate_stage_c_execution_contract(
                self.execution_mode,
                self.publishable,
                location="C3 config",
            )
        except ValueError:
            return None, False, "stage_c_execution_invalid"
        provenance = rec.provenance if isinstance(rec.provenance, Mapping) else {}
        observed = provenance.get("stage_c_execution")
        if not isinstance(observed, Mapping):
            return execution_mode, False, "stage_c_execution_missing"
        if dict(observed) != execution_mode:
            return execution_mode, False, "stage_c_execution_mismatch"
        # dry-run/disabled 是有效执行身份，只是不具备发布资格；仍允许 C3 产出
        # 审计裁决，发布链再根据 publishable=false fail-closed。
        return execution_mode, publishable, None

    def _validate_fingerprints(self, rec: AudioRecord) -> tuple[Optional[dict], Optional[str]]:
        """精确复验 C1/C2 来源 identity 与五项发布资产，不改写来源。"""
        if not isinstance(self.upstream_fingerprints, Mapping):
            return None, "artifact_fingerprints_missing"
        try:
            validated = validate_artifact_fingerprints(
                self.upstream_fingerprints, location="C3"
            )
        except ValueError:
            # 非发布 dry-run/disabled 身份允许模型/checkpoint 使用确定性 unset
            # 占位；policy 与 label-space 仍必须是真实 SHA。正式发布不走此分支。
            if self.publishable is not False:
                return None, "artifact_fingerprints_invalid"
            required = {
                "voice_policy_sha256",
                "audiotag_label_dict_sha256",
                "audiotag_checkpoint_sha256",
                "qwen_model_sha256",
                "c1_prompt_sha256",
            }
            if set(self.upstream_fingerprints) != required:
                return None, "artifact_fingerprints_invalid"
            validated = dict(self.upstream_fingerprints)
            optional_unset = {
                "audiotag_checkpoint_sha256",
                "qwen_model_sha256",
            }
            for key, value in validated.items():
                if key in optional_unset and value == f"unset:{key}":
                    continue
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)
                ):
                    return None, "artifact_fingerprints_invalid"
        if validated["voice_policy_sha256"] != self.policy_sha256:
            return None, "policy_hash_mismatch"

        c1_stage = rec.stages.get("C1")
        if not isinstance(c1_stage, Mapping) or not isinstance(self.c1_identity, Mapping):
            return None, "c1_identity_missing"
        expected_c1_keys = {"qwen_model_sha256", "c1_prompt_sha256"}
        if set(self.c1_identity) != expected_c1_keys:
            return None, "c1_identity_invalid"
        observed_c1 = {key: c1_stage.get(key) for key in expected_c1_keys}
        if observed_c1 != self.c1_identity:
            return None, "c1_identity_mismatch"

        c2_stage = rec.stages.get("C2")
        if not isinstance(c2_stage, Mapping) or not isinstance(self.c2_identity, Mapping):
            return None, "c2_identity_missing"
        expected_c2_keys = {
            "audiotag_label_dict_sha256",
            "audiotag_checkpoint_sha256",
            "evidence_contract_version",
            "score_contract",
            "class_count",
            "top_k",
        }
        if set(self.c2_identity) != expected_c2_keys:
            return None, "c2_identity_invalid"
        expected_c2 = {
            "label_space_sha256": self.c2_identity["audiotag_label_dict_sha256"],
            "audiotag_checkpoint_sha256": self.c2_identity[
                "audiotag_checkpoint_sha256"
            ],
            "evidence_contract_version": self.c2_identity[
                "evidence_contract_version"
            ],
            "score_contract": self.c2_identity["score_contract"],
            "class_count": self.c2_identity["class_count"],
            "top_k": self.c2_identity["top_k"],
        }
        observed_c2 = {key: c2_stage.get(key) for key in expected_c2}
        if observed_c2 != expected_c2:
            return None, "c2_identity_mismatch"
        if any(
            validated[key] != expected
            for key, expected in (
                ("qwen_model_sha256", self.c1_identity["qwen_model_sha256"]),
                ("c1_prompt_sha256", self.c1_identity["c1_prompt_sha256"]),
                (
                    "audiotag_label_dict_sha256",
                    self.c2_identity["audiotag_label_dict_sha256"],
                ),
                (
                    "audiotag_checkpoint_sha256",
                    self.c2_identity["audiotag_checkpoint_sha256"],
                ),
            )
        ):
            return None, "artifact_identity_mismatch"
        return validated, None

    def _decide(self, rec: AudioRecord) -> tuple[AdmissionDecision, Optional[dict]]:
        """执行 content 决策；身份/证据非法一律 fail-closed reject。

        返回 (decision, validated_fingerprints)。
        """
        c1_stage = rec.stages.get("C1")
        if not isinstance(c1_stage, Mapping):
            return (
                _reject_decision("c1_stage_missing", policy=self.policy),
                None,
            )
        intelligibility = c1_stage.get("intelligibility")
        if intelligibility not in ("noise", "babble", "speech"):
            return (
                _reject_decision("c1_intelligibility_invalid", policy=self.policy),
                None,
            )
        # 防御性路径同样保持 C1 speech 绝对终态，不允许被伪造的 C2 状态改写。
        if intelligibility == "speech":
            return (
                _reject_decision("c1_speech", policy=self.policy),
                None,
            )
        if c1_stage.get("status") != "passed":
            return (
                _reject_decision("c1_not_passed", policy=self.policy),
                None,
            )

        fingerprints, fp_error = self._validate_fingerprints(rec)
        if fp_error is not None:
            return (
                _reject_decision(fp_error, policy=self.policy),
                None,
            )
        c2_stage = rec.stages.get("C2")
        if not isinstance(c2_stage, Mapping):
            return (
                _reject_decision("c2_stage_missing", policy=self.policy),
                fingerprints,
            )
        if c2_stage.get("status") != "passed":
            return (
                _reject_decision("c2_not_passed", policy=self.policy),
                fingerprints,
            )

        # C2 source identity 额外含 checkpoint/top_k；共享 evidence parser 只接收
        # 事实 schema 字段，避免把来源绑定误解释为未知证据字段。
        evidence_identity = {
            key: c2_stage.get(key)
            for key in (
                "status",
                "evidence_contract_version",
                "score_contract",
                "class_count",
                "label_space_sha256",
            )
        }
        try:
            bundle = parse_c2_evidence(rec.labels, self.policy, evidence_identity)
        except ValueError:
            return (
                _reject_decision("c2_fact_invalid", policy=self.policy),
                fingerprints,
            )
        return (
            evaluate_noise_admission(
                c1_intelligibility=intelligibility,
                evidence=bundle,
                policy=self.policy,
            ),
            fingerprints,
        )

    def _extra(
        self,
        decision: AdmissionDecision,
        *,
        execution_mode: Optional[Mapping[str, str]],
        publishable: bool,
        fingerprints: Optional[Mapping[str, str]],
    ) -> Dict[str, Any]:
        return decision_to_admission_extra(
            decision,
            execution_mode=execution_mode,
            publishable=publishable,
            artifact_fingerprints=fingerprints,
            evidence_contract_version=self.evidence_contract_version,
        )

    def _process_record(self, rec: AudioRecord) -> AudioRecord:
        # 防御上游/旧版 manifest 的脏记录：labels/stages 为 None 时按空 dict 处理
        if rec.labels is None:
            rec.labels = {}
        if rec.stages is None:
            rec.stages = {}
        execution_mode, publishable, execution_error = self._execution_contract(rec)
        if execution_error is not None:
            decision = _reject_decision(
                execution_error, policy=self.policy
            )
            fingerprints = None
        else:
            decision, fingerprints = self._decide(rec)
        extra = self._extra(
            decision,
            execution_mode=execution_mode,
            publishable=publishable,
            fingerprints=fingerprints,
        )
        if decision.decision in ("accept_noise", "accept_babble"):
            rec.mark_stage(self.stage_name, "passed", extra=extra)
            rec.rejected_reason = None
        else:
            rec.mark_stage(self.stage_name, "rejected", extra=extra)
            rec.rejected_reason = (
                f"{self.stage_name.lower()}_{decision.reason_codes[0]}"
                if decision.reason_codes else f"{self.stage_name.lower()}_rejected"
            )
        return rec

    def process(
        self,
        input_manifest: str,
        output_dir: str,
        fingerprint_input: Optional[str] = None,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_manifest = os.path.join(output_dir, f"{self.stage_name.lower()}_manifest.jsonl")

        total = 0
        passed = 0
        rejected = 0

        def record_generator() -> Iterator[AudioRecord]:
            nonlocal total, passed, rejected
            for rec in read_manifest_strict(input_manifest, reject_duplicate_ids=True):
                total += 1
                # 脏记录防御：labels/stages 为 None 时按空 dict 处理，
                # 否则 should_skip/mark_stage 会在空值上崩溃杀停整个 stage
                if rec.labels is None:
                    rec.labels = {}
                if rec.stages is None:
                    rec.stages = {}
                # C3 是发布前最终门禁：即使输入携带旧 C3 passed，也必须按本次
                # execution contract 重新裁决，禁止历史状态绕过新增门禁。
                if not self.enable:
                    # C3 disabled 无法形成最终准入证据，必须 fail-closed，禁止透传发布。
                    execution_mode, _, _ = self._execution_contract(rec)
                    decision = _reject_decision("c3_disabled", policy=self.policy)
                    extra = self._extra(
                        decision,
                        execution_mode=execution_mode,
                        publishable=False,
                        fingerprints=None,
                    )
                    extra["enabled"] = False
                    rec.mark_stage(
                        self.stage_name,
                        "rejected",
                        extra=extra,
                    )
                    rec.rejected_reason = "c3_disabled"
                    rejected += 1
                    yield rec
                    continue
                processed = self._process_record(rec)
                if processed.is_stage_passed(self.stage_name):
                    passed += 1
                else:
                    rejected += 1
                yield processed

        with PerDatasetManifestWriter(output_manifest, batch_size=1000) as writer:
            for rec in record_generator():
                writer.append(rec)
        logger.info(
            "%s 噪声准入决策完成：%d 条记录，通过 %d，拒绝 %d，输出 %s",
            self.stage_name,
            total,
            passed,
            rejected,
            output_manifest,
        )
        return output_dir