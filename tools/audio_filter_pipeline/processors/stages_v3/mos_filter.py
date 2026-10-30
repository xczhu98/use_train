"""Artifact-backed Generation-3 Step4 MOS threshold filtering."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Optional

from ...config import PipelineConfig
from ..mos import mos_model_passed, normalize_selectable_mos_models
from ...utils.artifacts import ArtifactIndex, PipelineStep
from ...utils.identity_index import unique_identity_index
from .base import ArtifactStage, StageRunContext
from .segment import STEP2_ROW_FIELDS
from .select import STEP3_ROW_FIELDS, Step3MosSelectionStage


STEP4_ROW_FIELDS = STEP3_ROW_FIELDS + ("mos_selection",)


class Step4MosFilterStage(ArtifactStage):
    """消费 Step3 分数并按当前配置策略生成 MOS 筛选决策，不运行模型。

    当前数据集可达策略的阈值进入 config digest：改本数据集阈值后 Step4
    artifact 自动失效需重算，而 Step3 已打分结果保持可复用。
    """

    step = PipelineStep.STEP4
    upstream_step = PipelineStep.STEP3

    def __init__(self, *, dataset_id: str | None = None) -> None:
        if dataset_id is not None and (
            type(dataset_id) is not str or not dataset_id.strip()
        ):
            raise ValueError("Step4 dataset_id must be a non-empty string or None")
        self._dataset_id = dataset_id

    def _validate_contract(self, context: StageRunContext) -> None:
        super()._validate_contract(context)
        if self._dataset_id is not None and self._dataset_id != context.dataset_id:
            raise ValueError("Step4 stage dataset_id does not match context")

    def config_payload(self, config: PipelineConfig) -> Mapping[str, Any]:
        selection = {
            "resolver": "PipelineConfig.resolve_selection_policy",
            "scorer_execution": False,
        }
        if self._dataset_id is None:
            selection["selection_policies"] = config.to_dict()["selection_policies"]
        else:
            config.dataset_spec(self._dataset_id)
            selection["dataset_binding"] = config.dataset_mos_decision_binding(
                self._dataset_id
            )
        return {
            "row_contract": {"version": 4, "fields": list(STEP4_ROW_FIELDS)},
            "upstream_row_contract": {"version": 4, "fields": list(STEP3_ROW_FIELDS)},
            "selection": selection,
            "waveform_policy": {"read": False, "write": False},
        }

    @classmethod
    def _validate_policy(cls, row: Mapping[str, Any], context: StageRunContext) -> Any:
        policy = context.config.resolve_selection_policy(
            row["dataset_id"],
            row["language"],
            source_relative_path=row["source_relative_path"],
        )
        if (
            policy.category != row["category"]
            or policy.quality_profile != row["quality_profile"]
        ):
            raise ValueError("Step3 row identity does not match resolved selection policy")
        projection = context.config.mos_scoring_projection(
            row["dataset_id"],
            row["language"],
            source_relative_path=row["source_relative_path"],
        )
        models = normalize_selectable_mos_models(row["mos_models"])
        if (
            models != policy.mos_models
            or row["mos_scoring_policy_hash"] != projection["scoring_policy_hash"]
        ):
            raise ValueError("Step3 row MOS scoring projection does not match current policy")
        return policy

    def produce_rows(
        self,
        context: StageRunContext,
        upstream_index: Optional[ArtifactIndex],
        upstream_rows: Iterable[Mapping[str, Any]],
    ) -> Iterable[Mapping[str, Any]]:
        if upstream_index is None or upstream_index.step != PipelineStep.STEP3.value:
            raise ValueError("Step4 MOS filter requires a Step3 upstream artifact")
        # 重复检测走临时 sqlite：流式设计下不累积全量 segment_id。
        with unique_identity_index() as reserve_identity:
            for upstream_row in upstream_rows:
                if not isinstance(upstream_row, Mapping):
                    raise ValueError("Step3 upstream row must be a mapping")
                if set(upstream_row) != set(STEP3_ROW_FIELDS):
                    raise ValueError("Step3 upstream row fields do not match the exact contract")
                step2_fields = {field: upstream_row[field] for field in STEP2_ROW_FIELDS}
                Step3MosSelectionStage._validate_upstream_row(step2_fields, context.dataset_id)
                if upstream_row["language"] != context.config.dataset_spec(context.dataset_id).language:
                    raise ValueError("Step3 upstream row language does not exactly match dataset policy identity")
                if not reserve_identity("segment_id", upstream_row["segment_id"]):
                    raise ValueError("duplicate segment_id in Step3 artifact")
                models = normalize_selectable_mos_models(upstream_row["mos_models"])
                scores = Step3MosSelectionStage._validate_scores(
                    upstream_row["mos_scores"], models
                )
                policy = self._validate_policy(upstream_row, context)
                passed = passed_models(scores, policy)
                yield {
                    **{field: upstream_row[field] for field in STEP3_ROW_FIELDS},
                    "mos_selection": {
                        "decision": "keep" if passed >= policy.min_mos_models else "reject",
                        "passed_models": passed,
                        "required_models": policy.min_mos_models,
                        "policy_id": policy.policy_id,
                        "policy_hash": policy.policy_hash,
                    },
                }


def passed_models(scores: Mapping[str, Any], policy: Any) -> int:
    """按 registry 语义计算动态 K/N 策略的通过模型数。"""
    return sum(
        int(mos_model_passed(model, scores[model], policy.thresholds[model]))
        for model in policy.mos_models
    )
