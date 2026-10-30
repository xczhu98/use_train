"""Artifact-backed Generation-3 Step3 MOS scoring."""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import sqlite3
import sys
import tempfile
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Optional, Protocol, runtime_checkable

from tqdm import tqdm

from ...config import PipelineConfig
from ..mos_contract import (
    MOS_MODEL_ORDER,
    MOS_SHIPPED_MODELS,
    MOS_THRESHOLD_DIMENSIONS,
)
from ..mos import (
    list_mos_specs,
    normalize_selectable_mos_models,
    validate_mos_artifact_scores,
)
from ...utils.artifacts import ArtifactIndex, PipelineStep
from ...utils.strict_jsonl import (
    add_cleanup_note,
    decode_json_line,
    encode_json_line,
    finalize_streams,
    iter_bounded_jsonl,
    open_new_regular_stream,
    record_cleanup_error,
)
from .base import ArtifactStage, StageRunContext, _canonical_value
from .mos_lpt import (
    MAX_DISTINCT_MODEL_TUPLES,
    MAX_WORKER_INPUT_SUBSHARDS,
    STEP3_JSONL_MAX_FILE_BYTES,
    STEP3_SPOOL_MAX_TOTAL_BYTES,
)
from .mos_resume import (
    Step3ResumeIdentity,
    Step3ResumeStore,
    step3_task_identity,
)
from .segment import STEP2_ROW_FIELDS


@runtime_checkable
class Step3MosScorer(Protocol):
    """Step3 唯一生产 scorer 契约。"""

    def provenance_payload(self) -> Mapping[str, Any]:
        """返回可稳定写入 config digest 的严格 JSON provenance。"""

    def score_task_groups(
        self, groups: Sequence[Mapping[str, Any]],
    ) -> Iterable[Mapping[str, Any]]:
        """按 ordinal 顺序返回与 task identity 精确绑定的评分结果。"""


_SHIPPED_MOS_MODELS = MOS_SHIPPED_MODELS
_DNSMOS_DIMENSIONS = MOS_THRESHOLD_DIMENSIONS["dnsmos"]
_MAX_DISTINCT_MODEL_TUPLES = MAX_DISTINCT_MODEL_TUPLES
TUPLE_SPOOL_MAX_FILE_BYTES = STEP3_JSONL_MAX_FILE_BYTES
_COMPACT_INDEX_MAX_FILE_BYTES = 512 * 1024 * 1024
_STEP3_SPOOL_MAX_TOTAL_BYTES = STEP3_SPOOL_MAX_TOTAL_BYTES
_SEGMENT_INDEX_CACHE_BYTES = 8 * 1024 * 1024
_SEGMENT_INDEX_ENTRY_OVERHEAD_BYTES = 128
# spool 体积随 segment 数线性增长，聚合预算按行数伸缩（4 KiB/行覆盖
# tuple payload + compact index 行 + segment index 预留的宽松上界），
# 固定 4 GiB 下限只作为异常增长的 sanity guard；真实资源约束由
# check_total_spool_budget 里的磁盘剩余空间检查兜底。
_STEP3_SPOOL_BUDGET_BYTES_PER_ROW = 4 * 1024
_COMPACT_INDEX_BUDGET_BYTES_PER_ROW = 512
_SPOOL_DISK_CHECK_INTERVAL_BYTES = 256 * 1024 * 1024
_SPOOL_DISK_FREE_MARGIN_BYTES = 512 * 1024 * 1024
_SPOOL_BUDGET_ENV_VAR = "AUDIO_FILTER_STEP3_SPOOL_MAX_TOTAL_BYTES"
_EXECUTION_ATTESTATION_SCHEMA = "audio_filter_pipeline.step3_execution_attestation.v1"
_EXECUTION_ATTESTATION_MAX_BYTES = 1024 * 1024


def _task_digest(
    ordinal: int,
    row: Mapping[str, Any],
    models: Sequence[str],
    scoring_policy_hash: str,
) -> str:
    """兼容旧调用点的统一 Step3 task identity 入口。"""
    return step3_task_identity(ordinal, row, models, scoring_policy_hash)


def _scaled_budget_bytes(row_count: int, floor: int, per_row: int) -> int:
    if type(row_count) is not int or row_count < 0:
        raise ValueError("budget row_count must be a non-negative integer")
    return max(floor, row_count * per_row)


def _spool_budget_bytes(row_count: int) -> int:
    budget = _scaled_budget_bytes(
        row_count,
        _STEP3_SPOOL_MAX_TOTAL_BYTES,
        _STEP3_SPOOL_BUDGET_BYTES_PER_ROW,
    )
    override = os.environ.get(_SPOOL_BUDGET_ENV_VAR)
    if override is None:
        return budget
    try:
        value = int(override)
    except ValueError:
        value = 0
    if value <= 0:
        raise ValueError(
            f"{_SPOOL_BUDGET_ENV_VAR} must be a positive integer byte count"
        )
    return value


def _compact_index_budget_bytes(row_count: int) -> int:
    return _scaled_budget_bytes(
        row_count,
        _COMPACT_INDEX_MAX_FILE_BYTES,
        _COMPACT_INDEX_BUDGET_BYTES_PER_ROW,
    )


# Step3 记录 threshold-free 模型路由身份与分数；阈值、quorum、fusion 均由
# Step4/Step6 消费当前配置，修改这些决策参数无需重跑 MOS。
STEP3_ROW_FIELDS = STEP2_ROW_FIELDS + (
    "mos_models", "mos_scoring_policy_hash", "mos_scores",
)


class Step3MosSelectionStage(ArtifactStage):
    """按 immutable Step2 segment 及已解析策略记录 MOS 分数。"""

    step = PipelineStep.STEP3
    upstream_step = PipelineStep.STEP2
    progress_mode = "self_managed"
    progress_unit = "rows"

    def __init__(
        self,
        *,
        scorer: Step3MosScorer | None = None,
        dataset_id: str | None = None,
    ) -> None:
        if dataset_id is not None and (
            type(dataset_id) is not str or not dataset_id.strip()
        ):
            raise ValueError("Step3 dataset_id must be a non-empty string or None")
        self._dataset_id = dataset_id
        if scorer is None:
            raise TypeError("Step3 MOS scorer must be explicitly configured")
        if (not isinstance(scorer, Step3MosScorer)
                or not callable(scorer.provenance_payload)
                or not callable(scorer.score_task_groups)):
            raise TypeError(
                "Step3 MOS scorer must implement provenance_payload() and "
                "score_task_groups()"
            )
        provenance = scorer.provenance_payload()
        if not isinstance(provenance, Mapping):
            raise TypeError("Step3 scorer provenance payload must be a mapping")
        if not provenance:
            raise ValueError("Step3 scorer provenance payload must not be empty")
        self._scorer = scorer
        self._provenance = _canonical_value(
            provenance, "Step3 scorer provenance payload",
        )
        self._execution_attestation: Mapping[str, Any] | None = None
        self._resume_store: Step3ResumeStore | None = None
        self._resume_capability: Any = None
        self._resume_configured = False

    def _validate_execution_attestation(self, value: Any) -> Mapping[str, Any]:
        if not isinstance(value, Mapping) or not value:
            raise ValueError(
                "Step3 scorer must return a nonempty execution attestation on completion"
            )
        frozen = _canonical_value(value, "Step3 scorer execution attestation")
        payload = encode_json_line(
            frozen,
            "Step3 scorer execution attestation",
            max_line_bytes=_EXECUTION_ATTESTATION_MAX_BYTES,
        )
        if len(payload) > _EXECUTION_ATTESTATION_MAX_BYTES:
            raise ValueError("Step3 scorer execution attestation exceeds size limit")
        for field in ("provider_class", "device_class"):
            if type(frozen.get(field)) is not str or not frozen[field].strip():
                raise ValueError(
                    f"Step3 scorer execution attestation {field} must be nonblank"
                )
            expected = self._provenance.get(field)
            if expected is not None and frozen[field] != expected:
                raise ValueError(
                    f"Step3 scorer execution attestation {field} differs from scoring identity"
                )
        return frozen

    @staticmethod
    def execution_attestation_path(index: ArtifactIndex) -> str:
        return (
            f"{index.artifact_dir}.{index.artifact_digest}."
            "execution-attestation.json"
        )

    def _sidecar_payload(
        self,
        index: ArtifactIndex,
        scorer_provenance: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self._execution_attestation is None:
            raise RuntimeError("Step3 execution attestation is unavailable after scoring")
        scorer_identity = encode_json_line(
            scorer_provenance,
            "Step3 scorer identity",
            max_line_bytes=_EXECUTION_ATTESTATION_MAX_BYTES,
        )
        return {
            "schema": _EXECUTION_ATTESTATION_SCHEMA,
            "producer": {
                "step": index.step,
                "dataset_id": index.dataset_id,
                "config_digest": index.config_digest,
                "lineage_digest": index.lineage_digest,
                "artifact_digest": index.artifact_digest,
                "scorer_identity_sha256": hashlib.sha256(scorer_identity).hexdigest(),
            },
            "execution": self._execution_attestation,
        }

    def _read_bound_sidecar(
        self,
        index: ArtifactIndex,
        scorer_provenance: Mapping[str, Any],
        *,
        match_execution: bool = True,
    ) -> Mapping[str, Any]:
        path = self.execution_attestation_path(index)
        payloads = list(iter_bounded_jsonl(
            path,
            "Step3 execution attestation sidecar",
            max_line_bytes=_EXECUTION_ATTESTATION_MAX_BYTES,
            max_file_bytes=_EXECUTION_ATTESTATION_MAX_BYTES,
        ))
        if len(payloads) != 1:
            raise ValueError(
                "Step3 execution attestation sidecar must contain exactly one row"
            )
        value = decode_json_line(
            payloads[0],
            "Step3 execution attestation sidecar",
            max_line_bytes=_EXECUTION_ATTESTATION_MAX_BYTES,
        )
        expected = self._sidecar_payload(index, scorer_provenance)
        if (
            not isinstance(value, Mapping)
            or set(value) != {"schema", "producer", "execution"}
            or value["schema"] != expected["schema"]
            or value["producer"] != expected["producer"]
            or match_execution and value["execution"] != expected["execution"]
        ):
            raise ValueError("Step3 execution attestation sidecar binding mismatch")
        return value

    def _publish_sidecar(
        self, index: ArtifactIndex, scorer_provenance: Mapping[str, Any],
    ) -> None:
        path = self.execution_attestation_path(index)
        payload = encode_json_line(
            self._sidecar_payload(index, scorer_provenance),
            "Step3 execution attestation sidecar",
            max_line_bytes=_EXECUTION_ATTESTATION_MAX_BYTES,
        )
        temporary = f"{path}.tmp-{secrets.token_hex(16)}"
        stream = open_new_regular_stream(
            temporary, "Step3 execution attestation temporary sidecar",
        )
        try:
            stream.write(payload)
            finalize_streams((stream,), None, sync=True)
            stream = None
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError:
                winner = self._read_bound_sidecar(
                    index, scorer_provenance, match_execution=False,
                )
                self._execution_attestation = self._validate_execution_attestation(
                    winner["execution"]
                )
            parent_fd = os.open(
                os.path.dirname(path),
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        finally:
            if stream is not None:
                finalize_streams((stream,), sys.exc_info()[1], sync=False)
            if os.path.lexists(temporary):
                os.unlink(temporary)
        self._read_bound_sidecar(index, scorer_provenance)

    def before_publish(self, context: StageRunContext, writer: Any) -> None:
        if self._resume_store is None or self._resume_capability is None:
            return
        if self._execution_attestation is None:
            raise RuntimeError("Step3 execution attestation is unavailable before publish")
        preview = writer.preview_index()
        scorer_provenance = self._scorer_provenance(context.config)
        scorer_identity = encode_json_line(
            scorer_provenance,
            "Step3 scorer identity",
            max_line_bytes=_EXECUTION_ATTESTATION_MAX_BYTES,
        )
        assembly = {
            "artifact_hash": preview.artifact_digest,
            "artifact_path": preview.artifact_dir,
            "config_digest": preview.config_digest,
            "lineage_digest": preview.lineage_digest,
            "scorer_identity_sha256": hashlib.sha256(scorer_identity).hexdigest(),
            "sidecar_path": self.execution_attestation_path(preview),
            "proof": {
                "schema": _EXECUTION_ATTESTATION_SCHEMA,
                "execution": self._execution_attestation,
            },
        }
        self._resume_store.record_assembly_attestation(
            self._resume_capability, assembly,
        )

    def run(self, context: StageRunContext) -> ArtifactIndex:
        self._execution_attestation = None
        self._validate_contract(context)
        scorer_provenance = self._scorer_provenance(context.config)
        self._validate_configured_model_tuples(context)
        try:
            index = super().run(context)
        except BaseException:
            self._clear_resume()
            raise
        if self._execution_attestation is None:
            # cache hit 不得加载 estimator；优先经 artifact reader gate 恢复 pending sidecar。
            if context.persistent_write_capability is not None:
                upstream_index, _lineage = self._open_upstream_index(context)
                if upstream_index is None:
                    raise RuntimeError("Step3 cache hit has no upstream index")
                identity = Step3ResumeIdentity.create(Step3ResumeStore.identity_payload(
                    dataset_id=context.dataset_id,
                    step2_artifact_digest=upstream_index.artifact_digest,
                    step2_rows_sha256=upstream_index.rows_sha256,
                    step2_row_count=upstream_index.row_count,
                    step2_lineage_digest=upstream_index.lineage_digest,
                    step3_config=_canonical_value(
                        self.config_payload(context.config),
                        "Step3 resume config payload",
                    ),
                    model_provider=self._provenance,
                    execution_policy=self._resume_execution_policy(),
                ))
                store = Step3ResumeStore(context.output_root, context.dataset_id, identity)
                scorer_identity = encode_json_line(
                    scorer_provenance,
                    "Step3 scorer identity",
                    max_line_bytes=_EXECUTION_ATTESTATION_MAX_BYTES,
                )
                assembly = {
                    "artifact_hash": index.artifact_digest,
                    "artifact_path": index.artifact_dir,
                    "config_digest": index.config_digest,
                    "lineage_digest": index.lineage_digest,
                    "scorer_identity_sha256": hashlib.sha256(scorer_identity).hexdigest(),
                    "sidecar_path": self.execution_attestation_path(index),
                    "proof": None,
                }
                recovered = store.recover_assembly_attestation(assembly)
                if recovered is not None:
                    proof = recovered["proof"]
                    if not isinstance(proof, Mapping) or set(proof) != {"schema", "execution"}:
                        raise ValueError("Step3 pending sidecar proof is invalid")
                    if proof["schema"] != _EXECUTION_ATTESTATION_SCHEMA:
                        raise ValueError("Step3 pending sidecar proof schema is invalid")
                    self._execution_attestation = self._validate_execution_attestation(
                        proof["execution"]
                    )
                    self._publish_sidecar(index, scorer_provenance)
                    assembly["proof"] = proof
                    store.consume_assembly_attestation(
                        context.persistent_write_capability, assembly,
                    )
                    return index
            # 无 pending record 时只验证既有 producer runtime sidecar。
            scorer_identity = encode_json_line(
                scorer_provenance,
                "Step3 scorer identity",
                max_line_bytes=_EXECUTION_ATTESTATION_MAX_BYTES,
            )
            path = self.execution_attestation_path(index)
            payloads = list(iter_bounded_jsonl(
                path,
                "Step3 execution attestation sidecar",
                max_line_bytes=_EXECUTION_ATTESTATION_MAX_BYTES,
                max_file_bytes=_EXECUTION_ATTESTATION_MAX_BYTES,
            ))
            if len(payloads) != 1:
                raise ValueError("Step3 execution attestation sidecar is invalid")
            value = decode_json_line(
                payloads[0], "Step3 execution attestation sidecar",
                max_line_bytes=_EXECUTION_ATTESTATION_MAX_BYTES,
            )
            producer = value.get("producer") if isinstance(value, Mapping) else None
            expected_producer = {
                "step": index.step,
                "dataset_id": index.dataset_id,
                "config_digest": index.config_digest,
                "lineage_digest": index.lineage_digest,
                "artifact_digest": index.artifact_digest,
                "scorer_identity_sha256": hashlib.sha256(scorer_identity).hexdigest(),
            }
            if (
                not isinstance(value, Mapping)
                or set(value) != {"schema", "producer", "execution"}
                or value["schema"] != _EXECUTION_ATTESTATION_SCHEMA
                or producer != expected_producer
            ):
                raise ValueError("Step3 execution attestation sidecar binding mismatch")
            self._execution_attestation = self._validate_execution_attestation(
                value["execution"]
            )
            return index
        try:
            self._publish_sidecar(index, scorer_provenance)
            if self._resume_store is not None and self._resume_capability is not None:
                scorer_identity = encode_json_line(
                    scorer_provenance,
                    "Step3 scorer identity",
                    max_line_bytes=_EXECUTION_ATTESTATION_MAX_BYTES,
                )
                assembly = {
                    "artifact_hash": index.artifact_digest,
                    "artifact_path": index.artifact_dir,
                    "config_digest": index.config_digest,
                    "lineage_digest": index.lineage_digest,
                    "scorer_identity_sha256": hashlib.sha256(scorer_identity).hexdigest(),
                    "sidecar_path": self.execution_attestation_path(index),
                    "proof": {
                        "schema": _EXECUTION_ATTESTATION_SCHEMA,
                        "execution": self._execution_attestation,
                    },
                }
                self._resume_store.consume_assembly_attestation(
                    self._resume_capability, assembly,
                )
            return index
        finally:
            self._clear_resume()

    def _resume_execution_policy(self) -> Mapping[str, Any]:
        resolve = getattr(self._scorer, "resume_execution_policy", None)
        if callable(resolve):
            return _canonical_value(
                resolve(), "Step3 scorer resume execution policy",
            )
        return {
            "scheduler": "weighted_lpt_ordinals",
            "request_max_subshards": MAX_WORKER_INPUT_SUBSHARDS,
            "resume_protocol": "request_parent_verified_seal.v1",
        }

    def _configure_resume(
        self, context: StageRunContext, upstream_index: ArtifactIndex,
    ) -> None:
        if self._resume_configured:
            return
        self._resume_store = None
        self._resume_capability = None
        capability = context.persistent_write_capability
        configure = getattr(self._scorer, "configure_resume", None)
        if capability is None or not callable(configure):
            return
        identity = Step3ResumeIdentity.create(Step3ResumeStore.identity_payload(
            dataset_id=context.dataset_id,
            step2_artifact_digest=upstream_index.artifact_digest,
            step2_rows_sha256=upstream_index.rows_sha256,
            step2_row_count=upstream_index.row_count,
            step2_lineage_digest=upstream_index.lineage_digest,
            step3_config=_canonical_value(
                self.config_payload(context.config),
                "Step3 resume config payload",
            ),
            model_provider=self._provenance,
            execution_policy=self._resume_execution_policy(),
        ))
        store = Step3ResumeStore(context.output_root, context.dataset_id, identity)
        configure(store, capability)
        self._resume_store = store
        self._resume_capability = capability
        self._resume_configured = True

    def _clear_resume(self) -> None:
        if not self._resume_configured:
            return
        clear = getattr(self._scorer, "clear_resume", None)
        if callable(clear):
            clear()
        self._resume_store = None
        self._resume_capability = None
        self._resume_configured = False

    def _bound_dataset_id(self, config: PipelineConfig) -> str | None:
        if self._dataset_id is None:
            return None
        config.dataset_spec(self._dataset_id)
        return self._dataset_id

    def _validate_contract(self, context: StageRunContext) -> None:
        super()._validate_contract(context)
        if self._dataset_id is not None and self._dataset_id != context.dataset_id:
            raise ValueError("Step3 stage dataset_id does not match context")

    def config_payload(self, config: PipelineConfig) -> Mapping[str, Any]:
        dataset_id = self._bound_dataset_id(config)
        policy = {
            "resolver": "PipelineConfig.dataset_mos_scoring_binding",
            "score_schema": {
                name: {
                    "primary_score_key": spec.primary_score_key,
                    "artifact_score_keys": [list(pair) for pair in spec.artifact_score_keys],
                    "version": spec.version,
                }
                for name, spec in sorted(list_mos_specs().items())
                if spec.selectable
            },
            "max_distinct_model_tuples": _MAX_DISTINCT_MODEL_TUPLES,
        }
        if dataset_id is None:
            projections = []
            source_routes = config.selection_policies.get("sources", {})
            for configured_id, spec in sorted(config.dataset_specs().items()):
                projections.append(dict(
                    config.mos_scoring_projection(configured_id, spec.language)
                ))
                for source_path in source_routes.get(configured_id, {}):
                    projections.append(dict(config.mos_scoring_projection(
                        configured_id,
                        spec.language,
                        source_relative_path=source_path,
                    )))
            policy["projections"] = projections
        else:
            policy["dataset_binding"] = config.dataset_mos_scoring_binding(dataset_id)
        return {
            "row_contract": {"version": 4, "fields": list(STEP3_ROW_FIELDS)},
            "upstream_row_contract": {"version": 3, "fields": list(STEP2_ROW_FIELDS)},
            "policy": policy,
            # thresholds、quorum、mos_fusion 刻意不进 Step3 digest。
            "scorer_execution": self._scorer_provenance(config),
            "waveform_policy": {"read": False, "write": False},
        }

    @staticmethod
    def _dataset_scoring_models(
        config: PipelineConfig, dataset_id: str,
    ) -> tuple[str, ...]:
        binding = config.dataset_mos_scoring_binding(dataset_id)
        selected = {
            model
            for projection in binding["projections"]
            for model in projection["mos_models"]
        }
        return tuple(model for model in MOS_MODEL_ORDER if model in selected)

    @staticmethod
    def _filter_model_mapping(
        value: Any, models: tuple[str, ...], label: str,
    ) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a mapping")
        return {
            model: value[model]
            for model in models
            if model in value
        }

    def _scorer_provenance(self, config: PipelineConfig) -> Mapping[str, Any]:
        """返回当前数据集选中模型与全局运行时身份绑定的 provenance。"""
        dataset_id = self._bound_dataset_id(config)
        provenance = _canonical_value(
            self._provenance, "Step3 scorer provenance payload",
        )
        if dataset_id is None:
            return provenance
        models = self._dataset_scoring_models(config, dataset_id)
        projected = dict(provenance)
        configured_models = provenance.get("models")
        if isinstance(configured_models, list):
            projected["models"] = [model for model in models if model in configured_models]
        if "model_options" in provenance:
            projected["model_options"] = self._filter_model_mapping(
                provenance["model_options"], models, "Step3 scorer model_options",
            )
        if "assets" in provenance:
            assets = provenance["assets"]
            if not isinstance(assets, Mapping):
                raise TypeError("Step3 scorer assets must be a mapping")
            projected_assets = dict(assets)
            if "selected_models" in assets:
                selected_models = assets["selected_models"]
                if not isinstance(selected_models, list):
                    raise TypeError("Step3 scorer assets.selected_models must be a list")
                projected_assets["selected_models"] = [
                    model for model in models if model in selected_models
                ]
            if "models" in assets:
                projected_assets["models"] = self._filter_model_mapping(
                    assets["models"], models, "Step3 scorer assets.models",
                )
            projected["assets"] = projected_assets
        for model in MOS_MODEL_ORDER:
            for suffix in ("_model_dir", "_model_tree_sha256"):
                field = f"{model}{suffix}"
                if field in provenance and model not in models:
                    projected.pop(field, None)
            assets_field = f"{model}_assets"
            if assets_field in provenance and model not in models:
                projected.pop(assets_field, None)
        return _canonical_value(
            projected, "Step3 projected scorer provenance payload",
        )

    def _validate_configured_model_tuples(self, context: StageRunContext) -> None:
        """在 artifact writer 和 tuple spool 副作用前验证全部静态路由。"""
        validate = getattr(self._scorer, "validate_model_tuples", None)
        if not callable(validate):
            return
        spec = context.config.dataset_spec(context.dataset_id)
        source_routes = context.config.selection_policies.get("sources", {}).get(
            context.dataset_id, {},
        )
        projections = [context.config.mos_scoring_projection(
            context.dataset_id, spec.language,
        )]
        projections.extend(
            context.config.mos_scoring_projection(
                context.dataset_id,
                spec.language,
                source_relative_path=source_path,
            )
            for source_path in source_routes
        )
        validate([projection["mos_models"] for projection in projections])

    @staticmethod
    def _validate_upstream_row(row: Mapping[str, Any], dataset_id: str) -> None:
        if set(row) != set(STEP2_ROW_FIELDS):
            raise ValueError("Step2 upstream row fields do not match the exact contract")
        if row["dataset_id"] != dataset_id:
            raise ValueError("Step2 upstream row dataset_id does not match context")
        for field in ("source_key", "segment_id", "language"):
            if type(row[field]) is not str or not row[field].strip():
                raise ValueError(f"Step2 upstream row {field} must be a nonblank string")
        for field in ("sample_rate", "frames", "start_sample", "end_sample"):
            if type(row[field]) is not int:
                raise ValueError(f"Step2 upstream row {field} must be a Python integer")
        if not 0 <= row["start_sample"] < row["end_sample"] <= row["frames"]:
            raise ValueError("Step2 upstream segment bounds are invalid")

    @classmethod
    def _validate_scores(
        cls, scores: Any, models: tuple[str, ...],
    ) -> dict[str, Any]:
        return validate_mos_artifact_scores(scores, models)

    def produce_rows(
        self,
        context: StageRunContext,
        upstream_index: Optional[ArtifactIndex],
        upstream_rows: Iterable[Mapping[str, Any]],
    ) -> Iterable[Mapping[str, Any]]:
        if upstream_index is None or upstream_index.step != PipelineStep.STEP2.value:
            raise ValueError("Step3 MOS scoring requires a Step2 upstream artifact")
        max_line_bytes = 8 * 1024 * 1024
        row_count = upstream_index.row_count
        spool_budget = _spool_budget_bytes(row_count)
        compact_index_budget = _compact_index_budget_bytes(row_count)

        def task_for(
            ordinal: int,
            row: Mapping[str, Any],
            segment_index: sqlite3.Connection,
            reserve_segment_id: Any,
        ) -> dict[str, Any]:
            if not isinstance(row, Mapping):
                raise ValueError("Step2 upstream row must be a mapping")
            self._validate_upstream_row(row, context.dataset_id)
            if row["language"] != context.config.dataset_spec(context.dataset_id).language:
                raise ValueError(
                    "Step2 upstream row language does not exactly match dataset policy identity"
                )
            policy = context.config.resolve_selection_policy(
                row["dataset_id"],
                row["language"],
                source_relative_path=row["source_relative_path"],
            )
            if (policy.category != row["category"]
                    or policy.quality_profile != row["quality_profile"]):
                raise ValueError("Step2 row identity does not match resolved selection policy")
            models = normalize_selectable_mos_models(policy.mos_models)
            projection = context.config.mos_scoring_projection(
                row["dataset_id"],
                row["language"],
                source_relative_path=row["source_relative_path"],
            )
            segment_id = row["segment_id"]
            reserve_segment_id(segment_id)
            try:
                segment_index.execute(
                    "INSERT INTO segment_ids (segment_id) VALUES (?)", (segment_id,),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("duplicate segment_id in Step2 artifact") from exc
            return {
                "ordinal": ordinal,
                "row": row,
                "models": list(models),
                "scoring_policy_hash": projection["scoring_policy_hash"],
            }

        def encode(value: Mapping[str, Any], label: str) -> bytes:
            return encode_json_line(
                value, label, max_line_bytes=max_line_bytes,
            )

        def is_sha256(value: Any) -> bool:
            return (
                type(value) is str
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
            )

        def iter_compact(path: str) -> Iterator[dict[str, Any]]:
            payloads = iter(iter_bounded_jsonl(
                path,
                "Step3 compact task index",
                max_line_bytes=max_line_bytes,
                max_file_bytes=compact_index_budget,
            ))
            try:
                for payload in payloads:
                    value = decode_json_line(
                        payload, "Step3 compact task index row",
                        max_line_bytes=max_line_bytes,
                    )
                    if (
                        not isinstance(value, dict)
                        or set(value) != {
                            "ordinal", "models", "scoring_policy_hash", "task_digest",
                        }
                        or type(value["ordinal"]) is not int
                        or not is_sha256(value["scoring_policy_hash"])
                        or not is_sha256(value["task_digest"])
                    ):
                        raise ValueError("Step3 compact task index row is invalid")
                    yield value
            finally:
                primary = sys.exc_info()[1]
                close = getattr(payloads, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException as cleanup_error:
                        if isinstance(primary, GeneratorExit):
                            raise cleanup_error
                        if primary is not None:
                            add_cleanup_note(
                                primary,
                                "Step3 compact task index payload cleanup",
                                cleanup_error,
                            )
                        else:
                            raise

        def render(
            row: Mapping[str, Any],
            models: tuple[str, ...],
            scoring_policy_hash: str,
            raw_scores: Any,
        ) -> Mapping[str, Any]:
            return {
                **{field: row[field] for field in STEP2_ROW_FIELDS},
                "mos_models": list(models),
                "mos_scoring_policy_hash": scoring_policy_hash,
                "mos_scores": self._validate_scores(raw_scores, models),
            }

        with tempfile.TemporaryDirectory(prefix="audio-filter-step3-tuples-") as directory:
            segment_index = sqlite3.connect(
                os.path.join(directory, "segment-ids.sqlite3"),
            )
            tuple_states: dict[tuple[str, ...], dict[str, Any]] = {}
            tuple_descriptors: list[dict[str, Any]] = []
            compact_path = os.path.join(directory, "compact-task-index.jsonl")
            try:
                compact_stream = open_new_regular_stream(
                    compact_path, "Step3 compact task index",
                )
            except BaseException as primary:
                try:
                    segment_index.close()
                except BaseException as cleanup_error:
                    add_cleanup_note(primary, "Step3 segment index cleanup", cleanup_error)
                raise
            compact_size = 0
            total_tuple_bytes = 0
            count = 0

            def segment_index_bytes() -> int:
                page_count = segment_index.execute("PRAGMA page_count").fetchone()[0]
                page_size = segment_index.execute("PRAGMA page_size").fetchone()[0]
                return page_count * page_size

            reserved_segment_index_bytes = 0
            disk_check_watermark = 0

            def check_total_spool_budget(
                extra_bytes: int = 0, *, index_reservation: int = 0,
            ) -> None:
                nonlocal disk_check_watermark
                logical_index_bytes = segment_index_bytes()
                estimated_index_bytes = max(
                    logical_index_bytes,
                    reserved_segment_index_bytes + index_reservation,
                )
                total_bytes = (
                    total_tuple_bytes + compact_size + estimated_index_bytes
                    + _SEGMENT_INDEX_CACHE_BYTES + extra_bytes
                )
                if total_bytes > spool_budget:
                    raise ValueError(
                        "Step3 tuple spool aggregate budget exceeded: "
                        f"projected {total_bytes} bytes > budget {spool_budget} "
                        f"bytes for {row_count} rows; set "
                        f"{_SPOOL_BUDGET_ENV_VAR} to raise the cap, or point "
                        "TMPDIR at a larger filesystem"
                    )
                if total_bytes >= disk_check_watermark:
                    disk_check_watermark = (
                        total_bytes + _SPOOL_DISK_CHECK_INTERVAL_BYTES
                    )
                    free_bytes = shutil.disk_usage(directory).free
                    if total_bytes + _SPOOL_DISK_FREE_MARGIN_BYTES > free_bytes:
                        raise ValueError(
                            "Step3 tuple spool exceeds free disk space: "
                            f"projected {total_bytes} bytes plus "
                            f"{_SPOOL_DISK_FREE_MARGIN_BYTES} bytes safety margin "
                            f"vs {free_bytes} bytes free on the spool filesystem; "
                            "point TMPDIR at a larger filesystem or reduce the "
                            "dataset"
                        )

            def reserve_segment_id(segment_id: str) -> None:
                nonlocal reserved_segment_index_bytes
                key_bytes = len(segment_id.encode("utf-8"))
                reserve_bytes = key_bytes + _SEGMENT_INDEX_ENTRY_OVERHEAD_BYTES
                check_total_spool_budget(index_reservation=reserve_bytes)
                reserved_segment_index_bytes += reserve_bytes

            def open_tuple_part(state: dict[str, Any]) -> None:
                path = os.path.join(
                    directory,
                    f"tuple-{state['tuple_index']}-part-{state['part_index']}.jsonl",
                )
                state["path"] = path
                state["stream"] = open_new_regular_stream(
                    path, "Step3 tuple task spool",
                )
                state["size"] = 0
                state["rows"] = 0

            def finalize_tuple_part(state: dict[str, Any]) -> None:
                stream = state["stream"]
                if stream is None:
                    return
                finalize_streams((stream,), None, sync=True)
                state["stream"] = None
                state["descriptors"].append({
                    "models": list(state["models"]),
                    "path": state["path"],
                    "row_count": state["rows"],
                })
                state["part_index"] += 1

            try:
                segment_index.execute("PRAGMA journal_mode=OFF")
                segment_index.execute("PRAGMA synchronous=OFF")
                segment_index.execute("PRAGMA cache_size=-8192")
                segment_index.execute(
                    "CREATE TABLE segment_ids (segment_id TEXT PRIMARY KEY) WITHOUT ROWID"
                )
                for ordinal, row in enumerate(upstream_rows):
                    if ordinal >= upstream_index.row_count:
                        raise ValueError("Step2 stream exceeds verified artifact row_count")
                    task = task_for(
                        ordinal, row, segment_index, reserve_segment_id,
                    )
                    check_total_spool_budget()
                    models = tuple(task["models"])
                    state = tuple_states.get(models)
                    if state is None:
                        if len(tuple_states) >= _MAX_DISTINCT_MODEL_TUPLES:
                            raise ValueError("Step3 distinct MOS model tuple limit exceeded")
                        state = {
                            "models": models,
                            "tuple_index": len(tuple_states),
                            "part_index": 0,
                            "stream": None,
                            "path": None,
                            "size": 0,
                            "rows": 0,
                            "descriptors": [],
                        }
                        tuple_states[models] = state
                        open_tuple_part(state)
                    payload = encode(task, "Step3 tuple task")
                    if len(payload) > TUPLE_SPOOL_MAX_FILE_BYTES:
                        raise ValueError("Step3 tuple JSONL row exceeds physical part limit")
                    if state["rows"] and (
                        state["size"] + len(payload) > TUPLE_SPOOL_MAX_FILE_BYTES
                    ):
                        finalize_tuple_part(state)
                        open_tuple_part(state)
                    check_total_spool_budget(len(payload))
                    state["stream"].write(payload)
                    state["size"] += len(payload)
                    state["rows"] += 1
                    total_tuple_bytes += len(payload)
                    compact_payload = encode({
                        "ordinal": ordinal,
                        "models": task["models"],
                        "scoring_policy_hash": task["scoring_policy_hash"],
                        "task_digest": _task_digest(
                            ordinal,
                            task["row"],
                            models,
                            task["scoring_policy_hash"],
                        ),
                    }, "Step3 compact task index row")
                    if (compact_size + len(compact_payload)
                            > compact_index_budget):
                        raise ValueError(
                            "Step3 compact task index exceeds file limit: "
                            f"projected {compact_size + len(compact_payload)} "
                            f"bytes > budget {compact_index_budget} bytes for "
                            f"{row_count} rows"
                        )
                    check_total_spool_budget(len(compact_payload))
                    compact_stream.write(compact_payload)
                    compact_size += len(compact_payload)
                    count += 1
                if count != upstream_index.row_count:
                    raise ValueError("Step2 stream differs from verified artifact row_count")
                for state in tuple_states.values():
                    finalize_tuple_part(state)
                    tuple_descriptors.extend(state["descriptors"])
                finalize_streams((compact_stream,), None, sync=True)
                compact_stream = None
            finally:
                active_streams = tuple(
                    state["stream"]
                    for state in tuple_states.values()
                    if state["stream"] is not None
                )
                for state in tuple_states.values():
                    state["stream"] = None
                if compact_stream is not None:
                    active_streams += (compact_stream,)
                    compact_stream = None
                primary = sys.exc_info()[1]
                closing_generator = isinstance(primary, GeneratorExit)
                cleanup_failure = None
                try:
                    finalize_streams(
                        active_streams,
                        None if closing_generator else primary,
                        sync=True,
                    )
                except BaseException as cleanup_error:
                    cleanup_failure = record_cleanup_error(
                        cleanup_failure, cleanup_error, "Step3 tuple spool cleanup",
                    )
                try:
                    segment_index.close()
                except BaseException as cleanup_error:
                    cleanup_failure = record_cleanup_error(
                        cleanup_failure, cleanup_error, "Step3 segment index cleanup",
                    )
                if closing_generator and cleanup_failure is not None:
                    raise cleanup_failure
                if primary is not None:
                    if cleanup_failure is not None:
                        add_cleanup_note(
                            primary, "Step3 spool ownership cleanup", cleanup_failure,
                        )
                elif cleanup_failure is not None:
                    raise cleanup_failure

            progress = tqdm(
                total=upstream_index.row_count,
                unit=self.progress_unit,
                desc=f"{self.step.value}:{context.dataset_id}",
                file=sys.stdout,
                disable=None,
            )
            progress_lock = threading.Lock()
            displayed_rows = 0

            def update_progress(completed_rows: int) -> None:
                nonlocal displayed_rows
                with progress_lock:
                    remaining_rows = upstream_index.row_count - displayed_rows
                    if remaining_rows <= 0:
                        return
                    increment = min(completed_rows, remaining_rows)
                    progress.update(increment)
                    displayed_rows += increment

            set_progress_callback = None
            callback_set = False
            scored_iterator = None
            compact_iterator = None
            try:
                self._configure_resume(context, upstream_index)
                set_progress_callback = getattr(
                    self._scorer, "set_progress_callback", None,
                )
                callback_set = callable(set_progress_callback)
                if callback_set:
                    set_progress_callback(update_progress)
                scored = self._scorer.score_task_groups(tuple_descriptors)
                try:
                    scored_iterator = iter(scored)
                except TypeError as exc:
                    raise ValueError("Step3 scorer results must be iterable") from exc
                compact_iterator = iter(iter_compact(compact_path))
                processed_rows = 0
                for expected in range(upstream_index.row_count):
                    try:
                        compact = next(compact_iterator)
                    except StopIteration:
                        raise ValueError("Step3 compact task index is incomplete") from None
                    try:
                        scored_row = next(scored_iterator)
                    except StopIteration:
                        raise ValueError("Step3 scorer result count mismatch") from None
                    models = normalize_selectable_mos_models(compact["models"])
                    if (not isinstance(scored_row, Mapping)
                            or set(scored_row) != {
                                "ordinal", "row", "scores", "models",
                            }
                            or compact["ordinal"] != expected
                            or type(scored_row["ordinal"]) is not int
                            or scored_row["ordinal"] != expected
                            or scored_row["models"] != compact["models"]
                            or not isinstance(scored_row["row"], Mapping)
                            or _task_digest(
                                expected,
                                scored_row["row"],
                                models,
                                compact["scoring_policy_hash"],
                            ) != compact["task_digest"]):
                        raise ValueError("Step3 scorer result identity is invalid")
                    yield render(
                        scored_row["row"], models,
                        compact["scoring_policy_hash"], scored_row["scores"],
                    )
                    processed_rows += 1
                    with progress_lock:
                        missing_rows = processed_rows - displayed_rows
                        if missing_rows > 0:
                            progress.update(missing_rows)
                            displayed_rows += missing_rows
                try:
                    next(compact_iterator)
                except StopIteration:
                    pass
                else:
                    raise ValueError("Step3 compact task index exceeds artifact row_count")
                try:
                    next(scored_iterator)
                except StopIteration as completion:
                    self._execution_attestation = self._validate_execution_attestation(
                        completion.value
                    )
                else:
                    raise ValueError("Step3 scorer result count mismatch")
            finally:
                primary = sys.exc_info()[1]
                cleanup_error = None
                for label, iterator in (
                    ("Step3 compact task iterator", compact_iterator),
                    ("Step3 scorer iterator", scored_iterator),
                ):
                    close = getattr(iterator, "close", None)
                    if callable(close):
                        try:
                            close()
                        except BaseException as exc:
                            cleanup_error = record_cleanup_error(
                                cleanup_error, exc, label,
                            )
                if callback_set:
                    try:
                        set_progress_callback(None)
                    except BaseException as exc:
                        cleanup_error = record_cleanup_error(
                            cleanup_error, exc, "Step3 scorer progress callback",
                        )
                try:
                    progress.close()
                except BaseException as exc:
                    cleanup_error = record_cleanup_error(
                        cleanup_error, exc, "Step3 progress display",
                    )
                if isinstance(primary, GeneratorExit) and cleanup_error is not None:
                    raise cleanup_error
                if primary is not None:
                    if cleanup_error is not None:
                        add_cleanup_note(primary, "Step3 scoring cleanup", cleanup_error)
                elif cleanup_error is not None:
                    raise cleanup_error
