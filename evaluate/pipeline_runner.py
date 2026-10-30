"""评估任务适配器：只统一产物生命周期，不统一指标语义。"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from core.utils.inference_artifacts import atomic_write_json, fingerprint_path
from evaluate.artifact_lifecycle import (
    EVALUATION_RESERVATION_TOKEN_ENV,
    abort_artifact_run,
    attach_run_id,
    begin_unreserved_artifact_run,
    build_artifact_envelope,
    claim_artifact_run,
    new_run_id,
    normalize_ownership_binding,
    publish_artifact_commit,
    publish_artifact_envelope,
    validate_artifact_summary_filename,
)

_ContextT = TypeVar("_ContextT")


def _absolute_optional(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_absolute_optional(item) for item in value]
    return str(Path(value).expanduser().resolve())


def _fingerprint_optional(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_fingerprint_optional(item) for item in value]
    return fingerprint_path(Path(value).expanduser())


@dataclass(frozen=True)
class PipelineTaskAdapter:
    """把任务专属 summary 投影到公共生命周期字段。"""

    task: str
    pipeline: str
    step: str
    output_dir: str | Path
    provenance: Mapping[str, Any] = field(default_factory=dict)
    summary_filename: str = "summary.json"

    def _provenance(self) -> dict[str, Any]:
        provenance = dict(self.provenance)
        for key in ("config", "checkpoint", "model", "meta"):
            if key in provenance:
                value = provenance[key]
                provenance[key] = _absolute_optional(value)
                provenance[f"{key}_fingerprint"] = _fingerprint_optional(value)
        return provenance

    def validate_summary_binding(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        """复核 summary、当前资产 provenance 与 effective identity 同源。"""
        current = self._provenance()
        declared = summary.get("provenance")
        if declared is not None:
            if not isinstance(declared, Mapping) or dict(declared) != current:
                raise RuntimeError("evaluation summary provenance 与当前 adapter 不一致")
        identity = summary.get("effective_config_identity")
        if identity is not None:
            if not isinstance(identity, Mapping) or dict(identity) != current.get(
                "resolved_config_fingerprint"
            ):
                raise RuntimeError("evaluation summary effective config provenance 不一致")
        return current

    def ownership_binding(self) -> dict[str, Any]:
        return normalize_ownership_binding(
            {
                "task": self.task,
                "pipeline": self.pipeline,
                "step": self.step,
                "output_dir": str(Path(self.output_dir).expanduser().resolve()),
                "summary_filename": self.summary_filename,
                "provenance": self._provenance(),
            }
        )

    def build_envelope(
        self,
        summary: Mapping[str, Any],
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        provenance = self.validate_summary_binding(summary)
        status = str(summary.get("status", "unknown"))
        error = summary.get("reason") if status == "error" else None
        metric_coverage = summary.get("metric_coverage")
        if metric_coverage is None and self.task == "tse":
            metric_coverage = summary.get("coverage")
        if metric_coverage is None:
            datasets = summary.get("datasets", {})
            if isinstance(datasets, Mapping):
                per_dataset = {
                    name: item["metric_coverage"]
                    for name, item in datasets.items()
                    if isinstance(item, Mapping) and "metric_coverage" in item
                }
                metric_coverage = per_dataset or None
        return build_artifact_envelope(
            task=self.task,
            pipeline=self.pipeline,
            step=self.step,
            status=status,
            provenance=provenance,
            inference_coverage=summary.get("inference", summary.get("coverage")),
            metric_coverage=metric_coverage,
            summary=summary,
            run_id=run_id,
            error=None if error is None else str(error),
        )

    def publish(
        self,
        summary: Mapping[str, Any],
        *,
        run_id: str | None = None,
    ) -> Path:
        return publish_artifact_envelope(
            self.output_dir,
            self.build_envelope(summary, run_id=run_id),
        )


def run_pipeline(
    adapter: PipelineTaskAdapter,
    *,
    prepare: Callable[[], _ContextT],
    run: Callable[[_ContextT], Mapping[str, Any]],
    on_complete: Callable[[Mapping[str, Any]], None] | None = None,
    on_failure: Callable[[str, Exception], Mapping[str, Any]] | None = None,
    preflight: Callable[[], None] | None = None,
    precommit: Callable[[Mapping[str, Any]], None] | None = None,
    require_result_manifest: bool = False,
    requested_run_id: str | None = None,
    reraise: bool = True,
) -> Mapping[str, Any]:
    """按锁内 begin、执行、持久化、锁内 precommit/commit 执行生命周期。"""
    configured_output = Path(adapter.output_dir).expanduser()
    if configured_output.is_symlink():
        raise ValueError("pipeline 输出目录不允许符号链接")
    summary_filename = validate_artifact_summary_filename(adapter.summary_filename)
    reservation_token = os.environ.pop(EVALUATION_RESERVATION_TOKEN_ENV, None)
    if reservation_token is None:
        run_id = new_run_id() if requested_run_id is None else requested_run_id
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("pipeline requested_run_id must be a non-empty string")
        start = begin_unreserved_artifact_run(
            configured_output, run_id, preflight=preflight
        )
    else:
        if requested_run_id is not None:
            raise ValueError("reserved pipeline run_id is owned by the reservation")
        start = claim_artifact_run(
            configured_output, reservation_token, adapter.ownership_binding()
        )
    run_id = start.run_id
    previous_commit = start.previous_commit

    def failure_summary(stage: str, exc: Exception) -> dict[str, Any]:
        failed = (
            on_failure(stage, exc)
            if on_failure is not None
            else {
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
        return attach_run_id(failed, run_id)

    execution_error: Exception | None = None
    stage = "prepare"
    try:
        context = prepare()
        stage = "run"
        summary = attach_run_id(run(context), run_id)
    except Exception as exc:
        execution_error = exc
        try:
            summary = failure_summary(stage, exc)
        except Exception:
            abort_artifact_run(
                configured_output,
                run_id=run_id,
                previous_commit=previous_commit,
                reason=f"{stage}_failure_handler_failed",
            )
            raise

    if execution_error is not None and previous_commit is not None:
        abort_artifact_run(
            configured_output,
            run_id=run_id,
            previous_commit=previous_commit,
            reason=f"{stage}_failed",
        )
        if reraise:
            raise execution_error
        return summary

    summary_path = configured_output.resolve() / summary_filename
    try:
        if on_complete is not None:
            on_complete(summary)
        adapter.validate_summary_binding(summary)
        atomic_write_json(summary_path, dict(summary))
        adapter.publish(summary, run_id=run_id)

        def validate_commit_snapshot() -> None:
            if precommit is not None:
                precommit(summary)
            adapter.validate_summary_binding(summary)

        publish_artifact_commit(
            configured_output,
            run_id=run_id,
            summary_filename=summary_filename,
            precommit=validate_commit_snapshot,
            require_result_manifest=require_result_manifest,
        )
    except Exception as exc:
        abort_artifact_run(
            configured_output,
            run_id=run_id,
            previous_commit=previous_commit,
            reason="publish_failed",
        )
        try:
            failed = failure_summary("publish", exc)
        except Exception:
            if reraise:
                raise
            return attach_run_id(
                {"status": "error", "reason": f"{type(exc).__name__}: {exc}"},
                run_id,
            )
        if reraise:
            raise
        return failed

    if execution_error is not None and reraise:
        raise execution_error
    return summary
