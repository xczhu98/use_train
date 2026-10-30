"""Generation-3 CLI orchestration, isolated from the legacy v2 entrypoint."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from ...config import PipelineConfig, validation_context_for_step
from ...utils.artifacts import (
    PipelineStep,
    drop_artifact,
    iter_artifact_rows,
)
from ...utils.path_utils import cleanup_expired_temp, v3_artifact_dir
from ...utils.strict_jsonl import add_cleanup_note


_LAZY_IMPORTS = {
    "DatasetRunContext": (".runner", "DatasetRunContext"),
    "MosExecutionPlan": (".mos_execution", "MosExecutionPlan"),
    "SpeakerGroupExtractorPool": (".speaker_shadow", "SpeakerGroupExtractorPool"),
    "StageRunContext": (".base", "StageRunContext"),
    "Step0IngestStage": (".ingest", "Step0IngestStage"),
    "Step1AnalyzeStage": (".analyze", "Step1AnalyzeStage"),
    "Step2SegmentStage": (".segment", "Step2SegmentStage"),
    "Step3MosRuntime": (".mos_runtime", "Step3MosRuntime"),
    "Step3MosScorerPool": (".mos_pool", "Step3MosScorerPool"),
    "Step3MosWorkerFleet": (".mos_pool", "Step3MosWorkerFleet"),
    "Step3MosSelectionStage": (".select", "Step3MosSelectionStage"),
    "Step4AudioExporter": (".audio_export", "Step4AudioExporter"),
    "Step4MosFilterStage": (".mos_filter", "Step4MosFilterStage"),
    "Step5SpeakerShadowStage": (".speaker_shadow", "Step5SpeakerShadowStage"),
    "Step6AudioExporter": (".audio_export", "Step6AudioExporter"),
    "preflight_export_nofile": (".audio_export", "preflight_export_nofile"),
    "Step6MultirateExportStage": (".export", "Step6MultirateExportStage"),
    "TenVadRuntimeOptions": (".ten_vad", "TenVadRuntimeOptions"),
    "V3PipelineRunner": (".runner", "V3PipelineRunner"),
    "VerifiedMosExecutionPlan": (".mos_asset_preflight", "VerifiedMosExecutionPlan"),
    "WeSpeakerEmbeddingRuntime": (".wespeaker_runtime", "WeSpeakerEmbeddingRuntime"),
    "WeSpeakerRuntimeOptions": (".wespeaker_runtime", "WeSpeakerRuntimeOptions"),
    "WorkerLimits": ("..optional_runtime", "WorkerLimits"),
    "_FINAL_SELECTION_VERSION": (".export", "_FINAL_SELECTION_VERSION"),
    "normalize_selectable_mos_models": ("..mos", "normalize_selectable_mos_models"),
    "verify_mos_assets": (".mos_asset_preflight", "verify_mos_assets"),
}


def __getattr__(name: str) -> Any:
    """仅在实际执行相应 stage 时加载可选 MOS/torch 依赖。"""
    try:
        module_name, attribute = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(importlib.import_module(module_name, __package__), attribute)
    globals()[name] = value
    return value


def _dependency(name: str) -> Any:
    return getattr(sys.modules[__name__], name)


_STEP_ORDER = tuple(PipelineStep)
_STAGE_UPSTREAM = {
    PipelineStep.STEP0: None,
    PipelineStep.STEP1: PipelineStep.STEP0,
    PipelineStep.STEP2: PipelineStep.STEP1,
    PipelineStep.STEP3: PipelineStep.STEP2,
    PipelineStep.STEP4: PipelineStep.STEP3,
    PipelineStep.STEP5: PipelineStep.STEP3,
    PipelineStep.STEP6: PipelineStep.STEP5,
}
logger = logging.getLogger(__name__)
_GENERATION_3_CONSUMED_FIELDS = frozenset({
    "audio_export_max_inflight_bytes",
    "audio_export_max_pending",
    "audio_export_workers",
    "datasets",
    "dnsmos_model_dir",
    "min_source_sample_rate_hz",
    "nisqa_model_path",
    "nisqa_model_sha256",
    "nisqa_model_version",
    "output_root",
    "output_sample_rate_hz",
    "pipeline_generation",
    "scoreq_model_dir",
    "selection_policies",
    "speaker_consistency",
    "step",
    "step0_workers",
    "step2_max_segment_duration_sec",
    "step2_min_segment_duration_sec",
    "step3_gpu_ids",
    "step3_worker_hard_timeout_seconds",
    "step3_worker_idle_timeout_seconds",
    "step3_worker_stderr_tail_bytes",
    "step4_export_root",
    "step4_mode",
    "step5_apply_speaker_filter",
    "step6_export_root",
    "step6_mode",
    "urgent_pk_backbone",
    "urgent_pk_checkpoint",
    "urgent_pk_encoder",
    "urgent_pk_expected_commit",
    "urgent_pk_expected_sha256",
    "urgent_pk_score_range",
    "urgent_pk_source_dir",
    "use_gpu",
    "utmos_model_path",
    "utmos_model_sha256",
    "utmos_model_version",
    "vad",
})


def _validate_top_level_config_fields(raw_config: Mapping[str, Any]) -> None:
    """拒绝未被 generation-3 runtime 消费的顶层配置字段。"""
    unconsumed = set(raw_config) - _GENERATION_3_CONSUMED_FIELDS
    if unconsumed:
        raise ValueError(
            "unknown generation-3 configuration fields (unconsumed): "
            f"{sorted(unconsumed)!r}"
        )
    generation = raw_config.get("pipeline_generation")
    if type(generation) is not int or generation != 3:
        raise ValueError(
            f"pipeline_generation must be explicit integer 3, got {generation!r}"
        )


def _required_steps(target_step: PipelineStep) -> frozenset[PipelineStep]:
    """返回 CLI 固定 stage DAG 的 target 依赖闭包。"""
    required: set[PipelineStep] = set()
    current: PipelineStep | None = target_step
    while current is not None:
        if current in required:
            raise ValueError(f"cycle detected in CLI stage dependencies at {current.value}")
        required.add(current)
        current = _STAGE_UPSTREAM[current]
    return frozenset(required)


def _will_run_export(target_step: PipelineStep, config: PipelineConfig) -> bool:
    """本次 CLI 运行是否会在 pipeline 后执行 FLAC export。"""
    if target_step is PipelineStep.STEP4:
        return config.step4_mode == "filter"
    if target_step is PipelineStep.STEP6:
        return config.step6_mode == "filter"
    return False


def _step6_selection_statistics(rows: Iterable[Mapping[str, Any]]) -> dict[str, object]:
    """从已发布 Step6 artifact 聚合最终选择，不读取任何音频。"""
    final_selection_version = _dependency("_FINAL_SELECTION_VERSION")
    reasons: Counter[str] = Counter()
    total = 0
    eligible = 0
    eligible_duration_sec = 0.0
    rejected_duration_sec = 0.0
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Step6 artifact row must be a mapping")
        final_selection = row.get("final_selection")
        if not isinstance(final_selection, Mapping):
            raise ValueError("Step6 artifact row final_selection must be a mapping")
        if final_selection.get("version") != final_selection_version:
            raise ValueError(
                "Step6 artifact row final_selection version must be exactly "
                f"{final_selection_version}"
            )
        export_eligible = final_selection.get("export_eligible")
        reason = final_selection.get("reason")
        if type(export_eligible) is not bool or type(reason) is not str or not reason:
            raise ValueError("Step6 artifact row final_selection is invalid")
        sample_rate = row.get("sample_rate")
        start_sample = row.get("start_sample")
        end_sample = row.get("end_sample")
        if (type(sample_rate) is not int or sample_rate <= 0
                or type(start_sample) is not int or type(end_sample) is not int
                or not 0 <= start_sample < end_sample):
            raise ValueError("Step6 artifact row segment duration fields are invalid")
        duration_sec = (end_sample - start_sample) / sample_rate
        total += 1
        reasons[reason] += 1
        if export_eligible:
            eligible += 1
            eligible_duration_sec += duration_sec
        else:
            rejected_duration_sec += duration_sec
    return {
        "total_segments": total,
        "eligible_segments": eligible,
        "rejected_segments": total - eligible,
        "eligible_duration_sec": eligible_duration_sec,
        "rejected_duration_sec": rejected_duration_sec,
        "reasons": dict(sorted(reasons.items())),
    }


def _step4_dry_run_statistics(
    decision_rows: Iterable[Mapping[str, Any]],
) -> dict[str, object]:
    """从 Step4 决策行聚合 MOS 筛选统计，不读取任何音频、不发布 artifact。"""
    total = 0
    keep = 0
    keep_duration_sec = 0.0
    reject_duration_sec = 0.0
    by_profile: dict[str, dict[str, int]] = {}
    passed_histogram: Counter[int] = Counter()
    policy_hashes: set[str] = set()
    max_selected_models = 0
    normalize_models = _dependency("normalize_selectable_mos_models")
    for row in decision_rows:
        if not isinstance(row, Mapping):
            raise ValueError("Step4 decision row must be a mapping")
        mos_selection = row.get("mos_selection")
        if not isinstance(mos_selection, Mapping):
            raise ValueError("Step4 decision row mos_selection must be a mapping")
        decision = mos_selection.get("decision")
        if decision not in {"keep", "reject"}:
            raise ValueError("Step4 decision must be exactly 'keep' or 'reject'")
        selected_models = normalize_models(row.get("mos_models"))
        selected_count = len(selected_models)
        passed_models = mos_selection.get("passed_models")
        required_models = mos_selection.get("required_models")
        if (type(required_models) is not int
                or not 1 <= required_models <= selected_count
                or type(passed_models) is not int
                or not 0 <= passed_models <= selected_count):
            raise ValueError("Step4 MOS model counts are invalid")
        max_selected_models = max(max_selected_models, selected_count)
        row_hash = mos_selection.get("policy_hash")
        if type(row_hash) is not str or not row_hash:
            raise ValueError("Step4 policy_hash must be a nonblank string")
        policy_hashes.add(row_hash)
        profile = row.get("quality_profile")
        if type(profile) is not str or not profile:
            raise ValueError("Step4 decision row quality_profile must be a nonblank string")
        sample_rate = row.get("sample_rate")
        start_sample = row.get("start_sample")
        end_sample = row.get("end_sample")
        if (type(sample_rate) is not int or sample_rate <= 0
                or type(start_sample) is not int or type(end_sample) is not int
                or not 0 <= start_sample < end_sample):
            raise ValueError("Step4 decision row segment duration fields are invalid")
        duration_sec = (end_sample - start_sample) / sample_rate
        total += 1
        passed_histogram[passed_models] += 1
        profile_stats = by_profile.setdefault(
            profile, {"total": 0, "keep": 0, "reject": 0},
        )
        profile_stats["total"] += 1
        if decision == "keep":
            keep += 1
            keep_duration_sec += duration_sec
            profile_stats["keep"] += 1
        else:
            reject_duration_sec += duration_sec
            profile_stats["reject"] += 1
    return {
        "total_segments": total,
        "keep_segments": keep,
        "reject_segments": total - keep,
        "keep_duration_sec": keep_duration_sec,
        "reject_duration_sec": reject_duration_sec,
        "by_quality_profile": {
            profile: dict(stats) for profile, stats in sorted(by_profile.items())
        },
        "passed_models_histogram": {
            count: passed_histogram.get(count, 0)
            for count in range(max_selected_models + 1)
        },
        "policy_hash": (
            next(iter(policy_hashes)) if len(policy_hashes) == 1 else None
        ),
        "policy_hashes": sorted(policy_hashes),
    }


def _requested_step(value: Any) -> PipelineStep:
    if type(value) is not str:
        raise ValueError("generation-3 step must be a string")
    normalized = value.strip().lower()
    if normalized == "all":
        return PipelineStep.STEP6
    try:
        return PipelineStep(normalized)
    except ValueError as exc:
        allowed = ", ".join(step.value for step in _STEP_ORDER) + ", all"
        raise ValueError(f"generation-3 step must be one of: {allowed}") from exc


def _drop_steps_argument(value: Any) -> tuple[PipelineStep, ...]:
    """解析 --drop-steps（逗号分隔 step0..step6），空值返回空元组。"""
    if value is None:
        return ()
    if type(value) is not str:
        raise ValueError("drop_steps must be a comma-separated step list")
    parts = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("drop_steps must be a comma-separated step list")
    steps = []
    for part in parts:
        try:
            steps.append(PipelineStep(part))
        except ValueError as exc:
            allowed = ", ".join(step.value for step in _STEP_ORDER)
            raise ValueError(
                f"drop_steps entries must be one of: {allowed}"
            ) from exc
    if len(set(steps)) != len(steps):
        raise ValueError("drop_steps must not contain duplicates")
    return tuple(steps)


def _cleanup_dataset_temp_artifacts(output_root: str, dataset_id: str) -> int:
    """仅清理当前 dataset 的 v3 artifact staging 目录，不触碰已发布产物。"""
    total = 0
    for step in _STEP_ORDER:
        artifact_dir = v3_artifact_dir(output_root, step, dataset_id)
        artifact_parent = os.path.dirname(artifact_dir)
        staging_prefix = f".{os.path.basename(artifact_dir)}.stage-"
        removed = cleanup_expired_temp(
            artifact_parent,
            prefixes=(staging_prefix,),
        )
        total += removed
    if total:
        logger.info(
            "cleaned pipeline temp artifacts dataset=%s count=%d",
            dataset_id, total,
        )
    return total


def _value(args: Any, config: Mapping[str, Any], name: str, default=None):
    value = getattr(args, name, None)
    return value if value is not None else config.get(name, default)


def _output_root(args: Any, raw_config: Mapping[str, Any]) -> Any:
    """按 canonical CLI `--output_root`、YAML `output_root` 顺序选择输出根。"""
    for value in (
        getattr(args, "output_root", None),
        raw_config.get("output_root"),
    ):
        if value is not None:
            return value
    return None


def _ten_vad_options(config: Mapping[str, Any]) -> Any:
    vad = config.get("vad", {})
    if not isinstance(vad, Mapping):
        raise ValueError("vad configuration must be a mapping")
    allowed = {"runtime", "hop_size", "threshold", "workers", "max_pending"}
    unexpected = set(vad) - allowed
    if unexpected:
        raise ValueError(f"unsupported TEN VAD configuration fields: {sorted(unexpected)}")
    runtime = vad.get("runtime", "ten_vad_native_cpu")
    if runtime != "ten_vad_native_cpu":
        raise ValueError("Step1 supports only vad.runtime=ten_vad_native_cpu")
    return _dependency("TenVadRuntimeOptions")(
        hop_size=vad.get("hop_size", 256),
        threshold=vad.get("threshold", 0.5),
    )


def _parse_step3_gpu_ids(value: Any) -> list[int] | Any:
    """Parse the v3 CLI override without inheriting legacy ``gpu_ids`` semantics."""
    if value is None or isinstance(value, (list, tuple)):
        return value
    if type(value) is not str:
        raise ValueError("step3_gpu_ids CLI value must be a comma-separated integer list")
    parts = value.split(",")
    if not parts or any(not part or not part.isdecimal() for part in parts):
        raise ValueError("step3_gpu_ids must be comma-separated nonnegative integers")
    gpu_ids = [int(part) for part in parts]
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("step3_gpu_ids must not contain duplicates")
    return gpu_ids


def _positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive Python integer")
    return value


def _use_gpu(args: Any, raw_config: Mapping[str, Any]) -> bool:
    value = _value(args, raw_config, "use_gpu", False)
    if type(value) is not bool:
        raise ValueError("use_gpu must be bool")
    no_use_gpu = getattr(args, "no_use_gpu", False)
    if type(no_use_gpu) is not bool:
        raise ValueError("no_use_gpu must be bool")
    return value and not no_use_gpu


def _wespeaker_options_payload(options: Any) -> dict[str, Any]:
    """调用 WeSpeaker canonical payload 接口并校验跨进程 JSON 边界。"""
    payload = options.to_payload()
    if not isinstance(payload, dict):
        raise TypeError("WeSpeaker options payload must be a dict")
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "WeSpeaker options payload must contain only JSON-safe values"
        ) from exc
    return payload


def _build_config(args: Any, raw_config: Mapping[str, Any], step: PipelineStep) -> PipelineConfig:
    _validate_top_level_config_fields(raw_config)
    _positive_int(_value(args, raw_config, "step0_workers", 32), "step0_workers")
    output_root = _output_root(args, raw_config)
    speaker_consistency = raw_config.get("speaker_consistency")
    legacy_filter = raw_config.get("step5_apply_speaker_filter")
    vad = raw_config.get("vad", {})
    if not isinstance(vad, Mapping):
        raise ValueError("vad configuration must be a mapping")
    return PipelineConfig(
        pipeline_generation=3,
        step=step.value,
        output_root=output_root,
        output_sample_rate_hz=raw_config.get("output_sample_rate_hz"),
        min_source_sample_rate_hz=raw_config.get("min_source_sample_rate_hz"),
        datasets=raw_config.get("datasets"),
        selection_policies=raw_config.get("selection_policies"),
        speaker_consistency=speaker_consistency,
        step5_apply_speaker_filter=legacy_filter,
        step3_gpu_ids=_parse_step3_gpu_ids(
            _value(args, raw_config, "step3_gpu_ids", ())
        ),
        step3_worker_idle_timeout_seconds=raw_config.get(
            "step3_worker_idle_timeout_seconds"
        ),
        step3_worker_hard_timeout_seconds=raw_config.get(
            "step3_worker_hard_timeout_seconds"
        ),
        step3_worker_stderr_tail_bytes=raw_config.get("step3_worker_stderr_tail_bytes", 4_096),
        step6_mode=raw_config.get("step6_mode"),
        step4_mode=raw_config.get("step4_mode"),
        step4_export_root=raw_config.get("step4_export_root"),
        step6_export_root=raw_config.get("step6_export_root"),
        audio_export_workers=raw_config.get("audio_export_workers", 32),
        audio_export_max_pending=raw_config.get("audio_export_max_pending", 64),
        audio_export_max_inflight_bytes=raw_config.get(
            "audio_export_max_inflight_bytes", 4 * 1024 * 1024 * 1024,
        ),
        step1_vad_workers=vad.get("workers"),
        step1_vad_max_pending=vad.get("max_pending"),
        step2_max_segment_duration_sec=raw_config.get(
            "step2_max_segment_duration_sec", 60.0,
        ),
        step2_min_segment_duration_sec=raw_config.get(
            "step2_min_segment_duration_sec", 0.0,
        ),
        urgent_pk_checkpoint=raw_config.get("urgent_pk_checkpoint"),
        urgent_pk_source_dir=raw_config.get("urgent_pk_source_dir"),
        urgent_pk_encoder=raw_config.get("urgent_pk_encoder"),
        urgent_pk_backbone=raw_config.get("urgent_pk_backbone"),
        urgent_pk_score_range=raw_config.get("urgent_pk_score_range"),
        urgent_pk_expected_sha256=raw_config.get("urgent_pk_expected_sha256"),
        urgent_pk_expected_commit=raw_config.get("urgent_pk_expected_commit"),
        scoreq_model_dir=raw_config.get("scoreq_model_dir"),
        dnsmos_model_dir=raw_config.get("dnsmos_model_dir"),
        nisqa_model_path=raw_config.get("nisqa_model_path"),
        nisqa_model_version=raw_config.get("nisqa_model_version"),
        nisqa_model_sha256=raw_config.get("nisqa_model_sha256"),
        utmos_model_path=raw_config.get("utmos_model_path"),
        utmos_model_version=raw_config.get("utmos_model_version"),
        utmos_model_sha256=raw_config.get("utmos_model_sha256"),
    )


def _worker_sha256(worker_script: str) -> str:
    digest = hashlib.sha256()
    with open(worker_script, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_config_base_dir(config_base_dir: Any) -> str | None:
    """校验显式配置目录，不把入口路径上下文写入 PipelineConfig。"""
    if config_base_dir is None:
        return None
    try:
        path = os.fspath(config_base_dir)
    except TypeError as exc:
        raise ValueError("config_base_dir must be an absolute existing canonical directory") from exc
    if type(path) is not str or not path or not os.path.isabs(path):
        raise ValueError("config_base_dir must be an absolute existing canonical directory")
    canonical = os.path.realpath(path)
    if path != canonical or not os.path.isdir(path):
        raise ValueError("config_base_dir must be an absolute existing canonical directory")
    return canonical


def _factories(
    args: Any,
    raw_config: Mapping[str, Any],
    config: PipelineConfig,
    target_step: PipelineStep,
    plan: MosExecutionPlan | None = None,
    verified: VerifiedMosExecutionPlan | None = None,
    *,
    config_base_dir: str | None = None,
):
    config_base_dir = _canonical_config_base_dir(config_base_dir)
    required_steps = _required_steps(target_step)
    ten_vad_options = (
        _ten_vad_options(raw_config)
        if PipelineStep.STEP1 in required_steps else None
    )
    requires_step3 = PipelineStep.STEP3 in required_steps
    if requires_step3:
        plan_type = _dependency("MosExecutionPlan")
        verified_type = _dependency("VerifiedMosExecutionPlan")
        if type(plan) is not plan_type:
            raise TypeError("Step3 factories require a MosExecutionPlan")
        if type(verified) is not verified_type:
            raise TypeError(
                "Step3 factories require a VerifiedMosExecutionPlan"
            )
        if plan.selected_models != verified.selected_models:
            raise ValueError("verified MOS models must exactly match the execution plan")
        topology = plan.topology
    else:
        # Step0-2 不要求 MOS capability，也不解析任何 MOS 资产或设备拓扑。
        topology = None
    speaker_policy = config.speaker_consistency
    speaker_options = None
    speaker_gpu_ids: tuple[int, ...] = ()
    speaker_worker_timeout_seconds = 0.0
    speaker_worker_stderr_tail_bytes = 0
    if PipelineStep.STEP5 in required_steps and speaker_policy.get("enabled"):
        worker_script = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "workers", "wespeaker_worker.py",
        ))
        speaker_gpu_ids = tuple(speaker_policy.get("gpu_ids", ()))
        speaker_worker_timeout_seconds = float(
            speaker_policy["worker_timeout_seconds"]
        )
        speaker_worker_stderr_tail_bytes = speaker_policy[
            "worker_stderr_tail_bytes"
        ]
        speaker_options = _dependency("WeSpeakerRuntimeOptions")(
            model_path=speaker_policy["model_dir"],
            model_version=speaker_policy["model_version"],
            model_sha256=speaker_policy["model_sha256"],
            runtime_version=speaker_policy["runtime_version"],
            target_sample_rate=speaker_policy["target_sample_rate"],
            worker_script=worker_script,
            worker_script_sha256=_worker_sha256(worker_script),
            worker_limits=_dependency("WorkerLimits")(
                **dict(speaker_policy["worker_limits"])
            ),
            cuda_visible_devices=(
                str(speaker_gpu_ids[0]) if len(speaker_gpu_ids) == 1 else None
            ),
        )

    step0_workers = _positive_int(
        _value(args, raw_config, "step0_workers", 32), "step0_workers",
    )

    def step0_factory(config: PipelineConfig, dataset_id: str):
        return _dependency("Step0IngestStage")(
            config.dataset_spec(dataset_id),
            mapping_base_dir=config_base_dir,
            workers=step0_workers,
        )

    def step1_factory(config: PipelineConfig, _dataset_id: str):
        assert ten_vad_options is not None
        return _dependency("Step1AnalyzeStage")(
            ten_vad_options=ten_vad_options,
            workers=getattr(config, "step1_vad_workers", 1),
            max_pending=getattr(config, "step1_vad_max_pending", 1),
        )

    def step2_factory(_config: PipelineConfig, _dataset_id: str):
        return _dependency("Step2SegmentStage")()

    def step3_factory(_config: PipelineConfig, dataset_id: str):
        assert topology is not None
        assert verified is not None
        if topology["kind"] == "gpu_process_pool":
            gpu_ids = tuple(topology["gpu_ids"])
            if not gpu_ids:
                raise ValueError(
                    "Step3 use_gpu=True requires a nonempty step3_gpu_ids list"
                )
            scorer = _dependency("Step3MosScorerPool")(
                gpu_ids,
                verified_plan=verified,
                dataset_id=dataset_id,
                worker_idle_timeout_seconds=(
                    config.step3_worker_idle_timeout_seconds
                ),
                worker_hard_timeout_seconds=(
                    config.step3_worker_hard_timeout_seconds
                ),
                worker_stderr_tail_bytes=topology["worker_stderr_tail_bytes"],
            )
        else:
            scorer = _dependency("Step3MosRuntime")(verified_plan=verified)
        return _dependency("Step3MosSelectionStage")(
            scorer=scorer, dataset_id=dataset_id,
        )

    def step4_factory(_config: PipelineConfig, dataset_id: str):
        return _dependency("Step4MosFilterStage")(dataset_id=dataset_id)

    def step5_factory(_config: PipelineConfig, _dataset_id: str):
        speaker_runtime = None
        if speaker_options is not None:
            speaker_runtime = (
                _dependency("SpeakerGroupExtractorPool")(
                    _wespeaker_options_payload(speaker_options),
                    list(speaker_gpu_ids),
                    worker_timeout_seconds=speaker_worker_timeout_seconds,
                    worker_stderr_tail_bytes=speaker_worker_stderr_tail_bytes,
                )
                if len(speaker_gpu_ids) > 1
                else _dependency("WeSpeakerEmbeddingRuntime")(speaker_options)
            )
        return _dependency("Step5SpeakerShadowStage")(
            embedding_extractor=speaker_runtime
        )

    def step6_factory(_config: PipelineConfig, dataset_id: str):
        return _dependency("Step6MultirateExportStage")(dataset_id=dataset_id)

    return {
        PipelineStep.STEP0: step0_factory,
        PipelineStep.STEP1: step1_factory,
        PipelineStep.STEP2: step2_factory,
        PipelineStep.STEP3: step3_factory,
        PipelineStep.STEP4: step4_factory,
        PipelineStep.STEP5: step5_factory,
        PipelineStep.STEP6: step6_factory,
    }


def run_v3_cli(
    args: Any,
    raw_config: Mapping[str, Any],
    *,
    config_base_dir: str | None = None,
) -> int:
    """Run each configured dataset through the requested v3 artifact target."""
    if not isinstance(raw_config, Mapping):
        raise ValueError("generation-3 configuration must be a mapping")
    config_base_dir = _canonical_config_base_dir(config_base_dir)
    target_step = _requested_step(_value(args, raw_config, "step", "step0"))
    config = _build_config(args, raw_config, target_step)

    dataset_ids = tuple(config.dataset_specs())
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("generation-3 configuration contains duplicate dataset ids")

    # 独立清理命令：不构造任何 stage，也不做执行期校验（含 MOS 资产
    # preflight）——资产缺失或被移动时仍必须能清理已发布 artifact。
    drop_steps = _drop_steps_argument(getattr(args, "drop_steps", None))
    if drop_steps:
        for dataset_id in dataset_ids:
            with _dependency("DatasetRunContext")(config.output_root, dataset_id):
                _cleanup_dataset_temp_artifacts(config.output_root, dataset_id)
                for step in drop_steps:
                    drop_artifact(config.output_root, step, dataset_id)
        logger.info(
            "dropped artifacts steps=%s datasets=%d",
            ",".join(step.value for step in drop_steps), len(dataset_ids),
        )
        return 0

    required_steps = _required_steps(target_step)
    # 显式 stage_names 跳过旧 MOS 资产 builder，但 Step5/6 speaker execution
    # preflight 仍由 config.validate 保留；结构与策略语义先于任何 factory。
    config.validate(
        context=validation_context_for_step(target_step.value, 3),
        stage_names=[],
    )
    plan = None
    verified = None
    step3_use_gpu = False
    if PipelineStep.STEP3 in required_steps:
        step3_use_gpu = _use_gpu(args, raw_config)
        plan = _dependency("MosExecutionPlan").from_config(
            config,
            use_gpu=step3_use_gpu,
        )
        verified = _dependency("verify_mos_assets")(plan)
    factories = _factories(
        args,
        raw_config,
        config,
        target_step,
        plan,
        verified,
        config_base_dir=config_base_dir,
    )

    fleet = None
    if step3_use_gpu:
        assert plan is not None
        assert verified is not None
        topology = plan.topology
        gpu_ids = tuple(topology["gpu_ids"])
        fleet = _dependency("Step3MosWorkerFleet")(
            gpu_ids,
            verified_plan=verified,
            worker_idle_timeout_seconds=config.step3_worker_idle_timeout_seconds,
            worker_hard_timeout_seconds=config.step3_worker_hard_timeout_seconds,
            worker_stderr_tail_bytes=topology["worker_stderr_tail_bytes"],
        )
        factories = dict(factories)

        def fleet_step3_factory(_config: PipelineConfig, dataset_id: str):
            return _dependency("Step3MosSelectionStage")(
                scorer=fleet.for_dataset(dataset_id), dataset_id=dataset_id,
            )

        factories[PipelineStep.STEP3] = fleet_step3_factory

    primary_error = None
    try:
        for dataset_id in dataset_ids:
            with _dependency("DatasetRunContext")(config.output_root, dataset_id):
                _cleanup_dataset_temp_artifacts(config.output_root, dataset_id)
                if target_step is PipelineStep.STEP4 and config.step4_mode == "dry_run":
                    # dry_run：复用/补齐 Step0-3 后直接对已发布 Step3 分数应用当前
                    # 阈值做统计预览，不发布 Step4 artifact（调阈值迭代用）。
                    result = _dependency("V3PipelineRunner")(
                        config, factories
                    ).run(dataset_id, "step3")
                    step3_index = result[PipelineStep.STEP3]
                    rows = iter_artifact_rows(step3_index)
                    context = _dependency("StageRunContext")(
                        config.output_root, dataset_id, config
                    )
                    decisions = _dependency("Step4MosFilterStage")(
                        dataset_id=dataset_id,
                    ).produce_rows(context, step3_index, rows)
                    statistics = _step4_dry_run_statistics(decisions)
                    logger.info(
                        "Step4 dry_run dataset=%s total=%d keep=%d reject=%d "
                        "keep_duration_sec=%.3f reject_duration_sec=%.3f "
                        "by_quality_profile=%s passed_models_histogram=%s policy_hash=%s",
                        dataset_id,
                        statistics["total_segments"], statistics["keep_segments"],
                        statistics["reject_segments"], statistics["keep_duration_sec"],
                        statistics["reject_duration_sec"],
                        statistics["by_quality_profile"],
                        statistics["passed_models_histogram"], statistics["policy_hash"],
                    )
                    continue
                result = _dependency("V3PipelineRunner")(
                    config, factories
                ).run(dataset_id, target_step)
                # 最后一个 dataset 的 Step3 GPU fleet 必须在 DatasetRunContext
                # 释放 run lock 之前关闭；否则 fleet 持有的资源会把 fd 使用量
                # 推到 export preflight 设置的软限制边界，锁释放时重新打开
                # output-root 目录可能以 EMFILE 失败。
                if fleet is not None and dataset_id == dataset_ids[-1]:
                    final_fleet = fleet
                    fleet = None
                    final_fleet.close()
                # 只能在 runner 完成、且最后一个 dataset 的 GPU fleet 已关闭后
                # 采样 fd 基线；启动前的预检无法覆盖 Step3 cache/runtime
                # 及 worker fleet 在运行期间新增的描述符。
                if _will_run_export(target_step, config):
                    _dependency("preflight_export_nofile")(
                        config.audio_export_workers,
                        max_pending=config.audio_export_max_pending,
                    )
                if target_step is PipelineStep.STEP4 and config.step4_mode == "filter":
                    # Step4 filter：发布 Step4 artifact 后导出 MOS keep 段，
                    # 即语音增强（SE）任务的干净数据集。
                    _dependency("Step4AudioExporter")(
                        config.output_root,
                        export_root=config.step4_export_root,
                        workers=config.audio_export_workers,
                        max_pending=config.audio_export_max_pending,
                        max_inflight_bytes=config.audio_export_max_inflight_bytes,
                        **({"output_sample_rate_hz": config.output_sample_rate_hz}
                           if config.output_sample_rate_hz is not None else {}),
                    ).export_artifact(result.target)
                if target_step is PipelineStep.STEP6:
                    if config.step6_mode == "filter":
                        _dependency("Step6AudioExporter")(
                            config.output_root,
                            export_root=config.step6_export_root,
                            workers=config.audio_export_workers,
                            max_pending=config.audio_export_max_pending,
                            max_inflight_bytes=config.audio_export_max_inflight_bytes,
                            **({"output_sample_rate_hz": config.output_sample_rate_hz}
                               if config.output_sample_rate_hz is not None else {}),
                        ).export_artifact(result.target)
                    else:
                        rows = iter_artifact_rows(result.target)
                        statistics = _step6_selection_statistics(rows)
                        logger.info(
                            "Step6 dry_run dataset=%s total=%d eligible=%d rejected=%d "
                            "eligible_duration_sec=%.3f rejected_duration_sec=%.3f reasons=%s",
                            dataset_id,
                            statistics["total_segments"], statistics["eligible_segments"],
                            statistics["rejected_segments"], statistics["eligible_duration_sec"],
                            statistics["rejected_duration_sec"], statistics["reasons"],
                        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if fleet is not None:
            try:
                fleet.close()
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                add_cleanup_note(primary_error, "MOS fleet shutdown", cleanup_error)
    return 0
