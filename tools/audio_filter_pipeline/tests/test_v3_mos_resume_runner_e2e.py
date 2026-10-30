"""Step3 MOS persistent-resume coverage through the real v3 runner and stage."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional

import pytest

from audio_filter_pipeline.config import PipelineConfig
from audio_filter_pipeline.processors.mos_contract import (
    MOS_ASSET_LOAD_BINDING,
    MOS_SCORING_EXECUTION_POLICY,
    MOS_WORKER_PROTOCOL,
)
from audio_filter_pipeline.processors.stages_v3.base import ArtifactStage, StageRunContext
import audio_filter_pipeline.processors.stages_v3.mos_gpu_job as mos_gpu_job
import audio_filter_pipeline.processors.stages_v3.mos_pool as mos_pool
from audio_filter_pipeline.processors.stages_v3.mos_pool import Step3MosScorerPool
from audio_filter_pipeline.processors.stages_v3.mos_runtime import _plain_json
from audio_filter_pipeline.processors.stages_v3.runner import V3PipelineRunner
from audio_filter_pipeline.processors.stages_v3.segment import STEP2_ROW_FIELDS
from audio_filter_pipeline.processors.stages_v3.select import Step3MosSelectionStage
from audio_filter_pipeline.utils.artifacts import ArtifactIndex, ArtifactWriter, PipelineStep


pytestmark = pytest.mark.unit


class _ForwardingStage(ArtifactStage):
    def __init__(self, step: PipelineStep, rows: list[dict[str, Any]]) -> None:
        self.step = step
        self.upstream_step = (
            None if step is PipelineStep.STEP0 else PipelineStep(
                f"step{int(step.value[-1]) - 1}"
            )
        )
        self._rows = rows

    def config_payload(self, _config: PipelineConfig) -> Mapping[str, Any]:
        return {"test_stage": self.step.value, "contract": 1}

    def produce_rows(
        self,
        _context: StageRunContext,
        _upstream_index: Optional[ArtifactIndex],
        _upstream_rows: Iterable[Mapping[str, Any]],
    ) -> Iterable[Mapping[str, Any]]:
        yield from self._rows


def _config(output_root: Path, *, scoreq_threshold: float = 4.2) -> PipelineConfig:
    profiles = {
        name: {"thresholds": {
            "scoreq": scoreq_threshold,
            "dnsmos": {"ovrl": 3.3, "sig": 3.4, "bak": 4.0, "p808": 3.8},
        }}
        for name in ("quality_balanced", "quality_high", "quality_premium")
    }
    return PipelineConfig(
        pipeline_generation=3,
        step="step3",
        output_root=str(output_root),
        datasets={"alpha": {
            "input_root": "/resume-test/input",
            "language": "en",
            "category": "read_speech",
            "quality_profile": "quality_high",
            "speaker": {"source": "none", "required": False},
        }},
        selection_policies={
            "global": {"mos_models": ["scoreq"], "min_mos_models": 1},
            "quality_profiles": profiles,
            "sources": {"alpha": {
                "b.wav": {"mos_models": ["dnsmos"], "min_mos_models": 1},
            }},
        },
    )


def _step2_row(ordinal: int, source_relative_path: str) -> dict[str, Any]:
    row = {
        "source_key": f"source-{ordinal}", "dataset_id": "alpha", "language": "en",
        "category": "read_speech", "quality_profile": "quality_high",
        "input_root": "/resume-test/input",
        "original_path": f"/resume-test/input/{source_relative_path}",
        "original_relative_path": source_relative_path,
        "source_relative_path": source_relative_path,
        "source_size_bytes": 42, "source_sha256": "a" * 64,
        "sample_rate": 16000, "frames": 32000, "channels": 1,
        "duration_sec": 2.0, "format": "WAV", "subtype": "PCM_16",
        "speaker_id": None, "speaker_scope": None,
        "source_start_sample": 0, "source_end_sample": 32000,
        "signal_finite": True, "signal_zero_ratio": 0.0, "signal_peak": 0.5,
        "signal_clipping_ratio": 0.0, "signal_dc_offset": 0.0,
        "signal_rms_dbfs": -20.0, "signal_num_samples": 32000,
        "bandwidth_hz": 7000.0, "estimated_sr": 16000,
        "vad_speech_ratio": 0.9, "vad_timeline_sample_rate": 16000,
        "vad_timeline_num_samples": 32000, "segment_id": f"segment-{ordinal}",
        "start_sample": 0, "end_sample": 32000, "boundary_reason": "source_end",
    }
    assert tuple(row) == STEP2_ROW_FIELDS
    return row


def _write_attestation(request: Mapping[str, Any]) -> None:
    providers = {
        "scoreq": {
            "execution_mode": "cuda_primary_hybrid_ep",
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "device": "cuda",
        },
        "dnsmos": {
            "onnx_sess": ["CUDAExecutionProvider"],
            "p808_onnx_sess": ["CUDAExecutionProvider"],
            "cpu_ep_fallback_disabled": False,
        },
    }
    tuples = []
    worker_models = []
    for subshard in request["subshards"]:
        models = subshard["models"]
        for model in models:
            if model not in worker_models:
                worker_models.append(model)
        tuples.append({
            "models": models,
            "attestation": {
                "requested_device": "cuda",
                "actual_providers": {model: providers[model] for model in models},
                "fallback": False,
                "selected_models": models,
                "asset_load_binding": _plain_json(MOS_ASSET_LOAD_BINDING),
                "scoring_execution_policy": _plain_json(MOS_SCORING_EXECUTION_POLICY),
            },
        })
    Path(request["attestation_path"]).write_text(json.dumps({
        "protocol": MOS_WORKER_PROTOCOL,
        "asset_load_binding": _plain_json(MOS_ASSET_LOAD_BINDING),
        "scoring_execution_policy": _plain_json(MOS_SCORING_EXECUTION_POLICY),
        "worker_models": worker_models,
        "model_tuple_attestations": tuples,
    }) + "\n", encoding="utf-8")


def _request_runner(calls: list[str], *, fail_dnsmos: bool = False):
    def run(_gpu_id: int, request_path: str, _cwd: str) -> None:
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        models = request["subshards"][0]["models"]
        calls.append("+".join(models))
        if fail_dnsmos and models == ["dnsmos"]:
            raise RuntimeError("intentional dnsmos request failure")
        for subshard in request["subshards"]:
            scores = (
                {"scoreq": 4.5}
                if subshard["models"] == ["scoreq"]
                else {"dnsmos": {"ovrl": 3.5, "sig": 3.6, "bak": 4.2, "p808": 4.0}}
            )
            with open(subshard["output_path"], "x", encoding="utf-8") as output:
                with open(subshard["input_path"], encoding="utf-8") as source:
                    for line in source:
                        output.write(json.dumps({
                            "ordinal": json.loads(line)["ordinal"], "scores": scores,
                        }) + "\n")
        _write_attestation(request)
    return run


def _runner(
    config: PipelineConfig,
    request_runner,
    *,
    fail_sidecar_publish: bool = False,
    gpu_ids: tuple[int, ...] = (0,),
) -> V3PipelineRunner:
    assets_root = Path(config.output_root).parent / "mos-assets"
    for name in ("scoreq", "dnsmos"):
        (assets_root / name).mkdir(parents=True, exist_ok=True)
    rows = [_step2_row(0, "a.wav"), _step2_row(1, "b.wav")]

    def step3_factory(_config: PipelineConfig, dataset_id: str) -> ArtifactStage:
        pool = Step3MosScorerPool(
            gpu_ids, dataset_id=dataset_id, request_runner=request_runner,
            scoreq_options={
                "model_dir": str(assets_root / "scoreq"),
                "verified_model_tree_sha256": "1" * 64,
            },
            dnsmos_options={
                "model_dir": str(assets_root / "dnsmos"),
                "verified_model_tree_sha256": "2" * 64,
            },
        )
        stage = Step3MosSelectionStage(scorer=pool, dataset_id=dataset_id)
        if fail_sidecar_publish:
            def fail_publish(*_args, **_kwargs):
                raise RuntimeError("intentional sidecar publish failure")
            stage._publish_sidecar = fail_publish
        return stage

    return V3PipelineRunner(config, {
        PipelineStep.STEP0: lambda _config, _dataset: _ForwardingStage(
            PipelineStep.STEP0, rows,
        ),
        PipelineStep.STEP1: lambda _config, _dataset: _ForwardingStage(
            PipelineStep.STEP1, rows,
        ),
        PipelineStep.STEP2: lambda _config, _dataset: _ForwardingStage(
            PipelineStep.STEP2, rows,
        ),
        PipelineStep.STEP3: step3_factory,
    })


def _fail_if_loaded(*_args, **_kwargs):
    pytest.fail("full sealed Step3 cache must not load a scorer or GPU worker")


def test_runner_stage_recovers_request_and_publish_crashes_without_rescoring(tmp_path, monkeypatch):
    monkeypatch.setattr(mos_pool, "local_model_tree_sha256", lambda path: (
        "1" * 64 if path.endswith("scoreq") else "2" * 64
    ))
    monkeypatch.setattr(mos_gpu_job, "MAX_WORKER_INPUT_SUBSHARDS", 1)
    interrupted_root = tmp_path / "interrupted"
    config = _config(interrupted_root)
    first_calls = []
    with pytest.raises(RuntimeError, match="intentional dnsmos request failure"):
        _runner(config, _request_runner(first_calls, fail_dnsmos=True)).run(
            "alpha", PipelineStep.STEP3,
        )
    assert first_calls == ["scoreq", "dnsmos"]
    assert not (interrupted_root / "artifacts" / "step3" / "alpha").exists()

    retry_calls = []
    retried = _runner(config, _request_runner(retry_calls)).run("alpha", PipelineStep.STEP3)
    assert retry_calls == ["dnsmos"]
    assert tuple(retried) == (
        PipelineStep.STEP0, PipelineStep.STEP1, PipelineStep.STEP2, PipelineStep.STEP3,
    )
    assert all(
        retried[step].artifact_digest == _runner(config, _fail_if_loaded).run(
            "alpha", PipelineStep.STEP3,
        )[step].artifact_digest
        for step in (PipelineStep.STEP0, PipelineStep.STEP1, PipelineStep.STEP2)
    )

    uninterrupted_calls = []
    uninterrupted = _runner(
        _config(tmp_path / "uninterrupted"), _request_runner(uninterrupted_calls),
    ).run("alpha", PipelineStep.STEP3)
    assert uninterrupted_calls == ["scoreq", "dnsmos"]
    assert uninterrupted.target.artifact_digest == retried.target.artifact_digest

    shutil.rmtree(retried.target.artifact_dir)
    Path(Step3MosSelectionStage.execution_attestation_path(retried.target)).unlink()
    rebuilt = _runner(config, _fail_if_loaded).run("alpha", PipelineStep.STEP3)
    assert rebuilt.target.artifact_digest == retried.target.artifact_digest

    threshold_hit = _runner(
        _config(interrupted_root, scoreq_threshold=4.9), _fail_if_loaded,
    ).run("alpha", PipelineStep.STEP3)
    assert threshold_hit.target.artifact_digest == rebuilt.target.artifact_digest

    before_publish_root = tmp_path / "before-publish"
    before_config = _config(before_publish_root)
    before_calls = []
    real_finalize = ArtifactWriter.finalize

    def fail_step3_finalize(writer):
        if writer._step is PipelineStep.STEP3:
            raise RuntimeError("intentional artifact publish failure")
        return real_finalize(writer)

    monkeypatch.setattr(ArtifactWriter, "finalize", fail_step3_finalize)
    with pytest.raises(RuntimeError, match="intentional artifact publish failure"):
        _runner(before_config, _request_runner(before_calls)).run("alpha", PipelineStep.STEP3)
    assert before_calls == ["scoreq", "dnsmos"]
    assert not (before_publish_root / "artifacts" / "step3" / "alpha").exists()
    monkeypatch.setattr(ArtifactWriter, "finalize", real_finalize)
    before_recovered = _runner(before_config, _fail_if_loaded).run("alpha", PipelineStep.STEP3)
    assert before_recovered.target.artifact_digest == uninterrupted.target.artifact_digest

    after_publish_root = tmp_path / "after-publish"
    after_config = _config(after_publish_root)
    after_calls = []
    with pytest.raises(RuntimeError, match="intentional sidecar publish failure"):
        _runner(
            after_config, _request_runner(after_calls), fail_sidecar_publish=True,
        ).run("alpha", PipelineStep.STEP3)
    assert after_calls == ["scoreq", "dnsmos"]
    after_recovered = _runner(after_config, _fail_if_loaded).run("alpha", PipelineStep.STEP3)
    assert after_recovered.target.artifact_digest == uninterrupted.target.artifact_digest
    assert Path(
        Step3MosSelectionStage.execution_attestation_path(after_recovered.target),
    ).exists()


def test_runner_stage_does_not_reuse_partial_plan_when_parallel_capacity_changes(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(mos_pool, "local_model_tree_sha256", lambda path: (
        "1" * 64 if path.endswith("scoreq") else "2" * 64
    ))
    monkeypatch.setattr(mos_gpu_job, "MAX_WORKER_INPUT_SUBSHARDS", 1)
    config = _config(tmp_path / "capacity")
    first_calls = []
    with pytest.raises(RuntimeError, match="intentional dnsmos request failure"):
        _runner(
            config, _request_runner(first_calls, fail_dnsmos=True), gpu_ids=(0,),
        ).run("alpha", PipelineStep.STEP3)
    assert first_calls == ["scoreq", "dnsmos"]

    resumed_calls = []
    _runner(
        config, _request_runner(resumed_calls), gpu_ids=(0, 1),
    ).run("alpha", PipelineStep.STEP3)
    assert resumed_calls == ["scoreq", "dnsmos"]
