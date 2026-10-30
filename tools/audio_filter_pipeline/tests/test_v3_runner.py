"""Generation-3 pure artifact orchestration tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional

import pytest

from audio_filter_pipeline.config import PipelineConfig
from audio_filter_pipeline.processors.stages_v3.base import (
    ArtifactExpectation,
    ArtifactStage,
    StageRunContext,
)
from audio_filter_pipeline.utils.artifacts import ArtifactIndex, PipelineStep


pytestmark = pytest.mark.unit


class _FakeStage(ArtifactStage):
    def __init__(
        self,
        step: PipelineStep,
        calls: list[tuple],
        *,
        bad_upstream=None,
        payload_stage: ArtifactStage | None = None,
    ):
        self.step = step
        self.upstream_step = bad_upstream if bad_upstream is not None else (
            None if step is PipelineStep.STEP0 else PipelineStep(step.value[:-1] + str(int(step.value[-1]) - 1))
        )
        self._calls = calls
        self._payload_stage = payload_stage

    def config_payload(self, config: PipelineConfig) -> Mapping[str, Any]:
        if self._payload_stage is not None:
            return self._payload_stage.config_payload(config)
        return {"step": self.step.value}

    def produce_rows(
        self,
        context: StageRunContext,
        upstream_index: Optional[ArtifactIndex],
        upstream_rows: Iterable[Mapping[str, Any]],
    ) -> Iterable[Mapping[str, Any]]:
        self._calls.append((self.step, context.upstream))
        yield {"step": self.step.value}


def _config(
    output_root: Path,
    *,
    generation: int = 3,
    mos_models=("scoreq", "urgent_pk", "dnsmos"),
    scoreq_threshold: float = 4.0,
    sources=None,
) -> PipelineConfig:
    selection_policies = None
    if generation == 3:
        selection_policies = {
            "global": {
                "mos_models": mos_models,
                "min_mos_models": 3,
            },
            "quality_profiles": {
                name: {"thresholds": {
                    "scoreq": scoreq_threshold,
                    "urgent_pk": 3.5,
                    "dnsmos": {
                        "ovrl": 3.2, "sig": 3.0, "bak": 3.8, "p808": 3.6,
                    },
                }}
                for name in ("quality_balanced", "quality_high", "quality_premium")
            },
        }
        if sources is not None:
            selection_policies["sources"] = sources
    return PipelineConfig(
        pipeline_generation=generation,
        step="step5" if generation == 3 else "step3",
        output_root=str(output_root),
        datasets={"alpha": {
            "input_root": "/missing/input",
            "language": "en",
            "category": "read_speech",
            "quality_profile": "quality_high",
            "speaker": {"source": "none", "required": False},
        }},
        selection_policies=selection_policies,
    )


def _factories(calls: list[tuple]):
    created = []

    def factory_for(step: PipelineStep):
        def factory(_config: PipelineConfig, _dataset_id: str) -> ArtifactStage:
            created.append(step)
            return _FakeStage(step, calls)
        return factory

    return {step: factory_for(step) for step in PipelineStep}, created


def _policy_variant_config(output_root: Path, variant: str) -> PipelineConfig:
    kwargs = {}
    if variant == "thresholds":
        kwargs["scoreq_threshold"] = 4.8
    elif variant == "model-tuple":
        kwargs["mos_models"] = ("dnsmos", "urgent_pk", "scoreq")
    elif variant == "exact-source-route":
        kwargs["sources"] = {
            "alpha": {
                "speaker/one.wav": {"thresholds": {"scoreq": 4.9}},
            },
        }
    elif variant != "baseline":
        raise ValueError(f"unknown policy variant: {variant}")
    return _config(output_root, **kwargs)


def _policy_projection_stages(config: PipelineConfig):
    from audio_filter_pipeline.processors.stages_v3.analyze import Step1AnalyzeStage
    from audio_filter_pipeline.processors.stages_v3.ingest import Step0IngestStage
    from audio_filter_pipeline.processors.stages_v3.segment import Step2SegmentStage

    return {
        PipelineStep.STEP0: Step0IngestStage(config.dataset_spec("alpha")),
        PipelineStep.STEP1: Step1AnalyzeStage(),
        PipelineStep.STEP2: Step2SegmentStage(),
    }


def _policy_projection_factories(config: PipelineConfig, calls: list[tuple]):
    payload_stages = _policy_projection_stages(config)

    def factory_for(step: PipelineStep):
        def factory(_config: PipelineConfig, _dataset_id: str) -> ArtifactStage:
            return _FakeStage(step, calls, payload_stage=payload_stages[step])
        return factory

    return {step: factory_for(step) for step in payload_stages}


def _artifact_snapshot(result) -> dict[str, tuple[int, int, bytes]]:
    snapshot = {}
    for step, index in result.items():
        for path in sorted(Path(index.artifact_dir).iterdir()):
            stat = path.stat()
            snapshot[f"{step.value}/{path.name}"] = (
                stat.st_ino, stat.st_mtime_ns, path.read_bytes(),
            )
    return snapshot


class _ProvenanceScorer:
    def __init__(self, payload, calls):
        self._payload = payload
        self._calls = calls

    def provenance_payload(self):
        return self._payload

    def score_task_groups(self, *_args, **_kwargs):
        self._calls.append("score")
        raise AssertionError("stale Step3 lineage must fail before scoring")


def _lineage_factories(config, scorer, producer_calls):
    from audio_filter_pipeline.processors.stages_v3.select import (
        Step3MosSelectionStage,
    )

    step3_payload = Step3MosSelectionStage(scorer=scorer)

    def stage(step, *, upstream=None, payload_stage=None):
        return lambda _config, _dataset: _FakeStage(
            step,
            producer_calls,
            bad_upstream=upstream,
            payload_stage=payload_stage,
        )

    return {
        PipelineStep.STEP0: stage(PipelineStep.STEP0),
        PipelineStep.STEP1: stage(PipelineStep.STEP1),
        PipelineStep.STEP2: stage(PipelineStep.STEP2),
        PipelineStep.STEP3: stage(
            PipelineStep.STEP3, payload_stage=step3_payload,
        ),
        PipelineStep.STEP4: stage(PipelineStep.STEP4),
        PipelineStep.STEP5: stage(
            PipelineStep.STEP5, upstream=PipelineStep.STEP3,
        ),
        PipelineStep.STEP6: stage(PipelineStep.STEP6),
    }


@pytest.mark.parametrize(
    "requested_step",
    [
        PipelineStep.STEP3,
        PipelineStep.STEP4,
        PipelineStep.STEP5,
        PipelineStep.STEP6,
    ],
)
def test_runner_rejects_legacy_step3_lineage_before_any_producer_without_rewrite(
    tmp_path, requested_step,
):
    from audio_filter_pipeline.processors.stages_v3.runner import V3PipelineRunner

    config = _config(tmp_path / "out")
    legacy_execution = {
        "scorer_execution": "single_process",
        "cpu_fallback": False,
        "runtime": {"requested_gpu": False},
    }
    current_execution = {
        "scorer_execution": "single_process",
        "cpu_fallback": False,
        "runtime": {
            "requested_device": "cpu",
            "provider_validation": "on_first_score",
        },
        "asset_load_binding": {"version": 2},
        "assets": {"selected_models": [], "models": {}},
    }
    legacy_score_calls = []
    legacy_producer_calls = []
    legacy_runner = V3PipelineRunner(
        config,
        _lineage_factories(
            config,
            _ProvenanceScorer(legacy_execution, legacy_score_calls),
            legacy_producer_calls,
        ),
    )
    step4_result = legacy_runner.run("alpha", PipelineStep.STEP4)
    step6_result = legacy_runner.run("alpha", PipelineStep.STEP6)
    legacy_artifacts = dict(step4_result)
    legacy_artifacts.update(step6_result)
    before = _artifact_snapshot(legacy_artifacts)

    current_score_calls = []
    current_producer_calls = []
    current_runner = V3PipelineRunner(
        config,
        _lineage_factories(
            config,
            _ProvenanceScorer(current_execution, current_score_calls),
            current_producer_calls,
        ),
    )

    with pytest.raises(ValueError, match="expected config digest mismatch"):
        current_runner.run("alpha", requested_step)

    assert legacy_score_calls == []
    assert current_score_calls == []
    assert current_producer_calls == []
    assert _artifact_snapshot(legacy_artifacts) == before


@pytest.mark.parametrize(
    "variant",
    ["thresholds", "model-tuple", "exact-source-route"],
)
def test_step0_through_step2_identity_and_resume_ignore_selection_policy(
    tmp_path, variant,
):
    from audio_filter_pipeline.processors.stages_v3.runner import V3PipelineRunner

    output_root = tmp_path / "out"
    baseline = _policy_variant_config(output_root, "baseline")
    changed = _policy_variant_config(output_root, variant)
    assert baseline.selection_policies != changed.selection_policies
    baseline_stages = _policy_projection_stages(baseline)
    changed_stages = _policy_projection_stages(changed)
    assert {
        step: stage.config_digest(baseline)
        for step, stage in baseline_stages.items()
    } == {
        step: stage.config_digest(changed)
        for step, stage in changed_stages.items()
    }

    first_calls = []
    first = V3PipelineRunner(
        baseline, _policy_projection_factories(baseline, first_calls),
    ).run("alpha", PipelineStep.STEP2)
    before = _artifact_snapshot(first)

    resumed_calls = []
    resumed = V3PipelineRunner(
        changed, _policy_projection_factories(changed, resumed_calls),
    ).run("alpha", PipelineStep.STEP2)

    assert [step for step, _upstream in first_calls] == [
        PipelineStep.STEP0, PipelineStep.STEP1, PipelineStep.STEP2,
    ]
    assert {
        step: index.config_digest for step, index in resumed.items()
    } == {
        step: index.config_digest for step, index in first.items()
    }
    assert resumed_calls == []
    assert {
        step: index.serialized() for step, index in resumed.items()
    } == {
        step: index.serialized() for step, index in first.items()
    }
    assert _artifact_snapshot(resumed) == before


def test_runner_executes_dependency_chain_with_bound_expectations(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.runner import V3PipelineRunner

    calls = []
    factories, created = _factories(calls)
    result = V3PipelineRunner(_config(tmp_path / "out"), factories).run("alpha", "step3")

    assert tuple(result) == tuple(PipelineStep)[:4]
    # 依赖解析从 target 反向回溯，factory discovery 顺序可不同；实际执行顺序
    # 由返回的拓扑闭包保证。
    assert set(created) == set(list(PipelineStep)[:4])
    assert [step for step, _ in calls] == list(PipelineStep)[:4]
    assert calls[0][1] is None
    for position in range(1, 4):
        upstream = calls[position][1]
        previous = result[PipelineStep(list(PipelineStep)[position - 1].value)]
        assert upstream == ArtifactExpectation(
            PipelineStep(list(PipelineStep)[position - 1].value),
            previous.config_digest,
            previous.lineage_digest,
        )
    assert result.target is result[PipelineStep.STEP3]


def test_runner_only_requires_factories_through_requested_step(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.runner import V3PipelineRunner

    calls = []
    factories, created = _factories(calls)
    del factories[PipelineStep.STEP2]

    result = V3PipelineRunner(_config(tmp_path / "out"), factories).run("alpha", PipelineStep.STEP1)

    assert tuple(result) == (PipelineStep.STEP0, PipelineStep.STEP1)
    assert set(created) == {PipelineStep.STEP0, PipelineStep.STEP1}
    assert [step for step, _ in calls] == [PipelineStep.STEP0, PipelineStep.STEP1]


def test_runner_follows_non_adjacent_upstream_without_running_unrelated_branch(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.runner import V3PipelineRunner

    calls = []
    factories, created = _factories(calls)

    def direct_step5(_config, _dataset_id):
        created.append(PipelineStep.STEP5)
        stage = _FakeStage(PipelineStep.STEP5, calls)
        stage.upstream_step = PipelineStep.STEP3
        return stage

    factories[PipelineStep.STEP5] = direct_step5
    result = V3PipelineRunner(_config(tmp_path / "out"), factories).run(
        "alpha", PipelineStep.STEP5,
    )

    expected = [
        PipelineStep.STEP0, PipelineStep.STEP1, PipelineStep.STEP2,
        PipelineStep.STEP3, PipelineStep.STEP5,
    ]
    assert tuple(result) == tuple(expected)
    assert set(created) == set(expected)
    assert [step for step, _ in calls] == expected


def test_runner_rejects_dependency_cycle(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.runner import V3PipelineRunner

    calls = []
    factories, _ = _factories(calls)

    def cyclic_step3(_config, _dataset_id):
        stage = _FakeStage(PipelineStep.STEP3, calls)
        stage.upstream_step = PipelineStep.STEP5
        return stage

    factories[PipelineStep.STEP3] = cyclic_step3
    with pytest.raises(ValueError, match="cycle"):
        V3PipelineRunner(_config(tmp_path / "out"), factories).run(
            "alpha", PipelineStep.STEP5,
        )


def test_runner_resumes_full_chain_from_existing_artifacts(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.runner import V3PipelineRunner

    config = _config(tmp_path / "out")
    first_calls = []
    factories, _ = _factories(first_calls)
    first = V3PipelineRunner(config, factories).run("alpha", PipelineStep.STEP2)

    resumed_calls = []
    factories, _ = _factories(resumed_calls)
    resumed = V3PipelineRunner(config, factories).run("alpha", PipelineStep.STEP2)

    assert resumed.target.serialized() == first.target.serialized()
    assert resumed_calls == []


@pytest.mark.parametrize("requested_step", [3, object()])
def test_runner_rejects_invalid_requested_step(tmp_path, requested_step):
    from audio_filter_pipeline.processors.stages_v3.runner import V3PipelineRunner

    calls = []
    factories, _ = _factories(calls)
    with pytest.raises(ValueError, match="requested_step"):
        V3PipelineRunner(_config(tmp_path / "out"), factories).run("alpha", requested_step)
    assert calls == []


def test_runner_fails_closed_on_missing_or_wrong_factory_stage(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.runner import V3PipelineRunner

    config = _config(tmp_path / "out")
    with pytest.raises(ValueError, match="missing factory"):
        V3PipelineRunner(config, {}).run("alpha", PipelineStep.STEP0)

    def wrong_factory(_config, _dataset_id):
        return _FakeStage(PipelineStep.STEP1, [])

    with pytest.raises(ValueError, match="returned stage"):
        V3PipelineRunner(config, {PipelineStep.STEP0: wrong_factory}).run("alpha", PipelineStep.STEP0)


def test_runner_fails_closed_on_wrong_upstream_declaration_or_generation(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.runner import V3PipelineRunner

    calls = []
    factories, _ = _factories(calls)

    def malformed_step1(_config, _dataset_id):
        return _FakeStage(PipelineStep.STEP1, calls, bad_upstream=PipelineStep.STEP2)

    factories[PipelineStep.STEP1] = malformed_step1
    with pytest.raises(ValueError, match="upstream_step"):
        V3PipelineRunner(_config(tmp_path / "out"), factories).run("alpha", PipelineStep.STEP1)

    valid_factories, _ = _factories([])
    with pytest.raises(ValueError, match="pipeline_generation must be 3"):
        _config(tmp_path / "legacy", generation=2)
