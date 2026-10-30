"""Generation-3 Step3 MOS selection artifact contract tests."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from audio_filter_pipeline.config import PipelineConfig
from audio_filter_pipeline.processors.stages_v3.base import (
    ArtifactExpectation,
    StageRunContext,
)
from audio_filter_pipeline.processors.stages_v3.mos_execution import MosExecutionPlan
from audio_filter_pipeline.processors.stages_v3.segment import STEP2_ROW_FIELDS
import audio_filter_pipeline.processors.stages_v3.select as select_module
from audio_filter_pipeline.processors.stages_v3.select import (
    STEP3_ROW_FIELDS,
    Step3MosSelectionStage,
)
from audio_filter_pipeline.tests.stage_test_helpers import runner_expectation_for_test
import audio_filter_pipeline.tests.step3_test_helpers as step3_test_helpers
from audio_filter_pipeline.tests.step3_test_helpers import CanonicalStep3Scorer
from audio_filter_pipeline.utils.artifacts import (
    ArtifactWriter,
    PipelineStep,
    iter_artifact_rows,
)


pytestmark = pytest.mark.unit


def _quality_profiles():
    return {
        "quality_balanced": {"thresholds": {
            "scoreq": 3.6, "urgent_pk": 3.5, "nisqa": 3.5, "utmos": 3.5,
            "dnsmos": {"ovrl": 3.2, "sig": 3.0, "bak": 3.8, "p808": 3.6},
        }},
        "quality_high": {"thresholds": {
            "scoreq": 4.2, "urgent_pk": 3.5, "nisqa": 3.7, "utmos": 3.9,
            "dnsmos": {"ovrl": 3.3, "sig": 3.4, "bak": 4.0, "p808": 3.8},
        }},
        "quality_premium": {"thresholds": {
            "scoreq": 4.5, "urgent_pk": 3.5, "nisqa": 4.0, "utmos": 4.2,
            "dnsmos": {"ovrl": 3.4, "sig": 3.6, "bak": 4.2, "p808": 4.0},
        }},
    }


def _config(
    output_root,
    *,
    models=("scoreq", "urgent_pk", "dnsmos"),
    min_mos_models=None,
    source_policies=None,
):
    return PipelineConfig(
        pipeline_generation=3,
        step="step3",
        output_root=str(output_root),
        datasets={"alpha": {
            "input_root": "/missing/input",
            "language": "en",
            "category": "read_speech",
            "quality_profile": "quality_high",
            "speaker": {"source": "none", "required": False},
        }},
        selection_policies={
            "global": {
                "mos_models": list(models),
                "min_mos_models": min_mos_models or len(models),
            },
            "quality_profiles": _quality_profiles(),
            **({"sources": {"alpha": source_policies}} if source_policies else {}),
        },
    )


def _two_dataset_config(
    output_root,
    *,
    alpha_model="scoreq",
    alpha_threshold=4.0,
    beta_model="utmos",
    beta_threshold=3.5,
    datasets_order=("alpha", "beta"),
):
    profiles = {
        name: {"thresholds": {
            "scoreq": alpha_threshold,
            "utmos": 3.5,
            "dnsmos": {"ovrl": 3.2, "sig": 3.0, "bak": 3.8, "p808": 3.6},
        }}
        for name in ("quality_balanced", "quality_high", "quality_premium")
    }
    return PipelineConfig(
        pipeline_generation=3,
        step="step3",
        output_root=str(output_root),
        datasets={
            dataset_id: {
                "input_root": f"/missing/{dataset_id}",
                "language": "en",
                "category": "read_speech",
                "quality_profile": "quality_high",
                "speaker": {"source": "none", "required": False},
            }
            for dataset_id in datasets_order
        },
        selection_policies={
            "global": {"mos_models": ["scoreq"], "min_mos_models": 1},
            "quality_profiles": profiles,
            "datasets": {
                "alpha": {
                    "mos_models": [alpha_model],
                    "min_mos_models": 1,
                    "thresholds": {alpha_model: alpha_threshold},
                },
                "beta": {
                    "mos_models": [beta_model],
                    "min_mos_models": 1,
                    "thresholds": {beta_model: beta_threshold},
                },
            },
        },
    )


def test_step3_routes_single_selected_model_to_scorer(tmp_path):
    config = _config(tmp_path / "out", models=("scoreq",))
    upstream = _publish_step2(config.output_root, [_row()])
    calls = []

    def scorer(row, models):
        calls.append((row["segment_id"], models))
        return {"scoreq": 4.3}

    index = Step3MosSelectionStage(scorer=_test_scorer(scorer)).run(_context(config, upstream))
    result = list(iter_artifact_rows(index))[0]

    assert calls == [("segment-1", ("scoreq",))]
    assert result["mos_models"] == ["scoreq"]
    assert result["mos_scores"] == {"scoreq": 4.3}
    assert len(result["mos_scoring_policy_hash"]) == 64


def test_step3_persists_bound_execution_attestation_sidecar_without_row_change(
    tmp_path,
):
    config = _config(tmp_path / "out", models=("scoreq",))
    upstream = _publish_step2(config.output_root, [_row()])
    stage = Step3MosSelectionStage(
        scorer=_test_scorer(lambda _row, _models: {"scoreq": 4.3}),
    )

    index = stage.run(_context(config, upstream))
    sidecar_path = Path(stage.execution_attestation_path(index))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert set(sidecar) == {"schema", "producer", "execution"}
    assert sidecar["producer"] == {
        "step": "step3",
        "dataset_id": "alpha",
        "config_digest": index.config_digest,
        "lineage_digest": index.lineage_digest,
        "artifact_digest": index.artifact_digest,
        "scorer_identity_sha256": sidecar["producer"]["scorer_identity_sha256"],
    }
    assert len(sidecar["producer"]["scorer_identity_sha256"]) == 64
    assert sidecar["execution"]["provider_class"] == "test_canonical"
    assert set(list(iter_artifact_rows(index))[0]) == set(STEP3_ROW_FIELDS)

    resumed = Step3MosSelectionStage(
        scorer=_test_scorer(
            lambda *_args: pytest.fail("cache hit must not score")
        ),
    ).run(_context(config, upstream))
    assert resumed.serialized() == index.serialized()


def test_step3_routes_two_selected_models_in_declared_order(tmp_path):
    config = _config(tmp_path / "out", models=("utmos", "nisqa"), min_mos_models=1)
    upstream = _publish_step2(config.output_root, [_row()])
    calls = []

    def scorer(_row, models):
        calls.append(models)
        return {"utmos": 4.0, "nisqa": 3.8}

    index = Step3MosSelectionStage(scorer=_test_scorer(scorer)).run(_context(config, upstream))
    result = list(iter_artifact_rows(index))[0]

    assert calls == [("utmos", "nisqa")]
    assert result["mos_models"] == ["utmos", "nisqa"]
    assert result["mos_scores"] == {"utmos": 4.0, "nisqa": 3.8}


def _row(*, segment_id="segment-1", dataset_id="alpha"):
    values = {
        "source_key": "source-1",
        "dataset_id": dataset_id,
        "language": "en",
        "category": "read_speech",
        "quality_profile": "quality_high",
        "input_root": "/missing/input",
        "original_path": "/missing/input/source.wav",
        "original_relative_path": "source.wav",
        "source_relative_path": "source.wav",
        "source_size_bytes": 42,
        "source_sha256": hashlib.sha256(b"source-1").hexdigest(),
        "sample_rate": 16000,
        "frames": 160000,
        "channels": 1,
        "duration_sec": 10.0,
        "format": "WAV",
        "subtype": "PCM_16",
        "speaker_id": None,
        "speaker_scope": None,
        "source_start_sample": 0,
        "source_end_sample": 160000,
        "signal_finite": True,
        "signal_zero_ratio": 0.0,
        "signal_peak": 0.5,
        "signal_clipping_ratio": 0.0,
        "signal_dc_offset": 0.0,
        "signal_rms_dbfs": -20.0,
        "signal_num_samples": 160000,
        "bandwidth_hz": 7000.0,
        "estimated_sr": 16000,
        "vad_speech_ratio": 0.9,
        "vad_timeline_sample_rate": 16000,
        "vad_timeline_num_samples": 160000,
        "segment_id": segment_id,
        "start_sample": 0,
        "end_sample": 160000,
        "boundary_reason": "source_end",
    }
    assert tuple(values) == STEP2_ROW_FIELDS
    return values


def _expectation(index):
    return runner_expectation_for_test(index)


def _publish_step2(output_root, rows):
    writer = ArtifactWriter(
        str(output_root), PipelineStep.STEP2, "alpha", "1" * 64, "2" * 64,
    )
    for row in rows:
        writer.append(row)
    return writer.finalize()


def _context(config, upstream):
    return StageRunContext(
        config.output_root, "alpha", config, upstream=_expectation(upstream),
    )


def _scores(_row, _models):
    return {
        "scoreq": 4.3,
        "urgent_pk": 3.6,
        "dnsmos": {"ovrl": 3.4, "sig": 3.5, "bak": 4.1, "p808": 3.9},
    }


def _test_scorer(scorer):
    return CanonicalStep3Scorer(scorer)


def test_step3_select_module_exposes_only_canonical_scorer_contract():
    assert not hasattr(select_module, "LegacyStep3MosScorerAdapter")


def test_step3_stage_requires_complete_canonical_scorer_protocol():
    class NonCallableProvenance:
        provenance_payload = {"scorer_execution": "test"}

        @staticmethod
        def score_task_groups(_groups):
            return ()

    class NonCallableScoring:
        score_task_groups = None

        @staticmethod
        def provenance_payload():
            return {"scorer_execution": "test"}

    with pytest.raises(TypeError, match="explicitly configured"):
        Step3MosSelectionStage()
    for scorer in (_scores, NonCallableProvenance(), NonCallableScoring()):
        with pytest.raises(TypeError, match="provenance_payload.*score_task_groups"):
            Step3MosSelectionStage(scorer=scorer)


@pytest.mark.parametrize("provenance", [None, {}, {"bad": object()}])
def test_step3_stage_fails_closed_on_missing_or_unstable_provenance(provenance):
    class Scorer:
        @staticmethod
        def provenance_payload():
            return provenance

        @staticmethod
        def score_task_groups(_groups):
            return ()

    with pytest.raises((TypeError, ValueError)):
        Step3MosSelectionStage(scorer=Scorer())


def test_step3_stage_snapshots_scorer_provenance_at_construction():
    provenance = {"scorer_execution": "test", "nested": {"version": 1}}

    class Scorer:
        @staticmethod
        def provenance_payload():
            return provenance

        @staticmethod
        def score_task_groups(_groups):
            return ()

    stage = Step3MosSelectionStage(scorer=Scorer())
    provenance["nested"]["version"] = 2

    assert stage.config_payload(_config("/tmp/unused"))["scorer_execution"] == {
        "scorer_execution": "test", "nested": {"version": 1},
    }


def test_step3_scores_selected_three_model_tuple_and_preserves_step2(tmp_path):
    config = _config(tmp_path / "out")
    upstream = _publish_step2(config.output_root, [_row()])

    index = Step3MosSelectionStage(scorer=_test_scorer(_scores)).run(_context(config, upstream))
    rows = list(iter_artifact_rows(index))

    assert tuple(STEP3_ROW_FIELDS[:len(STEP2_ROW_FIELDS)]) == STEP2_ROW_FIELDS
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == set(STEP3_ROW_FIELDS)
    assert {field: row[field] for field in STEP2_ROW_FIELDS} == _row()
    assert "selection" not in row
    assert "selection_policy" not in row
    assert row["mos_scores"] == _scores(_row(), ())


def test_step3_progress_callback_does_not_double_count_rows(tmp_path, monkeypatch):
    config = _config(tmp_path / "out", models=("scoreq",), min_mos_models=1)
    upstream = _publish_step2(config.output_root, [
        _row(segment_id="segment-1"), _row(segment_id="segment-2"),
    ])
    updates = []
    callbacks = []

    class Progress:
        def __init__(self, **_kwargs):
            pass

        def update(self, amount=1):
            updates.append(amount)

        def close(self):
            updates.append("closed")

    class Scorer:
        @staticmethod
        def provenance_payload():
            return {
                "scorer_execution": "test_canonical",
                "provider_class": "test_canonical",
                "device_class": "cpu",
                "cpu_fallback": False,
            }

        def set_progress_callback(self, callback):
            callbacks.append(callback)
            self._callback = callback

        def score_task_groups(self, groups):
            for group in groups:
                for line in Path(group["path"]).read_text(encoding="utf-8").splitlines():
                    task = json.loads(line)
                    self._callback(1)
                    yield {
                        "ordinal": task["ordinal"], "row": task["row"],
                        "models": task["models"], "scores": {"scoreq": 4.2},
                    }
            return {"provider_class": "test_canonical", "device_class": "cpu"}

    monkeypatch.setattr(select_module, "tqdm", Progress)
    index = Step3MosSelectionStage(scorer=Scorer()).run(_context(config, upstream))

    assert list(iter_artifact_rows(index))
    assert updates == [1, 1, "closed"]
    assert callable(callbacks[0])
    assert callbacks[-1] is None


def test_step3_scoring_cleanup_exhausts_all_resources_after_primary_error(
    tmp_path, monkeypatch,
):
    config = _config(tmp_path / "out", models=("scoreq",), min_mos_models=1)
    row = _row(segment_id="segment-cleanup")
    upstream = _publish_step2(config.output_root, [row])
    calls = []
    real_iter_bounded_jsonl = select_module.iter_bounded_jsonl

    class CloseFailIterator:
        def __init__(self, values, label):
            self._values = iter(values)
            self._label = label

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._values)

        def close(self):
            calls.append(f"close:{self._label}")
            raise RuntimeError(f"{self._label} close failed")

    def iter_jsonl(path, label, **kwargs):
        values = list(real_iter_bounded_jsonl(path, label, **kwargs))
        return CloseFailIterator(values, "compact")

    class Progress:
        def __init__(self, **_kwargs):
            pass

        def update(self, _amount=1):
            pass

        def close(self):
            calls.append("close:progress")
            raise RuntimeError("progress close failed")

    class Scorer:
        @staticmethod
        def provenance_payload():
            return {
                "scorer_execution": "test_canonical",
                "provider_class": "test_canonical",
                "device_class": "cpu",
            }

        def set_progress_callback(self, callback):
            calls.append("callback:set" if callback is not None else "callback:unset")
            if callback is None:
                raise RuntimeError("callback unset failed")

        @staticmethod
        def score_task_groups(_groups):
            return CloseFailIterator([{"invalid": True}], "scored")

    monkeypatch.setattr(select_module, "iter_bounded_jsonl", iter_jsonl)
    monkeypatch.setattr(select_module, "tqdm", Progress)
    stage = Step3MosSelectionStage(scorer=Scorer())
    rows = stage.produce_rows(_context(config, upstream), upstream, [row])

    with pytest.raises(ValueError, match="result identity is invalid") as raised:
        next(rows)

    assert calls == [
        "callback:set",
        "close:compact",
        "close:scored",
        "callback:unset",
        "close:progress",
    ]
    assert any("compact close failed" in note for note in raised.value.__notes__)


def test_step3_consumer_cancellation_finishes_all_cleanup_before_raising_failure(
    tmp_path, monkeypatch,
):
    config = _config(tmp_path / "out", models=("scoreq",), min_mos_models=1)
    row = _row(segment_id="segment-cancel")
    upstream = _publish_step2(config.output_root, [row])
    calls = []
    real_iter_bounded_jsonl = select_module.iter_bounded_jsonl

    class CloseFailIterator:
        def __init__(self, values, label):
            self._values = iter(values)
            self._label = label

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._values)

        def close(self):
            calls.append(f"close:{self._label}")
            raise RuntimeError(f"{self._label} close failed")

    def iter_jsonl(path, label, **kwargs):
        values = list(real_iter_bounded_jsonl(path, label, **kwargs))
        return CloseFailIterator(values, "compact")

    class Progress:
        def __init__(self, **_kwargs):
            pass

        def update(self, _amount=1):
            pass

        def close(self):
            calls.append("close:progress")
            raise RuntimeError("progress close failed")

    class Scorer:
        @staticmethod
        def provenance_payload():
            return {
                "scorer_execution": "test_canonical",
                "provider_class": "test_canonical",
                "device_class": "cpu",
            }

        def set_progress_callback(self, callback):
            calls.append("callback:set" if callback is not None else "callback:unset")
            if callback is None:
                raise RuntimeError("callback unset failed")

        @staticmethod
        def score_task_groups(groups):
            task = json.loads(Path(groups[0]["path"]).read_text(encoding="utf-8"))
            return CloseFailIterator([{
                "ordinal": task["ordinal"],
                "row": task["row"],
                "models": task["models"],
                "scores": {"scoreq": 4.2},
            }], "scored")

    monkeypatch.setattr(select_module, "iter_bounded_jsonl", iter_jsonl)
    monkeypatch.setattr(select_module, "tqdm", Progress)
    stage = Step3MosSelectionStage(scorer=Scorer())
    rows = stage.produce_rows(_context(config, upstream), upstream, [row])

    assert next(rows)["segment_id"] == "segment-cancel"
    with pytest.raises(RuntimeError, match="compact close failed"):
        rows.close()

    assert calls == [
        "callback:set",
        "close:compact",
        "close:scored",
        "callback:unset",
        "close:progress",
    ]


def test_step3_progress_fallback_completes_when_scorer_emits_no_frames(tmp_path, monkeypatch):
    config = _config(tmp_path / "out", models=("scoreq",), min_mos_models=1)
    upstream = _publish_step2(config.output_root, [
        _row(segment_id="segment-1"), _row(segment_id="segment-2"),
    ])
    updates = []

    class Progress:
        def __init__(self, **_kwargs):
            pass

        def update(self, amount=1):
            updates.append(amount)

        def close(self):
            updates.append("closed")

    class Scorer:
        @staticmethod
        def provenance_payload():
            return {
                "scorer_execution": "test_canonical",
                "provider_class": "test_canonical",
                "device_class": "cpu",
                "cpu_fallback": False,
            }

        def set_progress_callback(self, _callback):
            pass

        @staticmethod
        def score_task_groups(groups):
            for group in groups:
                for line in Path(group["path"]).read_text(encoding="utf-8").splitlines():
                    task = json.loads(line)
                    yield {
                        "ordinal": task["ordinal"], "row": task["row"],
                        "models": task["models"], "scores": {"scoreq": 4.2},
                    }
            return {"provider_class": "test_canonical", "device_class": "cpu"}

    monkeypatch.setattr(select_module, "tqdm", Progress)
    index = Step3MosSelectionStage(scorer=Scorer()).run(_context(config, upstream))

    assert len(list(iter_artifact_rows(index))) == 2
    assert updates == [1, 1, "closed"]


def test_step3_records_below_threshold_scores_without_a_selection_decision(tmp_path):
    config = _config(tmp_path / "out")
    upstream = _publish_step2(config.output_root, [_row()])

    def dns_dimension_fails(row, models):
        scores = _scores(row, models)
        scores["dnsmos"] = {**scores["dnsmos"], "bak": 3.9}
        return scores

    index = Step3MosSelectionStage(
        scorer=_test_scorer(dns_dimension_fails),
    ).run(_context(config, upstream))
    row = list(iter_artifact_rows(index))[0]

    assert row["mos_scores"]["dnsmos"]["bak"] == 3.9
    assert "selection" not in row
    assert row["segment_id"] == "segment-1"


def test_step3_fails_closed_on_wrong_step2_shape_before_scoring(tmp_path):
    config = _config(tmp_path / "out")
    malformed = _row()
    malformed.pop("boundary_reason")
    upstream = _publish_step2(config.output_root, [malformed])
    called = []

    with pytest.raises(ValueError, match="exact contract"):
        Step3MosSelectionStage(
            scorer=_test_scorer(lambda *_: called.append(True)),
        ).run(_context(config, upstream))

    assert called == []


def test_step3_rejects_language_not_exactly_bound_to_dataset_policy_before_scoring(tmp_path):
    config = _config(tmp_path / "out")
    mismatched_language = _row()
    mismatched_language["language"] = "fr"
    upstream = _publish_step2(config.output_root, [mismatched_language])
    called = []

    with pytest.raises(ValueError, match="language"):
        Step3MosSelectionStage(
            scorer=_test_scorer(lambda *_: called.append(True)),
        ).run(_context(config, upstream))

    assert called == []


def test_step3_rejects_unverified_gpu_route_before_any_spool_side_effect(
    tmp_path, monkeypatch,
):
    config = _config(tmp_path / "out", models=("nisqa",))
    upstream = _publish_step2(config.output_root, [_row()])
    validations = []

    class GpuScorer:
        @staticmethod
        def provenance_payload():
            return {
                "scorer_execution": "test_gpu",
                "provider_class": "cuda_subprocess",
                "device_class": "gpu",
            }

        @staticmethod
        def validate_model_tuples(model_tuples):
            validations.append(tuple(tuple(models) for models in model_tuples))
            raise ValueError(
                "requested MOS models must be a subset of parent-verified models"
            )

        @staticmethod
        def score_task_groups(_groups):
            pytest.fail("unverified route must not reach scorer dispatch")

    monkeypatch.setattr(
        select_module.tempfile,
        "TemporaryDirectory",
        lambda **_kwargs: pytest.fail("tuple spool must not be created"),
    )

    with pytest.raises(ValueError, match="subset of parent-verified"):
        Step3MosSelectionStage(scorer=GpuScorer()).run(_context(config, upstream))

    assert validations == [(('nisqa',),)]
    assert not (Path(config.output_root) / "step3").exists()


def test_step3_requires_bound_step2_artifact(tmp_path):
    config = _config(tmp_path / "out")
    wrong = ArtifactExpectation(PipelineStep.STEP1, "1" * 64, "2" * 64)
    context = StageRunContext(config.output_root, "alpha", config, upstream=wrong)

    with pytest.raises(ValueError, match="upstream step mismatch"):
        Step3MosSelectionStage(scorer=_test_scorer(_scores)).run(context)


def test_step3_config_digest_and_scoring_projection_ignore_threshold_changes(tmp_path):
    """MOS 打分与阈值无关：thresholds 不进入 Step3 投影或 config digest。"""
    base = _config(tmp_path / "out")
    loosened = _quality_profiles()
    for name in loosened:
        loosened[name]["thresholds"] = {
            "scoreq": 1.0,
            "urgent_pk": 1.0,
            "dnsmos": {"ovrl": 1.0, "sig": 1.0, "bak": 1.0, "p808": 1.0},
        }
    mutated = PipelineConfig(
        pipeline_generation=3,
        step="step3",
        output_root=str(tmp_path / "out"),
        datasets={"alpha": {
            "input_root": "/missing/input",
            "language": "en",
            "category": "read_speech",
            "quality_profile": "quality_high",
            "speaker": {"source": "none", "required": False},
        }},
        selection_policies={
            "global": {
                "mos_models": ["scoreq", "urgent_pk", "dnsmos"],
                "min_mos_models": 3,
            },
            "quality_profiles": loosened,
        },
    )
    stage = Step3MosSelectionStage(scorer=_test_scorer(_scores))
    expected_projection = {
        "dataset_id": "alpha",
        "language": "en",
        "category": "read_speech",
        "quality_profile": "quality_high",
        "route_identity": None,
        "mos_models": ["scoreq", "urgent_pk", "dnsmos"],
        "scoring_policy_hash": (
            "2d2a3f36d2a5b72191ee2f682db744a8335750c1cf81124ee0b8a3b0d1fcd4d4"
        ),
    }

    assert dict(base.mos_scoring_projection("alpha", "en")) == expected_projection
    assert dict(mutated.mos_scoring_projection("alpha", "en")) == expected_projection
    assert base.resolve_selection_policy("alpha", "en").mos_models == (
        "scoreq", "urgent_pk", "dnsmos",
    )
    assert mutated.resolve_selection_policy("alpha", "en").mos_models == (
        "scoreq", "urgent_pk", "dnsmos",
    )
    assert stage.config_digest(mutated) == stage.config_digest(base)


def test_step3_dataset_local_binding_ignores_other_dataset_policy_changes(tmp_path):
    base = _two_dataset_config(tmp_path / "out")
    beta_threshold_changed = _two_dataset_config(
        tmp_path / "out", beta_threshold=4.2,
    )
    beta_models_changed = _two_dataset_config(
        tmp_path / "out", beta_model="nisqa", beta_threshold=3.2,
    )
    alpha_models_changed = _two_dataset_config(
        tmp_path / "out", alpha_model="nisqa", alpha_threshold=3.2,
    )
    stage = Step3MosSelectionStage(
        scorer=_test_scorer(_scores), dataset_id="alpha",
    )

    payload = stage.config_payload(base)

    assert payload["policy"]["dataset_binding"]["dataset_id"] == "alpha"
    assert "beta" not in str(payload["policy"]["dataset_binding"])
    assert stage.config_digest(base) == stage.config_digest(beta_threshold_changed)
    assert stage.config_digest(base) == stage.config_digest(beta_models_changed)
    assert stage.config_digest(base) != stage.config_digest(alpha_models_changed)


def test_step3_unbound_stage_does_not_retain_context_dataset(tmp_path):
    config = _two_dataset_config(tmp_path / "out")
    stage = Step3MosSelectionStage(
        scorer=_test_scorer(lambda _row, models: {model: 4.2 for model in models}),
    )

    alpha_upstream = _publish_step2(config.output_root, [_row()])
    beta_writer = ArtifactWriter(
        config.output_root, PipelineStep.STEP2, "beta", "3" * 64, "4" * 64,
    )
    beta_writer.append(_row(dataset_id="beta"))
    beta_upstream = beta_writer.finalize()
    alpha_context = _context(config, alpha_upstream)
    beta_context = StageRunContext(
        config.output_root,
        "beta",
        config,
        upstream=_expectation(beta_upstream),
    )

    assert stage.run(alpha_context).dataset_id == "alpha"
    assert stage.run(beta_context).dataset_id == "beta"
    assert stage._dataset_id is None
    assert "projections" in stage.config_payload(config)["policy"]


def test_step3_production_plan_global_model_order_does_not_change_local_digest(
    tmp_path,
):
    def config_for(beta_models):
        config = PipelineConfig(
            pipeline_generation=3,
            step="step3",
            output_root=str(tmp_path / "out"),
            datasets={
                "aardvark": {
                    "input_root": "/missing/aardvark",
                    "language": "en",
                    "category": "read_speech",
                    "quality_profile": "quality_high",
                    "speaker": {"source": "none", "required": False},
                },
                "alpha": {
                    "input_root": "/missing/alpha",
                    "language": "en",
                    "category": "read_speech",
                    "quality_profile": "quality_high",
                    "speaker": {"source": "none", "required": False},
                },
            },
            selection_policies={
                "global": {"mos_models": ["scoreq"], "min_mos_models": 1},
                "quality_profiles": _quality_profiles(),
                "datasets": {
                    "alpha": {
                        "mos_models": ["utmos", "scoreq"],
                        "min_mos_models": 2,
                    },
                    "aardvark": {
                        "mos_models": beta_models,
                        "min_mos_models": len(beta_models),
                    },
                },
            },
        )
        config.scoreq_model_dir = "/models/scoreq"
        config.utmos_model_path = "/models/utmos.pt"
        config.utmos_model_version = "utmos-v1"
        config.utmos_model_sha256 = "a" * 64
        config.nisqa_model_path = "/models/nisqa.pt"
        config.nisqa_model_version = "nisqa-v1"
        config.nisqa_model_sha256 = "b" * 64
        return config

    class PlanScorer:
        def __init__(self, plan):
            self._plan = plan

        def provenance_payload(self):
            models = self._plan.selected_models
            return {
                "scorer_execution": "step3_mos_pool.v6",
                "provider_class": "cuda_subprocess",
                "device_class": "gpu",
                "protocol": "audio_filter_pipeline.step3_mos_worker.v6",
                "models": list(models),
                "model_options": {
                    asset.model: asset.as_mapping() for asset in self._plan.assets
                },
                "assets": {
                    "selected_models": list(models),
                    "models": {
                        model: {"identity": f"{model}-asset"} for model in models
                    },
                },
                "asset_load_binding": {"version": "shared"},
                "worker_script": {"sha256": "shared-worker"},
                "cpu_fallback": False,
            }

        @staticmethod
        def score_task_groups(_groups):
            return ()

    baseline = config_for(["scoreq", "utmos"])
    changed = config_for(["nisqa", "utmos", "scoreq"])
    baseline_plan = MosExecutionPlan.from_config(baseline)
    changed_plan = MosExecutionPlan.from_config(changed)
    baseline_stage = Step3MosSelectionStage(
        scorer=PlanScorer(baseline_plan), dataset_id="alpha",
    )
    changed_stage = Step3MosSelectionStage(
        scorer=PlanScorer(changed_plan), dataset_id="alpha",
    )

    assert baseline_plan.selected_models == ("scoreq", "utmos")
    assert changed_plan.selected_models == ("nisqa", "utmos", "scoreq")
    assert baseline.dataset_mos_scoring_binding("alpha")["projections"][0]["mos_models"] == (
        "utmos", "scoreq",
    )
    assert changed.dataset_mos_scoring_binding("alpha") == (
        baseline.dataset_mos_scoring_binding("alpha")
    )
    assert baseline_stage.config_digest(baseline) == changed_stage.config_digest(changed)


def test_step3_projects_model_scoped_runtime_provenance_by_dataset(tmp_path):
    config = _two_dataset_config(
        tmp_path / "out", alpha_model="scoreq", beta_model="dnsmos",
    )
    beta_models_changed = _two_dataset_config(
        tmp_path / "out", alpha_model="scoreq", beta_model="nisqa",
    )

    class Scorer:
        def __init__(self, *, assets_by_model, protocol):
            self._assets_by_model = assets_by_model
            self._protocol = protocol

        def provenance_payload(self):
            models = list(self._assets_by_model)
            return {
                "scorer_execution": "step3_mos_pool.v6",
                "provider_class": "cuda_subprocess",
                "device_class": "gpu",
                "protocol": self._protocol,
                "models": models,
                "model_options": {
                    model: {"asset_identity": identity}
                    for model, identity in self._assets_by_model.items()
                },
                "assets": {
                    "selected_models": models,
                    "models": {
                        model: {"verified_model_tree_sha256": identity}
                        for model, identity in self._assets_by_model.items()
                    },
                },
                "scoreq_model_dir": "/models/scoreq",
                "scoreq_model_tree_sha256": self._assets_by_model["scoreq"],
                "dnsmos_model_dir": "/models/dnsmos" if "dnsmos" in models else None,
                "dnsmos_model_tree_sha256": self._assets_by_model.get("dnsmos"),
                "worker_script": {"sha256": "shared-worker"},
                "asset_load_binding": {"version": "shared"},
                "cpu_fallback": False,
            }

        @staticmethod
        def score_task_groups(_groups):
            return ()

    baseline = Step3MosSelectionStage(
        scorer=Scorer(
            assets_by_model={"scoreq": "scoreq-v1", "dnsmos": "dnsmos-v1"},
            protocol="v1",
        ),
        dataset_id="alpha",
    )
    beta_asset_changed = Step3MosSelectionStage(
        scorer=Scorer(
            assets_by_model={"scoreq": "scoreq-v1", "dnsmos": "dnsmos-v2"},
            protocol="v1",
        ),
        dataset_id="alpha",
    )
    beta_tuple_changed = Step3MosSelectionStage(
        scorer=Scorer(
            assets_by_model={"scoreq": "scoreq-v1", "nisqa": "nisqa-v1"},
            protocol="v1",
        ),
        dataset_id="alpha",
    )
    alpha_asset_changed = Step3MosSelectionStage(
        scorer=Scorer(
            assets_by_model={"scoreq": "scoreq-v2", "dnsmos": "dnsmos-v1"},
            protocol="v1",
        ),
        dataset_id="alpha",
    )
    protocol_changed = Step3MosSelectionStage(
        scorer=Scorer(
            assets_by_model={"scoreq": "scoreq-v1", "dnsmos": "dnsmos-v1"},
            protocol="v2",
        ),
        dataset_id="alpha",
    )

    payload = baseline.config_payload(config)["scorer_execution"]

    assert payload["models"] == ["scoreq"]
    assert payload["model_options"] == {"scoreq": {"asset_identity": "scoreq-v1"}}
    assert payload["assets"] == {
        "selected_models": ["scoreq"],
        "models": {"scoreq": {"verified_model_tree_sha256": "scoreq-v1"}},
    }
    assert "dnsmos_model_dir" not in payload
    assert "dnsmos_model_tree_sha256" not in payload
    assert baseline.config_digest(config) == beta_asset_changed.config_digest(config)
    assert baseline.config_digest(config) == beta_tuple_changed.config_digest(beta_models_changed)
    assert baseline.config_digest(config) != alpha_asset_changed.config_digest(config)
    assert baseline.config_digest(config) != protocol_changed.config_digest(config)


def test_step3_projected_scorer_identity_allows_alpha_cache_hit_after_beta_tuple_change(
    tmp_path,
):
    base = _two_dataset_config(
        tmp_path / "out", alpha_model="scoreq", beta_model="utmos",
    )
    beta_changed = _two_dataset_config(
        tmp_path / "out", alpha_model="scoreq", beta_model="nisqa",
    )
    upstream = _publish_step2(base.output_root, [_row()])

    class Scorer(CanonicalStep3Scorer):
        def __init__(self, models, scorer):
            super().__init__(scorer)
            self._models = models

        def provenance_payload(self):
            return {
                "scorer_execution": "test_canonical",
                "provider_class": "test_canonical",
                "device_class": "cpu",
                "models": self._models,
                "model_options": {
                    model: {"identity": f"{model}-v1"} for model in self._models
                },
                "assets": {
                    "selected_models": self._models,
                    "models": {
                        model: {"identity": f"{model}-v1"} for model in self._models
                    },
                },
            }

    published = Step3MosSelectionStage(
        scorer=Scorer(
            ["scoreq", "utmos"], lambda _row, _models: {"scoreq": 4.2},
        ),
        dataset_id="alpha",
    ).run(_context(base, upstream))
    resumed = Step3MosSelectionStage(
        scorer=Scorer(
            ["scoreq", "nisqa"],
            lambda *_args: pytest.fail("B tuple change must not rescore Alpha"),
        ),
        dataset_id="alpha",
    ).run(_context(beta_changed, upstream))

    assert resumed.serialized() == published.serialized()


def test_step3_keeps_shared_scorer_provenance_globally_bound(tmp_path):
    config = _two_dataset_config(tmp_path / "out")

    class Scorer:
        def __init__(self, asset_identity):
            self._asset_identity = asset_identity

        def provenance_payload(self):
            return {
                "scorer_execution": "test_canonical",
                "provider_class": "test_canonical",
                "device_class": "cpu",
                "verified_shared_asset": self._asset_identity,
            }

        @staticmethod
        def score_task_groups(_groups):
            return ()

    first = Step3MosSelectionStage(
        scorer=Scorer("asset-v1"), dataset_id="alpha",
    )
    changed_shared_asset = Step3MosSelectionStage(
        scorer=Scorer("asset-v2"), dataset_id="alpha",
    )

    assert first.config_digest(config) != changed_shared_asset.config_digest(config)


def test_step3_published_alpha_artifact_resumes_after_beta_policy_change(tmp_path):
    base = _two_dataset_config(tmp_path / "out")
    beta_changed = _two_dataset_config(
        tmp_path / "out", beta_model="nisqa", beta_threshold=4.0,
    )
    upstream = _publish_step2(base.output_root, [_row()])
    published = Step3MosSelectionStage(
        scorer=_test_scorer(lambda _row, _models: {"scoreq": 4.2}),
        dataset_id="alpha",
    ).run(_context(base, upstream))

    resumed = Step3MosSelectionStage(
        scorer=_test_scorer(lambda *_args: pytest.fail("B policy change must resume A")),
        dataset_id="alpha",
    ).run(_context(beta_changed, upstream))

    assert resumed.serialized() == published.serialized()


def test_step3_alpha_model_change_fails_closed_before_rescoring(tmp_path):
    base = _two_dataset_config(tmp_path / "out")
    alpha_changed = _two_dataset_config(
        tmp_path / "out", alpha_model="nisqa", alpha_threshold=3.2,
    )
    upstream = _publish_step2(base.output_root, [_row()])
    Step3MosSelectionStage(
        scorer=_test_scorer(lambda _row, _models: {"scoreq": 4.2}),
        dataset_id="alpha",
    ).run(_context(base, upstream))

    with pytest.raises(ValueError, match="expected config digest mismatch"):
        Step3MosSelectionStage(
            scorer=_test_scorer(lambda *_args: pytest.fail("must not rescore")),
            dataset_id="alpha",
        ).run(_context(alpha_changed, upstream))


def test_step3_scoring_projection_changes_with_model_tuple(tmp_path):
    base = _config(tmp_path / "out")
    changed = _config(
        tmp_path / "out",
        models=("dnsmos", "scoreq"),
        min_mos_models=1,
    )

    base_projection = dict(base.mos_scoring_projection("alpha", "en"))
    changed_projection = dict(changed.mos_scoring_projection("alpha", "en"))

    assert tuple(base_projection["mos_models"]) == ("scoreq", "urgent_pk", "dnsmos")
    assert tuple(changed_projection["mos_models"]) == ("dnsmos", "scoreq")
    assert changed_projection != base_projection
    assert changed_projection["scoring_policy_hash"] != base_projection["scoring_policy_hash"]


def test_step3_config_digest_is_sensitive_to_row_contract_fields():
    """行契约字段属于 Step3 digest：未来新增字段必须 bump row_contract 版本。"""
    config = _config("/tmp/unused")
    stage = Step3MosSelectionStage(scorer=_test_scorer(_scores))
    payload = stage.config_payload(config)

    assert payload["row_contract"]["version"] == 4
    assert payload["row_contract"]["fields"] == list(STEP3_ROW_FIELDS)
    assert "selection_policies" not in payload
    assert "datasets" not in payload


def test_step3_dispatches_all_model_tuples_in_one_pool_call(tmp_path):
    config = _config(
        tmp_path / "out",
        models=("scoreq",),
        min_mos_models=1,
        source_policies={
            "speaker/two.wav": {
                "mos_models": ["utmos", "nisqa"],
                "min_mos_models": 1,
            },
        },
    )
    rows = [
        {**_row(segment_id="segment-1"), "source_relative_path": "speaker/one.wav"},
        {**_row(segment_id="segment-2"), "source_relative_path": "speaker/two.wav"},
        {**_row(segment_id="segment-3"), "source_relative_path": "speaker/one.wav"},
    ]
    upstream = _publish_step2(config.output_root, rows)
    calls = []

    class Scorer:
        def __getattr__(self, name):
            if name in {"score_rows", "open_source_group", "source_identity", "score_batch"}:
                pytest.fail(f"Stage must not probe legacy scorer capability: {name}")
            raise AttributeError(name)

        @staticmethod
        def provenance_payload():
            return {
                "scorer_execution": "test_canonical",
                "provider_class": "test_canonical",
                "device_class": "cpu",
                "cpu_fallback": False,
            }

        @staticmethod
        def score_task_groups(groups):
            calls.append([(tuple(group["models"]), group["row_count"]) for group in groups])
            tasks = []
            for group in groups:
                for line in Path(group["path"]).read_text(encoding="utf-8").splitlines():
                    task = __import__("json").loads(line)
                    scores = ({"scoreq": 4.2} if task["models"] == ["scoreq"]
                              else {"utmos": 4.0, "nisqa": 3.9})
                    tasks.append({
                        "ordinal": task["ordinal"], "row": task["row"],
                        "models": task["models"], "scores": scores,
                    })
            yield from sorted(tasks, key=lambda task: task["ordinal"])
            return {
                "provider_class": "test_canonical",
                "device_class": "cpu",
            }

    index = Step3MosSelectionStage(scorer=Scorer()).run(_context(config, upstream))
    output = list(iter_artifact_rows(index))

    assert calls == [[(("scoreq",), 2), (("utmos", "nisqa"), 1)]]
    assert [row["segment_id"] for row in output] == [
        "segment-1", "segment-2", "segment-3",
    ]


def test_step3_uses_one_compact_index_scan_and_preserves_rows(tmp_path, monkeypatch):
    config = _config(tmp_path / "out", models=("scoreq",))
    rows = [
        _row(segment_id="segment-1"),
        _row(segment_id="segment-2"),
    ]
    upstream = _publish_step2(config.output_root, rows)
    scan_labels = []
    real_iter_bounded_jsonl = select_module.iter_bounded_jsonl

    def track_scans(path, label, **kwargs):
        scan_labels.append((Path(path).name, label))
        yield from real_iter_bounded_jsonl(path, label, **kwargs)

    class Scorer:
        @staticmethod
        def provenance_payload():
            return {
                "scorer_execution": "test_canonical",
                "provider_class": "test_canonical",
                "device_class": "cpu",
                "cpu_fallback": False,
            }

        @staticmethod
        def score_task_groups(groups):
            tasks = []
            for group in groups:
                with open(group["path"], encoding="utf-8") as stream:
                    tasks.extend(json.loads(line) for line in stream)
            for task in sorted(tasks, key=lambda item: item["ordinal"]):
                yield {
                    "ordinal": task["ordinal"], "row": task["row"],
                    "models": task["models"], "scores": {"scoreq": 4.2},
                }
            return {
                "provider_class": "test_canonical",
                "device_class": "cpu",
            }

    monkeypatch.setattr(select_module, "iter_bounded_jsonl", track_scans)
    index = Step3MosSelectionStage(scorer=Scorer()).run(_context(config, upstream))
    output = list(iter_artifact_rows(index))

    assert scan_labels == [
        ("compact-task-index.jsonl", "Step3 compact task index"),
        (
            Path(Step3MosSelectionStage.execution_attestation_path(index)).name,
            "Step3 execution attestation sidecar",
        ),
    ]
    assert [
        {field: row[field] for field in STEP2_ROW_FIELDS} for row in output
    ] == rows
    assert [row["mos_scores"] for row in output] == [
        {"scoreq": 4.2}, {"scoreq": 4.2},
    ]


def test_step3_compact_digest_rejects_scorer_row_mutation(tmp_path):
    config = _config(tmp_path / "out", models=("scoreq",))
    upstream = _publish_step2(config.output_root, [_row()])

    class Scorer:
        @staticmethod
        def provenance_payload():
            return {
                "scorer_execution": "test_canonical",
                "provider_class": "test_canonical",
                "device_class": "cpu",
                "cpu_fallback": False,
            }

        @staticmethod
        def score_task_groups(groups):
            task = json.loads(Path(groups[0]["path"]).read_text(encoding="utf-8"))
            task["row"]["segment_id"] = "replaced-segment"
            yield {
                "ordinal": task["ordinal"], "row": task["row"],
                "models": task["models"], "scores": {"scoreq": 4.2},
            }

    with pytest.raises(ValueError, match="result identity"):
        Step3MosSelectionStage(scorer=Scorer()).run(_context(config, upstream))


def test_step3_captures_generator_completion_attestation(tmp_path):
    config = _config(tmp_path / "out", models=("scoreq",))
    upstream = _publish_step2(config.output_root, [_row()])
    attestation = {
        "provider_class": "test_canonical",
        "device_class": "cpu",
        "runtime": {"execution_state": "scored"},
    }

    class Scorer:
        @staticmethod
        def provenance_payload():
            return {
                "scorer_execution": "test_canonical",
                "provider_class": "test_canonical",
                "device_class": "cpu",
                "cpu_fallback": False,
            }

        @staticmethod
        def score_task_groups(groups):
            task = json.loads(Path(groups[0]["path"]).read_text(encoding="utf-8"))
            yield {
                "ordinal": task["ordinal"], "row": task["row"],
                "models": task["models"], "scores": {"scoreq": 4.2},
            }
            return attestation

    stage = Step3MosSelectionStage(scorer=Scorer())
    index = stage.run(_context(config, upstream))
    sidecar = json.loads(
        Path(stage.execution_attestation_path(index)).read_text(encoding="utf-8")
    )

    assert sidecar["execution"] == attestation


@pytest.mark.parametrize("target", ["compact", "scorer"])
def test_step3_rejects_boolean_ordinal_identity(tmp_path, monkeypatch, target):
    config = _config(tmp_path / "out", models=("scoreq",))
    upstream = _publish_step2(config.output_root, [_row()])
    real_decode_json_line = select_module.decode_json_line

    def decode_json_line(payload, label, **kwargs):
        value = real_decode_json_line(payload, label, **kwargs)
        if target == "compact" and label == "Step3 compact task index row":
            value["ordinal"] = False
        return value

    class Scorer:
        @staticmethod
        def provenance_payload():
            return {
                "scorer_execution": "test_canonical",
                "provider_class": "test_canonical",
                "device_class": "cpu",
                "cpu_fallback": False,
            }

        @staticmethod
        def score_task_groups(groups):
            task = json.loads(Path(groups[0]["path"]).read_text(encoding="utf-8"))
            yield {
                "ordinal": False if target == "scorer" else task["ordinal"],
                "row": task["row"], "models": task["models"],
                "scores": {"scoreq": 4.2},
            }
            return {
                "provider_class": "test_canonical",
                "device_class": "cpu",
            }

    monkeypatch.setattr(select_module, "decode_json_line", decode_json_line)
    with pytest.raises(ValueError, match="compact task index row|result identity"):
        Step3MosSelectionStage(scorer=Scorer()).run(_context(config, upstream))


def test_step3_rejects_malformed_compact_task_digest(tmp_path, monkeypatch):
    config = _config(tmp_path / "out", models=("scoreq",))
    upstream = _publish_step2(config.output_root, [_row()])
    real_decode_json_line = select_module.decode_json_line

    def decode_json_line(payload, label, **kwargs):
        value = real_decode_json_line(payload, label, **kwargs)
        if label == "Step3 compact task index row":
            value["task_digest"] = "not-a-sha256"
        return value

    monkeypatch.setattr(select_module, "decode_json_line", decode_json_line)
    with pytest.raises(ValueError, match="compact task index row"):
        Step3MosSelectionStage(
            scorer=_test_scorer(lambda _row, _models: {"scoreq": 4.2}),
        ).run(_context(config, upstream))


def test_step3_disk_segment_index_rejects_duplicate_segment_id(tmp_path):
    config = _config(tmp_path / "out", models=("scoreq",))
    upstream = _publish_step2(config.output_root, [_row(), _row()])
    calls = []

    with pytest.raises(ValueError, match="duplicate segment_id"):
        Step3MosSelectionStage(
            scorer=_test_scorer(lambda *_args: calls.append(True)),
        ).run(_context(config, upstream))

    assert calls == []


def test_step3_aggregate_spool_budget_fails_before_scorer_dispatch(
    tmp_path, monkeypatch,
):
    config = _config(tmp_path / "out", models=("scoreq",))
    upstream = _publish_step2(config.output_root, [_row()])
    monkeypatch.setattr(select_module, "_STEP3_SPOOL_MAX_TOTAL_BYTES", 1)
    calls = []

    with pytest.raises(ValueError, match="aggregate budget"):
        Step3MosSelectionStage(
            scorer=_test_scorer(lambda *_args: calls.append(True)),
        ).run(_context(config, upstream))

    assert calls == []


def test_step3_rolls_tuple_spool_into_bounded_ordered_parts(tmp_path, monkeypatch):
    config = _config(tmp_path / "out", models=("scoreq",))
    rows = [_row(segment_id=f"segment-{index}") for index in range(5)]
    upstream = _publish_step2(config.output_root, rows)
    descriptors = []
    part_sizes = []

    class Scorer(CanonicalStep3Scorer):
        def score_task_groups(self, groups):
            descriptors.extend(dict(group) for group in groups)
            part_sizes.extend(Path(group["path"]).stat().st_size for group in groups)
            return (yield from super().score_task_groups(groups))

    monkeypatch.setattr(select_module, "TUPLE_SPOOL_MAX_FILE_BYTES", 2_200)
    index = Step3MosSelectionStage(
        scorer=Scorer(lambda _row, _models: {"scoreq": 4.2}),
    ).run(_context(config, upstream))

    assert select_module._STEP3_SPOOL_MAX_TOTAL_BYTES == 4 * 1024 * 1024 * 1024
    assert select_module._COMPACT_INDEX_MAX_FILE_BYTES == 512 * 1024 * 1024
    assert len(descriptors) == 3
    assert [descriptor["row_count"] for descriptor in descriptors] == [2, 2, 1]
    assert [Path(descriptor["path"]).name for descriptor in descriptors] == [
        "tuple-0-part-0.jsonl", "tuple-0-part-1.jsonl", "tuple-0-part-2.jsonl",
    ]
    assert all(0 < size <= 2_200 for size in part_sizes)
    assert index.row_count == 5


def test_step3_tuple_spool_exact_boundary_does_not_roll_early(tmp_path, monkeypatch):
    config = _config(tmp_path / "out", models=("scoreq",))
    rows = [_row(segment_id=f"segment-{index}") for index in range(3)]
    upstream = _publish_step2(config.output_root, rows)
    descriptors = []
    part_sizes = []
    real_encode = select_module.encode_json_line

    def encode(value, label, **kwargs):
        payload = real_encode(value, label, **kwargs)
        if label in {"Step3 tuple task", "Step3 task digest payload"}:
            target = 1_500
            assert len(payload) <= target
            payload = payload[:-1] + b" " * (target - len(payload)) + b"\n"
        return payload

    class Scorer(CanonicalStep3Scorer):
        def score_task_groups(self, groups):
            descriptors.extend(dict(group) for group in groups)
            part_sizes.extend(Path(group["path"]).stat().st_size for group in groups)
            return (yield from super().score_task_groups(groups))

    monkeypatch.setattr(select_module, "encode_json_line", encode)
    monkeypatch.setattr(select_module, "TUPLE_SPOOL_MAX_FILE_BYTES", 3_000)
    index = Step3MosSelectionStage(
        scorer=Scorer(lambda _row, _models: {"scoreq": 4.2}),
    ).run(_context(config, upstream))

    assert [descriptor["row_count"] for descriptor in descriptors] == [2, 1]
    assert part_sizes == [3_000, 1_500]
    assert index.row_count == 3


def test_step3_aggregate_budget_keeps_closed_part_bytes(tmp_path, monkeypatch):
    config = _config(tmp_path / "out", models=("scoreq",))
    rows = [_row(segment_id=f"segment-{index}") for index in range(20)]
    upstream = _publish_step2(config.output_root, rows)
    opened_tuple_parts = []
    scorer_calls = []
    real_encode = select_module.encode_json_line
    real_open = select_module.open_new_regular_stream

    def encode(value, label, **kwargs):
        payload = real_encode(value, label, **kwargs)
        if label in {"Step3 tuple task", "Step3 task digest payload"}:
            target = 1_500
            assert len(payload) <= target
            payload = payload[:-1] + b" " * (target - len(payload)) + b"\n"
        return payload

    def open_stream(path, label):
        if label == "Step3 tuple task spool":
            opened_tuple_parts.append(Path(path).name)
        return real_open(path, label)

    monkeypatch.setattr(select_module, "encode_json_line", encode)
    monkeypatch.setattr(select_module, "open_new_regular_stream", open_stream)
    monkeypatch.setattr(select_module, "TUPLE_SPOOL_MAX_FILE_BYTES", 1_500)
    monkeypatch.setattr(
        select_module, "_STEP3_SPOOL_MAX_TOTAL_BYTES",
        select_module._SEGMENT_INDEX_CACHE_BYTES + 30_000,
    )

    with pytest.raises(ValueError, match="aggregate budget"):
        Step3MosSelectionStage(
            scorer=_test_scorer(lambda *_args: scorer_calls.append(True)),
        ).run(_context(config, upstream))

    assert len(opened_tuple_parts) > 1
    assert scorer_calls == []


def test_step3_tuple_spool_limit_fails_before_scorer_dispatch(
    tmp_path, monkeypatch,
):
    config = _config(tmp_path / "out", models=("scoreq",))
    upstream = _publish_step2(config.output_root, [_row()])
    monkeypatch.setattr(select_module, "TUPLE_SPOOL_MAX_FILE_BYTES", 1)
    calls = []

    with pytest.raises(ValueError, match="tuple JSONL row exceeds physical part limit"):
        Step3MosSelectionStage(
            scorer=_test_scorer(lambda *_args: calls.append(True)),
        ).run(_context(config, upstream))

    assert calls == []


def test_canonical_test_scorer_keeps_one_open_part_per_logical_tuple(
    tmp_path, monkeypatch,
):
    models = ["scoreq"]
    groups = []
    for ordinal in range(12):
        path = tmp_path / f"part-{ordinal}.jsonl"
        path.write_text(json.dumps({
            "ordinal": ordinal,
            "row": _row(segment_id=f"segment-{ordinal}"),
            "models": models,
            "scoring_policy_hash": "a" * 64,
        }) + "\n", encoding="utf-8")
        groups.append({"models": models, "path": str(path), "row_count": 1})
    real_iter = step3_test_helpers.iter_bounded_jsonl
    active = 0
    peak = 0

    def iter_jsonl(*args, **kwargs):
        iterator = iter(real_iter(*args, **kwargs))

        def tracked():
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                yield from iterator
            finally:
                active -= 1
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()

        return tracked()

    monkeypatch.setattr(step3_test_helpers, "iter_bounded_jsonl", iter_jsonl)
    scorer = CanonicalStep3Scorer(lambda *_args: {"scoreq": 4.0})

    assert [row["ordinal"] for row in scorer.score_task_groups(groups)] == list(range(12))
    assert peak == 1
    assert active == 0


def test_canonical_test_scorer_rejects_invalid_part_descriptors_before_open(tmp_path):
    scorer = CanonicalStep3Scorer(lambda *_args: pytest.fail("must not score"))
    first = str(tmp_path / "first.jsonl")
    second = str(tmp_path / "second.jsonl")

    with pytest.raises(ValueError, match="path is duplicate"):
        list(scorer.score_task_groups([
            {"models": ["scoreq"], "path": first, "row_count": 1},
            {"models": ["scoreq"], "path": first, "row_count": 1},
        ]))
    with pytest.raises(ValueError, match="parts must be contiguous"):
        list(scorer.score_task_groups([
            {"models": ["scoreq"], "path": first, "row_count": 1},
            {"models": ["utmos"], "path": second, "row_count": 1},
            {
                "models": ["scoreq"],
                "path": str(tmp_path / "third.jsonl"),
                "row_count": 1,
            },
        ]))


def test_step3_logical_sqlite_budget_rejects_long_segment_id_before_dispatch(
    tmp_path, monkeypatch,
):
    config = _config(tmp_path / "out", models=("scoreq",))
    upstream = _publish_step2(
        config.output_root, [_row(segment_id="segment-" + "x" * 16_384)],
    )
    monkeypatch.setattr(
        select_module,
        "_STEP3_SPOOL_MAX_TOTAL_BYTES",
        select_module._SEGMENT_INDEX_CACHE_BYTES + 8 * 1024,
    )
    calls = []

    with pytest.raises(ValueError, match="aggregate budget"):
        Step3MosSelectionStage(
            scorer=_test_scorer(lambda *_args: calls.append(True)),
        ).run(_context(config, upstream))

    assert calls == []


def test_step3_active_tuple_writer_cleanup_failure_replaces_generator_exit(
    tmp_path, monkeypatch,
):
    config = _config(tmp_path / "out", models=("scoreq",))
    upstream = _publish_step2(config.output_root, [_row()])
    real_open = select_module.open_new_regular_stream
    failing_streams = []

    class CloseFailingStream:
        def __init__(self, stream):
            self._stream = stream

        def __getattr__(self, name):
            return getattr(self._stream, name)

        def close(self):
            self._stream.close()
            raise RuntimeError("active tuple writer close failed")

    def open_stream(path, label):
        stream = real_open(path, label)
        if label == "Step3 tuple task spool":
            stream = CloseFailingStream(stream)
            failing_streams.append(stream)
        return stream

    def interrupted_rows():
        yield _row()
        raise GeneratorExit

    monkeypatch.setattr(select_module, "open_new_regular_stream", open_stream)
    rows = Step3MosSelectionStage(
        scorer=_test_scorer(lambda *_args: pytest.fail("must not score")),
    ).produce_rows(_context(config, upstream), upstream, interrupted_rows())

    with pytest.raises(RuntimeError, match="active tuple writer close failed"):
        next(iter(rows))

    assert len(failing_streams) == 1


def test_step3_aggregate_budget_cleanup_removes_owned_spools(tmp_path, monkeypatch):
    config = _config(tmp_path / "out", models=("scoreq",))
    upstream = _publish_step2(config.output_root, [_row()])
    temporary_root = tmp_path / "owned-spools"

    class OwnedTemporaryDirectory:
        def __init__(self, **_kwargs):
            temporary_root.mkdir()

        def __enter__(self):
            return str(temporary_root)

        def __exit__(self, *_args):
            for path in temporary_root.iterdir():
                path.unlink()
            temporary_root.rmdir()

    monkeypatch.setattr(select_module.tempfile, "TemporaryDirectory", OwnedTemporaryDirectory)
    monkeypatch.setattr(select_module, "_STEP3_SPOOL_MAX_TOTAL_BYTES", 1)

    with pytest.raises(ValueError, match="aggregate budget"):
        Step3MosSelectionStage(
            scorer=_test_scorer(lambda *_args: pytest.fail("must not dispatch")),
        ).run(_context(config, upstream))

    assert not temporary_root.exists()


def test_step3_spool_budget_scales_with_row_count(tmp_path, monkeypatch):
    # 回归：固定聚合上限曾让 ~5M 行的 gigaspeech 在 Step3 起始即失败；
    # 预算必须随 row_count 伸缩，floor 不再随数据集规模误伤。
    config = _config(tmp_path / "out", models=("scoreq",))
    rows = [_row(segment_id=f"segment-{index}") for index in range(5_000)]
    upstream = _publish_step2(config.output_root, rows)
    monkeypatch.setattr(
        select_module,
        "_STEP3_SPOOL_MAX_TOTAL_BYTES",
        select_module._SEGMENT_INDEX_CACHE_BYTES + 1_000,
    )
    calls = []

    def score(_row, _models):
        calls.append(True)
        return {"scoreq": 4.2}

    index = Step3MosSelectionStage(
        scorer=_test_scorer(score),
    ).run(_context(config, upstream))

    assert index.row_count == 5_000
    assert len(calls) == 5_000


def test_step3_spool_budget_scales_compact_index_limit(tmp_path, monkeypatch):
    config = _config(tmp_path / "out", models=("scoreq",))
    rows = [_row(segment_id=f"segment-{index}") for index in range(5_000)]
    upstream = _publish_step2(config.output_root, rows)
    monkeypatch.setattr(
        select_module,
        "_COMPACT_INDEX_MAX_FILE_BYTES",
        1,
    )

    index = Step3MosSelectionStage(
        scorer=_test_scorer(lambda _row, _models: {"scoreq": 4.2}),
    ).run(_context(config, upstream))

    assert index.row_count == 5_000


def test_step3_spool_budget_env_override(tmp_path, monkeypatch):
    env_var = select_module._SPOOL_BUDGET_ENV_VAR
    config = _config(tmp_path / "fail-out", models=("scoreq",))
    upstream = _publish_step2(config.output_root, [_row()])
    calls = []
    monkeypatch.setenv(env_var, "1")

    with pytest.raises(ValueError, match="aggregate budget"):
        Step3MosSelectionStage(
            scorer=_test_scorer(lambda *_args: calls.append(True)),
        ).run(_context(config, upstream))

    assert calls == []
    for bad_value in ("0", "-1", "abc", ""):
        monkeypatch.setenv(env_var, bad_value)
        with pytest.raises(ValueError, match=env_var):
            select_module._spool_budget_bytes(1)


def test_step3_spool_budget_env_override_allows_large_run(tmp_path, monkeypatch):
    env_var = select_module._SPOOL_BUDGET_ENV_VAR
    config = _config(tmp_path / "out", models=("scoreq",))
    rows = [_row(segment_id=f"segment-{index}") for index in range(5_000)]
    upstream = _publish_step2(config.output_root, rows)
    monkeypatch.setattr(select_module, "_STEP3_SPOOL_MAX_TOTAL_BYTES", 1)
    monkeypatch.setenv(env_var, str(64 * 1024**3))

    index = Step3MosSelectionStage(
        scorer=_test_scorer(lambda _row, _models: {"scoreq": 4.2}),
    ).run(_context(config, upstream))

    assert index.row_count == 5_000


def test_step3_spool_budget_helpers_scale_and_keep_floor():
    assert select_module._spool_budget_bytes(0) == 4 * 1024**3
    # floor 与按行伸缩的交叠点是 4 GiB / 4 KiB = 1_048_576 行
    assert select_module._spool_budget_bytes(10**6) == 4 * 1024**3
    assert select_module._spool_budget_bytes(2 * 10**6) == 2 * 10**6 * 4 * 1024
    assert select_module._compact_index_budget_bytes(0) == 512 * 1024**2
    assert select_module._compact_index_budget_bytes(10**7) == 10**7 * 512
    with pytest.raises(ValueError, match="row_count"):
        select_module._scaled_budget_bytes(-1, 1, 1)


def test_step3_spool_disk_space_failure_is_actionable(tmp_path, monkeypatch):
    config = _config(tmp_path / "out", models=("scoreq",))
    upstream = _publish_step2(config.output_root, [_row()])
    calls = []

    class ExhaustedDiskShutil:
        @staticmethod
        def disk_usage(_path):
            return SimpleNamespace(free=1024)

    monkeypatch.setattr(select_module, "shutil", ExhaustedDiskShutil)

    with pytest.raises(ValueError, match="exceeds free disk space"):
        Step3MosSelectionStage(
            scorer=_test_scorer(lambda *_args: calls.append(True)),
        ).run(_context(config, upstream))

    assert calls == []


def test_step3_v2_artifact_resume_fails_before_scoring_without_rewrite(tmp_path):
    config = _config(tmp_path / "out")
    upstream = _publish_step2(config.output_root, [_row()])

    class LegacyStep3V2(Step3MosSelectionStage):
        def config_payload(self, current_config):
            payload = dict(super().config_payload(current_config))
            payload["row_contract"] = {
                "version": 2,
                "fields": [*STEP2_ROW_FIELDS, "mos_scores"],
            }
            return payload

    legacy_stage = LegacyStep3V2(scorer=_test_scorer(_scores))
    writer = ArtifactWriter(
        config.output_root,
        PipelineStep.STEP3,
        "alpha",
        legacy_stage.config_digest(config),
        upstream.artifact_digest,
    )
    writer.append({**_row(), "mos_scores": _scores(_row(), ())})
    writer.finalize()
    artifact_dir = Path(config.output_root) / "step3" / "alpha"
    before = {}
    for name in ("rows.jsonl", "index.json", "_SUCCESS"):
        path = artifact_dir / name
        stat_result = path.stat()
        before[name] = (
            stat_result.st_ino, stat_result.st_mtime_ns, path.read_bytes(),
        )
    scorer_calls = []

    with pytest.raises(ValueError, match="expected config digest mismatch"):
        Step3MosSelectionStage(
            scorer=_test_scorer(lambda *_args: scorer_calls.append(True)),
        ).run(_context(config, upstream))

    assert scorer_calls == []
    after = {}
    for name in before:
        path = artifact_dir / name
        stat_result = path.stat()
        after[name] = (
            stat_result.st_ino, stat_result.st_mtime_ns, path.read_bytes(),
        )
    assert after == before
