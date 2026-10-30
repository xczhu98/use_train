"""Generation-3 Step4 MOS filtering artifact contract tests."""

import hashlib
from pathlib import Path

import pytest

from audio_filter_pipeline.config import PipelineConfig
from audio_filter_pipeline.processors.stages_v3.base import (
    ArtifactExpectation,
    StageRunContext,
)
from audio_filter_pipeline.processors.stages_v3.mos_filter import (
    STEP4_ROW_FIELDS,
    Step4MosFilterStage,
)
from audio_filter_pipeline.processors.stages_v3.segment import STEP2_ROW_FIELDS
from audio_filter_pipeline.utils.artifacts import (
    ArtifactWriter,
    PipelineStep,
    iter_artifact_rows,
)
from stage_test_helpers import runner_expectation_for_test


pytestmark = pytest.mark.unit


def _config(
    output_root,
    *,
    models=("scoreq", "urgent_pk", "dnsmos"),
    min_mos_models=3,
):
    return PipelineConfig(
        pipeline_generation=3,
        step="step4",
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
                "mos_models": list(models), "min_mos_models": min_mos_models,
            },
            "quality_profiles": {
                name: {"thresholds": {
                    "scoreq": 4.2,
                    "urgent_pk": 3.5,
                    "dnsmos": {"ovrl": 3.3, "sig": 3.4, "bak": 4.0, "p808": 3.8},
                    "nisqa": 3.5,
                    "utmos": 3.5,
                }}
                for name in ("quality_balanced", "quality_high", "quality_premium")
            },
        },
    )


def _row():
    values = {
        "source_key": "source-1", "dataset_id": "alpha", "language": "en",
        "category": "read_speech", "quality_profile": "quality_high",
        "input_root": "/missing/input", "original_path": "/missing/input/source.wav",
        "original_relative_path": "source.wav", "source_relative_path": "source.wav",
        "source_size_bytes": 42, "source_sha256": hashlib.sha256(b"source-1").hexdigest(),
        "sample_rate": 16000, "frames": 160000, "channels": 1, "duration_sec": 10.0,
        "format": "WAV", "subtype": "PCM_16", "speaker_id": None, "speaker_scope": None,
        "source_start_sample": 0,
        "source_end_sample": 160000, "signal_finite": True, "signal_zero_ratio": 0.0,
        "signal_peak": 0.5, "signal_clipping_ratio": 0.0, "signal_dc_offset": 0.0,
        "signal_rms_dbfs": -20.0, "signal_num_samples": 160000, "bandwidth_hz": 7000.0,
        "estimated_sr": 16000, "vad_speech_ratio": 0.9, "vad_timeline_sample_rate": 16000,
        "vad_timeline_num_samples": 160000, "segment_id": "segment-1", "start_sample": 0,
        "end_sample": 160000, "boundary_reason": "source_end",
    }
    assert tuple(values) == STEP2_ROW_FIELDS
    return values


def _step3_row(config, *, scores=None):
    row = _row()
    projection = config.mos_scoring_projection("alpha", "en")
    defaults = {
        "scoreq": 4.3, "urgent_pk": 3.6,
        "dnsmos": {"ovrl": 3.4, "sig": 3.5, "bak": 4.1, "p808": 3.9},
        "nisqa": 3.8, "utmos": 4.0,
    }
    models = tuple(projection["mos_models"])
    return {
        **row,
        "mos_models": list(models),
        "mos_scoring_policy_hash": projection["scoring_policy_hash"],
        "mos_scores": scores or {model: defaults[model] for model in models},
    }


def _expectation(index):
    return runner_expectation_for_test(index)


def _run(tmp_path, row, *, config=None):
    config = config or _config(tmp_path / "out")
    writer = ArtifactWriter(config.output_root, PipelineStep.STEP3, "alpha", "1" * 64, "2" * 64)
    writer.append(row)
    upstream = writer.finalize()
    context = StageRunContext(config.output_root, "alpha", config, upstream=_expectation(upstream))
    index = Step4MosFilterStage().run(context)
    return list(iter_artifact_rows(index))[0]


def test_step4_rejects_duplicate_segment_ids_before_emitting_rows(tmp_path):
    config = _config(tmp_path / "out")
    row = _step3_row(config)
    writer = ArtifactWriter(config.output_root, PipelineStep.STEP3, "alpha", "1" * 64, "2" * 64)
    writer.append(row)
    writer.append(dict(row))
    upstream = writer.finalize()
    context = StageRunContext(
        config.output_root, "alpha", config, upstream=_expectation(upstream),
    )

    with pytest.raises(ValueError, match="duplicate segment_id in Step3 artifact"):
        Step4MosFilterStage().run(context)


def test_step4_decides_keep_from_step3_scores_without_model_runtime(tmp_path):
    config = _config(tmp_path / "out")
    upstream_row = _step3_row(config)
    row = _run(tmp_path, upstream_row)

    assert set(row) == set(STEP4_ROW_FIELDS)
    assert {field: row[field] for field in upstream_row} == upstream_row
    assert row["mos_selection"] == {
        "decision": "keep", "passed_models": 3, "required_models": 3,
        "policy_id": "alpha:en",
        "policy_hash": config.resolve_selection_policy("alpha", "en").policy_hash,
    }


def test_step4_supports_single_model_one_of_one_policy(tmp_path):
    config = _config(
        tmp_path / "single", models=("scoreq",), min_mos_models=1,
    )

    row = _run(tmp_path, _step3_row(config), config=config)

    assert row["mos_selection"]["decision"] == "keep"
    assert row["mos_selection"]["passed_models"] == 1
    assert row["mos_selection"]["required_models"] == 1


def test_step4_supports_two_model_one_of_two_policy(tmp_path):
    config = _config(
        tmp_path / "two", models=("utmos", "nisqa"), min_mos_models=1,
    )

    row = _run(
        tmp_path,
        _step3_row(config, scores={"utmos": 2.0, "nisqa": 3.8}),
        config=config,
    )

    assert row["mos_selection"]["decision"] == "keep"
    assert row["mos_selection"]["passed_models"] == 1
    assert row["mos_selection"]["required_models"] == 1


def test_step4_rejects_when_one_dnsmos_dimension_fails(tmp_path):
    config = _config(tmp_path / "out")
    row = _run(tmp_path, _step3_row(config, scores={
        "scoreq": 4.3, "urgent_pk": 3.6,
        "dnsmos": {"ovrl": 3.4, "sig": 3.5, "bak": 3.9, "p808": 3.9},
    }))

    assert row["mos_selection"]["decision"] == "reject"
    assert row["mos_selection"]["passed_models"] == 2


@pytest.mark.parametrize("mutate, error", [
    (lambda row: row.__setitem__("mos_scores", {"scoreq": 4.3}), "scores"),
    (lambda row: row.__setitem__("selection", {"decision": "keep"}), "exact contract"),
])
def test_step4_fails_closed_on_invalid_step3_scores_or_provenance(tmp_path, mutate, error):
    config = _config(tmp_path / "out")
    row = _step3_row(config)
    mutate(row)

    with pytest.raises(ValueError, match=error):
        _run(tmp_path, row)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda row: row.__setitem__("mos_scoring_policy_hash", "0" * 64),
            id="policy-hash",
        ),
        pytest.param(
            lambda row: row.update({
                "mos_models": ["scoreq"],
                "mos_scores": {"scoreq": 4.3},
            }),
            id="model-route",
        ),
    ],
)
def test_step4_stale_scoring_projection_fails_before_selection_side_effect(
    tmp_path, monkeypatch, mutate,
):
    config = _config(tmp_path / "out")
    row = _step3_row(config)
    mutate(row)
    selection_calls = []
    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.mos_filter.passed_models",
        lambda *_args: selection_calls.append(True) or 3,
    )

    with pytest.raises(ValueError) as error:
        _run(tmp_path, row, config=config)

    assert str(error.value) == (
        "Step3 row MOS scoring projection does not match current policy"
    )
    assert selection_calls == []
    assert not (Path(config.output_root) / "step4" / "alpha").exists()


def test_step4_requires_bound_step3_upstream(tmp_path):
    config = _config(tmp_path / "out")
    context = StageRunContext(config.output_root, "alpha", config)

    with pytest.raises(ValueError, match="requires an upstream expectation"):
        Step4MosFilterStage().run(context)


def _mutated_threshold_config(tmp_path):
    """与 _config 相同但 quality_high 的 scoreq 阈值从 4.2 放宽到 3.0。"""
    return PipelineConfig(
        pipeline_generation=3,
        step="step4",
        output_root=str(tmp_path),
        datasets={"alpha": {
            "input_root": "/missing/input",
            "language": "en",
            "category": "read_speech",
            "quality_profile": "quality_high",
            "speaker": {"source": "none", "required": False},
        }},
        selection_policies={
            "global": {"mos_models": ["scoreq", "urgent_pk", "dnsmos"], "min_mos_models": 3},
            "quality_profiles": {
                name: {"thresholds": {
                    "scoreq": 3.0 if name == "quality_high" else 4.2,
                    "urgent_pk": 3.5,
                    "dnsmos": {"ovrl": 3.3, "sig": 3.4, "bak": 4.0, "p808": 3.8},
                }}
                for name in ("quality_balanced", "quality_high", "quality_premium")
            },
        },
    )


def test_step4_unbound_stage_does_not_retain_context_dataset(tmp_path):
    raw = _config(tmp_path / "out").to_dict()
    raw["datasets"]["beta"] = {
        "input_root": "/missing/beta",
        "language": "en",
        "category": "read_speech",
        "quality_profile": "quality_high",
        "speaker": {"source": "none", "required": False},
    }
    config = PipelineConfig(**raw)
    stage = Step4MosFilterStage()

    alpha_writer = ArtifactWriter(
        config.output_root, PipelineStep.STEP3, "alpha", "1" * 64, "2" * 64,
    )
    alpha_writer.append({})
    alpha_upstream = alpha_writer.finalize()
    beta_writer = ArtifactWriter(
        config.output_root, PipelineStep.STEP3, "beta", "3" * 64, "4" * 64,
    )
    beta_writer.append({})
    beta_upstream = beta_writer.finalize()
    stage._validate_contract(StageRunContext(
        config.output_root, "alpha", config, upstream=_expectation(alpha_upstream),
    ))
    stage._validate_contract(StageRunContext(
        config.output_root, "beta", config, upstream=_expectation(beta_upstream),
    ))

    assert stage._dataset_id is None
    assert "selection_policies" in stage.config_payload(config)["selection"]


def test_step4_dataset_local_binding_ignores_other_dataset_policy_changes(tmp_path):
    base = _config(tmp_path / "out").to_dict()
    for config in (base,):
        config["datasets"]["beta"] = {
            "input_root": "/missing/beta",
            "language": "en",
            "category": "read_speech",
            "quality_profile": "quality_high",
            "speaker": {"source": "none", "required": False},
        }
        config["selection_policies"]["datasets"] = {
            "beta": {
                "mos_models": ["utmos"],
                "min_mos_models": 1,
                "thresholds": {"utmos": 3.5},
            },
        }
    beta_threshold = {
        **base,
        "datasets": {**base["datasets"]},
        "selection_policies": {
            **base["selection_policies"],
            "datasets": {
                "beta": {
                    "mos_models": ["utmos"],
                    "min_mos_models": 1,
                    "thresholds": {"utmos": 4.2},
                },
            },
        },
    }
    beta_models = {
        **base,
        "datasets": {**base["datasets"]},
        "selection_policies": {
            **base["selection_policies"],
            "datasets": {
                "beta": {
                    "mos_models": ["nisqa"],
                    "min_mos_models": 1,
                    "thresholds": {"nisqa": 3.5},
                },
            },
        },
    }
    stage = Step4MosFilterStage(dataset_id="alpha")

    base_config = PipelineConfig(**base)

    assert stage.config_payload(base_config)["selection"]["dataset_binding"]["dataset_id"] == "alpha"
    assert stage.config_digest(base_config) == stage.config_digest(PipelineConfig(**beta_threshold))
    assert stage.config_digest(base_config) == stage.config_digest(PipelineConfig(**beta_models))


def test_step4_config_digest_changes_with_thresholds(tmp_path):
    """解耦后阈值 digest 归属 Step4：改阈值必须改变 Step4 config digest。"""
    base = _config(tmp_path / "out")
    mutated = _mutated_threshold_config(str(tmp_path / "out"))
    stage = Step4MosFilterStage(dataset_id="alpha")

    assert stage.config_digest(base) != stage.config_digest(mutated)


def test_step4_published_artifact_fails_closed_after_threshold_change(tmp_path):
    """改阈值后已发布 Step4 artifact 恢复必须 fail-closed（提示重建）。"""
    config = _config(tmp_path / "out")
    writer = ArtifactWriter(
        config.output_root, PipelineStep.STEP3, "alpha", "1" * 64, "2" * 64,
    )
    writer.append(_step3_row(config))
    upstream = writer.finalize()
    published = Step4MosFilterStage(dataset_id="alpha").run(
        StageRunContext(config.output_root, "alpha", config, upstream=_expectation(upstream))
    )
    assert published.row_count == 1

    mutated = _mutated_threshold_config(str(tmp_path / "out"))
    context = StageRunContext(
        mutated.output_root, "alpha", mutated, upstream=_expectation(upstream),
    )
    # Step3 artifact 复用成功（解耦），Step4 自身 digest 不匹配 → 拒绝恢复
    with pytest.raises(ValueError, match="expected config digest mismatch"):
        Step4MosFilterStage(dataset_id="alpha").run(context)


def test_step4_v2_artifact_resume_fails_before_producing_without_rewrite(tmp_path):
    config = _config(tmp_path / "out")
    upstream_writer = ArtifactWriter(
        config.output_root, PipelineStep.STEP3, "alpha", "1" * 64, "2" * 64,
    )
    upstream_writer.append(_step3_row(config))
    upstream = upstream_writer.finalize()

    class LegacyStep4V2(Step4MosFilterStage):
        def config_payload(self, current_config):
            payload = dict(super().config_payload(current_config))
            payload["row_contract"] = {
                "version": 2, "fields": list(STEP4_ROW_FIELDS),
            }
            return payload

    legacy = LegacyStep4V2()
    writer = ArtifactWriter(
        config.output_root, PipelineStep.STEP4, "alpha",
        legacy.config_digest(config), upstream.artifact_digest,
    )
    writer.append({**_step3_row(config), "mos_selection": {"decision": "keep"}})
    writer.finalize()
    artifact_dir = Path(config.output_root) / "step4" / "alpha"
    before = {
        name: (artifact_dir / name).read_bytes()
        for name in ("rows.jsonl", "index.json", "_SUCCESS")
    }
    producer_calls = []

    class CurrentStage(Step4MosFilterStage):
        def produce_rows(self, *args, **kwargs):
            producer_calls.append(True)
            return super().produce_rows(*args, **kwargs)

    with pytest.raises(ValueError, match="expected config digest mismatch"):
        CurrentStage().run(StageRunContext(
            config.output_root, "alpha", config, upstream=_expectation(upstream),
        ))

    assert producer_calls == []
    assert {
        name: (artifact_dir / name).read_bytes() for name in before
    } == before


def test_step4_declares_v4_output_and_exact_step3_upstream_contract(tmp_path):
    payload = Step4MosFilterStage().config_payload(_config(tmp_path / "out"))

    assert payload["row_contract"] == {
        "version": 4,
        "fields": list(STEP4_ROW_FIELDS),
    }
    assert payload["upstream_row_contract"] == {
        "version": 4,
        "fields": list(STEP4_ROW_FIELDS[:-1]),
    }
