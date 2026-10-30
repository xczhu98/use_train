"""Generation-3 Step6 metadata-only multirate export contract tests."""

import hashlib
from pathlib import Path

import pytest

from audio_filter_pipeline.config import PipelineConfig
from audio_filter_pipeline.processors.stages_v3.base import StageRunContext
from audio_filter_pipeline.tests.stage_test_helpers import (
    runner_expectation_for_test,
)
from audio_filter_pipeline.utils.artifacts import (
    ArtifactWriter,
    PipelineStep,
    iter_artifact_rows,
    open_artifact_for_read,
)


pytestmark = pytest.mark.unit


def _config(
    output_root,
    *,
    models=("scoreq", "urgent_pk", "dnsmos"),
    min_mos_models=None,
    **overrides,
):
    return PipelineConfig(
        pipeline_generation=3,
        step="step6",
        output_root=str(output_root),
        datasets={
            "alpha": {
                "input_root": "/input/alpha",
                "language": "en",
                "category": "read_speech",
                "quality_profile": "quality_high",
                "speaker": {"source": "none", "required": False},
            }
        },
        selection_policies={
            "global": {
                "mos_models": list(models),
                "min_mos_models": min_mos_models or len(models),
            },
            "quality_profiles": {
                name: {
                    "thresholds": {
                        "scoreq": 4.2,
                        "urgent_pk": 3.5,
                        "nisqa": 3.7,
                        "utmos": 3.9,
                        "dnsmos": {
                            "ovrl": 3.3,
                            "sig": 3.4,
                            "bak": 4.0,
                            "p808": 3.8,
                        },
                    }
                }
                for name in ("quality_balanced", "quality_high", "quality_premium")
            },
        },
        **overrides,
    )


def _step6_row(
    *, segment_id, sample_rate, bandwidth_hz, estimated_sr, passed=True,
    config=None, scores=None,
):
    config = config or _config("/tmp/audio-filter-step6-test")
    projection = config.mos_scoring_projection("alpha", "en")
    models = config.resolve_selection_policy("alpha", "en").mos_models
    default_scores = {
        "scoreq": 4.5 if passed else 2.0,
        "urgent_pk": 3.8 if passed else 2.0,
        "nisqa": 3.8 if passed else 2.0,
        "utmos": 4.0 if passed else 2.0,
        "dnsmos": (
            {"ovrl": 3.5, "sig": 3.6, "bak": 4.2, "p808": 4.0}
            if passed else
            {"ovrl": 2.0, "sig": 2.0, "bak": 2.0, "p808": 2.0}
        ),
    }
    return {
        "dataset_id": "alpha",
        "language": "en",
        "category": "read_speech",
        "quality_profile": "quality_high",
        "source_key": f"source-{segment_id}",
        "segment_id": segment_id,
        "input_root": "/input/alpha",
        "original_path": f"/input/alpha/{segment_id}.wav",
        "original_relative_path": "",
        "source_relative_path": f"{segment_id}.wav",
        "source_size_bytes": 1,
        "source_sha256": "a" * 64,
        "frames": sample_rate,
        "channels": 1,
        "duration_sec": 1.0,
        "format": "WAV",
        "subtype": "PCM_16",
        "sample_rate": sample_rate,
        "speaker_id": None,
        "speaker_scope": None,
        "source_start_sample": 0,
        "source_end_sample": sample_rate,
        "signal_finite": True,
        "signal_zero_ratio": 0.0,
        "signal_peak": 0.5,
        "signal_clipping_ratio": 0.0,
        "signal_dc_offset": 0.0,
        "signal_rms_dbfs": -12.0,
        "signal_num_samples": sample_rate,
        "bandwidth_hz": bandwidth_hz,
        "estimated_sr": estimated_sr,
        "vad_speech_ratio": 1.0,
        "vad_timeline_sample_rate": sample_rate,
        "vad_timeline_num_samples": sample_rate,
        "start_sample": 0,
        "end_sample": sample_rate,
        "boundary_reason": "source_end",
        "mos_models": list(models),
        "mos_scoring_policy_hash": projection["scoring_policy_hash"],
        "mos_scores": scores or {model: default_scores[model] for model in models},
        "speaker_shadow": {"status": "disabled"},
    }


def _publish_step5(tmp_path, rows):
    writer = ArtifactWriter(
        str(tmp_path),
        PipelineStep.STEP5,
        "alpha",
        hashlib.sha256(b"step4-config").hexdigest(),
        hashlib.sha256(b"step3-lineage").hexdigest(),
    )
    for row in rows:
        writer.append(row)
    writer.finalize()
    return open_artifact_for_read(
        str(tmp_path), PipelineStep.STEP5, "alpha",
        expected_config_digest=hashlib.sha256(b"step4-config").hexdigest(),
        expected_lineage_digest=hashlib.sha256(b"step3-lineage").hexdigest(),
    )


def _context(config, upstream):
    return StageRunContext(
        config.output_root,
        "alpha",
        config,
        upstream=runner_expectation_for_test(upstream),
    )


def _run_step6(stage, config, upstream):
    return stage.run(_context(config, upstream))


def test_source_identity_validation_uses_only_frozen_metadata(monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import export

    directory_row = _step6_row(
        segment_id="directory", sample_rate=16_000,
        bandwidth_hz=3_900.0, estimated_sr=8_000,
    )
    monkeypatch.setattr(
        export.os.path, "isfile",
        lambda _path: (_ for _ in ()).throw(AssertionError("live filesystem probe")),
    )
    export._validate_dataset_and_source_identity(
        directory_row, "alpha", None, label="directory row",
    )

    single_file_row = dict(directory_row)
    single_file_row.update({
        "input_root": "/input/source.wav",
        "original_path": "/input/source.wav",
        "source_relative_path": "input/source.wav",
        "original_relative_path": "",
    })
    export._validate_dataset_and_source_identity(
        single_file_row, "alpha", None, label="single-file row",
        require_original_relative_path=True,
    )

    invalid_row = dict(single_file_row, original_path="/input/other.wav")
    with pytest.raises(ValueError, match="not unambiguously bound"):
        export._validate_dataset_and_source_identity(
            invalid_row, "alpha", None, label="invalid row",
        )


def test_effective_bandwidth_routes_to_lowest_preserving_standard_rate():
    from audio_filter_pipeline.processors.stages_v3.export import (
        target_sample_rate_for_effective_bandwidth,
    )

    assert target_sample_rate_for_effective_bandwidth(3_900.0, 48_000) == 16_000
    assert target_sample_rate_for_effective_bandwidth(7_900.0, 48_000) == 16_000
    assert target_sample_rate_for_effective_bandwidth(11_900.0, 48_000) == 24_000
    assert target_sample_rate_for_effective_bandwidth(15_900.0, 48_000) == 32_000
    assert target_sample_rate_for_effective_bandwidth(21_000.0, 48_000) == 44_100
    assert target_sample_rate_for_effective_bandwidth(23_000.0, 48_000) == 48_000
    assert target_sample_rate_for_effective_bandwidth(5_000.0, 11_025) == 11_025


def test_target_sample_rate_keeps_8k_source_at_its_native_rate():
    from audio_filter_pipeline.processors.stages_v3.export import (
        target_sample_rate_for_effective_bandwidth,
    )

    assert target_sample_rate_for_effective_bandwidth(3_900.0, 8_000) == 8_000


def test_step6_preserves_rows_and_routes_each_segment_without_audio_io(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.export import (
        STEP6_ROW_FIELDS,
        Step6MultirateExportStage,
    )

    config = _config(tmp_path, step6_mode="dry_run")
    upstream = _publish_step5(
        tmp_path,
        [
            _step6_row(
                segment_id="narrowband", sample_rate=48_000,
                bandwidth_hz=3_900.0, estimated_sr=8_000,
            ),
            _step6_row(
                segment_id="wideband", sample_rate=48_000,
                bandwidth_hz=11_900.0, estimated_sr=24_000,
            ),
            _step6_row(
                segment_id="zero-bandwidth", sample_rate=48_000,
                bandwidth_hz=0.0, estimated_sr=None,
            ),
        ],
    )

    result = _run_step6(Step6MultirateExportStage(), config, upstream)
    rows = list(iter_artifact_rows(result))

    assert [row["segment_id"] for row in rows] == [
        "narrowband", "wideband", "zero-bandwidth",
    ]
    assert [row["target_sample_rate"] for row in rows] == [16_000, 24_000, 16_000]
    for row in rows:
        assert set(row) == set(STEP6_ROW_FIELDS)
        assert row["source_sample_rate"] == 48_000
        assert row["segment_sample_rate"] == 48_000
        assert row["export_routing"] == {
            "version": 3,
            "layout_version": 3,
            "layout_template": "{step}/{language}/{dataset_id}/{artifact_digest}/{manifest_or_ratehz_path}",
            "manifest_schema": "audio_filter_pipeline.export_manifest.v2",
            "manifest_filename": "_export_manifest.v2.jsonl",
            "transaction_schema": "audio_filter_pipeline.audio_export_transaction.v2",
            "mode": "metadata_only",
            "audio_write": False,
            "minimum_target_sample_rate_hz": 16_000,
            "source_below_minimum_rule": "native",
            "upsampling_allowed": False,
            "source_path": row["original_path"],
            "segment_interval_samples": [0, 48_000],
        }
        assert "mos_selection" not in row
        assert row["speaker_shadow"] == {"status": "disabled"}
        assert row["final_selection"] == {
            "version": 3,
            "export_eligible": True,
            "reason": "mos_selected",
        }

    payload = Step6MultirateExportStage().config_payload(config)
    assert payload["final_selection"] == {
        "version": 3,
        "mode": "dry_run",
        "min_cosine_similarity": None,
        "speaker_shadow_algorithm": "l2_normalized_segment_embedding_to_frozen_enrollment_cosine_v1",
        "enrollment_bundle_sha256": None,
    }
    assert "selection_policies" in payload
    assert payload["export"] == {
        "routing_version": 3,
        "manifest_schema": "audio_filter_pipeline.export_manifest.v2",
        "manifest_filename": "_export_manifest.v2.jsonl",
        "transaction_schema": "audio_filter_pipeline.audio_export_transaction.v2",
        "layout_version": 3,
        "layout_template": "{step}/{language}/{dataset_id}/{artifact_digest}/{manifest_or_ratehz_path}",
        "mode": "metadata_only",
        "audio_read": False,
        "audio_write": False,
        "minimum_target_sample_rate_hz": 16_000,
        "source_below_minimum_rule": "native",
        "upsampling_allowed": False,
    }
    assert payload["artifact_bound_export"]["producer_identity_fields"] == [
        "step", "dataset_id", "config_digest", "lineage_digest",
        "rows_sha256", "artifact_digest",
    ]
    assert payload["effective_bandwidth_rule"] == {
        "version": 3,
        "formula": "0 if bandwidth_hz == 0 else min(bandwidth_hz, estimated_sr / 2, source_sample_rate / 2)",
        "zero_bandwidth_estimated_sr": None,
        "target_rule": "source_below_16000_native_else_lowest_standard_rate_at_least_max_16000_and_twice_effective_bandwidth_without_upsampling",
        "minimum_target_sample_rate_hz": 16_000,
        "source_below_minimum_rule": "native",
        "upsampling_allowed": False,
        "standard_sample_rates_hz": [8000, 16000, 24000, 32000, 44100, 48000],
    }


def test_step6_dry_run_applies_candidate_threshold_without_audio_export(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.export import Step6MultirateExportStage

    policy = {
        "enabled": True, "runtime": "wespeaker",
        "model_dir": "/opt/wespeaker", "model_version": "model-v1",
        "model_sha256": "a" * 64, "runtime_version": "1.0",
        "target_sample_rate": 16_000, "enrollment_bundle_path": "/bundles/alpha.jsonl",
        "enrollment_bundle_sha256": "c" * 64, "enrollment_exclusion_path": "/bundles/alpha-exclusions.jsonl", "enrollment_exclusion_sha256": "d" * 64, "min_cosine_similarity": 0.8,
    }
    config = PipelineConfig(
        **{**_config(tmp_path).to_dict(), "step6_mode": "dry_run", "speaker_consistency": policy},
    )
    low = _step6_row(segment_id="low", sample_rate=16_000, bandwidth_hz=3_900.0, estimated_sr=8_000)
    low["speaker_shadow"] = {"status": "computed", "cosine_similarity_to_enrollment": 0.7, "provenance": {"algorithm": "l2_normalized_segment_embedding_to_frozen_enrollment_cosine_v1", "enrollment_bundle_sha256": "c" * 64}}
    upstream = _publish_step5(tmp_path, [low])
    result = _run_step6(Step6MultirateExportStage(), config, upstream)

    assert next(iter_artifact_rows(result))["final_selection"] == {
        "version": 3, "export_eligible": False, "reason": "speaker_inconsistent",
    }


def test_step6_rejects_legacy_or_mismatched_computed_enrollment_contract(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.export import Step6MultirateExportStage

    policy = {
        "enabled": True, "runtime": "wespeaker", "model_dir": "/opt/wespeaker",
        "model_version": "model-v1", "model_sha256": "a" * 64, "runtime_version": "1.0",
        "target_sample_rate": 16_000, "enrollment_bundle_path": "/bundles/alpha.jsonl",
        "enrollment_bundle_sha256": "c" * 64, "enrollment_exclusion_path": "/bundles/alpha-exclusions.jsonl", "enrollment_exclusion_sha256": "d" * 64, "min_cosine_similarity": 0.8,
    }
    config = PipelineConfig(**{**_config(tmp_path).to_dict(), "speaker_consistency": policy, "step6_mode": "filter"})
    row = _step6_row(segment_id="legacy", sample_rate=16_000, bandwidth_hz=3_900.0, estimated_sr=8_000)
    row["speaker_shadow"] = {"status": "computed", "cosine_similarity_to_centroid": 0.9}
    upstream = _publish_step5(tmp_path, [row])

    with pytest.raises(ValueError, match="cosine_similarity_to_enrollment"):
        _run_step6(Step6MultirateExportStage(), config, upstream)


def test_step6_filter_mode_rejects_only_computed_similarity_below_threshold(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.export import Step6MultirateExportStage

    speaker_consistency = {
        "enabled": True,
        "runtime": "wespeaker",
        "model_dir": "/opt/models/wespeaker/english",
        "model_version": "model-v1",
        "model_sha256": "a" * 64,
        "runtime_version": "1.2.3",
        "target_sample_rate": 16_000,
        "enrollment_bundle_path": "/bundles/alpha.jsonl",
        "enrollment_bundle_sha256": "c" * 64,
        "enrollment_exclusion_path": "/bundles/alpha-exclusions.jsonl",
        "enrollment_exclusion_sha256": "d" * 64,
        "min_cosine_similarity": 0.8,
    }
    config = PipelineConfig(**{
        **_config(tmp_path).to_dict(),
        "speaker_consistency": speaker_consistency,
        "step6_mode": "filter",
    })
    rows = [
        _step6_row(segment_id="below", sample_rate=16_000, bandwidth_hz=3_900.0, estimated_sr=8_000),
        _step6_row(segment_id="boundary", sample_rate=16_000, bandwidth_hz=3_900.0, estimated_sr=8_000),
        _step6_row(segment_id="insufficient", sample_rate=16_000, bandwidth_hz=3_900.0, estimated_sr=8_000),
        _step6_row(segment_id="disabled", sample_rate=16_000, bandwidth_hz=3_900.0, estimated_sr=8_000),
        _step6_row(segment_id="unavailable", sample_rate=16_000, bandwidth_hz=3_900.0, estimated_sr=8_000),
        _step6_row(segment_id="unsupported", sample_rate=16_000, bandwidth_hz=3_900.0, estimated_sr=8_000),
        _step6_row(segment_id="not-trial", sample_rate=16_000, bandwidth_hz=3_900.0, estimated_sr=8_000),
        _step6_row(segment_id="unknown-status", sample_rate=16_000, bandwidth_hz=3_900.0, estimated_sr=8_000),
        _step6_row(segment_id="rejected", sample_rate=16_000, bandwidth_hz=3_900.0, estimated_sr=8_000),
    ]
    computed_provenance = {
        "algorithm": "l2_normalized_segment_embedding_to_frozen_enrollment_cosine_v1",
        "enrollment_bundle_sha256": "c" * 64,
    }
    rows[0]["speaker_shadow"] = {"status": "computed", "cosine_similarity_to_enrollment": 0.79, "provenance": computed_provenance}
    rows[1]["speaker_shadow"] = {"status": "computed", "cosine_similarity_to_enrollment": 0.8, "provenance": computed_provenance}
    rows[2]["speaker_shadow"] = {"status": "insufficient_group"}
    rows[3]["speaker_shadow"] = {"status": "disabled"}
    rows[4]["speaker_shadow"] = {"status": "unavailable"}
    rows[5]["speaker_shadow"] = {"status": "unsupported"}
    rows[6]["speaker_shadow"] = {"status": "not_trial"}
    rows[7]["speaker_shadow"] = {"status": "future_unrecognized_status"}
    rows[8] = _step6_row(segment_id="rejected", sample_rate=16_000, bandwidth_hz=3_900.0, estimated_sr=8_000, passed=False)

    upstream = _publish_step5(tmp_path, rows)
    result = _run_step6(Step6MultirateExportStage(), config, upstream)
    final = {row["segment_id"]: row["final_selection"] for row in iter_artifact_rows(result)}

    assert final == {
        "below": {"version": 3, "export_eligible": False, "reason": "speaker_inconsistent"},
        "boundary": {"version": 3, "export_eligible": True, "reason": "speaker_consistent"},
        "insufficient": {"version": 3, "export_eligible": False, "reason": "speaker_shadow_not_exportable"},
        "disabled": {"version": 3, "export_eligible": False, "reason": "speaker_shadow_not_exportable"},
        "unavailable": {"version": 3, "export_eligible": False, "reason": "speaker_shadow_not_exportable"},
        "unsupported": {"version": 3, "export_eligible": False, "reason": "speaker_shadow_not_exportable"},
        "not-trial": {"version": 3, "export_eligible": False, "reason": "speaker_shadow_not_exportable"},
        "unknown-status": {"version": 3, "export_eligible": False, "reason": "speaker_shadow_not_exportable"},
        "rejected": {"version": 3, "export_eligible": False, "reason": "mos_rejected"},
    }
    assert Step6MultirateExportStage().config_payload(config)["final_selection"] == {
        "version": 3,
        "mode": "filter",
        "min_cosine_similarity": 0.8,
        "speaker_shadow_algorithm": "l2_normalized_segment_embedding_to_frozen_enrollment_cosine_v1",
        "enrollment_bundle_sha256": "c" * 64,
    }


def test_step6_rejects_duplicate_segment_ids_before_emitting_rows(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.export import Step6MultirateExportStage

    config = _config(tmp_path)
    row = _step6_row(
        segment_id="dup", sample_rate=16_000, bandwidth_hz=3_900.0, estimated_sr=8_000,
    )
    upstream = _publish_step5(tmp_path, [row, dict(row)])

    with pytest.raises(ValueError, match="duplicate segment_id in Step6 artifact"):
        _run_step6(Step6MultirateExportStage(), config, upstream)


def test_step6_filter_mode_fails_closed_when_every_shadow_is_unavailable(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.export import Step6MultirateExportStage

    policy = {
        "enabled": True, "runtime": "wespeaker", "model_dir": "/opt/wespeaker",
        "model_version": "model-v1", "model_sha256": "a" * 64, "runtime_version": "1.0",
        "target_sample_rate": 16_000, "enrollment_bundle_path": "/bundles/alpha.jsonl",
        "enrollment_bundle_sha256": "c" * 64,
        "enrollment_exclusion_path": "/bundles/alpha-exclusions.jsonl",
        "enrollment_exclusion_sha256": "d" * 64, "min_cosine_similarity": 0.8,
    }
    config = PipelineConfig(**{
        **_config(tmp_path).to_dict(),
        "speaker_consistency": policy,
        "step6_mode": "filter",
    })
    rows = [
        _step6_row(segment_id=segment_id, sample_rate=16_000, bandwidth_hz=3_900.0, estimated_sr=8_000)
        for segment_id in ("one", "two")
    ]
    for row in rows:
        row["speaker_shadow"] = {"status": "unavailable"}
    upstream = _publish_step5(tmp_path, rows)

    with pytest.raises(ValueError, match="speaker shadow is unavailable for every"):
        _run_step6(Step6MultirateExportStage(), config, upstream)
    assert not (tmp_path / "step6" / "alpha").exists()


def test_step6_final_selection_applies_only_computed_scores_below_threshold(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.export import Step6MultirateExportStage

    config = _config(tmp_path)
    config = PipelineConfig(
        **{**config.to_dict(), "step6_mode": "filter",
           "speaker_consistency": {
               "enabled": True, "runtime": "wespeaker", "model_dir": "/opt/wespeaker",
               "model_version": "model-v1", "model_sha256": "b" * 64, "runtime_version": "1.0", "target_sample_rate": 16000,
               "enrollment_bundle_path": "/bundles/alpha.jsonl", "enrollment_bundle_sha256": "c" * 64,
               "enrollment_exclusion_path": "/bundles/alpha-exclusions.jsonl", "enrollment_exclusion_sha256": "d" * 64,
               "min_cosine_similarity": 0.8,
           }},
    )
    low = _step6_row(segment_id="low", sample_rate=16_000, bandwidth_hz=3_900.0, estimated_sr=8_000)
    low["speaker_shadow"] = {"status": "computed", "cosine_similarity_to_enrollment": 0.7, "provenance": {"algorithm": "l2_normalized_segment_embedding_to_frozen_enrollment_cosine_v1", "enrollment_bundle_sha256": "c" * 64}}
    unavailable = _step6_row(segment_id="unknown", sample_rate=16_000, bandwidth_hz=3_900.0, estimated_sr=8_000)
    unavailable["speaker_shadow"] = {"status": "unavailable"}
    rejected = _step6_row(segment_id="rejected", sample_rate=16_000, bandwidth_hz=3_900.0, estimated_sr=8_000, passed=False)
    upstream = _publish_step5(tmp_path, [low, unavailable, rejected])
    result = _run_step6(Step6MultirateExportStage(), config, upstream)

    assert [(row["final_selection"]["export_eligible"], row["final_selection"]["reason"])
            for row in iter_artifact_rows(result)] == [
        (False, "speaker_inconsistent"), (False, "speaker_shadow_not_exportable"), (False, "mos_rejected"),
    ]


def test_step6_applies_single_model_one_of_one_quorum(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.export import Step6MultirateExportStage

    config = _config(
        tmp_path, models=("scoreq",), min_mos_models=1, step6_mode="dry_run",
    )
    accepted = _step6_row(
        segment_id="accepted", sample_rate=16_000, bandwidth_hz=3_900.0,
        estimated_sr=8_000, config=config, scores={"scoreq": 4.2},
    )
    rejected = _step6_row(
        segment_id="rejected", sample_rate=16_000, bandwidth_hz=3_900.0,
        estimated_sr=8_000, config=config, scores={"scoreq": 4.19},
    )

    upstream = _publish_step5(tmp_path, [accepted, rejected])
    result = _run_step6(Step6MultirateExportStage(), config, upstream)

    assert [row["final_selection"] for row in iter_artifact_rows(result)] == [
        {"version": 3, "export_eligible": True, "reason": "mos_selected"},
        {"version": 3, "export_eligible": False, "reason": "mos_rejected"},
    ]


def test_step6_applies_two_model_one_of_two_quorum(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.export import Step6MultirateExportStage

    config = _config(
        tmp_path, models=("utmos", "nisqa"), min_mos_models=1,
        step6_mode="dry_run",
    )
    one_passes = _step6_row(
        segment_id="one-passes", sample_rate=16_000, bandwidth_hz=3_900.0,
        estimated_sr=8_000, config=config,
        scores={"utmos": 3.9, "nisqa": 3.69},
    )
    none_pass = _step6_row(
        segment_id="none-pass", sample_rate=16_000, bandwidth_hz=3_900.0,
        estimated_sr=8_000, config=config,
        scores={"utmos": 3.89, "nisqa": 3.69},
    )

    upstream = _publish_step5(tmp_path, [one_passes, none_pass])
    result = _run_step6(Step6MultirateExportStage(), config, upstream)

    assert [row["final_selection"]["export_eligible"] for row in iter_artifact_rows(result)] == [
        True, False,
    ]


def test_step6_unbound_stage_does_not_retain_context_dataset(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.export import Step6MultirateExportStage

    raw = _config(tmp_path, step6_mode="dry_run").to_dict()
    raw["datasets"]["beta"] = {
        "input_root": "/input/beta",
        "language": "en",
        "category": "read_speech",
        "quality_profile": "quality_high",
        "speaker": {"source": "none", "required": False},
    }
    config = PipelineConfig(**raw)
    stage = Step6MultirateExportStage()

    alpha_writer = ArtifactWriter(
        config.output_root, PipelineStep.STEP5, "alpha", "1" * 64, "2" * 64,
    )
    alpha_writer.append({})
    alpha_upstream = alpha_writer.finalize()
    beta_writer = ArtifactWriter(
        config.output_root, PipelineStep.STEP5, "beta", "3" * 64, "4" * 64,
    )
    beta_writer.append({})
    beta_upstream = beta_writer.finalize()
    stage._validate_contract(StageRunContext(
        config.output_root,
        "alpha",
        config,
        upstream=runner_expectation_for_test(alpha_upstream),
    ))
    stage._validate_contract(StageRunContext(
        config.output_root,
        "beta",
        config,
        upstream=runner_expectation_for_test(beta_upstream),
    ))

    assert stage._dataset_id is None
    assert "selection_policies" in stage.config_payload(config)


def test_step6_dataset_local_binding_ignores_other_dataset_policy_changes(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.export import Step6MultirateExportStage

    base = _config(tmp_path, step6_mode="dry_run").to_dict()
    base["datasets"]["beta"] = {
        "input_root": "/input/beta",
        "language": "en",
        "category": "read_speech",
        "quality_profile": "quality_high",
        "speaker": {"source": "none", "required": False},
    }
    base["selection_policies"]["datasets"] = {
        "beta": {
            "mos_models": ["utmos"],
            "min_mos_models": 1,
            "thresholds": {"utmos": 3.9},
        },
    }
    beta_changed = {
        **base,
        "datasets": {**base["datasets"]},
        "selection_policies": {
            **base["selection_policies"],
            "datasets": {
                "beta": {
                    "mos_models": ["nisqa"],
                    "min_mos_models": 1,
                    "thresholds": {"nisqa": 4.1},
                },
            },
        },
    }
    stage = Step6MultirateExportStage(dataset_id="alpha")
    base_config = PipelineConfig(**base)

    payload = stage.config_payload(base_config)

    assert payload["selection"]["dataset_binding"]["dataset_id"] == "alpha"
    assert stage.config_digest(base_config) == stage.config_digest(
        PipelineConfig(**beta_changed)
    )


def test_step6_alpha_threshold_change_invalidates_published_artifact(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.export import Step6MultirateExportStage

    config = _config(tmp_path, step6_mode="dry_run")
    changed = config.to_dict()
    changed["selection_policies"]["quality_profiles"]["quality_high"][
        "thresholds"
    ]["scoreq"] = 3.0
    changed_config = PipelineConfig(**changed)
    upstream = _publish_step5(tmp_path, [
        _step6_row(
            segment_id="threshold", sample_rate=16_000,
            bandwidth_hz=7_000.0, estimated_sr=16_000, config=config,
        ),
    ])
    stage = Step6MultirateExportStage(dataset_id="alpha")
    stage.run(_context(config, upstream))

    with pytest.raises(ValueError, match="expected config digest mismatch"):
        Step6MultirateExportStage(dataset_id="alpha").run(
            _context(changed_config, upstream)
        )


def test_step6_rejects_invalid_bandwidth_before_publishing(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.export import Step6MultirateExportStage

    config = _config(tmp_path)
    upstream = _publish_step5(
        tmp_path,
        [_step6_row(
            segment_id="invalid", sample_rate=16_000,
            bandwidth_hz=8_001.0, estimated_sr=16_000,
        )],
    )

    with pytest.raises(ValueError, match="bandwidth_hz"):
        _run_step6(Step6MultirateExportStage(), config, upstream)


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
                "mos_scores": {"scoreq": 4.5},
            }),
            id="model-route",
        ),
    ],
)
def test_step6_stale_scoring_projection_fails_before_export_routing_side_effect(
    tmp_path, monkeypatch, mutate,
):
    from audio_filter_pipeline.processors.stages_v3 import export

    config = _config(tmp_path, step6_mode="dry_run")
    row = _step6_row(
        segment_id="stale", sample_rate=16_000,
        bandwidth_hz=7_000.0, estimated_sr=16_000, config=config,
    )
    mutate(row)
    routing_calls = []
    monkeypatch.setattr(
        export,
        "_effective_bandwidth_hz",
        lambda _row: routing_calls.append(True) or 7_000.0,
    )
    upstream = _publish_step5(tmp_path, [row])

    with pytest.raises(ValueError) as error:
        _run_step6(export.Step6MultirateExportStage(), config, upstream)

    assert str(error.value) == "Step5 upstream row MOS scoring projection is stale"
    assert routing_calls == []
    assert not (Path(config.output_root) / "step6" / "alpha").exists()


def test_step6_v2_artifact_resume_fails_before_producing_without_rewrite(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.export import (
        Step6MultirateExportStage,
    )

    config = _config(tmp_path, step6_mode="dry_run")
    upstream = _publish_step5(tmp_path, [
        _step6_row(
            segment_id="legacy", sample_rate=16_000,
            bandwidth_hz=7_000.0, estimated_sr=16_000, config=config,
        ),
    ])

    class LegacyStep6V2(Step6MultirateExportStage):
        def config_payload(self, current_config):
            payload = dict(super().config_payload(current_config))
            payload["row_contract"] = {
                "version": 2, "preserves_upstream_fields": True,
                "appended_fields": ["target_sample_rate"],
            }
            return payload

    legacy = LegacyStep6V2()
    writer = ArtifactWriter(
        config.output_root, PipelineStep.STEP6, "alpha",
        legacy.config_digest(config), upstream.artifact_digest,
    )
    writer.append({
        **_step6_row(
            segment_id="legacy", sample_rate=16_000,
            bandwidth_hz=7_000.0, estimated_sr=16_000, config=config,
        ),
        "target_sample_rate": 16_000,
    })
    writer.finalize()
    artifact_dir = Path(config.output_root) / "step6" / "alpha"
    before = {
        name: (artifact_dir / name).read_bytes()
        for name in ("rows.jsonl", "index.json", "_SUCCESS")
    }
    producer_calls = []

    class CurrentStage(Step6MultirateExportStage):
        def produce_rows(self, *args, **kwargs):
            producer_calls.append(True)
            return super().produce_rows(*args, **kwargs)

    with pytest.raises(ValueError, match="expected config digest mismatch"):
        _run_step6(CurrentStage(), config, upstream)

    assert producer_calls == []
    assert {
        name: (artifact_dir / name).read_bytes() for name in before
    } == before


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_step6_rejects_step5_rows_outside_exact_contract(tmp_path, mutation):
    from audio_filter_pipeline.processors.stages_v3.export import (
        Step6MultirateExportStage,
    )

    config = _config(tmp_path, step6_mode="dry_run")
    row = _step6_row(
        segment_id="contract", sample_rate=16_000,
        bandwidth_hz=7_000.0, estimated_sr=16_000, config=config,
    )
    if mutation == "missing":
        row.pop("signal_peak")
    else:
        row["legacy_passthrough"] = True
    upstream = _publish_step5(tmp_path, [row])

    with pytest.raises(ValueError, match="fields do not match the exact contract"):
        _run_step6(Step6MultirateExportStage(), config, upstream)


def test_step6_declares_exact_step5_and_output_contract(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.export import (
        STEP5_ROW_FIELDS,
        STEP6_ROW_FIELDS,
        Step6MultirateExportStage,
    )

    payload = Step6MultirateExportStage().config_payload(_config(tmp_path))

    assert payload["upstream_row_contract"] == {
        "version": 5,
        "fields": list(STEP5_ROW_FIELDS),
    }
    assert payload["row_contract"] == {
        "version": 5,
        "fields": list(STEP6_ROW_FIELDS),
    }
