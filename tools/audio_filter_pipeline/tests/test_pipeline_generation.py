"""Generation 3 管线配置边界测试。"""

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from audio_filter_pipeline.config import (
    PipelineConfig,
    ValidationContext,
    validation_context_for_step,
)
from audio_filter_pipeline.processors.optional_runtime import local_model_tree_sha256

pytestmark = pytest.mark.unit


def _quality_profiles():
    return {
        "quality_balanced": {"thresholds": {
            "dnsmos": {"ovrl": 3.2, "sig": 3.0, "bak": 3.8, "p808": 3.6},
            "scoreq": 3.6, "urgent_pk": 3.5,
        }},
        "quality_high": {"thresholds": {
            "dnsmos": {"ovrl": 3.3, "sig": 3.4, "bak": 4.0, "p808": 3.8},
            "scoreq": 4.2, "urgent_pk": 3.5,
        }},
        "quality_premium": {"thresholds": {
            "dnsmos": {"ovrl": 3.4, "sig": 3.6, "bak": 4.2, "p808": 4.0},
            "scoreq": 4.5, "urgent_pk": 3.5,
        }},
    }


def _config(**kwargs):
    values = {"pipeline_generation": 3, "output_root": "/tmp/test-output"}
    values.update(kwargs)
    if values.get("datasets") is not None and "selection_policies" not in values:
        values["selection_policies"] = {"quality_profiles": _quality_profiles()}
    return PipelineConfig(**values)


def _validate_structural(config):
    config.validate(
        require_scores_file=False,
        context=ValidationContext.STRUCTURAL,
        stage_names=[],
    )


@pytest.mark.parametrize("step", ["step0", "step1", "step2", "step3", "step4", "step5", "step6"])
def test_generation3_accepts_all_v3_step_names(step):
    cfg = _config(step=step)
    _validate_structural(cfg)
    assert cfg.pipeline_generation == 3


def test_generation3_rejects_filter_only():
    with pytest.raises(ValueError, match="filter_only.*generation 3"):
        _validate_structural(_config(step="filter_only"))


@pytest.mark.parametrize("generation", [0, 1, 2, 4, "3", None])
def test_invalid_pipeline_generation_fails_clearly(generation):
    with pytest.raises(ValueError, match="pipeline_generation"):
        PipelineConfig(pipeline_generation=generation)


def test_pipeline_generation_is_serialized_and_round_trips():
    serialized = PipelineConfig(pipeline_generation=3, step="step4").to_dict()
    assert serialized["pipeline_generation"] == 3
    assert PipelineConfig(**serialized).pipeline_generation == 3


@pytest.mark.parametrize(
    ("legacy_argument", "explicit_value"),
    [
        ("mos_types", None),
        ("mos_threshold", 3.0),
        ("mos_thresholds", None),
        ("enable_multi_mos", True),
        ("min_mos_models", 2),
        ("mos_fusion", "average"),
        ("step2_direct_from_input", False),
        ("step2_mos_early_exit_threshold", None),
    ],
)
def test_generation3_rejects_each_explicit_legacy_selection_argument(
    legacy_argument, explicit_value,
):
    with pytest.raises(
        ValueError,
        match=rf"generation 3.*{legacy_argument}.*selection_policies",
    ):
        PipelineConfig(
            pipeline_generation=3,
            **{legacy_argument: explicit_value},
        )


def test_generation3_omitted_legacy_selection_arguments_keep_compatibility_defaults():
    config = PipelineConfig(pipeline_generation=3)

    assert config.mos_types == ["scoreq", "urgent_pk", "dnsmos"]
    assert config.mos_threshold == 3.0
    assert config.mos_thresholds == {}
    assert config.enable_multi_mos is True
    assert config.min_mos_models == 3
    assert config.mos_fusion == "average"
    assert config.step2_direct_from_input is False
    assert config.step2_mos_early_exit_threshold is None


def test_shipped_yaml_route_is_distinct_from_bare_legacy_constructor_default():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "audio_filter_zh_16k.yaml"
    )
    shipped_route = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )["selection_policies"]["global"]
    legacy = PipelineConfig(pipeline_generation=3)

    assert shipped_route["mos_models"] == ["scoreq", "dnsmos"]
    assert shipped_route["min_mos_models"] == 2
    assert shipped_route["mos_models"] != legacy.mos_types
    assert shipped_route["min_mos_models"] != legacy.min_mos_models


def test_validation_context_mapping_rejects_invalid_generation():
    with pytest.raises(ValueError, match="pipeline_generation"):
        validation_context_for_step("step1", pipeline_generation=4)


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        ("step0", "V3_STEP0_EXECUTION"),
        ("step1", "V3_STEP1_EXECUTION"),
        ("step2", "V3_STEP2_EXECUTION"),
        ("step3", "V3_STEP3_EXECUTION"),
        ("step4", "V3_STEP4_EXECUTION"),
        ("step5", "V3_STEP5_EXECUTION"),
        ("step6", "V3_STEP6_EXECUTION"),
    ],
)
def test_validation_context_mapping_distinguishes_generation3_steps(step, expected):
    context = validation_context_for_step(step, pipeline_generation=3)
    assert context is getattr(ValidationContext, expected)


@pytest.mark.parametrize("step", ["unknown", "step-1"])
def test_generation3_rejects_unknown_steps(step):
    with pytest.raises(ValueError, match="generation 3"):
        _validate_structural(_config(step=step))


def test_validation_context_enum_values_are_unique():
    members = list(ValidationContext.__members__.values())
    assert len({member.value for member in members}) == len(members)


@pytest.mark.parametrize("generation", [0, 1, 2, 4, "3", True, None])
def test_validate_rejects_mutated_invalid_generation(generation):
    cfg = _config()
    cfg.pipeline_generation = generation
    with pytest.raises(ValueError, match="pipeline_generation"):
        _validate_structural(cfg)


@pytest.mark.parametrize(
    "context",
    [
        ValidationContext.STEP1_EXECUTION,
        ValidationContext.STEP2_EXECUTION,
        ValidationContext.STEP3_EXECUTION,
    ],
)
def test_validate_rejects_non_v3_contexts(context):
    with pytest.raises(ValueError, match="context.*generation"):
        _config().validate(
            require_scores_file=False,
            context=context,
            stage_names=[],
        )


@pytest.mark.parametrize("step", ["step0", "step1", "step2", "step3", "step4", "step5", "step6"])
def test_default_validate_uses_generation_execution_context(step):
    cfg = _config(step=step, datasets=None)
    with pytest.raises(ValueError, match="datasets.*non-empty mapping"):
        cfg.validate(require_scores_file=False, stage_names=[])


def _minimal_datasets(tmp_path, *, input_root=None):
    return {
        "test": {
            "input_root": str(tmp_path / "source" if input_root is None else input_root),
            "language": "unknown",
            "category": "read_speech",
            "quality_profile": "quality_high",
            "speaker": {"source": "none", "required": False},
        },
    }


def test_v3_step3_default_validation_does_not_require_scores(tmp_path):
    cfg = _config(step="step3", datasets=_minimal_datasets(tmp_path))
    cfg.validate(stage_names=[])

    # 同一 config 在 chain 含 mos 时必须执行 MOS 资产 preflight 并因缺资产报错，
    # 证明上面的通过来自"跳过 preflight"而不是校验整体空转。
    with pytest.raises(ValueError, match="urgent_pk"):
        cfg.validate(stage_names=["mos"])


def test_v3_to_dict_is_exact_milestone_a_view(tmp_path):
    cfg = PipelineConfig(
        pipeline_generation=3,
        step="step4",
        output_root=str(tmp_path / "output"),
        datasets=_minimal_datasets(tmp_path),
        selection_policies={
            "quality_profiles": _quality_profiles(),
            "global": {"models": ("DNSMOS",)},
        },
    )

    assert set(cfg.to_dict()) == {
        "pipeline_generation",
        "step",
        "output_root",
        "output_sample_rate_hz",
        "datasets",
        "selection_policies",
        "speaker_consistency",
        "step3_gpu_ids",
        "step3_worker_idle_timeout_seconds",
        "step3_worker_hard_timeout_seconds",
        "step3_worker_stderr_tail_bytes",
        "step6_mode",
        "step4_mode",
        "step4_export_root",
        "step6_export_root",
        "audio_export_workers",
        "audio_export_max_pending",
        "audio_export_max_inflight_bytes",
        "step1_vad_workers",
        "step1_vad_max_pending",
        "step2_max_segment_duration_sec",
        "step2_min_segment_duration_sec",
        "urgent_pk_checkpoint",
        "urgent_pk_source_dir",
        "urgent_pk_encoder",
        "urgent_pk_backbone",
        "urgent_pk_score_range",
        "urgent_pk_expected_sha256",
        "urgent_pk_expected_commit",
        "scoreq_model_dir",
        "dnsmos_model_dir",
        "nisqa_model_path",
        "nisqa_model_version",
        "nisqa_model_sha256",
        "utmos_model_path",
        "utmos_model_version",
        "utmos_model_sha256",
    }


def test_v3_step3_worker_timeouts_are_canonical_and_round_trip():
    config = PipelineConfig(
        pipeline_generation=3,
        step3_worker_idle_timeout_seconds=7_200,
        step3_worker_hard_timeout_seconds=None,
    )

    payload = config.to_dict()
    assert payload["step3_worker_idle_timeout_seconds"] == 7_200.0
    assert payload["step3_worker_hard_timeout_seconds"] is None
    assert "step3_worker_timeout_seconds" not in payload
    assert PipelineConfig(**payload).to_dict() == payload


def test_v3_removed_step3_worker_timeout_argument_is_rejected():
    """`step3_worker_timeout_seconds` 已移除，构造期直接 TypeError。"""
    with pytest.raises(TypeError, match="step3_worker_timeout_seconds"):
        PipelineConfig(
            pipeline_generation=3,
            step3_worker_timeout_seconds=7_200,
        )

    assert not hasattr(
        PipelineConfig(pipeline_generation=3), "step3_worker_timeout_seconds",
    )


@pytest.mark.parametrize("field", [
    "step3_worker_idle_timeout_seconds",
    "step3_worker_hard_timeout_seconds",
])
@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_v3_rejects_invalid_step3_worker_timeouts(field, value):
    with pytest.raises(ValueError, match=field):
        PipelineConfig(pipeline_generation=3, **{field: value})


def test_v3_exact_source_override_resolves_distinct_immutable_row_policies(tmp_path):
    profiles = _quality_profiles()
    for profile in profiles.values():
        profile["thresholds"].update({"nisqa": 3.7, "utmos": 3.9})
    config = PipelineConfig(
        pipeline_generation=3,
        step="step3",
        output_root=str(tmp_path / "output"),
        datasets=_minimal_datasets(tmp_path),
        selection_policies={
            "global": {"mos_models": ["scoreq"], "min_mos_models": 1},
            "quality_profiles": profiles,
            "sources": {
                "test": {
                    "speaker/one.wav": {
                        "mos_models": ["utmos", "nisqa"],
                        "min_mos_models": 1,
                    },
                    "speaker/two.wav": {"mos_models": ["nisqa"]},
                },
            },
        },
    )

    first = config.resolve_selection_policy(
        "test", "unknown", source_relative_path="speaker/one.wav",
    )
    second = config.resolve_selection_policy(
        "test", "unknown", source_relative_path="speaker/two.wav",
    )
    default = config.resolve_selection_policy(
        "test", "unknown", source_relative_path="speaker/other.wav",
    )

    assert first.mos_models == ("utmos", "nisqa")
    assert first.min_mos_models == 1
    assert second.mos_models == ("nisqa",)
    assert default.mos_models == ("scoreq",)
    assert config.mos_scoring_projection(
        "test", "unknown", source_relative_path="speaker/one.wav",
    )["route_identity"] == {
        "dataset_id": "test", "source_relative_path": "speaker/one.wav",
    }
    assert config.resolve_selection_policy(
        "test", "unknown", source_relative_path="speaker/./one.wav",
    ) is first
    assert config.selected_mos_models() == ("scoreq", "utmos", "nisqa")


def test_v3_baseline_and_exact_source_policy_hashes_match_independent_goldens(
    tmp_path,
):
    config = PipelineConfig(
        pipeline_generation=3,
        datasets=_minimal_datasets(tmp_path),
        selection_policies={
            "quality_profiles": _quality_profiles(),
            "sources": {"test": {"speaker/one.wav": {
                "mos_models": ["dnsmos"],
                "min_mos_models": 1,
            }}},
        },
    )
    thresholds = {
        "dnsmos": {"ovrl": 3.3, "sig": 3.4, "bak": 4.0, "p808": 3.8},
        "scoreq": 4.2,
        "urgent_pk": 3.5,
    }
    baseline_payload = json.dumps(
        {
            "dataset_id": "test",
            "language": "unknown",
            "category": "read_speech",
            "quality_profile": "quality_high",
            "route_identity": None,
            "effective_policy": {
                "mos_models": ["scoreq", "urgent_pk", "dnsmos"],
                "thresholds": thresholds,
                "min_mos_models": 3,
                "early_exit": None,
            },
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    exact_source_payload = json.dumps(
        {
            "dataset_id": "test",
            "language": "unknown",
            "category": "read_speech",
            "quality_profile": "quality_high",
            "route_identity": {
                "dataset_id": "test",
                "source_relative_path": "speaker/one.wav",
            },
            "effective_policy": {
                "mos_models": ["dnsmos"],
                "thresholds": thresholds,
                "min_mos_models": 1,
                "early_exit": None,
            },
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    assert baseline_payload == (
        '{"category":"read_speech","dataset_id":"test","effective_policy":'
        '{"early_exit":null,"min_mos_models":3,"mos_models":'
        '["scoreq","urgent_pk","dnsmos"],"thresholds":{"dnsmos":'
        '{"bak":4.0,"ovrl":3.3,"p808":3.8,"sig":3.4},"scoreq":4.2,'
        '"urgent_pk":3.5}},"language":"unknown","quality_profile":'
        '"quality_high","route_identity":null}'
    )
    assert exact_source_payload == (
        '{"category":"read_speech","dataset_id":"test","effective_policy":'
        '{"early_exit":null,"min_mos_models":1,"mos_models":["dnsmos"],'
        '"thresholds":{"dnsmos":{"bak":4.0,"ovrl":3.3,"p808":3.8,'
        '"sig":3.4},"scoreq":4.2,"urgent_pk":3.5}},"language":"unknown",'
        '"quality_profile":"quality_high","route_identity":{"dataset_id":'
        '"test","source_relative_path":"speaker/one.wav"}}'
    )
    baseline = config.resolve_selection_policy("test", "unknown")
    exact_source = config.resolve_selection_policy(
        "test", "unknown", source_relative_path="speaker/one.wav",
    )
    assert hashlib.sha256(baseline_payload.encode("utf-8")).hexdigest() == (
        "7aa593a1e3e7e0f54dbeea0ea12e4a5b45de640666b8df8bf12dce04d1f7e37b"
    )
    assert baseline.policy_hash == (
        "7aa593a1e3e7e0f54dbeea0ea12e4a5b45de640666b8df8bf12dce04d1f7e37b"
    )
    assert hashlib.sha256(exact_source_payload.encode("utf-8")).hexdigest() == (
        "518904cbadd1c279942d861dda09b7d06b9d96b22bf4104bb7b6d6d5b09dcabf"
    )
    assert exact_source.policy_hash == (
        "518904cbadd1c279942d861dda09b7d06b9d96b22bf4104bb7b6d6d5b09dcabf"
    )


@pytest.mark.parametrize(
    "source_path",
    ["", "/abs.wav", "../x.wav", "a/../b.wav", r"a\b.wav"],
)
def test_v3_exact_source_override_rejects_unsafe_paths(tmp_path, source_path):
    with pytest.raises(ValueError, match="source|route|path"):
        PipelineConfig(
            pipeline_generation=3,
            output_root=str(tmp_path / "output"),
            datasets=_minimal_datasets(tmp_path),
            selection_policies={
                "quality_profiles": _quality_profiles(),
                "sources": {"test": {source_path: {"min_mos_models": 1}}},
            },
        )


def test_v3_exact_source_override_rejects_canonical_route_collision(tmp_path):
    with pytest.raises(ValueError, match="duplicate source route"):
        PipelineConfig(
            pipeline_generation=3,
            output_root=str(tmp_path / "output"),
            datasets=_minimal_datasets(tmp_path),
            selection_policies={
                "quality_profiles": _quality_profiles(),
                "sources": {"test": {
                    "speaker/one.wav": {"min_mos_models": 1},
                    "speaker/./one.wav": {"min_mos_models": 2},
                }},
            },
        )


def test_v3_mos_asset_preflight_accesses_only_selected_models(tmp_path, monkeypatch):
    profiles = _quality_profiles()
    for profile in profiles.values():
        profile["thresholds"]["nisqa"] = 3.5
    model_dir = tmp_path / "nisqa-model"
    model_dir.mkdir()
    model_path = model_dir / "nisqa.tar"
    model_path.write_bytes(b"verified nisqa weights")
    config = PipelineConfig(
        pipeline_generation=3,
        step="step3",
        output_root=str(tmp_path / "output"),
        datasets=_minimal_datasets(tmp_path),
        selection_policies={
            "quality_profiles": profiles,
            "global": {"mos_models": ["nisqa"], "min_mos_models": 1},
        },
        nisqa_model_path=str(model_path),
        nisqa_model_version="NISQA-test",
        nisqa_model_sha256=local_model_tree_sha256(str(model_dir.resolve())),
    )
    from audio_filter_pipeline.processors import optional_runtime

    verified = []
    real_verify = optional_runtime.verify_local_model_asset

    def record_verify(asset):
        verified.append(asset.local_path)
        return real_verify(asset)

    monkeypatch.setattr(optional_runtime, "verify_local_model_asset", record_verify)

    config.validate(context=ValidationContext.V3_STEP3_EXECUTION)

    # 只有被选中的 nisqa 资产被访问：多访问任何一个模型都会让该列表变长。
    assert verified == [str(model_dir.resolve())]


def test_v3_selected_local_model_asset_digest_mismatch_fails_closed(tmp_path):
    profiles = _quality_profiles()
    for profile in profiles.values():
        profile["thresholds"]["utmos"] = 3.5
    model_dir = tmp_path / "utmos-model"
    model_dir.mkdir()
    model_path = model_dir / "utmos.pt"
    model_path.write_bytes(b"utmos weights")
    config = PipelineConfig(
        pipeline_generation=3,
        step="step3",
        output_root=str(tmp_path / "output"),
        datasets=_minimal_datasets(tmp_path),
        selection_policies={
            "quality_profiles": profiles,
            "global": {"mos_models": ["utmos"], "min_mos_models": 1},
        },
        utmos_model_path=str(model_path),
        utmos_model_version="UTMOS-test",
        utmos_model_sha256="0" * 64,
    )

    with pytest.raises(RuntimeError, match="digest mismatch"):
        config.validate(context=ValidationContext.V3_STEP3_EXECUTION)


@pytest.mark.parametrize("route_model", ["nisqa", "utmos"])
def test_v3_route_only_local_model_missing_assets_fail_step3_preflight(
    tmp_path, route_model,
):
    profiles = _quality_profiles()
    for profile in profiles.values():
        profile["thresholds"][route_model] = 3.5
    scoreq_dir = tmp_path / "scoreq"
    scoreq_dir.mkdir()
    config = PipelineConfig(
        pipeline_generation=3,
        step="step3",
        output_root=str(tmp_path / "output"),
        datasets=_minimal_datasets(tmp_path),
        selection_policies={
            "quality_profiles": profiles,
            "global": {"mos_models": ["scoreq"], "min_mos_models": 1},
            "sources": {"test": {
                "speaker/route.wav": {
                    "mos_models": [route_model], "min_mos_models": 1,
                },
            }},
        },
        scoreq_model_dir=str(scoreq_dir),
    )

    assert config.selected_mos_models() == ("scoreq", route_model)
    with pytest.raises(ValueError, match=route_model):
        config.validate(context=ValidationContext.V3_STEP3_EXECUTION)


def test_v3_mos_asset_preflight_does_not_access_unselected_local_asset(tmp_path):
    scoreq_dir = tmp_path / "scoreq"
    scoreq_dir.mkdir()
    config = PipelineConfig(
        pipeline_generation=3,
        step="step3",
        output_root=str(tmp_path / "output"),
        datasets=_minimal_datasets(tmp_path),
        selection_policies={
            "quality_profiles": _quality_profiles(),
            "global": {"mos_models": ["scoreq"], "min_mos_models": 1},
        },
        scoreq_model_dir=str(scoreq_dir),
        nisqa_model_path=str(tmp_path / "must-not-be-opened.tar"),
        nisqa_model_version="unused",
        nisqa_model_sha256="0" * 64,
    )

    assert config.selected_mos_models() == ("scoreq",)
    config.validate(context=ValidationContext.V3_STEP3_EXECUTION)


def _speaker_consistency():
    return {
        "enabled": True,
        "runtime": "wespeaker",
        "model_dir": "/opt/models/wespeaker/./english",
        "model_version": " model-v1 ",
        "model_sha256": "a" * 64,
        "runtime_version": "1.2.3",
        "target_sample_rate": 16_000,
        "enrollment_bundle_path": "/opt/enrollments/./alpha.jsonl",
        "enrollment_bundle_sha256": "b" * 64,
        "enrollment_exclusion_path": "/opt/enrollments/./alpha-exclusions.jsonl",
        "enrollment_exclusion_sha256": "c" * 64,
    }


def _real_speaker_consistency(tmp_path):
    model_dir = tmp_path / "wespeaker-model"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"verified wespeaker weights")
    bundle = tmp_path / "enrollment.jsonl"
    bundle.write_bytes(b'{"bundle":"verified"}\n')
    exclusion = tmp_path / "exclusion.jsonl"
    exclusion.write_bytes(b'{"exclusion":"verified"}\n')
    return {
        "enabled": True,
        "runtime": "wespeaker",
        "model_dir": str(model_dir),
        "model_version": "model-v1",
        "model_sha256": local_model_tree_sha256(str(model_dir)),
        "runtime_version": "runtime-v1",
        "target_sample_rate": 16_000,
        "enrollment_bundle_path": str(bundle),
        "enrollment_bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "enrollment_exclusion_path": str(exclusion),
        "enrollment_exclusion_sha256": hashlib.sha256(exclusion.read_bytes()).hexdigest(),
    }


def _speaker_config(tmp_path, policy, *, step="step5"):
    return _config(
        step=step,
        datasets=_minimal_datasets(tmp_path),
        speaker_consistency=policy,
        step6_mode="dry_run",
    )


def test_v3_speaker_placeholders_are_structural_but_fail_step5_execution(tmp_path):
    policy = {
        **_speaker_consistency(),
        "model_dir": "/REPLACE_ME/audio_filter_pipeline/wespeaker_model",
        "model_version": "replace-me",
        "model_sha256": "0" * 64,
        "runtime_version": "replace-me",
        "enrollment_bundle_path": "/REPLACE_ME/audio_filter_pipeline/enrollment.jsonl",
        "enrollment_bundle_sha256": "0" * 64,
        "enrollment_exclusion_path": "/REPLACE_ME/audio_filter_pipeline/exclusion.jsonl",
        "enrollment_exclusion_sha256": "0" * 64,
    }
    config = _speaker_config(tmp_path, policy)

    config.validate(context=ValidationContext.STRUCTURAL, stage_names=[])
    for context in (
        ValidationContext.V3_STEP0_EXECUTION,
        ValidationContext.V3_STEP1_EXECUTION,
        ValidationContext.V3_STEP2_EXECUTION,
        ValidationContext.V3_STEP3_EXECUTION,
        ValidationContext.V3_STEP4_EXECUTION,
    ):
        config.validate(context=context, stage_names=[])
    for context in (
        ValidationContext.V3_STEP5_EXECUTION,
        ValidationContext.V3_STEP6_EXECUTION,
    ):
        with pytest.raises(ValueError, match="placeholder|REPLACE_ME"):
            config.validate(context=context, stage_names=[])


@pytest.mark.parametrize(
    ("field", "placeholder"),
    [
        ("model_dir", "/REPLACE_ME/wespeaker"),
        ("enrollment_bundle_path", "/REPLACE_ME/enrollment.jsonl"),
        ("enrollment_exclusion_path", "/REPLACE_ME/exclusion.jsonl"),
        ("model_version", "replace-me"),
        ("runtime_version", "replace-me"),
        ("model_sha256", "0" * 64),
        ("enrollment_bundle_sha256", "0" * 64),
        ("enrollment_exclusion_sha256", "0" * 64),
    ],
)
def test_v3_speaker_execution_rejects_each_placeholder(tmp_path, field, placeholder):
    policy = _real_speaker_consistency(tmp_path)
    policy[field] = placeholder
    config = _speaker_config(tmp_path, policy)

    with pytest.raises(ValueError, match=f"{field}.*placeholder"):
        config.validate(context=ValidationContext.V3_STEP5_EXECUTION, stage_names=[])


def test_v3_speaker_execution_accepts_digest_bound_regular_assets(tmp_path, monkeypatch):
    policy = _real_speaker_consistency(tmp_path)
    config = _speaker_config(tmp_path, policy)
    from audio_filter_pipeline.processors import optional_runtime

    verified = []
    real_verify = optional_runtime.verify_local_model_asset

    def record_verify(asset):
        verified.append(asset.local_path)
        return real_verify(asset)

    monkeypatch.setattr(optional_runtime, "verify_local_model_asset", record_verify)

    config.validate(context=ValidationContext.V3_STEP5_EXECUTION, stage_names=[])
    config.validate(context=ValidationContext.V3_STEP6_EXECUTION, stage_names=[])

    # 两个 execution context 都必须真正校验一次 speaker 模型资产。
    assert verified == [str(Path(policy["model_dir"]).resolve())] * 2


@pytest.mark.parametrize(
    "field",
    ["model_sha256", "enrollment_bundle_sha256", "enrollment_exclusion_sha256"],
)
def test_v3_speaker_execution_rejects_asset_digest_mismatch(tmp_path, field):
    policy = _real_speaker_consistency(tmp_path)
    policy[field] = "f" * 64
    config = _speaker_config(tmp_path, policy)

    with pytest.raises(RuntimeError, match="digest mismatch"):
        config.validate(context=ValidationContext.V3_STEP5_EXECUTION, stage_names=[])


@pytest.mark.parametrize(
    "path_field",
    ["enrollment_bundle_path", "enrollment_exclusion_path"],
)
@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_v3_speaker_execution_rejects_unsafe_enrollment_files(
    tmp_path, path_field, kind,
):
    policy = _real_speaker_consistency(tmp_path)
    original = policy[path_field]
    unsafe = tmp_path / f"unsafe-{path_field}.jsonl"
    if kind == "symlink":
        unsafe.symlink_to(original)
    else:
        os.mkfifo(unsafe)
    policy[path_field] = str(unsafe)
    config = _speaker_config(tmp_path, policy)

    with pytest.raises(ValueError, match="symlink|regular file"):
        config.validate(context=ValidationContext.V3_STEP5_EXECUTION, stage_names=[])


@pytest.mark.parametrize("kind", ["symlink", "regular_file"])
def test_v3_speaker_execution_rejects_unsafe_model_tree(tmp_path, kind):
    policy = _real_speaker_consistency(tmp_path)
    original = policy["model_dir"]
    unsafe = tmp_path / "unsafe-model"
    if kind == "symlink":
        unsafe.symlink_to(original, target_is_directory=True)
    else:
        unsafe.write_bytes(b"not a model directory")
    policy["model_dir"] = str(unsafe)
    config = _speaker_config(tmp_path, policy)

    with pytest.raises(RuntimeError, match="asset|scan|directory|symlink"):
        config.validate(context=ValidationContext.V3_STEP5_EXECUTION, stage_names=[])


def _default_speaker_worker_limits():
    return {
        "timeout_ms": 900_000,
        "terminate_grace_ms": 500,
        "kill_wait_ms": 500,
        "rlimit_cpu_seconds": 60,
        "rlimit_nofile": 256,
        "rlimit_fsize_bytes": 128 * 1024 * 1024,
        "rlimit_as_bytes": 32 * 1024**3,
        "max_request_bytes": 1024 * 1024,
        "max_stdout_bytes": 4 * 1024 * 1024,
        "max_stderr_bytes": 4_096,
        "max_worker_script_bytes": 1024 * 1024,
        "max_json_depth": 32,
        "io_chunk_bytes": 64 * 1024,
    }


def test_v3_speaker_consistency_is_canonical_frozen_and_round_trips(tmp_path):
    raw = _speaker_consistency()
    config = PipelineConfig(
        pipeline_generation=3,
        step="step5",
        output_root=str(tmp_path / "output"),
        datasets=_minimal_datasets(tmp_path),
        selection_policies={"quality_profiles": _quality_profiles()},
        speaker_consistency=raw,
        step6_mode="dry_run",
    )

    raw["model_dir"] = "/mutated"
    assert config.speaker_consistency["model_dir"] == "/opt/models/wespeaker/english"
    with pytest.raises(TypeError):
        config.speaker_consistency["enabled"] = False
    expected = _speaker_consistency()
    expected["model_dir"] = "/opt/models/wespeaker/english"
    expected["model_version"] = "model-v1"
    expected["enrollment_bundle_path"] = "/opt/enrollments/alpha.jsonl"
    expected["enrollment_exclusion_path"] = "/opt/enrollments/alpha-exclusions.jsonl"
    expected["worker_timeout_seconds"] = 900.0
    expected["worker_stderr_tail_bytes"] = 4_096
    expected["worker_limits"] = _default_speaker_worker_limits()
    assert config.to_dict()["speaker_consistency"] == expected
    assert config.to_dict()["step6_mode"] == "dry_run"
    assert PipelineConfig(**config.to_dict()).to_dict() == config.to_dict()


def test_v3_speaker_consistency_canonicalizes_optional_gpu_ids():
    config = PipelineConfig(pipeline_generation=3, step6_mode="dry_run", speaker_consistency={**_speaker_consistency(), "gpu_ids": [7, 2]})

    assert config.speaker_consistency["gpu_ids"] == (7, 2)
    assert config.to_dict()["speaker_consistency"]["gpu_ids"] == [7, 2]


def test_v3_speaker_consistency_worker_limits_default_and_round_trip():
    config = PipelineConfig(pipeline_generation=3, step6_mode="dry_run", speaker_consistency=_speaker_consistency())

    assert config.speaker_consistency["worker_timeout_seconds"] == 900.0
    assert config.speaker_consistency["worker_stderr_tail_bytes"] == 4_096
    assert dict(config.speaker_consistency["worker_limits"]) == _default_speaker_worker_limits()
    assert PipelineConfig(**config.to_dict()).to_dict() == config.to_dict()


def test_v3_speaker_consistency_merges_and_round_trips_nested_worker_limits():
    config = PipelineConfig(
        pipeline_generation=3,
        step6_mode="dry_run",
        speaker_consistency={
            **_speaker_consistency(),
            "worker_limits": {
                "timeout_ms": 12_500,
                "max_stderr_bytes": 123,
                "rlimit_nofile": 128,
                "max_json_depth": 16,
            },
        },
    )

    limits = config.to_dict()["speaker_consistency"]["worker_limits"]
    assert limits == {
        **_default_speaker_worker_limits(),
        "timeout_ms": 12_500,
        "max_stderr_bytes": 123,
        "rlimit_nofile": 128,
        "max_json_depth": 16,
    }
    assert config.speaker_consistency["worker_timeout_seconds"] == 12.5
    assert config.speaker_consistency["worker_stderr_tail_bytes"] == 123
    assert PipelineConfig(**config.to_dict()).to_dict() == config.to_dict()


@pytest.mark.parametrize(("legacy_field", "legacy_value", "nested_field", "nested_value"), [
    ("worker_timeout_seconds", 10.0, "timeout_ms", 11_000),
    ("worker_stderr_tail_bytes", 100, "max_stderr_bytes", 101),
])
def test_v3_speaker_consistency_rejects_conflicting_legacy_and_nested_worker_limits(
    legacy_field, legacy_value, nested_field, nested_value,
):
    with pytest.raises(ValueError, match="conflicts"):
        PipelineConfig(
            pipeline_generation=3,
            step6_mode="dry_run",
            speaker_consistency={
                **_speaker_consistency(),
                legacy_field: legacy_value,
                "worker_limits": {nested_field: nested_value},
            },
        )


@pytest.mark.parametrize("worker_limits", [
    {"unknown": 1},
    {1: 1},
    {"timeout_ms": True},
    {"timeout_ms": 3_600_001},
    {"max_stderr_bytes": 65_537},
])
def test_v3_speaker_consistency_rejects_invalid_nested_worker_limits(worker_limits):
    with pytest.raises(ValueError, match="worker_limits"):
        PipelineConfig(
            pipeline_generation=3,
            step6_mode="dry_run",
            speaker_consistency={
                **_speaker_consistency(), "worker_limits": worker_limits,
            },
        )


@pytest.mark.parametrize("field,value", [
    ("worker_timeout_seconds", 0), ("worker_timeout_seconds", float("inf")),
    ("worker_timeout_seconds", True), ("worker_timeout_seconds", 3601),
    ("worker_stderr_tail_bytes", 0), ("worker_stderr_tail_bytes", True),
    ("worker_stderr_tail_bytes", 65_537),
])
def test_v3_speaker_consistency_rejects_invalid_worker_limits(field, value):
    with pytest.raises(ValueError, match=field):
        PipelineConfig(
            pipeline_generation=3,
            step6_mode="dry_run",
            speaker_consistency={**_speaker_consistency(), field: value},
        )


@pytest.mark.parametrize("gpu_ids", [[0, 0], [-1], ["0"], None])
def test_v3_speaker_consistency_rejects_invalid_gpu_ids(gpu_ids):
    with pytest.raises(ValueError, match="gpu_ids"):
        PipelineConfig(pipeline_generation=3, step6_mode="dry_run", speaker_consistency={**_speaker_consistency(), "gpu_ids": gpu_ids})


def test_v3_speaker_consistency_defaults_to_disabled_with_top_level_filter_mode():
    config = PipelineConfig(pipeline_generation=3)
    assert config.speaker_consistency == {"enabled": False}
    assert config.step6_mode == "filter"


def test_v3_step3_gpu_ids_are_frozen_serialized_and_round_trip():
    raw = [7, 2]
    config = PipelineConfig(pipeline_generation=3, step3_gpu_ids=raw)

    raw.append(3)
    assert config.step3_gpu_ids == (7, 2)
    assert config.to_dict()["step3_gpu_ids"] == [7, 2]
    assert PipelineConfig(**config.to_dict()).to_dict() == config.to_dict()


@pytest.mark.parametrize("gpu_ids", [None, {0}, [0, 0], [-1], ["0"], [0.0], [True]])
def test_v3_rejects_invalid_step3_gpu_ids(gpu_ids):
    with pytest.raises(ValueError, match="step3_gpu_ids"):
        PipelineConfig(pipeline_generation=3, step3_gpu_ids=gpu_ids)


def test_v3_audio_export_concurrency_defaults_and_round_trip():
    config = PipelineConfig(pipeline_generation=3)

    assert config.audio_export_workers == 32
    assert config.audio_export_max_pending == 64
    assert config.audio_export_max_inflight_bytes == 4 * 1024 * 1024 * 1024
    assert PipelineConfig(**config.to_dict()).to_dict() == config.to_dict()


@pytest.mark.parametrize(
    "legacy_field",
    [
        "step6_export_workers",
        "step6_export_max_pending",
        "step6_export_max_inflight_bytes",
    ],
)
def test_v3_constructor_rejects_removed_step6_audio_export_fields(legacy_field):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        PipelineConfig(pipeline_generation=3, **{legacy_field: 1})


def test_v3_step4_mode_defaults_to_filter_and_rejects_invalid_values():
    assert PipelineConfig(pipeline_generation=3).step4_mode == "filter"
    assert PipelineConfig(pipeline_generation=3, step4_mode="dry_run").step4_mode == "dry_run"

    with pytest.raises(ValueError, match="step4_mode"):
        PipelineConfig(pipeline_generation=3, step4_mode="bogus")
    with pytest.raises(ValueError, match="step4_mode"):
        PipelineConfig(pipeline_generation=3, step4_mode="publish")


def test_v3_step4_mode_dry_run_round_trips():
    config = PipelineConfig(pipeline_generation=3, step4_mode="dry_run")

    assert config.step4_mode == "dry_run"
    assert config.to_dict()["step4_mode"] == "dry_run"
    assert PipelineConfig(**config.to_dict()).to_dict() == config.to_dict()


def test_v3_export_roots_default_null_and_round_trip(tmp_path):
    config = PipelineConfig(pipeline_generation=3)
    assert config.step4_export_root is None
    assert config.step6_export_root is None

    rooted = PipelineConfig(
        pipeline_generation=3,
        step4_export_root=str(tmp_path / "se"),
        step6_export_root=str(tmp_path / "tse"),
    )
    assert rooted.step4_export_root == str(tmp_path / "se")
    assert rooted.step6_export_root == str(tmp_path / "tse")
    assert PipelineConfig(**rooted.to_dict()).to_dict() == rooted.to_dict()

    with pytest.raises(ValueError, match="step4_export_root"):
        PipelineConfig(pipeline_generation=3, step4_export_root="  ")


def test_v3_export_roots_reject_output_and_dataset_ancestor_conflicts_without_existing_paths(
    tmp_path,
):
    output_root = tmp_path / "artifacts"
    input_root = tmp_path / "source"
    datasets = _minimal_datasets(tmp_path, input_root=input_root)
    policies = {"quality_profiles": _quality_profiles()}

    with pytest.raises(ValueError, match="step4_export_root.*output_root"):
        PipelineConfig(
            pipeline_generation=3,
            output_root=str(output_root),
            datasets=datasets,
            selection_policies=policies,
            step4_export_root=str(output_root / "exports"),
        )

    with pytest.raises(ValueError, match="step6_export_root.*output_root"):
        PipelineConfig(
            pipeline_generation=3,
            output_root=str(output_root / "nested"),
            datasets=datasets,
            selection_policies=policies,
            step6_export_root=str(output_root),
        )

    with pytest.raises(ValueError, match="step4_export_root.*input_root"):
        PipelineConfig(
            pipeline_generation=3,
            output_root=str(output_root),
            datasets=datasets,
            selection_policies=policies,
            step4_export_root=str(input_root / "exports"),
        )


def test_v3_export_roots_reject_mutual_overlap(tmp_path):
    with pytest.raises(ValueError, match="step4_export_root.*step6_export_root"):
        PipelineConfig(
            pipeline_generation=3,
            step4_export_root=str(tmp_path / "shared"),
            step6_export_root=str(tmp_path / "shared" / "nested"),
        )

    with pytest.raises(ValueError, match="step4_export_root.*step6_export_root"):
        PipelineConfig(
            pipeline_generation=3,
            step4_export_root=str(tmp_path / "same"),
            step6_export_root=str(tmp_path / "same"),
        )


@pytest.mark.parametrize(
    ("workers", "max_pending"),
    [(0, 32), (True, 32), (8, 7), (8, True)],
)
def test_v3_rejects_invalid_audio_export_concurrency(workers, max_pending):
    with pytest.raises(ValueError, match="audio_export"):
        PipelineConfig(
            pipeline_generation=3,
            audio_export_workers=workers,
            audio_export_max_pending=max_pending,
        )


@pytest.mark.parametrize("max_inflight_bytes", [0, True, 1.0])
def test_v3_rejects_invalid_audio_export_inflight_budget(max_inflight_bytes):
    with pytest.raises(ValueError, match="audio_export_max_inflight_bytes"):
        PipelineConfig(
            pipeline_generation=3,
            audio_export_max_inflight_bytes=max_inflight_bytes,
        )


def test_v3_rejects_relative_dataset_input_root():
    with pytest.raises(ValueError, match="input_root must be an absolute path"):
        PipelineConfig(
            pipeline_generation=3,
            datasets={
                "alpha": {
                    "input_root": "relative/input",
                    "language": "en",
                    "category": "read_speech",
                    "quality_profile": "quality_high",
                    "speaker": {"source": "none", "required": False},
                }
            },
            selection_policies={"quality_profiles": _quality_profiles()},
        )


def test_v3_step1_vad_concurrency_round_trips():
    config = PipelineConfig(
        pipeline_generation=3,
        step1_vad_workers=3,
        step1_vad_max_pending=6,
    )

    assert config.step1_vad_workers == 3
    assert config.step1_vad_max_pending == 6
    assert PipelineConfig(**config.to_dict()).to_dict() == config.to_dict()


@pytest.mark.parametrize(
    ("workers", "max_pending"),
    [(0, 2), (True, 2), (2, 1), (2, 5), (65, 130), (2, True)],
)
def test_v3_rejects_invalid_step1_vad_concurrency(workers, max_pending):
    with pytest.raises(ValueError, match="step1_vad"):
        PipelineConfig(
            pipeline_generation=3,
            step1_vad_workers=workers,
            step1_vad_max_pending=max_pending,
        )


@pytest.mark.parametrize(
    ("speaker_consistency", "step6_mode", "error"),
    [
        ({"enabled": True, "unknown": 1}, "dry_run", "unknown fields"),
        ({"enabled": True, "runtime": "other"}, "dry_run", "requires fields"),
        ({**_speaker_consistency(), "model_version": "   "}, "dry_run", "model_version"),
        ({"enabled": True, "runtime": "other", "model_dir": "/model", "model_version": "v1", "model_sha256": "a" * 64,
          "runtime_version": "1", "target_sample_rate": 16000, "enrollment_bundle_path": "/bundle.jsonl",
          "enrollment_bundle_sha256": "b" * 64, "enrollment_exclusion_path": "/exclusion.jsonl", "enrollment_exclusion_sha256": "c" * 64}, "dry_run", "runtime"),
        ({"enabled": True, "runtime": "wespeaker", "model_dir": "relative", "model_version": "", "model_sha256": "A" * 64,
          "runtime_version": "", "target_sample_rate": 0, "enrollment_bundle_path": "/bundle.jsonl",
          "enrollment_bundle_sha256": "b" * 64, "enrollment_exclusion_path": "/exclusion.jsonl", "enrollment_exclusion_sha256": "c" * 64}, "dry_run", "model_dir"),
        (_speaker_consistency(), "filter", "step6_mode='filter' requires speaker_consistency.min_cosine_similarity"),
        ({**_speaker_consistency(), "min_cosine_similarity": float("nan")}, "dry_run", "min_cosine_similarity"),
        ({**_speaker_consistency(), "min_cosine_similarity": 1.1}, "dry_run", "min_cosine_similarity"),
        ({**_speaker_consistency(), "enrollment_bundle_path": "/bundles/alpha.json"}, "dry_run", "jsonl bundle"),
    ],
)
def test_v3_speaker_consistency_rejects_invalid_schema(speaker_consistency, step6_mode, error):
    with pytest.raises(ValueError, match=error):
        PipelineConfig(
            pipeline_generation=3,
            speaker_consistency=speaker_consistency,
            step6_mode=step6_mode,
        )


def test_v3_step6_filter_mode_requires_threshold_and_round_trips():
    speaker_consistency = {**_speaker_consistency(), "min_cosine_similarity": 0.8}
    config = PipelineConfig(
        pipeline_generation=3,
        speaker_consistency=speaker_consistency,
        step6_mode="filter",
    )
    assert "mode" not in config.to_dict()["speaker_consistency"]
    assert config.to_dict()["step6_mode"] == "filter"


def test_v3_speaker_consistency_rejects_nested_mode_key():
    with pytest.raises(ValueError, match="unknown fields"):
        PipelineConfig(
            pipeline_generation=3,
            step6_mode="dry_run",
            speaker_consistency={**_speaker_consistency(), "mode": "dry_run"},
        )



def test_v3_to_dict_is_json_safe_with_pathlike_output_root(tmp_path):
    import json

    serialized = PipelineConfig(
        pipeline_generation=3,
        step="step4",
        output_root=tmp_path / "output",
        datasets=_minimal_datasets(tmp_path),
        selection_policies={"quality_profiles": _quality_profiles()},
    ).to_dict()

    assert serialized["output_root"] == str(tmp_path / "output")
    json.dumps(serialized, allow_nan=False)


def test_v3_materializes_complete_global_policy_defaults(tmp_path):
    config = PipelineConfig(
        pipeline_generation=3,
        step="step1",
        output_root=str(tmp_path / "output"),
        datasets=_minimal_datasets(tmp_path),
        selection_policies={"quality_profiles": _quality_profiles()},
    )

    assert config.to_dict()["selection_policies"] == {
        "global": {
            "mos_models": ["scoreq", "urgent_pk", "dnsmos"],
            "thresholds": {},
            "min_mos_models": 3,
            "mos_fusion": "min",
            "early_exit": None,
        },
        "quality_profiles": _quality_profiles(),
    }


def test_v3_global_overrides_defaults_and_round_trips_exactly(tmp_path):
    config = PipelineConfig(
        pipeline_generation=3,
        step="step1",
        output_root=str(tmp_path / "output"),
        datasets=_minimal_datasets(tmp_path),
        selection_policies={
            "quality_profiles": _quality_profiles(),
            "global": {"models": ["DNSMOS"], "min_mos_models": 1},
        },
    )

    expected = {
        "global": {
            "mos_models": ["dnsmos"],
            "thresholds": {},
            "min_mos_models": 1,
            "mos_fusion": "min",
            "early_exit": None,
        },
        "quality_profiles": _quality_profiles(),
    }
    assert config.to_dict()["selection_policies"] == expected
    assert PipelineConfig(**config.to_dict()).to_dict() == config.to_dict()



@pytest.mark.parametrize(
    ("global_policy", "dataset_policy", "error"),
    [
        # registry 支持五个已声明模型；unknown 不在 registry 且无最终阈值，
        # 完整性 gate 在 resolve 阶段即拒绝，消息为
        # "model(s) [...] are missing final threshold(s)"
        ({"mos_models": ["unknown"], "min_mos_models": 1}, None,
         r"model\(s\) \['unknown'\].*missing final threshold"),
        ({"mos_models": ["dnsmos"], "min_mos_models": 2}, None,
         "number of MOS models"),
        ({"mos_models": ["dnsmos"], "min_mos_models": 1, "mos_fusion": "bogus"}, None,
         "mos_fusion"),
        (
            {
                "mos_models": ["dnsmos"],
                "thresholds": {"dnsmos": {"bogus": 3.0}},
                "min_mos_models": 1,
            },
            None,
            "未知维度键",
        ),
        # 越界维度值必须放在 dataset 层（最末合并层）：放在 global 层会被
        # 显式 profile 的合法阈值覆盖，失去拒绝语义
        (
            {"mos_models": ["dnsmos"], "min_mos_models": 1},
            {"thresholds": {"dnsmos": {"ovrl": 6.0}}},
            r"\[1.0, 5.0\]",
        ),
    ],
)
def test_v3_validate_rejects_invalid_resolved_policy_semantics(
    tmp_path, global_policy, dataset_policy, error,
):
    selection_policies = {
        "quality_profiles": _quality_profiles(),
        "global": global_policy,
    }
    if dataset_policy is not None:
        selection_policies["datasets"] = {"test": dataset_policy}
    config = PipelineConfig(
        pipeline_generation=3,
        step="step3",
        output_root=str(tmp_path / "output"),
        datasets=_minimal_datasets(tmp_path),
        selection_policies=selection_policies,
    )

    with pytest.raises(ValueError, match=error):
        config.validate(stage_names=[])


@pytest.mark.parametrize("output_root", [b"/tmp/output", "", "   "])
def test_v3_rejects_non_json_safe_or_empty_output_root(tmp_path, output_root):
    with pytest.raises(ValueError, match="output_root.*non-empty string"):
        PipelineConfig(
            pipeline_generation=3,
            step="step1",
            output_root=output_root,
            datasets=_minimal_datasets(tmp_path),
        )


def test_v3_execution_validation_requires_output_root(tmp_path):
    cfg = PipelineConfig(
        pipeline_generation=3,
        step="step1",
        datasets=_minimal_datasets(tmp_path),
        selection_policies={"quality_profiles": _quality_profiles()},
    )

    with pytest.raises(ValueError, match="output_root.*non-empty string"):
        cfg.validate(stage_names=[])



def test_v3_still_validates_shared_dataset_registry(tmp_path):
    invalid = _minimal_datasets(tmp_path)
    invalid["test"]["language"] = ""

    with pytest.raises(ValueError, match="dataset 'test'.*language"):
        _config(
            pipeline_generation=3,
            step="step1",
            datasets=invalid,
        )


@pytest.mark.parametrize(
    ("selection_policies", "error"),
    [
        ({"global": {"mos_models": "dnsmos"}}, "mos_models"),
        ({"global": {"min_mos_models": "bad"}}, "min_mos_models"),
        ({"datasets": {"ghost": {}}}, "unknown dataset"),
        ({"categories": {"ghost": {}}}, "unknown category"),
    ],
)
def test_v3_validates_shared_selection_policy_schema_at_construction(
    tmp_path, selection_policies, error,
):
    policies = {"quality_profiles": _quality_profiles(), **selection_policies}
    with pytest.raises(ValueError, match=error):
        _config(
            pipeline_generation=3,
            step="step1",
            datasets=_minimal_datasets(tmp_path),
            selection_policies=policies,
        )


@pytest.mark.parametrize(
    ("policy", "error"),
    [
        (
            {"models": ["dnsmos"], "mos_models": ["scoreq"]},
            "models.*mos_models",
        ),
        ({"thresholds": {"dnsmos": object()}}, "canonical JSON"),
        ({"thresholds": {"dnsmos": {1: 3.0}}}, "keys.*strings"),
        ({"thresholds": {"dnsmos": float("nan")}}, "finite"),
    ],
)
def test_v3_shared_policy_payload_is_unambiguous_and_canonical(
    tmp_path, policy, error,
):
    with pytest.raises(ValueError, match=error):
        _config(
            pipeline_generation=3,
            step="step1",
            datasets=_minimal_datasets(tmp_path),
            selection_policies={
                "quality_profiles": _quality_profiles(),
                "global": policy,
            },
        )



def _v3_profile_config(tmp_path, quality_profiles):
    return PipelineConfig(
        pipeline_generation=3,
        output_root=str(tmp_path / "output"),
        datasets={"test": {
            "input_root": str(tmp_path / "source"),
            "language": "en",
            "category": "read_speech",
            "quality_profile": "quality_high",
            "speaker": {"source": "none", "required": False},
        }},
        selection_policies={"quality_profiles": quality_profiles},
    )


@pytest.mark.parametrize(
    "profiles",
    [
        lambda value: value.pop("quality_balanced"),
        lambda value: value.__setitem__("quality_extra", value["quality_high"]),
    ],
)
def test_v3_quality_profile_registry_requires_exact_profile_names(tmp_path, profiles):
    quality_profiles = _quality_profiles()
    profiles(quality_profiles)
    with pytest.raises(ValueError, match="quality_profiles.*quality_balanced|quality_extra"):
        _v3_profile_config(tmp_path, quality_profiles)


def test_v3_quality_profile_accepts_any_registered_selectable_threshold_model(tmp_path):
    quality_profiles = _quality_profiles()
    quality_profiles["quality_high"]["thresholds"]["utmos"] = 4.0

    config = _v3_profile_config(tmp_path, quality_profiles)

    assert config.to_dict()["selection_policies"]["quality_profiles"]["quality_high"]["thresholds"]["utmos"] == 4.0


def test_v3_quality_profile_rejects_unknown_threshold_model(tmp_path):
    quality_profiles = _quality_profiles()
    quality_profiles["quality_high"]["thresholds"]["unknown_mos"] = 4.0

    with pytest.raises(ValueError, match="quality_profiles.*unknown model"):
        _v3_profile_config(tmp_path, quality_profiles)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["quality_high"]["thresholds"]["dnsmos"].pop("p808"),
        lambda value: value["quality_high"]["thresholds"]["dnsmos"].__setitem__("extra", 4.0),
    ],
)
def test_v3_quality_profile_requires_exact_dnsmos_dimensions(tmp_path, mutate):
    quality_profiles = _quality_profiles()
    mutate(quality_profiles)
    with pytest.raises(ValueError, match="quality_profiles.*dnsmos.*p808|extra"):
        _v3_profile_config(tmp_path, quality_profiles)


@pytest.mark.parametrize("forbidden", ["mos_models", "min_mos_models", "mos_fusion", "early_exit"])
def test_v3_quality_profile_allows_thresholds_field_only(tmp_path, forbidden):
    quality_profiles = _quality_profiles()
    quality_profiles["quality_high"][forbidden] = None
    with pytest.raises(ValueError, match=rf"quality_profiles.*{forbidden}"):
        _v3_profile_config(tmp_path, quality_profiles)


@pytest.mark.parametrize("bad", [object(), float("nan"), float("inf")])
def test_v3_quality_profile_thresholds_are_canonical_finite_json(tmp_path, bad):
    quality_profiles = _quality_profiles()
    quality_profiles["quality_high"]["thresholds"]["scoreq"] = bad
    with pytest.raises(ValueError, match="canonical JSON|finite"):
        _v3_profile_config(tmp_path, quality_profiles)


def test_v3_quality_profile_registry_canonicalizes_and_round_trips(tmp_path):
    cfg = _v3_profile_config(tmp_path, _quality_profiles())
    serialized = cfg.to_dict()
    assert serialized["selection_policies"]["quality_profiles"] == _quality_profiles()
    assert PipelineConfig(**serialized).to_dict() == serialized


def test_v3_quality_profile_preserves_integer_thresholds_at_construction(tmp_path):
    quality_profiles = _quality_profiles()
    quality_profiles["quality_high"]["thresholds"]["scoreq"] = 4
    quality_profiles["quality_high"]["thresholds"]["dnsmos"]["bak"] = 4

    config = _v3_profile_config(tmp_path, quality_profiles)
    serialized_thresholds = config.to_dict()["selection_policies"][
        "quality_profiles"
    ]["quality_high"]["thresholds"]
    resolved_thresholds = config.resolve_selection_policy("test").thresholds

    assert serialized_thresholds["scoreq"] == 4
    assert type(serialized_thresholds["scoreq"]) is int
    assert serialized_thresholds["dnsmos"]["bak"] == 4
    assert type(serialized_thresholds["dnsmos"]["bak"]) is int
    assert type(resolved_thresholds["scoreq"]) is int
    assert type(resolved_thresholds["dnsmos"]["bak"]) is int
    config.validate(context=ValidationContext.STRUCTURAL, stage_names=[])


@pytest.mark.parametrize(
    ("invalid_threshold", "error"),
    [
        (6, r"\[1.0, 5.0\]"),
        (True, "阈值"),
    ],
)
def test_v3_quality_profile_semantic_errors_are_deferred_until_validate(
    tmp_path, invalid_threshold, error,
):
    quality_profiles = _quality_profiles()
    quality_profiles["quality_high"]["thresholds"]["scoreq"] = invalid_threshold

    config = _v3_profile_config(tmp_path, quality_profiles)

    stored = config.to_dict()["selection_policies"]["quality_profiles"][
        "quality_high"
    ]["thresholds"]["scoreq"]
    assert stored == invalid_threshold
    assert type(stored) is type(invalid_threshold)
    with pytest.raises(ValueError, match=error):
        config.validate(context=ValidationContext.STRUCTURAL, stage_names=[])


def test_v3_resolved_thresholds_are_deeply_immutable(tmp_path):
    config = _v3_profile_config(tmp_path, _quality_profiles())
    resolved = config.resolve_selection_policy("test")
    before = config.to_dict()

    with pytest.raises(TypeError):
        resolved.thresholds["scoreq"] = 1.0
    with pytest.raises(TypeError):
        resolved.thresholds["dnsmos"]["ovrl"] = 1.0

    assert config.to_dict() == before
    assert resolved.thresholds["scoreq"] == 4.2
    assert resolved.thresholds["dnsmos"]["ovrl"] == 3.3


def test_v3_policy_snapshot_isolated_from_nested_caller_mutation(tmp_path):
    quality_profiles = _quality_profiles()
    policies = {
        "quality_profiles": quality_profiles,
        "sources": {"test": {"speaker/one.wav": {
            "mos_models": ["dnsmos"],
            "min_mos_models": 1,
        }}},
    }
    config = PipelineConfig(
        pipeline_generation=3,
        datasets=_minimal_datasets(tmp_path),
        selection_policies=policies,
    )
    before = config.to_dict()["selection_policies"]

    quality_profiles["quality_high"]["thresholds"]["dnsmos"]["ovrl"] = 1.0
    policies["sources"]["test"]["speaker/one.wav"]["mos_models"].append(
        "scoreq"
    )
    policies["sources"]["test"]["speaker/one.wav"]["min_mos_models"] = 2

    assert config.to_dict()["selection_policies"] == before
    baseline = config.resolve_selection_policy("test")
    exact_source = config.resolve_selection_policy(
        "test", source_relative_path="speaker/one.wav",
    )
    assert baseline.thresholds["dnsmos"]["ovrl"] == 3.3
    assert exact_source.thresholds["dnsmos"]["ovrl"] == 3.3
    assert exact_source.mos_models == ("dnsmos",)
    assert exact_source.min_mos_models == 1


def test_v3_dataset_registry_requires_quality_profiles(tmp_path):
    with pytest.raises(ValueError, match="quality_profiles"):
        PipelineConfig(
            pipeline_generation=3,
            datasets=_minimal_datasets(tmp_path),
        )


def test_v3_no_dataset_global_only_config_may_omit_quality_profiles():
    cfg = PipelineConfig(
        pipeline_generation=3,
        selection_policies={"global": {
            "thresholds": {
                "scoreq": 3.6,
                "urgent_pk": 3.5,
                "dnsmos": {"ovrl": 3.2},
            },
        }},
    )
    assert "quality_profiles" not in cfg.to_dict()["selection_policies"]
