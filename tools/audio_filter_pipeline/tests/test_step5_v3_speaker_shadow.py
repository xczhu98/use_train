"""Generation-3 Step5 frozen enrollment verification tests."""

import hashlib
import json
from pathlib import Path

import pytest

from audio_filter_pipeline.config import PipelineConfig
from audio_filter_pipeline.processors.stages_v3 import speaker_shadow
from audio_filter_pipeline.processors.stages_v3.base import (
    ArtifactExpectation,
    StageRunContext,
)
from audio_filter_pipeline.processors.stages_v3.export import Step6MultirateExportStage
from audio_filter_pipeline.processors.stages_v3.segment import STEP2_ROW_FIELDS
from audio_filter_pipeline.processors.stages_v3.select import Step3MosSelectionStage
from audio_filter_pipeline.processors.stages_v3.speaker_shadow import (
    STEP5_ROW_FIELDS,
    SpeakerGroupExtractorPool,
    Step5SpeakerShadowStage,
    _load_enrollment_bundle,
    _load_enrollment_exclusions,
)
from audio_filter_pipeline.processors.workers import wespeaker_group_worker
from audio_filter_pipeline.tests.stage_test_helpers import runner_expectation_for_test
from audio_filter_pipeline.tests.step3_test_helpers import CanonicalStep3Scorer
from audio_filter_pipeline.utils.artifacts import ArtifactWriter, PipelineStep, iter_artifact_rows

pytestmark = pytest.mark.unit


def _policy(bundle_path, bundle_sha256, exclusion_path, exclusion_sha256):
    return {"enabled": True, "runtime": "wespeaker", "model_dir": "/opt/wespeaker", "model_version": "model-v1", "model_sha256": "b" * 64, "runtime_version": "1.0", "target_sample_rate": 16000, "enrollment_bundle_path": str(bundle_path), "enrollment_bundle_sha256": bundle_sha256, "enrollment_exclusion_path": str(exclusion_path), "enrollment_exclusion_sha256": exclusion_sha256}


def _config(output_root, bundle_path, bundle_sha256, exclusion_path, exclusion_sha256):
    return PipelineConfig(pipeline_generation=3, step="step5", step6_mode="dry_run", output_root=str(output_root), datasets={"alpha": {"input_root": "/missing/audio", "language": "en", "category": "read_speech", "quality_profile": "quality_high", "speaker": {"source": "none", "required": False}}}, selection_policies={"quality_profiles": {name: {"thresholds": {"scoreq": 4.2, "urgent_pk": 3.5, "dnsmos": {"ovrl": 3.3, "sig": 3.4, "bak": 4.0, "p808": 3.8}}} for name in ("quality_balanced", "quality_high", "quality_premium")}}, speaker_consistency=_policy(bundle_path, bundle_sha256, exclusion_path, exclusion_sha256))


def _pool_options(**overrides):
    worker_script = Path(wespeaker_group_worker.__file__).with_name("wespeaker_worker.py")
    options = {
        "model_path": "/opt/model",
        "model_version": "model-v1",
        "model_sha256": "b" * 64,
        "runtime_version": "1.0",
        "target_sample_rate": 16000,
        "worker_script": str(worker_script.resolve()),
        "worker_script_sha256": hashlib.sha256(worker_script.read_bytes()).hexdigest(),
    }
    options.update(overrides)
    return options


def _row(segment_id="segment-1", speaker_id="speaker-a", speaker_scope="alpha"):
    config = _config(
        "/tmp/audio-filter-step5-test", "/tmp/bundle.jsonl", "a" * 64,
        "/tmp/exclusions.jsonl", "b" * 64,
    )
    projection = config.mos_scoring_projection("alpha", "en")
    return {
        "source_key": f"source-{segment_id}",
        "dataset_id": "alpha",
        "language": "en",
        "category": "read_speech",
        "quality_profile": "quality_high",
        "input_root": "/missing/audio",
        "original_path": f"/missing/audio/{segment_id}.wav",
        "original_relative_path": "",
        "source_relative_path": f"{segment_id}.wav",
        "source_size_bytes": 1,
        "source_sha256": hashlib.sha256(segment_id.encode()).hexdigest(),
        "sample_rate": 16000,
        "frames": 32000,
        "channels": 1,
        "duration_sec": 2.0,
        "format": "WAV",
        "subtype": "PCM_16",
        "speaker_id": speaker_id,
        "speaker_scope": speaker_scope,
        "source_start_sample": 0,
        "source_end_sample": 32000,
        "signal_finite": True,
        "signal_zero_ratio": 0.0,
        "signal_peak": 0.5,
        "signal_clipping_ratio": 0.0,
        "signal_dc_offset": 0.0,
        "signal_rms_dbfs": -20.0,
        "signal_num_samples": 32000,
        "bandwidth_hz": 7000.0,
        "estimated_sr": 16000,
        "vad_speech_ratio": 0.9,
        "vad_timeline_sample_rate": 16000,
        "vad_timeline_num_samples": 32000,
        "segment_id": segment_id,
        "start_sample": 0,
        "end_sample": 32000,
        "boundary_reason": "source_end",
        "mos_models": ["scoreq", "urgent_pk", "dnsmos"],
        "mos_scoring_policy_hash": projection["scoring_policy_hash"],
        "mos_scores": {
            "scoreq": 4.5,
            "urgent_pk": 3.8,
            "dnsmos": {"ovrl": 3.5, "sig": 3.6, "bak": 4.2, "p808": 4.0},
        },
    }


def _entry(speaker_id="speaker-a", embedding=(1.0, 0.0), **provenance):
    return {"dataset_id": "alpha", "speaker_scope": "alpha", "speaker_id": speaker_id, "embedding": list(embedding), "provenance": {"runtime": "wespeaker", "model_version": "model-v1", "model_sha256": "b" * 64, "runtime_version": "1.0", "target_sample_rate": 16000, **provenance}}


class _StableExtractor:
    provenance = {
        "capability": "injected_embedding_extractor",
        "implementation": "tests.stable_extractor.v1",
    }

    def __init__(self, callback):
        self._callback = callback

    def __call__(self, row):
        return self._callback(row)


def test_speaker_vectors_use_stable_l2_normalization_for_large_finite_values():
    from audio_filter_pipeline.processors.stages_v3 import speaker_shadow

    normalized = speaker_shadow._l2_normalize((1e308, -1e308))

    assert normalized == pytest.approx((2 ** -0.5, -(2 ** -0.5)))
    assert speaker_shadow._cosine(normalized, normalized) == pytest.approx(1.0)


def _files(tmp_path, entries, artifact_digest, enrollment_ids=("segment-1",), *, schema="v2", algorithm="mean_cosine_linear_sum.v1"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    bundle = tmp_path / "enrollment.jsonl"
    bundle.write_text("".join(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n" for entry in entries), encoding="utf-8")
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    manifest = {"schema": f"audio_filter_pipeline.enrollment_exclusion.{schema}", "algorithm": algorithm, "bundle_sha256": digest, "step3": {"dataset_id": "alpha", "config_digest": "1" * 64, "lineage_digest": "2" * 64, "artifact_digest": artifact_digest}, "provenance": entries[0]["provenance"], "enrollments": [{"dataset_id": "alpha", "speaker_scope": "alpha", "speaker_id": entry["speaker_id"], "segment_ids": list(enrollment_ids if entry["speaker_id"] == "speaker-a" else ()), "candidate_count": len(enrollment_ids), "stability": {segment_id: 1.0 for segment_id in enrollment_ids}} for entry in entries]}
    path = tmp_path / "exclusions.jsonl"
    path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return bundle, digest, path, hashlib.sha256(path.read_bytes()).hexdigest()


def _run(tmp_path, rows, bundle_entries, stage=None, enrollment_ids=("segment-1",)):
    output_root = tmp_path / "out"
    writer = ArtifactWriter(str(output_root), PipelineStep.STEP3, "alpha", "1" * 64, "2" * 64)
    for row in rows: writer.append(row)
    upstream = writer.finalize()
    bundle, digest, exclusions, exclusion_digest = _files(
        tmp_path, bundle_entries, upstream.artifact_digest, enrollment_ids,
    )
    config = _config(output_root, bundle, digest, exclusions, exclusion_digest)
    context = StageRunContext(
        str(output_root), "alpha", config,
        upstream=runner_expectation_for_test(upstream),
    )
    return list(iter_artifact_rows((stage or Step5SpeakerShadowStage()).run(context)))


def test_exact_source_mixed_tuples_flow_through_step3_step5_step6(tmp_path):
    profiles = {
        name: {"thresholds": {
            "scoreq": 4.2, "utmos": 3.9, "nisqa": 3.7,
            "urgent_pk": 3.5,
            "dnsmos": {"ovrl": 3.3, "sig": 3.4, "bak": 4.0, "p808": 3.8},
        }}
        for name in ("quality_balanced", "quality_high", "quality_premium")
    }
    config = PipelineConfig(
        pipeline_generation=3,
        step="step6",
        step6_mode="filter",
        output_root=str(tmp_path / "out"),
        datasets={"alpha": {
            "input_root": "/missing/audio", "language": "en",
            "category": "read_speech", "quality_profile": "quality_high",
            "speaker": {"source": "none", "required": False},
        }},
        selection_policies={
            "global": {"mos_models": ["scoreq"], "min_mos_models": 1},
            "quality_profiles": profiles,
            "sources": {"alpha": {"segment-2.wav": {
                "mos_models": ["utmos", "nisqa"], "min_mos_models": 1,
            }}},
        },
    )
    step2_rows = []
    for segment_id in ("segment-1", "segment-2"):
        row = _row(segment_id)
        step2_rows.append({field: row[field] for field in STEP2_ROW_FIELDS})
    writer = ArtifactWriter(
        config.output_root, PipelineStep.STEP2, "alpha", "0" * 64, "1" * 64,
    )
    for row in step2_rows:
        writer.append(row)
    step2 = writer.finalize()

    def expectation(index, step):
        assert PipelineStep(index.step) is step
        return runner_expectation_for_test(index)

    def scorer(_row, models):
        return ({"scoreq": 4.5} if models == ("scoreq",)
                else {"utmos": 4.1, "nisqa": 3.8})

    step3 = Step3MosSelectionStage(
        scorer=CanonicalStep3Scorer(scorer),
    ).run(StageRunContext(
        config.output_root, "alpha", config,
        upstream=expectation(step2, PipelineStep.STEP2),
    ))
    step5 = Step5SpeakerShadowStage().run(StageRunContext(
        config.output_root, "alpha", config,
        upstream=expectation(step3, PipelineStep.STEP3),
    ))
    step6 = Step6MultirateExportStage().run(StageRunContext(
        config.output_root, "alpha", config,
        upstream=expectation(step5, PipelineStep.STEP5),
    ))
    rows = list(iter_artifact_rows(step6))

    assert [row["mos_models"] for row in rows] == [
        ["scoreq"], ["utmos", "nisqa"],
    ]
    assert [row["speaker_shadow"]["status"] for row in rows] == [
        "disabled", "disabled",
    ]
    assert all(row["final_selection"]["export_eligible"] for row in rows)


def test_step5_excludes_enrollment_and_scores_only_remaining_trials(tmp_path):
    embeddings = {"segment-2": (0.0, 1.0)}
    rows = _run(
        tmp_path, [_row(), _row("segment-2")], [_entry()],
        Step5SpeakerShadowStage(_StableExtractor(lambda row: embeddings[row["segment_id"]])),
    )
    assert rows[0]["speaker_shadow"]["status"] == "not_trial"
    assert rows[1]["speaker_shadow"]["cosine_similarity_to_enrollment"] == pytest.approx(0.0)


def test_step5_exclusion_lookup_remains_correct_with_many_nonexcluded_rows(tmp_path):
    rows = [_row()]
    rows.extend(_row(f"segment-{index}", speaker_id="another-speaker") for index in range(2, 102))
    scored = _run(
        tmp_path, rows, [_entry()],
        Step5SpeakerShadowStage(_StableExtractor(lambda _row: (1.0, 0.0))),
    )

    assert scored[0]["speaker_shadow"]["status"] == "not_trial"
    assert len(scored) == 101
    assert all(row["speaker_shadow"]["status"] == "not_enrolled" for row in scored[1:])


def test_step5_enrollment_ids_never_reach_extractor(tmp_path):
    calls = []
    extractor = _StableExtractor(
        lambda row: calls.append(row["segment_id"]) or (1.0, 0.0)
    )
    rows = _run(
        tmp_path, [_row(), _row("segment-2")], [_entry()],
        Step5SpeakerShadowStage(extractor),
    )
    assert rows[0]["speaker_shadow"]["status"] == "not_trial"
    assert calls == ["segment-2"]


def test_step5_missing_enrollment_never_extracts_candidate(tmp_path):
    calls = []
    extractor = _StableExtractor(lambda row: calls.append(row) or (1.0, 0.0))
    rows = _run(
        tmp_path, [_row("segment-1"), _row("segment-2", "another-speaker")],
        [_entry()], Step5SpeakerShadowStage(extractor),
    )
    assert rows[1]["speaker_shadow"]["status"] == "not_enrolled"
    assert calls == []


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
def test_step5_stale_scoring_projection_fails_before_embedding_side_effect(
    tmp_path, mutate,
):
    stale_row = _row("segment-2")
    mutate(stale_row)
    embedding_calls = []
    stage = Step5SpeakerShadowStage(_StableExtractor(
        lambda row: embedding_calls.append(row["segment_id"]) or (1.0, 0.0)
    ))

    with pytest.raises(ValueError) as error:
        _run(tmp_path, [_row(), stale_row], [_entry()], stage)

    assert str(error.value) == "Step3 upstream row MOS scoring projection is stale"
    assert embedding_calls == []
    assert not (tmp_path / "out" / "step5" / "alpha").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda raw: raw.replace(
                b'{"dataset_id":', b'{"dataset_id":"alpha","dataset_id":', 1,
            ),
            id="duplicate-top-level-key",
        ),
        pytest.param(
            lambda raw: raw.replace(
                b'"provenance":{', b'"provenance":{"runtime":"wespeaker",', 1,
            ),
            id="duplicate-nested-key",
        ),
        pytest.param(
            lambda raw: raw.replace(b'"embedding":[1.0,0.0]', b'"embedding":[NaN,0.0]'),
            id="nan",
        ),
        pytest.param(
            lambda raw: raw.replace(b'"embedding":[1.0,0.0]', b'"embedding":[Infinity,0.0]'),
            id="infinity",
        ),
    ],
)
def test_step5_bundle_uses_strict_jsonl_decoder(tmp_path, mutation):
    raw = json.dumps(_entry(), sort_keys=True, separators=(",", ":")).encode() + b"\n"
    raw = mutation(raw)
    bundle = tmp_path / "enrollment.jsonl"
    bundle.write_bytes(raw)
    policy = _policy(
        bundle, hashlib.sha256(raw).hexdigest(), tmp_path / "unused.jsonl", "a" * 64,
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        _load_enrollment_bundle(policy)


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda raw: raw.replace(
                b'"bundle_sha256":',
                b'"bundle_sha256":"duplicate","bundle_sha256":', 1,
            ),
            id="duplicate-top-level-key",
        ),
        pytest.param(
            lambda raw: raw.replace(
                b'"step3":{', b'"step3":{"dataset_id":"alpha",', 1,
            ),
            id="duplicate-nested-key",
        ),
        pytest.param(
            lambda raw: raw.replace(b'"stability":{"segment-1":1.0}', b'"stability":{"segment-1":NaN}'),
            id="nan",
        ),
        pytest.param(
            lambda raw: raw.replace(b'"stability":{"segment-1":1.0}', b'"stability":{"segment-1":Infinity}'),
            id="infinity",
        ),
    ],
)
def test_step5_exclusion_manifest_uses_strict_json_decoder(tmp_path, mutation):
    writer = ArtifactWriter(
        str(tmp_path / "out"), PipelineStep.STEP3, "alpha", "1" * 64, "2" * 64,
    )
    writer.append(_row())
    upstream = writer.finalize()
    bundle, digest, exclusions, _ = _files(tmp_path / "assets", [_entry()], upstream.artifact_digest)
    raw = mutation(exclusions.read_bytes())
    exclusions.write_bytes(raw)
    policy = _policy(bundle, digest, exclusions, hashlib.sha256(raw).hexdigest())
    enrollments = _load_enrollment_bundle(policy)

    with pytest.raises(ValueError, match="invalid JSON"):
        _load_enrollment_exclusions(policy, enrollments, upstream)


def test_step5_loads_current_v2_enrollment_exclusion_manifest(tmp_path):
    writer = ArtifactWriter(
        str(tmp_path / "out"), PipelineStep.STEP3, "alpha", "1" * 64, "2" * 64,
    )
    writer.append(_row())
    upstream = writer.finalize()
    bundle, digest, exclusions, exclusion_digest = _files(
        tmp_path / "assets", [_entry()], upstream.artifact_digest,
    )
    policy = _policy(bundle, digest, exclusions, exclusion_digest)

    exclusions_by_segment = _load_enrollment_exclusions(
        policy, _load_enrollment_bundle(policy), upstream,
    )

    assert exclusions_by_segment == {"segment-1": ("alpha", "alpha", "speaker-a")}


def test_step5_rejects_old_v1_enrollment_exclusion_manifest(tmp_path):
    writer = ArtifactWriter(
        str(tmp_path / "out"), PipelineStep.STEP3, "alpha", "1" * 64, "2" * 64,
    )
    writer.append(_row())
    upstream = writer.finalize()
    bundle, digest, exclusions, exclusion_digest = _files(
        tmp_path / "assets", [_entry()], upstream.artifact_digest, schema="v1",
    )
    policy = _policy(bundle, digest, exclusions, exclusion_digest)

    with pytest.raises(ValueError, match="schema is unsupported"):
        _load_enrollment_exclusions(policy, _load_enrollment_bundle(policy), upstream)


def test_step5_rejects_v2_enrollment_exclusion_manifest_with_invalid_algorithm(tmp_path):
    writer = ArtifactWriter(
        str(tmp_path / "out"), PipelineStep.STEP3, "alpha", "1" * 64, "2" * 64,
    )
    writer.append(_row())
    upstream = writer.finalize()
    bundle, digest, exclusions, _ = _files(
        tmp_path / "assets", [_entry()], upstream.artifact_digest, schema="v2",
    )
    manifest = json.loads(exclusions.read_text())
    manifest["algorithm"] = "mean_pairwise_cosine.v1"
    exclusions.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    policy = _policy(bundle, digest, exclusions, hashlib.sha256(exclusions.read_bytes()).hexdigest())

    with pytest.raises(ValueError, match="selection algorithm"):
        _load_enrollment_exclusions(policy, _load_enrollment_bundle(policy), upstream)


def test_step5_injected_extractor_without_stable_provenance_fails_closed(tmp_path):
    config = _config(
        tmp_path / "out", tmp_path / "bundle.jsonl", "a" * 64,
        tmp_path / "exclusions.jsonl", "b" * 64,
    )

    with pytest.raises(ValueError, match="requires stable provenance"):
        Step5SpeakerShadowStage(lambda _row: (1.0, 0.0)).config_digest(config)


def test_step5_rejects_manifest_bundle_binding(tmp_path):
    output_root = tmp_path / "out"
    writer = ArtifactWriter(str(output_root), PipelineStep.STEP3, "alpha", "1" * 64, "2" * 64)
    writer.append(_row())
    upstream = writer.finalize()
    bundle, digest, exclusions, _ = _files(tmp_path, [_entry()], upstream.artifact_digest)
    raw = json.loads(exclusions.read_text())
    raw["bundle_sha256"] = "0" * 64
    exclusions.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n")
    config = _config(output_root, bundle, digest, exclusions, hashlib.sha256(exclusions.read_bytes()).hexdigest())
    context = StageRunContext(
        str(output_root), "alpha", config,
        upstream=runner_expectation_for_test(upstream),
    )
    with pytest.raises(ValueError, match="bound"):
        Step5SpeakerShadowStage().run(context)


def test_step5_rejects_manifest_for_another_step4_artifact(tmp_path):
    output_root = tmp_path / "out"
    writer = ArtifactWriter(str(output_root), PipelineStep.STEP3, "alpha", "1" * 64, "2" * 64)
    writer.append(_row())
    upstream = writer.finalize()
    bundle, digest, exclusions, exclusion_digest = _files(tmp_path, [_entry()], "0" * 64)
    config = _config(output_root, bundle, digest, exclusions, exclusion_digest)
    context = StageRunContext(
        str(output_root), "alpha", config,
        upstream=runner_expectation_for_test(upstream),
    )
    with pytest.raises(ValueError, match="Step3 artifact binding"):
        Step5SpeakerShadowStage().run(context)


def _pool_groups():
    rows = [_row("segment-2"), _row("segment-3", "speaker-b")]
    groups = {
        ("alpha", "alpha", "speaker-a"): [(0, rows[0])],
        ("alpha", "alpha", "speaker-b"): [(1, rows[1])],
    }
    enrollments = {
        ("alpha", "alpha", "speaker-a"): {"embedding": (1.0, 0.0)},
        ("alpha", "alpha", "speaker-b"): {"embedding": (1.0, 0.0)},
    }
    return groups, enrollments


def test_step5_gpu_pool_provenance_binds_runtime_assets_and_ordered_devices():
    pool = SpeakerGroupExtractorPool(_pool_options(), [7, 2])

    assert pool.provenance["gpu_ids"] == [7, 2]
    assert pool.provenance["model"] == {
        "path": "/opt/model", "version": "model-v1", "sha256": "b" * 64,
    }
    assert pool.provenance["runtime"] == {
        "name": "wespeaker", "version": "1.0", "target_sample_rate": 16000,
    }
    assert pool.provenance["worker"]["protocol"] == (
        "audio_filter_pipeline.wespeaker_group_jsonl.v3"
    )
    sources = pool.provenance["worker"]
    assert sources["group_script"] == {
        "path": str(Path(wespeaker_group_worker.__file__).resolve()),
        "sha256": hashlib.sha256(Path(wespeaker_group_worker.__file__).read_bytes()).hexdigest(),
    }
    assert sources["runtime_script"] == {
        "path": _pool_options()["worker_script"],
        "sha256": _pool_options()["worker_script_sha256"],
    }
    assert sources["runtime_script_config_sha256"] == _pool_options()["worker_script_sha256"]
    assert {name for name in sources if name.endswith("module")} == {
        "runtime_module", "optional_runtime_module",
    }
    assert all(
        Path(sources[name]["path"]).is_file()
        and sources[name]["sha256"] == hashlib.sha256(
            Path(sources[name]["path"]).read_bytes()
        ).hexdigest()
        for name in ("runtime_module", "optional_runtime_module")
    )


@pytest.mark.parametrize(
    "options,gpu_ids",
    [
        pytest.param(_pool_options(model_path="/opt/model-v2"), [7, 2], id="model-path"),
        pytest.param(_pool_options(model_version="model-v2"), [7, 2], id="model-version"),
        pytest.param(_pool_options(model_sha256="c" * 64), [7, 2], id="model-sha"),
        pytest.param(_pool_options(runtime_version="2.0"), [7, 2], id="runtime-version"),
        pytest.param(_pool_options(target_sample_rate=32000), [7, 2], id="target-sample-rate"),
        pytest.param(_pool_options(worker_script_sha256="c" * 64), [7, 2], id="worker-script-sha"),
        pytest.param(_pool_options(), [2, 7], id="ordered-gpu-ids"),
    ],
)
def test_step5_gpu_pool_identity_changes_cache_digest(tmp_path, options, gpu_ids):
    config = _config(
        tmp_path / "out", tmp_path / "bundle.jsonl", "a" * 64,
        tmp_path / "exclusions.jsonl", "b" * 64,
    )
    baseline = Step5SpeakerShadowStage(
        SpeakerGroupExtractorPool(_pool_options(), [7, 2])
    ).config_digest(config)
    changed = Step5SpeakerShadowStage(
        SpeakerGroupExtractorPool(options, gpu_ids)
    ).config_digest(config)

    assert changed != baseline


def test_step5_speaker_shadow_producer_version_is_incremented(tmp_path):
    config = _config(
        tmp_path / "out", tmp_path / "bundle.jsonl", "a" * 64,
        tmp_path / "exclusions.jsonl", "b" * 64,
    )

    assert Step5SpeakerShadowStage().config_payload(config)["speaker_shadow"]["version"] == 9


def test_step5_runtime_batch_and_gpu_pool_receive_trials_only(tmp_path, monkeypatch):
    class Runtime:
        provenance = {"capability": "wespeaker_optional_worker"}
        def extract_batch(self, members): return [(1.0, 0.0)] * len(members)
    rows = _run(tmp_path, [_row(), _row("segment-2")], [_entry()], Step5SpeakerShadowStage(Runtime()))
    assert rows[1]["speaker_shadow"]["status"] == "computed"
    captured = []

    class Child:
        returncode = 0
        pid = 1234
        def poll(self): return 0
        def communicate(self, input, timeout):
            payload = json.loads(input)
            captured.append(payload)
            result = [{"schema": "audio_filter_pipeline.wespeaker_group_result.v3", "group_key": group["key"], "segment_id": row["segment_id"], "enrollment_bundle_sha256": payload["enrollment_bundle_sha256"], "shadow": {"status": "computed", "cosine_similarity_to_enrollment": 1.0, "provenance": {"algorithm": "l2_normalized_segment_embedding_to_frozen_enrollment_cosine_v1", "inference_performed": True, "enrollment_bundle_sha256": payload["enrollment_bundle_sha256"]}}} for group in payload["groups"] for row in group["segments"]]
            return b"".join(json.dumps(item).encode() + b"\n" for item in result), b""

    monkeypatch.setattr("audio_filter_pipeline.processors.stages_v3.speaker_shadow.subprocess.Popen", lambda *_args, **_kwargs: Child())
    pool = SpeakerGroupExtractorPool(_pool_options(), [0, 1])
    groups, enrollments = _pool_groups()
    result = pool.extract_groups(groups, enrollments, "a" * 64)

    assert set(result) == set(groups)
    assert len(captured) == 2
    assert {payload["schema"] for payload in captured} == {"audio_filter_pipeline.wespeaker_group_request.v3"}
    assert all(payload["enrollment_bundle_sha256"] == "a" * 64 for payload in captured)
    assert all(group["enrollment_bundle_sha256"] == "a" * 64 for payload in captured for group in payload["groups"])


@pytest.mark.parametrize("mutation", [
    lambda result: result.__setitem__("enrollment_bundle_sha256", "b" * 64),
    lambda result: result["shadow"]["provenance"].__setitem__("algorithm", "other"),
    lambda result: result["shadow"]["provenance"].__setitem__("inference_performed", False),
    lambda result: result["shadow"].__setitem__("cosine_similarity_to_enrollment", float("nan")),
])
def test_step5_gpu_pool_rejects_tampered_provenance(monkeypatch, mutation):
    class Child:
        returncode = 0
        pid = 1234
        def poll(self): return 0
        def communicate(self, input, timeout):
            payload = json.loads(input)
            if not payload["groups"]:
                return b"", b""
            group, row = payload["groups"][0], payload["groups"][0]["segments"][0]
            result = {"schema": "audio_filter_pipeline.wespeaker_group_result.v3", "group_key": group["key"], "segment_id": row["segment_id"], "enrollment_bundle_sha256": payload["enrollment_bundle_sha256"], "shadow": {"status": "computed", "cosine_similarity_to_enrollment": 1.0, "provenance": {"algorithm": "l2_normalized_segment_embedding_to_frozen_enrollment_cosine_v1", "inference_performed": True, "enrollment_bundle_sha256": payload["enrollment_bundle_sha256"]}}}
            mutation(result)
            return json.dumps(result, allow_nan=True).encode() + b"\n", b""

    monkeypatch.setattr("audio_filter_pipeline.processors.stages_v3.speaker_shadow.subprocess.Popen", lambda *_args, **_kwargs: Child())
    groups, enrollments = _pool_groups()
    with pytest.raises(ValueError, match="bundle|provenance"):
        SpeakerGroupExtractorPool(_pool_options(), [0, 1]).extract_groups(
            {next(iter(groups)): next(iter(groups.values()))},
            {next(iter(enrollments)): next(iter(enrollments.values()))}, "a" * 64,
        )


def test_step5_gpu_pool_timeout_includes_bounded_stderr_tail(monkeypatch):
    from subprocess import TimeoutExpired

    class Child:
        returncode = -9
        pid = 1234
        def poll(self): return self.returncode
        def wait(self, timeout): return self.returncode
        def communicate(self, input=None, timeout=None):
            if input is not None: raise TimeoutExpired("worker", timeout)
            return b"", b"0123456789"

    monkeypatch.setattr("audio_filter_pipeline.processors.stages_v3.speaker_shadow.subprocess.Popen", lambda *_args, **_kwargs: Child())
    groups, enrollments = _pool_groups()
    first_key = next(iter(groups))
    with pytest.raises(ValueError, match="stderr_tail='6789'"):
        SpeakerGroupExtractorPool(_pool_options(), [0, 1], worker_timeout_seconds=0.1, worker_stderr_tail_bytes=4).extract_groups(
            {first_key: groups[first_key]}, {first_key: enrollments[first_key]}, "a" * 64,
        )


def test_step5_gpu_pool_timeout_preserves_partial_stderr_from_initial_timeout(monkeypatch):
    from subprocess import TimeoutExpired

    calls = []

    class Child:
        returncode = None
        pid = 5678

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = -15
            return self.returncode

        def communicate(self, input=None, timeout=None):
            calls.append((input, timeout))
            if input is not None:
                raise TimeoutExpired("worker", timeout, stderr=b"initial-tail")
            return b"", b""

    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.speaker_shadow.subprocess.Popen",
        lambda *_args, **_kwargs: Child(),
    )
    groups, enrollments = _pool_groups()
    first_key = next(iter(groups))
    with pytest.raises(ValueError, match="stderr_tail='initial-tail'"):
        SpeakerGroupExtractorPool(
            _pool_options(), [0], worker_timeout_seconds=0.1,
        ).extract_groups(
            {first_key: groups[first_key]}, {first_key: enrollments[first_key]}, "a" * 64,
        )
    assert len(calls) == 2
    assert calls[0][0] is not None and calls[0][1] == 0.1
    assert calls[1] == (None, 0.05)


def test_step5_gpu_pool_sends_one_bounded_request_per_shard_and_merges(monkeypatch):
    key = ("alpha", "alpha", "speaker-a")
    rows = [_row(f"segment-{index}") for index in range(2, 10)]
    groups = {key: [(index, row) for index, row in enumerate(rows)]}
    enrollments = {key: {"embedding": (1.0, 0.0)}}
    captured = []
    request_limit = 16_000

    class Child:
        returncode = 0
        pid = 1234

        def __init__(self, environment):
            self.environment = environment

        def poll(self):
            return 0

        def communicate(self, input, timeout):
            payload = json.loads(input)
            captured.append((self.environment, input, payload))
            assert len(input) <= request_limit
            result = [
                {
                    "schema": "audio_filter_pipeline.wespeaker_group_result.v3",
                    "group_key": group["key"],
                    "segment_id": row["segment_id"],
                    "enrollment_bundle_sha256": payload["enrollment_bundle_sha256"],
                    "shadow": {
                        "status": "computed",
                        "cosine_similarity_to_enrollment": 1.0,
                        "provenance": {
                            "algorithm": "l2_normalized_segment_embedding_to_frozen_enrollment_cosine_v1",
                            "inference_performed": True,
                            "enrollment_bundle_sha256": payload["enrollment_bundle_sha256"],
                        },
                    },
                }
                for group in payload["groups"]
                for row in group["segments"]
            ]
            return b"".join(json.dumps(item).encode() + b"\n" for item in result), b""

    def popen(*_args, **kwargs):
        return Child(kwargs["env"])

    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.speaker_shadow._GROUP_REQUEST_MAX_BYTES",
        request_limit,
    )
    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.speaker_shadow.subprocess.Popen",
        popen,
    )
    result = SpeakerGroupExtractorPool(
        _pool_options(model_assets={"sha256": "x" * 64}), [7],
    ).extract_groups(groups, enrollments, "a" * 64)

    assert len(captured) == 1
    assert [
        row["segment_id"]
        for group in captured[0][2]["groups"]
        for row in group["segments"]
    ] == [row["segment_id"] for row in rows]
    assert all("model_assets" not in payload["options"] for _env, _input, payload in captured)
    assert all(payload["options"]["cuda_visible_devices"] == "7" for _env, _input, payload in captured)
    assert all(environment["CUDA_VISIBLE_DEVICES"] == "7" for environment, _input, _payload in captured)
    assert set(result) == {key}
    assert set(result[key]) == {row["segment_id"] for row in rows}


def test_step5_gpu_pool_rejects_frozen_dependency_drift_before_starting_child(monkeypatch):
    groups, enrollments = _pool_groups()
    popen_calls = []
    pool = SpeakerGroupExtractorPool(_pool_options(), [0])
    frozen_runtime_module = pool._frozen_sources["runtime_module"]["path"]
    original_hash = speaker_shadow._file_sha256

    def hash_with_drift(path):
        return "0" * 64 if path == frozen_runtime_module else original_hash(path)

    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.speaker_shadow._file_sha256",
        hash_with_drift,
    )
    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.speaker_shadow.subprocess.Popen",
        lambda *_args, **_kwargs: popen_calls.append(True),
    )

    with pytest.raises(ValueError, match="runtime_module source changed"):
        pool.extract_groups(groups, enrollments, "a" * 64)

    assert popen_calls == []


def test_step5_gpu_pool_rejects_single_oversized_segment_before_starting_child(monkeypatch):
    key = ("alpha", "alpha", "speaker-a")
    row = _row("oversized-segment")
    row["payload_padding"] = "x" * 2_048
    groups = {key: [(0, row)]}
    enrollments = {key: {"embedding": (1.0, 0.0)}}
    popen_calls = []

    def popen(*_args, **_kwargs):
        popen_calls.append(True)
        raise AssertionError("oversized request must fail before starting a child")

    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.speaker_shadow._GROUP_REQUEST_MAX_BYTES",
        1_024,
    )
    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.speaker_shadow.subprocess.Popen",
        popen,
    )
    with pytest.raises(
        ValueError, match="GPU shard 0 exceeds safe GPU child request payload limit",
    ):
        SpeakerGroupExtractorPool(_pool_options(), [0]).extract_groups(
            groups, enrollments, "a" * 64,
        )
    assert popen_calls == []


def test_step5_group_child_extracts_in_lazy_batches_of_at_most_32(monkeypatch):
    batch_sizes, sessions = [], []

    class Options:
        @staticmethod
        def from_payload(payload):
            assert payload == {}
            return payload

    class Runtime:
        provenance = {"capability": "wespeaker_optional_worker"}

        def __init__(self, options):
            assert options == {}

        def session(self):
            sessions.append(self)
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_batch(self, rows):
            batch_sizes.append(len(rows))
            return [(1.0, 0.0)] * len(rows)

    monkeypatch.setattr(wespeaker_group_worker, "WeSpeakerRuntimeOptions", Options)
    monkeypatch.setattr(wespeaker_group_worker, "WeSpeakerEmbeddingRuntime", Runtime)
    segments = [{"segment_id": f"segment-{index}"} for index in range(33)]
    request = {
        "schema": "audio_filter_pipeline.wespeaker_group_request.v3",
        "options": {},
        "enrollment_bundle_sha256": "a" * 64,
        "groups": [{
            "key": ["alpha", "alpha", "speaker-a"],
            "segments": segments,
            "enrollment_embedding": [1.0, 0.0],
            "enrollment_bundle_sha256": "a" * 64,
        }],
    }

    result = wespeaker_group_worker._handle(request)

    assert batch_sizes == [32, 1]
    assert len(sessions) == 1
    assert [row["segment_id"] for row in result] == [
        row["segment_id"] for row in segments
    ]


@pytest.mark.parametrize(
    "groups",
    [
        [{
            "key": ["alpha", "alpha", "speaker-a"],
            "segments": [{"segment_id": "duplicate"}, {"segment_id": "duplicate"}],
            "enrollment_embedding": [1.0, 0.0],
            "enrollment_bundle_sha256": "a" * 64,
        }],
        [
            {
                "key": ["alpha", "alpha", "speaker-a"],
                "segments": [{"segment_id": "duplicate"}],
                "enrollment_embedding": [1.0, 0.0],
                "enrollment_bundle_sha256": "a" * 64,
            },
            {
                "key": ["alpha", "alpha", "speaker-b"],
                "segments": [{"segment_id": "duplicate"}],
                "enrollment_embedding": [1.0, 0.0],
                "enrollment_bundle_sha256": "a" * 64,
            },
        ],
    ],
    ids=["same-group", "cross-group"],
)
def test_step5_group_child_rejects_duplicate_segment_identity_across_full_request(
    monkeypatch, groups,
):
    class Options:
        @staticmethod
        def from_payload(payload):
            return payload

    class Runtime:
        provenance = {"capability": "wespeaker_optional_worker"}

        def __init__(self, _options):
            pass

        def extract_batch(self, rows):
            return [(1.0, 0.0)] * len(rows)

    monkeypatch.setattr(wespeaker_group_worker, "WeSpeakerRuntimeOptions", Options)
    monkeypatch.setattr(wespeaker_group_worker, "WeSpeakerEmbeddingRuntime", Runtime)
    request = {
        "schema": "audio_filter_pipeline.wespeaker_group_request.v3",
        "options": {},
        "enrollment_bundle_sha256": "a" * 64,
        "groups": groups,
    }

    with pytest.raises(ValueError, match="duplicate segment identity"):
        wespeaker_group_worker._handle(request)


def test_step5_group_child_late_runtime_failure_replaces_partial_shard_results(monkeypatch):
    class Options:
        @staticmethod
        def from_payload(payload):
            return payload

    class Runtime:
        provenance = {"capability": "wespeaker_optional_worker"}

        def __init__(self, _options):
            self.calls = 0

        def session(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_batch(self, rows):
            self.calls += 1
            if self.calls == 2:
                raise wespeaker_group_worker.WeSpeakerRuntimeError("late failure")
            return [(1.0, 0.0)] * len(rows)

    monkeypatch.setattr(wespeaker_group_worker, "WeSpeakerRuntimeOptions", Options)
    monkeypatch.setattr(wespeaker_group_worker, "WeSpeakerEmbeddingRuntime", Runtime)
    segments = [{"segment_id": f"segment-{index}"} for index in range(33)]
    request = {
        "schema": "audio_filter_pipeline.wespeaker_group_request.v3",
        "options": {},
        "enrollment_bundle_sha256": "a" * 64,
        "groups": [{
            "key": ["alpha", "alpha", "speaker-a"],
            "segments": segments,
            "enrollment_embedding": [1.0, 0.0],
            "enrollment_bundle_sha256": "a" * 64,
        }],
    }

    result = wespeaker_group_worker._handle(request)

    assert [row["segment_id"] for row in result] == [
        row["segment_id"] for row in segments
    ]
    assert all(row["shadow"]["status"] == "unavailable" for row in result)


def test_step5_gpu_pool_retains_stubborn_child_for_coordinated_cleanup(monkeypatch):
    class Child:
        pid = 1234
        returncode = None

        def poll(self):
            return None

    child = Child()
    terminated = []
    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.speaker_shadow.subprocess.Popen",
        lambda *_args, **_kwargs: child,
    )
    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.speaker_shadow.run_bounded_subprocess",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            wespeaker_group_worker.WeSpeakerRuntimeError("bounded I/O failed")
        ),
    )
    groups, enrollments = _pool_groups()
    first_key = next(iter(groups))
    pool = SpeakerGroupExtractorPool(_pool_options(), [0])
    pool._terminate_process_group = lambda proc: terminated.append(proc)

    with pytest.raises(wespeaker_group_worker.WeSpeakerRuntimeError, match="bounded I/O failed"):
        pool.extract_groups(
            {first_key: groups[first_key]}, {first_key: enrollments[first_key]}, "a" * 64,
        )

    assert terminated == [child]
    assert child in pool._processes


def test_step5_gpu_pool_cleanup_attempts_every_child_and_prunes_only_exited():
    class Child:
        def __init__(self, pid, returncode=None):
            self.pid = pid
            self.returncode = returncode

        def poll(self):
            return self.returncode

    failed = Child(1001)
    exited = Child(1002)
    stubborn = Child(1003)
    attempts = []
    pool = SpeakerGroupExtractorPool(_pool_options(), [0])
    pool._processes.update((failed, exited, stubborn))

    def terminate(proc):
        attempts.append(proc)
        if proc is failed:
            raise RuntimeError("first cleanup failed")
        if proc is exited:
            proc.returncode = -15

    pool._terminate_process_group = terminate

    pool._terminate_active_children()

    assert set(attempts) == {failed, exited, stubborn}
    assert pool._processes == {failed, stubborn}


def test_step5_closes_upstream_iterator_when_consumer_cancels(tmp_path):
    output_root = tmp_path / "out"
    writer = ArtifactWriter(output_root, PipelineStep.STEP3, "alpha", "1" * 64, "2" * 64)
    writer.append(_row())
    upstream = writer.finalize()
    bundle, digest, exclusions, exclusion_digest = _files(
        tmp_path, [_entry()], upstream.artifact_digest,
    )
    config = _config(output_root, bundle, digest, exclusions, exclusion_digest)
    context = StageRunContext(
        str(output_root), "alpha", config,
        upstream=ArtifactExpectation(
            PipelineStep.STEP3, upstream.config_digest, upstream.lineage_digest,
        ),
    )

    class TrackingIterator:
        def __init__(self, values):
            self._values = iter(values)
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._values)

        def close(self):
            self.closed = True

    upstream_rows = TrackingIterator([_row()])
    produced = Step5SpeakerShadowStage().produce_rows(context, upstream, upstream_rows)

    next(produced)
    produced.close()

    assert upstream_rows.closed


def test_step5_v2_artifact_resume_fails_before_producing_without_rewrite(tmp_path):
    output_root = tmp_path / "out"
    upstream_writer = ArtifactWriter(
        str(output_root), PipelineStep.STEP3, "alpha", "1" * 64, "2" * 64,
    )
    upstream_writer.append(_row())
    upstream = upstream_writer.finalize()
    bundle, digest, exclusions, exclusion_digest = _files(
        tmp_path / "assets", [_entry()], upstream.artifact_digest,
    )
    config = _config(output_root, bundle, digest, exclusions, exclusion_digest)

    class LegacyStep5V2(Step5SpeakerShadowStage):
        def config_payload(self, current_config):
            payload = dict(super().config_payload(current_config))
            payload["row_contract"] = {
                "version": 2,
                "fields": [*STEP5_ROW_FIELDS[:-1], "speaker_shadow"],
            }
            return payload

    legacy = LegacyStep5V2()
    writer = ArtifactWriter(
        config.output_root, PipelineStep.STEP5, "alpha",
        legacy.config_digest(config), upstream.artifact_digest,
    )
    writer.append({**_row(), "speaker_shadow": {"status": "legacy"}})
    writer.finalize()
    artifact_dir = Path(config.output_root) / "step5" / "alpha"
    before = {
        name: (artifact_dir / name).read_bytes()
        for name in ("rows.jsonl", "index.json", "_SUCCESS")
    }
    producer_calls = []

    class CurrentStage(Step5SpeakerShadowStage):
        def produce_rows(self, *args, **kwargs):
            producer_calls.append(True)
            return super().produce_rows(*args, **kwargs)

    with pytest.raises(ValueError, match="expected config digest mismatch"):
        CurrentStage().run(StageRunContext(
            config.output_root, "alpha", config,
            upstream=runner_expectation_for_test(upstream),
        ))

    assert producer_calls == []
    assert {
        name: (artifact_dir / name).read_bytes() for name in before
    } == before


def test_step5_declares_v3_output_and_exact_step3_upstream_contract(tmp_path):
    config = _config(
        tmp_path / "out", tmp_path / "bundle.jsonl", "a" * 64,
        tmp_path / "exclusions.jsonl", "b" * 64,
    )
    payload = Step5SpeakerShadowStage().config_payload(config)

    assert payload["row_contract"] == {
        "version": 3,
        "fields": list(STEP5_ROW_FIELDS),
    }
    assert payload["upstream_row_contract"] == {
        "version": 3,
        "fields": list(STEP5_ROW_FIELDS[:-1]),
    }
    assert "step3_forward_compatible" not in str(payload)
