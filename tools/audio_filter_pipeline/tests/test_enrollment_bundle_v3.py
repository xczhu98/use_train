"""离线 enrollment bundle 的确定性选择与来源绑定测试。"""

import hashlib
import json
import math
import os
import random
import threading

import pytest

import audio_filter_pipeline.processors.stages_v3.enrollment_bundle as enrollment_bundle
from audio_filter_pipeline.processors.stages_v3.enrollment_bundle import (
    _candidate_rows,
    _choose_enrollment,
    _canonical_bytes,
    _transaction_marker_path,
    build_enrollment_bundle,
    enrollment_exclusion_manifest_algorithm,
    write_enrollment_bundle,
)
from audio_filter_pipeline.processors.stages_v3.base import ArtifactStage
from audio_filter_pipeline.processors.stages_v3.select import STEP3_ROW_FIELDS
from audio_filter_pipeline.utils.artifacts import (
    ArtifactWriter,
    PipelineStep,
    inspect_artifact_unbound,
    open_artifact_for_read,
)

pytestmark = pytest.mark.unit


def _row(segment_id, speaker="speaker-a"):
    values = {
        "source_key": f"source-{segment_id}",
        "dataset_id": "alpha",
        "language": "en",
        "category": "read_speech",
        "quality_profile": "quality_high",
        "input_root": "/missing/audio",
        "original_path": f"/missing/audio/{segment_id}.wav",
        "original_relative_path": f"{segment_id}.wav",
        "source_relative_path": f"{segment_id}.wav",
        "source_size_bytes": 1,
        "source_sha256": hashlib.sha256(segment_id.encode()).hexdigest(),
        "sample_rate": 16000,
        "frames": 32000,
        "channels": 1,
        "duration_sec": 2.0,
        "format": "WAV",
        "subtype": "PCM_16",
        "speaker_id": speaker,
        "speaker_scope": "alpha" if speaker is not None else None,
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
        "end_sample": 16000,
        "boundary_reason": "silence",
        "mos_models": ["scoreq"],
        "mos_scoring_policy_hash": "b" * 64,
        "mos_scores": {"scoreq": 4.5},
    }
    assert tuple(values) == STEP3_ROW_FIELDS
    return values


def _published_step3(tmp_path, rows):
    writer = ArtifactWriter(
        str(tmp_path), PipelineStep.STEP3, "alpha", "1" * 64, "2" * 64,
    )
    for row in rows:
        writer.append(row)
    writer.finalize()
    return open_artifact_for_read(
        tmp_path,
        PipelineStep.STEP3,
        "alpha",
        expected_config_digest="1" * 64,
        expected_lineage_digest="2" * 64,
    )


class _IdentityStage(ArtifactStage):
    def __init__(self, step, identity):
        self.step = step
        self.upstream_step = None if step is PipelineStep.STEP0 else PipelineStep(
            f"step{int(step.value[-1]) - 1}"
        )
        self.identity = identity

    def config_payload(self, _config):
        return {"producer_identity": self.identity}

    def produce_rows(self, _context, _upstream_index, _upstream_rows):
        return ()


def _publish_identity_chain(tmp_path, stages):
    lineage = hashlib.sha256(b"").hexdigest()
    indexes = {}
    for step in PipelineStep:
        if step is PipelineStep.STEP4:
            break
        rows = [_row("a")] if step is PipelineStep.STEP3 else [{"step": step.value}]
        writer = ArtifactWriter(
            str(tmp_path), step, "alpha", stages[step].config_digest(object()), lineage,
        )
        for row in rows:
            writer.append(row)
        indexes[step] = writer.finalize()
        lineage = indexes[step].artifact_digest
    return indexes


def _write_inputs(tmp_path):
    bundle = [{"speaker_id": "speaker-a", "embedding": [1.0, 0.0]}]
    bundle_payload = _canonical_bytes(bundle[0]) + b"\n"
    manifest = {"bundle_sha256": hashlib.sha256(bundle_payload).hexdigest(), "schema": "test"}
    return bundle, manifest, tmp_path / "bundle.jsonl", tmp_path / "exclusion.jsonl"


def _marker_path(bundle_path, exclusion_path):
    return _transaction_marker_path(tuple(sorted((bundle_path, exclusion_path), key=lambda path: str(path))))


def _legacy_pairwise_enrollment(rows, embeddings, top_k):
    vectors = tuple(enrollment_bundle._normalize(embedding) for embedding in embeddings)
    scores = []
    for index, vector in enumerate(vectors):
        similarities = [
            math.fsum(left * right for left, right in zip(vector, other))
            for other_index, other in enumerate(vectors)
            if other_index != index
        ]
        scores.append(math.fsum(similarities) / len(similarities) if similarities else 1.0)
    selected = sorted(
        range(len(rows)), key=lambda index: (-scores[index], rows[index]["segment_id"]),
    )[:min(top_k, len(rows))]

    def medoid_score(index):
        similarities = [
            math.fsum(left * right for left, right in zip(vectors[index], vectors[other]))
            for other in selected
            if other != index
        ]
        return math.fsum(similarities) / len(similarities) if similarities else 1.0

    medoid = min(
        selected, key=lambda index: (-medoid_score(index), rows[index]["segment_id"]),
    )
    return (
        vectors[medoid],
        tuple(rows[index]["segment_id"] for index in selected),
        {rows[index]["segment_id"]: scores[index] for index in range(len(rows))},
    )


def _seeded_embeddings(count, dimension):
    generator = random.Random(20260908)
    return [
        tuple(generator.uniform(-1.0, 1.0) for _ in range(dimension))
        for _ in range(count)
    ]


def test_prepare_targets_requires_canonical_absolute_jsonl_paths(tmp_path):
    bundle_path = tmp_path / "bundle.jsonl"
    exclusion_path = tmp_path / "exclusion.jsonl"

    invalid_paths = (
        (str(tmp_path / "nested" / ".." / "bundle.jsonl"), str(exclusion_path)),
        (str(bundle_path), f"{tmp_path}/./exclusion.jsonl"),
        ("relative.jsonl", str(exclusion_path)),
        (str(tmp_path / "bundle.txt"), str(exclusion_path)),
    )
    for invalid_bundle, invalid_exclusion in invalid_paths:
        with pytest.raises(ValueError, match="path|canonical|absolute"):
            enrollment_bundle._prepare_targets(invalid_bundle, invalid_exclusion)


def test_prepare_targets_rejects_parent_symlink_and_target_symlink(tmp_path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="canonical|symlink"):
        enrollment_bundle._prepare_targets(
            str(linked_parent / "bundle.jsonl"),
            str(tmp_path / "exclusion.jsonl"),
        )

    outside = tmp_path / "outside.jsonl"
    outside.write_text("outside\n", encoding="utf-8")
    linked_target = tmp_path / "bundle.jsonl"
    linked_target.symlink_to(outside)
    with pytest.raises(ValueError, match="canonical|symlink"):
        enrollment_bundle._prepare_targets(
            str(linked_target),
            str(tmp_path / "exclusion.jsonl"),
        )


@pytest.mark.parametrize("target_kind", ["directory", "fifo"])
def test_prepare_targets_rejects_non_regular_targets(tmp_path, target_kind):
    bundle_path = tmp_path / "bundle.jsonl"
    exclusion_path = tmp_path / "exclusion.jsonl"
    if target_kind == "directory":
        bundle_path.mkdir()
    else:
        os.mkfifo(bundle_path)

    with pytest.raises(ValueError, match="unsupported type"):
        enrollment_bundle._prepare_targets(str(bundle_path), str(exclusion_path))


def test_prepare_targets_closes_already_opened_target_when_later_target_fails(tmp_path, monkeypatch):
    bundle_path = tmp_path / "bundle.jsonl"
    exclusion_path = tmp_path / "exclusion.jsonl"
    os.mkfifo(exclusion_path)
    real_open = enrollment_bundle.open_safe_output_path
    opened = []

    def recording_open(*args, **kwargs):
        safe = real_open(*args, **kwargs)
        opened.append(safe)
        return safe

    monkeypatch.setattr(enrollment_bundle, "open_safe_output_path", recording_open)
    with pytest.raises(ValueError, match="unsupported type"):
        enrollment_bundle._prepare_targets(str(bundle_path), str(exclusion_path))

    assert len(opened) == 1
    with pytest.raises(ValueError, match="closed"):
        _ = opened[0].parent_fd


def test_write_bundle_closes_prepared_targets_when_writer_fails(tmp_path, monkeypatch):
    bundle_path = tmp_path / "bundle.jsonl"
    exclusion_path = tmp_path / "exclusion.jsonl"
    captured = []

    def fail_writer(_bundle, _manifest, prepared):
        captured.extend(prepared)
        raise RuntimeError("writer failure")

    monkeypatch.setattr(enrollment_bundle, "_write_enrollment_bundle_prepared", fail_writer)
    with pytest.raises(RuntimeError, match="writer failure"):
        write_enrollment_bundle([], {}, str(bundle_path), str(exclusion_path))

    assert len(captured) == 2
    for target in captured:
        with pytest.raises(ValueError, match="closed"):
            _ = target.safe.parent_fd


def test_prepare_targets_closes_both_descriptors_for_duplicate_targets(tmp_path):
    target_path = tmp_path / "bundle.jsonl"
    with pytest.raises(ValueError, match="differ"):
        enrollment_bundle._prepare_targets(str(target_path), str(target_path))


def test_candidate_filter_keeps_all_explicit_speakers():
    candidates = _candidate_rows([
        _row("b"), _row("a"), _row("c"),
        {**_row("unknown"), "speaker_id": None, "speaker_scope": None},
    ], "alpha")
    assert list(candidates) == [("alpha", "alpha", "speaker-a")]
    assert [row["segment_id"] for row in candidates[("alpha", "alpha", "speaker-a")]] == ["a", "b", "c"]


def test_top_k_medoid_selection_is_deterministic_and_records_all_stability():
    rows = [_row("c"), _row("a"), _row("b")]
    embedding, segment_ids, stability = _choose_enrollment(rows, [(0.0, 1.0), (1.0, 0.0), (0.99, 0.1)], 2)
    assert segment_ids == ("b", "a")
    assert embedding == pytest.approx((1.0, 0.0))
    assert set(stability) == {"a", "b", "c"}


@pytest.mark.parametrize(
    ("embeddings", "top_k"),
    [
        ([(3.0, 4.0)], 1),
        ([(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)], 2),
        ([(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)], 3),
        (_seeded_embeddings(96, 48), 17),
    ],
    ids=["singleton", "repeated", "symmetric", "seeded-large-random"],
)
def test_sum_stability_matches_legacy_scores_and_is_deterministic(embeddings, top_k):
    rows = [_row(f"segment-{index:03d}") for index in reversed(range(len(embeddings)))]

    _expected_embedding, _expected_segment_ids, expected_stability = _legacy_pairwise_enrollment(
        rows, embeddings, top_k,
    )
    result = _choose_enrollment(rows, embeddings, top_k)

    assert result == _choose_enrollment(rows, embeddings, top_k)
    assert result[2].keys() == expected_stability.keys()
    assert result[2] == pytest.approx(expected_stability, abs=1e-12)
    assert all(math.isfinite(score) for score in result[2].values())


def test_linear_sum_algorithm_is_deterministic_for_near_tied_candidates():
    embeddings = [
        (1, -1.8876871532617207e-7, -7.237165745713061e-7),
        (1, 2.2205193181297563e-7, 8.041437829975525e-7),
        (1, -3.605691152698688e-8, 9.312261791878567e-7),
        (1, 5.602848663552904e-7, -7.852994424955306e-7),
        (1, -5.558980702550537e-7, 7.320396290348518e-7),
        (1, -1.5784969182718055e-7, -1.9922471425759226e-7),
    ]
    rows = [_row(f"segment-{index}") for index in range(len(embeddings))]

    legacy = _legacy_pairwise_enrollment(rows, embeddings, 3)
    result = _choose_enrollment(rows, embeddings, 3)

    assert legacy[1] == ("segment-5", "segment-1", "segment-4")
    assert result[1] == ("segment-5", "segment-1", "segment-2")
    assert result == _choose_enrollment(rows, embeddings, 3)


def test_enrollment_exclusion_manifest_algorithm_accepts_only_current_v2_contract():
    common = {
        "bundle_sha256": "a" * 64,
        "step3": {},
        "provenance": {},
        "enrollments": [],
    }
    manifest = {
        "schema": "audio_filter_pipeline.enrollment_exclusion.v2",
        "algorithm": "mean_cosine_linear_sum.v1",
        **common,
    }

    assert enrollment_exclusion_manifest_algorithm(manifest) == "mean_cosine_linear_sum.v1"

    with pytest.raises(ValueError, match="schema is unsupported"):
        enrollment_exclusion_manifest_algorithm({
            "schema": "audio_filter_pipeline.enrollment_exclusion.v1", **common,
        })
    with pytest.raises(ValueError, match="selection algorithm"):
        enrollment_exclusion_manifest_algorithm({
            **manifest, "algorithm": "mean_pairwise_cosine.v1",
        })
    with pytest.raises(ValueError, match="v2 schema"):
        enrollment_exclusion_manifest_algorithm({
            key: value for key, value in manifest.items() if key != "algorithm"
        })


def test_enrollment_normalization_accepts_large_finite_embeddings():
    embedding, segment_ids, stability = _choose_enrollment(
        [_row("a"), _row("b")],
        [(1e308, -1e308), (1e308, -1e308)],
        1,
    )

    assert embedding == pytest.approx((2 ** -0.5, -(2 ** -0.5)))
    assert segment_ids == ("a",)
    assert stability == pytest.approx({"a": 1.0, "b": 1.0})


class _Runtime:
    provenance = {
        "runtime": "wespeaker",
        "model_version": "v1",
        "model_sha256": "a" * 64,
        "runtime_version": "1",
        "target_sample_rate": 16000,
    }

    def extract_batch(self, values):
        return tuple((1.0, index / 10) for index, _value in enumerate(values))


def test_build_bundle_binds_selected_segments_and_step3_provenance(tmp_path):
    index = _published_step3(tmp_path, [_row("a"), _row("b")])

    bundle, manifest = build_enrollment_bundle(index, "alpha", _Runtime(), 1)

    assert set(bundle[0]) == {"dataset_id", "speaker_scope", "speaker_id", "embedding", "provenance"}
    assert manifest["schema"] == "audio_filter_pipeline.enrollment_exclusion.v2"
    assert manifest["algorithm"] == "mean_cosine_linear_sum.v1"
    assert enrollment_exclusion_manifest_algorithm(manifest) == "mean_cosine_linear_sum.v1"
    assert manifest["bundle_sha256"]
    assert manifest["step3"]["artifact_digest"] == index.artifact_digest
    assert manifest["enrollments"][0]["segment_ids"] == ["a", "b"]
    assert set(manifest["enrollments"][0]["stability"]) == {"a", "b"}


def test_build_bundle_rejects_unbound_step3_inspection(tmp_path):
    _published_step3(tmp_path, [_row("a")])
    index = inspect_artifact_unbound(tmp_path, PipelineStep.STEP3, "alpha")

    with pytest.raises(ValueError, match="production-bound Step3"):
        build_enrollment_bundle(index, "alpha", _Runtime(), 1)


def test_current_producer_identity_chain_returns_production_bound_step3(tmp_path):
    stages = {
        step: _IdentityStage(step, "current")
        for step in tuple(PipelineStep)[:4]
    }
    indexes = _publish_identity_chain(tmp_path, stages)

    index = enrollment_bundle.open_current_step3_for_enrollment(
        str(tmp_path), "alpha", object(), stages,
    )

    assert index.serialized() == indexes[PipelineStep.STEP3].serialized()
    assert build_enrollment_bundle(index, "alpha", _Runtime(), 1)[0]


def test_current_producer_identity_rejects_caller_selected_legacy_digest(tmp_path):
    published_stages = {
        step: _IdentityStage(step, "published")
        for step in tuple(PipelineStep)[:4]
    }
    _publish_identity_chain(tmp_path, published_stages)
    current_stages = dict(published_stages)
    current_stages[PipelineStep.STEP3] = _IdentityStage(
        PipelineStep.STEP3, "current",
    )

    with pytest.raises(ValueError, match="config digest"):
        enrollment_bundle.open_current_step3_for_enrollment(
            str(tmp_path), "alpha", object(), current_stages,
        )


def test_enrollment_cli_requires_config_and_removes_caller_digest_flags():
    actions = enrollment_bundle._parser()._option_string_actions

    assert "--config" in actions
    assert "--output-root" not in actions
    assert "--step3-config-digest" not in actions
    assert "--step3-lineage-digest" not in actions


def test_candidate_rows_rejects_legacy_field_subset():
    legacy = {
        field: value for field, value in _row("a").items()
        if field not in {"mos_models", "mos_scoring_policy_hash", "mos_scores"}
    }

    with pytest.raises(ValueError, match="exact current contract"):
        _candidate_rows([legacy], "alpha")


def test_build_bundle_empty_candidates_reports_step3_artifact(tmp_path):
    index = _published_step3(
        tmp_path, [{**_row("a", speaker=None), "speaker_scope": None}],
    )

    with pytest.raises(
        ValueError,
        match="verified Step3 artifact contains no enrollment candidates",
    ):
        build_enrollment_bundle(index, "alpha", object(), 1)


def test_write_bundle_publishes_both_files_and_removes_marker(tmp_path):
    bundle, manifest, bundle_path, exclusion_path = _write_inputs(tmp_path)

    bundle_sha256, exclusion_sha256 = write_enrollment_bundle(
        bundle, manifest, str(bundle_path), str(exclusion_path),
    )

    assert hashlib.sha256(bundle_path.read_bytes()).hexdigest() == bundle_sha256
    assert hashlib.sha256(exclusion_path.read_bytes()).hexdigest() == exclusion_sha256
    assert not _marker_path(bundle_path, exclusion_path).exists()
    assert not list(tmp_path.glob(".*.jsonl.*"))


def test_second_link_failure_rolls_back_targets_temps_and_marker(tmp_path, monkeypatch):
    bundle, manifest, bundle_path, exclusion_path = _write_inputs(tmp_path)
    real_link = enrollment_bundle._link_target
    calls = 0

    def fail_second(source_fd, source_name, target_fd, target_name):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("second link failed")
        return real_link(source_fd, source_name, target_fd, target_name)

    monkeypatch.setattr(enrollment_bundle, "_link_target", fail_second)
    with pytest.raises(OSError, match="second link failed"):
        write_enrollment_bundle(bundle, manifest, str(bundle_path), str(exclusion_path))

    assert not bundle_path.exists()
    assert not exclusion_path.exists()
    assert not _marker_path(bundle_path, exclusion_path).exists()
    assert not list(tmp_path.glob(".*.jsonl.*"))


def test_next_call_recovers_partial_publish_after_process_exception(tmp_path, monkeypatch):
    bundle, manifest, bundle_path, exclusion_path = _write_inputs(tmp_path)
    real_link = enrollment_bundle._link_target
    calls = 0

    def interrupt_after_first_link(source_fd, source_name, target_fd, target_name):
        nonlocal calls
        calls += 1
        real_link(source_fd, source_name, target_fd, target_name)
        if calls == 1:
            raise KeyboardInterrupt("simulated process interruption")

    monkeypatch.setattr(enrollment_bundle, "_link_target", interrupt_after_first_link)
    with pytest.raises(KeyboardInterrupt, match="simulated process interruption"):
        write_enrollment_bundle(bundle, manifest, str(bundle_path), str(exclusion_path))

    marker = _marker_path(bundle_path, exclusion_path)
    assert marker.exists()
    assert bundle_path.exists()
    assert not exclusion_path.exists()

    monkeypatch.setattr(enrollment_bundle, "_link_target", real_link)
    write_enrollment_bundle(bundle, manifest, str(bundle_path), str(exclusion_path))

    assert bundle_path.exists()
    assert exclusion_path.exists()
    assert not marker.exists()
    assert not list(tmp_path.glob(".*.jsonl.*"))


@pytest.mark.parametrize("field", ["token", "targets", "temp_path", "temp_ino", "content_sha256"])
def test_tampered_marker_fails_closed_without_deleting_user_file(tmp_path, field, monkeypatch):
    bundle, manifest, bundle_path, exclusion_path = _write_inputs(tmp_path)
    real_link = enrollment_bundle._link_target

    def interrupt_after_first_link(source_fd, source_name, target_fd, target_name):
        real_link(source_fd, source_name, target_fd, target_name)
        raise KeyboardInterrupt("simulated process interruption")

    monkeypatch.setattr(enrollment_bundle, "_link_target", interrupt_after_first_link)
    with pytest.raises(KeyboardInterrupt, match="simulated process interruption"):
        write_enrollment_bundle(bundle, manifest, str(bundle_path), str(exclusion_path))

    marker = _marker_path(bundle_path, exclusion_path)
    marker_value = json.loads(marker.read_text())
    user_file = tmp_path / "user.jsonl"
    if field == "token":
        marker_value[field] = "0" * 32
    elif field == "targets":
        marker_value[field][0]["target_path"] = str(user_file)
    elif field == "temp_path":
        marker_value["targets"][0][field] = str(user_file)
    elif field == "temp_ino":
        marker_value["targets"][0][field] += 1
    else:
        marker_value["targets"][0][field] = "0" * 64
    marker.write_text(json.dumps(marker_value, sort_keys=True, separators=(",", ":")) + "\n")
    user_file.write_text("user data\n")

    with pytest.raises(
        ValueError,
        match="marker|target paths|identity|content hash",
    ):
        write_enrollment_bundle(bundle, manifest, str(bundle_path), str(exclusion_path))

    assert user_file.read_text() == "user data\n"
    assert bundle_path.exists()
    assert marker.exists()


def test_marker_symlink_is_rejected_without_touching_target(tmp_path, monkeypatch):
    bundle, manifest, bundle_path, exclusion_path = _write_inputs(tmp_path)
    real_link = enrollment_bundle._link_target

    def interrupt_after_first_link(source_fd, source_name, target_fd, target_name):
        real_link(source_fd, source_name, target_fd, target_name)
        raise KeyboardInterrupt("simulated process interruption")

    monkeypatch.setattr(enrollment_bundle, "_link_target", interrupt_after_first_link)
    with pytest.raises(KeyboardInterrupt, match="simulated process interruption"):
        write_enrollment_bundle(bundle, manifest, str(bundle_path), str(exclusion_path))
    marker = _marker_path(bundle_path, exclusion_path)
    marker.unlink()
    marker_target = tmp_path / "marker-user-file"
    marker_target.write_text("user data\n")
    marker.symlink_to(marker_target)

    with pytest.raises(ValueError, match="marker must not be a symlink"):
        write_enrollment_bundle(bundle, manifest, str(bundle_path), str(exclusion_path))

    assert marker.is_symlink()
    assert marker_target.read_text() == "user data\n"
    assert bundle_path.exists()


def test_existing_target_is_never_overwritten(tmp_path):
    bundle, manifest, bundle_path, exclusion_path = _write_inputs(tmp_path)
    bundle_path.write_text("existing bundle\n")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_enrollment_bundle(bundle, manifest, str(bundle_path), str(exclusion_path))

    assert bundle_path.read_text() == "existing bundle\n"
    assert not exclusion_path.exists()


def test_concurrent_writer_cannot_enter_active_transaction(tmp_path, monkeypatch):
    bundle, manifest, bundle_path, exclusion_path = _write_inputs(tmp_path)
    entered_link = threading.Event()
    release_link = threading.Event()
    result = []
    real_link = enrollment_bundle._link_target

    def pause_first_link(source_fd, source_name, target_fd, target_name):
        entered_link.set()
        assert release_link.wait(timeout=5)
        return real_link(source_fd, source_name, target_fd, target_name)

    monkeypatch.setattr(enrollment_bundle, "_link_target", pause_first_link)

    def first_writer():
        try:
            write_enrollment_bundle(bundle, manifest, str(bundle_path), str(exclusion_path))
        except BaseException as exc:
            result.append(exc)

    thread = threading.Thread(target=first_writer)
    thread.start()
    assert entered_link.wait(timeout=5)
    with pytest.raises(FileExistsError, match="transaction is active"):
        write_enrollment_bundle(bundle, manifest, str(bundle_path), str(exclusion_path))
    release_link.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result == []
    assert bundle_path.exists()
    assert exclusion_path.exists()


@pytest.mark.parametrize("payload", [b"{invalid json", b"\xff\xfe{"])
def test_read_marker_wraps_malformed_payload(tmp_path, payload):
    """损坏的 marker 必须统一转换为 unreadable，而不是泄漏原始解码异常。"""
    marker = tmp_path / "marker.bin"
    marker.write_bytes(payload)
    descriptor = os.open(marker, os.O_RDONLY)
    try:
        with pytest.raises(ValueError, match="transaction marker is unreadable"):
            enrollment_bundle._read_marker(descriptor)
    finally:
        os.close(descriptor)
