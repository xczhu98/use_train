"""Step3 持久 MOS resume store 的有界、fail-closed 行为。"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from audio_filter_pipeline.processors.stages_v3 import mos_resume
from audio_filter_pipeline.processors.stages_v3.mos_resume import (
    ResumeStoreError,
    Step3ResumeIdentity,
    Step3ResumeStore,
    source_boundary_fingerprint,
)
from audio_filter_pipeline.processors.stages_v3.runner import DatasetRunContext


pytestmark = pytest.mark.unit


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity(dataset_id: str = "alpha", **overrides):
    fields = {
        "dataset_id": dataset_id,
        "step2_artifact_digest": _digest("artifact"),
        "step2_rows_sha256": _digest("rows"),
        "step2_row_count": 3,
        "step2_lineage_digest": _digest("lineage"),
        "step3_config": {"route": "alpha", "sample_rate": 16000},
        "model_provider": {"models": ["scoreq"], "provider": "cuda"},
        "execution_policy": {"scheduler": "weighted_lpt", "fallback": False},
    }
    fields.update(overrides)
    return Step3ResumeIdentity.create(Step3ResumeStore.identity_payload(**fields))


def _descriptor(name: str, ordinals: list[int]):
    bindings = [
        {"models": ["scoreq"], "scoring_policy_hash": _digest("policy")}
        for _ in ordinals
    ]
    return {
        "shard_id": name,
        "ordinals": ordinals,
        "task_identities": [_digest(f"task-{ordinal}") for ordinal in ordinals],
        "models": ["scoreq"],
        "scoring_policy_hash": _digest("policy"),
        "task_bindings": bindings,
    }


def _rows(descriptor):
    return [
        {
            "ordinal": ordinal,
            "task_identity": identity,
            "payload": {"scores": {"scoreq": ordinal + 0.5}},
        }
        for ordinal, identity in zip(
            descriptor["ordinals"], descriptor["task_identities"], strict=True,
        )
    ]


def _store_and_capability(tmp_path, identity=None):
    identity = identity or _identity()
    context = DatasetRunContext(str(tmp_path), identity.payload["dataset_id"])
    context.__enter__()
    return Step3ResumeStore(str(tmp_path), identity.payload["dataset_id"], identity), context


def test_identity_is_canonical_complete_and_source_bound():
    base = _identity()
    assert _identity().run_key == base.run_key
    for key, replacement in (
        ("step2_artifact_digest", _digest("another-artifact")),
        ("step2_rows_sha256", _digest("another-rows")),
        ("step2_row_count", 4),
        ("step2_lineage_digest", _digest("another-lineage")),
        ("step3_config", {"route": "beta"}),
        ("model_provider", {"models": ["dnsmos"]}),
        ("execution_policy", {"scheduler": "other"}),
    ):
        assert _identity(**{key: replacement}).run_key != base.run_key

    payload = Step3ResumeStore.identity_payload(
        dataset_id="alpha",
        step2_artifact_digest=_digest("artifact"),
        step2_rows_sha256=_digest("rows"),
        step2_row_count=3,
        step2_lineage_digest=_digest("lineage"),
        step3_config={}, model_provider={}, execution_policy={},
    )
    payload["source_boundary"] = {"schema": "fake", "files": {}, "sha256": _digest("fake")}
    with pytest.raises(ResumeStoreError, match="source boundary"):
        Step3ResumeIdentity.create(payload)
    boundary = source_boundary_fingerprint()
    assert boundary["schema"].endswith("v1")
    assert "core/metrics/mos_models.py" in boundary["files"]


def test_source_boundary_drift_fails_before_persistent_write(tmp_path, monkeypatch):
    identity = _identity()
    store, context = _store_and_capability(tmp_path, identity)
    try:
        mutated = source_boundary_fingerprint()
        mutated["files"]["processors/mos.py"] = _digest("changed-source")
        mutated["sha256"] = _digest("changed-boundary")
        monkeypatch.setattr(mos_resume, "source_boundary_fingerprint", lambda: mutated)
        with pytest.raises(ResumeStoreError, match="source boundary"):
            store.begin_attempt(context.persistent_write_capability())
    finally:
        context.__exit__(None, None, None)


def test_store_rejects_persistent_writes_without_active_lock(tmp_path):
    identity = _identity()
    store = Step3ResumeStore(str(tmp_path), "alpha", identity)
    with pytest.raises(TypeError, match="DatasetRunLockCapability"):
        store.begin_attempt(None)  # type: ignore[arg-type]


def test_seal_reuse_complete_coverage_and_attempt_cleanup(tmp_path):
    store, context = _store_and_capability(tmp_path)
    try:
        capability = context.persistent_write_capability()
        first_descriptor = _descriptor("first", [0, 1])
        with store.begin_attempt(capability) as attempt:
            first = attempt.seal_shard(first_descriptor, _rows(first_descriptor))
            staging_path = attempt.path
            assert first.rows_count == 2
            assert os.path.isdir(first.path)
        assert not os.path.lexists(staging_path)
        assert os.path.isdir(first.path)

        second_descriptor = _descriptor("second", [2])
        with store.begin_attempt(capability) as attempt:
            reused = attempt.seal_shard(first_descriptor, _rows(first_descriptor))
            second = attempt.seal_shard(second_descriptor, _rows(second_descriptor))
        assert reused.path == first.path
        assert second.rows_count == 1
        assert [row["ordinal"] for row in store.iter_complete_rows()] == [0, 1, 2]
        assert store.sealed_manifest_digest() == store.sealed_manifest_digest()
    finally:
        context.__exit__(None, None, None)


def test_legacy_shard_descriptor_without_task_bindings_is_rejected():
    """缺 task_bindings 的旧 shard descriptor 不再兼容读取，必须直接拒绝。"""
    legacy = {
        key: value
        for key, value in _descriptor("legacy", [0, 1]).items()
        if key != "task_bindings"
    }

    with pytest.raises(ValueError, match="descriptor fields are invalid"):
        Step3ResumeStore._validated_descriptor(legacy)


def test_complete_coverage_accepts_interleaved_lpt_shards(tmp_path):
    """LPT 调度按权重交错分配，shard ordinals 非连续块但仍必须判定为完整覆盖。"""
    store, context = _store_and_capability(
        tmp_path, _identity(step2_row_count=5),
    )
    try:
        capability = context.persistent_write_capability()
        first_descriptor = _descriptor("interleaved-a", [0, 2, 4])
        second_descriptor = _descriptor("interleaved-b", [1, 3])
        with store.begin_attempt(capability) as attempt:
            attempt.seal_shard(first_descriptor, _rows(first_descriptor))
            second = attempt.seal_shard(second_descriptor, _rows(second_descriptor))
        assert [shard.shard_id for shard in store.complete_shards()] == [
            "interleaved-a", "interleaved-b",
        ]
        assert second.rows_count == 2
    finally:
        context.__exit__(None, None, None)


def test_complete_coverage_rejects_duplicate_ordinals_across_shards(tmp_path):
    store, context = _store_and_capability(
        tmp_path, _identity(step2_row_count=4),
    )
    try:
        capability = context.persistent_write_capability()
        first_descriptor = _descriptor("dup-a", [0, 1, 2])
        second_descriptor = _descriptor("dup-b", [2])
        with store.begin_attempt(capability) as attempt:
            attempt.seal_shard(first_descriptor, _rows(first_descriptor))
            attempt.seal_shard(second_descriptor, _rows(second_descriptor))
        with pytest.raises(ResumeStoreError, match="complete global ordinal"):
            store.complete_shards()
    finally:
        context.__exit__(None, None, None)


def test_staging_is_not_complete_and_corrupt_sealed_is_fail_closed(tmp_path):
    store, context = _store_and_capability(tmp_path)
    try:
        capability = context.persistent_write_capability()
        descriptor = _descriptor("only", [0, 1, 2])
        attempt = store.begin_attempt(capability)
        Path(attempt.path, "orphan.jsonl").write_text("partial", encoding="utf-8")
        with pytest.raises(ResumeStoreError, match="complete global ordinal"):
            store.complete_shards()
        attempt.abort()
        assert not os.path.lexists(attempt.path)

        with store.begin_attempt(capability) as sealing:
            sealed = sealing.seal_shard(descriptor, _rows(descriptor))
        Path(sealed.path, "rows.jsonl").write_text("{}\n", encoding="utf-8")
        with pytest.raises(ResumeStoreError):
            store.sealed_shards()
    finally:
        context.__exit__(None, None, None)


def test_rejects_mismatched_task_identity_and_non_owned_cleanup(tmp_path):
    store, context = _store_and_capability(tmp_path)
    try:
        capability = context.persistent_write_capability()
        descriptor = _descriptor("wrong", [0])
        rows = _rows(descriptor)
        rows[0]["task_identity"] = _digest("other")
        with store.begin_attempt(capability) as attempt:
            with pytest.raises(ResumeStoreError, match="task identity"):
                attempt.seal_shard(descriptor, rows)

        foreign = Path(store.staging_dir, "attempt-foreign")
        foreign.mkdir()
        Path(foreign, "attempt.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ResumeStoreError, match="non-owned"):
            store._remove_owned_attempt(str(foreign), "unknown")
        assert foreign.exists()
    finally:
        context.__exit__(None, None, None)


def test_assembly_recovery_requires_full_artifact_and_sealed_binding(tmp_path, monkeypatch):
    store, context = _store_and_capability(tmp_path)
    try:
        capability = context.persistent_write_capability()
        descriptor = _descriptor("all", [0, 1, 2])
        with store.begin_attempt(capability) as attempt:
            attempt.seal_shard(descriptor, _rows(descriptor))
        artifact_path = tmp_path / "step3" / "alpha"
        artifact_hash = _digest("artifact-final")
        sidecar = tmp_path / (
            f"step3/alpha.{artifact_hash}.execution-attestation.json"
        )
        config_digest = _digest("config")
        lineage_digest = _digest("lineage-final")

        class _Index:
            artifact_digest = artifact_hash

        def open_verified_artifact(root, step, dataset, **kwargs):
            assert root == str(tmp_path)
            assert step == "step3" and dataset == "alpha"
            assert kwargs == {
                "expected_config_digest": config_digest,
                "expected_lineage_digest": lineage_digest,
            }
            return _Index()

        monkeypatch.setattr(mos_resume, "open_artifact_for_read", open_verified_artifact)
        assembly = {
            "artifact_hash": artifact_hash,
            "artifact_path": str(artifact_path),
            "config_digest": config_digest,
            "lineage_digest": lineage_digest,
            "scorer_identity_sha256": _digest("scorer"),
            "sidecar_path": str(sidecar),
            "proof": {"execution": "attested"},
        }
        store.record_assembly_attestation(capability, assembly)
        assert not artifact_path.exists()
        artifact_path.mkdir(parents=True)
        recovered = store.recover_assembly_attestation(assembly)
        assert recovered is not None
        assert recovered["proof"] == assembly["proof"]
        changed = {**assembly, "artifact_hash": _digest("other")}
        with pytest.raises(ResumeStoreError, match="binding mismatch"):
            store.recover_assembly_attestation(changed)
        sidecar.write_text("published", encoding="utf-8")
        assert store.recover_assembly_attestation(assembly) is None
        store.consume_assembly_attestation(capability, assembly)
        assert not os.path.exists(store._assembly_path())
    finally:
        context.__exit__(None, None, None)


def test_sealed_manifest_digest_streams_beyond_single_json_size_limit(tmp_path):
    """逐 shard manifest 散列不受 8 MiB 单 JSON 上限约束（多个大 descriptor）。"""
    from types import SimpleNamespace

    store = Step3ResumeStore(str(tmp_path), "alpha", _identity())
    padding = "x" * (4 * 1024 * 1024)
    shards = tuple(
        SimpleNamespace(
            shard_id=f"shard-{index}",
            descriptor={"index": index, "padding": padding},
            rows_sha256=_digest(f"rows-{index}"),
            rows_count=1,
            rows_bytes=1,
            attestation_sha256=_digest(f"attestation-{index}"),
            seal_sha256=_digest(f"seal-{index}"),
        )
        for index in range(3)
    )
    store.complete_shards = lambda: shards
    digest = store.sealed_manifest_digest()
    assert len(digest) == 64
    assert digest == store.sealed_manifest_digest()
