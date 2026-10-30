"""SE/AEC/TSE 公共 artifact 生命周期与任务适配器契约。"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import evaluate.artifact_lifecycle as artifact_lifecycle
from evaluate.artifact_lifecycle import (
    ARTIFACT_FILENAME,
    ARTIFACT_SCHEMA,
    CURRENT_FILENAME,
    EVALUATION_RESERVATION_TOKEN_ENV,
    RESERVATION_FILENAME,
    begin_artifact_run,
    build_artifact_envelope,
    load_committed_artifact,
    publish_artifact_commit,
    publish_artifact_envelope,
    release_artifact_reservation,
    reserve_artifact_run,
)
import evaluate.common as eval_common
import evaluate.eval_pipeline_aec as pipeline_aec
from evaluate.eval_pipeline_common import (
    effective_config_fingerprint,
    guard_explicit_eval_output,
    resolve_eval_output_name,
    save_summary,
)
import evaluate.eval_pipeline_se as pipeline_se
from evaluate.eval_pipeline_tse import _write_tse_summary
from core.inference import FileInferenceItem, PreparedOutput
from evaluate.inference_components import run_adapter_inference
from evaluate.pipeline_runner import PipelineTaskAdapter, run_pipeline


def _reservation_parent_worker(output_dir: str, ready, start, results) -> None:
    ready.put(True)
    start.wait(timeout=10)
    try:
        reservation = reserve_artifact_run(output_dir, {"owner": "parent"})
    except RuntimeError as exc:
        results.put(("rejected", str(exc)))
    else:
        results.put(("reserved", reservation.run_id))


class _EvaluateAdapter:
    def infer(self, item) -> PreparedOutput:
        def writer(path: Path) -> None:
            path.write_bytes(b"RIFF" + b"evaluate-bridge")

        return PreparedOutput(item.uid, writer)


def test_evaluate_adapter_bridge_publishes_core_manifest_and_reuses_output(tmp_path: Path):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    source = tmp_path / "input.wav"
    config.write_text("model: bridge\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    import soundfile as sf
    import numpy as np
    sf.write(source, np.zeros(16, dtype=np.float32), 16000)
    output_dir = tmp_path / "enhanced"
    output = output_dir / "utt.wav"
    item = FileInferenceItem("utt", {"noisy": source}, output)
    artifact = {
        "algorithm": "sha256",
        "files": {
            "config": {"path": str(config.resolve()), "size": config.stat().st_size,
                       "sha256": __import__("hashlib").sha256(config.read_bytes()).hexdigest()},
            "checkpoint": {"path": str(checkpoint.resolve()), "size": checkpoint.stat().st_size,
                           "sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()},
        },
        "parameters": {"bridge": True},
    }
    from core.utils.inference_artifacts import fingerprint_json
    payload = {"algorithm": "sha256", "files": artifact["files"], "parameters": artifact["parameters"]}
    artifact["sha256"] = fingerprint_json(payload)

    first = run_adapter_inference(
        adapter=_EvaluateAdapter(), config=config, checkpoint=checkpoint,
        output_dir=output_dir, items=[item], artifact_fingerprint=artifact,
    )
    assert first.results[0].status == "ok"
    second = run_adapter_inference(
        adapter=_EvaluateAdapter(), config=config, checkpoint=checkpoint,
        output_dir=output_dir, items=[item], artifact_fingerprint=artifact,
        skip_existing=True,
    )
    assert second.results[0].status == "skipped"
    assert (output_dir / "run_manifest.json").is_file()
    assert (output_dir / "inf.scp").read_text(encoding="utf-8").startswith("utt ")


@pytest.mark.parametrize("task", ["se", "aec", "tse"])
def test_adapter_preserves_task_payload_and_normalizes_provenance(
    tmp_path: Path, task: str
):
    model = tmp_path / "model.tar"
    model.write_bytes(b"model")
    metric_coverage = {"task_metric": {"computed": 2, "eligible": 3}}
    summary = {
        "status": "partial",
        "inference": {"expected": 3, "ok": 2, "failed": 1},
        "metric_coverage": metric_coverage,
        "task_payload": {"native": task},
    }
    adapter = PipelineTaskAdapter(
        task=task,
        pipeline=f"eval_pipeline_{task}",
        step="all",
        output_dir=tmp_path / task,
        provenance={"model": model, "entry": f"eval_pipeline_{task}"},
    )

    envelope = adapter.build_envelope(summary)

    assert envelope["schema"] == ARTIFACT_SCHEMA
    assert envelope["task"] == task
    assert envelope["status"] == "partial"
    assert envelope["provenance"]["model"] == str(model.resolve())
    assert envelope["provenance"]["model_fingerprint"]["sha256"]
    assert envelope["inference_coverage"] == summary["inference"]
    assert envelope["metric_coverage"] == metric_coverage
    assert envelope["summary"]["task_payload"] == {"native": task}


def test_se_summary_dual_writes_legacy_and_common_artifacts(tmp_path: Path):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    summary = {
        "pipeline": "eval_pipeline_se",
        "model_type": "se",
        "step": "evaluate",
        "config": str(config),
        "checkpoint": str(checkpoint),
        "status": "done",
        "coverage": {"expected": 1, "produced": 1, "failed": 0, "evaluated": 1},
        "datasets": {
            "voicebank": {
                "status": "done",
                "metric_coverage": {"PESQ": {"eligible": 1, "computed": 1}},
            }
        },
    }

    legacy_path = save_summary(summary, tmp_path / "se")
    envelope = json.loads(
        (tmp_path / "se" / ARTIFACT_FILENAME).read_text(encoding="utf-8")
    )

    assert legacy_path.name == "summary.json"
    persisted = json.loads(legacy_path.read_text(encoding="utf-8"))
    assert {key: value for key, value in persisted.items() if key != "run_id"} == summary
    assert persisted["run_id"] == envelope["run_id"]
    assert envelope["summary"]["run_id"] == envelope["run_id"]
    assert envelope["task"] == "se"
    assert envelope["metric_coverage"] == {
        "voicebank": {"PESQ": {"eligible": 1, "computed": 1}}
    }


def test_tse_summary_keeps_sidecar_and_publishes_error_envelope(tmp_path: Path):
    model = tmp_path / "model.tar"
    model.write_bytes(b"model")
    payload = {
        "pipeline": "eval_pipeline_tse",
        "model": str(model),
        "metrics": [],
        "summary": {
            "status": "error",
            "reason": "inference_failed",
            "coverage": {},
            "inference": {"expected": 1, "ok": 0, "failed": 1},
        },
    }

    sidecar = _write_tse_summary(
        tmp_path / "tse",
        payload,
        model=model,
        step="inference",
    )
    envelope = json.loads(
        (tmp_path / "tse" / ARTIFACT_FILENAME).read_text(encoding="utf-8")
    )

    assert sidecar.name == "tse_eval_summary.json"
    persisted = json.loads(sidecar.read_text(encoding="utf-8"))
    assert persisted["run_id"] == envelope["run_id"]
    assert persisted["summary"]["run_id"] == envelope["run_id"]
    assert envelope["task"] == "tse"
    assert envelope["status"] == "error"
    assert envelope["error"] == "inference_failed"
    assert envelope["inference_coverage"] == payload["summary"]["inference"]


def test_run_pipeline_publishes_error_before_reraising(tmp_path: Path):
    adapter = PipelineTaskAdapter(
        task="se",
        pipeline="eval_pipeline_se",
        step="all",
        output_dir=tmp_path,
    )

    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline(
            adapter,
            prepare=lambda: "context",
            run=lambda _context: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    envelope = json.loads(
        (tmp_path / ARTIFACT_FILENAME).read_text(encoding="utf-8")
    )
    assert envelope["status"] == "error"
    assert envelope["error"] == "RuntimeError: boom"


def test_run_pipeline_commits_summary_and_envelope_as_one_consumable_run(
    tmp_path: Path,
):
    adapter = PipelineTaskAdapter(
        task="se",
        pipeline="eval_pipeline_se",
        step="all",
        output_dir=tmp_path,
    )

    result = run_pipeline(
        adapter,
        prepare=lambda: "context",
        run=lambda _context: {"status": "done"},
    )

    current = json.loads((tmp_path / CURRENT_FILENAME).read_text(encoding="utf-8"))
    envelope = load_committed_artifact(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert current["status"] == "committed"
    assert current["run_id"] == result["run_id"]
    assert summary["run_id"] == envelope["run_id"] == result["run_id"]
    assert current["summary"]["filename"] == "summary.json"
    assert current["envelope"]["filename"] == ARTIFACT_FILENAME
    assert len(current["summary"]["sha256"]) == 64
    assert len(current["envelope"]["sha256"]) == 64

    summary["status"] = "error"
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="summary digest mismatch"):
        load_committed_artifact(tmp_path)


@pytest.mark.parametrize(
    "summary_filename",
    [
        "../outside.json",
        "nested/summary.json",
        "nested\\summary.json",
        CURRENT_FILENAME,
        ARTIFACT_FILENAME,
    ],
)
def test_run_pipeline_rejects_unsafe_summary_filename_before_begin(
    tmp_path: Path, summary_filename: str
):
    adapter = PipelineTaskAdapter(
        task="se",
        pipeline="eval_pipeline_se",
        step="all",
        output_dir=tmp_path / "output",
        summary_filename=summary_filename,
    )

    with pytest.raises(ValueError, match="summary filename"):
        run_pipeline(
            adapter,
            prepare=lambda: pytest.fail("invalid filename must stop before prepare"),
            run=lambda _context: {"status": "done"},
        )

    assert not (tmp_path / "output" / CURRENT_FILENAME).exists()
    assert not (tmp_path / "outside.json").exists()


@pytest.mark.parametrize("requested_run_id", ["", 0, False])
def test_run_pipeline_rejects_invalid_requested_run_id_before_begin(
    tmp_path: Path, requested_run_id
):
    adapter = PipelineTaskAdapter(
        task="se",
        pipeline="eval_pipeline_se",
        step="all",
        output_dir=tmp_path / "output",
    )

    with pytest.raises(ValueError, match="requested_run_id"):
        run_pipeline(
            adapter,
            prepare=lambda: pytest.fail("invalid run id must stop before prepare"),
            run=lambda _context: {"status": "done"},
            requested_run_id=requested_run_id,
        )

    assert not (tmp_path / "output" / CURRENT_FILENAME).exists()


@pytest.mark.parametrize("drift", ["summary", "provenance"])
def test_artifact_commit_rejects_summary_envelope_drift_before_current(
    tmp_path: Path, drift: str
):
    run_id = "run-1"
    identity = {
        "schema": "effective-merged-config-v1",
        "algorithm": "sha256-json-v1",
        "sha256": "a" * 64,
    }
    provenance = {"resolved_config_fingerprint": identity}
    summary = {
        "run_id": run_id,
        "status": "done",
        "effective_config_identity": identity,
        "provenance": provenance,
    }
    begin_artifact_run(tmp_path, run_id)
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    embedded = dict(summary)
    envelope_provenance = provenance
    if drift == "summary":
        embedded = {**embedded, "status": "partial"}
    else:
        envelope_provenance = {
            "resolved_config_fingerprint": {**identity, "sha256": "b" * 64}
        }
    publish_artifact_envelope(
        tmp_path,
        build_artifact_envelope(
            task="se",
            pipeline="eval_pipeline_se",
            step="all",
            status="done",
            provenance=envelope_provenance,
            inference_coverage=None,
            metric_coverage=None,
            summary=embedded,
            run_id=run_id,
        ),
    )

    with pytest.raises(RuntimeError, match="summary.*envelope|provenance"):
        publish_artifact_commit(
            tmp_path, run_id=run_id, summary_filename="summary.json"
        )

    current = json.loads((tmp_path / CURRENT_FILENAME).read_text(encoding="utf-8"))
    assert current["status"] == "in_progress"


def test_run_pipeline_rejects_on_complete_provenance_drift_without_overwriting_commit(
    tmp_path: Path
):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.pt"
    config.write_text("model: {type: Test}\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    identity = effective_config_fingerprint(config)
    adapter = PipelineTaskAdapter(
        task="se",
        pipeline="eval_pipeline_se",
        step="all",
        output_dir=tmp_path / "output",
        provenance={
            "config": config,
            "checkpoint": checkpoint,
            "resolved_config_fingerprint": identity,
        },
    )
    first = run_pipeline(
        adapter,
        prepare=lambda: None,
        run=lambda _context: {
            "status": "done",
            "effective_config_identity": identity,
            "provenance": adapter._provenance(),
        },
    )
    output = Path(adapter.output_dir)
    current_before = (output / CURRENT_FILENAME).read_bytes()
    summary_before = (output / "summary.json").read_bytes()
    envelope_before = (output / ARTIFACT_FILENAME).read_bytes()

    with pytest.raises(RuntimeError, match="provenance"):
        run_pipeline(
            adapter,
            prepare=lambda: None,
            run=lambda _context: {
                "status": "done",
                "effective_config_identity": identity,
                "provenance": adapter._provenance(),
            },
            on_complete=lambda _summary: config.write_text(
                "model: {type: Replaced}\n", encoding="utf-8"
            ),
        )

    assert (output / CURRENT_FILENAME).read_bytes() == current_before
    assert (output / "summary.json").read_bytes() == summary_before
    assert (output / ARTIFACT_FILENAME).read_bytes() == envelope_before
    assert load_committed_artifact(output)["run_id"] == first["run_id"]


def _explicit_adapter(
    tmp_path: Path,
    *,
    task: str = "se",
    config: Path | None = None,
    checkpoint: Path | None = None,
    output_name: str = "demo@source-digest",
) -> PipelineTaskAdapter:
    config = config or (tmp_path / "config.yaml")
    checkpoint = checkpoint or (tmp_path / "model.model_only.pt")
    config.write_text("model: fixture\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    return PipelineTaskAdapter(
        task=task,
        pipeline=f"eval_pipeline_{task}",
        step="all",
        output_dir=tmp_path / output_name,
        provenance={
            "config": config,
            "checkpoint": checkpoint,
            "parameters": {"output_name": output_name},
        },
    )


def _commit_adapter(adapter: PipelineTaskAdapter) -> None:
    output_name = adapter.provenance["parameters"]["output_name"]
    run_pipeline(
        adapter,
        prepare=lambda: None,
        run=lambda _context: {"status": "done", "output_name": output_name},
    )


def test_run_pipeline_executes_preflight_under_lock_before_begin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    adapter = _explicit_adapter(tmp_path)
    events: list[str] = []

    @contextmanager
    def fake_lock(output: Path):
        events.append("lock-enter")
        output.mkdir(parents=True, exist_ok=True)
        try:
            yield output
        finally:
            events.append("lock-exit")

    monkeypatch.setattr("evaluate.artifact_lifecycle.inference_output_lock", fake_lock)

    def preflight() -> None:
        events.append("preflight")
        assert not (adapter.output_dir / CURRENT_FILENAME).exists()

    def run(_context: object) -> dict:
        events.append("run")
        current = json.loads(
            (adapter.output_dir / CURRENT_FILENAME).read_text(encoding="utf-8")
        )
        assert current["status"] == "in_progress"
        return {"status": "done"}

    def precommit(_summary: dict) -> None:
        events.append("precommit")

    run_pipeline(
        adapter,
        prepare=lambda: None,
        run=run,
        preflight=preflight,
        precommit=precommit,
    )

    assert events == [
        "lock-enter",
        "preflight",
        "lock-exit",
        "run",
        "lock-enter",
        "precommit",
        "lock-exit",
    ]


def test_reserved_child_claims_bound_run_before_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    adapter = _explicit_adapter(tmp_path)
    reservation = reserve_artifact_run(adapter.output_dir, adapter.ownership_binding())
    monkeypatch.setenv(EVALUATION_RESERVATION_TOKEN_ENV, reservation.token)

    def prepare() -> None:
        current = json.loads(
            (adapter.output_dir / CURRENT_FILENAME).read_text(encoding="utf-8")
        )
        assert current["run_id"] == reservation.run_id
        assert current["status"] == "in_progress"
        assert not (adapter.output_dir / RESERVATION_FILENAME).exists()

    result = run_pipeline(adapter, prepare=prepare, run=lambda _context: {"status": "done"})

    assert result["run_id"] == reservation.run_id


@pytest.mark.parametrize("drift", ["token", "binding"])
def test_reserved_child_rejects_claim_drift_without_consuming_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
):
    adapter = _explicit_adapter(tmp_path)
    reservation = reserve_artifact_run(adapter.output_dir, adapter.ownership_binding())
    claimed_adapter = adapter
    token = reservation.token
    if drift == "token":
        token = "wrong-token"
    else:
        claimed_adapter = PipelineTaskAdapter(
            **{**adapter.__dict__, "step": "evaluate"}
        )
    monkeypatch.setenv(EVALUATION_RESERVATION_TOKEN_ENV, token)

    with pytest.raises(RuntimeError, match=f"reservation {drift}"):
        run_pipeline(
            claimed_adapter,
            prepare=lambda: pytest.fail("invalid claim must stop before prepare"),
            run=lambda _context: {"status": "done"},
        )

    assert (adapter.output_dir / RESERVATION_FILENAME).exists()
    assert not (adapter.output_dir / CURRENT_FILENAME).exists()


def test_active_reservation_excludes_other_parents_and_legacy_direct_run(tmp_path: Path):
    adapter = _explicit_adapter(tmp_path)
    first = reserve_artifact_run(adapter.output_dir, adapter.ownership_binding())

    with pytest.raises(RuntimeError, match="active reservation"):
        reserve_artifact_run(adapter.output_dir, adapter.ownership_binding())
    with pytest.raises(RuntimeError, match="active reservation"):
        run_pipeline(
            adapter,
            prepare=lambda: None,
            run=lambda _context: {"status": "done"},
        )

    assert release_artifact_reservation(first) is True


@pytest.mark.subprocess
def test_concurrent_parents_create_exactly_one_reservation(tmp_path: Path):
    output_dir = tmp_path / "shared-output"
    context = multiprocessing.get_context("fork")
    ready = context.Queue()
    start = context.Event()
    results = context.Queue()
    parents = [
        context.Process(
            target=_reservation_parent_worker,
            args=(str(output_dir), ready, start, results),
        )
        for _ in range(2)
    ]
    for parent in parents:
        parent.start()
    assert ready.get(timeout=5) is True
    assert ready.get(timeout=5) is True
    start.set()
    outcomes = [results.get(timeout=5), results.get(timeout=5)]
    for parent in parents:
        parent.join(timeout=5)
        assert parent.exitcode == 0

    assert sorted(outcome for outcome, _detail in outcomes) == ["rejected", "reserved"]
    rejected = next(detail for outcome, detail in outcomes if outcome == "rejected")
    # 锁改为 LOCK_NB fail-fast 后，竞态输家可能直接拿到「已有活跃 writer」报错。
    assert "active reservation" in rejected or "已有活跃 writer" in rejected
    assert (output_dir / RESERVATION_FILENAME).is_file()
    assert not (output_dir / CURRENT_FILENAME).exists()


def test_reservation_persists_verifiable_owner_and_does_not_ttl_steal_live_owner(
    tmp_path: Path,
):
    output = tmp_path / "output"
    first = reserve_artifact_run(output, {"owner": "parent"})
    payload = json.loads((output / RESERVATION_FILENAME).read_text(encoding="utf-8"))

    assert payload["owner"] == first.owner
    assert payload["owner"]["pid"] > 0
    assert payload["owner"]["process_start_ticks"] > 0
    assert payload["owner"]["hostname"]
    assert payload["owner"]["boot_id"]
    with pytest.raises(RuntimeError, match="active reservation"):
        reserve_artifact_run(output, {"owner": "other"})


def test_reservation_reclaims_only_proven_dead_local_owner(tmp_path: Path):
    output = tmp_path / "output"
    first = reserve_artifact_run(output, {"owner": "parent"})
    payload_path = output / RESERVATION_FILENAME
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["owner"]["process_start_ticks"] += 1
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    second = reserve_artifact_run(output, {"owner": "replacement"})

    assert second.run_id != first.run_id
    assert release_artifact_reservation(first) is False
    assert release_artifact_reservation(second) is True


def test_reservation_preserves_readable_previous_commit(tmp_path: Path):
    adapter = _explicit_adapter(tmp_path)
    _commit_adapter(adapter)
    committed_before = load_committed_artifact(adapter.output_dir)
    current_before = (adapter.output_dir / CURRENT_FILENAME).read_bytes()

    reservation = reserve_artifact_run(
        adapter.output_dir,
        adapter.ownership_binding(),
        preflight=lambda: guard_explicit_eval_output(adapter),
    )

    assert (adapter.output_dir / CURRENT_FILENAME).read_bytes() == current_before
    assert load_committed_artifact(adapter.output_dir) == committed_before
    assert release_artifact_reservation(reservation) is True


def test_parent_release_does_not_roll_back_claimed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    adapter = _explicit_adapter(tmp_path)
    reservation = reserve_artifact_run(adapter.output_dir, adapter.ownership_binding())
    monkeypatch.setenv(EVALUATION_RESERVATION_TOKEN_ENV, reservation.token)
    run_pipeline(adapter, prepare=lambda: None, run=lambda _context: {"status": "done"})
    current_before = (adapter.output_dir / CURRENT_FILENAME).read_bytes()

    assert release_artifact_reservation(reservation) is False
    assert (adapter.output_dir / CURRENT_FILENAME).read_bytes() == current_before


def test_claimed_parent_release_does_not_disturb_newer_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    adapter = _explicit_adapter(tmp_path)
    first = reserve_artifact_run(adapter.output_dir, adapter.ownership_binding())
    monkeypatch.setenv(EVALUATION_RESERVATION_TOKEN_ENV, first.token)
    run_pipeline(adapter, prepare=lambda: None, run=lambda _context: {"status": "done"})
    second = reserve_artifact_run(adapter.output_dir, adapter.ownership_binding())

    assert release_artifact_reservation(first) is False
    assert (adapter.output_dir / RESERVATION_FILENAME).is_file()
    assert release_artifact_reservation(second) is True


@pytest.mark.parametrize("task", ["aec", "se"])
def test_explicit_output_guard_allows_same_committed_pair_and_rejects_foreign_pair(
    tmp_path: Path, task: str
):
    adapter = _explicit_adapter(tmp_path, task=task)
    _commit_adapter(adapter)

    guard_explicit_eval_output(adapter)
    current_before = (adapter.output_dir / CURRENT_FILENAME).read_bytes()

    foreign_checkpoint = tmp_path / f"foreign-{task}.model_only.pt"
    foreign_checkpoint.write_bytes(b"foreign")
    foreign = _explicit_adapter(
        tmp_path,
        task=task,
        checkpoint=foreign_checkpoint,
    )
    with pytest.raises(ValueError, match="checkpoint provenance"):
        run_pipeline(
            foreign,
            prepare=lambda: None,
            run=lambda _context: {"status": "done"},
            preflight=lambda: guard_explicit_eval_output(foreign),
        )
    assert (adapter.output_dir / CURRENT_FILENAME).read_bytes() == current_before


def test_explicit_output_guard_rejects_changed_effective_config_without_replacing_current(
    tmp_path: Path,
):
    base = tmp_path / "base.yaml"
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.model_only.pt"
    base.write_text("stft: {n_fft: 320}\n", encoding="utf-8")
    config.write_text(
        "base_config: base.yaml\nmodel: {type: DeepVQE}\n", encoding="utf-8"
    )
    checkpoint.write_bytes(b"checkpoint")
    output_name = "demo@merged-config"
    adapter = PipelineTaskAdapter(
        task="se",
        pipeline="eval_pipeline_se",
        step="all",
        output_dir=tmp_path / output_name,
        provenance={
            "config": config,
            "checkpoint": checkpoint,
            "resolved_config_fingerprint": effective_config_fingerprint(config),
            "parameters": {"output_name": output_name},
        },
    )
    _commit_adapter(adapter)
    current_before = (adapter.output_dir / CURRENT_FILENAME).read_bytes()

    base.write_text("stft: {n_fft: 512}\n", encoding="utf-8")
    changed = PipelineTaskAdapter(
        task="se",
        pipeline="eval_pipeline_se",
        step="all",
        output_dir=adapter.output_dir,
        provenance={
            **adapter.provenance,
            "resolved_config_fingerprint": effective_config_fingerprint(config),
        },
    )

    with pytest.raises(ValueError, match="resolved_config_fingerprint provenance"):
        guard_explicit_eval_output(changed)
    assert (adapter.output_dir / CURRENT_FILENAME).read_bytes() == current_before



@pytest.mark.parametrize("task", ["aec", "se"])
def test_explicit_output_guard_rejects_changed_asset_content_without_replacing_current(
    tmp_path: Path, task: str
):
    adapter = _explicit_adapter(tmp_path, task=task)
    _commit_adapter(adapter)
    current_before = (adapter.output_dir / CURRENT_FILENAME).read_bytes()

    config = Path(adapter.provenance["config"])
    config.write_text("model: changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="config_fingerprint provenance"):
        guard_explicit_eval_output(adapter)

    assert (adapter.output_dir / CURRENT_FILENAME).read_bytes() == current_before


@pytest.mark.parametrize("task", ["aec", "se"])
def test_explicit_output_same_pair_can_rerun_through_preflight(
    tmp_path: Path, task: str
):
    adapter = _explicit_adapter(tmp_path, task=task)
    _commit_adapter(adapter)
    first_run_id = json.loads(
        (adapter.output_dir / CURRENT_FILENAME).read_text(encoding="utf-8")
    )["run_id"]
    output_name = adapter.provenance["parameters"]["output_name"]

    result = run_pipeline(
        adapter,
        prepare=lambda: None,
        run=lambda _context: {"status": "done", "output_name": output_name},
        preflight=lambda: guard_explicit_eval_output(adapter),
    )

    current = json.loads(
        (adapter.output_dir / CURRENT_FILENAME).read_text(encoding="utf-8")
    )
    assert result["run_id"] != first_run_id
    assert current["status"] == "committed"
    assert current["run_id"] == result["run_id"]


@pytest.mark.parametrize("task", ["aec", "se"])
def test_explicit_output_guard_rejects_in_progress_without_changing_current(
    tmp_path: Path, task: str
):
    adapter = _explicit_adapter(tmp_path, task=task)
    adapter.output_dir.mkdir(parents=True)
    begin_artifact_run(adapter.output_dir, "active-run")
    current_before = (adapter.output_dir / CURRENT_FILENAME).read_bytes()

    with pytest.raises(ValueError, match="in-progress"):
        guard_explicit_eval_output(adapter)

    assert (adapter.output_dir / CURRENT_FILENAME).read_bytes() == current_before


def test_explicit_output_guard_rejects_output_symlink(tmp_path: Path):
    adapter = _explicit_adapter(tmp_path)
    target = tmp_path / "symlink-target"
    target.mkdir()
    Path(adapter.output_dir).symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="不允许符号链接"):
        guard_explicit_eval_output(adapter)


def test_explicit_output_guard_accepts_parent_log_only_and_rejects_orphan_summary(
    tmp_path: Path,
):
    adapter = _explicit_adapter(tmp_path)
    adapter.output_dir.mkdir(parents=True)
    (adapter.output_dir / "eval_pipeline.log").write_text("parent log\n", encoding="utf-8")
    guard_explicit_eval_output(adapter)

    (adapter.output_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lifecycle artifact 不完整"):
        guard_explicit_eval_output(adapter)


def test_run_pipeline_publish_failure_restores_old_committed_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    adapter = PipelineTaskAdapter(
        task="se",
        pipeline="eval_pipeline_se",
        step="all",
        output_dir=tmp_path,
    )
    first = run_pipeline(
        adapter,
        prepare=lambda: "context",
        run=lambda _context: {"status": "done"},
    )
    old_envelope = json.loads(
        (tmp_path / ARTIFACT_FILENAME).read_text(encoding="utf-8")
    )
    persisted: list[dict] = []

    monkeypatch.setattr(
        "evaluate.pipeline_runner.publish_artifact_envelope",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("publish boom")),
    )
    result = run_pipeline(
        adapter,
        prepare=lambda: "context",
        run=lambda _context: {"status": "done"},
        on_complete=lambda summary: persisted.append(dict(summary)),
        reraise=False,
    )

    current = json.loads((tmp_path / CURRENT_FILENAME).read_text(encoding="utf-8"))
    assert first["run_id"] != result["run_id"]
    assert [item["run_id"] for item in persisted] == [result["run_id"]]
    assert result["status"] == "error"
    assert result["reason"] == "RuntimeError: publish boom"
    assert current["status"] == "committed"
    assert current["run_id"] == first["run_id"]
    assert json.loads(
        (tmp_path / ARTIFACT_FILENAME).read_text(encoding="utf-8")
    ) == old_envelope
    assert load_committed_artifact(tmp_path)["run_id"] == first["run_id"]


def test_failed_rerun_restores_every_manifested_metric_artifact(
    tmp_path: Path,
):
    adapter = PipelineTaskAdapter(
        task="se",
        pipeline="eval_pipeline_se",
        step="all",
        output_dir=tmp_path,
    )
    metric = tmp_path / "dataset" / "RESULTS.txt"
    sidecar = tmp_path / "dataset" / "per_utterance.json"
    metric.parent.mkdir()
    metric.write_text("PESQ: 2.0\n", encoding="utf-8")
    sidecar.write_text('{"utt": 2.0}\n', encoding="utf-8")

    def publish_manifest(summary: dict) -> None:
        summary["datasets"] = {
            "dataset": {
                "status": "done",
                "results_file": str(metric),
            }
        }
        eval_common.publish_result_manifest(tmp_path, summary)

    first = run_pipeline(
        adapter,
        prepare=lambda: None,
        run=lambda _context: {"status": "done"},
        on_complete=publish_manifest,
        require_result_manifest=True,
    )
    old_files = {
        path: path.read_bytes()
        for path in (
            tmp_path / CURRENT_FILENAME,
            tmp_path / "summary.json",
            tmp_path / ARTIFACT_FILENAME,
            tmp_path / "result_manifest.json",
            metric,
            sidecar,
        )
    }

    def corrupt_then_fail(_summary: dict) -> None:
        metric.write_text("PESQ: 9.9\n", encoding="utf-8")
        sidecar.write_text('{"utt": 9.9}\n', encoding="utf-8")
        raise RuntimeError("metric publish failed")

    result = run_pipeline(
        adapter,
        prepare=lambda: None,
        run=lambda _context: {"status": "done"},
        on_complete=corrupt_then_fail,
        reraise=False,
    )

    assert result["status"] == "error"
    assert load_committed_artifact(tmp_path)["run_id"] == first["run_id"]
    assert {path: path.read_bytes() for path in old_files} == old_files


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside.txt",
        "/tmp/outside.txt",
        "nested\\file.txt",
        "nested/./file.txt",
        "nested//file.txt",
        "nested/file.txt/",
        ".inference.lock",
    ],
)
def test_current_commit_rejects_unsafe_result_manifest_paths(
    tmp_path: Path, unsafe_path: str
):
    adapter = PipelineTaskAdapter(
        task="se",
        pipeline="eval_pipeline_se",
        step="all",
        output_dir=tmp_path,
    )

    def publish_unsafe_manifest(summary: dict) -> None:
        manifest_path = tmp_path / "result_manifest.json"
        manifest = {
            "schema": "evaluation-results-v1",
            "run_id": summary["run_id"],
            "files": {
                unsafe_path: {
                    "size": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                }
            },
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        content = manifest_path.read_bytes()
        summary["result_manifest"] = {
            "filename": manifest_path.name,
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    result = run_pipeline(
        adapter,
        prepare=lambda: None,
        run=lambda _context: {"status": "done"},
        on_complete=publish_unsafe_manifest,
        require_result_manifest=True,
        reraise=False,
    )

    current = json.loads((tmp_path / CURRENT_FILENAME).read_text(encoding="utf-8"))
    assert result["status"] == "error"
    assert current["status"] == "aborted"


def test_current_commit_rejects_symlinked_metric_artifact(tmp_path: Path):
    adapter = PipelineTaskAdapter(
        task="se",
        pipeline="eval_pipeline_se",
        step="all",
        output_dir=tmp_path,
    )
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside\n", encoding="utf-8")

    def publish_symlink_manifest(summary: dict) -> None:
        metric = tmp_path / "dataset" / "RESULTS.txt"
        metric.parent.mkdir()
        metric.symlink_to(outside)
        content = outside.read_bytes()
        manifest_path = tmp_path / "result_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": "evaluation-results-v1",
                    "run_id": summary["run_id"],
                    "files": {
                        "dataset/RESULTS.txt": {
                            "size": len(content),
                            "sha256": hashlib.sha256(content).hexdigest(),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        manifest_content = manifest_path.read_bytes()
        summary["result_manifest"] = {
            "filename": manifest_path.name,
            "size": len(manifest_content),
            "sha256": hashlib.sha256(manifest_content).hexdigest(),
        }

    result = run_pipeline(
        adapter,
        prepare=lambda: None,
        run=lambda _context: {"status": "done"},
        on_complete=publish_symlink_manifest,
        require_result_manifest=True,
        reraise=False,
    )

    assert result["status"] == "error"
    assert json.loads((tmp_path / CURRENT_FILENAME).read_text(encoding="utf-8"))[
        "status"
    ] == "aborted"


def test_first_publish_failure_transitions_current_to_aborted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    adapter = PipelineTaskAdapter(
        task="se",
        pipeline="eval_pipeline_se",
        step="all",
        output_dir=tmp_path,
    )
    monkeypatch.setattr(
        "evaluate.pipeline_runner.publish_artifact_envelope",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("publish boom")),
    )

    result = run_pipeline(
        adapter,
        prepare=lambda: None,
        run=lambda _context: {"status": "done"},
        reraise=False,
    )

    current = json.loads((tmp_path / CURRENT_FILENAME).read_text(encoding="utf-8"))
    assert result["status"] == "error"
    assert current == {
        "schema": "evaluation-current-v1",
        "status": "aborted",
        "run_id": result["run_id"],
        "reason": "publish_failed",
    }


def test_failure_summary_exception_still_aborts_active_run(tmp_path: Path):
    adapter = PipelineTaskAdapter(
        task="se",
        pipeline="eval_pipeline_se",
        step="all",
        output_dir=tmp_path,
    )

    with pytest.raises(RuntimeError, match="handler boom"):
        run_pipeline(
            adapter,
            prepare=lambda: (_ for _ in ()).throw(RuntimeError("prepare boom")),
            run=lambda _context: {"status": "done"},
            on_failure=lambda *_args: (_ for _ in ()).throw(RuntimeError("handler boom")),
            reraise=False,
        )

    current = json.loads((tmp_path / CURRENT_FILENAME).read_text(encoding="utf-8"))
    assert current["status"] == "aborted"
    assert current["reason"] == "prepare_failure_handler_failed"


def test_stale_failure_cannot_roll_back_newer_commit(tmp_path: Path):
    adapter = PipelineTaskAdapter(
        task="se",
        pipeline="eval_pipeline_se",
        step="all",
        output_dir=tmp_path,
    )
    first = run_pipeline(
        adapter,
        prepare=lambda: None,
        run=lambda _context: {"status": "done", "version": 1},
    )
    stale_snapshot = artifact_lifecycle.capture_committed_artifact(tmp_path)
    second = run_pipeline(
        adapter,
        prepare=lambda: None,
        run=lambda _context: {"status": "done", "version": 2},
    )
    current_before = (tmp_path / CURRENT_FILENAME).read_bytes()

    assert artifact_lifecycle.abort_artifact_run(
        tmp_path,
        run_id=first["run_id"],
        previous_commit=stale_snapshot,
        reason="late failure",
    ) is False
    assert (tmp_path / CURRENT_FILENAME).read_bytes() == current_before
    assert load_committed_artifact(tmp_path)["run_id"] == second["run_id"]


def test_run_pipeline_execution_failure_restores_old_committed_artifact(
    tmp_path: Path,
):
    adapter = PipelineTaskAdapter(
        task="se",
        pipeline="eval_pipeline_se",
        step="all",
        output_dir=tmp_path,
    )
    first = run_pipeline(
        adapter,
        prepare=lambda: "context",
        run=lambda _context: {"status": "done"},
    )
    old_files = {
        filename: (tmp_path / filename).read_bytes()
        for filename in (CURRENT_FILENAME, "summary.json", ARTIFACT_FILENAME)
    }

    result = run_pipeline(
        adapter,
        prepare=lambda: "context",
        run=lambda _context: (_ for _ in ()).throw(RuntimeError("run boom")),
        reraise=False,
    )

    assert result["status"] == "error"
    assert result["reason"] == "RuntimeError: run boom"
    assert result["run_id"] != first["run_id"]
    assert {
        filename: (tmp_path / filename).read_bytes()
        for filename in old_files
    } == old_files
    assert load_committed_artifact(tmp_path)["run_id"] == first["run_id"]


def test_se_main_uses_runner_and_publishes_partial_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    assets = tmp_path / "assets"
    assets.mkdir()
    config = assets / "config.yaml"
    checkpoint = assets / "model.tar"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    output_name = "demo@model_0060-deadbeefcafe"
    args = SimpleNamespace(
        exp_dir=None,
        config=str(config),
        checkpoint=str(checkpoint),
        output_name=output_name,
        datasets="voicebank",
        include_aec_sim=False,
        include_mc_sim=False,
        step="all",
        device="cpu",
        inference_devices=None,
        urgent_noisy_dir="unused",
        urgent_clean_dir="unused",
        voicebank_noisy_scp="unused",
        voicebank_clean_scp="unused",
        dns_challenge_noisy_dir="unused",
    )
    runner_calls: list[dict] = []

    monkeypatch.setattr(pipeline_se, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline_se, "parse_args", lambda: args)
    monkeypatch.setattr(pipeline_se, "detect_model_type", lambda _config, **_kwargs: "se")
    monkeypatch.setattr(
        pipeline_se, "prepare_testset_scps", lambda **_kwargs: {"voicebank": {}}
    )
    monkeypatch.setattr(
        pipeline_se,
        "process_voicebank",
        lambda _args, _config, _checkpoint, _root, datasets, _scps: datasets.update(
            {
                "voicebank": {
                    "status": "partial",
                    "coverage": {
                        "expected": 2,
                        "produced": 1,
                        "failed": 1,
                        "evaluated": 1,
                    },
                    "metric_coverage": {
                        "PESQ": {"eligible": 1, "computed": 1}
                    },
                }
            }
        ),
    )
    monkeypatch.setattr(
        pipeline_se,
        "run_pipeline",
        lambda *runner_args, **runner_kwargs: (
            runner_calls.append(dict(runner_kwargs)),
            run_pipeline(*runner_args, **runner_kwargs),
        )[1],
    )
    monkeypatch.setattr(pipeline_se, "print_summary_table", lambda _summary: None)

    assert pipeline_se.main() == 0
    assert len(runner_calls) == 1
    assert callable(runner_calls[0]["preflight"])
    assert callable(runner_calls[0]["precommit"])
    assert runner_calls[0]["require_result_manifest"] is True
    output = tmp_path / "exp/eval/se" / output_name
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    envelope = json.loads((output / ARTIFACT_FILENAME).read_text(encoding="utf-8"))
    assert summary["status"] == "partial"
    assert summary["output_name"] == output_name
    assert envelope["status"] == "partial"
    assert summary["provenance"] == envelope["provenance"]
    assert summary["provenance"]["parameters"]["output_name"] == output_name
    assert envelope["inference_coverage"] == summary["coverage"]
    assert envelope["metric_coverage"] == {
        "voicebank": {"PESQ": {"eligible": 1, "computed": 1}}
    }


def test_aec_main_uses_runner_and_publishes_prepare_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    assets = tmp_path / "assets"
    assets.mkdir()
    config = assets / "config.yaml"
    checkpoint = assets / "model.tar"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    output_name = "demo@model_0060-deadbeefcafe"
    args = SimpleNamespace(
        exp_dir=None,
        config=str(config),
        checkpoint=str(checkpoint),
        output_name=output_name,
        datasets="voicebank",
        aec_subsets="all",
        step="evaluate",
        device="cpu",
        inference_devices=None,
        aec_testset_root="unused",
        urgent_noisy_dir="unused",
        urgent_clean_dir="unused",
        voicebank_noisy_scp="unused",
        voicebank_clean_scp="unused",
        dns_challenge_noisy_dir="unused",
    )
    runner_calls: list[dict] = []

    monkeypatch.setattr(pipeline_aec, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline_aec, "parse_args", lambda: args)
    monkeypatch.setattr(pipeline_aec, "detect_model_type", lambda _config, **_kwargs: "aec")
    monkeypatch.setattr(pipeline_aec, "detect_aec_subtype", lambda _config, **_kwargs: "joint")
    monkeypatch.setattr(
        pipeline_aec,
        "prepare_testset_scps",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("missing testset")),
    )
    monkeypatch.setattr(
        pipeline_aec,
        "run_pipeline",
        lambda *runner_args, **runner_kwargs: (
            runner_calls.append(dict(runner_kwargs)),
            run_pipeline(*runner_args, **runner_kwargs),
        )[1],
    )
    monkeypatch.setattr(pipeline_aec, "print_summary_table", lambda _summary: None)

    assert pipeline_aec.main() == 1
    assert len(runner_calls) == 1
    assert callable(runner_calls[0]["preflight"])
    assert callable(runner_calls[0]["precommit"])
    assert runner_calls[0]["require_result_manifest"] is True
    output = tmp_path / "exp/eval/aec" / output_name
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    envelope = json.loads((output / ARTIFACT_FILENAME).read_text(encoding="utf-8"))
    assert summary["status"] == "error"
    assert summary["output_name"] == output_name
    assert summary["datasets"]["voicebank"]["reason"] == (
        "scp_preparation_failed: missing testset"
    )
    assert envelope["status"] == "error"
    assert summary["provenance"] == envelope["provenance"]
    assert summary["provenance"]["parameters"]["output_name"] == output_name
    assert envelope["inference_coverage"] == summary["coverage"]


@pytest.mark.parametrize("task", ["se", "aec"])
@pytest.mark.parametrize("changed_asset", ["config", "base_config", "checkpoint"])
def test_eval_main_revalidates_assets_before_commit_and_preserves_previous_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    changed_asset: str,
):
    module = pipeline_se if task == "se" else pipeline_aec
    assets = tmp_path / "assets"
    assets.mkdir()
    base = assets / "base.yaml"
    config = assets / "config.yaml"
    checkpoint = assets / "model.tar"
    base.write_text("stft: {n_fft: 320}\n", encoding="utf-8")
    config.write_text(
        "base_config: base.yaml\nmodel: {type: Test}\n", encoding="utf-8"
    )
    checkpoint.write_bytes(b"checkpoint-v1")
    output_name = f"{task}-drift"
    common = dict(
        exp_dir=None,
        config=str(config),
        checkpoint=str(checkpoint),
        output_name=output_name,
        datasets="voicebank",
        step="all",
        device="cpu",
        inference_devices=None,
        urgent_noisy_dir="unused",
        urgent_clean_dir="unused",
        voicebank_noisy_scp="unused",
        voicebank_clean_scp="unused",
        dns_challenge_noisy_dir="unused",
    )
    args = SimpleNamespace(
        **common,
        **(
            {"include_aec_sim": False, "include_mc_sim": False}
            if task == "se"
            else {"aec_subsets": "all", "aec_testset_root": "unused"}
        ),
    )
    mutate = False

    def process(
        _args, _config, _checkpoint, _root, datasets, _scps
    ) -> None:
        nonlocal mutate
        datasets["voicebank"] = {"status": "done"}
        if not mutate:
            return
        if changed_asset == "config":
            config.write_text(
                "base_config: base.yaml\nmodel: {type: Changed}\n",
                encoding="utf-8",
            )
        elif changed_asset == "base_config":
            base.write_text("stft: {n_fft: 512}\n", encoding="utf-8")
        else:
            checkpoint.write_bytes(b"checkpoint-v2")

    monkeypatch.setattr(module, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(
        module, "detect_model_type", lambda _config, **_kwargs: task
    )
    if task == "aec":
        monkeypatch.setattr(
            module, "detect_aec_subtype", lambda _config, **_kwargs: "joint"
        )
    monkeypatch.setattr(
        module, "prepare_testset_scps", lambda **_kwargs: {"voicebank": {}}
    )
    monkeypatch.setattr(module, "process_voicebank", process)
    monkeypatch.setattr(module, "print_summary_table", lambda _summary: None)

    assert module.main() == 0
    output = tmp_path / "exp/eval" / task / output_name
    current_before = (output / CURRENT_FILENAME).read_bytes()
    summary_before = (output / "summary.json").read_bytes()
    envelope_before = (output / ARTIFACT_FILENAME).read_bytes()
    result_before = (output / "result_manifest.json").read_bytes()
    committed_run = load_committed_artifact(output)["run_id"]

    mutate = True
    assert module.main() == 1

    assert (output / CURRENT_FILENAME).read_bytes() == current_before
    assert (output / "summary.json").read_bytes() == summary_before
    assert (output / ARTIFACT_FILENAME).read_bytes() == envelope_before
    assert (output / "result_manifest.json").read_bytes() == result_before
    assert load_committed_artifact(output)["run_id"] == committed_run


@pytest.mark.parametrize("output_name", ["", ".", "..", "../x", "x/y", "x\\y", "has space", "中文"])
def test_resolve_eval_output_name_rejects_unsafe_names(tmp_path: Path, output_name: str):
    config = tmp_path / "assets" / "config.yaml"
    config.parent.mkdir()
    config.touch()

    with pytest.raises(ValueError, match="安全 basename"):
        resolve_eval_output_name(output_name, exp_dir="demo", config=config)


def test_resolve_eval_output_name_preserves_legacy_fallback(tmp_path: Path):
    config = tmp_path / "assets" / "config.yaml"
    config.parent.mkdir()
    config.touch()

    assert resolve_eval_output_name(None, exp_dir="exp/aec/demo", config=config) == "demo"
    assert resolve_eval_output_name(None, exp_dir=None, config=config) == "assets"
    assert (
        resolve_eval_output_name(
            "demo@model_0060-deadbeefcafe", exp_dir="exp/aec/demo", config=config
        )
        == "demo@model_0060-deadbeefcafe"
    )


@pytest.mark.parametrize("legacy_name", ["中文 名称", "baseline+ablation", "模型 (final)"])
def test_resolve_eval_output_name_keeps_legacy_unvalidated_names(
    tmp_path: Path, legacy_name: str
):
    config = tmp_path / legacy_name / "config.yaml"
    config.parent.mkdir()
    config.touch()

    assert (
        resolve_eval_output_name(
            None, exp_dir=f"exp/aec/{legacy_name}", config=config
        )
        == legacy_name
    )
    assert resolve_eval_output_name(None, exp_dir=None, config=config) == legacy_name


pytestmark = [pytest.mark.unit]
