"""Generation-3 Step3 GPU pool protocol tests without real MOS/GPU runtime."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import audio_filter_pipeline.processors.mos as mos
import audio_filter_pipeline.processors.stages_v3.mos_gpu_job as mos_gpu_job
import audio_filter_pipeline.processors.stages_v3.mos_pool as mos_pool
import audio_filter_pipeline.processors.stages_v3.mos_runtime as mos_runtime
import audio_filter_pipeline.processors.workers.step3_mos_worker as mos_worker

from audio_filter_pipeline.processors.stages_v3.mos_asset_preflight import verify_mos_assets
from audio_filter_pipeline.processors.stages_v3.mos_execution import MosAssetSpec, MosExecutionPlan
from audio_filter_pipeline.processors.stages_v3.mos_pool import (
    Step3MosScorerPool,
    Step3MosWorkerFleet,
)
from audio_filter_pipeline.processors.stages_v3.mos_resume import (
    Step3ResumeIdentity,
    Step3ResumeStore,
)
from audio_filter_pipeline.processors.stages_v3.runner import DatasetRunContext
from audio_filter_pipeline.processors.stages_v3.segment import STEP2_ROW_FIELDS
from audio_filter_pipeline.processors.stages_v3.select import (
    STEP3_ROW_FIELDS,
    Step3MosSelectionStage,
)
from audio_filter_pipeline.utils import artifacts


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _legacy_direct_pool_urgent_identity(monkeypatch):
    """隔离未持有 VerifiedMosExecutionPlan 的旧测试资源。"""
    def verify(_checkpoint, _source_dir, expected_sha256, expected_commit):
        return {
            "checkpoint_sha256": expected_sha256,
            "source_commit": expected_commit,
            "source_package_sha256": "9" * 64,
        }

    monkeypatch.setattr(mos_pool, "verify_urgent_pk_trusted_assets", verify)
    monkeypatch.setattr(mos_gpu_job, "verify_urgent_pk_trusted_assets", verify)


def _row(ordinal: int) -> dict[str, object]:
    values = {
        "source_key": f"source-{ordinal}", "dataset_id": "alpha", "language": "en",
        "category": "read_speech", "quality_profile": "quality_high",
        "input_root": "/input", "original_path": f"/input/{ordinal}.wav",
        "original_relative_path": f"{ordinal}.wav", "source_relative_path": f"{ordinal}.wav",
        "source_size_bytes": 42, "source_sha256": "a" * 64,
        "sample_rate": 16000, "frames": 32000, "channels": 1, "duration_sec": 2.0,
        "format": "WAV", "subtype": "PCM_16", "speaker_id": None, "speaker_scope": None,
        "source_start_sample": 0, "source_end_sample": 32000,
        "signal_finite": True, "signal_zero_ratio": 0.0, "signal_peak": 0.5,
        "signal_clipping_ratio": 0.0, "signal_dc_offset": 0.0, "signal_rms_dbfs": -20.0,
        "signal_num_samples": 32000, "bandwidth_hz": 7000.0, "estimated_sr": 16000,
        "vad_speech_ratio": 0.9, "vad_timeline_sample_rate": 16000,
        "vad_timeline_num_samples": 32000, "segment_id": f"segment-{ordinal}",
        "start_sample": 0, "end_sample": 32000, "boundary_reason": "source_end",
    }
    assert tuple(values) == STEP2_ROW_FIELDS
    return values


def _scores():
    return {"scoreq": 4.1, "urgent_pk": 3.7,
            "dnsmos": {"ovrl": 3.5, "sig": 3.4, "bak": 4.2, "p808": 3.9}}


def _consume_generator(generator):
    rows = []
    while True:
        try:
            rows.append(next(generator))
        except StopIteration as stop:
            return rows, stop.value


def _write_ordinal_sidecar(path, ordinals):
    path = os.fspath(path)
    digest = hashlib.sha256()
    size_bytes = 0
    with open(path, "xb") as stream:
        for ordinal in ordinals:
            payload = mos_gpu_job._encode_json_line(
                {"ordinal": ordinal}, "MOS sparse ordinal sidecar row",
            )
            stream.write(payload)
            digest.update(payload)
            size_bytes += len(payload)
    return {
        "ordinal_sidecar_path": path,
        "ordinal_sidecar_sha256": digest.hexdigest(),
        "ordinal_sidecar_size_bytes": size_bytes,
    }


def _score_rows(pool, rows, models, *, job_hook=None):
    rows = list(rows)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "tasks.jsonl")
        with open(path, "x", encoding="utf-8") as stream:
            for ordinal, row in enumerate(rows):
                stream.write(json.dumps({
                    "ordinal": ordinal,
                    "row": row,
                    "models": list(models),
                    "scoring_policy_hash": "a" * 64,
                }) + "\n")
        groups = [{
            "models": list(models), "path": path, "row_count": len(rows),
        }]
        if job_hook is None:
            return list(pool.score_task_groups(groups))
        job = _owned_job(pool)
        job_hook(job)
        return list(job.run(groups))


def test_asset_binding_break_keeps_step3_row_schema_unchanged():
    assert STEP3_ROW_FIELDS == STEP2_ROW_FIELDS + (
        "mos_models", "mos_scoring_policy_hash", "mos_scores",
    )


def test_pool_exposes_only_canonical_step3_scoring_entrypoint():
    pool = Step3MosScorerPool([0], dataset_id="alpha")

    assert callable(pool.provenance_payload)
    assert callable(pool.score_task_groups)
    assert not hasattr(pool, "score_rows")


def test_pool_does_not_own_per_run_gpu_transaction_methods():
    for name in (
        "_run_worker_process",
        "_read_result",
        "_terminate_active_children",
        "_cleanup_worker_process",
        "_write_json",
        "_iter_ordinal_sidecar",
    ):
        assert not hasattr(Step3MosScorerPool, name)


def _tree_options(tmp_path, model, digest):
    model_dir = tmp_path / f"{model}-assets"
    model_dir.mkdir(exist_ok=True)
    return {
        "model_dir": str(model_dir),
        "verified_model_tree_sha256": digest,
    }


def _urgent_options(tmp_path):
    return {
        "checkpoint": str(tmp_path / "urgent.ckpt"),
        "source_dir": str(tmp_path / "urgent-source"),
        "encoder": "wav2vec2",
        "backbone": "cnn",
        "score_range": {"min": 1, "max": 5},
        "expected_sha256": "7" * 64,
        "expected_commit": "8" * 40,
    }


def _default_pool_options():
    return {
        "scoreq_options": {
            "model_dir": "/assets/scoreq",
            "verified_model_tree_sha256": "1" * 64,
        },
        "urgent_options": {
            "checkpoint": "/assets/urgent.ckpt",
            "source_dir": "/assets/urgent-source",
            "encoder": "wav2vec2",
            "backbone": "cnn",
            "score_range": {"min": 1, "max": 5},
            "expected_sha256": "7" * 64,
            "expected_commit": "8" * 40,
        },
        "dnsmos_options": {
            "model_dir": "/assets/dnsmos",
            "verified_model_tree_sha256": "2" * 64,
        },
    }


def _owned_job(pool):
    options_by_model = {
        model: getattr(pool, f"_{field}")
        for model, field in mos_pool.MOS_MODEL_REQUEST_OPTION_FIELDS.items()
    }
    settings = mos_gpu_job._GpuScoringJobSettings(
        gpu_ids=pool._gpu_ids,
        dataset_id=pool._dataset_id,
        verified_models=pool._verified_models,
        options_by_model=options_by_model,
        asset_provenance=pool._asset_provenance,
        worker_idle_timeout_seconds=pool._worker_idle_timeout_seconds,
        worker_hard_timeout_seconds=pool._worker_hard_timeout_seconds,
        worker_stderr_tail_bytes=pool._worker_stderr_tail_bytes,
        popen=pool._popen,
        progress_callback=pool._notify_progress,
    )
    return mos_gpu_job._OwnedGpuScoringJob(settings, pool._request_runner)


def _checkpoint_model_options(tmp_path, model):
    return {
        "model_path": str(tmp_path / f"{model}.pt"),
        "model_version": f"{model}-test",
        "model_sha256": "3" * 64,
        "checkpoint_sha256": "4" * 64,
        "checkpoint_size_bytes": 1,
    }


def _worker_request(tmp_path, input_path, output_path, row_count, *, models=None):
    models = tuple(models or mos_worker._MODELS)
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    options = {
        "urgent_options": _urgent_options(tmp_path) if "urgent_pk" in models else None,
        "scoreq_options": (
            _tree_options(tmp_path, "scoreq", "1" * 64)
            if "scoreq" in models else None
        ),
        "dnsmos_options": (
            _tree_options(tmp_path, "dnsmos", "2" * 64)
            if "dnsmos" in models else None
        ),
        "nisqa_options": (
            _checkpoint_model_options(tmp_path, "nisqa")
            if "nisqa" in models else None
        ),
        "utmos_options": (
            _checkpoint_model_options(tmp_path, "utmos")
            if "utmos" in models else None
        ),
    }
    return {
        "protocol": mos_worker._PROTOCOL,
        "asset_load_binding": dict(mos_worker.ASSET_LOAD_BINDING),
        "scoring_execution_policy": mos_runtime._plain_json(
            mos_worker.MOS_SCORING_EXECUTION_POLICY,
        ),
        "private_dir": str(tmp_path),
        "attestation_path": str(tmp_path / "runtime-attestation.json"),
        "subshards": [{
            "models": list(models),
            "input_path": str(input_path),
            "output_path": str(output_path),
            "row_count": row_count,
            "input_sha256": input_sha256,
            "start_ordinal": 0,
            "end_ordinal": row_count,
        }],
        **options,
    }


def _fake_provider_attestation(models, checkpoint_digests=None):
    digests = {
        "nisqa": "4" * 64,
        "utmos": "4" * 64,
    }
    if checkpoint_digests is not None:
        digests.update(checkpoint_digests)
    providers = {
        "scoreq": {
            "execution_mode": "cuda_primary_hybrid_ep",
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "device": "gpu",
        },
        "urgent_pk": {"device": "cuda", "parameters": "cuda"},
        "dnsmos": {
            "onnx_sess": ["CUDAExecutionProvider"],
            "p808_onnx_sess": ["CUDAExecutionProvider"],
            "cpu_ep_fallback_disabled": False,
        },
        "nisqa": {
            "device": "cuda", "parameters": "cuda",
            "checkpoint_sha256": digests["nisqa"],
        },
        "utmos": {
            "device": "cuda", "parameters": "cuda",
            "checkpoint_sha256": digests["utmos"],
        },
    }
    value = {
        "requested_device": "cuda",
        "actual_providers": {model: providers[model] for model in models},
        "fallback": False,
        "selected_models": list(models),
        "asset_load_binding": dict(mos_worker.ASSET_LOAD_BINDING),
        "scoring_execution_policy": mos_runtime._plain_json(
            mos_worker.MOS_SCORING_EXECUTION_POLICY,
        ),
    }
    if "urgent_pk" in models:
        value["urgent_pk_assets"] = {
            "checkpoint_sha256": "7" * 64,
            "source_commit": "8" * 40,
            "source_package_sha256": "9" * 64,
        }
    return value


class _FakeWaveformCoordinator:
    def __init__(self, runtime, models):
        self._runtime = runtime
        self._models = tuple(models)
        self._rows = []
        self._source_identity = None
        self._source_context = None
        self._source_group = None
        self._closed = False

    def _close_source(self, *exc_info):
        context, self._source_context = self._source_context, None
        self._source_group = None
        self._source_identity = None
        if context is not None:
            context.__exit__(*exc_info)

    def _flush(self):
        if not self._rows:
            return []
        rows, self._rows = self._rows, []
        if self._source_group is None:
            raise RuntimeError("fake waveform coordinator has no source group")
        scores = self._source_group.score_batch(
            [row for _ordinal, row in rows],
        )
        return list(zip((ordinal for ordinal, _row in rows), scores, strict=True))

    def push(self, row, ordinal):
        if self._closed:
            raise RuntimeError("fake waveform coordinator is closed")
        if len(self._rows) == mos_runtime._MAX_BATCH_ROWS:
            completed = self._flush()
        else:
            completed = []
        reservation = (row["end_sample"] - row["start_sample"]) * (
            row["channels"] + 1
        ) * 4
        if reservation > mos_runtime._MAX_BATCH_FLOAT32_BYTES:
            raise ValueError("decoded MOS interval exceeds the float32 batch byte limit")
        if self._rows and sum(
            (item[1]["end_sample"] - item[1]["start_sample"]) * (
                item[1]["channels"] + 1
            ) * 4
            for item in self._rows
        ) + reservation > mos_runtime._MAX_BATCH_FLOAT32_BYTES:
            completed.extend(self._flush())
        identity = self._runtime.source_identity(row)
        if identity != self._source_identity:
            self._close_source(None, None, None)
            context = self._runtime.open_source_group(row, self._models)
            self._source_group = context.__enter__()
            self._source_context = context
            self._source_identity = identity
        self._rows.append((ordinal, row))
        return completed

    def finish(self):
        if self._closed:
            raise RuntimeError("fake waveform coordinator is closed")
        try:
            return self._flush()
        finally:
            self._closed = True
            self._close_source(*sys.exc_info())

    def abort(self, *exc_info):
        if self._closed:
            return
        self._closed = True
        self._rows.clear()
        self._close_source(*exc_info)


class _EmptyWaveformCoordinator:
    def push(self, _row, _ordinal):
        return []

    def finish(self):
        return []

    def abort(self, *_exc_info):
        return None


class _FakeRuntimeAttestation:
    @staticmethod
    def ensure_model_provenance(models):
        return _fake_provider_attestation(tuple(models))

    def runtime_attestation(self, models):
        return self.ensure_model_provenance(models)

    def open_waveform_coordinator(self, models):
        return _FakeWaveformCoordinator(self, models)



def _write_fake_worker_attestation(
    request, checkpoint_digests=None, urgent_pk_assets=None,
):
    expected_digests = {
        model: request[f"{model}_options"]["checkpoint_sha256"]
        for model in ("nisqa", "utmos")
        if request[f"{model}_options"] is not None
    }
    if checkpoint_digests is not None:
        expected_digests.update(checkpoint_digests)
    tuples = []
    worker_models = []
    seen = set()
    for subshard in request["subshards"]:
        models = tuple(subshard["models"])
        for model in models:
            if model not in worker_models:
                worker_models.append(model)
        if models not in seen:
            attestation = _fake_provider_attestation(
                models, expected_digests,
            )
            if "urgent_pk" in models and urgent_pk_assets is not None:
                attestation["urgent_pk_assets"] = dict(urgent_pk_assets)
            tuples.append({
                "models": list(models),
                "attestation": attestation,
            })
            seen.add(models)
    with open(request["attestation_path"], "x", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "protocol": mos_worker._PROTOCOL,
            "asset_load_binding": dict(mos_worker.ASSET_LOAD_BINDING),
            "scoring_execution_policy": json.loads(json.dumps(
                mos_worker.MOS_SCORING_EXECUTION_POLICY,
                default=lambda value: dict(value),
            )),
            "worker_models": worker_models,
            "model_tuple_attestations": tuples,
        }) + "\n")


def _verified_pool_plan(tmp_path, *, score_range=None, suffix=""):
    root = tmp_path / f"verified-assets{suffix}"
    scoreq_dir = root / "scoreq"
    dnsmos_dir = root / "dnsmos"
    nisqa_dir = root / "nisqa"
    utmos_dir = root / "utmos"
    urgent_source = root / "urgent-source"
    for directory in (
        scoreq_dir, dnsmos_dir, nisqa_dir, utmos_dir, urgent_source,
    ):
        directory.mkdir(parents=True)
    urgent_checkpoint = root / "urgent.ckpt"
    nisqa_model = nisqa_dir / "model.tar"
    utmos_model = utmos_dir / "model.pt"
    urgent_checkpoint.write_bytes(b"urgent checkpoint")
    nisqa_model.write_bytes(b"nisqa checkpoint")
    utmos_model.write_bytes(b"utmos checkpoint")
    tree_digests = {
        os.path.realpath(scoreq_dir): "1" * 64,
        os.path.realpath(dnsmos_dir): "2" * 64,
        os.path.realpath(nisqa_dir): "3" * 64,
        os.path.realpath(utmos_dir): "4" * 64,
    }
    checkpoint_identities = {
        os.path.realpath(nisqa_model): {
            "checkpoint_sha256": hashlib.sha256(b"nisqa checkpoint").hexdigest(),
            "checkpoint_size_bytes": len(b"nisqa checkpoint"),
        },
        os.path.realpath(utmos_model): {
            "checkpoint_sha256": hashlib.sha256(b"utmos checkpoint").hexdigest(),
            "checkpoint_size_bytes": len(b"utmos checkpoint"),
        },
    }
    urgent_identity = {
        "checkpoint_sha256": "7" * 64,
        "source_commit": "8" * 40,
        "source_package_sha256": "9" * 64,
    }
    assets = (
        MosAssetSpec("scoreq", {"model_dir": str(scoreq_dir)}),
        MosAssetSpec(
            "urgent_pk",
            {"checkpoint": str(urgent_checkpoint), "source_dir": str(urgent_source)},
            {
                "encoder": "wav2vec2", "backbone": "cnn",
                "score_range": score_range or {"min": 1, "max": 5},
                "expected_sha256": urgent_identity["checkpoint_sha256"],
                "expected_commit": urgent_identity["source_commit"],
            },
        ),
        MosAssetSpec("dnsmos", {"model_dir": str(dnsmos_dir)}),
        MosAssetSpec(
            "nisqa", {"model_path": str(nisqa_model)},
            {"model_version": "nisqa-v2", "model_sha256": tree_digests[os.path.realpath(nisqa_dir)]},
        ),
        MosAssetSpec(
            "utmos", {"model_path": str(utmos_model)},
            {"model_version": "utmos-v1", "model_sha256": tree_digests[os.path.realpath(utmos_dir)]},
        ),
    )
    plan = MosExecutionPlan(
        selected_models=("scoreq", "urgent_pk", "dnsmos", "nisqa", "utmos"),
        assets=assets,
        topology={"kind": "single_process", "device": "cpu"},
    )
    return verify_mos_assets(
        plan,
        tree_digest=lambda path: tree_digests[os.path.realpath(path)],
        checkpoint_snapshot=lambda path: checkpoint_identities[os.path.realpath(path)],
        urgent_verifier=lambda *_args: dict(urgent_identity),
    )


def _pool_options(verified):
    options = verified.as_mapping()["options_by_model"]
    return {
        "scoreq_options": options["scoreq"],
        "urgent_options": options["urgent_pk"],
        "dnsmos_options": options["dnsmos"],
        "nisqa_options": options["nisqa"],
        "utmos_options": options["utmos"],
    }


def _five_model_scores():
    return {**_scores(), "nisqa": 3.9, "utmos": 4.0}


def test_pool_and_worker_jsonl_readers_reject_symlink(tmp_path):
    target = tmp_path / "target.jsonl"
    target.write_bytes(b"{}\n")
    linked = tmp_path / "linked.jsonl"
    linked.symlink_to(target)

    with pytest.raises(ValueError, match="regular|symlink"):
        list(mos_gpu_job._iter_bounded_jsonl(str(linked), "MOS result"))
    with pytest.raises(ValueError, match="regular|symlink"):
        list(mos_worker._iter_bounded_jsonl(str(linked), "MOS input"))


def test_worker_request_rejects_symlink(tmp_path):
    target = tmp_path / "request.json"
    target.write_text("{}", encoding="utf-8")
    linked = tmp_path / "request-link.json"
    linked.symlink_to(target)

    with pytest.raises(ValueError, match="regular|symlink"):
        mos_worker._load_request(str(linked))


@pytest.mark.parametrize("model", ["scoreq", "dnsmos"])
@pytest.mark.parametrize(
    ("protocol", "digest", "expected_error"),
    [
        ("audio_filter_pipeline.step3_mos_worker.v1", "1" * 64, "protocol"),
        ("audio_filter_pipeline.step3_mos_worker.v3", "1" * 64, "protocol"),
        ("audio_filter_pipeline.step3_mos_worker.v4", "1" * 64, "protocol"),
        ("audio_filter_pipeline.step3_mos_worker.v6", None, "field|option|digest"),
        ("audio_filter_pipeline.step3_mos_worker.v6", "A" * 64, "option|hexadecimal"),
        ("audio_filter_pipeline.step3_mos_worker.v6", "1" * 63, "option|hexadecimal"),
        ("audio_filter_pipeline.step3_mos_worker.v6", "g" * 64, "option|hexadecimal"),
    ],
)
def test_worker_rejects_legacy_protocol_and_unbound_tree_digest(
    tmp_path, model, protocol, digest, expected_error,
):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    request_path = tmp_path / "request.json"
    input_path.write_text(
        json.dumps({"ordinal": 0, "row": _row(0)}) + "\n", encoding="utf-8",
    )
    request = _worker_request(
        tmp_path, input_path, output_path, 1, models=(model,),
    )
    request["protocol"] = protocol
    option_field = f"{model}_options"
    if digest is None:
        request[option_field].pop("verified_model_tree_sha256")
    else:
        request[option_field]["verified_model_tree_sha256"] = digest
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        mos_worker._load_request(str(request_path))


def test_worker_v6_requires_complete_sparse_ordinal_sidecar_schema(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    request_path = tmp_path / "request.json"
    input_path.write_text(
        json.dumps({"ordinal": 0, "row": _row(0)}) + "\n", encoding="utf-8",
    )
    request = _worker_request(
        tmp_path, input_path, output_path, 1, models=("scoreq",),
    )
    request["subshards"][0]["ordinal_sidecar_path"] = str(tmp_path / "ordinals.jsonl")
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ValueError, match="subshard schema"):
        mos_worker._load_request(str(request_path))


def test_worker_accepts_more_than_32_bounded_physical_subshards(tmp_path):
    request_path = tmp_path / "request.json"
    initial_input = tmp_path / "input-0.jsonl"
    initial_input.write_text(
        json.dumps({"ordinal": 0, "row": _row(0)}) + "\n", encoding="utf-8",
    )
    request = _worker_request(
        tmp_path, initial_input, tmp_path / "output-0.jsonl", 1,
        models=("scoreq",),
    )
    subshards = []
    for ordinal in range(33):
        input_path = tmp_path / f"input-{ordinal}.jsonl"
        if ordinal:
            input_path.write_text(
                json.dumps({"ordinal": ordinal, "row": _row(ordinal)}) + "\n",
                encoding="utf-8",
            )
        subshards.append({
            "models": ["scoreq"],
            "input_path": str(input_path),
            "output_path": str(tmp_path / f"output-{ordinal}.jsonl"),
            "row_count": 1,
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "start_ordinal": ordinal,
            "end_ordinal": ordinal + 1,
        })
    request["subshards"] = subshards
    request_path.write_text(json.dumps(request), encoding="utf-8")

    loaded = mos_worker._load_request(str(request_path))

    assert len(loaded["subshards"]) == 33
    assert mos_worker.MAX_WORKER_INPUT_SUBSHARDS == 40


def test_worker_rejects_41_subshards_before_opening_any_input(tmp_path, monkeypatch):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps({"ordinal": 0, "row": _row(0)}) + "\n", encoding="utf-8",
    )
    request = _worker_request(
        tmp_path, input_path, tmp_path / "output.jsonl", 1,
        models=("scoreq",),
    )
    request["subshards"] = [dict(request["subshards"][0]) for _ in range(41)]
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    real_open = mos_worker._open_regular_stream
    opened_labels = []

    def open_stream(path, label):
        opened_labels.append(label)
        return real_open(path, label)

    monkeypatch.setattr(mos_worker, "_open_regular_stream", open_stream)

    with pytest.raises(ValueError, match="subshard count exceeds canonical capacity"):
        mos_worker._load_request(str(request_path))

    assert opened_labels == ["worker request"]


@pytest.mark.parametrize(
    "alias_case",
    ["duplicate_input", "input_output", "input_sidecar", "request_attestation"],
)
def test_worker_rejects_global_owned_path_aliases(tmp_path, alias_case):
    request_path = tmp_path / "request.json"
    first_input = tmp_path / "input-0.jsonl"
    first_input.write_text(
        json.dumps({"ordinal": 0, "row": _row(0)}) + "\n", encoding="utf-8",
    )
    request = _worker_request(
        tmp_path, first_input, tmp_path / "output-0.jsonl", 1,
        models=("scoreq",),
    )
    if alias_case == "duplicate_input":
        duplicate = dict(request["subshards"][0])
        duplicate.update({
            "output_path": str(tmp_path / "output-1.jsonl"),
            "start_ordinal": 1,
            "end_ordinal": 2,
        })
        request["subshards"].append(duplicate)
    elif alias_case == "input_output":
        request["subshards"][0]["output_path"] = str(first_input)
    elif alias_case == "input_sidecar":
        request["subshards"][0].update({
            "ordinal_sidecar_path": str(first_input),
            "ordinal_sidecar_sha256": hashlib.sha256(first_input.read_bytes()).hexdigest(),
            "ordinal_sidecar_size_bytes": first_input.stat().st_size,
        })
        request["subshards"][0]["end_ordinal"] = 2
    else:
        request["attestation_path"] = str(request_path)
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        mos_worker._load_request(str(request_path))


def test_worker_rejects_non_path_sparse_sidecar_without_type_error(tmp_path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps({"ordinal": 0, "row": _row(0)}) + "\n", encoding="utf-8",
    )
    request_path = tmp_path / "request.json"
    request = _worker_request(
        tmp_path, input_path, tmp_path / "output.jsonl", 1,
        models=("scoreq",),
    )
    request["subshards"][0].update({
        "ordinal_sidecar_path": None,
        "ordinal_sidecar_sha256": "a" * 64,
        "ordinal_sidecar_size_bytes": 1,
    })
    request["subshards"][0]["end_ordinal"] = 2
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ValueError, match="subshard values"):
        mos_worker._load_request(str(request_path))


def test_tree_bound_options_use_each_models_canonical_schema(monkeypatch):
    common = frozenset({"model_dir", "verified_model_tree_sha256"})
    monkeypatch.setattr(mos_pool, "_TREE_BOUND_OPTION_FIELDS", {
        "scoreq": common | {"scoreq_only"},
        "dnsmos": common,
    })
    options = {
        "model_dir": "/assets/dnsmos",
        "verified_model_tree_sha256": "2" * 64,
    }

    mos_pool._validate_tree_bound_options(
        "dnsmos", options, "dnsmos_options",
    )
    with pytest.raises(ValueError, match="scoreq_only"):
        mos_pool._validate_tree_bound_options(
            "scoreq", options, "scoreq_options",
        )


def test_worker_rejects_asset_load_binding_v1(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    request_path = tmp_path / "request.json"
    input_path.write_text(
        json.dumps({"ordinal": 0, "row": _row(0)}) + "\n", encoding="utf-8",
    )
    request = _worker_request(
        tmp_path, input_path, output_path, 1, models=("scoreq",),
    )
    request["asset_load_binding"] = {"version": 1}
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ValueError, match="binding"):
        mos_worker._load_request(str(request_path))


@pytest.mark.parametrize("model", ["scoreq", "dnsmos"])
def test_worker_rejects_tree_mismatch_before_snapshot_loader(
    tmp_path, monkeypatch, model,
):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    request_path = tmp_path / "request.json"
    input_path.write_text(
        json.dumps({"ordinal": 0, "row": _row(0)}) + "\n", encoding="utf-8",
    )
    request = _worker_request(
        tmp_path, input_path, output_path, 1, models=(model,),
    )
    option_field = f"{model}_options"
    model_dir = request[option_field]["model_dir"]
    asset_path = os.path.join(model_dir, "model.onnx")
    with open(asset_path, "wb") as stream:
        stream.write(b"verified model tree")
    verified_snapshot = mos.open_owned_model_tree_snapshot(model_dir)
    try:
        expected_digest = verified_snapshot.sha256
    finally:
        verified_snapshot.close()
    request[option_field]["verified_model_tree_sha256"] = expected_digest
    request_path.write_text(json.dumps(request), encoding="utf-8")
    # parent request 固化后替换同路径内容，worker 必须在私有 loader 前拒绝。
    with open(asset_path, "wb") as stream:
        stream.write(b"replaced model tree")

    snapshot_attempts = []
    loader_calls = []
    create_snapshot = mos._create_owned_model_tree_snapshot

    def track_snapshot(path, expected, **kwargs):
        snapshot_attempts.append((os.path.realpath(path), expected))
        return create_snapshot(path, expected, **kwargs)

    monkeypatch.setattr(mos, "_create_owned_model_tree_snapshot", track_snapshot)
    if model == "scoreq":
        import core.metrics.mos_models as mos_models

        monkeypatch.setattr(
            mos_models, "load_scoreq",
            lambda **_kwargs: loader_calls.append(True),
        )
    else:
        monkeypatch.setattr(
            mos, "_load_private_dnsmos_module",
            lambda _snapshot: loader_calls.append(True),
        )

    with pytest.raises(
        (ValueError, RuntimeError),
        match=f"{model}|{model.upper()}|tree|digest|snapshot",
    ):
        mos_worker.handle_request(str(request_path))

    assert snapshot_attempts == [(os.path.realpath(model_dir), expected_digest)]
    assert loader_calls == []
    assert not output_path.exists()


@pytest.mark.parametrize("model", ["scoreq", "dnsmos"])
def test_owned_snapshot_is_reused_by_loader_reset_and_cache_hit(
    tmp_path, monkeypatch, model,
):
    model_dir = tmp_path / model
    model_dir.mkdir()
    replacement_dir = tmp_path / f"{model}-replacement"
    replacement_dir.mkdir()
    held_dir = tmp_path / f"{model}-held"
    loaded_paths = []
    loaded_payloads = []
    private_module_paths = []

    if model == "scoreq":
        model_name = "adapt_nr_telephone.onnx"
        (model_dir / model_name).write_bytes(b"scoreq-original")
        (replacement_dir / model_name).write_bytes(b"scoreq-replacement")
    else:
        for root, suffix in (
            (model_dir, b"original"),
            (replacement_dir, b"replacement"),
        ):
            (root / "sig_bak_ovr.onnx").write_bytes(b"primary-" + suffix)
            (root / "model_v8.onnx").write_bytes(b"p808-" + suffix)
            (root / "dnsmos_local.py").write_text(
                "class ComputeScore:\n"
                "    def __init__(self, primary, p808, use_gpu=False):\n"
                "        self.paths = (primary, p808)\n"
                "    def run_batch_audio(self, items, **kwargs):\n"
                "        return [dict(P808_MOS=4.0, SIG=4.0, BAK=4.0, OVRL=4.0) "
                "for _ in items]\n",
                encoding="utf-8",
            )

    verified_snapshot = mos.open_owned_model_tree_snapshot(str(model_dir))
    try:
        expected_digest = verified_snapshot.sha256
    finally:
        verified_snapshot.close()

    snapshot_creations = []
    create_snapshot = mos._create_owned_model_tree_snapshot

    def track_snapshot(path, expected, **kwargs):
        snapshot = create_snapshot(path, expected, **kwargs)
        stable_path = mos._owned_model_tree_root_descriptor_path(
            model.upper(), snapshot,
        )
        snapshot_creations.append(
            (os.path.realpath(path), snapshot.path, stable_path)
        )
        return snapshot

    monkeypatch.setattr(mos, "_create_owned_model_tree_snapshot", track_snapshot)
    monkeypatch.setattr(mos, "_MODEL_CACHE", {})
    if model == "scoreq":
        import core.metrics.mos_models as mos_models

        class Scoreq:
            @staticmethod
            def run_batch(items):
                return [{"MOS": 4.0} for _item in items]

        def load_scoreq(*, model_dir, use_gpu):
            assert use_gpu is False
            loaded_paths.append(model_dir)
            loaded_payloads.append(
                Path(model_dir, model_name).read_bytes()
            )
            return Scoreq()

        monkeypatch.setattr(mos_models, "load_scoreq", load_scoreq)
        binding = mos.bind_scoreq_options
    else:
        load_private_module = mos._load_private_dnsmos_module

        def track_private_module(snapshot):
            module_name, module = load_private_module(snapshot)
            private_module_paths.append(module.__file__)
            original_init = module.ComputeScore.__init__

            def track_init(instance, primary, p808, use_gpu=False):
                assert use_gpu is False
                loaded_paths.extend((primary, p808))
                loaded_payloads.extend((
                    Path(primary).read_bytes(), Path(p808).read_bytes(),
                ))
                original_init(instance, primary, p808, use_gpu=use_gpu)

            module.ComputeScore.__init__ = track_init
            return module_name, module

        monkeypatch.setattr(mos, "_load_private_dnsmos_module", track_private_module)
        binding = mos.bind_dnsmos_options

    options = {
        "model_dir": str(model_dir),
        "verified_model_tree_sha256": expected_digest,
    }
    snapshot_path = stable_path = None
    try:
        with binding(options):
            estimator = mos.get_mos_estimator(
                model, use_gpu=False, use_cache=True, allow_fallback=False,
            )
            assert len(snapshot_creations) == 1
            snapshot_path = snapshot_creations[0][1]
            stable_path = snapshot_creations[0][2]
            assert snapshot_path != os.path.realpath(model_dir)
            assert re.fullmatch(r"/proc/self/fd/\d+", stable_path)
            assert os.path.realpath(stable_path) == snapshot_path
            assert mos.get_mos_estimator(
                model, use_gpu=False, use_cache=True, allow_fallback=False,
            ) is estimator
            assert len(snapshot_creations) == 1

            model_dir.rename(held_dir)
            replacement_dir.rename(model_dir)
            estimator.compute_batch([
                (np.zeros(160, dtype=np.float32), 16_000),
            ])
            estimator.reset()

        assert len(snapshot_creations) == 1
        if model == "scoreq":
            assert loaded_payloads == [b"scoreq-original"] * 2
            assert loaded_paths == [stable_path] * 2
        else:
            assert private_module_paths == [
                os.path.join(stable_path, "dnsmos_local.py"),
            ]
            assert loaded_payloads == [
                b"primary-original", b"p808-original",
            ] * 2
            assert all(
                os.path.dirname(path) == stable_path for path in loaded_paths
            )
        assert os.path.isdir(snapshot_path)
        assert os.path.isdir(stable_path)
    finally:
        mos.clear_model_cache()

    assert snapshot_path is not None and stable_path is not None
    assert not os.path.exists(snapshot_path)
    assert not os.path.exists(stable_path)
    with pytest.raises(OSError):
        os.open(stable_path, os.O_RDONLY | os.O_DIRECTORY)


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "mkfifo"),
    reason="FIFO resources are unavailable",
)
def test_pool_worker_readers_and_request_reject_fifo(tmp_path):
    fifo = tmp_path / "input.fifo"
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="regular"):
        list(mos_gpu_job._iter_bounded_jsonl(str(fifo), "MOS result"))
    with pytest.raises(ValueError, match="regular"):
        list(mos_worker._iter_bounded_jsonl(str(fifo), "MOS input"))
    with pytest.raises(ValueError, match="regular"):
        mos_worker._load_request(str(fifo))


@pytest.mark.parametrize("module", [mos_gpu_job, mos_worker])
def test_new_output_rejects_symlink(tmp_path, module):
    target = tmp_path / "target.jsonl"
    target.write_bytes(b"")
    linked = tmp_path / "output.jsonl"
    linked.symlink_to(target)

    with pytest.raises(ValueError, match="new regular"):
        module._open_new_regular_stream(str(linked), "MOS worker result output")


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "mkfifo"),
    reason="FIFO resources are unavailable",
)
@pytest.mark.parametrize("module", [mos_gpu_job, mos_worker])
def test_new_output_rejects_fifo(tmp_path, module):
    fifo = tmp_path / "output.fifo"
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="new regular"):
        module._open_new_regular_stream(str(fifo), "MOS worker result output")


@pytest.mark.parametrize("module", [mos_gpu_job, mos_worker])
def test_jsonl_reader_rejects_same_size_mtime_change(tmp_path, monkeypatch, module):
    path = tmp_path / "input.jsonl"
    path.write_bytes(b"{}\n")
    original_fstat = module.os.fstat
    calls = 0

    def fstat_with_changed_mtime(fd):
        nonlocal calls
        value = original_fstat(fd)
        calls += 1
        if calls == 2:
            return SimpleNamespace(
                st_dev=value.st_dev, st_ino=value.st_ino, st_mode=value.st_mode,
                st_size=value.st_size, st_mtime_ns=value.st_mtime_ns + 1,
                st_ctime_ns=value.st_ctime_ns,
            )
        return value

    monkeypatch.setattr(module.os, "fstat", fstat_with_changed_mtime)
    if module is mos_gpu_job:
        iterator = module._iter_bounded_jsonl(str(path), "MOS result")
        with pytest.raises(ValueError, match="changed"):
            list(iterator)
    else:
        with path.open("rb") as stream:
            iterator = module._iter_bounded_jsonl(stream, "MOS input")
            with pytest.raises(ValueError, match="changed"):
                list(iterator)


def test_worker_rejects_bad_input_digest_before_runtime_output_or_progress(
    tmp_path, monkeypatch,
):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    request_path = tmp_path / "request.json"
    attestation_path = tmp_path / "runtime-attestation.json"
    input_path.write_text(
        json.dumps({"ordinal": 0, "row": _row(0)}) + "\n", encoding="utf-8",
    )
    request = _worker_request(
        tmp_path, input_path, output_path, 1, models=("scoreq",),
    )
    request["subshards"][0]["input_sha256"] = "f" * 64
    request_path.write_text(json.dumps(request), encoding="utf-8")
    runtime_calls = []
    progress = []

    monkeypatch.setattr(
        mos_worker,
        "Step3MosRuntime",
        lambda **_kwargs: runtime_calls.append(True),
    )
    monkeypatch.setattr(mos_worker, "_emit_progress", progress.append)

    with pytest.raises(ValueError, match="digest mismatch"):
        mos_worker.handle_request(str(request_path))

    assert runtime_calls == []
    assert progress == []
    assert not output_path.exists()
    assert not attestation_path.exists()


def test_worker_rejects_same_content_inode_swap_during_prevalidation(
    tmp_path, monkeypatch,
):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    request_path = tmp_path / "request.json"
    held_path = tmp_path / "held-input.jsonl"
    payload = json.dumps({"ordinal": 0, "row": _row(0)}) + "\n"
    input_path.write_text(payload, encoding="utf-8")
    request = _worker_request(
        tmp_path, input_path, output_path, 1, models=("scoreq",),
    )
    request_path.write_text(json.dumps(request), encoding="utf-8")
    real_signature = mos_worker._input_path_signature
    signature_calls = 0
    runtime_calls = []
    progress = []

    def swap_before_signature(path):
        nonlocal signature_calls
        signature_calls += 1
        if signature_calls == 2:
            input_path.rename(held_path)
            input_path.write_text(payload, encoding="utf-8")
            assert input_path.stat().st_ino != held_path.stat().st_ino
        return real_signature(path)

    monkeypatch.setattr(mos_worker, "_input_path_signature", swap_before_signature)
    monkeypatch.setattr(
        mos_worker,
        "Step3MosRuntime",
        lambda **_kwargs: runtime_calls.append(True),
    )
    monkeypatch.setattr(mos_worker, "_emit_progress", progress.append)

    with pytest.raises(ValueError, match="pathname changed during validation"):
        mos_worker.handle_request(str(request_path))

    assert signature_calls == 2
    assert runtime_calls == []
    assert progress == []
    assert not output_path.exists()
    assert not Path(request["attestation_path"]).exists()


def test_worker_scores_from_the_same_prevalidated_file_descriptor(
    tmp_path, monkeypatch,
):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    request_path = tmp_path / "request.json"
    input_path.write_text(
        json.dumps({"ordinal": 0, "row": _row(0)}) + "\n", encoding="utf-8",
    )
    request_path.write_text(json.dumps(_worker_request(
        tmp_path, input_path, output_path, 1, models=("scoreq",),
    )), encoding="utf-8")
    input_stream_ids = []
    real_iter = mos_worker._iter_bounded_jsonl

    def track_input_stream(source, label):
        if label == "MOS worker input shard":
            input_stream_ids.append(id(source))
        return real_iter(source, label)

    class Runtime(_FakeRuntimeAttestation):
        def __init__(self, **_kwargs):
            assert len(input_stream_ids) == 1

        @staticmethod
        def source_identity(row):
            return row["source_key"]

        @contextmanager
        def open_source_group(self, _row, _models):
            yield self

        @staticmethod
        def score_batch(rows):
            return [{"scoreq": 4.1} for _row in rows]

    monkeypatch.setattr(mos_worker, "_iter_bounded_jsonl", track_input_stream)
    monkeypatch.setattr(mos_worker, "Step3MosRuntime", Runtime)
    monkeypatch.setattr(mos_worker, "_emit_progress", lambda _count: None)

    mos_worker.handle_request(str(request_path))

    assert len(input_stream_ids) == 2
    assert input_stream_ids[0] == input_stream_ids[1]
    assert output_path.exists()


class _CloseTrackingStream:
    def __init__(self, stream, *, fail=False):
        self._stream = stream
        self.fail = fail
        self.close_calls = 0

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def close(self):
        self.close_calls += 1
        self._stream.close()
        if self.fail:
            raise RuntimeError("stream close failed")


def test_pool_stream_close_failure_does_not_skip_other_streams(tmp_path, monkeypatch):
    opened = []
    real_open = mos_gpu_job._open_new_regular_stream

    def open_tracking_stream(path, label):
        stream = real_open(path, label)
        if label == "MOS worker input shard":
            stream = _CloseTrackingStream(stream, fail=not opened)
            opened.append(stream)
        return stream

    monkeypatch.setattr(mos_gpu_job, "_open_new_regular_stream", open_tracking_stream)
    pool = Step3MosScorerPool(
        [0, 1], dataset_id="alpha", **_default_pool_options(),
    )

    with pytest.raises(RuntimeError, match="stream close failed"):
        _score_rows(
            pool,
            (_row(index) for index in range(2)),
            ("scoreq", "urgent_pk", "dnsmos"),
        )

    assert len(opened) == 2
    assert all(stream.close_calls == 1 for stream in opened)


def test_worker_streams_contiguous_source_in_bounded_group_batches(tmp_path, monkeypatch):
    rows = [_row(index) for index in range(65)]
    for row in rows:
        row["source_key"] = "shared-source"
        row["original_path"] = "/input/shared.wav"
        row["original_relative_path"] = "shared.wav"
        row["source_relative_path"] = "shared.wav"
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    request_path = tmp_path / "request.json"
    input_path.write_text("".join(
        json.dumps({"ordinal": index, "row": row}) + "\n" for index, row in enumerate(rows)
    ), encoding="utf-8")
    request_path.write_text(json.dumps(
        _worker_request(tmp_path, input_path, output_path, len(rows))
    ), encoding="utf-8")
    opens, batch_sizes = [], []
    progress_output = tempfile.SpooledTemporaryFile()
    monkeypatch.setattr(
        mos_worker.sys, "stdout", SimpleNamespace(buffer=progress_output),
    )

    class Runtime(_FakeRuntimeAttestation):
        def __init__(self, **_kwargs):
            pass
        def source_identity(self, row):
            return (row["source_relative_path"], row["source_sha256"])
        @contextmanager
        def open_source_group(self, row, models):
            assert models == mos_worker._MODELS
            opens.append(self.source_identity(row))
            yield self
        def score_batch(self, batch):
            batch_sizes.append(len(batch))
            return [_scores() for _row in batch]

    monkeypatch.setattr(mos_worker, "Step3MosRuntime", Runtime)
    mos_worker.handle_request(str(request_path))

    progress_output.seek(0)
    assert progress_output.read().splitlines() == [
        mos_worker._PROGRESS_PREFIX + b"32",
        mos_worker._PROGRESS_PREFIX + b"32",
        mos_worker._PROGRESS_PREFIX + b"1",
    ]
    progress_output.close()
    output = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert opens == [("shared.wav", "a" * 64)]
    assert batch_sizes == [32, 32, 1]
    assert [item["ordinal"] for item in output] == list(range(65))
    assert all(set(item) == {"ordinal", "scores"} for item in output)
    legacy_bytes = sum(len(json.dumps({
        "ordinal": index, "row": row, "scores": _scores(),
    }) + "\n") for index, row in enumerate(rows))
    assert output_path.stat().st_size < legacy_bytes / 4


def test_worker_reuses_one_runtime_for_multiple_model_tuple_subshards(tmp_path, monkeypatch):
    first_input = tmp_path / "scoreq.jsonl"
    second_input = tmp_path / "utmos-nisqa.jsonl"
    first_output = tmp_path / "scoreq-result.jsonl"
    second_output = tmp_path / "utmos-nisqa-result.jsonl"
    request_path = tmp_path / "request.json"
    first_input.write_text(
        json.dumps({"ordinal": 0, "row": _row(0)}) + "\n", encoding="utf-8",
    )
    second_input.write_text(
        json.dumps({"ordinal": 1, "row": _row(1)}) + "\n", encoding="utf-8",
    )
    scoreq_options = _tree_options(tmp_path, "scoreq", "1" * 64)
    nisqa_options = _checkpoint_model_options(tmp_path, "nisqa")
    utmos_options = _checkpoint_model_options(tmp_path, "utmos")
    request = {
        "protocol": "audio_filter_pipeline.step3_mos_worker.v6",
        "asset_load_binding": dict(mos_worker.ASSET_LOAD_BINDING),
        "scoring_execution_policy": mos_runtime._plain_json(
            mos_worker.MOS_SCORING_EXECUTION_POLICY,
        ),
        "private_dir": str(tmp_path),
        "attestation_path": str(tmp_path / "runtime-attestation.json"),
        "subshards": [
            {
                "models": ["scoreq"], "input_path": str(first_input),
                "output_path": str(first_output), "row_count": 1,
                "input_sha256": hashlib.sha256(first_input.read_bytes()).hexdigest(),
                "start_ordinal": 0, "end_ordinal": 1,
            },
            {
                "models": ["utmos", "nisqa"], "input_path": str(second_input),
                "output_path": str(second_output), "row_count": 1,
                "input_sha256": hashlib.sha256(second_input.read_bytes()).hexdigest(),
                "start_ordinal": 1, "end_ordinal": 2,
            },
        ],
        "urgent_options": None, "scoreq_options": scoreq_options,
        "dnsmos_options": None, "nisqa_options": nisqa_options,
        "utmos_options": utmos_options,
    }
    request_path.write_text(json.dumps(request), encoding="utf-8")
    constructions, tuples, batches = [], [], []

    class Runtime(_FakeRuntimeAttestation):
        def __init__(self, **kwargs):
            constructions.append(kwargs)
        @staticmethod
        def source_identity(row):
            return row["source_key"]
        @staticmethod
        def ensure_model_provenance(models):
            tuples.append(tuple(models))
            return _fake_provider_attestation(tuple(models))
        @contextmanager
        def open_source_group(self, _row, models):
            selected = tuple(models)
            class Group:
                @staticmethod
                def score_batch(batch):
                    batches.append((selected, len(batch)))
                    scores = ({"scoreq": 4.1} if selected == ("scoreq",)
                              else {"utmos": 4.0, "nisqa": 3.9})
                    return [scores for _item in batch]
            yield Group()

    monkeypatch.setattr(mos_worker, "Step3MosRuntime", Runtime)
    mos_worker.handle_request(str(request_path))

    assert len(constructions) == 1
    runtime_options = constructions[0]["options"]
    assert runtime_options.scoreq_options == scoreq_options
    assert runtime_options.nisqa_options == nisqa_options
    assert runtime_options.utmos_options == utmos_options
    assert tuples == [("scoreq",), ("utmos", "nisqa")]
    assert batches == [(("scoreq",), 1), (("utmos", "nisqa"), 1)]
    first_result = json.loads(first_output.read_text(encoding="utf-8"))
    second_result = json.loads(second_output.read_text(encoding="utf-8"))
    assert set(first_result) == set(second_result) == {"ordinal", "scores"}
    assert (first_result["ordinal"], first_result["scores"]) == (
        0, {"scoreq": 4.1},
    )
    assert (second_result["ordinal"], second_result["scores"]) == (
        1, {"utmos": 4.0, "nisqa": 3.9},
    )
    sidecar = json.loads(
        (tmp_path / "runtime-attestation.json").read_text(encoding="utf-8"),
    )
    assert set(sidecar) == {
        "protocol", "asset_load_binding", "scoring_execution_policy",
        "worker_models", "model_tuple_attestations",
    }
    assert sidecar["protocol"] == "audio_filter_pipeline.step3_mos_worker.v6"
    assert sidecar["asset_load_binding"] == dict(mos_worker.ASSET_LOAD_BINDING)
    assert sidecar["worker_models"] == ["scoreq", "utmos", "nisqa"]
    assert sidecar["model_tuple_attestations"] == [
        {
            "models": ["scoreq"],
            "attestation": _fake_provider_attestation(("scoreq",)),
        },
        {
            "models": ["utmos", "nisqa"],
            "attestation": _fake_provider_attestation(("utmos", "nisqa")),
        },
    ]


def test_worker_runtime_attestation_is_o_excl(tmp_path):
    attestation_path = tmp_path / "runtime-attestation.json"
    attestation_path.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        mos_worker._write_runtime_attestation(
            str(attestation_path), ("scoreq",), [{
                "models": ["scoreq"],
                "attestation": _fake_provider_attestation(("scoreq",)),
            }],
        )

    assert attestation_path.read_text(encoding="utf-8") == "sentinel\n"


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda payload: payload.update(asset_load_binding={"version": 1}), "binding"),
        (lambda payload: payload.update(scoring_execution_policy={"version": 1}), "scoring policy"),
        (lambda payload: payload.update(unexpected=True), "schema"),
        (
            lambda payload: payload["model_tuple_attestations"][0][
                "attestation"
            ].update(unexpected=True),
            "provider attestation schema",
        ),
        (
            lambda payload: payload["model_tuple_attestations"][0][
                "attestation"
            ].update(scoring_execution_policy={"version": 1}),
            "provider attestation scoring policy",
        ),
        (
            lambda payload: payload["model_tuple_attestations"][0][
                "attestation"
            ]["actual_providers"]["scoreq"].update(unexpected=True),
            "scoreq provider schema",
        ),
    ],
)
def test_parent_rejects_nonexact_or_legacy_worker_attestation(
    tmp_path, mutation, expected_error,
):
    path = tmp_path / "runtime-attestation.json"
    payload = {
        "protocol": "audio_filter_pipeline.step3_mos_worker.v6",
        "asset_load_binding": dict(mos_worker.ASSET_LOAD_BINDING),
        "scoring_execution_policy": mos_runtime._plain_json(
            mos_worker.MOS_SCORING_EXECUTION_POLICY,
        ),
        "worker_models": ["scoreq"],
        "model_tuple_attestations": [{
            "models": ["scoreq"],
            "attestation": _fake_provider_attestation(("scoreq",)),
        }],
    }
    mutation(payload)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        mos_gpu_job._read_runtime_attestation(str(path), ("scoreq",))


@pytest.mark.parametrize("model", ["urgent_pk", "nisqa", "utmos"])
@pytest.mark.parametrize("field", ["device", "parameters"])
@pytest.mark.parametrize("forged_value", ["not-cuda", "cuda-fallback-disabled"])
def test_parent_rejects_cuda_substring_device_attestation(
    tmp_path, model, field, forged_value,
):
    path = tmp_path / "runtime-attestation.json"
    attestation = _fake_provider_attestation((model,))
    attestation["actual_providers"][model][field] = forged_value
    payload = {
        "protocol": "audio_filter_pipeline.step3_mos_worker.v6",
        "asset_load_binding": dict(mos_worker.ASSET_LOAD_BINDING),
        "scoring_execution_policy": mos_runtime._plain_json(
            mos_worker.MOS_SCORING_EXECUTION_POLICY,
        ),
        "worker_models": [model],
        "model_tuple_attestations": [{
            "models": [model],
            "attestation": attestation,
        }],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    expected_digests = (
        {model: "4" * 64} if model in {"nisqa", "utmos"} else None
    )
    expected_urgent_assets = (
        dict(attestation["urgent_pk_assets"])
        if model == "urgent_pk" else None
    )

    with pytest.raises(ValueError, match=f"{model} provider values"):
        mos_gpu_job._read_runtime_attestation(
            str(path), (model,), expected_digests, expected_urgent_assets,
        )


@pytest.mark.parametrize("model", ["nisqa", "utmos"])
def test_parent_rejects_valid_but_unexpected_checkpoint_digest(tmp_path, model):
    path = tmp_path / "runtime-attestation.json"
    expected_digest = "4" * 64
    attestation = _fake_provider_attestation(
        (model,), {model: "5" * 64},
    )
    payload = {
        "protocol": "audio_filter_pipeline.step3_mos_worker.v6",
        "asset_load_binding": dict(mos_worker.ASSET_LOAD_BINDING),
        "scoring_execution_policy": mos_runtime._plain_json(
            mos_worker.MOS_SCORING_EXECUTION_POLICY,
        ),
        "worker_models": [model],
        "model_tuple_attestations": [{
            "models": [model],
            "attestation": attestation,
        }],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=f"{model} checkpoint digest mismatch"):
        mos_gpu_job._read_runtime_attestation(
            str(path), (model,), {model: expected_digest},
        )


def test_parent_checkpoint_gate_covers_selected_only_pool_request(tmp_path):
    task_path = tmp_path / "nisqa-tasks.jsonl"
    task_path.write_text(json.dumps({
        "ordinal": 0, "row": _row(0), "models": ["nisqa"],
        "scoring_policy_hash": "a" * 64,
    }) + "\n", encoding="utf-8")
    options = _checkpoint_model_options(tmp_path, "nisqa")

    def run_request(_gpu_id, request_path, _cwd):
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        assert request["nisqa_options"]["checkpoint_sha256"] == (
            options["checkpoint_sha256"]
        )
        subshard = request["subshards"][0]
        with open(subshard["output_path"], "x", encoding="utf-8") as output:
            output.write(json.dumps({
                "ordinal": 0, "scores": {"nisqa": 3.9},
            }) + "\n")
        _write_fake_worker_attestation(
            request, {"nisqa": "5" * 64},
        )

    pool = Step3MosScorerPool(
        [0], dataset_id="alpha", nisqa_options=options,
        request_runner=run_request,
    )

    with pytest.raises(ValueError, match="nisqa checkpoint digest mismatch"):
        list(pool.score_task_groups([{
            "models": ["nisqa"], "path": str(task_path), "row_count": 1,
        }]))


def test_parent_checkpoint_gate_covers_multi_tuple_subshards(tmp_path):
    scoreq_path = tmp_path / "scoreq-tasks.jsonl"
    checkpoint_path = tmp_path / "checkpoint-tasks.jsonl"
    scoreq_path.write_text(json.dumps({
        "ordinal": 0, "row": _row(0), "models": ["scoreq"],
        "scoring_policy_hash": "a" * 64,
    }) + "\n", encoding="utf-8")
    checkpoint_path.write_text(json.dumps({
        "ordinal": 1, "row": _row(1), "models": ["utmos", "nisqa"],
        "scoring_policy_hash": "b" * 64,
    }) + "\n", encoding="utf-8")

    def run_request(_gpu_id, request_path, _cwd):
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        assert [item["models"] for item in request["subshards"]] == [
            ["scoreq"], ["utmos", "nisqa"],
        ]
        for subshard in request["subshards"]:
            scores = (
                {"scoreq": 4.1}
                if subshard["models"] == ["scoreq"]
                else {"utmos": 4.0, "nisqa": 3.9}
            )
            with open(subshard["output_path"], "x", encoding="utf-8") as output:
                output.write(json.dumps({
                    "ordinal": subshard["start_ordinal"], "scores": scores,
                }) + "\n")
        _write_fake_worker_attestation(
            request, {"utmos": "5" * 64},
        )

    pool = Step3MosScorerPool(
        [0], dataset_id="alpha",
        scoreq_options=_tree_options(tmp_path, "scoreq", "1" * 64),
        nisqa_options=_checkpoint_model_options(tmp_path, "nisqa"),
        utmos_options=_checkpoint_model_options(tmp_path, "utmos"),
        request_runner=run_request,
    )

    with pytest.raises(ValueError, match="utmos checkpoint digest mismatch"):
        list(pool.score_task_groups([
            {"models": ["scoreq"], "path": str(scoreq_path), "row_count": 1},
            {
                "models": ["utmos", "nisqa"],
                "path": str(checkpoint_path), "row_count": 1,
            },
        ]))


def test_parent_checkpoint_gate_covers_persistent_fleet(tmp_path, monkeypatch):
    verified = _verified_pool_plan(tmp_path)
    task_path = tmp_path / "fleet-nisqa-tasks.jsonl"
    task_path.write_text(json.dumps({
        "ordinal": 0, "row": _row(0), "models": ["nisqa"],
        "scoring_policy_hash": "a" * 64,
    }) + "\n", encoding="utf-8")
    fleet = Step3MosWorkerFleet([0], verified_plan=verified)

    def run_request(_dataset_id, _gpu_id, request_path, _cwd):
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        expected_digest = request["nisqa_options"]["checkpoint_sha256"]
        assert expected_digest == verified.options_by_model["nisqa"][
            "checkpoint_sha256"
        ]
        subshard = request["subshards"][0]
        with open(subshard["output_path"], "x", encoding="utf-8") as output:
            output.write(json.dumps({
                "ordinal": 0, "scores": {"nisqa": 3.9},
            }) + "\n")
        wrong_digest = "0" * 64 if expected_digest != "0" * 64 else "1" * 64
        _write_fake_worker_attestation(
            request, {"nisqa": wrong_digest},
        )

    monkeypatch.setattr(fleet, "_run_request", run_request)
    try:
        with pytest.raises(ValueError, match="nisqa checkpoint digest mismatch"):
            list(fleet.for_dataset("alpha").score_task_groups([{
                "models": ["nisqa"], "path": str(task_path), "row_count": 1,
            }]))
    finally:
        fleet.close()


@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        ("checkpoint_sha256", "a" * 64),
        ("source_commit", "a" * 40),
        ("source_package_sha256", "a" * 64),
    ],
)
def test_parent_urgent_identity_gate_covers_selected_only_pool_request(
    tmp_path, field, forged_value,
):
    task_path = tmp_path / "urgent-tasks.jsonl"
    task_path.write_text(json.dumps({
        "ordinal": 0, "row": _row(0), "models": ["urgent_pk"],
        "scoring_policy_hash": "a" * 64,
    }) + "\n", encoding="utf-8")
    options = _urgent_options(tmp_path)

    def run_request(_gpu_id, request_path, _cwd):
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        expected_assets = {
            "checkpoint_sha256": request["urgent_options"]["expected_sha256"],
            "source_commit": request["urgent_options"]["expected_commit"],
            "source_package_sha256": "9" * 64,
        }
        forged_assets = {**expected_assets, field: forged_value}
        subshard = request["subshards"][0]
        with open(subshard["output_path"], "x", encoding="utf-8") as output:
            output.write(json.dumps({
                "ordinal": 0, "scores": {"urgent_pk": 3.7},
            }) + "\n")
        _write_fake_worker_attestation(
            request, urgent_pk_assets=forged_assets,
        )

    pool = Step3MosScorerPool(
        [0], dataset_id="alpha", urgent_options=options,
        request_runner=run_request,
    )

    with pytest.raises(ValueError, match=f"urgent_pk {field} mismatch"):
        list(pool.score_task_groups([{
            "models": ["urgent_pk"], "path": str(task_path), "row_count": 1,
        }]))


def test_parent_urgent_identity_gate_covers_multi_tuple_subshards(tmp_path):
    scoreq_path = tmp_path / "scoreq-tasks.jsonl"
    urgent_path = tmp_path / "urgent-tasks.jsonl"
    scoreq_path.write_text(json.dumps({
        "ordinal": 0, "row": _row(0), "models": ["scoreq"],
        "scoring_policy_hash": "a" * 64,
    }) + "\n", encoding="utf-8")
    urgent_path.write_text(json.dumps({
        "ordinal": 1, "row": _row(1), "models": ["urgent_pk"],
        "scoring_policy_hash": "b" * 64,
    }) + "\n", encoding="utf-8")

    def run_request(_gpu_id, request_path, _cwd):
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        assert [item["models"] for item in request["subshards"]] == [
            ["scoreq"], ["urgent_pk"],
        ]
        for subshard in request["subshards"]:
            scores = (
                {"scoreq": 4.1}
                if subshard["models"] == ["scoreq"] else {"urgent_pk": 3.7}
            )
            with open(subshard["output_path"], "x", encoding="utf-8") as output:
                output.write(json.dumps({
                    "ordinal": subshard["start_ordinal"], "scores": scores,
                }) + "\n")
        _write_fake_worker_attestation(request, urgent_pk_assets={
            "checkpoint_sha256": request["urgent_options"]["expected_sha256"],
            "source_commit": "a" * 40,
            "source_package_sha256": "9" * 64,
        })

    pool = Step3MosScorerPool(
        [0], dataset_id="alpha",
        scoreq_options=_tree_options(tmp_path, "scoreq", "1" * 64),
        urgent_options=_urgent_options(tmp_path), request_runner=run_request,
    )

    with pytest.raises(ValueError, match="urgent_pk source_commit mismatch"):
        list(pool.score_task_groups([
            {"models": ["scoreq"], "path": str(scoreq_path), "row_count": 1},
            {"models": ["urgent_pk"], "path": str(urgent_path), "row_count": 1},
        ]))


def test_parent_urgent_identity_gate_covers_persistent_fleet(tmp_path, monkeypatch):
    verified = _verified_pool_plan(tmp_path)
    task_path = tmp_path / "fleet-urgent-tasks.jsonl"
    task_path.write_text(json.dumps({
        "ordinal": 0, "row": _row(0), "models": ["urgent_pk"],
        "scoring_policy_hash": "a" * 64,
    }) + "\n", encoding="utf-8")
    fleet = Step3MosWorkerFleet([0], verified_plan=verified)

    def run_request(_dataset_id, _gpu_id, request_path, _cwd):
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        subshard = request["subshards"][0]
        with open(subshard["output_path"], "x", encoding="utf-8") as output:
            output.write(json.dumps({
                "ordinal": 0, "scores": {"urgent_pk": 3.7},
            }) + "\n")
        _write_fake_worker_attestation(request, urgent_pk_assets={
            "checkpoint_sha256": "a" * 64,
            "source_commit": request["urgent_options"]["expected_commit"],
            "source_package_sha256": "9" * 64,
        })

    monkeypatch.setattr(fleet, "_run_request", run_request)
    try:
        with pytest.raises(ValueError, match="urgent_pk checkpoint_sha256 mismatch"):
            list(fleet.for_dataset("alpha").score_task_groups([{
                "models": ["urgent_pk"], "path": str(task_path), "row_count": 1,
            }]))
    finally:
        fleet.close()


def test_parent_rejects_contradictory_scoreq_execution_mode(tmp_path):
    path = tmp_path / "runtime-attestation.json"
    attestation = _fake_provider_attestation(("scoreq",))
    attestation["actual_providers"]["scoreq"]["execution_mode"] = "cuda_only_ep"
    payload = {
        "protocol": "audio_filter_pipeline.step3_mos_worker.v6",
        "asset_load_binding": dict(mos_worker.ASSET_LOAD_BINDING),
        "scoring_execution_policy": mos_runtime._plain_json(
            mos_worker.MOS_SCORING_EXECUTION_POLICY,
        ),
        "worker_models": ["scoreq"],
        "model_tuple_attestations": [{
            "models": ["scoreq"],
            "attestation": attestation,
        }],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="scoreq provider"):
        mos_gpu_job._read_runtime_attestation(str(path), ("scoreq",))


def test_parent_rejects_oversized_worker_runtime_attestation(tmp_path):
    path = tmp_path / "runtime-attestation.json"
    path.write_bytes(b"{" + b" " * (1024 * 1024) + b"}\n")

    with pytest.raises(ValueError, match="exceeds JSONL file limit"):
        mos_gpu_job._read_runtime_attestation(str(path), ("scoreq",))


def test_worker_flushes_same_source_batch_before_row_limit_when_float32_bytes_exceed_cap(
    tmp_path, monkeypatch,
):
    interval_samples = mos_runtime._MAX_BATCH_FLOAT32_BYTES // 4 // 2
    rows = [_row(index) for index in range(2)]
    for row in rows:
        row["source_key"] = "shared-source"
        row["original_path"] = "/input/shared.wav"
        row["original_relative_path"] = "shared.wav"
        row["source_relative_path"] = "shared.wav"
        row["frames"] = interval_samples
        row["end_sample"] = interval_samples
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    request_path = tmp_path / "request.json"
    input_path.write_text("".join(
        json.dumps({"ordinal": index, "row": row}) + "\n" for index, row in enumerate(rows)
    ), encoding="utf-8")
    request_path.write_text(json.dumps(
        _worker_request(tmp_path, input_path, output_path, len(rows))
    ), encoding="utf-8")
    opens, batch_sizes = [], []

    class Runtime(_FakeRuntimeAttestation):
        def __init__(self, **_kwargs):
            pass
        def source_identity(self, row):
            return (row["source_relative_path"], row["source_sha256"])
        @contextmanager
        def open_source_group(self, row, _models):
            opens.append(self.source_identity(row))
            yield self
        def score_batch(self, batch):
            batch_sizes.append(len(batch))
            return [_scores() for _row in batch]

    monkeypatch.setattr(mos_worker, "Step3MosRuntime", Runtime)
    mos_worker.handle_request(str(request_path))

    output = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert opens == [("shared.wav", "a" * 64)]
    assert batch_sizes == [1, 1]
    assert [item["ordinal"] for item in output] == [0, 1]


def test_worker_exits_entered_group_once_when_source_switch_close_fails(tmp_path, monkeypatch):
    rows = [_row(0), _row(1)]
    rows[0]["source_key"] = "first"
    rows[1]["source_key"] = "second"
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    request_path = tmp_path / "request.json"
    input_path.write_text("".join(
        json.dumps({"ordinal": index, "row": row}) + "\n" for index, row in enumerate(rows)
    ), encoding="utf-8")
    request_path.write_text(json.dumps(
        _worker_request(tmp_path, input_path, output_path, len(rows))
    ), encoding="utf-8")
    exits = []

    class Group:
        def __init__(self, identity):
            self.identity = identity
        def __enter__(self):
            return self
        def __exit__(self, *_exc_info):
            exits.append(self.identity)
            raise RuntimeError("close failed")
        def score_batch(self, batch):
            return [_scores() for _row in batch]

    class Runtime(_FakeRuntimeAttestation):
        def __init__(self, **_kwargs):
            pass
        def source_identity(self, row):
            return row["source_key"]
        def open_source_group(self, row, _models):
            return Group(self.source_identity(row))

    monkeypatch.setattr(mos_worker, "Step3MosRuntime", Runtime)

    with pytest.raises(RuntimeError, match="close failed"):
        mos_worker.handle_request(str(request_path))

    assert exits == ["first"]


def test_worker_exits_entered_group_once_when_flush_fails(tmp_path, monkeypatch):
    row = _row(0)
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    request_path = tmp_path / "request.json"
    input_path.write_text(json.dumps({"ordinal": 0, "row": row}) + "\n", encoding="utf-8")
    request_path.write_text(json.dumps(
        _worker_request(tmp_path, input_path, output_path, 1)
    ), encoding="utf-8")
    exits = []

    class Group:
        def __enter__(self):
            return self
        def __exit__(self, *_exc_info):
            exits.append("group")
        def score_batch(self, _batch):
            raise RuntimeError("flush failed")

    class Runtime(_FakeRuntimeAttestation):
        def __init__(self, **_kwargs):
            pass
        def source_identity(self, _row):
            return "source"
        def open_source_group(self, _row, _models):
            return Group()

    monkeypatch.setattr(mos_worker, "Step3MosRuntime", Runtime)

    with pytest.raises(RuntimeError, match="flush failed"):
        mos_worker.handle_request(str(request_path))

    assert exits == ["group"]


class _FakeProcess:
    _next_pid = 1000

    def __init__(self, *, returncode=0, stdout=b"", stderr=b"", timeout=False, blocked=False):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._timeout = timeout
        self._blocked = blocked
        self.started = threading.Event()
        self._released = threading.Event()
        self.pid = self._next_pid
        type(self)._next_pid += 1

    def poll(self):
        if self._timeout and not self._released.is_set():
            return None
        return self.returncode if self._released.is_set() or not self._blocked else None

    def wait(self, timeout=None):
        self._released.set()
        return self.returncode

    def communicate(self, timeout=None):
        self.started.set()
        if self._timeout and timeout is not None:
            raise subprocess.TimeoutExpired("worker", timeout)
        if self._blocked and not self._released.is_set():
            self._released.wait(timeout=1.0)
        return self._stdout, self._stderr


def _popen(calls, *, duplicate=False):
    def popen(command, **kwargs):
        assert command[2].startswith("/")
        assert kwargs["shell"] is False
        assert kwargs["start_new_session"] is True
        request_path = command[2]
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        assert request["private_dir"] == kwargs["cwd"]
        assert request["protocol"] == mos_pool._PROTOCOL
        assert len(request["subshards"]) == 1
        subshard = request["subshards"][0]
        with open(subshard["input_path"], encoding="utf-8") as source:
            source_lines = source.readlines()
        calls.append((
            subshard,
            kwargs["env"]["CUDA_VISIBLE_DEVICES"],
            [json.loads(line)["ordinal"] for line in source_lines],
        ))
        with open(subshard["output_path"], "x", encoding="utf-8") as output:
            for line in source_lines:
                item = json.loads(line)
                output.write(json.dumps({"ordinal": item["ordinal"], "scores": _scores()}) + "\n")
                if duplicate:
                    output.write(json.dumps({"ordinal": item["ordinal"], "scores": _scores()}) + "\n")
        _write_fake_worker_attestation(request)
        return _FakeProcess()
    return popen


def test_pool_uses_balanced_contiguous_ordinals_and_preserves_input_order(tmp_path):
    calls = []
    pool = Step3MosScorerPool(
        [2, 5], dataset_id="alpha", popen=_popen(calls),
        verified_plan=_verified_pool_plan(tmp_path),
    )
    provenance_before = pool.provenance_payload()
    with pytest.raises(RuntimeError, match="before successful scoring"):
        pool.runtime_attestation()

    results = _score_rows(
        pool,
        (_row(index) for index in range(5)),
        ("scoreq", "urgent_pk", "dnsmos"),
    )

    expected_results = [{
        "ordinal": index,
        "row": _row(index),
        "scores": _scores(),
        "models": ["scoreq", "urgent_pk", "dnsmos"],
    } for index in range(5)]
    assert artifacts._canonical_json_bytes(results) == (
        artifacts._canonical_json_bytes(expected_results)
    )
    observed_calls = sorted(calls, key=lambda call: call[0]["start_ordinal"])
    assert [(call[0]["start_ordinal"], call[0]["end_ordinal"]) for call in observed_calls] == [(0, 5), (1, 4)]
    assert [call[1] for call in observed_calls] == ["2", "5"]
    assert len({call[0]["input_path"] for call in observed_calls}) == 2
    assert [call[2] for call in observed_calls] == [[0, 2, 4], [1, 3]]
    assert all(call[0]["input_path"].startswith("/") for call in calls)
    provenance = pool.provenance_payload()
    assert mos_pool._PROTOCOL == mos_worker._PROTOCOL == (
        "audio_filter_pipeline.step3_mos_worker.v6"
    )
    assert provenance["protocol"] == "audio_filter_pipeline.step3_mos_worker.v6"
    assert provenance["scorer_execution"] == "step3_mos_pool.v6"
    assert provenance["scoring_execution_policy"] == mos_runtime._plain_json(
        mos_worker.MOS_SCORING_EXECUTION_POLICY,
    )
    gpu_binding = provenance["asset_load_binding"]
    cpu_provenance = mos_runtime.Step3MosRuntime(
        options=mos_runtime.MosRuntimeOptions(),
    ).provenance_payload()
    assert gpu_binding == cpu_provenance["asset_load_binding"]
    assert gpu_binding == dict(mos_worker.ASSET_LOAD_BINDING)
    assert provenance["cpu_fallback"] is False
    assert provenance["provider_class"] == "cuda_subprocess"
    assert provenance["device_class"] == "gpu"
    for dynamic_field in (
        "gpu_ids", "worker_idle_timeout_seconds", "worker_hard_timeout_seconds",
        "worker_stderr_tail_bytes", "sharding", "worker_heartbeat_protocol",
    ):
        assert dynamic_field not in provenance
    assert len(provenance["worker_script"]["sha256"]) == 64
    assert provenance == provenance_before
    assert "runtime_attestation" not in provenance
    runtime_attestation = pool.runtime_attestation()
    assert runtime_attestation["provider_class"] == "cuda_subprocess"
    assert runtime_attestation["device_class"] == "gpu"
    assert runtime_attestation["gpu_ids"] == [2, 5]
    assert runtime_attestation["worker_idle_timeout_seconds"] == 900.0
    assert runtime_attestation["worker_hard_timeout_seconds"] is None
    assert runtime_attestation["worker_stderr_tail_bytes"] == 4_096
    runtime = runtime_attestation["runtime"]
    assert runtime["protocol"] == "audio_filter_pipeline.step3_mos_worker.v6"
    assert runtime["asset_load_binding"] == dict(mos_worker.ASSET_LOAD_BINDING)
    assert [worker["gpu_id"] for worker in runtime["workers"]] == [2, 5]
    assert all(
        worker["worker_models"] == ["scoreq", "urgent_pk", "dnsmos"]
        for worker in runtime["workers"]
    )


def test_pool_generator_return_value_is_complete_execution_attestation(tmp_path):
    calls = []
    pool = Step3MosScorerPool(
        [3], dataset_id="alpha", popen=_popen(calls),
        verified_plan=_verified_pool_plan(tmp_path),
    )
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "tasks.jsonl")
        with open(path, "x", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "ordinal": 0,
                "row": _row(0),
                "models": ["scoreq", "urgent_pk", "dnsmos"],
                "scoring_policy_hash": "a" * 64,
            }) + "\n")
        rows, returned = _consume_generator(pool.score_task_groups([{
            "models": ["scoreq", "urgent_pk", "dnsmos"],
            "path": path,
            "row_count": 1,
        }]))

    assert [item["ordinal"] for item in rows] == [0]
    assert returned == pool.runtime_attestation()
    assert set(returned) == {
        "provider_class", "device_class", "gpu_ids", "sharding",
        "worker_idle_timeout_seconds", "worker_hard_timeout_seconds",
        "worker_heartbeat_protocol", "worker_heartbeat_launcher_sha256",
        "worker_stderr_tail_bytes", "runtime",
    }
    assert returned["runtime"]["protocol"] == mos_pool._PROTOCOL
    assert returned["runtime"]["workers"][0]["worker_models"] == [
        "scoreq", "urgent_pk", "dnsmos",
    ]


def test_pool_empty_groups_returns_no_tasks_without_publishing_attestation():
    requests = []
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha", request_runner=lambda *_args: requests.append(True),
    )

    rows, returned = _consume_generator(pool.score_task_groups([]))

    assert rows == []
    assert returned["runtime"] == {"execution_state": "no_tasks"}
    assert returned["gpu_ids"] == []
    assert requests == []
    with pytest.raises(RuntimeError, match="before successful scoring"):
        pool.runtime_attestation()


def _digest_only_config():
    return SimpleNamespace(selection_policies={}, dataset_specs=lambda: {})


def test_step3_config_digest_ignores_physical_gpu_and_timeout_topology(tmp_path):
    verified = _verified_pool_plan(tmp_path)
    first = Step3MosSelectionStage(scorer=Step3MosScorerPool(
        [0], dataset_id="alpha", verified_plan=verified,
        worker_idle_timeout_seconds=10,
        worker_hard_timeout_seconds=100,
        worker_stderr_tail_bytes=1024,
    ))
    second = Step3MosSelectionStage(scorer=Step3MosScorerPool(
        [3, 7], dataset_id="alpha", verified_plan=verified,
        worker_idle_timeout_seconds=999,
        worker_hard_timeout_seconds=None,
        worker_stderr_tail_bytes=65536,
    ))

    assert first.config_digest(_digest_only_config()) == second.config_digest(
        _digest_only_config()
    )


def test_step3_config_digest_changes_for_asset_or_provider_class(tmp_path):
    first_verified = _verified_pool_plan(tmp_path, suffix="-first")
    second_verified = _verified_pool_plan(tmp_path, suffix="-second")
    pool_stage = Step3MosSelectionStage(scorer=Step3MosScorerPool(
        [0], dataset_id="alpha", verified_plan=first_verified,
    ))
    changed_asset_stage = Step3MosSelectionStage(scorer=Step3MosScorerPool(
        [0], dataset_id="alpha", verified_plan=second_verified,
    ))
    options = first_verified.as_mapping()["options_by_model"]
    in_process_stage = Step3MosSelectionStage(scorer=mos_runtime.Step3MosRuntime(
        options=mos_runtime.MosRuntimeOptions(
            use_gpu=True,
            scoreq_options=options["scoreq"],
            urgent_options=options["urgent_pk"],
            dnsmos_options=options["dnsmos"],
            nisqa_options=options["nisqa"],
            utmos_options=options["utmos"],
        ),
        verified_plan=first_verified,
    ))

    base_digest = pool_stage.config_digest(_digest_only_config())
    assert changed_asset_stage.config_digest(_digest_only_config()) != base_digest
    assert in_process_stage.config_digest(_digest_only_config()) != base_digest


def test_pool_rejects_unverified_model_before_reading_task_spool(tmp_path):
    started = []
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha",
        popen=lambda *_args, **_kwargs: started.append(True),
        scoreq_options={
            "model_dir": "/assets/scoreq",
            "verified_model_tree_sha256": "1" * 64,
        },
    )

    with pytest.raises(ValueError, match="subset of parent-verified"):
        pool.validate_model_tuples([("nisqa",)])
    with pytest.raises(ValueError, match="subset of parent-verified"):
        list(pool.score_task_groups([{
            "models": ["nisqa"],
            "path": str(tmp_path / "missing-task-spool.jsonl"),
            "row_count": 1,
        }]))

    assert started == []


def test_sqlite_lpt_matches_stable_python_reference_across_small_batches(
    tmp_path, monkeypatch,
):
    from audio_filter_pipeline.processors.stages_v3 import mos_lpt

    monkeypatch.setattr(mos_lpt, "_LPT_BATCH_MAX_ROWS", 2)
    groups = [(b"a", 9), (b"b", 9), (b"c", 4), (b"a", 3), (b"d", 1)]
    planner = mos_lpt._DiskBackedLptPlanner(
        str(tmp_path / "lpt.sqlite3"), len(groups), 3,
    )
    try:
        payloads = []
        for ordinal, (identity, weight) in enumerate(groups):
            payload = f"task-{ordinal}".encode()
            payloads.append(payload)
            planner.add(
                ordinal=ordinal, identity=identity, weight=weight, payload=payload,
            )
        assert planner.finalize() == 3
        observed = []
        records = [
            (index, identity, payloads[index])
            for index, (identity, _weight) in enumerate(groups)
        ]
        for start in range(0, len(records), 2):
            observed.extend(planner.assignments(records[start:start + 2]))
    finally:
        planner.close()

    weights = {b"a": 12, b"b": 9, b"c": 4, b"d": 1}
    loads = [0, 0, 0]
    expected_by_identity = {}
    for identity, weight in sorted(weights.items(), key=lambda item: item[1], reverse=True):
        worker = min(range(3), key=lambda index: (loads[index], index))
        expected_by_identity[identity] = worker
        loads[worker] += weight
    assert observed == [expected_by_identity[identity] for identity, _weight in groups]
    assert not hasattr(mos_pool, "_LPT_METADATA_MAX_BYTES")


def test_sqlite_lpt_rejects_duplicate_ordinal_before_starting_worker(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.mos_lpt import _DiskBackedLptPlanner

    planner = _DiskBackedLptPlanner(str(tmp_path / "lpt.sqlite3"), 2, 1)
    try:
        planner.add(ordinal=0, identity=b"a", weight=1, payload=b"first")
        planner.add(ordinal=0, identity=b"b", weight=1, payload=b"duplicate")
        with pytest.raises(ValueError, match="duplicate"):
            planner.finalize()
        for operation in (
            lambda: planner.add(
                ordinal=1, identity=b"c", weight=1, payload=b"third",
            ),
            planner.finalize,
            lambda: planner.assignments([(0, b"a", b"first")]),
        ):
            with pytest.raises(RuntimeError, match="poisoned"):
                operation()
    finally:
        planner.close()


def test_sqlite_lpt_uses_byte_bounded_batches_and_binds_second_scan(
    tmp_path, monkeypatch,
):
    from audio_filter_pipeline.processors.stages_v3 import mos_lpt

    real_flush = mos_lpt._DiskBackedLptPlanner._flush
    flush_sizes = []

    def flush(planner):
        flush_sizes.append((len(planner._pending), planner._pending_bytes))
        return real_flush(planner)

    monkeypatch.setattr(mos_lpt, "_LPT_BATCH_MAX_BYTES", 220)
    monkeypatch.setattr(mos_lpt._DiskBackedLptPlanner, "_flush", flush)
    planner = mos_lpt._DiskBackedLptPlanner(
        str(tmp_path / "lpt.sqlite3"), 2, 1,
    )
    try:
        planner.add(ordinal=0, identity=b"a", weight=1, payload=b"x" * 30)
        planner.add(ordinal=1, identity=b"b", weight=1, payload=b"y" * 30)
        assert planner.finalize() == 1
        assert planner.assignments([(0, b"a", b"x" * 30)]) == [0]
        with pytest.raises(ValueError, match="changed between LPT scans"):
            planner.assignments([(1, b"b", b"changed")])
    finally:
        planner.close()

    assert [rows for rows, _size in flush_sizes if rows] == [1, 1]
    assert all(size <= 220 for rows, size in flush_sizes if rows)


def test_sqlite_lpt_rejects_missing_ordinal_and_uint64_overflow(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.mos_lpt import _DiskBackedLptPlanner

    incomplete = _DiskBackedLptPlanner(str(tmp_path / "incomplete.sqlite3"), 2, 1)
    try:
        incomplete.add(ordinal=0, identity=b"a", weight=1, payload=b"first")
        with pytest.raises(ValueError, match="ordinals are incomplete"):
            incomplete.finalize()
        with pytest.raises(RuntimeError, match="poisoned"):
            incomplete.add(ordinal=1, identity=b"b", weight=1, payload=b"second")
    finally:
        incomplete.close()

    overflow = _DiskBackedLptPlanner(str(tmp_path / "overflow.sqlite3"), 2, 1)
    try:
        overflow.add(
            ordinal=0, identity=b"a", weight=(1 << 64) - 1, payload=b"first",
        )
        overflow.add(ordinal=1, identity=b"a", weight=1, payload=b"second")
        with pytest.raises(ValueError, match="exceeds uint64"):
            overflow.finalize()
        with pytest.raises(RuntimeError, match="poisoned"):
            overflow.assignments([(0, b"a", b"first")])
    finally:
        overflow.close()


def test_sqlite_lpt_flush_failure_poison_is_terminal(tmp_path, monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import mos_lpt

    monkeypatch.setattr(mos_lpt, "_LPT_BATCH_MAX_ROWS", 1)
    planner = mos_lpt._DiskBackedLptPlanner(
        str(tmp_path / "flush-failure.sqlite3"), 2, 1,
    )
    try:
        planner._connection.execute(
            "CREATE TRIGGER reject_tasks BEFORE INSERT ON task_assignments "
            "BEGIN SELECT RAISE(ABORT, 'forced flush failure'); END"
        )
        planner.add(ordinal=0, identity=b"a", weight=1, payload=b"first")
        with pytest.raises(sqlite3.IntegrityError, match="forced flush failure"):
            planner.add(ordinal=1, identity=b"b", weight=1, payload=b"second")
        for operation in (
            lambda: planner.add(
                ordinal=1, identity=b"b", weight=1, payload=b"second",
            ),
            planner.finalize,
            lambda: planner.assignments([(0, b"a", b"first")]),
        ):
            with pytest.raises(RuntimeError, match="poisoned"):
                operation()
    finally:
        planner.close()


def test_sqlite_result_store_orders_rows_with_bounded_flushes(tmp_path, monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import mos_lpt

    real_flush = mos_lpt._DiskBackedResultStore._flush
    flush_sizes = []

    def flush(store):
        flush_sizes.append((len(store._pending), store._pending_bytes))
        return real_flush(store)

    monkeypatch.setattr(mos_lpt, "_LPT_BATCH_MAX_ROWS", 2)
    monkeypatch.setattr(mos_lpt._DiskBackedResultStore, "_flush", flush)
    store = mos_lpt._DiskBackedResultStore(
        str(tmp_path / "results.sqlite3"), 3,
    )
    try:
        store.add(2, b"two")
        store.add(0, b"zero")
        store.add(1, b"one")
        store.finalize()
        assert list(store.iter_payloads()) == [
            (0, b"zero"), (1, b"one"), (2, b"two"),
        ]
    finally:
        store.close()

    assert [rows for rows, _size in flush_sizes if rows] == [2, 1]
    assert all(rows <= 2 for rows, _size in flush_sizes)


def test_sqlite_result_store_flush_failure_poison_is_terminal(
    tmp_path, monkeypatch,
):
    from audio_filter_pipeline.processors.stages_v3 import mos_lpt

    monkeypatch.setattr(mos_lpt, "_LPT_BATCH_MAX_ROWS", 1)
    store = mos_lpt._DiskBackedResultStore(
        str(tmp_path / "result-flush-failure.sqlite3"), 2,
    )
    try:
        store._connection.execute(
            "CREATE TRIGGER reject_results BEFORE INSERT ON results "
            "BEGIN SELECT RAISE(ABORT, 'forced result flush failure'); END"
        )
        store.add(0, b"first")
        with pytest.raises(sqlite3.IntegrityError, match="forced result flush failure"):
            store.add(1, b"second")
        for operation in (
            lambda: store.add(1, b"second"),
            store.finalize,
            lambda: list(store.iter_payloads()),
        ):
            with pytest.raises(RuntimeError, match="poisoned"):
                operation()
    finally:
        store.close()


def test_sqlite_result_store_failures_are_terminal(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.mos_lpt import (
        _DiskBackedResultStore,
    )

    duplicate = _DiskBackedResultStore(str(tmp_path / "duplicate.sqlite3"), 2)
    try:
        duplicate.add(0, b"first")
        duplicate.add(0, b"duplicate")
        with pytest.raises(ValueError, match="duplicate"):
            duplicate.finalize()
        for operation in (
            lambda: duplicate.add(1, b"second"),
            duplicate.finalize,
            lambda: list(duplicate.iter_payloads()),
        ):
            with pytest.raises(RuntimeError, match="poisoned"):
                operation()
    finally:
        duplicate.close()

    incomplete = _DiskBackedResultStore(str(tmp_path / "incomplete-result.sqlite3"), 2)
    try:
        incomplete.add(0, b"first")
        with pytest.raises(ValueError, match="coverage"):
            incomplete.finalize()
        with pytest.raises(RuntimeError, match="poisoned"):
            list(incomplete.iter_payloads())
    finally:
        incomplete.close()


def test_sqlite_result_store_consumer_cancel_propagates_cursor_cleanup_failure(
    tmp_path,
):
    from audio_filter_pipeline.processors.stages_v3.mos_lpt import (
        _DiskBackedResultStore,
    )

    class Cursor:
        def __init__(self):
            self._rows = iter(((0, b"zero"),))

        def fetchone(self):
            return next(self._rows, None)

        @staticmethod
        def close():
            raise RuntimeError("cursor close failed")

    class Connection:
        @staticmethod
        def execute(_query):
            return Cursor()

        @staticmethod
        def close():
            pass

    store = _DiskBackedResultStore(str(tmp_path / "cursor.sqlite3"), 1)
    store.add(0, b"zero")
    store.finalize()
    store._connection.close()
    store._connection = Connection()
    payloads = iter(store.iter_payloads())
    assert next(payloads) == (0, b"zero")

    with pytest.raises(RuntimeError, match="cursor close failed"):
        payloads.close()
    with pytest.raises(RuntimeError, match="poisoned"):
        list(store.iter_payloads())
    store.close()


@pytest.mark.parametrize("kind", ("planner", "result"))
def test_sqlite_owned_state_close_failure_keeps_retryable_ownership(tmp_path, kind):
    from audio_filter_pipeline.processors.stages_v3.mos_lpt import (
        _DiskBackedLptPlanner,
        _DiskBackedResultStore,
    )

    state = (
        _DiskBackedLptPlanner(str(tmp_path / "planner.sqlite3"), 1, 1)
        if kind == "planner"
        else _DiskBackedResultStore(str(tmp_path / "result.sqlite3"), 1)
    )
    real_connection = state._connection

    class FailOnceConnection:
        def __init__(self):
            self.calls = 0

        def close(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("SQLite close failed once")
            real_connection.close()

    proxy = FailOnceConnection()
    state._connection = proxy

    with pytest.raises(RuntimeError, match="close failed once"):
        state.close()
    assert state._connection is proxy
    assert state._poisoned is True

    state.close()
    assert proxy.calls == 2
    assert state._connection is None


@pytest.mark.parametrize("primary", (ValueError("business failed"), GeneratorExit()))
def test_sqlite_context_cleanup_preserves_business_error_and_replaces_generator_exit(
    tmp_path, primary,
):
    from audio_filter_pipeline.processors.stages_v3.mos_lpt import _DiskBackedResultStore

    state = _DiskBackedResultStore(str(tmp_path / "result.sqlite3"), 1)
    real_connection = state._connection

    class FailOnceConnection:
        calls = 0

        def close(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("SQLite context close failed")
            real_connection.close()

    proxy = FailOnceConnection()
    state._connection = proxy
    try:
        with pytest.raises(type(primary)) as raised:
            with state:
                raise primary
        if isinstance(primary, GeneratorExit):
            pytest.fail("GeneratorExit must be replaced by cleanup failure")
        assert str(raised.value) == "business failed"
        assert any("context close failed" in note for note in raised.value.__notes__)
    except RuntimeError as error:
        assert isinstance(primary, GeneratorExit)
        assert "context close failed" in str(error)
    finally:
        state.close()


def test_pool_rejects_nonincreasing_ordinals_across_tuple_parts_before_worker(
    tmp_path,
):
    models = ["scoreq"]
    paths = []
    for index, ordinal in enumerate((1, 0)):
        path = tmp_path / f"part-{index}.jsonl"
        path.write_text(json.dumps({
            "ordinal": ordinal,
            "row": _row(ordinal),
            "models": models,
            "scoring_policy_hash": "a" * 64,
        }) + "\n", encoding="utf-8")
        paths.append(path)
    started = []
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha",
        scoreq_options=_tree_options(tmp_path, "scoreq", "1" * 64),
        request_runner=lambda *_args: started.append(True),
    )

    with pytest.raises(ValueError, match="part ordinals must be strictly increasing"):
        list(pool.score_task_groups([
            {"models": models, "path": str(path), "row_count": 1}
            for path in paths
        ]))

    assert started == []


@pytest.mark.parametrize("failed_scan", [1, 2])
def test_pool_lpt_scan_propagates_payload_iterator_cleanup_failure(
    tmp_path, monkeypatch, failed_scan,
):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps({
        "ordinal": 0,
        "row": _row(0),
        "models": ["scoreq"],
        "scoring_policy_hash": "a" * 64,
    }) + "\n", encoding="utf-8")
    real_iter = mos_gpu_job._shared_iter_bounded_jsonl
    scan_count = 0
    close_calls = []

    class CloseTrackingIterator:
        def __init__(self, iterator, scan):
            self._iterator = iterator
            self._scan = scan

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._iterator)

        def close(self):
            close_calls.append(self._scan)
            self._iterator.close()
            if self._scan == failed_scan:
                raise RuntimeError(f"scan {self._scan} close failed")

    def track_scans(source, label, **kwargs):
        nonlocal scan_count
        iterator = real_iter(source, label, **kwargs)
        if label != "Step3 tuple task spool":
            return iterator
        scan_count += 1
        return CloseTrackingIterator(iterator, scan_count)

    started = []
    monkeypatch.setattr(mos_gpu_job, "_shared_iter_bounded_jsonl", track_scans)
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha",
        scoreq_options=_tree_options(tmp_path, "scoreq", "1" * 64),
        request_runner=lambda *_args: started.append(True),
    )

    with pytest.raises(RuntimeError, match=f"scan {failed_scan} close failed"):
        list(pool.score_task_groups([{
            "models": ["scoreq"], "path": str(path), "row_count": 1,
        }]))

    assert close_calls == list(range(1, failed_scan + 1))
    assert started == []


@pytest.mark.parametrize("failed_scan", [1, 2])
def test_pool_lpt_scan_propagates_owned_stream_close_failure(
    tmp_path, monkeypatch, failed_scan,
):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps({
        "ordinal": 0,
        "row": _row(0),
        "models": ["scoreq"],
        "scoring_policy_hash": "a" * 64,
    }) + "\n", encoding="utf-8")
    real_open = mos_gpu_job._open_regular_stream
    scan_count = 0
    opened = []

    class CloseFailingStream:
        def __init__(self, stream, scan):
            self._stream = stream
            self._scan = scan
            self.close_calls = 0

        def __getattr__(self, name):
            return getattr(self._stream, name)

        def close(self):
            self.close_calls += 1
            self._stream.close()
            if self._scan == failed_scan:
                raise RuntimeError(f"scan {self._scan} stream close failed")

    def open_stream(stream_path, label):
        nonlocal scan_count
        stream, signature = real_open(stream_path, label)
        if label == "Step3 tuple task spool":
            scan_count += 1
            stream = CloseFailingStream(stream, scan_count)
            opened.append(stream)
        return stream, signature

    started = []
    monkeypatch.setattr(mos_gpu_job, "_open_regular_stream", open_stream)
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha",
        scoreq_options=_tree_options(tmp_path, "scoreq", "1" * 64),
        request_runner=lambda *_args: started.append(True),
    )

    with pytest.raises(
        RuntimeError, match=f"scan {failed_scan} stream close failed",
    ):
        list(pool.score_task_groups([{
            "models": ["scoreq"], "path": str(path), "row_count": 1,
        }]))

    assert len(opened) == failed_scan
    assert all(stream.close_calls == 1 for stream in opened)
    assert started == []


def test_pool_rejects_same_content_inode_replacement_between_lpt_scans(
    tmp_path, monkeypatch,
):
    from audio_filter_pipeline.processors.stages_v3 import mos_lpt

    path = tmp_path / "tasks.jsonl"
    payload = json.dumps({
        "ordinal": 0,
        "row": _row(0),
        "models": ["scoreq"],
        "scoring_policy_hash": "a" * 64,
    }) + "\n"
    path.write_text(payload, encoding="utf-8")
    held = tmp_path / "held.jsonl"
    original_finalize = mos_lpt._DiskBackedLptPlanner.finalize

    def finalize(planner):
        workers = original_finalize(planner)
        path.rename(held)
        path.write_text(payload, encoding="utf-8")
        assert path.stat().st_ino != held.stat().st_ino
        return workers

    started = []
    monkeypatch.setattr(mos_lpt._DiskBackedLptPlanner, "finalize", finalize)
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha",
        scoreq_options=_tree_options(tmp_path, "scoreq", "1" * 64),
        request_runner=lambda *_args: started.append(True),
    )

    with pytest.raises(ValueError, match="changed between LPT scans"):
        list(pool.score_task_groups([{
            "models": ["scoreq"], "path": str(path), "row_count": 1,
        }]))

    assert started == []


def test_pool_rejects_pathname_replacement_during_second_lpt_scan(
    tmp_path, monkeypatch,
):
    path = tmp_path / "tasks.jsonl"
    payload = "".join(
        json.dumps({
            "ordinal": ordinal,
            "row": _row(ordinal),
            "models": ["scoreq"],
            "scoring_policy_hash": "a" * 64,
        }) + "\n"
        for ordinal in range(2)
    )
    path.write_text(payload, encoding="utf-8")
    held = tmp_path / "held.jsonl"
    real_iter = mos_gpu_job._shared_iter_bounded_jsonl
    scan_count = 0

    def replace_during_second_scan(source, label, **kwargs):
        nonlocal scan_count
        iterator = real_iter(source, label, **kwargs)
        if label != "Step3 tuple task spool":
            return iterator
        scan_count += 1
        current_scan = scan_count

        def values():
            for index, item in enumerate(iterator):
                yield item
                if current_scan == 2 and index == 0:
                    path.rename(held)
                    path.write_text(payload, encoding="utf-8")

        return values()

    started = []
    monkeypatch.setattr(
        mos_gpu_job, "_shared_iter_bounded_jsonl", replace_during_second_scan,
    )
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha",
        scoreq_options=_tree_options(tmp_path, "scoreq", "1" * 64),
        request_runner=lambda *_args: started.append(True),
    )

    with pytest.raises(ValueError, match="changed while being read|pathname changed"):
        list(pool.score_task_groups([{
            "models": ["scoreq"], "path": str(path), "row_count": 2,
        }]))

    assert scan_count == 2
    assert started == []


def test_pool_result_collection_opens_only_one_physical_iterator_at_a_time(
    tmp_path, monkeypatch,
):
    row_count = 12
    task_path = tmp_path / "tasks.jsonl"
    task_path.write_text("".join(
        json.dumps({
            "ordinal": ordinal,
            "row": _row(ordinal),
            "models": ["scoreq"],
            "scoring_policy_hash": "a" * 64,
        }) + "\n"
        for ordinal in range(row_count)
    ), encoding="utf-8")
    active = 0
    peak = 0
    iterators = []

    class TrackingResultIterator:
        def __init__(self, values):
            nonlocal active, peak
            self._values = iter(values)
            self.fd = os.open(os.devnull, os.O_RDONLY)
            self.closed = False
            active += 1
            peak = max(peak, active)
            iterators.append(self)

        def __iter__(self):
            return self

        def __next__(self):
            os.fstat(self.fd)
            return next(self._values)

        def close(self):
            nonlocal active
            if not self.closed:
                os.close(self.fd)
                self.closed = True
                active -= 1

    def run_request(_gpu_id, request_path, _cwd):
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        for subshard in request["subshards"]:
            Path(subshard["output_path"]).write_text("placeholder\n", encoding="utf-8")
        _write_fake_worker_attestation(request)

    def read_result(
        _path, _input_path, _input_sha256,
        start, stop, ordinals=None, models=("scoreq",),
    ):
        assert ordinals is None
        expected = range(start, stop)
        return TrackingResultIterator([
            {
                "ordinal": ordinal,
                "row": _row(ordinal),
                "scores": {"scoreq": 4.1},
            }
            for ordinal in expected
        ])

    monkeypatch.setattr(mos_gpu_job, "_JSONL_MAX_FILE_BYTES", 1000)
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha",
        scoreq_options=_tree_options(tmp_path, "scoreq", "1" * 64),
        request_runner=run_request,
    )
    monkeypatch.setattr(
        mos_gpu_job._OwnedGpuScoringJob, "_read_result",
        staticmethod(read_result),
    )

    output = list(pool.score_task_groups([{
        "models": ["scoreq"], "path": str(task_path), "row_count": row_count,
    }]))

    assert [item["ordinal"] for item in output] == list(range(row_count))
    assert len(iterators) == row_count
    assert peak == 1
    assert active == 0
    assert all(iterator.closed for iterator in iterators)


def test_pool_lpt_aggregates_source_weight_across_physical_tuple_parts(tmp_path):
    models = ["scoreq"]
    rows = [_row(0), _row(1), _row(2)]
    rows[0]["end_sample"] = 100
    rows[2].update({
        field: rows[0][field]
        for field in mos_runtime._SOURCE_IDENTITY_FIELDS
    })
    rows[2]["original_path"] = rows[0]["original_path"]
    rows[2]["end_sample"] = 100
    rows[1]["end_sample"] = 150
    paths = []
    for part_index, ordinals in enumerate(((0, 1), (2,))):
        path = tmp_path / f"part-{part_index}.jsonl"
        path.write_text("".join(json.dumps({
            "ordinal": ordinal,
            "row": rows[ordinal],
            "models": models,
            "scoring_policy_hash": "a" * 64,
        }) + "\n" for ordinal in ordinals), encoding="utf-8")
        paths.append(path)
    worker_ordinals = []

    def run_request(_gpu_id, request_path, _cwd):
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        assigned = []
        for subshard in request["subshards"]:
            values = [
                json.loads(line)
                for line in Path(subshard["input_path"]).read_text(
                    encoding="utf-8",
                ).splitlines()
            ]
            assigned.extend(item["ordinal"] for item in values)
            with open(subshard["output_path"], "x", encoding="utf-8") as output:
                for item in values:
                    output.write(json.dumps({
                        "ordinal": item["ordinal"], "scores": {"scoreq": 4.1},
                    }) + "\n")
        worker_ordinals.append(assigned)
        _write_fake_worker_attestation(request)

    pool = Step3MosScorerPool(
        [0, 1], dataset_id="alpha",
        scoreq_options=_tree_options(tmp_path, "scoreq", "1" * 64),
        request_runner=run_request,
    )
    output = list(pool.score_task_groups([
        {"models": models, "path": str(paths[0]), "row_count": 2},
        {"models": models, "path": str(paths[1]), "row_count": 1},
    ]))

    assert sorted(worker_ordinals) == [[0, 2], [1]]
    assert [item["ordinal"] for item in output] == [0, 1, 2]


def test_pool_rolls_sparse_worker_input_and_sidecar_at_small_physical_cap(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(mos_gpu_job, "_JSONL_MAX_FILE_BYTES", 3_000)
    real_encode = mos_gpu_job._encode_json_line

    def encode(value, label):
        payload = real_encode(value, label)
        if label == "MOS sparse ordinal sidecar row":
            assert len(payload) <= 1_000
            payload = payload[:-1] + b" " * (1_000 - len(payload)) + b"\n"
        return payload

    monkeypatch.setattr(mos_gpu_job, "_encode_json_line", encode)
    path = tmp_path / "tasks.jsonl"
    path.write_text("".join(json.dumps({
        "ordinal": ordinal,
        "row": _row(ordinal),
        "models": ["scoreq"],
        "scoring_policy_hash": "a" * 64,
    }) + "\n" for ordinal in range(8)), encoding="utf-8")
    observed = []

    def run_request(_gpu_id, request_path, _cwd):
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        for subshard in request["subshards"]:
            input_path = Path(subshard["input_path"])
            input_rows = [
                json.loads(line)
                for line in input_path.read_text(encoding="utf-8").splitlines()
            ]
            item = {
                "input_size": input_path.stat().st_size,
                "input_ordinals": [value["ordinal"] for value in input_rows],
                "sidecar_size": None,
                "sidecar_ordinals": None,
            }
            if "ordinal_sidecar_path" in subshard:
                sidecar_path = Path(subshard["ordinal_sidecar_path"])
                sidecar_rows = [
                    json.loads(line)
                    for line in sidecar_path.read_text(encoding="utf-8").splitlines()
                ]
                item.update({
                    "sidecar_size": sidecar_path.stat().st_size,
                    "sidecar_ordinals": [value["ordinal"] for value in sidecar_rows],
                })
            observed.append(item)
            with open(subshard["output_path"], "x", encoding="utf-8") as output:
                for value in input_rows:
                    output.write(json.dumps({
                        "ordinal": value["ordinal"], "scores": {"scoreq": 4.1},
                    }) + "\n")
        _write_fake_worker_attestation(request)

    pool = Step3MosScorerPool(
        [0, 1], dataset_id="alpha",
        scoreq_options=_tree_options(tmp_path, "scoreq", "1" * 64),
        request_runner=run_request,
    )
    output = list(pool.score_task_groups([{
        "models": ["scoreq"], "path": str(path), "row_count": 8,
    }]))

    sparse = [item for item in observed if item["sidecar_size"] is not None]
    assert len(observed) == 4
    assert len(sparse) == 2
    assert all(0 < item["input_size"] <= 3_000 for item in observed)
    assert all(item["sidecar_size"] == 3_000 for item in sparse)
    assert all(
        item["input_ordinals"] == item["sidecar_ordinals"]
        for item in sparse
    )
    assert [item["ordinal"] for item in output] == list(range(8))


def test_parent_tuple_spool_readers_use_dedicated_file_budget(tmp_path, monkeypatch):
    observed = []
    path = tmp_path / "tasks.jsonl"
    path.write_bytes(b"")

    def iter_jsonl(source, label, *, max_line_bytes, max_file_bytes):
        observed.append({
            "source": source,
            "label": label,
            "max_line_bytes": max_line_bytes,
            "max_file_bytes": max_file_bytes,
        })
        return iter(())

    monkeypatch.setattr(mos_gpu_job, "_shared_iter_bounded_jsonl", iter_jsonl)
    monkeypatch.setattr(mos_runtime, "iter_bounded_jsonl", iter_jsonl)

    assert list(mos_gpu_job._iter_tuple_task_spool(str(path))) == []
    assert list(mos_runtime.Step3MosRuntime._iter_task_group(
        str(path), ("scoreq",), 0,
    )) == []
    assert observed[0]["source"] == str(path)
    assert hasattr(observed[1]["source"], "read")
    assert [
        (item["label"], item["max_line_bytes"], item["max_file_bytes"])
        for item in observed
    ] == [(
        "Step3 tuple task spool",
        mos_pool._JSONL_MAX_LINE_BYTES,
        512 * 1024 * 1024,
    )] * 2


def test_pool_dispatches_multiple_model_tuple_subshards_in_one_gpu_request(tmp_path):
    first_path = tmp_path / "scoreq-tasks.jsonl"
    second_path = tmp_path / "utmos-nisqa-tasks.jsonl"
    first_row, second_row = _row(0), _row(1)
    first_path.write_text(json.dumps({
        "ordinal": 0, "row": first_row, "models": ["scoreq"],
        "scoring_policy_hash": "a" * 64,
    }) + "\n", encoding="utf-8")
    second_path.write_text(json.dumps({
        "ordinal": 1, "row": second_row, "models": ["utmos", "nisqa"],
        "scoring_policy_hash": "b" * 64,
    }) + "\n", encoding="utf-8")
    requests = []

    def popen(command, **kwargs):
        request = json.loads(Path(command[2]).read_text(encoding="utf-8"))
        requests.append((request, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        for subshard in request["subshards"]:
            scores = ({"scoreq": 4.1} if subshard["models"] == ["scoreq"]
                      else {"utmos": 4.0, "nisqa": 3.9})
            with open(subshard["input_path"], encoding="utf-8") as source:
                rows = [json.loads(line) for line in source]
            with open(subshard["output_path"], "x", encoding="utf-8") as output:
                for item in rows:
                    output.write(json.dumps({
                        "ordinal": item["ordinal"], "scores": scores,
                    }) + "\n")
        _write_fake_worker_attestation(request)
        return _FakeProcess()

    scoreq_options = _tree_options(tmp_path, "scoreq", "1" * 64)
    pool = Step3MosScorerPool(
        [7], dataset_id="alpha", scoreq_options=scoreq_options,
        utmos_options=_checkpoint_model_options(tmp_path, "utmos"),
        nisqa_options=_checkpoint_model_options(tmp_path, "nisqa"), popen=popen,
    )
    output = list(pool.score_task_groups([
        {"models": ["scoreq"], "path": str(first_path), "row_count": 1},
        {"models": ["utmos", "nisqa"], "path": str(second_path), "row_count": 1},
    ]))

    assert len(requests) == 1
    assert requests[0][1] == "7"
    request = requests[0][0]
    assert [item["models"] for item in request["subshards"]] == [
        ["scoreq"], ["utmos", "nisqa"],
    ]
    assert {
        model
        for subshard in request["subshards"]
        for model in subshard["models"]
    } == {"scoreq", "nisqa", "utmos"}
    assert {
        model
        for model, value in {
            "scoreq": request["scoreq_options"],
            "urgent_pk": request["urgent_options"],
            "dnsmos": request["dnsmos_options"],
            "nisqa": request["nisqa_options"],
            "utmos": request["utmos_options"],
        }.items()
        if value is not None
    } == {"scoreq", "nisqa", "utmos"}
    assert [item["ordinal"] for item in output] == [0, 1]
    assert [item["models"] for item in output] == [
        ["scoreq"], ["utmos", "nisqa"],
    ]


def test_pool_closes_shard_iterators_when_canonical_consumer_cancels(
    tmp_path, monkeypatch,
):
    class TrackingResultIterator:
        def __init__(self, values):
            self._values = iter(values)
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._values)

        def close(self):
            self.closed = True

    task_path = tmp_path / "tasks.jsonl"
    task_path.write_text("".join(
        json.dumps({
            "ordinal": ordinal,
            "row": _row(ordinal),
            "models": ["scoreq", "urgent_pk", "dnsmos"],
            "scoring_policy_hash": "a" * 64,
        }) + "\n"
        for ordinal in range(2)
    ), encoding="utf-8")
    calls = []
    pool = Step3MosScorerPool(
        [0, 1], dataset_id="alpha", popen=_popen(calls),
        **_default_pool_options(),
    )
    result_iterators = []

    def read_result(
        path, input_path, input_sha256,
        start, stop, ordinals=None, models=mos_pool._MODELS,
    ):
        expected = range(start, stop) if ordinals is None else ordinals
        values = [
            {"ordinal": ordinal, "row": _row(ordinal), "scores": _scores()}
            for ordinal in expected
        ]
        iterator = TrackingResultIterator(values)
        result_iterators.append(iterator)
        return iterator

    monkeypatch.setattr(
        mos_gpu_job._OwnedGpuScoringJob, "_read_result",
        staticmethod(read_result),
    )
    stream = pool.score_task_groups([{
        "models": ["scoreq", "urgent_pk", "dnsmos"],
        "path": str(task_path),
        "row_count": 2,
    }])

    next(stream)
    assert result_iterators
    assert all(iterator.closed for iterator in result_iterators)
    stream.close()

    assert all(iterator.closed for iterator in result_iterators)
    with pytest.raises(RuntimeError, match="before successful scoring"):
        pool.runtime_attestation()


def test_pool_provenance_records_scoreq_and_dnsmos_tree_digests(tmp_path):
    scoreq_dir = tmp_path / "scoreq"
    dnsmos_dir = tmp_path / "dnsmos"
    scoreq_dir.mkdir()
    dnsmos_dir.mkdir()
    (scoreq_dir / "model.onnx").write_bytes(b"scoreq-model")
    (dnsmos_dir / "model.onnx").write_bytes(b"dnsmos-model")
    scoreq_digest = mos_pool.local_model_tree_sha256(str(scoreq_dir))
    dnsmos_digest = mos_pool.local_model_tree_sha256(str(dnsmos_dir))
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha",
        scoreq_options={
            "model_dir": str(scoreq_dir),
            "verified_model_tree_sha256": scoreq_digest,
        },
        dnsmos_options={
            "model_dir": str(dnsmos_dir),
            "verified_model_tree_sha256": dnsmos_digest,
        },
    )

    provenance = pool.provenance_payload()

    assert provenance["scoreq_model_tree_sha256"] == scoreq_digest
    assert provenance["dnsmos_model_tree_sha256"] == dnsmos_digest
    for field in ("scoreq_model_tree_sha256", "dnsmos_model_tree_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", provenance[field])
    assert provenance["scoreq_model_dir"] == str(scoreq_dir)
    assert provenance["dnsmos_model_dir"] == str(dnsmos_dir)
    assert set(provenance["assets"]) == {"selected_models", "models"}
    assert set(provenance["assets"]["models"]["scoreq"]) == {
        "model_dir", "verified_model_tree_sha256",
    }
    assert set(provenance["assets"]["models"]["dnsmos"]) == {
        "model_dir", "verified_model_tree_sha256",
    }


@pytest.mark.parametrize("model", ["scoreq", "dnsmos"])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"model_dir": "/tmp/model"},
        {"model_dir": "/tmp/model", "verified_model_tree_sha256": "A" * 64},
        {"model_dir": "/tmp/model", "verified_model_tree_sha256": "1" * 63},
        {
            "model_dir": "/tmp/model",
            "verified_model_tree_sha256": "1" * 64,
            "provider": "cuda",
        },
    ],
)
def test_pool_rejects_unbound_or_non_exact_tree_options(model, options):
    with pytest.raises((TypeError, ValueError), match="option|field|digest|sha256"):
        Step3MosScorerPool(
            [0], dataset_id="alpha", **{f"{model}_options": options},
        )


def test_pool_provenance_hashes_configured_model_dirs_once_without_mutating_options(
    tmp_path, monkeypatch,
):
    scoreq_dir = tmp_path / "scoreq"
    dnsmos_dir = tmp_path / "dnsmos"
    scoreq_dir.mkdir()
    dnsmos_dir.mkdir()
    scoreq_options = {
        "model_dir": str(scoreq_dir),
        "verified_model_tree_sha256": "a" * 64,
    }
    dnsmos_options = {
        "model_dir": str(dnsmos_dir),
        "verified_model_tree_sha256": "b" * 64,
    }
    calls = []

    def fake_local_model_tree_sha256(path):
        calls.append(path)
        return "a" * 64 if path == str(scoreq_dir) else "b" * 64

    monkeypatch.setattr(mos_pool, "local_model_tree_sha256", fake_local_model_tree_sha256)
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha",
        scoreq_options=scoreq_options,
        dnsmos_options=dnsmos_options,
    )

    provenance = pool.provenance_payload()

    assert calls == [str(scoreq_dir), str(dnsmos_dir)]
    assert provenance["scoreq_model_tree_sha256"] == "a" * 64
    assert provenance["dnsmos_model_tree_sha256"] == "b" * 64
    assert pool._scoreq_options == scoreq_options
    assert pool._dnsmos_options == dnsmos_options


def test_pool_provenance_fails_closed_when_model_tree_scan_fails(tmp_path, monkeypatch):
    scoreq_dir = tmp_path / "scoreq"
    scoreq_dir.mkdir()

    def fail_local_model_tree_sha256(_path):
        raise RuntimeError("model tree scan failed")

    monkeypatch.setattr(mos_pool, "local_model_tree_sha256", fail_local_model_tree_sha256)
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha", scoreq_options={
            "model_dir": str(scoreq_dir),
            "verified_model_tree_sha256": "a" * 64,
        },
    )

    with pytest.raises(RuntimeError, match="model tree scan failed"):
        pool.provenance_payload()


def test_pool_without_verified_plan_keeps_five_model_canonical_fallback(
    tmp_path, monkeypatch,
):
    verified = _verified_pool_plan(tmp_path)
    options = _pool_options(verified)
    expected_models = verified.as_mapping()["provenance"]["models"]
    tree_calls, urgent_calls = [], []

    def tree_digest(path):
        tree_calls.append(path)
        canonical = os.path.realpath(path)
        for model in ("scoreq", "dnsmos", "nisqa", "utmos"):
            asset = expected_models[model]
            asset_root = asset.get("model_dir", os.path.dirname(asset.get("model_path", "")))
            if canonical == asset_root:
                return asset["verified_model_tree_sha256"]
        raise AssertionError(f"unexpected tree path: {path}")

    def urgent_verifier(*args):
        urgent_calls.append(args)
        return dict(expected_models["urgent_pk"])

    monkeypatch.setattr(mos_pool, "local_model_tree_sha256", tree_digest)
    monkeypatch.setattr(mos_pool, "verify_urgent_pk_trusted_assets", urgent_verifier)
    fallback = Step3MosScorerPool([7], dataset_id="alpha", **options)
    fallback_payload = fallback.provenance_payload()

    assert tree_calls == [
        options["scoreq_options"]["model_dir"],
        options["dnsmos_options"]["model_dir"],
        os.path.dirname(options["nisqa_options"]["model_path"]),
        os.path.dirname(options["utmos_options"]["model_path"]),
    ]
    assert len(urgent_calls) == 1
    assert tuple(fallback_payload["assets"]["models"]) == (
        "scoreq", "urgent_pk", "dnsmos", "nisqa", "utmos",
    )

    tree_calls.clear()
    urgent_calls.clear()
    injected = Step3MosScorerPool(
        [7], dataset_id="alpha", verified_plan=verified,
    )
    injected_payload = injected.provenance_payload()

    assert tree_calls == []
    assert urgent_calls == []
    assert artifacts._canonical_json_bytes(injected_payload) == (
        artifacts._canonical_json_bytes(fallback_payload)
    )
    for model in ("nisqa", "utmos"):
        assert injected_payload["assets"]["models"][model] == expected_models[model]


def test_pool_verified_plan_rejects_cross_bound_options_and_provenance(tmp_path):
    trusted = _verified_pool_plan(tmp_path, suffix="-trusted")
    foreign = _verified_pool_plan(
        tmp_path, score_range={"min": 0, "max": 10}, suffix="-foreign",
    )

    with pytest.raises((TypeError, ValueError), match="verified|option|identity|match"):
        Step3MosScorerPool(
            [0], dataset_id="alpha", verified_plan=trusted,
            **_pool_options(foreign),
        )
    with pytest.raises((TypeError, ValueError), match="verified|provenance|match"):
        Step3MosScorerPool(
            [0], dataset_id="alpha", verified_plan=trusted,
            preverified_provenance=foreign.provenance,
        )


def test_pool_verified_and_fallback_snapshot_nested_options_and_keep_protocol_golden(
    tmp_path,
):
    verified = _verified_pool_plan(tmp_path)
    options = _pool_options(verified)
    expected_options = verified.as_mapping()["options_by_model"]
    requests = []

    def popen(command, **_kwargs):
        with open(command[2], encoding="utf-8") as stream:
            request = json.load(stream)
        requests.append(request)
        subshard = request["subshards"][0]
        with open(subshard["input_path"], encoding="utf-8") as source:
            items = [json.loads(line) for line in source]
        with open(subshard["output_path"], "x", encoding="utf-8") as output:
            for item in items:
                output.write(json.dumps({
                    "ordinal": item["ordinal"],
                    "scores": _five_model_scores(),
                }) + "\n")
        _write_fake_worker_attestation(request)
        return _FakeProcess()

    fallback = Step3MosScorerPool(
        [0], dataset_id="alpha", popen=popen, **options,
    )
    injected = Step3MosScorerPool(
        [0], dataset_id="alpha", popen=popen, verified_plan=verified,
        **options,
    )
    options["urgent_options"]["score_range"]["max"] = 99
    options["urgent_options"]["score_range"]["unit"] = "mutated"
    models = tuple(expected_options)
    _score_rows(fallback, [_row(0)], models)
    _score_rows(injected, [_row(0)], models)

    assert len(requests) == 2
    expected_keys = {
        "protocol", "asset_load_binding", "scoring_execution_policy",
        "private_dir", "attestation_path",
        "subshards", "urgent_options", "scoreq_options", "dnsmos_options",
        "nisqa_options", "utmos_options",
    }
    for request in requests:
        assert set(request) == expected_keys
        assert request["protocol"] == (
            "audio_filter_pipeline.step3_mos_worker.v6"
        )
        assert request["asset_load_binding"] == dict(mos_worker.ASSET_LOAD_BINDING)
        assert request["subshards"][0]["models"] == list(expected_options)
        assert set(request["scoreq_options"]) == {
            "model_dir", "verified_model_tree_sha256",
        }
        assert set(request["dnsmos_options"]) == {
            "model_dir", "verified_model_tree_sha256",
        }
        for model in ("scoreq", "dnsmos"):
            request_option = request[f"{model}_options"]
            expected_asset = verified.as_mapping()["provenance"]["models"][model]
            assert request_option["verified_model_tree_sha256"] == (
                expected_asset["verified_model_tree_sha256"]
            )
        request_options = {
            "scoreq": request["scoreq_options"],
            "urgent_pk": request["urgent_options"],
            "dnsmos": request["dnsmos_options"],
            "nisqa": request["nisqa_options"],
            "utmos": request["utmos_options"],
        }
        assert artifacts._canonical_json_bytes(request_options) == (
            artifacts._canonical_json_bytes(expected_options)
        )


def test_pool_fails_closed_on_duplicate_child_result():
    pool = Step3MosScorerPool(
        [0, 1], dataset_id="alpha", popen=_popen([], duplicate=True),
        **_default_pool_options(),
    )

    with pytest.raises(ValueError, match="ordinal"):
        _score_rows(
            pool, [_row(0), _row(1)],
            ("scoreq", "urgent_pk", "dnsmos"),
        )


def _read_result_fixture(tmp_path, result_items, input_ordinals=(0,)):
    input_path = tmp_path / "input.jsonl"
    result_path = tmp_path / "result.jsonl"
    input_path.write_text("".join(
        json.dumps({"ordinal": ordinal, "row": _row(ordinal)}) + "\n"
        for ordinal in input_ordinals
    ), encoding="utf-8")
    result_path.write_text("".join(
        json.dumps(item) + "\n" for item in result_items
    ), encoding="utf-8")
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    return input_path, result_path, digest


@pytest.mark.parametrize("ordinal", [True, 0.0, "0"])
def test_pool_rejects_non_integer_result_ordinal(tmp_path, ordinal):
    input_path, result_path, digest = _read_result_fixture(
        tmp_path, [{"ordinal": ordinal, "scores": _scores()}],
    )

    with pytest.raises(ValueError, match="ordinal"):
        list(mos_gpu_job._OwnedGpuScoringJob._read_result(
            str(result_path), str(input_path), digest, 0, 1,
        ))


@pytest.mark.parametrize(
    ("result_ordinals", "expected_error"),
    [
        ([0, 0], "ordinal"),
        ([0], "row count"),
        ([1, 0], "ordinal"),
        ([0, 1, 2], "row count"),
    ],
)
def test_parent_rejects_duplicate_missing_unordered_or_extra_result_ordinals(
    tmp_path, result_ordinals, expected_error,
):
    input_path, result_path, digest = _read_result_fixture(
        tmp_path,
        [{"ordinal": ordinal, "scores": _scores()} for ordinal in result_ordinals],
        input_ordinals=(0, 1),
    )

    with pytest.raises(ValueError, match=expected_error):
        list(mos_gpu_job._OwnedGpuScoringJob._read_result(
            str(result_path), str(input_path), digest, 0, 2,
        ))


@pytest.mark.parametrize(
    ("input_ordinals", "start", "stop"),
    [((0,), 0, 2), ((0, 1), 0, 1)],
)
def test_parent_rejects_input_shard_count_mismatch(
    tmp_path, input_ordinals, start, stop,
):
    result_ordinals = tuple(range(start, stop))
    input_path, result_path, digest = _read_result_fixture(
        tmp_path,
        [{"ordinal": ordinal, "scores": _scores()} for ordinal in result_ordinals],
        input_ordinals=input_ordinals,
    )

    with pytest.raises(ValueError, match="input shard row count"):
        list(mos_gpu_job._OwnedGpuScoringJob._read_result(
            str(result_path), str(input_path), digest, start, stop,
        ))


def test_parent_rejects_input_shard_digest_mismatch_before_restoring_rows(tmp_path):
    input_path, result_path, _digest = _read_result_fixture(
        tmp_path, [{"ordinal": 0, "scores": _scores()}],
    )

    with pytest.raises(ValueError, match="digest mismatch"):
        list(mos_gpu_job._OwnedGpuScoringJob._read_result(
            str(result_path), str(input_path), "f" * 64, 0, 1,
        ))


def test_parent_result_cleanup_preserves_primary_error_when_sidecar_close_fails(
    tmp_path, monkeypatch,
):
    input_path, result_path, digest = _read_result_fixture(
        tmp_path, [{"ordinal": 0, "scores": _scores()}],
    )
    close_calls = []
    result_close_calls = []
    finalized = []
    real_finalize = mos_gpu_job._finalize_streams

    class BrokenOrdinalIterator:
        def __iter__(self):
            return self

        def __next__(self):
            return 1

        def close(self):
            close_calls.append(True)
            raise RuntimeError("ordinal close failed")

    def finalize(streams, primary=None, *, sync):
        stream_list = tuple(streams)
        finalized.extend(stream_list)
        assert primary is not None
        real_finalize(stream_list, primary, sync=sync)

    monkeypatch.setattr(
        mos_gpu_job._OwnedGpuScoringJob,
        "_iter_ordinal_sidecar",
        staticmethod(lambda *_args, **_kwargs: BrokenOrdinalIterator()),
    )
    class TrackingResultIterator:
        def __iter__(self):
            return self

        def __next__(self):
            return next(iter(()))

        def close(self):
            result_close_calls.append(True)

    real_iter_jsonl = mos_gpu_job._shared_iter_bounded_jsonl

    def iter_jsonl(source, label, *, max_line_bytes, max_file_bytes):
        if label == "MOS worker result output":
            return TrackingResultIterator()
        return real_iter_jsonl(
            source, label, max_line_bytes=max_line_bytes,
            max_file_bytes=max_file_bytes,
        )

    monkeypatch.setattr(mos_gpu_job, "_shared_iter_bounded_jsonl", iter_jsonl)
    monkeypatch.setattr(mos_gpu_job, "_finalize_streams", finalize)
    sidecar = {
        "ordinal_sidecar_path": str(tmp_path / "unused.jsonl"),
        "ordinal_sidecar_sha256": "a" * 64,
        "ordinal_sidecar_size_bytes": 1,
    }

    with pytest.raises(ValueError, match="ordinal or row") as raised:
        list(mos_gpu_job._OwnedGpuScoringJob._read_result(
            str(result_path), str(input_path), digest, 0, 2, sidecar,
        ))

    assert close_calls == [True]
    assert result_close_calls == [True]
    assert len(finalized) == 2
    assert any("ordinal close failed" in note for note in raised.value.__notes__)


def test_parent_result_consumer_cancel_propagates_result_close_failure(
    tmp_path, monkeypatch,
):
    input_path, result_path, digest = _read_result_fixture(
        tmp_path, [{"ordinal": 0, "scores": _scores()}],
    )
    real_open = mos_gpu_job._open_regular_stream
    opened = []

    class CloseFailingStream:
        def __init__(self, stream):
            self._stream = stream
            self.close_calls = 0

        def __getattr__(self, name):
            return getattr(self._stream, name)

        def close(self):
            self.close_calls += 1
            self._stream.close()
            raise RuntimeError("result close failed")

    def open_stream(path, label):
        stream, signature = real_open(path, label)
        if label == "MOS worker result output":
            stream = CloseFailingStream(stream)
            opened.append(stream)
        return stream, signature

    monkeypatch.setattr(mos_gpu_job, "_open_regular_stream", open_stream)
    results = mos_gpu_job._OwnedGpuScoringJob._read_result(
        str(result_path), str(input_path), digest, 0, 1,
    )

    assert next(results)["ordinal"] == 0
    with pytest.raises(RuntimeError, match="result close failed"):
        results.close()
    assert len(opened) == 1
    assert opened[0].close_calls == 1


@pytest.mark.parametrize("owner", ("parent", "worker"))
def test_sparse_sidecar_size_error_survives_stream_close_failure(
    tmp_path, monkeypatch, owner,
):
    class CloseFailingStream:
        close_calls = 0

        def close(self):
            self.close_calls += 1
            raise RuntimeError("sidecar close failed")

    stream = CloseFailingStream()
    module = mos_gpu_job if owner == "parent" else mos_worker
    monkeypatch.setattr(
        module,
        "_open_regular_stream",
        lambda *_args, **_kwargs: (stream, SimpleNamespace(st_size=2)),
    )
    sidecar = {
        "ordinal_sidecar_path": str(tmp_path / "ordinals.jsonl"),
        "ordinal_sidecar_sha256": "a" * 64,
        "ordinal_sidecar_size_bytes": 1,
    }
    if owner == "parent":
        ordinals = mos_gpu_job._OwnedGpuScoringJob._iter_ordinal_sidecar(sidecar, 0, 1, 1)
    else:
        ordinals = mos_worker._iter_ordinal_sidecar(
            sidecar, 0, 1, 1, os.path.realpath(tmp_path),
        )

    with pytest.raises(ValueError, match="size mismatch") as raised:
        next(ordinals)

    assert stream.close_calls == 1
    assert any("sidecar close failed" in note for note in raised.value.__notes__)


def test_active_child_cleanup_failure_replaces_generator_exit(monkeypatch):
    process = object()
    job = _owned_job(Step3MosScorerPool([0], dataset_id="alpha"))
    job._processes.add(process)

    def fail_termination(_process):
        raise RuntimeError("active termination failed")

    monkeypatch.setattr(job, "_terminate_process_group", fail_termination)

    with pytest.raises(RuntimeError, match="active termination failed"):
        job._terminate_active_children(GeneratorExit())


def test_worker_process_cleanup_failure_replaces_generator_exit(monkeypatch):
    class Process:
        stdin = None
        stdout = None
        stderr = None

        @staticmethod
        def wait(timeout=None):
            return 0

        @staticmethod
        def poll():
            return 0

    process = Process()
    job = _owned_job(Step3MosScorerPool([0], dataset_id="alpha"))
    job._processes.add(process)

    def fail_termination(_process):
        raise RuntimeError("worker termination failed")

    monkeypatch.setattr(job, "_terminate_process_group", fail_termination)

    with pytest.raises(RuntimeError, match="worker termination failed"):
        job._cleanup_worker_process(process, GeneratorExit())


def test_score_subshard_generator_exit_propagates_target_cleanup_failure(
    tmp_path, monkeypatch,
):
    target = object()
    finalized = []

    monkeypatch.setattr(mos_worker, "_open_new_regular_stream", lambda *_args: target)

    def interrupt_scoring(*_args, **_kwargs):
        raise GeneratorExit

    monkeypatch.setattr(mos_worker, "_iter_bounded_jsonl", interrupt_scoring)
    monkeypatch.setattr(mos_worker, "_verify_validated_subshard", lambda *_args: None)

    def fail_finalize(streams, primary=None, *, sync):
        finalized.extend(streams)
        assert primary is None
        assert sync is True
        raise RuntimeError("target cleanup failed")

    monkeypatch.setattr(mos_worker, "_finalize_streams", fail_finalize)
    subshard = {
        "models": ["scoreq"],
        "output_path": str(tmp_path / "output.jsonl"),
        "start_ordinal": 0,
        "end_ordinal": 1,
    }

    runtime = SimpleNamespace(
        open_waveform_coordinator=lambda _models: _EmptyWaveformCoordinator(),
    )
    with pytest.raises(RuntimeError, match="target cleanup failed"):
        mos_worker._score_subshard(
            runtime, subshard, input_stream=object(),
            input_signature=(1, 2, 3, 4, 5),
        )

    assert finalized == [target]


def test_handle_request_generator_exit_propagates_input_cleanup_failure(
    monkeypatch,
):
    stream = object()
    request = {
        "private_dir": "/tmp",
        "subshards": [{"row_count": 1}],
        "urgent_options": None,
        "scoreq_options": None,
        "dnsmos_options": None,
        "nisqa_options": None,
        "utmos_options": None,
    }
    monkeypatch.setattr(mos_worker, "_load_request", lambda _path: request)
    monkeypatch.setattr(
        mos_worker,
        "_prevalidate_subshard_input",
        lambda *_args: (stream, (1, 2, 3, 4, 5), None, None),
    )
    registry = SimpleNamespace(finalize=lambda _count: None, close=lambda: None)
    monkeypatch.setattr(mos_worker, "DiskBackedOrdinalRegistry", lambda _path: registry)
    monkeypatch.setattr(mos_worker, "_verify_validated_subshard", lambda *_args: None)

    def interrupt_runtime(**_kwargs):
        raise GeneratorExit

    monkeypatch.setattr(mos_worker, "Step3MosRuntime", interrupt_runtime)
    finalized = []

    def fail_finalize(streams, primary=None, *, sync):
        finalized.extend(streams)
        assert primary is None
        assert sync is False
        raise RuntimeError("input cleanup failed")

    monkeypatch.setattr(mos_worker, "_finalize_streams", fail_finalize)

    with pytest.raises(RuntimeError, match="input cleanup failed"):
        mos_worker.handle_request("/unused/request.json")

    assert finalized == [stream]


def _heartbeat_process_popen(processes, code):
    def popen(command, **kwargs):
        request_path = command[-1]
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        subshard = request["subshards"][0]
        with open(subshard["input_path"], encoding="utf-8") as source:
            items = [json.loads(line) for line in source]
        with open(subshard["output_path"], "x", encoding="utf-8") as output:
            for item in items:
                output.write(json.dumps({
                    "ordinal": item["ordinal"], "scores": _scores(),
                }) + "\n")
        _write_fake_worker_attestation(request)
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        processes.append(process)
        return process

    return popen


def test_pool_reports_completed_progress_frames_without_exposing_them_as_results():
    processes = []
    code = (
        "import sys,time\n"
        "print('audio_filter_pipeline.step3_mos_worker.progress.v1:2', flush=True)\n"
        "time.sleep(0.01)\n"
        "print('audio_filter_pipeline.step3_mos_worker.progress.v1:3', flush=True)\n"
    )
    progress = []
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha", popen=_heartbeat_process_popen(processes, code),
        **_default_pool_options(),
    )
    pool.set_progress_callback(progress.append)
    jobs = []

    results = _score_rows(
        pool, [_row(0)], ("scoreq", "urgent_pk", "dnsmos"),
        job_hook=jobs.append,
    )

    assert len(results) == 1
    assert progress == [2, 3, 1]
    assert processes[0].poll() == 0
    assert jobs[0]._processes == set()


def test_pool_rejects_invalid_progress_frames():
    processes = []
    code = (
        "print('audio_filter_pipeline.step3_mos_worker.progress.v1:0', flush=True)\n"
    )
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha", popen=_heartbeat_process_popen(processes, code),
        **_default_pool_options(),
    )

    with pytest.raises(RuntimeError, match="progress frame is invalid"):
        _score_rows(pool, [_row(0)], ("scoreq", "urgent_pk", "dnsmos"))

    assert processes[0].poll() is not None
    with pytest.raises(RuntimeError, match="before successful scoring"):
        pool.runtime_attestation()


def test_pool_heartbeat_resets_idle_timeout_beyond_old_total_duration():
    processes = []
    code = (
        "import sys,time\n"
        "for _ in range(8):\n"
        " print('audio_filter_pipeline.step3_mos_worker.heartbeat.v1', flush=True)\n"
        " time.sleep(0.1)\n"
    )
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha", worker_idle_timeout_seconds=0.5,
        popen=_heartbeat_process_popen(processes, code),
        **_default_pool_options(),
    )

    results = _score_rows(
        pool, [_row(0)], ("scoreq", "urgent_pk", "dnsmos"),
    )

    assert len(results) == 1
    assert processes[0].poll() == 0
    assert processes[0].stdout.closed and processes[0].stderr.closed


def test_pool_without_heartbeat_hits_idle_timeout_and_cleans_child():
    processes = []
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha", worker_idle_timeout_seconds=1.0,
        popen=_heartbeat_process_popen(
            processes,
            "import sys,time; print('stalled-tail', file=sys.stderr, flush=True); time.sleep(5)",
        ),
        **_default_pool_options(),
    )
    jobs = []

    with pytest.raises(RuntimeError, match="idle timeout.*stalled-tail"):
        _score_rows(
            pool, [_row(0)], ("scoreq", "urgent_pk", "dnsmos"),
            job_hook=jobs.append,
        )

    assert processes[0].poll() is not None
    assert processes[0].stdout.closed and processes[0].stderr.closed
    assert jobs[0]._processes == set()


def test_pool_hard_timeout_stops_continuous_heartbeat_and_cleans_child():
    processes = []
    code = (
        "import time\n"
        "for _ in range(100):\n"
        " print('audio_filter_pipeline.step3_mos_worker.heartbeat.v1', flush=True)\n"
        " time.sleep(0.01)\n"
    )
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha", worker_idle_timeout_seconds=0.2,
        worker_hard_timeout_seconds=0.3,
        popen=_heartbeat_process_popen(processes, code),
        **_default_pool_options(),
    )
    jobs = []

    with pytest.raises(RuntimeError, match="hard timeout"):
        _score_rows(
            pool, [_row(0)], ("scoreq", "urgent_pk", "dnsmos"),
            job_hook=jobs.append,
        )

    assert processes[0].poll() is not None
    assert processes[0].stdout.closed and processes[0].stderr.closed
    assert jobs[0]._processes == set()


def test_pool_timeout_reports_bounded_stderr_tail(monkeypatch):
    terminated = []
    proc = _FakeProcess(stderr=b"discard-this-tail-kept", timeout=True)
    monkeypatch.setattr(mos_gpu_job.os, "killpg", lambda pid, sig: terminated.append((pid, sig)))
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha", worker_idle_timeout_seconds=1,
        worker_stderr_tail_bytes=9, popen=lambda *_args, **_kwargs: proc,
        **_default_pool_options(),
    )

    with pytest.raises(RuntimeError, match="idle timeout.*tail-kept"):
        _score_rows(pool, [_row(0)], ("scoreq", "urgent_pk", "dnsmos"))

    assert terminated == [(proc.pid, signal.SIGTERM)]


def test_pool_timeout_preserves_partial_stderr_from_initial_timeout(monkeypatch):
    calls = []
    terminated = []

    class Process:
        pid = 4321
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = -15
            return self.returncode

        def communicate(self, timeout=None):
            calls.append(timeout)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired(
                    "worker", timeout, output=b"partial-stdout", stderr=b"initial-tail",
                )
            return b"", b""

    monkeypatch.setattr(mos_gpu_job.os, "killpg", lambda pid, sig: terminated.append((pid, sig)))
    process = Process()
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha", worker_idle_timeout_seconds=1,
        popen=lambda *_args, **_kwargs: process,
        **_default_pool_options(),
    )

    with pytest.raises(RuntimeError, match="stderr_tail='initial-tail'"):
        _score_rows(pool, [_row(0)], ("scoreq", "urgent_pk", "dnsmos"))

    assert calls == [1.0, 0.05]
    assert terminated == [(process.pid, signal.SIGTERM)]


def test_pool_failure_terminates_other_active_child(monkeypatch):
    blocked = _FakeProcess(blocked=True)
    failed = _FakeProcess(returncode=7, stderr=b"failure")
    calls = 0

    def popen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return blocked
        assert blocked.started.wait(timeout=1.0)
        return failed

    terminated = []
    monkeypatch.setattr(mos_gpu_job.os, "killpg", lambda pid, sig: terminated.append((pid, sig)))
    pool = Step3MosScorerPool(
        [0, 1], dataset_id="alpha", popen=popen,
        **_default_pool_options(),
    )

    with pytest.raises(RuntimeError, match="return code 7"):
        _score_rows(
            pool, [_row(0), _row(1)],
            ("scoreq", "urgent_pk", "dnsmos"),
        )

    assert (blocked.pid, signal.SIGTERM) in terminated


def test_second_executor_submit_failure_terminates_registered_child_without_idle_wait(
    monkeypatch,
):
    blocked = _FakeProcess(blocked=True)
    real_executor = mos_gpu_job.ThreadPoolExecutor

    class FailSecondSubmitExecutor:
        def __init__(self, max_workers):
            self._executor = real_executor(max_workers=max_workers)
            self._submit_count = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self._executor.shutdown(wait=True)
            return False

        def shutdown(self, wait=True, cancel_futures=False):
            # 生产 abort 路径调用 executor.shutdown(wait=True)；替身缺该方法时
            # 工作线程会在断言之后才清理进程注册表，造成时序假失败。
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

        def submit(self, function, *args):
            self._submit_count += 1
            if self._submit_count == 2:
                assert blocked.started.wait(timeout=1.0)
                raise RuntimeError("second submit failed")
            return self._executor.submit(function, *args)

    terminated = []
    monkeypatch.setattr(mos_gpu_job, "ThreadPoolExecutor", FailSecondSubmitExecutor)
    monkeypatch.setattr(
        mos_gpu_job.os, "killpg",
        lambda pid, sig: terminated.append((pid, sig)),
    )
    pool = Step3MosScorerPool(
        [0, 1], dataset_id="alpha", worker_idle_timeout_seconds=30,
        popen=lambda *_args, **_kwargs: blocked,
        **_default_pool_options(),
    )
    jobs = []

    with pytest.raises(RuntimeError, match="second submit failed"):
        _score_rows(
            pool, [_row(0), _row(1)],
            ("scoreq", "urgent_pk", "dnsmos"), job_hook=jobs.append,
        )

    assert (blocked.pid, signal.SIGTERM) in terminated
    assert blocked.poll() is not None
    assert jobs[0]._processes == set()


def test_worker_still_alive_after_term_and_kill_remains_registered(monkeypatch):
    class SurvivingProcess:
        pid = 4321
        returncode = None
        stdin = None
        stdout = None
        stderr = None

        def poll(self):
            return None

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("worker", timeout)

    process = SurvivingProcess()
    signals = []
    monkeypatch.setattr(
        mos_gpu_job.os, "killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )
    job = _owned_job(Step3MosScorerPool([0], dataset_id="alpha"))
    job._processes.add(process)

    with pytest.raises(subprocess.TimeoutExpired, match="worker"):
        job._cleanup_worker_process(process, None)

    assert signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]
    assert process in job._processes


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, RuntimeError])
def test_pool_cleans_process_when_registration_is_interrupted(monkeypatch, error_type):
    class TrackingStream:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FailingProcessSet(set):
        def __init__(self):
            super().__init__()
            self._failed = False

        def add(self, process):
            if not self._failed:
                self._failed = True
                raise error_type("process registration interrupted")
            super().add(process)

    process = _FakeProcess(blocked=True)
    process.stdin = TrackingStream()
    process.stdout = TrackingStream()
    process.stderr = TrackingStream()
    terminated = []
    monkeypatch.setattr(mos_gpu_job.os, "killpg", lambda pid, sig: terminated.append((pid, sig)))
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha", popen=lambda *_args, **_kwargs: process,
        **_default_pool_options(),
    )
    jobs = []

    def configure_job(job):
        job._processes = FailingProcessSet()
        jobs.append(job)

    with pytest.raises(error_type, match="process registration interrupted"):
        _score_rows(
            pool, [_row(0)], ("scoreq", "urgent_pk", "dnsmos"),
            job_hook=configure_job,
        )

    assert (process.pid, signal.SIGTERM) in terminated
    assert process.poll() == process.returncode
    assert process.stdin.closed and process.stdout.closed and process.stderr.closed
    assert process not in jobs[0]._processes


@pytest.fixture
def spooled_output():
    with tempfile.SpooledTemporaryFile() as output:
        yield output


def test_worker_fleet_loop_handles_requests_sequentially_and_clears_cache(
    monkeypatch, spooled_output,
):
    handled = []
    cleared = []
    output = spooled_output
    lines = iter([
        json.dumps({"dataset_id": "alpha", "request_path": "/tmp/alpha.json"}).encode() + b"\n",
        json.dumps({"dataset_id": "beta", "request_path": "/tmp/beta.json"}).encode() + b"\n",
        b"",
    ])
    monkeypatch.setattr(
        mos_worker.sys, "stdin",
        SimpleNamespace(buffer=SimpleNamespace(
            readline=lambda _limit: next(lines),
        )),
    )
    monkeypatch.setattr(mos_worker.sys, "stdout", SimpleNamespace(buffer=output))
    monkeypatch.setattr(
        mos_worker, "handle_request",
        lambda path, **kwargs: handled.append((path, kwargs["expected_dataset_id"])),
    )
    monkeypatch.setattr(mos_worker, "clear_model_cache", lambda: cleared.append(True))

    mos_worker.serve_requests()

    output.seek(0)
    assert output.read() == (
        mos_worker._FLEET_READY + mos_worker._FLEET_DONE + mos_worker._FLEET_DONE
    )
    assert handled == [
        ("/tmp/alpha.json", "alpha"), ("/tmp/beta.json", "beta"),
    ]
    assert cleared == [True]


def test_worker_heartbeat_is_request_scoped_and_stops_before_fleet_done(
    monkeypatch, spooled_output,
):
    output = spooled_output
    lines = iter([
        json.dumps({
            "dataset_id": "alpha", "request_path": "/tmp/alpha.json",
        }).encode() + b"\n",
        b"",
    ])
    monkeypatch.setenv("AUDIO_FILTER_STEP3_HEARTBEAT_SECONDS", "0.01")
    monkeypatch.setattr(
        mos_worker.sys, "stdin",
        SimpleNamespace(buffer=SimpleNamespace(readline=lambda _limit: next(lines))),
    )
    monkeypatch.setattr(mos_worker.sys, "stdout", SimpleNamespace(buffer=output))
    monkeypatch.setattr(
        mos_worker, "handle_request", lambda *_args, **_kwargs: time.sleep(0.5),
    )
    monkeypatch.setattr(mos_worker, "clear_model_cache", lambda: None)

    mos_worker.serve_requests()

    output.seek(0)
    frames = output.readlines()
    assert frames[0] == mos_worker._FLEET_READY
    assert frames[-1] == mos_worker._FLEET_DONE
    assert frames[1:-1]
    assert set(frames[1:-1]) == {mos_pool._HEARTBEAT_PROTOCOL}


def test_heartbeat_launcher_does_not_duplicate_worker_heartbeat_loop():
    launcher = mos_gpu_job._heartbeat_launcher_payload().decode("utf-8")

    assert "def heartbeat" not in launcher
    assert "threading.Thread" not in launcher
    assert "step3_mos_worker.main" in launcher


def test_heartbeat_launcher_imports_core_with_tools_only_pythonpath(tmp_path):
    launcher_path = tmp_path / "heartbeat_launcher.py"
    launcher_path.write_bytes(mos_gpu_job._heartbeat_launcher_payload())
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])

    completed = subprocess.run(
        [sys.executable, str(launcher_path), "--serve"],
        input=b"",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        cwd=tmp_path,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert completed.stdout == b"audio_filter_pipeline.step3_mos_worker.fleet.ready.v1\n"


def test_worker_fleet_loop_failure_stops_before_later_request_and_clears_cache(
    monkeypatch, spooled_output,
):
    handled = []
    cleared = []
    output = spooled_output
    lines = iter([
        json.dumps({"dataset_id": "fail", "request_path": "/tmp/fail.json"}).encode() + b"\n",
        json.dumps({"dataset_id": "later", "request_path": "/tmp/later.json"}).encode() + b"\n",
    ])
    monkeypatch.setattr(
        mos_worker.sys, "stdin",
        SimpleNamespace(buffer=SimpleNamespace(readline=lambda _limit: next(lines))),
    )
    monkeypatch.setattr(mos_worker.sys, "stdout", SimpleNamespace(buffer=output))

    def fail(request_path, *, expected_dataset_id):
        handled.append((request_path, expected_dataset_id))
        raise RuntimeError("request failed")

    monkeypatch.setattr(mos_worker, "handle_request", fail)
    monkeypatch.setattr(mos_worker, "clear_model_cache", lambda: cleared.append(True))

    with pytest.raises(RuntimeError, match="request failed"):
        mos_worker.serve_requests()

    output.seek(0)
    assert output.read() == mos_worker._FLEET_READY
    assert handled == [("/tmp/fail.json", "fail")]
    assert cleared == [True]


_FAKE_FLEET_WORKER = r'''
import json
import sys

log_path = sys.argv[1]
print("audio_filter_pipeline.step3_mos_worker.fleet.ready.v1", flush=True)
for control_line in sys.stdin:
    control = json.loads(control_line)
    with open(control["request_path"], encoding="utf-8") as request_stream:
        request = json.load(request_stream)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(json.dumps({
            "dataset_id": control["dataset_id"],
            "private_dir": request["private_dir"],
            "pid": __import__("os").getpid(),
        }) + "\n")
    worker_models = []
    tuples = []
    seen = set()
    completed_rows = 0
    for shard in request["subshards"]:
        models = tuple(shard["models"])
        for model in models:
            if model not in worker_models:
                worker_models.append(model)
        if models not in seen:
            tuples.append({
                "models": list(models),
                "attestation": {
                    "requested_device": "cuda", "fallback": False,
                    "selected_models": list(models),
                    "asset_load_binding": {"version": 3},
                    "scoring_execution_policy": {
                        "version": 1,
                        "waveform": {"channel": "first", "dtype": "float32", "padding": "none"},
                        "batching": {"max_rows": 32, "max_float32_bytes": 67108864, "decode_reservation": "interval_channels_plus_mono_float32", "cross_source": True, "flush": "tuple_change_or_limit_or_eof"},
                        "source_session": {"reuse": "same_source_across_batches", "switch": "close_reverify_then_open"},
                        "worker": {"physical_subshard_boundary": "flush"},
                    },
                    "actual_providers": {
                        "scoreq": {
                            "execution_mode": "cuda_primary_hybrid_ep",
                            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                            "device": "gpu",
                        },
                    },
                },
            })
            seen.add(models)
        with open(shard["input_path"], encoding="utf-8") as source:
            rows = [json.loads(line) for line in source]
        if any(item["row"]["dataset_id"] == "fail" for item in rows):
            raise RuntimeError("requested failure")
        completed_rows += len(rows)
        with open(shard["output_path"], "x", encoding="utf-8") as output:
            for item in rows:
                output.write(json.dumps({
                    "ordinal": item["ordinal"], "scores": {"scoreq": 4.1},
                }) + "\n")
    with open(request["attestation_path"], "x", encoding="utf-8") as output:
        output.write(json.dumps({
            "protocol": "audio_filter_pipeline.step3_mos_worker.v6",
            "asset_load_binding": {"version": 3},
            "scoring_execution_policy": {
                "version": 1,
                "waveform": {"channel": "first", "dtype": "float32", "padding": "none"},
                "batching": {"max_rows": 32, "max_float32_bytes": 67108864, "decode_reservation": "interval_channels_plus_mono_float32", "cross_source": True, "flush": "tuple_change_or_limit_or_eof"},
                "source_session": {"reuse": "same_source_across_batches", "switch": "close_reverify_then_open"},
                "worker": {"physical_subshard_boundary": "flush"},
            },
            "worker_models": worker_models,
            "model_tuple_attestations": tuples,
        }) + "\n")
    print(
        "audio_filter_pipeline.step3_mos_worker.progress.v1:" + str(completed_rows),
        flush=True,
    )
    print("audio_filter_pipeline.step3_mos_worker.fleet.done.v1", flush=True)
with open(log_path, "a", encoding="utf-8") as log:
    log.write(json.dumps({"cleared": True}) + "\n")
'''


def _fake_fleet_popen(log_path, starts):
    def popen(_command, **kwargs):
        starts.append(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
        return subprocess.Popen(
            [sys.executable, "-u", "-c", _FAKE_FLEET_WORKER, str(log_path)],
            stdin=kwargs["stdin"], stdout=kwargs["stdout"], stderr=kwargs["stderr"],
            start_new_session=kwargs["start_new_session"], env=kwargs["env"],
        )
    return popen


def test_fleet_reuses_one_lazy_gpu_worker_across_two_dataset_views(tmp_path):
    verified = _verified_pool_plan(tmp_path)
    log_path = tmp_path / "fleet.jsonl"
    starts = []
    progress = []
    fleet = Step3MosWorkerFleet(
        [7], verified_plan=verified, popen=_fake_fleet_popen(log_path, starts),
    )
    fleet.set_progress_callback(progress.append)
    assert starts == []
    try:
        for dataset_id in ("alpha", "beta"):
            row = _row(0)
            row["dataset_id"] = dataset_id
            output = _score_rows(fleet.for_dataset(dataset_id), [row], ("scoreq",))
            assert output[0]["row"]["dataset_id"] == dataset_id
            assert output[0]["scores"] == {"scoreq": 4.1}
        assert starts == ["7"]
        assert progress == [1, 1]
        requests = [
            json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        assert [request["dataset_id"] for request in requests] == ["alpha", "beta"]
        assert {request["pid"] for request in requests} == {requests[0]["pid"]}
        assert requests[0]["private_dir"] != requests[1]["private_dir"]
    finally:
        fleet.close()
    lifecycle = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert lifecycle[-1] == {"cleared": True}


def test_fleet_close_is_idempotent_and_rejects_existing_or_new_dataset_views(tmp_path):
    verified = _verified_pool_plan(tmp_path)
    fleet = Step3MosWorkerFleet([0], verified_plan=verified)
    view = fleet.for_dataset("alpha")

    fleet.close()
    fleet.close()
    fleet.close()

    with pytest.raises(RuntimeError, match="closed"):
        fleet.for_dataset("beta")
    with pytest.raises(RuntimeError, match="closed"):
        _score_rows(view, [_row(0)], ("scoreq",))


def test_fleet_request_failure_closes_worker_and_rejects_later_dataset(tmp_path):
    verified = _verified_pool_plan(tmp_path)
    starts = []
    fleet = Step3MosWorkerFleet(
        [3], verified_plan=verified,
        worker_idle_timeout_seconds=1,
        popen=_fake_fleet_popen(tmp_path / "failed-fleet.jsonl", starts),
    )
    row = _row(0)
    row["dataset_id"] = "fail"

    with pytest.raises(RuntimeError, match="exited|control"):
        _score_rows(fleet.for_dataset("fail"), [row], ("scoreq",))
    with pytest.raises(RuntimeError, match="broken"):
        fleet.for_dataset("later")

    assert starts == ["3"]
    assert all(slot._proc is None for slot in fleet._slots.values())
    fleet.close()


def test_fleet_view_surfaces_root_cause_over_masked_secondary_error(tmp_path):
    verified = _verified_pool_plan(tmp_path)
    fleet = Step3MosWorkerFleet([0], verified_plan=verified)
    view = fleet.for_dataset("alpha")
    root = ValueError("dnsmos sig score must be in [1.0, 5.0], got 5.2")
    secondary = RuntimeError("MOS fleet GPU 0 exited with return code -15")

    def masked_score_task_groups(_groups):
        # 模拟真实竞态：根因先触发 fleet._break，连锁中止产生的次生 -15
        # 错误随后经 future.result() 浮出；view 必须改抛根因。
        fleet._break(root)
        raise secondary
        yield  # pragma: no cover

    view._pool.score_task_groups = masked_score_task_groups
    try:
        with pytest.raises(ValueError, match=r"dnsmos sig score") as raised:
            list(view.score_task_groups([]))
    finally:
        fleet.close()
    assert raised.value.__cause__ is secondary


def test_fleet_close_failure_keeps_process_and_second_close_converges(tmp_path):
    class RetryStream:
        def __init__(self, *, fail_once=False):
            self.fail_once = fail_once
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("pipe close failed")

    class Process:
        pid = 4321
        returncode = 0

        def __init__(self):
            self.stdin = RetryStream()
            self.stdout = RetryStream(fail_once=True)
            self.stderr = RetryStream()
            self.wait_calls = 0

        def wait(self, timeout=None):
            self.wait_calls += 1
            return self.returncode

        def poll(self):
            return self.returncode

    fleet = Step3MosWorkerFleet(
        [0], verified_plan=_verified_pool_plan(tmp_path),
    )
    launcher_cleanup = []
    launcher = SimpleNamespace(cleanup=lambda: launcher_cleanup.append(True))
    fleet._launcher_temp = launcher
    slot = fleet._slots[0]
    process = Process()
    slot._proc = process

    with pytest.raises(RuntimeError, match="pipe close failed"):
        fleet.close()

    assert slot._proc is process
    assert fleet._launcher_temp is launcher
    assert launcher_cleanup == []
    assert fleet._closed is False
    with pytest.raises(RuntimeError, match="closing"):
        fleet.for_dataset("later")

    fleet.close()

    assert slot._proc is None
    assert fleet._launcher_temp is None
    assert launcher_cleanup == [True]
    assert fleet._closed is True
    assert process.wait_calls == 2
    assert process.stdout.close_calls == 2


def test_fleet_launcher_cleanup_failure_keeps_resource_for_retry(tmp_path):
    class RetryLauncher:
        def __init__(self):
            self.cleanup_calls = 0

        def cleanup(self):
            self.cleanup_calls += 1
            if self.cleanup_calls == 1:
                raise RuntimeError("launcher cleanup failed")

    fleet = Step3MosWorkerFleet(
        [0], verified_plan=_verified_pool_plan(tmp_path),
    )
    launcher = RetryLauncher()
    fleet._launcher_temp = launcher

    with pytest.raises(RuntimeError, match="launcher cleanup failed"):
        fleet.close()

    assert fleet._launcher_temp is launcher
    assert fleet._closed is False

    fleet.close()

    assert launcher.cleanup_calls == 2
    assert fleet._launcher_temp is None
    assert fleet._closed is True


def test_sparse_ordinal_sidecars_keep_1p2m_32tuple_request_below_limit(tmp_path):
    tuples = [
        (model,)
        for model in mos_worker._MODELS
    ]
    assert tuples
    tuples = [tuples[index % len(tuples)] for index in range(32)]
    row_count = 1_200_000 // len(tuples)
    subshards = []
    for index, models in enumerate(tuples):
        path = tmp_path / f"ordinals-{index}.jsonl"
        start = index
        sidecar = _write_ordinal_sidecar(
            str(path), range(start, 1_200_000, len(tuples)),
        )
        subshards.append({
            "models": list(models), "start_ordinal": start,
            "end_ordinal": 1_200_000, "row_count": row_count, **sidecar,
        })

    request = {"subshards": subshards}
    assert len(json.dumps(request).encode("utf-8")) < mos_pool._JSONL_MAX_LINE_BYTES
    observed = mos_gpu_job._OwnedGpuScoringJob._iter_ordinal_sidecar(
        {field: subshards[0][field] for field in mos_pool._ORDINAL_SIDECAR_FIELDS},
        0, 1_200_000, row_count,
    )
    assert next(observed) == 0
    for _ordinal in observed:
        pass
    assert sum(path.stat().st_size for path in tmp_path.glob("ordinals-*.jsonl")) > 8 * 1024 * 1024


@pytest.mark.parametrize("payload, error", [
    (b'{"ordinal":0}\n{"ordinal":0}\n', "ordinals"),
    (b'{"ordinal":0}\n{"ordinal":3}\n', "ordinals"),
    (b'{"ordinal":0}\n', "row count"),
])
def test_sparse_ordinal_sidecar_rejects_duplicate_out_of_range_or_short_rows(
    tmp_path, payload, error,
):
    path = tmp_path / "ordinals.jsonl"
    path.write_bytes(payload)
    sidecar = {
        "ordinal_sidecar_path": str(path),
        "ordinal_sidecar_sha256": hashlib.sha256(payload).hexdigest(),
        "ordinal_sidecar_size_bytes": len(payload),
    }

    with pytest.raises(ValueError, match=error):
        list(mos_gpu_job._OwnedGpuScoringJob._iter_ordinal_sidecar(sidecar, 0, 3, 2))


def test_sparse_ordinal_sidecar_rejects_digest_tampering(tmp_path):
    path = tmp_path / "ordinals.jsonl"
    sidecar = _write_ordinal_sidecar(str(path), [0, 2])
    path.write_bytes(b'{"ordinal":0}\n{"ordinal":1}\n')

    with pytest.raises(ValueError, match="digest mismatch"):
        list(mos_gpu_job._OwnedGpuScoringJob._iter_ordinal_sidecar(sidecar, 0, 3, 2))


def test_worker_request_sidecar_cleanup_preserves_primary_error(tmp_path, monkeypatch):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    request_path = tmp_path / "request.json"
    input_path.write_text(
        json.dumps({"ordinal": 0, "row": _row(0)}) + "\n", encoding="utf-8",
    )
    sidecar = _write_ordinal_sidecar(
        str(tmp_path / "ordinals.jsonl"), [0],
    )
    request = _worker_request(tmp_path, input_path, output_path, 1, models=("scoreq",))
    request["subshards"][0].update({
        "start_ordinal": 0, "end_ordinal": 2, **sidecar,
    })
    request_path.write_text(json.dumps(request), encoding="utf-8")
    close_calls = []

    class BrokenOrdinalIterator:
        def __iter__(self):
            return self

        def __next__(self):
            raise ValueError("sidecar traversal failed")

        def close(self):
            close_calls.append(True)
            raise RuntimeError("ordinal close failed")

    monkeypatch.setattr(
        mos_worker,
        "_iter_ordinal_sidecar",
        lambda *_args, **_kwargs: BrokenOrdinalIterator(),
    )

    subshard = mos_worker._load_request(str(request_path))["subshards"][0]
    with pytest.raises(ValueError, match="sidecar traversal failed") as raised:
        mos_worker._prevalidate_subshard_input(subshard, None)

    assert close_calls == [True]
    assert any("ordinal close failed" in note for note in raised.value.__notes__)


def test_worker_scores_sparse_sidecar_in_lockstep_and_rejects_tampered_input(
    tmp_path, monkeypatch,
):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    request_path = tmp_path / "request.json"
    input_path.write_text("".join(
        json.dumps({"ordinal": ordinal, "row": _row(ordinal)}) + "\n"
        for ordinal in (0, 2)
    ), encoding="utf-8")
    sidecar = _write_ordinal_sidecar(
        str(tmp_path / "ordinals.jsonl"), [0, 2],
    )
    request = _worker_request(tmp_path, input_path, output_path, 2, models=("scoreq",))
    request["subshards"][0].update({
        "start_ordinal": 0, "end_ordinal": 3, **sidecar,
    })
    request_path.write_text(json.dumps(request), encoding="utf-8")

    class Runtime(_FakeRuntimeAttestation):
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def source_identity(row):
            return row["source_key"]

        @contextmanager
        def open_source_group(self, _row, _models):
            yield self

        @staticmethod
        def score_batch(rows):
            return [{"scoreq": 4.1} for _row in rows]

    real_open = mos_worker._open_regular_stream
    opens = []

    def open_stream(path, label):
        opens.append((path, label))
        return real_open(path, label)

    monkeypatch.setattr(mos_worker, "_open_regular_stream", open_stream)
    monkeypatch.setattr(mos_worker, "Step3MosRuntime", Runtime)
    mos_worker.handle_request(str(request_path))
    assert [json.loads(line)["ordinal"] for line in output_path.read_text().splitlines()] == [0, 2]
    assert [label for _path, label in opens].count("MOS worker input shard") == 1
    assert [label for _path, label in opens].count("MOS sparse ordinal sidecar") == 1

    bad_input = input_path.read_text(encoding="utf-8").replace('"ordinal": 2', '"ordinal": 1')
    input_path.write_text(bad_input, encoding="utf-8")
    request["subshards"][0]["output_path"] = str(tmp_path / "tampered-output.jsonl")
    request["attestation_path"] = str(tmp_path / "tampered-attestation.json")
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="assigned ordinal assignment|digest mismatch"):
        mos_worker.handle_request(str(request_path))


def test_worker_rejects_cross_subshard_duplicate_ordinals_before_runtime(tmp_path, monkeypatch):
    inputs = []
    for index in range(2):
        path = tmp_path / f"input-{index}.jsonl"
        path.write_text(
            json.dumps({"ordinal": 0, "row": _row(index)}) + "\n",
            encoding="utf-8",
        )
        inputs.append(path)
    request = _worker_request(
        tmp_path, inputs[0], tmp_path / "output-0.jsonl", 1,
        models=("scoreq",),
    )
    request["subshards"] = [
        {
            "models": ["scoreq"],
            "input_path": str(path),
            "output_path": str(tmp_path / f"output-{index}.jsonl"),
            "row_count": 1,
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "start_ordinal": 0,
            "end_ordinal": 1,
        }
        for index, path in enumerate(inputs)
    ]
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setattr(
        mos_worker, "Step3MosRuntime",
        lambda **_kwargs: pytest.fail("runtime must not be constructed"),
    )
    monkeypatch.setattr(
        mos_worker, "_emit_progress",
        lambda *_args: pytest.fail("progress must not be emitted"),
    )

    with pytest.raises(ValueError, match="ordinals are not globally unique"):
        mos_worker.handle_request(str(request_path))

    assert not any((tmp_path / f"output-{index}.jsonl").exists() for index in range(2))
    assert not list(tmp_path.glob("audio-filter-step3-worker-ordinals-*"))


@pytest.mark.parametrize("tamper", ("replace_same_content", "append_tail"))
def test_worker_sidecar_barrier_precedes_runtime_output_and_progress(
    tmp_path, monkeypatch, tamper,
):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        json.dumps({"ordinal": 0, "row": _row(0)}) + "\n", encoding="utf-8",
    )
    sidecar_path = tmp_path / "ordinals.jsonl"
    sidecar = _write_ordinal_sidecar(str(sidecar_path), [0])
    request = _worker_request(
        tmp_path, input_path, output_path, 1, models=("scoreq",),
    )
    request["subshards"][0].update({"end_ordinal": 2, **sidecar})
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    real_verify = mos_worker._verify_validated_subshard
    barrier_calls = []

    def tamper_then_verify(subshard, validated):
        if not barrier_calls:
            if tamper == "replace_same_content":
                replacement = tmp_path / "replacement.jsonl"
                replacement.write_bytes(sidecar_path.read_bytes())
                os.replace(replacement, sidecar_path)
            else:
                with sidecar_path.open("ab") as stream:
                    stream.write(b'{"ordinal":1}\n')
        barrier_calls.append(True)
        return real_verify(subshard, validated)

    monkeypatch.setattr(mos_worker, "_verify_validated_subshard", tamper_then_verify)
    monkeypatch.setattr(
        mos_worker, "Step3MosRuntime",
        lambda **_kwargs: pytest.fail("runtime must not be constructed"),
    )
    monkeypatch.setattr(
        mos_worker, "_emit_progress",
        lambda *_args: pytest.fail("progress must not be emitted"),
    )

    with pytest.raises(ValueError, match="sidecar.*changed after validation"):
        mos_worker.handle_request(str(request_path))

    assert barrier_calls == [True]
    assert not output_path.exists()
    assert not Path(request["attestation_path"]).exists()


def test_worker_sidecar_cleanup_preserves_input_error_without_creating_target(
    tmp_path, monkeypatch,
):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        json.dumps({"ordinal": 1, "row": _row(0)}) + "\n", encoding="utf-8",
    )
    sidecar = _write_ordinal_sidecar(
        str(tmp_path / "ordinals.jsonl"), [0],
    )
    subshard = {
        "models": ["scoreq"], "input_path": str(input_path),
        "output_path": str(output_path), "row_count": 1,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "start_ordinal": 0, "end_ordinal": 2, **sidecar,
    }
    close_calls = []
    finalized = []

    class BrokenOrdinalIterator:
        def __iter__(self):
            return self

        def __next__(self):
            return 0

        def close(self):
            close_calls.append(True)
            raise RuntimeError("ordinal close failed")

    def finalize(streams, primary=None, *, sync):
        finalized.extend(streams)
        assert primary is not None
        for stream in streams:
            stream.close()

    monkeypatch.setattr(
        mos_worker,
        "_iter_ordinal_sidecar",
        lambda *_args, **_kwargs: BrokenOrdinalIterator(),
    )
    monkeypatch.setattr(mos_worker, "_finalize_streams", finalize)

    runtime = SimpleNamespace(
        open_waveform_coordinator=lambda _models: _EmptyWaveformCoordinator(),
    )
    with pytest.raises(ValueError, match="assigned ordinal assignment") as raised:
        mos_worker._score_subshard(runtime, subshard)

    assert close_calls == [True]
    assert len(finalized) == 2
    assert not output_path.exists()
    assert any("ordinal close failed" in note for note in raised.value.__notes__)


def test_parent_real_sidecar_finalize_failure_reaches_input_primary(tmp_path, monkeypatch):
    input_path, result_path, digest = _read_result_fixture(
        tmp_path,
        [{"ordinal": 0, "scores": _scores()}],
        input_ordinals=(1,),
    )
    sidecar_path = tmp_path / "ordinals.jsonl"
    sidecar = _write_ordinal_sidecar(
        str(sidecar_path), [0],
    )
    finalized = []
    sidecar_identity = os.stat(sidecar_path)
    real_finalize = mos_gpu_job._finalize_streams

    def finalize(streams, primary=None, *, sync):
        stream_list = tuple(streams)
        closes_sidecar = any(
            (os.fstat(stream.fileno()).st_dev, os.fstat(stream.fileno()).st_ino)
            == (sidecar_identity.st_dev, sidecar_identity.st_ino)
            for stream in stream_list
        )
        finalized.append((closes_sidecar, sync))
        real_finalize(stream_list, primary, sync=sync)
        if closes_sidecar:
            raise RuntimeError("sidecar finalization failed")

    monkeypatch.setattr(mos_gpu_job, "_finalize_streams", finalize)

    with pytest.raises(ValueError, match="ordinal or row") as raised:
        list(mos_gpu_job._OwnedGpuScoringJob._read_result(
            str(result_path), str(input_path), digest, 0, 2, sidecar,
        ))

    assert any(closes_sidecar for closes_sidecar, _sync in finalized)
    assert any("sidecar finalization failed" in note for note in raised.value.__notes__)


def test_worker_real_sidecar_finalize_failure_reaches_input_primary_before_target(
    tmp_path, monkeypatch,
):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        json.dumps({"ordinal": 1, "row": _row(0)}) + "\n", encoding="utf-8",
    )
    sidecar_path = tmp_path / "ordinals.jsonl"
    sidecar = _write_ordinal_sidecar(
        str(sidecar_path), [0],
    )
    subshard = {
        "models": ["scoreq"], "input_path": str(input_path),
        "output_path": str(output_path), "row_count": 1,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "start_ordinal": 0, "end_ordinal": 2, **sidecar,
    }
    finalized = []
    sidecar_identity = os.stat(sidecar_path)
    output_identity = None
    real_finalize = mos_worker._finalize_streams

    def finalize(streams, primary=None, *, sync):
        nonlocal output_identity
        stream_list = tuple(streams)
        identities = tuple(
            (os.fstat(stream.fileno()).st_dev, os.fstat(stream.fileno()).st_ino)
            for stream in stream_list
        )
        closes_sidecar = (
            sidecar_identity.st_dev, sidecar_identity.st_ino,
        ) in identities
        if sync:
            output_identity = identities[0]
        finalized.append((closes_sidecar, sync, identities))
        real_finalize(stream_list, primary, sync=sync)
        if closes_sidecar:
            raise RuntimeError("sidecar finalization failed")

    monkeypatch.setattr(mos_worker, "_finalize_streams", finalize)

    runtime = SimpleNamespace(
        open_waveform_coordinator=lambda _models: _EmptyWaveformCoordinator(),
    )
    with pytest.raises(ValueError, match="assigned ordinal assignment") as raised:
        mos_worker._score_subshard(runtime, subshard)

    assert any(closes_sidecar for closes_sidecar, _sync, _identities in finalized)
    assert not any(sync for _closes_sidecar, sync, _identities in finalized)
    assert output_identity is None
    assert not output_path.exists()
    assert any("sidecar finalization failed" in note for note in raised.value.__notes__)


def test_pool_resume_seals_completed_request_and_skips_it_on_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(mos_gpu_job, "MAX_WORKER_INPUT_SUBSHARDS", 1)
    task_path = tmp_path / "resume-tasks.jsonl"
    task_path.write_text("".join(
        json.dumps({
            "ordinal": ordinal,
            "row": _row(ordinal),
            "models": ["scoreq"],
            "scoring_policy_hash": ("a" if ordinal == 0 else "b") * 64,
        }) + "\n"
        for ordinal in range(2)
    ), encoding="utf-8")
    identity = Step3ResumeIdentity.create(Step3ResumeStore.identity_payload(
        dataset_id="alpha",
        step2_artifact_digest="1" * 64,
        step2_rows_sha256="2" * 64,
        step2_row_count=2,
        step2_lineage_digest="3" * 64,
        step3_config={"route": "scoreq"},
        model_provider={"provider": "test"},
        execution_policy={"scheduler": "weighted_lpt_ordinals"},
    ))
    context = DatasetRunContext(str(tmp_path), "alpha")
    calls = []

    def runner(_gpu, request_path, _cwd):
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        calls.append(request["subshards"][0]["start_ordinal"])
        subshard = request["subshards"][0]
        with open(subshard["output_path"], "x", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "ordinal": subshard["start_ordinal"],
                "scores": {"scoreq": 4.0},
            }) + "\n")
        _write_fake_worker_attestation(request)
        if len(calls) == 2:
            raise RuntimeError("interrupt after first request")

    def run_once(expect_error):
        pool = Step3MosScorerPool(
            [0], dataset_id="alpha",
            scoreq_options=_tree_options(tmp_path, "scoreq", "1" * 64),
            request_runner=runner,
        )
        store = Step3ResumeStore(str(tmp_path), "alpha", identity)
        pool.configure_resume(store, context.persistent_write_capability())
        try:
            if expect_error:
                with pytest.raises(RuntimeError, match="interrupt"):
                    list(pool.score_task_groups([{
                        "models": ["scoreq"], "path": str(task_path), "row_count": 2,
                    }]))
            else:
                return list(pool.score_task_groups([{
                    "models": ["scoreq"], "path": str(task_path), "row_count": 2,
                }]))
        finally:
            pool.clear_resume()

    context.__enter__()
    try:
        run_once(expect_error=True)
        assert calls == [0, 1]
        rows = run_once(expect_error=False)
        full_hit_rows = run_once(expect_error=False)
    finally:
        context.__exit__(None, None, None)
    assert calls == [0, 1, 1]
    assert [row["ordinal"] for row in rows] == [0, 1]
    assert full_hit_rows == rows


@pytest.mark.parametrize("gpu_ids", [[0, 0], [-1], [True], "0,1"])
def test_pool_rejects_invalid_physical_gpu_ids(gpu_ids):
    with pytest.raises(ValueError, match="gpu_ids"):
        Step3MosScorerPool(gpu_ids, dataset_id="alpha")


def _descriptor_budget_task_file(tmp_path, rows: int):
    task_path = tmp_path / "descriptor-budget-tasks.jsonl"
    task_path.write_text("".join(
        json.dumps({
            "ordinal": ordinal,
            "row": _row(ordinal),
            "models": ["scoreq"],
            "scoring_policy_hash": "b" * 64,
        }) + "\n"
        for ordinal in range(rows)
    ), encoding="utf-8")
    return task_path


def _recording_runner(requests_seen):
    def runner(_gpu, request_path, _cwd):
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        requests_seen.append([
            subshard["row_count"] for subshard in request["subshards"]
        ])
        for subshard in request["subshards"]:
            with open(subshard["input_path"], encoding="utf-8") as stream:
                ordinals = [json.loads(line)["ordinal"] for line in stream]
            with open(subshard["output_path"], "x", encoding="utf-8") as stream:
                for ordinal in ordinals:
                    stream.write(json.dumps({
                        "ordinal": ordinal,
                        "scores": {"scoreq": 4.0},
                    }) + "\n")
        _write_fake_worker_attestation(request)

    return runner


def test_pool_splits_requests_by_descriptor_byte_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(mos_gpu_job, "_REQUEST_DESCRIPTOR_BUDGET_BYTES", 700)
    task_path = _descriptor_budget_task_file(tmp_path, 6)
    requests_seen = []
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha",
        scoreq_options=_tree_options(tmp_path, "scoreq", "1" * 64),
        request_runner=_recording_runner(requests_seen),
    )
    try:
        rows = list(pool.score_task_groups([{
            "models": ["scoreq"], "path": str(task_path), "row_count": 6,
        }]))
    finally:
        pool.clear_resume()
    assert [row["ordinal"] for row in rows] == list(range(6))
    assert requests_seen == [[2], [2], [2]]


def test_pool_merges_file_size_parts_into_single_request(tmp_path, monkeypatch):
    monkeypatch.setattr(mos_gpu_job, "_JSONL_MAX_FILE_BYTES", 2_500)
    task_path = _descriptor_budget_task_file(tmp_path, 6)
    requests_seen = []
    pool = Step3MosScorerPool(
        [0], dataset_id="alpha",
        scoreq_options=_tree_options(tmp_path, "scoreq", "1" * 64),
        request_runner=_recording_runner(requests_seen),
    )
    try:
        rows = list(pool.score_task_groups([{
            "models": ["scoreq"], "path": str(task_path), "row_count": 6,
        }]))
    finally:
        pool.clear_resume()
    assert [row["ordinal"] for row in rows] == list(range(6))
    assert len(requests_seen) == 1
    assert len(requests_seen[0]) >= 2
    assert sum(requests_seen[0]) == 6
