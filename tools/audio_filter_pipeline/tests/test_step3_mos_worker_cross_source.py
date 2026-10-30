"""Step3 MOS worker 跨 source waveform 队列测试。"""

from __future__ import annotations

import json

import pytest

from audio_filter_pipeline.processors.workers import step3_mos_worker


pytestmark = pytest.mark.unit


def test_worker_batches_cross_source_within_physical_subshard_and_flushes_boundary(
    tmp_path, monkeypatch,
):
    created = []

    class Coordinator:
        def __init__(self, models):
            self.models = models
            self.queued = []
            self.finished = 0
            self.aborted = 0

        def push(self, row, ordinal):
            self.queued.append((ordinal, row["source"]))
            return []

        def finish(self):
            self.finished += 1
            return [
                (ordinal, {"scoreq": 4.0 + ordinal / 10})
                for ordinal, _source in self.queued
            ]

        def abort(self, *_args):
            self.aborted += 1

    class Runtime:
        @staticmethod
        def open_waveform_coordinator(models):
            coordinator = Coordinator(tuple(models))
            created.append(coordinator)
            return coordinator

    monkeypatch.setattr(step3_mos_worker, "_verify_validated_subshard", lambda *_args: None)
    monkeypatch.setattr(
        step3_mos_worker,
        "_validate_input_row",
        lambda payload, _ordinal, _dataset: json.loads(payload)["row"],
    )
    monkeypatch.setattr(step3_mos_worker, "_emit_progress", lambda _count: None)

    def score_subshard(start, sources):
        output_path = tmp_path / f"result-{start}.jsonl"
        input_path = tmp_path / f"input-{start}.jsonl"
        input_path.write_bytes(b"".join(
            json.dumps({"ordinal": start + index, "row": {"source": source}}).encode()
            + b"\n"
            for index, source in enumerate(sources)
        ))
        subshard = {
            "models": ["scoreq"],
            "output_path": str(output_path),
            "row_count": len(sources),
            "start_ordinal": start,
            "end_ordinal": start + len(sources),
        }
        with input_path.open("rb") as input_stream:
            step3_mos_worker._score_subshard(
                Runtime(), subshard,
                input_stream=input_stream, input_signature=(1, 2, 3, 4, 5),
            )
        return [json.loads(line) for line in output_path.read_text().splitlines()]

    first = score_subshard(0, ["source-a", "source-b", "source-a"])
    second = score_subshard(3, ["source-c"])

    assert [coordinator.models for coordinator in created] == [("scoreq",), ("scoreq",)]
    assert [coordinator.queued for coordinator in created] == [
        [(0, "source-a"), (1, "source-b"), (2, "source-a")],
        [(3, "source-c")],
    ]
    assert [coordinator.finished for coordinator in created] == [1, 1]
    assert [coordinator.aborted for coordinator in created] == [1, 1]
    assert [row["ordinal"] for row in first + second] == [0, 1, 2, 3]
    assert [row["scores"]["scoreq"] for row in first + second] == [4.0, 4.1, 4.2, 4.3]



def test_worker_requires_runtime_waveform_coordinator(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text('{"ordinal":0,"row":{}}\n', encoding="utf-8")
    subshard = {
        "models": ["scoreq"],
        "output_path": str(output_path),
        "row_count": 1,
        "start_ordinal": 0,
        "end_ordinal": 1,
    }

    with input_path.open("rb") as input_stream:
        with pytest.raises(AttributeError, match="open_waveform_coordinator"):
            step3_mos_worker._score_subshard(
                object(), subshard,
                input_stream=input_stream, input_signature=(1, 2, 3, 4, 5),
            )

    assert not output_path.exists()



def test_worker_request_and_attestation_bind_exact_execution_policy(tmp_path):
    request_path = tmp_path / "request.json"
    request = {
        "protocol": step3_mos_worker._PROTOCOL,
        "asset_load_binding": dict(step3_mos_worker.ASSET_LOAD_BINDING),
        "scoring_execution_policy": json.loads(json.dumps(
            step3_mos_worker.plain_json(step3_mos_worker.SCORING_EXECUTION_POLICY),
        )),
        "private_dir": str(tmp_path),
        "attestation_path": str(tmp_path / "attestation.jsonl"),
        "subshards": [{
            "models": ["scoreq"],
            "input_path": str(tmp_path / "input.jsonl"),
            "output_path": str(tmp_path / "output.jsonl"),
            "row_count": 1,
            "input_sha256": "a" * 64,
            "start_ordinal": 0,
            "end_ordinal": 1,
        }],
        "scoreq_options": {
            "model_dir": str(tmp_path / "scoreq-assets"),
            "verified_model_tree_sha256": "b" * 64,
        },
        "urgent_options": None,
        "dnsmos_options": None,
        "nisqa_options": None,
        "utmos_options": None,
    }
    request_path.write_text(json.dumps(request), encoding="utf-8")

    assert step3_mos_worker._load_request(str(request_path)) == request
    request["scoring_execution_policy"]["batching"]["cross_source"] = False
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="scoring execution policy"):
        step3_mos_worker._load_request(str(request_path))

    attestation_path = tmp_path / "attestation.jsonl"
    step3_mos_worker._write_runtime_attestation(
        str(attestation_path), ("scoreq",), [{
            "models": ["scoreq"], "attestation": {"selected_models": ["scoreq"]},
        }],
    )
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    assert set(attestation) == {
        "protocol", "asset_load_binding", "scoring_execution_policy", "worker_models",
        "model_tuple_attestations",
    }
    assert attestation["scoring_execution_policy"] == dict(
        step3_mos_worker.SCORING_EXECUTION_POLICY
    )
