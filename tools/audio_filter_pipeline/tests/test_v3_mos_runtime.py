"""Generation-3 Step3 MOS runtime adapter tests."""

from __future__ import annotations

import contextlib
import hashlib
import io
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import types
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from audio_filter_pipeline.processors import mos
from audio_filter_pipeline.processors.optional_runtime import (
    local_model_tree_sha256,
    open_owned_model_tree_snapshot,
)
from audio_filter_pipeline.processors.stages_v3 import mos_gpu_job, mos_pool, mos_runtime
from audio_filter_pipeline.processors.workers import step3_mos_worker
from audio_filter_pipeline.processors.stages_v3.mos_asset_preflight import (
    verify_mos_assets,
)
from audio_filter_pipeline.processors.stages_v3.mos_execution import (
    MosAssetSpec,
    MosExecutionPlan,
)
from audio_filter_pipeline.processors.stages_v3.mos_runtime import (
    MosRuntimeOptions,
    Step3MosRuntime,
)


pytestmark = pytest.mark.unit


def test_mos_runtime_does_not_expose_removed_result_helpers():
    assert not hasattr(mos_runtime, "_finite_real")
    assert not hasattr(mos_runtime, "_model_result")


@pytest.fixture
def row(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"placeholder")
    return {
        "input_root": str(tmp_path),
        "original_path": str(source),
        "source_relative_path": "source.wav",
        "source_size_bytes": len(b"placeholder"),
        "source_sha256": hashlib.sha256(b"placeholder").hexdigest(),
        "sample_rate": 16_000,
        "frames": 40,
        "channels": 1,
        "duration_sec": 40 / 16_000,
        "format": "WAV",
        "subtype": "PCM_16",
        "start_sample": 10,
        "end_sample": 30,
    }


@pytest.mark.parametrize(
    ("model", "raw_key", "score"),
    [
        ("nisqa", "nisqa_mos", 3.8),
        ("utmos", "utmos_score", 4.0),
    ],
)
def test_runtime_maps_registered_scalar_model_subset(row, model, raw_key, score):
    runtime = Step3MosRuntime(
        decoder=lambda _row: (np.zeros(40, dtype=np.float32), 16_000),
        compute_scores=lambda **_kwargs: {
            "audio_path": "<in-memory>",
            model: {raw_key: score},
        },
        options=MosRuntimeOptions(),
    )

    assert runtime(row, (model,)) == {model: score}


def test_runtime_decodes_exact_interval_and_maps_real_mos_result(row):
    decoded = np.column_stack((np.arange(40), np.arange(40) + 1000)).astype(np.float32)
    calls = []

    def decoder(source_row):
        assert source_row is row
        return decoded, 16_000

    def compute(*, audio_data, mos_types, **options):
        calls.append((audio_data, mos_types, options))
        return {
            "audio_path": "<in-memory>",
            "scoreq": {"scoreq_mos": 4.1},
            "urgent_pk": {"score": 3.7},
            "dnsmos": {
                "dnsmos_ovr_mos": 3.5,
                "dnsmos_sig_mos": 3.4,
                "dnsmos_bak_mos": 4.2,
                "dnsmos_p808_mos": 3.9,
            },
        }

    runtime = Step3MosRuntime(
        decoder=decoder,
        compute_scores=compute,
        options=MosRuntimeOptions(use_gpu=True, urgent_options=_URGENT_PROVIDER_OPTIONS),
    )

    assert runtime(row, ("scoreq", "urgent_pk", "dnsmos")) == {
        "scoreq": 4.1,
        "urgent_pk": 3.7,
        "dnsmos": {"ovrl": 3.5, "sig": 3.4, "bak": 4.2, "p808": 3.9},
    }
    assert len(calls) == 1
    audio_data, mos_types, options = calls[0]
    np.testing.assert_array_equal(audio_data[0], decoded[10:30, 0])
    assert audio_data[0].flags.owndata
    assert audio_data[0].base is None
    assert audio_data[1] == 16_000
    assert mos_types == ["scoreq", "urgent_pk", "dnsmos"]
    assert options == {
        "use_gpu": True,
        "use_cache": True,
        "model_specific_sr": True,
        "allow_fallback": False,
        "urgent_options": _URGENT_PROVIDER_OPTIONS,
        "scoreq_options": None,
        "dnsmos_options": None,
        "nisqa_options": None,
        "utmos_options": None,
    }


@pytest.mark.parametrize(
    ("result", "error"),
    [
        ({"scoreq": {"scoreq_mos": 4.1}}, "exactly match selected models"),
        ({
            "scoreq": {"scoreq_mos": 4.1},
            "urgent_pk": {"urgent_pk_mos": 3.7, "fallback": True},
            "dnsmos": {
                "dnsmos_ovr_mos": 3.5,
                "dnsmos_sig_mos": 3.4,
                "dnsmos_bak_mos": 4.2,
                "dnsmos_p808_mos": 3.9,
            },
        }, "urgent_pk result is fallback"),
        ({
            "scoreq": {"scoreq_mos": 4.1},
            "urgent_pk": {"score": 3.7},
            "dnsmos": {
                "dnsmos_ovr_mos": 3.5,
                "dnsmos_sig_mos": 3.4,
                "dnsmos_bak_mos": 4.2,
            },
        }, "dnsmos result keys"),
    ],
)
def test_runtime_rejects_partial_fallback_or_wrong_real_result_schema(row, result, error):
    runtime = Step3MosRuntime(
        decoder=lambda _row: (np.zeros(40, dtype=np.float32), 16_000),
        compute_scores=lambda **_kwargs: result,
        options=MosRuntimeOptions(),
    )

    with pytest.raises(ValueError, match=error):
        runtime(row, ("scoreq", "urgent_pk", "dnsmos"))


def test_runtime_rejects_path_or_interval_that_is_not_bound_to_source(row):
    runtime = Step3MosRuntime(
        decoder=lambda _row: (np.zeros(40, dtype=np.float32), 16_000),
        compute_scores=lambda **_kwargs: pytest.fail("compute must not run"),
        options=MosRuntimeOptions(),
    )
    row["original_path"] = str(Path(row["input_root"]).parent / "outside.wav")

    with pytest.raises(ValueError, match="original_path"):
        runtime(row, ("scoreq", "urgent_pk", "dnsmos"))


def _raw_result(score):
    return {
        "audio_path": "<in-memory>",
        "scoreq": {"scoreq_mos": score},
        "urgent_pk": {"score": score - 0.1},
        "dnsmos": {
            "dnsmos_ovr_mos": score - 0.2, "dnsmos_sig_mos": score - 0.3,
            "dnsmos_bak_mos": score - 0.4, "dnsmos_p808_mos": score - 0.5,
        },
    }


def test_runtime_batch_reuses_cross_source_sessions_and_maps_in_order(row):
    rows = [dict(row, start_sample=start, end_sample=end) for start, end in (
        (0, 5), (5, 10), (10, 15),
    )]
    other = dict(rows[2], source_relative_path="other.wav")
    other["original_path"] = str(Path(row["input_root"]) / "other.wav")
    rows = [rows[0], rows[1], other, rows[2]]
    sessions, batch_calls = [], []

    class Session:
        def __init__(self, source_row):
            self.source_row = source_row
            self.intervals = []
        def __enter__(self):
            sessions.append(self)
            return self
        def __exit__(self, *_args):
            return False
        def decode_interval(self, start, end):
            self.intervals.append((start, end))
            return np.arange(start, end, dtype=np.float32), 16_000

    def compute(items, *, mos_types, **options):
        batch_calls.append((items, mos_types, options))
        return [_raw_result(4.0 + index / 10) for index in range(len(items))]

    runtime = Step3MosRuntime(
        open_source_session=Session, compute_scores_batch=compute,
        options=MosRuntimeOptions(urgent_options=_URGENT_PROVIDER_OPTIONS),
    )

    scores = runtime.score_batch(rows, ("scoreq", "urgent_pk", "dnsmos"))

    assert [session.intervals for session in sessions] == [[(0, 5), (5, 10)], [(10, 15)], [(10, 15)]]
    assert [len(call[0]) for call in batch_calls] == [4]
    assert [score["scoreq"] for score in scores] == [4.0, 4.1, 4.2, 4.3]
    for items, mos_types, options in batch_calls:
        assert mos_types == ["scoreq", "urgent_pk", "dnsmos"]
        assert options == {
            "use_gpu": False, "use_cache": True, "batch_size": len(items),
            "model_specific_sr": True, "allow_fallback": False,
            "urgent_options": _URGENT_PROVIDER_OPTIONS,
            "scoreq_options": None,
            "dnsmos_options": None,
            "nisqa_options": None,
            "utmos_options": None,
        }
        assert all(audio.flags.owndata and audio.base is None for audio, _sr in items)


def test_source_group_reuses_one_session_across_bounded_calls_and_rejects_other_source(row):
    rows = [dict(row, start_sample=index, end_sample=index + 1) for index in range(32)]
    opened, batch_sizes = [], []

    class Session:
        def __enter__(self):
            opened.append(self)
            return self
        def __exit__(self, *_args):
            return False
        def decode_interval(self, start, end):
            return np.arange(start, end, dtype=np.float32), 16_000

    runtime = Step3MosRuntime(
        open_source_session=lambda _row: Session(),
        compute_scores_batch=lambda items, **_kwargs: (
            batch_sizes.append(len(items)) or [_raw_result(4.0) for _item in items]
        ),
        options=MosRuntimeOptions(),
    )
    models = ("scoreq", "urgent_pk", "dnsmos")

    with runtime.open_source_group(rows[0], models) as group:
        assert len(group.score_batch(rows[:20])) == 20
        assert len(group.score_batch(rows[20:])) == 12
        other = dict(rows[0], source_relative_path="other.wav")
        other["original_path"] = str(Path(row["input_root"]) / "other.wav")
        with pytest.raises(ValueError, match="immutable source identity"):
            group.score_batch([other])

    assert len(opened) == 1
    assert batch_sizes == [20, 12]


def test_source_group_microbatches_by_decoded_float32_bytes(row, monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import mos_runtime

    rows = [dict(row, start_sample=index, end_sample=index + 1) for index in range(3)]
    batch_sizes = []

    class Session:
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
        def decode_interval(self, _start, _end):
            return np.zeros(3, dtype=np.float32), 16_000

    monkeypatch.setattr(mos_runtime, "_MAX_BATCH_FLOAT32_BYTES", 20)
    runtime = Step3MosRuntime(
        open_source_session=lambda _row: Session(),
        compute_scores_batch=lambda items, **_kwargs: (
            batch_sizes.append(len(items)) or [_raw_result(4.0) for _item in items]
        ),
        options=MosRuntimeOptions(),
    )
    with runtime.open_source_group(rows[0], ("scoreq", "urgent_pk", "dnsmos")) as group:
        assert len(group.score_batch(rows)) == 3

    assert batch_sizes == [1, 1, 1]


def test_source_group_selects_first_channel_and_owns_mono_waveform(row):
    owned = np.column_stack((np.arange(5), np.arange(5) + 100)).astype(np.float32)
    observed = []

    class Session:
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
        def decode_interval(self, _start, _end):
            return owned, 16_000

    def compute(items, **_kwargs):
        observed.append(items[0][0])
        return [_raw_result(4.0)]

    runtime = Step3MosRuntime(
        open_source_session=lambda _row: Session(),
        compute_scores_batch=compute,
        options=MosRuntimeOptions(),
    )
    with runtime.open_source_group(row, ("scoreq", "urgent_pk", "dnsmos")) as group:
        group.score_batch([dict(row, start_sample=0, end_sample=5)])

    np.testing.assert_array_equal(observed[0], owned[:, 0])
    assert observed[0].ndim == 1
    assert observed[0].dtype == np.float32
    assert observed[0].flags.owndata
    assert observed[0].base is None
    assert observed[0] is not owned


def test_runtime_canonical_task_groups_batch_same_source_and_preserve_schema(
    row, tmp_path,
):
    opened = []

    class Session(_FakeSession):
        def __enter__(self):
            opened.append(self)
            return self

    tasks = [
        {
            "ordinal": ordinal,
            "row": dict(row, start_sample=ordinal, end_sample=ordinal + 1),
            "models": ["scoreq", "urgent_pk", "dnsmos"],
            "scoring_policy_hash": "a" * 64,
        }
        for ordinal in range(2)
    ]
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        "".join(__import__("json").dumps(task) + "\n" for task in tasks),
        encoding="utf-8",
    )
    batch_sizes = []
    runtime = Step3MosRuntime(
        open_source_session=lambda _row: Session(),
        compute_scores_batch=lambda items, **_kwargs: (
            batch_sizes.append(len(items))
            or [_raw_result(4.0 + index / 10) for index in range(len(items))]
        ),
        options=MosRuntimeOptions(),
    )

    output = list(runtime.score_task_groups([{
        "models": tasks[0]["models"], "path": str(path), "row_count": 2,
    }]))

    assert [item["ordinal"] for item in output] == [0, 1]
    assert all(set(item) == {"ordinal", "row", "models", "scores"} for item in output)
    assert [item["scores"]["scoreq"] for item in output] == [4.0, 4.1]
    assert batch_sizes == [2]
    assert len(opened) == 1


def test_runtime_task_groups_batch_cross_source_and_preserve_ordinal_order(row, tmp_path):
    first = dict(row, start_sample=0, end_sample=1)
    second = dict(first, source_relative_path="other.wav")
    second["original_path"] = str(Path(row["input_root"]) / "other.wav")
    tasks = [
        {
            "ordinal": ordinal,
            "row": source_row,
            "models": ["scoreq", "urgent_pk", "dnsmos"],
            "scoring_policy_hash": "a" * 64,
        }
        for ordinal, source_row in enumerate((first, second, first))
    ]
    path = tmp_path / "tasks-cross-source.jsonl"
    path.write_text(
        "".join(__import__("json").dumps(task) + "\n" for task in tasks),
        encoding="utf-8",
    )
    opens, batch_sizes = [], []

    class Session(_FakeSession):
        def __init__(self, source_row):
            self.source = source_row["source_relative_path"]

        def __enter__(self):
            opens.append(self.source)
            return self

    runtime = Step3MosRuntime(
        open_source_session=Session,
        compute_scores_batch=lambda items, **_kwargs: (
            batch_sizes.append(len(items))
            or [_raw_result(4.0 + index / 10) for index in range(len(items))]
        ),
        options=MosRuntimeOptions(),
    )

    output = list(runtime.score_task_groups([{
        "models": tasks[0]["models"], "path": str(path), "row_count": len(tasks),
    }]))

    assert opens == ["source.wav", "other.wav", "source.wav"]
    assert batch_sizes == [3]
    assert [item["ordinal"] for item in output] == [0, 1, 2]
    assert [item["scores"]["scoreq"] for item in output] == [4.0, 4.1, 4.2]


@pytest.mark.parametrize("mode", ("normal", "business", "cancel"))
def test_runtime_task_group_owned_stream_cleanup_semantics(
    row, tmp_path, monkeypatch, mode,
):
    task = {
        "ordinal": 0,
        "row": row,
        "models": ["scoreq", "urgent_pk", "dnsmos"],
        "scoring_policy_hash": "a" * 64,
    }
    path = tmp_path / "tasks.jsonl"
    if mode == "business":
        path.write_text("{}\n", encoding="utf-8")
    else:
        path.write_text(__import__("json").dumps(task) + "\n", encoding="utf-8")
    real_open = mos_runtime.open_regular_stream
    opened = []

    class CloseFailingStream:
        def __init__(self, stream):
            self._stream = stream

        def __getattr__(self, name):
            return getattr(self._stream, name)

        def close(self):
            self._stream.close()
            raise RuntimeError("owned task stream close failed")

    def open_stream(path_value, label):
        stream, initial = real_open(path_value, label)
        wrapped = CloseFailingStream(stream)
        opened.append(wrapped)
        return wrapped, initial

    monkeypatch.setattr(mos_runtime, "open_regular_stream", open_stream)
    tasks = mos_runtime.Step3MosRuntime._iter_task_group(
        str(path), tuple(task["models"]), 1,
    )
    if mode == "normal":
        with pytest.raises(RuntimeError, match="owned task stream close failed"):
            list(tasks)
    elif mode == "business":
        with pytest.raises(ValueError, match="task row is invalid") as raised:
            list(tasks)
        assert any("owned task stream close failed" in note for note in raised.value.__notes__)
    else:
        assert next(tasks)["ordinal"] == 0
        with pytest.raises(RuntimeError, match="owned task stream close failed"):
            tasks.close()
    assert len(opened) == 1


def test_runtime_chains_physical_parts_as_one_logical_tuple(row, tmp_path):
    models = ["scoreq", "urgent_pk", "dnsmos"]
    tasks = [
        {
            "ordinal": ordinal,
            "row": dict(row, start_sample=ordinal, end_sample=ordinal + 1),
            "models": models,
            "scoring_policy_hash": "a" * 64,
        }
        for ordinal in range(2)
    ]
    paths = []
    for index, task in enumerate(tasks):
        path = tmp_path / f"part-{index}.jsonl"
        path.write_text(__import__("json").dumps(task) + "\n", encoding="utf-8")
        paths.append(path)
    batch_sizes = []
    runtime = Step3MosRuntime(
        open_source_session=lambda _row: _FakeSession(),
        compute_scores_batch=lambda items, **_kwargs: (
            batch_sizes.append(len(items))
            or [_raw_result(4.0 + index / 10) for index in range(len(items))]
        ),
        options=MosRuntimeOptions(),
    )

    iterator = runtime.score_task_groups([
        {"models": models, "path": str(path), "row_count": 1}
        for path in paths
    ])
    output = []
    while True:
        try:
            output.append(next(iterator))
        except StopIteration as completion:
            attestation = completion.value
            break

    assert [item["ordinal"] for item in output] == [0, 1]
    assert batch_sizes == [2]
    assert len(attestation["model_tuple_attestations"]) == 1
    assert attestation["model_tuple_attestations"][0]["models"] == models


def test_runtime_tuple_chain_cancel_propagates_part_cleanup_failure(row, tmp_path, monkeypatch):
    task = {
        "ordinal": 0,
        "row": row,
        "models": ["scoreq", "urgent_pk", "dnsmos"],
        "scoring_policy_hash": "a" * 64,
    }

    class PartIterator:
        def __init__(self):
            self.ordinal = 0

        def __iter__(self):
            return self

        def __next__(self):
            current = dict(task, ordinal=self.ordinal)
            self.ordinal += 1
            return current

        @staticmethod
        def close():
            raise RuntimeError("tuple part close failed")

    runtime = Step3MosRuntime(
        open_source_session=lambda _row: _FakeSession(),
        compute_scores_batch=lambda items, **_kwargs: [
            _raw_result(4.0) for _item in items
        ],
        options=MosRuntimeOptions(),
    )
    monkeypatch.setattr(runtime, "_iter_task_group", lambda *_args: PartIterator())
    results = runtime.score_task_groups([{
        "models": task["models"],
        "path": str(tmp_path / "unused.jsonl"),
        "row_count": 1,
    }])

    assert next(results)["ordinal"] == 0
    with pytest.raises(RuntimeError, match="tuple part close failed"):
        results.close()


def test_runtime_rejects_nonincreasing_ordinals_across_tuple_parts(row, tmp_path):
    models = ["scoreq", "urgent_pk", "dnsmos"]
    paths = []
    for index in range(2):
        path = tmp_path / f"part-{index}.jsonl"
        path.write_text(__import__("json").dumps({
            "ordinal": 0,
            "row": dict(row, start_sample=index, end_sample=index + 1),
            "models": models,
            "scoring_policy_hash": "a" * 64,
        }) + "\n", encoding="utf-8")
        paths.append(path)
    runtime = Step3MosRuntime(
        open_source_session=lambda _row: _FakeSession(),
        compute_scores_batch=lambda items, **_kwargs: [
            _raw_result(4.0) for _item in items
        ],
        options=MosRuntimeOptions(),
    )

    with pytest.raises(ValueError, match="part ordinals must be strictly increasing"):
        list(runtime.score_task_groups([
            {"models": models, "path": str(path), "row_count": 1}
            for path in paths
        ]))


@pytest.mark.parametrize("descriptor_case", ["duplicate_path", "tuple_a_b_a"])
def test_runtime_rejects_invalid_part_descriptor_chains_before_read(
    row, tmp_path, descriptor_case,
):
    runtime = Step3MosRuntime(
        open_source_session=lambda _row: pytest.fail("must not open source"),
        compute_scores_batch=lambda *_args, **_kwargs: pytest.fail("must not score"),
        options=MosRuntimeOptions(),
    )
    first = str(tmp_path / "missing-first.jsonl")
    second = str(tmp_path / "missing-second.jsonl")
    if descriptor_case == "duplicate_path":
        groups = [
            {"models": ["scoreq"], "path": first, "row_count": 1},
            {"models": ["scoreq"], "path": first, "row_count": 1},
        ]
        error = "path is duplicate"
    else:
        groups = [
            {"models": ["scoreq"], "path": first, "row_count": 1},
            {"models": ["utmos"], "path": second, "row_count": 1},
            {
                "models": ["scoreq"],
                "path": str(tmp_path / "missing-third.jsonl"),
                "row_count": 1,
            },
        ]
        error = "parts must be contiguous"

    with pytest.raises(ValueError, match=error):
        list(runtime.score_task_groups(groups))


@pytest.mark.parametrize("batch_result", [[], [_raw_result(4.0), _raw_result(4.1)], [{"scoreq": {}}]])
def test_runtime_batch_rejects_mismatched_or_invalid_result_schema(row, batch_result):
    runtime = Step3MosRuntime(
        open_source_session=lambda _row: _FakeSession(),
        compute_scores_batch=lambda *_args, **_kwargs: batch_result,
        options=MosRuntimeOptions(),
    )

    with pytest.raises(ValueError):
        runtime.score_batch(
            [dict(row, start_sample=0, end_sample=5)],
            ("scoreq", "urgent_pk", "dnsmos"),
        )


class _FakeSession:
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        return False
    def decode_interval(self, start, end):
        return np.zeros(end - start, dtype=np.float32), 16_000


def test_runtime_batch_rejects_unbounded_input_before_opening_session(row):
    opened = False
    runtime = Step3MosRuntime(
        open_source_session=lambda _row: pytest.fail("session must not open"),
        compute_scores_batch=lambda *_args, **_kwargs: pytest.fail("compute must not run"),
        options=MosRuntimeOptions(),
    )

    models = ("scoreq", "urgent_pk", "dnsmos")
    with pytest.raises(ValueError, match="1 to 32"):
        runtime.score_batch((row for _ in range(1)), models)
    with pytest.raises(ValueError, match="1 to 32"):
        runtime.score_batch([row] * 33, models)
    assert not opened


class _FakeParameters:
    def __init__(self, device):
        self._device = device

    def parameters(self):
        return iter((types.SimpleNamespace(device=self._device),))


class _FakeOnnxSession:
    def __init__(self, providers):
        self._providers = providers

    def get_providers(self):
        return list(self._providers)


def test_scoreq_gpu_session_hardening_preserves_cuda_primary_hybrid_ep(monkeypatch, tmp_path):
    calls = []

    class FakeSessionOptions:
        def __init__(self):
            self.entries = {}

        def add_session_config_entry(self, key, value):
            self.entries[key] = value

    class FakeSession:
        def __init__(self, path, options, providers):
            calls.append((path, options, providers))

        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        InferenceSession=FakeSession,
    ))
    scorer = types.SimpleNamespace(_sess=object(), _device_str="CPU")

    mos._harden_scoreq_gpu_session(scorer, str(tmp_path))

    path, options, providers = calls[0]
    assert path == str(tmp_path / "adapt_nr_telephone.onnx")
    assert "session.disable_cpu_ep_fallback" not in options.entries
    assert providers == [
        ("CUDAExecutionProvider", {
            "arena_extend_strategy": "kSameAsRequested",
            "gpu_mem_limit": 16 * 1024 * 1024 * 1024,
        }),
        "CPUExecutionProvider",
    ]
    assert scorer._sess.get_providers() == [
        "CUDAExecutionProvider", "CPUExecutionProvider",
    ]
    assert scorer._device_str == "GPU"


def test_scoreq_reset_uses_owned_snapshot_manifest_and_single_loader_lease(
    monkeypatch,
):
    from core.metrics import mos_models

    calls = []

    class Snapshot:
        @mos.contextlib.contextmanager
        def loader_path_lease(self):
            calls.append("lease-enter")
            yield "/proc/self/fd/123"
            calls.append("lease-exit")

    replacement = object()
    estimator = mos.SCOREQEstimator.__new__(mos.SCOREQEstimator)
    estimator._snapshot = Snapshot()
    estimator._scoreq = object()
    estimator.use_gpu = False
    estimator._allow_fallback = False
    monkeypatch.setattr(
        mos,
        "_verify_owned_model_tree_snapshot",
        lambda *_args: pytest.fail("reset must not perform a full content hash"),
    )
    monkeypatch.setattr(
        mos,
        "_verify_owned_model_tree_manifest",
        lambda *_args: calls.append("manifest"),
    )
    monkeypatch.setattr(
        mos_models,
        "load_scoreq",
        lambda **kwargs: calls.append(("load", kwargs)) or replacement,
    )

    estimator.reset()

    assert estimator._scoreq is replacement
    assert calls == [
        "lease-enter",
        "manifest",
        ("load", {"model_dir": "/proc/self/fd/123", "use_gpu": False}),
        "manifest",
        "lease-exit",
    ]


def test_scoreq_inference_failure_is_not_replaced_by_energy_fallback():
    estimator = mos.SCOREQEstimator.__new__(mos.SCOREQEstimator)
    estimator._scoreq = lambda _audio, _sr: (_ for _ in ()).throw(
        RuntimeError("GPU inference failed")
    )
    estimator._allow_fallback = False

    with pytest.raises(RuntimeError, match="SCOREQ inference failed.*fallback"):
        estimator.compute(audio_data=(np.zeros(160, dtype=np.float32), 16_000))


def test_scoreq_direct_single_and_batch_select_first_channel():
    observed_single = []
    observed_batch = []

    class FakeScoreq:
        def __call__(self, audio, _sr):
            observed_single.append(audio)
            return {"MOS": 4.0}

        def run_batch(self, items):
            observed_batch.extend(audio for audio, _sr in items)
            return [{"MOS": 4.0} for _item in items]

    estimator = mos.SCOREQEstimator.__new__(mos.SCOREQEstimator)
    estimator._scoreq = FakeScoreq()
    estimator._allow_fallback = False
    estimator._batch_since_reset = 0
    stereo = np.column_stack((np.arange(160), np.arange(160) + 1000)).astype(np.float32)

    assert estimator.compute(audio_data=(stereo, 16_000)) == {"scoreq_mos": 4.0}
    assert estimator.compute_batch([(stereo, 16_000)]) == [{"scoreq_mos": 4.0}]
    for received in (*observed_single, *observed_batch):
        np.testing.assert_array_equal(received, stereo[:, 0])
        assert received.ndim == 1
    assert observed_single[0].dtype == np.float64
    assert observed_batch[0].dtype == np.float32


@pytest.mark.parametrize(
    "invalid_result",
    [{}, {"MOS": True}, {"MOS": "3.0"}, {"MOS": float("nan")},
     {"MOS": float("inf")}],
)
def test_scoreq_single_result_schema_is_fail_closed(invalid_result):
    estimator = mos.SCOREQEstimator.__new__(mos.SCOREQEstimator)
    estimator._scoreq = lambda _audio, _sr: invalid_result
    estimator._allow_fallback = False

    with pytest.raises(RuntimeError, match="SCOREQ inference failed.*fallback"):
        estimator.compute(audio_data=(np.zeros(160, dtype=np.float32), 16_000))


@pytest.mark.parametrize(
    "invalid_result",
    [{}, {"MOS": True}, {"MOS": "3.0"}, {"MOS": float("nan")},
     {"MOS": float("inf")}],
)
def test_scoreq_batch_result_schema_is_fail_closed(invalid_result):
    class FakeScoreq:
        def run_batch(self, _items):
            return [invalid_result]

        def __call__(self, _audio, _sr):
            pytest.fail("invalid batch schema must not retry through per-item inference")

    estimator = mos.SCOREQEstimator.__new__(mos.SCOREQEstimator)
    estimator._scoreq = FakeScoreq()
    estimator._allow_fallback = False
    estimator._batch_since_reset = 0

    with pytest.raises(mos.MosResultContractError, match="SCOREQ"):
        estimator.compute_batch([(np.zeros(160, dtype=np.float32), 16_000)])


def test_scoreq_schema_failure_uses_explicit_compatibility_fallback():
    estimator = mos.SCOREQEstimator.__new__(mos.SCOREQEstimator)
    estimator._scoreq = lambda _audio, _sr: {}
    estimator._allow_fallback = True

    result = estimator.compute(audio_data=(np.zeros(160, dtype=np.float32), 16_000))

    assert result["fallback"] is True
    assert result["scoreq_mos"] == pytest.approx(3.0)


@pytest.mark.parametrize(
    "batch_result",
    [
        [[3.0]],
        [{}],
        [{"MOS": True}],
        [{"MOS": "3.0"}],
        [{"MOS": float("nan")}],
        [{"MOS": float("inf")}],
        [],
    ],
)
def test_generation3_batch_contract_error_never_retries_single(
    monkeypatch, batch_result,
):
    single_calls = []

    class FakeScoreq:
        def run_batch(self, _items):
            return batch_result

        def __call__(self, _audio, _sr):
            single_calls.append(True)
            return {"MOS": 4.0}

    estimator = mos.SCOREQEstimator.__new__(mos.SCOREQEstimator)
    estimator._scoreq = FakeScoreq()
    estimator._allow_fallback = False
    estimator._batch_since_reset = 0
    monkeypatch.setattr(mos, "get_mos_estimator", lambda *_args, **_kwargs: estimator)

    with pytest.raises(mos.MosResultContractError):
        mos.compute_mos_scores_batch(
            [(np.zeros(160, dtype=np.float32), 16_000)],
            ["scoreq"],
            use_gpu=True,
            use_cache=False,
            allow_fallback=False,
        )

    assert single_calls == []


def test_generation3_batch_execution_error_still_retries_single(monkeypatch):
    single_calls = []

    class FakeScoreq:
        def run_batch(self, _items):
            raise RuntimeError("transient execution failure")

        def __call__(self, _audio, _sr):
            single_calls.append(True)
            return {"MOS": 4.0}

    estimator = mos.SCOREQEstimator.__new__(mos.SCOREQEstimator)
    estimator._scoreq = FakeScoreq()
    estimator._allow_fallback = False
    estimator._batch_since_reset = 0
    monkeypatch.setattr(mos, "get_mos_estimator", lambda *_args, **_kwargs: estimator)

    scores = mos.compute_mos_scores_batch(
        [(np.zeros(160, dtype=np.float32), 16_000)],
        ["scoreq"],
        use_gpu=True,
        use_cache=False,
        allow_fallback=False,
    )

    assert scores[0]["scoreq"] == {"scoreq_mos": 4.0}
    assert single_calls == [True]


def test_clear_model_cache_waits_for_active_compute(monkeypatch):
    compute_started = threading.Event()
    release_compute = threading.Event()

    class BlockingScoreq:
        def __call__(self, _audio, _sr):
            compute_started.set()
            assert release_compute.wait(timeout=5)
            return {"MOS": 4.0}

    estimator = mos.SCOREQEstimator.__new__(mos.SCOREQEstimator)
    scorer = BlockingScoreq()
    estimator._scoreq = scorer
    estimator._allow_fallback = False
    estimator._closed = False
    estimator._snapshot = None
    cleanup_entered = threading.Event()
    real_retire = mos._retire_estimators

    def retire_with_signal(estimators, *, wait_for_leases):
        if wait_for_leases:
            cleanup_entered.set()
        return real_retire(estimators, wait_for_leases=wait_for_leases)

    monkeypatch.setattr(mos, "_retire_estimators", retire_with_signal)
    monkeypatch.setattr(mos, "_MODEL_CACHE", {"scoreq": estimator})
    monkeypatch.setattr(
        mos, "clear_urgent_pk_checkpoint_metadata_cache", lambda: None,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        compute_future = executor.submit(
            estimator.compute,
            audio_data=(np.zeros(160, dtype=np.float32), 16_000),
        )
        assert compute_started.wait(timeout=5)
        clear_future = executor.submit(mos.clear_model_cache)
        # 等清理路径真正进入租约等待后再断言，固定 sleep 会让未调度的竞争线程假通过。
        assert cleanup_entered.wait(timeout=5)
        assert not clear_future.done()
        assert estimator._scoreq is scorer
        release_compute.set()
        assert compute_future.result(timeout=5) == {"scoreq_mos": 4.0}
        clear_future.result(timeout=5)

    assert estimator._scoreq is None


def test_concurrent_estimator_close_executes_cleanup_once():
    close_started = threading.Event()
    release_close = threading.Event()

    class Snapshot:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            close_started.set()
            assert release_close.wait(timeout=5)

    estimator = mos.SCOREQEstimator.__new__(mos.SCOREQEstimator)
    snapshot = Snapshot()
    estimator._scoreq = object()
    estimator._snapshot = snapshot
    estimator._closed = False
    real_close = estimator.close
    second_close_entered = threading.Event()

    def close_and_signal():
        # close_started 已由第一次 close 置位，因此这里必定是第二次调用。
        if close_started.is_set():
            second_close_entered.set()
        return real_close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(estimator.close)
        assert close_started.wait(timeout=5)
        second = executor.submit(close_and_signal)
        # 等第二次 close 真正进入互斥等待后再断言，固定 sleep 会让未调度的竞争线程假通过。
        assert second_close_entered.wait(timeout=5)
        assert not second.done()
        release_close.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert snapshot.close_calls == 1
    assert estimator._closed is True


def _write_scoreq_asset(root: Path, payload: bytes = b"scoreq-original") -> str:
    root.mkdir()
    (root / "adapt_nr_telephone.onnx").write_bytes(payload)
    return local_model_tree_sha256(str(root))


def _write_dnsmos_asset(
    root: Path,
    asset_id: str,
    *,
    score: float = 4.0,
    fail_exec: bool = False,
    fail_gpu: bool = False,
    fail_inference: bool = False,
) -> str:
    root.mkdir()
    (root / "sig_bak_ovr.onnx").write_bytes(f"primary-{asset_id}".encode())
    (root / "model_v8.onnx").write_bytes(f"p808-{asset_id}".encode())
    if fail_exec:
        source = 'raise RuntimeError("module exec exploded")\n'
    else:
        source = f'''\
from pathlib import Path

ASSET_ID = {asset_id!r}
ASSET_SCORE = {score!r}
MODULE_TOKEN = object()
FAIL_GPU = {fail_gpu!r}
FAIL_INFERENCE = {fail_inference!r}

class _Session:
    def get_providers(self):
        return ["CPUExecutionProvider"]

class ComputeScore:
    def __init__(self, primary, p808, use_gpu=False):
        if use_gpu and FAIL_GPU:
            raise RuntimeError("strict CUDA session failed")
        self.asset_id = ASSET_ID
        self.module_token = MODULE_TOKEN
        self.primary_path = primary
        self.p808_path = p808
        self.primary_bytes = Path(primary).read_bytes()
        self.p808_bytes = Path(p808).read_bytes()
        self.use_gpu = use_gpu
        self.onnx_sess = _Session()
        self.p808_onnx_sess = _Session()

    def run_batch_audio(self, items, **_kwargs):
        assert Path(self.primary_path).read_bytes() == self.primary_bytes
        assert Path(self.p808_path).read_bytes() == self.p808_bytes
        if FAIL_INFERENCE:
            raise RuntimeError("inference exploded")
        return [{{
            "P808_MOS": ASSET_SCORE,
            "SIG": ASSET_SCORE,
            "BAK": ASSET_SCORE,
            "OVRL": ASSET_SCORE,
        }} for _item in items]
'''
    (root / "dnsmos_local.py").write_text(source, encoding="utf-8")
    return local_model_tree_sha256(str(root))


def _install_fake_scoreq_loader(monkeypatch, loaded, *, inference_error=False):
    from core.metrics import mos_models

    class FakeScoreq:
        _device_str = "CPU"
        _sess = _FakeOnnxSession(["CPUExecutionProvider"])

        def __init__(self, model_dir):
            self.model_dir = model_dir
            self.model_bytes = (
                Path(model_dir) / "adapt_nr_telephone.onnx"
            ).read_bytes()

        def __call__(self, _audio, _sr):
            assert (
                Path(self.model_dir) / "adapt_nr_telephone.onnx"
            ).read_bytes() == self.model_bytes
            if inference_error:
                raise RuntimeError("inference exploded")
            return {"MOS": 4.0}

        def run_batch(self, items):
            if inference_error:
                raise RuntimeError("inference exploded")
            return [{"MOS": 4.0} for _item in items]

    def load_scoreq(*, model_dir, use_gpu):
        scorer = FakeScoreq(model_dir)
        loaded.append((model_dir, scorer.model_bytes, use_gpu))
        return scorer

    monkeypatch.setattr(mos_models, "load_scoreq", load_scoreq)


def _track_owned_snapshots(monkeypatch):
    created = []

    def tracked_open(path, **kwargs):
        snapshot = open_owned_model_tree_snapshot(path, **kwargs)
        created.append(snapshot)
        return snapshot

    monkeypatch.setattr(mos, "open_owned_model_tree_snapshot", tracked_open)
    return created


class _SwapPublicPathAfterVerify:
    def __init__(self, snapshot, install_replacement, *, attack_after):
        self._snapshot = snapshot
        self._install_replacement = install_replacement
        self._attack_after = attack_after
        self._verify_calls = 0
        self._held_path = Path(snapshot.path + "-owned-held")
        self._replacement_held_path = Path(snapshot.path + "-replacement-held")
        self.attacked = False
        self.close_errors = []

    def __getattr__(self, name):
        return getattr(self._snapshot, name)

    def verify(self):
        digest = self._snapshot.verify()
        self._verify_calls += 1
        if not self.attacked and self._verify_calls == self._attack_after:
            Path(self.path).rename(self._held_path)
            self._install_replacement(Path(self.path))
            self.attacked = True
        return digest

    def close(self):
        try:
            return self._snapshot.close()
        except BaseException as exc:
            self.close_errors.append(exc)
            raise

    def restore_and_close(self):
        import shutil

        replacement_held_by_test = False
        try:
            if Path(self.path).exists():
                Path(self.path).rename(self._replacement_held_path)
                replacement_held_by_test = True
            if self._held_path.exists():
                self._held_path.rename(self.path)
            self._snapshot.close()
        finally:
            if replacement_held_by_test:
                shutil.rmtree(self._replacement_held_path)


@pytest.mark.parametrize("close_raises", (False, True))
def test_swap_public_snapshot_cleanup_removes_only_its_replacement(
    tmp_path, close_raises,
):
    import shutil

    public_path = tmp_path / "snapshot"
    public_path.mkdir()
    (public_path / "model.bin").write_bytes(b"owned")
    historical = tmp_path / "historical-replacement-held"
    historical.mkdir()
    (historical / "sentinel").write_bytes(b"keep")

    class Snapshot:
        path = str(public_path)

        def verify(self):
            return "digest"

        def close(self):
            if close_raises:
                raise RuntimeError("injected close failure")
            shutil.rmtree(self.path)

    def install_replacement(path):
        path.mkdir()
        (path / "model.bin").write_bytes(b"replacement")

    snapshot = _SwapPublicPathAfterVerify(
        Snapshot(), install_replacement, attack_after=1,
    )
    snapshot.verify()
    try:
        if close_raises:
            with pytest.raises(RuntimeError, match="injected close failure"):
                snapshot.restore_and_close()
        else:
            snapshot.restore_and_close()
        replacement_was_left = snapshot._replacement_held_path.exists()
    finally:
        if snapshot._replacement_held_path.exists():
            shutil.rmtree(snapshot._replacement_held_path)

    assert not replacement_was_left
    assert (historical / "sentinel").read_bytes() == b"keep"


def _swap_public_snapshot_path_after_verify(
    monkeypatch, install_replacement, *, attack_after=1,
):
    created = []

    def attacking_open(path, **kwargs):
        snapshot = _SwapPublicPathAfterVerify(
            open_owned_model_tree_snapshot(path, **kwargs),
            install_replacement,
            attack_after=attack_after,
        )
        created.append(snapshot)
        return snapshot

    monkeypatch.setattr(mos, "open_owned_model_tree_snapshot", attacking_open)
    return created


def _install_fail_once_snapshot(monkeypatch):
    created = []

    class FailOnceSnapshot:
        def __init__(self, snapshot):
            self._snapshot = snapshot
            self.close_calls = 0

        def __getattr__(self, name):
            return getattr(self._snapshot, name)

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("injected snapshot close failure")
            return self._snapshot.close()

    def fail_once_open(path, **kwargs):
        snapshot = FailOnceSnapshot(
            open_owned_model_tree_snapshot(path, **kwargs)
        )
        created.append(snapshot)
        return snapshot

    monkeypatch.setattr(mos, "open_owned_model_tree_snapshot", fail_once_open)
    return created


def _run_posix_fork_probe(
    probe, *, timeout: float = 10.0, before_start=None, after_start=None,
):
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)

    def child_main():
        receiver.close()
        try:
            payload = {"result": probe(), "error": None}
        except BaseException as exc:
            payload = {
                "result": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        try:
            sender.send(payload)
        finally:
            sender.close()

    process = context.Process(target=child_main)
    if before_start is not None:
        before_start()
    try:
        process.start()
    finally:
        if after_start is not None:
            after_start()
    sender.close()
    try:
        if not receiver.poll(timeout):
            process.terminate()
            process.join(timeout=5)
            pytest.fail(f"fork child did not report within {timeout} seconds")
        try:
            payload = receiver.recv()
        except EOFError:
            payload = {"result": None, "error": "child pipe closed without a result"}
    finally:
        receiver.close()
    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("fork child reported but did not exit")
    assert process.exitcode == 0
    assert payload["error"] is None, payload["error"]
    return payload["result"]


class _FailOnceCloseEstimator:
    def __init__(self, name="estimator"):
        self.name = name
        self.close_calls = 0
        self.closed = False

    def close(self):
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError(f"injected {self.name} close failure")
        self.closed = True

    def compute(self, *, audio_path=None, audio_data=None):
        return {"name": self.name, "audio_path": audio_path, "audio_data": audio_data}


def test_clear_model_cache_retries_pending_close_on_next_clear(monkeypatch):
    estimator = _FailOnceCloseEstimator("cached")
    monkeypatch.setattr(mos, "_MODEL_CACHE", {("fake", False): estimator})
    monkeypatch.setattr(mos, "_PENDING_ESTIMATOR_CLEANUP", {})

    with pytest.raises(RuntimeError, match="injected cached close failure"):
        mos.clear_model_cache()

    assert mos._MODEL_CACHE == {}
    assert list(mos._PENDING_ESTIMATOR_CLEANUP.values()) == [estimator]
    mos.clear_model_cache()
    assert estimator.close_calls == 2
    assert estimator.closed is True
    assert mos._PENDING_ESTIMATOR_CLEANUP == {}


def test_uncached_scope_retains_handle_when_close_fails(monkeypatch):
    estimator = _FailOnceCloseEstimator("uncached")
    monkeypatch.setattr(mos, "_PENDING_ESTIMATOR_CLEANUP", {})
    monkeypatch.setattr(mos, "create_mos_estimator", lambda *_args, **_kwargs: estimator)

    @mos._close_uncached_estimators
    def create_uncached():
        return mos.get_mos_estimator("fake", use_cache=False)

    with pytest.raises(RuntimeError, match="injected uncached close failure"):
        create_uncached()

    assert list(mos._PENDING_ESTIMATOR_CLEANUP.values()) == [estimator]
    mos.clear_model_cache()
    assert estimator.closed is True
    assert mos._PENDING_ESTIMATOR_CLEANUP == {}


def test_dnsmos_worker_cleanup_retries_failed_close(monkeypatch):
    estimator = _FailOnceCloseEstimator("worker")
    monkeypatch.setattr(mos, "_WORKER_DNSMOS", estimator)
    monkeypatch.setattr(mos, "_WORKER_DNSMOS_ASSET_KEY", (os.getpid(), "asset"))
    monkeypatch.setattr(mos, "_PENDING_ESTIMATOR_CLEANUP", {})

    with pytest.raises(RuntimeError, match="injected worker close failure"):
        mos._cleanup_dnsmos_worker_estimator()

    assert mos._WORKER_DNSMOS is None
    assert mos._WORKER_DNSMOS_ASSET_KEY is None
    assert list(mos._PENDING_ESTIMATOR_CLEANUP.values()) == [estimator]
    mos._cleanup_dnsmos_worker_estimator()
    assert estimator.closed is True
    assert mos._PENDING_ESTIMATOR_CLEANUP == {}


def test_dnsmos_worker_replacement_retains_failed_previous_cleanup(monkeypatch):
    created = []

    def make_estimator(*, model_dir, **_kwargs):
        estimator = _FailOnceCloseEstimator(Path(model_dir).name)
        created.append(estimator)
        return estimator

    monkeypatch.setattr(mos, "DNSMOSEstimator", make_estimator)
    monkeypatch.setattr(mos, "_WORKER_DNSMOS", None)
    monkeypatch.setattr(mos, "_WORKER_DNSMOS_ASSET_KEY", None)
    monkeypatch.setattr(mos, "_WORKER_DNSMOS_ATEXIT_REGISTERED", True)
    monkeypatch.setattr(mos, "_PENDING_ESTIMATOR_CLEANUP", {})
    item = (np.zeros(8, dtype=np.float32), 16_000)

    _, first_result = mos._dnsmos_worker_compute(0, item, "/asset/a")
    with pytest.raises(RuntimeError, match="injected a close failure"):
        mos._dnsmos_worker_compute(1, item, "/asset/b")

    assert first_result["name"] == "a"
    assert mos._WORKER_DNSMOS is created[1]
    assert list(mos._PENDING_ESTIMATOR_CLEANUP.values()) == [created[0]]
    _, second_result = mos._dnsmos_worker_compute(2, item, "/asset/b")
    assert second_result["name"] == "b"
    assert created[0].closed is True
    assert mos._PENDING_ESTIMATOR_CLEANUP == {}


def test_cache_identity_replacement_is_bounded_and_failed_eviction_is_pending(
    monkeypatch,
):
    identity = ["a"]
    created = []

    def make_estimator(*_args, **_kwargs):
        estimator = _FailOnceCloseEstimator(identity[0])
        created.append(estimator)
        return estimator

    monkeypatch.setattr(
        mos, "_get_cache_key",
        lambda mos_type, use_gpu, allow_fallback: (
            mos_type, bool(use_gpu), bool(allow_fallback), identity[0],
        ),
    )
    monkeypatch.setattr(mos, "create_mos_estimator", make_estimator)
    monkeypatch.setattr(mos, "_MODEL_CACHE", {})
    monkeypatch.setattr(mos, "_PENDING_ESTIMATOR_CLEANUP", {})

    first = mos.get_mos_estimator("fake", use_cache=True)
    identity[0] = "b"
    with pytest.raises(RuntimeError, match="injected a close failure"):
        mos.get_mos_estimator("fake", use_cache=True)

    assert len(mos._MODEL_CACHE) == 1
    assert list(mos._MODEL_CACHE.values()) == [created[1]]
    assert list(mos._PENDING_ESTIMATOR_CLEANUP.values()) == [first]
    with pytest.raises(RuntimeError, match="injected b close failure"):
        mos.clear_model_cache()
    mos.clear_model_cache()
    assert first.closed is True
    assert created[1].closed is True
    assert mos._PENDING_ESTIMATOR_CLEANUP == {}


def test_urgent_pk_cache_hit_revalidates_commit_and_package_identity(monkeypatch):
    options = {
        "checkpoint": "/assets/urgent.ckpt",
        "source_dir": "/assets/urgent-source",
        "encoder": "wav2vec2",
        "backbone": "cnn",
        "score_range": {"min": 1, "max": 5},
        "expected_sha256": "7" * 64,
        "expected_commit": "8" * 40,
    }
    verified = {
        "checkpoint_sha256": options["expected_sha256"],
        "source_commit": options["expected_commit"],
        "source_package_sha256": "9" * 64,
    }
    calls = []
    created = []

    class Estimator:
        use_gpu = False

        def __init__(self):
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

    def verify(checkpoint, source_dir, expected_sha256, expected_commit):
        calls.append((checkpoint, source_dir, expected_sha256, expected_commit))
        return dict(verified)

    monkeypatch.setattr(mos, "_URGENT_PK_OPTIONS", options)
    monkeypatch.setattr(mos, "verify_urgent_pk_trusted_assets", verify)
    monkeypatch.setattr(
        mos, "urgent_pk_verified_assets", lambda *_args: dict(verified),
    )
    monkeypatch.setattr(mos, "create_mos_estimator", lambda *_args, **_kwargs: Estimator())
    monkeypatch.setattr(mos, "_MODEL_CACHE", {})

    first = mos.get_mos_estimator("urgent_pk", use_cache=True)
    second = mos.get_mos_estimator("urgent_pk", use_cache=True)

    assert second is first
    verified["source_package_sha256"] = "a" * 64
    replacement = mos.get_mos_estimator("urgent_pk", use_cache=True)

    assert replacement is not first
    assert len(created) == 2
    assert len(calls) == 3
    assert all(call[3] == options["expected_commit"] for call in calls)


def test_cache_identity_replacement_retires_until_canonical_compute_releases_lease(
    monkeypatch,
):
    identity = ["a"]
    compute_started = threading.Event()
    release_compute = threading.Event()
    old_closed = threading.Event()
    created = []

    class Estimator:
        use_gpu = False

        def __init__(self, name):
            self.name = name
            self.close_calls = 0

        def compute(self, *, audio_data):
            if self.name == "a":
                compute_started.set()
                assert release_compute.wait(timeout=5)
                assert not old_closed.is_set()
            return {"value": self.name}

        def close(self):
            self.close_calls += 1
            if self.name == "a":
                old_closed.set()

    def make_estimator(*_args, **_kwargs):
        estimator = Estimator(identity[0])
        created.append(estimator)
        return estimator

    monkeypatch.setattr(
        mos, "_get_cache_key",
        lambda mos_type, use_gpu, allow_fallback: (
            mos_type, bool(use_gpu), bool(allow_fallback), identity[0],
        ),
    )
    monkeypatch.setattr(mos, "create_mos_estimator", make_estimator)
    monkeypatch.setattr(mos, "_MODEL_CACHE", {})
    monkeypatch.setattr(mos, "_RETIRED_ESTIMATORS", {})
    monkeypatch.setattr(mos, "_PENDING_ESTIMATOR_CLEANUP", {})

    with ThreadPoolExecutor(max_workers=1) as executor:
        compute_future = executor.submit(
            mos.compute_mos_scores,
            mos_types=["fake"],
            audio_data=(np.zeros(8, dtype=np.float32), 16_000),
        )
        assert compute_started.wait(timeout=5)
        first = created[0]
        identity[0] = "b"
        replacement = mos.get_mos_estimator("fake", use_cache=True)
        assert replacement is created[1]
        assert first.close_calls == 0
        assert not old_closed.is_set()
        assert list(mos._RETIRED_ESTIMATORS.values()) == [first]
        release_compute.set()
        result = compute_future.result(timeout=5)

    assert result["fake"] == {"value": "a"}
    assert old_closed.is_set()
    assert first.close_calls == 1
    assert mos._RETIRED_ESTIMATORS == {}
    mos.clear_model_cache()
    assert replacement.close_calls == 1


def test_mos_atexit_retries_transient_pending_cleanup(monkeypatch):
    estimator = _FailOnceCloseEstimator("atexit")
    monkeypatch.setattr(mos, "_MODEL_CACHE", {("fake", False): estimator})
    monkeypatch.setattr(mos, "_PENDING_ESTIMATOR_CLEANUP", {})

    mos._cleanup_mos_at_exit()

    assert estimator.close_calls == 2
    assert estimator.closed is True
    assert mos._PENDING_ESTIMATOR_CLEANUP == {}


@pytest.mark.skipif(
    os.name != "posix" or "fork" not in multiprocessing.get_all_start_methods(),
    reason="POSIX fork is unavailable",
)
def test_fork_child_discards_inherited_pending_cleanup_without_close(monkeypatch):
    estimator = _FailOnceCloseEstimator("parent-pending")
    monkeypatch.setattr(mos, "_MODEL_CACHE", {})
    monkeypatch.setattr(mos, "_PENDING_ESTIMATOR_CLEANUP", {id(estimator): estimator})

    def child_probe():
        mos.clear_model_cache()
        return {
            "close_calls": estimator.close_calls,
            "pending_count": len(mos._PENDING_ESTIMATOR_CLEANUP),
        }

    assert _run_posix_fork_probe(child_probe) == {
        "close_calls": 0,
        "pending_count": 0,
    }
    assert estimator.close_calls == 0
    assert list(mos._PENDING_ESTIMATOR_CLEANUP.values()) == [estimator]
    with pytest.raises(RuntimeError, match="injected parent-pending close failure"):
        mos.clear_model_cache()
    mos.clear_model_cache()
    assert estimator.closed is True


def test_mos_at_fork_hook_is_registered_once_across_importlib_reload():
    script = r'''
import importlib
import json
import os

from audio_filter_pipeline.processors import mos

del mos._MOS_AT_FORK_DISPATCH
registrations = []
original = os.register_at_fork

def record_registration(**kwargs):
    registrations.append(kwargs)

os.register_at_fork = record_registration
try:
    importlib.reload(mos)
    importlib.reload(mos)
finally:
    os.register_at_fork = original
print(json.dumps({"registrations": len(registrations)}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.stderr == ""
    assert completed.stdout.strip() == '{"registrations": 1}'


def test_scoreq_close_retries_snapshot_cleanup_after_transient_failure(
    monkeypatch, tmp_path,
):
    origin = tmp_path / "scoreq"
    digest = _write_scoreq_asset(origin)
    loaded = []
    snapshots = _install_fail_once_snapshot(monkeypatch)
    _install_fake_scoreq_loader(monkeypatch, loaded)
    estimator = mos.SCOREQEstimator(
        model_dir=str(origin),
        verified_model_tree_sha256=digest,
        allow_fallback=False,
    )
    snapshot = snapshots[0]
    snapshot_path = Path(snapshot.path)

    try:
        with pytest.raises(RuntimeError, match="injected snapshot close failure"):
            estimator.close()
        assert snapshot.close_calls == 1
        assert snapshot_path.exists()
        assert estimator._snapshot is snapshot
        assert estimator._closed is False
        with pytest.raises(RuntimeError, match="only close retry is allowed"):
            estimator.compute(
                audio_data=(np.zeros(160, dtype=np.float32), 16_000),
            )
        with pytest.raises(RuntimeError, match="only close retry is allowed"):
            estimator.reset()

        estimator.close()
        assert snapshot.close_calls == 2
        assert not snapshot_path.exists()
        assert estimator._snapshot is None
        assert estimator._closed is True
    finally:
        snapshot._snapshot.close()


def test_dnsmos_close_retries_only_snapshot_without_repeating_other_cleanup(
    monkeypatch, tmp_path,
):
    origin = tmp_path / "dnsmos"
    digest = _write_dnsmos_asset(origin, "close-retry")
    snapshots = _install_fail_once_snapshot(monkeypatch)
    monkeypatch.setattr(mos, "_apply_dnsmos_vram_fix", lambda _module: None)
    module_cleanup_calls = []
    original_remove_module = mos._remove_private_dnsmos_module

    def track_remove_module(name, module):
        module_cleanup_calls.append((name, module))
        return original_remove_module(name, module)

    monkeypatch.setattr(mos, "_remove_private_dnsmos_module", track_remove_module)
    estimator = mos.DNSMOSEstimator(
        model_dir=str(origin),
        verified_model_tree_sha256=digest,
        allow_fallback=False,
    )
    snapshot = snapshots[0]
    snapshot_path = Path(snapshot.path)
    module_name = estimator._dnsmos_module_name

    class FakePool:
        def __init__(self):
            self.shutdown_calls = 0

        def shutdown(self, *, wait):
            assert wait is True
            assert estimator._pool_lock.acquire(blocking=False)
            estimator._pool_lock.release()
            self.shutdown_calls += 1

    pool = FakePool()
    estimator._process_pool = pool

    try:
        with pytest.raises(RuntimeError, match="injected snapshot close failure"):
            estimator.close()
        assert pool.shutdown_calls == 1
        assert len(module_cleanup_calls) == 1
        assert module_name not in sys.modules
        assert snapshot.close_calls == 1
        assert snapshot_path.exists()
        assert estimator._snapshot is snapshot
        assert estimator._closed is False

        estimator.close()
        assert pool.shutdown_calls == 1
        assert len(module_cleanup_calls) == 1
        assert snapshot.close_calls == 2
        assert not snapshot_path.exists()
        assert estimator._snapshot is None
        assert estimator._closed is True
    finally:
        original_remove_module(module_name, estimator._dnsmos_module)
        snapshot._snapshot.close()


def test_scoreq_resets_use_manifest_without_repeating_full_digest(
    monkeypatch, tmp_path,
):
    origin = tmp_path / "scoreq"
    digest = _write_scoreq_asset(origin)
    loaded = []
    wrappers = []
    _install_fake_scoreq_loader(monkeypatch, loaded)

    class CountingSnapshot:
        def __init__(self, snapshot):
            self._snapshot = snapshot
            self.full_verify_calls = 0
            self.manifest_verify_calls = 0
            self.lease_depth = 0

        def __getattr__(self, name):
            return getattr(self._snapshot, name)

        @contextlib.contextmanager
        def loader_path_lease(self):
            with self._snapshot.loader_path_lease() as path:
                self.lease_depth += 1
                try:
                    yield path
                finally:
                    self.lease_depth -= 1

        def verify(self):
            assert self.lease_depth == 1
            self.full_verify_calls += 1
            return self._snapshot.verify()

        def verify_manifest(self):
            assert self.lease_depth == 1
            self.manifest_verify_calls += 1
            return self._snapshot.verify_manifest()

        def close(self):
            return self._snapshot.close()

    def counting_open(path, **kwargs):
        wrapper = CountingSnapshot(
            open_owned_model_tree_snapshot(path, **kwargs)
        )
        wrappers.append(wrapper)
        return wrapper

    monkeypatch.setattr(mos, "open_owned_model_tree_snapshot", counting_open)
    estimator = mos.SCOREQEstimator(
        model_dir=str(origin),
        verified_model_tree_sha256=digest,
        allow_fallback=False,
    )
    snapshot = wrappers[0]
    try:
        assert snapshot.full_verify_calls == 1
        assert snapshot.manifest_verify_calls == 1
        estimator.reset()
        estimator.reset()
        assert snapshot.full_verify_calls == 1
        assert snapshot.manifest_verify_calls == 5
        assert len(loaded) == 3
    finally:
        estimator.close()


@pytest.mark.parametrize("origin_change", ("replace", "remove"))
def test_scoreq_loader_hardening_and_reset_stay_on_owned_snapshot(
    monkeypatch, tmp_path, origin_change,
):
    origin = tmp_path / "scoreq"
    digest = _write_scoreq_asset(origin)
    loaded = []
    hardened = []
    _install_fake_scoreq_loader(monkeypatch, loaded)

    def harden(scorer, model_dir):
        hardened.append((model_dir, scorer.model_bytes))
        assert (Path(model_dir) / "adapt_nr_telephone.onnx").read_bytes() == (
            b"scoreq-original"
        )

    monkeypatch.setattr(mos, "_harden_scoreq_gpu_session", harden)
    estimator = mos.SCOREQEstimator(
        use_gpu=True,
        model_dir=str(origin),
        verified_model_tree_sha256=digest,
        allow_fallback=False,
    )
    snapshot = estimator._snapshot
    stable_path = loaded[0][0]
    public_path = Path(snapshot.path)
    held = tmp_path / "scoreq-held"
    origin.rename(held)
    if origin_change == "replace":
        _write_scoreq_asset(origin, b"scoreq-replacement")

    estimator.reset()

    assert re.fullmatch(r"/proc/self/fd/\d+", stable_path)
    assert [entry[1] for entry in loaded] == [b"scoreq-original"] * 2
    assert [entry[0] for entry in loaded] == [stable_path] * 2
    assert hardened == [(stable_path, b"scoreq-original")] * 2
    assert Path(stable_path).resolve() == public_path
    assert public_path != origin
    assert public_path.exists()
    estimator.close()
    assert not public_path.exists()
    assert not Path(stable_path).exists()
    with pytest.raises(OSError):
        os.open(stable_path, os.O_RDONLY | os.O_DIRECTORY)


def test_scoreq_public_snapshot_path_swap_never_passes_replacement_to_loader_or_hardener(
    monkeypatch, tmp_path,
):
    origin = tmp_path / "scoreq"
    digest = _write_scoreq_asset(origin)
    loaded = []
    hardened = []

    def install_replacement(path):
        path.mkdir()
        (path / "adapt_nr_telephone.onnx").write_bytes(b"malicious-replacement")

    snapshots = _swap_public_snapshot_path_after_verify(
        monkeypatch, install_replacement,
    )
    _install_fake_scoreq_loader(monkeypatch, loaded)

    def harden(scorer, model_dir):
        hardened.append((model_dir, scorer.model_bytes))
        assert (Path(model_dir) / "adapt_nr_telephone.onnx").read_bytes() == (
            b"scoreq-original"
        )

    monkeypatch.setattr(mos, "_harden_scoreq_gpu_session", harden)
    estimator = None
    try:
        try:
            estimator = mos.SCOREQEstimator(
                use_gpu=True,
                model_dir=str(origin),
                verified_model_tree_sha256=digest,
                allow_fallback=False,
            )
        except (mos.MosModelAssetBindingError, RuntimeError):
            pass

        snapshot = snapshots[0]
        assert snapshot.attacked is True
        assert loaded and [entry[1] for entry in loaded] == [b"scoreq-original"]
        stable_path = loaded[0][0]
        assert re.fullmatch(r"/proc/self/fd/\d+", stable_path)
        assert Path(stable_path).resolve() == snapshot._held_path
        assert hardened == [(stable_path, b"scoreq-original")]
        assert (Path(snapshot.path) / "adapt_nr_telephone.onnx").read_bytes() == (
            b"malicious-replacement"
        )
    finally:
        if estimator is not None:
            try:
                estimator.close()
            except (mos.MosModelAssetBindingError, RuntimeError):
                pass
        if snapshots:
            snapshots[0].restore_and_close()


def test_dnsmos_public_snapshot_path_swap_never_compiles_replacement(
    monkeypatch, tmp_path,
):
    import builtins

    origin = tmp_path / "dnsmos"
    digest = _write_dnsmos_asset(origin, "owned")
    compiled = []

    def install_replacement(path):
        _write_dnsmos_asset(path, "malicious-replacement")

    snapshots = _swap_public_snapshot_path_after_verify(
        monkeypatch, install_replacement,
    )
    original_compile = builtins.compile

    def track_compile(source, filename, *args, **kwargs):
        if str(filename).endswith("dnsmos_local.py"):
            compiled.append((source, str(filename)))
        return original_compile(source, filename, *args, **kwargs)

    monkeypatch.setattr(builtins, "compile", track_compile)
    monkeypatch.setattr(mos, "_apply_dnsmos_vram_fix", lambda _module: None)
    estimator = None
    try:
        try:
            estimator = mos.DNSMOSEstimator(
                model_dir=str(origin),
                verified_model_tree_sha256=digest,
                allow_fallback=False,
            )
        except (mos.MosModelAssetBindingError, RuntimeError):
            pass

        snapshot = snapshots[0]
        assert snapshot.attacked is True
        assert len(compiled) == 1
        source, module_path = compiled[0]
        if isinstance(source, str):
            source = source.encode("utf-8")
        assert b"ASSET_ID = 'owned'" in source
        assert b"malicious-replacement" not in source
        stable_root = os.path.dirname(module_path)
        assert re.fullmatch(r"/proc/self/fd/\d+", stable_root)
        assert Path(stable_root).resolve() == snapshot._held_path
        assert b"malicious-replacement" in (
            Path(snapshot.path) / "dnsmos_local.py"
        ).read_bytes()
    finally:
        if estimator is not None:
            try:
                estimator.close()
            except (mos.MosModelAssetBindingError, RuntimeError):
                pass
        if snapshots:
            snapshots[0].restore_and_close()


def test_dnsmos_ignores_foreign_bare_module_and_isolates_two_assets(
    monkeypatch, tmp_path,
):
    first_dir = tmp_path / "dnsmos-a"
    second_dir = tmp_path / "dnsmos-b"
    first_digest = _write_dnsmos_asset(first_dir, "asset-a", score=3.1)
    second_digest = _write_dnsmos_asset(second_dir, "asset-b", score=4.2)
    foreign = types.ModuleType("dnsmos_local")
    foreign.ComputeScore = lambda *_args, **_kwargs: pytest.fail(
        "foreign bare dnsmos_local must not be reused"
    )
    monkeypatch.setitem(sys.modules, "dnsmos_local", foreign)
    monkeypatch.setattr(mos, "_apply_dnsmos_vram_fix", lambda _module: None)

    first = mos.DNSMOSEstimator(
        model_dir=str(first_dir),
        verified_model_tree_sha256=first_digest,
        allow_fallback=False,
    )
    second = mos.DNSMOSEstimator(
        model_dir=str(second_dir),
        verified_model_tree_sha256=second_digest,
        allow_fallback=False,
    )
    first_module_name = type(first.compute_score).__module__
    second_module_name = type(second.compute_score).__module__

    assert sys.modules["dnsmos_local"] is foreign
    assert first_module_name != "dnsmos_local"
    assert second_module_name != "dnsmos_local"
    assert first_module_name != second_module_name
    assert sys.modules[first_module_name] is not sys.modules[second_module_name]
    assert first.compute_score.asset_id == "asset-a"
    assert second.compute_score.asset_id == "asset-b"
    assert first.compute_score.primary_bytes == b"primary-asset-a"
    assert second.compute_score.primary_bytes == b"primary-asset-b"
    first.close()
    second.close()
    assert first_module_name not in sys.modules
    assert second_module_name not in sys.modules


def test_dnsmos_module_exec_failure_rolls_back_module_and_snapshot(
    monkeypatch, tmp_path,
):
    controlled_tmp = tmp_path / "controlled-tmp"
    controlled_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(controlled_tmp))
    monkeypatch.setattr(tempfile, "tempdir", None)
    origin = tmp_path / "dnsmos"
    digest = _write_dnsmos_asset(origin, "broken", fail_exec=True)
    before = set(sys.modules)

    with pytest.raises(RuntimeError, match="module exec exploded"):
        mos.DNSMOSEstimator(
            model_dir=str(origin),
            verified_model_tree_sha256=digest,
            allow_fallback=False,
        )

    leaked = {
        name for name in set(sys.modules) - before
        if "dnsmos" in name.lower()
    }
    assert leaked == set()
    assert list(controlled_tmp.iterdir()) == []


def test_dnsmos_reset_and_cpu_downgrade_reuse_private_snapshot_module(
    monkeypatch, tmp_path,
):
    origin = tmp_path / "dnsmos"
    digest = _write_dnsmos_asset(origin, "stable")
    foreign = types.ModuleType("dnsmos_local")
    monkeypatch.setitem(sys.modules, "dnsmos_local", foreign)
    monkeypatch.setattr(mos, "_apply_dnsmos_vram_fix", lambda _module: None)
    monkeypatch.setattr(mos, "cpu_downgrade_allowed", lambda: True)
    estimator = mos.DNSMOSEstimator(
        use_gpu=True,
        model_dir=str(origin),
        verified_model_tree_sha256=digest,
        allow_fallback=False,
    )
    first = estimator.compute_score
    private_name = type(first).__module__
    private_module = sys.modules[private_name]
    stable_root = os.path.dirname(first.primary_path)
    stable_paths = (first.primary_path, first.p808_path, private_module.__file__)
    held = tmp_path / "dnsmos-held"
    origin.rename(held)

    estimator.reset()
    second = estimator.compute_score
    estimator._downgrade_to_cpu("test")
    third = estimator.compute_score

    assert re.fullmatch(r"/proc/self/fd/\d+", stable_root)
    assert os.path.dirname(private_module.__file__) == stable_root
    assert sys.modules["dnsmos_local"] is foreign
    assert sys.modules[private_name] is private_module
    assert type(first) is type(second) is type(third)
    assert first.module_token is second.module_token is third.module_token
    assert [first.use_gpu, second.use_gpu, third.use_gpu] == [True, True, False]
    assert [first.primary_path, second.primary_path, third.primary_path] == (
        [stable_paths[0]] * 3
    )
    assert [first.p808_path, second.p808_path, third.p808_path] == (
        [stable_paths[1]] * 3
    )
    assert [first.primary_bytes, second.primary_bytes, third.primary_bytes] == (
        [b"primary-stable"] * 3
    )
    estimator.close()
    assert private_name not in sys.modules
    assert all(not Path(path).exists() for path in stable_paths)
    for path in stable_paths:
        with pytest.raises(OSError):
            os.open(path, os.O_RDONLY)


def test_dnsmos_cpu_downgrade_failure_preserves_gpu_session_without_fallback(
    monkeypatch,
):
    estimator = mos.DNSMOSEstimator.__new__(mos.DNSMOSEstimator)

    class FailingGpuSession:
        def run_batch_audio(self, *_args, **_kwargs):
            raise RuntimeError("injected GPU inference failure")

    gpu_session = FailingGpuSession()
    estimator.compute_score = gpu_session
    estimator.use_gpu = True
    estimator._is_fallback = False
    estimator._allow_fallback = True
    monkeypatch.setattr(mos, "cpu_downgrade_allowed", lambda: True)

    def fail_cpu_replacement(*, use_gpu):
        assert use_gpu is False
        raise RuntimeError("injected CPU replacement validation failure")

    monkeypatch.setattr(estimator, "_new_compute_score", fail_cpu_replacement)

    with pytest.raises(
        RuntimeError,
        match="DNSMOS CPU downgrade failed; GPU session preserved",
    ) as error:
        estimator.compute(audio_data=(np.zeros(160, dtype=np.float32), 16_000))

    assert "injected CPU replacement validation failure" in str(error.value.__cause__)
    assert estimator.compute_score is gpu_session
    assert estimator.use_gpu is True
    assert estimator._is_fallback is False


@pytest.mark.parametrize("model", ("nisqa", "utmos"))
def test_local_checkpoint_cache_hit_skips_asset_scan_and_survives_deleted_path(
    monkeypatch, tmp_path, model,
):
    model_path = (tmp_path / f"{model}.pt").resolve()
    model_path.write_bytes(b"checkpoint")
    options = {
        "model_path": str(model_path),
        "model_version": "v1",
        "model_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "checkpoint_size_bytes": len(b"checkpoint"),
    }
    verify_calls = []
    snapshot_calls = []
    construct_calls = []
    created = []

    class Estimator:
        use_gpu = False

        def __init__(self, identity):
            self.identity = identity
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    def verify(asset):
        verify_calls.append(asset)

    @mos.contextlib.contextmanager
    def checkpoint_snapshot(path):
        snapshot_calls.append(path)
        yield types.SimpleNamespace(path=path)

    def create(mos_type, **kwargs):
        construct_calls.append(kwargs["model_path"])
        with mos._open_local_checkpoint_snapshot(kwargs["model_path"]):
            pass
        estimator = Estimator((mos_type, kwargs["checkpoint_sha256"]))
        created.append(estimator)
        return estimator

    binder = mos.bind_nisqa_options if model == "nisqa" else mos.bind_utmos_options
    monkeypatch.setattr(mos, "verify_local_model_asset", verify)
    monkeypatch.setattr(mos, "_open_local_checkpoint_snapshot", checkpoint_snapshot)
    monkeypatch.setattr(mos, "create_mos_estimator", create)
    monkeypatch.setattr(mos, "_MODEL_CACHE", {})

    with binder(options):
        first = mos.get_mos_estimator(
            model, use_gpu=False, use_cache=True, allow_fallback=False,
        )
        model_path.unlink()
        second = mos.get_mos_estimator(
            model, use_gpu=False, use_cache=True, allow_fallback=False,
        )

    assert first is second
    assert len(verify_calls) == 1
    assert snapshot_calls == [str(model_path)]
    assert construct_calls == [str(model_path)]

    model_path.write_bytes(b"replacement")
    changed = dict(options, model_version="v2", model_sha256="3" * 64)
    with binder(changed):
        replacement = mos.get_mos_estimator(
            model, use_gpu=False, use_cache=True, allow_fallback=False,
        )

    assert replacement is created[1]
    assert replacement is not first
    assert len(verify_calls) == 2
    assert snapshot_calls == [str(model_path), str(model_path)]
    assert construct_calls == [str(model_path), str(model_path)]
    assert first.close_calls == 1
    assert len(mos._MODEL_CACHE) == 1


@pytest.mark.parametrize("model", ("nisqa", "utmos"))
def test_local_checkpoint_concurrent_cold_miss_verifies_and_constructs_once(
    monkeypatch, tmp_path, model,
):
    model_path = (tmp_path / f"{model}.pt").resolve()
    model_path.write_bytes(b"checkpoint")
    options = {
        "model_path": str(model_path),
        "model_version": "v1",
        "model_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "checkpoint_size_bytes": len(b"checkpoint"),
    }
    verify_calls = []
    snapshot_calls = []
    construct_calls = []
    start = threading.Barrier(8)
    estimator = types.SimpleNamespace(use_gpu=False)

    def verify(asset):
        verify_calls.append(asset)
        time.sleep(0.02)

    @mos.contextlib.contextmanager
    def checkpoint_snapshot(path):
        snapshot_calls.append(path)
        yield types.SimpleNamespace(path=path)

    def create(_mos_type, **kwargs):
        construct_calls.append(kwargs["model_path"])
        with mos._open_local_checkpoint_snapshot(kwargs["model_path"]):
            pass
        time.sleep(0.02)
        return estimator

    binder = mos.bind_nisqa_options if model == "nisqa" else mos.bind_utmos_options
    monkeypatch.setattr(mos, "verify_local_model_asset", verify)
    monkeypatch.setattr(mos, "_open_local_checkpoint_snapshot", checkpoint_snapshot)
    monkeypatch.setattr(mos, "create_mos_estimator", create)
    monkeypatch.setattr(mos, "_MODEL_CACHE", {})

    def load():
        start.wait(timeout=5)
        return mos.get_mos_estimator(
            model, use_gpu=False, use_cache=True, allow_fallback=False,
        )

    with binder(options), ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(load) for _ in range(8)]
        results = [future.result(timeout=5) for future in futures]

    assert results == [estimator] * 8
    assert len(verify_calls) == 1
    assert snapshot_calls == [str(model_path)]
    assert construct_calls == [str(model_path)]


@pytest.mark.parametrize("model", ("scoreq", "dnsmos"))
def test_directory_model_cache_hit_does_not_rebuild_owned_snapshot(
    monkeypatch, tmp_path, model,
):
    origin = tmp_path / model
    digest = (
        _write_scoreq_asset(origin)
        if model == "scoreq"
        else _write_dnsmos_asset(origin, "cached")
    )
    loaded = []
    _install_fake_scoreq_loader(monkeypatch, loaded)
    monkeypatch.setattr(mos, "_apply_dnsmos_vram_fix", lambda _module: None)
    snapshots = _track_owned_snapshots(monkeypatch)
    monkeypatch.setattr(mos, "_MODEL_CACHE", {})
    options = {
        "model_dir": str(origin),
        "verified_model_tree_sha256": digest,
    }
    binder = mos.bind_scoreq_options if model == "scoreq" else mos.bind_dnsmos_options

    with binder(options):
        first = mos.get_mos_estimator(
            model, use_gpu=False, use_cache=True, allow_fallback=False,
        )
        second = mos.get_mos_estimator(
            model, use_gpu=False, use_cache=True, allow_fallback=False,
        )

    assert first is second
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    snapshot_path = Path(snapshot.path)
    stable_path = mos._owned_model_tree_root_descriptor_path(model.upper(), snapshot)
    assert re.fullmatch(r"/proc/self/fd/\d+", stable_path)
    if model == "scoreq":
        assert [entry[0] for entry in loaded] == [stable_path]
    else:
        assert os.path.dirname(first.compute_score.primary_path) == stable_path
        assert os.path.dirname(first.compute_score.p808_path) == stable_path
    assert snapshot_path.exists()
    assert Path(stable_path).resolve() == snapshot_path
    mos.clear_model_cache()
    assert not snapshot_path.exists()
    assert not Path(stable_path).exists()


@pytest.mark.skipif(
    os.name != "posix" or "fork" not in multiprocessing.get_all_start_methods(),
    reason="POSIX fork is unavailable",
)
@pytest.mark.parametrize("model", ("scoreq", "dnsmos"))
def test_directory_model_cache_rebuilds_pid_owned_estimator_after_fork(
    monkeypatch, tmp_path, model,
):
    origin = tmp_path / model
    digest = (
        _write_scoreq_asset(origin)
        if model == "scoreq"
        else _write_dnsmos_asset(origin, "fork-cache")
    )
    loaded = []
    _install_fake_scoreq_loader(monkeypatch, loaded)
    monkeypatch.setattr(mos, "_apply_dnsmos_vram_fix", lambda _module: None)
    monkeypatch.setattr(mos, "_MODEL_CACHE", {})
    options = {
        "model_dir": str(origin),
        "verified_model_tree_sha256": digest,
    }
    binder = mos.bind_scoreq_options if model == "scoreq" else mos.bind_dnsmos_options

    with binder(options):
        parent_estimator = mos.get_mos_estimator(
            model, use_gpu=False, use_cache=True, allow_fallback=False,
        )
        parent_snapshot = parent_estimator._snapshot
        parent_snapshot_path = Path(parent_snapshot.path)
        parent_cache_lock = mos._MODEL_CACHE_LOCK

        def child_probe():
            child_estimator = mos.get_mos_estimator(
                model, use_gpu=False, use_cache=True, allow_fallback=False,
            )
            child_snapshot = child_estimator._snapshot
            child_snapshot_path = Path(child_snapshot.path)
            before_reset = child_snapshot.verify()
            child_estimator.reset()
            after_reset = child_snapshot.verify()
            child_estimator.close()
            child_removed = not child_snapshot_path.exists()
            parent_still_exists = parent_snapshot_path.exists()
            mos.clear_model_cache()
            return {
                "new_estimator": child_estimator is not parent_estimator,
                "new_snapshot": child_snapshot is not parent_snapshot,
                "different_snapshot_path": (
                    child_snapshot_path != parent_snapshot_path
                ),
                "cache_lock_replaced": mos._MODEL_CACHE_LOCK is not parent_cache_lock,
                "parent_estimator_untouched": (
                    parent_estimator._closed is False
                    and parent_estimator._snapshot is parent_snapshot
                ),
                "before_reset": before_reset,
                "after_reset": after_reset,
                "child_removed": child_removed,
                "parent_still_exists": parent_still_exists,
            }

        try:
            child = _run_posix_fork_probe(
                child_probe,
                before_start=parent_cache_lock.acquire,
                after_start=parent_cache_lock.release,
            )
            assert child == {
                "new_estimator": True,
                "new_snapshot": True,
                "different_snapshot_path": True,
                "cache_lock_replaced": True,
                "parent_estimator_untouched": True,
                "before_reset": digest,
                "after_reset": digest,
                "child_removed": True,
                "parent_still_exists": True,
            }
            assert parent_snapshot_path.exists()
            assert parent_snapshot.verify() == digest
        finally:
            mos.clear_model_cache()

    assert not parent_snapshot_path.exists()


@pytest.mark.parametrize("inference_error", (False, True), ids=("success", "error"))
def test_scoreq_use_cache_false_cleans_snapshot_after_call(
    monkeypatch, tmp_path, inference_error,
):
    controlled_tmp = tmp_path / "controlled-tmp"
    controlled_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(controlled_tmp))
    monkeypatch.setattr(tempfile, "tempdir", None)
    origin = tmp_path / "scoreq"
    digest = _write_scoreq_asset(origin)
    loaded = []
    snapshots = _track_owned_snapshots(monkeypatch)
    _install_fake_scoreq_loader(
        monkeypatch, loaded, inference_error=inference_error,
    )

    result = mos.compute_mos_scores(
        mos_types=["scoreq"],
        use_cache=False,
        allow_fallback=False,
        scoreq_options={
            "model_dir": str(origin),
            "verified_model_tree_sha256": digest,
        },
        audio_data=(np.zeros(160, dtype=np.float32), 16_000),
    )["scoreq"]

    if inference_error:
        assert "inference exploded" in result["error"]
    else:
        assert result == {"scoreq_mos": 4.0}
    assert len(snapshots) == 1
    assert not Path(snapshots[0].path).exists()
    assert list(controlled_tmp.iterdir()) == []


@pytest.mark.parametrize("inference_error", (False, True), ids=("success", "error"))
def test_dnsmos_use_cache_false_cleans_snapshot_and_private_module(
    monkeypatch, tmp_path, inference_error,
):
    controlled_tmp = tmp_path / "controlled-tmp"
    controlled_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(controlled_tmp))
    monkeypatch.setattr(tempfile, "tempdir", None)
    origin = tmp_path / "dnsmos"
    digest = _write_dnsmos_asset(
        origin, "ephemeral", fail_inference=inference_error,
    )
    snapshots = _track_owned_snapshots(monkeypatch)
    monkeypatch.setattr(mos, "_apply_dnsmos_vram_fix", lambda _module: None)
    before = set(sys.modules)

    result = mos.compute_mos_scores(
        mos_types=["dnsmos"],
        use_cache=False,
        allow_fallback=False,
        dnsmos_options={
            "model_dir": str(origin),
            "verified_model_tree_sha256": digest,
        },
        audio_data=(np.zeros(160, dtype=np.float32), 16_000),
    )["dnsmos"]

    if inference_error:
        assert "inference exploded" in result["error"]
    else:
        assert result["dnsmos_ovr_mos"] == pytest.approx(4.0)
    assert len(snapshots) == 1
    assert not Path(snapshots[0].path).exists()
    assert not {
        name for name in set(sys.modules) - before
        if name.startswith("_audio_filter_dnsmos_")
    }
    assert list(controlled_tmp.iterdir()) == []


@pytest.mark.parametrize("model", ("scoreq", "dnsmos"))
def test_directory_model_replacement_after_preflight_fails_before_real_loader(
    monkeypatch, tmp_path, model,
):
    model_dir = tmp_path / model
    expected = (
        _write_scoreq_asset(model_dir)
        if model == "scoreq"
        else _write_dnsmos_asset(model_dir, "before")
    )
    loaded = []
    asset = MosAssetSpec(model, {"model_dir": str(model_dir)})
    verified = verify_mos_assets(
        MosExecutionPlan(
            (model,), (asset,), {"kind": "single_process", "device": "cpu"},
        ),
        tree_digest=lambda _path: expected,
    )
    options = verified.as_mapping()["options_by_model"][model]
    if model == "scoreq":
        (model_dir / "adapt_nr_telephone.onnx").write_bytes(b"replacement")
    else:
        (model_dir / "sig_bak_ovr.onnx").write_bytes(b"replacement")
    _install_fake_scoreq_loader(monkeypatch, loaded)
    monkeypatch.setattr(mos, "_apply_dnsmos_vram_fix", lambda _module: None)
    monkeypatch.setattr(mos, "_MODEL_CACHE", {})
    binder = mos.bind_scoreq_options if model == "scoreq" else mos.bind_dnsmos_options

    with binder(options), pytest.raises(
        mos.MosModelAssetBindingError, match="digest mismatch",
    ):
        mos.get_mos_estimator(
            model, use_gpu=False, use_cache=False, allow_fallback=False,
        )

    assert loaded == []


def test_directory_model_snapshot_is_created_only_for_selected_model(
    monkeypatch, tmp_path,
):
    scoreq_dir = tmp_path / "scoreq"
    dnsmos_dir = tmp_path / "dnsmos"
    scoreq_digest = _write_scoreq_asset(scoreq_dir)
    dnsmos_digest = _write_dnsmos_asset(dnsmos_dir, "unselected")
    loaded = []
    snapshots = _track_owned_snapshots(monkeypatch)
    _install_fake_scoreq_loader(monkeypatch, loaded)

    with mos.bind_scoreq_options({
        "model_dir": str(scoreq_dir),
        "verified_model_tree_sha256": scoreq_digest,
    }), mos.bind_dnsmos_options({
        "model_dir": str(dnsmos_dir),
        "verified_model_tree_sha256": dnsmos_digest,
    }):
        estimator = mos.get_mos_estimator(
            "scoreq", use_gpu=False, use_cache=False, allow_fallback=False,
        )

    assert [snapshot.source_path for snapshot in snapshots] == [
        str(scoreq_dir.resolve())
    ]
    estimator.close()


def _spawn_dnsmos_asset_isolation_probe(asset_specs):
    from audio_filter_pipeline.processors import mos as child_mos

    observed = []
    previous_module_name = None
    try:
        for index, (model_dir, digest) in enumerate(asset_specs):
            _, result = child_mos._dnsmos_worker_compute(
                index,
                (np.zeros(160, dtype=np.float32), 16_000),
                model_dir,
                digest,
                False,
            )
            estimator = child_mos._WORKER_DNSMOS
            module_name = type(estimator.compute_score).__module__
            observed.append({
                "asset_id": estimator.compute_score.asset_id,
                "module_name": module_name,
                "module_is_private": module_name != "dnsmos_local",
                "previous_module_removed": (
                    previous_module_name is None
                    or previous_module_name not in sys.modules
                ),
                "primary_bytes": estimator.compute_score.primary_bytes,
                "score": result["dnsmos_ovr_mos"],
            })
            previous_module_name = module_name
        return os.getpid(), observed
    finally:
        estimator = child_mos._WORKER_DNSMOS
        if estimator is not None:
            estimator.close()
        child_mos._WORKER_DNSMOS = None
        child_mos._WORKER_DNSMOS_ASSET_KEY = None


def test_dnsmos_process_pool_uses_spawn_context(monkeypatch, tmp_path):
    origin = tmp_path / "dnsmos"
    digest = _write_dnsmos_asset(origin, "spawn")
    monkeypatch.setattr(mos, "_apply_dnsmos_vram_fix", lambda _module: None)
    estimator = mos.DNSMOSEstimator(
        model_dir=str(origin),
        verified_model_tree_sha256=digest,
        allow_fallback=False,
    )

    pool = estimator._get_process_pool(1)
    try:
        assert pool._mp_context.get_start_method() == "spawn"
    finally:
        estimator.close()


def test_dnsmos_spawn_child_rebinds_private_module_per_asset_without_residue(
    monkeypatch, tmp_path,
):
    controlled_tmp = tmp_path / "controlled-tmp"
    controlled_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(controlled_tmp))
    monkeypatch.setattr(tempfile, "tempdir", None)
    first_dir = tmp_path / "dnsmos-a"
    second_dir = tmp_path / "dnsmos-b"
    first_digest = _write_dnsmos_asset(first_dir, "child-a", score=3.2)
    second_digest = _write_dnsmos_asset(second_dir, "child-b", score=4.3)
    context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        child_pid, observed = pool.submit(
            _spawn_dnsmos_asset_isolation_probe,
            ((str(first_dir), first_digest), (str(second_dir), second_digest)),
        ).result(timeout=30)

    assert child_pid != os.getpid()
    assert [entry["asset_id"] for entry in observed] == ["child-a", "child-b"]
    assert [entry["score"] for entry in observed] == [3.2, 4.3]
    assert [entry["primary_bytes"] for entry in observed] == [
        b"primary-child-a", b"primary-child-b",
    ]
    assert all(entry["module_is_private"] for entry in observed)
    assert all(entry["previous_module_removed"] for entry in observed)
    assert observed[0]["module_name"] != observed[1]["module_name"]
    assert list(controlled_tmp.iterdir()) == []


def test_dnsmos_gpu_load_failure_never_constructs_cpu_session(monkeypatch, tmp_path):
    calls = []
    origin = tmp_path / "dnsmos"
    digest = _write_dnsmos_asset(origin, "gpu-failure", fail_gpu=True)

    def record_calls(module):
        compute_score = module.ComputeScore

        def construct(primary, p808, use_gpu=False):
            calls.append(use_gpu)
            return compute_score(primary, p808, use_gpu=use_gpu)

        module.ComputeScore = construct

    monkeypatch.setattr(mos, "_apply_dnsmos_vram_fix", record_calls)
    estimator = mos.DNSMOSEstimator(
        use_gpu=True,
        model_dir=str(origin),
        verified_model_tree_sha256=digest,
    )

    assert calls == [True]
    assert estimator.compute_score is None
    assert estimator._is_fallback is True
    estimator.close()


def _provider_attestation(
    model, *, device="cuda", dnsmos_p808_device="cuda", scoreq_providers=None,
):
    if model == "scoreq":
        providers = scoreq_providers or (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda" else ["CPUExecutionProvider"]
        )
        return {
            "execution_mode": (
                "cuda_primary_hybrid_ep" if len(providers) > 1
                else "cuda_only_ep" if device == "cuda" else "cpu_ep"
            ),
            "providers": providers,
            "device": "gpu" if device == "cuda" else "cpu",
        }
    if model == "dnsmos":
        main = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        p808 = (["CUDAExecutionProvider"] if dnsmos_p808_device == "cuda"
                else ["CPUExecutionProvider"])
        value = {"onnx_sess": main, "p808_onnx_sess": p808}
        if device == "cuda":
            value["cpu_ep_fallback_disabled"] = False
        return value
    if model == "nisqa":
        return {
            "device": device,
            "parameters": device,
            "checkpoint_sha256": _NISQA_PROVIDER_OPTIONS["checkpoint_sha256"],
        }
    value = {"device": device, "parameters": device}
    if model == "utmos":
        value["checkpoint_sha256"] = _UTMOS_PROVIDER_OPTIONS[
            "checkpoint_sha256"
        ]
    return value


class _PublicAttestationEstimator:
    """只暴露 Step3 允许依赖的 estimator 公共证明接口。"""

    def __init__(self, attestation):
        self.attestation = attestation
        self.calls = 0

    def runtime_attestation(self):
        self.calls += 1
        return self.attestation


def _provider_estimator(
    model, *, device="cuda", dnsmos_p808_device="cuda", scoreq_providers=None,
):
    return _PublicAttestationEstimator(_provider_attestation(
        model,
        device=device,
        dnsmos_p808_device=dnsmos_p808_device,
        scoreq_providers=scoreq_providers,
    ))


def _patch_provider_dependencies(monkeypatch, estimators):
    monkeypatch.setattr(
        mos_runtime, "get_mos_estimator", lambda model, **_kwargs: estimators[model],
    )
    monkeypatch.setattr(
        mos_runtime, "verify_urgent_pk_trusted_assets",
        lambda *_args, **_kwargs: {"checkpoint": "verified"},
    )
    monkeypatch.setattr(
        mos_runtime, "urgent_pk_verified_assets",
        lambda *_args, **_kwargs: {"checkpoint": "verified"},
    )


_FIVE_MODELS = ("scoreq", "urgent_pk", "dnsmos", "nisqa", "utmos")
_TREE_DIGESTS = {
    "scoreq": "1" * 64,
    "dnsmos": "2" * 64,
    "nisqa": "3" * 64,
    "utmos": "4" * 64,
}
_CHECKPOINT_IDENTITIES = {
    "nisqa": {"checkpoint_sha256": "5" * 64, "checkpoint_size_bytes": 501},
    "utmos": {"checkpoint_sha256": "6" * 64, "checkpoint_size_bytes": 601},
}
_URGENT_IDENTITY = {
    "checkpoint_sha256": "7" * 64,
    "source_commit": "8" * 40,
    "source_package_sha256": "9" * 64,
}
_SCOREQ_PROVIDER_OPTIONS = {
    "model_dir": "/assets/scoreq",
    "verified_model_tree_sha256": "1" * 64,
}
_DNSMOS_PROVIDER_OPTIONS = {
    "model_dir": "/assets/dnsmos",
    "verified_model_tree_sha256": "2" * 64,
}
_URGENT_PROVIDER_OPTIONS = {
    "checkpoint": "u",
    "source_dir": "s",
    "encoder": "wav2vec2",
    "backbone": "cnn",
    "score_range": {"min": 1, "max": 5},
    "expected_sha256": "7" * 64,
    "expected_commit": "8" * 40,
}
_NISQA_PROVIDER_OPTIONS = {
    "model_path": "/assets/nisqa/model.tar",
    "model_version": "nisqa-v2",
    "model_sha256": "3" * 64,
    "checkpoint_sha256": "5" * 64,
    "checkpoint_size_bytes": 501,
}
_UTMOS_PROVIDER_OPTIONS = {
    "model_path": "/assets/utmos/model.pt",
    "model_version": "utmos-v1",
    "model_sha256": "4" * 64,
    "checkpoint_sha256": "6" * 64,
    "checkpoint_size_bytes": 601,
}
_PROVIDER_OPTION_CASES = {
    "scoreq": ("scoreq_options", _SCOREQ_PROVIDER_OPTIONS),
    "urgent_pk": ("urgent_options", _URGENT_PROVIDER_OPTIONS),
    "dnsmos": ("dnsmos_options", _DNSMOS_PROVIDER_OPTIONS),
    "nisqa": ("nisqa_options", _NISQA_PROVIDER_OPTIONS),
    "utmos": ("utmos_options", _UTMOS_PROVIDER_OPTIONS),
}


@pytest.mark.parametrize("model", _FIVE_MODELS)
@pytest.mark.parametrize("change", ("missing", "extra"))
def test_runtime_options_apply_exact_schema_to_all_models(model, change):
    option_name, valid = _PROVIDER_OPTION_CASES[model]
    invalid = dict(valid)
    if change == "missing":
        invalid.pop(next(iter(invalid)))
    else:
        invalid["unexpected"] = True

    with pytest.raises(ValueError, match="fields must be exactly"):
        MosRuntimeOptions(**{option_name: invalid})


def _verified_runtime_plan(tmp_path, models=_FIVE_MODELS):
    roots = {
        model: tmp_path / model
        for model in _FIVE_MODELS
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    paths = {
        "scoreq": roots["scoreq"] / "model",
        "urgent_pk": roots["urgent_pk"] / "model.ckpt",
        "urgent_source": roots["urgent_pk"] / "source",
        "dnsmos": roots["dnsmos"] / "model",
        "nisqa": roots["nisqa"] / "model.tar",
        "utmos": roots["utmos"] / "model.pt",
    }
    paths["scoreq"].mkdir(exist_ok=True)
    paths["urgent_source"].mkdir(exist_ok=True)
    paths["dnsmos"].mkdir(exist_ok=True)
    for name in ("urgent_pk", "nisqa", "utmos"):
        paths[name].write_bytes(name.encode("ascii"))

    specs = {
        "scoreq": MosAssetSpec(
            "scoreq", {"model_dir": str(paths["scoreq"])},
        ),
        "urgent_pk": MosAssetSpec(
            "urgent_pk",
            {
                "checkpoint": str(paths["urgent_pk"]),
                "source_dir": str(paths["urgent_source"]),
            },
            {
                "encoder": "wav2vec2",
                "backbone": "cnn",
                "score_range": {"min": 1, "max": 5},
                "expected_sha256": _URGENT_IDENTITY["checkpoint_sha256"],
                "expected_commit": _URGENT_IDENTITY["source_commit"],
            },
        ),
        "dnsmos": MosAssetSpec(
            "dnsmos", {"model_dir": str(paths["dnsmos"])},
        ),
        "nisqa": MosAssetSpec(
            "nisqa",
            {"model_path": str(paths["nisqa"])},
            {"model_version": "nisqa-v2", "model_sha256": _TREE_DIGESTS["nisqa"]},
        ),
        "utmos": MosAssetSpec(
            "utmos",
            {"model_path": str(paths["utmos"])},
            {"model_version": "utmos-v1", "model_sha256": _TREE_DIGESTS["utmos"]},
        ),
    }
    plan = MosExecutionPlan(
        tuple(models),
        tuple(specs[model] for model in models),
        {"kind": "single_process", "device": "cpu"},
    )

    def tree_digest(path):
        resolved = Path(path).resolve()
        for model in ("scoreq", "dnsmos", "nisqa", "utmos"):
            if resolved == roots[model].resolve() or resolved == paths[model].resolve():
                return _TREE_DIGESTS[model]
        raise AssertionError(f"unexpected tree digest path: {resolved}")

    return verify_mos_assets(
        plan,
        tree_digest=tree_digest,
        checkpoint_snapshot=lambda path: dict(
            _CHECKPOINT_IDENTITIES[Path(path).parent.name]
        ),
        urgent_verifier=lambda *_args: dict(_URGENT_IDENTITY),
    )


def _runtime_options_for_verified(verified):
    options = {
        model: dict(model_options)
        for model, model_options in verified.as_mapping()["options_by_model"].items()
    }
    return MosRuntimeOptions(
        urgent_options=options.get("urgent_pk"),
        scoreq_options=options.get("scoreq"),
        dnsmos_options=options.get("dnsmos"),
        nisqa_options=options.get("nisqa"),
        utmos_options=options.get("utmos"),
    )


def _five_model_raw_result():
    return {
        "audio_path": "<in-memory>",
        "scoreq": {"scoreq_mos": 4.5},
        "urgent_pk": {"score": 4.4},
        "dnsmos": {
            "dnsmos_ovr_mos": 4.3,
            "dnsmos_sig_mos": 4.2,
            "dnsmos_bak_mos": 4.1,
            "dnsmos_p808_mos": 4.0,
        },
        "nisqa": {"nisqa_mos": 3.9},
        "utmos": {"utmos_score": 3.8},
    }


def test_runtime_verified_plan_bypasses_legacy_provenance_helper(monkeypatch, tmp_path):
    verified = _verified_runtime_plan(tmp_path, ("scoreq",))
    monkeypatch.setattr(
        mos_runtime,
        "selected_mos_asset_provenance",
        lambda *_args, **_kwargs: pytest.fail(
            "verified plan must bypass the legacy helper"
        ),
    )

    runtime = Step3MosRuntime(
        options=_runtime_options_for_verified(verified),
        verified_plan=verified,
    )

    assert runtime.provenance_payload()["assets"] == verified.as_mapping()["provenance"]


@pytest.mark.parametrize(
    ("model", "field", "replacement"),
    [
        ("scoreq", "model_dir", "/other/scoreq"),
        ("scoreq", "verified_model_tree_sha256", "a" * 64),
        ("urgent_pk", "expected_commit", "9" * 40),
        ("dnsmos", "model_dir", "/other/dnsmos"),
        ("dnsmos", "verified_model_tree_sha256", "b" * 64),
        ("nisqa", "checkpoint_sha256", "a" * 64),
        ("utmos", "model_version", "other-utmos"),
    ],
)
def test_runtime_rejects_options_detached_from_verified_provenance(
    tmp_path, model, field, replacement,
):
    verified = _verified_runtime_plan(tmp_path)
    detached = verified.as_mapping()["options_by_model"]
    detached[model][field] = replacement
    options = MosRuntimeOptions(
        urgent_options=detached["urgent_pk"],
        scoreq_options=detached["scoreq"],
        dnsmos_options=detached["dnsmos"],
        nisqa_options=detached["nisqa"],
        utmos_options=detached["utmos"],
    )

    with pytest.raises(ValueError):
        Step3MosRuntime(options=options, verified_plan=verified)


@pytest.mark.parametrize("model", ("scoreq", "dnsmos"))
def test_runtime_rejects_provenance_digest_detached_from_options(tmp_path, model):
    verified = _verified_runtime_plan(tmp_path, (model,))
    snapshot = verified.as_mapping()
    detached_provenance = snapshot["provenance"]
    detached_provenance["models"][model]["verified_model_tree_sha256"] = "f" * 64
    options = _runtime_options_for_verified(verified)

    with pytest.raises(ValueError, match="digest.*detached|tree digest mismatch"):
        Step3MosRuntime(
            options=options,
            preverified_provenance=detached_provenance,
        )


@pytest.mark.parametrize("model", ("scoreq", "dnsmos"))
@pytest.mark.parametrize(
    "digest",
    ("a" * 63, "a" * 65, "A" * 64, "g" * 64, b"a" * 64),
    ids=("short", "long", "uppercase", "nonhex", "bytes"),
)
def test_runtime_rejects_illegal_directory_model_digest(model, digest, tmp_path):
    verified = _verified_runtime_plan(tmp_path, (model,))
    detached = verified.as_mapping()["options_by_model"][model]
    detached["verified_model_tree_sha256"] = digest
    option_name = "scoreq_options" if model == "scoreq" else "dnsmos_options"

    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        Step3MosRuntime(
            options=MosRuntimeOptions(**{option_name: detached}),
            verified_plan=verified,
        )


def test_runtime_snapshots_nested_options_before_scoring(row, tmp_path):
    verified = _verified_runtime_plan(tmp_path)
    mutable = verified.as_mapping()["options_by_model"]
    expected = verified.as_mapping()["options_by_model"]
    requests = []
    runtime = Step3MosRuntime(
        decoder=lambda _row: (np.zeros(40, dtype=np.float32), 16_000),
        compute_scores=lambda **kwargs: requests.append(kwargs) or _five_model_raw_result(),
        options=MosRuntimeOptions(
            urgent_options=mutable["urgent_pk"],
            scoreq_options=mutable["scoreq"],
            dnsmos_options=mutable["dnsmos"],
            nisqa_options=mutable["nisqa"],
            utmos_options=mutable["utmos"],
        ),
        verified_plan=verified,
    )

    mutable["urgent_pk"]["score_range"]["min"] = -100
    mutable["nisqa"]["checkpoint_sha256"] = "f" * 64
    mutable["scoreq"]["model_dir"] = "/mutated/after/construction"
    mutable["scoreq"]["verified_model_tree_sha256"] = "a" * 64
    mutable["dnsmos"]["verified_model_tree_sha256"] = "b" * 64

    assert runtime(row, _FIVE_MODELS) == {
        "scoreq": 4.5,
        "urgent_pk": 4.4,
        "dnsmos": {"ovrl": 4.3, "sig": 4.2, "bak": 4.1, "p808": 4.0},
        "nisqa": 3.9,
        "utmos": 3.8,
    }
    request = requests[0]
    assert request["urgent_options"] == expected["urgent_pk"]
    assert request["scoreq_options"] == expected["scoreq"]
    assert request["dnsmos_options"] == expected["dnsmos"]
    assert request["nisqa_options"] == expected["nisqa"]
    assert request["utmos_options"] == expected["utmos"]


@pytest.mark.parametrize("model", _FIVE_MODELS)
def test_runtime_verified_five_model_result_schema_is_exact(row, tmp_path, model):
    verified = _verified_runtime_plan(tmp_path)
    raw = _five_model_raw_result()
    raw[model]["unexpected"] = 0.0
    runtime = Step3MosRuntime(
        decoder=lambda _row: (np.zeros(40, dtype=np.float32), 16_000),
        compute_scores=lambda **_kwargs: raw,
        options=_runtime_options_for_verified(verified),
        verified_plan=verified,
    )

    with pytest.raises(ValueError, match=f"{model} result keys"):
        runtime(row, _FIVE_MODELS)


def test_runtime_verified_five_model_exact_schema_and_golden_parity(
    monkeypatch, row, tmp_path,
):
    from audio_filter_pipeline.utils.artifacts import _canonical_json_bytes

    verified = _verified_runtime_plan(tmp_path)
    expected = verified.as_mapping()
    options = _runtime_options_for_verified(verified)
    legacy_calls = []

    def legacy_helper(selected_models, options_by_model):
        legacy_calls.append((selected_models, options_by_model))
        return expected["provenance"]

    monkeypatch.setattr(mos_runtime, "selected_mos_asset_provenance", legacy_helper)
    legacy_payload = Step3MosRuntime(options=options).provenance_payload()
    requests = []
    runtime = Step3MosRuntime(
        decoder=lambda _row: (np.zeros(40, dtype=np.float32), 16_000),
        compute_scores=lambda **kwargs: requests.append(kwargs) or _five_model_raw_result(),
        options=options,
        verified_plan=verified,
    )

    assert _canonical_json_bytes(runtime.provenance_payload()) == (
        _canonical_json_bytes(legacy_payload)
    )
    assert len(legacy_calls) == 1
    assert runtime(row, _FIVE_MODELS) == {
        "scoreq": 4.5,
        "urgent_pk": 4.4,
        "dnsmos": {"ovrl": 4.3, "sig": 4.2, "bak": 4.1, "p808": 4.0},
        "nisqa": 3.9,
        "utmos": 3.8,
    }
    request = requests[0]
    assert set(request) == {
        "audio_data", "mos_types", "use_gpu", "use_cache", "model_specific_sr",
        "allow_fallback", "urgent_options", "scoreq_options", "dnsmos_options",
        "nisqa_options", "utmos_options",
    }
    assert request["mos_types"] == list(_FIVE_MODELS)
    assert request["use_gpu"] is False
    assert request["use_cache"] is True
    assert request["model_specific_sr"] is True
    assert request["allow_fallback"] is False
    assert {
        "scoreq": request["scoreq_options"],
        "urgent_pk": request["urgent_options"],
        "dnsmos": request["dnsmos_options"],
        "nisqa": request["nisqa_options"],
        "utmos": request["utmos_options"],
    } == expected["options_by_model"]


def test_runtime_direct_caller_without_verified_plan_uses_legacy_helper(
    monkeypatch, tmp_path,
):
    calls = []
    provenance = {
        "selected_models": ["scoreq"],
        "models": {
            "scoreq": {
                "model_dir": str((tmp_path / "scoreq").resolve()),
                "verified_model_tree_sha256": "a" * 64,
            },
        },
    }

    def legacy_helper(selected_models, options_by_model):
        calls.append((selected_models, options_by_model))
        return provenance

    monkeypatch.setattr(mos_runtime, "selected_mos_asset_provenance", legacy_helper)
    scoreq_options = {
        "model_dir": str(tmp_path / "scoreq"),
        "verified_model_tree_sha256": "a" * 64,
    }
    runtime = Step3MosRuntime(
        options=MosRuntimeOptions(scoreq_options=scoreq_options),
    )

    payload = runtime.provenance_payload()
    assert payload["assets"] == provenance
    assert payload["asset_load_binding"] == {"version": 3}
    assert payload["scoring_execution_policy"] == dict(
        mos_runtime.SCORING_EXECUTION_POLICY
    )
    assert calls == [(("scoreq",), {
        "scoreq": scoreq_options,
        "urgent_pk": None,
        "dnsmos": None,
        "nisqa": None,
        "utmos": None,
    })]


def test_runtime_execution_policy_provenance_is_canonical_and_isolated(monkeypatch, tmp_path):
    provenance = {
        "selected_models": ["scoreq"],
        "models": {
            "scoreq": {
                "model_dir": str((tmp_path / "scoreq").resolve()),
                "verified_model_tree_sha256": "a" * 64,
            },
        },
    }
    monkeypatch.setattr(
        mos_runtime, "selected_mos_asset_provenance", lambda *_args: provenance,
    )
    runtime = Step3MosRuntime(options=MosRuntimeOptions(scoreq_options={
        "model_dir": str(tmp_path / "scoreq"),
        "verified_model_tree_sha256": "a" * 64,
    }))

    first = runtime.provenance_payload()
    first["scoring_execution_policy"]["batching"]["cross_source"] = False
    second = runtime.provenance_payload()

    assert second["scoring_execution_policy"] == dict(
        mos_runtime.SCORING_EXECUTION_POLICY
    )
    assert second["scoring_execution_policy"] is not first["scoring_execution_policy"]


@pytest.mark.parametrize("change", (
    ("version", 2),
    ("waveform", {"channel": "first", "dtype": "float32", "padding": "zero_right"}),
    ("batching", {
        "max_rows": 32,
        "max_float32_bytes": 64 * 1024 * 1024,
        "decode_reservation": "interval_channels_plus_mono_float32",
        "cross_source": False,
        "flush": "tuple_change_or_limit_or_eof",
    }),
))
def test_runtime_provenance_changes_when_execution_policy_changes(
    monkeypatch, tmp_path, change,
):
    provenance = {
        "selected_models": ["scoreq"],
        "models": {
            "scoreq": {
                "model_dir": str((tmp_path / "scoreq").resolve()),
                "verified_model_tree_sha256": "a" * 64,
            },
        },
    }
    monkeypatch.setattr(
        mos_runtime, "selected_mos_asset_provenance", lambda *_args: provenance,
    )
    runtime = Step3MosRuntime(options=MosRuntimeOptions(scoreq_options={
        "model_dir": str(tmp_path / "scoreq"),
        "verified_model_tree_sha256": "a" * 64,
    }))
    baseline = runtime.provenance_payload()
    changed_policy = __import__("json").loads(__import__("json").dumps(
        baseline["scoring_execution_policy"], sort_keys=True,
    ))
    changed_policy[change[0]] = change[1]
    monkeypatch.setattr(mos_runtime, "SCORING_EXECUTION_POLICY", changed_policy)

    assert runtime.provenance_payload() != baseline


def test_gpu_provider_provenance_uses_public_attestation_for_all_models(monkeypatch):
    estimators = {model: _provider_estimator(model) for model in _FIVE_MODELS}
    _patch_provider_dependencies(monkeypatch, estimators)
    options = MosRuntimeOptions(
        use_gpu=True,
        scoreq_options=_SCOREQ_PROVIDER_OPTIONS,
        urgent_options=_URGENT_PROVIDER_OPTIONS,
        dnsmos_options=_DNSMOS_PROVIDER_OPTIONS,
        nisqa_options=_NISQA_PROVIDER_OPTIONS,
        utmos_options=_UTMOS_PROVIDER_OPTIONS,
    )

    provenance = mos_runtime._runtime_provider_provenance(options, _FIVE_MODELS)

    assert provenance["requested_device"] == "cuda"
    assert provenance["asset_load_binding"] == {"version": 3}
    assert provenance["scoring_execution_policy"] == dict(
        mos_runtime.SCORING_EXECUTION_POLICY
    )
    assert provenance["actual_providers"] == {
        model: _provider_attestation(model) for model in _FIVE_MODELS
    }
    assert all(estimator.calls == 1 for estimator in estimators.values())


@pytest.mark.parametrize("model", _FIVE_MODELS)
@pytest.mark.parametrize("change", ("missing", "extra"))
def test_provider_provenance_rejects_nonexact_public_attestation_schema(
    monkeypatch, model, change,
):
    attestation = _provider_attestation(model)
    if change == "missing":
        attestation.pop(next(iter(attestation)))
    else:
        attestation["unexpected"] = True
    estimator = _PublicAttestationEstimator(attestation)
    _patch_provider_dependencies(monkeypatch, {model: estimator})
    option_name, model_options = _PROVIDER_OPTION_CASES[model]

    with pytest.raises(ValueError, match=f"{model} runtime attestation schema"):
        mos_runtime._runtime_provider_provenance(
            MosRuntimeOptions(use_gpu=True, **{option_name: model_options}),
            (model,),
        )


def test_cpu_and_gpu_scorer_provenance_share_asset_load_binding(monkeypatch, tmp_path):
    cpu_estimators = {
        model: _provider_estimator(model, device="cpu", dnsmos_p808_device="cpu")
        for model in ("scoreq", "dnsmos")
    }
    _patch_provider_dependencies(monkeypatch, cpu_estimators)
    cpu = mos_runtime._runtime_provider_provenance(
        MosRuntimeOptions(
            scoreq_options={
                "model_dir": "/assets/scoreq",
                "verified_model_tree_sha256": "1" * 64,
            },
            dnsmos_options={
                "model_dir": "/assets/dnsmos",
                "verified_model_tree_sha256": "2" * 64,
            },
        ),
        ("scoreq", "dnsmos"),
    )
    gpu = mos_pool.Step3MosScorerPool(
        [0], dataset_id="alpha",
        verified_plan=_verified_runtime_plan(tmp_path, ("scoreq", "dnsmos")),
    ).provenance_payload()

    assert cpu["asset_load_binding"] == {"version": 3}
    assert gpu["asset_load_binding"] == {"version": 3}
    assert gpu["scorer_execution"] == "step3_mos_pool.v6"
    assert gpu["protocol"] == step3_mos_worker._PROTOCOL
    assert gpu["protocol"].endswith(".v6")


@pytest.mark.parametrize(
    "providers",
    [
        ["CPUExecutionProvider"],
        ["CPUExecutionProvider", "CUDAExecutionProvider"],
    ],
)
def test_gpu_provider_provenance_rejects_scoreq_without_cuda_primary(
    monkeypatch, providers,
):
    estimator = _provider_estimator("scoreq", scoreq_providers=providers)
    _patch_provider_dependencies(monkeypatch, {"scoreq": estimator})

    with pytest.raises(ValueError, match="scoreq provider is not cuda-primary"):
        mos_runtime._runtime_provider_provenance(
            MosRuntimeOptions(use_gpu=True, scoreq_options=_SCOREQ_PROVIDER_OPTIONS), ("scoreq",),
        )


def test_gpu_provider_provenance_rejects_scoreq_cpu_device_marker(monkeypatch):
    estimator = _provider_estimator("scoreq")
    estimator.attestation["device"] = "cpu"
    _patch_provider_dependencies(monkeypatch, {"scoreq": estimator})

    with pytest.raises(ValueError, match="scoreq provider is not cuda-primary"):
        mos_runtime._runtime_provider_provenance(
            MosRuntimeOptions(use_gpu=True, scoreq_options=_SCOREQ_PROVIDER_OPTIONS), ("scoreq",),
        )


def test_gpu_provider_provenance_rejects_contradictory_scoreq_execution_mode(
    monkeypatch,
):
    estimator = _provider_estimator("scoreq")
    estimator.attestation["execution_mode"] = "cuda_only_ep"
    _patch_provider_dependencies(monkeypatch, {"scoreq": estimator})

    with pytest.raises(ValueError, match="scoreq provider is not cuda-primary"):
        mos_runtime._runtime_provider_provenance(
            MosRuntimeOptions(
                use_gpu=True, scoreq_options=_SCOREQ_PROVIDER_OPTIONS,
            ),
            ("scoreq",),
        )


def test_runtime_provenance_rejects_local_checkpoint_digest_mismatch(monkeypatch):
    estimator = _provider_estimator("nisqa", device="cpu")
    estimator.attestation["checkpoint_sha256"] = "b" * 64
    _patch_provider_dependencies(monkeypatch, {"nisqa": estimator})

    with pytest.raises(ValueError, match="nisqa checkpoint digest mismatch"):
        mos_runtime._runtime_provider_provenance(
            MosRuntimeOptions(
                use_gpu=False,
                nisqa_options={
                    **_NISQA_PROVIDER_OPTIONS,
                    "checkpoint_sha256": "a" * 64,
                },
            ),
            ("nisqa",),
        )


def test_gpu_provider_provenance_rejects_unhardened_dnsmos_cpu_ep(monkeypatch):
    estimator = _provider_estimator("dnsmos")
    estimator.attestation["onnx_sess"] = [
        "CUDAExecutionProvider", "CPUExecutionProvider",
    ]
    _patch_provider_dependencies(monkeypatch, {"dnsmos": estimator})

    with pytest.raises(ValueError, match="dnsmos .*not cuda"):
        mos_runtime._runtime_provider_provenance(
            MosRuntimeOptions(use_gpu=True, dnsmos_options=_DNSMOS_PROVIDER_OPTIONS), ("dnsmos",),
        )


def test_gpu_provider_provenance_accepts_dnsmos_with_disabled_cpu_fallback(monkeypatch):
    attestation = _provider_attestation("dnsmos")
    attestation.update({
        "onnx_sess": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "p808_onnx_sess": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cpu_ep_fallback_disabled": True,
    })
    estimator = _PublicAttestationEstimator(attestation)
    _patch_provider_dependencies(monkeypatch, {"dnsmos": estimator})

    provenance = mos_runtime._runtime_provider_provenance(
        MosRuntimeOptions(use_gpu=True, dnsmos_options=_DNSMOS_PROVIDER_OPTIONS), ("dnsmos",),
    )

    assert provenance["actual_providers"]["dnsmos"] == attestation


def test_gpu_provider_provenance_rejects_dnsmos_p808_cpu_fallback(monkeypatch):
    estimators = {"dnsmos": _provider_estimator(
        "dnsmos", dnsmos_p808_device="cpu",
    )}
    _patch_provider_dependencies(monkeypatch, estimators)

    with pytest.raises(ValueError, match="p808_onnx_sess provider is not cuda"):
        mos_runtime._runtime_provider_provenance(
            MosRuntimeOptions(use_gpu=True, dnsmos_options=_DNSMOS_PROVIDER_OPTIONS), ("dnsmos",),
        )


@pytest.mark.parametrize("model", ["scoreq", "nisqa", "utmos", "urgent_pk"])
def test_gpu_provider_provenance_rejects_model_cpu_fallback(monkeypatch, model):
    estimators = {model: _provider_estimator(model, device="cpu")}
    _patch_provider_dependencies(monkeypatch, estimators)
    option_name, model_options = _PROVIDER_OPTION_CASES[model]
    options = {option_name: model_options}

    with pytest.raises(ValueError, match=f"{model} .*not cuda"):
        mos_runtime._runtime_provider_provenance(
            MosRuntimeOptions(use_gpu=True, **options), (model,),
        )


@pytest.mark.parametrize("model", ["nisqa", "utmos", "urgent_pk"])
@pytest.mark.parametrize("field", ["device", "parameters"])
@pytest.mark.parametrize(
    ("requested", "invalid_device"),
    [("cpu", "not-cpu"), ("cuda", "not-cuda"), ("cuda", "not-gpu")],
)
def test_runtime_provider_entry_rejects_negated_device_markers(
    model, field, requested, invalid_device,
):
    attestation = _provider_attestation(model, device=requested)
    attestation[field] = invalid_device
    expected_digest = (
        _PROVIDER_OPTION_CASES[model][1].get("checkpoint_sha256")
    )

    with pytest.raises(ValueError, match=f"{model} device is not {requested}"):
        mos_runtime._runtime_provider_entry(
            model,
            attestation,
            requested=requested,
            expected_checkpoint_sha256=expected_digest,
        )


@pytest.mark.parametrize("model", ["nisqa", "utmos", "urgent_pk"])
@pytest.mark.parametrize(
    ("requested", "device"),
    [("cpu", "cpu"), ("cuda", "cuda"), ("cuda", "cuda:7")],
)
def test_runtime_provider_entry_accepts_canonical_device_markers(
    model, requested, device,
):
    attestation = _provider_attestation(model, device=device)
    expected_digest = (
        _PROVIDER_OPTION_CASES[model][1].get("checkpoint_sha256")
    )

    assert mos_runtime._runtime_provider_entry(
        model,
        attestation,
        requested=requested,
        expected_checkpoint_sha256=expected_digest,
    ) == attestation


def test_runtime_provider_provenance_passes_parent_attestation_reader(
    monkeypatch, tmp_path,
):
    """真实 writer 的 tuple attestation 必须原样通过 parent reader，防止双侧 schema 漂移。"""
    estimators = {"scoreq": _provider_estimator("scoreq")}
    monkeypatch.setattr(
        mos_runtime, "get_mos_estimator", lambda model, **_kwargs: estimators[model],
    )
    attestation = mos_runtime._runtime_provider_provenance(
        MosRuntimeOptions(use_gpu=True, scoreq_options=_SCOREQ_PROVIDER_OPTIONS),
        ("scoreq",),
    )

    sidecar = tmp_path / "runtime-attestation.json"
    step3_mos_worker._write_runtime_attestation(
        str(sidecar), ("scoreq",), [{
            "models": ["scoreq"], "attestation": attestation,
        }],
    )

    value = mos_gpu_job._read_runtime_attestation(str(sidecar), ("scoreq",))
    assert value["worker_models"] == ["scoreq"]


def test_unselected_urgent_provider_does_not_access_assets(monkeypatch):
    estimators = {"scoreq": _provider_estimator("scoreq")}
    monkeypatch.setattr(
        mos_runtime, "get_mos_estimator", lambda model, **_kwargs: estimators[model],
    )
    monkeypatch.setattr(
        mos_runtime, "verify_urgent_pk_trusted_assets",
        lambda *_args, **_kwargs: pytest.fail("unselected URGENT assets must remain lazy"),
    )

    provenance = mos_runtime._runtime_provider_provenance(
        MosRuntimeOptions(
            use_gpu=True, scoreq_options=_SCOREQ_PROVIDER_OPTIONS,
            urgent_options=_URGENT_PROVIDER_OPTIONS,
        ),
        ("scoreq",),
    )

    assert provenance["selected_models"] == ["scoreq"]


@pytest.mark.parametrize(
    ("outcome", "error"),
    (("success", None), ("mismatch", ValueError), ("exception", RuntimeError)),
)
def test_uncached_provider_preflight_always_closes_estimator(
    monkeypatch, outcome, error,
):
    estimator = _provider_estimator(
        "scoreq", device="cpu",
        scoreq_providers=["CPUExecutionProvider"],
    )
    close_calls = []
    estimator.close = lambda: close_calls.append(True)
    if outcome == "mismatch":
        estimator.attestation["device"] = "gpu"
    elif outcome == "exception":
        estimator.runtime_attestation = lambda: (_ for _ in ()).throw(
            RuntimeError("provider attestation exploded")
        )
    get_calls = []

    def get_estimator(model, **kwargs):
        get_calls.append((model, kwargs))
        return estimator

    monkeypatch.setattr(mos_runtime, "get_mos_estimator", get_estimator)
    options = MosRuntimeOptions(
        use_gpu=outcome == "mismatch",
        use_cache=False,
        scoreq_options=_SCOREQ_PROVIDER_OPTIONS,
    )

    if error is None:
        result = mos_runtime._runtime_provider_provenance(options, ("scoreq",))
        assert result["actual_providers"]["scoreq"]["execution_mode"] == "cpu_ep"
    else:
        with pytest.raises(error):
            mos_runtime._runtime_provider_provenance(options, ("scoreq",))

    assert get_calls[0][1]["use_cache"] is False
    assert close_calls == [True]


@pytest.mark.parametrize("retry", ("clear", "atexit"))
def test_uncached_provider_preflight_retains_failed_cleanup_and_primary_error(
    monkeypatch, retry,
):
    estimator = _provider_estimator(
        "scoreq", device="cpu",
        scoreq_providers=["CPUExecutionProvider"],
    )
    close_calls = []

    def close():
        close_calls.append(True)
        if len(close_calls) == 1:
            raise RuntimeError("provider estimator close exploded")

    estimator.close = close
    monkeypatch.setattr(
        mos_runtime, "get_mos_estimator", lambda *_args, **_kwargs: estimator,
    )
    monkeypatch.setattr(mos, "_MODEL_CACHE", {})
    monkeypatch.setattr(mos, "_PENDING_ESTIMATOR_CLEANUP", {})
    options = MosRuntimeOptions(
        use_gpu=True,
        use_cache=False,
        scoreq_options=_SCOREQ_PROVIDER_OPTIONS,
    )

    with pytest.raises(ValueError, match="scoreq provider is not cuda-primary") as caught:
        mos_runtime._runtime_provider_provenance(options, ("scoreq",))

    assert any(
        "provider estimator cleanup failed: provider estimator close exploded" in note
        for note in caught.value.__notes__
    )
    assert list(mos._PENDING_ESTIMATOR_CLEANUP.values()) == [estimator]
    if retry == "clear":
        mos.clear_model_cache()
    else:
        mos._cleanup_mos_at_exit()
    assert close_calls == [True, True]
    assert mos._PENDING_ESTIMATOR_CLEANUP == {}


def test_provider_preflight_rejects_missing_selected_options_before_loading(monkeypatch):
    monkeypatch.setattr(
        mos_runtime, "get_mos_estimator",
        lambda *_args, **_kwargs: pytest.fail("missing options must fail before loading"),
    )

    with pytest.raises(ValueError, match="selected nisqa is missing runtime options"):
        mos_runtime._runtime_provider_provenance(
            MosRuntimeOptions(scoreq_options=_SCOREQ_PROVIDER_OPTIONS),
            ("scoreq", "nisqa"),
        )


def test_runtime_attestation_rejects_nonverified_subset_before_preflight(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(
        mos_runtime, "_runtime_provider_provenance",
        lambda *_args, **_kwargs: pytest.fail(
            "nonverified selection must fail before provider preflight"
        ),
    )
    verified = _verified_runtime_plan(tmp_path, ("scoreq",))
    runtime = Step3MosRuntime(
        options=_runtime_options_for_verified(verified),
        verified_plan=verified,
    )

    with pytest.raises(ValueError, match="subset of verified selected models"):
        runtime.runtime_attestation(("scoreq", "nisqa"))


def test_runtime_attestation_concurrent_first_call_runs_preflight_once(monkeypatch):
    calls = []
    entered = threading.Event()

    def attest(_options, selected):
        calls.append(tuple(selected))
        entered.set()
        time.sleep(0.05)
        return {
            "requested_device": "cpu",
            "actual_providers": {"scoreq": {"providers": ["CPUExecutionProvider"]}},
            "fallback": False,
            "selected_models": ["scoreq"],
            "asset_load_binding": {"version": 3},
        }

    monkeypatch.setattr(mos_runtime, "_runtime_provider_provenance", attest)
    runtime = Step3MosRuntime(
        options=MosRuntimeOptions(scoreq_options=_SCOREQ_PROVIDER_OPTIONS),
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(runtime.runtime_attestation, ("scoreq",))
            for _ in range(8)
        ]
        assert entered.wait(timeout=1)
        results = [future.result(timeout=2) for future in futures]

    assert calls == [("scoreq",)]
    assert all(result == results[0] for result in results)


def test_runtime_attestation_returns_deeply_isolated_copies(monkeypatch):
    source = {
        "requested_device": "cpu",
        "actual_providers": {"scoreq": {"providers": ["CPUExecutionProvider"]}},
        "fallback": False,
        "selected_models": ["scoreq"],
        "asset_load_binding": {"version": 3},
    }
    monkeypatch.setattr(
        mos_runtime, "_runtime_provider_provenance",
        lambda *_args, **_kwargs: source,
    )
    runtime = Step3MosRuntime(
        options=MosRuntimeOptions(scoreq_options=_SCOREQ_PROVIDER_OPTIONS),
    )

    first = runtime.runtime_attestation(("scoreq",))
    source["actual_providers"]["scoreq"]["providers"].append("mutated-source")
    first["actual_providers"]["scoreq"]["providers"].append("mutated-return")
    second = runtime.runtime_attestation(("scoreq",))

    assert second["actual_providers"]["scoreq"]["providers"] == [
        "CPUExecutionProvider"
    ]
    assert second is not first
    assert second["actual_providers"] is not first["actual_providers"]


def test_worker_uses_canonical_runtime_attestation(monkeypatch):
    calls = []
    request = {
        "private_dir": "/tmp",
        "subshards": [{"models": ["scoreq"], "row_count": 1}],
        "urgent_options": None,
        "scoreq_options": _SCOREQ_PROVIDER_OPTIONS,
        "dnsmos_options": None,
        "nisqa_options": None,
        "utmos_options": None,
        "attestation_path": "/tmp/attestation.jsonl",
    }

    class Runtime:
        def __init__(self, *, options):
            calls.append(("init", options.use_gpu))

        def runtime_attestation(self, models):
            calls.append(("attest", tuple(models)))
            return {"selected_models": list(models)}

    monkeypatch.setattr(step3_mos_worker, "_load_request", lambda _path: request)
    monkeypatch.setattr(
        step3_mos_worker,
        "_prevalidate_subshard_input",
        lambda *_args: (io.BytesIO(), (1, 2, 3, 4, 5), None, None),
    )
    registry = types.SimpleNamespace(finalize=lambda _count: None, close=lambda: None)
    monkeypatch.setattr(
        step3_mos_worker, "DiskBackedOrdinalRegistry", lambda _path: registry,
    )
    monkeypatch.setattr(
        step3_mos_worker, "_verify_validated_subshard", lambda *_args: None,
    )
    monkeypatch.setattr(step3_mos_worker, "Step3MosRuntime", Runtime)
    monkeypatch.setattr(
        step3_mos_worker, "_score_subshard",
        lambda runtime, subshard, expected, **_kwargs: calls.append(
            ("score", runtime, tuple(subshard["models"]), expected)
        ),
    )
    monkeypatch.setattr(
        step3_mos_worker, "_write_runtime_attestation",
        lambda path, models, attestations: calls.append(
            ("write", path, models, attestations)
        ),
    )

    step3_mos_worker.handle_request("/tmp/request.json", expected_dataset_id="dataset")

    assert calls[0] == ("init", True)
    assert calls[1] == ("attest", ("scoreq",))
    assert calls[2][0] == "score"
    assert calls[2][2:] == (("scoreq",), "dataset")
    assert calls[3] == (
        "write",
        "/tmp/attestation.jsonl",
        ("scoreq",),
        [{
            "models": ["scoreq"],
            "attestation": {"selected_models": ["scoreq"]},
        }],
    )


def _install_tiny_utmos(monkeypatch):
    class TinyUTMOS(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

    module_names = (
        "speechmos",
        "speechmos.utmos22",
        "speechmos.utmos22.strong",
        "speechmos.utmos22.strong.model",
    )
    modules = [types.ModuleType(name) for name in module_names]
    modules[-1].UTMOS22Strong = TinyUTMOS
    modules[0].utmos22 = modules[1]
    modules[1].strong = modules[2]
    modules[2].model = modules[3]
    for name, module in zip(module_names, modules):
        monkeypatch.setitem(sys.modules, name, module)


def test_nisqa_loads_descriptor_snapshot_across_path_swap_restore(monkeypatch, tmp_path):
    checkpoint = tmp_path / "nisqa.tar"
    checkpoint.write_bytes(b"checkpoint-a")
    replacement = tmp_path / "replacement.tar"
    replacement.write_bytes(b"checkpoint-b")
    held_original = tmp_path / "held-original.tar"
    calls = []
    snapshot_paths = []
    original_torch_load = torch.load

    package = types.ModuleType("nisqa")
    module = types.ModuleType("nisqa.NISQA_model")
    moved = []

    class TinyModel:
        @staticmethod
        def to(device):
            moved.append(device)

    class TinyNisqa:
        def __init__(self, args):
            self.args = args
            self.dev = args["tr_device"]
            self.model = TinyModel()
            torch.load(args["pretrained_model"], map_location=self.dev)

    module.nisqaModel = TinyNisqa
    package.NISQA_model = module
    monkeypatch.setitem(sys.modules, "nisqa", package)
    monkeypatch.setitem(sys.modules, "nisqa.NISQA_model", module)
    monkeypatch.setattr(mos.sf, "write", lambda *_args, **_kwargs: None)

    def fake_safe_load(path, **kwargs):
        snapshot = Path(path)
        snapshot_paths.append(snapshot)
        assert snapshot.resolve() != checkpoint.resolve()
        assert snapshot.read_bytes() == b"checkpoint-a"
        assert snapshot.stat().st_mode & 0o777 in {0o400, 0o600}
        assert re.fullmatch(r"/proc/self/fd/\d+", os.fspath(snapshot))
        os.replace(checkpoint, held_original)
        os.replace(replacement, checkpoint)
        assert checkpoint.read_bytes() == b"checkpoint-b"
        os.replace(checkpoint, replacement)
        os.replace(held_original, checkpoint)
        calls.append(kwargs)
        return {"model": {}}

    monkeypatch.setattr(mos, "safe_torch_load", fake_safe_load)
    estimator = mos.NISQAEstimator(model_path=str(checkpoint))

    assert estimator._nisqa_model is not None
    assert estimator._checkpoint_sha256 == hashlib.sha256(b"checkpoint-a").hexdigest()
    assert calls == [{
        "map_location": "cpu",
        "purpose": "load verified NISQA weights safely",
    }]
    assert moved == ["cpu"]
    assert torch.load is original_torch_load
    assert snapshot_paths and all(not path.exists() for path in snapshot_paths)


def test_nisqa_safe_load_failure_is_fail_closed_and_restores_torch_load(
    monkeypatch, tmp_path,
):
    checkpoint = tmp_path / "nisqa.tar"
    checkpoint.write_bytes(b"placeholder")
    original_torch_load = torch.load

    package = types.ModuleType("nisqa")
    module = types.ModuleType("nisqa.NISQA_model")

    class TinyNisqa:
        def __init__(self, args):
            torch.load(args["pretrained_model"], map_location=args["tr_device"])

    module.nisqaModel = TinyNisqa
    package.NISQA_model = module
    monkeypatch.setitem(sys.modules, "nisqa", package)
    monkeypatch.setitem(sys.modules, "nisqa.NISQA_model", module)
    monkeypatch.setattr(mos.sf, "write", lambda *_args, **_kwargs: None)
    snapshot_paths = []

    def fail_safe_load(path, **_kwargs):
        snapshot_paths.append(Path(path))
        raise RuntimeError("Cannot safely load checkpoint: requires pickled globals")

    monkeypatch.setattr(mos, "safe_torch_load", fail_safe_load)

    with pytest.raises(RuntimeError, match="Cannot safely load"):
        mos.NISQAEstimator(model_path=str(checkpoint))
    assert torch.load is original_torch_load
    assert snapshot_paths and all(not path.exists() for path in snapshot_paths)


def test_utmos_uses_private_snapshot_and_strict_state_dict(monkeypatch, tmp_path):
    _install_tiny_utmos(monkeypatch)
    checkpoint = tmp_path / "utmos.pt"
    checkpoint.write_bytes(b"checkpoint-a")
    calls = []
    snapshot_paths = []

    def fake_safe_load(path, **kwargs):
        snapshot = Path(path)
        snapshot_paths.append(snapshot)
        assert snapshot.resolve() != checkpoint.resolve()
        assert snapshot.read_bytes() == b"checkpoint-a"
        assert snapshot.stat().st_mode & 0o777 in {0o400, 0o600}
        assert re.fullmatch(r"/proc/self/fd/\d+", os.fspath(snapshot))
        calls.append(kwargs)
        return {"weight": torch.ones(1)}

    monkeypatch.setattr(mos, "safe_torch_load", fake_safe_load)
    estimator = mos.UTMOSEstimator(model_path=str(checkpoint))

    assert estimator.model.weight.item() == pytest.approx(1.0)
    assert estimator._checkpoint_sha256 == hashlib.sha256(b"checkpoint-a").hexdigest()
    assert calls == [{
        "map_location": "cpu",
        "purpose": "load bundled UTMOS weights safely",
    }]
    assert "allow_unsafe" not in calls[0]
    assert snapshot_paths and all(not path.exists() for path in snapshot_paths)


def test_utmos_loads_descriptor_snapshot_across_path_swap_restore(monkeypatch, tmp_path):
    _install_tiny_utmos(monkeypatch)
    checkpoint = tmp_path / "utmos.pt"
    checkpoint.write_bytes(b"checkpoint-a")
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"checkpoint-b")
    held_original = tmp_path / "held-original.pt"

    def swap_restore_during_load(path, **_kwargs):
        assert Path(path).read_bytes() == b"checkpoint-a"
        os.replace(checkpoint, held_original)
        os.replace(replacement, checkpoint)
        assert checkpoint.read_bytes() == b"checkpoint-b"
        os.replace(checkpoint, replacement)
        os.replace(held_original, checkpoint)
        return {"weight": torch.ones(1)}

    monkeypatch.setattr(mos, "safe_torch_load", swap_restore_during_load)

    estimator = mos.UTMOSEstimator(model_path=str(checkpoint))
    assert estimator.model.weight.item() == pytest.approx(1.0)
    assert estimator._checkpoint_sha256 == hashlib.sha256(b"checkpoint-a").hexdigest()


def test_utmos_rejects_checkpoint_replacement_during_safe_load(monkeypatch, tmp_path):
    _install_tiny_utmos(monkeypatch)
    checkpoint = tmp_path / "utmos.pt"
    checkpoint.write_bytes(b"before")

    def replace_during_load(path, **_kwargs):
        assert Path(path).read_bytes() == b"before"
        checkpoint.write_bytes(b"after")
        return {"weight": torch.ones(1)}

    monkeypatch.setattr(mos, "safe_torch_load", replace_during_load)

    with pytest.raises(RuntimeError, match="changed while loading"):
        mos.UTMOSEstimator(model_path=str(checkpoint))


def test_utmos_rejects_snapshot_digest_that_differs_from_preflight(monkeypatch, tmp_path):
    _install_tiny_utmos(monkeypatch)
    checkpoint = tmp_path / "utmos.pt"
    checkpoint.write_bytes(b"runtime-bytes")
    monkeypatch.setattr(
        mos,
        "safe_torch_load",
        lambda *_args, **_kwargs: pytest.fail("mismatched snapshot must not load"),
    )

    with pytest.raises(RuntimeError, match="checkpoint digest mismatch"):
        mos.UTMOSEstimator(
            model_path=str(checkpoint), checkpoint_sha256="0" * 64,
        )


def test_utmos_rejects_state_dict_shape_mismatch(monkeypatch, tmp_path):
    _install_tiny_utmos(monkeypatch)
    checkpoint = tmp_path / "utmos.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(
        mos,
        "safe_torch_load",
        lambda *_args, **_kwargs: {"weight": torch.ones(2)},
    )

    with pytest.raises(RuntimeError, match="shape_mismatches"):
        mos.UTMOSEstimator(model_path=str(checkpoint))


def test_utmos_safe_load_failure_does_not_fallback(monkeypatch, tmp_path):
    _install_tiny_utmos(monkeypatch)
    checkpoint = tmp_path / "utmos.pt"
    checkpoint.write_bytes(b"placeholder")

    def fail_safe_load(*_args, **_kwargs):
        raise RuntimeError(
            "Cannot safely load checkpoint: requires non-numpy pickled global(s)"
        )

    monkeypatch.setattr(mos, "safe_torch_load", fail_safe_load)

    with pytest.raises(RuntimeError, match="Cannot safely load"):
        mos.UTMOSEstimator(model_path=str(checkpoint))


def test_urgent_pk_delegates_safe_and_verified_unsafe_loads(monkeypatch):
    calls = []
    gates = []

    def fake_safe_load(path, **kwargs):
        calls.append((path, kwargs))
        if len(calls) == 1:
            raise RuntimeError(
                "Cannot safely load checkpoint: requires non-numpy pickled global(s)"
            )
        return {"model": {}}

    monkeypatch.setattr(mos, "safe_torch_load", fake_safe_load)
    monkeypatch.setattr(mos, "_urgent_pk_trusted_load_gate", gates.append)

    assert mos._load_urgent_pk_checkpoint("urgent.ckpt") == {"model": {}}
    assert gates == ["urgent.ckpt"]
    assert calls == [
        (
            "urgent.ckpt",
            {
                "map_location": "cpu",
                "purpose": "read URGENT-PK checkpoint metadata safely",
            },
        ),
        (
            "urgent.ckpt",
            {
                "map_location": "cpu",
                "allow_unsafe": True,
                "purpose": "load digest/commit-verified URGENT-PK checkpoint",
            },
        ),
    ]


def test_urgent_pk_gate_failure_prevents_unsafe_load(monkeypatch):
    calls = []

    def fake_safe_load(path, **kwargs):
        calls.append((path, kwargs))
        raise RuntimeError(
            "Cannot safely load checkpoint: requires non-numpy pickled global(s)"
        )

    def reject_gate(_path):
        raise mos.UrgentPKTrustError("digest mismatch")

    monkeypatch.setattr(mos, "safe_torch_load", fake_safe_load)
    monkeypatch.setattr(mos, "_urgent_pk_trusted_load_gate", reject_gate)

    with pytest.raises(mos.UrgentPKTrustError, match="digest mismatch"):
        mos._load_urgent_pk_checkpoint("urgent.ckpt")
    assert len(calls) == 1
    assert "allow_unsafe" not in calls[0][1]


def _write_urgent_pk_source(root: Path, asset_id: str) -> None:
    package = root / "urgentpk"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(
        "from .urgentpk_model import AbTestModel\n", encoding="utf-8",
    )
    (package / "urgentpk_model.py").write_text(
        f"ASSET_ID = {asset_id!r}\nclass AbTestModel:\n    asset_id = ASSET_ID\n",
        encoding="utf-8",
    )


def _init_pinned_urgent_pk_source(tmp_path, asset_id="approved"):
    source_dir = tmp_path / "urgent-source"
    _write_urgent_pk_source(source_dir, asset_id)
    subprocess.run(
        ["git", "init", str(source_dir)], check=True, capture_output=True, timeout=30,
    )
    subprocess.run(
        ["git", "-C", str(source_dir), "add", "urgentpk"], check=True, timeout=30,
    )
    subprocess.run(
        ["git", "-C", str(source_dir), "-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture"],
        check=True,
        capture_output=True,
        timeout=30,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True, timeout=30,
    ).strip()
    return source_dir, commit


@pytest.mark.parametrize("mutation", ("tracked", "untracked"))
def test_urgent_pk_pinned_verifier_rejects_dirty_and_untracked_package_content(
    tmp_path, mutation,
):
    source_dir, commit = _init_pinned_urgent_pk_source(tmp_path)
    checkpoint = tmp_path / "urgent.ckpt"
    checkpoint.write_bytes(b"approved checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if mutation == "tracked":
        (source_dir / "urgentpk" / "urgentpk_model.py").write_text(
            "class AbTestModel: pass\n", encoding="utf-8",
        )
    else:
        (source_dir / "urgentpk" / "untracked.py").write_text(
            "VALUE = 1\n", encoding="utf-8",
        )

    with pytest.raises(mos.UrgentPKTrustError, match="批准 (commit|blob) 不一致"):
        mos.verify_urgent_pk_trusted_assets(
            str(checkpoint), str(source_dir), digest, commit,
        )


def test_urgent_pk_pinned_verifier_records_approved_package_snapshot_identity(tmp_path):
    source_dir, commit = _init_pinned_urgent_pk_source(tmp_path)
    checkpoint = tmp_path / "urgent.ckpt"
    checkpoint.write_bytes(b"approved checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    verified = mos.verify_urgent_pk_trusted_assets(
        str(checkpoint), str(source_dir), digest, commit,
    )

    assert verified["checkpoint_sha256"] == digest
    assert verified["source_commit"] == commit
    assert re.fullmatch(r"[0-9a-f]{64}", verified["source_package_sha256"])
    assert mos.urgent_pk_verified_assets(str(checkpoint), str(source_dir)) == verified


def test_urgent_pk_loader_rejects_source_replacement_after_snapshot(
    monkeypatch, tmp_path,
):
    source_dir, commit = _init_pinned_urgent_pk_source(tmp_path, "first")
    original_open = mos._open_urgent_pk_pinned_package_snapshot
    calls = 0

    def mutate_before_post_import_snapshot(source, expected_commit):
        nonlocal calls
        calls += 1
        if calls == 2:
            (Path(source) / "urgentpk" / "injected.py").write_text(
                "VALUE = 1\n", encoding="utf-8",
            )
        return original_open(source, expected_commit)

    monkeypatch.setattr(
        mos, "_open_urgent_pk_pinned_package_snapshot",
        mutate_before_post_import_snapshot,
    )
    before = dict(sys.modules)
    try:
        with pytest.raises(mos.UrgentPKTrustError, match="批准 commit 不一致"):
            mos.load_urgent_pk_model_module(str(source_dir), commit)
        assert not mos._URGENT_PK_PRIVATE_MODULES
        assert all(
            name not in sys.modules or sys.modules[name] is module
            for name, module in before.items()
        )
    finally:
        mos.clear_model_cache()
        for name in tuple(sys.modules):
            if name not in before and (
                name == "urgentpk" or name.startswith("urgentpk.")
                or name == "model_mod" or name.startswith("_urgent_pk_")
            ):
                sys.modules.pop(name, None)


def test_urgent_pk_private_namespace_loads_only_pinned_clean_package(monkeypatch, tmp_path):
    source_dir, commit = _init_pinned_urgent_pk_source(tmp_path, "first")
    before = dict(sys.modules)

    try:
        module = mos.load_urgent_pk_model_module(str(source_dir), commit)
        owned = dict(mos._URGENT_PK_PRIVATE_MODULES)
        assert module.AbTestModel.asset_id == "first"
        assert owned
        assert all(
            "/proc/self/fd/" in str(getattr(item, "__file__", ""))
            for item in owned.values() if getattr(item, "__file__", None)
        )
        (source_dir / "urgentpk" / "injected.py").write_text("VALUE = 1\n", encoding="utf-8")
        with pytest.raises(mos.UrgentPKTrustError, match="批准 commit 不一致"):
            mos.load_urgent_pk_model_module(str(source_dir), commit)
        assert all(sys.modules.get(name) is module for name, module in owned.items())
    finally:
        mos.clear_model_cache()
        for name in tuple(sys.modules):
            if name not in before and (
                name == "urgentpk" or name.startswith("urgentpk.")
                or name == "model_mod" or name.startswith("_urgent_pk_")
            ):
                sys.modules.pop(name, None)


def test_dnsmos_private_package_supports_relative_sibling_import(
    monkeypatch, tmp_path,
):
    origin = tmp_path / "dnsmos-relative"
    _write_dnsmos_asset(origin, "relative")
    (origin / "helper.py").write_text("SCORE_OFFSET = 0.25\n", encoding="utf-8")
    script = origin / "dnsmos_local.py"
    script.write_text(
        "from .helper import SCORE_OFFSET\n" + script.read_text(encoding="utf-8")
        + "\nASSET_SCORE += SCORE_OFFSET\n",
        encoding="utf-8",
    )
    digest = local_model_tree_sha256(str(origin))
    monkeypatch.setattr(mos, "_apply_dnsmos_vram_fix", lambda _module: None)

    estimator = mos.DNSMOSEstimator(
        model_dir=str(origin), verified_model_tree_sha256=digest,
        allow_fallback=False,
    )
    module_name = type(estimator.compute_score).__module__
    package_name = module_name.rpartition(".")[0]

    try:
        assert package_name
        assert sys.modules[module_name].__package__ == package_name
        assert sys.modules[f"{package_name}.helper"].SCORE_OFFSET == pytest.approx(0.25)
        assert estimator.compute_score.asset_id == "relative"
        assert estimator.compute_score.run_batch_audio([(np.zeros(1), 16_000)])[0][
            "OVRL"
        ] == pytest.approx(4.25)
    finally:
        estimator.close()
    assert package_name not in sys.modules
    assert module_name not in sys.modules
    assert f"{package_name}.helper" not in sys.modules


def test_urgent_pk_checkpoint_race_keeps_digest_metadata_and_model_on_one_snapshot(
    monkeypatch, tmp_path,
):
    checkpoint = tmp_path / "urgent.ckpt"
    replacement = tmp_path / "replacement.ckpt"
    held = tmp_path / "held.ckpt"
    checkpoint.write_bytes(b"checkpoint-original")
    replacement.write_bytes(b"checkpoint-replacement")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    seen = []

    class ModelClass:
        @staticmethod
        def load_from_checkpoint(path):
            seen.append(("model", path, Path(path).read_bytes()))
            assert checkpoint.read_bytes() == b"checkpoint-replacement"
            os.replace(checkpoint, replacement)
            os.replace(held, checkpoint)
            return object()

    def verify(path, source, expected_sha256, expected_commit):
        seen.append(("digest", path, Path(path).read_bytes()))
        assert source == str(source_dir)
        assert expected_sha256 == hashlib.sha256(b"checkpoint-original").hexdigest()
        assert expected_commit == "a" * 40
        return {
            "checkpoint_sha256": expected_sha256,
            "source_commit": expected_commit,
            "source_package_sha256": "b" * 64,
        }

    def metadata(path):
        seen.append(("metadata", path, Path(path).read_bytes()))
        os.replace(checkpoint, held)
        os.replace(replacement, checkpoint)
        return {"score_range": [1.0, 5.0]}

    monkeypatch.setattr(mos, "verify_urgent_pk_trusted_assets", verify)
    monkeypatch.setattr(mos, "read_urgent_pk_checkpoint_metadata", metadata)
    estimator = mos.URGENTPKEstimator(
        checkpoint=str(checkpoint), source_dir=str(source_dir),
        encoder="wav2vec2", backbone="cnn",
        expected_sha256=hashlib.sha256(b"checkpoint-original").hexdigest(),
        expected_commit="a" * 40,
    )

    result_metadata, loaded = estimator._load_bound_checkpoint(ModelClass)

    assert result_metadata == {"score_range": [1.0, 5.0]}
    assert loaded is not None
    assert [entry[0] for entry in seen] == ["digest", "metadata", "model"]
    assert len({entry[1] for entry in seen}) == 1
    assert all(entry[2] == b"checkpoint-original" for entry in seen)
    assert not Path(seen[0][1]).exists()
    assert checkpoint.read_bytes() == b"checkpoint-original"


def test_urgent_pk_real_estimator_result_contains_only_registered_score():
    class FakeModel:
        @staticmethod
        def model(_audios):
            return None, torch.tensor([[3.8, 4.2]], dtype=torch.float32)

    estimator = mos.URGENTPKEstimator.__new__(mos.URGENTPKEstimator)
    estimator._model = FakeModel()
    estimator._device = torch.device("cpu")
    estimator.use_gpu = False
    estimator._allow_fallback = False
    estimator.resolved_score_range = (1.0, 5.0)
    estimator._load_model = lambda: None

    assert estimator._predict_batch([np.zeros(160, dtype=np.float32)]) == [
        {"score": pytest.approx(4.0)},
    ]


def _install_tiny_nisqa(monkeypatch):
    package = types.ModuleType("nisqa")
    module = types.ModuleType("nisqa.NISQA_model")

    class TinyModel:
        @staticmethod
        def to(_device):
            return None

    class TinyNisqa:
        def __init__(self, args):
            self.args = args
            self.dev = args["tr_device"]
            self.model = TinyModel()
            torch.load(args["pretrained_model"], map_location=self.dev)

    module.nisqaModel = TinyNisqa
    package.NISQA_model = module
    monkeypatch.setitem(sys.modules, "nisqa", package)
    monkeypatch.setitem(sys.modules, "nisqa.NISQA_model", module)
    monkeypatch.setattr(mos.sf, "write", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mos, "safe_torch_load", lambda *_args, **_kwargs: {"model": {}})


def test_local_checkpoint_snapshot_cleanup_preserves_primary_and_retries(
    monkeypatch, tmp_path,
):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    real_rmtree = mos.shutil.rmtree
    failed_paths = []

    def fail_once(path, *args, **kwargs):
        if Path(path).name.startswith("audio-filter-mos-checkpoint-") and not failed_paths:
            failed_paths.append(os.fspath(path))
            raise OSError("injected checkpoint snapshot cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(mos, "_PENDING_CHECKPOINT_SNAPSHOT_CLEANUP", {})
    monkeypatch.setattr(mos.shutil, "rmtree", fail_once)

    with pytest.raises(RuntimeError, match="primary checkpoint load failure") as caught:
        with mos._open_local_checkpoint_snapshot(str(checkpoint)):
            raise RuntimeError("primary checkpoint load failure")

    assert any(
        "checkpoint snapshot cleanup failed" in note
        and "injected checkpoint snapshot cleanup failure" in note
        for note in caught.value.__notes__
    )
    assert list(mos._PENDING_CHECKPOINT_SNAPSHOT_CLEANUP) == failed_paths
    assert Path(failed_paths[0]).exists()

    mos._retry_pending_checkpoint_snapshot_cleanup()

    assert mos._PENDING_CHECKPOINT_SNAPSHOT_CLEANUP == {}
    assert not Path(failed_paths[0]).exists()


@pytest.mark.parametrize("model", ("nisqa", "utmos"))
def test_local_estimator_snapshot_cleanup_failure_is_retained_and_retryable(
    monkeypatch, tmp_path, model,
):
    checkpoint = tmp_path / f"{model}.pt"
    checkpoint.write_bytes(b"checkpoint")
    if model == "nisqa":
        _install_tiny_nisqa(monkeypatch)
    else:
        _install_tiny_utmos(monkeypatch)
        monkeypatch.setattr(
            mos, "safe_torch_load", lambda *_args, **_kwargs: {"weight": torch.ones(1)},
        )

    real_rmtree = mos.shutil.rmtree
    failed_paths = []

    def fail_once(path, *args, **kwargs):
        if Path(path).name.startswith("audio-filter-mos-checkpoint-") and not failed_paths:
            failed_paths.append(os.fspath(path))
            raise OSError(f"injected {model} snapshot cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(mos, "_PENDING_CHECKPOINT_SNAPSHOT_CLEANUP", {})
    monkeypatch.setattr(mos.shutil, "rmtree", fail_once)

    estimator_type = mos.NISQAEstimator if model == "nisqa" else mos.UTMOSEstimator
    with pytest.raises(OSError, match=f"injected {model} snapshot cleanup failure"):
        estimator_type(model_path=str(checkpoint), allow_fallback=False)

    assert list(mos._PENDING_CHECKPOINT_SNAPSHOT_CLEANUP) == failed_paths
    assert Path(failed_paths[0]).exists()
    mos._retry_pending_checkpoint_snapshot_cleanup()
    assert mos._PENDING_CHECKPOINT_SNAPSHOT_CLEANUP == {}
    assert not Path(failed_paths[0]).exists()


def test_nisqa_concurrent_compute_single_flight_owns_one_output_dir_and_close_cleans_it(
    monkeypatch, tmp_path,
):
    compute_barrier = threading.Barrier(2)
    second_output_allocation = threading.Event()
    allocation_lock = threading.Lock()
    output_dirs = []
    real_mkdtemp = mos.tempfile.mkdtemp

    class FakeNisqa:
        def __init__(self):
            self.args = {}
            self.ds_val = object()
            self.model = object()

        @staticmethod
        def _loadDatasetsFile():
            pass

        @staticmethod
        def predict():
            return _NisqaResultFrame([{"mos_pred": 4.0}])

    def synchronized_write(*_args, **_kwargs):
        compute_barrier.wait(timeout=5)

    def tracked_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        if kwargs.get("prefix") == "nisqa_out_":
            with allocation_lock:
                output_dirs.append(path)
                allocation_index = len(output_dirs)
            if allocation_index == 1:
                second_output_allocation.wait(timeout=0.2)
            else:
                second_output_allocation.set()
        return path

    monkeypatch.setattr(mos.sf, "write", synchronized_write)
    monkeypatch.setattr(mos.tempfile, "mkdtemp", tracked_mkdtemp)
    estimator = mos.NISQAEstimator.__new__(mos.NISQAEstimator)
    model = FakeNisqa()
    estimator._nisqa_model = model
    estimator._nisqa_tmpdir = None
    estimator._tmpdir_base = str(tmp_path)
    estimator.nisqa_dir = str(tmp_path)
    estimator._allow_fallback = False
    estimator.use_gpu = False
    estimator._is_fallback = False
    estimator._closed = False
    estimator._pending_tmp_paths = {}
    estimator._nisqa_cleanup_lock = threading.RLock()
    estimator._atexit_callback = None

    item = (np.zeros(8, dtype=np.float32), 16_000)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(estimator.compute, audio_data=item) for _ in range(2)]
        assert [future.result(timeout=5) for future in futures] == [
            {"nisqa_mos": 4.0}, {"nisqa_mos": 4.0},
        ]

    assert output_dirs == [estimator._nisqa_tmpdir]
    assert Path(output_dirs[0]).is_dir()

    estimator.close()

    assert not Path(output_dirs[0]).exists()
    assert estimator._nisqa_tmpdir is None
    assert estimator._nisqa_model is None
    assert model.ds_val is None
    assert model.model is None
    assert estimator._closed is True


def test_nisqa_single_input_cleanup_failure_is_retained_noted_and_retried(
    monkeypatch, tmp_path,
):
    estimator = mos.NISQAEstimator.__new__(mos.NISQAEstimator)
    estimator._nisqa_model = object()
    estimator._nisqa_tmpdir = None
    estimator._tmpdir_base = str(tmp_path)
    estimator._allow_fallback = False
    estimator.use_gpu = False
    estimator._is_fallback = False
    estimator._closed = False
    estimator._pending_tmp_paths = {}
    estimator._nisqa_cleanup_lock = threading.RLock()
    estimator._atexit_callback = None
    estimator._run_nisqa = lambda _path: (_ for _ in ()).throw(
        RuntimeError("primary NISQA inference failure")
    )
    monkeypatch.setattr(mos.sf, "write", lambda *_args, **_kwargs: None)
    real_unlink = mos.os.unlink
    failed_paths = []

    def fail_once(path, *args, **kwargs):
        if Path(path).name.startswith("nisqa_in_") and not failed_paths:
            failed_paths.append(os.fspath(path))
            raise OSError("injected NISQA unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(mos.os, "unlink", fail_once)

    with pytest.raises(RuntimeError, match="primary NISQA inference failure") as caught:
        estimator.compute(audio_data=(np.zeros(160, dtype=np.float32), 16_000))

    assert any(
        "NISQA temporary WAV cleanup failed" in note
        and "injected NISQA unlink failure" in note
        for note in caught.value.__notes__
    )
    assert estimator._pending_tmp_paths == {failed_paths[0]: False}
    assert Path(failed_paths[0]).exists()

    estimator._run_nisqa = lambda _path: {"nisqa_mos": 4.0}
    assert estimator.compute(
        audio_data=(np.zeros(160, dtype=np.float32), 16_000),
    ) == {"nisqa_mos": 4.0}
    assert estimator._pending_tmp_paths == {}
    assert not Path(failed_paths[0]).exists()


def test_nisqa_batch_cleanup_failure_is_retained_and_close_retries(
    monkeypatch, tmp_path,
):
    estimator = mos.NISQAEstimator.__new__(mos.NISQAEstimator)
    model = types.SimpleNamespace(ds_val=object(), model=object())
    estimator._nisqa_model = model
    estimator._nisqa_tmpdir = str(tmp_path / "nisqa-output")
    Path(estimator._nisqa_tmpdir).mkdir()
    estimator._tmpdir_base = str(tmp_path)
    estimator._allow_fallback = False
    estimator.use_gpu = False
    estimator._is_fallback = False
    estimator._closed = False
    estimator._pending_tmp_paths = {}
    estimator._nisqa_cleanup_lock = threading.RLock()
    estimator._atexit_callback = None
    estimator._run_nisqa_batch = lambda _paths: (_ for _ in ()).throw(
        RuntimeError("primary NISQA batch failure")
    )
    monkeypatch.setattr(mos.sf, "write", lambda *_args, **_kwargs: None)
    real_rmtree = mos.shutil.rmtree
    failed_paths = []

    def fail_once(path, *args, **kwargs):
        if Path(path).name.startswith("nisqa_batch_in_") and not failed_paths:
            failed_paths.append(os.fspath(path))
            raise OSError("injected NISQA batch rmtree failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(mos.shutil, "rmtree", fail_once)

    with pytest.raises(RuntimeError, match="primary NISQA batch failure") as caught:
        estimator.compute_batch([(np.zeros(160, dtype=np.float32), 16_000)])

    assert any(
        "NISQA batch temporary directory cleanup failed" in note
        and "injected NISQA batch rmtree failure" in note
        for note in caught.value.__notes__
    )
    assert estimator._pending_tmp_paths == {failed_paths[0]: True}
    assert Path(failed_paths[0]).exists()

    estimator.close()

    assert estimator._pending_tmp_paths == {}
    assert not Path(failed_paths[0]).exists()
    assert not Path(tmp_path / "nisqa-output").exists()
    assert estimator._nisqa_tmpdir is None
    assert estimator._nisqa_model is None
    assert model.ds_val is None
    assert model.model is None
    assert estimator._closed is True


def test_nisqa_close_stays_broken_until_all_cleanup_converges(
    monkeypatch, tmp_path,
):
    pending_file = tmp_path / "pending.wav"
    pending_file.write_bytes(b"wav")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    estimator = mos.NISQAEstimator.__new__(mos.NISQAEstimator)
    estimator._nisqa_model = types.SimpleNamespace(ds_val=object(), model=object())
    estimator._nisqa_tmpdir = str(output_dir)
    estimator._pending_tmp_paths = {str(pending_file): False}
    estimator._nisqa_cleanup_lock = threading.RLock()
    estimator._atexit_callback = None
    estimator._closed = False
    real_unlink = mos.os.unlink
    calls = []

    def fail_once(path, *args, **kwargs):
        if os.fspath(path) == str(pending_file) and not calls:
            calls.append(os.fspath(path))
            raise OSError("injected pending WAV retry failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(mos.os, "unlink", fail_once)

    with pytest.raises(OSError, match="injected pending WAV retry failure"):
        estimator.close()

    assert estimator._pending_tmp_paths == {str(pending_file): False}
    assert pending_file.exists()
    assert estimator._closed is False
    with pytest.raises(RuntimeError, match="only close retry is allowed"):
        estimator.compute(audio_path="unused.wav")

    estimator.close()

    assert estimator._pending_tmp_paths == {}
    assert not pending_file.exists()
    assert not output_dir.exists()
    assert estimator._nisqa_tmpdir is None
    assert estimator._nisqa_model is None
    assert estimator._closed is True


class _NisqaResultFrame:
    def __init__(self, rows):
        self._rows = rows
        self.iloc = self

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        return self._rows[index]

    def iterrows(self):
        return iter(enumerate(self._rows))


def _nisqa_estimator_with_result(tmp_path, rows):
    class FakeNisqa:
        args = {}

        @staticmethod
        def _loadDatasetsFile():
            pass

        @staticmethod
        def predict():
            return _NisqaResultFrame(rows)

    estimator = mos.NISQAEstimator.__new__(mos.NISQAEstimator)
    estimator._nisqa_model = FakeNisqa()
    estimator._nisqa_tmpdir = str(tmp_path)
    estimator._tmpdir_base = None
    estimator.nisqa_dir = str(tmp_path)
    estimator._allow_fallback = False
    return estimator


def test_nisqa_real_estimator_result_contains_only_registered_score(tmp_path):
    estimator = _nisqa_estimator_with_result(
        tmp_path,
        [{"mos_pred": 3.9, "noi_pred": 4.0, "dis_pred": 3.8}],
    )

    assert estimator._run_nisqa("audio.wav") == {
        "nisqa_mos": pytest.approx(3.9),
    }


@pytest.mark.parametrize(
    "row",
    [{}, {"mos_pred": True}, {"mos_pred": "3.0"},
     {"mos_pred": float("nan")}, {"mos_pred": float("inf")}],
)
def test_nisqa_rejects_missing_or_invalid_mos_pred(tmp_path, row):
    estimator = _nisqa_estimator_with_result(tmp_path, [row])

    with pytest.raises(mos.MosResultContractError, match="mos_pred"):
        estimator._run_nisqa("audio.wav")


@pytest.mark.parametrize("estimator_type", [mos.NISQAEstimator, mos.UTMOSEstimator])
def test_local_estimator_load_failure_respects_fallback_false(
    monkeypatch, tmp_path, estimator_type,
):
    missing = tmp_path / "missing-model"
    if estimator_type is mos.UTMOSEstimator:
        _install_tiny_utmos(monkeypatch)

    with pytest.raises((FileNotFoundError, RuntimeError), match="missing-model"):
        estimator_type(model_path=str(missing), allow_fallback=False)


def test_nisqa_default_model_load_failure_respects_fallback_false(
    monkeypatch, tmp_path,
):
    model_path = tmp_path / "nisqa.tar"
    model_path.write_bytes(b"placeholder")
    monkeypatch.setattr(mos.os.path, "exists", lambda path: path == str(model_path))
    monkeypatch.setattr(
        mos.os.path, "realpath",
        lambda path: str(model_path) if str(path).endswith("nisqa.tar") else str(path),
    )

    @mos.contextlib.contextmanager
    def fail_snapshot(_path):
        raise RuntimeError("default NISQA load exploded")
        yield

    monkeypatch.setattr(mos, "_open_local_checkpoint_snapshot", fail_snapshot)

    with pytest.raises(RuntimeError, match="default NISQA load exploded"):
        mos.NISQAEstimator(allow_fallback=False)


@pytest.mark.parametrize("model", ["nisqa", "utmos"])
def test_estimator_inference_failure_with_fallback_false_preserves_bounded_root_cause(
    monkeypatch, model, tmp_path,
):
    root_cause = "native inference exploded " + "x" * 2000
    if model == "nisqa":
        class FailingNisqa:
            args = {}

            @staticmethod
            def _loadDatasetsFile():
                pass

            @staticmethod
            def predict():
                raise RuntimeError(root_cause)

        estimator = mos.NISQAEstimator.__new__(mos.NISQAEstimator)
        estimator._nisqa_model = FailingNisqa()
        estimator._nisqa_tmpdir = str(tmp_path)
        estimator._tmpdir_base = None
        estimator.nisqa_dir = str(tmp_path)
        estimator._is_fallback = False
        estimator._allow_fallback = False
        estimator.use_gpu = False
        monkeypatch.setattr(
            mos,
            "sf",
            type("FakeSoundFile", (), {
                "read": staticmethod(lambda *_args, **_kwargs: (
                    np.zeros((160, 2), dtype=np.float32), 16_000,
                )),
                "write": staticmethod(lambda *_args, **_kwargs: None),
            }),
        )
    else:
        class FailingModel:
            def __call__(self, _waveform, _sr):
                raise RuntimeError(root_cause)

        estimator = mos.UTMOSEstimator.__new__(mos.UTMOSEstimator)
        estimator.model = FailingModel()
        estimator._device = torch.device("cpu")
        estimator._is_fallback = False
        estimator._allow_fallback = False
        estimator.use_gpu = False
        estimator._load_waveform = lambda *_args: (torch.zeros(1, 160), 16_000)

    with pytest.raises(RuntimeError) as raised:
        estimator.compute(audio_path="audio.wav")

    message = str(raised.value)
    assert "native inference exploded" in message
    assert len(message) <= 600


@pytest.mark.parametrize(
    ("raw", "models", "expected"),
    [
        ({"nisqa": {"nisqa_mos": 5.01}}, ("nisqa",), 5.01),
        ({"urgent_pk": {"score": 0.5}}, ("urgent_pk",), 0.5),
        ({"utmos": {"utmos_score": 6.0}}, ("utmos",), 6.0),
        ({"nisqa": {"nisqa_mos": 0.8177040118447554}}, ("nisqa",), 0.8177040118447554),
    ],
)
def test_artifact_mapping_preserves_out_of_scale_model_scores(raw, models, expected):
    # artifact 记录模型真实输出（DNSMOS 多项式外推可合法超出 [1, 5] 量表），
    # keep/reject 由下游选择阈值决定，本层不得钳位或拦截有限分数。
    assert mos.map_mos_raw_result(raw, models) == {models[0]: expected}


@pytest.mark.parametrize("raw_score", [float("nan"), float("inf"), float("-inf")])
def test_artifact_mapping_rejects_nonfinite_mos_scores(raw_score):
    raw = {"nisqa": {"nisqa_mos": raw_score}}

    with pytest.raises(ValueError, match="finite"):
        mos.map_mos_raw_result(raw, ("nisqa",))


def test_utmos_real_score_is_not_silently_clamped():
    class OutOfRangeModel:
        @staticmethod
        def __call__(_waveform, _sr):
            return torch.tensor([5.2])

    estimator = mos.UTMOSEstimator.__new__(mos.UTMOSEstimator)
    estimator.model = OutOfRangeModel()
    estimator._device = torch.device("cpu")
    estimator._is_fallback = False
    estimator._allow_fallback = False
    estimator.use_gpu = False
    estimator._load_waveform = lambda *_args: (torch.zeros(1, 160), 16_000)

    assert estimator.compute(audio_path="audio.wav") == {
        "utmos_score": pytest.approx(5.2),
    }


def test_nisqa_batch_rejects_missing_mos_pred_column(tmp_path):
    estimator = _nisqa_estimator_with_result(tmp_path, [])
    estimator._run_nisqa_batch = lambda _paths: _NisqaResultFrame([{}])

    with pytest.raises(mos.MosResultContractError, match="mos_pred"):
        estimator.compute_batch([
            (np.zeros(160, dtype=np.float32), 16_000),
        ])


def test_strict_api_error_mapping_preserves_bounded_root_cause(monkeypatch):
    root_cause = "native loader exploded " + "x" * 2000
    monkeypatch.setattr(
        mos,
        "get_mos_estimator",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(root_cause)),
    )

    single = mos.compute_mos_scores(
        mos_types=["nisqa"], allow_fallback=False,
        audio_data=(np.zeros(160, dtype=np.float32), 16_000),
    )["nisqa"]["error"]
    batch = mos.compute_mos_scores_batch(
        [(np.zeros(160, dtype=np.float32), 16_000)],
        ["nisqa"],
        allow_fallback=False,
    )[0]["nisqa"]["error"]

    for error in (single, batch):
        assert error.startswith("RuntimeError: native loader exploded")
        assert len(error) <= 384


def test_clear_model_cache_retries_cleanup_created_while_waiting_for_lease(
    monkeypatch,
):
    compute_started = threading.Event()
    release_compute = threading.Event()

    class Estimator:
        use_gpu = False

        def __init__(self):
            self.close_calls = 0

        def compute(self, *, audio_data):
            compute_started.set()
            assert release_compute.wait(timeout=5)
            return {"value": float(len(audio_data[0]))}

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("injected lease cleanup failure")

    estimator = Estimator()
    cleanup_entered = threading.Event()
    real_retire = mos._retire_estimators

    def retire_with_signal(estimators, *, wait_for_leases):
        if wait_for_leases:
            cleanup_entered.set()
        return real_retire(estimators, wait_for_leases=wait_for_leases)

    monkeypatch.setattr(mos, "_retire_estimators", retire_with_signal)
    monkeypatch.setattr(mos, "_MODEL_CACHE", {"fake_False_True": estimator})
    monkeypatch.setattr(mos, "_RETIRED_ESTIMATORS", {})
    monkeypatch.setattr(mos, "_PENDING_ESTIMATOR_CLEANUP", {})
    monkeypatch.setattr(
        mos, "clear_urgent_pk_checkpoint_metadata_cache", lambda: None,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        compute_future = executor.submit(
            mos.compute_mos_scores,
            mos_types=["fake"],
            audio_data=(np.zeros(8, dtype=np.float32), 16_000),
        )
        assert compute_started.wait(timeout=5)
        clear_future = executor.submit(mos.clear_model_cache)
        # 等清理路径真正进入租约等待后再断言，固定 sleep 会让未调度的竞争线程假通过。
        assert cleanup_entered.wait(timeout=5)
        assert not clear_future.done()
        release_compute.set()
        with pytest.raises(RuntimeError, match="injected lease cleanup failure"):
            compute_future.result(timeout=5)
        clear_future.result(timeout=5)

    assert estimator.close_calls == 2
    assert mos._PENDING_ESTIMATOR_CLEANUP == {}


def test_nisqa_in_memory_temp_file_is_closed_and_removed_when_write_fails(
    monkeypatch, tmp_path,
):
    temp_path = tmp_path / "nisqa_in.wav"
    temp_path.write_bytes(b"")

    class TempFile:
        name = str(temp_path)
        closed = False

        def close(self):
            self.closed = True

    temp_file = TempFile()
    monkeypatch.setattr(
        mos.tempfile, "NamedTemporaryFile", lambda **_kwargs: temp_file,
    )

    def fail_write(path, _audio, _sr):
        assert path == str(temp_path)
        assert temp_file.closed is True
        raise RuntimeError("injected sf.write failure")

    monkeypatch.setattr(mos.sf, "write", fail_write)
    estimator = mos.NISQAEstimator.__new__(mos.NISQAEstimator)
    estimator._tmpdir_base = None

    with pytest.raises(RuntimeError, match="injected sf.write failure"):
        estimator.compute(
            audio_data=(np.zeros(8, dtype=np.float32), 16_000),
        )

    assert temp_file.closed is True
    assert not temp_path.exists()


def test_nisqa_single_temp_unlink_failure_is_fail_closed_and_notes_primary(
    monkeypatch, tmp_path,
):
    created = []

    class TempFile:
        def __init__(self, path):
            self.name = str(path)

        @staticmethod
        def close():
            pass

    def create_temp_file(**_kwargs):
        path = tmp_path / f"nisqa_in_{len(created)}.wav"
        path.write_bytes(b"")
        created.append(path)
        return TempFile(path)

    monkeypatch.setattr(mos.tempfile, "NamedTemporaryFile", create_temp_file)
    monkeypatch.setattr(mos.sf, "write", lambda *_args, **_kwargs: None)
    original_unlink = mos.os.unlink
    unlink_calls = []

    def fail_selected_unlink(path):
        unlink_calls.append(path)
        if len(unlink_calls) in {1, 4}:
            raise OSError("injected NISQA input unlink failure")
        original_unlink(path)

    monkeypatch.setattr(mos.os, "unlink", fail_selected_unlink)
    estimator = mos.NISQAEstimator.__new__(mos.NISQAEstimator)
    estimator._tmpdir_base = None
    estimator._allow_fallback = False
    estimator._is_fallback = False
    estimator.use_gpu = False
    estimator._run_nisqa = lambda _path: {"nisqa_mos": 4.0}
    item = (np.zeros(8, dtype=np.float32), 16_000)

    try:
        with pytest.raises(OSError, match="injected NISQA input unlink failure"):
            estimator.compute(audio_data=item)
        assert created[0].exists()

        assert estimator.compute(audio_data=item) == {"nisqa_mos": 4.0}
        assert not created[1].exists()

        estimator._run_nisqa = lambda _path: (_ for _ in ()).throw(
            ValueError("injected NISQA business failure")
        )
        with pytest.raises(
            ValueError, match="injected NISQA business failure",
        ) as caught:
            estimator.compute(audio_data=item)

        assert any(
            "temporary WAV cleanup failed: "
            "injected NISQA input unlink failure" in note
            for note in caught.value.__notes__
        )
    finally:
        for path in created:
            if path.exists():
                original_unlink(path)


def test_nisqa_batch_temp_rmtree_failure_is_fail_closed_and_notes_primary(
    monkeypatch, tmp_path,
):
    estimator = mos.NISQAEstimator.__new__(mos.NISQAEstimator)
    estimator._nisqa_model = object()
    estimator._tmpdir_base = str(tmp_path)
    estimator._allow_fallback = False
    estimator._run_nisqa_batch = lambda _paths: _NisqaResultFrame([
        {"mos_pred": 4.0},
    ])
    monkeypatch.setattr(mos.sf, "write", lambda *_args, **_kwargs: None)
    original_rmtree = mos.shutil.rmtree
    created = []
    original_mkdtemp = mos.tempfile.mkdtemp

    def tracked_mkdtemp(**kwargs):
        path = original_mkdtemp(**kwargs)
        created.append(path)
        return path

    monkeypatch.setattr(mos.tempfile, "mkdtemp", tracked_mkdtemp)
    rmtree_calls = []

    def fail_selected_rmtree(path):
        rmtree_calls.append(path)
        if len(rmtree_calls) in {1, 4}:
            raise OSError("injected NISQA batch rmtree failure")
        original_rmtree(path)

    monkeypatch.setattr(mos.shutil, "rmtree", fail_selected_rmtree)
    item = (np.zeros(8, dtype=np.float32), 16_000)

    try:
        with pytest.raises(OSError, match="injected NISQA batch rmtree failure"):
            estimator.compute_batch([item])
        assert Path(created[0]).exists()

        assert estimator.compute_batch([item]) == [{"nisqa_mos": 4.0}]
        assert not Path(created[1]).exists()

        estimator._run_nisqa_batch = lambda _paths: (_ for _ in ()).throw(
            ValueError("injected NISQA batch business failure")
        )
        with pytest.raises(
            RuntimeError, match="injected NISQA batch business failure",
        ) as caught:
            estimator.compute_batch([item])

        assert any(
            "batch temporary directory cleanup failed: "
            "injected NISQA batch rmtree failure" in note
            for note in caught.value.__notes__
        )
    finally:
        for path in created:
            if Path(path).exists():
                original_rmtree(path)


def test_nisqa_close_failure_keeps_retry_path_and_breaks_lifecycle(
    monkeypatch, tmp_path,
):
    output_dir = tmp_path / "nisqa_out_retry"
    output_dir.mkdir()
    callback = object()
    unregister_calls = []
    estimator = mos.NISQAEstimator.__new__(mos.NISQAEstimator)
    estimator._nisqa_tmpdir = str(output_dir)
    estimator._atexit_callback = callback
    monkeypatch.setattr(mos, "_NISQA_ATEXIT_CALLBACKS", [callback])
    monkeypatch.setattr(
        mos.atexit, "unregister", lambda value: unregister_calls.append(value),
    )
    original_rmtree = mos.shutil.rmtree
    rmtree_calls = []

    def fail_once_rmtree(path):
        rmtree_calls.append(path)
        if len(rmtree_calls) == 1:
            raise OSError("injected NISQA close rmtree failure")
        original_rmtree(path)

    monkeypatch.setattr(mos.shutil, "rmtree", fail_once_rmtree)

    with pytest.raises(OSError, match="injected NISQA close rmtree failure"):
        estimator.close()

    assert estimator._nisqa_tmpdir == str(output_dir)
    assert output_dir.exists()
    assert estimator._atexit_callback is callback
    assert mos._NISQA_ATEXIT_CALLBACKS == [callback]
    assert mos._estimator_lifecycle_state(estimator).phase == "broken"
    with pytest.raises(RuntimeError, match="only close retry is allowed"):
        estimator.compute(audio_path="unused.wav")

    estimator.close()

    assert rmtree_calls == [str(output_dir), str(output_dir)]
    assert not output_dir.exists()
    assert estimator._nisqa_tmpdir is None
    assert estimator._atexit_callback is None
    assert unregister_calls == [callback]
    assert mos._NISQA_ATEXIT_CALLBACKS == []
    assert mos._estimator_lifecycle_state(estimator).phase == "closed"


@pytest.mark.parametrize("failure_stage", ("inference", "prefetch"))
def test_overlap_resample_executor_always_shuts_down_without_masking_primary(
    monkeypatch, failure_stage,
):
    shutdown_calls = []

    class Future:
        def __init__(self, result=None, error=None):
            self._result = result
            self._error = error

        def result(self):
            if self._error is not None:
                raise self._error
            return self._result

    class Executor:
        def __init__(self, *, max_workers):
            assert max_workers == 1

        def submit(self, function, *args):
            try:
                result = function(*args)
            except BaseException as exc:
                return Future(error=exc)
            return Future(result=result)

        def shutdown(self, *, wait):
            shutdown_calls.append(wait)
            if failure_stage == "inference":
                raise RuntimeError("injected executor shutdown failure")

    class Estimator:
        def __init__(self, invalid_count=False):
            self.invalid_count = invalid_count

        def compute_batch(self, batch):
            if self.invalid_count:
                return []
            return [{"MOS": 4.0} for _item in batch]

    estimators = {
        "scoreq": Estimator(invalid_count=failure_stage == "inference"),
        "nisqa": Estimator(),
    }

    def prepare(batch, target_sr, _method):
        if failure_stage == "prefetch" and target_sr == 48_000:
            raise RuntimeError("injected prefetch failure")
        return list(batch)

    monkeypatch.setattr(mos, "ThreadPoolExecutor", Executor)
    monkeypatch.setattr(mos, "get_mos_estimator", lambda model, **_kwargs: estimators[model])
    monkeypatch.setattr(mos, "_prepare_batch_items_for_model", prepare)

    expected = (
        mos.MosResultContractError if failure_stage == "inference" else RuntimeError
    )
    message = (
        "compute_batch returned" if failure_stage == "inference"
        else "injected prefetch failure"
    )
    with pytest.raises(expected, match=message) as caught:
        mos.compute_mos_scores_batch(
            [(np.zeros(8, dtype=np.float32), 16_000)],
            ["scoreq", "nisqa"],
            use_gpu=True,
            use_cache=True,
            model_specific_sr=True,
            overlap_resample=True,
            allow_fallback=False,
        )

    assert shutdown_calls == [True]
    if failure_stage == "inference":
        assert any(
            "executor shutdown failed: injected executor shutdown failure" in note
            for note in caught.value.__notes__
        )


def _runtime_attestation_estimator(model):
    if model == "scoreq":
        estimator = mos.SCOREQEstimator.__new__(mos.SCOREQEstimator)
        estimator.use_gpu = True
        estimator._is_fallback = False
        estimator._scoreq = types.SimpleNamespace(
            _sess=_FakeOnnxSession([
                "CUDAExecutionProvider", "CPUExecutionProvider",
            ]),
            _device_str="GPU",
        )
        return estimator, {
            "execution_mode": "cuda_primary_hybrid_ep",
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "device": "gpu",
        }
    if model == "dnsmos":
        estimator = mos.DNSMOSEstimator.__new__(mos.DNSMOSEstimator)
        estimator.use_gpu = True
        estimator._is_fallback = False
        estimator.compute_score = types.SimpleNamespace(
            onnx_sess=_FakeOnnxSession(["CUDAExecutionProvider"]),
            p808_onnx_sess=_FakeOnnxSession(["CUDAExecutionProvider"]),
            _cpu_ep_fallback_disabled=False,
        )
        return estimator, {
            "onnx_sess": ["CUDAExecutionProvider"],
            "p808_onnx_sess": ["CUDAExecutionProvider"],
            "cpu_ep_fallback_disabled": False,
        }
    if model == "nisqa":
        estimator = mos.NISQAEstimator.__new__(mos.NISQAEstimator)
        estimator.use_gpu = True
        estimator._is_fallback = False
        estimator._expected_checkpoint_sha256 = "a" * 64
        estimator._checkpoint_sha256 = "a" * 64
        estimator._nisqa_model = types.SimpleNamespace(
            dev="cuda", model=_FakeParameters("cuda"),
        )
        return estimator, {
            "device": "cuda",
            "parameters": "cuda",
            "checkpoint_sha256": "a" * 64,
        }
    if model == "utmos":
        estimator = mos.UTMOSEstimator.__new__(mos.UTMOSEstimator)
        estimator.use_gpu = True
        estimator._is_fallback = False
        estimator._expected_checkpoint_sha256 = "b" * 64
        estimator._checkpoint_sha256 = "b" * 64
        estimator._device = "cuda"
        estimator.model = _FakeParameters("cuda")
        return estimator, {
            "device": "cuda",
            "parameters": "cuda",
            "checkpoint_sha256": "b" * 64,
        }
    estimator = mos.URGENTPKEstimator.__new__(mos.URGENTPKEstimator)
    estimator.use_gpu = True
    estimator._is_fallback = False
    estimator._device = "cuda"
    estimator._model = _FakeParameters("cuda")
    estimator._load_model = lambda: None
    estimator.checkpoint = "urgent.ckpt"
    estimator.source_dir = "urgent-source"
    estimator.expected_sha256 = "c" * 64
    estimator.expected_commit = "d" * 40
    return estimator, {"device": "cuda", "parameters": "cuda"}


@pytest.mark.parametrize("model", _FIVE_MODELS)
def test_all_estimator_wrappers_expose_existing_runtime_attestation_wire_payload(
    model, monkeypatch,
):
    estimator, expected = _runtime_attestation_estimator(model)
    if model == "urgent_pk":
        assets = {
            "checkpoint_sha256": estimator.expected_sha256,
            "source_commit": estimator.expected_commit,
            "source_package_sha256": "e" * 64,
        }
        monkeypatch.setattr(
            mos, "verify_urgent_pk_trusted_assets", lambda *_args: assets,
        )
        monkeypatch.setattr(mos, "urgent_pk_verified_assets", lambda *_args: assets)

    assert estimator.runtime_attestation() == expected


def test_runtime_attestation_is_exclusive_with_active_compute():
    compute_started = threading.Event()
    release_compute = threading.Event()

    class BlockingScoreq:
        _sess = _FakeOnnxSession(["CPUExecutionProvider"])
        _device_str = "CPU"

        def __call__(self, _audio, _sr):
            compute_started.set()
            assert release_compute.wait(timeout=5)
            return {"MOS": 4.0}

    estimator = mos.SCOREQEstimator.__new__(mos.SCOREQEstimator)
    estimator.use_gpu = False
    estimator._is_fallback = False
    estimator._allow_fallback = False
    estimator._scoreq = BlockingScoreq()
    attestation_entered = threading.Event()

    def attest_and_signal():
        attestation_entered.set()
        return estimator.runtime_attestation()

    with ThreadPoolExecutor(max_workers=2) as executor:
        compute_future = executor.submit(
            estimator.compute,
            audio_data=(np.zeros(8, dtype=np.float32), 16_000),
        )
        assert compute_started.wait(timeout=5)
        attestation_future = executor.submit(attest_and_signal)
        # 等 attestation 真正进入独占等待后再断言，固定 sleep 会让未调度的竞争线程假通过。
        assert attestation_entered.wait(timeout=5)
        assert not attestation_future.done()
        release_compute.set()
        assert compute_future.result(timeout=5) == {"scoreq_mos": 4.0}
        assert attestation_future.result(timeout=5) == {
            "execution_mode": "cpu_ep",
            "providers": ["CPUExecutionProvider"],
            "device": "cpu",
        }
