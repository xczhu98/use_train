"""Unit tests for bounded real-model MOS batching benchmark helpers."""

import json

import numpy as np
import pytest

from tools.verify import benchmark_mos_batching as benchmark

pytestmark = pytest.mark.unit


def _item(index: int, samples: int, sample_rate: int = 16_000) -> benchmark.AudioItem:
    return benchmark.AudioItem(
        path=f"sample-{index}.wav",
        waveform=np.arange(samples, dtype=np.float32),
        sample_rate=sample_rate,
        original_samples=samples,
        source_capped=False,
    )


def test_first_channel_is_contiguous_float32_and_rejects_invalid_inputs():
    stereo = np.array([[1.0, 99.0], [2.0, 98.0]], dtype=np.float64)
    mono = benchmark._first_channel_float32(stereo)

    assert mono.dtype == np.float32
    assert mono.flags.c_contiguous
    np.testing.assert_array_equal(mono, [1.0, 2.0])
    with pytest.raises(ValueError, match="at least one sample"):
        benchmark._first_channel_float32(np.array([], dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        benchmark._first_channel_float32(np.array([np.nan], dtype=np.float32))


def test_preprocessing_lengths_report_exact_buckets_and_scoreq_cap():
    short = _item(0, 80)
    long = _item(1, benchmark._SCOREQ_MAX_SAMPLES + 1)

    utmos = benchmark.preprocessing_report("utmos", [short, long])
    scoreq = benchmark.preprocessing_report("scoreq", [short, long])

    assert [record["model_preprocessed_samples"] for record in utmos] == [16_000, 160_001]
    assert scoreq[0]["model_preprocessed_samples"] == 320
    assert scoreq[1]["post_resample_samples"] == 160_001
    assert scoreq[1]["model_preprocessed_samples"] == 160_000
    assert scoreq[1]["ten_second_capped"] is True
    assert benchmark.batch_histogram(scoreq) == [
        {"preprocessed_samples": 320, "count": 1},
        {"preprocessed_samples": 160_000, "count": 1},
    ]


def test_derived_chunks_are_explicitly_bounded_by_exact_length_buckets():
    records = [
        {"model_preprocessed_samples": 320},
        {"model_preprocessed_samples": 320},
        {"model_preprocessed_samples": 640},
    ]

    assert benchmark.derived_expected_chunks("utmos", records, batch_size=2, use_gpu=False) == 2
    assert benchmark.derived_expected_chunks("scoreq", records, batch_size=2, use_gpu=False) == 2


def test_compare_scores_preserves_original_positions_and_serializes_without_nan():
    comparison = benchmark.compare_scores(
        "scoreq",
        [{"scoreq_mos": 3.0}, {"scoreq_mos": 2.5}],
        [{"scoreq_mos": 3.0}, {"scoreq_mos": 2.500001}],
        tolerance=1e-5,
    )

    assert [entry["index"] for entry in comparison["items"]] == [0, 1]
    assert comparison["all_within_tolerance"] is True
    assert comparison["max_absolute_error"] == pytest.approx(1e-6)
    assert json.loads(json.dumps(comparison, allow_nan=False)) == comparison


def test_bounded_batch_helper_checks_counts_and_keeps_chunk_order():
    class Estimator:
        def compute_batch(self, items):
            return [{"utmos_score": item[0][0]} for item in items]

    inputs = [
        (np.array([float(index)], dtype=np.float32), 16_000)
        for index in range(5)
    ]
    results = benchmark._compute_in_bounded_batches(Estimator(), inputs, batch_size=2)

    assert [result["utmos_score"] for result in results] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_model_selection_and_unavailable_telemetry_fail_closed():
    assert benchmark.selected_models(None) == ("utmos", "scoreq")
    with pytest.raises(ValueError, match="duplicates"):
        benchmark.selected_models(("scoreq", "scoreq"))

    telemetry = benchmark._session_telemetry("utmos", object(), {})
    assert telemetry["actual_onnx_run_count"] is None
    assert telemetry["actual_session_build_count"] is None
    assert "no ONNX session telemetry" in telemetry["unavailable_reason"]


def test_model_failure_report_is_json_safe_and_never_claims_a_comparison():
    report = benchmark.model_failure_report(
        "scoreq",
        [_item(0, 160)],
        batch_size=1,
        use_gpu=False,
        error=ValueError("asset digest mismatch"),
    )

    assert report["status"] == "failed"
    assert report["comparison"] is None
    assert report["error"] == {
        "type": "ValueError",
        "message": "asset digest mismatch",
    }
    assert json.loads(json.dumps(report, allow_nan=False)) == report
