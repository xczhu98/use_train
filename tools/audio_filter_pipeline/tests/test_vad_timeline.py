"""Silero VAD 概率序列到不可变语音时间线的纯后处理测试。"""

import json
import math

import numpy as np
import pytest

from audio_filter_pipeline.processors.vad_filter import SpeechInterval, speech_timeline

pytestmark = pytest.mark.unit

FRAME = 512
SR = 16000


def _timeline(probabilities, *, total_samples=None, **kwargs):
    probs = np.asarray(probabilities, dtype=np.float32)
    if total_samples is None:
        total_samples = len(probs) * FRAME
    return speech_timeline(probs, total_samples=total_samples, **kwargs)


def test_empty_probabilities_return_empty_tuple():
    result = speech_timeline(np.array([], dtype=np.float32), total_samples=0)

    assert result == ()
    assert isinstance(result, tuple)


def test_threshold_equality_counts_as_speech():
    result = _timeline(
        [0.49, 0.5, 0.75],
        min_speech_ms=0,
        max_silence_gap_ms=0,
        padding_ms=0,
    )

    assert result == (SpeechInterval(512, 1536, pytest.approx(0.625)),)


def test_default_minimum_speech_drops_three_frames_and_keeps_four():
    three_frames = _timeline([0.9, 0.9, 0.9], padding_ms=0)
    four_frames = _timeline([0.9, 0.9, 0.9, 0.9], padding_ms=0)

    assert three_frames == ()
    assert four_frames == (SpeechInterval(0, 2048, pytest.approx(0.9)),)


def test_minimum_speech_is_applied_before_gap_merge():
    result = _timeline(
        [0.9] * 3 + [0.0] + [0.8] * 3,
        padding_ms=0,
    )

    assert result == ()


def test_five_frame_silence_gap_merges_without_affecting_mean_probability():
    result = _timeline(
        [0.8] * 4 + [0.1] * 5 + [0.6] * 4,
        padding_ms=0,
    )

    assert result == (SpeechInterval(0, 6656, pytest.approx(0.7)),)


def test_seven_frame_silence_gap_remains_after_padding():
    result = _timeline([0.9] * 4 + [0.0] * 7 + [0.7] * 4)

    assert result == (
        SpeechInterval(0, 3648, pytest.approx(0.9)),
        SpeechInterval(4032, 7680, pytest.approx(0.7)),
    )
    assert result[0].end_sample < result[1].start_sample


def test_padding_clamps_to_zero_and_non_frame_aligned_total_samples():
    probs = np.array([0.8] * 4 + [0.0] * 8 + [0.6] * 4, dtype=np.float32)

    result = speech_timeline(probs, total_samples=7900, max_silence_gap_ms=0)

    assert result == (
        SpeechInterval(0, 3648, pytest.approx(0.8)),
        SpeechInterval(4544, 7900, pytest.approx(0.6)),
    )


def test_padding_union_produces_ordered_non_overlapping_intervals():
    result = _timeline(
        [0.9] * 4 + [0.0] * 7 + [0.7] * 4 + [0.0] * 9 + [0.6] * 4,
        max_silence_gap_ms=0,
        padding_ms=120,
    )

    assert result == (
        SpeechInterval(0, 9600, pytest.approx(0.8)),
        SpeechInterval(10368, 14336, pytest.approx(0.6)),
    )
    assert all(
        left.end_sample < right.start_sample
        for left, right in zip(result, result[1:])
    )


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.5],
        np.array([[0.5]], dtype=np.float32),
        np.array([1], dtype=np.int64),
        np.array([True], dtype=np.bool_),
        np.array([0.5 + 0j], dtype=np.complex64),
        np.array([0.5], dtype=object),
    ],
)
def test_invalid_probability_container_shape_or_dtype_is_rejected(probabilities):
    with pytest.raises((TypeError, ValueError)):
        speech_timeline(probabilities, total_samples=0)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -0.01, 1.01])
def test_non_finite_or_out_of_range_probability_is_rejected(value):
    with pytest.raises(ValueError):
        speech_timeline(np.array([value], dtype=np.float64), total_samples=0)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("total_samples", True),
        ("total_samples", -1),
        ("total_samples", 1.5),
        ("sample_rate", True),
        ("sample_rate", 0),
        ("sample_rate", 16000.0),
        ("frame_samples", True),
        ("frame_samples", 0),
        ("frame_samples", 512.0),
        ("speech_threshold", np.nan),
        ("speech_threshold", np.inf),
        ("speech_threshold", -0.01),
        ("speech_threshold", 1.01),
        ("speech_threshold", True),
        ("min_speech_ms", True),
        ("min_speech_ms", -1),
        ("min_speech_ms", 100.0),
        ("max_silence_gap_ms", True),
        ("max_silence_gap_ms", -1),
        ("max_silence_gap_ms", 200.0),
        ("padding_ms", True),
        ("padding_ms", -1),
        ("padding_ms", 100.0),
    ],
)
def test_invalid_scalar_parameter_is_rejected(parameter, value):
    kwargs = {"total_samples": 0, parameter: value}

    with pytest.raises((TypeError, ValueError)):
        speech_timeline(np.array([0.5], dtype=np.float32), **kwargs)


def test_total_samples_cannot_exceed_probability_frame_coverage():
    with pytest.raises(ValueError):
        speech_timeline(np.array([0.5], dtype=np.float32), total_samples=513)


def test_empty_probabilities_reject_nonzero_total_samples():
    with pytest.raises(ValueError):
        speech_timeline(np.array([], dtype=np.float32), total_samples=1)


def test_short_audio_may_use_a_zero_padded_final_frame():
    result = speech_timeline(
        np.array([0.75], dtype=np.float32),
        total_samples=123,
        min_speech_ms=0,
        padding_ms=100,
    )

    assert result == (SpeechInterval(0, 123, pytest.approx(0.75)),)


def test_input_probability_array_is_not_modified():
    probabilities = np.array([0.8] * 4 + [0.1] * 5 + [0.6] * 4, dtype=np.float32)
    original = probabilities.copy()

    result = speech_timeline(probabilities, total_samples=len(probabilities) * FRAME)

    assert result == (SpeechInterval(0, 6656, pytest.approx(0.7)),)
    assert np.array_equal(probabilities, original)


def test_minimum_speech_equal_to_sample_floor_is_retained():
    result = speech_timeline(
        np.array([0.8], dtype=np.float32),
        total_samples=100,
        sample_rate=1000,
        frame_samples=100,
        min_speech_ms=100,
        max_silence_gap_ms=0,
        padding_ms=0,
    )

    assert result == (SpeechInterval(0, 100, pytest.approx(0.8)),)


def test_gap_equal_to_sample_floor_merges_non_equal_runs_by_frame_count():
    result = speech_timeline(
        np.array([0.9, 0.0, 0.0, 0.6, 0.6, 0.6], dtype=np.float32),
        total_samples=600,
        sample_rate=1000,
        frame_samples=100,
        min_speech_ms=100,
        max_silence_gap_ms=200,
        padding_ms=0,
    )

    assert result == (SpeechInterval(0, 600, pytest.approx(0.675)),)


def test_padding_touch_unions_non_equal_runs_by_frame_count():
    result = speech_timeline(
        np.array([0.9, 0.0, 0.0, 0.6, 0.6, 0.6], dtype=np.float32),
        total_samples=600,
        sample_rate=1000,
        frame_samples=100,
        min_speech_ms=100,
        max_silence_gap_ms=0,
        padding_ms=100,
    )

    assert result == (SpeechInterval(0, 600, pytest.approx(0.675)),)


def test_numpy_integer_parameters_use_python_integer_arithmetic():
    result = speech_timeline(
        np.array([0.9, 0.9, 0.9], dtype=np.float32),
        total_samples=np.int32(1536),
        sample_rate=np.int16(16000),
        frame_samples=np.int16(512),
        min_speech_ms=np.int16(100),
        max_silence_gap_ms=np.int16(200),
        padding_ms=np.int16(100),
    )

    assert result == ()


@pytest.mark.parametrize("integer_type", [np.int16, np.int32, np.int64])
@pytest.mark.parametrize("threshold", [np.float32(0.5), np.float64(0.5)])
def test_numpy_scalar_parameters_emit_finite_json_safe_python_fields(
    integer_type, threshold
):
    result = speech_timeline(
        np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
        total_samples=integer_type(2048),
        sample_rate=integer_type(16000),
        frame_samples=integer_type(512),
        speech_threshold=threshold,
        min_speech_ms=integer_type(100),
        max_silence_gap_ms=integer_type(0),
        padding_ms=integer_type(0),
    )

    assert result == (SpeechInterval(0, 2048, pytest.approx(0.65)),)
    interval = result[0]
    assert type(interval.start_sample) is int
    assert type(interval.end_sample) is int
    assert type(interval.mean_probability) is float
    assert math.isfinite(interval.mean_probability)
    assert json.dumps(
        {
            "start_sample": interval.start_sample,
            "end_sample": interval.end_sample,
            "mean_probability": interval.mean_probability,
        }
    )


@pytest.mark.parametrize(
    "parameter",
    [
        "total_samples",
        "sample_rate",
        "frame_samples",
        "speech_threshold",
        "min_speech_ms",
        "max_silence_gap_ms",
        "padding_ms",
    ],
)
def test_numpy_bool_scalar_parameters_are_rejected(parameter):
    kwargs = {"total_samples": 0, parameter: np.bool_(False)}

    with pytest.raises((TypeError, ValueError)):
        speech_timeline(np.array([0.5], dtype=np.float32), **kwargs)


def test_clamped_run_mean_includes_full_partial_and_out_of_domain_tail_frames():
    result = speech_timeline(
        np.array([0.6, 0.8, 1.0], dtype=np.float32),
        total_samples=150,
        sample_rate=1000,
        frame_samples=100,
        min_speech_ms=100,
        max_silence_gap_ms=0,
        padding_ms=0,
    )

    assert result == (SpeechInterval(0, 150, pytest.approx(0.8)),)


def test_nonempty_probabilities_with_zero_total_samples_return_empty_tuple():
    result = speech_timeline(np.array([0.9], dtype=np.float32), total_samples=0)

    assert result == ()
