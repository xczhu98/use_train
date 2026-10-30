"""Step1 纯分析事实映射契约测试。"""

import json
import random
from dataclasses import replace

import numpy as np
import pytest

import audio_filter_pipeline.processors.segmentation as segmentation_module
import audio_filter_pipeline.processors.stages_v3.analyze as analyze_module
import audio_filter_pipeline.processors.vad_filter as vad_module
import audio_filter_pipeline.utils.resample as resample_module
from audio_filter_pipeline.processors.segmentation import (
    STEP1_STABLE_SILENCE_PARAMETERS,
    StableSilenceParameters,
    detect_stable_silence_intervals,
)
from audio_filter_pipeline.processors.signal_metrics import SignalMetrics
from audio_filter_pipeline.processors.stages_v3.analyze import (
    STEP1_ANALYSIS_FIELDS,
    Step1AnalyzeStage,
    analyze_decoded_source,
)
from audio_filter_pipeline.processors.stages_v3.source_io import DecodedSource
from audio_filter_pipeline.processors.vad_filter import SpeechInterval, VadAnalysis
from audio_filter_pipeline.utils.bandwidth import estimate_audio_bandwidth
from audio_filter_pipeline.utils.resample import resample_audio


pytestmark = pytest.mark.unit


def _decoded(audio: np.ndarray, sample_rate: int = 16000, **changes) -> DecodedSource:
    values = {
        "audio": audio,
        "sample_rate": sample_rate,
        "frames": audio.shape[0] if audio.ndim >= 1 else 1,
        "channels": 1,
        "format": "WAV",
        "subtype": "FLOAT",
        "source_size_bytes": 123,
        "source_sha256": "a" * 64,
        "vad_capacity_samples": max(1, 2 * audio.shape[0] + 1),
    }
    values.update(changes)
    return DecodedSource(**values)


class FakeVad:
    def __init__(self, result: VadAnalysis):
        self.result = result
        self.calls = []

    def analyze(self, audio: np.ndarray, sample_rate: int) -> VadAnalysis:
        self.calls.append((audio, sample_rate, audio.copy()))
        return self.result


def _vad_result(
    num_samples: int,
    *,
    speech_ratio: float = 0.5,
    sample_rate: int = 16000,
    timeline: tuple[SpeechInterval, ...] | None = None,
) -> VadAnalysis:
    if timeline is None:
        timeline = (SpeechInterval(0, num_samples, 0.75),)
    return VadAnalysis(speech_ratio, timeline, sample_rate, num_samples)


def test_mono_emits_complete_ordered_json_safe_python_facts():
    frames = 4096
    time = np.arange(frames, dtype=np.float32) / 16000
    audio = np.ascontiguousarray(0.25 * np.sin(2 * np.pi * 440 * time), dtype=np.float32)
    vad = FakeVad(
        _vad_result(
            frames,
            speech_ratio=0.123456789,
            timeline=(SpeechInterval(10, 100, 0.987654321),),
        )
    )

    facts = analyze_decoded_source(_decoded(audio), vad)

    assert tuple(facts) == STEP1_ANALYSIS_FIELDS
    assert facts["source_start_sample"] == 0
    assert facts["source_end_sample"] == frames
    assert facts["signal_finite"] is True
    assert facts["signal_num_samples"] == frames
    assert facts["vad_speech_ratio"] == 0.123457
    assert facts["vad_timeline"] == [
        {"start_sample": 10, "end_sample": 100, "mean_probability": 0.987654}
    ]
    assert facts["vad_timeline_sample_rate"] == 16000
    assert facts["vad_timeline_num_samples"] == frames
    assert facts["stable_silence_timeline"] == []
    assert facts["bandwidth_hz"] > 0.0
    assert facts["estimated_sr"] is not None
    assert "snr_db" not in facts
    assert len(vad.calls) == 1
    assert vad.calls[0][0].shape == (frames,)
    json.dumps(facts, allow_nan=False)
    for key, value in facts.items():
        assert not isinstance(value, np.generic), key


def test_stereo_vad_receives_first_channel_not_channel_mean():
    frames = 4096
    left = np.linspace(-0.8, 0.4, frames, dtype=np.float32)
    right = np.linspace(0.6, -0.2, frames, dtype=np.float32)
    # 多通道源在 Step1 契约中以 channels 元数据 + channel 0 mono 表示。
    audio = np.ascontiguousarray(left)
    expected_mono = left
    vad = FakeVad(_vad_result(frames))

    facts = analyze_decoded_source(_decoded(audio, channels=2), vad)

    assert len(vad.calls) == 1
    received, received_sr, received_copy = vad.calls[0]
    assert received_sr == 16000
    assert received.ndim == 1
    assert received.shape == (frames,)
    np.testing.assert_array_equal(received, expected_mono)
    np.testing.assert_array_equal(received, received_copy)
    assert facts["signal_num_samples"] == frames * 2
    assert 0.0 <= facts["bandwidth_hz"] <= 8000.0


def test_stable_silence_parameters_are_fixed_and_payload_is_complete():
    assert STEP1_STABLE_SILENCE_PARAMETERS == StableSilenceParameters(
        frame_ms=10,
        rms_threshold_dbfs=-45.0,
        peak_threshold_dbfs=-30.0,
        min_silence_ms=200,
        guard_ms=100,
    )
    payload = Step1AnalyzeStage().config_payload(object())
    assert payload["row_contract"] == {
        "version": 7,
        "fields": list(analyze_module.STEP1_ROW_FIELDS),
    }
    assert payload["stable_silence"] == {
        "version": 3,
        "source_domain": "original_sample_rate",
        "channel_policy": "first_channel_index_0",
        "working_set_max_bytes": 8 * 1024 * 1024,
        "algorithm": "float64_inplace_square_reduceat",
        "frame_ms": 10,
        "rms_threshold_dbfs": -45.0,
        "peak_threshold_dbfs": -30.0,
        "min_silence_ms": 200,
        "guard_ms": 100,
        "interval": "half_open",
    }


def test_stable_silence_analysis_uses_first_channel_only():
    sample_rate = 8000
    audio = np.full((4000, 2), 0.1, dtype=np.float32)
    audio[800:3200] = 0.0
    assert detect_stable_silence_intervals(audio[:, :1], sample_rate) == ((800, 3200),)

    # Step1 只将第 1 通道（索引 0）传给检测；其他通道的活动不应影响结果。
    audio[1960:2040, 1] = 0.1
    assert detect_stable_silence_intervals(audio[:, :1], sample_rate) == ((800, 3200),)


def test_stable_silence_rms_can_fail_while_peak_passes():
    audio = np.full((3200, 1), 10.0 ** (-40.0 / 20.0), dtype=np.float32)
    assert detect_stable_silence_intervals(audio, 16000) == ()


def test_stable_silence_peak_can_fail_while_rms_passes():
    audio = np.zeros((3200, 1), dtype=np.float32)
    audio[::160, 0] = 10.0 ** (-29.0 / 20.0)
    assert detect_stable_silence_intervals(audio, 16000) == ()


def test_stable_silence_accepts_threshold_equality():
    rms_limit = 10.0 ** (-45.0 / 20.0)
    parameters = StableSilenceParameters(
        rms_threshold_dbfs=-45.0,
        peak_threshold_dbfs=-45.0,
    )
    audio = np.full((3200, 2), rms_limit, dtype=np.float64)
    assert detect_stable_silence_intervals(
        audio, 16000, parameters=parameters
    ) == ((0, 3200),)


@pytest.mark.parametrize(
    ("duration_ms", "expected"),
    [(199, ()), (200, ((1600, 4800),))],
)
def test_stable_silence_minimum_duration_boundary(duration_ms, expected):
    sample_rate = 16000
    audio = np.full((6400, 1), 0.1, dtype=np.float32)
    end = 1600 + duration_ms * sample_rate // 1000
    audio[1600:end] = 0.0
    assert detect_stable_silence_intervals(audio, sample_rate) == expected


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"frame_ms": True}, TypeError),
        ({"frame_ms": 10.0}, TypeError),
        ({"frame_ms": 0}, ValueError),
        ({"rms_threshold_dbfs": -45}, TypeError),
        ({"rms_threshold_dbfs": float("nan")}, ValueError),
        ({"peak_threshold_dbfs": float("inf")}, ValueError),
        ({"min_silence_ms": False}, TypeError),
        ({"min_silence_ms": 0}, ValueError),
        ({"guard_ms": 0.0}, TypeError),
        ({"guard_ms": -1}, ValueError),
    ],
)
def test_stable_silence_parameters_reject_malformed_values(changes, error):
    with pytest.raises(error):
        StableSilenceParameters(**changes)


def test_stable_silence_rejects_zero_channels_even_when_empty():
    with pytest.raises(ValueError, match="channel"):
        detect_stable_silence_intervals(np.empty((0, 0), dtype=np.float32), 16000)


def test_stable_silence_192k_eight_channel_block_respects_byte_budget(monkeypatch):
    block_frames = segmentation_module._stable_silence_block_frames(192000, 10, 8)
    assert block_frames == 68
    bytes_per_frame = 1920 * 8 * np.dtype(np.float64).itemsize
    assert block_frames * bytes_per_frame <= 8 * 1024 * 1024
    assert (block_frames + 1) * bytes_per_frame > 8 * 1024 * 1024

    calls = []
    original = segmentation_module._stable_silence_block_flags

    def recording(audio, starts, ends, *limits):
        calls.append(len(starts))
        return original(audio, starts, ends, *limits)

    monkeypatch.setattr(segmentation_module, "_stable_silence_block_flags", recording)
    frame_count = 2 * block_frames + 1
    audio = np.zeros((frame_count * 1920, 8), dtype=np.float32)
    assert detect_stable_silence_intervals(audio, 192000) == (
        (0, audio.shape[0]),
    )
    assert calls == [68, 68, 1]


def test_stable_silence_preserves_nonintegral_frame_bounds_and_tail():
    parameters = StableSilenceParameters(
        frame_ms=1,
        rms_threshold_dbfs=-45.0,
        peak_threshold_dbfs=-30.0,
        min_silence_ms=1,
        guard_ms=0,
    )
    audio = np.zeros((67, 2), dtype=np.float32)
    audio[22, 1] = 0.1

    # 22-sample 首帧不足 ceil(22.05)，尾部 23 samples 含最后的不完整帧。
    assert detect_stable_silence_intervals(
        audio, 22050, parameters=parameters
    ) == ((44, 67),)


def test_stable_silence_randomized_matches_scalar_oracle():
    rng = random.Random(0x51AB1E)
    parameters = StableSilenceParameters(min_silence_ms=20)
    rms_limit = 10.0 ** (parameters.rms_threshold_dbfs / 20.0)
    peak_limit = 10.0 ** (parameters.peak_threshold_dbfs / 20.0)

    for _ in range(100):
        sample_rate = rng.choice((8000, 16000, 22050, 44100, 192000))
        channels = rng.randint(1, 4)
        frame_count = rng.randint(1, 50)
        num_samples = rng.randint(1, frame_count * sample_rate // 100)
        audio = np.asarray(
            [
                [rng.uniform(-0.04, 0.04) for _channel in range(channels)]
                for _sample in range(num_samples)
            ],
            dtype=np.float32,
        )
        denominator = sample_rate * parameters.frame_ms
        actual_frame_count = (num_samples * 1000 + denominator - 1) // denominator
        silent = []
        for frame_index in range(actual_frame_count):
            start = frame_index * denominator // 1000
            end = min(num_samples, (frame_index + 1) * denominator // 1000)
            frame = audio[start:end].astype(np.float64)
            per_channel_peak = np.sqrt(np.max(np.square(frame), axis=0))
            per_channel_rms = np.sqrt(np.sum(np.square(frame), axis=0) / len(frame))
            silent.append(bool(np.all(
                (per_channel_peak <= peak_limit) & (per_channel_rms <= rms_limit)
            )))

        expected = []
        run_start = None
        for frame_index, is_silent in enumerate([*silent, False]):
            if is_silent and run_start is None:
                run_start = frame_index
            elif not is_silent and run_start is not None:
                start = run_start * denominator // 1000
                end = min(num_samples, frame_index * denominator // 1000)
                if end - start >= (parameters.min_silence_ms * sample_rate + 999) // 1000:
                    expected.append((start, end))
                run_start = None

        assert detect_stable_silence_intervals(
            audio, sample_rate, parameters=parameters
        ) == tuple(expected)


def test_ten_second_internal_pause_preserves_waveform_bounds_and_timeline_gap():
    frames = 160000
    audio = np.full(frames, 0.1, dtype=np.float32)
    audio[40000:120000] = 0.0
    original_object = audio
    original_content = audio.copy()
    timeline = (
        SpeechInterval(0, 40000, 0.9),
        SpeechInterval(120000, frames, 0.8),
    )
    vad = FakeVad(_vad_result(frames, timeline=timeline))

    facts = analyze_decoded_source(_decoded(audio), vad)

    assert audio is original_object
    np.testing.assert_array_equal(audio, original_content)
    assert facts["source_start_sample"] == 0
    assert facts["source_end_sample"] == frames
    assert facts["vad_timeline"] == [
        {"start_sample": 0, "end_sample": 40000, "mean_probability": 0.9},
        {"start_sample": 120000, "end_sample": frames, "mean_probability": 0.8},
    ]
    assert facts["stable_silence_timeline"] == [
        {"start_sample": 40000, "end_sample": 120000}
    ]
    assert set(facts) == set(STEP1_ANALYSIS_FIELDS)


def test_policy_only_extremes_are_emitted_as_facts_not_rejected():
    audio = np.zeros(4096, dtype=np.float32)
    audio[0] = 1.0
    audio[1] = -1.0
    audio[2] = 0.9
    vad = FakeVad(_vad_result(4096, speech_ratio=0.0, timeline=()))

    facts = analyze_decoded_source(_decoded(audio), vad)

    assert facts["signal_zero_ratio"] > 0.99
    assert facts["signal_peak"] == 1.0
    assert facts["signal_clipping_ratio"] > 0.0
    assert facts["bandwidth_hz"] >= 0.0
    assert facts["vad_speech_ratio"] == 0.0
    assert facts["vad_timeline"] == []


def test_short_nonempty_tone_uses_zero_padded_bandwidth_window():
    sample_rate = 16000
    frames = 256
    time = np.arange(frames, dtype=np.float32) / sample_rate
    audio = (0.25 * np.sin(2 * np.pi * 1000 * time)).astype(np.float32)
    vad = FakeVad(_vad_result(frames, timeline=()))

    facts = analyze_decoded_source(_decoded(audio), vad)

    assert facts["bandwidth_hz"] > 0.0
    assert facts["estimated_sr"] is not None
    assert facts["estimated_sr"] <= sample_rate


def test_zero_rms_and_zero_bandwidth_are_preserved_as_analysis_facts():
    audio = np.zeros(4096, dtype=np.float32)
    vad = FakeVad(_vad_result(4096, speech_ratio=0.0, timeline=()))

    facts = analyze_decoded_source(_decoded(audio), vad)

    assert facts["signal_rms_dbfs"] is None
    assert facts["bandwidth_hz"] == 0.0
    assert facts["estimated_sr"] is None
    assert facts["vad_speech_ratio"] == 0.0
    assert facts["vad_timeline"] == []
    assert len(vad.calls) == 1


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_audio_fails_closed(bad_value):
    audio = np.zeros(32, dtype=np.float32)
    audio[3] = bad_value
    vad = FakeVad(_vad_result(32))

    with pytest.raises(ValueError, match="finite"):
        analyze_decoded_source(_decoded(audio), vad)

    assert vad.calls == []


@pytest.mark.parametrize(
    "decoded",
    [
        object(),
        _decoded(np.zeros(8, dtype=np.float64)),
        _decoded(np.zeros(8, dtype=np.float32)),
        _decoded(np.zeros((8, 1), dtype=np.float32)),  # 二维必须拒绝
        _decoded(np.zeros((8, 1, 1), dtype=np.float32)),
        _decoded(np.zeros(8, dtype=np.float32), frames=7),
        _decoded(np.zeros(8, dtype=np.float32), channels=0),
        _decoded(np.zeros(0, dtype=np.float32)),
        _decoded(np.zeros(16, dtype=np.float32)[::2]),
        _decoded(np.zeros(8, dtype=np.float32), sample_rate=True),
        _decoded(np.zeros(8, dtype=np.float32), frames=np.int64(8)),
        _decoded(np.zeros(8, dtype=np.float32), channels=False),
        _decoded(np.zeros(8, dtype=np.float32), format=""),
        _decoded(np.zeros(8, dtype=np.float32), subtype=""),
        _decoded(np.zeros(8, dtype=np.float32), source_size_bytes=-1),
        _decoded(np.zeros(8, dtype=np.float32), source_size_bytes=True),
        _decoded(np.zeros(8, dtype=np.float32), source_sha256="A" * 64),
        _decoded(np.zeros(8, dtype=np.float32), source_sha256="a" * 63),
    ],
)
def test_malformed_decoded_source_fails_closed_before_dependencies(decoded):
    class ForbiddenVad:
        def analyze(self, *_args):
            raise AssertionError("VAD must not run for malformed decoded input")

    with pytest.raises(ValueError):
        analyze_decoded_source(decoded, ForbiddenVad())


@pytest.mark.parametrize(
    "metrics",
    [
        SignalMetrics(True, 0.0, 0.0, 0.0, 0.0, None, 7),
        SignalMetrics(False, None, None, None, None, None, 8),
        SignalMetrics(True, None, 0.0, 0.0, 0.0, None, 8),
        SignalMetrics(True, np.nan, 0.0, 0.0, 0.0, None, 8),
        SignalMetrics(True, 0.0, np.inf, 0.0, 0.0, None, 8),
        SignalMetrics(True, 0.0, 0.0, True, 0.0, None, 8),
        SignalMetrics(True, 0.0, 0.0, 0.0, "bad", None, 8),
        SignalMetrics(True, 0.0, 0.0, 0.0, 0.0, -np.inf, 8),
    ],
)
def test_malformed_signal_metrics_fail_closed(monkeypatch, metrics):
    monkeypatch.setattr(analyze_module, "compute_signal_metrics", lambda _audio: metrics)

    with pytest.raises(ValueError, match="signal metrics"):
        analyze_decoded_source(
            _decoded(np.zeros(8, dtype=np.float32)),
            FakeVad(_vad_result(8)),
        )


@pytest.mark.parametrize(
    "bandwidth_result",
    [
        (8000, 100.0),
        (16000, np.nan),
        (16000, -0.1),
        (16000, 8000.1),
        (16000, True),
        (16000, 100.0, 1.0),
    ],
)
def test_malformed_bandwidth_facts_fail_closed(monkeypatch, bandwidth_result):
    monkeypatch.setattr(
        analyze_module,
        "estimate_audio_bandwidth",
        lambda _audio, sr, **_kwargs: bandwidth_result,
    )

    with pytest.raises(ValueError, match="bandwidth"):
        analyze_decoded_source(
            _decoded(np.zeros(8, dtype=np.float32)),
            FakeVad(_vad_result(8)),
        )


@pytest.mark.parametrize("optimal_sr", [True, 0, -1, 16001, np.int64(8000)])
def test_invalid_optimal_sample_rate_fails_closed(monkeypatch, optimal_sr):
    monkeypatch.setattr(
        analyze_module,
        "estimate_audio_bandwidth",
        lambda _audio, sr, **_kwargs: (sr, 100.0),
    )
    monkeypatch.setattr(
        analyze_module,
        "get_optimal_sample_rate",
        lambda _bandwidth, _source_sr: optimal_sr,
    )

    with pytest.raises(ValueError, match="optimal sample rate"):
        analyze_decoded_source(
            _decoded(np.zeros(8, dtype=np.float32)),
            FakeVad(_vad_result(8)),
        )


@pytest.mark.parametrize(
    "vad_result",
    [
        object(),
        VadAnalysis(np.nan, (), 16000, 8),
        VadAnalysis(-0.1, (), 16000, 8),
        VadAnalysis(1.1, (), 16000, 8),
        VadAnalysis(True, (), 16000, 8),
        VadAnalysis(0.0, (), True, 8),
        VadAnalysis(0.0, (), 0, 8),
        VadAnalysis(0.0, (), 16000, True),
        VadAnalysis(0.0, (), 16000, -1),
        VadAnalysis(0.5, [], 16000, 8),
        VadAnalysis(0.5, (object(),), 16000, 8),
        VadAnalysis(0.5, (SpeechInterval(-1, 2, 0.5),), 16000, 8),
        VadAnalysis(0.5, (SpeechInterval(2, 2, 0.5),), 16000, 8),
        VadAnalysis(0.5, (SpeechInterval(2, 9, 0.5),), 16000, 8),
        VadAnalysis(
            0.5,
            (SpeechInterval(0, 5, 0.5), SpeechInterval(4, 7, 0.5)),
            16000,
            8,
        ),
        VadAnalysis(
            0.5,
            (SpeechInterval(4, 7, 0.5), SpeechInterval(0, 2, 0.5)),
            16000,
            8,
        ),
        VadAnalysis(0.5, (SpeechInterval(True, 2, 0.5),), 16000, 8),
        VadAnalysis(0.5, (SpeechInterval(0, 2, np.nan),), 16000, 8),
        VadAnalysis(0.5, (SpeechInterval(0, 2, 1.1),), 16000, 8),
        VadAnalysis(0.5, (SpeechInterval(0, 2, True),), 16000, 8),
    ],
)
def test_malformed_vad_analysis_fails_closed(monkeypatch, vad_result):
    vad = FakeVad(vad_result)

    with pytest.raises(ValueError, match="VAD"):
        analyze_decoded_source(_decoded(np.zeros(8, dtype=np.float32)), vad)

    assert len(vad.calls) == 1


def test_vad_exception_becomes_clear_analysis_error_and_is_called_once(monkeypatch):

    class ExplodingVad:
        def __init__(self):
            self.calls = 0

        def analyze(self, _audio, _sample_rate):
            self.calls += 1
            raise RuntimeError("model unavailable")

    vad = ExplodingVad()

    with pytest.raises(ValueError, match="VAD analysis failed: model unavailable"):
        analyze_decoded_source(_decoded(np.zeros(8, dtype=np.float32)), vad)

    assert vad.calls == 1


def test_max_float_stereo_uses_stable_finite_mono_for_vad_and_bandwidth():
    maximum = np.finfo(np.float32).max
    alternating = np.where(np.arange(4096) % 2 == 0, maximum, -maximum).astype(
        np.float32
    )
    audio = np.ascontiguousarray(alternating)
    stable_mono64 = alternating.astype(np.float64)
    stable_bandwidth = estimate_audio_bandwidth(stable_mono64, sr=16000)
    vad = FakeVad(_vad_result(4096))

    facts = analyze_decoded_source(_decoded(audio, channels=2), vad)

    assert facts["signal_finite"] is True
    received = vad.calls[0][0]
    assert received.ndim == 1
    assert received.dtype == np.float32
    assert received.flags.c_contiguous
    assert np.all(np.isfinite(received))
    np.testing.assert_array_equal(received, stable_mono64.astype(np.float32))
    assert facts["bandwidth_hz"] == stable_bandwidth[1] == 8000.0
    assert "snr_db" not in facts


def test_bandwidth_helper_reduces_max_float_stereo_with_float64_accumulator():
    maximum = np.finfo(np.float32).max
    alternating = np.where(np.arange(4096) % 2 == 0, maximum, -maximum).astype(
        np.float32
    )
    stereo = np.ascontiguousarray(np.column_stack((alternating, alternating)))
    stable_mono64 = stereo.astype(np.float64).mean(axis=1)

    actual = estimate_audio_bandwidth(stereo, sr=16000)
    expected = estimate_audio_bandwidth(stable_mono64, sr=16000)

    assert actual == expected
    assert actual[1] == 8000.0


def test_source_and_vad_timeline_keep_distinct_sample_domains(monkeypatch):
    audio = np.zeros(48000, dtype=np.float32)
    vad = FakeVad(
        _vad_result(
            16000,
            sample_rate=16000,
            timeline=(SpeechInterval(12000, 16000, 0.8),),
        )
    )

    facts = analyze_decoded_source(_decoded(audio, sample_rate=48000), vad)

    assert facts["source_start_sample"] == 0
    assert facts["source_end_sample"] == 48000
    assert facts["vad_timeline_sample_rate"] == 16000
    assert facts["vad_timeline_num_samples"] == 16000
    assert facts["vad_timeline"] == [
        {"start_sample": 12000, "end_sample": 16000, "mean_probability": 0.8}
    ]
    assert facts["stable_silence_timeline"] == [
        {"start_sample": 0, "end_sample": 48000}
    ]
    assert vad.calls[0][1] == 16000
    assert vad.calls[0][0].shape == (16000,)


@pytest.mark.parametrize(
    "source_sr,frames,expected_length",
    [(8000, 800, 1600), (44100, 4410, 1600), (192000, 19200, 1600)],
)
def test_non_16k_vad_uses_captured_direct_method_once_within_capacity(
    monkeypatch, source_sr, frames, expected_length
):
    audio = np.linspace(-0.5, 0.5, frames, dtype=np.float32)
    real_resample = analyze_module.resample_audio
    calls = []

    def recording_resample(waveform, orig_sr, target_sr, *, method):
        calls.append((orig_sr, target_sr, method))
        return real_resample(waveform, orig_sr, target_sr, method=method)

    monkeypatch.setattr(analyze_module, "resample_audio", recording_resample)
    vad = FakeVad(_vad_result(expected_length, timeline=()))
    parameters = replace(vad_module.STEP1_VAD_PARAMETERS, resample_method="scipy")
    capacity = (frames * 16000 + source_sr - 1) // source_sr + 1

    facts = analyze_decoded_source(
        _decoded(
            audio,
            sample_rate=source_sr,
            vad_capacity_samples=capacity,
        ),
        vad,
        vad_parameters=parameters,
    )

    assert calls == [(source_sr, 16000, "scipy")]
    assert len(vad.calls) == 1
    assert vad.calls[0][1] == 16000
    assert vad.calls[0][0].size == expected_length <= capacity
    assert facts["vad_timeline_num_samples"] == expected_length


def test_soxr_failure_is_closed_without_librosa_or_scipy_fallback(monkeypatch):
    audio = np.zeros(800, dtype=np.float32)
    backend_failure = RuntimeError("soxr backend exploded")
    fallback_calls = []

    monkeypatch.setattr(resample_module, "_soxr_available", lambda: True)
    monkeypatch.setattr(
        resample_module,
        "_resample_soxr",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(backend_failure),
    )

    def forbidden_fallback(*_args, **_kwargs):
        fallback_calls.append(True)
        raise AssertionError("fallback backend must not run")

    monkeypatch.setattr(resample_module, "_resample_librosa", forbidden_fallback)
    monkeypatch.setattr(resample_module, "_resample_scipy", forbidden_fallback)
    vad = FakeVad(_vad_result(1600, timeline=()))

    with pytest.raises(ValueError, match="VAD resampling failed") as caught:
        analyze_decoded_source(
            _decoded(audio, sample_rate=8000, vad_capacity_samples=1601), vad
        )

    assert caught.value.__cause__ is backend_failure
    assert fallback_calls == []
    assert vad.calls == []


def test_identity_16k_real_processor_skips_both_resample_seams_and_infers_once(
    monkeypatch,
):
    audio = np.zeros(8, dtype=np.float32)

    def forbidden_resample(*_args, **_kwargs):
        raise AssertionError("identity VAD must not resample")

    monkeypatch.setattr(analyze_module, "resample_audio", forbidden_resample)
    monkeypatch.setattr(vad_module, "resample_audio", forbidden_resample)
    processor = object.__new__(vad_module.SileroVadProcessor)
    processor.device = pytest.importorskip("torch").device("cpu")
    processor.resample_method = vad_module.STEP1_VAD_PARAMETERS.resample_method
    processor.analysis_parameters = vad_module.STEP1_VAD_PARAMETERS
    inference_calls = []

    def one_inference(waveform):
        inference_calls.append(waveform.copy())
        return np.array([0.0], dtype=np.float32)

    processor._frame_probabilities_16k = one_inference

    facts = analyze_decoded_source(
        _decoded(audio, vad_capacity_samples=8),
        processor,
    )

    assert facts["vad_timeline_num_samples"] == 8
    assert len(inference_calls) == 1
    np.testing.assert_array_equal(inference_calls[0], audio)


def test_non_16k_real_processor_resamples_outer_once_and_infers_once(monkeypatch):
    audio = np.zeros(4, dtype=np.float32)
    outer_calls = []

    def fixed_outer_resample(_waveform, orig_sr, target_sr, *, method):
        outer_calls.append((orig_sr, target_sr, method))
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(analyze_module, "resample_audio", fixed_outer_resample)
    monkeypatch.setattr(
        vad_module,
        "resample_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("processor must receive the prepared VAD domain")
        ),
    )
    processor = object.__new__(vad_module.SileroVadProcessor)
    processor.device = pytest.importorskip("torch").device("cpu")
    processor.resample_method = vad_module.STEP1_VAD_PARAMETERS.resample_method
    processor.analysis_parameters = vad_module.STEP1_VAD_PARAMETERS
    inference_calls = []

    def one_inference(waveform):
        inference_calls.append(waveform.copy())
        return np.array([0.0], dtype=np.float32)

    processor._frame_probabilities_16k = one_inference

    facts = analyze_decoded_source(
        _decoded(audio, sample_rate=8000, vad_capacity_samples=9),
        processor,
    )

    assert outer_calls == [(8000, 16000, "soxr_hq")]
    assert facts["vad_timeline_num_samples"] == 8
    assert len(inference_calls) == 1
    assert inference_calls[0].size == 8


@pytest.mark.parametrize("capacity,should_pass", [(8, True), (7, False)])
def test_actual_vad_output_capacity_boundary_precedes_inference(
    monkeypatch, capacity, should_pass
):
    audio = np.zeros(4, dtype=np.float32)
    resample_calls = []

    def fixed_resample(_waveform, _orig_sr, _target_sr, *, method):
        resample_calls.append(method)
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(analyze_module, "resample_audio", fixed_resample)
    vad = FakeVad(_vad_result(8, timeline=()))
    decoded = _decoded(
        audio,
        sample_rate=8000,
        vad_capacity_samples=capacity,
    )

    if should_pass:
        facts = analyze_decoded_source(decoded, vad)
        assert facts["vad_timeline_num_samples"] == 8
        assert len(vad.calls) == 1
    else:
        with pytest.raises(ValueError, match="VAD.*capacity"):
            analyze_decoded_source(decoded, vad)
        assert vad.calls == []
    assert resample_calls == ["soxr_hq"]


def test_missing_vad_capacity_fails_before_resample_or_inference(monkeypatch):
    audio = np.zeros(4, dtype=np.float32)
    monkeypatch.setattr(
        analyze_module,
        "resample_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("missing capacity must fail before resampling")
        ),
    )
    vad = FakeVad(_vad_result(8, timeline=()))

    with pytest.raises(ValueError, match="VAD capacity"):
        analyze_decoded_source(
            _decoded(audio, sample_rate=8000, vad_capacity_samples=None), vad
        )

    assert vad.calls == []


@pytest.mark.parametrize(
    "reported_length,should_pass",
    [(16000, False), (16001, True)],
    ids=["formula-length-rejected", "actual-length-accepted"],
)
def test_non_integral_scipy_vad_length_uses_actual_resampler_output(
    monkeypatch, reported_length, should_pass,
):
    audio = np.linspace(-0.5, 0.5, 44101, dtype=np.float32)
    expected_vad_audio = resample_audio(
        audio, 44100, 16000, method="scipy",
    )
    assert expected_vad_audio.size == 16001
    real_resample = analyze_module.resample_audio
    resample_calls = []

    def recording_resample(waveform, orig_sr, target_sr, *, method):
        result = real_resample(waveform, orig_sr, target_sr, method=method)
        resample_calls.append((waveform.copy(), orig_sr, target_sr, method, result.copy()))
        return result

    monkeypatch.setattr(analyze_module, "resample_audio", recording_resample)
    vad = FakeVad(_vad_result(reported_length, timeline=()))
    parameters = replace(vad_module.STEP1_VAD_PARAMETERS, resample_method="scipy")

    if should_pass:
        facts = analyze_decoded_source(
            _decoded(audio, sample_rate=44100),
            vad,
            vad_parameters=parameters,
        )
        assert facts["vad_timeline_num_samples"] == expected_vad_audio.size
    else:
        with pytest.raises(ValueError, match="VAD num_samples.*resampled waveform"):
            analyze_decoded_source(
                _decoded(audio, sample_rate=44100),
                vad,
                vad_parameters=parameters,
            )

    assert len(resample_calls) == 1
    _, orig_sr, target_sr, method, actual_vad_audio = resample_calls[0]
    assert (orig_sr, target_sr, method) == (44100, 16000, "scipy")
    np.testing.assert_array_equal(actual_vad_audio, expected_vad_audio)
    assert len(vad.calls) == 1
    np.testing.assert_array_equal(vad.calls[0][0], expected_vad_audio)


def test_16k_vad_rejects_processor_reported_length_mismatch(monkeypatch):
    audio = np.zeros(4096, dtype=np.float32)
    vad = FakeVad(_vad_result(4095, timeline=()))

    with pytest.raises(ValueError, match="VAD num_samples.*decoded frames"):
        analyze_decoded_source(_decoded(audio), vad)

    assert len(vad.calls) == 1


def test_vad_rejects_non_pinned_analysis_sample_rate(monkeypatch):
    audio = np.zeros(4096, dtype=np.float32)
    vad = FakeVad(_vad_result(4096, sample_rate=8000, timeline=()))

    with pytest.raises(ValueError, match="VAD sample_rate must be 16000"):
        analyze_decoded_source(_decoded(audio), vad)

    assert len(vad.calls) == 1


def test_vad_rejects_mutated_non_pinned_parameter_before_processor_call(monkeypatch):
    audio = np.zeros(4096, dtype=np.float32)
    vad = FakeVad(_vad_result(2048, sample_rate=8000, timeline=()))
    parameters = replace(vad_module.STEP1_VAD_PARAMETERS, analysis_sample_rate=8000)

    with pytest.raises(ValueError, match="VAD analysis_sample_rate must be pinned to 16000"):
        analyze_decoded_source(_decoded(audio), vad, vad_parameters=parameters)

    assert vad.calls == []


@pytest.mark.parametrize(
    "method",
    [None, True, 1, np.str_("scipy"), "", " ", "auto", "unknown"],
)
def test_vad_parameters_reject_non_exact_or_non_direct_resample_method(method):
    with pytest.raises((TypeError, ValueError), match="resample_method"):
        replace(vad_module.STEP1_VAD_PARAMETERS, resample_method=method)


def test_processor_cache_separates_resample_method_parameter_identity(monkeypatch):
    cache = {}
    load_calls = []
    monkeypatch.setattr(vad_module, "_thread_cache", lambda _name: cache)

    def fake_model_loader(model_path, device):
        load_calls.append((model_path, str(device)))
        return object()

    monkeypatch.setattr(vad_module, "_load_vad_model", fake_model_loader)
    baseline_parameters = vad_module.STEP1_VAD_PARAMETERS
    method_parameters = replace(baseline_parameters, resample_method="scipy")
    shape_parameters = replace(baseline_parameters, frame_samples=256)

    baseline = vad_module.get_vad_processor(
        device="cpu",
        resample_method=baseline_parameters.resample_method,
        model_path="/model/vad.jit",
        analysis_parameters=baseline_parameters,
    )
    resumed = vad_module.get_vad_processor(
        device="cpu",
        resample_method=baseline_parameters.resample_method,
        model_path="/model/vad.jit",
        analysis_parameters=baseline_parameters,
    )
    method_changed = vad_module.get_vad_processor(
        device="cpu",
        resample_method=method_parameters.resample_method,
        model_path="/model/vad.jit",
        analysis_parameters=method_parameters,
    )
    shape_changed = vad_module.get_vad_processor(
        device="cpu",
        resample_method=baseline_parameters.resample_method,
        model_path="/model/vad.jit",
        analysis_parameters=shape_parameters,
    )

    assert resumed is baseline
    assert method_changed is not baseline
    assert shape_changed is not baseline
    assert method_changed is not shape_changed
    assert baseline.analysis_parameters is baseline_parameters
    assert method_changed.analysis_parameters is method_parameters
    assert shape_changed.analysis_parameters is shape_parameters
    assert baseline.resample_method == baseline_parameters.resample_method
    assert method_changed.resample_method == method_parameters.resample_method
    assert shape_changed.resample_method == baseline_parameters.resample_method
    assert load_calls == [("/model/vad.jit", "cpu")] * 3


@pytest.mark.parametrize(
    "amplitude,expected_dc,expected_rms",
    [(0.75, 0.75, 20.0 * np.log10(0.75)), (1e-20, 1e-20, -400.0)],
    ids=["high-dc", "low-nonzero-rms"],
)
def test_dc_and_low_nonzero_rms_are_facts_not_rejection(
    amplitude, expected_dc, expected_rms
):
    audio = np.full(4096, amplitude, dtype=np.float32)
    original = audio.copy()
    vad = FakeVad(_vad_result(4096, speech_ratio=0.0, timeline=()))

    facts = analyze_decoded_source(_decoded(audio), vad)

    assert tuple(facts) == STEP1_ANALYSIS_FIELDS
    assert facts["signal_dc_offset"] == pytest.approx(expected_dc)
    assert facts["signal_rms_dbfs"] == pytest.approx(expected_rms)
    assert facts["vad_speech_ratio"] == 0.0
    assert facts["vad_timeline"] == []
    assert not ({"rejected", "path", "output_path", "waveform", "audio"} & facts.keys())
    np.testing.assert_array_equal(audio, original)
    json.dumps(facts, allow_nan=False)


@pytest.mark.parametrize(
    "dependency,expected_label",
    [("signal", "signal metrics analysis failed"), ("bandwidth", "bandwidth analysis failed")],
)
def test_dependency_exceptions_have_clear_category_and_chained_cause(
    monkeypatch, dependency, expected_label
):
    def explode(*_args, **_kwargs):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(
        analyze_module,
        "compute_signal_metrics" if dependency == "signal" else "estimate_audio_bandwidth",
        explode,
    )

    with pytest.raises(ValueError, match=expected_label) as caught:
        analyze_decoded_source(
            _decoded(np.zeros(8, dtype=np.float32)),
            FakeVad(_vad_result(8)),
        )

    assert isinstance(caught.value.__cause__, RuntimeError)


def test_vad_exception_preserves_chained_cause(monkeypatch):

    class ExplodingVad:
        def analyze(self, _audio, _sample_rate):
            raise RuntimeError("backend exploded")

    with pytest.raises(ValueError, match="VAD analysis failed") as caught:
        analyze_decoded_source(
            _decoded(np.zeros(8, dtype=np.float32)), ExplodingVad()
        )

    assert isinstance(caught.value.__cause__, RuntimeError)


def test_unconvertible_real_fact_becomes_labeled_value_error(monkeypatch):
    metrics = SignalMetrics(True, 10**1000, 0.0, 0.0, 0.0, None, 8)
    monkeypatch.setattr(analyze_module, "compute_signal_metrics", lambda _audio: metrics)

    with pytest.raises(ValueError, match="signal metrics zero_ratio"):
        analyze_decoded_source(
            _decoded(np.zeros(8, dtype=np.float32)),
            FakeVad(_vad_result(8)),
        )


def test_nyquist_bandwidth_equality_is_accepted(monkeypatch):
    monkeypatch.setattr(
        analyze_module,
        "estimate_audio_bandwidth",
        lambda _audio, sr, **_kwargs: (sr, sr / 2),
    )

    facts = analyze_decoded_source(
        _decoded(np.zeros(8, dtype=np.float32)), FakeVad(_vad_result(8))
    )

    assert facts["bandwidth_hz"] == 8000.0
    assert facts["estimated_sr"] == 16000


def test_touching_vad_intervals_and_nested_key_order_are_accepted(monkeypatch):
    vad = FakeVad(
        _vad_result(
            8,
            timeline=(
                SpeechInterval(0, 4, 0.25),
                SpeechInterval(4, 8, 0.75),
            ),
        )
    )

    facts = analyze_decoded_source(_decoded(np.zeros(8, dtype=np.float32)), vad)

    assert [tuple(interval) for interval in facts["vad_timeline"]] == [
        ("start_sample", "end_sample", "mean_probability"),
        ("start_sample", "end_sample", "mean_probability"),
    ]
    for interval in facts["vad_timeline"]:
        assert all(type(value) in (int, float) for value in interval.values())


@pytest.mark.parametrize(
    "metrics",
    [object(), SignalMetrics(True, 0.0, 0.0, 0.0, 0.0, None, True)],
    ids=["wrong-result-type", "bool-num-samples"],
)
def test_metrics_result_type_and_bool_num_samples_fail_closed(monkeypatch, metrics):
    monkeypatch.setattr(analyze_module, "compute_signal_metrics", lambda _audio: metrics)

    with pytest.raises(ValueError, match="signal metrics"):
        analyze_decoded_source(
            _decoded(np.zeros(8, dtype=np.float32)),
            FakeVad(_vad_result(8)),
        )


@pytest.mark.parametrize(
    "bandwidth_result",
    [(16000, 0.0, 0.0), [16000, 0.0], (True, 0.0)],
    ids=["wrong-tuple-length", "wrong-result-type", "bool-returned-sr"],
)
def test_bandwidth_shape_type_and_bool_returned_sr_fail_closed(
    monkeypatch, bandwidth_result
):
    monkeypatch.setattr(
        analyze_module,
        "estimate_audio_bandwidth",
        lambda _audio, sr, **_kwargs: bandwidth_result,
    )

    with pytest.raises(ValueError, match="bandwidth"):
        analyze_decoded_source(
            _decoded(np.zeros(8, dtype=np.float32)),
            FakeVad(_vad_result(8)),
        )
