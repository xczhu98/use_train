"""Focused Generation-3 TEN VAD migration tests."""

from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audio_filter_pipeline.processors.stages_v3 import cli
from audio_filter_pipeline.processors.stages_v3.analyze import Step1AnalyzeStage
from audio_filter_pipeline.processors.stages_v3.ten_vad import (
    TenVadProcessor,
    TenVadRuntimeOptions,
)
from audio_filter_pipeline.processors.vad_filter import VadAnalysisParameters
from audio_filter_pipeline.utils.artifacts import PipelineStep


pytestmark = pytest.mark.unit


class _FakeTenVad:
    instances = []

    def __init__(self, *, hop_size, threshold):
        self.hop_size = hop_size
        self.threshold = threshold
        self.frames = []
        self.instances.append(self)
        self.probabilities = iter((0.9, 0.1, 0.8))

    def process(self, frame):
        self.frames.append(frame.copy())
        probability = next(self.probabilities)
        return probability, int(probability >= self.threshold)


def test_ten_processor_preserves_step1_vad_analysis_contract(monkeypatch):
    _FakeTenVad.instances.clear()
    fake_module = SimpleNamespace(TenVad=_FakeTenVad)
    monkeypatch.setitem(sys.modules, "ten_vad", fake_module)
    parameters = VadAnalysisParameters(
        min_speech_ms=0,
        max_silence_gap_ms=0,
        padding_ms=0,
    )
    processor = TenVadProcessor(
        options=TenVadRuntimeOptions(hop_size=4, threshold=0.5),
        analysis_parameters=parameters,
    )

    analysis = processor.analyze(
        np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 0.25, -0.25, 1.2, 0.1], dtype=np.float32),
        16000,
    )

    assert analysis.speech_ratio == pytest.approx(2 / 3)
    assert analysis.sample_rate == 16000
    assert analysis.num_samples == 9
    assert [(item.start_sample, item.end_sample) for item in analysis.timeline] == [
        (0, 4), (8, 9),
    ]
    assert processor._vad is None
    np.testing.assert_array_equal(
        np.stack(_FakeTenVad.instances[0].frames),
        np.array(
            [
                [-32767, -16384, 0, 16384],
                [32767, 8192, -8192, 32767],
                [3277, 0, 0, 0],
            ],
            dtype=np.int16,
        ),
    )


def test_ten_processor_creates_a_fresh_native_handler_per_source(monkeypatch):
    class StatefulTenVad:
        instances = []

        def __init__(self, *, hop_size, threshold):
            self.threshold = threshold
            self.processed_frames = 0
            self.instances.append(self)

        def process(self, _frame):
            self.processed_frames += 1
            probability = 0.9 if self.processed_frames == 1 else 0.1
            return probability, int(probability >= self.threshold)

    monkeypatch.setitem(sys.modules, "ten_vad", SimpleNamespace(TenVad=StatefulTenVad))
    parameters = VadAnalysisParameters(
        min_speech_ms=0,
        max_silence_gap_ms=0,
        padding_ms=0,
    )
    processor = TenVadProcessor(
        options=TenVadRuntimeOptions(hop_size=4, threshold=0.5),
        analysis_parameters=parameters,
    )
    source_a = np.ones(8, dtype=np.float32)
    source_b = np.ones(4, dtype=np.float32)

    processor.analyze(source_a, 16000)
    after_a_b = processor.analyze(source_b, 16000)
    fresh_b = TenVadProcessor(
        options=TenVadRuntimeOptions(hop_size=4, threshold=0.5),
        analysis_parameters=parameters,
    ).analyze(source_b, 16000)

    assert after_a_b == fresh_b
    assert len(StatefulTenVad.instances) == 3
    assert [instance.processed_frames for instance in StatefulTenVad.instances] == [2, 1, 1]


def test_ten_processor_releases_handler_after_success_empty_and_failure(monkeypatch):
    class ReleasableTenVad:
        references = []

        def __init__(self, *, hop_size, threshold):
            self.threshold = threshold
            self.fail = False
            self.references.append(weakref.ref(self))

        def process(self, _frame):
            if self.fail:
                raise RuntimeError("injected native failure")
            return 0.9, 1

    monkeypatch.setitem(sys.modules, "ten_vad", SimpleNamespace(TenVad=ReleasableTenVad))
    processor = TenVadProcessor(
        options=TenVadRuntimeOptions(hop_size=4, threshold=0.5),
        analysis_parameters=VadAnalysisParameters(min_speech_ms=0, max_silence_gap_ms=0, padding_ms=0),
    )

    processor.analyze(np.ones(4, dtype=np.float32), 16000)
    processor.analyze(np.empty(0, dtype=np.float32), 16000)
    processor._vad_factory = lambda **_kwargs: SimpleNamespace(
        process=lambda _frame: (_ for _ in ()).throw(RuntimeError("injected native failure"))
    )
    with pytest.raises(RuntimeError, match="injected native failure"):
        processor.analyze(np.ones(4, dtype=np.float32), 16000)

    assert processor._vad is None
    gc.collect()
    assert all(reference() is None for reference in ReleasableTenVad.references)


def test_native_ten_pcm_wav_b_matches_fresh_processor_after_a(tmp_path):
    pytest.importorskip("ten_vad")
    sample_rate = 16000
    source_a = tmp_path / "a.wav"
    source_b = tmp_path / "b.wav"
    time_a = np.arange(4096, dtype=np.float32) / sample_rate
    time_b = np.arange(2048, dtype=np.float32) / sample_rate
    sf.write(source_a, 0.2 * np.sin(2 * np.pi * 220 * time_a), sample_rate, subtype="PCM_16")
    sf.write(source_b, 0.2 * np.sin(2 * np.pi * 440 * time_b), sample_rate, subtype="PCM_16")
    waveform_a, read_rate_a = sf.read(source_a, dtype="float32")
    waveform_b, read_rate_b = sf.read(source_b, dtype="float32")
    assert (read_rate_a, read_rate_b) == (sample_rate, sample_rate)
    parameters = VadAnalysisParameters()
    options = TenVadRuntimeOptions()
    reused = TenVadProcessor(options=options, analysis_parameters=parameters)

    reused.analyze(waveform_a, sample_rate)
    after_a_b = reused.analyze(waveform_b, sample_rate)
    fresh_b = TenVadProcessor(options=options, analysis_parameters=parameters).analyze(
        waveform_b, sample_rate,
    )

    assert after_a_b == fresh_b


def test_ten_processor_rejects_divergent_native_and_fact_thresholds(monkeypatch):
    monkeypatch.setitem(sys.modules, "ten_vad", SimpleNamespace(TenVad=_FakeTenVad))

    with pytest.raises(ValueError, match="threshold"):
        TenVadProcessor(
            options=TenVadRuntimeOptions(threshold=0.7),
            analysis_parameters=VadAnalysisParameters(speech_threshold=0.5),
        )


def test_step1_payload_records_native_cpu_ten_runtime():
    stage = Step1AnalyzeStage(ten_vad_options=TenVadRuntimeOptions(hop_size=128))
    payload = stage.config_payload(SimpleNamespace())

    assert payload["vad"] == {
        "version": 5,
        "channel_policy": "first_channel_index_0",
        "native_cpu": True,
        "native_handler_lifecycle": "per_source",
        "asset": None,
        "hop_size": 128,
        "native_threshold": 0.5,
        "workers": 1,
        "max_pending": 1,
        "analysis_sample_rate": 16000,
        "resample_method": "soxr_hq",
        "speech_threshold": 0.5,
        "min_speech_ms": 100,
        "max_silence_gap_ms": 200,
        "padding_ms": 100,
    }


def test_cli_uses_default_ten_without_silero_assets():
    config = cli.PipelineConfig(pipeline_generation=3, step="step1", output_root="/tmp/out")
    factories = cli._factories(
        SimpleNamespace(use_gpu=None, no_use_gpu=False), {}, config, PipelineStep.STEP1
    )
    stage = factories[PipelineStep.STEP1](
        SimpleNamespace(step1_vad_workers=3, step1_vad_max_pending=6), "alpha"
    )

    assert isinstance(stage, Step1AnalyzeStage)
    assert stage.ten_vad_options == TenVadRuntimeOptions()
    assert stage.workers == 3
    assert stage.max_pending == 6


def test_cli_accepts_ten_cpu_concurrency_options():
    raw_config = {
        "pipeline_generation": 3,
        "output_root": "/tmp/out",
        "vad": {"workers": 3, "max_pending": 6},
    }
    config = cli._build_config(SimpleNamespace(), raw_config, PipelineStep.STEP1)
    options = cli._ten_vad_options(raw_config)

    assert (config.step1_vad_workers, config.step1_vad_max_pending) == (3, 6)
    assert options == TenVadRuntimeOptions()


@pytest.mark.parametrize(
    "vad, error",
    [
        ({"runtime": "silero"}, "ten_vad_native_cpu"),
        ({"hop_size": 0}, "hop_size"),
        ({"threshold": 1.1}, "threshold"),
        ({"unknown": 2}, "unsupported"),
    ],
)
def test_cli_rejects_unsupported_ten_runtime_options(vad, error):
    with pytest.raises(ValueError, match=error):
        cli._ten_vad_options({"vad": vad})
