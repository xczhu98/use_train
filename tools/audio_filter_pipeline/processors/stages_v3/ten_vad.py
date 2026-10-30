"""TEN native CPU VAD adapter for the Generation-3 Step1 contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..vad_filter import VadAnalysis, VadAnalysisParameters, speech_timeline


@dataclass(frozen=True)
class TenVadRuntimeOptions:
    """CPU-native TEN VAD options owned by Step1."""

    hop_size: int = 256
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if type(self.hop_size) is not int or self.hop_size <= 0:
            raise ValueError("TEN VAD hop_size must be a positive integer")
        if type(self.threshold) not in (int, float) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("TEN VAD threshold must be within [0, 1]")


def _to_pcm16(audio: np.ndarray) -> np.ndarray:
    """Convert finite normalized float audio to contiguous signed PCM16."""

    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(waveform)):
        raise ValueError("TEN VAD waveform must be finite")
    return np.ascontiguousarray(
        np.rint(np.clip(waveform, -1.0, 1.0) * 32767.0), dtype=np.int16
    )


class TenVadProcessor:
    """Adapt bundled ``ten_vad`` frame results to the established VAD facts."""

    def __init__(
        self,
        *,
        options: TenVadRuntimeOptions = TenVadRuntimeOptions(),
        analysis_parameters: VadAnalysisParameters,
    ) -> None:
        try:
            from ten_vad import TenVad
        except ImportError as exc:
            raise RuntimeError("TEN VAD runtime is unavailable; install ten_vad in the active environment") from exc
        if options.threshold != analysis_parameters.speech_threshold:
            raise ValueError(
                "TEN VAD native threshold and speech_threshold must be identical"
            )
        self.options = options
        self.analysis_parameters = analysis_parameters
        self._vad_factory: Any = TenVad
        self._vad: Any | None = None

    def analyze(self, waveform: np.ndarray, sample_rate: int) -> VadAnalysis:
        parameters = self.analysis_parameters
        if sample_rate != parameters.analysis_sample_rate:
            raise ValueError("TEN VAD input must already be in the configured analysis sample rate")
        vad = self._vad_factory(
            hop_size=self.options.hop_size,
            threshold=self.options.threshold,
        )
        self._vad = vad
        try:
            pcm = _to_pcm16(waveform)
            num_samples = int(pcm.size)
            if not num_samples:
                return VadAnalysis(0.0, (), sample_rate, 0)

            frame_count = math.ceil(num_samples / self.options.hop_size)
            padded = np.zeros(frame_count * self.options.hop_size, dtype=np.int16)
            padded[:num_samples] = pcm
            probabilities = np.empty(frame_count, dtype=np.float32)
            for index in range(frame_count):
                probability, _speech_flag = vad.process(
                    padded[index * self.options.hop_size : (index + 1) * self.options.hop_size]
                )
                probabilities[index] = probability

            speech_ratio = float((probabilities >= parameters.speech_threshold).mean())
            timeline = speech_timeline(
                probabilities,
                total_samples=num_samples,
                sample_rate=sample_rate,
                frame_samples=self.options.hop_size,
                speech_threshold=parameters.speech_threshold,
                min_speech_ms=parameters.min_speech_ms,
                max_silence_gap_ms=parameters.max_silence_gap_ms,
                padding_ms=parameters.padding_ms,
            )
            return VadAnalysis(speech_ratio, timeline, sample_rate, num_samples)
        finally:
            self._vad = None
