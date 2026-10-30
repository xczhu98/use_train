"""共享 Layer C acoustic rendering facade。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.data.simulation_trace import SourceSimulationTrace
from core.data.waveform_bundle import WaveformBundle


@dataclass(frozen=True)
class AcousticBundleResult:
    """同步 waveform bundle 与 Layer C 声学事实。"""

    bundle: WaveformBundle
    acoustic_output: Any

    def record_stems(self, trace: SourceSimulationTrace | None) -> None:
        """显式 export Adapter；不改变 Dataset sample Interface。"""
        if trace is None:
            return
        for role, waveform in self.bundle.tracks.items():
            trace.replace_stem(role, waveform)


class AcousticRenderer:
    """把 Dataset 的 C 层实现投影为 task-neutral acoustic bundle。"""

    def render_bundle(
        self,
        dataset,
        speech: WaveformBundle,
        sample_rate: int,
        max_duration_samples: int,
        rng: np.random.Generator,
        need_noise: bool,
        *,
        forced_source_volume_gain: float | None = None,
        source_trace: SourceSimulationTrace | None = None,
    ) -> AcousticBundleResult:
        speech_2d = speech.require("speech")
        if speech_2d.ndim != 2 or speech_2d.shape[0] != 1:
            raise ValueError(
                "acoustic speech track must have mono shape (1, T), got "
                f"{speech_2d.shape}"
            )
        actual_len = speech.num_samples
        output = dataset._mix_acoustic(
            speech_2d,
            sample_rate,
            actual_len,
            max_duration_samples,
            rng,
            need_noise,
            forced_source_volume_gain=forced_source_volume_gain,
        )
        noisy, clean, after_rir, additive = dataset._align_mix_acoustic_outputs(
            output.noisy_speech,
            output.clean_target,
            output.noisy_after_rir,
            output.additive_noise_2d,
            actual_len,
            sample_rate,
        )
        bundle = WaveformBundle(
            {
                "noisy_speech": noisy,
                "clean_target": clean,
                "noisy_after_rir": after_rir,
                "additive_noise": additive,
            },
            metadata={"sample_rate": int(sample_rate), "need_noise": bool(need_noise)},
        )
        result = AcousticBundleResult(bundle=bundle, acoustic_output=output)
        result.record_stems(source_trace)
        return result


__all__ = ["AcousticBundleResult", "AcousticRenderer"]
