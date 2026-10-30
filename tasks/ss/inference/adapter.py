"""SS task inference adapter with canonical waveform and spectrum outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from core.inference import FileInferenceItem, PreparedOutput
from tasks.ss.model_io import SSModelRuntime


class SSInferenceAdapter:
    """Run TDANet waveform-native or TIGER external-STFT inference.

    The returned mapping always contains ``est_sources`` and contains
    ``est_sources_spectra`` when the model emits spectra.  Spectra use the
    framework layout ``[B, K, F, T, 2]``.
    """

    def __init__(self, model, *, plan, stft_module=None):
        self.runtime = SSModelRuntime(model.eval(), plan=plan, stft_module=stft_module)

    def infer(self, item: FileInferenceItem) -> PreparedOutput:
        """按公共 runner 合同分离一个 mixture WAV 并写出指定 source。"""
        input_path = item.inputs.get("mixture")
        if input_path is None:
            raise ValueError("SS inference item requires a mixture path")
        source_index = item.attributes.get("source_index", 0)
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError("SS source_index must be an integer")
        try:
            import torchaudio
        except ImportError as exc:  # pragma: no cover - 基础环境通常已提供。
            raise RuntimeError("SS file inference requires torchaudio") from exc
        mixture, sample_rate = torchaudio.load(str(Path(input_path)))
        if mixture.ndim != 2 or mixture.shape[0] < 1:
            raise ValueError("SS file inference requires waveform shape [C, T]")
        batched_mixture = mixture.unsqueeze(0)
        if batched_mixture.shape[1] > 1:
            if self.runtime.plan.input_domain != "spectrum":
                raise ValueError(
                    "multi-channel SS file inference requires a spectrum-domain model"
                )
            model_mixture = batched_mixture
        else:
            model_mixture = batched_mixture[:, 0, :]
        separated = self(model_mixture, sample_rate=int(sample_rate))["est_sources"]
        if separated.ndim != 3 or not 0 <= source_index < separated.shape[1]:
            raise IndexError(
                f"source_index {source_index} is outside [0, {separated.shape[1]})"
            )
        waveform = separated[0, source_index].detach().cpu()

        def writer(path: Path) -> None:
            torchaudio.save(str(path), waveform.unsqueeze(0), int(sample_rate))

        return PreparedOutput(item.uid, writer)

    @torch.no_grad()
    def __call__(self, mixture: torch.Tensor, *, sample_rate: int | None = None) -> dict[str, torch.Tensor]:
        if mixture.ndim == 1:
            mixture = mixture.unsqueeze(0)
        if mixture.ndim not in (2, 3):
            raise ValueError("mixture must have shape [B, T], [B, C, T], or [T]")
        lengths = torch.full(
            (mixture.shape[0],), mixture.shape[-1], dtype=torch.int64, device=mixture.device
        )
        result = self.runtime.separate(mixture, lengths, sample_rate)
        output = {"est_sources": result.sources}
        if result.source_spectra is not None:
            output["est_sources_spectra"] = result.source_spectra
        output.update(result.diagnostics)
        return output
