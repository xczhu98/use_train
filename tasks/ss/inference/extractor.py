"""Unified inference adapter for waveform and external-STFT SS models."""

from __future__ import annotations

import torch

from core.models.execution_plan import ResolvedExecutionPlan
from tasks.ss.model_io import SSModelRuntime


class SSExtractor:
    """Run one fixed-source-count separator and return ``[K, T]`` waveforms."""

    def __init__(
        self,
        model,
        *,
        plan: ResolvedExecutionPlan,
        stft_module=None,
    ) -> None:
        self.model = model.eval()
        self.stft_module = stft_module
        self.model_runtime = SSModelRuntime(
            self.model,
            plan=plan,
            stft_module=stft_module,
        )

    @torch.no_grad()
    def __call__(self, mixture: torch.Tensor, *, sample_rate: int | None = None) -> torch.Tensor:
        if mixture.ndim == 1:
            mixture = mixture.unsqueeze(0)
        if mixture.ndim != 2:
            raise ValueError("mixture must have shape [B, T] or [T]")
        lengths = torch.full(
            (mixture.shape[0],), mixture.shape[-1], dtype=torch.int64, device=mixture.device
        )
        return self.model_runtime.separate(
            mixture,
            lengths,
            sample_rate,
        ).sources
