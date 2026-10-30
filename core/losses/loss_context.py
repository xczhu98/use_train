"""Task-neutral loss context and projection at the loss Interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class LossContext:
    """All optional references a loss may consume for one forward call."""

    observation: torch.Tensor
    supervision_target: torch.Tensor
    estimate: torch.Tensor
    lengths: torch.Tensor | None = None
    sample_rate: int | None = None
    undesired_ref: torch.Tensor | None = None
    undesired_estimate: torch.Tensor | None = None
    estimate_spectra: torch.Tensor | None = None
    target_spectra: torch.Tensor | None = None

    def to_kwargs(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "noisy": self.observation,
            "sample_rate": self.sample_rate,
            "undesired_ref": self.undesired_ref,
            "undesired_enhanced": self.undesired_estimate,
            "enhanced_spectra": self.estimate_spectra,
            "target_spectra": self.target_spectra,
        }
        return {key: value for key, value in result.items() if value is not None}

    @classmethod
    def from_waveforms(
        cls,
        observation: torch.Tensor,
        target: torch.Tensor,
        estimate: torch.Tensor,
        *,
        lengths: torch.Tensor | None = None,
        sample_rate: int | None = None,
    ) -> "LossContext":
        return cls(
            observation=observation,
            supervision_target=target,
            estimate=estimate,
            lengths=lengths,
            sample_rate=sample_rate,
            undesired_ref=observation - target,
            undesired_estimate=observation - estimate,
        )


__all__ = ["LossContext"]
