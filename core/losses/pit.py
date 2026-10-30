"""与统一 Loss 配置兼容的 permutation-invariant loss wrapper。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from core.losses.loss_contract import LossContract
from core.losses.permutation import solve_source_permutation
from core.utils.registry import LOSS_REGISTRY


@dataclass(frozen=True)
class PITResult:
    loss: torch.Tensor
    assignment: torch.Tensor
    reordered_estimates: torch.Tensor


@LOSS_REGISTRY.register("PITLoss")
class PITLoss(nn.Module):
    """用一个注册的 pairwise loss 构造 utterance-level PIT 标量目标。"""

    SEPARATION_LOSS = True
    LOSS_CONTRACT = LossContract(
        inputs=("est_sources", "sources", "lengths", "source_mask"),
        permutation_invariant=True,
        supports_lengths=True,
        supports_source_mask=True,
    )

    def __init__(
        self,
        pairwise_loss: str = "PairwiseNegSISDRLoss",
        pairwise_loss_args: dict | None = None,
        assignment: str = "exhaustive",
        reduction: str = "mean",
        max_sources: int = 4,
    ) -> None:
        super().__init__()
        if assignment != "exhaustive":
            raise ValueError("PITLoss currently supports assignment='exhaustive' only")
        if reduction != "mean":
            raise ValueError("PITLoss currently supports reduction='mean' only")
        if type(max_sources) is not int or max_sources < 1:
            raise ValueError("max_sources must be an exact positive int")
        LOSS_REGISTRY.ensure_registered(pairwise_loss)
        pairwise_cls = LOSS_REGISTRY.get(pairwise_loss)
        self.pairwise_loss = pairwise_cls(**dict(pairwise_loss_args or {}))
        self.pairwise_loss_name = pairwise_loss
        self.assignment = assignment
        self.reduction = reduction
        self.max_sources = max_sources

    def compute(
        self,
        estimated: torch.Tensor,
        reference: torch.Tensor,
        *,
        source_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> PITResult:
        if estimated.ndim != 3 or reference.ndim != 3:
            raise ValueError("estimated and reference must have shape [B, K, T]")
        if estimated.shape != reference.shape:
            raise ValueError(
                f"estimated/reference shape mismatch: {tuple(estimated.shape)} vs "
                f"{tuple(reference.shape)}"
            )
        batch, sources, _ = estimated.shape
        if sources > self.max_sources:
            raise ValueError(
                f"PIT source_count={sources} exceeds max_sources={self.max_sources}"
            )
        if source_mask is not None:
            if source_mask.shape != (batch, sources) or source_mask.dtype is not torch.bool:
                raise ValueError("source_mask must be bool with shape [B, K]")
            if not bool(source_mask.all().item()):
                raise ValueError("variable source count is not supported in the first SS contract")
        pairwise = self.pairwise_loss(estimated, reference, lengths=lengths)
        expected = (batch, sources, sources)
        if tuple(pairwise.shape) != expected:
            raise ValueError(
                f"pairwise loss must return shape {expected}, got {tuple(pairwise.shape)}"
            )
        permutation = solve_source_permutation(
            pairwise,
            estimates=estimated,
            source_mask=source_mask,
            max_sources=self.max_sources,
        )
        assert permutation.aligned_estimates is not None
        return PITResult(
            permutation.per_sample_cost.mean(),
            permutation.assignment,
            permutation.aligned_estimates,
        )

    def forward(self, estimated, reference, *, source_mask=None, lengths=None):
        return self.compute(
            estimated,
            reference,
            source_mask=source_mask,
            lengths=lengths,
        ).loss
