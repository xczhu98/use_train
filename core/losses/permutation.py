"""固定 source count 的 permutation assignment 原语。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations

import torch


@dataclass(frozen=True)
class PermutationResult:
    """最优 estimate-to-reference assignment 及可选对齐输出。"""

    assignment: torch.Tensor
    aligned_estimates: torch.Tensor | None
    per_sample_cost: torch.Tensor


@lru_cache(maxsize=None)
def _permutation_table(source_count: int, device: torch.device) -> torch.Tensor:
    """按 source count 和 device 复用不可变精确排列表。"""
    return torch.tensor(
        list(permutations(range(source_count))),
        dtype=torch.long,
        device=device,
    )


def solve_source_permutation(
    pairwise_cost: torch.Tensor,
    *,
    estimates: torch.Tensor | None = None,
    source_mask: torch.Tensor | None = None,
    max_sources: int = 4,
) -> PermutationResult:
    """从 ``[B,K,K]`` pairwise cost 求最小平均代价的精确排列。"""
    if not isinstance(pairwise_cost, torch.Tensor) or pairwise_cost.ndim != 3:
        raise ValueError("pairwise_cost must have shape [B, K, K]")
    batch, estimates_count, references_count = pairwise_cost.shape
    if estimates_count < 1 or estimates_count != references_count:
        raise ValueError("pairwise_cost must be non-empty and square on source axes")
    if estimates_count > max_sources:
        raise ValueError(
            f"source_count={estimates_count} exceeds max_sources={max_sources}"
        )
    if not torch.isfinite(pairwise_cost).all():
        raise ValueError("pairwise_cost must contain only finite values")
    if source_mask is not None:
        if source_mask.shape != (batch, estimates_count) or source_mask.dtype is not torch.bool:
            raise ValueError("source_mask must be bool with shape [B, K]")
        if not bool(source_mask.all().item()):
            raise ValueError("variable source count is not supported")
    if estimates is not None:
        if estimates.ndim < 2 or estimates.shape[:2] != (batch, estimates_count):
            raise ValueError("estimates must start with shape [B, K, ...]")

    permutation_table = _permutation_table(
        estimates_count, torch.device(pairwise_cost.device)
    )
    estimate_index = torch.arange(estimates_count, device=pairwise_cost.device)
    permutation_cost = pairwise_cost[:, estimate_index, permutation_table].mean(dim=-1)
    per_sample_cost, best_index = permutation_cost.min(dim=-1)
    assignment = permutation_table.index_select(0, best_index)

    aligned = None
    if estimates is not None:
        inverse = torch.empty_like(assignment)
        inverse.scatter_(
            1,
            assignment,
            torch.arange(estimates_count, device=assignment.device)
            .view(1, -1)
            .expand(batch, -1),
        )
        gather_index = inverse.view(batch, estimates_count, *([1] * (estimates.ndim - 2)))
        gather_index = gather_index.expand_as(estimates)
        aligned = torch.gather(estimates, 1, gather_index)
    return PermutationResult(assignment, aligned, per_sample_cost)
