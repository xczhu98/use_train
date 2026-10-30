"""Permutation-aware source separation metrics."""

from __future__ import annotations

import torch

from core.losses.pairwise import PairwiseNegSISDRLoss
from core.losses.permutation import solve_source_permutation


def _si_sdr(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    lengths: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if lengths is not None:
        mask = (
            torch.arange(estimate.shape[-1], device=estimate.device)
            .view(1, 1, -1)
            < lengths.view(-1, 1, 1)
        ).to(estimate.dtype)
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        estimate = estimate * mask
        reference = reference * mask
        estimate = (estimate - estimate.sum(dim=-1, keepdim=True) / count) * mask
        reference = (reference - reference.sum(dim=-1, keepdim=True) / count) * mask
    else:
        estimate = estimate - estimate.mean(dim=-1, keepdim=True)
        reference = reference - reference.mean(dim=-1, keepdim=True)
    reference_energy = reference.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    target = (estimate * reference).sum(dim=-1, keepdim=True) * reference / reference_energy
    residual = estimate - target
    return 10.0 * torch.log10(
        target.square().sum(dim=-1).clamp_min(eps)
        / residual.square().sum(dim=-1).clamp_min(eps)
    )


def si_sdr(
    estimated: torch.Tensor,
    reference: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return PIT-aligned SI-SDR per source and the selected assignment."""
    if estimated.shape != reference.shape or estimated.ndim != 3:
        raise ValueError("estimated/reference must have the same shape [B, K, T]")
    pairwise = PairwiseNegSISDRLoss()(estimated, reference, lengths=lengths)
    result = solve_source_permutation(
        pairwise,
        estimates=estimated,
    )
    assert result.aligned_estimates is not None
    aligned = result.aligned_estimates
    values = _si_sdr(aligned, reference, lengths=lengths)
    return values, result.assignment


def si_sdr_improvement(
    estimated: torch.Tensor,
    reference: torch.Tensor,
    mixture: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return PIT-aligned SI-SDR improvement over the mixture baseline."""
    values, assignment = si_sdr(estimated, reference, lengths=lengths)
    baseline = _si_sdr(
        mixture.unsqueeze(1).expand_as(reference), reference, lengths=lengths
    )
    return values - baseline, assignment


def build_separation_metric_pairs(
    estimated: torch.Tensor,
    reference: torch.Tensor,
    lengths: torch.Tensor,
    source_mask: torch.Tensor,
    sample_rate: int,
    *,
    max_pairs: int,
) -> tuple[list[tuple[object, object, int]], torch.Tensor]:
    """PIT-align source waveforms and build one-dimensional metric pairs."""
    if type(max_pairs) is not int or max_pairs < 0:
        raise ValueError("max_pairs must be an exact non-negative int")
    pairwise = PairwiseNegSISDRLoss()(estimated, reference, lengths=lengths)
    permutation = solve_source_permutation(
        pairwise,
        estimates=estimated,
        source_mask=source_mask,
    )
    assert permutation.aligned_estimates is not None
    aligned = permutation.aligned_estimates.detach().float().cpu()
    reference_cpu = reference.detach().float().cpu()
    lengths_cpu = lengths.detach().cpu()
    source_mask_cpu = source_mask.detach().cpu()
    pairs = []
    for batch_index in range(reference_cpu.shape[0]):
        utterance_length = int(lengths_cpu[batch_index].item())
        for source_index in range(reference_cpu.shape[1]):
            if len(pairs) >= max_pairs:
                return pairs, permutation.assignment
            if not bool(source_mask_cpu[batch_index, source_index]):
                continue
            ref_np = reference_cpu[
                batch_index, source_index, :utterance_length
            ].numpy()
            if abs(ref_np).max(initial=0.0) < 1e-6:
                continue
            enh_np = aligned[
                batch_index, source_index, :utterance_length
            ].numpy()
            pairs.append((ref_np, enh_np, sample_rate))
    return pairs, permutation.assignment
