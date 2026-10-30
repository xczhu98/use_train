import pytest
import torch

from core.metrics.separation import build_separation_metric_pairs


@pytest.mark.unit
def test_ss_validation_metric_pairs_are_pit_aligned_and_source_flattened():
    reference = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]
    )
    estimated = reference.flip(1)
    pairs, assignment = build_separation_metric_pairs(
        estimated,
        reference,
        torch.tensor([4]),
        torch.ones(1, 2, dtype=torch.bool),
        16000,
        max_pairs=8,
    )
    assert assignment.tolist() == [[1, 0]]
    assert len(pairs) == 2
    assert pairs[0][0].ndim == pairs[0][1].ndim == 1
    assert pairs[0][0].tolist() == pairs[0][1].tolist()
