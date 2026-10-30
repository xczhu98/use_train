"""Unit tests for tasks/aec/metrics/state_metrics.py."""

from __future__ import annotations

import pytest
import torch

from tasks.aec.trainer.aec_trainer import AECTrainer
from tasks.aec.metrics.state_metrics import (
    compute_state_counts,
    state_metrics_from_counts,
)


def _compute_state_metrics(logits, labels, *, ignore_index: int = -1):
    """测试侧单 batch 便捷封装（生产侧只保留 counts / from_counts 两步 API）。"""
    counts = compute_state_counts(logits, labels, ignore_index=ignore_index)
    metrics = state_metrics_from_counts(counts["tp"], counts["fp"], counts["fn"])
    valid_mask = labels != ignore_index
    if bool(valid_mask.any()):
        preds = logits.argmax(dim=-1)
        metrics["state_accuracy"] = (
            (preds[valid_mask] == labels[valid_mask]).float().mean().item()
        )
    return metrics


def test_compute_state_counts_perfect_predictions():
    """All predictions correct -> tp=1 per class, fp=fn=0."""
    # 3 classes, 2 frames each.
    logits = torch.zeros(2, 3, 3)
    logits[0, 0, 0] = 1.0
    logits[0, 1, 1] = 1.0
    logits[0, 2, 2] = 1.0
    logits[1, 0, 0] = 1.0
    logits[1, 1, 1] = 1.0
    logits[1, 2, 2] = 1.0
    labels = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    counts = compute_state_counts(logits, labels)
    assert torch.equal(counts["tp"], torch.tensor([2, 2, 2]))
    assert torch.equal(counts["fp"], torch.tensor([0, 0, 0]))
    assert torch.equal(counts["fn"], torch.tensor([0, 0, 0]))


def test_compute_state_counts_with_ignore_index():
    """Padding frames marked with -1 should not contribute."""
    logits = torch.zeros(1, 4, 2)
    logits[0, 0, 0] = 1.0
    logits[0, 1, 1] = 1.0
    # frames 2 and 3 are padding
    labels = torch.tensor([[0, 1, -1, -1]], dtype=torch.long)
    counts = compute_state_counts(logits, labels)
    assert torch.equal(counts["tp"], torch.tensor([1, 1]))
    assert torch.equal(counts["fp"], torch.tensor([0, 0]))
    assert torch.equal(counts["fn"], torch.tensor([0, 0]))


def test_compute_state_metrics_basic():
    """Smoke test for the test-side single-batch convenience wrapper."""
    logits = torch.zeros(1, 3, 3)
    logits[0, 0, 0] = 1.0
    logits[0, 1, 1] = 1.0
    logits[0, 2, 2] = 1.0
    labels = torch.tensor([[0, 1, 2]], dtype=torch.long)
    metrics = _compute_state_metrics(logits, labels)
    assert metrics["state_accuracy"] == pytest.approx(1.0)
    assert metrics["state_macro_f1"] == pytest.approx(1.0)
    for c in range(3):
        assert metrics[f"state_f1_class_{c}"] == pytest.approx(1.0)


def test_state_metrics_from_counts_empty():
    """Zero counts should expose support without fabricating performance."""
    zeros = torch.zeros(3, dtype=torch.int64)
    metrics = state_metrics_from_counts(zeros, zeros, zeros)
    assert metrics == {
        "state_support_count": 0.0,
        "state_supported_class_count": 0.0,
        "state_support_class_0": 0.0,
        "state_support_class_1": 0.0,
        "state_support_class_2": 0.0,
    }


def test_state_metrics_reports_all_class_and_supported_class_macro_f1():
    tp = torch.tensor([2, 0, 0], dtype=torch.int64)
    fp = torch.tensor([0, 1, 0], dtype=torch.int64)
    fn = torch.zeros(3, dtype=torch.int64)

    metrics = state_metrics_from_counts(tp, fp, fn)

    assert metrics["state_macro_f1"] == pytest.approx(1.0 / 3.0)
    assert metrics["state_macro_f1_supported"] == pytest.approx(1.0)
    assert metrics["state_support_count"] == pytest.approx(2.0)
    assert metrics["state_supported_class_count"] == pytest.approx(1.0)
    assert metrics["state_support_class_0"] == pytest.approx(2.0)
    assert metrics["state_support_class_1"] == pytest.approx(0.0)


def test_compute_state_counts_rejects_time_dimension_mismatch():
    logits = torch.zeros(2, 4, 3)
    labels = torch.zeros(2, 3, dtype=torch.long)

    with pytest.raises(ValueError, match="time dimension mismatch"):
        compute_state_counts(logits, labels)


def _state_extra(tp, fp, fn, *, correct=0, valid=0):
    return {
        "state_tp": torch.tensor(tp, dtype=torch.int64),
        "state_fp": torch.tensor(fp, dtype=torch.int64),
        "state_fn": torch.tensor(fn, dtype=torch.int64),
        "state_correct": torch.tensor([correct], dtype=torch.int64),
        "state_valid": torch.tensor([valid], dtype=torch.int64),
        "state_metrics": None,
    }


def test_trainer_state_empty_global_support_does_not_record_performance():
    trainer = object.__new__(AECTrainer)
    trainer.world_size = 1
    extra = _state_extra([0, 0, 0], [0, 0, 0], [0, 0, 0])

    trainer._val_post_allreduce_extra(extra, val_x=1)

    assert extra["state_metrics"]["state_support_count"] == 0.0
    assert "state_accuracy" not in extra["state_metrics"]
    assert "state_macro_f1" not in extra["state_metrics"]


def test_trainer_state_ddp_empty_rank_uses_global_support(monkeypatch):
    trainer = object.__new__(AECTrainer)
    trainer.world_size = 2
    extra = _state_extra([0, 0, 0], [0, 0, 0], [0, 0, 0])
    remote = iter(
        [
            torch.tensor([2, 0, 0]),
            torch.tensor([0, 1, 0]),
            torch.tensor([0, 0, 0]),
            torch.tensor([2]),
            torch.tensor([2]),
        ]
    )

    def fake_all_reduce(tensor, op=None):
        del op
        tensor.add_(next(remote).to(dtype=tensor.dtype))

    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.dist.all_reduce", fake_all_reduce)
    trainer._val_post_allreduce_extra(extra, val_x=1)

    metrics = extra["state_metrics"]
    assert metrics["state_support_count"] == pytest.approx(2.0)
    assert metrics["state_accuracy"] == pytest.approx(1.0)
    assert metrics["state_macro_f1"] == pytest.approx(1.0 / 3.0)
    assert metrics["state_macro_f1_supported"] == pytest.approx(1.0)


def test_trainer_without_state_head_skips_state_metrics():
    trainer = object.__new__(AECTrainer)
    trainer.world_size = 2
    extra = {"state_tp": None, "state_metrics": None}

    trainer._val_post_allreduce_extra(extra, val_x=1)

    assert extra["state_metrics"] is None


pytestmark = [pytest.mark.unit]
