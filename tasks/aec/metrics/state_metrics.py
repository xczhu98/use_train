"""Frame-level state classification metrics for AEC state detectors.

These helpers compute per-class and macro-averaged precision/recall/F1 from
frame-wise class logits and labels.  They are designed to be accumulated across
batches and all-reduced across DDP ranks before the final metric is computed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def compute_state_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -1,
    num_classes: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Return per-class true/false positive/negative counts.

    Args:
        logits: Class logits of shape ``(B, T, C)``.
        labels: Ground-truth class indices of shape ``(B, T)``.
        ignore_index: Label value to ignore (typically ``-1`` for padded frames).
        num_classes: Number of classes; inferred from ``logits`` if omitted.

    Returns:
        A dict with keys ``tp``, ``fp``, ``fn`` mapping to 1-D tensors of shape
        ``(C,)``.  These counts can be accumulated over batches and all-reduced
        across DDP ranks before calling ``state_metrics_from_counts``.
    """
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("state logits and labels must be torch.Tensor")
    if not logits.is_floating_point() or logits.is_complex():
        raise TypeError(f"state logits must use a real floating dtype, got {logits.dtype}")
    if labels.is_floating_point() or labels.is_complex():
        raise TypeError(f"state labels must use an integer dtype, got {labels.dtype}")
    if logits.device != labels.device:
        raise ValueError(f"state logits/labels device mismatch: {logits.device} != {labels.device}")
    if not torch.isfinite(logits).all():
        raise ValueError("state logits contain NaN or Inf")
    if logits.ndim != 3:
        raise ValueError(f"state logits must have shape (B, T, C), got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"state labels must have shape (B, T), got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "state logits/labels batch or time dimension mismatch: "
            f"logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
        )
    inferred_classes = int(logits.shape[-1])
    if num_classes is None:
        num_classes = inferred_classes
    elif int(num_classes) != inferred_classes:
        raise ValueError(
            f"state num_classes={num_classes} != logits classes={inferred_classes}"
        )
    preds = logits.argmax(dim=-1)  # (B, T)
    valid_mask = labels != ignore_index
    if not bool(valid_mask.any()):
        zeros = torch.zeros(num_classes, dtype=torch.int64, device=logits.device)
        return {"tp": zeros, "fp": zeros, "fn": zeros}

    valid_preds = preds[valid_mask]
    valid_labels = labels[valid_mask]
    if bool(((valid_labels < 0) | (valid_labels >= num_classes)).any()):
        raise ValueError(
            f"state labels must be in [0, {num_classes - 1}] or equal ignore_index={ignore_index}"
        )
    pred_oh = F.one_hot(valid_preds, num_classes=num_classes).float()
    label_oh = F.one_hot(valid_labels, num_classes=num_classes).float()
    tp = (pred_oh * label_oh).sum(dim=0).to(torch.int64)
    fp = (pred_oh * (1.0 - label_oh)).sum(dim=0).to(torch.int64)
    fn = ((1.0 - pred_oh) * label_oh).sum(dim=0).to(torch.int64)
    return {"tp": tp, "fp": fp, "fn": fn}


def state_metrics_from_counts(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
) -> dict[str, float]:
    """Convert accumulated per-class counts into metric scalars.

    Args:
        tp, fp, fn: 1-D tensors of shape ``(C,)``.

    Returns:
        Dict with per-class support counts.  When support is non-zero, it also
        contains per-class precision/recall/F1, the historical all-class
        ``state_macro_f1``, and ``state_macro_f1_supported`` averaged only over
        classes present in the labels.  Empty counts intentionally expose only
        support metadata rather than fabricated zero-valued performance.
    """
    if not all(isinstance(value, torch.Tensor) for value in (tp, fp, fn)):
        raise TypeError("state tp/fp/fn must be torch.Tensor")
    if tp.ndim != 1 or fp.shape != tp.shape or fn.shape != tp.shape:
        raise ValueError(
            "state tp/fp/fn must be 1-D tensors with identical shapes, got "
            f"tp={tuple(tp.shape)}, fp={tuple(fp.shape)}, fn={tuple(fn.shape)}"
        )
    if tp.device != fp.device or tp.device != fn.device:
        raise ValueError("state tp/fp/fn must be on the same device")
    if any(value.is_floating_point() or value.is_complex() for value in (tp, fp, fn)):
        raise TypeError("state tp/fp/fn must use integer dtypes")
    if any(bool((value < 0).any()) for value in (tp, fp, fn)):
        raise ValueError("state tp/fp/fn must be non-negative")
    support = tp + fn
    support_count = int(support.sum().item())
    supported = support > 0
    metrics: dict[str, float] = {
        "state_support_count": float(support_count),
        "state_supported_class_count": float(supported.sum().item()),
    }
    for i in range(int(tp.shape[0])):
        metrics[f"state_support_class_{i}"] = float(support[i].item())
    if support_count == 0:
        return metrics

    eps = 1e-12
    precision = tp.float() / (tp + fp + eps).float()
    recall = tp.float() / (tp + fn + eps).float()
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    metrics["state_macro_f1"] = f1.mean().item()
    metrics["state_macro_f1_supported"] = f1[supported].mean().item()
    for i in range(int(tp.shape[0])):
        metrics[f"state_precision_class_{i}"] = precision[i].item()
        metrics[f"state_recall_class_{i}"] = recall[i].item()
        metrics[f"state_f1_class_{i}"] = f1[i].item()
    return metrics
