"""Frame-level state detection cross-entropy loss for AEC models.

The current project supervision scheme is formally named **frame-level state CE
engineering mode**.  It provides the mutually-exclusive cross-entropy loss used
by DBGNet's auxiliary state detector and is distinct from the original paper's
segment-level ``L_SD`` with temporal pooling.  It supports the 3-class
FST / DT / NST view as well as the project-specific 8-class scenario labels
derived from ``[near_present, echo_present, noise_present]`` presence bits.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.registry import LOSS_REGISTRY


# 只接受有符号整型：ignore_index 是负数 sentinel，无符号 dtype 既无法表达它，
# 也会把 uint8 的 255 之类 padding 编码误判为「非法标签」而非 ignore。
_INTEGER_DTYPES = frozenset(
    dtype
    for name in ("int8", "int16", "int32", "int64")
    if (dtype := getattr(torch, name, None)) is not None
)


@LOSS_REGISTRY.register("StateCrossEntropyLoss")
class StateCrossEntropyLoss(nn.Module):
    """Cross-entropy loss for frame-level AEC state classification.

    Expects ``state_probs`` of shape ``(B, T, C)`` (class logits, **not**
    softmax probabilities) and ``state`` of shape ``(B, T)`` (class indices),
    with exactly matching batch and time dimensions. Padding frames are marked
    with ``ignore_index`` and do not contribute to the loss. ``C`` may be 3
    (FST/DT/NST), 8 (project-specific scenario classes), or any other positive
    number of classes.

    Args:
        ignore_index: Label value to ignore (default ``-1`` for padded frames).
        label_smoothing: Optional label smoothing for the CE target. Must be
            finite and satisfy ``0 <= label_smoothing < 1``.

    ``state`` 必须是 **有符号** 整型：``ignore_index`` 是负数 sentinel，无符号
    dtype 无法表达它，且会把 uint8 padding 编码误判为非法标签。
    """

    def __init__(self, ignore_index: int = -1, label_smoothing: float = 0.0):
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)
        if not math.isfinite(self.label_smoothing) or not (
            0.0 <= self.label_smoothing < 1.0
        ):
            raise ValueError(
                "label_smoothing must be finite and satisfy 0 <= x < 1, "
                f"got {self.label_smoothing}"
            )

    def _validate_inputs(
        self,
        state_probs: torch.Tensor | None,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(state_probs, torch.Tensor):
            raise TypeError("state_probs must be present and be a Tensor")
        if not isinstance(state, torch.Tensor):
            raise TypeError("state must be present and be a Tensor")
        if state_probs.dim() != 3:
            raise ValueError(
                f"state_probs must be (B, T, C), got {tuple(state_probs.shape)}"
            )
        if not state_probs.is_floating_point():
            raise TypeError(
                f"state_probs must have a floating dtype, got {state_probs.dtype}"
            )
        if state_probs.shape[2] <= 0:
            raise ValueError("state_probs class dimension C must be positive")
        if state.dim() != 2:
            raise ValueError(f"state must be (B, T), got {tuple(state.shape)}")
        if state.dtype not in _INTEGER_DTYPES:
            raise TypeError(
                f"state must have a signed integer dtype, got {state.dtype}"
            )
        if state.shape != state_probs.shape[:2]:
            raise ValueError(
                "state_probs and state must have the same batch/time dimensions, "
                f"got {tuple(state_probs.shape[:2])} and {tuple(state.shape)}"
            )
        if state.device != state_probs.device:
            raise ValueError(
                "state_probs and state must be on the same device, "
                f"got {state_probs.device} and {state.device}"
            )

        valid = state != self.ignore_index
        state_long = state.to(dtype=torch.long)
        invalid = valid & ((state_long < 0) | (state_long >= state_probs.shape[2]))
        if bool(invalid.any()):
            raise ValueError("non-ignore state labels must satisfy 0 <= label < C")
        return state_probs, state_long, valid

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor | None = None,
        state_probs: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute state classification loss.

        ``enhanced`` and ``clean`` are ignored; they are present only to match the
        generic loss signature used by ``CompositeLoss``.
        """
        state_probs, state, valid = self._validate_inputs(state_probs, state)
        if not bool(valid.any()):
            # 空有效选择不读取 logits 数值；sum 仍保留 autograd/DDP 连接，
            # 即使 logits 含 NaN/Inf 也返回精确有限零。
            return state_probs[valid].sum()
        return F.cross_entropy(
            state_probs.transpose(1, 2),  # (B, C, T)
            state.long(),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )

    def ddp_global_mean_mass(
        self,
        enhanced: torch.Tensor | None = None,
        clean: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        state_probs: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return the local valid-frame count used by the forward mean."""
        _, _, valid = self._validate_inputs(state_probs, state)
        return valid.sum(dtype=torch.float32)
