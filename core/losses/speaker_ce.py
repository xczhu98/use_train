"""Utterance-level speaker classification cross-entropy loss for TSE models.

TSE multi-task 辅助任务：模型在句级 speaker embedding 后接分类头输出
``spk_logits``，对本批目标说话人索引
（``TSEDynamicMixingDataset.speaker_identity_to_idx``）
做 CE；未知说话人（label=-1，如占位样本或 identity 缺失）经 ``ignore_index``
跳过。模式照 ``core/losses/state_ce.py`` 的 StateCrossEntropyLoss（AEC
state detector 的帧级 CE），分量调用约定（CompositeLoss kwargs 透传）一致。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("SpeakerCrossEntropyLoss")
class SpeakerCrossEntropyLoss(nn.Module):
    """Cross-entropy loss for utterance-level speaker classification (TSE multi-task).

    Expects ``spk_logits`` of shape ``(B, C)`` (class logits, **not** softmax
    probabilities) and ``spk_labels`` of shape ``(B,)`` (int64 类别索引，
    ``ignore_index`` = 未知说话人）。两者缺失时 fail-fast，避免非零 CE 分量
    静默返回 detached zero、令分类头断开梯度。整批均为未知说话人时返回不读取
    logits 数值、但保留 autograd/DDP 连接的精确有限 0（包括 logits 含 NaN/Inf）。

    Args:
        ignore_index: Label value to ignore (default ``-1`` = 未知说话人).
        label_smoothing: Optional label smoothing for the CE target.
    """

    def __init__(self, ignore_index: int = -1, label_smoothing: float = 0.0):
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)
        if not math.isfinite(self.label_smoothing) or not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be finite and satisfy 0 <= x < 1")

    def _validate_inputs(
        self,
        spk_logits: torch.Tensor | None,
        spk_labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """统一校验 forward 与 DDP 全局均值分母所需的分类输入。"""
        missing = [
            name
            for name, value in (
                ("spk_logits", spk_logits),
                ("spk_labels", spk_labels),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "SpeakerCrossEntropyLoss requires "
                f"{', '.join(missing)}; a non-zero speaker CE component cannot "
                "run without them"
            )
        if not isinstance(spk_logits, torch.Tensor):
            raise TypeError(
                f"spk_logits must be a Tensor, got {type(spk_logits).__name__}"
            )
        if not isinstance(spk_labels, torch.Tensor):
            raise TypeError(
                f"spk_labels must be a Tensor, got {type(spk_labels).__name__}"
            )
        if spk_logits.dim() != 2:
            raise ValueError(
                f"spk_logits must be (B, C), got {tuple(spk_logits.shape)}"
            )
        if spk_logits.shape[1] <= 0:
            raise ValueError(
                f"spk_logits must have C > 0, got shape {tuple(spk_logits.shape)}"
            )
        if not spk_logits.is_floating_point():
            raise TypeError(
                f"spk_logits must have a floating dtype, got {spk_logits.dtype}"
            )
        if spk_labels.dim() != 1 or spk_labels.shape[0] != spk_logits.shape[0]:
            raise ValueError(
                f"spk_labels must be (B,) with B={spk_logits.shape[0]}, "
                f"got {tuple(spk_labels.shape)}"
            )
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if spk_labels.dtype not in integer_dtypes:
            raise TypeError(
                f"spk_labels must have an integer dtype, got {spk_labels.dtype}"
            )
        if spk_labels.device != spk_logits.device:
            raise ValueError(
                f"spk_labels device {spk_labels.device} must match "
                f"spk_logits device {spk_logits.device}"
            )

        labels = spk_labels.long()
        valid = labels != self.ignore_index
        valid_labels = labels[valid]
        if valid_labels.numel() > 0:
            # min/max 合并后一次传回 CPU，避免 CUDA 上逐 scalar ``item()`` 同步。
            min_label, max_label = torch.stack(torch.aminmax(valid_labels)).cpu().tolist()
            num_classes = spk_logits.shape[1]
            if min_label < 0 or max_label >= num_classes:
                raise ValueError(
                    f"spk_labels valid label range {min_label}..{max_label} is outside "
                    f"[0, C) for C={num_classes} (ignore_index={self.ignore_index})"
                )
        return spk_logits, labels, valid

    def ddp_global_mean_mass(
        self,
        spk_logits: torch.Tensor | None = None,
        spk_labels: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """返回与 CE mean 分母一致的本地有效 utterance 数。"""
        logits, _, valid = self._validate_inputs(spk_logits, spk_labels)
        return valid.sum(dtype=torch.float32).to(device=logits.device)

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor | None = None,
        spk_logits: torch.Tensor | None = None,
        spk_labels: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute speaker classification loss.

        ``enhanced`` / ``clean`` / ``lengths`` 均被忽略；它们存在仅为匹配
        ``CompositeLoss`` 的通用分量签名（照 StateCrossEntropyLoss 先例）。
        """
        logits, labels, valid = self._validate_inputs(spk_logits, spk_labels)
        if not bool(valid.any()):
            # 空有效选择不读取 logits 数值；sum 仍保留 autograd/DDP 连接。
            return logits[valid].sum()
        return F.cross_entropy(
            logits[valid],
            labels[valid],
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )
