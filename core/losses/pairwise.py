"""Speech Separation 的 pairwise source cost。"""

from __future__ import annotations

import torch
import torch.nn as nn

from core.utils.registry import LOSS_REGISTRY


def _time_mask(lengths: torch.Tensor | None, batch: int, time: int, device) -> torch.Tensor:
    if lengths is None:
        return torch.ones(batch, 1, 1, time, dtype=torch.bool, device=device)
    if not isinstance(lengths, torch.Tensor) or lengths.shape != (batch,):
        raise ValueError("lengths must have shape [B]")
    index = torch.arange(time, device=device).view(1, 1, 1, time)
    return index < lengths.view(batch, 1, 1, 1)


@LOSS_REGISTRY.register("PairwiseL1Loss")
class PairwiseL1Loss(nn.Module):
    """返回 ``[B, K_est, K_ref]`` 的逐 source L1 cost。"""

    def forward(self, estimated, reference, *, lengths=None):
        if estimated.ndim != 3 or reference.ndim != 3:
            raise ValueError("estimated/reference must have shape [B, K, T]")
        if estimated.shape[0] != reference.shape[0] or estimated.shape[-1] != reference.shape[-1]:
            raise ValueError("estimated/reference batch and time dimensions must match")
        mask = _time_mask(lengths, estimated.shape[0], estimated.shape[-1], estimated.device)
        error = (estimated[:, :, None, :] - reference[:, None, :, :]).abs()
        denominator = mask.sum(dim=-1).clamp_min(1).to(error.dtype)
        return (error * mask).sum(dim=-1) / denominator


@LOSS_REGISTRY.register("PairwiseNegSISDRLoss")
class PairwiseNegSISDRLoss(nn.Module):
    """返回 ``[B, K_est, K_ref]`` cost；静音参考使用输出均方能量抑制。

    SI-SDR 对零参考没有定义。静音分支保留预测的 DC 与有效长度，只最小化
    输出能量，避免固定 80 dB cost 和零梯度。
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, estimated, reference, *, lengths=None):
        if estimated.ndim != 3 or reference.ndim != 3:
            raise ValueError("estimated/reference must have shape [B, K, T]")
        if estimated.shape[0] != reference.shape[0] or estimated.shape[-1] != reference.shape[-1]:
            raise ValueError("estimated/reference batch and time dimensions must match")
        mask = _time_mask(lengths, estimated.shape[0], estimated.shape[-1], estimated.device)
        mask_f = mask.to(estimated.dtype)
        count = mask_f.sum(dim=-1).clamp_min(1.0)
        est = estimated[:, :, None, :] * mask_f
        ref = reference[:, None, :, :] * mask_f
        silent_reference = ref.square().sum(dim=-1) / count <= self.eps
        silence_cost = est.square().sum(dim=-1) / count
        est = (est - est.sum(dim=-1, keepdim=True) / count.unsqueeze(-1)) * mask_f
        ref = (ref - ref.sum(dim=-1, keepdim=True) / count.unsqueeze(-1)) * mask_f
        ref_energy = ref.square().sum(dim=-1, keepdim=True)
        projection = (est * ref).sum(dim=-1, keepdim=True) * ref / (ref_energy + self.eps)
        noise = est - projection
        ratio = projection.square().sum(dim=-1) / noise.square().sum(dim=-1).clamp_min(self.eps)
        return torch.where(
            silent_reference,
            silence_cost,
            -10.0 * torch.log10(ratio.clamp_min(self.eps)),
        )
