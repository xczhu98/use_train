"""时域波形损失。

  SIL1Loss         — 尺度不变 L1；近静音 clean 样本不参与平均
  SIL1WavSeTrainLoss — se_train 配方 SI-L1（零参考分母 0.1）
  SISNRLoss        — 负 SI-SNR（dB）
  L1WavLoss        — 普通 MAE

接口: forward(enhanced, clean, lengths=None) -> 标量；enhanced/clean: (B,T)；lengths: (B,) 可选
"""

import torch
import torch.nn as nn
from core.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("SIL1WavSeTrainLoss")
class SIL1WavSeTrainLoss(nn.Module):
    """Scale-invariant L1 waveform (se_train recipe). For CompositeLoss with MR losses."""

    def forward(
        self, enhanced: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        if lengths is not None:
            t = enhanced.shape[1]
            t_idx = torch.arange(t, device=enhanced.device).unsqueeze(0)
            mask = (t_idx < lengths.unsqueeze(1).clamp(min=0)).to(enhanced.dtype)
            n = mask.sum(dim=1).clamp(min=1.0)
            diff = (torch.abs(clean - enhanced) * mask).sum(dim=1) / n
            ref = (torch.abs(clean) * mask).sum(dim=1) / n
            return torch.where(
                ref > 1e-12,
                10.0 * diff / (ref + 1e-8),
                10.0 * diff / (0.1 + 1e-8),
            ).mean()
        diff = torch.mean(torch.abs(clean - enhanced), dim=1)
        ref = torch.mean(torch.abs(clean), dim=1)
        num = torch.where(ref > 1e-12, 10.0 * diff / (ref + 1e-8), 10.0 * diff / (0.1 + 1e-8))
        return num.mean()


@LOSS_REGISTRY.register("SIL1Loss")
class SIL1Loss(nn.Module):
    """Scale-invariant L1 waveform loss (Scale-Invariant L1).

    Formula: SI-L1 = 10 * mean(|clean - enhanced|) / mean(|clean|)
    Source: se_train, avoids excessive gradient for small-magnitude signals, more stable than pure L1.
    """
    EPS = 1e-4  # threshold below which clean is considered silent

    def forward(self, enhanced, clean, lengths=None):
        if lengths is not None:
            mask = torch.arange(clean.shape[1], device=clean.device)[None, :] < lengths[:, None]
            n = mask.sum(dim=1).float().clamp(min=1)
            diff = (torch.abs(clean - enhanced) * mask).sum(dim=1) / n
            ref = (torch.abs(clean) * mask).sum(dim=1) / n
        else:
            diff = torch.mean(torch.abs(clean - enhanced), dim=1)
            ref = torch.mean(torch.abs(clean), dim=1)
        # Guard against near-silent clean signals: 10*diff/(ref≈0) → inf.
        # Skip samples where mean |clean| < EPS (same strategy as SISNRLoss).
        valid = ref > self.EPS
        loss = 10.0 * diff / ref.clamp(min=self.EPS)
        # 掩码归约无需 valid.any() 分支：全无效 batch 同样得到 graph-connected 零，
        # 避免每个 micro-batch 一次 host sync（与 hybrid.py 同一范式）。
        valid_f = valid.to(loss.dtype)
        return (loss * valid_f).sum() / valid_f.sum().clamp(min=1.0)


@LOSS_REGISTRY.register("SISNRLoss")
class SISNRLoss(nn.Module):
    """Scale-invariant SNR loss (negative form for minimization).

    Formula: SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    where s_target = (<clean, enhanced> / ||clean||^2) * clean  (optimal scale projection)
          e_noise  = enhanced - s_target
    Returns -SI-SNR, minimizing this loss equals maximizing SI-SNR.
    """
    EPS = 1e-8
    # 显式 capability：子类化后仍保留 pure metadata 门控，不依赖类名 fallback
    REQUIRES_SPEECH_REFERENCE = True

    def forward(self, enhanced, clean, lengths=None):
        # Standard SI-SNR requires zero-mean signals
        if lengths is not None:
            mask = torch.arange(clean.shape[1], device=clean.device)[None, :] < lengths[:, None]
            clean, enhanced = clean * mask, enhanced * mask
            # 零均值只在有效样点上计算(均值 = sum / n_valid), 减均值后再乘
            # mask, padding 区严格为 0 (与 hybrid.py 同一 JAE-H02 范式)
            n_valid = lengths.to(clean.dtype).unsqueeze(-1).clamp(min=1.0)
            mask_f = mask.to(clean.dtype)
            clean = (clean - clean.sum(dim=-1, keepdim=True) / n_valid) * mask_f
            enhanced = (enhanced - enhanced.sum(dim=-1, keepdim=True) / n_valid) * mask_f
        else:
            clean = clean - clean.mean(dim=-1, keepdim=True)
            enhanced = enhanced - enhanced.mean(dim=-1, keepdim=True)
        clean_energy = torch.sum(clean ** 2, dim=-1, keepdim=True)
        # Guard against silent/near-zero clean: return 0 loss for those samples
        valid = (clean_energy.squeeze(-1) > self.EPS)
        s_target = (torch.sum(clean * enhanced, dim=-1, keepdim=True)
                    / (clean_energy + self.EPS)) * clean
        e_noise = enhanced - s_target
        snr = 10 * torch.log10(
            torch.sum(s_target ** 2, dim=-1).clamp(min=self.EPS)
            / torch.sum(e_noise ** 2, dim=-1).clamp(min=self.EPS)
        )
        snr = snr.clamp(max=80.0)  # upper bound: avoid extreme SNR dominating loss
        # 掩码归约无需 valid.any() 分支：全无效 batch 同样得到 graph-connected 零，
        # 避免每个 micro-batch 一次 host sync（与 hybrid.py 同一范式）。
        valid_f = valid.to(snr.dtype)
        return -(snr * valid_f).sum() / valid_f.sum().clamp(min=1.0)


@LOSS_REGISTRY.register("L1WavLoss")
class L1WavLoss(nn.Module):
    """L1 waveform loss (Mean Absolute Error on waveform).

    Simplest time-domain loss, direct mean of per-sample absolute error.
    """
    def forward(self, enhanced, clean, lengths=None):
        if lengths is not None:
            mask = torch.arange(clean.shape[1], device=clean.device)[None, :] < lengths[:, None]
            return (torch.abs(enhanced - clean) * mask).sum() / mask.sum().clamp(min=1)
        return torch.mean(torch.abs(enhanced - clean))


@LOSS_REGISTRY.register("AbsentTargetLoss")
class AbsentTargetLoss(nn.Module):
    """仅对零目标样本约束输出能量；全 present batch 保留连接计算图的零。"""

    SUPPORTS_SILENT_TARGET = True

    def ddp_global_mean_mass(self, enhanced, clean, lengths=None, **kwargs):
        """返回本 batch 的 absent-target 数，供 Composite 全局归约。"""
        del enhanced, kwargs
        clean = clean.float()
        mask = torch.ones_like(clean)
        if lengths is not None:
            mask = (
                torch.arange(clean.shape[-1], device=clean.device)[None, :]
                < lengths[:, None]
            ).to(clean.dtype)
        count = mask.sum(dim=-1).clamp(min=1)
        absent = (((clean.square() * mask).sum(dim=-1) / count) <= 1e-10).to(clean.dtype)
        return absent.sum()

    def forward(self, enhanced, clean, lengths=None):
        clean, enhanced = clean.float(), enhanced.float()
        mask = torch.ones_like(clean)
        if lengths is not None:
            mask = (torch.arange(clean.shape[-1], device=clean.device)[None, :] < lengths[:, None]).to(clean.dtype)
        count = mask.sum(dim=-1).clamp(min=1)
        absent = (((clean.square() * mask).sum(dim=-1) / count) <= 1e-10).to(clean.dtype)
        energy = (enhanced.square() * mask).sum(dim=-1) / count
        return (energy * absent).sum() / absent.sum().clamp(min=1)
