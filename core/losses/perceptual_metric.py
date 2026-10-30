"""Psychoacoustic metric-based losses (require external dependencies).

PESQLoss, STOILoss
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from core.utils.registry import LOSS_REGISTRY

logger = logging.getLogger(__name__)


# =============================================================================
# PESQ Loss (Differentiable)
# =============================================================================
@LOSS_REGISTRY.register("PESQLoss")
class PESQLoss(nn.Module):
    """Differentiable PESQ loss (negative form for minimization).

    Source: pytorch-pesq (https://github.com/ludlows/pytorch-pesq)

    PESQ (Perceptual Evaluation of Speech Quality) is a standard metric for
    speech quality assessment. This implementation provides a differentiable
    approximation suitable for training.

    Args:
        sample_rate: Audio sample rate (8000 or 16000 Hz)
        mode: 'wb' (wideband, 16kHz) or 'nb' (narrowband, 8kHz)

    Returns:
        Negative PESQ score (minimize to maximize PESQ)
        Range: PESQ is typically 1.0-4.5, so loss is -4.5 to -1.0
    """

    # 显式 capability：子类化后仍保留 pure metadata 门控，不依赖类名 fallback
    REQUIRES_SPEECH_REFERENCE = True

    def __init__(self, sample_rate: int = 16000, mode: str = "wb", allow_missing: bool = False):
        super().__init__()
        self._missing_logged = False
        try:
            from pytorch_pesq import PesqLoss as _PESQLoss
            self.pesq_fn = _PESQLoss(sample_rate, mode)
            self.available = True
        except ImportError as e:
            if allow_missing:
                logger.error(
                    "pytorch-pesq not installed; PESQLoss disabled. "
                    "Install: pip install pytorch-pesq"
                )
                self.available = False
            else:
                raise RuntimeError(
                    "PESQLoss requires pytorch-pesq. Install via: "
                    "pip install pytorch-pesq "
                    "or pass allow_missing=True via YAML args."
                ) from e

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor | None = None):
        if not self.available:
            # 每次实例调用时最多记一次 ERROR，避免训练热路径刷日志
            if not self._missing_logged:
                logger.error("PESQLoss unavailable (pytorch-pesq missing); returning zero loss")
                self._missing_logged = True
            return (enhanced.float() * 0.0).sum()

        # pytorch-pesq expects (B, T) normalized to [-1, 1]
        # Apply length masking if provided
        if lengths is not None:
            mask = torch.arange(clean.shape[1], device=clean.device)[None, :] < lengths[:, None]
            enhanced = enhanced * mask
            clean = clean * mask

        # PESQ loss returns negative PESQ score (higher PESQ = lower loss)
        loss = self.pesq_fn(enhanced, clean)
        return loss


# =============================================================================
# STOI Loss (Differentiable)
# =============================================================================
@LOSS_REGISTRY.register("STOILoss")
class STOILoss(nn.Module):
    """Differentiable STOI loss (negative form for minimization).

    Source: differentiable_stoi (https://github.com/etzints/differentiable_stoi)

    STOI (Short-Time Objective Intelligibility) measures speech intelligibility.
    This differentiable version can be used as a training objective.

    Args:
        sample_rate: Audio sample rate (default: 16000 Hz)
        use_vad: Whether to apply voice activity detection (default: False)

    Returns:
        Negative STOI score (minimize to maximize STOI)
        Range: STOI is 0-1, so loss is -1 to 0
    """

    # 显式 capability：子类化后仍保留 pure metadata 门控，不依赖类名 fallback
    REQUIRES_SPEECH_REFERENCE = True

    def __init__(self, sample_rate: int = 16000, use_vad: bool = False, allow_missing: bool = False):
        super().__init__()
        self.sample_rate = sample_rate
        self.use_vad = use_vad
        self._missing_logged = False
        try:
            from torch_stoi import NegSTOILoss
            self.stoi_fn = NegSTOILoss(sample_rate=sample_rate, use_vad=use_vad)
            self.available = True
        except ImportError as e:
            if allow_missing:
                logger.error(
                    "torch-stoi not installed; STOILoss disabled. "
                    "Install: pip install pystoi torch-stoi"
                )
                self.available = False
            else:
                raise RuntimeError(
                    "STOILoss requires torch-stoi. Install via: "
                    "pip install pystoi torch-stoi "
                    "or pass allow_missing=True via YAML args."
                ) from e

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor | None = None):
        if not self.available:
            # 每次实例调用时最多记一次 ERROR，避免训练热路径刷日志
            if not self._missing_logged:
                logger.error("STOILoss unavailable (torch-stoi missing); returning zero loss")
                self._missing_logged = True
            return (enhanced.float() * 0.0).sum()

        # Apply length masking if provided
        if lengths is not None:
            mask = torch.arange(clean.shape[1], device=clean.device)[None, :] < lengths[:, None]
            enhanced = enhanced * mask
            clean = clean * mask

        # NegSTOILoss already returns negative STOI
        loss = self.stoi_fn(enhanced, clean)
        return loss


