"""Third-party library wrapper losses (auraloss).

AuralossMultiResSTFTLoss, AuralossSDRLoss
"""

import logging

import torch
import torch.nn as nn
from core.utils.registry import LOSS_REGISTRY

logger = logging.getLogger(__name__)


@LOSS_REGISTRY.register("AuralossMultiResSTFTLoss")
class AuralossMultiResSTFTLoss(nn.Module):
    """Multi-resolution STFT loss from auraloss library.

    Source: auraloss (https://github.com/csteinmetz1/auraloss)

    Computes spectral convergence and log-magnitude distance across multiple
    STFT resolutions. Widely used in audio generation models.

    Args:
        fft_sizes: List of FFT sizes (default: [1024, 2048, 512])
        hop_sizes: List of hop sizes (default: [120, 240, 50])
        win_lengths: List of window lengths (default: [600, 1200, 240])
        w_sc: Weight for spectral convergence (default: 1.0)
        w_log_mag: Weight for log-magnitude distance (default: 1.0)
        w_lin_mag: Weight for linear magnitude distance (default: 0.0)

    Returns:
        Weighted sum of spectral convergence and magnitude distances
    """

    def __init__(
        self,
        fft_sizes: list[int] | None = None,
        hop_sizes: list[int] | None = None,
        win_lengths: list[int] | None = None,
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        allow_missing: bool = False,
    ):
        super().__init__()
        if fft_sizes is None:
            fft_sizes = [1024, 2048, 512]
        if hop_sizes is None:
            hop_sizes = [120, 240, 50]
        if win_lengths is None:
            win_lengths = [600, 1200, 240]
        self._missing_logged = False
        try:
            from auraloss.freq import MultiResolutionSTFTLoss
            self.loss_fn = MultiResolutionSTFTLoss(
                fft_sizes=fft_sizes,
                hop_sizes=hop_sizes,
                win_lengths=win_lengths,
                w_sc=w_sc,
                w_log_mag=w_log_mag,
                w_lin_mag=w_lin_mag,
            )
            self.available = True
        except ImportError as e:
            if allow_missing:
                logger.error(
                    "auraloss not installed; AuralossMultiResSTFTLoss disabled. "
                    "Install: pip install auraloss"
                )
                self.available = False
            else:
                raise RuntimeError(
                    "AuralossMultiResSTFTLoss requires auraloss. Install via: "
                    "pip install auraloss "
                    "or pass allow_missing=True via YAML args."
                ) from e

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor | None = None):
        if not self.available:
            # 每次实例调用时最多记一次 ERROR，避免训练热路径刷日志
            if not self._missing_logged:
                logger.error(
                    "AuralossMultiResSTFTLoss unavailable (auraloss missing); returning zero loss"
                )
                self._missing_logged = True
            return (enhanced.float() * 0.0).sum()

        # Auraloss expects (B, 1, T) for mono audio
        if enhanced.ndim == 2:
            enhanced = enhanced.unsqueeze(1)
            clean = clean.unsqueeze(1)

        # Apply length masking if provided
        if lengths is not None:
            mask = torch.arange(clean.shape[2], device=clean.device)[None, None, :] < lengths[:, None, None]
            enhanced = enhanced * mask
            clean = clean * mask

        loss = self.loss_fn(enhanced, clean)
        return loss


@LOSS_REGISTRY.register("AuralossSDRLoss")
class AuralossSDRLoss(nn.Module):
    """Signal-to-Distortion Ratio (SDR) loss from auraloss.

    Source: auraloss (https://github.com/csteinmetz1/auraloss)

    Computes negative SDR (Source-to-Distortion Ratio) for minimization.
    SDR measures the ratio of signal power to distortion power.

    Args:
        sdr_type: Type of SDR ('snr', 'sisdr', 'sdsdr') (default: 'sisdr')
        zero_mean: Whether to zero-mean signals before computing (default: True)

    Returns:
        Negative SDR value (minimize to maximize SDR)
    """

    REQUIRES_SPEECH_REFERENCE = True

    def __init__(self, sdr_type: str = "sisdr", zero_mean: bool = True, allow_missing: bool = False):
        super().__init__()
        self._missing_logged = False
        try:
            import auraloss
            loss_map = {
                "snr": auraloss.time.SNRLoss,
                "sisdr": auraloss.time.SISDRLoss,
                "sdsdr": auraloss.time.SDSDRLoss,
            }
            if sdr_type not in loss_map:
                raise ValueError(f"sdr_type must be one of {list(loss_map)}, got {sdr_type!r}")
            cls = loss_map[sdr_type]
            if sdr_type == "sdsdr":
                self.loss_fn = cls()
            else:
                self.loss_fn = cls(zero_mean=zero_mean)
            self.sdr_type = sdr_type
            self.available = True
        except ImportError as e:
            if allow_missing:
                logger.error(
                    "auraloss not installed; AuralossSDRLoss disabled. "
                    "Install: pip install auraloss"
                )
                self.available = False
            else:
                raise RuntimeError(
                    "AuralossSDRLoss requires auraloss. Install via: "
                    "pip install auraloss "
                    "or pass allow_missing=True via YAML args."
                ) from e

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor | None = None):
        if not self.available:
            # 每次实例调用时最多记一次 ERROR，避免训练热路径刷日志
            if not self._missing_logged:
                logger.error("AuralossSDRLoss unavailable (auraloss missing); returning zero loss")
                self._missing_logged = True
            return (enhanced.float() * 0.0).sum()

        # Auraloss expects (B, C, T), add channel dim if needed
        if enhanced.ndim == 2:
            enhanced = enhanced.unsqueeze(1)
            clean = clean.unsqueeze(1)

        # Apply length masking if provided
        if lengths is not None:
            mask = torch.arange(clean.shape[2], device=clean.device)[None, None, :] < lengths[:, None, None]
            enhanced = enhanced * mask
            clean = clean * mask

        loss = self.loss_fn(enhanced, clean)
        return loss
