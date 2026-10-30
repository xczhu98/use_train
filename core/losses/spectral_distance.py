"""Simple spectral distance losses (no perceptual model, basic STFT math).

SpectralConvergenceLoss, LogMagnitudeSTFTLoss
"""

import torch
import torch.nn as nn
from core.losses.utils import create_frame_mask, prepare_centered_stft_inputs
from core.utils.registry import LOSS_REGISTRY


def _centered_stft_magnitudes(
    enhanced: torch.Tensor,
    clean: torch.Tensor,
    lengths: torch.Tensor | None,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 trim 等价幅度谱，并返回共享的有效帧 mask。"""
    (enhanced, clean), lengths, _ = prepare_centered_stft_inputs(
        enhanced, clean, lengths=lengths, n_fft=n_fft
    )
    kwargs = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "window": window,
        "return_complex": True,
        "center": True,
    }
    enhanced_spec = torch.stft(enhanced.float(), **kwargs).abs()
    clean_spec = torch.stft(clean.float(), **kwargs).abs()
    frame_mask = create_frame_mask(
        lengths,
        clean_spec.shape[-1],
        n_fft,
        win_length,
        hop_length,
        clean_spec.device,
    ).unsqueeze(1).to(clean_spec.dtype)
    return enhanced_spec, clean_spec, frame_mask


def _masked_log_magnitude_error(
    enhanced_spec: torch.Tensor,
    clean_spec: torch.Tensor,
    frame_mask: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """返回逐样本有效帧 log-magnitude L1。"""
    error = (
        torch.log(enhanced_spec + eps) - torch.log(clean_spec + eps)
    ).abs() * frame_mask
    denominator = (
        frame_mask.sum(dim=(1, 2)) * enhanced_spec.shape[1]
    ).clamp_min(1.0)
    return error.sum(dim=(1, 2)) / denominator


@LOSS_REGISTRY.register("SpectralConvergenceLoss")
class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss (simplified, no external dependencies).

    Measures the spectral distance between enhanced and clean spectrograms.

    Formula: SC = ||X_clean - X_enhanced||_F / ||X_clean||_F
    where X is the magnitude spectrogram and ||·||_F is Frobenius norm.
    For a per-sample raw target norm at or below 1e-8, SC falls back to the
    valid-frame log-magnitude L1 used by ``LogMagnitudeSTFTLoss``. This matches
    the MR-STFT zero-target strategy without dividing by a near-zero norm.

    Args:
        n_fft: FFT size (default: 1024)
        hop_length: Hop length (default: 256)
        win_length: Window length (default: 1024)

    Returns:
        Non-negative spectral distance (lower is better)
    """

    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window, persistent=False)

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor | None = None):
        enhanced_spec, clean_spec, frame_mask = _centered_stft_magnitudes(
            enhanced,
            clean,
            lengths,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )
        raw_target_norm = torch.linalg.vector_norm(
            clean_spec * frame_mask, dim=(1, 2)
        )
        enhanced_spec = enhanced_spec.clamp_min(1e-7)
        clean_spec = clean_spec.clamp_min(1e-7)
        numerator = torch.linalg.vector_norm(
            (clean_spec - enhanced_spec) * frame_mask, dim=(1, 2)
        )
        target_norm = torch.linalg.vector_norm(
            clean_spec * frame_mask, dim=(1, 2)
        )
        valid_target = raw_target_norm > 1e-8
        spectral_convergence = numerator / torch.where(
            valid_target, target_norm, torch.ones_like(target_norm)
        )
        log_magnitude = _masked_log_magnitude_error(
            enhanced_spec, clean_spec, frame_mask, eps=1e-8
        )
        return torch.where(
            valid_target, spectral_convergence, log_magnitude
        ).mean()


@LOSS_REGISTRY.register("LogMagnitudeSTFTLoss")
class LogMagnitudeSTFTLoss(nn.Module):
    """Log-magnitude STFT distance (simplified, no external dependencies).

    Measures L1 distance between log-magnitude spectrograms.

    Formula: LogMag = mean(|log(|X_clean|+eps) - log(|X_enhanced|+eps)|)

    Args:
        n_fft: FFT size (default: 1024)
        hop_length: Hop length (default: 256)
        win_length: Window length (default: 1024)
        eps: Small constant for numerical stability (default: 1e-8)

    Returns:
        Log-magnitude distance
    """

    def __init__(
        self, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024, eps: float = 1e-8
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.eps = eps
        window = torch.hann_window(win_length)
        self.register_buffer("window", window, persistent=False)

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor | None = None):
        enhanced_spec, clean_spec, frame_mask = _centered_stft_magnitudes(
            enhanced,
            clean,
            lengths,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )
        enhanced_spec = enhanced_spec.clamp_min(1e-7)
        clean_spec = clean_spec.clamp_min(1e-7)
        return _masked_log_magnitude_error(
            enhanced_spec, clean_spec, frame_mask, eps=self.eps
        ).mean()
