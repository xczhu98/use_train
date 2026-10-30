"""基于目标空间协方差的 MVDR oracle 评估基线。"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch


def _batched_channel_first(value, *, name: str) -> tuple[torch.Tensor, bool]:
    tensor = torch.as_tensor(value)
    if tensor.is_complex() or tensor.ndim not in (2, 3):
        raise ValueError(f"{name} must be a real [C,T] or [B,C,T] waveform")
    if tensor.shape[-2] < 1 or tensor.shape[-1] < 1:
        raise ValueError(f"{name} must have positive channel and time dimensions")
    tensor = tensor.to(dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    unbatched = tensor.ndim == 2
    return (tensor.unsqueeze(0) if unbatched else tensor), unbatched


def _window(name: str, win_length: int, *, device: torch.device) -> torch.Tensor:
    if name == "hann":
        return torch.hann_window(win_length, periodic=True, device=device)
    if name == "hamming":
        return torch.hamming_window(win_length, periodic=True, device=device)
    raise ValueError("window must be 'hann' or 'hamming'")


def oracle_mvdr(
    mixture,
    target,
    *,
    n_fft: int = 512,
    hop_length: int | None = None,
    win_length: int | None = None,
    reference_channel: int = 0,
    diagonal_loading: float = 1e-6,
    window: Literal["hann", "hamming"] = "hann",
) -> torch.Tensor:
    """Estimate target steering from a clean multichannel image and apply MVDR.

    Args:
        mixture: Multichannel observation with shape ``[C,T]`` or ``[B,C,T]``.
        target: Clean target image with the same multichannel shape as ``mixture``.
        n_fft: Framework-compatible STFT size used for covariance estimation.
        hop_length: STFT hop; defaults to ``n_fft // 2``.
        win_length: Window length; defaults to ``n_fft``.
        reference_channel: Channel whose target image defines the distortionless output.
        diagonal_loading: Relative identity loading after trace normalization.
        window: Periodic Hann or Hamming window.

    Returns:
        Enhanced mono waveform, shape ``[T]`` or ``[B,T]`` matching input batching.
    """
    mixture, was_unbatched = _batched_channel_first(mixture, name="mixture")
    target, target_unbatched = _batched_channel_first(target, name="target")
    if was_unbatched != target_unbatched or mixture.shape != target.shape:
        raise ValueError(
            f"target shape {tuple(target.shape)} must match mixture {tuple(mixture.shape)}"
        )
    if type(n_fft) is not int or n_fft < 2:
        raise ValueError("n_fft must be an exact integer >= 2")
    hop_length = n_fft // 2 if hop_length is None else hop_length
    win_length = n_fft if win_length is None else win_length
    if type(hop_length) is not int or not 1 <= hop_length <= n_fft:
        raise ValueError("hop_length must be an exact integer in [1,n_fft]")
    if type(win_length) is not int or not 1 <= win_length <= n_fft:
        raise ValueError("win_length must be an exact integer in [1,n_fft]")
    if type(reference_channel) is not int or not 0 <= reference_channel < mixture.shape[1]:
        raise ValueError("reference_channel must be an exact int in [0,C)")
    if (
        isinstance(diagonal_loading, bool)
        or not isinstance(diagonal_loading, (int, float, np.number))
        or not np.isfinite(diagonal_loading)
        or diagonal_loading <= 0.0
    ):
        raise ValueError("diagonal_loading must be a positive finite number")

    batch, channels, samples = mixture.shape
    window_tensor = _window(window, win_length, device=mixture.device)
    mixture_spectrum = torch.stft(
        mixture.reshape(batch * channels, samples),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_tensor,
        center=True,
        pad_mode="constant",
        return_complex=True,
    ).reshape(batch, channels, n_fft // 2 + 1, -1)
    target_spectrum = torch.stft(
        target.reshape(batch * channels, samples),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_tensor,
        center=True,
        pad_mode="constant",
        return_complex=True,
    ).reshape(batch, channels, n_fft // 2 + 1, -1)

    frame_count = mixture_spectrum.shape[-1]
    target_covariance = torch.einsum(
        "bcft,bdft->bfcd", target_spectrum, target_spectrum.conj()
    ) / max(frame_count, 1)
    target_covariance = (target_covariance + target_covariance.mH) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(target_covariance)
    steering = eigenvectors[..., -1]
    reference_response = steering[..., reference_channel]
    target_power = target_covariance.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1)
    valid_steering = (target_power > torch.finfo(target_power.dtype).eps) & (
        reference_response.abs() > torch.finfo(reference_response.real.dtype).eps
    )
    safe_reference = torch.where(
        valid_steering, reference_response, torch.ones_like(reference_response)
    )
    steering = steering / safe_reference.unsqueeze(-1)
    fallback_steering = torch.ones_like(steering)
    steering = torch.where(valid_steering.unsqueeze(-1), steering, fallback_steering)

    mixture_covariance = torch.einsum(
        "bcft,bdft->bfcd", mixture_spectrum, mixture_spectrum.conj()
    ) / max(frame_count, 1)
    mixture_covariance = (mixture_covariance + mixture_covariance.mH) * 0.5
    trace = mixture_covariance.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1)
    normalizer = trace.clamp_min(torch.finfo(trace.dtype).eps)
    covariance = mixture_covariance / normalizer[..., None, None]
    identity = torch.eye(channels, dtype=covariance.dtype, device=covariance.device)
    covariance = (covariance + float(diagonal_loading) * identity / channels) / (
        1.0 + float(diagonal_loading)
    )

    solved = torch.linalg.solve(covariance, steering.unsqueeze(-1)).squeeze(-1)
    denominator = torch.sum(steering.conj() * solved, dim=-1)
    if not torch.isfinite(denominator).all() or (denominator.abs() <= 1e-12).any():
        raise ValueError("MVDR covariance produced an invalid distortionless denominator")
    weights = solved / denominator.unsqueeze(-1)
    enhanced_spectrum = torch.einsum(
        "bfc,bcft->bft", weights.conj(), mixture_spectrum
    )
    enhanced = torch.istft(
        enhanced_spectrum,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_tensor,
        center=True,
        length=samples,
    )
    if not torch.isfinite(enhanced).all():
        raise ValueError("MVDR output contains NaN or Inf")
    return enhanced[0] if was_unbatched else enhanced


__all__ = ["oracle_mvdr"]
