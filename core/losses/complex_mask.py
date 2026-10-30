"""Complex ideal ratio mask regression loss for spectrum-output SE models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.losses.utils import create_frame_mask
from core.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("ComplexIdealRatioMaskLoss")
class ComplexIdealRatioMaskLoss(nn.Module):
    """MSE between compressed ideal and output-derived complex ratio masks."""

    SUPPORTS_DIRECT_SPECTRA = True
    REQUIRES_NOISY_REFERENCE = True
    REQUIRES_MATCHING_STFT_CONTEXT = True
    SUPPORTS_SS_UNDESIRED_ENERGY = False

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        reference_channel: int = 0,
        compression_k: float = 10.0,
        compression_c: float = 0.1,
        eps: float = 1.1920928955078125e-7,
    ) -> None:
        super().__init__()
        if type(n_fft) is not int or n_fft < 2:
            raise ValueError("n_fft must be an exact integer >= 2")
        if type(hop_length) is not int or not 1 <= hop_length <= n_fft:
            raise ValueError("hop_length must be an exact integer in [1,n_fft]")
        if type(win_length) is not int or not 1 <= win_length <= n_fft:
            raise ValueError("win_length must be an exact integer in [1,n_fft]")
        if window not in {"hann", "hamming"}:
            raise ValueError("window must be 'hann' or 'hamming'")
        if type(center) is not bool or type(normalized) is not bool:
            raise TypeError("center and normalized must be bool")
        if type(reference_channel) is not int or reference_channel < 0:
            raise ValueError("reference_channel must be a non-negative exact int")
        for name, value in (("compression_k", compression_k), ("compression_c", compression_c), ("eps", eps)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window_name = window
        self.center = center
        self.normalized = normalized
        self.reference_channel = reference_channel
        self.compression_k = float(compression_k)
        self.compression_c = float(compression_c)
        self.eps = float(eps)
        if window == "hann":
            window_tensor = torch.hann_window(win_length)
        else:
            window_tensor = torch.hamming_window(win_length)
        self.register_buffer("_window", window_tensor, persistent=False)

    @staticmethod
    def _complex_spectrum(value: torch.Tensor, *, name: str) -> torch.Tensor:
        if value.is_complex():
            return value.to(dtype=torch.complex64)
        if value.ndim < 1 or value.shape[-1] != 2:
            raise ValueError(f"{name} must be complex or end in a real/imag axis of length 2")
        return torch.view_as_complex(value.float().contiguous())

    def _compressed_mask(self, clean: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        numerator = clean * mixture.conj()
        denominator = mixture.abs().square() + self.eps
        ratio = numerator / denominator
        components = torch.view_as_real(ratio).float().clamp(min=-100.0)
        compressed = self.compression_k * torch.tanh(
            0.5 * self.compression_c * components
        )
        return compressed

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        enhanced_spectra: torch.Tensor | None = None,
        target_spectra: torch.Tensor | None = None,
        noisy: torch.Tensor | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        del enhanced
        if enhanced_spectra is None or target_spectra is None:
            raise ValueError(
                "ComplexIdealRatioMaskLoss requires enhanced_spectra and target_spectra"
            )
        if noisy is None or not torch.is_tensor(noisy):
            raise ValueError("ComplexIdealRatioMaskLoss requires a noisy waveform reference")
        if not torch.is_tensor(clean) or clean.ndim != 2:
            raise ValueError("clean must have shape [B,T]")
        if noisy.ndim == 2:
            noisy_ref = noisy
        elif noisy.ndim == 3:
            if self.reference_channel >= noisy.shape[1]:
                raise ValueError("reference_channel is outside noisy microphone channels")
            noisy_ref = noisy[:, self.reference_channel, :]
        else:
            raise ValueError("noisy must have shape [B,T] or [B,C,T]")
        if noisy_ref.shape != clean.shape:
            raise ValueError(
                f"noisy reference/clean shape mismatch: {tuple(noisy_ref.shape)} != "
                f"{tuple(clean.shape)}"
            )

        predicted = self._complex_spectrum(enhanced_spectra, name="enhanced_spectra")
        target = self._complex_spectrum(target_spectra, name="target_spectra")
        if predicted.shape != target.shape or predicted.ndim != 3:
            raise ValueError(
                "direct spectra must share [B,F,T] shape, got "
                f"{tuple(predicted.shape)} and {tuple(target.shape)}"
            )
        if predicted.shape[0] != clean.shape[0]:
            raise ValueError("direct spectrum batch dimension must match clean waveform")

        resolved_lengths = None
        noisy_input = noisy_ref.float()
        if lengths is not None:
            if (
                not torch.is_tensor(lengths)
                or lengths.ndim != 1
                or lengths.shape[0] != noisy_input.shape[0]
                or lengths.dtype not in (
                    torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
                )
            ):
                raise ValueError("lengths must be an integer [B] tensor")
            resolved_lengths = lengths.to(device=noisy_input.device, dtype=torch.long)
            if not bool(((resolved_lengths >= 1) & (resolved_lengths <= noisy_input.shape[-1])).all()):
                raise ValueError("lengths must lie in [1,T]")
            samples = torch.arange(noisy_input.shape[-1], device=noisy_input.device)
            noisy_input = noisy_input * (samples[None, :] < resolved_lengths[:, None])
        if not self.center:
            minimum_frames = max(noisy_input.shape[-1], self.n_fft)
            padded_length = self.n_fft + (
                (minimum_frames - self.n_fft + self.hop_length - 1) // self.hop_length
            ) * self.hop_length
            if padded_length > noisy_input.shape[-1]:
                noisy_input = F.pad(noisy_input, (0, padded_length - noisy_input.shape[-1]))
        window = self._window.to(device=noisy_input.device, dtype=noisy_input.dtype)
        mixture_spectrum = torch.stft(
            noisy_input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            return_complex=True,
            onesided=True,
        )
        if mixture_spectrum.shape != predicted.shape:
            raise ValueError(
                "mixture and direct model spectra must have matching [B,F,T] geometry: "
                f"{tuple(mixture_spectrum.shape)} != {tuple(predicted.shape)}"
            )

        predicted_mask = self._compressed_mask(predicted, mixture_spectrum)
        target_mask = self._compressed_mask(target, mixture_spectrum)
        error = (predicted_mask - target_mask).square()
        if resolved_lengths is None:
            return error.mean()
        frame_mask = create_frame_mask(
            resolved_lengths,
            predicted.shape[-1],
            self.n_fft,
            self.win_length,
            self.hop_length,
            predicted.device,
            center=self.center,
        ).unsqueeze(1).unsqueeze(-1).to(error.dtype)
        denominator = (
            frame_mask.sum(dim=(1, 2, 3)) * predicted.shape[1] * 2
        ).clamp(min=1.0)
        per_sample = (error * frame_mask).sum(dim=(1, 2, 3)) / denominator
        return per_sample.mean()


__all__ = ["ComplexIdealRatioMaskLoss"]
