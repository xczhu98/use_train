"""ClearerVoice-Studio style spectral ops (framework ``(B, F, T, 2)`` tensors + Kaldi fbank)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def power_compress(x: torch.Tensor) -> torch.Tensor:
    """(B, F, T, 2) -> (B, 2, F, T) compressed real/imag channels (ClearerVoice ``power_compress``)."""
    real, imag = x[..., 0], x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_c = mag * torch.cos(phase)
    imag_c = mag * torch.sin(phase)
    return torch.stack([real_c, imag_c], dim=1)


def power_uncompress(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    """Compressed ``real``/``imag`` tensors -> (..., 2) last dim [Re, Im] (inverse of ``power_compress``)."""
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1.0 / 0.3)
    real_u = mag * torch.cos(phase)
    imag_u = mag * torch.sin(phase)
    return torch.stack([real_u, imag_u], dim=-1)


def kaldi_fbank_plus_deltas(
    wav_1d: torch.Tensor,
    *,
    sampling_rate: int,
    win_len_samples: int,
    hop_samples: int,
    num_mels: int,
    win_type: str,
    dither: float = 1.0,
) -> torch.Tensor:
    """ClearerVoice ``compute_fbank`` + Δ + ΔΔ: (T_wav,) float -> (T_frame, 3 * num_mels).

    Uses ``torchaudio.compliance.kaldi.fbank`` (mel filterbank + log energy path per Kaldi recipe).
    ``win_len_samples`` / ``hop_samples`` match framework ``win_length`` / ``hop_length`` in samples.
    """
    import torchaudio

    frame_length = win_len_samples / float(sampling_rate) * 1000.0
    frame_shift = hop_samples / float(sampling_rate) * 1000.0
    wt = (win_type or "hamming").lower().replace("hanning", "hann")
    fb = torchaudio.compliance.kaldi.fbank(
        wav_1d.unsqueeze(0),
        dither=dither,
        frame_length=frame_length,
        frame_shift=frame_shift,
        num_mel_bins=num_mels,
        sample_frequency=float(sampling_rate),
        window_type=wt,
    )
    fbank_tr = fb.transpose(0, 1)
    d1 = torchaudio.functional.compute_deltas(fbank_tr)
    d2 = torchaudio.functional.compute_deltas(d1)
    d1 = d1.transpose(0, 1)
    d2 = d2.transpose(0, 1)
    return torch.cat([fb, d1, d2], dim=1)


def interpolate_mask_time(x: torch.Tensor, target_t: int) -> torch.Tensor:
    """``x`` (B, C, T) -> (B, C, target_t) linear interpolation along last dim."""
    if x.size(-1) == target_t:
        return x
    b, c, t0 = x.shape
    x2 = x.reshape(b * c, 1, t0)
    y = F.interpolate(x2, size=target_t, mode="linear", align_corners=False)
    return y.reshape(b, c, target_t)
