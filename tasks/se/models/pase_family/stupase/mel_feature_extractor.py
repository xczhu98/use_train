"""Mel-spectrogram feature extractor for StuPASE's Mel-vocoder pretraining phase.

Ported from ``stupase/models/mel/feature_extractor.py`` in the source repo.
"""

import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram


class Mel_feat(nn.Module):
    """Normalize by the batch-global peak then extract a log-mel spectrogram.

    The peak is taken over the whole batch, matching official
    ``stupase/models/mel/feature_extractor.py`` (``x.abs().max()``), not
    per sample.

    Args:
        n_fft, hop_length, win_length: STFT parameters.
        n_mels: Number of mel bands.
        sample_rate: Input sample rate.
        f_min, f_max: Mel filterbank frequency range.
        power: Spectrogram power (1.0 = magnitude).
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 320,
        win_length: int = 1024,
        n_mels: int = 100,
        sample_rate: int = 16000,
        f_min: float = 0,
        f_max: float = None,
        power: float = 1.0,
    ):
        super().__init__()
        self.mel_spec = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=power,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, L) or (B, 1, L). Returns: (B, n_mels, T) log-mel.

        峰值归一化使用整个 batch 的 global max，与官方
        ``stupase/models/mel/feature_extractor.py:35`` 一致。
        """
        if x.dim() == 3:
            x = x.squeeze(1)

        x = x / (x.abs().max() + 1e-8) * 0.9

        feat = self.mel_spec(x)
        feat = torch.log(torch.clamp(feat, min=1e-7))

        return feat
