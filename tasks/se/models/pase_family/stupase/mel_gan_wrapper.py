"""StuPASE Phase-1 (StuP1) Mel-vocoder GAN generator.

Wraps ``Mel_feat`` (parameter-free log-mel extraction) + trainable
``VocosVocoder``. Mirrors source ``stupase/train/train_melvocoder.py``'s
generator step: ``feat = encoder(wav); esti_wav = generator(feat)`` -- an
unconditional (clean-to-clean) vocoder-reconstruction autoencoder, not
enhancement. Used as the ``generator`` in a ``PASEGANTrainer`` config where
the data pipeline sets ``noisy == clean`` (``simulation_cfg.prob_target_only:
1.0``), so ``PASEGANTrainer._forward_generator``'s ``self.generator(noisy,
lengths, fs=fs)`` call effectively reconstructs the same signal it discriminates
against, matching source's ``true_wav`` usage exactly. ``fs`` 是 batch 实际
采样率（None 视为 16000）：非 16k 输入先重采样再走 Mel 前端。
"""

import torch
import torch.nn as nn
import torchaudio

from core.utils.registry import MODEL_REGISTRY

from ..common.utils import align_waveform_length, resample_waveform_and_lengths
from .mel_feature_extractor import Mel_feat
from .vocos_vocoder import VocosVocoder


@MODEL_REGISTRY.register("StuPASEMelVocoderGenerator")
class StuPASEMelVocoderGenerator(nn.Module):
    """Mel-feature extraction (fixed) + Vocos vocoder (trainable).

    ``output_sample_rate = 16000``：多采样率训练时 PASEGANTrainer 据此把
    clean/lengths 对齐到 16k，再计算重建/判别损失。
    """

    output_sample_rate = 16000
    val_audio_sample_rate = 16000

    # validation 验证音频：官方 train_melvocoder.py:360-369（StuPASE Phase 1）。
    val_audio_stage = "vocoder"
    val_audio_vocoder_roles = ()

    def __init__(
        self,
        mel_kwargs: dict = None,
        input_channels: int = 100,
        dim: int = 768,
        intermediate_dim: int = 2304,
        num_layers: int = 12,
        num_res: int = 4,
        num_attn: int = 1,
        n_fft: int = 1280,
        hop_length: int = 320,
        **kwargs,
    ):
        super().__init__()
        for key in ("win_length", "window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"StuPASEMelVocoderGenerator got unexpected keyword arguments: {sorted(kwargs)}")
        self.encoder = Mel_feat(**(mel_kwargs or {}))
        self.vocoder = VocosVocoder(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            num_res=num_res,
            num_attn=num_attn,
            n_fft=n_fft,
            hop_length=hop_length,
        )

    def forward(
        self,
        wav: torch.Tensor,
        lengths: torch.Tensor = None,
        *,
        fs=None,
    ) -> torch.Tensor:
        """Args: wav (B, L)，采样率 ``fs``（None 视为 16000）。

        Returns: (B, L') 16000 波形，L' 与重采样后的输入等长。多采样率训练时
        先按 batch 实际采样率 ``fs`` 重采样到 16000，再走 Mel_feat + vocoder；
        ``fs == 16000`` 时不做重采样，行为与单采样率路径一致。
        """
        sr = 16000 if fs is None else int(fs)
        wav, _ = resample_waveform_and_lengths(wav, None, sr, 16000)
        n_samples = wav.shape[-1]
        feat = self.encoder(wav)
        esti = self.vocoder(feat)
        return align_waveform_length(esti, n_samples)

    def val_audio_outputs(
        self,
        noisy_wav: torch.Tensor,
        clean_wav: torch.Tensor,
        lengths: torch.Tensor = None,
        *,
        fs=None,
        vocoders=None,
        sample_settings=None,
        forward_dict=None,
        raw_enhanced: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """官方 StuPASE Phase-1 验证音频：``{uid}_true.wav`` + ``{uid}_esti_epochNNN.wav``。"""
        if raw_enhanced is None:
            raise ValueError(
                "StuPASEMelVocoderGenerator val audio requires the validation forward output"
            )
        sr = int(fs) if fs is not None else self.val_audio_sample_rate
        clean = clean_wav
        if sr != self.val_audio_sample_rate:
            clean = torchaudio.functional.resample(
                clean_wav, orig_freq=sr, new_freq=self.val_audio_sample_rate
            )
        return {
            "true": clean,
            "esti_epoch{epoch:03d}": raw_enhanced,
        }
