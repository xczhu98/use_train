"""PASE (Phonologically Anchored Speech Enhancer) model for use_train framework.

A generative SE model consisting of:
  1. DeWavLM (WavLM encoder fine-tuned via distillation)
  2. Dual-stream Vocoder (phonetic + acoustic -> waveform)

This file registers ``PASE`` as a model type and ``PASEDiscriminator`` as
a discriminator type for Phase-3 GAN training.
"""

import logging
from typing import Union, List

import torch
import torch.nn as nn
import torchaudio

from tasks.se.models.base_model import BaseSEModel
from core.utils.registry import MODEL_REGISTRY

from ..common.wavlm_frontend import WavLM_feat
from ..common.vocoder_backend import WavLMDec
from ..common.discriminator import CombinedDiscriminator
from ..common.utils import align_waveform_length, scale_waveform_to_reference_peak

logger = logging.getLogger(__name__)


@MODEL_REGISTRY.register("PASE")
class PASE(BaseSEModel):
    """PASE generative speech enhancement model.

    Args:
        wavlm_ckpt_path: Path to WavLM-Large.pt (or DeWavLM checkpoint).
        output_layer: Layer(s) to extract, e.g., [1, 24] for dual-stream.
        vocoder_ckpt_path: Optional path to pretrained vocoder (Phase 3 init).
        input_channels: WavLM feature dim (1024 for Large).
        vocoder_dim: Vocoder backbone hidden dim.
        vocoder_intermediate_dim: ConvNeXt intermediate dim.
        vocoder_num_layers: Number of ConvNeXt blocks.
        n_fft: ISTFT FFT size.
        hop_length: ISTFT hop length (should match WavLM downsampling).
        frozen_encoder: If True, freeze DeWavLM during vocoder training.
            Defaults to True, matching the reference PASE Phase-3 behaviour
            (frozen encoder, trainable decoder). Set to ``False`` for Phase-2
            DeWavLM distillation (where the encoder itself is the training
            target) or for end-to-end fine-tuning.
    """

    input_domain = "waveform"
    output_domain = "waveform"
    output_sample_rate = 16000
    forward_kind = "estimate"
    frontend_owner = "model"

    # validation 验证音频：PASE P1（单流 Vocoder）与 P3（双流 Vocoder GAN）。
    val_audio_sample_rate = 16000
    val_audio_vocoder_roles = ()

    @property
    def val_audio_stage(self) -> str:
        """单流 Phase 1 用 ``vocoder``；双流 Phase 3 用 ``pase_dual``。"""
        return "pase_dual" if self.use_dual_stream else "vocoder"


    def __init__(
        self,
        wavlm_ckpt_path: str = "",
        output_layer: Union[int, List[int]] = None,  # R-07: 默认 [1,24] 双流自洽（对齐官方 dewavlm_output_layer=[1,24]）；用 None 避免可变默认参数
        vocoder_ckpt_path: str = "",
        input_channels: int = 1024,
        vocoder_dim: int = 768,
        vocoder_intermediate_dim: int = 2304,
        vocoder_num_layers: int = 12,
        n_fft: int = 1280,
        hop_length: int = 320,
        frozen_encoder: bool = True,
        frozen_decoder: bool = False,
        use_dual_stream: bool = True,
        allow_random_init: bool = False,
        feature_norm: str = "batch",
        **kwargs,
    ):
        super().__init__()
        for key in ("win_length", "window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"PASE got unexpected keyword arguments: {sorted(kwargs)}")
        if output_layer is None:
            output_layer = [1, 24]  # R-07: 默认双流自洽（对齐官方 dewavlm_output_layer=[1,24]）
        if use_dual_stream and not (isinstance(output_layer, list) and len(output_layer) == 2):
            raise ValueError(
                "use_dual_stream=True requires output_layer to be a 2-element list "
                f"(e.g. [1, 24]), got {output_layer!r}. Single-stream (Phase 1) configs "
                "must pass use_dual_stream=False with an int output_layer."
            )
        if not use_dual_stream and isinstance(output_layer, list) and len(output_layer) > 1:
            raise ValueError(
                "use_dual_stream=False requires a single output_layer (int or 1-element list), "
                f"got {output_layer!r}: a single-stream decoder would silently drop the "
                "acoustic branch. Use use_dual_stream=True for multi-layer output."
            )
        if not allow_random_init and (not wavlm_ckpt_path or not vocoder_ckpt_path):
            raise ValueError(
                "PASE inference requires wavlm_ckpt_path and vocoder_ckpt_path; "
                "set allow_random_init=True only for explicit training initialization"
            )
        self.output_layer = output_layer
        self.use_dual_stream = bool(use_dual_stream)
        self.sample_rate = 16000  # PASE operates at 16kHz
        self._artifacts_ready = bool(wavlm_ckpt_path and vocoder_ckpt_path)
        self._allow_random_init = allow_random_init

        # Encoder: DeWavLM (initialized from WavLM-Large, then fine-tuned in Phase 2)
        self.encoder = WavLM_feat(
            wavlm_ckpt_path=wavlm_ckpt_path,
            output_layer=output_layer,
            load_pretrained=bool(wavlm_ckpt_path),
            frozen=frozen_encoder,
            feature_norm=feature_norm,
        )

        # Decoder: load from checkpoint if provided, otherwise construct from args
        if vocoder_ckpt_path:
            self.decoder = WavLMDec.from_pretrained(
                vocoder_ckpt_path,
                frozen=frozen_decoder,
                use_dual_stream=use_dual_stream,
            )
            logger.info("[PASE] Loaded vocoder weights from %s", vocoder_ckpt_path)
        else:
            self.decoder = WavLMDec(
                input_channels=input_channels,
                dim=vocoder_dim,
                intermediate_dim=vocoder_intermediate_dim,
                num_layers=vocoder_num_layers,
                n_fft=n_fft,
                hop_length=hop_length,
                use_dual_stream=use_dual_stream,
                frozen=frozen_decoder,
            )

    def forward(
        self,
        noisy_wav: torch.Tensor,
        lengths: torch.Tensor = None,
        *,
        fs=None,
        normalize_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            noisy_wav: (B, L) or (B, 1, L) waveform at 16kHz.
            lengths: (B,) valid lengths (optional).
            fs: sample rate (optional, should be 16000).
            normalize_output: 推理时把输出峰值缩放到输入峰值（官方 inference.py）；
                训练默认 False，避免改变重建/判别目标。

        Returns:
            enhanced_wav: (B, L') enhanced waveform at 16kHz.
        """
        # Guard：无 artifact 且未声明训练初始化时禁止 forward（防部署误用随机权重）。
        # allow_random_init=True 是显式的“从头训练”声明：训练期 validation 会把模型
        # 切到 eval 再 forward（GAN trainer 的 val loss/指标都依赖这条路径），
        # 因此不能再用 self.training 区分——否则 Phase-1/Phase-3 在第一个 validation
        # epoch（val_metrics.enabled=false 也会跑 val loss）必然炸。
        if not self._artifacts_ready and not self._allow_random_init:
            raise RuntimeError("PASE cannot run inference with randomly initialized encoder or vocoder")
        n_samples = noisy_wav.shape[-1]
        input_sr = int(fs) if fs is not None else self.sample_rate
        if input_sr <= 0:
            raise ValueError(f"PASE fs must be positive, got {fs!r}")
        out_samples = (
            n_samples
            if input_sr == self.sample_rate
            else int(round(n_samples * self.sample_rate / float(input_sr)))
        )

        # PASE operates at 16kHz; warn if the caller passes a different rate
        if input_sr != self.sample_rate:
            logger.warning(
                "[PASE] Received fs=%d, but PASE expects %d Hz. "
                "The encoder will resample and output length follows %d Hz.",
                input_sr, self.sample_rate, self.sample_rate,
            )

        # Extract features from DeWavLM (encoder handles dim & sr internally)
        feats = self.encoder(noisy_wav, sr=fs, lengths=lengths)  # single tensor or list of tensors

        if isinstance(feats, list):
            feat_p, feat_a = feats[1], feats[0]  # [1, 24]: 0=layer1(acoustic), 1=layer24(phonetic)
            enhanced = self.decoder(feat_p, feat_a)
        else:
            # Single-stream (Phase 1)
            enhanced = self.decoder(feats)

        # Length alignment: pad or crop to output-rate sample count
        enhanced = align_waveform_length(enhanced, out_samples)
        if normalize_output:
            enhanced = scale_waveform_to_reference_peak(enhanced, noisy_wav)
        return enhanced

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
        """官方 P1/P3 验证音频。

        - P1（单流，``phase1_vocoder_single.yaml``，官方 ``train_vocoder.py:356-363``）：
          ``{uid}_true.wav`` + ``{uid}_esti_epochNNN.wav``；
        - P3（双流，``phase3_vocoder_dual_gan.yaml``，官方
          ``train_vocoder_dual.py:359-372``）：``{uid}_noisy.wav`` +
          ``{uid}_clean.wav`` + ``{uid}_enh_epochNNN.wav``。

        所有输出统一为 16 kHz；多采样率 batch 的 noisy/clean 先重采样到 16 kHz。
        """
        if raw_enhanced is None:
            raise ValueError("PASE val audio requires the validation forward output")
        sr = int(fs) if fs is not None else self.sample_rate
        noisy = noisy_wav
        clean = clean_wav
        if sr != self.val_audio_sample_rate:
            noisy = torchaudio.functional.resample(
                noisy, orig_freq=sr, new_freq=self.val_audio_sample_rate
            )
            clean = torchaudio.functional.resample(
                clean, orig_freq=sr, new_freq=self.val_audio_sample_rate
            )
        if self.use_dual_stream:
            return {
                "noisy": noisy,
                "clean": clean,
                "enh_epoch{epoch:03d}": raw_enhanced,
            }
        return {
            "true": clean,
            "esti_epoch{epoch:03d}": raw_enhanced,
        }



@MODEL_REGISTRY.register("PASEDiscriminator")
class PASEDiscriminator(nn.Module):
    """Combined MPD + MBD discriminator for PASE Phase-3 training."""

    def __init__(
        self,
        mpd_reshapes: List[int] = None,
        mbd_fft_sizes: List[int] = None,
        channel_mult: int = 1,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.discriminator = CombinedDiscriminator(
            mpd_reshapes=mpd_reshapes,
            mbd_fft_sizes=mbd_fft_sizes,
            channel_mult=channel_mult,
            use_spectral_norm=use_spectral_norm,
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        return self.discriminator(y, y_hat)
