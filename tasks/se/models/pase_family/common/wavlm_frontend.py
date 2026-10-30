"""DeWavLM: WavLM-based frontend for PASE with layer-selection and distillation support.

This module provides:
  - WavLM_feat: a wrapper around the fairseq-style WavLM model.
  - Support for loading official WavLM-Large checkpoints.
  - Multi-layer feature extraction (e.g., layer 1 for acoustic, layer 24 for phonetic).
  - Teacher/Student mode for Phase-2 representation distillation.

Reference:
  - PASE: Leveraging the Phonological Prior of WavLM for Low-Hallucination
    Generative Speech Enhancement (AAAI 2026)
  - WavLM: https://github.com/microsoft/unilm/tree/master/wavlm
"""

import logging
from pathlib import Path
from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from core.utils.safe_load import safe_torch_load

from .wavlm.WavLM import WavLM, WavLMConfig

logger = logging.getLogger(__name__)


class WavLM_feat(nn.Module):
    """WavLM feature extractor with configurable output layers.

    Args:
        wavlm_ckpt_path: Path to the official WavLM checkpoint (e.g., WavLM-Large.pt).
        output_layer: Which transformer layer(s) to return. int for single layer,
            List[int] for multiple layers (e.g., [1, 24] for dual-stream).
        load_pretrained: Whether to load weights from ``wavlm_ckpt_path``.
        frozen: If True, set requires_grad=False for all WavLM parameters.
        feature_norm: ``per_sample`` 按每个样本有效帧归一（UniPASE/GAP 官方口径）；
            ``batch`` 按 ``F.layer_norm(feat, feat.shape)`` 整批共享统计量
            （PASE/StuPASE 官方口径）。
    """

    def __init__(
        self,
        wavlm_ckpt_path: str = "",
        output_layer: Union[int, List[int]] = 24,
        load_pretrained: bool = True,
        frozen: bool = True,
        feature_norm: str = "per_sample",
    ):
        super().__init__()
        if feature_norm not in ("per_sample", "batch"):
            raise ValueError(
                f"feature_norm must be 'per_sample' or 'batch', got {feature_norm!r}"
            )
        self.feature_norm = feature_norm
        self.wavlm_ckpt_path = str(Path(wavlm_ckpt_path).expanduser()) if wavlm_ckpt_path else ""
        self.output_layer = output_layer
        self.load_pretrained = load_pretrained
        self._frozen = frozen

        # Build model
        if load_pretrained and self.wavlm_ckpt_path:
            cpt = safe_torch_load(self.wavlm_ckpt_path, map_location="cpu")
            self.cfg = WavLMConfig(cpt.get("cfg", {}))
            self.wavlm = WavLM(self.cfg)
            self.wavlm.load_state_dict(cpt["model"], strict=True)
            logger.info("[WavLM] Loaded pretrained weights from %s", self.wavlm_ckpt_path)
        else:
            # Build with default Large config if no checkpoint provided
            self.cfg = WavLMConfig()
            # Override to WavLM-Large defaults
            self.cfg.encoder_layers = 24
            self.cfg.encoder_embed_dim = 1024
            self.cfg.encoder_ffn_embed_dim = 4096
            self.cfg.encoder_attention_heads = 16
            self.cfg.layer_norm_first = True
            self.cfg.gru_rel_pos = True
            self.cfg.relative_position_embedding = True
            self.wavlm = WavLM(self.cfg)
            logger.info("[WavLM] Initialized from scratch (Large config)")

        if frozen:
            self.wavlm.eval()
            for p in self.wavlm.parameters():
                p.requires_grad = False

        logger.info("[WavLM] output_layer=%s, embed_dim=%d", output_layer, self.cfg.encoder_embed_dim)

    def train(self, mode: bool = True):
        """Override ``train()`` to keep a frozen WavLM in eval mode.

        ``BaseTrainer`` 会在每个 epoch 调用 ``model.train()``。冻结时必须将
        ``self.wavlm`` 保持在 eval，避免 dropout/LayerDrop 使前向非确定。
        """
        super().train(mode)
        if self._frozen:
            self.wavlm.eval()
        return self

    @staticmethod
    def pad_to_multiple(x: torch.Tensor, multiple: int = 320, offset: int = 80) -> torch.Tensor:
        """Replicate official ``pad``: crop when the next grid point is behind.

        Official PASE/StuPASE/UniPASE ``pad`` computes
        ``L//320*320 + 80 - L``; for ``L % 320 > 80`` this is negative and
        ``F.pad`` truncates the tail. The resulting length is always
        ``L//320*320 + 80``. We keep the official behavior for ``L >= 400``
        and retain the defensive 400-sample minimum for shorter inputs, where
        the official formula would crop below WavLM's conv receptive field.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        L = x.shape[-1]
        if L < 400:
            target_length = 400
        elif L % multiple == offset:
            return x
        else:
            target_length = L // multiple * multiple + offset
        if target_length > L:
            return F.pad(x, (0, target_length - L))
        return x[..., :target_length]

    def forward(
        self,
        wav: torch.Tensor,
        sr: int = None,
        mask: bool = False,
        mask_indices=None,
        lengths: torch.Tensor | None = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Extract WavLM features without allowing right padding to affect samples.

        ``lengths`` 表示输入采样率下的有效波形长度。前端将其换算到 16 kHz，
        构造并传递 WavLM waveform padding mask；随后只对每个样本的有效帧做
        LayerNorm，避免同批样本、排列或右侧 padding 改变表示。
        """
        if wav.ndim == 3:
            if wav.shape[1] != 1:
                raise ValueError(f"Expected (B, 1, L) waveform, got {tuple(wav.shape)}")
            wav = wav.squeeze(1)
        if wav.ndim != 2 or not wav.is_floating_point():
            raise TypeError("WavLM input must be a floating-point tensor shaped (B, L)")
        batch, input_length = wav.shape
        if lengths is None:
            lengths = torch.full((batch,), input_length, device=wav.device, dtype=torch.long)
        else:
            lengths = torch.as_tensor(lengths, device=wav.device, dtype=torch.long)
            if lengths.shape != (batch,) or (lengths <= 0).any() or (lengths > input_length).any():
                raise ValueError(f"lengths must be within [1, {input_length}] for each batch sample")
        time_index = torch.arange(input_length, device=wav.device)
        wav = wav.masked_fill(time_index.unsqueeze(0) >= lengths.unsqueeze(1), 0.0)
        if sr is not None and sr != 16000:
            if sr <= 0:
                raise ValueError(f"sr must be positive, got {sr}")
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
            lengths = torch.div(lengths * 16000 + sr - 1, sr, rounding_mode="floor")
        wav = self.pad_to_multiple(wav)
        waveform_padding_mask = (
            torch.arange(wav.shape[-1], device=wav.device).unsqueeze(0) >= lengths.unsqueeze(1)
        )

        max_layer = self.output_layer if isinstance(self.output_layer, int) else max(self.output_layer)
        res = self.wavlm.extract_features(
            wav,
            padding_mask=waveform_padding_mask,
            output_layer=max_layer,
            mask=mask,
            mask_indices=mask_indices,
        )[0]
        layer_reps = res["layer_reps"]  # List[Tensor], each (B, T, D)
        feature_padding_mask = res["padding_mask"]

        if isinstance(self.output_layer, int) or len(self.output_layer) == 1:
            return self.normalize_features(
                layer_reps[max_layer], feature_padding_mask
            )
        return [
            self.normalize_features(layer_reps[idx], feature_padding_mask)
            for idx in self.output_layer
        ]

    def normalize_features(
        self,
        feat: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the family-specific LayerNorm口径。

        PASE/StuPASE 官方使用 ``F.layer_norm(feat, feat.shape)``，normalized_shape
        含 batch 维，整批共享统计量；UniPASE/GAP 官方使用 ``feat.shape[1:]``。
        本方法只负责选择口径，不改变 upstream WavLM 输出。
        """
        if getattr(self, "feature_norm", "per_sample") == "batch":
            return F.layer_norm(feat, feat.shape, eps=1e-6)
        if padding_mask is None:
            return F.layer_norm(feat, feat.shape[1:], eps=1e-6)
        valid = ~padding_mask
        mask = valid.unsqueeze(-1).to(feat.dtype)
        count = mask.expand_as(feat).sum(dim=(1, 2), keepdim=True).clamp(min=1)
        mean = (feat * mask).sum(dim=(1, 2), keepdim=True) / count
        centered = (feat - mean) * mask
        var = (centered * centered).sum(dim=(1, 2), keepdim=True) / count
        return centered / torch.sqrt(var + 1e-6)

    def load_pretrained_weights(self, ckpt_path: str) -> None:
        """Load or reload pretrained WavLM weights (used for Phase-2 student init)."""
        ckpt_path = Path(ckpt_path).expanduser()
        cpt = safe_torch_load(ckpt_path, map_location="cpu")
        self.wavlm.load_state_dict(cpt["model"], strict=True)
        logger.info("[WavLM] Reloaded weights from %s", ckpt_path)

    def set_frozen(self, frozen: bool) -> None:
        """Toggle grad requirement for all WavLM parameters."""
        self._frozen = frozen
        for p in self.wavlm.parameters():
            p.requires_grad = not frozen
        if frozen:
            self.wavlm.eval()
        else:
            self.wavlm.train()
