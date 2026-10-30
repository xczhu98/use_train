"""Vocoder backend for PASE: single-stream and dual-stream decoders.

Maps WavLM features to waveform via Vocos backbone + ISTFT head.
"""

import logging
from pathlib import Path

import torch
from torch import nn

from core.utils.safe_load import safe_torch_load

from .vocos.backbone import VocosBackbone
from .vocos.head import ISTFTHead

logger = logging.getLogger(__name__)


class WavLMDec(nn.Module):
    """Vocoder decoder for PASE (single-stream or dual-stream).

    Args:
        input_channels: WavLM feature dimension (1024 for Large).
        dim: Backbone hidden dimension (default 768).
        intermediate_dim: ConvNeXt intermediate dimension (default 2304).
        num_layers: Number of ConvNeXt blocks (default 12).
        n_fft: ISTFT FFT size (default 1280).
        hop_length: ISTFT hop length (default 320, matching WavLM downsample).
        use_dual_stream: If True, include projection layer for acoustic features.
        frozen: If True, freeze all parameters after construction.
    """

    def __init__(
        self,
        input_channels: int = 1024,
        dim: int = 768,
        intermediate_dim: int = 2304,
        num_layers: int = 12,
        n_fft: int = 1280,
        hop_length: int = 320,
        use_dual_stream: bool = True,
        frozen: bool = False,
    ):
        super().__init__()
        self.use_dual_stream = use_dual_stream
        self.decoder = VocosBackbone(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
        )
        self.head = ISTFTHead(dim=dim, n_fft=n_fft, hop_length=hop_length)
        if use_dual_stream:
            self.proj = nn.Linear(input_channels, input_channels)

        if frozen:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False
            logger.info("[Vocoder] Frozen (%d params)", sum(p.numel() for p in self.parameters()))

    @classmethod
    def from_pretrained(cls, ckpt_path: str, frozen: bool = True, use_dual_stream: bool = None) -> "WavLMDec":
        """Load a WavLMDec from a PASE-style checkpoint, inferring architecture from the
        saved config.

        .. note::
            ``frozen`` defaults to ``True`` here (unlike ``__init__`` where it
            defaults to ``False``) because pretrained vocoders are typically
            loaded for inference or as a frozen initialisation for GAN training.

        Args:
            ckpt_path: Path to the vocoder checkpoint (``.tar``).
            frozen: If True, freeze all parameters after loading.
            use_dual_stream: If None, auto-detect from the checkpoint (``proj`` key in
                state_dict). Pass True/False to override.

        Returns:
            WavLMDec with loaded weights.
        """
        ckpt_path = Path(ckpt_path).expanduser()
        logger.info("[Vocoder] Loading from: %s", ckpt_path)
        ckpt = safe_torch_load(ckpt_path, map_location="cpu")

        # Auto-detect dual-stream from state_dict
        state_dict = ckpt.get("model", ckpt)
        if use_dual_stream is None:
            use_dual_stream = any(k.startswith("proj.") for k in state_dict)

        # Build model from the saved config (if available), falling back to defaults
        cfg = ckpt.get("cfg", {})
        # Defensive: prevent cfg keys from shadowing explicit constructor args
        cfg.pop("frozen", None)
        cfg.pop("use_dual_stream", None)
        model = cls(use_dual_stream=use_dual_stream, frozen=frozen, **cfg)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        n_sd = len(state_dict)
        matched = n_sd - len(unexpected)
        if matched == 0 or missing or unexpected:
            raise RuntimeError(
                f"[Vocoder.from_pretrained] {ckpt_path}: state_dict 不匹配 "
                f"(matched={matched}/{n_sd}, missing={len(missing)}, "
                f"unexpected={len(unexpected)})；"
                f"missing[:5]={missing[:5]}, unexpected[:5]={unexpected[:5]}"
            )

        logger.info(
            "[Vocoder] Loaded %s-stream vocoder (%d params)",
            "dual" if use_dual_stream else "single",
            sum(p.numel() for p in model.parameters()),
        )
        return model

    def forward(self, embed_p: torch.Tensor, embed_a: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            embed_p: (B, T, D) phonetic features.
            embed_a: (B, T, D) acoustic features. If None, single-stream mode;
                a non-None value with ``use_dual_stream=False`` is a hard error
                (never silently dropped).

        Returns:
            audio: (B, T_audio) time-domain waveform.
        """
        if embed_a is not None:
            if not self.use_dual_stream:
                raise ValueError(
                    "WavLMDec was built with use_dual_stream=False but received acoustic "
                    "features (embed_a); refusing to silently drop the acoustic branch."
                )
            embed_a = self.proj(embed_a)
            embed = embed_p + embed_a
        else:
            embed = embed_p
        x = embed.transpose(1, 2)  # (B, D, T)
        x = self.decoder(x)
        audio = self.head(x)
        return audio
