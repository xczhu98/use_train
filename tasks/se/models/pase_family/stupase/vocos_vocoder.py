"""VocosVocoder: Mel-conditioned Vocos vocoder for StuPASE.

Reuses the shared, generalized ``VocosBackbone``
(tasks/se/models/pase_family/common/vocos/backbone.py, extended with
num_res/num_attn to match this model's topology) and StuPASE's own
``ISTFTHead`` (vocos_head.py, numerically distinct from PASE's head).

Ported from ``stupase/models/vocoder/vocos/vocoder.py``.
"""

import logging

import torch
from torch import nn

from ..common.utils import load_artifact_module
from ..common.vocos.backbone import VocosBackbone
from .vocos_head import ISTFTHead

logger = logging.getLogger(__name__)


class VocosVocoder(nn.Module):
    """Mel-spectrogram-conditioned Vocos vocoder.

    Args:
        input_channels: Mel-spectrogram channel count (n_mels).
        dim: Backbone hidden dimension.
        intermediate_dim: ConvNeXt intermediate dimension.
        num_layers: Number of ConvNeXt blocks.
        num_res: Number of ResNet blocks in the positional pre-net (must be even).
        num_attn: Number of self-attention blocks in the positional pre-net.
        n_fft: ISTFT FFT size.
        hop_length: ISTFT hop length.
    """

    def __init__(
        self,
        input_channels: int = 100,
        dim: int = 768,
        intermediate_dim: int = 2304,
        num_layers: int = 12,
        num_res: int = 4,
        num_attn: int = 1,
        n_fft: int = 1280,
        hop_length: int = 320,
    ):
        super().__init__()
        self.decoder = VocosBackbone(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            num_res=num_res,
            num_attn=num_attn,
        )
        self.head = ISTFTHead(dim=dim, n_fft=n_fft, hop_length=hop_length)

    @classmethod
    def from_pretrained(cls, ckpt_path: str, frozen: bool = True) -> "VocosVocoder":
        """加载 ``{cfg, generator|model}`` 形式的 Mel vocoder 发布 artifact。

        加载语义统一走 ``common.utils.load_artifact_module``
        （weights_only=True + key 匹配报告，任何 missing/unexpected 都 fail closed）。
        """
        return load_artifact_module(cls, ckpt_path, frozen=frozen)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Args: mel (B, n_mels, T). Returns: audio (B, T_audio)."""
        x = self.decoder(mel)
        return self.head(x)
