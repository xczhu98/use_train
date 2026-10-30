"""VocosVocoder: UniPASE 的 WavLM-L1 特征声码器（16kHz 波形输出）。

上游来源：<https://github.com/Xiaobin-Rong/UniPASE> 的
``models/vocoder/vocos/vocoder.py``；本仓训练 recipe：
``configs/se/urgent2026/pase/unipase/unipase_phase2_vocoder_gan.yaml``。

复用本框架共享的 ``VocosBackbone``（common/vocos/backbone.py，已参数化 num_res/num_attn，
与官方 models/vocoder/vocos/backbone.py:150-200 拓扑一致）和 ``ISTFTHead``
（common/vocos/head.py，``mag_clip="log"`` 即官方 log-mag 裁剪行为）。
"""

import torch
from torch import nn

from ..common.vocos.backbone import VocosBackbone
from ..common.vocos.head import ISTFTHead


class VocosVocoder(nn.Module):
    """Vocos vocoder mapping WavLM L1 features to a 16kHz waveform.

    Args:
        input_channels: WavLM feature dim (1024 for Large).
        dim: Backbone hidden dimension.
        intermediate_dim: ConvNeXt intermediate dimension.
        num_layers: Number of ConvNeXt blocks.
        num_res: Backbone positional pre-net 的 ResNet 块数（必须为偶数）。
        num_attn: Backbone positional pre-net 的 self-attention 块数。
        n_fft: ISTFT FFT size（win_length = n_fft = 1280）。
        hop_length: ISTFT hop length（320，对齐 WavLM 下采样率）。

    默认值为官方发布配置（configs/cfg_train_vocoder.yaml:5-11: dim=1024,
    intermediate_dim=3072）；官方 vocoder.py 构造默认 dim=768/intermediate_dim=2304
    与其实际训练配置不一致，这里以训练配置为准。
    """

    def __init__(
        self,
        input_channels: int = 1024,
        dim: int = 1024,
        intermediate_dim: int = 3072,
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
        self.head = ISTFTHead(dim=dim, n_fft=n_fft, hop_length=hop_length, mag_clip="log")

    @classmethod
    def from_pretrained(cls, ckpt_path: str, frozen: bool = True) -> "VocosVocoder":
        """Load from a UniPASE vocoder checkpoint.

        checkpoint 需含 'cfg' 以及 'generator' 或 'model' 权重（官方 :39-60）。
        加载语义统一走 tasks.se.models.pase_family.common.utils.load_artifact_module
        （weights_only=True + key 匹配报告，任何 missing/unexpected 都 fail closed）。
        """
        from ..common.utils import load_artifact_module

        return load_artifact_module(cls, ckpt_path, frozen=frozen)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Args: input (B, D, T) WavLM L1 特征（官方 unipase.py:48 传入
        enh_feat_a.transpose(1, 2)）。Returns: audio (B, T_audio)。"""
        x = self.decoder(input)
        audio_output = self.head(x)

        return audio_output
