"""VocosAdapter: UniPASE 的声学-语义适配器（WavLM L24 + L1 -> 增强 L1 特征）。

上游来源：<https://github.com/Xiaobin-Rong/UniPASE> 的
``models/adapter/vocos/adapter.py``；本仓训练 recipe：
``configs/se/urgent2026/pase/unipase/unipase_phase3_adapter_gan.yaml``。

结构（对齐官方）：
  Linear(input_channels->input_channels) 投影 L24 语义特征 + 加上 L1 声学特征
  -> 共享 VocosBackbone（官方用 models/adapter/vocos/backbone.py，与本框架
  common/vocos/backbone.py 参数化后拓扑一致）
  -> Linear(dim->output_channels) 头。

官方在 (B, T, D) 特征上操作，进入 backbone 前 transpose 成 (B, D, T)，
出来再 transpose 回 (B, T, D)（官方 :60-62）。
"""

import torch
from torch import nn

from ..common.vocos.backbone import VocosBackbone


class VocosAdapter(nn.Module):
    """Vocos-style adapter mapping noisy L1 + enhanced L24 to enhanced L1 features.

    Args:
        input_channels: WavLM feature dim (1024 for Large).
        dim: Backbone hidden dimension.
        intermediate_dim: ConvNeXt intermediate dimension.
        num_layers: Number of ConvNeXt blocks.
        num_res: Backbone positional pre-net 的 ResNet 块数（必须为偶数）。
        num_attn: Backbone positional pre-net 的 self-attention 块数。
        output_channels: Output feature dim (1024, i.e. enhanced L1).

    默认值为官方发布配置（configs/cfg_train_adapter.yaml:11-16: dim=1024,
    intermediate_dim=3072, num_layers=12）；官方 adapter.py 构造默认
    intermediate_dim=4096 与其实际训练配置不一致，这里以训练配置为准。
    """

    def __init__(
        self,
        input_channels: int = 1024,
        dim: int = 1024,
        intermediate_dim: int = 3072,
        num_layers: int = 12,
        num_res: int = 4,
        num_attn: int = 1,
        output_channels: int = 1024,
    ):
        super().__init__()
        self.proj = nn.Linear(input_channels, input_channels)
        self.decoder = VocosBackbone(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            num_res=num_res,
            num_attn=num_attn,
        )
        self.head = nn.Linear(dim, output_channels)

    @classmethod
    def from_pretrained(cls, ckpt_path: str, frozen: bool = True) -> "VocosAdapter":
        """Load from a UniPASE adapter checkpoint.

        checkpoint 需含 'cfg' 以及 'generator' 或 'model' 权重（官方 :26-47）。
        加载语义统一走 tasks.se.models.pase_family.common.utils.load_artifact_module
        （weights_only=True + key 匹配报告，任何 missing/unexpected 都 fail closed）。
        """
        from ..common.utils import load_artifact_module

        return load_artifact_module(cls, ckpt_path, frozen=frozen)

    def forward(self, embed_a: torch.Tensor, embed_p: torch.Tensor) -> torch.Tensor:
        """Args: embed_a (B, T, D) noisy L1 acoustic feat; embed_p (B, T, D)
        enhanced L24 phonetic feat. Returns: (B, T, D) enhanced L1 feat."""
        embed_p = self.proj(embed_p)

        x = embed_p + embed_a

        x = x.transpose(1, 2)
        x = self.decoder(x)
        x = x.transpose(1, 2)

        x = self.head(x)

        return x  # (B, T, D)
