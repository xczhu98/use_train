"""Vendored WeSpeaker ResNet34 speaker encoder（Apache-2.0）。

模块名与参数键保持 WeSpeaker ``wespeaker/models/resnet.py`` 兼容，避免把
WeSpeaker 的 CLI、VAD、聚类等可选依赖带入 TSE 基础 import 路径。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TSTP(nn.Module):
    """Temporal statistics pooling。"""

    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1).flatten(start_dim=1)
        correction = 0 if x.shape[-1] == 1 else 1
        std = torch.sqrt(
            x.var(dim=-1, correction=correction) + 1e-7
        ).flatten(start_dim=1)
        return torch.cat((mean, std), dim=1)

    def get_out_dim(self) -> int:
        return self.in_dim * 2


class BasicBlock(nn.Module):
    """WeSpeaker ResNet18/34 使用的 basic residual block。"""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    """WeSpeaker 小通道 speaker ResNet。"""

    def __init__(
        self,
        block: type[BasicBlock],
        num_blocks: tuple[int, int, int, int],
        *,
        m_channels: int = 32,
        feat_dim: int = 80,
        embed_dim: int = 256,
        pooling_func: str = "TSTP",
        two_emb_layer: bool = False,
    ):
        super().__init__()
        if pooling_func != "TSTP":
            raise ValueError("ResNet34 currently supports pooling_func='TSTP' only")
        self.in_planes = m_channels
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.frame_dim = self.stats_dim * block.expansion
        self.two_emb_layer = two_emb_layer
        self.conv1 = nn.Conv2d(1, m_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, m_channels * 8, num_blocks[3], stride=2)
        self.pool = TSTP(self.stats_dim * block.expansion)
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)
        if two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(
        self, block: type[BasicBlock], planes: int, count: int, stride: int
    ) -> nn.Sequential:
        layers = []
        for current_stride in (stride, *([1] * (count - 1))):
            layers.append(block(self.in_planes, planes, current_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _get_frame_level_feat(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def get_frame_level_feat(self, x: torch.Tensor) -> torch.Tensor:
        out = self._get_frame_level_feat(x).transpose(1, 3)
        return torch.flatten(out, 2, -1)

    def forward(self, x: torch.Tensor):
        stats = self.pool(self._get_frame_level_feat(x))
        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            embed_b = self.seg_2(self.seg_bn_1(F.relu(embed_a)))
            return embed_a, embed_b
        return torch.zeros((), device=x.device, dtype=x.dtype), embed_a


def ResNet34(
    feat_dim: int,
    embed_dim: int,
    pooling_func: str = "TSTP",
    two_emb_layer: bool = False,
) -> ResNet:
    """WeSpeaker ``ResNet34-TSTP-emb256-fbank80``。"""
    return ResNet(
        BasicBlock,
        (3, 4, 6, 3),
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer,
    )


__all__ = ["ResNet34"]
