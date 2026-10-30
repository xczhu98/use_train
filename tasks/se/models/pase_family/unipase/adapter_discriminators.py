"""UniPASE adapter 判别器：作用在 WavLM 特征上的多尺度 1D 判别器。

Ported from ``unipase/models/adapter/discriminators.py``（Discriminator1D +
MultiScaleDiscriminator，6 个尺度，embed_dims 32->1024）。
官方源自 hifi-gan（MIT license）。

注册名 "UniPASEMultiScaleDiscriminator"（与 PASEDiscriminator 的注册模式一致）。
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import spectral_norm, weight_norm

from core.utils.registry import MODEL_REGISTRY


class Discriminator1D(nn.Module):
    """Single-scale 1D discriminator operating on (B, T, D) features."""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(in_dim, embed_dim, kernel_size=1)),
                norm_f(Conv1d(embed_dim, embed_dim, kernel_size=7, stride=2, padding=3)),
                norm_f(Conv1d(embed_dim, embed_dim, kernel_size=7, stride=2, padding=3)),
                norm_f(Conv1d(embed_dim, embed_dim, kernel_size=7, stride=2, padding=3)),
                norm_f(Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)),
            ]
        )
        self.conv_post = norm_f(Conv1d(embed_dim, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """x: (B, T, D)"""
        x = x.transpose(1, 2)
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


@MODEL_REGISTRY.register("UniPASEMultiScaleDiscriminator")
class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator over WavLM features (官方默认 6 个尺度).

    Args:
        in_dim: 特征维度（WavLM-Large 为 1024）。
        sub_dims: 各尺度的 embed_dim（官方默认 [32, 64, 128, 256, 512, 1024]）。
        use_spectral_norm: 是否用 spectral_norm 替代 weight_norm。
    """

    def __init__(
        self,
        in_dim: int = 1024,
        sub_dims: List[int] = None,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        if sub_dims is None:
            sub_dims = [32, 64, 128, 256, 512, 1024]
        self.discriminators = nn.ModuleList(
            [
                Discriminator1D(in_dim, embed_dim, use_spectral_norm=use_spectral_norm)
                for embed_dim in sub_dims
            ]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        """Args: y / y_hat (B, T, D) 真实 / 生成特征。"""
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
