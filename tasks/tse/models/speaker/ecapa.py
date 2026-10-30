"""Vendored ECAPA-TDNN speaker encoder（对齐 wespeaker ``ECAPA_TDNN_GLOB_c512``）.

来源：wespeaker ``wespeaker/models/ecapa_tdnn.py`` + ``pooling_layers.py::ASTP``
（https://github.com/wenet-e2e/wespeaker，Apache-2.0），按 use_train 风格 vendor
进来以去掉对 wespeaker 包的 git 依赖。**模块层级与属性名与 wespeaker 完全一致**
（``layer1..layer4 / conv / pool / bn / linear``），因此 wespeaker VoxCeleb 预训练
checkpoint（``avg_model.pt``）的 state_dict 键可原样映射加载。

比赛 baseline 配置：80 维 fbank 输入、C=512、ASTP(global_context_att=True)、
embed_dim=192（即 ``ECAPA_TDNN_GLOB_c512(feat_dim=80, embed_dim=192)``）。

Author: Xincheng Zhu
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Res2Conv1dReluBn(nn.Module):
    """Res2Conv1d + BatchNorm1d + ReLU（in_channels == out_channels == channels）。"""

    def __init__(
        self,
        channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        scale=4,
    ):
        super().__init__()
        if channels % scale != 0:
            raise ValueError(f"channels={channels} must be divisible by scale={scale}")
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        convs = []
        bns = []
        for _ in range(self.nums):
            convs.append(
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    bias=bias,
                )
            )
            bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            # 顺序：conv -> relu -> bn
            if i >= 1:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out


class Conv1dReluBn(nn.Module):
    """Conv1d + BatchNorm1d + ReLU。"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    """1D 情况的 SE 连接。"""

    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out


class SE_Res2Block(nn.Module):
    """ECAPA-TDNN 的 SE-Res2Block。"""

    def __init__(self, channels, kernel_size, stride, padding, dilation, scale):
        super().__init__()
        self.se_res2block = nn.Sequential(
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            Res2Conv1dReluBn(
                channels, kernel_size, stride, padding, dilation, scale=scale
            ),
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            SE_Connect(channels),
        )

    def forward(self, x):
        return x + self.se_res2block(x)


class ASTP(nn.Module):
    """Attentive statistics pooling（ECAPA-TDNN 使用，通道与上下文相关统计池化）。"""

    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.global_context_att = global_context_att

        # 用 stride=1 的 Conv1d 代替 Linear，省去 transpose
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, bottleneck_dim, kernel_size=1)
        else:
            self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        """
        x: tdnn 架构下的 3 维张量 (B, F, T)，最后一维为时间维
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        if len(x.shape) != 3:
            raise ValueError(
                "ASTP expects (B, F, T) or (B, C, F, T) input, got shape "
                f"{tuple(x.shape)}"
            )

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            # 单帧时无偏方差无定义，使用 correction=0。
            correction = 0 if x.shape[-1] == 1 else 1
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True, correction=correction) + 1e-7
            ).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-7))
        return torch.cat([mean, std], dim=1)

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim


_POOLING_LAYERS = {"ASTP": ASTP}


class ECAPA_TDNN(nn.Module):
    """ECAPA-TDNN（模块命名与 wespeaker 一致，保证 ckpt 键名兼容）。"""

    def __init__(
        self,
        channels=512,
        feat_dim=80,
        embed_dim=192,
        pooling_func="ASTP",
        global_context_att=False,
        emb_bn=False,
    ):
        super().__init__()
        if pooling_func not in _POOLING_LAYERS:
            raise ValueError(
                f"Unsupported pooling_func {pooling_func!r}, "
                f"available: {list(_POOLING_LAYERS)}"
            )

        self.layer1 = Conv1dReluBn(feat_dim, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(
            channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8
        )
        self.layer3 = SE_Res2Block(
            channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8
        )
        self.layer4 = SE_Res2Block(
            channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8
        )

        cat_channels = channels * 3
        out_channels = 512 * 3
        self.conv = nn.Conv1d(cat_channels, out_channels, kernel_size=1)
        self.pool = _POOLING_LAYERS[pooling_func](
            in_dim=out_channels, global_context_att=global_context_att
        )
        self.pool_out_dim = self.pool.get_out_dim()
        self.bn = nn.BatchNorm1d(self.pool_out_dim)
        self.linear = nn.Linear(self.pool_out_dim, embed_dim)
        self.emb_bn = emb_bn
        if emb_bn:  # SSL 场景下对 embedding 再加 BN
            self.bn2 = nn.BatchNorm1d(embed_dim)
        else:
            self.bn2 = nn.Identity()

    def _get_frame_level_feat(self, x):
        # 内部使用；x: (B, T, F) fbank
        x = x.permute(0, 2, 1)  # (B,T,F) -> (B,F,T)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = torch.cat([out2, out3, out4], dim=1)
        out = self.conv(out)

        return out, out4

    def get_frame_level_feat(self, x):
        # 外部接口；返回 (B, T, D)
        out = self._get_frame_level_feat(x)[0].permute(0, 2, 1)
        return out

    def forward(self, x):
        """x: (B, T, F) fbank → (out4 帧级特征 (B,C,T), 句级 embedding (B,embed_dim))。"""
        out, out4 = self._get_frame_level_feat(x)
        out = F.relu(out)
        out = self.bn(self.pool(out))
        out = self.linear(out)
        if self.emb_bn:
            out = self.bn2(out)
        return out4, out


def ECAPA_TDNN_GLOB_c512(feat_dim, embed_dim, pooling_func="ASTP", emb_bn=False):
    """比赛 baseline 同款：C=512 + global context ASTP，80 维 fbank → 192 维 embedding。"""
    return ECAPA_TDNN(
        channels=512,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        global_context_att=True,
        emb_bn=emb_bn,
    )
