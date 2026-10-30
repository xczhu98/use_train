# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
"""MuSE 视觉前端 kernel 的 canonical 移植。

来源：WeSep commit ``dc11cb18`` 的 ``wesep/modules/visual/muse.py``（Apache-2.0，
Copyright (c) 2026 Ke Zhang）。上游注明 lip 预处理、视觉 encoder 与时间卷积移植
自 MuSE（Z. Pan, R. Tao, C. Xu, and H. Li, "MuSE: Multi-Modal Target Speaker
Extraction with Visual Cues", ICASSP 2021, https://github.com/zexupan/MuSE）；
第三方授权与署名条款沿用仓库既有的 third-party notice 约定，本文件只做 kernel
来源说明。

本模块只保留 kernel：不注册模型、不构造 separator、不持有 STFT、不解码视频文件。
checkpoint 一律经 ``core.utils.safe_load.safe_torch_load`` 以 ``weights_only=True``
加载，并在复制权重前严格校验键集合与 shape，禁止静默回退或部分加载。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.safe_load import safe_torch_load


# MuSE 视觉 encoder 的帧级特征维度；raw 输入路径的默认 adapter 宽度。
MUSE_FRONTEND_FEATURE_DIM = 512

_ADAPTER_TYPES = frozenset({"direct", "muse", "clearer"})
_ADAPTER_KEYS = frozenset({"type", "input_dim", "output_dim", "hidden_dim", "num_layers"})


class Muse_LipROIProcessor(nn.Module):
    """把 RGB video ``[B,H,W,3,T]`` 转成归一化 lip ROI ``[B,T,roi,roi]``。"""

    def __init__(self,
                 roi_size=112,
                 resize_size=224,
                 norm_mean=0.4161,
                 norm_std=0.1688):
        super().__init__()
        self.roi_size = roi_size
        self.resize_size = resize_size
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def forward(self, video):
        B, H, W, C, T = video.shape
        if C != 3:
            raise ValueError(f"Expected three RGB channels, got {C}")
        if video.dtype not in (torch.uint8, torch.float32):
            raise TypeError(
                f"raw video must be uint8 or float32, got {video.dtype}")

        # Convert compact RGB frames into normalized grayscale images.
        video = video.permute(0, 4, 3, 1, 2).contiguous()
        scale = 1.0 / 255.0 if video.dtype == torch.uint8 else 1.0
        gray = video[:, :, 0].float().mul_(0.299 * scale)
        gray.add_(video[:, :, 1], alpha=0.587 * scale)
        gray.add_(video[:, :, 2], alpha=0.114 * scale)

        # Resize each frame and take the centered lip region.
        gray = gray.view(B * T, 1, H, W)
        if H != self.resize_size or W != self.resize_size:
            gray = F.interpolate(
                gray,
                size=(self.resize_size, self.resize_size),
                mode="bilinear",
                align_corners=False,
            )
        start = self.resize_size // 2 - self.roi_size // 2
        roi = gray[:, :, start:start + self.roi_size,
                   start:start + self.roi_size]
        roi = roi.view(B, T, self.roi_size, self.roi_size)
        return (roi - self.norm_mean) / self.norm_std


class ResNetLayer(nn.Module):
    """MuSE 视觉 ResNet 的两个残差单元。"""

    def __init__(self, inplanes, outplanes, stride):
        super().__init__()
        self.conv1a = nn.Conv2d(inplanes,
                                outplanes,
                                3,
                                stride=stride,
                                padding=1,
                                bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, 3, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes,
                                    outplanes,
                                    1,
                                    stride=stride,
                                    bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, 3, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, 3, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

    def forward(self, x):
        residual = x if self.stride == 1 else self.downsample(x)
        x = self.conv2a(F.relu(self.bn1a(self.conv1a(x))))
        x = x + residual
        intermediate = x
        x = F.relu(self.outbna(x))

        x = self.conv2b(F.relu(self.bn1b(self.conv1b(x))))
        return F.relu(self.outbnb(x + intermediate))


class ResNet(nn.Module):
    """视觉前端使用的 ResNet-18 风格帧 encoder。"""

    def __init__(self):
        super().__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4, 4), stride=(1, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x)


class Muse_VisualFrontend(nn.Module):
    """把每个 lip 帧编码为 ``[B,512,T]`` 表示。"""

    def __init__(self):
        super().__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                64,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
        )
        self.resnet = ResNet()

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        batch_size = x.shape[0]
        x = self.frontend3D(x).transpose(1, 2)
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.resnet(x)
        return x.reshape(batch_size, -1, MUSE_FRONTEND_FEATURE_DIM).transpose(1, 2)


def load_muse_frontend(frontend, checkpoint):
    """把 MuSE 帧 encoder 权重严格加载进 ``frontend``。

    只接受 ``{key: tensor}`` 或 ``{"state_dict": {...}}`` 形式，允许去 ``module.``
    DDP 前缀；缺失键、多余键、重复键、shape/dtype 不匹配以及 0 匹配一律 fail-closed，
    不做静默部分加载，也不接受 pickle 出来的模型对象。
    """
    if not isinstance(frontend, nn.Module):
        raise TypeError("frontend must be an nn.Module")
    if not isinstance(checkpoint, (str, os.PathLike)):
        raise TypeError("MuSE frontend checkpoint must be a filesystem path")
    path = Path(checkpoint).expanduser()
    state = safe_torch_load(
        path,
        map_location="cpu",
        purpose="load MuSE visual frontend encoder weights",
    )
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError(
            f"MuSE frontend checkpoint {path} must contain a non-empty state dict"
        )

    model_state = frontend.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    for raw_key, value in state.items():
        if not isinstance(raw_key, str):
            unexpected.append(repr(raw_key))
            continue
        key = raw_key[len("module."):] if raw_key.startswith("module.") else raw_key
        if key not in model_state:
            unexpected.append(key)
            continue
        if key in adapted:
            raise ValueError(f"MuSE frontend checkpoint {path} repeats key {key!r}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"MuSE frontend checkpoint {path} entry {key!r} must be a tensor"
            )
        expected = model_state[key]
        if tuple(value.shape) != tuple(expected.shape):
            raise ValueError(
                f"MuSE frontend checkpoint {path} key {key!r} has shape "
                f"{tuple(value.shape)}, expected {tuple(expected.shape)}"
            )
        if value.dtype != expected.dtype:
            raise TypeError(
                f"MuSE frontend checkpoint {path} key {key!r} has dtype "
                f"{value.dtype}, expected {expected.dtype}"
            )
        adapted[key] = value

    missing = sorted(set(model_state) - set(adapted))
    if missing or unexpected:
        raise ValueError(
            f"MuSE frontend checkpoint {path} does not match the encoder: "
            f"{len(adapted)} matched, missing={missing}, unexpected={unexpected}"
        )
    frontend.load_state_dict(adapted, strict=True)


def compute_muse_frontend(video, roi, frontend):
    """把 raw video ``[B,H,W,3,T]`` 转成 ``[B,512,T]`` 特征。"""
    lip_roi = roi(video)
    return frontend(lip_roi.transpose(0, 1).unsqueeze(2))


class Muse_VisualConv1D(nn.Module):
    """MuSE 的单层残差时间视觉卷积块。"""

    def __init__(self, channels=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.PReLU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x) + x


class DirectVisualAdapter(nn.Module):
    """原样透传帧级视觉特征。"""

    def __init__(self, config):
        super().__init__()
        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]

    def forward(self, x):
        return x


class MuseVisualAdapter(nn.Module):
    """应用 MuSE 的时间视觉卷积栈。"""

    def __init__(self, config):
        super().__init__()
        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        self.net = nn.Sequential(*[
            Muse_VisualConv1D(channels=self.output_dim)
            for _ in range(config["num_layers"])
        ])

    def forward(self, x):
        return self.net(x)


class ClearerVisualConv1D(nn.Module):
    """ClearerVoice 风格的单层残差时间卷积块。"""

    def __init__(self, channels=256, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, hidden_dim, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                3,
                padding=1,
                groups=hidden_dim,
                bias=True,
            ),
            nn.PReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, channels, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x) + x


class ClearerVisualAdapter(nn.Module):
    """先用 1x1 投影再走 ClearerVoice 风格 VTCN 块处理视觉 cue。"""

    def __init__(self, config):
        super().__init__()
        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        self.net = nn.Sequential(
            nn.Conv1d(
                self.input_dim,
                self.output_dim,
                1,
                bias=False,
            ),
            *[
                ClearerVisualConv1D(
                    channels=self.output_dim,
                    hidden_dim=config["hidden_dim"],
                ) for _ in range(config["num_layers"])
            ],
        )

    def forward(self, x):
        return self.net(x)


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact int, got {value!r}")
    return value


def build_visual_adapter(config: Mapping[str, object]) -> nn.Module:
    """构造配置指定的时间视觉 adapter，并先校验全部被消费的键。

    ``direct`` / ``muse`` 不做通道投影，因此要求 ``input_dim == output_dim``；
    ``clearer`` 需要显式 ``hidden_dim``；未被对应 adapter 消费的键一律拒绝。
    """
    if not isinstance(config, Mapping):
        raise TypeError("visual adapter config must be a mapping")
    unknown = sorted(set(config) - _ADAPTER_KEYS)
    if unknown:
        raise ValueError(f"visual adapter config contains unknown keys: {unknown}")
    adapter_type = config.get("type")
    if adapter_type not in _ADAPTER_TYPES:
        raise ValueError(f"unsupported visual adapter: {adapter_type!r}")
    for key in ("input_dim", "output_dim"):
        if key not in config:
            raise ValueError(f"visual adapter requires {key}")
    input_dim = _positive_int(config["input_dim"], "visual adapter input_dim")
    output_dim = _positive_int(config["output_dim"], "visual adapter output_dim")

    if adapter_type == "direct":
        unused = sorted(set(config) & {"hidden_dim", "num_layers"})
        if unused:
            raise ValueError(f"direct visual adapter does not consume {unused}")
        if input_dim != output_dim:
            raise ValueError(
                "direct visual adapter passes features through unchanged and requires "
                f"input_dim == output_dim, got {input_dim} and {output_dim}"
            )
        return DirectVisualAdapter(config)

    if adapter_type == "muse":
        if "hidden_dim" in config:
            raise ValueError("muse visual adapter does not consume hidden_dim")
        if input_dim != output_dim:
            raise ValueError(
                "muse visual adapter keeps the channel count and requires "
                f"input_dim == output_dim, got {input_dim} and {output_dim}"
            )
        if "num_layers" not in config:
            raise ValueError("muse visual adapter requires num_layers")
        _positive_int(config["num_layers"], "visual adapter num_layers")
        return MuseVisualAdapter(config)

    for key in ("hidden_dim", "num_layers"):
        if key not in config:
            raise ValueError(f"clearer visual adapter requires {key}")
    _positive_int(config["hidden_dim"], "visual adapter hidden_dim")
    _positive_int(config["num_layers"], "visual adapter num_layers")
    return ClearerVisualAdapter(config)


__all__ = [
    "ClearerVisualAdapter",
    "ClearerVisualConv1D",
    "DirectVisualAdapter",
    "MUSE_FRONTEND_FEATURE_DIM",
    "MuseVisualAdapter",
    "Muse_LipROIProcessor",
    "Muse_VisualConv1D",
    "Muse_VisualFrontend",
    "build_visual_adapter",
    "compute_muse_frontend",
    "load_muse_frontend",
]
