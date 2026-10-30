"""Encoder and decoders for ZipEnhancer.

- DenseEncoder: 2-D conv + dilated dense block + stride-2 downsampling
- MappingDecoder: magnitude spectrum decoder with sub-pixel upsampling
- PhaseDecoder: phase spectrum decoder with sub-pixel upsampling

All architectures follow MP-SENet (Interspeech 2023) as adapted in
ZipEnhancer (arXiv:2501.05183).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sub-pixel ConvTranspose2d (upsample only in frequency axis)
# ---------------------------------------------------------------------------

class SubPixelConvTranspose2d(nn.Module):
    """Sub-pixel upsampling in the width (frequency) dimension only.

    Equivalent to PixelShuffle along one axis, used to restore the
    frequency resolution halved by the encoder's stride-2 conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (1, 3),
        stride: tuple = (1, 2),
        padding: tuple = (0, 1),
    ):
        super().__init__()
        self.upscale_width_factor = stride[1]
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels * self.upscale_width_factor,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, f = x.size()
        x = self.conv1(x)
        x = x.view(b, c, self.upscale_width_factor, t, f)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(b, c, t, f * self.upscale_width_factor)
        return x


# ---------------------------------------------------------------------------
# DenseBlockV2: dilated dense block used in encoder & decoders
# ---------------------------------------------------------------------------

class DenseBlockV2(nn.Module):
    """4-layer dense block with exponentially increasing dilation.

    Each layer uses dilation ``(2**i, 1)`` along the time axis to expand
    the temporal receptive field without increasing parameters.
    """

    def __init__(self, dense_channel: int, kernel_size: tuple = (2, 3), depth: int = 4):
        super().__init__()
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2**i
            pad_length = kernel_size[0] + (dil - 1) * (kernel_size[0] - 1) - 1
            dense_conv = nn.Sequential(
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
                nn.Conv2d(
                    dense_channel * (i + 1),
                    dense_channel,
                    kernel_size,
                    dilation=(dil, 1),
                ),
                nn.InstanceNorm2d(dense_channel, affine=True),
                nn.PReLU(dense_channel),
            )
            self.dense_block.append(dense_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


# ---------------------------------------------------------------------------
# DenseEncoder
# ---------------------------------------------------------------------------

class DenseEncoder(nn.Module):
    """ZipEnhancer encoder: maps 2-channel (mag, pha) input to hidden features.

    Architecture:
      Conv(2->C, 1x1) + IN + PReLU
      -> DenseBlockV2(depth=4)
      -> Conv(C->C, 1x3, stride=(1,2)) + IN + PReLU
    Output frequency is halved: ``F -> F // 2``.
    """

    def __init__(self, dense_channel: int, in_channel: int = 2):
        super().__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, (1, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
        )
        self.dense_block = DenseBlockV2(dense_channel, depth=4)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, in_channel, T, F). Returns: (B, dense_channel, T, F//2)."""
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_2(x)
        return x


# ---------------------------------------------------------------------------
# BaseDecoder
# ---------------------------------------------------------------------------

class BaseDecoder(nn.Module):
    """Base decoder with dense block and sub-pixel upsampling."""

    def __init__(self, dense_channel: int):
        super().__init__()
        self.upsample_module_class = SubPixelConvTranspose2d
        self.dense_block = DenseBlockV2(dense_channel, depth=4)


# ---------------------------------------------------------------------------
# MappingDecoder -> compressed magnitude spectrum
# ---------------------------------------------------------------------------

class MappingDecoder(BaseDecoder):
    """Magnitude decoder: restores the full-frequency magnitude spectrum.

    Output is passed through ReLU to ensure non-negativity.
    """

    def __init__(self, dense_channel: int, out_channel: int = 1):
        super().__init__(dense_channel)
        self.mask_conv = nn.Sequential(
            self.upsample_module_class(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
            nn.Conv2d(dense_channel, out_channel, (1, 2)),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, C, T, F'). Returns: (B, out_channel, T, 2*F')."""
        x = self.dense_block(x)
        x = self.mask_conv(x)
        return self.relu(x)


# ---------------------------------------------------------------------------
# PhaseDecoder -> wrapped phase spectrum via arctan2
# ---------------------------------------------------------------------------

class PhaseDecoder(BaseDecoder):
    """Phase decoder: restores the full-frequency wrapped phase spectrum.

    Uses two output convolutions (real / imag) followed by atan2.
    """

    def __init__(self, dense_channel: int, out_channel: int = 1):
        super().__init__(dense_channel)
        self.phase_conv = nn.Sequential(
            self.upsample_module_class(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
        )
        self.phase_conv_r = nn.Conv2d(dense_channel, out_channel, (1, 2))
        self.phase_conv_i = nn.Conv2d(dense_channel, out_channel, (1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, C, T, F'). Returns: (B, out_channel, T, 2*F')."""
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        return torch.atan2(x_i, x_r)
