"""MTFAA-Net 的频谱主干模块。

本文件按上游非官方实现重写，仅保留网络拓扑；STFT/iSTFT 由仓库顶层完成。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ComplexConv2d(nn.Module):
    """由两组实卷积构成的复数卷积。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (1, 1),
        *,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if in_channels % 2 or out_channels % 2:
            raise ValueError("complex channels must be even")
        self.time_pad = kernel_size[1] - 1
        self.causal = causal
        self.real_conv = nn.Conv2d(in_channels // 2, out_channels // 2, kernel_size)
        self.imag_conv = nn.Conv2d(in_channels // 2, out_channels // 2, kernel_size)
        nn.init.normal_(self.real_conv.weight, std=0.05)
        nn.init.normal_(self.imag_conv.weight, std=0.05)
        nn.init.zeros_(self.real_conv.bias)
        nn.init.zeros_(self.imag_conv.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.time_pad:
            if self.causal:
                inputs = F.pad(inputs, (self.time_pad, 0, 0, 0))
            else:
                left = self.time_pad // 2
                inputs = F.pad(inputs, (left, self.time_pad - left, 0, 0))
        real, imag = inputs.chunk(2, dim=1)
        output_real = self.real_conv(real) - self.imag_conv(imag)
        output_imag = self.imag_conv(real) + self.real_conv(imag)
        return torch.cat([output_real, output_imag], dim=1)


class PhaseEncoder(nn.Module):
    """将复频谱映射为压缩幅度特征。"""

    def __init__(self, channels: int = 4, compression: float = 0.5) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("phase encoder channels must be even")
        if compression <= 0:
            raise ValueError("compression must be positive")
        self.temporal_conv = ComplexConv2d(2, channels, (1, 3), causal=True)
        self.projection = ComplexConv2d(channels, channels)
        self.compression = compression

    def forward(self, spectrum: Tensor) -> Tensor:
        encoded = self.temporal_conv(spectrum)
        projected = self.projection(encoded)
        real, imag = projected.chunk(2, dim=1)
        magnitude = torch.sqrt(real.square() + imag.square() + 1e-8)
        return magnitude.pow(self.compression)


def _linear_filter_bank(n_filters: int, n_frequencies: int) -> Tensor:
    """构造不依赖第三方库的线性三角 filter bank。"""
    points = torch.linspace(0, n_frequencies - 1, n_filters + 2)
    bins = torch.arange(n_frequencies, dtype=torch.float32)
    filters = []
    for left, center, right in zip(points[:-2], points[1:-1], points[2:]):
        rising = (bins - left) / (center - left).clamp_min(1e-8)
        falling = (right - bins) / (right - center).clamp_min(1e-8)
        filters.append(torch.minimum(rising, falling).clamp_min(0.0))
    return torch.stack(filters)


class LinearFilterBank(nn.Module):
    """线性频带压缩及伪逆展开。"""

    def __init__(self, n_filters: int, n_frequencies: int) -> None:
        super().__init__()
        filters = _linear_filter_bank(n_filters, n_frequencies)
        self.register_buffer("filters", filters * 1.3)
        self.register_buffer("inverse_filters", torch.linalg.pinv(filters))

    def compress(self, inputs: Tensor) -> Tensor:
        return torch.einsum("bcft,kf->bckt", inputs, self.filters)

    def expand(self, inputs: Tensor) -> Tensor:
        return torch.einsum("bckt,fk->bcft", inputs, self.inverse_filters)


class FrequencyDownsample(nn.Module):
    """按四倍频率步长下采样。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (7, 1), (4, 1), (2, 0), groups=2),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class FrequencyUpsample(nn.Module):
    """融合 encoder skip 后按四倍频率步长上采样。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.Tanh(),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                out_channels,
                out_channels,
                (7, 1),
                (4, 1),
                (2, 0),
                (1, 0),
                groups=2,
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        output = self.gate(torch.cat([inputs, skip], dim=1)) * skip
        return self.upsample(self.projection(output))


class TFCMBlock(nn.Module):
    """带频率深度卷积和时间膨胀卷积的残差块。"""

    def __init__(self, channels: int, dilation: int, causal: bool) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
        )
        time_context = 2 * dilation
        time_left = time_context if causal else time_context // 2
        time_right = 0 if causal else time_context - time_left
        self.depthwise = nn.Sequential(
            nn.ConstantPad2d((time_left, time_right, 1, 1), 0.0),
            nn.Conv2d(
                channels,
                channels,
                (3, 3),
                dilation=(1, dilation),
                groups=channels,
            ),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
        )
        self.post = nn.Conv2d(channels, channels, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.post(self.depthwise(self.pre(inputs)))


class TFCM(nn.Module):
    """指数膨胀率堆叠的 TFCM。"""

    def __init__(self, channels: int, layers: int, causal: bool) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *(TFCMBlock(channels, 2**index, causal) for index in range(layers))
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.blocks(inputs)


class AxialSoftAttention(nn.Module):
    """先频率后时间的 axial soft attention。"""

    def __init__(self, channels: int, causal: bool) -> None:
        super().__init__()
        attention_channels = channels // 4
        if attention_channels < 1:
            raise ValueError("attention channels require channels >= 4")
        self.frequency_qkv = nn.Sequential(
            nn.Conv2d(channels, attention_channels * 3, 1, bias=False),
            nn.BatchNorm2d(attention_channels * 3),
            nn.PReLU(attention_channels * 3),
        )
        self.temporal_qk = nn.Sequential(
            nn.Conv2d(channels, attention_channels * 2, 1, bias=False),
            nn.BatchNorm2d(attention_channels * 2),
            nn.PReLU(attention_channels * 2),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(attention_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
        )
        self.scale = attention_channels**-0.5
        self.causal = causal

    def forward(self, inputs: Tensor) -> Tensor:
        frequency_query, frequency_key, value = self.frequency_qkv(inputs).chunk(3, dim=1)
        frequency_score = torch.einsum(
            "bcft,bcyt->btfy", frequency_query, frequency_key
        ) * self.scale
        frequency_output = torch.einsum(
            "btfy,bcyt->bcft", frequency_score.softmax(dim=-1), value
        )

        temporal_query, temporal_key = self.temporal_qk(inputs).chunk(2, dim=1)
        temporal_score = torch.einsum(
            "bcft,bcfy->bfty", temporal_query, temporal_key
        ) * self.scale
        if self.causal:
            time = temporal_score.shape[-1]
            mask = torch.ones(
                time, time, dtype=torch.bool, device=temporal_score.device
            ).triu(1)
            temporal_score = temporal_score.masked_fill(mask, -torch.finfo(temporal_score.dtype).max)
        temporal_output = torch.einsum(
            "bfty,bcfy->bcft", temporal_score.softmax(dim=-1), frequency_output
        )
        return inputs + self.projection(temporal_output)


class TemporalFrequencyStage(nn.Sequential):
    """TFCM 与 axial attention 的固定组合。"""

    def __init__(self, channels: int, tfcm_layers: int, causal: bool) -> None:
        super().__init__(
            TFCM(channels, tfcm_layers, causal),
            AxialSoftAttention(channels, causal),
        )


class MTFAABackbone(nn.Module):
    """MTFAA encoder-bottleneck-decoder 主干，输出线性频点特征。"""

    def __init__(
        self,
        n_frequencies: int,
        *,
        phase_channels: int = 4,
        channels: Sequence[int] = (48, 96, 192),
        filter_bands: int = 64,
        bottleneck_layers: int = 2,
        tfcm_layers: int = 6,
        causal: bool = True,
    ) -> None:
        super().__init__()
        channels = tuple(channels)
        if not channels or any(channel % 4 for channel in channels):
            raise ValueError("channels must be a non-empty sequence divisible by 4")
        if phase_channels != 4:
            raise ValueError("phase_channels must be 4 to match the shared MTFAA mask head")
        downsample_factor = 4 ** len(channels)
        if filter_bands < downsample_factor or filter_bands % downsample_factor:
            raise ValueError(
                f"filter_bands must be divisible by {downsample_factor} for {len(channels)} stages"
            )
        if bottleneck_layers < 1 or tfcm_layers < 1:
            raise ValueError("bottleneck_layers and tfcm_layers must be positive")

        self.phase_encoder = PhaseEncoder(phase_channels)
        self.filter_bank = LinearFilterBank(filter_bands, n_frequencies)
        encoder_channels = (phase_channels // 2, *channels)
        self.encoder_downsamples = nn.ModuleList(
            FrequencyDownsample(encoder_channels[index], encoder_channels[index + 1])
            for index in range(len(channels))
        )
        self.encoder_stages = nn.ModuleList(
            TemporalFrequencyStage(channel, tfcm_layers, causal) for channel in channels
        )
        self.bottleneck = nn.Sequential(
            *(
                TemporalFrequencyStage(channels[-1], tfcm_layers, causal)
                for _ in range(bottleneck_layers)
            )
        )

        decoder_channels = (phase_channels, *channels)
        self.decoder_upsamples = nn.ModuleList(
            FrequencyUpsample(decoder_channels[index], decoder_channels[index - 1])
            for index in range(len(channels), 0, -1)
        )
        self.decoder_stages = nn.ModuleList(
            TemporalFrequencyStage(decoder_channels[index - 1], tfcm_layers, causal)
            for index in range(len(channels), 0, -1)
        )

    def forward(self, spectrum: Tensor) -> Tensor:
        """输入 ``(B,2,F,T)``，输出 ``(B,4,T,F)``。"""
        output = self.filter_bank.compress(self.phase_encoder(spectrum))
        encoder_outputs = []
        for downsample, stage in zip(self.encoder_downsamples, self.encoder_stages):
            output = downsample(output)
            encoder_outputs.append(output)
            output = stage(output)
        output = self.bottleneck(output)
        for upsample, stage, skip in zip(
            self.decoder_upsamples, self.decoder_stages, reversed(encoder_outputs)
        ):
            output = stage(upsample(output, skip))
        return self.filter_bank.expand(output).permute(0, 1, 3, 2).contiguous()
