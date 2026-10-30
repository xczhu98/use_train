"""Fast-ULCNet: spectrum-in / spectrum-out (框架 STFT 统一做 wav<->spec).

本文件不包含任何 STFT/iSTFT 或窗函数计算；仅接收 (B,F,T,2) 频谱、输出 (B,F,T,2)。

方法来源: Fast-ULCNet（Comfi-FastGRNN、压缩域 CRM、Channel Reorientation 等）与公开实现对齐；
继承 ULCNet 类频带重排与掩膜思路并以 FastGRNN 族降低复杂度。固定上游 commit、
复制/适配范围和 MIT 许可见 ``fast_ulcnet_impl/PROVENANCE`` 及同目录 ``LICENSE.*``。

结构与官方对齐要点:
- 输入为框架 STFT 后的 (B,F,T,2)；功率律压缩 0.3；reorientation window_size=48, overlap=0.33。
- Conv 4 层 32->64->96->128 + MaxPool(1,2)×3；Freq-ComfiFastGRNN 64 单元 bidir；子带 2×2 层 128 单元；FC 257；Stage2 CNN 2×32+pointwise 2；CRM 输入为压缩域 real/imag。
- YAML 需 stft 块；train.py 在配置含 stft 时统一走框架 STFT → 模型 → iSTFT。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks.se.models.base_model import BaseSEModel
from tasks.se.models.modules.fast_ulcnet_impl import ComfiFastGRNN, ComplexRatioMask
from core.utils.registry import MODEL_REGISTRY


def _zero_safe_signed_power(x: torch.Tensor, exponent: float) -> torch.Tensor:
    """非零值保持 signed power 精确值，零值的输出和梯度均为零。"""
    result = torch.zeros_like(x)
    nonzero = x != 0
    values = x[nonzero]
    result[nonzero] = torch.sign(values) * torch.pow(torch.abs(values), exponent)
    return result


class _ChannelWiseReorientation(nn.Module):
    """Channel-wise feature reorientation: (B, T, F) -> (B, T, n_bands, window_size). Paper: 1.5kHz, overlap 0.33."""

    def __init__(self, input_freq_dim: int, window_size: int = 48, overlap: float = 0.33):
        super().__init__()
        self.input_freq_dim = int(input_freq_dim)
        self.window_size = window_size
        self.hop_size = max(1, math.ceil(window_size * (1 - overlap)))
        self.n_bands = math.ceil(((self.input_freq_dim - self.window_size) / self.hop_size) + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        subbands = []
        for i in range(self.n_bands):
            start = i * self.hop_size
            end = start + self.window_size
            if end > self.input_freq_dim:
                subband = x[:, :, start:self.input_freq_dim]
                pad = torch.zeros((B, T, end - self.input_freq_dim), device=x.device, dtype=x.dtype)
                subband = torch.cat([subband, pad], dim=-1)
            else:
                subband = x[:, :, start:end]
            subbands.append(subband)
        return torch.stack(subbands, dim=2)


class _SeparableConv2d(nn.Module):
    """Depthwise separable 2D conv: (1,3) depthwise + (1,1) pointwise."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.pointwise(self.depthwise(x)))


class _ConvBlock(nn.Module):
    """4 depthwise-separable layers 32->64->96->128 with MaxPool(1,2) after 2nd/3rd/4th."""

    def __init__(self, in_channels: int = 8):
        super().__init__()
        self.sep1 = _SeparableConv2d(in_channels, 32)
        self.sep2 = _SeparableConv2d(32, 64)
        self.sep3 = _SeparableConv2d(64, 96)
        self.sep4 = _SeparableConv2d(96, 128)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.sep1(x))
        x = F.relu(self.sep2(x))
        x = self.pool(x)
        x = F.relu(self.sep3(x))
        x = self.pool(x)
        x = F.relu(self.sep4(x))
        x = self.pool(x)
        return x


@MODEL_REGISTRY.register("FastULCNet")
class FastULCNet(BaseSEModel):
    """Fast-ULCNet: spectrum-in / spectrum-out。STFT/iSTFT 由框架 stft_module 完成。

    输入: noisy_spec (B, F, T, 2)，F = n_fft//2+1，与 YAML stft 一致。
    输出: enhanced_spec (B, F, T, 2)。

    Args（YAML model.args；STFT 参数由顶层 ``stft`` 注入但模型内部不做 STFT）:
      n_fft: 固定为 512，对应官方网络的 257 个频点
      hop_length/win_length/window/center: 显式消费并记录框架 STFT 契约
      compression_factor: 0.3
      bidirectional_frnn_units: 64
      sub_band_rnn_units: 128
      CRM_type: 'E' | 'R' | 'C'
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        window: str = "hann",
        center: bool = True,
        compression_factor: float = 0.3,
        bidirectional_frnn_units: int = 64,
        sub_band_rnn_units: int = 128,
        CRM_type: str = "E",
    ):
        super().__init__()
        if n_fft != 512:
            raise ValueError(
                "FastULCNet has a fixed 257-bin architecture and requires n_fft=512"
            )
        if hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if win_length != n_fft:
            raise ValueError("FastULCNet requires win_length == n_fft == 512")
        if not isinstance(window, str) or not window:
            raise ValueError("window must be a non-empty string")
        if not isinstance(center, bool):
            raise TypeError("center must be a bool")
        if compression_factor <= 0:
            raise ValueError("compression_factor must be positive")

        # 这些字段只绑定框架 STFT 契约；模型内部不执行 STFT/iSTFT。
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.compression_factor = compression_factor
        freq_dim = n_fft // 2 + 1

        # reorientation: 论文 1.5kHz / overlap 0.33 → window_size=48, hop_size=ceil(48*0.67)=33, n_bands=8
        self.reorientation = _ChannelWiseReorientation(input_freq_dim=freq_dim, window_size=48, overlap=0.33)
        self.conv_block = _ConvBlock(in_channels=8)
        self.crm_layer = ComplexRatioMask(masking_mode=CRM_type)

        self.freq_rnn = ComfiFastGRNN(
            input_size=128,
            hidden_size=bidirectional_frnn_units,
            bidirectional=True,
            batch_first=True,
            lambda_init=0.0,
            gamma_init=0.999,
        )
        self.pointwise_conv = nn.Conv2d(2 * bidirectional_frnn_units, 64, kernel_size=(1, 1), bias=False)

        self.sub_band_rnn1 = ComfiFastGRNN(
            input_size=192,
            hidden_size=sub_band_rnn_units,
            num_layers=2,
            batch_first=True,
            lambda_init=0.0,
            gamma_init=0.999,
        )
        self.sub_band_rnn2 = ComfiFastGRNN(
            input_size=192,
            hidden_size=sub_band_rnn_units,
            num_layers=2,
            batch_first=True,
            lambda_init=0.0,
            gamma_init=0.999,
        )

        self.fc1 = nn.Linear(2 * sub_band_rnn_units, freq_dim)
        self.fc2 = nn.Linear(freq_dim, freq_dim)

        self.cnn_block = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.ReLU(),
        )
        self.complex_mask_conv = nn.Conv2d(32, 2, kernel_size=(1, 1), bias=False)

    def _feature_preprocessing(self, x: torch.Tensor):
        real, imag = x.real, x.imag
        c = self.compression_factor
        # 零值显式隔离，避免 0**c 的奇异梯度。
        comp_real = _zero_safe_signed_power(real, c)
        comp_imag = _zero_safe_signed_power(imag, c)
        mag = torch.sqrt(comp_real**2 + comp_imag**2)
        phase = torch.atan2(comp_imag, comp_real)
        return mag, phase, comp_real, comp_imag

    def _intermediate_feature(self, mask: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        inter_r = mask * torch.cos(phase)
        inter_i = mask * torch.sin(phase)
        return torch.stack([inter_r, inter_i], dim=1)

    def _power_law_decompression(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = x.real, x.imag
        inv_c = 1.0 / self.compression_factor
        dec_real = _zero_safe_signed_power(real, inv_c)
        dec_imag = _zero_safe_signed_power(imag, inv_c)
        return torch.complex(dec_real, dec_imag)

    def forward(
        self,
        noisy_spec: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        fs: int | None = None,
    ) -> torch.Tensor:
        """noisy_spec: (B, F, T, 2)；返回 enhanced_spec (B, F, T, 2)。"""
        if noisy_spec.dim() != 4 or noisy_spec.shape[-1] != 2:
            raise ValueError(
                "FastULCNet expects spectrum input (B, F, T, 2). "
                "Ensure YAML has top-level 'stft' block for framework STFT pipeline."
            )
        if noisy_spec.shape[1] != 257:
            raise ValueError(
                "FastULCNet requires exactly 257 frequency bins from n_fft=512; "
                f"got F={noisy_spec.shape[1]}"
            )
        # (B, F, T, 2) -> complex (B, F, T) -> (B, T, F) 供内部 pipeline
        real = noisy_spec[..., 0].float()
        imag = noisy_spec[..., 1].float()
        x_complex = torch.complex(real, imag)
        x_btf = x_complex.permute(0, 2, 1)

        mag, phase, real, imag = self._feature_preprocessing(x_btf)

        # Reorientation -> [B, C, T, F]
        features = self.reorientation(mag)
        features = features.permute(0, 2, 1, 3)

        x = self.conv_block(features)

        B, C, T, F_red = x.shape
        x_rnn = x.permute(0, 2, 3, 1).reshape(B * T, F_red, C)
        frnn_out, _ = self.freq_rnn(x_rnn)
        frnn_out = frnn_out.view(B, T, F_red, -1).permute(0, 3, 1, 2)

        x = F.relu(self.pointwise_conv(frnn_out))
        x = x.permute(0, 2, 3, 1).reshape(B, T, -1)
        sub1, sub2 = torch.chunk(x, 2, dim=-1)
        r1, _ = self.sub_band_rnn1(sub1)
        r2, _ = self.sub_band_rnn2(sub2)
        concatenated = torch.cat([r1, r2], dim=-1)

        mask = F.relu(self.fc1(concatenated))
        mask = F.relu(self.fc2(mask))

        inter_feat = self._intermediate_feature(mask, phase)
        cnn_out = self.cnn_block(inter_feat)
        c_mask = F.relu(self.complex_mask_conv(cnn_out))

        m_real, m_imag = c_mask[:, 0, :, :], c_mask[:, 1, :, :]
        est_comp = self.crm_layer(real, imag, m_real, m_imag)
        est_spec = self._power_law_decompression(est_comp)

        # (B, T, F) complex -> (B, F, T, 2) 与框架约定一致
        est_spec_bft = est_spec.permute(0, 2, 1)
        return torch.view_as_real(est_spec_bft)
