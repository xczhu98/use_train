"""无内部 STFT 的 DCCRN 编解码网络。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_layers import (
    ComplexBatchNorm,
    ComplexConv2d,
    ComplexConvTranspose2d,
    NaiveComplexLSTM,
    complex_cat,
)

_MASKING_MODES = frozenset({"E", "C", "R"})


def apply_complex_mask(
    noisy_spec: torch.Tensor,
    mask_real: torch.Tensor,
    mask_imag: torch.Tensor,
    masking_mode: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    """按 DCCRN 的 E/C/R 规则把复掩码应用到 ``(B,F,T,2)`` 频谱。"""
    if masking_mode not in _MASKING_MODES:
        raise ValueError(f"masking_mode must be one of {sorted(_MASKING_MODES)}")
    noisy_real, noisy_imag = noisy_spec.unbind(dim=-1)
    if masking_mode == "E":
        # ``atan2(0, 0)`` 的反向传播为 NaN；原点处固定相位为 0，不改变前向值。
        mask_is_zero = (mask_real == 0) & (mask_imag == 0)
        noisy_is_zero = (noisy_real == 0) & (noisy_imag == 0)
        safe_mask_real = torch.where(mask_is_zero, torch.ones_like(mask_real), mask_real)
        safe_mask_imag = torch.where(mask_is_zero, torch.zeros_like(mask_imag), mask_imag)
        safe_noisy_real = torch.where(noisy_is_zero, torch.ones_like(noisy_real), noisy_real)
        safe_noisy_imag = torch.where(noisy_is_zero, torch.zeros_like(noisy_imag), noisy_imag)
        mask_magnitude = torch.tanh(
            torch.linalg.vector_norm(torch.stack((mask_real, mask_imag)), dim=0)
        )
        mask_phase = torch.atan2(safe_mask_imag, safe_mask_real)
        noisy_magnitude = torch.sqrt(noisy_real.square() + noisy_imag.square() + eps)
        noisy_phase = torch.atan2(safe_noisy_imag, safe_noisy_real)
        enhanced_magnitude = mask_magnitude * noisy_magnitude
        enhanced_phase = noisy_phase + mask_phase
        enhanced_real = enhanced_magnitude * torch.cos(enhanced_phase)
        enhanced_imag = enhanced_magnitude * torch.sin(enhanced_phase)
    elif masking_mode == "C":
        enhanced_real = noisy_real * mask_real - noisy_imag * mask_imag
        enhanced_imag = noisy_real * mask_imag + noisy_imag * mask_real
    else:
        enhanced_real = noisy_real * mask_real
        enhanced_imag = noisy_imag * mask_imag
    return torch.stack((enhanced_real, enhanced_imag), dim=-1)


class DCCRNCore(nn.Module):
    """DCCRN encoder–RNN–decoder 核心，不执行 STFT/iSTFT。"""

    def __init__(
        self,
        n_fft: int = 512,
        rnn_layers: int = 2,
        rnn_units: int = 256,
        masking_mode: str = "E",
        use_clstm: bool = True,
        use_cbn: bool = False,
        kernel_size: int = 5,
        kernel_num: Sequence[int] = (16, 32, 64, 128, 256, 256),
    ) -> None:
        super().__init__()
        if n_fft <= 0 or n_fft % 2:
            raise ValueError("n_fft must be a positive even integer")
        if rnn_layers <= 0:
            raise ValueError("rnn_layers must be positive")
        if rnn_units <= 0 or rnn_units % 2:
            raise ValueError("rnn_units must be a positive even integer")
        if masking_mode not in _MASKING_MODES:
            raise ValueError(f"masking_mode must be one of {sorted(_MASKING_MODES)}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        channels = tuple(int(value) for value in kernel_num)
        if not channels or any(value <= 0 or value % 2 for value in channels):
            raise ValueError("kernel_num must contain positive even channel counts")
        frequency_without_dc = n_fft // 2
        frequency_divisor = 2 ** len(channels)
        if frequency_without_dc % frequency_divisor:
            raise ValueError(
                "n_fft//2 must be divisible by 2**len(kernel_num) for DCCRN skips"
            )
        self.n_fft = n_fft
        self.masking_mode = masking_mode
        self.use_clstm = use_clstm
        complex_channels = (2, *channels)
        frequency_padding = kernel_size // 2
        norm = ComplexBatchNorm if use_cbn else nn.BatchNorm2d
        self.encoder = nn.ModuleList(
            nn.Sequential(
                ComplexConv2d(
                    complex_channels[index],
                    complex_channels[index + 1],
                    kernel_size=(kernel_size, 2),
                    stride=(2, 1),
                    padding=(frequency_padding, 1),
                    causal=True,
                ),
                norm(complex_channels[index + 1]),
                nn.PReLU(),
            )
            for index in range(len(channels))
        )
        hidden_frequency = frequency_without_dc // frequency_divisor
        recurrent_dim = hidden_frequency * channels[-1]
        if use_clstm:
            recurrent_layers: list[NaiveComplexLSTM] = []
            for index in range(rnn_layers):
                recurrent_layers.append(
                    NaiveComplexLSTM(
                        input_size=recurrent_dim if index == 0 else rnn_units,
                        hidden_size=rnn_units,
                        projection_dim=recurrent_dim if index == rnn_layers - 1 else None,
                    )
                )
            self.enhance = nn.ModuleList(recurrent_layers)
            self.transform = None
        else:
            self.enhance = nn.LSTM(
                input_size=recurrent_dim,
                hidden_size=rnn_units,
                num_layers=rnn_layers,
            )
            self.transform = nn.Linear(rnn_units, recurrent_dim)
        decoder_layers: list[nn.Module] = []
        for index in range(len(channels), 0, -1):
            output_channels = complex_channels[index - 1]
            layers: list[nn.Module] = [
                ComplexConvTranspose2d(
                    complex_channels[index] * 2,
                    output_channels,
                    kernel_size=(kernel_size, 2),
                    stride=(2, 1),
                    padding=(frequency_padding, 0),
                    output_padding=(1, 0),
                )
            ]
            if index != 1:
                layers.extend((norm(output_channels), nn.PReLU()))
            decoder_layers.append(nn.Sequential(*layers))
        self.decoder = nn.ModuleList(decoder_layers)
        self.flatten_parameters()

    def flatten_parameters(self) -> None:
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()
        else:
            for layer in self.enhance:
                layer.flatten_parameters()

    def _run_recurrent(self, encoded: torch.Tensor) -> torch.Tensor:
        batch, channels, frequencies, frames = encoded.shape
        sequence = encoded.permute(3, 0, 1, 2)
        if isinstance(self.enhance, nn.ModuleList):
            real = sequence[:, :, : channels // 2].reshape(
                frames, batch, channels // 2 * frequencies
            )
            imag = sequence[:, :, channels // 2 :].reshape(
                frames, batch, channels // 2 * frequencies
            )
            for layer in self.enhance:
                real, imag = layer((real, imag))
            sequence = torch.cat(
                (
                    real.reshape(frames, batch, channels // 2, frequencies),
                    imag.reshape(frames, batch, channels // 2, frequencies),
                ),
                dim=2,
            )
        else:
            sequence = sequence.reshape(frames, batch, channels * frequencies)
            sequence = self.enhance(sequence)[0]
            sequence = self.transform(sequence)
            sequence = sequence.reshape(frames, batch, channels, frequencies)
        return sequence.permute(1, 2, 3, 0)

    def forward(self, noisy_spec: torch.Tensor) -> torch.Tensor:
        expected_frequencies = self.n_fft // 2 + 1
        if noisy_spec.ndim != 4 or noisy_spec.shape[-1] != 2:
            raise ValueError("DCCRNCore expects noisy_spec shaped (B,F,T,2)")
        if noisy_spec.shape[1] != expected_frequencies:
            raise ValueError(
                f"DCCRNCore expects F={expected_frequencies}, got F={noisy_spec.shape[1]}"
            )
        # 上游网络不编码 DC bin；内部布局为 [real channels, imag channels]。
        encoded = noisy_spec[:, 1:].permute(0, 3, 1, 2)
        skip_connections: list[torch.Tensor] = []
        for layer in self.encoder:
            encoded = layer(encoded)
            skip_connections.append(encoded)
        decoded = self._run_recurrent(encoded)
        for layer, skip in zip(self.decoder, reversed(skip_connections)):
            decoded = layer(complex_cat((decoded, skip), axis=1))
            decoded = decoded[..., 1:]
        mask_real = F.pad(decoded[:, 0], (0, 0, 1, 0))
        mask_imag = F.pad(decoded[:, 1], (0, 0, 1, 0))
        return apply_complex_mask(
            noisy_spec, mask_real, mask_imag, masking_mode=self.masking_mode
        )
