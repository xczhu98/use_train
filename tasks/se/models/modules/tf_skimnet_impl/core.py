"""TF-SkiMNet 的时频 skipping-memory 主干。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """只在时间轴左侧补零的一维因果卷积。"""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        *,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.kernel_size - 1, 0)))


class CausalPointwiseEncoder(nn.Module):
    """逐频点执行的因果时间卷积。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frequency, frames, channels = x.shape
        sequence = x.reshape(batch * frequency, frames, channels).transpose(1, 2)
        sequence = self.conv(F.pad(sequence, (self.kernel_size - 1, 0)))
        return sequence.transpose(1, 2).reshape(batch, frequency, frames, -1)


class LocalFrequencyModule(nn.Module):
    """以 stride 1 保持频率分辨率的邻频卷积。"""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.permute(0, 3, 2, 1)
        x = self.activation(self.conv(x))
        return x.permute(0, 3, 2, 1) + residual


class GlobalFrequencyModule(nn.Module):
    """逐帧沿频率轴运行 BiLSTM，并导出 frame-wise hidden/cell。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=channels,
            batch_first=True,
            bidirectional=True,
        )
        self.projection = nn.Linear(2 * channels, channels)

    def _initial_state(
        self,
        state: tuple[torch.Tensor, torch.Tensor] | None,
        *,
        batch: int,
        frames: int,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            shape = (2, batch * frames, self.channels)
            zeros = reference.new_zeros(shape)
            return zeros, zeros.clone()

        hidden, cell = state
        expected = (batch, frames, 2 * self.channels)
        if hidden.shape != expected or cell.shape != expected:
            raise ValueError(
                "TF-SkiM state must have shape "
                f"{expected}, got hidden={tuple(hidden.shape)}, cell={tuple(cell.shape)}"
            )
        hidden = hidden.reshape(batch * frames, 2, self.channels).permute(1, 0, 2)
        cell = cell.reshape(batch * frames, 2, self.channels).permute(1, 0, 2)
        return hidden.contiguous(), cell.contiguous()

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, frequency, frames, channels = x.shape
        residual = x
        sequence = self.norm(x.permute(0, 2, 1, 3)).reshape(
            batch * frames, frequency, channels
        )
        initial = self._initial_state(
            state,
            batch=batch,
            frames=frames,
            reference=sequence,
        )
        sequence, (hidden, cell) = self.lstm(sequence, initial)
        sequence = self.projection(sequence)
        sequence = sequence.reshape(batch, frames, frequency, channels).permute(0, 2, 1, 3)

        hidden = hidden.permute(1, 0, 2).reshape(batch, frames, 2 * channels)
        cell = cell.permute(1, 0, 2).reshape(batch, frames, 2 * channels)
        return sequence + residual, (hidden, cell)


class TemporalStateModule(nn.Module):
    """沿时间递推 frame-wise hidden 或 cell embedding。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2 * channels,
            hidden_size=2 * channels,
            batch_first=True,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(2 * channels)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(state)
        return self.norm(output) + state


class LocalTemporalModule(nn.Module):
    """逐频点执行 grouped causal Conv1d 的局部时间建模。"""

    def __init__(self, channels: int, kernel_size: int, groups: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.conv = CausalConv1d(channels, kernel_size, groups=groups)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frequency, frames, channels = x.shape
        residual = x
        sequence = self.norm(x).reshape(batch * frequency, frames, channels)
        sequence = sequence.transpose(1, 2)
        sequence = self.activation(self.conv(sequence))
        sequence = sequence.transpose(1, 2).reshape(batch, frequency, frames, channels)
        return sequence + residual


class TFSkiMBlock(nn.Module):
    """Local-F、Global-F 和可选 Local-T 组成的单个 TF-SkiM block。"""

    def __init__(
        self,
        channels: int,
        *,
        local_frequency_kernel: int,
        local_temporal_kernel: int,
        local_temporal_groups: int,
        use_local_temporal: bool,
    ) -> None:
        super().__init__()
        self.local_frequency = LocalFrequencyModule(channels, local_frequency_kernel)
        self.global_frequency = GlobalFrequencyModule(channels)
        self.local_temporal = (
            LocalTemporalModule(
                channels,
                local_temporal_kernel,
                local_temporal_groups,
            )
            if use_local_temporal
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.local_frequency(x)
        x, state = self.global_frequency(x, state)
        if self.local_temporal is not None:
            x = self.local_temporal(x)
        return x, state


class TFSkiMBackbone(nn.Module):
    """官方 TF-SkiMNet 拓扑的框架无关谱特征主干。"""

    def __init__(
        self,
        *,
        num_blocks: int,
        input_channels: int,
        output_channels: int,
        hidden_size: int,
        encoder_kernel_size: int,
        local_frequency_kernel: int,
        local_temporal_kernel: int,
        local_temporal_groups: int,
        use_hidden_state_memory: bool,
        use_cell_state_memory: bool,
        use_global_temporal: bool,
        use_local_temporal: bool,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.use_global_temporal = use_global_temporal
        self.encoder = CausalPointwiseEncoder(
            input_channels,
            hidden_size,
            encoder_kernel_size,
        )
        self.blocks = nn.ModuleList(
            [
                TFSkiMBlock(
                    hidden_size,
                    local_frequency_kernel=local_frequency_kernel,
                    local_temporal_kernel=local_temporal_kernel,
                    local_temporal_groups=local_temporal_groups,
                    use_local_temporal=use_local_temporal,
                )
                for _ in range(num_blocks)
            ]
        )
        transition_count = num_blocks - 1
        self.hidden_state_networks = nn.ModuleList(
            [TemporalStateModule(hidden_size) for _ in range(transition_count)]
            if use_global_temporal and use_hidden_state_memory
            else []
        )
        self.cell_state_networks = nn.ModuleList(
            [TemporalStateModule(hidden_size) for _ in range(transition_count)]
            if use_global_temporal and use_cell_state_memory
            else []
        )
        self.use_hidden_state_memory = use_hidden_state_memory
        self.use_cell_state_memory = use_cell_state_memory
        self.decoder = CausalPointwiseEncoder(
            hidden_size,
            output_channels,
            encoder_kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        state: tuple[torch.Tensor, torch.Tensor] | None = None
        for index, block in enumerate(self.blocks):
            x, (hidden, cell) = block(x, state)
            if not self.use_global_temporal or index == self.num_blocks - 1:
                state = None
                continue
            if self.use_hidden_state_memory:
                hidden = self.hidden_state_networks[index](hidden)
            if self.use_cell_state_memory:
                cell = self.cell_state_networks[index](cell)
            state = (hidden, cell)
        return self.decoder(x)
