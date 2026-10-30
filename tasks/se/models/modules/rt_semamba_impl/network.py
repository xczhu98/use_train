"""RT-SEMamba 的因果 cTF-Mamba 主体与显式流式状态。

该模块不包含 STFT/iSTFT。参数层级保留官方 KD1/KD8 checkpoint 的命名，
流式推理则把卷积历史、Mamba ``conv_state`` 与 ``ssm_state`` 全部交给调用方。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.models.mixer_seq_simple import _init_weights
    from mamba_ssm.modules.mamba_simple import Mamba
except (ImportError, OSError) as exc:  # pragma: no cover - 由 lazy dependency guard 覆盖
    raise ImportError("RT_SEMamba requires optional dependency mamba_ssm") from exc


MixerFactory = Callable[[int, int, int, int, int], nn.Module]


class RMSNorm(nn.Module):
    """纯 PyTorch RMSNorm；参数名与 Mamba 官方实现兼容且支持 CPU。"""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().square().mean(dim=-1, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + self.eps)
        return normalized.to(dtype=x.dtype) * self.weight


@dataclass(frozen=True)
class RTSEMambaStreamingState:
    """固定大小流式状态；所有 tensor 的容量均与已处理帧数无关。"""

    encoder_conv: tuple[torch.Tensor, ...]
    time_mamba: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    magnitude_conv: tuple[torch.Tensor, ...]
    phase_conv: tuple[torch.Tensor, ...]


def _default_mixer_factory(
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    layer_idx: int,
) -> nn.Module:
    return Mamba(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        layer_idx=layer_idx,
        use_fast_path=False,
    )


class LearnableSigmoid2D(nn.Module):
    """按频点学习 sigmoid 斜率，与官方 mask decoder 参数布局一致。"""

    def __init__(self, in_features: int, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta * torch.sigmoid(self.slope * x)


class ChannelLayerNorm2d(nn.Module):
    """只归一化 channel，因而不聚合未来时间帧。"""

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DenseBlock(nn.Module):
    """四层时间因果 dense convolution，支持显式外置历史帧。"""

    def __init__(self, hid_feature: int, norm_epsilon: float, depth: int = 4):
        super().__init__()
        self.hid_feature = hid_feature
        self.depth = depth
        self.dense_block = nn.ModuleList()
        for index in range(depth):
            dilation = 2**index
            self.dense_block.append(
                nn.Sequential(
                    nn.ConstantPad2d((1, 1, dilation, 0), value=0.0),
                    nn.Conv2d(
                        hid_feature * (index + 1),
                        hid_feature,
                        kernel_size=(2, 3),
                        dilation=(dilation, 1),
                    ),
                    ChannelLayerNorm2d(hid_feature, eps=norm_epsilon),
                    nn.PReLU(hid_feature),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        for layer in self.dense_block:
            x = layer(skip)
            skip = torch.cat([x, skip], dim=1)
        return x

    def init_state(
        self,
        batch_size: int,
        freq_bins: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.zeros(
                batch_size,
                self.hid_feature * (index + 1),
                2**index,
                freq_bins,
                device=device,
                dtype=dtype,
            )
            for index in range(self.depth)
        )

    def forward_frame(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if x.shape[2] != 1 or len(state) != self.depth:
            raise ValueError("DenseBlock.forward_frame expects one frame and one cache per layer")
        skip = x
        next_state: list[torch.Tensor] = []
        for index, layer in enumerate(self.dense_block):
            pad, conv, norm, act = layer
            del pad
            cache = state[index]
            expected = (skip.shape[0], skip.shape[1], 2**index, skip.shape[3])
            if tuple(cache.shape) != expected:
                raise ValueError(
                    f"Dense cache {index} shape mismatch: expected {expected}, got {tuple(cache.shape)}"
                )
            context = torch.cat([cache, skip], dim=2)
            y = conv(F.pad(context, (1, 1, 0, 0)))
            y = act(norm(y))
            next_state.append(context[:, :, -2**index :, :])
            skip = torch.cat([y, skip], dim=1)
        return y, tuple(next_state)


class DenseEncoder(nn.Module):
    """Magnitude/phase encoder。"""

    def __init__(self, hid_feature: int, norm_epsilon: float):
        super().__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(2, hid_feature, (1, 1)),
            ChannelLayerNorm2d(hid_feature, eps=norm_epsilon),
            nn.PReLU(hid_feature),
        )
        self.dense_block = DenseBlock(hid_feature, norm_epsilon)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(hid_feature, hid_feature, (1, 3), (1, 2), padding=(0, 1)),
            ChannelLayerNorm2d(hid_feature, eps=norm_epsilon),
            nn.PReLU(hid_feature),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_conv_2(self.dense_block(self.dense_conv_1(x)))

    def forward_frame(
        self, x: torch.Tensor, state: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x, next_state = self.dense_block.forward_frame(self.dense_conv_1(x), state)
        return self.dense_conv_2(x), next_state


class SPConvTranspose2d(nn.Module):
    """官方 frequency sub-pixel upsampling。"""

    def __init__(self, in_channels: int, out_channels: int, r: int = 2):
        super().__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=(1, 3))
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(self.pad1(x))
        batch, channels, time, freq = out.shape
        out = out.view(batch, self.r, channels // self.r, time, freq)
        return out.permute(0, 2, 3, 4, 1).contiguous().view(
            batch, channels // self.r, time, -1
        )


class MagDecoder(nn.Module):
    """预测压缩幅度 mask。"""

    def __init__(
        self, hid_feature: int, norm_epsilon: float, n_fft: int, beta: float
    ):
        super().__init__()
        self.dense_block = DenseBlock(hid_feature, norm_epsilon)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(hid_feature, hid_feature, 2),
            ChannelLayerNorm2d(hid_feature, eps=norm_epsilon),
            nn.PReLU(hid_feature),
            nn.Conv2d(hid_feature, 1, (1, 2)),
        )
        self.lsigmoid = LearnableSigmoid2D(n_fft // 2 + 1, beta=beta)

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mask_conv(x).permute(0, 3, 2, 1).squeeze(-1)
        return self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._decode(self.dense_block(x))

    def forward_frame(
        self, x: torch.Tensor, state: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x, next_state = self.dense_block.forward_frame(x, state)
        return self._decode(x), next_state


class PhaseDecoder(nn.Module):
    """以实部/虚部头的 atan2 预测相位。"""

    def __init__(self, hid_feature: int, norm_epsilon: float):
        super().__init__()
        self.dense_block = DenseBlock(hid_feature, norm_epsilon)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(hid_feature, hid_feature, 2),
            ChannelLayerNorm2d(hid_feature, eps=norm_epsilon),
            nn.PReLU(hid_feature),
        )
        self.phase_conv_r = nn.Conv2d(hid_feature, 1, (1, 2))
        self.phase_conv_i = nn.Conv2d(hid_feature, 1, (1, 2))

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.phase_conv(x)
        return torch.atan2(self.phase_conv_i(x), self.phase_conv_r(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._decode(self.dense_block(x))

    def forward_frame(
        self, x: torch.Tensor, state: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x, next_state = self.dense_block.forward_frame(x, state)
        return self._decode(x), next_state


def _mamba_step_reference(
    mixer: Mamba,
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """无 Triton 的单帧 Mamba 更新；不原地修改调用方持有的 state。"""
    dtype = hidden_states.dtype
    if hidden_states.shape[1] != 1:
        raise ValueError("Mamba streaming step expects exactly one token")
    xz = mixer.in_proj(hidden_states.squeeze(1))
    x, z = xz.chunk(2, dim=-1)

    next_conv = torch.cat((conv_state[:, :, 1:], x.unsqueeze(-1)), dim=-1)
    x = torch.sum(next_conv * mixer.conv1d.weight.squeeze(1), dim=-1)
    if mixer.conv1d.bias is not None:
        x = x + mixer.conv1d.bias
    x = mixer.act(x).to(dtype=dtype)

    x_db = mixer.x_proj(x)
    dt, b_state, c_state = torch.split(
        x_db, [mixer.dt_rank, mixer.d_state, mixer.d_state], dim=-1
    )
    dt = F.linear(dt, mixer.dt_proj.weight)
    transition = -torch.exp(mixer.A_log.float())
    dt = F.softplus(dt + mixer.dt_proj.bias.to(dtype=dt.dtype))
    discrete_a = torch.exp(torch.einsum("bd,dn->bdn", dt, transition))
    discrete_b = torch.einsum("bd,bn->bdn", dt, b_state)
    next_ssm = ssm_state * discrete_a + x.unsqueeze(-1) * discrete_b
    y = torch.einsum("bdn,bn->bd", next_ssm.to(dtype), c_state)
    y = (y + mixer.D.to(dtype) * x) * mixer.act(z)
    return mixer.out_proj(y).unsqueeze(1), next_conv, next_ssm


class _MambaLayer(nn.Module):
    """保持官方 ``Block.mixer``/``Block.norm`` state_dict 路径。"""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        norm_epsilon: float,
        layer_idx: int,
        mixer_factory: MixerFactory,
    ):
        super().__init__()
        self.mixer = mixer_factory(d_model, d_state, d_conv, expand, layer_idx)
        self.norm = RMSNorm(d_model, eps=norm_epsilon)

    def forward(
        self, hidden: torch.Tensor, residual: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden + residual if residual is not None else hidden
        return self.mixer(self.norm(residual)), residual

    def step(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden + residual if residual is not None else hidden
        conv_state, ssm_state = state
        normalized = self.norm(residual)
        if isinstance(self.mixer, Mamba):
            hidden, next_conv, next_ssm = _mamba_step_reference(
                self.mixer, normalized, conv_state, ssm_state
            )
        else:
            hidden, next_conv, next_ssm = self.mixer.step(
                normalized, conv_state, ssm_state
            )
        return hidden, residual, (next_conv, next_ssm)


class MambaBlock(nn.Module):
    """单向或双向 Mamba scan。"""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        norm_epsilon: float,
        bidirectional: bool,
        layer_offset: int,
        mixer_factory: MixerFactory,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.forward_blocks = nn.ModuleList(
            [
                _MambaLayer(
                    d_model,
                    d_state,
                    d_conv,
                    expand,
                    norm_epsilon,
                    layer_offset,
                    mixer_factory,
                )
            ]
        )
        self.backward_blocks = (
            nn.ModuleList(
                [
                    _MambaLayer(
                        d_model,
                        d_state,
                        d_conv,
                        expand,
                        norm_epsilon,
                        layer_offset + 1,
                        mixer_factory,
                    )
                ]
            )
            if bidirectional
            else None
        )
        self.total_layers = 2 if bidirectional else 1

    @staticmethod
    def _finish(hidden: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
        return hidden + residual if residual is not None else hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, residual = x, None
        for layer in self.forward_blocks:
            hidden, residual = layer(hidden, residual)
        forward = self._finish(hidden, residual)
        if not self.bidirectional:
            return forward
        hidden, residual = torch.flip(x, [1]), None
        assert self.backward_blocks is not None
        for layer in self.backward_blocks:
            hidden, residual = layer(hidden, residual)
        backward = torch.flip(self._finish(hidden, residual), [1])
        return torch.cat([forward, backward], dim=-1)

    def step(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.bidirectional or len(self.forward_blocks) != 1:
            raise RuntimeError("MambaBlock.step only supports one unidirectional layer")
        hidden, residual, next_state = self.forward_blocks[0].step(x, None, state)
        return self._finish(hidden, residual), next_state


class TFMambaBlock(nn.Module):
    """时间单向、当前帧频率双向的 cTF-Mamba block。"""

    def __init__(
        self,
        hid_feature: int,
        d_state: int,
        d_conv: int,
        expand: int,
        norm_epsilon: float,
        layer_offset: int,
        mixer_factory: MixerFactory,
    ):
        super().__init__()
        self.time_mamba = MambaBlock(
            hid_feature,
            d_state,
            d_conv,
            expand,
            norm_epsilon,
            False,
            layer_offset,
            mixer_factory,
        )
        self.freq_mamba = MambaBlock(
            hid_feature,
            d_state,
            d_conv,
            expand,
            norm_epsilon,
            True,
            layer_offset + 1,
            mixer_factory,
        )
        self.total_layers = 3
        self.tlinear = nn.ConvTranspose1d(hid_feature, hid_feature, 1)
        self.flinear = nn.ConvTranspose1d(hid_feature * 2, hid_feature, 1)

    def _frequency_forward(self, x: torch.Tensor, batch: int, time: int) -> torch.Tensor:
        _, freq, channels = x.shape
        freq_residual = x
        x = self.freq_mamba(x)
        x = self.flinear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return (freq_residual + x).view(batch, time, freq, channels).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, time, freq = x.shape
        temporal = x.permute(0, 3, 2, 1).contiguous().view(batch * freq, time, channels)
        temporal_out = self.time_mamba(temporal)
        temporal_out = self.tlinear(temporal_out.permute(0, 2, 1)).permute(0, 2, 1)
        temporal = temporal + temporal_out
        frequency = temporal.view(batch, freq, time, channels).permute(0, 2, 1, 3)
        return self._frequency_forward(
            frequency.contiguous().view(batch * time, freq, channels), batch, time
        )

    def forward_frame(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, channels, time, freq = x.shape
        if time != 1:
            raise ValueError("TFMambaBlock.forward_frame expects exactly one frame")
        temporal = x.permute(0, 3, 2, 1).contiguous().view(batch * freq, 1, channels)
        temporal_out, next_state = self.time_mamba.step(temporal, state)
        temporal_out = self.tlinear(temporal_out.permute(0, 2, 1)).permute(0, 2, 1)
        temporal = temporal + temporal_out
        frequency = temporal.view(batch, freq, 1, channels).permute(0, 2, 1, 3)
        return (
            self._frequency_forward(frequency.reshape(batch, freq, channels), batch, 1),
            next_state,
        )


class MLPWithNorm(nn.Module):
    """每个 cTF-Mamba 后的 residual MLP。"""

    def __init__(self, hid_feature: int, norm_epsilon: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hid_feature, hid_feature * 4),
            nn.GELU(),
            nn.Linear(hid_feature * 4, hid_feature),
            nn.Dropout(0.1),
        )
        self.norm = nn.LayerNorm(hid_feature, eps=norm_epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, time, freq = x.shape
        flat = x.permute(0, 2, 3, 1).contiguous().view(-1, channels)
        update = self.norm(self.mlp(flat)).view(batch, time, freq, channels)
        return x + update.permute(0, 3, 1, 2)


class RTSEMambaNetwork(nn.Module):
    """官方纯 Mamba KD1/KD8 网络主体。"""

    def __init__(
        self,
        *,
        num_tfmamba: int,
        n_fft: int = 400,
        hid_feature: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 4,
        norm_epsilon: float = 1e-5,
        beta: float = 2.0,
        mixer_factory: MixerFactory | None = None,
    ):
        super().__init__()
        if num_tfmamba <= 0:
            raise ValueError(f"num_tfmamba must be positive, got {num_tfmamba}")
        self.num_tscblocks = num_tfmamba
        self.n_fft = n_fft
        self.hid_feature = hid_feature
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        factory = mixer_factory or _default_mixer_factory

        self.dense_encoder = DenseEncoder(hid_feature, norm_epsilon)
        self.TSMamba = nn.ModuleList()
        layer_offset = 0
        for _ in range(num_tfmamba):
            block = TFMambaBlock(
                hid_feature,
                d_state,
                d_conv,
                expand,
                norm_epsilon,
                layer_offset,
                factory,
            )
            self.TSMamba.append(block)
            layer_offset += block.total_layers
        self.mlp_after_mamba = nn.ModuleList(
            MLPWithNorm(hid_feature, norm_epsilon) for _ in range(num_tfmamba)
        )
        self.mask_decoder = MagDecoder(hid_feature, norm_epsilon, n_fft, beta)
        self.phase_decoder = PhaseDecoder(hid_feature, norm_epsilon)
        if mixer_factory is None:
            self.apply(partial(_init_weights, n_layer=1))

    @property
    def encoded_freq_bins(self) -> int:
        return (self.n_fft // 2 + 1 + 1) // 2

    def init_streaming_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> RTSEMambaStreamingState:
        freq_in = self.n_fft // 2 + 1
        freq_hidden = self.encoded_freq_bins
        encoder = self.dense_encoder.dense_block.init_state(
            batch_size, freq_in, device=device, dtype=dtype
        )
        magnitude = self.mask_decoder.dense_block.init_state(
            batch_size, freq_hidden, device=device, dtype=dtype
        )
        phase = self.phase_decoder.dense_block.init_state(
            batch_size, freq_hidden, device=device, dtype=dtype
        )
        mamba_states: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.TSMamba:
            mixer = block.time_mamba.forward_blocks[0].mixer
            conv, ssm = mixer.allocate_inference_cache(
                batch_size=batch_size * freq_hidden,
                max_seqlen=1,
                dtype=dtype,
                device=device,
            )
            mamba_states.append((conv, ssm))
        return RTSEMambaStreamingState(
            encoder_conv=encoder,
            time_mamba=tuple(mamba_states),
            magnitude_conv=magnitude,
            phase_conv=phase,
        )

    @staticmethod
    def _features(
        blocks: list[torch.Tensor], x: torch.Tensor
    ) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        return {"tfmamba_blocks": blocks, "pre_decoder": x}

    def _decode(
        self,
        x: torch.Tensor,
        noisy_mag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noisy_mag_4d = noisy_mag.permute(0, 2, 1).unsqueeze(1)
        denoised_mag = (self.mask_decoder(x) * noisy_mag_4d).squeeze(1).permute(0, 2, 1)
        denoised_pha = self.phase_decoder(x).squeeze(1).permute(0, 2, 1)
        denoised_com = torch.stack(
            (
                denoised_mag * torch.cos(denoised_pha),
                denoised_mag * torch.sin(denoised_pha),
            ),
            dim=-1,
        )
        return denoised_mag, denoised_pha, denoised_com

    def forward(
        self,
        noisy_mag: torch.Tensor,
        noisy_pha: torch.Tensor,
        *,
        return_features: bool = False,
    ):
        mag_4d = noisy_mag.permute(0, 2, 1).unsqueeze(1)
        pha_4d = noisy_pha.permute(0, 2, 1).unsqueeze(1)
        x = self.dense_encoder(torch.cat((mag_4d, pha_4d), dim=1))
        features: list[torch.Tensor] = []
        for block, mlp in zip(self.TSMamba, self.mlp_after_mamba):
            x = mlp(block(x))
            features.append(x)
        outputs = self._decode(x, noisy_mag)
        if return_features:
            return (*outputs, self._features(features, x))
        return outputs

    def forward_stream_frame(
        self,
        noisy_mag: torch.Tensor,
        noisy_pha: torch.Tensor,
        state: RTSEMambaStreamingState,
        *,
        return_features: bool = False,
    ):
        if noisy_mag.ndim != 3 or noisy_mag.shape[-1] != 1:
            raise ValueError("stream frame magnitude must have shape (B,F,1)")
        mag_4d = noisy_mag.permute(0, 2, 1).unsqueeze(1)
        pha_4d = noisy_pha.permute(0, 2, 1).unsqueeze(1)
        x, encoder_state = self.dense_encoder.forward_frame(
            torch.cat((mag_4d, pha_4d), dim=1), state.encoder_conv
        )
        features: list[torch.Tensor] = []
        mamba_states: list[tuple[torch.Tensor, torch.Tensor]] = []
        if len(state.time_mamba) != len(self.TSMamba):
            raise ValueError("stream state block count does not match model depth")
        for block, mlp, block_state in zip(
            self.TSMamba, self.mlp_after_mamba, state.time_mamba
        ):
            x, next_block_state = block.forward_frame(x, block_state)
            x = mlp(x)
            features.append(x)
            mamba_states.append(next_block_state)
        mag_mask, magnitude_state = self.mask_decoder.forward_frame(
            x, state.magnitude_conv
        )
        pha_raw, phase_state = self.phase_decoder.forward_frame(x, state.phase_conv)
        denoised_mag = (mag_mask * mag_4d).squeeze(1).permute(0, 2, 1)
        denoised_pha = pha_raw.squeeze(1).permute(0, 2, 1)
        denoised_com = torch.stack(
            (
                denoised_mag * torch.cos(denoised_pha),
                denoised_mag * torch.sin(denoised_pha),
            ),
            dim=-1,
        )
        next_state = RTSEMambaStreamingState(
            encoder_conv=encoder_state,
            time_mamba=tuple(mamba_states),
            magnitude_conv=magnitude_state,
            phase_conv=phase_state,
        )
        outputs = (denoised_mag, denoised_pha, denoised_com, next_state)
        if return_features:
            return (*outputs, self._features(features, x))
        return outputs

    def forward_stream_chunk(
        self,
        noisy_mag: torch.Tensor,
        noisy_pha: torch.Tensor,
        state: RTSEMambaStreamingState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RTSEMambaStreamingState]:
        if noisy_mag.shape != noisy_pha.shape or noisy_mag.ndim != 3:
            raise ValueError("stream chunk magnitude/phase must share shape (B,F,T)")
        mags: list[torch.Tensor] = []
        phases: list[torch.Tensor] = []
        complexes: list[torch.Tensor] = []
        next_state = state
        for frame_index in range(noisy_mag.shape[-1]):
            mag, pha, com, next_state = self.forward_stream_frame(
                noisy_mag[:, :, frame_index : frame_index + 1],
                noisy_pha[:, :, frame_index : frame_index + 1],
                next_state,
            )
            mags.append(mag)
            phases.append(pha)
            complexes.append(com)
        return (
            torch.cat(mags, dim=-1),
            torch.cat(phases, dim=-1),
            torch.cat(complexes, dim=2),
            next_state,
        )
