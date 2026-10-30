"""MeanSE 的纯 PyTorch complex-STFT 项目实现。

本模块参考 Wang et al., "MeanSE: Efficient Generative Speech Enhancement with
Mean Flows"（arXiv:2509.21214）及其官方仓库固定 commit
``a4ced60cf3aa365c481319632e8ea762d178922b``，按本仓谱进谱出接口重新实现。

MeanSE 与本仓已有的 MeanFlowSE（Li et al., arXiv:2509.14858）不是同一篇论文、
同一个官方仓库或同一份实现。这里不复制上游 Lightning、ESPnet、数据模拟和
checkpoint 代码，也不声明官方 checkpoint 兼容性。来源与许可证状态见
同目录 ``PROVENANCE``。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.registry import MODEL_REGISTRY
from tasks.se.models.base_model import BaseSEModel
from tasks.se.models.generative.meanflowse.model import (
    GaussianFourierEmbedding,
    ResidualBlock,
)


def _as_complex(spec: torch.Tensor) -> torch.Tensor:
    """把框架 ``(B,F,T,2)`` 谱转换为 complex tensor。"""
    if spec.is_complex():
        if spec.ndim != 3:
            raise ValueError(
                f"complex spectrum must have shape (B,F,T), got {tuple(spec.shape)}"
            )
        return spec
    if spec.ndim != 4 or spec.shape[-1] != 2:
        raise ValueError(
            f"spectrum must have shape (B,F,T,2), got {tuple(spec.shape)}"
        )
    return torch.view_as_complex(spec.float().contiguous())


def _as_real(spec: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(spec.contiguous()) if spec.is_complex() else spec


def spectral_transform(
    spec: torch.Tensor,
    *,
    factor: float = 0.065,
    exponent: float = 0.667,
) -> torch.Tensor:
    """执行官方配置使用的幅度幂压缩并保留相位。"""
    if factor <= 0.0 or exponent <= 0.0:
        raise ValueError("factor and exponent must be positive")
    value = _as_complex(spec)
    return factor * value.abs().pow(exponent) * torch.exp(1j * torch.angle(value))


def inverse_spectral_transform(
    spec: torch.Tensor,
    *,
    factor: float = 0.065,
    exponent: float = 0.667,
) -> torch.Tensor:
    """逆转幅度幂压缩，保持输入的 complex/real-pair 表示。"""
    if factor <= 0.0 or exponent <= 0.0:
        raise ValueError("factor and exponent must be positive")
    was_complex = spec.is_complex()
    value = _as_complex(spec) / factor
    restored = value.abs().pow(1.0 / exponent) * torch.exp(1j * torch.angle(value))
    return restored if was_complex else _as_real(restored)


class JVPCompatibleSpatialAttention(nn.Module):
    """不用 fused SDPA 的显式多头注意力，保留 JVP/二阶导支持。"""

    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        if channels % num_heads:
            raise ValueError(
                f"channels={channels} must be divisible by num_heads={num_heads}"
            )
        groups = next(
            (value for value in (32, 16, 8, 4, 2) if channels % value == 0),
            1,
        )
        self.num_heads = int(num_heads)
        self.head_dim = channels // self.num_heads
        self.norm = nn.GroupNorm(groups, channels, eps=1.0e-6)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, freq, frames = x.shape
        sequence = self.norm(x).flatten(2).transpose(1, 2)
        q, k, v = self.qkv(sequence).chunk(3, dim=-1)
        tokens = sequence.shape[1]
        q = q.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        weights = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1
        )
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(
            batch, tokens, channels
        )
        attended = self.proj(attended).transpose(1, 2).reshape(
            batch, channels, freq, frames
        )
        return (x + attended) / math.sqrt(2.0)


class MeanSENCSNppBackbone(nn.Module):
    """带共享 ``t/r`` 编码和线性融合的 NCSN++ 风格 U-Net。"""

    def __init__(
        self,
        *,
        nf: int = 128,
        ch_mult: Sequence[int] = (1, 1, 2, 2, 2, 2, 2),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        fourier_scale: float = 16.0,
        attention_heads: int = 4,
    ):
        super().__init__()
        if nf < 4 or nf % 2:
            raise ValueError(f"nf must be an even integer >= 4, got {nf}")
        if not ch_mult or any(int(mult) <= 0 for mult in ch_mult):
            raise ValueError(f"ch_mult must contain positive integers, got {ch_mult}")
        if num_res_blocks < 1:
            raise ValueError(f"num_res_blocks must be >= 1, got {num_res_blocks}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        channels = [nf * int(mult) for mult in ch_mult]
        temb_channels = nf * 4
        self.pad_multiple = 2 ** (len(channels) - 1)
        self.time_embed = GaussianFourierEmbedding(nf, fourier_scale)
        self.time_mlp = nn.Sequential(
            nn.Linear(nf, temb_channels),
            nn.SiLU(),
            nn.Linear(temb_channels, temb_channels),
        )
        self.time_fuse = nn.Linear(2 * temb_channels, temb_channels)
        with torch.no_grad():
            self.time_fuse.weight.zero_()
            self.time_fuse.weight[:, :temb_channels].copy_(
                torch.eye(temb_channels)
            )
            self.time_fuse.bias.zero_()

        self.input_conv = nn.Conv2d(4, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_channels = channels[0]
        for level, out_channels in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    ResidualBlock(in_channels, out_channels, temb_channels, dropout)
                )
                in_channels = out_channels
            self.down_blocks.append(blocks)
            if level < len(channels) - 1:
                self.downsamples.append(
                    nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
                )

        self.mid1 = ResidualBlock(in_channels, in_channels, temb_channels, dropout)
        self.mid_attn = JVPCompatibleSpatialAttention(in_channels, attention_heads)
        self.mid2 = ResidualBlock(in_channels, in_channels, temb_channels, dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(len(channels))):
            out_channels = channels[level]
            blocks = nn.ModuleList(
                [
                    ResidualBlock(
                        in_channels + out_channels,
                        out_channels,
                        temb_channels,
                        dropout,
                    )
                ]
            )
            in_channels = out_channels
            for _ in range(num_res_blocks - 1):
                blocks.append(
                    ResidualBlock(in_channels, out_channels, temb_channels, dropout)
                )
            self.up_blocks.append(blocks)
            if level > 0:
                self.upsamples.append(
                    nn.Conv2d(in_channels, in_channels, 3, padding=1)
                )

        groups = next(
            (value for value in (32, 16, 8, 4, 2) if channels[0] % value == 0),
            1,
        )
        self.output_norm = nn.GroupNorm(groups, channels[0], eps=1.0e-6)
        self.output_conv = nn.Conv2d(channels[0], 2, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def fused_time_embedding(
        self, time_t: torch.Tensor, time_r: torch.Tensor
    ) -> torch.Tensor:
        """共享编码 ``t/r``，并线性融合回单时间 embedding 维度。"""
        time_embedding = self.time_mlp(self.time_embed(time_t))
        interval_embedding = self.time_mlp(self.time_embed(time_r))
        return self.time_fuse(torch.cat((time_embedding, interval_embedding), dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        time_r: torch.Tensor,
        time_t: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(
                f"backbone input must have shape (B,4,F,T), got {tuple(x.shape)}"
            )
        if time_r.shape != time_t.shape or torch.any(time_r > time_t):
            raise ValueError("time_r and time_t must have equal shapes and satisfy r <= t")
        temb = self.fused_time_embedding(time_t, time_r)
        freq, frames = x.shape[-2:]
        pad_f = (-freq) % self.pad_multiple
        pad_t = (-frames) % self.pad_multiple
        h = self.input_conv(F.pad(x, (0, pad_t, 0, pad_f)))

        skips: list[torch.Tensor] = []
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, temb)
            skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        upsample_index = 0
        for reverse_level, blocks in enumerate(self.up_blocks):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat((h, skip), dim=1)
            for block in blocks:
                h = block(h, temb)
            if reverse_level < len(self.upsamples):
                h = F.interpolate(h, scale_factor=2.0, mode="nearest")
                h = self.upsamples[upsample_index](h)
                upsample_index += 1

        h = F.silu(self.output_norm(h))
        h = h / time_t.clamp_min(1.0e-4)[:, None, None, None]
        output = self.output_conv(h)
        return output[:, :, :freq, :frames]


@MODEL_REGISTRY.register("MeanSE")
class MeanSE(BaseSEModel):
    """MeanSE 生成式 complex-STFT 语音增强模型。

    推理遵循标准谱进谱出接口；训练时由 ``MeanFlowSETrainer`` 显式传入
    ``clean_spec``，模型返回 scalar MeanFlow objective 与 diagnostics。这是生成式
    loss-returning 特例，而非普通判别式 SE forward。

    本类实现 Wang et al. 的 MeanSE 路线；它与本仓 ``MeanFlowSE`` 类对应的
    Li et al. 工作不是同一篇论文、同一官方仓库或同一实现。
    """

    def __init__(
        self,
        *,
        n_fft: int = 1022,
        hop_length: int = 320,
        win_length: int = 1022,
        window: str = "hann",
        center: bool = True,
        strict_preset: bool = False,
        spec_factor: float = 0.065,
        spec_abs_exponent: float = 0.667,
        sigma: float = 0.5,
        t_eps: float = 0.0,
        t_max: float = 1.0,
        flow_ratio: float = 0.75,
        max_interval: float = 1.0,
        loss_type: str = "adaptive_l2",
        adaptive_l2_c: float = 1.0e-3,
        sampling_steps: int = 1,
        inference_seed: int = 42,
        nf: int = 128,
        ch_mult: Sequence[int] = (1, 1, 2, 2, 2, 2, 2),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        fourier_scale: float = 16.0,
        attention_heads: int = 4,
    ):
        super().__init__()
        self._validate_config(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            strict_preset=strict_preset,
            spec_factor=spec_factor,
            spec_abs_exponent=spec_abs_exponent,
            sigma=sigma,
            t_eps=t_eps,
            t_max=t_max,
            flow_ratio=flow_ratio,
            max_interval=max_interval,
            loss_type=loss_type,
            adaptive_l2_c=adaptive_l2_c,
            sampling_steps=sampling_steps,
            inference_seed=inference_seed,
        )
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.expected_freq_bins = self.n_fft // 2 + 1
        self.spec_factor = float(spec_factor)
        self.spec_abs_exponent = float(spec_abs_exponent)
        self.sigma = float(sigma)
        self.t_eps = float(t_eps)
        self.t_max = float(t_max)
        self.flow_ratio = float(flow_ratio)
        self.max_interval = float(max_interval)
        self.loss_type = str(loss_type)
        self.adaptive_l2_c = float(adaptive_l2_c)
        self.sampling_steps = int(sampling_steps)
        self.inference_seed = int(inference_seed)
        self.backbone = MeanSENCSNppBackbone(
            nf=nf,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            fourier_scale=fourier_scale,
            attention_heads=attention_heads,
        )

    @staticmethod
    def _validate_config(**cfg: object) -> None:
        if int(cfg["n_fft"]) <= 0 or int(cfg["n_fft"]) % 2:
            raise ValueError("n_fft must be a positive even integer")
        if int(cfg["hop_length"]) <= 0:
            raise ValueError("hop_length must be positive")
        if int(cfg["win_length"]) != int(cfg["n_fft"]):
            raise ValueError("MeanSE requires win_length == n_fft")
        if str(cfg["window"]).lower() != "hann":
            raise ValueError("MeanSE supports only the official hann window")
        if not bool(cfg["center"]):
            raise ValueError("MeanSE requires center=True")
        if float(cfg["spec_factor"]) <= 0.0 or float(cfg["spec_abs_exponent"]) <= 0.0:
            raise ValueError("spec_factor and spec_abs_exponent must be positive")
        if float(cfg["sigma"]) < 0.0:
            raise ValueError("sigma must be non-negative")
        if not 0.0 <= float(cfg["t_eps"]) < float(cfg["t_max"]) <= 1.0:
            raise ValueError("times must satisfy 0 <= t_eps < t_max <= 1")
        if not 0.0 <= float(cfg["flow_ratio"]) <= 1.0:
            raise ValueError("flow_ratio must be in [0, 1]")
        if not 0.0 < float(cfg["max_interval"]) <= 1.0:
            raise ValueError("max_interval must be in (0, 1]")
        if str(cfg["loss_type"]) not in {"mse", "mae", "adaptive_l2"}:
            raise ValueError("loss_type must be one of: mse, mae, adaptive_l2")
        if float(cfg["adaptive_l2_c"]) <= 0.0:
            raise ValueError("adaptive_l2_c must be positive")
        if int(cfg["sampling_steps"]) < 1:
            raise ValueError("sampling_steps must be >= 1")
        if int(cfg["inference_seed"]) < 0:
            raise ValueError("inference_seed must be non-negative")
        if bool(cfg["strict_preset"]):
            actual = (
                int(cfg["n_fft"]),
                int(cfg["hop_length"]),
                int(cfg["win_length"]),
                str(cfg["window"]).lower(),
                bool(cfg["center"]),
            )
            expected = (1022, 320, 1022, "hann", True)
            if actual != expected:
                raise ValueError(f"strict MeanSE preset requires {expected}, got {actual}")

    def _check_spectrum(self, spec: torch.Tensor, name: str) -> torch.Tensor:
        value = _as_complex(spec)
        if value.shape[1] != self.expected_freq_bins:
            raise ValueError(
                f"{name} has {value.shape[1]} frequency bins, expected "
                f"{self.expected_freq_bins} for n_fft={self.n_fft}"
            )
        if value.shape[0] == 0 or value.shape[2] == 0:
            raise ValueError(f"{name} requires non-empty batch and time dimensions")
        return value

    def _transform(self, spec: torch.Tensor) -> torch.Tensor:
        return spectral_transform(
            spec,
            factor=self.spec_factor,
            exponent=self.spec_abs_exponent,
        )

    def _inverse_transform(self, spec: torch.Tensor) -> torch.Tensor:
        return inverse_spectral_transform(
            spec,
            factor=self.spec_factor,
            exponent=self.spec_abs_exponent,
        )

    def _frame_mask(
        self, lengths: torch.Tensor | None, spec: torch.Tensor
    ) -> torch.Tensor | None:
        if lengths is None:
            return None
        if lengths.ndim != 1 or lengths.shape[0] != spec.shape[0]:
            raise ValueError(
                f"lengths must have shape ({spec.shape[0]},), got {tuple(lengths.shape)}"
            )
        if torch.any(lengths <= 0):
            raise ValueError("lengths must contain positive waveform lengths")
        valid_frames = torch.div(
            lengths.to(device=spec.device, dtype=torch.long),
            self.hop_length,
            rounding_mode="floor",
        ) + 1
        if torch.any(valid_frames > spec.shape[-1]):
            raise ValueError("lengths imply more STFT frames than the supplied spectrum")
        frame_ids = torch.arange(spec.shape[-1], device=spec.device)
        return (frame_ids[None, None, :] < valid_frames[:, None, None]).to(
            spec.real.dtype
        )

    @staticmethod
    def _channels(x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return torch.stack((x.real, x.imag, condition.real, condition.imag), dim=1)

    def velocity(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        output = -self.backbone(self._channels(x, condition), r, t)
        return torch.complex(output[:, 0], output[:, 1])

    def sample_r_t(
        self,
        batch_size: int,
        device: torch.device,
        *,
        force_flow: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """按官方 batch-level mix-up 采样 ``r <= t``。"""
        samples = torch.rand(batch_size, 2, device=device)
        t = samples.amax(dim=1)
        r = samples.amin(dim=1)
        train_flow = (
            bool(torch.rand((), device=device) < self.flow_ratio)
            if force_flow is None
            else bool(force_flow)
        )
        if train_flow:
            r = t
        else:
            t = r + (t - r) * self.max_interval
        scale = self.t_max - self.t_eps
        return self.t_eps + scale * r, self.t_eps + scale * t, train_flow

    def flow_path(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """构造 ``x_t=(1-t)x_0+t e``，其中 ``e~N(y,sigma^2 I)``。"""
        shape = (t.shape[0],) + (1,) * (clean.ndim - 1)
        t_view = t.view(shape)
        prior = noisy + self.sigma * noise
        x_t = (1.0 - t_view) * clean + t_view * prior
        return x_t, prior - clean, prior

    def _loss(
        self,
        error: torch.Tensor,
        frame_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if frame_mask is not None:
            error = error * frame_mask
        if self.loss_type == "mse":
            losses = error.abs().square()
        elif self.loss_type == "mae":
            losses = error.abs()
        else:
            delta_sq = error.abs().square().reshape(error.shape[0], -1).mean(dim=1)
            weights = (delta_sq + self.adaptive_l2_c).detach().reciprocal()
            return 0.5 * (weights * delta_sq).mean()
        return 0.5 * losses.reshape(error.shape[0], -1).sum(dim=1).mean()

    def compute_objective(
        self,
        noisy_spec: torch.Tensor,
        clean_spec: torch.Tensor,
        *,
        lengths: torch.Tensor | None = None,
        r: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        force_flow: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        clean = self._transform(self._check_spectrum(clean_spec, "clean_spec"))
        noisy = self._transform(self._check_spectrum(noisy_spec, "noisy_spec"))
        if clean.shape != noisy.shape:
            raise ValueError(
                f"clean/noisy spectrum shapes differ: {clean.shape} vs {noisy.shape}"
            )
        frame_mask = self._frame_mask(lengths, clean)
        if frame_mask is not None:
            clean = clean * frame_mask
            noisy = noisy * frame_mask
        batch = clean.shape[0]
        train_flow = False
        if r is None or t is None:
            if r is not None or t is not None:
                raise ValueError("r and t must be supplied together")
            r, t, train_flow = self.sample_r_t(
                batch, clean.device, force_flow=force_flow
            )
        else:
            if r.shape != (batch,) or t.shape != (batch,):
                raise ValueError(f"r and t must have shape ({batch},)")
            if r.device != clean.device or t.device != clean.device:
                raise ValueError("r and t must be on the spectrum device")
            if torch.any(r < self.t_eps) or torch.any(t > self.t_max) or torch.any(r > t):
                raise ValueError("r and t must satisfy t_eps <= r <= t <= t_max")
            train_flow = bool(torch.equal(r, t))
        if noise is None:
            noise = torch.randn_like(clean)
        elif noise.shape != clean.shape or noise.device != clean.device:
            raise ValueError("noise must match the transformed clean spectrum")
        if frame_mask is not None:
            noise = noise * frame_mask

        x_t, conditional_velocity, _ = self.flow_path(clean, noisy, t, noise)
        # CPU flash-attention 尚不支持 forward AD；官方实现也提供 autograd JVP。
        prediction, directional_derivative = torch.autograd.functional.jvp(
            lambda x_value, r_value, t_value: self.velocity(
                x_value, r_value, t_value, noisy
            ),
            (x_t, r, t),
            (
                conditional_velocity,
                torch.zeros_like(r),
                torch.ones_like(t),
            ),
            create_graph=True,
        )
        interval = (t - r).view(batch, *((1,) * (clean.ndim - 1)))
        target = (conditional_velocity - interval * directional_derivative).detach()
        loss = self._loss(prediction - target, frame_mask)
        diagnostics = {
            "loss_meanflow": loss.detach(),
            "train_flow": loss.new_tensor(float(train_flow)),
            "mean_interval": (t - r).mean().detach(),
        }
        return loss, diagnostics

    @torch.no_grad()
    def sample(
        self,
        noisy_spec: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        steps: int | None = None,
    ) -> torch.Tensor:
        """使用均匀 MeanFlow 区间执行 ``steps`` 次网络求值。"""
        num_steps = self.sampling_steps if steps is None else int(steps)
        if num_steps < 1:
            raise ValueError("steps must be >= 1")
        noisy = self._transform(self._check_spectrum(noisy_spec, "noisy_spec"))
        frame_mask = self._frame_mask(lengths, noisy)
        if frame_mask is not None:
            noisy = noisy * frame_mask
        generator = torch.Generator(device=noisy.device)
        generator.manual_seed(self.inference_seed)
        prior_noise = torch.randn(
            noisy.shape,
            dtype=noisy.dtype,
            device=noisy.device,
            generator=generator,
        )
        if frame_mask is not None:
            prior_noise = prior_noise * frame_mask
        state = noisy + self.sigma * prior_noise
        batch = noisy.shape[0]
        boundaries = torch.linspace(
            self.t_max,
            self.t_eps,
            num_steps + 1,
            device=noisy.device,
            dtype=noisy.real.dtype,
        )
        for index in range(num_steps):
            t = boundaries[index].expand(batch)
            r = boundaries[index + 1].expand(batch)
            shape = (batch,) + (1,) * (state.ndim - 1)
            state = state - (t - r).view(shape) * self.velocity(
                state, r, t, noisy
            )
        if frame_mask is not None:
            state = state * frame_mask
        return _as_real(self._inverse_transform(state))

    def forward(
        self,
        noisy_spec: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        fs: int | None = None,
        clean_spec: torch.Tensor | None = None,
        progress: float = 1.0,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del fs, progress
        if clean_spec is None:
            return self.sample(noisy_spec, lengths)
        return self.compute_objective(noisy_spec, clean_spec, lengths=lengths)
