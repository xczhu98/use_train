"""MeanFlowSE 的纯 PyTorch complex-STFT 实现。

参考固定上游 commit ``6e0fe649e9916046ca1147343fb4a2fb69f87e19``。
本文件保留其 spectral transform、Flow Matching path、MeanFlow 目标、FD-JVP
与一步 ``euler_mf`` 语义；网络改写为只依赖标准 PyTorch 层的 NCSN++ 风格
backbone。STFT/iSTFT 由框架顶层管理。

官方 checkpoint 未做实物核验，本实现不声明 checkpoint 兼容性。
第三方来源、项目适配边界与许可证见同目录可跟踪文件 ``PROVENANCE``。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.registry import MODEL_REGISTRY
from tasks.se.models.amp_utils import fp32_autocast
from tasks.se.models.base_model import BaseSEModel


def _as_complex(spec: torch.Tensor) -> torch.Tensor:
    """把框架 ``(B,F,T,2)`` 谱转换为 complex tensor。"""
    if spec.is_complex():
        if spec.ndim != 3:
            raise ValueError(f"complex spectrum must have shape (B,F,T), got {tuple(spec.shape)}")
        return spec
    if spec.ndim != 4 or spec.shape[-1] != 2:
        raise ValueError(f"spectrum must have shape (B,F,T,2), got {tuple(spec.shape)}")
    return torch.view_as_complex(spec.float().contiguous())


def _as_real(spec: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(spec.contiguous()) if spec.is_complex() else spec


def spectral_transform(
    spec: torch.Tensor,
    *,
    factor: float = 0.15,
    exponent: float = 0.5,
) -> torch.Tensor:
    """官方压缩谱变换 ``factor * |S|**exponent * exp(j angle(S))``。"""
    if factor <= 0.0:
        raise ValueError(f"factor must be positive, got {factor}")
    if exponent <= 0.0:
        raise ValueError(f"exponent must be positive, got {exponent}")
    z = _as_complex(spec)
    return factor * z.abs().pow(exponent) * torch.exp(1j * torch.angle(z))


def inverse_spectral_transform(
    spec: torch.Tensor,
    *,
    factor: float = 0.15,
    exponent: float = 0.5,
) -> torch.Tensor:
    """官方逆变换，返回与输入相同的 complex 或 real-pair 表示。"""
    was_complex = spec.is_complex()
    scaled = _as_complex(spec) / factor
    out = scaled.abs().pow(1.0 / exponent) * torch.exp(1j * torch.angle(scaled))
    return out if was_complex else _as_real(out)


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class GaussianFourierEmbedding(nn.Module):
    """固定 Gaussian Fourier 时间特征。"""

    def __init__(self, embedding_size: int, scale: float = 16.0):
        super().__init__()
        if embedding_size < 2 or embedding_size % 2:
            raise ValueError("embedding_size must be an even integer >= 2")
        self.register_buffer(
            "weight",
            torch.randn(embedding_size // 2) * scale,
            persistent=True,
        )

    def forward(self, value: torch.Tensor, *, log1p: bool = False) -> torch.Tensor:
        value = value.clamp_min(0.0)
        encoded = torch.log1p(value) if log1p else torch.log(value.clamp_min(1.0e-4))
        projected = encoded[:, None] * self.weight[None, :] * (2.0 * math.pi)
        return torch.cat((projected.sin(), projected.cos()), dim=-1)


class ResidualBlock(nn.Module):
    """带时间条件的 NCSN++ 风格残差块。"""

    def __init__(self, in_channels: int, out_channels: int, temb_channels: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels, eps=1.0e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.temb = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels, eps=1.0e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return (self.skip(x) + h) / math.sqrt(2.0)


class SpatialAttention(nn.Module):
    """NCSN++ bottleneck 的轻量空间自注意力。"""

    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        if channels % num_heads:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.norm = nn.GroupNorm(_group_count(channels), channels, eps=1.0e-6)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, freq, frames = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = self.proj(h).transpose(1, 2).reshape(batch, channels, freq, frames)
        return (x + h) / math.sqrt(2.0)


class NCSNppBackbone(nn.Module):
    """纯 PyTorch NCSN++ 风格 U-Net，支持 ``t`` 与 ``d=t-r`` 双时间条件。"""

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
        if nf < 4:
            raise ValueError(f"nf must be >= 4, got {nf}")
        if not ch_mult or any(int(mult) <= 0 for mult in ch_mult):
            raise ValueError(f"ch_mult must contain positive integers, got {ch_mult}")
        if num_res_blocks < 1:
            raise ValueError(f"num_res_blocks must be >= 1, got {num_res_blocks}")
        channels = [nf * int(mult) for mult in ch_mult]
        temb_channels = nf * 4
        self.pad_multiple = 2 ** (len(channels) - 1)
        self.time_embed = GaussianFourierEmbedding(nf, fourier_scale)
        self.delta_embed = GaussianFourierEmbedding(nf, fourier_scale)
        self.time_mlp = nn.Sequential(
            nn.Linear(nf, temb_channels),
            nn.SiLU(),
            nn.Linear(temb_channels, temb_channels),
        )
        self.delta_mlp = nn.Sequential(
            nn.Linear(nf, temb_channels),
            nn.SiLU(),
            nn.Linear(temb_channels, temb_channels),
        )
        self.input_conv = nn.Conv2d(4, channels[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_channels = channels[0]
        for level, out_channels in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(in_channels, out_channels, temb_channels, dropout))
                in_channels = out_channels
            self.down_blocks.append(blocks)
            if level < len(channels) - 1:
                self.downsamples.append(nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1))

        self.mid1 = ResidualBlock(in_channels, in_channels, temb_channels, dropout)
        self.mid_attn = SpatialAttention(in_channels, attention_heads)
        self.mid2 = ResidualBlock(in_channels, in_channels, temb_channels, dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(len(channels))):
            out_channels = channels[level]
            blocks = nn.ModuleList()
            blocks.append(
                ResidualBlock(in_channels + out_channels, out_channels, temb_channels, dropout)
            )
            in_channels = out_channels
            for _ in range(num_res_blocks - 1):
                blocks.append(ResidualBlock(in_channels, out_channels, temb_channels, dropout))
            self.up_blocks.append(blocks)
            if level > 0:
                self.upsamples.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))

        self.output_norm = nn.GroupNorm(_group_count(channels[0]), channels[0], eps=1.0e-6)
        self.output_conv = nn.Conv2d(channels[0], 2, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        freq, frames = x.shape[-2:]
        pad_f = (-freq) % self.pad_multiple
        pad_t = (-frames) % self.pad_multiple
        return F.pad(x, (0, pad_t, 0, pad_f)), (freq, frames)

    def forward(
        self,
        x: torch.Tensor,
        time_t: torch.Tensor,
        time_d: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(f"backbone input must have shape (B,4,F,T), got {tuple(x.shape)}")
        temb = self.time_mlp(self.time_embed(time_t))
        temb = temb + self.delta_mlp(self.delta_embed(time_d, log1p=True))
        h, original_shape = self._pad(x)
        h = self.input_conv(h)
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

        h = self.output_conv(F.silu(self.output_norm(h)))
        freq, frames = original_shape
        return h[:, :, :freq, :frames]


@MODEL_REGISTRY.register("MeanFlowSE")
class MeanFlowSE(BaseSEModel):
    """MeanFlowSE spectrum-in/spectrum-out 模型。

    推理：``forward(noisy_spec) -> enhanced_spec``。公共 waveform→STFT 推理边界
    根据 ``inference_peak_normalization`` 执行 noisy-peak normalization，并在
    iSTFT 后按同一尺度还原；模型内部仍只处理频谱。
    训练：``forward(noisy_spec, clean_spec=..., progress=...) -> (loss, diagnostics)``。
    """

    def __init__(
        self,
        *,
        n_fft: int = 510,
        hop_length: int = 128,
        win_length: int = 510,
        window: str = "hann",
        center: bool = True,
        strict_preset: bool = False,
        spec_factor: float = 0.15,
        spec_abs_exponent: float = 0.5,
        sigma_min: float = 0.0,
        sigma_max: float = 0.487,
        t_eps: float = 0.03,
        t_max: float = 1.0,
        mf_weight_final: float = 0.25,
        mf_warmup_frac: float = 0.5,
        mf_delta_gamma_start: float = 8.0,
        mf_delta_gamma_end: float = 1.0,
        mf_delta_warmup_frac: float = 0.7,
        mf_r_equals_t_prob: float = 0.1,
        mf_jvp_clip: float = 5.0,
        mf_jvp_eps: float = 1.0e-3,
        mf_jvp_chunk: int = 1,
        mf_skip_weight_thresh: float = 0.05,
        mf_first_order_coef: float = 0.5,
        inference_seed: int = 42,
        inference_peak_normalization: bool = True,
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
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            t_eps=t_eps,
            t_max=t_max,
            spec_factor=spec_factor,
            spec_abs_exponent=spec_abs_exponent,
            mf_weight_final=mf_weight_final,
            mf_warmup_frac=mf_warmup_frac,
            mf_delta_gamma_start=mf_delta_gamma_start,
            mf_delta_gamma_end=mf_delta_gamma_end,
            mf_delta_warmup_frac=mf_delta_warmup_frac,
            mf_r_equals_t_prob=mf_r_equals_t_prob,
            mf_jvp_clip=mf_jvp_clip,
            mf_jvp_eps=mf_jvp_eps,
            mf_jvp_chunk=mf_jvp_chunk,
            mf_skip_weight_thresh=mf_skip_weight_thresh,
            inference_seed=inference_seed,
            inference_peak_normalization=inference_peak_normalization,
        )
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.expected_freq_bins = self.n_fft // 2 + 1
        self.spec_factor = float(spec_factor)
        self.spec_abs_exponent = float(spec_abs_exponent)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.t_eps = float(t_eps)
        self.t_max = float(t_max)
        self.mf_weight_final = float(mf_weight_final)
        self.mf_warmup_frac = float(mf_warmup_frac)
        self.mf_delta_gamma_start = float(mf_delta_gamma_start)
        self.mf_delta_gamma_end = float(mf_delta_gamma_end)
        self.mf_delta_warmup_frac = float(mf_delta_warmup_frac)
        self.mf_r_equals_t_prob = float(mf_r_equals_t_prob)
        self.mf_jvp_clip = float(mf_jvp_clip)
        self.mf_jvp_eps = float(mf_jvp_eps)
        self.mf_jvp_chunk = int(mf_jvp_chunk)
        self.mf_skip_weight_thresh = float(mf_skip_weight_thresh)
        self.mf_first_order_coef = float(mf_first_order_coef)
        self.inference_seed = int(inference_seed)
        self.inference_peak_normalization = inference_peak_normalization
        self.backbone = NCSNppBackbone(
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
            raise ValueError("MeanFlowSE requires win_length == n_fft")
        if str(cfg["window"]).lower() != "hann":
            raise ValueError("MeanFlowSE supports only the official hann window")
        if not bool(cfg["center"]):
            raise ValueError("MeanFlowSE requires center=True")
        if not 0.0 <= float(cfg["sigma_min"]) <= float(cfg["sigma_max"]):
            raise ValueError("sigma_min and sigma_max must satisfy 0 <= sigma_min <= sigma_max")
        if not 0.0 < float(cfg["t_eps"]) < float(cfg["t_max"]):
            raise ValueError("t_eps and t_max must satisfy 0 < t_eps < t_max")
        if float(cfg["spec_factor"]) <= 0.0 or float(cfg["spec_abs_exponent"]) <= 0.0:
            raise ValueError("spec_factor and spec_abs_exponent must be positive")
        if not 0.0 <= float(cfg["mf_weight_final"]) <= 1.0:
            raise ValueError("mf_weight_final must be in [0, 1]")
        if float(cfg["mf_warmup_frac"]) < 0.0 or float(cfg["mf_delta_warmup_frac"]) < 0.0:
            raise ValueError("MeanFlowSE warmup fractions must be non-negative")
        if float(cfg["mf_delta_gamma_start"]) <= 0.0 or float(cfg["mf_delta_gamma_end"]) <= 0.0:
            raise ValueError("MeanFlowSE delta gamma values must be positive")
        if not 0.0 <= float(cfg["mf_r_equals_t_prob"]) <= 1.0:
            raise ValueError("mf_r_equals_t_prob must be in [0, 1]")
        if float(cfg["mf_jvp_clip"]) <= 0.0 or float(cfg["mf_jvp_eps"]) <= 0.0:
            raise ValueError("mf_jvp_clip and mf_jvp_eps must be positive")
        if int(cfg["mf_jvp_chunk"]) < 0:
            raise ValueError("mf_jvp_chunk must be non-negative")
        if float(cfg["mf_skip_weight_thresh"]) < 0.0:
            raise ValueError("mf_skip_weight_thresh must be non-negative")
        if int(cfg["inference_seed"]) < 0:
            raise ValueError("inference_seed must be non-negative")
        if not isinstance(cfg["inference_peak_normalization"], bool):
            raise TypeError("inference_peak_normalization must be bool")
        if bool(cfg["strict_preset"]):
            actual = (
                int(cfg["n_fft"]),
                int(cfg["hop_length"]),
                int(cfg["win_length"]),
                str(cfg["window"]).lower(),
                bool(cfg["center"]),
            )
            expected = (510, 128, 510, "hann", True)
            if actual != expected:
                raise ValueError(f"strict MeanFlowSE preset requires {expected}, got {actual}")

    def _check_spectrum(self, spec: torch.Tensor, name: str) -> torch.Tensor:
        z = _as_complex(spec)
        if z.shape[1] != self.expected_freq_bins:
            raise ValueError(
                f"{name} has {z.shape[1]} frequency bins, expected {self.expected_freq_bins} "
                f"for n_fft={self.n_fft}"
            )
        if z.shape[0] == 0 or z.shape[2] == 0:
            raise ValueError(f"{name} requires non-empty batch and time dimensions")
        return z

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
            raise ValueError(
                "lengths imply more centered-STFT frames than the supplied spectrum: "
                f"max={int(valid_frames.max().item())}, frames={spec.shape[-1]}"
            )
        frame_ids = torch.arange(spec.shape[-1], device=spec.device)
        return (frame_ids[None, None, :] < valid_frames[:, None, None]).to(spec.real.dtype)

    @staticmethod
    def _channels(x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return torch.stack((x.real, x.imag, condition.real, condition.imag), dim=1)

    def velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        r: torch.Tensor | None = None,
    ) -> torch.Tensor:
        d = torch.zeros_like(t) if r is None else (t - r).clamp_min(0.0)
        output = -self.backbone(self._channels(x, condition), t, d)
        return torch.complex(output[:, 0], output[:, 1])

    def flow_path(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """构造官方 Gaussian probability path 及条件速度。"""
        shape = (t.shape[0],) + (1,) * (clean.ndim - 1)
        t_view = t.view(shape)
        sigma = (1.0 - t) * self.sigma_min + t * self.sigma_max
        x_t = (1.0 - t_view) * clean + t_view * noisy + sigma.view(shape) * noise
        velocity = noisy - clean + (self.sigma_max - self.sigma_min) * noise
        return x_t, velocity

    @staticmethod
    def _field_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        error = (prediction - target).abs().square()
        if frame_mask is not None:
            error = error * frame_mask
        return 0.5 * error.reshape(prediction.shape[0], -1).sum(dim=-1).mean()

    def _mf_weight(self, progress: float) -> float:
        progress = min(max(float(progress), 0.0), 1.0)
        if self.mf_warmup_frac <= 0.0:
            return self.mf_weight_final
        return self.mf_weight_final * min(progress / self.mf_warmup_frac, 1.0)

    def _delta_gamma(self, progress: float) -> float:
        if self.mf_delta_warmup_frac <= 0.0:
            return self.mf_delta_gamma_end
        frac = min(max(float(progress), 0.0) / self.mf_delta_warmup_frac, 1.0)
        return self.mf_delta_gamma_start + frac * (
            self.mf_delta_gamma_end - self.mf_delta_gamma_start
        )

    def sample_r(self, t: torch.Tensor, progress: float) -> torch.Tensor:
        u = torch.rand_like(t).pow(self._delta_gamma(progress))
        r = t - u * (t - self.t_eps)
        force_equal = torch.rand_like(t) < self.mf_r_equals_t_prob
        return torch.where(force_equal, t, r)

    def finite_difference_jvp(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        r: torch.Tensor,
        velocity_direction: torch.Tensor,
        time_direction: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """中心有限差分计算 ``(v,1)·(∂x u, ∂t u)``，结果不反传二阶梯度。"""
        batch = x_t.shape[0]
        dims = (1,) * (x_t.ndim - 1)
        scale = (x_t.abs().mean(dim=tuple(range(1, x_t.ndim))) +
                 velocity_direction.abs().mean(dim=tuple(range(1, x_t.ndim))))
        eps = (self.mf_jvp_eps * scale.clamp_min(1.0e-6)).view(batch, *dims)
        eps_t = eps.reshape(batch)
        chunk_size = batch if self.mf_jvp_chunk <= 0 else self.mf_jvp_chunk
        chunks: list[torch.Tensor] = []
        with torch.no_grad(), fp32_autocast(x_t):
            for start in range(0, batch, chunk_size):
                stop = min(start + chunk_size, batch)
                e = eps[start:stop]
                et = eps_t[start:stop]
                positive = self.velocity(
                    x_t[start:stop] + e * velocity_direction[start:stop],
                    t[start:stop] + et * time_direction[start:stop],
                    condition[start:stop],
                    r[start:stop],
                )
                negative = self.velocity(
                    x_t[start:stop] - e * velocity_direction[start:stop],
                    t[start:stop] - et * time_direction[start:stop],
                    condition[start:stop],
                    r[start:stop],
                )
                chunks.append((positive - negative) / (2.0 * e))
        result = torch.cat(chunks, dim=0)
        if frame_mask is not None:
            result = result * frame_mask
        norm = torch.linalg.vector_norm(result.abs().reshape(batch, -1), dim=1).clamp_min(1.0e-12)
        clip = (self.mf_jvp_clip / norm).clamp(max=1.0).view(batch, *dims)
        return (result * clip).detach()

    def compute_objective(
        self,
        noisy_spec: torch.Tensor,
        clean_spec: torch.Tensor,
        *,
        lengths: torch.Tensor | None = None,
        progress: float = 1.0,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        r: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        clean = self._transform(self._check_spectrum(clean_spec, "clean_spec"))
        noisy = self._transform(self._check_spectrum(noisy_spec, "noisy_spec"))
        if clean.shape != noisy.shape:
            raise ValueError(f"clean/noisy spectrum shapes differ: {clean.shape} vs {noisy.shape}")
        batch = clean.shape[0]
        frame_mask = self._frame_mask(lengths, clean)
        if frame_mask is not None:
            clean = clean * frame_mask
            noisy = noisy * frame_mask
        if t is None:
            t = self.t_eps + torch.rand(batch, device=clean.device) * (self.t_max - self.t_eps)
        elif t.shape != (batch,) or t.device != clean.device:
            raise ValueError(f"t must have shape ({batch},) on {clean.device}, got {tuple(t.shape)} on {t.device}")
        if noise is None:
            noise = torch.randn_like(clean)
        elif noise.shape != clean.shape or noise.device != clean.device:
            raise ValueError("noise must match the transformed clean spectrum shape and device")
        if frame_mask is not None:
            noise = noise * frame_mask
        if r is not None:
            if r.shape != (batch,) or r.device != clean.device:
                raise ValueError(f"r must have shape ({batch},) on {clean.device}, got {tuple(r.shape)} on {r.device}")
            if torch.any(r < self.t_eps) or torch.any(r > t):
                raise ValueError("r must satisfy t_eps <= r <= t")
        x_t, conditional_velocity = self.flow_path(clean, noisy, t, noise)
        cfm_prediction = self.velocity(x_t, t, noisy, r=t)
        loss_cfm = self._field_loss(cfm_prediction, conditional_velocity, frame_mask)
        weight = self._mf_weight(progress)
        loss_mf = loss_cfm.new_zeros(())
        skipped = weight < self.mf_skip_weight_thresh
        if not skipped:
            if r is None:
                r = self.sample_r(t, progress)
            jvp = self.finite_difference_jvp(
                x_t,
                t,
                noisy,
                r,
                conditional_velocity,
                torch.ones_like(t),
                frame_mask,
            )
            delta = (t - r).view(batch, *((1,) * (clean.ndim - 1)))
            target = (conditional_velocity - self.mf_first_order_coef * delta * jvp).detach()
            loss_mf = self._field_loss(
                self.velocity(x_t, t, noisy, r), target, frame_mask
            )
        total = (1.0 - weight) * loss_cfm + weight * loss_mf if not skipped else loss_cfm
        diagnostics = {
            "loss_cfm": loss_cfm.detach(),
            "loss_mf": loss_mf.detach(),
            "mf_weight": total.new_tensor(weight),
            "mf_skipped": total.new_tensor(float(skipped)),
        }
        return total, diagnostics

    @torch.no_grad()
    def euler_mf(
        self, noisy_spec: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """从 ``t_max`` 到 ``t_eps`` 执行一次 MeanFlow 位移式 Euler。"""
        noisy = self._transform(self._check_spectrum(noisy_spec, "noisy_spec"))
        frame_mask = self._frame_mask(lengths, noisy)
        if frame_mask is not None:
            noisy = noisy * frame_mask
        batch = noisy.shape[0]
        t = noisy.real.new_full((batch,), self.t_max)
        r = noisy.real.new_full((batch,), self.t_eps)
        sigma_t = (1.0 - t) * self.sigma_min + t * self.sigma_max
        shape = (batch,) + (1,) * (noisy.ndim - 1)
        prior_generator = torch.Generator(device=noisy.device)
        prior_generator.manual_seed(self.inference_seed)
        prior_noise = torch.randn(
            noisy.shape,
            dtype=noisy.dtype,
            device=noisy.device,
            generator=prior_generator,
        )
        if frame_mask is not None:
            prior_noise = prior_noise * frame_mask
        x_t = noisy + sigma_t.view(shape) * prior_noise
        enhanced = x_t - (t - r).view(shape) * self.velocity(x_t, t, noisy, r)
        if frame_mask is not None:
            enhanced = enhanced * frame_mask
        return _as_real(self._inverse_transform(enhanced))

    def forward(
        self,
        noisy_spec: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        fs: int | None = None,
        clean_spec: torch.Tensor | None = None,
        progress: float = 1.0,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del fs
        if clean_spec is None:
            return self.euler_mf(noisy_spec, lengths)
        return self.compute_objective(
            noisy_spec, clean_spec, lengths=lengths, progress=progress
        )
