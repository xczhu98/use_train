"""ZipEnhancer scaling utilities (simplified from Zipformer2 scaling).

This module provides parameter-scaled initializations, custom normalization,
activation functions, and upsampling/downsampling utilities required by the
Zipformer2 dual-path encoder.

All parameter shapes and initialization scales are kept strictly aligned with
the original Zipformer2 implementation to ensure 2.04 M parameter count.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# ScheduledFloat: training-time schedule for dropout / skip rates
# ---------------------------------------------------------------------------

class PiecewiseLinear:
    """Piecewise-linear schedule used by ScheduledFloat."""

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], PiecewiseLinear):
            self.pairs = list(args[0].pairs)
        else:
            self.pairs = [(float(x), float(y)) for x, y in args]
        for i in range(len(self.pairs) - 1):
            assert self.pairs[i + 1][0] > self.pairs[i][0]

    def __call__(self, x):
        if x <= self.pairs[0][0]:
            return self.pairs[0][1]
        elif x >= self.pairs[-1][0]:
            return self.pairs[-1][1]
        else:
            cur_x, cur_y = self.pairs[0]
            for i in range(1, len(self.pairs)):
                next_x, next_y = self.pairs[i]
                if cur_x <= x <= next_x:
                    return cur_y + (next_y - cur_y) * (x - cur_x) / (next_x - cur_x)
                cur_x, cur_y = next_x, next_y
        assert False

    def __mul__(self, alpha):
        return PiecewiseLinear(*[(x, y * alpha) for x, y in self.pairs])

    def __add__(self, x):
        if isinstance(x, (float, int)):
            return PiecewiseLinear(*[(p[0], p[1] + x) for p in self.pairs])
        s, x = self.get_common_basis(x)
        return PiecewiseLinear(*[(sp[0], sp[1] + xp[1]) for sp, xp in zip(s.pairs, x.pairs)])

    def max(self, x):
        if isinstance(x, (float, int)):
            x = PiecewiseLinear((0, x))
        s, x = self.get_common_basis(x, include_crossings=True)
        return PiecewiseLinear(*[(sp[0], max(sp[1], xp[1])) for sp, xp in zip(s.pairs, x.pairs)])

    def min(self, x):
        if isinstance(x, (float, int)):
            x = PiecewiseLinear((0, x))
        s, x = self.get_common_basis(x, include_crossings=True)
        return PiecewiseLinear(*[(sp[0], min(sp[1], xp[1])) for sp, xp in zip(s.pairs, x.pairs)])

    def __eq__(self, other):
        return self.pairs == other.pairs

    def get_common_basis(self, p: 'PiecewiseLinear', include_crossings: bool = False):
        """Return equivalent piecewise-linear functions with the same x values."""
        assert isinstance(p, PiecewiseLinear), type(p)
        x_vals = sorted(set([x for x, _ in self.pairs] + [x for x, _ in p.pairs]))
        y_vals1 = [self(x) for x in x_vals]
        y_vals2 = [p(x) for x in x_vals]
        if include_crossings:
            extra_x_vals = []
            for i in range(len(x_vals) - 1):
                _compare_results1 = (y_vals1[i] > y_vals2[i])
                _compare_results2 = (y_vals1[i + 1] > y_vals2[i + 1])
                if _compare_results1 != _compare_results2:
                    diff_cur = abs(y_vals1[i] - y_vals2[i])
                    diff_next = abs(y_vals1[i + 1] - y_vals2[i + 1])
                    pos = diff_cur / (diff_cur + diff_next)
                    extra_x_val = x_vals[i] + pos * (x_vals[i + 1] - x_vals[i])
                    extra_x_vals.append(extra_x_val)
            if len(extra_x_vals) > 0:
                x_vals = sorted(set(x_vals + extra_x_vals))
            y_vals1 = [self(x) for x in x_vals]
            y_vals2 = [p(x) for x in x_vals]
        return (
            PiecewiseLinear(*zip(x_vals, y_vals1)),
            PiecewiseLinear(*zip(x_vals, y_vals2)),
        )


class ScheduledFloat(nn.Module):
    """Floating point schedule controlled by training batch count.

    In use_train the ``batch_count`` is never set, so this always returns
    ``default`` (typically 0.0), which disables layer-drop and other
    training-specific stochastic behaviours at inference time.
    """

    def __init__(self, *args, default: float = 0.0):
        super().__init__()
        self.batch_count: Optional[int] = None
        self.default = default
        self.schedule = PiecewiseLinear(*args)

    def __float__(self):
        if self.batch_count is None or not self.training:
            return float(self.default)
        return self.schedule(self.batch_count)

    def __repr__(self):
        return f"ScheduledFloat(default={self.default}, schedule={self.schedule.pairs})"

    def __add__(self, x):
        if isinstance(x, (float, int)):
            return ScheduledFloat(self.schedule + x, default=self.default)
        else:
            return ScheduledFloat(
                self.schedule + x.schedule, default=self.default + x.default)

    def max(self, x):
        if isinstance(x, (float, int)):
            return ScheduledFloat(self.schedule.max(x), default=self.default)
        else:
            return ScheduledFloat(
                self.schedule.max(x.schedule),
                default=max(self.default, x.default))


FloatLike = Union[float, ScheduledFloat]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def logaddexp_onnx(x: Tensor, y: Tensor) -> Tensor:
    max_value = torch.max(x, y)
    diff = torch.abs(x - y)
    return max_value + torch.log1p(torch.exp(-diff))


def logaddexp(x: Tensor, y: Tensor) -> Tensor:
    if torch.jit.is_scripting():
        return torch.logaddexp(x, y)
    elif getattr(torch, "onnx", None) is not None and torch.onnx.is_in_onnx_export():
        return logaddexp_onnx(x, y)
    else:
        return torch.logaddexp(x, y)


# ---------------------------------------------------------------------------
# Training auxiliary functions: parameter limiting, loss penalties
# ---------------------------------------------------------------------------

class WithLoss(torch.autograd.Function):
    """Returns x unchanged, but adds y.sum() to the loss in backward."""

    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor, name: str):
        ctx.y_shape = y.shape
        return x

    @staticmethod
    def backward(ctx, ans_grad: Tensor):
        return (
            ans_grad,
            torch.ones(
                ctx.y_shape, dtype=ans_grad.dtype, device=ans_grad.device),
            None,
        )


def with_loss(x, y, name):
    """Returns x but adds y.sum() to the loss function."""
    return WithLoss.apply(x, y, name)


class LimitParamValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, min: float, max: float):
        ctx.save_for_backward(x)
        assert max >= min
        ctx.min = min
        ctx.max = max
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        x_grad = x_grad * torch.where(
            torch.logical_and(x_grad > 0, x < ctx.min), -1.0, 1.0)
        x_grad *= torch.where(
            torch.logical_and(x_grad < 0, x > ctx.max), -1.0, 1.0)
        return x_grad, None, None


def limit_param_value(x: Tensor,
                      min: float,
                      max: float,
                      prob: float = 0.6,
                      training: bool = True):
    """Limit parameter value range by modifying gradients."""
    if training and torch.rand(1).item() < prob:
        return LimitParamValue.apply(x, min, max)
    else:
        return x


def penalize_abs_values_gt(x: Tensor,
                           limit: float,
                           penalty: float,
                           name: str = None) -> Tensor:
    """Returns x unmodified, but penalizes abs values > limit in backprop."""
    x_sign = x.sign()
    over_limit = (x.abs() - limit) > 0
    aux_loss = penalty * ((x_sign * over_limit).to(torch.int8) * x)
    return with_loss(x, aux_loss, name)


# ---------------------------------------------------------------------------
# Identity module (used extensively in original code for diagnostics)
# ---------------------------------------------------------------------------

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


# ---------------------------------------------------------------------------
# Custom activation functions: SwooshL / SwooshR
# ---------------------------------------------------------------------------

class SwooshLFunction(torch.autograd.Function):
    """swoosh_l(x) = log(1 + exp(x - 4)) - 0.08 * x - 0.035"""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        requires_grad = x.requires_grad
        original_dtype = x.dtype
        if original_dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        coeff = -0.08
        with torch.amp.autocast("cuda", enabled=False):
            with torch.enable_grad():
                x = x.detach()
                x.requires_grad = True
                y = torch.logaddexp(zero, x - 4.0) + coeff * x - 0.035
                if not requires_grad:
                    if original_dtype in (torch.float16, torch.bfloat16):
                        y = y.to(original_dtype)
                    return y
                y.backward(gradient=torch.ones_like(y))
                grad = x.grad
                floor = coeff
                ceil = 1.0 + coeff + 0.005
                _diff = (grad - floor) * (255.0 / (ceil - floor))
                d_scaled = _diff + torch.rand_like(grad)
                d_int = d_scaled.to(torch.uint8)
                ctx.save_for_backward(d_int)
                if original_dtype in (torch.float16, torch.bfloat16):
                    y = y.to(original_dtype)
                return y

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        (d,) = ctx.saved_tensors
        coeff = -0.08
        floor = coeff
        ceil = 1.0 + coeff + 0.005
        d = d * ((ceil - floor) / 255.0) + floor
        return y_grad * d


class SwooshL(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
            return logaddexp(zero, x - 4.0) - 0.08 * x - 0.035
        return SwooshLFunction.apply(x)


def SwooshLForward(x: Tensor) -> Tensor:
    """Forward-only SwooshL (used by ActivationDropoutAndLinear)."""
    x_offset = x - 4.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)
    return log_sum - 0.08 * x - 0.035


class SwooshRFunction(torch.autograd.Function):
    """swoosh_r(x) = log(1 + exp(x - 1)) - 0.08 * x - 0.313261687"""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        requires_grad = x.requires_grad
        original_dtype = x.dtype
        if original_dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        with torch.amp.autocast("cuda", enabled=False):
            with torch.enable_grad():
                x = x.detach()
                x.requires_grad = True
                y = torch.logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687
                if not requires_grad:
                    if original_dtype in (torch.float16, torch.bfloat16):
                        y = y.to(original_dtype)
                    return y
                y.backward(gradient=torch.ones_like(y))
                grad = x.grad
                floor = -0.08
                ceil = 0.925
                _diff = (grad - floor) * (255.0 / (ceil - floor))
                d_scaled = _diff + torch.rand_like(grad)
                d_int = d_scaled.to(torch.uint8)
                ctx.save_for_backward(d_int)
                if original_dtype in (torch.float16, torch.bfloat16):
                    y = y.to(original_dtype)
                return y

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        (d,) = ctx.saved_tensors
        floor = -0.08
        ceil = 0.925
        d = d * ((ceil - floor) / 255.0) + floor
        return y_grad * d


class SwooshR(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
            return logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687
        return SwooshRFunction.apply(x)


def SwooshRForward(x: Tensor) -> Tensor:
    """Forward-only SwooshR."""
    x_offset = x - 1.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)
    return log_sum - 0.08 * x - 0.313261687


# ---------------------------------------------------------------------------
# Dropout2: supports scheduled dropout rate
# ---------------------------------------------------------------------------

class Dropout2(nn.Module):
    def __init__(self, p: FloatLike):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.dropout(x, p=float(self.p), training=self.training)


# ---------------------------------------------------------------------------
# ScaledLinear: nn.Linear with controlled initial scale
# ---------------------------------------------------------------------------

def ScaledLinear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    initial_scale: float = 1.0,
) -> nn.Linear:
    """Create nn.Linear with weights scaled by ``initial_scale``."""
    ans = nn.Linear(in_features, out_features, bias=bias)
    with torch.no_grad():
        ans.weight[:] *= initial_scale
        if ans.bias is not None:
            nn.init.uniform_(ans.bias, -0.1 * initial_scale, 0.1 * initial_scale)
    return ans


# ---------------------------------------------------------------------------
# BiasNorm: LayerNorm replacement with learnable bias and scalar scale
# ---------------------------------------------------------------------------

class BiasNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        bias: Tensor,
        log_scale: Tensor,
        channel_dim: int,
        store_output_for_backprop: bool,
    ) -> Tensor:
        if channel_dim < 0:
            channel_dim = channel_dim + x.ndim
        ctx.store_output_for_backprop = store_output_for_backprop
        ctx.channel_dim = channel_dim
        for _ in range(channel_dim + 1, x.ndim):
            bias = bias.unsqueeze(-1)
        _x_bias_square = torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True)
        scales = (_x_bias_square ** -0.5) * log_scale.exp()
        ans = x * scales
        ctx.save_for_backward(
            ans.detach() if store_output_for_backprop else x,
            scales.detach(),
            bias.detach(),
            log_scale.detach(),
        )
        return ans

    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tensor:
        ans_or_x, scales, bias, log_scale = ctx.saved_tensors
        if ctx.store_output_for_backprop:
            x = ans_or_x / scales
        else:
            x = ans_or_x
        x = x.detach()
        x.requires_grad = True
        bias.requires_grad = True
        log_scale.requires_grad = True
        with torch.enable_grad():
            _x_bias_square = torch.mean(
                (x - bias) ** 2, dim=ctx.channel_dim, keepdim=True
            )
            scales = (_x_bias_square ** -0.5) * log_scale.exp()
            ans = x * scales
            ans.backward(gradient=ans_grad)
        return x.grad, bias.grad.flatten(), log_scale.grad, None, None


class BiasNorm(nn.Module):
    """BiasNorm replaces LayerNorm to avoid fixed large channel values.

    Formulation: ``BiasNorm(x) = x / RMS[x - b] * exp(gamma)``
    where ``b`` is learnable per-channel bias and ``gamma`` is a scalar.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,
        log_scale: float = 1.0,
        log_scale_min: float = -1.5,
        log_scale_max: float = 1.5,
        store_output_for_backprop: bool = False,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.log_scale = nn.Parameter(torch.tensor(log_scale))
        self.bias = nn.Parameter(torch.empty(num_channels).normal_(mean=0, std=1e-4))
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max
        self.store_output_for_backprop = store_output_for_backprop

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.channel_dim] == self.num_channels
        # Clamp log_scale to [min, max] for both JIT and training paths.
        log_scale = self.log_scale.clamp(
            min=float(self.log_scale_min), max=float(self.log_scale_max)
        )
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            channel_dim = self.channel_dim
            if channel_dim < 0:
                channel_dim += x.ndim
            bias = self.bias
            for _ in range(channel_dim + 1, x.ndim):
                bias = bias.unsqueeze(-1)
            _x_bias_square = torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True)
            scales = (_x_bias_square ** -0.5) * log_scale.exp()
            return x * scales
        return BiasNormFunction.apply(
            x, self.bias, log_scale, self.channel_dim, self.store_output_for_backprop
        )


# ---------------------------------------------------------------------------
# ActivationDropoutAndLinear: fused activation + dropout + linear
# ---------------------------------------------------------------------------

class ActivationDropoutAndLinear(nn.Module):
    """Fused Swoosh activation, dropout, and linear projection.

    Parameter count is exactly ``in_channels * out_channels + out_channels``
    (same as nn.Linear), because we steal the weight/bias from a temporary
    ScaledLinear module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        activation: str = "SwooshL",
        dropout_p: FloatLike = 0.0,
        dropout_shared_dim: Optional[int] = -1,
        initial_scale: float = 1.0,
    ):
        super().__init__()
        linear_module = ScaledLinear(
            in_channels, out_channels, bias=bias, initial_scale=initial_scale
        )
        self.weight = linear_module.weight
        self.register_parameter("bias", linear_module.bias)
        self.activation = activation
        self.dropout_p = dropout_p
        self.dropout_shared_dim = dropout_shared_dim

    def forward(self, x: Tensor) -> Tensor:
        if self.activation == "SwooshL":
            x = SwooshLForward(x)
        elif self.activation == "SwooshR":
            x = SwooshRForward(x)
        else:
            raise ValueError(f"Unknown activation {self.activation}")
        return nn.functional.linear(x, self.weight, self.bias)


# ---------------------------------------------------------------------------
# Depthwise 1-D convolution (causal / non-causal)
# ---------------------------------------------------------------------------

class ChunkCausalDepthwiseConv1d(nn.Module):
    """Depthwise 1-D conv with chunk-causal support.

    Not used in non-causal ZipEnhancer, but kept for parameter alignment.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        initial_scale: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        half_kernel_size = (kernel_size + 1) // 2
        self.causal_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=half_kernel_size,
            padding=0,
            bias=True,
        )
        self.chunkwise_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )
        self.chunkwise_conv_scale = nn.Parameter(
            torch.zeros(2, channels, kernel_size)
        )
        self.kernel_size = kernel_size
        with torch.no_grad():
            self.causal_conv.weight[:] *= initial_scale
            self.chunkwise_conv.weight[:] *= initial_scale
            if bias:
                nn.init.uniform_(
                    self.causal_conv.bias, -0.1 * initial_scale, 0.1 * initial_scale
                )

    def forward(self, x: Tensor, chunk_size: int = -1) -> Tensor:
        batch_size, num_channels, seq_len = x.shape
        left_pad = self.kernel_size // 2
        if chunk_size < 0 or chunk_size > seq_len:
            chunk_size = seq_len
        right_pad = -seq_len % chunk_size
        x = nn.functional.pad(x, (left_pad, right_pad))
        x_causal = self.causal_conv(x[..., : left_pad + seq_len])
        assert x_causal.shape == (batch_size, num_channels, seq_len)
        x_chunk = x[..., left_pad:]
        num_chunks = x_chunk.shape[2] // chunk_size
        x_chunk = x_chunk.reshape(batch_size, num_channels, num_chunks, chunk_size)
        x_chunk = x_chunk.permute(0, 2, 1, 3).reshape(
            batch_size * num_chunks, num_channels, chunk_size
        )
        x_chunk = self.chunkwise_conv(x_chunk)
        chunk_scale = self._get_chunk_scale(chunk_size)
        x_chunk = x_chunk * chunk_scale
        x_chunk = x_chunk.reshape(batch_size, num_chunks, num_channels, chunk_size)
        x_chunk = x_chunk.permute(0, 2, 1, 3)
        x_chunk = x_chunk.reshape(batch_size, num_channels, num_chunks * chunk_size)[
            ..., :seq_len
        ]
        return x_chunk + x_causal

    def _get_chunk_scale(self, chunk_size: int) -> Tensor:
        left_edge = self.chunkwise_conv_scale[0]
        right_edge = self.chunkwise_conv_scale[1]
        if chunk_size < self.kernel_size:
            left_edge = left_edge[:, :chunk_size]
            right_edge = right_edge[:, -chunk_size:]
        else:
            t = chunk_size - self.kernel_size
            channels = left_edge.shape[0]
            pad = torch.zeros(
                channels, t, device=left_edge.device, dtype=left_edge.dtype
            )
            left_edge = torch.cat((left_edge, pad), dim=-1)
            right_edge = torch.cat((pad, right_edge), dim=-1)
        return 1.0 + (left_edge + right_edge)


# ---------------------------------------------------------------------------
# Simple downsample / upsample (no learnable parameters except downsample bias)
# ---------------------------------------------------------------------------

class SimpleDownsample(nn.Module):
    """Downsample by weighted averaging with learnable softmax weights."""

    def __init__(self, channels: int, downsample: int, dropout: FloatLike):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(downsample))
        self.dropout = dropout
        self.downsample = downsample

    def forward(self, src: Tensor) -> Tensor:
        """Args: src (seq_len, batch, channels). Returns (seq_len//ds, batch, channels)."""
        seq_len, batch_size, in_channels = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds
        pad = d_seq_len * ds - seq_len
        if pad > 0:
            src_extra = src[src.shape[0] - 1 :].expand(pad, src.shape[1], src.shape[2])
            src = torch.cat((src, src_extra), dim=0)
        src = src.reshape(d_seq_len, ds, batch_size, in_channels)
        weights = self.bias.softmax(dim=0).view(1, ds, 1, 1)
        return (src * weights).sum(dim=1)


class SimpleUpsample(nn.Module):
    """Upsample by simple repetition."""

    def __init__(self, num_channels: int, upsample: int):
        super().__init__()
        self.upsample = upsample

    def forward(self, src: Tensor) -> Tensor:
        seq_len, batch_size, num_channels = src.shape
        src = src.unsqueeze(1).expand(seq_len, self.upsample, batch_size, num_channels)
        return src.reshape(seq_len * self.upsample, batch_size, num_channels)


# ---------------------------------------------------------------------------
# BypassModule: learnable residual scale
# ---------------------------------------------------------------------------

class BypassModule(nn.Module):
    """Learnable bypass: ``out = x + (y - x) * c`` where ``c`` is per-channel."""

    def __init__(
        self,
        embed_dim: int,
        skip_rate: FloatLike = 0.0,
        straight_through_rate: FloatLike = 0.0,
        scale_min: FloatLike = ScheduledFloat((0.0, 0.9), (20000.0, 0.2), default=0),
        scale_max: FloatLike = 1.0,
    ):
        super().__init__()
        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))
        self.skip_rate = skip_rate
        self.straight_through_rate = straight_through_rate
        self.scale_min = scale_min
        self.scale_max = scale_max

    def _get_bypass_scale(self, batch_size: int) -> Tensor:
        if not self.training:
            return self.bypass_scale
        ans = self.bypass_scale.clamp(
            min=float(self.scale_min), max=float(self.scale_max)
        )
        skip_rate = float(self.skip_rate)
        if skip_rate != 0.0:
            mask = (
                torch.rand((batch_size, 1), device=ans.device) > skip_rate
            ).to(ans.dtype)
            ans = ans * mask
        straight_through_rate = float(self.straight_through_rate)
        if straight_through_rate != 0.0:
            _rand = torch.rand((batch_size, 1), device=ans.device)
            mask = _rand < straight_through_rate
            ans = torch.maximum(ans, mask.to(ans.dtype))
        return ans

    def forward(self, src_orig: Tensor, src: Tensor) -> Tensor:
        # src_orig, src: (seq_len, batch_size, num_channels)
        bypass_scale = self._get_bypass_scale(src.shape[-2])
        return src_orig + (src - src_orig) * bypass_scale


# ---------------------------------------------------------------------------
# Channel conversion helper
# ---------------------------------------------------------------------------

def convert_num_channels(x: Tensor, num_channels: int) -> Tensor:
    """Adjust channel dimension by truncation or zero-padding.

    Expects ``x`` to be at least 2-D with channels at dim 1.
    """
    assert x.dim() >= 2, f"Expected x.dim() >= 2, got {x.dim()}"
    if num_channels <= x.shape[1]:
        return x[:, :num_channels, :, :]
    shape = list(x.shape)
    shape[1] = num_channels - shape[1]
    zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat((x, zeros), dim=1)
