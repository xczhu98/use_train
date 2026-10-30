"""DeepFilterNet 底层实现：ERB/DPRNN/DeepFiltering 等模块。

本文件为 DeepFilterNet 的**算法实现层**，对外接口见 `deepfilternet.py`。
STFT/iSTFT 由 YAML stft 与 trainer 统一处理，本文件仅做 spectrum-in 处理。

文件职责：
  - ERB 滤波器组与归一化
  - Conv/GRU/DPRNN 层
  - Deep Filtering 算子
  - Encoder/Decoder 架构
  - 完整的 DeepFilterNet 骨干

架构说明：
  基线为 DeepFilterNet 统一结构：ERB 包络 + Deep Filtering 双路径。
  通过 YAML 中不同参数得到：
    - DeepFilterNet2: enc_concat=true, enc_lin_groups=8, lin_groups=8, erb_apply_above_bins=null, dprnn_num_blocks=0
    - DeepFilterNet3: enc_concat=false, enc_lin_groups=32, lin_groups=16, erb_apply_above_bins=96, dprnn_num_blocks=0
    - DPDFNet 变体:   enc_concat=true, enc_lin_groups=32, lin_groups=16, erb_apply_above_bins=null, dprnn_num_blocks=2|4|8

参考:
  - DeepFilterNet: https://github.com/Rikorose/DeepFilterNet
  - DPDFNet: https://github.com/ceva-ip/DPDFNet
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable, Final, Iterable, List, Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function

# ---------------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------------

def as_complex(x: Tensor) -> Tensor:
    if torch.is_complex(x):
        return x
    if x.shape[-1] != 2:
        raise ValueError(f"Last dim must be 2 (re+im), got {x.shape}")
    if x.stride(-1) != 1:
        x = x.contiguous()
    return torch.view_as_complex(x)


def as_real(x: Tensor) -> Tensor:
    if torch.is_complex(x):
        return torch.view_as_real(x)
    return x


def to_db(x: Tensor) -> Tensor:
    return 10 * torch.log10(x + 1e-10)


def get_magnitude(x: Tensor) -> Tensor:
    return as_complex(x).abs()


class _Angle(Function):
    """torch.angle with numerically stable gradient near zero magnitude."""
    @staticmethod
    def forward(ctx, x: Tensor):
        ctx.save_for_backward(x)
        return torch.atan2(x.imag, x.real)

    @staticmethod
    def backward(ctx, grad: Tensor):
        (x,) = ctx.saved_tensors
        grad_inv = grad / (x.real.square() + x.imag.square()).clamp_min_(1e-10)
        return torch.view_as_complex(
            torch.stack((-x.imag * grad_inv, x.real * grad_inv), dim=-1)
        )


def get_angle(x: Tensor) -> Tensor:
    return _Angle.apply(as_complex(x))


def erb_filter_banks(
    n_filters: int = 32,
    nfft: int = 512,
    fs: int = 16000,
    low_freq: int = 0,
    high_freq: Optional[int] = None,
    min_nb_freqs: int = 2,
) -> np.ndarray:
    """Build ERB (Equivalent Rectangular Bandwidth) triangular filter bank."""
    def freq2erb(f):
        return 9.265 * np.log1p(f / (24.7 * 9.265))

    def erb2freq(e):
        return 24.7 * 9.265 * (np.exp(e / 9.265) - 1)

    high_freq = high_freq or fs // 2
    if high_freq > fs // 2:
        raise ValueError(f"high_freq ({high_freq}) > Nyquist ({fs // 2})")
    if not (0 <= low_freq < high_freq):
        raise ValueError(f"low_freq ({low_freq}) must be in [0, high_freq={high_freq})")

    nyq = fs / 2
    fw = fs / nfft
    erb_lo = freq2erb(0.0)
    erb_hi = freq2erb(nyq)
    step = (erb_hi - erb_lo) / n_filters
    bins = np.zeros(n_filters + 1, dtype=np.int32)
    for i in range(n_filters + 1):
        bins[i] = int(round(erb2freq(erb_lo + i * step) / fw))
    bins[-1] = nfft // 2 + 1

    fbank = np.zeros([n_filters, nfft // 2 + 1])
    freq_over = 0
    for j in range(n_filters):
        alpha, beta = bins[j] + freq_over, bins[j + 1]
        if (beta - alpha) < min_nb_freqs:
            freq_over = min_nb_freqs - (beta - alpha)
            beta = min(beta + freq_over, nfft // 2 + 1)
        else:
            freq_over = 0
        fbank[j, alpha:beta] = 1.0

    if not (fbank.sum(axis=1) > 0).all():
        raise ValueError("Some ERB bands are empty; decrease n_filters or increase nfft")
    return np.abs(fbank)


# ---------------------------------------------------------------------------
#  Normalization layers
# ---------------------------------------------------------------------------

class ErbNorm(nn.Module):
    """Temporal recursive ERB normalization via exponential moving average."""
    def __init__(self, alpha: float, eps: float = 1e-12,
                 stateful: bool = False, dynamic_var: bool = False):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.init_vals = [-60.0, -90.0]
        self.stateful = stateful
        self.dynamic_var = dynamic_var
        self.mu = None
        self.var = None

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3, f"Expected [B,T,F], got ndim={x.ndim}"
        B, T, nf = x.shape
        if self.mu is None or not self.training or not self.stateful:
            step = (self.init_vals[1] - self.init_vals[0]) / (nf - 1)
            mu = self.init_vals[0] + torch.arange(nf, device=x.device) * step
            mu = mu[None, :].expand(B, nf)
            var = torch.zeros_like(mu) + 40 ** 2
        else:
            mu, var = self.mu, self.var

        out = []
        for t in range(T):
            mu = self.alpha * mu + (1 - self.alpha) * x[:, t]
            if self.dynamic_var:
                var = self.alpha * var + (1 - self.alpha) * ((x[:, t] - mu) ** 2)
            out.append((x[:, t] - mu) / (var.sqrt() + self.eps))
        result = torch.stack(out, dim=1)

        if self.mu is None or not self.training or not self.stateful:
            self.mu = self.var = None
        else:
            self.mu, self.var = mu, var
        return result


class SpecNorm(nn.Module):
    """Temporal recursive complex spec normalization via exponential moving average."""
    def __init__(self, alpha: float, eps: float = 1e-12, stateful: bool = False):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.stateful = stateful
        self.init_vals = [0.001, 0.0001]
        self.s = None

    def forward(self, x: Tensor) -> Tensor:
        assert not torch.is_complex(x), "Input must be float [B,T,F,2]"
        assert x.ndim == 4, f"Expected [B,T,F,2], got ndim={x.ndim}"
        if self.s is None or not self.training or not self.stateful:
            B, T, nf, _ = x.shape
            step = (self.init_vals[1] - self.init_vals[0]) / (nf - 1)
            s = self.init_vals[0] + torch.arange(nf, device=x.device) * step
            s = s[None, :].expand(B, nf)
        else:
            s = self.s

        x_abs = as_complex(x).abs()
        r_norm, i_norm = [], []
        for t in range(x_abs.shape[1]):
            s = self.alpha * s + (1 - self.alpha) * x_abs[:, t]
            denom = (s + self.eps).sqrt()
            r_norm.append(x[:, t, :, 0] / denom)
            i_norm.append(x[:, t, :, 1] / denom)
        result = torch.stack([torch.stack(r_norm, 1), torch.stack(i_norm, 1)], dim=-1)

        if self.s is None or not self.training or not self.stateful:
            self.s = None
        else:
            self.s = s
        return result


class Mask(nn.Module):
    def __init__(self, erb_inv_fb: Tensor, post_filter: bool = False, eps: float = 1e-12):
        super().__init__()
        self.erb_inv_fb: Tensor
        self.register_buffer("erb_inv_fb", erb_inv_fb)
        self.clamp_tensor = tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (1, 9)
        self.post_filter = post_filter
        self.eps = eps

    def pf(self, mask: Tensor, beta: float = 0.02) -> Tensor:
        mask_sin = mask * torch.sin(np.pi * mask / 2)
        return (1 + beta) * mask / (1 + beta * mask.div(mask_sin.clamp_min(self.eps)).pow(2))

    def forward(self, spec: Tensor, mask: Tensor,
                atten_lim: Tensor | None = None) -> Tensor:
        if not self.training and self.post_filter:
            mask = self.pf(mask)
        if atten_lim is not None:
            atten_lim = 10 ** (-atten_lim / 20)
            if self.clamp_tensor:
                mask = mask.clamp(min=atten_lim.view(-1, 1, 1, 1))
            else:
                mask = torch.stack(
                    [mask[i].clamp_min(atten_lim[i].item()) for i in range(atten_lim.shape[0])],
                    dim=0,
                )
        mask = mask.matmul(self.erb_inv_fb)
        if not spec.is_complex():
            mask = mask.unsqueeze(4)
        return spec * mask


# ---------------------------------------------------------------------------
#  Conv / Linear building blocks
# ---------------------------------------------------------------------------

class Conv2DPointWiseAsLinear(nn.Module):
    def __init__(self, input_channel, output_channel, bias=True):
        super().__init__()
        self.cnn_fc = nn.Linear(input_channel, output_channel, bias=bias)
        self.reset_parameters()

    def forward(self, x):
        B, C, T, F = x.shape
        x = einops.rearrange(x, 'b c t f -> (b t f) c')
        x = self.cnn_fc(x)
        return einops.rearrange(x, '(b t f) c -> b c t f', b=B, t=T, f=F)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.cnn_fc.weight, a=math.sqrt(5))
        if self.cnn_fc.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.cnn_fc.weight)
            if fan_in != 0:
                nn.init.uniform_(self.cnn_fc.bias, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))


class GroupedConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding='valid',
                 dilation=(1, 1), bias=True, groups=1):
        super().__init__()
        assert in_ch % groups == 0 and out_ch % groups == 0
        self.groups = groups
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch // groups, out_ch // groups, kernel_size,
                      stride=stride, padding=padding, dilation=dilation, bias=bias)
            for _ in range(groups)
        ])

    def forward(self, x):
        if self.groups > 1:
            splits = torch.chunk(x, self.groups, dim=1)
            return torch.cat([c(s) for c, s in zip(self.convs, splits)], dim=1)
        return self.convs[0](x)


class Conv2dNormAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: Union[int, Iterable[int]],
                 fstride: int = 1, dilation: int = 1, fpad: bool = True,
                 bias: bool = True, separable: bool = False,
                 norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
                 activation_layer: Callable[..., nn.Module] | None = nn.ReLU,
                 point_wise_type: str = 'cnn'):
        ks = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        fpad_ = ks[1] // 2 + dilation - 1 if fpad else 0
        pad = (0, 0, ks[0] - 1, 0)
        layers: list[nn.Module] = []
        if any(x > 0 for x in pad):
            layers.append(nn.ConstantPad2d(pad, 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        if max(ks) == 1:
            separable = False
        if groups > 1 and not (in_ch == out_ch == groups):
            layers.append(GroupedConv2D(in_ch, out_ch, kernel_size=ks, padding=(0, fpad_),
                                       stride=(1, fstride), dilation=(1, dilation),
                                       groups=groups, bias=bias))
        else:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=ks, padding=(0, fpad_),
                                    stride=(1, fstride), dilation=(1, dilation),
                                    groups=groups, bias=bias))
        if separable:
            pw = nn.Conv2d(out_ch, out_ch, 1, bias=False) if point_wise_type == 'cnn' \
                else Conv2DPointWiseAsLinear(out_ch, out_ch, bias=False)
            layers.append(pw)
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class ConvTranspose2dNormAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 fstride: int = 1, dilation: int = 1, fpad: bool = True,
                 bias: bool = True, separable: bool = False,
                 norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
                 activation_layer: Callable[..., nn.Module] | None = nn.ReLU,
                 point_wise_type: str = 'cnn'):
        ks = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        fpad_ = ks[1] // 2 if fpad else 0
        pad = (0, 0, ks[0] - 1, 0)
        layers: list[nn.Module] = []
        if any(x > 0 for x in pad):
            layers.append(nn.ConstantPad2d(pad, 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        layers.append(nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=ks,
            padding=(ks[0] - 1, fpad_ + dilation - 1),
            output_padding=(0, fpad_), stride=(1, fstride),
            dilation=(1, dilation), groups=groups, bias=bias,
        ))
        if separable:
            pw = nn.Conv2d(out_ch, out_ch, 1, bias=False) if point_wise_type == 'cnn' \
                else Conv2DPointWiseAsLinear(out_ch, out_ch, bias=False)
            layers.append(pw)
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class SubPixelConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, fstride=2,
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=True):
        super().__init__()
        assert fstride > 1
        self.fstride = fstride
        self.out_channels = out_channels
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias,
                      padding=padding, dilation=dilation, groups=groups)
            for _ in range(fstride)
        ])

    def forward(self, inputs):
        out = torch.cat([c(inputs) for c in self.convs], dim=1)
        return einops.rearrange(out, 'b (s c) t f -> b c t (f s)',
                                s=self.fstride, c=self.out_channels)


class SubPixelConv2dNormAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 fstride: int = 1, dilation: int = 1, fpad: bool = True,
                 bias: bool = True, separable: bool = False,
                 norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
                 activation_layer: Callable[..., nn.Module] | None = nn.ReLU,
                 point_wise_type: str = 'cnn'):
        ks = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        fpad_ = ks[1] // 2 if fpad else 0
        pad = (0, 0, ks[0] - 1, 0)
        layers: list[nn.Module] = []
        if any(x > 0 for x in pad):
            layers.append(nn.ConstantPad2d(pad, 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        layers.append(SubPixelConv2D(in_ch, out_ch, kernel_size=ks,
                                     padding=(0, fpad_ + dilation - 1),
                                     fstride=fstride, dilation=(1, dilation),
                                     groups=groups, bias=bias))
        if separable:
            pw = nn.Conv2d(out_ch, out_ch, 1, bias=False) if point_wise_type == 'cnn' \
                else Conv2DPointWiseAsLinear(out_ch, out_ch, bias=False)
            layers.append(pw)
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


# ---------------------------------------------------------------------------
#  Grouped Linear layers
# ---------------------------------------------------------------------------

class GroupedLinearEinsum(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, groups: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.groups = groups
        assert input_size % groups == 0 and hidden_size % groups == 0
        self.ws = input_size // groups
        self.weight = nn.Parameter(
            torch.zeros(groups, input_size // groups, hidden_size // groups))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            nn.init.uniform_(self.bias, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        x = x.view(b, t, self.groups, self.ws)
        x = torch.einsum("btgi,gih->btgh", x, self.weight)
        return x.flatten(2, 3) + self.bias

    def __repr__(self):
        return (f"{self.__class__.__name__}(input_size={self.input_size}, "
                f"hidden_size={self.hidden_size}, groups={self.groups})")


class GroupedLinear(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, groups: int = 1,
                 shuffle: bool = False):
        super().__init__()
        assert input_size % groups == 0 and hidden_size % groups == 0
        self.groups = groups
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        if groups == 1:
            shuffle = False
        self.shuffle = shuffle
        self.layers = nn.ModuleList(
            nn.Linear(self.input_size, self.hidden_size) for _ in range(groups))

    def forward(self, x: Tensor) -> Tensor:
        outs = [layer(x[..., i * self.input_size:(i + 1) * self.input_size])
                for i, layer in enumerate(self.layers)]
        output = torch.cat(outs, dim=-1)
        if self.shuffle:
            s = output.shape
            output = output.view(-1, self.hidden_size, self.groups).transpose(-1, -2).reshape(s)
        return output

    def __repr__(self):
        return (f"{self.__class__.__name__}(input_size={self.input_size}, "
                f"hidden_size={self.hidden_size}, groups={self.groups})")


# ---------------------------------------------------------------------------
#  GRU variants
# ---------------------------------------------------------------------------

class GroupedGRULayer(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]
    groups: Final[int]
    batch_first: Final[bool]
    bidirectional: Final[bool]
    num_directions: Final[int]

    def __init__(self, input_size: int, hidden_size: int, groups: int,
                 batch_first: bool = True, bias: bool = True,
                 dropout: float = 0, bidirectional: bool = False):
        super().__init__()
        assert input_size % groups == 0 and hidden_size % groups == 0
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        self.out_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.groups = groups
        self.batch_first = batch_first
        kw = dict(bias=bias, batch_first=batch_first, dropout=dropout,
                  bidirectional=bidirectional)
        self.layers = nn.ModuleList(
            nn.GRU(self.input_size, self.hidden_size, **kw) for _ in range(groups))

    def flatten_parameters(self):
        for layer in self.layers:
            layer.flatten_parameters()

    def get_h0(self, batch_size: int = 1, device=torch.device("cpu")):
        return torch.zeros(self.groups * self.num_directions, batch_size,
                           self.hidden_size, device=device)

    def forward(self, input: Tensor, h0: Tensor | None = None) -> Tuple[Tensor, Tensor]:
        if h0 is None:
            d0, d1 = input.shape[:2]
            h0 = self.get_h0(d0 if self.batch_first else d1, device=input.device)
        outputs, states = [], []
        for i, layer in enumerate(self.layers):
            o, s = layer(
                input[..., i * self.input_size:(i + 1) * self.input_size],
                h0[i * self.num_directions:(i + 1) * self.num_directions],
            )
            outputs.append(o)
            states.append(s)
        return torch.cat(outputs, dim=-1), torch.cat(states, dim=0)


class GroupedGRU(nn.Module):
    groups: Final[int]
    num_layers: Final[int]
    batch_first: Final[bool]
    hidden_size: Final[int]
    shuffle: Final[bool]
    add_outputs: Final[bool]

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 groups: int = 4, bias: bool = True, batch_first: bool = True,
                 dropout: float = 0, bidirectional: bool = False,
                 shuffle: bool = True, add_outputs: bool = False):
        super().__init__()
        kw = dict(groups=groups, bias=bias, batch_first=batch_first,
                  dropout=dropout, bidirectional=bidirectional)
        assert input_size % groups == 0 and hidden_size % groups == 0 and num_layers > 0
        self.groups = groups
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.hidden_size = hidden_size // groups
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        if groups == 1:
            shuffle = False
        self.shuffle = shuffle
        self.add_outputs = add_outputs
        self.grus = nn.ModuleList()
        self.grus.append(GroupedGRULayer(input_size, hidden_size, **kw))
        for _ in range(1, num_layers):
            self.grus.append(GroupedGRULayer(hidden_size, hidden_size, **kw))
        self.flatten_parameters()

    def flatten_parameters(self):
        for g in self.grus:
            g.flatten_parameters()

    def get_h0(self, batch_size: int, device=torch.device("cpu")) -> Tensor:
        return torch.zeros(
            self.num_layers * self.groups * self.num_directions,
            batch_size, self.hidden_size, device=device)

    def forward(self, input: Tensor, state: Tensor | None = None) -> Tuple[Tensor, Tensor]:
        d0, d1, _ = input.shape
        b = d0 if self.batch_first else d1
        if state is None:
            state = self.get_h0(b, input.device)
        output = torch.zeros(d0, d1, self.hidden_size * self.num_directions * self.groups,
                             device=input.device)
        outstates = []
        h = self.groups * self.num_directions
        for i, gru in enumerate(self.grus):
            input, s = gru(input, state[i * h:(i + 1) * h])
            outstates.append(s)
            if self.shuffle and i < self.num_layers - 1:
                input = input.view(d0, d1, -1, self.groups).transpose(2, 3).reshape(d0, d1, -1)
            if self.add_outputs:
                output += input
            else:
                output = input
        return output, torch.cat(outstates, dim=0)


def _execute_rnn_gru(
    x: Tensor, rnn_layer: nn.Module, rnn_states: Tensor | None,
    stateful: bool, training: bool,
) -> Tuple[Tensor, Tensor | None]:
    """Shared RNN step for SqueezedGRU / SqueezedGRU_S."""
    if stateful and training and rnn_states is not None:
        if x.shape[0] != rnn_states.shape[1]:
            r = torch.randint(0, rnn_states.shape[1], (x.shape[0],), device=x.device)
            rnn_states = rnn_states[:, r]
    if stateful and training:
        output, nxt = rnn_layer(x, rnn_states)
        nxt = nxt.detach()
    else:
        output, _ = rnn_layer(x)
        nxt = None
    return output, nxt


class SqueezedGRU(nn.Module):
    """旧版 SqueezedGRU：skip 接在 GRU 输出上（再经 linear_out）。用于复现原始 DeepFilterNet2。"""

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int | None = None, num_layers: int = 1,
                 linear_groups: int = 8, batch_first: bool = True,
                 gru_skip_op: Callable[..., nn.Module] | None = None,
                 linear_act_layer: Callable[..., nn.Module] = nn.Identity,
                 group_linear_layer: Callable[..., nn.Module] = GroupedLinearEinsum,
                 stateful: bool = False, group_gru: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_in = nn.Sequential(
            group_linear_layer(input_size, hidden_size, linear_groups),
            linear_act_layer(),
        )
        self.stateful = stateful
        self.states: Tensor | None = None
        gru_cls = nn.GRU if group_gru == 1 else GroupedGRU
        self.gru = gru_cls(hidden_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None
        if output_size is not None:
            self.linear_out = nn.Sequential(
                group_linear_layer(hidden_size, output_size, linear_groups),
                linear_act_layer(),
            )
        else:
            self.linear_out = nn.Identity()

    def forward(self, input: Tensor, h: Tensor | None = None) -> Tuple[Tensor, Tensor]:
        x = self.linear_in(input)
        x, self.states = _execute_rnn_gru(x, self.gru, self.states, self.stateful, self.training)
        if self.gru_skip is not None:
            x = x + self.gru_skip(input)
        x = self.linear_out(x)
        return x, self.states


class SqueezedGRU_S(nn.Module):
    """Compressed GRU: GroupedLinear -> GRU -> GroupedLinear + optional skip.

    Skip 接在 linear_out 之后（x + gru_skip(input)），与 DFN3/DPDFNet 一致。原始 DFN2（Rikorose
    deepfilternet2.py）用的是旧版 SqueezedGRU（skip 接在 GRU 输出上）；DPDFNet 作者采用 DFN3 的 _S 变体，
    本实现默认与 DPDFNet-main 一致；复现原版 DFN2 时可设 gru_variant='squeezed'。
    """
    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int | None = None, num_layers: int = 1,
                 linear_groups: int = 8, batch_first: bool = True,
                 gru_skip_op: Callable[..., nn.Module] | None = None,
                 linear_act_layer: Callable[..., nn.Module] = nn.Identity,
                 group_linear_layer: Callable[..., nn.Module] = GroupedLinearEinsum,
                 stateful: bool = False, group_gru: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_in = nn.Sequential(
            group_linear_layer(input_size, hidden_size, linear_groups),
            linear_act_layer(),
        )
        self.stateful = stateful
        self.states = None
        gru_cls = nn.GRU if group_gru == 1 else GroupedGRU
        self.gru = gru_cls(hidden_size, hidden_size, num_layers=num_layers,
                           batch_first=batch_first)
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None
        if output_size is not None:
            self.linear_out = nn.Sequential(
                group_linear_layer(hidden_size, output_size, linear_groups),
                linear_act_layer(),
            )
        else:
            self.linear_out = nn.Identity()

    def forward(self, input: Tensor, h: Tensor | None = None) -> Tuple[Tensor, Tensor]:
        x = self.linear_in(input)
        x, self.states = _execute_rnn_gru(x, self.gru, self.states, self.stateful, self.training)
        x = self.linear_out(x)
        if self.gru_skip is not None:
            x = x + self.gru_skip(input)
        return x, self.states


# ---------------------------------------------------------------------------
#  DPRNN (Dual-Path RNN)
# ---------------------------------------------------------------------------

class DPRNNBlock(nn.Module):
    """Intra (freq-dim bidirectional GRU) + inter (time-dim unidirectional GRU) with residual."""
    def __init__(self, hidden_dim: int, num_gru_layers: int = 1, stateful: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stateful = stateful
        self.intra_gru = nn.GRU(hidden_dim, hidden_dim, num_gru_layers,
                                batch_first=True, bidirectional=True)
        self.fc_intra = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln_intra = nn.LayerNorm(hidden_dim)
        self.inter_gru = nn.GRU(hidden_dim, hidden_dim, num_gru_layers, batch_first=True)
        self.inter_states: Tensor | None = None
        self.fc_inter = nn.Linear(hidden_dim, hidden_dim)
        self.ln_inter = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        B, C, T, F = inputs.shape
        x_intra = einops.rearrange(inputs, 'b c t f -> (b t) f c')
        x_intra, _ = self._run_rnn(x_intra, self.intra_gru, None)
        x_intra = self.ln_intra(self.fc_intra(x_intra))
        x_intra = einops.rearrange(x_intra, '(b t) f c -> b c t f', b=B, t=T, f=F)
        x = inputs + x_intra

        x_inter = einops.rearrange(x, 'b c t f -> (b f) t c')
        x_inter, self.inter_states = self._run_rnn(x_inter, self.inter_gru, self.inter_states)
        x_inter = self.ln_inter(self.fc_inter(x_inter))
        x_inter = einops.rearrange(x_inter, '(b f) t c -> b c t f', b=B, t=T, f=F)
        return x + x_inter

    def _run_rnn(self, x: Tensor, rnn: nn.GRU,
                 states: Tensor | None) -> Tuple[Tensor, Tensor | None]:
        if self.stateful and self.training and states is not None:
            if x.size(0) != states.size(1):
                idx = torch.randint(0, states.size(1), (x.size(0),), device=x.device)
                states = states[:, idx]
        if self.stateful and self.training:
            out, nxt = rnn(x, states)
            return out, nxt.detach()
        out, _ = rnn(x)
        return out, None


class DPRNN(nn.Module):
    def __init__(self, ch_in: int, hidden_dim: int, ch_out: int,
                 num_gru_layers: int = 1, num_blocks: int = 6, stateful: bool = True):
        super().__init__()
        self.input_proj = nn.Identity() if ch_in == hidden_dim else nn.Conv2d(ch_in, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            DPRNNBlock(hidden_dim, num_gru_layers, stateful) for _ in range(num_blocks))
        self.output_proj = nn.Identity() if hidden_dim == ch_out else nn.Conv2d(hidden_dim, ch_out, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.input_proj(inputs)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, conv_layer_fn, depth, conv_ch, point_wise_type):
        super().__init__()
        self.first_layer = conv_layer_fn
        block_fn = partial(Conv2dNormAct, in_ch=conv_ch, out_ch=conv_ch, kernel_size=(1, 3),
                           fstride=1, bias=False, separable=True,
                           point_wise_type=point_wise_type, activation_layer=nn.Identity)
        self.res_blocks = nn.ModuleList([block_fn() for _ in range(depth - 1)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.first_layer(x)
        for block in self.res_blocks:
            out = self.relu(block(out) + out)
        return out


# ---------------------------------------------------------------------------
#  Multi-frame Deep Filtering
# ---------------------------------------------------------------------------

def _as_windowed(x: Tensor, window_length: int, step: int = 1, dim: int = 1) -> Tensor:
    shape = list(x.shape)
    stride = list(x.stride())
    shape[dim] = (shape[dim] - window_length) // step + 1
    if dim > 0:
        shape.insert(dim + 1, window_length)
        stride.insert(dim + 1, stride[dim])
    elif dim == -1:
        shape.append(window_length)
        stride.append(stride[dim])
    else:
        shape.insert(dim, window_length)
        stride.insert(dim, stride[dim])
    stride[dim] = stride[dim] * step
    return x.as_strided(shape, stride)


class MultiFrameModule(nn.Module):
    num_freqs: Final[int]
    frame_size: Final[int]
    need_unfold: Final[bool]
    real: Final[bool]

    def __init__(self, num_freqs: int, frame_size: int, lookahead: int = 0,
                 real: bool = False):
        super().__init__()
        self.num_freqs = num_freqs
        self.frame_size = frame_size
        self.real = real
        if real:
            self.pad = nn.ConstantPad3d((0, 0, 0, 0, frame_size - 1 - lookahead, lookahead), 0.0)
        else:
            self.pad = nn.ConstantPad2d((0, 0, frame_size - 1 - lookahead, lookahead), 0.0)
        self.need_unfold = frame_size > 1
        self.lookahead = lookahead

    def spec_unfold_real(self, spec: Tensor):
        if self.need_unfold:
            spec = self.pad(spec).unfold(-3, self.frame_size, 1)
            return spec.permute(0, 1, 5, 2, 3, 4)
        return spec.unsqueeze(-1)

    def spec_unfold(self, spec: Tensor):
        if self.need_unfold:
            return self.pad(spec).unfold(2, self.frame_size, 1)
        return spec.unsqueeze(-1)

    @staticmethod
    def apply_coefs(spec: Tensor, coefs: Tensor) -> Tensor:
        return torch.einsum("...n,...n->...", spec, coefs)


def _df(spec: Tensor, coefs: Tensor) -> Tensor:
    return torch.einsum("...tfn,...ntf->...tf", spec, coefs)


def _df_real(spec: Tensor, coefs: Tensor) -> Tensor:
    b, c, _, t, f, _ = spec.shape
    out = torch.empty((b, c, t, f, 2), dtype=spec.dtype, device=spec.device)
    out[..., 0] = (spec[..., 0] * coefs[..., 0]).sum(dim=2) - (spec[..., 1] * coefs[..., 1]).sum(dim=2)
    out[..., 1] = (spec[..., 0] * coefs[..., 1]).sum(dim=2) + (spec[..., 1] * coefs[..., 0]).sum(dim=2)
    return out


class DF(MultiFrameModule):
    """Complex-valued Deep Filtering."""
    conj: Final[bool]

    def __init__(self, num_freqs: int, frame_size: int, lookahead: int = 0,
                 conj: bool = False):
        super().__init__(num_freqs, frame_size, lookahead)
        self.conj = conj

    def forward(self, spec: Tensor, coefs: Tensor):
        spec_u = self.spec_unfold(torch.view_as_complex(spec))
        coefs = torch.view_as_complex(coefs)
        spec_f = spec_u.narrow(-2, 0, self.num_freqs)
        coefs = coefs.view(coefs.shape[0], -1, self.frame_size, coefs.shape[2], coefs.shape[3])
        if self.conj:
            coefs = coefs.conj()
        spec_f = _df(spec_f, coefs)
        if self.training:
            spec = spec.clone()
        spec[..., :self.num_freqs, :] = torch.view_as_real(spec_f)
        return spec


class DFreal(MultiFrameModule):
    """Real-valued Deep Filtering."""
    conj: Final[bool]

    def __init__(self, num_freqs: int, frame_size: int, lookahead: int = 0,
                 conj: bool = False):
        super().__init__(num_freqs, frame_size, lookahead, real=True)
        self.conj = conj

    def forward(self, spec: Tensor, coefs: Tensor):
        spec_u = self.spec_unfold_real(spec)
        spec_f = spec_u.narrow(-2, 0, self.num_freqs)
        coefs = coefs.view([coefs.shape[0], -1, self.frame_size] + list(coefs.shape[2:]))
        if self.conj:
            coefs = coefs.conj()
        spec_f = _df_real(spec_f, coefs)
        spec[..., :self.num_freqs, :] = spec_f
        return spec


# ---------------------------------------------------------------------------
#  Encoder / Decoder / top-level DeepFilterNet
# ---------------------------------------------------------------------------

PI = 3.1415926535897932384626433


class _Add(nn.Module):
    def forward(self, a, b):
        return a + b


class _Concat(nn.Module):
    def forward(self, a, b):
        return torch.cat((a, b), dim=-1)


class Encoder(nn.Module):
    """Dual-path encoder: ERB conv path + DF conv path -> merge -> GRU -> lsnr."""
    def __init__(self, nb_erb: int, nb_df: int, conv_ch: int,
                 conv_kernel_inp: Tuple[int, int], conv_kernel: Tuple[int, int],
                 enc_concat: bool, emb_hidden_dim: int, enc_lin_groups: int,
                 emb_num_layers: int, lin_groups: int,
                 emb_gru_skip_enc: str = 'none', stateful: bool = False,
                 group_linear_type: str = 'einsum', point_wise_type: str = 'cnn',
                 group_gru: int = 1, separable_first_conv: bool = True,
                 lsnr_min: float = -15., lsnr_max: float = 35.,
                 dprnn_num_blocks: int = 0, gru_variant: str = 'squeezed_s'):
        super().__init__()
        assert nb_erb % 4 == 0
        self.erb_conv0 = Conv2dNormAct(1, conv_ch, conv_kernel_inp, bias=False,
                                       separable=separable_first_conv,
                                       point_wise_type=point_wise_type)
        _cv = partial(Conv2dNormAct, in_ch=conv_ch, out_ch=conv_ch, kernel_size=conv_kernel,
                      bias=False, separable=True, point_wise_type=point_wise_type)
        self.erb_conv1 = _cv(fstride=2)
        self.erb_conv2 = _cv(fstride=2)
        self.erb_conv3 = _cv(fstride=1)
        self.df_conv0 = Conv2dNormAct(2, conv_ch, conv_kernel_inp, bias=False,
                                      separable=separable_first_conv,
                                      point_wise_type=point_wise_type)
        self.df_conv1 = _cv(fstride=2)
        self.dprnn_erb = DPRNN(conv_ch, conv_ch, conv_ch, 1, dprnn_num_blocks,
                               stateful) if dprnn_num_blocks > 0 else nn.Identity()
        self.dprnn_df = DPRNN(conv_ch, conv_ch, conv_ch, 1, dprnn_num_blocks,
                              stateful) if dprnn_num_blocks > 0 else nn.Identity()
        self.erb_bins = nb_erb
        self.emb_in_dim = conv_ch * nb_erb // 4
        self.emb_dim = emb_hidden_dim
        self.emb_out_dim = conv_ch * nb_erb // 4
        gl = GroupedLinearEinsum if group_linear_type == 'einsum' else GroupedLinear
        self.df_fc_emb = nn.Sequential(
            gl(conv_ch * nb_df // 2, self.emb_in_dim, enc_lin_groups), nn.ReLU(inplace=True))
        if enc_concat:
            self.emb_in_dim *= 2
            self.combine = _Concat()
        else:
            self.combine = _Add()
        self.emb_n_layers = emb_num_layers
        skip_op = None
        if emb_gru_skip_enc == "identity":
            assert self.emb_in_dim == self.emb_out_dim
            skip_op = partial(nn.Identity)
        elif emb_gru_skip_enc == "groupedlinear":
            skip_op = partial(gl, input_size=self.emb_out_dim,
                              hidden_size=self.emb_out_dim, groups=lin_groups)
        elif emb_gru_skip_enc != "none":
            raise NotImplementedError()
        gru_cls = SqueezedGRU_S if gru_variant == 'squeezed_s' else SqueezedGRU
        self.emb_gru = gru_cls(
            self.emb_in_dim, self.emb_dim, output_size=self.emb_out_dim,
            num_layers=1, batch_first=True, gru_skip_op=skip_op,
            linear_groups=lin_groups, linear_act_layer=partial(nn.ReLU, inplace=True),
            group_linear_layer=gl, stateful=stateful, group_gru=group_gru)
        self.lsnr_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())
        self.lsnr_scale = lsnr_max - lsnr_min
        self.lsnr_offset = lsnr_min
        # DF 应用门控 α（0~1），供 DfAlphaLoss / DfAlphaLossComponent 与训练对齐官方 L_α
        self.df_alpha_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())

    def forward(self, feat_erb: Tensor, feat_spec: Tensor):
        e0 = self.erb_conv0(feat_erb)
        e1 = self.erb_conv1(e0)
        e2 = self.erb_conv2(e1)
        e3 = self.erb_conv3(e2)
        e3d = self.dprnn_erb(e3)
        c0 = self.df_conv0(feat_spec)
        c1 = self.dprnn_df(self.df_conv1(c0))
        cemb = self.df_fc_emb(c1.permute(0, 2, 3, 1).flatten(2))
        emb = self.combine(e3d.permute(0, 2, 3, 1).flatten(2), cemb)
        emb, _ = self.emb_gru(emb)
        lsnr = self.lsnr_fc(emb).squeeze(-1) * self.lsnr_scale + self.lsnr_offset
        df_alpha = self.df_alpha_fc(emb).squeeze(-1)
        return e0, e1, e2, e3, emb, c0, lsnr, df_alpha


class ErbDecoder(nn.Module):
    """ERB mask decoder: GRU embedding + multi-scale upsample -> Sigmoid mask."""
    def __init__(self, nb_erb: int, conv_ch: int, conv_kernel: Tuple[int, int],
                 convt_kernel: Tuple[int, int], emb_num_layers: int,
                 emb_hidden_dim: int, lin_groups: int,
                 emb_gru_skip: str = 'none', stateful: bool = False,
                 group_linear_type: str = 'einsum',
                 upsample_conv_type: str = 'transpose',
                 point_wise_type: str = 'cnn', group_gru: int = 1,
                 gru_variant: str = 'squeezed_s'):
        super().__init__()
        assert nb_erb % 8 == 0
        self.emb_in_dim = conv_ch * nb_erb // 4
        self.emb_dim = emb_hidden_dim
        self.emb_out_dim = conv_ch * nb_erb // 4
        gl = GroupedLinearEinsum if group_linear_type == 'einsum' else GroupedLinear
        skip_op = None
        if emb_gru_skip == "identity":
            assert self.emb_in_dim == self.emb_out_dim
            skip_op = partial(nn.Identity)
        elif emb_gru_skip == "groupedlinear":
            skip_op = partial(gl, input_size=self.emb_in_dim,
                              hidden_size=self.emb_out_dim, groups=lin_groups)
        elif emb_gru_skip != "none":
            raise NotImplementedError()
        gru_cls = SqueezedGRU_S if gru_variant == 'squeezed_s' else SqueezedGRU
        self.emb_gru = gru_cls(
            self.emb_in_dim, self.emb_dim, output_size=self.emb_out_dim,
            num_layers=emb_num_layers, batch_first=True, gru_skip_op=skip_op,
            linear_groups=lin_groups, linear_act_layer=partial(nn.ReLU, inplace=True),
            group_linear_layer=gl, stateful=stateful, group_gru=group_gru)
        up = ConvTranspose2dNormAct if upsample_conv_type == 'transpose' else SubPixelConv2dNormAct
        _tc = partial(up, kernel_size=convt_kernel, bias=False, separable=True,
                      point_wise_type=point_wise_type)
        _cv = partial(Conv2dNormAct, bias=False, separable=True,
                      point_wise_type=point_wise_type)
        self.conv3p = _cv(conv_ch, conv_ch, kernel_size=1)
        self.convt3 = _cv(conv_ch, conv_ch, kernel_size=conv_kernel)
        self.conv2p = _cv(conv_ch, conv_ch, kernel_size=1)
        self.convt2 = _tc(conv_ch, conv_ch, fstride=2)
        self.conv1p = _cv(conv_ch, conv_ch, kernel_size=1)
        self.convt1 = _tc(conv_ch, conv_ch, fstride=2)
        self.conv0p = _cv(conv_ch, conv_ch, kernel_size=1)
        self.conv0_out = _cv(conv_ch, 1, kernel_size=conv_kernel, activation_layer=nn.Sigmoid)

    def forward(self, emb: Tensor, e3: Tensor, e2: Tensor,
                e1: Tensor, e0: Tensor) -> Tensor:
        b, _, t, f8 = e3.shape
        emb, _ = self.emb_gru(emb)
        emb = emb.view(b, t, f8, -1).permute(0, 3, 1, 2)
        e3 = self.convt3(self.conv3p(e3) + emb)
        e2 = self.convt2(self.conv2p(e2) + e3)
        e1 = self.convt1(self.conv1p(e1) + e2)
        return self.conv0_out(self.conv0p(e0) + e1)


class DfOutputReshapeMF(nn.Module):
    """Reshape DF coeffs (B,T,F,O*2) -> (B,O,T,F,2)."""
    def __init__(self, df_order: int, df_bins: int):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins

    def forward(self, coefs: Tensor) -> Tensor:
        s = list(coefs.shape)
        s[-1] = -1
        s.append(2)
        return coefs.view(s).permute(0, 3, 1, 2, 4)


class DfDecoder(nn.Module):
    """DF coefficients decoder: GRU + skip + conv path -> DF coefficients."""
    def __init__(self, nb_erb: int, nb_df: int, conv_ch: int,
                 df_hidden_dim: int, emb_hidden_dim: int,
                 df_order: int, df_num_layers: int,
                 df_pathway_kernel_size_t: int, lin_groups: int,
                 df_gru_skip: str = 'groupedlinear', stateful: bool = False,
                 group_linear_type: str = 'einsum', point_wise_type: str = 'cnn',
                 group_gru: int = 1, gru_variant: str = 'squeezed_s'):
        super().__init__()
        self.emb_in_dim = conv_ch * nb_erb // 4
        self.emb_dim = df_hidden_dim
        self.df_order = df_order
        self.df_bins = nb_df
        self.df_out_ch = df_order * 2
        _cv = partial(Conv2dNormAct, separable=True, bias=False, point_wise_type=point_wise_type)
        self.df_convp = _cv(conv_ch, self.df_out_ch, fstride=1,
                            kernel_size=(df_pathway_kernel_size_t, 1))
        gl = GroupedLinearEinsum if group_linear_type == 'einsum' else GroupedLinear
        gru_cls = SqueezedGRU_S if gru_variant == 'squeezed_s' else SqueezedGRU
        self.df_gru = gru_cls(
            self.emb_in_dim, self.emb_dim, num_layers=df_num_layers,
            batch_first=True, gru_skip_op=None,
            linear_act_layer=partial(nn.ReLU, inplace=True),
            group_linear_layer=gl, stateful=stateful, group_gru=group_gru)
        df_gru_skip = df_gru_skip.lower()
        assert df_gru_skip in ("none", "identity", "groupedlinear")
        self.df_skip: nn.Module | None
        if df_gru_skip == "none":
            self.df_skip = None
        elif df_gru_skip == "identity":
            assert emb_hidden_dim == df_hidden_dim
            self.df_skip = nn.Identity()
        elif df_gru_skip == "groupedlinear":
            self.df_skip = gl(self.emb_in_dim, self.emb_dim, groups=lin_groups)
        else:
            raise NotImplementedError()
        out_dim = nb_df * self.df_out_ch
        self.df_out = nn.Sequential(gl(df_hidden_dim, out_dim, groups=lin_groups), nn.Tanh())

    def forward(self, emb: Tensor, c0: Tensor) -> Tensor:
        b, t, _ = emb.shape
        c, _ = self.df_gru(emb)
        if self.df_skip is not None:
            c = c + self.df_skip(emb)
        c0 = self.df_convp(c0).permute(0, 2, 3, 1)
        c = self.df_out(c)
        return c.view(b, t, self.df_bins, self.df_out_ch) + c0


class DeepFilterNet(nn.Module):
    """DeepFilterNet 统一骨干：ERB 包络 + Deep Filtering 双路径。

    配置不同参数即得 DeepFilterNet2 / DeepFilterNet3 / DPDFNet 变体（见模块 docstring）。
    双路径：
      ERB 路径：宽带掩码，覆盖全频带或仅高频（erb_apply_above_bins）
      DF 路径：低频 FIR 深度滤波（freq_df 以下）
    """
    def __init__(
        self,
        samplerate: int = 16000,
        freq_df: int = 4800,
        nb_erb: int = 32,
        min_nb_freqs: int = 1,
        erb_to_db: bool = True,
        alpha_norm: float = 0.98,
        conv_ch: int = 64,
        conv_kernel_inp: Tuple[int, int] = (3, 3),
        conv_kernel: Tuple[int, int] = (1, 3),
        convt_kernel: Tuple[int, int] = (1, 3),
        enc_gru_dim: int = 256,
        erb_dec_gru_dim: int = 256,
        df_dec_gru_dim: int = 256,
        enc_lin_groups: int = 32,
        emb_gru_skip_enc: str = 'none',
        lin_groups: int = 16,
        df_order: int = 5,
        df_pathway_kernel_size_t: int = 5,
        df_gru_skip: str = 'groupedlinear',
        df_lookahead: int = 2,
        conv_lookahead: int = 2,
        enc_concat: bool = True,
        emb_num_layers: int = 2,
        df_num_layers: int = 2,
        stateful: bool = False,
        mask_method: str = 'before_df',
        erb_dynamic_var: bool = False,
        norm_stateful: bool = False,
        upsample_conv_type: str = 'subpixel',
        group_linear_type: str = 'loop',
        point_wise_type: str = 'cnn',
        group_gru: int = 1,
        separable_first_conv: bool = True,
        lsnr_min: float = -15.,
        lsnr_max: float = 35.,
        dprnn_num_blocks: int = 0,
        erb_apply_above_bins: int | None = None,
        gru_variant: str = 'squeezed_s',
        **kwargs,
    ):
        """DeepFilterNet 统一实现。n_fft/win_length/hop_length 由 YAML stft 注入。erb_apply_above_bins=96 时为 DF3。
        gru_variant: 'squeezed_s'（DFN3/DPDFNet，skip 在 linear_out 后）| 'squeezed'（原版 DFN2，skip 在 GRU 输出上）。
        """
        super().__init__()
        n_fft = kwargs.pop("n_fft", 320)
        win_length = kwargs.pop("win_length", 0.02)
        hop_length = kwargs.pop("hop_length", 0.01)
        if kwargs:
            raise TypeError(
                f"DeepFilterNet implementation got unexpected keyword arguments: "
                f"{sorted(kwargs)}"
            )
        assert upsample_conv_type in ('transpose', 'subpixel')
        assert group_linear_type in ('einsum', 'loop')
        assert point_wise_type in ('cnn', 'linear')

        self.mask_method = mask_method
        self.run_df = True
        self.nb_erb = nb_erb
        self.erb_to_db = erb_to_db

        ef = erb_filter_banks(nfft=n_fft, low_freq=0, fs=samplerate,
                              n_filters=nb_erb, min_nb_freqs=min_nb_freqs)
        ef = torch.tensor(ef, dtype=torch.float32)
        inv_ef = ef.clone().t()
        ef = ef / ef.sum(-1, keepdims=True)
        self.register_buffer('erb_fb', ef.t())
        self.register_buffer('erb_inv_fb', inv_ef.t())

        assert nb_erb % 8 == 0
        self.n_fft = n_fft
        self.df_lookahead = df_lookahead
        self.freq_bins = n_fft // 2 + 1
        self.nb_df = int((freq_df / (samplerate // 2)) * self.freq_bins)
        self.emb_dim = conv_ch * nb_erb
        self.erb_bins = nb_erb

        if conv_lookahead > 0:
            assert conv_lookahead >= df_lookahead
            self.pad_feat = nn.ConstantPad2d((0, 0, -conv_lookahead, conv_lookahead), 0.0)
        else:
            self.pad_feat = nn.Identity()
        self.pad_spec = nn.ConstantPad3d(
            (0, 0, 0, 0, -df_lookahead, df_lookahead), 0.0) if df_lookahead > 0 else nn.Identity()

        self.enc = Encoder(
            nb_erb=nb_erb, nb_df=self.nb_df, conv_ch=conv_ch,
            conv_kernel_inp=conv_kernel_inp, conv_kernel=conv_kernel,
            enc_concat=enc_concat, emb_hidden_dim=enc_gru_dim,
            enc_lin_groups=enc_lin_groups, emb_num_layers=emb_num_layers - 1,
            lin_groups=lin_groups, emb_gru_skip_enc=emb_gru_skip_enc,
            stateful=stateful, group_linear_type=group_linear_type,
            point_wise_type=point_wise_type, group_gru=group_gru,
            separable_first_conv=separable_first_conv,
            lsnr_min=lsnr_min, lsnr_max=lsnr_max, dprnn_num_blocks=dprnn_num_blocks,
            gru_variant=gru_variant)
        self.erb_dec = ErbDecoder(
            nb_erb=nb_erb, conv_ch=conv_ch, conv_kernel=conv_kernel,
            convt_kernel=convt_kernel, emb_num_layers=emb_num_layers,
            emb_hidden_dim=erb_dec_gru_dim, lin_groups=lin_groups,
            emb_gru_skip=emb_gru_skip_enc, stateful=stateful,
            upsample_conv_type=upsample_conv_type,
            group_linear_type=group_linear_type, point_wise_type=point_wise_type,
            group_gru=group_gru, gru_variant=gru_variant)
        self.mask = Mask(self.erb_inv_fb)
        self.df_order = df_order
        self.df_op = DF(num_freqs=self.nb_df, frame_size=df_order, lookahead=df_lookahead)
        self.df_dec = DfDecoder(
            nb_erb=nb_erb, nb_df=self.nb_df, conv_ch=conv_ch,
            df_hidden_dim=df_dec_gru_dim, emb_hidden_dim=erb_dec_gru_dim,
            df_order=df_order, df_num_layers=df_num_layers,
            df_pathway_kernel_size_t=df_pathway_kernel_size_t, lin_groups=lin_groups,
            df_gru_skip=df_gru_skip, stateful=stateful,
            group_linear_type=group_linear_type, point_wise_type='cnn',
            group_gru=group_gru, gru_variant=gru_variant)
        self.df_out_transform = DfOutputReshapeMF(df_order, self.nb_df)
        self.run_erb = self.nb_df + 1 < self.freq_bins
        self.erb_apply_above_bins = erb_apply_above_bins  # None = 全频 ERB (DF2/DPDFNet)，96 = 仅高频 (DF3)
        self.erb_norm = ErbNorm(alpha=alpha_norm, dynamic_var=erb_dynamic_var,
                                stateful=norm_stateful)
        self.spec_norm = SpecNorm(alpha=alpha_norm, stateful=norm_stateful)
        self._init_gru_weight()

    def forward_from_spec(
        self, noisy_spec: Tensor, return_alpha: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """Spectrum-in, spectrum-out. noisy_spec (B, F, T, 2) -> (enhanced_spec, lsnr) 或 (+ df_alpha)。"""
        spec_btf = as_complex(noisy_spec.contiguous()).permute(0, 2, 1)
        spec, feat_erb, feat_spec = self._feature_extraction_from_spec(spec_btf)
        feat_spec = feat_spec.squeeze(1).permute(0, 3, 1, 2)
        feat_erb = self.pad_feat(feat_erb)
        feat_spec = self.pad_feat(feat_spec)
        e0, e1, e2, e3, emb, c0, lsnr, df_alpha = self.enc(feat_erb, feat_spec)
        m = self.erb_dec(emb, e3, e2, e1, e0)
        df_coefs = self.df_out_transform(self.df_dec(emb, c0))
        erb_above = self.erb_apply_above_bins
        spec_orig = spec.clone() if erb_above is not None else None
        if self.mask_method == 'before_df':
            spec = self.mask(spec, m)
            if erb_above is not None:
                spec[..., :erb_above, :] = spec_orig[..., :erb_above, :]
            spec_e = self.df_op(spec.clone(), df_coefs)
        elif self.mask_method == 'separate':
            spec_m = self.mask(spec, m)
            if erb_above is not None:
                spec_m[..., :erb_above, :] = spec_orig[..., :erb_above, :]
            spec_e = self.df_op(spec.clone(), df_coefs)
            spec_e[..., self.nb_df:, :] = spec_m[..., self.nb_df:, :]
        elif self.mask_method == 'after_df':
            spec_e = self.df_op(spec.clone(), df_coefs)
            spec_e_df_only = spec_e.clone() if erb_above is not None else None
            spec_e = self.mask(spec_e, m)
            if erb_above is not None:
                spec_e[..., :erb_above, :] = spec_e_df_only[..., :erb_above, :]
        else:
            raise ValueError(f'Unknown mask_method: {self.mask_method}')
        spec_e = torch.view_as_complex(spec_e).squeeze(1)
        out = as_real(spec_e.permute(0, 2, 1).contiguous())
        if return_alpha:
            return out, lsnr, df_alpha
        return out, lsnr

    def _feature_extraction_from_spec(self, spec_btf: Tensor):
        """From (B, T, F) complex spec, same outputs as _feature_extraction (no no_grad)."""
        feat_erb = (get_magnitude(spec_btf) ** 2) @ self.erb_fb
        if self.erb_to_db:
            feat_erb = to_db(feat_erb)
        feat_spec = as_real(spec_btf[..., :self.nb_df])
        feat_erb = self.erb_norm(feat_erb)
        feat_spec = self.spec_norm(feat_spec)
        spec = as_real(spec_btf.unsqueeze(1))
        feat_erb = feat_erb.unsqueeze(1)
        feat_spec = feat_spec.unsqueeze(1)
        return spec, feat_erb, feat_spec

    def _init_gru_weight(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
