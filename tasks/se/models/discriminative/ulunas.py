"""UL-UNAS: Ultra-Lightweight U-Nets for Real-Time Speech Enhancement via NAS.

论文: "UL-UNAS: Ultra-Lightweight U-Nets for Real-Time Speech Enhancement via
      Network Architecture Search", Xiaobin Rong, Leyan Yang, Dahan Wang, Yuxiang Hu,
      Changbao Zhu, Kai Chen, Jing Lu. arXiv:2503.00340.
      代码与音频: https://github.com/Xiaobin-Rong/ul-unas

前序工作 GTCRN (ICASSP 2024, ``tasks.se.models.discriminative.gtcrn``) 为同一作者线；UL-UNAS 保留 ERB(BM/BS) 与
G-DPRNN 式瓶颈，编码器/解码器改为 NAS 搜出的 XConv / XDWS / XMB，并引入 APReLU、cTFA 与 IRM 幅度掩膜。

模块分区（速查）:
  - 与 GTCRN 对齐/同构思路: ``ERB``, ``GRNN``, ``DPGRNN``（``DPGRNN`` 的 FC/LN 按论文写为通道维 C）
  - 仅 UL-UNAS: ``AffinePReLU``, ``FA``, ``cTFA``, ``Shuffle``, ``XConvBlock``, ``XDWSBlock``, ``XMBBlocks``

谱域接口: 输入/输出 noisy_spec (B, F, T, 2)；STFT/iSTFT 由训练框架完成。

NAS 默认结构 (Table IV): types [XConv, XMB, XDWS, XMB, XDWS], strides [2,2,1,1,1],
groups [1,2,2,2,2], channels [12,24,24,32,16], kernels [(3,3),(2,3),(2,3),(1,5),(1,5)]。
"""
from __future__ import annotations

import os
import sys
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

import logging

from tasks.se.models.amp_utils import fp32_forward
from tasks.se.models.base_model import BaseSEModel
from core.utils.registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)

_ERB_16K_WARNED = False


def _warn_erb_16k_once(model_name: str) -> None:
    """ERB 16k 假设告警，每进程只打一次（避免 DDP 多 rank/重复构建刷屏）。"""
    global _ERB_16K_WARNED
    if not _ERB_16K_WARNED:
        _ERB_16K_WARNED = True
        logger.warning(
            "%s assumes fs=16000 Hz: its ERB filterbank is hardcoded to "
            "high_lim=8000/fs=16000 and is NOT adapted to the actual sample rate.",
            model_name,
        )


# =============================================================================
# SHARED LINEAGE (1/2) — ERB
# GRNN / DPGRNN 紧随「UL-UNAS-only」X 块之后（本文件中部）。与 ``gtcrn`` 同脉络；ERB 两处请保持同步。
# =============================================================================


class ERB(nn.Module):
    """Equivalent Rectangular Bandwidth：低频保持分辨率，高频经固定滤波器组压缩。

    ``bm`` / ``bs`` 与 ``gtcrn.ERB`` 同构（Band Merge / Band Split）。修改一处时请同步另一文件。
    """

    def __init__(self, erb_subband_1: int, erb_subband_2: int, nfft: int = 512, high_lim: float = 8000, fs: int = 16000):
        super().__init__()
        erb_filters = self._erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs - erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs - erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    @staticmethod
    def _hz2erb(freq_hz: float) -> float:
        return 21.4 * np.log10(0.00437 * freq_hz + 1)

    @staticmethod
    def _erb2hz(erb_f: float) -> float:
        return (10 ** (erb_f / 21.4) - 1) / 0.00437

    def _erb_filter_banks(self, erb_subband_1: int, erb_subband_2: int, nfft: int, high_lim: float, fs: int):
        low_lim = erb_subband_1 / nfft * fs
        erb_low, erb_high = self._hz2erb(low_lim), self._hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self._erb2hz(erb_points) / fs * nfft).astype(np.int32)
        f = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)
        f[0, bins[0] : bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2 - 2):
            f[i + 1, bins[i] : bins[i + 1]] = (
                np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12
            ) / (bins[i + 1] - bins[i] + 1e-12)
            f[i + 1, bins[i + 1] : bins[i + 2]] = (
                bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12
            ) / (bins[i + 2] - bins[i + 1] + 1e-12)
        f[-1, bins[-2] : bins[-1] + 1] = 1 - f[-2, bins[-2] : bins[-1] + 1]
        f = f[:, erb_subband_1:]
        return torch.from_numpy(np.abs(f))

    def bm(self, x: torch.Tensor) -> torch.Tensor:
        """Band Merge: 频域 -> ERB（低频保留 + 高频线性压缩）。"""
        return torch.cat([x[..., : self.erb_subband_1], self.erb_fc(x[..., self.erb_subband_1 :])], dim=-1)

    def bs(self, x_erb: torch.Tensor) -> torch.Tensor:
        """Band Split: ERB -> 频域（低频保留 + 高频展开）。"""
        return torch.cat([x_erb[..., : self.erb_subband_1], self.ierb_fc(x_erb[..., self.erb_subband_1 :])], dim=-1)


# =============================================================================
# UL-UNAS-only — APReLU / cTFA / NAS 卷积块（GTCRN 使用 SFE+TRA+GTConvBlock，不包含本节）
# 论文 Sec. III-C2 (APReLU), III-C3 (cTFA), Fig. 3 (XConv/XDWS/XMB)
# =============================================================================
class AffinePReLU(nn.Module):
    """Affine PReLU: h(x) = gamma*x + beta + PReLU(x), gamma/beta (C,F), alpha (C)."""

    def __init__(self, channels: int, width: int, init: float = 0.25):
        super().__init__()
        self.affine_weight = nn.Parameter(torch.ones(channels, width))
        self.affine_bias = nn.Parameter(torch.zeros(channels, width))
        self.slope_weight = nn.Parameter(torch.empty(channels))
        nn.init.constant_(self.slope_weight, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.affine_weight[None, :, None, :] * x + self.affine_bias[None, :, None, :]
        y = y + torch.where(x > 0, x, self.slope_weight.view(1, -1, 1, 1) * x)
        return y


class FA(nn.Module):
    """Frequency attention branch in cTFA: channel-pooled -> fold -> BiGRU -> FC."""

    def __init__(self, nfreq: int, freq_comp_ratio: int = 4):
        super().__init__()
        self.r = freq_comp_ratio
        self.gru = nn.GRU(self.r, self.r, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * self.r, self.r)
        remainder = nfreq % self.r
        self.pad_len = (self.r - remainder) if remainder != 0 else 0
        self.F_pad = nfreq + self.pad_len
        self.H = self.F_pad // self.r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, F = x.shape
        x = torch.mean(x.pow(2), dim=1)
        x = nn.functional.pad(x, (0, self.pad_len))
        x = x.view(B, T, self.H, self.r)
        x = rearrange(x, "b t h c -> (b t) h c")
        x, _ = self.gru(x)
        x = self.fc(x)
        x = rearrange(x, "(b t) h c -> b t h c", b=B)
        x = x.reshape(B, T, self.F_pad)
        if self.pad_len > 0:
            x = x[..., :F]
        return x


class cTFA(nn.Module):
    """Causal time-frequency attention: TA (GRU on time) * FA (channel-pooled freq)."""

    def __init__(self, channels: int, width: int):
        super().__init__()
        self.channels = channels
        self.ta_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.ta_fc = nn.Linear(channels * 2, channels)
        self.fa = FA(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, F)
        zt = torch.mean(x.pow(2), dim=-1)  # (B, C, T)
        at = self.ta_gru(zt.transpose(1, 2))[0]  # (B, C, T)
        at = self.ta_fc(at).transpose(1, 2)  # (B, C, T)
        at = torch.sigmoid(at)  # (B, C, T)
        af = self.fa(x)  # (B, C, T, F)
        af = torch.sigmoid(af)
        return at[..., None] * x * af[:, None]


class Shuffle(nn.Module):
    """Channel shuffle for groups=2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()
        x = rearrange(x, "b c g t f -> b (c g) t f")
        return x


class XConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int,
        kernel_size: int | tuple,
        stride: int = 1,
        groups: int = 1,
        use_deconv: bool = False,
        is_last: bool = False,
    ):
        super().__init__()
        self.g = groups
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        kt, kf = kernel_size[0], kernel_size[1]
        pt = (kt - 1) if use_deconv else 0
        pf = kf // 2
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.ops = nn.Sequential(
            nn.ZeroPad2d([0, 0, kt - 1, 0]),
            conv_module(in_channels, out_channels, kernel_size, stride=(1, stride), padding=(pt, pf), groups=groups),
            nn.BatchNorm2d(out_channels),
            AffinePReLU(out_channels, width) if not is_last else nn.Identity(),
            cTFA(out_channels, width),
            Shuffle() if (not is_last and groups == 2) else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ops(x)


class XDWSBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int,
        kernel_size: int | tuple,
        stride: int = 1,
        groups: int = 1,
        use_deconv: bool = False,
        is_last: bool = False,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        kt, kf = kernel_size[0], kernel_size[1]
        pt = (kt - 1) if use_deconv else 0
        pf = kf // 2
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        in_width = (width * 2 - 1) if (stride == 2 and not use_deconv) else (width // 2 + 1 if stride == 2 else width)
        self.pconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, groups=groups),
            nn.BatchNorm2d(out_channels),
            AffinePReLU(out_channels, in_width),
            Shuffle() if groups == 2 else nn.Identity(),
        )
        self.dconv = nn.Sequential(
            nn.ZeroPad2d([0, 0, kt - 1, 0]),
            conv_module(out_channels, out_channels, kernel_size, stride=(1, stride), padding=(pt, pf), groups=out_channels),
            nn.BatchNorm2d(out_channels),
            AffinePReLU(out_channels, width) if not is_last else nn.Identity(),
            cTFA(out_channels, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pconv(x)
        return self.dconv(h)


class XMBBlocks(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int,
        kernel_size: int | tuple,
        stride: int = 1,
        groups: int = 1,
        use_deconv: bool = False,
        is_last: bool = False,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        kt, kf = kernel_size[0], kernel_size[1]
        pt = (kt - 1) if use_deconv else 0
        pf = kf // 2
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        in_width = (width * 2 - 1) if (stride == 2 and not use_deconv) else (width // 2 + 1 if stride == 2 else width)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, groups=groups),
            nn.BatchNorm2d(out_channels),
            AffinePReLU(out_channels, in_width),
            Shuffle() if groups == 2 else nn.Identity(),
        )
        self.dconv = nn.Sequential(
            nn.ZeroPad2d([0, 0, kt - 1, 0]),
            conv_module(out_channels, out_channels, kernel_size, stride=(1, stride), padding=(pt, pf), groups=out_channels),
            nn.BatchNorm2d(out_channels),
            AffinePReLU(out_channels, width),
        )
        self.pconv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, groups=groups),
            nn.BatchNorm2d(out_channels),
            cTFA(out_channels, width),
        )
        self.shuffle = Shuffle() if (not is_last and groups == 2) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.pconv1(x)
        x = self.dconv(x)
        x = self.pconv2(x)
        if x.shape == identity.shape:
            x = x + identity
        return self.shuffle(x)


# =============================================================================
# SHARED LINEAGE (2/2) — GRNN / DPGRNN（G-DPRNN 瓶颈；``gtcrn`` 中同名类同构）
# =============================================================================


class GRNN(nn.Module):
    """Grouped GRU：通道维一分为二，两组独立 GRU 再拼接。

    与 ``gtcrn.GRNN`` 同构；供 G-DPRNN 的 intra / inter 子路径使用。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(
            input_size // 2, hidden_size // 2, num_layers, batch_first=batch_first, bidirectional=bidirectional
        )
        self.rnn2 = nn.GRU(
            input_size // 2, hidden_size // 2, num_layers, batch_first=batch_first, bidirectional=bidirectional
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if h is None:
            h = torch.zeros(
                self.num_layers * (2 if self.bidirectional else 1),
                x.shape[0],
                self.hidden_size,
                device=x.device,
                dtype=x.dtype,
            )
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        return torch.cat([y1, y2], dim=-1), torch.cat([h1, h2], dim=-1)


class DPGRNN(nn.Module):
    """Dual-Path Grouped RNN（G-DPRNN）：频向 intra（双向）+ 时向 inter（因果）。

    论文 UL-UNAS 瓶颈形式：intra/inter 后 ``Linear(H, C)`` 与 ``LayerNorm((F, C))``（C=input_size）。
    ``gtcrn.DPGRNN`` 为 ICASSP GTCRN 实现：``Linear(H,H)`` 与 ``LayerNorm(..., H)``；
    默认通道下二者形状可对齐，对比实现时请分文件阅读。
    """

    def __init__(self, input_size: int, width: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size
        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, input_size)
        self.intra_ln = nn.LayerNorm((width, input_size), eps=1e-8)
        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, input_size)
        self.inter_ln = nn.LayerNorm((width, input_size), eps=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        B, T, F, C = x.shape
        intra_x = x.reshape(B * T, F, C)
        intra_x, _ = self.intra_rnn(intra_x)
        intra_x = self.intra_fc(intra_x)
        intra_x = intra_x.reshape(B, T, self.width, self.input_size)
        intra_x = self.intra_ln(intra_x)
        intra_out = x + intra_x
        inter_x = intra_out.permute(0, 2, 1, 3).reshape(B * F, T, C)
        inter_x, _ = self.inter_rnn(inter_x)
        inter_x = self.inter_fc(inter_x)
        inter_x = inter_x.reshape(B, self.width, T, self.input_size).permute(0, 2, 1, 3)
        inter_x = self.inter_ln(inter_x)
        inter_out = intra_out + inter_x
        return inter_out.permute(0, 3, 1, 2)


# =============================================================================
# U-Net 编码器/解码器（NAS Table IV 默认；块类型见上方 UL-UNAS-only）
# =============================================================================


class _Encoder(nn.Module):
    def __init__(
        self,
        types: list[int],
        channels: list[int],
        widths: list[int],
        kernels: list[tuple[int, int]],
        strides: list[int],
        groups: list[int],
    ):
        super().__init__()
        block_types = [XConvBlock, XDWSBlock, XMBBlocks]
        en_convs = []
        in_ch = 1
        for i in range(len(types)):
            out_ch = channels[i]
            en_convs.append(
                block_types[types[i]](
                    in_ch, out_ch, widths[i], kernels[i], strides[i], groups=groups[i]
                )
            )
            in_ch = out_ch
        self.en_convs = nn.ModuleList(en_convs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        en_outs = []
        for layer in self.en_convs:
            x = layer(x)
            en_outs.append(x)
        return x, en_outs


class _Decoder(nn.Module):
    def __init__(
        self,
        types: list[int],
        channels: list[int],
        widths: list[int],
        kernels: list[tuple[int, int]],
        strides: list[int],
        groups: list[int],
        erb_freq: int,
    ):
        super().__init__()
        block_types = [XConvBlock, XDWSBlock, XMBBlocks]
        n_blocks = len(types)
        de_convs = []
        in_ch = channels[-1]
        for i in range(n_blocks - 1, 0, -1):
            out_ch = channels[i - 1]
            de_convs.append(
                block_types[types[i]](
                    in_ch, out_ch, widths[i - 1], kernels[i], strides[i], groups[i], use_deconv=True
                )
            )
            in_ch = out_ch
        de_convs.append(
            block_types[types[0]](
                in_ch, 1, erb_freq, kernels[0], strides[0], groups[0], use_deconv=True, is_last=True
            )
        )
        self.de_convs = nn.ModuleList(de_convs)

    def forward(self, x: torch.Tensor, en_outs: list[torch.Tensor]) -> torch.Tensor:
        n = len(self.de_convs)
        for i in range(n):
            x = self.de_convs[i](x + en_outs[n - 1 - i])
        return torch.sigmoid(x)


# --- 注册模型与 NAS 默认超参 (Table IV) ----------------------------------------
_DEFAULT_TYPES = [0, 2, 1, 2, 1]   # XConv, XMB, XDWS, XMB, XDWS
_DEFAULT_STRIDES = [2, 2, 1, 1, 1]
_DEFAULT_GROUPS = [1, 2, 2, 2, 2]
_DEFAULT_CHANNELS = [12, 24, 24, 32, 16]
_DEFAULT_KERNELS = [(3, 3), (2, 3), (2, 3), (1, 5), (1, 5)]
_DEFAULT_WIDTHS = [65, 33, 33, 33, 33]
_ERB_LOW, _ERB_HIGH = 65, 64
_ERB_FREQ = _ERB_LOW + _ERB_HIGH  # 129


@MODEL_REGISTRY.register("ULUNAS")
class ULUNAS(BaseSEModel):
    """UL-UNAS: spectrum-in / spectrum-out. No STFT/iSTFT; trainer does wav<->spec.

    Input:  noisy_spec (B, F, T, 2), F = n_fft//2+1.
    Output: enhanced_spec (B, F, T, 2). Magnitude mask (IRM) applied to noisy phase.
    """

    def __init__(
        self,
        n_fft: int = 512,
        types: list[int] | None = None,
        strides: list[int] | None = None,
        groups: list[int] | None = None,
        channels: list[int] | None = None,
        kernels: list[tuple[int, int]] | None = None,
        widths: list[int] | None = None,
        num_dpgrnn: int = 2,
        **kwargs,
    ):
        super().__init__()
        target_fs = kwargs.pop("target_fs", 16000)
        for key in ("hop_length", "win_length", "window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"ULUNAS got unexpected keyword arguments: {sorted(kwargs)}")
        if target_fs != 16000:
            raise ValueError(f"ULUNAS requires target_fs=16000, got {target_fs}")
        self.n_fft = n_fft
        self.target_fs = target_fs
        types = types or _DEFAULT_TYPES
        strides = strides or _DEFAULT_STRIDES
        groups = groups or _DEFAULT_GROUPS
        channels = channels or _DEFAULT_CHANNELS
        kernels = kernels or _DEFAULT_KERNELS
        widths = widths or _DEFAULT_WIDTHS

        # ERB 滤波器组固定为 fs=16000 / high_lim=8000；其他采样率会静默产生错误子带划分。
        _warn_erb_16k_once("ULUNAS")
        self.erb = ERB(_ERB_LOW, _ERB_HIGH, nfft=n_fft)
        self.encoder = _Encoder(types, channels, widths, kernels, strides, groups)
        self.dpgrnn = nn.Sequential(
            *[DPGRNN(channels[-1], widths[-1], channels[-1]) for _ in range(num_dpgrnn)]
        )
        self.decoder = _Decoder(types, channels, widths, kernels, strides, groups, _ERB_FREQ)

    @fp32_forward
    def forward(self, noisy_spec: torch.Tensor, lengths=None, *, fs: int | None = None) -> torch.Tensor:
        """noisy_spec: (B, F, T, 2). Returns enhanced_spec (B, F, T, 2)."""
        if fs is not None and fs != self.target_fs:
            raise ValueError(f"ULUNAS requires fs={self.target_fs}, got {fs}")
        if noisy_spec.dim() != 4 or noisy_spec.shape[-1] != 2:
            raise ValueError(f"Expected [B, F, T, 2], got {tuple(noisy_spec.shape)}")
        if noisy_spec.shape[1] != self.n_fft // 2 + 1:
            raise ValueError(f"Expected F={self.n_fft // 2 + 1}, got F={noisy_spec.shape[1]}")
        if noisy_spec.shape[0] == 0 or noisy_spec.shape[2] == 0:
            raise ValueError("ULUNAS requires non-empty batch and time dimensions")
        if not noisy_spec.is_floating_point():
            raise TypeError("ULUNAS noisy_spec must use a floating-point dtype")
        x = noisy_spec.float()
        spec_2ch = x.permute(0, 3, 2, 1)
        mag = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-12)
        feat = torch.log10(mag.clamp(1e-12)).permute(0, 2, 1).unsqueeze(1)
        feat = self.erb.bm(feat)
        feat, en_outs = self.encoder(feat)
        feat = self.dpgrnn(feat)
        m_erb = self.decoder(feat, en_outs)
        m = self.erb.bs(m_erb)
        spec_enh = spec_2ch * m
        return spec_enh.permute(0, 3, 2, 1)


if __name__ == "__main__":
    from core.utils.model_profile import profile_model, make_profile_parser
    from core.stft import build_stft_istft

    args = make_profile_parser(default_sr=16000).parse_args()
    STFT_CFG = {"n_fft": 512, "hop_length": 256, "win_length": 512, "window": "hann"}
    model = ULUNAS(n_fft=STFT_CFG["n_fft"])
    stft_module = build_stft_istft(STFT_CFG)

    class _W(nn.Module):
        def __init__(self, m, stft_m):
            super().__init__()
            self._m, self._stft = m, stft_m
        def forward(self, wav):
            spec = self._stft(wav)
            return self._stft.inverse(self._m(spec), wav.shape[-1])

    wrapped = _W(model, stft_module)
    print("\n[Config] STFT:", STFT_CFG, "| sample_rate:", args.sr)
    profile_model(
        wrapped, sample_rate=args.sr, duration=args.dur, batch_size=args.batch,
        device=args.device, repeats=args.repeats, label="ULUNAS",
    )
