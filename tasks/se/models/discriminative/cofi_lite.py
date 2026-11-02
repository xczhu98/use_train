"""CoFi-Lite: 超轻量因果语音增强（waveform-native，模型自有 STFT/iSTFT 前端）。

论文: "CoFi-Lite: Pushing the Limits of Ultra-Lightweight Speech Enhancement",
      Leyan Yang, Dahan Wang, Xiaobin Rong, Jiadong Lu, Jing Lu,
      IEEE Signal Processing Letters, 2026.
官方实现: https://github.com/Acceleration123/CoFi-Lite
移植自上游 ``model/cofilite.py``（2026-10-19 发布的 DNS3 checkpoint 版本）。

结构（与上游逐模块对齐）:
  - ``STFTEncoder`` / ``STFTDecoder``: 512/256/512、sqrt-Hann 窗、center=True；
  - ``ERB``: 低频 64 个线性 bin + 高频 65 个 ERB 子带（``low_lim=2000``，
    与上游 released code/checkpoint 的 64+65 划分一致；论文正文写作 65+64）；
  - 双路径编码器: ERB 包络路径（en_erb1..3）+ 低频细节路径（en_lfb1）；
  - ``CrossPathFusion``: 两路特征拼接后经 FC + LayerNorm + ELU + GRU 交互；
  - ``InterRNN``: 频向独立、时向因果的 GRU 细化；
  - 双路径解码器: ERB 包络掩膜（sigmoid）与低频细节掩膜（sigmoid）分别作用后
    合成增强复谱，再由内部 iSTFT 还原波形。

I/O:
  输入 ``noisy_wav`` [B, T]，输出 ``enhanced_wav`` [B, T]。
  与仓库标准判别式谱进谱出模型的区别是：本模型按上游方案在模型内部完成
  STFT/iSTFT（``frontend_owner="model"``），因此 YAML 使用
  ``meta.allow_no_stft: true`` 且不声明顶层 ``stft``。

规模（与上游 released code 对齐）:
  trainable 83,124 params（README 的 83.12k）；
  total 108,214 params（含两个 requires_grad=False 的固定 ERB 滤波器组）；
  上游报告 12.87M MACs/s @ 16 kHz。本文件不重复声明未实测的独立 FLOPs。

官方 checkpoint 兼容性:
  模块属性名、子模块层级和 state_dict key 均保持上游命名（``stft_encoder``、
  ``ERB``、``en_erb1`` ... ``de_lfb1``、``stft_decoder``），可直接
  ``load_state_dict(..., strict=True)`` 加载上游 ``model_on_dns3.tar`` 的
  ``model`` 字段。修改本文件时不要重命名这些属性。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch import Tensor

from core.models.io_contract import ModelIOContract, TensorField
from core.utils.registry import MODEL_REGISTRY
from tasks.se.models.amp_utils import fp32_forward
from tasks.se.models.base_model import BaseSEModel


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


class STFTEncoder(nn.Module):
    """上游 sqrt-Hann 分析前端；输出 [B, 2, T, F]（real/imag 在 channel 轴）。"""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        return_complex: bool = False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.return_complex = return_complex

        window = torch.hann_window(win_length).pow(0.5)
        self.register_buffer("window", window)

    def forward(self, wav: Tensor) -> Tensor:
        # wav: (B, T)
        # 上游使用 return_complex=False；这里先取 complex 再 view_as_real，
        # 数值逐位等价，同时避开 PyTorch 对 return_complex=False 的弃用路径。
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )

        if self.return_complex:
            return spec  # (B, F, T) -> complex spectrogram
        return torch.view_as_real(spec).permute(0, 3, 2, 1)  # (B, 2, T, F)


class STFTDecoder(nn.Module):
    """上游 sqrt-Hann 合成前端；接受 [B, F, T] complex 或 [B, F, T, 2] real。"""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        window = torch.hann_window(win_length).pow(0.5)
        self.register_buffer("window", window)

    def forward(self, spec: Tensor) -> Tensor:
        # spec: (B, F, T) or (B, F, T, 2)
        if len(spec.shape) == 3:
            wav = torch.istft(
                spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
            )
        elif len(spec.shape) == 4:
            if spec.shape[-1] != 2:
                raise ValueError(
                    "CoFi-Lite STFTDecoder expects the last dimension to be 2 "
                    f"for real/imaginary parts, got {tuple(spec.shape)}"
                )
            spec_complex = torch.complex(spec[..., 0], spec[..., 1])
            wav = torch.istft(
                spec_complex,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
            )
        else:
            raise ValueError(
                "CoFi-Lite STFTDecoder input must be 3D or 4D, "
                f"got {tuple(spec.shape)}"
            )

        return wav  # (B, T)


class SFE(nn.Module):
    """Shallow Feature Extraction：沿频轴 unfold，把邻域展开到 channel 维。"""

    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.unfold = nn.Unfold(
            kernel_size=(1, kernel_size),
            stride=stride,
            padding=(0, (kernel_size - 1) // 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T, F)
        x_unfold = self.unfold(x)  # (B, C*kernel_size, F * T)
        x_sfe = x_unfold.contiguous().view(
            x.shape[0], x_unfold.shape[1], x.shape[2], x.shape[3]
        )
        return x_sfe  # (B, C*kernel_size, T, F)


class ERB(nn.Module):
    """固定 ERB 滤波器组：低频保持 FFT bin，高频线性压缩到 ERB 子带。

    上游 ``erb_fc`` / ``ierb_fc`` 的 ``nn.Linear`` 声明形状（257→129、129→257）
    只用于占位；实际注册的固定权重分别是 [65, 193] 与 [193, 65]，前向只作用在
    ``x[..., 64:]`` 的 193 个高频 bin 上。保持这一写法以兼容上游 state_dict。
    """

    def __init__(
        self,
        erb_subband_num: int,
        nfft: int = 512,
        fs: float = 16000.0,
        low_lim: int = 2000,
        high_lim: int = 8000,
    ):
        super().__init__()
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = round(low_lim / fs * nfft)
        erb_subband_2 = erb_subband_num - self.erb_subband_1
        erb_filters = self.erb_filter_banks(
            self.erb_subband_1,
            erb_subband_2,
            nfft=nfft,
            fs=fs,
            low_lim=low_lim,
            high_lim=high_lim,
        )
        self.erb_fc = nn.Linear(nfreqs, erb_subband_num, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_num, nfreqs, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

        self.act = nn.Sigmoid()

    @staticmethod
    def hz2erb(freq_hz):
        return 21.4 * np.log10(0.00437 * freq_hz + 1)

    @staticmethod
    def erb2hz(erb_f):
        return (10 ** (erb_f / 21.4) - 1) / 0.00437

    def erb_filter_banks(
        self,
        erb_subband_1,
        erb_subband_2,
        nfft=512,
        fs=16000,
        low_lim=2000,
        high_lim=8000,
    ):
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (
            (bins[1] - np.arange(bins[0], bins[1]) + 1e-12)
            / (bins[1] - bins[0] + 1e-12)
        )
        for i in range(erb_subband_2 - 2):
            erb_filters[i + 1, bins[i]:bins[i + 1]] = (
                (np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12)
                / (bins[i + 1] - bins[i] + 1e-12)
            )
            erb_filters[i + 1, bins[i + 1]:bins[i + 2]] = (
                (bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12)
                / (bins[i + 2] - bins[i + 1] + 1e-12)
            )
        erb_filters[-1, bins[-2]:bins[-1] + 1] = 1 - erb_filters[
            -2, bins[-2]:bins[-1] + 1
        ]

        return torch.from_numpy(np.abs(erb_filters[:, erb_subband_1:]))

    def erb(self, x):
        """x: (B,C,T,F)"""
        return torch.concat(
            (x[..., :self.erb_subband_1], self.erb_fc(x[..., self.erb_subband_1:])),
            dim=-1,
        )

    def ierb(self, x_erb, flag=False):
        """x: (B,C,T,F_erb)"""
        y = torch.concat(
            (x_erb[..., :self.erb_subband_1], self.ierb_fc(x_erb[..., self.erb_subband_1:])),
            dim=-1,
        )
        if flag:
            return self.act(y)
        return y


class TRA(nn.Module):
    """Temporal Recurrent Attention：时向 GRU 生成逐帧逐通道注意力。"""

    def __init__(self, channels: int, channel_mult: int = 2):
        super().__init__()
        inter_channels = channels * channel_mult

        self.att_gru = nn.GRU(channels, inter_channels, 1, batch_first=True)
        self.att_fc = nn.Linear(inter_channels, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T, F)
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_gru(zt.transpose(1, 2))[0]
        at = self.att_fc(at).transpose(1, 2)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)

        return x * At


class LayerNorm(nn.Module):
    """上游辅助模块；released CoFiLite 图未调用，保留以保持文件级对齐。"""

    def __init__(self, c: int, f: int):
        super().__init__()
        self.ln = nn.LayerNorm([c, f], eps=1e-8)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T, F]
        return self.ln(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)


class ChompT(nn.Module):
    """去掉 ConvTranspose2d 在时间轴产生的尾部多余帧。"""

    def __init__(self, t: int):
        super().__init__()
        self.t = t

    def forward(self, x: Tensor) -> Tensor:
        return x[..., :-self.t, :]


class AffinePReLU(nn.Module):
    """逐通道、逐频点的仿射 PReLU（上游 affine_prelu）。"""

    def __init__(self, channels: int, width: int, init: float = 0.25):
        super().__init__()
        self.affine_weight = nn.Parameter(torch.ones(channels, width))
        self.affine_bias = nn.Parameter(torch.zeros(channels, width))
        self.slope_weight = nn.Parameter(torch.empty(channels))
        nn.init.constant_(self.slope_weight, init)

    def forward(self, x: Tensor) -> Tensor:
        y = self.affine_weight[None, :, None, :] * x + self.affine_bias[None, :, None, :]
        y = y + torch.where(x > 0, x, self.slope_weight.view(1, -1, 1, 1) * x)
        return y


class ConvBlock(nn.Sequential):
    """Conv2d / ConvTranspose2d + BatchNorm + 激活；时轴因果 padding/chomp。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nfreq: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        use_deconv: bool = False,
        separable: bool = False,
        bn: bool = True,
        act: bool = True,
        act_type: str = "affine_prelu",
    ):
        layers = []
        tpad_size = (kernel_size[0] - 1) * dilation[0]
        Groups = out_channels if separable else groups
        In_channels = out_channels if separable else in_channels
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        if separable:
            layers.append(nn.Conv2d(in_channels, out_channels, (1, 1)))
        if not use_deconv:
            layers.append(torch.nn.ConstantPad2d([0, 0, tpad_size, 0], 0.0))
        layers.append(
            conv_module(
                In_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=Groups,
            )
        )
        if use_deconv and tpad_size != 0:
            layers.append(ChompT(tpad_size))

        if bn:
            layers.append(nn.BatchNorm2d(out_channels))

        if act:
            if act_type not in ("prelu", "affine_prelu"):
                raise ValueError(
                    f"CoFi-Lite ConvBlock act_type must be 'prelu' or "
                    f"'affine_prelu', got {act_type!r}"
                )
            if act_type == "prelu":
                layers.append(nn.PReLU(out_channels))
            else:
                layers.append(AffinePReLU(out_channels, nfreq))

        super().__init__(*layers)


class MobileNetBlockv2(nn.Module):
    """MobileNetV2 风格块：1x1 升维 + depthwise/转置卷积 + 1x1 降维 + TRA。"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        hidden_channels: int,
        nfreq: int,
        kernel_size: Tuple[int, int] = (3, 3),
        padding: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        use_deconv: bool = False,
        act_type: str = "prelu",
    ):
        super().__init__()
        freq_before = 2 * nfreq - 1 if not use_deconv else (nfreq + 1) // 2

        self.pwconv1 = ConvBlock(
            in_channels=in_ch,
            out_channels=hidden_channels,
            nfreq=freq_before,
            kernel_size=(1, 1),
            bn=True,
            act=True,
            act_type=act_type,
        )

        self.dwconv = ConvBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            nfreq=nfreq,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=hidden_channels,
            bn=True,
            act=True,
            use_deconv=use_deconv,
            act_type=act_type,
        )

        self.pwconv2 = ConvBlock(
            in_channels=hidden_channels,
            out_channels=out_ch,
            nfreq=nfreq,
            kernel_size=(1, 1),
            bn=True,
            act=False,
            act_type=act_type,
        )

        self.tra = TRA(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pwconv1(x)
        x = self.dwconv(x)
        x = self.pwconv2(x)
        x = self.tra(x)
        return x


class GRNN(nn.Module):
    """Grouped GRU：通道一分为二，两组独立 GRU 后拼接。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.rnn1 = nn.GRU(
            input_size // 2,
            hidden_size // 2,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.rnn2 = nn.GRU(
            input_size // 2,
            hidden_size // 2,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        y1, h1 = self.rnn1(x1)
        y2, h2 = self.rnn2(x2)
        return torch.cat((y1, y2), dim=-1), torch.cat((h1, h2), dim=0)


class InterRNN(nn.Module):
    """频向独立、时向因果的 GRU 细化（上游 InterRNN）。"""

    def __init__(self, channels: int, nfreqs: int):
        super().__init__()
        self.inter_rnn = nn.GRU(
            input_size=channels,
            hidden_size=channels,
            bidirectional=False,
            batch_first=True,
        )
        self.inter_fc = nn.Linear(channels, channels)
        self.inter_ln = nn.LayerNorm(((nfreqs, channels)), eps=1e-8)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T, F]
        bs, ch, nt, nf = x.shape
        ref = x
        inter_x = x.permute(0, 3, 2, 1)  # [B, F, T, C]
        inter_x = inter_x.reshape(bs * nf, nt, ch)  # [B*F, T, C]
        inter_x = self.inter_rnn(inter_x)[0]  # [B*F, T, C]
        inter_x = self.inter_fc(inter_x)  # [B*F, T, C]
        inter_x = inter_x.reshape(bs, nf, nt, ch)  # [B, F, T, C]
        inter_x = inter_x.permute(0, 2, 1, 3)  # [B, T, F, C]
        inter_x = self.inter_ln(inter_x)  # [B, T, F, C]
        inter_out = torch.add(ref, inter_x.permute(0, 3, 1, 2))  # [B, C, T, F]
        return inter_out


class CrossPathFusion(nn.Module):
    """Cross-Path Fusion：两路特征拼接后经 FC+LN+ELU+GRU 交互，再拆回两路。"""

    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=hidden_size)
        self.ln = nn.LayerNorm([hidden_size], eps=1e-8)
        self.act = nn.ELU()

        self.rnn = GRNN(input_size=hidden_size, hidden_size=hidden_size, bidirectional=False)

        self.fc_out = nn.Linear(hidden_size, out_features)

    def forward(self, x1: Tensor, x2: Tensor | None = None) -> Tensor:
        # x1: (B, C, T, F)
        # x2: (B, C, T, F)
        x1_res, x2_res = x1, x2

        bs1, ch1, nt1, nf1 = x1.shape
        x1 = x1.permute(0, 2, 1, 3).flatten(2)  # (B, C, T, F) -> (B, T, C*F)

        if x2 is not None:
            bs2, ch2, nt2, nf2 = x2.shape
            x2 = x2.permute(0, 2, 1, 3).flatten(2)  # (B, C, T, F) -> (B, T, C*F)
            x = torch.cat((x1, x2), dim=-1)  # (B, T, H)
        else:
            x = x1

        x = self.fc(x)  # (B, T, H)
        x = self.ln(x)
        x = self.act(x)
        x = self.rnn(x)[0]  # (B, T, H)
        x = self.fc_out(x)  # (B, T, F)

        if x2 is not None:
            x1, x2 = torch.split(x, [ch1 * nf1, ch2 * nf2], dim=-1)  # (B, T, F1), (B, T, F2)
            x1 = x1.reshape(bs1, nt1, ch1, nf1).permute(0, 2, 1, 3)  # -> (B, C, T, F1)
            x2 = x2.reshape(bs2, nt2, ch2, nf2).permute(0, 2, 1, 3)  # -> (B, C, T, F2)
            return x1 + x1_res, x2 + x2_res

        x = x.reshape(bs1, nt1, ch1, nf1).permute(0, 2, 1, 3)  # -> (B, C, T, F)
        return x


@MODEL_REGISTRY.register("CoFiLite")
class CoFiLite(BaseSEModel):
    """Waveform-in / waveform-out CoFi-Lite。

    Args:
        n_fft / hop_length / win_length / window / center: 仅用于显式消费
            ``build_model`` 注入的 STFT 上下文并校验其与上游一致；本模型内部
            固定使用 512/256/512、sqrt-Hann 窗和 center=True，不能用这些参数
            改变结构。默认值即上游值；传其他值会在构造期报错。

    输入:  ``noisy_wav`` [B, T]（单通道）。
    输出:  ``enhanced_wav`` [B, T]。
    """

    input_domain = "waveform"
    output_domain = "waveform"
    forward_kind = "estimate"
    frontend_owner = "model"

    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({"se"}),
        inputs=(TensorField("noisy", "waveform", ("batch", "time")),),
        outputs=(TensorField("enhanced", "waveform", ("batch", "time")),),
        forward_kind="estimate",
        frontend_owner="model",
    )

    # 上游固定 16 kHz：ERB 滤波器组硬编码 fs=16000 / high_lim=8000。
    target_fs = 16000

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        window: str = "hann",
        center: bool = True,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            raise TypeError(
                f"CoFiLite got unexpected keyword arguments: {sorted(kwargs)}"
            )
        if type(n_fft) is not int or n_fft != 512:
            raise ValueError(
                f"CoFiLite fixes n_fft=512 (upstream 257-bin frontend), got {n_fft!r}"
            )
        if type(hop_length) is not int or hop_length != 256:
            raise ValueError(
                f"CoFiLite fixes hop_length=256, got {hop_length!r}"
            )
        if type(win_length) is not int or win_length != 512:
            raise ValueError(
                f"CoFiLite fixes win_length=512, got {win_length!r}"
            )
        if not isinstance(window, str) or window.lower() != "hann":
            raise ValueError(
                f"CoFiLite fixes the sqrt-Hann analysis window, got {window!r}"
            )
        if center is not True:
            raise ValueError(
                "CoFiLite requires center=True to match the released checkpoint "
                f"and algorithmic latency, got {center!r}"
            )

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window.lower()
        self.center = center
        # 上游固定结构参数（与 released checkpoint 严格对应）。
        hc1 = 6
        hc2 = 6
        hh1 = 6
        hh2 = 6
        hf1 = 17
        hf2 = 33
        hidden_dim = 76
        act_type = "affine_prelu"

        self.stft_encoder = STFTEncoder(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        self.ERB = ERB(
            erb_subband_num=129,
            nfft=n_fft,
            fs=float(self.target_fs),
            low_lim=2000,
            high_lim=8000,
        )
        self.sfe = SFE(kernel_size=3, stride=1)

        self.en_erb1 = MobileNetBlockv2(
            3, hc1, hh1, 65, kernel_size=(3, 5), stride=(1, 2), padding=(0, 2),
            act_type=act_type,
        )
        self.en_erb2 = MobileNetBlockv2(
            hc1, hc1, hh1, 33, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2),
            act_type=act_type,
        )
        self.en_erb3 = MobileNetBlockv2(
            hc1, hc1, hh1, 17, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2),
            act_type=act_type,
        )

        self.en_lfb1 = MobileNetBlockv2(
            9, hc2, hh2, 33, kernel_size=(3, 3), stride=(1, 2), padding=(0, 1),
            act_type=act_type,
        )

        self.spgrnn = CrossPathFusion(
            hf1 * hc1 + hf2 * hc2, hidden_dim, hf1 * hc1 + hf2 * hc2
        )

        self.inter_rnn1 = InterRNN(hc1, hf1)
        self.inter_rnn2 = InterRNN(hc2, hf2)
        self.inter_rnn3 = InterRNN(hc1, hf1)
        self.inter_rnn4 = InterRNN(hc2, hf2)

        self.de_erb3 = MobileNetBlockv2(
            hc1, hc1, hh1, 33, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2),
            use_deconv=True, act_type=act_type,
        )
        self.de_erb2 = MobileNetBlockv2(
            hc1, hc1, hh1, 65, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2),
            use_deconv=True, act_type=act_type,
        )
        self.de_erb1 = MobileNetBlockv2(
            hc1, 1, hh1, 129, kernel_size=(3, 5), stride=(1, 2), padding=(0, 2),
            use_deconv=True, act_type=act_type,
        )

        self.de_lfb1 = MobileNetBlockv2(
            hc2, 1, hh2, 65, kernel_size=(3, 3), stride=(1, 2), padding=(0, 1),
            use_deconv=True, act_type=act_type,
        )

        self.out_act = nn.Sigmoid()
        self.stft_decoder = STFTDecoder(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )

    @staticmethod
    def _validate_waveform(noisy_wav: Tensor) -> None:
        if not torch.is_tensor(noisy_wav):
            raise TypeError(
                f"CoFiLite input must be a torch.Tensor, got {type(noisy_wav).__name__}"
            )
        if noisy_wav.ndim != 2:
            raise ValueError(
                "CoFiLite expects a mono waveform [B, T]; multichannel input is "
                f"not supported, got {tuple(noisy_wav.shape)}"
            )
        if noisy_wav.shape[0] < 1 or noisy_wav.shape[1] < 1:
            raise ValueError(
                "CoFiLite requires non-empty batch and time dimensions, "
                f"got {tuple(noisy_wav.shape)}"
            )
        if not noisy_wav.is_floating_point():
            raise TypeError(
                f"CoFiLite input must be floating point, got {noisy_wav.dtype}"
            )

    @staticmethod
    def _validate_lengths(lengths: Tensor | None, noisy_wav: Tensor) -> None:
        if lengths is None:
            return
        if not torch.is_tensor(lengths):
            raise TypeError(
                f"CoFiLite lengths must be a torch.Tensor, got {type(lengths).__name__}"
            )
        if lengths.ndim != 1 or lengths.shape[0] != noisy_wav.shape[0]:
            raise ValueError(
                f"CoFiLite lengths must have shape [B] with B={noisy_wav.shape[0]}, "
                f"got {tuple(lengths.shape)}"
            )
        if lengths.dtype not in _INTEGER_DTYPES:
            raise TypeError(f"CoFiLite lengths must be integer dtype, got {lengths.dtype}")
        if lengths.device != noisy_wav.device:
            raise ValueError(
                f"CoFiLite lengths device {lengths.device} does not match input "
                f"device {noisy_wav.device}"
            )

    def _validate_fs(self, fs: int | None) -> None:
        if fs is not None and fs != self.target_fs:
            raise ValueError(f"CoFiLite requires fs={self.target_fs}, got {fs}")

    @fp32_forward
    def forward(
        self,
        noisy_wav: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        fs: int | None = None,
    ) -> torch.Tensor:
        """noisy_wav: [B, T]. Returns enhanced_wav [B, T]."""
        self._validate_fs(fs)
        self._validate_waveform(noisy_wav)
        self._validate_lengths(lengths, noisy_wav)
        # 模型参数与官方训练均为 fp32；显式归一化输入 dtype，避免 float64/float16
        # 在内部 STFT/卷积处报低层 dtype 错误。
        noisy_wav = noisy_wav.float()

        n_samples = noisy_wav.shape[-1]
        x = self.stft_encoder(noisy_wav)  # [B, 2, T, F]
        x_ref = x.permute(0, 3, 2, 1)  # [B, F, T, 2]

        X_amp = torch.sqrt(x[:, 0].pow(2) + x[:, 1].pow(2) + 1e-12).unsqueeze(1)  # [B, 1, T, F]
        X_ric = x.div(X_amp.pow(0.7))  # [B, 2, T, F]
        X_log = torch.log10(X_amp)  # [B, 1, T, F]
        x = torch.cat([X_log, X_ric], dim=1)  # [B, 3, T, F]

        x_erb = self.ERB.erb(X_amp)  # [B, 1, T, F=129]
        x_erb = torch.log10(x_erb)
        x_erb = self.sfe(x_erb)  # [B, 3, T, F=129]

        en_x1 = self.en_erb1(x_erb)  # [B, C, T, F=65]
        en_x2 = self.en_erb2(en_x1)  # [B, C, T, F=33]
        en_x3 = self.en_erb3(en_x2)  # [B, C, T, F=17]
        emb_erb = self.inter_rnn1(en_x3)  # [B, C, T, F=17]

        x_f = x[..., :65]  # [B, 3, T, 65]
        x_f = self.sfe(x_f)  # [B, 9, T, 65]

        enf_x1 = self.en_lfb1(x_f)  # [B, C, T, F=33]
        emb_lfb = self.inter_rnn2(enf_x1)  # [B, C, T, F=33]

        emb1, emb2 = self.spgrnn(emb_erb, emb_lfb)

        emb_erb = self.inter_rnn3(emb1)  # [B, C, T, F=17]
        de_x3 = self.de_erb3(emb_erb + en_x3)  # [B, C, T, F=33]
        de_x2 = self.de_erb2(de_x3 + en_x2)  # [B, C, T, F=65]
        de_x1 = self.de_erb1(de_x2 + en_x1)  # [B, C=1, T, F=129]
        de_x1 = self.out_act(de_x1)

        emb_lfb = self.inter_rnn4(emb2)  # [B, C, T, F=33]
        def_x1 = self.de_lfb1(emb_lfb + enf_x1)  # [B, C=1, T, F=65]
        def_x1 = self.out_act(def_x1)

        est_erb = self.ERB.ierb(de_x1).permute(0, 3, 2, 1).contiguous()  # [B, F, T, 1]
        enh0 = x_ref * est_erb  # envelope enhancement
        mask = def_x1.permute(0, 3, 2, 1)  # [B, F_low, T, 1]
        enh = torch.cat([enh0[:, :65] * mask, enh0[:, 65:]], dim=1)  # low-frequency detail
        enh_wav = self.stft_decoder(enh)
        enh_wav = Func.pad(enh_wav, (0, n_samples - enh_wav.shape[-1]))

        if enh_wav.ndim != 2 or enh_wav.shape != noisy_wav.shape:
            raise RuntimeError(
                "CoFiLite output shape mismatch: expected "
                f"{tuple(noisy_wav.shape)}, got {tuple(enh_wav.shape)}"
            )
        return enh_wav


if __name__ == "__main__":
    model = CoFiLite()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    x = torch.randn(1, 16000)
    with torch.no_grad():
        y = model(x)
    print(
        f"CoFiLite params: total={total} ({total / 1e3:.2f}k), "
        f"trainable={trainable} ({trainable / 1e3:.2f}k), "
        f"output={tuple(y.shape)}"
    )
