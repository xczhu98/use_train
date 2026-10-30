"""DBGNet - Dual-Branch Guidance Encoder for Robust Acoustic Echo Suppression.

Reference: Hyewon Han et al., "Dual-Branch Guidance Encoder for Robust Acoustic
Echo Suppression", IEEE/ACM TASLP, vol. 33, 2025.
Paper/demo: https://aec-project.github.io
Unofficial implementation reference: https://github.com/vkeep/DBGNet

Implementation notes:
- Spectrum-in / spectrum-out: AECTrainer feeds ``mic_spectra`` / ``farend_spectra``
  of shape ``(B, F, T, 2)`` and expects ``nearend_estimate_spectra`` of the same shape.
- ``mask_domain`` 选择掩码域：``"mag"``（默认）估计实值 sigmoid 掩码并乘到复数
  麦克风谱上，保留麦克风相位（论文 DBG-Mag）；``"com"`` 估计 tanh 复数比值掩码，
  与麦克风复谱做复数乘法（论文 DBG-Com）。
- 论文主线是 DBG-Mag，DBG-Com 只是论文报告的对照变体。论文仅把 DBG-Com 描述为
  "modifying the input and output configurations from magnitude to complex
  spectrum"，没有给出掩码参数化与激活函数，也没有官方源码；`vkeep/DBGNet` 同样
  只有实值 magnitude-mask 路径。本模块的 com 实现按该句字面描述采用 2 通道 RI
  输入 + tanh 复数比值掩码，属于项目对论文缺口的显式选择，不声称与作者实现等价。
- F-GRU 宽度由 ``fgru_hidden`` 决定，默认 ``mic_channels[-1] + far_channels[-1]``，
  即论文 Table I 的每方向宽度（论文通道下 192，双向输出 384；生产 small 通道下 96）。
  非官方参考实现 vkeep/DBGNet 使用 ``(mic[-1]+far[-1]) // 2``（论文通道下 96），与论文
  Table I 不一致；按该旧规则训练出的 checkpoint 形状更窄，必须按当前规则重训。
  T-GRU 输出 128 与论文一致（输入宽度 = 2 × fgru_hidden）。
- Causal along time: encoder uses left-only time padding; decoder upsampling adds
  one future frame which is cropped to maintain causality.
- Optional auxiliary frame-level state detector.  The detector head is now
  imported from the reusable module ``tasks.aec.models.state_detector`` so that other
  AEC models can share the same supervision.  It supports frame-level
  mutually-exclusive class logits (3-class FST/DT/NST or 8-class scenario)
  consumed by ``StateCrossEntropyLoss``.  The original DBGNet paper uses
  segment-level 3-class softmax; here the temporal resolution is frame-level.
  When enabled, ``batch["state_probs"]`` of shape ``(B, T, state_nclass)`` is
  populated; the standard AEC trainer ignores it unless a dedicated multi-task
  loss consumes a corresponding ``state`` label.

Usage (YAML):
    model:
      type: DBGNet
      args:
        n_fft: 512
        hop_length: 256
        win_length: 512
        sample_rate: 16000
        erb_low_bins: 65
        erb_bands: 64
        use_state_detector: false

Run directly to test params and compute:
    python tasks/aec/models/discriminative/dbgnet.py [--sr 16000] [--dur 1.0] [--device cpu]
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tasks.aec.models.state_detector import AECStateDetectorHead
from core.utils.registry import MODEL_REGISTRY


def _causal_pad_t(x: Tensor, pad_left: int) -> Tensor:
    """Causal time padding for tensors with layout ``(B, C, F, T)``.

    Pads only the left side of the last (time) dimension.
    """
    return F.pad(x, (pad_left, 0, 0, 0))


_MASK_DOMAINS = ("mag", "com")


def _validate_mask_domain(value: str) -> str:
    """校验掩码域；只接受论文报告的 DBG-Mag / DBG-Com 两种变体。"""
    if not isinstance(value, str):
        raise TypeError(f"mask_domain must be a string, got {type(value).__name__}")
    domain = value.strip().lower()
    if domain not in _MASK_DOMAINS:
        raise ValueError(f"mask_domain must be one of {list(_MASK_DOMAINS)}, got {value!r}")
    return domain


def _apply_complex_mask(mic_spec: Tensor, mask: Tensor) -> Tensor:
    """复数掩码乘法：``mask`` 为 ``(B, F, T, 2)`` 实/虚分量。"""
    mic_re, mic_im = mic_spec[..., 0], mic_spec[..., 1]
    mask_re, mask_im = mask[..., 0], mask[..., 1]
    return torch.stack(
        [mic_re * mask_re - mic_im * mask_im, mic_re * mask_im + mic_im * mask_re],
        dim=-1,
    )


class _ERB(nn.Module):
    """Equivalent Rectangular Bandwidth (ERB) filterbank used by DBGNet.

    Splits the full frequency axis into ``low_bins`` low-frequency bins kept as-is
    and ``erb_bands`` ERB-compressed high-frequency bands. ``bm`` compresses the
    full magnitude spectrum; ``bs`` synthesizes back to the full frequency axis.

    Args:
        low_bins: Number of lowest frequency bins kept uncompressed.
        erb_bands: Number of ERB bands for the remaining frequencies.
        n_fft: FFT size; full frequency bin count is ``n_fft // 2 + 1``.
        high_freq: Upper frequency limit for ERB warping (Hz).
        sample_rate: Sampling rate (Hz).
    """

    def __init__(
        self,
        low_bins: int,
        erb_bands: int,
        n_fft: int = 512,
        high_freq: int = 8000,
        sample_rate: int = 16000,
    ):
        super().__init__()
        for name, value in (
            ("low_bins", low_bins),
            ("erb_bands", erb_bands),
            ("n_fft", n_fft),
            ("sample_rate", sample_rate),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {value!r}")
        if n_fft <= 0 or n_fft % 2 != 0:
            raise ValueError(f"n_fft must be a positive even integer, got {n_fft}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if not isinstance(high_freq, (int, float)) or isinstance(high_freq, bool):
            raise TypeError(f"high_freq must be a real number, got {high_freq!r}")
        high_freq = float(high_freq)
        if not math.isfinite(high_freq) or not 0.0 < high_freq <= sample_rate / 2:
            raise ValueError(
                f"high_freq must be finite and in (0, {sample_rate / 2}], got {high_freq!r}"
            )

        nfreqs = n_fft // 2 + 1
        if not 1 <= low_bins < nfreqs:
            raise ValueError(f"low_bins must be in [1, {nfreqs - 1}], got {low_bins}")
        if erb_bands < 2:
            raise ValueError(f"erb_bands must be at least 2, got {erb_bands}")
        low_freq = low_bins / n_fft * sample_rate
        if high_freq <= low_freq:
            raise ValueError(
                f"high_freq={high_freq} must exceed the first ERB frequency {low_freq}"
            )

        self.low_bins = low_bins
        self.erb_bands = erb_bands
        self.n_fft = n_fft
        self.sample_rate = sample_rate

        erb_filters = self._build_filterbank(
            low_bins, erb_bands, nfreqs, high_freq, sample_rate
        )

        self.erb_fc = nn.Linear(nfreqs - low_bins, erb_bands, bias=False)
        self.ierb_fc = nn.Linear(erb_bands, nfreqs - low_bins, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    @staticmethod
    def _hz_to_erb(freq_hz: float) -> float:
        return 24.7 * math.log10(0.00437 * freq_hz + 1.0)

    @staticmethod
    def _erb_to_hz(erb_f: float) -> float:
        return (10.0 ** (erb_f / 24.7) - 1.0) / 0.00437

    def _build_filterbank(
        self,
        low_bins: int,
        erb_bands: int,
        nfreqs: int,
        high_freq: float,
        sample_rate: float,
    ) -> Tensor:
        low_freq = low_bins / self.n_fft * sample_rate
        erb_low = self._hz_to_erb(low_freq)
        erb_high = self._hz_to_erb(high_freq)
        erb_points = np.linspace(erb_low, erb_high, erb_bands)
        bins = np.round(self._erb_to_hz(erb_points) / sample_rate * self.n_fft).astype(np.int32)
        if bins[0] < low_bins or bins[-1] >= nfreqs:
            raise ValueError(
                f"ERB centers must stay in [{low_bins}, {nfreqs - 1}], got "
                f"[{int(bins[0])}, {int(bins[-1])}]"
            )
        if np.any(np.diff(bins) <= 0):
            raise ValueError(
                "ERB band centers must be strictly increasing after bin rounding; "
                f"got centers={bins.tolist()}"
            )

        filters = np.zeros([erb_bands, nfreqs], dtype=np.float32)

        # First band: decreasing ramp from bins[0] to bins[1].
        filters[0, bins[0] : bins[1]] = (
            bins[1] - np.arange(bins[0], bins[1]) + 1e-12
        ) / (bins[1] - bins[0] + 1e-12)

        # Middle bands: triangular filters.
        for i in range(erb_bands - 2):
            filters[i + 1, bins[i] : bins[i + 1]] = (
                np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12
            ) / (bins[i + 1] - bins[i] + 1e-12)
            filters[i + 1, bins[i + 1] : bins[i + 2]] = (
                bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12
            ) / (bins[i + 2] - bins[i + 1] + 1e-12)

        # Last band: complement of the second-to-last band.
        filters[-1, bins[-2] : bins[-1] + 1] = 1.0 - filters[-2, bins[-2] : bins[-1] + 1]

        filters = filters[:, low_bins:]
        return torch.from_numpy(np.abs(filters))

    def bm(self, x: Tensor) -> Tensor:
        """Band-merge: ``(B, F, T)`` -> ``(B, low_bins + erb_bands, T)``."""
        x_low = x[..., : self.low_bins, :]
        x_high = self.erb_fc(x[..., self.low_bins :, :].transpose(-1, -2)).transpose(-1, -2)
        return torch.cat([x_low, x_high], dim=-2)

    def bs(self, x_erb: Tensor) -> Tensor:
        """Band-split: ``(B, low_bins + erb_bands, T)`` -> ``(B, F, T)``."""
        x_low = x_erb[..., : self.low_bins, :]
        x_high = self.ierb_fc(x_erb[..., self.low_bins :, :].transpose(-1, -2)).transpose(-1, -2)
        return torch.cat([x_low, x_high], dim=-2)


class _FTGBlock(nn.Module):
    """Far-end Tracking GRU.

    Applies a small causal GRU over the time dimension after flattening channels,
    then reshapes back and adds a residual connection.

    Args:
        num_features: Number of frequency bins at the current resolution.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=num_features,
            bidirectional=False,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, F, T)
        b, c, f, t = x.shape
        y = x.flatten(0, 1).permute(0, 2, 1)  # (B*C, T, F)
        y, _ = self.gru(y)
        y = y.permute(0, 2, 1).view(b, c, f, t)
        return y + x


class _SLMBlock(nn.Module):
    """Separative Latent Masking block.

    Generates a guidance map from mic and far features, splits the mic feature
    into echo-related and echo-unrelated parts, and refines both branches.  Both
    guided outputs feed the next encoder layer.

    Args:
        mic_channels: Number of channels in the microphone branch.
        far_channels: Number of channels in the far-end branch.
    """

    def __init__(self, mic_channels: int, far_channels: int):
        super().__init__()
        self.gmap_conv_mic = nn.Sequential(
            nn.Conv2d(mic_channels, mic_channels, kernel_size=1),
            nn.BatchNorm2d(mic_channels),
            nn.PReLU(),
        )
        self.gmap_conv_far = nn.Sequential(
            nn.Conv2d(far_channels, far_channels, kernel_size=1),
            nn.BatchNorm2d(far_channels),
            nn.PReLU(),
        )
        self.gmap_conv_concat = nn.Sequential(
            nn.Conv2d(mic_channels + far_channels, mic_channels, kernel_size=1),
            nn.BatchNorm2d(mic_channels),
            nn.Sigmoid(),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(mic_channels, mic_channels, kernel_size=1),
            nn.BatchNorm2d(mic_channels),
            nn.PReLU(),
        )
        self.guiding_conv_mic = nn.Sequential(
            nn.Conv2d(mic_channels * 2, mic_channels, kernel_size=1),
            nn.BatchNorm2d(mic_channels),
            nn.PReLU(),
        )
        self.guiding_conv_far = nn.Sequential(
            nn.Conv2d(mic_channels + far_channels, far_channels, kernel_size=1),
            nn.BatchNorm2d(far_channels),
            nn.PReLU(),
        )

    def forward(self, dy: Tensor, dx: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # dy: (B, C_mic, F, T), dx: (B, C_far, F, T)
        qy = self.gmap_conv_mic(dy)
        qx = self.gmap_conv_far(dx)
        mg = self.gmap_conv_concat(torch.cat([qy, qx], dim=1))

        vy = self.conv1x1(dy)
        uy_echo = vy * mg
        uy_speech = vy * (1.0 - mg)

        gy = self.guiding_conv_mic(torch.cat([dy, uy_speech], dim=1))
        gx = self.guiding_conv_far(torch.cat([dx, uy_echo], dim=1))
        return gy, gx, mg


class _DecoderBlock(nn.Module):
    """DBGNet decoder block with gated skip fusion.

    Args:
        in_channels: Number of input / skip channels.
        out_channels: Number of output channels.
        kernel_size: ConvTranspose2d kernel size.
        stride: ConvTranspose2d stride.
        padding: ConvTranspose2d padding.
        output_padding: ConvTranspose2d output padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        output_padding: Tuple[int, int] = (0, 0),
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                2 * in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x: Tensor, x_enc: Tensor) -> Tensor:
        m_enc = self.conv(torch.cat([x_enc, x], dim=1))
        x_enc = m_enc * x_enc
        return self.deconv(torch.cat([x_enc, x], dim=1))


@MODEL_REGISTRY.register("DBGNet")
class DBGNet(nn.Module):
    """DBGNet AEC model.

    通过 ``@MODEL_REGISTRY.register("DBGNet")`` 注册，并与 ``tasks.aec.models``
    lazy map 中的公开名称保持一致。

    Args:
        n_fft: FFT 大小，决定频率 bin 数 ``n_fft // 2 + 1``。
        hop_length: STFT 帧移（仅存储参考；STFT 由外部执行）。
        win_length: STFT 窗长。
        sample_rate: 采样率 (Hz)。
        erb_low_bins: ERB 保留的低频不压缩 bin 数。
        erb_bands: ERB 压缩带数。
        mic_channels: 麦克风编码器各层通道数，长度 4。
            默认 (32, 64, 96, 128)。缩小可降低算力，如 (16, 32, 48, 64)。
        far_channels: 远端编码器各层通道数，长度 4。
            默认 (16, 32, 48, 64)。通常为 mic_channels 的一半。
        fgru_hidden: 频率维 GRU 隐层大小，默认自动推导 ``mic[-1]+far[-1]``，即论文
            Table I 的每方向宽度（论文通道下 192，双向输出 384）。历史 vkeep 参考
            实现用 ``(mic[-1]+far[-1])//2``；依赖旧推导的配置必须显式写出旧值。
        tgru_hidden: 时间维 GRU 隐层大小，默认自动推导 mic[-1]（= 论文的 128）。
        state_feat_dim: 状态检测器每级特征维度，默认 64。
        state_hidden_dim: 状态检测器 RNN 隐层大小，默认 128。
        use_state_detector: 是否启用辅助帧级状态分类器。
        state_nclass: 状态类别数（3=FST/DT/NST 或 8=场景分类）。
        mask_domain: 掩码域，``"mag"``（默认，DBG-Mag：实值 sigmoid 掩码 + 保留
            麦克风相位）或 ``"com"``（DBG-Com：2 通道 RI 输入 + tanh 复数比值掩码）。
        eps: 数值稳定小常数。
    """

    aec_input_capabilities = frozenset({"mic_spectra", "farend_spectra"})

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        sample_rate: int = 16000,
        erb_low_bins: int = 65,
        erb_bands: int = 64,
        mic_channels: tuple | list = (32, 64, 96, 128),
        far_channels: tuple | list = (16, 32, 48, 64),
        fgru_hidden: int | None = None,
        tgru_hidden: int | None = None,
        state_feat_dim: int = 64,
        state_hidden_dim: int = 128,
        use_state_detector: bool = True,
        state_nclass: int = 3,
        mask_domain: str = "mag",
        eps: float = 1e-12,
        **kwargs,
    ):
        super().__init__()
        kwargs.pop("center", None)
        kwargs.pop("window", None)
        if kwargs:
            raise TypeError(f"DBGNet received unexpected kwargs: {list(kwargs.keys())}")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.use_state_detector = use_state_detector
        self.state_nclass = int(state_nclass)
        self.mask_domain = _validate_mask_domain(mask_domain)
        # mag：1 通道幅度进、1 通道实掩码出；com：2 通道 RI 进、2 通道复数掩码出。
        # mag 下所有模块的构造参数与历史 checkpoint 逐键一致。
        in_ch = 1 if self.mask_domain == "mag" else 2
        out_ch = in_ch
        if isinstance(eps, bool) or not isinstance(eps, Real):
            raise TypeError(f"eps must be a finite positive number, got {eps!r}")
        self.eps = float(eps)
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError(f"eps must be a finite positive number, got {eps!r}")
        # topology marker 使 strict loader 拒绝不兼容 checkpoint。
        self.register_buffer("_recursive_topology_v1", torch.empty(0))

        mc = [int(c) for c in mic_channels]
        fc = [int(c) for c in far_channels]
        if len(mc) != 4 or len(fc) != 4:
            raise ValueError(f"mic_channels and far_channels must have 4 elements, got {len(mc)} and {len(fc)}")

        # 默认 F-GRU 每方向宽度 = 拼接输入宽度（论文 Table I：Concatenation 192 →
        # F-GRU 每方向 192、双向输出 384）。历史 vkeep 参考实现用 (mic[-1]+far[-1])//2，
        # 依赖该旧推导的配置必须显式写出旧值以保持 checkpoint 兼容。
        _fgru_hidden = fgru_hidden if fgru_hidden is not None else mc[3] + fc[3]
        _tgru_hidden = tgru_hidden if tgru_hidden is not None else mc[3]
        # The tgru bottleneck output feeds deconv4, whose skip fusion concatenates
        # the ``gy4`` mic skip (mc[3] channels). ``_DecoderBlock`` therefore expects
        # exactly ``mic_channels[-1]`` input channels, so the tgru hidden size must
        # match. Reject non-default values that would break this to fail fast with a
        # clear error instead of an opaque downstream shape mismatch.
        if _tgru_hidden != mc[3]:
            raise ValueError(
                f"tgru_hidden={_tgru_hidden} is incompatible with the decoder: it must "
                f"equal mic_channels[-1]={mc[3]} because the tgru bottleneck output "
                f"feeds deconv4, whose skip fusion assumes mic_channels[-1] channels. "
                f"Leave tgru_hidden as None (default) or set it to {mc[3]}."
            )
        _fgru_input = mc[3] + fc[3]

        high_freq = sample_rate // 2
        self.erb = _ERB(
            low_bins=erb_low_bins,
            erb_bands=erb_bands,
            n_fft=n_fft,
            high_freq=high_freq,
            sample_rate=sample_rate,
        )

        f0 = erb_low_bins + erb_bands
        f1 = self._conv_out_size(f0, kernel=5, stride=1, padding=2)
        f2 = self._conv_out_size(f1, kernel=3, stride=2, padding=0)
        f3 = self._conv_out_size(f2, kernel=3, stride=2, padding=0)
        f4 = self._conv_out_size(f3, kernel=3, stride=2, padding=0)
        self._freq_sizes = (f0, f1, f2, f3, f4)

        if f4 <= 0:
            raise ValueError(
                f"DBGNet encoder collapsed for n_fft={n_fft}, "
                f"erb_low_bins={erb_low_bins}, erb_bands={erb_bands}: freq sizes={self._freq_sizes}"
            )

        # FNP/FPB：com 下按 2 通道 RI 投影；mag 下与历史实现逐键一致。
        self.fpb = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU(),
        )

        # 麦克风编码器
        self.conv_mic1 = nn.Sequential(
            nn.Conv2d(in_ch, mc[0], kernel_size=(5, 2), stride=(1, 1), padding=(2, 0)),
            nn.BatchNorm2d(mc[0]),
            nn.PReLU(),
        )
        self.conv_mic2 = nn.Sequential(
            nn.Conv2d(mc[0], mc[1], kernel_size=(3, 2), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(mc[1]),
            nn.PReLU(),
        )
        self.conv_mic3 = nn.Sequential(
            nn.Conv2d(mc[1], mc[2], kernel_size=(3, 2), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(mc[2]),
            nn.PReLU(),
        )
        self.conv_mic4 = nn.Sequential(
            nn.Conv2d(mc[2], mc[3], kernel_size=(3, 2), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(mc[3]),
            nn.PReLU(),
        )

        # 远端编码器 + FTG
        self.conv_far1 = nn.Sequential(
            nn.Conv2d(in_ch, fc[0], kernel_size=(5, 2), stride=(1, 1), padding=(2, 0)),
            nn.BatchNorm2d(fc[0]),
            nn.PReLU(),
        )
        self.ftg1 = _FTGBlock(f1)

        self.conv_far2 = nn.Sequential(
            nn.Conv2d(fc[0], fc[1], kernel_size=(3, 2), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(fc[1]),
            nn.PReLU(),
        )
        self.ftg2 = _FTGBlock(f2)

        self.conv_far3 = nn.Sequential(
            nn.Conv2d(fc[1], fc[2], kernel_size=(3, 2), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(fc[2]),
            nn.PReLU(),
        )
        self.ftg3 = _FTGBlock(f3)

        self.conv_far4 = nn.Sequential(
            nn.Conv2d(fc[2], fc[3], kernel_size=(3, 2), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(fc[3]),
            nn.PReLU(),
        )
        self.ftg4 = _FTGBlock(f4)

        # 双分支引导模块：每层 gy/gx 递归馈入下一层编码器
        self.slm1 = _SLMBlock(mc[0], fc[0])
        self.slm2 = _SLMBlock(mc[1], fc[1])
        self.slm3 = _SLMBlock(mc[2], fc[2])
        self.slm4 = _SLMBlock(mc[3], fc[3])

        # 解码器
        self.deconv4 = _DecoderBlock(mc[3], mc[2], kernel_size=(3, 2), stride=(2, 1), padding=(0, 0))
        self.deconv3 = _DecoderBlock(
            mc[2], mc[1], kernel_size=(3, 2), stride=(2, 1), padding=(0, 0), output_padding=(1, 0)
        )
        self.deconv2 = _DecoderBlock(mc[1], mc[0], kernel_size=(3, 2), stride=(2, 1), padding=(0, 0))
        self.deconv1 = _DecoderBlock(mc[0], 1, kernel_size=(1, 2), stride=(1, 1), padding=(0, 0))
        self.out_conv = nn.Conv2d(1, out_ch, kernel_size=1)
        self._validate_decoder_geometry()

        # 瓶颈 F-T GRU
        self.fgru = nn.GRU(input_size=_fgru_input, hidden_size=_fgru_hidden, bidirectional=True, batch_first=True)
        _tgru_input = _fgru_hidden * 2
        self.tgru = nn.GRU(input_size=_tgru_input, hidden_size=_tgru_hidden, bidirectional=False, batch_first=True)

        # 可选帧级状态检测器（封装多分辨率对齐 + 分类）
        if self.use_state_detector:
            self.state_head = AECStateDetectorHead(
                feature_freq_sizes=[f1, f2, f3, f4],
                projection_size=state_feat_dim,
                hidden_size=state_hidden_dim,
                nclass=self.state_nclass,
                spatial_layout="F_T",
            )

    @staticmethod
    def _conv_out_size(n: int, kernel: int, stride: int, padding: int) -> int:
        return (n + 2 * padding - kernel) // stride + 1

    @staticmethod
    def _deconv_out_size(n: int, conv: nn.ConvTranspose2d, axis: int) -> int:
        return (
            (n - 1) * conv.stride[axis]
            - 2 * conv.padding[axis]
            + conv.dilation[axis] * (conv.kernel_size[axis] - 1)
            + conv.output_padding[axis]
            + 1
        )

    def _validate_decoder_geometry(self) -> None:
        """按实际反卷积参数校验 encoder/decoder 镜像几何。"""
        sizes = self._freq_sizes
        decoder_specs = (
            ("deconv4", self.deconv4, sizes[4], sizes[3]),
            ("deconv3", self.deconv3, sizes[3], sizes[2]),
            ("deconv2", self.deconv2, sizes[2], sizes[1]),
            ("deconv1", self.deconv1, sizes[1], sizes[0]),
        )
        for name, block, input_freq, expected_freq in decoder_specs:
            conv = block.deconv[0]
            actual_freq = self._deconv_out_size(input_freq, conv, axis=0)
            if actual_freq != expected_freq:
                raise ValueError(
                    f"{name} frequency geometry is not the encoder mirror: "
                    f"{input_freq} -> {actual_freq}, expected {expected_freq}"
                )
            actual_time = self._deconv_out_size(1, conv, axis=1)
            if actual_time != 2:
                raise ValueError(
                    f"{name} time geometry must produce 2 frames before causal crop, "
                    f"got {actual_time}"
                )

    def _validate_spectra_inputs(self, mic_spec: Tensor, far_spec: Tensor) -> None:
        """统一校验离线 RI 复谱输入合同。"""
        # ONNX trace 会固化 Python shape 比较；eager 路径保持 fail-fast。
        is_legacy_onnx_export = getattr(
            torch.onnx, "is_in_onnx_export", lambda: False
        )()
        for name, spec in (("mic_spec", mic_spec), ("far_spec", far_spec)):
            if not isinstance(spec, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(spec).__name__}")
            if not is_legacy_onnx_export:
                if spec.ndim != 4:
                    raise ValueError(
                        f"{name} must have rank 4 (B, F, T, 2), got shape {tuple(spec.shape)}"
                    )
                if spec.shape[-1] != 2:
                    raise ValueError(
                        f"{name} RI dimension must be 2, got shape {tuple(spec.shape)}"
                    )
                expected_freq = self.n_fft // 2 + 1
                if spec.shape[1] != expected_freq:
                    raise ValueError(
                        f"{name} frequency dimension must be {expected_freq} for n_fft={self.n_fft}, "
                        f"got {spec.shape[1]}"
                    )
                if spec.shape[0] <= 0 or spec.shape[2] <= 0:
                    raise ValueError(
                        f"{name} batch and time dimensions must be positive, got "
                        f"B={spec.shape[0]}, T={spec.shape[2]}"
                    )
            if not spec.is_floating_point():
                raise TypeError(f"{name} must use a floating-point RI dtype, got {spec.dtype}")

        if not is_legacy_onnx_export and mic_spec.shape != far_spec.shape:
            raise ValueError(
                "mic_spec and far_spec must have identical (B, F, T, 2) shapes, "
                f"got {tuple(mic_spec.shape)} and {tuple(far_spec.shape)}"
            )
        if mic_spec.dtype != far_spec.dtype:
            raise TypeError(
                f"mic_spec and far_spec must have the same dtype, got "
                f"{mic_spec.dtype} and {far_spec.dtype}"
            )
        if mic_spec.device != far_spec.device:
            raise ValueError(
                f"mic_spec and far_spec must be on the same device, got "
                f"{mic_spec.device} and {far_spec.device}"
            )

        parameter = next(self.parameters())
        if mic_spec.dtype != parameter.dtype:
            raise TypeError(
                f"spectra dtype {mic_spec.dtype} must match model dtype {parameter.dtype}"
            )
        if mic_spec.device != parameter.device:
            raise ValueError(
                f"spectra device {mic_spec.device} must match model device {parameter.device}"
            )

    def _magnitude(self, spec: Tensor) -> Tensor:
        """计算 RI 复谱幅度；低精度输入仅在此处提升到 FP32。"""
        work = spec.float() if spec.dtype in (torch.float16, torch.bfloat16) else spec
        magnitude = torch.sqrt(work[..., 0].square() + work[..., 1].square() + self.eps)
        if work is spec:
            return magnitude
        # BF16 的有限最大值在 FP32 中平方也会溢出；先将 +inf/过大幅度
        # 截到目标低精度 dtype 的最大有限值，再转换，避免 cast 重新产生 inf。
        magnitude = magnitude.clamp(max=torch.finfo(spec.dtype).max)
        return magnitude.to(dtype=spec.dtype)

    def _erb_bm_ri(self, spec: Tensor) -> Tensor:
        """DBG-Com 输入：实/虚分量分别 ERB band-merge，返回 ``(B, 2, F_erb, T)``。"""
        return torch.stack(
            [self.erb.bm(spec[..., 0]), self.erb.bm(spec[..., 1])], dim=1
        )

    def _erb_bs_ri(self, y: Tensor) -> Tensor:
        """DBG-Com 输出：``(B, 2, F_erb, T)`` 实/虚分量分别 band-split 成 ``(B, F, T, 2)``。"""
        parts = [self.erb.bs(y[:, index]) for index in range(y.shape[1])]
        return torch.stack(parts, dim=-1)

    @staticmethod
    def _validate_enhanced_output(enhanced: Tensor) -> None:
        """Python eager 输出边界拒绝非有限增强谱；图 runtime 负责同类门禁。"""
        compiler = getattr(torch, "compiler", None)
        is_graph_capture = (
            torch.jit.is_scripting()
            or torch.jit.is_tracing()
            or getattr(compiler, "is_compiling", lambda: False)()
            or getattr(torch.onnx, "is_in_onnx_export", lambda: False)()
        )
        if not is_graph_capture and not torch.isfinite(enhanced).all().item():
            raise FloatingPointError(
                "DBGNet produced a non-finite enhanced spectrogram; "
                "input magnitude or low-precision activations exceeded the model's numerical range"
            )

    def forward_spectra(self, mic_spec: Tensor, far_spec: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """DBGNet spectrum-to-spectrum forward.

        Args:
            mic_spec: ``(B, F, T, 2)`` complex microphone spectrogram.
            far_spec: ``(B, F, T, 2)`` complex far-end spectrogram.

        Returns:
            Enhanced complex spectrogram ``(B, F, T, 2)``.
            If ``use_state_detector`` is True, also returns the frame-level state
            detector logits ``(B, T, state_nclass)`` for ``StateCrossEntropyLoss``.
        """
        self._validate_spectra_inputs(mic_spec, far_spec)
        x_ref = mic_spec
        b = mic_spec.shape[0]

        if self.mask_domain == "mag":
            # Magnitude + ERB band-merge → (B, 1, F_erb, T)。
            x_mic = self.erb.bm(self._magnitude(mic_spec)).unsqueeze(1)
            x_far = self.erb.bm(self._magnitude(far_spec)).unsqueeze(1)
        else:
            # DBG-Com：实/虚分量分别 band-merge，直接得到 (B, 2, F_erb, T)。
            x_mic = self._erb_bm_ri(mic_spec)
            x_far = self._erb_bm_ri(far_spec)

        x_far = self.fpb(x_far)

        # 逐层执行双分支编码、far-end tracking 和 SLM；每层 guidance
        # 作为下一层编码器输入，使离线前向遵循 encoder_i -> SLM_i -> encoder_(i+1)。
        dy1 = self.conv_mic1(_causal_pad_t(x_mic, 1))
        dx1 = self.ftg1(self.conv_far1(_causal_pad_t(x_far, 1)))
        gy1, gx1, mg1 = self.slm1(dy1, dx1)

        dy2 = self.conv_mic2(_causal_pad_t(gy1, 1))
        dx2 = self.ftg2(self.conv_far2(_causal_pad_t(gx1, 1)))
        gy2, gx2, mg2 = self.slm2(dy2, dx2)

        dy3 = self.conv_mic3(_causal_pad_t(gy2, 1))
        dx3 = self.ftg3(self.conv_far3(_causal_pad_t(gx2, 1)))
        gy3, gx3, mg3 = self.slm3(dy3, dx3)

        dy4 = self.conv_mic4(_causal_pad_t(gy3, 1))
        dx4 = self.ftg4(self.conv_far4(_causal_pad_t(gx3, 1)))
        gy4, gx4, mg4 = self.slm4(dy4, dx4)

        # 多分辨率状态检测：各层频率分辨率不同，时间轴均保持 STFT 帧率。
        if self.use_state_detector:
            ps = self.state_head([(mg1, gx1), (mg2, gx2), (mg3, gx3), (mg4, gx4)])

        # Bottleneck F-T GRU.
        y = torch.cat([gy4, gx4], dim=1)  # (B, 192, F4, T)
        y = y.permute(0, 3, 2, 1).flatten(0, 1)  # (B*T, F4, 192)
        y, _ = self.fgru(y)
        y = y.view(b, -1, *y.shape[1:])  # (B, T, F4, 192)
        y = y.permute(0, 2, 1, 3).flatten(0, 1)  # (B*F4, T, 192)
        y, _ = self.tgru(y)
        y = y.view(b, -1, *y.shape[1:]).permute(0, 3, 1, 2)  # (B, 128, F4, T)

        # Decoder with causal time-dimension trimming.
        y = self.deconv4(y, gy4)[..., :-1]
        y = self.deconv3(y, gy3)[..., :-1]
        y = self.deconv2(y, gy2)[..., :-1]
        y = self.deconv1(y, gy1)[..., :-1]
        y = self.out_conv(y)  # (B, 1, F_erb, T)

        # ERB band-split + mask.
        if self.mask_domain == "mag":
            m = torch.sigmoid(self.erb.bs(y.squeeze(1)))  # (B, F, T)
            enhanced = x_ref * m[..., None]  # (B, F, T, 2)
        else:
            mask = torch.tanh(self._erb_bs_ri(y))  # (B, F, T, 2)
            enhanced = _apply_complex_mask(x_ref, mask)
        self._validate_enhanced_output(enhanced)

        if self.use_state_detector:
            return enhanced, ps
        return enhanced

    def forward(self, batch: dict) -> dict:
        """Dict-based forward compatible with AECTrainer.

        Required keys: ``mic_spectra``, ``farend_spectra`` of shape ``(B, F, T, 2)``.
        Sets key: ``nearend_estimate_spectra`` of the same shape.
        If ``use_state_detector`` is True, also sets ``state_probs`` of shape
        ``(B, T, state_nclass)`` with class logits for ``StateCrossEntropyLoss``.
        """
        if "mic_spectra" not in batch or "farend_spectra" not in batch:
            raise KeyError("batch must contain mic_spectra and farend_spectra (B, F, T, 2)")

        out = self.forward_spectra(batch["mic_spectra"], batch["farend_spectra"])
        if self.use_state_detector:
            batch["nearend_estimate_spectra"] = out[0]
            batch["state_probs"] = out[1]
        else:
            batch["nearend_estimate_spectra"] = out
        return batch


if __name__ == "__main__":
    import os as _os
    import sys as _sys
    import time

    import torch.nn as nn
    from core.utils.model_profile import _count_params, _fmt_num, make_profile_parser

    # 仅直接运行本文件时需要仓库根在 sys.path；模块导入不得有 sys.path 副作用。
    _sys.path.insert(
        0,
        _os.path.dirname(
            _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
        ),
    )

    args = make_profile_parser(default_sr=16000).parse_args()

    model = DBGNet(use_state_detector=False).to(args.device).eval()
    total_p, train_p = _count_params(model)

    n_fft = model.n_fft
    hop = model.hop_length
    f_bins = n_fft // 2 + 1
    n_frames = max(2, int(args.sr * args.dur / hop))
    mic = torch.randn(args.batch, f_bins, n_frames, 2, device=args.device)
    far = torch.randn(args.batch, f_bins, n_frames, 2, device=args.device)

    class _SpecThopWrap(nn.Module):
        def __init__(self, inner: DBGNet):
            super().__init__()
            self.inner = inner

        def forward(self, mic_s: torch.Tensor, far_s: torch.Tensor) -> torch.Tensor:
            b = {"mic_spectra": mic_s, "farend_spectra": far_s}
            return self.inner(b)["nearend_estimate_spectra"]

    wrap = _SpecThopWrap(model)
    macs_str = "N/A"
    try:
        from thop import profile as thop_profile

        macs, _ = thop_profile(wrap, inputs=(mic, far), verbose=False)
        macs_str = _fmt_num(float(macs), "MACs")
    except Exception as e:
        macs_str = f"N/A ({e})"

    with torch.inference_mode():
        t0 = time.perf_counter()
        for _ in range(max(1, args.repeats)):
            _ = model({"mic_spectra": mic, "farend_spectra": far})
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / max(1, args.repeats)

    print(f"=== DBGNet (spectrum batch B={args.batch}, F={f_bins}, T={n_frames}) ===")
    print(f"  Params   : {total_p / 1e6:.3f} M (trainable {train_p / 1e6:.3f} M)")
    print(f"  MACs     : {macs_str} (thop on forward(batch) spectrum path)")
    print(f"  Latency  : {dt * 1000:.2f} ms / forward (mean of {args.repeats}, device={args.device})")
