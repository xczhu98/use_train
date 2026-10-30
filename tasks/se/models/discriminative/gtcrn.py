"""GTCRN: Group Temporal Convolutional Recurrent Network (实时语音增强，超轻量).

论文: "GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources",
      Xiaobin Rong, Tianchi Sun, Xu Zhang, Yuxiang Hu, Changbao Zhang, Jing Lu,
      IEEE ICASSP 2024. 官方实现: https://github.com/Xiaobin-Rong/gtcrn

与本仓库 ``tasks.se.models.discriminative.ulunas`` (UL-UNAS, arXiv:2503.00340) 同源作者体系；二者在 ERB、
分组双路径 RNN 瓶颈等设计上延续同一脉络，但编码器块、注意力与掩膜形式不同——见各分区注释。

模块分区（速查）:
  - 与 UL-UNAS 对齐/同构思路: ``ERB``, ``GRNN``, ``DPGRNN``（实现细节见类 docstring）
  - 仅 GTCRN: ``SFE``, ``TRA``, ``ConvBlock``, ``GTConvBlock``, ``_Mask``（复数掩膜）

规模（与 gtcrn-main 一致量级; 详见 docs/architecture/SE_Architecture.md、tools/model_tools/audit_metrics.py）: ~23.7K 可训练参数, ~48K 总参数（含固定 ERB）, ~33 MMACs/s。

谱域接口:
  输入 noisy_spec (B, F, T, 2)；STFT/iSTFT 由训练框架完成。流式/ONNX 见 ``onnx/streaming_gtcrn.py``。

脚本: python tasks/se/models/gtcrn.py [--sr 16000] [--dur 1.0] [--device cpu]
"""

# When run directly, add project root to sys.path
if __name__ == "__main__":
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

from typing import Dict, List, Optional, Tuple

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

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
# GRNN / DPGRNN 紧随「GTCRN-only」卷积块之后（本文件中部）。与 ``ulunas`` 同脉络；ERB 两处请保持同步。
# =============================================================================


class ERB(nn.Module):
    """Equivalent Rectangular Bandwidth：低频保持分辨率，高频经固定滤波器组压缩。

    ``bm`` / ``bs`` 与 ``ulunas.ERB`` 同构（Band Merge / Band Split）。修改一处时请同步另一文件。

    Args:
        erb_subband_1: 保持原 FFT bin 数的低频个数。
        erb_subband_2: 高频经 ERB 压缩后的子带数。
    """
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self._erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs - erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs - erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    @staticmethod
    def _hz2erb(freq_hz):
        return 21.4 * np.log10(0.00437 * freq_hz + 1)

    @staticmethod
    def _erb2hz(erb_f):
        return (10 ** (erb_f / 21.4) - 1) / 0.00437

    def _erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft, high_lim, fs):
        low_lim = erb_subband_1 / nfft * fs
        erb_low, erb_high = self._hz2erb(low_lim), self._hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self._erb2hz(erb_points) / fs * nfft).astype(np.int32)
        f = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)
        f[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2 - 2):
            f[i+1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12) / (bins[i+1] - bins[i] + 1e-12)
            f[i+1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i+2]) + 1e-12) / (bins[i+2] - bins[i+1] + 1e-12)
        f[-1, bins[-2]:bins[-1]+1] = 1 - f[-2, bins[-2]:bins[-1]+1]
        f = f[:, erb_subband_1:]
        return torch.from_numpy(np.abs(f))

    def bm(self, x):
        """Band Merge: 频域 -> ERB（低频保留 + 高频线性压缩）。"""
        return torch.cat([x[..., :self.erb_subband_1], self.erb_fc(x[..., self.erb_subband_1:])], dim=-1)

    def bs(self, x_erb):
        """Band Split: ERB -> 频域（低频保留 + 高频展开）。"""
        return torch.cat([x_erb[..., :self.erb_subband_1], self.ierb_fc(x_erb[..., self.erb_subband_1:])], dim=-1)


# =============================================================================
# GTCRN-only — 编码器/解码器卷积与掩膜（UL-UNAS 使用 XConv/XDWS/XMB + IRM，不经过本节模块）
# =============================================================================


class SFE(nn.Module):
    """Shallow Feature Extraction.

    Uses Unfold for local feature extraction along freq dim (similar to ShuffleNetV2 Block),
    expands each freq bin neighborhood (kernel_size values) into channel dim.
    Input (B, C, T, F) -> Output (B, C*kernel_size, T, F).
    """
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, (kernel_size-1)//2))

    def forward(self, x):
        return self.unfold(x).reshape(x.shape[0], x.shape[1]*self.kernel_size, x.shape[2], x.shape[3])


class TRA(nn.Module):
    """Temporal Recurrent Attention.

    Channel attention weights along time axis:
    1. Energy mean over freq dim -> (B, C, T)
    2. GRU models temporal dependency -> FC + Sigmoid for attention weights
    3. Per-frame per-channel weighted features
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        zt = torch.mean(x.pow(2), dim=-1)  #  (B, C, T) freq-dim energy
        at = self.att_act(self.att_fc(self.att_gru(zt.transpose(1, 2))[0])).transpose(1, 2).unsqueeze(-1)  # (B, C, T, 1)
        return x * at


class ConvBlock(nn.Module):
    """Standard Conv2d / ConvTranspose2d + BatchNorm + activation (PReLU / Tanh)."""
    def __init__(self, in_ch, out_ch, kernel, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        C = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = C(in_ch, out_ch, kernel, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.Tanh() if is_last else nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """Grouped Temporal Conv Block - channel grouping + SFE + depthwise separable conv + TRA + channel shuffle."""
    def __init__(self, in_ch, hid, kernel, stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.pad_size = (kernel[0]-1) * dilation[0]
        self.use_deconv = use_deconv
        C = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.sfe = SFE(3, 1)
        self.pc1 = C(in_ch//2*3, hid, 1); self.bn1 = nn.BatchNorm2d(hid); self.act1 = nn.PReLU()
        self.dc = C(hid, hid, kernel, stride=stride, padding=padding, dilation=dilation, groups=hid)
        self.dbn = nn.BatchNorm2d(hid); self.dact = nn.PReLU()
        self.pc2 = C(hid, in_ch//2, 1); self.bn2 = nn.BatchNorm2d(in_ch//2)
        self.tra = TRA(in_ch//2)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        h = self.act1(self.bn1(self.pc1(self.sfe(x1))))
        h = nn.functional.pad(h, [0, 0, self.pad_size, 0])
        h = self.dact(self.dbn(self.dc(h)))
        h = self.tra(self.bn2(self.pc2(h)))
        s = torch.stack([h, x2], dim=1).transpose(1, 2).contiguous()
        return rearrange(s, 'b c g t f -> b (c g) t f')


# =============================================================================
# SHARED LINEAGE (2/2) — GRNN / DPGRNN（G-DPRNN 瓶颈；``ulunas`` 中同名类同构）
# =============================================================================


class GRNN(nn.Module):
    """Grouped GRU：通道维一分为二，两组独立 GRU 再拼接，降低参数量。

    UL-UNAS 中 G-DPRNN 的 intra/inter 子路径使用同一分组思想；参见 ``ulunas.GRNN``。
    """
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        if h is None:
            d = 2 if self.bidirectional else 1
            h = torch.zeros(self.num_layers*d, x.shape[0], self.hidden_size,
                            device=x.device, dtype=x.dtype)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        h1, h2 = torch.chunk(h, 2, dim=-1)
        y1, h1 = self.rnn1(x1, h1.contiguous())
        y2, h2 = self.rnn2(x2, h2.contiguous())
        return torch.cat([y1, y2], dim=-1), torch.cat([h1, h2], dim=-1)


class DPGRNN(nn.Module):
    """Dual-Path Grouped RNN（G-DPRNN）：频向 intra（双向）+ 时向 inter（因果）。

    与 UL-UNAS 论文中瓶颈同名；本实现 residual 投影为 ``Linear(H, H)`` 与 ``LayerNorm(..., H)``。
    ``ulunas.DPGRNN`` 为论文写法：``Linear(H, C)`` 与 ``LayerNorm(..., C)``（C 为通道宽）；
    在 GTCRN 默认 C=H=16 时张量形状一致，公式不完全相同，对比时请分文件阅读。
    """
    def __init__(self, input_size, width, hidden_size):
        super().__init__()
        self.width = width
        self.hidden_size = hidden_size
        self.intra_rnn = GRNN(input_size, hidden_size//2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)
        self.inter_rnn = GRNN(input_size, hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, T, F, C = x.shape
        intra_x = self.intra_fc(self.intra_rnn(x.reshape(B*T, F, C))[0])
        intra_x = self.intra_ln(intra_x.reshape(B, T, self.width, self.hidden_size))
        intra_out = x + intra_x
        x2 = intra_out.permute(0, 2, 1, 3)
        inter_x = self.inter_fc(self.inter_rnn(x2.reshape(B*F, T, C))[0])
        inter_x = inter_x.reshape(B, self.width, T, self.hidden_size).permute(0, 2, 1, 3)
        return (intra_out + self.inter_ln(inter_x)).permute(0, 3, 1, 2)


# --- GTCRN 固定 U-Net 编码器/解码器（非 NAS；与 ``ulunas._Encoder`` 不同）----------------


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvBlock(9, 16, (1,5), (1,2), (0,2)),
            ConvBlock(16, 16, (1,5), (1,2), (0,2), groups=2),
            GTConvBlock(16, 16, (3,3), (1,1), (0,1), (1,1)),
            GTConvBlock(16, 16, (3,3), (1,1), (0,1), (2,1)),
            GTConvBlock(16, 16, (3,3), (1,1), (0,1), (5,1)),
        ])

    def forward(self, x):
        skips = []
        for layer in self.layers:
            x = layer(x); skips.append(x)
        return x, skips


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            GTConvBlock(16, 16, (3,3), (1,1), (10,1), (5,1), use_deconv=True),
            GTConvBlock(16, 16, (3,3), (1,1), (4,1), (2,1), use_deconv=True),
            GTConvBlock(16, 16, (3,3), (1,1), (2,1), (1,1), use_deconv=True),
            ConvBlock(16, 16, (1,5), (1,2), (0,2), groups=2, use_deconv=True),
            ConvBlock(16, 2, (1,5), (1,2), (0,2), use_deconv=True, is_last=True),
        ])

    def forward(self, x, skips):
        n = len(self.layers)
        for i in range(n):
            x = self.layers[i](x + skips[n-1-i])
        return x


class _Mask(nn.Module):
    def forward(self, mask, spec):
        sr = spec[:,0]*mask[:,0] - spec[:,1]*mask[:,1]
        si = spec[:,1]*mask[:,0] + spec[:,0]*mask[:,1]
        return torch.stack([sr, si], dim=1)


@MODEL_REGISTRY.register("GTCRN")
class GTCRN(BaseSEModel):
    """Spectrum-in / spectrum-out GTCRN. No STFT/iSTFT inside; trainer does wav<->spec.

    Input:  noisy_spec (B, F, T, 2) from framework STFT.
    Output: enhanced_spec (B, F, T, 2); framework iSTFT gives waveform.

    Freq dimension (freq_bins = n_fft//2+1) and STFT/window come from YAML ``stft`` block;
    no STFT/window parameters in constructor.
    For ONNX export and streaming deployment, use StreamingGTCRN from onnx/streaming_gtcrn.py.
    """

    input_domain = "spectrum"
    output_domain = "spectrum"
    forward_kind = "estimate"
    frontend_owner = "runtime"

    def __init__(self, target_fs: int = 16000, **kwargs):
        super().__init__()
        n_fft = kwargs.pop("n_fft", 512)
        for key in ("hop_length", "win_length", "window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"GTCRN got unexpected keyword arguments: {sorted(kwargs)}")
        if target_fs != 16000:
            raise ValueError(f"GTCRN requires target_fs=16000, got {target_fs}")
        self.n_fft = n_fft
        self.target_fs = target_fs
        # ERB 滤波器组固定为 fs=16000 / high_lim=8000；其他采样率会静默产生错误子带划分。
        _warn_erb_16k_once("GTCRN")
        self.erb = ERB(65, 64, nfft=n_fft)
        self.sfe = SFE(3, 1)
        self.encoder = _Encoder()
        self.dpgrnn1 = DPGRNN(16, 33, 16)
        self.dpgrnn2 = DPGRNN(16, 33, 16)
        self.decoder = _Decoder()
        self.mask = _Mask()

    @staticmethod
    def _spec_to_feat(spec: torch.Tensor):
        """(B, F, T, 2) -> feat (B, 3, T, F), spec_2ch (B, 2, T, F)."""
        sr = spec[..., 0].permute(0, 2, 1)
        si = spec[..., 1].permute(0, 2, 1)
        mag = torch.sqrt(sr ** 2 + si ** 2 + 1e-12)
        feat = torch.stack([mag, sr, si], dim=1)
        spec_2ch = spec.permute(0, 3, 2, 1)
        return feat, spec_2ch

    @fp32_forward
    def forward(self, noisy_spec: torch.Tensor, lengths=None, *, fs: int | None = None):
        """noisy_spec: (B, F, T, 2). Returns enhanced_spec (B, F, T, 2)."""
        if fs is not None and fs != self.target_fs:
            raise ValueError(f"GTCRN requires fs={self.target_fs}, got {fs}")
        if noisy_spec.dim() != 4 or noisy_spec.shape[-1] != 2:
            raise ValueError(f"Expected [B, F, T, 2], got {tuple(noisy_spec.shape)}")
        if noisy_spec.shape[1] != self.n_fft // 2 + 1:
            raise ValueError(f"Expected F={self.n_fft // 2 + 1}, got F={noisy_spec.shape[1]}")
        if noisy_spec.shape[0] == 0 or noisy_spec.shape[2] == 0:
            raise ValueError("GTCRN requires non-empty batch and time dimensions")
        if not noisy_spec.is_floating_point():
            raise TypeError("GTCRN noisy_spec must use a floating-point dtype")
        if lengths is not None and (
            lengths.dim() != 1 or lengths.shape[0] != noisy_spec.shape[0]
        ):
            raise ValueError(
                f"lengths must be a 1-D tensor of batch size {noisy_spec.shape[0]}, "
                f"got shape {tuple(lengths.shape)}"
            )
        feat, spec_2ch = self._spec_to_feat(noisy_spec.float())
        feat = self.sfe(self.erb.bm(feat))
        feat, skips = self.encoder(feat)
        feat = self.dpgrnn2(self.dpgrnn1(feat))
        m = self.erb.bs(self.decoder(feat, skips))
        enh = self.mask(m, spec_2ch).permute(0, 3, 2, 1)
        return enh


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 验证本模型参数量与算力。修改下面配置后运行: python tasks/se/models/gtcrn.py [--sr 16000] [--dur 1.0] [--device cpu]
    # -------------------------------------------------------------------------
    from core.utils.model_profile import profile_model, make_profile_parser
    from core.stft import build_stft_istft

    args = make_profile_parser(default_sr=16000).parse_args()

    # 完整配置：STFT、采样率、模型参数（与 YAML 对齐，便于对比）
    STFT_CFG = {
        "n_fft": 512,
        "hop_length": 256,
        "win_length": 512,
        "window": "hann",
    }
    SAMPLE_RATE = args.sr
    MODEL_ARGS = {
        "n_fft": STFT_CFG["n_fft"],
    }

    model = GTCRN(**MODEL_ARGS)
    stft_module = build_stft_istft(STFT_CFG)

    class _W(nn.Module):
        def __init__(self, m, stft_m):
            super().__init__()
            self._m, self._stft = m, stft_m
        def forward(self, wav):
            spec = self._stft(wav)
            return self._stft.inverse(self._m(spec), wav.shape[-1])

    wrapped = _W(model, stft_module)
    print("\n[Config] STFT:", STFT_CFG, "| sample_rate:", SAMPLE_RATE, "| model args:", MODEL_ARGS)
    profile_model(
        wrapped,
        sample_rate=SAMPLE_RATE,
        duration=args.dur,
        batch_size=args.batch,
        device=args.device,
        repeats=args.repeats,
        label="GTCRN",
    )
