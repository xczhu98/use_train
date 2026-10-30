"""CAGCRN spectrum-in / spectrum-out AEC model.

训练和推理只通过 ``forward(batch)`` 接收/输出复数谱；STFT/iSTFT 位于模型外部。
主体结构沿用公开 CAGCRN 实现的 ERB、CATA、TFGRU×3、Decoder、Mask 与 TFAG，
同时按论文的 no-lookahead 声明确保时间卷积严格因果。CATA 候选窗口由
``align_delay`` 固定配置，cross-attention 在该历史窗口内执行软对齐；本实现不保留
无法获得梯度的伪可学习 delay 参数。

输入/输出形状为 (B, F, T, 2)，F = n_fft//2+1（如 n_fft=512 时 F=257）。
本仓库保留 BatchNorm，与训练态逐帧严格等价无关；部署与验证均使用 eval 模式。
"""

from __future__ import annotations

import math
import sys as _sys
import os as _os

if __name__ == "__main__":
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.registry import MODEL_REGISTRY


CAGCRN_DELAY_D = 11
CAGCRN_ALGORITHMIC_DELAY_FRAMES = 0


# ---------------------------------------------------------------------------
# CAGCRN components
# ---------------------------------------------------------------------------


class ERB(nn.Module):
    """Equivalent Rectangular Bandwidth (ERB) filterbank. Official default: nfft=512, high_freq=8000, sample_rate=16000."""

    def __init__(self, low_freq_idx, num_erb_bands, nfft=512, high_freq=8000, sample_rate=16000):
        super().__init__()
        self.low_freq_idx = low_freq_idx
        self.num_erb_bands = num_erb_bands
        erb_filters = self._create_erb_filterbank(low_freq_idx, num_erb_bands, nfft, high_freq, sample_rate)
        num_freqs = nfft // 2 + 1
        self.analysis = nn.Linear(num_freqs - low_freq_idx, num_erb_bands, bias=False)
        self.synthesis = nn.Linear(num_erb_bands, num_freqs - low_freq_idx, bias=False)
        self.analysis.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.synthesis.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def _hz_to_erb(self, freq_hz):
        return 21.4 * torch.log10(0.00437 * freq_hz + 1)

    def _erb_to_hz(self, erb_scale):
        return (10 ** (erb_scale / 21.4) - 1) / 0.00437

    def _create_erb_filterbank(self, low_freq_idx, num_erb_bands, nfft, high_freq, sample_rate):
        low_freq = low_freq_idx / nfft * sample_rate
        erb_low = self._hz_to_erb(torch.tensor(low_freq))
        erb_high = self._hz_to_erb(torch.tensor(high_freq))
        erb_points = torch.linspace(erb_low, erb_high, num_erb_bands)
        freq_bins = torch.round(self._erb_to_hz(erb_points) / sample_rate * nfft).int()
        erb_filters = torch.zeros((num_erb_bands, nfft // 2 + 1), dtype=torch.float32)
        for i in range(num_erb_bands):
            if i == 0:
                start, end = freq_bins[i], freq_bins[i + 1]
                erb_filters[i, start:end] = (end - torch.arange(start, end)) / (end - start + 1e-8)
            elif i == num_erb_bands - 1:
                start, end = freq_bins[i - 1], freq_bins[i]
                erb_filters[i, start : end + 1] = 1 - erb_filters[i - 1, start : end + 1]
            else:
                start, mid, end = freq_bins[i - 1], freq_bins[i], freq_bins[i + 1]
                erb_filters[i, start:mid] = (torch.arange(start, mid) - start) / (mid - start + 1e-8)
                erb_filters[i, mid:end] = (end - torch.arange(mid, end)) / (end - mid + 1e-8)
        return erb_filters[:, low_freq_idx:]

    def bm(self, x):
        x_low = x[..., : self.low_freq_idx]
        x_high = self.analysis(x[..., self.low_freq_idx :])
        return torch.cat([x_low, x_high], dim=-1)

    def bs(self, x_erb):
        x_low = x_erb[..., : self.low_freq_idx]
        x_high = self.synthesis(x_erb[..., self.low_freq_idx :])
        return torch.cat([x_low, x_high], dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DGConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0] - 1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.point_conv1 = conv_module(in_channels, out_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(out_channels)
        self.point_act = nn.PReLU()
        self.depth_conv = conv_module(
            out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=out_channels
        )
        self.depth_bn = nn.BatchNorm2d(out_channels)
        self.depth_act = nn.PReLU()
        self.point_conv2 = conv_module(out_channels, out_channels, 1)
        self.point_bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = x
        x = self.point_act(self.point_bn1(self.point_conv1(x)))
        x = F.pad(x, [0, 0, self.pad_size, 0])
        x = self.depth_act(self.depth_bn(self.depth_conv(x)))
        x = self.point_bn2(self.point_conv2(x))
        return x + x1


class CATA(nn.Module):
    """Cross-Attention Temporal Alignment.

    ``delay_frames`` 表示包含当前帧在内的 farend 候选窗口大小。Cross-attention
    沿 delay/window 轴归一化；窗口是固定配置，不是可学习参数。
    """

    def __init__(self, in_channels, hidden_channels, delay_frames=CAGCRN_DELAY_D):
        super().__init__()
        if isinstance(delay_frames, bool) or not isinstance(delay_frames, int) or delay_frames < 1:
            raise ValueError(f"CATA delay_frames must be a positive integer, got {delay_frames!r}")
        self.delay_frames = delay_frames
        self.pconv_mic = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, groups=in_channels),
            nn.Conv2d(in_channels, hidden_channels, 1),
        )
        self.pconv_mic2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1, groups=hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
        )
        self.pconv_ref = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, groups=in_channels),
            nn.Conv2d(in_channels, hidden_channels, 1),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, in_channels * 2, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.PReLU(),
        )

    def sliding_window(self, x, window_size, step=1):
        B, C, T, F = x.shape
        return x.unfold(2, window_size, step).permute(0, 1, 2, 4, 3)

    def forward(self, x_mic, x_ref):
        Q_mic = self.pconv_mic(x_mic)
        K_mic = self.pconv_mic2(Q_mic)
        V_mic = x_mic

        # Compute attention scores for mic signal
        attn_scores_mic = torch.matmul(Q_mic.permute(0,2,3,1), K_mic.permute(0,2,1,3))  # (B,T,F,F)
        attn_scores_mic = F.softmax(attn_scores_mic / math.sqrt(Q_mic.shape[1]), dim=-1)
        mic_out = torch.matmul(attn_scores_mic, V_mic.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        d = self.delay_frames

        K_ref = self.pconv_ref(x_ref)
        K_ref = F.pad(K_ref, (0, 0, d - 1, 0))
        Ku = self.sliding_window(K_ref, d)
        ref_score = Q_mic.unsqueeze(-2) * Ku
        # 论文公式沿 sliding delay/window 轴归一化，而不是沿频率轴。
        ref_score = F.softmax(ref_score, dim=-2)
        V_ref = F.pad(x_ref, (0, 0, d - 1, 0))
        V_ref = self.sliding_window(V_ref, d)
        ref_out = torch.sum(V_ref * ref_score, dim=-2)
        combined = torch.cat([mic_out, ref_out], dim=1)
        out = self.fusion(combined)
        return out, mic_out


class Encoder(nn.Module):
    def __init__(self, align_delay=CAGCRN_DELAY_D):
        super().__init__()
        self.mic_convs_pre = nn.ModuleList([
            ConvBlock(9, 12, (1, 5), stride=(1, 2), padding=(0, 2), groups=1, use_deconv=False, is_last=False),
            DGConvBlock(12, 12, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), use_deconv=False),
        ])
        self.ref_convs_pre = nn.ModuleList([
            ConvBlock(9, 12, (1, 5), stride=(1, 2), padding=(0, 2), groups=1, use_deconv=False, is_last=False),
            DGConvBlock(12, 12, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), use_deconv=False),
        ])
        self.mix_convs = nn.ModuleList([
            ConvBlock(24, 24, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=False, is_last=False),
            DGConvBlock(24, 24, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), use_deconv=False),
            ConvBlock(24, 24, (1, 5), stride=(1, 1), padding=(0, 2), groups=2, use_deconv=False, is_last=False),
            DGConvBlock(24, 24, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1), use_deconv=False),
            ConvBlock(24, 24, (1, 5), stride=(1, 1), padding=(0, 2), groups=2, use_deconv=False, is_last=False),
            DGConvBlock(24, 24, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(4, 1), use_deconv=False),
        ])
        self.mic_convs = nn.ModuleList([
            ConvBlock(12, 12, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=False, is_last=False),
            DGConvBlock(12, 12, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), use_deconv=False),
            ConvBlock(12, 12, (1, 5), stride=(1, 1), padding=(0, 2), groups=2, use_deconv=False, is_last=False),
            DGConvBlock(12, 12, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1), use_deconv=False),
            ConvBlock(12, 24, (1, 5), stride=(1, 1), padding=(0, 2), groups=2, use_deconv=False, is_last=False),
            DGConvBlock(24, 24, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(4, 1), use_deconv=False),
        ])
        self.align = CATA(12, 12, delay_frames=align_delay)
        self.unfold_mic = nn.Unfold(kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.unfold_ref = nn.Unfold(kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))

    def _extract_subband_features(self, x, unfold):
        B, C, T, F = x.shape
        x_unfolded = unfold(x)
        return x_unfolded.reshape(B, C * 3, T, F)

    def forward(self, mic, ref):
        en_outs = []
        mic = self._extract_subband_features(mic, self.unfold_mic)
        ref = self._extract_subband_features(ref, self.unfold_ref)
        for i in range(len(self.mic_convs_pre)):
            mic = self.mic_convs_pre[i](mic)
            ref = self.ref_convs_pre[i](ref)
            en_outs.append(mic)
        mix, _ = self.align(mic, ref)
        for conv in self.mix_convs:
            mix = conv(mix)
            en_outs.append(mix)
        for conv in self.mic_convs:
            mic = conv(mic)
        return mix, en_outs, mic


class TFGRU(nn.Module):
    """Time-Frequency Grouped RNN aligned with official _create_grouped_gru.

    intra_gru: each path GRU(input//2, hidden//4, bidir) → output hidden//2 per path → concat hidden_size.
    intra_fc: Linear(hidden_size, hidden_size). Matches official two-level //2 in _create_grouped_gru.
    """

    def __init__(self, input_size, width, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size
        self.intra_gru = nn.ModuleDict({
            "path1": nn.GRU(input_size // 2, hidden_size // 4, batch_first=True, bidirectional=True),
            "path2": nn.GRU(input_size // 2, hidden_size // 4, batch_first=True, bidirectional=True),
        })
        # intra: two paths, each bidir → hidden//4*2=hidden//2 per path → concat hidden_size
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_norm = nn.LayerNorm((width, hidden_size), eps=1e-8)
        self.inter_gru = nn.ModuleDict({
            "path1": nn.GRU(input_size // 2, hidden_size // 2, batch_first=True, bidirectional=False),
            "path2": nn.GRU(input_size // 2, hidden_size // 2, batch_first=True, bidirectional=False),
        })
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_norm = nn.LayerNorm((width, hidden_size), eps=1e-8)

    def _grouped_gru_forward(self, x, gru_dict, h=None):
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        if h is not None:
            h1, h2 = torch.chunk(h, chunks=2, dim=-1)
            h1, h2 = h1.contiguous(), h2.contiguous()
        else:
            h1 = h2 = None
        y1, h1 = gru_dict["path1"](x1, h1)
        y2, h2 = gru_dict["path2"](x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1) if h1 is not None else None
        return y, h

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        batch_size = x.shape[0]
        intra_x = x.reshape(-1, x.shape[2], x.shape[3])
        intra_x, _ = self._grouped_gru_forward(intra_x, self.intra_gru)
        intra_x = self.intra_fc(intra_x)
        intra_x = intra_x.reshape(batch_size, -1, self.width, self.hidden_size)
        intra_x = self.intra_norm(intra_x)
        intra_out = x + intra_x
        x = intra_out.permute(0, 2, 1, 3)
        inter_x = x.reshape(-1, x.shape[2], x.shape[3])
        inter_x, _ = self._grouped_gru_forward(inter_x, self.inter_gru)
        inter_x = self.inter_fc(inter_x)
        inter_x = inter_x.reshape(batch_size, self.width, -1, self.hidden_size)
        inter_x = inter_x.permute(0, 2, 1, 3)
        inter_x = self.inter_norm(inter_x)
        inter_out = intra_out + inter_x
        return inter_out.permute(0, 3, 1, 2)


class TFAG(nn.Module):
    def __init__(self, channels):
        super().__init__()
        reduced_channels = channels // 2
        self.temporal_left_pads = (4, 8)
        self.temporal_scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, reduced_channels, (5, 1), dilation=(1, 1), groups=reduced_channels),
                nn.BatchNorm2d(reduced_channels),
                nn.PReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(channels, reduced_channels, (5, 1), dilation=(2, 1), groups=reduced_channels),
                nn.BatchNorm2d(reduced_channels),
                nn.PReLU(),
            ),
        ])
        self.freq_scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, reduced_channels, (1, 3), padding=(0, 1), dilation=(1, 1), groups=reduced_channels),
                nn.BatchNorm2d(reduced_channels),
                nn.PReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(channels, reduced_channels, (1, 3), padding=(0, 2), dilation=(1, 2), groups=reduced_channels),
                nn.BatchNorm2d(reduced_channels),
                nn.PReLU(),
            ),
        ])
        self.gate_generator = nn.Sequential(
            nn.Conv2d(reduced_channels * 4, reduced_channels, 1, groups=reduced_channels),
            nn.BatchNorm2d(reduced_channels),
            nn.PReLU(),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Conv2d(channels, channels, 1, groups=channels)

    def forward(self, x1, x2):
        mix = x1 + x2
        temp_feats = [
            conv(F.pad(mix, (0, 0, left_pad, 0)))
            for conv, left_pad in zip(self.temporal_scales, self.temporal_left_pads)
        ]
        freq_feats = [conv(mix) for conv in self.freq_scales]
        multi_scale_feats = torch.cat(temp_feats + freq_feats, dim=1)
        gate = self.gate_generator(multi_scale_feats)
        out = gate * x1 + (1 - gate) * x2
        return self.fusion(out)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            DGConvBlock(24, 24, (3, 3), stride=(1, 1), padding=(2 * 4, 1), dilation=(4, 1), use_deconv=True),
            ConvBlock(24, 24, (1, 5), stride=(1, 1), padding=(0, 2), groups=2, use_deconv=True, is_last=False),
            DGConvBlock(24, 24, (3, 3), stride=(1, 1), padding=(2 * 2, 1), dilation=(2, 1), use_deconv=True),
            ConvBlock(24, 24, (1, 5), stride=(1, 1), padding=(0, 2), groups=2, use_deconv=True, is_last=False),
            DGConvBlock(24, 24, (3, 3), stride=(1, 1), padding=(2 * 1, 1), dilation=(1, 1), use_deconv=True),
            ConvBlock(24, 12, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True, is_last=False),
            DGConvBlock(12, 12, (3, 3), stride=(1, 1), padding=(2 * 1, 1), dilation=(1, 1), use_deconv=True),
            ConvBlock(12, 2, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, is_last=True),
        ])

    def forward(self, x, en_outs):
        for i in range(len(self.de_convs)):
            x = self.de_convs[i](x + en_outs[len(self.de_convs) - 1 - i])
        return x


class _Mask(nn.Module):
    def forward(self, mask, spec):
        s_real = spec[:, 0] * mask[:, 0] - spec[:, 1] * mask[:, 1]
        s_imag = spec[:, 1] * mask[:, 0] + spec[:, 0] * mask[:, 1]
        return torch.stack([s_real, s_imag], dim=1)


class _CAGCRNNet(nn.Module):
    """Core CAGCRN network. Input/output: (B, F, T, 2)."""

    def __init__(self, nfft=512, high_freq=8000, sample_rate=16000, align_delay=CAGCRN_DELAY_D):
        super().__init__()
        self.nfft = nfft
        self.erb = ERB(65, 64, nfft=nfft, high_freq=high_freq, sample_rate=sample_rate)
        self.encoder = Encoder(align_delay=align_delay)
        self.tfgru1 = TFGRU(24, 33, 24)
        self.tfgru2 = TFGRU(24, 33, 24)
        self.tfgru3 = TFGRU(24, 33, 24)
        self.decoder = Decoder()
        self.mask = _Mask()
        self.multi_scale_gate = TFAG(24)

    def forward(self, mic_spec, ref_spec):
        for name, spec in (("mic_spec", mic_spec), ("ref_spec", ref_spec)):
            if not isinstance(spec, torch.Tensor):
                raise TypeError(f"CAGCRN {name} must be a torch.Tensor, got {type(spec).__name__}")
            if spec.ndim != 4 or spec.shape[-1] != 2:
                raise ValueError(
                    f"CAGCRN expects {name} with shape (B, F, T, 2), got {tuple(spec.shape)}"
                )
            if not spec.is_floating_point():
                raise TypeError(f"CAGCRN {name} must use a floating dtype, got {spec.dtype}")
        if mic_spec.shape != ref_spec.shape:
            raise ValueError(
                f"CAGCRN requires mic_spec and ref_spec shapes to match, got "
                f"{tuple(mic_spec.shape)} and {tuple(ref_spec.shape)}"
            )
        expected_freqs = self.nfft // 2 + 1
        if mic_spec.shape[1] != expected_freqs:
            raise ValueError(
                f"CAGCRN n_fft={self.nfft} requires F={expected_freqs}, got F={mic_spec.shape[1]}"
            )
        if mic_spec.dtype != ref_spec.dtype:
            raise TypeError(
                f"CAGCRN mic_spec and ref_spec dtypes must match, got "
                f"{mic_spec.dtype} and {ref_spec.dtype}"
            )
        if mic_spec.device != ref_spec.device:
            raise ValueError(
                f"CAGCRN mic_spec and ref_spec devices must match, got "
                f"{mic_spec.device} and {ref_spec.device}"
            )
        spec_ori = mic_spec
        mic_real = mic_spec[..., 0].permute(0, 2, 1)
        mic_imag = mic_spec[..., 1].permute(0, 2, 1)
        mic_mag = torch.sqrt(mic_real**2 + mic_imag**2 + 1e-12)
        mic_feat = torch.stack([mic_mag, mic_real, mic_imag], dim=1)
        ref_real = ref_spec[..., 0].permute(0, 2, 1)
        ref_imag = ref_spec[..., 1].permute(0, 2, 1)
        ref_mag = torch.sqrt(ref_real**2 + ref_imag**2 + 1e-12)
        ref_feat = torch.stack([ref_mag, ref_real, ref_imag], dim=1)
        mic_feat = self.erb.bm(mic_feat)
        ref_feat = self.erb.bm(ref_feat)
        feat_mix, en_outs, feat_mic = self.encoder(mic_feat, ref_feat)
        feat_mix = self.tfgru2(self.tfgru1(feat_mix))
        feat_mic = self.tfgru3(feat_mic)
        feat_mix = self.multi_scale_gate(feat_mix, feat_mic)
        m_feat = self.decoder(feat_mix, en_outs)
        m = self.erb.bs(m_feat)
        spec_enh = self.mask(m, spec_ori.permute(0, 3, 2, 1))
        spec_enh = spec_enh.permute(0, 3, 2, 1)
        return spec_enh


@MODEL_REGISTRY.register("CAGCRN")
class CAGCRN(nn.Module):
    """CAGCRN for use_train AEC: spectrum-in/spectrum-out.

    ``forward(batch)`` expects ``mic_spectra`` / ``farend_spectra`` (trainer or caller STFT)
    and writes ``nearend_estimate_spectra``. 图外 STFT 参数会完整保存，供 ONNX runtime
    metadata 与部署前端校验；网络数值仅依赖 ``n_fft`` 与 ``sample_rate``。
    """

    aec_input_capabilities = frozenset({"mic_spectra", "farend_spectra"})

    def __init__(
        self,
        n_fft=512,
        hop_length=256,
        win_length=512,
        window="hann",
        center=True,
        high_freq=None,
        sample_rate=16000,
        align_delay: int = CAGCRN_DELAY_D,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            name = next(iter(kwargs))
            raise TypeError(f"CAGCRN.__init__() got an unexpected keyword argument {name!r}")
        int_params = {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "sample_rate": sample_rate,
        }
        invalid_types = {
            name: value
            for name, value in int_params.items()
            if isinstance(value, bool) or not isinstance(value, int)
        }
        if invalid_types:
            raise TypeError(
                f"CAGCRN STFT parameters and sample_rate must be integers, got {invalid_types}"
            )
        if not isinstance(window, str):
            raise TypeError(f"CAGCRN window must be str, got {type(window).__name__}")
        if not isinstance(center, bool):
            raise TypeError(f"CAGCRN center must be bool, got {type(center).__name__}")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.sample_rate = sample_rate
        self.target_fs = self.sample_rate
        if min(self.n_fft, self.hop_length, self.win_length, self.sample_rate) <= 0:
            raise ValueError("CAGCRN STFT parameters and sample_rate must be positive")
        if self.win_length != self.n_fft:
            raise ValueError(
                f"CAGCRN requires win_length == n_fft, got {self.win_length} and {self.n_fft}"
            )
        if self.window != "hann":
            raise ValueError(f"CAGCRN supports only window='hann', got {self.window!r}")
        if isinstance(align_delay, bool) or not isinstance(align_delay, int) or align_delay < 1:
            raise ValueError(f"CAGCRN align_delay must be a positive integer, got {align_delay!r}")
        self.align_delay = align_delay
        self.high_freq = self.sample_rate / 2 if high_freq is None else high_freq
        if (
            isinstance(self.high_freq, bool)
            or not isinstance(self.high_freq, (int, float))
            or not math.isfinite(self.high_freq)
            or not 0 < self.high_freq <= self.sample_rate / 2
        ):
            raise ValueError(
                f"CAGCRN high_freq must be in (0, sample_rate/2]={self.sample_rate / 2}, "
                f"got {self.high_freq!r}"
            )
        self._net = _CAGCRNNet(
            nfft=self.n_fft,
            high_freq=self.high_freq,
            sample_rate=self.sample_rate,
            align_delay=self.align_delay,
        )

    def forward(self, batch: dict) -> dict:
        """Spectrum-in spectrum-out. Expects trainer-filled mic_spectra/farend_spectra; no STFT here."""
        if not isinstance(batch, dict):
            raise TypeError(f"CAGCRN forward expects a dict batch, got {type(batch).__name__}")
        missing = self.aec_input_capabilities.difference(batch)
        if missing:
            raise KeyError(f"CAGCRN batch is missing required keys: {sorted(missing)}")
        mic_spec = batch["mic_spectra"]
        ref_spec = batch["farend_spectra"]
        out = self._net(mic_spec, ref_spec)
        batch["nearend_estimate_spectra"] = out
        return batch


if __name__ == "__main__":
    import time

    import torch.nn as nn
    from core.utils.model_profile import _count_params, _fmt_num, make_profile_parser

    args = make_profile_parser(default_sr=16000).parse_args()
    model = CAGCRN().to(args.device).eval()
    total_p, train_p = _count_params(model)
    hop = 256
    n_fft = model.n_fft
    f_bins = n_fft // 2 + 1
    n_frames = max(2, int(args.sr * args.dur / hop))
    mic = torch.randn(args.batch, f_bins, n_frames, 2, device=args.device)
    far = torch.randn(args.batch, f_bins, n_frames, 2, device=args.device)

    class _SpecThopWrap(nn.Module):
        def __init__(self, inner: CAGCRN):
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

    print(f"=== CAGCRN (spectrum batch B={args.batch}, F={f_bins}, T={n_frames}) ===")
    print(f"  Params   : {total_p / 1e6:.3f} M (trainable {train_p / 1e6:.3f} M)")
    print(f"  MACs     : {macs_str} (thop on forward(batch) spectrum path)")
    print(f"  Latency  : {dt * 1000:.2f} ms / forward (mean of {args.repeats}, device={args.device})")
