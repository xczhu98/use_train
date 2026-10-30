"""Ultra-DPC-AEC: Ultra Dual-Path Compression for Joint Echo Cancellation and Noise Suppression.

Reference: Chen et al., "Ultra Dual-Path Compression For Joint Echo Cancellation And Noise Suppression",
           Tencent AI Lab, ICASSP 2023.
           Code: https://github.com/tencent-ailab/UltraDualPathCompression

Architecture:
  - Online DPT-FSNet backbone (2D Conv Encoder/Decoder + Dual-Path Transformer)
  - Inputs: mic spectrum, ref spectrum, error spectrum (from linear AEC)
  - Outputs: individual complex masks per input, summed as final enhanced spectrum
  - Optional frequency/time/dual-path compression for tunable complexity

Key differences from offline DPT-FSNet:
  1. LSTM -> single-layer GRU after first attention layer
  2. Standard attention -> Linear attention (causal for sub-band, bidirectional for full-band)
  3. Sub-band path unidirectional (causal), full-band path bidirectional
  4. Multi-input (mic/ref/error) with per-input mask output

================================================================================
NOTE on the "error" input (err_spec / error_spectra)
================================================================================
The third input "err_spec" is the RESIDUAL from a LINEAR AEC front-end
(e.g. NLMS / FDAF / Kalman filter in the frequency domain).  It is NOT an
independent microphone signal.  The paper's formulation is:

    err = mic - y_hat        where y_hat = linear_AEC(ref)

This residual carries the near-end speech + noise + non-linear echo leakage,
while the reference carries the far-end signal.  The model learns to jointly
suppress residual echo (from err) and noise (from mic) using three masks.

CURRENT IMPLEMENTATION STATUS
-----------------------------
LAEC (Speex MDF via ctypes) is integrated into the data pipeline
(``tasks/aec/data/linear_aec_frontend.py``).  When ``linear_aec`` is configured in
YAML, ``error_spectra`` is generated on-the-fly and injected into the batch
dict by AECTrainer.

  - Training with LAEC: ``error_spectra`` is present → three-input mode.
  - Training/Inference without LAEC (e2e): ``error_spectra`` is absent
    → falls back to ``mic_spectra``.  The model degrades to two-input
    (mic+ref) joint AEC+NS, which still works but does NOT reproduce the
    paper's reported performance.

  - forward_spectra(..., err_spec=None) : same fallback for backward compat.

Compression status (aligned with the official repo's released modules):
  - FreqCompression / FreqDecompression implement the official TrainMel
    semantics (HTK-Mel band boundaries, overlap-add with /2 normalization and
    a final tanh on decompression).
  - TimeCompression / TimeDecompression implement the official skip-prediction
    semantics (left padding, trans1/trans2, repeat and pad removal).
  - ``time_skip > 1`` activates the paper's PostNet on the error spectrum and
    the pre-mask output (active in both train and eval).
  - Streaming wrapper (``onnx/models/streaming_ultra_dpc.py``) natively
    supports frequency compression; time compression falls back to an
    explicit per-frame offline call (no cross-frame token grouping, documented
    in the wrapper), so offline-vs-streaming equivalence only holds for
    ``time_skip == 1``.

================================================================================
"""

from __future__ import annotations

import math
import os as _os
import sys as _sys
import typing as tp

if __name__ == "__main__":
    _sys.path.insert(
        0,
        _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..")),
    )

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.utils.registry import MODEL_REGISTRY


# ──────────────────────────────────────────────────────────────
# Linear Attention (Katharopoulos et al., 2020)
# ──────────────────────────────────────────────────────────────
class LinearAttention(nn.Module):
    """Linear-complexity attention using feature maps.

    For causal mode, uses cumulative sum for autoregressive masking.
    """

    def __init__(self, dim: int, n_heads: int = 4, causal: bool = False, eps: float = 1e-6):
        super().__init__()
        if not isinstance(n_heads, int) or isinstance(n_heads, bool) or n_heads < 1:
            raise ValueError(f"LinearAttention n_heads must be a positive integer, got {n_heads!r}")
        if dim % n_heads:
            raise ValueError(f"LinearAttention dim ({dim}) must be divisible by n_heads ({n_heads})")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.causal = causal
        self.eps = eps

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def _feature_map(self, x: Tensor) -> Tensor:
        # elu + 1 feature map
        return F.elu(x) + 1.0

    def forward(self, x: Tensor) -> Tensor:
        """Args: x (B, L, D) -> (B, L, D)"""
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # (B, L, n_heads, head_dim)

        q = self._feature_map(q)
        k = self._feature_map(k)

        if self.causal:
            # Causal linear attention via cumulative sum
            # KV = cumsum(K * V, dim=1), Z = cumsum(K, dim=1)
            kv = torch.einsum("blhd,blhe->blhde", k, v)
            kv_cum = torch.cumsum(kv, dim=1)
            k_cum = torch.cumsum(k, dim=1)

            num = torch.einsum("blhd,blhde->blhe", q, kv_cum)
            den = torch.einsum("blhd,blhd->blh", q, k_cum).clamp_min(self.eps)
            out = num / den.unsqueeze(-1)
        else:
            # Non-causal: global aggregation
            # Memory-opt: fuse einsum+sum so full kv tensor is not retained
            kv_sum = torch.einsum("blhd,blhe->blhde", k, v).sum(dim=1, keepdim=True)
            k_sum = k.sum(dim=1, keepdim=True)

            num = torch.einsum("blhd,blhde->blhe", q, kv_sum)
            den = torch.einsum("blhd,blhd->blh", q, k_sum).clamp_min(self.eps)
            out = num / den.unsqueeze(-1)

        out = out.reshape(B, L, D)
        return self.out(out)


# ──────────────────────────────────────────────────────────────
# Dual-Path Transformer Block
# ──────────────────────────────────────────────────────────────
class DualPathTransformerBlock(nn.Module):
    """One dual-path block: sub-band (causal, temporal) + full-band (bidirectional, spectral).

    Paper: only the *first* block has a GRU (single GRU placed after first attention layer).
    Subsequent blocks have attention only.

    Input/Output: (B, T, F, E)
    """

    def __init__(self, dim: int, n_heads: int = 4, use_gru: bool = False, gru_dim: int | None = None):
        super().__init__()
        self.dim = dim
        self.use_gru = use_gru
        gru_dim = gru_dim or dim

        # Sub-band path: along time axis, causal
        self.subband_norm1 = nn.LayerNorm(dim)
        self.subband_attn = LinearAttention(dim, n_heads, causal=True)
        if use_gru:
            self.subband_gru = nn.GRU(dim, gru_dim, num_layers=1, batch_first=True, bidirectional=False)
            self.subband_proj = nn.Linear(gru_dim, dim) if gru_dim != dim else nn.Identity()

        # Full-band path: along frequency axis, bidirectional
        self.fullband_norm = nn.LayerNorm(dim)
        self.fullband_attn = LinearAttention(dim, n_heads, causal=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, F, E = x.shape

        # --- Sub-band path (temporal, causal) ---
        x_sb = x.permute(0, 2, 1, 3).reshape(B * F, T, E)
        x_sb = self.subband_norm1(x_sb)
        # Gradient checkpointing: recompute subband_attn during backward to
        # avoid retaining the huge kv-cumsum activations (saves ~12 GB VRAM).
        from torch.utils.checkpoint import checkpoint
        x_sb = x_sb + checkpoint(self.subband_attn, x_sb, use_reentrant=False)
        if self.use_gru:
            x_sb = x_sb + self.subband_proj(self.subband_gru(x_sb)[0])
        x_sb = x_sb.reshape(B, F, T, E).permute(0, 2, 1, 3)
        x = x + x_sb

        # --- Full-band path (spectral, bidirectional) ---
        x_fb = x.reshape(B * T, F, E)
        x_fb = self.fullband_norm(x_fb)
        x_fb = x_fb + self.fullband_attn(x_fb)
        x_fb = x_fb.reshape(B, T, F, E)
        x = x + x_fb

        return x


# ──────────────────────────────────────────────────────────────
# Frequency Compression / Decompression (TrainMel)
# ──────────────────────────────────────────────────────────────
def _mel_band_boundaries(n_fft: int, n_bands: int, sr: int = 16000) -> tuple[list[int], list[int]]:
    """按官方 HTK-Mel 规则返回重叠频带的左右边界。

    官方实现先在 ``[0, sr // 2]`` 上建立 ``nfreq`` 个等间隔频点，
    再用相邻的三个 Mel 边界组成每个重叠 band；这里保持该取整语义。
    """
    if n_fft < 2:
        raise ValueError(f"n_fft must be >= 2, got {n_fft}")
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    if sr < 1:
        raise ValueError(f"sr must be >= 1, got {sr}")
    n_freq = n_fft // 2 + 1
    if n_bands > n_freq:
        raise ValueError(f"n_bands ({n_bands}) must not exceed frequency bins ({n_freq})")

    def hz_to_mel(freq: float) -> float:
        return 2595.0 * math.log10(1.0 + freq / 700.0)

    def mel_to_hz(mels: float) -> float:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # 与官方 ``torch.linspace(0, sample_rate // 2, nfreq)`` 完全一致。
    all_freqs = [float(sr // 2) * i / (n_freq - 1) for i in range(n_freq)]
    mel_min = hz_to_mel(0.0)
    mel_max = hz_to_mel(float(sr) / 2.0)
    mel_points = [mel_min + (mel_max - mel_min) * i / (n_bands + 1)
                  for i in range(n_bands + 2)]
    freq_points = [mel_to_hz(mel) for mel in mel_points]
    bounds = [0]
    for freq in freq_points[1:-1]:
        # 官方 argmax 在没有命中时返回 0；正常 band 配置不会触发该分支。
        bound = next((i for i, value in enumerate(all_freqs) if value > freq), 0)
        bounds.append(bound)
    bounds.append(n_freq)
    if any(bounds[i] >= bounds[i + 2] for i in range(n_bands)):
        raise ValueError(f"Mel bounds contain an empty band: {bounds}")
    return bounds[:-2], bounds[2:]


class FreqCompression(nn.Module):
    """官方频率压缩：输入输出布局均为 ``(B,C,T,F)``。"""

    def __init__(self, n_fft: int, n_bands: int, in_channels: int, embed_dim: int, sr: int = 16000):
        super().__init__()
        self.n_fft = n_fft
        self.n_bands = n_bands
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        low, high = _mel_band_boundaries(n_fft, n_bands, sr)
        self.register_buffer("low", torch.tensor(low, dtype=torch.long))
        self.register_buffer("high", torch.tensor(high, dtype=torch.long))
        self.proj = nn.ModuleList([
            nn.Linear((high[i] - low[i]) * in_channels, embed_dim, bias=False)
            for i in range(n_bands)
        ])

    def forward(self, x: Tensor) -> Tensor:
        B, _, T, _ = x.shape
        bands = []
        for i in range(self.n_bands):
            lo, hi = int(self.low[i]), int(self.high[i])
            band = x[:, :, :, lo:hi].permute(0, 2, 1, 3).reshape(B, T, -1)
            bands.append(self.proj[i](band))
        return torch.stack(bands, dim=-1).permute(0, 2, 1, 3).contiguous()


class FreqDecompression(nn.Module):
    """官方频率解压：重叠 band 做 overlap-add 后输出 ``(B,C,T,F)``。"""

    def __init__(self, n_fft: int, n_bands: int, out_channels: int, embed_dim: int, sr: int = 16000):
        super().__init__()
        self.n_fft = n_fft
        self.n_bands = n_bands
        self.out_channels = out_channels
        low, high = _mel_band_boundaries(n_fft, n_bands, sr)
        self.register_buffer("low", torch.tensor(low, dtype=torch.long))
        self.register_buffer("high", torch.tensor(high, dtype=torch.long))
        self.proj = nn.ModuleList([
            nn.Conv1d(embed_dim, out_channels * (high[i] - low[i]), 1, bias=True)
            for i in range(n_bands)
        ])

    def forward(self, x: Tensor) -> Tensor:
        B, _, T, _ = x.shape
        n_freq = self.n_fft // 2 + 1
        out = x.new_zeros(B, self.out_channels, T, n_freq)
        for i in range(self.n_bands):
            lo, hi = int(self.low[i]), int(self.high[i])
            band = self.proj[i](x[:, :, :, i])
            band = band.reshape(B, self.out_channels, hi - lo, T).permute(0, 1, 3, 2)
            out[:, :, :, lo:hi] += band
        if self.n_bands > 1:
            out[:, :, :, int(self.low[1]):int(self.high[-2])] /= 2.0
        return out.permute(0, 1, 2, 3).contiguous().tanh()


# ──────────────────────────────────────────────────────────────
# Time Compression / Decompression (Skip Prediction + PostNet)
# ──────────────────────────────────────────────────────────────
class TimeCompression(nn.Module):
    """官方时间压缩，输入输出布局均为 ``(B,C,T,F)``。

    ``proj`` 对应官方 ``TimeCompression.trans1``；逆向的 ``trans2`` 由
    :class:`TimeDecompression` 持有。两者共同组成官方的时间压缩模块。
    """

    def __init__(self, channels: int, skip: int, out_channels: int | None = None):
        super().__init__()
        if skip < 1:
            raise ValueError(f"skip must be >= 1, got {skip}")
        self.channels = channels
        self.out_channels = out_channels or channels
        self.skip = skip
        self.proj = nn.Conv2d(channels * skip, self.out_channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, int]:
        """返回 ``(压缩结果, pad)``，由调用方把 pad 传给解压侧。

        不在模块内保存 ``pad``：forward 保持无状态，重复调用 / 并发流 /
        ``torch.compile`` 下都依赖调用方显式传递。
        """
        B, C, T, F_bins = x.shape
        # 固定相位使分组不依赖整段长度：第 0、skip、2*skip... 帧产生新 token，
        # 每个 token 只拼接当前帧与历史帧。尾部右 pad 只生成最终不会被消费的 token。
        pad = self.skip - 1
        right_pad = (-(T + pad)) % self.skip
        x = F.pad(x, (0, 0, pad, right_pad))
        x = x.reshape(B, C, -1, self.skip, F_bins)
        x = x.permute(0, 1, 3, 2, 4).reshape(B, C * self.skip, -1, F_bins)
        return self.proj(x), pad


class TimeDecompression(nn.Module):
    """官方时间解压：``trans2``、repeat、左补零、裁剪和去除左 pad。"""

    def __init__(self, channels: int, skip: int, out_channels: int | None = None):
        super().__init__()
        if skip < 1:
            raise ValueError(f"skip must be >= 1, got {skip}")
        self.skip = skip
        self.out_channels = out_channels or channels
        self.proj = nn.Conv2d(channels, self.out_channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor, pad: int = 0) -> Tensor:
        x = self.proj(x)
        B, C, T, F_bins = x.shape
        x = x.unsqueeze(3).expand(B, C, T, self.skip, F_bins)
        x = x.reshape(B, C, T * self.skip, F_bins)
        x = F.pad(x, (0, 0, self.skip - 1, 0))
        x = x[:, :, :-(self.skip - 1), :] if self.skip > 1 else x
        return x[:, :, pad:, :] if pad else x


class PostProcessingNet(nn.Module):
    """官方 PostNet：log-power 多通道输入、频率压缩、GRU 和 stacked mask head。

    官方示例将三路输入谱与前级输出拼成 ``C+1`` 个 log-power 通道；
    ``input_channels=2`` 仍可用于只传入 error/previous 两路的独立调用。
    """

    def __init__(
        self,
        n_fft: int,
        input_channels: int = 2,
        hidden_channels: int = 4,
        freq_ratio: int = 2,
        sr: int = 16000,
    ):
        super().__init__()
        if input_channels < 1 or hidden_channels < 1:
            raise ValueError("input_channels and hidden_channels must be >= 1")
        self.n_fft = n_fft
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.freq_ratio = freq_ratio
        n_freq = n_fft // 2 + 1
        n_bands = max(1, n_freq // freq_ratio)
        self.n_bands = n_bands
        self.freq_comp = FreqCompression(
            n_fft, n_bands, in_channels=input_channels, embed_dim=1, sr=sr
        )
        self.gru = nn.GRU(
            n_bands, n_bands, num_layers=1, batch_first=True, bidirectional=False
        )
        self.freq_decomp = FreqDecompression(
            n_fft, n_bands, out_channels=hidden_channels, embed_dim=1, sr=sr
        )
        # 与官方 notebook 的 Linear(F,F) -> Conv2d -> Sigmoid -> Linear(F,F)
        # -> Conv2d -> Sigmoid 顺序一致。
        self.out = nn.Sequential(
            nn.Linear(n_freq, n_freq),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
            nn.Linear(n_freq, n_freq),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward_with_state(
        self,
        logpow: Tensor,
        prev_logpow: Tensor | None = None,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """计算实值修正 mask，并显式返回 causal GRU 状态。"""
        if prev_logpow is not None:
            logpow = torch.cat([logpow, prev_logpow], dim=1)
        if logpow.shape[1] != self.input_channels:
            raise ValueError(
                f"PostNet expects {self.input_channels} log-power channels, "
                f"got {logpow.shape[1]}"
            )
        x = self.freq_comp(logpow).squeeze(1).contiguous()
        x, hidden = self.gru(x, hidden)
        x = self.freq_decomp(x.unsqueeze(1))
        return self.out(x), hidden

    def forward(self, logpow: Tensor, prev_logpow: Tensor | None = None) -> Tensor:
        mask, _ = self.forward_with_state(logpow, prev_logpow)
        return mask


# ──────────────────────────────────────────────────────────────
# 2D Conv Encoder / Decoder
# ──────────────────────────────────────────────────────────────
class Conv2DEncoder(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, n_layers: int = 2):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(n_layers):
            # Use (3,1) kernel: convolve only along frequency, not time
            layers += [
                nn.Conv2d(ch, embed_dim, kernel_size=(3, 1), padding=(1, 0), bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.SiLU(inplace=True),
            ]
            ch = embed_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Conv2DDecoder(nn.Module):
    def __init__(self, embed_dim: int, out_ch: int, n_layers: int = 2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            out_c = out_ch if i == n_layers - 1 else embed_dim
            layers += [
                nn.Conv2d(embed_dim, out_c, kernel_size=(3, 1), padding=(1, 0), bias=True if i == n_layers - 1 else False),
                nn.BatchNorm2d(out_c) if i < n_layers - 1 else nn.Identity(),
                nn.SiLU(inplace=True) if i < n_layers - 1 else nn.Identity(),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────
# Ultra-DPC-AEC Model
# ──────────────────────────────────────────────────────────────
@MODEL_REGISTRY.register("UltraDPCAEC")
class UltraDPCAEC(nn.Module):
    """Ultra Dual-Path Compression AEC.

    Args:
        n_fft: STFT n_fft (default 320 for 20ms@16kHz)
        hop_length: STFT hop (default 160 for 10ms@16kHz)
        embed_dim: feature dimension E (default 48)
        n_dp_blocks: number of dual-path transformer blocks (default 4)
        n_heads: attention heads (default 4)
        encoder_layers: 2D conv encoder depth (default 2)
        decoder_layers: 2D conv decoder depth (default 2)
        freq_ratio: frequency compression ratio (1 = no compression)
        time_skip: time compression skip interval (1 = no compression)
        use_postnet: whether to use post-processing net for time compression
        sr: sample rate
    """

    aec_input_capabilities = frozenset(
        {"mic_spectra", "farend_spectra", "error_spectra"}
    )

    def __init__(
        self,
        n_fft: int = 320,
        hop_length: int = 160,
        embed_dim: int = 48,
        n_dp_blocks: int = 4,
        n_heads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        gru_dim: int | None = None,
        freq_ratio: int = 1,
        time_skip: int = 1,
        use_postnet: bool = True,
        sr: int = 16000,
        win_length: int | None = None,
        window: str = "hann",
        center: bool = True,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            raise TypeError(
                f"UltraDPCAEC got unexpected keyword arguments: {sorted(kwargs)}"
            )
        win_length = n_fft if win_length is None else win_length
        positive_ints = {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "embed_dim": embed_dim,
            "n_dp_blocks": n_dp_blocks,
            "n_heads": n_heads,
            "encoder_layers": encoder_layers,
            "decoder_layers": decoder_layers,
            "freq_ratio": freq_ratio,
            "time_skip": time_skip,
            "sr": sr,
        }
        invalid = {key: value for key, value in positive_ints.items()
                   if type(value) is not int or value < 1}
        if invalid:
            raise ValueError(f"UltraDPCAEC expects positive integer parameters, got {invalid}")
        if not hop_length <= win_length <= n_fft:
            raise ValueError("UltraDPCAEC requires hop_length <= win_length <= n_fft")
        if type(window) is not str or window.lower() not in {"hann", "hamming"}:
            raise ValueError(f"UltraDPCAEC supports hann/hamming windows, got {window!r}")
        if type(center) is not bool:
            raise TypeError(f"UltraDPCAEC center must be bool, got {center!r}")
        if embed_dim % n_heads:
            raise ValueError("UltraDPCAEC embed_dim must be divisible by n_heads")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window.lower()
        self.center = center
        self.embed_dim = embed_dim
        self.freq_ratio = freq_ratio
        self.time_skip = time_skip
        self.use_postnet = use_postnet and time_skip > 1
        self.sr = sr
        self.n_signals = 3  # mic, ref, error
        in_ch = self.n_signals * 2  # 每路信号的 real/imag
        n_freq = n_fft // 2 + 1
        has_freq = freq_ratio > 1
        has_time = time_skip > 1

        # 压缩链路严格按 time -> freq，逆向链路按 freq -> time。
        if has_time:
            time_out_ch = in_ch * 2 if has_freq else embed_dim
            self.time_comp = TimeCompression(in_ch, time_skip, time_out_ch)
        else:
            time_out_ch = in_ch
            self.time_comp = None
        self.time_decomp = (
            TimeDecompression(time_out_ch, time_skip, in_ch) if has_time else None
        )

        if has_freq:
            n_bands = max(1, n_freq // freq_ratio)
            self.freq_comp = FreqCompression(n_fft, n_bands, time_out_ch, embed_dim, sr)
            self.freq_decomp = FreqDecompression(n_fft, n_bands, time_out_ch, embed_dim, sr)
        else:
            self.freq_comp = None
            self.freq_decomp = None

        compressed = has_freq or has_time
        encoder_in_ch = embed_dim if compressed else in_ch
        # 压缩模式的 decoder 先还原到 latent channel，再由逆压缩模块恢复输入 channel。
        decoder_out_ch = embed_dim if compressed else in_ch
        if self.use_postnet:
            self.postnet = PostProcessingNet(n_fft, freq_ratio=2, sr=sr)
        else:
            self.postnet = None

        # --- 2D Conv Encoder ---
        self.encoder = Conv2DEncoder(encoder_in_ch, embed_dim, encoder_layers)

        # --- Dual-Path Transformer blocks ---
        self.dp_blocks = nn.ModuleList([
            DualPathTransformerBlock(embed_dim, n_heads, use_gru=(i == 0), gru_dim=gru_dim)
            for i in range(n_dp_blocks)
        ])

        # --- 2D Conv Decoder ---
        self.decoder = Conv2DDecoder(embed_dim, decoder_out_ch, decoder_layers)

        # --- Output layer: 1x1 conv to produce three路复数 masks ---
        self.mask_out = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True)

    def _apply_complex_masks(
        self,
        features: Tensor,
        mic_spec: Tensor,
        ref_spec: Tensor,
        err_spec: Tensor,
    ) -> Tensor:
        """将 ``(B,6,T,F)`` 特征映射为三路 complex mask 并融合。"""
        batch, freq_bins, frames, _ = mic_spec.shape
        masks = torch.tanh(self.mask_out(features.permute(0, 1, 3, 2).contiguous()))
        masks = masks.reshape(batch, self.n_signals, 2, freq_bins, frames)
        masks = masks.permute(0, 1, 3, 4, 2)
        inputs = torch.stack([mic_spec, ref_spec, err_spec], dim=1)
        real = inputs[..., 0] * masks[..., 0] - inputs[..., 1] * masks[..., 1]
        imag = inputs[..., 0] * masks[..., 1] + inputs[..., 1] * masks[..., 0]
        return torch.stack([real, imag], dim=-1).sum(dim=1)

    def _apply_postnet(
        self,
        enhanced: Tensor,
        mic_spec: Tensor,
        err_spec: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """以 error/前级输出 log-power 逐帧修正，并透传 PostNet 状态。"""
        if self.postnet is None:
            return enhanced, hidden
        err_logpow = torch.log(err_spec.square().sum(dim=-1).clamp_min(1e-8))
        prev_logpow = torch.log(enhanced.square().sum(dim=-1).clamp_min(1e-8))
        mask, hidden = self.postnet.forward_with_state(
            err_logpow.transpose(1, 2).unsqueeze(1),
            prev_logpow.transpose(1, 2).unsqueeze(1),
            hidden,
        )
        return enhanced + mic_spec * mask.permute(0, 3, 2, 1), hidden

    def forward_spectra(
        self, mic_spec: Tensor, ref_spec: Tensor, err_spec: Tensor | None = None
    ) -> Tensor:
        """Spectrum-in / spectrum-out core.

        Args:
            mic_spec: (B, F, T, 2) mic complex spectrum (noisy near-end + echo)
            ref_spec: (B, F, T, 2) reference (far-end) complex spectrum
            err_spec: (B, F, T, 2) error complex spectrum = residual of a
                      LINEAR AEC front-end (e.g. FDAF / NLMS / Kalman).
                      This is NOT an independent mic; it is mic minus the
                      linear echo estimate:  err = mic - linear_AEC(ref).
                      If None, uses mic_spec as fallback (linear AEC bypassed).
                      In training the LAEC residual is produced by the data
                      pipeline / PFDKF frontend and injected as
                      ``error_spectra`` by AECTrainer.

        Returns:
            enhanced_spec: (B, F, T, 2) enhanced near-end complex spectrum
        """
        if err_spec is None:
            err_spec = mic_spec  # fallback：训练时 LAEC 关闭/推理无线性前端
        for name, spec in (("mic_spec", mic_spec), ("ref_spec", ref_spec), ("err_spec", err_spec)):
            if not isinstance(spec, Tensor):
                raise TypeError(
                    f"UltraDPCAEC {name} must be a torch.Tensor, got {type(spec).__name__}"
                )
            if spec.ndim != 4 or spec.shape[-1] != 2:
                raise ValueError(
                    f"UltraDPCAEC expects {name} with shape (B, F, T, 2), got {tuple(spec.shape)}"
                )
            if not spec.is_floating_point():
                raise TypeError(f"UltraDPCAEC {name} must use a floating dtype, got {spec.dtype}")
        if not mic_spec.shape == ref_spec.shape == err_spec.shape:
            raise ValueError(
                "UltraDPCAEC requires mic_spec, ref_spec and err_spec to have identical shapes, "
                f"got {tuple(mic_spec.shape)}, {tuple(ref_spec.shape)} and {tuple(err_spec.shape)}"
            )
        expected_freqs = self.n_fft // 2 + 1
        if mic_spec.shape[1] != expected_freqs:
            raise ValueError(
                f"UltraDPCAEC n_fft={self.n_fft} requires F={expected_freqs}, got F={mic_spec.shape[1]}"
            )
        if mic_spec.dtype != ref_spec.dtype or mic_spec.dtype != err_spec.dtype:
            raise TypeError("UltraDPCAEC mic_spec, ref_spec and err_spec dtypes must match")
        if mic_spec.device != ref_spec.device or mic_spec.device != err_spec.device:
            raise ValueError("UltraDPCAEC mic_spec, ref_spec and err_spec devices must match")
        B, F_bins, T, _ = mic_spec.shape

        # ===== 1. 输入堆叠，统一为官方内部布局 (B,C,T,F) =====
        x = torch.stack([mic_spec, ref_spec, err_spec], dim=1)
        x = x.permute(0, 1, 4, 3, 2).reshape(B, self.n_signals * 2, T, F_bins)

        # ===== 2. 官方联合顺序：time -> freq =====
        time_pad = 0
        if self.time_comp is not None:
            x, time_pad = self.time_comp(x)
        if self.freq_comp is not None:
            x = self.freq_comp(x)

        # Conv2d 使用 (B,C,F,T)，dual-path 再切回 (B,T,F,E)。
        x = self.encoder(x.permute(0, 1, 3, 2).contiguous())
        x = x.permute(0, 3, 2, 1).contiguous()
        for blk in self.dp_blocks:
            x = blk(x)

        # ===== 3. 逆序解压：freq -> time =====
        x = self.decoder(x.permute(0, 3, 2, 1).contiguous())
        x = x.permute(0, 1, 3, 2).contiguous()
        if self.freq_decomp is not None:
            x = self.freq_decomp(x)
        if self.time_decomp is not None:
            x = self.time_decomp(x, time_pad)
        x = x[:, :, :T, :]

        # ===== 4. 三路 complex mask 融合；PostNet 再做逐帧实值修正 =====
        enhanced = self._apply_complex_masks(x, mic_spec, ref_spec, err_spec)
        enhanced, _ = self._apply_postnet(enhanced, mic_spec, err_spec)
        return enhanced

    def forward(self, batch: dict) -> dict:
        """Dict-based forward compatible with AEC training loop.

        Expected keys:
            mic_spectra    : (B, F, T, 2)
            farend_spectra : (B, F, T, 2)
        Optional keys:
            error_spectra  : (B, F, T, 2) — from linear AEC pre-processing.
                             If absent, mic_spectra is used as fallback.

        Sets:
            nearend_estimate_spectra : (B, F, T, 2)
        """
        if not isinstance(batch, dict):
            raise TypeError(f"UltraDPCAEC forward expects a dict batch, got {type(batch).__name__}")
        required_inputs = {"mic_spectra", "farend_spectra"}
        missing = required_inputs.difference(batch)
        if missing:
            raise KeyError(f"UltraDPCAEC batch is missing required keys: {sorted(missing)}")
        mic_spec = batch["mic_spectra"]
        ref_spec = batch["farend_spectra"]
        err_spec = batch.get("error_spectra")
        batch["nearend_estimate_spectra"] = self.forward_spectra(mic_spec, ref_spec, err_spec)
        return batch


if __name__ == "__main__":
    # Quick sanity check
    model = UltraDPCAEC(n_fft=320, hop_length=160, embed_dim=48, n_dp_blocks=4)
    B, f_bins, T = 2, 161, 100
    mic = torch.randn(B, f_bins, T, 2)
    ref = torch.randn(B, f_bins, T, 2)
    err = torch.randn(B, f_bins, T, 2)
    out = model({"mic_spectra": mic, "farend_spectra": ref, "error_spectra": err})
    print(f"Input:  mic={mic.shape}, ref={ref.shape}, err={err.shape}")
    print(f"Output: {out['nearend_estimate_spectra'].shape}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"Params: {n_params:.1f}K")
