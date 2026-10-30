"""Self-contained Band-Split RNN (BSRNN) for speech enhancement.

Ported from espnet2 (enh/layers/bsrnn.py + enh/layers/tcn.py) with all external
dependencies removed.  Only requires PyTorch.

本模块支持两套预设，便于与不同实现对齐：
  - 模型 A（481 bins / 48k）：n_fft=960, hop=480，34 子带，mask+residual；
    与 espnet / URGENT2026 baseline 一致。
  - 模型 B（1025 bins / 48k）：n_fft=2048, hop=512，41 子带，仅 mask；
    与 USR-BSRNN 论文 (bsrnn_sfi, ICASSP 2023) 一致；论文 0.35M/0.52G MACs，本实现 0.475M/0.529G（口径见 docs/architecture/SE_Architecture.md、tools/model_tools/audit_metrics.py）。

References:
    [1] J. Yu et al., "High fidelity speech enhancement with band-split RNN,"
        Proc. ISCA Interspeech, 2023.
    [2] J. Yu & Y. Luo, "Efficient monaural speech enhancement with universal
        sample rate band-split RNN," Proc. ICASSP, 2023.
"""

from itertools import accumulate
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

EPS = torch.finfo(torch.get_default_dtype()).eps


# ---------------------------------------------------------------------------
#  Normalization helpers
# ---------------------------------------------------------------------------

class ChannelwiseLayerNorm1d(nn.Module):
    """Channel-wise Layer Norm for 3-D input (B, C, T)."""

    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1))

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        mean = y.mean(dim=1, keepdim=True)
        var = y.var(dim=1, keepdim=True, unbiased=False)
        return self.gamma * (y - mean) / (var + EPS).sqrt() + self.beta


def choose_norm_1d(norm_type: str, channel_size: int) -> nn.Module:
    """Return a 1-D (B, C, T) normalization layer."""
    if norm_type == "cLN":
        return ChannelwiseLayerNorm1d(channel_size)
    if norm_type == "BN":
        return nn.BatchNorm1d(channel_size)
    if norm_type == "GN":
        return nn.GroupNorm(1, channel_size, eps=1e-8)
    raise ValueError(f"Unsupported 1D norm type: {norm_type}")


class ChannelwiseLayerNorm2d(nn.Module):
    """Channel-wise Layer Norm for 4-D input (B, C, T, F)."""

    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1, 1))

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        mean = y.mean(dim=1, keepdim=True)
        var = y.var(dim=1, keepdim=True, unbiased=False)
        return self.gamma * (y - mean) / (var + EPS).sqrt() + self.beta


class ChannelFreqwiseLayerNorm(nn.Module):
    """Channel-and-Frequency-wise Layer Norm for 4-D input (B, C, T, F)."""

    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1, 1))

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        mean = y.mean(dim=(1, 3), keepdim=True)
        var = ((y - mean) ** 2).mean(dim=(1, 3), keepdim=True)
        return self.gamma * (y - mean) / (var + EPS).sqrt() + self.beta


def choose_norm_2d(norm_type: str, channel_size: int) -> nn.Module:
    """Return a 2-D (B, C, T, F) normalization layer."""
    if norm_type == "cfLN":
        return ChannelFreqwiseLayerNorm(channel_size)
    if norm_type == "cLN":
        return ChannelwiseLayerNorm2d(channel_size)
    if norm_type == "BN":
        return nn.BatchNorm2d(channel_size)
    if norm_type == "GN":
        return nn.GroupNorm(1, channel_size)
    raise ValueError(f"Unsupported 2D norm type: {norm_type}")


# ---------------------------------------------------------------------------
#  BandSplit
# ---------------------------------------------------------------------------

class BandSplit(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_fs: int = 48000,
        subbands: Optional[List[Tuple[int, int]]] = None,
        channels: int = 128,
        norm_type: str = "GN",
    ):
        super().__init__()
        assert input_dim % 2 == 1, f"input_dim must be odd, got {input_dim}"
        n_fft = (input_dim - 1) * 2
        freqs = torch.fft.rfftfreq(n_fft, 1.0 / target_fs).tolist()

        if subbands is not None:
            assert isinstance(subbands, (list, tuple)) and len(subbands) > 0
            self.subbands = tuple(tuple(t) for t in subbands)
        elif input_dim == 481 and target_fs == 48000:
            # ---------- 模型 A：espnet / URGENT2026 对齐 ----------
            # n_fft=960, hop=480（20ms 窗），34 子带（5 + 19×4 + 6×10 + 7×40 + 60 = 481）
            sizes = tuple([5] + [4] * 19 + [10] * 6 + [40] * 7 + [60])
            self.subbands = self._sizes_to_indices(sizes, input_dim)
        elif input_dim == 1025 and target_fs == 48000:
            # ---------- 模型 B：USR-BSRNN 论文 (ICASSP 2023) 对齐 ----------
            # n_fft=2048, hop=512（~42ms 窗），41 子带
            # freq_res = 48000/2048 ≈ 23.4 Hz/bin；子带按 Hz 划分后映射到 bin 数
            #   < 1kHz:   100Hz bands → ~4-5 bins × 10 bands
            #   1-4kHz:   250Hz bands → ~10-11 bins × 12 bands
            #   4-8kHz:   500Hz bands → ~21 bins × 8 bands
            #   8-16kHz:  1kHz  bands → ~43 bins × 8 bands
            #   16-20kHz: 2kHz  bands → ~85 bins × 2 bands
            #   20-24kHz: 剩余 bin 归为 1 个子带
            # 论文 Table 2：48k → 0.52G MACs（本实现 0.529G 一致）；参数量论文 0.35M，本实现 0.475M（口径见 docs/architecture/SE_Architecture.md、tools/model_tools/audit_metrics.py）。
            # 低采样率：本实现按 48k 固定 bin 布局，forward 时跳过 >fs/2 的子带；
            # 论文或按 Hz 固定子带，低 SR 时每子带 bin 数不同，故 8k/16k MACs 可能与表 2 略有差异。
            sizes = self._compute_usr_bsrnn_subbands(n_fft, target_fs)
            self.subbands = self._sizes_to_indices(sizes, input_dim)
        elif input_dim == 257 and target_fs == 16000:
            # 16kHz 预设：n_fft=512，31 子带；低频更细（1–10 bins × 10 段），其余 8–10 bins × 21 段
            sizes = tuple([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [9] * 8 + [10] * 13)
            self.subbands = self._sizes_to_indices(sizes, input_dim)
        else:
            raise NotImplementedError(
                f"Define subbands for input_dim={input_dim}, target_fs={target_fs}. "
                f"Pass explicit `subbands` list or use n_fft=960/2048 with target_fs=48000, or n_fft=512 with target_fs=16000."
            )

        self.subband_freqs = tuple((freqs[s], freqs[e]) for s, e in self.subbands)

        self.norm = nn.ModuleList()
        self.fc = nn.ModuleList()
        for st, et in self.subbands:
            sb = et - st + 1
            self.norm.append(choose_norm_1d(norm_type, int(sb * 2)))
            self.fc.append(nn.Conv1d(int(sb * 2), channels, 1))

    @staticmethod
    def _sizes_to_indices(sizes, input_dim):
        assert sum(sizes) == input_dim, (sum(sizes), input_dim)
        cumsum = tuple(accumulate((0,) + sizes))
        return tuple((cumsum[i], cumsum[i + 1] - 1) for i in range(len(cumsum) - 1))

    @staticmethod
    def _compute_usr_bsrnn_subbands(n_fft: int, fs: int) -> tuple:
        """按 Hz 边界计算 USR-BSRNN 各子带 bin 数（模型 B，论文 Yu & Luo ICASSP 2023）。

        子带划分（Hz）：
            <1kHz:    100Hz 带宽 × 10 段
            1-4kHz:   250Hz 带宽 × 12 段
            4-8kHz:   500Hz 带宽 × 8 段
            8-16kHz:  1kHz  带宽 × 8 段
            16-20kHz: 2kHz  带宽 × 2 段
            20kHz 以上: 剩余 bin 合并为 1 段 → 共 41 段
        返回各段对应的 bin 数量元组，总和 = n_fft//2 + 1。
        """
        freq_res = fs / n_fft
        input_dim = n_fft // 2 + 1
        boundaries_hz = (
            [(i * 100, (i + 1) * 100) for i in range(10)]           # 0-1kHz
            + [(1000 + i * 250, 1000 + (i + 1) * 250) for i in range(12)]  # 1-4kHz
            + [(4000 + i * 500, 4000 + (i + 1) * 500) for i in range(8)]   # 4-8kHz
            + [(8000 + i * 1000, 8000 + (i + 1) * 1000) for i in range(8)] # 8-16kHz
            + [(16000 + i * 2000, 16000 + (i + 1) * 2000) for i in range(2)] # 16-20kHz
        )
        sizes = []
        used = 0
        for lo, hi in boundaries_hz:
            bin_lo = round(lo / freq_res)
            bin_hi = round(hi / freq_res)
            size = bin_hi - bin_lo
            if size <= 0:
                size = 1
            sizes.append(size)
            used += size
        sizes.append(input_dim - used)  # remaining bins → last band
        assert sum(sizes) == input_dim, f"Subband sum {sum(sizes)} != {input_dim}"
        return tuple(sizes)

    def forward(self, x: torch.Tensor, fs: Optional[int] = None) -> torch.Tensor:
        """将频谱按子带切分并映射到通道维。(B, T, F, 2) -> (B, N, T, K)。

        当传入 fs 时，会跳过中心频率 >= fs/2 的子带（多采样率时子带数可变）。
        """
        parts = []
        for i, (st, et) in enumerate(self.subbands):
            if st >= x.size(2):
                break
            if fs is not None and self.subband_freqs[i][0] >= fs / 2:
                break
            sb = et - st + 1
            x_band = x[:, :, st: et + 1, :]
            if sb > x_band.size(2):
                x_band = nn.functional.pad(x_band, (0, 0, 0, sb - x_band.size(2)))
            x_band = x_band.reshape(x_band.size(0), x_band.size(1), -1)
            out = self.fc[i](self.norm[i](x_band.transpose(1, 2)))
            parts.append(out.unsqueeze(-1))
        return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
#  MaskDecoder
# ---------------------------------------------------------------------------

class MaskDecoder(nn.Module):
    """按子带估计复数 mask（及可选的 residual）。

    - use_residual=True：预测 mask + residual（模型 A / espnet/URGENT 一致）。
    - use_residual=False：仅预测 mask（模型 B / 论文轻量版，参数量与算力更小）。
    """

    def __init__(self, freq_dim: int, subbands, channels: int = 128,
                 num_spk: int = 1, norm_type: str = "GN",
                 use_residual: bool = True):
        super().__init__()
        assert freq_dim == subbands[-1][1] + 1
        self.subbands = subbands
        self.freq_dim = freq_dim
        self.num_spk = num_spk
        self.use_residual = use_residual
        self.mlp_mask = nn.ModuleList()
        if use_residual:
            self.mlp_residual = nn.ModuleList()
        for st, et in self.subbands:
            sb = et - st + 1
            self.mlp_mask.append(nn.Sequential(
                choose_norm_1d(norm_type, channels),
                nn.Conv1d(channels, 4 * channels, 1), nn.Tanh(),
                nn.Conv1d(4 * channels, int(sb * 4 * num_spk), 1), nn.GLU(dim=1),
            ))
            if use_residual:
                self.mlp_residual.append(nn.Sequential(
                    choose_norm_1d(norm_type, channels),
                    nn.Conv1d(channels, 4 * channels, 1), nn.Tanh(),
                    nn.Conv1d(4 * channels, int(sb * 4 * num_spk), 1), nn.GLU(dim=1),
                ))

    def forward(self, x: torch.Tensor):
        """(B, N, T, K) -> mask (B, S, T, F, 2), residual (B, S, T, F, 2)

        When use_residual=False, residual is all zeros.
        """
        m = x.new_zeros(x.size(0), x.size(2), self.num_spk, self.freq_dim, 2)
        r = x.new_zeros(x.size(0), x.size(2), self.num_spk, self.freq_dim, 2)
        for i, (st, et) in enumerate(self.subbands):
            if i >= x.size(-1):
                break
            x_band = x[:, :, :, i]
            out = self.mlp_mask[i](x_band).transpose(1, 2).contiguous()
            out = out.reshape(out.size(0), out.size(1), self.num_spk, -1, 2)
            m[..., st: et + 1, :] += out
            if self.use_residual:
                res = self.mlp_residual[i](x_band).transpose(1, 2).contiguous()
                res = res.reshape(res.size(0), res.size(1), self.num_spk, -1, 2)
                r[..., st: et + 1, :] += res
        return m.moveaxis(1, 2), r.moveaxis(1, 2)


# ---------------------------------------------------------------------------
#  Core BSRNN
# ---------------------------------------------------------------------------

class BSRNN(nn.Module):
    """Band-Split RNN，无 espnet2 依赖。

    输入: (B, T, F, 2) 实部/虚部最后一维的 STFT
    输出: (B, num_spk, T, F, 2)

    常用参数：
        input_dim: 由 STFT 注入，481（模型 A）或 1025（模型 B）
        num_channel (N): 子带特征维度；模型 A 常用 196，模型 B 论文 16
        hidden_factor: LSTM hidden = hidden_factor * N，通常为 2
        use_residual: 模型 A 用 True（mask+residual），模型 B 用 False（仅 mask）
    """

    def __init__(
        self,
        input_dim: int = 481,
        num_channel: int = 16,
        num_layer: int = 6,
        target_fs: int = 48000,
        subbands: Optional[List[Tuple[int, int]]] = None,
        causal: bool = False,
        num_spk: int = 1,
        norm_type: str = "GN",
        hidden_factor: int = 2,
        use_residual: bool = True,
    ):
        super().__init__()
        norm1d_type = norm_type if norm_type != "cfLN" else "cLN"
        self.num_layer = num_layer
        self.band_split = BandSplit(
            input_dim, target_fs=target_fs, subbands=subbands,
            channels=num_channel, norm_type=norm1d_type,
        )
        self.target_fs = target_fs
        self.causal = causal
        self.num_spk = num_spk

        self.norm_time = nn.ModuleList()
        self.rnn_time = nn.ModuleList()
        self.fc_time = nn.ModuleList()
        self.norm_freq = nn.ModuleList()
        self.rnn_freq = nn.ModuleList()
        self.fc_freq = nn.ModuleList()
        hdim = hidden_factor * num_channel
        for _ in range(num_layer):
            self.norm_time.append(choose_norm_2d(norm_type, num_channel))
            self.rnn_time.append(nn.LSTM(
                num_channel, hdim, batch_first=True, bidirectional=not causal,
            ))
            self.fc_time.append(nn.Linear(hdim if causal else hdim * 2, num_channel))
            self.norm_freq.append(choose_norm_2d(norm_type, num_channel))
            self.rnn_freq.append(nn.LSTM(
                num_channel, hdim, batch_first=True, bidirectional=True,
            ))
            self.fc_freq.append(nn.Linear(hdim * 2, num_channel))  # freq LSTM always bidir

        self.mask_decoder = MaskDecoder(
            input_dim, self.band_split.subbands,
            channels=num_channel, num_spk=num_spk, norm_type=norm1d_type,
            use_residual=use_residual,
        )

    @property
    def num_subbands(self) -> int:
        """Number of frequency subbands (K)."""
        return len(self.band_split.subbands)

    @property
    def hidden_dim(self) -> int:
        """Hidden dimension of time LSTM."""
        return self.rnn_time[0].hidden_size

    # ------------------------------------------------------------------
    #  Full-sequence forward (training & offline inference)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, fs: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, F, 2)
            fs: optional sampling rate for sub-Nyquist truncation
        Returns:
            (B, num_spk, T, F, 2)
        """
        z = self.band_split(x, fs=fs)
        B, N, T, K = z.shape
        skip = z
        for i in range(self.num_layer):
            out = self.norm_time[i](skip)
            out = out.transpose(1, 3).reshape(B * K, T, N)
            out, _ = self.rnn_time[i](out)
            out = self.fc_time[i](out)
            out = out.reshape(B, K, T, N).transpose(1, 3)
            skip = skip + out

            out = self.norm_freq[i](skip)
            out = out.permute(0, 2, 3, 1).contiguous().reshape(B * T, K, N)
            out, _ = self.rnn_freq[i](out)
            out = self.fc_freq[i](out)
            out = out.reshape(B, T, K, N).permute(0, 3, 1, 2).contiguous()
            skip = skip + out

        m, r = self.mask_decoder(skip)
        m = torch.view_as_complex(m)
        r = torch.view_as_complex(r)
        x_c = torch.view_as_complex(x)
        m = m[..., :x_c.size(-1)]
        r = r[..., :x_c.size(-1)]
        return torch.view_as_real(m * x_c.unsqueeze(1) + r)
