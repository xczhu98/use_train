"""批量矢量化 kaldi-fbank（数值对齐 ``torchaudio.compliance.kaldi.fbank``）.

移植来源：wesep ``wesep/modules/speaker/encoder.py::Fbank_kaldi``。原实现逐样本
Python 循环调用 ``kaldi.fbank``（该接口仅支持单声道 ``(c, n)``，``c < 2``），是
speaker frontend 的吞吐瓶颈。本实现按 ``torchaudio.compliance.kaldi`` 内部流程
（分帧 → 去直流 → 预加重 → hamming 窗 → 功率谱 → mel 滤波 → log）改写为批量
无循环版本，dither=0 时与逐样本调用误差 < 1e-4（见提交时的 /tmp 验证脚本）。

与 wesep 调用配置一致（其余 kaldi 参数取其默认值）：
``window_type="hamming"``、``use_energy=False``、``snip_edges=True``、
``remove_dc_offset=True``、``preemphasis_coefficient=0.97``、
``round_to_power_of_two=True``、``use_power=True``、``use_log_fbank=True``、
``low_freq=20``、``high_freq=0``（即 nyquist）。

Author: Xincheng Zhu
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 与 torchaudio.compliance.kaldi 相同的 float32 epsilon（防止 log(0)）
_EPSILON = torch.tensor(torch.finfo(torch.float32).eps)


def _next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _mel_scale(freq: torch.Tensor) -> torch.Tensor:
    return 1127.0 * (1.0 + freq / 700.0).log()


def _get_mel_banks(
    num_bins: int,
    window_length_padded: int,
    sample_freq: float,
    low_freq: float,
    high_freq: float,
) -> torch.Tensor:
    """生成 kaldi 三角 mel 滤波器组 ``(num_bins, window_length_padded // 2)``.

    逐行复刻 ``torchaudio.compliance.kaldi.get_mel_banks``（vtln_warp=1.0 分支），
    保证 mel 权重与逐样本参考完全一致。
    """
    if num_bins <= 3:
        raise ValueError(f"num_bins must be > 3, got {num_bins}")
    if window_length_padded % 2 != 0:
        raise ValueError(
            f"window_length_padded must be even, got {window_length_padded}"
        )
    num_fft_bins = window_length_padded // 2
    nyquist = 0.5 * sample_freq
    if high_freq <= 0.0:
        high_freq += nyquist
    if not 0.0 <= low_freq < high_freq <= nyquist:
        raise ValueError(
            f"Bad low_freq={low_freq} / high_freq={high_freq} vs nyquist {nyquist}"
        )

    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = 1127.0 * math.log(1.0 + low_freq / 700.0)
    mel_high_freq = 1127.0 * math.log(1.0 + high_freq / 700.0)
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    bin_ = torch.arange(num_bins).unsqueeze(1)
    left_mel = mel_low_freq + bin_ * mel_freq_delta
    center_mel = mel_low_freq + (bin_ + 1.0) * mel_freq_delta
    right_mel = mel_low_freq + (bin_ + 2.0) * mel_freq_delta

    # (1, num_fft_bins)：各 FFT bin 中心频率的 mel 刻度
    mel = _mel_scale(fft_bin_width * torch.arange(num_fft_bins)).unsqueeze(0)

    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    # left < center < right，取两斜率最小值并截断负值
    bins = torch.clamp(torch.min(up_slope, down_slope), min=0.0)
    return bins


class FbankKaldi(nn.Module):
    """批量 kaldi-fbank（80 mel / 25ms / 10ms）+ 句级 mean-CMN。

    输入 ``(B, T)`` 波形（float，[-1, 1] 量级），输出 ``(B, T_f, num_mel_bins)``。
    无参数、无持久 buffer（state_dict 干净，不影响比赛 checkpoint 键名兼容）。
    """

    def __init__(
        self,
        num_mel_bins: int = 80,
        frame_length: float = 25.0,
        frame_shift: float = 10.0,
        dither: float = 0.0,
        sample_rate: int = 16000,
        norm_mean: bool = True,
        norm_var: bool = False,
    ):
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.dither = dither
        self.norm_mean = norm_mean
        self.norm_var = norm_var

        self.window_size = int(sample_rate * frame_length * 0.001)
        self.window_shift = int(sample_rate * frame_shift * 0.001)
        self.padded_window_size = _next_power_of_2(self.window_size)

        # kaldi hamming：periodic=False, alpha=0.54, beta=0.46
        window = torch.hamming_window(
            self.window_size, periodic=False, alpha=0.54, beta=0.46
        )
        # mel 滤波器组：(num_mel_bins, n_fft//2) → 右补一列零 → (num_mel_bins, n_fft//2+1)
        mel = _get_mel_banks(
            num_mel_bins, self.padded_window_size, float(sample_rate), 20.0, 0.0
        )
        mel = F.pad(mel, (0, 1), mode="constant", value=0)

        # fbank 无权重，不进入 state_dict。
        self.register_buffer("window", window, persistent=False)
        self.register_buffer("mel_energies", mel, persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` → ``(B, T_f, num_mel_bins)``，``T_f = 1 + (T - win) // shift``。"""
        if wav.dim() != 2:
            raise ValueError(f"FbankKaldi expects (B, T) waveform, got {wav.shape}")
        B, T = wav.shape
        if T < self.window_size:
            raise ValueError(
                f"Waveform too short for fbank: T={T} < window_size={self.window_size}"
            )
        eps = _EPSILON.to(device=wav.device, dtype=wav.dtype)

        # 与 kaldi 对齐：波形先乘 2^15
        wav = wav * (1 << 15)

        # 分帧（snip_edges=True）：(B, m, window_size)
        frames = wav.unfold(1, self.window_size, self.window_shift)

        if self.training and self.dither != 0.0:
            frames = frames + torch.randn_like(frames) * self.dither

        # 去直流（每帧减均值）
        frames = frames - frames.mean(dim=-1, keepdim=True)

        # 预加重：y[j] = x[j] - 0.97 * x[max(0, j-1)]
        shifted = F.pad(frames, (1, 0), mode="replicate")[..., :-1]
        frames = frames - 0.97 * shifted

        # 加窗 + 零填充到 padded_window_size
        frames = frames * self.window
        if self.padded_window_size != self.window_size:
            frames = F.pad(frames, (0, self.padded_window_size - self.window_size))

        # 功率谱（use_power=True）：(B, m, n_fft//2+1)
        spectrum = torch.fft.rfft(frames).abs().pow(2.0)

        # mel 滤波 + log（use_log_fbank=True）
        mel_energies = torch.matmul(spectrum, self.mel_energies.transpose(0, 1))
        mel_energies = torch.max(mel_energies, eps).log()

        # 句级 CMN（mean；wesep 默认 norm_mean=True / norm_var=False）
        if self.norm_mean:
            mel_energies = mel_energies - mel_energies.mean(dim=1, keepdim=True)
        if self.norm_var:
            # 单帧时无偏方差无定义；多帧仍与 wesep apply_cmvn 默认语义一致。
            correction = 0 if mel_energies.shape[1] == 1 else 1
            mel_energies = mel_energies / torch.sqrt(
                mel_energies.var(dim=1, keepdim=True, correction=correction) + 1e-8
            )
        return mel_energies
