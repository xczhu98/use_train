"""Estimate effective audio bandwidth and derive 真实/有效采样率 (true sample rate)."""
import math
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from typing import Tuple

# 每块处理的帧数：分块流式累积，避免一次性物化 n_frames × fft_size 的
# float64 帧矩阵与 complex128 频谱（1 小时 48kHz 音频约 5GB+10GB）。
# 只影响内存峰值，不影响数值结果（逐帧计算口径不变）。
_FFT_FRAME_BLOCK = 4096


@dataclass(frozen=True)
class BandwidthAnalysisParameters:
    """Step1 bandwidth 执行与 config digest 共用的不可变参数。"""

    fft_size: int = 2048
    threshold_db: float = -40.0


STEP1_BANDWIDTH_PARAMETERS = BandwidthAnalysisParameters()


def estimate_audio_bandwidth(
    audio_path_or_data,
    sr: int = None,
    fft_size: int = STEP1_BANDWIDTH_PARAMETERS.fft_size,
    threshold_db: float = STEP1_BANDWIDTH_PARAMETERS.threshold_db,
) -> Tuple[int, float]:
    """Estimate effective audio bandwidth via spectral analysis.

    Used to record 真实采样率/有效频谱：部分音频标称 48k 但有效频谱有限，
    通过带宽估计得到 bandwidth_hz 与 derived sample rate，便于下游区分。

    Args:
        audio_path_or_data: file path (str) or pre-loaded numpy array.
        sr: sample rate (required when passing numpy array, ignored for file path).
        fft_size: FFT window size.
        threshold_db: magnitude threshold for bandwidth detection.

    Returns:
        (sample_rate, bandwidth_hz)
    """
    if isinstance(audio_path_or_data, (str, bytes)):
        audio, sr = sf.read(audio_path_or_data)
    else:
        audio = audio_path_or_data
        if sr is None:
            raise ValueError("sr is required when passing pre-loaded audio array")

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1, dtype=np.float64)

    if audio.size == 0:
        return sr, 0.0

    audio = audio.astype(np.float64)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    hop_size = fft_size // 4
    if audio.size < fft_size:
        audio = np.pad(audio, (0, fft_size - audio.size))
        n_frames = 1
    else:
        n_frames = 1 + (len(audio) - fft_size) // hop_size

    threshold_linear = 10 ** (threshold_db / 20.0)
    window = np.hanning(fft_size)

    # 分块流式：按块构造帧并批量 FFT，逐块累积 frame_max_bins，
    # 全量帧矩阵与频谱不驻留内存（长音频内存峰值与时长解耦）。
    # 数值口径与一次性全量计算完全一致：逐帧 max-bin 独立计算。
    frame_starts = np.arange(n_frames, dtype=np.int64) * hop_size
    bin_offsets = np.arange(fft_size, dtype=np.int64)
    n_bins = fft_size // 2 + 1
    frame_max_bins = np.empty(n_frames, dtype=np.int64)
    for bstart in range(0, n_frames, _FFT_FRAME_BLOCK):
        bend = min(bstart + _FFT_FRAME_BLOCK, n_frames)
        indices = frame_starts[bstart:bend, None] + bin_offsets
        windowed = audio[indices] * window
        magnitude = np.abs(np.fft.rfft(windowed, axis=1))

        magnitude_normalized = magnitude / (np.max(magnitude, axis=1, keepdims=True) + 1e-10)
        # 逐帧取过阈最高 bin，再取 95 分位数：避免单个 click/爆音帧
        # 经逐帧归一化后把全局 max 顶到 Nyquist
        above = magnitude_normalized > threshold_linear
        has_above = above.any(axis=1)
        frame_max_bins[bstart:bend] = np.where(
            has_above,
            n_bins - 1 - np.argmax(above[:, ::-1], axis=1),
            0,
        )

    max_freq_bin = int(np.percentile(frame_max_bins, 95))

    freq_resolution = sr / fft_size
    bandwidth_hz = max_freq_bin * freq_resolution

    return sr, bandwidth_hz


# 与 Urgent2026 resample_to_estimated_bandwidth 对齐的标准采样率列表
STANDARD_SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)


def get_estimated_sample_rate_for_resample(bandwidth_hz: float) -> int:
    """按「最小满足 2×bandwidth 的标准采样率」作为 est_fs（估计/真实有效采样率），与 Urgent2026 一致。

    在标准率中取最小的 sr 使得 est_bandwidth * 2 <= sr；若无则用列表最大值（48000）。
    用于第一步按真实有效采样率落盘时的目标采样率。
    """
    for sr in STANDARD_SAMPLE_RATES:
        if bandwidth_hz * 2 <= sr:
            return sr
    return STANDARD_SAMPLE_RATES[-1]


def get_optimal_sample_rate(bandwidth_hz: float, sample_rate: int) -> int:
    """根据有效带宽得到匹配的真实/有效采样率（用于 48k 但有效频谱不多的场景）。

    与 ``get_estimated_sample_rate_for_resample`` 共用同一张标准采样率表
    （``STANDARD_SAMPLE_RATES``），避免两处口径不一致。
    """
    # 用 ceil 而非 int() 截断：bw=8000.24 时 2*bw=16000.48 > 16000，
    # 截断会错选 16000 违反 Nyquist；与 get_estimated_sample_rate_for_resample 的
    # 「2*bandwidth_hz <= sr」比较口径保持一致
    min_required_sr = math.ceil(bandwidth_hz * 2)
    for rate in STANDARD_SAMPLE_RATES:
        if rate >= min_required_sr and rate <= sample_rate:
            return rate
    return sample_rate
