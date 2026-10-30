"""Internal implementation of URGENT2026 audio simulation primitives.

This module provides audio augmentation functions for dynamic mixing without
requiring the external urgent2026_challenge_track1-main repository.

Core functions ported from:
  - urgent2026/simulation/simulate_data_from_param.py
  - urgent2026/simulation/rir_utils.py

DataLoader worker 的 Layer E codec 通过独立 ffmpeg 子进程执行，避免
torchaudio 内置旧版 FFmpeg native 崩溃直接终止 worker；非 worker 调用仍保留
torchaudio backend。

Author: xincheng.zhu (USE_Train framework)
Original source: URGENT2026 Challenge Baseline
License: Apache 2.0 (matching URGENT2026)
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np
import scipy.fft
import scipy.signal
import soundfile as sf

from core.data.simulation.energy_ratio import EnergyRatioScaler
from core.data.simulation.level_audit import LevelAudit

logger = logging.getLogger(__name__)

# fd 2 是进程级资源：并发进入抑制窗口时，后进入者若直接读回 fd 2 再恢复，
# 会把 stderr 永久指向 devnull。用可重入锁串行化整个窗口。
_STDERR_LOCK = threading.RLock()


@contextmanager
def _suppress_stderr():
    """Temporarily suppress stderr (used to silence torchaudio C++ vorbis warning).

    抑制作用于进程级 fd 2，窗口由 ``_STDERR_LOCK`` 串行化，因此同一窗口内其它
    线程的 stderr 也会被吞掉；并发 codec 调用只能串行执行。
    """
    with _STDERR_LOCK:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        try:
            os.dup2(devnull, 2)
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(devnull)
            os.close(old_stderr)


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int, res_type: str = "soxr_hq") -> np.ndarray:
    """Resample without resampy: soxr first, then librosa.

    Input ``(C, T)`` is converted to ``(T, C)`` for soxr (expects samples, channels).

    ``res_type`` is a soxr quality string (``soxr_hq``, ``soxr_vhq``, ...)
    or a librosa ``res_type`` (``kaiser_best``, ``polyphase``, ``fft``, ``scipy``).
    The default ``None`` or ``"soxr_*"`` forces soxr first; all other types skip soxr.
    """
    if orig_sr == target_sr:
        return audio
    need_transpose = audio.ndim == 2 and audio.shape[0] < audio.shape[1]

    use_soxr = (res_type is None or res_type.startswith("soxr_") or res_type == "polyphase")
    if use_soxr:
        # soxr expects (samples, channels) — transpose (C, T) → (T, C)
        _in = audio.T if need_transpose else audio
        try:
            import soxr
            quality = "HQ"
            if res_type and res_type.startswith("soxr_"):
                quality = res_type[5:].upper()  # "soxr_hq" → "HQ"
            out = soxr.resample(_in, orig_sr, target_sr, quality=quality)
            out = out.astype(np.float32)
            return out.T if need_transpose else out  # (T, C) → (C, T) if needed
        except ImportError:
            pass

    # Fallback / non-soxr types: use librosa (expects (C, T) or 1D)
    import librosa
    _rtype = "polyphase" if use_soxr else res_type
    if audio.ndim == 2:
        if need_transpose:
            audio = audio  # already (C, T) — no change needed
        out = np.stack(
            [librosa.resample(audio[c], orig_sr=orig_sr, target_sr=target_sr, res_type=_rtype)
             for c in range(audio.shape[0])],
            axis=0,
        )
        return out.astype(np.float32)
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr, res_type=_rtype).astype(np.float32)

# torchaudio 的 codec backend 延迟加载：DataLoader worker 不应加载旧版
# libav* 动态库，worker 路径改走外部 ffmpeg 子进程，避免 native 崩溃带走
# 整个 worker。测试可以继续通过这三个模块级 seam 注入 fake backend。
torch = None
AudioEffector = None
CodecConfig = None
HAS_TORCHAUDIO_IO: bool | None = None


def _in_dataloader_worker() -> bool:
    """返回当前进程是否为 PyTorch DataLoader worker。"""
    try:
        from torch.utils.data import get_worker_info
    except (ImportError, RuntimeError):
        return False
    return get_worker_info() is not None


def _load_torchaudio_io() -> bool:
    """按需加载 torchaudio codec backend，且保留测试注入的 fake seam。"""
    global AudioEffector, CodecConfig, HAS_TORCHAUDIO_IO, torch
    if HAS_TORCHAUDIO_IO is False:
        return False
    if AudioEffector is not None and CodecConfig is not None:
        if torch is None:
            import torch as torch_module

            torch = torch_module
        HAS_TORCHAUDIO_IO = True
        return True
    try:
        import torch as torch_module
        from torchaudio.io import AudioEffector as audio_effector
        from torchaudio.io import CodecConfig as codec_config
    except (ImportError, AttributeError, OSError):
        HAS_TORCHAUDIO_IO = False
        return False
    torch = torch_module
    AudioEffector = audio_effector
    CodecConfig = codec_config
    HAS_TORCHAUDIO_IO = True
    return True


# =============================================================================
# High-pass filter taps (DC removal at 70Hz)
# =============================================================================
def _filter_designs(fs: int, cutoff: int = 70, transition_width: int = 15, attenuation: int = 10) -> np.ndarray:
    """Design highpass FIR filter using firwin2.
    
    Args:
        fs: Sampling rate in Hz
        cutoff: Cutoff frequency in Hz
        transition_width: Transition band width in Hz
        attenuation: Stopband attenuation in dB
        
    Returns:
        Filter taps (1D array)
    """
    nyq = 0.5 * fs
    
    stop = cutoff - transition_width
    if stop < 0:
        stop = 0
        transition_width = cutoff
    
    pass_start = cutoff
    if pass_start > nyq:
        pass_start = nyq
    
    norm_stop = stop / nyq
    norm_pass = pass_start / nyq
    freq_points = [0, norm_stop, norm_pass, 1.0]
    gain_points = [0, 0, 1, 1]
    
    # numtaps ≈ (attenuation * fs) / (22 * transition_width)
    numtaps = int((attenuation * fs) / (22 * transition_width))
    numtaps = max(numtaps, 101)
    if numtaps % 2 == 0:
        numtaps += 1
    
    taps = scipy.signal.firwin2(numtaps, freq=freq_points, gain=gain_points)
    return taps


# Pre-compute highpass filter taps for common sample rates
HIGH_PASS_TAPS = {
    fs: _filter_designs(fs) for fs in [8000, 16000, 22050, 24000, 32000, 44100, 48000]
}


def get_highpass_taps(fs: int = 16000) -> np.ndarray:
    """Get pre-computed highpass filter taps for given sample rate.
    
    Args:
        fs: Sampling rate in Hz
        
    Returns:
        Filter taps
    """
    if fs not in HIGH_PASS_TAPS:
        return _filter_designs(fs)
    return HIGH_PASS_TAPS[fs]


def fast_filtfilt_fir(b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Zero-phase FIR filtering via fftconvolve.

    Workaround for scipy >=1.14 ``lfilter_zi`` pathology: when ``a=[1]``
    (FIR) and ``len(b)`` is large (~500), ``scipy.signal.filtfilt`` spends
    ~16s inside ``numpy.linalg.solve`` during initial-condition setup.
    This FFT-convolve replacement runs in ~25ms (640x speed-up) while
    matching the middle 90%% of the signal to >130dB SNR.
    """
    y_fwd = scipy.signal.fftconvolve(x, b, mode="same")
    y_rev = scipy.signal.fftconvolve(y_fwd[::-1], b, mode="same")
    return y_rev[::-1]


# =============================================================================
# Audio I/O
# =============================================================================
def read_audio(
    filename: str | Path,
    force_1ch: bool = True,
    fs: int | None = None,
    max_duration: int = -1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, int]:
    """Read audio file with optional resampling and cropping.
    
    Args:
        filename: Path to audio file
        force_1ch: If True, return mono audio (first channel)
        fs: Target sampling rate (if different from file, resample)
        max_duration: Maximum duration in samples (-1 = no limit)
        rng: 可选随机数生成器；未提供时使用全局 numpy RNG
        
    Returns:
        Tuple of (audio, sampling_rate)
        audio shape: (1, T) if force_1ch else (C, T)
    """
    audio, fs_ = sf.read(filename, always_2d=True)
    audio = audio[:, :1].T if force_1ch else audio.T  # (T, C) -> (C, T)
    
    if fs is not None and fs != fs_:
        audio = _resample_audio(audio, fs_, fs)
        fs_ = fs
    
    if max_duration > 0 and audio.shape[1] > max_duration:
        # 包含最后一个合法起点，避免末尾样本永远不可被裁到。
        crop_upper = audio.shape[1] - max_duration + 1
        start = int(rng.integers(crop_upper)) if rng is not None else np.random.randint(crop_upper)
        audio = audio[:, start : start + max_duration]
    
    return audio, fs_


def save_audio(audio: np.ndarray, filename: str | Path, fs: int):
    """Save audio to file.
    
    Args:
        audio: Audio array (1, T) or (T,)
        filename: Output path
        fs: Sampling rate in Hz
    """
    if audio.ndim != 1:
        audio = audio[0] if audio.shape[0] == 1 else audio.T
    sf.write(filename, audio, samplerate=fs)


# =============================================================================
# Non-silence detection (from espnet2)
# =============================================================================
def detect_non_silence(audio: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """Detect non-silent regions in audio.

    与 URGENT2026 track1 ``simulation/simulate_data_from_param.py`` 一致：赛道上由
    ``espnet2.train.preprocessor.detect_non_silence`` 提供；本仓库为 **无 espnet 依赖** 的
    轻量替代（逐采样 RMS、跨通道均值），供 ``mix_noise`` / ``wind_noise`` 估计非静音段功率。

    Args:
        audio: Audio array (C, T)
        threshold: RMS threshold for silence

    Returns:
        Boolean mask of non-silent samples (length T; 与 ``speech[:, mask]`` 索引配套)
    """
    # Simple RMS-based detection
    rms = np.sqrt(np.mean(audio ** 2, axis=0, keepdims=True))
    return (rms > threshold).ravel()


# =============================================================================
# Core augmentation functions
# =============================================================================
def _validate_finite_samples(name: str, samples: np.ndarray) -> None:
    """拒绝会把非有限值传播进能量计算或卷积的音频 samples。"""
    try:
        all_finite = bool(np.all(np.isfinite(samples)))
    except TypeError as exc:
        raise ValueError(f"{name} samples must be finite numeric values") from exc
    if not all_finite:
        raise ValueError(f"{name} samples must be finite")


def mix_noise(
    speech_sample: np.ndarray,
    noise_sample: np.ndarray,
    snr: float = 5.0,
    rng: np.random.Generator | None = None,
    snr_reference: np.ndarray | None = None,
    audit_out: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mix speech with additive noise at given SNR.
    
    Args:
        speech_sample: Speech audio (C, T); **output** is ``speech_sample + scaled_noise``.
        noise_sample: Noise audio (C, T)
        snr: Signal-to-noise ratio in dB (noise power vs reference active-region power)
        rng: Random number generator (default: use global RNG)
        snr_reference: If set, active-region power for SNR scaling is taken from this array
            (same shape as ``speech_sample``), e.g. early-RIR **clean_target** while still
            mixing into full-RIR ``speech_sample``. If ``None``, uses ``speech_sample``.
        audit_out: 可选审计容器；仅请求时写入 active-power calibration 数值。
        
    Returns:
        Tuple of (noisy_speech, scaled_noise), both (C, T)
    """
    try:
        snr_value = float(snr)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"mix_noise snr must be finite, got {snr!r}") from exc
    if not np.isfinite(snr_value):
        raise ValueError(f"mix_noise snr must be finite, got {snr!r}")

    _validate_finite_samples("mix_noise speech_sample", speech_sample)
    _validate_finite_samples("mix_noise noise_sample", noise_sample)
    if snr_reference is not None:
        _validate_finite_samples("mix_noise snr_reference", snr_reference)

    if rng is None:
        rng = np.random.default_rng()

    len_speech = speech_sample.shape[-1]
    len_noise = noise_sample.shape[-1]
    if len_speech == 0:
        return speech_sample, noise_sample[:, :0] if noise_sample.shape[-1] > 0 else speech_sample

    if len_noise != len_speech:
        raise ValueError(
            f"mix_noise expects pre-aligned noise (see DynamicMixingDataset._align_noise_2d_to_length): "
            f"speech T={len_speech}, noise T={len_noise}"
        )

    ref = speech_sample if snr_reference is None else snr_reference
    if snr_reference is not None:
        if ref.shape[0] != speech_sample.shape[0]:
            raise ValueError(
                f"mix_noise: snr_reference channels {ref.shape} vs speech_sample {speech_sample.shape}"
            )
        T_sp = int(speech_sample.shape[1])
        if ref.shape[1] < T_sp:
            raise ValueError(
                f"mix_noise: snr_reference T={ref.shape[1]} shorter than speech_sample T={T_sp}"
            )
        if ref.shape[1] > T_sp:
            ref = ref[:, :T_sp]

    # active mask 仍沿用原 dtype 判定；数学 Module 内固定 float64 累加。
    with np.errstate(over="ignore", invalid="ignore"):
        speech_mask = detect_non_silence(ref)
        noise_mask = detect_non_silence(noise_sample)
    try:
        power_speech = EnergyRatioScaler.measure(
            ref,
            policy="active_mean_power",
            active_mask=speech_mask,
        )
        power_noise = EnergyRatioScaler.measure(
            noise_sample,
            policy="active_mean_power",
            active_mask=noise_mask,
        )
    except ValueError as exc:
        raise ValueError("mix_noise active power must be finite") from exc

    # 先校验原始 gain，再保留现有 1e4 上限；Adapter 保持乘法与相加顺序。
    try:
        ratio = EnergyRatioScaler.power_ratio(
            power_speech,
            power_noise,
            snr_value,
            policy="active_mean_power",
            denominator_floor=1e-10,
            gain_cap=1e4,
        )
    except ValueError as exc:
        raise ValueError(f"mix_noise scale must be finite, got {exc}") from exc
    power_noise = ratio.denominator_energy
    scale = ratio.gain
    noise = scale * noise_sample
    noisy_speech = speech_sample + noise
    if audit_out is not None:
        audit_out.update({
            "nominal_snr_db": float(snr),
            "reference_active_power": float(power_speech),
            "noise_active_power_before_scale": float(power_noise),
            "noise_scale": float(scale),
            "calibration_realized_snr_db": float(ratio.realized_db),
            "level_audit": LevelAudit(
                name="mixture.snr",
                candidate_drawn_db=float(snr),
                selected_nominal_db=float(snr),
                calibration_realized_db=float(ratio.realized_db),
                final_realized_db=None,
                numerator="snr_reference_active",
                denominator="scaled_additive_noise_active",
                policy="active_mean_power",
                calibration_measure_point="pre_mix_active_calibration",
                final_measure_point="not_measured",
                applicable=False,
                reason="final_not_measured",
                observations={
                    "numerator_energy": float(power_speech),
                    "denominator_energy_before_scale": float(power_noise),
                    "gain": float(scale),
                },
            ).to_dict(),
        })
    
    return noisy_speech, noise


def _validate_rir_time_parameters(*, early_rir_sec: Real, fs: Real) -> tuple[float, float]:
    """校验 early RIR 时间参数，拒绝 bool、复数及非有限值。"""
    if isinstance(early_rir_sec, (bool, np.bool_)) or not isinstance(early_rir_sec, Real):
        raise TypeError(
            f"early_rir_sec must be a real number, got {type(early_rir_sec).__name__}"
        )
    if isinstance(fs, (bool, np.bool_)) or not isinstance(fs, Real):
        raise TypeError(f"fs must be a real number, got {type(fs).__name__}")
    try:
        early_value = float(early_rir_sec)
        fs_value = float(fs)
    except (OverflowError, ValueError) as error:
        raise ValueError("early_rir_sec and fs must be finite") from error
    if not np.isfinite(early_value) or early_value < 0:
        raise ValueError(f"early_rir_sec must be finite and >= 0, got {early_rir_sec!r}")
    if not np.isfinite(fs_value) or fs_value <= 0:
        raise ValueError(f"fs must be finite and > 0, got {fs!r}")
    early_samples = early_value * fs_value
    if not np.isfinite(early_samples) or early_samples > np.iinfo(np.intp).max:
        raise ValueError(
            "early_rir_sec * fs must fit the platform sample-index range"
        )
    return early_value, fs_value


def _normalize_rir_with_peak(
    rir_sample: np.ndarray, normalize_rir: bool, peak: float | None
) -> np.ndarray:
    """按给定 full-RIR 峰值归一化，保留 full/early 相对增益。"""
    if not normalize_rir or peak is None or peak <= 1e-8:
        return rir_sample
    return rir_sample / peak


def add_paired_reverberation(
    speech_sample: np.ndarray,
    rir_sample: np.ndarray,
    early_rir_sec: float,
    *,
    fs: int,
    normalize_rir: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """用一次 speech FFT 生成 full/early 静态 mono RIR 卷积结果。

    ``normalize_rir=True`` 时 full 与 early 共用 full-RIR peak，避免 early
    独立归一化破坏二者相对增益；moving RIR 不应调用本函数。返回值保持统一的
    浮点 ``np.result_type`` dtype、speech shape 与原 ``scipy.signal.convolve``
    的前缀截断。
    """
    if speech_sample.ndim != 2 or speech_sample.shape[0] != 1:
        raise ValueError(
            f"add_paired_reverberation expects mono speech (1, T), got shape {speech_sample.shape}"
        )
    if rir_sample.ndim != 2 or rir_sample.shape[0] != 1:
        raise ValueError(
            f"add_paired_reverberation expects mono RIR (1, T_rir), got shape {rir_sample.shape}"
        )
    early_rir_sec, fs = _validate_rir_time_parameters(
        early_rir_sec=early_rir_sec, fs=fs
    )
    _validate_finite_samples("add_paired_reverberation speech_sample", speech_sample)
    _validate_finite_samples("add_paired_reverberation rir_sample", rir_sample)
    output_dtype = np.result_type(speech_sample.dtype, rir_sample.dtype)
    if not np.issubdtype(output_dtype, np.floating):
        output_dtype = np.dtype(np.float32)
    if speech_sample.shape[1] == 0:
        empty = np.empty(speech_sample.shape, dtype=output_dtype)
        return empty, empty.copy()
    if rir_sample.shape[1] == 0:
        empty = np.zeros(speech_sample.shape, dtype=output_dtype)
        return empty, empty.copy()

    early_rir = estimate_early_rir(rir_sample, early_rir_sec=early_rir_sec, fs=fs)
    nfft = scipy.fft.next_fast_len(speech_sample.shape[1] + rir_sample.shape[1] - 1)
    work_dtype = np.result_type(output_dtype, np.float32)
    speech_fft = scipy.fft.rfft(np.asarray(speech_sample[0], dtype=work_dtype), n=nfft)
    full_peak = float(np.max(np.abs(rir_sample)))
    full_norm = _normalize_rir_with_peak(rir_sample, normalize_rir, full_peak)
    early_norm = _normalize_rir_with_peak(early_rir, normalize_rir, full_peak)
    full_fft = scipy.fft.rfft(
        np.asarray(full_norm[0], dtype=np.result_type(full_norm.dtype, np.float32)),
        n=nfft,
    )
    early_fft = scipy.fft.rfft(
        np.asarray(early_norm[0], dtype=np.result_type(early_norm.dtype, np.float32)),
        n=nfft,
    )
    full = scipy.fft.irfft(speech_fft * full_fft, n=nfft)[: speech_sample.shape[1]]
    early = scipy.fft.irfft(speech_fft * early_fft, n=nfft)[: speech_sample.shape[1]]
    return (
        np.asarray(full, dtype=output_dtype).reshape(1, -1),
        np.asarray(early, dtype=output_dtype).reshape(1, -1),
    )


def add_reverberation(
    speech_sample: np.ndarray,
    rir_sample: np.ndarray,
    normalize_rir: bool = True,
) -> np.ndarray:
    """Apply reverberation via RIR convolution.

    Args:
        speech_sample: Speech audio (1, T)
        rir_sample: Room impulse response (1, T_rir)，单通道 RIR（多通道文件需先下混或拆文件）
        normalize_rir: If True (默认, 生产配置), 卷积前把 RIR 峰值归一化到 1.0，消除不同
            RIR 录音的增益差异。If False, 用原始 RIR 增益——对齐官方 PASE/StuPASE
            ``simulate_utils.add_reverberation``（不归一化），保持 full-RIR(noisy) 与
            early-RIR(target) 共用同一原始增益，相对能量由 RIR 物理特性决定（审计 D-05）。
            注意：当 full_peak == early_peak（direct path 是峰值，SLR26 类 RIR 常见）时，
            归一化不改变 full/early 相对增益，只重缩放绝对能量（后续被 peak guard 吸收）；
            normalize_rir=False 严格对齐官方绝对 scaling，并消除 late-reflection>direct-path
            时相对增益被破坏的条件性风险。

    Returns:
        Reverberant speech (1, T)
    """
    if speech_sample.ndim != 2 or speech_sample.shape[0] != 1:
        raise ValueError(
            f"add_reverberation expects mono speech (1, T), got shape {speech_sample.shape}"
        )
    if rir_sample.ndim != 2 or rir_sample.shape[0] != 1:
        raise ValueError(
            f"add_reverberation expects mono RIR (1, T_rir), got shape {rir_sample.shape}"
        )
    _validate_finite_samples("add_reverberation speech_sample", speech_sample)
    _validate_finite_samples("add_reverberation rir_sample", rir_sample)
    if speech_sample.shape[1] == 0:
        return speech_sample.copy()

    if normalize_rir:
        # Peak-normalize RIR to remove arbitrary recording-gain differences
        rir_peak = float(np.max(np.abs(rir_sample)))
        rir_norm = rir_sample / rir_peak if rir_peak > 1e-8 else rir_sample
    else:
        # 官方 PASE/StuPASE: 用原始 RIR 增益，保持 full/early 相对增益（审计 D-05）
        rir_norm = rir_sample

    reverberant = scipy.signal.convolve(speech_sample, rir_norm, mode="full")
    return reverberant[:, : speech_sample.shape[1]]


def add_moving_reverberation(
    speech_sample: np.ndarray,
    rir_seq: np.ndarray,
    segment_sec: float,
    fs: int,
    normalize_rir: bool = True,
    normalization_peak: float | None = None,
) -> np.ndarray:
    """移动声源时变混响：speech 按 ``segment_sec`` 分块，每块用对应段 RIR 卷积后按绝对时间轴 overlap-add。

    模拟说话人（或麦克风）在房间内移动：``rir_seq`` 为 ``[T_seg, n_rir]`` 的分段
    时变 RIR 序列（由 gpuRIR 移动仿真生成，pkl ``moving: True``）。语音超过轨迹长度
    的部分复用最后一段 RIR（声源停在终点）；各段保留 ``n_h - 1`` 的完整卷积尾音并在
    绝对时间轴上相加，因此不存在段间 click，也不会丢失晚期混响。

    Args:
        speech_sample: Speech audio (1, T)
        rir_seq: 移动 RIR 序列 (T_seg, n_rir)，每行是一条静态 RIR
        segment_sec: 每段时间（秒），与生成侧一致
        fs: 采样率
        normalize_rir: 是否用完整 full sequence 的共同峰值归一化，保留跨段增益
        normalization_peak: 可选共同 full-sequence peak；early sequence 复用该值，
            保持 full/early 在同一尺度。仅在 ``normalize_rir=True`` 时生效。

    Returns:
        Moving-reverberant speech (1, T)
    """
    if speech_sample.ndim != 2 or speech_sample.shape[0] != 1:
        raise ValueError(
            f"add_moving_reverberation expects mono speech (1, T), got shape {speech_sample.shape}"
        )
    if rir_seq.ndim != 2:
        raise ValueError(
            f"add_moving_reverberation expects RIR sequence [T_seg, n_rir], got shape {rir_seq.shape}"
        )
    _validate_finite_samples("add_moving_reverberation speech_sample", speech_sample)
    _validate_finite_samples("add_moving_reverberation rir_seq", rir_seq)
    T = speech_sample.shape[1]
    T_seg, n_h = rir_seq.shape
    if T == 0 or T_seg == 0:
        return speech_sample.copy()

    seg_len = max(1, int(round(segment_sec * fs)))
    n_seg = int(np.ceil(T / seg_len))
    if normalize_rir and normalization_peak is None:
        normalization_peak = float(np.max(np.abs(rir_seq)))
    elif normalize_rir:
        try:
            normalization_peak = float(normalization_peak)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"normalization_peak must be finite, got {normalization_peak!r}"
            ) from exc
        if not np.isfinite(normalization_peak):
            raise ValueError(
                f"normalization_peak must be finite, got {normalization_peak!r}"
            )

    # 每段 full convolution 都按绝对时间轴 overlap-add，完整尾音长度必须为
    # ``n_h - 1``；这也保证恒定 RIR 序列严格退化为静态截断卷积。
    out = np.zeros((1, T + n_h - 1), dtype=np.float32)
    for i in range(n_seg):
        start = i * seg_len
        end = min(start + seg_len, T)
        s = speech_sample[:, start:end]
        if s.shape[1] == 0:
            break
        h = rir_seq[min(i, T_seg - 1)]
        h = _normalize_rir_with_peak(h, normalize_rir, normalization_peak)
        conv = scipy.signal.convolve(s, h.reshape(1, -1), mode="full")
        out[:, start : start + conv.shape[1]] += conv.astype(np.float32, copy=False)

    return out[:, :T]


def bandwidth_limitation(
    speech_sample: np.ndarray,
    fs: int,
    fs_new: int,
    res_type: str = "soxr_hq",
) -> np.ndarray:
    """Apply bandwidth limitation via resampling.
    
    Args:
        speech_sample: Speech audio (C, T)
        fs: Original sampling rate in Hz
        fs_new: Effective (limited) sampling rate in Hz
        res_type: Resampling method (librosa). Default: 'soxr_hq' (fast, high quality)
                  Alternatives: 'kaiser_best', 'kaiser_fast', 'fft', 'scipy'
        
    Returns:
        Bandwidth-limited speech (C, T)
        
    Note:
        Recommended to use 'soxr_hq' (fastest) or 'kaiser_best' (best quality).
        Install soxr: pip install soxr
    """
    if fs == fs_new:
        return speech_sample
    
    if not fs > fs_new:
        raise ValueError(f"fs ({fs}) must be greater than fs_new ({fs_new})")
    
    # Downsample to fs_new then upsample back (soxr/librosa, no resampy)
    downsampled = _resample_audio(speech_sample, fs, fs_new, res_type=res_type)
    upsampled = _resample_audio(downsampled, fs_new, fs, res_type=res_type)
    n_time = speech_sample.shape[1]
    if upsampled.shape[1] >= n_time:
        return upsampled[:, :n_time]
    # Pad if resampling rounded down (avoid returning shorter than input)
    pad_width = ((0, 0), (0, n_time - upsampled.shape[1]))
    return np.pad(upsampled, pad_width, mode="constant", constant_values=0)


def clipping_aug(
    speech_sample: np.ndarray,
    min_quantile: float = 0.0,
    max_quantile: float = 0.9,
) -> np.ndarray:
    """Apply amplitude clipping based on quantiles.
    
    Args:
        speech_sample: Speech audio (C, T)
        min_quantile: Lower quantile threshold (0-1)
        max_quantile: Upper quantile threshold (0-1)
        
    Returns:
        Clipped speech (C, T)
    """
    q = np.array([min_quantile, max_quantile])
    min_vals, max_vals = np.quantile(speech_sample, q, axis=-1, keepdims=False)
    min_vals, max_vals = np.minimum(min_vals, max_vals), np.maximum(min_vals, max_vals)

    # Per-channel clipping
    clipped = np.stack(
        [
            np.clip(speech_sample[i], min_vals[i], max_vals[i])
            for i in range(speech_sample.shape[0])
        ],
        axis=0,
    )
    return clipped


@dataclass(frozen=True)
class CodecCompressionResult:
    """Codec 执行结果。"""

    waveform: np.ndarray
    observation_changed: bool
    reason: str | None = None


def _codec_ffmpeg_timeout_sec() -> float:
    """读取 worker codec 子进程超时，默认 60 秒。"""
    raw = os.environ.get("USE_TRAIN_CODEC_TIMEOUT_SEC", "60")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"USE_TRAIN_CODEC_TIMEOUT_SEC must be a number, got {raw!r}"
        ) from exc
    if not np.isfinite(timeout) or timeout <= 0.0:
        raise ValueError(
            "USE_TRAIN_CODEC_TIMEOUT_SEC must be a finite positive number, "
            f"got {raw!r}"
        )
    return timeout


def _codec_ffmpeg_spec(
    format: str,
    encoder: str | None,
    qscale: int | None,
    channels: int,
) -> tuple[str, tuple[str, ...], str]:
    """返回外部 ffmpeg 的编码器、质量参数和容器。"""
    quality = 5 if qscale is None else int(qscale)
    if format == "mp3":
        return "libmp3lame", ("-q:a", str(max(0, min(10, quality)))), "mp3"
    if encoder in (None, "vorbis"):
        return "libvorbis", ("-q:a", str(max(-1, min(10, quality)))), "ogg"
    # libopus 不接受与 torchaudio CodecConfig 同口径的 qscale；按通道数
    # 映射到稳定的总码率，仍保留质量档位越高码率越高的单调关系。
    if encoder == "opus":
        per_channel_kbps = 16 + (max(1, min(10, quality)) - 1) * 8
        bitrate_kbps = max(16, min(512, per_channel_kbps * max(1, channels)))
        return "libopus", ("-b:a", f"{bitrate_kbps}k"), "ogg"
    raise ValueError(f"Unsupported encoder for ffmpeg codec backend: {encoder}")


def _codec_compression_ffmpeg(
    speech_sample: np.ndarray,
    inp_ct: np.ndarray,
    fs: int,
    format: str,
    encoder: str | None,
    qscale: int | None,
) -> CodecCompressionResult:
    """在独立 ffmpeg 进程中执行 codec，隔离 DataLoader worker 的 native 库。"""
    n_ch, n_time = inp_ct.shape
    try:
        codec, quality_args, container = _codec_ffmpeg_spec(
            format, encoder, qscale, n_ch
        )
        common = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-threads",
            "1",
        ]
        encoded = subprocess.run(
            [
                *common,
                "-f",
                "f32le",
                "-ar",
                str(fs),
                "-ac",
                str(n_ch),
                "-i",
                "pipe:0",
                "-c:a",
                codec,
                *quality_args,
                "-f",
                container,
                "pipe:1",
            ],
            input=np.ascontiguousarray(inp_ct.T).tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=_codec_ffmpeg_timeout_sec(),
        )
        if encoded.returncode != 0 or not encoded.stdout:
            detail = encoded.stderr.decode("utf-8", errors="replace")[-300:]
            raise RuntimeError(
                f"ffmpeg encode failed (code={encoded.returncode}): {detail}"
            )
        decoded = subprocess.run(
            [
                *common,
                "-f",
                container,
                "-i",
                "pipe:0",
                "-f",
                "f32le",
                "-ar",
                str(fs),
                "-ac",
                str(n_ch),
                "-c:a",
                "pcm_f32le",
                "pipe:1",
            ],
            input=encoded.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=_codec_ffmpeg_timeout_sec(),
        )
        if decoded.returncode != 0 or not decoded.stdout:
            detail = decoded.stderr.decode("utf-8", errors="replace")[-300:]
            raise RuntimeError(
                f"ffmpeg decode failed (code={decoded.returncode}): {detail}"
            )
        decoded_values = np.frombuffer(decoded.stdout, dtype=np.float32)
        frame_count = decoded_values.size // n_ch
        if frame_count < 1:
            raise RuntimeError("ffmpeg decode returned no complete frames")
        output = np.ascontiguousarray(
            decoded_values[: frame_count * n_ch].reshape(frame_count, n_ch).T
        )
    except FileNotFoundError as error:
        logger.warning("Codec compression disabled: ffmpeg is unavailable: %s", error)
        return CodecCompressionResult(speech_sample, False, "dependency_unavailable")
    except subprocess.TimeoutExpired as error:
        logger.warning(
            "Codec compression timed out (%s, %s, qscale=%s): %s",
            format,
            encoder,
            qscale,
            error,
        )
        return CodecCompressionResult(speech_sample, False, "codec_error")
    except Exception as error:
        logger.warning(
            "Codec compression failed in ffmpeg worker backend (%s, %s, qscale=%s): %s",
            format,
            encoder,
            qscale,
            error,
        )
        return CodecCompressionResult(speech_sample, False, "codec_error")

    # 对齐长度；外部 ffmpeg 的编码帧可能比输入多出尾部 padding。
    out_time = output.shape[1]
    if out_time < n_time:
        output = np.pad(
            output,
            ((0, 0), (0, n_time - out_time)),
            mode="constant",
            constant_values=0,
        )
    elif out_time > n_time:
        output = output[:, :n_time]
    changed = output.shape != inp_ct.shape or not np.array_equal(output, inp_ct)
    return CodecCompressionResult(output, changed, None if changed else "no_change")


def codec_compression_result(
    speech_sample: np.ndarray,
    fs: int,
    format: str,
    encoder: str | None = None,
    qscale: int | None = None,
) -> CodecCompressionResult:
    """Apply codec compression and report whether the observation changed."""
    if format not in ("mp3", "ogg"):
        raise ValueError(f"Unsupported format: {format}")
    if encoder not in (None, "None", "vorbis", "opus"):
        raise ValueError(f"Unsupported encoder: {encoder}")

    encoder = None if encoder == "None" else encoder

    # 空输入不能形成有效 codec observation，且无需探测可选依赖。
    if speech_sample.size == 0:
        return CodecCompressionResult(speech_sample, False, "empty")
    if speech_sample.ndim == 2 and speech_sample.shape[0] == 0:
        return CodecCompressionResult(speech_sample, False, "empty")

    # Build (C, T) for both backends; AudioEffector and ffmpeg consume
    # (T, C) only at their respective call boundaries.
    if speech_sample.ndim == 2:
        inp_ct = np.ascontiguousarray(speech_sample.astype(np.float32))
    else:
        inp_ct = np.ascontiguousarray(
            np.reshape(speech_sample, (1, -1)).astype(np.float32)
        )
    n_ch, n_time = inp_ct.shape
    if n_ch == 0 or n_time == 0:
        return CodecCompressionResult(speech_sample, False, "empty")
    if format == "ogg" and encoder == "vorbis" and qscale is not None and qscale == 0:
        qscale = 1

    # DataLoader worker 内不进入 torchaudio 自带的旧版 FFmpeg 动态库。该库
    # 的 native 崩溃无法由 Python try/except 捕获，必须由外部进程隔离。
    if _in_dataloader_worker():
        return _codec_compression_ffmpeg(
            speech_sample,
            inp_ct,
            fs,
            format,
            encoder,
            qscale,
        )

    if not _load_torchaudio_io():
        import warnings
        warnings.warn(
            "torchaudio.io not available, codec_compression disabled. "
            "Install torchaudio >= 2.1.0 for codec support.",
            UserWarning,
        )
        return CodecCompressionResult(
            speech_sample, False, "dependency_unavailable"
        )

    # (C, T) -> (T, C) for torchaudio FFmpeg backend
    inp_tc = np.ascontiguousarray(inp_ct.T)

    try:
        module = AudioEffector(
            format=format,
            encoder=encoder,
            codec_config=CodecConfig(qscale=qscale),
            pad_end=True,
        )
        # torchaudio >=2.4 unconditionally prints a C++ warning when vorbis is
        # selected ("Enabling '-strict experimental'").  The encoder works fine;
        # we just suppress the stderr spam during apply().
        with _suppress_stderr():
            out_tc = module.apply(torch.from_numpy(inp_tc), fs).numpy()
        if out_tc.ndim == 1:
            output = np.reshape(out_tc, (1, -1))
        else:
            output = np.ascontiguousarray(out_tc.T)  # (T, C) -> (C, T)
    except Exception as error:
        logger.warning(
            "Codec compression failed (%s, %s, qscale=%s): %s",
            format,
            encoder,
            qscale,
            error,
            exc_info=True,
        )
        return CodecCompressionResult(speech_sample, False, "codec_error")

    # Align length; output is (C, T)
    out_time = output.shape[1]
    if out_time < n_time:
        output = np.pad(output, ((0, 0), (0, n_time - out_time)), mode="constant", constant_values=0)
    elif out_time > n_time:
        output = output[:, :n_time]
    changed = output.shape != inp_ct.shape or not np.array_equal(output, inp_ct)
    return CodecCompressionResult(
        output,
        changed,
        None if changed else "no_change",
    )


def packet_loss(
    speech_sample: np.ndarray,
    fs: int,
    packet_loss_indices: list[int],
    packet_duration_ms: int = 20,
) -> np.ndarray:
    """Simulate packet loss by zeroing specific time intervals.
    
    Args:
        speech_sample: Speech audio (C, T)
        fs: Sampling rate in Hz
        packet_loss_indices: List of packet indices to drop
        packet_duration_ms: Packet duration in milliseconds
        
    Returns:
        Speech with packet loss (C, T)
    """
    speech_sample = speech_sample.copy()
    n_samples = speech_sample.shape[1]
    for idx in packet_loss_indices:
        start = idx * packet_duration_ms * fs // 1000
        end = (idx + 1) * packet_duration_ms * fs // 1000
        if start >= n_samples:
            continue
        end = min(end, n_samples)
        speech_sample[:, start:end] = 0
    return speech_sample


def _hermite_interpolation(
    x: np.ndarray, x0: float, x1: float, p0: float, p1: float, m0: float, m1: float
) -> np.ndarray:
    """Cubic Hermite 插值，与 ffmpeg ``libavfilter/hermite.h`` 逐式一致。"""
    width = x1 - x0
    t = (x - x0) / width
    m0 = m0 * width
    m1 = m1 * width
    t2 = t * t
    t3 = t2 * t
    ct0 = p0
    ct1 = m0
    ct2 = -3 * p0 - 2 * m0 + 3 * p1 - m1
    ct3 = 2 * p0 + m0 - 2 * p1 + m1
    return ct3 * t3 + ct2 * t2 + ct1 * t + ct0


def _sidechain_envelope_follower(
    level: np.ndarray, attack_coeff: float, release_coeff: float
) -> np.ndarray:
    """ffmpeg sidechaincompress 同款单极点包络跟随器（电平上升走 attack，下降走 release）。

    系数依赖当前电平与包络的大小关系，是逐样本递归，无法向量化；纯 float
    标量循环 64k 样本约 30ms，远快于每样本 spawn ffmpeg + 3 个临时 WAV 的 I/O。
    """
    env = np.empty_like(level)
    e = 0.0
    for i in range(level.shape[0]):
        v = level[i]
        e += (v - e) * (attack_coeff if v > e else release_coeff)
        env[i] = e
    return env


def _sidechain_compress(
    speech: np.ndarray,
    sidechain: np.ndarray,
    fs: int,
    threshold: float,
    ratio: float,
    attack: float,
    release: float,
    sc_gain: float,
    knee: float = 2.82843,
) -> np.ndarray:
    """内存版 sidechain 压缩，逐参数对齐 ffmpeg ``af_sidechaincompress.c``。

    等价于滤镜 ``sidechaincompress=threshold=...:ratio=...:level_sc=...:
    attack=...:release=...``，未显式参数取 ffmpeg 默认值：mode=downward,
    link=average, detection=rms, makeup=1, mix=1, level_in=1, knee=2.82843。

    Args:
        speech: 主输入 (C, T)，float64
        sidechain: 侧链输入 (C, T)，float64
        fs: 采样率
        threshold: 线性幅度阈值（ffmpeg threshold 为线性，非 dB）
        ratio: 压缩比
        attack/release: 包络时间常数 (ms)
        sc_gain: 侧链增益（ffmpeg level_sc）
        knee: 膝点宽度（线性比，ffmpeg 默认 2.82843）

    Returns:
        压缩后的 speech (C, T)，float64
    """
    # ── 常量（对应 compressor_config_output）──
    thres = np.log(threshold)
    lin_knee_start = threshold / np.sqrt(knee)
    lin_knee_stop = threshold * np.sqrt(knee)
    adj_knee_start = lin_knee_start ** 2  # detection=rms 在平方域判定膝点
    knee_start = np.log(lin_knee_start)
    knee_stop = np.log(lin_knee_stop)
    compressed_knee_stop = (knee_stop - thres) / ratio + thres
    attack_coeff = min(1.0, 1.0 / (attack * fs / 4000.0))
    release_coeff = min(1.0, 1.0 / (release * fs / 4000.0))

    # ── 电平检测：link=average（通道间 |x| 平均），detection=rms（整体取平方）──
    level = np.mean(np.abs(sidechain * sc_gain), axis=0) ** 2  # (T,)
    env = _sidechain_envelope_follower(level, attack_coeff, release_coeff)

    # ── 增益计算（对应 output_gain，mode=downward）──
    gain = np.ones_like(env)
    active = (env > 0.0) & (env > adj_knee_start)
    slope = np.zeros_like(env)
    slope[active] = 0.5 * np.log(env[active])  # detection=rms → log 斜率减半
    g = (slope - thres) / ratio + thres
    knee_mask = active & (knee > 1.0) & (slope < knee_stop)
    if np.any(knee_mask):
        g[knee_mask] = _hermite_interpolation(
            slope[knee_mask], knee_start, knee_stop,
            knee_start, compressed_knee_stop, 1.0, 1.0 / ratio,
        )
    gain[active] = np.exp(g[active] - slope[active])

    return speech * gain  # level_in=1, makeup=1, mix=1


def wind_noise(
    speech_sample: np.ndarray,
    noise_sample: np.ndarray,
    fs: int,
    threshold: float,
    ratio: float,
    attack: float,
    release: float,
    sc_gain: float,
    clipping: bool,
    clipping_threshold: float,
    snr: float,
    rng: np.random.Generator | None = None,
    snr_reference: np.ndarray | None = None,
    audit_out: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply wind noise with in-memory sidechain compression.

    非加性叠加：side-chain 压缩直接改变 ``speech_sample`` 波形本身，输出不再是
    "语音 + 可分离噪声分量"。基于未失真（undistorted）target 的 loss 与
    stem 分解在该路径下会系统性失配，见 SimulationConfigs 的 C-2 说明。

    Args:
        speech_sample: Speech audio (C, T)
        noise_sample: Wind noise audio (C, T)
        fs: Sampling rate in Hz
        threshold: Compression threshold（线性幅度，与 ffmpeg 一致，非 dB）
        ratio: Compression ratio
        attack: Attack time (ms)
        release: Release time (ms)
        sc_gain: Sidechain gain（线性，ffmpeg level_sc）
        clipping: Whether to apply clipping
        clipping_threshold: Clipping threshold (0-1)
        snr: Signal-to-noise ratio (dB)
        rng: Random number generator
        snr_reference: Optional (C, T) same shape as ``speech_sample``; wind gain uses its
            active-region power instead of ``speech_sample`` (e.g. clean_target vs full RIR mic).

    Returns:
        Tuple of (noisy_speech, scaled_noise), both (C, T)。``scaled_noise``
        等于实际混入 ``noisy_speech`` 的风噪分量（amix 减半后的量）。
    """
    _validate_finite_samples("wind_noise speech_sample", speech_sample)
    _validate_finite_samples("wind_noise noise_sample", noise_sample)
    if snr_reference is not None:
        _validate_finite_samples("wind_noise snr_reference", snr_reference)

    numeric_params = {
        "fs": fs,
        "threshold": threshold,
        "ratio": ratio,
        "attack": attack,
        "release": release,
        "sc_gain": sc_gain,
        "clipping_threshold": clipping_threshold,
        "snr": snr,
    }
    validated_params: dict[str, float] = {}
    for name, value in numeric_params.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(
                f"wind_noise {name} must be a finite real number, got {value!r}"
            )
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            raise ValueError(f"wind_noise {name} must be finite, got {value!r}")
        validated_params[name] = numeric_value
    fs = validated_params["fs"]
    threshold = validated_params["threshold"]
    ratio = validated_params["ratio"]
    attack = validated_params["attack"]
    release = validated_params["release"]
    sc_gain = validated_params["sc_gain"]
    clipping_threshold = validated_params["clipping_threshold"]
    snr = validated_params["snr"]

    if fs <= 0.0:
        raise ValueError(f"wind_noise fs must be positive, got {fs!r}")
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"wind_noise threshold must be in (0.0, 1.0], got {threshold!r}")
    if ratio < 1.0:
        raise ValueError(f"wind_noise ratio must be at least 1.0, got {ratio!r}")
    for name, value in (("attack", attack), ("release", release), ("sc_gain", sc_gain)):
        if value <= 0.0:
            raise ValueError(f"wind_noise {name} must be positive, got {value!r}")
    if not 0.0 <= clipping_threshold <= 1.0:
        raise ValueError(
            "wind_noise clipping_threshold must be in [0.0, 1.0], "
            f"got {clipping_threshold!r}"
        )

    if rng is None:
        rng = np.random.default_rng()

    len_speech = speech_sample.shape[-1]
    len_noise = noise_sample.shape[-1]
    if len_noise != len_speech:
        raise ValueError(
            f"wind_noise expects pre-aligned noise (see DynamicMixingDataset._align_noise_2d_to_length): "
            f"speech T={len_speech}, noise T={len_noise}"
        )

    ref_for_pow = speech_sample if snr_reference is None else snr_reference
    if snr_reference is not None:
        if ref_for_pow.shape[0] != speech_sample.shape[0]:
            raise ValueError(
                f"wind_noise: snr_reference channels {ref_for_pow.shape} vs speech_sample {speech_sample.shape}"
            )
        T_sp = int(speech_sample.shape[1])
        if ref_for_pow.shape[1] < T_sp:
            raise ValueError(
                f"wind_noise: snr_reference T={ref_for_pow.shape[1]} shorter than speech_sample T={T_sp}"
            )
        if ref_for_pow.shape[1] > T_sp:
            ref_for_pow = ref_for_pow[:, :T_sp]

    speech_active = ref_for_pow[:, detect_non_silence(ref_for_pow)]
    noise_active = noise_sample[:, detect_non_silence(noise_sample)]
    power_speech = np.mean(speech_active ** 2) if speech_active.size > 0 else 1e-10
    power_noise = np.mean(noise_active ** 2) if noise_active.size > 0 else 1e-10
    ratio_result = EnergyRatioScaler.power_ratio(
        power_speech,
        power_noise,
        snr,
        policy="active_mean_power",
        denominator_floor=1e-10,
        preserve_scalar_dtype=True,
    )
    scale = ratio_result.gain
    noise = scale * noise_sample
    if audit_out is not None:
        audit_out.update(
            LevelAudit(
                name="mixture.wind_snr",
                candidate_drawn_db=float(snr),
                selected_nominal_db=float(snr),
                calibration_realized_db=float(ratio_result.realized_db),
                final_realized_db=None,
                numerator="wind_snr_reference_active",
                denominator="scaled_wind_active_before_amix",
                policy="active_mean_power_legacy_dtype",
                calibration_measure_point="pre_sidechain_pre_amix",
                final_measure_point="not_measured",
                applicable=False,
                reason="final_not_measured",
                observations={
                    "numerator_energy": float(power_speech),
                    "denominator_energy_before_scale": float(power_noise),
                    "gain": float(scale),
                },
            ).to_dict()
        )
    
    # 归一化防止压缩时削顶
    scale_norm = 0.9 / max(np.max(np.abs(speech_sample)), np.max(np.abs(noise)), 1e-10)
    speech_norm = speech_sample.astype(np.float64) * scale_norm
    noise_norm = noise.astype(np.float64) * scale_norm

    # 压缩后按 ffmpeg amix 默认（normalize=1）将压缩语音与风噪各按 1/2 混合。
    compr = _sidechain_compress(
        speech_norm, noise_norm, fs, threshold, ratio, attack, release, sc_gain
    )
    mix = (compr + noise_norm) / 2.0
    # noise stem 必须等于实际混入的风噪分量。
    noise_out = noise_norm / 2.0

    scale_norm = max(scale_norm, 1e-10)
    mix = mix / scale_norm
    noise_out = noise_out / scale_norm
    if clipping:
        # 与 ``urgent2026_challenge_track1-main/simulation/simulate_data_from_param.py`` 中
        # ``wind_noise`` 在 ``clipping=True`` 时等价：该处为
        # ``maximum(c*min, mix)`` + ``minimum(c*max, mix)``；``np.clip(mix, low, high)`` 与之
        # 逐元相同，仅写法不同。
        mix = np.clip(mix, clipping_threshold * np.min(mix), clipping_threshold * np.max(mix))
    return mix, noise_out


# =============================================================================
# RIR utilities
# =============================================================================
def get_rir_start_sample(h: np.ndarray, level_ratio: float = 1e-1) -> int:
    """Find start sample in RIR where energy exceeds threshold.
    
    Args:
        h: RIR (T,) or (C, T)
        level_ratio: Threshold as ratio of max value
        
    Returns:
        Start sample index
    """
    if not level_ratio < 1:
        raise ValueError(f"level_ratio must be < 1, got {level_ratio}")
    
    if h.ndim > 1:
        if not h.shape[0] < 20:
            raise ValueError(f"Too many channels: {h.shape}")
        h = np.reshape(h, (-1, h.shape[-1]))
        return np.min([get_rir_start_sample(h_, level_ratio) for h_ in h])
    
    abs_h = np.abs(h)
    max_index = np.argmax(abs_h)
    max_abs_value = abs_h[max_index]
    
    # Find first sample exceeding threshold before peak
    larger_than_threshold = abs_h[: max_index + 1] > level_ratio * max_abs_value
    rir_start_sample = np.argmax(larger_than_threshold)
    
    return int(rir_start_sample)


def estimate_early_rir(
    rir_sample: np.ndarray,
    early_rir_sec: float = 0.05,
    *,
    fs: int,
) -> np.ndarray:
    """Extract early reflections from RIR (first N ms after direct path).
    
    Args:
        rir_sample: RIR (C, T)
        early_rir_sec: Duration of early reflections in seconds
        fs: Sampling rate in Hz (keyword-only, must be explicitly provided)
        
    Returns:
        Early RIR (C, T) with tail zeroed out
    """
    early_rir_sec, fs = _validate_rir_time_parameters(
        early_rir_sec=early_rir_sec, fs=fs
    )
    rir_start_samples = np.array([get_rir_start_sample(h) for h in rir_sample])
    early_rir_samples = int(early_rir_sec * fs)
    rir_stop_samples = rir_start_samples + early_rir_samples
    
    rir_early = rir_sample.copy()
    for i in range(rir_sample.shape[0]):
        rir_early[i, rir_stop_samples[i] :] = 0
    
    return rir_early
