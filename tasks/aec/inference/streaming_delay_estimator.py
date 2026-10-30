"""Causal streaming delay estimator for AEC inference frontend.

Provides a lightweight, real-time capable delay estimator suitable as a frontend
for neural AEC models such as DBGNet that do not have an internal alignment
module.  The estimator maintains ring buffers of recent reference (farend) and
microphone samples, periodically computes a phase-transform (GCC-PHAT style)
cross-correlation over the buffered window, and median-smooths accepted delay
estimates.

Usage:
    cfg = StreamingDelayEstimatorConfig(
        sample_rate=16000,
        coarse_delay_ms=32.0,        # known hardware group delay
        max_residual_delay_ms=50.0,  # fine search around coarse
        buffer_duration_ms=300.0,
        update_interval_ms=100.0,
    )
    estimator = StreamingDelayEstimator(cfg)

    for mic_frame, ref_frame in stream:
        estimator.push_ref(ref_frame)
        estimator.push_mic(mic_frame)
        aligned_ref = estimator.get_aligned_ref_frame(len(mic_frame))
        # aligned_ref is delayed so that it matches echo in mic_frame
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import math
import os
import stat
import sys
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np


class _RingBuffer:
    """Simple 1-D float32 ring buffer for streaming audio samples."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"RingBuffer capacity must be positive, got {capacity}")
        self._capacity = int(capacity)
        self._buf = np.zeros(self._capacity, dtype=np.float32)
        self._size = 0
        self._head = 0  # index of next write position

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size

    def push(self, frame: np.ndarray) -> None:
        """Append a 1-D frame to the buffer, discarding oldest samples if full."""
        x = np.asarray(frame, dtype=np.float32).ravel()
        n = x.size
        if n == 0:
            return
        if n >= self._capacity:
            # Frame larger than capacity: keep only the newest samples.
            x = x[-self._capacity :]
            n = x.size
            self._buf[:] = x
            self._head = 0
            self._size = self._capacity
            return

        # Handle wrap-around for the write.
        first = min(n, self._capacity - self._head)
        self._buf[self._head : self._head + first] = x[:first]
        if first < n:
            self._buf[: n - first] = x[first:]
        self._head = (self._head + n) % self._capacity
        self._size = min(self._size + n, self._capacity)

    def read_oldest(self, n: int) -> np.ndarray:
        """Read the oldest ``n`` samples in chronological order.

        If fewer than ``n`` samples are available, the returned array is padded
        with leading zeros.  The output length always matches the request, even
        when ``n`` exceeds the physical ring-buffer capacity.
        """
        requested = max(0, int(n))
        if requested == 0:
            return np.zeros(0, dtype=np.float32)
        avail = min(requested, self._size)
        out = np.zeros(requested, dtype=np.float32)
        if avail == 0:
            return out

        # Start index of the available block (oldest sample).
        if self._size < self._capacity:
            start = 0
        else:
            start = self._head

        # Read ``avail`` samples starting from ``start``.
        idx = np.arange(avail)
        src = (start + idx) % self._capacity
        out[-avail:] = self._buf[src]
        return out

    def read_newest(self, n: int) -> np.ndarray:
        """Read the newest ``n`` samples in chronological order.

        If fewer than ``n`` samples are available, the returned array is padded
        with leading zeros.  The output length always matches the request, even
        when ``n`` exceeds the physical ring-buffer capacity.
        """
        requested = max(0, int(n))
        if requested == 0:
            return np.zeros(0, dtype=np.float32)
        avail = min(requested, self._size)
        out = np.zeros(requested, dtype=np.float32)
        if avail == 0:
            return out

        tail = self._head - 1 if self._head > 0 else self._capacity - 1
        idx = np.arange(avail - 1, -1, -1)
        src = (tail - idx) % self._capacity
        out[-avail:] = self._buf[src]
        return out


@dataclass
class StreamingDelayEstimatorConfig:
    """Configuration for :class:`StreamingDelayEstimator`.

    ``smooth_alpha`` 控制 median 窗口长度 ``max(3, floor(1 / alpha + 0.5))``；
    它不是 EMA。
    """

    sample_rate: int = 16000
    buffer_duration_ms: float = 500.0
    # 默认搜索窗为 [coarse-residual, coarse+residual] = [0, 100] ms：
    # 用户基于硬件实测延迟（8-36ms）决策的窄搜索窗 profile。
    coarse_delay_ms: float = 50.0
    max_residual_delay_ms: float = 50.0
    update_interval_ms: float = 100.0
    smooth_alpha: float = 0.15
    ref_energy_threshold_db: float = -50.0
    min_observation_ms: float = 200.0
    peak_threshold: float = 0.05
    peak_prominence_db: float = 6.0
    false_accept_prob: float = 1e-4
    acps_alpha: float = 0.8
    algorithm: str = "gcc_phat"
    eps: float = 1e-12

    # Matched-filter (AEC3-style) tunables.  These are ignored when
    # ``algorithm`` is ``"gcc_phat"``.
    mf_down_sampling_factor: int = 4
    mf_num_filters: int = 5
    mf_filter_length_subblocks: int = 32
    mf_lag_shift_subblocks: int = 24
    mf_smoothing: float = 0.7
    mf_excitation_limit: float = 0.005
    mf_histogram_window: int = 250
    mf_initial_threshold: int = 5
    mf_converged_threshold: int = 20
    mf_block_size: int = 64

    def __post_init__(self) -> None:
        integer_names = (
            "sample_rate",
            "mf_down_sampling_factor",
            "mf_num_filters",
            "mf_filter_length_subblocks",
            "mf_lag_shift_subblocks",
            "mf_histogram_window",
            "mf_initial_threshold",
            "mf_converged_threshold",
            "mf_block_size",
        )
        for name in integer_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            setattr(self, name, int(value))

        float_names = (
            "buffer_duration_ms",
            "coarse_delay_ms",
            "max_residual_delay_ms",
            "update_interval_ms",
            "smooth_alpha",
            "ref_energy_threshold_db",
            "min_observation_ms",
            "peak_threshold",
            "peak_prominence_db",
            "false_accept_prob",
            "acps_alpha",
            "eps",
            "mf_smoothing",
            "mf_excitation_limit",
        )
        for name in float_names:
            raw_value = getattr(self, name)
            if isinstance(raw_value, bool) or not isinstance(
                raw_value, (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"{name} must be a finite number, got {raw_value!r}")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
            setattr(self, name, value)

        if self.buffer_duration_ms <= 0:
            raise ValueError("buffer_duration_ms must be > 0")
        if self.update_interval_ms <= 0:
            raise ValueError("update_interval_ms must be > 0")
        if self.min_observation_ms <= 0:
            raise ValueError("min_observation_ms must be > 0")
        if self.min_observation_ms > self.buffer_duration_ms:
            raise ValueError("min_observation_ms must be <= buffer_duration_ms")
        if self.coarse_delay_ms < 0:
            raise ValueError("coarse_delay_ms must be >= 0")
        if self.max_residual_delay_ms < 0:
            raise ValueError("max_residual_delay_ms must be >= 0")
        if self.peak_prominence_db < 0:
            raise ValueError("peak_prominence_db must be >= 0")
        if not 0 < self.smooth_alpha <= 1:
            raise ValueError("smooth_alpha must be in (0, 1]")
        if not 0 <= self.acps_alpha < 1:
            raise ValueError("acps_alpha must be in [0, 1)")
        if not 0 < self.peak_threshold <= 1:
            raise ValueError("peak_threshold must be in (0, 1]")
        if not 0 < self.false_accept_prob < 1:
            raise ValueError("false_accept_prob must be in (0, 1)")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")
        if not 0 <= self.mf_smoothing <= 1:
            raise ValueError("mf_smoothing must be in [0, 1]")
        if self.mf_excitation_limit < 0:
            raise ValueError("mf_excitation_limit must be >= 0")
        if self.mf_block_size % self.mf_down_sampling_factor != 0:
            raise ValueError("mf_block_size must be divisible by mf_down_sampling_factor")
        if self.mf_initial_threshold > self.mf_converged_threshold:
            raise ValueError("mf_initial_threshold must be <= mf_converged_threshold")

        if not isinstance(self.algorithm, str):
            raise ValueError(f"algorithm must be a string, got {self.algorithm!r}")
        algorithm = self.algorithm.strip().lower()
        if algorithm not in {"gcc_phat", "matched_filter"}:
            raise ValueError(
                f"algorithm must be 'gcc_phat' or 'matched_filter', got {algorithm!r}"
            )
        self.algorithm = algorithm


@dataclass
class AECStreamAlignConfig:
    """Strict YAML-facing schema shared by training and inference."""

    enabled: bool = False
    algorithm: str = "gcc_phat"
    buffer_ms: float = 500.0
    # 与 StreamingDelayEstimatorConfig 共享的用户决策窄搜索窗默认：
    # 搜索范围 [coarse-residual, coarse+residual] = [0, 100] ms（硬件实测 8-36ms）。
    coarse_ms: float = 50.0
    residual_ms: float = 50.0
    update_ms: float = 100.0
    alpha: float = 0.15
    acps_alpha: float = 0.8
    ref_thresh_db: float = -50.0
    min_obs_ms: float = 200.0
    peak_thresh: float = 0.05
    prominence_db: float = 6.0
    false_accept_prob: float = 1e-4
    frame_size: int = 160

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(
                f"aec_frontend.delay_alignment.enabled must be a bool, got {self.enabled!r}"
            )
        if (
            isinstance(self.frame_size, bool)
            or not isinstance(self.frame_size, (int, np.integer))
            or self.frame_size <= 0
        ):
            raise ValueError(
                "aec_frontend.delay_alignment.frame_size must be a positive integer, "
                f"got {self.frame_size!r}"
            )
        self.frame_size = int(self.frame_size)
        # 统一配置与运行时校验，避免语义漂移。
        validated = self.to_estimator_config(sample_rate=16000)
        self.algorithm = validated.algorithm
        self.buffer_ms = validated.buffer_duration_ms
        self.coarse_ms = validated.coarse_delay_ms
        self.residual_ms = validated.max_residual_delay_ms
        self.update_ms = validated.update_interval_ms
        self.alpha = validated.smooth_alpha
        self.acps_alpha = validated.acps_alpha
        self.ref_thresh_db = validated.ref_energy_threshold_db
        self.min_obs_ms = validated.min_observation_ms
        self.peak_thresh = validated.peak_threshold
        self.prominence_db = validated.peak_prominence_db
        self.false_accept_prob = validated.false_accept_prob

    @classmethod
    def from_dict(cls, raw: dict | None) -> "AECStreamAlignConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(
                f"aec_frontend.delay_alignment must be a mapping, got {type(raw).__name__}"
            )
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(
                f"aec_frontend.delay_alignment has unknown key(s) {unknown}; "
                f"expected subset of {sorted(known)}"
            )
        values = {field.name: raw.get(field.name, field.default) for field in fields(cls)}
        return cls(**values)

    def to_estimator_config(self, sample_rate: int) -> StreamingDelayEstimatorConfig:
        return StreamingDelayEstimatorConfig(
            sample_rate=sample_rate,
            buffer_duration_ms=self.buffer_ms,
            coarse_delay_ms=self.coarse_ms,
            max_residual_delay_ms=self.residual_ms,
            update_interval_ms=self.update_ms,
            smooth_alpha=self.alpha,
            ref_energy_threshold_db=self.ref_thresh_db,
            min_observation_ms=self.min_obs_ms,
            peak_threshold=self.peak_thresh,
            peak_prominence_db=self.prominence_db,
            false_accept_prob=self.false_accept_prob,
            acps_alpha=self.acps_alpha,
            algorithm=self.algorithm,
        )

    def to_dict(self) -> dict:
        return asdict(self)


class StreamingDelayEstimator:
    """Causal streaming delay estimator for AEC frontends.

    The estimator tracks the delay between the reference (farend) signal and its
    echo component in the microphone signal.  Positive delay means the echo in
    the microphone lags behind the reference.  The estimate is updated
    periodically by computing a phase-transform cross-correlation (GCC-PHAT)
    over a sliding window; accepted raw estimates are median-smoothed.

    Args:
        config: Estimator configuration.  See :class:`StreamingDelayEstimatorConfig`.
    """

    def __init__(self, config: Optional[StreamingDelayEstimatorConfig] = None):
        self.cfg = config or StreamingDelayEstimatorConfig()
        self._sr = self.cfg.sample_rate

        self._buffer_samples = int(round(self.cfg.buffer_duration_ms * self._sr / 1000.0))
        self._coarse_samples = int(round(self.cfg.coarse_delay_ms * self._sr / 1000.0))
        self._max_residual_samples = int(
            round(self.cfg.max_residual_delay_ms * self._sr / 1000.0)
        )
        self._update_interval_samples = int(
            round(self.cfg.update_interval_ms * self._sr / 1000.0)
        )
        self._min_observation_samples = max(
            256, int(round(self.cfg.min_observation_ms * self._sr / 1000.0))
        )
        if self._buffer_samples <= 0 or self._update_interval_samples <= 0:
            raise ValueError("duration values must resolve to at least one sample")
        if self._min_observation_samples > self._buffer_samples:
            raise ValueError("min_observation_ms resolves beyond the analysis buffer")
        self._ref_energy_threshold = 10.0 ** (self.cfg.ref_energy_threshold_db / 20.0)
        self._peak_threshold = self.cfg.peak_threshold
        self._peak_prominence_linear = 10.0 ** (-self.cfg.peak_prominence_db / 20.0)
        self._alpha = self.cfg.smooth_alpha
        # Python round uses ties-to-even; use the same half-up formula as C++.
        self._history_len = max(3, int(math.floor(1.0 / self._alpha + 0.5)))
        self._history: list[float] = []

        # Ref buffer needs extra capacity for aligned output: it must hold at
        # least coarse + max_residual + one frame of history.
        required_ref_capacity = (
            self._buffer_samples
            + self._coarse_samples
            + self._max_residual_samples
            + 1024
        )
        self._ref_buf = _RingBuffer(max(required_ref_capacity, self._buffer_samples))
        self._mic_buf = _RingBuffer(self._buffer_samples)

        self._current_delay_samples = float(self._coarse_samples)
        self._samples_since_update = 0
        self._frame_count = 0

        # ACPS: accumulated cross-power spectrum for robust GCC-PHAT.
        self._acps_alpha = float(self.cfg.acps_alpha)
        self._R_accum: Optional[np.ndarray] = None
        self._acps_nfft: int = 0

    @property
    def backend_identity(self) -> str:
        """Stable identity for the native Python backend."""
        return "python"

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def delay_samples(self) -> int:
        """Current smoothed delay estimate in samples (echo lags ref)."""
        return int(round(self._current_delay_samples))

    @property
    def delay_ms(self) -> float:
        """Current smoothed delay estimate in milliseconds."""
        return self._current_delay_samples * 1000.0 / self._sr

    @staticmethod
    def _finite_frame(frame: np.ndarray, name: str) -> np.ndarray:
        """Normalize one 1-D frame and reject invalid input before state changes."""
        array = np.asarray(frame)
        if array.ndim != 1:
            raise ValueError(
                f"{name} frame must be one-dimensional, got shape {array.shape}"
            )
        normalized = np.asarray(array, dtype=np.float32)
        if not np.all(np.isfinite(normalized)):
            raise ValueError(f"{name} frame must contain only finite samples")
        return normalized

    def push_ref(self, frame: np.ndarray) -> None:
        """Push a finite reference (farend) frame atomically."""
        normalized = self._finite_frame(frame, "reference")
        self._ref_buf.push(normalized)

    def push_mic(self, frame: np.ndarray) -> None:
        """Push a finite microphone frame and trigger a delay update.

        The update is performed after both the reference and microphone frames
        for the current iteration are buffered, so that the two windows are
        time-aligned.  Cadence is based on accepted microphone samples: each
        call triggers at most one update, while modulo arithmetic preserves any
        overshoot for subsequent calls.
        """
        normalized = self._finite_frame(frame, "microphone")
        self._mic_buf.push(normalized)
        self._frame_count += 1
        self._samples_since_update += normalized.size
        if self._samples_since_update >= self._update_interval_samples:
            self._samples_since_update %= self._update_interval_samples
            self._update_delay()

    def get_aligned_ref_frame(self, frame_size: int) -> np.ndarray:
        """Return a reference frame aligned to the most recent microphone frame.

        The returned frame is read from the reference ring buffer delayed by the
        current delay estimate.  If insufficient history is available, leading
        samples are zero-padded.

        Args:
            frame_size: Number of samples to return.

        Returns:
            ``(frame_size,)`` float32 array.
        """
        if (
            isinstance(frame_size, bool)
            or not isinstance(frame_size, (int, np.integer))
            or frame_size < 0
        ):
            raise ValueError(
                f"frame_size must be a non-negative integer, got {frame_size!r}"
            )
        frame_size = int(frame_size)
        if frame_size == 0:
            return np.zeros(0, dtype=np.float32)

        delay = self.delay_samples
        if frame_size + delay > self._ref_buf.capacity:
            raise ValueError(
                f"frame_size + delay exceeds reference buffer capacity: "
                f"{frame_size} + {delay} > {self._ref_buf.capacity}"
            )
        # We want ref aligned with mic, so read ref delayed by ``delay``.
        # Read newest ``frame_size + delay`` samples, then take the oldest
        # ``frame_size`` of that block.
        total = frame_size + delay
        if total <= 0:
            return self._ref_buf.read_newest(frame_size)

        block = self._ref_buf.read_newest(total)
        return block[:frame_size].copy()

    def reset(self) -> None:
        """Reset internal state (buffers, delay estimate and history)."""
        self._ref_buf = _RingBuffer(self._ref_buf.capacity)
        self._mic_buf = _RingBuffer(self._mic_buf.capacity)
        self._current_delay_samples = float(self._coarse_samples)
        self._samples_since_update = 0
        self._frame_count = 0
        self._R_accum = None
        self._acps_nfft = 0
        self._history.clear()

    def _selected_lag_confidence(
        self,
        mic: np.ndarray,
        ref: np.ndarray,
        lag: int,
        candidate_count: int,
    ) -> Optional[bool]:
        """用一阶差分、去均值的绝对 NCC 校验 selected lag。

        ``None`` 表示数值或边界观测无效；``False`` 表示有限观测未达到
        family-wise 误接受概率门。正 lag 使用 ``mic[lag:]`` 对齐
        ``ref[:-lag]``；负 lag 则使用 ``mic[:lag]`` 对齐 ``ref[-lag:]``。
        两种情况的一阶差分样本数均为 ``N = n - abs(lag) - 1``。
        """
        if (
            isinstance(lag, bool)
            or not isinstance(lag, (int, np.integer))
            or candidate_count <= 0
            or mic.size != ref.size
        ):
            return None
        lag = int(lag)
        overlap = mic.size - abs(lag)
        if overlap < 3:
            return None

        mic_start = max(lag, 0)
        ref_start = max(-lag, 0)
        mic_overlap = mic[mic_start : mic_start + overlap].astype(
            np.float64, copy=False
        )
        ref_overlap = ref[ref_start : ref_start + overlap].astype(
            np.float64, copy=False
        )
        mic_diff = np.diff(mic_overlap)
        ref_diff = np.diff(ref_overlap)
        n_overlap = mic_diff.size
        if n_overlap < 2:
            return None

        mic_raw_energy = float(np.dot(mic_diff, mic_diff))
        ref_raw_energy = float(np.dot(ref_diff, ref_diff))
        mic_diff = mic_diff - np.mean(mic_diff)
        ref_diff = ref_diff - np.mean(ref_diff)
        numerator = float(np.dot(mic_diff, ref_diff))
        mic_energy = float(np.dot(mic_diff, mic_diff))
        ref_energy = float(np.dot(ref_diff, ref_diff))
        if not all(
            math.isfinite(value)
            for value in (
                numerator,
                mic_raw_energy,
                ref_raw_energy,
                mic_energy,
                ref_energy,
            )
        ):
            return None

        relative_eps = np.finfo(np.float64).eps
        if (
            mic_energy <= relative_eps * mic_raw_energy
            or ref_energy <= relative_eps * ref_raw_energy
        ):
            return None
        denominator = math.sqrt(mic_energy * ref_energy)
        if not math.isfinite(denominator) or denominator <= 0.0:
            return None

        abs_ncc = abs(numerator / denominator)
        threshold = math.sqrt(
            2.0
            * math.log(2.0 * float(candidate_count) / self.cfg.false_accept_prob)
            / float(n_overlap)
        )
        if (
            not math.isfinite(abs_ncc)
            or not math.isfinite(threshold)
            or threshold >= 1.0
        ):
            return None
        return abs_ncc >= threshold

    def _update_delay(self) -> None:
        """Compute GCC-PHAT over buffered window and update smoothed delay."""
        if self._mic_buf.size < self._min_observation_samples:
            return
        if self._ref_buf.size < self._min_observation_samples:
            return

        n = min(self._mic_buf.size, self._ref_buf.size)
        n = min(n, self._buffer_samples)
        mic = self._mic_buf.read_newest(n)
        ref = self._ref_buf.read_newest(n)

        if not np.all(np.isfinite(mic)) or not np.all(np.isfinite(ref)):
            return
        ref64 = ref.astype(np.float64, copy=False)
        ref_rms = float(np.sqrt(np.mean(ref64 * ref64)))
        if not math.isfinite(ref_rms) or ref_rms < self._ref_energy_threshold:
            return

        search_center = int(round(self._current_delay_samples))
        min_lag = max(0, search_center - self._max_residual_samples)
        max_lag = min(search_center + self._max_residual_samples, n - 1)
        candidate_count = max_lag - min_lag + 1

        previous_accum = None if self._R_accum is None else self._R_accum.copy()
        previous_nfft = self._acps_nfft
        if self._acps_alpha > 0:
            result = self._gcc_phat_tau_acps(mic, ref, search_center)
        else:
            result = self._gcc_phat_tau(mic, ref, search_center)
        if result is None:
            self._R_accum = previous_accum
            self._acps_nfft = previous_nfft
            return
        tau, peak_mag, mean_mag = result

        selected_lag = int(math.floor(float(tau) + 0.5))
        confidence = self._selected_lag_confidence(
            mic, ref, selected_lag, candidate_count
        )
        if confidence is None:
            # 无效边界或非有限观测不得改变 ACPS 状态。
            self._R_accum = previous_accum
            self._acps_nfft = previous_nfft
            return
        if not confidence:
            return

        # Quality gate: peak magnitude and prominence.
        if peak_mag < self._peak_threshold:
            return
        if peak_mag < mean_mag / self._peak_prominence_linear:
            return

        tau = max(0.0, float(tau))
        if self._history:
            deviation = abs(tau - self._current_delay_samples)
            if deviation > self._max_residual_samples and peak_mag < 0.5:
                return

        self._history.append(tau)
        if len(self._history) > self._history_len:
            self._history.pop(0)

        self._current_delay_samples = float(np.median(self._history))

    # ------------------------------------------------------------------
    # GCC-PHAT with accumulated cross-power spectrum (ACPS)
    # ------------------------------------------------------------------
    def _gcc_phat_tau_acps(
        self,
        sig: np.ndarray,
        refsig: np.ndarray,
        search_center: Optional[int] = None,
    ) -> Optional[tuple[float, float, float]]:
        """GCC-PHAT with cross-power spectrum accumulation.

        Accumulates the cross-power spectrum across updates before PHAT
        normalization, effectively extending the analysis window without
        increasing the buffer size.  This significantly improves peak
        reliability when the delay approaches the buffer length.
        """
        sig = np.asarray(sig, dtype=np.float32).ravel()
        refsig = np.asarray(refsig, dtype=np.float32).ravel()
        if not np.all(np.isfinite(sig)) or not np.all(np.isfinite(refsig)):
            return None
        n = sig.size + refsig.size - 1
        if n <= 0:
            return None

        nfft = 1 << (n - 1).bit_length()
        SIG = np.fft.rfft(sig, n=nfft)
        REF = np.fft.rfft(refsig, n=nfft)
        R = SIG * np.conj(REF)
        if not np.all(np.isfinite(R)):
            return None

        old_valid = False
        if self._R_accum is not None:
            try:
                old_valid = (
                    self._acps_nfft == nfft
                    and self._R_accum.shape == R.shape
                    and bool(np.all(np.isfinite(self._R_accum)))
                )
            except (AttributeError, TypeError):
                old_valid = False
            if not old_valid:
                # 丢弃形状不匹配或已污染的遗留状态；当前观测从零重新起步。
                self._R_accum = None
                self._acps_nfft = 0

        if old_valid:
            a = self._acps_alpha
            candidate = a * self._R_accum + (1.0 - a) * R
        else:
            candidate = R.copy()
        if not np.all(np.isfinite(candidate)):
            return None
        denom = np.abs(candidate) + self.cfg.eps
        normalized = candidate / denom
        if not np.all(np.isfinite(denom)) or not np.all(np.isfinite(normalized)):
            return None
        cc_fft = np.fft.irfft(normalized, n=nfft)
        if not np.all(np.isfinite(cc_fft)):
            return None

        sig_size = sig.size
        neg_lags = cc_fft[-(sig_size - 1) :] if sig_size > 1 else np.array([])
        pos_lags = cc_fft[: refsig.size]
        cc = np.concatenate([neg_lags, pos_lags])
        if not np.all(np.isfinite(cc)):
            return None
        result = self._find_peak_in_cc(cc, sig_size, refsig.size, search_center)
        # 仅在本次候选状态及其完整归一化路径通过后提交持久 ACPS。
        self._R_accum = candidate
        self._acps_nfft = nfft
        return result

    def _gcc_phat_tau(
        self,
        sig: np.ndarray,
        refsig: np.ndarray,
        search_center: Optional[int] = None,
    ) -> Optional[tuple[float, float, float]]:
        """Return (delay, peak_magnitude, mean_magnitude) via single-window GCC-PHAT."""
        sig = np.asarray(sig, dtype=np.float32).ravel()
        refsig = np.asarray(refsig, dtype=np.float32).ravel()
        if not np.all(np.isfinite(sig)) or not np.all(np.isfinite(refsig)):
            return None
        n = sig.size + refsig.size - 1
        if n <= 0:
            return None

        nfft = 1 << (n - 1).bit_length()
        SIG = np.fft.rfft(sig, n=nfft)
        REF = np.fft.rfft(refsig, n=nfft)
        R = SIG * np.conj(REF)
        if not np.all(np.isfinite(R)):
            return None
        denom = np.abs(R) + self.cfg.eps
        normalized = R / denom
        if not np.all(np.isfinite(denom)) or not np.all(np.isfinite(normalized)):
            return None
        cc_fft = np.fft.irfft(normalized, n=nfft)
        if not np.all(np.isfinite(cc_fft)):
            return None

        neg_lags = cc_fft[-(sig.size - 1) :] if sig.size > 1 else np.array([])
        pos_lags = cc_fft[: refsig.size]
        cc = np.concatenate([neg_lags, pos_lags])
        if not np.all(np.isfinite(cc)):
            return None

        return self._find_peak_in_cc(cc, sig.size, refsig.size, search_center)

    def _find_peak_in_cc(
        self,
        cc: np.ndarray,
        sig_size: int,
        ref_size: int,
        search_center: Optional[int] = None,
    ) -> Optional[tuple[float, float, float]]:
        """Find the best delay peak in a full-layout correlation array.

        The correlation *cc* has layout matching ``np.correlate(..., 'full')``:
        lags from ``-(sig_size - 1)`` to ``+(ref_size - 1)``.

        Returns ``(peak_lag, selected_peak_magnitude, mean_magnitude)`` or ``None``.
        The returned magnitude belongs to the selected (final) peak, not the
        global maximum of the search window.
        """
        try:
            cc = np.asarray(cc).ravel()
            center = self._coarse_samples if search_center is None else int(search_center)
        except (TypeError, ValueError, OverflowError):
            return None
        if sig_size <= 0 or ref_size <= 0 or cc.size != sig_size + ref_size - 1:
            return None
        if not np.all(np.isfinite(cc)):
            return None

        min_lag = max(0, center - self._max_residual_samples)
        max_lag = center + self._max_residual_samples
        min_lag = max(min_lag, -(sig_size - 1))
        max_lag = min(max_lag, ref_size - 1)
        if min_lag > max_lag:
            return None

        offset = sig_size - 1
        start_idx = offset + min_lag
        end_idx = offset + max_lag + 1
        search_cc = cc[start_idx:end_idx]

        abs_cc = np.abs(search_cc)
        peak_mag = float(np.max(abs_cc))
        mean_mag = float(np.mean(abs_cc))
        if peak_mag <= self.cfg.eps:
            return None

        center_rel = center - min_lag
        strong_threshold = max(
            self._peak_threshold,
            peak_mag * 0.7,
        )

        if abs_cc.size == 1:
            peak_rels = np.array([0], dtype=np.int64)
        else:
            interior = (abs_cc[1:-1] >= abs_cc[:-2]) & (abs_cc[1:-1] >= abs_cc[2:])
            peak_rels = np.where(interior)[0] + 1
            if abs_cc[0] >= abs_cc[1]:
                peak_rels = np.concatenate(([0], peak_rels))
            if abs_cc[-1] >= abs_cc[-2]:
                peak_rels = np.concatenate((peak_rels, [abs_cc.size - 1]))

        strong_rels = peak_rels[abs_cc[peak_rels] >= strong_threshold]
        if strong_rels.size == 0:
            strong_rels = np.array([int(np.argmax(abs_cc))])

        prominence_half = max(1, min(8, len(abs_cc) // 16))
        prominences = []
        for pr in strong_rels:
            lo = max(0, pr - prominence_half)
            hi = min(len(abs_cc), pr + prominence_half + 1)
            neighbourhood = np.concatenate([abs_cc[lo:pr], abs_cc[pr + 1 : hi]])
            neighbourhood_mean = float(np.mean(neighbourhood)) if neighbourhood.size else 0.0
            prominences.append(float(abs_cc[pr]) - neighbourhood_mean)
        prominences = np.array(prominences)

        max_prominence = float(np.max(prominences))
        top_rels = strong_rels[prominences >= max_prominence * 0.9]
        best_rel = int(top_rels[np.argmin(np.abs(top_rels - center_rel))])

        # Sub-sample peak refinement via parabolic interpolation around the
        # three nearest samples.  This reduces integer-sample quantization
        # jitter without increasing FFT size.
        best_rel_float = float(best_rel)
        if 0 < best_rel < len(abs_cc) - 1:
            y_m1 = float(abs_cc[best_rel - 1])
            y_0 = float(abs_cc[best_rel])
            y_p1 = float(abs_cc[best_rel + 1])
            denom = y_m1 - 2.0 * y_0 + y_p1
            if abs(denom) > self.cfg.eps:
                delta = 0.5 * (y_m1 - y_p1) / denom
                # Clamp to the nearest sample interval for robustness.
                delta = max(-0.5, min(0.5, delta))
                best_rel_float += delta

        peak_lag = min_lag + best_rel_float
        # Quality gates must use the magnitude of the selected peak, not the
        # global maximum, so that confidence is tied to the delay we actually emit.
        selected_peak = float(abs_cc[best_rel])
        return float(peak_lag), selected_peak, mean_mag


_MATCHED_FILTER_INPUT_ABS_LIMIT = 16.0


class _ValidatedDelayEstimator:
    """Validate the shared public contract before delegating to a backend."""

    def __init__(
        self,
        estimator,
        aligned_ref_capacity: int,
        backend_identity: str,
        input_abs_limit: Optional[float],
    ):
        if backend_identity not in {"cpp", "python"}:
            raise ValueError(f"unknown delay estimator backend identity: {backend_identity!r}")
        self._estimator = estimator
        self._aligned_ref_capacity = aligned_ref_capacity
        self._backend_identity = backend_identity
        self._input_abs_limit = input_abs_limit

    def __getattr__(self, name):
        return getattr(self._estimator, name)

    @property
    def backend_identity(self) -> str:
        """Stable backend identity independent of implementation class names."""
        return self._backend_identity

    @property
    def input_abs_limit(self) -> Optional[float]:
        """稳定的输入绝对幅度上限；``None`` 表示除 finite 外无限制。"""
        return self._input_abs_limit

    def preflight_input(self, frame: np.ndarray, name: str) -> np.ndarray:
        """无状态预检并规范化单侧输入，供调用方在提交 pair 前使用。"""
        normalized = StreamingDelayEstimator._finite_frame(frame, name)
        if (
            self._input_abs_limit is not None
            and normalized.size
            and float(np.max(np.abs(normalized))) > self._input_abs_limit
        ):
            raise ValueError(
                f"{name} frame exceeds normalized audio range "
                f"[-{self._input_abs_limit:g}, {self._input_abs_limit:g}]"
            )
        return normalized

    def push_ref(self, frame: np.ndarray) -> None:
        normalized = self.preflight_input(frame, "reference")
        self._estimator.push_ref(normalized)

    def push_mic(self, frame: np.ndarray) -> None:
        normalized = self.preflight_input(frame, "microphone")
        self._estimator.push_mic(normalized)

    def get_aligned_ref_frame(self, frame_size: int) -> np.ndarray:
        if (
            isinstance(frame_size, bool)
            or not isinstance(frame_size, (int, np.integer))
            or frame_size < 0
        ):
            raise ValueError(
                f"frame_size must be a non-negative integer, got {frame_size!r}"
            )
        frame_size = int(frame_size)
        delay = self.delay_samples
        if frame_size + delay > self._aligned_ref_capacity:
            raise ValueError(
                "frame_size + delay exceeds reference buffer capacity: "
                f"{frame_size} + {delay} > {self._aligned_ref_capacity}"
            )
        return self._estimator.get_aligned_ref_frame(frame_size)


def _extension_file_identity(path: Path) -> tuple[int, int, int]:
    """返回扩展文件的 (inode, size, mtime_ns) 身份，用于检测加载前后的替换。"""
    stat_result = path.stat()
    return (
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
    )


def _validate_delay_estimator_extension(
    extension_path: Path, source_path: Path, native_dir: Path
) -> Path:
    """校验扩展位置与源码新旧关系，并返回规范化路径。

    ``_native/`` 是仓库内、与源码同属一个 Unix 用户的受信构建产物目录；
    校验通过 ``os.open`` + ``os.fstat`` 读取真实文件身份（含常规文件检查），
    加载后再复核一次身份，使检查-加载窗口内的替换 fail-closed。
    """
    extension_path = extension_path.resolve()
    native_dir = native_dir.resolve()
    if extension_path.parent != native_dir:
        raise ImportError(f"refusing non-native-dir extension: {extension_path}")
    try:
        fd = os.open(extension_path, os.O_RDONLY)
    except OSError as exc:
        raise ImportError(
            f"native-dir extension is unreadable: {extension_path}"
        ) from exc
    try:
        extension_stat = os.fstat(fd)
    finally:
        os.close(fd)
    if not stat.S_ISREG(extension_stat.st_mode):
        raise ImportError(
            f"native-dir extension is not a regular file: {extension_path}"
        )
    try:
        source_stat = source_path.stat()
    except OSError as exc:
        raise ImportError(f"extension source is missing: {source_path}") from exc
    if extension_stat.st_mtime_ns < source_stat.st_mtime_ns:
        raise ImportError(
            f"native-dir extension is stale: {extension_path}; "
            f"source is newer: {source_path}"
        )
    return extension_path


def _load_delay_estimator_extension():
    """仅从 tasks/aec/inference/_native/ 加载通过源码时间戳校验的 C++ 扩展。

    ``_native/`` 属受信边界（仓库内、同用户可写的构建产物目录）：扩展不做
    内容寻址，加载前后各校验一次文件身份，检测检查-加载窗口内的替换。
    """
    module_name = "delay_estimator_ext"
    project_root = Path(__file__).resolve().parents[2]
    native_dir = project_root / "aec" / "inference" / "_native"
    source_path = project_root / "aec" / "inference" / "csrc" / f"{module_name}.cpp"
    candidates = [
        native_dir / f"{module_name}{suffix}"
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ]
    extension_path = next((path for path in candidates if path.is_file()), None)
    if extension_path is None:
        raise ImportError(f"native-dir {module_name} extension is not built")
    extension_path = _validate_delay_estimator_extension(
        extension_path, source_path, native_dir
    )
    pre_load_identity = _extension_file_identity(extension_path)

    loaded = sys.modules.get(module_name)
    if loaded is not None:
        loaded_path = getattr(loaded, "__file__", None)
        if loaded_path is None or Path(loaded_path).resolve() != extension_path:
            raise ImportError(
                f"refusing previously loaded non-native-dir extension: {loaded_path}"
            )
        return loaded

    spec = importlib.util.spec_from_file_location(module_name, extension_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create extension loader for {extension_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        loaded_path = Path(module.__file__).resolve()
        if loaded_path != extension_path:
            raise ImportError(
                f"loaded unexpected {module_name} path: {loaded_path}; "
                f"expected {extension_path}"
            )
        if _extension_file_identity(extension_path) != pre_load_identity:
            raise ImportError(
                f"{module_name} extension changed while loading: {extension_path}"
            )
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _aligned_ref_capacity(config: StreamingDelayEstimatorConfig) -> int:
    """Return the factory-level aligned-reference capacity for one backend."""
    if config.algorithm == "gcc_phat":
        raw_samples = (
            config.buffer_duration_ms * config.sample_rate / 1000.0,
            config.coarse_delay_ms * config.sample_rate / 1000.0,
            config.max_residual_delay_ms * config.sample_rate / 1000.0,
        )
        python_capacity = sum(int(round(value)) for value in raw_samples) + 1024
        cpp_capacity = sum(
            int(math.floor(value + 0.5)) for value in raw_samples
        ) + 1024
        return min(python_capacity, cpp_capacity)

    sub_block_size = config.mf_block_size // config.mf_down_sampling_factor
    max_lag = (
        (config.mf_num_filters - 1)
        * config.mf_lag_shift_subblocks
        * sub_block_size
        + config.mf_filter_length_subblocks * sub_block_size
        - 1
    )
    return (
        max_lag * config.mf_down_sampling_factor
        + config.mf_block_size * 32
    )


def create_delay_estimator(config: Optional[StreamingDelayEstimatorConfig] = None):
    """Factory: create a delay estimator based on ``config.algorithm``.

    Returns an object with the same interface: ``push_ref``, ``push_mic``,
    ``delay_samples``, ``delay_ms``, ``get_aligned_ref_frame``, ``reset`` and
    the stable read-only ``backend_identity`` (``"cpp"`` or ``"python"``).
    ``input_abs_limit`` and the stateless ``preflight_input`` expose the stable
    estimator-specific input contract: GCC has no extra amplitude limit, while
    matched-filter input is bounded to an absolute amplitude of 16.

    Supported algorithms:
      - ``"gcc_phat"`` (default): GCC-PHAT with optional ACPS accumulation.
        Uses the C++ backend (delay_estimator_ext) when available; benchmark the
        speedup on the target platform. Falls back to Python numpy otherwise.
      - ``"matched_filter"``: AEC3-style NLMS matched filter + histogram voting.
        Uses the C++ backend (delay_estimator_ext) when available; benchmark the
        speedup on the target platform. Falls back to Python numpy/scipy otherwise.
    """
    config = config or StreamingDelayEstimatorConfig()
    algo = config.algorithm.strip().lower()
    capacity = _aligned_ref_capacity(config)
    if algo == "gcc_phat":
        try:
            extension = _load_delay_estimator_extension()
            backend = extension.GccPhatEstimatorCpp(
                sr=config.sample_rate,
                buffer_ms=config.buffer_duration_ms,
                coarse_ms=config.coarse_delay_ms,
                residual_ms=config.max_residual_delay_ms,
                update_ms=config.update_interval_ms,
                smooth_alpha=config.smooth_alpha,
                ref_thresh_db=config.ref_energy_threshold_db,
                peak_thresh=config.peak_threshold,
                prominence_db=config.peak_prominence_db,
                false_accept_prob=config.false_accept_prob,
                acps_alpha=config.acps_alpha,
                min_obs_ms=config.min_observation_ms,
                eps=config.eps,
            )
            backend_identity = "cpp"
        except ImportError as exc:
            warnings.warn(
                "C++ extension delay_estimator_ext was rejected; "
                "falling back to Python GCC-PHAT implementation. "
                f"Reason: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            backend = StreamingDelayEstimator(config)
            backend_identity = "python"
        return _ValidatedDelayEstimator(
            backend, capacity, backend_identity, input_abs_limit=None
        )
    if algo == "matched_filter":
        cpp_supported = (
            config.sample_rate == 16000 and config.mf_down_sampling_factor == 4
        )
        if cpp_supported:
            try:
                extension = _load_delay_estimator_extension()
                backend = extension.MatchedFilterEstimatorCpp(
                    sr=config.sample_rate,
                    ds=config.mf_down_sampling_factor,
                    num_filters=config.mf_num_filters,
                    flen_subblocks=config.mf_filter_length_subblocks,
                    lag_shift_subblocks=config.mf_lag_shift_subblocks,
                    smoothing=config.mf_smoothing,
                    excitation_limit=config.mf_excitation_limit,
                    hist_window=config.mf_histogram_window,
                    init_thresh=config.mf_initial_threshold,
                    conv_thresh=config.mf_converged_threshold,
                    block_size=config.mf_block_size,
                )
                return _ValidatedDelayEstimator(
                    backend,
                    capacity,
                    "cpp",
                    input_abs_limit=_MATCHED_FILTER_INPUT_ABS_LIMIT,
                )
            except ImportError as exc:
                warnings.warn(
                    "C++ extension delay_estimator_ext was rejected; "
                    "falling back to Python matched-filter implementation. "
                    f"Reason: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "C++ matched-filter backend only supports sample_rate=16000 "
                "and mf_down_sampling_factor=4; falling back to Python "
                "matched-filter implementation.",
                RuntimeWarning,
                stacklevel=2,
            )
        from tasks.aec.inference.matched_filter_delay_estimator import (
            MatchedFilterDelayEstimator,
            MatchedFilterDelayEstimatorConfig,
        )
        mf_cfg = MatchedFilterDelayEstimatorConfig(
            sample_rate=config.sample_rate,
            down_sampling_factor=config.mf_down_sampling_factor,
            num_filters=config.mf_num_filters,
            filter_length_subblocks=config.mf_filter_length_subblocks,
            lag_shift_subblocks=config.mf_lag_shift_subblocks,
            smoothing=config.mf_smoothing,
            excitation_limit=config.mf_excitation_limit,
            histogram_window=config.mf_histogram_window,
            initial_threshold=config.mf_initial_threshold,
            converged_threshold=config.mf_converged_threshold,
            block_size=config.mf_block_size,
        )
        backend = MatchedFilterDelayEstimator(mf_cfg)
        return _ValidatedDelayEstimator(
            backend,
            capacity,
            "python",
            input_abs_limit=_MATCHED_FILTER_INPUT_ABS_LIMIT,
        )
    raise ValueError(f"Unknown delay estimator algorithm: {algo!r}. Use 'gcc_phat' or 'matched_filter'.")


if __name__ == "__main__":
    """Self-test: estimate a known delay on synthetic data."""
    sr = 16000
    duration_s = 3.0
    true_delay_ms = 87.5
    true_delay_samples = int(round(true_delay_ms * sr / 1000.0))

    # Synthetic ref: shaped noise.
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1.0, int(duration_s * sr)).astype(np.float32)
    ref = np.convolve(ref, np.ones(8) / 8, mode="same").astype(np.float32)
    ref *= 0.5 / (np.std(ref) + 1e-12)

    # Synthetic mic = delayed echo + near-end speech + noise.
    echo = np.zeros_like(ref)
    echo[true_delay_samples:] = ref[:-true_delay_samples] * 0.3
    near = rng.normal(0, 0.05, ref.size).astype(np.float32)
    noise = rng.normal(0, 0.02, ref.size).astype(np.float32)
    mic = echo + near + noise

    # 自检沿用共享默认（窄搜索窗 [0,100]ms 覆盖 87.5ms 合成延迟），
    # 仅显式加快更新节奏以缩短收敛时间。
    cfg = StreamingDelayEstimatorConfig(
        sample_rate=sr,
        update_interval_ms=50.0,
    )
    est = StreamingDelayEstimator(cfg)

    frame_size = 256
    estimates = []
    for i in range(0, ref.size - frame_size, frame_size):
        est.push_ref(ref[i : i + frame_size])
        est.push_mic(mic[i : i + frame_size])
        estimates.append(est.delay_ms)

    # Use the last 30% of estimates after convergence.
    tail = estimates[len(estimates) // 2 :]
    mean_est = float(np.mean(tail))
    print(f"True delay: {true_delay_ms:.2f} ms")
    print(f"Estimated delay (mean of last half): {mean_est:.2f} ms")
    print(f"Error: {abs(mean_est - true_delay_ms):.2f} ms")

    # Also verify aligned ref extraction.
    est.reset()
    aligned_ref = []
    for i in range(0, ref.size - frame_size, frame_size):
        est.push_ref(ref[i : i + frame_size])
        est.push_mic(mic[i : i + frame_size])
        aligned_ref.append(est.get_aligned_ref_frame(frame_size))
    aligned_ref = np.concatenate(aligned_ref)
    print(f"Aligned ref length: {aligned_ref.size} samples")
