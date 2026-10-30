"""AEC3-style matched filter delay estimator.

Reimplements the delay estimation pipeline from WebRTC AEC3:
  Decimator → MatchedFilter (NLMS) → LagAggregator (histogram voting)

Reference: WebRTC AEC3 (chromium), simplified for single-channel 16kHz AEC.

Compared to GCC-PHAT:
  - Time-domain adaptive filter (NLMS) instead of frequency-domain correlation
  - Inherent temporal smoothing via recursive filter update
  - Histogram voting for stable delay selection
  - 4x downsampling for efficiency
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfilt


# 24 dB headroom above normalized full scale accepts ordinary clipping and
# moderate upstream gain while rejecting values that can destabilize IIR/NLMS.
_NORMALIZED_AUDIO_ABS_LIMIT = 16.0
# One second at the supported 16 kHz C++ rate; also exceeds the default 9728
# sample matched-filter search span, without allowing an absent peer to grow.
_MAX_PENDING_SAMPLES = 16384


# ---------------------------------------------------------------------------
# Decimator: anti-aliasing + downsampling
# ---------------------------------------------------------------------------
class _Decimator:
    """4x decimator with Butterworth anti-aliasing and high-pass noise reduction."""

    def __init__(self, factor: int = 4, sample_rate: int = 16000):
        self._factor = factor
        nyq = sample_rate / 2.0
        self._aa_sos = butter(3, 0.85 / factor, btype="low", output="sos")
        self._hp_sos = butter(2, 200.0 / nyq, btype="high", output="sos")
        self._aa_state = np.zeros((self._aa_sos.shape[0], 2))
        self._hp_state = np.zeros((self._hp_sos.shape[0], 2))

    def process(self, block: np.ndarray) -> np.ndarray:
        filtered, aa_state = sosfilt(self._aa_sos, block, zi=self._aa_state)
        filtered, hp_state = sosfilt(self._hp_sos, filtered, zi=self._hp_state)
        downsampled = filtered[:: self._factor].astype(np.float32)
        if not all(
            np.all(np.isfinite(value))
            for value in (filtered, aa_state, hp_state, downsampled)
        ):
            raise RuntimeError("matched-filter internal non-finite decimator state")
        self._aa_state = aa_state
        self._hp_state = hp_state
        return downsampled

    def reset(self) -> None:
        self._aa_state = np.zeros((self._aa_sos.shape[0], 2))
        self._hp_state = np.zeros((self._hp_sos.shape[0], 2))


# ---------------------------------------------------------------------------
# Ring buffer for downsampled render signal
# ---------------------------------------------------------------------------
class _DownsampledRenderBuffer:

    def __init__(self, capacity: int):
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._head = 0

    def insert(self, block: np.ndarray) -> None:
        n = block.size
        self._head = (self._head - n) % self._capacity
        end = self._head + n
        if end <= self._capacity:
            self._buf[self._head:end] = block
        else:
            first = self._capacity - self._head
            self._buf[self._head:] = block[:first]
            self._buf[:n - first] = block[first:]

    def read(self, offset: int, length: int) -> np.ndarray:
        out = np.empty(length, dtype=np.float32)
        start = (self._head + offset) % self._capacity
        end = start + length
        if end <= self._capacity:
            out[:] = self._buf[start:end]
        else:
            first = self._capacity - start
            out[:first] = self._buf[start:]
            out[first:] = self._buf[:length - first]
        return out

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._head = 0


# ---------------------------------------------------------------------------
# Single NLMS adaptive matched filter
# ---------------------------------------------------------------------------
class _SingleMatchedFilter:

    def __init__(self, length: int, smoothing: float = 0.7):
        self._length = length
        self._smoothing = smoothing
        self._h = np.zeros(length, dtype=np.float32)

    @property
    def length(self) -> int:
        return self._length

    def update(
        self,
        capture_block: np.ndarray,
        render_buf: _DownsampledRenderBuffer,
        alignment_shift: int,
        excitation_limit: float,
    ) -> tuple[bool, int, float, float]:
        """Run NLMS update and return (updated, lag, accuracy, error_sum)."""
        n = capture_block.size
        x2_threshold = self._length * (excitation_limit ** 2)
        error_sum = 0.0
        y_sum = 0.0

        x_matrix = np.stack([
            render_buf.read(alignment_shift + i, self._length) for i in range(n)
        ])
        y_vec = capture_block.astype(np.float32)
        candidate_h = self._h.copy()
        s_vec = x_matrix @ candidate_h
        e_vec = y_vec - s_vec
        x2_vec = np.sum(x_matrix * x_matrix, axis=1)

        error_sum = float(np.dot(e_vec, e_vec))
        y_sum = float(np.dot(y_vec, y_vec))

        for i in range(n):
            if x2_vec[i] > x2_threshold and abs(y_vec[i]) < 0.95:
                alpha = self._smoothing * e_vec[i] / (x2_vec[i] + 1e-12)
                candidate_h += alpha * x_matrix[i]
                if i + 1 < n:
                    s_vec[i+1:] = x_matrix[i+1:] @ candidate_h
                    e_vec[i+1:] = y_vec[i+1:] - s_vec[i+1:]

        values = (x_matrix, y_vec, s_vec, e_vec, x2_vec, candidate_h)
        if not all(np.all(np.isfinite(value)) for value in values) or not all(
            math.isfinite(value) for value in (error_sum, y_sum)
        ):
            raise RuntimeError("matched-filter internal non-finite NLMS state")

        lag = int(np.argmax(np.abs(candidate_h)))
        reliable = (
            lag > 2
            and lag < self._length - 10
            and y_sum > 1e-12
            and error_sum < 0.2 * y_sum
        )
        accuracy = y_sum - error_sum if y_sum > 1e-12 else 0.0
        if not math.isfinite(accuracy):
            raise RuntimeError("matched-filter internal non-finite NLMS candidate")

        self._h = candidate_h
        return reliable, lag, accuracy, error_sum

    def reset(self) -> None:
        self._h[:] = 0.0


# ---------------------------------------------------------------------------
# Multi-filter matched filter bank
# ---------------------------------------------------------------------------
class _MatchedFilterBank:

    def __init__(
        self,
        num_filters: int = 5,
        filter_length: int = 512,
        lag_shift: int = 384,
        smoothing: float = 0.7,
        excitation_limit: float = 0.005,
    ):
        self._num_filters = num_filters
        self._filter_length = filter_length
        self._lag_shift = lag_shift
        self._excitation_limit = excitation_limit
        self._filters = [_SingleMatchedFilter(filter_length, smoothing) for _ in range(num_filters)]

    @property
    def max_lag(self) -> int:
        return self._num_filters * self._lag_shift + self._filter_length

    def update(
        self,
        capture_block: np.ndarray,
        render_buf: _DownsampledRenderBuffer,
    ) -> list[dict]:
        results = []
        for i, filt in enumerate(self._filters):
            shift = i * self._lag_shift
            reliable, lag, accuracy, error_sum = filt.update(
                capture_block, render_buf, shift, self._excitation_limit,
            )
            results.append({
                "filter_idx": i,
                "updated": True,
                "reliable": reliable,
                "lag": shift + lag,
                "accuracy": accuracy,
                "error_sum": error_sum,
            })
        return results

    def reset(self) -> None:
        for f in self._filters:
            f.reset()


# ---------------------------------------------------------------------------
# Histogram lag aggregator
# ---------------------------------------------------------------------------
class _LagAggregator:

    def __init__(self, max_lag: int, window_size: int = 250,
                 initial_threshold: int = 5, converged_threshold: int = 20):
        self._max_lag = max_lag
        self._histogram = np.zeros(max_lag + 1, dtype=np.int32)
        self._window_size = window_size
        self._ring = np.zeros(window_size, dtype=np.int32)
        self._ring_idx = 0
        self._ring_count = 0
        self._initial_threshold = initial_threshold
        self._converged_threshold = converged_threshold
        self._significant_found = False

    @property
    def histogram(self) -> np.ndarray:
        """只读直方图视图，供调用方做有限性校验而不暴露可变内部数组。"""
        view = self._histogram.view()
        view.flags.writeable = False
        return view

    def aggregate(self, lag_estimates: list[dict]) -> Optional[tuple[int, str]]:
        best = None
        for est in lag_estimates:
            if est["reliable"]:
                if best is None or est["accuracy"] > best["accuracy"]:
                    best = est

        if best is not None:
            lag = best["lag"]
            if 0 <= lag <= self._max_lag:
                if self._ring_count >= self._window_size:
                    old_lag = int(self._ring[self._ring_idx])
                    self._histogram[old_lag] -= 1
                self._ring[self._ring_idx] = lag
                self._histogram[lag] += 1
                self._ring_idx = (self._ring_idx + 1) % self._window_size
                self._ring_count = min(self._ring_count + 1, self._window_size)

        if self._ring_count == 0:
            return None

        candidate = int(np.argmax(self._histogram))
        count = int(self._histogram[candidate])

        threshold = self._converged_threshold if self._significant_found else self._initial_threshold
        if count >= threshold:
            quality = "refined" if self._significant_found else "coarse"
            if count >= self._converged_threshold:
                self._significant_found = True
                quality = "refined"
            return candidate, quality

        return None

    def reset(self, hard: bool = True) -> None:
        self._histogram[:] = 0
        self._ring[:] = 0
        self._ring_idx = 0
        self._ring_count = 0
        if hard:
            self._significant_found = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class MatchedFilterDelayEstimatorConfig:
    """Configuration for the AEC3-style matched filter delay estimator."""
    sample_rate: int = 16000
    down_sampling_factor: int = 4
    num_filters: int = 5
    filter_length_subblocks: int = 32
    lag_shift_subblocks: int = 24
    smoothing: float = 0.7
    excitation_limit: float = 0.005
    histogram_window: int = 250
    initial_threshold: int = 5
    converged_threshold: int = 20
    block_size: int = 64


# ---------------------------------------------------------------------------
# Main estimator class
# ---------------------------------------------------------------------------
class MatchedFilterDelayEstimator:
    """AEC3-style delay estimator using decimation + matched filter + histogram voting.

    Provides the same external interface as StreamingDelayEstimator:
    push_ref, push_mic, delay_samples, delay_ms, get_aligned_ref_frame, reset.
    """

    def __init__(self, config: Optional[MatchedFilterDelayEstimatorConfig] = None):
        self.cfg = config or MatchedFilterDelayEstimatorConfig()
        self._sr = self.cfg.sample_rate
        self._ds = self.cfg.down_sampling_factor
        self._block_size = self.cfg.block_size
        self._sub_block_size = self._block_size // self._ds

        filter_length = self.cfg.filter_length_subblocks * self._sub_block_size
        lag_shift = self.cfg.lag_shift_subblocks * self._sub_block_size

        self._filter_bank = _MatchedFilterBank(
            num_filters=self.cfg.num_filters,
            filter_length=filter_length,
            lag_shift=lag_shift,
            smoothing=self.cfg.smoothing,
            excitation_limit=self.cfg.excitation_limit,
        )

        render_buf_capacity = self._filter_bank.max_lag + filter_length + 1024
        self._render_buf = _DownsampledRenderBuffer(render_buf_capacity)
        self._render_decimator = _Decimator(self._ds, self._sr)
        self._capture_decimator = _Decimator(self._ds, self._sr)

        self._lag_aggregator = _LagAggregator(
            max_lag=self._filter_bank.max_lag,
            window_size=self.cfg.histogram_window,
            initial_threshold=self.cfg.initial_threshold,
            converged_threshold=self.cfg.converged_threshold,
        )

        self._current_delay_samples = 0
        self._ref_accum = np.array([], dtype=np.float32)
        self._mic_accum = np.array([], dtype=np.float32)
        self._max_pending_samples = _MAX_PENDING_SAMPLES
        self._frame_count = 0

        from tasks.aec.inference.streaming_delay_estimator import _RingBuffer
        max_delay_samples = self._filter_bank.max_lag * self._ds
        self._ref_raw_buf = _RingBuffer(max_delay_samples + self._block_size * 32)

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def delay_samples(self) -> int:
        return self._current_delay_samples

    @property
    def delay_ms(self) -> float:
        return self._current_delay_samples * 1000.0 / self._sr

    @property
    def current_delay(self) -> int:
        return self._current_delay_samples

    @staticmethod
    def _validated_frame(frame: np.ndarray, name: str) -> np.ndarray:
        """整帧校验归一化音频，确保 public push 原子拒绝。"""
        normalized = np.asarray(frame, dtype=np.float32)
        if normalized.ndim != 1:
            raise ValueError(f"{name} frame must be one-dimensional")
        if not np.all(np.isfinite(normalized)):
            raise ValueError(f"{name} frame must contain only finite samples")
        if normalized.size and float(np.max(np.abs(normalized))) > _NORMALIZED_AUDIO_ABS_LIMIT:
            raise ValueError(
                f"{name} frame exceeds normalized audio range "
                f"[-{_NORMALIZED_AUDIO_ABS_LIMIT:g}, {_NORMALIZED_AUDIO_ABS_LIMIT:g}]"
            )
        return normalized

    def _check_pending_limit(self, new_size: int, name: str) -> None:
        if new_size > self._max_pending_samples:
            raise ValueError(
                f"{name} pending skew exceeds fixed limit "
                f"{self._max_pending_samples} samples"
            )

    @staticmethod
    def _append_pending(pending: np.ndarray, frame: np.ndarray) -> np.ndarray:
        if frame.size == 0:
            return pending
        if pending.size == 0:
            return frame.copy()
        return np.concatenate((pending, frame))

    def push_ref(self, frame: np.ndarray) -> None:
        frame = self._validated_frame(frame, "reference")
        self._check_pending_limit(self._ref_accum.size + frame.size, "reference")
        self._ref_raw_buf.push(frame)
        self._ref_accum = self._append_pending(self._ref_accum, frame)
        self._process_blocks_fail_closed()

    def push_mic(self, frame: np.ndarray) -> None:
        frame = self._validated_frame(frame, "microphone")
        self._check_pending_limit(self._mic_accum.size + frame.size, "microphone")
        self._mic_accum = self._append_pending(self._mic_accum, frame)
        self._process_blocks_fail_closed()

    def _process_blocks_fail_closed(self) -> None:
        try:
            self._process_blocks()
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            self.reset()
            raise RuntimeError(
                "matched-filter internal non-finite state; estimator reset"
            ) from exc

    def _process_blocks(self) -> None:
        bs = self._block_size
        while self._ref_accum.size >= bs and self._mic_accum.size >= bs:
            ref_block = self._ref_accum[:bs]
            mic_block = self._mic_accum[:bs]

            ds_ref = self._render_decimator.process(ref_block)
            ds_mic = self._capture_decimator.process(mic_block)
            if not np.all(np.isfinite(ds_ref)) or not np.all(np.isfinite(ds_mic)):
                raise RuntimeError("matched-filter internal non-finite decimator output")

            self._render_buf.insert(ds_ref)
            lag_estimates = self._filter_bank.update(ds_mic, self._render_buf)
            for estimate in lag_estimates:
                if not all(
                    math.isfinite(float(estimate[key]))
                    for key in ("lag", "accuracy", "error_sum")
                ):
                    raise RuntimeError("matched-filter internal non-finite candidate")

            result = self._lag_aggregator.aggregate(lag_estimates)
            if not np.all(np.isfinite(self._lag_aggregator.histogram)):
                raise RuntimeError("matched-filter internal non-finite histogram")
            if result is not None:
                ds_lag, _quality = result
                candidate_delay = ds_lag * self._ds
                if not math.isfinite(candidate_delay):
                    raise RuntimeError("matched-filter internal non-finite delay")
                self._current_delay_samples = candidate_delay

            self._ref_accum = self._ref_accum[bs:].copy()
            self._mic_accum = self._mic_accum[bs:].copy()
            self._frame_count += 1

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
        if frame_size == 0:
            return np.zeros(0, dtype=np.float32)
        delay = self.delay_samples
        # delay 由直方图 argmax（>= 0）乘以下采样因子得到，frame_size > 0，
        # 因此 total 恒为正。
        total = frame_size + delay
        if total > self._ref_raw_buf.capacity:
            raise ValueError(
                "frame_size + delay exceeds reference buffer capacity: "
                f"{frame_size} + {delay} > {self._ref_raw_buf.capacity}"
            )
        block = self._ref_raw_buf.read_newest(total)
        return block[:frame_size].copy()

    def reset(self) -> None:
        self._filter_bank.reset()
        self._lag_aggregator.reset(hard=True)
        self._render_buf.reset()
        self._render_decimator.reset()
        self._capture_decimator.reset()
        self._current_delay_samples = 0
        self._ref_accum = np.array([], dtype=np.float32)
        self._mic_accum = np.array([], dtype=np.float32)
        self._frame_count = 0
        from tasks.aec.inference.streaming_delay_estimator import _RingBuffer
        self._ref_raw_buf = _RingBuffer(self._ref_raw_buf.capacity)
