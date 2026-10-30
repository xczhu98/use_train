"""Tests for tasks.aec.inference.streaming_delay_estimator."""
import math
import pytest

import unittest
from unittest.mock import patch

import numpy as np

from tasks.aec.inference.streaming_delay_estimator import (
    AECStreamAlignConfig,
    StreamingDelayEstimator,
    StreamingDelayEstimatorConfig,
    _RingBuffer,
    create_delay_estimator,
)


class TestRingBuffer(unittest.TestCase):
    def test_basic_push_read(self):
        buf = _RingBuffer(10)
        buf.push(np.arange(5, dtype=np.float32))
        assert buf.size == 5
        np.testing.assert_array_equal(
            buf.read_oldest(5), np.arange(5, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            buf.read_newest(5), np.arange(5, dtype=np.float32)
        )

    def test_overwrite(self):
        buf = _RingBuffer(5)
        buf.push(np.arange(8, dtype=np.float32))  # 0..7
        assert buf.size == 5
        np.testing.assert_array_equal(buf.read_oldest(5), np.arange(3, 8))
        np.testing.assert_array_equal(buf.read_newest(5), np.arange(3, 8))

    def test_read_pads_with_zeros(self):
        buf = _RingBuffer(10)
        buf.push(np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(
            buf.read_oldest(5),
            np.array([0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float32),
        )


class TestStreamingDelayEstimator(unittest.TestCase):
    def _make_stream(self, sr, duration_s, delay_ms, echo_scale=0.3):
        rng = np.random.default_rng(42)
        delay_samples = int(round(delay_ms * sr / 1000.0))
        ref = rng.normal(0, 1.0, int(duration_s * sr)).astype(np.float32)
        ref = np.convolve(ref, np.ones(8) / 8, mode="same").astype(np.float32)
        ref *= 0.5 / (np.std(ref) + 1e-12)

        echo = np.zeros_like(ref)
        echo[delay_samples:] = ref[:-delay_samples] * echo_scale
        near = rng.normal(0, 0.05, ref.size).astype(np.float32)
        noise = rng.normal(0, 0.02, ref.size).astype(np.float32)
        mic = echo + near + noise
        return ref, mic

    def test_estimates_known_delay(self):
        sr = 16000
        delay_ms = 87.5
        ref, mic = self._make_stream(sr, 3.0, delay_ms)

        cfg = StreamingDelayEstimatorConfig(
            sample_rate=sr,
            buffer_duration_ms=500.0,
            coarse_delay_ms=0.0,
            max_residual_delay_ms=100.0,
            update_interval_ms=50.0,
            smooth_alpha=0.3,
        )
        est = StreamingDelayEstimator(cfg)

        frame_size = 256
        estimates = []
        for i in range(0, ref.size - frame_size, frame_size):
            est.push_ref(ref[i : i + frame_size])
            est.push_mic(mic[i : i + frame_size])
            estimates.append(est.delay_ms)

        tail = estimates[len(estimates) // 2 :]
        mean_est = float(np.mean(tail))
        assert abs(mean_est - delay_ms) < 20.0, (
            f"Estimated delay {mean_est:.2f} ms too far from true {delay_ms:.2f} ms"
        )

    def test_coarse_delay_reduces_search(self):
        sr = 16000
        true_delay_ms = 120.0
        ref, mic = self._make_stream(sr, 3.0, true_delay_ms)

        cfg = StreamingDelayEstimatorConfig(
            sample_rate=sr,
            buffer_duration_ms=500.0,
            coarse_delay_ms=100.0,       # known coarse
            max_residual_delay_ms=50.0,  # fine search around 100ms
            update_interval_ms=50.0,
            smooth_alpha=0.3,
        )
        est = StreamingDelayEstimator(cfg)

        frame_size = 256
        estimates = []
        for i in range(0, ref.size - frame_size, frame_size):
            est.push_ref(ref[i : i + frame_size])
            est.push_mic(mic[i : i + frame_size])
            estimates.append(est.delay_ms)

        tail = estimates[len(estimates) // 2 :]
        mean_est = float(np.mean(tail))
        assert abs(mean_est - true_delay_ms) < 5.0

    def test_zero_residual_delay_locks_to_coarse_without_crashing(self):
        sr = 16000
        coarse_delay_ms = 25.0
        ref, mic = self._make_stream(sr, 2.0, coarse_delay_ms, echo_scale=0.5)
        cfg = StreamingDelayEstimatorConfig(
            sample_rate=sr,
            buffer_duration_ms=500.0,
            coarse_delay_ms=coarse_delay_ms,
            max_residual_delay_ms=0.0,
            update_interval_ms=50.0,
            smooth_alpha=0.3,
        )
        est = StreamingDelayEstimator(cfg)

        frame_size = 256
        expected_delay = int(round(coarse_delay_ms * sr / 1000.0))
        for i in range(0, ref.size - frame_size, frame_size):
            est.push_ref(ref[i : i + frame_size])
            est.push_mic(mic[i : i + frame_size])
            self.assertEqual(est.delay_samples, expected_delay)

        self.assertEqual(est.get_aligned_ref_frame(frame_size).size, frame_size)

    def test_silent_ref_skips_update(self):
        sr = 16000
        cfg = StreamingDelayEstimatorConfig(sample_rate=sr)
        est = StreamingDelayEstimator(cfg)
        silence = np.zeros(256, dtype=np.float32)
        mic = np.random.normal(0, 0.1, 256).astype(np.float32)
        for _ in range(50):
            est.push_ref(silence)
            est.push_mic(mic)
        # 静默 ref 永不触发更新：估计值停留在 coarse 初始值。
        self.assertEqual(est.delay_samples, est._coarse_samples)

    def test_nearend_only_multiple_seeds_never_locks(self):
        # 显式 coarse=0：本测试语义是“永不锁定”，估计值必须停留在 0。
        cfg = StreamingDelayEstimatorConfig(
            buffer_duration_ms=500.0,
            coarse_delay_ms=0.0,
            max_residual_delay_ms=400.0,
            update_interval_ms=100.0,
            min_observation_ms=100.0,
            acps_alpha=0.9,
        )
        for seed in range(10):
            rng = np.random.default_rng(seed)
            ref = rng.normal(size=16000).astype(np.float32)
            mic = rng.normal(size=16000).astype(np.float32)
            est = StreamingDelayEstimator(cfg)
            for start in range(0, ref.size, 160):
                est.push_ref(ref[start : start + 160])
                est.push_mic(mic[start : start + 160])
            with self.subTest(seed=seed):
                self.assertEqual(est.delay_samples, 0)
                self.assertEqual(est._history, [])

    def test_confidence_gate_captures_normal_positive_delay(self):
        sr = 16000
        delay_samples = 640
        rng = np.random.default_rng(731)
        ref = rng.normal(size=2 * sr).astype(np.float32)
        mic = np.zeros_like(ref)
        mic[delay_samples:] = 0.3 * ref[:-delay_samples]
        mic += rng.normal(0.0, 0.05, mic.size).astype(np.float32)
        est = StreamingDelayEstimator(
            StreamingDelayEstimatorConfig(
                max_residual_delay_ms=100.0,
                update_interval_ms=50.0,
            )
        )
        for start in range(0, ref.size, 160):
            est.push_ref(ref[start : start + 160])
            est.push_mic(mic[start : start + 160])
        self.assertLessEqual(abs(est.delay_samples - delay_samples), 2)

    def test_confidence_helper_accepts_exact_positive_and_negative_integer_shifts(self):
        rng = np.random.default_rng(193)
        ref = rng.normal(size=128)
        est = StreamingDelayEstimator(StreamingDelayEstimatorConfig(acps_alpha=0.0))

        for lag in (17, -17):
            mic = np.zeros_like(ref)
            if lag > 0:
                mic[lag:] = ref[:-lag]
            else:
                mic[:lag] = ref[-lag:]
            with self.subTest(lag=lag):
                self.assertTrue(
                    est._selected_lag_confidence(mic, ref, lag, candidate_count=9)
                )

    def test_confidence_helper_rejects_each_low_variance_side_after_scaling(self):
        n = 128
        pattern = np.tile(np.array([-1.0, 1.0]), (n - 1 + 1) // 2)[: n - 1]
        nearly_constant_diff = 1.0 + 1.0e-10 * pattern
        healthy_diff = 1.0e6 * pattern
        nearly_constant = np.concatenate(([0.0], np.cumsum(nearly_constant_diff)))
        healthy = np.concatenate(([0.0], np.cumsum(healthy_diff)))
        est = StreamingDelayEstimator(StreamingDelayEstimatorConfig(acps_alpha=0.0))

        for low_variance_side, mic, ref in (
            ("mic", nearly_constant * 1.0e4, healthy * 1.0e-4),
            ("ref", healthy * 1.0e-4, nearly_constant * 1.0e4),
        ):
            with self.subTest(low_variance_side=low_variance_side):
                self.assertIsNone(
                    est._selected_lag_confidence(mic, ref, 0, candidate_count=1)
                )

    def test_confidence_helper_fails_closed_when_threshold_reaches_one(self):
        rng = np.random.default_rng(194)
        signal = rng.normal(size=128)
        n_overlap = signal.size - 1

        for threshold in (1.0, 1.01):
            false_accept_prob = 2.0 * math.exp(
                -0.5 * n_overlap * threshold * threshold
            )
            est = StreamingDelayEstimator(
                StreamingDelayEstimatorConfig(
                    acps_alpha=0.0,
                    false_accept_prob=false_accept_prob,
                )
            )
            with self.subTest(threshold=threshold):
                self.assertIsNone(
                    est._selected_lag_confidence(
                        signal, signal, 0, candidate_count=1
                    )
                )

    def test_invalid_confidence_boundary_does_not_pollute_acps_state(self):
        cfg = StreamingDelayEstimatorConfig(
            sample_rate=1000,
            buffer_duration_ms=300.0,
            coarse_delay_ms=0.0,
            max_residual_delay_ms=300.0,
            update_interval_ms=300.0,
            min_observation_ms=256.0,
        )
        est = StreamingDelayEstimator(cfg)
        rng = np.random.default_rng(99)
        ref = rng.normal(size=300).astype(np.float32)
        est.push_ref(ref)

        def invalid_candidate(*_args):
            est._R_accum = np.ones(513, dtype=np.complex128)
            est._acps_nfft = 1024
            return 299.0, 1.0, 0.1

        est._gcc_phat_tau_acps = invalid_candidate
        est.push_mic(ref.copy())
        self.assertIsNone(est._R_accum)
        self.assertEqual(est._acps_nfft, 0)
        self.assertEqual(est._history, [])
        self.assertEqual(est.delay_samples, 0)

    def test_reset(self):
        cfg = StreamingDelayEstimatorConfig(sample_rate=16000)
        est = StreamingDelayEstimator(cfg)
        est.push_ref(np.ones(256, dtype=np.float32))
        est.push_mic(np.ones(256, dtype=np.float32))
        est.reset()
        self.assertEqual(est._ref_buf.size, 0)
        self.assertEqual(est._mic_buf.size, 0)
        # reset 后估计值回到 coarse 初始值（默认 coarse=50ms → 800 样本）。
        self.assertEqual(est.delay_samples, est._coarse_samples)
        self.assertEqual(len(est._history), 0)

    def test_aligned_ref_frame(self):
        sr = 16000
        delay_ms = 64.0
        delay_samples = int(round(delay_ms * sr / 1000.0))
        ref, mic = self._make_stream(sr, 2.0, delay_ms, echo_scale=0.5)

        cfg = StreamingDelayEstimatorConfig(
            sample_rate=sr,
            buffer_duration_ms=500.0,
            max_residual_delay_ms=100.0,
            update_interval_ms=50.0,
            smooth_alpha=0.3,
        )
        est = StreamingDelayEstimator(cfg)

        frame_size = 256
        for i in range(0, ref.size - frame_size, frame_size):
            est.push_ref(ref[i : i + frame_size])
            est.push_mic(mic[i : i + frame_size])

        # Wait for convergence and then check aligned ref.
        aligned = est.get_aligned_ref_frame(frame_size)
        assert aligned.size == frame_size
        # Aligned ref should be delayed relative to newest ref by ~delay_samples.
        newest_ref = est._ref_buf.read_newest(frame_size)
        # The very first samples of aligned should be old ref content (or zeros
        # if not enough history), while newest_ref starts with recent content.
        assert not np.allclose(aligned, newest_ref)

    def test_acps_uses_current_sig_size(self):
        """ACPS reconstruction must track the current signal size, not a stale one.

        When the buffered window length changes while the FFT size stays the
        same, the previous code kept a stale signal size and reconstructed the
        cross-correlation with the wrong negative-lag length.  This could
        shift the peak by one sample and break Python/C++ parity at small
        delays.
        """
        cfg = StreamingDelayEstimatorConfig(
            sample_rate=16000,
            buffer_duration_ms=500.0,
            max_residual_delay_ms=100.0,
            update_interval_ms=50.0,
            smooth_alpha=0.3,
            acps_alpha=0.9,
        )
        est = StreamingDelayEstimator(cfg)

        def _make_pair(n_samples, delay_samples):
            rng = np.random.default_rng(42)
            ref = rng.normal(0, 0.5, n_samples).astype(np.float32)
            ref = np.convolve(ref, np.ones(8) / 8, mode="same").astype(np.float32)
            sig = np.zeros_like(ref)
            if delay_samples > 0:
                sig[delay_samples:] = ref[:-delay_samples] * 0.3
            sig += rng.normal(0, 0.05, n_samples).astype(np.float32)
            return sig, ref

        # Prime ACPS with a 400-sample window (nfft=1024, delay=50 samples).
        sig1, ref1 = _make_pair(400, 50)
        result1 = est._gcc_phat_tau_acps(sig1, ref1, search_center=0)
        self.assertIsNotNone(result1)
        self.assertEqual(est._acps_nfft, 1024)

        # A 401-sample window still fits in nfft=1024.  With the stale-bug,
        # the reconstructed correlation length would be 400 + 401 - 1 = 800
        # instead of 801, potentially shifting the peak.
        sig2, ref2 = _make_pair(401, 50)
        result2 = est._gcc_phat_tau_acps(sig2, ref2, search_center=0)
        self.assertIsNotNone(result2)
        tau, _, _ = result2
        self.assertGreaterEqual(tau, 0.0)
        # The true delay is 50 samples; allow a few samples of interpolation jitter.
        self.assertLess(abs(tau - 50.0), 5.0)

    @staticmethod
    def _state_snapshot(est):
        return (
            est._ref_buf.size,
            est._ref_buf.read_newest(est._ref_buf.capacity),
            est._mic_buf.size,
            est._mic_buf.read_newest(est._mic_buf.capacity),
            est._samples_since_update,
            est._frame_count,
            est.delay_samples,
            tuple(est._history),
            None if est._R_accum is None else est._R_accum.copy(),
            est._acps_nfft,
        )

    def _assert_state_equal(self, actual, expected):
        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actual[2], expected[2])
        np.testing.assert_array_equal(actual[3], expected[3])
        self.assertEqual(actual[4:8], expected[4:8])
        if expected[8] is None:
            self.assertIsNone(actual[8])
        else:
            np.testing.assert_array_equal(actual[8], expected[8])
        self.assertEqual(actual[9], expected[9])

    def test_nonfinite_push_is_atomic_and_recovers(self):
        est = StreamingDelayEstimator(
            StreamingDelayEstimatorConfig(update_interval_ms=10.0)
        )
        valid = np.linspace(-1.0, 1.0, 160, dtype=np.float32)
        est.push_ref(valid)
        est.push_mic(valid)

        for push, bad_value in (
            (est.push_ref, np.nan),
            (est.push_ref, np.inf),
            (est.push_mic, np.nan),
            (est.push_mic, -np.inf),
        ):
            before = self._state_snapshot(est)
            with self.subTest(push=push.__name__, bad_value=bad_value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    push(np.array([0.0, bad_value, 1.0], dtype=np.float32))
                self._assert_state_equal(self._state_snapshot(est), before)

        est.push_ref(valid)
        est.push_mic(valid)
        self.assertTrue(np.isfinite(est.delay_ms))

    def test_public_push_requires_one_dimensional_frames(self):
        est = StreamingDelayEstimator()
        for push in (est.push_ref, est.push_mic):
            with self.subTest(push=push.__name__, shape=(0,)):
                push(np.zeros(0, dtype=np.float32))
            for bad in (
                np.array(0.0, dtype=np.float32),
                np.zeros((1, 2), dtype=np.float32),
            ):
                with self.subTest(push=push.__name__, shape=bad.shape):
                    with self.assertRaisesRegex(ValueError, "one-dimensional"):
                        push(bad)

        self.assertEqual(est._ref_buf.size, 0)
        self.assertEqual(est._mic_buf.size, 0)
        self.assertEqual(est._frame_count, 1)
        self.assertEqual(est._samples_since_update, 0)

    def test_matched_filter_fallback_rejects_invalid_frames_atomically(self):
        def snapshot(estimator):
            return {
                "ref_raw": estimator._ref_raw_buf._buf.copy(),
                "ref_raw_size": estimator._ref_raw_buf._size,
                "ref_raw_head": estimator._ref_raw_buf._head,
                "ref_accum": estimator._ref_accum.copy(),
                "mic_accum": estimator._mic_accum.copy(),
                "render": estimator._render_buf._buf.copy(),
                "render_head": estimator._render_buf._head,
                "render_aa": estimator._render_decimator._aa_state.copy(),
                "render_hp": estimator._render_decimator._hp_state.copy(),
                "capture_aa": estimator._capture_decimator._aa_state.copy(),
                "capture_hp": estimator._capture_decimator._hp_state.copy(),
                "filters": tuple(
                    filt._h.copy() for filt in estimator._filter_bank._filters
                ),
                "histogram": estimator._lag_aggregator._histogram.copy(),
                "histogram_ring": estimator._lag_aggregator._ring.copy(),
                "histogram_ring_idx": estimator._lag_aggregator._ring_idx,
                "histogram_ring_count": estimator._lag_aggregator._ring_count,
                "histogram_significant": (
                    estimator._lag_aggregator._significant_found
                ),
                "delay": estimator.delay_samples,
                "frame_count": estimator._frame_count,
            }

        def assert_snapshot_equal(actual, expected):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                if isinstance(actual[key], np.ndarray):
                    np.testing.assert_array_equal(actual[key], expected[key])
                elif key == "filters":
                    for actual_filter, expected_filter in zip(
                        actual[key], expected[key]
                    ):
                        np.testing.assert_array_equal(actual_filter, expected_filter)
                else:
                    self.assertEqual(actual[key], expected[key], key)

        cfg = StreamingDelayEstimatorConfig(
            algorithm="matched_filter",
            sample_rate=8000,
            mf_initial_threshold=1,
            mf_converged_threshold=2,
        )
        with self.assertWarnsRegex(RuntimeWarning, "falling back to Python"):
            est = create_delay_estimator(cfg)
        with self.assertWarnsRegex(RuntimeWarning, "falling back to Python"):
            control = create_delay_estimator(cfg)

        rng = np.random.default_rng(23)
        ref = rng.normal(0.0, 0.2, 64 * 80).astype(np.float32)
        mic = np.zeros_like(ref)
        mic[64:] = 0.8 * ref[:-64]
        for start in range(0, ref.size, 64):
            ref_frame = ref[start : start + 64]
            mic_frame = mic[start : start + 64]
            est.push_ref(ref_frame)
            est.push_mic(mic_frame)
            control.push_ref(ref_frame)
            control.push_mic(mic_frame)

        self.assertTrue(any(np.any(filt._h) for filt in est._filter_bank._filters))
        self.assertGreater(est._lag_aggregator._ring_count, 0)
        self.assertGreater(est.delay_samples, 0)

        before_empty = snapshot(est)
        est.push_ref(np.zeros(0, dtype=np.float32))
        est.push_mic(np.zeros(0, dtype=np.float32))
        assert_snapshot_equal(snapshot(est), before_empty)

        for method_name, bad_value in (
            ("push_ref", np.nan),
            ("push_ref", np.inf),
            ("push_mic", np.nan),
            ("push_mic", -np.inf),
        ):
            bad = np.zeros(64, dtype=np.float32)
            bad[17] = bad_value
            before = snapshot(est)
            with self.subTest(method=method_name, bad=bad_value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    getattr(est, method_name)(bad)
                assert_snapshot_equal(snapshot(est), before)

        for method_name in ("push_ref", "push_mic"):
            for bad in (
                np.array(0.0, dtype=np.float32),
                np.zeros((1, 2), dtype=np.float32),
            ):
                before = snapshot(est)
                with self.subTest(method=method_name, shape=bad.shape):
                    with self.assertRaisesRegex(ValueError, "one-dimensional"):
                        getattr(est, method_name)(bad)
                    assert_snapshot_equal(snapshot(est), before)

        recovery_ref = rng.normal(0.0, 0.2, 64).astype(np.float32)
        recovery_mic = rng.normal(0.0, 0.2, 64).astype(np.float32)
        est.push_ref(recovery_ref)
        est.push_mic(recovery_mic)
        control.push_ref(recovery_ref)
        control.push_mic(recovery_mic)
        assert_snapshot_equal(snapshot(est), snapshot(control))
        self.assertTrue(np.isfinite(est.delay_ms))

    def test_matched_filter_fallback_rejects_huge_finite_atomically_and_recovers(self):
        cfg = StreamingDelayEstimatorConfig(
            algorithm="matched_filter",
            sample_rate=8000,
            mf_initial_threshold=1,
            mf_converged_threshold=2,
        )
        with self.assertWarnsRegex(RuntimeWarning, "falling back to Python"):
            est = create_delay_estimator(cfg)
        with self.assertWarnsRegex(RuntimeWarning, "falling back to Python"):
            control = create_delay_estimator(cfg)

        valid = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
        est.push_ref(valid)
        est.push_mic(valid)
        control.push_ref(valid)
        control.push_mic(valid)

        for method_name in ("push_ref", "push_mic"):
            before_ref = est._ref_accum.copy()
            before_mic = est._mic_accum.copy()
            before_aligned = est.get_aligned_ref_frame(64).copy()
            with self.subTest(method=method_name):
                with self.assertRaisesRegex(ValueError, "normalized audio range"):
                    getattr(est, method_name)(
                        np.array([0.0, 1.0e20, -1.0e20], dtype=np.float32)
                    )
                np.testing.assert_array_equal(est._ref_accum, before_ref)
                np.testing.assert_array_equal(est._mic_accum, before_mic)
                np.testing.assert_array_equal(
                    est.get_aligned_ref_frame(64), before_aligned
                )

        est.push_ref(valid)
        est.push_mic(valid)
        control.push_ref(valid)
        control.push_mic(valid)
        np.testing.assert_array_equal(est._ref_accum, control._ref_accum)
        np.testing.assert_array_equal(est._mic_accum, control._mic_accum)
        np.testing.assert_array_equal(
            est.get_aligned_ref_frame(64), control.get_aligned_ref_frame(64)
        )
        self.assertEqual(est.delay_samples, control.delay_samples)

    def test_matched_filter_fallback_one_sided_pending_is_bounded(self):
        cfg = StreamingDelayEstimatorConfig(
            algorithm="matched_filter",
            sample_rate=8000,
        )
        with self.assertWarnsRegex(RuntimeWarning, "falling back to Python"):
            est = create_delay_estimator(cfg)

        frame = np.zeros(100, dtype=np.float32)
        accepted = 0
        for _ in range(1000):  # 100k attempted one-sided samples.
            try:
                est.push_ref(frame)
                accepted += frame.size
            except ValueError as exc:
                self.assertRegex(str(exc), "pending skew")

        self.assertEqual(est._max_pending_samples, 16384)
        self.assertLessEqual(est._ref_accum.size, est._max_pending_samples)
        self.assertLessEqual(accepted, est._max_pending_samples)
        self.assertEqual(est._mic_accum.size, 0)

        # Draining the accepted reference samples restores normal paired flow.
        est.push_mic(np.zeros(accepted, dtype=np.float32))
        self.assertLess(est._ref_accum.size, est._block_size)
        self.assertLess(est._mic_accum.size, est._block_size)

    def test_matched_filter_internal_nonfinite_resets_and_raises(self):
        cfg = StreamingDelayEstimatorConfig(
            algorithm="matched_filter",
            sample_rate=8000,
        )
        with self.assertWarnsRegex(RuntimeWarning, "falling back to Python"):
            est = create_delay_estimator(cfg)

        est._render_decimator._aa_state[0, 0] = np.nan
        est.push_ref(np.zeros(64, dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "internal non-finite state"):
            est.push_mic(np.zeros(64, dtype=np.float32))

        self.assertEqual(est._ref_accum.size, 0)
        self.assertEqual(est._mic_accum.size, 0)
        self.assertEqual(est.delay_samples, 0)
        self.assertTrue(np.all(np.isfinite(est._render_decimator._aa_state)))
        self.assertTrue(
            all(
                np.all(np.isfinite(filt._h))
                for filt in est._filter_bank._filters
            )
        )

    def test_private_gcc_rejects_nonfinite_without_polluting_acps(self):
        est = StreamingDelayEstimator(StreamingDelayEstimatorConfig())
        rng = np.random.default_rng(8)
        ref = rng.normal(size=512).astype(np.float32)
        sig = np.roll(ref, 32)
        self.assertIsNotNone(est._gcc_phat_tau_acps(sig, ref, search_center=0))
        before = est._R_accum.copy()

        for method in (est._gcc_phat_tau, est._gcc_phat_tau_acps):
            for bad_value in (np.nan, np.inf, -np.inf):
                bad_sig = sig.copy()
                bad_sig[17] = bad_value
                with self.subTest(method=method.__name__, bad_value=bad_value):
                    self.assertIsNone(method(bad_sig, ref, search_center=0))
                    np.testing.assert_array_equal(est._R_accum, before)

    def test_acps_pollution_is_discarded_and_restarted(self):
        est = StreamingDelayEstimator(StreamingDelayEstimatorConfig())
        rng = np.random.default_rng(8)
        ref = rng.normal(size=512).astype(np.float32)
        sig = np.roll(ref, 32)
        est._R_accum = np.full(513, np.nan + 0j)
        est._acps_nfft = 1024
        result = est._gcc_phat_tau_acps(sig, ref, search_center=0)
        self.assertIsNotNone(result)
        self.assertTrue(np.all(np.isfinite(est._R_accum)))

        fresh = StreamingDelayEstimator(StreamingDelayEstimatorConfig())
        fresh_result = fresh._gcc_phat_tau_acps(sig, ref, search_center=0)
        self.assertIsNotNone(fresh_result)
        np.testing.assert_array_equal(est._R_accum, fresh._R_accum)
        np.testing.assert_allclose(result, fresh_result, rtol=0.0, atol=0.0)

    def test_strict_config_contract_and_shared_defaults(self):
        defaults = AECStreamAlignConfig.from_dict({})
        cfg = defaults.to_estimator_config(16000)
        self.assertEqual(defaults.acps_alpha, 0.8)
        self.assertEqual(cfg.coarse_delay_ms, 50.0)
        self.assertEqual(cfg.max_residual_delay_ms, 50.0)
        self.assertEqual(cfg.smooth_alpha, 0.15)
        self.assertEqual(cfg.acps_alpha, 0.8)
        self.assertEqual(cfg.peak_threshold, 0.05)
        self.assertEqual(cfg.false_accept_prob, 1e-4)
        self.assertEqual(cfg.min_observation_ms, 200.0)

        self.assertEqual(AECStreamAlignConfig.from_dict(None), defaults)
        yaml_cases = (
            [],
            {"unknown": 1},
            {"enabled": "false"},
            {"enabled": 1},
            {"frame_size": True},
            {"frame_size": 0},
            {"frame_size": 160.0},
            {"coarse_ms": -1.0},
            {"buffer_ms": float("nan")},
        )
        for raw in yaml_cases:
            with self.subTest(yaml_config=raw), self.assertRaises(ValueError):
                AECStreamAlignConfig.from_dict(raw)

        finite_fields = (
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
        for field in finite_fields:
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    StreamingDelayEstimatorConfig(**{field: value})

        estimator_cases = (
            {"sample_rate": True},
            {"sample_rate": 16000.0},
            {"sample_rate": 0},
            {"buffer_duration_ms": 0.0},
            {"coarse_delay_ms": -1.0},
            {"max_residual_delay_ms": -1.0},
            {"update_interval_ms": 0.0},
            {"min_observation_ms": 0.0},
            {"min_observation_ms": 501.0, "buffer_duration_ms": 500.0},
            {"smooth_alpha": 0.0},
            {"smooth_alpha": 1.01},
            {"acps_alpha": -0.01},
            {"acps_alpha": 1.0},
            {"peak_threshold": 0.0},
            {"peak_threshold": 1.01},
            {"peak_prominence_db": -1.0},
            {"false_accept_prob": 0.0},
            {"false_accept_prob": 1.0},
            {"eps": 0.0},
            {"algorithm": "not-gcc"},
        )
        for kwargs in estimator_cases:
            with self.subTest(estimator_config=kwargs), self.assertRaises(ValueError):
                StreamingDelayEstimatorConfig(**kwargs)

    def test_update_interval_runs_once_per_push_and_preserves_remainder(self):
        est = StreamingDelayEstimator(
            StreamingDelayEstimatorConfig(
                sample_rate=1000,
                buffer_duration_ms=500.0,
                update_interval_ms=160.0,
                min_observation_ms=100.0,
            )
        )
        updates = []
        est._update_delay = lambda: updates.append(est._samples_since_update)

        # 单个超大帧即使跨过两个边界，也只触发一次更新；modulo 保留
        # 90 个样本的 cadence 余数。
        est.push_mic(np.ones(410, dtype=np.float32))
        self.assertEqual(len(updates), 1)
        self.assertEqual(est._samples_since_update, 90)

        est.push_mic(np.ones(69, dtype=np.float32))
        self.assertEqual(len(updates), 1)
        self.assertEqual(est._samples_since_update, 159)

        est.push_mic(np.ones(1, dtype=np.float32))
        self.assertEqual(len(updates), 2)
        self.assertEqual(est._samples_since_update, 0)

    def test_aligned_frame_rejects_oversized_request(self):
        est = StreamingDelayEstimator(StreamingDelayEstimatorConfig())
        capacity = est._ref_buf.capacity
        with self.assertRaisesRegex(ValueError, "capacity"):
            est.get_aligned_ref_frame(capacity + 1)

        est._current_delay_samples = 1.0
        with self.assertRaisesRegex(ValueError, "capacity"):
            est.get_aligned_ref_frame(capacity)

    def test_factory_fallbacks_share_aligned_frame_argument_contract(self):
        with patch(
            "tasks.aec.inference.streaming_delay_estimator._load_delay_estimator_extension",
            side_effect=ImportError("forced Python fallback"),
        ), self.assertWarnsRegex(RuntimeWarning, "falling back to Python"):
            gcc = create_delay_estimator(StreamingDelayEstimatorConfig())

        with self.assertWarnsRegex(RuntimeWarning, "falling back to Python"):
            matched = create_delay_estimator(
                StreamingDelayEstimatorConfig(
                    algorithm="matched_filter",
                    sample_rate=8000,
                )
            )

        for estimator in (gcc, matched):
            with self.subTest(backend=type(estimator._estimator).__name__):
                self.assertEqual(
                    estimator.get_aligned_ref_frame(np.int64(0)).shape, (0,)
                )
                for frame_size, message in (
                    (True, "integer"),
                    (1.0, "integer"),
                    (-1, "non-negative"),
                    (estimator._aligned_ref_capacity + 1, "capacity"),
                    (10**18, "capacity"),
                ):
                    with self.subTest(frame_size=frame_size):
                        with self.assertRaisesRegex(ValueError, message):
                            estimator.get_aligned_ref_frame(frame_size)

    def test_factory_input_preflight_is_algorithm_specific_and_read_only(self):
        with patch(
            "tasks.aec.inference.streaming_delay_estimator._load_delay_estimator_extension",
            side_effect=ImportError("forced Python fallback"),
        ), self.assertWarnsRegex(RuntimeWarning, "falling back to Python"):
            gcc = create_delay_estimator(StreamingDelayEstimatorConfig())
        with self.assertWarnsRegex(RuntimeWarning, "falling back to Python"):
            matched = create_delay_estimator(
                StreamingDelayEstimatorConfig(
                    algorithm="matched_filter",
                    sample_rate=8000,
                )
            )

        self.assertIsNone(gcc.input_abs_limit)
        self.assertEqual(matched.input_abs_limit, 16.0)
        for estimator in (gcc, matched):
            with self.subTest(algorithm=estimator._estimator.__class__.__name__):
                with self.assertRaises(AttributeError):
                    estimator.input_abs_limit = 1.0

        large_finite = np.array([-1.0e20, 1.0e20], dtype=np.float32)
        np.testing.assert_array_equal(
            gcc.preflight_input(large_finite, "microphone"), large_finite
        )
        gcc.push_ref(large_finite)
        gcc.push_mic(large_finite)

        with self.assertRaisesRegex(ValueError, "normalized audio range"):
            matched.preflight_input(
                np.array([0.0, 16.01], dtype=np.float32), "microphone"
            )


if __name__ == "__main__":
    unittest.main()

pytestmark = [pytest.mark.unit]
