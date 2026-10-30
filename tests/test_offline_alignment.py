"""Unit tests for offline AEC ref/mic alignment."""

from __future__ import annotations
import pytest

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tasks.aec.inference.gcc_phat_core import gcc_phat_tau_with_confidence
from tasks.aec.inference.offline_alignment import align_farend_to_mic, gcc_phat_tau_seconds


class TestOfflineAlignment(unittest.TestCase):
    """Tests for offline GCC-PHAT alignment."""

    def _synthetic_echo(self, sr: int = 16000, delay_ms: float = 87.5):
        true_delay_samp = int(round(delay_ms * sr / 1000.0))
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1.0, sr).astype(np.float32)
        ref = np.convolve(ref, np.ones(8) / 8, mode="same").astype(np.float32)
        ref *= 0.5 / (np.std(ref) + 1e-12)
        echo = np.zeros_like(ref)
        echo[true_delay_samp:] = ref[:-true_delay_samp] * 0.3
        mic = echo + rng.normal(0, 0.05, ref.size).astype(np.float32)
        return mic, ref, true_delay_samp

    def test_gcc_phat_positive_delay(self):
        """GCC-PHAT returns positive delay when echo lags ref."""
        mic, ref, true_delay_samp = self._synthetic_echo()
        tau = gcc_phat_tau_seconds(mic, ref, 16000, bias_seconds=0.0)
        self.assertGreater(tau, 0.0)
        self.assertLess(abs(tau * 16000 - true_delay_samp), 10)

    def test_align_pads_ref_front(self):
        """apply policy pads ref front by the estimated delay."""
        mic, ref, true_delay_samp = self._synthetic_echo()
        mic_out, ref_aligned, tau = align_farend_to_mic(
            mic, ref, 16000, pad_policy="apply", bias_seconds=0.0
        )
        self.assertEqual(mic_out.size, mic.size)
        self.assertGreater(ref_aligned.size, ref.size)
        self.assertGreaterEqual(ref_aligned.size - ref.size, true_delay_samp - 5)
        self.assertEqual(ref_aligned.size - ref.size, int(tau * 16000))

    def test_removed_pad_policies_raise(self):
        """已删除的 bypass policy 必须 fail-fast。"""
        mic, ref, _ = self._synthetic_echo()
        for policy in ("skip", "log_only"):
            with self.subTest(policy=policy), self.assertRaisesRegex(
                ValueError, "only supports 'apply'"
            ):
                align_farend_to_mic(
                    mic, ref, 16000, pad_policy=policy, bias_seconds=0.0
                )

    def test_effective_tau_uses_truncated_and_clamped_samples(self):
        """apply 返回值与实际补齐都采用 truncate/clamp 后的整数采样点。"""
        mic = np.zeros(32, dtype=np.float32)
        far = np.ones(32, dtype=np.float32)
        estimate = SimpleNamespace(low_confidence=False, tau_sec=0.026)
        with patch(
            "tasks.aec.inference.offline_alignment.estimate_gcc_tau_windowed",
            return_value=estimate,
        ):
            _, ref_aligned, tau = align_farend_to_mic(
                mic,
                far,
                fs=100,
                max_tau_seconds=0.02,
                bias_seconds=0.004,
            )
        self.assertEqual(tau, 0.02)
        self.assertEqual(ref_aligned.size - far.size, 2)

    def test_min_pad_samples_returns_zero_effective_tau(self):
        """被 min_pad_samples 抑制的估计不得报告为已生效。"""
        mic = np.zeros(32, dtype=np.float32)
        far = np.ones(32, dtype=np.float32)
        estimate = SimpleNamespace(low_confidence=False, tau_sec=0.014)
        with patch(
            "tasks.aec.inference.offline_alignment.estimate_gcc_tau_windowed",
            return_value=estimate,
        ):
            _, ref_out, tau = align_farend_to_mic(
                mic,
                far,
                fs=100,
                min_pad_samples=1,
            )
        self.assertEqual(tau, 0.0)
        np.testing.assert_array_equal(ref_out, far)

    def test_disabled_returns_unchanged(self):
        """enabled=False returns inputs unchanged and delay 0."""
        mic, ref, _ = self._synthetic_echo()
        mic_out, ref_out, tau = align_farend_to_mic(
            mic, ref, 16000, enabled=False, pad_policy="apply"
        )
        np.testing.assert_array_equal(mic_out, mic)
        np.testing.assert_array_equal(ref_out, ref)
        self.assertEqual(tau, 0.0)

    def test_invalid_pad_policy_raises(self):
        """Unknown pad_policy raises ValueError."""
        with self.assertRaises(ValueError):
            align_farend_to_mic(np.zeros(100), np.zeros(100), 16000, pad_policy="unknown")

    def test_short_input_returns_zero(self):
        """Very short inputs return zero delay unchanged."""
        mic = np.zeros(4, dtype=np.float32)
        ref = np.zeros(4, dtype=np.float32)
        _, _, tau = align_farend_to_mic(mic, ref, 16000)
        self.assertEqual(tau, 0.0)

    def test_bias_seconds_subtracted(self):
        """bias_seconds is subtracted from the raw GCC-PHAT estimate."""
        mic, ref, _ = self._synthetic_echo()
        tau_no_bias = gcc_phat_tau_seconds(mic, ref, 16000, bias_seconds=0.0)
        tau_bias = gcc_phat_tau_seconds(mic, ref, 16000, bias_seconds=0.005)
        self.assertAlmostEqual(tau_no_bias - tau_bias, 0.005, places=5)

    def test_max_tau_clamps_estimate(self):
        """max_tau_seconds caps the search window."""
        mic, ref, true_delay_samp = self._synthetic_echo(delay_ms=150.0)
        tau = gcc_phat_tau_seconds(mic, ref, 16000, max_tau_seconds=0.05, bias_seconds=0.0)
        # With max_tau=50ms, a 150ms delay cannot be found; estimate is clamped.
        self.assertLessEqual(abs(tau), 0.05)

    def test_symmetric_zero_max_shift_searches_only_zero_lag(self):
        """双向模式的零搜索窗只能包含 lag 0，不能因 ``-0`` 展开全段。"""
        rng = np.random.default_rng(9)
        sig = rng.standard_normal(64).astype(np.float32)
        ref = np.roll(sig, 7)
        tau, _, _ = gcc_phat_tau_with_confidence(
            sig,
            ref,
            fs=16000,
            max_tau_seconds=0.0,
            non_negative=False,
        )
        self.assertEqual(tau, 0.0)

    def test_gcc_rejects_invalid_numeric_parameters_and_samples(self):
        sig = np.ones(16, dtype=np.float32)
        for fs in (0, -1, np.nan, np.inf):
            with self.subTest(fs=fs), self.assertRaises(ValueError):
                gcc_phat_tau_with_confidence(sig, sig, fs=fs)
        for kwargs in (
            {"interp": np.nan},
            {"interp": 1.5},
            {"max_tau_seconds": np.nan},
            {"max_tau_seconds": np.inf},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                gcc_phat_tau_with_confidence(sig, sig, fs=16000, **kwargs)
        invalid = sig.copy()
        invalid[0] = np.nan
        with self.assertRaises(ValueError):
            gcc_phat_tau_with_confidence(invalid, sig, fs=16000)

    def test_non_negative_constraint(self):
        """non_negative=True restricts estimate to non-negative lags."""
        rng = np.random.default_rng(123)
        ref = rng.normal(0, 1.0, 16000).astype(np.float32)
        # sig leads ref by 50 samples; physical echo cannot lead farend.
        sig = np.zeros_like(ref)
        sig[:-50] = ref[50:] * 0.3
        tau_sym = gcc_phat_tau_seconds(sig, ref, 16000, bias_seconds=0.0, non_negative=False)
        tau_nonneg = gcc_phat_tau_seconds(sig, ref, 16000, bias_seconds=0.0, non_negative=True)
        self.assertLess(tau_sym, 0.0)
        self.assertGreaterEqual(tau_nonneg, 0.0)

    def test_non_negative_preserves_positive_delay(self):
        """non_negative=True still estimates positive echo delay correctly."""
        mic, ref, true_delay_samp = self._synthetic_echo()
        tau = gcc_phat_tau_seconds(mic, ref, 16000, bias_seconds=0.0, non_negative=True)
        self.assertGreater(tau, 0.0)
        self.assertLess(abs(tau * 16000 - true_delay_samp), 10)

    @staticmethod
    def _speech_like(n: int, seed: int = 0) -> np.ndarray:
        """类语音信号：低通噪声 × 慢变包络（宽带、有调制，适合 GCC-PHAT）。"""
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n)
        x = np.convolve(x, np.ones(8) / 8.0, mode="same")
        env = 0.3 + 0.7 * np.abs(
            np.convolve(rng.standard_normal(n), np.ones(400) / 400, mode="same")
        )
        x = x * env
        return (x / (np.max(np.abs(x)) + 1e-9) * 0.3).astype(np.float32)

    def test_windowed_handles_silent_head(self):
        """前 2s 静音 + 正文 80ms 延迟：窗式估计跳过静音 head 窗，正确锁定 80ms。

        旧的 head-2s 单次估计在静音片段上会锁 0（错误）；窗式内核的能量门
        使估计落在正文窗上。
        """
        sr = 16000
        true_delay_ms = 80.0
        d = int(true_delay_ms / 1000.0 * sr)
        far = np.concatenate([np.zeros(2 * sr, np.float32), self._speech_like(4 * sr, seed=2)])
        mic = np.concatenate([np.zeros(d, np.float32), far])[: far.size]

        _, ref_aligned, tau = align_farend_to_mic(mic, far, sr, bias_seconds=0.0)
        self.assertAlmostEqual(tau * 1000.0, true_delay_ms, delta=2.0)
        self.assertEqual(ref_aligned.size - far.size, int(round(tau * sr)))

    def test_windowed_uncorrelated_pair_no_pad(self):
        """不相关白噪声对：窗式质量门全灭 → tau=0，不补零（样本保留）。"""
        sr = 16000
        rng = np.random.default_rng(21)
        far = (rng.standard_normal(4 * sr) * 0.2).astype(np.float32)
        mic = (rng.standard_normal(4 * sr) * 0.2).astype(np.float32)
        mic_out, ref_out, tau = align_farend_to_mic(mic, far, sr)
        self.assertEqual(tau, 0.0)
        np.testing.assert_array_equal(ref_out, far)
        np.testing.assert_array_equal(mic_out, mic)


def test_offline_alignment_adaptive_and_explicit_downsample_consistent():
    """22050/44100 默认 adaptive 与显式倍率都保持相同物理延迟。"""
    rng = np.random.default_rng(91)
    for sr in (22050, 44100):
        delay_ms = 47.0
        n = int(sr * 1.5)
        delay = int(round(delay_ms * sr / 1000.0))
        far = (rng.standard_normal(n) * 0.2).astype(np.float32)
        mic = np.zeros_like(far)
        mic[delay:] = far[:-delay]
        _, _, tau_auto = align_farend_to_mic(mic, far, sr)
        _, _, tau_explicit = align_farend_to_mic(mic, far, sr, downsample=4)
        assert tau_auto == pytest.approx(delay_ms / 1000.0, abs=0.002)
        assert tau_explicit == pytest.approx(delay_ms / 1000.0, abs=0.002)
        assert tau_auto == pytest.approx(tau_explicit, abs=0.002)


def test_offline_alignment_default_passes_adaptive_downsample():
    """offline 默认不强制旧的 downsample=4。"""
    mic = np.zeros(64, dtype=np.float32)
    far = np.ones(64, dtype=np.float32)
    estimate = SimpleNamespace(low_confidence=False, tau_sec=0.0)
    with patch(
        "tasks.aec.inference.offline_alignment.estimate_gcc_tau_windowed",
        return_value=estimate,
    ) as estimate_mock:
        align_farend_to_mic(mic, far, 22050)
    assert estimate_mock.call_args.kwargs["downsample"] is None


if __name__ == "__main__":
    unittest.main()

pytestmark = [pytest.mark.unit]
