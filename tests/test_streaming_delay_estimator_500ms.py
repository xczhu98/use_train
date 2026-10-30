"""Standalone numpy-only test with speech-like excitation."""
import pytest
import importlib.util
import sys
from pathlib import Path

import numpy as np

module_path = Path(__file__).resolve().parent.parent / "tasks" / "aec" / "inference" / "streaming_delay_estimator.py"
spec = importlib.util.spec_from_file_location("streaming_delay_estimator", module_path)
estimator_mod = importlib.util.module_from_spec(spec)
sys.modules["streaming_delay_estimator"] = estimator_mod
spec.loader.exec_module(estimator_mod)

StreamingDelayEstimator = estimator_mod.StreamingDelayEstimator
StreamingDelayEstimatorConfig = estimator_mod.StreamingDelayEstimatorConfig


def _synthesize_speech_like(rng, n_samples, sr):
    """Generate a pseudo-speech signal with harmonic + formant + noise."""
    t = np.arange(n_samples) / sr
    # Fundamental with jitter
    f0 = 120.0 + 20.0 * np.sin(2 * np.pi * 2.0 * t)
    phase = np.cumsum(2 * np.pi * f0 / sr)
    voiced = np.cos(phase).astype(np.float32)
    # Formant-like resonances (vocal tract shaping)
    formants = (
        np.sin(2 * np.pi * 500 * t) * 0.6
        + np.sin(2 * np.pi * 1500 * t) * 0.4
        + np.sin(2 * np.pi * 2500 * t) * 0.2
    ).astype(np.float32)
    # Amplitude modulation ~ syllable rate
    mod = (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)) * (
        0.7 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
    )
    noise = rng.normal(0, 0.25, n_samples).astype(np.float32)
    x = ((voiced + formants) * mod + noise).astype(np.float32)
    x *= 0.5 / (np.std(x) + 1e-12)
    return x


def _sweep_delay_test(true_delay_ms, buffer_ms, residual_ms, seed=42, duration_s=8.0, sr=16000):
    rng = np.random.default_rng(seed)
    n_samples = int(duration_s * sr)
    ref = _synthesize_speech_like(rng, n_samples, sr)

    true_delay_samples = int(round(true_delay_ms * sr / 1000.0))
    echo = ref * 0.3
    if true_delay_samples > 0:
        echo = np.zeros_like(ref)
        echo[true_delay_samples:] = ref[:-true_delay_samples] * 0.3
    mic = echo + rng.normal(0, 0.05, ref.size).astype(np.float32)

    cfg = StreamingDelayEstimatorConfig(
        sample_rate=sr,
        buffer_duration_ms=buffer_ms,
        coarse_delay_ms=0.0,
        max_residual_delay_ms=residual_ms,
        update_interval_ms=100.0,
        smooth_alpha=0.15,
        ref_energy_threshold_db=-50.0,
        min_observation_ms=100.0,
        peak_threshold=0.05,
        peak_prominence_db=6.0,
    )
    est = StreamingDelayEstimator(cfg)

    hop = 160
    for i in range(0, n_samples - hop + 1, hop):
        # Update is triggered after the reference frame is pushed so that mic
        # and ref windows are time-aligned.
        est.push_ref(ref[i : i + hop])
        est.push_mic(mic[i : i + hop])
        _ = est.get_aligned_ref_frame(hop)

    est_ms = est.delay_ms
    error_ms = abs(est_ms - true_delay_ms)
    return est_ms, error_ms


def test_500ms_delay_sweep():
    """Pytest entry point: sweep delays up to 400 ms with production-like config."""
    delays = [0, 25, 50, 87.5, 100, 150, 200, 250, 300, 400]
    configs = [
        # Production config mirrors
        # dbgnet_small_16k_10ms_delay80ms_loss_hybrid.yaml;
        # this wider 500 ms test buffer additionally covers the 400 ms sweep,
        # while residual_ms keeps the true delay away from the search boundary.
        ("500/500", 500.0, 500.0),
        # Larger buffer sanity check.
        ("800/500", 800.0, 500.0),
    ]
    for name, buffer_ms, residual_ms in configs:
        for d in delays:
            est, err = _sweep_delay_test(d, buffer_ms, residual_ms, seed=int(d) + 1)
            # Allow one STFT hop (10 ms) of quantization for non-zero delays and
            # a slightly wider band for the zero-delay case, where the synthetic
            # periodic signal can bias the first accepted peak by a few hops.
            tol = 35.0 if d == 0 else 15.0
            assert err <= tol, (
                f"config {name}, true_delay={d}ms: estimate={est:.2f}ms, "
                f"error={err:.2f}ms > tol={tol:.2f}ms"
            )


pytestmark = [pytest.mark.unit]
