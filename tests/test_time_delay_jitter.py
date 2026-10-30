"""Tests for tasks/aec/data/time_delay_jitter.py.

Covers trajectory generation, fractional retiming, echo_path reconstruction,
mask semantics, transition ramps, amplitude correlation, log-uniform
amplitudes, config validation and reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.data.timebase import apply_delay_trajectory
from tasks.aec.data.time_delay_jitter import (
    TimeDelayJitterConfig,
    _sample_event_times_ms,
    _sample_signed_log_uniform_ms,
    apply_time_delay_jitter,
)

SR = 16000
T = 1600  # 100 ms


def _impulse(t: int = T // 2, amplitude: float = 1.0) -> np.ndarray:
    x = np.zeros(T, dtype=np.float32)
    x[t] = amplitude
    return x


def _speech_like(n: int, seed: int = 0) -> np.ndarray:
    """A simple coloured noise signal that is non-zero and non-periodic."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    # gentle low-pass colouration so it resembles speech more than white noise
    x[1:] += 0.7 * x[:-1]
    return x / (np.max(np.abs(x)) + 1e-8)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _cfg(**kwargs) -> TimeDelayJitterConfig:
    defaults = dict(
        enabled=True,
        prob=1.0,
        jitter_type="sudden_jump",
        jitter_max_ms=500.0,
        # High rate + tiny min segment so the first event lands well before 50 ms.
        jitter_rate=(1000.0, 1000.0),
        jitter_corr_length=(100.0, 2000.0),
        jump_ms_range=(10.0, 10.0),  # fixed +10ms
        drift_ppm_range=(-100.0, -100.0),
        periodic_amp_ms_range=(1.0, 1.0),
        periodic_hz_range=(100.0, 100.0),
        min_segment_ms=0.5,
        transition_ms=0.0,
        # Small cap so multiple high-rate jumps do not accumulate into huge shifts
        # that push the whole signal out of bounds in unit tests.
        max_relative_jitter_ms=5.0,
    )
    defaults.update(kwargs)
    return TimeDelayJitterConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. disabled / gate
# ---------------------------------------------------------------------------
def test_disabled_returns_identity():
    mic = np.ones(T, dtype=np.float32)
    echo = np.zeros(T, dtype=np.float32)
    cfg = _cfg(enabled=False)
    r = apply_time_delay_jitter(mic, echo, 0, sample_rate=SR, rng=_rng(), config=cfg)
    np.testing.assert_array_equal(r.mic_jittered, mic)
    np.testing.assert_array_equal(r.echo_jittered, echo)
    assert not r.meta["applied"]


def test_prob_gate_miss_returns_identity():
    mic = np.ones(T, dtype=np.float32)
    echo = np.zeros(T, dtype=np.float32)
    cfg = _cfg(prob=0.0)
    r = apply_time_delay_jitter(mic, echo, 0, sample_rate=SR, rng=_rng(), config=cfg)
    np.testing.assert_array_equal(r.mic_jittered, mic)
    np.testing.assert_array_equal(r.echo_jittered, echo)
    assert not r.meta["applied"]


# ---------------------------------------------------------------------------
# 2. sudden_jump echo_path
# ---------------------------------------------------------------------------
def test_sudden_jump_positive_echo_path():
    echo = _impulse(t=T // 2)
    near = np.zeros(T, dtype=np.float32)
    mic = near + echo
    cfg = _cfg(jitter_type="sudden_jump")
    r = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(123), config=cfg,
    )
    assert r.meta["applied"]
    assert r.meta["jitter_type"] == "sudden_jump"
    # delay_gt increased after the event
    diff = np.diff(r.delay_gt, prepend=r.delay_gt[0])
    assert np.any(diff > 0)
    # echo peak moved to the right (clipped to max_relative_jitter_ms=5ms)
    orig_peak = int(np.argmax(echo))
    new_peak = int(np.argmax(r.echo_jittered))
    max_shift = int(round(5.0 / 1000.0 * SR))
    assert 0 < new_peak - orig_peak <= max_shift
    # mic is reconstructed: original mic - echo + jittered echo
    np.testing.assert_allclose(
        r.mic_jittered, mic - echo + r.echo_jittered, atol=1e-6
    )


def test_sudden_jump_negative_echo_path():
    """Negative delta advances echo, but total delay stays physically >= 0."""
    echo = _impulse(t=T // 2 + 160)  # leave headroom for skip
    near = np.zeros(T, dtype=np.float32)
    mic = near + echo
    cfg = _cfg(jitter_type="sudden_jump", jump_ms_range=(-10.0, -10.0))
    # Use base_delay=20ms so a -10ms jump keeps total delay at 10ms (>= 0).
    r = apply_time_delay_jitter(
        mic, echo, 20.0 / 1000.0 * SR,
        sample_rate=SR, rng=_rng(124), config=cfg,
    )
    new_peak = int(np.argmax(r.echo_jittered))
    orig_peak = int(np.argmax(echo))
    max_shift = int(round(5.0 / 1000.0 * SR))
    assert -max_shift <= new_peak - orig_peak < 0
    # Total delay must remain non-negative.
    assert np.all(r.delay_gt >= 0.0)


def test_total_delay_never_negative():
    """With base_delay=0, a negative jump is clipped so echo never leads ref."""
    echo = _impulse(t=T // 2)
    mic = echo.copy()
    cfg = _cfg(jitter_type="sudden_jump", jump_ms_range=(-20.0, -20.0))
    r = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(126), config=cfg,
    )
    assert np.all(r.delay_gt >= -1e-6)
    # No actual shift since the negative delta is fully clipped.
    np.testing.assert_allclose(r.echo_jittered, echo, atol=1e-6)


def test_mic_contains_near_and_noise():
    """mic may contain near+noise+echo; only echo stem is retimed."""
    near = _impulse(t=T // 2 - 80)
    noise = _impulse(t=T // 2 + 80)
    echo = _impulse(t=T // 2)
    mic = near + noise + echo
    cfg = _cfg(jitter_type="sudden_jump", jump_ms_range=(10.0, 10.0))
    r = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(125), config=cfg,
    )
    # near/noise components are untouched; only the echo component moves.
    np.testing.assert_allclose(
        r.mic_jittered - r.echo_jittered, near + noise, atol=1e-6
    )


# ---------------------------------------------------------------------------
# 3. gradual_drift
# ---------------------------------------------------------------------------
def test_gradual_drift_slope():
    # 使用非零 echo，确保 trajectory retiming 对 waveform 可观测。
    echo = _speech_like(T, seed=5)
    mic = echo.copy()
    cfg = _cfg(
        jitter_type="gradual_drift",
        drift_ppm_range=(100.0, 100.0),
    )
    r = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(), config=cfg,
    )
    # delay_gt[t] ~= t * ppm * 1e-6
    expected_slope = 100.0 * 1e-6
    actual_slope = float(r.delay_gt[-1] - r.delay_gt[0]) / (T - 1)
    assert actual_slope == pytest.approx(expected_slope, rel=1e-3)
    # 正 drift 读取更早样点，当前短轨迹仍全部落在 source window 内。
    assert r.valid_mask.all()
    # Echo must be modified by drift.
    assert not np.allclose(r.echo_jittered, echo)


def test_gradual_drift_waveform_and_mask_follow_exported_trajectory():
    echo = _speech_like(T, seed=17)
    cfg = _cfg(
        jitter_type="gradual_drift",
        drift_ppm_range=(10000.0, 10000.0),
        max_relative_jitter_ms=0.2,
    )

    result = apply_time_delay_jitter(
        None, echo, 0,
        sample_rate=SR, rng=_rng(91), config=cfg,
    )
    expected = apply_delay_trajectory(
        echo,
        total_delay_samples=result.delay_gt,
        base_delay_samples=0.0,
        edge_mode=cfg.edge_mode,
    )

    np.testing.assert_array_equal(result.echo_jittered, expected.waveform)
    np.testing.assert_array_equal(result.valid_mask, expected.valid_mask)


def test_gradual_drift_negative_ppm_valid_mask_tail_false():
    """Negative ppm compresses the timebase when baseline delay permits it."""
    echo = _speech_like(T, seed=5)
    mic = echo.copy()
    cfg = _cfg(
        jitter_type="gradual_drift",
        drift_ppm_range=(-1000.0, -1000.0),
    )
    r = apply_time_delay_jitter(
        mic, echo, 32.0,
        sample_rate=SR, rng=_rng(), config=cfg,
    )
    # Negative drift reads increasingly later source positions, so the tail
    # leaves the source window while the non-negative total delay stays valid.
    assert not r.valid_mask.all()
    assert np.all(r.echo_jittered[~r.valid_mask] == 0.0)
    assert r.valid_mask[: T // 2].all()


# ---------------------------------------------------------------------------
# 4. periodic_jitter
# ---------------------------------------------------------------------------
def test_periodic_jitter_amplitude_and_frequency():
    echo = np.zeros(T, dtype=np.float32)
    mic = echo.copy()
    cfg = _cfg(
        jitter_type="periodic_jitter",
        periodic_amp_ms_range=(2.0, 2.0),
        periodic_hz_range=(100.0, 100.0),
        periodic_max_equivalent_ppm=1e9,
    )
    r = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(), config=cfg,
    )
    # amplitude in samples
    amp_samp = 2.0 / 1000.0 * SR
    assert np.max(np.abs(r.delay_gt)) == pytest.approx(amp_samp, rel=0.05)
    # 100 Hz -> period 160 samples -> ~10 cycles in T=1600 -> ~20 sign changes
    zero_crossings = np.sum(np.diff(np.sign(r.delay_gt)) != 0)
    assert 15 < zero_crossings < 30


def test_periodic_jitter_respects_equivalent_ppm():
    """High-frequency periodic jitter is amplitude-limited by equivalent ppm."""
    echo = np.zeros(T, dtype=np.float32)
    mic = echo.copy()
    hz = 100.0
    ppm_limit = 1000.0
    max_amp_ms = ppm_limit / (2.0 * np.pi * hz * 1000.0)
    cfg = TimeDelayJitterConfig(
        enabled=True,
        prob=1.0,
        jitter_type="periodic_jitter",
        periodic_amp_ms_range=(10.0, 10.0),
        periodic_hz_range=(hz, hz),
        periodic_max_equivalent_ppm=ppm_limit,
    )
    r = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(), config=cfg,
    )
    amp_ms = np.max(np.abs(r.delay_gt)) / SR * 1000.0
    assert amp_ms <= max_amp_ms + 1e-3


# ---------------------------------------------------------------------------
# 5. mixed
# ---------------------------------------------------------------------------
def test_mixed_contains_jump_and_drift():
    echo = np.zeros(T, dtype=np.float32)
    mic = echo.copy()
    cfg = _cfg(
        jitter_type="mixed",
        drift_ppm_range=(0.0, 0.0),
        periodic_amp_ms_range=(0.0, 0.0),
        jump_ms_range=(5.0, 5.0),
    )
    r = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(7), config=cfg,
    )
    # at least one jump event
    assert r.event_mask.any()
    # delay_gt is not a simple constant
    assert np.std(r.delay_gt) > 0.1


def test_mixed_drift_uses_asrc():
    """mixed 中关闭 jump/periodic 时，drift 部分与纯 gradual_drift 一致。"""
    echo = _speech_like(T, seed=5)
    mic = echo.copy()
    ppm = 100.0
    cfg_mixed = TimeDelayJitterConfig(
        enabled=True,
        prob=1.0,
        jitter_type="mixed",
        drift_ppm_range=(ppm, ppm),
        jump_ms_range=(0.0, 0.0),
        periodic_amp_ms_range=(0.0, 0.0),
        periodic_hz_range=(1.0, 1.0),
        jitter_rate=(0.02, 0.5),
        min_segment_ms=200.0,
        max_relative_jitter_ms=500.0,
    )
    cfg_drift = TimeDelayJitterConfig(
        enabled=True,
        prob=1.0,
        jitter_type="gradual_drift",
        drift_ppm_range=(ppm, ppm),
        max_relative_jitter_ms=500.0,
    )
    r_mixed = apply_time_delay_jitter(
        mic.copy(), echo.copy(), 0,
        sample_rate=SR, rng=_rng(42), config=cfg_mixed,
    )
    r_drift = apply_time_delay_jitter(
        mic.copy(), echo.copy(), 0,
        sample_rate=SR, rng=_rng(42), config=cfg_drift,
    )
    np.testing.assert_allclose(r_mixed.delay_gt, r_drift.delay_gt, atol=1e-6)
    np.testing.assert_allclose(r_mixed.echo_jittered, r_drift.echo_jittered, atol=1e-5)


def test_mixed_with_jump():
    """mixed 同时存在 drift + jump 时，跳变事件仍然出现。"""
    echo = _speech_like(T, seed=5)
    mic = echo.copy()
    cfg = TimeDelayJitterConfig(
        enabled=True,
        prob=1.0,
        jitter_type="mixed",
        drift_ppm_range=(50.0, 50.0),
        jump_ms_range=(5.0, 5.0),
        jitter_rate=(1000.0, 1000.0),
        min_segment_ms=0.5,
        max_relative_jitter_ms=200.0,
    )
    r = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(11), config=cfg,
    )
    assert r.meta["effective_jitter_type"] == "mixed"
    assert r.event_mask.any()
    # delay_gt contains both drift slope and jump steps.
    assert np.std(np.diff(r.delay_gt)) > 1e-3


def test_mixed_all_components():
    """mixed 同时开启 drift + jump + periodic 时不产生异常，且输出 shape/valid 语义正确。"""
    echo = _speech_like(T, seed=5)
    mic = echo.copy()
    cfg = TimeDelayJitterConfig(
        enabled=True,
        prob=1.0,
        jitter_type="mixed",
        drift_ppm_range=(50.0, 50.0),
        jump_ms_range=(5.0, 5.0),
        jitter_rate=(1000.0, 1000.0),
        min_segment_ms=0.5,
        periodic_amp_ms_range=(1.0, 1.0),
        periodic_hz_range=(10.0, 10.0),
        periodic_max_equivalent_ppm=1e9,
        max_relative_jitter_ms=200.0,
    )
    r = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(12), config=cfg,
    )
    assert r.meta["effective_jitter_type"] == "mixed"
    assert r.echo_jittered.shape == (T,)
    assert r.mic_jittered.shape == (T,)
    assert np.all(np.isfinite(r.echo_jittered))
    assert np.all(np.isfinite(r.delay_gt))
    assert r.meta["num_events"] >= 1
    assert r.valid_mask.dtype == bool


# ---------------------------------------------------------------------------
# 6. transition ramps
# ---------------------------------------------------------------------------
def test_transition_ramp_smooths_step():
    """With transition_ms > 0 the delay change is a linear ramp, not a step."""
    echo = _impulse(t=T // 2)
    mic = echo.copy()
    cfg = _cfg(
        jitter_type="sudden_jump",
        jump_ms_range=(10.0, 10.0),
        transition_ms=5.0,  # 80 samples ramp
        max_relative_jitter_ms=20.0,
    )
    r = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(13), config=cfg,
    )
    diff = np.diff(r.delay_gt)
    # With a ramp, the derivative is spread over many consecutive samples.
    # With an instant step, only one sample has non-zero diff.
    non_zero_diff = np.sum(np.abs(diff) > 1e-6)
    assert non_zero_diff > 1
    assert r.meta["transition_ms"] == 5.0
    assert r.meta["transition_samples"] > 0


def test_event_mask_tracks_realized_transition_ramp_changes():
    """event_mask 仅标记最终轨迹中仍实际变化的 ramp 样点。"""
    echo = _impulse(t=T // 2)
    mic = echo.copy()
    cfg = _cfg(
        jitter_type="sudden_jump",
        jump_ms_range=(10.0, 10.0),
        transition_ms=5.0,
        max_relative_jitter_ms=20.0,
    )
    r = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(14), config=cfg,
    )
    assert r.meta.get("events")
    assert r.meta["transition_samples"] > 0
    assert r.event_mask.any()
    assert not r.event_mask[0]
    np.testing.assert_array_equal(r.event_mask[1:], np.diff(r.delay_gt) != 0.0)


# ---------------------------------------------------------------------------
# 7. amplitude correlation and log-uniform distribution
# ---------------------------------------------------------------------------
def test_correlated_amplitudes_same_direction():
    """Long correlation length makes successive jumps similar in sign/magnitude."""
    echo = np.zeros(T, dtype=np.float32)
    mic = echo.copy()
    cfg = _cfg(
        jitter_type="sudden_jump",
        jump_ms_range=(50.0, 50.0),
        jitter_rate=(500.0, 500.0),
        jitter_corr_length=(10000.0, 10000.0),  # very long -> strong correlation
        min_segment_ms=0.1,
        max_relative_jitter_ms=200.0,
    )
    r = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(20), config=cfg,
    )
    events = np.where(r.event_mask)[0]
    if len(events) >= 2:
        amps = [r.delay_gt[e] - r.delay_gt[max(0, e - 1)] for e in events]
        # All realized step amplitudes should have the same sign under strong correlation.
        assert all(a >= 0 for a in amps) or all(a <= 0 for a in amps)


def test_log_uniform_prefers_small_jumps():
    """Log-uniform sampling produces more small-magnitude jumps than uniform."""
    echo = np.zeros(T, dtype=np.float32)
    mic = echo.copy()
    cfg = _cfg(
        jitter_type="sudden_jump",
        jump_ms_range=(1.0, 100.0),
        jitter_rate=(200.0, 200.0),
        min_segment_ms=0.1,
        max_relative_jitter_ms=200.0,
    )
    small_count = 0
    total = 0
    for seed in range(30):
        r = apply_time_delay_jitter(
            mic, echo, 0,
            sample_rate=SR, rng=_rng(seed), config=cfg,
        )
        events = np.where(r.event_mask)[0]
        for e in events:
            amp_ms = abs(r.delay_gt[e] - r.delay_gt[max(0, e - 1)]) / SR * 1000.0
            total += 1
            if amp_ms < 10.0:
                small_count += 1
    # With log-uniform over [1, 100] ms, the median is ~10 ms, so at least
    # half of the realized jumps should be small.
    assert total > 0
    assert small_count / total >= 0.4


@pytest.mark.parametrize(
    ("jump_ms_range", "expected_ms"),
    [
        ((-20.0, 20.0), 4.221477182747914),
        ((-20.0, -1.0), -12.493277221965736),
        ((1.0, 20.0), 12.493277221965736),
    ],
    ids=["symmetric_zero_crossing", "nonpositive", "nonnegative"],
)
def test_jump_sampler_fixed_seed_regression(jump_ms_range, expected_ms):
    amplitude_ms = _sample_signed_log_uniform_ms(
        _rng(20260828), *jump_ms_range
    )
    assert amplitude_ms == pytest.approx(expected_ms, abs=1e-12)


def test_symmetric_jump_range_fixed_seed_keeps_cumulative_random_walk():
    """固定种子的 sudden jumps 继续按事件振幅累积，而非覆盖前值。"""
    sample_rate = 1000
    echo = np.zeros(sample_rate, dtype=np.float32)
    cfg = TimeDelayJitterConfig.from_dict({
        "enabled": True,
        "prob": 1.0,
        "jitter_type": "sudden_jump",
        "jitter_rate": (4.0, 4.0),
        "min_segment_ms": 100.0,
        "jitter_corr_length": (0.0, 0.0),
        "jump_ms_range": (-20.0, 20.0),
        "jitter_max_ms": 50.0,
        "max_relative_jitter_ms": 100.0,
    })
    result = apply_time_delay_jitter(
        None,
        echo,
        100.0,
        sample_rate=sample_rate,
        rng=_rng(20260828),
        config=cfg,
    )

    amplitudes_ms = np.asarray(
        [event["amp_ms"] for event in result.meta["events"]]
    )
    np.testing.assert_allclose(
        amplitudes_ms,
        [
            1.7810490742556566,
            0.2374527083571828,
            -0.0036557861962213008,
            -0.0017224221792928056,
            -0.016978105373809876,
        ],
        rtol=0.0,
        atol=1e-12,
    )
    assert result.delay_gt[-1] == pytest.approx(
        100.0 + float(amplitudes_ms.sum()), abs=1e-12
    )


def test_event_interval_distribution_honors_rate():
    """Truncated exponential sampling keeps the mean interval close to 1/rate."""
    rate = 1.0  # 1 event/s -> mean interval 1000 ms
    min_seg = 500.0
    duration_ms = 100_000.0
    intervals: list[float] = []
    for seed in range(100):
        events = _sample_event_times_ms(
            np.random.default_rng(seed),
            (rate, rate),
            min_seg,
            duration_ms,
        )
        prev = 0.0
        for t in events:
            intervals.append(t - prev)
            prev = t
    assert intervals
    mean_interval = float(np.mean(intervals))
    # Truncated-exponential mean should be ~1000 ms; allow 15% Monte Carlo tolerance.
    assert mean_interval == pytest.approx(1000.0, rel=0.15)


def test_event_sampling_rejects_implausible_rate():
    """误配的超高 jitter_rate 必须显式拒绝，而不是无界累积事件。"""
    with pytest.raises(ValueError, match="jump events for one sample"):
        _sample_event_times_ms(
            np.random.default_rng(0), (1e7, 1e7), 0.0, 1000.0
        )


# ---------------------------------------------------------------------------
# 8. masks and bounds
# ---------------------------------------------------------------------------
def test_valid_mask_excludes_out_of_bounds():
    echo = _impulse(t=0)
    mic = echo.copy()
    # Large positive jump moves read_pos outside bounds early.
    cfg = _cfg(
        jitter_type="sudden_jump",
        jump_ms_range=(100.0, 100.0),
        jitter_rate=(100.0, 100.0),
        min_segment_ms=1.0,
        max_relative_jitter_ms=200.0,  # allow enough accumulation to hit bounds
    )
    r = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(11), config=cfg,
    )
    assert not r.valid_mask.all()
    # invalid positions are zero-padded
    assert np.all(r.echo_jittered[~r.valid_mask] == 0.0)


def test_max_relative_jitter_clipping():
    echo = np.zeros(T, dtype=np.float32)
    mic = echo.copy()
    cfg = _cfg(
        jitter_type="gradual_drift",
        drift_ppm_range=(10000.0, 10000.0),  # would overshoot
        max_relative_jitter_ms=50.0,
    )
    r = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(), config=cfg,
    )
    max_samp = 50.0 / 1000.0 * SR
    assert np.max(np.abs(r.delay_gt)) <= max_samp + 1e-6


def test_base_delay_is_included_in_delay_gt():
    echo = np.zeros(T, dtype=np.float32)
    mic = echo.copy()
    cfg = _cfg(jitter_type="sudden_jump", jump_ms_range=(10.0, 10.0))
    base = 32.0
    r = apply_time_delay_jitter(
        mic, echo, base,
        sample_rate=SR, rng=_rng(15), config=cfg,
    )
    assert r.meta["base_delay_samples"] == base
    assert np.all(r.delay_gt >= base - 1e-3)


# ---------------------------------------------------------------------------
# 9. reproducibility
# ---------------------------------------------------------------------------
def test_same_seed_same_result():
    echo = _impulse(t=T // 2)
    mic = echo.copy()
    cfg = _cfg()
    r1 = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(42), config=cfg,
    )
    r2 = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(42), config=cfg,
    )
    np.testing.assert_array_equal(r1.delay_gt, r2.delay_gt)
    np.testing.assert_array_equal(r1.mic_jittered, r2.mic_jittered)
    np.testing.assert_array_equal(r1.echo_jittered, r2.echo_jittered)


# ---------------------------------------------------------------------------
# 10. config validation
# ---------------------------------------------------------------------------
def test_config_rejects_unknown_key():
    with pytest.raises(ValueError, match="Unknown key"):
        TimeDelayJitterConfig.from_dict({"enabled": True, "bad_key": 1})


def test_config_rejects_bad_jitter_type():
    with pytest.raises(ValueError, match="jitter_type"):
        TimeDelayJitterConfig.from_dict({"jitter_type": "random_walk"})


def test_config_rejects_prob_out_of_range():
    with pytest.raises(ValueError, match="prob"):
        TimeDelayJitterConfig.from_dict({"prob": 1.5})


def test_config_rejects_negative_transition_ms():
    with pytest.raises(ValueError, match="transition_ms"):
        TimeDelayJitterConfig.from_dict({"transition_ms": -1.0})


def test_config_allows_zero_max_relative_jitter_ms():
    cfg = TimeDelayJitterConfig.from_dict({"max_relative_jitter_ms": 0.0})
    assert cfg.max_relative_jitter_ms == 0.0


@pytest.mark.parametrize(
    "raw",
    [
        {"max_abs_delay_ms": 5.0},
        {"max_abs_delay_ms": 5.0, "max_relative_jitter_ms": 10.0},
    ],
    ids=["legacy_only", "legacy_and_canonical"],
)
def test_config_rejects_removed_legacy_max_abs_delay_ms(raw):
    with pytest.raises(ValueError) as exc_info:
        TimeDelayJitterConfig.from_dict(raw)
    message = str(exc_info.value)
    assert "max_abs_delay_ms" in message
    assert "max_relative_jitter_ms" in message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("jitter_rate", (-1.0, 0.5)),
        ("jitter_corr_length", (-1.0, 1.0)),
        ("periodic_hz_range", (0.0, 1.0)),
        ("periodic_amp_ms_range", (-1.0, 1.0)),
        ("jump_ms_range", (2.0, -1.0)),
    ],
)
def test_config_rejects_invalid_jitter_ranges(field, value):
    with pytest.raises(ValueError, match=field):
        TimeDelayJitterConfig.from_dict({field: value})


@pytest.mark.parametrize(
    "jump_ms_range",
    [(-1.0, 2.0), (-2.0, 1.0)],
    ids=["positive_side_larger", "negative_side_larger"],
)
def test_config_rejects_asymmetric_zero_crossing_jump_range(jump_ms_range):
    with pytest.raises(ValueError, match="jump_ms_range.*symmetric"):
        TimeDelayJitterConfig.from_dict({"jump_ms_range": jump_ms_range})


@pytest.mark.parametrize(
    "jump_ms_range",
    [(-2.0, 2.0), (-2.0, 0.0), (0.0, 2.0)],
    ids=["symmetric_zero_crossing", "nonpositive", "nonnegative"],
)
def test_config_allows_supported_jump_range_boundaries(jump_ms_range):
    cfg = TimeDelayJitterConfig.from_dict({"jump_ms_range": jump_ms_range})
    assert cfg.jump_ms_range == jump_ms_range


def test_direct_config_rejects_asymmetric_zero_crossing_jump_range():
    with pytest.raises(ValueError, match="jump_ms_range.*symmetric"):
        TimeDelayJitterConfig(jump_ms_range=(-1.0, 2.0))


def test_config_rejects_bad_edge_mode():
    with pytest.raises(ValueError, match="edge_mode"):
        TimeDelayJitterConfig.from_dict({"edge_mode": "mirror"})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_config_rejects_nonfinite_periodic_max_equivalent_ppm(value):
    with pytest.raises(ValueError, match="periodic_max_equivalent_ppm"):
        TimeDelayJitterConfig.from_dict({"periodic_max_equivalent_ppm": value})


def test_config_edge_mode_round_trip():
    cfg = TimeDelayJitterConfig.from_dict({"edge_mode": "nearest"})
    assert cfg.edge_mode == "nearest"


@pytest.mark.parametrize(
    "gate_config",
    [_cfg(enabled=False), _cfg(prob=0.0)],
    ids=["disabled", "prob_zero"],
)
@pytest.mark.parametrize("sample_rate", [0, -1, 1.5, True])
def test_apply_validation_rejects_invalid_sample_rate_before_fast_gate(
    sample_rate, gate_config
):
    with pytest.raises(ValueError, match="sample_rate"):
        apply_time_delay_jitter(
            None,
            np.zeros(T, dtype=np.float32),
            0.0,
            sample_rate=sample_rate,
            rng=_rng(),
            config=gate_config,
        )


@pytest.mark.parametrize(
    "gate_config",
    [_cfg(enabled=False), _cfg(prob=0.0)],
    ids=["disabled", "prob_zero"],
)
@pytest.mark.parametrize("base_delay", [-1.0, float("nan"), float("inf")])
def test_apply_validation_rejects_invalid_base_delay_before_fast_gate(
    base_delay, gate_config
):
    with pytest.raises(ValueError, match="base_delay"):
        apply_time_delay_jitter(
            None,
            np.zeros(T, dtype=np.float32),
            base_delay,
            sample_rate=SR,
            rng=_rng(),
            config=gate_config,
        )


@pytest.mark.parametrize(
    "gate_config",
    [_cfg(enabled=False), _cfg(prob=0.0)],
    ids=["disabled", "prob_zero"],
)
@pytest.mark.parametrize("max_total_delay_ms", [-1.0, float("nan"), float("inf")])
def test_apply_validation_rejects_invalid_max_total_delay_before_fast_gate(
    max_total_delay_ms, gate_config
):
    with pytest.raises(ValueError, match="max_total_delay_ms"):
        apply_time_delay_jitter(
            None,
            np.zeros(T, dtype=np.float32),
            0.0,
            sample_rate=SR,
            rng=_rng(),
            config=gate_config,
            max_total_delay_ms=max_total_delay_ms,
        )


def test_max_total_delay_ms_hard_cap():
    """When base_delay + max_relative_jitter_ms exceeds max_total_delay_ms, the total
    delay is clipped to max_total_delay_ms (the model's supported delay range).
    """
    echo = np.zeros(T, dtype=np.float32)
    mic = echo.copy()
    cfg = _cfg(
        jitter_type="sudden_jump",
        jump_ms_range=(100.0, 100.0),
        jitter_rate=(1000.0, 1000.0),
        min_segment_ms=0.1,
        max_relative_jitter_ms=300.0,
    )
    # base_delay=800ms, max_relative_jitter=300ms -> raw peak could be 1100ms,
    # but max_total_delay_ms=1000ms must cap it.
    base_ms = 800.0
    max_total_ms = 1000.0
    r = apply_time_delay_jitter(
        mic, echo, base_ms / 1000.0 * SR,
        sample_rate=SR, rng=_rng(99), config=cfg,
        max_total_delay_ms=max_total_ms,
    )
    max_total_samp = max_total_ms / 1000.0 * SR
    assert np.all(r.delay_gt <= max_total_samp + 1e-6)
    assert np.all(r.delay_gt >= 0.0)


def test_jitter_type_probs_selects_type():
    echo = np.zeros(T, dtype=np.float32)
    mic = echo.copy()
    cfg = TimeDelayJitterConfig.from_dict({
        "enabled": True,
        "prob": 1.0,
        "jitter_type_probs": {"sudden_jump": 1.0, "gradual_drift": 0.0, "periodic_jitter": 0.0},
        "jitter_rate": (1000.0, 1000.0),
        "min_segment_ms": 0.5,
        "max_relative_jitter_ms": 5.0,
    })
    r = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(200), config=cfg,
    )
    assert r.meta["effective_jitter_type"] == "sudden_jump"
    assert r.event_mask.any()


def test_jitter_type_probs_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        TimeDelayJitterConfig.from_dict({
            "jitter_type_probs": {"sudden_jump": 0.5, "gradual_drift": 0.3},
        })


def test_jitter_max_ms_clamps_event_amplitude():
    """jitter_max_ms caps the magnitude of each raw event before smoothing."""
    echo = _impulse(t=T // 2)
    mic = echo.copy()
    cfg = TimeDelayJitterConfig(
        enabled=True,
        prob=1.0,
        jitter_type="sudden_jump",
        jitter_rate=(1000.0, 1000.0),
        min_segment_ms=0.5,
        jump_ms_range=(10.0, 10.0),
        jitter_max_ms=5.0,
        max_relative_jitter_ms=200.0,
    )
    r = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(123), config=cfg,
    )
    events = r.meta.get("events", [])
    assert len(events) >= 1
    for ev in events:
        assert abs(ev["amp_ms"]) <= 5.0 + 1e-3
    # Realized step on delay_gt is also bounded by the per-event cap.
    diffs = np.diff(r.delay_gt)
    assert np.max(np.abs(diffs)) <= 5.0 / 1000.0 * SR + 1.0


def test_meta_realized_amplitude_matches_delay_gt():
    """meta 同时记录请求振幅和裁剪后实际生效振幅。"""
    echo = _impulse(t=T // 2)
    mic = echo.copy()
    cfg = _cfg(
        jitter_type="sudden_jump",
        jump_ms_range=(100.0, 100.0),
        jitter_rate=(1000.0, 1000.0),
        min_segment_ms=0.5,
        max_relative_jitter_ms=50.0,
    )
    r = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(99), config=cfg,
    )
    events = r.meta.get("events", [])
    assert events
    for ev in events:
        # realized must not exceed requested, and is clipped by max_relative_jitter_ms.
        assert abs(ev["realized_amp_ms"]) <= abs(ev["amp_ms"]) + 1e-3
        assert abs(ev["realized_amp_ms"]) <= 50.0 + 1e-3
        # With transition_ms=0, instantaneous and total are the same.
        assert abs(ev["realized_total_amp_ms"] - ev["realized_amp_ms"]) <= 1e-6
        # realized matches the actual delay_gt step at the event index.
        idx = int(round(ev["pos_ms"] / 1000.0 * SR))
        idx = max(0, min(T - 1, idx))
        if idx > 0:
            step_ms = float(r.delay_gt[idx] - r.delay_gt[idx - 1]) / SR * 1000.0
            assert step_ms == pytest.approx(ev["realized_amp_ms"], abs=1e-2)
        else:
            # Event at idx=0: realized is relative to the pre-event baseline.
            assert abs(ev["realized_amp_ms"]) > 1e-6


def test_meta_realized_total_covers_transition_ramp():
    """transition_ms > 0 时 realized_total 字段存在且被 max_relative_jitter_ms 限幅。"""
    echo = _impulse(t=T // 2)
    mic = echo.copy()
    cfg = _cfg(
        jitter_type="sudden_jump",
        jump_ms_range=(20.0, 20.0),
        jitter_rate=(1000.0, 1000.0),
        min_segment_ms=0.5,
        transition_ms=5.0,
        max_relative_jitter_ms=50.0,
    )
    r = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(21), config=cfg,
    )
    events = r.meta.get("events", [])
    assert events
    for ev in events:
        assert "realized_total_amp_ms" in ev
        assert np.isfinite(ev["realized_total_amp_ms"])
        assert abs(ev["realized_total_amp_ms"]) <= 50.0 + 1e-3


# ---------------------------------------------------------------------------
# 11. mic=None echo-only path
# ---------------------------------------------------------------------------
def test_mic_none_skips_reconstruction():
    """When mic is None, only echo is retimed; mic_jittered equals echo_jittered."""
    echo = _impulse(t=T // 2)
    cfg = _cfg(jitter_type="sudden_jump", jump_ms_range=(10.0, 10.0))
    r = apply_time_delay_jitter(
        None, echo, 0,
        sample_rate=SR, rng=_rng(31), config=cfg,
    )
    assert r.meta["applied"]
    np.testing.assert_array_equal(r.mic_jittered, r.echo_jittered)


def test_mic_none_preserves_invariant_when_mic_provided():
    """Providing the real mic preserves mic - echo + echo_jittered reconstruction."""
    near = _impulse(t=T // 2 - 80)
    echo = _impulse(t=T // 2)
    mic = near + echo
    cfg = _cfg(jitter_type="sudden_jump", jump_ms_range=(10.0, 10.0))
    r = apply_time_delay_jitter(
        mic, echo, 0,
        sample_rate=SR, rng=_rng(32), config=cfg,
    )
    np.testing.assert_allclose(
        r.mic_jittered, mic - echo + r.echo_jittered, atol=1e-6
    )


# ---------------------------------------------------------------------------
# 12. event_mask coverage for drift/periodic
# ---------------------------------------------------------------------------
def test_event_mask_marks_only_realized_drift_changes_before_clipped_plateau():
    echo = np.zeros(T, dtype=np.float32)
    cfg = _cfg(
        jitter_type="gradual_drift",
        drift_ppm_range=(10000.0, 10000.0),
        max_relative_jitter_ms=10.0,
    )
    r = apply_time_delay_jitter(
        echo.copy(), echo, 0,
        sample_rate=SR, rng=_rng(33), config=cfg,
        max_total_delay_ms=0.01,
    )
    assert r.meta["effective_jitter_type"] == "gradual_drift"
    assert not r.event_mask[0]
    assert r.event_mask[1:17].all()
    assert not r.event_mask[17:].any()


def test_event_mask_excludes_clipped_periodic_flat_regions():
    echo = np.zeros(T, dtype=np.float32)
    cfg = _cfg(
        jitter_type="periodic_jitter",
        periodic_amp_ms_range=(1.0, 1.0),
        periodic_hz_range=(10.0, 10.0),
    )
    r = apply_time_delay_jitter(
        echo.copy(), echo, 0,
        sample_rate=SR, rng=_rng(34), config=cfg,
    )
    assert r.meta["effective_jitter_type"] == "periodic_jitter"
    assert not r.event_mask[0]
    assert r.event_mask.any()
    assert (~r.event_mask).any()
    np.testing.assert_array_equal(r.event_mask[1:], np.diff(r.delay_gt) != 0.0)


def test_event_mask_is_empty_when_negative_drift_is_fully_clipped():
    echo = np.zeros(T, dtype=np.float32)
    cfg = _cfg(
        jitter_type="gradual_drift",
        drift_ppm_range=(-1000.0, -1000.0),
    )
    r = apply_time_delay_jitter(
        echo.copy(), echo, 0,
        sample_rate=SR, rng=_rng(35), config=cfg,
    )
    np.testing.assert_array_equal(r.delay_gt, np.zeros(T, dtype=np.float64))
    assert not r.event_mask.any()


# ---------------------------------------------------------------------------
# 13. edge_mode for boundary continuation
# ---------------------------------------------------------------------------
def test_edge_mode_nearest_repeats_boundary():
    """nearest edge mode repeats the last sample instead of zero-padding."""
    echo = np.ones(T, dtype=np.float32)
    cfg = _cfg(
        jitter_type="sudden_jump",
        jump_ms_range=(100.0, 100.0),
        jitter_rate=(100.0, 100.0),
        min_segment_ms=1.0,
        max_relative_jitter_ms=200.0,
    )
    r_const = apply_time_delay_jitter(
        echo.copy(), echo, 0,
        sample_rate=SR, rng=_rng(36), config=cfg,
        edge_mode="constant",
    )
    r_near = apply_time_delay_jitter(
        echo.copy(), echo, 0,
        sample_rate=SR, rng=_rng(36), config=cfg,
        edge_mode="nearest",
    )
    # Constant mode zero-pads invalid tail; nearest repeats ones, so tail energy is higher.
    invalid = ~r_const.valid_mask
    assert invalid.any()
    assert np.sum(np.abs(r_near.echo_jittered[invalid])) > np.sum(np.abs(r_const.echo_jittered[invalid]))


def test_edge_mode_rejects_invalid_value():
    echo = np.zeros(T, dtype=np.float32)
    with pytest.raises(ValueError, match="edge_mode"):
        apply_time_delay_jitter(
            echo.copy(), echo, 0,
            sample_rate=SR, rng=_rng(), config=_cfg(),
            edge_mode="mirror",
        )


# ---------------------------------------------------------------------------
# 14. AECDeploymentFrontend (streaming delay estimator) interaction evaluation
# ---------------------------------------------------------------------------
def _run_estimator_on_jittered_pair(
    mic: np.ndarray,
    farend: np.ndarray,
    cfg: StreamingDelayEstimatorConfig,
    frame_size: int = 160,
) -> tuple[list[np.ndarray], StreamingDelayEstimator]:
    """Push mic/farend through StreamingDelayEstimator frame by frame."""
    from tasks.aec.inference.streaming_delay_estimator import StreamingDelayEstimator

    est = StreamingDelayEstimator(cfg)
    aligned = []
    num_frames = len(mic) // frame_size
    for i in range(num_frames):
        est.push_ref(farend[i * frame_size:(i + 1) * frame_size])
        est.push_mic(mic[i * frame_size:(i + 1) * frame_size])
        aligned.append(est.get_aligned_ref_frame(frame_size))
    return aligned, est


def test_jitter_output_compatible_with_streaming_estimator():
    """Jittered echo + farend can be pushed through StreamingDelayEstimator.

    This verifies the data format and delay range are compatible with the
    input frontend; it does not assert estimator accuracy.
    """
    from tasks.aec.inference.streaming_delay_estimator import StreamingDelayEstimatorConfig

    rng = _rng(123)
    farend = rng.standard_normal(4 * SR).astype(np.float32)
    base_delay_samp = int(0.05 * SR)  # 50 ms
    echo = np.zeros_like(farend)
    echo[base_delay_samp:] = farend[:-base_delay_samp]
    near = rng.standard_normal(4 * SR).astype(np.float32) * 0.1

    cfg = TimeDelayJitterConfig.from_dict({
        "enabled": True,
        "prob": 1.0,
        "jitter_type": "sudden_jump",
        "jitter_rate": (0.5, 0.5),
        "min_segment_ms": 200.0,
        "jump_ms_range": (-20.0, 20.0),
        "max_relative_jitter_ms": 100.0,
    })
    r = apply_time_delay_jitter(
        near + echo, echo, float(base_delay_samp),
        sample_rate=SR, rng=rng, config=cfg,
    )
    mic = r.mic_jittered

    estimator_cfg = StreamingDelayEstimatorConfig(
        sample_rate=SR,
        buffer_duration_ms=200.0,
        coarse_delay_ms=50.0,
        max_residual_delay_ms=100.0,
        update_interval_ms=50.0,
    )
    aligned, est = _run_estimator_on_jittered_pair(mic, farend, estimator_cfg)
    assert len(aligned) == len(mic) // 160
    assert all(a.size == 160 for a in aligned)
    # Estimator should produce some delay estimate after warm-up.
    assert est.delay_ms >= 0.0


def test_streaming_estimator_tracks_sudden_jump_from_jitter():
    """A deliberate +50 ms jump created by TimeDelayJitter is tracked by AECDeploymentFrontend."""
    from tasks.aec.inference.streaming_delay_estimator import StreamingDelayEstimatorConfig

    rng = _rng(0)
    duration_s = 6
    n = duration_s * SR
    farend = rng.standard_normal(n).astype(np.float32)
    base_delay_samp = int(0.05 * SR)  # 50 ms
    echo = np.zeros_like(farend)
    echo[base_delay_samp:] = farend[:-base_delay_samp]
    near = rng.standard_normal(n).astype(np.float32) * 0.05

    # Force a single +50 ms jump in the middle of the sample.
    cfg = TimeDelayJitterConfig.from_dict({
        "enabled": True,
        "prob": 1.0,
        "jitter_type": "sudden_jump",
        "jitter_rate": (0.5, 0.5),
        "min_segment_ms": 3000.0,  # at most one event in 6 s
        "jump_ms_range": (50.0, 50.0),
        "transition_ms": 0.0,
        "max_relative_jitter_ms": 100.0,
    })
    r = apply_time_delay_jitter(
        near + echo, echo, float(base_delay_samp),
        sample_rate=SR, rng=rng, config=cfg,
    )
    # Verify exactly one jump happened and final delay is ~100 ms.
    assert r.meta["num_events"] == 1, f"num_events={r.meta['num_events']}"
    final_delay_ms = float(r.delay_gt[-1]) / SR * 1000.0
    assert 90.0 < final_delay_ms < 110.0, f"final_delay_ms={final_delay_ms}"

    estimator_cfg = StreamingDelayEstimatorConfig(
        sample_rate=SR,
        buffer_duration_ms=200.0,
        coarse_delay_ms=50.0,
        max_residual_delay_ms=100.0,
        update_interval_ms=30.0,
    )
    _, est = _run_estimator_on_jittered_pair(r.mic_jittered, farend, estimator_cfg)
    # After warm-up and the jump, the estimator should end up closer to 100 ms
    # than to the initial 50 ms.  Allow generous tolerance because of smoothing.
    assert 65.0 < est.delay_ms < 130.0, f"estimator final delay={est.delay_ms}ms"

pytestmark = [pytest.mark.unit]
