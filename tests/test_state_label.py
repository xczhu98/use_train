"""Unit tests for tasks/aec/data/state_label.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tasks.aec.data.state_label import (
    STATE_LABEL_DT,
    STATE_LABEL_FST,
    STATE_LABEL_NST,
    _smooth_state_sequence,
    _state_frame_activity_mask,
    compute_aec_state_attributes,
    compute_aec_state_label,
    compute_aec_state_scenario_label,
    compute_state_activity_diagnostics,
)


SR = 16000
HOP = 256
WIN = 512


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def test_all_zero_signal_returns_inactive():
    x = np.zeros(SR, dtype=np.float32)
    mask = _state_frame_activity_mask(x, HOP, WIN)
    assert mask.sum() == 0
    assert len(mask) == SR // HOP + 1


def test_activity_diagnostics_identify_peak_normalization_short_circuit():
    diagnostics = compute_state_activity_diagnostics(
        np.zeros(SR, dtype=np.float32), HOP, WIN
    )

    assert diagnostics["normalization_applied"] is False
    assert diagnostics["pre_normalization_peak"] == 0.0
    assert diagnostics["pre_normalization_rms"] == 0.0
    assert diagnostics["floor_hit_frames"] == 0
    assert diagnostics["floor_hit_rate"] == 0.0


def test_full_scale_sine_all_active():
    t = np.arange(SR)
    x = np.sin(2 * np.pi * 440 * t / SR).astype(np.float32)
    mask = _state_frame_activity_mask(x, HOP, WIN)
    assert mask.all()


def test_half_silence_half_sine():
    """A signal that is silent for the first half and active for the second half
    should have inactive frames in the first half and active frames in the second.
    """
    half = SR // 2
    x = np.concatenate([
        np.zeros(half, dtype=np.float32),
        np.sin(2 * np.pi * 440 * np.arange(half) / SR).astype(np.float32),
    ])
    mask = _state_frame_activity_mask(x, HOP, WIN)
    n_frames = len(mask)
    # With centered windows, the transition region spans a few frames; require
    # that early frames are mostly inactive and late frames are mostly active.
    assert mask[: n_frames // 3].sum() == 0
    assert mask[2 * n_frames // 3 :].all()


def test_peak_normalization_makes_threshold_independent_of_scale():
    """A sine wave at two different amplitudes should produce identical masks."""
    t = np.arange(SR)
    sine = np.sin(2 * np.pi * 440 * t / SR).astype(np.float32)
    mask1 = _state_frame_activity_mask(sine * 0.1, HOP, WIN)
    mask2 = _state_frame_activity_mask(sine * 0.9, HOP, WIN)
    assert np.array_equal(mask1, mask2)


def test_activity_diagnostics_expose_pre_normalization_levels_without_changing_mask():
    t = np.arange(SR)
    sine = np.sin(2 * np.pi * 440 * t / SR).astype(np.float32)
    quiet = sine * 0.1
    loud = sine * 0.9

    quiet_diag = compute_state_activity_diagnostics(
        quiet, HOP, WIN, pre_emphasis=0.0
    )
    loud_diag = compute_state_activity_diagnostics(
        loud, HOP, WIN, pre_emphasis=0.0
    )

    assert quiet_diag["pre_normalization_peak"] == pytest.approx(0.1, rel=1e-3)
    assert loud_diag["pre_normalization_peak"] == pytest.approx(0.9, rel=1e-3)
    assert loud_diag["pre_normalization_rms"] == pytest.approx(
        9.0 * quiet_diag["pre_normalization_rms"], rel=1e-5
    )
    assert quiet_diag["floor_hit_rate"] == loud_diag["floor_hit_rate"]
    assert np.array_equal(
        _state_frame_activity_mask(quiet, HOP, WIN),
        _state_frame_activity_mask(loud, HOP, WIN),
    )


def test_activity_diagnostics_report_normalized_floor_suppression_rate():
    x = np.zeros(4 * HOP, dtype=np.float32)
    x[:HOP] = 1.0
    x[HOP : 2 * HOP] = 0.1
    relative_only = _state_frame_activity_mask(
        x,
        HOP,
        HOP,
        threshold_db=-80.0,
        hangover_frames=0,
        absolute_floor_after_norm=0.0,
        pre_emphasis=0.0,
    )
    with_floor = _state_frame_activity_mask(
        x,
        HOP,
        HOP,
        threshold_db=-80.0,
        hangover_frames=0,
        absolute_floor_after_norm=0.2,
        pre_emphasis=0.0,
    )
    diagnostics = compute_state_activity_diagnostics(
        x,
        HOP,
        HOP,
        threshold_db=-80.0,
        absolute_floor_after_norm=0.2,
        pre_emphasis=0.0,
    )

    suppressed_frames = int((relative_only & ~with_floor).sum())
    assert diagnostics["floor_controls_threshold"] is True
    assert diagnostics["floor_suppressed_frames"] == suppressed_frames
    assert diagnostics["floor_suppression_rate"] == pytest.approx(
        suppressed_frames / len(relative_only)
    )


def test_boundary_frames_use_valid_sample_count():
    """A signal active only at the very beginning should still mark the first
    centered frame active (it contains half padding but enough valid samples).
    """
    x = np.zeros(SR, dtype=np.float32)
    x[: WIN] = np.sin(2 * np.pi * 440 * np.arange(WIN) / SR).astype(np.float32)
    mask = _state_frame_activity_mask(x, HOP, WIN)
    # First frame has half padding but should still see strong energy.
    assert mask[0]


def test_normalized_floor_prevents_sparse_noise_boost():
    """A single spike must not make zero frames pass the normalized floor."""
    x = np.zeros(SR, dtype=np.float32)
    x[SR // 2] = 1e-7  # single spike determines peak
    mask = _state_frame_activity_mask(x, HOP, WIN, absolute_floor_after_norm=1e-4)
    # Only the frame containing the spike (and possibly its hangover neighbour)
    # should be active; all-zero frames must be inactive.
    assert mask.sum() <= 3


def test_hangover_fills_short_gaps():
    """A single-frame gap between active frames should be filled."""
    n_frames = 10
    # Construct a signal with active frames except frame 5.
    x = np.zeros(n_frames * HOP, dtype=np.float32)
    for i in range(n_frames):
        if i != 5:
            start = i * HOP
            x[start : start + HOP] = np.sin(2 * np.pi * 440 * np.arange(HOP) / SR).astype(np.float32)
    mask = _state_frame_activity_mask(x, HOP, HOP, hangover_frames=1)
    assert mask[5]


def test_compute_aec_state_label_basic_states():
    """Manually construct nearend/echo segments and verify state mapping."""
    n_samples = 10 * HOP
    nearend = np.zeros(n_samples, dtype=np.float32)
    echo = np.zeros(n_samples, dtype=np.float32)

    # Segment 0: only echo active -> FST
    echo[0 * HOP : 2 * HOP] = 1.0
    # Segment 1: only nearend active -> NST
    nearend[3 * HOP : 5 * HOP] = 1.0
    # Segment 2: both active -> DT
    echo[6 * HOP : 8 * HOP] = 1.0
    nearend[6 * HOP : 8 * HOP] = 1.0

    state = compute_aec_state_label(nearend, echo, HOP, WIN)
    assert state.shape[0] == n_samples // HOP + 1

    # Use a central frame from each segment to avoid transition effects.
    assert state[1] == STATE_LABEL_FST
    assert state[4] == STATE_LABEL_NST
    assert state[7] == STATE_LABEL_DT


def test_3class_collapses_silence_and_noise_to_nst():
    """3-class FST/DT/NST is a coarse collapse of the 8-class scenario label.

    Frames without both near-end and echo activity all map to NST, including
    silence and noise-only frames.  This test documents the intentional
    inconsistency between 8-class and 3-class silence definitions.
    """
    n = 3 * HOP
    nearend = np.zeros(n, dtype=np.float32)
    echo = np.zeros(n, dtype=np.float32)
    noise = np.zeros(n, dtype=np.float32)

    # silence -> 8-class 0, 3-class NST
    label_8 = compute_aec_state_scenario_label(nearend, echo, noise, HOP, WIN)
    label_3 = compute_aec_state_label(nearend, echo, HOP, WIN)
    assert (label_8 == 0).all()
    assert (label_3 == STATE_LABEL_NST).all()

    # noise only -> 8-class 1, 3-class NST
    noise[HOP : 2 * HOP] = 1.0
    label_8 = compute_aec_state_scenario_label(nearend, echo, noise, HOP, WIN)
    label_3 = compute_aec_state_label(nearend, echo, HOP, WIN)
    assert label_8[1] == 1
    assert label_3[1] == STATE_LABEL_NST

    # near only -> 8-class 4, 3-class NST
    noise[:] = 0.0
    nearend[HOP : 2 * HOP] = 1.0
    label_8 = compute_aec_state_scenario_label(nearend, echo, noise, HOP, WIN)
    label_3 = compute_aec_state_label(nearend, echo, HOP, WIN)
    assert label_8[1] == 4
    assert label_3[1] == STATE_LABEL_NST


def test_compute_aec_state_label_echo_gets_more_lenient_threshold():
    """Echo 20 dB below nearend should still be detected as active."""
    n_samples = 5 * HOP
    nearend = np.zeros(n_samples, dtype=np.float32)
    echo = np.zeros(n_samples, dtype=np.float32)
    nearend[1 * HOP : 4 * HOP] = 1.0
    echo[1 * HOP : 4 * HOP] = 0.1  # -20 dB relative to nearend

    state = compute_aec_state_label(nearend, echo, HOP, WIN)
    # The overlapping region should be DT because echo is well above -50 dB
    # relative to its own peak (0.1 -> threshold 0.1 * 10^(-50/20) ≈ 3e-4).
    central = state[2 * HOP : 3 * HOP]
    assert (central == STATE_LABEL_DT).all()


def test_state_label_exported_to_torch():
    """Ensure state labels can be converted to torch tensors for collation."""
    state = compute_aec_state_label(
        np.zeros(2 * HOP, dtype=np.float32),
        np.zeros(2 * HOP, dtype=np.float32),
        HOP,
        WIN,
    )
    t = torch.from_numpy(state.copy())
    assert t.dtype == torch.int64
    assert t.shape[0] == 2 * HOP // HOP + 1


def test_pre_emphasis_reduces_low_frequency_bias():
    """A loud low-frequency bias should not mark silence frames as active.

    Without pre-emphasis the near-DC drift dominates the peak and the silent
    portions can exceed a relative threshold.  With pre-emphasis enabled the
    drift is attenuated and only the actual high-energy sine segment is active.
    """
    n = 5 * HOP
    t = np.arange(n)
    # Strong DC offset + a short high-frequency sine burst.
    x = np.ones(n, dtype=np.float32) * 0.9
    x[1 * HOP : 4 * HOP] += np.sin(2 * np.pi * 1000 * t[1 * HOP : 4 * HOP] / SR)

    mask_no_pre = _state_frame_activity_mask(x, HOP, WIN, pre_emphasis=0.0)
    mask_pre = _state_frame_activity_mask(x, HOP, WIN, pre_emphasis=0.97)
    # Pre-emphasis must not increase activity and should suppress pure-drift frames.
    assert mask_pre.sum() <= mask_no_pre.sum()
    # The active sine region (frames 1..3) should still be detected.
    assert mask_pre[1:4].any()


def test_compute_aec_state_attributes_all_zero():
    x = np.zeros(2 * HOP, dtype=np.float32)
    attrs = compute_aec_state_attributes(x, x, x, HOP, WIN)
    assert attrs.shape == (2 * HOP // HOP + 1, 3)
    assert attrs.sum() == 0


def test_compute_aec_state_attributes_single_components():
    n = 5 * HOP
    nearend = np.zeros(n, dtype=np.float32)
    echo = np.zeros(n, dtype=np.float32)
    noise = np.zeros(n, dtype=np.float32)
    nearend[1 * HOP : 4 * HOP] = 1.0
    echo[2 * HOP : 5 * HOP] = 1.0
    noise[0 * HOP : 2 * HOP] = 1.0

    attrs = compute_aec_state_attributes(nearend, echo, noise, HOP, WIN)
    # Use central frames to avoid transition boundaries.
    assert attrs[2, 0] == 1.0  # near
    assert attrs[3, 1] == 1.0  # echo
    assert attrs[1, 2] == 1.0  # noise


def test_compute_aec_state_attributes_noise_optional():
    n = 3 * HOP
    nearend = np.zeros(n, dtype=np.float32)
    nearend[1 * HOP : 2 * HOP] = 1.0
    attrs = compute_aec_state_attributes(nearend, np.zeros(n), None, HOP, WIN)
    assert attrs[:, 0].any()
    assert not attrs[:, 1].any()
    assert not attrs[:, 2].any()


def test_compute_aec_state_attributes_seven_scenario_mapping():
    """Verify that the 7 AEC branches produce the expected attribute patterns."""
    n = 6 * HOP
    speech = np.zeros(n, dtype=np.float32)
    speech[1 * HOP : 4 * HOP] = 1.0
    echo = np.zeros(n, dtype=np.float32)
    echo[2 * HOP : 5 * HOP] = 1.0
    noise = np.zeros(n, dtype=np.float32)
    noise[0 * HOP : 3 * HOP] = 1.0

    # prob_target_noise_echo: all three can be active.
    attrs = compute_aec_state_attributes(speech, echo, noise, HOP, WIN)
    central = attrs[3]
    assert central[0] == 1.0
    assert central[1] == 1.0
    assert central[2] == 1.0

    # prob_noise_only: only noise active.
    attrs = compute_aec_state_attributes(np.zeros(n), np.zeros(n), noise, HOP, WIN)
    assert attrs[:, 2].any()
    assert not attrs[:, 0].any()
    assert not attrs[:, 1].any()

    # echo_only: only echo active.
    attrs = compute_aec_state_attributes(np.zeros(n), echo, np.zeros(n), HOP, WIN)
    assert attrs[:, 1].any()
    assert not attrs[:, 0].any()
    assert not attrs[:, 2].any()


def test_state_attributes_exported_to_torch():
    """Ensure state attributes can be converted to torch tensors for collation."""
    attrs = compute_aec_state_attributes(
        np.zeros(2 * HOP, dtype=np.float32),
        np.zeros(2 * HOP, dtype=np.float32),
        np.zeros(2 * HOP, dtype=np.float32),
        HOP,
        WIN,
    )
    t = torch.from_numpy(attrs.copy())
    assert t.dtype == torch.float32
    assert t.shape == (2 * HOP // HOP + 1, 3)


def test_compute_aec_state_scenario_label_mapping():
    """Verify the 8-class scenario mapping from presence attributes."""
    n = 3 * HOP
    nearend = np.zeros(n, dtype=np.float32)
    echo = np.zeros(n, dtype=np.float32)
    noise = np.zeros(n, dtype=np.float32)

    # silence -> class 0
    label = compute_aec_state_scenario_label(nearend, echo, noise, HOP, WIN)
    assert label.shape == (n // HOP + 1,)
    assert (label == 0).all()

    # noise only -> class 1
    noise[HOP : 2 * HOP] = 1.0
    label = compute_aec_state_scenario_label(nearend, echo, noise, HOP, WIN)
    assert label[1] == 1

    # echo only -> class 2 (FST)
    noise[:] = 0.0
    echo[HOP : 2 * HOP] = 1.0
    label = compute_aec_state_scenario_label(nearend, echo, noise, HOP, WIN)
    assert label[1] == 2

    # near only -> class 4 (NST / prob_target_only active frame)
    echo[:] = 0.0
    nearend[HOP : 2 * HOP] = 1.0
    label = compute_aec_state_scenario_label(nearend, echo, noise, HOP, WIN)
    assert label[1] == 4

    # near + echo -> class 6 (DT)
    echo[HOP : 2 * HOP] = 1.0
    label = compute_aec_state_scenario_label(nearend, echo, noise, HOP, WIN)
    assert label[1] == 6

    # all three -> class 7
    noise[HOP : 2 * HOP] = 1.0
    label = compute_aec_state_scenario_label(nearend, echo, noise, HOP, WIN)
    assert label[1] == 7


def test_compute_aec_state_scenario_label_integer_dtype():
    """Ensure scenario labels are int64 for collation."""
    label = compute_aec_state_scenario_label(
        np.zeros(2 * HOP, dtype=np.float32),
        np.zeros(2 * HOP, dtype=np.float32),
        np.zeros(2 * HOP, dtype=np.float32),
        HOP,
        WIN,
    )
    t = torch.from_numpy(label.copy())
    assert t.dtype == torch.int64


def test_smooth_state_sequence_identity_for_window_one():
    seq = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    assert np.array_equal(_smooth_state_sequence(seq, 1), seq)


def test_smooth_state_sequence_removes_isolated_frame():
    # A single-frame DT burst surrounded by FST should be smoothed to FST.
    seq = np.array(
        [STATE_LABEL_FST] * 4
        + [STATE_LABEL_DT]
        + [STATE_LABEL_FST] * 4,
        dtype=np.int64,
    )
    smoothed = _smooth_state_sequence(seq, 3)
    assert (smoothed == STATE_LABEL_FST).all()


def test_smooth_state_sequence_preserves_real_segment():
    # A contiguous DT segment of length 3 should survive a 3-frame mode filter.
    seq = np.array(
        [STATE_LABEL_FST] * 3
        + [STATE_LABEL_DT] * 3
        + [STATE_LABEL_FST] * 3,
        dtype=np.int64,
    )
    smoothed = _smooth_state_sequence(seq, 3)
    assert (smoothed[3:6] == STATE_LABEL_DT).all()


def test_compute_aec_state_scenario_label_smoothing_reduces_flicker():
    n = 10 * HOP
    nearend = np.zeros(n, dtype=np.float32)
    echo = np.zeros(n, dtype=np.float32)
    # Long echo-only segment.
    echo[1 * HOP : 9 * HOP] = 1.0
    # Short nearend spike in the middle creates a brief DT flicker.
    nearend[5 * HOP] = 5.0

    no_smooth = compute_aec_state_scenario_label(
        nearend, echo, None, HOP, WIN, smoothing_window_frames=1
    )
    # Use a larger smoothing window so the short spike (<=2 frames) is absorbed.
    smooth = compute_aec_state_scenario_label(
        nearend, echo, None, HOP, WIN, smoothing_window_frames=5
    )

    # Without smoothing there is a brief near+echo (DT, class 6) flicker.
    assert 6 in no_smooth
    # With smoothing the spike is absorbed back to echo_only.
    assert 6 not in smooth
    assert set(np.unique(smooth)) == {0, 2}
    assert (smooth[1:9] == 2).all()


def test_compute_aec_state_label_smoothing_reduces_flicker():
    n = 10 * HOP
    nearend = np.zeros(n, dtype=np.float32)
    echo = np.zeros(n, dtype=np.float32)
    echo[1 * HOP : 9 * HOP] = 1.0
    nearend[5 * HOP] = 5.0

    no_smooth = compute_aec_state_label(
        nearend, echo, HOP, WIN, smoothing_window_frames=1
    )
    smooth = compute_aec_state_label(
        nearend, echo, HOP, WIN, smoothing_window_frames=5
    )

    # FST -> DT flicker without smoothing.
    assert STATE_LABEL_DT in no_smooth
    # Smoothed result keeps the dominant FST state.
    assert STATE_LABEL_DT not in smooth
    assert (smooth[1:9] == STATE_LABEL_FST).all()

pytestmark = [pytest.mark.unit]
