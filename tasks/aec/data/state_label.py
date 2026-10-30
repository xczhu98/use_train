"""Frame-level state label helpers for AEC state detectors.

Provides both the legacy mutually-exclusive FST/DT/NST labels and the newer
factorized multi-label presence attributes ``[near_present, echo_present,
noise_present]``.  Labels are derived from signal energy on each STFT frame,
not from a general voice-activity detector.
"""

from __future__ import annotations

import numbers

import numpy as np


STATE_LABEL_FST = 0  # far-end single talk (echo only)
STATE_LABEL_DT = 1   # double talk (near + echo)
# NST in the 3-class view is the coarse collapse of all "no echo" frames in the
# 8-class scenario view: near_only (4) + prob_noise_only (1) + silence (0).
STATE_LABEL_NST = 2  # near-end single talk / no echo / silence / noise-only

# Human-readable names for frame-level state classes, used for logging.
FST_DT_NST_CLASS_NAMES = ("FST", "DT", "NST")
SCENARIO_CLASS_NAMES = (
    "silence",
    "prob_noise_only",
    "echo_only",
    "noise_echo",
    "near_only",
    "near_noise",
    "near_echo",
    "all",
)


def _validate_state_label_inputs(
    signal: np.ndarray,
    hop_length: int,
    win_length: int | None,
    *,
    threshold_db: float | None = None,
    hangover_frames: int | None = None,
    absolute_floor_after_norm: float | None = None,
    pre_emphasis: float | None = None,
    smoothing_window_frames: int | None = None,
) -> np.ndarray:
    """校验 state label 公共 API 的帧参数和 waveform 数值域。"""
    value = np.asarray(signal)
    if value.ndim != 1:
        raise ValueError(f"state signal must be 1-D, got {value.shape}")
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError("state signal must contain only finite numeric values")
    for name, size in (("hop_length", hop_length), ("win_length", win_length)):
        if size is not None and (isinstance(size, bool) or not isinstance(size, numbers.Integral) or size <= 0):
            raise ValueError(f"{name} must be a positive integer, got {size!r}")
    for name, value_to_check, minimum in (
        ("threshold_db", threshold_db, None),
        ("absolute_floor_after_norm", absolute_floor_after_norm, 0.0),
        ("pre_emphasis", pre_emphasis, None),
    ):
        if value_to_check is not None and (
            not isinstance(value_to_check, numbers.Real)
            or not np.isfinite(value_to_check)
            or (minimum is not None and value_to_check < minimum)
        ):
            raise ValueError(f"{name} must be finite{' and >= 0' if minimum is not None else ''}, got {value_to_check!r}")
    for name, count in (("hangover_frames", hangover_frames), ("smoothing_window_frames", smoothing_window_frames)):
        if count is not None and (isinstance(count, bool) or not isinstance(count, numbers.Integral) or count < 0):
            raise ValueError(f"{name} must be a non-negative integer, got {count!r}")
    return value.astype(np.float32, copy=False)


def _smooth_state_sequence(state: np.ndarray, window_frames: int = 1) -> np.ndarray:
    """Apply a sliding-window mode filter to a categorical state sequence.

    This suppresses isolated 1-2 frame state flicker caused by hard thresholding
    in ``_state_frame_activity_mask``.  Ties are resolved by keeping the original
    center frame class when it is among the modes, otherwise choosing the smallest
    class id.

    Args:
        state: Integer state label sequence, shape ``(T,)``.
        window_frames: Window length in frames.  Values <= 1 disable smoothing;
            even values are rounded up to the next odd integer.

    Returns:
        Smoothed integer state sequence of the same shape.
    """
    if window_frames <= 1:
        return state
    # Force odd window for symmetric padding around the center frame.
    if window_frames % 2 == 0:
        window_frames += 1
    half = window_frames // 2
    n = state.shape[0]
    if n == 0:
        return state.copy()

    padded = np.pad(state, (half, half), mode="edge")
    out = state.copy()
    max_class = int(state.max())
    for i in range(n):
        win = padded[i : i + window_frames]
        counts = np.bincount(win, minlength=max_class + 1)
        max_count = int(counts.max())
        modes = np.where(counts == max_count)[0]
        center = int(state[i])
        if center in modes:
            out[i] = center
        else:
            out[i] = int(modes[0])
    return out


def compute_state_activity_diagnostics(
    signal: np.ndarray,
    hop_length: int,
    win_length: int | None = None,
    threshold_db: float = -40.0,
    absolute_floor_after_norm: float = 1e-4,
    pre_emphasis: float = 0.97,
) -> dict[str, float | int | bool]:
    """Summarize the levels and normalized-floor effect of state activity detection.

    This pure diagnostic helper uses the same pre-emphasis, peak normalization,
    and centered frame-RMS definitions as ``_state_frame_activity_mask``.  It
    does not alter label generation or its return ABI.  ``pre_normalization_*``
    describes the detector input after pre-emphasis but before peak
    normalization; it is therefore the appropriate place to audit absolute
    waveform level.  Floor statistics are measured in the peak-normalized
    domain before hangover filling.

    Returns:
        JSON-compatible scalar metadata.  ``floor_hit_rate`` is the fraction of
        frames at or below ``absolute_floor_after_norm``;
        ``floor_suppression_rate`` is the fraction that would pass the relative
        dB threshold but is rejected by the normalized-domain floor.
    """
    signal = _validate_state_label_inputs(
        signal,
        hop_length,
        win_length,
        threshold_db=threshold_db,
        absolute_floor_after_norm=absolute_floor_after_norm,
        pre_emphasis=pre_emphasis,
    )
    n_samples = len(signal)
    n_frames = n_samples // hop_length + 1 if n_samples > 0 else 0

    # 与标签路径一致：先做预加重，再审计峰值和 RMS。
    if pre_emphasis > 0.0 and n_samples > 1:
        emphasized = np.empty_like(signal)
        emphasized[0] = signal[0]
        emphasized[1:] = signal[1:] - pre_emphasis * signal[:-1]
        signal = emphasized

    peak = float(np.max(np.abs(signal))) if n_samples > 0 else 0.0
    pre_norm_rms = (
        float(np.sqrt(np.mean(signal * signal))) if n_samples > 0 else 0.0
    )
    relative_threshold = float(10.0 ** (threshold_db / 20.0))
    normalized_floor = float(absolute_floor_after_norm)
    effective_threshold = max(relative_threshold, normalized_floor)

    # 全零/极小信号与标签路径一样不执行峰值归一化，floor 统计也不伪造。
    if n_samples == 0 or peak < 1e-8:
        return {
            "n_samples": n_samples,
            "n_frames": n_frames,
            "pre_normalization_peak": peak,
            "pre_normalization_rms": pre_norm_rms,
            "normalization_applied": False,
            "relative_threshold_linear": relative_threshold,
            "floor_after_norm": normalized_floor,
            "effective_threshold_linear": effective_threshold,
            "floor_controls_threshold": normalized_floor > relative_threshold,
            "floor_hit_frames": 0,
            "floor_hit_rate": 0.0,
            "floor_suppressed_frames": 0,
            "floor_suppression_rate": 0.0,
        }

    normalized = signal / peak
    resolved_win_length = win_length if win_length is not None else hop_length
    half_win = resolved_win_length // 2
    padded = np.pad(normalized, (half_win, half_win), mode="constant")
    frame_rms = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_length
        seg = padded[start : start + resolved_win_length]
        valid_start = max(0, half_win - start)
        valid_end = min(resolved_win_length, half_win + n_samples - start)
        valid_seg = seg[valid_start:valid_end]
        frame_rms[i] = float(np.sqrt(np.mean(valid_seg * valid_seg)))

    floor_hits = frame_rms <= normalized_floor
    floor_suppressed = (frame_rms > relative_threshold) & (
        frame_rms <= effective_threshold
    )
    floor_hit_frames = int(floor_hits.sum())
    floor_suppressed_frames = int(floor_suppressed.sum())
    return {
        "n_samples": n_samples,
        "n_frames": n_frames,
        "pre_normalization_peak": peak,
        "pre_normalization_rms": pre_norm_rms,
        "normalization_applied": True,
        "relative_threshold_linear": relative_threshold,
        "floor_after_norm": normalized_floor,
        "effective_threshold_linear": effective_threshold,
        "floor_controls_threshold": normalized_floor > relative_threshold,
        "floor_hit_frames": floor_hit_frames,
        "floor_hit_rate": float(floor_hit_frames / n_frames),
        "floor_suppressed_frames": floor_suppressed_frames,
        "floor_suppression_rate": float(floor_suppressed_frames / n_frames),
    }


def _state_frame_activity_mask(
    signal: np.ndarray,
    hop_length: int,
    win_length: int | None = None,
    threshold_db: float = -40.0,
    hangover_frames: int = 1,
    absolute_floor_after_norm: float = 1e-4,
    pre_emphasis: float = 0.97,
) -> np.ndarray:
    """Return a per-STFT-frame activity mask for AEC state labelling.

    This is not a general voice activity detector; it simply decides whether a
    signal (near-end target or echo) has enough energy on a given frame to be
    considered "present" when constructing FST/DT/NST labels.

    A light pre-emphasis high-pass filter is applied first to reduce low-
    frequency bias (e.g. DC drift, rumble).  The signal is then peak-normalized
    so that the threshold is applied to relative presence rather than absolute
    level.  The energy window matches a centered STFT frame (``win_length``
    samples centred at ``i * hop_length``, zero-padded at the boundaries).
    Boundary frames are normalized by the number of valid (non-padded) samples
    so that padding does not attenuate the estimate.  A frame is active if its
    RMS is within ``threshold_db`` of the normalized peak, subject to a second
    floor in the same peak-normalized domain.  This floor is not an absolute
    waveform-level or dBFS gate.  A short hangover suppresses spurious 1-frame
    gaps.
    """
    signal = _validate_state_label_inputs(
        signal,
        hop_length,
        win_length,
        threshold_db=threshold_db,
        hangover_frames=hangover_frames,
        absolute_floor_after_norm=absolute_floor_after_norm,
        pre_emphasis=pre_emphasis,
    )
    n_samples = len(signal)
    if n_samples == 0:
        return np.zeros(0, dtype=np.bool_)

    # Light pre-emphasis high-pass: y[n] = x[n] - alpha * x[n-1].
    # Reduces low-frequency bias while keeping the signal length unchanged.
    if pre_emphasis > 0.0 and n_samples > 1:
        emphasized = np.empty_like(signal)
        emphasized[0] = signal[0]
        emphasized[1:] = signal[1:] - pre_emphasis * signal[:-1]
        signal = emphasized

    # Peak-normalize so the mask reflects temporal presence, not absolute level.
    # This makes the state labels independent of SER / volume perturbations.
    peak = float(np.max(np.abs(signal)))
    if peak < 1e-8:
        return np.zeros(n_samples // hop_length + 1, dtype=np.bool_)
    signal = signal / peak

    n_frames = n_samples // hop_length + 1
    win_length = win_length if win_length is not None else hop_length
    half_win = win_length // 2
    # Match centered-STFT boundary padding.
    padded = np.pad(signal, (half_win, half_win), mode="constant")

    # Precompute which padded positions correspond to original signal samples.
    # Original samples occupy [half_win, half_win + n_samples).
    rms = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_length
        seg = padded[start : start + win_length]
        # Count valid (non-padded) samples in this window.
        valid_start = max(0, half_win - start)
        valid_end = min(win_length, half_win + n_samples - start)
        valid_count = max(1, valid_end - valid_start)
        valid_seg = seg[valid_start:valid_end]
        rms[i] = float(np.sqrt(np.mean(valid_seg * valid_seg)))

    threshold_linear = max(
        10.0 ** (threshold_db / 20.0),
        float(absolute_floor_after_norm),
    )
    active = rms > threshold_linear

    if hangover_frames > 0 and n_frames > 0:
        # Close short gaps: if active before and after a gap <= hangover, fill it.
        out = active.copy()
        gap_count = 0
        for i in range(1, n_frames):
            if active[i]:
                if gap_count > 0 and gap_count <= hangover_frames:
                    out[i - gap_count : i] = True
                gap_count = 0
            else:
                gap_count += 1
        active = out
    return active


def compute_aec_state_attributes(
    nearend: np.ndarray,
    echo: np.ndarray,
    noise: np.ndarray | None = None,
    hop_length: int = 256,
    win_length: int | None = None,
    near_threshold_db: float = -40.0,
    echo_threshold_db: float = -50.0,
    noise_threshold_db: float = -40.0,
    hangover_frames: int = 1,
    absolute_floor_after_norm: float = 1e-4,
    pre_emphasis: float = 0.97,
) -> np.ndarray:
    """Compute frame-level multi-label presence attributes for AEC state detector.

    Returns a binary matrix of shape ``(n_frames, 3)`` where the three columns
    correspond to ``[near_present, echo_present, noise_present]``.

    Args:
        nearend: Clean near-end (target) waveform, shape ``(T,)``.
        echo: Echo waveform actually present in mic, shape ``(T,)``.
        noise: Additive noise waveform, shape ``(T,)``.  If ``None`` or empty,
            treated as silence (all ``noise_present=0``).
        hop_length: STFT hop length; determines frame rate.
        win_length: STFT window length for centred energy computation; defaults
            to ``hop_length`` if not provided.
        near_threshold_db: Relative energy threshold for near-end (dB below
            normalized peak).
        echo_threshold_db: Relative energy threshold for echo.
        noise_threshold_db: Relative energy threshold for additive noise.
        hangover_frames: Hangover for gap filling.
        absolute_floor_after_norm: RMS floor in the peak-normalized domain; not
            an absolute waveform-level or dBFS threshold.
        pre_emphasis: First-order high-pass pre-emphasis coefficient applied
            inside the activity mask.

    Returns:
        Float32 array of shape ``(n_frames, 3)`` with values ``0.0`` or ``1.0``.
    """
    _validate_state_label_inputs(
        nearend, hop_length, win_length,
        threshold_db=near_threshold_db,
        hangover_frames=hangover_frames,
        absolute_floor_after_norm=absolute_floor_after_norm,
        pre_emphasis=pre_emphasis,
    )
    _validate_state_label_inputs(
        echo, hop_length, win_length,
        threshold_db=echo_threshold_db,
        hangover_frames=hangover_frames,
        absolute_floor_after_norm=absolute_floor_after_norm,
        pre_emphasis=pre_emphasis,
    )
    if noise is not None:
        _validate_state_label_inputs(
            noise, hop_length, win_length,
            threshold_db=noise_threshold_db,
            hangover_frames=hangover_frames,
            absolute_floor_after_norm=absolute_floor_after_norm,
            pre_emphasis=pre_emphasis,
        )
    near_vad = _state_frame_activity_mask(
        nearend, hop_length, win_length, near_threshold_db, hangover_frames,
        absolute_floor_after_norm, pre_emphasis,
    )
    echo_vad = _state_frame_activity_mask(
        echo, hop_length, win_length, echo_threshold_db, hangover_frames,
        absolute_floor_after_norm, pre_emphasis,
    )
    if noise is not None and len(np.asarray(noise).ravel()) > 0:
        noise_vad = _state_frame_activity_mask(
            noise, hop_length, win_length, noise_threshold_db, hangover_frames,
            absolute_floor_after_norm, pre_emphasis,
        )
    else:
        noise_vad = np.zeros_like(near_vad)

    n_frames = max(len(near_vad), len(echo_vad), len(noise_vad))
    attrs = np.zeros((n_frames, 3), dtype=np.float32)
    if len(near_vad) < n_frames:
        near_vad = np.pad(near_vad, (0, n_frames - len(near_vad)), constant_values=False)
    if len(echo_vad) < n_frames:
        echo_vad = np.pad(echo_vad, (0, n_frames - len(echo_vad)), constant_values=False)
    if len(noise_vad) < n_frames:
        noise_vad = np.pad(noise_vad, (0, n_frames - len(noise_vad)), constant_values=False)

    attrs[:, 0] = near_vad.astype(np.float32)
    attrs[:, 1] = echo_vad.astype(np.float32)
    attrs[:, 2] = noise_vad.astype(np.float32)
    return attrs


# 8-class scenario mapping derived from [near_present, echo_present, noise_present].
#   bit 2 (weight 4) = near, bit 1 (weight 2) = echo, bit 0 (weight 1) = noise.
#   class 0: silence, 1: prob_noise_only, 2: echo_only, 3: noise_echo,
#   class 4: near_only, 5: near_noise, 6: near_echo (DT), 7: all.
_STATE_SCENARIO_SILENCE = 0
_STATE_SCENARIO_NOISE_ONLY = 1
_STATE_SCENARIO_ECHO_ONLY = 2
_STATE_SCENARIO_NOISE_ECHO = 3
_STATE_SCENARIO_NEAR_ONLY = 4
_STATE_SCENARIO_NEAR_NOISE = 5
_STATE_SCENARIO_NEAR_ECHO = 6
_STATE_SCENARIO_ALL = 7


def compute_aec_state_scenario_label(
    nearend: np.ndarray,
    echo: np.ndarray,
    noise: np.ndarray | None = None,
    hop_length: int = 256,
    win_length: int | None = None,
    near_threshold_db: float = -40.0,
    echo_threshold_db: float = -50.0,
    noise_threshold_db: float = -40.0,
    hangover_frames: int = 1,
    absolute_floor_after_norm: float = 1e-4,
    pre_emphasis: float = 0.97,
    smoothing_window_frames: int = 1,
) -> np.ndarray:
    """Compute a frame-level 8-class scenario label from presence attributes.

    The class index is formed from the three binary presence bits
    ``[near_present, echo_present, noise_present]`` as
    ``class = near*4 + echo*2 + noise*1``.

    This maps the mutually-exclusive branches of ``DynamicMixingDataset`` to
    frame-level presence combinations:

    ===== =============== ==========================================
    class bits [n,e,ns]   meaning
    ===== =============== ==========================================
      0   [0, 0, 0]       silence / clean background
      1   [0, 0, 1]       prob_noise_only
      2   [0, 1, 0]       echo_only  (FST)
      3   [0, 1, 1]       noise + echo
      4   [1, 0, 0]       near_only  (NST, e.g. prob_target_only branch)
      5   [1, 0, 1]       near + noise
      6   [1, 1, 0]       near + echo (DT)
      7   [1, 1, 1]       all three present
    ===== =============== ==========================================

    Args:
        nearend, echo, noise: Waveforms, shape ``(T,)``.
        hop_length, win_length: STFT framing; see ``compute_aec_state_attributes``.
        near_threshold_db, echo_threshold_db, noise_threshold_db: Activity
            thresholds passed through to ``compute_aec_state_attributes``.
        hangover_frames, absolute_floor_after_norm, pre_emphasis: See
            ``compute_aec_state_attributes``.
        smoothing_window_frames: Optional sliding-window mode filter width
            applied to the final 8-class sequence to suppress isolated frame
            flicker.  Values <= 1 disable smoothing.

    Returns:
        Integer array of shape ``(n_frames,)`` with values in ``[0, 7]``.
    """
    _validate_state_label_inputs(
        nearend, hop_length, win_length, smoothing_window_frames=smoothing_window_frames
    )
    attrs = compute_aec_state_attributes(
        nearend,
        echo,
        noise,
        hop_length=hop_length,
        win_length=win_length,
        near_threshold_db=near_threshold_db,
        echo_threshold_db=echo_threshold_db,
        noise_threshold_db=noise_threshold_db,
        hangover_frames=hangover_frames,
        absolute_floor_after_norm=absolute_floor_after_norm,
        pre_emphasis=pre_emphasis,
    )
    bits = (attrs > 0.5).astype(np.int64)
    state = bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]
    state = _smooth_state_sequence(state.astype(np.int64), smoothing_window_frames)
    return state.astype(np.int64)


def compute_aec_state_label(
    nearend: np.ndarray,
    echo: np.ndarray,
    hop_length: int,
    win_length: int | None = None,
    near_threshold_db: float = -40.0,
    echo_threshold_db: float = -50.0,
    hangover_frames: int = 1,
    absolute_floor_after_norm: float = 1e-4,
    pre_emphasis: float = 0.97,
    smoothing_window_frames: int = 1,
) -> np.ndarray:
    """Compute a frame-level FST/DT/NST label for AEC state detector.

    This is the coarse 3-class view of the 8-class scenario label.  The mapping
    from 8-class ``[near, echo, noise]`` presence bits to 3-class is:

    ===== =============== ==========================================
    class presence bits   3-class label
    ===== =============== ==========================================
      2   [0, 1, 0]       FST  (echo only)
      6   [1, 1, 0]       DT   (near + echo)
      0   [0, 0, 0]       NST  (silence / clean background)
      1   [0, 0, 1]       NST  (noise only)
      4   [1, 0, 0]       NST  (near only)
      5   [1, 0, 1]       NST  (near + noise)
    ===== =============== ==========================================

    Because this function does not inspect the noise channel, any frame without
    both near-end and echo activity collapses to ``STATE_LABEL_NST``.

    Args:
        nearend: Near-end (target) waveform, shape ``(T,)``.
        echo: Echo waveform actually present in mic, shape ``(T,)``.
        hop_length: STFT hop length; determines frame rate.
        win_length: STFT window length for centred energy computation; defaults
            to ``hop_length`` if not provided.
        near_threshold_db: Relative energy threshold for near-end (dB below
            normalized peak).  Echo is usually weaker, so it gets a more
            lenient threshold by default.
        echo_threshold_db: Relative energy threshold for echo.
        hangover_frames: Hangover for gap filling.
        absolute_floor_after_norm: RMS floor in the peak-normalized domain; not
            an absolute waveform-level or dBFS threshold.
        pre_emphasis: First-order high-pass pre-emphasis coefficient applied
            inside the activity mask.  Use ``0.0`` to disable.
        smoothing_window_frames: Optional sliding-window mode filter width
            applied to the final 3-class sequence to suppress isolated frame
            flicker.  Values <= 1 disable smoothing.

    Returns:
        Integer array of shape ``(n_frames,)`` where each element is one of
        ``STATE_LABEL_FST``, ``STATE_LABEL_DT``, ``STATE_LABEL_NST``.
    """
    _validate_state_label_inputs(
        nearend,
        hop_length,
        win_length,
        threshold_db=near_threshold_db,
        hangover_frames=hangover_frames,
        absolute_floor_after_norm=absolute_floor_after_norm,
        pre_emphasis=pre_emphasis,
        smoothing_window_frames=smoothing_window_frames,
    )
    _validate_state_label_inputs(
        echo,
        hop_length,
        win_length,
        threshold_db=echo_threshold_db,
        hangover_frames=hangover_frames,
        absolute_floor_after_norm=absolute_floor_after_norm,
        pre_emphasis=pre_emphasis,
    )
    near_vad = _state_frame_activity_mask(
        nearend, hop_length, win_length, near_threshold_db, hangover_frames,
        absolute_floor_after_norm, pre_emphasis,
    )
    echo_vad = _state_frame_activity_mask(
        echo, hop_length, win_length, echo_threshold_db, hangover_frames,
        absolute_floor_after_norm, pre_emphasis,
    )
    n_frames = max(len(near_vad), len(echo_vad))
    if len(near_vad) < n_frames:
        near_vad = np.pad(near_vad, (0, n_frames - len(near_vad)), constant_values=False)
    if len(echo_vad) < n_frames:
        echo_vad = np.pad(echo_vad, (0, n_frames - len(echo_vad)), constant_values=False)

    state = np.full(n_frames, STATE_LABEL_NST, dtype=np.int64)
    state[echo_vad & (~near_vad)] = STATE_LABEL_FST
    state[near_vad & echo_vad] = STATE_LABEL_DT
    # near_vad & ~echo_vad -> NST (already default)
    # ~near_vad & ~echo_vad -> NST (already default)
    state = _smooth_state_sequence(state, smoothing_window_frames)
    return state
