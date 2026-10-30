"""Time-delay jitter augmentation for AEC training (echo_path only).

This module models time-base disturbances on the echo path:
- sudden_jump: piecewise-constant delay jumps (buffer xrun, path switch, clock
  step compensation)
- gradual_drift: sample-rate offset / clock drift (SRO)
- periodic_jitter: sinusoidal scheduling/PLL jitter
- mixed: sum of the above

It is intentionally separate from ``tasks/aec/data/dynamic_dataset.py`` so that the
trajectory generation, fractional retiming and mask bookkeeping can be tested
in isolation.

Sign convention
---------------
``delay_gt[t]`` is positive when mic/echo lags the reference (far) signal.
The caller passes the current base delay ``base_delay`` (samples); the module
generates a new per-sample delay trajectory and retimes the echo stem so that
the resulting ref/echo/mic relationship has that trajectory.

The training pipeline only supports ``echo_path``: the reference (farend) is
kept unchanged and the echo stem is retimed; the mic is rebuilt as
``mic - echo + echo_jittered``. ``reference_capture`` / ``capture_buffer``
distortions belong to A–F dataset simulation or the AEC input/deployment frontend and are not wired here.

Retiming implementation
-----------------------
All jitter types export one authoritative ``delay_gt`` trajectory. Waveform
retiming and valid-mask bookkeeping delegate to ``core.data.timebase`` using
that same trajectory, including after non-negative / relative / total-delay
clipping. This keeps waveform, labels and audit metadata on one timebase.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from numbers import Real
from typing import Any

import numpy as np

from core.data.simulation.timebase_impairment import (
    DelayTrajectoryPlan,
    apply_delay_trajectory_plan,
)

# Small constants used to avoid division-by-zero / log(0) without obscuring intent.
_MIN_LOG_ARG_MS = 1e-3  # 1 us lower bound for log-uniform magnitude sampling
_MIN_POSITIVE_HZ = 1e-9  # avoid division by zero in periodic ppm cap
_MIN_TAU_MS = 1e-6  # avoid division by zero in correlation smoothing

# 单样本跳变事件数上限：防御误配的超高 jitter_rate 造成无界循环/内存增长。
_MAX_EVENT_COUNT = 100_000

# ---------------------------------------------------------------------------
# Config / result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeDelayJitterConfig:
    """Configuration for time-delay jitter augmentation.

    Units are ms where the field name ends with ``_ms``; ``_ppm`` is parts per
    million; ``_hz`` is cycles per second. ``jitter_rate`` is in events per
    second. Ranges are inclusive two-element sequences ``[lo, hi]``.

    ``jitter_corr_length`` controls how strongly adjacent jump amplitudes are
    correlated: the smoothing factor between event ``i-1`` and ``i`` is
    ``alpha = exp(-dt_ms / tau_ms)`` where ``tau_ms`` is drawn uniformly from
    ``jitter_corr_length``. Short correlation length => independent jumps;
    long correlation length => successive jumps move in similar directions.

    ``transition_ms`` linearly ramps the delay change at each jump event over
    the specified duration, instead of an instantaneous step. This models
    real-world route switches / clock adjustments that take a finite time.

    ``jump_ms_range`` gives the raw signed jump amplitude bounds. Magnitudes are
    sampled log-uniformly by default so small jumps are more frequent than huge
    jumps; the sign follows the interval (negative when both bounds are
    negative, random when the interval straddles zero). A range that strictly
    straddles zero must currently be symmetric to keep both sampled signs within
    the configured bounds.

    ``jitter_max_ms`` caps the magnitude of a single raw event amplitude
    (sign is preserved). It is applied before the correlation smoothing, so the
    final realized amplitude may be smaller but never exceeds this cap. This is
    useful for preventing occasional huge jumps even when ``jump_ms_range`` is
    wide.
    """

    enabled: bool = False
    prob: float = 0.0
    jitter_type: str = "sudden_jump"  # sudden_jump|gradual_drift|periodic_jitter|mixed
    # Optional per-type probabilities, like aec_echo_clipping.path_probs.
    # If provided, it overrides the single jitter_type string for sample-level
    # type sampling. Keys must be a subset of the four allowed types and sum to 1.0.
    jitter_type_probs: dict[str, float] | None = None
    jitter_max_ms: float = 500.0
    jitter_rate: tuple[float, float] = (0.02, 0.5)  # events/s
    jitter_corr_length: tuple[float, float] = (100.0, 2000.0)  # ms
    jump_ms_range: tuple[float, float] = (-500.0, 500.0)
    drift_ppm_range: tuple[float, float] = (-200.0, 200.0)
    periodic_amp_ms_range: tuple[float, float] = (0.1, 10.0)
    periodic_hz_range: tuple[float, float] = (0.1, 5.0)
    # Cap the equivalent peak sample-rate offset of periodic jitter.
    # delay(t) = A_ms * sin(2π f t). The local stretch factor is
    # d(delay_sec)/dt = (A_ms / 1000) * 2π f. Expressed as ppm:
    # ppm_peak = A_ms * 2π f * 1000. Therefore:
    #   A_ms <= ppm_limit / (2π f * 1000).
    periodic_max_equivalent_ppm: float = 1000.0
    min_segment_ms: float = 200.0
    transition_ms: float = 0.0
    # Boundary handling for fractional reads outside [0, T-1].
    # "constant" zero-pads; "nearest" repeats the boundary sample.
    edge_mode: str = "constant"
    # Maximum absolute deviation of the generated delay trajectory from base_delay.
    # The total delay is also clamped to be non-negative, so echo never leads ref.
    max_relative_jitter_ms: float = 1000.0

    def __post_init__(self) -> None:
        lo_ms, hi_ms = self.jump_ms_range
        if lo_ms < 0.0 < hi_ms and abs(lo_ms) != abs(hi_ms):
            raise ValueError(
                "aec_time_delay_jitter.jump_ms_range must be symmetric when it "
                f"straddles zero, got {self.jump_ms_range!r}"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "TimeDelayJitterConfig":
        """Build from a YAML-parsed mapping, with strict validation."""
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError(
                "aec_time_delay_jitter must be a mapping, "
                f"got {type(raw).__name__}"
            )
        cfg = dict(raw)
        legacy_key = "max_abs_delay_ms"
        canonical_key = "max_relative_jitter_ms"
        if legacy_key in cfg:
            raise ValueError(
                f"aec_time_delay_jitter.{legacy_key} was removed; use "
                f"aec_time_delay_jitter.{canonical_key} instead."
            )
        known = {f.name for f in fields(cls)}
        for key in cfg:
            if not isinstance(key, str) or key not in known:
                raise ValueError(f"Unknown key in aec_time_delay_jitter: {key!r}")

        def _finite_real(name: str, value: Any) -> float:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(
                    f"aec_time_delay_jitter.{name} must be a finite real number, "
                    f"got {value!r}"
                )
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(
                    f"aec_time_delay_jitter.{name} must be finite, got {value!r}"
                )
            return result

        def _pair(name: str) -> tuple[float, float]:
            value = cfg.get(name, getattr(cls, name))
            if not (isinstance(value, (list, tuple)) and len(value) == 2):
                raise ValueError(
                    f"aec_time_delay_jitter.{name} must be a two-element range, "
                    f"got {value!r}"
                )
            lo = _finite_real(f"{name}[0]", value[0])
            hi = _finite_real(f"{name}[1]", value[1])
            if lo > hi:
                raise ValueError(
                    f"aec_time_delay_jitter.{name} must satisfy lo <= hi, got {value!r}"
                )
            return (lo, hi)

        enabled_raw = cfg.get("enabled", cls.enabled)
        if not isinstance(enabled_raw, (bool, np.bool_)):
            raise ValueError(
                "aec_time_delay_jitter.enabled must be a boolean, "
                f"got {enabled_raw!r}"
            )
        enabled = bool(enabled_raw)
        prob = _finite_real("prob", cfg.get("prob", cls.prob))
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"aec_time_delay_jitter.prob must be in [0,1], got {prob}")

        jitter_type_raw = cfg.get("jitter_type", cls.jitter_type)
        if not isinstance(jitter_type_raw, str):
            raise ValueError(
                "aec_time_delay_jitter.jitter_type must be a string, "
                f"got {jitter_type_raw!r}"
            )
        jitter_type = jitter_type_raw.strip().lower()
        _allowed_types = ("sudden_jump", "gradual_drift", "periodic_jitter", "mixed")
        if jitter_type not in _allowed_types:
            raise ValueError(
                f"aec_time_delay_jitter.jitter_type must be one of "
                f"sudden_jump|gradual_drift|periodic_jitter|mixed, got {jitter_type!r}"
            )

        jitter_type_probs_raw = cfg.get("jitter_type_probs")
        jitter_type_probs: dict[str, float] | None = None
        if jitter_type_probs_raw is not None:
            if not isinstance(jitter_type_probs_raw, Mapping) or not jitter_type_probs_raw:
                raise ValueError(
                    "aec_time_delay_jitter.jitter_type_probs must be a non-empty mapping, "
                    f"got {jitter_type_probs_raw!r}"
                )
            normalized_probs: dict[str, float] = {}
            for key, value in jitter_type_probs_raw.items():
                if key not in _allowed_types:
                    raise ValueError(
                        f"aec_time_delay_jitter.jitter_type_probs key must be one of "
                        f"sudden_jump|gradual_drift|periodic_jitter|mixed, got {key!r}"
                    )
                probability = _finite_real(f"jitter_type_probs[{key!r}]", value)
                if not (0.0 <= probability <= 1.0):
                    raise ValueError(
                        f"aec_time_delay_jitter.jitter_type_probs[{key!r}] "
                        f"must be in [0,1], got {value!r}"
                    )
                normalized_probs[key] = probability
            total = sum(normalized_probs.values())
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-5):
                raise ValueError(
                    f"aec_time_delay_jitter.jitter_type_probs must sum to 1.0, got {total}"
                )
            jitter_type_probs = normalized_probs

        jitter_max_ms = _finite_real(
            "jitter_max_ms", cfg.get("jitter_max_ms", cls.jitter_max_ms)
        )
        max_relative_jitter_ms = _finite_real(
            "max_relative_jitter_ms",
            cfg.get("max_relative_jitter_ms", cls.max_relative_jitter_ms),
        )
        min_segment_ms = _finite_real(
            "min_segment_ms", cfg.get("min_segment_ms", cls.min_segment_ms)
        )
        transition_ms = _finite_real(
            "transition_ms", cfg.get("transition_ms", cls.transition_ms)
        )
        if jitter_max_ms <= 0.0:
            raise ValueError(f"aec_time_delay_jitter.jitter_max_ms must be > 0, got {jitter_max_ms}")
        if max_relative_jitter_ms < 0.0:
            raise ValueError(
                f"aec_time_delay_jitter.max_relative_jitter_ms must be >= 0, got {max_relative_jitter_ms}"
            )
        if min_segment_ms < 0.0:
            raise ValueError(f"aec_time_delay_jitter.min_segment_ms must be >= 0, got {min_segment_ms}")
        if transition_ms < 0.0:
            raise ValueError(f"aec_time_delay_jitter.transition_ms must be >= 0, got {transition_ms}")
        jitter_rate = _pair("jitter_rate")
        jitter_corr_length = _pair("jitter_corr_length")
        jump_ms_range = _pair("jump_ms_range")
        drift_ppm_range = _pair("drift_ppm_range")
        periodic_amp_ms_range = _pair("periodic_amp_ms_range")
        periodic_hz_range = _pair("periodic_hz_range")
        if jitter_rate[0] < 0.0:
            raise ValueError(f"jitter_rate must satisfy 0 <= lo <= hi, got {jitter_rate!r}")
        if jitter_corr_length[0] < 0.0:
            raise ValueError(
                "jitter_corr_length must satisfy 0 <= lo <= hi, "
                f"got {jitter_corr_length!r}"
            )
        if periodic_hz_range[0] <= 0.0:
            raise ValueError(
                "periodic_hz_range must satisfy 0 < lo <= hi, "
                f"got {periodic_hz_range!r}"
            )
        if periodic_amp_ms_range[0] < 0.0:
            raise ValueError(
                "periodic_amp_ms_range must satisfy 0 <= lo <= hi, "
                f"got {periodic_amp_ms_range!r}"
            )

        edge_mode_raw = cfg.get("edge_mode", cls.edge_mode)
        if not isinstance(edge_mode_raw, str):
            raise ValueError(
                "aec_time_delay_jitter.edge_mode must be a string, "
                f"got {edge_mode_raw!r}"
            )
        edge_mode = edge_mode_raw.strip().lower()
        if edge_mode not in ("constant", "nearest"):
            raise ValueError(
                f"aec_time_delay_jitter.edge_mode must be 'constant' or 'nearest', got {edge_mode!r}"
            )
        periodic_max_equivalent_ppm = _finite_real(
            "periodic_max_equivalent_ppm",
            cfg.get("periodic_max_equivalent_ppm", cls.periodic_max_equivalent_ppm),
        )
        if periodic_max_equivalent_ppm <= 0.0:
            raise ValueError(
                f"aec_time_delay_jitter.periodic_max_equivalent_ppm must be finite and > 0, "
                f"got {periodic_max_equivalent_ppm}"
            )
        return cls(
            enabled=enabled,
            prob=prob,
            jitter_type=jitter_type,
            jitter_type_probs=jitter_type_probs,
            jitter_max_ms=jitter_max_ms,
            jitter_rate=jitter_rate,
            jitter_corr_length=jitter_corr_length,
            jump_ms_range=jump_ms_range,
            drift_ppm_range=drift_ppm_range,
            periodic_amp_ms_range=periodic_amp_ms_range,
            periodic_hz_range=periodic_hz_range,
            periodic_max_equivalent_ppm=periodic_max_equivalent_ppm,
            min_segment_ms=min_segment_ms,
            transition_ms=transition_ms,
            edge_mode=edge_mode,
            max_relative_jitter_ms=max_relative_jitter_ms,
        )


@dataclass
class JitterResult:
    """Output of ``apply_time_delay_jitter`` (echo_path only)."""

    mic_jittered: np.ndarray  # [T]; echo_jittered copy when mic was None
    echo_jittered: np.ndarray  # [T]
    delay_gt: np.ndarray  # [T], samples, positive = mic/echo lags ref
    valid_mask: np.ndarray  # [T], bool
    event_mask: np.ndarray  # [T], bool; True where delay is non-static
    meta: dict[str, Any]


# ---------------------------------------------------------------------------
# Delay trajectory generators
# ---------------------------------------------------------------------------


def _sample_event_times_ms(
    rng: np.random.Generator,
    rate_range: tuple[float, float],
    min_segment_ms: float,
    duration_ms: float,
) -> list[float]:
    """Sample jump event times in milliseconds with a minimum dwell between events.

    Intervals are drawn from a truncated exponential distribution with lower
    bound ``min_seg``. This keeps the expected interval equal to ``1/rate``
    when ``min_seg`` is small, and avoids the artificial rate inflation caused
    by hard-clamping sub-minimum intervals.

    超过 ``_MAX_EVENT_COUNT`` 个事件时抛 ``ValueError``（误配的 jitter_rate
    防护），不会静默截断事件序列。
    """
    if duration_ms <= 0.0:
        return []
    rate = float(rng.uniform(*rate_range))
    if rate <= 0.0:
        return []
    min_seg = max(0.0, min_segment_ms)
    mean_interval = 1000.0 / rate  # desired mean interval
    # For a truncated exponential with lower bound min_seg, the mean is
    # min_seg + adjusted_mean. We choose adjusted_mean so that the total mean
    # matches mean_interval (unless min_seg already exceeds it).
    adjusted_mean = max(mean_interval - min_seg, 1e-9)
    intervals_ms: list[float] = []
    total = 0.0
    while total < duration_ms:
        # 误配的超高 jitter_rate（或 min_segment_ms=0 且指数抽样极小）会让事件数
        # 无界增长；达到上限即显式拒绝，而不是继续累积。
        if len(intervals_ms) >= _MAX_EVENT_COUNT:
            raise ValueError(
                "aec_time_delay_jitter generated more than "
                f"{_MAX_EVENT_COUNT} jump events for one sample "
                f"(jitter_rate={rate:g} events/s, min_segment_ms={min_seg:g}, "
                f"duration_ms={duration_ms:g}); check jitter_rate/min_segment_ms."
            )
        dt = min_seg + float(rng.exponential(adjusted_mean))
        total += dt
        intervals_ms.append(dt)
    event_ms = np.cumsum(intervals_ms).tolist()
    # Drop events past duration and the first if it lands exactly at 0.
    event_ms = [t for t in event_ms if 0.0 < t < duration_ms]
    return event_ms


def _sample_signed_log_uniform_ms(
    rng: np.random.Generator,
    lo_ms: float,
    hi_ms: float,
    max_mag_ms: float | None = None,
) -> float:
    """Sample a signed jump amplitude (ms) with log-uniform magnitude.

    - If ``lo`` and ``hi`` have the same sign, the sign is fixed and the
      magnitude is log-uniform between ``|lo|`` and ``|hi|``.
    - If the interval straddles zero, the magnitude is log-uniform between a
      small epsilon and ``max(|lo|, |hi|)`` and the sign is random.
    - If ``lo == hi`` (including zero), return that value exactly.
    - ``max_mag_ms`` optionally clamps the magnitude of the returned sample
      (sign is preserved). This is used to enforce ``jitter_max_ms`` as a
      per-event amplitude cap.
    """
    if lo_ms == hi_ms:
        amp = float(lo_ms)
        if max_mag_ms is not None:
            amp = math.copysign(min(abs(amp), max_mag_ms), amp)
        return amp
    eps = _MIN_LOG_ARG_MS
    if lo_ms >= 0.0:
        mag = math.exp(rng.uniform(math.log(max(eps, lo_ms)), math.log(max(eps, hi_ms))))
        amp = float(mag)
    elif hi_ms <= 0.0:
        mag = math.exp(rng.uniform(math.log(max(eps, abs(hi_ms))), math.log(max(eps, abs(lo_ms)))))
        amp = float(-mag)
    else:
        # Interval straddles zero: sample magnitude and random sign.
        max_mag = max(abs(lo_ms), abs(hi_ms))
        mag = math.exp(rng.uniform(math.log(eps), math.log(max(eps, max_mag))))
        amp = float(mag if rng.random() < 0.5 else -mag)
    if max_mag_ms is not None:
        amp = math.copysign(min(abs(amp), max_mag_ms), amp)
    return amp


def _correlated_jump_amplitudes(
    rng: np.random.Generator,
    event_ms: list[float],
    jump_ms_range: tuple[float, float],
    corr_length_ms_range: tuple[float, float],
    max_mag_ms: float | None = None,
) -> list[float]:
    """Generate exponentially-smoothed jump amplitudes.

    ``amp[i] = alpha * amp[i-1] + (1-alpha) * raw_amp[i]`` where
    ``alpha = exp(-dt / tau)`` and ``tau`` is drawn per gap from
    ``corr_length_ms_range``.

    ``max_mag_ms`` clamps each raw event amplitude before the smoothing mix,
    preserving sign. The smoothed result stays inside the same bound because
    the mix is convex.
    """
    if not event_ms:
        return []
    amplitudes: list[float] = []
    prev_amp = 0.0
    for i, t_ms in enumerate(event_ms):
        if i == 0:
            alpha = 0.0
        else:
            dt_ms = max(0.0, t_ms - event_ms[i - 1])
            tau_ms = float(rng.uniform(*corr_length_ms_range))
            alpha = math.exp(-dt_ms / max(tau_ms, _MIN_TAU_MS))
        raw_amp = _sample_signed_log_uniform_ms(
            rng, *jump_ms_range, max_mag_ms=max_mag_ms
        )
        amp = alpha * prev_amp + (1.0 - alpha) * raw_amp
        amplitudes.append(float(amp))
        prev_amp = amp
    return amplitudes


def _apply_transition_ramps(
    delay_gt: np.ndarray,
    event_indices: list[int],
    transition_len: int,
) -> np.ndarray:
    """Replace each instantaneous step with a linear ramp.

    The ramp starts at the event index and spans ``transition_len`` samples,
    transitioning from the old delay value to the new delay value. If events
    are closer than ``transition_len``, the ramps overlap and their contributions
    are summed.
    """
    if transition_len <= 0 or not event_indices:
        return delay_gt
    T = delay_gt.size
    # rate[t] = delay_pc[t] - delay_pc[t-1]。每个跳变的均匀 ramp 通过差分
    # 数组一次累计，避免逐事件切片写入导致 O(T * events) 的 Python 调度。
    rate = np.diff(delay_gt, prepend=float(delay_gt[0]))
    nonzero = np.flatnonzero(rate)
    if nonzero.size == 0:
        return delay_gt.astype(np.float64, copy=True)
    ends = np.minimum(T, nonzero + transition_len)
    masses = rate[nonzero] / (ends - nonzero)
    diff = np.zeros(T + 1, dtype=np.float64)
    np.add.at(diff, nonzero, masses)
    np.add.at(diff, ends, -masses)
    smoothed_rate = np.cumsum(diff[:-1])
    return float(delay_gt[0]) + np.cumsum(smoothed_rate)


def _generate_delay_trajectory(
    T: int,
    base_delay: float | np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    cfg: TimeDelayJitterConfig,
    jitter_type: str | None = None,
    max_total_delay_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate a per-sample delay trajectory, event mask and drift component.

    Returns ``(delay_gt_samples, event_mask, drift_delta, meta)`` where
    ``delay_gt_samples`` has shape ``(T,)`` and ``drift_delta`` is the delay
    contribution of the drift component alone (zero if absent). ``event_mask``
    is derived after all clipping from the final trajectory: index 0 is False,
    and each later index is True exactly when its delay differs from the previous
    sample. Thus fully clipped mechanisms and clipped plateaus remain unmarked.

    Detailed jump metadata (position and amplitude) is available in ``meta``.

    ``max_total_delay_ms`` is an optional hard cap on the total absolute delay.
    When provided, ``delay_gt`` is clipped to ``[0, max_total_delay_ms]`` after
    the relative jitter bound is applied.
    """
    _jitter_type = (jitter_type or cfg.jitter_type).strip().lower()
    if isinstance(base_delay, np.ndarray):
        d0 = np.asarray(base_delay, dtype=np.float64).ravel()
        if d0.size != T:
            raise ValueError(f"base_delay array length {d0.size} != signal length {T}")
    else:
        d0 = np.full(T, float(base_delay), dtype=np.float64)

    delay_gt = d0.copy()
    drift_delta = np.zeros(T, dtype=np.float64)
    duration_ms = T / max(1, sample_rate) * 1000.0

    has_jump = _jitter_type in ("sudden_jump", "mixed")
    has_drift = _jitter_type in ("gradual_drift", "mixed")
    has_periodic = _jitter_type in ("periodic_jitter", "mixed")

    meta_events: list[dict[str, Any]] = []
    drift_ppm: float | None = None
    periodic_amp_ms: float | None = None
    periodic_hz: float | None = None
    transition_len = int(round(cfg.transition_ms / 1000.0 * sample_rate))

    if has_jump:
        event_ms = _sample_event_times_ms(
            rng, cfg.jitter_rate, cfg.min_segment_ms, duration_ms
        )
        amplitudes_ms = _correlated_jump_amplitudes(
            rng,
            event_ms,
            cfg.jump_ms_range,
            cfg.jitter_corr_length,
            max_mag_ms=cfg.jitter_max_ms,
        )
        event_indices = [
            max(0, min(T - 1, int(round(t / 1000.0 * sample_rate))))
            for t in event_ms
        ]
        jump_deltas = np.zeros(T, dtype=np.float64)
        prior_jump_delta = 0.0
        for idx, t_ms, amp_ms in zip(event_indices, event_ms, amplitudes_ms):
            amp_samp = amp_ms / 1000.0 * sample_rate
            # event_indices 按时间有序；累计先前事件即可保留同点事件的顺序语义。
            # 实际 trajectory 在循环后一次 cumsum，避免每个事件切片累加。
            pre_delay_samples = float(delay_gt[idx] + prior_jump_delta)
            jump_deltas[idx] += amp_samp
            prior_jump_delta += amp_samp
            meta_events.append({
                "pos_ms": float(t_ms),
                "amp_ms": float(amp_ms),
                "amp_samples": float(amp_samp),
                "pre_delay_samples": pre_delay_samples,
            })
        delay_gt += np.cumsum(jump_deltas)
        # Replace instantaneous steps with linear ramps.
        if transition_len > 0:
            delay_gt = _apply_transition_ramps(delay_gt, event_indices, transition_len)

    if has_drift:
        drift_ppm = float(rng.uniform(*cfg.drift_ppm_range))
        per_sample = drift_ppm * 1e-6
        # delay_gt += cumulative SRO offset（0-indexed，与共享 trajectory apply 对齐）。
        drift_delta = np.arange(T, dtype=np.float64) * per_sample
        delay_gt += drift_delta

    if has_periodic:
        periodic_amp_ms = float(rng.uniform(*cfg.periodic_amp_ms_range))
        periodic_hz = float(rng.uniform(*cfg.periodic_hz_range))
        # Limit equivalent peak sample-rate offset to avoid unrealistic local
        # stretching. Correct ppm relation: A_ms * 2π * hz * 1000 <= ppm_limit.
        max_amp_ms = cfg.periodic_max_equivalent_ppm / (
            2.0 * math.pi * max(periodic_hz, _MIN_POSITIVE_HZ) * 1000.0
        )
        periodic_amp_ms = min(periodic_amp_ms, max_amp_ms)
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        t = np.arange(T, dtype=np.float64) / max(1, sample_rate)
        delay_gt += (periodic_amp_ms / 1000.0 * sample_rate) * np.sin(2.0 * np.pi * periodic_hz * t + phase)

    # Bound the jitter magnitude relative to base_delay, keep the total delay
    # physically non-negative (echo cannot lead ref), and optionally enforce a
    # hard upper cap (e.g. the model's maximum supported delay).
    max_samp = cfg.max_relative_jitter_ms / 1000.0 * sample_rate
    max_total_samp = (
        float(max_total_delay_ms) / 1000.0 * sample_rate
        if max_total_delay_ms is not None
        else None
    )
    delta = delay_gt - d0
    delta_clipped = np.clip(delta, -max_samp, max_samp)
    clipped = np.clip(d0 + delta_clipped, 0.0, None)
    if max_total_samp is not None:
        clipped = np.clip(clipped, 0.0, max_total_samp)

    # 最终轨迹是标签的唯一事实源：仅标记当前样点相对前一样点发生的
    # 实际时延变化。被非负/相对/总时延裁剪抹平的机制不应留下事件标签。
    event_mask = np.zeros(T, dtype=bool)
    if T > 1:
        event_mask[1:] = np.diff(clipped) != 0.0

    # Record the realized amplitude after clipping (may differ from requested).
    for ev in meta_events:
        idx = int(round(float(ev["pos_ms"]) / 1000.0 * sample_rate))
        idx = max(0, min(T - 1, idx))
        # Pre-event delay clipped with the same bounds that were applied to the
        # full trajectory. This handles idx=0 correctly and isolates the jump.
        pre = float(ev["pre_delay_samples"])
        base_at_idx = float(d0[idx])
        pre_delta_clipped = float(np.clip(pre - base_at_idx, -max_samp, max_samp))
        pre_clipped = pre_delta_clipped + base_at_idx
        if max_total_samp is not None:
            pre_clipped = min(pre_clipped, max_total_samp)
        # Instantaneous realized step at the event onset.
        realized_samp = float(clipped[idx]) - pre_clipped
        ev["realized_amp_samples"] = realized_samp
        ev["realized_amp_ms"] = realized_samp / sample_rate * 1000.0
        # Total realized change over the transition ramp (or equal to the above
        # when transition_ms == 0).
        post_idx = min(T - 1, idx + transition_len)
        realized_total_samp = float(clipped[post_idx]) - pre_clipped
        ev["realized_total_amp_samples"] = realized_total_samp
        ev["realized_total_amp_ms"] = realized_total_samp / sample_rate * 1000.0

    meta_out: dict[str, Any] = {
        "events": meta_events,
        "transition_ms": float(cfg.transition_ms),
        "transition_samples": int(transition_len),
    }
    if drift_ppm is not None:
        meta_out["drift_ppm"] = drift_ppm
    if periodic_amp_ms is not None:
        meta_out["periodic_amp_ms"] = periodic_amp_ms
        meta_out["periodic_hz"] = periodic_hz
    return clipped.astype(np.float64), event_mask, drift_delta, meta_out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_time_delay_jitter(
    mic: np.ndarray | None,
    echo: np.ndarray,
    base_delay: float | np.ndarray,
    *,
    sample_rate: int,
    rng: np.random.Generator,
    config: TimeDelayJitterConfig | dict | None,
    max_total_delay_ms: float | None = None,
    edge_mode: str | None = None,
) -> JitterResult:
    """Apply configurable time-delay jitter to an AEC sample (echo_path only).

    Args:
        mic: optional microphone signal, shape ``(T,)``, that already contains
            ``echo``. When provided, the function returns ``mic_jittered`` by
            replacing the original echo component with the jittered echo. When
            ``None``, only ``echo_jittered`` is produced and ``mic_jittered`` is
            set to a copy of ``echo_jittered``. Passing ``None`` is the correct
            choice when the caller needs branch-aware mic reconstruction (e.g.
            ``echo_only`` vs ``prob_target_echo``).
        echo: echo stem, shape ``(T,)``.
        base_delay: current base delay in **samples** (scalar only).
            Positive means echo lags ref. Per-sample base delay arrays are not
            supported by the echo_path retimer.
        sample_rate: sample rate in Hz.
        rng: NumPy random generator.
        config: ``TimeDelayJitterConfig`` instance or dict. ``None`` means
            disabled.
        max_total_delay_ms: optional hard cap on the total delay in ms. When
            provided, ``delay_gt`` is clipped to ``[0, max_total_delay_ms]``.
        edge_mode: how to handle fractional reads outside ``[0, T-1]``. If
            ``None``, uses ``config.edge_mode``. ``constant`` zero-pads;
            ``nearest`` repeats the boundary sample.

    Returns:
        ``JitterResult`` with jittered waveforms, delay_gt (samples), masks and
        metadata. ``delay_gt`` is the single source of truth used by the shared
        core timebase primitive to produce both waveform and valid mask.
    """
    cfg = (
        config
        if isinstance(config, TimeDelayJitterConfig)
        else TimeDelayJitterConfig.from_dict(config)
    )
    if edge_mode is None:
        edge_mode = cfg.edge_mode

    # 在快路径和 RNG 消费前校验数值合同。
    if (
        isinstance(sample_rate, (bool, np.bool_))
        or not isinstance(sample_rate, (int, np.integer))
        or int(sample_rate) <= 0
    ):
        raise ValueError(f"sample_rate must be a positive integer, got {sample_rate!r}")
    sample_rate = int(sample_rate)

    if isinstance(base_delay, (bool, np.bool_)) or not np.isscalar(base_delay):
        raise ValueError(
            "apply_time_delay_jitter only supports a finite, non-negative scalar "
            "base_delay; per-sample arrays are not supported."
        )
    try:
        base_delay = float(base_delay)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"base_delay must be a finite, non-negative scalar, got {base_delay!r}"
        ) from exc
    if not math.isfinite(base_delay) or base_delay < 0.0:
        raise ValueError(
            f"base_delay must be a finite, non-negative scalar, got {base_delay!r}"
        )

    if max_total_delay_ms is not None:
        try:
            max_total_delay_ms = float(max_total_delay_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "max_total_delay_ms must be finite and non-negative, "
                f"got {max_total_delay_ms!r}"
            ) from exc
        if not math.isfinite(max_total_delay_ms) or max_total_delay_ms < 0.0:
            raise ValueError(
                "max_total_delay_ms must be finite and non-negative, "
                f"got {max_total_delay_ms!r}"
            )

    echo = np.asarray(echo, dtype=np.float32).ravel()
    T = echo.size
    if mic is not None:
        mic = np.asarray(mic, dtype=np.float32).ravel()
        if mic.size != T:
            raise ValueError(f"mic length {mic.size} != echo length {T}")

    # Short-circuit when disabled or gated out.
    if not cfg.enabled or float(rng.random()) >= cfg.prob:
        delay_gt_short = np.full(T, float(base_delay), dtype=np.float64)
        echo_copy = echo.copy()
        return JitterResult(
            mic_jittered=mic.copy() if mic is not None else echo_copy.copy(),
            echo_jittered=echo_copy,
            delay_gt=delay_gt_short,
            valid_mask=np.ones(T, dtype=bool),
            event_mask=np.zeros(T, dtype=bool),
            meta={"applied": False, "jitter_type": cfg.jitter_type},
        )

    # Sample the concrete jitter type for this sample when per-type probabilities are given.
    effective_jitter_type = cfg.jitter_type
    if cfg.jitter_type_probs:
        _types = list(cfg.jitter_type_probs.keys())
        _probs = [cfg.jitter_type_probs[t] for t in _types]
        effective_jitter_type = str(rng.choice(_types, p=_probs))

    delay_gt, event_mask, _drift_delta, jump_meta = _generate_delay_trajectory(
        T, base_delay, sample_rate, rng, cfg,
        jitter_type=effective_jitter_type,
        max_total_delay_ms=max_total_delay_ms,
    )

    base0 = float(base_delay)
    # 委托 immutable core plan：导出的 total-delay trajectory 同时决定 waveform
    # 与 valid_mask，避免在 AEC facade 中保留第二份 timebase mechanics。
    timebase_result = apply_delay_trajectory_plan(
        echo,
        DelayTrajectoryPlan(
            total_delay_samples=delay_gt,
            base_delay_samples=base0,
            edge_mode=edge_mode,
        ),
    )
    echo_j = timebase_result.waveform
    valid_mask = timebase_result.valid_mask

    if mic is not None:
        # Rebuild mic by replacing the original echo with the jittered echo.
        mic_j = mic.astype(np.float32) - echo.astype(np.float32) + echo_j
    else:
        # Caller handles branch-aware mixing; return echo_jittered as mic placeholder.
        mic_j = echo_j.copy()

    meta = {
        "applied": True,
        "jitter_type": cfg.jitter_type,
        "effective_jitter_type": effective_jitter_type,
        "jitter_type_probs": cfg.jitter_type_probs,
        "base_delay_samples": base0,
        "max_relative_jitter_ms": cfg.max_relative_jitter_ms,
        "num_events": len(jump_meta.get("events", [])),
        "edge_mode": edge_mode,
        **jump_meta,
    }

    return JitterResult(
        mic_jittered=mic_j,
        echo_jittered=echo_j,
        delay_gt=delay_gt.astype(np.float64),
        valid_mask=valid_mask,
        event_mask=event_mask,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_from_read_pos(read_pos: np.ndarray, T: int) -> np.ndarray:
    """A sample is valid when its read position lies inside the source array."""
    return (read_pos >= 0.0) & (read_pos <= float(T - 1)) & np.isfinite(read_pos)
