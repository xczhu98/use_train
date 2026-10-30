from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np


BoundaryReason = Literal[
    "whole",
    "stable_silence",
    "no_stable_silence",
    "tail_merged",
    "short_tail_unmerged",
    "skipped_overlong",
    "skipped_short",
]

_BOUNDARY_REASONS = {
    "whole",
    "stable_silence",
    "no_stable_silence",
    "tail_merged",
    "short_tail_unmerged",
    "skipped_overlong",
    "skipped_short",
}
_VAD_SAMPLE_RATE = 16_000
_STABLE_SILENCE_WORKING_SET_MAX_BYTES = 8 * 1024 * 1024
_STABLE_SILENCE_ALGORITHM = "float64_inplace_square_reduceat"
_FLOAT64_BYTES = np.dtype(np.float64).itemsize


@dataclass(frozen=True, slots=True)
class StableSilenceParameters:
    """Step1 稳定静音检测与 Step2 安全切点共用的固定参数。"""

    frame_ms: int = 10
    rms_threshold_dbfs: float = -45.0
    peak_threshold_dbfs: float = -30.0
    min_silence_ms: int = 200
    guard_ms: int = 100

    def __post_init__(self) -> None:
        for field in ("frame_ms", "min_silence_ms", "guard_ms"):
            if type(getattr(self, field)) is not int:
                raise TypeError(f"{field} must be an exact Python int")
        for field in ("rms_threshold_dbfs", "peak_threshold_dbfs"):
            value = getattr(self, field)
            if type(value) is not float:
                raise TypeError(f"{field} must be an exact Python float")
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
        if self.frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        if self.min_silence_ms <= 0:
            raise ValueError("min_silence_ms must be positive")
        if self.guard_ms < 0:
            raise ValueError("guard_ms must be nonnegative")


STEP1_STABLE_SILENCE_PARAMETERS = StableSilenceParameters()


def _stable_silence_block_frames(
    sample_rate: int,
    frame_ms: int,
    channels: int,
) -> int:
    """按最坏单帧 float64 工作区计算固定 byte budget 内的帧数。"""

    max_samples_per_frame = (sample_rate * frame_ms + 999) // 1000
    bytes_per_frame = max_samples_per_frame * channels * _FLOAT64_BYTES
    return max(1, _STABLE_SILENCE_WORKING_SET_MAX_BYTES // bytes_per_frame)


def _stable_silence_block_flags(
    audio: np.ndarray,
    frame_starts: np.ndarray,
    frame_ends: np.ndarray,
    rms_limit: float,
    peak_limit: float,
) -> np.ndarray:
    """使用一个音频块大小的 float64 工作区批量归约连续帧。"""

    flags = np.ones(frame_starts.size, dtype=np.bool_)
    nonempty = np.flatnonzero(frame_ends > frame_starts)
    if nonempty.size == 0:
        return flags

    sample_start = int(frame_starts[nonempty[0]])
    sample_end = int(frame_ends[nonempty[-1]])
    # copy=True 保证原地平方不会修改 decoded audio；这是唯一与音频块等大的
    # float64 scratch，reduceat 结果仅为 frame×channel 规模。
    values = np.array(
        audio[sample_start:sample_end], dtype=np.float64, order="C", copy=True
    )
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("audio must contain only finite samples")

    local_starts = frame_starts[nonempty] - sample_start
    lengths = frame_ends[nonempty] - frame_starts[nonempty]
    # 闭区间阈值允许归约后的单个 float64 舍入误差，不放宽实际 dB 门限。
    peak_limit_inclusive = np.nextafter(peak_limit, math.inf)
    rms_limit_inclusive = np.nextafter(rms_limit, math.inf)
    np.square(values, out=values)
    channel_square_max = np.maximum.reduceat(values, local_starts, axis=0)
    np.sqrt(channel_square_max, out=channel_square_max)
    peak_ok = np.all(channel_square_max <= peak_limit_inclusive, axis=1)
    del channel_square_max
    channel_square_sum = np.add.reduceat(values, local_starts, axis=0)
    channel_square_sum /= lengths[:, None]
    np.sqrt(channel_square_sum, out=channel_square_sum)
    rms_ok = np.all(channel_square_sum <= rms_limit_inclusive, axis=1)
    flags[nonempty] = peak_ok & rms_ok
    return flags


def detect_stable_silence_intervals(
    audio: np.ndarray,
    sample_rate: int,
    *,
    parameters: StableSilenceParameters = STEP1_STABLE_SILENCE_PARAMETERS,
) -> tuple[tuple[int, int], ...]:
    """在源采样域检测稳定静音；任一声道活跃时整帧视为非静音。"""

    if not isinstance(audio, np.ndarray) or audio.ndim != 2:
        raise TypeError("audio must be a two-dimensional NumPy array")
    if not np.issubdtype(audio.dtype, np.floating):
        raise TypeError("audio must use a real floating dtype")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive Python integer")
    if not isinstance(parameters, StableSilenceParameters):
        raise TypeError("parameters must be StableSilenceParameters")
    if audio.shape[1] == 0:
        raise ValueError("audio must contain at least one channel")
    if audio.shape[0] == 0:
        return ()

    rms_limit = 10.0 ** (parameters.rms_threshold_dbfs / 20.0)
    peak_limit = 10.0 ** (parameters.peak_threshold_dbfs / 20.0)
    frame_denominator = sample_rate * parameters.frame_ms
    frame_count = (audio.shape[0] * 1000 + frame_denominator - 1) // (
        frame_denominator
    )
    silent_frames = np.empty(frame_count, dtype=np.bool_)
    block_frames = _stable_silence_block_frames(
        sample_rate, parameters.frame_ms, audio.shape[1]
    )

    # 分块仅受固定 byte budget 和音频形状影响；边界仍逐项使用整数 floor 语义。
    for block_start in range(0, frame_count, block_frames):
        block_end = min(frame_count, block_start + block_frames)
        frame_indices = np.arange(block_start, block_end, dtype=np.int64)
        frame_starts = frame_indices * frame_denominator // 1000
        frame_ends = np.minimum(
            audio.shape[0], (frame_indices + 1) * frame_denominator // 1000
        )
        silent_frames[block_start:block_end] = _stable_silence_block_flags(
            audio, frame_starts, frame_ends, rms_limit, peak_limit
        )

    transitions = np.flatnonzero(
        np.concatenate(([False], silent_frames, [False]))[1:]
        != np.concatenate(([False], silent_frames, [False]))[:-1]
    )
    minimum_samples = (
        parameters.min_silence_ms * sample_rate + 999
    ) // 1000
    intervals: list[tuple[int, int]] = []
    for run_start, run_end in zip(transitions[::2], transitions[1::2]):
        start = int(run_start) * frame_denominator // 1000
        end = min(audio.shape[0], int(run_end) * frame_denominator // 1000)
        if end - start >= minimum_samples:
            intervals.append((start, end))
    return tuple(intervals)


@dataclass(frozen=True, slots=True)
class SegmentBoundary:
    start_sample: int
    end_sample: int
    boundary_reason: BoundaryReason

    def __post_init__(self) -> None:
        if type(self.start_sample) is not int or type(self.end_sample) is not int:
            raise TypeError("segment bounds must be exact Python ints")
        if self.start_sample < 0:
            raise ValueError("start_sample must be nonnegative")
        if self.end_sample <= self.start_sample:
            raise ValueError("end_sample must be greater than start_sample")
        if type(self.boundary_reason) is not str:
            raise TypeError("boundary_reason must be an exact Python str")
        if self.boundary_reason not in _BOUNDARY_REASONS:
            raise ValueError("boundary_reason is not supported")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "boundary_reason": self.boundary_reason,
        }


def _require_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact Python int")
    return value


def _validate_timeline(
    timeline: Sequence[Mapping[str, object]],
    vad_num_samples: int,
) -> list[tuple[int, int]]:
    if isinstance(timeline, (str, bytes, bytearray)) or not isinstance(
        timeline, Sequence
    ):
        raise TypeError("vad_timeline must be a non-string Sequence")

    intervals: list[tuple[int, int]] = []
    previous_end = 0
    expected_keys = {"start_sample", "end_sample", "mean_probability"}
    for index, entry in enumerate(timeline):
        if not isinstance(entry, Mapping):
            raise TypeError(f"vad_timeline[{index}] must be a Mapping")
        if set(entry.keys()) != expected_keys:
            raise ValueError(f"vad_timeline[{index}] must have exactly the required keys")

        start = _require_int("start_sample", entry["start_sample"])
        end = _require_int("end_sample", entry["end_sample"])
        probability = entry["mean_probability"]
        if isinstance(probability, bool) or not isinstance(probability, Real):
            raise TypeError("mean_probability must be a non-bool real number")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("mean_probability must be finite and in [0, 1]")
        if start < 0 or end <= start or end > vad_num_samples:
            raise ValueError("VAD intervals must be nonempty and within VAD bounds")
        if intervals and start < previous_end:
            raise ValueError("VAD intervals must be ordered and non-overlapping")

        intervals.append((start, end))
        previous_end = end
    return intervals


def _map_and_union_speech(
    intervals: Sequence[tuple[int, int]],
    source_num_samples: int,
    source_sample_rate: int,
) -> list[tuple[int, int]]:
    mapped: list[tuple[int, int]] = []
    for vad_start, vad_end in intervals:
        source_start = (vad_start * source_sample_rate) // _VAD_SAMPLE_RATE
        source_end = (
            vad_end * source_sample_rate + _VAD_SAMPLE_RATE - 1
        ) // _VAD_SAMPLE_RATE
        source_start = min(source_num_samples, source_start)
        source_end = min(source_num_samples, source_end)
        if source_start >= source_end:
            continue
        if mapped and source_start <= mapped[-1][1]:
            mapped[-1] = (mapped[-1][0], max(mapped[-1][1], source_end))
        else:
            mapped.append((source_start, source_end))
    return mapped


def _silence_gaps(
    speech: Sequence[tuple[int, int]], source_num_samples: int
) -> list[tuple[int, int]]:
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start, end in speech:
        if cursor < start:
            gaps.append((cursor, start))
        cursor = end
    if cursor < source_num_samples:
        gaps.append((cursor, source_num_samples))
    return gaps


def _validate_stable_silence_timeline(
    timeline: Sequence[Mapping[str, object]],
    source_num_samples: int,
) -> list[tuple[int, int]]:
    if isinstance(timeline, (str, bytes, bytearray)) or not isinstance(
        timeline, Sequence
    ):
        raise TypeError("stable_silence_timeline must be a non-string Sequence")

    intervals: list[tuple[int, int]] = []
    previous_end = 0
    expected_keys = {"start_sample", "end_sample"}
    for index, entry in enumerate(timeline):
        if not isinstance(entry, Mapping):
            raise TypeError(f"stable_silence_timeline[{index}] must be a Mapping")
        if set(entry.keys()) != expected_keys:
            raise ValueError(
                f"stable_silence_timeline[{index}] must have exactly the required keys"
            )
        start = _require_int("start_sample", entry["start_sample"])
        end = _require_int("end_sample", entry["end_sample"])
        if start < 0 or end <= start or end > source_num_samples:
            raise ValueError(
                "stable silence intervals must be nonempty and within source bounds"
            )
        if intervals and start < previous_end:
            raise ValueError(
                "stable silence intervals must be ordered and non-overlapping"
            )
        intervals.append((start, end))
        previous_end = end
    return intervals


def _safe_cut_intervals(
    vad_silence: Sequence[tuple[int, int]],
    stable_silence: Sequence[tuple[int, int]],
    guard_samples: int,
) -> list[tuple[int, int]]:
    safe: list[tuple[int, int]] = []
    vad_index = 0
    stable_index = 0
    while vad_index < len(vad_silence) and stable_index < len(stable_silence):
        intersection_start = max(
            vad_silence[vad_index][0], stable_silence[stable_index][0]
        )
        intersection_end = min(
            vad_silence[vad_index][1], stable_silence[stable_index][1]
        )
        start = intersection_start + guard_samples
        end = intersection_end - guard_samples
        if start <= end:
            safe.append((start, end))
        if vad_silence[vad_index][1] <= stable_silence[stable_index][1]:
            vad_index += 1
        else:
            stable_index += 1
    return safe


def plan_single_speaker_segments(
    *,
    source_num_samples: int,
    source_sample_rate: int,
    vad_timeline: Sequence[Mapping[str, object]],
    vad_timeline_sample_rate: int,
    vad_timeline_num_samples: int,
    stable_silence_timeline: Sequence[Mapping[str, object]],
    max_segment_duration_sec: float = 60.0,
    min_segment_duration_sec: float = 0.0,
) -> tuple[SegmentBoundary, ...]:
    source_num_samples = _require_int("source_num_samples", source_num_samples)
    source_sample_rate = _require_int("source_sample_rate", source_sample_rate)
    vad_timeline_sample_rate = _require_int(
        "vad_timeline_sample_rate", vad_timeline_sample_rate
    )
    vad_timeline_num_samples = _require_int(
        "vad_timeline_num_samples", vad_timeline_num_samples
    )

    if source_num_samples < 0:
        raise ValueError("source_num_samples must be nonnegative")
    if source_sample_rate <= 0:
        raise ValueError("source_sample_rate must be positive")
    # 超过上限的连续不可切段保留为 skipped_overlong 覆盖边界，由 stage 跳过、
    # 不产生 segment 行；planner 输出仍保持全源连续覆盖。
    if isinstance(max_segment_duration_sec, bool) or not isinstance(
        max_segment_duration_sec, Real
    ):
        raise TypeError("max_segment_duration_sec must be a real number")
    if not math.isfinite(max_segment_duration_sec) or max_segment_duration_sec <= 0:
        raise ValueError("max_segment_duration_sec must be positive and finite")
    # 低于 min_segment_duration_sec 的段保留为 skipped_short 覆盖边界，由 stage
    # 跳过、不产生 segment 行；0 表示不过滤（保持历史行为）。
    if isinstance(min_segment_duration_sec, bool) or not isinstance(
        min_segment_duration_sec, Real
    ):
        raise TypeError("min_segment_duration_sec must be a real number")
    if not math.isfinite(min_segment_duration_sec) or min_segment_duration_sec < 0:
        raise ValueError("min_segment_duration_sec must be nonnegative and finite")
    if vad_timeline_sample_rate != _VAD_SAMPLE_RATE:
        raise ValueError("vad_timeline_sample_rate must be exactly 16000")
    if vad_timeline_num_samples < 0:
        raise ValueError("vad_timeline_num_samples must be nonnegative")
    if isinstance(vad_timeline, (str, bytes, bytearray)) or not isinstance(
        vad_timeline, Sequence
    ):
        raise TypeError("vad_timeline must be a non-string Sequence")
    if isinstance(stable_silence_timeline, (str, bytes, bytearray)) or not isinstance(
        stable_silence_timeline, Sequence
    ):
        raise TypeError("stable_silence_timeline must be a non-string Sequence")

    if source_num_samples == 0:
        if (
            vad_timeline_num_samples != 0
            or len(vad_timeline) != 0
            or len(stable_silence_timeline) != 0
        ):
            raise ValueError("zero source requires zero lengths and empty timelines")
        return ()

    if vad_timeline_num_samples <= 0:
        raise ValueError("nonempty source requires positive VAD length")
    if source_sample_rate == _VAD_SAMPLE_RATE:
        if vad_timeline_num_samples != source_num_samples:
            raise ValueError("16 kHz source and VAD lengths must match")
    else:
        scaled = source_num_samples * _VAD_SAMPLE_RATE
        minimum = max(1, scaled // source_sample_rate)
        maximum = (scaled + source_sample_rate - 1) // source_sample_rate + 1
        if not minimum <= vad_timeline_num_samples <= maximum:
            raise ValueError("VAD length is outside the source-rate tolerance")

    vad_intervals = _validate_timeline(vad_timeline, vad_timeline_num_samples)
    stable_intervals = _validate_stable_silence_timeline(
        stable_silence_timeline, source_num_samples
    )
    speech = _map_and_union_speech(
        vad_intervals, source_num_samples, source_sample_rate
    )
    guard_samples = (
        STEP1_STABLE_SILENCE_PARAMETERS.guard_ms * source_sample_rate + 999
    ) // 1000
    safe_intervals = _safe_cut_intervals(
        _silence_gaps(speech, source_num_samples), stable_intervals, guard_samples
    )

    target_length = 10 * source_sample_rate
    search_radius = source_sample_rate
    cap_samples = int(max_segment_duration_sec * source_sample_rate)
    if cap_samples < 1:
        raise ValueError(
            "max_segment_duration_sec is too small for source_sample_rate"
        )
    min_samples = int(min_segment_duration_sec * source_sample_rate)
    if min_samples > cap_samples:
        raise ValueError(
            "min_segment_duration_sec must not exceed max_segment_duration_sec"
        )
    segments: list[SegmentBoundary] = []
    current_start = 0
    safe_cursor = 0

    def append_bounded(start: int, end: int, reason: BoundaryReason) -> None:
        # 长度超上限的段标记为 skipped_overlong（不产生输出行），
        # 低于 min_segment_duration_sec 的段标记为 skipped_short（不产生输出行），
        # 其余按原语义落地。
        if end - start > cap_samples:
            segments.append(SegmentBoundary(start, end, "skipped_overlong"))
        elif end - start < min_samples:
            segments.append(SegmentBoundary(start, end, "skipped_short"))
        else:
            segments.append(SegmentBoundary(start, end, reason))

    while source_num_samples - current_start > target_length:
        target = current_start + target_length
        search_start = max(current_start + 1, target - search_radius)
        search_end = min(source_num_samples - 1, target + search_radius)
        while (
            safe_cursor < len(safe_intervals)
            and safe_intervals[safe_cursor][1] < search_start
        ):
            safe_cursor += 1

        best: tuple[int, int] | None = None
        interval_index = safe_cursor
        while (
            interval_index < len(safe_intervals)
            and safe_intervals[interval_index][0] <= search_end
        ):
            safe_start, safe_end = safe_intervals[interval_index]
            intersection_start = max(safe_start, search_start)
            intersection_end = min(safe_end, search_end)
            if intersection_start <= intersection_end:
                candidate = min(max(target, intersection_start), intersection_end)
                rank = (abs(candidate - target), candidate)
                if best is None or rank < best:
                    best = rank
            interval_index += 1

        if best is None:
            forward_index = safe_cursor
            while (
                forward_index < len(safe_intervals)
                and safe_intervals[forward_index][1] <= search_end
            ):
                forward_index += 1
            if forward_index == len(safe_intervals):
                append_bounded(
                    current_start, source_num_samples, "no_stable_silence"
                )
                return tuple(segments)
            safe_start, _ = safe_intervals[forward_index]
            cut = max(safe_start, search_end + 1)
            if cut >= source_num_samples:
                append_bounded(
                    current_start, source_num_samples, "no_stable_silence"
                )
                return tuple(segments)
        else:
            cut = best[1]
        append_bounded(current_start, cut, "stable_silence")
        current_start = cut

    if not segments:
        if source_num_samples > cap_samples:
            return (SegmentBoundary(0, source_num_samples, "skipped_overlong"),)
        if source_num_samples < min_samples:
            return (SegmentBoundary(0, source_num_samples, "skipped_short"),)
        return (SegmentBoundary(0, source_num_samples, "whole"),)

    tail_length = source_num_samples - current_start
    if tail_length < 2 * source_sample_rate:
        previous = segments[-1]
        # skipped_overlong / skipped_short 的已跳过区间不参与合并，否则跳过段
        # 会被 tail_merged 重新变成输出行，且合并路径不复查 min_samples。
        if (
            previous.boundary_reason not in ("skipped_overlong", "skipped_short")
            and source_num_samples - previous.start_sample
            <= min(12 * source_sample_rate, cap_samples)
        ):
            segments[-1] = SegmentBoundary(
                previous.start_sample, source_num_samples, "tail_merged"
            )
        else:
            append_bounded(
                current_start, source_num_samples, "short_tail_unmerged"
            )
    else:
        append_bounded(current_start, source_num_samples, "whole")

    return tuple(segments)
