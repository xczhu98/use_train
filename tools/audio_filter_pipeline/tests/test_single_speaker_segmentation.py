from __future__ import annotations

import copy
import json
import random
from collections import UserDict, UserList
from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

import audio_filter_pipeline.processors.segmentation as segmentation_module
from audio_filter_pipeline.processors.segmentation import (
    SegmentBoundary,
    _map_and_union_speech,
    plan_single_speaker_segments,
)

pytestmark = pytest.mark.unit

VAD_RATE = 16_000


def _speech(start: int, end: int, probability: object = 0.75) -> dict[str, object]:
    return {
        "start_sample": start,
        "end_sample": end,
        "mean_probability": probability,
    }


def _stable(start: int, end: int) -> dict[str, int]:
    return {"start_sample": start, "end_sample": end}


def _plan(
    num_samples: int,
    *,
    sample_rate: int = VAD_RATE,
    timeline: object = (),
    stable_timeline: object | None = None,
    vad_num_samples: int | None = None,
    max_segment_duration_sec: float = 60.0,
    min_segment_duration_sec: float = 0.0,
):
    if stable_timeline is None:
        # 测试必须显式向 planner 提供稳定静音事实；默认 fixture 表示全源稳定。
        stable_timeline = [] if num_samples == 0 else [_stable(0, num_samples)]
    if vad_num_samples is None:
        vad_num_samples = num_samples if sample_rate == VAD_RATE else max(
            1, (num_samples * VAD_RATE) // sample_rate
        )
    return plan_single_speaker_segments(
        source_num_samples=num_samples,
        source_sample_rate=sample_rate,
        vad_timeline=timeline,  # type: ignore[arg-type]
        vad_timeline_sample_rate=VAD_RATE,
        vad_timeline_num_samples=vad_num_samples,
        stable_silence_timeline=stable_timeline,  # type: ignore[arg-type]
        max_segment_duration_sec=max_segment_duration_sec,
        min_segment_duration_sec=min_segment_duration_sec,
    )


def _rows(result: tuple[SegmentBoundary, ...]) -> list[tuple[int, int, str]]:
    return [
        (segment.start_sample, segment.end_sample, segment.boundary_reason)
        for segment in result
    ]


def _vad_length_bounds(source_num_samples: int, source_rate: int) -> tuple[int, int]:
    scaled = source_num_samples * VAD_RATE
    return max(1, scaled // source_rate), (scaled + source_rate - 1) // source_rate + 1


def _independent_reference(
    source_num_samples: int,
    source_rate: int,
    vad_timeline: list[dict[str, object]],
) -> list[tuple[int, int, str]]:
    """Small brute-force oracle; intentionally does not call production helpers."""
    if source_num_samples == 0:
        return []

    speech: list[tuple[int, int]] = []
    for row in vad_timeline:
        start_16k = int(row["start_sample"])
        end_16k = int(row["end_sample"])
        start = min(source_num_samples, (start_16k * source_rate) // VAD_RATE)
        end = min(
            source_num_samples,
            (end_16k * source_rate + VAD_RATE - 1) // VAD_RATE,
        )
        if start >= end:
            continue
        if speech and start <= speech[-1][1]:
            speech[-1] = (speech[-1][0], max(speech[-1][1], end))
        else:
            speech.append((start, end))

    gaps: list[tuple[int, int]] = []
    cursor = 0
    for speech_start, speech_end in speech:
        if cursor < speech_start:
            gaps.append((cursor, speech_start))
        cursor = speech_end
    if cursor < source_num_samples:
        gaps.append((cursor, source_num_samples))
    guard = (100 * source_rate + 999) // 1000
    safe = [(start + guard, end - guard) for start, end in gaps if start + guard <= end - guard]

    target_size = 10 * source_rate
    radius = source_rate
    planned: list[tuple[int, int, str]] = []
    current = 0
    while source_num_samples - current > target_size:
        target = current + target_size
        low = max(current + 1, target - radius)
        high = min(source_num_samples - 1, target + radius)
        candidates = [
            point
            for safe_start, safe_end in safe
            for point in range(max(safe_start, low), min(safe_end, high) + 1)
        ]
        if candidates:
            cut = min(candidates, key=lambda point: (abs(point - target), point))
        else:
            cut = next(
                (
                    max(safe_start, high + 1)
                    for safe_start, safe_end in safe
                    if safe_end > high and max(safe_start, high + 1) < source_num_samples
                ),
                None,
            )
            if cut is None:
                planned.append((current, source_num_samples, "no_stable_silence"))
                return planned
        planned.append((current, cut, "stable_silence"))
        current = cut

    if not planned:
        return [(0, source_num_samples, "whole")]

    tail_length = source_num_samples - current
    if tail_length < 2 * source_rate:
        previous_start, _, _ = planned[-1]
        if source_num_samples - previous_start <= 12 * source_rate:
            planned[-1] = (previous_start, source_num_samples, "tail_merged")
        else:
            planned.append((current, source_num_samples, "short_tail_unmerged"))
    else:
        planned.append((current, source_num_samples, "whole"))
    return planned


@pytest.mark.parametrize(
    ("num_samples", "want"),
    [
        (0, []),
        (1, [(0, 1, "whole")]),
        (10 * VAD_RATE - 1, [(0, 10 * VAD_RATE - 1, "whole")]),
        (10 * VAD_RATE, [(0, 10 * VAD_RATE, "whole")]),
    ],
)
def test_zero_tiny_and_at_most_target_sources_are_whole(num_samples, want):
    assert _rows(_plan(num_samples, vad_num_samples=num_samples)) == want


def test_segment_boundary_is_frozen_slotted_and_json_safe():
    boundary = SegmentBoundary(1, 2, "stable_silence")
    assert boundary.to_dict() == {
        "start_sample": 1,
        "end_sample": 2,
        "boundary_reason": "stable_silence",
    }
    assert json.loads(json.dumps(boundary.to_dict())) == boundary.to_dict()
    assert not hasattr(boundary, "__dict__")
    with pytest.raises(FrozenInstanceError):
        boundary.end_sample = 3  # type: ignore[misc]


class _StrSubclass(str):
    pass


@pytest.mark.parametrize(
    ("start", "end", "reason", "error"),
    [
        (True, 2, "whole", TypeError),
        (0, False, "whole", TypeError),
        (0.0, 2, "whole", TypeError),
        (0, 2.0, "whole", TypeError),
        (-1, 2, "whole", ValueError),
        (2, 2, "whole", ValueError),
        (3, 2, "whole", ValueError),
        (0, 2, 1, TypeError),
        (0, 2, _StrSubclass("whole"), TypeError),
        (0, 2, "hard_cut", ValueError),
        (0, 2, "unknown", ValueError),
    ],
)
def test_segment_boundary_rejects_invalid_direct_construction(
    start, end, reason, error
):
    with pytest.raises(error):
        SegmentBoundary(start, end, reason)


def test_full_speech_keeps_remaining_source_without_hard_cut():
    total = 21 * VAD_RATE + 1
    timeline = [_speech(0, total)]
    assert _rows(_plan(total, timeline=timeline)) == [
        (0, total, "no_stable_silence"),
    ]


@pytest.mark.parametrize(
    ("safe_start", "safe_end", "expected_cut"),
    [
        (10 * VAD_RATE - 100, 10 * VAD_RATE + 100, 10 * VAD_RATE),
        (10 * VAD_RATE - 200, 10 * VAD_RATE - 100, 10 * VAD_RATE - 100),
        (10 * VAD_RATE + 100, 10 * VAD_RATE + 200, 10 * VAD_RATE + 100),
        (9 * VAD_RATE - 50, 9 * VAD_RATE, 9 * VAD_RATE),
        (11 * VAD_RATE, 11 * VAD_RATE + 50, 11 * VAD_RATE),
    ],
)
def test_guarded_silence_at_before_after_and_radius_is_selected(
    safe_start, safe_end, expected_cut
):
    total = 24 * VAD_RATE
    guard = VAD_RATE // 10
    timeline = [
        _speech(0, safe_start - guard),
        _speech(safe_end + guard, total),
    ]
    result = _plan(total, timeline=timeline)
    assert _rows(result)[0] == (0, expected_cut, "stable_silence")


@pytest.mark.parametrize(
    ("safe_start", "safe_end", "expected_cut", "expected_reason"),
    [
        (9 * VAD_RATE - 51, 9 * VAD_RATE - 1, 24 * VAD_RATE,
         "no_stable_silence"),
        (11 * VAD_RATE + 1, 11 * VAD_RATE + 51, 11 * VAD_RATE + 1,
         "stable_silence"),
    ],
)
def test_no_window_candidate_searches_forward_without_hard_cut(
    safe_start, safe_end, expected_cut, expected_reason
):
    total = 24 * VAD_RATE
    guard = VAD_RATE // 10
    timeline = [
        _speech(0, safe_start - guard),
        _speech(safe_end + guard, total),
    ]
    assert _rows(_plan(total, timeline=timeline))[0] == (
        0, expected_cut, expected_reason
    )


def test_equal_distance_safe_intervals_choose_earlier_boundary():
    guard = VAD_RATE // 10
    before = 10 * VAD_RATE - 2_000
    after = 10 * VAD_RATE + 2_000
    total = before + 10 * VAD_RATE
    stable_timeline = [
        _stable(before - guard, before + guard),
        _stable(after - guard, after + guard),
    ]
    assert _rows(_plan(total, stable_timeline=stable_timeline))[0] == (
        0,
        before,
        "stable_silence",
    )


def test_empty_timeline_makes_entire_source_silence():
    total = 22 * VAD_RATE
    assert _rows(_plan(total)) == [
        (0, 10 * VAD_RATE, "stable_silence"),
        (10 * VAD_RATE, 20 * VAD_RATE, "stable_silence"),
        (20 * VAD_RATE, total, "whole"),
    ]


def test_each_target_is_rebased_from_previous_shifted_cut():
    total = 31 * VAD_RATE
    guard = VAD_RATE // 10
    first_cut = 10 * VAD_RATE + 500
    rebased_target = first_cut + 10 * VAD_RATE
    second_cut = rebased_target + 100
    stable_timeline = [
        _stable(first_cut - guard, first_cut + guard),
        _stable(second_cut - guard, second_cut + guard),
    ]
    rows = _rows(_plan(total, stable_timeline=stable_timeline))
    assert rows[:2] == [
        (0, first_cut, "stable_silence"),
        (first_cut, second_cut, "stable_silence"),
    ]


def test_planner_consumes_safe_intervals_with_monotonic_cursor(monkeypatch):
    interval_count = 2000
    intervals = [
        (index * 10 * VAD_RATE, index * 10 * VAD_RATE)
        for index in range(1, interval_count + 1)
    ]

    class CountingIntervals:
        def __init__(self, values):
            self.values = values
            self.accesses = 0

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            self.accesses += 1
            return self.values[index]

    counting = CountingIntervals(intervals)
    monkeypatch.setattr(
        segmentation_module, "_safe_cut_intervals", lambda *_args: counting
    )
    total = (interval_count * 10 + 2) * VAD_RATE

    rows = _rows(_plan(total))

    assert len(rows) == interval_count + 1
    assert counting.accesses < interval_count * 10


def test_touching_speech_intervals_do_not_create_silence_gap():
    total = 22 * VAD_RATE
    target = 10 * VAD_RATE
    timeline = [_speech(0, target), _speech(target, total)]
    assert _rows(_plan(total, timeline=timeline)) == [
        (0, total, "no_stable_silence")
    ]


def test_trailing_silence_guard_moves_cut_inside_silence():
    total = 23 * VAD_RATE
    speech_end = 10 * VAD_RATE + 123
    result = _plan(total, timeline=[_speech(0, speech_end)])
    assert _rows(result)[0] == (
        0, speech_end + VAD_RATE // 10, "stable_silence"
    )


@pytest.mark.parametrize(
    ("tail", "last_reason"),
    [
        (2 * VAD_RATE, "whole"),
        (2 * VAD_RATE - 1, "tail_merged"),
    ],
)
def test_two_second_tail_threshold_is_strict(tail, last_reason):
    cut = 10 * VAD_RATE
    total = cut + tail
    guard = VAD_RATE // 10
    rows = _rows(
        _plan(total, stable_timeline=[_stable(cut - guard, cut + guard)])
    )
    if tail == 2 * VAD_RATE:
        assert rows == [
            (0, cut, "stable_silence"),
            (cut, total, last_reason),
        ]
    else:
        assert rows == [(0, total, last_reason)]


@pytest.mark.parametrize(
    ("previous_length", "expected"),
    [
        (10 * VAD_RATE + 1, "tail_merged"),
        (10 * VAD_RATE + 2, "short_tail_unmerged"),
    ],
)
def test_twelve_second_merged_length_boundary(previous_length, expected):
    tail = 2 * VAD_RATE - 1
    total = previous_length + tail
    guard = VAD_RATE // 10
    rows = _rows(
        _plan(
            total,
            stable_timeline=[
                _stable(previous_length - guard, previous_length + guard)
            ],
        )
    )
    if expected == "tail_merged":
        assert rows == [(0, total, "tail_merged")]
        assert total == 12 * VAD_RATE
    else:
        assert rows == [
            (0, previous_length, "stable_silence"),
            (previous_length, total, "short_tail_unmerged"),
        ]
        assert total == 12 * VAD_RATE + 1


def test_boundary_reason_belongs_to_segment_ending_at_that_boundary():
    cut = 10 * VAD_RATE + 50
    total = cut + 10 * VAD_RATE
    guard = VAD_RATE // 10
    assert _rows(
        _plan(total, stable_timeline=[_stable(cut - guard, cut + guard)])
    ) == [
        (0, cut, "stable_silence"),
        (cut, total, "whole"),
    ]


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 22_050, 44_100, 48_000, 192_000])
def test_cross_sample_rates_cover_source_exactly(sample_rate):
    total = 23 * sample_rate + 7
    low, _ = _vad_length_bounds(total, sample_rate)
    result = _plan(total, sample_rate=sample_rate, vad_num_samples=low)
    rows = _rows(result)
    assert rows[0][0] == 0
    assert rows[-1][1] == total
    assert all(left[1] == right[0] for left, right in zip(rows, rows[1:]))
    assert all(start < end for start, end, _ in rows)


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        pytest.param((1, 2), (1, 3), id="floor-start"),
        pytest.param((2, 3), (2, 5), id="ceil-end"),
    ],
)
def test_source_mapping_uses_exact_floor_start_and_ceil_end(interval, expected):
    assert _map_and_union_speech([interval], 100, 22_050) == [expected]


def test_floor_start_and_ceil_end_mapping_are_conservative():
    sample_rate = 22_050
    total = 24 * sample_rate
    vad_total = (total * VAD_RATE) // sample_rate
    target = 10 * sample_rate
    guard = sample_rate // 10
    vad_start = ((target - guard - 1) * VAD_RATE + sample_rate - 1) // sample_rate
    vad_end = ((target + guard + 1) * VAD_RATE) // sample_rate + 1
    timeline = [_speech(0, vad_start), _speech(vad_end, vad_total)]
    mapped_left_end = (vad_start * sample_rate + VAD_RATE - 1) // VAD_RATE
    mapped_right_start = (vad_end * sample_rate) // VAD_RATE
    result = _plan(
        total,
        sample_rate=sample_rate,
        timeline=timeline,
        vad_num_samples=vad_total,
    )
    expected_cut = min(
        range(mapped_left_end + guard, mapped_right_start - guard + 1),
        key=lambda point: (abs(point - target), point),
    )
    assert _rows(result)[0] == (0, expected_cut, "stable_silence")


def test_mapped_touching_intervals_are_unioned():
    intervals = [(0, 20 * VAD_RATE), (20 * VAD_RATE, 44 * VAD_RATE)]
    assert _map_and_union_speech(intervals, 22 * VAD_RATE, 8_000) == [
        (0, 22 * VAD_RATE)
    ]


def test_mapped_rounding_overlap_and_touch_are_unioned_without_false_gap():
    sample_rate = 8_000
    total = 22 * sample_rate
    vad_total = total * 2
    target_vad = 20 * sample_rate
    # VAD-domain gap of one sample collapses after mapping to 8 kHz.
    timeline = [_speech(0, target_vad), _speech(target_vad + 1, vad_total)]
    assert _rows(
        _plan(
            total,
            sample_rate=sample_rate,
            timeline=timeline,
            vad_num_samples=vad_total,
        )
    ) == [(0, total, "no_stable_silence")]


@pytest.mark.parametrize("which_bound", ["lower", "upper"])
def test_backend_vad_length_plus_one_tolerance_is_inclusive(which_bound):
    source_rate = 44_100
    total = 11 * source_rate + 17
    low, high = _vad_length_bounds(total, source_rate)
    vad_length = low if which_bound == "lower" else high
    result = _plan(
        total,
        sample_rate=source_rate,
        vad_num_samples=vad_length,
    )
    assert result[-1].end_sample == total


def test_mapping_does_not_normalize_coordinates_by_reported_vad_total():
    source_rate = 48_000
    total = 23 * source_rate
    low, high = _vad_length_bounds(total, source_rate)
    assert high > low
    guard_16k = VAD_RATE // 10
    gap_start_16k = 10 * VAD_RATE - guard_16k
    gap_end_16k = 10 * VAD_RATE + guard_16k
    timeline = [_speech(0, gap_start_16k), _speech(gap_end_16k, low)]
    first_low = _rows(
        _plan(total, sample_rate=source_rate, timeline=timeline, vad_num_samples=low)
    )[0]
    first_high = _rows(
        _plan(total, sample_rate=source_rate, timeline=timeline, vad_num_samples=high)
    )[0]
    assert first_low == first_high == (0, 10 * source_rate, "stable_silence")


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("source_num_samples", True),
        ("source_num_samples", 1.0),
        ("source_num_samples", "1"),
        ("source_sample_rate", False),
        ("source_sample_rate", 16_000.0),
        ("vad_timeline_sample_rate", True),
        ("vad_timeline_sample_rate", 16_000.0),
        ("vad_timeline_num_samples", False),
        ("vad_timeline_num_samples", 1.0),
    ],
)
def test_scalar_fields_require_exact_python_int(field, bad):
    kwargs = dict(
        source_num_samples=1,
        source_sample_rate=VAD_RATE,
        vad_timeline=(),
        vad_timeline_sample_rate=VAD_RATE,
        vad_timeline_num_samples=1,
        stable_silence_timeline=(),
    )
    kwargs[field] = bad
    with pytest.raises(TypeError):
        plan_single_speaker_segments(**kwargs)


def test_stable_silence_timeline_has_no_implicit_default():
    with pytest.raises(TypeError, match="stable_silence_timeline"):
        plan_single_speaker_segments(
            source_num_samples=1,
            source_sample_rate=VAD_RATE,
            vad_timeline=(),
            vad_timeline_sample_rate=VAD_RATE,
            vad_timeline_num_samples=1,
        )


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("source_num_samples", -1),
        ("source_sample_rate", 0),
        ("source_sample_rate", -1),
        ("vad_timeline_sample_rate", 8_000),
        ("vad_timeline_sample_rate", 16_001),
        ("vad_timeline_num_samples", -1),
    ],
)
def test_scalar_ranges_and_fixed_vad_rate_are_enforced(field, bad):
    kwargs = dict(
        source_num_samples=1,
        source_sample_rate=VAD_RATE,
        vad_timeline=(),
        vad_timeline_sample_rate=VAD_RATE,
        vad_timeline_num_samples=1,
        stable_silence_timeline=(),
    )
    kwargs[field] = bad
    with pytest.raises(ValueError):
        plan_single_speaker_segments(**kwargs)


@pytest.mark.parametrize(
    ("timeline", "vad_length"),
    [
        ([_speech(0, 1)], 0),
        ((), 1),
    ],
)
def test_zero_source_requires_zero_vad_length_and_empty_timeline(timeline, vad_length):
    with pytest.raises(ValueError):
        _plan(0, timeline=timeline, vad_num_samples=vad_length)


def test_nonempty_source_requires_positive_vad_length():
    with pytest.raises(ValueError):
        _plan(1, vad_num_samples=0)


@pytest.mark.parametrize("vad_length", [159, 161])
def test_16k_source_requires_exact_vad_length(vad_length):
    with pytest.raises(ValueError):
        _plan(160, vad_num_samples=vad_length)


@pytest.mark.parametrize("side", ["below", "above"])
def test_non16k_vad_length_outside_tolerance_is_rejected(side):
    source_rate = 44_100
    total = 123_457
    low, high = _vad_length_bounds(total, source_rate)
    bad = low - 1 if side == "below" else high + 1
    with pytest.raises(ValueError):
        _plan(total, sample_rate=source_rate, vad_num_samples=bad)


@pytest.mark.parametrize(
    "timeline",
    [
        "not-a-sequence",
        b"bytes",
        bytearray(b"bytes"),
        {"start_sample": 0, "end_sample": 1, "mean_probability": 0.5},
        42,
        None,
    ],
)
def test_timeline_must_be_a_non_string_sequence(timeline):
    with pytest.raises(TypeError):
        _plan(1, timeline=timeline)


def test_sequence_and_mapping_protocol_implementations_are_accepted():
    timeline = UserList([UserDict(_speech(0, 1))])
    assert _rows(_plan(1, timeline=timeline)) == [(0, 1, "whole")]


@pytest.mark.parametrize(
    "entry",
    [
        1,
        (0, 1, 0.5),
        {"start_sample": 0, "end_sample": 1},
        {
            "start_sample": 0,
            "end_sample": 1,
            "mean_probability": 0.5,
            "extra": 1,
        },
        {"start_sample": 0, "end_sample": 1, 2: 0.5},
    ],
)
def test_timeline_entries_are_exact_key_mappings(entry):
    with pytest.raises((TypeError, ValueError)):
        _plan(1, timeline=[entry])


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (True, 1),
        (0, False),
        (0.0, 1),
        (0, 1.0),
        ("0", 1),
        (0, "1"),
    ],
)
def test_timeline_bounds_require_exact_ints(start, end):
    with pytest.raises(TypeError):
        _plan(1, timeline=[_speech(start, end)])


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1, 1), (0, 0), (1, 1), (1, 0), (0, 2), (2, 3)],
)
def test_timeline_intervals_are_nonempty_and_inside_vad_domain(start, end):
    with pytest.raises(ValueError):
        _plan(1, timeline=[_speech(start, end)])


@pytest.mark.parametrize(
    "probability",
    [
        True,
        False,
        "0.5",
        None,
        complex(0.5, 0),
        float("nan"),
        float("inf"),
        -float("inf"),
        -0.0001,
        1.0001,
    ],
)
def test_probability_must_be_finite_non_bool_real_in_closed_unit_interval(probability):
    with pytest.raises((TypeError, ValueError)):
        _plan(1, timeline=[_speech(0, 1, probability)])


@pytest.mark.parametrize("probability", [0, 1, 0.0, 1.0, Fraction(1, 3)])
def test_probability_closed_interval_accepts_real_numeric_types(probability):
    assert _rows(_plan(1, timeline=[_speech(0, 1, probability)])) == [
        (0, 1, "whole")
    ]


def test_timeline_must_be_ordered():
    with pytest.raises(ValueError):
        _plan(4, timeline=[_speech(2, 3), _speech(0, 1)])


def test_timeline_must_not_overlap_but_may_touch():
    with pytest.raises(ValueError):
        _plan(4, timeline=[_speech(0, 3), _speech(2, 4)])
    assert _rows(_plan(4, timeline=[_speech(0, 2), _speech(2, 4)])) == [
        (0, 4, "whole")
    ]


def test_valid_input_timeline_is_not_mutated():
    timeline = [_speech(0, 5, 0.25), _speech(7, 10, 0.9)]
    before = copy.deepcopy(timeline)
    _plan(10, timeline=timeline)
    assert timeline == before


def test_result_fields_are_builtin_int_and_str():
    result = _plan(1)
    assert type(result) is tuple
    assert type(result[0].start_sample) is int
    assert type(result[0].end_sample) is int
    assert type(result[0].boundary_reason) is str


def test_randomized_valid_timelines_match_independent_brute_force_oracle():
    rng = random.Random(0xD1)
    rates = [1, 2, 3, 7, 8_000, 16_000, 22_050, 44_100, 48_000, 192_000]

    for _ in range(300):
        source_rate = rng.choice(rates)
        source_num_samples = rng.randint(1, 35) * source_rate + rng.randrange(source_rate)
        low, high = _vad_length_bounds(source_num_samples, source_rate)
        vad_length = source_num_samples if source_rate == VAD_RATE else rng.randint(low, high)

        points = sorted(rng.sample(range(vad_length + 1), k=min(rng.randint(0, 8), vad_length + 1)))
        timeline: list[dict[str, object]] = []
        cursor = 0
        for point in points:
            if point > cursor and rng.random() < 0.65:
                timeline.append(_speech(cursor, point, rng.random()))
            cursor = point
        if cursor < vad_length and rng.random() < 0.65:
            timeline.append(_speech(cursor, vad_length, rng.random()))

        before = copy.deepcopy(timeline)
        result = _plan(
            source_num_samples,
            sample_rate=source_rate,
            timeline=timeline,
            vad_num_samples=vad_length,
        )
        second = _plan(
            source_num_samples,
            sample_rate=source_rate,
            timeline=timeline,
            vad_num_samples=vad_length,
        )
        rows = _rows(result)

        assert rows == _independent_reference(source_num_samples, source_rate, timeline)
        assert result == second
        assert timeline == before
        assert rows[0][0] == 0
        assert rows[-1][1] == source_num_samples
        assert all(start < end for start, end, _ in rows)
        assert all(left[1] == right[0] for left, right in zip(rows, rows[1:]))
        assert {reason for _, _, reason in rows} <= {
            "whole",
            "stable_silence",
            "no_stable_silence",
            "tail_merged",
            "short_tail_unmerged",
        }


def test_overlong_source_without_safe_silence_is_marked_skipped_overlong():
    num_samples = 100 * VAD_RATE
    result = _plan(
        num_samples,
        timeline=[_speech(0, num_samples)],
        stable_timeline=[],
    )
    # 整源无安全切点且超过默认 60s 上限：planner 仍保持连续覆盖，
    # 但唯一边界被标记为 skipped_overlong，由 stage 跳过、不产生行。
    assert _rows(result) == [(0, num_samples, "skipped_overlong")]


def test_max_segment_duration_cap_boundary_is_inclusive():
    at_cap = 60 * VAD_RATE
    assert _rows(_plan(
        at_cap, timeline=[_speech(0, at_cap)], stable_timeline=[],
    )) == [(0, at_cap, "no_stable_silence")]
    over_cap = at_cap + 1
    assert _rows(_plan(
        over_cap, timeline=[_speech(0, over_cap)], stable_timeline=[],
    )) == [(0, over_cap, "skipped_overlong")]


def test_custom_cap_keeps_overlong_default_stretch_emitted():
    num_samples = 100 * VAD_RATE
    result = _plan(
        num_samples,
        timeline=[_speech(0, num_samples)],
        stable_timeline=[],
        max_segment_duration_sec=200.0,
    )
    assert _rows(result) == [(0, num_samples, "no_stable_silence")]


def test_skip_resumes_planning_after_future_safe_cut():
    # 90s 源：前 70.1s 无安全切点（70s-71s 稳定静音扣除两侧 100ms guard），
    # 超长段被跳过；planner 从安全切点继续，余量 19.9s 无未来安全点 →
    # no_stable_silence。
    num_samples = 90 * VAD_RATE
    result = _plan(
        num_samples,
        timeline=[_speech(0, 70 * VAD_RATE), _speech(71 * VAD_RATE, num_samples)],
        stable_timeline=[_stable(70 * VAD_RATE, 71 * VAD_RATE)],
    )
    assert _rows(result) == [
        (0, 1_121_600, "skipped_overlong"),
        (1_121_600, num_samples, "no_stable_silence"),
    ]


def test_short_tail_after_skip_is_not_merged_into_skipped_stretch():
    # 跳过段之后不足 2s 的尾巴不得 tail_merged 进已跳过区间（否则跳过段
    # 会重新变成输出行）；按 short_tail_unmerged 独立落地。
    num_samples = 72 * VAD_RATE
    result = _plan(
        num_samples,
        timeline=[_speech(0, 70 * VAD_RATE), _speech(71 * VAD_RATE, num_samples)],
        stable_timeline=[_stable(70 * VAD_RATE, 71 * VAD_RATE)],
    )
    assert _rows(result) == [
        (0, 1_121_600, "skipped_overlong"),
        (1_121_600, num_samples, "short_tail_unmerged"),
    ]


def test_short_tail_is_not_merged_into_skipped_short_stretch():
    # skipped_short 段同样不得被 tail_merged 复活：10.7s 源 + min=11.0s，
    # 安全切点 10.2s 处前段标记 skipped_short，尾巴 0.5s；合并路径既会复活
    # 已跳过区间，又不复查 min_samples，会把低于 min 的段变成输出行。
    num_samples = 171_200
    timeline = [_speech(0, 161_600), _speech(164_800, num_samples)]
    stable_timeline = [_stable(161_600, 164_800)]
    assert _rows(_plan(
        num_samples,
        timeline=timeline,
        stable_timeline=stable_timeline,
        min_segment_duration_sec=11.0,
    )) == [
        (0, 163_200, "skipped_short"),
        (163_200, num_samples, "skipped_short"),
    ]


def test_short_source_exceeding_tiny_cap_is_marked_skipped_overlong():
    num_samples = 5 * VAD_RATE
    result = _plan(num_samples, max_segment_duration_sec=4.0)
    assert _rows(result) == [(0, num_samples, "skipped_overlong")]


def test_short_source_below_min_segment_duration_is_marked_skipped_short():
    num_samples = 5 * VAD_RATE
    result = _plan(num_samples, min_segment_duration_sec=10.0)
    # 整源低于 min：planner 仍保持连续覆盖，但唯一边界被标记为
    # skipped_short，由 stage 跳过、不产生行。
    assert _rows(result) == [(0, num_samples, "skipped_short")]


def test_min_segment_duration_default_keeps_short_source_whole():
    num_samples = 5 * VAD_RATE
    assert _rows(_plan(num_samples)) == [(0, num_samples, "whole")]


def test_min_segment_duration_boundary_is_exclusive():
    # 段长恰好等于 min 时保留（跳过条件是严格小于）。
    num_samples = 10 * VAD_RATE
    assert _rows(_plan(num_samples, min_segment_duration_sec=10.0)) == [
        (0, num_samples, "whole"),
    ]


def test_short_tail_below_min_segment_duration_is_marked_skipped_short():
    # 20.5s 源：前 18.8s 语音，18.8s–19.2s 静音（VAD ∩ 稳定静音），
    # 尾部 1.6s 语音；前段经 forward safe cut 落地 18.9s，尾部低于 2s 且
    # 合并会超过 12s 上限 → 默认 short_tail_unmerged；min=2.0s 时尾部
    # 改标记 skipped_short。
    num_samples = 328_000
    timeline = [_speech(0, 300_800), _speech(307_200, num_samples)]
    stable_timeline = [_stable(300_800, 307_200)]
    assert _rows(_plan(
        num_samples, timeline=timeline, stable_timeline=stable_timeline,
    )) == [
        (0, 302_400, "stable_silence"),
        (302_400, num_samples, "short_tail_unmerged"),
    ]
    assert _rows(_plan(
        num_samples, timeline=timeline, stable_timeline=stable_timeline,
        min_segment_duration_sec=2.0,
    )) == [
        (0, 302_400, "stable_silence"),
        (302_400, num_samples, "skipped_short"),
    ]


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_min_segment_duration_sec_must_be_nonnegative_finite(value):
    with pytest.raises(ValueError):
        _plan(1, min_segment_duration_sec=value)


@pytest.mark.parametrize("value", [True, "1"])
def test_min_segment_duration_sec_must_be_real_number(value):
    with pytest.raises(TypeError):
        _plan(1, min_segment_duration_sec=value)


def test_min_segment_duration_sec_must_not_exceed_max():
    with pytest.raises(ValueError):
        _plan(
            100 * VAD_RATE,
            max_segment_duration_sec=10.0,
            min_segment_duration_sec=20.0,
        )


@pytest.mark.parametrize("value", [0, -1.0, float("nan"), float("inf")])
def test_max_segment_duration_sec_must_be_positive_finite(value):
    with pytest.raises(ValueError):
        _plan(1, max_segment_duration_sec=value)


@pytest.mark.parametrize("value", [True, "60"])
def test_max_segment_duration_sec_must_be_real_number(value):
    with pytest.raises(TypeError):
        _plan(1, max_segment_duration_sec=value)


def test_max_segment_duration_sec_too_small_for_sample_rate():
    with pytest.raises(ValueError):
        _plan(1, sample_rate=192_000, max_segment_duration_sec=1e-9)
