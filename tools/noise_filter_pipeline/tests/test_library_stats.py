"""Tests for core/library_stats（LibraryStats 聚合口径）。"""
import os
import sys

import pytest

_TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from noise_filter_pipeline.core.library_stats import LibraryStats  # noqa: E402
from noise_filter_pipeline.utils.sanitize import sanitize_basename_for_rename  # noqa: E402

pytestmark = pytest.mark.unit


def test_stats_label_key_collapses_sanitize_variants():
    """人类标签按 sanitize key 聚合，非单射变体合并为一个类别。"""
    stats = LibraryStats()
    for display, duration in (("Inside, small room", 1.0), ("Inside small room", 2.0)):
        stats.add(
            "ds_c",
            sanitize_basename_for_rename(display, is_file=False),
            display,
            16000,
            duration,
        )

    stats_dict = stats.to_dict()
    assert stats_dict["by_label"]["Inside_small_room"]["chunks"] == 2
    assert stats_dict["by_label"]["Inside_small_room"]["display"] == "Inside, small room"
    assert stats_dict["label_count"] == 1
    assert stats_dict["total_chunks"] == 2
    assert stats_dict["total_duration_hours"] == round(3.0 / 3600.0, 2)


def test_stats_aggregates_dataset_and_sample_rate():
    stats = LibraryStats()
    stats.add("ds_a", "Music", "Music", 16000, 1.0)
    stats.add("ds_a", "Music", "Music", 44100, 2.0)
    stats.add("ds_b", "Vehicle", "Vehicle", None, None)

    stats_dict = stats.to_dict()
    assert stats_dict["dataset_count"] == 2
    assert stats_dict["by_dataset"]["ds_a"]["chunks"] == 2
    assert stats_dict["by_dataset"]["ds_a"]["sample_rates"] == {"16000": 1, "44100": 1}
    assert stats_dict["by_sample_rate"] == {"16000": 1, "44100": 1}
    # sr/duration 缺失的记录仍计入 chunk 数，但不计入采样率统计
    assert stats_dict["by_dataset"]["ds_b"]["chunks"] == 1
    assert stats_dict["by_dataset"]["ds_b"]["sample_rates"] == {}


@pytest.mark.parametrize(
    "duration",
    [float("nan"), float("inf"), float("-inf"), -0.1, True, "1.0"],
)
def test_stats_rejects_invalid_duration_without_partial_update(duration):
    stats = LibraryStats()

    with pytest.raises(ValueError, match="duration"):
        stats.add("ds", "Rain", "Rain", 16000, duration)

    assert stats.total_chunks == 0
    assert stats.total_duration_s == 0.0
    assert dict(stats.by_dataset) == {}
    assert dict(stats.by_sample_rate) == {}
    assert dict(stats.by_label) == {}


@pytest.mark.parametrize(
    ("field", "args"),
    [
        ("dataset_id", (None, "Rain", "Rain")),
        ("dataset_id", ("", "Rain", "Rain")),
        ("dataset_id", (1, "Rain", "Rain")),
        ("label_key", ("ds", "", "Rain")),
        ("label_key", ("ds", None, "Rain")),
        ("label_key", ("ds", 1, "Rain")),
        ("label_display", ("ds", "Rain", "")),
        ("label_display", ("ds", "Rain", None)),
        ("label_display", ("ds", "Rain", 1)),
    ],
)
def test_stats_rejects_invalid_identity_fields_without_partial_update(field, args):
    stats = LibraryStats()

    with pytest.raises(ValueError, match=field):
        stats.add(*args, 16000, 1.0)

    assert stats.total_chunks == 0
    assert stats.total_duration_s == 0.0
    assert dict(stats.by_dataset) == {}
    assert dict(stats.by_sample_rate) == {}
    assert dict(stats.by_label) == {}


@pytest.mark.parametrize("sr", [0, -1, True, 16000.0, "16000"])
def test_stats_rejects_invalid_sample_rate_without_partial_update(sr):
    stats = LibraryStats()

    with pytest.raises(ValueError, match="sr"):
        stats.add("ds", "Rain", "Rain", sr, 1.0)

    assert stats.total_chunks == 0
    assert stats.total_duration_s == 0.0
    assert dict(stats.by_dataset) == {}
    assert dict(stats.by_sample_rate) == {}
    assert dict(stats.by_label) == {}
