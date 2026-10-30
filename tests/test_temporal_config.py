"""共享 temporal canvas 值对象合同。"""

import pytest

from core.data.temporal_config import CanvasSpec


pytestmark = pytest.mark.unit


def test_canvas_spec_resolves_seconds_before_sample_cap():
    spec = CanvasSpec(
        max_duration_seconds=0.5,
        max_duration_samples=8000,
        repeat_to_max_duration=True,
        repeat_tile_silence_gap=False,
        repeat_tile_silence_gap_range=(0.02, 0.2),
    )
    assert spec.has_duration_cap
    assert spec.samples_for_sr(16000) == 8000


def test_canvas_spec_uses_sample_cap_without_seconds():
    spec = CanvasSpec(
        max_duration_seconds=None,
        max_duration_samples=4000,
        repeat_to_max_duration=False,
        repeat_tile_silence_gap=False,
        repeat_tile_silence_gap_range=(0.02, 0.2),
    )
    assert spec.samples_for_sr(16000) == 4000


def test_canvas_spec_uses_floor_for_fractional_seconds():
    """任务 Dataset 与共享画布均不能把 fractional seconds 四舍五入。"""
    spec = CanvasSpec(
        max_duration_seconds=0.1000625,
        max_duration_samples=0,
        repeat_to_max_duration=False,
        repeat_tile_silence_gap=False,
        repeat_tile_silence_gap_range=(0.02, 0.2),
    )
    assert spec.samples_for_sr(8000) == int(0.1000625 * 8000)


def test_canvas_spec_rejects_invalid_gap_range():
    with pytest.raises(ValueError, match="ordered finite non-negative"):
        CanvasSpec(
            max_duration_seconds=None,
            max_duration_samples=4000,
            repeat_to_max_duration=False,
            repeat_tile_silence_gap=False,
            repeat_tile_silence_gap_range=(0.2, 0.1),
        )
