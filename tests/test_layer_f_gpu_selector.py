"""CPU Layer-F selector、realized clipping facts 与 observation 合同测试。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.data.post_augment import apply_post_augmentations
from core.data.simulation_configs import SimulationConfigs

pytestmark = pytest.mark.unit


def _clip_cfg(**overrides) -> SimulationConfigs:
    values = {
        "noisy_post_mix_degradations_enabled": True,
        "prob_clip": 1.0,
        "clip_variants": ["sox"],
        "clip_variant_probs": {"sox": 1.0},
        "clip_sox_threshold_db_range": [-6.0, -6.0],
    }
    values.update(overrides)
    return SimulationConfigs.from_dict(values, strict=True)


def _fixed_rail_cfg(**overrides) -> SimulationConfigs:
    values = {
        "noisy_post_mix_degradations_enabled": True,
        "prob_clip": 1.0,
        "clip_variants": ["fixed_rail"],
        "clip_variant_probs": {"fixed_rail": 1.0},
        "clip_pre_drive_db_range": [0.0, 0.0],
    }
    values.update(overrides)
    return SimulationConfigs.from_dict(values, strict=True)


def test_explicit_clip_selector_canonical_form():
    cfg = SimulationConfigs.from_dict(
        {"clip_variants": ["hard"], "clip_variant_probs": {"hard": 1.0}},
        strict=True,
    )

    assert cfg.clip_variants == ["hard"]


def test_map_only_clip_variant_probs_fails_closed():
    with pytest.raises(ValueError, match="clip_variant_probs keys must exactly match"):
        SimulationConfigs.from_dict({"clip_variant_probs": {"sox": 1.0}}, strict=True)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"clip_variants": ["soft", "sox"], "clip_variant_probs": {"soft": 1.0}},
            "clip_variant_probs.*exactly match",
        ),
        (
            {"clip_variants": ["soft", "soft"], "clip_variant_probs": {"soft": 1.0}},
            "clip_variants.*duplicate",
        ),
        (
            {"clip_variants": ["unknown"], "clip_variant_probs": {"unknown": 1.0}},
            "clip_variants.*unsupported",
        ),
        (
            {"clip_variants": ["soft"], "clip_variant_probs": {"soft": math.nan}},
            "clip_variant_probs.*finite non-negative",
        ),
        (
            {"clip_variants": ["soft"], "clip_variant_probs": {"soft": True}},
            "clip_variant_probs.*finite non-negative",
        ),
        (
            {
                "clip_variants": ["soft", "sox"],
                "clip_variant_probs": {"soft": 1.1, "sox": -0.1},
            },
            "clip_variant_probs.*finite non-negative",
        ),
        (
            {
                "clip_variants": ["soft", "sox"],
                "clip_variant_probs": {"soft": 0.2, "sox": 0.2},
            },
            "clip_variant_probs.*sum to 1",
        ),
    ],
)
def test_clipping_selector_is_strictly_validated(overrides, match):
    with pytest.raises(ValueError, match=match):
        SimulationConfigs.from_dict(overrides, strict=True)


def test_layer_f_master_switch_blocks_cpu_post_mix_degradations():
    cfg = _clip_cfg(noisy_post_mix_degradations_enabled=False)
    waveform = np.array([0.5, -0.5], dtype=np.float32)

    actual = apply_post_augmentations(
        waveform.copy(), 16_000, cfg, rng=np.random.default_rng(9)
    )

    np.testing.assert_array_equal(actual, waveform)


def test_cpu_clipping_selector_honors_clip_variants_and_probabilities():
    cfg = _clip_cfg()
    waveform = np.array([-0.5, -0.1, 0.0, 0.1, 0.5], dtype=np.float32)
    threshold = np.float32(10.0 ** (-6.0 / 20.0))
    facts = {}
    applied_ops = []

    actual = apply_post_augmentations(
        waveform,
        16_000,
        cfg,
        rng=np.random.default_rng(3),
        applied_ops_out=applied_ops,
        clipping_facts_out=facts,
    )

    expected = threshold * np.tanh(waveform / threshold)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert applied_ops == ["clip"]
    assert facts["variant"] == "sox"
    assert facts["applied"] is True
    assert facts["changed_sample_count"] == 4


def test_fixed_rail_cpu_facts_use_preclip_waveform_and_include_tail_frame():
    drive_db = 6.020599913465065
    cfg = _fixed_rail_cfg(clip_pre_drive_db_range=[drive_db, drive_db])
    waveform = np.full(65, 0.25, dtype=np.float32)
    waveform[18:22] = 0.75
    waveform[64] = -0.75
    facts = {}

    actual = apply_post_augmentations(
        waveform.copy(),
        1_000,
        cfg,
        rng=np.random.default_rng(0),
        clipping_facts_out=facts,
    )

    expected = np.clip(waveform * 2.0, -1.0, 1.0)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    assert facts["changed_sample_count"] == 65
    assert facts["changed_sample_ratio"] == pytest.approx(1.0)
    assert facts["clipped_sample_count"] == 5
    assert facts["clipped_sample_ratio"] == pytest.approx(5 / 65)
    assert facts["clipped_frame_count"] == 3
    assert facts["clipped_frame_ratio"] == pytest.approx(3 / 4)
    assert facts["longest_clipped_run_samples"] == 4


def test_non_fixed_variants_expose_unavailable_rail_facts():
    cfg = _clip_cfg()
    waveform = np.array([-0.75, 0.25, 0.75], dtype=np.float32)
    facts = {}

    apply_post_augmentations(
        waveform.copy(),
        16_000,
        cfg,
        rng=np.random.default_rng(0),
        clipping_facts_out=facts,
    )

    unavailable_fields = (
        "clipped_sample_count",
        "clipped_sample_ratio",
        "clipped_frame_count",
        "clipped_frame_ratio",
        "longest_clipped_run_samples",
    )
    assert all(facts[field] is None for field in unavailable_fields)


def test_fixed_rail_zero_change_is_not_reported_as_applied():
    cfg = _fixed_rail_cfg()
    waveform = np.zeros(4, dtype=np.float32)
    facts = {}
    applied_ops = []

    actual = apply_post_augmentations(
        waveform.copy(),
        16_000,
        cfg,
        rng=np.random.default_rng(0),
        applied_ops_out=applied_ops,
        clipping_facts_out=facts,
    )

    np.testing.assert_array_equal(actual, waveform)
    assert facts["changed_sample_count"] == 0
    assert facts["clipped_sample_count"] == 0
    assert facts["clipped_frame_count"] == 0
    assert facts["longest_clipped_run_samples"] == 0
    assert facts["applied"] is False
    assert applied_ops == []


def test_clipping_facts_are_cleared_when_selector_does_not_fire():
    cfg = _fixed_rail_cfg(prob_clip=0.0)
    facts = {"variant": "stale", "applied": True}

    actual = apply_post_augmentations(
        np.zeros(4, dtype=np.float32),
        16_000,
        cfg,
        rng=np.random.default_rng(0),
        clipping_facts_out=facts,
    )

    np.testing.assert_array_equal(actual, np.zeros(4, dtype=np.float32))
    assert facts == {}


def test_layer_f_keeps_target_outside_noisy_observation_path():
    noisy = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
    target = np.array([-0.25, -0.1, 0.1, 0.25], dtype=np.float32)
    target_before = target.copy()

    apply_post_augmentations(
        noisy.copy(), 16_000, _clip_cfg(), rng=np.random.default_rng(0)
    )

    np.testing.assert_array_equal(target, target_before)
