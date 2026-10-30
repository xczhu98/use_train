"""公共 Layer D/E/F gate 与 deterministic safety guard 合同测试。"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import core.data.dynamic_dataset_base as dynamic_base
from core.data.dynamic_dataset_base import (
    BaseDynamicMixingDataset,
    LayerEOperationResult,
    LayerEResult,
    apply_deterministic_peak_guard,
)
from core.data.post_augment import apply_post_augmentations
from core.data.simulation_configs import SimulationConfigs


pytestmark = pytest.mark.unit


def test_peak_guard_exposes_roles_threshold_peak_and_scale_facts():
    waveforms = {
        "mic": np.array([2.0, -2.0], dtype=np.float32),
        "nearend": np.array([1.0, -1.0], dtype=np.float32),
        "farend": np.array([10.0, -10.0], dtype=np.float32),
    }

    result = apply_deterministic_peak_guard(
        waveforms,
        roles=("mic", "nearend"),
        threshold=0.5,
    )

    assert result.facts.to_metadata() == {
        "roles": ["mic", "nearend"],
        "threshold": 0.5,
        "peak": 2.0,
        "scale": 0.25,
        "applied": True,
    }
    np.testing.assert_array_equal(result.waveforms["mic"], [0.5, -0.5])
    np.testing.assert_array_equal(result.waveforms["nearend"], [0.25, -0.25])
    np.testing.assert_array_equal(result.waveforms["farend"], waveforms["farend"])


def test_peak_guard_float32_threshold_is_idempotent_and_reports_no_scale():
    threshold = 0.99
    representable_threshold = float(np.float32(threshold))
    waveform = np.array([representable_threshold, -representable_threshold], dtype=np.float32)

    result = apply_deterministic_peak_guard(
        {"mic": waveform},
        roles=("mic",),
        threshold=threshold,
    )

    assert result.facts.to_metadata() == {
        "roles": ["mic"],
        "threshold": representable_threshold,
        "peak": representable_threshold,
        "scale": 1.0,
        "applied": False,
    }
    np.testing.assert_array_equal(result.waveforms["mic"], waveform)


def test_peak_guard_rejects_nonfinite_input():
    with pytest.raises(ValueError, match="must be finite"):
        apply_deterministic_peak_guard(
            {"mic": np.array([np.nan], dtype=np.float32)},
            roles=("mic",),
        )


def test_skip_clip_blocks_only_layer_f_clipping_operation():
    cfg = SimulationConfigs.from_dict(
        {
            "noisy_post_mix_degradations_enabled": True,
            "prob_clip": 1.0,
            "clip_variants": ["fixed_rail"],
            "clip_variant_probs": {"fixed_rail": 1.0},
            "clip_pre_drive_db_range": [12.0, 12.0],
            "prob_dc_offset": 1.0,
            "dc_offset_range": [0.25, 0.25],
        },
        strict=True,
    )
    clipping_facts = {"stale": True}
    applied_ops = []

    actual = apply_post_augmentations(
        np.zeros(8, dtype=np.float32),
        16_000,
        cfg,
        skip_clip=True,
        rng=np.random.default_rng(3),
        applied_ops_out=applied_ops,
        clipping_facts_out=clipping_facts,
    )

    assert applied_ops == ["dc_offset"]
    assert clipping_facts == {}
    np.testing.assert_allclose(np.abs(actual), 0.25, rtol=0, atol=1e-7)


def test_layer_f_strict_mode_does_not_silently_sanitize_nonfinite(monkeypatch):
    import core.data.post_augment as post_augment

    cfg = SimulationConfigs.from_dict(
        {"noisy_post_mix_degradations_enabled": True, "prob_drc": 1.0},
        strict=True,
    )
    monkeypatch.setattr(
        post_augment,
        "drc",
        lambda *args, **kwargs: np.full_like(args[0], np.nan),
    )
    strict_value = apply_post_augmentations(
        np.ones(8, dtype=np.float32), 16_000, cfg,
        rng=np.random.default_rng(0),
    )
    lenient_value = apply_post_augmentations(
        np.ones(8, dtype=np.float32), 16_000, cfg,
        rng=np.random.default_rng(0), sanitize_nonfinite=True,
    )
    assert not np.isfinite(strict_value).all()
    assert np.isfinite(lenient_value).all()


def test_layer_f_lenient_sanitize_keeps_realized_facts(monkeypatch):
    import core.data.post_augment as post_augment

    cfg = SimulationConfigs.from_dict(
        {
            "noisy_post_mix_degradations_enabled": True,
            "prob_clip": 1.0,
            "clip_variants": ["fixed_rail"],
            "clip_variant_probs": {"fixed_rail": 1.0},
            "clip_pre_drive_db_range": [12.0, 12.0],
            "prob_post_eq": 1.0,
            "post_eq_db_range": [0.0, 0.0],
        },
        strict=True,
    )
    monkeypatch.setattr(
        post_augment,
        "post_eq",
        lambda *args, **kwargs: np.full_like(args[0], np.nan),
    )
    operations = []
    clipping_facts = {}
    value = apply_post_augmentations(
        np.full(8, 0.5, dtype=np.float32),
        16_000,
        cfg,
        rng=np.random.default_rng(0),
        applied_ops_out=operations,
        clipping_facts_out=clipping_facts,
        sanitize_nonfinite=True,
    )

    assert np.isfinite(value).all()
    assert operations == ["clip", "post_eq"]
    assert clipping_facts["variant"] == "fixed_rail"
    assert clipping_facts["clipped_sample_count"] == 8


@pytest.mark.parametrize("layer_e_clip_applied", [False, True])
def test_common_layer_ef_gate_uses_actual_layer_e_clipping(
    monkeypatch,
    layer_e_clip_applied,
):
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = SimpleNamespace()
    waveform = np.array([0.25, -0.25], dtype=np.float32)
    dataset._apply_channel_degradation_result = lambda *_args: LayerEResult(
        waveform=waveform,
        enabled=True,
        num_augmentations_drawn=1,
        selected=("clipping",),
        operations=(
            LayerEOperationResult(
                name="clipping",
                applied=layer_e_clip_applied,
                observation_changed=layer_e_clip_applied,
                reason=None if layer_e_clip_applied else "no_change",
            ),
        ),
        observation_changed=layer_e_clip_applied,
    )
    observed = {}

    def _post(noisy, sr, cfg, **kwargs):
        observed.update(kwargs)
        return noisy

    monkeypatch.setattr(dynamic_base, "apply_post_augmentations", _post)

    result = dataset._apply_layer_ef_result(
        waveform,
        16_000,
        len(waveform),
        np.random.default_rng(0),
        False,
    )

    assert observed["skip_clip"] is layer_e_clip_applied
    assert observed["skip_codec"] is False
    np.testing.assert_array_equal(result.waveform, waveform)


def test_layer_ef_exposes_only_current_layer_inputs():
    assert tuple(inspect.signature(BaseDynamicMixingDataset._apply_layer_ef).parameters) == (
        "self",
        "noisy_1d",
        "sr",
        "actual_len",
        "rng",
        "used_sidechain_wind",
    )
