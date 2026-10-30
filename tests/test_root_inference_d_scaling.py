"""根 inference 与 spatial Layer D 的多通道边界回归。"""

from types import SimpleNamespace
import json

import numpy as np
import pytest

from core.data.post_mix_pipeline import PostMixState
from core.data.simulation.scene_renderer import SourceScenePostMixAdapter
from core.data.simulation_configs import SimulationConfigs
from core.data.simulation_views import project_simulation_view
from core.data.waveform_bundle import WaveformBundle

pytestmark = pytest.mark.unit


def _spatial_adapter(policy: str, mode: str):
    config = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 1,
            "spatial_layer_d_policy": policy,
            "peak_normalization_mode": mode,
            "peak_normalization_scale_range": [0.5, 0.5],
        }
    )
    dataset = SimpleNamespace(
        sim_cfg=config,
        export_simulation_stems=False,
        _simulation_view=project_simulation_view(config, "ss"),
    )
    return dataset, SourceScenePostMixAdapter(dataset, 8000, np.random.default_rng(3))


def _consistent_bundle(*, large: bool):
    if large:
        mixture = [[2.0, -2.0], [0.35, -0.35]]
        full = [[1.5, -1.5], [0.25, -0.25]]
        target = [1.5, -1.5]
    else:
        mixture = [[1.5, -1.5], [0.35, -0.35]]
        full = [[1.0, -1.0], [0.25, -0.25]]
        target = [1.0, -1.0]
    return WaveformBundle(
        {
            "mixture": np.asarray(mixture, dtype=np.float32),
            "background": np.asarray([[0.5, -0.5], [0.1, -0.1]], dtype=np.float32),
            "source_full_rir/0": np.asarray(full, dtype=np.float32),
            "target/0": np.asarray(target, dtype=np.float32),
        }
    )


@pytest.mark.parametrize("policy", ["shared", "per_channel"])
def test_spatial_d_normalize_follows_reference_target_gain(policy):
    dataset, adapter = _spatial_adapter(policy, "joint")
    state = PostMixState._from_frozen_waveforms(_consistent_bundle(large=False).tracks)
    normalized = adapter.apply_stage("D-normalize", state).state

    np.testing.assert_allclose(
        normalized.waveforms["mixture"],
        normalized.waveforms["background"]
        + normalized.waveforms["source_full_rir/0"],
    )
    facts = adapter.facts["normalize"]
    assert facts["reference_channel"] == dataset.sim_cfg.reference_channel
    assert facts["target_roles"] == ["target/0"]
    expected_target_scale = facts["scale"] if policy == "shared" else facts["scale"][1]
    assert facts["target_scale"] == pytest.approx(expected_target_scale)
    np.testing.assert_allclose(
        normalized.waveforms["target/0"],
        np.asarray([1.0, -1.0], dtype=np.float32) * expected_target_scale,
    )


@pytest.mark.parametrize("policy", ["shared", "per_channel"])
def test_spatial_d_guard_couples_full_tracks_and_mono_target(policy):
    _, adapter = _spatial_adapter(policy, "off")
    state = PostMixState._from_frozen_waveforms(_consistent_bundle(large=True).tracks)
    guarded = adapter.apply_stage("D-guard", state).state

    np.testing.assert_allclose(
        guarded.waveforms["mixture"],
        guarded.waveforms["background"]
        + guarded.waveforms["source_full_rir/0"],
    )
    assert np.max(np.abs(guarded.waveforms["source_full_rir/0"])) <= 0.99 + 1e-6
    facts = adapter.facts["D-guard"]
    assert facts["target_roles"] == ["target/0"]
    expected = 0.99 / 2.0 if policy == "shared" else 0.99 / 1.5
    np.testing.assert_allclose(
        guarded.waveforms["target/0"],
        np.asarray([1.5, -1.5], dtype=np.float32) * expected,
        atol=1e-6,
    )
    assert json.loads(json.dumps(adapter.facts))["D-guard"]["preserves_decomposition"] is True
