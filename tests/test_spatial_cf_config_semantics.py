"""Spatial supplied-RIR 与 Layer C-F 配置语义测试。"""

from types import SimpleNamespace

import numpy as np
import pytest

from core.data.simulation.scene_renderer import render_speech_scene
from core.data.simulation.role_policy import AcousticActor, AcousticRole, RIRPolicy
from core.data.simulation_configs import SimulationConfigs
from core.data.simulation_views import project_simulation_view


pytestmark = pytest.mark.unit


def _dataset(cfg):
    return SimpleNamespace(
        sim_cfg=cfg,
        export_simulation_stems=False,
        _simulation_view=project_simulation_view(cfg, "ss"),
    )


def _actors(sources):
    roles = (
        AcousticRole.TARGET_SPEECH,
        *([AcousticRole.INTERFERENCE_SPEECH] * (len(sources) - 1)),
    )
    return tuple(
        AcousticActor(
            uid=f"source/{index}",
            role=roles[index],
            rir_policy=RIRPolicy.SPATIAL_SOURCE,
            waveform=source,
        )
        for index, source in enumerate(sources)
    )


def test_spatial_policies_are_projected_and_legacy_non_default_is_rejected():
    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "spatial_layer_d_policy": "per_channel",
            "spatial_layer_e_policy": "shared",
            "spatial_layer_f_policy": "per_channel",
        }
    )
    view = project_simulation_view(cfg, "ss")
    assert view.shared.spatial.spatial_layer_d_policy == "per_channel"
    assert view.shared.spatial.spatial_layer_e_policy == "shared"
    with pytest.raises(ValueError, match="requires spatial_enabled"):
        SimulationConfigs.from_dict({"spatial_layer_d_policy": "per_channel"})
    with pytest.raises(ValueError, match="spatial simulation keys are removed"):
        SimulationConfigs.from_dict(
            {"spatial_enabled": True, "array_profile": "linear_2ch_80mm", "reference_channel": 0, "spatial_geometry": "linear"},
            strict=True,
        )


def test_spatial_renderer_requires_the_configured_supplied_rir_seam():
    cfg = SimulationConfigs.from_dict(
        {"spatial_enabled": True, "array_profile": "linear_2ch_80mm", "reference_channel": 0}
    )
    dataset = _dataset(cfg)
    with pytest.raises(ValueError, match="explicit supplied"):
        render_speech_scene(
            dataset,
            _actors((np.ones(8, dtype=np.float32),)),
            8000,
            np.random.default_rng(0),
            sir_range=None,
            sir_mode="none",
            need_noise=False,
        )


def test_spatial_renderer_rejects_unresolved_point_source_background():
    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "rir_backend": "online",
            "prob_point_source_background": 0.5,
        }
    )
    with pytest.raises(ValueError, match="background point-source policy"):
        render_speech_scene(
            _dataset(cfg),
            _actors((np.ones(8, dtype=np.float32),)),
            8000,
            np.random.default_rng(0),
            sir_range=None,
            sir_mode="none",
            need_noise=True,
            spatial_rirs=np.ones((1, 2, 1), dtype=np.float32),
        )


def test_supplied_spatial_sir_uses_all_microphone_channels():
    cfg = SimulationConfigs.from_dict(
        {"spatial_enabled": True, "array_profile": "linear_2ch_80mm", "reference_channel": 0, "prob_reverberation": 1.0}
    )
    dataset = _dataset(cfg)
    sources = (
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )
    rirs = np.asarray(
        [
            [[1.0], [0.0]],
            [[0.0], [1.0]],
        ],
        dtype=np.float32,
    )
    scene = render_speech_scene(
        dataset,
        _actors(sources),
        8000,
        np.random.default_rng(1),
        sir_range=[6.0, 6.0],
        sir_mode="aggregate",
        need_noise=False,
        spatial_rirs=rirs,
    )
    interference = scene.bundle.require("source_full_rir/1")
    assert scene.facts["sir"][0]["realized_db"] == pytest.approx(6.0)
    assert np.max(np.abs(interference)) == pytest.approx(10.0 ** (-6.0 / 20.0))


def test_observation_geometry_accepts_absolute_room_coordinates():
    cfg = SimulationConfigs.from_dict(
        {"spatial_enabled": True, "array_profile": "linear_2ch_80mm", "reference_channel": 0}
    )
    dataset = _dataset(cfg)
    scene = render_speech_scene(
        dataset,
        _actors((np.ones(4, dtype=np.float32),)),
        8000,
        np.random.default_rng(3),
        sir_range=None,
        sir_mode="none",
        need_noise=False,
        spatial_rirs=np.ones((1, 2, 1), dtype=np.float32),
        observation_geometry={
            "positions_m": [[7.82, 7.9, 1.5], [7.9, 7.9, 1.5]],
            "coordinate_system": "cartesian_xyz_m",
        },
    )
    np.testing.assert_allclose(
        scene.facts["geometry"]["positions_m"],
        [[7.82, 7.9, 1.5], [7.9, 7.9, 1.5]],
    )


def test_supplied_spatial_source_volume_scales_full_early_and_mono_target():
    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "prob_reverberation": 1.0,
            "source_volume_normalization_enabled": True,
            "source_volume_db_range": [-20.0, -20.0],
            "source_volume_use_rms": False,
        }
    )
    dataset = _dataset(cfg)
    rirs = np.asarray([[[1.0], [0.5]]], dtype=np.float32)
    scene = render_speech_scene(
        dataset,
        _actors((np.ones(4, dtype=np.float32),)),
        8000,
        np.random.default_rng(2),
        sir_range=None,
        sir_mode="none",
        need_noise=False,
        spatial_rirs=rirs,
    )
    full = scene.bundle.require("source_full_rir/0")
    target = scene.bundle.require("target/0")
    assert full.shape == (2, 4)
    assert target.shape == (4,)
    np.testing.assert_allclose(full[0], target)
    np.testing.assert_allclose(full[1], target * 0.5)
    assert scene.facts["source_volume_calibration"][0]["stage"] == "post_rir_pre_noise_echo"
