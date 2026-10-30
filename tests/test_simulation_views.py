"""SimulationConfigs immutable typed view contracts。"""

from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from core.data.simulation_configs import SimulationConfigs
from core.data.simulation_views import (
    AECSimulationView,
    BackgroundSimulationView,
    FrozenMapping,
    GeometrySimulationView,
    SESimulationView,
    TSESimulationView,
    SSSimulationView,
    SSLSimulationView,
    TemporalSimulationView,
    TaskSemanticSimulationView,
    project_simulation_view,
    simulation_view_for,
)


def test_shared_view_preserves_resolved_values_and_freezes_nested_state():
    cfg = SimulationConfigs.from_dict(
        {
            "speed_rate_range": [0.9, 1.1],
            "wind_noise_config": {"threshold": [0.2, 0.3]},
            "augmentations": {
                "codec": {
                    "weight": 1.0,
                    "config": [{"format": "mp3", "qscale": [2, 7]}],
                }
            },
        }
    )

    view = project_simulation_view(cfg, "se")

    assert isinstance(view, SESimulationView)
    assert view.shared.layer_a.speed_rate_range == tuple(cfg.speed_rate_range)
    assert view.shared.acoustic.wind_noise_config["threshold"] == (0.2, 0.3)
    assert isinstance(view.shared.background, BackgroundSimulationView)
    assert isinstance(view.shared.temporal, TemporalSimulationView)
    assert view.shared.temporal.overlap_ratio_range == tuple(cfg.overlap_ratio_range)
    assert dict(view.shared.temporal.timeline_mode_weights) == cfg.timeline_mode_weights
    assert view.shared.background.wind_noise_config["threshold"] == (0.2, 0.3)
    assert view.shared.background.noise_mix_ratio_range == tuple(cfg.noise_mix_ratio_range)
    assert view.shared.post_mix.clip_variants == ("fixed_rail",)
    assert view.shared.post_mix.clip_variant_probs["fixed_rail"] == 1.0
    assert view.shared.post_mix.clip_pre_drive_db_range == (0.0, 12.0)
    assert isinstance(view.shared.channel.augmentations, FrozenMapping)
    codec = view.shared.channel.augmentations["codec"]
    assert codec["config"][0]["qscale"] == (2, 7)
    with pytest.raises(TypeError):
        codec["config"][0]["qscale"][0] = 3
    with pytest.raises(TypeError):
        view.shared.channel.augmentations["codec"] = {}
    with pytest.raises(FrozenInstanceError):
        view.shared.layer_a = view.shared.layer_a


def test_geometry_view_projects_validated_array_topology_without_mutating_legacy_spatial_view():
    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_4ch_40mm",
            "reference_channel": 2,
            "geometry_provenance": {"external_commit": "abc123"},
            "validation_recipe": {"name": "geometry-smoke", "version": 1},
        },
        strict=True,
    )

    view = project_simulation_view(cfg, "ss")
    geometry = view.shared.geometry

    assert isinstance(geometry, GeometrySimulationView)
    assert geometry.spatial_enabled is True
    assert geometry.geometry_kind == "profile"
    assert geometry.array_profile == "linear_4ch_40mm"
    assert geometry.channels == 4
    assert geometry.reference_channel == 2
    assert geometry.spacing_m == pytest.approx(0.04)
    assert geometry.coordinate_system == "cartesian_xyz_m"
    assert geometry.positions_m == view.shared.spatial.microphone_positions
    assert geometry.geometry_provenance["external_commit"] == "abc123"
    assert geometry.validation_recipe["version"] == 1
    assert geometry.resolved_array is not None
    assert geometry.resolved_array.channels == geometry.channels

    with pytest.raises(FrozenInstanceError):
        geometry.channels = 1
    with pytest.raises(TypeError):
        geometry.geometry_provenance["external_commit"] = "mutated"


def test_geometry_view_keeps_mono_configs_explicitly_geometry_free():
    geometry = project_simulation_view(SimulationConfigs.from_dict({}), "se").shared.geometry

    assert isinstance(geometry, GeometrySimulationView)
    assert geometry.spatial_enabled is False
    assert geometry.channels == 1
    assert geometry.positions_m is None
    assert geometry.geometry_kind is None
    assert geometry.resolved_array is None


def test_single_source_view_rejects_active_relative_timeline():
    cfg = SimulationConfigs.from_dict(
        {"prob_timeline": 1.0, "overlap_ratio_range": [0.2, 0.8]},
        strict=True,
    )
    with pytest.raises(ValueError, match="active temporal features"):
        project_simulation_view(cfg, "se")


def test_views_are_pickleable_for_spawn_dataloader_workers():
    view = project_simulation_view(SimulationConfigs.from_dict({}), "tse")

    restored = pickle.loads(pickle.dumps(view))

    assert restored == view
    assert restored.shared.channel.augmentations["codec"]["config"][0]["format"] == "mp3"


def test_task_views_expose_only_owned_task_slices():
    cfg = SimulationConfigs.from_dict(
        {
            "prob_echo_only": 0.2,
            "ser_range": [-5, 5],
            "num_interferers": {1: 0.4, 2: 0.6},
            "sir_range": [-10, 10],
        }
    )

    se_view = project_simulation_view(cfg, "se")
    aec_view = project_simulation_view(cfg, "aec")
    tse_view = project_simulation_view(cfg, "tse")

    assert isinstance(aec_view, AECSimulationView)
    assert aec_view.scenario.prob_echo_only == 0.2
    assert aec_view.echo.ser_range == (-5, 5)
    assert not hasattr(aec_view, "interference")
    assert isinstance(tse_view, TSESimulationView)
    assert dict(tse_view.scenario.num_interferers) == {1: 0.4, 2: 0.6}
    assert tse_view.scenario.sir_range == (-10, 10)
    assert not hasattr(tse_view, "enrollment")
    assert not hasattr(tse_view, "echo")
    assert not hasattr(se_view.scenario, "prob_echo_only")
    assert not hasattr(se_view, "enrollment")


def test_task_semantic_views_normalize_presence_without_changing_legacy_scenarios():
    cfg = SimulationConfigs.from_dict(
        {
            "prob_target_only": 0.10,
            "prob_noise_only": 0.20,
            "prob_target_noise": 0.70,
            "prob_echo_only": 0.05,
            "prob_noise_echo_only": 0.05,
            "prob_target_echo": 0.10,
            "prob_target_noise_echo": 0.10,
            "prob_interference_only": 0.05,
            "prob_interference_noise_only": 0.05,
            "prob_target_interference": 0.10,
            "prob_target_interference_noise": 0.10,
            "target_type": "dry",
        }
    )

    se_view = project_simulation_view(cfg, "se")
    aec_view = project_simulation_view(cfg, "aec")
    tse_view = project_simulation_view(cfg, "tse")
    ss_view = project_simulation_view(cfg, "ss")
    ssl_view = project_simulation_view(cfg, "ssl")

    assert isinstance(se_view.semantic, TaskSemanticSimulationView)
    assert dict(se_view.semantic.branch_probabilities) == {
        "target_only": 0.10,
        "noise_only": 0.20,
        "target_noise": 0.70,
    }
    assert se_view.semantic.target_presence_probabilities["present"] == pytest.approx(0.80)
    assert se_view.semantic.target_presence_probabilities["absent"] == pytest.approx(0.20)
    assert se_view.semantic.target_type == "dry"
    assert se_view.semantic.target_roles == ("target",)
    assert se_view.scenario.prob_target_noise == 0.70

    assert aec_view.semantic.target_presence_probabilities["present"] == pytest.approx(1.0)
    assert aec_view.semantic.target_presence_probabilities["absent"] == pytest.approx(0.30)
    assert aec_view.semantic.target_roles == ("nearend",)

    assert tse_view.semantic.target_presence_probabilities["present"] == pytest.approx(1.0)
    assert tse_view.semantic.target_presence_probabilities["absent"] == pytest.approx(0.30)
    assert tse_view.semantic.target_roles == ("target",)

    assert ss_view.semantic.branch_probabilities == FrozenMapping({})
    assert ss_view.semantic.target_presence_probabilities["present"] == pytest.approx(0.80)
    assert ss_view.semantic.target_presence_probabilities["absent"] == pytest.approx(0.20)
    assert ss_view.semantic.fixed_source_count is True

    assert ssl_view.semantic.branch_probabilities == FrozenMapping({})
    assert dict(ssl_view.semantic.target_presence_probabilities) == {
        "present": 1.0,
        "absent": 0.0,
    }
    assert ssl_view.semantic.fixed_source_count is True
    assert ssl_view.semantic.spatial_supervision is True


def test_task_semantic_view_is_optional_for_legacy_direct_construction():
    cfg = SimulationConfigs.from_dict({})
    shared = project_simulation_view(cfg, "se").shared
    legacy = SESimulationView(
        task="se",
        shared=shared,
        scenario=project_simulation_view(cfg, "se").scenario,
    )

    assert legacy.semantic is None


def test_ss_view_exposes_only_ss_layer_b_scenario():
    cfg = SimulationConfigs.from_dict(
        {"prob_target_only": 0.75, "prob_target_noise": 0.20, "prob_noise_only": 0.05}
    )
    view = project_simulation_view(cfg, "ss")
    assert isinstance(view, SSSimulationView)
    assert view.scenario.prob_target_only == 0.75
    assert view.scenario.prob_target_noise == 0.20
    assert view.scenario.prob_noise_only == 0.05


def test_simulation_view_for_accepts_owner_attached_ssl_view():
    owner = object.__new__(type("SSLDynamicMixingDataset", (), {}))
    owner._simulation_view = project_simulation_view(
        SimulationConfigs.from_dict({}), "ssl"
    )

    assert isinstance(owner._simulation_view, SSLSimulationView)
    assert simulation_view_for(owner, task="ssl") is owner._simulation_view


def test_projector_requires_validated_simulation_configs():
    with pytest.raises(TypeError, match="validated SimulationConfigs"):
        project_simulation_view(SimpleNamespace(prob_target_only=0.1), "se")
    with pytest.raises(ValueError, match="se/aec/tse"):
        project_simulation_view(SimulationConfigs.from_dict({}), "unknown")


def test_simulation_view_for_accepts_validated_simulation_configs():
    cfg = SimulationConfigs.from_dict({"prob_target_only": 0.25, "prob_noise_only": 0.5})

    view = simulation_view_for(cfg, task="se")

    assert view.scenario.prob_target_only == 0.25
    assert view.scenario.prob_noise_only == 0.5


def test_simulation_view_for_rejects_legacy_namespace_config():
    legacy = SimpleNamespace(
        prob_target_only=0.25,
        prob_noise_only=0.5,
        speed_rate_range=[0.95, 1.05],
    )

    with pytest.raises(TypeError, match="validated SimulationConfigs"):
        simulation_view_for(legacy, task="se")


def test_compatibility_adapter_prefers_owner_attached_view():
    owner = object.__new__(type("LegacyDataset", (), {}))
    owner._simulation_view = project_simulation_view(
        SimulationConfigs.from_dict({"prob_target_only": 0.3}), "tse"
    )

    assert simulation_view_for(owner, task="tse") is owner._simulation_view
    with pytest.raises(ValueError, match="task mismatch"):
        simulation_view_for(owner, task="se")


def test_spatial_background_unsupported_branches_fail_at_view_projection():
    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "prob_wind_noise_sidechain": 0.1,
        }
    )
    for task in ("se", "ss", "aec", "tse"):
        with pytest.raises(ValueError, match="prob_wind_noise_sidechain"):
            project_simulation_view(cfg, task)
    # SSL/DOA 是第一阶段允许接入 non-decomposable spatial sidechain 的路径。
    ssl_view = project_simulation_view(cfg, "ssl")
    assert ssl_view.shared.background.prob_wind_noise_sidechain == 0.1

    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "prob_colored_noise": 0.1,
        }
    )
    # C-3 彩色噪声 spatial 渲染已对全部五任务开放（capabilities 矩阵），
    # view 投影不再拒绝；渲染仍须走 BackgroundSpec 链。
    for task in ("se", "ss", "aec", "tse", "ssl"):
        view = project_simulation_view(cfg, task)
        assert view.shared.background.prob_colored_noise == 0.1


@pytest.mark.parametrize("task", ["se", "ss", "aec", "tse", "ssl"])
def test_spatial_wind_background_is_shared_across_task_views(task):
    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "prob_wind_noise_background": 0.5,
        },
        strict=True,
    )

    view = project_simulation_view(cfg, task)

    assert view.shared.background.prob_wind_noise_background == 0.5


pytestmark = [pytest.mark.unit]
