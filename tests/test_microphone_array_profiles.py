"""物理 microphone array profile 与配置投影测试。"""

import numpy as np
import pytest

from core.data.simulation.array_profiles import (
    ArrayTopology,
    CANONICAL_ARRAY_PROFILES,
    MicrophoneArrayProfile,
    build_geometry_metadata,
    get_array_profile,
    resolve_microphone_positions,
)
from core.data.simulation.rir_config import OnlineRIRConfig
from core.data.simulation.rir_geometry import sample_rir_scene
from core.data.simulation_configs import SimulationConfigs
from core.data.simulation_views import project_simulation_view


pytestmark = pytest.mark.unit


def test_canonical_profiles_are_task_neutral_and_physically_spaced():
    assert set(CANONICAL_ARRAY_PROFILES) == {
        "linear_2ch_80mm",
        "linear_4ch_40mm",
        "linear_6ch_40mm",
        "circular_8ch_r80mm",
        "chime3_6ch_rect_100x190mm",
    }
    chime3 = get_array_profile("chime3_6ch_rect_100x190mm")
    assert chime3.num_channels == 6
    assert chime3.spacing_m is None
    np.testing.assert_allclose(
        chime3.positions_m,
        [[-0.1, 0.095, 0.0], [0.0, 0.095, 0.0], [0.1, 0.095, 0.0],
         [-0.1, -0.095, 0.0], [0.0, -0.095, 0.0], [0.1, -0.095, 0.0]],
    )
    for name, expected_channels, expected_spacing in (
        ("linear_2ch_80mm", 2, 0.08),
        ("linear_4ch_40mm", 4, 0.04),
        ("linear_6ch_40mm", 6, 0.04),
    ):
        profile = get_array_profile(name)
        assert isinstance(profile, MicrophoneArrayProfile)
        assert profile.num_channels == expected_channels
        assert profile.spacing_m == pytest.approx(expected_spacing)
        assert profile.coordinate_system == "cartesian_xyz_m"
        assert "FN-SSL" not in profile.name and "NBSS" not in profile.name
        positions = np.asarray(profile.positions_m)
        np.testing.assert_allclose(np.diff(positions[:, 0]), expected_spacing)
        np.testing.assert_allclose(positions[:, 1:], 0.0)


def test_circular_profile_exposes_radius_chord_and_coplanar_geometry():
    profile = get_array_profile("circular_8ch_r80mm")
    assert profile.num_channels == 8
    assert profile.coordinate_system == "cartesian_xyz_m"
    positions = np.asarray(profile.positions_m)
    np.testing.assert_allclose(np.linalg.norm(positions, axis=1), 0.08, atol=1e-12)
    np.testing.assert_allclose(positions[:, 2], 0.0)
    chord = 2.0 * 0.08 * np.sin(np.pi / 8.0)
    adjacent = np.linalg.norm(np.diff(np.vstack((positions, positions[:1])), axis=0), axis=1)
    np.testing.assert_allclose(adjacent, chord, atol=1e-12)
    assert profile.spacing_m == pytest.approx(chord)


def test_topology_and_profile_validation_is_fail_closed():
    with pytest.raises(ValueError, match="adjacent microphone distance"):
        ArrayTopology(((0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.2, 0.0, 0.0)), spacing_m=0.04)
    with pytest.raises(ValueError, match="must match"):
        ArrayTopology(((-0.02, 0.0, 0.0), (0.02, 0.0, 0.0)), spacing_m=0.08)
    with pytest.raises(ValueError, match="unknown array_profile"):
        get_array_profile("FN-SSL")
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_microphone_positions(
            array_profile="linear_2ch_80mm",
            custom_geometry=[[0.0, 0.0, 0.0]],
        )


def test_config_resolves_profile_and_projects_geometry_provenance():
    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "reference_channel": 1,
            "array_profile": "linear_4ch_40mm",
            "geometry_provenance": {
                "external_repository": "https://example.invalid/reference",
                "external_path": "arrays/linear4.json",
                "external_commit": "abc123",
            },
            "validation_recipe": {"name": "array-room-smoke", "version": 1},
        },
        strict=True,
    )
    view = project_simulation_view(cfg, "ss")
    assert view.shared.spatial.array_profile == "linear_4ch_40mm"
    assert len(view.shared.spatial.microphone_positions) == 4
    assert view.shared.spatial.geometry_provenance["external_commit"] == "abc123"
    metadata = build_geometry_metadata(
        array_profile=cfg.array_profile,
        reference_channel=cfg.reference_channel,
        external=cfg.geometry_provenance,
        validation_recipe=cfg.validation_recipe,
    )
    assert metadata["array_profile"] == "linear_4ch_40mm"
    assert metadata["num_channels"] == 4
    assert metadata["provenance"]["external_repository"].startswith("https://")
    assert metadata["validation_recipe"]["name"] == "array-room-smoke"


def test_profile_requires_explicit_reference_for_even_array():
    with pytest.raises(ValueError, match="reference_channel"):
        SimulationConfigs.from_dict(
            {"spatial_enabled": True, "array_profile": "linear_2ch_80mm"},
            strict=True,
        )
    with pytest.raises(ValueError, match="reference_channel"):
        SimulationConfigs.from_dict(
            {
                "spatial_enabled": True,
                "array_profile": "linear_2ch_80mm",
                "reference_channel": 2,
            },
            strict=True,
        )


def test_online_rir_plan_carries_profile_geometry_and_reference():
    cfg = SimulationConfigs.from_dict(
        {
            "rir_backend": "online",
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 1,
            "geometry_provenance": {
                "external_path": "reference/array.json",
                "external_commit": "deadbeef",
            },
            "validation_recipe": {"name": "cpu-plan"},
        },
        strict=True,
    )
    online = OnlineRIRConfig.from_simulation(cfg)
    plan = sample_rir_scene(online, 8000, 1, np.random.default_rng(7))
    assert plan["array_profile"] == "linear_2ch_80mm"
    assert plan["reference_channel"] == 1
    assert plan["geometry_provenance"]["external_commit"] == "deadbeef"
    assert plan["validation_recipe"]["name"] == "cpu-plan"
    assert np.asarray(plan["microphone_offsets"]).shape == (2, 3)
