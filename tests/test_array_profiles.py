"""物理麦克风阵列 profile、配置解析与 provenance 合同测试。"""

import numpy as np
import pytest

from core.data.simulation.array_profiles import (
    ArrayTopology,
    build_geometry_metadata,
    get_array_profile,
    normalize_geometry_provenance,
    resolve_microphone_array,
)
from core.data.simulation.rir_config import OnlineRIRConfig
from core.data.simulation.rir_geometry import sample_rir_scene
from core.data.simulation_configs import SimulationConfigs


pytestmark = pytest.mark.unit


def test_canonical_profiles_are_task_neutral_physical_topologies():
    profile = get_array_profile("linear_4ch_40mm")
    assert profile.name == "linear_4ch_40mm"
    assert profile.num_channels == 4
    assert profile.topology.coordinate_system == "cartesian_xyz_m"
    assert profile.topology.spacing_m == pytest.approx(0.04)
    assert np.diff(np.asarray(profile.positions_m)[:, 0]).tolist() == pytest.approx([0.04] * 3)


def test_profile_and_reference_channel_resolve_to_rir_geometry():
    cfg = SimulationConfigs.from_dict(
        {
            "rir_backend": "online",
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 1,
                    },
        strict=True,
    )
    online = OnlineRIRConfig.from_simulation(cfg)
    plan = sample_rir_scene(online, 8000, 1, np.random.default_rng(3))
    assert plan["array_profile"] == "linear_2ch_80mm"
    assert plan["reference_channel"] == 1
    assert plan["array_topology"]["spacing_m"] == pytest.approx(0.08)
    assert np.asarray(plan["mic_loc"]).shape == (2, 3)


def test_custom_geometry_is_explicit_and_spacing_is_checked():
    resolved = resolve_microphone_array(
        custom_geometry={
            "positions_m": [[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]],
            "spacing_m": 0.04,
            "coordinate_system": "cartesian_xyz_m",
        },
        reference_channel=0,
    )
    assert resolved.profile.name == "custom"
    assert resolved.channels == 2
    with pytest.raises(ValueError, match="spacing_m"):
        ArrayTopology(
            ((-0.02, 0.0, 0.0), (0.02, 0.0, 0.0)),
            spacing_m=0.08,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_microphone_array(
            array_profile="linear_2ch_80mm",
            custom_geometry=[[0.0, 0.0, 0.0]],
        )


def test_external_names_are_not_array_profiles_but_are_valid_provenance():
    with pytest.raises(ValueError, match="unknown array_profile"):
        get_array_profile("FN-SSL")
    with pytest.raises(ValueError, match="unknown array_profile"):
        get_array_profile("NBSS")
    metadata = build_geometry_metadata(
        array_profile="linear_2ch_80mm",
        reference_channel=0,
        external={
            "external_repository": "Audio-WestlakeU/NBSS",
            "external_path": "config/array.yaml",
            "external_commit": "0123456789abcdef",
        },
        validation_recipe={"name": "geometry_parity", "command": "pytest"},
    )
    assert metadata["array_profile"] == "linear_2ch_80mm"
    assert metadata["provenance"]["external_repository"] == "Audio-WestlakeU/NBSS"
    assert metadata["provenance"]["external_path"] == "config/array.yaml"
    assert metadata["provenance"]["external_commit"] == "0123456789abcdef"


def test_geometry_provenance_rejects_short_external_key_aliases():
    for key in ("repository", "path", "commit"):
        with pytest.raises(ValueError, match="geometry_provenance has unknown keys"):
            normalize_geometry_provenance({key: "value"})


def test_spatial_config_keeps_c1_legacy_and_rejects_profile_channel_mismatch():
    legacy = SimulationConfigs.from_dict({"spatial_enabled": False}, strict=True)
    assert not hasattr(legacy, "num_channels") and legacy.reference_channel == 0
    with pytest.raises(ValueError, match="reference_channel must be explicit"):
        SimulationConfigs.from_dict(
            {
                "spatial_enabled": True,
                "array_profile": "linear_4ch_40mm",
            },
            strict=True,
        )


def test_supplied_rir_manifest_can_return_geometry_metadata(tmp_path):
    from core.data.spatial import load_offline_rir_bank_with_metadata

    rir = tmp_path / "rir.npy"
    np.save(rir, np.ones((2, 4), dtype=np.float32))
    manifest = tmp_path / "rir.json"
    manifest.write_text(
        '{"entries":{"source":"rir.npy"},"metadata":{"array_profile":"linear_2ch_80mm"}}',
        encoding="utf-8",
    )
    bank, metadata = load_offline_rir_bank_with_metadata(
        manifest,
        ("source",),
        geometry={"positions_m": [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]]},
    )
    assert bank.shape == (1, 2, 4)
    assert metadata == {"array_profile": "linear_2ch_80mm"}


def test_chime3_profile_matches_nbss_microphone_order_and_coordinates():
    profile = get_array_profile("chime3_6ch_rect_100x190mm")

    assert profile.channels == 6
    assert profile.topology.spacing_m is None
    assert profile.positions_m == (
        (-0.10, 0.095, 0.0),
        (0.0, 0.095, 0.0),
        (0.10, 0.095, 0.0),
        (-0.10, -0.095, 0.0),
        (0.0, -0.095, 0.0),
        (0.10, -0.095, 0.0),
    )


def test_array_rotation_is_seeded_and_bound_into_geometry_identity():
    cfg = SimulationConfigs.from_dict(
        {
            "rir_backend": "online",
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "prob_array_rotation": 1.0,
            "array_rotation_z_deg_range": [90.0, 90.0],
        },
        strict=True,
    )
    online = OnlineRIRConfig.from_simulation(cfg)
    first = sample_rir_scene(online, 8000, 1, np.random.default_rng(11))
    replay = sample_rir_scene(online, 8000, 1, np.random.default_rng(11))

    assert first["group_uid"] == replay["group_uid"]
    assert first["array_augmentation"]["rotation_z_rad"] == pytest.approx(np.pi / 2)
    np.testing.assert_allclose(
        np.asarray(first["microphone_offsets"]), [[0.0, -0.04, 0.0], [0.0, 0.04, 0.0]],
        atol=1e-12,
    )
    assert first["spacing_m"] == pytest.approx(0.08)


def test_microphone_jitter_is_seeded_and_removes_uniform_spacing_claim():
    cfg = SimulationConfigs.from_dict(
        {
            "rir_backend": "online",
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "prob_microphone_position_jitter": 1.0,
            "microphone_position_jitter_m_range": [0.005, 0.005],
        },
        strict=True,
    )
    online = OnlineRIRConfig.from_simulation(cfg)
    first = sample_rir_scene(online, 8000, 1, np.random.default_rng(12))
    replay = sample_rir_scene(online, 8000, 1, np.random.default_rng(12))

    assert first["group_uid"] == replay["group_uid"]
    assert first["array_augmentation"]["microphone_jitter_max_m"] == pytest.approx(0.005)
    assert first["spacing_m"] is None
    assert first["array_topology"]["spacing_m"] is None
