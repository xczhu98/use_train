"""方案 B 分层动态仿真配置的边界合同。"""

import pytest

from core.data.simulation_configs import SimulationConfigs
from core.data.simulation_schema import (
    normalize_config_simulation_sections,
    normalize_simulation_cfg,
)


pytestmark = pytest.mark.unit


def test_layered_sections_project_common_fields_to_runtime_fields():
    raw = {
        "speech": {
            "splice": {"enabled": False},
            "speed": {"prob": 0.25, "rate_range": [0.9, 1.1]},
        },
        "acoustic": {
            "rir": {"backend": "online", "reverberation_prob": 1.0},
        },
        "spatial": {
            "enabled": True,
            "geometry": {"profile": "linear_2ch_80mm", "reference_mic": 0},
            "coherence": {"policy": "habets_spherical"},
            "background": {
                "point_source_prob": 0.3,
                "wind_background_prob": 0.0,
                "wind_sidechain_prob": 0.0,
            },
        },
    }

    normalized = normalize_simulation_cfg(raw)
    assert normalized == {
        "speech_splice": False,
        "prob_speed_perturb": 0.25,
        "speed_rate_range": [0.9, 1.1],
        "rir_backend": "online",
        "prob_reverberation": 1.0,
        "spatial_enabled": True,
        "array_profile": "linear_2ch_80mm",
        "reference_channel": 0,
        "spatial_diffuse_coherence_model": "habets_spherical",
        "prob_point_source_background": 0.3,
        "prob_wind_noise_background": 0.0,
        "prob_wind_noise_sidechain": 0.0,
    }

    cfg = SimulationConfigs.from_dict(raw, strict=True)
    assert cfg.spatial_enabled is True
    assert cfg.prob_point_source_background == 0.3
    assert cfg.prob_wind_noise_sidechain == 0.0


def test_layered_sections_allow_canonical_layer_e_field_inside_post_mix():
    raw = {
        "post_mix": {
            "prob_clip": 0.2,
            "clip_variants": ["fixed_rail"],
            "clip_variant_probs": {"fixed_rail": 1.0},
        },
    }
    normalized = normalize_simulation_cfg(raw)
    assert normalized["prob_clip"] == 0.2
    assert normalized["clip_variants"] == ["fixed_rail"]
    assert normalized["clip_variant_probs"] == {"fixed_rail": 1.0}


def test_layered_sections_allow_mapping_aliases_and_task_extensions():
    normalized = normalize_simulation_cfg(
        {
            "scene": {
                "timeline": {
                    "mode_weights": {"head": 1.0},
                },
            },
            "spatial": {
                "geometry": {
                    "custom": {
                        "coordinate_system": "cartesian_xyz_m",
                        "positions_m": [[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]],
                    },
                },
            },
            "aec": {
                "aec_time_delay_jitter": {"enabled": False},
            },
            "tse": {
                "enrollment_augment": {"enabled": False},
            },
            "se": {
                "pase_preprocess": {"peak_normalization_mode": "single"},
            },
        }
    )
    assert normalized["timeline_mode_weights"] == {"head": 1.0}
    assert normalized["custom_geometry"]["positions_m"] == [
        [0.0, 0.0, 0.0],
        [0.08, 0.0, 0.0],
    ]
    assert normalized["aec_time_delay_jitter"] == {"enabled": False}
    assert normalized["enrollment_augment"] == {"enabled": False}
    assert normalized["pase_preprocess"] == {"peak_normalization_mode": "single"}


def test_layered_sections_do_not_require_version_key():
    assert normalize_simulation_cfg({"spatial": {"enabled": True}}) == {
        "spatial_enabled": True
    }


def test_layered_spatial_yaw_rad_alias_is_retired_fail_closed():
    with pytest.raises(ValueError, match="array_yaw_deg_range"):
        normalize_simulation_cfg(
            {
                "spatial": {
                    "enabled": True,
                    "augmentation": {
                        "array_yaw_prob": 1.0,
                        "array_yaw_rad_range": [0.0, 6.283185307179586],
                    },
                }
            }
        )


def test_layered_spatial_yaw_deg_alias_projects_to_canonical_field():
    normalized = normalize_simulation_cfg(
        {
            "spatial": {
                "enabled": True,
                "augmentation": {
                    "array_yaw_prob": 1.0,
                    "array_yaw_deg_range": [0.0, 360.0],
                },
            }
        }
    )
    assert normalized["prob_array_rotation"] == 1.0
    assert normalized["array_rotation_z_deg_range"] == [0.0, 360.0]


def test_layered_section_projection_follows_layer_order():
    normalized = normalize_simulation_cfg(
        {
            "device": {"enabled": False},
            "channel": {"enabled": False},
            "acoustic": {"rir": {"backend": "online"}},
            "speech": {"splice": {"enabled": False}},
        }
    )
    assert list(normalized) == [
        "speech_splice",
        "rir_backend",
        "prob_channel_enabled",
        "noisy_post_mix_degradations_enabled",
    ]


def test_layered_sections_reject_flat_nested_mix_and_duplicate_fields():
    with pytest.raises(ValueError, match="cannot mix sections with flat keys"):
        normalize_simulation_cfg(
            {"spatial_enabled": True, "spatial": {"enabled": True}}
        )

    with pytest.raises(ValueError, match="duplicate field 'spatial_enabled'"):
        normalize_simulation_cfg(
            {
                "spatial": {
                    "enabled": True,
                    "other": {"spatial_enabled": False},
                },
            }
        )


def test_layered_sections_reject_unknown_nested_key():
    with pytest.raises(ValueError, match="unknown key 'spatial.geometry.reference_microphone'"):
        normalize_simulation_cfg(
            {
                "spatial": {"geometry": {"reference_microphone": 0}},
            }
        )


def test_config_normalizer_updates_train_and_validation_in_place():
    config = {
        "data": {
            "train": {
                "simulation_cfg": {
                    "acoustic": {"rir": {"backend": "online"}},
                }
            },
            "validation": {
                "simulation_cfg": {
                    "spatial": {"enabled": True},
                }
            },
        }
    }
    assert normalize_config_simulation_sections(config) is config
    assert config["data"]["train"]["simulation_cfg"] == {"rir_backend": "online"}
    assert config["data"]["validation"]["simulation_cfg"] == {"spatial_enabled": True}
