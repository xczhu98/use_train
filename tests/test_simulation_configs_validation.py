"""Unit tests for SimulationConfigs validation invariants."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.data.simulation_configs import SimulationConfigs


class TestRangeValidation:
    """All *_range fields must be 2-element lists with lo <= hi."""

    def test_range_lo_must_be_le_hi(self):
        with pytest.raises(ValueError, match="lo <= hi"):
            SimulationConfigs.from_dict({"source_volume_db_range": [-20, -40]})

    def test_range_must_have_two_elements(self):
        with pytest.raises(ValueError, match="2-element"):
            SimulationConfigs.from_dict({"source_volume_db_range": [-40]})

    def test_valid_range_loads(self):
        cfg = SimulationConfigs.from_dict({"source_volume_db_range": [-40, -20]})
        assert cfg.source_volume_db_range == [-40, -20]

    def test_room_dimension_range_loads_as_three_ordered_ranges(self):
        cfg = SimulationConfigs.from_dict(
            {"gpurir_room_dim_range_m": [[3.0, 4.0], [4.0, 5.0], [2.5, 3.0]]},
            strict=True,
        )
        assert cfg.gpurir_room_dim_range_m == [[3.0, 4.0], [4.0, 5.0], [2.5, 3.0]]

    @pytest.mark.parametrize(
        "value",
        [
            [[3.0, 4.0], [4.0, 5.0]],
            [[3.0, 4.0], [4.0, 5.0], [2.5]],
            [3.0, 4.0, 2.5],
        ],
    )
    def test_room_dimension_range_requires_three_pairs(self, value):
        with pytest.raises(ValueError, match="gpurir_room_dim_range_m"):
            SimulationConfigs.from_dict(
                {"gpurir_room_dim_range_m": value}, strict=True
            )

    def test_clip_pre_drive_range_requires_ordered_endpoints(self):
        with pytest.raises(ValueError, match="clip_pre_drive_db_range.*lo <= hi"):
            SimulationConfigs.from_dict({"clip_pre_drive_db_range": [12.0, 0.0]})

    def test_none_range_is_allowed(self):
        cfg = SimulationConfigs.from_dict({"ser_range": None})
        assert cfg.ser_range is None

    @pytest.mark.parametrize("bool_value", [True, np.bool_(True)])
    def test_source_volume_range_rejects_bool_endpoint(self, bool_value):
        with pytest.raises(ValueError, match="source_volume_db_range.*numeric"):
            SimulationConfigs.from_dict(
                {"source_volume_db_range": [-40.0, bool_value]}
            )

    @pytest.mark.parametrize("field", ["snr_low", "snr_high"])
    @pytest.mark.parametrize("bool_value", [True, np.bool_(True)])
    def test_primary_snr_bounds_reject_bool(self, field, bool_value):
        with pytest.raises(ValueError, match=rf"{field}.*real number"):
            SimulationConfigs.from_dict({field: bool_value})

    @pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
    @pytest.mark.parametrize(
        "field",
        [
            "source_volume_db_range",
            "ser_range",
            "colored_noise_snr_range",
            "peak_normalization_scale_range",
            "clip_pre_drive_db_range",
        ],
    )
    def test_range_rejects_nonfinite_endpoints(self, field, nonfinite):
        with pytest.raises(ValueError, match=rf"{field}.*finite"):
            SimulationConfigs.from_dict({field: [0.0, nonfinite]})

    @pytest.mark.parametrize("field", ["snr_low", "snr_high"])
    @pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
    def test_primary_snr_bounds_reject_nonfinite_values(self, field, nonfinite):
        with pytest.raises(ValueError, match=rf"{field}.*finite"):
            SimulationConfigs.from_dict({field: nonfinite})

    @pytest.mark.parametrize(
        "field",
        [
            "threshold",
            "ratio",
            "attack",
            "release",
            "sc_gain",
            "clipping_threshold",
            "wind_noise_snr_range",
        ],
    )
    @pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
    def test_wind_ranges_reject_nonfinite_endpoints(self, field, nonfinite):
        with pytest.raises(ValueError, match=rf"wind_noise_config\.{field}.*finite"):
            SimulationConfigs.from_dict(
                {"wind_noise_config": {field: [0.0, nonfinite]}}
            )


_RETIRED_FAREND_VOLUME_KEYS = (
    "farend_random_volume",
    "farend_volume_db_range",
    "farend_use_rms",
    "farend_volume_distribution",
    "farend_volume_beta_a",
    "farend_volume_beta_b",
)


@pytest.mark.parametrize("key", _RETIRED_FAREND_VOLUME_KEYS)
def test_retired_farend_volume_keys_fail_fast_without_strict_mode(key):
    with pytest.raises(ValueError, match=rf"{key}.*far_db_range.*erl_range"):
        SimulationConfigs.from_dict({key: False})


class TestBetaValidation:
    """Beta shape parameters and distribution selector must be valid."""

    def test_beta_params_must_be_positive(self):
        with pytest.raises(ValueError, match="source_volume_beta_a.*positive"):
            SimulationConfigs.from_dict({"source_volume_beta_a": 0.0})

    @pytest.mark.parametrize(
        "field",
        [
            "source_volume_beta_a",
            "source_volume_beta_b",
            "erl_beta_a",
            "erl_beta_b",
            "silence_pad_beta_a",
            "silence_pad_beta_b",
        ],
    )
    @pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
    def test_all_beta_params_reject_nonfinite_values(self, field, nonfinite):
        with pytest.raises(ValueError, match=rf"{field}.*finite"):
            SimulationConfigs.from_dict({field: nonfinite})

    def test_unknown_distribution_rejected(self):
        with pytest.raises(ValueError, match="source_volume_distribution"):
            SimulationConfigs.from_dict({"source_volume_distribution": "gamma"})

    def test_valid_beta_config_loads(self):
        cfg = SimulationConfigs.from_dict(
            {
                "source_volume_distribution": "beta",
                "source_volume_beta_a": 1.5,
                "source_volume_beta_b": 1.5,
            }
        )
        assert cfg.source_volume_distribution == "beta"
        assert cfg.source_volume_beta_a == 1.5


def _bool_fields(config_cls=SimulationConfigs):
    """按最终 MRO 注解枚举严格 bool 字段，子类覆盖优先。"""
    from typing import get_type_hints

    return tuple(
        sorted(
            field
            for field, annotation in get_type_hints(config_cls).items()
            if annotation is bool
        )
    )


SIMULATION_BOOL_FIELDS = _bool_fields()


class TestStrictBooleanValidation:
    """所有 bool 注解字段只接受 Python bool，不依赖字段名白名单。"""

    def test_annotation_discovery_covers_switches_without_float_probabilities(self):
        assert {
            "use_high_pass",
            "reuse_rir",
            "rir_normalize",
            "reuse_noise",
        } <= set(SIMULATION_BOOL_FIELDS)
        assert "prob_reverberation" not in SIMULATION_BOOL_FIELDS

    @pytest.mark.parametrize("field", SIMULATION_BOOL_FIELDS)
    @pytest.mark.parametrize(
        "invalid",
        ["false", "true", 0, 1, 0.0, 1.0, math.nan, math.inf, np.bool_(True)],
    )
    def test_all_bool_annotated_fields_reject_non_bool_values(self, field, invalid):
        with pytest.raises(ValueError, match=rf"{field}.*bool"):
            SimulationConfigs.from_dict({field: invalid})

    @pytest.mark.parametrize("field", SIMULATION_BOOL_FIELDS)
    @pytest.mark.parametrize("valid", [False, True])
    def test_all_bool_annotated_fields_accept_python_bool(self, field, valid):
        values = {field: valid}
        if field == "spatial_enabled" and valid:
            values.update(
                {
                    "array_profile": "linear_2ch_80mm",
                    "reference_channel": 0,
                }
            )
        cfg = SimulationConfigs.from_dict(values)
        assert getattr(cfg, field) is valid

    def test_bool_annotation_discovery_respects_inheritance_and_overrides(self):
        class DerivedSimulationConfigs(SimulationConfigs):
            derived_switch: bool = False
            use_high_pass: float = 1.0
            prob_derived: float = 0.5

        with pytest.raises(ValueError, match=r"derived_switch.*bool"):
            DerivedSimulationConfigs.from_dict({"derived_switch": "false"})
        with pytest.raises(ValueError, match=r"reuse_noise.*bool"):
            DerivedSimulationConfigs.from_dict({"reuse_noise": 1})

        cfg = DerivedSimulationConfigs.from_dict(
            {"use_high_pass": 0.5, "prob_derived": 0.5}
        )
        assert cfg.use_high_pass == 0.5
        assert cfg.prob_derived == 0.5


class TestPostRirVolumeCompatibility:
    """post-RIR 音量语义沿用稳定字段名。"""

    def test_post_rir_volume_fields_keep_legacy_names(self):
        cfg = SimulationConfigs.from_dict(
            {
                "source_volume_normalization_enabled": True,
                "source_volume_db_range": [-38, -18],
                "source_volume_use_rms": True,
                "source_volume_distribution": "beta",
                "source_volume_beta_a": 2.0,
                "source_volume_beta_b": 3.0,
            }
        )

        assert cfg.source_volume_normalization_enabled is True
        assert cfg.source_volume_db_range == [-38, -18]
        assert cfg.source_volume_use_rms is True
        assert cfg.source_volume_distribution == "beta"
        assert cfg.source_volume_beta_a == 2.0
        assert cfg.source_volume_beta_b == 3.0

    @pytest.mark.parametrize(
        "field",
        ["source_volume_normalization_enabled", "source_volume_use_rms"],
    )
    @pytest.mark.parametrize("string_bool", ["false", "true"])
    def test_post_rir_volume_bool_fields_reject_strings(self, field, string_bool):
        with pytest.raises(ValueError, match=rf"{field}.*bool"):
            SimulationConfigs.from_dict({field: string_bool}, strict=True)


class TestErlValidation:
    """far_db_range / erl_range / erl_distribution / erl_beta_* 纳入统一校验。"""

    def test_valid_erl_config_loads(self):
        cfg = SimulationConfigs.from_dict(
            {
                "far_db_range": [-40, -12],
                "erl_range": [-20, 40],
                "erl_distribution": "beta",
                "erl_beta_a": 1.5,
                "erl_beta_b": 1.5,
            }
        )
        assert cfg.far_db_range == [-40, -12]
        assert cfg.erl_range == [-20, 40]
        assert cfg.erl_distribution == "beta"

    def test_far_db_range_lo_must_be_le_hi(self):
        with pytest.raises(ValueError, match="far_db_range.*lo <= hi"):
            SimulationConfigs.from_dict({"far_db_range": [-12, -40]})

    def test_erl_range_lo_must_be_le_hi(self):
        with pytest.raises(ValueError, match="erl_range.*lo <= hi"):
            SimulationConfigs.from_dict({"erl_range": [40, -20]})

    def test_unknown_erl_distribution_rejected(self):
        with pytest.raises(ValueError, match="erl_distribution"):
            SimulationConfigs.from_dict({"erl_distribution": "gamma"})

    def test_erl_beta_params_must_be_positive(self):
        with pytest.raises(ValueError, match="erl_beta_a.*positive"):
            SimulationConfigs.from_dict({"erl_beta_a": 0.0})


class TestProbabilityValidation:
    """prob_* / *_prob 标量字段必须在 [0, 1] 内，超界报错而非静默钳位。"""

    def test_moving_rir_probability_defaults_to_zero(self):
        cfg = SimulationConfigs.from_dict({})
        assert cfg.prob_moving_rir == 0.0

    def test_moving_rir_requires_online_backend(self):
        with pytest.raises(
            ValueError,
            match="moving RIR requires rir_backend='online'",
        ):
            SimulationConfigs.from_dict(
                {
                    "prob_moving_rir": 0.1,
                    "rir_backend": "offline",
                    "spatial_enabled": True,
                    "array_profile": "linear_2ch_80mm",
                    "reference_channel": 0,
                }
            )

    def test_online_mono_moving_rir_is_accepted(self):
        cfg = SimulationConfigs.from_dict(
            {"prob_moving_rir": 0.3, "rir_backend": "online"}
        )
        assert cfg.prob_moving_rir == 0.3
        assert cfg.spatial_enabled is False

    def test_online_spatial_moving_rir_is_accepted(self):
        cfg = SimulationConfigs.from_dict(
            {
                "prob_moving_rir": 1.0,
                "rir_backend": "online",
                "spatial_enabled": True,
                "array_profile": "linear_2ch_80mm",
                "reference_channel": 0,
            }
        )
        assert cfg.prob_moving_rir == 1.0

    def test_prob_above_one_rejected_with_field_name(self):
        with pytest.raises(ValueError, match=r"prob_clip.*1\.5"):
            SimulationConfigs.from_dict({"prob_clip": 1.5})

    def test_negative_prob_rejected(self):
        with pytest.raises(ValueError, match="silence_pad_prob"):
            SimulationConfigs.from_dict({"silence_pad_prob": -0.1})

    @pytest.mark.parametrize("prob_clip", [0.0, 1.0])
    def test_boundary_probs_load(self, prob_clip):
        cfg = SimulationConfigs.from_dict({"prob_clip": prob_clip})
        assert cfg.prob_clip == prob_clip

    @pytest.mark.parametrize(
        ("key", "value"),
        [("adc_hard_clip_prob", 0.1), ("adc_peak_guard_trigger", "all_stems")],
    )
    def test_removed_adc_config_keys_fail_fast(self, key, value):
        with pytest.raises(
            ValueError,
            match=(
                rf"retired Layer D ADC keys.*{key}.*Layer F.*prob_clip.*"
                r"fixed_rail.*clip_pre_drive_db_range"
            ),
        ):
            SimulationConfigs.from_dict({key: value})

    def test_bool_switch_not_treated_as_probability(self):
        cfg = SimulationConfigs.from_dict({"prob_channel_enabled": True})
        assert cfg.prob_channel_enabled is True

    @pytest.mark.parametrize("bool_value", [True, np.bool_(True)])
    def test_numeric_probability_rejects_bool(self, bool_value):
        with pytest.raises(ValueError, match="prob_reverberation.*numeric probability"):
            SimulationConfigs.from_dict({"prob_reverberation": bool_value})

    def test_retired_prob_wind_noise_requires_sidechain_key(self):
        with pytest.raises(
            ValueError,
            match=r"prob_wind_noise.*no longer supported.*prob_wind_noise_sidechain",
        ):
            SimulationConfigs.from_dict({"prob_wind_noise": 0.5})

    def test_retired_layer_a_requires_flat_simulation_cfg(self):
        with pytest.raises(
            ValueError,
            match=r"layer_a.*no longer supported.*top level of simulation_cfg",
        ):
            SimulationConfigs.from_dict({"layer_a": {"prob_speed_perturb": 0.5}})


class TestLayerEValidation:
    """Layer E 开关和抽样分布必须在加载期严格校验。"""

    @pytest.mark.parametrize(
        "field",
        ["prob_channel_enabled", "noisy_post_mix_degradations_enabled"],
    )
    @pytest.mark.parametrize(
        "invalid",
        ["false", "true", 0, 1, 2, math.nan, math.inf, -math.inf, np.bool_(True)],
    )
    def test_master_switches_require_real_bool(self, field, invalid):
        with pytest.raises(ValueError, match=rf"{field}.*bool"):
            SimulationConfigs.from_dict({field: invalid})

    def test_master_switches_accept_real_bool(self):
        cfg = SimulationConfigs.from_dict(
            {
                "prob_channel_enabled": False,
                "noisy_post_mix_degradations_enabled": True,
            }
        )
        assert cfg.prob_channel_enabled is False
        assert cfg.noisy_post_mix_degradations_enabled is True

    def test_num_augmentations_must_not_be_empty(self):
        with pytest.raises(ValueError, match="num_augmentations.*non-empty"):
            SimulationConfigs.from_dict({"num_augmentations": {}})

    @pytest.mark.parametrize("key", [True, -1, 1.5, "1"])
    def test_num_augmentations_keys_are_non_negative_ints(self, key):
        with pytest.raises(ValueError, match="num_augmentations.*non-negative int"):
            SimulationConfigs.from_dict({"num_augmentations": {key: 1.0}})

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -0.1, True])
    def test_num_augmentations_probabilities_are_finite_non_negative(self, value):
        with pytest.raises(ValueError, match="num_augmentations.*finite non-negative"):
            SimulationConfigs.from_dict({"num_augmentations": {0: value}})

    def test_num_augmentations_probabilities_must_sum_to_one(self):
        with pytest.raises(ValueError, match="num_augmentations.*sum to 1"):
            SimulationConfigs.from_dict({"num_augmentations": {0: 0.2, 1: 0.2}})

    def test_explicit_num_augmentations_replaces_default_distribution(self):
        cfg = SimulationConfigs.from_dict({"num_augmentations": {0: 1.0}})
        assert cfg.num_augmentations == {0: 1.0}

    @pytest.mark.parametrize("target_type", ["early", "dry"])
    def test_target_type_accepts_supported_selectors(self, target_type):
        cfg = SimulationConfigs.from_dict({"target_type": target_type})
        assert cfg.target_type == target_type

    @pytest.mark.parametrize(
        "augmentation_name",
        ["bandwidth_limitation", "clipping", "codec", "packet_loss"],
    )
    def test_augmentation_name_accepts_runtime_selectors(self, augmentation_name):
        cfg = SimulationConfigs.from_dict(
            {
                "augmentations_name": [augmentation_name],
                "num_augmentations": {0: 1.0},
            }
        )
        assert cfg.augmentations_name == [augmentation_name]

    def test_unreferenced_augmentation_mapping_extension_remains_compatible(self):
        cfg = SimulationConfigs.from_dict(
            {
                "augmentations": {
                    "legacy_extension": {"weight": 0.0, "option": "kept"},
                }
            }
        )
        assert cfg.augmentations["legacy_extension"] == {
            "weight": 0.0,
            "option": "kept",
        }

    @pytest.mark.parametrize("target_type", ["", "full", None, 1])
    def test_target_type_rejects_unsupported_selectors(self, target_type):
        with pytest.raises(ValueError, match=r"target_type.*dry.*early"):
            SimulationConfigs.from_dict({"target_type": target_type})

    @pytest.mark.parametrize(
        "augmentation_name", ["reverb", "legacy_extension"],
    )
    def test_augmentation_name_rejects_unsupported_runtime_selectors(
        self, augmentation_name
    ):
        with pytest.raises(ValueError, match="augmentations_name.*unsupported"):
            SimulationConfigs.from_dict({"augmentations_name": [augmentation_name]})

    def test_duplicate_augmentation_names_are_rejected(self):
        with pytest.raises(ValueError, match="augmentations_name.*duplicate.*clipping"):
            SimulationConfigs.from_dict(
                {"augmentations_name": ["clipping", "clipping"]}
            )

    def test_positive_augmentation_count_cannot_exceed_unique_names(self):
        with pytest.raises(
            ValueError,
            match="num_augmentations.*positive-probability count 2.*1 unique",
        ):
            SimulationConfigs.from_dict(
                {
                    "augmentations_name": ["bandwidth_limitation"],
                    "num_augmentations": {0: 0.5, 1: 0.25, 2: 0.25, 3: 0.0},
                }
            )

    def test_zero_probability_augmentation_count_may_exceed_unique_names(self):
        cfg = SimulationConfigs.from_dict(
            {
                "augmentations_name": ["bandwidth_limitation"],
                "num_augmentations": {0: 0.85, 1: 0.15, 2: 0.0, 3: 0.0},
            }
        )
        assert cfg.num_augmentations[3] == 0.0

    def test_runtime_rejects_sampled_count_above_available_names(self):
        from core.data.dynamic_dataset_base import BaseDynamicMixingDataset

        dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
        dataset.sim_cfg = SimulationConfigs.from_dict(
            {
                "augmentations_name": ["bandwidth_limitation"],
                "num_augmentations": {1: 1.0},
            }
        )
        dataset.sim_cfg.num_augmentations = {2: 1.0}
        dataset._aug_names = ["bandwidth_limitation"]
        dataset._aug_weights = np.array([1.0])

        with pytest.raises(ValueError, match="sampled num_augmentations count 2.*only 1"):
            dataset._apply_channel_degradation_result(
                np.zeros(16, dtype=np.float32),
                sr=16000,
                actual_len=16,
                rng=np.random.default_rng(0),
                used_sidechain_wind=False,
            )

    def test_referenced_augmentation_must_exist(self):
        with pytest.raises(ValueError, match="augmentations_name.*unknown"):
            SimulationConfigs.from_dict({"augmentations_name": ["unknown"]})

    def test_referenced_augmentation_weight_is_required(self):
        with pytest.raises(ValueError, match="bandwidth_limitation.*weight"):
            SimulationConfigs.from_dict(
                {
                    "augmentations_name": ["bandwidth_limitation"],
                    "augmentations": {
                        "bandwidth_limitation": {"resample_methods": "random"}
                    },
                }
            )

    @pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf, -0.1, True])
    def test_augmentation_weights_are_finite_non_negative(self, weight):
        with pytest.raises(ValueError, match="bandwidth_limitation.*weight"):
            SimulationConfigs.from_dict(
                {
                    "augmentations_name": ["bandwidth_limitation"],
                    "augmentations": {"bandwidth_limitation": {"weight": weight}},
                }
            )

    def test_referenced_augmentation_weights_cannot_all_be_zero(self):
        with pytest.raises(ValueError, match="augmentation weights.*greater than 0"):
            SimulationConfigs.from_dict(
                {
                    "augmentations_name": ["bandwidth_limitation", "clipping"],
                    "augmentations": {
                        "bandwidth_limitation": {"weight": 0.0},
                        "clipping": {"weight": 0.0},
                    },
                }
            )


class TestNumericAndDomainValidation:
    """数值配置拒绝字符串/布尔值，并在加载时校验消费侧领域约束。"""

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            ({"source_volume_db_range": ["-40", -20]}, "source_volume_db_range.*numeric"),
            ({"snr_low": "-5"}, "snr_low.*real number"),
            ({"wind_noise_config": {"threshold": ["0.1", 0.3]}}, "threshold.*numeric"),
            ({"wind_noise_config": {"clipping_chance": "0.5"}}, "clipping_chance.*real number"),
            ({"silence_pad_short_ratio": "0.5"}, "silence_pad_short_ratio.*real number"),
        ],
    )
    def test_numeric_strings_are_rejected(self, payload, match):
        with pytest.raises(ValueError, match=match):
            SimulationConfigs.from_dict(payload)

    @pytest.mark.parametrize(
        "field",
        [
            "source_volume_beta_a",
            "source_volume_beta_b",
            "erl_beta_a",
            "erl_beta_b",
            "silence_pad_beta_a",
            "silence_pad_beta_b",
        ],
    )
    def test_beta_shapes_reject_bool(self, field):
        with pytest.raises(ValueError, match=rf"{field}.*positive number"):
            SimulationConfigs.from_dict({field: True})

    def test_primary_snr_requires_ordered_bounds(self):
        with pytest.raises(ValueError, match="snr_low.*snr_high"):
            SimulationConfigs.from_dict({"snr_low": 10.0, "snr_high": -5.0})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("threshold", [0.0, 0.3]),
            ("threshold", [0.1, 1.1]),
            ("ratio", [0.5, 2.0]),
            ("attack", [0.0, 10.0]),
            ("release", [-1.0, 10.0]),
            ("sc_gain", [0.0, 1.0]),
            ("clipping_threshold", [-0.1, 0.8]),
            ("clipping_threshold", [0.8, 1.1]),
        ],
    )
    def test_wind_ranges_enforce_runtime_domains(self, field, value):
        with pytest.raises(ValueError, match=rf"wind_noise_config\.{field}"):
            SimulationConfigs.from_dict({"wind_noise_config": {field: value}})

    @pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf, True])
    def test_wind_clipping_chance_is_finite_probability(self, value):
        with pytest.raises(ValueError, match="clipping_chance"):
            SimulationConfigs.from_dict(
                {"wind_noise_config": {"clipping_chance": value}}
            )

    @pytest.mark.parametrize("field", ["silence_pad_position", "silence_pad_mode"])
    def test_silence_selectors_reject_unknown_values(self, field):
        with pytest.raises(ValueError, match=field):
            SimulationConfigs.from_dict({field: "unsupported"})

    @pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf, True])
    def test_silence_short_ratio_is_finite_probability(self, value):
        with pytest.raises(ValueError, match="silence_pad_short_ratio"):
            SimulationConfigs.from_dict({"silence_pad_short_ratio": value})

    @pytest.mark.parametrize(
        "field",
        ["silence_pad_total_range", "silence_pad_short_range", "silence_pad_long_range"],
    )
    def test_silence_duration_ranges_are_non_negative(self, field):
        with pytest.raises(ValueError, match=rf"{field}.*non-negative"):
            SimulationConfigs.from_dict({field: [-0.1, 0.5]})


class TestSchemaAndDiscreteDspValidation:
    @pytest.mark.parametrize("value", [2, True, 1.0, "2", 1, 3])
    def test_schema_version_is_retired(self, value):
        with pytest.raises(ValueError, match="schema_version is retired"):
            SimulationConfigs.from_dict({"schema_version": value})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("speech_splice_max_repeat_times", 1.0),
            ("band_reject_max_n", 0),
            ("spectral_holes_frame_length", True),
            ("spectral_holes_hop_length", 0),
            ("spectral_leakage_max_time_shift", -1),
            ("loudness_max_n_intervals", "5"),
            ("lowpass_order_range", [4.0, 20]),
            ("bit_crush_range", [0, 16]),
            ("spectral_holes_num_range", [1, 2.0]),
        ],
    )
    def test_discrete_dsp_parameters_require_exact_integer_domains(self, field, value):
        with pytest.raises(ValueError, match=field):
            SimulationConfigs.from_dict({field: value})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("colored_noise_types", ["white", "speech"]),
            ("codecs_types", ["mp3", "flac"]),
            ("webrtc_ns_levels", [0, 4]),
            ("webrtc_ns_levels", [True, 1]),
            ("spectral_leakage_window_lengths", [1024, 1024]),
        ],
    )
    def test_runtime_selector_lists_are_strict(self, field, value):
        with pytest.raises(ValueError, match=field):
            SimulationConfigs.from_dict({field: value})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("speech_splice_seconds", 0),
            ("early_rir_sec", -0.1),
            ("segment_sec", math.inf),
            ("speech_splice_gap_range", [-0.1, 0.2]),
            ("drc_attack_ms_range", [0.0, 5.0]),
        ],
    )
    def test_consumed_duration_parameters_enforce_domains(self, field, value):
        with pytest.raises(ValueError, match=field):
            SimulationConfigs.from_dict({field: value})

    @pytest.mark.parametrize("value", [0.02, 0.0, True])
    def test_removed_rir_overlap_sec_is_rejected_explicitly(self, value):
        with pytest.raises(
            ValueError,
            match=r"simulation keys are removed \(no compatibility\).*rir_overlap_sec",
        ):
            SimulationConfigs.from_dict({"rir_overlap_sec": value})


class TestLayerEConsumedParameterValidation:
    @pytest.mark.parametrize(
        ("name", "override", "match"),
        [
            (
                "clipping",
                {"clipping_min_quantile": [-0.1, 0.1]},
                "clipping_min_quantile",
            ),
            (
                "clipping",
                {"clipping_max_quantile": [0.9, 1.1]},
                "clipping_max_quantile",
            ),
            (
                "codec",
                {
                    "config": [
                        {"format": "flac", "encoder": None, "qscale": [1, 10]}
                    ]
                },
                "format",
            ),
            (
                "codec",
                {
                    "config": [
                        {
                            "format": "ogg",
                            "encoder": ["aac"],
                            "qscale": [1, 10],
                        }
                    ]
                },
                "encoder",
            ),
            (
                "codec",
                {
                    "config": [
                        {
                            "format": "ogg",
                            "encoder": ("vorbis",),
                            "qscale": [1, 10],
                        }
                    ]
                },
                "encoder",
            ),
            (
                "codec",
                {
                    "config": [
                        {
                            "format": "mp3",
                            "encoder": None,
                            "qscale": [1.0, 10],
                        }
                    ]
                },
                "qscale",
            ),
            ("packet_loss", {"packet_duration_ms": 20.0}, "packet_duration_ms"),
            (
                "packet_loss",
                {"max_continuous_packet_loss": 0},
                "max_continuous_packet_loss",
            ),
            ("packet_loss", {"packet_loss_rate": [-0.1, 0.2]}, "packet_loss_rate"),
        ],
    )
    def test_referenced_layer_e_parameters_fail_fast(self, name, override, match):
        base = {
            "clipping": {
                "weight": 1.0,
                "clipping_min_quantile": [0.0, 0.1],
                "clipping_max_quantile": [0.9, 1.0],
            },
            "codec": {
                "weight": 1.0,
                "config": [{"format": "mp3", "encoder": None, "qscale": [1, 10]}],
            },
            "packet_loss": {
                "weight": 1.0,
                "packet_duration_ms": 20,
                "max_continuous_packet_loss": 10,
                "packet_loss_rate": [0.05, 0.25],
            },
        }
        with pytest.raises(ValueError, match=match):
            SimulationConfigs.from_dict(
                {
                    "augmentations_name": [name],
                    "num_augmentations": {0: 1.0},
                    "augmentations": {name: {**base[name], **override}},
                }
            )

@pytest.mark.parametrize(
    "field",
    [
        "num_channels",
        "spatial_rir_policy",
        "spatial_rir_scp",
        "spatial_geometry",
        "microphone_array",
    ],
)
@pytest.mark.parametrize("strict", [False, True])
def test_removed_spatial_keys_fail_closed_even_when_parser_is_non_strict(field, strict):
    with pytest.raises(ValueError, match="spatial simulation keys are removed"):
        SimulationConfigs.from_dict({field: None}, strict=strict)


def test_spatial_nbss_fields_are_separate_from_existing_speech_speed_perturbation():
    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "spatial_diffuse_coherence_model": "habets_spherical",
            "prob_array_rotation": 1.0,
            "array_rotation_z_deg_range": [0.0, 360.0],
            "prob_microphone_position_jitter": 0.25,
            "microphone_position_jitter_m_range": [0.0, 0.005],
            "target_type": "rts",
            "rts_target_t60_sec": 0.15,
            "rts_time_after_max_sec": 0.002,
            "moving_source_speed_range_mps": [0.12, 0.4],
        },
        strict=True,
    )

    assert cfg.spatial_diffuse_coherence_model == "habets_spherical"
    assert cfg.array_rotation_z_deg_range == [0.0, 360.0]
    assert cfg.moving_source_speed_range_mps == [0.12, 0.4]
    assert cfg.prob_speed_perturb == 0.0
    assert cfg.speed_rate_range == [0.8, 1.2]


@pytest.mark.parametrize("strict", [False, True])
def test_array_rotation_rad_range_is_retired_fail_closed(strict):
    """旧 rad 键必须显式拒绝并给出迁移说明，不做静默单位换算。"""
    with pytest.raises(ValueError, match="array_rotation_z_deg_range"):
        SimulationConfigs.from_dict(
            {"array_rotation_z_rad_range": [0.0, 2 * math.pi]},
            strict=strict,
        )


def test_array_rotation_deg_range_defaults_to_full_circle():
    cfg = SimulationConfigs.from_dict({})
    assert cfg.array_rotation_z_deg_range == [0.0, 360.0]


def test_array_rotation_deg_range_converts_to_radians_at_rir_boundary():
    """配置侧 degree 在 OnlineRIRConfig 边界转为 rad；[90, 90] 即 π/2。"""
    from core.data.simulation.rir_config import OnlineRIRConfig

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
    assert online.array_rotation_z_rad_range == pytest.approx((math.pi / 2, math.pi / 2))


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"spatial_diffuse_coherence_model": "cylindrical"}, "spatial_diffuse_coherence_model"),
        ({"moving_source_speed_range_mps": [0.0, 0.4]}, "moving_source_speed_range_mps.*positive"),
        ({"microphone_position_jitter_m_range": [-0.001, 0.005]}, "microphone_position_jitter_m_range lower bound must be non-negative"),
        ({"prob_array_rotation": 1.1}, "prob_array_rotation.*probability"),
        ({"rts_target_t60_sec": 0.0}, "rts_target_t60_sec.*positive"),
        ({"target_type": "rts"}, "target_type='rts'.*spatial_enabled"),
        ({"prob_microphone_position_jitter": 0.1}, "jitter require spatial_enabled"),
    ],
)
def test_spatial_nbss_fields_fail_fast_on_invalid_semantics(overrides, match):
    spatial = bool(
        "target_type" not in overrides
        and "prob_microphone_position_jitter" not in overrides
    )
    values = dict(overrides)
    if spatial:
        values.update(
            {
                "spatial_enabled": True,
                "array_profile": "linear_2ch_80mm",
                "reference_channel": 0,
            }
        )
    with pytest.raises(ValueError, match=match):
        SimulationConfigs.from_dict(values, strict=True)


def test_background_point_source_defaults_to_zero_without_spatial():
    cfg = SimulationConfigs.from_dict({}, strict=True)
    assert cfg.prob_point_source_background == 0.0
    assert cfg.prob_wind_noise_background == 0.0


def test_background_point_source_legacy_key_is_rejected_without_aliasing():
    with pytest.raises(ValueError, match="renamed to prob_point_source_background"):
        SimulationConfigs.from_dict(
            {"prob_background_point_source": 0.5}, strict=True
        )


def test_wind_noise_background_loads_with_spatial():
    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "prob_wind_noise_background": 0.4,
        },
        strict=True,
    )
    assert cfg.prob_wind_noise_background == 0.4


def test_background_point_source_loads_with_spatial_online():
    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "rir_backend": "online",
            "prob_point_source_background": 0.3,
        },
        strict=True,
    )
    assert cfg.prob_point_source_background == 0.3


@pytest.mark.parametrize(
    "overrides,match",
    [
        (
            {"prob_point_source_background": 0.5},
            "prob_point_source_background requires spatial_enabled",
        ),
        (
            {
                "spatial_enabled": True,
                "array_profile": "linear_2ch_80mm",
                "reference_channel": 0,
                "prob_point_source_background": 0.5,
            },
            "prob_point_source_background requires rir_backend='online'",
        ),
        (
            {
                "spatial_enabled": True,
                "array_profile": "linear_2ch_80mm",
                "reference_channel": 0,
                "rir_backend": "online",
                "prob_point_source_background": 1.5,
            },
            "prob_point_source_background.*probability",
        ),
        (
            {"prob_wind_noise_background": 0.5},
            "prob_wind_noise_background requires spatial_enabled",
        ),
        (
            {
                "spatial_enabled": True,
                "array_profile": "linear_2ch_80mm",
                "reference_channel": 0,
                "prob_wind_noise_background": 1.5,
            },
            "prob_wind_noise_background.*probability",
        ),
    ],
)
def test_background_point_source_fail_fast_on_invalid_semantics(overrides, match):
    with pytest.raises(ValueError, match=match):
        SimulationConfigs.from_dict(overrides, strict=True)


class TestBandwidthLimitationTargetSR:
    """Layer E bandwidth_limitation 可固定 target_sr（UniPASE/GAP 固定 8000）。"""

    def test_explicit_target_sr_is_accepted(self):
        cfg = SimulationConfigs.from_dict(
            {
                "augmentations": {
                    "bandwidth_limitation": {"weight": 1.0, "target_sr": 8000},
                }
            },
            strict=True,
        )
        assert cfg.augmentations["bandwidth_limitation"]["target_sr"] == 8000

    @pytest.mark.parametrize(
        "target_sr",
        [0, -8000, 8000.0, -1.5, True, np.bool_(True), "8000", None],
    )
    def test_non_exact_positive_int_target_sr_fails_fast(self, target_sr):
        with pytest.raises(
            ValueError, match=r"bandwidth_limitation.*target_sr.*positive integer"
        ):
            SimulationConfigs.from_dict(
                {
                    "augmentations": {
                        "bandwidth_limitation": {
                            "weight": 1.0,
                            "target_sr": target_sr,
                        },
                    }
                },
                strict=True,
            )


def _bandwidth_dataset(augmentations: dict):
    """用已校验 SimulationConfigs 构造只带 sim_cfg 的 dummy Dataset。"""
    from core.data.dynamic_dataset_base import BaseDynamicMixingDataset

    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = SimulationConfigs.from_dict(
        {"augmentations": augmentations}, strict=True
    )
    return dataset


def test_gen_bandwidth_params_returns_fixed_target_sr():
    from core.data.dynamic_dataset_base import _BANDWIDTH_RESAMPLE_METHODS

    dataset = _bandwidth_dataset(
        {"bandwidth_limitation": {"weight": 1.0, "target_sr": 8000}}
    )
    fs_new, res_type = dataset._gen_bandwidth_params(
        16000, rng=np.random.default_rng(0)
    )
    assert fs_new == 8000
    assert res_type in _BANDWIDTH_RESAMPLE_METHODS


@pytest.mark.parametrize("sr", [16000, 8000])
def test_gen_bandwidth_params_skips_non_lower_fixed_target_sr(sr):
    dataset = _bandwidth_dataset(
        {"bandwidth_limitation": {"weight": 1.0, "target_sr": 16000}}
    )
    assert dataset._gen_bandwidth_params(sr, rng=np.random.default_rng(0)) == (
        None,
        None,
    )


def test_gen_bandwidth_params_falls_back_to_random_lower_sample_rate():
    from core.data.dynamic_dataset_base import (
        _BANDWIDTH_RESAMPLE_METHODS,
        _BANDWIDTH_SAMPLE_RATES,
    )

    dataset = _bandwidth_dataset({"bandwidth_limitation": {"weight": 1.0}})
    rng = np.random.default_rng(0)
    fs_new, res_type = dataset._gen_bandwidth_params(48000, rng=rng)
    assert fs_new in _BANDWIDTH_SAMPLE_RATES and fs_new < 48000
    assert res_type in _BANDWIDTH_RESAMPLE_METHODS

    # 未配置 target_sr 时保持旧 RNG 序列：先抽采样率，再抽 resample 方法。
    expected_rng = np.random.default_rng(0)
    expected_fs_opts = [fs for fs in _BANDWIDTH_SAMPLE_RATES if fs < 48000]
    expected_fs = expected_fs_opts[
        int(expected_rng.integers(0, len(expected_fs_opts)))
    ]
    assert fs_new == expected_fs
    assert res_type == _BANDWIDTH_RESAMPLE_METHODS[
        int(expected_rng.integers(0, len(_BANDWIDTH_RESAMPLE_METHODS)))
    ]


@pytest.mark.parametrize("target_sr", [0, -8000, 8000.0, True, "8000"])
def test_gen_bandwidth_params_rejects_non_exact_positive_target_sr(target_sr):
    from types import SimpleNamespace

    from core.data.dynamic_dataset_base import BaseDynamicMixingDataset

    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = SimpleNamespace(
        augmentations={
            "bandwidth_limitation": {"weight": 1.0, "target_sr": target_sr}
        }
    )
    with pytest.raises(ValueError, match=r"target_sr.*positive integer"):
        dataset._gen_bandwidth_params(16000, rng=np.random.default_rng(0))


pytestmark = [pytest.mark.unit]
