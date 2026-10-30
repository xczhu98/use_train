"""Task-specific simulation config ownership 与 capability 合同。"""

from __future__ import annotations

import pytest

from tasks.aec.data.dynamic_aec_build import resolve_aec_dataset_kwargs
from core.data.dataset_build import DatasetBuildSpec, build_dataset
from core.data.simulation_config_contract import validate_simulation_config_contract
from core.data.simulation_configs import SimulationConfigs


class _DatasetSpy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _section(simulation_cfg: dict) -> dict:
    return {"speech_scp": "speech.scp", "simulation_cfg": simulation_cfg}


@pytest.mark.parametrize("task", ["se", "tse"])
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("prob_echo_only", 0.0),
        ("ser_range", None),
        ("erl_range", [-20, 40]),
    ],
)
def test_aec_only_explicit_defaults_are_rejected(task, key, value):
    with pytest.raises(
        ValueError,
        match=rf"data\.train\.simulation_cfg\.{key}.*AEC-only.*{task!r}.*train-local",
    ):
        validate_simulation_config_contract(
            task=task,
            split="train",
            train_section=_section({key: value}),
        )


_RETIRED_FAREND_VOLUME_KEYS = (
    "farend_random_volume",
    "farend_volume_db_range",
    "farend_use_rms",
    "farend_volume_distribution",
    "farend_volume_beta_a",
    "farend_volume_beta_b",
)


@pytest.mark.parametrize("task", ["se", "aec", "tse"])
@pytest.mark.parametrize("key", _RETIRED_FAREND_VOLUME_KEYS)
def test_retired_farend_volume_keys_are_rejected_for_every_task(task, key):
    with pytest.raises(
        ValueError,
        match=rf"data\.train\.simulation_cfg\.{key}.*far_db_range.*erl_range.*train-local",
    ):
        validate_simulation_config_contract(
            task=task,
            split="train",
            train_section=_section({key: False}),
        )


@pytest.mark.parametrize("task", ["se", "aec"])
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("prob_interference_only", 0.0),
        ("sir_range", None),
        ("spk2enroll_json", None),
        ("prob_target_interference", 0.0),
    ],
)
def test_tse_only_explicit_defaults_are_rejected(task, key, value):
    with pytest.raises(
        ValueError,
        match=rf"data\.train\.simulation_cfg\.{key}.*TSE(?:/SS)?-only.*{task!r}.*train-local",
    ):
        validate_simulation_config_contract(
            task=task,
            split="train",
            train_section=_section({key: value}),
        )


def test_validation_inherited_error_preserves_train_path_and_source():
    with pytest.raises(
        ValueError,
        match=(
            r"data\.train\.simulation_cfg\.ser_range.*AEC-only.*'se'.*"
            r"train inherited"
        ),
    ):
        validate_simulation_config_contract(
            task="se",
            split="val",
            train_section=_section({"ser_range": None}),
            validation_section={"simulation_cfg": {"snr_low": 0.0}},
        )


def test_validation_local_error_preserves_validation_path_and_source():
    with pytest.raises(
        ValueError,
        match=(
            r"data\.validation\.simulation_cfg\.prob_interference_only.*TSE-only.*'aec'.*"
            r"validation-local"
        ),
    ):
        validate_simulation_config_contract(
            task="aec",
            split="val",
            train_section=_section({}),
            validation_section={"simulation_cfg": {"prob_interference_only": 0.0}},
        )


@pytest.mark.parametrize("mode", ["off", "joint"])
def test_aec_accepts_wired_peak_normalization(mode):
    validate_simulation_config_contract(
        task="aec",
        split="train",
        train_section=_section(
            {
                "peak_normalization_mode": mode,
                "peak_normalization_scale_range": [0.4, 0.9],
            }
        ),
    )


@pytest.mark.parametrize("mode", ["separate", "single"])
def test_aec_rejects_non_follower_peak_normalization(mode):
    with pytest.raises(ValueError, match="supports only 'off' or 'joint'"):
        validate_simulation_config_contract(
            task="aec",
            split="train",
            train_section=_section({"peak_normalization_mode": mode}),
        )


@pytest.mark.parametrize("task", ["se", "aec", "tse"])
@pytest.mark.parametrize(
    ("key", "value"),
    [("adc_hard_clip_prob", 0.1), ("adc_peak_guard_trigger", "all_stems")],
)
def test_removed_adc_config_keys_are_rejected_for_every_task(task, key, value):
    with pytest.raises(
        ValueError,
        match=(
            rf"data\.train\.simulation_cfg\.{key}.*retired Layer D ADC.*"
            r"Layer F.*prob_clip.*fixed_rail.*clip_pre_drive_db_range.*train-local"
        ),
    ):
        validate_simulation_config_contract(
            task=task,
            split="train",
            train_section=_section({key: value}),
        )


@pytest.mark.parametrize("invalid", ["disabled", "Joint"])
def test_peak_mode_schema_rejects_unknown_selector_before_capability(invalid):
    with pytest.raises(ValueError, match="peak_normalization_mode.*one of"):
        validate_simulation_config_contract(
            task="aec",
            split="train",
            train_section=_section({"peak_normalization_mode": invalid}),
        )


@pytest.mark.parametrize("invalid", [[0.0, 0.95], [-0.1, 0.9], [0.5, 1.1]])
def test_peak_scale_schema_rejects_invalid_amplitude_domain(invalid):
    with pytest.raises(ValueError, match=r"peak_normalization_scale_range.*0 < lo"):
        SimulationConfigs.from_dict({"peak_normalization_scale_range": invalid})


def test_shared_builder_enforces_ownership_for_foreign_task_keys():
    cfg = {
        "mode": "dynamic",
        "train": _section({"ser_range": None}),
    }
    spec = DatasetBuildSpec(
        task="se",
        dynamic_dataset_cls=_DatasetSpy,
        simulation_cfg_strict_from_cfg=True,
    )

    with pytest.raises(ValueError, match=r"ser_range.*AEC-only.*train-local"):
        build_dataset(cfg, spec=spec)


def test_shared_builder_rejects_legacy_layer_a_before_dataset_construction():
    cfg = {
        "mode": "dynamic",
        "train": _section({"layer_a": {"ser_range": None}}),
    }
    spec = DatasetBuildSpec(task="se", dynamic_dataset_cls=_DatasetSpy)

    with pytest.raises(
        ValueError,
        match=r"layer_a.*no longer supported.*top level of simulation_cfg",
    ):
        build_dataset(cfg, spec=spec)


def test_shared_validation_builder_reports_train_inherited_provenance():
    cfg = {
        "mode": "dynamic",
        "train": _section({"ser_range": None}),
        "validation": {"mode": "dynamic", "simulation_cfg": {"snr_low": 0.0}},
    }
    spec = DatasetBuildSpec(task="se", dynamic_dataset_cls=_DatasetSpy)

    with pytest.raises(
        ValueError,
        match=r"data\.train\.simulation_cfg\.ser_range.*train inherited",
    ):
        build_dataset(cfg, mode="val", spec=spec)


def test_aec_specialized_resolver_uses_same_adapter():
    cfg = {
        "data": {
            "mode": "dynamic_aec",
            "train": {
                "real_farend_echo_pair_json": "pairs.json",
                "simulation_cfg": {"prob_interference_only": 0.0},
            },
        }
    }

    with pytest.raises(ValueError, match=r"prob_interference_only.*TSE-only.*train-local"):
        resolve_aec_dataset_kwargs(cfg, "train")


def test_aec_specialized_validation_preserves_validation_local_provenance():
    cfg = {
        "data": {
            "mode": "dynamic_aec",
            "train": {
                "real_farend_echo_pair_json": "train.json",
                "simulation_cfg": {},
            },
            "validation": {
                "real_farend_echo_pair_json": "val.json",
                "simulation_cfg": {"prob_interference_only": 0.0},
            },
        }
    }

    with pytest.raises(
        ValueError,
        match=r"data\.validation\.simulation_cfg\.prob_interference_only.*validation-local",
    ):
        resolve_aec_dataset_kwargs(cfg, "val")


pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize("task", ["se", "aec", "tse", "ss"])
@pytest.mark.parametrize("key", ["prob_speech_only", "prob_speech_noise"])
def test_retired_ss_scenario_keys_are_rejected(task, key):
    with pytest.raises(ValueError, match=r"retired SS scenario key"):
        validate_simulation_config_contract(
            task=task,
            split="train",
            train_section=_section({key: 1.0}),
        )


@pytest.mark.parametrize("task", ["se", "aec"])
@pytest.mark.parametrize(
    "key, owner",
    [
        ("prob_timeline", "TSE/SS/SSL multi-source-only"),
        ("overlap_ratio_range", "TSE/SS/SSL multi-source-only"),
        ("snr_reference", "TSE/SS multi-source-only"),
    ],
)
def test_multi_source_keys_are_not_silently_ignored_by_single_source_tasks(task, key, owner):
    value = [0.1, 0.5] if key == "overlap_ratio_range" else (
        "all_speech" if key == "snr_reference" else 0.0
    )
    with pytest.raises(ValueError, match=owner):
        validate_simulation_config_contract(
            task=task,
            split="train",
            train_section=_section({key: value}),
        )


def test_ssl_owns_relative_timeline_configuration():
    validate_simulation_config_contract(
        task="ssl",
        split="train",
        train_section=_section(
            {
                "prob_timeline": 1.0,
                "overlap_ratio_range": [0.2, 0.8],
                "timeline_mode_weights": {
                    "head": 0.2,
                    "tail": 0.2,
                    "middle": 0.2,
                    "crossing": 0.2,
                    "containment": 0.2,
                },
            }
        ),
    )
