"""Core data mode and dataset-builder protocol contracts."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tasks.aec.data import build_dataset as build_aec_dataset
from tasks.aec.data.dynamic_aec_build import resolve_aec_dataset_kwargs
from tasks.aec.data.offline_dataset import build_offline_aec_dataset
from core.data.config_flags import (
    resolve_placeholder_num_samples,
    resolve_simulation_cfg_strict,
    resolve_strict_load,
)
from core.data.constants import normalize_data_mode
from core.data.dataset_build import (
    VALIDATION_DYNAMIC_DATA_SOURCE_KEYS,
    DatasetBuildSpec,
    build_dataset,
    expand_simulation_cfg_aec_ref_echo_dataset_keys,
    expand_simulation_cfg_layer_a,
    project_flat_simulation_cfg,
    resolve_dynamic_dataset_build_request,
)
from tasks.se.data import build_dataset as build_se_dataset
from tasks.tse.data import build_dataset as build_tse_dataset
from tasks.tse.data.offline_dataset import build_offline_tse_dataset


@pytest.mark.parametrize(
    ("raw_mode", "task", "storage", "task_mode"),
    [
        ("dynamic", "se", "dynamic", "dynamic"),
        ("offline", "se", "offline", "offline"),
        ("dynamic_aec", "aec", "dynamic", "dynamic_aec"),
        ("offline_aec", "aec", "offline", "offline_aec"),
        ("dynamic_tse", "tse", "dynamic", "dynamic_tse"),
        ("offline_tse", "tse", "offline", "offline_tse"),
        ("dynamic_ssl", "ssl", "dynamic", "dynamic_ssl"),
        ("  DYNAMIC_TSE  ", "tse", "dynamic", "dynamic_tse"),
    ],
)
def test_normalize_data_mode_contract(raw_mode, task, storage, task_mode):
    normalized = normalize_data_mode(raw_mode)

    assert normalized.task == task
    assert normalized.storage == storage
    assert normalized.task_mode == task_mode


@pytest.mark.parametrize("raw_mode", [None, "", "streaming", 1, True])
def test_normalize_data_mode_has_one_error_contract(raw_mode):
    with pytest.raises(
        ValueError,
        match=r"^Unsupported data\.mode .*supported modes: dynamic, offline, "
        r"dynamic_aec, offline_aec, dynamic_tse, offline_tse, "
        r"dynamic_ss, offline_ss, dynamic_ssl$",
    ):
        normalize_data_mode(raw_mode)


@pytest.mark.parametrize(
    "builder",
    [
        build_dataset,
        build_se_dataset,
        build_aec_dataset,
        build_tse_dataset,
        resolve_aec_dataset_kwargs,
        build_offline_aec_dataset,
        build_offline_tse_dataset,
    ],
)
def test_all_dataset_entrypoints_share_unknown_mode_error(builder):
    cfg = {"data": {"mode": "streaming", "train": {"placeholder": True}}}
    kwargs = {"spec": DatasetBuildSpec(task="se", dynamic_dataset_cls=_DatasetSpy)}
    if builder is resolve_aec_dataset_kwargs:
        kwargs = {"split": "train"}
    elif builder is not build_dataset:
        kwargs = {}

    with pytest.raises(
        ValueError,
        match=r"^Unsupported data\.mode 'streaming'; supported modes: dynamic, "
        r"offline, dynamic_aec, offline_aec, dynamic_tse, offline_tse, "
        r"dynamic_ss, offline_ss, dynamic_ssl$",
    ):
        builder(cfg, **kwargs)


class _DatasetSpy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _dynamic_cfg(mode: str = "dynamic") -> dict:
    return {
        "mode": mode,
        "sample_rate": 16000,
        "train": {
            "speech_scp": "speech.scp",
            "noise_scp": "noise.scp",
            "simulation_cfg": {},
            "num_per_epoch": 12,
        },
    }


def test_core_dynamic_request_owns_aec_split_inheritance_and_common_kwargs():
    cfg = {
        "data": {
            "mode": "dynamic_aec",
            "sample_rate": 16000,
            "strict_load": False,
            "placeholder_num_samples": 321,
            "train": {
                "speech_scp": "speech.scp",
                "noise_scp": "noise.scp",
                "rir_scp": "rir.scp",
                "real_farend_echo_pair_json": "train-pairs.json",
                "max_duration_seconds": 3.0,
                "sampling": {"speech": {"language_probs": {"zh": 1}}},
                "num_per_epoch": 12,
                "simulation_cfg": {
                    "aec_ref_echo_random_delay_prob": 0.0,
                },
            },
            "validation": {
                "inherit_simulation": True,
                "speech_scp": "val-speech.scp",
                "real_farend_echo_pair_json": "val-pairs.json",
            },
        },
        "meta": {"seed": 19},
    }
    spec = DatasetBuildSpec(
        task="aec",
        dynamic_dataset_cls=_DatasetSpy,
        simulation_cfg_expanders=(
            expand_simulation_cfg_layer_a,
            expand_simulation_cfg_aec_ref_echo_dataset_keys,
        ),
        simulation_cfg_strict_from_cfg=True,
        dynamic_validation_default=True,
        validation_data_source_keys=VALIDATION_DYNAMIC_DATA_SOURCE_KEYS
        + ("real_farend_echo_pair_json",),
    )

    request = resolve_dynamic_dataset_build_request(cfg, "val", spec=spec)

    assert request.seed == 19
    assert request.dataset_kwargs["speech_scp"] == "val-speech.scp"
    assert request.dataset_kwargs["real_farend_echo_pair_json"] == "val-pairs.json"
    assert request.dataset_kwargs["max_duration_seconds"] == 3.0
    assert request.dataset_kwargs["aec_ref_echo_random_delay_prob"] == 0.0
    assert request.dataset_kwargs["target_sr"] == 16000
    assert request.dataset_kwargs["strict_load"] is False
    assert request.dataset_kwargs["placeholder_num_samples"] == 321
    assert request.dataset_kwargs["simulation_cfg_strict"] is True
    assert "sampling" not in request.dataset_kwargs
    assert "sampling_cfg" not in request.dataset_kwargs
    assert "num_per_epoch" not in request.dataset_kwargs
    assert cfg["data"]["train"]["max_duration_seconds"] == 3.0


def test_tse_builder_rejects_generic_dynamic_mode():
    cfg = _dynamic_cfg("dynamic")
    cfg["train"]["speech_scp"] = "speech.jsonl"
    cfg["train"]["spk2enroll_json"] = "enroll.json"
    cfg["train"]["simulation_cfg"] = {}

    with pytest.raises(ValueError, match=r"belongs to task 'se', not 'tse'"):
        build_tse_dataset(cfg)


def test_tse_builder_rejects_removed_utt2spk_key():
    cfg = _dynamic_cfg("dynamic_tse")
    cfg["train"]["speech_scp"] = "speech.jsonl"
    cfg["train"]["spk2enroll_json"] = "enroll.json"
    cfg["train"]["simulation_cfg"] = {
        "utt2spk": "utt2spk",
    }

    with pytest.raises(ValueError, match="utt2spk"):
        build_tse_dataset(cfg)


def test_flat_simulation_projection_is_non_destructive_and_auditable():
    raw = {
        "max_duration_seconds": 3.0,
        "prob_speed_perturb": 0.25,
        "rir_weight_by_duration": [1.0, 2.0],
        "spk2enroll_json": "enroll.json",
    }

    with pytest.raises(ValueError, match="data split"):
        project_flat_simulation_cfg(raw, task="tse")


@pytest.mark.parametrize(
    ("task", "foreign_key", "owner"),
    [
        ("se", "spk2enroll_json", "TSE"),
        ("tse", "aec_ref_echo_random_delay_prob", "AEC"),
    ],
)
def test_flat_simulation_projection_rejects_foreign_dataset_keys(
    task, foreign_key, owner
):
    if foreign_key == "spk2enroll_json":
        pattern = rf"{foreign_key}.*data split"
    else:
        pattern = rf"{foreign_key}.*{owner}-only"
    with pytest.raises(ValueError, match=pattern):
        project_flat_simulation_cfg({foreign_key: {}}, task=task)


def test_flat_simulation_projection_rejects_unknown_task_and_legacy_layer_a():
    with pytest.raises(ValueError, match="se/aec/tse"):
        project_flat_simulation_cfg({}, task="asr")
    with pytest.raises(
        ValueError,
        match=r"layer_a.*no longer supported.*top level of simulation_cfg",
    ):
        project_flat_simulation_cfg({"layer_a": {}}, task="se")


def test_dataset_build_spec_replaces_boolean_task_switches():
    spec = DatasetBuildSpec(
        task="se",
        dynamic_dataset_cls=_DatasetSpy,
        simulation_cfg_strict_from_cfg=True,
    )

    dataset = build_dataset(_dynamic_cfg(), spec=spec)

    assert isinstance(dataset, _DatasetSpy)
    assert dataset.kwargs["target_sr"] == 16000
    assert "num_per_epoch" not in dataset.kwargs


@pytest.mark.parametrize(
    "legacy_kwargs",
    [
        {"dynamic_dataset_cls": _DatasetSpy},
        {"expand_aec_ref_echo": True},
        {"expand_tse_keys": True},
        {"sim_cfg_strict_from_cfg": True},
        {"sampling_capabilities": None},
    ],
)
def test_build_dataset_rejects_removed_legacy_strategy_kwargs(legacy_kwargs):
    spec = DatasetBuildSpec(task="se", dynamic_dataset_cls=_DatasetSpy)

    with pytest.raises(TypeError):
        build_dataset(_dynamic_cfg(), spec=spec, **legacy_kwargs)


@pytest.mark.parametrize(
    ("task", "top_mode", "validation_mode"),
    [
        ("se", "dynamic", "dynamic_tse"),
        ("aec", "dynamic_aec", "dynamic_tse"),
        ("tse", "dynamic_tse", "dynamic_aec"),
    ],
)
def test_validation_rejects_other_task_mode(task, top_mode, validation_mode):
    cfg = _dynamic_cfg(top_mode)
    cfg["validation"] = {"mode": validation_mode, "speech_scp": "val.scp"}
    spec = DatasetBuildSpec(task=task, dynamic_dataset_cls=_DatasetSpy)

    with pytest.raises(ValueError, match=rf"validation.mode .*not {task!r}"):
        build_dataset(cfg, mode="val", spec=spec)


def test_retired_val_section_alias_is_rejected():
    cfg = _dynamic_cfg()
    cfg["val"] = {"num_per_epoch": 8}
    spec = DatasetBuildSpec(task="se", dynamic_dataset_cls=_DatasetSpy)

    with pytest.raises(ValueError, match=r"data\.val is no longer supported"):
        build_dataset(cfg, mode="val", spec=spec)

    nested = {"data": cfg}
    with pytest.raises(ValueError, match=r"data\.val is no longer supported"):
        resolve_dynamic_dataset_build_request(nested, "train", spec=spec)


def test_dynamic_validation_section_target_sr_overrides_data_sample_rate():
    cfg = _dynamic_cfg()
    cfg["validation"] = {
        "mode": "dynamic",
        "speech_scp": "val.scp",
        "target_sr": 24000,
    }
    spec = DatasetBuildSpec(task="se", dynamic_dataset_cls=_DatasetSpy)

    dataset = build_dataset(cfg, mode="val", spec=spec)

    assert dataset.kwargs["target_sr"] == 24000


def test_offline_train_section_target_sr_overrides_data_sample_rate():
    cfg = {
        "mode": "offline",
        "sample_rate": 16000,
        "train": {
            "clean_scp": "clean.scp",
            "noisy_scp": "noisy.scp",
            "target_sr": 24000,
        },
    }
    spec = DatasetBuildSpec(task="se", dynamic_dataset_cls=_DatasetSpy)

    with patch("core.data.dataset_build.OfflineDataset") as offline:
        build_dataset(cfg, spec=spec)

    assert offline.call_args.kwargs["target_sr"] == 24000
    assert offline.call_args.kwargs["clean_scp"] == "clean.scp"


def test_offline_shared_builder_strictly_parses_simulation_cfg():
    cfg = {
        "mode": "offline",
        "train": {
            "clean_scp": "clean.scp",
            "noisy_scp": "noisy.scp",
            "simulation_cfg": {"spatial_enabld": True},
        },
    }
    with pytest.raises(ValueError, match="SimulationConfigs unknown keys"):
        build_dataset(cfg, spec=DatasetBuildSpec(task="se", dynamic_dataset_cls=_DatasetSpy))


@pytest.mark.parametrize("retired_key", ["clean_list", "noisy_list", "clean_json", "noisy_json", "list_file_sample_rate"])
def test_offline_build_rejects_retired_source_keys(retired_key):
    cfg = {
        "mode": "offline",
        "sample_rate": 16000,
        "train": {"clean_scp": "clean.scp", "noisy_scp": "noisy.scp", retired_key: "retired"},
    }
    spec = DatasetBuildSpec(task="se", dynamic_dataset_cls=_DatasetSpy)

    with pytest.raises(ValueError, match=retired_key):
        build_dataset(cfg, spec=spec)


@pytest.mark.parametrize(
    ("builder", "mode", "section", "patch_target", "expected_key"),
    [
        (
            build_se_dataset,
            "offline",
            {"clean_scp": "clean.scp", "noisy_scp": "noisy.scp"},
            "core.data.dataset_build.OfflineDataset",
            "clean_scp",
        ),
        (
            build_aec_dataset,
            "offline_aec",
            {
                "mic_scp": "mic.scp",
                "farend_scp": "farend.scp",
                "nearend_scp": "nearend.scp",
            },
            "tasks.aec.data.AECOfflineDataset",
            "mic_scp",
        ),
        (
            build_tse_dataset,
            "offline_tse",
            {"samples_list": "raw.list", "cues_yaml": "cues.yaml"},
            "tasks.tse.data.TSEOfflineDataset",
            "samples_list",
        ),
    ],
)
def test_task_specs_route_offline_builds_to_owned_dataset_factory(
    builder, mode, section, patch_target, expected_key
):
    cfg = {
        "mode": mode,
        "sample_rate": 16000,
        "train": section,
        "meta": {"seed": 23},
    }

    with patch(patch_target, _DatasetSpy):
        dataset = builder(cfg)

    assert isinstance(dataset, _DatasetSpy)
    assert dataset.kwargs[expected_key] == section[expected_key]
    assert dataset.kwargs["target_sr"] == 16000
    assert dataset.kwargs["seed"] == 23


def test_dynamic_validation_rejects_retired_offline_json_sources():
    cfg = _dynamic_cfg()
    cfg["train"].update(
        {"clean_json": "offline-clean.json", "noisy_json": "offline-noisy.json"}
    )
    cfg["validation"] = {"mode": "dynamic", "speech_scp": "val.scp"}
    spec = DatasetBuildSpec(task="se", dynamic_dataset_cls=_DatasetSpy)

    with pytest.raises(ValueError, match="clean_json"):
        build_dataset(cfg, mode="val", spec=spec)


def test_aec_builder_keeps_canonical_frontend_out_of_dataset_kwargs():
    cfg = {
        "aec_frontend": {
            "delay_alignment": {"enabled": True},
            "linear_aec": {"backend": "pfdkf_torch"},
            "input_routing": "mic_farend_error",
        },
        "data": {
            "mode": "dynamic_aec",
            "train": {
                "real_farend_echo_pair_json": "pairs.json",
                "speech_scp": "speech.scp",
                "noise_scp": "noise.scp",
                "rir_scp": "rir.scp",
                "simulation_cfg": {"prob_target_only": 1.0},
            },
        },
    }

    kwargs = resolve_aec_dataset_kwargs(cfg, "train")

    assert "aec_frontend" not in kwargs
    assert "linear_aec" not in kwargs
    assert "aec_stream_align" not in kwargs


@pytest.mark.parametrize("invalid", ["false", "true", 0, 1, None])
def test_strict_load_flag_requires_exact_bool_when_configured(invalid):
    with pytest.raises(ValueError, match="must be a bool"):
        resolve_strict_load({"data": {"strict_load": invalid}})


def test_retired_nested_data_layer_is_rejected():
    with pytest.raises(ValueError, match=r"data\.data is no longer supported"):
        resolve_strict_load({"data": {"data": {"strict_load": False}}})

    with pytest.raises(ValueError, match=r"data\.data is no longer supported"):
        resolve_placeholder_num_samples(
            {"data": {"data": {"placeholder_num_samples": 8}}}
        )


@pytest.mark.parametrize("invalid", [True, False, "true", 0, None])
def test_retired_simulation_cfg_strict_key_is_rejected_everywhere(invalid):
    for cfg in (
        {"data": {"simulation_cfg_strict": invalid}},
        {"data": {"train": {"simulation_cfg_strict": invalid}}},
        {"simulation_cfg_strict": invalid},
    ):
        with pytest.raises(ValueError, match="simulation_cfg_strict is no longer supported"):
            resolve_simulation_cfg_strict(cfg)


@pytest.mark.parametrize("invalid", [True, 1.0, "1", 0, -1])
def test_placeholder_num_samples_requires_exact_positive_int(invalid):
    with pytest.raises(ValueError, match="exact int > 0"):
        resolve_placeholder_num_samples(
            {"data": {"placeholder_num_samples": invalid}}
        )


@pytest.mark.parametrize("invalid", [True, 16000.0, "16000", 0, -1])
def test_data_sample_rate_requires_exact_positive_int(invalid):
    cfg = _dynamic_cfg()
    cfg["sample_rate"] = invalid
    with pytest.raises(ValueError, match=r"data\.sample_rate.*exact int > 0"):
        build_dataset(
            cfg,
            spec=DatasetBuildSpec(task="se", dynamic_dataset_cls=_DatasetSpy),
        )


@pytest.mark.parametrize(
    ("key", "invalid"),
    [
        ("repeat_to_max_duration", "true"),
        ("repeat_tile_silence_gap", 1),
        ("max_duration", 16000.0),
        ("max_duration_seconds", 0),
        ("max_duration_seconds", float("nan")),
        ("repeat_tile_silence_gap_range", [0.2, 0.1]),
        ("repeat_tile_silence_gap_range", [0.1, float("inf")]),
    ],
)
def test_dynamic_layer_a_dataset_kwargs_fail_fast(key, invalid):
    cfg = _dynamic_cfg()
    cfg["train"]["simulation_cfg"][key] = invalid
    with pytest.raises(ValueError, match=key):
        build_dataset(
            cfg,
            spec=DatasetBuildSpec(task="se", dynamic_dataset_cls=_DatasetSpy),
        )


def test_dynamic_validation_target_sr_requires_exact_positive_int():
    cfg = _dynamic_cfg()
    cfg["validation"] = {
        "mode": "dynamic",
        "speech_scp": "val.scp",
        "target_sr": "24000",
    }
    with pytest.raises(ValueError, match=r"validation\.target_sr.*exact int > 0"):
        build_dataset(
            cfg,
            mode="val",
            spec=DatasetBuildSpec(task="se", dynamic_dataset_cls=_DatasetSpy),
        )


pytestmark = [pytest.mark.unit]
