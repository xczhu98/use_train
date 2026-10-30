"""任务级 ``simulation_cfg`` ownership 与 capability 合同 Adapter。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from core.data.simulation.capabilities import TEMPORAL_FEATURE_CAPABILITIES


# 退役 key 集合的单一事实来源：``SimulationConfigs.from_dict`` 与投影拒绝都引用这里，
# 避免多处各写一份、漏改即静默放行旧键。
RETIRED_FAREND_VOLUME_KEYS = frozenset(
    {
        "farend_random_volume",
        "farend_volume_db_range",
        "farend_use_rms",
        "farend_volume_distribution",
        "farend_volume_beta_a",
        "farend_volume_beta_b",
    }
)

RETIRED_LAYER_D_ADC_KEYS = frozenset(
    {"adc_hard_clip_prob", "adc_peak_guard_trigger"}
)


_AEC_ONLY_KEYS = frozenset(
    {
        # Layer B AEC 场景。
        "prob_echo_only",
        "prob_noise_echo_only",
        "prob_target_echo",
        "prob_target_noise_echo",
        # C-aec SER / ERL。
        "ser_range",
        "far_db_range",
        "erl_range",
        "erl_distribution",
        "erl_beta_a",
        "erl_beta_b",
        # AEC Dataset 从平面 simulation_cfg 提升的 ref/echo 参数。
        "aec_ref_echo_fine_gcc_align",
        "aec_ref_echo_gcc_bias_seconds",
        "aec_ref_echo_gcc_max_tau_seconds",
        "aec_ref_echo_gcc_max_pad_seconds",
        "aec_ref_echo_gcc_farend_pad_policy",
        "aec_ref_echo_gcc_min_pad_samples",
        "aec_ref_echo_gcc_peak_threshold",
        "aec_ref_echo_gcc_prominence_db",
        "aec_ref_echo_gcc_tv_iqr_ms",
        "aec_ref_echo_gcc_max_windows",
        "aec_ref_echo_gcc_downsample",
        "aec_ref_echo_random_delay_prob",
        "aec_ref_echo_random_delay_mode",
        "aec_ref_echo_random_delay_distribution",
        "aec_ref_echo_random_delay_beta_a",
        "aec_ref_echo_random_delay_beta_b",
        "aec_ref_echo_random_delay_uniform_prob",
        "aec_ref_echo_delay_range_ms",
        "aec_time_delay_jitter",
        "aec_echo_device_eq",
        "aec_echo_clipping",
    }
)

_RETIRED_LAYER_G_KEYS = {
    "aec_stream_align": "aec_frontend.delay_alignment.enabled",
    "linear_aec": "aec_frontend.linear_aec",
}


_REMOVED_SIMULATION_KEYS = frozenset({"utt2spk"})

from core.data.simulation.tse_scenarios import (
    REMOVED_TSE_KEYS, REMOVED_SHARED_KEYS, TSE_SCENARIO_KEYS,
)

_SS_ONLY_KEYS = frozenset()
_RETIRED_SS_SCENARIO_KEYS = frozenset({"prob_speech_only", "prob_speech_noise"})
_RELATIVE_TIMELINE_KEYS = frozenset(
    {"prob_timeline", "overlap_ratio_range", "timeline_mode_weights"}
)
_RELATIVE_TIMELINE_TASKS = frozenset().union(
    *(capability.dynamic_tasks for capability in TEMPORAL_FEATURE_CAPABILITIES)
)

_REMOVED_TSE_SIMULATION_KEYS = REMOVED_TSE_KEYS | REMOVED_SHARED_KEYS
_TSE_ONLY_KEYS = (TSE_SCENARIO_KEYS - {"prob_noise_only", "prob_target_only", "prob_target_noise"}) | frozenset({
    "num_interferers", "sir_range", "spk2enroll_json", "enrollment_duration",
    "enrollment_db", "enroll_policy", "enrollment_augment", "targets_per_mixture",
})

@dataclass(frozen=True)
class _ExplicitSimulationValue:
    key: str
    value: object
    path: str
    provenance: str


def _simulation_mapping(section: Mapping | None) -> Mapping:
    if not isinstance(section, Mapping):
        return {}
    simulation_cfg = section.get("simulation_cfg")
    if not isinstance(simulation_cfg, Mapping):
        return {}
    from core.data.simulation_schema import normalize_simulation_cfg

    return normalize_simulation_cfg(simulation_cfg)


def _explicit_values(
    *,
    split: str,
    train_section: Mapping,
    validation_section: Mapping | None,
) -> tuple[_ExplicitSimulationValue, ...]:
    train_simulation = _simulation_mapping(train_section)
    if split == "train":
        return tuple(
            _ExplicitSimulationValue(
                key=key,
                value=value,
                path=f"data.train.simulation_cfg.{key}",
                provenance="train-local",
            )
            for key, value in train_simulation.items()
        )

    validation_simulation = _simulation_mapping(validation_section)
    local_keys = frozenset(validation_simulation)
    values = [
        _ExplicitSimulationValue(
            key=key,
            value=value,
            path=f"data.validation.simulation_cfg.{key}",
            provenance="validation-local",
        )
        for key, value in validation_simulation.items()
    ]
    values.extend(
        _ExplicitSimulationValue(
            key=key,
            value=value,
            path=f"data.train.simulation_cfg.{key}",
            provenance="train inherited",
        )
        for key, value in train_simulation.items()
        if key not in local_keys
    )
    return tuple(values)


def _raise_ownership(record: _ExplicitSimulationValue, *, task: str, owner: str) -> None:
    raise ValueError(
        f"{record.path} is {owner}-only and cannot be configured for task {task!r} "
        f"(source: {record.provenance}, explicit value: {record.value!r})"
    )


def _raise_retired_farend_volume(record: _ExplicitSimulationValue) -> None:
    raise ValueError(
        f"{record.path} is retired; migrate to far_db_range and erl_range "
        f"(source: {record.provenance}, explicit value: {record.value!r})"
    )


def _validate_schema_field(record: _ExplicitSimulationValue) -> None:
    from core.data.simulation_configs import SimulationConfigs

    try:
        SimulationConfigs.from_dict({record.key: record.value}, strict=True)
    except (TypeError, ValueError) as exc:
        raise type(exc)(
            f"{record.path} is invalid (source: {record.provenance}, explicit value: "
            f"{record.value!r}): {exc}"
        ) from exc


def _validate_capability_value(record: _ExplicitSimulationValue, *, task: str) -> None:
    """校验已接线 D-normalize 能力，并保留 AEC 模式边界。"""
    if record.key in {
        "peak_normalization_mode",
        "peak_normalization_scale_range",
    }:
        if record.key == "peak_normalization_mode" and record.value in {
            "separate",
            "single",
        }:
            raise ValueError(
                f"{record.path} is not supported by the shared Layer D contract: "
                "D-normalize supports only 'off' or 'joint' "
                f"(source: {record.provenance}, explicit value: {record.value!r})"
            )
        _validate_schema_field(record)


def validate_simulation_config_contract(
    *,
    task: str,
    split: str,
    train_section: Mapping,
    validation_section: Mapping | None = None,
) -> None:
    """校验 raw 显式键的任务 ownership、来源与当前 runtime capability。

    ``validation`` 在 merge 前传入；validation-local 键覆盖同名 train 键，其他
    train 显式键保留 ``train inherited`` 来源。该合同与 unknown-key strict 开关
    正交，显式默认值、``None``、``False`` 与 ``0`` 均不会被视为“未配置”。
    """
    if task not in {"se", "aec", "tse", "ss", "ssl"}:
        raise ValueError(f"simulation config task must be se/aec/tse/ss/ssl, got {task!r}")
    if split not in {"train", "val"}:
        raise ValueError(f"simulation config split must be train/val, got {split!r}")
    if not isinstance(train_section, Mapping):
        raise TypeError("train_section must be a mapping")

    for record in _explicit_values(
        split=split,
        train_section=train_section,
        validation_section=validation_section,
    ):
        if record.key in _REMOVED_SIMULATION_KEYS:
            raise ValueError(
                f"{record.path} is no longer supported; remove this configuration key "
                f"(source: {record.provenance}, explicit value: {record.value!r})"
            )
        if record.key in _REMOVED_TSE_SIMULATION_KEYS:
            raise ValueError(
                f"{record.path} is removed; use "
                "the current Layer A–F fields and seven TSE scenarios "
                f"(source: {record.provenance}, explicit value: {record.value!r})"
            )
        if record.key in _RETIRED_SS_SCENARIO_KEYS:
            raise ValueError(
                f"{record.path} is a retired SS scenario key; use the canonical "
                "prob_target_only/prob_target_noise/prob_noise_only names "
                f"(source: {record.provenance})"
            )
        if task == "ss" and record.key in _TSE_ONLY_KEYS:
            _raise_ownership(record, task=task, owner="TSE")
        if (
            record.key in _RELATIVE_TIMELINE_KEYS
            and task not in _RELATIVE_TIMELINE_TASKS
        ):
            _raise_ownership(record, task=task, owner="TSE/SS/SSL multi-source")
        if task in {"se", "aec"} and record.key == "snr_reference":
            _raise_ownership(record, task=task, owner="TSE/SS multi-source")
        if record.key in _RETIRED_LAYER_G_KEYS:
            raise ValueError(
                f"{record.path} is no longer supported; configure top-level "
                f"{_RETIRED_LAYER_G_KEYS[record.key]} instead "
                f"(source: {record.provenance}, explicit value: {record.value!r})"
            )
        if record.key in RETIRED_FAREND_VOLUME_KEYS:
            _raise_retired_farend_volume(record)
        if record.key in RETIRED_LAYER_D_ADC_KEYS:
            raise ValueError(
                f"{record.path} is a retired Layer D ADC key; migrate "
                "observation/device clipping to Layer F with prob_clip, fixed_rail, "
                "and clip_pre_drive_db_range "
                f"(source: {record.provenance}, explicit value: {record.value!r})"
            )
        if task != "aec" and record.key in _AEC_ONLY_KEYS:
            _raise_ownership(record, task=task, owner="AEC")
        if task != "tse" and record.key in _TSE_ONLY_KEYS:
            _raise_ownership(record, task=task, owner="TSE")
        _validate_capability_value(record, task=task)
