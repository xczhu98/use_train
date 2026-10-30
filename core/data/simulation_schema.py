"""Dynamic simulation configuration section adapters.

配置文件可以使用标准分层 section；运行时仍接收已经校验的
``SimulationConfigs`` 平面属性。这个模块只负责一次性的结构投影，不提供默认值，
也不接受同一字段的平面/嵌套重复声明。

方案 B 的公共结构如下（示例）：

.. code-block:: yaml

   simulation_cfg:
     speech:
       splice: {enabled: false}
       speed: {prob: 0.0, rate_range: [0.9, 1.1]}
     acoustic:
       rir:
         backend: online
         reverberation_prob: 1.0
       background:
         snr_low: -5.0
         snr_high: 15.0
     post_mix:
       normalization: {mode: off}
     channel: {enabled: false}
     device: {enabled: false}
     spatial:
       enabled: true
       geometry: {profile: linear_2ch_80mm, reference_mic: 0}
       coherence: {policy: habets_spherical}
       background:
         point_source_prob: 0.3
         wind_background_prob: 0.0
         wind_sidechain_prob: 0.0

分层 YAML 在边界一次投影为 canonical flat runtime mapping；已经归一化的 flat
mapping 只作为内部 ``SimulationConfigs`` / Dataset API 输入，不代表旧 YAML
配置兼容。检测到标准 section 后，任意未知 mapping 都会在配置边界报错。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_LAYERED_SECTION_NAMES = frozenset(
    {
        "speech",      # Layer A
        "scene",       # Layer B
        "acoustic",    # Layer C: rir/background/target
        "rir",         # 兼容 acoustic.rir 的独立迁移 section
        "background",  # 兼容 acoustic.background 的独立迁移 section
        "target",      # 兼容 acoustic.target 的独立迁移 section
        "post_mix",    # Layer D
        "channel",     # Layer E
        "device",      # Layer F
        "spatial",     # 跨 Layer C-F 的几何与空间策略
        "aec",         # AEC 专属
        "tse",         # TSE 专属
        "se",          # SE 专属 Dataset 预处理
        "layer_e",     # 仅用于迁移旧 Layer E 字段
        "layer_f",     # 仅用于迁移旧 Layer F 字段
    }
)

# 解析顺序与运行时 Layer A-F 一致。section 的顺序不改变随机数或 renderer 行为，
# 但固定顺序可以让错误路径、配置快照和文档审查结果稳定可读。
_LAYERED_SECTION_ORDER = (
    "speech",
    "scene",
    "acoustic",
    "rir",
    "background",
    "target",
    "post_mix",
    "channel",
    "device",
    "spatial",
    "aec",
    "tse",
    "se",
    "layer_e",
    "layer_f",
)

# 任务 Dataset 在构造前会从 ``simulation_cfg`` 提升的扩展字段。它们不是
# ``SimulationConfigs`` 的 runtime 属性，但仍属于动态仿真合同；放进 ``aec`` /
# ``tse`` section 后必须先通过这里的结构校验，再由各任务 builder 消费。
_TASK_EXTENSION_FIELDS = frozenset(
    {
        # AEC ref/echo 对齐与 echo-device 变换。
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
        # TSE 的 Dataset-only enrollment / speaker-axis 参数。
        "enrollment_duration",
        "enrollment_db",
        "enroll_policy",
        "enrollment_augment",
        "targets_per_mixture",
        # SE PASE / vocoder 数据集的输入预处理。
        "pase_preprocess",
    }
)


# 这些 aliases 是公开分层配置的稳定词汇；runtime 内部字段名称暂时保持不变，
# 便于逐步迁移五个任务而不把 section 语义散落到各 Dataset。
_PATH_ALIASES: dict[tuple[str, ...], str] = {
    # Layer A: speech preparation.
    ("speech", "splice", "enabled"): "speech_splice",
    ("speech", "splice", "seconds"): "speech_splice_seconds",
    ("speech", "splice", "seconds_range"): "speech_splice_seconds_range",
    ("speech", "splice", "gap_range"): "speech_splice_gap_range",
    ("speech", "splice", "equal_volume"): "speech_splice_equal_volume",
    ("speech", "splice", "equal_volume_range"): "speech_splice_equal_volume_range",
    ("speech", "splice", "max_repeat_times"): "speech_splice_max_repeat_times",
    ("speech", "random_start"): "speech_random_start",
    ("speech", "speed", "prob"): "prob_speed_perturb",
    ("speech", "speed", "rate_range"): "speed_rate_range",
    ("speech", "pitch", "prob"): "prob_pitch_shift",
    ("speech", "pitch", "semitone_range"): "semitone_range",
    ("speech", "eq", "prob"): "prob_eq_perturb",
    ("speech", "eq", "db_range"): "eq_db_range",
    ("speech", "extra_space_prob"): "prob_add_extra_space",
    ("speech", "silence", "prob"): "silence_pad_prob",
    ("speech", "silence", "total_range"): "silence_pad_total_range",
    ("speech", "silence", "position"): "silence_pad_position",
    ("speech", "silence", "mode"): "silence_pad_mode",
    ("speech", "silence", "short_range"): "silence_pad_short_range",
    ("speech", "silence", "long_range"): "silence_pad_long_range",
    ("speech", "silence", "short_ratio"): "silence_pad_short_ratio",
    ("speech", "silence", "beta_a"): "silence_pad_beta_a",
    ("speech", "silence", "beta_b"): "silence_pad_beta_b",
    ("speech", "high_pass", "enabled"): "use_high_pass",
    # Layer B: scene sampling.
    ("scene", "timeline", "prob"): "prob_timeline",
    ("scene", "timeline", "overlap_ratio_range"): "overlap_ratio_range",
    ("scene", "timeline", "mode_weights"): "timeline_mode_weights",
    ("scene", "snr_reference"): "snr_reference",
    # Layer C-0: RIR and target semantics.
    ("rir", "backend"): "rir_backend",
    ("rir", "reverberation_prob"): "prob_reverberation",
    ("rir", "moving_prob"): "prob_moving_rir",
    ("rir", "source_speed_mps_range"): "moving_source_speed_range_mps",
    ("rir", "reuse"): "reuse_rir",
    ("rir", "early_window_sec"): "early_rir_sec",
    ("rir", "target_type"): "target_type",
    ("rir", "rts_target_rt60_sec"): "rts_target_t60_sec",
    ("rir", "rts_tail_after_max_sec"): "rts_time_after_max_sec",
    ("rir", "normalize"): "rir_normalize",
    ("rir", "segment_sec"): "segment_sec",
    # Spatial geometry and background policies.
    ("spatial", "enabled"): "spatial_enabled",
    ("spatial", "geometry", "profile"): "array_profile",
    ("spatial", "geometry", "reference_mic"): "reference_channel",
    ("spatial", "geometry", "custom"): "custom_geometry",
    ("spatial", "geometry", "provenance"): "geometry_provenance",
    ("spatial", "coherence", "policy"): "spatial_diffuse_coherence_model",
    ("spatial", "augmentation", "array_yaw_prob"): "prob_array_rotation",
    ("spatial", "augmentation", "array_yaw_deg_range"): "array_rotation_z_deg_range",
    ("spatial", "augmentation", "mic_jitter_prob"): "prob_microphone_position_jitter",
    ("spatial", "augmentation", "mic_jitter_m_range"): "microphone_position_jitter_m_range",
    ("spatial", "policies", "layer_d"): "spatial_layer_d_policy",
    ("spatial", "policies", "layer_e"): "spatial_layer_e_policy",
    ("spatial", "policies", "layer_f"): "spatial_layer_f_policy",
    ("spatial", "background", "point_source_prob"): "prob_point_source_background",
    ("spatial", "background", "wind_background_prob"): "prob_wind_noise_background",
    ("spatial", "background", "wind_sidechain_prob"): "prob_wind_noise_sidechain",
    # Shared acoustic background.
    ("background", "snr_reference"): "snr_reference",
    ("background", "snr_low"): "snr_low",
    ("background", "snr_high"): "snr_high",
    ("background", "noise_mix_prob"): "noise_mix_prob",
    ("background", "noise_mix_ratio_range"): "noise_mix_ratio_range",
    ("background", "reuse_noise"): "reuse_noise",
    ("background", "colored_noise_prob"): "prob_colored_noise",
    ("background", "colored_noise_snr_range"): "colored_noise_snr_range",
    ("background", "colored_noise_types"): "colored_noise_types",
    ("background", "wind_config"): "wind_noise_config",
    # Target level and observation post-mix contracts.
    ("target", "volume", "enabled"): "source_volume_normalization_enabled",
    ("target", "volume", "db_range"): "source_volume_db_range",
    ("target", "volume", "use_rms"): "source_volume_use_rms",
    ("target", "volume", "distribution"): "source_volume_distribution",
    ("target", "volume", "beta_a"): "source_volume_beta_a",
    ("target", "volume", "beta_b"): "source_volume_beta_b",
    ("post_mix", "normalization", "mode"): "peak_normalization_mode",
    ("post_mix", "normalization", "scale_range"): "peak_normalization_scale_range",
    ("post_mix", "observation", "degradations_enabled"): "noisy_post_mix_degradations_enabled",
    # Layer E/F 全局开关；其余 canonical augmentation 字段可直接放在 section 内。
    ("channel", "enabled"): "prob_channel_enabled",
    ("device", "enabled"): "noisy_post_mix_degradations_enabled",
    # Task-specific scenario fields stay under a task-owned namespace.
    ("aec", "scenario", "echo_only_prob"): "prob_echo_only",
    ("aec", "scenario", "noise_echo_only_prob"): "prob_noise_echo_only",
    ("aec", "scenario", "target_echo_prob"): "prob_target_echo",
    ("aec", "scenario", "target_noise_echo_prob"): "prob_target_noise_echo",
    ("aec", "ser_range"): "ser_range",
    ("aec", "far_db_range"): "far_db_range",
    ("aec", "erl_range"): "erl_range",
    ("aec", "erl_distribution"): "erl_distribution",
    ("aec", "erl_beta_a"): "erl_beta_a",
    ("aec", "erl_beta_b"): "erl_beta_b",
    ("tse", "scenario", "interference_only_prob"): "prob_interference_only",
    ("tse", "scenario", "interference_noise_only_prob"): "prob_interference_noise_only",
    ("tse", "scenario", "target_interference_prob"): "prob_target_interference",
    ("tse", "scenario", "target_interference_noise_prob"): "prob_target_interference_noise",
    ("tse", "num_interferers"): "num_interferers",
    ("tse", "sir_range"): "sir_range",
}

# 已退役的分层路径：值给出迁移说明，命中即在配置边界 fail-fast。
_RETIRED_PATHS: dict[tuple[str, ...], str] = {
    (
        "spatial",
        "augmentation",
        "array_yaw_rad_range",
    ): "use spatial.augmentation.array_yaw_deg_range with degrees (0-360)",
}


# Layer C 使用 ``acoustic`` 作为阅读顺序上的总 section；保留 rir/background/target
# 语义子域，自动复用同一组公开 aliases，避免维护两份字段表。
for _path, _target in tuple(_PATH_ALIASES.items()):
    if _path[0] in {"rir", "background", "target"}:
        _PATH_ALIASES[("acoustic", *_path)] = _target


def _simulation_fields() -> frozenset[str]:
    # 延迟导入避免 ``SimulationConfigs`` 导入本模块时出现循环依赖。
    from core.data.simulation_configs import SimulationConfigs

    return frozenset(
        name
        for name in dir(SimulationConfigs)
        if not name.startswith("_")
        and not callable(getattr(SimulationConfigs, name))
    ) | _TASK_EXTENSION_FIELDS


def _flatten_layered_section(
    value: Mapping[str, Any],
    path: tuple[str, ...],
    out: dict[str, Any],
    canonical_fields: frozenset[str],
) -> None:
    for key, child in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"nested simulation_cfg keys must be non-empty strings, got {key!r}")
        child_path = path + (key,)
        if child_path in _RETIRED_PATHS:
            raise ValueError(
                "nested simulation_cfg retired key "
                f"'{'.'.join(child_path)}'; { _RETIRED_PATHS[child_path] }"
            )
        target = _PATH_ALIASES.get(child_path)
        if target is not None:
            # ``custom_geometry`` / ``timeline_mode_weights`` 等 canonical
            # 字段本身就是 mapping；只有短 alias 指向标量时才拒绝 mapping，
            # 避免把合法的结构值误判为嵌套 section。
            if isinstance(child, Mapping) and target not in canonical_fields:
                raise TypeError(
                    f"simulation_cfg.{'.'.join(child_path)} must be a scalar/list value"
                )
            if target in out:
                raise ValueError(
                    "nested simulation_cfg declares duplicate field "
                    f"{target!r} at {'.'.join(child_path)}"
                )
            out[target] = child
            continue
        # 完整分布（例如 clip_variant_probs / num_interferers）本身就是 mapping；
        # 先按 canonical field 识别，不能继续把其内部权重当作 section。
        if key in canonical_fields:
            if key in out:
                raise ValueError(
                    "nested simulation_cfg declares duplicate field "
                    f"{key!r} at {'.'.join(child_path)}"
                )
            out[key] = child
            continue
        if isinstance(child, Mapping):
            _flatten_layered_section(child, child_path, out, canonical_fields)
            continue
        # 允许 section 内直接使用 canonical runtime field，便于迁移 Layer E/F
        # 的长配置；短别名必须显式登记到 _PATH_ALIASES。
        if key not in canonical_fields:
            raise ValueError(
                "nested simulation_cfg unknown key "
                f"'{'.'.join(child_path)}'; add an explicit field alias or use a "
                "canonical SimulationConfigs field"
            )
        if key in out:
            raise ValueError(
                "nested simulation_cfg declares duplicate field "
                f"{key!r} at {'.'.join(child_path)}"
            )
        out[key] = child


def normalize_simulation_cfg(raw: Mapping[str, Any]) -> dict[str, Any]:
    """将标准分层配置归一化为 runtime flat mapping。

    canonical flat runtime mapping 原样复制；带有任一分层 section 时不允许 flat
    key 与 nested key 混用。这样可以让 YAML 只保留方案 B 的公开结构，同时让
    已校验的 runtime mapping 在内部组件之间保持稳定。
    """
    if not isinstance(raw, Mapping):
        raise TypeError(f"simulation_cfg must be a mapping, got {type(raw).__name__}")
    values = dict(raw)
    sections = {
        key for key, value in values.items()
        if key in {
            "speech", "scene", "acoustic", "rir", "background", "target",
            "post_mix", "channel", "device", "spatial", "aec", "tse", "se",
            "layer_e", "layer_f",
        }
        and isinstance(value, Mapping)
    }
    if "schema" in values:
        raise ValueError(
            "simulation_cfg.schema is not configurable; order fields by Layer A-F "
            "sections without an extra version key"
        )
    if not sections:
        return values
    flat_keys = set(values) - sections
    if flat_keys:
        raise ValueError(
            "nested simulation_cfg cannot mix sections with flat keys: "
            f"{sorted(flat_keys)!r}"
        )
    out: dict[str, Any] = {}
    canonical_fields = _simulation_fields()
    for section in _LAYERED_SECTION_ORDER:
        if section not in sections:
            continue
        child = values[section]
        if not isinstance(child, Mapping):
            raise TypeError(f"simulation_cfg.{section} must be a mapping")
        _flatten_layered_section(child, (section,), out, canonical_fields)
    return out


def normalize_config_simulation_sections(config: dict[str, Any]) -> dict[str, Any]:
    """原地归一化 ``data.train/validation.simulation_cfg`` 并返回 config。

    入口层调用该函数后，任务校验、Dataset builder 和任务专属 capability 检查
    都只看到同一种 flat runtime mapping。
    """
    data = config.get("data") if isinstance(config, Mapping) else None
    if not isinstance(data, dict):
        return config
    for split in ("train", "validation"):
        section = data.get(split)
        if not isinstance(section, dict) or "simulation_cfg" not in section:
            continue
        raw = section["simulation_cfg"]
        if isinstance(raw, Mapping):
            section["simulation_cfg"] = normalize_simulation_cfg(raw)
    return config


__all__ = ["normalize_config_simulation_sections", "normalize_simulation_cfg"]
