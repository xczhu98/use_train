"""已校验 ``SimulationConfigs`` 的 immutable runtime typed slices。

本模块只投影 ``simulation_cfg`` 经当前 strict parser 解析和校验后的对象；
外部配置的分层解析留在配置边界，不在这里提供默认值或复制 validator。
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from core.data.simulation.array_profiles import (
    ResolvedMicrophoneArray,
    resolve_microphone_array,
)
from core.data.simulation.capabilities import triggered_temporal_features
from core.data.simulation.spatial_background import validate_spatial_background_capability
from core.data.simulation_configs import SimulationConfigs


class FrozenMapping(Mapping):
    """可 pickle 的递归 immutable mapping。"""

    __slots__ = ("_items", "_data")

    def __init__(self, values: Mapping) -> None:
        items = tuple((key, _freeze(value)) for key, value in values.items())
        self._items = items
        self._data = dict(items)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __reduce__(self):
        return (type(self), (dict(self._items),))

    def __repr__(self) -> str:
        return f"FrozenMapping({self._data!r})"


def _freeze(value):
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class LayerASimulationView:
    speech_splice: bool
    speech_splice_seconds: float
    speech_splice_seconds_range: tuple | None
    speech_splice_gap_range: tuple
    speech_splice_equal_volume: bool
    speech_splice_equal_volume_range: tuple
    speech_splice_max_repeat_times: int
    speech_random_start: bool
    prob_speed_perturb: float
    speed_rate_range: tuple
    prob_pitch_shift: float
    semitone_range: tuple
    prob_eq_perturb: float
    eq_db_range: tuple
    prob_add_extra_space: float
    silence_pad_prob: float
    silence_pad_total_range: tuple
    silence_pad_position: str
    silence_pad_mode: str
    silence_pad_short_range: tuple | None
    silence_pad_long_range: tuple | None
    silence_pad_short_ratio: float
    silence_pad_beta_a: float
    silence_pad_beta_b: float
    use_high_pass: bool


@dataclass(frozen=True)
class TemporalSimulationView:
    """共享的多源相对时间线只读视图。

    时间线发生在每个 source 完成 Layer A 画布之后，属于 scene layout，
    不属于单源 speech preparation 的 Layer A。``timeline_mode_weights``
    使用递归冻结后的 mapping，避免 runtime 消费者绕过配置边界修改权重。
    """

    prob_timeline: float
    overlap_ratio_range: tuple
    timeline_mode_weights: FrozenMapping


@dataclass(frozen=True)
class AcousticSimulationView:
    music_ratio: float
    rir_backend: str
    snr_reference: str
    prob_reverberation: float
    prob_moving_rir: float
    moving_source_speed_range_mps: tuple
    reuse_rir: bool
    early_rir_sec: float
    target_type: str
    rts_target_t60_sec: float
    rts_time_after_max_sec: float
    rir_normalize: bool
    segment_sec: float
    source_volume_normalization_enabled: bool
    source_volume_db_range: tuple
    source_volume_use_rms: bool
    source_volume_distribution: str
    source_volume_beta_a: float
    source_volume_beta_b: float
    snr_low: float
    snr_high: float
    reuse_noise: bool
    noise_mix_prob: float
    noise_mix_ratio_range: tuple
    prob_wind_noise_sidechain: float
    wind_noise_config: FrozenMapping
    prob_colored_noise: float
    colored_noise_snr_range: tuple
    colored_noise_types: tuple


@dataclass(frozen=True)
class BackgroundSimulationView:
    """跨任务共享的背景策略只读视图。"""

    prob_point_source_background: float
    prob_wind_noise_background: float
    spatial_diffuse_coherence_model: str
    noise_mix_prob: float
    noise_mix_ratio_range: tuple
    reuse_noise: bool
    prob_wind_noise_sidechain: float
    wind_noise_config: FrozenMapping
    prob_colored_noise: float
    colored_noise_snr_range: tuple
    colored_noise_types: tuple


@dataclass(frozen=True)
class GeometrySimulationView:
    """跨任务共享的只读 microphone geometry 视图。

    ``SimulationConfigs`` 仍是配置边界的唯一校验入口；该 view 只保存解析后的
    topology、reference channel 和 provenance，供 renderer、provider 与 task
    projection 逐步迁移使用。没有 geometry 的 mono 配置仍显式保留 ``channels=1``
    和 ``positions_m=None``。
    """

    spatial_enabled: bool
    array_profile: str | None
    custom_geometry: FrozenMapping | None
    reference_channel: int
    channels: int
    positions_m: tuple[tuple[float, float, float], ...] | None
    spacing_m: float | None
    coordinate_system: str | None
    geometry_kind: str | None
    geometry_provenance: FrozenMapping | None
    validation_recipe: FrozenMapping | None
    resolved_array: ResolvedMicrophoneArray | None


@dataclass(frozen=True)
class SpatialSimulationView:
    """共享 spatial microphone observation 与 Layer D-F 粒度。"""

    spatial_enabled: bool
    channels: int
    reference_channel: int
    array_profile: str | None
    microphone_positions: tuple | None
    custom_geometry: FrozenMapping | None
    geometry_provenance: FrozenMapping | None
    validation_recipe: FrozenMapping | None
    spatial_diffuse_coherence_model: str
    prob_array_rotation: float
    array_rotation_z_rad_range: tuple
    prob_microphone_position_jitter: float
    microphone_position_jitter_m_range: tuple
    spatial_layer_d_policy: str
    spatial_layer_e_policy: str
    spatial_layer_f_policy: str


@dataclass(frozen=True)
class LayerCConfig:
    """Layer C 的声学混合视图。

    Layer C 的背景策略和 RIR/音量/SNR 参数在历史代码中分别由
    ``acoustic`` 与 ``background`` 两个视图承载；这里提供一个只读组合视图，
    让新 renderer 以 A-F 术语访问，而不复制或重新解析配置字段。
    """

    acoustic: AcousticSimulationView
    background: BackgroundSimulationView


@dataclass(frozen=True)
class PeakSimulationView:
    """Layer D-normalize 配置切片；D-guard 无配置字段且固定执行。"""

    peak_normalization_mode: str
    peak_normalization_scale_range: tuple


@dataclass(frozen=True)
class ChannelSimulationView:
    prob_channel_enabled: bool
    augmentations_name: tuple
    num_augmentations: FrozenMapping
    augmentations: FrozenMapping


@dataclass(frozen=True)
class PostMixSimulationView:
    noisy_post_mix_degradations_enabled: bool
    prob_clip: float
    clip_variants: tuple
    clip_variant_probs: FrozenMapping
    clip_pre_drive_db_range: tuple
    clip_hard_portion: float
    clip_hard_on_rate: bool
    clip_rate_range: tuple
    clip_threshold_db_range: tuple
    clip_soft_slope_range: tuple
    clip_sigmoid_slope_range: tuple
    clip_sox_threshold_db_range: tuple
    clip_pedal_threshold_db_range: tuple
    prob_post_eq: float
    post_eq_db_range: tuple
    prob_lowpass: float
    lowpass_cutoff_range: tuple
    lowpass_order_range: tuple
    prob_band_reject: float
    band_reject_center_freq_range: tuple
    band_reject_freq_bw_range: tuple
    band_reject_use_stft: bool
    band_reject_max_n: int
    prob_dc_offset: float
    dc_offset_range: tuple
    prob_drc: float
    drc_threshold_db_range: tuple
    drc_ratio_range: tuple
    drc_attack_ms_range: tuple
    drc_release_ms_range: tuple
    prob_codecs: float
    codecs_types: tuple
    prob_bit_crush: float
    bit_crush_range: tuple
    prob_spectral_holes: float
    spectral_holes_frame_length: int
    spectral_holes_hop_length: int
    spectral_holes_num_range: tuple
    spectral_holes_width_freq_range: tuple
    spectral_holes_width_time_range: tuple
    spectral_holes_cutoff_freq: float
    prob_webrtc_ns: float
    webrtc_ns_levels: tuple
    prob_webrtc_agc: float
    webrtc_agc_target_dbfs_range: tuple
    prob_bass_boost: float
    bass_boost_highpass_cutoff_range: tuple
    bass_boost_attenuation_min_db: float
    prob_spectral_leakage: float
    spectral_leakage_window_lengths: tuple
    spectral_leakage_max_time_shift: int
    prob_loudness_perturb: float
    loudness_factor_range: tuple
    loudness_max_n_intervals: int


@dataclass(frozen=True)
class SharedSimulationView:
    layer_a: LayerASimulationView
    temporal: TemporalSimulationView
    acoustic: AcousticSimulationView
    background: BackgroundSimulationView
    peak: PeakSimulationView
    channel: ChannelSimulationView
    post_mix: PostMixSimulationView
    geometry: GeometrySimulationView
    spatial: SpatialSimulationView


@dataclass(frozen=True)
class SEScenarioSimulationView:
    prob_target_only: float
    prob_noise_only: float
    prob_target_noise: float


@dataclass(frozen=True)
class AECScenarioSimulationView:
    prob_target_only: float
    prob_noise_only: float
    prob_echo_only: float
    prob_noise_echo_only: float
    prob_target_noise: float
    prob_target_echo: float
    prob_target_noise_echo: float


@dataclass(frozen=True)
class SSScenarioSimulationView:
    prob_target_only: float
    prob_target_noise: float
    prob_noise_only: float


@dataclass(frozen=True)
class AECEchoSimulationView:
    ser_range: tuple | None
    far_db_range: tuple
    erl_range: tuple
    erl_distribution: str
    erl_beta_a: float
    erl_beta_b: float


@dataclass(frozen=True)
class TSEScenarioSimulationView:
    prob_target_only: float
    prob_interference_only: float
    prob_noise_only: float
    prob_interference_noise_only: float
    prob_target_noise: float
    prob_target_interference: float
    prob_target_interference_noise: float
    num_interferers: FrozenMapping
    sir_range: tuple | None


@dataclass(frozen=True)
class TaskSemanticSimulationView:
    """任务场景和监督语义的只读投影视图。

    ``branch_probabilities`` 与 ``target_presence_probabilities`` 只保存当前
    ``SimulationConfigs`` 已校验的值，不负责重新校验或补默认值。角色字段是
    任务语义类别（例如 ``nearend``、``interference``、``source``），不是
    ``WaveformSceneBundle`` 的具体 role key；具体 role 仍由任务 adapter 负责。
    ``semantic`` 字段以默认 ``None`` 追加到各任务 view，保留旧构造方式，便于
    逐步迁移消费者。
    """

    task: str
    branch_probabilities: FrozenMapping
    target_presence_probabilities: FrozenMapping
    target_type: str | None
    target_roles: tuple[str, ...]
    fixed_source_count: bool
    spatial_supervision: bool


# 公共命名以 A-F 层为主；保留旧的 *SimulationView 名称，避免任务侧已有
# import 在迁移期间失效。Layer B 是任务语义切片，具体 branch 仍由任务 view
# 校验，Layer C 则由 RIR/背景两个已校验切片组合而成。
LayerAConfig = LayerASimulationView
LayerBConfig = TaskSemanticSimulationView
LayerDConfig = PeakSimulationView
LayerEConfig = ChannelSimulationView
LayerFConfig = PostMixSimulationView


@dataclass(frozen=True)
class SimulationLayerViews:
    """任务无关的 A-F typed runtime 入口。

    外部 YAML 仍保持平面 ``simulation_cfg``；该对象只由
    :func:`simulation_layers_for` 从严格解析后的 task view 投影，不负责默认值
    或二次校验。``layer_b`` 使用任务语义 view，因而不会把 AEC/SE/TSE 的
    branch 概率混成一个错误的共享场景合同。
    """

    task: str
    layer_a: LayerAConfig
    layer_b: LayerBConfig
    layer_c: LayerCConfig
    layer_d: LayerDConfig
    layer_e: LayerEConfig
    layer_f: LayerFConfig


@dataclass(frozen=True)
class SESimulationView:
    task: str
    shared: SharedSimulationView
    scenario: SEScenarioSimulationView
    semantic: TaskSemanticSimulationView | None = None


@dataclass(frozen=True)
class AECSimulationView:
    task: str
    shared: SharedSimulationView
    scenario: AECScenarioSimulationView
    echo: AECEchoSimulationView
    semantic: TaskSemanticSimulationView | None = None


@dataclass(frozen=True)
class TSESimulationView:
    task: str
    shared: SharedSimulationView
    scenario: TSEScenarioSimulationView
    semantic: TaskSemanticSimulationView | None = None


@dataclass(frozen=True)
class SSSimulationView:
    task: str
    shared: SharedSimulationView
    scenario: SSScenarioSimulationView
    semantic: TaskSemanticSimulationView | None = None


@dataclass(frozen=True)
class SSLSimulationView:
    """SSL 的 task-neutral shared simulation view。

    SSL 的 source-count、trajectory 和 direct-path target 由 task adapter 管理；
    Layer A/C/D/E/F 的公共字段仍通过 ``shared`` 使用，避免借用 SS scenario
    字段造成任务语义污染。
    """

    task: str
    shared: SharedSimulationView
    semantic: TaskSemanticSimulationView | None = None


TaskSimulationView: TypeAlias = SESimulationView | AECSimulationView | TSESimulationView | SSSimulationView | SSLSimulationView


def _values(cfg: SimulationConfigs, names: tuple[str, ...]) -> dict[str, Any]:
    return {name: _freeze(getattr(cfg, name)) for name in names}


def _shared_view(cfg: SimulationConfigs) -> SharedSimulationView:
    layer_a = LayerASimulationView(
        **_values(cfg, tuple(LayerASimulationView.__annotations__))
    )
    temporal = TemporalSimulationView(
        **_values(cfg, tuple(TemporalSimulationView.__annotations__))
    )
    acoustic = AcousticSimulationView(
        **_values(cfg, tuple(AcousticSimulationView.__annotations__))
    )
    background = BackgroundSimulationView(
        **_values(cfg, tuple(BackgroundSimulationView.__annotations__))
    )
    peak = PeakSimulationView(
        **_values(cfg, tuple(PeakSimulationView.__annotations__))
    )
    channel = ChannelSimulationView(
        **_values(cfg, tuple(ChannelSimulationView.__annotations__))
    )
    post_mix = PostMixSimulationView(
        **_values(cfg, tuple(PostMixSimulationView.__annotations__))
    )
    resolved_array = resolve_microphone_array(
        array_profile=cfg.array_profile,
        custom_geometry=cfg.custom_geometry,
        reference_channel=cfg.reference_channel,
    )
    resolved_positions = None if resolved_array is None else resolved_array.positions_m
    geometry = GeometrySimulationView(
        spatial_enabled=cfg.spatial_enabled,
        array_profile=cfg.array_profile,
        custom_geometry=(FrozenMapping(cfg.custom_geometry) if cfg.custom_geometry is not None else None),
        reference_channel=cfg.reference_channel,
        channels=len(resolved_positions) if resolved_positions is not None else 1,
        positions_m=resolved_positions,
        spacing_m=(None if resolved_array is None else resolved_array.profile.spacing_m),
        coordinate_system=(None if resolved_array is None else resolved_array.profile.coordinate_system),
        geometry_kind=(
            None
            if resolved_array is None
            else ("custom" if resolved_array.profile.name == "custom" else "profile")
        ),
        geometry_provenance=(FrozenMapping(cfg.geometry_provenance) if cfg.geometry_provenance is not None else None),
        validation_recipe=(FrozenMapping(cfg.validation_recipe) if cfg.validation_recipe is not None else None),
        resolved_array=resolved_array,
    )
    spatial = SpatialSimulationView(
        spatial_enabled=cfg.spatial_enabled,
        channels=geometry.channels,
        reference_channel=cfg.reference_channel,
        array_profile=cfg.array_profile,
        microphone_positions=resolved_positions,
        custom_geometry=(FrozenMapping(cfg.custom_geometry) if cfg.custom_geometry is not None else None),
        geometry_provenance=(FrozenMapping(cfg.geometry_provenance) if cfg.geometry_provenance is not None else None),
        validation_recipe=(FrozenMapping(cfg.validation_recipe) if cfg.validation_recipe is not None else None),
        spatial_diffuse_coherence_model=cfg.spatial_diffuse_coherence_model,
        prob_array_rotation=cfg.prob_array_rotation,
        # view 保持 rad 口径；配置侧 SimulationConfigs 单位为 degree，在此转换。
        array_rotation_z_rad_range=tuple(
            math.radians(v) for v in cfg.array_rotation_z_deg_range
        ),
        prob_microphone_position_jitter=cfg.prob_microphone_position_jitter,
        microphone_position_jitter_m_range=tuple(cfg.microphone_position_jitter_m_range),
        spatial_layer_d_policy=cfg.spatial_layer_d_policy,
        spatial_layer_e_policy=cfg.spatial_layer_e_policy,
        spatial_layer_f_policy=cfg.spatial_layer_f_policy,
    )
    return SharedSimulationView(
        layer_a=layer_a,
        temporal=temporal,
        acoustic=acoustic,
        background=background,
        peak=peak,
        channel=channel,
        post_mix=post_mix,
        geometry=geometry,
        spatial=spatial,
    )


def _task_semantic_view(cfg: SimulationConfigs, task: str) -> TaskSemanticSimulationView:
    """投影任务语义；不改变既有 task-specific scenario view。"""

    def branch_probabilities(*names: str) -> FrozenMapping:
        return FrozenMapping({name: float(getattr(cfg, f"prob_{name}")) for name in names})

    def target_presence(present: float, absent: float) -> FrozenMapping:
        return FrozenMapping({"present": float(present), "absent": float(absent)})

    if task == "se":
        branches = branch_probabilities("target_only", "noise_only", "target_noise")
        presence = target_presence(
            cfg.prob_target_only + cfg.prob_target_noise,
            cfg.prob_noise_only,
        )
        return TaskSemanticSimulationView(
            task=task,
            branch_probabilities=branches,
            target_presence_probabilities=presence,
            target_type=str(cfg.target_type),
            target_roles=("target",),
            fixed_source_count=False,
            spatial_supervision=False,
        )
    if task == "aec":
        branch_names = (
            "target_only",
            "noise_only",
            "echo_only",
            "noise_echo_only",
            "target_noise",
            "target_echo",
            "target_noise_echo",
        )
        branches = branch_probabilities(*branch_names)
        presence = target_presence(
            cfg.prob_target_only
            + cfg.prob_target_noise
            + cfg.prob_target_echo
            + cfg.prob_target_noise_echo,
            cfg.prob_noise_only + cfg.prob_echo_only + cfg.prob_noise_echo_only,
        )
        return TaskSemanticSimulationView(
            task=task,
            branch_probabilities=branches,
            target_presence_probabilities=presence,
            target_type=str(cfg.target_type),
            target_roles=("nearend",),
            fixed_source_count=False,
            spatial_supervision=False,
        )
    if task == "tse":
        branch_names = (
            "target_only",
            "interference_only",
            "noise_only",
            "interference_noise_only",
            "target_noise",
            "target_interference",
            "target_interference_noise",
        )
        branches = branch_probabilities(*branch_names)
        presence = target_presence(
            cfg.prob_target_only
            + cfg.prob_target_noise
            + cfg.prob_target_interference
            + cfg.prob_target_interference_noise,
            cfg.prob_interference_only
            + cfg.prob_noise_only
            + cfg.prob_interference_noise_only,
        )
        return TaskSemanticSimulationView(
            task=task,
            branch_probabilities=branches,
            target_presence_probabilities=presence,
            target_type=str(cfg.target_type),
            target_roles=("target",),
            fixed_source_count=False,
            spatial_supervision=False,
        )
    if task == "ss":
        return TaskSemanticSimulationView(
            task=task,
            branch_probabilities=FrozenMapping({}),
            target_presence_probabilities=target_presence(
                cfg.prob_target_only + cfg.prob_target_noise,
                cfg.prob_noise_only,
            ),
            target_type=str(cfg.target_type),
            target_roles=("source",),
            fixed_source_count=True,
            spatial_supervision=False,
        )
    return TaskSemanticSimulationView(
        task=task,
        branch_probabilities=FrozenMapping({}),
        target_presence_probabilities=target_presence(1.0, 0.0),
        target_type=str(cfg.target_type),
        target_roles=("target",),
        fixed_source_count=True,
        spatial_supervision=True,
    )


def project_simulation_view(cfg: SimulationConfigs, task: str) -> TaskSimulationView:
    """从已校验的当前 SimulationConfigs 对象投影任务 view。"""
    if not isinstance(cfg, SimulationConfigs):
        raise TypeError(
            "project_simulation_view requires a validated SimulationConfigs instance"
        )
    if task not in {"se", "aec", "tse", "ss", "ssl"}:
        raise ValueError(f"simulation view task must be se/aec/tse/ss/ssl, got {task!r}")
    validate_spatial_background_capability(task, cfg)
    unsupported_temporal = triggered_temporal_features(task, cfg)
    if unsupported_temporal:
        raise ValueError(
            f"task {task!r} does not support active temporal features: "
            f"{unsupported_temporal!r}"
        )
    shared = _shared_view(cfg)
    semantic = _task_semantic_view(cfg, task)
    common_scenario = SEScenarioSimulationView(
        **_values(cfg, tuple(SEScenarioSimulationView.__annotations__))
    )
    if task == "se":
        return SESimulationView(
            task=task, shared=shared, scenario=common_scenario, semantic=semantic
        )
    if task == "ss":
        return SSSimulationView(
            task=task,
            shared=shared,
            scenario=SSScenarioSimulationView(
                **_values(cfg, tuple(SSScenarioSimulationView.__annotations__))
            ),
            semantic=semantic,
        )
    if task == "ssl":
        return SSLSimulationView(task=task, shared=shared, semantic=semantic)
    if task == "aec":
        return AECSimulationView(
            task=task,
            shared=shared,
            scenario=AECScenarioSimulationView(
                **_values(cfg, tuple(AECScenarioSimulationView.__annotations__))
            ),
            echo=AECEchoSimulationView(
                **_values(cfg, tuple(AECEchoSimulationView.__annotations__))
            ),
            semantic=semantic,
        )
    return TSESimulationView(
        task=task,
        shared=shared,
        scenario=TSEScenarioSimulationView(
            **_values(cfg, tuple(TSEScenarioSimulationView.__annotations__))
        ),
        semantic=semantic,
    )


def simulation_view_for(source: object, task: str | None = None) -> TaskSimulationView:
    """返回 owner/config 的 typed view。

    只接受已校验的 ``SimulationConfigs``，或暴露 ``_simulation_view`` /
    ``sim_cfg`` 的 owner 对象；其他对象（含只提供零散属性的旧 namespace）直接
    ``TypeError``，不再就地拼装兼容配置。
    """
    existing = getattr(source, "_simulation_view", None)
    if isinstance(
        existing,
        (SESimulationView, AECSimulationView, TSESimulationView, SSSimulationView, SSLSimulationView),
    ):
        if task is not None and existing.task != task:
            raise ValueError(
                f"simulation view task mismatch: expected {task!r}, got {existing.task!r}"
            )
        return existing
    if hasattr(source, "sim_cfg"):
        return simulation_view_for(getattr(source, "sim_cfg"), task=task)
    if task is None:
        raise ValueError("task is required when projecting a SimulationConfigs source")
    return project_simulation_view(source, task)


def simulation_layers_for(source: object, task: str | None = None) -> SimulationLayerViews:
    """返回 A-F typed 视图，供 renderer/provider 逐层消费。

    该入口复用 ``simulation_view_for`` 的任务校验和空间能力门禁；它不会再次
    解析 YAML，也不会从 owner 的零散属性拼装配置。``Layer C`` 组合 acoustic
    与 background 两个已有只读切片，保持历史字段分组兼容。
    """

    view = simulation_view_for(source, task=task)
    semantic = getattr(view, "semantic", None)
    if semantic is None:
        raise ValueError(
            f"task simulation view {view.task!r} does not expose Layer B semantics"
        )
    shared = view.shared
    return SimulationLayerViews(
        task=view.task,
        layer_a=shared.layer_a,
        layer_b=semantic,
        layer_c=LayerCConfig(
            acoustic=shared.acoustic,
            background=shared.background,
        ),
        layer_d=shared.peak,
        layer_e=shared.channel,
        layer_f=shared.post_mix,
    )


__all__ = [
    "AECSimulationView",
    "BackgroundSimulationView",
    "ChannelSimulationView",
    "FrozenMapping",
    "GeometrySimulationView",
    "LayerAConfig",
    "LayerBConfig",
    "LayerCConfig",
    "LayerDConfig",
    "LayerEConfig",
    "LayerFConfig",
    "PostMixSimulationView",
    "SESimulationView",
    "SharedSimulationView",
    "SimulationLayerViews",
    "SpatialSimulationView",
    "TemporalSimulationView",
    "TaskSemanticSimulationView",
    "TSESimulationView",
    "SSSimulationView",
    "SSLSimulationView",
    "SSScenarioSimulationView",
    "TaskSimulationView",
    "project_simulation_view",
    "simulation_layers_for",
    "simulation_view_for",
]
