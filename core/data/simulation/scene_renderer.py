"""多语音源共享 Layer C–F；源图像、监督目标和 observation 始终分开。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from core.data.simulation.channel_effects import (
    aggregate_layer_e_results,
    apply_post_augmentations_channels,
    dispatch_observation_channels,
)
from core.data.simulation.contracts import apply_deterministic_peak_guard
from core.data.post_augment import (
    scale_source_volume_stem,
    source_volume_normalize,
    source_volume_shared_max_abs,
)
from core.data.post_mix_pipeline import PostMixStageResult
from core.data.simulation.background import mix_background
from core.data.simulation.energy_ratio import EnergyRatioScaler
from core.data.simulation.reference_resolver import resolve_sir_reference, resolve_snr_reference
from core.data.simulation.role_policy import (
    AcousticActor,
    AcousticRole,
    build_background_spec,
    validate_scene_actor_render_mode,
)
from core.data.simulation.scene_context import (
    SceneRenderContext,
    coerce_scene_render_context,
)
from core.data.simulation.scene_spec import SceneSpec
from core.data.simulation.scene_plan import ScenePlan
from core.data.simulation.array_profiles import (
    build_geometry_metadata,
    resolve_microphone_positions,
    resolve_scene_microphone_array,
)
from core.data.speech_scene import place_scene_sources, scene_activity_facts
from core.data.waveform_bundle import WaveformBundle
from core.data.waveform_scene_bundle import WaveformSceneBundle, WaveformSceneContract
from core.data.spatial import render_spatial_scene

def _resolve_observation_geometry(cfg, observation_geometry):
    """解析 scene observation geometry，禁止调用方修改共享 config。"""
    geometry_profile = cfg.array_profile
    geometry_custom = cfg.custom_geometry
    if observation_geometry is not None:
        geometry_profile = None
        geometry_custom = observation_geometry
        topology = (
            observation_geometry.get("topology", {})
            if isinstance(observation_geometry, Mapping)
            else {}
        )
        if not isinstance(topology, Mapping):
            topology = {}
        if isinstance(observation_geometry, Mapping):
            positions = observation_geometry.get("positions_m")
            if positions is None:
                positions = topology.get("positions_m")
            spacing = observation_geometry.get("spacing_m", topology.get("spacing_m"))
            coordinate_system = observation_geometry.get(
                "coordinate_system", topology.get("coordinate_system", "cartesian_xyz_m")
            )
            room_dim = observation_geometry.get("room_dim_m")
            if room_dim is None and isinstance(observation_geometry.get("room"), Mapping):
                room_dim = observation_geometry["room"].get("room_dim_m")
        else:
            positions = observation_geometry
            spacing = None
            coordinate_system = "cartesian_xyz_m"
            room_dim = None
        resolved = resolve_scene_microphone_array(
            positions_m=positions,
            room_dim_m=room_dim,
            spacing_m=spacing,
            coordinate_system=coordinate_system,
            reference_channel=cfg.reference_channel,
        )
        microphone_positions = resolved.positions_m
    else:
        microphone_positions = resolve_microphone_positions(
            array_profile=geometry_profile,
            custom_geometry=geometry_custom,
            reference_channel=cfg.reference_channel,
        )
    return geometry_profile, geometry_custom, microphone_positions


def _render_spatial_background(
    context,
    speech_sum,
    snr_reference,
    sample_rate,
    length,
    rng,
    need_noise,
    microphone_positions,
    background_spec,
):
    """执行 spatial background mix，并返回实际生效的 BackgroundSpec。"""
    context = coerce_scene_render_context(context)
    cfg = context.config
    resolved_spec = (
        background_spec
        if background_spec is not None
        else build_background_spec(
            image_geometry=microphone_positions,
            shared_scene=True,
        )
    )
    background = mix_background(
        context,
        speech_sum.copy(),
        snr_reference,
        sample_rate,
        length,
        length,
        rng,
        need_noise,
        float(rng.uniform(cfg.snr_low, cfg.snr_high)) if need_noise else None,
        image_geometry=microphone_positions,
        background_spec=resolved_spec,
    )
    return background, resolved_spec


def _render_mono_background(
    context,
    speech_sum,
    reference,
    sample_rate,
    length,
    rng,
    need_noise,
    background_spec,
    *,
    has_sources,
):
    """执行 mono scene 的 background mix，并保持既有 RNG/shape 语义。

    ``render_speech_scene`` 的 mono 路径同时支持有源和纯噪声场景。这个 helper
    只收拢编排，保留 source 分支在 ``need_noise=False`` 时仍抽取一次 SNR 的
    既有行为。
    """
    context = coerce_scene_render_context(context)
    cfg = context.config
    if not has_sources:
        if not need_noise:
            raise ValueError("scene must contain speech or background")
        speech_image = np.zeros((1, length), dtype=np.float32)
        reference_image = np.asarray(reference, dtype=np.float32).reshape(1, -1)
        effective_need_noise = True
    else:
        speech_image = np.asarray(speech_sum, dtype=np.float32).reshape(1, -1)
        reference_image = np.asarray(reference, dtype=np.float32).reshape(1, -1)
        effective_need_noise = bool(need_noise)
    resolved_spec = (
        background_spec
        if background_spec is not None
        else build_background_spec(image_geometry=None)
    )
    snr = float(rng.uniform(cfg.snr_low, cfg.snr_high))
    background = mix_background(
        context,
        speech_image,
        reference_image,
        sample_rate,
        length,
        length,
        rng,
        effective_need_noise,
        snr,
        background_spec=resolved_spec,
    )
    return background, resolved_spec


def _validate_speech_scene_actors(actors):
    """校验 speech scene actor 合同并投影 actor/source facts。"""
    actor_items = tuple(actors)
    if any(not isinstance(item, AcousticActor) for item in actor_items):
        raise TypeError("render_speech_scene requires AcousticActor entries")
    if len({actor.uid for actor in actor_items}) != len(actor_items):
        raise ValueError("AcousticActor.uid must be unique within a scene")
    allowed_roles = {
        AcousticRole.TARGET_SPEECH,
        AcousticRole.INTERFERENCE_SPEECH,
        AcousticRole.SOURCE_SPEECH,
        AcousticRole.NEAREND,
        AcousticRole.LOUDSPEAKER_ECHO,
    }
    invalid_roles = [
        actor.role.value for actor in actor_items if actor.role not in allowed_roles
    ]
    if invalid_roles:
        raise ValueError(f"roles {invalid_roles!r} cannot enter speech scene renderer")
    sources = tuple(actor.waveform for actor in actor_items)
    if sources and len({np.size(source) for source in sources}) != 1:
        raise ValueError("scene requires non-empty aligned speech sources")
    return actor_items, [actor.to_dict() for actor in actor_items], sources, tuple(
        actor.uid for actor in actor_items
    )


def _prepare_spatial_tracks(
    *,
    cfg,
    sources,
    source_uids,
    spatial_rirs,
    pre_rendered_spatial,
    expected_channels,
    length,
    sample_rate,
    rng,
):
    """校验 supplied/pre-rendered spatial 输入并生成 full/early 轨道。"""
    supplied_rirs = None if spatial_rirs is None else np.asarray(spatial_rirs, dtype=np.float32)
    if supplied_rirs is not None and (
        supplied_rirs.ndim != 3
        or supplied_rirs.shape[:2] != (len(sources), expected_channels)
    ):
        raise ValueError(
            "supplied spatial RIR must have shape [num_sources, num_channels, rir_samples]"
        )
    source_array = np.stack(
        [np.asarray(source, dtype=np.float32).reshape(-1) for source in sources]
    )
    use_reverb = bool(rng.random() < cfg.prob_reverberation)
    rts_tracks = None
    if pre_rendered_spatial is not None:
        use_reverb = True
        from core.data.simulation.spatial_scene import SpatialSceneResult
        from core.data.spatial import SpatialRenderResult

        if not isinstance(pre_rendered_spatial, (SpatialRenderResult, SpatialSceneResult)):
            raise TypeError("pre_rendered_spatial must be SpatialRenderResult or SpatialSceneResult")
        if pre_rendered_spatial.full_tracks.shape != (len(sources), expected_channels, length):
            raise ValueError("pre-rendered spatial tracks do not match source/channel/time axes")
        declared_order = pre_rendered_spatial.provenance.get("source_order")
        rendered_source_uids = getattr(pre_rendered_spatial, "source_uids", None)
        expected_order = tuple(source_uids)
        if declared_order is not None:
            order = tuple(declared_order)
            if len(order) != len(sources) or len(set(order)) != len(order):
                raise ValueError("pre-rendered spatial source_order must identify each source uniquely")
            if order != expected_order:
                raise ValueError("pre-rendered spatial source_order does not match sources")
        if isinstance(pre_rendered_spatial, SpatialSceneResult):
            if rendered_source_uids is None or len(tuple(rendered_source_uids)) != len(sources):
                raise ValueError("pre-rendered spatial source_uids must match sources")
            if tuple(rendered_source_uids) != tuple(source_uids):
                raise ValueError("pre-rendered spatial source_uids do not match sources")
            if pre_rendered_spatial.geometry and expected_channels != len(
                pre_rendered_spatial.geometry.get("positions_m", ())
            ):
                raise ValueError("pre-rendered spatial geometry channel count does not match request")
            if pre_rendered_spatial.trajectory is not None:
                if pre_rendered_spatial.timebase is None:
                    raise ValueError("pre-rendered moving scene requires trajectory timebase")
                if pre_rendered_spatial.timebase.num_samples != length:
                    raise ValueError("pre-rendered trajectory timebase does not match scene length")
        source_full_rirs = pre_rendered_spatial.full_tracks.copy()
        early_tracks = pre_rendered_spatial.early_tracks.copy()
        rts_tracks = getattr(pre_rendered_spatial, "rts_tracks", None)
    else:
        if cfg.target_type == "rts":
            raise ValueError(
                "RTS target requires a pre-rendered spatial scene with the sampled room RT60"
            )
        spatial = render_spatial_scene(
            source_array,
            supplied_rirs,
            early_boundary=max(0, int(round(cfg.early_rir_sec * sample_rate))),
            reference_channel=cfg.reference_channel,
            num_samples=length,
        )
        source_full_rirs = spatial.full_tracks.copy()
        early_tracks = spatial.early_tracks.copy()
        rts_tracks = spatial.rts_tracks
    return source_array, source_full_rirs, early_tracks, rts_tracks, use_reverb, supplied_rirs


@dataclass
class _MonoAcousticTrackPreparation:
    """mono speech scene 的 C 层轨道准备结果。

    该结果只包含 RIR/target 轨道和在线 RIR provenance；SIR 标定、background
    mix 以及 facts 组装仍由 ``render_speech_scene`` 负责。列表字段保持可变，
    以便调用方沿用原有的 SIR 缩放顺序和就地更新语义。
    """

    source_full_rirs: list[np.ndarray]
    targets: list[np.ndarray]
    plans: list[dict]
    online_facts: object | None
    online_moving: bool
    online_segment_sec: float | None
    online_normalization_peak: float | None
    latent_sir_full_rir: np.ndarray | None
    latent_sir_plan: dict | None


def _prepare_mono_acoustic_tracks(
    context,
    sources,
    source_uids,
    sample_rate,
    length,
    rng,
    *,
    sir_reference_speech,
):
    """准备 mono speech scene 的 online RIR、source image 和 target 轨道。

    helper 保持原 ``render_speech_scene`` 的随机数和临时 dataset 状态顺序：
    先抽样共享 online RIR group，再渲染 latent SIR reference，最后按 source
    顺序渲染 speech。SIR 缩放和 background mix 不在这里执行。
    """
    context = coerce_scene_render_context(context)
    runtime = context.acoustic
    cfg = context.config
    source_full_rirs = []
    targets = []
    plans = []
    online_group = None
    online_facts = None
    online_moving = False
    online_segment_sec = None
    online_normalization_peak = None
    latent_sir_full_rir = None
    latent_sir_plan = None
    latent_sir_count = 1 if sir_reference_speech is not None else 0
    if (sources or latent_sir_count) and cfg.rir_backend == "online" and rng.random() < cfg.prob_reverberation:
        if runtime.online_rir_provider is None:
            from core.data.simulation.rir_service import OnlineRIRError

            raise OnlineRIRError("selected speech scene requires an online provider initialized by the Dataset owner")
        source_count = len(sources) + latent_sir_count
        if (
            not cfg.spatial_enabled
            and cfg.prob_moving_rir > 0.0
            and rng.random() < cfg.prob_moving_rir
        ):
            speech_actor_uids = source_uids
            actor_uids = (
                ("sir_reference/target",) if latent_sir_count else ()
            ) + tuple(f"speech/{uid}" for uid in speech_actor_uids)
            moving_tracks = runtime.online_rir_provider.sample_moving_spatial_tracks(
                sample_rate,
                source_count,
                rng,
                num_samples=int(length),
                source_uids=actor_uids,
                normalize=cfg.rir_normalize,
                track_kinds=("full",),
            )
            online_group = np.asarray(moving_tracks.full, dtype=np.float32)
            online_facts = moving_tracks.facts
            online_moving = True
            online_segment_sec = (
                float(moving_tracks.plan["trajectory_timebase"]["hop_samples"])
                / float(sample_rate)
            )
            if cfg.rir_normalize:
                online_normalization_peak = float(np.max(np.abs(online_group)))
        else:
            online_group, online_facts = runtime.online_rir_provider.sample_scene(
                sample_rate, source_count, rng, normalize=cfg.rir_normalize,
            )

    def online_source_override(index):
        asset = None
        if online_group is not None:
            asset = online_group[:, index, :] if online_moving else online_group[index]
        return {
            "enabled": online_group is not None,
            "asset": asset,
            "facts": online_facts,
            "index": index,
            "moving": online_moving,
            "segment_sec": online_segment_sec,
            "normalization_peak": online_normalization_peak,
        }

    if sir_reference_speech is not None:
        if cfg.rir_backend == "online":
            with runtime.online_rir_source(online_source_override(0)):
                acoustic = runtime.mix_acoustic(
                    np.asarray(sir_reference_speech, dtype=np.float32).reshape(1, -1),
                    sample_rate,
                    length,
                    length,
                    rng,
                    False,
                    resolved_asset_role="sir_reference/target",
                )
        else:
            acoustic = runtime.mix_acoustic(
                np.asarray(sir_reference_speech, dtype=np.float32).reshape(1, -1),
                sample_rate,
                length,
                length,
                rng,
                False,
                resolved_asset_role="sir_reference/target",
            )
        latent_sir_full_rir = acoustic.noisy_after_rir.reshape(-1).astype(np.float32)
        latent_sir_plan = acoustic.acoustic_scene_plan.to_dict()

    for index, source in enumerate(sources):
        if cfg.rir_backend == "online":
            with runtime.online_rir_source(online_source_override(index + latent_sir_count)):
                acoustic = runtime.mix_acoustic(
                    np.asarray(source, dtype=np.float32).reshape(1, -1),
                    sample_rate,
                    length,
                    length,
                    rng,
                    False,
                    resolved_asset_role=f"source/{index}",
                )
        else:
            acoustic = runtime.mix_acoustic(
                np.asarray(source, dtype=np.float32).reshape(1, -1),
                sample_rate,
                length,
                length,
                rng,
                False,
                resolved_asset_role=f"source/{index}",
            )
        source_full_rirs.append(
            acoustic.noisy_after_rir.reshape(-1).astype(np.float32)
        )
        targets.append(acoustic.clean_target.reshape(-1).astype(np.float32))
        plans.append(acoustic.acoustic_scene_plan.to_dict())

    return _MonoAcousticTrackPreparation(
        source_full_rirs=source_full_rirs,
        targets=targets,
        plans=plans,
        online_facts=online_facts,
        online_moving=online_moving,
        online_segment_sec=online_segment_sec,
        online_normalization_peak=online_normalization_peak,
        latent_sir_full_rir=latent_sir_full_rir,
        latent_sir_plan=latent_sir_plan,
    )


def _build_spatial_scene_tracks(
    rendered_plan,
    background,
    source_full_rirs,
    targets,
):
    """从已经渲染的 spatial 轨道构造 task-neutral scene track map。

    该 helper 只组合 role 到 waveform 的映射，不读取配置、不消费 RNG，也不
    改写任何输入。背景 observation、background additive、full-RIR source 和
    target 轨道的 role 命名由 ``ScenePlan`` 统一提供。
    """
    tracks = {
        rendered_plan.observation_role: background.observation,
        rendered_plan.background_role: background.additive,
    }
    tracks.update(
        {
            rendered_plan.source_full_rir_role(index): source_full_rirs[index]
            for index in range(len(source_full_rirs))
        }
    )
    tracks.update(
        {
            rendered_plan.target_role(index): targets[index]
            for index in range(len(targets))
        }
    )
    return tracks


def _build_mono_scene_tracks(
    rendered_plan,
    background,
    source_full_rirs,
    targets,
):
    """从已经渲染的 mono 轨道构造 task-neutral scene track map。"""
    tracks = {
        rendered_plan.observation_role: background.observation.reshape(-1),
        rendered_plan.background_role: background.additive.reshape(-1),
    }
    tracks.update(
        {
            rendered_plan.source_full_rir_role(index): full_rir
            for index, full_rir in enumerate(source_full_rirs)
        }
    )
    tracks.update(
        {
            rendered_plan.target_role(index): target
            for index, target in enumerate(targets)
        }
    )
    return tracks


def _build_spatial_scene_facts(
    *,
    cfg,
    actor_facts,
    speech_sum,
    snr_reference,
    background_additive,
    background,
    supplied_rirs,
    use_reverb,
    geometry,
    sir_facts,
    source_volume_calibration,
    resolved_background_spec,
    rendered_plan,
    sample_rate,
    length,
    pre_rendered_spatial,
    timeline,
):
    """投影 spatial scene provenance/facts。

    该函数只基于已完成的声学轨道和 provenance 计算 JSON-safe facts，不执行
    背景、SIR、target 或 post-mix 操作，因此不会改变主 renderer 的随机数和
    异常顺序。
    """
    speech_reference = snr_reference[cfg.reference_channel]
    background_reference = background_additive[cfg.reference_channel]
    speech_power = float(np.mean(speech_reference.astype(np.float64) ** 2))
    background_power = float(np.mean(background_reference.astype(np.float64) ** 2))
    rendered_reference_snr = (
        None
        if speech_power <= 1e-12 or background_power <= 1e-12
        else float(10.0 * np.log10(speech_power / background_power))
    )
    full_speech_power = float(np.sum(snr_reference.astype(np.float64) ** 2))
    full_noise_power = float(np.sum(background_additive.astype(np.float64) ** 2))
    rendered_image_snr = (
        float(10 * np.log10(full_speech_power / full_noise_power))
        if full_speech_power > 1e-12 and full_noise_power > 1e-12
        else None
    )
    facts = {
        "version": 1,
        "spatial": {
            "enabled": True,
            "num_channels": int(speech_sum.shape[0]),
            "reference_channel": cfg.reference_channel,
            # 预渲染入口只有 source image，不把 image shape 冒充 RIR shape。
            "rir_shape": list(supplied_rirs.shape) if supplied_rirs is not None else None,
            "reverberation_applied": use_reverb,
            "geometry": geometry,
        },
        "geometry": geometry,
        "geometry_provenance": geometry,
        "sir": sir_facts,
        "sir_measure_point": "post_rir_pre_background",
        "sir_policy": "full_sum_energy",
        "source_volume_calibration": source_volume_calibration or None,
        "background_snr_values": (
            [] if rendered_image_snr is None else [rendered_image_snr]
        ),
        "background_snr_nominal_values": list(background.snr_values),
        "background_snr_measure_point": "post_render_full_image",
        "background_snr_reference_channel_db": rendered_reference_snr,
        "noise_image_provenance": background.noise_image_provenance,
        "background_policy": resolved_background_spec.to_dict(),
        "wind_background_applied": bool(background.used_wind_background),
        "rendered_scene_plan": rendered_plan.to_dict(),
        "sample_rate": int(sample_rate),
        "num_samples": int(length),
        "decomposable_before_post_mix": not background.used_sidechain_wind,
        "acoustic_actors": actor_facts,
        "timeline": timeline,
    }
    if pre_rendered_spatial is not None:
        # 保留预渲染 provider 的完整 provenance，避免 bundle 构建丢失 source
        # order、trajectory 和 room geometry 等身份信息。
        facts["spatial"]["pre_rendered_provenance"] = dict(
            pre_rendered_spatial.provenance
        )
    return facts


def _build_mono_scene_facts(
    *,
    cfg,
    actor_facts,
    source_full_rirs,
    plans,
    latent_sir_plan,
    online_facts,
    timeline,
    sir_facts,
    sir_reference_source,
    reference_facts,
    background,
    record,
    rendered_plan,
    sample_rate,
    length,
    sources,
):
    """投影 mono scene provenance/facts，不触碰声学轨道。"""
    return {
        "version": 1,
        "rir_backend": cfg.rir_backend,
        "source_plans": plans,
        "sir_reference_plan": latent_sir_plan,
        "online_rir_scene": online_facts,
        "timeline": timeline,
        "activity": (
            scene_activity_facts(tuple(source_full_rirs))
            if source_full_rirs
            else {
                "active_ratios": [],
                "overlap_ratio": 0.0,
                "active_source_count": 0,
            }
        ),
        "sir": sir_facts,
        "sir_measure_point": "post_rir_pre_background",
        "sir_policy": "full_sum_energy",
        "sir_reference_source": sir_reference_source,
        "reference_facts": reference_facts,
        "snr_reference": cfg.snr_reference if sources else "latent_speech",
        "snr_reference_track": "full_rir" if sources else "latent_speech",
        "background_snr_values": background.snr_values,
        "rendered_scene_plan": rendered_plan.to_dict(),
        "sample_rate": int(sample_rate),
        "num_samples": int(length),
        "noise_sources": [source.to_dict() for source in record["noise_sources"]],
        "background_audit": background.audit,
        "wind_background_applied": bool(background.used_wind_background),
        "decomposable_before_post_mix": not background.used_sidechain_wind,
        "acoustic_actors": actor_facts,
    }


@dataclass(frozen=True)
class SpeechScene:
    bundle: WaveformBundle
    facts: dict
    used_sidechain_wind: bool

    def to_scene_bundle(self) -> WaveformSceneBundle:
        """将共享 scene 投影到 task-neutral role bundle 并执行统一合同校验。"""
        return WaveformSceneBundle.from_tracks(
            self.bundle.tracks,
            metadata=self.bundle.metadata,
            contract=WaveformSceneContract(
                required_roles=("mixture", "background"),
                observation_role="mixture",
            ),
        )


@dataclass(frozen=True, slots=True)
class _PreparedSpeechScene:
    """actor 校验与时间窗放置完成后的共享 speech scene 状态。"""

    context: SceneRenderContext
    actor_facts: list[dict]
    sources: tuple[np.ndarray, ...]
    source_uids: tuple[str, ...]
    timeline: object
    length: int
    geometry_profile: object
    geometry_custom: object
    microphone_positions: np.ndarray | None


def _prepare_speech_scene(
    dataset,
    actors,
    rng,
    *,
    sir_mode,
    need_noise,
    num_samples,
    spatial_rirs,
    pre_rendered_spatial,
    sources_preplaced,
    timeline,
    background_spec,
    observation_geometry,
) -> _PreparedSpeechScene:
    """执行 speech scene 的公共合同校验、actor 投影和 source placement。"""
    context = coerce_scene_render_context(dataset)
    actor_items, actor_facts, sources, source_uids = _validate_speech_scene_actors(actors)
    if sir_mode not in {"aggregate", "per_source", "none"}:
        raise ValueError("sir_mode must be aggregate, per_source, or none")
    cfg = context.config
    geometry_profile, geometry_custom, resolved_geometry = _resolve_observation_geometry(
        cfg, observation_geometry
    )
    validate_scene_actor_render_mode(actor_items, spatial_enabled=bool(cfg.spatial_enabled))
    if cfg.spatial_enabled and spatial_rirs is None and pre_rendered_spatial is None:
        raise ValueError(
            "spatial_enabled=True requires an explicit supplied source-by-microphone RIR bank"
        )
    if not cfg.spatial_enabled and spatial_rirs is not None:
        raise ValueError("spatial_rirs requires spatial_enabled=True")
    if spatial_rirs is not None and cfg.rir_backend not in {"offline", "online"}:
        raise ValueError("spatial_rirs requires rir_backend='offline' or 'online'")
    if pre_rendered_spatial is not None and (not sources_preplaced or not cfg.spatial_enabled):
        raise ValueError('pre-rendered spatial tracks require preplaced spatial sources')
    if sources_preplaced:
        sources = tuple(sources)
        timeline = () if timeline is None else timeline
    else:
        if timeline is not None:
            raise ValueError("timeline can only be supplied with sources_preplaced=True")
        temporal = getattr(
            getattr(context.simulation_view, "shared", None),
            "temporal",
            cfg,
        )
        sources, timeline = place_scene_sources(sources, temporal, rng)
    length = np.size(sources[0]) if sources else num_samples
    if not length or length < 1:
        raise ValueError("empty scene requires positive num_samples")
    if spatial_rirs is not None or pre_rendered_spatial is not None:
        if (
            need_noise
            and float(getattr(cfg, "prob_point_source_background", 0.0)) > 0.0
            and background_spec is None
        ):
            raise ValueError(
                "spatial background point-source policy must be resolved before "
                "render_speech_scene when noise is enabled"
            )

    return _PreparedSpeechScene(
        context=context,
        actor_facts=actor_facts,
        sources=sources,
        source_uids=source_uids,
        timeline=timeline,
        length=length,
        geometry_profile=geometry_profile,
        geometry_custom=geometry_custom,
        microphone_positions=resolved_geometry,
    )


def calibrate_spatial_source_volume(full_tracks, early_tracks, targets, cfg, sample_rate, rng):
    """按 early reference 标定一次，共同缩放 [K,C,T] 图像与 [K,T] 目标。

    不修改输入；关闭时原样返回且不消费随机数。启用时沿用共享有限值检查、
    三类 stem 的共同峰值保护与 post-RIR 标定事实。
    """
    if not cfg.source_volume_normalization_enabled:
        return full_tracks, early_tracks, targets, []
    full_tracks = full_tracks.copy()
    early_tracks = early_tracks.copy()
    targets = targets.copy()
    calibrations = []
    for index in range(len(full_tracks)):
        calibration = {}
        reference = early_tracks[index, cfg.reference_channel]
        source_volume_normalize(
            reference,
            cfg.source_volume_db_range[0],
            cfg.source_volume_db_range[1],
            cfg.source_volume_use_rms,
            sample_rate,
            rng=rng,
            distribution=cfg.source_volume_distribution,
            beta_a=cfg.source_volume_beta_a,
            beta_b=cfg.source_volume_beta_b,
            shared_stem_max_abs=source_volume_shared_max_abs(
                full_tracks[index], early_tracks[index], targets[index]
            ),
            calibration_out=calibration,
        )
        gain = float(calibration["gain"])
        full_tracks[index] = scale_source_volume_stem(
            full_tracks[index], gain, stem_name="full_rir"
        )
        early_tracks[index] = scale_source_volume_stem(
            early_tracks[index], gain, stem_name="early_rir"
        )
        targets[index] = scale_source_volume_stem(
            targets[index], gain, stem_name="clean_target"
        )
        calibration["stage"] = "post_rir_pre_noise_echo"
        calibrations.append(calibration)
    return full_tracks, early_tracks, targets, calibrations


def _render_spatial_speech_scene(
    prepared: _PreparedSpeechScene,
    sample_rate,
    rng,
    *,
    sir_range,
    sir_mode,
    need_noise,
    sir_reference_speech,
    spatial_rirs,
    pre_rendered_spatial,
    background_spec,
    observation_geometry,
):
    """渲染 spatial source/RIR、SIR、background 与 scene bundle。"""
    context = prepared.context
    cfg = context.config
    sources = prepared.sources
    source_uids = prepared.source_uids
    actor_facts = prepared.actor_facts
    length = prepared.length
    timeline = prepared.timeline
    microphone_positions = prepared.microphone_positions
    if sir_reference_speech is not None or not cfg.spatial_enabled:
        raise ValueError(
            "supplied spatial RIR requires spatial_enabled=true and no latent SIR reference"
        )
    if len(sources) < 1:
        raise ValueError("supplied spatial RIR requires at least one source")
    if sir_mode != "none" and sir_range is None:
        raise ValueError("spatial SIR requires an explicit sir_range")
    expected_channels = len(microphone_positions) if microphone_positions is not None else None
    if expected_channels is None:
        raise ValueError("spatial scene requires resolved microphone geometry")
    (
        source_array,
        source_full_rirs,
        early_tracks,
        rts_tracks,
        use_reverb,
        supplied_rirs,
    ) = _prepare_spatial_tracks(
        cfg=cfg,
        sources=sources,
        source_uids=source_uids,
        spatial_rirs=spatial_rirs,
        pre_rendered_spatial=pre_rendered_spatial,
        expected_channels=expected_channels,
        length=length,
        sample_rate=sample_rate,
        rng=rng,
    )

    if cfg.target_type == "dry":
        targets = source_array.copy()
    elif cfg.target_type == "rts":
        if rts_tracks is None:
            raise ValueError("RTS target was requested but pre-rendered spatial result has no rts_tracks")
        targets = rts_tracks[:, cfg.reference_channel].copy()
    else:
        targets = early_tracks[:, cfg.reference_channel].copy()
    source_full_rirs, early_tracks, targets, source_volume_calibration = (
        calibrate_spatial_source_volume(
            source_full_rirs, early_tracks, targets, cfg, sample_rate, rng
        )
    )

    sir_facts = []
    if len(sources) > 1 and sir_mode != "none":
        interference_indices = tuple(range(1, len(sources)))
        groups = (
            [interference_indices]
            if sir_mode == "aggregate"
            else [(index,) for index in interference_indices]
        )
        # SIR 必须在完整 microphone observation 上标定；只看 reference
        # channel 会把其它通道有效的 source 错判为 inactive。
        numerator = EnergyRatioScaler.measure(
            source_full_rirs[0], policy="full_sum_energy"
        )
        for group in groups:
            denominator = EnergyRatioScaler.measure(
                source_full_rirs[list(group)].sum(axis=0),
                policy="full_sum_energy",
            )
            selected = float(rng.uniform(*sir_range))
            if numerator <= 1e-10 or denominator <= 1e-10:
                sir_facts.append({
                    "sources": list(group), "reference": "target", "selected_db": selected,
                    "realized_db": None, "reason": "inactive_source",
                })
                continue
            result = EnergyRatioScaler.sum_energy_ratio(numerator, denominator, selected)
            for index in group:
                source_full_rirs[index] *= np.float32(result.gain)
                early_tracks[index] *= np.float32(result.gain)
                targets[index] *= np.float32(result.gain)
            sir_facts.append({
                "sources": list(group), "reference": "target", "selected_db": selected,
                "realized_db": result.realized_db, "gain": result.gain, "reason": None,
            })

    speech_sum = source_full_rirs.sum(axis=0, dtype=np.float32)
    snr_reference = source_full_rirs[0] if cfg.snr_reference == "primary" else speech_sum
    background, resolved_background_spec = _render_spatial_background(
        context,
        speech_sum,
        snr_reference,
        sample_rate,
        length,
        rng,
        need_noise,
        microphone_positions,
        background_spec,
    )
    background_additive = background.additive
    if background_additive.shape != speech_sum.shape:
        raise ValueError(
            f"spatial background shape {background_additive.shape} does not match speech {speech_sum.shape}"
        )
    scene_spec = SceneSpec(
        "reference_scene", True, len(sources) > 1, need_noise, False,
        len(sources), task=None,
    )
    rendered_plan = ScenePlan(scene_spec)
    tracks = _build_spatial_scene_tracks(
        rendered_plan, background, source_full_rirs, targets
    )
    geometry = build_geometry_metadata(
        array_profile=prepared.geometry_profile,
        custom_geometry=prepared.geometry_custom,
        reference_channel=cfg.reference_channel,
        num_channels=expected_channels,
        external=cfg.geometry_provenance,
        validation_recipe=cfg.validation_recipe,
        scene_geometry=observation_geometry is not None,
    )
    if pre_rendered_spatial is not None:
        # RIR 的 scene 几何使用房间坐标；不能用配置中的相对拓扑覆盖它。
        geometry = {**geometry, **dict(pre_rendered_spatial.geometry)}
    facts = _build_spatial_scene_facts(
        cfg=cfg,
        actor_facts=actor_facts,
        speech_sum=speech_sum,
        snr_reference=snr_reference,
        background_additive=background_additive,
        background=background,
        supplied_rirs=supplied_rirs,
        use_reverb=use_reverb,
        geometry=geometry,
        sir_facts=sir_facts,
        source_volume_calibration=source_volume_calibration,
        resolved_background_spec=resolved_background_spec,
        rendered_plan=rendered_plan,
        sample_rate=sample_rate,
        length=length,
        pre_rendered_spatial=pre_rendered_spatial,
        timeline=timeline,
    )
    return SpeechScene(WaveformBundle(tracks), facts, background.used_sidechain_wind)


def _render_mono_speech_scene(
    prepared: _PreparedSpeechScene,
    sample_rate,
    rng,
    *,
    sir_range,
    sir_mode,
    need_noise,
    reference_speech,
    sir_reference_speech,
    background_spec,
):
    """渲染 mono acoustic tracks、SIR、background 与 scene bundle。"""
    context = prepared.context
    cfg = context.config
    sources = prepared.sources
    source_uids = prepared.source_uids
    actor_facts = prepared.actor_facts
    timeline = prepared.timeline
    length = prepared.length
    acoustic_tracks = _prepare_mono_acoustic_tracks(
        context,
        sources,
        source_uids,
        sample_rate,
        length,
        rng,
        sir_reference_speech=sir_reference_speech,
    )
    source_full_rirs = acoustic_tracks.source_full_rirs
    targets = acoustic_tracks.targets
    plans = acoustic_tracks.plans
    online_facts = acoustic_tracks.online_facts
    latent_sir_full_rir = acoustic_tracks.latent_sir_full_rir
    latent_sir_plan = acoustic_tracks.latent_sir_plan
    sir_facts = []
    sir_reference_source = "latent_target" if latent_sir_full_rir is not None else "target"
    sir_numerator_full_rir = (
        latent_sir_full_rir
        if latent_sir_full_rir is not None
        else (source_full_rirs[0] if source_full_rirs else None)
    )
    interference_indices = (
        tuple(range(len(source_full_rirs)))
        if latent_sir_full_rir is not None
        else tuple(range(1, len(source_full_rirs)))
    )
    if sir_numerator_full_rir is not None and interference_indices and sir_mode != "none":
        groups = [interference_indices] if sir_mode == "aggregate" else [(i,) for i in interference_indices]
        numerator = EnergyRatioScaler.measure(
            sir_numerator_full_rir, policy="full_sum_energy"
        )
        for group in groups:
            denominator = EnergyRatioScaler.measure(
                sum(source_full_rirs[i] for i in group),
                policy="full_sum_energy",
            )
            selected = float(rng.uniform(*sir_range))
            if numerator <= 1e-10 or denominator <= 1e-10:
                sir_facts.append({"sources": list(group), "reference": sir_reference_source, "selected_db": selected, "realized_db": None, "reason": "inactive_source"})
                continue
            result = EnergyRatioScaler.sum_energy_ratio(numerator, denominator, selected)
            for index in group:
                source_full_rirs[index] = (
                    source_full_rirs[index] * result.gain
                ).astype(np.float32)
                targets[index] = (targets[index] * result.gain).astype(np.float32)
            sir_facts.append({"sources": list(group), "reference": sir_reference_source, "selected_db": selected, "realized_db": result.realized_db, "gain": result.gain, "reason": None})
    speech_sum = (
        np.sum(source_full_rirs, axis=0, dtype=np.float32)
        if source_full_rirs
        else np.zeros(length, dtype=np.float32)
    )
    reference = (
        source_full_rirs[0]
        if source_full_rirs and cfg.snr_reference == "primary"
        else speech_sum
    )
    record = {"noise_sources": [], "noise_source_policies": [], "noise_assets": [],
              "colored_enabled": False, "colored_type": None, "colored_snr_db": None, "mix_ratio": None}
    if not sources and need_noise:
        if reference_speech is None:
            raise ValueError(
                "speech-free scene requires a clean speech reference for SNR scaling"
            )
        reference = np.asarray(reference_speech, dtype=np.float32).reshape(-1)
        if reference.size != length:
            raise ValueError(
                "speech-free scene SNR reference must match the scene canvas length"
            )
    with context.acoustic.active_acoustic_plan_record(record):
        background, _ = _render_mono_background(
            context,
            speech_sum,
            reference,
            sample_rate,
            length,
            rng,
            need_noise,
            background_spec,
            has_sources=bool(sources),
        )
    reference_scene = SceneSpec(
        # 这是 SNR/SIR resolver 的 task-neutral reference，不是 Layer B scene。
        scene_name="reference_scene",
        target_present=bool(sources) and sir_reference_speech is None,
        interference_present=bool(
            latent_sir_full_rir is not None or len(source_full_rirs) > 1
        ),
        background_present=need_noise,
        echo_present=False,
        source_count=len(source_full_rirs),
        task=None,
    )
    rendered_plan = ScenePlan(reference_scene)
    tracks = _build_mono_scene_tracks(
        rendered_plan, background, source_full_rirs, targets
    )
    # timeline 的零窗口允许产生无活动源；不把合法静音误报成资源错误。
    sir_sources = source_full_rirs if latent_sir_full_rir is not None else source_full_rirs[1:]
    sir_active = (
        sir_numerator_full_rir is not None and np.any(sir_numerator_full_rir)
        and bool(sir_sources) and np.any(np.sum(sir_sources, axis=0))
    )
    snr_active = bool(np.any(reference))
    reference_facts = {
        "snr": resolve_snr_reference(
            reference_scene,
            [reference] if source_full_rirs else None,
            [reference] if not source_full_rirs else None,
        ) if snr_active else None,
        "sir": resolve_sir_reference(
            reference_scene, sir_numerator_full_rir, sir_sources,
        ) if sir_active and sir_mode != "none" else None,
        "snr_unavailable_reason": "inactive_reference" if need_noise and not snr_active else None,
        "sir_unavailable_reason": "inactive_source" if reference_scene.interference_present and not sir_active else None,
    }
    facts = _build_mono_scene_facts(
        cfg=cfg,
        actor_facts=actor_facts,
        source_full_rirs=source_full_rirs,
        plans=plans,
        latent_sir_plan=latent_sir_plan,
        online_facts=online_facts,
        timeline=timeline,
        sir_facts=sir_facts,
        sir_reference_source=sir_reference_source,
        reference_facts=reference_facts,
        background=background,
        record=record,
        rendered_plan=rendered_plan,
        sample_rate=sample_rate,
        length=length,
        sources=sources,
    )
    scene = SpeechScene(WaveformBundle(tracks), facts, background.used_sidechain_wind)
    scene.to_scene_bundle()
    return scene


def render_speech_scene(
    dataset,
    actors,
    sample_rate,
    rng,
    *,
    sir_range,
    sir_mode="aggregate",
    need_noise=True,
    num_samples=None,
    reference_speech=None,
    sir_reference_speech=None,
    spatial_rirs=None,
    pre_rendered_spatial=None,
    sources_preplaced=False,
    timeline=None,
    background_spec=None,
    observation_geometry=None,
):
    """按显式声学角色渲染 speech scene，并产出 ``observation [C,T]``。

    supplied RIR 要求显式 source×microphone bank；背景噪声仍通过 shared background seam
    混入每个 observation channel，任务 target 始终投影为 reference channel。
    ``background_spec`` 可显式注入 point-source RIR provider；未提供时按
    observation geometry 使用 diffuse 或 mono 默认策略；空间噪声开启且配置要求
    point-source background 时，调用方必须先解析并传入该 spec。
    ``observation_geometry`` 可显式注入已解析的 scene microphone geometry；提供后
    不会读取或修改 dataset config 中的 ``array_profile/custom_geometry``。
    ``timeline`` 只允许随 ``sources_preplaced=True`` 传入；renderer 只发布该
    已完成的 source-window plan，不会再次抽样或重放 timeline。
    ``actors`` 必须是进入该 scene 的完整 ``AcousticActor`` 序列；source UID、角色和
    RIR policy 均从 actor 派生，不能通过旁路参数覆盖。
    """
    prepared = _prepare_speech_scene(
        dataset,
        actors,
        rng,
        sir_mode=sir_mode,
        need_noise=need_noise,
        num_samples=num_samples,
        spatial_rirs=spatial_rirs,
        pre_rendered_spatial=pre_rendered_spatial,
        sources_preplaced=sources_preplaced,
        timeline=timeline,
        background_spec=background_spec,
        observation_geometry=observation_geometry,
    )
    if spatial_rirs is not None or pre_rendered_spatial is not None:
        return _render_spatial_speech_scene(
            prepared,
            sample_rate,
            rng,
            sir_range=sir_range,
            sir_mode=sir_mode,
            need_noise=need_noise,
            sir_reference_speech=sir_reference_speech,
            spatial_rirs=spatial_rirs,
            pre_rendered_spatial=pre_rendered_spatial,
            background_spec=background_spec,
            observation_geometry=observation_geometry,
        )
    return _render_mono_speech_scene(
        prepared,
        sample_rate,
        rng,
        sir_range=sir_range,
        sir_mode=sir_mode,
        need_noise=need_noise,
        reference_speech=reference_speech,
        sir_reference_speech=sir_reference_speech,
        background_spec=background_spec,
    )


class SourceScenePostMixAdapter:
    """SS/TSE 多源场景的 D–F Adapter；全部源共用 D 增益，E/F 仅改变 observation。"""

    def __init__(self, dataset, sample_rate, rng, *, used_sidechain_wind=False):
        self.context = coerce_scene_render_context(dataset)
        self.sample_rate, self.rng = sample_rate, rng
        self.used_sidechain_wind = used_sidechain_wind
        self.facts = {}
        self.layer_e = None
        self._layer_e_results = ()

    def _spatial_observation_roles(self, tracks):
        """返回 spatial observation 侧 ``[C,T]`` roles。"""
        return tuple(
            role for role, waveform in tracks.items()
            if (
                role == "mixture"
                or role == "background"
                or role.startswith("source_full_rir/")
            )
            and np.asarray(waveform).ndim == 2
        )

    @staticmethod
    def _spatial_target_roles(tracks):
        """返回与 reference channel 对应的 mono target roles。"""
        return tuple(
            role
            for role, waveform in tracks.items()
            if role.startswith("target/") and np.asarray(waveform).ndim == 1
        )

    def _spatial_peak_normalize(self, tracks, view):
        roles = self._spatial_observation_roles(tracks)
        target_roles = self._spatial_target_roles(tracks)
        policy = view.spatial.spatial_layer_d_policy
        reference_channel = int(view.spatial.reference_channel)
        base_facts = {
            "mode": view.peak.peak_normalization_mode,
            "policy": policy,
            "reference_channel": reference_channel,
            "target_roles": list(target_roles),
            "preserves_decomposition": True,
        }
        if not roles or np.asarray(tracks["mixture"]).ndim != 2:
            return {}, {**base_facts, "mode": "off", "scale": 1.0, "target_scale": 1.0}
        if view.peak.peak_normalization_mode == "off":
            return {}, {**base_facts, "scale": 1.0, "target_scale": 1.0}
        arrays = {role: np.asarray(tracks[role], dtype=np.float32) for role in roles}
        targets = {
            role: np.asarray(tracks[role], dtype=np.float32) for role in target_roles
        }
        target_peak = max(
            (float(np.max(np.abs(array))) for array in targets.values()),
            default=0.0,
        )
        if policy == "shared":
            peak = max(
                *(float(np.max(np.abs(array))) for array in arrays.values()),
                target_peak,
                1e-8,
            )
            scale = float(
                self.rng.uniform(*view.peak.peak_normalization_scale_range)
            ) / peak
            scales = np.asarray([scale], dtype=np.float32)
        elif policy == "per_channel":
            peaks = np.maximum.reduce(
                [np.max(np.abs(array), axis=-1) for array in arrays.values()]
            )
            peaks[reference_channel] = max(peaks[reference_channel], target_peak)
            desired = self.rng.uniform(
                *view.peak.peak_normalization_scale_range, size=peaks.shape
            )
            scales = (desired / np.maximum(peaks, 1e-8)).astype(np.float32)
        else:
            raise ValueError(f"unsupported spatial_layer_d_policy: {policy!r}")
        channel_scale = scales if scales.size > 1 else np.repeat(scales, view.spatial.channels)
        changes = {
            role: (array * channel_scale.reshape(-1, 1)).astype(np.float32)
            for role, array in arrays.items()
        }
        reference_scale = float(channel_scale[reference_channel])
        changes.update(
            {
                role: (array * np.float32(reference_scale)).astype(np.float32)
                for role, array in targets.items()
            }
        )
        return changes, {
            **base_facts,
            "scale": float(scales[0]) if scales.size == 1 else scales.tolist(),
            "target_scale": reference_scale,
        }

    def _spatial_peak_guard(self, tracks, roles, policy, *, reference_channel):
        target_roles = self._spatial_target_roles(tracks)
        coupled_roles = (*roles, *target_roles)
        if policy == "shared":
            result = apply_deterministic_peak_guard(tracks, roles=coupled_roles)
            facts = result.facts.to_metadata()
            facts.update(
                {
                    "policy": policy,
                    "reference_channel": reference_channel,
                    "target_roles": list(target_roles),
                    "target_scale": float(facts["scale"]),
                    "preserves_decomposition": True,
                }
            )
            return {
                role: result.waveforms[role] for role in coupled_roles
            }, facts
        if policy != "per_channel":
            raise ValueError(f"unsupported spatial layer D policy: {policy!r}")
        threshold = np.float32(0.99)
        arrays = {role: np.asarray(tracks[role], dtype=np.float32) for role in roles}
        targets = {
            role: np.asarray(tracks[role], dtype=np.float32) for role in target_roles
        }
        peaks = np.maximum.reduce(
            [np.max(np.abs(array), axis=-1) for array in arrays.values()]
        )
        target_peak = max(
            (float(np.max(np.abs(array))) for array in targets.values()),
            default=0.0,
        )
        peaks[reference_channel] = max(peaks[reference_channel], target_peak)
        scales = np.minimum(1.0, threshold / np.maximum(peaks, 1e-8)).astype(
            np.float32
        )
        changes = {
            role: (array * scales[:, None]).astype(np.float32)
            for role, array in arrays.items()
        }
        reference_scale = float(scales[reference_channel])
        changes.update(
            {
                role: (array * np.float32(reference_scale)).astype(np.float32)
                for role, array in targets.items()
            }
        )
        return changes, {
            "roles": list(coupled_roles),
            "threshold": float(threshold),
            "peak": float(np.max(peaks, initial=0.0)),
            "scale": scales.tolist(),
            "applied": bool(np.any(scales < 1.0)),
            "policy": policy,
            "reference_channel": reference_channel,
            "target_roles": list(target_roles),
            "target_scale": reference_scale,
            "preserves_decomposition": True,
        }

    def apply_stage(self, stage, state):
        tracks = state.waveforms
        changes = {}
        cfg = self.context.config
        if stage == "D-normalize":
            view = self.context.simulation_view.shared
            if cfg.peak_normalization_mode not in {"off", "joint"}:
                raise ValueError("multi-source scene supports only off/joint peak normalization")
            if view.spatial.spatial_enabled and np.asarray(tracks["mixture"]).ndim == 2:
                changes, normalize_facts = self._spatial_peak_normalize(tracks, view)
            else:
                scale = 1.0
                if cfg.peak_normalization_mode == "joint":
                    peak = max(float(np.max(np.abs(track))) for track in tracks.values())
                    desired = float(self.rng.uniform(*cfg.peak_normalization_scale_range))
                    scale = desired / max(peak, 1e-8)
                    changes = {role: (track * scale).astype(np.float32) for role, track in tracks.items()}
                normalize_facts = {
                    "mode": cfg.peak_normalization_mode,
                    "scale": scale,
                    "preserves_decomposition": True,
                }
            self.facts["normalize"] = normalize_facts
        elif stage in {"D-guard", "final-guard"}:
            view = self.context.simulation_view.shared
            if stage == "D-guard" and view.spatial.spatial_enabled and np.asarray(tracks["mixture"]).ndim == 2:
                roles = self._spatial_observation_roles(tracks)
                changes, guard_facts = self._spatial_peak_guard(
                    tracks,
                    roles,
                    view.spatial.spatial_layer_d_policy,
                    reference_channel=int(view.spatial.reference_channel),
                )
            else:
                roles = tuple(tracks) if stage == "D-guard" else ("mixture",)
                result = apply_deterministic_peak_guard(tracks, roles=roles)
                changes = {role: result.waveforms[role] for role in roles}
                guard_facts = result.facts.to_metadata()
            self.facts[stage] = guard_facts
        elif stage == "E":
            mixture = tracks["mixture"]
            spatial = self.context.simulation_view.shared.spatial
            policy = spatial.spatial_layer_e_policy if mixture.ndim == 2 else "shared"
            dispatch = dispatch_observation_channels(
                mixture,
                policy=policy,
                rng=self.rng,
                apply_channel=lambda channel, channel_rng, _index: self.context.acoustic.apply_channel_degradation(
                    channel, self.sample_rate, mixture.shape[-1], channel_rng,
                    self.used_sidechain_wind,
                ),
            )
            self._layer_e_results = tuple(dispatch.channel_results)
            self.layer_e = aggregate_layer_e_results(dispatch)
            changes["mixture"] = self.layer_e.waveform
            self.facts["E"] = self.layer_e.to_metadata()
            self.facts["E"]["channel_facts"] = list(dispatch.channel_facts)
            if spatial.spatial_enabled and mixture.ndim == 2:
                self.facts["E"]["policy"] = spatial.spatial_layer_e_policy
        elif stage == "F":
            if self.layer_e is None:
                raise RuntimeError("Layer F requires completed Layer E")
            shared = self.context.simulation_view.shared
            view = shared.post_mix
            mixture = tracks["mixture"]
            policy = shared.spatial.spatial_layer_f_policy if mixture.ndim == 2 else "shared"
            result = apply_post_augmentations_channels(
                mixture,
                self.sample_rate,
                view,
                policy=policy,
                rng=self.rng,
                layer_e_results=self._layer_e_results,
                sanitize_nonfinite=not self.context.strict_load,
            )
            changes["mixture"] = result.waveform
            self.facts["F"] = {
                "noisy_post_mix_degradations_enabled": view.noisy_post_mix_degradations_enabled,
                "applied_ops": list(result.applied_ops),
                "channel_facts": list(result.channel_facts),
                "clipping_facts": result.clipping_facts,
                "observation_changed": not np.array_equal(tracks["mixture"], changes["mixture"]),
                "clipping_skipped_due_to_layer_e": any(
                    bool(facts.get("skip_clip", False)) for facts in result.channel_facts
                ),
            }
            if shared.spatial.spatial_enabled and mixture.ndim == 2:
                self.facts["F"]["policy"] = shared.spatial.spatial_layer_f_policy
        elif stage not in {"pre-finite", "post-finite", "task-post-EF"}:
            raise ValueError(f"unknown post-mix stage: {stage}")
        output = state._replace_waveforms(changes)
        return PostMixStageResult(stage, output, output is not state)


def _speech_scene_post_mix_roles(scene_bundle):
    """从 scene bundle 按既有轨道顺序投影 post-mix role 集合。"""
    bundle_roles = tuple(scene_bundle.tracks)
    observation_roles = tuple(
        role for role in ("mixture", "background") if role in bundle_roles
    )
    protected_roles = tuple(
        role for role in bundle_roles if role.startswith("target/")
    )
    amplitude_roles = tuple(
        role for role in bundle_roles if role.startswith("source_full_rir/")
    )
    return observation_roles, protected_roles, amplitude_roles


def _build_speech_scene_post_mix_request(context, scene, sample_rate, rng):
    """纯构造 speech scene 的 observation post-mix 请求。"""
    from core.data.simulation.observation_post_mix import (
        NonDecomposableObservationContract,
        ObservationPostMixRequest,
    )

    context = coerce_scene_render_context(context)
    observation_roles, protected_roles, amplitude_roles = _speech_scene_post_mix_roles(
        scene.bundle
    )
    non_decomposable_contract = None
    if scene.used_sidechain_wind:
        # 现有 mono sidechain 在 C-2 已消费 wind 控制信号，scene bundle 不重复暴露
        # 一条可被误认为 additive background 的 control waveform；用合同记录其
        # 非分解语义，后续若需要空间 sidechain 必须提供显式 control role。
        non_decomposable_contract = NonDecomposableObservationContract(
            observation_roles=observation_roles,
            protected_roles=protected_roles,
            transform_id="wind_noise_sidechain_v1",
            mutable_stage="C-2",
            rng_stream="background.wind_sidechain",
            provenance={"wind_sidechain_applied": True},
        )
    return ObservationPostMixRequest(
        scene_bundle=scene.bundle,
        sample_rate=sample_rate,
        rng=rng,
        task_policy=context,
        observation_roles=observation_roles,
        protected_roles=protected_roles,
        amplitude_roles=amplitude_roles,
        retain_stage_states=False,
        used_sidechain_wind=scene.used_sidechain_wind,
        non_decomposable_contract=non_decomposable_contract,
    )


def _build_finished_speech_scene_facts(scene, post_mix):
    """纯合并 post-mix facts，保留固定 D/E/F trace 字段。"""
    return {
        **scene.facts,
        "post_mix_stage_order": list(post_mix.trace.stage_order),
        "layer_d_facts": post_mix.facts["layer_d_facts"],
        "layer_e": post_mix.facts["layer_e"],
        "layer_f": post_mix.facts["layer_f"],
        "final_peak_guard": post_mix.facts["final_peak_guard"],
        "post_mix_facts": dict(post_mix.facts),
        "role_contract": post_mix.facts["role_contract"],
        "non_decomposable_observation": post_mix.facts.get("non_decomposable_observation"),
        "decomposable_final": bool(post_mix.facts["decomposable_final"]),
    }


def finish_speech_scene(dataset, scene, sample_rate, rng):
    scene.to_scene_bundle()
    context = coerce_scene_render_context(dataset)
    cfg = context.config
    tracks = scene.bundle.tracks
    mixture = np.asarray(tracks["mixture"])
    if cfg.spatial_enabled:
        microphone_positions = resolve_microphone_positions(
            array_profile=cfg.array_profile,
            custom_geometry=cfg.custom_geometry,
            reference_channel=cfg.reference_channel,
        )
        if microphone_positions is None:
            raise ValueError("spatial scene requires resolved microphone geometry")
        expected_channels = len(microphone_positions)
        if mixture.ndim != 2 or mixture.shape[0] != expected_channels:
            raise ValueError(
                "spatial scene mixture must have shape [num_channels, T]"
            )
        for role, waveform in tracks.items():
            value = np.asarray(waveform)
            if role.startswith("target/"):
                if value.ndim != 1:
                    raise ValueError("spatial scene targets must have shape [T]")
            elif value.ndim != 2 or value.shape[0] != expected_channels:
                raise ValueError(
                    f"spatial scene observation stem {role!r} must have shape [C, T]"
                )
    elif mixture.ndim != 1:
        raise ValueError("mono scene mixture must have shape [T]")
    # 统一走 observation-only post-mix seam；任务层仍通过 scene bundle 的 role
    # 声明哪些轨道属于 observation、amplitude coupling 和 protected target。
    from core.data.simulation.observation_post_mix import ObservationPostMixRenderer

    post_mix = ObservationPostMixRenderer().apply(
        _build_speech_scene_post_mix_request(context, scene, sample_rate, rng)
    )
    facts = _build_finished_speech_scene_facts(scene, post_mix)
    finished = SpeechScene(post_mix.bundle, facts, scene.used_sidechain_wind)
    finished.to_scene_bundle()
    return finished
