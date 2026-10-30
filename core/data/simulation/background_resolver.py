"""背景点源 RIR 的生产解析 seam。

``prob_point_source_background`` 配置门在这里被解析为显式 provider 并注入
``build_background_spec``：噪声 actor 与 speech scene 共享 room/array/timebase，
位置按 speech 源相同的分布族在同一房间内采样。门关闭（概率 0）、样本不需要
噪声或单通道观测时退化为 ``build_background_spec`` 默认策略，且零随机数消耗，
保持既有 seeded replay 字节一致。
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from core.data.acoustic_scene_plan import (
    AcousticActorPlan,
    AcousticScenePlan,
    NoisePlan,
    RIRPlan,
    SourceGainPlan,
)
from core.data.simulation.point_source import (
    MovingPointSourceRIRProvider,
    PointSourceRIRProvider,
)
from core.data.simulation.rir_config import OnlineRIRConfig
from core.data.simulation.role_policy import BackgroundPolicy, build_background_spec
from core.data.simulation.scene_context import coerce_scene_render_context
from core.data.simulation.trajectory import sample_source_trajectories_by_speed

BACKGROUND_NOISE_ACTOR_UID = "background_noise/0"
BACKGROUND_NOISE_ACTOR_ROLE = "background_noise"
_MAX_POSITION_ATTEMPTS = 500


def _require_scene_uid(scene_plan) -> str:
    uid = scene_plan.identity.get("scene_uid")
    if not isinstance(uid, str) or not uid.strip():
        raise ValueError("background point source requires scene identity with scene_uid")
    return uid.strip()


def _sample_noise_position(*, scene_plan, rir_cfg, rng) -> np.ndarray:
    """在与 speech 源相同的约束族内采样噪声源位置。

    距离从阵列质心按 ``distance_range`` 抽取，接受条件复用
    ``sample_rir_scene`` 的 margin/height/min_separation 约定，并与 scene 内
    既有 actor 的起始位置保持距离。
    """

    room = np.asarray(scene_plan.room["room_dim_m"], dtype=np.float64)
    center = np.asarray(scene_plan.array.positions_m, dtype=np.float64).mean(axis=0)
    existing = []
    for actor in scene_plan.actors:
        if actor.position_m is not None:
            existing.append(np.asarray(actor.position_m, dtype=np.float64))
        elif actor.trajectory is not None:
            existing.append(np.asarray(actor.trajectory.positions_m[0, 0], dtype=np.float64))
    margin = float(rir_cfg.margin)
    min_separation = float(rir_cfg.min_separation)
    for _ in range(_MAX_POSITION_ATTEMPTS):
        direction = rng.normal(size=3)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        position = center + direction * float(rng.uniform(*rir_cfg.distance_range))
        if (
            np.all(position > margin)
            and np.all(position < room - margin)
            and rir_cfg.source_height[0] <= position[2] <= rir_cfg.source_height[1]
            and all(
                float(np.linalg.norm(position - other)) >= min_separation
                for other in existing
            )
        ):
            return position
    raise ValueError(
        "cannot place background point source within the configured room/distance constraints"
    )


def sample_background_noise_actor(*, scene_plan, cfg, rng) -> AcousticActorPlan:
    """在同一 room 内采样背景噪声 actor；moving scene 共享 timebase 规划轨迹。"""
    rir_cfg = OnlineRIRConfig.from_simulation(cfg)
    position = _sample_noise_position(scene_plan=scene_plan, rir_cfg=rir_cfg, rng=rng)
    if scene_plan.timebase is not None:
        trajectory, _speeds = sample_source_trajectories_by_speed(
            position[None, :],
            room_dim_m=np.asarray(scene_plan.room["room_dim_m"], dtype=np.float64),
            timebase=scene_plan.timebase,
            speed_range_mps=tuple(cfg.moving_source_speed_range_mps),
            rng=rng,
            wall_margin_m=float(rir_cfg.margin),
            min_separation_m=float(rir_cfg.min_separation),
            source_height_range_m=tuple(rir_cfg.source_height),
        )
        return AcousticActorPlan(
            actor_uid=BACKGROUND_NOISE_ACTOR_UID,
            role=BACKGROUND_NOISE_ACTOR_ROLE,
            trajectory=trajectory,
        )
    return AcousticActorPlan(
        actor_uid=BACKGROUND_NOISE_ACTOR_UID,
        role=BACKGROUND_NOISE_ACTOR_ROLE,
        position_m=tuple(float(value) for value in position),
    )


def build_background_noise_plan(*, scene_plan, noise_actor) -> AcousticScenePlan:
    """克隆 speech scene 的 room/array/timebase，只保留噪声 actor 的派生 plan。

    派生 identity 使噪声 RIR 的 provenance 可追溯到原 scene，同时让噪声
    scene 拥有独立的 scene_uid，不与 speech 资产混用。
    """
    base_uid = _require_scene_uid(scene_plan)
    instance_uid = str(scene_plan.identity.get("scene_instance_uid") or base_uid)
    payload = {"scene_instance_uid": instance_uid, "actor": noise_actor.to_dict()}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()[:24]
    return AcousticScenePlan(
        sample_rate=scene_plan.sample_rate,
        num_samples=scene_plan.num_samples,
        need_noise=False,
        rir=RIRPlan.disabled(),
        source_gain=SourceGainPlan.disabled(),
        noise=NoisePlan.disabled(),
        identity={
            **scene_plan.identity,
            "scene_uid": f"{base_uid}/noise/{digest}",
            "scene_instance_uid": f"{instance_uid}/noise/{digest}",
            "background_noise": True,
        },
        room=scene_plan.room,
        array=scene_plan.array,
        timebase=scene_plan.timebase,
        actors=(noise_actor,),
    )


def resolve_point_source_provider(*, provider, noise_plan, sample_rate):
    """按 noise plan 是否 moving 构造对应点源 provider。"""
    source_uid = noise_plan.actors[0].actor_uid
    if noise_plan.timebase is not None:
        return MovingPointSourceRIRProvider.from_scene_plan(
            provider, noise_plan, source_uid=source_uid, sample_rate=sample_rate
        )
    return PointSourceRIRProvider.from_scene_plan(
        provider, noise_plan, source_uid=source_uid, sample_rate=sample_rate
    )


def resolve_background_spec(
    *,
    cfg,
    background_view=None,
    scene_plan,
    provider,
    rng,
    sample_rate,
    image_geometry,
    shared_scene=False,
    enabled=True,
):
    """按 ``prob_point_source_background`` 门解析背景策略。

    ``background_view`` 是 ``SharedSimulationView.background`` 的只读投影；提供时
    点源门控只从该 view 读取，``cfg`` 仍用于构造 moving 轨迹所需的完整物理参数。
    门关闭、样本不需要噪声（``enabled=False``）或单通道观测时退化为
    ``build_background_spec`` 默认策略且不消费随机数；门打开时在同一 room 内
    采样噪声 actor，构造点源 provider 注入显式 seam。
    """
    point_source_probability = (
        float(background_view.prob_point_source_background)
        if background_view is not None
        else float(cfg.prob_point_source_background)
    )
    if (
        not enabled
        or image_geometry is None
        or point_source_probability <= 0.0
    ):
        return build_background_spec(image_geometry=image_geometry, shared_scene=shared_scene)
    if float(rng.random()) >= point_source_probability:
        return build_background_spec(image_geometry=image_geometry, shared_scene=shared_scene)
    if not callable(getattr(provider, "sample_scene_tracks", None)):
        raise TypeError(
            "background point source requires a scene RIR provider with sample_scene_tracks"
        )
    noise_actor = sample_background_noise_actor(scene_plan=scene_plan, cfg=cfg, rng=rng)
    noise_plan = build_background_noise_plan(scene_plan=scene_plan, noise_actor=noise_actor)
    point_source = resolve_point_source_provider(
        provider=provider, noise_plan=noise_plan, sample_rate=sample_rate
    )
    return build_background_spec(
        image_geometry=image_geometry,
        shared_scene=shared_scene,
        point_source_provider=point_source,
    )


def resolve_and_mix_background(
    *,
    dataset,
    speech_image,
    reference_image,
    sample_rate,
    actual_length,
    max_duration_samples,
    rng,
    mix_rng=None,
    need_noise,
    snr,
    scene_plan,
    provider,
    image_geometry,
    mix_image_geometry=None,
    shared_scene=False,
):
    """解析背景策略并完成一次背景混合。

    该 seam 保持 ``resolve_background_spec → mix_background`` 的调用顺序，
    让 SE/AEC/SS 共享相同的点源门控、SNR 标定和 provenance 入口。返回实际
    使用的 ``BackgroundSpec``，调用方可把它写入 scene facts。
    """
    context = coerce_scene_render_context(dataset)
    spec = resolve_background_spec(
        cfg=context.config,
        background_view=(
            getattr(getattr(context.simulation_view, "shared", None), "background", None)
        ),
        scene_plan=scene_plan,
        provider=provider,
        rng=rng,
        sample_rate=int(sample_rate),
        image_geometry=image_geometry,
        shared_scene=shared_scene,
        enabled=bool(need_noise),
    )
    # 延迟导入避免 background.py 与 resolver 之间形成模块初始化环。
    from core.data.simulation.background import mix_background

    effective_image_geometry = (
        image_geometry
        if mix_image_geometry is None or spec.policy == BackgroundPolicy.POINT_SOURCE_RIR
        else mix_image_geometry
    )
    background = mix_background(
        context,
        np.asarray(speech_image, dtype=np.float32).copy(),
        np.asarray(reference_image, dtype=np.float32),
        int(sample_rate),
        int(actual_length),
        int(max_duration_samples),
        rng if mix_rng is None else mix_rng,
        bool(need_noise),
        snr,
        image_geometry=effective_image_geometry,
        background_spec=spec,
    )
    return background, spec


__all__ = [
    "BACKGROUND_NOISE_ACTOR_ROLE",
    "BACKGROUND_NOISE_ACTOR_UID",
    "build_background_noise_plan",
    "resolve_and_mix_background",
    "resolve_background_spec",
    "resolve_point_source_provider",
    "sample_background_noise_actor",
]
