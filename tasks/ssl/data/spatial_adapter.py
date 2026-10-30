"""SSL 对共享空间场景与 observation post-mix seam 的任务侧适配器。

空间场景渲染与 observation post-mix 已分别复用 core 的
``SpatialSceneExecutor`` 和 ``ObservationPostMixRenderer``；本模块只负责把 SSL
Dataset 的 ``render``/``apply`` 请求与结果适配到共享实现。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.data.simulation.scene_renderer import render_speech_scene
from core.data.simulation import scene_sampler
from core.data.simulation.role_policy import AcousticActor, AcousticRole, RIRPolicy
from core.data.simulation.spatial_executor import SpatialSceneExecutor
from core.data.simulation.spatial_scene import SpatialSceneRequest as CoreSpatialSceneRequest
from core.data.simulation.observation_post_mix import (
    NonDecomposableObservationContract,
    ObservationPostMixRenderer as CoreObservationPostMixRenderer,
    ObservationPostMixRequest,
)
from core.data.simulation.rng import SimulationRNG
from core.data.simulation.scene_context import SceneRenderContext
from core.data.spatial import SpatialRenderResult


def _has_spatial_background_resource(
    *, noise_available: bool, wind_available: bool, config: Any
) -> bool:
    """按显式资源事实判断 SSL spatial scene 是否需要背景层。

    ordinary noise 与 wind pool 是两个独立资源；仅配置 wind pool 时也必须
    进入 Layer C，不能把 ``noise_flt`` 误当成所有背景资源的总开关。
    """
    if noise_available:
        return True
    if not wind_available:
        return False
    return bool(
        float(getattr(config, "prob_wind_noise_sidechain", 0.0)) > 0.0
        or float(getattr(config, "prob_wind_noise_background", 0.0)) > 0.0
    )


@dataclass(frozen=True)
class SpatialSceneRenderRequest:
    context: SceneRenderContext
    rng: SimulationRNG
    background_enabled: bool
    sources: tuple[np.ndarray, ...]
    source_uids: tuple[str, ...]
    index: int
    segment_length: int
    sample_rate: int
    provider: Any
    moving: bool
    trajectory_points: int
    max_displacement_m: float
    reference_channel: int
    early_boundary: int
    timeline: Any

    def __post_init__(self) -> None:
        if not isinstance(self.context, SceneRenderContext):
            raise TypeError("context must be SceneRenderContext")
        if not isinstance(self.rng, SimulationRNG) or self.rng.task != "ssl":
            raise TypeError("rng must be the SSL SimulationRNG")
        if type(self.background_enabled) is not bool:
            raise TypeError("background_enabled must be bool")


@dataclass(frozen=True)
class SpatialSceneRenderResult:
    pre_mix_scene: Any
    direct_tracks: np.ndarray
    mic_positions: np.ndarray
    source_positions: np.ndarray
    provenance: dict[str, Any]
    backend: str
    scene_plan: Any


@dataclass(frozen=True)
class ObservationPostMixResult:
    observation: np.ndarray
    direct_tracks: np.ndarray
    mic_positions: np.ndarray
    source_positions: np.ndarray
    facts: dict[str, Any]
    scene: Any


class SSLSpatialSceneAdapter:
    """把 SSL 请求适配到 core SpatialSceneExecutor 的任务侧 Adapter。

    ``SpatialSceneRenderer`` 是 core 中真正执行 RIR/轨迹渲染的实现名称；
    SSL 这一层只负责请求和结果的任务语义适配，因此使用显式的 Adapter 名称。
    旧名称在文件末尾保留兼容别名，避免破坏已发布的测试和外部调用方。
    """

    def render(self, request: SpatialSceneRenderRequest) -> SpatialSceneRenderResult:
        named_rng = request.rng
        rng = named_rng.stream("scene_plan")
        config = request.context.config
        plan = scene_sampler.prepare_scene_plan(
            provider=request.provider,
            task="ssl",
            sample_index=request.index,
            sample_rate=request.sample_rate,
            num_samples=request.segment_length,
            source_uids=request.source_uids,
            spatial_view=request.context.simulation_view.shared.spatial,
            simulation_cfg=config,
            rng=rng,
            moving=request.moving,
            trajectory_points=request.trajectory_points,
            max_displacement_m=request.max_displacement_m,
            roles=tuple("target" for _ in request.source_uids),
        )
        # offline moving manifest 的 mode 由 plan timebase 派生，与抽样布尔无关。
        moving = plan.timebase is not None
        method = "sample_moving_scene_tracks" if moving else "sample_scene_tracks"
        if not callable(getattr(request.provider, method, None)):
            raise TypeError(f"SSL scene provider must implement {method}")
        result = SpatialSceneExecutor().execute(
            CoreSpatialSceneRequest(
                sources=np.stack(request.sources).astype(np.float32, copy=False),
                sample_rate=request.sample_rate,
                provider=request.provider,
                source_uids=request.source_uids,
                mode='moving' if moving else 'static',
                track_kinds=('full', 'direct', 'early'),
                early_boundary=request.early_boundary,
                reference_channel=request.reference_channel,
                rng=rng,
                scene_plan=plan,
                normalize=config.rir_normalize,
                num_samples=request.segment_length,
                trajectory_points=request.trajectory_points,
                max_displacement_m=request.max_displacement_m,
            )
        )
        rendered = SpatialRenderResult(
            result.full_tracks,
            result.early_tracks,
            result.full_tracks[:, request.reference_channel],
            result.mixture,
            direct_tracks=result.direct_tracks,
            geometry=result.geometry,
            provenance=result.provenance,
        )
        facts = dict(result.provenance)
        facts["rng"] = named_rng.stream_facts()
        track_facts = facts.get("track_facts")
        if isinstance(track_facts, dict):
            facts.setdefault("rir", dict(track_facts))
        if not moving:
            positions = np.asarray([actor.position_m for actor in plan.actors], dtype=np.float32)
        else:
            if result.trajectory is None or result.timebase != plan.timebase:
                raise ValueError("SSL moving result must preserve scene trajectory timebase")
            expected = np.concatenate([actor.trajectory.positions_m for actor in plan.actors], axis=1)
            if not np.allclose(result.trajectory.positions_m, expected):
                raise ValueError("SSL moving result trajectory differs from scene plan")
            positions = result.trajectory.positions_m
            # DOA 投影消费同一条已验证的轨迹，保留原始 full/direct identity。
            facts["rir"] = {
                **facts.get("rir", {}),
                "trajectory": positions.tolist(),
                "trajectory_timebase": result.timebase.to_dict(),
            }
        # array 与 actor 都来自 sampler，不能混用局部阵列与房间绝对坐标。
        mic_positions = np.asarray(plan.array.positions_m, dtype=np.float32)
        return SpatialSceneRenderResult(
            pre_mix_scene=rendered,
            direct_tracks=rendered.direct_tracks.copy(),
            mic_positions=mic_positions,
            source_positions=positions,
            provenance={**facts, 'timeline': request.timeline},
            backend=config.rir_backend,
            scene_plan=plan,
        )



class SSLObservationPostMixAdapter:
    """把 SSL 请求适配到 core ObservationPostMixRenderer 的任务侧 Adapter。

    core renderer 负责 D/E/F 执行；本类仅负责 SSL role、background 和
    direct-path 投影的请求装配。
    """

    def apply(
        self,
        request: SpatialSceneRenderRequest,
        rendered: SpatialSceneRenderResult,
    ) -> ObservationPostMixResult:
        named_rng = request.rng
        actors = tuple(
            AcousticActor(
                uid=source_uid,
                role=AcousticRole.TARGET_SPEECH,
                rir_policy=RIRPolicy.SPATIAL_SOURCE,
                waveform=source,
            )
            for source_uid, source in zip(request.source_uids, request.sources, strict=True)
        )
        from core.data.simulation.background_resolver import resolve_background_spec
        shared_view = getattr(request.context.simulation_view, "shared", None)
        background_enabled = request.background_enabled
        background_spec = resolve_background_spec(
            cfg=request.context.config,
            background_view=getattr(shared_view, "background", None),
            scene_plan=rendered.scene_plan,
            provider=request.provider,
            rng=named_rng.stream("background_resolution"),
            sample_rate=request.sample_rate,
            image_geometry=rendered.scene_plan.array.positions_m,
            shared_scene=True,
            enabled=background_enabled,
        )
        scene = render_speech_scene(
            request.context,
            actors,
            request.sample_rate,
            named_rng.stream("speech_scene"),
            sir_range=None,
            sir_mode="none",
            need_noise=background_enabled,
            num_samples=request.segment_length,
            spatial_rirs=rendered.pre_mix_scene.full_tracks,
            pre_rendered_spatial=rendered.pre_mix_scene,
            sources_preplaced=True,
            timeline=request.timeline,
            background_spec=background_spec,
            observation_geometry={
                "positions_m": [
                    list(position) for position in rendered.scene_plan.array.positions_m
                ],
            },
        )
        direct_tracks = rendered.direct_tracks.copy()
        # direct 监督必须保持 source 和 microphone 轴。
        if direct_tracks.ndim != 3 or direct_tracks.shape[0] != len(request.sources):
            raise ValueError(f'spatial direct track/source mismatch: {direct_tracks.shape} vs {len(request.sources)}')
        for index, calibration in enumerate(scene.facts.get("source_volume_calibration") or ()):
            direct_tracks[index] *= calibration["gain"]
        tracks = dict(scene.bundle.tracks)
        for index, direct in enumerate(direct_tracks):
            tracks[f"direct/{index}"] = direct.astype(np.float32, copy=True)
        bundle = scene.bundle.replace(tracks=tracks)
        protected = tuple(
            [f"target/{index}" for index in range(len(request.sources))]
            + [f"direct/{index}" for index in range(len(request.sources))]
        )
        non_decomposable_contract = None
        if bool(getattr(scene, "used_sidechain_wind", False)):
            non_decomposable_contract = NonDecomposableObservationContract(
                observation_roles=("mixture", "background"),
                protected_roles=protected,
                transform_id="wind_noise_sidechain_v1",
                mutable_stage="C-2",
                rng_stream="background.wind_sidechain",
                provenance={
                    "wind_sidechain_applied": True,
                    "task": "ssl",
                },
            )
        post_mix = CoreObservationPostMixRenderer().apply(
            ObservationPostMixRequest(
                scene_bundle=bundle,
                sample_rate=request.sample_rate,
                rng=named_rng.stream("post_mix"),
                task_policy=request.context,
                # D-guard 与 source decomposition 需要同步缩放 background；
                # E/F 仍只由 core adapter 改写 mixture。
                observation_roles=("mixture", "background"),
                protected_roles=protected,
                amplitude_roles=tuple(f"source_full_rir/{index}" for index in range(len(request.sources))),
                retain_stage_states=False,
                used_sidechain_wind=bool(getattr(scene, "used_sidechain_wind", False)),
                non_decomposable_contract=non_decomposable_contract,
            )
        )
        final_bundle = post_mix.bundle
        post_mix_facts = dict(post_mix.facts)
        scene_facts = {
            **post_mix_facts,
            "post_mix_stage_order": list(post_mix.trace.stage_order),
            "layer_d_facts": dict(post_mix_facts["layer_d_facts"]),
            "layer_e": dict(post_mix_facts["layer_e"]),
            "layer_f": dict(post_mix_facts["layer_f"]),
            "final_peak_guard": dict(post_mix_facts["final_peak_guard"]),
            "decomposable_final": bool(post_mix_facts["decomposable_final"]),
        }
        scene_render_facts = dict(scene.facts)
        facts = {
            **rendered.provenance,
            "background_policy": scene.facts.get("background_policy"),
            # 背景是在 shared speech scene 内完成投影和 SNR 标定的；这些事实
            # 必须穿过 post-mix adapter，避免 SSL 样本只保留策略名而丢失
            # [C,T] shape、coherence、SNR 与 source identity。
            "noise_image_provenance": scene_render_facts.get(
                "noise_image_provenance"
            ),
            "background_snr_values": scene_render_facts.get("background_snr_values"),
            "background_snr_nominal_values": scene_render_facts.get(
                "background_snr_nominal_values"
            ),
            "background_snr_reference_channel_db": scene_render_facts.get(
                "background_snr_reference_channel_db"
            ),
            "background_audit": scene_render_facts.get("background_audit"),
            "decomposable_before_post_mix": scene_render_facts.get(
                "decomposable_before_post_mix"
            ),
            "scene_render": scene_render_facts,
            "post_mix_stage_order": scene_facts["post_mix_stage_order"],
            "post_mix": scene_facts,
            "layer_d_facts": scene_facts["layer_d_facts"],
            "layer_e": scene_facts["layer_e"],
            "layer_f": scene_facts["layer_f"],
            "final_peak_guard": scene_facts["final_peak_guard"],
            "decomposable_final": scene_facts["decomposable_final"],
            "scene": scene_facts,
            "direct_path_exact": True,
            "direct_path_contract": "core_spatial_direct_rir",
            "backend": rendered.backend,
        }
        return ObservationPostMixResult(
            observation=np.asarray(final_bundle.require("mixture")),
            direct_tracks=np.stack([final_bundle.require(f"direct/{index}") for index in range(len(request.sources))]),
            mic_positions=rendered.mic_positions,
            source_positions=rendered.source_positions,
            facts=facts,
            scene=post_mix,
        )


## 保留旧公开名称，避免迁移期破坏外部 import；新代码应使用带任务前缀的名称。
SpatialSceneRenderer = SSLSpatialSceneAdapter
ObservationPostMixRenderer = SSLObservationPostMixAdapter


__all__ = [
    "SSLObservationPostMixAdapter",
    "SSLSpatialSceneAdapter",
    "ObservationPostMixRenderer",
    "ObservationPostMixResult",
    "SpatialSceneRenderRequest",
    "SpatialSceneRenderResult",
    "SpatialSceneRenderer",
]
