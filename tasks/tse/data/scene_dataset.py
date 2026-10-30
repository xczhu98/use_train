"""TSE 多源场景 Adapter：只决定目标身份、缺席场景和 enrollment，复用共享渲染。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import torch

from core.data.dynamic_dataset_base import SampleBuildResult, SampleBuildContractError
from core.data.simulation.tse_scenarios import select_tse_scenario
from core.data.simulation.role_policy import AcousticActor, AcousticRole, RIRPolicy
from core.data.simulation.scene_renderer import render_speech_scene
from core.data.simulation.spatial_executor import SpatialSceneExecutor
from core.data.simulation.spatial_scene import SpatialSceneRequest
from core.data.simulation.observation_post_mix import (
    ObservationPostMixRenderer,
    ObservationPostMixRequest,
    infer_post_mix_role_contract,
)
from core.data.speech_scene import scene_rng, scene_activity_facts, place_scene_sources
from core.data.waveform_bundle import WaveformBundle
from core.data.simulation.scene_spec import SceneSpec
from core.data.simulation.scene_plan import ScenePlan
from core.data.simulation.noise_image import adapt_noise_waveform
from core.data.simulation.scene_sampler import prepare_scene_plan
from core.data.simulation.background_resolver import resolve_background_spec
from core.data.simulation.rng import SimulationRNG
from tasks.tse.cue_contract import TSECue, TSECues
from tasks.tse.data.collate import make_tse_sample, make_tse_target
from tasks.tse.data.spatial_cues import (
    build_static_spatial_cue_from_scene,
    build_track_spatial_cue_from_scene,
)


def _validate_scene_geometry(dataset, scene_plan):
    geometry = getattr(dataset, "resolved_microphone_array", None)
    if geometry is None or scene_plan.array is None:
        raise SampleBuildContractError("spatial TSE requires resolved scene and dataset geometry")
    expected = np.asarray(geometry.positions_m)
    actual = np.asarray(scene_plan.array.positions_m)
    if (actual.shape != expected.shape
            or scene_plan.array.reference_channel != geometry.reference_channel
            or not np.allclose(actual - actual[0], expected - expected[0], rtol=0, atol=1e-6)):
        raise SampleBuildContractError("scene geometry topology/reference differs from dataset; heterogeneous arrays cannot share a TSE model")
    speed = getattr(dataset, "spatial_speed_of_sound_mps", None)
    actual_speed = (scene_plan.room or {}).get("sound_speed")
    if speed is not None and actual_speed is not None and actual_speed != speed:
        raise SampleBuildContractError("scene sound_speed differs from dataset spatial physics")


def _missing_spatial_cue(representation: str) -> TSECue:
    if representation == "static":
        return TSECue("static", None, False, None, None)
    if representation == "track":
        return TSECue("track", None, False, 0, None)
    raise ValueError(f"unsupported dynamic spatial representation: {representation!r}")


def _build_target_spatial_cue(dataset, scene_plan, actor_uid: str, output_num_samples: int):
    representation = getattr(dataset, "spatial_representation", None)
    if representation is None:
        return None, None
    if scene_plan is None or getattr(dataset, "resolved_microphone_array", None) is None:
        return _missing_spatial_cue(representation), None
    if not isinstance(scene_plan.room, Mapping) or scene_plan.room.get("sound_speed") is None:
        return _missing_spatial_cue(representation), None
    try:
        if representation == "static":
            built = build_static_spatial_cue_from_scene(scene_plan, actor_uid=actor_uid)
            return built.to_tse_cue(), {
                "representation": "static",
                "actor_uid": built.actor_uid,
                "scene_uid": built.scene_uid,
                "coordinate_frame": built.coordinate_frame,
                "sound_speed_mps": built.sound_speed_mps,
                "array_origin_m": list(built.array_origin_m),
                "reference_channel": built.reference_channel,
            }
        if representation != "track":
            raise ValueError(f"unsupported dynamic spatial representation: {representation!r}")
        if scene_plan.timebase is not None:
            built = build_track_spatial_cue_from_scene(
                scene_plan,
                actor_uid=actor_uid,
                output_start_sample=0,
                output_num_samples=output_num_samples,
            )
            return built.to_tse_cue(), built.metadata
        static = build_static_spatial_cue_from_scene(scene_plan, actor_uid=actor_uid)
        return TSECue(
            "track",
            static.value.unsqueeze(1),
            True,
            1,
            torch.zeros(1, dtype=torch.int64),
        ), {
            "representation": "track",
            "actor_uid": static.actor_uid,
            "scene_uid": static.scene_uid,
            "coordinate_frame": static.coordinate_frame,
            "sound_speed_mps": static.sound_speed_mps,
            "array_origin_m": list(static.array_origin_m),
            "reference_channel": static.reference_channel,
            "source_sample_anchors": [0],
            "output_start_sample": 0,
            "output_num_samples": int(output_num_samples),
            "renderer": "static_segment",
        }
    except KeyError:
        return _missing_spatial_cue(representation), None


def build_source_scene_sample(dataset, index, uid, sample_rate, force_branch, trace):
    cfg = dataset.sim_cfg
    named_rng = SimulationRNG.for_task(dataset, index, "tse", rng_factory=scene_rng)
    rng = named_rng.stream("scenario")
    scenario, draw = select_tse_scenario(cfg, rng, force_branch)
    branch = scenario.name
    absent = not scenario.target
    requested = 0
    if scenario.interference:
        counts = tuple(cfg.num_interferers)
        requested = int(rng.choice(counts, p=[cfg.num_interferers[k] for k in counts]))
    interferers = dataset._sample_interference_uids(uid, sample_rate, requested, rng) if requested else []
    identities = [dataset.uid_to_speaker_identity[u] for u in interferers]
    if (len(interferers) != requested or len(set(identities)) != requested
            or dataset.uid_to_speaker_identity[uid] in identities):
        raise SampleBuildContractError("TSE sampler must return the requested distinct non-target identities")
    source_uids = ([uid] if scenario.target else []) + interferers
    length = dataset._max_duration_for_sr(sample_rate)
    if length <= 0:
        raise ValueError("multi-source TSE requires positive max_duration_seconds/max_duration")
    prepared = []
    latent_target_reference = None
    excluded_paths = set()
    for position, source_uid in enumerate(source_uids):
        source_trace = trace.source(f"source/{position}")
        source_trace.replace_metadata("uid", source_uid)
        item = dataset._prepare_clean_speech_result(
            source_uid, sample_rate, length, named_rng.stream("speech_preparation", offset=position),
            source_trace=source_trace,
        )
        prepared.append(item.speech_1d)
        for used_uid in [source_uid, *item.speech_splice_export.get("extra_uids", [])]:
            excluded_paths.add(dataset.speech_flt[used_uid])
    if not scenario.target and (scenario.noise or scenario.interference):
        # 目标仍缺席；clean speech 只作为隐式 SNR/SIR reference，不进入 scene。
        latent_target_reference = dataset._prepare_clean_speech_result(
            uid,
            sample_rate,
            length,
            named_rng.stream("latent_reference"),
        ).speech_1d
    render_sources = tuple(prepared)
    render_source_uids = tuple(source_uids)
    latent_spatial = bool(cfg.spatial_enabled and absent)
    if latent_spatial:
        if latent_target_reference is None:
            raise SampleBuildContractError(
                "spatial target-absent TSE scenes require a latent target reference"
            )
        render_sources = (latent_target_reference, *render_sources)
        render_source_uids = (uid, *render_source_uids)
    spatial_rirs = None
    pre_rendered_spatial = None
    background_spec = None
    # timeline 必须在任务层统一使用独立子流；mono 和 spatial 两条路径都先
    # 完成 source window placement，再交给 shared renderer，避免 timeline
    # 抽样与 speech_scene 的声学随机数耦合。
    render_sources, spatial_timeline = place_scene_sources(
        render_sources,
        dataset._simulation_view.shared.temporal,
        named_rng.stream("timeline"),
    )
    sources_preplaced = True
    scene_plan = None
    if cfg.spatial_enabled:
        provider = getattr(dataset, "_spatial_rir_provider", None)
        if provider is None:
            raise ValueError("TSE spatial scene requires a scene-level RIR provider")
        moving = bool(
            cfg.prob_moving_rir > 0.0
            and named_rng.stream("moving_gate").random() < cfg.prob_moving_rir
        )
        scene_plan = prepare_scene_plan(
            provider=provider, task="tse", sample_index=index, sample_rate=sample_rate,
            num_samples=length, source_uids=tuple(render_source_uids),
            spatial_view=dataset._simulation_view.shared.spatial, simulation_cfg=cfg,
            rng=named_rng.stream("scene_plan"), moving=moving,
            roles=tuple(("latent_target" if absent else "target") if i == 0 else "interference"
                        for i in range(len(render_source_uids))),
        )
        _validate_scene_geometry(dataset, scene_plan)
        pre_rendered_spatial = SpatialSceneExecutor().execute(SpatialSceneRequest(
            sources=np.stack(render_sources).astype(np.float32), sample_rate=sample_rate,
            provider=provider, source_uids=tuple(render_source_uids),
            mode="moving" if scene_plan.timebase is not None else "static", track_kinds=("full", "early"),
            early_boundary=max(0, int(round(cfg.early_rir_sec * sample_rate))),
            reference_channel=cfg.reference_channel, rng=named_rng.stream("rir"),
            num_samples=length, normalize=cfg.rir_backend == "online", scene_plan=scene_plan,
        ))
        background_spec = resolve_background_spec(
            cfg=cfg,
            background_view=dataset._simulation_view.shared.background,
            scene_plan=scene_plan,
            provider=provider,
            rng=named_rng.stream("background_resolution"),
            sample_rate=sample_rate,
            image_geometry=scene_plan.array.positions_m,
            shared_scene=True,
            enabled=scenario.noise,
        )
        sources_preplaced = True
    actor_policy = RIRPolicy.SPATIAL_SOURCE if cfg.spatial_enabled else RIRPolicy.MONO_SOURCE
    actor_inputs = tuple(
        AcousticActor(
            uid=source_uid,
            role=(
                AcousticRole.TARGET_SPEECH
                if position == 0 and (scenario.target or latent_spatial)
                else AcousticRole.INTERFERENCE_SPEECH
            ),
            rir_policy=actor_policy,
            waveform=source,
            metadata={"latent_target": bool(position == 0 and latent_spatial and not scenario.target)},
        )
        for position, (source_uid, source) in enumerate(zip(render_source_uids, render_sources))
    )
    scene = render_speech_scene(
        dataset, actor_inputs, sample_rate, named_rng.stream("speech_scene"),
        sir_range=cfg.sir_range, sir_mode="aggregate" if scenario.interference else "none",
        need_noise=scenario.noise, num_samples=length,
        reference_speech=(latent_target_reference if not render_sources else None),
        sir_reference_speech=(
            latent_target_reference
            if (not cfg.spatial_enabled and not scenario.target and scenario.interference)
            else None
        ),
        spatial_rirs=spatial_rirs,
        pre_rendered_spatial=pre_rendered_spatial,
        sources_preplaced=sources_preplaced,
        timeline=spatial_timeline if sources_preplaced else None,
        background_spec=background_spec,
        observation_geometry=(
            {"positions_m": [list(position) for position in scene_plan.array.positions_m]}
            if scene_plan is not None else None
        ),
    )
    noise_image_provenance = {"projection": "absent", "rir_applied": False}
    if cfg.spatial_enabled and scenario.noise:
        # shared scene renderer 已产出最终 [C,T] image；禁止在任务侧再次空间化。
        background = scene.bundle.require("background")
        if np.asarray(background).ndim != 2:
            raise ValueError("shared TSE spatial renderer must produce background [C,T]")
        noise_image_provenance = dict(scene.facts.get("noise_image_provenance", {}))
    tracks = dict(scene.bundle.tracks)
    if latent_spatial:
        latent_track = tracks.pop("source_full_rir/0")
        tracks.pop("target/0")
        tracks["mixture"] = tracks["mixture"] - latent_track
        for source_index in range(len(prepared)):
            tracks[f"source_full_rir/{source_index}"] = tracks.pop(
                f"source_full_rir/{source_index + 1}"
            )
            tracks[f"target/{source_index}"] = tracks.pop(
                f"target/{source_index + 1}"
            )
        observed_plan = ScenePlan(
            SceneSpec(
                scene_name="reference_scene",
                target_present=False,
                interference_present=bool(prepared),
                background_present=scenario.noise,
                echo_present=False,
                source_count=len(prepared),
                task=None,
            )
        )
        sir_facts = [
            {
                **fact,
                "sources": [source_index - 1 for source_index in fact["sources"]],
                "reference": "latent_target",
            }
            for fact in scene.facts.get("sir", [])
        ]
        scene = replace(
            scene,
            bundle=WaveformBundle(tracks),
            facts={
                **scene.facts,
                "activity": (
                    scene_activity_facts(tuple(prepared))
                    if prepared
                    else {"active_ratios": [], "overlap_ratio": 0.0, "active_source_count": 0}
                ),
                "sir": sir_facts,
                "rendered_scene_plan": observed_plan.to_dict(),
                "latent_target_removed_from_observation": True,
            },
        )
    if absent:
        tracks["target/absent"] = np.zeros(length, dtype=np.float32)
    facts = dict(scene.facts)
    scene = replace(scene, bundle=WaveformBundle(tracks), facts=facts)
    target_roles = [(uid, "target/absent" if absent else "target/0")]
    if source_uids:
        start = 0 if absent else 1
        target_roles.extend((source_uids[i], f"target/{i}") for i in range(start, len(source_uids)))
    target_roles = target_roles[:dataset.targets_per_mixture]

    scene_bundle = scene.to_scene_bundle()
    post_mix_contract = infer_post_mix_role_contract(
        scene_bundle,
        protected_prefixes=("target/",),
    )
    post_mix = ObservationPostMixRenderer().apply(
        ObservationPostMixRequest(
            scene_bundle=scene_bundle,
            sample_rate=sample_rate,
            rng=named_rng.stream("post_mix"),
            task_policy=dataset,
            observation_roles=post_mix_contract.observation_roles,
            # D joint gain 必须覆盖全部 target stem；E/F/final 仍只改变 observation。
            protected_roles=post_mix_contract.protected_roles,
            amplitude_roles=post_mix_contract.amplitude_roles,
            role_contract=post_mix_contract,
            retain_stage_states=False,
        )
    )
    post_mix_facts = dict(post_mix.facts)
    # ``adapter`` describes observation-only invariance; ``delegate`` keeps
    # the canonical Layer D/E/F facts published by SourceScenePostMixAdapter.
    adapter_facts = dict(post_mix_facts.get("delegate", post_mix_facts["adapter"]))
    layer_facts = {key: dict(value) if isinstance(value, dict) else value
                   for key, value in adapter_facts.items()}
    layer_f = dict(layer_facts["F"])
    # 任务 audit 保留原 canonical schema；逐 channel facts 留在 post_mix_facts。
    layer_f.pop("channel_facts", None)
    if not layer_f.get("applied_ops") and not layer_f.get("observation_changed"):
        layer_f["clipping_facts"] = {}
    scene = replace(
        scene,
        bundle=post_mix.bundle.bundle,
        facts={
            **scene.facts,
            "post_mix_stage_order": list(post_mix.trace.stage_order),
            "layer_d_facts": {
                "stage_order": ("D-normalize", "D-guard"),
                "normalize": adapter_facts.get("D-normalize", adapter_facts["normalize"]),
                "guard": adapter_facts["D-guard"],
            },
            "layer_e": layer_facts["E"],
            "layer_f": layer_f,
            "final_peak_guard": adapter_facts["final-guard"],
            "decomposable_final": bool(post_mix_facts["decomposable_final"]),
            "post_mix_facts": post_mix_facts,
            "rng": named_rng.stream_facts(),
        },
    )
    targets = []
    spatial_cue_metadata = []
    for position, (target_uid, role) in enumerate(target_roles):
        from tasks.tse.data.dynamic_dataset import EnrollmentAugmentationProviderError

        try:
            enrollment = dataset._build_enrollment(
                target_uid, sample_rate, named_rng.stream("enrollment", offset=position),
                role=f"enrollment/{position}", trace=trace,
                excluded_paths=excluded_paths, audit_name=f"enrollment.{position}.snr",
            )
        except EnrollmentAugmentationProviderError:
            raise
        except Exception as exc:
            kind = "alternate" if position else "primary"
            raise EnrollmentAugmentationProviderError(
                f"failed to construct enrollment for {kind} target {target_uid!r}: {exc}"
            ) from exc
        waveform = scene.bundle.require(role).copy()
        spatial_cue, spatial_metadata = _build_target_spatial_cue(
            dataset, scene_plan, target_uid, length
        )
        if spatial_metadata is not None:
            spatial_cue_metadata.append(spatial_metadata)
        else:
            spatial_cue_metadata.append({"actor_uid": target_uid, "present": spatial_cue is not None and spatial_cue.present})
        targets.append(
            make_tse_target(
                target=waveform,
                cues=TSECues(
                    speaker_waveform=TSECue(
                        "waveform",
                        enrollment,
                        True,
                        len(enrollment),
                        None,
                    ),
                    spatial=spatial_cue,
                ),
                speaker_id=target_uid,
                spk_label=dataset._spk_label(target_uid),
                target_present=role != "target/absent",
            )
        )
        target_trace = trace.source("target" if position == 0 else f"alternate_target/{position - 1}")
        target_trace.replace_metadata("final_track", {"sample_rate": sample_rate, "num_samples": length})
        if trace.capture_stems:
            trace.source(f"enrollment/{position}").replace_stem("final_enrollment", enrollment)
            trace.source("target" if position == 0 else f"alternate_target/{position - 1}").replace_stem("final_target", waveform)
    present_uids = source_uids
    facts = {**scene.facts, "branch": branch, "target_present": scenario.target,
             "interference_present": scenario.interference, "noise_present": scenario.noise,
             "requested_interferers": requested, "actual_interferers": len(interferers),
             "source_uids": present_uids, "source_identities": [dataset.uid_to_speaker_identity[u] for u in present_uids],
             "num_speakers_in_scene": len(present_uids),
             "supervised_targets": len(targets), "sample_rate": sample_rate, "num_samples": length,
             "final_time_transform": "shared_layer_a_fixed_canvas"}
    if getattr(dataset, "spatial_representation", None) is not None:
        facts["spatial_cues"] = tuple(spatial_cue_metadata)
    facts["noise_image"] = noise_image_provenance
    facts["scene_spec"] = SceneSpec(
        scene_name=branch,
        target_present=scenario.target,
        interference_present=scenario.interference,
        background_present=scenario.noise,
        echo_present=False,
        source_count=len(source_uids),
        task="tse",
    ).to_dict()
    trace.replace_metadata("branch", branch)
    trace.replace_metadata("scene_name", branch)
    trace.replace_metadata("scene", facts)
    trace.source("scene").replace_metadata("scene", facts)
    trace.replace_metadata("branch_selection", {"selected": branch, "uniform_u": draw, "force_branch": force_branch})
    trace.set_legacy_snapshot("_last_run_branch", branch)
    if trace.capture_stems:
        for role, waveform in scene.bundle.tracks.items():
            trace.source("scene").replace_stem(role, waveform)
        trace.source("scene").replace_stem("final_mixture", scene.bundle.require("mixture"))
    mixture = np.asarray(scene.bundle.require("mixture"), dtype=np.float32).copy()
    if mixture.ndim == 1:
        mixture = mixture[None, :]
    sample = make_tse_sample(
        key=uid,
        mixture=mixture,
        sr=sample_rate,
        length=length,
        targets=targets,
        geometry=getattr(dataset, "resolved_microphone_array", None),
        timebase=scene_plan.timebase if scene_plan is not None else None,
        provenance=facts,
    )
    return SampleBuildResult(sample, trace)
