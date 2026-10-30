"""固定 K 的动态 SS：共享 Layer A、逐源 C、D–F 和 worker/trace 生命周期。"""

from __future__ import annotations

import math
from dataclasses import replace
import numpy as np

import torch
from core.data.dynamic_dataset_base import BaseDynamicMixingDataset, SampleBuildResult
from core.data.simulation.scene_renderer import (
    SpeechScene,
    calibrate_spatial_source_volume,
    render_speech_scene,
)
from core.data.simulation.role_policy import (
    AcousticActor,
    AcousticRole,
    RIRPolicy,
)
from core.data.simulation.spatial_executor import SpatialSceneExecutor
from core.data.simulation.spatial_scene import SpatialSceneRequest
from core.data.simulation.rng import SimulationRNG
from core.data.simulation.observation_post_mix import (
    ObservationPostMixRenderer,
    ObservationPostMixRequest,
)
from core.data.simulation.energy_ratio import EnergyRatioScaler
from core.data.waveform_bundle import WaveformBundle
from core.data.source_pool import sample_distinct_identities
from core.data.speech_scene import place_scene_sources, scene_activity_facts, scene_rng
from core.data.simulation.ss_scenarios import select_ss_scenario, validate_ss_scenarios

from core.data.offline_dataset import validate_loaded_audio_metadata
from core.data.offline_multitrack import ensure_finite, ensure_mono, load_audio, resample
from tasks.ss.data.source_pool import SourcePoolRecord, parse_source_pool
from tasks.ss.data.config import build_ss_dataset_contract
from core.data.simulation.scene_names import validate_scene_name
from core.data.simulation.scene_spec import SceneSpec
from core.data.simulation.scene_plan import ScenePlan
from core.data.simulation.scene_sampler import prepare_scene_plan


class SSDynamicMixingDataset(BaseDynamicMixingDataset):
    """K 个身份或匿名 source；身份只参与采样，不作为模型标签。"""

    _simulation_task = "ss"

    def __init__(
        self,
        speech_scp: str,
        *,
        num_sources: int,
        sample_rate: int,
        segment_seconds: float = 3.0,
        sir_range: tuple[float, float] | list[float] = (-5.0, 5.0),
        seed: int = 0,
        noise_scp: str | None = None,
        rir_scp: str | None = None,
        windnoise_scp: str | None = None,
        music_scp: str | None = None,
        simulation_cfg: dict | None = None,
        strict_load: bool = True,
        repeat_to_max_duration: bool = False,
        repeat_tile_silence_gap: bool = False,
        repeat_tile_silence_gap_range: list | None = None,
        filter_by_duration: bool = False,
        filter_min_duration: float = 0.5,
        filter_max_duration: float = 30.0,
        speech_length_file: str | None = None,
        target_sr: int | None = None,
        sampling_cfg: object = None,
        placeholder_num_samples: int = 1024,
        export_simulation_stems: bool = False,
        simulation_cfg_strict: bool = True,
    ) -> None:
        if type(num_sources) is not int or num_sources < 1:
            raise ValueError("num_sources must be an exact positive int")
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("sample_rate must be an exact positive int")
        if target_sr is not None and (type(target_sr) is not int or target_sr != sample_rate):
            raise ValueError("SS target_sr must match the output sample_rate")
        if (
            isinstance(segment_seconds, bool)
            or not isinstance(segment_seconds, (int, float))
            or not math.isfinite(segment_seconds)
            or segment_seconds <= 0
        ):
            raise ValueError("segment_seconds must be a finite positive number")
        if (
            not isinstance(sir_range, (list, tuple))
            or len(sir_range) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in sir_range
            )
            or sir_range[0] > sir_range[1]
        ):
            raise ValueError("sir_range must be [low, high]")
        if type(seed) is not int:
            raise TypeError("seed must be an exact int")
        self.records: tuple[SourcePoolRecord, ...] = parse_source_pool(speech_scp)
        groups = {}
        for record in self.records:
            groups.setdefault(record.identity, []).append(record.uid)
        if len(groups) < num_sources:
            raise ValueError("source pool must contain at least num_sources distinct speakers or anonymous sources")
        self._speaker_records = {identity: tuple(uids) for identity, uids in groups.items()}
        self._records_by_path = {str(record.path): record for record in self.records}
        self._identity_by_uid = {record.uid: record.identity for record in self.records}
        self.num_sources = num_sources
        self.sample_rate = sample_rate
        # 与共享 CanvasSpec 保持一致：固定画布按秒数乘采样率向下取整。
        self.segment_length = int(float(segment_seconds) * sample_rate)
        if self.segment_length < 1:
            raise ValueError("segment_seconds must produce at least one sample")
        self.sir_range = (float(sir_range[0]), float(sir_range[1]))
        self.seed = seed
        # 无资源的最小 SS 默认保持干净混合；显式参数始终覆盖默认值。
        sim = {"prob_reverberation": 0.0,
               "prob_channel_enabled": False, "use_high_pass": False,
            "rir_backend": "offline", "prob_target_only": 1.0,
            "prob_target_noise": 0.0, "prob_noise_only": 0.0,
            **(simulation_cfg or {})}
        self._has_background = noise_scp is not None or windnoise_scp is not None or music_scp is not None
        super().__init__(
            speech_scp, noise_scp, rir_scp, windnoise_scp=windnoise_scp, music_scp=music_scp,
            simulation_cfg=sim, max_duration_seconds=float(segment_seconds),
            strict_load=strict_load,
            repeat_to_max_duration=repeat_to_max_duration,
            repeat_tile_silence_gap=repeat_tile_silence_gap,
            repeat_tile_silence_gap_range=repeat_tile_silence_gap_range,
            speech_length_file=speech_length_file,
            filter_by_duration=filter_by_duration,
            filter_min_duration=filter_min_duration,
            filter_max_duration=filter_max_duration,
            target_sr=target_sr or sample_rate,
            sampling_cfg=sampling_cfg,
            placeholder_num_samples=placeholder_num_samples,
            export_simulation_stems=export_simulation_stems,
            simulation_cfg_strict=simulation_cfg_strict,
        )
        eligible_uids = {self.speech_uids[fs][idx] for fs, idx in self._filtered_indices}
        groups = {}
        for record in self.records:
            if record.uid in eligible_uids:
                groups.setdefault(record.identity, []).append(record.uid)
        self._speaker_records = {identity: tuple(uids) for identity, uids in groups.items()}
        if len(self._speaker_records) < num_sources:
            raise ValueError("SS duration filter leaves fewer than num_sources identities")
        self.set_epoch(0)
        if (not self.sim_cfg.spatial_enabled and self.sim_cfg.rir_backend == "offline"
                and self.sim_cfg.prob_reverberation > 0 and not self.rirs_flt):
            raise ValueError("SS prob_reverberation > 0 requires rir_scp")
        if self.sim_cfg.peak_normalization_mode not in {"off", "joint"}:
            raise ValueError("SS supports off/joint peak normalization")
        validate_ss_scenarios(self.sim_cfg)
        if (self.sim_cfg.prob_target_noise + self.sim_cfg.prob_noise_only) > 0 and not self._has_background:
            raise ValueError("SS background scenes require noise_scp, windnoise_scp, or music_scp")
        self.contract = build_ss_dataset_contract(
            sample_rate=sample_rate,
            num_sources=num_sources,
        )

    def _render_spatial_scene(self, sources, source_uids, scenario, work_sr, index, latent_reference):
        """timeline → 单次空间卷积 → 共增益音量 → SS SIR → background。

        timeline/volume 独占 303/304 子流，不改变已有 scene/SIR/background 随机流。
        dry 监督使用 placed sources，轨迹/timebase 与固定 K 的 source 顺序保持不变。
        """
        spatial = self._simulation_view.shared.spatial
        provider = getattr(self, "_spatial_rir_provider", None)
        if provider is None:
            raise ValueError("spatial RIR provider is not configured")
        named_rng = SimulationRNG.for_task(self, index, "ss", rng_factory=scene_rng)
        rng = named_rng.stream("speech_scene")
        moving = bool(
            self.sim_cfg.prob_moving_rir > 0.0
            and rng.random() < self.sim_cfg.prob_moving_rir
        )
        scene_plan = prepare_scene_plan(
            provider=provider,
            task="ss", sample_index=index, sample_rate=work_sr,
            num_samples=self.segment_length, source_uids=tuple(source_uids),
            spatial_view=spatial, simulation_cfg=self.sim_cfg, rng=rng, moving=moving,
        )
        sources, timeline = place_scene_sources(
            sources, self._simulation_view.shared.temporal, named_rng.stream("timeline")
        )
        request = SpatialSceneRequest(
            sources=np.stack(tuple(np.asarray(source, dtype=np.float32) for source in sources)),
            sample_rate=work_sr,
            provider=provider,
            source_uids=tuple(source_uids),
            mode="moving" if scene_plan.timebase is not None else "static",
            # SS 的监督只需要 full observation 与 early target 两类 track。
            track_kinds=("full", "early"),
            early_boundary=max(0, int(round(self.sim_cfg.early_rir_sec * work_sr))),
            reference_channel=spatial.reference_channel,
            rng=named_rng.stream("scene_plan"),
            num_samples=self.segment_length,
            normalize=self.sim_cfg.rir_backend == "online",
            trajectory_points=int(getattr(self.sim_cfg, "trajectory_points", 4)),
            max_displacement_m=float(getattr(self.sim_cfg, "max_displacement_m", 0.25)),
            scene_plan=scene_plan,
        )
        rendered = SpatialSceneExecutor().execute(request)
        full_tracks = rendered.full_tracks.copy()
        early_tracks = rendered.early_tracks.copy()
        targets = (
            np.stack([np.asarray(source, dtype=np.float32) for source in sources])
            if self.sim_cfg.target_type == "dry"
            else early_tracks[:, spatial.reference_channel].copy()
        )
        full_tracks, early_tracks, targets, volume_calibration = calibrate_spatial_source_volume(
            full_tracks, early_tracks, targets, self.sim_cfg, work_sr,
            named_rng.stream("source_volume"),
        )
        sir_facts = []
        if len(sources) > 1:
            numerator = EnergyRatioScaler.measure(full_tracks[0], policy="full_sum_energy")
            for source_index in range(1, len(sources)):
                denominator = EnergyRatioScaler.measure(
                    full_tracks[source_index], policy="full_sum_energy"
                )
                selected_db = float(rng.uniform(*self.sir_range))
                if numerator <= 1e-10 or denominator <= 1e-10:
                    sir_facts.append({"sources": [source_index], "selected_db": selected_db,
                                      "realized_db": None, "reason": "inactive_source"})
                    continue
                scaled = EnergyRatioScaler.sum_energy_ratio(numerator, denominator, selected_db)
                full_tracks[source_index] *= np.float32(scaled.gain)
                early_tracks[source_index] *= np.float32(scaled.gain)
                targets[source_index] *= np.float32(scaled.gain)
                sir_facts.append({"sources": [source_index], "selected_db": selected_db,
                                  "realized_db": scaled.realized_db, "gain": scaled.gain,
                                  "reason": None})
        speech_sum = full_tracks.sum(axis=0, dtype=np.float32)
        reference = full_tracks[0] if self.sim_cfg.snr_reference == "primary" else speech_sum
        from core.data.simulation.background_resolver import resolve_and_mix_background
        resolve_rng = named_rng.stream("background_resolution")
        mix_rng = named_rng.stream("background_mix")
        # 点源 provider 的 geometry 绑定 scene_plan.array；diffuse 路径保持既有
        # view geometry，确保默认行为字节级不变。
        background_mix, background_spec = resolve_and_mix_background(
            dataset=self,
            speech_image=speech_sum,
            reference_image=reference,
            sample_rate=work_sr,
            actual_length=self.segment_length,
            max_duration_samples=self.segment_length,
            rng=resolve_rng,
            mix_rng=mix_rng,
            need_noise=scenario.noise,
            snr=(
                float(named_rng.stream("snr").uniform(self.sim_cfg.snr_low, self.sim_cfg.snr_high))
                if scenario.noise else None
            ),
            scene_plan=scene_plan,
            provider=provider,
            image_geometry=scene_plan.array.positions_m,
            mix_image_geometry=spatial.microphone_positions,
            shared_scene=True,
        )
        background = background_mix.additive
        noise_provenance = background_mix.noise_image_provenance
        if not scenario.speech:
            background = np.asarray(background, dtype=np.float32)
            targets = np.zeros((self.num_sources, self.segment_length), dtype=np.float32)
            observation = background
            full_tracks = np.zeros_like(full_tracks)
            early_tracks = np.zeros_like(early_tracks)
        else:
            observation = background_mix.observation
        scene_spec = SceneSpec(
            scene_name=validate_scene_name("ss", scenario.name),
            target_present=scenario.speech, interference_present=False,
            background_present=scenario.noise, echo_present=False,
            source_count=self.num_sources if scenario.speech else 0, task="ss",
        )
        tracks = {"mixture": observation.astype(np.float32), "background": background.astype(np.float32)}
        tracks.update({f"source_full_rir/{i}": full_tracks[i] for i in range(full_tracks.shape[0])})
        tracks.update({f"target/{i}": targets[i] for i in range(targets.shape[0])})
        facts = {"version": 1, "rir_backend": self.sim_cfg.rir_backend,
                 "sir": sir_facts, "sir_mode": "per_source", "sir_range": list(self.sir_range),
                 "source_volume_calibration": volume_calibration or None,
                 "sir_measure_point": "post_rir_pre_background", "sir_policy": "full_sum_energy",
                 "source_count": self.num_sources if scenario.speech else 0,
                 "timeline": timeline,
                 "activity": {
                     **scene_activity_facts(tuple(np.max(np.abs(track), axis=0) for track in full_tracks)),
                     "measure_point": "post_rir_pre_background",
                 },
                 "noise_image": noise_provenance,
                 "background_policy": background_spec.to_dict(),
                 "rng": named_rng.stream_facts(),
                 "rendered_scene_plan": ScenePlan(scene_spec).to_dict(),
                 "spatial_scene": rendered.provenance}
        return SpeechScene(WaveformBundle(tracks), facts, background_mix.used_sidechain_wind)

    def set_epoch(self, epoch: int, base_seed: int | None = None) -> None:
        if type(epoch) is not int or (base_seed is not None and type(base_seed) is not int):
            raise TypeError("epoch and base_seed must be exact ints")
        self._external_base_seed = getattr(self, "_external_base_seed", 0) if base_seed is None else base_seed
        super().set_epoch(epoch, base_seed=self.seed * 1_000_003 + self._external_base_seed)

    def _speech_splice_candidate_uids(self, anchor_uid):
        return self._speaker_records[self._identity_by_uid[anchor_uid]]

    def get_source_length(self):
        """metadata 描述最终固定画布，不使用原始短语音的长度。"""
        return [self.segment_length] * len(self)

    def get_srs(self):
        return [self.sample_rate] * len(self)

    def _validate_simulation_cfg(self, cfg):
        validate_ss_scenarios(cfg)
        if cfg.rir_backend == "online" and self.num_sources > cfg.gpurir_max_sources:
            raise ValueError("SS num_sources exceeds gpurir_max_sources budget")

    def _scenario_probabilities(self):
        scenario = self._simulation_view.scenario
        return {
            "prob_target_only": float(scenario.prob_target_only),
            "prob_target_noise": float(scenario.prob_target_noise),
            "prob_noise_only": float(scenario.prob_noise_only),
        }

    def _wind_noise_scenario_probs(self):
        scenario = self._simulation_view.scenario
        return (scenario.prob_target_noise, scenario.prob_noise_only)

    def _is_nonrecoverable_sample_error(self, exc):
        """SS 资源/schema 错误保留原异常类型，禁止降级为静音样本。"""
        return super()._is_nonrecoverable_sample_error(exc)

    def _read_one_speech(self, path, sr):
        waveform, source_sr = load_audio(path)
        waveform = ensure_mono(ensure_finite(waveform, strict=True, track="source", sample=path))
        record = self._records_by_path[path]
        try:
            validate_loaded_audio_metadata(
                waveform,
                source_sr,
                expected_sr=record.sample_rate,
                expected_length=record.length,
                source=f"SS dynamic source {record.uid}",
            )
        except ValueError as exc:
            from core.data.errors import SimulationContractError

            raise SimulationContractError(str(exc)) from exc
        if source_sr != sr:
            waveform = resample(waveform, source_sr, sr)
        return waveform[0].numpy().astype(np.float32)

    def _placeholder_sample(self, work_sr: int, index: int, uid: str) -> dict:
        """可恢复资源失败保留固定 K 交付合同，不能降级成 K-1。"""
        observation, length = self._make_observation_placeholder(work_sr)
        return {
            "uid": f"placeholder-{uid}-{index}",
            "mixture": torch.from_numpy(observation.copy()),
            "sources": torch.zeros(
                (self.num_sources, length), dtype=torch.float32
            ),
            "length": length,
            "sr": work_sr,
            "source_ids": ["absent"] * self.num_sources,
        }

    def _build_sample(self, index, uid, slen, work_sr, force_branch=None, *, trace):
        named_rng = SimulationRNG.for_task(self, index, "ss", rng_factory=scene_rng)
        scenario, draw = select_ss_scenario(
            self.sim_cfg, named_rng.stream("scenario"), force_branch
        )
        rng = named_rng.stream("scenario")
        anchor_identity = self._identity_by_uid[uid]
        selected = [uid, *sample_distinct_identities(
            self._speaker_records,
            self.num_sources - 1,
            rng,
            exclude=(anchor_identity,),
        )]
        prepared = []
        latent_reference = None
        if scenario.speech:
            for source_index, source_uid in enumerate(selected):
                source_trace = trace.source(f"source/{source_index}")
                source_trace.replace_metadata("uid", source_uid)
                item = self._prepare_clean_speech_result(
                    source_uid, work_sr, self.segment_length,
                    named_rng.stream("speech_preparation", offset=source_index),
                    source_trace=source_trace,
                )
                prepared.append(item.speech_1d)
        else:
            latent_reference = self._prepare_clean_speech_result(
                selected[0], work_sr, self.segment_length,
                named_rng.stream("latent_reference"),
            ).speech_1d
        render_sources = tuple(prepared)
        render_source_uids = tuple(selected)
        latent_spatial = bool(
            self._simulation_view.shared.spatial.spatial_enabled and not scenario.speech
        )
        if latent_spatial:
            if latent_reference is None:
                raise ValueError("SS spatial noise-only scene requires a latent SNR reference")
            render_sources = (latent_reference,)
            render_source_uids = (selected[0],)
        if self._simulation_view.shared.spatial.spatial_enabled:
            scene = self._render_spatial_scene(
                render_sources, render_source_uids, scenario, work_sr, index, latent_reference
            )
        else:
            render_sources, timeline = place_scene_sources(
                render_sources,
                self._simulation_view.shared.temporal,
                named_rng.stream("timeline"),
            )
            actor_inputs = tuple(
                AcousticActor(
                    uid=source_uid,
                    role=AcousticRole.SOURCE_SPEECH,
                    rir_policy=RIRPolicy.MONO_SOURCE,
                    waveform=source,
                )
                for source_uid, source in zip(render_source_uids, render_sources)
            )
            scene = render_speech_scene(
                self, actor_inputs, work_sr, named_rng.stream("speech_scene"),
                sir_range=self.sir_range, sir_mode="per_source",
                need_noise=scenario.noise, num_samples=self.segment_length,
                reference_speech=(latent_reference if not render_sources else None),
                sources_preplaced=True,
                timeline=timeline,
            )
        if latent_spatial:
            background = scene.bundle.require("background").copy()
            observed_plan = ScenePlan(
                SceneSpec(
                    scene_name="reference_scene",
                    target_present=False,
                    interference_present=False,
                    background_present=True,
                    echo_present=False,
                    source_count=0,
                    task=None,
                )
            )
            scene = replace(
                scene,
                bundle=scene.bundle.replace(
                    tracks={"mixture": background, "background": background.copy()}
                ),
                facts={
                    **scene.facts,
                    "activity": {
                        "active_ratios": [], "overlap_ratio": 0.0, "active_source_count": 0
                    },
                    "rendered_scene_plan": observed_plan.to_dict(),
                    "latent_snr_reference_removed_from_observation": True,
                },
            )
        if not scenario.speech:
            zero_targets = dict(scene.bundle.tracks)
            zero_targets.update({
                f"target/{source_index}": np.zeros(self.segment_length, dtype=np.float32)
                for source_index in range(self.num_sources)
            })
            scene = replace(scene, bundle=scene.bundle.replace(tracks=zero_targets))
        bundle_roles = tuple(scene.bundle.tracks)
        observation_roles = tuple(
            role for role in ("mixture", "background") if role in bundle_roles
        )
        protected_roles = tuple(
            role
            for role in bundle_roles
            if role.startswith("target/") or role.startswith("direct/")
        )
        amplitude_roles = tuple(
            role for role in bundle_roles if role.startswith("source_full_rir/")
        )
        post_mix = ObservationPostMixRenderer().apply(
            ObservationPostMixRequest(
                scene_bundle=scene.bundle,
                sample_rate=work_sr,
                rng=named_rng.stream("post_mix"),
                task_policy=self,
                observation_roles=observation_roles,
                protected_roles=protected_roles,
                amplitude_roles=amplitude_roles,
            )
        )
        post_mix_facts = dict(post_mix.facts)
        layer_f = dict(post_mix_facts["layer_f"])
        layer_f.pop("channel_facts", None)
        if not layer_f.get("applied_ops") and not layer_f.get("observation_changed"):
            layer_f["clipping_facts"] = {}
        scene = SpeechScene(
            post_mix.bundle,
            {
                **scene.facts,
                "post_mix_stage_order": list(post_mix.trace.stage_order),
                "post_mix": post_mix_facts,
                "layer_d_facts": dict(post_mix_facts["layer_d_facts"]),
                "layer_e": dict(post_mix_facts["layer_e"]),
                "layer_f": layer_f,
                "final_peak_guard": dict(post_mix_facts["final_peak_guard"]),
                "decomposable_final": bool(post_mix_facts["decomposable_final"]),
            },
            scene.used_sidechain_wind,
        )
        scene.to_scene_bundle()
        scene.facts["branch"] = scenario.name
        scene.facts["scene_name"] = validate_scene_name("ss", scenario.name)
        scene.facts["speech_present"] = scenario.speech
        scene.facts["noise_present"] = scenario.noise
        scene.facts["scenario_uniform_u"] = draw
        scene.facts["source_count"] = self.num_sources
        scene.facts["scene_spec"] = SceneSpec(
            scene_name=validate_scene_name("ss", scenario.name),
            target_present=scenario.speech,
            interference_present=False,
            background_present=scenario.noise,
            echo_present=False,
            source_count=self.num_sources if scenario.speech else 0,
            task="ss",
        ).to_dict()
        trace.replace_metadata("branch", scenario.name)
        trace.replace_metadata("scene_name", validate_scene_name("ss", scenario.name))
        trace.replace_metadata("scene", scene.facts)
        if trace.capture_stems:
            trace.source("scene").replace_metadata("scene", scene.facts)
            for role, waveform in scene.bundle.tracks.items():
                trace.source("scene").replace_stem(role, waveform)
        return SampleBuildResult({
            "uid": f"dynamic-{int(self._worker_epoch[0])}-{index}",
            "mixture": torch.from_numpy(scene.bundle.require("mixture").copy()),
            "sources": [
                torch.from_numpy(scene.bundle.require(f"target/{i}").copy())
                if scenario.speech else torch.zeros(self.segment_length)
                for i in range(self.num_sources)
            ],
            "length": self.segment_length, "sr": work_sr,
            "source_ids": selected if scenario.speech else tuple("absent" for _ in range(self.num_sources)),
        }, trace)
