"""SE 在线动态混合数据集的任务侧编排入口。

``DynamicMixingDataset`` 在训练时按需合成 clean/noisy 对。底层 Layer A/C 仿真、
音频 I/O、缓存和通用 DSP 已迁移至 ``BaseDynamicMixingDataset`` 及 ``core.data``；
本模块仅负责 SE 分支、固定 post-mix Adapter、审计 metadata 与最终时间轴投影。
无需预生成样本对。

Six-layer simulation pipeline (``_run``, aligned with YAML ``simulation_cfg``):

  Layer A  Clean-speech pre-processing (clean only)
           A-1 Speech splice (concatenation with gaps)
           A-2 Speed / pitch / EQ / silence-gap augmentations
           A-3 Fixed max-duration canvas: crop/repeat, or place one utterance
               once when silence-pad mode is selected
           A-4 High-pass filter (70 Hz filtfilt)
  Layer B  Scenario branch selection (after Layer A, before acoustic mixing)
           prob_target_only:  noisy = clean_target (discards mixed noisy)
           prob_noise_only:   target = 0, noisy = additive noise label
           prob_target_noise: target = clean_target, noisy = full mixed signal
           The branch is decided here but only *applied* once Layer C
           has produced clean_target/noisy_speech/additive_noise.
  Layer C  Acoustic mixing (noisy; C-0 RIR also produces clean_target)
           C-0 RIR convolution (clean_target = early RIR reference; noisy = full RIR)
           C-0b Post-RIR target volume (shared gain for full/early/target)
           C-1 Additive noise mix_noise (SNR-controlled)
           C-2 Wind noise (side-chain or additive)
           C-3 Colored noise (white / pink / brown / equalized)
           All branches continue through the fixed post-mix pipeline.
  Layer D Post-mix amplitude conditioning, with fixed internal order:
           D-normalize optional random peak normalization, then D-guard
           deterministic peak guard using one shared scale for all related stems;
           volume control lives at C-0b. PASE/StuPASE 的独立 peak normalization
           默认在 D-normalize 执行；``pase_preprocess.apply_after_channel=true``
           时推迟到 task-post-EF。
  Layer E URGENT2026 channel degradation (bandwidth / clipping / codec / packet loss)
           Applied **only** to noisy.
  Layer F Post-mix distortion simulation (14 degradations, noisy only)
  Dataset boundary: the SE Dataset ends after Layer F. The AEC deployment frontend
           is AEC-only and deferred to Trainer/Inference rather than executed here.

Post-mix order: pre-finite -> D-normalize -> D-guard -> E -> F -> task-post-EF ->
observation-only final peak guard -> post-finite.

Note: After Layer E/F, ``noisy`` (model input) is intentionally **not** identical to
the linear acoustic generation without E/F. This is by design: the model learns to
recover the clean supervision (from Layer C) from a degraded observation.
"""

from __future__ import annotations

from collections.abc import Mapping

from core.data.dynamic_dataset_base import (
    BaseDynamicMixingDataset,
    SampleBuildResult,
    signal_active_rms_dbfs,
    signal_rms_dbfs,
)

import copy
import math
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.data.sampling import SamplingConfig

import numpy as np

from core.data.constants import DEFAULT_PLACEHOLDER_NUM_SAMPLES
from core.data.simulation_audio_utils import _resample_audio
from core.data.simulation.scene_renderer import finish_speech_scene, render_speech_scene
from core.data.simulation.role_policy import AcousticActor, AcousticRole, RIRPolicy
from core.data.simulation.spatial_executor import SpatialSceneExecutor
from core.data.simulation.spatial_scene import SpatialSceneRequest
from core.data.simulation.array_profiles import resolve_microphone_positions
from tasks.se.data.post_mix_adapter import (
    ADDITIVE,
    OBSERVATION,
    RESIDUAL,
    TARGET,
    TaskPostMixAdapter,
    run_se_post_mix_pipeline,
)
from core.data.simulation_configs import SimulationConfigs
from core.data.time_transform import apply_scene_time_transform
from core.data.waveform_bundle import WaveformBundle
from core.data.simulation_views import simulation_view_for
from core.utils.pure_branch_flags import branch_to_pure_flags_se
from core.data.simulation.scene_names import validate_scene_name
from core.data.simulation.scene_spec import SceneSpec
from core.data.simulation.scene_sampler import prepare_scene_plan
from core.data.speech_scene import place_scene_sources

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Layer-D stem keys that need resampling when target_sr differs from work_sr
_LAYER_D_STEM_KEYS: tuple[str, ...] = (
    "layer_d_noisy",
    "layer_d_clean",
    "layer_d_noise_additive",
    "layer_d_noise_residual_mixed",
)


def _normalize_se_force_branch(force_branch: str) -> str:
    """校验 SE 的三路新场景名，不接受已删除的旧 branch 名。"""
    branch = str(force_branch)
    allowed = ("prob_target_only", "prob_noise_only", "prob_target_noise")
    if branch not in allowed:
        raise ValueError(
            f"SE force_branch must be one of {list(allowed)}, got {force_branch!r}"
        )
    return branch


# =====================================================================
# Dataset
# =====================================================================

class DynamicMixingDataset(BaseDynamicMixingDataset):
    """On-the-fly mixing with a full augmentation pipeline.

    When ``export_simulation_stems=True`` (tools only), ``_last_simulation_stems``
    retains a minimal waveform set consistent with training delivery:
    ``final_noisy`` / ``final_clean`` (model input & supervision after target_sr
    and alignment), Layer-D snapshots ``layer_d_*``, and JSON summaries
    ``speech_splice_export``, ``repeat_export``, ``sim_meta``, ``simulation_by_layer``.
    """

    _simulation_task = "se"

    @staticmethod
    def _restore_layer_d_stage_order(metadata: dict[str, object]) -> None:
        """在 legacy 投影中恢复 Layer D 顺序字段的 tuple 契约。"""
        layer_d = metadata.get("layer_d_facts")
        if isinstance(layer_d, dict):
            layer_d["stage_order"] = ("D-normalize", "D-guard")
        simulation_by_layer = metadata.get("simulation_by_layer")
        if isinstance(simulation_by_layer, dict):
            layer_d = simulation_by_layer.get("D")
            if isinstance(layer_d, dict):
                layer_d["stage_order"] = ("D-normalize", "D-guard")

    def _legacy_stems_from_trace(self, trace) -> dict | None:
        """只转换 legacy sidecar，canonical trace 继续使用 JSON list。"""
        legacy = super()._legacy_stems_from_trace(trace)
        if legacy is not None:
            self._restore_layer_d_stage_order(legacy)
        return legacy

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        speech_scp: str,
        noise_scp: str,
        rir_scp: str,
        windnoise_scp: Optional[str] = None,
        speech_length_file: Optional[str] = None,
        simulation_cfg: Optional[dict] = None,
        strict_load: bool = True,
        max_duration: int = 240000,
        max_duration_seconds: Optional[float] = None,
        repeat_to_max_duration: bool = False,
        repeat_tile_silence_gap: bool = False,
        repeat_tile_silence_gap_range: Optional[list] = None,
        filter_by_duration: bool = True,
        filter_min_duration: float = 0.5,
        filter_max_duration: float = 30.0,
        target_sr: Optional[int] = None,
        speech_min_sample_rate: Optional[int] = None,
        speech_selected_sets: Optional[list] = None,
        placeholder_num_samples: int = DEFAULT_PLACEHOLDER_NUM_SAMPLES,
        export_simulation_stems: bool = False,
        simulation_cfg_strict: bool = True,
        sampling_cfg: "SamplingConfig | None" = None,
        music_scp: Optional[str] = None,
        pase_preprocess: Optional[dict] = None,
    ):
        # 缺省混噪概率在任务边界只解析一次；显式值必须参与三路和校验。
        simulation_cfg = dict(simulation_cfg or {})
        if "prob_target_noise" not in simulation_cfg:
            simulation_cfg["prob_target_noise"] = max(
                0.0,
                1.0 - float(simulation_cfg.get("prob_target_only", 0.0))
                - float(simulation_cfg.get("prob_noise_only", 0.0)),
            )
        if pase_preprocess is not None and not isinstance(pase_preprocess, dict):
            raise TypeError("pase_preprocess must be a mapping")
        self.pase_preprocess = copy.deepcopy(pase_preprocess or {})
        unknown_keys = set(self.pase_preprocess) - {
            "peak_normalization_mode",
            "peak_normalization_scale_range",
            "apply_after_channel",
        }
        if unknown_keys:
            raise ValueError(
                "pase_preprocess only supports peak_normalization_mode / "
                "peak_normalization_scale_range / apply_after_channel, "
                f"got unknown keys {sorted(unknown_keys)}"
            )
        apply_after_channel = self.pase_preprocess.get("apply_after_channel", False)
        if type(apply_after_channel) is not bool:
            raise ValueError(
                "pase_preprocess.apply_after_channel must be an exact bool, "
                f"got {apply_after_channel!r}"
            )
        mode = self.pase_preprocess.get("peak_normalization_mode")
        if mode is not None and mode not in {"separate", "single"}:
            raise ValueError(
                "pase_preprocess.peak_normalization_mode must be 'separate' or 'single'"
            )
        scale_range = self.pase_preprocess.get("peak_normalization_scale_range")
        if scale_range is not None:
            # 采样期直接索引 scale_range[1]，长度/递增性/符号必须在构造边界校验，
            # 否则单元素配置在首次采样时 IndexError，反向或负值静默产出错误峰值。
            # 判据与公共 Layer D 的 SimulationConfigs 合同保持一致。
            try:
                scale_values = list(scale_range)
            except TypeError as exc:
                raise ValueError(
                    "pase_preprocess.peak_normalization_scale_range must be a "
                    f"2-element sequence, got {scale_range!r}"
                ) from exc
            if len(scale_values) != 2:
                raise ValueError(
                    "pase_preprocess.peak_normalization_scale_range must be a "
                    f"2-element sequence, got {scale_range!r}"
                )
            try:
                scale_lo, scale_hi = (float(value) for value in scale_values)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "pase_preprocess.peak_normalization_scale_range must contain "
                    f"two finite numbers, got {scale_range!r}"
                ) from exc
            if not (
                math.isfinite(scale_lo)
                and math.isfinite(scale_hi)
                and 0.0 < scale_lo <= scale_hi <= 1.0
            ):
                raise ValueError(
                    "pase_preprocess.peak_normalization_scale_range must satisfy "
                    f"0 < lo <= hi <= 1, got {scale_range!r}"
                )
        super().__init__(
            speech_scp=speech_scp,
            noise_scp=noise_scp,
            rir_scp=rir_scp,
            windnoise_scp=windnoise_scp,
            music_scp=music_scp,
            speech_length_file=speech_length_file,
            simulation_cfg=simulation_cfg,
            strict_load=strict_load,
            max_duration=max_duration,
            max_duration_seconds=max_duration_seconds,
            repeat_to_max_duration=repeat_to_max_duration,
            repeat_tile_silence_gap=repeat_tile_silence_gap,
            repeat_tile_silence_gap_range=repeat_tile_silence_gap_range,
            filter_by_duration=filter_by_duration,
            filter_min_duration=filter_min_duration,
            filter_max_duration=filter_max_duration,
            target_sr=target_sr,
            speech_min_sample_rate=speech_min_sample_rate,
            speech_selected_sets=speech_selected_sets,
            placeholder_num_samples=placeholder_num_samples,
            export_simulation_stems=export_simulation_stems,
            simulation_cfg_strict=simulation_cfg_strict,
            sampling_cfg=sampling_cfg,
        )

    def _validate_simulation_cfg(self, cfg: SimulationConfigs) -> None:
        """Validate SE scenario probabilities.

        SE 的目标-only 与纯噪声概率之和允许恰好为 1.0；剩余概率进入
        ``prob_target_noise``，并由 Layer B 的三路新 branch 名记录。
        """
        def _prob_unit_interval(name: str, raw: float) -> float:
            v = float(raw)
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"simulation_cfg {name} must be in [0.0, 1.0], got {raw!r}"
                )
            return v

        scenario = simulation_view_for(self, task="se").scenario
        prob_target = _prob_unit_interval(
            "prob_target_only", float(scenario.prob_target_only)
        )
        prob_noise = _prob_unit_interval(
            "prob_noise_only", float(scenario.prob_noise_only)
        )
        prob_sum = prob_target + prob_noise
        if prob_sum > 1.0:
            raise ValueError(
                f"simulation_cfg: prob_target_only + prob_noise_only must be <= 1.0 "
                f"(== 1.0 leaves zero mass on prob_target_noise). Got {prob_sum:.6f}"
            )
        total = prob_sum + _prob_unit_interval("prob_target_noise", cfg.prob_target_noise)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("SE three scenario probabilities must sum to 1")

    def _scenario_probabilities(self):
        scenario = self._simulation_view.scenario
        return {
            name: float(getattr(scenario, name))
            for name in ("prob_target_only", "prob_noise_only", "prob_target_noise")
        }


    # ------------------------------------------------------------------
    # Channel augmentation parameter generation (Layer E)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Export helpers (for verify_simulation tools)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Layer E: channel degradation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Full simulation pipeline (_run)
    # ------------------------------------------------------------------
    def _run(
        self,
        speech_uid: str,
        speech_length: int,
        sr: int,
        index: int | None = None,
        force_branch: str | None = None,
        *,
        source_trace=None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Execute the full A–F simulation pipeline.

        Returns:
            (clean_1d, noisy_1d, sr)
        """
        cfg = self.sim_cfg
        max_dur_sr = self._max_duration_for_sr(sr)
        if rng is None:
            rng = self._get_rng(index)
        # ========== Layer A: clean-speech pre-processing ==========
        prepared_speech = self._prepare_clean_speech_result(
            speech_uid,
            sr,
            max_dur_sr,
            rng,
            source_trace=source_trace,
        )
        speech_bundle = prepared_speech.to_bundle() if source_trace is not None else None
        speech_1d = prepared_speech.speech_1d
        speech_2d = prepared_speech.speech_2d
        actual_len = prepared_speech.actual_len

        # ========== Layer B: scenario branch selection (decided here, applied after Layer C) ==========
        prob_noise_cfg = float(getattr(cfg, "prob_noise_only", 0.0))
        p_target = float(getattr(cfg, "prob_target_only", 0.0))
        p_noise = prob_noise_cfg
        branch_path, _ = self._select_branch(
            [
                ("prob_target_only", p_target),
                ("prob_noise_only", p_noise),
                ("prob_target_noise", cfg.prob_target_noise),
            ],
            rng,
            force_branch=force_branch,
            normalize_force=_normalize_se_force_branch,
        )

        # ========== Layer C: acoustic mixing ==========
        need_noise = branch_path in ("prob_noise_only", "prob_target_noise")
        if speech_bundle is None:
            ac = self._mix_acoustic(
                speech_2d, sr, actual_len, max_dur_sr, rng, need_noise
            )
            (
                noisy_speech,
                clean_target,
                noisy_after_rir,
                additive_noise_2d,
            ) = self._align_mix_acoustic_outputs(
                ac.noisy_speech,
                ac.clean_target,
                ac.noisy_after_rir,
                ac.additive_noise_2d,
                actual_len,
                sr,
            )
        else:
            acoustic_bundle = self._mix_acoustic_bundle(
                speech_bundle,
                sr,
                max_dur_sr,
                rng,
                need_noise,
                source_trace=source_trace,
            )
            ac = acoustic_bundle.acoustic_output
            noisy_speech = acoustic_bundle.bundle.require("noisy_speech")
            clean_target = acoustic_bundle.bundle.require("clean_target")
            noisy_after_rir = acoustic_bundle.bundle.require("noisy_after_rir")
            additive_noise_2d = acoustic_bundle.bundle.require("additive_noise")

        noise_additive_1d = additive_noise_2d.flatten().astype(np.float32)
        noise_residual_1d = (noisy_speech - noisy_after_rir).flatten().astype(np.float32)

        # ========== Apply Layer B branch selection to Layer C mixing outputs ==========
        if branch_path == "prob_target_only":
            noisy_1d = clean_target.flatten().astype(np.float32)
            clean_1d = clean_target.flatten().astype(np.float32)
        elif branch_path == "prob_noise_only":
            noisy_1d = noise_additive_1d.copy()
            clean_1d = np.zeros(actual_len, dtype=np.float32)
        else:
            noisy_1d = noisy_speech.flatten().astype(np.float32)
            clean_1d = clean_target.flatten().astype(np.float32)

        # ========== Post-mix: pre-finite → D-normalize → D-guard → E → F → task-post-EF → final-guard → post-finite ==========
        adapter = TaskPostMixAdapter(
            dataset=self,
            rng=rng,
            sample_rate=sr,
            actual_len=actual_len,
            speech_uid=speech_uid,
            branch=branch_path,
            used_sidechain_wind=ac.used_sidechain_wind,
            metadata={"branch": branch_path, "sample_rate": sr},
        )
        initial_bundle = adapter.prepare_initial_bundle(
            observation=noisy_1d,
            target=clean_1d,
            additive=noise_additive_1d,
            residual=noise_residual_1d,
            metadata={"branch": branch_path, "sample_rate": sr},
        )
        post_mix_trace = run_se_post_mix_pipeline(
            adapter.state_from_bundle(initial_bundle),
            adapter,
            retain_stage_states=bool(
                source_trace is not None and source_trace.capture_stems
            ),
        )
        post_mix_result = adapter.build_result(post_mix_trace)
        clean_1d, noisy_1d, _ = post_mix_result.to_run_tuple(sr)
        layer_e_result = post_mix_result.layer_e
        layer_e_metadata = layer_e_result.to_metadata()
        layer_f_applied_ops = post_mix_result.layer_f_applied_ops
        layer_f_metadata = {
            "noisy_post_mix_degradations_enabled": bool(
                getattr(cfg, "noisy_post_mix_degradations_enabled", False)
            ),
            "applied_ops": list(layer_f_applied_ops),
            "observation_changed": post_mix_result.layer_f_observation_changed,
            "clipping_skipped_due_to_layer_e": layer_e_result.operation_applied(
                "clipping"
            ),
            "clipping_facts": dict(post_mix_result.layer_f_clipping_facts),
        }
        channel_aug_names = list(layer_e_result.selected)
        layer_d_facts = copy.deepcopy(dict(post_mix_result.layer_d_facts))

        layer_d_noisy = post_mix_result.layer_d_bundle.require(OBSERVATION).copy()
        layer_d_clean = post_mix_result.layer_d_bundle.require(TARGET).copy()
        noise_additive_after_d = post_mix_result.layer_d_bundle.require(ADDITIVE).copy()
        noise_residual_after_d = post_mix_result.layer_d_bundle.require(RESIDUAL).copy()
        noise_additive_1d = noise_additive_after_d.copy()
        simulation_by_layer = {
            "B": {"branch": branch_path, "force_branch": force_branch},
            "C": {
                "need_additive_noise": bool(need_noise),
                "c0_reverb_sampled": bool(ac.use_reverb),
                "source_volume_calibration": ac.source_volume_calibration,
                "acoustic_audit": ac.acoustic_audit,
                "c1_c2_snr_primary_drawn_db": ac.snr_primary_drawn_db,
                "c1_c2_wind_noise_snr_candidate_drawn_db": ac.wind_noise_snr_candidate_db,
                "c1_c2_additive_snr_db_each_mix": [
                    float(x) for x in ac.additive_snr_db_list
                ],
                "c1_c2_wind_sidechain_applied": bool(ac.used_sidechain_wind),
                "c1_c2_mix_two_segments_drawn": bool(ac.mix_two),
                "c1_c2_want_sidechain_drawn": bool(ac.want_sidechain),
            },
            "D": copy.deepcopy(layer_d_facts),
            "E": layer_e_metadata,
            "F": layer_f_metadata,
            "final_peak_normalize": dict(post_mix_result.final_guard_facts),
            "realized": {
                "clean_dbfs_rms": signal_rms_dbfs(clean_1d),
                "mic_dbfs_rms": signal_rms_dbfs(noisy_1d),
                "noise_dbfs_rms": signal_rms_dbfs(noise_additive_1d),
                "clean_active_dbfs": signal_active_rms_dbfs(clean_1d),
                "mic_active_dbfs": signal_active_rms_dbfs(noisy_1d),
                "mic_peak_abs": float(np.max(np.abs(noisy_1d))) if noisy_1d.size else 0.0,
                "mic_clip_frame_ratio": (
                    float(np.mean(np.abs(noisy_1d) >= 1.0 - 0.01))
                    if noisy_1d.size else 0.0
                ),
                "mic_near_fullscale_ratio": (
                    float(np.mean(np.abs(noisy_1d) > 0.99)) if noisy_1d.size else 0.0
                ),
                "measure_point": "post_layer_f",
            },
        }
        sim_meta = {
            "path": branch_path,
            "reverb": bool(ac.use_reverb),
            "wind_sidechain": bool(ac.used_sidechain_wind),
            "wind_noise_snr_db": float(ac.wn_snr),
            "additive_snr_mix_db": [float(x) for x in ac.additive_snr_db_list],
            "mix_two_segments": bool(ac.mix_two),
            "want_sidechain_draw": bool(ac.want_sidechain),
            "n_channel_aug": len(channel_aug_names),
            "channel_aug": list(channel_aug_names),
            "force_branch": force_branch,
        }
        layer_d_stems = {
            "final_clean": clean_1d,
            "final_noisy": noisy_1d,
            "layer_d_noisy": layer_d_noisy,
            "layer_d_clean": layer_d_clean,
            "layer_d_noise_additive": noise_additive_after_d,
            "layer_d_noise_residual_mixed": noise_residual_after_d,
        }
        if source_trace is None and self._last_simulation_stems is not None:
            legacy_stems = dict(self._last_simulation_stems)
            legacy_stems.update(
                {
                    name: np.asarray(waveform, dtype=np.float32).copy()
                    for name, waveform in layer_d_stems.items()
                }
            )
            legacy_stems["sim_meta"] = sim_meta
            legacy_stems["simulation_by_layer"] = simulation_by_layer
            self._last_simulation_stems = legacy_stems
            self._last_run_branch = branch_path
        elif source_trace is not None:
            source_trace.replace_metadata("branch", branch_path)
            if source_trace.capture_stems:
                source_trace.replace_metadata("sim_meta", sim_meta)
                source_trace.replace_metadata("simulation_by_layer", simulation_by_layer)
            source_trace.replace_metadata("acoustic_scene_plan", ac.acoustic_scene_plan.to_dict())
            source_trace.replace_metadata("post_mix_stage_order", list(post_mix_trace.stage_order))
            source_trace.replace_metadata(
                "post_mix_stage_applied",
                {result.stage: result.applied for result in post_mix_trace.results},
            )
            source_trace.replace_metadata("layer_d_facts", copy.deepcopy(layer_d_facts))
            source_trace.replace_metadata("layer_e", layer_e_metadata)
            source_trace.replace_metadata("layer_f", layer_f_metadata)
            source_trace.replace_metadata("final_peak_guard", dict(post_mix_result.final_guard_facts))
            for name, waveform in layer_d_stems.items():
                source_trace.replace_stem(name, waveform)
        return clean_1d, noisy_1d, sr

    @staticmethod
    def _background_geometry_from_rir(geometry) -> dict | None:
        """把 supplied RIR 的 geometry 投影为背景 renderer 可消费的拓扑。"""
        if not isinstance(geometry, Mapping):
            return None
        positions = geometry.get("positions_m")
        if positions is None:
            positions = geometry.get("microphone_positions_m")
        topology = geometry.get("topology")
        if positions is None and isinstance(topology, Mapping):
            positions = topology.get("positions_m")
        if positions is None:
            return None
        result = {"positions_m": positions}
        for key in ("spacing_m", "coordinate_system"):
            value = geometry.get(key)
            if value is None and isinstance(topology, Mapping):
                value = topology.get(key)
            if value is not None:
                result[key] = value
        return result

    def _build_spatial_sample(
        self, index: int, uid: str, work_sr: int, force_branch: str | None, trace
    ) -> SampleBuildResult:
        scene_trace = trace.source("scene")
        rng = self._get_rng(index)
        prepared = self._prepare_clean_speech_result(
            uid,
            work_sr,
            self._max_duration_for_sr(work_sr),
            rng,
            source_trace=scene_trace,
        )
        branch, _ = self._select_branch(
            [
                ("prob_target_only", float(self.sim_cfg.prob_target_only)),
                ("prob_noise_only", float(self.sim_cfg.prob_noise_only)),
                ("prob_target_noise", float(self.sim_cfg.prob_target_noise)),
            ],
            rng,
            force_branch=force_branch,
            normalize_force=_normalize_se_force_branch,
        )
        spatial = self._simulation_view.shared.spatial
        provider = getattr(self, "_spatial_rir_provider", None)
        if provider is None:
            raise ValueError(
                "spatial RIR provider is not configured; configure split-level rir_scp "
                "for offline or use rir_backend='online'"
            )
        moving = bool(
            self.sim_cfg.prob_moving_rir > 0.0
            and rng.random() < self.sim_cfg.prob_moving_rir
        )
        source_waveforms, timeline = place_scene_sources(
            (prepared.speech_1d,), self._simulation_view.shared.temporal, rng
        )
        scene_plan = prepare_scene_plan(
            provider=provider, task="se", sample_index=index, sample_rate=work_sr,
            num_samples=prepared.speech_1d.size, source_uids=(uid,),
            spatial_view=spatial, simulation_cfg=self.sim_cfg, rng=rng, moving=moving,
        )
        rendered = SpatialSceneExecutor().execute(SpatialSceneRequest(
            sources=np.stack(source_waveforms), sample_rate=work_sr, provider=provider,
            source_uids=(uid,), mode="moving" if scene_plan.timebase is not None else "static",
            track_kinds=("full", "early"),
            early_boundary=max(0, int(round(self.sim_cfg.early_rir_sec * work_sr))),
            rts_target_t60_sec=(
                self.sim_cfg.rts_target_t60_sec
                if self.sim_cfg.target_type == "rts" else None
            ),
            rts_time_after_max_sec=self.sim_cfg.rts_time_after_max_sec,
            reference_channel=spatial.reference_channel, rng=rng,
            num_samples=prepared.speech_1d.size, normalize=self.sim_cfg.rir_backend == "online",
            scene_plan=scene_plan,
        ))
        pre_rendered_spatial = rendered
        rirs = None
        background_geometry = dict(rendered.geometry)
        cfg = self.sim_cfg
        actor_inputs = (
            AcousticActor(
                uid=uid,
                role=AcousticRole.TARGET_SPEECH,
                rir_policy=RIRPolicy.SPATIAL_SOURCE,
                waveform=source_waveforms[0],
            ),
        )
        scene = render_speech_scene(
            self,
            actor_inputs,
            work_sr,
            rng,
            sir_range=None,
            sir_mode="none",
            # 空间路径的 noise_scp 由本任务统一适配为 [C,T]；避免
            # renderer 再把已经适配的 observation-side image 重复空间化。
            need_noise=False,
            num_samples=prepared.speech_1d.size,
            spatial_rirs=rirs,
            pre_rendered_spatial=pre_rendered_spatial,
            sources_preplaced=True,
            observation_geometry=self._background_geometry_from_rir(background_geometry),
        )
        tracks = dict(scene.bundle.tracks)
        from core.data.simulation.background_resolver import resolve_and_mix_background
        background, background_spec = resolve_and_mix_background(
            dataset=self,
            speech_image=tracks["mixture"],
            reference_image=tracks["source_full_rir/0"],
            sample_rate=work_sr,
            actual_length=prepared.speech_1d.size,
            max_duration_samples=prepared.speech_1d.size,
            rng=rng,
            need_noise=branch != "prob_target_only",
            snr=(
                float(rng.uniform(cfg.snr_low, cfg.snr_high))
                if branch != "prob_target_only" else None
            ),
            scene_plan=scene_plan,
            provider=provider,
            image_geometry=scene_plan.array.positions_m,
            shared_scene=True,
        )
        tracks["mixture"] = background.observation
        tracks["background"] = background.additive
        image_facts = background.noise_image_provenance or {}
        realized_snr = image_facts.get("snr_realized_db")
        scene.facts["background_snr_values"] = [] if realized_snr is None else [realized_snr]
        scene.facts["background_snr_nominal_values"] = list(background.snr_values)
        scene.facts["background_snr_measure_point"] = "post_render_full_image"
        ref_power = float(np.sum(tracks["source_full_rir/0"][cfg.reference_channel].astype(np.float64) ** 2))
        noise_power = float(np.sum(background.additive[cfg.reference_channel].astype(np.float64) ** 2))
        scene.facts["background_snr_reference_channel_db"] = (
            float(10 * np.log10(ref_power / noise_power)) if ref_power > 0 and noise_power > 0 else None
        )
        scene.facts["noise_image_provenance"] = image_facts
        scene.facts["background_policy"] = background_spec.to_dict()
        scene = type(scene)(scene.bundle, scene.facts, background.used_sidechain_wind)
        if branch == "prob_noise_only":
            tracks["mixture"] = tracks["background"].copy()
            tracks["target/0"] = np.zeros_like(tracks["target/0"])
            tracks["source_full_rir/0"] = np.zeros_like(tracks["source_full_rir/0"])
        scene.facts.update(
            {
                "branch": branch,
                "target_present": branch != "prob_noise_only",
                "background_present": branch != "prob_target_only",
                "source_count": 0 if branch == "prob_noise_only" else 1,
                "decomposable_before_post_mix": not background.used_sidechain_wind,
            }
        )
        scene = type(scene)(scene.bundle.replace(tracks=tracks), scene.facts, scene.used_sidechain_wind)
        scene = finish_speech_scene(self, scene, work_sr, rng)
        observation = scene.bundle.require("mixture").copy()
        target = scene.bundle.require("target/0").copy()
        scene_time_result = apply_scene_time_transform(
            bundle=WaveformBundle(
                {OBSERVATION: observation, TARGET: target},
                metadata={"branch": branch, "sample_rate": int(work_sr)},
            ),
            source_sample_rate=work_sr,
            target_sample_rate=self.target_sr,
            rng=rng,
            fixed_silence_canvas_active=bool(getattr(self, "_fixed_silence_canvas_active", False)),
            build_final_time_plan=self._build_final_time_plan,
            resample_adapter=_resample_audio,
            policy_prefix="se_target_sr",
        )
        final_bundle = scene_time_result.bundle
        observation = final_bundle.require(OBSERVATION).copy()
        target = final_bundle.require(TARGET).copy()
        fs_out = scene_time_result.sample_rate
        length = scene_time_result.plan.output_length
        scene_trace.replace_metadata("branch", branch)
        scene_trace.replace_metadata("scene", scene.facts)
        scene_trace.replace_metadata("final_time_transform", scene_time_result.to_metadata())
        scene_trace.replace_stem("final_clean", target)
        scene_trace.replace_stem("final_noisy", observation)
        trace.replace_metadata("branch", branch)
        trace.set_legacy_snapshot("_last_run_branch", branch)
        return SampleBuildResult(
            sample=(
                target,
                observation,
                int(fs_out),
                int(length),
                *branch_to_pure_flags_se(branch),
            ),
            trace=trace,
        )

    # ------------------------------------------------------------------
    # __getitem__ implementation
    # ------------------------------------------------------------------
    def _build_sample(
        self,
        index: int,
        uid: str,
        slen: int,
        work_sr: int,
        force_branch: str | None = None,
        *,
        trace,
    ) -> SampleBuildResult:
        """运行 SE 仿真并返回 sample 与调用方提供的 trace。"""
        if self._simulation_view.shared.spatial.spatial_enabled:
            return self._build_spatial_sample(index, uid, work_sr, force_branch, trace)
        rng_align = self._get_rng(index)

        scene_trace = trace.source("scene")
        speech, noisy, fs_out = self._run(
            uid,
            slen,
            work_sr,
            index=index,
            force_branch=force_branch,
            source_trace=scene_trace,
            rng=rng_align,
        )

        branch = scene_trace.metadata.get("branch")
        scene_trace.replace_metadata("scene_name", validate_scene_name("se", branch))
        scene_trace.replace_metadata(
            "scene_spec",
            SceneSpec(
                scene_name=validate_scene_name("se", branch),
                target_present=branch != "prob_noise_only",
                interference_present=False,
                background_present=branch != "prob_target_only",
                echo_present=False,
                source_count=0 if branch == "prob_noise_only" else 1,
                task="se",
            ).to_dict(),
        )
        delivery_tracks = {
            OBSERVATION: np.asarray(noisy, dtype=np.float32).ravel(),
            TARGET: np.asarray(speech, dtype=np.float32).ravel(),
        }
        if trace.capture_stems:
            for stem_key in _LAYER_D_STEM_KEYS:
                waveform = scene_trace.stems.get(stem_key)
                if waveform is not None:
                    delivery_tracks[f"export/{stem_key}"] = np.asarray(
                        waveform, dtype=np.float32
                    ).ravel()
        scene_time_result = apply_scene_time_transform(
            bundle=WaveformBundle(
                delivery_tracks,
                metadata={"branch": branch, "sample_rate": int(fs_out)},
            ),
            source_sample_rate=fs_out,
            target_sample_rate=self.target_sr,
            rng=rng_align,
            fixed_silence_canvas_active=bool(
                getattr(self, "_fixed_silence_canvas_active", False)
            ),
            build_final_time_plan=self._build_final_time_plan,
            resample_adapter=_resample_audio,
            policy_prefix="se_target_sr",
        )
        final_bundle = scene_time_result.bundle
        speech = final_bundle.require(TARGET).copy()
        noisy = final_bundle.require(OBSERVATION).copy()
        fs_out = scene_time_result.sample_rate
        length = scene_time_result.plan.output_length
        resampled = scene_time_result.source_sample_rate != fs_out
        len_pre_final = scene_time_result.plan.input_length
        if scene_time_result.plan.operations and resampled:
            len_pre_final = scene_time_result.plan.operations[0].output_length
        if trace.capture_stems:
            for stem_key in _LAYER_D_STEM_KEYS:
                role = f"export/{stem_key}"
                if role in final_bundle.tracks:
                    scene_trace.replace_stem(stem_key, final_bundle.require(role))
            repeat_export = scene_trace.metadata.get("repeat_export")
            if isinstance(repeat_export, dict):
                repeat_export = dict(repeat_export)
                if resampled:
                    repeat_export["sample_rate"] = int(fs_out)
                    repeat_export["max_duration_samples"] = int(
                        self._max_duration_for_sr(fs_out)
                    )
                    repeat_export["samples_after_repeat_step"] = int(len_pre_final)
                self._enrich_repeat_export_final(
                    repeat_export, fs_out, int(len_pre_final), length
                )
                scene_trace.replace_metadata("repeat_export", repeat_export)

        delivery_bundle = TaskPostMixAdapter.make_delivery_bundle(
            observation=noisy,
            target=speech,
            metadata={
                "branch": branch,
                "sample_rate": fs_out,
                "time_transform_policy": scene_time_result.plan.policy,
            },
        )
        scene_trace.replace_metadata(
            "final_time_transform", scene_time_result.to_metadata()
        )
        scene_trace.replace_stem("final_clean", delivery_bundle.require(TARGET))
        scene_trace.replace_stem("final_noisy", delivery_bundle.require(OBSERVATION))
        pure_flags = branch_to_pure_flags_se(branch)
        trace.replace_metadata("branch", branch)
        trace.set_legacy_snapshot("_last_run_branch", branch)
        return SampleBuildResult(
            sample=TaskPostMixAdapter.project_sample(
                delivery_bundle,
                sample_rate=fs_out,
                length=length,
                pure_flags=pure_flags,
            ),
            trace=trace,
        )

    def _placeholder_sample(
        self, work_sr: int, index: int, uid: str
    ) -> tuple[np.ndarray, np.ndarray, int, int, bool, bool, bool]:
        """Return a silence placeholder when mixing fails."""
        observation, out_len = self._make_observation_placeholder(work_sr)
        clean = np.zeros(out_len, dtype=np.float32)
        return clean, observation, work_sr, out_len, False, False, False
