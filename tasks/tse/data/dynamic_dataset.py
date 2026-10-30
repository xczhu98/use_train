"""TSE 动态 Dataset：唯一七路场景管线、身份约束及独立 enrollment。

源选择在本模块，身份校验复用 tasks.tse.manifest_contract，场景编排在
scene_dataset，声学渲染复用 core。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.data.sampling import SamplingConfig

from core.data.acoustic_scene_plan import AcousticScenePlan, C0AssetReplayError, NoisePlan, RIRPlan, SourceGainPlan
from core.data.constants import DEFAULT_PLACEHOLDER_NUM_SAMPLES
from core.data.errors import SimulationContractError
from core.data.dynamic_dataset_base import BaseDynamicMixingDataset, DataSummary, _compute_source_summary
from core.data.simulation_audio_utils import _align_noise_2d_to_length, _read_cached_2d
from core.data.simulation_configs import SimulationConfigs
from core.data.simulation_utils import is_jsonl_source_manifest, is_strict_jsonl_manifest, read_source_scp, rng_choice
from tools.shared.pickle_io import load_pkl_file
from core.data.resource_paths import resolve_config_resource_path
from core.data.simulation.array_profiles import resolve_microphone_array
from core.data.simulation.level_audit import LevelAudit
from core.data.simulation.rir_provider import validate_spatial_rir_provider
from tasks.tse.cue_contract import TSECue, TSECueSpec, TSECueSpecs, TSECues
from tasks.tse.data.collate import make_tse_sample, make_tse_target
from tasks.tse.data.enrollment_augment import EnrollmentAugmentor
from tasks.tse.manifest_contract import (
    SpeakerIdentity,
    build_recording_ids,
    build_speaker_identities,
    parse_enrollment_manifest,
)

logger = logging.getLogger(__name__)


class EnrollmentAugmentationProviderError(SimulationContractError):
    """enrollment 资源或覆盖读取失败，禁止降级为整样本 placeholder。"""


def _is_moving_rir_path(path: str) -> bool:
    """读取 RIR 元数据，识别静态卷积增强器不支持的 moving RIR。"""
    if Path(path).suffix.lower() != ".pkl":
        return False
    try:
        payload = load_pkl_file(path, purpose="inspect TSE enrollment RIR metadata")
    except OSError:
        # 仅「文件缺失/不可读」视为非 moving；解析失败（损坏或非 dict）必须
        # 向上抛，否则会被误判为 static 而绕过 static-only 门禁。
        return False
    return isinstance(payload, dict) and bool(payload.get("moving", False))

# enroll_policy 合法取值：random（train，每样本随机抽注册音）/
# fixed（val，固定取 spk 注册音列表首条，保证可复现）
_ENROLL_POLICIES = ("random", "fixed")

class TSEDynamicMixingDataset(BaseDynamicMixingDataset):
    """On-the-fly TSE scene simulation with per-source acoustic rendering.

    ``export_simulation_stems`` 与 ``get_sample_with_trace(..., capture_stems=True)``
    共享基类开关，按 task-local 合同捕获既有 role-local waveform，不改变 canonical
    dict item。Phase 2C 已隔离 target/interference/alternate_target/enrollment metadata
    role。
    """

    _simulation_task = "tse"




    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        speech_scp: str,
        noise_scp: str | None = None,
        rir_scp: str | None = None,
        spk2enroll_json: str | None = None,
        windnoise_scp: Optional[str] = None,
        music_scp: Optional[str] = None,
        enrollment_duration: float = 8.0,
        enrollment_db: float = -30.0,
        enroll_policy: str = "random",
        enrollment_augment: Mapping[str, object] | None = None,
        targets_per_mixture: int = 1,
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
        placeholder_num_samples: int = DEFAULT_PLACEHOLDER_NUM_SAMPLES,
        export_simulation_stems: bool = False,
        simulation_cfg_strict: bool = True,
        sampling_cfg: "SamplingConfig | None" = None,
        spatial_rir_provider=None,
    ):
        # TSE dynamic identity 仅接受 JSONL 主事实，必须早于基类解析 fail-fast。
        from core.data.simulation.tse_scenarios import prepare_tse_simulation_cfg

        simulation_cfg = prepare_tse_simulation_cfg(simulation_cfg)
        if isinstance(simulation_cfg, Mapping) and "prob_clean_only" in simulation_cfg:
            raise ValueError("prob_clean_only is removed for TSE; use prob_target_only")
        if not is_strict_jsonl_manifest(speech_scp):
            raise ValueError(
                "TSE dynamic speech_scp must be a strict JSONL manifest "
                "(.jsonl with language/dataset_id/speaker_id metadata); "
                "SCP and JSON manifests are not supported"
            )
        self._spk2enroll_json = spk2enroll_json
        if not isinstance(spk2enroll_json, str) or not spk2enroll_json.strip():
            raise ValueError("spk2enroll_json must be a non-empty path")
        self._music_scp = music_scp
        for name, value in (("enrollment_duration", enrollment_duration), ("enrollment_db", enrollment_db)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
        if isinstance(targets_per_mixture, bool) or not isinstance(targets_per_mixture, int):
            raise TypeError(f"targets_per_mixture must be an integer, got {targets_per_mixture!r}")
        self.enrollment_duration = float(enrollment_duration)
        self.enrollment_db = float(enrollment_db)
        self.spatial_rir_provider = (
            validate_spatial_rir_provider(spatial_rir_provider)
            if spatial_rir_provider is not None
            else None
        )
        self.enroll_policy = str(enroll_policy)
        # TSE-local mapping，不进入共享 SimulationConfigs；disabled 不消费主 RNG。
        self.enrollment_augmentor = EnrollmentAugmentor.from_config(enrollment_augment)
        augment_cfg = enrollment_augment if isinstance(enrollment_augment, Mapping) else {}
        self._enrollment_augment_rirs: tuple[tuple[str, str], ...] = ()
        self._enrollment_augment_noises: tuple[tuple[str, str], ...] = ()
        if self.enrollment_augmentor.requires_rir_provider():
            if self.enrollment_augmentor.rir_backend == "offline":
                local_rir_scp = augment_cfg.get("rir_scp")
                if not isinstance(local_rir_scp, str) or not local_rir_scp.strip():
                    raise ValueError(
                        "enrollment_augment.prob_reverb > 0 requires non-empty rir_scp"
                    )
                local_rir_scp = resolve_config_resource_path(local_rir_scp)
                _, _, rir_flat, _ = read_source_scp(local_rir_scp)
                self._enrollment_augment_rirs = tuple(rir_flat.items())
                if not self._enrollment_augment_rirs:
                    raise ValueError(
                        "enrollment_augment.prob_reverb > 0 requires a non-empty RIR provider"
                    )
                moving_paths = [
                    path
                    for _, path in self._enrollment_augment_rirs
                    if _is_moving_rir_path(path)
                ]
                if moving_paths and self.enrollment_augmentor.rir_static_only:
                    moving = set(moving_paths)
                    self._enrollment_augment_rirs = tuple(
                        (uid, path) for uid, path in self._enrollment_augment_rirs if path not in moving
                    )
                    if not self._enrollment_augment_rirs:
                        raise ValueError("enrollment rir_static_only selected an empty static RIR pool")
                elif moving_paths:
                    raise ValueError(
                        "enrollment augmentation supports static RIR only; moving RIR "
                        f"provider is not supported: {moving_paths[0]!r}"
                    )
        if self.enrollment_augmentor.requires_noise_provider():
            local_noise_scp = augment_cfg.get("noise_scp")
            if not isinstance(local_noise_scp, str) or not local_noise_scp.strip():
                raise ValueError(
                    "enrollment_augment.prob_noise > 0 requires non-empty noise_scp"
                )
            local_noise_scp = resolve_config_resource_path(local_noise_scp)
            _, _, noise_flat, _ = read_source_scp(local_noise_scp)
            self._enrollment_augment_noises = tuple(noise_flat.items())
            if not self._enrollment_augment_noises:
                raise ValueError(
                    "enrollment_augment.prob_noise > 0 requires a non-empty noise provider"
                )
        # speaker-axis 多目标（省内存版）：一条 mixture 最多对应
        # targets_per_mixture 个可选目标（1 主目标 + 干扰说话人）；1 = 现状。
        # mixture 不复制，target/enroll 按目标各一份（详见 _build_sample docstring）。
        self.targets_per_mixture = int(targets_per_mixture)
        if self.targets_per_mixture < 1:
            raise ValueError(
                f"targets_per_mixture must be >= 1, got {targets_per_mixture!r}"
            )
        if self.enrollment_duration <= 0:
            raise ValueError(
                f"enrollment_duration must be > 0, got {enrollment_duration!r}"
            )
        if self.enroll_policy not in _ENROLL_POLICIES:
            raise ValueError(
                f"enroll_policy must be one of {list(_ENROLL_POLICIES)}, got {enroll_policy!r}"
            )
        super().__init__(
            speech_scp=speech_scp,
            noise_scp=noise_scp,
            rir_scp=rir_scp,
            windnoise_scp=windnoise_scp,
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
            placeholder_num_samples=placeholder_num_samples,
            export_simulation_stems=export_simulation_stems,
            simulation_cfg_strict=simulation_cfg_strict,
            sampling_cfg=sampling_cfg,
            spatial_rir_provider=spatial_rir_provider,
        )
        default_enroll_sr = self.target_sr or min(self.get_srs())
        default_enroll_samples = max(
            1, int(round(self.enrollment_duration * default_enroll_sr))
        )
        # 多目标 Layer D 的 normalize stage 只允许可对备选目标施加 follower
        # scale 的 joint，或完全直通的 off；separate/single 没有可保持
        # speaker-axis 相对电平的一致缩放合同。
        peak_mode = str(getattr(self.sim_cfg, "peak_normalization_mode", "off"))
        if self.targets_per_mixture > 1 and peak_mode not in {"off", "joint"}:
            raise ValueError(
                "targets_per_mixture > 1 supports Layer D normalize "
                "peak_normalization_mode only "
                f"in ['off', 'joint'], got {peak_mode!r}"
            )
        if (
            self.enrollment_augmentor.requires_rir_provider()
            and self.enrollment_augmentor.rir_backend == "online"
            and self._online_rir_provider is None
        ):
            raise ValueError(
                "enrollment_augment.rir_backend='online' requires the online RIR "
                "provider; check gpurir_* geometry config and dataset init order"
            )
        spatial = self._simulation_view.shared.spatial
        self.spatial_rir_provider = (
            self._spatial_rir_provider
            if spatial.spatial_enabled
            else spatial_rir_provider
        )
        self.resolved_microphone_array = (
            resolve_microphone_array(
                array_profile=self.sim_cfg.array_profile,
                custom_geometry=self.sim_cfg.custom_geometry,
                reference_channel=self.sim_cfg.reference_channel,
            )
            if spatial.spatial_enabled
            else None
        )
        self.spatial_representation = None
        self.spatial_speed_of_sound_mps = None
        if spatial.spatial_enabled:
            moving = self.sim_cfg.prob_moving_rir > 0.0
            if self.sim_cfg.rir_backend == "online":
                self.spatial_speed_of_sound_mps = self.sim_cfg.gpurir_sound_speed_mps
            else:
                from core.data.simulation.scene_manifest import OfflineSceneIndex

                manifest = getattr(self.spatial_rir_provider, "scene_manifest", None)
                scenes = (
                    [manifest.get(key) for key in manifest.records]
                    if isinstance(manifest, OfflineSceneIndex) else [manifest]
                )
                moving = any(getattr(scene, "metadata", {}).get("moving", False) for scene in scenes)
                speeds = [getattr(scene, "room", {}).get("sound_speed") for scene in scenes]
                if speeds and all(speed is not None for speed in speeds):
                    if len(set(speeds)) != 1:
                        raise ValueError("dynamic TSE spatial scenes require one actual sound_speed")
                    self.spatial_speed_of_sound_mps = speeds[0]
            self.spatial_representation = "track" if moving else "static"
        if self.spatial_representation == "static":
            spatial_spec = TSECueSpec("static", (2,))
        elif self.spatial_representation == "track":
            spatial_spec = TSECueSpec("track", (2, 1))
        else:
            spatial_spec = None
        self.cue_specs = TSECueSpecs(
            speaker_waveform=TSECueSpec("waveform", (default_enroll_samples,)),
            spatial=spatial_spec,
        )

        # ---- canonical speaker identity（uid -> (dataset_id, speaker_id)）----
        self.uid_to_speaker_identity: dict[str, SpeakerIdentity] = (
            build_speaker_identities(
                self._speech_manifest_metadata, speech_scp
            )
        )
        self.spk2enroll = parse_enrollment_manifest(
            spk2enroll_json, set(self.uid_to_speaker_identity.values())
        )
        self._recording_ids = build_recording_ids(
            self._speech_manifest_metadata, self.spk2enroll, speech_scp
        )
        # canonical identity -> manifest-order UID tuple，供目标和每路干扰的
        # speaker-aware splice 共用；同裸 speaker 跨 dataset 不得合池。
        identity_to_uids: dict[SpeakerIdentity, list[str]] = {}
        for uid in self.speech_flt:
            identity = self.uid_to_speaker_identity.get(uid)
            if identity is not None:
                identity_to_uids.setdefault(identity, []).append(uid)
        self.speaker_identity_to_uids: dict[SpeakerIdentity, tuple[str, ...]] = {
            identity: tuple(uids) for identity, uids in identity_to_uids.items()
        }
        # canonical identity -> 稳定连续整数索引（sorted 枚举，跨进程/跨轮次
        # 恒定；查询缺失时回退 -1），供 multi-task CE 辅助任务消费 spk_label。
        # validation 构造后由 entry 注入 train 的 dense 映射，已知 identity 与
        # train label-space 对齐；validation-only identity 保持 -1。
        self.speaker_identity_to_idx: dict[SpeakerIdentity, int] = {
            identity: i
            for i, identity in enumerate(
                sorted(set(self.uid_to_speaker_identity.values()))
            )
        }
        from core.data.background_sources import initialize_music_pool

        if noise_scp is not None or music_scp is not None:
            initialize_music_pool(self, music_scp, extract_embedded=is_jsonl_source_manifest(noise_scp) if noise_scp else False)
        else:
            from core.data.simulation.tse_scenarios import TSE_SCENARIOS

            if any(s.noise and getattr(self.sim_cfg, s.probability_key) > 0 for s in TSE_SCENARIOS):
                raise ValueError("TSE scenes containing background require a noise/music source pool")
            self.music_source, self.music_uids, self.music_flt, self._music_lengths = {}, {}, {}, {}
            self._build_noise_sampling_state()

        # ---- 构造期廉价 join 校验：identity / enrollment 覆盖度 ----
        # 采样期 _build_enrollment 对无候选样本仍会报错；此处提前暴露覆盖率问题。
        speech_uid_set = set(self.speech_flt.keys())
        missing_identity = speech_uid_set - set(self.uid_to_speaker_identity.keys())
        covered_uids = speech_uid_set - missing_identity
        if speech_uid_set and not covered_uids:
            raise ValueError(
                f"speaker identity covers 0/{len(speech_uid_set)} speech uids "
                f"(manifest={speech_scp!r})"
            )
        if missing_identity:
            logger.warning(
                "speaker identity missing %d/%d speech uids (e.g. %s); those samples "
                "will fail at enrollment build time",
                len(missing_identity),
                len(speech_uid_set),
                sorted(missing_identity)[:3],
            )
        covered_identities = {self.uid_to_speaker_identity[u] for u in covered_uids}
        missing_spk2enroll = covered_identities - set(self.spk2enroll.keys())
        if covered_identities and covered_identities == missing_spk2enroll:
            raise ValueError(
                f"spk2enroll_json covers 0/{len(covered_identities)} speaker identities "
                f"referenced by the speech manifest: {spk2enroll_json!r}"
            )
        if missing_spk2enroll:
            logger.warning(
                "spk2enroll_json missing %d/%d speaker identities (e.g. %s); samples of "
                "those speakers will fail at enrollment build time",
                len(missing_spk2enroll),
                len(covered_identities),
                sorted(missing_spk2enroll)[:3],
            )

        # 多目标模式（targets_per_mixture > 1）：干扰说话人也作为备选提取目标，
        # 其 identity 在 spk2enroll 中无候选时采样期 _build_enrollment 会抛错毁掉
        # 整个样本，构造期 fail fast 并要求干扰池（speech 池涉及的全部 identity）
        # 100% 覆盖（含"键存在但候选列表为空"的情况）
        if self.targets_per_mixture > 1:
            missing_alt = sorted(
                identity
                for identity in covered_identities
                if not self.spk2enroll.get(identity)
            )
            if missing_alt:
                raise ValueError(
                    f"targets_per_mixture > 1 requires every speech-pool speaker to "
                    f"have enrollment candidates in spk2enroll_json; missing "
                    f"{len(missing_alt)}/{len(covered_identities)} speaker identities "
                    f"(e.g. {missing_alt[:3]})"
                )

        if self._max_duration_for_sr(self.target_sr or next(iter(self.samplerates))) <= 0:
            raise ValueError("TSE requires a positive fixed canvas duration")

    def _is_nonrecoverable_sample_error(self, exc: Exception) -> bool:
        """asset replay 与 enrollment 合同错误不得降级为 placeholder。"""
        return isinstance(exc, (C0AssetReplayError, EnrollmentAugmentationProviderError)) or super()._is_nonrecoverable_sample_error(exc)

    def _always_load_speech_manifest_metadata(self, speech_scp) -> bool:
        """strict JSONL 的 canonical speaker identity 不依赖 sampling 是否启用。"""
        return is_strict_jsonl_manifest(speech_scp)

    def get_source_length(self) -> list[int]:
        return [self._max_duration_for_sr(sr) for sr in self.get_srs()]

    def _speech_splice_candidate_uids(self, anchor_uid: str) -> tuple[str, ...]:
        """仅返回 anchor canonical identity 内的 splice 候选。"""
        identity = self.uid_to_speaker_identity.get(anchor_uid)
        if identity is None:
            return (anchor_uid,)
        return self.speaker_identity_to_uids.get(identity, (anchor_uid,))

    def _defer_noise_sampling_state(self) -> bool:
        """TSE 需先把内嵌 music role 移出 ordinary pool。"""
        return True

    def _wind_noise_scenario_probs(self):
        from core.data.simulation.tse_scenarios import TSE_SCENARIOS

        return tuple(
            getattr(self.sim_cfg, s.probability_key)
            for s in TSE_SCENARIOS
            if s.noise
        )

    def _needs_online_rir_provider(self, cfg):
        if super()._needs_online_rir_provider(cfg):
            return True
        # enrollment 在线近场 RIR 生成复用 mixture 的在线服务（enrollment_augmentor
        # 在 super().__init__ 之前已构造）。
        augmentor = getattr(self, "enrollment_augmentor", None)
        return (
            augmentor is not None
            and augmentor.rir_backend == "online"
            and augmentor.requires_rir_provider()
        )

    def _validate_simulation_cfg(self, cfg: SimulationConfigs) -> None:
        from core.data.simulation.tse_scenarios import validate_tse_scenarios

        validate_tse_scenarios(cfg)

    def _scenario_probabilities(self):
        from core.data.simulation.tse_scenarios import TSE_SCENARIOS

        return {s.name: getattr(self.sim_cfg, s.probability_key) for s in TSE_SCENARIOS}

    def _has_speech_for_rir(self):
        from core.data.simulation.tse_scenarios import TSE_SCENARIOS

        return any((s.target or s.interference) and getattr(self.sim_cfg, s.probability_key) > 0 for s in TSE_SCENARIOS)

    # ------------------------------------------------------------------
    # Layer C 噪声源扩展：每次 source read 独立按 music_ratio 决定 family
    # ------------------------------------------------------------------
    def _select_tse_additive_source(self, sr, rng, **kwargs):
        from core.data.background_sources import select_source

        return select_source(self, sr, rng, **kwargs)

    def _select_tse_dual_sources(self, sr, rng):
        from core.data.background_sources import select_dual_sources

        return select_dual_sources(self, sr, rng, selector=self._select_tse_additive_source)

    def _read_one_additive_noise_scp_only(self, sr, max_duration, rng, **kwargs):
        from core.data.background_sources import read_source

        source = self._select_tse_additive_source(sr, rng)
        if source is None:
            raise RuntimeError(f"No additive noise source for sr={sr}.")
        return read_source(self, source, sr, max_duration, rng, spatial=False, **kwargs)

    def _read_dual_additive_noise_scp_blended(self, sr, max_duration, rng):
        from core.data.background_sources import read_dual

        return read_dual(
            self,
            sr,
            max_duration,
            rng,
            selector=self._select_tse_dual_sources,
            spatial=False,
        )

    # ------------------------------------------------------------------
    # 含 I 的场景严格采样不同身份；不得因不足而改变场景组成。
    # ------------------------------------------------------------------
    def _sample_interference_uids(
        self, target_uid: str, sr: int, n: int, rng: np.random.Generator
    ) -> list[str]:
        """按 native-SR 池（无该 SR 时用全池）选择 n 个不同的非目标身份。"""
        from core.data.source_pool import sample_distinct_identities

        target_spk = self.uid_to_speaker_identity.get(target_uid)
        cache = getattr(self, "_interference_group_cache", None)
        if cache is None:
            cache = self._interference_group_cache = {}
        if sr not in cache:
            pool = self.speech_uids.get(sr) or [u for group in self.speech_uids.values() for u in group]
            groups = {}
            for uid in pool:
                identity = self.uid_to_speaker_identity[uid]
                groups.setdefault(identity, []).append(uid)
            cache[sr] = {identity: tuple(uids) for identity, uids in groups.items()}
        from core.data.dynamic_dataset_base import SampleBuildContractError

        try:
            return sample_distinct_identities(cache[sr], n, rng, exclude=(target_spk,), shortage="error")
        except ValueError as exc:
            raise SampleBuildContractError(str(exc)) from exc



    # ------------------------------------------------------------------
    # enrollment 仿真（Layer A-enroll + Layer C-enroll）
    # ------------------------------------------------------------------
    def _build_enrollment(
        self,
        target_uid: str,
        sr: int,
        rng: np.random.Generator,
        *,
        role: str = "enrollment/0",
        trace=None,
        level_audits: dict[str, dict] | None = None,
        audit_name: str = "enrollment.snr",
        excluded_paths: set[str] | None = None,
    ) -> np.ndarray:
        """按目标 spk 采样注册音并做 Layer A-enroll + Layer C-enroll 仿真。

        在输出采样率（target_sr 之后）上构建，产出定长
        ``int(enrollment_duration * sr)`` 的 1-D float32。

        设计前提：注册音按样本 uncached 全文件解码（``_read_one_speech``），
        不做缓存——``spk2enroll_json`` 应指向预切好的短注册音（数秒级），
        若直接指向长音频，每样本的重复解码开销会显著拖慢数据管线。
        """
        cfg = self.sim_cfg
        enroll_rir_plan = RIRPlan.disabled(
            early_rir_sec=float(getattr(cfg, "early_rir_sec", 0.0)),
            target_type="enrollment_full",
        )
        enroll_noise_record: dict[str, object] = {
            "noise_sources": [],
            "noise_source_policies": [],
            "noise_assets": [],
        }
        audits = level_audits if level_audits is not None else {}
        audits[audit_name] = LevelAudit(
            name=audit_name,
            candidate_drawn_db=None,
            selected_nominal_db=None,
            calibration_realized_db=None,
            final_realized_db=None,
            numerator="enrollment_reference_active",
            denominator="enrollment_additive_noise_active",
            policy="active_mean_power",
            calibration_measure_point="not_measured",
            final_measure_point="not_measured",
            applicable=False,
            reason="enrollment_noise_not_applied",
        ).to_dict()
        spk = self.uid_to_speaker_identity.get(target_uid)
        cands = self.spk2enroll.get(spk) if spk is not None else None
        if not cands:
            raise EnrollmentAugmentationProviderError(
                f"no enrollment candidates for target uid={target_uid} (identity={spk!r}); "
                "check speech manifest identity / spk2enroll_json coverage"
            )
        if excluded_paths is not None:
            excluded = {str(Path(path).expanduser().resolve()) for path in excluded_paths}
            excluded_recordings = {self._recording_ids[path] for path in excluded if path in self._recording_ids}
            cands = tuple(
                path for path in cands
                if (resolved := str(Path(path).expanduser().resolve())) not in excluded
                and self._recording_ids.get(resolved) not in excluded_recordings
            )
            if not cands:
                raise EnrollmentAugmentationProviderError(
                    f"no independent enrollment recording for target {target_uid!r}; all candidates occur in mixture"
                )
        # enroll_policy: random（train，随机抽）/ fixed（val，固定取首条，可复现）
        enroll_path = (
            cands[0] if self.enroll_policy == "fixed" else rng_choice(rng, cands)
        )
        n_target = max(1, int(round(self.enrollment_duration * sr)))
        enroll = self._read_one_speech(enroll_path, sr)

        # 完整 waveform 链由 TSE-local 无状态执行器承担；Dataset 仅注入 I/O。
        if self.enrollment_augmentor.enabled:
            def _local_rir_provider(
                sample_rate: int, provider_rng: np.random.Generator
            ):
                if self.enrollment_augmentor.rir_backend == "online":
                    provider = self._online_rir_provider
                    if provider is None:
                        raise EnrollmentAugmentationProviderError(
                            "enrollment online RIR provider is not initialized"
                        )
                    group, rir_facts = provider.sample_scene(
                        sample_rate,
                        1,
                        provider_rng,
                        normalize=self.enrollment_augmentor.rir_normalize,
                        distance_range=tuple(
                            self.sim_cfg.gpurir_enrollment_distance_range_m
                        ),
                    )
                    # add_reverberation 要求 mono RIR 形状 (1, T_rir)，与 manifest 静态分支一致。
                    group_uid = str(rir_facts.get("group_uid"))
                    return group[0:1], {
                        "uid": group_uid,
                        "path": f"online://{group_uid}",
                        "backend": "online",
                        "dist": rir_facts.get("dist"),
                        "moving": False,
                    }
                uid, path = self._enrollment_augment_rirs[
                    int(provider_rng.integers(0, len(self._enrollment_augment_rirs)))
                ]
                # local augment 不得随 Dataset ``strict_load=False`` 降级为静音。
                try:
                    rir_2d = _read_cached_2d(
                        path, sample_rate, strict=True, rng=provider_rng
                    )
                except Exception as exc:
                    raise EnrollmentAugmentationProviderError(
                        f"failed to read enrollment RIR provider {path!r}"
                    ) from exc
                return rir_2d, {
                    "uid": str(uid),
                    "path": str(path),
                    "moving": bool(rir_2d.ndim == 2 and rir_2d.shape[0] > 1),
                }

            def _local_noise_provider(
                sample_rate: int, length: int, provider_rng: np.random.Generator
            ):
                uid, path = self._enrollment_augment_noises[
                    int(provider_rng.integers(0, len(self._enrollment_augment_noises)))
                ]
                # local provider 使用自己的资源池，不改变普通 mixture 的读取语义。
                try:
                    noise_2d = _read_cached_2d(
                        path, sample_rate, length, strict=True, rng=provider_rng
                    )
                    noise_2d = _align_noise_2d_to_length(
                        noise_2d, length, provider_rng
                    )
                except Exception as exc:
                    raise EnrollmentAugmentationProviderError(
                        f"failed to read enrollment noise provider {path!r}"
                    ) from exc
                return noise_2d, {"uid": str(uid), "path": str(path)}

        augmentor = self.enrollment_augmentor.for_dynamic(
            duration_seconds=self.enrollment_duration,
            target_db=self.enrollment_db, fit_policy=self.enroll_policy,
        )
        rir_provider = _local_rir_provider if augmentor.requires_rir_provider() else None
        noise_provider = _local_noise_provider if augmentor.requires_noise_provider() else None

        enroll, enrollment_augment_facts = augmentor.apply(
            enroll.astype(np.float32, copy=False), sr, rng,
            rir_provider=rir_provider, noise_provider=noise_provider,
        )
        rir_facts = enrollment_augment_facts.get("rir")
        if rir_facts is not None:
            enroll_rir_plan = RIRPlan(
                enabled=True, uid=rir_facts["uid"], path=rir_facts["path"],
                moving=rir_facts["moving"], reuse_enabled=False, cache_hit=False,
                segment_sec=None, normalize=augmentor.rir_normalize,
                early_rir_sec=float(getattr(cfg, "early_rir_sec", 0.0)),
                target_type="enrollment_full",
            )
        augment_facts = enrollment_augment_facts
        enroll_mix_audit = augment_facts.get("noise_audit")
        enroll_snr_db = augment_facts.get("snr_db")
        enroll_eq_applied = bool(augment_facts.get("eq_post_applied"))
        enroll_peak_rescaled = bool(augment_facts["peak_guard_rescaled"])
        if enroll_mix_audit is not None:
            base_audit = enroll_mix_audit["level_audit"]
            calibration = base_audit["calibration_realized_db"]
            audits[audit_name] = LevelAudit(
                name=audit_name,
                candidate_drawn_db=base_audit["candidate_drawn_db"],
                selected_nominal_db=base_audit["selected_nominal_db"],
                calibration_realized_db=calibration,
                final_realized_db=None if enroll_eq_applied else calibration,
                numerator="enrollment_reference_active",
                denominator="enrollment_additive_noise_active",
                policy="active_mean_power",
                calibration_measure_point="post_reverb_pre_enrollment_eq",
                final_measure_point="post_enrollment_peak_guard",
                applicable=not enroll_eq_applied,
                reason="post_mix_eq_components_unavailable" if enroll_eq_applied else None,
                observations={
                    **base_audit["observations"],
                    "peak_guard_rescaled": float(enroll_peak_rescaled),
                },
            ).to_dict()
        if trace is not None:
            noise_sources = tuple(
                replace(source, snr_db=enroll_snr_db)
                for source in enroll_noise_record["noise_sources"]
            )
            enrollment_plan = AcousticScenePlan(
                sample_rate=int(sr),
                num_samples=int(len(enroll)),
                need_noise=enroll_snr_db is not None,
                rir=enroll_rir_plan,
                source_gain=SourceGainPlan.disabled(),
                noise=(
                    NoisePlan(
                        enabled=True,
                        mode="single",
                        mix_ratio=None,
                        sources=noise_sources,
                        colored_enabled=False,
                        colored_type=None,
                        colored_snr_db=None,
                    )
                    if enroll_snr_db is not None
                    else NoisePlan.disabled()
                ),
                identity={
                    "role": role,
                    "source_path": str(enroll_path),
                    "independent_from_mixture": True,
                },
            )
            source_trace = trace.source(role)
            source_trace.replace_metadata("uid", target_uid)
            source_trace.replace_metadata("level_audit", audits[audit_name])
            source_trace.replace_metadata(
                "acoustic_scene_plan", enrollment_plan.to_dict()
            )
            source_trace.replace_metadata(
                "noise_source_policies",
                [policy.to_dict() for policy in enroll_noise_record["noise_source_policies"]],
            )
            source_trace.replace_metadata(
                "enrollment",
                {
                    "sample_rate": sr,
                    "num_samples": len(enroll),
                    "source_path": str(enroll_path),
                },
            )
            if enrollment_augment_facts is not None:
                source_trace.replace_metadata(
                    "enrollment_augment", enrollment_augment_facts
                )
        return enroll.astype(np.float32, copy=False)

    def _spk_label(self, uid: str) -> int:
        """uid -> 稳定 canonical identity 整数索引；identity 缺失时回退 -1。"""
        spk = self.uid_to_speaker_identity.get(uid)
        return self.speaker_identity_to_idx.get(spk, -1) if spk is not None else -1

    def set_speaker_label_mapping(self, mapping) -> None:
        """复用外部（train）canonical identity → label 映射（validation label
        空间对齐）。key 必须是 canonical ``(dataset_id, speaker_id)`` tuple，
        value 必须是从 0 开始连续且唯一的 int；validation-only identity 经
        ``_spk_label`` 回退 ``-1``，不动态扩展映射。
        """
        if not isinstance(mapping, Mapping):
            raise ValueError("speaker label mapping must be a mapping")
        validated: dict[SpeakerIdentity, int] = {}
        seen_values: set[int] = set()
        for key, value in mapping.items():
            if not (
                isinstance(key, tuple)
                and len(key) == 2
                and all(isinstance(part, str) and part for part in key)
            ):
                raise ValueError(
                    "speaker label mapping keys must be canonical "
                    "(dataset_id, speaker_id) tuples"
                )
            if type(value) is not int or value < 0:
                raise ValueError(
                    "speaker label mapping values must be non-negative ints"
                )
            if value in seen_values:
                raise ValueError("speaker label mapping values must be unique")
            seen_values.add(value)
            validated[key] = value
        if sorted(seen_values) != list(range(len(seen_values))):
            raise ValueError(
                "speaker label mapping values must form a contiguous range "
                "starting at 0"
            )
        self.speaker_identity_to_idx = validated

    def get_data_summary(
        self,
        split: str,
        n_batches: int,
        dataloader_config: dict[str, Any],
    ) -> DataSummary:
        """TSE 数据分布报告：在基类基础上追加 music 池与 speaker/enroll 统计。"""
        summary = super().get_data_summary(split, n_batches, dataloader_config)
        sources = dict(summary.sources)
        if self.music_uids:
            sources["music"] = _compute_source_summary(
                self.music_uids, self._music_lengths
            )
        # 干扰人池与目标语音共享 speech 池，但独立展示便于核对分布。
        sources["interference"] = _compute_source_summary(
            self.speech_uids, self.source_length
        )

        extra: dict[str, Any] = dict(summary.extra or {})
        n_speakers = len(self.spk2enroll)
        extra["speakers"] = n_speakers
        extra["interference_identities"] = len(
            set(self.uid_to_speaker_identity.values())
        )
        if self.spk2enroll:
            cand_counts = [len(cands) for cands in self.spk2enroll.values()]
            extra["enroll_candidates"] = {
                "min": int(min(cand_counts)),
                "max": int(max(cand_counts)),
                "mean": float(np.mean(cand_counts)),
            }
        else:
            extra["enroll_candidates"] = {"min": 0, "max": 0, "mean": 0.0}

        return DataSummary(
            split=summary.split,
            n_samples=summary.n_samples,
            n_batches=summary.n_batches,
            sources=sources,
            dataloader_config=summary.dataloader_config,
            noise_categories=summary.noise_categories,
            scenario_probs=summary.scenario_probs,
            extra=extra,
        )

    # ------------------------------------------------------------------
    # Full simulation pipeline
    # ------------------------------------------------------------------
    def _build_sample(self, index, uid, slen, work_sr, force_branch=None, *, trace):
        """唯一动态入口：七路场景 → 逐源声学 → 独立注册音 → canonical item。"""
        from tasks.tse.data.scene_dataset import build_source_scene_sample

        return build_source_scene_sample(self, index, uid, work_sr, force_branch, trace)

    def _missing_cues(self) -> TSECues:
        values: dict[str, TSECue] = {
            "speaker_waveform": TSECue("waveform", None, False, 0, None),
        }
        if self.spatial_representation == "static":
            values["spatial"] = TSECue("static", None, False, None, None)
        elif self.spatial_representation == "track":
            values["spatial"] = TSECue("track", None, False, 0, None)
        return TSECues(**values)

    def _placeholder_sample(self, work_sr: int, index: int, uid: str):
        """Return a canonical silence placeholder when mixing fails."""
        observation, out_len = self._make_observation_placeholder(work_sr)
        mixture = np.asarray(observation, dtype=np.float32)
        if mixture.ndim == 1:
            mixture = mixture[None, :]
        target = make_tse_target(
            target=np.zeros(out_len, dtype=np.float32),
            cues=self._missing_cues(),
            speaker_id=uid,
            spk_label=-1,
            target_present=True,
        )
        return make_tse_sample(
            key=uid,
            mixture=mixture,
            sr=work_sr,
            length=out_len,
            targets=[target],
            geometry=getattr(self, "resolved_microphone_array", None),
            provenance={"placeholder": True},
        )
