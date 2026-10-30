"""AEC 对共享固定顺序 post-mix pipeline 的任务侧 Adapter。"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from core.data.dynamic_dataset_base import (
    LayerEResult,
    apply_deterministic_peak_guard,
)
from core.data.noise_source_policy import NoiseSourcePolicy
from core.data.peak_normalization import apply_joint_role_peak_normalization
from core.data.post_augment import apply_post_augmentations
from core.data.simulation.channel_effects import (
    aggregate_layer_e_results,
    apply_post_augmentations_channels,
    dispatch_observation_channels,
)
from core.data.post_mix_pipeline import (
    PostMixStage,
    PostMixStageResult,
    PostMixState,
    PostMixTrace,
    POST_MIX_STAGE_ORDER,
)
from core.data.simulation.observation_post_mix import PostMixRoleContract
from core.data.simulation_audio_utils import _resample_audio
from core.data.simulation_views import simulation_view_for
from core.data.time_transform import (
    ResampleAdapter,
    ResampleOp,
    TimeTransformPlan,
    apply_scene_time_transform as apply_shared_scene_time_transform,
)
from core.data.waveform_bundle import WaveformBundle


NEAREND = "nearend"
MIC = "mic"
FAREND = "farend"
ECHO = "echo"
NOISE = "noise"
_REQUIRED_ROLES = (NEAREND, MIC, FAREND, ECHO, NOISE)


@dataclass
class AECPostMixContext:
    """共享 pipeline 不可见的 AEC branch、DSP 开关与 disk metadata。"""

    branch: str
    task: str
    aec_disk_role: str | None
    aec_disk_extras: dict[str, object]
    used_sidechain_wind: bool
    joint_target_rir_mismatch: bool
    sidechain_wind_mismatch_reason: str | None
    index: int | None

    @classmethod
    def from_aec_disk(
        cls,
        *,
        branch: str,
        task: str,
        aec_disk: dict[str, dict[str, Any]] | None,
        used_sidechain_wind: bool,
        joint_target_rir_mismatch: bool,
        sidechain_wind_mismatch_reason: str | None,
        index: int | None,
    ) -> "AECPostMixContext":
        """从 legacy disk context 提取非波形字段，波形只留在 bundle。"""
        role: str | None = None
        extras: dict[str, object] = {}
        if aec_disk:
            if len(aec_disk) != 1:
                raise ValueError("AEC disk context must contain exactly one role")
            role, role_context = next(iter(aec_disk.items()))
            if not isinstance(role_context, dict):
                raise TypeError("AEC disk role context must be a dict")
            c3_meta = role_context.get("c3_meta")
            if c3_meta is not None:
                extras["c3_meta"] = c3_meta
        return cls(
            branch=str(branch),
            task=str(task),
            aec_disk_role=role,
            aec_disk_extras=extras,
            used_sidechain_wind=bool(used_sidechain_wind),
            joint_target_rir_mismatch=bool(joint_target_rir_mismatch),
            sidechain_wind_mismatch_reason=sidechain_wind_mismatch_reason,
            index=index,
        )

    def build_aec_disk(
        self,
        bundle: WaveformBundle,
        *,
        resampled_from: int | None = None,
    ) -> dict[str, dict[str, Any]] | None:
        """由 bundle 投影 legacy disk context；metadata 不参与时间变换。"""
        if self.aec_disk_role is None:
            return None
        role_context = dict(self.aec_disk_extras)
        if resampled_from is not None and isinstance(role_context.get("c3_meta"), dict):
            c3_meta = dict(role_context["c3_meta"])
            c3_meta["sample_rate"] = int(resampled_from)
            role_context["c3_meta"] = c3_meta
        role_context["farend"] = bundle.require(FAREND).copy()
        role_context["echo"] = bundle.require(ECHO).copy()
        return {self.aec_disk_role: role_context}


@dataclass(frozen=True)
class MicExtraNoiseResult:
    """task-post-EF mic-extra 的波形与来源 policy。"""

    mic: np.ndarray
    extra_noise: np.ndarray | None
    policy: NoiseSourcePolicy | None


@dataclass(frozen=True)
class AECPostMixResult:
    """一次完整 AEC post-mix 执行的波形与显式 stage facts。"""

    bundle: WaveformBundle
    role_contract: PostMixRoleContract
    trace: PostMixTrace
    context: AECPostMixContext
    layer_e: LayerEResult
    layer_e_channel_facts: tuple[Mapping[str, object], ...]
    layer_f_applied_ops: tuple[str, ...]
    layer_f_clipping_facts: Mapping[str, object]
    decomposable: bool
    decomposable_reason_codes: tuple[str, ...]
    layer_d_facts: Mapping[str, object]
    final_guard_facts: Mapping[str, object]
    mic_extra_noise_policy: NoiseSourcePolicy | None
    aec_disk: dict[str, dict[str, Any]] | None


@dataclass(frozen=True)
class AECSceneTimeResult:
    """target-SR 与 final alignment 的单一 scene plan 执行结果。"""

    bundle: WaveformBundle
    plan: TimeTransformPlan
    sample_rate: int
    fixed_silence_canvas_active: bool

    def to_metadata(self) -> dict[str, object]:
        return {
            "policy": self.plan.policy,
            "source_sample_rate": int(
                next(
                    (
                        operation.source_sample_rate
                        for operation in self.plan.operations
                        if isinstance(operation, ResampleOp)
                    ),
                    self.sample_rate,
                )
            ),
            "target_sample_rate": self.sample_rate,
            "input_length": self.plan.input_length,
            "output_length": self.plan.output_length,
            "operations": [type(operation).__name__ for operation in self.plan.operations],
            "required_tracks": list(self.plan.required_tracks),
            "fixed_silence_canvas_active": self.fixed_silence_canvas_active,
        }


def project_training_tuple(
    *,
    dataset: Any,
    bundle: WaveformBundle,
    sample_rate: int,
    index: int,
    uid: str,
    rng: np.random.Generator,
    context: AECPostMixContext,
    resampled_from: int | None = None,
) -> tuple:
    """在最终时间窗把 AEC bundle 投影回现有 11/12 tuple。"""
    missing = tuple(role for role in _REQUIRED_ROLES if role not in bundle.tracks)
    if missing:
        raise ValueError(f"AEC delivery bundle missing roles: {missing}")
    return dataset._assemble_training_tuple(
        bundle.require(NEAREND).copy(),
        bundle.require(MIC).copy(),
        int(sample_rate),
        int(index),
        str(uid),
        context.build_aec_disk(bundle, resampled_from=resampled_from),
        rng,
        bundle.require(NOISE).copy(),
        branch=context.branch,
    )


def apply_scene_time_transform(
    *,
    dataset: Any,
    bundle: WaveformBundle,
    source_sample_rate: int,
    rng: np.random.Generator,
    fixed_silence_canvas_active: bool,
    resample_adapter: ResampleAdapter = _resample_audio,
) -> AECSceneTimeResult:
    """委托共享 scene-time seam，保留 AEC target-SR 策略和结果名称。"""
    source_sample_rate = int(source_sample_rate)
    target_sample_rate = (
        source_sample_rate if dataset.target_sr is None else int(dataset.target_sr)
    )
    result = apply_shared_scene_time_transform(
        bundle=bundle,
        source_sample_rate=source_sample_rate,
        target_sample_rate=target_sample_rate,
        rng=rng,
        fixed_silence_canvas_active=fixed_silence_canvas_active,
        build_final_time_plan=dataset._build_final_time_plan,
        resample_adapter=resample_adapter,
        policy_prefix="aec_target_sr",
    )
    return AECSceneTimeResult(
        bundle=result.bundle,
        plan=result.plan,
        sample_rate=result.sample_rate,
        fixed_silence_canvas_active=result.fixed_silence_canvas_active,
    )


class TaskPostMixAdapter:
    """把 AEC scene roles 映射到共享固定 stage 顺序，不复制 DSP。"""

    def __init__(
        self,
        *,
        dataset: Any,
        rng: np.random.Generator,
        sample_rate: int,
        actual_len: int,
        speech_uid: str,
        context: AECPostMixContext,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self._dataset = dataset
        self._rng = rng
        self._sample_rate = int(sample_rate)
        self._actual_len = int(actual_len)
        self._speech_uid = speech_uid
        self._context = context
        self._metadata = {} if metadata is None else dict(metadata)
        self._simulation_view = simulation_view_for(dataset, task="aec")
        self._layer_d_normalize_stage_facts: dict[str, object] | None = None
        self._layer_e: LayerEResult | None = None
        self._layer_e_results: tuple[LayerEResult, ...] = ()
        self._layer_e_channel_facts: tuple[Mapping[str, object], ...] = ()
        self._layer_f_applied_ops: tuple[str, ...] = ()
        self._layer_f_clipping_facts: dict[str, object] = {}
        self._decomposable: bool | None = None
        self._decomposable_reason_codes: tuple[str, ...] = ()
        self._layer_d_facts: dict[str, object] | None = None
        self._final_guard_facts: dict[str, object] | None = None
        self._mic_extra_noise_policy: NoiseSourcePolicy | None = None
        self._pre_finite_completed = False
        self._post_finite_completed = False
        self._role_contract: PostMixRoleContract | None = None

    @property
    def decomposable(self) -> bool | None:
        return self._decomposable

    def _set_decomposable_reasons(self, reasons: list[str] | tuple[str, ...]) -> None:
        """按 stage 顺序去重并同步最终 decomposable facts。"""
        self._decomposable_reason_codes = tuple(dict.fromkeys(reasons))
        self._decomposable = not self._decomposable_reason_codes

    def _append_decomposable_reason(self, reason: str) -> None:
        self._set_decomposable_reasons((*self._decomposable_reason_codes, reason))

    def prepare_initial_bundle(
        self,
        *,
        nearend: np.ndarray,
        mic: np.ndarray,
        farend: np.ndarray,
        echo: np.ndarray,
        noise: np.ndarray,
    ) -> WaveformBundle:
        """执行 legacy pre-finite guard 后建立五个基础 AEC waveform role。"""
        mic, nearend = self._dataset._guard_finite(
            mic, nearend, self._speech_uid, "mix pipeline"
        )
        self._pre_finite_completed = True
        mic_array = np.asarray(mic, dtype=np.float32)
        spatial = self._simulation_view.shared.spatial
        if mic_array.ndim not in (1, 2) or mic_array.shape[-1] < 1:
            raise ValueError(f"AEC microphone must be [T] or [C,T], got {mic_array.shape}")
        if not spatial.spatial_enabled and mic_array.ndim == 2:
            if mic_array.shape[0] != 1:
                raise ValueError(
                    "multichannel AEC microphone requires spatial_enabled=True"
                )
            mic_array = mic_array[0]
        elif spatial.spatial_enabled and mic_array.ndim == 2:
            if mic_array.shape[0] != spatial.channels:
                raise ValueError(
                    "AEC spatial microphone channel count does not match configuration"
                )
        bundle = WaveformBundle(
            {
                NEAREND: np.asarray(nearend, dtype=np.float32).ravel(),
                MIC: mic_array.copy(),
                FAREND: np.asarray(farend, dtype=np.float32).ravel(),
                ECHO: np.asarray(echo, dtype=np.float32).ravel(),
                NOISE: np.asarray(noise, dtype=np.float32).ravel(),
            },
            metadata=self._metadata,
        )
        self._role_contract = self._build_role_contract(bundle)
        return bundle

    def _build_role_contract(self, bundle: WaveformBundle) -> PostMixRoleContract:
        """为 AEC 的 coupled 五路 scene 建立显式 role/stage 合同。"""
        if self._context.task == "aec":
            nearend_stages = (
                "D-normalize",
                "D-guard",
                "task-post-EF",
                "post-finite",
            )
        else:
            nearend_stages = ("D-normalize", "D-guard", "post-finite")
        mutable_stages = {
            MIC: tuple(stage for stage in POST_MIX_STAGE_ORDER if stage != "pre-finite"),
            NEAREND: nearend_stages,
            FAREND: ("D-normalize", "D-guard"),
            ECHO: ("D-normalize", "D-guard"),
            NOISE: ("D-normalize", "D-guard", "task-post-EF"),
        }
        return PostMixRoleContract(
            observation_roles=(MIC,),
            protected_roles=(FAREND, ECHO),
            amplitude_roles=(NEAREND, NOISE),
            role_shapes={role: tuple(np.asarray(bundle.require(role)).shape) for role in _REQUIRED_ROLES},
            mutable_stages=mutable_stages,
        )

    @staticmethod
    def state_from_bundle(bundle: WaveformBundle) -> PostMixState:
        missing = tuple(role for role in _REQUIRED_ROLES if role not in bundle.tracks)
        if missing:
            raise ValueError(f"AEC post-mix bundle missing roles: {missing}")
        return PostMixState._from_frozen_waveforms(bundle.tracks)

    @staticmethod
    def _changed(before: PostMixState, after: PostMixState) -> bool:
        return before is not after

    @staticmethod
    def _state(state: PostMixState, **changes: np.ndarray) -> PostMixState:
        return state._replace_waveforms(changes)

    def _build_aec_disk(
        self, state: PostMixState
    ) -> dict[str, dict[str, Any]] | None:
        return self._context.build_aec_disk(WaveformBundle(state.waveforms))

    def apply_stage(
        self, stage: PostMixStage, state: PostMixState
    ) -> PostMixStageResult:
        if stage == "pre-finite":
            if not self._pre_finite_completed:
                raise RuntimeError("prepare_initial_bundle must run before the pipeline")
            next_state = state
        elif stage == "D-normalize":
            next_state = self._apply_layer_d_normalize(state)
        elif stage == "D-guard":
            next_state = self._apply_layer_d_guard(state)
        elif stage == "E":
            next_state = self._apply_layer_e(state)
        elif stage == "F":
            next_state = self._apply_layer_f(state)
        elif stage == "task-post-EF":
            next_state = self._apply_task_post_ef(state)
        elif stage == "final-guard":
            next_state = self._apply_final_peak_guard(state)
        elif stage == "post-finite":
            next_state = self._apply_post_finite(state)
        else:  # pragma: no cover - 共享 pipeline 只传固定 stage
            raise ValueError(f"unsupported AEC post-mix stage: {stage!r}")
        if self._role_contract is None:
            # 保留 Adapter 的低层单 stage 测试 seam；完整 pipeline 仍在
            # prepare_initial_bundle 建立同一份合同并先执行 pre-finite。
            self._role_contract = self._build_role_contract(
                WaveformBundle(state.waveforms)
            )
        changed_roles = tuple(
            role
            for role in state.waveforms
            if not np.array_equal(next_state.waveforms[role], state.waveforms[role])
        )
        unknown_roles = tuple(
            role
            for role in changed_roles
            if role not in self._role_contract.mutable_stages
        )
        if unknown_roles:
            raise ValueError(
                f"AEC post-mix stage {stage!r} changed roles outside role contract: "
                f"{unknown_roles!r}"
            )
        stage_rejects = tuple(
            role
            for role in changed_roles
            if stage not in self._role_contract.mutable_stages[role]
        )
        if stage_rejects:
            raise ValueError(
                f"AEC post-mix stage {stage!r} changed roles outside mutable_stages: "
                f"{stage_rejects!r}"
            )
        return PostMixStageResult(stage, next_state, self._changed(state, next_state))

    def _apply_layer_d_normalize(self, state: PostMixState) -> PostMixState:
        """Layer D 保持 AEC coupled normalization；空间非参考声道独立增益。"""
        shared = self._simulation_view.shared
        mode = shared.peak.peak_normalization_mode
        policy = shared.spatial.spatial_layer_d_policy
        if mode not in {"off", "joint"}:
            raise ValueError("AEC Layer D normalization supports only ['off', 'joint']")
        observation = np.asarray(state.waveforms[MIC], dtype=np.float32)
        if observation.ndim == 2 and shared.spatial.spatial_enabled:
            if observation.shape[0] != shared.spatial.channels:
                raise ValueError("AEC spatial mic channel count does not match configuration")
            if policy == "shared":
                policy_result = apply_joint_role_peak_normalization(
                    state.waveforms, anchor_roles=(MIC, NEAREND, ECHO, NOISE),
                    mode=mode, scale_range=shared.peak.peak_normalization_scale_range,
                    rng=self._rng,
                )
                changes = policy_result.waveforms
                facts = dict(policy_result.scale_facts)
                facts["scale"] = float(facts["scale"])
            elif policy == "per_channel":
                peaks = np.max(np.abs(observation), axis=-1)
                if mode == "off":
                    scales = np.ones(observation.shape[0], dtype=np.float32)
                else:
                    desired = self._rng.uniform(
                        *shared.peak.peak_normalization_scale_range,
                        size=observation.shape[0],
                    )
                    scales = (desired / np.maximum(peaks, 1e-8)).astype(np.float32)
                reference_scale = float(scales[shared.spatial.reference_channel])
                changes = {
                    MIC: (observation * scales[:, None]).astype(np.float32),
                    NEAREND: state.waveforms[NEAREND] * reference_scale,
                    ECHO: state.waveforms[ECHO] * reference_scale,
                    NOISE: state.waveforms[NOISE] * reference_scale,
                    FAREND: state.waveforms[FAREND] * reference_scale,
                }
                facts = {
                    "target_peak": float(np.max(peaks, initial=0.0)),
                    "scale": scales.tolist(),
                    "reference_scale": reference_scale,
                }
            else:
                raise ValueError(f"unsupported AEC spatial_layer_d_policy: {policy!r}")
            facts.update({"mode": mode, "policy": policy, "reference_channel": shared.spatial.reference_channel,
                          "preserves_decomposition": True})
            self._layer_d_normalize_stage_facts = facts
            return self._state(state, **changes)

        if policy != "shared":
            raise ValueError(
                "AEC spatial_layer_d_policy='per_channel' requires a channel-first mic observation"
            )
        policy_result = apply_joint_role_peak_normalization(
            state.waveforms, anchor_roles=(MIC, NEAREND, ECHO, NOISE), mode=mode,
            scale_range=shared.peak.peak_normalization_scale_range, rng=self._rng,
        )
        self._layer_d_normalize_stage_facts = {
            "mode": mode, "target_peak": policy_result.scale_facts["target_peak"],
            "scale": float(policy_result.scale_facts["scale"]),
            "preserves_decomposition": True,
        }
        return self._state(state, **policy_result.waveforms)

    def _apply_layer_d_guard(self, state: PostMixState) -> PostMixState:
        """执行 Layer D guard；空间 per_channel 仍让 mono AEC stems 跟随参考增益。"""
        if self._layer_d_normalize_stage_facts is None:
            raise RuntimeError("Layer D guard requires completed Layer D normalization")
        shared = self._simulation_view.shared
        observation = np.asarray(state.waveforms[MIC], dtype=np.float32)
        policy = shared.spatial.spatial_layer_d_policy
        if observation.ndim == 2 and shared.spatial.spatial_enabled and policy == "per_channel":
            roles = (MIC, NEAREND, ECHO, NOISE, FAREND)
            peaks = np.max(np.abs(observation), axis=-1)
            for role in roles[1:]:
                peaks[shared.spatial.reference_channel] = max(
                    float(peaks[shared.spatial.reference_channel]),
                    float(np.max(np.abs(state.waveforms[role]))),
                )
            scales = np.minimum(1.0, 0.99 / np.maximum(peaks, 1e-8)).astype(np.float32)
            reference_scale = float(scales[shared.spatial.reference_channel])
            changes = {
                MIC: observation * scales[:, None],
                NEAREND: state.waveforms[NEAREND] * reference_scale,
                ECHO: state.waveforms[ECHO] * reference_scale,
                NOISE: state.waveforms[NOISE] * reference_scale,
                FAREND: state.waveforms[FAREND] * reference_scale,
            }
            coupled_facts = {
                "roles": list(roles), "threshold": 0.99,
                "peak": float(np.max(peaks, initial=0.0)), "scale": scales.tolist(),
                "applied": bool(np.any(scales < 1.0)), "policy": policy,
                "reference_channel": shared.spatial.reference_channel,
                "reference_scale": reference_scale,
            }
            next_state = self._state(state, **changes)
            mic_facts = dict(coupled_facts)
            mic_facts["roles"] = list(roles[:-1])
            farend_facts = dict(coupled_facts)
            farend_facts["roles"] = [FAREND]
        else:
            if policy != "shared":
                raise ValueError("AEC spatial_layer_d_policy='per_channel' requires spatial mic observation")
            coupled_roles = (MIC, NEAREND, ECHO, NOISE, FAREND)
            coupled_guard = apply_deterministic_peak_guard(state.waveforms, roles=coupled_roles)
            coupled_facts = coupled_guard.facts.to_metadata()
            mic_facts = dict(coupled_facts)
            mic_facts["roles"] = list(coupled_roles[:-1])
            farend_facts = dict(coupled_facts)
            farend_facts["roles"] = [FAREND]
            next_state = self._state(state, **{
                role: coupled_guard.waveforms[role] for role in coupled_roles
            })
        self._layer_d_facts = {
            "stage_order": ("D-normalize", "D-guard"),
            "normalize": dict(self._layer_d_normalize_stage_facts),
            "guard": {"mic_related": mic_facts, "farend": farend_facts},
        }
        self._dataset._export_layer_d_stems(
            layer_d_noisy=next_state.waveforms[MIC], layer_d_clean=next_state.waveforms[NEAREND],
            layer_d_noise=next_state.waveforms[NOISE], layer_d_echo=next_state.waveforms[ECHO],
            farend_ref=next_state.waveforms[FAREND],
        )
        return next_state

    def _apply_final_peak_guard(self, state: PostMixState) -> PostMixState:
        """仅保护最终 mic observation，不改写可分解 stems。"""
        mic_guard = apply_deterministic_peak_guard(
            state.waveforms,
            roles=(MIC,),
        )
        next_state = self._state(state, mic=mic_guard.waveforms[MIC])
        self._final_guard_facts = mic_guard.facts.to_metadata()
        if mic_guard.facts.applied:
            self._append_decomposable_reason("final_guard_observation_only_scale")
        return next_state

    def _apply_layer_e(self, state: PostMixState) -> PostMixState:
        observation = state.waveforms[MIC]
        spatial = self._simulation_view.shared.spatial
        policy = spatial.spatial_layer_e_policy if observation.ndim == 2 else "shared"
        dispatch = dispatch_observation_channels(
            observation,
            policy=policy,
            rng=self._rng,
            apply_channel=lambda channel, channel_rng, _index: self._dataset._apply_channel_degradation_result(
                channel,
                self._sample_rate,
                self._actual_len,
                channel_rng,
                self._context.used_sidechain_wind,
            ),
        )
        self._layer_e_results = tuple(dispatch.channel_results)
        self._layer_e_channel_facts = dispatch.channel_facts
        self._layer_e = aggregate_layer_e_results(dispatch)
        return self._state(state, mic=self._layer_e.waveform)

    def _apply_layer_f(self, state: PostMixState) -> PostMixState:
        if self._layer_e is None or not self._layer_e_results:
            raise RuntimeError("Layer F requires completed Layer E channel results")
        observation = state.waveforms[MIC]
        spatial = self._simulation_view.shared.spatial
        policy = spatial.spatial_layer_f_policy if observation.ndim == 2 else "shared"
        result = apply_post_augmentations_channels(
            observation,
            self._sample_rate,
            self._simulation_view.shared.post_mix,
            policy=policy,
            rng=self._rng,
            layer_e_results=self._layer_e_results,
            apply_fn=apply_post_augmentations,
        )
        self._layer_f_applied_ops = result.applied_ops
        self._layer_f_clipping_facts = result.clipping_facts
        return self._state(state, mic=result.waveform)

    def _apply_task_post_ef(self, state: PostMixState) -> PostMixState:
        if self._layer_e is None:
            raise RuntimeError("task-post-EF requires a completed Layer E result")
        reasons: list[str] = []
        if self._layer_e.observation_changed:
            reasons.append("layer_e_nonlinear")
        if self._layer_f_applied_ops:
            reasons.append("layer_f_nonlinear")
        if self._context.joint_target_rir_mismatch:
            reasons.append("joint_target_rir_mismatch")
        if self._context.sidechain_wind_mismatch_reason is not None:
            reasons.append(self._context.sidechain_wind_mismatch_reason)
        self._set_decomposable_reasons(reasons)

        build_result = getattr(self._dataset, "_maybe_mic_extra_noise_result", None)
        if build_result is None:
            mic, extra_noise = self._dataset._maybe_mic_extra_noise(
                state.waveforms[MIC],
                self._sample_rate,
                self._context.index,
                state.waveforms[NEAREND],
            )
            result = MicExtraNoiseResult(mic, extra_noise, None)
        else:
            result = build_result(
                state.waveforms[MIC],
                self._sample_rate,
                self._context.index,
                state.waveforms[NEAREND],
            )
        if not isinstance(result, MicExtraNoiseResult):
            raise TypeError("mic-extra result must be MicExtraNoiseResult")
        if result.policy is not None and not isinstance(
            result.policy, NoiseSourcePolicy
        ):
            raise TypeError("mic-extra source policy must be NoiseSourcePolicy or None")
        mic = result.mic
        extra_noise = result.extra_noise
        self._mic_extra_noise_policy = result.policy
        if extra_noise is None:
            return self._state(state, mic=mic)
        self._append_decomposable_reason("task_post_ef_mic_extra_noise")
        changes = {
            MIC: mic,
            NOISE: (state.waveforms[NOISE] + extra_noise).astype(
                np.float32, copy=False
            ),
        }
        if self._context.task == "aec":
            changes[NEAREND] = (state.waveforms[NEAREND] + extra_noise).astype(
                np.float32, copy=False
            )
        return self._state(state, **changes)

    def _apply_post_finite(self, state: PostMixState) -> PostMixState:
        mic, nearend = self._dataset._guard_finite(
            state.waveforms[MIC],
            state.waveforms[NEAREND],
            self._speech_uid,
            "after augmentation",
        )
        self._post_finite_completed = True
        return self._state(state, mic=mic, nearend=nearend)

    def build_result(self, trace: PostMixTrace) -> AECPostMixResult:
        """完整成功后才返回 waveforms 与 stage facts。"""
        if not self._post_finite_completed:
            raise RuntimeError("AEC post-mix pipeline did not complete post-finite")
        if (
            self._layer_d_normalize_stage_facts is None
            or self._layer_e is None
            or self._decomposable is None
        ):
            raise RuntimeError(
                "AEC post-mix pipeline did not complete D-normalize/E/task-post-EF"
            )
        if self._layer_d_facts is None or self._final_guard_facts is None:
            raise RuntimeError("AEC post-mix pipeline did not complete D-guard/final")
        bundle = WaveformBundle(trace.final_state.waveforms, self._metadata)
        aec_disk = self._build_aec_disk(trace.final_state)
        return AECPostMixResult(
            bundle=bundle,
            role_contract=self._role_contract,
            trace=trace,
            context=self._context,
            layer_e=self._layer_e,
            layer_e_channel_facts=self._layer_e_channel_facts,
            layer_f_applied_ops=self._layer_f_applied_ops,
            layer_f_clipping_facts=MappingProxyType(
                dict(self._layer_f_clipping_facts)
            ),
            decomposable=self._decomposable,
            decomposable_reason_codes=self._decomposable_reason_codes,
            layer_d_facts=MappingProxyType(dict(self._layer_d_facts)),
            final_guard_facts=MappingProxyType(dict(self._final_guard_facts)),
            mic_extra_noise_policy=self._mic_extra_noise_policy,
            aec_disk=aec_disk,
        )


__all__ = [
    "AECPostMixContext",
    "AECPostMixResult",
    "AECSceneTimeResult",
    "MicExtraNoiseResult",
    "ECHO",
    "FAREND",
    "MIC",
    "NEAREND",
    "NOISE",
    "TaskPostMixAdapter",
    "apply_scene_time_transform",
    "project_training_tuple",
]
