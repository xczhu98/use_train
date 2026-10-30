"""SE 对共享固定顺序 post-mix pipeline 的 runtime Adapter。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from core.data.dynamic_dataset_base import (
    LayerEResult,
    apply_deterministic_peak_guard,
)
from core.data.peak_normalization import PeakNormalizationResult
from core.data.post_augment import apply_post_augmentations
from core.data.post_mix_pipeline import (
    POST_MIX_STAGE_ORDER,
    PostMixStage,
    PostMixStageResult,
    PostMixEngine,
    PostMixState,
)
from core.data.simulation_views import simulation_view_for
from core.data.waveform_bundle import WaveformBundle
from tasks.se.data.pase_preprocess import apply_pase_peak_preprocess


# SE 保留历史导出名，但 stage 顺序由共享 pipeline 提供单一事实源。
SE_POST_MIX_STAGE_ORDER: tuple[PostMixStage, ...] = POST_MIX_STAGE_ORDER


@dataclass(frozen=True, eq=False)
class SEPostMixTrace:
    """SE 专属 A–F post-mix 执行结果；不包含 AEC 部署前端边界。"""

    initial_state: PostMixState
    results: tuple[PostMixStageResult, ...]
    retains_stage_states: bool = True

    def __post_init__(self) -> None:
        results = tuple(self.results)
        if tuple(result.stage for result in results) != SE_POST_MIX_STAGE_ORDER:
            raise ValueError("SE trace must contain every A–F stage in fixed order")
        if type(self.retains_stage_states) is not bool:
            raise TypeError("retains_stage_states must be bool")
        if not self.retains_stage_states:
            final_state = results[-1].state
            if any(result.state is not final_state for result in results):
                raise ValueError("compact SE trace results must share the final state")
        object.__setattr__(self, "results", results)

    @property
    def stage_order(self) -> tuple[PostMixStage, ...]:
        return tuple(result.stage for result in self.results)

    @property
    def final_state(self) -> PostMixState:
        return self.results[-1].state


def run_se_post_mix_pipeline(
    initial_state: PostMixState,
    adapter: Any,
    *,
    retain_stage_states: bool = True,
) -> SEPostMixTrace:
    """按共享 Engine 执行 SE 的固定 A–F 顺序并保留旧返回类型。"""
    trace = PostMixEngine().run(
        initial_state,
        adapter,
        retain_stage_states=retain_stage_states,
    )
    return SEPostMixTrace(
        initial_state=initial_state,
        results=trace.results,
        retains_stage_states=retain_stage_states,
    )


OBSERVATION = "observation"
TARGET = "target"
ADDITIVE = "additive"
RESIDUAL = "residual"
_REQUIRED_ROLES = (OBSERVATION, TARGET, ADDITIVE, RESIDUAL)


@dataclass(frozen=True)
class SEPostMixResult:
    """一次完整 SE post-mix 执行的显式波形与审计事实。"""

    bundle: WaveformBundle
    trace: SEPostMixTrace
    layer_e: LayerEResult
    layer_f_applied_ops: tuple[str, ...]
    layer_f_observation_changed: bool
    layer_f_clipping_facts: Mapping[str, object]
    layer_d_bundle: WaveformBundle
    layer_d_facts: Mapping[str, object]
    final_guard_facts: Mapping[str, object]

    def to_run_tuple(self, sample_rate: int) -> tuple[np.ndarray, np.ndarray, int]:
        """投影回 `_run` 的 legacy `(clean, noisy, sr)` Interface。"""
        return (
            self.bundle.require(TARGET).copy(),
            self.bundle.require(OBSERVATION).copy(),
            int(sample_rate),
        )


class TaskPostMixAdapter:
    """将 SE helper 映射到共享固定 stage 顺序，不复制任何 DSP。"""

    def __init__(
        self,
        *,
        dataset: Any,
        rng: np.random.Generator,
        sample_rate: int,
        actual_len: int,
        speech_uid: str,
        branch: str,
        used_sidechain_wind: bool,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self._dataset = dataset
        self._rng = rng
        self._sample_rate = int(sample_rate)
        self._actual_len = int(actual_len)
        self._speech_uid = speech_uid
        self._branch = branch
        self._used_sidechain_wind = bool(used_sidechain_wind)
        self._metadata = {} if metadata is None else dict(metadata)
        self._simulation_view = simulation_view_for(dataset, task="se")
        self._peak_normalization: PeakNormalizationResult | None = None
        self._layer_e: LayerEResult | None = None
        self._layer_f_applied_ops: tuple[str, ...] = ()
        self._layer_f_observation_changed = False
        self._layer_f_clipping_facts: dict[str, object] = {}
        self._layer_d_bundle: WaveformBundle | None = None
        self._layer_d_normalize_stage_facts: dict[str, object] | None = None
        self._layer_d_facts: dict[str, object] | None = None
        self._final_guard_facts: dict[str, object] | None = None
        self._pre_finite_completed = False
        self._post_finite_completed = False

    def prepare_initial_bundle(
        self,
        *,
        observation: np.ndarray,
        target: np.ndarray,
        additive: np.ndarray,
        residual: np.ndarray,
        metadata: dict[str, object] | None = None,
    ) -> WaveformBundle:
        """在 immutable state 构造前执行 legacy pre-finite guard。"""
        observation, target = self._dataset._guard_finite(
            observation, target, self._speech_uid, "mix pipeline"
        )
        self._pre_finite_completed = True
        return self.make_initial_bundle(
            observation=observation,
            target=target,
            additive=additive,
            residual=residual,
            metadata=metadata,
        )

    @staticmethod
    def make_initial_bundle(
        *,
        observation: np.ndarray,
        target: np.ndarray,
        additive: np.ndarray,
        residual: np.ndarray,
        metadata: dict[str, object] | None = None,
    ) -> WaveformBundle:
        """显式建立四 role 同 timeline seam。"""
        return WaveformBundle(
            {
                OBSERVATION: np.asarray(observation, dtype=np.float32).ravel(),
                TARGET: np.asarray(target, dtype=np.float32).ravel(),
                ADDITIVE: np.asarray(additive, dtype=np.float32).ravel(),
                RESIDUAL: np.asarray(residual, dtype=np.float32).ravel(),
            },
            metadata={} if metadata is None else metadata,
        )

    @staticmethod
    def state_from_bundle(bundle: WaveformBundle) -> PostMixState:
        missing = tuple(role for role in _REQUIRED_ROLES if role not in bundle.tracks)
        if missing:
            raise ValueError(f"SE post-mix bundle missing roles: {missing}")
        return PostMixState._from_frozen_waveforms(
            {role: bundle.require(role) for role in _REQUIRED_ROLES}
        )

    @staticmethod
    def make_delivery_bundle(
        *, observation: np.ndarray, target: np.ndarray, metadata: dict[str, object]
    ) -> WaveformBundle:
        """在 target-SR/final TimeTransformPlan 后建立最终交付快照。"""
        return WaveformBundle(
            {
                OBSERVATION: np.asarray(observation, dtype=np.float32).ravel(),
                TARGET: np.asarray(target, dtype=np.float32).ravel(),
            },
            metadata=metadata,
        )

    @staticmethod
    def project_sample(
        bundle: WaveformBundle,
        *,
        sample_rate: int,
        length: int,
        pure_flags: tuple[bool, bool, bool],
    ) -> tuple[np.ndarray, np.ndarray, int, int, bool, bool, bool]:
        """把最终 bundle 投影为 SE 7 元组样本，交共享 collate 生成 SEBatch。"""
        if bundle.num_samples != int(length):
            raise ValueError(
                "SE delivery bundle length mismatch: "
                f"bundle={bundle.num_samples}, declared={length}"
            )
        return (
            bundle.require(TARGET).copy(),
            bundle.require(OBSERVATION).copy(),
            int(sample_rate),
            int(length),
            *pure_flags,
        )

    @staticmethod
    def _changed(before: PostMixState, after: PostMixState) -> bool:
        return before is not after

    @staticmethod
    def _state(state: PostMixState, **changes: np.ndarray) -> PostMixState:
        return state._replace_waveforms(changes)

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
            next_state = self._apply_final_guard(state)
        elif stage == "post-finite":
            next_state = self._apply_finite(state, "after augmentation")
            self._post_finite_completed = True
        else:  # pragma: no cover - 共享 pipeline 只会传固定 stage
            raise ValueError(f"unsupported SE post-mix stage: {stage!r}")
        return PostMixStageResult(stage, next_state, self._changed(state, next_state))

    def _apply_finite(self, state: PostMixState, label: str) -> PostMixState:
        observation, target = self._dataset._guard_finite(
            state.waveforms[OBSERVATION],
            state.waveforms[TARGET],
            self._speech_uid,
            label,
        )
        return self._state(state, observation=observation, target=target)

    def _pase_preprocess_cfg(self) -> Mapping[str, object]:
        """返回 Dataset 上的 PASE 专属预处理配置；未配置时为空 mapping。"""
        cfg = getattr(self._dataset, "pase_preprocess", None)
        return {} if cfg is None else cfg

    @staticmethod
    def _peak_normalization_facts(
        result: PeakNormalizationResult,
    ) -> dict[str, object]:
        """投影归一化缩放事实；D-normalize 与 PASE post-E 共用同一组键。"""
        return {
            "mode": result.mode,
            "target_peak": result.target_peak,
            "noisy_scale": result.noisy_scale,
            "clean_scale": result.clean_scale,
            "noise_scale": result.noise_scale,
            "preserves_decomposition": result.preserves_decomposition,
        }

    def _apply_pase_peak_normalize(self, state: PostMixState) -> PostMixState:
        """执行 PASE/StuPASE 独立 peak normalization 并写回四 role 波形。"""
        pase_cfg = self._pase_preprocess_cfg()
        result = apply_pase_peak_preprocess(
            state.waveforms[OBSERVATION],
            state.waveforms[TARGET],
            state.waveforms[ADDITIVE],
            mode=pase_cfg.get("peak_normalization_mode"),
            scale_range=pase_cfg.get(
                "peak_normalization_scale_range",
                self._simulation_view.shared.peak.peak_normalization_scale_range,
            ),
            rng=self._rng,
        )
        residual_scale = result.noisy_scale
        observation = result.noisy
        if result.mode == "single" and self._branch == "prob_target_only":
            residual_scale = result.clean_scale
            observation = np.asarray(result.clean, dtype=np.float32).copy()
        self._peak_normalization = result
        return self._state(
            state,
            observation=observation,
            target=result.clean,
            additive=result.noise,
            residual=(state.waveforms[RESIDUAL] * residual_scale).astype(np.float32),
        )

    def _apply_layer_d_normalize(self, state: PostMixState) -> PostMixState:
        """执行 Layer D 的随机 peak normalization。

        ``pase_preprocess.apply_after_channel=True`` 时 PASE 独立归一化推迟到
        ``task-post-EF``：Layer D 只执行公共 normalize（配置为 off 时等效 no-op），
        对齐官方 UniPASE/GAP 在增强之后逐信号 peak 归一化的顺序。
        """
        pase_cfg = self._pase_preprocess_cfg()
        pase_mode = pase_cfg.get("peak_normalization_mode")
        apply_after_channel = pase_cfg.get("apply_after_channel", False) is True
        if pase_mode is not None and not apply_after_channel:
            next_state = self._apply_pase_peak_normalize(state)
            stage_facts = self._peak_normalization_facts(self._peak_normalization)
        else:
            result = self._dataset._apply_peak_normalization_result(
                state.waveforms[OBSERVATION],
                state.waveforms[TARGET],
                state.waveforms[ADDITIVE],
                self._rng,
            )
            residual_scale = result.noisy_scale
            observation = result.noisy
            if result.mode == "single" and self._branch == "prob_target_only":
                residual_scale = result.clean_scale
                observation = np.asarray(result.clean, dtype=np.float32).copy()
            self._peak_normalization = result
            stage_facts = self._peak_normalization_facts(result)
            if pase_mode is not None:
                # 记录 PASE 被推迟的事实；此时 Layer D 波形不含 PASE 缩放。
                stage_facts["pase_deferred"] = {
                    "configured_mode": pase_mode,
                    "apply_after_channel": True,
                    "applied_at": "task-post-EF",
                }
            next_state = self._state(
                state,
                observation=observation,
                target=result.clean,
                additive=result.noise,
                residual=(state.waveforms[RESIDUAL] * residual_scale).astype(np.float32),
            )
        self._layer_d_normalize_stage_facts = stage_facts
        return next_state

    def _apply_task_post_ef(self, state: PostMixState) -> PostMixState:
        """apply_after_channel=True 时在 Layer E/F 之后补做 PASE 独立归一化。"""
        pase_cfg = self._pase_preprocess_cfg()
        if pase_cfg.get("apply_after_channel", False) is not True:
            return state
        if pase_cfg.get("peak_normalization_mode") is None:
            return state
        if self._layer_d_facts is None:
            raise RuntimeError(
                "task-post-EF PASE normalization requires completed Layer D"
            )
        next_state = self._apply_pase_peak_normalize(state)
        facts = self._peak_normalization_facts(self._peak_normalization)
        facts["apply_after_channel"] = True
        # D-guard 已在 D 阶段执行；post-E 波形交由 final-guard 保护，不重复 D-guard。
        self._layer_d_facts["pase_post_e"] = facts
        return next_state

    def _apply_layer_d_guard(self, state: PostMixState) -> PostMixState:
        """在 normalize 后执行 Layer D 的共同、确定性峰值保护。"""
        if self._layer_d_normalize_stage_facts is None:
            raise RuntimeError("Layer D guard requires completed Layer D normalization")
        guarded = apply_deterministic_peak_guard(
            {
                OBSERVATION: state.waveforms[OBSERVATION],
                TARGET: state.waveforms[TARGET],
                ADDITIVE: state.waveforms[ADDITIVE],
                RESIDUAL: state.waveforms[RESIDUAL],
            }
        )
        next_state = self._state(state, **guarded.waveforms)
        self._layer_d_bundle = WaveformBundle(next_state.waveforms, self._metadata)
        self._layer_d_facts = {
            "stage_order": ("D-normalize", "D-guard"),
            "normalize": dict(self._layer_d_normalize_stage_facts),
            "guard": guarded.facts.to_metadata(),
        }
        return next_state

    def _apply_layer_e(self, state: PostMixState) -> PostMixState:
        result = self._dataset._apply_channel_degradation_result(
            state.waveforms[OBSERVATION],
            self._sample_rate,
            self._actual_len,
            self._rng,
            self._used_sidechain_wind,
        )
        changed = not np.array_equal(state.waveforms[OBSERVATION], result.waveform)
        self._layer_e = (
            result
            if result.observation_changed == changed
            else replace(result, observation_changed=changed)
        )
        return self._state(state, observation=result.waveform)

    def _apply_layer_f(self, state: PostMixState) -> PostMixState:
        if self._layer_e is None:
            raise RuntimeError("Layer F requires a completed Layer E result")
        ops: list[str] = []
        post_mix = self._simulation_view.shared.post_mix
        skip_clip = self._layer_e.operation_applied("clipping")
        layer_f_cfg = replace(post_mix, prob_clip=0.0) if skip_clip else post_mix
        waveform = apply_post_augmentations(
            state.waveforms[OBSERVATION],
            self._sample_rate,
            layer_f_cfg,
            skip_clip=skip_clip,
            skip_codec=self._layer_e.operation_applied("codec"),
            rng=self._rng,
            applied_ops_out=ops,
            clipping_facts_out=self._layer_f_clipping_facts,
            sanitize_nonfinite=not getattr(self._dataset, "strict_load", True),
        )
        changed = not np.array_equal(state.waveforms[OBSERVATION], waveform)
        self._layer_f_observation_changed = changed
        realized_ops = tuple(str(op) for op in ops)
        # dispatcher 仅在操作真实执行后写入 applied_ops_out；波形变化时完整保留
        # 已执行记录，避免与其他操作混合时漏报 drc/codec。
        self._layer_f_applied_ops = realized_ops if changed else ()
        if not changed:
            self._layer_f_clipping_facts.clear()
        return self._state(state, observation=waveform)

    def _apply_final_guard(self, state: PostMixState) -> PostMixState:
        guarded = apply_deterministic_peak_guard(
            {OBSERVATION: state.waveforms[OBSERVATION]}
        )
        self._final_guard_facts = guarded.facts.to_metadata()
        return self._state(state, observation=guarded.waveforms[OBSERVATION])

    def build_result(self, trace: SEPostMixTrace) -> SEPostMixResult:
        """只在完整 pipeline 成功后构造结果，禁止 partial facts 外泄。"""
        if not self._post_finite_completed:
            raise RuntimeError("SE post-mix pipeline did not complete post-finite")
        if self._peak_normalization is None or self._layer_e is None:
            raise RuntimeError(
                "SE post-mix pipeline did not complete D-normalize/Layer E"
            )
        if (
            self._layer_d_bundle is None
            or self._layer_d_normalize_stage_facts is None
            or self._layer_d_facts is None
            or self._final_guard_facts is None
        ):
            raise RuntimeError("SE post-mix pipeline did not complete Layer D/final guard")
        return SEPostMixResult(
            bundle=WaveformBundle(trace.final_state.waveforms, self._metadata),
            trace=trace,
            layer_e=self._layer_e,
            layer_f_applied_ops=self._layer_f_applied_ops,
            layer_f_observation_changed=self._layer_f_observation_changed,
            layer_f_clipping_facts=MappingProxyType(
                dict(self._layer_f_clipping_facts)
            ),
            layer_d_bundle=self._layer_d_bundle,
            layer_d_facts=MappingProxyType(dict(self._layer_d_facts)),
            final_guard_facts=MappingProxyType(dict(self._final_guard_facts)),
        )


__all__ = [
    "ADDITIVE",
    "OBSERVATION",
    "RESIDUAL",
    "SE_POST_MIX_STAGE_ORDER",
    "SEPostMixResult",
    "SEPostMixTrace",
    "run_se_post_mix_pipeline",
    "TARGET",
    "TaskPostMixAdapter",
]
