"""Observation-side Layer D/E/F deep Module。

该 Module 只组合现有 ``post_mix_pipeline`` 和
``SourceScenePostMixAdapter``，不复制任何 Layer E/F DSP。调用方提交一个
scene bundle、sample rate、RNG 和 task policy；返回固定顺序的 post-mix trace。
D-normalize/D-guard 对 target 与 source decomposition 同步增益；E/F/final-guard
仍只改变 observation，其他 protected roles 保持 bitwise 不变。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from core.data.post_mix_pipeline import (
    POST_MIX_STAGE_ORDER,
    PostMixEngine,
    PostMixStageResult,
    PostMixState,
    PostMixTrace,
)
from core.data.simulation.scene_renderer import SourceScenePostMixAdapter
from core.data.waveform_bundle import WaveformBundle
from core.data.waveform_scene_bundle import WaveformSceneBundle


_OBSERVATION_MUTABLE_STAGES = frozenset(
    (*POST_MIX_STAGE_ORDER, "C-0", "C-1", "C-2", "C-3")
)


@dataclass(frozen=True)
class PostMixRoleContract:
    """Observation post-mix 的 typed role 合同。

    旧调用方仍可通过 ``ObservationPostMixRequest`` 的 tuple 字段提交角色；
    request 会在边界把它们转换成该合同，内部 orchestration 只消费规范化后的
    role 集合。``derived_roles`` 为任务侧保留的派生 observation role，不自动
    进入 protected 集合。
    """

    observation_roles: tuple[str, ...]
    protected_roles: tuple[str, ...] = ()
    amplitude_roles: tuple[str, ...] = ()
    derived_roles: tuple[str, ...] = ()
    role_shapes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    mutable_stages: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        observation = _role_tuple(self.observation_roles, "observation_roles")
        protected = _role_tuple(self.protected_roles, "protected_roles")
        amplitude = _role_tuple(self.amplitude_roles, "amplitude_roles")
        derived = _role_tuple(self.derived_roles, "derived_roles")
        overlap = (
            (set(observation) & set(protected))
            | (set(amplitude) & set(protected))
            | (set(derived) & set(protected))
        )
        if overlap:
            raise ValueError(f"post-mix role contract has overlapping protected roles: {sorted(overlap)!r}")
        if not observation:
            raise ValueError("observation_roles must not be empty")
        role_shapes = dict(self.role_shapes)
        for role, shape in role_shapes.items():
            if role not in set(observation) | set(protected) | set(amplitude) | set(derived):
                raise ValueError(f"role_shapes contains unknown role: {role!r}")
            if not isinstance(shape, tuple) or any(type(dim) is not int or dim < 0 for dim in shape):
                raise ValueError(f"role_shapes[{role!r}] must be a tuple of non-negative exact ints")
        mutable_stages = {
            role: _role_tuple(stages, f"mutable_stages[{role!r}]")
            for role, stages in dict(self.mutable_stages).items()
        }
        unknown_stages = {
            stage
            for stages in mutable_stages.values()
            for stage in stages
            if stage not in POST_MIX_STAGE_ORDER
        }
        if unknown_stages:
            raise ValueError(
                f"mutable_stages contains unknown post-mix stages: "
                f"{sorted(unknown_stages)!r}"
            )
        known_roles = set(observation) | set(protected) | set(amplitude) | set(derived)
        unknown_mutable = set(mutable_stages) - known_roles
        if unknown_mutable:
            raise ValueError(f"mutable_stages contains unknown roles: {sorted(unknown_mutable)!r}")
        object.__setattr__(self, "observation_roles", observation)
        object.__setattr__(self, "protected_roles", protected)
        object.__setattr__(self, "amplitude_roles", amplitude)
        object.__setattr__(self, "derived_roles", derived)
        object.__setattr__(self, "role_shapes", MappingProxyType(role_shapes))
        object.__setattr__(self, "mutable_stages", MappingProxyType(mutable_stages))

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_roles": list(self.observation_roles),
            "protected_roles": list(self.protected_roles),
            "amplitude_roles": list(self.amplitude_roles),
            "derived_roles": list(self.derived_roles),
            "role_shapes": {role: list(shape) for role, shape in self.role_shapes.items()},
            "mutable_stages": {role: list(stages) for role, stages in self.mutable_stages.items()},
        }


@dataclass(frozen=True)
class NonDecomposableObservationContract:
    """非加性 observation 变换的显式合同。

    该合同描述 side-chain compressor 等会改变 mixture、但不能写成
    ``source + background`` 的变换。它只登记边界事实，不会自动启用任何
    side-chain 路径；真正的变换接入仍需由任务 adapter 显式消费。
    """

    observation_roles: tuple[str, ...]
    transform_id: str
    mutable_stage: str
    control_roles: tuple[str, ...] = ()
    protected_roles: tuple[str, ...] = ()
    role_shapes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    channel_order: tuple[str, ...] = ()
    timebase_identity: str | None = None
    rng_stream: str | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        observation = _role_tuple(self.observation_roles, "observation_roles")
        control = _role_tuple(self.control_roles, "control_roles")
        protected = _role_tuple(self.protected_roles, "protected_roles")
        role_groups = (set(observation), set(control), set(protected))
        overlap = (
            (role_groups[0] & role_groups[1])
            | (role_groups[0] & role_groups[2])
            | (role_groups[1] & role_groups[2])
        )
        if overlap:
            raise ValueError(
                "non-decomposable observation contract has overlapping roles: "
                f"{sorted(overlap)!r}"
            )
        if not observation:
            raise ValueError("observation_roles must not be empty")
        if not isinstance(self.transform_id, str) or not self.transform_id.strip():
            raise ValueError("transform_id must be a non-empty string")
        if not isinstance(self.mutable_stage, str) or not self.mutable_stage.strip():
            raise ValueError("mutable_stage must be a non-empty string")
        if self.mutable_stage not in _OBSERVATION_MUTABLE_STAGES:
            raise ValueError(
                "mutable_stage must identify a known C-stage or post-mix stage: "
                f"{tuple(sorted(_OBSERVATION_MUTABLE_STAGES))!r}"
            )
        if self.timebase_identity is not None and (
            not isinstance(self.timebase_identity, str) or not self.timebase_identity.strip()
        ):
            raise ValueError("timebase_identity must be a non-empty string or None")
        if self.rng_stream is not None and (
            not isinstance(self.rng_stream, str) or not self.rng_stream.strip()
        ):
            raise ValueError("rng_stream must be a non-empty string or None")
        channel_order = _role_tuple(self.channel_order, "channel_order")
        role_shapes = dict(self.role_shapes)
        known_roles = set(observation) | set(control) | set(protected)
        for role, shape in role_shapes.items():
            if role not in known_roles:
                raise ValueError(f"role_shapes contains unknown role: {role!r}")
            if not isinstance(shape, tuple) or any(type(dim) is not int or dim < 0 for dim in shape):
                raise ValueError(
                    f"role_shapes[{role!r}] must be a tuple of non-negative exact ints"
                )
        provenance = dict(self.provenance)
        object.__setattr__(self, "observation_roles", observation)
        object.__setattr__(self, "control_roles", control)
        object.__setattr__(self, "protected_roles", protected)
        object.__setattr__(self, "channel_order", channel_order)
        object.__setattr__(self, "role_shapes", MappingProxyType(role_shapes))
        object.__setattr__(self, "provenance", MappingProxyType(provenance))

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_roles": list(self.observation_roles),
            "control_roles": list(self.control_roles),
            "protected_roles": list(self.protected_roles),
            "transform_id": self.transform_id,
            "mutable_stage": self.mutable_stage,
            "role_shapes": {role: list(shape) for role, shape in self.role_shapes.items()},
            "channel_order": list(self.channel_order),
            "timebase_identity": self.timebase_identity,
            "rng_stream": self.rng_stream,
            "provenance": dict(self.provenance),
            "decomposable": False,
        }


@dataclass(frozen=True)
class ObservationPostMixRequest:
    """Observation-side post-mix 请求。

    ``task_policy`` 必须是当前 ``SourceScenePostMixAdapter`` 所需的 task
    policy/dataset adapter，至少提供 ``sim_cfg``、``_simulation_view`` 和
    ``_apply_channel_degradation_result``。它们是内部 Adapter 的既有 seam，
    本 Module 不重新实现 Layer E/F。``used_sidechain_wind`` 显式携带 C-2
    sidechain 事实，避免从 task policy 反射读取不存在的 Dataset 字段。
    """

    scene_bundle: WaveformBundle | WaveformSceneBundle
    sample_rate: int
    rng: Any
    task_policy: Any
    observation_roles: tuple[str, ...]
    protected_roles: tuple[str, ...] = ()
    amplitude_roles: tuple[str, ...] = ()
    retain_stage_states: bool = True
    role_contract: PostMixRoleContract | None = None
    non_decomposable_contract: NonDecomposableObservationContract | None = None
    used_sidechain_wind: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.scene_bundle, (WaveformBundle, WaveformSceneBundle)):
            raise TypeError("scene_bundle must be WaveformBundle or WaveformSceneBundle")
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, (int, np.integer)):
            raise TypeError("sample_rate must be an integer")
        if int(self.sample_rate) <= 0:
            raise ValueError("sample_rate must be positive")
        if self.rng is None or not callable(getattr(self.rng, "random", None)):
            raise TypeError("rng must provide random()")
        for name in ("sim_cfg", "_simulation_view", "_apply_channel_degradation_result"):
            if not hasattr(self.task_policy, name):
                raise TypeError(f"task_policy must provide {name}")
        observation = _role_tuple(self.observation_roles, "observation_roles")
        protected = _role_tuple(self.protected_roles, "protected_roles")
        amplitude = _role_tuple(self.amplitude_roles, "amplitude_roles")
        contract = self.role_contract
        if contract is None:
            contract = PostMixRoleContract(
                observation_roles=observation,
                protected_roles=protected,
                amplitude_roles=amplitude,
            )
        elif not isinstance(contract, PostMixRoleContract):
            raise TypeError("role_contract must be PostMixRoleContract or None")
        elif (
            observation != contract.observation_roles
            or protected != contract.protected_roles
            or amplitude != contract.amplitude_roles
        ):
            raise ValueError("role_contract disagrees with explicit post-mix role fields")
        observation = contract.observation_roles
        protected = contract.protected_roles
        amplitude = contract.amplitude_roles
        roles = _bundle(self.scene_bundle).tracks
        for name, values in (("observation_roles", observation), ("protected_roles", protected)):
            missing = tuple(role for role in values if role not in roles)
            if missing:
                raise KeyError(f"{name} contain missing scene roles: {missing!r}")
        for role, expected_shape in contract.role_shapes.items():
            if role not in roles:
                raise KeyError(f"role_shapes contains missing scene role: {role!r}")
            actual_shape = tuple(np.asarray(roles[role]).shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"role_shapes[{role!r}] expects {expected_shape!r}, "
                    f"got {actual_shape!r}"
                )
        overlap = (set(observation) & set(protected)) | (set(amplitude) & set(protected))
        if overlap:
            raise ValueError(f"observation and protected roles overlap: {sorted(overlap)!r}")
        if not observation:
            raise ValueError("observation_roles must not be empty")
        if type(self.retain_stage_states) is not bool:
            raise TypeError("retain_stage_states must be bool")
        if type(self.used_sidechain_wind) is not bool:
            raise TypeError("used_sidechain_wind must be bool")
        object.__setattr__(self, "sample_rate", int(self.sample_rate))
        object.__setattr__(self, "observation_roles", observation)
        object.__setattr__(self, "protected_roles", protected)
        object.__setattr__(self, "amplitude_roles", amplitude)
        object.__setattr__(self, "role_contract", contract)
        non_decomposable = self.non_decomposable_contract
        if non_decomposable is not None:
            if not isinstance(non_decomposable, NonDecomposableObservationContract):
                raise TypeError(
                    "non_decomposable_contract must be "
                    "NonDecomposableObservationContract or None"
                )
            if non_decomposable.observation_roles != observation:
                raise ValueError(
                    "non_decomposable_contract disagrees with observation_roles"
                )
            if non_decomposable.protected_roles != protected:
                raise ValueError(
                    "non_decomposable_contract disagrees with protected_roles"
                )
            roles = _bundle(self.scene_bundle).tracks
            missing_controls = tuple(
                role for role in non_decomposable.control_roles if role not in roles
            )
            if missing_controls:
                raise KeyError(
                    "non_decomposable_contract control_roles contain missing scene roles: "
                    f"{missing_controls!r}"
                )
        object.__setattr__(self, "non_decomposable_contract", non_decomposable)


@dataclass(frozen=True)
class ObservationPostMixResult:
    """固定 D/E/F trace 及 observation-only 结果。"""

    initial_bundle: WaveformBundle | WaveformSceneBundle
    bundle: WaveformBundle | WaveformSceneBundle
    trace: PostMixTrace
    facts: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.initial_bundle, (WaveformBundle, WaveformSceneBundle)):
            raise TypeError("initial_bundle must be a supported scene bundle")
        if not isinstance(self.bundle, (WaveformBundle, WaveformSceneBundle)):
            raise TypeError("bundle must be a supported scene bundle")
        if not isinstance(self.trace, PostMixTrace):
            raise TypeError("trace must be PostMixTrace")
        if self.trace.stage_order != POST_MIX_STAGE_ORDER:
            raise ValueError("trace does not use the fixed D/E/F stage order")
        if not isinstance(self.facts, Mapping):
            raise TypeError("facts must be a mapping")
        object.__setattr__(self, "facts", MappingProxyType(dict(self.facts)))


class _ObservationOnlyAdapter:
    """包裹既有 task Adapter；D 耦合 target，E/F/final-guard 保护 observation 之外的 role。"""

    def __init__(
        self,
        delegate: Any,
        observation_roles: tuple[str, ...],
        protected_roles: tuple[str, ...],
        amplitude_roles: tuple[str, ...],
        mutable_stages: Mapping[str, tuple[str, ...]],
    ):
        self._delegate = delegate
        self._observation_roles = frozenset(observation_roles)
        self._protected_roles = frozenset(protected_roles)
        self._amplitude_roles = frozenset(amplitude_roles)
        self._mutable_stages = dict(mutable_stages)
        self.facts: dict[str, object] = {}
        self._after_d_waveforms: dict[str, np.ndarray] | None = None

    def apply_stage(self, stage: str, state: PostMixState) -> PostMixStageResult:
        result = self._delegate.apply_stage(stage, state)
        target_roles = {
            role for role in state.waveforms if role.startswith("target/")
        }
        if stage in {"D-normalize", "D-guard"}:
            # D 的增益必须同时作用于全部 target（包括未选作监督输出的
            # target/1 等角色）和 source decomposition，否则后续选择目标时
            # target/source 的相对电平会失配。
            changes = {
                role: state.waveforms[role]
                for role in self._protected_roles - target_roles
                if not np.array_equal(result.state.waveforms[role], state.waveforms[role])
            }
        else:
            # E/F/final-guard 只能改变 observation；即使调用方没有把备用
            # target 明列为 protected，也必须恢复全部 target 角色。
            restore_roles = self._protected_roles | target_roles
            changes = {
                role: state.waveforms[role]
                for role in restore_roles
                if not np.array_equal(result.state.waveforms[role], state.waveforms[role])
            }
        restored = result.state._replace_waveforms(changes)
        if stage == "D-guard":
            # compact trace 不保留各阶段 waveform，单独保存 D 之后的 target 快照供 facts 使用。
            self._after_d_waveforms = {
                role: restored.waveforms[role]
                for role in self._protected_roles
            }
        changed_roles = tuple(
            role for role in restored.waveforms
            if not np.array_equal(restored.waveforms[role], state.waveforms[role])
        )
        allowed = self._observation_roles
        if stage in {"D-normalize", "D-guard"}:
            allowed = allowed | self._amplitude_roles | target_roles
        unexpected = tuple(role for role in changed_roles if role not in allowed)
        if unexpected:
            raise ValueError(
                f"post-mix stage {stage!r} changed non-observation roles: {unexpected!r}"
            )
        stage_rejects = tuple(
            role
            for role in changed_roles
            if role in self._mutable_stages
            and stage not in self._mutable_stages[role]
        )
        if stage_rejects:
            raise ValueError(
                f"post-mix stage {stage!r} changed roles outside mutable_stages: "
                f"{stage_rejects!r}"
            )
        self.facts[str(stage)] = {
            "delegate_applied": bool(result.applied),
            "observation_changed": bool(changed_roles),
            "changed_roles": list(changed_roles),
            "protected_roles_restored": list(changes),
            "coupled_roles": sorted(target_roles),
        }
        return PostMixStageResult(stage, restored, restored is not state)


def _role_tuple(values: Sequence[str], field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a sequence of role names")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{field} must be a sequence of role names") from exc
    if any(not isinstance(role, str) or not role.strip() for role in result):
        raise ValueError(f"{field} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must not contain duplicates")
    return result


def _bundle(scene_bundle: WaveformBundle | WaveformSceneBundle) -> WaveformBundle:
    return scene_bundle if isinstance(scene_bundle, WaveformBundle) else scene_bundle.bundle


def infer_post_mix_role_contract(
    scene_bundle: WaveformBundle | WaveformSceneBundle,
    *,
    observation_roles: Sequence[str] = ("mixture", "background"),
    protected_prefixes: Sequence[str] = ("target/", "direct/"),
    amplitude_prefixes: Sequence[str] = ("source_full_rir/",),
) -> PostMixRoleContract:
    """从 scene bundle 的稳定 role 命名推导 observation post-mix 合同。

    不同 Dataset 的场景轨道集合不同，但 observation、target/direct 和 source
    decomposition 的 role 选择规则相同。此 helper 只做角色发现，
    不改变 waveform，也不会把缺失的 observation role 静默省略。
    """

    if not isinstance(scene_bundle, (WaveformBundle, WaveformSceneBundle)):
        raise TypeError("scene_bundle must be WaveformBundle or WaveformSceneBundle")

    def _prefix_tuple(values: Sequence[str], field: str) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{field} must be a sequence of role prefixes")
        try:
            result = tuple(values)
        except TypeError as exc:
            raise TypeError(f"{field} must be a sequence of role prefixes") from exc
        if any(not isinstance(prefix, str) or not prefix for prefix in result):
            raise ValueError(f"{field} must contain non-empty strings")
        if len(set(result)) != len(result):
            raise ValueError(f"{field} must not contain duplicates")
        return result

    observation = _role_tuple(observation_roles, "observation_roles")
    protected_prefixes = _prefix_tuple(protected_prefixes, "protected_prefixes")
    amplitude_prefixes = _prefix_tuple(amplitude_prefixes, "amplitude_prefixes")
    tracks = _bundle(scene_bundle).tracks
    missing = tuple(role for role in observation if role not in tracks)
    if missing:
        raise KeyError(f"observation_roles contain missing scene roles: {missing!r}")
    protected = tuple(
        role for role in tracks
        if role.startswith(protected_prefixes)
    )
    amplitude = tuple(
        role for role in tracks
        if role.startswith(amplitude_prefixes)
    )
    return PostMixRoleContract(
        observation_roles=observation,
        protected_roles=protected,
        amplitude_roles=amplitude,
    )


def _rebuild_scene(original: WaveformBundle | WaveformSceneBundle, bundle: WaveformBundle):
    if isinstance(original, WaveformBundle):
        return bundle
    return WaveformSceneBundle(bundle, original.contract)


def _canonical_layer_facts(delegate_facts: Mapping[str, object]) -> dict[str, object]:
    """从 Adapter 的最终 stage facts 投影出统一 D/E/F schema。"""
    normalize = delegate_facts.get("D-normalize", delegate_facts.get("normalize", {}))
    guard = delegate_facts.get("D-guard", {})
    layer_e = delegate_facts.get("E", {})
    layer_f = delegate_facts.get("F", {})
    final_guard = delegate_facts.get("final-guard", {})
    return {
        "layer_d_facts": {
            "stage_order": ["D-normalize", "D-guard"],
            "normalize": dict(normalize) if isinstance(normalize, Mapping) else normalize,
            "guard": dict(guard) if isinstance(guard, Mapping) else guard,
        },
        "layer_e": dict(layer_e) if isinstance(layer_e, Mapping) else layer_e,
        "layer_f": dict(layer_f) if isinstance(layer_f, Mapping) else layer_f,
        "final_peak_guard": (
            dict(final_guard) if isinstance(final_guard, Mapping) else final_guard
        ),
    }


def _decomposable_final(bundle: WaveformBundle) -> bool:
    """按最终 bundle 检查 source_full_rir + background 是否重建 mixture。"""
    mixture = np.asarray(bundle.require("mixture"))
    background = bundle.tracks.get("background")
    source_roles = tuple(
        role for role in bundle.tracks if role.startswith("source_full_rir/")
    )
    if background is None:
        return False
    components = (np.asarray(background),) + tuple(
        np.asarray(bundle.require(role)) for role in source_roles
    )
    if any(component.shape != mixture.shape for component in components):
        return False
    reconstructed = np.zeros_like(mixture, dtype=np.float32)
    for component in components:
        reconstructed = reconstructed + component
    return bool(np.allclose(reconstructed, mixture, rtol=1e-5, atol=1e-7))


class ObservationPostMixRenderer:
    """组合既有 post-mix pipeline 的 observation-only deep Module。"""

    def apply(self, request: ObservationPostMixRequest) -> ObservationPostMixResult:
        if not isinstance(request, ObservationPostMixRequest):
            raise TypeError("request must be ObservationPostMixRequest")
        source = _bundle(request.scene_bundle)
        delegate = SourceScenePostMixAdapter(
            request.task_policy,
            request.sample_rate,
            request.rng,
            used_sidechain_wind=request.used_sidechain_wind,
        )
        adapter = _ObservationOnlyAdapter(
            delegate,
            request.observation_roles,
            request.protected_roles,
            request.amplitude_roles,
            request.role_contract.mutable_stages,
        )
        post_mix = PostMixEngine().run_bundle(
            source,
            adapter,
            context={
                "observation_roles": list(request.observation_roles),
                "protected_roles": list(request.protected_roles),
                "derived_roles": list(request.role_contract.derived_roles),
                "non_decomposable_observation": (
                    None
                    if request.non_decomposable_contract is None
                    else request.non_decomposable_contract.to_dict()
                ),
            },
            retain_stage_states=request.retain_stage_states,
        )
        final = _rebuild_scene(request.scene_bundle, post_mix.bundle)
        initial_tracks = source.tracks
        final_tracks = _bundle(final).tracks
        if adapter._after_d_waveforms is None:
            raise RuntimeError("observation post-mix did not record the D-guard state")
        d_tracks = adapter._after_d_waveforms
        target_invariant_after_d = all(
            np.array_equal(initial_tracks[role], d_tracks[role])
            for role in request.protected_roles
        )
        target_invariant_after_observation_stages = all(
            np.array_equal(d_tracks[role], final_tracks[role])
            for role in request.protected_roles
        )
        target_direct_invariant = all(
            np.array_equal(initial_tracks[role], final_tracks[role])
            for role in request.protected_roles
        )
        decomposition_preserved = (
            False
            if request.non_decomposable_contract is not None
            else _decomposable_final(_bundle(final))
        )
        delegate_facts = dict(getattr(adapter._delegate, "facts", {}))
        canonical = _canonical_layer_facts(delegate_facts)
        facts = {
            "stage_order": list(post_mix.trace.stage_order),
            "observation_roles": list(request.observation_roles),
            "protected_roles": list(request.protected_roles),
            "derived_roles": list(request.role_contract.derived_roles),
            "role_contract": request.role_contract.to_dict(),
            "non_decomposable_observation": (
                None
                if request.non_decomposable_contract is None
                else request.non_decomposable_contract.to_dict()
            ),
            "observation_changed": any(
                not np.array_equal(initial_tracks[role], final_tracks[role])
                for role in request.observation_roles
            ),
            "target_invariant_after_d": target_invariant_after_d,
            "target_invariant_after_observation_stages": target_invariant_after_observation_stages,
            "target_direct_invariant": target_direct_invariant,
            "decomposition_preserved": decomposition_preserved,
            "decomposable_final": decomposition_preserved,
            **canonical,
            "adapter": dict(adapter.facts),
            "delegate": delegate_facts,
            "post_mix": post_mix.context.to_dict(),
        }
        return ObservationPostMixResult(
            initial_bundle=request.scene_bundle,
            bundle=final,
            trace=post_mix.trace,
            facts=facts,
        )


__all__ = [
    "infer_post_mix_role_contract",
    "NonDecomposableObservationContract",
    "ObservationPostMixRenderer",
    "ObservationPostMixRequest",
    "ObservationPostMixResult",
    "PostMixRoleContract",
]
