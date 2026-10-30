"""动态仿真 Layer A、D、E、F 的公共数据合同。

这些类型属于仿真层的值对象，不依赖具体 Dataset。Dataset 仍可从本模块
重新导出它们以保持历史 import API；仿真组件内部应直接依赖本模块，避免
``core.data.simulation`` 与 ``dynamic_dataset_base`` 形成反向依赖。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, NamedTuple

import numpy as np

from core.data.waveform_bundle import WaveformBundle


@dataclass(frozen=True)
class PreparedSpeech:
    """Layer A 的显式结果。"""

    speech_uid: str
    sample_rate: int
    speech_1d: np.ndarray
    speech_2d: np.ndarray
    actual_len: int
    fixed_silence_canvas_active: bool
    speech_splice_export: dict[str, Any]
    repeat_export: dict[str, Any]

    def to_bundle(self) -> WaveformBundle:
        """在 Layer A→C seam 创建一次只读快照；默认热路径不自动复制。"""
        return WaveformBundle(
            {"speech": self.speech_2d},
            metadata={
                "speech_uid": self.speech_uid,
                "sample_rate": self.sample_rate,
                "fixed_silence_canvas_active": self.fixed_silence_canvas_active,
            },
        )


@dataclass(frozen=True)
class PeakGuardFacts:
    """Layer D-guard 的角色、阈值、峰值与缩放事实。"""

    roles: tuple[str, ...]
    threshold: float
    peak: float
    scale: float
    applied: bool

    def to_metadata(self) -> dict[str, Any]:
        return {
            "roles": list(self.roles),
            "threshold": self.threshold,
            "peak": self.peak,
            "scale": self.scale,
            "applied": self.applied,
        }


@dataclass(frozen=True)
class PeakGuardResult:
    """Layer D-guard 的波形与审计事实。"""

    waveforms: dict[str, np.ndarray]
    facts: PeakGuardFacts


def apply_deterministic_peak_guard(
    waveforms: Mapping[str, np.ndarray],
    *,
    roles: tuple[str, ...] | None = None,
    threshold: float = 0.99,
) -> PeakGuardResult:
    """对显式 roles 按共同峰值执行 Layer D-guard。"""
    if not isinstance(threshold, Real) or isinstance(threshold, (bool, np.bool_)):
        raise TypeError(f"D-guard threshold must be a real number, got {threshold!r}")
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(
            f"D-guard threshold must be finite and positive, got {threshold!r}"
        )
    # guard 输出固定为 float32；比较与审计也使用该 dtype 中真正可表示的阈值，
    # 避免 np.float32(0.99) 被二进制 float64 字面量 0.99 误判为越界。
    effective_threshold = float(np.float32(threshold))
    if not np.isfinite(effective_threshold) or effective_threshold <= 0.0:
        raise ValueError(
            "D-guard threshold must be representable as a finite positive "
            f"float32 value, got {threshold!r}"
        )

    selected_roles = tuple(waveforms) if roles is None else tuple(roles)
    if len(set(selected_roles)) != len(selected_roles):
        raise ValueError(f"D-guard roles must be unique, got {selected_roles!r}")
    missing = tuple(role for role in selected_roles if role not in waveforms)
    if missing:
        raise KeyError(f"D-guard roles are missing from waveforms: {missing!r}")

    arrays = {
        role: np.asarray(waveform, dtype=np.float32)
        for role, waveform in waveforms.items()
    }
    nonfinite_roles = tuple(
        role for role in selected_roles if not np.isfinite(arrays[role]).all()
    )
    if nonfinite_roles:
        raise ValueError(
            f"D-guard waveforms must be finite for roles: {nonfinite_roles!r}"
        )
    peak = max(
        (
            float(np.max(np.abs(arrays[role])))
            for role in selected_roles
            if arrays[role].size
        ),
        default=0.0,
    )
    applied = peak > effective_threshold
    scale = effective_threshold / peak if applied else 1.0
    guarded = {
        role: (
            array * np.float32(scale)
            if applied and role in selected_roles
            else array
        ).astype(np.float32, copy=False)
        for role, array in arrays.items()
    }
    return PeakGuardResult(
        waveforms=guarded,
        facts=PeakGuardFacts(
            roles=selected_roles,
            threshold=effective_threshold,
            peak=float(peak),
            scale=float(scale),
            applied=applied,
        ),
    )


@dataclass(frozen=True)
class LayerEOperationResult:
    """一个被 Layer E selector 选中后的稳定执行结果。"""

    name: str
    applied: bool
    observation_changed: bool
    reason: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "applied": self.applied,
            "observation_changed": self.observation_changed,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LayerEResult:
    """Layer E selector 与 execution 的结构化事实。"""

    waveform: np.ndarray
    enabled: bool
    num_augmentations_drawn: int
    selected: tuple[str, ...]
    operations: tuple[LayerEOperationResult, ...]
    observation_changed: bool
    skip_reason: str | None = None

    @property
    def applied(self) -> tuple[str, ...]:
        return tuple(operation.name for operation in self.operations if operation.applied)

    @property
    def skipped(self) -> tuple[str, ...]:
        return tuple(operation.name for operation in self.operations if not operation.applied)

    def operation_applied(self, name: str) -> bool:
        return any(
            operation.name == name and operation.applied
            for operation in self.operations
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "channel_augmentation_enabled": self.enabled,
            "num_augmentations_drawn": self.num_augmentations_drawn,
            "selected": list(self.selected),
            "applied": list(self.applied),
            "skipped": list(self.skipped),
            "operations": [operation.to_metadata() for operation in self.operations],
            "observation_changed": self.observation_changed,
            "skip_reason": self.skip_reason,
            "augmentation_names_chosen": list(self.selected),
            "skip_codec_redundant_with_e": self.operation_applied("codec"),
            "skip_clip_redundant_with_e": self.operation_applied("clipping"),
        }


@dataclass(frozen=True)
class LayerEFResult:
    """Layer E/F 波形与审计事实。"""

    waveform: np.ndarray
    layer_e: LayerEResult
    layer_f_applied_ops: tuple[str, ...]
    layer_f_clipping_facts: dict[str, Any] | None

    @property
    def layer_e_codec_applied(self) -> bool:
        return self.layer_e.operation_applied("codec")

    @property
    def layer_e_clip_applied(self) -> bool:
        return self.layer_e.operation_applied("clipping")


class _LayerEFOutputTuple(NamedTuple):
    waveform: np.ndarray
    selected: list[str]
    layer_e_codec_applied: bool
    layer_e_clip_applied: bool


class LayerEFOutput(_LayerEFOutputTuple):
    """保持四 positional；布尔位明确表示 Layer E 的实际 execution facts。"""

    layer_e: LayerEResult
    layer_f_applied_ops: tuple[str, ...]
    layer_f_clipping_facts: dict[str, Any] | None

    def __new__(cls, result: LayerEFResult):
        output = super().__new__(
            cls,
            result.waveform,
            list(result.layer_e.selected),
            result.layer_e_codec_applied,
            result.layer_e_clip_applied,
        )
        output.layer_e = result.layer_e
        output.layer_f_applied_ops = result.layer_f_applied_ops
        output.layer_f_clipping_facts = result.layer_f_clipping_facts
        return output


__all__ = [
    "LayerEFOutput",
    "LayerEFResult",
    "LayerEOperationResult",
    "LayerEResult",
    "PeakGuardFacts",
    "PeakGuardResult",
    "PreparedSpeech",
    "apply_deterministic_peak_guard",
]
