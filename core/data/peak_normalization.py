"""混合后随机峰值归一化的结构化纯函数 Module。"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from core.data.simulation_utils import rng_uniform


@dataclass(frozen=True)
class PeakNormalizationResult:
    """Layer D-normalize 峰值归一化波形与显式缩放事实。"""

    noisy: np.ndarray
    clean: np.ndarray
    noise: np.ndarray
    mode: str
    target_peak: float | None
    noisy_scale: float
    clean_scale: float
    noise_scale: float
    preserves_decomposition: bool


def _max_abs(waveform: np.ndarray) -> float:
    return float(np.max(np.abs(waveform))) if waveform.size else 0.0


class _FrozenJsonDict(dict[str, object]):
    """可供 ``json.dumps`` 编码、但拒绝原地修改的 JSON object。"""

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("scale facts are immutable")

    __delitem__ = _immutable
    __ior__ = _immutable
    __setitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


@dataclass(frozen=True)
class JointRolePeakNormalizationResult:
    """任意角色 joint 峰值归一化的冻结 role mapping 与缩放事实。"""

    waveforms: Mapping[str, np.ndarray]
    scale_facts: Mapping[str, object]


def apply_joint_role_peak_normalization(
    waveforms: Mapping[str, np.ndarray],
    *,
    anchor_roles: Sequence[str],
    mode: str,
    scale_range: Sequence[float],
    rng: np.random.Generator,
) -> JointRolePeakNormalizationResult:
    """对任意 waveform roles 执行 joint D-normalize，anchor roles 决定唯一 scale。

    ``off`` 不消费 RNG。``joint`` 只抽一次 ``target_peak``；followers 始终
    使用由 anchors 共同峰值确定的同一 scale，且不影响该峰值。返回的 mapping
    不可写，缩放事实只包含 JSON-safe Python 标量和 tuple。
    """
    normalized = {
        str(role): np.asarray(waveform, dtype=np.float32)
        for role, waveform in waveforms.items()
    }
    if not normalized:
        raise ValueError("D-normalize joint role requires at least one waveform")
    anchors = tuple(str(role) for role in anchor_roles)
    if not anchors:
        raise ValueError("D-normalize joint role requires at least one anchor role")
    if len(set(anchors)) != len(anchors):
        raise ValueError("D-normalize joint role anchor roles must be unique")
    missing_anchors = tuple(role for role in anchors if role not in normalized)
    if missing_anchors:
        raise ValueError(
            "D-normalize joint role anchor roles are missing: "
            f"{missing_anchors!r}"
        )
    if mode not in {"off", "joint"}:
        raise ValueError(
            "D-normalize joint role mode must be one of "
            f"['off', 'joint'], got {mode!r}"
        )

    anchor_maxima = tuple(_max_abs(normalized[role]) for role in anchors)
    if mode == "off":
        scale = 1.0
        target_peak: float | None = None
    else:
        lo, hi = scale_range
        target_peak = float(rng_uniform(rng, float(lo), float(hi)))
        shared_max = float(max(*anchor_maxima, 1e-12))
        scale = (
            1.0
            if all(value == 0.0 for value in anchor_maxima)
            else target_peak / shared_max
        )

    scaled = {
        role: (waveform * scale).astype(np.float32, copy=False)
        for role, waveform in normalized.items()
    }
    scale_facts = _FrozenJsonDict(
        {
            "mode": mode,
            "target_peak": target_peak,
            "scale": float(scale),
            "anchor_roles": anchors,
            "anchor_maxima": anchor_maxima,
            "shared_max": float(max(*anchor_maxima, 1e-12)),
        }
    )
    return JointRolePeakNormalizationResult(
        waveforms=MappingProxyType(scaled),
        scale_facts=scale_facts,
    )


def apply_peak_normalization(
    noisy: np.ndarray,
    clean: np.ndarray,
    noise: np.ndarray,
    *,
    mode: str,
    scale_range: Sequence[float],
    rng: np.random.Generator,
) -> PeakNormalizationResult:
    """应用 Layer D-normalize，并返回波形及调用方所需的缩放事实。

    ``off`` 不消费 RNG；``joint`` 消费一次 ``target_peak`` uniform draw。
    公共 Layer D 只保持物理分解的 off/joint；PASE/StuPASE 的独立 clean/noisy
    归一化由 ``tasks.se.data.pase_preprocess`` 专用路径提供。
    """
    noisy_32 = np.asarray(noisy, dtype=np.float32)
    clean_32 = np.asarray(clean, dtype=np.float32)
    noise_32 = np.asarray(noise, dtype=np.float32)

    if mode == "off":
        return PeakNormalizationResult(
            noisy=noisy_32,
            clean=clean_32,
            noise=noise_32,
            mode=mode,
            target_peak=None,
            noisy_scale=1.0,
            clean_scale=1.0,
            noise_scale=1.0,
            preserves_decomposition=True,
        )
    if mode not in {"joint"}:
        raise ValueError(
            "D-normalize peak normalization mode must be one of "
            f"['off', 'joint'], got {mode!r}"
        )

    lo, hi = scale_range
    target_peak = float(rng_uniform(rng, float(lo), float(hi)))

    if mode == "joint":
        noisy_max = _max_abs(noisy_32)
        clean_max = _max_abs(clean_32)
        noise_max = _max_abs(noise_32)
        shared_max = float(max(noisy_max, clean_max, noise_max, 1e-12))
        all_silent = noisy_max == clean_max == noise_max == 0.0
        shared_scale = 1.0 if all_silent else target_peak / shared_max
        return PeakNormalizationResult(
            noisy=(noisy_32 * shared_scale).astype(np.float32, copy=False),
            clean=(clean_32 * shared_scale).astype(np.float32, copy=False),
            noise=(noise_32 * shared_scale).astype(np.float32, copy=False),
            mode=mode,
            target_peak=target_peak,
            noisy_scale=shared_scale,
            clean_scale=shared_scale,
            noise_scale=shared_scale,
            preserves_decomposition=True,
        )


__all__ = [
    "JointRolePeakNormalizationResult",
    "PeakNormalizationResult",
    "apply_joint_role_peak_normalization",
    "apply_peak_normalization",
]
