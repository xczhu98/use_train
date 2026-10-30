"""PASE/StuPASE 专属幅度预处理。

公共动态仿真 Layer D 只允许 ``off`` / ``joint``。PASE 系列的
``separate`` / ``single`` 是模型训练 recipe 的数据预处理，不参与
SE/AEC/TSE/SS 的共享 SimulationConfigs 合同。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from core.data.peak_normalization import PeakNormalizationResult
from core.data.simulation_utils import rng_uniform


def apply_pase_peak_preprocess(
    noisy: np.ndarray,
    clean: np.ndarray,
    noise: np.ndarray,
    *,
    mode: str,
    scale_range: Sequence[float],
    rng: np.random.Generator,
) -> PeakNormalizationResult:
    """执行 PASE/StuPASE 的独立 noisy/clean 峰值预处理。"""
    if mode not in {"separate", "single"}:
        raise ValueError(
            "PASE peak preprocessing mode must be 'separate' or 'single', "
            f"got {mode!r}"
        )
    noisy_32 = np.asarray(noisy, dtype=np.float32)
    clean_32 = np.asarray(clean, dtype=np.float32)
    noise_32 = np.asarray(noise, dtype=np.float32)
    try:
        scale_lo, scale_hi = (float(value) for value in scale_range)
    except (TypeError, ValueError) as exc:
        # 单元素/非法 range 不得在采样期直接 ``scale_range[1]`` IndexError；
        # 区间本身（0 < lo <= hi <= 1）由 Dataset 构造边界统一校验。
        raise ValueError(
            "PASE peak preprocessing scale_range must contain two numbers, "
            f"got {scale_range!r}"
        ) from exc
    target_peak = float(rng_uniform(rng, scale_lo, scale_hi))
    noisy_max = float(np.max(np.abs(noisy_32))) if noisy_32.size else 0.0
    clean_max = float(np.max(np.abs(clean_32))) if clean_32.size else 0.0
    noisy_scale = 1.0 if noisy_max == 0.0 else target_peak / (noisy_max + 1e-9)
    clean_scale = 1.0 if clean_max == 0.0 else target_peak / (clean_max + 1e-9)
    if mode == "separate":
        return PeakNormalizationResult(
            noisy=(noisy_32 * noisy_scale).astype(np.float32, copy=False),
            clean=(clean_32 * clean_scale).astype(np.float32, copy=False),
            noise=noise_32,
            mode=mode,
            target_peak=target_peak,
            noisy_scale=noisy_scale,
            clean_scale=clean_scale,
            noise_scale=1.0,
            preserves_decomposition=False,
        )
    return PeakNormalizationResult(
        noisy=noisy_32,
        clean=(clean_32 * clean_scale).astype(np.float32, copy=False),
        noise=noise_32,
        mode=mode,
        target_peak=target_peak,
        noisy_scale=1.0,
        clean_scale=clean_scale,
        noise_scale=1.0,
        preserves_decomposition=False,
    )


__all__ = ["apply_pase_peak_preprocess"]
