"""纯数学能量比缩放 Module；任务语义由调用侧 Adapter 决定。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


EnergyPolicy = Literal["active_mean_power", "full_mean_power", "full_sum_energy"]


@dataclass(frozen=True)
class EnergyRatioResult:
    """一次比例标定的纯数学结果。"""

    numerator_energy: float
    denominator_energy: float
    gain: float
    realized_db: float
    policy: EnergyPolicy


class EnergyRatioScaler:
    """集中实现能量测量和历史兼容的比例增益公式。"""

    @staticmethod
    def measure(
        samples: np.ndarray,
        *,
        policy: EnergyPolicy,
        active_mask: np.ndarray | None = None,
        empty_value: float = 1e-10,
    ) -> float:
        """以 float64 累加测量 mean power 或 sum energy。"""
        values = np.asarray(samples)
        if policy == "active_mean_power":
            if active_mask is None:
                raise ValueError("active_mean_power requires active_mask")
            mask = np.asarray(active_mask, dtype=bool).ravel()
            if values.ndim == 0 or values.shape[-1] != mask.size:
                raise ValueError("active_mask length must match the sample time axis")
            values = values[..., mask]
        elif policy not in ("full_mean_power", "full_sum_energy"):
            raise ValueError(f"unsupported energy policy: {policy!r}")

        active = values.astype(np.float64, copy=False)
        if active.size == 0:
            return float(empty_value)
        with np.errstate(over="ignore", invalid="ignore"):
            squared = np.square(active)
            if policy == "full_sum_energy":
                result = float(np.sum(squared, dtype=np.float64))
            else:
                result = float(np.mean(squared, dtype=np.float64))
        if not np.isfinite(result):
            raise ValueError(f"energy must be finite, got {result!r}")
        return result

    @staticmethod
    def power_ratio(
        numerator_power: float,
        denominator_power: float,
        target_db: float,
        *,
        policy: EnergyPolicy,
        denominator_floor: float = 1e-10,
        gain_cap: float | None = None,
        preserve_scalar_dtype: bool = False,
    ) -> EnergyRatioResult:
        """按 ``10**(-dB/20) * sqrt(num) / sqrt(max(den,floor))`` 标定。

        ``preserve_scalar_dtype`` 仅供旧 Adapter 使用：校验仍转 Python float，
        但 sqrt 前保留传入 numpy scalar dtype，从而不改变既有 waveform。
        """
        numerator = _finite_nonnegative(numerator_power, "numerator_power")
        denominator = _finite_nonnegative(denominator_power, "denominator_power")
        ratio_db = _finite(target_db, "target_db")
        floor = _finite_nonnegative(denominator_floor, "denominator_floor")
        numerator_operand = numerator_power if preserve_scalar_dtype else numerator
        denominator_operand = (
            max(denominator_power, denominator_floor)
            if preserve_scalar_dtype
            else max(denominator, floor)
        )
        denominator = max(denominator, floor)
        try:
            gain = (
                10 ** (-ratio_db / 20.0)
                * np.sqrt(numerator_operand)
                / np.sqrt(denominator_operand)
            )
        except OverflowError as exc:
            raise ValueError("gain must be finite, got overflow") from exc
        gain = float(gain)
        if not np.isfinite(gain):
            raise ValueError(f"gain must be finite, got {gain!r}")
        if gain_cap is not None:
            cap = _finite_nonnegative(gain_cap, "gain_cap")
            gain = min(gain, cap)
        realized = _ratio_db(numerator, denominator * gain**2)
        return EnergyRatioResult(numerator, denominator, gain, realized, policy)

    @staticmethod
    def energy_db_ratio(numerator_energy: float, denominator_energy: float) -> float:
        """返回两个已测有限非负能量的 dB 比。"""
        numerator = _finite_nonnegative(numerator_energy, "numerator_energy")
        denominator = _finite_nonnegative(denominator_energy, "denominator_energy")
        return _ratio_db(numerator, denominator)

    @staticmethod
    def target_rms_gain(mean_power: float, target_db: float) -> float:
        """返回把 full RMS 缩放到目标 dBFS 的增益。"""
        power = _finite_nonnegative(mean_power, "mean_power")
        target = _finite(target_db, "target_db")
        if power <= 0.0:
            return 0.0
        gain = (10.0 ** (target / 20.0)) / np.sqrt(power)
        if not np.isfinite(gain):
            raise ValueError(f"gain must be finite, got {gain!r}")
        return float(gain)

    @staticmethod
    def sum_energy_ratio(
        numerator_energy: float,
        denominator_energy: float,
        target_db: float,
        *,
        additive_floor: float = 1e-10,
    ) -> EnergyRatioResult:
        """保留 TSE 公式：``sqrt(num / (den * 10**(dB/10) + floor))``。"""
        numerator = _finite_nonnegative(numerator_energy, "numerator_energy")
        denominator = _finite_nonnegative(denominator_energy, "denominator_energy")
        ratio_db = _finite(target_db, "target_db")
        floor = _finite_nonnegative(additive_floor, "additive_floor")
        try:
            scaled_denominator = denominator * (10.0 ** (ratio_db / 10.0)) + floor
            gain = float(np.sqrt(numerator / scaled_denominator))
        except OverflowError as exc:
            raise ValueError("gain must be finite, got overflow") from exc
        if not np.isfinite(gain):
            raise ValueError(f"gain must be finite, got {gain!r}")
        realized = _ratio_db(numerator, denominator * gain**2)
        return EnergyRatioResult(
            numerator, denominator, gain, realized, "full_sum_energy"
        )


def _ratio_db(numerator: float, denominator: float) -> float:
    return 10.0 * float(
        np.log10(max(float(numerator), 1e-20) / max(float(denominator), 1e-20))
    )


def _finite(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return result
