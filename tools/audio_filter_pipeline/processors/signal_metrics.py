"""与筛选策略无关的纯音频信号测量。"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


_SIGNAL_METRICS_BLOCK_SAMPLES = 262_144
_FLOAT64_EXPONENT_BINS = 2047
_FLOAT64_FRACTION_MASK = (1 << 52) - 1
_SUPERACCUMULATOR_LIMB_BITS = 22
_SUPERACCUMULATOR_LIMB_MASK = (1 << _SUPERACCUMULATOR_LIMB_BITS) - 1
_SUPERACCUMULATOR_LIMBS = 100
_FLOAT64_UNIT_DENOMINATOR = 1 << 1074


@dataclass(frozen=True)
class SignalMetrics:
    finite: bool
    zero_ratio: Optional[float]
    peak: Optional[float]
    clipping_ratio: Optional[float]
    dc_offset: Optional[float]
    rms_dbfs: Optional[float]
    num_samples: int


def _audio_blocks(audio: np.ndarray):
    """按固定上限返回副本块，非连续输入也不会触发全量 flatten copy。"""
    for start in range(0, audio.size, _SIGNAL_METRICS_BLOCK_SAMPLES):
        yield audio.flat[start : start + _SIGNAL_METRICS_BLOCK_SAMPLES]


def _exact_float64_block_units(values: np.ndarray) -> int:
    """以 2**-1074 为单位精确聚合 IEEE-754 float64 块。"""
    bits = values.view(np.uint64)
    exponent = ((bits >> 52) & 0x7FF).astype(np.intp)
    significand = bits & _FLOAT64_FRACTION_MASK
    significand[exponent != 0] |= np.uint64(1 << 52)
    negative = (bits >> 63) != 0

    limb_bins = []
    for shift in (0, 22, 44):
        limb = ((significand >> shift) & _SUPERACCUMULATOR_LIMB_MASK).astype(
            np.int64
        )
        limb[negative] *= -1
        limb_bins.append(
            np.bincount(
                exponent,
                weights=limb.astype(np.float64),
                minlength=_FLOAT64_EXPONENT_BINS,
            )
        )

    total = 0
    occupied = np.flatnonzero(
        (limb_bins[0] != 0.0) | (limb_bins[1] != 0.0) | (limb_bins[2] != 0.0)
    )
    for raw_exponent in occupied:
        coefficient = sum(
            int(limb_bins[index][raw_exponent]) << shift
            for index, shift in enumerate((0, 22, 44))
        )
        total += coefficient << max(int(raw_exponent) - 1, 0)
    return total


def _add_exact_units(state: np.ndarray, value: int) -> None:
    """将 block exact units 加入固定正负 limbs，并立即归一化。"""
    row = state[1 if value < 0 else 0]
    remaining = abs(value)
    carry = 0
    for index in range(_SUPERACCUMULATOR_LIMBS):
        total = int(row[index]) + (remaining & _SUPERACCUMULATOR_LIMB_MASK) + carry
        row[index] = total & _SUPERACCUMULATOR_LIMB_MASK
        carry = total >> _SUPERACCUMULATOR_LIMB_BITS
        remaining >>= _SUPERACCUMULATOR_LIMB_BITS
    if remaining or carry:
        raise OverflowError("signal metrics superaccumulator capacity exceeded")


def _exact_units_from_state(state: np.ndarray) -> int:
    """仅在 reduction 结束后将固定 limbs 重建为 exact Python integer。"""
    positive = 0
    negative = 0
    for index in range(_SUPERACCUMULATOR_LIMBS - 1, -1, -1):
        positive = (positive << _SUPERACCUMULATOR_LIMB_BITS) | int(state[0, index])
        negative = (negative << _SUPERACCUMULATOR_LIMB_BITS) | int(state[1, index])
    return positive - negative


def _reduce_block(block: np.ndarray, peak: float | None = None):
    """以固定大小 NumPy workspace 归约单块。"""
    if not bool(np.all(np.isfinite(block))):
        return None
    values = np.asarray(block, dtype=np.float64)
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("audio values must be representable as finite float64")

    if peak is None:
        block_max = np.max(values)
        block_min = np.min(values)
        block_peak = float(max(block_max, -block_min))
        zero_count = int(np.count_nonzero((values > -1e-12) & (values < 1e-12)))
        clipping_count = int(np.count_nonzero((values <= -1.0) | (values >= 1.0)))
        return block_peak, zero_count, clipping_count

    values /= peak
    exact_sum = _exact_float64_block_units(values)
    np.square(values, out=values)
    square_sum = np.sum(values, dtype=np.longdouble)
    return exact_sum, square_sum


def _compensated_add(total, correction, value):
    """固定两个 scalar 的 Neumaier 聚合，保留跨块正负抵消残差。"""
    updated = total + value
    if abs(total) >= abs(value):
        correction += (total - updated) + value
    else:
        correction += (value - updated) + total
    return updated, correction


def _round_ratio_to_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder * 2
    if doubled > denominator or (doubled == denominator and quotient & 1):
        quotient += 1
    return quotient


def _rational_to_float(numerator: int, denominator: int) -> float:
    """将 exact rational 按 IEEE round-to-nearest-even 转成 Python float。"""
    if numerator == 0:
        return 0.0
    negative = numerator < 0
    numerator = abs(numerator)
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1

    if exponent < -1022:
        significand = _round_ratio_to_even(
            numerator << 1074, denominator
        )
        result = math.ldexp(float(significand), -1074)
    else:
        shift = 52 - exponent
        if shift >= 0:
            significand = _round_ratio_to_even(numerator << shift, denominator)
        else:
            significand = _round_ratio_to_even(
                numerator, denominator << -shift
            )
        if significand == 1 << 53:
            significand >>= 1
            exponent += 1
        result = math.ldexp(float(significand), exponent - 52)
    return -result if negative else result


def compute_signal_metrics(audio: np.ndarray) -> SignalMetrics:
    """使用固定大小 workspace 计算传入 waveform 的信号事实。"""
    if not isinstance(audio, np.ndarray):
        raise TypeError("audio must be a NumPy array")
    if not (
        np.issubdtype(audio.dtype, np.integer)
        or np.issubdtype(audio.dtype, np.floating)
    ):
        raise TypeError("audio dtype must be real numeric, not bool or complex")

    num_samples = int(audio.size)
    if num_samples == 0:
        return SignalMetrics(True, 1.0, 0.0, 0.0, 0.0, None, 0)

    peak = 0.0
    zero_count = 0
    clipping_count = 0
    for block in _audio_blocks(audio):
        reduced = _reduce_block(block)
        if reduced is None:
            return SignalMetrics(False, None, None, None, None, None, num_samples)
        block_peak, block_zeros, block_clipping = reduced
        peak = max(peak, block_peak)
        zero_count += block_zeros
        clipping_count += block_clipping

    zero_ratio = float(zero_count / num_samples)
    clipping_ratio = float(clipping_count / num_samples)
    if peak == 0.0:
        dc_offset = 0.0
        rms_dbfs = None
    else:
        exact_state = np.zeros((2, _SUPERACCUMULATOR_LIMBS), dtype=np.int64)
        square_sum = np.longdouble(0.0)
        square_correction = np.longdouble(0.0)
        for block in _audio_blocks(audio):
            block_exact_sum, block_square_sum = _reduce_block(block, peak)
            _add_exact_units(exact_state, block_exact_sum)
            square_sum, square_correction = _compensated_add(
                square_sum, square_correction, block_square_sum
            )

        square_sum += square_correction
        exact_sum = _exact_units_from_state(exact_state)
        scaled_sum = _rational_to_float(exact_sum, _FLOAT64_UNIT_DENOMINATOR)
        dc_offset = float((scaled_sum / num_samples) * peak)
        mean_square_scaled = float(square_sum / np.longdouble(num_samples))
        rms_dbfs = float(
            20.0 * (math.log10(peak) + 0.5 * math.log10(mean_square_scaled))
        )

    emitted_floats = (
        zero_ratio,
        peak,
        clipping_ratio,
        dc_offset,
        *(() if rms_dbfs is None else (rms_dbfs,)),
    )
    if not all(math.isfinite(value) for value in emitted_floats):
        raise ValueError("signal metrics must be finite Python floats")

    return SignalMetrics(
        finite=True,
        zero_ratio=zero_ratio,
        peak=peak,
        clipping_ratio=clipping_ratio,
        dc_offset=dc_offset,
        rms_dbfs=rms_dbfs,
        num_samples=num_samples,
    )
