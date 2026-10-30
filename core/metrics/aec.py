"""AEC 专用客观指标。"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _shift_sample_mask(mask: np.ndarray, delay_samples: int) -> np.ndarray:
    """按 far→mic 延时平移样本级门控；正延时表示 mic 中回声更晚到达。"""
    delay = int(delay_samples)
    if delay == 0:
        return mask
    shifted = np.zeros_like(mask)
    if delay > 0:
        if delay < mask.size:
            shifted[delay:] = mask[:-delay]
    elif -delay < mask.size:
        shifted[:delay] = mask[-delay:]
    return shifted


def _fst_gate_sample_mask(
    far: np.ndarray,
    near: np.ndarray,
    n_samples: int,
    fs: int,
    frame_ms: float,
    hop_ms: float,
    far_active_db: float,
    near_inactive_db: float,
    far_delay_samples: int = 0,
) -> Optional[np.ndarray]:
    """帧级能量 VAD → 样本级延时补偿 → FST 门控 mask。

    far 帧能量 > far 最大帧能量 -``far_active_db`` dB 视为 active；
    near 帧能量 <= near 最大帧能量 -``near_inactive_db`` dB 视为 inactive。
    帧标签按其 hop 起点展开：第 i 帧覆盖 ``[i * hop, (i + 1) * hop)``，
    最后一帧的标签延续到 ``n_samples``，因此完整覆盖 ``[0, n_samples)``。
    far active mask 展开到样本级后按 ``far_delay_samples`` 平移到 mic 时间轴，
    再与未平移的 near inactive mask 相交。
    """
    frame_len = max(int(round(fs * frame_ms / 1000.0)), 1)
    hop = max(int(round(fs * hop_ms / 1000.0)), 1)
    if n_samples < frame_len:
        frame_len = n_samples
        hop = n_samples
    n_frames = 1 + (n_samples - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    far_e = (far[idx] ** 2).mean(axis=1)
    near_e = (near[idx] ** 2).mean(axis=1)

    far_max = float(far_e.max())
    if not np.isfinite(far_max) or far_max <= 0:
        return None
    near_max = float(near_e.max())
    far_active = far_e > far_max * 10.0 ** (-far_active_db / 10.0)
    near_inactive = near_e <= near_max * 10.0 ** (-near_inactive_db / 10.0)

    # 每个 hop 使用对应起始帧的标签；最后一帧延续至末尾，避免遗漏
    # ``frame_len - hop`` 的完整帧尾部和不足一个 hop 的音频尾部。
    sample_frames = np.minimum(np.arange(n_samples) // hop, n_frames - 1)
    far_mask = far_active[sample_frames]
    near_mask = near_inactive[sample_frames]
    return _shift_sample_mask(far_mask, far_delay_samples) & near_mask


def _estimate_far_delay_samples(
    mic: np.ndarray,
    nearend: np.ndarray,
    far: np.ndarray,
    fs: int,
) -> int:
    """用共享 GCC-PHAT 内核估计非负 far→mic 延时，低置信度时退化为 0。"""
    try:
        from tasks.aec.inference.gcc_phat_core import (
            gcc_phat_tau_with_confidence,
            gcc_quality_ok,
        )

        echo_observed = np.asarray(mic, dtype=np.float32) - np.asarray(
            nearend, dtype=np.float32
        )
        max_tau = min(1.0, max((echo_observed.size - 1) / float(fs), 0.0))
        tau, peak, mean = gcc_phat_tau_with_confidence(
            echo_observed,
            far,
            fs,
            max_tau_seconds=max_tau,
            non_negative=True,
        )
        if gcc_quality_ok(peak, mean, peak_threshold=0.05, prominence_db=6.0):
            return max(int(round(tau * fs)), 0)
    except (ImportError, ValueError, FloatingPointError):
        pass
    return 0


def compute_erle_energy_sums(
    mic: np.ndarray,
    nearend: np.ndarray,
    enh: np.ndarray,
    fs: int = 0,
    far: Optional[np.ndarray] = None,
    gate_frame_ms: float = 20.0,
    gate_hop_ms: float = 10.0,
    far_active_db: float = 40.0,
    near_inactive_db: float = 40.0,
    far_delay_samples: Optional[int] = None,
) -> Optional[tuple[float, float, int]]:
    """返回 FST 门控后的线性能量 ``(sum_in, sum_out, count)``。

    提供 ``far`` 且 ``fs > 0`` 时启用 FST 门控。``far_delay_samples=None``
    会从 ``far`` 与 ``mic-nearend`` 自动估计 far→mic 延时；传入整数（包括 0）
    则使用调用方给定延时。只累计输入、输出都有限的样本。
    """
    mic = np.asarray(mic).reshape(-1)
    nearend = np.asarray(nearend).reshape(-1)
    enh = np.asarray(enh).reshape(-1)
    lengths = [mic.size, nearend.size, enh.size]
    if far is not None:
        far = np.asarray(far).reshape(-1)
        lengths.append(far.size)
    minlen = min(lengths)
    if minlen == 0:
        return None
    mic = mic[:minlen]
    nearend = nearend[:minlen]
    enh = enh[:minlen]

    if far is not None and fs > 0:
        far = far[:minlen]
        delay = (
            _estimate_far_delay_samples(mic, nearend, far, fs)
            if far_delay_samples is None
            else int(far_delay_samples)
        )
        mask = _fst_gate_sample_mask(
            far,
            nearend,
            minlen,
            fs,
            gate_frame_ms,
            gate_hop_ms,
            far_active_db,
            near_inactive_db,
            far_delay_samples=delay,
        )
        if mask is None or not mask.any():
            return None
        mic = mic[mask]
        nearend = nearend[mask]
        enh = enh[mask]

    echo_in = np.asarray(mic - nearend, dtype=np.float64)
    echo_out = np.asarray(enh - nearend, dtype=np.float64)
    finite = np.isfinite(echo_in) & np.isfinite(echo_out)
    count = int(finite.sum())
    if count == 0:
        return None
    sum_in = float(np.square(echo_in[finite]).sum(dtype=np.float64))
    sum_out = float(np.square(echo_out[finite]).sum(dtype=np.float64))
    if not np.isfinite(sum_in) or not np.isfinite(sum_out) or sum_in <= 0.0:
        return None
    return sum_in, sum_out, count


def compute_erle(
    mic: np.ndarray,
    nearend: np.ndarray,
    enh: np.ndarray,
    fs: int = 0,
    eps: float = 1e-10,
    far: Optional[np.ndarray] = None,
    gate_frame_ms: float = 20.0,
    gate_hop_ms: float = 10.0,
    far_active_db: float = 40.0,
    near_inactive_db: float = 40.0,
    far_delay_samples: Optional[int] = None,
    return_energy_sums: bool = False,
) -> Optional[float | tuple[float, float, float, int]]:
    """计算 ERLE（dB），兼容旧接口并复用线性能量累计实现。

    提供 ``far`` 时执行 delay-aware FST 门控；未显式提供
    ``far_delay_samples`` 时自动用共享 GCC-PHAT 内核估计 far→mic 延时。
    ``return_energy_sums=True`` 时返回 ``(erle_db, sum_in, sum_out, count)``，
    供调用方在保留逐 utterance dB 宏平均的同时计算全局 energy-micro。
    """
    stats = compute_erle_energy_sums(
        mic,
        nearend,
        enh,
        fs=fs,
        far=far,
        gate_frame_ms=gate_frame_ms,
        gate_hop_ms=gate_hop_ms,
        far_active_db=far_active_db,
        near_inactive_db=near_inactive_db,
        far_delay_samples=far_delay_samples,
    )
    if stats is None:
        return None
    sum_in, sum_out, count = stats
    denominator = sum_out + float(eps) * count
    if denominator <= 0.0:
        return None
    erle = 10.0 * np.log10(sum_in / denominator)
    if not np.isfinite(erle):
        return None
    erle_db = float(erle)
    if return_energy_sums:
        return erle_db, sum_in, sum_out, count
    return erle_db
