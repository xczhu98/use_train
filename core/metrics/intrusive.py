"""有参考信号的侵入式客观指标。"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.metrics.utils import _resample_np


def compute_sisdr(ref: np.ndarray, enh: np.ndarray, fs: int = 0) -> Optional[float]:
    """Scale-Invariant Signal-to-Distortion Ratio (dB)。

    仅接受非空、等长、一维且有限的波形；不满足契约时按可选指标失败返回
    ``None``，避免 NumPy 广播或空数组均值产生伪造的有限分数。
    """
    ref = np.asarray(ref)
    enh = np.asarray(enh)
    if (
        ref.ndim != 1
        or enh.ndim != 1
        or ref.size == 0
        or ref.shape != enh.shape
        or not np.isfinite(ref).all()
        or not np.isfinite(enh).all()
    ):
        return None
    ref = ref - np.mean(ref)
    enh = enh - np.mean(enh)
    dot = np.dot(ref, enh)
    s_ref_energy = np.dot(ref, ref) + 1e-8
    s_target = dot * ref / s_ref_energy
    e_noise = enh - s_target
    sisdr = 10 * np.log10(
        np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-8) + 1e-8
    )
    return float(sisdr) if np.isfinite(sisdr) else None


def compute_sisdr_np(ref: np.ndarray, enh: np.ndarray, fs: int = 0) -> Optional[float]:
    """compute_sisdr 的别名。"""
    return compute_sisdr(ref, enh, fs)


def compute_pesq(ref: np.ndarray, enh: np.ndarray, fs: int) -> Optional[float]:
    """PESQ (ITU-T P.862)。需要：pip install pesq。"""
    try:
        from pesq import PesqError, pesq
    except ImportError:
        return None
    try:
        if fs < 8000:
            return None
        if fs != 8000 and fs != 16000:
            target = 8000 if fs < 12000 else 16000
            ref = _resample_np(ref, fs, target)
            enh = _resample_np(enh, fs, target)
            fs = target
        mode = "nb" if fs == 8000 else "wb"
        score = pesq(fs, ref, enh, mode=mode, on_error=PesqError.RETURN_VALUES)
        error_values = {
            float(getattr(PesqError, name))
            for name in (
                "UNKNOWN",
                "INVALID_SAMPLE_RATE",
                "OUT_OF_MEMORY_REF",
                "OUT_OF_MEMORY_DEG",
                "OUT_OF_MEMORY_TMP",
                "BUFFER_TOO_SHORT",
                "NO_UTTERANCES_DETECTED",
            )
            if hasattr(PesqError, name)
        }
        score = float(score)
        if not np.isfinite(score) or score in error_values:
            return None
        return score
    except Exception:
        return None


def compute_stoi(
    ref: np.ndarray, enh: np.ndarray, fs: int, extended: bool = False
) -> Optional[float]:
    """STOI 或 ESTOI。需要：pip install pystoi。"""
    try:
        from pystoi import stoi
    except ImportError:
        return None
    try:
        if fs > 16000:
            ref = _resample_np(ref, fs, 16000)
            enh = _resample_np(enh, fs, 16000)
            fs = 16000
        score = float(stoi(ref, enh, fs_sig=fs, extended=extended))
        return score if np.isfinite(score) else None
    except Exception:
        return None


def compute_estoi(ref: np.ndarray, enh: np.ndarray, fs: int) -> Optional[float]:
    """扩展 STOI (ESTOI)。"""
    return compute_stoi(ref, enh, fs, extended=True)


_INTRUSIVE_FNS = {
    "SI-SDR": (compute_sisdr, True),
    "PESQ":   (compute_pesq, True),
    "STOI":   (compute_stoi, True),
    "ESTOI":  (compute_estoi, True),
}


def _compute_intrusive_one(args):
    """并行侵入式指标计算的工作函数。"""
    ref_np, enh_np, fs, metric_name = args
    fn, _ = _INTRUSIVE_FNS[metric_name]
    try:
        v = fn(ref_np, enh_np, fs)
        if v is not None and np.isfinite(v):
            return metric_name, v, True
    except Exception:
        pass
    return metric_name, 0.0, False
