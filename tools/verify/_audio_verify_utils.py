"""Audio utility functions shared between AEC and SE simulation verify scripts.

Not part of the public project API; only imported by sibling scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def _rms(x: np.ndarray) -> float:
    """计算 1-D 音频的 RMS（根均方）。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


def _to_numpy_f32(w: Any) -> np.ndarray:
    """将 ``torch.Tensor`` 或 ``np.ndarray`` 转为 float32 一维 numpy 数组。"""
    if hasattr(w, "detach"):
        w = w.detach().cpu().numpy()
    return np.asarray(w, dtype=np.float32).ravel()


def _save_wav(path: Path, x: np.ndarray, sr: int) -> None:
    """保存音频为 WAV（FLOAT 子格式），自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(x, dtype=np.float32), int(sr), subtype="FLOAT")


def _align_audio_1d_to_len(x: np.ndarray | None, n: int) -> np.ndarray:
    """将 1-D 音频裁切或零填充到 *n* 样点，便于与其他信号同长试听。"""
    if x is None:
        return np.zeros(n, dtype=np.float32)
    a = np.asarray(x, dtype=np.float32).ravel()
    if a.size >= n:
        return a[:n].copy()
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    out[: a.size] = a
    return out


def _to_dbfs(rms_val: float) -> float:
    """RMS 线性值 → dBFS（以 1.0 为 0 dBFS）。RMS≤0 返回 -120。"""
    if rms_val <= 0:
        return -120.0
    return 20.0 * np.log10(rms_val)


def _compute_active_rms(x: np.ndarray, threshold_db: float = -40.0) -> float:
    """基于短时能量门限的活跃段 RMS（排除静音/极低能量段）。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    frame_len = max(1, int(len(x) // 100))
    frames = np.array([np.mean(x[i : i + frame_len] ** 2) for i in range(0, len(x), frame_len)])
    if frames.size == 0:
        return _rms(x)
    peak = float(np.max(frames))
    if peak <= 0:
        return _rms(x)
    threshold_lin = peak * (10.0 ** (threshold_db / 10.0))
    active = frames[frames > threshold_lin]
    if active.size == 0:
        return _rms(x)
    return float(np.sqrt(np.mean(active)))


def _compute_snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    """从 clean 和 noise 分量计算 SNR = 10·log₁₀(clean_power / noise_power)。"""
    c = np.asarray(clean, dtype=np.float64).ravel()
    n = np.asarray(noise, dtype=np.float64).ravel()
    t = min(c.size, n.size)
    if t <= 0:
        return -120.0
    c, n = c[:t], n[:t]
    c_pow = np.mean(c ** 2)
    n_pow = np.mean(n ** 2)
    if n_pow <= 1e-20:
        return 120.0 if c_pow > 1e-20 else -120.0
    return float(10.0 * np.log10(c_pow / n_pow))


def _compute_ser_db(nearend: np.ndarray, echo: np.ndarray) -> float:
    """从 nearend 和 echo 分量计算 SER = 10·log₁₀(nearend_power / echo_power)。"""
    return _compute_snr_db(nearend, echo)


def _make_stereo_left_right(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """生成立体声 WAV（左 = left, 右 = right），长度对齐到最短。"""
    left = np.asarray(left, dtype=np.float32).ravel()
    right = np.asarray(right, dtype=np.float32).ravel()
    t = min(left.size, right.size)
    if t <= 0:
        return np.zeros((2, 1), dtype=np.float32)
    out = np.zeros((2, t), dtype=np.float32)
    out[0, :] = left[:t]
    out[1, :] = right[:t]
    return out
