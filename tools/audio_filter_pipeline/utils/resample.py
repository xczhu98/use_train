"""重采样到目标采样率，优先 soxr，fallback 到 librosa/scipy。与 Urgent2026 按真实有效采样率落盘一致。"""
from __future__ import annotations

import logging
from typing import Literal, get_args

import numpy as np

logger = logging.getLogger(__name__)

_SOXR_AVAILABLE = None
# 低于阈值的滤波器过冲按局部裁剪处理；明显过冲则整段等比缩放，避免削波。
_OVERSHOOT_NORMALIZE_THRESHOLD = 1.01

# 显式算法名（不含 auto）；用于 CLI/YAML 校验与文档。
ResampleMethod = Literal[
    "auto",
    "soxr",
    "soxr_hq",
    "soxr_vhq",
    "soxr_mq",
    "librosa",
    "librosa_kaiser_best",
    "scipy",
]


def resample_method_choices() -> tuple[str, ...]:
    return get_args(ResampleMethod)


def _soxr_available() -> bool:
    global _SOXR_AVAILABLE
    if _SOXR_AVAILABLE is not None:
        return _SOXR_AVAILABLE
    try:
        import soxr  # noqa: F401
        _SOXR_AVAILABLE = True
    except ImportError:
        _SOXR_AVAILABLE = False
    return _SOXR_AVAILABLE


def _resample_soxr(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
    quality: str = "HQ",
) -> np.ndarray:
    import soxr

    return soxr.resample(audio, orig_sr, target_sr, quality=quality)


def _resample_librosa(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
    *,
    res_type: str | None = "soxr_hq",
) -> np.ndarray:
    import librosa

    kw = dict(orig_sr=orig_sr, target_sr=target_sr)
    if res_type is not None:
        kw["res_type"] = res_type
    if audio.ndim == 1:
        return librosa.resample(audio.astype(np.float64), **kw).astype(np.float32)
    return librosa.resample(audio.T, **kw).T.astype(np.float32)


def _resample_scipy(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(orig_sr, target_sr)
    if audio.ndim == 1:
        return resample_poly(
            audio.astype(np.float64), target_sr // g, orig_sr // g
        ).astype(np.float32)
    return resample_poly(
        audio.astype(np.float64), target_sr // g, orig_sr // g, axis=0
    ).astype(np.float32)


def _normalize_resampled_length(
    audio: np.ndarray, input_frames: int, orig_sr: int, target_sr: int,
) -> np.ndarray:
    """统一后端输出到 ceil 帧数；轻微过冲裁剪，明显过冲整段归一化。"""
    if audio.ndim not in (1, 2):
        return audio
    expected_frames = (input_frames * target_sr + orig_sr - 1) // orig_sr
    actual_frames = audio.shape[0]
    if actual_frames == expected_frames:
        normalized = audio
    elif actual_frames > expected_frames:
        normalized = audio[:expected_frames]
    else:
        padding_shape = (expected_frames - actual_frames, *audio.shape[1:])
        padding = np.zeros(padding_shape, dtype=audio.dtype)
        normalized = np.concatenate((audio, padding), axis=0)
    if not normalized.size:
        return normalized
    peak = float(np.max(np.abs(normalized)))
    if peak > _OVERSHOOT_NORMALIZE_THRESHOLD:
        return normalized / peak
    if peak > 1.0:
        return np.clip(normalized, -1.0, 1.0)
    return normalized


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
    *,
    method: str = "auto",
) -> np.ndarray:
    """将音频从 orig_sr 重采样到 target_sr。

    Args:
        audio:  float 波形，shape (n_samples,) 或 (n_samples, n_channels)
        orig_sr: 当前采样率
        target_sr: 目标采样率
        method: 重采样策略。
            - ``auto``（默认）: 优先 soxr(HQ)，失败则 librosa(soxr_hq)，再 scipy 多相。
            - ``soxr`` / ``soxr_hq``: soxr 高质量（默认 HQ）。
            - ``soxr_vhq``: soxr 极高质（更慢，更保细节）。
            - ``soxr_mq``: soxr 中等质量（更快）。
            - ``librosa``: librosa 默认重采样器（不传 ``res_type``，随 librosa 版本而定）。
            - ``librosa_kaiser_best``: librosa ``kaiser_best``，较慢、通常更保真。
            - ``scipy``: ``scipy.signal.resample_poly``，有理数率比时高效。

    Returns:
        重采样后的 float 数组，shape 与 audio 一致（多通道时保持通道维）。
        非整数采样率比例时，帧数统一为 ``ceil(n * target_sr / orig_sr)``。
        轻微滤波器过冲（峰值 <= 1.01）裁剪到 [-1, 1]；明显过冲按峰值整段归一化。

    Raises:
        ValueError: 采样率非法或 ``method`` 未知。
        ImportError: 显式指定某后端但未安装对应依赖时。
    """
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(
            f"resample_audio: orig_sr and target_sr must be positive, got orig_sr={orig_sr}, target_sr={target_sr}"
        )
    if orig_sr == target_sr:
        return audio

    valid = set(resample_method_choices())
    if method not in valid:
        raise ValueError(
            f"resample_audio: unknown method {method!r}, expected one of {sorted(valid)}"
        )

    def finalize(result: np.ndarray) -> np.ndarray:
        return _normalize_resampled_length(
            result, audio.shape[0], orig_sr, target_sr,
        )

    if method == "auto":
        if _soxr_available():
            try:
                return finalize(_resample_soxr(audio, orig_sr, target_sr, quality="HQ"))
            except Exception as e:
                logger.warning("soxr resample failed: %s, falling back to librosa/scipy", e)
        try:
            return finalize(_resample_librosa(audio, orig_sr, target_sr, res_type="soxr_hq"))
        except Exception as e:
            logger.warning("librosa resample failed: %s, using scipy fallback", e)
        return finalize(_resample_scipy(audio, orig_sr, target_sr))

    if method in ("soxr", "soxr_hq"):
        if not _soxr_available():
            raise ImportError("method requires soxr; pip install soxr")
        return finalize(_resample_soxr(audio, orig_sr, target_sr, quality="HQ"))
    if method == "soxr_vhq":
        if not _soxr_available():
            raise ImportError("method requires soxr; pip install soxr")
        return finalize(_resample_soxr(audio, orig_sr, target_sr, quality="VHQ"))
    if method == "soxr_mq":
        if not _soxr_available():
            raise ImportError("method requires soxr; pip install soxr")
        return finalize(_resample_soxr(audio, orig_sr, target_sr, quality="MQ"))

    if method == "librosa":
        return finalize(_resample_librosa(audio, orig_sr, target_sr, res_type=None))
    if method == "librosa_kaiser_best":
        return finalize(_resample_librosa(audio, orig_sr, target_sr, res_type="kaiser_best"))

    if method == "scipy":
        return finalize(_resample_scipy(audio, orig_sr, target_sr))

    raise ValueError(f"resample_audio: unhandled method {method!r}")


def sample_rate_to_str(sr: int) -> str:
    """将采样率转换为短字符串，用于目录命名。

    Args:
        sr: 采样率，如 8000、16000、44100 等。

    Returns:
        短字符串，如 ``"8k"``、``"16k"``、``"44.1k"``、``"22.05k"``。
    """
    sr = int(sr)
    if sr % 1000 == 0:
        return f"{sr // 1000}k"
    s = f"{sr / 1000:.2f}".rstrip("0").rstrip(".")
    return f"{s}k"
