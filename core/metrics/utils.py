"""音频指标计算公用工具。"""

from __future__ import annotations

import numpy as np


def _resample_np(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """对 numpy 音频数组重采样（优先 soxr，回退 librosa）。"""
    if orig_sr == target_sr:
        return audio
    try:
        import soxr
        return soxr.resample(audio, orig_sr, target_sr)
    except ImportError:
        pass
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        raise ImportError(
            "Resampling requires soxr or librosa: pip install soxr or pip install librosa"
        )
