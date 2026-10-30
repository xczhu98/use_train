"""RIR artifact 和发布前校验。"""

from __future__ import annotations

from typing import Any

import numpy as np
import soundfile as sf

from tools.shared.pickle_io import load_pkl_file


def validate_rirs(
    entries: list[tuple[str, int, int, str]],
    threshold: float,
) -> list[str]:
    """检查采样率、长度、有限性和逐 segment 峰值。"""
    abnormal: list[str] = []
    for uid, fs, n_samples, path in entries:
        try:
            if path.endswith(".pkl"):
                data = load_pkl_file(path, purpose="validate generated simulated RIR")
                rir = data["rir"] if isinstance(data, dict) else data
                actual_fs = int(data.get("fs", fs)) if isinstance(data, dict) else fs
            else:
                rir, actual_fs = sf.read(path, dtype="float32")
            rir = np.asarray(rir, dtype=np.float32)
            if actual_fs != fs:
                abnormal.append(
                    f"{uid} ({path}): sample rate {actual_fs} != declared {fs}"
                )
                continue
            if rir.ndim not in {1, 2} or rir.size == 0 or rir.shape[-1] == 0:
                abnormal.append(f"{uid} ({path}): invalid RIR shape {rir.shape}")
                continue
            if rir.shape[-1] != n_samples:
                abnormal.append(
                    f"{uid} ({path}): frame count {rir.shape[-1]} != declared {n_samples}"
                )
                continue
            if not np.isfinite(rir).all():
                abnormal.append(f"{uid} ({path}): contains non-finite samples")
                continue

            segments = rir.reshape(1, -1) if rir.ndim == 1 else rir
            segment_peaks = np.max(np.abs(segments), axis=1)
            silent_segments = np.flatnonzero(segment_peaks < threshold)
            if silent_segments.size:
                indices = ",".join(str(int(index)) for index in silent_segments[:10])
                suffix = "..." if silent_segments.size > 10 else ""
                abnormal.append(
                    f"{uid} ({path}): segment peak below {threshold:.2e} "
                    f"at indices [{indices}{suffix}]"
                )
        except Exception as exc:
            abnormal.append(f"{uid} ({path}): read error {exc}")
    return abnormal


__all__ = ["validate_rirs"]
