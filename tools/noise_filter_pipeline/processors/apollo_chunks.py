"""Apollo 分片纯函数（从 apollo_superres.py 拆出）。

仅包含无状态的 chunk 切分与窗函数工具，由 ApolloSuperResProcessor
向后兼容 re-export（subprocess_daemon 模式下分片在 daemon 子进程内
按 --chunk_sec 完成）。
"""
from __future__ import annotations

import numpy as np


def _prepare_chunks(audio: np.ndarray, sr: int, chunk_sec: float = 10.0, overlap_sec: float = 2.0):
    """把音频切分为带重叠的 chunks，用于 Apollo 推理。

    Args:
        chunk_sec: 每个 chunk 的时长（秒）。
        overlap_sec: 相邻 chunk 之间的重叠时长（秒）。
    """
    if sr <= 0:
        raise ValueError("sr 必须大于 0")
    if chunk_sec <= 0:
        raise ValueError("chunk_sec 必须大于 0")
    if overlap_sec < 0 or overlap_sec >= chunk_sec:
        raise ValueError("overlap_sec 必须满足 0 <= overlap_sec < chunk_sec")

    chunk_samples = int(chunk_sec * sr)
    overlap_samples = int(overlap_sec * sr)
    if chunk_samples <= 0:
        raise ValueError("chunk_sec 对应的采样点数必须大于 0")
    step_samples = chunk_samples - overlap_samples
    fade = overlap_samples
    border = overlap_samples  # 首尾的 reflect pad 等于重叠长度

    original_len = audio.shape[0]
    if original_len > 2 * border and border > 0:
        audio = np.pad(audio, (border, border), mode="reflect")

    chunks = []
    positions = []
    i = 0
    while i < audio.shape[0]:
        part = audio[i : i + chunk_samples]
        length = part.shape[0]
        if length < chunk_samples:
            if length > chunk_samples // 2 + 1:
                part = np.pad(part, (0, chunk_samples - length), mode="reflect")
            else:
                part = np.pad(part, (0, chunk_samples - length), mode="constant")
        chunks.append(part)
        positions.append((i, length))
        i += step_samples

    return chunks, positions, audio.shape[0], original_len, fade, border



def _get_window(chunk_size: int, fade_size: int) -> np.ndarray:
    window = np.ones(chunk_size)
    if fade_size > 0:
        fade_in = np.linspace(0, 1, fade_size)
        fade_out = np.linspace(1, 0, fade_size)
        window[:fade_size] *= fade_in
        window[-fade_size:] *= fade_out
    # fade_size == 0（overlap_sec == 0，_prepare_chunks 允许的合法值）时
    # window[-0:] == window[0:] 是整个数组，乘以空数组会 ValueError
    return window
