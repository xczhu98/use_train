"""
音频分片核心算法（纯函数，无 IO）。

被 ChunkingProcessor 与 StageBUnifiedProcessor 共享，避免分片逻辑复制。
所有函数接收 numpy 音频数组，返回 chunk 元信息列表（含音频数据），
调用方自行决定落盘或内存复用。

注意（量纲）：本模块的 ``energy`` 为 **MSE**（mean(x²)），
``hive_energy_threshold``（默认 0.0005）同样是 MSE 量纲；
而 silence_processor 的 ``silent_energy_threshold``（默认 0.005）是
**RMS**（sqrt(mean(x²))）量纲。两者都叫 "energy" 但不可直接比较，
换算关系：MSE = RMS²（0.005 RMS ≈ 2.5e-5 MSE）。
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from tools.shared.audio_io import split_audio


def _mono_audio(audio: np.ndarray) -> np.ndarray:
    """如有多个通道，取平均转为单通道。"""
    if len(audio.shape) > 1:
        return audio.mean(axis=1)
    return audio


def cut_adaptive(
    audio: np.ndarray,
    sr: int,
    chunk_duration: float,
    min_chunk_duration: float,
) -> List[Dict[str, Any]]:
    """adaptive 模式：无重叠固定时长分片。

    Returns:
        chunk 元信息列表，字段包含 audio、sr、start_sample、end_sample、
        sample_rate、duration、energy、is_original、chunk_idx。
    """
    audio = _mono_audio(audio)
    chunks = split_audio(audio, sr, chunk_duration=chunk_duration, min_chunk_duration=min_chunk_duration)
    if not chunks:
        return []

    results = []
    for idx, (chunk_data, start_time, end_time) in enumerate(chunks, start=1):
        results.append({
            "audio": chunk_data,
            "sr": sr,
            "start_sample": int(start_time * sr),
            "end_sample": int(end_time * sr),
            "sample_rate": sr,
            "duration": end_time - start_time,
            "energy": float(np.mean(chunk_data.astype(np.float64) ** 2)),
            "is_original": False,
            "chunk_idx": idx,
        })
    return results


def cut_hive(
    audio: np.ndarray,
    sr: int,
    window_sec: float,
    step_sec: float,
    energy_threshold: float,
    min_tail_sec: float,
) -> List[Dict[str, Any]]:
    """hive 模式：固定窗口 / 步长，能量过滤，丢弃末尾残余。

    energy_threshold 为 MSE 量纲（mean(x²)，见模块 docstring）。
    """
    audio = _mono_audio(audio)
    total_samples = len(audio)
    duration = total_samples / sr

    if duration <= window_sec:
        energy = float(np.mean(audio.astype(np.float64) ** 2))
        if energy < energy_threshold:
            return []
        return [{
            "audio": audio,
            "sr": sr,
            "start_sample": 0,
            "end_sample": total_samples,
            "sample_rate": sr,
            "duration": duration,
            "energy": energy,
            "is_original": True,
            "chunk_idx": 1,
        }]

    window_size = int(window_sec * sr)
    step_size = int(step_sec * sr)
    min_tail_samples = int(min_tail_sec * sr)

    results = []
    chunk_index = 1
    for start in range(0, total_samples, step_size):
        end = min(start + window_size, total_samples)
        segment = audio[start:end]

        # 末尾残余不足 min_tail_sec 则丢弃
        if end == total_samples and (end - start) < min_tail_samples:
            break

        energy = float(np.mean(segment.astype(np.float64) ** 2))
        if energy < energy_threshold:
            continue

        results.append({
            "audio": segment,
            "sr": sr,
            "start_sample": start,
            "end_sample": end,
            "sample_rate": sr,
            "duration": len(segment) / sr,
            "energy": energy,
            "is_original": False,
            "chunk_idx": chunk_index,
        })
        chunk_index += 1

        if end == total_samples:
            break

    return results


def cut_none(
    audio: np.ndarray,
    sr: int,
) -> List[Dict[str, Any]]:
    """none 模式：不分片，原音频透传。"""
    audio = _mono_audio(audio)
    return [{
        "audio": audio,
        "sr": sr,
        "start_sample": 0,
        "end_sample": len(audio),
        "sample_rate": sr,
        "duration": len(audio) / sr,
        "energy": float(np.mean(audio.astype(np.float64) ** 2)),
        "is_original": True,
        "chunk_idx": 1,
    }]
