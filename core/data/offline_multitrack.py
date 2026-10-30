"""多轨离线 Dataset 的共享生命周期与音频 I/O primitives。

本模块只集中所有 task 都需要的 filtered index、sampler metadata cache、局部 RNG、
strict/fallback/placeholder 生命周期及波形规范化。manifest 解析、轨道含义、窗口策略和
外部 tuple/dict 合同仍由 SE/AEC/TSE Adapter 自己负责。
"""

from __future__ import annotations

import logging
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import soundfile as sf
import torch

_log = logging.getLogger(__name__)

_IO_POOL: ThreadPoolExecutor | None = None
_IO_POOL_PID: int | None = None
_IO_POOL_LOCK = threading.Lock()


@dataclass(frozen=True)
class OfflineLifecycleOptions:
    """共享离线生命周期配置；task 特有输出不进入该 Interface。"""

    strict_load: bool = True
    placeholder_num_samples: int = 16000
    seed: int = 0


class BaseOfflineMultiTrackDataset(torch.utils.data.Dataset):
    """多轨离线 Dataset 的共享生命周期基类。

    Adapter 只需提供输出采样率、source length、真实样本加载和 placeholder hooks。
    ``real_index`` 始终指 manifest 原始记录；公开 index 由 ``filtered_indices`` 映射。
    """

    def __init__(
        self,
        *,
        full_length: int,
        filtered_indices: list[int],
        lifecycle: OfflineLifecycleOptions,
    ) -> None:
        super().__init__()
        self._full_len = int(full_length)
        self._filtered_indices = list(filtered_indices)
        if self._full_len < 0:
            raise ValueError(f"full_length must be non-negative, got {full_length}")
        if any(i < 0 or i >= self._full_len for i in self._filtered_indices):
            raise ValueError("filtered_indices contains an out-of-range real index")
        self.strict_load = bool(lifecycle.strict_load)
        self.placeholder_num_samples = max(1, int(lifecycle.placeholder_num_samples))
        self._configured_seed = int(lifecycle.seed)
        # 主进程 set_epoch 后，persistent worker 持有的是 Dataset 副本；共享 tensor
        # 让 fork/spawn worker 都能观察最新 epoch/base seed，而不改变 RNG 混合公式。
        self._worker_rng_state = torch.zeros(2, dtype=torch.int64).share_memory_()
        self._cached_source_lengths: Optional[list[int]] = None
        self._cached_srs: Optional[list[int]] = None

    @property
    def _epoch(self) -> int:
        return int(self._worker_rng_state[0])

    @_epoch.setter
    def _epoch(self, value: int) -> None:
        self._worker_rng_state[0] = int(value)

    @property
    def _base_seed(self) -> int:
        return int(self._worker_rng_state[1])

    @_base_seed.setter
    def _base_seed(self, value: int) -> None:
        self._worker_rng_state[1] = int(value)

    def __len__(self) -> int:
        return len(self._filtered_indices)

    def set_epoch(self, epoch: int, base_seed: Optional[int] = None) -> None:
        """幂等设置共享 epoch/base seed，不累计修改构造时 seed。"""
        if base_seed is not None:
            self._base_seed = int(base_seed)
        self._epoch = int(epoch)

    def sample_rng(self, real_index: int, *, stream: int = 0) -> random.Random:
        """从 immutable seed、base seed、epoch、样本与 stream 派生局部 RNG。"""
        seed = (
            self._configured_seed * 1_000_003
            + self._base_seed * 100_003
            + self._epoch * 10_007
            + int(real_index) * 31
            + int(stream) * 1_009
        ) % (2**62)
        return random.Random(seed)

    def get_srs(self) -> list[int]:
        """返回 filtered 样本的输出 SR bucket，并缓存 Adapter 查询结果。"""
        if self._cached_srs is None:
            self._cached_srs = [
                int(self._output_sr(real_index))
                for real_index in self._filtered_indices
            ]
        return list(self._cached_srs)

    def get_source_length(self) -> list[int]:
        """返回 filtered 样本的输出域长度，并缓存 Adapter 查询结果。"""
        if self._cached_source_lengths is None:
            self._cached_source_lengths = [
                int(self._source_length(real_index))
                for real_index in self._filtered_indices
            ]
        return list(self._cached_source_lengths)

    def __getitem__(self, index: int):
        if not isinstance(index, int):
            raise TypeError(f"index must be int, got {type(index)}")
        if index < 0 or index >= len(self._filtered_indices):
            raise IndexError(f"index out of range: {index}, len={len(self)}")
        real_index = self._filtered_indices[index]
        try:
            return self._load_sample(real_index)
        except Exception as exc:
            if self.strict_load:
                raise RuntimeError(
                    f"strict_load=True: offline load failed index={index} "
                    f"{self._sample_description(real_index)}: {exc}"
                ) from exc
            _log.warning(
                "Failed to load offline sample index=%s real_index=%s %s: %s (%s)",
                index,
                real_index,
                self._sample_description(real_index),
                type(exc).__name__,
                exc,
            )
            return self._fallback_or_placeholder(index, real_index, exc)

    def _fallback_or_placeholder(self, index: int, real_index: int, primary_error):
        failed_sr = int(self._output_sr(real_index))
        public_candidates = list(range(index + 1, len(self))) + list(range(index))
        candidates = [
            self._filtered_indices[candidate]
            for candidate in public_candidates
            if int(self._output_sr(self._filtered_indices[candidate])) == failed_sr
        ]
        if not candidates:
            _log.warning(
                "No fallback sample in output SR bucket %s; returning placeholder",
                failed_sr,
            )
            return self._placeholder_sample(real_index)
        fallback_real = self.sample_rng(real_index, stream=2).choice(candidates)
        try:
            return self._load_sample(fallback_real)
        except Exception as fallback_error:
            _log.warning(
                "Both primary and fallback failed; returning placeholder. "
                "primary=%s fallback=%s",
                primary_error,
                fallback_error,
            )
            return self._placeholder_sample(real_index)

    # Adapter hooks。manifest、轨含义与输出合同不进入共享 Module。
    def _output_sr(self, real_index: int) -> int:
        raise NotImplementedError

    def _source_length(self, real_index: int) -> int:
        raise NotImplementedError

    def _load_sample(self, real_index: int):
        raise NotImplementedError

    def _placeholder_sample(self, real_index: int):
        raise NotImplementedError

    def _sample_description(self, real_index: int) -> str:
        return f"real_index={real_index}"


def get_io_pool() -> ThreadPoolExecutor:
    """惰性复用进程级 I/O 线程池；fork 后按 pid 重建。"""
    global _IO_POOL, _IO_POOL_PID
    pid = os.getpid()
    if _IO_POOL is None or _IO_POOL_PID != pid:
        with _IO_POOL_LOCK:
            if _IO_POOL is None or _IO_POOL_PID != pid:
                _IO_POOL = ThreadPoolExecutor(max_workers=2)
                _IO_POOL_PID = pid
    return _IO_POOL


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """整段读取音频，返回 channel-first float32 tensor。"""
    data, sr = sf.read(path, dtype="float32")
    data = data[np.newaxis, :] if data.ndim == 1 else data.T
    return torch.from_numpy(data), int(sr)


def load_audio_window(
    path: str, start: int, n_frames: int
) -> tuple[torch.Tensor, int]:
    """在 source domain 读取 ``[start, start+n_frames)``，末尾不足时 clip。"""
    data, sr = sf.read(
        path, dtype="float32", start=int(start), frames=int(n_frames)
    )
    data = data[np.newaxis, :] if data.ndim == 1 else data.T
    return torch.from_numpy(data), int(sr)


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    """把 channel-first 波形降为单声道，单声道输入零拷贝返回。"""
    if waveform.ndim != 2:
        raise ValueError(
            f"waveform must be channel-first 2D [C, T], got {tuple(waveform.shape)}"
        )
    return waveform if waveform.shape[0] == 1 else waveform.mean(0, keepdim=True)


def ensure_finite(
    waveform: torch.Tensor,
    *,
    strict: bool,
    track: str,
    sample: str,
) -> torch.Tensor:
    """严格模式拒绝 NaN/Inf；宽容模式原位语义地清零非有限值。"""
    if torch.isfinite(waveform).all():
        return waveform
    if strict:
        raise ValueError(f"NaN/Inf in {track} waveform ({sample})")
    _log.warning("NaN/Inf in %s waveform (%s), sanitizing with nan_to_num", track, sample)
    return torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)


@lru_cache(maxsize=32)
def _torchaudio_resampler(orig_sr: int, target_sr: int):
    import torchaudio

    return torchaudio.transforms.Resample(orig_sr, target_sr)


def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """按需重采样；torchaudio 不可用时回退 librosa。"""
    if int(orig_sr) == int(target_sr):
        return waveform
    try:
        return _torchaudio_resampler(int(orig_sr), int(target_sr))(waveform)
    except Exception:
        _log.warning("torchaudio resample failed, using librosa fallback", exc_info=True)
        import librosa

        array = waveform.detach().cpu().numpy()
        output = np.stack(
            [
                librosa.resample(channel, orig_sr=orig_sr, target_sr=target_sr)
                for channel in array
            ]
        )
        return torch.from_numpy(output).to(device=waveform.device, dtype=waveform.dtype)


def resampled_num_frames(num_frames: int, orig_sr: int, target_sr: int) -> int:
    """把 source-domain 帧数换算为 output-domain 帧数。"""
    if int(orig_sr) == int(target_sr):
        return int(num_frames)
    return (int(num_frames) * int(target_sr) + int(orig_sr) - 1) // int(orig_sr)


def frame_window_between_domains(
    *, start: int, length: int, source_sr: int, output_sr: int
) -> tuple[int, int]:
    """显式把 source-domain window 换算到 output domain。"""
    return (
        resampled_num_frames(start, source_sr, output_sr),
        resampled_num_frames(length, source_sr, output_sr),
    )


def slice_or_pad(
    waveform: torch.Tensor, *, start: int, length: int
) -> torch.Tensor:
    """截取同域窗口，文件末尾不足时右侧补零。"""
    if start < 0 or length < 0:
        raise ValueError(f"start/length must be non-negative, got {start}/{length}")
    segment = waveform[:, int(start) : int(start) + int(length)]
    if segment.shape[1] < length:
        segment = torch.nn.functional.pad(
            segment, (0, int(length) - segment.shape[1]), value=0.0
        )
    return segment
