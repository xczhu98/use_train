"""Shared audio I/O helpers for dynamic near-end simulation (SE + AEC ``dynamic_dataset``).

Extracted from ``tasks/se/data/dynamic_dataset.py`` so SE and AEC paths share one LRU cache
implementation. ``_align_to_max_duration`` stays in each dataset module (AEC adds
``extras``). ``_read_cached_2d`` requires an explicit ``rng`` for long-file random crops.

Environment:
  ``USE_TRAIN_AUDIO_CACHE_BYTES``: byte budget per DataLoader worker for decoded mono
  audio (default 512 MiB; ``0`` disables caching). The byte-budget LRU
  (``_ByteLRUCache``) evicts oldest entries once total ``nbytes`` exceeds the budget.
  ``USE_TRAIN_AUDIO_CACHE_MAXSIZE``: entry-count cap for the same cache (default
  2048, kept for backward compatibility). Paths are normalized with ``Path.resolve``
  before lookup so equivalent file paths share one cache entry.
  ``USE_TRAIN_AUDIO_WINDOW_READ_MIN_SECONDS``: long-file window-read threshold in
  seconds (default ``0`` = disabled). When enabled, same-sample-rate files at
  least this long are read via ``soundfile.read(start=..., frames=...)`` instead
  of decoding the whole file; the rng draw sequence is identical to the legacy
  full-read crop.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import soundfile

from core.data.acoustic_scene_plan import AudioCropPlan
from tools.shared.pickle_io import load_pkl_file

# Per-worker LRU for decoded audio; raise on large manifests / many workers (env override).
_AUDIO_CACHE_MAXSIZE = max(32, int(os.environ.get("USE_TRAIN_AUDIO_CACHE_MAXSIZE", "2048")))
# 每 worker 音频缓存字节预算（默认 512 MiB；0 = 禁用缓存）
_AUDIO_CACHE_BYTES = max(
    0,
    int(os.environ.get("USE_TRAIN_AUDIO_CACHE_BYTES", str(512 * 1024 * 1024))),
)
# 长文件窗口读阈值（秒，默认 0 = 关闭）：同 sr 且时长 ≥ 阈值时只读裁剪窗口
_AUDIO_WINDOW_READ_MIN_SECONDS = max(
    0.0,
    float(os.environ.get("USE_TRAIN_AUDIO_WINDOW_READ_MIN_SECONDS", "0")),
)
_AUDIO_LOAD_STATE = threading.local()


class _ByteLRUCache:
    """Thread-safe LRU cache bounded by total ``nbytes`` (plus an entry-count cap).

    Replaces ``functools.lru_cache`` for decoded-audio buffers so memory per
    DataLoader worker cannot exceed a fixed byte budget. Values are numpy arrays
    (or tuples/lists of them); entry size is the sum of ``value.nbytes``. An
    entry larger than the whole budget is never stored — the caller still gets
    the freshly loaded value.
    """

    def __init__(self, max_bytes: int, max_entries: int) -> None:
        self.max_bytes = int(max_bytes)
        self.max_entries = int(max_entries)
        self._data: OrderedDict = OrderedDict()
        self._total_bytes = 0
        self._lock = threading.Lock()

    @staticmethod
    def _value_nbytes(value) -> int:
        """Recursively measure entry size; numpy arrays use ``nbytes``."""
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, (tuple, list)):
            return sum(_ByteLRUCache._value_nbytes(v) for v in value)
        return int(getattr(value, "nbytes", 0))

    def get(self, key):
        """Return the cached value or ``None``; a hit moves the entry to the LRU tail."""
        with self._lock:
            try:
                value = self._data.pop(key)
            except KeyError:
                return None
            self._data[key] = value
            return value

    def put(self, key, value) -> None:
        """Insert *value* and evict from the head until both caps are satisfied."""
        nbytes = self._value_nbytes(value)
        with self._lock:
            old = self._data.pop(key, None)
            if old is not None:
                self._total_bytes -= self._value_nbytes(old)
            if nbytes > self.max_bytes:
                return  # 单条超过预算：不入缓存，调用方照常使用
            self._data[key] = value
            self._total_bytes += nbytes
            while len(self._data) > self.max_entries or self._total_bytes > self.max_bytes:
                _, evicted = self._data.popitem(last=False)
                self._total_bytes -= self._value_nbytes(evicted)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._total_bytes = 0

    cache_clear = clear

    def cache_info(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._data),
                "total_bytes": self._total_bytes,
                "max_bytes": self.max_bytes,
                "max_entries": self.max_entries,
            }


def _normalize_audio_cache_key(path: str) -> str:
    """Expand ``~`` and resolve paths before any audio/stat/cache operation."""
    expanded = os.path.expanduser(path)
    try:
        return str(Path(expanded).resolve(strict=False))
    except OSError:
        return os.path.normpath(expanded)


def _resample_audio(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio. For 2-D ``(C, T)`` arrays, transposes to ``(T, C)`` for soxr."""
    if orig_sr == target_sr:
        return data
    need_transpose = data.ndim == 2 and data.shape[0] < data.shape[1]
    if need_transpose:
        data = data.T  # (C, T) → (T, C) for soxr
    try:
        import soxr

        out = soxr.resample(data, orig_sr, target_sr)
        return out.astype(np.float32).T if need_transpose else out.astype(np.float32)
    except ImportError:
        pass
    import librosa

    if need_transpose:
        data = data.T  # (T, C) → (C, T) for librosa
    return librosa.resample(
        data, orig_sr=orig_sr, target_sr=target_sr, res_type="polyphase"
    ).astype(np.float32)


def _align_with_window(
    arr: np.ndarray, length: int, window_ms: float = 5.0, sr: int = 16000
) -> np.ndarray:
    """Align array to target length with smart padding and window smoothing."""
    is_1d = arr.ndim == 1
    if is_1d:
        arr = arr.reshape(1, -1)

    c, t = arr.shape

    if t < length:
        pad_len = length - t

        if t >= 10:
            repeats = (pad_len // t) + 1
            repeated = np.tile(arr, (1, repeats))
            padding = repeated[:, :pad_len]

            window_len = min(int(window_ms * sr / 1000), t // 4, pad_len // 2)
            if window_len > 1:
                hann_window = np.hanning(window_len * 2)
                fade_out = hann_window[:window_len]
                fade_in = hann_window[window_len:]
                arr = arr.copy()
                arr[:, -window_len:] *= fade_out
                padding[:, :window_len] *= fade_in
        else:
            padding = np.zeros((c, pad_len), dtype=arr.dtype)

        arr = np.concatenate([arr, padding], axis=1)

    arr = arr[:, :length]

    if is_1d:
        arr = arr.flatten()

    return arr


def _load_audio_impl(path: str, sr: int, strict: bool = False) -> np.ndarray:
    """Load & resample audio to 1D float32; expand ``~`` at the I/O boundary."""
    _log = logging.getLogger(__name__)
    _AUDIO_LOAD_STATE.failed = False
    path = os.path.expanduser(path)
    path_obj = Path(path)

    def _fail(msg: str, exc: BaseException | None = None) -> np.ndarray:
        _AUDIO_LOAD_STATE.failed = True
        if strict:
            raise RuntimeError(f"{msg}: {path}") from exc
        _log.warning("%s (%s), using silence placeholder", msg, path)
        return np.zeros(max(sr * 2, 1), dtype=np.float32)

    orig_sr = sr

    if path_obj.suffix.lower() == ".pkl":
        try:
            # 受限反序列化（仅 numpy/基础类型；设 PREPARE_SCP_ALLOW_UNSAFE_PICKLE=1 回退完整 pickle）
            data = load_pkl_file(
                path, purpose="load simulation audio or RIR pickle"
            )

            # 移动 RIR（pkl 含 moving: True，rir 为 [T_seg, n] 分段序列）时保留 2D 形状，
            # 由消费端 add_moving_reverberation 做分段卷积；静态 RIR 仍压成 1D。
            _moving = False
            if isinstance(data, dict):
                rir = None
                for key in (
                    "rir",
                    "impulse_response",
                    "h",
                    "source_rir",
                    "data",
                    "ir",
                    "signal",
                    "audio",
                ):
                    v = data.get(key)
                    if v is not None and isinstance(v, np.ndarray) and v.ndim >= 1:
                        rir = v
                        break
                if rir is None:
                    for v in data.values():
                        if isinstance(v, np.ndarray) and v.ndim >= 1:
                            rir = v
                            break
                if rir is None:
                    return _fail(
                        f"Pickle has no rir array key, keys={list(data.keys())}"
                    )
                _moving = bool(data.get("moving", False))
                orig_sr = int(data.get("fs", data.get("sample_rate", data.get("sr", sr))))
                data = rir
            elif isinstance(data, np.ndarray):
                orig_sr = sr
                data = data
            else:
                return _fail(f"Unsupported pickle content type {type(data)}")

            if data.ndim > 1 and not _moving:
                data = data[:, 0] if data.shape[1] < data.shape[0] else data[0, :]
            data = data.astype(np.float32)

        except RuntimeError:
            raise
        except Exception as e:
            return _fail("Failed to load pickle", e)

    else:
        # soundfile.LibsndfileError 继承自 RuntimeError，需先捕获，避免绕过 _fail 的
        # strict/placeholder 处理。对 libsndfile I/O 错误做短暂重试，以容忍 NFS/存储
        # 瞬时抖动；持续失败时按 strict 参数决定抛错或返回静音占位。
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                data, orig_sr = soundfile.read(path, dtype="float32", always_2d=True)
                break
            except soundfile.LibsndfileError as e:
                if attempt == max_attempts - 1:
                    return _fail("Failed to read audio file", e)
                time.sleep(0.05)
            except RuntimeError:
                raise
            except Exception as e:
                return _fail("Failed to read audio file", e)
        data = data[:, 0]

    if len(data) == 0:
        return _fail("Zero-length audio")

    if orig_sr != sr:
        data = _resample_audio(data, orig_sr, sr)

    out = data.astype(np.float32)
    if strict and not np.isfinite(out).all():
        raise ValueError(f"Non-finite samples after load/resample: {path}")
    if not strict and not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


_AUDIO_CACHE = _ByteLRUCache(_AUDIO_CACHE_BYTES, _AUDIO_CACHE_MAXSIZE)


def _load_audio_cached_impl(path: str, sr: int, strict: bool = False) -> np.ndarray:
    """Byte-budget LRU wrapper around ``_load_audio_impl`` (see ``_ByteLRUCache``).

    ``cache_clear()`` 属性供显式清理调用；解码音频 LRU 不随 ``set_epoch`` 或
    ``_worker_maybe_clear_caches`` 清空，仍由自身字节/条目预算约束。
    """
    key = (path, int(sr), bool(strict))
    cached = _AUDIO_CACHE.get(key)
    if cached is not None:
        return cached
    # monkeypatch/自定义 loader 可能不设置状态；每次调用前显式清零，避免把
    # 上一次线程中的失败标记泄漏到本次成功读取。
    _AUDIO_LOAD_STATE.failed = False
    value = _load_audio_impl(path, sr, strict)
    if not bool(getattr(_AUDIO_LOAD_STATE, "failed", False)):
        _AUDIO_CACHE.put(key, value)
    return value


def _clear_audio_cache() -> None:
    """Clear the module-level audio cache (dynamic lookup, monkeypatch-safe)."""
    _AUDIO_CACHE.clear()


def _audio_cache_info() -> dict:
    return _AUDIO_CACHE.cache_info()


_load_audio_cached_impl.cache_clear = _clear_audio_cache  # type: ignore[attr-defined]
_load_audio_cached_impl.cache_info = _audio_cache_info  # type: ignore[attr-defined]


def _load_audio_uncached(path: str, sr: int, strict: bool = False) -> np.ndarray:
    """Load & resample audio without touching either decoded-audio cache."""
    return _load_audio_impl(_normalize_audio_cache_key(path), sr, strict)


def _load_audio_cached(path: str, sr: int, strict: bool = False) -> np.ndarray:
    """Load & resample audio to 1D float32 with per-worker byte-budget LRU cache.

    Paths are normalized (``resolve``) before lookup so the same file under different spellings
    hits one cache entry. Budget: ``USE_TRAIN_AUDIO_CACHE_BYTES`` (default 512 MiB);
    entry cap: ``USE_TRAIN_AUDIO_CACHE_MAXSIZE`` (default 2048).

    .. warning::

       返回值是缓存内共享的同一个 ndarray，**禁止原地修改**（切片赋值、
       ``+=``、``np.nan_to_num(out=...)`` 等），否则会污染后续命中同一
       entry 的所有调用方。需要修改时请先 ``.copy()`` / ``.astype(...)``。
    """
    return _load_audio_cached_impl(_normalize_audio_cache_key(path), sr, strict)


def _align_noise_2d_to_length(
    noise_2d: np.ndarray,
    target_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise_2d.ndim != 2:
        raise ValueError(
            f"_align_noise_2d_to_length expects (C, T), got shape {noise_2d.shape}"
        )
    c, t = int(noise_2d.shape[0]), int(noise_2d.shape[1])
    if target_len <= 0:
        return np.zeros((c, 0), dtype=np.float32)
    if t == 0:
        return np.zeros((c, target_len), dtype=np.float32)
    if t == target_len:
        return np.ascontiguousarray(noise_2d, dtype=np.float32)
    if t > target_len:
        start = int(rng.integers(0, t - target_len + 1))
        return noise_2d[:, start : start + target_len].astype(np.float32, copy=False)
    reps = (target_len + t - 1) // t
    tiled = np.tile(noise_2d, (1, reps))
    return tiled[:, :target_len].astype(np.float32)


@lru_cache(maxsize=2048)
def _audio_file_info(path: str) -> tuple[int, int] | None:
    """Return ``(frames, samplerate)`` via ``soundfile.info``; ``None`` on failure."""
    try:
        info = soundfile.info(path)
    except Exception:
        return None
    return int(info.frames), int(info.samplerate)


def _validate_forced_crop_identity(
    forced_crop: AudioCropPlan | None, identity: str
) -> None:
    """在任何 decode/I/O 前校验 forced crop 的资源 identity。"""
    if forced_crop is not None and forced_crop.identity != identity:
        raise ValueError(
            "forced crop identity does not match requested audio: "
            f"{forced_crop.identity!r} != {identity!r}"
        )


def _resolve_crop_start(
    *,
    identity: str,
    sample_rate: int,
    source_num_samples: int,
    num_samples: int,
    rng: np.random.Generator,
    crop_record: dict[str, Any] | None,
    forced_crop: AudioCropPlan | None,
) -> int:
    """在 legacy draw 位置消费一次 draw，再记录或应用精确 crop。"""
    span = source_num_samples - num_samples
    if span < 0:
        raise ValueError("crop num_samples exceeds decoded source length")
    drawn_start = int(rng.integers(0, span + 1))
    start = drawn_start
    if forced_crop is not None:
        if forced_crop.source_sample_rate != sample_rate:
            raise ValueError("forced crop source_sample_rate does not match decoded audio")
        if forced_crop.source_num_samples != source_num_samples:
            raise ValueError("forced crop source_num_samples does not match decoded audio")
        if forced_crop.num_samples != num_samples:
            raise ValueError("forced crop num_samples does not match requested duration")
        start = forced_crop.start_sample
    crop = AudioCropPlan(
        identity=identity,
        source_sample_rate=sample_rate,
        source_num_samples=source_num_samples,
        start_sample=start,
        num_samples=num_samples,
    )
    if crop_record is not None:
        crop_record["crop"] = crop
        crop_record["drawn_start_sample"] = drawn_start
        crop_record["forced"] = forced_crop is not None
    return start


def _try_window_read(
    path: str,
    sr: int,
    max_duration: int,
    *,
    strict: bool = False,
    rng: np.random.Generator,
    crop_record: dict[str, Any] | None = None,
    forced_crop: AudioCropPlan | None = None,
) -> np.ndarray | None:
    """Window-read long same-sr files; return ``None`` when full read is needed."""
    key = _normalize_audio_cache_key(path)
    _validate_forced_crop_identity(forced_crop, key)
    info = _audio_file_info(key)
    if info is None:
        return None
    n_frames, file_sr = info
    if file_sr != sr or n_frames <= max_duration:
        return None
    if n_frames / file_sr < _AUDIO_WINDOW_READ_MIN_SECONDS:
        return None
    start = _resolve_crop_start(
        identity=key,
        sample_rate=file_sr,
        source_num_samples=n_frames,
        num_samples=max_duration,
        rng=rng,
        crop_record=crop_record,
        forced_crop=forced_crop,
    )
    try:
        data, read_sr = soundfile.read(
            key, start=start, frames=max_duration, dtype="float32", always_2d=True
        )
    except Exception:
        data, read_sr = None, None
    if data is not None and read_sr == file_sr and data.shape[0] == max_duration:
        out = np.ascontiguousarray(data[:, 0], dtype=np.float32)
        if strict and not np.isfinite(out).all():
            raise ValueError(f"Non-finite samples after window read: {path}")
        if not strict and not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.reshape(1, -1)
    # 窗口读失败、短读或采样率不符：整读回退但不进入 LRU，避免长文件大 buffer 挤出热项。
    full = _load_audio_uncached(key, sr, strict)
    if len(full) <= max_duration:
        return full.reshape(1, -1).copy()
    start = min(start, len(full) - max_duration)
    return full[start : start + max_duration].copy().reshape(1, -1)


def _read_spatial_noise_2d(
    path: str, sr: int, max_duration: int = -1, *, strict: bool = False,
    rng: np.random.Generator, crop_record=None, forced_crop=None,
) -> np.ndarray:
    """空间 additive noise 专用读取；保留磁盘通道，不影响 mono speech/RIR 缓存。"""
    key = _normalize_audio_cache_key(path)
    _validate_forced_crop_identity(forced_crop, key)
    cache_key = ("spatial-noise", key, int(sr), bool(strict))
    full = _AUDIO_CACHE.get(cache_key)
    if full is None:
        for attempt in range(2):
            try:
                wave, source_sr = soundfile.read(key, dtype="float32", always_2d=True)
                break
            except soundfile.LibsndfileError:
                if attempt == 1:
                    raise
                time.sleep(0.05)
        full = np.ascontiguousarray(wave.T)
        if full.shape[0] < 1 or full.shape[1] < 1:
            raise ValueError(f"empty spatial noise: {key}")
        if source_sr != sr:
            # 每个 channel 独立重采样时间轴；不能按短波形形状猜测 C/T。
            full = np.stack([_resample_audio(channel, source_sr, sr) for channel in full])
        if not np.isfinite(full).all():
            if strict:
                raise ValueError(f"non-finite spatial noise: {key}")
            full = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0)
        full = np.ascontiguousarray(full, dtype=np.float32)
        full.setflags(write=False)
        _AUDIO_CACHE.put(cache_key, full)
    length = full.shape[-1]
    if max_duration > 0 and length > max_duration:
        start = _resolve_crop_start(
            identity=key, sample_rate=sr, source_num_samples=length,
            num_samples=max_duration, rng=rng, crop_record=crop_record, forced_crop=forced_crop,
        )
        return full[:, start:start+max_duration].copy()
    if forced_crop is not None:
        raise ValueError("forced crop requires a source longer than max_duration")
    return full.copy()


def _read_cached_2d(
    path: str,
    sr: int,
    max_duration: int = -1,
    *,
    strict: bool = False,
    rng: np.random.Generator,
    crop_record: dict[str, Any] | None = None,
    forced_crop: AudioCropPlan | None = None,
) -> np.ndarray:
    """读取 owned crop；record/forced seam 不改变默认 draw 位置与次数。"""
    key = _normalize_audio_cache_key(path)
    _validate_forced_crop_identity(forced_crop, key)
    if max_duration > 0 and _AUDIO_WINDOW_READ_MIN_SECONDS > 0:
        window = _try_window_read(
            path,
            sr,
            max_duration,
            strict=strict,
            rng=rng,
            crop_record=crop_record,
            forced_crop=forced_crop,
        )
        if window is not None:
            return window
    full = _load_audio_cached(path, sr, strict)
    if full.ndim == 2:
        # 移动 RIR（pkl 含 moving 标记）：[T_seg, n] 分段序列，保留结构供分段卷积
        return full.copy()
    if max_duration > 0 and len(full) > max_duration:
        start = _resolve_crop_start(
            identity=key,
            sample_rate=sr,
            source_num_samples=len(full),
            num_samples=max_duration,
            rng=rng,
            crop_record=crop_record,
            forced_crop=forced_crop,
        )
        return full[start : start + max_duration].copy().reshape(1, -1)
    if forced_crop is not None:
        raise ValueError("forced crop requires a source longer than max_duration")
    # Copy so callers cannot mutate LRU-cached buffers.
    return full.reshape(1, -1).copy()


@lru_cache(maxsize=65536)
def _read_rir_meta_segment_sec(path: str) -> float | None:
    """读取移动 RIR pkl 的 ``segment_sec`` 元信息（受限反序列化，按 path 缓存）。

    移动 RIR 的分段时间（生成侧 ``--segment-sec``）写在 pkl 里；消费侧优先用它，
    ``cfg.segment_sec`` 仅作兜底，避免两侧参数不一致导致段边界错位。
    静态 RIR、缺该字段或文件不存在时返回 ``None``；其他反序列化/读取错误会记
    ``warning``（含 traceback）后仍回落到 ``cfg.segment_sec``。
    """
    try:
        data = load_pkl_file(
            _normalize_audio_cache_key(path), purpose="read moving RIR segment metadata"
        )
    except FileNotFoundError:
        return None
    except Exception:
        logging.getLogger(__name__).warning(
            "moving RIR metadata unreadable at %s; falling back to cfg.segment_sec",
            path,
            exc_info=True,
        )
        return None
    if isinstance(data, dict):
        try:
            return float(data["segment_sec"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


__all__ = [
    "_resample_audio",
    "_align_with_window",
    "_normalize_audio_cache_key",
    "_AUDIO_CACHE_MAXSIZE",
    "_AUDIO_CACHE_BYTES",
    "_ByteLRUCache",
    "_AUDIO_CACHE",
    "_load_audio_impl",
    "_load_audio_cached_impl",
    "_load_audio_cached",
    "_load_audio_uncached",
    "_align_noise_2d_to_length",
    "_read_cached_2d",
    "_read_rir_meta_segment_sec",
]
