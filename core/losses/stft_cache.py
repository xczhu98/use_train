"""Generic multi-resolution STFT caching utilities shared by loss implementations.

Used by ``core.losses.composite.CompositeLoss`` to compute a shared STFT once
per resolution when two or more loss components would otherwise redundantly
recompute the same (window, hop) STFT on the same (enhanced, clean) pair.

公共 helper（供各 loss 模块直接导入）：
``cached_hann_window``、``cached_arange``、``stft_real_view``。三者内部使用
模块级 LRU，读写均持锁，可安全地被多个线程（验证线程、DataLoader worker）共用。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence

import torch

# Module-level cache: avoid re-allocating Hann windows / small aranges every step
# (device/length space is bounded, LRU-evicted). 模块级 LRU 可能被多线程
# （DataLoader worker / 多验证线程）同时命中，读写必须持锁。
_MAX_HANN_CACHE = 48
_MAX_ARANGE_CACHE = 96
_hann_cache: OrderedDict[tuple[str, int, str], torch.Tensor] = OrderedDict()
_arange_cache: OrderedDict[tuple[str, int, str], torch.Tensor] = OrderedDict()
_cache_lock = threading.Lock()


def cached_hann_window(wl: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # key 必须含 dtype：不同 dtype 的同长度窗若共用 entry 会静默改变下游数值类型
    key = (str(device), wl, str(dtype))
    with _cache_lock:
        hit = _hann_cache.get(key)
        if hit is not None:
            _hann_cache.move_to_end(key)
            return hit
    w = torch.hann_window(wl, device=device, dtype=dtype)
    with _cache_lock:
        _hann_cache[key] = w
        while len(_hann_cache) > _MAX_HANN_CACHE:
            _hann_cache.popitem(last=False)
    return w


def cached_arange(
    n: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """1D arange(0..n-1) on device; dtype default int64 (index masks)."""
    dt = dtype or torch.int64
    key = (str(device), n, str(dt))
    with _cache_lock:
        hit = _arange_cache.get(key)
        if hit is not None:
            _arange_cache.move_to_end(key)
            return hit
    t = torch.arange(n, device=device, dtype=dt)
    with _cache_lock:
        _arange_cache[key] = t
        while len(_arange_cache) > _MAX_ARANGE_CACHE:
            _arange_cache.popitem(last=False)
    return t


@torch.amp.autocast("cuda", enabled=False)
def stft_real_view(x: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor) -> torch.Tensor:
    """torch.stft → view_as_real, shape (B, F, T, 2). Forced FP32."""
    return torch.view_as_real(
        torch.stft(x.float(), n_fft, hop_length=hop, win_length=n_fft,
                   window=window, return_complex=True, center=True, onesided=True)
    )


def compute_mr_stft_shared_cache(
    enhanced: torch.Tensor,
    clean: torch.Tensor,
    fft_sizes: Sequence[int],
    hop_sizes: Sequence[int],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Compute one shared STFT pair per exact ``(fft_size, hop_size)`` entry.

    ``fft_sizes`` and ``hop_sizes`` are the canonical interface and preserve
    schedule order.
    """
    if not isinstance(fft_sizes, Sequence) or not isinstance(hop_sizes, Sequence):
        raise TypeError("fft_sizes and hop_sizes must both be sequences")
    exact_fft_sizes = tuple(int(size) for size in fft_sizes)
    exact_hop_sizes = tuple(int(size) for size in hop_sizes)

    if not exact_fft_sizes or len(exact_fft_sizes) != len(exact_hop_sizes):
        raise ValueError("fft_sizes and hop_sizes must be non-empty and have equal length")
    if any(size <= 0 for size in exact_fft_sizes) or any(hop <= 0 for hop in exact_hop_sizes):
        raise ValueError("fft_sizes and hop_sizes must contain positive integers")

    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for fft_size, hop_size in zip(exact_fft_sizes, exact_hop_sizes):
        win = cached_hann_window(fft_size, enhanced.device, torch.float32)
        out.append(
            (
                stft_real_view(enhanced, fft_size, hop_size, win),
                stft_real_view(clean, fft_size, hop_size, win),
            )
        )
    return out
