"""Regression tests for the byte-budget audio cache (A16 L1/L2).

Covers ``_ByteLRUCache`` byte accounting + strict LRU order, oversized-entry
pass-through, ``cache_clear()`` compatibility, and the entry-count cap.
Arrays with known ``nbytes`` are injected via monkeypatch so no real audio
decoding is needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile

import core.data.simulation_audio_utils as sau
from core.data.simulation_audio_utils import (
    _ByteLRUCache,
    _load_audio_cached_impl,
    _load_audio_uncached,
)


def _arr(nbytes: int) -> np.ndarray:
    """Return a float32 array occupying exactly *nbytes* bytes."""
    assert nbytes % 4 == 0
    return np.zeros(nbytes // 4, dtype=np.float32)


@pytest.fixture
def small_cache(monkeypatch):
    """Swap the module-level audio cache for a small-budget instance."""
    cache = _ByteLRUCache(max_bytes=1024, max_entries=100)
    monkeypatch.setattr(sau, "_AUDIO_CACHE", cache)
    return cache


def test_total_bytes_never_exceeds_budget(small_cache, monkeypatch):
    monkeypatch.setattr(
        sau, "_load_audio_impl", lambda path, sr, strict=False: _arr(400)
    )
    for i in range(10):
        out = _load_audio_cached_impl(f"/fake/{i}.wav", 16000)
        assert out.nbytes == 400
        assert small_cache.cache_info()["total_bytes"] <= 1024
    # 1024 字节预算 / 每条 400 字节 → 最多保留 2 条
    assert small_cache.cache_info()["entries"] == 2


def test_lru_hit_prevents_eviction(small_cache, monkeypatch):
    monkeypatch.setattr(
        sau, "_load_audio_impl", lambda path, sr, strict=False: _arr(400)
    )
    _load_audio_cached_impl("/fake/a.wav", 16000)
    _load_audio_cached_impl("/fake/b.wav", 16000)  # 缓存: [a, b] 共 800B
    hit = _load_audio_cached_impl("/fake/a.wav", 16000)  # 命中 → a 移到队尾
    assert hit is not None
    _load_audio_cached_impl("/fake/c.wav", 16000)  # 超限 → 逐出队首 b
    assert small_cache.get(("/fake/a.wav", 16000, False)) is not None
    assert small_cache.get(("/fake/b.wav", 16000, False)) is None
    assert small_cache.get(("/fake/c.wav", 16000, False)) is not None


def test_oversized_entry_returned_but_not_cached(small_cache, monkeypatch):
    monkeypatch.setattr(
        sau, "_load_audio_impl", lambda path, sr, strict=False: _arr(4096)
    )
    out = _load_audio_cached_impl("/fake/big.wav", 16000)
    assert out.nbytes == 4096  # 正常返回
    info = small_cache.cache_info()
    assert info["entries"] == 0  # 但不入缓存
    assert info["total_bytes"] == 0


def test_cache_clear_zeroes_state(small_cache, monkeypatch):
    monkeypatch.setattr(
        sau, "_load_audio_impl", lambda path, sr, strict=False: _arr(400)
    )
    _load_audio_cached_impl("/fake/a.wav", 16000)
    # set_epoch / _worker_maybe_clear_caches 经由该属性清缓存
    _load_audio_cached_impl.cache_clear()
    info = small_cache.cache_info()
    assert info["entries"] == 0
    assert info["total_bytes"] == 0


def test_entry_cap_and_byte_cap_both_enforced(monkeypatch):
    cache = _ByteLRUCache(max_bytes=10**9, max_entries=3)
    monkeypatch.setattr(sau, "_AUDIO_CACHE", cache)
    monkeypatch.setattr(
        sau, "_load_audio_impl", lambda path, sr, strict=False: _arr(400)
    )
    for i in range(5):
        _load_audio_cached_impl(f"/fake/{i}.wav", 16000)
    info = cache.cache_info()
    assert info["entries"] == 3  # 条数上限与字节上限同时生效
    assert cache.get(("/fake/0.wav", 16000, False)) is None
    assert cache.get(("/fake/1.wav", 16000, False)) is None
    assert cache.get(("/fake/4.wav", 16000, False)) is not None


def test_tuple_value_nbytes_counted():
    """Tuple entries (e.g. AEC stereo pairs) are measured by summed nbytes."""
    cache = _ByteLRUCache(max_bytes=1000, max_entries=100)
    cache.put("p1", (_arr(400), _arr(400)))  # 800B
    assert cache.cache_info()["total_bytes"] == 800
    cache.put("p2", (_arr(400), _arr(400)))  # 1600B > 预算 → 逐出 p1
    assert cache.get("p1") is None
    assert cache.get("p2") is not None
    assert cache.cache_info()["total_bytes"] == 800


def test_load_audio_uncached_bypasses_cache(small_cache, monkeypatch):
    calls = []

    def fake_impl(path, sr, strict=False):
        calls.append(path)
        return _arr(400)

    monkeypatch.setattr(sau, "_load_audio_impl", fake_impl)
    out1 = _load_audio_uncached("/fake/spk.wav", 16000)
    out2 = _load_audio_uncached("/fake/spk.wav", 16000)
    assert out1 is not out2  # 每次重新加载，返回独立数组
    assert len(calls) == 2
    assert small_cache.cache_info()["entries"] == 0


def test_aec_stereo_cache_byte_budget(tmp_path, monkeypatch):
    """AEC stereo farend/echo cache evicts by byte budget; cache_clear() still works."""
    import tasks.aec.data.dynamic_dataset as aec_ds

    sr = 16000
    rng = np.random.default_rng(0)
    for name in ("pair1.wav", "pair2.wav"):
        data = rng.standard_normal((sr, 2)).astype(np.float32)
        soundfile.write(str(tmp_path / name), data, sr, subtype="FLOAT")
    p1 = str(tmp_path / "pair1.wav")
    p2 = str(tmp_path / "pair2.wav")

    # 每条目 2 × 64000B；预算只容得下 1 条
    cache = _ByteLRUCache(max_bytes=200_000, max_entries=100)
    monkeypatch.setattr(aec_ds, "_AEC_STEREO_CACHE", cache)

    f1, e1 = aec_ds._load_stereo_farend_echo_cached_impl(p1, sr, False, 1, 0)
    assert cache.cache_info()["entries"] == 1
    assert cache.cache_info()["total_bytes"] == f1.nbytes + e1.nbytes
    aec_ds._load_stereo_farend_echo_cached_impl(p2, sr, False, 1, 0)
    info = cache.cache_info()
    assert info["entries"] == 1  # 超出字节预算 → 逐出旧条目
    assert info["total_bytes"] <= 200_000
    # set_epoch 经由该属性清缓存
    aec_ds._load_stereo_farend_echo_cached_impl.cache_clear()
    assert cache.cache_info()["entries"] == 0


def test_aec_stereo_transient_read_retries_then_caches(monkeypatch):
    """瞬态 NFS 读取失败有限退避，成功结果才进入 stereo cache。"""
    import tasks.aec.data.dynamic_dataset as aec_ds

    cache = _ByteLRUCache(max_bytes=1_000_000, max_entries=10)
    monkeypatch.setattr(aec_ds, "_AEC_STEREO_CACHE", cache)
    audio = np.stack([np.arange(8), np.arange(8) + 10], axis=1).astype(np.float32)
    read_calls = []
    sleep_calls = []

    def flaky_read(*args, **kwargs):
        read_calls.append(1)
        if len(read_calls) == 1:
            raise soundfile.LibsndfileError("transient NFS read failure")
        return audio, 16000

    monkeypatch.setattr(soundfile, "read", flaky_read)
    monkeypatch.setattr(aec_ds.time, "sleep", sleep_calls.append)
    far, echo = aec_ds._load_stereo_farend_echo_cached_impl(
        "/fake/pair.wav", 16000, False, 1, 0
    )

    assert len(read_calls) == 2
    assert sleep_calls == [0.02]
    np.testing.assert_array_equal(far, audio[:, 1])
    np.testing.assert_array_equal(echo, audio[:, 0])
    assert cache.cache_info()["entries"] == 1


def test_aec_stereo_failure_placeholder_is_not_cached(monkeypatch):
    """non-strict 静音占位不得缓存，下一调用必须重新尝试读取。"""
    import tasks.aec.data.dynamic_dataset as aec_ds

    cache = _ByteLRUCache(max_bytes=1_000_000, max_entries=10)
    monkeypatch.setattr(aec_ds, "_AEC_STEREO_CACHE", cache)
    read_calls = []

    def failing_read(*args, **kwargs):
        read_calls.append(1)
        raise soundfile.LibsndfileError("persistent NFS read failure")

    monkeypatch.setattr(soundfile, "read", failing_read)
    monkeypatch.setattr(aec_ds.time, "sleep", lambda _: None)
    first = aec_ds._load_stereo_farend_echo_cached_impl(
        "/fake/pair.wav", 16000, False, 1, 0
    )
    second = aec_ds._load_stereo_farend_echo_cached_impl(
        "/fake/pair.wav", 16000, False, 1, 0
    )

    assert len(read_calls) == 6
    assert cache.cache_info()["entries"] == 0
    for pair in (first, second):
        assert pair[0].shape == pair[1].shape == (32000,)
        np.testing.assert_array_equal(pair[0], 0.0)
        np.testing.assert_array_equal(pair[1], 0.0)


def test_load_audio_libsndfile_error_goes_through_fail_path(monkeypatch):
    """LibsndfileError (RuntimeError subclass) must not bypass _fail handling."""
    from core.data.simulation_audio_utils import _load_audio_impl

    calls = []

    def failing_read(*args, **kwargs):
        calls.append(1)
        raise soundfile.LibsndfileError("Internal psf_fseek() failed.")

    monkeypatch.setattr(soundfile, "read", failing_read)

    # strict=False → 返回静音占位，且至少尝试重试一次
    out = _load_audio_impl("/fake/bad.wav", 16000, strict=False)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert len(calls) >= 2  # 重试机制

    calls.clear()
    # strict=True → 抛出 RuntimeError，消息包含文件路径
    with pytest.raises(RuntimeError, match="Failed to read audio file.*bad\\.wav"):
        _load_audio_impl("/fake/bad.wav", 16000, strict=True)


def test_mono_failure_placeholder_is_not_cached(monkeypatch):
    """mono 音频读取失败后文件恢复时，下一次调用必须重新解码。"""
    cache = _ByteLRUCache(max_bytes=1_000_000, max_entries=10)
    monkeypatch.setattr(sau, "_AUDIO_CACHE", cache)
    calls = []

    def flaky_read(*args, **kwargs):
        calls.append(1)
        if len(calls) <= 2:
            raise soundfile.LibsndfileError("transient NFS read failure")
        return np.ones((8, 1), dtype=np.float32), 16000

    monkeypatch.setattr(soundfile, "read", flaky_read)
    monkeypatch.setattr(sau.time, "sleep", lambda _: None)

    first = _load_audio_cached_impl("/fake/recover.wav", 16000, strict=False)
    assert np.all(first == 0.0)
    assert cache.cache_info()["entries"] == 0

    second = _load_audio_cached_impl("/fake/recover.wav", 16000, strict=False)
    np.testing.assert_array_equal(second, 1.0)
    assert cache.cache_info()["entries"] == 1


def test_spatial_noise_retries_transient_read(monkeypatch, tmp_path):
    """空间背景音频与 mono 路径一样容忍一次瞬态存储读取失败。"""
    from core.data.simulation_audio_utils import _read_spatial_noise_2d

    cache = _ByteLRUCache(max_bytes=1_000_000, max_entries=10)
    monkeypatch.setattr(sau, "_AUDIO_CACHE", cache)
    calls = []
    audio = np.arange(12, dtype=np.float32).reshape(6, 2)

    def flaky_read(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise soundfile.LibsndfileError("transient NFS read failure")
        return audio, 16000

    monkeypatch.setattr(soundfile, "read", flaky_read)
    monkeypatch.setattr(sau.time, "sleep", lambda _: None)

    out = _read_spatial_noise_2d(
        str(tmp_path / "spatial.wav"), 16000, strict=True,
        rng=np.random.default_rng(0),
    )

    assert len(calls) == 2
    np.testing.assert_array_equal(out, audio.T)

pytestmark = [pytest.mark.unit]
