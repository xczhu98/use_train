"""Tests for the optional long-file window-read path (A16 L4, default off).

``USE_TRAIN_AUDIO_WINDOW_READ_MIN_SECONDS`` enables reading only the cropped
region of long same-sample-rate files. These tests pin the contract:
window-read output is sample-identical to the legacy full-read crop, the rng
draw sequence is unchanged, and the default (0) keeps the legacy cached path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile

import core.data.simulation_audio_utils as sau
from core.data.simulation_audio_utils import _read_cached_2d


@pytest.fixture
def long_wav(tmp_path):
    """3 s mono float32 wav; returns (path, sr, samples)."""
    sr = 16000
    rng = np.random.default_rng(42)
    data = rng.standard_normal(3 * sr).astype(np.float32)
    path = tmp_path / "long.wav"
    soundfile.write(str(path), data, sr, subtype="FLOAT")
    return str(path), sr, data


@pytest.fixture(autouse=True)
def audio_cache(monkeypatch):
    """每个用例替换模块级音频 LRU，避免用例直接改写全局缓存状态。"""
    fresh = sau._ByteLRUCache(sau._AUDIO_CACHE_BYTES, sau._AUDIO_CACHE_MAXSIZE)
    monkeypatch.setattr(sau, "_AUDIO_CACHE", fresh)
    return fresh


def test_window_read_matches_full_read_crop(long_wav, monkeypatch):
    path, sr, data = long_wav
    monkeypatch.setattr(sau, "_AUDIO_WINDOW_READ_MIN_SECONDS", 2.0)
    sau._AUDIO_CACHE.clear()
    max_dur = sr  # 裁剪 1 s
    seed = 7
    got = _read_cached_2d(path, sr, max_dur, rng=np.random.default_rng(seed))
    # 参考：整读 + 同 seed 同参数抽取 start 后 crop
    span = len(data) - max_dur
    start = int(np.random.default_rng(seed).integers(0, span + 1))
    ref = data[start : start + max_dur].reshape(1, -1)
    np.testing.assert_array_equal(got, ref)
    # 窗口路径不触碰音频 LRU
    assert sau._AUDIO_CACHE.cache_info()["entries"] == 0


def test_window_read_failure_falls_back_without_cache(long_wav, monkeypatch):
    """窗口读失败时整读回退不应把长文件写入音频 LRU。"""
    path, sr, data = long_wav
    original_read = soundfile.read

    def fail_window_read(*args, **kwargs):
        if "start" in kwargs:
            raise RuntimeError("injected window read failure")
        return original_read(*args, **kwargs)

    monkeypatch.setattr(soundfile, "read", fail_window_read)
    monkeypatch.setattr(sau, "_AUDIO_WINDOW_READ_MIN_SECONDS", 2.0)
    sau._AUDIO_CACHE.clear()
    max_dur = sr
    seed = 11
    got = _read_cached_2d(path, sr, max_dur, rng=np.random.default_rng(seed))
    span = len(data) - max_dur
    start = int(np.random.default_rng(seed).integers(0, span + 1))
    np.testing.assert_array_equal(got, data[start : start + max_dur].reshape(1, -1))
    assert sau._AUDIO_CACHE.cache_info()["entries"] == 0


def test_window_read_rng_start_matches_legacy(long_wav, monkeypatch):
    """窗口路径与旧整读路径的 rng 抽取与输出必须完全一致。"""
    path, sr, _ = long_wav
    max_dur = sr
    seed = 123
    monkeypatch.setattr(sau, "_AUDIO_WINDOW_READ_MIN_SECONDS", 2.0)
    sau._AUDIO_CACHE.clear()
    got = _read_cached_2d(path, sr, max_dur, rng=np.random.default_rng(seed))
    monkeypatch.setattr(sau, "_AUDIO_WINDOW_READ_MIN_SECONDS", 0.0)
    sau._AUDIO_CACHE.clear()
    legacy = _read_cached_2d(path, sr, max_dur, rng=np.random.default_rng(seed))
    np.testing.assert_array_equal(got, legacy)


def test_window_read_disabled_by_default_uses_cache(long_wav, monkeypatch):
    """阈值 0（默认）= 关闭，走整读 LRU 路径。"""
    path, sr, _ = long_wav
    monkeypatch.setattr(sau, "_AUDIO_WINDOW_READ_MIN_SECONDS", 0.0)
    sau._AUDIO_CACHE.clear()
    out = _read_cached_2d(path, sr, sr, rng=np.random.default_rng(0))
    assert out.shape == (1, sr)
    assert sau._AUDIO_CACHE.cache_info()["entries"] == 1


def test_window_read_below_threshold_uses_cache(long_wav, monkeypatch):
    """文件时长低于阈值：整读旧路径。"""
    path, sr, _ = long_wav
    monkeypatch.setattr(sau, "_AUDIO_WINDOW_READ_MIN_SECONDS", 5.0)  # 3 s 文件不达标
    sau._AUDIO_CACHE.clear()
    out = _read_cached_2d(path, sr, sr, rng=np.random.default_rng(0))
    assert out.shape == (1, sr)
    assert sau._AUDIO_CACHE.cache_info()["entries"] == 1


def test_window_read_sr_mismatch_falls_back(tmp_path, monkeypatch):
    """无声明 metadata 时异采样率保持旧的整读回退语义。"""
    file_sr, target_sr = 8000, 16000
    rng = np.random.default_rng(1)
    data = rng.standard_normal(3 * file_sr).astype(np.float32)
    path = tmp_path / "long8k.wav"
    soundfile.write(str(path), data, file_sr, subtype="FLOAT")
    monkeypatch.setattr(sau, "_AUDIO_WINDOW_READ_MIN_SECONDS", 2.0)
    sau._AUDIO_CACHE.clear()
    out = _read_cached_2d(str(path), target_sr, target_sr, rng=np.random.default_rng(5))
    assert out.shape == (1, target_sr)
    assert sau._AUDIO_CACHE.cache_info()["entries"] == 1  # 未进入窗口路径，仍走整读缓存
    assert np.isfinite(out).all()


@pytest.mark.parametrize("window_enabled", [False, True])
def test_crop_record_and_forced_replay_preserve_waveform_and_rng_state(
    long_wav, monkeypatch, window_enabled
):
    """forced crop 消费等价 legacy draw，并复现 record 的精确窗口。"""
    path, sr, _ = long_wav
    max_dur = sr
    monkeypatch.setattr(
        sau, "_AUDIO_WINDOW_READ_MIN_SECONDS", 2.0 if window_enabled else 0.0
    )
    sau._AUDIO_CACHE.clear()
    record: dict = {}
    legacy_rng = np.random.default_rng(2026)
    expected = _read_cached_2d(
        path,
        sr,
        max_dur,
        rng=legacy_rng,
        crop_record=record,
    )
    expected_state = legacy_rng.bit_generator.state

    assert record["crop"].identity == str(Path(path).resolve())
    assert record["crop"].num_samples == max_dur

    sau._AUDIO_CACHE.clear()
    replay_rng = np.random.default_rng(2026)
    replay_record: dict = {}
    replayed = _read_cached_2d(
        path,
        sr,
        max_dur,
        rng=replay_rng,
        crop_record=replay_record,
        forced_crop=record["crop"],
    )

    np.testing.assert_array_equal(replayed, expected)
    assert replay_rng.bit_generator.state == expected_state
    assert replay_record["crop"] == record["crop"]


def test_forced_crop_rejects_wrong_identity_after_consuming_no_io(long_wav, monkeypatch):
    path, sr, _ = long_wav
    monkeypatch.setattr(sau, "_AUDIO_WINDOW_READ_MIN_SECONDS", 0.0)
    record: dict = {}
    _read_cached_2d(
        path, sr, sr, rng=np.random.default_rng(1), crop_record=record
    )
    wrong = record["crop"].__class__(
        identity="/wrong/audio.wav",
        source_sample_rate=record["crop"].source_sample_rate,
        source_num_samples=record["crop"].source_num_samples,
        start_sample=record["crop"].start_sample,
        num_samples=record["crop"].num_samples,
    )

    with pytest.raises(ValueError, match="identity"):
        _read_cached_2d(
            path,
            sr,
            sr,
            rng=np.random.default_rng(1),
            forced_crop=wrong,
        )


pytestmark = [pytest.mark.unit]
