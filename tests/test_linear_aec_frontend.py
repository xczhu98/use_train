"""Unit tests for Linear AEC frontend cache safety."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tasks.aec.data import linear_aec_frontend as linear_aec
from tasks.aec.data.linear_aec_frontend import LinearAECConfig, LinearAECFrontend


pytestmark = pytest.mark.unit


class _StubBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.frame_size = 4

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        self.calls += 1
        return (mic + self.calls).astype(np.float32)


def _frontend(cache_dir=None, cache_mem_max=0, sample_rate=16000):
    frontend = LinearAECFrontend(
        {
            "backend": None,
            "cache_dir": None if cache_dir is None else str(cache_dir),
            "cache_mem_max": cache_mem_max,
        },
        sample_rate=sample_rate,
    )
    frontend._backend = _StubBackend()
    return frontend


def test_zero_memory_cache_allows_disk_cache(tmp_path):
    mic = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    ref = np.zeros_like(mic)
    first = _frontend(tmp_path, cache_mem_max=0)

    expected = first.process(mic, ref)
    assert first._backend.calls == 1
    assert not first._cache_mem

    second = _frontend(tmp_path, cache_mem_max=0)
    actual = second.process(mic, ref)
    assert np.array_equal(actual, expected)
    assert second._backend.calls == 0
    assert not second._cache_mem


@pytest.mark.parametrize(
    "bad_array",
    [
        np.zeros((2, 2), dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.array([0.0, np.nan, 0.0], dtype=np.float32),
        np.array([0.0, np.inf, 0.0], dtype=np.float32),
    ],
)
def test_invalid_disk_cache_is_recomputed(tmp_path, bad_array):
    mic = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    ref = np.zeros_like(mic)
    frontend = _frontend(tmp_path)
    key = frontend._hash_inputs(mic, ref)
    np.save(tmp_path / f"{key}.npy", bad_array)

    result = frontend.process(mic, ref)

    assert frontend._backend.calls == 1
    assert result.ndim == 1
    assert len(result) == len(mic)
    assert np.isfinite(result).all()
    assert np.array_equal(np.load(tmp_path / f"{key}.npy"), result)


def test_cache_key_includes_configuration_and_version(tmp_path):
    mic = np.array([0.1, 0.2], dtype=np.float32)
    ref = np.zeros_like(mic)
    first = _frontend(tmp_path, sample_rate=16000)
    changed_rate = _frontend(tmp_path, sample_rate=8000)
    changed_config = _frontend(tmp_path, sample_rate=16000)
    changed_config._cache_key_config["N"] = 99

    key = first._hash_inputs(mic, ref)
    assert key != changed_rate._hash_inputs(mic, ref)
    assert key != changed_config._hash_inputs(mic, ref)
    previous_version = first._CACHE_KEY_VERSION
    first._CACHE_KEY_VERSION += 1
    try:
        assert key != first._hash_inputs(mic, ref)
    finally:
        first._CACHE_KEY_VERSION = previous_version


@pytest.mark.parametrize(
    "cfg",
    [
        {"backend": "bad"},
        {"backend": 1},
        {"backend": "speex", "type": "pfdkf"},
        {"frame_size": 0},
        {"frame_size": 256.5},
        {"frame_size": "256"},
        {"filter_length": 0},
        {"bulk_delay": 1},
        {"cache_mem_max": -1},
        {"cache_dir": 1},
        {"gpu_batch": "false"},
        {"use_res": 1},
        {"enable_tdc": True},
        {"tdc_max_delay_ms": 500.0},
        {"tdc_win_seconds": 2.0},
        {"A": 0.0},
        {"A": "0.9"},
        {"P_initial": 0},
        {"keep_m_gate": 1.1},
        {"device": "mps"},
        {"device": 0},
        {"backend": "pfdkf", "M": 1024, "hop": 256},
        {"backend": "speex", "gpu_batch": True},
    ],
)
def test_config_and_frontend_fail_fast(cfg):
    with pytest.raises(ValueError):
        LinearAECConfig.from_dict(cfg)
    with pytest.raises(ValueError):
        LinearAECFrontend(cfg)


@pytest.mark.parametrize("sample_rate", [True, 16000.0, "16000", 0, -1])
def test_frontend_sample_rate_requires_exact_positive_int(sample_rate):
    with pytest.raises(ValueError, match="sample_rate must be an exact int > 0"):
        LinearAECFrontend({"backend": None}, sample_rate=sample_rate)


def test_speex_helper_cache_is_content_addressed_and_outside_source_tree(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    first_so, first_lock = linear_aec._speex_batch_cache_paths()
    monkeypatch.setattr(
        linear_aec,
        "_SPEEX_BATCH_C_SOURCE",
        linear_aec._SPEEX_BATCH_C_SOURCE + "\n/* cache identity change */\n",
    )
    changed_so, changed_lock = linear_aec._speex_batch_cache_paths()

    source_dir = linear_aec.os.path.dirname(linear_aec.__file__)
    assert linear_aec.os.path.commonpath([first_so, source_dir]) != source_dir
    assert first_so.startswith(str(tmp_path))
    assert first_so.endswith(".so")
    assert first_lock.endswith(".lock")
    assert (first_so, first_lock) != (changed_so, changed_lock)


def test_speex_helper_build_uses_linux_lock_and_atomic_replace(tmp_path, monkeypatch):
    import fcntl

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(linear_aec.platform, "system", lambda: "Linux")
    # 该模块全局在构建成功后会保留已加载的库；登记恢复点，避免跨测试泄漏。
    monkeypatch.setattr(linear_aec, "_c_batch_lib", None)
    locks = []
    replaces = []
    commands = []
    real_replace = linear_aec.os.replace

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        output_path = cmd[cmd.index("-o") + 1]
        with open(output_path, "wb") as compiled:
            compiled.write(b"test shared object")
        return SimpleNamespace(returncode=0, stderr="")

    def fake_load(path):
        linear_aec._c_batch_lib = object()
        return True

    def record_lock(fd, operation):
        locks.append((fd, operation))

    def record_replace(src, dst):
        replaces.append((src, dst))
        real_replace(src, dst)

    monkeypatch.setattr(linear_aec.subprocess, "run", fake_run)
    monkeypatch.setattr(linear_aec, "_load_speex_batch", fake_load)
    monkeypatch.setattr(fcntl, "flock", record_lock)
    monkeypatch.setattr(linear_aec.os, "replace", record_replace)

    assert linear_aec._compile_speex_batch() is True
    so_path, lock_path = linear_aec._speex_batch_cache_paths()
    assert locks and locks[0][1] == fcntl.LOCK_EX
    assert replaces and replaces[0][1] == so_path
    assert commands
    assert commands[0][: len(linear_aec._SPEEX_BATCH_BUILD_FLAGS)] == list(
        linear_aec._SPEEX_BATCH_BUILD_FLAGS
    )
    assert all(flag in commands[0] for flag in linear_aec._SPEEX_BATCH_LINK_FLAGS)
    assert linear_aec.os.path.isfile(so_path)
    assert linear_aec.os.path.isfile(lock_path)
    assert not list(tmp_path.rglob(".speex_batch-*.tmp"))


def test_speex_helper_rebuilds_when_artifact_digest_mismatches(tmp_path, monkeypatch):
    """cache 内 .so 被替换后必须重建，而不是直接加载未验证二进制。"""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(linear_aec.platform, "system", lambda: "Linux")
    # 构建成功会写回模块全局；登记恢复点，避免跨测试泄漏。
    monkeypatch.setattr(linear_aec, "_c_batch_lib", None)
    loaded: list[str] = []

    def fake_run(cmd, **kwargs):
        output_path = cmd[cmd.index("-o") + 1]
        with open(output_path, "wb") as compiled:
            compiled.write(b"compiled-once")
        return SimpleNamespace(returncode=0, stderr="")

    def fake_load(path):
        loaded.append(path)
        linear_aec._c_batch_lib = object()
        return True

    monkeypatch.setattr(linear_aec.subprocess, "run", fake_run)
    monkeypatch.setattr(linear_aec, "_load_speex_batch", fake_load)

    assert linear_aec._compile_speex_batch() is True
    so_path, _ = linear_aec._speex_batch_cache_paths()
    digest_path = linear_aec._speex_batch_digest_path(so_path)
    assert linear_aec.os.path.isfile(digest_path)

    with open(so_path, "wb") as artifact:
        artifact.write(b"tampered")
    loaded.clear()

    assert linear_aec._compile_speex_batch() is True
    assert loaded == [so_path]
    with open(so_path, "rb") as artifact:
        assert artifact.read() == b"compiled-once"


def test_cache_key_includes_backend_implementation_identity(tmp_path):
    mic = np.array([0.1, 0.2], dtype=np.float32)
    ref = np.zeros_like(mic)
    frontend = _frontend(tmp_path)
    key = frontend._hash_inputs(mic, ref)

    frontend._backend_identity = {"backend": None, "module": "changed"}

    assert frontend._hash_inputs(mic, ref) != key


def test_speex_helper_build_failure_keeps_ctypes_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(linear_aec.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        linear_aec.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="compile failed"),
    )
    monkeypatch.setattr(linear_aec, "_c_batch_lib", object())

    assert linear_aec._compile_speex_batch() is False
    so_path, _ = linear_aec._speex_batch_cache_paths()
    assert linear_aec._c_batch_lib is None
    assert not linear_aec.os.path.exists(so_path)
    assert not list(tmp_path.rglob(".speex_batch-*.tmp"))


def test_none_backend_disables_frontend():
    config = LinearAECConfig.from_dict({"backend": "none"})
    frontend = LinearAECFrontend({"backend": "none"})
    mic = np.array([0.1, -0.2], dtype=np.float32)

    assert config.backend is None
    assert frontend.enabled is False
    assert np.array_equal(frontend.process(mic, np.zeros_like(mic)), np.zeros_like(mic))
    assert not hasattr(frontend, "apply_tdc")
