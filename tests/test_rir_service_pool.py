"""常驻 gpuRIR 生成池、并发派发与 RIR 缓存的回归测试。"""

from __future__ import annotations

import hashlib
import json
import queue
import socket
import threading
import time
from contextlib import contextmanager

import numpy as np
import pytest

from core.data.simulation import rir_service as service

pytestmark = pytest.mark.unit


def _plan(index=0):
    return {
        "room_dim": [3.0, 3.0, 2.5],
        "source_loc": [[1.0, 1.0, 1.2]],
        "mic_loc": [1.5, 1.5, 1.2],
        "num_samples": 8,
        "sample_rate": 8000,
        "rt60": 0.2,
        "diffuse_db": 15.0,
        "sound_speed": 340.0,
        "group_uid": f"pool-{index}",
    }


class _RecordingGenerator:
    """在 generate 内阻塞并记录并发峰值的 fake 生成器。"""

    def __init__(self, delay, size=2):
        self.delay = delay
        self.size = size
        self.peak = 0
        self.calls = 0
        self._active = 0
        self._lock = threading.Lock()

    def generate(self, plan, track_kind, expected, timeout, requested_timeout_sec=None):
        with self._lock:
            self.calls += 1
            self._active += 1
            self.peak = max(self.peak, self._active)
        try:
            time.sleep(self.delay)
        finally:
            with self._lock:
                self._active -= 1
        wave = np.ones(expected, dtype="<f4")
        facts = {
            **plan,
            "track_kind": track_kind,
            "rir_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
        }
        return {"full": (wave, facts)}


@contextmanager
def _server(tmp_path, generator):
    address = str(tmp_path / "pool.sock")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(address)
        listener.listen(8)
        failures = []

        def run():
            try:
                service._serve_requests(
                    listener, "token", None, 4, 30.0, generation_pool=generator
                )
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            yield service.RIRClient(address, "token", 30.0)
        finally:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(2)
                sock.connect(address)
                service._send(sock, json.dumps({"token": "token", "action": "close"}).encode())
            thread.join(5)
            assert not thread.is_alive(), "RIR pool fixture failed to stop"
            assert not failures, failures


def test_concurrent_requests_occupy_separate_pool_slots(tmp_path):
    generator = _RecordingGenerator(delay=0.5, size=2)
    with _server(tmp_path, generator) as client:
        results = []
        lock = threading.Lock()

        def call(index):
            out = client.generate(_plan(index))
            with lock:
                results.append(out)

        threads = [threading.Thread(target=call, args=(index,)) for index in range(2)]
        started = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed = time.monotonic() - started

    assert len(results) == 2
    # 峰值 2 证明两个请求并行占用两个 slot；串行实现只会有峰值 1。
    assert generator.peak == 2
    assert elapsed < 0.95


def test_concurrent_same_key_requests_use_single_flight(tmp_path):
    generator = _RecordingGenerator(delay=0.5, size=2)
    with _server(tmp_path, generator) as client:
        results = []
        threads = [
            threading.Thread(target=lambda: results.append(client.generate(_plan(0))))
            for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert len(results) == 2
    assert generator.calls == 1


def test_rir_cache_take_removes_entry(monkeypatch):
    cache = service._RIRCache(4)
    entry = {"full": (np.ones((1, 4), dtype="<f4"), {})}
    cache.put("key", entry)
    assert cache.take("key") is entry
    assert cache.take("key") is None


def test_rir_cache_evicts_oldest_entry_over_entry_limit(monkeypatch):
    cache = service._RIRCache(2)
    for key in ("a", "b"):
        cache.put(key, {"full": (np.ones((1, 4), dtype="<f4"), {})})
    cache.put("c", {"full": (np.ones((1, 4), dtype="<f4"), {})})
    assert cache.take("a") is None
    assert cache.take("b") is not None
    assert cache.take("c") is not None


def test_rir_cache_replacing_key_replaces_byte_accounting():
    cache = service._RIRCache(4)
    first = {"full": (np.ones((1, 4), dtype="<f4"), {})}
    second = {"full": (np.ones((1, 8), dtype="<f4"), {})}
    cache.put("same", first)
    cache.put("same", second)
    assert cache._bytes == service._RIRCache._size(second)
    assert cache.take("same") is second
    assert cache._bytes == 0


def test_generation_pool_queue_wait_consumes_generation_deadline(monkeypatch):
    """等待空闲 slot 不能把一次请求的总等待时间延长一个完整 timeout。"""
    pool = service._GenerationPool.__new__(service._GenerationPool)
    pool._size = 1
    pool._closed = False
    import queue as _queue

    pool._available = _queue.Queue()
    with pytest.raises(service.OnlineRIRError, match="worker"):
        pool.generate(_plan(), "full", (1, 8), 0.01)


def test_rir_cache_rejects_entry_over_byte_budget(monkeypatch):
    monkeypatch.setattr(service, "_CACHE_BYTES", 8)
    cache = service._RIRCache(4)
    cache.put("big", {"full": (np.ones((1, 8), dtype="<f4"), {})})
    assert cache.take("big") is None


def test_rir_cache_byte_budget_reads_environment(monkeypatch):
    monkeypatch.setenv("USE_TRAIN_GPURIR_CACHE_BYTES", "16")
    assert service._resolve_cache_bytes() == 16
    monkeypatch.setenv("USE_TRAIN_GPURIR_CACHE_BYTES", "0")
    assert service._resolve_cache_bytes() == 0


def test_generation_pool_reports_worker_shortage_without_hanging(monkeypatch):
    """池容量缩到 0 时必须在 deadline 内 fail-closed，不能无限阻塞。"""
    pool = service._GenerationPool.__new__(service._GenerationPool)
    pool._size = 1
    pool._closed = False
    import queue as _queue

    pool._available = _queue.Queue()
    with pytest.raises(service.OnlineRIRError, match="worker"):
        pool.generate(_plan(), "full", (1, 8), 0.05)


class _CorruptWorkerConnection:
    """只返回一组可控的 worker IPC 帧。"""

    def __init__(self, header, payload=None):
        self._frames = [header] if payload is None else [header, payload]

    def send_bytes(self, _payload):
        return None

    def poll(self, _timeout):
        return bool(self._frames)

    def recv_bytes(self):
        return self._frames.pop(0)

    def close(self):
        return None


@pytest.mark.parametrize(
    "header",
    [b"not-json", b"[]"],
    ids=["invalid-json", "non-object"],
)
def test_generation_pool_recycles_worker_on_protocol_corruption(monkeypatch, header):
    """协议污染不能把同一 slot 放回池中继续消费残留字节。"""
    pool = service._GenerationPool.__new__(service._GenerationPool)
    pool._size = 1
    pool._closed = False
    pool._available = queue.Queue()
    slot = service._WorkerSlot(process=object(), conn=_CorruptWorkerConnection(header))
    pool._available.put(slot)
    recycled = []
    monkeypatch.setattr(pool, "_recycle", lambda item: recycled.append(item))

    with pytest.raises(service.OnlineRIRError, match="header"):
        pool.generate(_plan(), "full", (1, 8), 1.0)

    assert recycled == [slot]
    assert pool._available.empty()


def test_generation_pool_recycles_worker_on_cuda_backend_error(monkeypatch):
    """CUDA context 发生致命错误后不能把同一 worker 放回池中。"""
    pool = service._GenerationPool.__new__(service._GenerationPool)
    pool._size = 1
    pool._closed = False
    pool._available = queue.Queue()
    slot = service._WorkerSlot(
        process=object(),
        conn=_CorruptWorkerConnection(
            json.dumps({"error": "RuntimeError: CUDA out of memory"}).encode()
        ),
    )
    pool._available.put(slot)
    recycled = []
    monkeypatch.setattr(pool, "_recycle", lambda item: recycled.append(item))

    with pytest.raises(service.OnlineRIRError, match="CUDA out of memory"):
        pool.generate(_plan(), "full", (1, 8), 1.0)

    assert recycled == [slot]
    assert pool._available.empty()


def test_generation_pool_concurrent_recycle_keeps_all_replacements(monkeypatch):
    """多个 handler 同时回收 slot 时，池状态不能丢失 replacement。"""
    pool = service._GenerationPool.__new__(service._GenerationPool)
    pool._size = 2
    pool._closed = False
    pool._state_lock = threading.Lock()
    pool._available = queue.Queue()
    slots = [
        service._WorkerSlot(process=object(), conn=_CorruptWorkerConnection(b"[]"))
        for _ in range(2)
    ]
    pool._slots = list(slots)
    replacements = iter(
        service._WorkerSlot(process=object(), conn=_CorruptWorkerConnection(b"[]"))
        for _ in range(2)
    )
    monkeypatch.setattr(service, "_stop_worker", lambda _process: None)
    monkeypatch.setattr(pool, "_spawn", lambda: next(replacements))

    threads = [threading.Thread(target=pool._recycle, args=(slot,)) for slot in slots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(pool._slots) == 2
    assert pool._available.qsize() == 2
