"""Step3 MOS fleet cancellation ordering tests."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import audio_filter_pipeline.processors.stages_v3.mos_gpu_job as mos_gpu_job
import audio_filter_pipeline.processors.stages_v3.mos_pool as mos_pool


pytestmark = pytest.mark.unit


class _Process:
    def __init__(self, pid=1234):
        self.pid = pid
        self.waited = []
        self._done = False
        self.stdin = self.stdout = self.stderr = None

    def poll(self):
        return 0 if self._done else None

    def wait(self, timeout=None):
        self.waited.append(timeout)
        self._done = True
        return 0


class _Fleet:
    def __init__(self):
        self._shutdown_timeout = 0.1


def _slot():
    return mos_pool._FleetWorkerSlot(_Fleet(), 0)


def test_slot_late_spawn_after_abort_is_signalled_and_reaped(monkeypatch):
    slot = _slot()
    proc = _Process()
    signals = []
    monkeypatch.setattr(mos_pool.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    slot.abort()

    assert slot._register_spawned(proc) is False
    assert signals == [(proc.pid, mos_pool.signal.SIGTERM)]
    assert proc.waited == [0.5]
    assert slot._proc is None


def test_slot_abort_before_popen_return_reaps_late_worker(monkeypatch):
    class SpawnFleet(_Fleet):
        _idle_timeout = 1.0
        _launcher_directory = staticmethod(lambda: "/tmp")

    slot = mos_pool._FleetWorkerSlot(SpawnFleet(), 0)
    proc = _Process()
    popen_entered = threading.Event()
    allow_popen_return = threading.Event()
    errors = []
    signals = []

    def popen(*_args, **_kwargs):
        popen_entered.set()
        assert allow_popen_return.wait(timeout=1.0)
        return proc

    slot._fleet._popen = popen
    monkeypatch.setattr(mos_pool.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    thread = threading.Thread(target=lambda: _capture(errors, slot._spawn, "/tmp"))
    thread.start()
    assert popen_entered.wait(timeout=1.0)
    slot.abort()
    allow_popen_return.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert signals == [(proc.pid, mos_pool.signal.SIGTERM)]
    assert slot._proc is None


def _capture(errors, function, *args):
    try:
        function(*args)
    except BaseException as error:
        errors.append(error)


def test_slot_abort_after_spawn_signals_registered_process(monkeypatch):
    slot = _slot()
    proc = _Process()
    signals = []
    monkeypatch.setattr(mos_pool.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    assert slot._register_spawned(proc) is True
    slot.abort()

    assert signals == [(proc.pid, mos_pool.signal.SIGTERM)]
    assert slot._proc is proc


def test_fleet_request_abort_records_first_error_and_broadcasts_without_join(monkeypatch):
    fleet = object.__new__(mos_pool.Step3MosWorkerFleet)
    fleet._state_lock = threading.RLock()
    fleet._broken = None
    calls = []
    fleet._slots = {
        0: SimpleNamespace(abort=lambda: calls.append(0)),
        1: SimpleNamespace(abort=lambda: calls.append(1)),
    }
    joined = []
    monkeypatch.setattr(fleet, "_shutdown_slots", lambda error: joined.append(error))
    first = RuntimeError("first")

    fleet.request_abort(first)
    fleet.request_abort(RuntimeError("second"))

    assert fleet.root_cause() is first
    assert calls == [0, 1]
    assert joined == []
    fleet.join_aborted(first)
    assert joined == [first]


@pytest.mark.parametrize("error", [KeyboardInterrupt(), GeneratorExit()])
def test_job_cancel_preserves_baseexception_and_calls_abort_once(error):
    aborts = []
    settings = mos_gpu_job._GpuScoringJobSettings(
        gpu_ids=(0,), dataset_id="alpha", verified_models=("scoreq",),
        options_by_model={
            "scoreq": None, "urgent_pk": None, "dnsmos": None,
            "nisqa": None, "utmos": None,
        },
        asset_provenance=None, worker_idle_timeout_seconds=1.0,
        worker_hard_timeout_seconds=None, worker_stderr_tail_bytes=1,
        popen=lambda *_args, **_kwargs: None, progress_callback=lambda _rows: None,
        request_abort=aborts.append,
    )
    job = mos_gpu_job._OwnedGpuScoringJob(settings)

    job._cancel(error)
    job._cancel(RuntimeError("secondary"))

    with pytest.raises(type(error)) as raised:
        job._raise_if_cancelled()
    assert raised.value is error
    assert aborts == [error]


def test_simultaneous_slot_failures_keep_one_root_cause_and_one_broadcast():
    fleet = object.__new__(mos_pool.Step3MosWorkerFleet)
    fleet._state_lock = threading.RLock()
    fleet._broken = None
    calls = []
    fleet._slots = {
        0: SimpleNamespace(abort=lambda: calls.append(0)),
        1: SimpleNamespace(abort=lambda: calls.append(1)),
    }
    barrier = threading.Barrier(2)
    errors = [RuntimeError("slot 0"), RuntimeError("slot 1")]
    threads = [
        threading.Thread(
            target=lambda error=error: (barrier.wait(), fleet.request_abort(error)),
        )
        for error in errors
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)

    assert all(not thread.is_alive() for thread in threads)
    assert fleet.root_cause() in errors
    assert sorted(calls) == [0, 1]


def test_repeat_close_does_not_repeat_successful_shutdown(monkeypatch):
    fleet = object.__new__(mos_pool.Step3MosWorkerFleet)
    fleet._close_lock = threading.Lock()
    fleet._state_lock = threading.RLock()
    fleet._closed = False
    fleet._closing = False
    calls = []
    monkeypatch.setattr(fleet, "_shutdown_slots", lambda: calls.append(True))

    fleet.close()
    fleet.close()

    assert calls == [True]
    assert fleet._closed is True


def test_slot_failure_broadcasts_after_releasing_its_request_lock(monkeypatch):
    class Fleet(_Fleet):
        def _check_usable(self):
            return None

        def request_abort(self, error):
            assert error.args == ("control failed",)
            acquired = slot._lock.acquire(blocking=False)
            assert acquired
            if acquired:
                slot._lock.release()

    class Input:
        def write(self, _payload):
            return None

        def flush(self):
            return None

    slot = mos_pool._FleetWorkerSlot(Fleet(), 0)
    monkeypatch.setattr(slot, "_spawn", lambda _cwd: SimpleNamespace(stdin=Input()))
    monkeypatch.setattr(
        slot, "_read_control",
        lambda _expected: (_ for _ in ()).throw(RuntimeError("control failed")),
    )

    with pytest.raises(RuntimeError, match="control failed"):
        slot.run_request("alpha", "/tmp/request.json", "/tmp")


def test_slot_signal_failure_does_not_prevent_other_slot_abort(monkeypatch):
    fleet = object.__new__(mos_pool.Step3MosWorkerFleet)
    fleet._state_lock = threading.RLock()
    fleet._broken = None
    first = SimpleNamespace(abort=lambda: (_ for _ in ()).throw(RuntimeError("signal failed")))
    called = []
    fleet._slots = {0: first, 1: SimpleNamespace(abort=lambda: called.append(True))}
    error = RuntimeError("root")

    fleet.request_abort(error)

    assert called == [True]
    assert any("signal failed" in note for note in error.__notes__)
