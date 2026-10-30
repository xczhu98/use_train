"""Audio exporter 的 bounded source-session 与 I/O 计数边界。"""

from __future__ import annotations

import threading
from collections import OrderedDict, Counter
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, Iterator

from .source_io import VerifiedSourceSession, open_verified_source_session

_SOURCE_SESSION_FIELDS = (
    "input_root", "source_relative_path", "source_size_bytes", "source_sha256",
    "sample_rate", "frames", "channels", "format", "subtype", "duration_sec",
)


def _add_exception_note(error: BaseException, note: str) -> None:
    """在不同 Python 版本上尽力保留清理诊断。"""
    try:
        add_note = getattr(error, "add_note", None)
        if callable(add_note):
            add_note(note)
            return
        notes = list(getattr(error, "__notes__", ()))
        notes.append(note)
        error.__notes__ = notes
    except BaseException:
        pass


def source_session_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """返回不含 segment interval 的 immutable source identity。"""
    try:
        return tuple(row[field] for field in _SOURCE_SESSION_FIELDS)
    except KeyError as exc:
        raise ValueError(f"source session row is missing {exc.args[0]}") from exc


class ExportIOCounters:
    """线程安全的 invocation 计数器；不参与安全决策。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: Counter[str] = Counter()
        self._phases: Counter[str] = Counter()

    def add(self, **values: int) -> None:
        with self._lock:
            self._values.update(values)

    def record_output_verify(
        self, phase: str, file_size: int, read_passes: int = 2,
    ) -> None:
        # read_passes 记录实际完整读取遍数：快照单遍路径为 1，流式 decode +
        # 分块 hash 路径为 2（libsndfile 完整解码 + raw SHA 完整读取）。
        with self._lock:
            self._values["output_verify_calls"] += 1
            self._values["output_full_read_passes"] += read_passes
            self._values["output_estimated_full_read_bytes"] += read_passes * file_size
            self._phases[phase] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = dict(self._values)
            result["output_verify_calls_by_phase"] = dict(sorted(self._phases.items()))
            return result


class VerifiedSourceSessionPool:
    """按 immutable source identity 复用且限制 held session 数。"""

    def __init__(self, max_sessions: int, counters: ExportIOCounters) -> None:
        if type(max_sessions) is not int or max_sessions <= 0:
            raise ValueError("max_sessions must be a positive Python integer")
        self._max_sessions = max_sessions
        self._counters = counters
        self._condition = threading.Condition()
        self._sessions: OrderedDict[tuple[Any, ...], VerifiedSourceSession] = OrderedDict()
        self._active: Counter[tuple[Any, ...]] = Counter()
        self._opening: set[tuple[Any, ...]] = set()
        self._closed = False

    def _evict_idle_unlocked(self) -> VerifiedSourceSession | None:
        for identity, session in tuple(self._sessions.items()):
            if self._active[identity] == 0:
                del self._sessions[identity]
                self._active.pop(identity, None)
                return session
        return None

    @contextmanager
    def acquire(self, row: Mapping[str, Any]) -> Iterator[VerifiedSourceSession]:
        identity = source_session_identity(row)
        session = None
        evicted = None
        must_open = False
        while session is None and not must_open:
            with self._condition:
                if self._closed:
                    raise RuntimeError("verified source session pool is closed")
                session = self._sessions.get(identity)
                if session is not None:
                    self._active[identity] += 1
                    self._sessions.move_to_end(identity)
                    break
                if identity in self._opening:
                    self._condition.wait()
                    continue
                if len(self._sessions) + len(self._opening) >= self._max_sessions:
                    evicted = self._evict_idle_unlocked()
                    if evicted is None:
                        self._condition.wait()
                        continue
                self._opening.add(identity)
                must_open = True
        if evicted is not None:
            counts = evicted.io_counts
            self._counters.add(
                interval_decode_calls=counts["interval_decode_calls"],
                interval_decode_frames=counts["interval_decode_frames"],
                interval_decode_bytes=counts["interval_decode_bytes"],
            )
            try:
                evicted.__exit__(None, None, None)
            except BaseException:
                with self._condition:
                    self._opening.discard(identity)
                    self._condition.notify_all()
                raise
            self._counters.add(source_sessions_closed=1)
        if must_open:
            try:
                opened = open_verified_source_session(row)
                session = opened.__enter__()
            except BaseException:
                with self._condition:
                    self._opening.discard(identity)
                    self._condition.notify_all()
                raise
            with self._condition:
                if self._closed:
                    self._opening.discard(identity)
                    self._condition.notify_all()
                    close_now = True
                else:
                    self._sessions[identity] = session
                    self._active[identity] = 1
                    self._opening.discard(identity)
                    self._condition.notify_all()
                    close_now = False
            if close_now:
                session.__exit__(None, None, None)
                raise RuntimeError("verified source session pool closed while opening")
            counts = session.io_counts
            self._counters.add(
                source_sessions_opened=1,
                source_full_sha_reads=counts["source_full_sha_reads"],
                source_full_sha_bytes=counts["source_full_sha_bytes"],
            )
        assert session is not None
        try:
            yield session
        finally:
            with self._condition:
                self._active[identity] -= 1
                self._condition.notify_all()

    def close(self, primary: BaseException | None = None) -> None:
        with self._condition:
            self._closed = True
            # 唤醒阻塞在 acquire() 等待中的线程，使其重新检查 _closed 并退出，
            # 否则 close 返回后将无人再 notify，acquirer 永久阻塞。
            self._condition.notify_all()
            while self._opening or any(self._active.values()):
                self._condition.wait()
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
            self._active.clear()
        first_error = None
        for session in sessions:
            counts = session.io_counts
            self._counters.add(
                interval_decode_calls=counts["interval_decode_calls"],
                interval_decode_frames=counts["interval_decode_frames"],
                interval_decode_bytes=counts["interval_decode_bytes"],
            )
            try:
                session.__exit__(None, None, None)
                self._counters.add(source_sessions_closed=1)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    _add_exception_note(
                        first_error,
                        f"additional source session cleanup failure: {exc!r}",
                    )
        if first_error is not None:
            if primary is not None:
                _add_exception_note(
                    primary,
                    f"source session cleanup failed: {first_error!r}",
                )
                return
            raise first_error
