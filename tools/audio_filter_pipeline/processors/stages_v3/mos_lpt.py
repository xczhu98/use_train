"""Step3 GPU source-group weighted LPT 的磁盘化规划器。"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
from collections.abc import Iterable, Sequence
from typing import Any

from ...utils.strict_jsonl import add_cleanup_note, encode_json_line

MAX_DISTINCT_MODEL_TUPLES = 32
STEP3_JSONL_MAX_FILE_BYTES = 512 * 1024 * 1024
STEP3_SPOOL_MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
MAX_WORKER_INPUT_SUBSHARDS = (
    MAX_DISTINCT_MODEL_TUPLES
    + STEP3_SPOOL_MAX_TOTAL_BYTES // STEP3_JSONL_MAX_FILE_BYTES
)
_LPT_BATCH_MAX_ROWS = 512
_LPT_BATCH_MAX_BYTES = 32 * 1024 * 1024
_MAX_UINT64 = (1 << 64) - 1


def _uint64_parts(value: int) -> tuple[int, int]:
    if type(value) is not int or not 0 <= value <= _MAX_UINT64:
        raise ValueError("LPT weight must be uint64")
    return value >> 32, value & 0xFFFFFFFF


def _uint64_value(high: int, low: int) -> int:
    if (
        type(high) is not int
        or type(low) is not int
        or not 0 <= high <= 0xFFFFFFFF
        or not 0 <= low <= 0xFFFFFFFF
    ):
        raise ValueError("stored LPT weight is invalid")
    return (high << 32) | low


def _is_unique_constraint(error: sqlite3.IntegrityError) -> bool:
    error_name = getattr(error, "sqlite_errorname", None)
    if error_name in {"SQLITE_CONSTRAINT_PRIMARYKEY", "SQLITE_CONSTRAINT_UNIQUE"}:
        return True
    return str(error).startswith("UNIQUE constraint failed:")


class _DiskBackedSqliteState:
    """共享 SQLite terminal-state 与穷尽关闭语义。"""

    _connection: sqlite3.Connection | None
    _poisoned: bool

    def _ensure_usable(self, label: str) -> sqlite3.Connection:
        if self._poisoned:
            raise RuntimeError(f"{label} is poisoned")
        connection = self._connection
        if connection is None:
            raise RuntimeError(f"{label} is closed")
        return connection

    def _poison(self) -> None:
        self._poisoned = True

    def close(self) -> None:
        connection = self._connection
        if connection is None:
            return
        try:
            connection.close()
        except BaseException:
            self._poison()
            raise
        self._connection = None

    def __enter__(self) -> "_DiskBackedSqliteState":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        primary: BaseException | None,
        traceback: Any,
    ) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if isinstance(primary, GeneratorExit) or primary is None:
                raise
            add_cleanup_note(primary, "SQLite ownership cleanup", cleanup_error)


class _DiskBackedLptPlanner(_DiskBackedSqliteState):
    """用 owned SQLite 聚合 source group，并以有界批次验证两遍扫描。"""

    def __init__(self, path: str, total_rows: int, worker_limit: int) -> None:
        if type(path) is not str or not path:
            raise ValueError("LPT SQLite path must be nonempty")
        if type(total_rows) is not int or total_rows <= 0:
            raise ValueError("LPT total_rows must be positive")
        if type(worker_limit) is not int or worker_limit <= 0:
            raise ValueError("LPT worker_limit must be positive")
        self._total_rows = total_rows
        self._worker_limit = worker_limit
        self._connection = sqlite3.connect(path)
        self._poisoned = False
        self._pending: list[tuple[int, bytes, int, bytes, int]] = []
        self._pending_bytes = 0
        self._discovery_order = 0
        self._finalized = False
        connection = self._connection
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute("PRAGMA mmap_size=0")
            connection.executescript(
                """
                CREATE TABLE source_groups (
                    group_id INTEGER PRIMARY KEY,
                    identity BLOB NOT NULL UNIQUE,
                    discovery_order INTEGER NOT NULL UNIQUE,
                    weight_hi INTEGER NOT NULL,
                    weight_lo INTEGER NOT NULL,
                    worker INTEGER
                );
                CREATE TABLE task_assignments (
                    ordinal INTEGER PRIMARY KEY,
                    group_id INTEGER NOT NULL,
                    payload_sha256 BLOB NOT NULL,
                    FOREIGN KEY(group_id) REFERENCES source_groups(group_id)
                );
                """
            )
        except BaseException as primary:
            self._poison()
            self._connection = None
            try:
                connection.close()
            except BaseException as cleanup_error:
                add_cleanup_note(primary, "LPT SQLite initialization", cleanup_error)
            raise

    @staticmethod
    def identity_bytes(
        models: Sequence[str], source_identity: Sequence[Any],
    ) -> bytes:
        return encode_json_line(
            {"models": list(models), "source_identity": list(source_identity)},
            "MOS LPT source identity",
            max_line_bytes=8 * 1024 * 1024,
        )

    def add(
        self,
        *,
        ordinal: int,
        identity: bytes,
        weight: int,
        payload: bytes,
    ) -> None:
        self._ensure_usable("LPT planner")
        if self._finalized:
            raise RuntimeError("LPT planner ingestion is already finalized")
        if type(ordinal) is not int or not 0 <= ordinal < self._total_rows:
            raise ValueError("Step3 tuple task ordinal is invalid")
        if type(identity) is not bytes or not identity:
            raise ValueError("LPT source identity must be nonempty bytes")
        if type(payload) is not bytes or not payload:
            raise ValueError("LPT task payload must be nonempty bytes")
        _uint64_parts(weight)
        record_bytes = len(identity) + len(payload) + 96
        if record_bytes > _LPT_BATCH_MAX_BYTES:
            raise ValueError("LPT ingestion row exceeds bounded batch byte limit")
        if self._pending and (
            len(self._pending) >= _LPT_BATCH_MAX_ROWS
            or self._pending_bytes + record_bytes > _LPT_BATCH_MAX_BYTES
        ):
            self._flush()
        self._pending.append((
            ordinal,
            identity,
            weight,
            hashlib.sha256(payload).digest(),
            self._discovery_order,
        ))
        self._pending_bytes += record_bytes
        self._discovery_order += 1

    def _flush(self) -> None:
        connection = self._ensure_usable("LPT planner")
        if not self._pending:
            return
        try:
            with connection:
                for ordinal, identity, weight, digest, discovery_order in self._pending:
                    connection.execute(
                        "INSERT OR IGNORE INTO source_groups "
                        "(identity, discovery_order, weight_hi, weight_lo, worker) "
                        "VALUES (?, ?, 0, 0, NULL)",
                        (identity, discovery_order),
                    )
                    stored = connection.execute(
                        "SELECT group_id, weight_hi, weight_lo FROM source_groups "
                        "WHERE identity = ?",
                        (identity,),
                    ).fetchone()
                    if stored is None:
                        raise RuntimeError("LPT source group insert was not observable")
                    group_id, weight_hi, weight_lo = stored
                    updated = _uint64_value(weight_hi, weight_lo) + weight
                    if updated > _MAX_UINT64:
                        raise ValueError("MOS source group weight exceeds uint64")
                    updated_hi, updated_lo = _uint64_parts(updated)
                    connection.execute(
                        "UPDATE source_groups SET weight_hi = ?, weight_lo = ? "
                        "WHERE group_id = ?",
                        (updated_hi, updated_lo, group_id),
                    )
                    try:
                        connection.execute(
                            "INSERT INTO task_assignments "
                            "(ordinal, group_id, payload_sha256) VALUES (?, ?, ?)",
                            (ordinal, group_id, digest),
                        )
                    except sqlite3.IntegrityError as exc:
                        if _is_unique_constraint(exc):
                            raise ValueError(
                                "Step3 tuple task ordinal is invalid or duplicate"
                            ) from exc
                        raise
        except BaseException:
            self._poison()
            raise
        finally:
            self._pending.clear()
            self._pending_bytes = 0

    def finalize(self) -> int:
        self._ensure_usable("LPT planner")
        if self._finalized:
            raise RuntimeError("LPT planner is already finalized")
        try:
            self._flush()
            connection = self._ensure_usable("LPT planner")
            count, minimum, maximum = connection.execute(
                "SELECT COUNT(*), MIN(ordinal), MAX(ordinal) FROM task_assignments"
            ).fetchone()
            if (
                count != self._total_rows
                or minimum != 0
                or maximum != self._total_rows - 1
            ):
                raise ValueError("Step3 tuple task ordinals are incomplete")
            group_count = connection.execute(
                "SELECT COUNT(*) FROM source_groups"
            ).fetchone()[0]
            workers = min(self._worker_limit, group_count)
            if workers <= 0:
                raise ValueError("MOS LPT source groups are empty")
            loads = [0] * workers
            updates: list[tuple[int, int]] = []
            rows = connection.execute(
                "SELECT group_id, weight_hi, weight_lo FROM source_groups "
                "ORDER BY weight_hi DESC, weight_lo DESC, discovery_order ASC"
            )
            with connection:
                for group_id, weight_hi, weight_lo in rows:
                    worker = min(range(workers), key=lambda index: (loads[index], index))
                    loads[worker] += _uint64_value(weight_hi, weight_lo)
                    updates.append((worker, group_id))
                    if len(updates) >= _LPT_BATCH_MAX_ROWS:
                        connection.executemany(
                            "UPDATE source_groups SET worker = ? WHERE group_id = ?",
                            updates,
                        )
                        updates.clear()
                if updates:
                    connection.executemany(
                        "UPDATE source_groups SET worker = ? WHERE group_id = ?",
                        updates,
                    )
            self._finalized = True
            return workers
        except BaseException:
            self._poison()
            raise

    def assignments(
        self,
        records: Iterable[tuple[int, bytes, bytes]],
    ) -> list[int]:
        connection = self._ensure_usable("LPT planner")
        if not self._finalized:
            raise RuntimeError("LPT planner must be finalized before lookup")
        try:
            materialized: list[tuple[int, bytes, bytes]] = []
            size = 0
            for record in records:
                if (
                    not isinstance(record, tuple)
                    or len(record) != 3
                    or type(record[0]) is not int
                    or type(record[1]) is not bytes
                    or type(record[2]) is not bytes
                ):
                    raise ValueError("LPT lookup record is invalid")
                if len(materialized) >= _LPT_BATCH_MAX_ROWS:
                    raise ValueError("LPT lookup batch row count is invalid")
                record_size = len(record[1]) + len(record[2]) + 64
                if record_size > _LPT_BATCH_MAX_BYTES - size:
                    raise ValueError("LPT lookup batch exceeds byte limit")
                materialized.append(record)
                size += record_size
            if not materialized:
                raise ValueError("LPT lookup batch row count is invalid")
            ordinals = [record[0] for record in materialized]
            placeholders = ",".join("?" for _ in ordinals)
            rows = connection.execute(
                "SELECT tasks.ordinal, groups.identity, tasks.payload_sha256, groups.worker "
                "FROM task_assignments AS tasks JOIN source_groups AS groups "
                "ON groups.group_id = tasks.group_id "
                f"WHERE tasks.ordinal IN ({placeholders})",
                ordinals,
            ).fetchall()
            found = {
                ordinal: (identity, digest, worker)
                for ordinal, identity, digest, worker in rows
            }
            if len(found) != len(materialized):
                raise ValueError("LPT lookup task coverage is invalid")
            result = []
            for ordinal, identity, payload in materialized:
                stored_identity, stored_digest, worker = found[ordinal]
                if (
                    stored_identity != identity
                    or stored_digest != hashlib.sha256(payload).digest()
                    or type(worker) is not int
                    or not 0 <= worker < self._worker_limit
                ):
                    raise ValueError("Step3 tuple task changed between LPT scans")
                result.append(worker)
            return result
        except BaseException:
            self._poison()
            raise


class _DiskBackedOrdinalRegistry(_DiskBackedSqliteState):
    """以 bounded batch 验证 worker request 内 ordinal 全局唯一性。"""

    def __init__(self, path: str) -> None:
        if type(path) is not str or not path:
            raise ValueError("ordinal registry SQLite path must be nonempty")
        self._connection = sqlite3.connect(path)
        self._poisoned = False
        self._pending: list[tuple[int]] = []
        self._finalized = False
        connection = self._connection
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute("PRAGMA mmap_size=0")
            connection.execute(
                "CREATE TABLE ordinals (ordinal INTEGER PRIMARY KEY)"
            )
        except BaseException as primary:
            self._poison()
            self._connection = None
            try:
                connection.close()
            except BaseException as cleanup_error:
                add_cleanup_note(primary, "ordinal registry SQLite initialization", cleanup_error)
            raise

    def add(self, ordinal: int) -> None:
        self._ensure_usable("ordinal registry")
        if self._finalized:
            raise RuntimeError("ordinal registry is already finalized")
        if type(ordinal) is not int or not 0 <= ordinal <= (1 << 63) - 1:
            raise ValueError("worker request ordinal is invalid")
        if len(self._pending) >= _LPT_BATCH_MAX_ROWS:
            self._flush()
        self._pending.append((ordinal,))

    def _flush(self) -> None:
        connection = self._ensure_usable("ordinal registry")
        if not self._pending:
            return
        try:
            with connection:
                try:
                    connection.executemany(
                        "INSERT INTO ordinals (ordinal) VALUES (?)",
                        self._pending,
                    )
                except sqlite3.IntegrityError as exc:
                    if _is_unique_constraint(exc):
                        raise ValueError(
                            "worker request ordinals are not globally unique"
                        ) from exc
                    raise
        except BaseException:
            self._poison()
            raise
        finally:
            self._pending.clear()

    def finalize(self, expected_count: int) -> None:
        self._ensure_usable("ordinal registry")
        if self._finalized:
            raise RuntimeError("ordinal registry is already finalized")
        if type(expected_count) is not int or expected_count <= 0:
            raise ValueError("worker request ordinal count is invalid")
        try:
            self._flush()
            connection = self._ensure_usable("ordinal registry")
            count = connection.execute("SELECT COUNT(*) FROM ordinals").fetchone()[0]
            if count != expected_count:
                raise ValueError("worker request ordinal count is invalid")
            self._finalized = True
        except BaseException:
            self._poison()
            raise


# Step3 GPU worker 直接消费的磁盘化 ordinal 登记表：以公开别名声明跨模块
# 契约，worker 不再 reach-through 本模块的私有符号。
DiskBackedOrdinalRegistry = _DiskBackedOrdinalRegistry


class _DiskBackedResultStore(_DiskBackedSqliteState):
    """按 ordinal 保存已验证结果，避免结果归并持有线性数量的文件。"""

    def __init__(self, path: str, total_rows: int) -> None:
        if type(path) is not str or not path:
            raise ValueError("MOS result SQLite path must be nonempty")
        if type(total_rows) is not int or total_rows <= 0:
            raise ValueError("MOS result total_rows must be positive")
        self._total_rows = total_rows
        self._connection = sqlite3.connect(path)
        self._poisoned = False
        self._pending: list[tuple[int, bytes]] = []
        self._pending_bytes = 0
        self._finalized = False
        connection = self._connection
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute("PRAGMA mmap_size=0")
            connection.execute(
                "CREATE TABLE results (ordinal INTEGER PRIMARY KEY, payload BLOB NOT NULL)"
            )
        except BaseException as primary:
            self._poison()
            self._connection = None
            try:
                connection.close()
            except BaseException as cleanup_error:
                add_cleanup_note(primary, "MOS result SQLite initialization", cleanup_error)
            raise

    def add(self, ordinal: int, payload: bytes) -> None:
        self._ensure_usable("MOS result store")
        if self._finalized:
            raise RuntimeError("MOS result store is already finalized")
        if type(ordinal) is not int or not 0 <= ordinal < self._total_rows:
            raise ValueError("MOS result ordinal is invalid")
        if type(payload) is not bytes or not payload:
            raise ValueError("MOS result payload must be nonempty bytes")
        record_bytes = len(payload) + 64
        if record_bytes > _LPT_BATCH_MAX_BYTES:
            raise ValueError("MOS result row exceeds bounded batch byte limit")
        if self._pending and (
            len(self._pending) >= _LPT_BATCH_MAX_ROWS
            or self._pending_bytes + record_bytes > _LPT_BATCH_MAX_BYTES
        ):
            self._flush()
        self._pending.append((ordinal, payload))
        self._pending_bytes += record_bytes

    def _flush(self) -> None:
        connection = self._ensure_usable("MOS result store")
        if not self._pending:
            return
        try:
            with connection:
                try:
                    connection.executemany(
                        "INSERT INTO results (ordinal, payload) VALUES (?, ?)",
                        self._pending,
                    )
                except sqlite3.IntegrityError as exc:
                    if _is_unique_constraint(exc):
                        raise ValueError("MOS result ordinal is duplicate") from exc
                    raise
        except BaseException:
            self._poison()
            raise
        finally:
            self._pending.clear()
            self._pending_bytes = 0

    def finalize(self) -> None:
        self._ensure_usable("MOS result store")
        if self._finalized:
            raise RuntimeError("MOS result store is already finalized")
        try:
            self._flush()
            connection = self._ensure_usable("MOS result store")
            count, minimum, maximum = connection.execute(
                "SELECT COUNT(*), MIN(ordinal), MAX(ordinal) FROM results"
            ).fetchone()
            if (
                count != self._total_rows
                or minimum != 0
                or maximum != self._total_rows - 1
            ):
                raise ValueError("MOS worker result global ordinal coverage is invalid")
            self._finalized = True
        except BaseException:
            self._poison()
            raise

    def iter_payloads(self) -> Iterable[tuple[int, bytes]]:
        connection = self._ensure_usable("MOS result store")
        if not self._finalized:
            raise RuntimeError("MOS result store must be finalized before iteration")
        cursor = None
        try:
            cursor = connection.execute(
                "SELECT ordinal, payload FROM results ORDER BY ordinal ASC"
            )
            expected = 0
            batch: list[tuple[int, bytes]] = []
            batch_bytes = 0
            while True:
                row = cursor.fetchone()
                if row is None:
                    break
                ordinal, payload = row
                if type(ordinal) is not int or type(payload) is not bytes:
                    raise ValueError("stored MOS result row is invalid")
                record_bytes = len(payload) + 64
                if record_bytes > _LPT_BATCH_MAX_BYTES:
                    raise ValueError("stored MOS result row exceeds bounded batch byte limit")
                if batch and (
                    len(batch) >= _LPT_BATCH_MAX_ROWS
                    or batch_bytes + record_bytes > _LPT_BATCH_MAX_BYTES
                ):
                    for item in batch:
                        yield item
                    batch.clear()
                    batch_bytes = 0
                if ordinal != expected:
                    raise ValueError("stored MOS result ordinal ordering is invalid")
                expected += 1
                batch.append((ordinal, payload))
                batch_bytes += record_bytes
            for item in batch:
                yield item
            if expected != self._total_rows:
                raise ValueError("stored MOS result ordinal coverage is invalid")
        except GeneratorExit:
            raise
        except BaseException:
            self._poison()
            raise
        finally:
            primary = sys.exc_info()[1]
            if cursor is not None:
                try:
                    cursor.close()
                except BaseException as cleanup_error:
                    self._poison()
                    if isinstance(primary, GeneratorExit):
                        raise cleanup_error
                    if primary is not None:
                        add_cleanup_note(
                            primary, "MOS result cursor cleanup", cleanup_error,
                        )
                    else:
                        raise
