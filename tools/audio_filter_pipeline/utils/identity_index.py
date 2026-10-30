"""临时 sqlite 唯一索引：流式重复检测（仅 stdlib）。

与 Step3 ``select`` 的 segment-ids sqlite 同构：索引只落在临时目录，不把
``source_key``/``segment_id`` 全量累积进 Python ``set``，避免流式设计中
O(rows) 的内存占用。
"""

from __future__ import annotations

import os
import re
import sqlite3
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager


_INDEX_PREFIX = "audio-filter-identity-"
_NAMESPACE_RE = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")


@contextmanager
def unique_identity_index() -> Iterator[Callable[[str, str], bool]]:
    """打开临时唯一索引，返回 ``reserve(namespace, value)`` 登记函数。

    每个 ``namespace`` 一张表；首次登记返回值 ``True``，重复登记返回
    ``False``，由调用方按各自的 artifact 合同 fail-fast。
    """
    with tempfile.TemporaryDirectory(prefix=_INDEX_PREFIX) as directory:
        connection = sqlite3.connect(os.path.join(directory, "identities.sqlite3"))
        try:
            tables: set[str] = set()

            def reserve(namespace: str, value: str) -> bool:
                if type(namespace) is not str or _NAMESPACE_RE.fullmatch(namespace) is None:
                    raise ValueError(f"identity index namespace is invalid: {namespace!r}")
                if type(value) is not str:
                    raise TypeError("identity index value must be a string")
                if namespace not in tables:
                    connection.execute(
                        f'CREATE TABLE IF NOT EXISTS "{namespace}" '
                        "(value TEXT PRIMARY KEY)",
                    )
                    tables.add(namespace)
                try:
                    connection.execute(
                        f'INSERT INTO "{namespace}" (value) VALUES (?)', (value,),
                    )
                except sqlite3.IntegrityError:
                    return False
                return True

            yield reserve
        finally:
            connection.close()
