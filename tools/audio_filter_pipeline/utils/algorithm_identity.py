"""Step1/Step2 分析算法源码身份（仅 stdlib）。

Step1/Step2 的 ``config_payload`` 里的算法语义版本是手工字面量；把算法源文件
sha256 一并并入 payload 后，segmentation / signal_metrics / vad_filter /
bandwidth 的实现改动无需人工同步字面量即可让旧 artifact 失配。
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

_CACHE_SIZE = 1
_ALGORITHM_FILES = (
    "processors/segmentation.py",
    "processors/signal_metrics.py",
    "processors/vad_filter.py",
    "utils/bandwidth.py",
)
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _source_sha256(relative_path: str) -> str:
    try:
        payload = (_PACKAGE_ROOT / relative_path).read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"analysis algorithm source is not readable: {relative_path}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=_CACHE_SIZE)
def analyzer_algorithm_source_sha256() -> dict[str, str]:
    """返回分析算法源文件到 sha256 的映射；进程内缓存单次读取结果。"""
    return {path: _source_sha256(path) for path in _ALGORITHM_FILES}


def analyzer_algorithm_payload() -> dict[str, object]:
    """返回可直接嵌入 ``config_payload`` 的算法源码身份。"""
    return {
        "version": 1,
        "source_sha256": dict(analyzer_algorithm_source_sha256()),
    }
