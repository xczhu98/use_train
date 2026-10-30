"""SS source pool 的兼容导入；共享实现支持可选 canonical speaker identity。"""

from __future__ import annotations

from core.data.source_pool import SourcePoolRecord, parse_source_pool

__all__ = ["SourcePoolRecord", "parse_source_pool"]
