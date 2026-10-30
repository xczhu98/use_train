"""Audio Filter Pipeline 的配置合同与 shipped 配置资产。"""
from __future__ import annotations

from . import schema as _schema


# 兼容原 ``audio_filter_pipeline.config`` 模块的公开及私有符号。
# 使用显式 globals 转发而不是 ``import *``，保留测试和内部合同依赖的
# ``_SELECTION_MOS_SPECS`` 等私有校验元数据。
for _name, _value in vars(_schema).items():
    if _name not in {"__name__", "__package__", "__loader__", "__spec__"}:
        globals()[_name] = _value

__all__ = tuple(
    name for name in vars(_schema)
    if not name.startswith("_")
)
