"""Shared data defaults and ``data.mode`` normalization."""

from __future__ import annotations

from typing import NamedTuple

DEFAULT_PLACEHOLDER_NUM_SAMPLES = 1024

# ── data.mode 取值常量 ──
# 集中管理 data.mode 的合法取值，避免在 core / se / aec / tse 各处散落字符串
# 字面量判断（新增模式或改名时只需改这一处）。
DATA_MODE_DYNAMIC = "dynamic"
DATA_MODE_DYNAMIC_AEC = "dynamic_aec"
DATA_MODE_DYNAMIC_TSE = "dynamic_tse"
DATA_MODE_DYNAMIC_SS = "dynamic_ss"
DATA_MODE_OFFLINE = "offline"
DATA_MODE_OFFLINE_AEC = "offline_aec"
DATA_MODE_OFFLINE_TSE = "offline_tse"
DATA_MODE_OFFLINE_SS = "offline_ss"
DATA_MODE_DYNAMIC_SSL = "dynamic_ssl"


class NormalizedDataMode(NamedTuple):
    """标准化后的任务、存储形态与公开 ``data.mode``。"""

    task: str
    storage: str
    task_mode: str


_DATA_MODE_PROTOCOL = {
    DATA_MODE_DYNAMIC: NormalizedDataMode("se", "dynamic", DATA_MODE_DYNAMIC),
    DATA_MODE_OFFLINE: NormalizedDataMode("se", "offline", DATA_MODE_OFFLINE),
    DATA_MODE_DYNAMIC_AEC: NormalizedDataMode(
        "aec", "dynamic", DATA_MODE_DYNAMIC_AEC
    ),
    DATA_MODE_OFFLINE_AEC: NormalizedDataMode(
        "aec", "offline", DATA_MODE_OFFLINE_AEC
    ),
    DATA_MODE_DYNAMIC_TSE: NormalizedDataMode(
        "tse", "dynamic", DATA_MODE_DYNAMIC_TSE
    ),
    DATA_MODE_OFFLINE_TSE: NormalizedDataMode(
        "tse", "offline", DATA_MODE_OFFLINE_TSE
    ),
    DATA_MODE_DYNAMIC_SS: NormalizedDataMode(
        "ss", "dynamic", DATA_MODE_DYNAMIC_SS
    ),
    DATA_MODE_OFFLINE_SS: NormalizedDataMode(
        "ss", "offline", DATA_MODE_OFFLINE_SS
    ),
    DATA_MODE_DYNAMIC_SSL: NormalizedDataMode(
        "ssl", "dynamic", DATA_MODE_DYNAMIC_SSL
    ),
}
_SUPPORTED_DATA_MODES = tuple(_DATA_MODE_PROTOCOL)


def normalize_data_mode(value: object) -> NormalizedDataMode:
    """纯函数：把公开 ``data.mode`` 归一为 task/storage/task_mode。"""
    mode = value.strip().lower() if isinstance(value, str) else None
    normalized = _DATA_MODE_PROTOCOL.get(mode)
    if normalized is None:
        supported = ", ".join(_SUPPORTED_DATA_MODES)
        raise ValueError(
            f"Unsupported data.mode {value!r}; supported modes: {supported}"
        )
    return normalized
