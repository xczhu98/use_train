"""样本构建事务的 last-completed 状态快照与回滚。

Dataset 负责事务边界；本模块负责恢复可变容器，避免失败样本污染已发布事实。
sealed trace 等非容器对象保持原引用，不复制整个声学场景。
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np


_MUTABLE_SNAPSHOT_TYPES = (np.ndarray, dict, list, tuple, set)


def capture_last_completed_state(owner: Any) -> dict[str, tuple[Any, Any]]:
    """保存原对象和内容；标量及 sealed trace 仅保存原引用。"""
    return {
        name: (
            value,
            copy.deepcopy(value) if isinstance(value, _MUTABLE_SNAPSHOT_TYPES) else value,
        )
        for name, value in vars(owner).items()
        if name.startswith("_last_")
    }


def _restore_value(target: Any, saved: Any) -> Any:
    """兼容容器尽量原地恢复；标量或不兼容数组返回应重新挂接的值。"""
    if isinstance(target, np.ndarray) and isinstance(saved, np.ndarray):
        if target.shape == saved.shape and target.dtype == saved.dtype and target.flags.writeable:
            target[...] = saved
            return target
        return copy.deepcopy(saved)
    if isinstance(target, dict) and isinstance(saved, dict):
        for key in tuple(target):
            if key not in saved:
                del target[key]
        for key, value in saved.items():
            target[key] = _restore_value(target[key], value) if key in target else copy.deepcopy(value)
        return target
    if isinstance(target, list) and isinstance(saved, list):
        for index in range(min(len(target), len(saved))):
            target[index] = _restore_value(target[index], saved[index])
        del target[len(saved):]
        target.extend(copy.deepcopy(saved[len(target):]))
        return target
    if isinstance(target, tuple) and isinstance(saved, tuple) and len(target) == len(saved):
        restored = tuple(_restore_value(current, value) for current, value in zip(target, saved))
        if all(current is value for current, value in zip(target, restored)):
            return target
        return restored
    if isinstance(target, set) and isinstance(saved, set):
        target.clear()
        target.update(copy.deepcopy(saved))
        return target
    return copy.deepcopy(saved)


def restore_last_completed_state(owner: Any, snapshot: Mapping[str, tuple[Any, Any]]) -> None:
    """撤销未提交字段，并恢复上一个完成样本的内容与可兼容的容器引用。"""
    for name in tuple(vars(owner)):
        if name.startswith("_last_") and name not in snapshot:
            delattr(owner, name)
    for name, (original, saved) in snapshot.items():
        restored = (
            _restore_value(original, saved)
            if isinstance(original, _MUTABLE_SNAPSHOT_TYPES)
            else original
        )
        setattr(owner, name, restored)
