"""严格递归更新 dict，未知配置键直接报错。

Author: Xincheng Zhu
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


def deep_update(default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """以 ``default`` 为基准递归合并 ``user``，未知键直接 ``KeyError``。

    返回新 dict，不修改入参。
    """
    merged = deepcopy(default)
    _deep_update_strict(merged, user, path="")
    return merged


def _deep_update_strict(merged: Dict[str, Any], user: Dict[str, Any], path: str) -> None:
    for k, v in user.items():
        key_path = f"{path}.{k}" if path else str(k)
        if k not in merged:
            raise KeyError(
                f"Unknown config key {key_path!r} (typo?). "
                f"Known keys at this level: {sorted(merged.keys())}"
            )
        if isinstance(merged[k], dict) and isinstance(v, dict):
            _deep_update_strict(merged[k], v, path=key_path)
        else:
            merged[k] = v
