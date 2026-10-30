"""测试辅助：把标准库原语替换限制在被测模块的命名空间内。"""
from __future__ import annotations

import os
import types
from typing import Any


def patched_os(**overrides: Any) -> types.ModuleType:
    """返回只覆盖指定属性的 os 代理，用作 monkeypatch 的替换值。

    `module.os` 就是真实 os 模块，直接 `setattr(module.os, ...)` 属于全局
    替换，会波及同进程的 pytest、并发线程与子进程；把被测模块的 `os`
    整体换成代理后，替换范围只限于该模块。
    """
    shim = types.ModuleType("_patched_os")
    shim.__dict__.update(vars(os))
    shim.__dict__.update(overrides)
    return shim
