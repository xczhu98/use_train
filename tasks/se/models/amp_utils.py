"""SE 模型族的 AMP 精度隔离工具。

原实现用 ``@torch.amp.autocast("cuda", enabled=False)`` 在**构造期**固定设备类型：
无 CUDA 的 torch 构建会在 import 期直接抛 ``RuntimeError``（``is_autocast_available``
为 False），CPU/XPU 等实际设备上的 autocast 也不会被关闭。这里改为按运行时输入张量
的真实设备类型关闭 autocast，保持「该模块始终以 FP32 前向」的精度契约不变。
"""

from __future__ import annotations

import contextlib
import functools
from typing import Callable, Iterator

import torch

# 只有这些设备类型支持 autocast；其他设备类型（如 meta）构造 autocast 会直接报错。
_AUTOCAST_DEVICE_TYPES = frozenset({"cuda", "cpu", "xpu", "mps", "hpu", "xla", "ipu"})


def _device_type(value: object) -> str | None:
    if isinstance(value, torch.Tensor):
        return value.device.type
    if isinstance(value, torch.device):
        return value.type
    return None


@contextlib.contextmanager
def fp32_autocast(*references: object) -> Iterator[None]:
    """按首个张量/``torch.device`` 实参的真实设备类型关闭 autocast。

    参数不足以判定设备类型（或设备类型不支持 autocast）时原样执行，不做任何精度干预。
    """
    device_type = None
    for reference in references:
        device_type = _device_type(reference)
        if device_type is not None:
            break
    if device_type is None or device_type not in _AUTOCAST_DEVICE_TYPES:
        yield
        return
    with torch.amp.autocast(device_type, enabled=False):
        yield


def fp32_forward(fn: Callable) -> Callable:
    """装饰器：按被装饰方法首个张量/``torch.device`` 实参的真实设备类型关闭 autocast。"""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with fp32_autocast(*args, *kwargs.values()):
            return fn(self, *args, **kwargs)

    return wrapper
