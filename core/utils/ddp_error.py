"""跨 DDP 边界传播异常前的不可恢复 CUDA 故障判定。"""

from __future__ import annotations

import torch


_UNRECOVERABLE_CUDA_MARKERS = (
    "cuda out of memory",
    "cuda assert",
    "device-side assert",
    "device-side error",
    "an illegal memory access was encountered",
    "misaligned address",
    "unspecified launch failure",
    "cuda error: launch failure",
)


def is_unrecoverable_cuda_exception(exc: BaseException) -> bool:
    """判断异常是否表示当前 CUDA context 不能安全参与后续 collective。"""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _UNRECOVERABLE_CUDA_MARKERS)
