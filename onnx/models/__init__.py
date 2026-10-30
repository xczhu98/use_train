"""ONNX streaming model implementations.

Contains frame-by-frame streaming wrappers for:
  - BSRNN (via StreamingBSRNNWrapper)
  - CAGCRN
  - DeepVQE
  - LocalVQE
  - GTCRN
  - OnlineSpatialNet backend adapter (multichannel observation, mono output)
  - Generic ONNX inference wrappers
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_ATTRS = {
    "StreamingBSRNNWrapper": ("onnx.models.streaming_bsrnn", "StreamingBSRNNWrapper"),
    "StreamingCAGCRN": ("onnx.models.streaming_cagcrn", "StreamingCAGCRN"),
    "StreamingCAGCRNFlat": ("onnx.models.streaming_cagcrn", "StreamingCAGCRNFlat"),
    "StreamingDeepVQE": ("onnx.models.streaming_deepvqe", "StreamingDeepVQE"),
    "StreamingDeepVQEFlat": ("onnx.models.streaming_deepvqe", "StreamingDeepVQEFlat"),
    "StreamingLocalVQE": ("onnx.models.streaming_localvqe", "StreamingLocalVQE"),
    "StreamingLocalVQEFlat": ("onnx.models.streaming_localvqe", "StreamingLocalVQEFlat"),
    "StreamingGTCRN": ("onnx.models.streaming_gtcrn", "StreamingGTCRN"),
    "StreamingTSEFlat": ("onnx.models.streaming_tse", "StreamingTSEFlat"),
    "StreamingOnlineSpatialNetFlat": (
        "onnx.models.streaming_online_spatialnet", "StreamingOnlineSpatialNetFlat"
    ),
    "OnlineSpatialNetSession": (
        "onnx.models.streaming_online_spatialnet", "OnlineSpatialNetSession"
    ),
    "convert_gtcrn_to_streaming": ("onnx.models.streaming_gtcrn", "convert_gtcrn_to_streaming"),
    "StreamingInferenceWrapper": ("onnx.models.streaming_wrapper", "StreamingInferenceWrapper"),
    "StreamingONNXWrapper": ("onnx.models.streaming_wrapper", "StreamingONNXWrapper"),
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name: str) -> Any:
    """按需加载模型 wrapper，避免无关模型的专属依赖阻断包导入。"""
    try:
        module_name, attr_name = _LAZY_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
