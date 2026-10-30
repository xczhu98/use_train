"""ONNX Export Module

This package contains all ONNX-specific implementations:
  - Streaming model wrappers (StreamingGTCRN, StreamingBSRNNWrapper,
    StreamingCAGCRN, StreamingDeepVQE)
  - ONNX export utilities
  - Streaming inference wrappers

These modules are separated from training code (tasks/se/) to maintain clear boundaries:
  - Training: tasks/se/models/*.py
  - Evaluation: evaluate/*.py
  - ONNX Export: onnx/*.py

Usage:
    # Export GTCRN to ONNX
    from onnx.models.streaming_gtcrn import StreamingGTCRN, convert_gtcrn_to_streaming
    from onnx.export import export_model

    # Verify streaming consistency
    from onnx.models.streaming_wrapper import StreamingInferenceWrapper
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_ATTRS = {
    "export_model": ("onnx.export", "export_model"),
    "StreamingGTCRN": ("onnx.models.streaming_gtcrn", "StreamingGTCRN"),
    "convert_gtcrn_to_streaming": ("onnx.models.streaming_gtcrn", "convert_gtcrn_to_streaming"),
    "StreamingInferenceWrapper": ("onnx.models.streaming_wrapper", "StreamingInferenceWrapper"),
    "StreamingONNXWrapper": ("onnx.models.streaming_wrapper", "StreamingONNXWrapper"),
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name: str) -> Any:
    """按需加载公开 API，避免导入无关模型的专属依赖。"""
    try:
        module_name, attr_name = _LAZY_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
