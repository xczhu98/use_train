"""TSE 推理接口包。

导出 :class:`TSEExtractor`（与 REAL-TSE-Challenge 比赛 Extractor 同语义的
整句推理器）、严格 JSONL 根清单读取器、公共文件 adapter 与 sidecar 发布入口。
"""

from tasks.tse.inference.extractor import TSEExtractor, load_model_local
from tasks.tse.inference.adapter import TSEInferenceAdapter, build_inference_adapter
from tasks.tse.inference.artifacts import TSEInferenceArtifactSnapshot
from tasks.tse.inference.root_manifest import TSERootManifestItem, read_tse_root_manifest
from tasks.tse.inference.sidecar import publish_sidecars_from_report
from tasks.tse.inference.streaming import (
    CenteredStreamingSTFTOLA,
    TSEStreamingWaveformSession,
    build_streaming_enrollment,
)
from tasks.tse.inference.torchscript import export_streaming_tse_torchscript

__all__ = [
    "CenteredStreamingSTFTOLA",
    "TSEExtractor",
    "TSEInferenceAdapter",
    "TSEInferenceArtifactSnapshot",
    "TSERootManifestItem",
    "build_inference_adapter",
    "publish_sidecars_from_report",
    "read_tse_root_manifest",
    "TSEStreamingWaveformSession",
    "build_streaming_enrollment",
    "export_streaming_tse_torchscript",
    "load_model_local",
]
