"""任务无关的共享推理基础接口。"""

from .artifact_binding import bind_artifacts, bind_config, verify_file_bindings
from .contracts import (
    ArtifactBinding,
    FileInferenceItem,
    InferenceAdapter,
    InferenceReport,
    InferenceSpec,
    PreparedOutput,
    ReusePolicy,
)
from .input_inventory import build_inventory, inventory_items, verify_inventory
from .runner import run_inference, run_inference_report

__all__ = [
    "InferenceAdapter",
    "ArtifactBinding",
    "FileInferenceItem",
    "InferenceReport",
    "InferenceSpec",
    "PreparedOutput",
    "ReusePolicy",
    "bind_artifacts",
    "bind_config",
    "verify_file_bindings",
    "build_inventory",
    "inventory_items",
    "verify_inventory",
    "run_inference",
    "run_inference_report",
]
