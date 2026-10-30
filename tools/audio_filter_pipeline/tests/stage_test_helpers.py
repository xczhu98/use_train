"""Stage 测试专用的 runner-minted upstream 辅助。"""

from __future__ import annotations

from audio_filter_pipeline.processors.stages_v3.base import (
    _mint_runner_expectation,
)
from audio_filter_pipeline.utils.artifacts import (
    ArtifactIndex,
    PipelineStep,
    open_artifact_for_read,
)


def runner_expectation_for_test(index: ArtifactIndex):
    """复验真实 artifact 后，为 direct stage 测试 mint 进程内 capability。"""
    step = PipelineStep(index.step)
    verified = open_artifact_for_read(
        index.output_root,
        step,
        index.dataset_id,
        expected_config_digest=index.config_digest,
        expected_lineage_digest=index.lineage_digest,
    )
    if verified.artifact_digest != index.artifact_digest:
        raise ValueError("test upstream artifact identity changed before capability mint")
    return _mint_runner_expectation(verified, step)
