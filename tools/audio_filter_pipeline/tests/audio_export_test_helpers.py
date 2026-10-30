"""Audio exporter 测试用 producer artifact 发布辅助。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from audio_filter_pipeline.utils.artifacts import (
    ArtifactWriter,
    PipelineStep,
    open_artifact_for_read,
)


def install_export_rows_stub(monkeypatch, *exporter_classes):
    """把测试发布入口挂到生产 exporter 上，并由 monkeypatch 在用例结束时撤销。

    生产类不得自己定义 ``_export_test_rows``：一旦定义，测试就会静默改用
    生产实现而无人察觉，因此这里先断言该名字未被占用。
    """
    for exporter_class in exporter_classes:
        assert not hasattr(exporter_class, "_export_test_rows"), (
            f"{exporter_class.__name__} 不得定义 _export_test_rows"
        )
        monkeypatch.setattr(
            exporter_class, "_export_test_rows", export_rows_as_artifact, raising=False,
        )


def export_rows_as_artifact(exporter, rows: Iterable[Mapping[str, Any]]):
    """将测试 rows 发布为真实 artifact，再调用唯一生产入口。"""
    materialized = list(rows)
    step = PipelineStep(exporter._EXPORT_STEP)
    dataset_ids = {row.get("dataset_id") for row in materialized}
    if len(dataset_ids) != 1 or not materialized:
        if not materialized:
            raise ValueError("test artifact helper requires at least one row")
        raise ValueError("test artifact rows must share one dataset_id")
    dataset_id = dataset_ids.pop()
    canonical = json.dumps(
        materialized, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    rows_key = hashlib.sha256(canonical).hexdigest()
    artifact_root = Path(exporter._export_root).parent / ".test-producer-artifacts" / rows_key
    config_digest = hashlib.sha256(f"{step.value}:config".encode()).hexdigest()
    lineage_digest = hashlib.sha256(f"{step.value}:lineage".encode()).hexdigest()
    existing: FileExistsError | None = None
    try:
        writer = ArtifactWriter(
            str(artifact_root), step, dataset_id, config_digest, lineage_digest,
        )
        for row in materialized:
            writer.append(row)
        writer.finalize()
    except FileExistsError as exc:
        existing = exc
    try:
        index = open_artifact_for_read(
            str(artifact_root), step, dataset_id,
            expected_config_digest=config_digest,
            expected_lineage_digest=lineage_digest,
        )
    except Exception:
        # 已有 artifact 未通过复用校验时，必须暴露写入冲突本身而不是静默复用。
        if existing is not None:
            raise existing
        raise
    return exporter.export_artifact(index)
