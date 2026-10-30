"""Stage orchestrator 输入快照与 marker 绑定回归测试。"""
import json
import os

import pytest

from noise_filter_pipeline.core.manifest_io import AudioRecord, read_manifest, write_manifest
from noise_filter_pipeline.core.processor_base import Processor
from noise_filter_pipeline.core.stage_orchestrator import StageOrchestratorBase

pytestmark = pytest.mark.unit


class _EchoProcessor(Processor):
    def __init__(self):
        super().__init__({}, stage_name="C1")

    def process(self, input_manifest, output_dir, fingerprint_input=None):
        records = list(read_manifest(input_manifest))
        for record in records:
            record.mark_stage("C1", "passed")
        write_manifest(iter(records), os.path.join(output_dir, "c1_manifest_ds.jsonl"))
        return output_dir


class _MutatingProcessor(_EchoProcessor):
    def __init__(self, input_path: str, replacement: bytes):
        super().__init__()
        self.input_path = input_path
        self.replacement = replacement

    def process(self, input_manifest, output_dir, fingerprint_input=None):
        before = os.stat(self.input_path)
        with open(self.input_path, "wb") as handle:
            handle.write(self.replacement)
        os.utime(self.input_path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return super().process(input_manifest, output_dir, fingerprint_input)


def _record(record_id: str = "r1") -> AudioRecord:
    return AudioRecord(
        record_id=record_id,
        dataset_id="ds",
        source_id=f"s-{record_id}",
        audio={"path": f"/tmp/{record_id}.wav", "sr": 16000, "duration": 1.0},
        stages={"B4": {"status": "passed"}},
    )


def _write_record(path, record_id: str) -> None:
    path.write_text(
        json.dumps(_record(record_id).to_dict(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _run_with_snapshot_mutation(monkeypatch, input_dir, output_dir, mutation):
    orchestrator = StageOrchestratorBase()
    real_snapshot = orchestrator._input_snapshot
    mutated = False

    def mutate_before_snapshot(path, *args, **kwargs):
        nonlocal mutated
        if not mutated:
            mutation()
            mutated = True
        return real_snapshot(path, *args, **kwargs)

    monkeypatch.setattr(orchestrator, "_input_snapshot", mutate_before_snapshot)
    orchestrator._run_substage(
        _EchoProcessor(),
        str(input_dir),
        str(output_dir),
        prev_stage="B4",
    )
    assert mutated


def _output_record_ids(output_dir) -> set[str]:
    return {record.record_id for record in read_manifest(str(output_dir))}


def test_processor_input_replacement_aborts_before_marker(tmp_path):
    """processor 后同 size/mtime 内容替换必须阻断 marker 发布。"""
    input_path = tmp_path / "input.jsonl"
    original = json.dumps(_record().to_dict(), ensure_ascii=False).encode("utf-8") + b"\n"
    replacement = original.replace(b'"record_id": "r1"', b'"record_id": "x1"')
    assert len(replacement) == len(original)
    input_path.write_bytes(original)
    output_dir = tmp_path / "out"

    with pytest.raises(RuntimeError, match="内容或 identity 发生变化"):
        StageOrchestratorBase()._run_substage(
            _MutatingProcessor(str(input_path), replacement),
            str(input_path),
            str(output_dir),
            prev_stage="B4",
        )

    assert not (output_dir / "c1_output.complete").exists()


def test_marker_binds_split_time_input_identity(tmp_path):
    """完成 marker 必须记录拆分时的输入 identity，不能绑定后续新输入。"""
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps(_record().to_dict()) + "\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    orchestrator = StageOrchestratorBase()
    orchestrator._run_substage(
        _EchoProcessor(),
        str(input_path),
        str(output_dir),
        prev_stage="B4",
    )

    marker = json.loads((output_dir / "c1_output.complete").read_text(encoding="utf-8"))
    assert marker["input_identities"]
    assert marker["input_fingerprints"]


def test_manifest_added_before_snapshot_is_split_from_same_collection(
    tmp_path, monkeypatch
):
    """snapshot 前新增的 manifest 必须同时进入拆分与 marker。"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_record(input_dir / "b_manifest_a.jsonl", "r1")
    added_path = input_dir / "b_manifest_b.jsonl"
    output_dir = tmp_path / "out"

    _run_with_snapshot_mutation(
        monkeypatch,
        input_dir,
        output_dir,
        lambda: _write_record(added_path, "r2"),
    )

    assert _output_record_ids(output_dir) == {"r1", "r2"}
    marker = json.loads((output_dir / "c1_output.complete").read_text(encoding="utf-8"))
    assert [item["name"] for item in marker["input_fingerprints"]] == [
        "b_manifest_a.jsonl",
        "b_manifest_b.jsonl",
    ]
    assert marker["line_count"] == 2


def test_manifest_deleted_before_snapshot_is_excluded_from_split_collection(
    tmp_path, monkeypatch
):
    """snapshot 前删除的 manifest 不得残留在拆分输入列表中。"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_record(input_dir / "b_manifest_a.jsonl", "r1")
    removed_path = input_dir / "b_manifest_b.jsonl"
    _write_record(removed_path, "r2")
    output_dir = tmp_path / "out"

    _run_with_snapshot_mutation(
        monkeypatch,
        input_dir,
        output_dir,
        removed_path.unlink,
    )

    assert _output_record_ids(output_dir) == {"r1"}
    marker = json.loads((output_dir / "c1_output.complete").read_text(encoding="utf-8"))
    assert [item["name"] for item in marker["input_fingerprints"]] == [
        "b_manifest_a.jsonl"
    ]
    assert marker["line_count"] == 1


def test_manifest_replaced_before_snapshot_uses_replacement_for_split_and_marker(
    tmp_path, monkeypatch
):
    """snapshot 前原子替换的 manifest 必须按替换后的内容处理和标记。"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    input_path = input_dir / "b_manifest_a.jsonl"
    _write_record(input_path, "r1")
    original_inode = input_path.stat().st_ino
    replacement_path = input_dir / "replacement.tmp"
    _write_record(replacement_path, "r2")
    output_dir = tmp_path / "out"

    _run_with_snapshot_mutation(
        monkeypatch,
        input_dir,
        output_dir,
        lambda: os.replace(replacement_path, input_path),
    )

    assert input_path.stat().st_ino != original_inode
    assert _output_record_ids(output_dir) == {"r2"}
    marker = json.loads((output_dir / "c1_output.complete").read_text(encoding="utf-8"))
    assert marker["input_identities"][0]["identity"][1] == input_path.stat().st_ino
    assert marker["line_count"] == 1
