"""
Tests for core.stats stage statistics reporting.
"""
import json
import os
import tempfile

import pytest

from noise_filter_pipeline.core.manifest_io import AudioRecord, write_manifest
from noise_filter_pipeline.core.stats import compute_stage_stats, write_stage_stats

pytestmark = pytest.mark.pipeline


def test_compute_stage_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "test.jsonl")
        recs = [
            AudioRecord(
                record_id=f"ds__src__{i}",
                dataset_id="ds",
                source_id="src",
                audio={"path": f"/tmp/{i}.wav", "sr": 16000, "duration": 1.0 + i},
                labels={"original": "noise" if i % 2 == 0 else "speech"},
            )
            for i in range(4)
        ]
        for i, rec in enumerate(recs):
            rec.mark_stage("A", "passed" if i < 2 else "rejected")
        write_manifest(iter(recs), manifest)

        stats = compute_stage_stats(manifest, "A")
        assert stats["stage"] == "A"
        assert stats["total_records"] == 4
        assert stats["status"]["passed"] == 2
        assert stats["status"]["rejected"] == 2
        assert stats["datasets"] == {"ds": 4}
        assert "noise" in stats["top_labels"]
        assert "speech" in stats["top_labels"]
        assert stats["duration"]["total_seconds"] == 10.0


def test_write_stage_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "test.jsonl")
        rec = AudioRecord(
            record_id="r1",
            dataset_id="ds",
            source_id="s",
            audio={"path": "/tmp/a.wav", "sr": 16000, "duration": 2.0},
        )
        rec.mark_stage("B", "passed")
        write_manifest(iter([rec]), manifest)

        stats_path = os.path.join(tmpdir, "b_stats.json")
        write_stage_stats(manifest, "B", stats_path)
        assert os.path.isfile(stats_path)
        with open(stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["stage"] == "B"
        assert data["total_records"] == 1


def test_stage_stats_aggregates_substage_marks():
    """records 只带 B1..B4 子阶段标记时，统计 'B' 不得恒为 0。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "test.jsonl")

        def _rec(rid):
            return AudioRecord(
                record_id=rid,
                dataset_id="ds",
                source_id="src",
                audio={"path": f"/tmp/{rid}.wav", "sr": 16000, "duration": 2.0},
            )

        # r1: 全流程通过
        r1 = _rec("r1")
        for s in ("B1", "B2", "B3", "B4"):
            r1.mark_stage(s, "passed")
        # r2: B4 被拒
        r2 = _rec("r2")
        for s in ("B1", "B2", "B3"):
            r2.mark_stage(s, "passed")
        r2.mark_stage("B4", "rejected")
        # r3: B1 被拒（源记录，无后续子阶段）
        r3 = _rec("r3")
        r3.mark_stage("B1", "rejected")
        write_manifest(iter([r1, r2, r3]), manifest)

        stats = compute_stage_stats(manifest, "B")
        assert stats["total_records"] == 3
        assert stats["status"]["passed"] == 1
        assert stats["status"]["rejected"] == 2
        assert stats["status"]["other"] == 0
        # 子阶段明细
        assert stats["substages"]["B4"] == {"passed": 1, "rejected": 1}
        assert stats["substages"]["B1"] == {"passed": 2, "rejected": 1}


def test_stage_stats_skipped_records():
    """Stage C 中被 skipped 透传的记录应计入 skipped。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "test.jsonl")
        r1 = AudioRecord(
            record_id="r1", dataset_id="ds", source_id="src",
            audio={"path": "/tmp/r1.wav", "sr": 16000, "duration": 2.0},
        )
        for s in ("C1", "C2", "C3"):
            r1.mark_stage(s, "passed")
        r2 = AudioRecord(
            record_id="r2", dataset_id="ds", source_id="src",
            audio={"path": "/tmp/r2.wav", "sr": 16000, "duration": 2.0},
        )
        for s in ("C1", "C2", "C3"):
            r2.mark_stage(s, "skipped")
        write_manifest(iter([r1, r2]), manifest)

        stats = compute_stage_stats(manifest, "C")
        assert stats["status"]["passed"] == 1
        assert stats["status"]["skipped"] == 1
        assert stats["status"]["rejected"] == 0


def test_pass_reject_rate_denominator_includes_skipped():
    """pass_rate/reject_rate 的分母为全部记录（含 skipped 透传）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "test.jsonl")

        def _rec(rid):
            return AudioRecord(
                record_id=rid, dataset_id="ds", source_id="src",
                audio={"path": f"/tmp/{rid}.wav", "sr": 16000, "duration": 2.0},
            )

        r_pass = _rec("r_pass")
        r_pass.mark_stage("A", "passed")
        r_rej = _rec("r_rej")
        r_rej.mark_stage("A", "rejected")
        skips = []
        for i in range(8):
            r = _rec(f"r_skip{i}")
            r.mark_stage("A", "skipped")
            skips.append(r)
        write_manifest(iter([r_pass, r_rej] + skips), manifest)

        stats = compute_stage_stats(manifest, "A")
        assert stats["total_records"] == 10
        assert stats["status"]["passed"] == 1
        assert stats["status"]["rejected"] == 1
        assert stats["status"]["skipped"] == 8
        # 分母含 skipped：1/10 = 0.1，而非 1/2 = 0.5
        assert stats["pass_rate"] == 0.1
        assert stats["reject_rate"] == 0.1


def test_pass_reject_rate_empty_manifest():
    """total=0 时 pass_rate/reject_rate 应为 0.0，不得除零。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "empty.jsonl")
        write_manifest(iter([]), manifest)

        stats = compute_stage_stats(manifest, "A")
        assert stats["total_records"] == 0
        assert stats["pass_rate"] == 0.0
        assert stats["reject_rate"] == 0.0


def test_stage_stats_derives_substage_keys_dynamically():
    """子阶段键从记录实际标记动态推导：新增子阶段（如 B5）无需改 stats 代码。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "test.jsonl")
        rec = AudioRecord(
            record_id="r1", dataset_id="ds", source_id="src",
            audio={"path": "/tmp/r1.wav", "sr": 16000, "duration": 2.0},
        )
        for s in ("B1", "B2", "B3", "B4", "B5"):
            rec.mark_stage(s, "passed")
        write_manifest(iter([rec]), manifest)

        stats = compute_stage_stats(manifest, "B")
        assert stats["status"]["passed"] == 1
        # 硬编码列表里没有的 B5 也应出现在子阶段明细中
        assert stats["substages"]["B5"] == {"passed": 1}
