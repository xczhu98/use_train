"""evaluate 残留安全与覆盖率回归测试。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from core.utils.inference_artifacts import fingerprint_file, inference_output_lock
from evaluate import eval_io, eval_pipeline_common
from evaluate.aec_pairing import build_farend_index, resolve_farend_path
from evaluate.checkpoint_loading import load_checkpoint_payload
from evaluate.common import (
    append_extra_metrics_to_results,
    load_normalized_score,
    metric_coverage_status,
)
from evaluate.eval_pipeline_common import _safe_output_uid
from evaluate.urgent_score import SCORE_FILENAME, SCORE_SCHEMA

pytestmark = [pytest.mark.unit]


def test_checkpoint_loader_rehashes_same_fd_after_metadata_preserving_mutation(
    tmp_path: Path,
) -> None:
    """加载器返回后即使 inode、大小和 mtime 未变，也必须拒绝内容篡改。"""
    checkpoint_path = tmp_path / "model.pt"
    original = b"original-checkpoint"
    replacement = b"modified-checkpoint"
    assert len(replacement) == len(original)
    checkpoint_path.write_bytes(original)
    expected_fingerprint = fingerprint_file(checkpoint_path)
    original_stat = checkpoint_path.stat()

    def mutate_during_load(_stream, **_kwargs):
        checkpoint_path.write_bytes(replacement)
        os.utime(
            checkpoint_path,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        return {"model": {}}

    with pytest.raises(RuntimeError, match="加载期间内容发生变化"):
        load_checkpoint_payload(
            checkpoint_path,
            expected_fingerprint=expected_fingerprint,
            checkpoint_loader=mutate_during_load,
        )


def test_not_requested_optional_metrics_do_not_lower_coverage_status() -> None:
    metric_coverage = {
        "PESQ": {"status": "done"},
        "AECMOS_ECHO": {"status": "skipped", "reason": "not_requested"},
        "EMOS": {"status": "skipped", "reason": "not_requested"},
    }

    assert metric_coverage_status(metric_coverage) == "done"


def test_requested_optional_metric_skip_remains_partial() -> None:
    metric_coverage = {
        "PESQ": {"status": "done"},
        "AECMOS_ECHO": {"status": "skipped", "reason": "dependency unavailable"},
    }

    assert metric_coverage_status(metric_coverage) == "partial"


def test_nested_input_dir_uid_is_encoded_but_scp_uid_rejects_path() -> None:
    nested_uid = "speaker_a/session_01/utt_001"

    assert _safe_output_uid(nested_uid, allow_path_separators=True) == (
        f"uid-{nested_uid.encode('utf-8').hex()}"
    )
    with pytest.raises(ValueError, match="unsafe output uid"):
        _safe_output_uid(nested_uid)


def test_extra_metrics_append_serializes_on_output_lock(tmp_path: Path) -> None:
    """RESULTS.txt 的 read-modify-write 必须在 output_dir 锁内，并发 writer 立即失败。"""
    with inference_output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="已有活跃 writer"):
            append_extra_metrics_to_results(tmp_path, {"ERLE_dB": 1.0})


def test_score_load_does_not_swallow_plain_value_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """控制流只认 ResultsBindingMismatch：文案相同的普通 ValueError 不得降级为 None。"""
    (tmp_path / "RESULTS.txt").write_text("Normalized_Average: 0.5000\n", encoding="utf-8")
    (tmp_path / SCORE_FILENAME).write_text(
        json.dumps({"schema": SCORE_SCHEMA}), encoding="utf-8"
    )

    def raise_message_only(*_args, **_kwargs):
        raise ValueError("URGENT normalized score RESULTS binding mismatch")

    monkeypatch.setattr("evaluate.common.validate_normalized_score", raise_message_only)
    with pytest.raises(ValueError, match="binding mismatch"):
        load_normalized_score(tmp_path)


def test_farend_index_is_consumed_without_rebuilding_dict(tmp_path: Path) -> None:
    """预建 index 必须被消费：far_files 为空也能解析，且与逐次建表结果一致。"""
    far_files = [("farend_speech_a", "/far/a.wav")]
    index = build_farend_index(far_files)

    assert resolve_farend_path(tmp_path / "nearend_mic_a.wav", [], far_index=index) == "/far/a.wav"
    assert resolve_farend_path(tmp_path / "nearend_mic_a.wav", far_files) == "/far/a.wav"


def test_scp_empty_source_policy_is_explicit_per_entrypoint(tmp_path: Path) -> None:
    """空源策略显式化：eval_io 默认放行，评估流水线入口必须 fail-fast。"""
    empty = tmp_path / "empty"
    empty.mkdir()

    generated = eval_io.generate_scp_from_dir(empty, scp_path=tmp_path / "empty.scp")
    assert generated.read_text(encoding="utf-8") == ""
    with pytest.raises(ValueError, match="zero entries"):
        eval_io.generate_scp_from_dir(
            empty, scp_path=tmp_path / "empty-required.scp", require_non_empty=True
        )
    with pytest.raises(ValueError, match="zero entries"):
        eval_pipeline_common.generate_scp_from_dir(empty, scp_path=tmp_path / "empty-pipe.scp")
