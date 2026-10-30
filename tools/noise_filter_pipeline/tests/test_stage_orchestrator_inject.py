"""
Stage Orchestrator 拆分/汇总与幂等逻辑的回归测试。

覆盖 code-review 发现：
- 空 stages dict（"stages": {}）注入不得产生尾逗号非法 JSON；
- 锚点不唯一（嵌套 "stages": { 或字符串值含锚点）时回退 parse+dump；
- 非法 JSON 坏行跳过并告警，不中断整个拆分。

子阶段输出契约（目录化，2026-08 起）：
- 子阶段输出 = output_dir 下 per-dataset 文件集合（<stage>_manifest_<ds>.jsonl），
  eligible 由 processor 写入，ineligible 在拆分后按 dataset 追加到同一组文件；
  不再落 <stage>_merged.jsonl 单文件（legacy merged 仅在有效 marker 下复用）；
- 幂等标记为 <stage>_output.complete（输入/配置/输出文件集合三重指纹）。

此外覆盖 Stage C 固定 C1 → C2 → C3 链路与资产指纹注入：
- C3 固定消费 C2 输出，不再构造旧 C4；
- C1/C2 各自只注入生成 identity，C3 独立绑定五项发布 identity；
- policy 内容变化只改变 C3，Qwen/AudioTag 内容变化只影响各自 checkpoint；
- 正式模式启动前聚合校验运行资产，dry-run/disabled 分支保留 unset 占位；
- Stage checkpoint fingerprint 只负责生成资产失效，strict manifest runtime 独立负责
  schema、canonical policy、签章格式和 manifest 内一致性。
"""
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pytest

from core.data.noise_admission_policy import (
    CLASSIFIER_LABEL_COUNT,
    EVIDENCE_CONTRACT_VERSION,
    SCORE_CONTRACT,
)
from noise_filter_pipeline.core.manifest_io import AudioRecord, read_manifest, write_manifest
from noise_filter_pipeline.core.pipeline_config import UnifiedPipelineConfig
from noise_filter_pipeline.core.processor_base import Processor
from noise_filter_pipeline.core.stage_orchestrator import StageOrchestratorBase
from noise_filter_pipeline.stages.stage_c_single_event import StageCOrchestrator
from third_party_scripts.icefall_audiotaq.audiotag_result import (
    format_audiotag_results,
    load_label_metadata,
)

pytestmark = pytest.mark.pipeline

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REAL_LABEL_DICT = str(
    _REPO_ROOT
    / "third_party_scripts"
    / "icefall_audiotaq"
    / "data"
    / "class_labels_indices.csv"
)
_LABEL_METADATA = load_label_metadata(_REAL_LABEL_DICT)


def _c2_rain_result() -> dict:
    scores = [0.0] * CLASSIFIER_LABEL_COUNT
    for rank, label in enumerate(("Rain", "Music", "Wind", "Thunder", "Water")):
        mid = _LABEL_METADATA.label_to_mid[label]
        scores[_LABEL_METADATA.index_of(mid)] = 1.0 - rank * 0.1
    result = format_audiotag_results([scores], _LABEL_METADATA)[0]
    return {"audioset_top5": result["audioset_top5"]}


class _EchoProcessor(Processor):
    """测试用 processor：eligible 记录原样透传并打本 stage passed 标记。

    输出按目录化契约写 <stage>_manifest_ds.jsonl（测试记录 dataset_id 恒为 ds），
    返回 output_dir（与真实 processor 的 per-dataset 输出一致）。
    """

    def __init__(self, stage_name: str = "C1", config=None):
        super().__init__(config or {}, stage_name=stage_name)

    def process(self, input_manifest: str, output_dir: str, fingerprint_input: Optional[str] = None) -> str:
        out = os.path.join(output_dir, f"{self.stage_name.lower()}_manifest_ds.jsonl")
        recs = list(read_manifest(input_manifest))
        for rec in recs:
            rec.mark_stage(self.stage_name, "passed")
        write_manifest(iter(recs), out)
        return output_dir


class _SkipAwareFailClosedProcessor(Processor):
    """模拟真实 processor 的 should_skip；真正执行的记录一律拒绝。"""

    def __init__(self, stage_name: str, config=None):
        super().__init__(config or {}, stage_name=stage_name)

    def process(self, input_manifest: str, output_dir: str, fingerprint_input: Optional[str] = None) -> str:
        out = os.path.join(output_dir, f"{self.stage_name.lower()}_manifest_ds.jsonl")
        recs = list(read_manifest(input_manifest))
        for rec in recs:
            if not self.should_skip(rec):
                rec.mark_stage(
                    self.stage_name,
                    "rejected",
                    {"reason": "processor_called"},
                )
        write_manifest(iter(recs), out)
        return output_dir


def _make_record(record_id: str, stages=None, **extra) -> dict:
    rec = AudioRecord(
        record_id=record_id,
        dataset_id="ds",
        source_id=record_id,
        audio={"path": f"/tmp/{record_id}.wav", "sr": 16000, "duration": 1.0},
        labels={"original": "noise"},
    )
    if stages is not None:
        rec.stages = stages
    data = rec.to_dict()
    data.update(extra)
    return data


def _read_dir_lines(output_dir: str) -> list:
    """读取目录化输出的全部原始行（per-dataset 文件按文件名排序，确定性）。"""
    raw_lines = []
    for name in sorted(os.listdir(output_dir)):
        if not (name.endswith(".jsonl") and "_manifest_" in name):
            continue
        with open(os.path.join(output_dir, name), "r", encoding="utf-8") as f:
            raw_lines.extend(line for line in f.read().splitlines() if line.strip())
    return raw_lines


def _run_substage(lines, tmp_path, stage_name: str = "C1"):
    """跑一遍拆分+汇总，返回 (output_dir, output_raw_lines)。"""
    input_manifest = os.path.join(tmp_path, "input.jsonl")
    with open(input_manifest, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    out_dir = os.path.join(tmp_path, "out")
    result = StageOrchestratorBase()._run_substage(
        _EchoProcessor(stage_name), input_manifest, out_dir, prev_stage="B4"
    )
    assert result == out_dir  # 目录化契约：返回子阶段目录
    return out_dir, _read_dir_lines(out_dir)


def test_empty_stages_dict_injection_produces_valid_json(tmp_path):
    """空 "stages": {} 的记录注入 skipped 后必须是合法 JSON，且记录不丢失。"""
    line = json.dumps(_make_record("r1"), ensure_ascii=False)
    assert '"stages": {}' in line  # 确认锚点为空 dict 形态
    _, raw_lines = _run_substage([line], str(tmp_path))

    # 输出必须包含该记录（P1：尾逗号非法 JSON 会导致记录被静默丢弃）
    assert len(raw_lines) == 1
    data = json.loads(raw_lines[0])  # 非法 JSON 会直接在这里抛错
    assert data["stages"]["C1"]["status"] == "skipped"
    assert data["stages"]["C1"]["reason"] == "prev_stage_B4_not_passed"

    # 中间产物 ineligible 文件同样逐行可解析（关闭自动清理以便检查）
    input_manifest = os.path.join(str(tmp_path), "input.jsonl")
    with open(input_manifest, "w", encoding="utf-8") as f:
        f.write(line + "\n")
    out_dir = os.path.join(str(tmp_path), "out")
    StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1", {"cleanup_intermediate": False}),
        input_manifest, out_dir, prev_stage="B4",
    )
    ineligible = os.path.join(str(tmp_path), "out", "c1_input_ineligible.jsonl")
    with open(ineligible, "r", encoding="utf-8") as f:
        for raw in f:
            json.loads(raw)


def test_nonempty_stages_injection_keeps_existing_entries(tmp_path):
    """非空 stages dict 注入带尾逗号，已有条目保留，结果合法。"""
    line = json.dumps(
        _make_record("r1", stages={"B1": {"status": "passed", "timestamp": "t0"}}),
        ensure_ascii=False,
    )
    _, raw_lines = _run_substage([line], str(tmp_path))
    assert len(raw_lines) == 1
    data = json.loads(raw_lines[0])
    assert data["stages"]["B1"]["status"] == "passed"
    assert data["stages"]["C1"]["status"] == "skipped"


def test_anchor_not_unique_falls_back_to_parse_dump(tmp_path):
    """行内出现多个 '"stages": {'（如 provenance 里嵌套 stages 对象）时回退 parse+dump。"""
    rec = _make_record("r1")
    rec["provenance"] = {"stages": {"nested": "嵌套 stages 对象"}}
    line = json.dumps(rec, ensure_ascii=False)
    assert line.count('"stages": {') == 2

    _, raw_lines = _run_substage([line], str(tmp_path))
    assert len(raw_lines) == 1
    data = json.loads(raw_lines[0])
    assert data["stages"]["C1"]["status"] == "skipped"
    # 回退路径不得损坏嵌套的 stages 对象
    assert data["provenance"]["stages"] == {"nested": "嵌套 stages 对象"}


def test_malformed_line_fails_split_by_default(tmp_path):
    """默认严格模式下，坏 JSON 不得被静默跳过。"""
    input_manifest = _write_input(tmp_path, [json.dumps(_make_record("good")), "{坏 JSON"])
    with pytest.raises(ValueError, match="allow_partial"):
        StageOrchestratorBase()._run_substage(
            _EchoProcessor("C1"), input_manifest, os.path.join(str(tmp_path), "out"), prev_stage="B4"
        )


def test_malformed_line_can_be_skipped_with_allow_partial(tmp_path, caplog):
    """仅显式 allow_partial 时，坏 JSON 可告警跳过，其余记录继续处理。"""
    good = json.dumps(_make_record("good"), ensure_ascii=False)
    passed = json.dumps(
        _make_record("passed", stages={"B4": {"status": "passed", "timestamp": "t0"}}),
        ensure_ascii=False,
    )
    bad = "{这不是合法 JSON"
    input_manifest = _write_input(tmp_path, [good, bad, passed])
    out_dir = StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1", {"allow_partial": True}), input_manifest,
        os.path.join(str(tmp_path), "out"), prev_stage="B4"
    )
    raw_lines = _read_dir_lines(out_dir)
    assert len(raw_lines) == 2
    by_id = {json.loads(line)["record_id"]: json.loads(line) for line in raw_lines}
    assert by_id["good"]["stages"]["C1"]["status"] == "skipped"
    assert by_id["passed"]["stages"]["C1"]["status"] == "passed"
    assert "非法 JSON 行" in caplog.text


def _structurally_broken_passed_line() -> str:
    """构造前缀命中 passed 正则、但整体结构损坏（截断）的行：行尾不是 '}'。"""
    good = json.dumps(
        _make_record("p_bad", stages={"B4": {"status": "passed", "timestamp": "t0"}}),
        ensure_ascii=False,
    )
    return good[:-5] + '"truncated'


def test_strict_mode_rejects_braced_non_json_before_processor(tmp_path):
    """首尾花括号不能冒充合法 JSON object，strict 模式必须在处理前中止。"""
    input_manifest = _write_input(tmp_path, ["{not valid json}"])
    processor = _CountingEchoProcessor("C1")
    out_dir = os.path.join(str(tmp_path), "out")

    with pytest.raises(ValueError, match="JSON object"):
        StageOrchestratorBase()._run_substage(
            processor, input_manifest, out_dir, prev_stage=None
        )

    assert processor.process_calls == 0
    assert not os.path.exists(os.path.join(out_dir, "c1_output.complete"))


def test_eligible_side_structurally_broken_line_raises(tmp_path):
    """P2 回归：前缀命中 passed 正则但整体结构损坏的行，严格模式下必须 raise，
    不得被 eligible 原始透传吞掉（processor 读取时会静默丢弃该记录）。"""
    broken = _structurally_broken_passed_line()
    assert broken.endswith("truncated")  # 确认行尾不是 '}'
    input_manifest = _write_input(tmp_path, [broken, _PASSED_LINE])
    with pytest.raises(ValueError, match="allow_partial"):
        StageOrchestratorBase()._run_substage(
            _EchoProcessor("C1"), input_manifest,
            os.path.join(str(tmp_path), "out"), prev_stage="B4",
        )


def test_eligible_side_broken_line_skipped_with_allow_partial(tmp_path, caplog):
    """allow_partial 时结构损坏的 eligible 行可告警跳过，其余记录正常处理。"""
    broken = _structurally_broken_passed_line()
    input_manifest = _write_input(tmp_path, [broken, _PASSED_LINE])
    out_dir = StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1", {"allow_partial": True}), input_manifest,
        os.path.join(str(tmp_path), "out"), prev_stage="B4",
    )
    raw_lines = _read_dir_lines(out_dir)
    assert len(raw_lines) == 1
    assert json.loads(raw_lines[0])["record_id"] == "p1"
    assert "结构不完整" in caplog.text


def test_utf8_truncated_line_skipped_with_allow_partial(tmp_path, caplog):
    """UTF-8 多字节被截断的行（进程被杀后中文标签末行）不得 UnicodeDecodeError。

    回归：decode 在结构检查之前抛 UnicodeDecodeError，allow_partial 无从
    生效，重跑该 stage 直接崩溃。errors="replace" 后由结构检查拦截跳过。
    """
    good = json.dumps(_make_record("good"), ensure_ascii=False)
    # 模拟崩溃截断：完整行末尾的半个多字节字符（无换行符结尾）
    partial = json.dumps(
        _make_record("partial", stages={"B4": {"status": "passed", "timestamp": "t0"}}),
        ensure_ascii=False,
    )
    manifest_path = tmp_path / "input.jsonl"
    with open(manifest_path, "wb") as f:
        f.write(good.encode("utf-8") + b"\n")
        f.write(partial.encode("utf-8")[:-3] + b"\xe4\xb8")  # 半个中文字符

    out_dir = StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1", {"allow_partial": True}), str(manifest_path),
        os.path.join(str(tmp_path), "out"), prev_stage="B4",
    )
    raw_lines = _read_dir_lines(out_dir)
    assert len(raw_lines) == 1
    assert json.loads(raw_lines[0])["record_id"] == "good"
    assert "结构不完整" in caplog.text


def test_utf8_truncated_line_raises_by_default(tmp_path):
    """严格模式下 UTF-8 截断行同样按坏行处理：raise 并提示 allow_partial。"""
    good = json.dumps(_make_record("good"), ensure_ascii=False)
    manifest_path = tmp_path / "input.jsonl"
    with open(manifest_path, "wb") as f:
        f.write(good.encode("utf-8") + b"\n")
        f.write(b"\xe4\xb8\xad\xe5")  # 半个中文 + 半个中文字符
    with pytest.raises(ValueError, match="allow_partial"):
        StageOrchestratorBase()._run_substage(
            _EchoProcessor("C1"), str(manifest_path),
            os.path.join(str(tmp_path), "out"), prev_stage="B4",
        )


def test_split_accounting_invariant_holds_without_prev_stage(tmp_path):
    """无 prev_stage 时全部行进 eligible（eligible == lines_read 恒成立），
    行数守恒校验不得误伤正常流程。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1"), input_manifest, os.path.join(str(tmp_path), "out")
    )
    assert len(_read_dir_lines(out_dir)) == 2


def test_merge_raw_copy_preserves_lines_and_progress_count(tmp_path):
    """汇总阶段：eligible 记录在前、ineligible 按 dataset 追加到同一 per-dataset
    文件，内容逐行可解析；progress_callback 触发次数恰好等于 eligible 记录数。"""
    passed_line = json.dumps(
        _make_record("p1", stages={"B4": {"status": "passed", "timestamp": "t0"}}),
        ensure_ascii=False,
    )
    skipped_line = json.dumps(_make_record("s1"), ensure_ascii=False)

    input_manifest = os.path.join(str(tmp_path), "input.jsonl")
    with open(input_manifest, "w", encoding="utf-8") as f:
        f.write(passed_line + "\n" + skipped_line + "\n")

    ticks = []
    out_dir = StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1"),
        input_manifest,
        os.path.join(str(tmp_path), "out"),
        prev_stage="B4",
        progress_callback=lambda: ticks.append(1),
    )
    # 两条记录同属 dataset "ds"：同一 per-dataset 文件内 eligible 在前、
    # ineligible 追加在后（与旧 merged 的段序一致）
    raw_lines = _read_dir_lines(out_dir)

    assert len(raw_lines) == 2
    first = json.loads(raw_lines[0])
    assert first["record_id"] == "p1"
    assert first["stages"]["C1"]["status"] == "passed"
    second = json.loads(raw_lines[1])
    assert second["record_id"] == "s1"
    assert second["stages"]["C1"]["status"] == "skipped"
    # 进度回调恰好每条 eligible 一次（ineligible 不触发）
    assert len(ticks) == 1


def test_processor_silent_line_loss_aborts_before_ineligible_and_marker(tmp_path):
    """processor 少写 eligible 行时必须中止，不得追加 ineligible 或写完成 marker。"""
    class _DroppingProcessor(_EchoProcessor):
        def process(self, input_manifest, output_dir, fingerprint_input=None):
            recs = list(read_manifest(input_manifest))[:-1]
            out = os.path.join(output_dir, "c1_manifest_ds.jsonl")
            for rec in recs:
                rec.mark_stage("C1", "passed")
            write_manifest(iter(recs), out)
            return output_dir

    passed2 = json.dumps(
        _make_record("p2", stages={"B4": {"status": "passed", "timestamp": "t0"}}),
        ensure_ascii=False,
    )
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, passed2, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")

    with pytest.raises(RuntimeError, match="eligible 输出行数"):
        StageOrchestratorBase()._run_substage(
            _DroppingProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
        )

    assert len(_read_dir_lines(out_dir)) == 1
    assert not os.path.exists(os.path.join(out_dir, "c1_output.complete"))


@pytest.mark.parametrize("mutation", ["duplicate-other", "wrong-dataset"])
def test_processor_same_count_identity_corruption_aborts_without_marker(tmp_path, mutation):
    """同数量输出若丢 A 重复 B 或篡改 dataset_id，也必须由身份守恒拦截。"""
    class _IdentityCorruptingProcessor(Processor):
        def __init__(self):
            super().__init__({}, stage_name="C1")

        def process(self, input_manifest, output_dir, fingerprint_input=None):
            recs = list(read_manifest(input_manifest))
            if mutation == "duplicate-other":
                recs[0] = recs[1]
            else:
                recs[0].dataset_id = "wrong_ds"
            for rec in recs:
                rec.mark_stage("C1", "passed")
            write_manifest(iter(recs), os.path.join(output_dir, "c1_manifest_ds.jsonl"))
            return output_dir

    passed2 = json.dumps(
        _make_record("p2", stages={"B4": {"status": "passed", "timestamp": "t0"}}),
        ensure_ascii=False,
    )
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, passed2])
    out_dir = os.path.join(str(tmp_path), "out")

    with pytest.raises(RuntimeError, match="eligible.*身份"):
        StageOrchestratorBase()._run_substage(
            _IdentityCorruptingProcessor(), input_manifest, out_dir, prev_stage="B4"
        )

    assert not os.path.exists(os.path.join(out_dir, "c1_output.complete"))


def test_duplicate_eligible_record_id_fails_before_processor(tmp_path):
    """eligible 输入 record_id 重复时身份不明确，必须在调用 processor 前失败。"""
    processor = _CountingEchoProcessor("C1")
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _PASSED_LINE])

    with pytest.raises(ValueError, match="record_id.*重复"):
        StageOrchestratorBase()._run_substage(
            processor, input_manifest, os.path.join(str(tmp_path), "out"), prev_stage="B4"
        )

    assert processor.process_calls == 0


@pytest.mark.parametrize("bad_line", ["{not valid json}", "[]"], ids=["invalid-json", "non-object"])
def test_processor_bad_output_raises_without_marker(tmp_path, bad_line):
    """processor 输出必须逐行是 JSON object，坏输出不得发布 marker。"""
    class _BadOutputProcessor(Processor):
        def __init__(self):
            super().__init__({}, stage_name="C1")

        def process(self, input_manifest, output_dir, fingerprint_input=None):
            out = os.path.join(output_dir, "c1_manifest_ds.jsonl")
            with open(out, "w", encoding="utf-8") as f:
                f.write(bad_line + "\n")
            return output_dir

    input_manifest = _write_input(tmp_path, [_PASSED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")

    with pytest.raises(ValueError, match="processor 输出"):
        StageOrchestratorBase()._run_substage(
            _BadOutputProcessor(), input_manifest, out_dir, prev_stage="B4"
        )

    assert not os.path.exists(os.path.join(out_dir, "c1_output.complete"))


def test_processor_truncated_output_raises_without_marker(tmp_path):
    """processor 截断输出也必须 fail closed，不得过滤后发布完成状态。"""
    class _TruncatingProcessor(Processor):
        def __init__(self):
            super().__init__({}, stage_name="C1")

        def process(self, input_manifest, output_dir, fingerprint_input=None):
            out = os.path.join(output_dir, "c1_manifest_ds.jsonl")
            recs = list(read_manifest(input_manifest))
            with open(out, "w", encoding="utf-8") as f:
                for rec in recs:
                    rec.mark_stage("C1", "passed")
                    f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                f.write('{"record_id": "truncated", "stages": {')
            return output_dir

    input_manifest = _write_input(tmp_path, [_PASSED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")

    with pytest.raises(ValueError, match="processor 输出"):
        StageOrchestratorBase()._run_substage(
            _TruncatingProcessor(), input_manifest, out_dir, prev_stage="B4"
        )

    assert not os.path.exists(os.path.join(out_dir, "c1_output.complete"))


@pytest.mark.parametrize("append_result", [(0, 1), (0, 0)])
def test_ineligible_append_mismatch_aborts_without_marker(tmp_path, monkeypatch, append_result):
    """追加坏行或追加数不等于 ineligible_count 时不得发布完成 marker。"""
    monkeypatch.setattr(
        StageOrchestratorBase,
        "_append_ineligible_per_dataset",
        staticmethod(lambda *_args: append_result),
    )
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")

    with pytest.raises(RuntimeError, match="ineligible 追加不守恒"):
        StageOrchestratorBase()._run_substage(
            _EchoProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
        )

    assert not os.path.exists(os.path.join(out_dir, "c1_output.complete"))


# ---------------------------------------------------------------------------
# 子阶段幂等（目录化输出复用）
# ---------------------------------------------------------------------------


class _RetryableProcessor(_EchoProcessor):
    def has_retryable_failures(self) -> bool:
        return True


def test_retryable_processor_does_not_write_marker(tmp_path):
    """可重试 processor 的部分输出不得带完成 marker，且必须留下未完成标记。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    with pytest.raises(RuntimeError, match="可重试"):
        StageOrchestratorBase()._run_substage(
            _RetryableProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
        )
    assert not os.path.exists(os.path.join(out_dir, "c1_output.complete"))
    assert os.path.isfile(os.path.join(out_dir, "c1_output.incomplete"))


class _CountingEchoProcessor(_EchoProcessor):
    """记录 process 调用次数的 Echo processor。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process_calls = 0

    def process(self, input_manifest: str, output_dir: str, fingerprint_input: Optional[str] = None) -> str:
        self.process_calls += 1
        return super().process(input_manifest, output_dir, fingerprint_input=fingerprint_input)


def test_retryable_failure_marks_incomplete_and_rerun_reprocesses(tmp_path):
    """P1 回归语义（目录化契约）：可重试失败时写 .incomplete 未完成标记，
    重跑不得复用上次不完整产物，必须重新处理并写出 marker、清除未完成标记。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")

    # 第一次：processor 存在可重试失败 -> raise，不写 marker
    with pytest.raises(RuntimeError, match="可重试"):
        StageOrchestratorBase()._run_substage(
            _RetryableProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
        )
    incomplete = os.path.join(out_dir, "c1_output.incomplete")
    complete = os.path.join(out_dir, "c1_output.complete")
    assert os.path.isfile(incomplete)
    assert not os.path.exists(complete)

    # 第二次：换正常 processor 同输入重跑 -> 必须再次触发处理并写出 marker
    echo = _CountingEchoProcessor("C1")
    result = StageOrchestratorBase()._run_substage(
        echo, input_manifest, out_dir, prev_stage="B4"
    )
    assert echo.process_calls == 1, "重跑必须重新处理，不得复用 .incomplete 产物"
    assert result == out_dir
    assert os.path.isfile(complete)
    assert not os.path.exists(incomplete), "成功重跑后未完成标记必须清除"
    assert len(_read_dir_lines(out_dir)) == 2


class _FailingProcessor(Processor):
    """process 被调用即失败的 processor：用于断言幂等路径没有触发处理。"""

    def __init__(self, stage_name: str = "C1"):
        super().__init__({}, stage_name=stage_name)
        self.called = False

    def process(self, input_manifest: str, output_dir: str, fingerprint_input: Optional[str] = None) -> str:
        self.called = True
        raise AssertionError("幂等复用时不得调用 process")


def _write_input(tmp_path, lines) -> str:
    input_manifest = os.path.join(tmp_path, "input.jsonl")
    with open(input_manifest, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return input_manifest


_PASSED_LINE = json.dumps(
    _make_record("p1", stages={"B4": {"status": "passed", "timestamp": "t0"}}),
    ensure_ascii=False,
)
_SKIPPED_LINE = json.dumps(_make_record("s1"), ensure_ascii=False)


def test_substage_rerun_reuses_dir_output_with_marker(tmp_path):
    """目录化输出 marker 校验通过时，重跑直接复用目录，不再拆分与处理。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")

    result1 = StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
    )
    assert result1 == out_dir
    assert os.path.isfile(os.path.join(out_dir, "c1_output.complete"))

    failing = _FailingProcessor("C1")
    result2 = StageOrchestratorBase()._run_substage(
        failing, input_manifest, out_dir, prev_stage="B4"
    )
    assert not failing.called
    assert result2 == out_dir


def test_substage_legacy_merged_without_marker_triggers_reprocess(tmp_path):
    """无 marker 的旧 merged 无法证明配置与输入一致，必须全量重算。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    os.makedirs(out_dir)
    # 手工构造 legacy merged（新代码已不再写 merged）
    merged = os.path.join(out_dir, "c1_merged.jsonl")
    with open(merged, "w", encoding="utf-8") as f:
        f.write(_PASSED_LINE + "\n" + _SKIPPED_LINE + "\n")

    processor = _CountingEchoProcessor("C1")
    result = StageOrchestratorBase()._run_substage(
        processor, input_manifest, out_dir, prev_stage="B4"
    )

    assert processor.process_calls == 1
    assert result == out_dir
    assert not os.path.exists(merged)
    assert not os.path.exists(merged + ".complete")
    assert os.path.isfile(os.path.join(out_dir, "c1_output.complete"))


def test_substage_legacy_merged_with_valid_marker_is_reused(tmp_path):
    """输入、配置与输出指纹有效时，legacy merged 仍可正常 resume。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    os.makedirs(out_dir)
    merged = os.path.join(out_dir, "c1_merged.jsonl")
    with open(merged, "w", encoding="utf-8") as f:
        f.write(_PASSED_LINE + "\n" + _SKIPPED_LINE + "\n")
    processor = _FailingProcessor("C1")
    StageOrchestratorBase._write_marker(
        merged + ".complete", input_manifest, processor, 2, merged
    )

    result = StageOrchestratorBase()._run_substage(
        processor, input_manifest, out_dir, prev_stage="B4"
    )

    assert not processor.called
    assert result == merged
    with open(merged + ".complete", "r", encoding="utf-8") as f:
        marker = json.load(f)
    assert marker["schema"] == "stage_orchestrator_complete"
    assert marker["version"] == 4
    assert marker["output_kind"] == "merged_file"
    assert marker["content_digests"] == [
        {
            "name": "c1_merged.jsonl",
            "sha256": sha256_of(merged),
            "line_count": 2,
        }
    ]


def test_merged_marker_without_content_digest_triggers_reprocess(tmp_path):
    """旧 merged marker 缺 content digest 时一律不得复用。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    os.makedirs(out_dir)
    merged = os.path.join(out_dir, "c1_merged.jsonl")
    with open(merged, "w", encoding="utf-8") as f:
        f.write(_PASSED_LINE + "\n" + _SKIPPED_LINE + "\n")
    processor = _CountingEchoProcessor("C1")
    marker_path = merged + ".complete"
    StageOrchestratorBase._write_marker(
        marker_path, input_manifest, processor, 2, merged
    )
    with open(marker_path, "r", encoding="utf-8") as f:
        marker = json.load(f)
    marker.pop("content_digests")
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(marker, f, ensure_ascii=False, indent=2)

    result = StageOrchestratorBase()._run_substage(
        processor, input_manifest, out_dir, prev_stage="B4"
    )

    assert processor.process_calls == 1
    assert result == out_dir
    assert not os.path.exists(merged)


def test_merged_marker_rejects_invalid_json_output(tmp_path):
    """merged 内容变成非法 JSON 后 marker 必须失效并触发重算。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    os.makedirs(out_dir)
    merged = os.path.join(out_dir, "c1_merged.jsonl")
    with open(merged, "w", encoding="utf-8") as f:
        f.write(_PASSED_LINE + "\n" + _SKIPPED_LINE + "\n")
    processor = _CountingEchoProcessor("C1")
    StageOrchestratorBase._write_marker(
        merged + ".complete", input_manifest, processor, 2, merged
    )
    with open(merged, "w", encoding="utf-8") as f:
        f.write("{坏 JSON}\n")

    result = StageOrchestratorBase()._run_substage(
        processor, input_manifest, out_dir, prev_stage="B4"
    )

    assert processor.process_calls == 1
    assert result == out_dir
    assert not os.path.exists(merged)


def test_merged_marker_rejects_same_size_same_mtime_content_replacement(tmp_path):
    """merged 即使 size/mtime 未变，内容替换也必须使 marker 失效并重算。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    os.makedirs(out_dir)
    merged = os.path.join(out_dir, "c1_merged.jsonl")
    with open(merged, "w", encoding="utf-8") as f:
        f.write(_PASSED_LINE + "\n" + _SKIPPED_LINE + "\n")
    processor = _CountingEchoProcessor("C1")
    StageOrchestratorBase._write_marker(
        merged + ".complete", input_manifest, processor, 2, merged
    )

    old_stat = os.stat(merged)
    with open(merged, "rb") as f:
        original = f.read()
    replaced = original.replace(b'"record_id": "p1"', b'"record_id": "x1"', 1)
    assert replaced != original
    assert len(replaced) == len(original)
    with open(merged, "wb") as f:
        f.write(replaced)
    os.utime(merged, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))

    result = StageOrchestratorBase()._run_substage(
        processor, input_manifest, out_dir, prev_stage="B4"
    )

    assert processor.process_calls == 1
    assert result == out_dir
    assert not os.path.exists(merged)


def test_marker_rejects_invalid_dir_output_content(tmp_path):
    """即使 marker 匹配，目录输出含非法 JSON 时也必须重新处理。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
    )
    ds_file = os.path.join(out_dir, "c1_manifest_ds.jsonl")
    with open(ds_file, "a", encoding="utf-8") as f:
        f.write("{坏 JSON}\n")

    StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
    )
    raw_lines = _read_dir_lines(out_dir)
    assert len(raw_lines) == 2
    for line in raw_lines:
        json.loads(line)


def test_dir_marker_rejects_same_size_same_mtime_content_replacement(tmp_path):
    """目录产物即使 size/mtime 未变，内容替换也必须使 marker 失效并重算。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
    )

    ds_file = os.path.join(out_dir, "c1_manifest_ds.jsonl")
    old_stat = os.stat(ds_file)
    with open(ds_file, "rb") as f:
        original = f.read()
    replaced = original.replace(b'"record_id": "p1"', b'"record_id": "x1"', 1)
    assert replaced != original
    assert len(replaced) == len(original)
    with open(ds_file, "wb") as f:
        f.write(replaced)
    os.utime(ds_file, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))

    processor = _CountingEchoProcessor("C1")
    result = StageOrchestratorBase()._run_substage(
        processor, input_manifest, out_dir, prev_stage="B4"
    )

    assert processor.process_calls == 1
    assert result == out_dir
    assert {json.loads(line)["record_id"] for line in _read_dir_lines(out_dir)} == {"p1", "s1"}


def test_marker_resume_streams_content_digest(tmp_path, monkeypatch):
    """正常 marker resume 必须重算内容摘要后复用，不调用 processor。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")

    StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
    )
    marker_path = os.path.join(out_dir, "c1_output.complete")
    with open(marker_path, "r", encoding="utf-8") as f:
        marker = json.load(f)
    assert marker["schema"] == "stage_orchestrator_complete"
    assert marker["version"] == 4
    assert marker["output_kind"] == "per_dataset_dir"
    assert marker["content_digests"] == [
        {
            "name": "c1_manifest_ds.jsonl",
            "sha256": sha256_of(os.path.join(out_dir, "c1_manifest_ds.jsonl")),
            "line_count": 2,
        }
    ]

    digest_calls = []
    original = StageOrchestratorBase._content_digests

    def _spy(path):
        digest_calls.append(path)
        return original(path)

    monkeypatch.setattr(StageOrchestratorBase, "_content_digests", staticmethod(_spy))
    failing = _FailingProcessor("C1")
    result = StageOrchestratorBase()._run_substage(
        failing, input_manifest, out_dir, prev_stage="B4"
    )

    assert not failing.called
    assert result == out_dir
    assert digest_calls == [out_dir]


def test_dir_marker_content_digests_are_sorted_by_relative_filename(tmp_path):
    """目录 marker 对每个输出文件写 SHA-256，并按相对文件名稳定排序。"""
    class _PerDatasetEchoProcessor(Processor):
        def __init__(self):
            super().__init__({}, stage_name="C1")

        def process(self, input_manifest, output_dir, fingerprint_input=None):
            by_dataset = {}
            for rec in read_manifest(input_manifest):
                rec.mark_stage("C1", "passed")
                by_dataset.setdefault(rec.dataset_id, []).append(rec)
            for dataset_id, recs in by_dataset.items():
                write_manifest(
                    iter(recs),
                    os.path.join(output_dir, f"c1_manifest_{dataset_id}.jsonl"),
                )
            return output_dir

    lines = []
    for record_id, dataset_id in (("r-z", "z_ds"), ("r-a", "a_ds")):
        rec = _make_record(
            record_id,
            stages={"B4": {"status": "passed", "timestamp": "t0"}},
            dataset_id=dataset_id,
        )
        lines.append(json.dumps(rec, ensure_ascii=False))
    input_manifest = _write_input(tmp_path, lines)
    out_dir = os.path.join(str(tmp_path), "out")
    StageOrchestratorBase()._run_substage(
        _PerDatasetEchoProcessor(), input_manifest, out_dir, prev_stage="B4"
    )

    with open(os.path.join(out_dir, "c1_output.complete"), "r", encoding="utf-8") as f:
        marker = json.load(f)
    assert marker["content_digests"] == [
        {
            "name": name,
            "sha256": sha256_of(os.path.join(out_dir, name)),
            "line_count": 1,
        }
        for name in ("c1_manifest_a_ds.jsonl", "c1_manifest_z_ds.jsonl")
    ]


def test_marker_resume_ignores_mtime_only_change(tmp_path):
    """无害 touch 不改变 SHA-256，流式校验后仍应正常复用。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
    )
    ds_file = os.path.join(out_dir, "c1_manifest_ds.jsonl")
    st = os.stat(ds_file)
    os.utime(ds_file, (st.st_atime, st.st_mtime + 10))

    failing = _FailingProcessor("C1")
    result = StageOrchestratorBase()._run_substage(
        failing, input_manifest, out_dir, prev_stage="B4"
    )

    assert not failing.called
    assert result == out_dir


def test_marker_rejects_same_size_same_mtime_input_replacement(tmp_path):
    """输入同 size/mtime 改写也必须由 SHA-256 指纹触发重算。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
    )

    old_stat = os.stat(input_manifest)
    with open(input_manifest, "rb") as handle:
        original = handle.read()
    replaced = original.replace(b'"record_id": "p1"', b'"record_id": "x1"', 1)
    assert replaced != original
    assert len(replaced) == len(original)
    with open(input_manifest, "wb") as handle:
        handle.write(replaced)
    os.utime(input_manifest, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))

    processor = _CountingEchoProcessor("C1")
    StageOrchestratorBase()._run_substage(
        processor, input_manifest, out_dir, prev_stage="B4"
    )
    assert processor.process_calls == 1
    assert {json.loads(line)["record_id"] for line in _read_dir_lines(out_dir)} == {
        "x1",
        "s1",
    }


def test_marker_without_content_digest_triggers_reprocess(tmp_path):
    """旧 marker 即使其他字段完整，缺 content digest 也必须失效并重算。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
    )
    marker_path = os.path.join(out_dir, "c1_output.complete")
    with open(marker_path, "r", encoding="utf-8") as f:
        marker = json.load(f)
    marker.pop("content_digests")
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(marker, f, ensure_ascii=False, indent=2)

    processor = _CountingEchoProcessor("C1")
    result = StageOrchestratorBase()._run_substage(
        processor, input_manifest, out_dir, prev_stage="B4"
    )

    assert processor.process_calls == 1
    assert result == out_dir


def test_substage_stale_input_triggers_reprocess(tmp_path):
    """输入 manifest 变更后 marker 失效，必须重新拆分与处理。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")

    StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
    )

    # 输入追加一条记录（size/mtime 变化 → marker 失效）
    with open(input_manifest, "a", encoding="utf-8") as f:
        f.write(_SKIPPED_LINE + "\n")

    echo = _EchoProcessor("C1")
    result = StageOrchestratorBase()._run_substage(
        echo, input_manifest, out_dir, prev_stage="B4"
    )
    assert len(_read_dir_lines(result)) == 3


def test_substage_truncated_legacy_merged_triggers_reprocess(tmp_path):
    """无 marker 且 legacy merged 末行截断（崩溃写入）时回退到重新处理；
    重处理后产出目录化输出，legacy merged 被清除（防止下轮被兼容路径误复用）。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    os.makedirs(out_dir)
    merged = os.path.join(out_dir, "c1_merged.jsonl")
    with open(merged, "w", encoding="utf-8") as f:
        f.write(_PASSED_LINE + "\n" + _SKIPPED_LINE + "\n")
    # 模拟崩溃导致的截断：砍掉末尾 10 字节
    with open(merged, "r+b") as f:
        f.truncate(os.path.getsize(merged) - 10)

    echo = _EchoProcessor("C1")
    result = StageOrchestratorBase()._run_substage(
        echo, input_manifest, out_dir, prev_stage="B4"
    )
    # 重新处理后输出恢复完整（目录化契约），legacy merged 已清除
    assert result == out_dir
    assert not os.path.exists(merged)
    raw_lines = _read_dir_lines(out_dir)
    assert len(raw_lines) == 2
    for line in raw_lines:
        json.loads(line)


def test_stale_per_dataset_files_removed_on_reprocess(tmp_path):
    """重跑时未被 processor 重写的陈旧 per-dataset 文件必须删除：

    陈旧文件（旧运行残留、数据集不在本轮输入中）若保留，会随目录读取
    错误流入下游。
    """
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    os.makedirs(out_dir)
    # 旧运行残留：数据集 old_ds 不在本轮输入中
    stale = os.path.join(out_dir, "c1_manifest_old_ds.jsonl")
    with open(stale, "w", encoding="utf-8") as f:
        f.write(_SKIPPED_LINE + "\n")

    StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
    )
    assert not os.path.exists(stale)
    # 本轮输出只有 ds 一个数据集的 per-dataset 文件
    assert [n for n in os.listdir(out_dir) if n.endswith(".jsonl")] == ["c1_manifest_ds.jsonl"]


def test_processor_must_rewrite_existing_eligible_dataset_file(tmp_path):
    """processor 完全不写时，旧同数据集文件不得冒充本轮输出。"""
    class _NoWriteProcessor(Processor):
        def __init__(self):
            super().__init__({}, stage_name="C1")

        def process(self, input_manifest, output_dir, fingerprint_input=None):
            return output_dir

    input_manifest = _write_input(tmp_path, [_PASSED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    os.makedirs(out_dir)
    old_output = os.path.join(out_dir, "c1_manifest_ds.jsonl")
    with open(old_output, "w", encoding="utf-8") as f:
        f.write(_PASSED_LINE + "\n")

    with pytest.raises(RuntimeError, match="未重写"):
        StageOrchestratorBase()._run_substage(
            _NoWriteProcessor(), input_manifest, out_dir, prev_stage="B4"
        )

    assert not os.path.exists(os.path.join(out_dir, "c1_output.complete"))


def test_processor_same_count_in_place_output_without_stat_change_is_rejected(tmp_path):
    """仅原地覆写同 count、同 size 并恢复 mtime，不算可证明的本轮重写。"""
    class _StatPreservingProcessor(Processor):
        def __init__(self):
            super().__init__({}, stage_name="C1")

        def process(self, input_manifest, output_dir, fingerprint_input=None):
            out = os.path.join(output_dir, "c1_manifest_ds.jsonl")
            old_stat = os.stat(out)
            with open(out, "r+", encoding="utf-8") as f:
                content = f.read()
                f.seek(0)
                f.write(content)
                f.truncate()
            os.utime(out, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
            return output_dir

    input_manifest = _write_input(tmp_path, [_PASSED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    os.makedirs(out_dir)
    old_output = os.path.join(out_dir, "c1_manifest_ds.jsonl")
    with open(old_output, "w", encoding="utf-8") as f:
        f.write(_PASSED_LINE + "\n")

    with pytest.raises(RuntimeError, match="未重写"):
        StageOrchestratorBase()._run_substage(
            _StatPreservingProcessor(), input_manifest, out_dir, prev_stage="B4"
        )

    assert not os.path.exists(os.path.join(out_dir, "c1_output.complete"))


def test_dir_output_atomic_rewrite_preserves_hardlinked_final(tmp_path):
    """P1 回归语义（目录化契约）：per-dataset 文件被硬链接为 stage 根目录输出后，
    重跑不得破坏旧副本。

    processor 经 PerDatasetManifestWriter / write_manifest 重写时产生新 inode，
    硬链接仍指向旧 inode；ineligible 只追加到本轮新写的文件（inode 新鲜）。
    """
    from noise_filter_pipeline.core.manifest_io import link_or_copy_file

    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
    )
    ds_file = os.path.join(out_dir, "c1_manifest_ds.jsonl")
    final = os.path.join(str(tmp_path), "c_manifest_ds.jsonl")
    link_or_copy_file(ds_file, final)
    old_ino = os.stat(final).st_ino
    old_lines = sum(1 for _ in open(final, "rb"))

    # 输入追加一条并删掉 marker，强制完整重跑（split+process+汇总）
    with open(input_manifest, "a", encoding="utf-8") as f:
        f.write(_SKIPPED_LINE + "\n")
    os.remove(os.path.join(out_dir, "c1_output.complete"))
    StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
    )

    # 旧副本（硬链接）内容未被第二次处理破坏：仍是旧的 2 行、旧 inode
    assert os.stat(final).st_ino == old_ino
    assert sum(1 for _ in open(final, "rb")) == old_lines
    # 新输出是完整的新内容（3 行）且已是不同 inode
    assert sum(1 for _ in open(ds_file, "rb")) == 3
    assert os.stat(ds_file).st_ino != old_ino


def test_cleanup_intermediate_default_removes_split_files(tmp_path):
    """默认 cleanup_intermediate=True：汇总+marker 落盘后删除拆分中间产物；
    目录化契约下不再产生 merged 单文件。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
    )
    assert os.path.isfile(os.path.join(out_dir, "c1_manifest_ds.jsonl"))
    assert os.path.isfile(os.path.join(out_dir, "c1_output.complete"))
    assert not os.path.exists(os.path.join(out_dir, "c1_merged.jsonl"))
    assert not os.path.exists(os.path.join(out_dir, "c1_input_eligible.jsonl"))
    assert not os.path.exists(os.path.join(out_dir, "c1_input_ineligible.jsonl"))


def test_cleanup_intermediate_disabled_keeps_files(tmp_path):
    """cleanup_intermediate: false 时保留拆分中间产物。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE, _SKIPPED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1", {"cleanup_intermediate": False}),
        input_manifest, out_dir, prev_stage="B4",
    )
    assert os.path.isfile(os.path.join(out_dir, "c1_input_eligible.jsonl"))
    assert os.path.isfile(os.path.join(out_dir, "c1_input_ineligible.jsonl"))


def test_cleanup_intermediate_failure_keeps_files(tmp_path):
    """处理失败时不清理：拆分现场保留供排查/续跑。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    with pytest.raises(AssertionError):
        StageOrchestratorBase()._run_substage(
            _FailingProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
        )
    assert os.path.isfile(os.path.join(out_dir, "c1_input_eligible.jsonl"))


# ---------------------------------------------------------------------------
# 直接单测（code-review 修复）：dataset_id 兜底 + 陈旧清理独立方法
# ---------------------------------------------------------------------------
def test_extract_dataset_id_fast_path_and_fallback():
    """dataset_id 提取：正则快路径 / JSON 兜底 / 非法行归 unknown。

    兜底只处理正则未命中的场景（顶层 dataset_id 缺失、键序异常导致
    首个匹配非顶层）；嵌套字段先出现而正则命中嵌套值时仍取正则结果
    ——正常 AudioRecord 序列化 dataset_id 恒为首批字段，该偏差仅存在于
    第三方异常 manifest，与追加路径（_append_ineligible_per_dataset）口径一致。
    """
    from noise_filter_pipeline.core.stage_orchestrator import _extract_dataset_id
    import re

    ds_re = re.compile(r'"dataset_id":\s*"([^"]+)"')

    # 快路径：AudioRecord 序列化顺序，首个匹配即顶层
    assert _extract_dataset_id('{"record_id": "r", "dataset_id": "dsA", "stages": {}}', ds_re) == "dsA"
    # 嵌套 dataset_id 排在顶层之前：必须取 depth=1 顶层值
    nested_first = (
        '{"record_id":"r","labels":{"dataset_id":"wrong"},'
        '"dataset_id":"correct","stages":{}}'
    )
    assert _extract_dataset_id(nested_first, ds_re) == "correct"
    # 正则未命中但 JSON 可解析：顶层 dataset_id 缺失 → unknown
    assert _extract_dataset_id('{"record_id": "r", "stages": {}}', ds_re) == "unknown"
    # 非法 JSON → unknown
    assert _extract_dataset_id("not a json line", ds_re) == "unknown"
    # JSON 非 dict → unknown
    assert _extract_dataset_id('"just a string"', ds_re) == "unknown"


def test_prune_stale_substage_files_rules(tmp_path):
    """陈旧清理三规则：不在输入中删 / 零 eligible 删 / eligible 保留。"""
    from noise_filter_pipeline.core.stage_orchestrator import StageOrchestratorBase

    out_dir = str(tmp_path)
    names = {}
    for ds in ("dsA", "dsB", "dsC"):
        path = os.path.join(out_dir, f"c1_manifest_{ds}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{}\n")
        names[ds] = path
    # dsC 本轮由 processor 重写（新 inode），名字仍在 pre_run 快照里
    with open(names["dsC"], "w", encoding="utf-8") as f:
        f.write('{"dataset_id": "dsC"}\n')

    removed = StageOrchestratorBase._prune_stale_substage_files(
        out_dir, "C1",
        pre_run_names={f"c1_manifest_{ds}.jsonl" for ds in ("dsA", "dsB", "dsC")},
        input_datasets={"dsB", "dsC"},       # dsA 不在本轮输入
        eligible_datasets={"dsC"},           # dsB 零 eligible
    )
    assert removed == 2
    assert not os.path.exists(names["dsA"])  # 旧运行残留
    assert not os.path.exists(names["dsB"])  # 零 eligible（由 ineligible 追加重建）
    assert os.path.exists(names["dsC"])      # eligible > 0 必已被本轮重写


def test_prune_stale_substage_files_delete_failure_aborts(tmp_path, monkeypatch):
    """陈旧文件删不掉时必须中止，不能带着旧数据写完成 marker。"""
    from noise_filter_pipeline.core.stage_orchestrator import StageOrchestratorBase

    stale_name = "c1_manifest_old_ds.jsonl"
    stale_path = tmp_path / stale_name
    stale_path.write_text("{}\n")
    real_remove = os.remove

    def fail_stale_remove(path):
        if os.path.basename(path) == stale_name:
            raise PermissionError("read-only filesystem")
        return real_remove(path)

    monkeypatch.setattr(os, "remove", fail_stale_remove)
    with pytest.raises(RuntimeError, match="清理陈旧 per-dataset 文件失败"):
        StageOrchestratorBase._prune_stale_substage_files(
            str(tmp_path), "C1", {stale_name}, input_datasets=set(), eligible_datasets=set()
        )
    assert stale_path.exists()


def test_reprocess_failure_invalidates_old_complete_marker(tmp_path):
    """输入变化触发重处理后失败，旧 complete marker 必须已失效。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
    )
    marker = os.path.join(out_dir, "c1_output.complete")
    assert os.path.isfile(marker)

    passed2 = json.dumps(
        _make_record("p2", stages={"B4": {"status": "passed", "timestamp": "t0"}}),
        ensure_ascii=False,
    )
    with open(input_manifest, "a", encoding="utf-8") as f:
        f.write(passed2 + "\n")
    with pytest.raises(AssertionError):
        StageOrchestratorBase()._run_substage(
            _FailingProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
        )
    assert not os.path.exists(marker)


def test_append_ineligible_rejects_unsafe_dataset_id(tmp_path):
    """dataset_id 不得通过任一平台路径分隔符逃逸输出目录。"""
    for unsafe in ("../escaped", "..\\escaped"):
        ineligible = tmp_path / "ineligible.jsonl"
        rec = _make_record("r1")
        rec["dataset_id"] = unsafe
        ineligible.write_text(json.dumps(rec) + "\n")

        with pytest.raises(ValueError, match="dataset_id"):
            StageOrchestratorBase._append_ineligible_per_dataset(
                str(tmp_path / "out"), "C1", str(ineligible)
            )
    assert not (tmp_path / "c1_manifest_escaped.jsonl").exists()


def test_per_dataset_writer_rejects_unsafe_dataset_id(tmp_path):
    """eligible processor 的共享 writer 同样必须拒绝路径逃逸。"""
    from noise_filter_pipeline.core.manifest_per_dataset import PerDatasetManifestWriter

    rec = AudioRecord(record_id="r", dataset_id="../escaped", source_id="s")
    with pytest.raises(ValueError, match="dataset_id"):
        with PerDatasetManifestWriter(str(tmp_path / "out" / "c1_manifest.jsonl")) as writer:
            writer.append(rec)


def test_reprocess_aborts_if_old_complete_marker_cannot_be_removed(tmp_path, monkeypatch):
    """无法失效旧 complete marker 时必须在 processor 前中止。"""
    input_manifest = _write_input(tmp_path, [_PASSED_LINE])
    out_dir = os.path.join(str(tmp_path), "out")
    StageOrchestratorBase()._run_substage(
        _EchoProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
    )
    marker = os.path.join(out_dir, "c1_output.complete")
    with open(input_manifest, "a", encoding="utf-8") as f:
        f.write(_PASSED_LINE + "\n")
    real_remove = os.remove

    def fail_marker(path):
        if path == marker:
            raise PermissionError("read-only")
        return real_remove(path)

    monkeypatch.setattr(os, "remove", fail_marker)
    with pytest.raises(RuntimeError, match="完成 marker"):
        StageOrchestratorBase()._run_substage(
            _FailingProcessor("C1"), input_manifest, out_dir, prev_stage="B4"
        )


# ---------------------------------------------------------------------------
# Stage C orchestrator：固定三阶段链路 + 资产指纹注入
# ---------------------------------------------------------------------------


def _stage_c_all_noise_cfg(tmp_path):
    """最小 Stage C dry-run 配置；测试中三个 processor 均替换为 echo。"""
    return UnifiedPipelineConfig(
        output_root=str(tmp_path / "out"),
        stage_c={
            "c1": {"dry_run": True},
            "c2": {"mode": "dry_run", "label_dict": _REAL_LABEL_DICT},
            "c3": {},
        },
    )


_STAGE_C_INVALID_C2_CONTRACT_CASES = [
    ("output-key", {"output_label_key": "audioset_top5"}, "output_label_key"),
    ("topk-key", {"topk_output_key": "audioset_top5"}, "topk_output_key"),
    ("top-k", {"topk_keep_k": 5}, "topk_keep_k"),
    ("policy-in-c2", {"policy_path": "policy.json"}, "policy_path"),
    ("calibration-in-c2", {"calibration_path": "legacy.json"}, "calibration_path"),
]


@pytest.mark.parametrize(
    ("c2_overrides", "error_match"),
    [
        (c2_overrides, error_match)
        for _, c2_overrides, error_match in _STAGE_C_INVALID_C2_CONTRACT_CASES
    ],
    ids=[case_id for case_id, _, _ in _STAGE_C_INVALID_C2_CONTRACT_CASES],
)
def test_stage_c_invalid_c2_contract_fails_before_side_effects(
    tmp_path, c2_overrides, error_match
):
    """C2 非 canonical facts 配置必须在 orchestrator 构造时无副作用失败。"""
    c2_cfg = {
        "mode": "dry_run",
        "label_dict": _REAL_LABEL_DICT,
        **c2_overrides,
    }
    output_dir = tmp_path / "out" / "stage_c"

    with pytest.raises(ValueError, match=error_match):
        StageCOrchestrator(UnifiedPipelineConfig(stage_c={
            "c1": {"dry_run": True},
            "c2": c2_cfg,
            "c3": {},
        }))

    assert not output_dir.exists()


def test_stage_c_rejects_forged_input_before_c1(monkeypatch, tmp_path):
    """源输入预置 Stage C 状态或准入证据时必须中止，不得清理后重算。"""
    import noise_filter_pipeline.stages.stage_c_single_event as stage_c_mod

    processor_calls = []

    class _ObservedProcessor(_EchoProcessor):
        def process(self, input_manifest, output_dir, fingerprint_input=None):
            processor_calls.append(self.stage_name)
            return super().process(input_manifest, output_dir, fingerprint_input)

    monkeypatch.setattr(
        stage_c_mod,
        "QwenSingleEventProcessor",
        lambda config: _ObservedProcessor("C1", config),
    )
    monkeypatch.setattr(
        stage_c_mod,
        "AudioTagClassifierProcessor",
        lambda config: _ObservedProcessor("C2", config),
    )
    monkeypatch.setattr(
        stage_c_mod,
        "NoiseAdmissionProcessor",
        lambda config: _ObservedProcessor("C3", config),
    )

    forged = _make_record(
        "forged",
        stages={
            "B4": {"status": "passed", "source": "original"},
            "C1": {"status": "passed", "intelligibility": "noise"},
            "C4": {"status": "passed", "publishable": True},
        },
        labels={"original": "source-noise", **_c2_rain_result()},
        provenance={
            "config_hash": "upstream-config",
            "stage_c_execution": {
                "c1": "production",
                "c2": "subprocess_daemon",
            },
        },
    )
    manifest = _write_input(tmp_path, [json.dumps(forged, ensure_ascii=False)])
    stage_out = os.path.join(str(tmp_path), "out", "stage_c")

    with pytest.raises(ValueError, match=r"stages\.C1|stages\.C4|labels\.audioset_top5"):
        StageCOrchestrator(_stage_c_all_noise_cfg(tmp_path)).run(
            manifest, stage_out, os.path.join(stage_out, "c_manifest.jsonl")
        )

    assert processor_calls == []
    source = next(iter(read_manifest(manifest)))
    assert source.stages["C1"]["status"] == "passed"
    assert source.labels["audioset_top5"][0]["label"] == "Rain"
    assert source.provenance["stage_c_execution"]["c1"] == "production"
    assert not os.path.exists(os.path.join(stage_out, "C1", "c1_output.complete"))


def test_stage_c_orchestrator_trusted_marker_resume_skips_processors(monkeypatch, tmp_path):
    """同输入与配置的第二轮应复用本轮可信 marker，不破坏正常 resume。"""
    import noise_filter_pipeline.stages.stage_c_single_event as stage_c_mod

    processors = []

    def factory(stage_name):
        def build(config):
            processor = _CountingEchoProcessor(stage_name, config)
            processors.append(processor)
            return processor

        return build

    monkeypatch.setattr(stage_c_mod, "QwenSingleEventProcessor", factory("C1"))
    monkeypatch.setattr(stage_c_mod, "AudioTagClassifierProcessor", factory("C2"))
    monkeypatch.setattr(stage_c_mod, "NoiseAdmissionProcessor", factory("C3"))

    manifest = _write_input(tmp_path, [_PASSED_LINE])
    stage_out = os.path.join(str(tmp_path), "out", "stage_c")
    orchestrator = StageCOrchestrator(_stage_c_all_noise_cfg(tmp_path))
    first = orchestrator.run(
        manifest, stage_out, os.path.join(stage_out, "c_manifest.jsonl")
    )
    first_record = next(iter(read_manifest(os.path.join(first, "c_manifest_ds.jsonl"))))

    second = orchestrator.run(
        manifest, stage_out, os.path.join(stage_out, "c_manifest.jsonl")
    )
    second_record = next(iter(read_manifest(os.path.join(second, "c_manifest_ds.jsonl"))))

    assert len(processors) == 6
    assert [processor.process_calls for processor in processors[:3]] == [1, 1, 1]
    assert [processor.process_calls for processor in processors[3:]] == [0, 0, 0]
    assert second_record.to_dict() == first_record.to_dict()
    for stage_name in ("C1", "C2", "C3"):
        assert os.path.isfile(
            os.path.join(stage_out, stage_name, f"{stage_name.lower()}_output.complete")
        )


def test_stage_c_c1_processor_checkpoint_resume_survives_input_sanitization(
    monkeypatch, tmp_path
):
    """C1 incomplete 重启应从本轮可信 checkpoint 恢复，不重复模型推理。"""
    import noise_filter_pipeline.stages.stage_c_single_event as stage_c_mod
    from noise_filter_pipeline.processors.qwen_single_event import (
        QwenSingleEventProcessor,
    )

    c1_processors = []

    class _CountingCheckpointQwen(QwenSingleEventProcessor):
        def __init__(self, config):
            super().__init__(config)
            self.inference_records = 0

        def _dry_run_results(self, records):
            self.inference_records += len(records)
            return ["noise"] * len(records)

    def build_c1(config):
        processor = _CountingCheckpointQwen(config)
        c1_processors.append(processor)
        return processor

    monkeypatch.setattr(stage_c_mod, "QwenSingleEventProcessor", build_c1)
    monkeypatch.setattr(
        stage_c_mod,
        "AudioTagClassifierProcessor",
        lambda config: _EchoProcessor("C2", config),
    )
    monkeypatch.setattr(
        stage_c_mod,
        "NoiseAdmissionProcessor",
        lambda config: _EchoProcessor("C3", config),
    )

    manifest = _write_input(tmp_path, [_PASSED_LINE])
    stage_out = os.path.join(str(tmp_path), "out", "stage_c")
    orchestrator = StageCOrchestrator(_stage_c_all_noise_cfg(tmp_path))
    orchestrator.run(
        manifest, stage_out, os.path.join(stage_out, "c_manifest.jsonl")
    )
    incomplete = os.path.join(stage_out, "C1", "c1_output.incomplete")
    with open(incomplete, "w", encoding="utf-8") as f:
        f.write("simulate interrupted orchestrator\n")

    orchestrator.run(
        manifest, stage_out, os.path.join(stage_out, "c_manifest.jsonl")
    )

    assert [processor.inference_records for processor in c1_processors] == [1, 0]
    resumed = next(iter(read_manifest(os.path.join(stage_out, "C1"))))
    assert resumed.stages["C1"]["status"] == "passed"
    assert resumed.stages["C1"]["intelligibility"] == "noise"


@pytest.mark.parametrize(
    ("c1_cfg", "c2_cfg", "c3_cfg", "expected_status", "expected_reason"),
    [
        (
            {"dry_run": False},
            {"enable": True, "mode": "subprocess_daemon"},
            {"enable": False},
            "rejected",
            "c3_disabled",
        ),
        (
            {"dry_run": False},
            {"enable": False, "mode": "subprocess_daemon"},
            {"enable": True},
            "passed",
            "admitted_noise",
        ),
        (
            {"dry_run": True},
            {"enable": True, "mode": "subprocess_daemon"},
            {"enable": True},
            "passed",
            "admitted_noise",
        ),
    ],
    ids=["c3-disabled", "c2-disabled", "c1-dry-run"],
)
def test_stage_c_reprocesses_unmarked_fake_production_c3_merged(
    monkeypatch, tmp_path, c1_cfg, c2_cfg, c3_cfg, expected_status, expected_reason
):
    """无 marker 的伪 production C3 merged 不得绕过本轮 C1/C2/C3 执行。"""
    import noise_filter_pipeline.stages.stage_c_single_event as stage_c_mod
    from core.data.noise_admission_policy import load_admission_policy
    from noise_filter_pipeline.processors.noise_label_filter import (
        NoiseAdmissionProcessor,
    )

    policy = load_admission_policy()

    class _StageCEvidenceProcessor(_EchoProcessor):
        def process(self, input_manifest, output_dir, fingerprint_input=None):
            out = os.path.join(output_dir, f"{self.stage_name.lower()}_manifest_ds.jsonl")
            recs = list(read_manifest(input_manifest))
            mode = self.config["execution_mode"]
            for rec in recs:
                execution = dict(rec.provenance.get("stage_c_execution", {}))
                execution[self.stage_name.lower()] = mode
                rec.provenance["stage_c_execution"] = execution
                if self.stage_name == "C1":
                    rec.mark_stage(
                        "C1",
                        "passed",
                        extra={
                            "intelligibility": "noise",
                            **self.config["artifact_fingerprints"],
                        },
                    )
                else:
                    result = _c2_rain_result()
                    rec.labels.update({
                        "audioset_top5": result["audioset_top5"],
                    })
                    rec.mark_stage(
                        "C2",
                        "passed",
                        extra={
                            "evidence_contract_version": self.config[
                                "evidence_contract_version"
                            ],
                            "score_contract": self.config["score_contract"],
                            "class_count": self.config["class_count"],
                            "top_k": self.config["artifact_fingerprints"]["top_k"],
                            "label_space_sha256": self.config[
                                "artifact_fingerprints"
                            ]["audiotag_label_dict_sha256"],
                            "audiotag_checkpoint_sha256": self.config[
                                "artifact_fingerprints"
                            ]["audiotag_checkpoint_sha256"],
                        },
                    )
            write_manifest(iter(recs), out)
            return output_dir

    c3_processors = []

    class _CountingC3Processor(NoiseAdmissionProcessor):
        def __init__(self, config):
            super().__init__(config)
            self.process_calls = 0

        def process(self, input_manifest, output_dir, fingerprint_input=None):
            self.process_calls += 1
            return super().process(input_manifest, output_dir, fingerprint_input)

    monkeypatch.setattr(
        stage_c_mod,
        "QwenSingleEventProcessor",
        lambda config: _StageCEvidenceProcessor("C1", config),
    )
    monkeypatch.setattr(
        stage_c_mod,
        "AudioTagClassifierProcessor",
        lambda config: _StageCEvidenceProcessor("C2", config),
    )

    def build_c3(config):
        processor = _CountingC3Processor(config)
        c3_processors.append(processor)
        return processor

    monkeypatch.setattr(stage_c_mod, "NoiseAdmissionProcessor", build_c3)
    monkeypatch.setattr(
        StageCOrchestrator,
        "_compute_artifact_fingerprints",
        lambda _self: {
            "voice_policy_sha256": policy.sha256,
            "audiotag_label_dict_sha256": policy.label_space_sha256,
            "audiotag_checkpoint_sha256": "3" * 64,
            "qwen_model_sha256": "4" * 64,
        },
    )

    manifest = _write_input(tmp_path, [_PASSED_LINE])
    stage_out = os.path.join(str(tmp_path), "out", "stage_c")
    c3_dir = os.path.join(stage_out, "C3")
    os.makedirs(c3_dir)
    fake = _make_record(
        "p1",
        stages={
            "B4": {"status": "passed", "timestamp": "t0"},
            "C1": {"status": "passed", "intelligibility": "noise"},
            "C2": {"status": "passed"},
            "C3": {"status": "passed", "publishable": True},
        },
        provenance={
            "stage_c_execution": {"c1": "production", "c2": "subprocess_daemon"}
        },
    )
    with open(os.path.join(c3_dir, "c3_merged.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(fake, ensure_ascii=False) + "\n")

    config = UnifiedPipelineConfig(
        output_root=str(tmp_path / "out"),
        stage_c={
            "c1": c1_cfg,
            "c2": {**c2_cfg, "label_dict": _REAL_LABEL_DICT},
            "c3": dict(c3_cfg),
        },
    )
    result = StageCOrchestrator(config).run(
        manifest, stage_out, os.path.join(stage_out, "c_manifest.jsonl")
    )

    assert c3_processors[0].process_calls == 1
    assert not os.path.exists(os.path.join(c3_dir, "c3_merged.jsonl"))
    rec = next(iter(read_manifest(os.path.join(result, "c_manifest_ds.jsonl"))))
    assert rec.stages["C3"]["status"] == expected_status
    assert rec.stages["C3"]["reason"] == expected_reason
    assert rec.stages["C3"]["publishable"] is False


@pytest.mark.unit
def test_stage_c_run_lock_covers_all_processors_and_root_publish(monkeypatch, tmp_path):
    """整次 Stage C run 锁必须覆盖 C1-C3、根 manifest 与 stats 发布。"""
    import noise_filter_pipeline.stages.stage_c_single_event as stage_c_mod

    events = []
    locked = False
    registry = tmp_path / "lock-registry"
    monkeypatch.setenv("USE_TRAIN_LOCK_REGISTRY", str(registry))
    real_stage_c_run_lock = stage_c_mod.stage_c_run_lock

    @contextmanager
    def observed_lock(output_dir):
        nonlocal locked
        assert not os.path.exists(output_dir)
        with real_stage_c_run_lock(output_dir, timeout=0):
            locked = True
            events.append("lock-enter")
            try:
                yield
            finally:
                events.append("lock-exit")
                locked = False

    def factory(stage_name):
        assert locked
        events.append(f"build-{stage_name}")
        return _EchoProcessor(stage_name, {})

    real_link = stage_c_mod.link_stage_output
    real_stats = stage_c_mod.write_stage_stats

    def observed_link(*args, **kwargs):
        assert locked
        events.append("publish-manifest")
        return real_link(*args, **kwargs)

    def observed_stats(*args, **kwargs):
        assert locked
        events.append("publish-stats")
        return real_stats(*args, **kwargs)

    monkeypatch.setattr(stage_c_mod, "stage_c_run_lock", observed_lock)
    monkeypatch.setattr(stage_c_mod, "QwenSingleEventProcessor", lambda _cfg: factory("C1"))
    monkeypatch.setattr(stage_c_mod, "AudioTagClassifierProcessor", lambda _cfg: factory("C2"))
    monkeypatch.setattr(stage_c_mod, "NoiseAdmissionProcessor", lambda _cfg: factory("C3"))
    monkeypatch.setattr(stage_c_mod, "link_stage_output", observed_link)
    monkeypatch.setattr(stage_c_mod, "write_stage_stats", observed_stats)

    manifest = _write_input(tmp_path, [_PASSED_LINE])
    stage_out = os.path.join(str(tmp_path), "out", "stage_c")
    StageCOrchestrator(_stage_c_all_noise_cfg(tmp_path)).run(
        manifest, stage_out, os.path.join(stage_out, "c_manifest.jsonl")
    )

    assert events == [
        "lock-enter", "build-C1", "build-C2", "build-C3",
        "publish-manifest", "publish-stats", "lock-exit",
    ]


def test_stage_c_orchestrator_sequence(monkeypatch, tmp_path):
    import noise_filter_pipeline.stages.stage_c_single_event as stage_c_mod

    manifest = _write_input(tmp_path, [_PASSED_LINE])
    calls = []
    captured = {}

    def factory(name, stage_name):
        def build(config):
            calls.append(name)
            captured[name] = dict(config)
            return _EchoProcessor(stage_name, config)
        return build

    monkeypatch.setattr(stage_c_mod, "QwenSingleEventProcessor", factory("C1", "C1"))
    monkeypatch.setattr(stage_c_mod, "AudioTagClassifierProcessor", factory("C2", "C2"))
    monkeypatch.setattr(stage_c_mod, "NoiseAdmissionProcessor", factory("C3", "C3"))

    stage_out = os.path.join(str(tmp_path), "out", "stage_c")
    StageCOrchestrator(_stage_c_all_noise_cfg(tmp_path)).run(
        manifest, stage_out, os.path.join(stage_out, "c_manifest.jsonl")
    )

    assert calls == ["C1", "C2", "C3"]
    import hashlib as _h

    assert captured["C1"]["artifact_fingerprints"] == {
        "qwen_model_sha256": "unset:qwen_model_sha256",
        "c1_prompt_sha256": _h.sha256(
            captured["C1"]["prompt"].encode("utf-8")).hexdigest(),
    }
    assert captured["C1"]["input_sanitization"]["version"] == 4
    c2_identity = captured["C2"]["artifact_fingerprints"]
    assert c2_identity == {
        "audiotag_label_dict_sha256": _LABEL_METADATA.label_space_sha256,
        "audiotag_checkpoint_sha256": "unset:audiotag_checkpoint_sha256",
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "score_contract": SCORE_CONTRACT,
        "class_count": CLASSIFIER_LABEL_COUNT,
        "top_k": 5,
    }
    c3_fingerprints = captured["C3"]["artifact_fingerprints"]
    assert set(c3_fingerprints) == {
        "qwen_model_sha256",
        "c1_prompt_sha256",
        "audiotag_label_dict_sha256",
        "audiotag_checkpoint_sha256",
        "voice_policy_sha256",
    }
    assert captured["C3"]["c1_identity"] == captured["C1"]["artifact_fingerprints"]
    assert captured["C3"]["c2_identity"] == c2_identity
    assert os.path.isdir(os.path.join(stage_out, "C3"))
    assert not os.path.exists(os.path.join(stage_out, "C4"))
    rec = next(iter(read_manifest(os.path.join(stage_out, "c_manifest_ds.jsonl"))))
    assert rec.is_stage_passed("C3")
    assert "C4" not in rec.stages


def test_stage_c_c1_speech_terminal_record_does_not_enter_c2(monkeypatch, tmp_path):
    """C1 Speech 终态记录必须在编排层跳过 C2 AudioTag。"""
    import noise_filter_pipeline.stages.stage_c_single_event as stage_c_mod

    class _SpeechC1(_EchoProcessor):
        def process(self, input_manifest, output_dir, fingerprint_input=None):
            out = os.path.join(output_dir, "c1_manifest_ds.jsonl")
            recs = list(read_manifest(input_manifest))
            for rec in recs:
                rec.mark_stage(
                    "C1",
                    "rejected",
                    extra={"intelligibility": "speech", "reason": "speech"},
                )
            write_manifest(iter(recs), out)
            return output_dir

    c2_record_ids = []

    class _ObservedC2(_EchoProcessor):
        def process(self, input_manifest, output_dir, fingerprint_input=None):
            c2_record_ids.extend(rec.record_id for rec in read_manifest(input_manifest))
            return super().process(input_manifest, output_dir, fingerprint_input)

    monkeypatch.setattr(
        stage_c_mod,
        "QwenSingleEventProcessor",
        lambda config: _SpeechC1("C1", config),
    )
    monkeypatch.setattr(
        stage_c_mod,
        "AudioTagClassifierProcessor",
        lambda config: _ObservedC2("C2", config),
    )
    monkeypatch.setattr(
        stage_c_mod,
        "NoiseAdmissionProcessor",
        lambda config: _EchoProcessor("C3", config),
    )

    manifest = _write_input(tmp_path, [_PASSED_LINE])
    stage_out = os.path.join(str(tmp_path), "out", "stage_c")
    StageCOrchestrator(_stage_c_all_noise_cfg(tmp_path)).run(
        manifest, stage_out, os.path.join(stage_out, "c_manifest.jsonl")
    )

    rec = next(iter(read_manifest(os.path.join(stage_out, "c_manifest_ds.jsonl"))))
    assert c2_record_ids == []
    assert rec.stages["C1"]["intelligibility"] == "speech"
    assert rec.stages["C2"]["status"] == "skipped"
    assert rec.stages["C2"]["reason"] == "prev_stage_C1_not_passed"


def test_stage_c_artifact_fingerprints_follow_substage_boundaries(monkeypatch, tmp_path):
    """C1/C2 绑定生成 identity，C3 独立绑定五项发布 identity。"""
    import core.data.noise_admission as noise_admission
    import noise_filter_pipeline.stages.stage_c_single_event as stage_c_mod

    from core.data.noise_admission import canonical_voice_policy_path

    policy = canonical_voice_policy_path()
    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_text("weights", encoding="utf-8")
    ckpt = tmp_path / "pretrained.pt"
    ckpt.write_text("ckpt", encoding="utf-8")
    label_dict = tmp_path / "class_labels_indices.csv"
    label_dict.write_text("index,mid,display_name\n", encoding="utf-8")

    # 指纹走确定性 stub：不依赖真实资产大小，且可验证「哪个资产进了哪个键」
    real_sha256_file = noise_admission.sha256_file
    real_sha256_path = noise_admission.sha256_path
    monkeypatch.setattr(noise_admission, "sha256_file", lambda p: "f:" + os.path.basename(p))
    monkeypatch.setattr(noise_admission, "sha256_path", lambda p: "d:" + os.path.basename(p))

    captured = {}

    def capture(name):
        def factory(config):
            captured[name] = dict(config)
            return _EchoProcessor(name, config)

        return factory

    for name in (
        "QwenSingleEventProcessor",
        "AudioTagClassifierProcessor",
        "NoiseAdmissionProcessor",
    ):
        monkeypatch.setattr(stage_c_mod, name, capture(name))

    manifest = _write_input(tmp_path, [_PASSED_LINE])
    stage_out = os.path.join(str(tmp_path), "out", "stage_c")
    cfg = UnifiedPipelineConfig(
        output_root=str(tmp_path / "out"),
        stage_c={
            "c1": {"model_path": str(model_dir), "dry_run": True},
            "c2": {
                "mode": "dry_run",
                "checkpoint": str(ckpt),
                "label_dict": str(label_dict),
                "dry_run_labels": ["Rain"],
            },
            "c3": {"policy_path": str(policy)},
        },
    )
    orch = StageCOrchestrator(cfg)
    orch.run(
        input_manifest=manifest,
        output_dir=stage_out,
        output_manifest=os.path.join(stage_out, "c_manifest.jsonl"),
    )

    assert set(captured) == {
        "QwenSingleEventProcessor",
        "AudioTagClassifierProcessor",
        "NoiseAdmissionProcessor",
    }
    import hashlib as _h

    assert captured["QwenSingleEventProcessor"]["artifact_fingerprints"] == {
        "qwen_model_sha256": "d:qwen",
        "c1_prompt_sha256": _h.sha256(
            captured["QwenSingleEventProcessor"]["prompt"].encode("utf-8")).hexdigest(),
    }
    c2_identity = captured["AudioTagClassifierProcessor"]["artifact_fingerprints"]
    assert c2_identity == {
        "audiotag_label_dict_sha256": "f:class_labels_indices.csv",
        "audiotag_checkpoint_sha256": "f:pretrained.pt",
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "score_contract": SCORE_CONTRACT,
        "class_count": CLASSIFIER_LABEL_COUNT,
        "top_k": 5,
    }
    c3 = captured["NoiseAdmissionProcessor"]
    c3_fingerprints = c3["artifact_fingerprints"]
    assert c3_fingerprints == {
        "qwen_model_sha256": "d:qwen",
        "c1_prompt_sha256": captured["QwenSingleEventProcessor"][
            "artifact_fingerprints"
        ]["c1_prompt_sha256"],
        "audiotag_label_dict_sha256": "f:class_labels_indices.csv",
        "audiotag_checkpoint_sha256": "f:pretrained.pt",
        "voice_policy_sha256": "f:audiotag_voice_policy.json",
    }
    assert c3["c1_identity"] == captured["QwenSingleEventProcessor"][
        "artifact_fingerprints"
    ]
    assert c3["c2_identity"] == c2_identity
    assert "calibration_identity" not in captured["QwenSingleEventProcessor"]
    assert "calibration_identity" not in captured["AudioTagClassifierProcessor"]
    assert "calibration_identity" not in c3

    # 内容变化 → 指纹变化 → checkpoint config hash 失效（路径不变也不复用旧断点）。
    # 先恢复真实哈希实现，再验证真实内容 hash 跟随文件内容变化
    monkeypatch.setattr(noise_admission, "sha256_file", real_sha256_file)
    monkeypatch.setattr(noise_admission, "sha256_path", real_sha256_path)
    fingerprints_after = orch._compute_artifact_fingerprints()
    assert fingerprints_after["voice_policy_sha256"] == sha256_of(policy)
    assert fingerprints_after["voice_policy_sha256"] != "f:audiotag_voice_policy.json"


def test_stage_c_c1_checkpoint_fingerprint_tracks_only_qwen_content(monkeypatch, tmp_path):
    """policy/AudioTag 内容不进入 C1 identity，Qwen 内容变化使 C1 失效。"""
    import noise_filter_pipeline.stages.stage_c_single_event as stage_c_mod
    from noise_filter_pipeline.core.manifest_io import build_checkpoint_fingerprint

    from core.data.noise_admission import canonical_voice_policy_path

    policy = canonical_voice_policy_path()
    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    model_file = model_dir / "model.safetensors"
    model_file.write_text("weights v1", encoding="utf-8")
    ckpt = tmp_path / "pretrained.pt"
    ckpt.write_text("ckpt v1", encoding="utf-8")
    label_dict = tmp_path / "class_labels_indices.csv"
    label_dict.write_text("labels v1", encoding="utf-8")

    # sha256_path 在生产代码中按路径做进程内缓存；这里用无缓存替身隔离验证
    # orchestrator 的注入语义，等价于资产变更后重新启动 pipeline 进程。
    monkeypatch.setattr(
        "core.data.noise_admission.sha256_path",
        lambda path: sha256_of(os.path.join(path, "model.safetensors")),
    )

    c1_configs = []

    def capture_c1(config):
        c1_configs.append(dict(config))
        return _EchoProcessor("C1", config)

    monkeypatch.setattr(stage_c_mod, "QwenSingleEventProcessor", capture_c1)
    monkeypatch.setattr(
        stage_c_mod,
        "AudioTagClassifierProcessor",
        lambda config: _EchoProcessor("C2", config),
    )
    monkeypatch.setattr(
        stage_c_mod,
        "NoiseAdmissionProcessor",
        lambda config: _EchoProcessor("C3", config),
    )

    manifest = _write_input(tmp_path, [_PASSED_LINE])
    cfg = UnifiedPipelineConfig(
        output_root=str(tmp_path / "out"),
        stage_c={
            "c1": {"model_path": str(model_dir)},
            "c2": {"checkpoint": str(ckpt), "label_dict": str(label_dict)},
            "c3": {"policy_path": str(policy)},
        },
    )
    orch = StageCOrchestrator(cfg)

    def run_and_c1_fingerprint(run_name):
        stage_out = os.path.join(str(tmp_path), "out", run_name)
        orch.run(manifest, stage_out, os.path.join(stage_out, "c_manifest.jsonl"))
        c1_config = c1_configs[-1]
        assert set(c1_config["artifact_fingerprints"]) == {"qwen_model_sha256", "c1_prompt_sha256"}
        return build_checkpoint_fingerprint(c1_config, manifest)

    initial = run_and_c1_fingerprint("initial")

    ckpt.write_text("ckpt v2", encoding="utf-8")
    label_dict.write_text("labels v2", encoding="utf-8")
    assert run_and_c1_fingerprint("policy_audiotag_changed") == initial

    model_file.write_text("weights v2", encoding="utf-8")
    assert run_and_c1_fingerprint("qwen_changed") != initial


def sha256_of(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_stage_c_fingerprints_unset_placeholder_when_assets_are_optional():
    """C1 dry-run + C2 disabled 时非必需模型资产使用确定性 unset 占位。"""
    orch = StageCOrchestrator(UnifiedPipelineConfig(
        stage_c={
            "c1": {"dry_run": True},
            "c2": {"enable": False, "mode": "dry_run"},
        }
    ))
    fingerprints = orch._compute_artifact_fingerprints()
    assert set(fingerprints) == {
        "voice_policy_sha256",
        "audiotag_label_dict_sha256",
        "audiotag_checkpoint_sha256",
        "qwen_model_sha256",
    }
    from core.data.noise_admission import canonical_voice_policy_path
    from core.data.noise_admission import sha256_file as real_hash

    assert fingerprints["voice_policy_sha256"] == real_hash(canonical_voice_policy_path())
    assert fingerprints["audiotag_checkpoint_sha256"] == "unset:audiotag_checkpoint_sha256"
    assert fingerprints["audiotag_label_dict_sha256"] == "unset:audiotag_label_dict_sha256"
    assert fingerprints["qwen_model_sha256"] == "unset:qwen_model_sha256"


def test_stage_c_production_missing_assets_fail_before_side_effects(monkeypatch, tmp_path):
    """正式运行缺失项须聚合报错，且早于输出、processor 构造和内容 hash。"""
    import core.data.noise_admission as noise_admission
    import noise_filter_pipeline.stages.stage_c_single_event as stage_c_mod

    missing_root = tmp_path / "missing"
    cfg = UnifiedPipelineConfig(stage_c={
        "c1": {"dry_run": False, "model_path": str(missing_root / "qwen")},
        "c2": {
            "enable": True,
            "mode": "subprocess_daemon",
            "checkpoint": str(missing_root / "audiotag.pt"),
            "label_dict": str(missing_root / "labels.csv"),
        },
        "c3": {},
    })
    processor_calls = []

    def fail_processor(_config):
        processor_calls.append(True)
        raise AssertionError("必需资产检查失败时不得构造 processor")

    for name in (
        "QwenSingleEventProcessor",
        "AudioTagClassifierProcessor",
        "NoiseAdmissionProcessor",
    ):
        monkeypatch.setattr(stage_c_mod, name, fail_processor)
    monkeypatch.setattr(
        noise_admission,
        "sha256_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("缺失检查前不得计算文件 hash")),
    )
    monkeypatch.setattr(
        noise_admission,
        "sha256_path",
        lambda _path: (_ for _ in ()).throw(AssertionError("缺失检查前不得计算目录 hash")),
    )

    output_dir = tmp_path / "out" / "stage_c"
    with pytest.raises(FileNotFoundError) as exc_info:
        StageCOrchestrator(cfg).run(
            input_manifest=str(tmp_path / "input.jsonl"),
            output_dir=str(output_dir),
            output_manifest=str(output_dir / "c_manifest.jsonl"),
        )

    message = str(exc_info.value)
    for asset in (
        "C1 Qwen model",
        "C2 AudioTag checkpoint",
        "C2 AudioTag label dict",
    ):
        assert asset in message
    assert not output_dir.exists()
    assert processor_calls == []


@pytest.mark.parametrize(
    ("c1_cfg", "c2_cfg", "expected_unset"),
    [
        (
            {"dry_run": True},
            {"enable": True, "mode": "subprocess_daemon", "with_assets": True},
            {"qwen_model_sha256"},
        ),
        (
            {"dry_run": False, "with_model": True},
            {"enable": False, "mode": "subprocess_daemon"},
            {"audiotag_checkpoint_sha256", "audiotag_label_dict_sha256"},
        ),
        (
            {"dry_run": False, "with_model": True},
            {"enable": True, "mode": "dry_run", "with_label_dict": True},
            {"audiotag_checkpoint_sha256"},
        ),
    ],
    ids=["c1-dry-c2-production", "c1-production-c2-disabled", "c1-production-c2-dry"],
)
def test_stage_c_required_assets_follow_independent_c1_c2_modes(
    tmp_path, c1_cfg, c2_cfg, expected_unset
):
    """C1/C2 模式独立决定模型资产是否必需，禁用分支不误触发校验。"""
    c1_cfg = dict(c1_cfg)
    c2_cfg = dict(c2_cfg)
    if c1_cfg.pop("with_model", False):
        model_dir = tmp_path / "qwen"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_text("weights", encoding="utf-8")
        c1_cfg["model_path"] = str(model_dir)
    if c2_cfg.pop("with_assets", False):
        checkpoint = tmp_path / "audiotag.pt"
        checkpoint.write_text("checkpoint", encoding="utf-8")
        label_dict = tmp_path / "labels.csv"
        label_dict.write_text("labels", encoding="utf-8")
        c2_cfg.update(checkpoint=str(checkpoint), label_dict=str(label_dict))
    if c2_cfg.pop("with_label_dict", False):
        c2_cfg["label_dict"] = _REAL_LABEL_DICT

    fingerprints = StageCOrchestrator(UnifiedPipelineConfig(
        stage_c={
            "c1": c1_cfg,
            "c2": c2_cfg,
            "c3": {},
        }
    ))._compute_artifact_fingerprints()

    actual_unset = {key for key, value in fingerprints.items() if value.startswith("unset:")}
    assert actual_unset == expected_unset


def test_stage_c_c2_production_assets_required_even_when_c1_is_dry_run(tmp_path):
    """混合模式中 C1 dry-run 不得放宽已启用 C2 subprocess_daemon 的资产要求。"""
    orch = StageCOrchestrator(UnifiedPipelineConfig(stage_c={
        "c1": {"dry_run": True},
        "c2": {"enable": True, "mode": "subprocess_daemon"},
        "c3": {},
    }))
    with pytest.raises(FileNotFoundError) as exc_info:
        orch._compute_artifact_fingerprints()
    message = str(exc_info.value)
    assert "C2 AudioTag checkpoint" in message
    assert "C2 AudioTag label dict" in message
    assert "C1 Qwen model" not in message
