"""
Tests for Stage C single-event mining components (dry-run / lightweight)。
"""
import hashlib
import json
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf
import yaml

from noise_filter_pipeline.core.execution_contract import build_stage_c_execution_contract
from noise_filter_pipeline.core.llm_cache import LLMResultCache
from noise_filter_pipeline.core.manifest_io import AudioRecord, write_manifest, read_manifest
from noise_filter_pipeline.core.pipeline_config import UnifiedPipelineConfig
from noise_filter_pipeline.processors.qwen_single_event import QwenSingleEventProcessor
from noise_filter_pipeline.processors.audiotag_classifier import AudioTagClassifierProcessor
from noise_filter_pipeline.processors.noise_label_filter import NoiseAdmissionProcessor
from noise_filter_pipeline.stages.stage_c_single_event import StageCOrchestrator
from core.data.noise_admission_policy import (
    ADMISSION_POLICY_VERSION,
    CLASSIFIER_LABEL_COUNT,
    EVIDENCE_CONTRACT_VERSION,
    SCORE_CONTRACT,
    load_admission_policy,
    validate_admission_contract,
)
from third_party_scripts.icefall_audiotaq.audiotag_result import (
    build_dry_run_result,
    format_audiotag_results,
    load_label_metadata as load_audiotag_label_metadata,
)

# Stage C 涉及 Qwen/AudioTag 外部模型与 daemon，归类为 pipeline 外部集成测试
pytestmark = [pytest.mark.pipeline, pytest.mark.integration, pytest.mark.external]


PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_YAML = os.path.join(PIPELINE_ROOT, "config", "pipeline.yaml")

POLICY_REL_PATH = "tools/noise_filter_pipeline/config/ontology/audiotag_voice_policy.json"
_REAL_LABEL_DICT = os.path.join(
    os.path.dirname(os.path.dirname(PIPELINE_ROOT)),
    "third_party_scripts",
    "icefall_audiotaq",
    "data",
    "class_labels_indices.csv",
)
_AUDIOTAG_METADATA = load_audiotag_label_metadata(_REAL_LABEL_DICT)


def _audiotag_labels(top_label: str | None, score_overrides=None) -> dict:
    """从完整 527 内部排序构造唯一持久化的固定 top-5 C2 事实。"""
    scores = [0.0] * CLASSIFIER_LABEL_COUNT
    # 用五个 canonical noise MID 填满 top-5，避免零分并列时低 index speech MID
    # 进入 fixture；调用方覆盖的标签仍按分数参与真实全 527 排序。
    for rank, label in enumerate(("Rain", "Music", "Wind", "Thunder", "Water")):
        index = _AUDIOTAG_METADATA.index_of(_AUDIOTAG_METADATA.label_to_mid[label])
        scores[index] = 0.05 - rank * 0.01
    if top_label is not None:
        scores[
            _AUDIOTAG_METADATA.index_of(_AUDIOTAG_METADATA.label_to_mid[top_label])
        ] = 1.0
    for label, prob in (score_overrides or {}).items():
        index = _AUDIOTAG_METADATA.index_of(_AUDIOTAG_METADATA.label_to_mid[label])
        scores[index] = float(prob)
    result = format_audiotag_results([scores], _AUDIOTAG_METADATA)[0]
    return {"audioset_top5": result["audioset_top5"]}


def _make_wav(path, duration=2.0, sr=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wav = np.random.randn(int(duration * sr)) * 0.01
    sf.write(path, wav, sr)


def _record_with_audio(tmpdir, idx, labels, stage_status=None):
    wav_path = os.path.join(tmpdir, f"audio_{idx}.wav")
    _make_wav(wav_path)
    rec = AudioRecord(
        record_id=f"ds__src__{idx}",
        dataset_id="ds",
        source_id="src",
        chunk_id=f"chunk_{idx}",
        audio={"path": wav_path, "sr": 16000, "duration": 2.0, "channels": 1},
        labels=labels,
        provenance={
            "stage_c_execution": {
                "c1": "production",
                "c2": "subprocess_daemon",
            }
        },
    )
    if stage_status:
        for stage, status in stage_status.items():
            extra = _c1_identity() if stage == "C1" else None
            rec.mark_stage(stage, status, extra=extra)
    return rec


def test_llm_cache_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = LLMResultCache(cache_dir=tmpdir, model_name="test")
        cache.set("/tmp/a.wav", "prompt", {"single_event": True})
        assert cache.get("/tmp/a.wav", "prompt") == {"single_event": True}
        assert cache.get("/tmp/a.wav", "other") is None


def test_qwen_single_event_dry_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [_record_with_audio(tmpdir, i, {"original": "noise"}) for i in range(4)]
        write_manifest(iter(recs), manifest)

        proc = QwenSingleEventProcessor(
            {"dry_run": True, "dry_run_distribution": {"noise": 0.5, "babble": 0.3, "speech": 0.2}, "batch_size": 2}
        )
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "c1"))
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 4
        # dry_run 结果确定但比例是随机的；至少保证记录状态被正确写入
        assert all("C1" in r.stages for r in out_recs)


def test_qwen_single_event_dry_run_deterministic_and_no_global_pollution():
    """dry_run 抽样必须可复现，且不得扰动全局 random 状态。"""
    import random as random_mod

    with tempfile.TemporaryDirectory() as tmpdir:
        recs = [_record_with_audio(tmpdir, i, {"original": "noise"}) for i in range(5)]
        proc = QwenSingleEventProcessor(
            {"dry_run": True, "dry_run_distribution": {"noise": 0.5, "babble": 0.3, "speech": 0.2}}
        )

        r1 = proc._dry_run_results(recs)
        # 相同输入必须可复现（不依赖按进程随机化的内置 hash()）
        assert proc._dry_run_results(recs) == r1
        # 全局 random 序列不得被 dry_run 扰动
        random_mod.seed(12345)
        expected_next = random_mod.random()
        random_mod.seed(12345)
        proc._dry_run_results(recs)
        assert random_mod.random() == expected_next


def test_qwen_single_event_default_prompt_defers_admission_to_c3():
    """C1 只产出候选类别，不得把 Babble/Noise 写成最终准入结论。"""
    from noise_filter_pipeline.processors.qwen_single_event import DEFAULT_PROMPT

    prompt = DEFAULT_PROMPT.lower()
    assert "c1 only produces candidates" in prompt
    assert "final admission" in prompt
    assert "downstream c3 admission policy" in prompt
    assert "- speech:" in prompt
    assert "must be rejected" in prompt
    assert "- babble candidate:" in prompt
    assert "babble is not final admission" in prompt
    assert "- noise candidate:" in prompt
    assert "noise is not final admission" in prompt
    for vocalization in (
        "breathing", "wheezing", "snoring", "gasping", "panting", "snorting",
        "sniffing", "coughing", "throat clearing", "sneezing", "hiccuping",
        "burping", "chewing", "gargling", "whistling", "yawning", "sighing",
        "groaning", "grunting",
    ):
        assert vocalization in prompt


def test_real_pipeline_yaml_uses_single_default_c1_prompt(monkeypatch, tmp_path):
    """生产配置不复制 prompt；编排器注入默认值并对同一内容取指纹。"""
    import hashlib
    import noise_filter_pipeline.stages.stage_c_single_event as stage_c_mod
    from noise_filter_pipeline.processors.qwen_single_event import DEFAULT_PROMPT

    config = UnifiedPipelineConfig.from_yaml(PIPELINE_YAML, active_stages=())
    assert "prompt" not in config.stage_c["c1"]

    orchestrator = StageCOrchestrator(config)
    monkeypatch.setattr(
        orchestrator,
        "_compute_artifact_fingerprints",
        lambda: {
            "voice_policy_sha256": "0" * 64,
            "audiotag_label_dict_sha256": "0" * 64,
            "audiotag_checkpoint_sha256": "0" * 64,
            "qwen_model_sha256": "0" * 64,
        },
    )
    captured = {}

    class C1ConfigCaptured(Exception):
        pass

    def capture_c1(config):
        captured.update(config)
        raise C1ConfigCaptured

    monkeypatch.setattr(stage_c_mod, "QwenSingleEventProcessor", capture_c1)
    with pytest.raises(C1ConfigCaptured):
        orchestrator.run(
            input_manifest="unused.jsonl",
            output_dir=str(tmp_path / "stage_c"),
            output_manifest=str(tmp_path / "stage_c" / "c_manifest.jsonl"),
        )

    expected_sha256 = hashlib.sha256(DEFAULT_PROMPT.encode("utf-8")).hexdigest()
    assert captured["prompt"] == DEFAULT_PROMPT
    assert captured["artifact_fingerprints"]["c1_prompt_sha256"] == expected_sha256


def test_qwen_single_event_parse_responses_requires_complete_label():
    """仅接受完整单标签；解释文本和复合结果必须 fail-closed。"""
    from noise_filter_pipeline.processors.qwen_single_event import UNPARSEABLE

    proc = QwenSingleEventProcessor({})
    texts = [
        # 合法：大小写、首尾空白、外层引号、单个句末标点。
        "<|im_start|>assistant\nSpeech",
        "<|im_start|>assistant\n  bAbBlE.  ",
        '<|im_start|>assistant\n"Noise"',
        "<|im_start|>assistant\n'Noise'?",
        '<|im_start|>assistant\n"Speech!"',
        # 危险输出：不得因标签出现在开头或句中而误收。
        "<|im_start|>assistant\nNoise and Speech",
        "<|im_start|>assistant\nProbably Noise",
        "<|im_start|>assistant\nNoise because no voice is audible",
        "<|im_start|>assistant\nSpeech or Babble",
        "<|im_start|>assistant\nNoise..",
        "<|im_start|>assistant\nUnknown",
        "not a valid reply",
        None,
    ]
    results = proc.parse_responses(texts)
    # 有原始输出但无法解析 → UNPARSEABLE（内容级终态）；
    # 无原始输出（None）→ None（基础设施失败，可重试）
    assert results == [
        "speech", "babble", "noise", "noise", "speech",
        UNPARSEABLE, UNPARSEABLE, UNPARSEABLE, UNPARSEABLE, UNPARSEABLE,
        UNPARSEABLE, UNPARSEABLE, None,
    ]


def test_qwen_single_event_babble_is_candidate_and_speech_is_rejected():
    """C1 Babble 仅作为候选下传，Speech 在 C1 终态拒绝。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [_record_with_audio(tmpdir, i, {"original": "noise"}) for i in range(2)]
        write_manifest(iter(recs), manifest)

        proc = QwenSingleEventProcessor({})
        proc.process_records = lambda pending: ["babble", "speech"][: len(pending)]
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "c1"))
        out_recs = {r.record_id: r for r in read_manifest(out_manifest)}

        babble = out_recs["ds__src__0"]
        assert babble.is_stage_passed("C1")
        assert babble.stages["C1"]["intelligibility"] == "babble"
        assert babble.rejected_reason is None

        speech = out_recs["ds__src__1"]
        assert speech.stages["C1"]["status"] == "rejected"
        assert speech.stages["C1"]["intelligibility"] == "speech"
        assert speech.rejected_reason == "c1_speech"


def test_qwen_single_event_failure_classification():
    """parse 失败为终态（rejected + checkpoint），infra 失败可重试（failed、无 checkpoint）。"""
    from noise_filter_pipeline.processors.qwen_single_event import UNPARSEABLE

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [_record_with_audio(tmpdir, i, {"original": "noise"}) for i in range(4)]
        write_manifest(iter(recs), manifest)

        proc = QwenSingleEventProcessor({})
        # 模拟推理结果：noise / speech / 不可解析 / 基础设施失败
        proc.process_records = lambda pending: ["noise", "speech", UNPARSEABLE, None][: len(pending)]
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "c1"))
        out_recs = {r.record_id: r for r in read_manifest(out_manifest)}
        assert len(out_recs) == 4

        assert out_recs["ds__src__0"].is_stage_passed("C1")
        assert out_recs["ds__src__1"].stages["C1"]["status"] == "rejected"
        assert out_recs["ds__src__1"].rejected_reason == "c1_speech"
        # 不可解析输出：终态 rejected，进入 checkpoint（resume 不再重试）
        assert out_recs["ds__src__2"].stages["C1"]["status"] == "rejected"
        assert out_recs["ds__src__2"].rejected_reason == "c1_parse_failed"
        # 基础设施失败：status=failed（非 rejected），不进 checkpoint
        assert out_recs["ds__src__3"].stages["C1"]["status"] == "failed"
        assert out_recs["ds__src__3"].stages["C1"]["reason"] == "inference_failed"
        assert out_recs["ds__src__3"].rejected_reason is None

        import json

        with open(os.path.join(tmpdir, "c1", "checkpoint.json"), encoding="utf-8") as f:
            done = set(json.load(f)["done"])
        assert done == {"ds__src__0", "ds__src__1", "ds__src__2"}

        # resume：infra 失败记录应被重新处理（本轮恢复为 noise），
        # parse 失败记录不再重算（从上次输出还原 rejected 状态）
        proc2 = QwenSingleEventProcessor({})
        seen_ids = []

        def fake_process_records(pending):
            seen_ids.extend(r.record_id for r in pending)
            return ["noise"] * len(pending)

        proc2.process_records = fake_process_records
        out_manifest2 = proc2.process(manifest, os.path.join(tmpdir, "c1"))
        out_recs2 = {r.record_id: r for r in read_manifest(out_manifest2)}
        assert seen_ids == ["ds__src__3"]
        assert out_recs2["ds__src__3"].is_stage_passed("C1")
        assert out_recs2["ds__src__2"].rejected_reason == "c1_parse_failed"


def test_qwen_single_event_infra_failure_summary_wording(caplog):
    """infra 失败汇总告警须对齐 raise 语义：stage 将中止，而非'进入后续阶段被跳过'。"""
    import logging

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [_record_with_audio(tmpdir, 0, {"original": "noise"})]
        write_manifest(iter(recs), manifest)

        proc = QwenSingleEventProcessor({})
        proc.process_records = lambda pending: [None] * len(pending)
        with caplog.at_level(logging.WARNING):
            proc.process(manifest, os.path.join(tmpdir, "c1"))

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "基础设施失败" in r.getMessage()
        ]
        assert warnings, "C1 结束时应输出基础设施失败汇总 warning"
        assert "中止" in warnings[0].getMessage()
        assert "将被跳过" not in warnings[0].getMessage()


def test_qwen_single_event_model_load_is_thread_safe():
    """并发调用 _load 时模型必须只加载一次（加载逻辑须持锁）。"""
    import sys
    import threading
    import time
    import types

    from noise_filter_pipeline.processors.qwen_single_event import QwenSingleEventModel

    load_count = [0]
    disable_count = [0]
    processor_kwargs = []

    class FakeLoadedModel:
        def __init__(self):
            self.speaker_map = {}

        def disable_talker(self):
            disable_count[0] += 1

    class FakeModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            load_count[0] += 1
            time.sleep(0.05)  # 放大竞态窗口
            return FakeLoadedModel()

    class FakeProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            processor_kwargs.append(kwargs)
            return object()

    fake_tf = types.SimpleNamespace(
        Qwen2_5OmniForConditionalGeneration=FakeModel,
        Qwen2_5OmniProcessor=FakeProcessor,
    )
    fake_utils = types.SimpleNamespace(process_mm_info=lambda *a, **k: None)
    saved = {k: sys.modules.get(k) for k in ("transformers", "qwen_omni_utils")}
    sys.modules["transformers"] = fake_tf
    sys.modules["qwen_omni_utils"] = fake_utils
    try:
        model = QwenSingleEventModel(model_path="/fake/qwen")
        threads = [threading.Thread(target=model._load) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert model._model is not None
        assert load_count[0] == 1
        assert disable_count[0] == 1
        assert processor_kwargs == [{"cache_dir": None, "use_fast": False}]
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_audiotag_classifier_dry_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [_record_with_audio(tmpdir, i, {"original": "noise"}) for i in range(3)]
        write_manifest(iter(recs), manifest)

        proc = AudioTagClassifierProcessor({
            "mode": "dry_run",
            "dry_run_labels": ["Speech", "Music"],
            "label_dict": _REAL_LABEL_DICT,
        })
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "c2"))
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 3
        assert all(r.is_stage_passed("C2") for r in out_recs)
        for rec in out_recs:
            assert set(rec.labels) == {"original", "audioset_top5"}
            assert len(rec.labels["audioset_top5"]) == 5
            assert rec.labels["audioset_top5"][0]["prob"] == 1.0
            assert set(rec.labels["audioset_top5"][0]) == {
                "index", "mid", "label", "prob",
            }
            assert rec.stages["C2"]["evidence_contract_version"] == 5
            assert rec.stages["C2"]["class_count"] == 527
            assert rec.stages["C2"]["score_contract"] == (
                "audioset_527_independent_sigmoid_v1"
            )


def test_audiotag_classifier_checkpoint_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [_record_with_audio(tmpdir, i, {"original": "noise"}) for i in range(4)]
        write_manifest(iter(recs), manifest)

        out_dir = os.path.join(tmpdir, "c2")
        proc = AudioTagClassifierProcessor({
            "mode": "dry_run",
            "dry_run_labels": ["Speech"],
            "label_dict": _REAL_LABEL_DICT,
        })
        out_manifest1 = proc.process(manifest, out_dir)
        out_recs1 = list(read_manifest(out_manifest1))
        assert len(out_recs1) == 4
        assert all(r.is_stage_passed("C2") for r in out_recs1)

        # 输出 manifest 仍在时重跑：checkpoint 命中，应恢复固定 top-5 事实。
        out_manifest2 = proc.process(manifest, out_dir)
        out_recs2 = list(read_manifest(out_manifest2))
        assert len(out_recs2) == 4
        assert all(r.is_stage_passed("C2") for r in out_recs2)
        assert all(len(r.labels["audioset_top5"]) == 5 for r in out_recs2)
        assert all("audioset_scores" not in r.labels for r in out_recs2)
        assert all("audioset_top10" not in r.labels for r in out_recs2)


def test_audiotag_classifier_checkpoint_resume_output_deleted():
    """上次输出被清理后重跑：checkpoint 命中但无输出可恢复，应回退为重新处理。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [_record_with_audio(tmpdir, i, {"original": "noise"}) for i in range(4)]
        write_manifest(iter(recs), manifest)

        out_dir = os.path.join(tmpdir, "c2")
        proc = AudioTagClassifierProcessor({
            "mode": "dry_run",
            "dry_run_labels": ["Speech"],
            "label_dict": _REAL_LABEL_DICT,
        })
        out_manifest1 = proc.process(manifest, out_dir)
        assert len(list(read_manifest(out_manifest1))) == 4

        # 模拟异常现场：上次 per-dataset 输出被清理（checkpoint 仍在但无输出可恢复）
        for f in os.listdir(out_dir):
            if f.startswith("c2_manifest_") and f.endswith(".jsonl"):
                os.remove(os.path.join(out_dir, f))
        out_manifest2 = proc.process(manifest, out_dir)
        out_recs2 = list(read_manifest(out_manifest2))
        assert len(out_recs2) == 4
        # 回退重新处理后记录仍应只带固定 C2 v5 top-5 事实。
        assert all(r.is_stage_passed("C2") for r in out_recs2)
        assert all(len(r.labels["audioset_top5"]) == 5 for r in out_recs2)
        assert all("audioset_scores" not in r.labels for r in out_recs2)
        assert all("audioset_top10" not in r.labels for r in out_recs2)


def test_audiotag_classifier_missing_model():
    with pytest.raises(FileNotFoundError):
        proc = AudioTagClassifierProcessor({
            "mode": "subprocess_daemon",
            "checkpoint": "/nonexistent/pretrained.pt",
            "label_dict": "/nonexistent/class_labels_indices.csv",
        })
        proc._validate_model_paths()


def test_audiotag_daemon_protocol_uses_fixed_v5_contract(tmp_path, monkeypatch):
    """daemon 不接收可变 top-k，infer 顶层必须携带有序 record_ids。"""
    from contextlib import nullcontext
    from noise_filter_pipeline.processors.audiotag_classifier import _AudioTagDaemonPool

    pool = _AudioTagDaemonPool(
        python_path="python",
        daemon_script_path="daemon.py",
        checkpoint="checkpoint.pt",
        label_dict=_REAL_LABEL_DICT,
        sample_rate=16000,
        batch_size=2,
        gpu_ids=[0],
        num_gpus=1,
        timeout=30,
        stage_name="C2",
    )
    cmd = pool._build_cmd(0)
    assert "--topk" not in cmd
    assert "--policy-path" not in cmd

    records = [
        _record_with_audio(str(tmp_path), index, {"original": "noise"})
        for index in range(2)
    ]
    sent = []

    def fake_send(_proc, payload):
        sent.append(payload)
        with open(payload["output_path"], "w", encoding="utf-8") as handle:
            json.dump([
                build_dry_run_result(
                    "Rain", _AUDIOTAG_METADATA, record_id=record.record_id
                )
                for record in records
            ], handle)

    monkeypatch.setattr(pool, "start", lambda _output_dir: None)
    monkeypatch.setattr(pool, "_pick", lambda: (0, object(), nullcontext()))
    monkeypatch.setattr(pool, "_send", fake_send)
    monkeypatch.setattr(pool, "_recv", lambda _proc: {"status": "ok"})

    output = pool.infer(records, str(tmp_path), batch_index=0)

    assert sent == [{
        "cmd": "infer",
        "input_path": str(tmp_path / "c2_input_gpu0_0000.json"),
        "output_path": str(tmp_path / "c2_output_gpu0_0000.json"),
        "record_ids": [record.record_id for record in records],
    }]
    assert [item["record_id"] for item in output] == [
        record.record_id for record in records
    ]


@pytest.mark.parametrize("contract_error", ["count", "id"])
def test_audiotag_output_contract_error_aborts_without_commit(contract_error):
    """输出数量或 ID 不守恒必须中止，当前 pending 不得部分提交。"""
    from noise_filter_pipeline.processors.audiotag_classifier import DaemonInferenceError

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [_record_with_audio(tmpdir, i, {"original": "noise"}) for i in range(3)]
        write_manifest(iter(recs), manifest)

        proc = AudioTagClassifierProcessor(
            {"mode": "dry_run", "label_dict": _REAL_LABEL_DICT}
        )
        outputs = proc._run_dry_run(recs)
        if contract_error == "count":
            outputs.pop()
        else:
            outputs[1]["record_id"] = "ds__src__wrong"
        proc._run_dry_run = lambda records: outputs
        output_dir = os.path.join(tmpdir, "c2")

        with pytest.raises(DaemonInferenceError) as raised:
            proc.process(manifest, output_dir)

        assert raised.value.error_kind == "internal"
        assert raised.value.error_code == "invalid_output_contract"
        assert raised.value.record_ids == [rec.record_id for rec in recs]
        assert not any(
            name.startswith("c2_manifest_") and name.endswith(".jsonl")
            for name in os.listdir(output_dir)
        )
        checkpoint = os.path.join(output_dir, "checkpoint.dry_run.json")
        assert not os.path.exists(checkpoint)


def test_audiotag_legacy_content_level_error_aborts_without_commit():
    """旧 ContentLevelError 不得再被解释为整批内容拒绝。"""
    from noise_filter_pipeline.core.exceptions import ContentLevelError

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [_record_with_audio(tmpdir, i, {"original": "noise"}) for i in range(3)]
        write_manifest(iter(recs), manifest)

        proc = AudioTagClassifierProcessor(
            {"mode": "dry_run", "label_dict": _REAL_LABEL_DICT}
        )
        proc._run_dry_run = lambda records: (_ for _ in ()).throw(
            ContentLevelError("AudioTag daemon (gpu=0) 推理失败: decode failed")
        )
        output_dir = os.path.join(tmpdir, "c2")

        with pytest.raises(ContentLevelError, match="decode failed"):
            proc.process(manifest, output_dir)

        assert not any(
            name.startswith("c2_manifest_") and name.endswith(".jsonl")
            for name in os.listdir(output_dir)
        )
        checkpoint = os.path.join(output_dir, "checkpoint.dry_run.json")
        assert not os.path.exists(checkpoint)


def test_stage_c_orchestrator_dry_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        label_dict, checkpoint, model_dir = _write_asset_fixtures(tmpdir)
        recs = [
            # Stage C seam 输入只携带上游原始标签；C2 top-5 必须由本轮生成。
            _record_with_audio(tmpdir, 0, {
                "original": "noise",
            }, {"B4": "passed"}),
            # 多标签记录（C1 已删除）直接进入 C1，不再被单标签过滤拦截
            _record_with_audio(tmpdir, 1, {
                "original": ["noise", "music"],
            }, {"B4": "passed"}),
            _record_with_audio(tmpdir, 2, {
                "original": "music",
            }, {"B4": "passed"}),
            # B4 未通过的记录应在 C1 被 skipped 透传，不再消耗 Qwen 算力
            _record_with_audio(tmpdir, 3, {
                "original": "noise",
            }, {"B4": "rejected"}),
        ]
        for record in recs:
            record.provenance = {}
        write_manifest(iter(recs), manifest)

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_c={
                "c1": {
                    "dry_run": True,
                    "dry_run_distribution": {"noise": 1.0, "babble": 0.0, "speech": 0.0},
                    "batch_size": 2,
                    "model_path": model_dir,
                },
                # C2 AudioTag 用 dry_run 产生稳定标签，避免测试环境加载模型
                "c2": {
                    "mode": "dry_run",
                    "dry_run_labels": ["Rain"],
                    "label_dict": label_dict,
                    "checkpoint": checkpoint,
                },
                "c3": {},
            },
        )
        orch = StageCOrchestrator(config)
        expected_fingerprints = orch._compute_artifact_fingerprints()
        assert not any(value.startswith("unset:") for value in expected_fingerprints.values())
        out_manifest = orch.run(
            input_manifest=manifest,
            output_dir=os.path.join(tmpdir, "out", "stage_c"),
            output_manifest=os.path.join(tmpdir, "out", "stage_c", "c_manifest.jsonl"),
        )

        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 4
        # C1 已删除：多标签记录直接进入 C1（dry-run 判 noise 通过）；
        # 仅 B4 rejected 记录在 C1 skipped 透传
        c1_skipped = [r for r in out_recs if r.stages.get("C1", {}).get("status") == "skipped"]
        assert len(c1_skipped) == 1
        # dry-run 仍执行完整 C3 truth table，便于审计；publishable=false 在发布链拦截。
        c3_evaluated = [
            r for r in out_recs
            if r.stages.get("C1", {}).get("status") == "passed"
        ]
        assert len(c3_evaluated) == 3
        for rec in c3_evaluated:
            assert rec.stages["C3"]["status"] == "passed"
            assert rec.stages["C3"]["admission_decision"] == "accept_noise"
            assert rec.stages["C3"]["publishable"] is False
            assert rec.stages["C3"]["execution_mode"] == {
                "c1": "dry_run",
                "c2": "dry_run",
            }
            assert rec.rejected_reason is None
        # B4 rejected 的记录从 C1 起即被 skipped，且不得通过 C3
        rejected_rec = [r for r in out_recs if r.stages.get("B4", {}).get("status") == "rejected"]
        assert len(rejected_rec) == 1
        assert rejected_rec[0].stages.get("C1", {}).get("status") == "skipped"
        assert not rejected_rec[0].is_stage_passed("C3")


def test_stage_c_dry_run_without_model_assets_is_not_publishable():
    """缺模型资产的 dry-run 可执行 C3，但始终标记不可发布。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        rec = _record_with_audio(
            tmpdir,
            0,
            {"original": "noise"},
            {"B4": "passed"},
        )
        rec.provenance = {}
        write_manifest(iter([rec]), manifest)

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_c={
                "c1": {
                    "dry_run": True,
                    "dry_run_distribution": {
                        "noise": 1.0,
                        "babble": 0.0,
                        "speech": 0.0,
                    },
                },
                "c2": {
                    "mode": "dry_run",
                    "dry_run_labels": ["Rain"],
                    "label_dict": os.path.join(
                        _repo_root(), "third_party_scripts", "icefall_audiotaq",
                        "data", "class_labels_indices.csv"),
                },
                "c3": {},
            },
        )
        orch = StageCOrchestrator(config)
        fingerprints = orch._compute_artifact_fingerprints()
        assert {
            value for value in fingerprints.values() if value.startswith("unset:")
        } == {
            "unset:audiotag_checkpoint_sha256",
            "unset:qwen_model_sha256",
        }

        output_dir = os.path.join(tmpdir, "out", "stage_c")
        out_manifest = orch.run(
            input_manifest=manifest,
            output_dir=output_dir,
            output_manifest=os.path.join(output_dir, "c_manifest.jsonl"),
        )

        out_rec = next(iter(read_manifest(out_manifest)))
        assert out_rec.is_stage_passed("C1")
        assert out_rec.is_stage_passed("C2")
        assert out_rec.stages["C3"]["status"] == "passed"
        assert out_rec.stages["C3"]["admission_decision"] == "accept_noise"
        assert out_rec.stages["C3"]["publishable"] is False
        assert out_rec.rejected_reason is None


# 固定 top-5 的低风险 C2 v5 事实。
C2_EVIDENCE_NONE = _audiotag_labels("Rain")


def _write_asset_fixtures(tmpdir):
    """创建 C1/C2 资产路径 fixture。

    orchestrator 启动时会对 policy/AudioTag label dict/checkpoint/Qwen
    模型目录做内容指纹注入；dry-run 测试不加载真实模型，只需路径下存在文件
    （内容任意），保证注入的指纹是合法 SHA-256 而非占位。label dict 直接使用
    repo 内 tracked 的真实 527 类 CSV，绑定 v5 top-5 标签身份。
    """
    label_dict = os.path.join(_repo_root(), "third_party_scripts", "icefall_audiotaq",
                              "data", "class_labels_indices.csv")
    checkpoint = os.path.join(tmpdir, "pretrained.pt")
    with open(checkpoint, "w", encoding="utf-8") as f:
        f.write("test checkpoint")
    model_dir = os.path.join(tmpdir, "qwen_model")
    os.makedirs(model_dir)
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
        f.write("{}")
    return label_dict, checkpoint, model_dir


def _c3_voice_policy_config():
    with open(PIPELINE_YAML, encoding="utf-8") as f:
        config = yaml.safe_load(f)["stage_c"]["c3"]
    config["execution_mode"] = {
        "c1": "production",
        "c2": "subprocess_daemon",
    }
    config["publishable"] = True
    return config


def _repo_root():
    return os.path.dirname(os.path.dirname(PIPELINE_ROOT))


def _policy_sha256(policy_path):
    """计算 policy 文件内容 SHA-256（相对路径相对仓库根解析）。"""
    from core.data.noise_admission import sha256_file

    if not os.path.isabs(policy_path):
        policy_path = os.path.join(_repo_root(), policy_path)
    return sha256_file(policy_path)


def _fake_artifact_fingerprints(policy_sha):
    """构造与当前无 calibration 契约一致的五项发布指纹。"""
    policy = load_admission_policy()
    return {
        "voice_policy_sha256": policy_sha,
        "audiotag_label_dict_sha256": policy.label_space_sha256,
        "audiotag_checkpoint_sha256": "3" * 64,
        "qwen_model_sha256": "4" * 64,
        "c1_prompt_sha256": "5" * 64,
    }


def _c1_identity():
    return {
        "qwen_model_sha256": "4" * 64,
        "c1_prompt_sha256": "5" * 64,
    }


def _c2_identity():
    policy = load_admission_policy()
    return {
        "audiotag_label_dict_sha256": policy.label_space_sha256,
        "audiotag_checkpoint_sha256": "3" * 64,
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "score_contract": SCORE_CONTRACT,
        "class_count": CLASSIFIER_LABEL_COUNT,
        "top_k": 5,
    }


def _c2_stage_extra():
    identity = _c2_identity()
    return {
        "evidence_contract_version": identity["evidence_contract_version"],
        "score_contract": identity["score_contract"],
        "class_count": identity["class_count"],
        "label_space_sha256": identity["audiotag_label_dict_sha256"],
        "audiotag_checkpoint_sha256": identity["audiotag_checkpoint_sha256"],
        "top_k": identity["top_k"],
    }


def _c3_test_config():
    """pipeline.yaml C3 段 + 合法 C1/C2/发布 identity。"""
    config = _c3_voice_policy_config()
    config["artifact_fingerprints"] = _fake_artifact_fingerprints(
        _policy_sha256(config["policy_path"])
    )
    config["c1_identity"] = _c1_identity()
    config["c2_identity"] = _c2_identity()
    config["execution_mode"] = {
        "c1": "production",
        "c2": "subprocess_daemon",
    }
    config["publishable"] = True
    return config


def _c2_labels(top1, *, scores=None):
    """按 label 从完整 527 内部排序生成固定 top-5。"""
    return _audiotag_labels(top1, scores)


def test_stage_c_rejects_custom_policy_path(tmp_path):
    """C3 policy 必须是 canonical policy 的同一 physical file。"""
    custom_policy = tmp_path / "copied_policy.json"
    with open(os.path.join(_repo_root(), POLICY_REL_PATH), "rb") as source:
        custom_policy.write_bytes(source.read())

    with pytest.raises(ValueError, match="canonical policy"):
        StageCOrchestrator(UnifiedPipelineConfig(stage_c={
            "c3": {"policy_path": str(custom_policy)},
        }))


def test_real_pipeline_yaml_references_single_voice_policy():
    with open(PIPELINE_YAML, encoding="utf-8") as f:
        stage_c = yaml.safe_load(f)["stage_c"]
    config = stage_c["c3"]

    assert config["policy_path"] == (
        "tools/noise_filter_pipeline/config/ontology/audiotag_voice_policy.json"
    )
    assert "policy_path" not in stage_c
    assert "calibration_path" not in stage_c
    for removed_key in (
        "calibration_path",
        "calibration_identity",
        "ontology_path",
        "reject_ontology_roots",
        "voice_prob_threshold",
        "mid_thresholds",
        "thresholds",
        "allow_voice_labels",
        "exact_reject_labels",
    ):
        assert removed_key not in config

    processor = NoiseAdmissionProcessor(dict(config))
    assert processor.policy.version == ADMISSION_POLICY_VERSION
    assert processor.policy_sha256 == _policy_sha256(config["policy_path"])
    assert len(processor.policy.mid_to_class) == CLASSIFIER_LABEL_COUNT
    assert set(processor.policy.mid_to_class.values()) == {"speech", "babble", "noise"}
    expected_classes = {
        "Babbling": "speech",
        "Chatter": "babble",
        "Hubbub, speech noise, speech babble": "babble",
        "Crowd": "babble",
        "Music": "noise",
        "Background music": "noise",
        "Inside, public space": "noise",
    }
    for label, expected in expected_classes.items():
        mid = _AUDIOTAG_METADATA.label_to_mid[label]
        assert processor.policy.class_of(mid) == expected


@pytest.mark.parametrize(
    ("c1", "c2", "execution_mode", "publishable"),
    [
        (
            {"dry_run": False},
            {"enable": True, "mode": "subprocess_daemon"},
            {"c1": "production", "c2": "subprocess_daemon"},
            True,
        ),
        (
            {"dry_run": True},
            {"enable": True, "mode": "subprocess_daemon"},
            {"c1": "dry_run", "c2": "subprocess_daemon"},
            False,
        ),
        (
            {"dry_run": False},
            {"enable": True, "mode": "dry_run"},
            {"c1": "production", "c2": "dry_run"},
            False,
        ),
        (
            {"dry_run": False},
            {"enable": False, "mode": "subprocess_daemon"},
            {"c1": "production", "c2": "disabled"},
            False,
        ),
    ],
)
def test_stage_c_execution_contract_is_derived_from_existing_modes(
    c1, c2, execution_mode, publishable
):
    assert build_stage_c_execution_contract(c1, c2) == {
        "execution_mode": execution_mode,
        "publishable": publishable,
    }


def test_stage_c_injects_shared_evidence_contract_version(monkeypatch, tmp_path):
    """C2/C3 版本注入必须跟随共享常量，避免复制固定版本字面量。"""
    import noise_filter_pipeline.stages.stage_c_single_event as stage_c_mod

    sentinel_version = EVIDENCE_CONTRACT_VERSION + 100
    monkeypatch.setattr(stage_c_mod, "EVIDENCE_CONTRACT_VERSION", sentinel_version)
    captured = {}

    def processor_factory(stage_name):
        class FakeProcessor:
            def __init__(self, config):
                self.config = config
                self.stage_name = stage_name

        return FakeProcessor

    monkeypatch.setattr(
        stage_c_mod, "QwenSingleEventProcessor", processor_factory("C1")
    )
    monkeypatch.setattr(
        stage_c_mod, "AudioTagClassifierProcessor", processor_factory("C2")
    )
    monkeypatch.setattr(
        stage_c_mod, "NoiseAdmissionProcessor", processor_factory("C3")
    )

    orchestrator = StageCOrchestrator(UnifiedPipelineConfig(stage_c={
        "c1": {"dry_run": True},
        "c2": {"mode": "dry_run"},
        "c3": {},
    }))
    monkeypatch.setattr(
        orchestrator,
        "_compute_artifact_fingerprints",
        lambda: {
            "voice_policy_sha256": "0" * 64,
            "audiotag_label_dict_sha256": "0" * 64,
            "audiotag_checkpoint_sha256": "0" * 64,
            "qwen_model_sha256": "0" * 64,
        },
    )

    def capture_substage(processor, input_manifest, **_kwargs):
        captured[processor.stage_name] = processor.config
        return input_manifest

    monkeypatch.setattr(orchestrator, "_run_substage", capture_substage)
    monkeypatch.setattr(stage_c_mod, "split_manifest_by_dataset", lambda **_kwargs: None)
    monkeypatch.setattr(
        stage_c_mod,
        "write_stage_stats",
        lambda **_kwargs: {"status": {"passed": 0}, "total_records": 0},
    )

    orchestrator.run(
        input_manifest="unused.jsonl",
        output_dir=str(tmp_path / "stage_c"),
        output_manifest=str(tmp_path / "stage_c" / "c_manifest.jsonl"),
    )

    assert captured["C2"]["evidence_contract_version"] == sentinel_version
    assert captured["C3"]["evidence_contract_version"] == sentinel_version


def test_c3_rejects_noncanonical_evidence_contract_version():
    """C3 仍拒绝任何非共享常量版本，不因编排器复用常量而放宽门禁。"""
    with pytest.raises(ValueError, match=str(EVIDENCE_CONTRACT_VERSION)):
        NoiseAdmissionProcessor({
            "evidence_contract_version": EVIDENCE_CONTRACT_VERSION + 1,
        })


@pytest.mark.parametrize(
    "legacy_config",
    [
        {"calibration_path": "legacy.json"},
        {"calibration_identity": {"sha256": "0" * 64}},
        {"voice_prob_threshold": 0.5},
        {"mid_thresholds": {}},
    ],
)
def test_c3_rejects_removed_calibration_and_threshold_config(legacy_config):
    """C3 对所有旧 calibration/threshold 配置 fail-closed。"""
    with pytest.raises(ValueError, match="calibration/threshold"):
        NoiseAdmissionProcessor(legacy_config)


def test_pipeline_config_rejects_legacy_stage_c_entries():
    """配置加载不得让旧顶层/C2/C3 契约潜伏到运行阶段。"""
    legacy_configs = (
        {"stage_c": {"calibration_path": "legacy.json"}},
        {"stage_c": {"c2": {"topk_keep_k": 10}}},
        {"stage_c": {"c2": {"output_label_key": "audioset_scores"}}},
        {"stage_c": {"c3": {"calibration_path": "legacy.json"}}},
    )
    for config in legacy_configs:
        with pytest.raises(ValueError, match="未知或已移除配置键"):
            UnifiedPipelineConfig.from_dict(config, active_stages=())


def _apply_c3(
    tmpdir,
    labels,
    c1_intelligibility=None,
    config=None,
    c2_extra=None,
    c1_status="passed",
    c2_status="passed",
):
    rec = _record_with_audio(tmpdir, 0, labels)
    if c1_intelligibility is not None:
        rec.mark_stage(
            "C1",
            c1_status,
            extra={"intelligibility": c1_intelligibility, **_c1_identity()},
        )
    if c2_status is not None:
        rec.mark_stage(
            "C2",
            c2_status,
            extra=_c2_stage_extra() if c2_extra is None else c2_extra,
        )
    return NoiseAdmissionProcessor(config or _c3_test_config())._process_record(rec)


@pytest.mark.parametrize(
    ("intelligibility", "scores", "decision", "reason"),
    [
        ("speech", {"Rain": 1.0}, "reject", "c1_speech"),
        ("noise", {"Rain": 1.0}, "accept_noise", "admitted_noise"),
        ("noise", {"Speech": 0.021, "Rain": 1.0}, "reject", "speech_mid_in_top5"),
        ("noise", {"Chatter": 1.0, "Rain": 0.8}, "accept_noise", "admitted_noise"),
        ("babble", {"Speech": 0.021, "Chatter": 1.0}, "reject", "speech_mid_in_top5"),
        ("babble", {"Chatter": 1.0}, "accept_babble", "admitted_babble"),
        ("babble", {"Rain": 1.0}, "accept_babble", "admitted_babble"),
    ],
)
def test_c3_frozen_truth_table(intelligibility, scores, decision, reason):
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = _apply_c3(
            tmpdir,
            _c2_labels(None, scores=scores),
            c1_intelligibility=intelligibility,
        )

    extra = rec.stages["C3"]
    assert extra["admission_decision"] == decision
    assert extra["admission_reason_codes"] == [reason]
    assert extra["reason"] == reason
    assert extra["status"] == (
        "passed" if decision in {"accept_noise", "accept_babble"} else "rejected"
    )
    assert "threshold_hits" not in extra
    assert "review_required" not in extra
    assert "content_kind" not in extra
    assert "quarantine" not in extra
    if decision == "reject":
        assert rec.rejected_reason == f"c3_{reason}"
        assert extra["balance_category"] is None
    else:
        assert rec.rejected_reason is None


def test_c3_has_no_probability_threshold_for_speech_veto():
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = _apply_c3(
            tmpdir,
            _c2_labels(None, scores={"Speech": 0.021, "Rain": 1.0}),
            c1_intelligibility="noise",
        )

    extra = rec.stages["C3"]
    assert extra["admission_decision"] == "reject"
    assert extra["reason"] == "speech_mid_in_top5"
    speech_hit = next(hit for hit in extra["top5_hits"] if hit["label"] == "Speech")
    assert speech_hit["prob"] == 0.021
    assert "threshold" not in speech_hit


@pytest.mark.parametrize(
    "label", ["Music", "Background music", "Inside, public space"]
)
def test_c3_hard_negative_labels_are_noise_categories(label):
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = _apply_c3(
            tmpdir,
            _c2_labels(None, scores={label: 0.8}),
            c1_intelligibility="noise",
        )

    extra = rec.stages["C3"]
    assert extra["admission_decision"] == "accept_noise"
    assert extra["balance_category"] == label
    assert extra["balance_category_mid"] == _AUDIOTAG_METADATA.label_to_mid[label]
    assert extra["balance_category_source"] == "noise_top5_hit"


def test_c3_category_selection_uses_highest_ranked_noise_mid():
    with tempfile.TemporaryDirectory() as tmpdir:
        highest = _apply_c3(
            tmpdir,
            _c2_labels(None, scores={"Rain": 0.7, "Music": 0.9}),
            c1_intelligibility="noise",
        )

    assert highest.stages["C3"]["balance_category"] == "Music"


def test_c3_babble_mid_is_non_decisive_for_decision_and_category():
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = _apply_c3(
            tmpdir,
            _c2_labels(None, scores={"Chatter": 0.99, "Rain": 0.9}),
            c1_intelligibility="babble",
        )

    extra = rec.stages["C3"]
    assert extra["admission_decision"] == "accept_babble"
    assert extra["balance_category"] == "Rain"
    assert extra["balance_category_source"] == "noise_top5_hit"
    chatter_hit = next(hit for hit in extra["top5_hits"] if hit["label"] == "Chatter")
    assert chatter_hit["kind"] == "babble"


def test_c3_c1_evidence_fail_closed():
    with tempfile.TemporaryDirectory() as tmpdir:
        missing = _apply_c3(tmpdir, _c2_labels("Rain"))
        assert missing.rejected_reason == "c3_c1_stage_missing"
        rejected = _apply_c3(
            tmpdir, _c2_labels("Rain"), c1_intelligibility="noise", c1_status="rejected"
        )
        assert rejected.rejected_reason == "c3_c1_not_passed"
        invalid = _apply_c3(tmpdir, _c2_labels("Rain"), c1_intelligibility="invalid")
        assert invalid.rejected_reason == "c3_c1_intelligibility_invalid"


def test_c3_c2_stage_status_fail_closed():
    """合法 labels 不能绕过 C2 stage 缺失或非 passed 状态。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        labels = _c2_labels("Rain")
        missing = _apply_c3(
            tmpdir, labels, c1_intelligibility="noise", c2_status=None
        )
        assert missing.rejected_reason == "c3_c2_identity_missing"
        for status in ("failed", "rejected"):
            not_passed = _apply_c3(
                tmpdir, labels, c1_intelligibility="noise", c2_status=status
            )
            assert not_passed.rejected_reason == "c3_c2_not_passed"

        speech = _apply_c3(
            tmpdir, labels, c1_intelligibility="speech", c2_status=None
        )
        assert speech.rejected_reason == "c3_c1_speech"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_top5",
        "short_top5",
        "bool_index",
        "nan_prob",
        "forged_mid",
        "bad_order",
        "extra_key",
    ],
)
def test_c3_malformed_top5_evidence_fail_closed(mutation):
    labels = _c2_labels("Rain")
    top5 = labels["audioset_top5"]
    if mutation == "missing_top5":
        labels.pop("audioset_top5")
    elif mutation == "short_top5":
        top5.pop()
    elif mutation == "bool_index":
        top5[0]["index"] = True
    elif mutation == "nan_prob":
        top5[0]["prob"] = float("nan")
    elif mutation == "forged_mid":
        top5[0]["mid"] = top5[1]["mid"]
    elif mutation == "bad_order":
        top5[0], top5[1] = top5[1], top5[0]
    else:
        top5[0]["threshold"] = 0.5

    with tempfile.TemporaryDirectory() as tmpdir:
        rec = _apply_c3(tmpdir, labels, c1_intelligibility="noise")
    assert rec.rejected_reason == "c3_c2_fact_invalid"


def test_c3_rejects_c2_identity_mismatch_before_parsing_scores():
    bad_identity = _c2_stage_extra()
    bad_identity["evidence_contract_version"] = EVIDENCE_CONTRACT_VERSION - 1
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = _apply_c3(
            tmpdir,
            _c2_labels("Rain"),
            c1_intelligibility="noise",
            c2_extra=bad_identity,
        )
    assert rec.rejected_reason == "c3_c2_identity_mismatch"


def test_c3_fingerprints_missing_or_mismatch_fail_closed():
    """artifact_fingerprints 缺失/不全/与 C3 实际 policy hash 不一致 → 拒绝。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        labels = _c2_labels("Rain")

        # config 无注入 + 记录无 C2 stage extra → 拒绝
        rec = _apply_c3(
            tmpdir, labels, c1_intelligibility="noise", config=_c3_voice_policy_config()
        )
        assert rec.rejected_reason == "c3_artifact_fingerprints_missing"

        # 五项发布指纹缺一个 key → 拒绝
        partial = _c3_test_config()
        partial["artifact_fingerprints"].pop("qwen_model_sha256")
        rec = _apply_c3(tmpdir, labels, c1_intelligibility="noise", config=partial)
        assert rec.rejected_reason == "c3_artifact_fingerprints_invalid"

        # voice_policy_sha256 与 C3 实际 policy 不一致 → 拒绝
        mismatched = _c3_test_config()
        mismatched["artifact_fingerprints"]["voice_policy_sha256"] = "f" * 64
        rec = _apply_c3(tmpdir, labels, c1_intelligibility="noise", config=mismatched)
        assert rec.rejected_reason == "c3_policy_hash_mismatch"

        # C2 stage 只携带来源/事实 identity，不携带业务 policy 指纹。
        assert set(_c2_stage_extra()) == {
            "evidence_contract_version",
            "score_contract",
            "class_count",
            "label_space_sha256",
            "audiotag_checkpoint_sha256",
            "top_k",
        }


def test_validate_admission_contract_binds_core_c3_fingerprints():
    """共享复验绑定当前 policy 与 label-space identity。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = _apply_c3(tmpdir, _c2_labels("Rain"), c1_intelligibility="noise")
        policy = load_admission_policy()
        c1 = rec.stages["C1"]
        labels = rec.labels
        valid = rec.stages["C3"]

        validate_admission_contract(
            c1_intelligibility=c1["intelligibility"],
            c2_labels=labels,
            c2_stage=rec.stages["C2"],
            admission_stage=valid,
            policy=policy,
            require_accepted=True,
            location="test C3",
        )

        cases = []
        forged_policy = dict(valid)
        forged_policy["artifact_fingerprints"] = {
            **valid["artifact_fingerprints"],
            "voice_policy_sha256": "f" * 64,
        }
        cases.append((forged_policy, "current policy.sha256"))
        forged_labels = dict(valid)
        forged_labels["artifact_fingerprints"] = {
            **valid["artifact_fingerprints"],
            "audiotag_label_dict_sha256": "e" * 64,
        }
        cases.append((forged_labels, "policy.label_space_sha256"))

        for stage, message in cases:
            with pytest.raises(ValueError, match=message):
                validate_admission_contract(
                    c1_intelligibility=c1["intelligibility"],
                    c2_labels=labels,
                    c2_stage=rec.stages["C2"],
                    admission_stage=stage,
                    policy=policy,
                    require_accepted=True,
                    location="test C3",
                )


def test_c3_top10_only_manifest_fail_closed():
    """旧 top-10 不能替代固定 top-5，必须 fail-closed。"""
    legacy_top10 = _c2_labels("Rain")["audioset_top5"] * 2
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = _apply_c3(
            tmpdir,
            {"audioset_top10": legacy_top10},
            c1_intelligibility="noise",
        )
    assert rec.rejected_reason == "c3_c2_fact_invalid"


@pytest.mark.parametrize(
    ("legacy_key", "legacy_value"),
    [
        ("topk", [{"label": "Speech", "prob": 1.0}]),
        ("audioset", "Speech"),
        ("audioset_mid", "/m/09x0r"),
    ],
)
def test_c3_rejects_legacy_evidence_fields_alongside_top5(
    legacy_key, legacy_value
):
    """旧 C2 字段即使与固定 top-5 共存也必须 fail-closed。"""
    labels = _c2_labels(None, scores={"Rain": 0.9, "Chatter": 0.8})
    labels[legacy_key] = legacy_value
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = _apply_c3(tmpdir, labels, c1_intelligibility="noise")
    assert rec.rejected_reason == "c3_c2_fact_invalid"


def test_c3_rejects_unknown_legacy_label_fields_without_substring_fallback():
    """v5 只接受固定 C2 schema；未知 leaf 字段不得参与或绕过裁决。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = _apply_c3(tmpdir, {
            **_c2_labels("Rain"),
            "leaf": "Male speech, man speaking",
            "original": "studio speech artifact",
        }, c1_intelligibility="noise")
        assert rec.rejected_reason == "c3_c2_fact_invalid"


@pytest.mark.parametrize(
    ("execution_mode", "publishable"),
    [
        ({"c1": "dry_run", "c2": "subprocess_daemon"}, False),
        ({"c1": "production", "c2": "dry_run"}, False),
        ({"c1": "production", "c2": "disabled"}, False),
    ],
)
def test_c3_non_publishable_execution_modes_remain_auditable(execution_mode, publishable):
    """dry-run/disabled 身份仍重算 truth table，但发布标记必须为 false。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _c3_test_config()
        config["execution_mode"] = execution_mode
        config["publishable"] = publishable
        rec = _record_with_audio(tmpdir, 0, _c2_labels("Rain"))
        rec.stages["C1"] = {
            "status": "passed",
            "intelligibility": "noise",
            **_c1_identity(),
        }
        rec.stages["C2"] = {"status": "passed", **_c2_stage_extra()}
        rec.provenance["stage_c_execution"] = dict(execution_mode)

        processed = NoiseAdmissionProcessor(config)._process_record(rec)

        assert processed.is_stage_passed("C3")
        assert processed.stages["C3"]["admission_decision"] == "accept_noise"
        assert processed.stages["C3"]["publishable"] is False
        assert processed.stages["C3"]["execution_mode"] == execution_mode


def test_c3_rechecks_legacy_passed_record_against_execution_contract():
    """旧 C3 passed 不能触发通用 should_skip 绕过新的执行模式门禁。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        execution_mode = {"c1": "dry_run", "c2": "subprocess_daemon"}
        rec = _record_with_audio(tmpdir, 0, _c2_labels("Rain"))
        rec.stages["C1"] = {
            "status": "passed",
            "intelligibility": "noise",
            **_c1_identity(),
        }
        rec.stages["C2"] = {"status": "passed", **_c2_stage_extra()}
        rec.stages["C3"] = {"status": "passed", "admission_allowed": True}
        rec.provenance["stage_c_execution"] = dict(execution_mode)
        write_manifest(iter([rec]), manifest)
        config = _c3_test_config()
        config["execution_mode"] = execution_mode
        config["publishable"] = False

        output = NoiseAdmissionProcessor(config).process(
            manifest, os.path.join(tmpdir, "c3")
        )
        processed = next(iter(read_manifest(output)))

        assert processed.stages["C3"]["status"] == "passed"
        assert processed.stages["C3"]["admission_decision"] == "accept_noise"
        assert processed.stages["C3"]["publishable"] is False


def test_c3_publishable_fixture_identity_passes():
    """合法 C2/C3 identity 与五项资产指纹可获得发布资格。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _c3_test_config()
        rec = _record_with_audio(
            tmpdir, 0, _c2_labels("Rain"), {"C1": "passed", "C2": "passed"}
        )
        rec.stages["C1"]["intelligibility"] = "noise"
        rec.stages["C2"].update(_c2_stage_extra())

        processed = NoiseAdmissionProcessor(config)._process_record(rec)

        assert processed.is_stage_passed("C3")
        assert processed.stages["C3"]["artifact_fingerprints"] == config[
            "artifact_fingerprints"
        ]
        assert processed.stages["C3"]["publishable"] is True


def test_c3_passed_extra_contract():
    with tempfile.TemporaryDirectory() as tmpdir:
        fingerprints = _fake_artifact_fingerprints(_policy_sha256(POLICY_REL_PATH))
        noise = _apply_c3(tmpdir, _c2_labels("Rain"), c1_intelligibility="noise")
        extra = noise.stages["C3"]
        rain_mid = _AUDIOTAG_METADATA.label_to_mid["Rain"]
        assert extra["status"] == "passed"
        assert extra["admission_decision"] == "accept_noise"
        assert extra["admission_reason_codes"] == ["admitted_noise"]
        assert extra["reason"] == "admitted_noise"
        assert extra["admission_allowed"] is True
        assert extra["balance_category"] == "Rain"
        assert extra["balance_category_mid"] == rain_mid
        assert extra["balance_category_source"] == "noise_top5_hit"
        assert len(extra["top5_hits"]) == 5
        assert extra["top5_hits"][0] == {
            "index": _AUDIOTAG_METADATA.index_of(rain_mid),
            "mid": rain_mid,
            "label": "Rain",
            "kind": "noise",
            "prob": 1.0,
        }
        assert all(
            set(hit) == {"index", "mid", "label", "kind", "prob"}
            for hit in extra["top5_hits"]
        )
        assert extra["admission_policy_version"] == ADMISSION_POLICY_VERSION
        assert extra["admission_policy_sha256"] == _policy_sha256(POLICY_REL_PATH)
        assert extra["evidence_contract_version"] == EVIDENCE_CONTRACT_VERSION
        assert extra["score_contract"] == SCORE_CONTRACT
        assert extra["label_space_sha256"] == load_admission_policy().label_space_sha256
        assert extra["artifact_fingerprints"] == fingerprints
        assert extra["execution_mode"] == {
            "c1": "production",
            "c2": "subprocess_daemon",
        }
        assert extra["publishable"] is True
        for removed_key in (
            "review_required",
            "threshold_hits",
            "admission_calibration_version",
            "admission_calibration_sha256",
            "content_kind",
            "quarantine",
        ):
            assert removed_key not in extra

        babble = _apply_c3(
            tmpdir,
            _c2_labels(None, scores={"Chatter": 0.9}),
            c1_intelligibility="babble",
        )
        extra = babble.stages["C3"]
        assert extra["admission_decision"] == "accept_babble"
        assert extra["balance_category"] == "Rain"
        assert extra["balance_category_mid"] == rain_mid
        assert extra["balance_category_source"] == "noise_top5_hit"


def test_noise_label_filter_process_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [
            _record_with_audio(
                tmpdir, 0, _c2_labels("Rain"), {"C1": "passed", "C2": "passed"}
            ),
            _record_with_audio(
                tmpdir, 1, _c2_labels("Speech"), {"C1": "passed", "C2": "passed"}
            ),
            _record_with_audio(
                tmpdir,
                2,
                _c2_labels("Rain", scores={"Singing": 0.8}),
                {"C1": "passed", "C2": "passed"},
            ),
            _record_with_audio(
                tmpdir,
                3,
                _c2_labels(None, scores={"Chatter": 0.9}),
                {"C1": "passed", "C2": "passed"},
            ),
        ]
        for rec, kind in zip(recs, ("noise", "noise", "noise", "babble")):
            rec.stages["C1"]["intelligibility"] = kind
            rec.stages["C2"].update(_c2_stage_extra())
        write_manifest(iter(recs), manifest)

        out_manifest = NoiseAdmissionProcessor(_c3_test_config()).process(
            manifest, os.path.join(tmpdir, "c3")
        )
        out_recs = list(read_manifest(out_manifest))
        assert out_recs[0].is_stage_passed("C3")
        assert out_recs[1].rejected_reason == "c3_speech_mid_in_top5"
        assert out_recs[2].rejected_reason == "c3_speech_mid_in_top5"
        assert out_recs[3].is_stage_passed("C3")
        assert out_recs[3].stages["C3"]["admission_decision"] == "accept_babble"
        assert out_recs[3].stages["C3"]["balance_category"] == "Rain"


def test_noise_label_filter_robust_to_malformed_records():
    """旧 topk 与固定 top-5 共存、或缺 C2 fact 时均 fail-closed。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [
            _record_with_audio(
                tmpdir,
                0,
                {**_c2_labels("Wind"), "topk": [{"label": "Speech", "prob": "high"}]},
                {"C1": "passed", "C2": "passed"},
            ),
            _record_with_audio(
                tmpdir,
                1,
                {
                    **_c2_labels("Wind"),
                    "topk": ["not-a-dict", 42, {"label": "Wind", "prob": 0.9}],
                },
                {"C1": "passed", "C2": "passed"},
            ),
            _record_with_audio(
                tmpdir, 2, {"audioset": "Rain"}, {"C1": "passed", "C2": "passed"}
            ),
            _record_with_audio(tmpdir, 3, {**_c2_labels("Rain")}),
        ]
        for rec in recs[:3]:
            rec.stages["C1"]["intelligibility"] = "noise"
            rec.stages["C2"].update(_c2_stage_extra())
        recs[2].labels = None
        recs[3].stages = None
        write_manifest(iter(recs), manifest)

        proc = NoiseAdmissionProcessor(_c3_test_config())
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "c3"))
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 4
        assert out_recs[0].rejected_reason == "c3_c2_fact_invalid"
        assert out_recs[1].rejected_reason == "c3_c2_fact_invalid"
        assert out_recs[2].rejected_reason == "c3_c2_fact_invalid"
        assert out_recs[3].rejected_reason == "c3_c1_stage_missing"


def test_noise_label_filter_disabled_fail_closed():
    """enable: false 时 C3 无法形成最终证据，必须明确拒绝。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [
            _record_with_audio(tmpdir, 0, {"audioset": "Speech"}),
            _record_with_audio(tmpdir, 1, {"audioset": "Noise"}),
        ]
        write_manifest(iter(recs), manifest)

        proc = NoiseAdmissionProcessor({"enable": False})
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "c3"))
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 2
        for rec in out_recs:
            assert rec.stages["C3"]["status"] == "rejected"
            assert rec.stages["C3"]["enabled"] is False
            assert rec.stages["C3"]["reason"] == "c3_disabled"
            assert rec.stages["C3"]["publishable"] is False
            assert rec.rejected_reason == "c3_disabled"


def test_c1_records_bind_qwen_model_and_prompt_identity():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        rec = _record_with_audio(tmpdir, 0, {"original": "noise"})
        write_manifest(iter([rec]), manifest)
        prompt = "Return only Noise."
        identity = {
            "qwen_model_sha256": "a" * 64,
            "c1_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
        proc = QwenSingleEventProcessor({
            "dry_run": True,
            "prompt": prompt,
            "artifact_fingerprints": identity,
            "dry_run_distribution": {"noise": 1.0},
        })

        output = proc.process(manifest, os.path.join(tmpdir, "c1"))
        processed = next(iter(read_manifest(output)))

        assert {
            key: processed.stages["C1"][key] for key in identity
        } == identity


def test_c3_rejects_exact_c1_and_c2_source_identity_mismatch():
    with tempfile.TemporaryDirectory() as tmpdir:
        c1_bad = _apply_c3(
            tmpdir, _c2_labels("Rain"), c1_intelligibility="noise"
        )
        c1_bad.stages.pop("C3")
        c1_bad.stages["C1"]["c1_prompt_sha256"] = "f" * 64
        c1_result = NoiseAdmissionProcessor(_c3_test_config())._process_record(c1_bad)
        assert c1_result.rejected_reason == "c3_c1_identity_mismatch"

        bad_c2 = _c2_stage_extra()
        bad_c2["audiotag_checkpoint_sha256"] = "e" * 64
        c2_result = _apply_c3(
            tmpdir,
            _c2_labels("Rain"),
            c1_intelligibility="noise",
            c2_extra=bad_c2,
        )
        assert c2_result.rejected_reason == "c3_c2_identity_mismatch"
