"""
Tests for Stage B curation (lightweight: VAD/classification disabled).
"""
import json
import os
import tempfile

import numpy as np
import soundfile as sf
import pytest

from noise_filter_pipeline.core.pipeline_config import UnifiedPipelineConfig
from noise_filter_pipeline.core.manifest_io import write_manifest, read_manifest, AudioRecord
from noise_filter_pipeline.stages.stage_b_curation import StageBOrchestrator

pytestmark = pytest.mark.pipeline


def _make_wav(path, duration=2.0, sr=16000, scale=0.05):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wav = np.random.randn(int(duration * sr)) * scale
    sf.write(path, wav, sr)


def _make_record(wav_path, duration, sr=16000):
    rec = AudioRecord(
        record_id="ds__src__source",
        dataset_id="ds",
        source_id="src",
        chunk_id="source",
        audio={"path": wav_path, "sr": sr, "duration": duration, "channels": 1},
        labels={"original": "noise"},
    )
    rec.mark_stage("A", "passed")
    return rec


def test_stage_b_adaptive_chunking():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "src.wav")
        _make_wav(wav_path, duration=12.0)

        a_manifest = os.path.join(tmpdir, "a_manifest.jsonl")
        write_manifest(iter([_make_record(wav_path, 12.0)]), a_manifest)

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_b={
                "b1": {"mode": "adaptive", "chunk_duration": 5.0, "min_chunk_duration": 2.0},
                "b2": {"enable": False},
                "b3": {"enable": False},
                "b5": {"enable": False},
                "b4": {"keep_noise": True, "keep_suspected": True, "keep_rejected": True},
                "num_workers": 1,
            },
        )
        orch = StageBOrchestrator(config)
        b_manifest = orch.run(
            input_manifest=a_manifest,
            output_dir=os.path.join(tmpdir, "out", "stage_b"),
            output_manifest=os.path.join(tmpdir, "out", "stage_b", "b_manifest.jsonl"),
        )

        b_recs = list(read_manifest(b_manifest))
        assert len(b_recs) >= 2
        for r in b_recs:
            assert r.is_stage_passed("B4")
            assert r.audio["duration"] >= 2.0


def test_stage_b_hive_chunking():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "src.wav")
        _make_wav(wav_path, duration=25.0)

        a_manifest = os.path.join(tmpdir, "a_manifest.jsonl")
        write_manifest(iter([_make_record(wav_path, 25.0)]), a_manifest)

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_b={
                "b1": {
                    "mode": "hive",
                    "hive_window_sec": 10.0,
                    "hive_step_sec": 5.0,
                },
                "b2": {"enable": False},
                "b3": {"enable": False},
                "b5": {"enable": False},
                "b4": {"keep_noise": True},
                "num_workers": 1,
            },
        )
        orch = StageBOrchestrator(config)
        b_manifest = orch.run(
            input_manifest=a_manifest,
            output_dir=os.path.join(tmpdir, "out", "stage_b"),
            output_manifest=os.path.join(tmpdir, "out", "stage_b", "b_manifest.jsonl"),
        )

        b_recs = list(read_manifest(b_manifest))
        # 25s -> 4 个 B1 hive chunks
        assert len(b_recs) >= 4
        for r in b_recs:
            assert r.is_stage_passed("B4")


def test_stage_b_legacy_hive_chunking_mapping():
    """验证旧配置 enable_hive_chunking + hive_chunking 能正确映射到 B1 hive 模式。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "src.wav")
        _make_wav(wav_path, duration=25.0)

        a_manifest = os.path.join(tmpdir, "a_manifest.jsonl")
        write_manifest(iter([_make_record(wav_path, 25.0)]), a_manifest)

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_b={
                "enable_hive_chunking": True,
                "hive_chunking": {"window_sec": 10.0, "step_sec": 5.0},
                "chunk_duration": 5.0,
                "min_chunk_duration": 2.0,
                "enable_vad": False,
                "enable_classification": False,
                "enable_silence_filter": False,
                "keep_noise": True,
                "num_workers": 1,
            },
        )
        orch = StageBOrchestrator(config)
        b_manifest = orch.run(
            input_manifest=a_manifest,
            output_dir=os.path.join(tmpdir, "out", "stage_b"),
            output_manifest=os.path.join(tmpdir, "out", "stage_b", "b_manifest.jsonl"),
        )

        b_recs = list(read_manifest(b_manifest))
        assert len(b_recs) >= 4
        for r in b_recs:
            assert r.is_stage_passed("B4")


def test_stage_b_unified_pipeline_lightweight():
    """验证 unified 单文件流水线与分阶段链路结果等价。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "src.wav")
        _make_wav(wav_path, duration=12.0)

        a_manifest = os.path.join(tmpdir, "a_manifest.jsonl")
        write_manifest(iter([_make_record(wav_path, 12.0)]), a_manifest)

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_b={
                "unified": True,
                "b1": {"mode": "adaptive", "chunk_duration": 5.0, "min_chunk_duration": 2.0},
                "b2": {"enable": True, "max_silence_ratio": 0.95},
                "b3": {"enable": False},
                "b5": {"enable": False},
                "b4": {"keep_noise": True, "keep_suspected": True, "keep_rejected": True},
                "num_workers": 1,
            },
        )
        orch = StageBOrchestrator(config)
        b_manifest = orch.run(
            input_manifest=a_manifest,
            output_dir=os.path.join(tmpdir, "out", "stage_b"),
            output_manifest=os.path.join(tmpdir, "out", "stage_b", "b_manifest.jsonl"),
        )

        b_recs = list(read_manifest(b_manifest))
        assert len(b_recs) >= 2
        for r in b_recs:
            assert r.is_stage_passed("B4")
            assert "B1" in r.stages
            assert "B2" in r.stages
            assert "B4" in r.stages
            # unified 模式下只保留最终音频
            assert os.path.isfile(r.audio["path"])


def test_stage_b_unified_output_shard_is_forwarded_by_orchestrator():
    """顶层 output_shard 必须传入 unified processor，避免海量音频平铺。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "src.wav")
        _make_wav(wav_path, duration=3.0)
        input_dir = os.path.join(tmpdir, "stage_a")
        os.makedirs(input_dir)
        write_manifest(iter([_make_record(wav_path, 3.0)]), os.path.join(input_dir, "a_manifest_ds.jsonl"))

        orch = StageBOrchestrator(UnifiedPipelineConfig(stage_b={
            "unified": True,
            "b1": {"mode": "none"},
            "b2": {"enable": False},
            "b3": {"enable": False},
            "b4": {"keep_noise": True},
            "output_shard": True,
            "num_workers": 1,
        }))
        output_dir = os.path.join(tmpdir, "stage_b")
        orch.run(input_dir, output_dir, os.path.join(output_dir, "b_manifest.jsonl"))

        out_rec = next(read_manifest(output_dir))
        audio_root = os.path.join(output_dir, "audio")
        assert len(os.path.dirname(os.path.relpath(out_rec.audio["path"], audio_root))) == 2


def test_stage_b_prunes_stale_output_for_empty_input_dataset():
    """本轮空上游数据集不得让旧 B 输出继续流入 C/D。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "stage_a")
        output_dir = os.path.join(tmpdir, "stage_b")
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        open(os.path.join(input_dir, "a_manifest_ds1.jsonl"), "w", encoding="utf-8").close()
        open(os.path.join(output_dir, "b_manifest_ds1.jsonl"), "w", encoding="utf-8").write("stale\n")
        open(os.path.join(output_dir, "b_manifest_ds2.jsonl"), "w", encoding="utf-8").write("keep\n")

        orch = StageBOrchestrator(UnifiedPipelineConfig(stage_b={
            "unified": True,
            "b1": {"mode": "none"},
            "b2": {"enable": False},
            "b3": {"enable": False},
            "b4": {"keep_noise": True},
            "num_workers": 1,
        }))
        orch.run(input_dir, output_dir, os.path.join(output_dir, "b_manifest.jsonl"))

        assert not os.path.exists(os.path.join(output_dir, "b_manifest_ds1.jsonl"))
        assert os.path.isfile(os.path.join(output_dir, "b_manifest_ds2.jsonl"))


def test_stage_b_unified_keeps_b2_rejection_for_audit():
    """B2 拒绝的 chunk 必须以 B2 原因写入 manifest，且不得误标 B1。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "silent.wav")
        sf.write(wav_path, np.full(3 * 16000, 0.001, dtype=np.float32), 16000)
        a_manifest = os.path.join(tmpdir, "a_manifest.jsonl")
        write_manifest(iter([_make_record(wav_path, 3.0)]), a_manifest)
        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_b={
                "unified": True,
                "b1": {"mode": "none"},
                "b2": {"enable": True},
                "b3": {"enable": False},
                "b4": {"keep_noise": True},
                "num_workers": 1,
            },
        )
        b_manifest = StageBOrchestrator(config).run(
            a_manifest, os.path.join(tmpdir, "out", "stage_b"),
            os.path.join(tmpdir, "out", "stage_b", "b_manifest.jsonl"),
        )
        recs = list(read_manifest(b_manifest))
        assert len(recs) == 1
        assert recs[0].is_stage_passed("B1")
        assert recs[0].stages["B2"]["status"] == "rejected"
        assert recs[0].rejected_reason.startswith("b2_")
        assert "B3" not in recs[0].stages
        assert "B4" not in recs[0].stages


def test_stage_b_unified_rejected_source_preserves_original():
    """B1 拒绝的源记录不得进入 B3/B4，且原始数据集文件必须保持完好。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1s 音频在 adaptive + min_chunk_duration=2.0 下会被 B1 以 no_chunks 拒绝
        wav_path = os.path.join(tmpdir, "src.wav")
        _make_wav(wav_path, duration=1.0)

        a_manifest = os.path.join(tmpdir, "a_manifest.jsonl")
        write_manifest(iter([_make_record(wav_path, 1.0)]), a_manifest)

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_b={
                "unified": True,
                "b1": {"mode": "adaptive", "chunk_duration": 10.0, "min_chunk_duration": 2.0},
                "b2": {"enable": True},
                "b3": {"enable": False},
                "b4": {"keep_noise": True},
                "num_workers": 1,
            },
        )
        orch = StageBOrchestrator(config)
        b_manifest = orch.run(
            input_manifest=a_manifest,
            output_dir=os.path.join(tmpdir, "out", "stage_b"),
            output_manifest=os.path.join(tmpdir, "out", "stage_b", "b_manifest.jsonl"),
        )

        b_recs = list(read_manifest(b_manifest))
        assert len(b_recs) == 1
        rec = b_recs[0]
        assert rec.stages.get("B1", {}).get("status") == "rejected"
        # 被拒绝的源记录不得被 B4 标记为通过
        assert not rec.is_stage_passed("B4")
        # 原始数据集文件必须完好，且记录仍指向原路径
        assert os.path.isfile(wav_path)
        assert rec.audio["path"] == wav_path


def test_stage_b_staged_b2_rejection_not_resurrected():
    """staged（unified:false）链路：B2 拒绝的记录不得被 B3/B4 复活。

    回归测试：旧 staged 链路顺序调用 B2/B3/B4 且无 prev_stage 门控，
    B2 rejected 的静音 chunk（音频文件仍在盘上）会被后续阶段放行，
    最终在 Stage C 复活。修复后 B3/B4 必须标 skipped 透传。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "silent.wav")
        sf.write(wav_path, np.full(3 * 16000, 0.001, dtype=np.float32), 16000)

        a_manifest = os.path.join(tmpdir, "a_manifest.jsonl")
        write_manifest(iter([_make_record(wav_path, 3.0)]), a_manifest)

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_b={
                "unified": False,
                "b1": {"mode": "none"},
                "b2": {"enable": True},
                "b3": {"enable": False},
                "b4": {"keep_noise": True, "keep_suspected": True, "keep_rejected": True},
                "num_workers": 1,
            },
        )
        orch = StageBOrchestrator(config)
        b_manifest = orch.run(
            input_manifest=a_manifest,
            output_dir=os.path.join(tmpdir, "out", "stage_b"),
            output_manifest=os.path.join(tmpdir, "out", "stage_b", "b_manifest.jsonl"),
        )

        recs = list(read_manifest(b_manifest))
        assert len(recs) == 1
        rec = recs[0]
        assert rec.stages["B2"]["status"] == "rejected"
        assert rec.stages.get("B3", {}).get("status") == "skipped"
        assert rec.stages.get("B4", {}).get("status") == "skipped"
        assert not rec.is_stage_passed("B4")


def test_stage_b_staged_b1_no_chunks_downstream_skipped():
    """staged 链路：B1 拒绝（no_chunks）的源记录，B2/B3/B4 必须全部 skipped。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1.5s 短源在 adaptive + min_chunk_duration=2.0 下被 B1 以 no_chunks 拒绝
        wav_path = os.path.join(tmpdir, "short.wav")
        _make_wav(wav_path, duration=1.5)

        a_manifest = os.path.join(tmpdir, "a_manifest.jsonl")
        write_manifest(iter([_make_record(wav_path, 1.5)]), a_manifest)

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_b={
                "unified": False,
                "b1": {"mode": "adaptive", "chunk_duration": 10.0, "min_chunk_duration": 2.0},
                "b2": {"enable": True},
                "b3": {"enable": False},
                "b4": {"keep_noise": True, "keep_suspected": True, "keep_rejected": True},
                "num_workers": 1,
            },
        )
        orch = StageBOrchestrator(config)
        b_manifest = orch.run(
            input_manifest=a_manifest,
            output_dir=os.path.join(tmpdir, "out", "stage_b"),
            output_manifest=os.path.join(tmpdir, "out", "stage_b", "b_manifest.jsonl"),
        )

        recs = list(read_manifest(b_manifest))
        assert len(recs) == 1
        rec = recs[0]
        assert rec.stages["B1"]["status"] == "rejected"
        assert rec.stages["B1"].get("reason") == "no_chunks"
        assert rec.stages.get("B2", {}).get("status") == "skipped"
        assert rec.stages.get("B3", {}).get("status") == "skipped"
        assert rec.stages.get("B4", {}).get("status") == "skipped"
        assert not rec.is_stage_passed("B4")


def test_stage_b_full_pipeline_lightweight():
    """lightweight 全链路：仅启用 B1、B2、B4，关闭 VAD/分类。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "src.wav")
        _make_wav(wav_path, duration=12.0)

        a_manifest = os.path.join(tmpdir, "a_manifest.jsonl")
        write_manifest(iter([_make_record(wav_path, 12.0)]), a_manifest)

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_b={
                "b1": {"mode": "adaptive", "chunk_duration": 5.0, "min_chunk_duration": 2.0},
                "b2": {"enable": True, "max_silence_ratio": 0.95},
                "b3": {"enable": False},
                "b5": {"enable": False},
                "b4": {"keep_noise": True, "keep_suspected": True, "keep_rejected": True},
                "num_workers": 1,
            },
        )
        orch = StageBOrchestrator(config)
        b_manifest = orch.run(
            input_manifest=a_manifest,
            output_dir=os.path.join(tmpdir, "out", "stage_b"),
            output_manifest=os.path.join(tmpdir, "out", "stage_b", "b_manifest.jsonl"),
        )

        b_recs = list(read_manifest(b_manifest))
        assert len(b_recs) >= 2
        for r in b_recs:
            assert r.is_stage_passed("B4")
            assert "B1" in r.stages
            assert "B2" in r.stages
            assert "B4" in r.stages


def test_stage_b_unified_infra_failure_never_writes_audio_even_keep_rejected():
    """keep_rejected=true 时 B3 infra 失败（vad_batch_failed）也不得写出音频。

    回归：策略层 keep_rejected=true 会把 vad_batch_failed 记录放行并写出
    音频，把未经 VAD 判定的数据送入下游。修复后无条件按 rejected 处理。
    """
    from noise_filter_pipeline.processors.stage_b_unified import StageBUnifiedProcessor

    with tempfile.TemporaryDirectory() as tmpdir:
        proc = StageBUnifiedProcessor({
            "b1": {"mode": "adaptive"},
            "b2": {"enable": False},
            "b3": {"enable": True},
            "b4": {"keep_noise": True, "keep_suspected": True, "keep_rejected": True},
            "num_workers": 1,
        })
        audio_dir = os.path.join(tmpdir, "audio")
        os.makedirs(audio_dir)

        def fake_fail(records, arrays):
            for rec in records:
                rec.mark_stage("B3", "rejected", extra={"reason": "vad_batch_failed"})
                rec.rejected_reason = "b3_vad_batch_failed"
            return records

        proc._vad_processor.process_records_with_arrays = fake_fail
        rec = _make_record(os.path.join(tmpdir, "fail.wav"), 2.0)
        audio = np.zeros(16000, dtype=np.float32)

        out = proc._process_batch([rec], [(audio, 16000)], audio_dir)
        assert out[0].stages["B4"]["status"] == "rejected"
        assert not out[0].is_stage_passed("B4")
        # 不得有任何音频文件落盘
        assert os.listdir(audio_dir) == []


def test_stage_b_unified_has_retryable_failures_on_vad_batch_failed():
    """vad_batch_failed 记录经 flush 落盘后 has_retryable_failures() 必须为真。

    回归：B unified 此前不暴露可重试失败计数，VAD 整批失败时 stage 照常
    "成功"完成（写统计/无告警），下游消费空 manifest。stage_b_curation
    依赖该计数中止 stage。
    """
    from noise_filter_pipeline.processors.stage_b_unified import StageBUnifiedProcessor

    with tempfile.TemporaryDirectory() as tmpdir:
        proc = StageBUnifiedProcessor({
            "b1": {"mode": "adaptive"},
            "b2": {"enable": False},
            "b3": {"enable": True},
            "b4": {"keep_noise": True, "keep_suspected": True, "keep_rejected": True},
            "num_workers": 1,
        })
        assert not proc.has_retryable_failures()

        rec = _make_record(os.path.join(tmpdir, "fail.wav"), 2.0)
        rec.mark_stage("B3", "rejected", extra={"reason": "vad_batch_failed"})
        rec.rejected_reason = "b3_vad_batch_failed"
        # 隔离 VAD 真实路径：直接 mock _prepare_batch 返回预标记记录，
        # 单独验证 _flush_batch 的可重试计数逻辑
        proc._prepare_batch = lambda records, arrays, vad_mask, audio_dir: (records, [])
        stats = {"total": 0, "kept": 0, "discarded": 0, "b2_rejected": 0}

        class _Writer:
            def append(self, r):
                pass

        class _Ckpt:
            def mark_done_batch(self, ids):
                pass

        audio_dir = os.path.join(tmpdir, "audio")
        os.makedirs(audio_dir)
        proc._flush_batch([rec], [], audio_dir, _Writer(), stats, _Ckpt())
        assert proc.has_retryable_failures()


def test_b2_rejection_reason_uses_max_silence_ratio_threshold():
    """silence_ratio 越过 max_silence_ratio（默认 0.9）即应归因 silence_ratio。

    回归：旧口径写死 >=1.0，0.9~1.0 之间的拒绝被误归为 silence_filter。
    """
    from noise_filter_pipeline.processors.stage_b_unified import StageBUnifiedProcessor

    with tempfile.TemporaryDirectory() as tmpdir:
        proc = StageBUnifiedProcessor({
            "b1": {"mode": "adaptive"},
            "b2": {"enable": True, "max_silence_ratio": 0.9},
            "b3": {"enable": False},
            "b4": {"keep_noise": True},
            "num_workers": 1,
        })
        result = {
            "silence_ratio": 0.95,
            "is_silent": False,
            "dc_offset": 0.0,
            "dc_offset_rejected": False,
        }
        assert proc._b2_rejection_reason(result) == "silence_ratio"


def test_stage_b_unified_checkpoint_resume_idempotent():
    """Stage B unified 应支持 checkpoint / resume：重跑时跳过已处理记录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        records = []
        for i in range(2):
            wav_path = os.path.join(tmpdir, f"src_{i}.wav")
            _make_wav(wav_path, duration=12.0)
            rec = AudioRecord(
                record_id=f"ds__src_{i}__source",
                dataset_id="ds",
                source_id=f"src_{i}",
                chunk_id="source",
                audio={"path": wav_path, "sr": 16000, "duration": 12.0, "channels": 1},
                labels={"original": "noise"},
            )
            rec.mark_stage("A", "passed")
            records.append(rec)

        a_manifest = os.path.join(tmpdir, "a_manifest.jsonl")
        write_manifest(iter(records), a_manifest)

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_b={
                "unified": True,
                "b1": {"mode": "adaptive", "chunk_duration": 5.0, "min_chunk_duration": 2.0},
                "b2": {"enable": True, "max_silence_ratio": 0.95},
                "b3": {"enable": False},
                "b4": {"keep_noise": True, "keep_suspected": True, "keep_rejected": True},
                "num_workers": 1,
                "checkpoint_save_interval": 1,
                "manifest_write_batch_size": 1,
            },
        )
        orch = StageBOrchestrator(config)
        output_dir = os.path.join(tmpdir, "out", "stage_b")
        output_manifest = os.path.join(output_dir, "b_manifest.jsonl")

        b_manifest1 = orch.run(
            input_manifest=a_manifest,
            output_dir=output_dir,
            output_manifest=output_manifest,
        )
        recs1 = list(read_manifest(b_manifest1))
        assert len(recs1) >= 2
        checkpoint_path = os.path.join(output_dir, "checkpoint.json")
        assert os.path.isfile(checkpoint_path)

        # 第二次重跑应通过 checkpoint 跳过所有记录，结果与第一次一致
        b_manifest2 = orch.run(
            input_manifest=a_manifest,
            output_dir=output_dir,
            output_manifest=output_manifest,
        )
        recs2 = list(read_manifest(b_manifest2))
        assert len(recs2) == len(recs1)
        for r in recs2:
            if r.is_stage_passed("B4"):
                assert os.path.isfile(r.audio["path"])


def test_stage_b_unified_vad_batch_failed_retryable():
    """VAD 整批失败（b3_vad_batch_failed）不写 checkpoint，resume 时可重试；
    正常处理（含内容级拒绝）的记录仍写入 checkpoint。"""
    import numpy as np

    from noise_filter_pipeline.core.checkpoint import CheckpointTracker
    from noise_filter_pipeline.core.manifest_io import BatchedManifestWriter
    from noise_filter_pipeline.processors.stage_b_unified import StageBUnifiedProcessor

    with tempfile.TemporaryDirectory() as tmpdir:
        proc = StageBUnifiedProcessor({
            "b1": {"mode": "adaptive"},
            "b2": {"enable": False},
            "b3": {"enable": True},
            "b4": {"keep_noise": True},
            "num_workers": 1,
        })

        audio_dir = os.path.join(tmpdir, "audio")
        os.makedirs(audio_dir)
        manifest_path = os.path.join(tmpdir, "b_manifest.jsonl.tmp")
        ckpt = CheckpointTracker(
            os.path.join(tmpdir, "checkpoint.json"), enabled=True, save_interval=1
        )
        stats = {"total": 0, "kept": 0, "discarded": 0}
        audio = np.zeros(16000, dtype=np.float32)

        # 1) VAD 整批失败：模拟 strict 路径的标记方式
        def fake_fail(records, arrays):
            for rec in records:
                rec.mark_stage("B3", "rejected", extra={"reason": "vad_batch_failed"})
                rec.rejected_reason = "b3_vad_batch_failed"
            return records

        proc._vad_processor.process_records_with_arrays = fake_fail
        rec_fail = _make_record(os.path.join(tmpdir, "fail.wav"), 2.0)

        writer = BatchedManifestWriter(manifest_path, batch_size=1)
        with writer:
            proc._flush_batch([rec_fail], [(audio, 16000)], audio_dir, writer, stats, ckpt)
        assert not ckpt.is_done(rec_fail.record_id), "infra 失败不应写入 checkpoint"

        # 2) 正常 VAD 通过：应写入 checkpoint
        def fake_pass(records, arrays):
            for rec in records:
                rec.mark_stage(
                    "B3", "passed", extra={"vad_result": {"speech_ratio": 0.0}}
                )
            return records

        proc._vad_processor.process_records_with_arrays = fake_pass
        rec_ok = _make_record(os.path.join(tmpdir, "ok.wav"), 2.0)

        writer2 = BatchedManifestWriter(manifest_path, batch_size=1)
        with writer2:
            proc._flush_batch([rec_ok], [(audio, 16000)], audio_dir, writer2, stats, ckpt)
        assert ckpt.is_done(rec_ok.record_id), "正常记录应写入 checkpoint"
        assert rec_ok.is_stage_passed("B4")


def test_stage_b_unified_resume_missing_prev_record_reprocesses():
    """checkpoint 命中但 prev_records 缺少条目时不得静默跳过，必须回退为重新处理。"""
    import json

    from noise_filter_pipeline.processors.stage_b_unified import StageBUnifiedProcessor

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "src.wav")
        _make_wav(wav_path, duration=3.0)

        a_manifest = os.path.join(tmpdir, "a_manifest.jsonl")
        write_manifest(iter([_make_record(wav_path, 3.0)]), a_manifest)

        cfg = {
            "b1": {"mode": "adaptive"},
            "b2": {"enable": False},
            "b3": {"enable": False},
            "b4": {"keep_noise": True},
            "num_workers": 1,
            "checkpoint_save_interval": 1,
            "manifest_write_batch_size": 1,
        }
        output_dir = os.path.join(tmpdir, "out", "stage_b")

        # 第一次正常跑完，拿到 record_id 与 checkpoint
        proc = StageBUnifiedProcessor(dict(cfg))
        b_manifest = proc.process(a_manifest, output_dir)
        recs1 = list(read_manifest(b_manifest))
        assert len(recs1) >= 1
        done_ids = [r.record_id for r in recs1]

        # 模拟异常现场：checkpoint 仍在，但输出 manifest 丢失（prev_records 为空）
        for f in os.listdir(output_dir):
            if f.startswith("b_manifest_") and f.endswith(".jsonl"):
                os.remove(os.path.join(output_dir, f))
        checkpoint_path = os.path.join(output_dir, "checkpoint.json")
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"done": done_ids}, f)

        # 第二次跑：不得因 checkpoint 命中而丢记录，应全部重新处理
        proc2 = StageBUnifiedProcessor(dict(cfg))
        b_manifest2 = proc2.process(a_manifest, output_dir)
        recs2 = list(read_manifest(b_manifest2))
        assert len(recs2) == len(recs1), "checkpoint 命中但 prev 缺失的记录必须重新处理"
        for r in recs2:
            assert r.is_stage_passed("B4")


def test_stage_b_unified_output_filename_deterministic():
    """同一 record_id / original_path 重跑 _process_batch 必须得到同一输出文件名。

    回归测试：输出文件名的 hash 源曾是随机临时 chunk 路径，崩溃 resume 后
    同一 chunk 会写出第二个不同名文件，audio 目录积累孤儿重复。
    """
    from noise_filter_pipeline.processors.stage_b_unified import StageBUnifiedProcessor

    with tempfile.TemporaryDirectory() as tmpdir:
        proc = StageBUnifiedProcessor({
            "b1": {"mode": "adaptive"},
            "b2": {"enable": False},
            "b3": {"enable": False},
            "b4": {"keep_noise": True},
            "num_workers": 1,
        })
        audio_dir = os.path.join(tmpdir, "audio")
        os.makedirs(audio_dir)
        audio = np.random.randn(16000).astype(np.float32) * 0.01

        def run_once(tmp_chunk_path):
            rec = AudioRecord(
                record_id="ds__src__chunk_0001",
                dataset_id="ds",
                source_id="src",
                chunk_id="chunk_0001",
                audio={
                    "path": tmp_chunk_path,
                    "original_path": "/data/original/src.wav",
                    "sr": 16000,
                    "duration": 1.0,
                    "channels": 1,
                },
                labels={"original": "noise"},
            )
            rec.mark_stage("A", "passed")
            # 与真实 pipeline 一致：B3 禁用时先标记 passed，否则策略过滤会按 vad_rejected 拒绝
            rec.mark_stage("B3", "passed", extra={"enabled": False})
            out = proc._process_batch([rec], [(audio, 16000)], audio_dir)
            assert out[0].is_stage_passed("B4")
            return out[0].audio["path"]

        # 两次运行的临时 chunk 路径不同（模拟不同 TemporaryDirectory），
        # 最终输出文件名必须一致
        p1 = run_once(os.path.join(tmpdir, "run1", "chunk.wav"))
        p2 = run_once(os.path.join(tmpdir, "run2", "chunk.wav"))
        assert p1 == p2
        assert os.path.isfile(p1)


def test_stage_b_unified_output_filename_distinct_chunks():
    """同一 original_path 的不同 chunk 必须得到不同输出文件名，不得互相覆盖。"""
    from noise_filter_pipeline.processors.stage_b_unified import StageBUnifiedProcessor

    with tempfile.TemporaryDirectory() as tmpdir:
        proc = StageBUnifiedProcessor({
            "b1": {"mode": "adaptive"},
            "b2": {"enable": False},
            "b3": {"enable": False},
            "b4": {"keep_noise": True},
            "num_workers": 1,
        })
        audio_dir = os.path.join(tmpdir, "audio")
        os.makedirs(audio_dir)
        audio = np.random.randn(16000).astype(np.float32) * 0.01

        recs = []
        arrays = []
        for chunk_id in ("chunk_0001", "chunk_0002"):
            rec = AudioRecord(
                record_id=f"ds__src__{chunk_id}",
                dataset_id="ds",
                source_id="src",
                chunk_id=chunk_id,
                audio={
                    "path": os.path.join(tmpdir, f"{chunk_id}.wav"),
                    "original_path": "/data/original/src.wav",
                    "sr": 16000,
                    "duration": 1.0,
                    "channels": 1,
                },
                labels={"original": "noise"},
            )
            rec.mark_stage("A", "passed")
            rec.mark_stage("B3", "passed", extra={"enabled": False})
            recs.append(rec)
            arrays.append((audio, 16000))

        out = proc._process_batch(recs, arrays, audio_dir)
        paths = [r.audio["path"] for r in out if r.is_stage_passed("B4")]
        assert len(paths) == 2
        assert paths[0] != paths[1]


def test_stage_b_unified_writes_kept_audio_concurrently(monkeypatch, tmp_path):
    """最终 FLAC 写出应使用文件级并发，避免串行编码成为 flush 主瓶颈。"""
    import threading
    import time

    from noise_filter_pipeline.processors import stage_b_unified
    from noise_filter_pipeline.processors.stage_b_unified import StageBUnifiedProcessor

    proc = StageBUnifiedProcessor({
        "b1": {"mode": "adaptive"},
        "b2": {"enable": False},
        "b3": {"enable": False},
        "b4": {"keep_noise": True},
        "num_workers": 2,
    })
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = np.zeros(16000, dtype=np.float32)

    active = 0
    max_active = 0
    lock = threading.Lock()
    release = threading.Event()

    def fake_save_audio(audio_array, sr, path):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            if max_active >= 2:
                release.set()
        release.wait(timeout=0.5)
        time.sleep(0.01)
        with lock:
            active -= 1

    monkeypatch.setattr(stage_b_unified, "save_audio_atomically", fake_save_audio)

    records = []
    arrays = []
    for i in range(2):
        rec = AudioRecord(
            record_id=f"ds__src__chunk_{i:04d}",
            dataset_id="ds",
            source_id="src",
            chunk_id=f"chunk_{i:04d}",
            audio={
                "path": str(tmp_path / f"tmp_{i}.wav"),
                "original_path": str(tmp_path / "original.wav"),
                "sr": 16000,
                "duration": 1.0,
                "channels": 1,
            },
        )
        rec.mark_stage("B3", "passed", extra={"enabled": False})
        records.append(rec)
        arrays.append((audio, 16000))

    out = proc._process_batch(records, arrays, str(audio_dir))

    assert all(rec.is_stage_passed("B4") for rec in out)
    assert max_active >= 2


@pytest.mark.parametrize(
    ("invalid_kind", "expected_error"),
    [
        ("malformed", "invalid JSON"),
        ("duplicate", "duplicate record_id"),
    ],
    ids=["malformed", "duplicate-id"],
)
def test_stage_b_fallback_preflight_rejects_invalid_input_without_side_effects(
    tmp_path, invalid_kind, expected_error
):
    """fallback 必须在 B1 落盘前完整严格预检输入。"""
    wav_path = tmp_path / "src.wav"
    _make_wav(str(wav_path), duration=2.0)
    record_line = json.dumps(_make_record(str(wav_path), 2.0).to_dict())
    invalid_line = "{broken json}" if invalid_kind == "malformed" else record_line
    input_text = f"{record_line}\n{invalid_line}\n"
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(input_text, encoding="utf-8")

    config = UnifiedPipelineConfig(stage_b={
        "unified": False,
        "b1": {"mode": "none"},
        "b2": {"enable": False},
        "b3": {"enable": False},
        "b4": {"keep_noise": True},
        "num_workers": 1,
    })
    output_dir = tmp_path / "stage_b"

    with pytest.raises(ValueError, match=expected_error):
        StageBOrchestrator(config).run(
            str(manifest), str(output_dir), str(output_dir / "b_manifest.jsonl")
        )

    assert manifest.read_text(encoding="utf-8") == input_text
    assert wav_path.is_file()
    assert not output_dir.exists()


def test_stage_b_unified_rejects_duplicate_record_ids_globally(tmp_path):
    """Stage B unified strict reader 必须在处理前拒绝重复 record_id。"""
    from noise_filter_pipeline.processors.stage_b_unified import StageBUnifiedProcessor

    wav_path = tmp_path / "src.wav"
    _make_wav(str(wav_path), duration=2.0)
    record = _make_record(str(wav_path), 2.0).to_dict()
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8"
    )
    proc = StageBUnifiedProcessor({
        "b1": {"mode": "none"},
        "b2": {"enable": False},
        "b3": {"enable": False},
        "b4": {"keep_noise": True},
        "num_workers": 1,
    })
    with pytest.raises(ValueError, match="duplicate record_id"):
        proc.process(str(manifest), str(tmp_path / "out"))


def test_stage_b_unified_rejects_malformed_manifest_line(tmp_path):
    """Stage B unified 不得沿用宽松 reader 静默吞掉坏行。"""
    from noise_filter_pipeline.processors.stage_b_unified import StageBUnifiedProcessor

    manifest = tmp_path / "input.jsonl"
    manifest.write_text("{broken json}\n", encoding="utf-8")
    proc = StageBUnifiedProcessor({
        "b1": {"mode": "none"},
        "b2": {"enable": False},
        "b3": {"enable": False},
        "b4": {"keep_noise": True},
        "num_workers": 1,
    })
    with pytest.raises(ValueError, match="invalid JSON"):
        proc.process(str(manifest), str(tmp_path / "out"))


def test_stage_b_unified_failure_does_not_promote_tmp():
    """处理中途失败时，tmp manifest 不得原子替换为最终 manifest（tmp 保留供 resume）。"""
    import pytest

    from noise_filter_pipeline.processors.stage_b_unified import StageBUnifiedProcessor

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "src.wav")
        _make_wav(wav_path, duration=4.0)
        a_manifest = os.path.join(tmpdir, "a.jsonl")
        write_manifest(iter([_make_record(wav_path, 4.0)]), a_manifest)

        proc = StageBUnifiedProcessor({
            "b1": {"mode": "adaptive", "chunk_duration": 2.0, "min_chunk_duration": 1.0},
            "b2": {"enable": False},
            "b3": {"enable": False},
            "b4": {"keep_noise": True},
            "num_workers": 1,
        })

        def boom(*args, **kwargs):
            raise RuntimeError("simulated prepare failure")

        # 新两级流水线下，prepare(batch N+1) 阶段异常即中止整个 stage；
        # 注入点从旧 _flush_batch 迁移到 _prepare_batch
        proc._prepare_batch = boom
        out_dir = os.path.join(tmpdir, "out")
        with pytest.raises(RuntimeError):
            proc.process(a_manifest, out_dir)

        assert not os.path.exists(os.path.join(out_dir, "b_manifest.jsonl")), \
            "失败时不得生成遗留单文件"
        assert not os.path.exists(os.path.join(out_dir, "b_manifest.jsonl.tmp")), \
            "per-dataset 架构无 tmp 中间文件（resume 靠 checkpoint + prev_records）"


def test_stage_b_silero_group_size_is_runtime_only_checkpoint_config(monkeypatch):
    """silero_group_size 是调度参数：运行时保留，但不得改变旧 checkpoint hash。"""
    from noise_filter_pipeline.core.manifest_io import config_hash
    from noise_filter_pipeline.processors import stage_b_unified
    from noise_filter_pipeline.processors.stage_b_unified import StageBUnifiedProcessor

    seen = []

    class _VADProcessor:
        def __init__(self, cfg):
            seen.append(dict(cfg))

        def close(self):
            pass

    monkeypatch.setattr(stage_b_unified, "VADProcessor", _VADProcessor)
    base = {
        "b1": {"mode": "none"},
        "b2": {"enable": False},
        "b3": {"enable": False},
        "b4": {"keep_noise": True},
        "num_workers": 1,
    }
    old = StageBUnifiedProcessor(dict(base))
    tuned_cfg = dict(base)
    tuned_cfg["b3"] = {"enable": False, "silero_group_size": 7}
    tuned = StageBUnifiedProcessor(tuned_cfg)

    assert config_hash(old.config) == config_hash(tuned.config)
    assert tuned.b3_cfg["silero_group_size"] == 7
    assert seen[-1]["silero_group_size"] == 7


def test_stage_b_overlaps_write_n_with_vad_n_plus_1_single_lane(
    monkeypatch, tmp_path
):
    """write(N) 阻塞期间 VAD(N+1) 应运行，且 VAD 最大并发始终为 1。"""
    import threading

    from noise_filter_pipeline.processors import stage_b_unified
    from noise_filter_pipeline.processors.stage_b_unified import StageBUnifiedProcessor

    records = []
    for i in range(2):
        wav_path = tmp_path / f"src_{i}.wav"
        _make_wav(str(wav_path), duration=1.0)
        rec = AudioRecord(
            record_id=f"ds__src_{i}__source",
            dataset_id="ds",
            source_id=f"src_{i}",
            chunk_id="source",
            audio={"path": str(wav_path), "sr": 16000, "duration": 1.0, "channels": 1},
        )
        rec.mark_stage("A", "passed")
        records.append(rec)
    manifest = tmp_path / "a.jsonl"
    write_manifest(iter(records), str(manifest))

    write_started = threading.Event()
    second_vad_started = threading.Event()
    allow_write = threading.Event()
    active_vad = 0
    max_active_vad = 0
    vad_calls = 0
    lock = threading.Lock()

    def fake_vad(batch_records, arrays):
        nonlocal active_vad, max_active_vad, vad_calls
        with lock:
            active_vad += 1
            max_active_vad = max(max_active_vad, active_vad)
            vad_calls += 1
            call = vad_calls
        if call == 2:
            second_vad_started.set()
        for rec in batch_records:
            rec.mark_stage("B3", "passed", extra={"enabled": False})
        with lock:
            active_vad -= 1
        return batch_records

    def fake_save(audio, sr, path):
        write_started.set()
        assert second_vad_started.wait(timeout=2), "write(N) 未与 VAD(N+1) 重叠"
        assert allow_write.wait(timeout=2)

    proc = StageBUnifiedProcessor({
        "b1": {"mode": "none"},
        "b2": {"enable": False},
        "b3": {"enable": True},
        "b4": {"keep_noise": True},
        "num_workers": 1,
        "vad_batch_chunks": 1,
        "checkpoint_save_interval": 100,
        "manifest_write_batch_size": 100,
    })
    proc._vad_processor.process_records_with_arrays = fake_vad
    monkeypatch.setattr(stage_b_unified, "save_audio_atomically", fake_save, raising=False)

    thread = threading.Thread(
        target=lambda: proc.process(str(manifest), str(tmp_path / "out")),
        daemon=True,
    )
    thread.start()
    assert write_started.wait(timeout=2)
    assert second_vad_started.wait(timeout=2)
    allow_write.set()
    thread.join(timeout=3)

    assert not thread.is_alive()
    assert vad_calls == 2
    assert max_active_vad == 1


def test_vad_batch_chunks_clamped_by_available_memory(monkeypatch):
    """批内 chunk 数必须随可用内存钳制；内存充足或估算不出时保持配置值。"""
    from noise_filter_pipeline.processors import stage_b_unified
    from noise_filter_pipeline.processors.stage_b_unified import (
        StageBUnifiedProcessor,
        _clamp_batch_chunks_by_memory,
    )

    budget_bytes = 2 * 1024 ** 3 * 0.25
    bytes_per_chunk = 10.0 * 48000 * 4  # chunk_duration × 最高常见采样率 × float32

    monkeypatch.setattr(stage_b_unified, "_available_memory_bytes", lambda: 2 * 1024 ** 3)
    clamped = _clamp_batch_chunks_by_memory(2000, chunk_duration=10.0)
    assert 0 < clamped < 2000
    # 钳制后的整批音频必须落在预算内（按 48k 最坏情况估算）
    assert clamped * bytes_per_chunk <= budget_bytes
    # 配置值本身就在预算内时不得下调
    assert _clamp_batch_chunks_by_memory(clamped, chunk_duration=10.0) == clamped

    # 平台不支持 sysconf（返回 None）时保持配置值
    monkeypatch.setattr(stage_b_unified, "_available_memory_bytes", lambda: None)
    assert _clamp_batch_chunks_by_memory(2000, chunk_duration=10.0) == 2000

    # 可用内存低于单批最小预算：兜底为 1，绝不返回 0
    monkeypatch.setattr(stage_b_unified, "_available_memory_bytes", lambda: 1)
    assert _clamp_batch_chunks_by_memory(2000, chunk_duration=10.0) == 1

    # 处理器构造路径必须应用钳制结果
    monkeypatch.setattr(stage_b_unified, "_available_memory_bytes", lambda: 2 * 1024 ** 3)
    proc = StageBUnifiedProcessor({
        "b1": {"mode": "none", "chunk_duration": 10.0},
        "b3": {"enable": True},
        "b4": {"keep_noise": True},
        "num_workers": 1,
        "vad_batch_chunks": 2000,
    })
    assert proc.vad_batch_chunks == clamped


def test_stage_b_unified_manifest_strict_input_order_with_mixed_types(
    tmp_path,
):
    """eligible / B2-rejected / B1-rejected / checkpoint 记录必须严格按输入顺序落盘。

    回归：旧实现由主线程直接 append rejected/resume 记录，会越过尚未完成
    的后台 batch，造成 per-dataset manifest 乱序。新 commit lane 必须保序。
    """
    from noise_filter_pipeline.processors.stage_b_unified import StageBUnifiedProcessor

    # 1) eligible：普通噪声 wav
    elig_wav = tmp_path / "elig.wav"
    _make_wav(str(elig_wav), duration=1.0)
    # 2) B2-rejected：纯静音 wav（max_silence_ratio 默认 0.9）
    silent_wav = tmp_path / "silent.wav"
    sf.write(str(silent_wav), np.full(16000, 0.001, dtype=np.float32), 16000)
    # 3) B1-rejected：缺失文件
    missing_wav = tmp_path / "missing.wav"

    def make(source_id, wav_path):
        return AudioRecord(
            record_id=f"ds__{source_id}__source",
            dataset_id="ds",
            source_id=source_id,
            chunk_id="source",
            audio={"path": str(wav_path), "sr": 16000, "duration": 1.0, "channels": 1},
            labels={"original": "noise"},
        )

    def make_2nd_eligible(source_id):
        """第二个 eligible 记录：直接构造 B1/B2 passed 的 chunk 记录。"""
        rec = make(source_id, elig_wav)
        rec.stages["B1"] = {"status": "passed", "extra": {}}
        rec.stages["B2"] = {"status": "passed", "extra": {}}
        return rec

    # 输入顺序：eligible-0, B2-rejected(silent), eligible-1, B1-rejected(missing)
    rec_elig0 = make_2nd_eligible("elig0")
    rec_silent = make("silent", silent_wav)
    rec_elig1 = make_2nd_eligible("elig1")
    rec_missing = make("missing", missing_wav)

    a_manifest = tmp_path / "a.jsonl"
    write_manifest(
        iter([rec_elig0, rec_silent, rec_elig1, rec_missing]), str(a_manifest)
    )

    proc = StageBUnifiedProcessor({
        "b1": {"mode": "adaptive", "chunk_duration": 10.0, "min_chunk_duration": 0.5},
        "b2": {"enable": True},
        "b3": {"enable": False},
        "b4": {"keep_noise": True},
        "num_workers": 1,
        "vad_batch_chunks": 1,  # 每条单独成批，最大化跨批交错
        "checkpoint_save_interval": 100,
        "manifest_write_batch_size": 100,
        "checkpoint_path": str(tmp_path / "out" / "checkpoint.json"),
    })
    out_dir = str(tmp_path / "out")
    proc.process(str(a_manifest), out_dir)

    from noise_filter_pipeline.core.manifest_io import read_manifest

    # per-dataset writer 产出 b_manifest_*.jsonl；read_manifest 支持目录。
    # 分片后 chunk 记录 id 形如 ds__<source>__source__chunk_0001；按 source 位断言
    # 相对顺序即可同时覆盖 chunk 化与 B1/B2 拒绝的保序
    ids = [r.record_id for r in read_manifest(out_dir)]
    sources = [rid.split("__")[1] for rid in ids]
    assert sources == ["elig0", "silent", "elig1", "missing"]


def test_stage_b_unified_write_infra_failure_aborts_without_commit(
    monkeypatch, tmp_path
):
    """最终写盘基础设施失败（ENOSPC/EIO）必须中止 stage，不得 commit/写 checkpoint。

    回归：旧实现把写盘失败当作内容级 b4_write_error 永久 checkpoint，
    磁盘故障会被误判为数据质量问题。
    """
    from noise_filter_pipeline.processors import stage_b_unified
    from noise_filter_pipeline.processors.stage_b_unified import StageBUnifiedProcessor

    wav_path = tmp_path / "src.wav"
    _make_wav(str(wav_path), duration=1.0)
    a_manifest = tmp_path / "a.jsonl"
    write_manifest(iter([_make_record(str(wav_path), 1.0)]), a_manifest)

    def boom(*args, **kwargs):
        raise OSError(28, "No space left on device")

    proc = StageBUnifiedProcessor({
        "b1": {"mode": "none"},
        "b2": {"enable": False},
        "b3": {"enable": False},
        "b4": {"keep_noise": True},
        "num_workers": 1,
    })
    monkeypatch.setattr(stage_b_unified, "save_audio_atomically", boom)

    out_dir = str(tmp_path / "out")
    with pytest.raises(OSError, match="No space left"):
        proc.process(str(a_manifest), out_dir)

    # 不得生成任何最终 manifest（commit 未发生），也无 checkpoint 记录
    assert not os.path.exists(os.path.join(out_dir, "b_manifest.jsonl"))
    ckpt_path = os.path.join(out_dir, "checkpoint.json")
    if os.path.isfile(ckpt_path):
        import json as _json

        assert _json.load(open(ckpt_path, encoding="utf-8"))["done"] == []
