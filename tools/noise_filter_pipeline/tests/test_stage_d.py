"""
Tests for Stage D enhancement (dry-run / lightweight)。
"""
import errno
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from noise_filter_pipeline.core.manifest_io import AudioRecord, write_manifest, read_manifest
from noise_filter_pipeline.core.pipeline_config import UnifiedPipelineConfig
from noise_filter_pipeline.processors import apollo_superres
from noise_filter_pipeline.processors.apollo_superres import ApolloSuperResProcessor
from noise_filter_pipeline.processors.loudness_normalizer import LoudnessNormalizerProcessor
from noise_filter_pipeline.stages import stage_d_enhancement as stage_d_module
from noise_filter_pipeline.stages.stage_d_enhancement import StageDOrchestrator

# 本文件使用 dry-run / mock，仅覆盖本地逻辑；真实 Apollo 模型另属 external。
pytestmark = pytest.mark.unit


def _make_wav(path, duration=2.0, sr=16000, rms=0.1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wav = np.random.randn(int(duration * sr)) * rms
    sf.write(path, wav, sr)


def _c2_labels(policy, hit_mid):
    """构造固定五项、按概率降序排列的 AudioSet 证据。"""
    filler_mids = [
        mid
        for mid in policy.index_to_mid.values()
        if mid != hit_mid and policy.class_of(mid) == "noise"
    ][:4]
    mids = [hit_mid, *filler_mids]
    probabilities = [1.0, 0.8, 0.6, 0.4, 0.2]
    return {
        "audioset_top5": [
            {
                "index": policy.mid_to_index[mid],
                "mid": mid,
                "label": policy.label_of(mid),
                "prob": probability,
            }
            for mid, probability in zip(mids, probabilities)
        ],
    }


def _record_with_audio(tmpdir, idx, labels, stage_status=None, sr=16000, duration=2.0):
    wav_path = os.path.join(tmpdir, f"audio_{idx}.wav")
    _make_wav(wav_path, duration=duration, sr=sr)
    rec = AudioRecord(
        record_id=f"ds__src__{idx}",
        dataset_id="ds",
        source_id="src",
        chunk_id=f"chunk_{idx}",
        audio={"path": wav_path, "sr": sr, "duration": duration, "channels": 1},
        labels=labels,
    )
    if stage_status:
        for stage, status in stage_status.items():
            extra = None
            if stage == "C3" and status == "passed":
                from core.data.noise_admission_policy import (
                    CLASSIFIER_LABEL_COUNT,
                    EVIDENCE_CONTRACT_VERSION,
                    SCORE_CONTRACT,
                    decision_to_admission_extra,
                    evaluate_noise_admission,
                    load_admission_policy,
                    parse_c2_evidence,
                )

                policy = load_admission_policy()
                requested_label = labels.get("audioset", "Rain")
                requested_mid = next(
                    (
                        mid
                        for mid, label in policy.mid_to_label.items()
                        if label == requested_label and policy.class_of(mid) == "noise"
                    ),
                    next(
                        mid
                        for mid, label in policy.mid_to_label.items()
                        if label == "Rain"
                    ),
                )
                labels.clear()
                labels.update(_c2_labels(policy, requested_mid))
                c2_extra = {
                    "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
                    "score_contract": SCORE_CONTRACT,
                    "class_count": CLASSIFIER_LABEL_COUNT,
                    "label_space_sha256": policy.label_space_sha256,
                }
                evidence = parse_c2_evidence(labels, policy, c2_extra)
                decision = evaluate_noise_admission(
                    c1_intelligibility="noise",
                    evidence=evidence,
                    policy=policy,
                )
                rec.mark_stage("C1", "passed", extra={"intelligibility": "noise"})
                rec.mark_stage("C2", "passed", extra=c2_extra)
                extra = decision_to_admission_extra(
                    decision,
                    execution_mode={"c1": "production", "c2": "subprocess_daemon"},
                    publishable=True,
                    artifact_fingerprints={
                        "voice_policy_sha256": policy.sha256,
                        "audiotag_label_dict_sha256": policy.label_space_sha256,
                        "audiotag_checkpoint_sha256": "2" * 64,
                        "qwen_model_sha256": "4" * 64,
                        "c1_prompt_sha256": "5" * 64,
                    },
                )
                rec.mark_stage("C3", "passed", extra=extra)
                continue
            rec.mark_stage(stage, status, extra=extra)
    return rec


def _read_ckpt_done(out_dir, name="checkpoint.json"):
    """读取 checkpoint 文件中已完成的 record_id 集合。"""
    ckpt_path = os.path.join(out_dir, name)
    if not os.path.isfile(ckpt_path):
        return set()
    with open(ckpt_path, "r", encoding="utf-8") as f:
        return set(json.load(f).get("done", []))


def _write_mock_apollo_audio_daemon(script_path):
    """写一个只生成可读中间音频的 Apollo daemon mock，不加载模型。"""
    code = r'''
import json
import os
import shutil
import sys

output_dir = "."
args = sys.argv[1:]
for index, arg in enumerate(args):
    if arg == "--output_audio_dir" and index + 1 < len(args):
        output_dir = args[index + 1]
os.makedirs(output_dir, exist_ok=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    command = json.loads(line)
    command_type = command.get("type")
    if command_type == "ping":
        print(json.dumps({"type": "pong"}), flush=True)
    elif command_type == "enhance":
        results = []
        for item in command.get("items", []):
            source = item["audio_path"]
            stem = os.path.splitext(os.path.basename(source))[0]
            enhanced = os.path.join(output_dir, stem + "_enhanced.wav")
            shutil.copyfile(source, enhanced)
            results.append({"audio_path": source, "enhanced_audio_path": enhanced})
        print(json.dumps({
            "type": "result",
            "batch_id": command.get("batch_id"),
            "results": results,
        }), flush=True)
    elif command_type == "stop":
        print(json.dumps({"type": "stopped"}), flush=True)
        break
'''
    Path(script_path).write_text(code, encoding="utf-8")


def test_apollo_superres_dry_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [_record_with_audio(tmpdir, i, {"leaf": "noise"}, {"D2": "passed"}, sr=44100) for i in range(2)]
        write_manifest(iter(recs), manifest)

        proc = ApolloSuperResProcessor({"mode": "dry_run", "target_sr": 44100})
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "d1"))
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 2
        assert all(r.is_stage_passed("D1") for r in out_recs)
        assert all(os.path.isfile(r.audio["path"]) for r in out_recs)


def test_apollo_superres_missing_model():
    with pytest.raises(FileNotFoundError):
        proc = ApolloSuperResProcessor({
            "mode": "subprocess_daemon",
            "config_path": "/nonexistent/config.yaml",
            "checkpoint": "/nonexistent/apollo_model_uni.ckpt",
        })
        proc._validate_model_paths()


def test_d2_same_basename_inputs_do_not_overwrite():
    """不同目录下的同名输入，D2 输出不得互相覆盖。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_a = os.path.join(tmpdir, "ds_a", "clip.wav")
        wav_b = os.path.join(tmpdir, "ds_b", "clip.wav")
        _make_wav(wav_a, duration=2.0)
        _make_wav(wav_b, duration=1.0)

        recs = [
            AudioRecord(
                record_id=f"ds_a__src__0", dataset_id="ds_a", source_id="src", chunk_id="c0",
                audio={"path": wav_a, "sr": 16000, "duration": 2.0, "channels": 1},
                labels={"original": "noise"},
            ),
            AudioRecord(
                record_id=f"ds_b__src__0", dataset_id="ds_b", source_id="src", chunk_id="c0",
                audio={"path": wav_b, "sr": 16000, "duration": 1.0, "channels": 1},
                labels={"original": "noise"},
            ),
        ]
        manifest = os.path.join(tmpdir, "input.jsonl")
        write_manifest(iter(recs), manifest)

        proc = LoudnessNormalizerProcessor({"target_loudness": -20.0})
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "d2"))
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 2
        paths = [r.audio["path"] for r in out_recs]
        assert paths[0] != paths[1]
        assert all(os.path.isfile(p) for p in paths)
        # 两条记录应各自指向时长正确的独立文件
        dur_a = sf.info(paths[0]).duration
        dur_b = sf.info(paths[1]).duration
        assert dur_a != pytest.approx(dur_b, rel=1e-3)


def test_loudness_normalizer():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [
            _record_with_audio(tmpdir, 0, {"leaf": "noise"}, {"D1": "passed"}, sr=44100, duration=3.0),
            _record_with_audio(tmpdir, 1, {"leaf": "noise"}, {"D1": "passed"}, sr=44100, duration=1.0),  # too short
        ]
        write_manifest(iter(recs), manifest)

        proc = LoudnessNormalizerProcessor({
            "target_rms": 0.1,
            "max_peak": 0.95,
            "min_duration": 2.0,
            "max_duration": 30.0,
            "target_sr": 44100,
        })
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "d2"))
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 2
        assert out_recs[0].is_stage_passed("D2")
        assert out_recs[1].rejected_reason == "d2_duration_out_of_range"


def test_apollo_superres_checkpoint_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [_record_with_audio(tmpdir, i, {"leaf": "noise"}, {"D2": "passed"}, sr=44100) for i in range(4)]
        write_manifest(iter(recs), manifest)

        out_dir = os.path.join(tmpdir, "d1")
        proc = ApolloSuperResProcessor({"mode": "dry_run", "target_sr": 44100})
        out_manifest1 = proc.process(manifest, out_dir)
        out_recs1 = list(read_manifest(out_manifest1))
        assert len(out_recs1) == 4

        # 第二次运行应通过 checkpoint 跳过已处理记录，且从上次输出恢复完整状态
        out_manifest2 = proc.process(manifest, out_dir)
        out_recs2 = list(read_manifest(out_manifest2))
        assert len(out_recs2) == 4
        # resume 后记录必须保留 D1 标记与增强后路径，不能退回“未处理”状态
        assert all(r.is_stage_passed("D1") for r in out_recs2)
        assert all(os.path.isfile(r.audio["path"]) for r in out_recs2)


def test_apollo_superres_post_normalize():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [_record_with_audio(tmpdir, i, {"leaf": "noise"}, {"D2": "passed"}, sr=44100, duration=3.0) for i in range(2)]
        write_manifest(iter(recs), manifest)

        proc = ApolloSuperResProcessor({
            "mode": "dry_run",
            "target_sr": 44100,
            "post_normalize": {"target_rms": 0.1, "max_peak": 0.95, "min_duration": 2.0, "max_duration": 30.0},
        })
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "d1_d2"))
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 2
        assert all(r.is_stage_passed("D1") for r in out_recs)
        assert all(r.is_stage_passed("D2") for r in out_recs)
        assert all(os.path.isfile(r.audio["path"]) for r in out_recs)


def test_stage_d_orchestrator_dry_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [
            _record_with_audio(tmpdir, 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0),
            _record_with_audio(tmpdir, 1, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=1.0),
        ]
        write_manifest(iter(recs), manifest)

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_d={
                "d1": {"mode": "dry_run", "target_sr": 44100},
                "d2": {"target_rms": 0.1, "max_peak": 0.95, "min_duration": 2.0, "max_duration": 30.0, "target_sr": 44100},
            },
        )
        orch = StageDOrchestrator(config)
        out_manifest = orch.run(
            input_manifest=manifest,
            output_dir=os.path.join(tmpdir, "out", "stage_d"),
            output_manifest=os.path.join(tmpdir, "out", "stage_d", "d_manifest.jsonl"),
        )
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 2
        passed = [r for r in out_recs if r.is_stage_passed("D2")]
        assert len(passed) == 1


@pytest.mark.parametrize("skip_d1", [False, True], ids=["with-d1", "skip-d1"])
def test_stage_d_orchestrator_rejects_empty_manifest_before_output(skip_d1, tmp_path):
    manifest = tmp_path / "empty.jsonl"
    write_manifest(iter([]), str(manifest))
    output_dir = tmp_path / "out" / "stage_d"
    config = UnifiedPipelineConfig(
        output_root=str(tmp_path / "out"),
        stage_d={
            "skip_d1": skip_d1,
            "d1": {"mode": "dry_run"},
            "d2": {},
        },
    )

    with pytest.raises(ValueError, match="至少一条输入记录"):
        StageDOrchestrator(config).run(
            input_manifest=str(manifest),
            output_dir=str(output_dir),
            output_manifest=str(output_dir / "d_manifest.jsonl"),
        )
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "execution_mode",
    [
        {"c1": "dry_run", "c2": "subprocess_daemon"},
        {"c1": "production", "c2": "dry_run"},
        {"c1": "production", "c2": "disabled"},
    ],
)
def test_stage_d_orchestrator_rejects_non_publishable_stage_c_before_output(execution_mode):
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        rec = _record_with_audio(
            tmpdir, 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0
        )
        rec.mark_stage(
            "C3",
            "passed",
            extra={"execution_mode": execution_mode, "publishable": False},
        )
        write_manifest(iter([rec]), manifest)

        output_dir = os.path.join(tmpdir, "out", "stage_d")
        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_d={"skip_d1": True, "d2": {"target_sr": 44100}},
        )

        with pytest.raises(ValueError, match="Stage C execution is not publishable"):
            StageDOrchestrator(config).run(
                input_manifest=manifest,
                output_dir=output_dir,
                output_manifest=os.path.join(output_dir, "d_manifest.jsonl"),
            )
        assert not os.path.exists(output_dir)


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("artifact", "current policy.sha256"),
        ("evidence", "label_space_sha256 does not match admission policy"),
    ],
)
def test_stage_d_orchestrator_rejects_tampered_contract_identity_before_output(
    mutation, expected
):
    """C2/C3 契约身份被篡改时，Stage D 必须在创建输出前拒绝。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        rec = _record_with_audio(
            tmpdir, 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0
        )
        if mutation == "artifact":
            rec.stages["C3"]["artifact_fingerprints"]["voice_policy_sha256"] = "f" * 64
        else:
            rec.stages["C2"]["label_space_sha256"] = "f" * 64
        write_manifest(iter([rec]), manifest)

        output_dir = os.path.join(tmpdir, "out", "stage_d")
        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_d={"skip_d1": True, "d2": {"target_sr": 44100}},
        )
        with pytest.raises(ValueError, match=expected):
            StageDOrchestrator(config).run(
                input_manifest=manifest,
                output_dir=output_dir,
                output_manifest=os.path.join(output_dir, "d_manifest.jsonl"),
            )
        assert not os.path.exists(output_dir)


def test_stage_d_orchestrator_rejects_speech_mid_in_top5_before_output(tmp_path):
    """top5 任一 speech MID 都必须使共享复验否决既有 passed 决策。"""
    from core.data.noise_admission_policy import load_admission_policy

    manifest = tmp_path / "input.jsonl"
    rec = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0
    )
    policy = load_admission_policy()
    speech_mid = "/m/01h8n0"
    rec.labels["audioset_top5"][0] = {
        "index": policy.mid_to_index[speech_mid],
        "mid": speech_mid,
        "label": policy.label_of(speech_mid),
        "prob": 1.0,
    }
    write_manifest(iter([rec]), str(manifest))
    output_dir = tmp_path / "out" / "stage_d"
    config = UnifiedPipelineConfig(
        output_root=str(tmp_path / "out"),
        stage_d={"skip_d1": True, "d2": {"target_sr": 44100}},
    )

    with pytest.raises(ValueError, match="does not match recomputed value"):
        StageDOrchestrator(config).run(
            input_manifest=str(manifest),
            output_dir=str(output_dir),
            output_manifest=str(output_dir / "d_manifest.jsonl"),
        )
    assert not output_dir.exists()


@pytest.mark.parametrize("legacy_location", ["labels", "c3"])
def test_stage_d_orchestrator_rejects_legacy_admission_fields_before_output(
    tmp_path, legacy_location
):
    rec = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0
    )
    if legacy_location == "labels":
        rec.labels["audioset_scores"] = [0.0] * 527
    else:
        rec.stages["C3"]["admission_calibration_version"] = 2
    manifest = tmp_path / "input.jsonl"
    write_manifest(iter([rec]), str(manifest))
    output_dir = tmp_path / "out" / "stage_d"
    config = UnifiedPipelineConfig(
        output_root=str(tmp_path / "out"),
        stage_d={"skip_d1": True, "d2": {"target_sr": 44100}},
    )

    with pytest.raises(ValueError, match="legacy admission fields are forbidden"):
        StageDOrchestrator(config).run(
            input_manifest=str(manifest),
            output_dir=str(output_dir),
            output_manifest=str(output_dir / "d_manifest.jsonl"),
        )
    assert not output_dir.exists()


@pytest.mark.parametrize("legacy_c4", [False, True], ids=["no-c3", "c4-only"])
def test_stage_d_orchestrator_requires_new_c3_before_output(legacy_c4):
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        rec = _record_with_audio(tmpdir, 0, {"leaf": "noise"}, sr=44100, duration=3.0)
        if legacy_c4:
            rec.mark_stage("C4", "passed")
        write_manifest(iter([rec]), manifest)

        output_dir = os.path.join(tmpdir, "out", "stage_d")
        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_d={"skip_d1": True, "d2": {"target_sr": 44100}},
        )

        match = "legacy stages" if legacy_c4 else "C3: stage is missing"
        with pytest.raises(ValueError, match=match):
            StageDOrchestrator(config).run(
                input_manifest=manifest,
                output_dir=output_dir,
                output_manifest=os.path.join(output_dir, "d_manifest.jsonl"),
            )
        assert not os.path.exists(output_dir)


def test_stage_d_orchestrator_requires_passed_c1_before_output(tmp_path):
    manifest = tmp_path / "input.jsonl"
    rec = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0
    )
    rec.mark_stage("C1", "rejected", extra={"intelligibility": "noise"})
    write_manifest(iter([rec]), str(manifest))
    output_dir = tmp_path / "out" / "stage_d"
    config = UnifiedPipelineConfig(
        output_root=str(tmp_path / "out"),
        stage_d={"skip_d1": True, "d2": {"target_sr": 44100}},
    )

    with pytest.raises(ValueError, match=r"record 'ds__src__0' C1"):
        StageDOrchestrator(config).run(
            input_manifest=str(manifest),
            output_dir=str(output_dir),
            output_manifest=str(output_dir / "d_manifest.jsonl"),
        )
    assert not output_dir.exists()


def test_stage_d_orchestrator_rejects_zero_c3_eligible(tmp_path):
    manifest = tmp_path / "input.jsonl"
    rec = _record_with_audio(str(tmp_path), 0, {}, sr=44100, duration=3.0)
    rec.mark_stage("C3", "rejected")
    write_manifest(iter([rec]), str(manifest))
    output_dir = tmp_path / "out" / "stage_d"
    config = UnifiedPipelineConfig(
        output_root=str(tmp_path / "out"),
        stage_d={"skip_d1": True, "d2": {}},
    )

    with pytest.raises(ValueError, match="没有 C3 passed eligible"):
        StageDOrchestrator(config).run(
            input_manifest=str(manifest),
            output_dir=str(output_dir),
            output_manifest=str(output_dir / "d_manifest.jsonl"),
        )
    assert not output_dir.exists()


@pytest.mark.parametrize("skip_d1", [False, True], ids=["d1", "direct-d2"])
def test_stage_d_consumes_private_audio_snapshot_across_source_aba(
    tmp_path, monkeypatch, skip_d1
):
    """源路径发生 A→B→A 时，D1/D2 仍只能消费预检时冻结的 A bytes。"""
    manifest = tmp_path / "input.jsonl"
    rec = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0
    )
    write_manifest(iter([rec]), str(manifest))
    audio = Path(rec.audio["path"])
    original_bytes = audio.read_bytes()
    backup = tmp_path / "audio_original.backup.wav"
    replacement = tmp_path / "audio_replacement.wav"
    _make_wav(str(replacement), duration=1.0, sr=44100)
    output_dir = tmp_path / "out" / "stage_d"
    orchestrator = StageDOrchestrator(
        UnifiedPipelineConfig(
            output_root=str(tmp_path / "out"),
            stage_d={
                "skip_d1": skip_d1,
                "d1": {"mode": "dry_run", "target_sr": 44100},
                "d2": {
                    "min_duration": 0.5,
                    "max_duration": 30.0,
                    "target_sr": 44100,
                    "num_workers": 1,
                },
            },
        )
    )
    real_run_locked = orchestrator._run_locked
    captured_snapshot_audio = []
    restored = False

    def swap_source_before_processing(snapshot_path, *args, **kwargs):
        with open(snapshot_path, "r", encoding="utf-8") as snapshot:
            payload = json.loads(next(snapshot))
        snapshot_audio = Path(payload["audio"]["path"])
        captured_snapshot_audio.append(
            (snapshot_audio, snapshot_audio.read_bytes(), Path(snapshot_path).parent)
        )
        audio.replace(backup)
        replacement.replace(audio)
        try:
            return real_run_locked(snapshot_path, *args, **kwargs)
        finally:
            if backup.exists():
                if audio.exists():
                    audio.replace(replacement)
                backup.replace(audio)

    def restore_source_before_publication(_source_manifest, _manifests, _audios):
        nonlocal restored
        audio.replace(replacement)
        backup.replace(audio)
        restored = True
        # 模拟 A→B→A 后 stat identity 也发生 ABA；本测试锁定实际消费 bytes，
        # source identity fail-closed 由相邻回归独立覆盖。

    monkeypatch.setattr(orchestrator, "_run_locked", swap_source_before_processing)
    monkeypatch.setattr(
        orchestrator, "_assert_source_snapshots", restore_source_before_publication
    )
    result = orchestrator.run(
        input_manifest=str(manifest),
        output_dir=str(output_dir),
        output_manifest=str(output_dir / "d_manifest.jsonl"),
    )

    output = next(rec for rec in read_manifest(result) if rec.is_stage_passed("D2"))
    assert sf.info(output.audio["path"]).duration == pytest.approx(3.0, rel=1e-3)
    assert restored
    assert len(captured_snapshot_audio) == 1
    snapshot_audio, snapshot_bytes, snapshot_root = captured_snapshot_audio[0]
    assert snapshot_audio != audio
    snapshot_audio.relative_to(snapshot_root)
    assert snapshot_bytes == original_bytes
    assert audio.read_bytes() == original_bytes
    assert not snapshot_root.exists()


@pytest.mark.parametrize("skip_d1", [False, True], ids=["d1", "direct-d2"])
def test_stage_d_removes_snapshot_references_from_terminal_artifacts(
    tmp_path, skip_d1
):
    """passed/rejected/skipped 终态及完成凭据均不得引用已回收快照。"""
    manifest = tmp_path / "input.jsonl"
    passed = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0
    )
    rejected = _record_with_audio(
        str(tmp_path), 1, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=1.0
    )
    skipped = _record_with_audio(
        str(tmp_path), 2, {}, {"C3": "rejected"}, sr=44100, duration=3.0
    )
    records = [passed, rejected, skipped]
    source_paths = {rec.record_id: rec.audio["path"] for rec in records}
    write_manifest(iter(records), str(manifest))
    output_dir = tmp_path / "out" / "stage_d"
    config = UnifiedPipelineConfig(
        output_root=str(tmp_path / "out"),
        stage_d={
            "skip_d1": skip_d1,
            "d1": {
                "mode": "dry_run",
                "target_sr": 44100,
                "cleanup_intermediate": False,
            },
            "d2": {
                "min_duration": 2.0,
                "max_duration": 30.0,
                "target_sr": 44100,
                "num_workers": 1,
                "output_layout": "sr_dataset",
                "cleanup_intermediate": False,
            },
        },
    )

    result = StageDOrchestrator(config).run(
        input_manifest=str(manifest),
        output_dir=str(output_dir),
        output_manifest=str(output_dir / "d_manifest.jsonl"),
    )

    terminal = {rec.record_id: rec for rec in read_manifest(result)}
    assert terminal[passed.record_id].is_stage_passed("D2")
    assert terminal[rejected.record_id].stages["D2"]["status"] == "rejected"
    assert terminal[skipped.record_id].stages["D2"]["status"] == "skipped"
    assert terminal[skipped.record_id].audio["path"] == source_paths[skipped.record_id]
    if not skip_d1:
        assert terminal[skipped.record_id].stages["D1"]["status"] == "skipped"
    else:
        assert terminal[rejected.record_id].audio["path"] == source_paths[rejected.record_id]
        assert (
            terminal[rejected.record_id].stages["D2"]["input_path"]
            == source_paths[rejected.record_id]
        )

    assert not (output_dir / ".stage_d_validated_snapshot").exists()
    private_tokens = (
        ".stage_d_validated_snapshot",
        ".stage_d_validated_staging_",
        "_stage_d_snapshot_source",
    )
    artifact_paths = {
        *output_dir.rglob("*.json"),
        *output_dir.rglob("*.jsonl"),
        *output_dir.rglob("*.complete"),
    }
    assert artifact_paths
    for path in artifact_paths:
        content = path.read_text(encoding="utf-8")
        assert not any(token in content for token in private_tokens), path


def test_stage_d_failure_retires_snapshot_and_sanitizes_resume_artifacts(
    tmp_path, monkeypatch
):
    """可重试失败保留现场时，JSONL 也不得继续引用已回收私有快照。"""
    manifest = tmp_path / "input.jsonl"
    source = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0
    )
    write_manifest(iter([source]), str(manifest))
    output_dir = tmp_path / "out" / "stage_d"
    config = UnifiedPipelineConfig(
        output_root=str(tmp_path / "out"),
        stage_d={
            "skip_d1": True,
            "d2": {
                "min_duration": 0.5,
                "max_duration": 30.0,
                "target_sr": 44100,
                "num_workers": 1,
            },
        },
    )
    real_normalize = LoudnessNormalizerProcessor._normalize
    calls = 0

    def fail_once(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected retryable failure")
        return real_normalize(self, *args, **kwargs)

    monkeypatch.setattr(LoudnessNormalizerProcessor, "_normalize", fail_once)
    with pytest.raises(RuntimeError, match="D2 存在可重试处理失败"):
        StageDOrchestrator(config).run(
            input_manifest=str(manifest),
            output_dir=str(output_dir),
            output_manifest=str(output_dir / "d_manifest.jsonl"),
        )

    private_tokens = (
        ".stage_d_validated_snapshot",
        ".stage_d_validated_staging_",
        "_stage_d_snapshot_source",
    )
    assert not (output_dir / ".stage_d_validated_snapshot").exists()
    assert not list(output_dir.parent.glob(".stage_d_validated_staging_*"))
    failed_artifacts = {
        *output_dir.rglob("*.json"),
        *output_dir.rglob("*.jsonl"),
        *output_dir.rglob("*.complete"),
    }
    assert failed_artifacts
    for path in failed_artifacts:
        content = path.read_text(encoding="utf-8")
        assert not any(token in content for token in private_tokens), path

    result = StageDOrchestrator(config).run(
        input_manifest=str(manifest),
        output_dir=str(output_dir),
        output_manifest=str(output_dir / "d_manifest.jsonl"),
    )
    output = next(iter(read_manifest(result)))
    assert output.is_stage_passed("D2")
    assert output.stages["D2"]["input_path"] == source.audio["path"]
    assert calls == 2
    assert not (output_dir / ".stage_d_validated_snapshot").exists()


@pytest.mark.parametrize("output_layout", ["flat", "dataset_label"])
def test_stage_d_checkpoint_resume_uses_restored_source_identity(
    tmp_path, monkeypatch, output_layout
):
    """去引用后的 D2 checkpoint 应稳定恢复，且不得重跑归一化。"""
    manifest = tmp_path / "input.jsonl"
    source = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0
    )
    write_manifest(iter([source]), str(manifest))
    output_dir = tmp_path / "out" / "stage_d"
    config = UnifiedPipelineConfig(
        output_root=str(tmp_path / "out"),
        stage_d={
            "skip_d1": True,
            "d2": {
                "min_duration": 0.5,
                "max_duration": 30.0,
                "target_sr": 44100,
                "num_workers": 1,
                "output_layout": output_layout,
            },
        },
    )

    first_result = StageDOrchestrator(config).run(
        input_manifest=str(manifest),
        output_dir=str(output_dir),
        output_manifest=str(output_dir / "d_manifest.jsonl"),
    )
    first = next(iter(read_manifest(first_result)))
    first_output_path = first.audio["path"]
    assert first.is_stage_passed("D2")
    assert first.stages["D2"]["input_path"] == source.audio["path"]
    assert source.record_id in _read_ckpt_done(str(output_dir / "D2"))
    assert not (output_dir / ".stage_d_validated_snapshot").exists()

    (output_dir / "D2" / "d2_output.complete").unlink()

    def reject_reprocessing(*_args, **_kwargs):
        raise AssertionError("checkpoint 命中时不得重新执行 D2 normalize")

    monkeypatch.setattr(LoudnessNormalizerProcessor, "_normalize", reject_reprocessing)
    second_result = StageDOrchestrator(config).run(
        input_manifest=str(manifest),
        output_dir=str(output_dir),
        output_manifest=str(output_dir / "d_manifest.jsonl"),
    )
    second = next(iter(read_manifest(second_result)))

    assert second.is_stage_passed("D2")
    assert second.audio["path"] == first_output_path
    assert second.stages["D2"]["input_path"] == source.audio["path"]
    assert source.record_id in _read_ckpt_done(str(output_dir / "D2"))
    assert not (output_dir / ".stage_d_validated_snapshot").exists()
    private_tokens = (
        ".stage_d_validated_snapshot",
        ".stage_d_validated_staging_",
        "_stage_d_snapshot_source",
    )
    artifact_paths = {
        *output_dir.rglob("*.json"),
        *output_dir.rglob("*.jsonl"),
        *output_dir.rglob("*.complete"),
    }
    for path in artifact_paths:
        content = path.read_text(encoding="utf-8")
        assert not any(token in content for token in private_tokens), path


def test_stage_d_rejects_source_manifest_change_before_publication(tmp_path, monkeypatch):
    manifest = tmp_path / "input.jsonl"
    original = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0
    )
    write_manifest(iter([original]), str(manifest))
    output_dir = tmp_path / "out" / "stage_d"
    output_dir.mkdir(parents=True)
    completion = output_dir / StageDOrchestrator.COMPLETION_FACT_NAME
    completion.write_text("stale", encoding="utf-8")
    orchestrator = StageDOrchestrator(
        UnifiedPipelineConfig(
            output_root=str(tmp_path / "out"),
            stage_d={"skip_d1": True, "d2": {}},
        )
    )

    def replace_before_publication(snapshot_path, *_args):
        replacement = manifest.with_suffix(".replacement")
        replacement.write_bytes(manifest.read_bytes())
        replacement.replace(manifest)
        source_manifest, manifests, audios = _args[3:6]
        orchestrator._assert_source_snapshots(source_manifest, manifests, audios)
        return str(output_dir)

    monkeypatch.setattr(orchestrator, "_run_locked", replace_before_publication)
    with pytest.raises(ValueError, match="manifest identity changed before publication"):
        orchestrator.run(
            input_manifest=str(manifest),
            output_dir=str(output_dir),
            output_manifest=str(output_dir / "d_manifest.jsonl"),
        )
    assert not completion.exists()


def test_stage_d_rejects_audio_change_before_publication(tmp_path, monkeypatch):
    manifest = tmp_path / "input.jsonl"
    rec = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0
    )
    write_manifest(iter([rec]), str(manifest))
    audio = Path(rec.audio["path"])
    output_dir = tmp_path / "out" / "stage_d"
    output_dir.mkdir(parents=True)
    completion = output_dir / StageDOrchestrator.COMPLETION_FACT_NAME
    completion.write_text("stale", encoding="utf-8")
    orchestrator = StageDOrchestrator(
        UnifiedPipelineConfig(
            output_root=str(tmp_path / "out"),
            stage_d={"skip_d1": True, "d2": {}},
        )
    )

    def replace_before_publication(snapshot_path, *_args):
        replacement = audio.with_suffix(".replacement.wav")
        _make_wav(str(replacement), duration=1.0, sr=8000)
        replacement.replace(audio)
        source_manifest, manifests, audios = _args[3:6]
        orchestrator._assert_source_snapshots(source_manifest, manifests, audios)
        return str(output_dir)

    monkeypatch.setattr(orchestrator, "_run_locked", replace_before_publication)
    with pytest.raises(ValueError, match="audio identity changed before publication"):
        orchestrator.run(
            input_manifest=str(manifest),
            output_dir=str(output_dir),
            output_manifest=str(output_dir / "d_manifest.jsonl"),
        )
    assert not completion.exists()


def test_stage_d_rejects_audio_path_replacement_after_fd_open(tmp_path, monkeypatch):
    manifest = tmp_path / "input.jsonl"
    rec = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0
    )
    write_manifest(iter([rec]), str(manifest))
    audio = Path(rec.audio["path"])
    replacement = audio.with_suffix(".replacement.wav")
    _make_wav(str(replacement), duration=1.0, sr=8000)
    orchestrator = StageDOrchestrator(
        UnifiedPipelineConfig(output_root=str(tmp_path / "out"), stage_d={"skip_d1": True})
    )
    real_open = os.open
    replaced = False

    def replace_after_open(path, flags, *args, **kwargs):
        nonlocal replaced
        descriptor = real_open(path, flags, *args, **kwargs)
        if not replaced and Path(path) == audio:
            replaced = True
            replacement.replace(audio)
        return descriptor

    monkeypatch.setattr(os, "open", replace_after_open)
    with pytest.raises(ValueError, match="file changed while reading stable snapshot"):
        orchestrator.run(
            input_manifest=str(manifest),
            output_dir=str(tmp_path / "out" / "stage_d"),
            output_manifest=str(tmp_path / "out" / "stage_d" / "d_manifest.jsonl"),
        )
    assert replaced


@pytest.mark.unit
def test_stage_d_stable_snapshot_rejects_fifo_without_blocking(tmp_path):
    fifo = tmp_path / "source.fifo"
    os.mkfifo(fifo)
    outcomes = []

    def inspect_fifo():
        try:
            StageDOrchestrator._stable_file_snapshot(fifo, include_bytes=False)
        except BaseException as exc:
            outcomes.append(exc)

    worker = threading.Thread(target=inspect_fifo, daemon=True)
    worker.start()
    worker.join(timeout=1)

    assert not worker.is_alive(), "FIFO snapshot inspection blocked in os.open"
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], ValueError)
    assert "snapshot source must be a regular file" in str(outcomes[0])


@pytest.mark.unit
@pytest.mark.parametrize(
    "unsupported_errno",
    sorted(StageDOrchestrator._REFLINK_UNSUPPORTED_ERRNOS),
)
def test_stage_d_snapshot_reflink_errno_falls_back(
    tmp_path, monkeypatch, unsupported_errno
):
    source = tmp_path / "source.wav"
    target = tmp_path / "snapshot.wav"
    payload = b"validated-snapshot" * 4096
    source.write_bytes(payload)
    methods = []

    def unsupported_clone(*_args):
        raise OSError(unsupported_errno, "reflink unsupported")

    monkeypatch.setattr(fcntl, "ioctl", unsupported_clone)
    _, identity = StageDOrchestrator._stable_file_snapshot(
        source,
        include_bytes=False,
        copy_path=target,
        copy_result=lambda method, size: methods.append((method, size)),
    )

    target_stat = target.stat()
    assert methods == [("fallback", len(payload))]
    assert identity[0] == hashlib.sha256(payload).hexdigest()
    assert target.read_bytes() == payload
    assert target_stat.st_size == len(payload)
    assert target_stat.st_nlink == 1
    assert (source.stat().st_dev, source.stat().st_ino) != (
        target_stat.st_dev,
        target_stat.st_ino,
    )


@pytest.mark.unit
def test_stage_d_snapshot_reflink_success_verifies_target_sha(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    target = tmp_path / "snapshot.wav"
    payload = b"reflink-content" * 2048
    source.write_bytes(payload)
    methods = []

    def successful_clone(target_fd, _request, source_fd):
        os.lseek(source_fd, 0, os.SEEK_SET)
        while True:
            chunk = os.read(source_fd, 8192)
            if not chunk:
                break
            os.write(target_fd, chunk)
        os.lseek(source_fd, 0, os.SEEK_SET)

    monkeypatch.setattr(fcntl, "ioctl", successful_clone)
    _, identity = StageDOrchestrator._stable_file_snapshot(
        source,
        include_bytes=False,
        copy_path=target,
        copy_result=lambda method, size: methods.append((method, size)),
    )

    assert methods == [("reflink", len(payload))]
    assert identity[0] == hashlib.sha256(payload).hexdigest()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == identity[0]


@pytest.mark.unit
def test_stage_d_snapshot_unexpected_reflink_errno_propagates(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    source.write_bytes(b"payload")

    def denied_clone(*_args):
        raise OSError(errno.EPERM, "reflink denied")

    monkeypatch.setattr(fcntl, "ioctl", denied_clone)
    with pytest.raises(ValueError, match="file changed while reading") as exc_info:
        StageDOrchestrator._stable_file_snapshot(
            source,
            include_bytes=False,
            copy_path=tmp_path / "snapshot.wav",
        )
    assert isinstance(exc_info.value.__cause__, OSError)
    assert exc_info.value.__cause__.errno == errno.EPERM


@pytest.mark.unit
@pytest.mark.parametrize("mutation", ["size", "path_identity", "nlink"])
def test_stage_d_snapshot_rejects_unsafe_target_identity(
    tmp_path, monkeypatch, mutation
):
    source = tmp_path / "source.wav"
    target = tmp_path / "snapshot.wav"
    replacement = tmp_path / "replacement.wav"
    hardlink = tmp_path / "snapshot.link"
    payload = b"identity-check" * 1024
    source.write_bytes(payload)
    replacement.write_bytes(payload)

    def unsafe_clone(target_fd, _request, source_fd):
        if mutation != "size":
            os.lseek(source_fd, 0, os.SEEK_SET)
            os.write(target_fd, os.read(source_fd, len(payload)))
            os.lseek(source_fd, 0, os.SEEK_SET)
        if mutation == "path_identity":
            os.replace(replacement, target)
        elif mutation == "nlink":
            os.link(target, hardlink)

    monkeypatch.setattr(fcntl, "ioctl", unsafe_clone)
    with pytest.raises(ValueError, match="invalid snapshot copy identity"):
        StageDOrchestrator._stable_file_snapshot(
            source, include_bytes=False, copy_path=target
        )


@pytest.mark.unit
def test_stage_d_audio_copy_skips_fsync_but_manifest_and_sentinel_sync(
    tmp_path, monkeypatch
):
    record = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=8000
    )
    manifest = tmp_path / "input.jsonl"
    write_manifest(iter([record]), str(manifest))
    staging = tmp_path / "staging"
    staging.mkdir()
    fsynced_paths = []
    real_fsync = os.fsync

    monkeypatch.setattr(
        fcntl,
        "ioctl",
        lambda *_args: (_ for _ in ()).throw(
            OSError(errno.EOPNOTSUPP, "reflink unsupported")
        ),
    )

    def capture_fsync(descriptor):
        fsynced_paths.append(os.readlink(f"/proc/self/fd/{descriptor}"))
        real_fsync(descriptor)

    monkeypatch.setattr(stage_d_module.os, "fsync", capture_fsync)
    snapshot = staging / "stage_c_snapshot.jsonl"
    StageDOrchestrator(
        UnifiedPipelineConfig(stage_d={"skip_d1": True, "snapshot_num_workers": 1})
    )._build_validated_snapshot(str(manifest), str(snapshot), tmp_path / "published")
    StageDOrchestrator._write_snapshot_sentinel(staging)

    assert any(path.endswith("stage_c_snapshot.jsonl.tmp") for path in fsynced_paths)
    assert any(path.endswith(".stage_d_owned") for path in fsynced_paths)
    assert not any(f"{os.sep}audio{os.sep}" in path for path in fsynced_paths)


@pytest.mark.unit
def test_stage_d_parallel_snapshot_is_bounded_and_byte_order_stable(
    tmp_path, monkeypatch
):
    records = [
        _record_with_audio(
            str(tmp_path), i, {"leaf": "noise"}, {"C3": "passed"}, sr=8000
        )
        for i in range(6)
    ]
    manifest = tmp_path / "input.jsonl"
    write_manifest(iter(records), str(manifest))
    published = tmp_path / "published"
    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir()
    baseline = baseline_root / "stage_c_snapshot.jsonl"
    StageDOrchestrator(
        UnifiedPipelineConfig(stage_d={"skip_d1": True, "snapshot_num_workers": 1})
    )._build_validated_snapshot(str(manifest), str(baseline), published)

    real_snapshot = StageDOrchestrator._stable_file_snapshot
    real_executor = stage_d_module.ThreadPoolExecutor
    rendezvous = threading.Barrier(3)
    release = [threading.Event(), threading.Event()]
    second_completed = threading.Event()
    initial_window_submitted = threading.Event()
    lock = threading.Lock()
    active = 0
    max_active = 0
    submitted = 0
    unfinished = 0
    max_unfinished = 0
    completion_order = []

    class TrackingExecutor:
        def __init__(self, *args, **kwargs):
            self._executor = real_executor(*args, **kwargs)

        def __enter__(self):
            self._executor.__enter__()
            return self

        def __exit__(self, *args):
            return self._executor.__exit__(*args)

        def submit(self, function, *args, **kwargs):
            nonlocal submitted, unfinished, max_unfinished
            with lock:
                submitted += 1
                unfinished += 1
                max_unfinished = max(max_unfinished, unfinished)
                if submitted == 4:
                    initial_window_submitted.set()
            try:
                future = self._executor.submit(function, *args, **kwargs)
            except BaseException:
                with lock:
                    unfinished -= 1
                raise

            def mark_finished(_future):
                nonlocal unfinished
                with lock:
                    unfinished -= 1

            future.add_done_callback(mark_finished)
            return future

    def coordinated_snapshot(cls, path, **kwargs):
        nonlocal active, max_active
        if kwargs.get("copy_path") is None:
            return real_snapshot(path, **kwargs)
        index = int(Path(path).stem.rsplit("_", 1)[1])
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            result = real_snapshot(path, **kwargs)
            if index < 2:
                rendezvous.wait(timeout=5)
                assert release[index].wait(timeout=5)
            with lock:
                completion_order.append(index)
            if index == 1:
                second_completed.set()
            return result
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(
        StageDOrchestrator, "_stable_file_snapshot", classmethod(coordinated_snapshot)
    )
    monkeypatch.setattr(stage_d_module, "ThreadPoolExecutor", TrackingExecutor)
    concurrent_root = tmp_path / "concurrent"
    concurrent_root.mkdir()
    concurrent = concurrent_root / "stage_c_snapshot.jsonl"
    errors = []

    def build_snapshot():
        try:
            StageDOrchestrator(
                UnifiedPipelineConfig(
                    stage_d={"skip_d1": True, "snapshot_num_workers": 2}
                )
            )._build_validated_snapshot(str(manifest), str(concurrent), published)
        except BaseException as exc:
            errors.append(exc)

    build_thread = threading.Thread(target=build_snapshot)
    build_thread.start()
    rendezvous.wait(timeout=5)
    assert initial_window_submitted.wait(timeout=5)
    with lock:
        assert submitted == 4
        assert unfinished <= 4
    release[1].set()
    assert second_completed.wait(timeout=5)
    release[0].set()
    build_thread.join(timeout=10)

    assert not build_thread.is_alive()
    assert not errors
    assert max_active == 2
    assert max_unfinished <= 4
    assert submitted == len(records)
    assert completion_order.index(1) < completion_order.index(0)
    assert concurrent.read_bytes() == baseline.read_bytes()


@pytest.mark.unit
def test_stage_d_duplicate_source_path_uses_single_snapshot_copy(tmp_path, monkeypatch):
    records = [
        _record_with_audio(
            str(tmp_path), i, {"leaf": "noise"}, {"C3": "passed"}, sr=8000
        )
        for i in range(2)
    ]
    records[1].audio = dict(records[0].audio)
    manifest = tmp_path / "input.jsonl"
    write_manifest(iter(records), str(manifest))
    real_snapshot = StageDOrchestrator._stable_file_snapshot
    copied_sources = []

    def count_copies(cls, path, **kwargs):
        if kwargs.get("copy_path") is not None:
            copied_sources.append(Path(path))
        return real_snapshot(path, **kwargs)

    monkeypatch.setattr(
        StageDOrchestrator, "_stable_file_snapshot", classmethod(count_copies)
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    snapshot = staging / "stage_c_snapshot.jsonl"
    _, audios, _, _ = StageDOrchestrator(
        UnifiedPipelineConfig(stage_d={"skip_d1": True, "snapshot_num_workers": 2})
    )._build_validated_snapshot(str(manifest), str(snapshot), tmp_path / "published")
    payloads = [json.loads(line) for line in snapshot.read_text().splitlines()]

    assert copied_sources == [Path(records[0].audio["path"])]
    assert len(audios) == 1
    assert payloads[0]["audio"]["path"] == payloads[1]["audio"]["path"]


@pytest.mark.unit
@pytest.mark.parametrize("workers", [True, 0, -1, 1.5, "2", None, 129])
def test_stage_d_snapshot_num_workers_requires_bounded_positive_integer(workers):
    with pytest.raises(ValueError, match=r"stage_d\.snapshot_num_workers"):
        UnifiedPipelineConfig.from_dict(
            {"stage_d": {"snapshot_num_workers": workers}}, active_stages=()
        )


@pytest.mark.unit
def test_stage_d_worker_defaults_are_runtime_only_and_decoupled(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    config = UnifiedPipelineConfig.from_dict(
        {"stage_d": {"skip_d1": True}}, active_stages=()
    )
    assert "snapshot_num_workers" not in config.stage_d
    assert "num_workers" not in config.stage_d
    orchestrator = StageDOrchestrator(config)
    assert orchestrator._snapshot_num_workers() == 4

    captured = []

    class StopAfterCapture(Exception):
        pass

    def capture_processor(*, processor, **_kwargs):
        captured.append(processor.num_workers)
        raise StopAfterCapture

    monkeypatch.setattr(orchestrator, "_run_snapshot_substage", capture_processor)
    with pytest.raises(StopAfterCapture):
        orchestrator._run_locked(
            "snapshot.jsonl",
            "output",
            "unused.jsonl",
            snapshot_sources={},
            record_sources={},
        )
    assert captured == [8]

    legacy_fallback = StageDOrchestrator(
        UnifiedPipelineConfig(
            stage_d={
                "skip_d1": True,
                "num_workers": 3,
                "snapshot_num_workers": 7,
            }
        )
    )
    captured.clear()
    monkeypatch.setattr(legacy_fallback, "_run_snapshot_substage", capture_processor)
    with pytest.raises(StopAfterCapture):
        legacy_fallback._run_locked(
            "snapshot.jsonl",
            "output",
            "unused.jsonl",
            snapshot_sources={},
            record_sources={},
        )
    assert legacy_fallback._snapshot_num_workers() == 7
    assert captured == [3]

    explicit_d2 = StageDOrchestrator(
        UnifiedPipelineConfig(
            stage_d={
                "skip_d1": True,
                "num_workers": 3,
                "snapshot_num_workers": 7,
                "d2": {"num_workers": 5},
            }
        )
    )
    captured.clear()
    monkeypatch.setattr(explicit_d2, "_run_snapshot_substage", capture_processor)
    with pytest.raises(StopAfterCapture):
        explicit_d2._run_locked(
            "snapshot.jsonl",
            "output",
            "unused.jsonl",
            snapshot_sources={},
            record_sources={},
        )
    assert explicit_d2._snapshot_num_workers() == 7
    assert captured == [5]


@pytest.mark.unit
def test_stage_d_snapshot_worker_failure_does_not_publish_completion(
    tmp_path, monkeypatch
):
    record = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=8000
    )
    manifest = tmp_path / "input.jsonl"
    write_manifest(iter([record]), str(manifest))
    output_dir = tmp_path / "out" / "stage_d"
    output_dir.mkdir(parents=True)
    completion = output_dir / StageDOrchestrator.COMPLETION_FACT_NAME
    completion.write_text("stale", encoding="utf-8")
    real_snapshot = StageDOrchestrator._stable_file_snapshot

    def fail_audio_copy(cls, path, **kwargs):
        if kwargs.get("copy_path") is not None:
            raise RuntimeError("injected snapshot worker failure")
        return real_snapshot(path, **kwargs)

    monkeypatch.setattr(
        StageDOrchestrator, "_stable_file_snapshot", classmethod(fail_audio_copy)
    )
    with pytest.raises(RuntimeError, match="snapshot worker failure"):
        StageDOrchestrator(
            UnifiedPipelineConfig(
                stage_d={"skip_d1": True, "snapshot_num_workers": 2}
            )
        ).run(
            str(manifest),
            str(output_dir),
            str(output_dir / "d_manifest.jsonl"),
        )

    assert not completion.exists()
    assert not (output_dir / ".stage_d_validated_snapshot").exists()
    assert not list(output_dir.parent.glob(".stage_d_validated_staging_*"))


@pytest.mark.unit
def test_stage_d_retire_failure_does_not_publish_completion(tmp_path, monkeypatch):
    record = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=8000
    )
    manifest = tmp_path / "input.jsonl"
    write_manifest(iter([record]), str(manifest))
    output_dir = tmp_path / "out" / "stage_d"
    completion = output_dir / StageDOrchestrator.COMPLETION_FACT_NAME
    orchestrator = StageDOrchestrator(
        UnifiedPipelineConfig(stage_d={"skip_d1": True, "snapshot_num_workers": 1})
    )
    fake_completion = {
        "schema": StageDOrchestrator.COMPLETION_FACT_SCHEMA,
        "version": StageDOrchestrator.COMPLETION_FACT_VERSION,
        "d2_passed": 1,
    }

    monkeypatch.setattr(
        orchestrator,
        "_run_locked",
        lambda *_args, **_kwargs: (str(output_dir), fake_completion),
    )
    monkeypatch.setattr(
        orchestrator,
        "_retire_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected retire failure")
        ),
    )
    with pytest.raises(RuntimeError, match="retire failure"):
        orchestrator.run(
            str(manifest),
            str(output_dir),
            str(output_dir / "d_manifest.jsonl"),
        )

    assert not completion.exists()


@pytest.mark.unit
def test_stage_d_temporary_cleanup_failure_does_not_publish_completion(
    tmp_path, monkeypatch
):
    record = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=8000
    )
    manifest = tmp_path / "input.jsonl"
    write_manifest(iter([record]), str(manifest))
    output_dir = tmp_path / "out" / "stage_d"
    completion_path = output_dir / StageDOrchestrator.COMPLETION_FACT_NAME
    orchestrator = StageDOrchestrator(
        UnifiedPipelineConfig(stage_d={"skip_d1": True, "snapshot_num_workers": 1})
    )
    fake_completion = {
        "schema": StageDOrchestrator.COMPLETION_FACT_SCHEMA,
        "version": StageDOrchestrator.COMPLETION_FACT_VERSION,
        "d2_passed": 1,
    }
    real_temporary_directory = tempfile.TemporaryDirectory

    class CleanupFailingTemporaryDirectory(real_temporary_directory):
        def cleanup(self):
            raise OSError("injected temporary cleanup failure")

    monkeypatch.setattr(
        stage_d_module.tempfile,
        "TemporaryDirectory",
        CleanupFailingTemporaryDirectory,
    )
    monkeypatch.setattr(
        orchestrator,
        "_run_locked",
        lambda *_args, **_kwargs: (str(output_dir), fake_completion),
    )

    with pytest.raises(OSError, match="temporary cleanup failure"):
        orchestrator.run(
            str(manifest),
            str(output_dir),
            str(output_dir / "d_manifest.jsonl"),
        )

    assert not completion_path.exists()


@pytest.mark.unit
def test_stage_d_snapshot_progress_bars_count_unique_sources(
    tmp_path, monkeypatch
):
    records = [
        _record_with_audio(
            str(tmp_path), i, {"leaf": "noise"}, {"C3": "passed"}, sr=8000
        )
        for i in range(3)
    ]
    records[2].audio = dict(records[0].audio)
    manifest = tmp_path / "input.jsonl"
    write_manifest(iter(records), str(manifest))
    progress_bars = []
    caller_thread = threading.get_ident()

    class CapturedProgress:
        def __init__(self, *, total, desc, unit):
            self.total = total
            self.desc = desc
            self.unit = unit
            self.n = 0
            self.update_threads = []
            progress_bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def update(self, amount=1):
            self.n += amount
            self.update_threads.append(threading.get_ident())

        def refresh(self):
            pass

    monkeypatch.setattr(stage_d_module, "tqdm", CapturedProgress)
    staging = tmp_path / "staging"
    staging.mkdir()
    snapshot = staging / "stage_c_snapshot.jsonl"
    orchestrator = StageDOrchestrator(
        UnifiedPipelineConfig(
            stage_d={"skip_d1": True, "snapshot_num_workers": 2}
        )
    )
    manifests, audios, _, _ = orchestrator._build_validated_snapshot(
        str(manifest), str(snapshot), tmp_path / "published"
    )
    orchestrator._assert_source_snapshots(str(manifest), manifests, audios)

    assert len(audios) == 2
    assert [(bar.desc, bar.unit, bar.total, bar.n) for bar in progress_bars] == [
        ("Stage D snapshot build", "unique-file", 2, 2),
        ("Stage D source revalidation", "unique-file", 2, 2),
    ]
    assert all(
        thread_id == caller_thread
        for bar in progress_bars
        for thread_id in bar.update_threads
    )


def test_stage_d_allows_empty_audio_path_for_ineligible_audit_record(tmp_path):
    """C3 skipped 的未落盘 chunk 可保留空路径作为审计记录。"""
    manifest = tmp_path / "input.jsonl"
    passed = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0
    )
    skipped = AudioRecord(
        record_id="ds__silent__chunk_0001",
        dataset_id="ds",
        source_id="silent",
        chunk_id="chunk_0001",
        audio={"path": "", "sr": 44100, "duration": 3.0, "channels": 1},
        labels={"original": "silent"},
        rejected_reason="b2_silent",
    )
    skipped.mark_stage(
        "C3", "skipped", extra={"reason": "prev_stage_C2_not_passed"}
    )
    write_manifest(iter([passed, skipped]), str(manifest))
    output_dir = tmp_path / "out" / "stage_d"

    result = StageDOrchestrator(
        UnifiedPipelineConfig(
            output_root=str(tmp_path / "out"),
            stage_d={
                "skip_d1": True,
                "d2": {"min_duration": 0.5, "max_duration": 30.0, "num_workers": 1},
            },
        )
    ).run(
        input_manifest=str(manifest),
        output_dir=str(output_dir),
        output_manifest=str(output_dir / "d_manifest.jsonl"),
    )

    records = {rec.record_id: rec for rec in read_manifest(result)}
    assert records[passed.record_id].is_stage_passed("D2")
    assert records[skipped.record_id].stages["D2"]["status"] == "skipped"
    assert records[skipped.record_id].audio["path"] == ""


def test_stage_d_rejects_empty_audio_path_for_c3_passed(tmp_path):
    """真正进入 D1/D2 的 C3 passed 记录仍必须有非空音频路径。"""
    manifest = tmp_path / "input.jsonl"
    rec = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0
    )
    rec.audio["path"] = ""
    write_manifest(iter([rec]), str(manifest))
    output_dir = tmp_path / "out" / "stage_d"

    with pytest.raises(ValueError, match="audio.path must be a non-empty string"):
        StageDOrchestrator(
            UnifiedPipelineConfig(
                output_root=str(tmp_path / "out"), stage_d={"skip_d1": True}
            )
        ).run(
            input_manifest=str(manifest),
            output_dir=str(output_dir),
            output_manifest=str(output_dir / "d_manifest.jsonl"),
        )


def test_stage_d_rejects_unknown_audiorecord_field(tmp_path):
    manifest = tmp_path / "input.jsonl"
    rec = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0
    )
    payload = rec.to_dict()
    payload["unknown_future_field"] = True
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    orchestrator = StageDOrchestrator(
        UnifiedPipelineConfig(output_root=str(tmp_path / "out"), stage_d={"skip_d1": True})
    )

    with pytest.raises(ValueError, match="invalid AudioRecord fields"):
        orchestrator.run(
            input_manifest=str(manifest),
            output_dir=str(tmp_path / "out" / "stage_d"),
            output_manifest=str(tmp_path / "out" / "stage_d" / "d_manifest.jsonl"),
        )


@pytest.mark.parametrize("dataset_id", ["../escape", "nested/name", "", ".", ".."])
def test_stage_d_rejects_unsafe_dataset_id(tmp_path, dataset_id):
    manifest = tmp_path / "input.jsonl"
    rec = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0
    )
    rec.dataset_id = dataset_id
    write_manifest(iter([rec]), str(manifest))
    output_dir = tmp_path / "out" / "stage_d"

    with pytest.raises(
        ValueError, match=r"非法 dataset_id|dataset_id must be a non-empty string"
    ):
        StageDOrchestrator(
            UnifiedPipelineConfig(
                output_root=str(tmp_path / "out"), stage_d={"skip_d1": True}
            )
        ).run(
            input_manifest=str(manifest),
            output_dir=str(output_dir),
            output_manifest=str(output_dir / "d_manifest.jsonl"),
        )
    assert not output_dir.exists()


def test_stage_d_writes_completion_fact_bound_to_manifests_and_stats(tmp_path):
    import hashlib

    from noise_filter_pipeline.core.manifest_io import config_hash

    manifest = tmp_path / "input.jsonl"
    rec = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0
    )
    write_manifest(iter([rec]), str(manifest))
    output_dir = tmp_path / "out" / "stage_d"
    orchestrator = StageDOrchestrator(
        UnifiedPipelineConfig(
            output_root=str(tmp_path / "out"),
            stage_d={"skip_d1": True, "d2": {"min_duration": 2.0}},
        )
    )

    orchestrator.run(
        input_manifest=str(manifest),
        output_dir=str(output_dir),
        output_manifest=str(output_dir / "d_manifest.jsonl"),
    )

    completion = json.loads((output_dir / "d_completion.json").read_text(encoding="utf-8"))
    assert set(completion) == {
        "schema",
        "version",
        "final_stage",
        "manifest_prefix",
        "stage_d_config_hash",
        "d2_passed",
        "manifest_files",
        "d_stats",
    }
    assert completion["schema"] == "noise_filter_stage_d_completion"
    assert completion["version"] == 1
    assert completion["final_stage"] == "D2"
    assert completion["manifest_prefix"] == "d2"
    assert completion["stage_d_config_hash"] == config_hash(orchestrator.stage_cfg)
    assert completion["d2_passed"] == 1
    d2_manifest = output_dir / "D2" / "d2_manifest_ds.jsonl"
    assert completion["manifest_files"] == [
        {
            "name": d2_manifest.name,
            "size": d2_manifest.stat().st_size,
            "sha256": hashlib.sha256(d2_manifest.read_bytes()).hexdigest(),
        }
    ]
    stats_path = output_dir / "d_stats.json"
    assert completion["d_stats"] == {
        "size": stats_path.stat().st_size,
        "sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
    }


def test_stage_d_zero_d2_passed_invalidates_completion_and_stats(tmp_path):
    manifest = tmp_path / "input.jsonl"
    rec = _record_with_audio(
        str(tmp_path), 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=1.0
    )
    write_manifest(iter([rec]), str(manifest))
    output_dir = tmp_path / "out" / "stage_d"
    output_dir.mkdir(parents=True)
    (output_dir / "d_stats.json").write_text("stale", encoding="utf-8")
    (output_dir / "d_completion.json").write_text("stale", encoding="utf-8")
    config = UnifiedPipelineConfig(
        output_root=str(tmp_path / "out"),
        stage_d={
            "skip_d1": True,
            "d2": {"min_duration": 2.0, "max_duration": 30.0},
        },
    )

    with pytest.raises(RuntimeError, match="没有 D2 passed"):
        StageDOrchestrator(config).run(
            input_manifest=str(manifest),
            output_dir=str(output_dir),
            output_manifest=str(output_dir / "d_manifest.jsonl"),
        )
    assert not (output_dir / "d_stats.json").exists()
    assert not (output_dir / "d_completion.json").exists()
    assert not (output_dir / "D2" / "d2_output.complete").exists()


@pytest.mark.parametrize("c2_status", [None, "failed", "rejected"], ids=["missing", "failed", "rejected"])
def test_stage_d_orchestrator_requires_passed_c2_before_output(c2_status):
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        rec = _record_with_audio(
            tmpdir, 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0
        )
        if c2_status is None:
            rec.stages.pop("C2")
        else:
            rec.mark_stage("C2", c2_status)
        write_manifest(iter([rec]), manifest)

        output_dir = os.path.join(tmpdir, "out", "stage_d")
        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_d={"skip_d1": True, "d2": {"target_sr": 44100}},
        )

        with pytest.raises(ValueError, match=r"record 'ds__src__0' C2"):
            StageDOrchestrator(config).run(
                input_manifest=manifest,
                output_dir=output_dir,
                output_manifest=os.path.join(output_dir, "d_manifest.jsonl"),
            )
        assert not os.path.exists(output_dir)


def test_apollo_superres_dry_run_output_sr_null():
    """output_sr 为 null 时，dry_run 仍应模拟 Apollo 输出到 target_sr。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [_record_with_audio(tmpdir, i, {"leaf": "noise"}, {"C3": "passed"}, sr=16000, duration=3.0) for i in range(2)]
        write_manifest(iter(recs), manifest)

        proc = ApolloSuperResProcessor({
            "mode": "dry_run",
            "target_sr": 44100,
            "output_sr": None,
            "output_format": "flac",
        })
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "d1"))
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 2
        for r in out_recs:
            assert r.is_stage_passed("D1")
            assert r.audio["sr"] == 44100
            info = sf.info(r.audio["path"])
            assert info.samplerate == 44100


def test_apollo_superres_skip_records_original_sr():
    """跳过 Apollo 的记录应在 extra 中保留正确的 original_sr。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [
            _record_with_audio(tmpdir, 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0),
            _record_with_audio(tmpdir, 1, {"leaf": "noise"}, {"C3": "passed"}, sr=48000, duration=3.0),
        ]
        write_manifest(iter(recs), manifest)

        proc = ApolloSuperResProcessor({"mode": "dry_run", "target_sr": 44100, "output_format": "flac"})
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "d1"))
        out_recs = list(read_manifest(out_manifest))
        extras = [r.stages.get("D1", {}) for r in out_recs]
        assert extras[0].get("original_sr") == 44100
        assert extras[1].get("original_sr") == 48000


def test_apollo_superres_outputs_flac():
    """D1 输出音频应为 FLAC 格式以节省空间。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [_record_with_audio(tmpdir, i, {"leaf": "noise"}, {"D2": "passed"}, sr=16000) for i in range(2)]
        write_manifest(iter(recs), manifest)

        proc = ApolloSuperResProcessor({"mode": "dry_run", "target_sr": 44100, "output_format": "flac"})
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "d1"))
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 2
        for r in out_recs:
            assert r.is_stage_passed("D1")
            assert r.audio["path"].endswith(".flac")
            assert os.path.isfile(r.audio["path"])
            info = sf.info(r.audio["path"])
            assert info.format == "FLAC"
            assert r.audio["sr"] == 44100


def test_apollo_superres_skip_high_sample_rate():
    """采样率 ≥ target_sr 的样本应跳过 Apollo，默认按 output_sr 输出统一采样率。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [
            _record_with_audio(tmpdir, 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0),
            _record_with_audio(tmpdir, 1, {"leaf": "noise"}, {"C3": "passed"}, sr=48000, duration=3.0),
        ]
        write_manifest(iter(recs), manifest)

        proc = ApolloSuperResProcessor({"mode": "dry_run", "target_sr": 44100, "output_format": "flac"})
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "d1"))
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 2
        for r in out_recs:
            assert r.is_stage_passed("D1")
            assert r.audio["path"].endswith(".flac")
            assert os.path.isfile(r.audio["path"])
            extra = r.stages.get("D1", {})
            assert extra.get("skipped") is True
        # 默认 output_sr == target_sr，跳过的音频也统一输出到 44.1kHz
        assert out_recs[0].audio["sr"] == 44100
        assert out_recs[1].audio["sr"] == 44100


def test_apollo_superres_skip_high_sample_rate_keep_original_sr():
    """output_sr 显式设为 null 时，跳过的样本应保留原始采样率。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [
            _record_with_audio(tmpdir, 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0),
            _record_with_audio(tmpdir, 1, {"leaf": "noise"}, {"C3": "passed"}, sr=48000, duration=3.0),
        ]
        write_manifest(iter(recs), manifest)

        proc = ApolloSuperResProcessor({
            "mode": "dry_run",
            "target_sr": 44100,
            "output_sr": None,
            "output_format": "flac",
        })
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "d1"))
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 2
        for r in out_recs:
            assert r.is_stage_passed("D1")
            extra = r.stages.get("D1", {})
            assert extra.get("skipped") is True
        assert out_recs[0].audio["sr"] == 44100
        assert out_recs[1].audio["sr"] == 48000


def test_loudness_normalizer_outputs_flac():
    """D2 最终输出音频应为 FLAC 格式。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [_record_with_audio(tmpdir, i, {"leaf": "noise"}, {"D1": "passed"}, sr=44100, duration=3.0) for i in range(2)]
        write_manifest(iter(recs), manifest)

        proc = LoudnessNormalizerProcessor({
            "target_rms": 0.1,
            "max_peak": 0.95,
            "min_duration": 2.0,
            "max_duration": 30.0,
            "output_format": "flac",
        })
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "d2"))
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 2
        for r in out_recs:
            assert r.is_stage_passed("D2")
            assert r.audio["path"].endswith(".flac")
            assert os.path.isfile(r.audio["path"])
            info = sf.info(r.audio["path"])
            assert info.format == "FLAC"


def test_loudness_normalizer_output_sr():
    """D2 应支持 output_sr，使用 soxr 重采样到目标采样率。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [_record_with_audio(tmpdir, i, {"leaf": "noise"}, {"D1": "passed"}, sr=16000, duration=3.0) for i in range(2)]
        write_manifest(iter(recs), manifest)

        proc = LoudnessNormalizerProcessor({
            "target_rms": 0.1,
            "max_peak": 0.95,
            "min_duration": 2.0,
            "max_duration": 30.0,
            "output_sr": 44100,
            "output_format": "flac",
        })
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "d2"))
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 2
        for r in out_recs:
            assert r.is_stage_passed("D2")
            assert r.audio["sr"] == 44100
            info = sf.info(r.audio["path"])
            assert info.samplerate == 44100


def test_stage_d_orchestrator_merged():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [
            _record_with_audio(tmpdir, 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0),
            _record_with_audio(tmpdir, 1, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=1.0),
        ]
        write_manifest(iter(recs), manifest)

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_d={
                "d1": {
                    "mode": "dry_run",
                    "target_sr": 44100,
                    "post_normalize": {"target_rms": 0.1, "max_peak": 0.95, "min_duration": 2.0, "max_duration": 30.0},
                },
                "d2": {"target_rms": 0.1, "max_peak": 0.95, "min_duration": 2.0, "max_duration": 30.0, "target_sr": 44100},
            },
        )
        orch = StageDOrchestrator(config)
        out_manifest = orch.run(
            input_manifest=manifest,
            output_dir=os.path.join(tmpdir, "out", "stage_d"),
            output_manifest=os.path.join(tmpdir, "out", "stage_d", "d_manifest.jsonl"),
        )
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 2
        passed = [r for r in out_recs if r.is_stage_passed("D2")]
        assert len(passed) == 1


def test_stage_d_orchestrator_merged_d2_output_sr():
    """D1+D2 合并模式下，D2 的 output_sr/output_format 应作为最终输出规格。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [
            _record_with_audio(tmpdir, 0, {"leaf": "noise"}, {"C3": "passed"}, sr=16000, duration=3.0),
        ]
        write_manifest(iter(recs), manifest)

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_d={
                "d1": {
                    "mode": "dry_run",
                    "target_sr": 44100,
                    "output_format": "flac",
                    "post_normalize": {"target_rms": 0.1, "max_peak": 0.95, "min_duration": 2.0, "max_duration": 30.0},
                },
                "d2": {"output_sr": 32000, "output_format": "wav"},
            },
        )
        orch = StageDOrchestrator(config)
        out_manifest = orch.run(
            input_manifest=manifest,
            output_dir=os.path.join(tmpdir, "out", "stage_d"),
            output_manifest=os.path.join(tmpdir, "out", "stage_d", "d_manifest.jsonl"),
        )
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 1
        rec = out_recs[0]
        assert rec.is_stage_passed("D1")
        assert rec.is_stage_passed("D2")
        # D2 output_sr 应覆盖 D1 output_sr 作为最终采样率
        assert rec.audio["sr"] == 32000
        # D2 output_format 应作为最终格式
        assert rec.audio["path"].endswith(".wav")
        info = sf.info(rec.audio["path"])
        assert info.samplerate == 32000


def test_stage_d_orchestrator_skip_d1():
    """skip_d1 为 true 时跳过 Apollo，直接用 D2 处理 Stage C 输出。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [
            _record_with_audio(tmpdir, 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0),
            _record_with_audio(tmpdir, 1, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=1.0),
        ]
        write_manifest(iter(recs), manifest)

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            stage_d={
                "skip_d1": True,
                "d2": {
                    "target_rms": 0.1,
                    "max_peak": 0.95,
                    "min_duration": 2.0,
                    "max_duration": 30.0,
                    "output_sr": 32000,
                    "output_format": "flac",
                },
            },
        )
        orch = StageDOrchestrator(config)
        out_manifest = orch.run(
            input_manifest=manifest,
            output_dir=os.path.join(tmpdir, "out", "stage_d"),
            output_manifest=os.path.join(tmpdir, "out", "stage_d", "d_manifest.jsonl"),
        )
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 2

        # D1 被跳过，不应出现 D1 stage 标记
        assert all("D1" not in r.stages for r in out_recs)

        passed = [r for r in out_recs if r.is_stage_passed("D2")]
        rejected = [r for r in out_recs if not r.is_stage_passed("D2")]
        assert len(passed) == 1
        assert len(rejected) == 1
        assert rejected[0].rejected_reason == "d2_duration_out_of_range"

        # 最终输出音频应由 D2 产生，采样率/格式与 d2 配置一致
        for rec in passed:
            assert rec.audio["sr"] == 32000
            assert rec.audio["path"].endswith(".flac")
            assert os.path.isfile(rec.audio["path"])
            info = sf.info(rec.audio["path"])
            assert info.samplerate == 32000
            assert info.format == "FLAC"
            # 输出目录应为 stage_d/D2/audio（sr_dataset 布局：audio/sr_xx/<dataset>/），而不是 D1
            assert f"{os.sep}D2{os.sep}audio{os.sep}" in rec.audio["path"]


def test_apollo_chunk_parameters_reject_invalid_overlap():
    """Apollo 分片必须拒绝非正 chunk 或 overlap >= chunk，不能静默退化。"""
    from noise_filter_pipeline.processors.apollo_chunks import _prepare_chunks

    audio = np.zeros(16000, dtype=np.float32)
    with pytest.raises(ValueError, match="chunk_sec"):
        _prepare_chunks(audio, 16000, chunk_sec=0, overlap_sec=0)
    with pytest.raises(ValueError, match="overlap_sec"):
        _prepare_chunks(audio, 16000, chunk_sec=1, overlap_sec=1)


def test_apollo_chunks_zero_overlap_no_fade_crash():
    """overlap_sec=0（合法配置）时窗函数不得崩溃。

    回归：fade=0 时 window[-0:] == window[0:] 是整段数组，乘以空数组抛
    ValueError，subprocess_daemon 模式下整批标 failed 无限重试。
    """
    from noise_filter_pipeline.processors.apollo_chunks import _get_window, _prepare_chunks

    # _prepare_chunks 接受 overlap_sec=0
    audio = np.zeros(16000, dtype=np.float32)
    chunks, _, _, _, fade, _ = _prepare_chunks(audio, 16000, chunk_sec=1, overlap_sec=0)
    assert fade == 0
    assert len(chunks) >= 1
    # 窗函数在 fade=0 时返回全 1，不抛异常
    window = _get_window(16000, 0)
    assert np.all(window == 1.0)


def test_apollo_superres_rejects_invalid_chunk_config():
    """D1 非法 chunk/overlap 配置必须在 __init__ fail-fast。"""
    from noise_filter_pipeline.processors.apollo_superres import ApolloSuperResProcessor

    with pytest.raises(ValueError, match="chunk_sec"):
        ApolloSuperResProcessor({"chunk_sec": 0, "overlap_sec": 0})
    with pytest.raises(ValueError, match="overlap_sec"):
        ApolloSuperResProcessor({"chunk_sec": 1, "overlap_sec": 1})
    with pytest.raises(ValueError, match="overlap_sec"):
        ApolloSuperResProcessor({"chunk_sec": 1, "overlap_sec": -0.5})
    # 合法配置（overlap_sec=0 允许）正常构造
    ApolloSuperResProcessor({"chunk_sec": 1, "overlap_sec": 0})


def test_make_unique_output_path_sanitizes_unsafe_stem():
    """源文件名含空格/括号/中文时，输出文件名应清洗为安全字符。"""
    from tools.shared.audio_io import make_unique_output_path

    out = make_unique_output_path("/tmp/out", "/opt/nonexistent/src/apostrophe(')hp chunk.wav", ext=".flac")
    name = os.path.basename(out)
    assert name.endswith(".flac")
    stem = name[: -len(".flac")]
    # 只允许 ASCII 字母数字、点、下划线、连字符
    assert all(c.isascii() and (c.isalnum() or c in "._-") for c in stem), name
    assert " " not in name and "(" not in name and "'" not in name

    # 中文 stem 不得透传
    out_zh = make_unique_output_path("/tmp/out", "/opt/nonexistent/src/中文 音频.wav", ext=".flac")
    name_zh = os.path.basename(out_zh)
    assert name_zh.isascii(), name_zh

    # 不同源路径即使清洗后 stem 趋同，hash 也能保证不碰撞
    a = make_unique_output_path("/tmp/out", "/opt/nonexistent/a/x(y).wav")
    b = make_unique_output_path("/tmp/out", "/opt/nonexistent/b/x y.wav")
    assert a != b

    # 同一源路径输出名确定（resume 友好）
    assert make_unique_output_path("/tmp/out", "/opt/nonexistent/src/中文 音频.wav", ext=".flac") == out_zh


def test_loudness_normalizer_rejects_nan():
    """含 NaN/Inf 的音频不得归一化写盘，应判 rejected。"""
    proc = LoudnessNormalizerProcessor({"target_rms": 0.1, "min_duration": 0.1, "max_duration": 30.0})
    audio = np.random.randn(16000).astype(np.float32) * 0.1
    audio[100] = np.nan
    _, _, info = proc._normalize(audio, 16000)
    assert info["rejected_reason"] == "nan_or_inf"

    audio_inf = np.random.randn(16000).astype(np.float32) * 0.1
    audio_inf[200] = np.inf
    _, _, info_inf = proc._normalize(audio_inf, 16000)
    assert info_inf["rejected_reason"] == "nan_or_inf"


def test_apollo_resume_probe_rejects_corrupt_audio():
    """resume 可读性探测：损坏/空文件不得被当作合法产物恢复。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        good = os.path.join(tmpdir, "good.wav")
        _make_wav(good)
        assert ApolloSuperResProcessor._is_readable_audio(good)

        corrupt = os.path.join(tmpdir, "corrupt.wav")
        with open(corrupt, "wb") as f:
            f.write(b"not an audio file")
        assert not ApolloSuperResProcessor._is_readable_audio(corrupt)

        empty = os.path.join(tmpdir, "empty.wav")
        open(empty, "wb").close()
        assert not ApolloSuperResProcessor._is_readable_audio(empty)

        assert not ApolloSuperResProcessor._is_readable_audio(os.path.join(tmpdir, "missing.wav"))
        assert not ApolloSuperResProcessor._is_readable_audio(None)


def test_d2_empty_input_produces_empty_manifest(tmp_path):
    """空输入 manifest：D2 产出存在的输出目录（下游 count/合并读空集不崩）。"""
    empty = os.path.join(str(tmp_path), "empty.jsonl")
    write_manifest(iter([]), empty)
    d2_dir = os.path.join(str(tmp_path), "d2")
    out = LoudnessNormalizerProcessor({}).process(empty, d2_dir)
    assert os.path.isdir(out)
    # 空输入不产出任何 per-dataset manifest 文件
    assert not [f for f in os.listdir(out) if f.endswith(".jsonl")]


def test_production_config_uses_dataset_label_output_layout():
    """正式配置固定使用 dataset/<category>/shard/ordinal 的输出布局。"""
    config_path = Path(__file__).resolve().parents[1] / "config" / "pipeline.yaml"
    config = UnifiedPipelineConfig.from_yaml(str(config_path), active_stages=())
    assert config.stage_d["snapshot_num_workers"] == 4
    assert "num_workers" not in config.stage_d
    assert config.stage_d["d2"]["num_workers"] == 32
    assert config.stage_d["d2"]["output_layout"] == "dataset_label"


def test_loudness_normalizer_output_layout_sr_dataset():
    """默认 sr_dataset 布局：输出音频落在 audio/sr_<rate>/<dataset_id>/ 子目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [
            _record_with_audio(tmpdir, 0, {"leaf": "noise"}, {"D1": "passed"}, sr=16000, duration=3.0),
            _record_with_audio(tmpdir, 1, {"leaf": "noise"}, {"D1": "passed"}, sr=44100, duration=3.0),
        ]
        write_manifest(iter(recs), manifest)

        proc = LoudnessNormalizerProcessor({})
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "d2"))
        out_recs = list(read_manifest(out_manifest))
        assert all(r.is_stage_passed("D2") for r in out_recs)
        audio_root = os.path.join(tmpdir, "d2", "audio")
        for r in out_recs:
            assert os.path.isfile(r.audio["path"])
            rel = os.path.relpath(r.audio["path"], audio_root)
            parts = rel.split(os.sep)
            assert len(parts) == 3, rel
            sr = r.audio["sr"]
            expected_sr_dir = f"sr_{sr // 1000}k" if sr % 1000 == 0 else f"sr_{sr}"
            assert parts[0] == expected_sr_dir, rel
            assert parts[1] == r.dataset_id, rel


def test_loudness_normalizer_output_layout_flat():
    """output_layout: flat 时保持旧的平铺行为（audio/ 直接是文件名）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [_record_with_audio(tmpdir, 0, {"leaf": "noise"}, {"D1": "passed"}, sr=16000, duration=3.0)]
        write_manifest(iter(recs), manifest)

        proc = LoudnessNormalizerProcessor({"output_layout": "flat"})
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "d2"))
        out_recs = list(read_manifest(out_manifest))
        rel = os.path.relpath(out_recs[0].audio["path"], os.path.join(tmpdir, "d2", "audio"))
        assert os.sep not in rel, rel


def test_loudness_normalizer_output_layout_sr_dataset_label():
    """sr_dataset_label 布局必须使用 C3 选定的 balance category。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [
            _record_with_audio(
                tmpdir,
                0,
                {"leaf": "noise", "audioset": "Rain"},
                {"C3": "passed"},
                sr=16000,
                duration=3.0,
            ),
            _record_with_audio(
                tmpdir,
                1,
                {"leaf": "noise", "audioset": "Wind"},
                {"C3": "passed"},
                sr=16000,
                duration=3.0,
            ),
        ]
        write_manifest(iter(recs), manifest)

        proc = LoudnessNormalizerProcessor({"output_layout": "sr_dataset_label"})
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "d2"))
        out_recs = list(read_manifest(out_manifest))
        audio_root = os.path.join(tmpdir, "d2", "audio")
        rel0 = os.path.relpath(out_recs[0].audio["path"], audio_root)
        rel1 = os.path.relpath(out_recs[1].audio["path"], audio_root)
        assert rel0.split(os.sep)[:3] == ["sr_16k", "ds", "Rain"], rel0
        assert rel1.split(os.sep)[:3] == ["sr_16k", "ds", "Wind"], rel1


def test_apollo_batch_failure_marks_failed_retryable():
    """D1 daemon 批失败：基础设施失败标 failed+retryable、不设 rejected_reason、不写 checkpoint。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [
            _record_with_audio(tmpdir, i, {"leaf": "noise"}, {"C3": "passed"}, sr=16000, duration=3.0)
            for i in range(2)
        ]
        write_manifest(iter(recs), manifest)

        ckpt = os.path.join(tmpdir, "apollo.pt")
        cfg = os.path.join(tmpdir, "apollo.yml")
        open(ckpt, "w").close()
        open(cfg, "w").close()

        proc = ApolloSuperResProcessor({
            "mode": "subprocess_daemon",
            "daemon_script_path": __file__,  # 仅需通过 isfile 检查，不会被真正启动
            "config_path": cfg,
            "checkpoint": ckpt,
            "target_sr": 44100,
            "batch_size": 1,
        })

        def _boom(*args, **kwargs):
            raise RuntimeError("cuda oom")

        proc._run_subprocess_daemon = _boom
        out_dir = os.path.join(tmpdir, "d1")
        out_manifest = proc.process(manifest, out_dir)
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 2
        for r in out_recs:
            d1 = r.stages.get("D1", {})
            assert d1.get("status") == "failed", d1
            assert d1.get("retryable") is True, d1
            assert not r.rejected_reason
        assert proc.has_retryable_failures()
        # 基础设施失败不写 checkpoint，resume 可重试
        done = _read_ckpt_done(out_dir)
        assert not {r.record_id for r in recs} & done


def test_apollo_skip_copy_failure_marks_failed_and_retried_on_rerun(monkeypatch):
    """_skip_apollo 复制/转码失败：标 failed 不写 checkpoint，重跑时应重新处理。"""
    import noise_filter_pipeline.processors.apollo_superres as apollo_mod

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [
            _record_with_audio(tmpdir, i, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0)
            for i in range(2)
        ]
        write_manifest(iter(recs), manifest)

        out_dir = os.path.join(tmpdir, "d1")
        proc = ApolloSuperResProcessor({"mode": "dry_run", "target_sr": 44100, "output_format": "flac"})

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        # 打桩只覆盖第一次跑：monkeypatch 默认作用域是整个测试函数，
        # 若不加 context 限制，下面的"重跑（不再故障）"仍会命中 OSError。
        with monkeypatch.context() as m:
            m.setattr(apollo_mod, "copy_or_convert_audio", _boom)
            out_manifest = proc.process(manifest, out_dir)

        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 2
        for r in out_recs:
            d1 = r.stages.get("D1", {})
            assert d1.get("status") == "failed", d1
            assert d1.get("retryable") is True, d1
            assert not r.rejected_reason
        assert proc.has_retryable_failures()
        done = _read_ckpt_done(out_dir, name="checkpoint.dry_run.json")
        assert not {r.record_id for r in recs} & done

        # 重跑（不再故障）：未写 checkpoint 的记录应被重新处理并通过
        proc2 = ApolloSuperResProcessor({"mode": "dry_run", "target_sr": 44100, "output_format": "flac"})
        out_manifest2 = proc2.process(manifest, out_dir)
        out_recs2 = list(read_manifest(out_manifest2))
        assert all(r.is_stage_passed("D1") for r in out_recs2)
        assert not proc2.has_retryable_failures()


def test_apollo_skip_content_error_marks_rejected_terminal(monkeypatch):
    """_skip_apollo 内容级失败（损坏音频等非 OSError）标 rejected 终态写 checkpoint。

    回归：此前所有异常统一标 failed+retryable——永久损坏文件不写 checkpoint，
    orchestrator 见可重试失败即中止 stage，pipeline 每轮跑不完。按 D2 的
    异常分类口径拆分为 OSError（基础设施）与其它（内容级终态）。
    """
    import noise_filter_pipeline.processors.apollo_superres as apollo_mod

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [
            _record_with_audio(tmpdir, i, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0)
            for i in range(1)
        ]
        write_manifest(iter(recs), manifest)
        out_dir = os.path.join(tmpdir, "d1")
        proc = ApolloSuperResProcessor({"mode": "dry_run", "target_sr": 44100, "output_format": "flac"})

        def _boom(*args, **kwargs):
            raise ValueError("corrupt audio frame")  # 内容级，非 OSError

        with monkeypatch.context() as m:
            m.setattr(apollo_mod, "copy_or_convert_audio", _boom)
            out_manifest = proc.process(manifest, out_dir)

        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 1
        d1 = out_recs[0].stages.get("D1", {})
        assert d1.get("status") == "rejected", d1
        assert out_recs[0].rejected_reason == "d1_copy_exception"
        # 内容级终态：写 checkpoint（resume 不再重试），且不计可重试失败
        assert not proc.has_retryable_failures()
        done = _read_ckpt_done(out_dir, name="checkpoint.dry_run.json")
        assert {r.record_id for r in recs} <= done


def _age_file(path, hours):
    """把文件/目录 mtime 调整为 N 小时前。"""
    old = os.path.getmtime(path) - hours * 3600
    os.utime(path, (old, old))


def test_cleanup_temp_dir_age_uses_subtree_latest_mtime():
    """stage_b_unified_* 目录年龄应按子树最新 mtime 判定：

    顶层目录 mtime 恒定（只在直接增删子项时变化），运行中的任务持续写
    子目录文件时顶层 mtime 不变，不得被误判为过期残留删除。
    """
    from noise_filter_pipeline.scripts.cleanup_pipeline_temp import _find_residuals

    with tempfile.TemporaryDirectory() as tmpdir:
        active = os.path.join(tmpdir, "stage_b_unified_active")
        os.makedirs(os.path.join(active, "sub"))
        fresh = os.path.join(active, "sub", "worker_0.jsonl")
        with open(fresh, "w") as f:
            f.write("x")
        # 顶层目录 mtime 调旧（48h 前），子树文件保持新鲜
        _age_file(active, 48)
        _age_file(os.path.join(active, "sub"), 48)

        stale = os.path.join(tmpdir, "stage_b_unified_stale")
        os.makedirs(stale)
        old_file = os.path.join(stale, "worker_0.jsonl")
        with open(old_file, "w") as f:
            f.write("x")
        _age_file(stale, 48)
        _age_file(old_file, 48)

        residuals = _find_residuals({tmpdir}, min_age_hours=24)
        names = {p.name for p in residuals}
        assert "stage_b_unified_stale" in names
        assert "stage_b_unified_active" not in names


def test_cleanup_temp_covers_atomic_write_tmp_files():
    """扫描应覆盖 tmp+replace 原子写的崩溃残留：audio mkstemp 隐藏临时文件、
    *.jsonl.tmp、*.complete.tmp、llm_cache 的 *.json.*.tmp。"""
    from noise_filter_pipeline.scripts.cleanup_pipeline_temp import _find_residuals

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = os.path.join(tmpdir, "stage_d", "D2", "audio", "sr_16k", "ds")
        os.makedirs(audio_dir)
        cache_dir = os.path.join(tmpdir, "llm_cache", "c1_qwen")
        os.makedirs(cache_dir)

        cases = {
            # audio_utils.save_audio_atomically 的 mkstemp 命名：.{basename}.<rand>{ext}
            os.path.join(audio_dir, ".clip.flac.ab12cd34.flac"): True,
            os.path.join(audio_dir, ".clip.wav.ab12cd34.wav"): True,
            # manifest/merged 原子写残留
            os.path.join(tmpdir, "stage_d", "d2_merged.jsonl.tmp"): True,
            # stage_runner .complete marker 原子写残留
            os.path.join(tmpdir, "stage_d", ".complete.tmp"): True,
            # llm_cache 原子写残留：<key>.json.<pid>.tmp
            os.path.join(cache_dir, "abcdef.json.12345.tmp"): True,
            # 正常文件不得命中
            os.path.join(audio_dir, "clip.flac"): False,
            os.path.join(tmpdir, "stage_d", "d2_manifest_ds.jsonl"): False,
            os.path.join(tmpdir, "stage_d", ".complete"): False,
            os.path.join(cache_dir, "abcdef.json"): False,
        }
        for path, _ in cases.items():
            with open(path, "w") as f:
                f.write("x")
            _age_file(path, 48)

        residuals = _find_residuals(set(), min_age_hours=24, recursive_roots={tmpdir})
        found = {os.path.abspath(str(p)) for p in residuals}
        for path, expected in cases.items():
            assert (os.path.abspath(path) in found) is expected, path


def test_cleanup_exposes_python_version_rmtree_capability_probe():
    """集成测试依赖的 rmtree 能力探针必须在所有 Python 版本可导入。"""
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    assert isinstance(module._RMTREE_SUPPORTS_DIR_FD, bool)


def test_cleanup_removes_stale_directory_with_dev_inode_identity(tmp_path):
    """普通目录 quarantine 必须使用 (st_dev, st_ino) 身份完成安全删除。"""
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    target = tmp_path / "stage_b_unified_stale"
    target.mkdir()
    (target / "worker.jsonl").write_text("stale", encoding="utf-8")
    old = os.path.getmtime(target) - 48 * 3600
    os.utime(target / "worker.jsonl", (old, old))
    os.utime(target, (old, old))

    candidates = module._find_residuals({str(tmp_path)}, min_age_hours=24)
    assert candidates == [target]
    assert module._remove_residuals(
        candidates, dry_run=False, min_age_hours=24
    ) == 1
    assert not target.exists()
    assert not list(tmp_path.glob("*.quarantine"))


def test_cleanup_final_rename_uses_stable_parent_dirfd(tmp_path, monkeypatch):
    """检查后 parent 路径被替换时，rename 仍只能作用于已打开的原 parent。"""
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    parent = tmp_path / "cache"
    parent.mkdir()
    target = parent / "checkpoint.json.tmp"
    target.write_text("original", encoding="utf-8")
    old = os.path.getmtime(target) - 48 * 3600
    os.utime(target, (old, old))
    candidates = module._find_residuals(
        set(), min_age_hours=24, recursive_roots={str(parent)}
    )
    assert candidates == [target]

    moved_parent = tmp_path / "cache.original"
    real_rename = module.os.rename
    swapped = False

    def swap_parent_at_quarantine_rename(source, destination, *args, **kwargs):
        nonlocal swapped
        if not swapped and ".cleanup-" in os.fspath(destination):
            swapped = True
            real_rename(parent, moved_parent)
            parent.mkdir()
            (parent / target.name).write_text("replacement", encoding="utf-8")
        return real_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(module.os, "rename", swap_parent_at_quarantine_rename)
    assert module._remove_residuals(
        candidates, dry_run=False, min_age_hours=24
    ) == 1
    assert swapped
    assert (parent / target.name).read_text(encoding="utf-8") == "replacement"
    assert not (moved_parent / target.name).exists()


def test_cleanup_rejects_candidate_replaced_by_symlink(tmp_path):
    """扫描后候选被 symlink 替换时，O_NOFOLLOW 最终边界必须 fail-closed。"""
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    target = tmp_path / "checkpoint.json.tmp"
    external = tmp_path / "external.txt"
    target.write_text("stale", encoding="utf-8")
    external.write_text("keep", encoding="utf-8")
    old = os.path.getmtime(target) - 48 * 3600
    os.utime(target, (old, old))
    candidates = module._find_residuals(
        set(), min_age_hours=24, recursive_roots={str(tmp_path)}
    )
    assert candidates == [target]

    target.unlink()
    target.symlink_to(external)
    assert module._remove_residuals(
        candidates, dry_run=False, min_age_hours=24
    ) == 0
    assert target.is_symlink()
    assert external.read_text(encoding="utf-8") == "keep"


def test_cleanup_revalidates_age_on_same_fd_after_rename(tmp_path, monkeypatch):
    """rename 瞬间候选变新时，同一候选 fd 的最终年龄复验必须拒绝删除。"""
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    target = tmp_path / "checkpoint.json.tmp"
    target.write_text("stale", encoding="utf-8")
    old = os.path.getmtime(target) - 48 * 3600
    os.utime(target, (old, old))
    candidates = module._find_residuals(
        set(), min_age_hours=24, recursive_roots={str(tmp_path)}
    )
    assert candidates == [target]

    real_rename = module.os.rename
    touched = False

    def touch_at_quarantine_rename(source, destination, *args, **kwargs):
        nonlocal touched
        if not touched and ".cleanup-" in os.fspath(destination):
            touched = True
            os.utime(target, None)
        return real_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(module.os, "rename", touch_at_quarantine_rename)
    assert module._remove_residuals(
        candidates, dry_run=False, min_age_hours=24
    ) == 0
    assert touched
    assert target.read_text(encoding="utf-8") == "stale"
    assert not list(tmp_path.glob("*.quarantine"))


def test_apollo_post_normalize_output_layout_sr_dataset_label():
    """D1+D2 合并模式应消费 D2 output_layout：最终音频落在
    audio/sr_*/<dataset>/<label>/ 下（skip 与 Apollo 两条写盘路径均覆盖）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [
            # sr=44100 >= target_sr：走 _skip_apollo 路径
            _record_with_audio(tmpdir, 0, {"audioset": "Rain"}, {"C3": "passed"}, sr=44100, duration=3.0),
            # sr=16000 < target_sr：走 Apollo dry_run 路径
            _record_with_audio(tmpdir, 1, {"audioset": "Wind"}, {"C3": "passed"}, sr=16000, duration=3.0),
        ]
        write_manifest(iter(recs), manifest)

        proc = ApolloSuperResProcessor({
            "mode": "dry_run",
            "target_sr": 44100,
            "post_normalize": {
                "target_rms": 0.1,
                "max_peak": 0.95,
                "min_duration": 2.0,
                "max_duration": 30.0,
                "output_sr": 32000,
                "output_layout": "sr_dataset_label",
            },
        })
        out_manifest = proc.process(manifest, os.path.join(tmpdir, "d1_d2"))
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 2
        audio_root = os.path.join(tmpdir, "d1_d2", "audio")
        for r, label in zip(out_recs, ["Rain", "Wind"]):
            assert r.is_stage_passed("D1"), r.stages
            assert r.is_stage_passed("D2"), r.stages
            assert os.path.isfile(r.audio["path"]), r.audio["path"]
            rel = os.path.relpath(r.audio["path"], audio_root)
            parts = rel.split(os.sep)
            assert parts[:3] == ["sr_32k", "ds", label], rel
        # 中间产物不得滞留在平铺层：audio/ 下只允许 sr_* 布局目录
        assert sorted(os.listdir(audio_root)) == ["sr_32k"]


def test_apollo_post_normalize_dataset_label_matches_standalone_d2_paths():
    """合并与独立 D2 均输出 audio/<dataset>/<category>/shard_<5位>/<4位>.flac。

    同一 dataset 的 ordinal 跨 category 全局递增，且覆盖 Apollo skip/dry-run 两条路径。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [
            # 交替覆盖 skip Apollo 与 Apollo dry_run 两条合并写盘路径；
            # category 交替变化，验证 ordinal 不随 category 重置。
            _record_with_audio(
                tmpdir, 0, {"audioset": "Rain"}, {"C3": "passed"},
                sr=44100, duration=3.0,
            ),
            _record_with_audio(
                tmpdir, 1, {"audioset": "Wind"}, {"C3": "passed"},
                sr=16000, duration=3.0,
            ),
            _record_with_audio(
                tmpdir, 2, {"audioset": "Rain"}, {"C3": "passed"},
                sr=44100, duration=3.0,
            ),
            _record_with_audio(
                tmpdir, 3, {"audioset": "Wind"}, {"C3": "passed"},
                sr=16000, duration=3.0,
            ),
        ]
        write_manifest(iter(recs), manifest)
        normalizer_cfg = {
            "target_rms": 0.1,
            "max_peak": 0.95,
            "min_duration": 2.0,
            "max_duration": 30.0,
            "output_sr": 32000,
            "output_format": "flac",
            "output_layout": "dataset_label",
        }

        standalone_dir = os.path.join(tmpdir, "standalone_d2")
        standalone_manifest = LoudnessNormalizerProcessor(normalizer_cfg).process(
            manifest, standalone_dir
        )
        merged_dir = os.path.join(tmpdir, "merged_d1_d2")
        merged_manifest = ApolloSuperResProcessor({
            "mode": "dry_run",
            "target_sr": 44100,
            "post_normalize": normalizer_cfg,
        }).process(manifest, merged_dir)

        def _relative_paths(output_manifest, output_dir):
            return {
                rec.record_id: os.path.relpath(
                    rec.audio["path"], os.path.join(output_dir, "audio")
                )
                for rec in read_manifest(output_manifest)
            }

        standalone_paths = _relative_paths(standalone_manifest, standalone_dir)
        merged_paths = _relative_paths(merged_manifest, merged_dir)
        expected_paths = {
            f"ds__src__{ordinal}": os.path.join(
                "ds",
                category,
                "shard_00000",
                f"{ordinal:04d}.flac",
            )
            for ordinal, category in enumerate(("Rain", "Wind", "Rain", "Wind"))
        }
        assert standalone_paths == expected_paths
        assert merged_paths == expected_paths

        ordinals_by_category = {}
        for relative_path in standalone_paths.values():
            parts = relative_path.split(os.sep)
            ordinals_by_category.setdefault(parts[1], []).append(int(Path(parts[3]).stem))
        assert ordinals_by_category == {"Rain": [0, 2], "Wind": [1, 3]}

        for output_dir, output_manifest in (
            (standalone_dir, standalone_manifest),
            (merged_dir, merged_manifest),
        ):
            for rec in read_manifest(output_manifest):
                assert rec.provenance["dataset_label_ordinal"] == int(
                    os.path.basename(rec.audio["path"])[-9:-5]
                )
                assert os.path.isfile(rec.audio["path"])
            assert sorted(os.listdir(os.path.join(output_dir, "audio", "ds"))) == [
                "Rain",
                "Wind",
            ]


def test_apollo_merged_dataset_label_checkpoint_rejects_category_identity_change(monkeypatch):
    """同 record_id/ordinal 但 C3 category 变化时不得恢复旧类别音频。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        rec = _record_with_audio(
            tmpdir, 0, {"audioset": "Rain"}, {"C3": "passed"}, sr=16000, duration=3.0
        )
        write_manifest(iter([rec]), manifest)
        cfg = {
            "mode": "dry_run",
            "target_sr": 44100,
            "post_normalize": {
                "target_rms": 0.1,
                "min_duration": 2.0,
                "max_duration": 30.0,
                "output_sr": 32000,
                "output_format": "flac",
                "output_layout": "dataset_label",
            },
        }
        out_dir = os.path.join(tmpdir, "d1_d2")
        output_manifest = ApolloSuperResProcessor(cfg).process(manifest, out_dir)
        previous = list(read_manifest(output_manifest))
        assert previous[0].provenance["dataset_label_ordinal"] == 0
        assert previous[0].audio["path"].endswith(
            os.path.join("ds", "Rain", "shard_00000", "0000.flac")
        )

        # 保持 input manifest 的 size/mtime 不变，确保 checkpoint 命中，
        # 只改变当前 C3 身份，覆盖真正的恢复身份校验而非配置指纹失效。
        input_stat = os.stat(manifest)
        current = list(read_manifest(manifest))
        assert current[0].record_id == previous[0].record_id
        current[0].stages["C3"]["balance_category"] = "Wind"
        write_manifest(iter(current), manifest)
        assert os.path.getsize(manifest) == input_stat.st_size
        os.utime(manifest, ns=(input_stat.st_atime_ns, input_stat.st_mtime_ns))

        calls = []
        proc = ApolloSuperResProcessor(cfg)
        original_run = proc._run_dry_run

        def _record_recompute(input_path, output_path):
            calls.append((input_path, output_path))
            return original_run(input_path, output_path)

        monkeypatch.setattr(proc, "_run_dry_run", _record_recompute)
        rerun_manifest = proc.process(manifest, out_dir)
        rerun = list(read_manifest(rerun_manifest))
        assert len(calls) == 1, "C3 category 变化时必须拒绝恢复并重算 D1+D2"
        assert len(rerun) == 1
        recomputed = rerun[0]
        assert recomputed.provenance["dataset_label_ordinal"] == 0
        assert recomputed.audio["path"].endswith(
            os.path.join("ds", "Wind", "shard_00000", "0000.flac")
        )
        assert os.path.isfile(recomputed.audio["path"])


@pytest.mark.parametrize("mutation", ["source_id", "input_fingerprint"])
def test_apollo_merged_dataset_label_checkpoint_rejects_source_identity_change(
    mutation, monkeypatch
):
    """同 record_id/dataset/category/ordinal 但源身份变化时必须重算。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        rec = _record_with_audio(
            tmpdir, 0, {"audioset": "Rain"}, {"C3": "passed"}, sr=16000, duration=3.0
        )
        write_manifest(iter([rec]), manifest)
        cfg = {
            "mode": "dry_run",
            "target_sr": 44100,
            "post_normalize": {
                "target_rms": 0.1,
                "min_duration": 2.0,
                "max_duration": 30.0,
                "output_sr": 32000,
                "output_format": "flac",
                "output_layout": "dataset_label",
            },
        }
        out_dir = os.path.join(tmpdir, "d1_d2")
        first_manifest = ApolloSuperResProcessor(cfg).process(manifest, out_dir)
        first = list(read_manifest(first_manifest))[0]
        assert first.audio["path"].endswith(
            os.path.join("ds", "Rain", "shard_00000", "0000.flac")
        )

        # 保持 manifest 的 size/mtime 不变，确保命中同一个 checkpoint；只改变
        # source_id 或原始输入文件 fingerprint，覆盖恢复身份校验本身。
        manifest_stat = os.stat(manifest)
        current = list(read_manifest(manifest))
        if mutation == "source_id":
            current[0].source_id = "s2c"
            assert current[0].record_id == first.record_id
        else:
            source_path = current[0].audio["path"]
            _make_wav(source_path, duration=3.0, sr=16000, rms=0.2)
        write_manifest(iter(current), manifest)
        assert os.path.getsize(manifest) == manifest_stat.st_size
        os.utime(manifest, ns=(manifest_stat.st_atime_ns, manifest_stat.st_mtime_ns))

        calls = []
        proc = ApolloSuperResProcessor(cfg)
        original_run = proc._run_dry_run

        def _record_recompute(input_path, output_path):
            calls.append((input_path, output_path))
            return original_run(input_path, output_path)

        monkeypatch.setattr(proc, "_run_dry_run", _record_recompute)
        rerun_manifest = proc.process(manifest, out_dir)
        rerun = list(read_manifest(rerun_manifest))
        assert len(calls) == 1, "源身份变化时不得恢复旧 dataset_label 产物"
        assert len(rerun) == 1
        assert rerun[0].audio["path"].endswith(
            os.path.join("ds", "Rain", "shard_00000", "0000.flac")
        )
        assert os.path.isfile(rerun[0].audio["path"])


def test_apollo_merged_normalize_content_exception_restores_ordinal_path_without_rerun(
    monkeypatch,
):
    """内容级归一化异常应清理中间产物，首次写入 ordinal 路径，后续可恢复。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_config = os.path.join(tmpdir, "apollo.yaml")
        mock_checkpoint = os.path.join(tmpdir, "apollo.ckpt")
        mock_daemon = os.path.join(tmpdir, "apollo_daemon.py")
        for path in (mock_config, mock_checkpoint, mock_daemon):
            Path(path).write_text("mock\n", encoding="utf-8")

        manifest = os.path.join(tmpdir, "input.jsonl")
        rec = _record_with_audio(
            tmpdir, 0, {"audioset": "Rain"}, {"C3": "passed"}, sr=16000, duration=3.0
        )
        write_manifest(iter([rec]), manifest)
        cfg = {
            "mode": "subprocess_daemon",
            "config_path": mock_config,
            "checkpoint": mock_checkpoint,
            "daemon_script_path": mock_daemon,
            "target_sr": 44100,
            "post_normalize": {
                "target_rms": 0.1,
                "min_duration": 2.0,
                "max_duration": 30.0,
                "output_sr": 32000,
                "output_format": "flac",
                "output_layout": "dataset_label",
            },
        }
        out_dir = os.path.join(tmpdir, "d1_d2")
        proc = ApolloSuperResProcessor(cfg)
        intermediate_paths = []

        def _mock_apollo(records, _output_dir, output_audio_dir, _batch_index):
            source = records[0].audio["path"]
            intermediate = os.path.join(output_audio_dir, "apollo_intermediate.wav")
            Path(intermediate).write_bytes(Path(source).read_bytes())
            intermediate_paths.append(intermediate)
            return {records[0].record_id: intermediate}

        def _content_error(*_args):
            raise ValueError("corrupt normalized content")

        monkeypatch.setattr(proc, "_run_subprocess_daemon", _mock_apollo)
        monkeypatch.setattr(proc._normalizer, "_normalize", _content_error)
        first_manifest = proc.process(manifest, out_dir)
        first = list(read_manifest(first_manifest))[0]
        expected_path = os.path.join(
            out_dir, "audio", "ds", "Rain", "shard_00000", "0000.flac"
        )
        assert first.stages["D2"]["status"] == "rejected"
        assert first.rejected_reason == "d2_exception"
        assert first.audio["path"] == expected_path
        assert not os.path.exists(intermediate_paths[0])
        assert rec.record_id in _read_ckpt_done(out_dir)

        retry_calls = []
        proc2 = ApolloSuperResProcessor(cfg)
        monkeypatch.setattr(
            proc2,
            "_run_subprocess_daemon",
            lambda *args: retry_calls.append(args) or {},
        )
        second_manifest = proc2.process(manifest, out_dir)
        second = list(read_manifest(second_manifest))[0]
        assert not retry_calls, "内容级 rejected 应从 checkpoint 恢复，不得重跑 Apollo"
        assert second.audio["path"] == expected_path
        assert second.stages["D2"]["status"] == "rejected"
        assert second.rejected_reason == "d2_exception"


def test_apollo_merged_dataset_label_sanitize_collision_fails_fast(tmp_path):
    """合并模式中清洗后相同的 dataset 目录必须 fail-fast，禁止覆盖音频。"""
    manifest = tmp_path / "input.jsonl"
    records = [
        _record_with_audio(
            str(tmp_path), 0, {"audioset": "Rain"}, {"C3": "passed"}, sr=44100, duration=3.0
        ),
        _record_with_audio(
            str(tmp_path), 1, {"audioset": "Rain"}, {"C3": "passed"}, sr=44100, duration=3.0
        ),
    ]
    records[0].dataset_id = "dataset a"
    records[1].dataset_id = "dataset,a"
    write_manifest(iter(records), str(manifest))

    cfg = {
        "mode": "dry_run",
        "target_sr": 44100,
        "post_normalize": {
            "target_rms": 0.1,
            "min_duration": 2.0,
            "max_duration": 30.0,
            "output_sr": 32000,
            "output_format": "flac",
            "output_layout": "dataset_label",
        },
    }
    with pytest.raises(ValueError, match=r"dataset.*(collision|冲突)|冲突.*dataset"):
        ApolloSuperResProcessor(cfg).process(
            str(manifest), str(tmp_path / "d1_d2")
        )


def test_apollo_merged_old_dataset_label_checkpoint_recomputes_missing_provenance(monkeypatch):
    """D1+D2 合并模式遇到旧顺序元数据时必须重算而非恢复旧音频。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        rec = _record_with_audio(
            tmpdir, 0, {"audioset": "Rain"}, {"C3": "passed"}, sr=16000, duration=3.0
        )
        write_manifest(iter([rec]), manifest)
        cfg = {
            "mode": "dry_run",
            "target_sr": 44100,
            "post_normalize": {
                "target_rms": 0.1,
                "min_duration": 2.0,
                "max_duration": 30.0,
                "output_sr": 32000,
                "output_format": "flac",
                "output_layout": "dataset_label",
            },
        }
        out_dir = os.path.join(tmpdir, "d1_d2")
        output_manifest = ApolloSuperResProcessor(cfg).process(manifest, out_dir)
        previous = list(read_manifest(output_manifest))
        assert previous[0].provenance["dataset_label_ordinal"] == 0
        assert os.path.isfile(previous[0].audio["path"])

        # 模拟旧 D1 checkpoint：旧输出记录缺少三项顺序命名元数据，但 checkpoint
        # 仍声明该 record 已完成；新实现必须拒绝恢复并重建最终路径。
        previous[0].provenance.pop("dataset_label_naming_version")
        previous[0].provenance.pop("dataset_label_ordinal")
        previous[0].provenance.pop("dataset_label_shard_size")
        per_dataset_manifest = os.path.join(out_dir, "d1_manifest_ds.jsonl")
        write_manifest(iter(previous), per_dataset_manifest)

        calls = []
        proc = ApolloSuperResProcessor(cfg)
        original_run = proc._run_dry_run

        def _record_recompute(input_path, output_path):
            calls.append((input_path, output_path))
            return original_run(input_path, output_path)

        monkeypatch.setattr(proc, "_run_dry_run", _record_recompute)
        rerun_manifest = proc.process(manifest, out_dir)
        rerun = list(read_manifest(rerun_manifest))
        assert len(calls) == 1, "缺少 provenance 的旧 checkpoint 必须重新执行 D1+D2"
        assert len(rerun) == 1
        restored = rerun[0]
        assert restored.is_stage_passed("D1")
        assert restored.is_stage_passed("D2")
        assert restored.provenance["dataset_label_naming_version"] == 1
        assert restored.provenance["dataset_label_ordinal"] == 0
        assert restored.provenance["dataset_label_shard_size"] == 10000
        assert restored.audio["path"].endswith(
            os.path.join("ds", "Rain", "shard_00000", "0000.flac")
        )
        assert os.path.isfile(restored.audio["path"])
        assert set(_read_ckpt_done(out_dir, name="checkpoint.dry_run.json")) == {rec.record_id}


def test_apollo_subprocess_daemon_success_runs_merged_dataset_label_path():
    """CPU-only mock daemon 覆盖增强、D2 归一化、顺序命名及 checkpoint。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_daemon = os.path.join(tmpdir, "mock_apollo_daemon.py")
        _write_mock_apollo_audio_daemon(mock_daemon)
        config_path = os.path.join(tmpdir, "apollo.yaml")
        checkpoint_path = os.path.join(tmpdir, "apollo.ckpt")
        Path(config_path).write_text("mock: true\n", encoding="utf-8")
        Path(checkpoint_path).write_bytes(b"mock checkpoint")

        manifest = os.path.join(tmpdir, "input.jsonl")
        records = [
            _record_with_audio(
                tmpdir, 0, {"audioset": "Rain"}, {"C3": "passed"}, sr=16000, duration=3.0
            ),
            _record_with_audio(
                tmpdir, 1, {"audioset": "Wind"}, {"C3": "passed"}, sr=16000, duration=3.0
            ),
        ]
        write_manifest(iter(records), manifest)
        proc = ApolloSuperResProcessor({
            "mode": "subprocess_daemon",
            "python_path": sys.executable,
            "daemon_script_path": mock_daemon,
            "config_path": config_path,
            "checkpoint": checkpoint_path,
            "batch_size": 2,
            "daemon_timeout": 30,
            "daemon_startup_timeout": 10,
            "target_sr": 44100,
            "post_normalize": {
                "target_rms": 0.1,
                "min_duration": 2.0,
                "max_duration": 30.0,
                "output_sr": 32000,
                "output_format": "flac",
                "output_layout": "dataset_label",
            },
        })

        out_dir = os.path.join(tmpdir, "d1_d2")
        output_manifest = proc.process(manifest, out_dir)
        output_records = list(read_manifest(output_manifest))
        assert [r.provenance["dataset_label_ordinal"] for r in output_records] == [0, 1]
        assert all(r.provenance["dataset_label_naming_version"] == 1 for r in output_records)
        assert all(r.provenance["dataset_label_shard_size"] == 10000 for r in output_records)
        assert all(r.is_stage_passed("D1") and r.is_stage_passed("D2") for r in output_records)
        assert [
            os.path.relpath(r.audio["path"], os.path.join(out_dir, "audio"))
            for r in output_records
        ] == [
            os.path.join("ds", "Rain", "shard_00000", "0000.flac"),
            os.path.join("ds", "Wind", "shard_00000", "0001.flac"),
        ]
        assert all(os.path.isfile(r.audio["path"]) for r in output_records)
        assert all(sf.info(r.audio["path"]).samplerate == 32000 for r in output_records)
        assert set(_read_ckpt_done(out_dir)) == {r.record_id for r in records}


def test_apollo_merged_resume_restores_d2_rejected_without_rerun():
    """合并模式 resume：D2 rejected 记录的 Apollo 中间产物已被 _normalize_record
    按预期删除，resume 应跳过产物可读性校验直接恢复，不得无限重跑。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        # duration=1.0 < min_duration=2.0 → D2 必拒；sr=44100 走 _skip_apollo 路径
        recs = [_record_with_audio(tmpdir, 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=1.0)]
        write_manifest(iter(recs), manifest)

        cfg = {
            "mode": "dry_run",
            "target_sr": 44100,
            "post_normalize": {"target_rms": 0.1, "max_peak": 0.95, "min_duration": 2.0, "max_duration": 30.0},
        }
        out_dir = os.path.join(tmpdir, "d1_d2")
        proc = ApolloSuperResProcessor(cfg)
        out_manifest = proc.process(manifest, out_dir)
        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 1
        assert out_recs[0].stages.get("D2", {}).get("status") == "rejected"
        # 中间产物已按预期删除
        assert not os.path.exists(out_recs[0].audio["path"])

        # 重跑：D2 rejected 是内容级终态，应直接恢复，不再触发 Apollo/复制
        proc2 = ApolloSuperResProcessor(cfg)
        called = []
        orig_skip = proc2._skip_apollo

        def _spy_skip(rec, output_audio_dir):
            called.append(rec.record_id)
            return orig_skip(rec, output_audio_dir)

        proc2._skip_apollo = _spy_skip
        orig_dry = proc2._run_dry_run

        def _spy_dry(input_path, output_path):
            called.append(input_path)
            return orig_dry(input_path, output_path)

        proc2._run_dry_run = _spy_dry
        out_manifest2 = proc2.process(manifest, out_dir)
        assert not called, called
        out_recs2 = list(read_manifest(out_manifest2))
        assert len(out_recs2) == 1
        assert out_recs2[0].stages.get("D2", {}).get("status") == "rejected"
        assert out_recs2[0].rejected_reason == "d2_duration_out_of_range"


def test_apollo_dry_run_writes_atomically():
    """dry_run 无 post_normalize 分支应走 save_audio_atomically（原子写），
    不得裸 sf.write 留下崩溃半成品。"""
    import noise_filter_pipeline.processors.apollo_superres as apollo_mod

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [_record_with_audio(tmpdir, 0, {"leaf": "noise"}, {"C3": "passed"}, sr=16000, duration=3.0)]
        write_manifest(iter(recs), manifest)

        calls = []
        orig_save = apollo_mod.save_audio_atomically

        def _spy_save(audio, sr, dst_path):
            calls.append(dst_path)
            return orig_save(audio, sr, dst_path)

        apollo_mod.save_audio_atomically = _spy_save
        try:
            proc = ApolloSuperResProcessor({"mode": "dry_run", "target_sr": 44100})
            out_manifest = proc.process(manifest, os.path.join(tmpdir, "d1"))
        finally:
            apollo_mod.save_audio_atomically = orig_save

        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 1
        assert out_recs[0].is_stage_passed("D1")
        assert out_recs[0].audio["path"] in calls


def _write_relayout_fixture(root, sr_field=16000):
    """构造最小 stage_d 目录：D2/d2_merged.jsonl + 二级布局音频 + 根 manifest。"""
    d2_dir = os.path.join(root, "D2")
    audio_dir = os.path.join(d2_dir, "audio", "sr_16k", "ds")
    os.makedirs(audio_dir)
    old_path = os.path.join(audio_dir, "clip.flac")
    with open(old_path, "wb") as f:
        f.write(b"fLaC")
    rec = {
        "record_id": "ds__src__0",
        "dataset_id": "ds",
        "audio": {"path": old_path, "sr": sr_field, "duration": 3.0},
        "labels": {"audioset": "Rain"},
        "stages": {"D2": {"status": "passed"}},
    }
    line = json.dumps(rec, ensure_ascii=False)
    merged = os.path.join(d2_dir, "d2_merged.jsonl")
    with open(merged, "w", encoding="utf-8") as f:
        f.write(line + "\n")
    for name in ("d_manifest.jsonl", "d_manifest_ds.jsonl"):
        with open(os.path.join(root, name), "w", encoding="utf-8") as f:
            f.write(line + "\n")
    return old_path, line


def test_relayout_rewrites_per_dataset_manifests(monkeypatch):
    """relayout 除重建遗留单文件外，还必须用同一套路径替换重写
    per-dataset d_manifest_<dataset>.jsonl。"""
    import sys

    from noise_filter_pipeline.scripts import relayout_stage_d_audio

    with tempfile.TemporaryDirectory() as root:
        old_path, line = _write_relayout_fixture(root)
        monkeypatch.setattr(sys, "argv", ["relayout_stage_d_audio.py", "--root", root])
        assert relayout_stage_d_audio.main() == 0

        new_path = old_path.replace(
            os.path.join("sr_16k", "ds", "clip.flac"),
            os.path.join("sr_16k", "ds", "Rain", "clip.flac"),
        )
        assert os.path.isfile(new_path)
        assert not os.path.exists(old_path)
        for name in ("d_manifest.jsonl", "d_manifest_ds.jsonl"):
            with open(os.path.join(root, name), "r", encoding="utf-8") as f:
                content = f.read()
            assert new_path in content, name
            assert old_path not in content, name
        with open(os.path.join(root, "D2", "d2_merged.jsonl"), "r", encoding="utf-8") as f:
            assert new_path in f.read()


def test_relayout_interrupted_rerun_heals_per_dataset_manifests(monkeypatch):
    """中断后重跑：merged 已在新布局（跳过分支）时，per-dataset 的旧路径
    残留必须仍被修复——不得依赖本次运行的 path_map（重跑时为空）。

    回归：path_map 只记录"本次实际移动"的路径；中断后重跑 merged 全命中
    already 分支、path_map 为空，per-dataset 文件中的旧路径永不修正，
    下游读到已移动的旧位置。
    """
    import sys

    from noise_filter_pipeline.scripts import relayout_stage_d_audio

    with tempfile.TemporaryDirectory() as root:
        old_path, line = _write_relayout_fixture(root)
        new_path = old_path.replace(
            os.path.join("sr_16k", "ds", "clip.flac"),
            os.path.join("sr_16k", "ds", "Rain", "clip.flac"),
        )
        # 模拟中断现场：音频已移动、merged 与根 manifest 已是新路径、
        # per-dataset d_manifest_ds.jsonl 仍是旧路径（中断发生在重写之前）
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        os.rename(old_path, new_path)
        new_line = line.replace(old_path, new_path)
        with open(os.path.join(root, "D2", "d2_merged.jsonl"), "w", encoding="utf-8") as f:
            f.write(new_line + "\n")
        with open(os.path.join(root, "d_manifest.jsonl"), "w", encoding="utf-8") as f:
            f.write(new_line + "\n")

        monkeypatch.setattr(sys, "argv", ["relayout_stage_d_audio.py", "--root", root])
        assert relayout_stage_d_audio.main() == 0

        with open(os.path.join(root, "d_manifest_ds.jsonl"), "r", encoding="utf-8") as f:
            content = f.read()
        assert new_path in content
        assert old_path not in content


def test_relayout_refuses_without_stage_d_run_lock(monkeypatch):
    """Stage D 写者持 run 锁时，relayout 必须拒绝改写（不得无锁改产物）。"""
    import sys

    from noise_filter_pipeline.core.run_lock import stage_d_run_lock
    from noise_filter_pipeline.scripts import relayout_stage_d_audio

    with tempfile.TemporaryDirectory() as root:
        old_path, _line = _write_relayout_fixture(root)
        monkeypatch.setattr(sys, "argv", ["relayout_stage_d_audio.py", "--root", root])
        with stage_d_run_lock(root):
            with pytest.raises(RuntimeError, match="run 锁"):
                relayout_stage_d_audio.main()
        # 锁冲突中止后现场保持原样：音频未移动、merged 未替换
        assert os.path.isfile(old_path)


def test_relayout_missing_sr_aborts_with_records(monkeypatch, capsys):
    """SR_RE 未命中（sr 缺失/null）不得静默归入 sr_0/：显式报错并列出记录，
    merged 与音频保持原样。"""
    import sys

    from noise_filter_pipeline.scripts import relayout_stage_d_audio

    with tempfile.TemporaryDirectory() as root:
        old_path, line = _write_relayout_fixture(root, sr_field=None)
        # json.dumps(None) → "sr": null，SR_RE 无法匹配数字
        assert '"sr": null' in line
        monkeypatch.setattr(sys, "argv", ["relayout_stage_d_audio.py", "--root", root])
        rc = relayout_stage_d_audio.main()
        assert rc == 1
        out = capsys.readouterr().out
        assert "ds__src__0" in out
        # 中止后现场保持原样：merged 未替换、音频未移动、不落入 sr_0/
        with open(os.path.join(root, "D2", "d2_merged.jsonl"), "r", encoding="utf-8") as f:
            assert f.read() == line + "\n"
        assert os.path.isfile(old_path)
        assert not os.path.exists(os.path.join(root, "D2", "audio", "sr_0"))


def test_scan_corrupted_parser_defaults_to_portable_registry(monkeypatch):
    """scan 无显式参数时使用便携 noise.yaml registry。"""
    from noise_filter_pipeline.scripts.scan_corrupted_audio import parse_args

    monkeypatch.setattr(sys, "argv", ["scan_corrupted_audio.py"])
    assert parse_args().registry == "tools/noise_filter_pipeline/config/noise.yaml"


def test_scan_corrupted_delete_handles_tab_in_filename():
    """delete 模式直接传递 (path, reason) 元组：含 tab 的文件名不被截断误删。"""
    from noise_filter_pipeline.scripts.scan_corrupted_audio import delete_bad_files

    with tempfile.TemporaryDirectory() as tmpdir:
        tab_name = os.path.join(tmpdir, "bad\tfile.wav")
        decoy = os.path.join(tmpdir, "bad")
        for p in (tab_name, decoy):
            with open(p, "w") as f:
                f.write("x")

        attempted, removed = delete_bad_files(
            [("ds", tab_name, "无法读取音频信息")],
            dry_run=False,
            dataset_roots={"ds": tmpdir},
        )
        assert (attempted, removed) == (1, 1)
        assert not os.path.exists(tab_name)
        # 旧字符串解析会把路径截断到 tab 前，误删 decoy；元组传递不得误删
        assert os.path.isfile(decoy)


def test_scan_corrupted_delete_skips_paths_outside_dataset_root():
    """--delete 只删 registry 数据集根目录内的文件（对照 B4 _safe_delete_audio）。"""
    from noise_filter_pipeline.scripts.scan_corrupted_audio import delete_bad_files

    with tempfile.TemporaryDirectory() as tmpdir:
        root = os.path.join(tmpdir, "root")
        outside = os.path.join(tmpdir, "outside")
        os.makedirs(root)
        os.makedirs(outside)
        inside_file = os.path.join(root, "bad.wav")
        outside_file = os.path.join(outside, "bad.wav")
        for p in (inside_file, outside_file):
            with open(p, "w") as f:
                f.write("x")

        attempted, removed = delete_bad_files(
            [("ds", inside_file, "r"), ("ds", outside_file, "r"), ("unknown", inside_file, "r")],
            dry_run=False,
            dataset_roots={"ds": root},
        )

        # 只有根目录内的那条被计数并删除，其余两条（越界/根目录未知）跳过
        assert (attempted, removed) == (1, 1)
        assert not os.path.exists(inside_file)
        assert os.path.isfile(outside_file)


def test_download_hive_models_atomic_tmp(monkeypatch):
    """下载先落 dest.tmp 再 os.replace：失败不留部分文件，成功无 tmp 残留。"""
    import urllib.request

    from noise_filter_pipeline.scripts.download_hive_models import _download

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = os.path.join(tmpdir, "sub", "model.ckpt")

        def _partial_fail(url, path):
            with open(path, "wb") as f:
                f.write(b"partial")
            raise RuntimeError("connection reset")

        monkeypatch.setattr(urllib.request, "urlretrieve", _partial_fail)
        assert _download("http://example/m", dest) is False
        assert not os.path.exists(dest)
        assert not os.path.exists(dest + ".tmp")

        def _ok(url, path):
            with open(path, "wb") as f:
                f.write(b"full-content")

        monkeypatch.setattr(urllib.request, "urlretrieve", _ok)
        assert _download("http://example/m", dest) is True
        with open(dest, "rb") as f:
            assert f.read() == b"full-content"
        assert not os.path.exists(dest + ".tmp")


def test_download_nemo_vad_onnx_atomic_tmp(monkeypatch):
    """NeMo VAD ONNX 下载失败清理随机临时文件，成功后原子发布。"""
    import hashlib

    from noise_filter_pipeline.scripts import download_nemo_vad_onnx as downloader

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "vad_marblenet.onnx")
        payload = b"onnx-bytes"
        monkeypatch.setenv("NEMO_VAD_ONNX_URL", "https://example/vad.onnx")
        monkeypatch.setenv("NEMO_VAD_ONNX_SHA256", hashlib.sha256(payload).hexdigest())
        monkeypatch.setattr(downloader, "_validate_onnx_file", lambda path: True)

        failed_tmp_paths = []

        def _partial_fail(url, path):
            failed_tmp_paths.append(path)
            Path(path).write_bytes(b"partial")
            raise RuntimeError("timeout")

        monkeypatch.setattr(downloader, "_download_to_temp", _partial_fail)
        assert downloader._try_download(tmpdir, out_path) is False
        assert failed_tmp_paths
        assert failed_tmp_paths[0].endswith(".download.tmp")
        assert not os.path.exists(out_path)
        assert not list(Path(tmpdir).glob("*.download.tmp"))

        published = []
        real_replace = os.replace

        def _replace(source, destination):
            published.append((source, destination))
            real_replace(source, destination)

        monkeypatch.setattr(downloader.os, "replace", _replace)

        def _ok(url, path):
            Path(path).write_bytes(payload)
            return hashlib.sha256(payload).hexdigest()

        monkeypatch.setattr(downloader, "_download_to_temp", _ok)
        assert downloader._try_download(tmpdir, out_path) is True
        assert Path(out_path).read_bytes() == payload
        assert len(published) == 1
        assert published[0][1] == out_path
        assert published[0][0].endswith(".download.tmp")
        assert not list(Path(tmpdir).glob("*.download.tmp"))

        # 无 URL 配置时直接返回 False，不触碰文件系统
        monkeypatch.delenv("NEMO_VAD_ONNX_URL")
        assert downloader._try_download(tmpdir, out_path) is False


def test_apollo_merged_d2_infra_failure_is_not_checkpointed(monkeypatch):
    """合并 D2 的 OSError 不得被 D1 checkpoint 恢复为完成结果。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        rec = _record_with_audio(tmpdir, 0, {"leaf": "noise"}, {"C3": "passed"}, sr=16000, duration=3.0)
        write_manifest(iter([rec]), manifest)
        config_path = os.path.join(tmpdir, "apollo.yml")
        checkpoint_path = os.path.join(tmpdir, "apollo.ckpt")
        open(config_path, "w").close()
        open(checkpoint_path, "w").close()
        cfg = {
            "mode": "subprocess_daemon",
            "daemon_script_path": apollo_superres.__file__,
            "config_path": config_path,
            "checkpoint": checkpoint_path,
            "target_sr": 44100,
            "post_normalize": {"target_rms": 0.1, "min_duration": 2.0, "max_duration": 30.0},
        }

        proc = ApolloSuperResProcessor(cfg)
        proc._run_subprocess_daemon = lambda records, *_args: {
            item.record_id: item.audio["path"] for item in records
        }
        monkeypatch.setattr(proc._normalizer, "_normalize", lambda *_args: (_ for _ in ()).throw(OSError("disk full")))
        out_dir = os.path.join(tmpdir, "d1")
        out_manifest = proc.process(manifest, out_dir)
        out_rec = next(read_manifest(out_manifest))
        assert out_rec.stages["D1"]["status"] == "failed"
        assert proc.has_retryable_failures()
        assert rec.record_id not in _read_ckpt_done(out_dir)

        retry_calls = []
        proc2 = ApolloSuperResProcessor(cfg)
        proc2._run_subprocess_daemon = lambda records, *_args: (
            retry_calls.append([item.record_id for item in records]) or {
                item.record_id: item.audio["path"] for item in records
            }
        )
        proc2.process(manifest, out_dir)
        assert retry_calls == [[rec.record_id]], "未 checkpoint 的基础设施失败必须重试"


def test_apollo_dry_run_content_failure_is_checkpointed(monkeypatch):
    """dry_run 内容级失败是终态，重跑不得再执行同一条记录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        rec = _record_with_audio(tmpdir, 0, {"leaf": "noise"}, {"C3": "passed"}, sr=16000, duration=3.0)
        write_manifest(iter([rec]), manifest)
        out_dir = os.path.join(tmpdir, "d1")
        proc = ApolloSuperResProcessor({"mode": "dry_run", "target_sr": 44100})
        monkeypatch.setattr(proc, "_run_dry_run", lambda *_args: (_ for _ in ()).throw(ValueError("corrupt audio")))
        out_manifest = proc.process(manifest, out_dir)
        out_rec = next(read_manifest(out_manifest))
        assert out_rec.stages["D1"]["status"] == "rejected"
        assert rec.record_id in _read_ckpt_done(out_dir, name="checkpoint.dry_run.json")

        proc2 = ApolloSuperResProcessor({"mode": "dry_run", "target_sr": 44100})
        retry_calls = []
        monkeypatch.setattr(proc2, "_run_dry_run", lambda *_args: retry_calls.append(True))
        proc2.process(manifest, out_dir)
        assert not retry_calls, "终态 rejected 应从 checkpoint 恢复，不得重复处理"


def test_apollo_merged_num_workers_excluded_from_fingerprint():
    """合并模式下 post_normalize 携带的 num_workers（orchestrator 合并 d2_cfg
    时注入，默认 min(8, cpu_count) 随机器变化）不得进入 D1 checkpoint 指纹。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = [_record_with_audio(tmpdir, 0, {"leaf": "noise"}, {"C3": "passed"}, sr=44100, duration=3.0)]
        write_manifest(iter(recs), manifest)

        hashes = []
        for workers in (1, 8):
            proc = ApolloSuperResProcessor({
                "mode": "dry_run",
                "target_sr": 44100,
                "post_normalize": {
                    "target_rms": 0.1, "min_duration": 2.0, "max_duration": 30.0,
                    "num_workers": workers,
                },
            })
            assert "num_workers" not in proc.config["post_normalize"]
            out_dir = os.path.join(tmpdir, f"d1_w{workers}")
            proc.process(manifest, out_dir)
            with open(os.path.join(out_dir, "checkpoint.dry_run.json"), "r", encoding="utf-8") as f:
                hashes.append(json.load(f)["config_hash"])
        assert hashes[0] == hashes[1]
