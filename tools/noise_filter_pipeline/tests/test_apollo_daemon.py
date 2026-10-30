"""
Tests for D1 Apollo subprocess_daemon mode。

- 真实 daemon 协议测试：用 CPU 启动 inference_apollo.py --daemon，验证
  ping/pong、enhance 单条短音频、stop 干净退出。
- round-robin 分发测试：mock daemon（不加载模型），验证多 GPU 分发、
  daemon 只启动一次、结果映射与中间产物清理。
- 配置测试：pipeline.yaml 的 d1 mode/num_gpus。
"""
import json
import os
import select
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
import yaml

from noise_filter_pipeline.core.manifest_io import AudioRecord, read_manifest, write_manifest
from noise_filter_pipeline.processors import apollo_superres
from noise_filter_pipeline.processors.apollo_superres import (
    ApolloSuperResProcessor,
    _ApolloDaemonPool,
    _ApolloDaemonRetryableError,
    _DaemonPoolExhaustedError,
)

# 真实 daemon 协议测试依赖独立 conda 环境（look2hear/torch 等），归类为 pipeline 外部集成测试
pytestmark = [pytest.mark.pipeline, pytest.mark.integration, pytest.mark.external]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
APOLLO_SCRIPT = os.path.join(_REPO_ROOT, "third_party_scripts", "Apollo", "inference_apollo.py")
APOLLO_CKPT = os.path.join(_REPO_ROOT, "third_party_scripts", "Apollo", "apollo_model_uni.ckpt")
APOLLO_CONFIG = os.path.join(_REPO_ROOT, "third_party_scripts", "Apollo", "config_apollo_uni.yaml")
# Apollo 依赖（look2hear/torch/ml_collections）只安装在独立 conda 环境中
APOLLO_PYTHON = "/root/miniconda3/envs/noise_filter_pipeline/bin/python"
PIPELINE_YAML = os.path.join(
    _REPO_ROOT, "tools", "noise_filter_pipeline", "config", "pipeline.yaml"
)


def _make_wav(path, duration=1.0, sr=16000, freq=440.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.arange(int(duration * sr)) / sr
    wav = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, wav, sr)


def _send(proc, msg):
    proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
    proc.stdin.flush()


def _recv(proc, timeout=300):
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    assert ready, f"daemon 响应超时（>{timeout}s）"
    line = proc.stdout.readline()
    assert line, "daemon 意外结束输出"
    return json.loads(line)


@pytest.mark.skipif(
    not (os.path.isfile(APOLLO_PYTHON) and os.path.isfile(APOLLO_CKPT) and os.path.isfile(APOLLO_CONFIG)),
    reason="Apollo conda 环境或模型文件缺失",
)
def test_apollo_daemon_protocol_real_model():
    """真实 Apollo daemon（CPU）：ping -> pong，enhance 一条 1s tone，stop 干净退出。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "tone.wav")
        _make_wav(src, duration=1.0, sr=16000)
        out_dir = os.path.join(tmpdir, "enhanced")

        proc = subprocess.Popen(
            [
                APOLLO_PYTHON,
                APOLLO_SCRIPT,
                "--daemon",
                "--output_audio_dir", out_dir,
                "--device", "cpu",
                "--batch_size", "1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            cwd=_REPO_ROOT,
        )
        try:
            # ping -> pong（首次响应包含模型加载，CPU 上可能较慢）
            _send(proc, {"type": "ping"})
            resp = _recv(proc, timeout=600)
            assert resp.get("type") == "pong"

            # enhance -> result，输出文件存在且采样率为 target_sr(44100)
            _send(proc, {"type": "enhance", "batch_id": 1, "items": [{"audio_path": src}]})
            resp = _recv(proc, timeout=600)
            assert resp.get("type") == "result"
            assert resp.get("batch_id") == 1
            assert len(resp["results"]) == 1
            item = resp["results"][0]
            assert item.get("audio_path") == src
            enhanced = item.get("enhanced_audio_path")
            assert enhanced and os.path.isfile(enhanced), f"增强产物缺失: {item}"
            _, sr = sf.read(enhanced)
            assert sr == 44100

            # stop -> stopped，进程干净退出
            _send(proc, {"type": "stop"})
            resp = _recv(proc, timeout=60)
            assert resp.get("type") == "stopped"
            code = proc.wait(timeout=60)
            assert code == 0
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def _write_mock_apollo_daemon(script_path: str, counter_dir: str) -> None:
    """写一个遵循 Apollo daemon 协议的 mock daemon（不加载模型，直接拷贝输入）。"""
    code = '''
import json
import os
import shutil
import sys

out_dir = "."
args = sys.argv[1:]
for i, a in enumerate(args):
    if a == "--output_audio_dir" and i + 1 < len(args):
        out_dir = args[i + 1]
os.makedirs(out_dir, exist_ok=True)

gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
counter_dir = {counter_dir!r}
def mark(tag):
    with open(os.path.join(counter_dir, "gpu_%s.log" % gpu), "a") as f:
        f.write(tag + "\\n")

# 进程启动即记录一次，用于验证 daemon 只启动一次（模型只加载一次）
mark("start")

def reply(resp):
    sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    cmd = json.loads(line)
    t = cmd.get("type")
    if t == "ping":
        reply({{"type": "pong"}})
    elif t == "stop":
        reply({{"type": "stopped"}})
        break
    elif t == "enhance":
        results = []
        for item in cmd.get("items", []):
            src = item["audio_path"]
            stem = os.path.splitext(os.path.basename(src))[0]
            dst = os.path.join(out_dir, stem + ".wav")
            n = 1
            while os.path.exists(dst):
                dst = os.path.join(out_dir, "%s_%d.wav" % (stem, n))
                n += 1
            shutil.copyfile(src, dst)
            results.append({{"audio_path": src, "enhanced_audio_path": dst}})
        mark("enhance:%d" % len(results))
        reply({{"type": "result", "batch_id": cmd.get("batch_id"), "results": results}})
    else:
        reply({{"type": "error", "message": "unknown type %s" % t}})
'''.format(counter_dir=counter_dir)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)


def test_apollo_subprocess_daemon_round_robin():
    """mock daemon 验证：多 GPU round-robin 分发、daemon 复用、结果映射与中间产物清理。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_daemon = os.path.join(tmpdir, "mock_apollo_daemon.py")
        counter_dir = os.path.join(tmpdir, "counters")
        os.makedirs(counter_dir)
        _write_mock_apollo_daemon(mock_daemon, counter_dir)

        # 8 条 16kHz 记录（< target_sr，走 Apollo 路径）；
        # batch_size=2, num_gpus=2 -> chunk_size=4，共 2 个 micro-batch
        audio_dir = os.path.join(tmpdir, "src_audio")
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = []
        for i in range(8):
            wav_path = os.path.join(audio_dir, f"a{i}.wav")
            _make_wav(wav_path, duration=2.5, sr=16000)
            recs.append(
                AudioRecord(
                    record_id=f"ds__src__{i}",
                    dataset_id="ds",
                    source_id="src",
                    audio={"path": wav_path, "sr": 16000, "duration": 2.5, "channels": 1},
                    labels={"original": "noise"},
                )
            )
        write_manifest(iter(recs), manifest)

        dummy_ckpt = os.path.join(tmpdir, "dummy.ckpt")
        dummy_cfg = os.path.join(tmpdir, "dummy.yaml")
        for p in (dummy_ckpt, dummy_cfg):
            with open(p, "w", encoding="utf-8") as f:
                f.write("")

        proc = ApolloSuperResProcessor(
            {
                "mode": "subprocess_daemon",
                "daemon_script_path": mock_daemon,
                "python_path": sys.executable,
                "checkpoint": dummy_ckpt,
                "config_path": dummy_cfg,
                "batch_size": 2,
                "num_gpus": 2,
                "checkpoint_save_interval": 10,
                "manifest_write_batch_size": 10,
            }
        )
        out_dir = os.path.join(tmpdir, "d1")
        output_manifest = proc.process(manifest, out_dir)

        # 全部记录 D1 passed，产物存在
        out_recs = list(read_manifest(output_manifest))
        assert len(out_recs) == 8
        for rec in out_recs:
            assert rec.is_stage_passed("D1"), rec.record_id
            assert os.path.isfile(rec.audio["path"]), rec.audio["path"]

        # 两个 GPU 的 daemon 都被使用（round-robin 分片并行）
        gpu_logs = {}
        for name in os.listdir(counter_dir):
            with open(os.path.join(counter_dir, name), "r", encoding="utf-8") as f:
                gpu_logs[name] = [line.strip() for line in f if line.strip()]
        assert len(gpu_logs) == 2, gpu_logs
        for name, lines in gpu_logs.items():
            # 每个 daemon 只启动一次（模型只加载一次）
            assert lines.count("start") == 1, (name, lines)
            # 每个 daemon 都处理过 enhance 请求
            assert any(line.startswith("enhance:") for line in lines), (name, lines)

        # 中间产物已清理：无 *_inputs symlink 目录、无 {hash}.wav 残留
        out_audio_dir = os.path.join(out_dir, "audio")
        leftovers = [
            name for name in os.listdir(out_audio_dir)
            if name.endswith("_inputs") or name.endswith(".wav")
        ]
        assert not leftovers, leftovers

        # 进程结束后 daemon 已被回收
        assert proc._daemon_pool is not None
        assert not proc._daemon_pool._daemon_procs


def _make_daemon_processor_for_ownership_test(tmp_path):
    """构造只执行响应归属校验、不启动真实 daemon 的 D1 processor。"""
    proc = ApolloSuperResProcessor(
        {
            "mode": "subprocess_daemon",
            "daemon_script_path": __file__,
            "python_path": sys.executable,
            "checkpoint": "dummy.ckpt",
            "config_path": "dummy.yaml",
            "batch_size": 2,
        }
    )
    proc._daemon_output_root = str(tmp_path / "daemon_outputs")
    os.makedirs(proc._daemon_output_root)
    return proc


def test_apollo_daemon_rejects_cross_item_output_claim(monkeypatch, tmp_path):
    """同一 request 内，item 不得认领另一 item basename 对应的输出。"""
    proc = _make_daemon_processor_for_ownership_test(tmp_path)
    records = []
    for index, freq in enumerate((220.0, 880.0)):
        source = str(tmp_path / "sources" / f"item_{index}.wav")
        _make_wav(source, duration=0.1, freq=freq)
        records.append(
            AudioRecord(
                record_id=f"ds__src__{index}",
                dataset_id="ds",
                source_id="src",
                audio={"path": source, "sr": 16000, "duration": 0.1, "channels": 1},
            )
        )

    def return_swapped_outputs(items, log_dir, output_root, batch_index):
        outputs = []
        for item in items:
            stem = os.path.splitext(os.path.basename(item["audio_path"]))[0]
            output = os.path.join(output_root, f"{stem}.wav")
            _make_wav(output, duration=0.1)
            outputs.append(output)
        return [
            {"audio_path": items[0]["audio_path"], "enhanced_audio_path": outputs[1]},
            {"audio_path": items[1]["audio_path"], "enhanced_audio_path": outputs[0]},
        ]

    monkeypatch.setattr(proc._daemon_pool, "enhance_parallel", return_swapped_outputs)
    final_root = tmp_path / "final"
    final_root.mkdir()

    with pytest.raises(_ApolloDaemonRetryableError, match="输出 basename"):
        proc._run_subprocess_daemon(
            records, str(tmp_path), str(final_root), batch_index=3
        )

    assert not list(final_root.iterdir())


def test_apollo_daemon_maps_success_by_record_id(monkeypatch, tmp_path):
    """成功结果必须按 record_id 归属，而不是按 source path 归属。"""
    proc = _make_daemon_processor_for_ownership_test(tmp_path)
    records = []
    for index in range(2):
        source = str(tmp_path / "sources" / f"item_{index}.wav")
        _make_wav(source, duration=0.1)
        records.append(
            AudioRecord(
                record_id=f"ds__src__{index}",
                dataset_id="ds",
                source_id="src",
                audio={"path": source, "sr": 16000, "duration": 0.1, "channels": 1},
            )
        )

    def return_owned_outputs(items, log_dir, output_root, batch_index):
        results = []
        for item in items:
            stem = os.path.splitext(os.path.basename(item["audio_path"]))[0]
            output = os.path.join(output_root, f"{stem}.wav")
            _make_wav(output, duration=0.1)
            results.append({"audio_path": item["audio_path"], "enhanced_audio_path": output})
        return results

    monkeypatch.setattr(proc._daemon_pool, "enhance_parallel", return_owned_outputs)
    final_root = tmp_path / "final"
    final_root.mkdir()

    path_map = proc._run_subprocess_daemon(
        records, str(tmp_path), str(final_root), batch_index=2
    )

    assert set(path_map) == {record.record_id for record in records}
    assert all(os.path.isfile(path_map[record.record_id]) for record in records)


def test_apollo_daemon_partial_response_reclaims_request_targets(monkeypatch, tmp_path):
    """partial response 必须失败并回收本 request 的正式/中间产物。"""
    proc = _make_daemon_processor_for_ownership_test(tmp_path)
    source_paths = [tmp_path / "sources" / f"item_{index}.wav" for index in range(2)]
    for source in source_paths:
        _make_wav(str(source), duration=0.1)
    records = [
        AudioRecord(
            record_id=f"ds__src__{index}",
            dataset_id="ds",
            source_id="src",
            audio={"path": str(source), "sr": 16000, "duration": 0.1, "channels": 1},
        )
        for index, source in enumerate(source_paths)
    ]
    final_root = tmp_path / "final"
    final_root.mkdir()
    preexisting = final_root / "keep.flac"
    preexisting.write_bytes(b"keep")

    def return_partial_output(items, log_dir, output_root, batch_index):
        response_input = items[0]["audio_path"]
        stem = os.path.splitext(os.path.basename(response_input))[0]
        output = os.path.join(output_root, f"{stem}.wav")
        _make_wav(output, duration=0.1)
        return [{"audio_path": response_input, "enhanced_audio_path": output}]

    monkeypatch.setattr(proc._daemon_pool, "enhance_parallel", return_partial_output)

    with pytest.raises(_ApolloDaemonRetryableError, match="partial response"):
        proc._run_subprocess_daemon(records, str(tmp_path), str(final_root), batch_index=6)

    assert list(final_root.iterdir()) == [preexisting]
    assert os.listdir(proc._daemon_output_root) == []


def test_apollo_daemon_rejects_malformed_result_item(monkeypatch, tmp_path):
    """非法 result item 必须走统一 retryable 失败路径。"""
    proc = _make_daemon_processor_for_ownership_test(tmp_path)
    source = tmp_path / "source.wav"
    _make_wav(str(source), duration=0.1)
    record = AudioRecord(
        record_id="ds__src__0",
        dataset_id="ds",
        source_id="src",
        audio={"path": str(source), "sr": 16000, "duration": 0.1, "channels": 1},
    )
    monkeypatch.setattr(
        proc._daemon_pool,
        "enhance_parallel",
        lambda *args, **kwargs: ["not-an-object"],
    )
    final_root = tmp_path / "final"
    final_root.mkdir()

    with pytest.raises(_ApolloDaemonRetryableError, match="非法 result item"):
        proc._run_subprocess_daemon([record], str(tmp_path), str(final_root), batch_index=7)

    assert not list(final_root.iterdir())


def test_apollo_daemon_reclaims_partial_targets_before_retry(monkeypatch, tmp_path):
    """首条发布、次条失败时只回收本 request 目标，重试可重新发布。"""
    proc = _make_daemon_processor_for_ownership_test(tmp_path)
    records = []
    for index in range(2):
        source = str(tmp_path / "sources" / f"item_{index}.wav")
        _make_wav(source, duration=0.1)
        records.append(
            AudioRecord(
                record_id=f"ds__src__{index}",
                dataset_id="ds",
                source_id="src",
                audio={"path": source, "sr": 16000, "duration": 0.1, "channels": 1},
            )
        )

    final_root = tmp_path / "final"
    final_root.mkdir()
    preexisting = final_root / "preexisting.flac"
    preexisting.write_bytes(b"keep-me")
    preexisting_inode = preexisting.stat().st_ino
    real_copy = apollo_superres.copy_or_convert_audio
    fail_second = True
    copy_calls = 0
    target_paths = {}

    def return_owned_outputs(items, log_dir, output_root, batch_index):
        results = []
        for item in items:
            stem = os.path.splitext(os.path.basename(item["audio_path"]))[0]
            output = os.path.join(output_root, f"{stem}.wav")
            _make_wav(output, duration=0.1)
            results.append({"audio_path": item["audio_path"], "enhanced_audio_path": output})
        return results

    def copy_with_injected_second_failure(src_path, dst_path, **kwargs):
        nonlocal copy_calls
        copy_calls += 1
        if fail_second and copy_calls == 2:
            raise OSError("injected second target failure")
        real_copy(src_path, dst_path, **kwargs)
        target_paths[copy_calls] = dst_path

    monkeypatch.setattr(proc._daemon_pool, "enhance_parallel", return_owned_outputs)
    monkeypatch.setattr(apollo_superres, "copy_or_convert_audio", copy_with_injected_second_failure)

    with pytest.raises(_ApolloDaemonRetryableError, match="injected second target failure"):
        proc._run_subprocess_daemon(
            records, str(tmp_path), str(final_root), batch_index=5
        )

    first_target = target_paths[1]
    assert not os.path.exists(first_target)
    assert preexisting.read_bytes() == b"keep-me"
    assert preexisting.stat().st_ino == preexisting_inode

    fail_second = False
    path_map = proc._run_subprocess_daemon(
        records, str(tmp_path), str(final_root), batch_index=5
    )

    assert set(path_map) == {record.record_id for record in records}
    assert all(os.path.isfile(path_map[record.record_id]) for record in records)
    assert preexisting.read_bytes() == b"keep-me"
    assert preexisting.stat().st_ino == preexisting_inode


def test_apollo_daemon_rejects_duplicate_source_path(monkeypatch, tmp_path):
    """同一 batch 的重复源路径必须 fail-closed，不能共享最后一个输出。"""
    proc = _make_daemon_processor_for_ownership_test(tmp_path)
    source = str(tmp_path / "source.wav")
    _make_wav(source, duration=0.1)
    records = [
        AudioRecord(
            record_id=f"ds__src__{index}",
            dataset_id="ds",
            source_id="src",
            audio={"path": source, "sr": 16000, "duration": 0.1, "channels": 1},
        )
        for index in range(2)
    ]
    daemon_called = False

    def unexpected_daemon_call(*args, **kwargs):
        nonlocal daemon_called
        daemon_called = True
        raise AssertionError("重复 source path 不应提交给 daemon")

    monkeypatch.setattr(proc._daemon_pool, "enhance_parallel", unexpected_daemon_call)
    final_root = tmp_path / "final"
    final_root.mkdir()

    with pytest.raises(_ApolloDaemonRetryableError, match="重复 source path"):
        proc._run_subprocess_daemon(
            records, str(tmp_path), str(final_root), batch_index=3
        )

    assert not daemon_called


def test_apollo_daemon_claims_each_response_input_once(monkeypatch, tmp_path):
    """同一输入首次响应即消耗 claim，后续伪造的补偿响应不得再认领。"""
    proc = _make_daemon_processor_for_ownership_test(tmp_path)
    source = str(tmp_path / "source.wav")
    _make_wav(source, duration=0.1)
    record = AudioRecord(
        record_id="ds__src__0",
        dataset_id="ds",
        source_id="src",
        audio={"path": source, "sr": 16000, "duration": 0.1, "channels": 1},
    )

    def return_duplicate_responses(items, log_dir, output_root, batch_index):
        response_input = items[0]["audio_path"]
        stem = os.path.splitext(os.path.basename(response_input))[0]
        valid_output = os.path.join(output_root, f"{stem}.wav")
        _make_wav(valid_output, duration=0.1)
        return [
            {"audio_path": response_input, "enhanced_audio_path": valid_output},
            {"audio_path": response_input, "enhanced_audio_path": valid_output},
        ]

    monkeypatch.setattr(proc._daemon_pool, "enhance_parallel", return_duplicate_responses)
    final_root = tmp_path / "final"
    final_root.mkdir()

    with pytest.raises(_ApolloDaemonRetryableError, match="重复 claim"):
        proc._run_subprocess_daemon(
            [record], str(tmp_path), str(final_root), batch_index=4
        )

    assert not list(final_root.iterdir())


def test_apollo_daemon_rejects_cross_request_output_claim(monkeypatch, tmp_path):
    """并发 request 即使 source 相同，也必须用 request UUID 隔离输入/输出 stem。"""
    proc = _make_daemon_processor_for_ownership_test(tmp_path)
    source = str(tmp_path / "source.wav")
    _make_wav(source, duration=0.1)
    records = [
        AudioRecord(
            record_id=f"ds__src__{index}",
            dataset_id="ds",
            source_id="src",
            audio={"path": source, "sr": 16000, "duration": 0.1, "channels": 1},
        )
        for index in range(2)
    ]
    calls = []
    calls_lock = threading.Lock()
    ready = threading.Barrier(2)

    def return_foreign_request_output(items, log_dir, output_root, batch_index):
        own_input = items[0]["audio_path"]
        with calls_lock:
            calls.append(own_input)
        ready.wait(timeout=5)
        with calls_lock:
            foreign_input = next(path for path in calls if path != own_input)
            foreign_stem = os.path.splitext(os.path.basename(foreign_input))[0]
            output = os.path.join(output_root, f"{foreign_stem}.wav")
            if not os.path.exists(output):
                _make_wav(output, duration=0.1)
        return [{"audio_path": own_input, "enhanced_audio_path": output}]

    monkeypatch.setattr(proc._daemon_pool, "enhance_parallel", return_foreign_request_output)
    final_roots = [tmp_path / "final_a", tmp_path / "final_b"]
    for root in final_roots:
        root.mkdir()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                proc._run_subprocess_daemon,
                [record],
                str(tmp_path),
                str(final_root),
                batch_index,
            )
            for batch_index, (record, final_root) in enumerate(
                zip(records, final_roots), start=10
            )
        ]
        results = []
        for future in futures:
            with pytest.raises(_ApolloDaemonRetryableError, match="输出 basename"):
                future.result()
            results.append(None)

    input_stems = [os.path.splitext(os.path.basename(path))[0] for path in calls]
    assert len(set(input_stems)) == 2
    assert results == [None, None]


def test_apollo_daemon_partial_response_marks_batch_failed_without_checkpoint(monkeypatch, tmp_path):
    """partial response 经 process/_flush_pending 后整批 failed 且不写 checkpoint。"""
    manifest = tmp_path / "input.jsonl"
    records = []
    for index in range(2):
        source = tmp_path / "sources" / f"item_{index}.wav"
        _make_wav(str(source), duration=0.1)
        records.append(
            AudioRecord(
                record_id=f"ds__src__{index}",
                dataset_id="ds",
                source_id="src",
                audio={"path": str(source), "sr": 16000, "duration": 0.1, "channels": 1},
            )
        )
    write_manifest(iter(records), str(manifest))
    config = {
        "mode": "subprocess_daemon",
        "daemon_script_path": __file__,
        "python_path": sys.executable,
        "checkpoint": str(tmp_path / "apollo.ckpt"),
        "config_path": str(tmp_path / "apollo.yaml"),
        "batch_size": 2,
        "checkpoint_save_interval": 1,
        "manifest_write_batch_size": 2,
    }
    for path in (config["checkpoint"], config["config_path"]):
        open(path, "w", encoding="utf-8").close()
    proc = ApolloSuperResProcessor(config)

    def return_partial_output(items, log_dir, output_root, batch_index):
        response_input = items[0]["audio_path"]
        stem = os.path.splitext(os.path.basename(response_input))[0]
        output = os.path.join(output_root, f"{stem}.wav")
        _make_wav(output, duration=0.1)
        return [{"audio_path": response_input, "enhanced_audio_path": output}]

    monkeypatch.setattr(proc._daemon_pool, "enhance_parallel", return_partial_output)
    out_dir = tmp_path / "d1"
    output_manifest = proc.process(str(manifest), str(out_dir))

    out_records = list(read_manifest(output_manifest))
    assert {record.record_id for record in out_records} == {
        record.record_id for record in records
    }
    for record in out_records:
        d1 = record.stages["D1"]
        assert d1["status"] == "failed"
        assert d1["retryable"] is True
        assert not record.rejected_reason
    checkpoint = out_dir / "checkpoint.json"
    assert not checkpoint.exists() or not (
        {record.record_id for record in records}
        & set(json.loads(checkpoint.read_text())["done"])
    )
    assert not list((out_dir / "audio").glob("*.flac"))


def test_apollo_daemon_config():
    """pipeline.yaml 的 d1 配置：mode=subprocess_daemon，num_gpus=8。"""
    with open(PIPELINE_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    d1 = cfg["stage_d"]["d1"]
    assert d1["mode"] == "subprocess_daemon"
    assert int(d1["num_gpus"]) == 8
    assert d1.get("gpu_ids", []) == []


# ---------------------------------------------------------------------------
# _ApolloDaemonPool 故障恢复逻辑（mock daemon，不启动真实进程）
# ---------------------------------------------------------------------------


def _make_mock_pool(n: int = 2) -> _ApolloDaemonPool:
    """构造不启动真实进程的 pool，daemon 槽位填 MagicMock。"""
    pool = _ApolloDaemonPool(
        python_path=sys.executable,
        daemon_script_path="dummy_daemon.py",
        config_path="dummy.yaml",
        checkpoint="dummy.ckpt",
        batch_size=1,
        target_sr=44100,
        chunk_sec=10.0,
        overlap_sec=2.0,
        gpu_ids=list(range(n)),
        num_gpus=n,
        timeout=5,
    )
    pool._daemon_procs = [MagicMock(name=f"proc{i}") for i in range(n)]
    pool._daemon_locks = [threading.Lock() for _ in range(n)]
    pool._daemon_gpu_ids = list(range(n))
    return pool


def test_restart_daemon_kill_then_wait(monkeypatch):
    """重启路径：kill 后必须 wait 回收进程，替代 daemon 换入原槽位。"""
    pool = _make_mock_pool(1)
    old_proc = pool._daemon_procs[0]
    new_proc = MagicMock(name="new_proc")
    monkeypatch.setattr(pool, "_launch_daemon", lambda gpu_id, log_dir: new_proc)

    assert pool._restart_daemon(0, old_proc) is True
    old_proc.kill.assert_called_once()
    old_proc.wait.assert_called_once_with(timeout=30)
    assert pool._daemon_procs[0] is new_proc


def test_restart_daemon_accepts_pick_token(monkeypatch):
    """Apollo 运行时仅传 _pick token 时，仍按绑定的进程身份重启。"""
    pool = _make_mock_pool(1)
    old_proc = pool._daemon_procs[0]
    daemon_idx, picked_proc, _lock = pool._pick()
    new_proc = MagicMock(name="new_proc")
    monkeypatch.setattr(pool, "_launch_daemon", lambda gpu_id, log_dir: new_proc)

    assert picked_proc is old_proc
    assert pool._restart_daemon(daemon_idx) is True
    old_proc.kill.assert_called_once()
    old_proc.wait.assert_called_once_with(timeout=30)
    assert pool._daemon_procs[0] is new_proc


def test_restart_daemon_stale_slot_recycles_redundant_proc(monkeypatch):
    """stale-index 防护：重启期间槽位被并发线程替换时，回收多余的新进程。"""
    pool = _make_mock_pool(2)
    old_proc = pool._daemon_procs[0]
    other_proc = MagicMock(name="other_proc")
    redundant_proc = MagicMock(name="redundant_proc")

    def fake_launch(gpu_id, log_dir):
        # 模拟另一线程在本线程 _launch_daemon 期间已把槽位换为 other_proc
        pool._daemon_procs[0] = other_proc
        return redundant_proc

    monkeypatch.setattr(pool, "_launch_daemon", fake_launch)

    assert pool._restart_daemon(0, old_proc) is True
    # 槽位保持并发线程换入的进程，不被覆盖
    assert pool._daemon_procs[0] is other_proc
    # 多余的替代进程被 kill + wait 回收，不泄漏占 GPU
    redundant_proc.kill.assert_called_once()
    redundant_proc.wait.assert_called_once_with(timeout=30)


def test_restart_daemon_failure_keeps_replaced_slot(monkeypatch):
    """stale-index 防护：重启失败时若槽位已被并发线程替换，不得误删新进程。"""
    pool = _make_mock_pool(2)
    old_proc = pool._daemon_procs[0]
    other_proc = MagicMock(name="other_proc")

    def fake_launch(gpu_id, log_dir):
        pool._daemon_procs[0] = other_proc
        raise RuntimeError("launch failed")

    monkeypatch.setattr(pool, "_launch_daemon", fake_launch)

    assert pool._restart_daemon(0, old_proc) is False
    assert len(pool._daemon_procs) == 2
    assert pool._daemon_procs[0] is other_proc


def test_enhance_retries_batch_once_on_daemon_failure(monkeypatch):
    """失败重试：首个 daemon 失败后重启并换 daemon 重试一次，不整批 reject。"""
    pool = _make_mock_pool(2)
    proc_a, proc_b = pool._daemon_procs
    sent_to = []

    def fake_send(proc, cmd):
        sent_to.append(proc)
        if proc is proc_a:
            raise BrokenPipeError("daemon dead")

    monkeypatch.setattr(pool, "_send", fake_send)
    monkeypatch.setattr(
        pool,
        "_recv",
        lambda proc, timeout=None: {
            "type": "result",
            "batch_id": 7,
            "results": [{"audio_path": "x", "enhanced_audio_path": "y"}],
        },
    )
    restarts = []
    monkeypatch.setattr(pool, "_restart_daemon", lambda idx: restarts.append(idx) or True)

    results = pool.enhance([{"audio_path": "x"}], batch_id=7)

    assert results == [{"audio_path": "x", "enhanced_audio_path": "y"}]
    assert restarts == [0]  # 只重启了故障的 daemon 0
    assert sent_to == [proc_a, proc_b]  # 重试打到另一个 daemon
    assert pool._consecutive_batch_failures == 0


def test_enhance_raises_exhausted_when_all_daemons_dead(monkeypatch):
    """熔断：所有 daemon 重启失败被移出轮询后，抛 _DaemonPoolExhaustedError。"""
    pool = _make_mock_pool(1)

    def fake_send(proc, cmd):
        raise BrokenPipeError("daemon dead")

    def fake_restart(idx):
        # 模拟重启失败：把 daemon 移出轮询
        del pool._daemon_procs[idx]
        del pool._daemon_locks[idx]
        del pool._daemon_gpu_ids[idx]
        return False

    monkeypatch.setattr(pool, "_send", fake_send)
    monkeypatch.setattr(pool, "_restart_daemon", fake_restart)

    with pytest.raises(_DaemonPoolExhaustedError):
        pool.enhance([{"audio_path": "x"}], batch_id=1)


def test_enhance_circuit_breaker_after_consecutive_failures(monkeypatch):
    """熔断：重试仍失败计入连续失败，达到上限后抛 _DaemonPoolExhaustedError。"""
    pool = _make_mock_pool(1)

    def fake_send(proc, cmd):
        raise BrokenPipeError("daemon dead")

    monkeypatch.setattr(pool, "_send", fake_send)
    # 重启“成功”但 daemon 仍旧不可用（每批都重新拉起又失败）
    monkeypatch.setattr(pool, "_restart_daemon", lambda idx: True)

    # 前 N-1 次：重试一次后仍失败，原样抛出业务异常
    for i in range(pool._MAX_CONSECUTIVE_BATCH_FAILURES - 1):
        with pytest.raises(BrokenPipeError):
            pool.enhance([{"audio_path": "x"}], batch_id=i)

    # 第 N 次：触发熔断
    with pytest.raises(_DaemonPoolExhaustedError):
        pool.enhance([{"audio_path": "x"}], batch_id=99)


def test_enhance_batch_id_mismatch_triggers_restart(monkeypatch):
    """batch_id 不匹配视为 daemon 故障：触发重启并换 daemon 重试。"""
    pool = _make_mock_pool(2)
    proc_a, proc_b = pool._daemon_procs

    monkeypatch.setattr(pool, "_send", lambda proc, cmd: None)

    def fake_recv(proc, timeout=None):
        # proc_a 返回错配的 batch_id；proc_b 返回正确响应
        if proc is proc_a:
            return {"type": "result", "batch_id": 999, "results": []}
        return {"type": "result", "batch_id": 3, "results": [{"audio_path": "x"}]}

    monkeypatch.setattr(pool, "_recv", fake_recv)
    restarts = []
    monkeypatch.setattr(pool, "_restart_daemon", lambda idx: restarts.append(idx) or True)

    results = pool.enhance([{"audio_path": "x"}], batch_id=3)
    assert results == [{"audio_path": "x"}]
    assert restarts == [0]


def test_flush_pending_aborts_on_pool_exhausted(monkeypatch, tmp_path):
    """_flush_pending 遇到 _DaemonPoolExhaustedError 必须上抛中止 stage。"""
    from noise_filter_pipeline.core.checkpoint import CheckpointTracker
    from noise_filter_pipeline.core.manifest_io import BatchedManifestWriter

    proc = ApolloSuperResProcessor(
        {
            "mode": "subprocess_daemon",
            "daemon_script_path": __file__,  # 仅占位，不会实际启动
            "checkpoint": "dummy.ckpt",
            "config_path": "dummy.yaml",
            "batch_size": 2,
        }
    )

    def fake_enhance_parallel(*args, **kwargs):
        raise _DaemonPoolExhaustedError("所有 Apollo daemon 均不可用")

    monkeypatch.setattr(proc._daemon_pool, "enhance_parallel", fake_enhance_parallel)

    rec = AudioRecord(
        record_id="ds__src__0",
        dataset_id="ds",
        source_id="src",
        audio={"path": "/tmp/a.wav", "sr": 16000, "duration": 1.0, "channels": 1},
    )
    writer = BatchedManifestWriter(os.path.join(str(tmp_path), "out.jsonl"), batch_size=10)
    ckpt = CheckpointTracker(checkpoint_path=None, enabled=False)

    with pytest.raises(_DaemonPoolExhaustedError):
        proc._flush_pending(
            [rec], str(tmp_path), str(tmp_path), writer, ckpt, [0]
        )


def test_apollo_rejects_removed_modes():
    """D1 只保留 dry_run / subprocess_daemon：inline/subprocess 启动时必须 ValueError。"""
    with pytest.raises(ValueError, match="mode"):
        ApolloSuperResProcessor({"mode": "inline"})
    with pytest.raises(ValueError, match="mode"):
        ApolloSuperResProcessor({"mode": "subprocess"})
