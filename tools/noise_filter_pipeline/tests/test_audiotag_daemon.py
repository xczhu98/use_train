"""
Tests for AudioTag subprocess_daemon mode (using a mock daemon).
"""
import errno
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

import third_party_scripts.icefall_audiotaq.audiotag_daemon as audiotag_daemon
import third_party_scripts.icefall_audiotaq.audiotag_result as audiotag_result
from noise_filter_pipeline.core.exceptions import InfraAbortError
from noise_filter_pipeline.core.manifest_io import AudioRecord, read_manifest, write_manifest
from noise_filter_pipeline.processors.audiotag_classifier import (
    AudioTagClassifierProcessor,
    DaemonInferenceError,
    _AudioTagDaemonPool,
)
from third_party_scripts.icefall_audiotaq.audiotag_daemon import AudioTagDaemon
from third_party_scripts.icefall_audiotaq.audiotag_result import (
    AUDIOTAG_TOP_K,
    CLASSIFIER_LABEL_COUNT,
    EVIDENCE_CONTRACT_VERSION,
    SCORE_CONTRACT,
    build_dry_run_result,
    format_audiotag_results,
    load_label_metadata,
)

_REAL_LABEL_DICT = str(
    Path(audiotag_result.__file__).resolve().parent
    / "data" / "class_labels_indices.csv"
)
_LABEL_METADATA = load_label_metadata(_REAL_LABEL_DICT)


def _make_v5_output(record_id: str = "ds__src__0", label: str = "Rain") -> dict:
    return build_dry_run_result(label, _LABEL_METADATA, record_id=record_id)

# 本文件使用 mock daemon / tensor 小样本，仅覆盖本地逻辑；真实模型测试另属 external。
pytestmark = pytest.mark.unit


def test_format_audiotag_results_emits_only_fixed_stable_top5():
    scores = [0.0] * CLASSIFIER_LABEL_COUNT
    scores[20] = 0.9
    scores[3] = 0.9
    scores[11] = 0.8

    result = format_audiotag_results(
        [scores], _LABEL_METADATA, record_ids=["record-0"]
    )[0]

    assert set(result) == {
        "record_id",
        "audioset_top5",
        "evidence_contract_version",
        "score_contract",
        "class_count",
        "label_space_sha256",
    }
    assert result["record_id"] == "record-0"
    assert len(result["audioset_top5"]) == AUDIOTAG_TOP_K == 5
    assert [item["index"] for item in result["audioset_top5"][:3]] == [3, 20, 11]
    assert all(
        set(item) == {"index", "mid", "label", "prob"}
        for item in result["audioset_top5"]
    )
    assert "audioset_scores" not in result
    assert "audioset_top10" not in result
    assert result["evidence_contract_version"] == EVIDENCE_CONTRACT_VERSION == 5
    assert result["score_contract"] == SCORE_CONTRACT
    assert result["class_count"] == CLASSIFIER_LABEL_COUNT
    assert result["label_space_sha256"] == _LABEL_METADATA.label_space_sha256


@pytest.mark.parametrize(
    "scores",
    [
        [0.0] * (CLASSIFIER_LABEL_COUNT - 1),
        [0.0] * CLASSIFIER_LABEL_COUNT + [0.0],
        [float("nan")] + [0.0] * (CLASSIFIER_LABEL_COUNT - 1),
        [1.1] + [0.0] * (CLASSIFIER_LABEL_COUNT - 1),
        [True] + [0.0] * (CLASSIFIER_LABEL_COUNT - 1),
    ],
)
def test_format_audiotag_results_rejects_invalid_fixed_scores(scores):
    with pytest.raises(ValueError):
        format_audiotag_results([scores], _LABEL_METADATA)



def test_top5_to_manifest_c2_v5_end_to_end():
    result = _make_v5_output()
    proc = AudioTagClassifierProcessor({"mode": "dry_run", "label_dict": _REAL_LABEL_DICT})
    rec = _make_record()
    rec.labels = {}

    proc._apply_output_to_record(rec, result)

    assert rec.labels == {"audioset_top5": result["audioset_top5"]}
    assert rec.labels["audioset_top5"][0]["label"] == "Rain"
    assert rec.is_stage_passed("C2")
    assert rec.stages["C2"]["evidence_contract_version"] == 5
    assert rec.stages["C2"]["class_count"] == CLASSIFIER_LABEL_COUNT



def test_apply_output_validates_complete_v5_contract_before_mutation():
    proc = AudioTagClassifierProcessor({"mode": "dry_run", "label_dict": _REAL_LABEL_DICT})
    valid = _make_v5_output()

    corruptions = []
    missing_top5 = dict(valid)
    missing_top5.pop("audioset_top5")
    corruptions.append(missing_top5)

    short_top5 = json.loads(json.dumps(valid))
    short_top5["audioset_top5"].pop()
    corruptions.append(short_top5)

    wrong_identity = json.loads(json.dumps(valid))
    wrong_identity["audioset_top5"][0]["mid"] = "/m/not-canonical"
    corruptions.append(wrong_identity)

    wrong_order = json.loads(json.dumps(valid))
    wrong_order["audioset_top5"][0], wrong_order["audioset_top5"][1] = (
        wrong_order["audioset_top5"][1],
        wrong_order["audioset_top5"][0],
    )
    corruptions.append(wrong_order)
    corruptions.append(dict(valid, record_id="other__id__9"))
    corruptions.append(dict(valid, evidence_contract_version=4))
    corruptions.append(dict(valid, audioset_scores=[0.0] * CLASSIFIER_LABEL_COUNT))
    corruptions.append(dict(valid, audioset_top10=valid["audioset_top5"]))

    for output in corruptions:
        rec = _make_record()
        rec.labels = {}
        with pytest.raises(ValueError):
            proc._apply_output_to_record(rec, output)
        assert rec.labels == {}
        assert not rec.is_stage_passed("C2")


def test_c2_rejects_removed_modes():
    """C2 只保留 dry_run / subprocess_daemon：inline/subprocess 启动时必须 ValueError。"""
    with pytest.raises(ValueError, match="mode"):
        AudioTagClassifierProcessor({"mode": "inline"})
    with pytest.raises(ValueError, match="mode"):
        AudioTagClassifierProcessor({"mode": "subprocess"})


def test_c2_rejects_policy_configuration():
    with pytest.raises(ValueError, match="policy_path"):
        AudioTagClassifierProcessor({"mode": "dry_run", "policy_path": "unused.json"})


def test_c2_daemon_command_has_no_policy_or_topk_options():
    proc = AudioTagClassifierProcessor({
        "mode": "subprocess_daemon",
        "daemon_script_path": "dummy_daemon.py",
    })

    cmd = proc._daemon_pool._build_cmd(0)

    assert "--policy-path" not in cmd
    assert "--topk" not in cmd
    audiotag_daemon.get_parser().parse_args(cmd)


def test_c2_daemon_command_carries_performance_options_to_cli_parser():
    proc = AudioTagClassifierProcessor({
        "mode": "subprocess_daemon",
        "daemon_script_path": "dummy_daemon.py",
        "checkpoint": "dummy.pt",
        "label_dict": "labels.csv",
        "cpu_threads": 2,
        "interop_threads": 3,
        "audio_workers": 6,
        "timing_log_interval": 9,
    })

    args = audiotag_daemon.get_parser().parse_args(proc._daemon_pool._build_cmd(0))

    assert args.cpu_threads == 2
    assert args.interop_threads == 3
    assert args.audio_workers == 6
    assert args.timing_log_interval == 9


@pytest.mark.parametrize("name", [
    "cpu_threads", "interop_threads", "audio_workers", "timing_log_interval",
    "request_batches",
])
@pytest.mark.parametrize("bad_value", [True, 0, -1, 1.0, "1"])
def test_c2_performance_options_require_positive_integer(name, bad_value):
    with pytest.raises(ValueError, match=name):
        AudioTagClassifierProcessor({"mode": "dry_run", name: bad_value})


@pytest.mark.parametrize("option", [
    "--cpu-threads", "--interop-threads", "--audio-workers", "--timing-log-interval",
])
@pytest.mark.parametrize("bad_value", ["0", "-1"])
def test_daemon_cli_performance_options_require_positive_integer(option, bad_value):
    with pytest.raises(SystemExit):
        audiotag_daemon.get_parser().parse_args([
            "--checkpoint", "dummy.pt",
            "--label-dict", "labels.csv",
            option, bad_value,
        ])


def _make_decode_daemon(audio_workers=2):
    daemon = object.__new__(AudioTagDaemon)
    daemon.sample_rate = 16000
    daemon.audio_workers = audio_workers
    daemon._audio_executor = MagicMock()
    daemon._audio_executor.map.side_effect = lambda fn, items: [fn(item) for item in items]
    return daemon


def _make_readable_audio_paths(tmp_path, names):
    paths = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"x")
        paths.append(str(path))
    return paths


def test_parallel_decode_preserves_input_order(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    daemon = _make_decode_daemon(audio_workers=2)
    paths = _make_readable_audio_paths(tmp_path, ["a", "b", "c", "d"])
    sample_rates = dict(zip(paths, [48000, 16000, 48000, 32000]))
    monkeypatch.setattr(audiotag_daemon, "_soundfile", None)  # 强制走 torchaudio 路径
    monkeypatch.setattr(
        audiotag_daemon.torchaudio, "load",
        lambda path: (
            torch.tensor([[float(ord(Path(path).name))]]), sample_rates[path]
        ),
    )

    decoded, decode_seconds = daemon._decode_waves(paths)

    assert [wave.item() for wave, _ in decoded] == [97.0, 98.0, 99.0, 100.0]
    assert [source_rate for _, source_rate in decoded] == [48000, 16000, 48000, 32000]
    assert decode_seconds >= 0.0


def test_decode_wave_prefers_soundfile_and_takes_first_channel(monkeypatch, tmp_path):
    """默认走 soundfile（GIL 释放、可并行解码），取第一声道，不触碰 torchaudio。"""
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"x")  # 只需通过存储探测，soundfile 被 mock
    fake_sf = MagicMock()
    fake_sf.read.return_value = (
        np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32), 48000,
    )
    monkeypatch.setattr(audiotag_daemon, "_soundfile", fake_sf)
    load = MagicMock()
    monkeypatch.setattr(audiotag_daemon.torchaudio, "load", load)

    wave, sample_rate = AudioTagDaemon._decode_wave(str(audio_path))

    assert sample_rate == 48000
    assert wave.tolist() == pytest.approx([0.1, 0.2])
    fake_sf.read.assert_called_once_with(
        str(audio_path), dtype="float32", always_2d=True
    )
    load.assert_not_called()


def test_decode_wave_falls_back_to_torchaudio_on_soundfile_runtime_error(
    monkeypatch, tmp_path
):
    """soundfile 不支持的编码（RuntimeError）落回 torchaudio 解码。"""
    torch = pytest.importorskip("torch")
    audio_path = tmp_path / "a.mp3"
    audio_path.write_bytes(b"x")
    fake_sf = MagicMock()
    fake_sf.read.side_effect = RuntimeError("unsupported format")
    monkeypatch.setattr(audiotag_daemon, "_soundfile", fake_sf)
    monkeypatch.setattr(
        audiotag_daemon.torchaudio, "load",
        lambda path: (torch.tensor([[0.3, 0.4]]), 8000),
    )

    wave, sample_rate = AudioTagDaemon._decode_wave(str(audio_path))

    assert sample_rate == 8000
    assert wave.tolist() == pytest.approx([0.3, 0.4])


def _make_device_resample_daemon():
    torch = pytest.importorskip("torch")
    daemon = object.__new__(AudioTagDaemon)
    daemon.sample_rate = 16000
    daemon.device = torch.device("cpu")
    daemon._resamplers = {}
    return daemon


def test_resample_on_device_caches_instance_per_source_rate(monkeypatch):
    torch = pytest.importorskip("torch")
    daemon = _make_device_resample_daemon()
    resample = MagicMock(side_effect=lambda wave: wave + 1)
    resample.to.return_value = resample
    resample_factory = MagicMock(return_value=resample)
    monkeypatch.setattr(
        audiotag_daemon.torchaudio.transforms, "Resample", resample_factory,
    )
    wave = torch.zeros(4)

    assert daemon._resample_on_device(wave, 48000).tolist() == [1.0] * 4
    daemon._resample_on_device(wave, 48000)
    daemon._resample_on_device(wave, 44100)

    # 每个 source rate 只构造一个实例，并搬到推理设备上
    assert resample_factory.call_count == 2
    resample_factory.assert_any_call(48000, 16000)
    resample_factory.assert_any_call(44100, 16000)
    assert resample.to.call_args_list[0].args == (daemon.device,)
    assert resample.call_count == 3


def test_resample_on_device_skips_16k_input(monkeypatch):
    torch = pytest.importorskip("torch")
    daemon = _make_device_resample_daemon()
    resample_factory = MagicMock()
    monkeypatch.setattr(
        audiotag_daemon.torchaudio.transforms, "Resample", resample_factory,
    )
    wave = torch.zeros(4)

    assert daemon._resample_on_device(wave, 16000) is wave
    resample_factory.assert_not_called()


def test_daemon_close_shuts_down_audio_executor():
    daemon = object.__new__(AudioTagDaemon)
    executor = MagicMock()
    prefetch_executor = MagicMock()
    daemon._audio_executor = executor
    daemon._prefetch_executor = prefetch_executor
    daemon._timing = {
        "decode": 0.1,
        "resample": 0.2,
        "h2d_fbank": 0.3,
        "model": 0.4,
        "serialize": 0.1,
        "total": 1.1,
    }
    daemon._timing_batches = 1
    daemon._timing_files = 2

    daemon.close()
    daemon.close()

    executor.shutdown.assert_called_once_with(wait=True)
    prefetch_executor.shutdown.assert_called_once_with(wait=True)
    assert daemon._audio_executor is None
    assert daemon._prefetch_executor is None
    assert all(value == 0.0 for value in daemon._timing.values())
    assert daemon._timing_batches == 0
    assert daemon._timing_files == 0


def test_format_top5_matches_legacy_full_score_sort():
    """_format_top5 必须与 format_audiotag_results 的全 527 排序输出逐字节等价。"""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    probs_np = rng.random((7, CLASSIFIER_LABEL_COUNT)).astype(np.float32)
    # 制造并列值，验证 prob 降序 + index 升序的排序契约等价
    probs_np[0, 3] = probs_np[0, 20] = np.float32(0.9)
    probs_np[1, 5] = probs_np[1, 7] = probs_np[1, 9] = np.float32(0.5)
    record_ids = [f"r{i}" for i in range(7)]
    daemon = object.__new__(AudioTagDaemon)
    daemon.label_metadata = _LABEL_METADATA

    got = daemon._format_top5(torch.from_numpy(probs_np), record_ids)
    expected = format_audiotag_results(
        probs_np.tolist(), _LABEL_METADATA, record_ids=record_ids
    )

    assert got == expected


def test_format_top5_rejects_non_finite_probs():
    torch = pytest.importorskip("torch")
    daemon = object.__new__(AudioTagDaemon)
    daemon.label_metadata = _LABEL_METADATA
    probs = torch.zeros(2, CLASSIFIER_LABEL_COUNT)
    probs[0, 3] = float("nan")

    with pytest.raises(RuntimeError, match="NaN"):
        daemon._format_top5(probs, ["r0", "r1"])


def test_infer_prefetches_next_batch_decode_during_model(monkeypatch):
    """双缓冲：模型处理 batch N 时，batch N+1 的解码必须已并行启动。"""
    torch = pytest.importorskip("torch")
    daemon = object.__new__(AudioTagDaemon)
    daemon.sample_rate = 16000
    daemon.batch_size = 2
    daemon.device = torch.device("cpu")
    daemon.label_metadata = _LABEL_METADATA
    daemon._expected_class_count = CLASSIFIER_LABEL_COUNT
    daemon._resamplers = {}
    daemon.timing_log_interval = 1_000_000  # 测试期间不落 timing 日志
    daemon._timing = {
        key: 0.0
        for key in ("decode", "resample", "h2d_fbank", "model", "serialize", "total")
    }
    daemon._timing_batches = 0
    daemon._timing_files = 0
    daemon._audio_executor = ThreadPoolExecutor(max_workers=2)
    daemon._prefetch_executor = ThreadPoolExecutor(max_workers=1)

    decode_calls = []
    decode2_started = threading.Event()
    model1_started = threading.Event()

    def fake_decode(paths):
        decode_calls.append(list(paths))
        if len(decode_calls) == 2:
            decode2_started.set()
            assert model1_started.wait(timeout=2), (
                "batch 2 解码启动后，batch 1 模型未并行执行"
            )
        return ([(torch.zeros(160), 16000) for _ in paths], 0.0)

    class _Model:
        calls = 0

        def forward_encoder(self, features, lengths):
            type(self).calls += 1
            if type(self).calls == 1:
                model1_started.set()
                assert decode2_started.wait(timeout=2), (
                    "模型处理 batch 1 时，batch 2 的解码未被预取"
                )
            return torch.zeros(features.size(0), 10, 16), lengths

        def forward_audio_tagging(self, encoder_out, encoder_out_lens):
            return torch.zeros(encoder_out.size(0), CLASSIFIER_LABEL_COUNT)

    monkeypatch.setattr(daemon, "_decode_waves", fake_decode)
    daemon.fbank = lambda waves: [torch.zeros(10, 80) for _ in waves]
    daemon.model = _Model()

    try:
        items = [
            {"record_id": f"r{i}", "audio_path": f"/tmp/x{i}.wav"}
            for i in range(4)
        ]
        results = daemon.infer(items)
    finally:
        daemon.close()

    assert [result["record_id"] for result in results] == [f"r{i}" for i in range(4)]
    assert decode_calls == [["/tmp/x0.wav", "/tmp/x1.wav"], ["/tmp/x2.wav", "/tmp/x3.wav"]]


@pytest.mark.parametrize("label", ["Rain", "Singing", "Speech", "Cough", "Chatter"])
def test_dry_run_outputs_fixed_v5_contract(label):
    proc = AudioTagClassifierProcessor({
        "mode": "dry_run",
        "dry_run_labels": [label],
        "label_dict": _REAL_LABEL_DICT,
    })

    result = proc._run_dry_run([_make_record()])[0]

    top_index = _LABEL_METADATA.mid_to_index[_LABEL_METADATA.label_to_mid[label]]
    assert len(result["audioset_top5"]) == AUDIOTAG_TOP_K
    assert result["audioset_top5"][0] == {
        "index": top_index,
        "mid": _LABEL_METADATA.index_to_mid[top_index],
        "label": label,
        "prob": 1.0,
    }
    assert "audioset_scores" not in result
    assert "audioset_top10" not in result
    assert result["record_id"] == "ds__src__0"


def _make_wav(path, duration=1.0, sr=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wav = np.random.randn(int(duration * sr)) * 0.01
    sf.write(path, wav, sr)


def _write_mock_daemon(script_path: str, counter_file: str | None = None) -> None:
    """写一个 mock daemon，遵循 audiotag_daemon.py 的 stdin/stdout 协议。"""
    counter_line = ""
    if counter_file:
        counter_line = f'''
        with open({counter_file!r}, "a") as _cf:
            _cf.write("infer\\n")'''
    result_template = _make_v5_output(record_id="placeholder")

    code = f'''
import json
import sys

def reply(resp):
    sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    cmd = json.loads(line)
    c = cmd.get("cmd")
    if c == "ping":
        reply({{"status": "ok", "message": "pong"}})
    elif c == "exit":
        reply({{"status": "ok", "message": "bye"}})
        break
    elif c == "infer":
        input_path = cmd["input_path"]
        output_path = cmd["output_path"]
        with open(input_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        assert cmd["record_ids"] == [item["record_id"] for item in items]
        result_template = {result_template!r}
        results = [dict(result_template, record_id=item["record_id"]) for item in items]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        {counter_line}
        reply({{"status": "ok"}})
    else:
        reply({{"status": "error", "message": f"unknown cmd {{c}}"}})
'''
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)


def test_audiotag_subprocess_daemon():
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_daemon = os.path.join(tmpdir, "mock_daemon.py")
        _write_mock_daemon(mock_daemon)

        audio_dir = os.path.join(tmpdir, "audio")
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = []
        for i in range(5):
            wav_path = os.path.join(audio_dir, f"a{i}.wav")
            _make_wav(wav_path)
            recs.append(
                AudioRecord(
                    record_id=f"ds__src__{i}",
                    dataset_id="ds",
                    source_id="src",
                    audio={"path": wav_path, "sr": 16000, "duration": 1.0, "channels": 1},
                    labels={"original": "noise"},
                )
            )
        write_manifest(iter(recs), manifest)

        proc = AudioTagClassifierProcessor(
            {
                "mode": "subprocess_daemon",
                "daemon_script_path": mock_daemon,
                "python_path": sys.executable,
                "checkpoint": os.path.join(tmpdir, "dummy.pt"),
                "label_dict": _REAL_LABEL_DICT,
                "batch_size": 2,
                "checkpoint_save_interval": 3,
                "manifest_write_batch_size": 3,
            }
        )
        # mock daemon 不读 checkpoint，但 processor 会校验文件存在。
        with open(proc.checkpoint, "w", encoding="utf-8") as f:
            f.write("")

        output_manifest = proc.process(manifest, os.path.join(tmpdir, "out"))

        out_recs = list(read_manifest(output_manifest))
        assert len(out_recs) == 5
        for rec in out_recs:
            assert rec.is_stage_passed(proc.stage_name)
            assert set(rec.labels) == {"original", "audioset_top5"}
            assert len(rec.labels["audioset_top5"]) == AUDIOTAG_TOP_K
            assert rec.labels["audioset_top5"][0]["label"] == "Rain"
            assert rec.stages["C2"]["evidence_contract_version"] == 5


def test_audiotag_subprocess_daemon_multiple_batches():
    """验证 daemon 只被启动一次，多个 batch 复用同一进程。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_daemon = os.path.join(tmpdir, "mock_daemon.py")
        counter_file = os.path.join(tmpdir, "counter.txt")
        _write_mock_daemon(mock_daemon, counter_file=counter_file)

        audio_dir = os.path.join(tmpdir, "audio")
        manifest = os.path.join(tmpdir, "input.jsonl")
        recs = []
        for i in range(7):
            wav_path = os.path.join(audio_dir, f"a{i}.wav")
            _make_wav(wav_path)
            recs.append(
                AudioRecord(
                    record_id=f"ds__src__{i}",
                    dataset_id="ds",
                    source_id="src",
                    audio={"path": wav_path, "sr": 16000, "duration": 1.0, "channels": 1},
                    labels={"original": "noise"},
                )
            )
        write_manifest(iter(recs), manifest)

        proc = AudioTagClassifierProcessor(
            {
                "mode": "subprocess_daemon",
                "daemon_script_path": mock_daemon,
                "python_path": sys.executable,
                "checkpoint": os.path.join(tmpdir, "dummy.pt"),
                "label_dict": _REAL_LABEL_DICT,
                "batch_size": 3,
                "checkpoint_save_interval": 10,
                "manifest_write_batch_size": 10,
            }
        )
        with open(proc.checkpoint, "w", encoding="utf-8") as f:
            f.write("")

        proc.process(manifest, os.path.join(tmpdir, "out"))

        # batch_size=3, 7 条记录 -> 3 个 batch
        with open(counter_file, "r", encoding="utf-8") as f:
            infer_count = sum(1 for line in f if line.strip() == "infer")
        assert infer_count == 3


# ---------------------------------------------------------------------------
# _AudioTagDaemonPool 故障恢复逻辑（mock 槽位，不启动真实进程）
# 与 test_apollo_daemon.py 的 Apollo 侧用例对等：历史上 AudioTag 池缺少
# kill+wait 与槽位身份校验，且无任何测试兜底，导致并发误删健康 daemon。
# ---------------------------------------------------------------------------


def _make_mock_pool(n: int = 2) -> _AudioTagDaemonPool:
    """构造不启动真实进程的 pool，daemon 槽位填 MagicMock。"""
    pool = _AudioTagDaemonPool(
        python_path=sys.executable,
        daemon_script_path="dummy_daemon.py",
        checkpoint="dummy.pt",
        label_dict="dummy.csv",
        sample_rate=16000,
        batch_size=1,
        cpu_threads=1,
        interop_threads=1,
        audio_workers=4,
        timing_log_interval=20,
        gpu_ids=list(range(n)),
        num_gpus=n,
        timeout=5,
        stage_name="C2",
    )
    pool._daemon_procs = [MagicMock(name=f"proc{i}") for i in range(n)]
    pool._daemon_locks = [threading.Lock() for _ in range(n)]
    pool._daemon_gpu_ids = list(range(n))
    return pool


def _make_record() -> AudioRecord:
    return AudioRecord(
        record_id="ds__src__0",
        dataset_id="ds",
        source_id="src",
        audio={"path": "/tmp/a.wav", "sr": 16000, "duration": 1.0, "channels": 1},
    )


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


def test_restart_daemon_relocates_high_slot_after_low_slot_removed(monkeypatch):
    """低槽删除后按身份重定位高槽，不得误杀漂移到旧下标的健康进程。"""
    pool = _make_mock_pool(3)
    low_proc, healthy_proc, failed_proc = pool._daemon_procs
    replacement_proc = MagicMock(name="replacement_proc")
    del pool._daemon_procs[0]
    del pool._daemon_locks[0]
    del pool._daemon_gpu_ids[0]
    monkeypatch.setattr(
        pool, "_launch_daemon", lambda gpu_id, log_dir: replacement_proc
    )

    assert pool._restart_daemon(2, failed_proc) is True

    low_proc.kill.assert_not_called()
    healthy_proc.kill.assert_not_called()
    failed_proc.kill.assert_called_once()
    assert pool._daemon_procs == [healthy_proc, replacement_proc]
    assert pool._daemon_gpu_ids == [1, 2]


def test_restart_daemon_stale_slot_recycles_redundant_proc(monkeypatch):
    """并发槽位已被处理时，回收本线程拉起的多余替代进程。"""
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
    """重启失败时若槽位已被并发线程替换，不得误删新进程。"""
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


def test_infer_retries_once_on_daemon_failure(monkeypatch, tmp_path):
    """失败重试：首个 daemon 失败后重启并换 daemon 重试一次，不整批 reject。"""
    pool = _make_mock_pool(2)
    pool._daemon_started = True  # 跳过 start，不启动真实进程
    proc_a, proc_b = pool._daemon_procs
    sent_to = []
    expected_output = _make_v5_output()

    def fake_send(proc, cmd):
        sent_to.append(proc)
        if proc is proc_a:
            raise BrokenPipeError("daemon dead")
        # proc_b：模拟 daemon 写出 v5 结果文件。
        with open(cmd["output_path"], "w", encoding="utf-8") as f:
            json.dump([expected_output], f)

    monkeypatch.setattr(pool, "_send", fake_send)
    monkeypatch.setattr(pool, "_recv", lambda proc, timeout=None: {"status": "ok"})
    restarts = []
    monkeypatch.setattr(
        pool,
        "_restart_daemon",
        lambda idx, expected_proc: restarts.append((idx, expected_proc)) or True,
    )

    results = pool.infer([_make_record()], str(tmp_path), 0)

    assert results == [expected_output]
    assert restarts == [(0, proc_a)]  # 只重启故障 daemon，并传递稳定进程身份
    assert sent_to == [proc_a, proc_b]  # 重试打到另一个 daemon


def test_infer_raises_infra_abort_when_all_daemons_dead(monkeypatch, tmp_path):
    """所有 daemon 重启失败被移出轮询后，抛 InfraAbortError 中止 stage。"""
    pool = _make_mock_pool(1)
    pool._daemon_started = True

    def fake_send(proc, cmd):
        raise BrokenPipeError("daemon dead")

    def fake_restart(idx, expected_proc):
        # 模拟重启失败：按调用方传递的故障进程身份移出轮询
        assert pool._daemon_procs[idx] is expected_proc
        del pool._daemon_procs[idx]
        del pool._daemon_locks[idx]
        del pool._daemon_gpu_ids[idx]
        return False

    monkeypatch.setattr(pool, "_send", fake_send)
    monkeypatch.setattr(pool, "_restart_daemon", fake_restart)

    with pytest.raises(InfraAbortError):
        pool.infer([_make_record()], str(tmp_path), 0)


@pytest.mark.parametrize(
    ("response", "expected_kind", "expected_code"),
    [
        ({"status": "error", "message": "legacy"}, "internal", "invalid_error_response"),
        ({"status": "error", "error_kind": "content", "error_code": "bad_audio",
          "message": "decode failed", "record_ids": ["ds__src__0"]},
         "content", "bad_audio"),
        ({"status": "error", "error_kind": "infra", "error_code": "disk_io",
          "message": "write failed", "record_ids": ["ds__src__0"]},
         "infra", "disk_io"),
        ({"status": "error", "error_kind": "internal", "error_code": "model_bug",
          "message": "unexpected", "record_ids": ["ds__src__0"]},
         "internal", "model_bug"),
    ],
)
def test_infer_error_response_is_structured_and_never_restarts(
    monkeypatch, tmp_path, response, expected_kind, expected_code
):
    """内容、基础设施、内部及 legacy 错误均结构化上抛且不重启健康进程。"""
    pool = _make_mock_pool(2)
    pool._daemon_started = True
    monkeypatch.setattr(pool, "_send", lambda proc, cmd: None)
    monkeypatch.setattr(pool, "_recv", lambda proc, timeout=None: response)
    restarts = []
    monkeypatch.setattr(pool, "_restart_daemon", lambda idx: restarts.append(idx) or True)

    with pytest.raises(DaemonInferenceError) as raised:
        pool.infer([_make_record()], str(tmp_path), 0)

    assert (raised.value.error_kind, raised.value.error_code) == (
        expected_kind, expected_code,
    )
    assert raised.value.record_ids == ["ds__src__0"]
    assert restarts == []


# ---------------------------------------------------------------------------
# BaseDaemonPool._recv：带剩余超时预算的行读取（真实子进程，不经过 buffered readline）
# ---------------------------------------------------------------------------


@pytest.mark.subprocess
def test_recv_times_out_on_hanging_half_line():
    """daemon 半行挂起：_recv 必须在超时预算内报错，而非被 readline 永久阻塞。"""
    pool = _make_mock_pool(1)
    proc = subprocess.Popen(
        [
            sys.executable, "-c",
            "import sys, time; sys.stdout.write('{\"status\":'); sys.stdout.flush(); time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        t0 = time.monotonic()
        with pytest.raises(RuntimeError, match="响应超时"):
            pool._recv(proc, timeout=1)
        # 必须在超时预算附近报错，而不是等子进程退出（30s）
        assert time.monotonic() - t0 < 10
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.subprocess
def test_recv_assembles_split_line():
    """daemon 分段写出一行：_recv 按换行符拼齐后正常解析。"""
    pool = _make_mock_pool(1)
    proc = subprocess.Popen(
        [
            sys.executable, "-c",
            "import sys, time; sys.stdout.write('{\"status\":'); sys.stdout.flush(); "
            "time.sleep(0.3); sys.stdout.write(' \"ok\"}\\n'); sys.stdout.flush()",
        ],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        assert pool._recv(proc, timeout=5) == {"status": "ok"}
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.subprocess
def test_recv_eof_raises_unexpected_eof():
    """daemon 无输出直接退出：_recv 报'意外结束输出'而非误报超时。"""
    pool = _make_mock_pool(1)
    proc = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        with pytest.raises(RuntimeError, match="意外结束输出"):
            pool._recv(proc, timeout=5)
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.parametrize("bad_value", [True, 0, 4, 6, 10, 527, 1.0, "5"])
def test_topk_keep_k_rejects_non_five_configuration(bad_value):
    with pytest.raises(ValueError, match="topk_keep_k"):
        AudioTagClassifierProcessor({
            "mode": "dry_run",
            "label_dict": _REAL_LABEL_DICT,
            "topk_keep_k": bad_value,
        })


def test_dry_run_requires_label_dict_and_outputs_fixed_top5(tmp_path):
    missing = AudioTagClassifierProcessor(
        {"mode": "dry_run", "label_dict": str(tmp_path / "missing.csv")}
    )
    with pytest.raises(FileNotFoundError, match="label dict"):
        missing._validate_model_paths()

    proc = AudioTagClassifierProcessor({
        "mode": "dry_run",
        "label_dict": _REAL_LABEL_DICT,
    })
    result = proc._run_dry_run([_make_record()])[0]
    assert result["record_id"] == "ds__src__0"
    assert len(result["audioset_top5"]) == AUDIOTAG_TOP_K


def _make_records(count):
    records = [_make_record() for _ in range(count)]
    for index, record in enumerate(records):
        record.record_id = f"ds__src__{index}"
    return records


def test_infer_command_carries_complete_ordered_record_ids(monkeypatch, tmp_path):
    pool = _make_mock_pool(1)
    pool._daemon_started = True
    records = _make_records(3)
    sent = {}

    def fake_send(proc, cmd):
        sent.update(cmd)
        with open(cmd["output_path"], "w", encoding="utf-8") as f:
            json.dump([], f)

    monkeypatch.setattr(pool, "_send", fake_send)
    monkeypatch.setattr(pool, "_recv", lambda proc, timeout=None: {"status": "ok"})

    assert pool.infer(records, str(tmp_path), 0) == []
    assert sent["record_ids"] == [record.record_id for record in records]


def test_request_file_missing_preserves_command_record_ids(monkeypatch, tmp_path):
    pool = _make_mock_pool(1)
    pool._daemon_started = True
    daemon = object.__new__(AudioTagDaemon)
    response = {}

    def fake_send(proc, cmd):
        missing_cmd = dict(cmd, input_path=f"{cmd['input_path']}.missing")
        response.update(daemon._handle_infer(missing_cmd))

    monkeypatch.setattr(pool, "_send", fake_send)
    monkeypatch.setattr(pool, "_recv", lambda proc, timeout=None: response)

    with pytest.raises(DaemonInferenceError) as raised:
        pool.infer(_make_records(2), str(tmp_path), 0)

    assert (raised.value.error_kind, raised.value.error_code) == (
        "infra", "request_io_failed",
    )
    assert raised.value.record_ids == ["ds__src__0", "ds__src__1"]


@pytest.mark.parametrize("command_ids", [
    None,
    "r0",
    [],
    ["r0", ""],
    ["r0", "r0"],
])
def test_daemon_rejects_invalid_command_record_ids(tmp_path, command_ids):
    daemon = object.__new__(AudioTagDaemon)
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps([{"record_id": "r0", "audio_path": "a.wav"}]),
        encoding="utf-8",
    )

    response = daemon._handle_infer({
        "input_path": str(input_path),
        "output_path": str(tmp_path / "output.json"),
        "record_ids": command_ids,
    })

    assert (response["error_kind"], response["error_code"]) == (
        "internal", "protocol_invalid",
    )
    assert response["record_ids"] == []


@pytest.mark.parametrize("command_ids", [
    ["r1", "r0"],
    ["r0"],
    ["r0", "other"],
])
def test_daemon_rejects_command_manifest_id_mismatch(
    monkeypatch, tmp_path, command_ids
):
    daemon = object.__new__(AudioTagDaemon)
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps([
            {"record_id": "r0", "audio_path": "a.wav"},
            {"record_id": "r1", "audio_path": "b.wav"},
        ]),
        encoding="utf-8",
    )
    infer = MagicMock()
    monkeypatch.setattr(daemon, "infer", infer)

    response = daemon._handle_infer({
        "input_path": str(input_path),
        "output_path": str(tmp_path / "output.json"),
        "record_ids": command_ids,
    })

    assert (response["error_kind"], response["error_code"]) == (
        "internal", "protocol_invalid",
    )
    assert response["record_ids"] == command_ids
    infer.assert_not_called()


def _handle_single_decode(monkeypatch, tmp_path, audio_path):
    daemon = object.__new__(AudioTagDaemon)
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps([{"record_id": "r0", "audio_path": str(audio_path)}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        daemon, "infer", lambda items: [daemon._decode_wave(items[0]["audio_path"])]
    )
    return daemon._handle_infer({
        "input_path": str(input_path),
        "output_path": str(tmp_path / "output.json"),
        "record_ids": ["r0"],
    })


@pytest.mark.parametrize(
    "storage_failure",
    [None, PermissionError("permission denied"), OSError(errno.EIO, "I/O error")],
    ids=["missing", "permission", "eio"],
)
def test_daemon_probes_unavailable_storage_before_torchaudio_load(
    monkeypatch, tmp_path, storage_failure
):
    audio_path = tmp_path / "unavailable.wav"
    if storage_failure is not None:
        real_open = open

        def fail_audio_probe(path, mode="r", *args, **kwargs):
            if os.fspath(path) == str(audio_path) and mode == "rb":
                raise storage_failure
            return real_open(path, mode, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fail_audio_probe)
    load = MagicMock(side_effect=RuntimeError("backend failed to open audio"))
    monkeypatch.setattr(audiotag_daemon.torchaudio, "load", load)

    response = _handle_single_decode(monkeypatch, tmp_path, audio_path)

    assert response["status"] == "error"
    assert (response["error_kind"], response["error_code"]) == (
        "infra", "audio_storage_unavailable",
    )
    assert response["record_ids"] == ["r0"]
    assert response["message"]
    load.assert_not_called()


def test_daemon_classifies_backend_runtime_error_as_content_after_probe(
    monkeypatch, tmp_path
):
    audio_path = tmp_path / "malformed.wav"
    audio_path.write_bytes(b"not-audio")
    load = MagicMock(side_effect=RuntimeError("invalid audio stream"))
    monkeypatch.setattr(audiotag_daemon.torchaudio, "load", load)

    response = _handle_single_decode(monkeypatch, tmp_path, audio_path)

    assert response["status"] == "error"
    assert (response["error_kind"], response["error_code"]) == (
        "content", "audio_decode_failed",
    )
    assert response["record_ids"] == ["r0"]
    load.assert_called_once_with(str(audio_path))


def test_daemon_classifies_empty_audio_as_content_without_backend(
    monkeypatch, tmp_path
):
    audio_path = tmp_path / "empty.wav"
    audio_path.write_bytes(b"")
    load = MagicMock()
    monkeypatch.setattr(audiotag_daemon.torchaudio, "load", load)

    response = _handle_single_decode(monkeypatch, tmp_path, audio_path)

    assert response["status"] == "error"
    assert (response["error_kind"], response["error_code"]) == (
        "content", "audio_decode_failed",
    )
    assert response["record_ids"] == ["r0"]
    load.assert_not_called()


@pytest.mark.parametrize(("kind", "code", "failure"), [
    ("internal", "unexpected_exception", RuntimeError("model bug")),
    ("infra", "request_io_failed", None),
])
def test_daemon_handle_infer_returns_structured_error(monkeypatch, tmp_path, kind, code, failure):
    daemon = object.__new__(AudioTagDaemon)
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps([{"record_id": "r0", "audio_path": "a.wav"}]),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.json"
    if failure is None:
        output_path = tmp_path  # 目录不可作为结果文件，覆盖 infra 写失败路径
        monkeypatch.setattr(daemon, "infer", lambda items: [])
    else:
        monkeypatch.setattr(daemon, "infer", MagicMock(side_effect=failure))
    response = daemon._handle_infer({
        "input_path": str(input_path),
        "output_path": str(output_path),
        "record_ids": ["r0"],
    })
    assert response["status"] == "error"
    assert (response["error_kind"], response["error_code"]) == (kind, code)
    assert response["record_ids"] == ["r0"] and response["message"]


def test_batch_identity_error_never_commits_partial_success(monkeypatch, tmp_path):
    proc = AudioTagClassifierProcessor(
        {"mode": "dry_run", "label_dict": _REAL_LABEL_DICT}
    )
    pending = _make_records(2)
    outputs = proc._run_dry_run(pending)
    outputs[1].pop("record_id")
    monkeypatch.setattr(proc, "_run_dry_run", lambda records: outputs)
    writer, checkpoint = MagicMock(), MagicMock()
    with pytest.raises(DaemonInferenceError) as raised:
        proc._flush_pending(pending, str(tmp_path), writer, checkpoint, [0])
    assert raised.value.error_code == "invalid_output_contract"
    writer.append.assert_not_called()
    checkpoint.mark_done_batch.assert_not_called()


def test_content_error_rejects_only_poison_singleton_and_preserves_order(monkeypatch, tmp_path):
    proc = AudioTagClassifierProcessor({
        "mode": "dry_run", "label_dict": _REAL_LABEL_DICT,
        "batch_size": 10,
    })
    pending = _make_records(10)
    expected_ids = [record.record_id for record in pending]
    real_run = proc._run_dry_run

    def infer(records):
        if any(record.record_id == "ds__src__4" for record in records):
            raise DaemonInferenceError("content", "audio_decode_failed", "bad wav")
        return real_run(records)

    monkeypatch.setattr(proc, "_run_dry_run", infer)
    writer, checkpoint = MagicMock(), MagicMock()
    proc._flush_pending(pending, str(tmp_path), writer, checkpoint, [0])
    written = [call.args[0] for call in writer.append.call_args_list]
    assert [record.record_id for record in written] == expected_ids
    assert [record.stages["C2"]["status"] for record in written].count("rejected") == 1
    assert written[4].rejected_reason == "c2_content_error"
    checkpoint.mark_done_batch.assert_called_once_with(expected_ids)


@pytest.mark.parametrize(("failed", "total"), [(1, 1), (1, 9)])
def test_tail_content_failure_commits_when_below_full_batch(
    monkeypatch, tmp_path, failed, total
):
    proc = AudioTagClassifierProcessor({
        "mode": "dry_run", "label_dict": _REAL_LABEL_DICT,
        "batch_size": 64,
    })
    pending = _make_records(total)
    expected_ids = [record.record_id for record in pending]
    poisoned = set(expected_ids[:failed])
    real_run = proc._run_dry_run

    def infer(records):
        if any(record.record_id in poisoned for record in records):
            raise DaemonInferenceError("content", "bad_audio", "bad")
        return real_run(records)

    monkeypatch.setattr(proc, "_run_dry_run", infer)
    writer, checkpoint = MagicMock(), MagicMock()
    batch_index = [0]

    proc._flush_pending(pending, str(tmp_path), writer, checkpoint, batch_index)

    written = [call.args[0] for call in writer.append.call_args_list]
    assert [record.record_id for record in written] == expected_ids
    assert [record.stages["C2"]["status"] for record in written].count("rejected") == failed
    checkpoint.mark_done_batch.assert_called_once_with(expected_ids)
    assert pending == []
    assert batch_index == [1]


@pytest.mark.parametrize("failure", [
    RuntimeError("legacy error"),
    DaemonInferenceError("infra", "request_io_failed", "disk error"),
    DaemonInferenceError("internal", "unexpected_exception", "model bug"),
])
def test_non_content_error_never_commits(monkeypatch, tmp_path, failure):
    proc = AudioTagClassifierProcessor(
        {"mode": "dry_run", "label_dict": _REAL_LABEL_DICT}
    )
    monkeypatch.setattr(proc, "_run_dry_run", MagicMock(side_effect=failure))
    writer, checkpoint = MagicMock(), MagicMock()
    with pytest.raises(RuntimeError):
        proc._flush_pending(_make_records(2), str(tmp_path), writer, checkpoint, [0])
    writer.append.assert_not_called()
    checkpoint.mark_done_batch.assert_not_called()


@pytest.mark.parametrize(("failed", "total"), [(8, 80), (2, 10)])
def test_content_failure_circuit_breaker_commits_then_aborts(monkeypatch, tmp_path, failed, total):
    """内容级熔断必须先落盘本批结果并写 checkpoint，再中止。

    回归：中止前不落盘时，重跑的同序输入构成同一批、同一条件再次熔断，
    stage 只能反复中止在同一批（永远无法收敛）。
    """
    proc = AudioTagClassifierProcessor({
        "mode": "dry_run", "label_dict": _REAL_LABEL_DICT,
        "batch_size": total,
    })
    pending = _make_records(total)
    expected_ids = [record.record_id for record in pending]
    poisoned = {record.record_id for record in pending[:failed]}
    real_run = proc._run_dry_run

    def infer(records):
        if any(record.record_id in poisoned for record in records):
            raise DaemonInferenceError("content", "bad_audio", "bad")
        return real_run(records)

    monkeypatch.setattr(proc, "_run_dry_run", infer)
    writer, checkpoint = MagicMock(), MagicMock()
    with pytest.raises(DaemonInferenceError, match="content_failure_circuit_breaker"):
        proc._flush_pending(pending, str(tmp_path), writer, checkpoint, [0])

    written = [call.args[0] for call in writer.append.call_args_list]
    assert [record.record_id for record in written] == expected_ids
    assert [record.stages["C2"]["status"] for record in written].count("rejected") == failed
    checkpoint.mark_done_batch.assert_called_once_with(expected_ids)
    # 中止前必须强制落盘：否则 checkpoint 可能未写入，重跑重复消费同一批
    checkpoint.save.assert_called_once_with(force=True)
    assert pending == []


def test_infer_parallel_uses_fixed_size_shards_for_tail_batch(monkeypatch, tmp_path):
    """用 31→8 个小分片同构验证生产 511→8 尾批，且分片确实并发。"""
    pool = _make_mock_pool(8)
    pool.batch_size = 4
    monkeypatch.setattr(pool, "start", lambda output_dir: None)
    calls, workers, lock = [], set(), threading.Lock()
    active = max_active = 0

    def infer(records, output_dir, batch_index):
        nonlocal active, max_active
        with lock:
            calls.append(len(records)); workers.add(threading.get_ident())
            active += 1; max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return [{"record_id": record.record_id} for record in records]

    monkeypatch.setattr(pool, "infer", infer)
    results = pool.infer_parallel(_make_records(31), str(tmp_path), 3)
    assert sorted(calls) == [3] + [4] * 7
    assert len(workers) > 1 and max_active > 1
    assert [item["record_id"] for item in results] == [f"ds__src__{i}" for i in range(31)]


def test_infer_parallel_shards_scale_with_request_batches(monkeypatch, tmp_path):
    """request_batches>1 时每个 daemon 请求含多个内部 batch，批间双缓冲才生效。"""
    pool = _make_mock_pool(2)
    pool.batch_size = 4
    pool.request_batches = 2
    monkeypatch.setattr(pool, "start", lambda output_dir: None)
    calls = []

    def infer(records, output_dir, batch_index):
        calls.append(len(records))
        return [{"record_id": record.record_id} for record in records]

    monkeypatch.setattr(pool, "infer", infer)
    results = pool.infer_parallel(_make_records(31), str(tmp_path), 3)
    assert sorted(calls) == [7, 8, 8, 8]
    assert [item["record_id"] for item in results] == [f"ds__src__{i}" for i in range(31)]


@pytest.mark.unit
@pytest.mark.subprocess
def test_daemon_bootstraps_repo_root_for_core_imports(tmp_path):
    """daemon 在任意 cwd、无 PYTHONPATH 时仍能解析仓库内安全加载模块。"""
    daemon_file = Path(audiotag_result.__file__).resolve().parent / "audiotag_daemon.py"
    assert daemon_file.is_file()
    script_dir = daemon_file.parent
    repo_root = script_dir.parents[1]
    assert (repo_root / "core" / "utils" / "safe_load.py").is_file()

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    bootstrap = f"""
import importlib.util
from pathlib import Path
path = Path({str(daemon_file)!r})
spec = importlib.util.spec_from_file_location("_audiotag_daemon_smoke", path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
from core.utils.safe_load import safe_torch_load
assert callable(safe_torch_load)
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", bootstrap],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


def test_daemon_non_object_json_returns_protocol_invalid(monkeypatch):
    daemon = object.__new__(AudioTagDaemon)
    replies = []
    monkeypatch.setattr(audiotag_daemon.sys, "stdin", io.StringIO("[]\n42\n"))
    monkeypatch.setattr(daemon, "_reply", replies.append)

    daemon._run_loop()

    assert len(replies) == 2
    assert all(
        (reply["error_kind"], reply["error_code"])
        == ("internal", "protocol_invalid")
        for reply in replies
    )


@pytest.mark.parametrize(
    ("name", "bad_value"),
    [
        ("enable", 1),
        ("enable_checkpoint", "true"),
        ("sample_rate", True),
        ("batch_size", 0),
        ("num_gpus", 1.0),
        ("daemon_timeout", "30"),
    ],
)
def test_c2_config_requires_exact_types(name, bad_value):
    with pytest.raises(ValueError, match=name):
        AudioTagClassifierProcessor({"mode": "dry_run", name: bad_value})


def test_c2_strict_reader_rejects_global_duplicate_before_output(tmp_path):
    manifest = tmp_path / "input.jsonl"
    records = _make_records(2)
    records[1].record_id = records[0].record_id
    write_manifest(iter(records), str(manifest))
    output_dir = tmp_path / "out"
    proc = AudioTagClassifierProcessor(
        {"mode": "dry_run", "label_dict": _REAL_LABEL_DICT}
    )

    with pytest.raises(ValueError, match="duplicate record_id"):
        proc.process(str(manifest), str(output_dir))

    assert not output_dir.exists()
