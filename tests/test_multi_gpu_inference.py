"""多 GPU 推理编排的最小 unit 契约。"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

import evaluate.multi_gpu_inference as multi
from core.utils.inference_artifacts import (
    InferenceResult,
    atomic_write_json,
    build_content_fingerprint,
    build_input_inventory,
    build_run_manifest,
    load_run_manifest,
    stat_audio_identity,
)
from evaluate.inference_components import effective_config_identity

pytestmark = pytest.mark.unit


def _artifact_fingerprint(config: Path, checkpoint: Path, mapping=None) -> dict:
    return build_content_fingerprint(
        {"config": config, "checkpoint": checkpoint},
        parameters={
            "effective_config_identity": effective_config_identity(
                {"model": "test"} if mapping is None else mapping
            )
        },
    )


def _wav(path: Path, value: float = 0.0) -> Path:
    sf.write(path, np.full(8, value, dtype=np.float32), 16000)
    return path


def test_script_entrypoint_bootstraps_repository_imports():
    script = Path(__file__).resolve().parents[1] / "evaluate" / "multi_gpu_inference.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parent.parent,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "multi GPU inference worker" in completed.stdout


def test_shard_and_validate_devices_are_stable():
    assert multi.shard_uids(["b", "a", "c", "d"], 2) == [["a", "c"], ["b", "d"]]
    assert multi.validate_inference_devices("cuda:0, cuda:1") == ["cuda:0", "cuda:1"]
    with pytest.raises(ValueError, match="重复"):
        multi.validate_inference_devices(["cuda:0", "cuda:0"])
    with pytest.raises(ValueError, match="不支持"):
        multi.validate_inference_devices(["gpu0"])


def test_worker_command_forwards_mode_specific_options(tmp_path: Path):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    worker_dir = tmp_path / "worker"
    worker_dir.mkdir()
    se_command = multi._worker_command(
        "se", config, checkpoint, worker_dir, "cuda:1", {"noisy": tmp_path / "in.scp"},
        sr=16000,
        allow_partial=True,
        expected_effective_config_identity={"sha256": "a" * 64},
    )
    assert se_command[se_command.index("--sr") : se_command.index("--sr") + 2] == ["--sr", "16000"]
    assert se_command[
        se_command.index("--expected-effective-config-sha256") + 1
    ] == "a" * 64
    assert "--allow-partial" in se_command
    aec_command = multi._worker_command(
        "aec_paired", config, checkpoint, worker_dir, "cuda:1",
        {"mic": tmp_path / "mic.scp"}, {"farend": tmp_path / "far.scp"},
        no_stream_align=True,
    )
    assert "--aec_no_stream_align" in aec_command
    assert "--skip-existing" not in aec_command


def test_normalize_devices_validates_metric_fallback():
    assert multi.normalize_inference_devices(None, fallback="cpu") == ["cpu"]
    with pytest.raises(ValueError, match="不支持"):
        multi.normalize_inference_devices("cuda:0,cuda:1", fallback="invalid")


def test_multi_gpu_skip_existing_reuses_verified_results_without_workers(tmp_path: Path, monkeypatch):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    input_path = _wav(tmp_path / "input.wav")
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    output_path = output_dir / "u.wav"
    _wav(output_path, 1.0)
    inventory = build_input_inventory([("u", {"noisy": input_path})])
    base = _artifact_fingerprint(config, checkpoint)
    artifact = multi._multi_fingerprint(base, ["cpu", "cuda:1"])
    manifest = build_run_manifest(
        config=config,
        checkpoint=checkpoint,
        artifact_fingerprint=artifact,
        input_inventory=inventory,
        results=[
            InferenceResult(
                uid="u",
                status="ok",
                output_path=str(output_path.resolve()),
                output_identity=stat_audio_identity(output_path),
            )
        ],
        allow_partial=False,
        created_at="2026-01-01T00:00:00",
    )
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    (output_dir / "inf.scp").write_text(f"u {output_path.resolve()}\n", encoding="utf-8")

    def fail_if_worker_runs(_command):
        raise AssertionError("verified reusable UID must not be dispatched to a worker")

    monkeypatch.setattr(multi, "_run_worker", fail_if_worker_runs)
    result = multi._run_multi(
        mode="se",
        config=config,
        checkpoint=checkpoint,
        output_dir=output_dir,
        devices=["cpu", "cuda:1"],
        input_maps={"noisy": {"u": input_path}},
        input_inventory=inventory,
        skip_existing=True,
    )
    assert result == output_dir.resolve()
    loaded, results = load_run_manifest(output_dir / "run_manifest.json")
    assert loaded["expected_uids"] == ["u"]
    assert results[0].status == "skipped"
    assert Path(results[0].output_path) == output_path.resolve()


def test_multi_gpu_skip_existing_merges_reused_and_worker_results(
    tmp_path: Path, monkeypatch
):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    inputs = {
        uid: _wav(tmp_path / f"{uid}-in.wav") for uid in ("a", "b")
    }
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    reused_output = output_dir / "a.wav"
    _wav(reused_output, 1.0)
    inventory = build_input_inventory(
        [(uid, {"noisy": path}) for uid, path in inputs.items()]
    )
    base = _artifact_fingerprint(config, checkpoint)
    existing_artifact = multi._multi_fingerprint(base, ["cpu", "cuda:1"])
    existing_manifest = build_run_manifest(
        config=config,
        checkpoint=checkpoint,
        artifact_fingerprint=existing_artifact,
        input_inventory=inventory,
        results=[
            InferenceResult(
                uid="a",
                status="ok",
                output_path=str(reused_output.resolve()),
                output_identity=stat_audio_identity(reused_output),
            ),
            InferenceResult(uid="b", status="failed", error="pending"),
        ],
        allow_partial=True,
        created_at="2026-01-01T00:00:00",
    )
    atomic_write_json(output_dir / "run_manifest.json", existing_manifest)

    def fake_run_worker(command):
        worker_dir = Path(command[command.index("--output-dir") + 1])
        assert "--skip-existing" not in command
        worker_dir.mkdir(exist_ok=True)
        _worker_manifest(
            worker_dir, config, checkpoint, "b", inputs["b"], worker_dir / "b.wav"
        )

    monkeypatch.setattr(multi, "_run_worker", fake_run_worker)
    multi._run_multi(
        mode="se",
        config=config,
        checkpoint=checkpoint,
        output_dir=output_dir,
        devices=["cpu", "cuda:1"],
        input_maps={"noisy": inputs},
        input_inventory=inventory,
        skip_existing=True,
    )
    _, results = load_run_manifest(output_dir / "run_manifest.json")
    by_uid = {item.uid: item for item in results}
    assert by_uid["a"].status == "skipped"
    assert by_uid["a"].output_path == str(reused_output.resolve())
    assert by_uid["b"].status == "ok"
    assert Path(by_uid["b"].output_path).is_file()


def test_single_device_wrapper_does_not_enter_multi_gpu(monkeypatch, tmp_path: Path):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    input_scp = tmp_path / "input.scp"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    input_scp.write_text("u /tmp/u.wav\n", encoding="utf-8")
    called = []
    monkeypatch.setattr(multi, "run_se_inference_multi_gpu", lambda *a, **k: called.append(1))
    monkeypatch.setattr(multi.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(
        "evaluate.eval_pipeline_common._validate_fresh_inference_manifest",
        lambda *a, **k: None,
    )
    from evaluate.eval_pipeline_common import run_inference_se

    run_inference_se(config, checkpoint, input_scp=input_scp, output_dir=tmp_path / "out", device="cpu")
    assert called == []


def _worker_manifest(worker_dir: Path, config: Path, checkpoint: Path, uid: str, input_path: Path, output: Path):
    worker_dir.mkdir(exist_ok=True)
    worker_dir_config = worker_dir / "config.yaml"
    worker_dir_checkpoint = worker_dir / "model.tar"
    worker_dir_config.write_bytes(config.read_bytes())
    worker_dir_checkpoint.write_bytes(checkpoint.read_bytes())
    output.parent.mkdir(parents=True, exist_ok=True)
    _wav(output, 1.0)
    inventory = build_input_inventory([(uid, {"noisy": input_path})])
    manifest = build_run_manifest(
        config=config,
        checkpoint=checkpoint,
        artifact_fingerprint=_artifact_fingerprint(config, checkpoint),
        input_inventory=inventory,
        results=[InferenceResult(uid=uid, status="ok", output_path=str(output.resolve()), output_identity=stat_audio_identity(output))],
        allow_partial=False,
        created_at="2026-01-01T00:00:00",
    )
    atomic_write_json(worker_dir / "run_manifest.json", manifest)
    (worker_dir / "inf.scp").write_text(f"{uid} {output.resolve()}\n", encoding="utf-8")


def test_merge_contract_publishes_complete_uid_set(monkeypatch, tmp_path: Path):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    inputs = {uid: _wav(tmp_path / f"{uid}-in.wav") for uid in ("a", "b")}
    input_scp = tmp_path / "input.scp"
    input_scp.write_text("a a-in.wav\nb b-in.wav\n", encoding="utf-8")

    def fake_run_worker(command):
        worker_dir = Path(command[command.index("--output-dir") + 1])
        worker_dir.mkdir(exist_ok=True)
        uid = "a" if "worker_00" in str(worker_dir) else "b"
        source = inputs[uid]
        output = worker_dir / f"{uid}.wav"
        _worker_manifest(worker_dir, config, checkpoint, uid, source, output)

    monkeypatch.setattr(multi, "_run_worker", fake_run_worker)
    output_dir = tmp_path / "out"
    result = multi.run_se_inference_multi_gpu(config, checkpoint, input_scp, output_dir, ["cpu", "cuda:1"])
    manifest, results = load_run_manifest(result / "run_manifest.json")
    assert manifest["expected_uids"] == ["a", "b"]
    assert {item.uid for item in results} == {"a", "b"}
    assert set((result / "inf.scp").read_text(encoding="utf-8").splitlines()[i].split()[0] for i in range(2)) == {"a", "b"}
    assert all(Path(item.output_path).is_file() for item in results)


def test_publish_copy_uses_distinct_unpredictable_exclusive_temporaries(tmp_path: Path):
    source = _wav(tmp_path / "source.wav", 1.0)
    destination = tmp_path / "result.wav"

    first = multi._copy_to_exclusive_temporary(source, destination)
    second = multi._copy_to_exclusive_temporary(source, destination)

    assert first != second
    assert first.parent == second.parent == tmp_path
    assert first.name.startswith(".result.wav.") and first.name.endswith(".tmp")
    assert second.name.startswith(".result.wav.") and second.name.endswith(".tmp")
    assert first.name != f".result.wav.{multi.os.getpid()}.tmp"
    assert second.name != f".result.wav.{multi.os.getpid()}.tmp"
    first.unlink()
    second.unlink()


def test_worker_failure_is_fail_closed_and_preserves_existing_metadata(monkeypatch, tmp_path: Path):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    input_scp = tmp_path / "input.scp"
    _wav(tmp_path / "a.wav")
    _wav(tmp_path / "b.wav")
    input_scp.write_text("a a.wav\nb b.wav\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    old_manifest = {"old": True}
    atomic_write_json(output_dir / "run_manifest.json", old_manifest)
    (output_dir / "inf.scp").write_text("old /tmp/old.wav\n", encoding="utf-8")
    monkeypatch.setattr(multi, "_run_worker", lambda command: (_ for _ in ()).throw(multi.MultiGPUInferenceError("boom")))

    with pytest.raises(multi.MultiGPUInferenceError):
        multi.run_se_inference_multi_gpu(config, checkpoint, input_scp, output_dir, ["cpu", "cuda:1"])
    assert json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8")) == old_manifest
    assert (output_dir / "inf.scp").read_text(encoding="utf-8") == "old /tmp/old.wav\n"
    assert not list(tmp_path.glob("multi_gpu_workers_*"))


def test_merge_rejects_missing_uid_before_publishing(tmp_path: Path):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    input_a = _wav(tmp_path / "a-in.wav")
    input_b = _wav(tmp_path / "b-in.wav")
    inventory = build_input_inventory(
        [("a", {"noisy": input_a}), ("b", {"noisy": input_b})]
    )
    output = _wav(tmp_path / "worker-a.wav", 1.0)
    result = InferenceResult(
        uid="a",
        status="ok",
        output_path=str(output.resolve()),
        output_identity=stat_audio_identity(output),
    )
    fingerprint = _artifact_fingerprint(config, checkpoint)
    manifests = [{
        "config": str(config.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "artifact_fingerprint": fingerprint,
        "created_at": "2026-01-01T00:00:00",
    }]

    with pytest.raises(multi.MultiGPUInferenceError, match="重复或缺失"):
        multi._publish_merged(
            tmp_path / "out",
            config,
            checkpoint,
            inventory,
            manifests,
            {"a": result},
            ["cpu", "cuda:1"],
        )
    assert not (tmp_path / "out" / "run_manifest.json").exists()


def test_merge_rejects_worker_fingerprint_mismatch(tmp_path: Path):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    input_path = _wav(tmp_path / "a-in.wav")
    inventory = build_input_inventory([("a", {"noisy": input_path})])
    output = _wav(tmp_path / "worker-a.wav", 1.0)
    result = InferenceResult(
        uid="a",
        status="ok",
        output_path=str(output.resolve()),
        output_identity=stat_audio_identity(output),
    )
    first_fingerprint = _artifact_fingerprint(config, checkpoint)
    other = tmp_path / "other.tar"
    other.write_bytes(b"different")
    second_fingerprint = _artifact_fingerprint(config, other)
    manifests = [
        {
            "config": str(config.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "artifact_fingerprint": first_fingerprint,
            "created_at": "2026-01-01T00:00:00",
        },
        {
            "config": str(config.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "artifact_fingerprint": second_fingerprint,
            "created_at": "2026-01-01T00:00:00",
        },
    ]

    with pytest.raises(multi.MultiGPUInferenceError, match="fingerprint 不一致"):
        multi._publish_merged(
            tmp_path / "out",
            config,
            checkpoint,
            inventory,
            manifests,
            {"a": result},
            ["cpu", "cuda:1"],
        )
    assert not (tmp_path / "out" / "run_manifest.json").exists()


def test_merge_rolls_back_wavs_when_publish_copy_fails(monkeypatch, tmp_path: Path):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    inputs = {uid: _wav(tmp_path / f"{uid}-in.wav") for uid in ("a", "b")}
    input_scp = tmp_path / "input.scp"
    input_scp.write_text("a a-in.wav\nb b-in.wav\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    old_manifest = {"old": True}
    atomic_write_json(output_dir / "run_manifest.json", old_manifest)
    (output_dir / "inf.scp").write_text("old /tmp/old.wav\n", encoding="utf-8")

    def fake_run_worker(command):
        worker_dir = Path(command[command.index("--output-dir") + 1])
        uid = "a" if "worker_00" in str(worker_dir) else "b"
        _worker_manifest(worker_dir, config, checkpoint, uid, inputs[uid], worker_dir / f"{uid}.wav")

    original_copy = multi._copy_to_exclusive_temporary
    calls = 0

    def fail_on_second_publish_copy(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated publish copy failure")
        return original_copy(source, destination)

    monkeypatch.setattr(multi, "_run_worker", fake_run_worker)
    monkeypatch.setattr(
        multi, "_copy_to_exclusive_temporary", fail_on_second_publish_copy
    )

    with pytest.raises(OSError, match="simulated publish copy failure"):
        multi.run_se_inference_multi_gpu(
            config, checkpoint, input_scp, output_dir, ["cpu", "cuda:1"]
        )
    assert json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8")) == old_manifest
    assert (output_dir / "inf.scp").read_text(encoding="utf-8") == "old /tmp/old.wav\n"
    assert not list(output_dir.glob("uid-*.wav"))
    assert not list(output_dir.glob(".*.tmp"))


def test_merge_rolls_back_installed_wav_when_final_input_gate_fails(
    monkeypatch, tmp_path: Path
):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    inputs = {uid: _wav(tmp_path / f"{uid}-in.wav") for uid in ("a", "b")}
    input_scp = tmp_path / "input.scp"
    input_scp.write_text("a a-in.wav\nb b-in.wav\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing = output_dir / "uid-61.wav"
    _wav(existing, -1.0)
    original_existing = existing.read_bytes()
    old_manifest = {"old": True}
    atomic_write_json(output_dir / "run_manifest.json", old_manifest)
    (output_dir / "inf.scp").write_text("old /tmp/old.wav\n", encoding="utf-8")

    def fake_run_worker(command):
        worker_dir = Path(command[command.index("--output-dir") + 1])
        uid = "a" if "worker_00" in str(worker_dir) else "b"
        _worker_manifest(
            worker_dir, config, checkpoint, uid, inputs[uid], worker_dir / f"{uid}.wav"
        )

    original_validate = multi._validate_input_inventory_current
    calls = 0

    def fail_after_install(inventory):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise multi.MultiGPUInferenceError("simulated final input gate failure")
        original_validate(inventory)

    monkeypatch.setattr(multi, "_run_worker", fake_run_worker)
    monkeypatch.setattr(multi, "_validate_input_inventory_current", fail_after_install)

    with pytest.raises(multi.MultiGPUInferenceError, match="final input gate failure"):
        multi.run_se_inference_multi_gpu(
            config, checkpoint, input_scp, output_dir, ["cpu", "cuda:1"]
        )

    assert existing.read_bytes() == original_existing
    assert not (output_dir / "uid-62.wav").exists()
    assert json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8")) == old_manifest
    assert (output_dir / "inf.scp").read_text(encoding="utf-8") == "old /tmp/old.wav\n"
    assert not list(output_dir.glob(".*.tmp"))


def test_merge_restores_existing_wav_when_install_replace_fails(monkeypatch, tmp_path: Path):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    inputs = {uid: _wav(tmp_path / f"{uid}-in.wav") for uid in ("a", "b")}
    input_scp = tmp_path / "input.scp"
    input_scp.write_text("a a-in.wav\nb b-in.wav\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing = output_dir / "uid-61.wav"
    _wav(existing, -1.0)
    original_existing = existing.read_bytes()

    def fake_run_worker(command):
        worker_dir = Path(command[command.index("--output-dir") + 1])
        uid = "a" if "worker_00" in str(worker_dir) else "b"
        _worker_manifest(worker_dir, config, checkpoint, uid, inputs[uid], worker_dir / f"{uid}.wav")

    original_replace = multi.os.replace

    def fail_install(source, destination, **kwargs):
        if Path(destination) == existing and Path(source).name.startswith(".uid-61.wav."):
            raise OSError("simulated install replace failure")
        return original_replace(source, destination, **kwargs)

    monkeypatch.setattr(multi, "_run_worker", fake_run_worker)
    monkeypatch.setattr(multi.os, "replace", fail_install)

    with pytest.raises(OSError, match="simulated install replace failure"):
        multi.run_se_inference_multi_gpu(
            config, checkpoint, input_scp, output_dir, ["cpu", "cuda:1"]
        )

    assert existing.read_bytes() == original_existing
    assert not list(output_dir.glob(".*.tmp"))
    assert not list(output_dir.glob("uid-*.wav.tmp"))


def test_merge_keeps_existing_wav_visible_until_atomic_install(monkeypatch, tmp_path: Path):
    """安装新文件前旧文件必须仍在原位可读，避免发布中途出现 output 空窗。"""
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    inputs = {uid: _wav(tmp_path / f"{uid}-in.wav") for uid in ("a", "b")}
    input_scp = tmp_path / "input.scp"
    input_scp.write_text("a a-in.wav\nb b-in.wav\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing = output_dir / "uid-61.wav"
    _wav(existing, -1.0)
    original_existing = existing.read_bytes()

    def fake_run_worker(command):
        worker_dir = Path(command[command.index("--output-dir") + 1])
        uid = "a" if "worker_00" in str(worker_dir) else "b"
        _worker_manifest(worker_dir, config, checkpoint, uid, inputs[uid], worker_dir / f"{uid}.wav")

    observed: list[tuple[bool, bytes]] = []
    original_replace = multi.os.replace

    def observe_install(source, destination, **kwargs):
        if Path(destination) == existing and Path(source).name.startswith(".uid-61.wav."):
            observed.append((existing.is_file(), existing.read_bytes()))
        return original_replace(source, destination, **kwargs)

    monkeypatch.setattr(multi, "_run_worker", fake_run_worker)
    monkeypatch.setattr(multi.os, "replace", observe_install)

    multi.run_se_inference_multi_gpu(config, checkpoint, input_scp, output_dir, ["cpu", "cuda:1"])

    assert observed == [(True, original_existing)]
    assert existing.read_bytes() != original_existing


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--worker-mode", "se"],
        [
            "--worker-mode", "aec_paired",
            "-C", "config.yaml",
            "--checkpoint", "model.tar",
            "--output-dir", "out",
            "--device", "cpu",
        ],
    ],
)
def test_worker_cli_reports_usage_error_instead_of_none_typeerror(monkeypatch, argv):
    """worker CLI 缺失必需参数时必须给出用法错误，而不是把 None 拼进子进程命令。"""
    monkeypatch.setattr(sys, "argv", ["multi_gpu_inference.py", *argv])
    with pytest.raises(SystemExit) as exit_info:
        multi.main()
    assert exit_info.value.code == 2


@pytest.mark.parametrize("output_kind", ["symlink", "outside"])
def test_validate_worker_rejects_output_escape_or_symlink(tmp_path: Path, output_kind: str):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    input_path = _wav(tmp_path / "input.wav")
    worker_dir = tmp_path / "worker"
    worker_dir.mkdir()
    outside = _wav(tmp_path / f"{output_kind}.wav", 1.0)
    output = worker_dir / "a.wav"
    if output_kind == "symlink":
        output.symlink_to(outside)
        output_for_manifest = output
    else:
        output_for_manifest = outside

    inventory = build_input_inventory([("a", {"noisy": input_path})])
    result = InferenceResult(
        uid="a",
        status="ok",
        output_path=str(output_for_manifest),
        output_identity=stat_audio_identity(output_for_manifest),
    )
    fingerprint = _artifact_fingerprint(config, checkpoint)
    manifest = build_run_manifest(
        config=config,
        checkpoint=checkpoint,
        artifact_fingerprint=fingerprint,
        input_inventory=inventory,
        results=[result],
        allow_partial=False,
        created_at="2026-01-01T00:00:00",
    )
    atomic_write_json(worker_dir / "run_manifest.json", manifest)
    (worker_dir / "inf.scp").write_text(
        f"a {output_for_manifest}\n", encoding="utf-8"
    )

    expected_error = "普通文件" if output_kind == "symlink" else "越出 worker 目录"
    with pytest.raises(multi.MultiGPUInferenceError, match=expected_error):
        multi._validate_worker(
            worker_dir,
            config,
            checkpoint,
            inventory,
            allow_partial=False,
        )


def test_merge_rejects_worker_effective_config_identity_mismatch(tmp_path: Path):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    manifests = [
        {
            "artifact_fingerprint": _artifact_fingerprint(
                config, checkpoint, {"model": "first"}
            ),
            "config": str(config.resolve()),
            "checkpoint": str(checkpoint.resolve()),
        },
        {
            "artifact_fingerprint": _artifact_fingerprint(
                config, checkpoint, {"model": "second"}
            ),
            "config": str(config.resolve()),
            "checkpoint": str(checkpoint.resolve()),
        },
    ]

    with pytest.raises(
        multi.MultiGPUInferenceError, match="effective config identity 不一致"
    ):
        multi._publish_merged(
            tmp_path / "out",
            config,
            checkpoint,
            {"items": []},
            manifests,
            {},
            ["cpu", "cuda:1"],
        )


def test_merge_rejects_legacy_worker_manifest_without_effective_identity(tmp_path: Path):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    legacy = build_content_fingerprint({"config": config, "checkpoint": checkpoint})

    with pytest.raises(
        multi.MultiGPUInferenceError, match="缺少 effective config identity"
    ):
        multi._publish_merged(
            tmp_path / "out",
            config,
            checkpoint,
            {"items": []},
            [{"artifact_fingerprint": legacy}],
            {},
            ["cpu", "cuda:1"],
        )
