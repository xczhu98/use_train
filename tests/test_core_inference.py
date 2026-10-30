"""core.inference 公共新 Interface 合同测试。"""

from pathlib import Path

import pytest

from core.inference import (
    ArtifactBinding,
    FileInferenceItem,
    InferenceSpec,
    PreparedOutput,
    ReusePolicy,
    bind_artifacts,
    bind_config,
    build_inventory,
    run_inference,
    run_inference_report,
    verify_file_bindings,
    verify_inventory,
)
from core.utils.inference_artifacts import load_run_manifest

pytestmark = pytest.mark.unit


def _spec(config: Path, source: Path, output_dir: Path, *, reuse=False, metadata=None):
    output = output_dir / "u1.wav"
    return InferenceSpec(
        config=config,
        output_dir=output_dir,
        items=[FileInferenceItem("u1", {"noisy": source}, output)],
        artifacts=ArtifactBinding(files={"config": config}, parameters={"task": "se"}),
        reuse=ReusePolicy(enabled=reuse),
        run_metadata=metadata or {},
    )


def test_inventory_and_bindings_are_revalidated(tmp_path: Path):
    source = tmp_path / "input.wav"
    artifact = tmp_path / "model.bin"
    source.write_bytes(b"input")
    artifact.write_bytes(b"model")
    inventory = build_inventory([("u1", {"noisy": source})])
    binding = bind_artifacts({"model": artifact}, parameters={"sr": 16000})
    assert inventory["items"][0]["uid"] == "u1"
    assert bind_config({"model": "test"})["algorithm"] == "sha256-json-v1"
    verify_inventory(inventory)
    verify_file_bindings(binding)
    source.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="输入"):
        verify_inventory(inventory)
    artifact.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="artifact"):
        verify_file_bindings(binding)


def test_runner_publishes_new_spec_and_report(tmp_path: Path):
    source = tmp_path / "input.wav"
    config = tmp_path / "config.yaml"
    output_dir = tmp_path / "outputs"
    source.write_bytes(b"input")
    config.write_text("model: test\n", encoding="utf-8")

    class Adapter:
        def infer(self, item):
            assert item.output == output_dir / "u1.wav"
            return PreparedOutput(item.uid, lambda path: path.write_bytes(b"output"))

    report = run_inference_report(_spec(config, source, output_dir), Adapter())
    assert report.status == "completed"
    manifest, loaded = load_run_manifest(output_dir / "run_manifest.json")
    assert manifest["expected_uids"] == ["u1"]
    assert loaded[0].output_path == str((output_dir / "u1.wav").resolve())
    assert report.scp_path.is_file()


def test_runner_allows_partial_failures(tmp_path: Path):
    source = tmp_path / "input.wav"
    config = tmp_path / "config.yaml"
    output_dir = tmp_path / "outputs"
    source.write_bytes(b"input")
    config.write_text("model: test\n", encoding="utf-8")

    class Adapter:
        def infer(self, item):
            raise RuntimeError("adapter failed")

    report = run_inference_report(
        InferenceSpec(
            **{**_spec(config, source, output_dir).__dict__, "allow_partial": True}
        ),
        Adapter(),
    )
    assert report.status == "failed"
    assert (output_dir / "run_manifest.json").is_file()


def test_runner_rejects_artifact_changed_during_inference(tmp_path: Path):
    source = tmp_path / "input.wav"
    artifact = tmp_path / "config.yaml"
    output_dir = tmp_path / "outputs"
    source.write_bytes(b"input")
    artifact.write_bytes(b"model")

    class Adapter:
        def infer(self, item):
            artifact.write_bytes(b"changed")
            return PreparedOutput(item.uid, lambda path: path.write_bytes(b"output"))

    with pytest.raises(RuntimeError, match="artifact"):
        run_inference(
            InferenceSpec(
                config=artifact,
                output_dir=output_dir,
                items=[FileInferenceItem("u1", {"noisy": source}, output_dir / "u1.wav")],
                artifacts=ArtifactBinding(files={"config": artifact}),
            ),
            Adapter(),
        )


def test_runner_reuses_matching_output_and_metadata_invalidates(tmp_path: Path):
    source = tmp_path / "input.wav"
    config = tmp_path / "config.yaml"
    output_dir = tmp_path / "outputs"
    source.write_bytes(b"input")
    config.write_text("model: test\n", encoding="utf-8")
    calls = []

    class Adapter:
        def infer(self, item):
            calls.append(item.uid)
            return PreparedOutput(item.uid, lambda path: path.write_bytes(b"output"))

    first = _spec(config, source, output_dir, metadata={"mode": "offline"})
    run_inference(first, Adapter())
    reused = run_inference(_spec(config, source, output_dir, reuse=True, metadata={"mode": "offline"}), Adapter())
    rerun = run_inference(_spec(config, source, output_dir, reuse=True, metadata={"mode": "streaming"}), Adapter())
    assert reused.results[0].status == "skipped"
    assert rerun.results[0].status == "ok"
    assert calls == ["u1", "u1"]


def test_runner_rejects_invalid_adapter_output_contract(tmp_path: Path):
    source = tmp_path / "input.wav"
    config = tmp_path / "config.yaml"
    output_dir = tmp_path / "outputs"
    source.write_bytes(b"input")
    config.write_text("model: test\n", encoding="utf-8")

    class BadAdapter:
        def infer(self, item):
            return object()

    with pytest.raises(TypeError, match="PreparedOutput"):
        run_inference(_spec(config, source, output_dir), BadAdapter())


def test_runner_rejects_empty_items_before_publishing_metadata(tmp_path: Path):
    config = tmp_path / "config.yaml"
    output_dir = tmp_path / "outputs"
    config.write_text("model: test\n", encoding="utf-8")
    spec = InferenceSpec(
        config=config,
        output_dir=output_dir,
        items=[],
        artifacts=ArtifactBinding(files={"config": config}),
    )

    with pytest.raises(ValueError, match="at least one inference item"):
        run_inference(spec, object())
    assert not output_dir.exists()


def test_runner_publishes_consistent_manifest_when_aborting(tmp_path: Path):
    """allow_partial=False 中止时必须发布与已落盘音频一致的结果集合。"""
    config = tmp_path / "config.yaml"
    output_dir = tmp_path / "outputs"
    config.write_text("model: test\n", encoding="utf-8")
    items = []
    for uid in ("u1", "u2", "u3"):
        source = tmp_path / f"{uid}.wav"
        source.write_bytes(b"input")
        items.append(FileInferenceItem(uid, {"noisy": source}, output_dir / f"{uid}.wav"))

    class Adapter:
        def infer(self, item):
            if item.uid == "u2":
                raise RuntimeError("adapter failed")
            return PreparedOutput(item.uid, lambda path: path.write_bytes(b"output"))

    with pytest.raises(RuntimeError, match="adapter failed"):
        run_inference(
            InferenceSpec(
                config=config,
                output_dir=output_dir,
                items=items,
                artifacts=ArtifactBinding(files={"config": config}),
            ),
            Adapter(),
        )

    assert (output_dir / "u1.wav").is_file()
    manifest, results = load_run_manifest(output_dir / "run_manifest.json")
    assert manifest["ok_uids"] == ["u1"]
    assert manifest["failed_uids"] == ["u2", "u3"]
    assert [result.status for result in results] == ["ok", "failed", "failed"]
    scp_text = (output_dir / "inf.scp").read_text(encoding="utf-8")
    assert "u1" in scp_text
    assert "u2" not in scp_text and "u3" not in scp_text


def test_runner_reinfers_when_reusable_output_disappears(monkeypatch, tmp_path: Path):
    """复用命中项在复核时消失必须降级为重推，而不是中断整个 run。"""
    import core.inference.runner as runner_module

    source = tmp_path / "input.wav"
    config = tmp_path / "config.yaml"
    output_dir = tmp_path / "outputs"
    source.write_bytes(b"input")
    config.write_text("model: test\n", encoding="utf-8")
    calls = []

    class Adapter:
        def infer(self, item):
            calls.append(item.uid)
            return PreparedOutput(item.uid, lambda path: path.write_bytes(b"output"))

    run_inference(_spec(config, source, output_dir), Adapter())

    real_stat = runner_module.stat_audio_identity
    remaining = {"failures": 1}

    def flaky_stat(path):
        if remaining["failures"]:
            remaining["failures"] -= 1
            raise FileNotFoundError(str(path))
        return real_stat(path)

    monkeypatch.setattr(runner_module, "stat_audio_identity", flaky_stat)

    report = run_inference(_spec(config, source, output_dir, reuse=True), Adapter())
    assert report.status == "completed"
    assert report.results[0].status == "ok"
    assert calls == ["u1", "u1"]
