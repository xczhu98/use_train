"""推理 artifact 指纹、原子写与严格 manifest 契约测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.utils.inference_artifacts import (
    InferenceResult,
    ManifestValidationError,
    atomic_write_audio,
    atomic_write_json,
    atomic_write_text,
    build_content_fingerprint,
    build_input_inventory,
    build_run_manifest,
    collect_source_tree_files,
    fingerprint_file,
    fingerprint_json,
    fingerprint_path,
    inference_output_lock,
    load_run_manifest,
    publish_inference_metadata,
    reusable_uids_from_manifest,
    stat_audio_identity,
    validate_run_manifest,
)
from evaluate.inference_components import (
    build_artifact_fingerprint,
    effective_config_identity,
    finalize_inference_artifacts,
    inference_source_files,
    load_effective_config,
    resolve_reusable_uids,
)

# 仓库根由测试文件位置推导，避免依赖 pytest 的启动 cwd。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _build_complete_manifest(tmp_path: Path):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    noisy = tmp_path / "noisy.wav"
    output = tmp_path / "u1.wav"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    noisy.write_bytes(b"input")
    output.write_bytes(b"output")

    artifact = build_content_fingerprint(
        {"config": config, "checkpoint": checkpoint},
        parameters={"sr": 16000, "chunk_seconds": 0.0},
    )
    inventory = build_input_inventory([("u1", {"noisy": noisy})])
    result = InferenceResult(
        uid="u1",
        status="ok",
        output_path=str(output.resolve()),
        output_identity=stat_audio_identity(output),
    )
    manifest = build_run_manifest(
        config=config,
        checkpoint=checkpoint,
        artifact_fingerprint=artifact,
        input_inventory=inventory,
        results=[result],
        allow_partial=False,
        created_at="2026-08-12T00:00:00",
    )
    return manifest, artifact, inventory, output


def test_content_fingerprint_binds_bytes_path_and_parameters(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"v1")

    first = build_content_fingerprint(
        {"model": artifact, "optional": None}, parameters={"sr": 16000}
    )
    same = build_content_fingerprint(
        {"optional": None, "model": artifact}, parameters={"sr": 16000}
    )
    changed_parameter = build_content_fingerprint(
        {"model": artifact, "optional": None}, parameters={"sr": 8000}
    )
    artifact.write_bytes(b"v2")
    changed_content = build_content_fingerprint(
        {"model": artifact, "optional": None}, parameters={"sr": 16000}
    )

    assert first == same
    assert first["files"]["model"]["path"] == str(artifact.resolve())
    assert first["sha256"] != changed_parameter["sha256"]
    assert first["sha256"] != changed_content["sha256"]


def test_path_and_json_fingerprints_bind_directory_layout_and_content(tmp_path):
    root = tmp_path / "artifact"
    root.mkdir()
    first_file = root / "a.py"
    first_file.write_text("VALUE = 1\n", encoding="utf-8")

    first = fingerprint_path(root)
    assert first["kind"] == "directory"
    assert first["files"] == 1
    assert fingerprint_json({"b": 2, "a": 1}) == fingerprint_json({"a": 1, "b": 2})

    first_file.write_text("VALUE = 2\n", encoding="utf-8")
    assert fingerprint_path(root)["sha256"] != first["sha256"]
    assert fingerprint_path(first_file)["kind"] == "file"

    with pytest.raises(ValueError, match="标准 JSON"):
        fingerprint_json({"bad": object()})


def test_effective_config_identity_hashes_the_loaded_mapping_once():
    mapping = {"model": {"type": "Test"}, "stft": {"n_fft": 320}}
    calls = []

    loaded, identity = load_effective_config(
        "config.yaml",
        loader=lambda path: calls.append(path) or mapping,
    )

    assert loaded is mapping
    assert calls == ["config.yaml"]
    assert identity == {
        "schema": "effective-merged-config-v1",
        "algorithm": "sha256-json-v1",
        "sha256": "03d6f6b6a25254be9d7b543e064d3413f5f1b1ce1ba48216d5f23bbf8a214bd1",
    }
    assert effective_config_identity(mapping) == identity


def test_source_tree_files_are_stable_scoped_and_content_bound(tmp_path):
    source_root = tmp_path / "project"
    model_dir = source_root / "tasks" / "se" / "models"
    cache_dir = model_dir / "__pycache__"
    data_dir = source_root / "data"
    model_dir.mkdir(parents=True)
    cache_dir.mkdir()
    data_dir.mkdir()
    model = model_dir / "model.py"
    model.write_text("VALUE = 1\n", encoding="utf-8")
    (cache_dir / "model.py").write_text("CACHE = 1\n", encoding="utf-8")
    (data_dir / "ignored.py").write_text("DATA = 1\n", encoding="utf-8")

    files = collect_source_tree_files(source_root, ["tasks/se/models"])
    first = build_content_fingerprint(files)
    assert list(files) == ["source:tasks/se/models/model.py"]

    model.write_text("VALUE = 2\n", encoding="utf-8")
    changed = build_content_fingerprint(
        collect_source_tree_files(source_root, ["tasks/se/models"])
    )
    assert first["sha256"] != changed["sha256"]


def test_main_artifact_fingerprint_binds_task_model_sources(tmp_path):
    from argparse import Namespace

    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    model_source = tmp_path / "model.py"
    config.write_text("model: {type: Test}\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    model_source.write_text("VALUE = 1\n", encoding="utf-8")
    source_files = {"source:tasks/aec/models/model.py": model_source}
    args = Namespace(
        checkpoint=str(checkpoint),
        config=str(config),
        device="cpu",
        sr=None,
        chunk_seconds=0.0,
        aec_stream_align=False,
        aec_no_stream_align=False,
        aec_review_multichannel=False,
    )

    fingerprint_kwargs = {
        "project_root": _PROJECT_ROOT,
        "inference_entry": Path("inference.py"),
        "is_aec": True,
        "is_waveform_native": False,
        "source_files": source_files,
    }
    first = build_artifact_fingerprint(args, {"model": {"type": "Test"}}, **fingerprint_kwargs)
    args.aec_review_multichannel = True
    review_changed = build_artifact_fingerprint(
        args, {"model": {"type": "Test"}}, **fingerprint_kwargs
    )
    model_source.write_text("VALUE = 2\n", encoding="utf-8")
    changed = build_artifact_fingerprint(args, {"model": {"type": "Test"}}, **fingerprint_kwargs)

    assert "source:tasks/aec/models/model.py" in first["files"]
    assert first["parameters"]["aec_review_multichannel"] is False
    assert review_changed["parameters"]["aec_review_multichannel"] is True
    assert first["sha256"] != review_changed["sha256"]
    assert review_changed["sha256"] != changed["sha256"]


def test_aec_review_multichannel_rejected_for_se(monkeypatch, tmp_path):
    from argparse import Namespace

    import core.utils.config
    import inference

    monkeypatch.setattr(
        core.utils.config,
        "load_merged_config",
        lambda _path: {
            "data": {"mode": "dynamic"},
            "model": {"type": "TestSE"},
        },
    )
    args = Namespace(
        config=str(tmp_path / "se.yaml"),
        output_relpath_scp=None,
        device="cpu",
        cpu_long_audio_seconds=0.0,
        aec_review_multichannel=True,
    )

    with pytest.raises(ValueError, match="only valid for AEC"):
        inference._main_locked(args, tmp_path)


def test_inference_source_boundaries_are_task_specific():
    project_root = _PROJECT_ROOT
    se_files = inference_source_files(project_root, is_aec=False)
    aec_files = inference_source_files(project_root, is_aec=True)

    assert "source:core/stft.py" in se_files
    assert "source:core/utils/config.py" in se_files
    assert "source:core/utils/inference_artifacts.py" in se_files
    assert "source:core/utils/safe_load.py" in se_files
    assert "source:evaluate/checkpoint_loading.py" in se_files
    assert "source:evaluate/inference_components.py" in se_files
    assert "source:tasks/se/models/__init__.py" in se_files
    assert not any(name.startswith("source:tasks/aec/") for name in se_files)
    assert "source:core/stft.py" in aec_files
    assert "source:tasks/aec/models/__init__.py" in aec_files
    assert "source:tasks/aec/inference/aec_frontend.py" in aec_files
    assert "source:tasks/aec/inference/layer_g_frontend.py" not in aec_files
    assert "source:tasks/aec/inference/streaming_delay_estimator.py" in aec_files
    assert "source:tasks/aec/inference/csrc/delay_estimator_ext.cpp" in aec_files
    assert "source:tasks/aec/inference/csrc/pocketfft_hdronly.h" in aec_files
    assert "source:tasks/aec/data/linear_aec_frontend.py" in aec_files
    assert not any(name.startswith("source:tasks/se/") for name in aec_files)


def test_input_inventory_preserves_uid_order_and_rejects_duplicates(tmp_path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    inventory = build_input_inventory(
        [("b", {"noisy": second}), ("a", {"noisy": first})]
    )
    assert [item["uid"] for item in inventory["items"]] == ["b", "a"]
    assert inventory["algorithm"] == "stat-v1"
    assert inventory["items"][0]["inputs"]["noisy"]["algorithm"] == "stat-v1"

    with pytest.raises(ValueError, match="重复 uid"):
        build_input_inventory(
            [("dup", {"noisy": first}), ("dup", {"noisy": second})]
        )


def test_audio_identity_and_inventory_never_open_payload(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"payload")
    original_open = Path.open

    def reject_audio_open(path, *args, **kwargs):
        if Path(path) == audio:
            raise AssertionError("audio payload must not be opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_audio_open)
    identity = stat_audio_identity(audio)
    inventory = build_input_inventory([("u1", {"noisy": audio})])

    assert identity["st_size"] == len(b"payload")
    assert inventory["items"][0]["inputs"]["noisy"] == identity


def test_atomic_audio_write_preserves_old_target_on_writer_failure(tmp_path):
    output = tmp_path / "u1.wav"
    output.write_bytes(b"old")

    def fail_after_staging(staging):
        staging.write_bytes(b"partial")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        atomic_write_audio(output, fail_after_staging)

    assert output.read_bytes() == b"old"
    assert not list(tmp_path.glob(".*.tmp.wav"))


def test_atomic_writes_replace_complete_files(tmp_path):
    text_path = tmp_path / "inf.scp"
    json_path = tmp_path / "run_manifest.json"
    text_path.write_text("old\n", encoding="utf-8")

    atomic_write_text(text_path, "u1 /tmp/u1.wav\n")
    atomic_write_json(json_path, {"ok": ["u1"]})

    assert text_path.read_text(encoding="utf-8") == "u1 /tmp/u1.wav\n"
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"ok": ["u1"]}
    assert not list(tmp_path.glob(".*.tmp"))


def test_output_lock_file_persists_after_release(tmp_path):
    with inference_output_lock(tmp_path) as output:
        assert output == tmp_path.resolve()
        assert (output / ".inference.lock").is_file()

    assert (tmp_path / ".inference.lock").is_file()


def test_metadata_publish_rolls_back_all_files(monkeypatch, tmp_path):
    import core.utils.file_transaction as transaction

    scp = tmp_path / "inf.scp"
    sidecar = tmp_path / "inf.scp.provenance.json"
    manifest = tmp_path / "run_manifest.json"
    scp.write_text("old-scp\n", encoding="utf-8")
    sidecar.write_text('{"old": true}\n', encoding="utf-8")
    manifest.write_text('{"old": true}\n', encoding="utf-8")
    real_replace = transaction._replace_at
    replacements = 0

    def fail_second_replace(source, target, directory_fds):
        nonlocal replacements
        replacements += 1
        if replacements == 2:
            raise RuntimeError("publish failed")
        return real_replace(source, target, directory_fds)

    monkeypatch.setattr(transaction, "_replace_at", fail_second_replace)
    with pytest.raises(RuntimeError, match="publish failed"):
        publish_inference_metadata(
            tmp_path,
            scp_content="u1 /tmp/new.wav\n",
            manifest={"new": True},
        )

    assert scp.read_text(encoding="utf-8") == "old-scp\n"
    assert sidecar.read_text(encoding="utf-8") == '{"old": true}\n'
    assert json.loads(manifest.read_text(encoding="utf-8")) == {"old": True}


def test_metadata_publish_refreshes_scp_provenance(tmp_path):
    scp = tmp_path / "inf.scp"
    sidecar = tmp_path / "inf.scp.provenance.json"
    sidecar.write_text('{"entries": 0, "scp_sha256": "stale"}\n', encoding="utf-8")
    content = "u1 /tmp/u1.wav\nu2 /tmp/u2.wav\n"

    publish_inference_metadata(tmp_path, scp_content=content, manifest={"new": True})

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["entries"] == 2
    assert payload["scp_sha256"] == __import__("hashlib").sha256(
        content.encode("utf-8")
    ).hexdigest()
    assert payload["source"] == str(tmp_path.resolve())


def test_inference_result_rejects_mismatched_output_path(tmp_path):
    output = tmp_path / "out.wav"
    other = tmp_path / "other.wav"
    output.write_bytes(b"out")
    other.write_bytes(b"other")

    with pytest.raises(ManifestValidationError, match="同一文件"):
        InferenceResult(
            uid="u1",
            status="ok",
            output_path=str(output.resolve()),
            output_identity=stat_audio_identity(other),
        )

    with pytest.raises(ManifestValidationError, match="failed result 必须包含"):
        InferenceResult(uid="u1", status="failed")


def test_manifest_roundtrip_and_reuse_requires_current_output_identity(tmp_path):
    manifest, artifact, inventory, output = _build_complete_manifest(tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    atomic_write_json(manifest_path, manifest)

    loaded, results = load_run_manifest(
        manifest_path,
        expected_artifact_fingerprint=artifact,
        expected_input_inventory=inventory,
    )
    reusable = reusable_uids_from_manifest(
        manifest_path,
        expected_artifact_fingerprint=artifact,
        expected_input_inventory=inventory,
        expected_output_paths={"u1": output},
    )

    assert loaded == manifest
    assert [result.uid for result in results] == ["u1"]
    assert reusable == {"u1"}

    output.write_bytes(b"tampered")
    assert reusable_uids_from_manifest(
        manifest_path,
        expected_artifact_fingerprint=artifact,
        expected_input_inventory=inventory,
        expected_output_paths={"u1": output},
    ) == set()


def test_manifest_reuse_accepts_json_normalized_parameter_keys(tmp_path):
    """JSON 序列化后的整数参数键不应破坏相同 artifact 的复用。"""
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    noisy = tmp_path / "noisy.wav"
    output = tmp_path / "u1.wav"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    noisy.write_bytes(b"input")
    output.write_bytes(b"output")

    artifact = build_content_fingerprint(
        {"config": config, "checkpoint": checkpoint},
        parameters={"simulation": {"num_augmentations": {0: 1.0}}},
    )
    inventory = build_input_inventory([("u1", {"noisy": noisy})])
    manifest = build_run_manifest(
        config=config,
        checkpoint=checkpoint,
        artifact_fingerprint=artifact,
        input_inventory=inventory,
        results=[
            InferenceResult(
                uid="u1",
                status="ok",
                output_path=str(output.resolve()),
                output_identity=stat_audio_identity(output),
            )
        ],
        allow_partial=False,
        created_at="2026-09-12T00:00:00",
    )
    manifest_path = tmp_path / "run_manifest.json"
    atomic_write_json(manifest_path, manifest)

    assert reusable_uids_from_manifest(
        manifest_path,
        expected_artifact_fingerprint=artifact,
        expected_input_inventory=inventory,
        expected_output_paths={"u1": output},
    ) == {"u1"}


def test_manifest_reuse_follows_nested_output_path_mapping(tmp_path):
    """manifest 的 output_path 可映射到递归输出目录，同时保留旧扁平映射。"""
    manifest, artifact, inventory, output = _build_complete_manifest(tmp_path)
    nested_output = tmp_path / "delay_0_50ms" / "nested" / output.name
    nested_output.parent.mkdir(parents=True)
    nested_output.hardlink_to(output)
    manifest["results"][0]["output_path"] = str(nested_output.resolve())
    manifest["results"][0]["output_identity"] = stat_audio_identity(nested_output)
    manifest_path = tmp_path / "run_manifest.json"
    atomic_write_json(manifest_path, manifest)

    assert reusable_uids_from_manifest(
        manifest_path,
        expected_artifact_fingerprint=artifact,
        expected_input_inventory=inventory,
        expected_output_paths={"u1": nested_output},
    ) == {"u1"}
    assert reusable_uids_from_manifest(
        manifest_path,
        expected_artifact_fingerprint=artifact,
        expected_input_inventory=inventory,
        expected_output_paths={"u1": output},
    ) == set()


def test_manifest_strict_validation_rejects_extra_and_derived_uid_tampering(tmp_path):
    manifest, _, _, _ = _build_complete_manifest(tmp_path)
    with_extra = dict(manifest, unexpected=True)
    with pytest.raises(ManifestValidationError, match="extra=.*unexpected"):
        validate_run_manifest(with_extra)

    tampered = json.loads(json.dumps(manifest))
    tampered["ok_uids"] = []
    with pytest.raises(ManifestValidationError, match="ok_uids 与 results 不一致"):
        validate_run_manifest(tampered)


def test_manifest_strict_validation_rejects_identity_tampering(tmp_path):
    manifest, _, _, _ = _build_complete_manifest(tmp_path)
    tampered = json.loads(json.dumps(manifest))
    tampered["input_inventory"]["items"][0]["inputs"]["noisy"]["st_size"] += 1

    with pytest.raises(ManifestValidationError, match="input_inventory.digest 与内容不一致"):
        validate_run_manifest(tampered)


def test_main_skip_helper_reuses_only_strict_matching_manifest(tmp_path):
    manifest, artifact, inventory, output = _build_complete_manifest(tmp_path)
    atomic_write_json(tmp_path / "run_manifest.json", manifest)
    reusable = resolve_reusable_uids(
        enabled=True,
        out_dir=tmp_path,
        expected_uids=["u1"],
        artifact_fingerprint=artifact,
        input_inventory=inventory,
    )
    assert reusable == {"u1"}

    stale_artifact = build_content_fingerprint(
        {"config": tmp_path / "config.yaml", "checkpoint": tmp_path / "model.tar"},
        parameters={"sr": 8000},
    )
    assert resolve_reusable_uids(
        enabled=True,
        out_dir=tmp_path,
        expected_uids=["u1"],
        artifact_fingerprint=stale_artifact,
        input_inventory=inventory,
    ) == set()
    assert output.is_file()


def test_main_skip_helper_rejects_legacy_manifest_without_effective_identity(tmp_path):
    manifest, _, inventory, _ = _build_complete_manifest(tmp_path)
    atomic_write_json(tmp_path / "run_manifest.json", manifest)
    current_artifact = build_content_fingerprint(
        {"config": tmp_path / "config.yaml", "checkpoint": tmp_path / "model.tar"},
        parameters={
            "sr": 16000,
            "chunk_seconds": 0.0,
            "effective_config_identity": effective_config_identity(
                {"model": "test"}
            ),
        },
    )

    assert resolve_reusable_uids(
        enabled=True,
        out_dir=tmp_path,
        expected_uids=["u1"],
        artifact_fingerprint=current_artifact,
        input_inventory=inventory,
    ) == set()


@pytest.mark.parametrize(
    "changed_asset",
    ["config", "base_config", "checkpoint", "source", "input"],
)
def test_finalize_rejects_runtime_snapshot_drift_without_replacing_manifest(
    tmp_path: Path, changed_asset: str
):
    base = tmp_path / "base.yaml"
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    source = tmp_path / "model.py"
    noisy = tmp_path / "input.wav"
    output = tmp_path / "u1.wav"
    base.write_text("stft: {n_fft: 320}\n", encoding="utf-8")
    config.write_text(
        "base_config: base.yaml\nmodel: {type: Test}\n", encoding="utf-8"
    )
    checkpoint.write_bytes(b"checkpoint-v1")
    source.write_text("VALUE = 1\n", encoding="utf-8")
    noisy.write_bytes(b"input-v1")
    output.write_bytes(b"output")
    effective, identity = load_effective_config(config)
    artifact = build_content_fingerprint(
        {"config": config, "checkpoint": checkpoint, "source:model.py": source},
        parameters={
            "resolved_config": effective,
            "effective_config_identity": identity,
        },
    )
    inventory = build_input_inventory([("u1", {"noisy": noisy})])
    manifest_path = tmp_path / "run_manifest.json"
    old_manifest = b'{"old": true}\n'
    manifest_path.write_bytes(old_manifest)

    if changed_asset == "config":
        config.write_text(
            "base_config: base.yaml\nmodel: {type: Changed}\n", encoding="utf-8"
        )
    elif changed_asset == "base_config":
        base.write_text("stft: {n_fft: 512}\n", encoding="utf-8")
    elif changed_asset == "checkpoint":
        checkpoint.write_bytes(b"checkpoint-v2")
    elif changed_asset == "source":
        source.write_text("VALUE = 2\n", encoding="utf-8")
    else:
        noisy.write_bytes(b"input-v2-longer")

    with pytest.raises(RuntimeError, match="发布前|effective config identity mismatch"):
        finalize_inference_artifacts(
            config=config,
            allow_partial=False,
            out_dir=tmp_path,
            expected_uids=["u1"],
            failed_uids=[],
            skipped_uids=[],
            artifact_fingerprint=artifact,
            input_inventory=inventory,
        )

    assert manifest_path.read_bytes() == old_manifest


def test_extracted_inference_components_write_standard_artifacts(tmp_path):
    from evaluate.inference_components import configured_checkpoint_files

    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    nested_checkpoint = tmp_path / "frontend.tar"
    noisy = tmp_path / "input.wav"
    output = tmp_path / "u1.wav"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    nested_checkpoint.write_bytes(b"frontend")
    noisy.write_bytes(b"input")
    output.write_bytes(b"output")
    assert configured_checkpoint_files(
        {"frontend": {"filter_ckpt_path": str(nested_checkpoint)}}
    ) == {"config.frontend.filter_ckpt_path": nested_checkpoint}

    effective, identity = load_effective_config(config)
    artifact = build_content_fingerprint(
        {"config": config, "checkpoint": checkpoint},
        parameters={
            "resolved_config": effective,
            "effective_config_identity": identity,
        },
    )
    inventory = build_input_inventory([("u1", {"noisy": noisy})])
    results = finalize_inference_artifacts(
        config=config,
        allow_partial=False,
        out_dir=tmp_path,
        expected_uids=["u1"],
        failed_uids=[],
        skipped_uids=[],
        artifact_fingerprint=artifact,
        input_inventory=inventory,
        created_at="2026-08-12T00:00:00",
    )

    assert results[0].status == "ok"
    assert (tmp_path / "inf.scp").read_text(encoding="utf-8") == (
        f"u1 {output.resolve()}\n"
    )
    manifest, loaded_results = load_run_manifest(tmp_path / "run_manifest.json")
    assert manifest["expected_uids"] == ["u1"]
    assert loaded_results[0].output_identity == stat_audio_identity(output)


def test_waveform_native_artifacts_are_rechecked_after_model_load(tmp_path):
    from evaluate.inference_components import (
        configured_checkpoint_files,
        verify_configured_checkpoint_fingerprints,
    )

    config = tmp_path / "config.yaml"
    encoder = tmp_path / "encoder.pt"
    decoder = tmp_path / "decoder.pt"
    config.write_text("model: waveform\n", encoding="utf-8")
    encoder.write_bytes(b"encoder-v1")
    decoder.write_bytes(b"decoder-v1")
    model_cfg = {
        "encoder_ckpt_path": str(encoder),
        "nested": {"decoder_ckpt_path": str(decoder)},
    }
    checkpoint_files = configured_checkpoint_files(model_cfg)
    artifact = build_content_fingerprint({"config": config, **checkpoint_files})

    verify_configured_checkpoint_fingerprints(checkpoint_files, artifact)
    decoder.write_bytes(b"decoder-v2")

    with pytest.raises(RuntimeError, match="加载期间内容发生变化"):
        verify_configured_checkpoint_fingerprints(checkpoint_files, artifact)


def test_root_model_loader_receives_precomputed_checkpoint_fingerprint(
    monkeypatch,
    tmp_path,
):
    import inference
    import core.models.build as build_module

    checkpoint = tmp_path / "model.tar"
    checkpoint.write_bytes(b"checkpoint")
    expected = fingerprint_file(checkpoint)
    captured = {}
    model = inference.torch.nn.Linear(2, 2)
    monkeypatch.setattr(build_module, "build_model", lambda *args, **kwargs: model)

    def _load_checkpoint(target, path, **kwargs):
        captured.update(kwargs)
        return target, {"model": target.state_dict()}

    monkeypatch.setattr(inference, "load_checkpoint", _load_checkpoint)

    loaded = inference._load_model(
        {"model": {"type": "GTCRN"}},
        checkpoint,
        inference.torch.device("cpu"),
        expected_checkpoint_fingerprint=expected,
    )

    assert loaded is model
    assert captured["expected_fingerprint"] == expected


def test_cli_defaults_fail_closed_and_allow_partial_is_opt_in(monkeypatch, tmp_path):
    import inference

    argv = [
        "inference.py",
        "-C",
        str(tmp_path / "config.yaml"),
        "--output_dir",
        str(tmp_path / "out"),
    ]
    monkeypatch.setattr("sys.argv", argv)
    defaults = inference.parse_args()
    assert defaults.allow_partial is False
    assert defaults.aec_review_multichannel is False

    monkeypatch.setattr("sys.argv", [*argv, "--allow-partial"])
    assert inference.parse_args().allow_partial is True

    monkeypatch.setattr("sys.argv", [*argv, "--aec-review-multichannel"])
    assert inference.parse_args().aec_review_multichannel is True

    monkeypatch.setattr("sys.argv", [*argv, "--allow-partial-load"])
    with pytest.raises(SystemExit):
        inference.parse_args()


pytestmark = [pytest.mark.unit]


def test_output_lock_fails_fast_when_already_held(tmp_path):
    """同进程二次获取持久锁必须 fail-fast 而非阻塞（与 TSE 语义一致）。"""
    with inference_output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="已有活跃 writer"):
            with inference_output_lock(tmp_path):
                pass


def test_output_lock_rejects_symlink_leaf(tmp_path):
    """锁文件为 symlink 时必须拒绝，避免 flock 落到链接目标的另一个 inode。"""
    target = tmp_path / "other.lock"
    target.write_text("", encoding="utf-8")
    (tmp_path / ".inference.lock").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        with inference_output_lock(tmp_path):
            pass
