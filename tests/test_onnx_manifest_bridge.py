"""ONNX 发布 manifest 与 core artifact binding 的兼容桥接测试。"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.unit


def test_onnx_manifest_binds_source_and_runtime_contract(tmp_path):
    from onnx.verify.manifest import (
        build_onnx_manifest,
        load_onnx_manifest,
        validate_onnx_manifest_spec,
        write_onnx_manifest,
    )

    checkpoint = tmp_path / "model.pt"
    config = tmp_path / "resolved.yaml"
    target = tmp_path / "model.onnx"
    checkpoint.write_bytes(b"checkpoint-v1")
    config.write_text("model: gtcrn\n", encoding="utf-8")
    target.write_bytes(b"onnx-v1")
    contract = {"model_id": "gtcrn", "runtime_spec_version": 1, "cache": ["conv_cache"]}
    manifest = build_onnx_manifest(
        target=target,
        model_id="gtcrn",
        source_files={"checkpoint": checkpoint, "config": config},
        runtime_contract=contract,
    )
    write_onnx_manifest(manifest)
    assert load_onnx_manifest(target)["model_id"] == "gtcrn"
    assert load_onnx_manifest(target)["source_binding"] == validate_onnx_manifest_spec(
        model_id="gtcrn",
        source_files={"checkpoint": checkpoint, "config": config},
        runtime_contract=contract,
    )

    checkpoint.write_bytes(b"checkpoint-v2")
    assert load_onnx_manifest(target)["source_binding"] != validate_onnx_manifest_spec(
        model_id="gtcrn",
        source_files={"checkpoint": checkpoint, "config": config},
        runtime_contract=contract,
    )


def test_onnx_manifest_rejects_missing_or_changed_target(tmp_path):
    from onnx.verify.manifest import (
        build_onnx_manifest,
        inspect_onnx_manifest_reuse,
        write_onnx_manifest,
    )

    source = tmp_path / "source.pt"
    target = tmp_path / "model.onnx"
    source.write_bytes(b"source")
    target.write_bytes(b"model")
    manifest = build_onnx_manifest(
        target=target,
        model_id="tse",
        source_files={"checkpoint": source},
        runtime_contract={"model_id": "tse", "cache_abi": 1},
    )
    write_onnx_manifest(manifest)
    target.write_bytes(b"tampered")
    decision = inspect_onnx_manifest_reuse(
        target, model_id="tse", source_files={"checkpoint": source},
        runtime_contract={"model_id": "tse", "cache_abi": 1},
    )
    assert not decision.reusable
    assert decision.reason == "artifact_mismatch"


def test_onnx_manifest_reuse_decision_exposes_fail_closed_reason(tmp_path):
    from onnx.verify.manifest import (
        build_onnx_manifest,
        inspect_onnx_manifest_reuse,
        require_reusable_onnx_manifest,
        write_onnx_manifest,
    )

    source = tmp_path / "checkpoint.pt"
    target = tmp_path / "model.onnx"
    source.write_bytes(b"checkpoint")
    target.write_bytes(b"onnx")
    contract = {"model_id": "gtcrn", "runtime": 1}
    write_onnx_manifest(
        build_onnx_manifest(
            target=target,
            model_id="gtcrn",
            source_files={"checkpoint": source},
            runtime_contract=contract,
        )
    )

    decision = inspect_onnx_manifest_reuse(
        target,
        model_id="gtcrn",
        source_files={"checkpoint": source},
        runtime_contract=contract,
    )
    assert decision.reusable
    assert decision.reason == "reusable"
    assert decision.manifest is not None
    assert require_reusable_onnx_manifest(
        target,
        model_id="gtcrn",
        source_files={"checkpoint": source},
        runtime_contract=contract,
    )["model_id"] == "gtcrn"

    source.write_bytes(b"changed")
    decision = inspect_onnx_manifest_reuse(
        target,
        model_id="gtcrn",
        source_files={"checkpoint": source},
        runtime_contract=contract,
    )
    assert not decision.reusable
    assert decision.reason == "source_binding_mismatch"
    with pytest.raises(RuntimeError, match="source_binding_mismatch"):
        require_reusable_onnx_manifest(
            target,
            model_id="gtcrn",
            source_files={"checkpoint": source},
            runtime_contract=contract,
        )


def test_formal_verifiers_use_one_strict_manifest_publish_contract():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "onnx" / "verify"
    verifier_names = {
        "cagcrn", "dbgnet", "deepvqe", "fastenhancer",
        "gtcrn", "tse", "ultra_dpc", "ulunas",
    }
    for name in verifier_names:
        text = (root / f"verify_3way_{name}.py").read_text(encoding="utf-8")
        assert "require_manifest=True" in text, name
        assert '"source_files":' in text, name
        assert '"runtime_contract": runtime_contract_from_spec' in text, name
        assert '"model_id":' in text, name


def test_missing_manifest_is_not_reusable(tmp_path):
    from onnx.verify.manifest import inspect_onnx_manifest_reuse

    decision = inspect_onnx_manifest_reuse(
        tmp_path / "missing.onnx",
        model_id="gtcrn",
        source_files={},
        runtime_contract={"runtime": 1},
    )
    assert not decision.reusable
    assert decision.reason == "manifest_unavailable"


def test_onnx_manifest_uses_shared_artifact_identity(tmp_path):
    from core.utils.artifact_identity import ArtifactIdentity, SourceBinding
    from onnx.verify.manifest import build_onnx_manifest

    source = tmp_path / "source.pt"
    target = tmp_path / "model.onnx"
    source.write_bytes(b"source")
    target.write_bytes(b"onnx")
    identity = ArtifactIdentity.from_path(source)
    binding = SourceBinding.bind({"checkpoint": source}, parameters={"model_id": "x"})
    manifest = build_onnx_manifest(
        target=target,
        model_id="x",
        source_files={"checkpoint": source},
        runtime_contract={"runtime": 1},
    )
    assert binding.as_dict() == manifest["source_binding"]
    assert identity.as_dict() == manifest["source_binding"]["files"]["checkpoint"]
