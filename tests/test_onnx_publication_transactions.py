"""ONNX 发布绑定实际加载来源，并以同一锁提交图和 manifest。"""

import numpy as np
import pytest
import torch

from core.utils.artifact_identity import SourceBinding
from onnx.verify import verify_common as common
from onnx.verify.manifest import inspect_onnx_manifest_reuse, load_onnx_manifest

pytestmark = pytest.mark.unit


def _graph(path, operation):
    from onnx.utils import pypi_onnx_scope

    with pypi_onnx_scope() as onnx:
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node(operation, ["x"], ["y"])], operation,
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
        model.ir_version = 9
        onnx.save(model, path)


def _spec(tmp_path, name):
    checkpoint = tmp_path / f"{name}.pt"
    checkpoint.write_bytes(name.encode())
    files = {"checkpoint": checkpoint}
    return {
        "model_id": "fixture",
        "source_files": files,
        "source_binding": SourceBinding.bind(files, parameters={"model_id": "fixture"}).as_dict(),
        "runtime_contract": {"operation": name},
    }


def _artifact(tmp_path, name):
    path = tmp_path / f"{name}.onnx"
    _graph(path, name)
    artifact = common.VerifiedOnnxArtifact(path)
    session = artifact.create_session()
    expected = 2.0 if name == "Identity" else -2.0
    np.testing.assert_array_equal(session.run(None, {"x": np.array([2.0], np.float32)})[0], [expected])
    return artifact


def test_loaded_source_change_cannot_be_rebound_at_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_ONNX_RELEASE_ROOT", tmp_path)
    spec = _spec(tmp_path, "Identity")
    artifact = _artifact(tmp_path, "Identity")
    target = tmp_path / "released.onnx"
    spec["source_files"]["checkpoint"].write_bytes(b"changed-after-load")
    try:
        with pytest.raises(RuntimeError, match="source artifact changed"):
            artifact.publish(target, all_pass=True, manifest=spec, require_manifest=True)
        assert not target.exists()
    finally:
        artifact.close()


def test_actual_loader_snapshot_rejects_checkpoint_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_ONNX_RELEASE_ROOT", tmp_path)

    class TinyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(1, 1)

    checkpoint = tmp_path / "loaded.tar"
    model = TinyNet()
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    source_binding = common.load_checkpoint_strict(
        TinyNet(), checkpoint, return_source_binding=True, model_id="fixture"
    )
    checkpoint.write_bytes(b"replacement-B")
    artifact = _artifact(tmp_path, "Identity")
    try:
        with pytest.raises(RuntimeError, match="source artifact changed"):
            artifact.publish(
                tmp_path / "released.onnx",
                all_pass=True,
                manifest={
                    "model_id": "fixture",
                    "source_files": {"checkpoint": checkpoint},
                    "source_binding": source_binding,
                    "runtime_contract": {"operation": "Identity"},
                },
                require_manifest=True,
            )
    finally:
        artifact.close()


def test_real_ort_interleaving_does_not_bind_neg_graph_to_identity_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_ONNX_RELEASE_ROOT", tmp_path)
    identity = _artifact(tmp_path, "Identity")
    negative = _artifact(tmp_path, "Neg")
    spec_a = _spec(tmp_path, "Identity")
    spec_b = _spec(tmp_path, "Neg")
    target = tmp_path / "released.onnx"
    real_publish = common.publish_verified_onnx

    def interleave(*args, **kwargs):
        result = real_publish(*args, **kwargs)
        negative.publish(target, all_pass=True, manifest=spec_b, require_manifest=True)
        return result

    try:
        identity.publish(target, all_pass=True, manifest=spec_a, require_manifest=True, publisher=interleave)
        from onnx.utils import create_ort_session

        session = create_ort_session(target)
        np.testing.assert_array_equal(session.run(None, {"x": np.array([2.0], np.float32)})[0], [-2.0])
        assert load_onnx_manifest(target)["runtime_contract"] == spec_b["runtime_contract"]
        decision = inspect_onnx_manifest_reuse(
            target, model_id="fixture", source_files=spec_a["source_files"],
            runtime_contract=spec_a["runtime_contract"],
        )
        assert not decision.reusable
    finally:
        identity.close()
        negative.close()


def test_manifest_commit_failure_rolls_back_onnx_and_contract_together(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_ONNX_RELEASE_ROOT", tmp_path)
    identity = _artifact(tmp_path, "Identity")
    negative = _artifact(tmp_path, "Neg")
    spec_a = _spec(tmp_path, "Identity")
    spec_b = _spec(tmp_path, "Neg")
    target = tmp_path / "released.onnx"
    try:
        identity.publish(target, all_pass=True, manifest=spec_a, require_manifest=True)
        real_exchange = common._rename_exchange
        calls = 0

        def fail_manifest_exchange(parent_fd, left_name, right_name):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected manifest commit failure")
            return real_exchange(parent_fd, left_name, right_name)

        monkeypatch.setattr(common, "_rename_exchange", fail_manifest_exchange)
        with pytest.raises(OSError, match="manifest commit failure"):
            negative.publish(target, all_pass=True, manifest=spec_b, require_manifest=True)

        from onnx.utils import create_ort_session

        session = create_ort_session(target)
        np.testing.assert_array_equal(session.run(None, {"x": np.array([2.0], np.float32)})[0], [2.0])
        assert load_onnx_manifest(target)["runtime_contract"] == spec_a["runtime_contract"]
    finally:
        identity.close()
        negative.close()


def test_manifest_required_rejects_publisher_without_manifest_keyword(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_ONNX_RELEASE_ROOT", tmp_path)
    artifact = _artifact(tmp_path, "Identity")
    spec = _spec(tmp_path, "Identity")
    target = tmp_path / "released.onnx"
    target.write_bytes(b"previous")

    def publisher_without_manifest(
        source,
        destination,
        *,
        all_pass,
        expected_sha256,
        expected_size_bytes,
        expected_device,
        expected_inode,
        source_fd,
    ):
        raise AssertionError("publisher must not be invoked without manifest capability")

    try:
        with pytest.raises(TypeError, match="manifest"):
            artifact.publish(
                target,
                all_pass=True,
                publisher=publisher_without_manifest,
                manifest=spec,
                require_manifest=True,
            )
        assert target.read_bytes() == b"previous"
        assert not (tmp_path / "released.onnx.manifest.json").exists()
    finally:
        artifact.close()


def test_manifest_capable_kwargs_publisher_commits_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_ONNX_RELEASE_ROOT", tmp_path)
    artifact = _artifact(tmp_path, "Identity")
    spec = _spec(tmp_path, "Identity")
    target = tmp_path / "released.onnx"
    calls = []

    def kwargs_publisher(source, destination, **kwargs):
        calls.append(kwargs["manifest"])
        return common.publish_verified_onnx(source, destination, **kwargs)

    try:
        result = artifact.publish(
            target,
            all_pass=True,
            publisher=kwargs_publisher,
            manifest=spec,
            require_manifest=True,
        )
        assert result["manifest"].endswith("released.onnx.manifest.json")
        assert calls == [spec]
        assert load_onnx_manifest(target)["model_id"] == "fixture"
    finally:
        artifact.close()


def test_manifest_argument_without_flag_is_formal_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_ONNX_RELEASE_ROOT", tmp_path)
    artifact = _artifact(tmp_path, "Identity")
    spec = _spec(tmp_path, "Identity")
    target = tmp_path / "released.onnx"
    try:
        result = artifact.publish(
            target,
            all_pass=True,
            manifest=spec,
            require_manifest=False,
        )
        assert result["manifest"].endswith("released.onnx.manifest.json")
        assert load_onnx_manifest(target)["model_id"] == "fixture"
    finally:
        artifact.close()
