"""官方 TAPLoss checkpoint 安全转换器的 hermetic 回归测试。"""

from __future__ import annotations

import ast
import hashlib
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from core.losses.tap import _TAPAcousticEstimator
from core.utils.safe_load import safe_torch_load
from tools.model_tools import convert_taploss_official_checkpoint as converter

pytestmark = pytest.mark.unit


def _official_like_checkpoint_bytes() -> bytes:
    checkpoint = {
        "model_state_dict": _TAPAcousticEstimator().state_dict(),
        "optimizer_state_dict": {
            "array": np.array([1.0], dtype=np.float32),
            "scalar": np.float64(2.0),
            "dtype": np.dtype(np.float64),
        },
        "metadata": "must not survive conversion",
    }
    serialized = io.BytesIO()
    torch.save(checkpoint, serialized)

    source = zipfile.ZipFile(io.BytesIO(serialized.getvalue()), "r")
    rewritten = io.BytesIO()
    with source, zipfile.ZipFile(rewritten, "w") as target:
        for member in source.infolist():
            payload = source.read(member.filename)
            if member.filename.endswith("/data.pkl"):
                payload = payload.replace(
                    b"numpy._core.multiarray\n", b"numpy.core.multiarray\n"
                )
            target.writestr(member, payload)
    return rewritten.getvalue()


@pytest.fixture(scope="module")
def official_like_bytes() -> bytes:
    payload = _official_like_checkpoint_bytes()
    assert converter._audit_pickle_globals(payload) == converter._EXPECTED_PICKLE_GLOBALS
    return payload


def _write_source(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _run_cli_with_fixture_sha(
    source: Path,
    output: Path,
    sha256: str,
    *,
    env: dict[str, str] | None = None,
):
    code = (
        "import sys; "
        "from tools.model_tools import convert_taploss_official_checkpoint as c; "
        f"c.OFFICIAL_SHA256={sha256!r}; "
        f"sys.argv=['converter', '--input', {str(source)!r}, '--output', {str(output)!r}]; "
        "raise SystemExit(c.main())"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_conversion_isolated_safe_and_direct_tensor_only(
    tmp_path, official_like_bytes
):
    source = tmp_path / "official.pt"
    output = tmp_path / "converted.pt"
    source_sha256 = _write_source(source, official_like_bytes)
    safe_globals_before = list(torch.serialization.get_safe_globals())

    process = _run_cli_with_fixture_sha(source, output, source_sha256)

    assert process.returncode == 0, process.stderr
    assert f"source_sha256={source_sha256}" in process.stdout
    assert "output_sha256=" in process.stdout
    assert "tensor_count=28" in process.stdout
    assert "parameter_count=" in process.stdout
    assert torch.serialization.get_safe_globals() == safe_globals_before

    state_dict = safe_torch_load(output, map_location="cpu")
    assert set(state_dict) == set(_TAPAcousticEstimator().state_dict())
    assert "model_state_dict" not in state_dict
    assert all(type(key) is str for key in state_dict)
    assert all(type(value) is torch.Tensor for value in state_dict.values())
    assert all(value.device.type == "cpu" for value in state_dict.values())
    _TAPAcousticEstimator().load_state_dict(state_dict, strict=True)


def test_converter_source_contains_no_direct_torch_load() -> None:
    source_path = Path(converter.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    direct_loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "load"
    ]
    assert direct_loads == []


def test_direct_conversion_does_not_pollute_parent_safe_globals(
    tmp_path, official_like_bytes, monkeypatch
):
    source = tmp_path / "official.pt"
    output = tmp_path / "converted.pt"
    source_sha256 = _write_source(source, official_like_bytes)
    monkeypatch.setattr(converter, "OFFICIAL_SHA256", source_sha256)
    safe_globals_before = list(torch.serialization.get_safe_globals())

    converter.convert_checkpoint(source, output)

    assert torch.serialization.get_safe_globals() == safe_globals_before


def test_worker_private_temp_files_are_cleaned(
    tmp_path, official_like_bytes, monkeypatch
):
    source = tmp_path / "official.pt"
    output = tmp_path / "converted.pt"
    worker_tmp = tmp_path / "worker-tmp"
    worker_tmp.mkdir()
    source_sha256 = _write_source(source, official_like_bytes)
    env = os.environ.copy()
    env["TMPDIR"] = str(worker_tmp)

    process = _run_cli_with_fixture_sha(
        source,
        output,
        source_sha256,
        env=env,
    )

    assert process.returncode == 0, process.stderr
    assert list(worker_tmp.iterdir()) == []

    monkeypatch.setattr(converter.tempfile, "tempdir", str(worker_tmp))

    def inspect_then_fail(path, **_kwargs):
        path = Path(path)
        assert path.is_file()
        assert not path.is_symlink()
        assert path.read_bytes() == official_like_bytes
        assert path.parent.stat().st_mode & 0o777 == 0o700
        raise RuntimeError("injected")

    monkeypatch.setattr(converter, "safe_torch_load", inspect_then_fail)
    with pytest.raises(RuntimeError, match="injected"):
        converter._load_worker_checkpoint(official_like_bytes)
    assert list(worker_tmp.iterdir()) == []


def test_conversion_deserializes_the_sha_bound_bytes(
    tmp_path, official_like_bytes, monkeypatch
):
    source = tmp_path / "official.pt"
    output = tmp_path / "converted.pt"
    source_sha256 = _write_source(source, official_like_bytes)
    monkeypatch.setattr(converter, "OFFICIAL_SHA256", source_sha256)
    original_audit = converter._audit_pickle_globals

    def replace_path_after_audit(source_bytes):
        globals_found = original_audit(source_bytes)
        source.write_bytes(b"replaced after SHA and audit")
        return globals_found

    monkeypatch.setattr(converter, "_audit_pickle_globals", replace_path_after_audit)
    reported_source_sha, _output_sha, tensor_count, _parameter_count = (
        converter.convert_checkpoint(source, output)
    )

    assert reported_source_sha == source_sha256
    assert source.read_bytes() == b"replaced after SHA and audit"
    assert tensor_count == 28
    _TAPAcousticEstimator().load_state_dict(safe_torch_load(output), strict=True)


@pytest.mark.subprocess
def test_cli_rejects_non_official_sha(tmp_path):
    source = tmp_path / "wrong.pt"
    output = tmp_path / "converted.pt"
    source.write_bytes(b"not the official checkpoint")

    process = subprocess.run(
        [
            sys.executable,
            "tools/model_tools/convert_taploss_official_checkpoint.py",
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode != 0
    assert "source SHA256 mismatch" in process.stderr
    assert not output.exists()


@pytest.mark.parametrize("kind", ["symlink", "directory", "empty"])
def test_source_rejects_non_regular_or_empty(tmp_path, kind):
    source = tmp_path / "source.pt"
    if kind == "symlink":
        target = tmp_path / "target.pt"
        target.write_bytes(b"data")
        source.symlink_to(target)
    elif kind == "directory":
        source.mkdir()
    else:
        source.touch()

    with pytest.raises(converter.ConversionError, match="symlink|regular file|empty"):
        converter._read_bound_source(source)


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_output_refuses_overwrite(tmp_path, kind):
    output = tmp_path / "converted.pt"
    if kind == "file":
        output.write_bytes(b"existing")
    else:
        target = tmp_path / "target.pt"
        target.write_bytes(b"existing")
        output.symlink_to(target)

    with pytest.raises(converter.ConversionError, match="refusing to overwrite"):
        converter._validate_output_path(output)


@pytest.mark.parametrize("role", ["source", "output"])
def test_paths_reject_parent_symlink(tmp_path, role):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    path = linked_parent / ("source.pt" if role == "source" else "converted.pt")
    if role == "source":
        (real_parent / "source.pt").write_bytes(b"data")
        call = lambda: converter._read_bound_source(path)
    else:
        call = lambda: converter._validate_output_path(path)

    with pytest.raises(converter.ConversionError, match="path component.*symlink"):
        call()


@pytest.mark.parametrize("role", ["source", "output"])
def test_paths_reject_non_parent_ancestor_symlink(tmp_path, role):
    real_ancestor = tmp_path / "real-ancestor"
    nested = real_ancestor / "nested"
    nested.mkdir(parents=True)
    linked_ancestor = tmp_path / "linked-ancestor"
    linked_ancestor.symlink_to(real_ancestor, target_is_directory=True)
    path = linked_ancestor / "nested" / (
        "source.pt" if role == "source" else "converted.pt"
    )
    if role == "source":
        (nested / "source.pt").write_bytes(b"data")
        call = lambda: converter._read_bound_source(path)
    else:
        call = lambda: converter._validate_output_path(path)

    with pytest.raises(converter.ConversionError, match="path component.*symlink"):
        call()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            b"cnumpy.core.multiarray\nscalar\n",
            b"cnumpy.core.multiarray\nunknown\n",
        ),
        (
            b"cnumpy.core.multiarray\nscalar\n",
            b"cnumpy.core.multiarray\n_reconstruct\n",
        ),
    ],
)
def test_pickle_global_set_must_match_exactly(official_like_bytes, old, new):
    source = zipfile.ZipFile(io.BytesIO(official_like_bytes), "r")
    rewritten = io.BytesIO()
    with source, zipfile.ZipFile(rewritten, "w") as target:
        for member in source.infolist():
            payload = source.read(member.filename)
            if member.filename.endswith("/data.pkl"):
                changed = payload.replace(old, new)
                assert changed != payload
                payload = changed
            target.writestr(member, payload)

    with pytest.raises(converter.ConversionError, match="globals do not exactly match"):
        converter._audit_pickle_globals(rewritten.getvalue())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("count", "exactly 28"),
        ("non_string_key", "keys must all be strings"),
        ("non_tensor", "plain Tensor"),
        ("non_cpu", "must be on CPU"),
        ("shape", "shape mismatch"),
        ("dtype", "dtype mismatch"),
    ],
)
def test_state_dict_structure_is_strictly_validated(mutation, message):
    state_dict = dict(_TAPAcousticEstimator().state_dict())
    key = next(iter(state_dict))
    value = state_dict[key]
    if mutation == "count":
        state_dict.pop(key)
    elif mutation == "non_string_key":
        state_dict[1] = state_dict.pop(key)
    elif mutation == "non_tensor":
        state_dict[key] = np.zeros(value.shape, dtype=np.float32)
    elif mutation == "non_cpu":
        state_dict[key] = torch.empty_like(value, device="meta")
    elif mutation == "shape":
        state_dict[key] = value.reshape(-1)[:-1]
    else:
        state_dict[key] = value.double()

    with pytest.raises(converter.ConversionError, match=message):
        converter._validate_state_dict(state_dict)


def test_worker_failure_does_not_publish(tmp_path, official_like_bytes, monkeypatch):
    source = tmp_path / "official.pt"
    output = tmp_path / "converted.pt"
    sha256 = _write_source(source, official_like_bytes)
    monkeypatch.setattr(converter, "OFFICIAL_SHA256", sha256)
    monkeypatch.setattr(
        converter,
        "_worker_convert",
        lambda _source: (_ for _ in ()).throw(converter.ConversionError("worker failed")),
    )

    with pytest.raises(converter.ConversionError, match="worker failed"):
        converter.convert_checkpoint(source, output)

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.tmp-*"))


def test_atomic_publish_failure_cleans_sibling_temp(tmp_path, monkeypatch):
    output = tmp_path / "converted.pt"

    def fail_link(_source, _target):
        raise OSError("injected link failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(converter.ConversionError, match="atomically publish"):
        converter._publish_and_verify(b"output", output)

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.tmp-*"))


def test_post_publish_safe_reload_failure_removes_output(tmp_path):
    output = tmp_path / "converted.pt"

    with pytest.raises(Exception):
        converter._publish_and_verify(b"not a torch checkpoint", output)

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.tmp-*"))
