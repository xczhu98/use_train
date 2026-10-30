"""safe_load 与 inference artifact 输出边界的安全回归测试。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from core.utils.inference_artifacts import atomic_write_audio, atomic_write_text
from core.utils.safe_load import safe_torch_load


class _NumpyCtypesLoadLibraryFixture:
    """仅序列化危险 numpy callable，安全加载不应执行或 allowlist 它。"""

    def __reduce__(self):
        import numpy.ctypeslib

        return numpy.ctypeslib.load_library, ("never-load", "/")


@pytest.mark.unit
def test_safe_load_rejects_numpy_ctypeslib_global_without_allowlist_pollution(tmp_path):
    checkpoint = tmp_path / "malicious_numpy.pt"
    torch.save({"payload": _NumpyCtypesLoadLibraryFixture()}, checkpoint)
    before = list(torch.serialization.get_safe_globals())

    with pytest.raises(RuntimeError, match="numpy.ctypeslib.load_library"):
        safe_torch_load(checkpoint)

    assert torch.serialization.get_safe_globals() == before


@pytest.mark.unit
def test_safe_load_scoped_numpy_globals_do_not_pollute_process_allowlist(tmp_path):
    if not hasattr(torch.serialization, "safe_globals"):
        pytest.skip("当前 PyTorch 不支持 scoped safe_globals")
    numpy = pytest.importorskip("numpy")
    checkpoint = tmp_path / "numpy_array.pt"
    torch.save({"array": numpy.array([1, 2, 3], dtype=numpy.int64)}, checkpoint)
    before = list(torch.serialization.get_safe_globals())

    loaded = safe_torch_load(checkpoint)

    assert loaded["array"].tolist() == [1, 2, 3]
    assert torch.serialization.get_safe_globals() == before


@pytest.mark.unit
def test_safe_load_accepts_torch_version_metadata_without_global_pollution(tmp_path):
    if not hasattr(torch.serialization, "safe_globals"):
        pytest.skip("当前 PyTorch 不支持 scoped safe_globals")
    from torch.torch_version import TorchVersion

    checkpoint = tmp_path / "torch_version.pt"
    torch.save({"version": TorchVersion(torch.__version__)}, checkpoint)
    before = list(torch.serialization.get_safe_globals())

    loaded = safe_torch_load(checkpoint)

    assert str(loaded["version"]) == str(torch.__version__)
    assert torch.serialization.get_safe_globals() == before


@pytest.mark.unit
@pytest.mark.parametrize(
    ("write", "target_name"),
    [
        (lambda target: atomic_write_audio(target, lambda staging: staging.write_bytes(b"audio")), "audio.wav"),
        (lambda target: atomic_write_text(target, "metadata\n"), "inf.scp"),
    ],
)
def test_atomic_writes_reject_leaf_symlink(tmp_path, write, target_name):
    outside = tmp_path / "outside"
    outside.write_text("preserve", encoding="utf-8")
    target = tmp_path / target_name
    target.symlink_to(outside)

    with pytest.raises(RuntimeError, match="symbolic links"):
        write(target)

    assert outside.read_text(encoding="utf-8") == "preserve"
    assert target.is_symlink()


@pytest.mark.unit
def test_atomic_write_text_rejects_ancestor_symlink(tmp_path):
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symbolic-link ancestors"):
        atomic_write_text(linked_dir / "inf.scp", "metadata\n")

    assert not (outside_dir / "inf.scp").exists()


@pytest.mark.unit
def test_atomic_audio_write_rejects_target_replaced_with_symlink(tmp_path):
    target = tmp_path / "audio.wav"
    target.write_bytes(b"old")
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"preserve")

    def replace_target_during_write(staging: Path) -> None:
        staging.write_bytes(b"new")
        target.unlink()
        target.symlink_to(outside)

    with pytest.raises(RuntimeError, match="symbolic links"):
        atomic_write_audio(target, replace_target_during_write)

    assert outside.read_bytes() == b"preserve"
    assert target.is_symlink()
    assert os.readlink(target) == str(outside)
