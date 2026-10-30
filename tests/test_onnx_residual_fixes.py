"""ONNX 残留修复的回归测试。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from onnx.verify import verify_common


pytestmark = pytest.mark.unit

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "onnx"
    / "download_official_fastenhancer_onnx.py"
)


def _load_downloader():
    spec = importlib.util.spec_from_file_location(
        "download_official_fastenhancer_onnx_residual", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fastenhancer_reconstruction_uses_artifact_center(monkeypatch):
    from onnx.infer import infer_streaming_fastenhancer_ort as fastenhancer

    captured = {}

    def fake_istft(spectra, **kwargs):
        captured["spectra"] = spectra
        captured.update(kwargs)
        return torch.zeros(1, kwargs["length"])

    monkeypatch.setattr(fastenhancer.torch, "istft", fake_istft)
    result = fastenhancer._reconstruct_waveform(
        torch.zeros(1, 9, 2, 2),
        n_fft=16,
        hop_length=8,
        win_length=16,
        window=torch.hann_window(16),
        center=False,
        length=24,
    )

    assert result.shape == (1, 24)
    assert torch.is_complex(captured["spectra"])
    assert captured["center"] is False


def test_cli_rejects_output_dir_outside_release_root(monkeypatch, tmp_path, capsys):
    downloader = _load_downloader()
    release_root = tmp_path / "release"
    release_root.mkdir()
    outside_dir = tmp_path / "outside"
    download_called = False
    monkeypatch.setattr(verify_common, "_ONNX_RELEASE_ROOT", release_root)

    def fail_download(*_args, **_kwargs):
        nonlocal download_called
        download_called = True
        raise AssertionError("output directory must be rejected before download")

    monkeypatch.setattr(downloader, "_download_atomic", fail_download)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(_SCRIPT_PATH), "--output-dir", str(outside_dir)],
    )

    with pytest.raises(SystemExit) as exc_info:
        downloader.main()

    assert exc_info.value.code == 2
    assert "must be under the ONNX release root" in capsys.readouterr().err
    assert not download_called
    assert not outside_dir.exists()


def test_validate_output_dir_allows_release_root_subdirectory(monkeypatch, tmp_path):
    downloader = _load_downloader()
    release_root = tmp_path / "release"
    release_root.mkdir()
    target_dir = release_root / "official" / "fastenhancer"
    monkeypatch.setattr(verify_common, "_ONNX_RELEASE_ROOT", release_root)

    assert downloader._validate_output_dir(target_dir) == target_dir
