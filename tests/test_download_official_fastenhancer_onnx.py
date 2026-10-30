"""官方 FastEnhancer 下载器的 TLS 安全合同测试。"""

from __future__ import annotations

import hashlib
import importlib.util
import ssl
import sys
import urllib.request
from pathlib import Path

import pytest

from onnx.verify import verify_common


pytestmark = pytest.mark.unit

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "onnx"
    / "download_official_fastenhancer_onnx.py"
)


class _FakeResponse:
    def __init__(self, payload=b"fake onnx payload", *, final_url="https://objects.example/model.onnx", content_length=None):
        self.payload = payload
        self.offset = 0
        self.final_url = final_url
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def geturl(self):
        return self.final_url

    def read(self, size=-1):
        if self.offset >= len(self.payload):
            return b""
        end = len(self.payload) if size < 0 else min(len(self.payload), self.offset + size)
        chunk = self.payload[self.offset:end]
        self.offset = end
        return chunk


def _load_downloader():
    spec = importlib.util.spec_from_file_location(
        "download_official_fastenhancer_onnx", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_does_not_use_custom_ssl_context(monkeypatch, tmp_path):
    downloader = _load_downloader()
    payload = b"fake onnx payload"
    calls = {}
    monkeypatch.setattr(verify_common, "_ONNX_RELEASE_ROOT", tmp_path)
    downloader._OFFICIAL_RELEASES["dns"]["waveform"]["b"] = (
        hashlib.sha256(payload).hexdigest(),
        len(payload),
    )

    def fail_create_default_context(*args, **kwargs):
        raise AssertionError("download script must not create a custom SSL context")

    def fake_urlopen(request, **kwargs):
        calls["request"] = request
        calls["kwargs"] = kwargs
        assert "context" not in kwargs
        return _FakeResponse(payload, content_length=len(payload))

    monkeypatch.setattr(ssl, "create_default_context", fail_create_default_context)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_SCRIPT_PATH),
            "--variant", "B",
            "--dataset", "dns",
            "--output-dir", str(tmp_path),
        ],
    )

    assert downloader.main() == 0
    assert calls["kwargs"] == {"timeout": 120}
    assert calls["request"].full_url == (
        "https://github.com/aask1357/fastenhancer/releases/download/"
        "onnx-dns-v1.0.0/fastenhancer_b.onnx"
    )
    assert (tmp_path / "fastenhancer_b_official.onnx").read_bytes() == payload
    assert any(path.suffix == ".lock" for path in tmp_path.iterdir())


@pytest.mark.parametrize("dataset", ["vctk-demand", "dns", "48khz"])
def test_official_manifest_pins_all_variants_and_modes(dataset):
    downloader = _load_downloader()
    release = downloader._OFFICIAL_RELEASES[dataset]
    assert release["tag"]
    for mode in ("waveform", "spec"):
        assert set(release[mode]) == {"t", "b", "s", "m", "l"}
        for digest, size_bytes in release[mode].values():
            assert len(digest) == 64
            assert set(digest) <= set("0123456789abcdef")
            assert size_bytes > 0


def test_download_requires_pinned_digest_and_size(tmp_path):
    downloader = _load_downloader()
    with pytest.raises(TypeError):
        downloader._download_atomic("https://example/model.onnx", tmp_path / "model.onnx")


def test_download_rejects_final_non_https_redirect_and_preserves_target(monkeypatch, tmp_path):
    downloader = _load_downloader()
    target = tmp_path / "model.onnx"
    target.write_bytes(b"old")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(final_url="http://objects.example/model.onnx"),
    )

    with pytest.raises(RuntimeError, match="non-HTTPS"):
        downloader._download_atomic(
            "https://example/model.onnx",
            target,
            expected_sha256="0" * 64,
            expected_size_bytes=1,
        )
    assert target.read_bytes() == b"old"


def test_download_rejects_pinned_size_mismatch_and_preserves_target(monkeypatch, tmp_path):
    downloader = _load_downloader()
    target = tmp_path / "model.onnx"
    target.write_bytes(b"old")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(b"short", content_length=10),
    )

    with pytest.raises(RuntimeError, match="Pinned size mismatch"):
        downloader._download_atomic(
            "https://example/model.onnx",
            target,
            expected_sha256=hashlib.sha256(b"short").hexdigest(),
            expected_size_bytes=5,
        )
    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob("*.tmp")) == []


def test_download_sha256_gate_and_shared_transaction_publish(monkeypatch, tmp_path):
    downloader = _load_downloader()
    payload = b"complete onnx payload"
    target = tmp_path / "model.onnx"
    target.write_bytes(b"old")
    monkeypatch.setattr(verify_common, "_ONNX_RELEASE_ROOT", tmp_path)

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(payload, content_length=len(payload)),
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        downloader._download_atomic(
            "https://example/model.onnx",
            target,
            expected_sha256="0" * 64,
            expected_size_bytes=len(payload),
        )
    assert target.read_bytes() == b"old"

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(payload, content_length=len(payload)),
    )
    result = downloader._download_atomic(
        "https://example/model.onnx",
        target,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size_bytes=len(payload),
    )
    assert result["size_bytes"] == len(payload)
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert target.read_bytes() == payload
    assert any(path.suffix == ".lock" for path in tmp_path.iterdir())


def test_download_cleanup_failure_is_visible(monkeypatch, tmp_path):
    downloader = _load_downloader()
    payload = b"bad digest"
    target = tmp_path / "model.onnx"
    real_unlink = Path.unlink
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(payload, content_length=len(payload)),
    )

    def retain_download(self, *args, **kwargs):
        if self.suffix == ".tmp":
            raise OSError("injected cleanup failure")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", retain_download)

    with pytest.raises(RuntimeError, match="download_cleanup=retained"):
        downloader._download_atomic(
            "https://example/model.onnx",
            target,
            expected_sha256="0" * 64,
            expected_size_bytes=len(payload),
        )


def test_cli_prints_cleanup_state(monkeypatch, tmp_path, capsys):
    downloader = _load_downloader()
    monkeypatch.setattr(verify_common, "_ONNX_RELEASE_ROOT", tmp_path)
    digest, size_bytes = downloader._OFFICIAL_RELEASES["dns"]["waveform"]["b"]
    monkeypatch.setattr(
        downloader,
        "_download_atomic",
        lambda *_args, **_kwargs: {
            "path": str(tmp_path / "model.onnx"),
            "sha256": digest,
            "size_bytes": size_bytes,
            "backup_cleanup": "retained",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(_SCRIPT_PATH), "--dataset", "dns", "--output-dir", str(tmp_path)],
    )

    assert downloader.main() == 0
    assert "cleanup={'backup_cleanup': 'retained'}" in capsys.readouterr().out
