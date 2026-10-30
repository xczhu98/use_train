"""SSL 外部数据下载器的完整性与 archive 安全合同。"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import io
import tarfile
import zipfile

import pytest

from tools.prepare_data_tools.generators.ssl import download_data


pytestmark = pytest.mark.unit


class _FakeResponse:
    def __init__(self, payload: bytes, *, content_length: int | None = None) -> None:
        self._stream = io.BytesIO(payload)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def close(self) -> None:
        self.closed = True


def test_https_validation_and_redirect_downgrade() -> None:
    assert download_data._validate_https_url("https://example.org/a") == "https://example.org/a"
    for url in (
        "http://example.org/a",
        "https://user:password@example.org/a",
        "https://example.org/a#fragment",
    ):
        with pytest.raises(ValueError):
            download_data._validate_https_url(url)

    handler = download_data._HTTPSRedirectHandler()
    request = download_data.Request("https://example.org/a")
    with pytest.raises(ValueError, match="only HTTPS"):
        handler.redirect_request(request, None, 302, "redirect", {}, "http://example.org/b")
    redirected = handler.redirect_request(request, None, 302, "redirect", {}, "/b")
    assert redirected is not None
    assert redirected.full_url == "https://example.org/b"


def test_custom_cli_requires_expected_hash() -> None:
    with pytest.raises(SystemExit):
        download_data.main(
            [
                "--download-url",
                "https://example.org/resource.bin",
                "--output",
                "resource.bin",
            ]
        )


def test_download_is_atomic_hash_checked_and_reuses_verified_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"ssl-data-payload"
    expected = sha256(payload).hexdigest()
    target = tmp_path / "resource.bin"
    opened = 0

    def fake_open(url: str, *, timeout: float) -> _FakeResponse:
        nonlocal opened
        opened += 1
        assert url == "https://example.org/resource.bin"
        return _FakeResponse(payload, content_length=len(payload))

    monkeypatch.setattr(download_data, "_open_url", fake_open)
    result = download_data.download_resource(
        "https://example.org/resource.bin",
        target,
        expected_sha256=expected,
        expected_bytes=len(payload),
    )
    assert result.reused is False
    assert result.sha256 == expected
    assert target.read_bytes() == payload
    assert opened == 1
    assert not list(tmp_path.glob("*.download"))

    def fail_open(*args, **kwargs):
        raise AssertionError("verified existing file should be reused")

    monkeypatch.setattr(download_data, "_open_url", fail_open)
    reused = download_data.download_resource(
        "https://example.org/resource.bin",
        target,
        expected_sha256=expected,
        expected_bytes=len(payload),
    )
    assert reused.reused is True
    assert reused.sha256 == expected


def test_download_rejects_hash_and_size_mismatch_without_publishing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"payload"
    monkeypatch.setattr(
        download_data,
        "_open_url",
        lambda url, *, timeout: _FakeResponse(payload, content_length=len(payload)),
    )
    target = tmp_path / "resource.bin"
    with pytest.raises(download_data.DownloadError, match="SHA256 mismatch"):
        download_data.download_resource("https://example.org/resource.bin", target, expected_sha256="0" * 64)
    assert not target.exists()
    assert not list(tmp_path.glob("*.download"))

    with pytest.raises(download_data.DownloadError, match="Content-Length mismatch"):
        download_data.download_resource(
            "https://example.org/resource.bin", target, expected_bytes=len(payload) + 1
        )
    assert not target.exists()


def test_safe_extract_rejects_traversal_and_symlinks(tmp_path: Path) -> None:
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../outside.txt", b"blocked")
    destination = tmp_path / "extract"
    with pytest.raises(download_data.DownloadError, match="escapes"):
        download_data.safe_extract_archive(traversal, destination)
    assert not destination.exists()
    assert not (tmp_path / "outside.txt").exists()

    symlink_archive = tmp_path / "symlink.tar"
    with tarfile.open(symlink_archive, "w") as archive:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
    with pytest.raises(download_data.DownloadError, match="link"):
        download_data.safe_extract_archive(symlink_archive, tmp_path / "tar-extract")


def test_noise92_mat_conversion_is_mono_and_16k(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    scipy_io = pytest.importorskip("scipy.io")
    sf = pytest.importorskip("soundfile")
    source = tmp_path / "noise.mat"
    scipy_io.savemat(source, {"noise": np.linspace(-2.0, 2.0, 19_980, dtype=np.float32)})
    output = tmp_path / "noise.wav"
    metadata = download_data.convert_noise92_mat_to_wav(source, output)
    info = sf.info(output)
    assert metadata["sample_rate"] == 16_000
    assert info.samplerate == 16_000
    assert info.channels == 1
    assert info.frames > 0


def test_noise92_cli_emits_four_column_wav_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(url: str, output: Path, **kwargs) -> download_data.DownloadResult:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"mat")
        return download_data.DownloadResult(
            path=str(output),
            url=url,
            bytes=3,
            sha256=sha256(b"mat").hexdigest(),
            reused=False,
            expected_sha256=None,
            content_length=3,
        )

    def fake_convert(source: Path, output: Path, **kwargs) -> dict[str, object]:
        output.write_bytes(b"wav")
        return {"path": str(output), "sample_rate": 16000, "frames": 12, "sha256": sha256(b"wav").hexdigest()}

    monkeypatch.setattr(download_data, "download_resource", fake_download)
    monkeypatch.setattr(download_data, "convert_noise92_mat_to_wav", fake_convert)
    assert download_data.main(["--download-noise92", "babble", "--convert-wav", "--output-dir", str(tmp_path)]) == 0
    scp = (tmp_path / "noise92.scp").read_text(encoding="utf-8")
    assert scp == "noise92:babble 16000 12 babble.wav\n"
    manifest = (tmp_path / "download_manifest.json").read_text(encoding="utf-8")
    assert '"wav_manifest"' in manifest
