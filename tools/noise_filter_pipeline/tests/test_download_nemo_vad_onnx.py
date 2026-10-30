"""NeMo VAD ONNX 下载脚本的离线安全回归测试。"""

import hashlib
import io
import sys
import types
import urllib.request

import pytest

from noise_filter_pipeline.scripts import download_nemo_vad_onnx as downloader


pytestmark = pytest.mark.unit


class _FakeResponse:
    def __init__(self, payload, content_length=None):
        self._stream = io.BytesIO(payload)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, size):
        return self._stream.read(size)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.test/vad.onnx",
        "file:///tmp/vad.onnx",
        "ftp://example.test/vad.onnx",
        "https:///vad.onnx",
    ],
)
def test_try_download_rejects_unsupported_url_before_network(monkeypatch, tmp_path, url):
    """非法初始 scheme 或缺少 hostname 时不得触发网络访问。"""
    calls = []
    monkeypatch.setenv("NEMO_VAD_ONNX_URL", url)
    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *args: calls.append(args),
    )

    out_path = tmp_path / "vad_marblenet.onnx"
    with pytest.raises(ValueError):
        downloader._try_download(str(tmp_path), str(out_path))

    assert calls == []
    assert not out_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_redirect_downgrade_is_rejected():
    """HTTPS 下载遇到 HTTP 重定向时必须 fail-closed。"""
    handler = downloader._HttpsOnlyRedirectHandler()
    request = urllib.request.Request("https://example.test/vad.onnx")

    with pytest.raises(ValueError, match="https scheme"):
        handler.redirect_request(request, None, 302, "Found", {}, "http://example.test/vad.onnx")


def test_try_download_https_success_streams_with_timeout_and_atomic_replace(monkeypatch, tmp_path):
    """下载应流式写入同目录唯一临时文件，校验后原子发布。"""
    url = "https://example.test/vad.onnx"
    payload = b"onnx-bytes"
    monkeypatch.setenv("NEMO_VAD_ONNX_URL", url)
    monkeypatch.setenv("NEMO_VAD_ONNX_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(downloader, "_validate_onnx_file", lambda path: True)

    calls = []

    class _FakeOpener:
        def open(self, request, timeout):
            calls.append((request.full_url, timeout))
            return _FakeResponse(payload, len(payload))

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: _FakeOpener())

    out_path = tmp_path / "vad_marblenet.onnx"
    out_path.write_bytes(b"old-model")
    assert downloader._try_download(str(tmp_path), str(out_path)) is True

    assert calls == [(url, downloader.DOWNLOAD_TIMEOUT_SECONDS)]
    assert out_path.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [out_path]


def test_checksum_mismatch_cleans_owned_temp_and_preserves_existing(monkeypatch, tmp_path):
    """checksum 不匹配时不得发布新文件，也不得留下临时文件。"""
    url = "https://example.test/vad.onnx"
    out_path = tmp_path / "vad_marblenet.onnx"
    out_path.write_bytes(b"trusted-existing")
    monkeypatch.setenv("NEMO_VAD_ONNX_URL", url)
    monkeypatch.setenv("NEMO_VAD_ONNX_SHA256", "0" * 64)

    def _write_wrong_payload(url_arg, tmp_path_arg):
        with open(tmp_path_arg, "wb") as file_obj:
            file_obj.write(b"wrong-payload")
        return hashlib.sha256(b"wrong-payload").hexdigest()

    monkeypatch.setattr(downloader, "_download_to_temp", _write_wrong_payload)
    assert downloader._try_download(str(tmp_path), str(out_path)) is False

    assert out_path.read_bytes() == b"trusted-existing"
    assert list(tmp_path.iterdir()) == [out_path]


def test_download_size_limit_cleans_temp(monkeypatch, tmp_path):
    """声明或实际大小超过上限时必须停止下载并清理临时文件。"""
    url = "https://example.test/vad.onnx"
    monkeypatch.setenv("NEMO_VAD_ONNX_URL", url)
    monkeypatch.setenv("NEMO_VAD_ONNX_SHA256", "0" * 64)

    class _FakeOpener:
        def open(self, request, timeout):
            return _FakeResponse(b"small", downloader.MAX_DOWNLOAD_BYTES + 1)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: _FakeOpener())
    out_path = tmp_path / "vad_marblenet.onnx"
    assert downloader._try_download(str(tmp_path), str(out_path)) is False
    assert list(tmp_path.iterdir()) == []


def test_corrupt_existing_file_is_not_accepted(monkeypatch, tmp_path):
    """损坏、空或非普通文件不得被已有文件短路逻辑接受。"""
    path = tmp_path / "vad_marblenet.onnx"
    path.write_bytes(b"corrupt")
    monkeypatch.setattr(
        downloader,
        "_validate_onnx_file",
        lambda candidate: (_ for _ in ()).throw(ValueError("invalid ONNX")),
    )

    assert downloader._existing_onnx_path(str(tmp_path)) is None


def test_export_validates_before_atomic_publish_and_cleans_temp(monkeypatch, tmp_path):
    """NeMo 导出必须先完成临时文件校验，再原子替换目标文件。"""
    models = types.ModuleType("nemo.collections.asr.models")

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

        def eval(self):
            return self

        def export(self, path):
            with open(path, "wb") as file_obj:
                file_obj.write(b"exported-model")

    models.EncDecClassificationModel = _FakeModel
    monkeypatch.setitem(sys.modules, "nemo", types.ModuleType("nemo"))
    monkeypatch.setitem(sys.modules, "nemo.collections", types.ModuleType("nemo.collections"))
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", types.ModuleType("nemo.collections.asr"))
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", models)

    validated = []
    monkeypatch.setattr(downloader, "_validate_onnx_file", lambda path: validated.append(path))
    out_path = tmp_path / "vad_marblenet.onnx"
    out_path.write_bytes(b"old-model")

    assert downloader._try_export_with_nemo(str(tmp_path), str(out_path)) is True
    assert out_path.read_bytes() == b"exported-model"
    assert len(validated) == 2
    assert validated[0] != str(out_path)
    assert validated[0].startswith(str(tmp_path))
    assert validated[1] == str(out_path)
    assert list(tmp_path.iterdir()) == [out_path]


def test_main_explicit_url_checksum_failure_does_not_fallback_to_export(monkeypatch, tmp_path):
    """显式 URL 下载 checksum 失败后必须直接失败，禁止回退 NeMo 导出。"""
    url = "https://example.test/vad.onnx"
    payload = b"wrong-payload"
    monkeypatch.setenv("NEMO_VAD_ONNX_URL", url)
    monkeypatch.setenv("NEMO_VAD_ONNX_SHA256", "0" * 64)
    monkeypatch.setattr(downloader, "_get_third_party_nemo_onnx_dir", lambda: str(tmp_path))
    monkeypatch.setattr(downloader, "_existing_onnx_path", lambda target_dir: None)

    def _write_wrong_payload(url_arg, tmp_path_arg):
        with open(tmp_path_arg, "wb") as file_obj:
            file_obj.write(payload)
        return hashlib.sha256(payload).hexdigest()

    export_calls = []
    monkeypatch.setattr(downloader, "_download_to_temp", _write_wrong_payload)
    monkeypatch.setattr(
        downloader,
        "_try_export_with_nemo",
        lambda *args: export_calls.append(args) or True,
    )

    assert downloader.main() == 1
    assert export_calls == []
    assert list(tmp_path.iterdir()) == []


def test_main_explicit_url_validation_failure_does_not_fallback_to_export(monkeypatch, tmp_path):
    """显式 URL 下载 ONNX 校验失败后必须直接失败，禁止回退 NeMo 导出。"""
    url = "https://example.test/vad.onnx"
    payload = b"invalid-onnx"
    monkeypatch.setenv("NEMO_VAD_ONNX_URL", url)
    monkeypatch.setenv("NEMO_VAD_ONNX_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(downloader, "_get_third_party_nemo_onnx_dir", lambda: str(tmp_path))
    monkeypatch.setattr(downloader, "_existing_onnx_path", lambda target_dir: None)

    def _write_invalid_onnx(url_arg, tmp_path_arg):
        with open(tmp_path_arg, "wb") as file_obj:
            file_obj.write(payload)
        return hashlib.sha256(payload).hexdigest()

    export_calls = []
    monkeypatch.setattr(downloader, "_download_to_temp", _write_invalid_onnx)
    monkeypatch.setattr(
        downloader,
        "_validate_onnx_file",
        lambda path: (_ for _ in ()).throw(ValueError("invalid ONNX")),
    )
    monkeypatch.setattr(
        downloader,
        "_try_export_with_nemo",
        lambda *args: export_calls.append(args) or True,
    )

    assert downloader.main() == 1
    assert export_calls == []
    assert list(tmp_path.iterdir()) == []


def test_main_help_exits_before_filesystem_side_effects(monkeypatch, tmp_path, capsys):
    """--help 必须在创建目标目录或检查模型前退出。"""
    target_dir = tmp_path / "NeMo-VAD-onnx"
    monkeypatch.setattr(downloader, "_get_third_party_nemo_onnx_dir", lambda: str(target_dir))

    with pytest.raises(SystemExit) as exc_info:
        downloader.main(["--help"])

    assert exc_info.value.code == 0
    assert "usage:" in capsys.readouterr().out
    assert not target_dir.exists()
