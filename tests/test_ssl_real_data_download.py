"""LOCATA/RealMAN 真实数据下载器的本地合同测试。"""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

import pytest

from tools.prepare_data_tools.generators.ssl import download_data, download_real_datasets


pytestmark = pytest.mark.unit


def test_md5_file(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"ssl-real-data")
    assert download_real_datasets.md5_file(path) == hashlib.md5(b"ssl-real-data").hexdigest()


def test_locata_all_merges_dev_and_eval_archives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resources = {
        "dev": {"filename": "dev.zip", "url": "https://example.org/dev.zip", "bytes": 1, "md5": ""},
        "eval": {"filename": "eval.zip", "url": "https://example.org/eval.zip", "bytes": 1, "md5": ""},
    }
    monkeypatch.setattr(download_real_datasets, "LOCATA_RESOURCES", resources)

    def fake_download(url: str, output: Path, **kwargs) -> download_data.DownloadResult:
        name = output.stem
        payload = name.encode("ascii")
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr(f"{name}/record.txt", payload)
        return download_data.DownloadResult(
            path=str(output),
            url=url,
            bytes=output.stat().st_size,
            sha256=download_data.sha256_file(output),
            reused=False,
            expected_sha256=None,
            content_length=output.stat().st_size,
        )

    monkeypatch.setattr(download_real_datasets, "download_resource", fake_download)
    for spec in resources.values():
        archive = tmp_path / "LOCATA" / "archives" / str(spec["filename"])
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr(f"{archive.stem}/record.txt", archive.stem.encode("ascii"))
        spec["md5"] = download_real_datasets.md5_file(archive)
        archive.unlink()

    payload = download_real_datasets.download_locata("all", tmp_path)
    assert {record["name"] for record in payload["resources"]} == {"dev", "eval"}
    assert (tmp_path / "LOCATA" / "dev" / "record.txt").read_bytes() == b"dev"
    assert (tmp_path / "LOCATA" / "eval" / "record.txt").read_bytes() == b"eval"


def test_locata_split_download_preserves_previous_manifest_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resources = {
        "dev": {"filename": "dev.zip", "url": "https://example.org/dev.zip", "bytes": 1, "md5": ""},
        "eval": {"filename": "eval.zip", "url": "https://example.org/eval.zip", "bytes": 1, "md5": ""},
    }
    monkeypatch.setattr(download_real_datasets, "LOCATA_RESOURCES", resources)

    def fake_download(url: str, output: Path, **kwargs) -> download_data.DownloadResult:
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr(f"{output.stem}/record.txt", output.stem.encode("ascii"))
        return download_data.DownloadResult(
            path=str(output),
            url=url,
            bytes=output.stat().st_size,
            sha256=download_data.sha256_file(output),
            reused=False,
            expected_sha256=None,
            content_length=output.stat().st_size,
        )

    monkeypatch.setattr(download_real_datasets, "download_resource", fake_download)
    for spec in resources.values():
        archive = tmp_path / "LOCATA" / "archives" / str(spec["filename"])
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr(f"{archive.stem}/record.txt", archive.stem.encode("ascii"))
        spec["md5"] = download_real_datasets.md5_file(archive)
        archive.unlink()

    download_real_datasets.download_locata("dev", tmp_path)
    payload = download_real_datasets.download_locata("eval", tmp_path)
    assert [record["name"] for record in payload["resources"]] == ["dev", "eval"]
    assert (tmp_path / "LOCATA" / "dev" / "record.txt").read_bytes() == b"dev"


def test_realman_eval_patterns_exclude_training_payloads() -> None:
    assert "val/**" in download_real_datasets.REALMAN_EVAL_PATTERNS
    assert "test/**" in download_real_datasets.REALMAN_EVAL_PATTERNS
    assert all(not pattern.startswith("train") for pattern in download_real_datasets.REALMAN_EVAL_PATTERNS)


def test_realman_extraction_uses_existing_rar_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "RealMAN" / "val" / "ma_noisy_speech" / "OfficeRoom1.rar"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"rar-placeholder")
    calls: list[list[str]] = []
    monkeypatch.setattr(download_real_datasets.shutil, "which", lambda name: "/usr/bin/unrar")
    monkeypatch.setattr(
        download_real_datasets.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command),
    )
    result = download_real_datasets.extract_realman_archives(tmp_path, selection="val")
    assert result["archives"] == 1
    assert calls == [[
        "/usr/bin/unrar",
        "x",
        "-idq",
        "-o-",
        str(archive),
        f"{archive.parent}{download_real_datasets.os.sep}",
    ]]
