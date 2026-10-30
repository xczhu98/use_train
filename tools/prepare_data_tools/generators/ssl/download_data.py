#!/usr/bin/env python3
"""安全下载和整理 SSL 外部数据。

该工具只处理外部资源的获取与本地登记，不把数据文件加入 Git。默认数据目录为
``data/ssl``。FN-SSL 官方 README 使用的 Noise92 SPIB 页面只提供 ``.mat``
文件且未提供发布方 SHA256，因此脚本会计算并记录下载后的摘要，但不会伪造
预期摘要。自定义 URL 必须使用 HTTPS；若需要可复现的强校验，请同时传入
``--sha256``。

示例::

    python tools/prepare_data_tools/generators/ssl/download_data.py --list
    python tools/prepare_data_tools/generators/ssl/download_data.py \
        --download-noise92 babble white --convert-wav
    python tools/prepare_data_tools/generators/ssl/download_data.py \
        --download-url https://example.org/archive.zip \
        --output data/ssl/external/archive.zip --sha256 <sha256> \
        --extract-dir data/ssl/external/archive

Noise92 原始文件的许可和再分发条件以 SPIB/TNO/RSRE 的原始页面为准；脚本只
记录来源，不宣称这些文件是 MIT 或 CC 数据。LOCATA 和 RealMAN 体积较大、
有额外许可或访问条件，脚本不提供自动下载入口。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import posixpath
import shlex
import shutil
import ssl
import stat
import tempfile
from typing import Iterator, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener
import zipfile
import tarfile


DEFAULT_OUTPUT_DIR = Path("data/ssl")
SPIB_NOISE_PAGE = "https://spib.linse.ufsc.br/noise.html"
SPIB_NOISE_BASE = "https://spib.linse.ufsc.br/data/noise/"


@dataclass(frozen=True)
class Noise92Resource:
    """SPIB Noise92 资源元数据。

    ``sha256`` 留空是有意的：SPIB 页面没有公布文件摘要，不能从文件名推断或
    伪造预期值。下载后仍会计算内容摘要并写入本地 manifest。
    """

    name: str
    filename: str
    url: str
    content_length: int | None
    sha256: str | None = None
    license_note: str = (
        "SPIB 页面说明文件由 TNO/RSRE 发布为 NOISE-ROM-0；"
        "请遵循原始版权和使用条款。"
    )


def _noise_resource(filename: str, content_length: int | None) -> Noise92Resource:
    name = Path(filename).stem
    return Noise92Resource(
        name=name,
        filename=filename,
        url=SPIB_NOISE_BASE + filename,
        content_length=content_length,
    )


# 页面列出的 15 个文件。Content-Length 来自 2026-10-01 的 HTTPS HEAD；由于
# SPIB 未发布 SHA256，保留 None，运行时会把实际摘要写入 manifest。
NOISE92_RESOURCES: dict[str, Noise92Resource] = {
    resource.name: resource
    for resource in (
        _noise_resource("white.mat", 9_400_000),
        _noise_resource("pink.mat", 9_399_992),
        _noise_resource("babble.mat", 12_541_832),
        _noise_resource("factory1.mat", 9_400_000),
        _noise_resource("factory2.mat", 9_400_000),
        _noise_resource("buccaneer1.mat", 9_400_008),
        _noise_resource("buccaneer2.mat", 9_400_008),
        _noise_resource("f16.mat", 9_399_992),
        _noise_resource("destroyerengine.mat", 9_400_008),
        _noise_resource("destroyerops.mat", 9_400_008),
        _noise_resource("leopard.mat", 9_400_000),
        _noise_resource("m109.mat", 9_399_992),
        _noise_resource("machinegun.mat", 9_400_008),
        _noise_resource("volvo.mat", 9_400_000),
        _noise_resource("hfchannel.mat", 9_400_008),
    )
}


class DownloadError(RuntimeError):
    """下载或完整性校验失败。"""


def _validate_https_url(url: str) -> str:
    """只接受无凭据的 HTTPS URL。"""

    if not isinstance(url, str) or not url.strip():
        raise ValueError("url must be a non-empty string")
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() != "https":
        raise ValueError(f"only HTTPS URLs are allowed: {url!r}")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("HTTPS URL must contain a hostname and no embedded credentials")
    if parsed.fragment:
        raise ValueError("URL fragments are not allowed for downloads")
    return url.strip()


class _HTTPSRedirectHandler(HTTPRedirectHandler):
    """拒绝从 HTTPS 降级到 HTTP 的重定向。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        resolved = urljoin(req.full_url, newurl)
        _validate_https_url(resolved)
        return super().redirect_request(req, fp, code, msg, headers, resolved)


def _open_url(url: str, *, timeout: float) -> object:
    """通过系统 CA 验证 TLS 并打开 HTTPS 资源。"""

    validated = _validate_https_url(url)
    context = ssl.create_default_context()
    opener = build_opener(HTTPSHandler(context=context), _HTTPSRedirectHandler())
    request = Request(validated, headers={"User-Agent": "use-train-ssl-data/1"})
    try:
        # HTTPSHandler 使用系统 CA 与 hostname 校验；没有 CERT_NONE 或降级路径。
        return opener.open(request, timeout=timeout)
    except (HTTPError, URLError) as exc:
        raise DownloadError(f"HTTPS download failed for {validated}: {exc}") from exc


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    """计算文件 SHA256。"""

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("sha256 must be exactly 64 hexadecimal characters")
    return normalized


def _header_int(response: object, name: str) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class DownloadResult:
    path: str
    url: str
    bytes: int
    sha256: str
    reused: bool
    expected_sha256: str | None
    content_length: int | None


def download_resource(
    url: str,
    output: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    max_bytes: int = 2 * 1024 * 1024 * 1024,
    timeout: float = 60.0,
    overwrite: bool = False,
) -> DownloadResult:
    """通过 HTTPS 下载一个文件并原子发布。

    已有文件在提供 ``expected_sha256`` 时必须匹配后才会复用；未提供发布方
    摘要时仍计算并返回本地摘要，调用方应把该摘要写入 provenance manifest。
    """

    normalized_url = _validate_https_url(url)
    expected_sha = _validate_sha256(expected_sha256)
    if isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if expected_bytes is not None and (isinstance(expected_bytes, bool) or expected_bytes < 0):
        raise ValueError("expected_bytes must be a non-negative integer")
    target = Path(output).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        if not target.is_file() or target.is_symlink():
            raise DownloadError(f"output exists but is not a regular file: {target}")
        existing_bytes = target.stat().st_size
        if expected_bytes is not None and existing_bytes != expected_bytes:
            if not overwrite:
                raise DownloadError(
                    f"existing file size mismatch for {target}: "
                    f"expected {expected_bytes}, got {existing_bytes}"
                )
        else:
            actual_sha = sha256_file(target)
            if expected_sha is not None and actual_sha != expected_sha:
                if not overwrite:
                    raise DownloadError(
                        f"existing file SHA256 mismatch for {target}: "
                        f"expected {expected_sha}, got {actual_sha}"
                    )
            else:
                return DownloadResult(
                    path=str(target),
                    url=normalized_url,
                    bytes=existing_bytes,
                    sha256=actual_sha,
                    reused=True,
                    expected_sha256=expected_sha,
                    content_length=expected_bytes,
                )

    response = _open_url(normalized_url, timeout=timeout)
    temporary: Path | None = None
    digest = hashlib.sha256()
    total = 0
    content_length = _header_int(response, "Content-Length")
    if content_length is not None and content_length > max_bytes:
        response.close()  # type: ignore[attr-defined]
        raise DownloadError(
            f"remote resource is larger than max_bytes ({content_length} > {max_bytes}): {normalized_url}"
        )
    if expected_bytes is not None and content_length is not None and content_length != expected_bytes:
        response.close()  # type: ignore[attr-defined]
        raise DownloadError(
            f"Content-Length mismatch for {normalized_url}: expected {expected_bytes}, got {content_length}"
        )
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".download", delete=False
        ) as stream:
            temporary = Path(stream.name)
            while True:
                block = response.read(1024 * 1024)  # type: ignore[attr-defined]
                if not block:
                    break
                total += len(block)
                if total > max_bytes:
                    raise DownloadError(
                        f"remote resource exceeded max_bytes ({max_bytes}): {normalized_url}"
                    )
                digest.update(block)
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        response.close()  # type: ignore[attr-defined]

    if expected_bytes is not None and total != expected_bytes:
        temporary.unlink(missing_ok=True)  # type: ignore[union-attr]
        raise DownloadError(
            f"download size mismatch for {normalized_url}: expected {expected_bytes}, got {total}"
        )
    actual_sha = digest.hexdigest()
    if expected_sha is not None and actual_sha != expected_sha:
        temporary.unlink(missing_ok=True)  # type: ignore[union-attr]
        raise DownloadError(
            f"download SHA256 mismatch for {normalized_url}: expected {expected_sha}, got {actual_sha}"
        )
    if target.exists() and not overwrite:
        temporary.unlink(missing_ok=True)  # type: ignore[union-attr]
        raise DownloadError(f"output appeared during download: {target}")
    os.replace(temporary, target)
    return DownloadResult(
        path=str(target),
        url=normalized_url,
        bytes=total,
        sha256=actual_sha,
        reused=False,
        expected_sha256=expected_sha,
        content_length=content_length,
    )


def _member_target(root: Path, name: str) -> Path:
    """解析 archive 成员并拒绝绝对路径及目录穿越。"""

    if not name or "\0" in name:
        raise DownloadError("archive member has an empty/NUL name")
    normalized = posixpath.normpath(name.replace("\\", "/"))
    if normalized in {"", "."}:
        return root
    if normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
        raise DownloadError(f"archive member escapes destination: {name!r}")
    target = (root / Path(*normalized.split("/"))).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise DownloadError(f"archive member escapes destination: {name!r}") from exc
    return target


def _zip_member_is_link(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o170000
    return stat.S_ISLNK(mode)


def _extract_zip(archive: Path, staging: Path) -> Iterator[str]:
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            _member_target(staging, info.filename)
            if _zip_member_is_link(info):
                raise DownloadError(f"archive symlink is not allowed: {info.filename!r}")
            target = _member_target(staging, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info, "r") as input_stream, open(target, "xb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream)
            yield info.filename


def _extract_tar(archive: Path, staging: Path) -> Iterator[str]:
    with tarfile.open(archive, mode="r:*") as source:
        for info in source.getmembers():
            _member_target(staging, info.name)
            if info.issym() or info.islnk():
                raise DownloadError(f"archive link is not allowed: {info.name!r}")
            if not (info.isdir() or info.isfile()):
                raise DownloadError(f"unsupported tar member type: {info.name!r}")
            target = _member_target(staging, info.name)
            if info.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = source.extractfile(info)
            if extracted is None:
                raise DownloadError(f"cannot read tar member: {info.name!r}")
            with extracted, open(target, "xb") as output_stream:
                shutil.copyfileobj(extracted, output_stream)
            yield info.name


def safe_extract_archive(
    archive: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> list[str]:
    """安全解压 zip/tar archive，并以目录事务发布。

    绝对路径、``..`` 穿越、symlink/hardlink 和特殊 tar 文件类型均拒绝。默认
    不覆盖已有目录，避免误删用户数据。
    """

    source_input = Path(archive).expanduser()
    if source_input.is_symlink():
        raise DownloadError(f"archive symlink is not allowed: {source_input}")
    source = source_input.resolve(strict=True)
    if not source.is_file():
        raise DownloadError(f"archive is not a regular file: {source}")
    target = Path(destination).expanduser()
    if target.exists() and not overwrite:
        raise DownloadError(f"extraction destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.extract-"))
    try:
        if zipfile.is_zipfile(source):
            members = list(_extract_zip(source, staging))
        elif tarfile.is_tarfile(source):
            members = list(_extract_tar(source, staging))
        else:
            raise DownloadError(f"unsupported archive format: {source}")
        if target.exists():
            if not overwrite:
                raise DownloadError(f"extraction destination appeared during extraction: {target}")
            if target.is_symlink() or not target.is_dir():
                raise DownloadError(f"cannot replace non-directory destination: {target}")
            # overwrite is explicit; remove only the destination selected by caller.
            shutil.rmtree(target)
        os.replace(staging, target)
        staging = None  # type: ignore[assignment]
        return members
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def convert_noise92_mat_to_wav(
    source: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    sample_rate: int = 16_000,
    source_sample_rate: int = 19_980,
    key: str | None = None,
) -> dict[str, object]:
    """把 SPIB 的 Noise92 ``.mat`` 单通道数组转换为 WAV。

    ``scipy`` 和 ``soundfile`` 仅在调用该函数时导入，缺失时给出清晰错误。
    原始 Noise92 采样率是 19.98 kHz，默认使用 polyphase 明确重采样到
    16 kHz，输出为单声道 WAV。设置 ``sample_rate=19_980`` 可保留原采样率。
    """

    try:
        import numpy as np
        import scipy.io
        import scipy.signal
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Noise92 .mat conversion requires optional scipy, numpy, and soundfile"
        ) from exc
    source_path = Path(source).expanduser().resolve(strict=True)
    if source_path.suffix.lower() != ".mat":
        raise ValueError(f"expected .mat source: {source_path}")
    payload = scipy.io.loadmat(source_path)
    candidates = [
        value
        for name, value in payload.items()
        if not name.startswith("__") and isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number)
    ]
    if key is not None:
        if key not in payload:
            raise ValueError(f"MAT variable not found: {key}")
        candidates = [payload[key]]
    if not candidates:
        raise ValueError(f"no numeric signal array found in {source_path}")
    signal = max(candidates, key=lambda value: int(value.size))
    signal = np.asarray(signal).squeeze()
    if signal.ndim != 1:
        raise ValueError(f"Noise92 MAT signal must be one-dimensional, got {signal.shape}")
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("Noise92 MAT signal is empty or contains non-finite values")
    signal = signal.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(signal)))
    if peak > 1.0:
        signal = signal / max(peak, 1.0)
    if source_sample_rate <= 0 or sample_rate <= 0:
        raise ValueError("sample rates must be positive")
    if sample_rate != source_sample_rate:
        signal = scipy.signal.resample_poly(signal, sample_rate, source_sample_rate).astype(
            np.float32, copy=False
        )
    target = Path(output).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
        sf.write(str(temporary), signal, sample_rate, subtype="PCM_16", format="WAV")
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "path": str(target),
        "sample_rate": sample_rate,
        "frames": int(signal.size),
        "sha256": sha256_file(target),
    }


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, content: str, *, overwrite: bool = False) -> bool:
    """原子写文本；已有同内容文件直接复用并返回 True。"""

    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise DownloadError(f"output exists but is not a regular file: {path}")
        if path.read_text(encoding="utf-8") == content:
            return True
        if not overwrite:
            raise DownloadError(f"output exists and differs; pass --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        return False
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_manifest_hashes(path: Path) -> dict[tuple[str, str], str]:
    """读取上次 manifest 的本地摘要，供无发布方 hash 的资源复用校验。"""

    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DownloadError(f"cannot read existing download manifest: {path}") from exc
    records = payload.get("resources") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise DownloadError(f"invalid download manifest resources: {path}")
    hashes: dict[tuple[str, str], str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        url = record.get("url")
        record_path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(url, str) or not isinstance(record_path, str):
            continue
        try:
            normalized = _validate_sha256(digest if isinstance(digest, str) else None)
        except ValueError as exc:
            raise DownloadError(f"invalid SHA256 in download manifest: {path}") from exc
        if normalized:
            hashes[(url, str(Path(record_path).expanduser().resolve()))] = normalized
    return hashes


def _parse_resource_names(values: Sequence[str]) -> list[Noise92Resource]:
    resources: list[Noise92Resource] = []
    for value in values:
        name = value.strip().lower()
        if name == "all":
            resources.extend(NOISE92_RESOURCES.values())
            continue
        if name not in NOISE92_RESOURCES:
            raise ValueError(
                f"unknown Noise92 resource {value!r}; choose from {', '.join(sorted(NOISE92_RESOURCES))} or all"
            )
        resources.append(NOISE92_RESOURCES[name])
    unique: dict[str, Noise92Resource] = {item.name: item for item in resources}
    return list(unique.values())


def _print_resources() -> None:
    for resource in NOISE92_RESOURCES.values():
        print(
            f"noise92:{resource.name}\t{resource.url}\t"
            f"{resource.content_length or 'unknown'} bytes\tsha256=publisher-not-provided"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="安全下载/转换 SSL 外部数据到 data/ssl",
        allow_abbrev=False,
    )
    parser.add_argument("--list", action="store_true", help="列出官方 Noise92 HTTPS 资源")
    parser.add_argument(
        "--download-noise92",
        nargs="+",
        metavar="NAME",
        help="下载 Noise92 资源名（或 all）；默认不自动下载",
    )
    parser.add_argument("--download-url", help="下载自定义 HTTPS URL（必须配 --output）")
    parser.add_argument("--output", help="自定义 URL 的输出路径")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Noise92 输出目录")
    parser.add_argument("--sha256", help="自定义 URL 的预期 SHA256（必填）；官方 SPIB 未提供摘要")
    parser.add_argument("--max-bytes", type=int, default=2 * 1024 * 1024 * 1024)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--convert-wav", action="store_true", help="将下载的 .mat 转换成同名 WAV")
    parser.add_argument("--wav-sample-rate", type=int, default=16_000)
    parser.add_argument("--extract-dir", help="将下载的 zip/tar 安全解压到指定目录")
    parser.add_argument("--manifest", help="下载后写出的 JSON provenance manifest")
    args = parser.parse_args(argv)

    if args.list:
        _print_resources()
        if not (args.download_noise92 or args.download_url):
            return 0
    if args.download_url and not args.output:
        parser.error("--download-url requires --output")
    if args.download_url and not args.sha256:
        parser.error("--download-url requires --sha256; 不接受未校验的自定义资源")
    if args.download_noise92 and args.download_url:
        parser.error("--download-noise92 and --download-url are mutually exclusive")
    if args.extract_dir and not (args.download_url or args.download_noise92):
        parser.error("--extract-dir requires a downloaded archive")
    if args.convert_wav and not args.download_noise92:
        parser.error("--convert-wav currently applies to --download-noise92")

    manifest = (
        Path(args.manifest).expanduser()
        if args.manifest
        else Path(args.output_dir).expanduser() / "download_manifest.json"
    )
    previous_hashes = _load_manifest_hashes(manifest)
    records: list[dict[str, object]] = []
    wav_entries: list[tuple[str, int, int, str]] = []
    if args.download_noise92:
        resources = _parse_resource_names(args.download_noise92)
        output_dir = Path(args.output_dir).expanduser()
        for resource in resources:
            target = output_dir / resource.filename
            manifest_key = (resource.url, str(target.resolve()))
            local_expected_sha = previous_hashes.get(manifest_key)
            expected_sha = resource.sha256 or local_expected_sha
            result = download_resource(
                resource.url,
                target,
                expected_sha256=expected_sha,
                expected_bytes=resource.content_length,
                max_bytes=args.max_bytes,
                timeout=args.timeout,
                overwrite=args.overwrite,
            )
            record = {
                "kind": "noise92",
                "name": resource.name,
                "source_page": SPIB_NOISE_PAGE,
                "license": resource.license_note,
                "hash_source": (
                    "publisher"
                    if resource.sha256
                    else ("previous_manifest" if local_expected_sha else "computed_only")
                ),
                **asdict(result),
            }
            if args.convert_wav:
                wav_result = convert_noise92_mat_to_wav(
                    result.path,
                    Path(result.path).with_suffix(".wav"),
                    sample_rate=args.wav_sample_rate,
                )
                record["wav"] = wav_result
                wav_path = Path(str(wav_result["path"])).expanduser().resolve(strict=True)
                wav_entries.append(
                    (
                        f"noise92:{resource.name}",
                        int(wav_result["sample_rate"]),
                        int(wav_result["frames"]),
                        shlex.quote(os.path.relpath(wav_path, output_dir.resolve())),
                    )
                )
            if args.extract_dir:
                if len(resources) != 1:
                    parser.error("--extract-dir with Noise92 requires exactly one resource")
                record["extracted_members"] = safe_extract_archive(
                    result.path,
                    args.extract_dir,
                    overwrite=args.overwrite,
                )
            records.append(record)
    elif args.download_url:
        target = Path(args.output).expanduser()
        manifest_key = (args.download_url, str(target.resolve()))
        local_expected_sha = previous_hashes.get(manifest_key)
        expected_sha = args.sha256 or local_expected_sha
        result = download_resource(
            args.download_url,
            target,
            expected_sha256=expected_sha,
            max_bytes=args.max_bytes,
            timeout=args.timeout,
            overwrite=args.overwrite,
        )
        record = {
            "kind": "custom",
            "hash_source": "provided" if args.sha256 else ("previous_manifest" if local_expected_sha else "computed_only"),
            **asdict(result),
        }
        if args.extract_dir:
            record["extracted_members"] = safe_extract_archive(
                result.path,
                args.extract_dir,
                overwrite=args.overwrite,
            )
        records.append(record)

    payload: dict[str, object] = {"schema_version": 1, "resources": records}
    if wav_entries:
        output_dir = Path(args.output_dir).expanduser()
        wav_manifest_path = output_dir / "noise92.scp"
        wav_manifest_text = "".join(
            f"{uid} {sample_rate} {frames} {path}\n"
            for uid, sample_rate, frames, path in sorted(wav_entries)
        )
        _atomic_write_text(wav_manifest_path, wav_manifest_text, overwrite=args.overwrite)
        payload["wav_manifest"] = {
            "path": str(wav_manifest_path),
            "records": len(wav_entries),
            "sha256": hashlib.sha256(wav_manifest_text.encode("utf-8")).hexdigest(),
            "sample_rate": args.wav_sample_rate,
            "format": "uid sample_rate frames relative_path",
        }
    if records or wav_entries:
        _atomic_write_json(manifest, payload)
        print(json.dumps({"manifest": str(manifest), **payload}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
