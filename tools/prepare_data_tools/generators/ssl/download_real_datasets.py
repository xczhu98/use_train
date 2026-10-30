#!/usr/bin/env python3
"""下载并登记 LOCATA 与 RealMAN 的 SSL 真实数据。

LOCATA 使用官方 Zenodo 归档，下载后可安全解压为 ``data/ssl/LOCATA/{dev,eval}``。
RealMAN 使用官方仓库 README 指向的 Hugging Face ``AISHELL/RealMAN``，默认只取
``val``、``test`` 和 ``dataset_info``，用于真实评估；完整训练集可显式传
``--realman full``，避免把 500 GB 级训练资源误当成一次小下载。

所有 HTTP 资源都必须是 HTTPS。LOCATA 归档同时校验官方 MD5 和本地 SHA256；
RealMAN 由 ``huggingface_hub.snapshot_download`` 负责 HTTPS、revision 和断点
续传，工具会将实际文件大小与 revision 写入本地 manifest。数据本体位于
``data/ssl`` 并由仓库忽略规则隔离，不进入 Git。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Sequence

from .download_data import DownloadError, download_resource, safe_extract_archive


DEFAULT_OUTPUT_DIR = Path("data/ssl")
LOCATA_RECORD_URL = "https://zenodo.org/api/records/3630471"
LOCATA_LICENSE = "Open Data Commons Open Database License (ODbL) 1.0；遵循 LOCATA 官方条款。"
LOCATA_RESOURCES: dict[str, dict[str, object]] = {
    "dev": {
        "filename": "dev.zip",
        "url": f"{LOCATA_RECORD_URL}/files/dev.zip/content",
        "bytes": 6_207_354_195,
        "md5": "d5a5417c3f6b2ed0e43581dd06f504e6",
    },
    "eval": {
        "filename": "eval.zip",
        "url": f"{LOCATA_RECORD_URL}/files/eval.zip/content",
        "bytes": 13_045_105_034,
        "md5": "46709713350bc16c106e788920d30a8b",
    },
}
REALMAN_REPO = "AISHELL/RealMAN"
REALMAN_SOURCE = "https://github.com/Audio-WestlakeU/RealMAN"
REALMAN_EVAL_PATTERNS = ("val/**", "test/**", "dataset_info/**", "transcriptions.trn")


def md5_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    """计算文件 MD5，用于核对 LOCATA 官方记录的归档摘要。"""

    digest = hashlib.md5()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
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


def _merge_extracted_tree(staging: Path, destination: Path, *, overwrite: bool) -> list[str]:
    """把单个归档的顶层目录合并到 LOCATA 根目录，拒绝覆盖已有成员。"""

    destination.mkdir(parents=True, exist_ok=True)
    # 保留归档顶层目录（LOCATA 归档分别包含 ``dev/`` 或 ``eval/``）；
    # 若归档直接放置文件，则这些文件直接进入目标根目录。
    source_root = staging
    moved: list[str] = []
    for child in sorted(source_root.iterdir(), key=lambda item: item.name):
        target = destination / child.name
        if target.exists() or target.is_symlink():
            if not overwrite or target.is_symlink():
                raise DownloadError(f"LOCATA extraction would overwrite existing path: {target}")
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        os.replace(child, target)
        moved.append(child.name)
    return moved


def _extract_locata(archive: Path, destination: Path, *, overwrite: bool) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".locata-extract-"))
    extracted_root = staging_parent / "payload"
    try:
        safe_extract_archive(archive, extracted_root)
        return _merge_extracted_tree(extracted_root, destination, overwrite=overwrite)
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def download_locata(
    selection: str,
    output_dir: str | os.PathLike[str],
    *,
    extract: bool = True,
    overwrite: bool = False,
) -> dict[str, object]:
    """下载 LOCATA dev/eval 官方归档并生成本地 manifest。"""

    if selection not in {"dev", "eval", "all"}:
        raise ValueError("LOCATA selection must be dev, eval, or all")
    names = ["dev", "eval"] if selection == "all" else [selection]
    root = Path(output_dir).expanduser() / "LOCATA"
    archive_dir = root / "archives"
    manifest_path = root / "download_manifest.json"
    existing_payload: dict[str, object] = {}
    if manifest_path.exists():
        candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
        if candidate.get("dataset") == "LOCATA":
            existing_payload = candidate
    existing_by_name = {
        str(item["name"]): item
        for item in existing_payload.get("resources", [])
        if isinstance(item, dict) and "name" in item
    }
    records: list[dict[str, object]] = []
    for name in names:
        spec = LOCATA_RESOURCES[name]
        archive = archive_dir / str(spec["filename"])
        result = download_resource(
            str(spec["url"]),
            archive,
            expected_bytes=int(spec["bytes"]),
            max_bytes=32 * 1024**3,
            overwrite=overwrite,
        )
        actual_md5 = md5_file(archive)
        if actual_md5 != str(spec["md5"]):
            raise DownloadError(
                f"LOCATA MD5 mismatch for {archive}: expected {spec['md5']}, got {actual_md5}"
            )
        record: dict[str, object] = {
            "kind": "locata",
            "name": name,
            "source": LOCATA_RECORD_URL,
            "license": LOCATA_LICENSE,
            "archive": str(archive),
            "md5": actual_md5,
            "sha256": result.sha256,
            "bytes": result.bytes,
            "reused": result.reused,
        }
        if extract:
            destination_member = root / name
            if destination_member.exists() and not overwrite:
                record["extracted_members"] = existing_by_name.get(name, {}).get(
                    "extracted_members", [name]
                )
                record["reused_extraction"] = True
            else:
                record["extracted_members"] = _extract_locata(
                    archive,
                    root,
                    overwrite=overwrite,
                )
        records.append(record)
    existing_records = [
        item
        for item in existing_payload.get("resources", [])
        if isinstance(item, dict) and item.get("name") not in names
    ]
    records = existing_records + records
    records.sort(key=lambda item: (0 if item.get("name") == "dev" else 1, str(item.get("name"))))
    manifest = {
        "schema_version": 1,
        "dataset": "LOCATA",
        "source": LOCATA_RECORD_URL,
        "license": LOCATA_LICENSE,
        "resources": records,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def download_realman(
    selection: str,
    output_dir: str | os.PathLike[str],
    *,
    revision: str = "main",
) -> dict[str, object]:
    """通过 Hugging Face 官方仓库下载 RealMAN 评估或完整资源。"""

    if selection not in {"eval", "full"}:
        raise ValueError("RealMAN selection must be eval or full")
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "RealMAN download requires optional huggingface_hub; use the setrain environment"
        ) from exc
    root = Path(output_dir).expanduser() / "RealMAN"
    allow_patterns = None if selection == "full" else list(REALMAN_EVAL_PATTERNS)
    snapshot_root = snapshot_download(
        repo_id=REALMAN_REPO,
        repo_type="dataset",
        revision=revision,
        local_dir=str(root),
        allow_patterns=allow_patterns,
    )
    info = HfApi().dataset_info(REALMAN_REPO, revision=revision)
    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size})
    manifest = {
        "schema_version": 1,
        "dataset": "RealMAN",
        "source": REALMAN_SOURCE,
        "repository": REALMAN_REPO,
        "revision": getattr(info, "sha", revision),
        "license": "CC BY 4.0（以官方仓库当前声明为准）",
        "selection": selection,
        "allow_patterns": allow_patterns,
        "snapshot_root": str(snapshot_root),
        "files": files,
        "total_bytes": sum(int(item["bytes"]) for item in files),
    }
    _atomic_json(root / "download_manifest.json", manifest)
    return manifest


def extract_realman_archives(
    output_dir: str | os.PathLike[str],
    *,
    selection: str = "all",
) -> dict[str, object]:
    """用系统 ``unrar`` 解压 RealMAN 的 val/test 归档到原目录。

    官方仓库按场景保存 RAR；解压目标固定为每个归档所在目录，
    ``unrar -o-`` 保留已有文件并支持中断后继续。缺少 unrar 时直接报错，
    不自动安装第三方程序。
    """

    if selection not in {"val", "test", "all"}:
        raise ValueError("RealMAN extraction selection must be val, test, or all")
    unrar = shutil.which("unrar")
    if unrar is None:
        raise RuntimeError("RealMAN extraction requires the system unrar command")
    root = Path(output_dir).expanduser() / "RealMAN"
    splits = ("val", "test") if selection == "all" else (selection,)
    archives = [archive for split in splits for archive in sorted((root / split).rglob("*.rar"))]
    if not archives:
        raise FileNotFoundError(f"no RealMAN RAR archives found under {root}")
    extracted = 0
    for archive in archives:
        subprocess.run(
            [unrar, "x", "-idq", "-o-", str(archive), f"{archive.parent}{os.sep}"],
            check=True,
        )
        extracted += 1
    result = {
        "dataset": "RealMAN",
        "selection": selection,
        "archives": extracted,
        "root": str(root),
    }
    manifest_path = root / "download_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["extraction"] = result
        _atomic_json(manifest_path, payload)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载并登记 LOCATA/RealMAN SSL 真实数据", allow_abbrev=False)
    parser.add_argument("--locata", choices=("dev", "eval", "all"), help="下载 LOCATA 官方归档")
    parser.add_argument("--realman", choices=("eval", "full"), help="下载 RealMAN 评估子集或完整仓库")
    parser.add_argument("--extract-realman", choices=("val", "test", "all"), help="解压已下载的 RealMAN RAR")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--revision", default="main", help="RealMAN Hugging Face revision")
    parser.add_argument("--no-extract", action="store_true", help="只下载 LOCATA 归档，不解压")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.locata and not args.realman and not args.extract_realman:
        parser.error("至少选择 --locata、--realman 或 --extract-realman")
    payload: dict[str, object] = {}
    if args.locata:
        payload["locata"] = download_locata(
            args.locata,
            args.output_dir,
            extract=not args.no_extract,
            overwrite=args.overwrite,
        )
    if args.realman:
        payload["realman"] = download_realman(args.realman, args.output_dir, revision=args.revision)
    if args.extract_realman:
        payload["realman_extraction"] = extract_realman_archives(
            args.output_dir,
            selection=args.extract_realman,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
