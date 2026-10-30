#!/usr/bin/env python3
"""从 Libri2Mix 的 ``s1/s2`` 音频目录生成 dynamic_tse source JSONL。

该脚本只负责把 Libri2Mix 的配对源音频转换为
``prepare_dynamic_tse_manifests.py`` 所需的 source contract；混响、噪声和
混合仍由 use_train 的 Dataset/仿真链路负责。每条 source 记录的 enrollment
候选仅来自同一 split、同一 speaker，并按原始录音身份排除当前 source。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Sequence

from tools.prepare_data_tools.generators.tse.mc_librimix.scan_librimix import (
    speaker_ids,
)


_MIXTURE_ID = re.compile(r"[0-9]+-[0-9]+-[0-9]+_[0-9]+-[0-9]+-[0-9]+")
GENERATOR_VERSION = 2


def _inventory(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Libri2Mix audio directory not found: {directory}")
    paths = {}
    for path in sorted(directory.glob("*.wav")):
        if path.is_symlink() or not path.is_file():
            continue
        if not _MIXTURE_ID.fullmatch(path.stem):
            raise ValueError(f"unexpected Libri2Mix filename: {path}")
        paths[path.stem] = path.resolve()
    if not paths:
        raise ValueError(f"no Libri2Mix WAV files found in {directory}")
    return paths


def build_source_records(
    dataset_root: str | os.PathLike[str],
    split: str,
    *,
    dataset_id: str | None = None,
    language: str = "en",
    enrollment_count: int = 2,
) -> list[dict[str, object]]:
    """扫描一个 split，返回 strict producer 的 source records。"""
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string")
    if (
        not isinstance(language, str)
        or not language.strip()
        or language != language.strip()
    ):
        raise ValueError("language must be a non-empty trimmed string")
    if (
        isinstance(enrollment_count, bool)
        or not isinstance(enrollment_count, int)
        or enrollment_count <= 0
    ):
        raise ValueError("enrollment_count must be a positive integer")
    root = Path(dataset_root).expanduser().resolve(strict=True) / split
    s1 = _inventory(root / "s1")
    s2 = _inventory(root / "s2")
    if set(s1) != set(s2):
        raise ValueError("Libri2Mix s1 and s2 filename sets must match exactly")
    resolved_dataset_id = dataset_id or f"librimix_{split}"
    if (
        not isinstance(resolved_dataset_id, str)
        or not resolved_dataset_id.strip()
        or resolved_dataset_id != resolved_dataset_id.strip()
    ):
        raise ValueError("dataset_id must be a non-empty trimmed string")

    pools: dict[str, list[Path]] = {}
    recording_ids: dict[Path, str] = {}
    rows: list[tuple[str, str, Path]] = []
    for mixture_id in sorted(s1):
        first, second = speaker_ids(mixture_id)
        for source_name, speaker, path in (
            ("s1", first, s1[mixture_id]),
            ("s2", second, s2[mixture_id]),
        ):
            recording_ids[path] = mixture_id.split("_")[0 if source_name == "s1" else 1]
            rows.append((source_name, speaker, path))
            bucket = pools.setdefault(speaker, [])
            if path not in bucket:
                bucket.append(path)

    records: list[dict[str, object]] = []
    for source_name, speaker, path in rows:
        candidates = list({
            recording_ids[candidate]: candidate for candidate in reversed(pools[speaker])
            if recording_ids[candidate] != recording_ids[path]
        }.values())
        candidates.sort()
        if not candidates:
            raise ValueError(
                f"speaker {speaker!r} has no enrollment candidate besides {path}"
            )
        records.append(
            {
                "uid": f"{source_name}:{path.stem}",
                "path": str(path),
                "language": language,
                "dataset_id": resolved_dataset_id,
                "speaker_id": speaker,
                "recording_id": recording_ids[path],
                "generator_version": GENERATOR_VERSION,
                "enrollments": [str(item) for item in candidates[:enrollment_count]],
            }
        )
    return records


def write_source_jsonl(records: Sequence[dict[str, object]], output: str | os.PathLike[str], *, overwrite: bool = False) -> int:
    """以临时文件事务发布 source JSONL。"""
    target = Path(output).expanduser().resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    if not records:
        raise ValueError("records must not be empty")
    target.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent,
        prefix=f".{target.name}.", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if target.exists() and not overwrite:
            raise FileExistsError(target)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return len(records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create dynamic_tse source JSONL from Libri2Mix s1/s2 WAVs",
        allow_abbrev=False,
    )
    parser.add_argument("--dataset-root", required=True, help="Libri2Mix wav16k/min root")
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-source-jsonl", required=True)
    parser.add_argument("--dataset-id")
    parser.add_argument("--language", default="en")
    parser.add_argument("--enrollment-count", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    records = build_source_records(
        args.dataset_root,
        args.split,
        dataset_id=args.dataset_id,
        language=args.language,
        enrollment_count=args.enrollment_count,
    )
    count = write_source_jsonl(records, args.output_source_jsonl, overwrite=args.overwrite)
    print(f"Saved {count} Libri2Mix source records to {args.output_source_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
