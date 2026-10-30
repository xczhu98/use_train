"""按 TIGER EchoSet 配对语义生成 use_train strict separation JSONL。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import soundfile as sf

from core.utils.file_transaction import write_text_files_transactionally


GENERATOR_VERSION = "1.0"
SPLITS = ("train", "val", "test")


def _canonical_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"EchoSet input root is not a directory: {root}")
    missing = [split for split in SPLITS if not (root / split).is_dir()]
    if missing:
        raise ValueError(f"EchoSet input root is missing splits: {missing}")
    return root


def _audio_header(path: Path, *, location: str) -> tuple[int, int]:
    info = sf.info(path)
    if int(info.channels) != 1:
        raise ValueError(f"{location}: expected mono audio, got {info.channels} channels")
    sample_rate = int(info.samplerate)
    frames = int(info.frames)
    if sample_rate <= 0 or frames <= 0:
        raise ValueError(
            f"{location}: invalid audio header sample_rate={sample_rate}, frames={frames}"
        )
    return sample_rate, frames


def _relative_path(path: Path, manifest_dir: Path) -> str:
    return os.path.relpath(path, start=manifest_dir).replace(os.sep, "/")


def build_split_records(
    input_root: str | Path,
    output_dir: str | Path,
    split: str,
) -> list[dict[str, Any]]:
    """Scan one EchoSet split and return strict mixture/source records."""
    if split not in SPLITS:
        raise ValueError(f"unsupported EchoSet split: {split!r}")
    root = _canonical_root(input_root)
    manifest_dir = Path(output_dir).expanduser().resolve()
    records: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    for mixture in sorted((root / split).rglob("mix.wav")):
        source_1 = mixture.with_name("spk1_reverb.wav")
        source_2 = mixture.with_name("spk2_reverb.wav")
        missing = [str(path) for path in (source_1, source_2) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"EchoSet {split} mixture has missing source files: {missing}"
            )
        relative_parent = mixture.parent.relative_to(root / split).as_posix()
        uid = f"echoset-{split}-{relative_parent.replace('/', '__')}"
        if uid in seen_uids:
            raise ValueError(f"duplicate EchoSet uid: {uid}")
        seen_uids.add(uid)
        mix_sr, mix_frames = _audio_header(mixture, location=f"{uid}.mixture")
        source_headers = [
            _audio_header(source_1, location=f"{uid}.source_0"),
            _audio_header(source_2, location=f"{uid}.source_1"),
        ]
        if any(header != (mix_sr, mix_frames) for header in source_headers):
            raise ValueError(
                f"{uid}: mixture/source header mismatch; "
                f"mixture={(mix_sr, mix_frames)}, sources={source_headers}"
            )
        records.append(
            {
                "uid": uid,
                "fs": mix_sr,
                "length": mix_frames,
                "mixture": _relative_path(mixture, manifest_dir),
                "sources": [
                    {
                        "source_id": "source_0",
                        "path": _relative_path(source_1, manifest_dir),
                        "length": mix_frames,
                    },
                    {
                        "source_id": "source_1",
                        "path": _relative_path(source_2, manifest_dir),
                        "length": mix_frames,
                    },
                ],
            }
        )
    if not records:
        raise ValueError(f"EchoSet split {split!r} contains no mix.wav files")
    return records


def _jsonl_content(records: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    )


def prepare_echoset(input_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Generate train/val/test JSONL plus a versioned publication summary."""
    root = _canonical_root(input_root)
    output = Path(output_dir).expanduser().resolve()
    split_records = {
        split: build_split_records(root, output, split) for split in SPLITS
    }
    contents = {split: _jsonl_content(records) for split, records in split_records.items()}
    summary = {
        "schema": "echoset-separation-publication-v1",
        "generator_version": GENERATOR_VERSION,
        "input_root": str(root),
        "splits": {
            split: {
                "records": len(split_records[split]),
                "sha256": hashlib.sha256(contents[split].encode("utf-8")).hexdigest(),
            }
            for split in SPLITS
        },
    }

    def writer(content: str):
        return lambda stream: stream.write(content)

    outputs = [
        (output / f"{split}.jsonl", writer(contents[split])) for split in SPLITS
    ]
    outputs.append(
        (
            output / "echoset_metadata.json",
            writer(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"),
        )
    )
    write_text_files_transactionally(outputs, overwrite=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare strict SS JSONL manifests from the official EchoSet layout."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_echoset(args.input_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
