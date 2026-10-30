#!/usr/bin/env python3
# Copyright (c) WeSep contributors.
# SPDX-License-Identifier: Apache-2.0
# 修改声明：迁入 use_train，增加严格音频校验、相对路径和无覆盖原子发布。
"""扫描多通道 Libri2Mix 音频并生成 WeSep 样本索引。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile


def speaker_ids(mixture_id):
    """解析 Libri2Mix ID，并拒绝同一说话人的双源混合。"""
    if not isinstance(mixture_id, str) or not re.fullmatch(
        r"[0-9]+-[0-9]+-[0-9]+_[0-9]+-[0-9]+-[0-9]+", mixture_id
    ):
        raise ValueError(f"Expected a two-speaker Libri2Mix ID, got: {mixture_id}")
    speakers = [part.split("-")[0] for part in mixture_id.split("_")]
    if speakers[0] == speakers[1]:
        raise ValueError(f"Speaker IDs must be distinct: {mixture_id}")
    return speakers


def _audio_inventory(directory):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {directory}")
    paths = {path.stem: path for path in directory.glob("*.wav") if path.is_file()}
    if not paths:
        raise ValueError(f"No WAV files found in {directory}")
    return paths


def scan_records(mix_dir, s1_dir, s2_dir, outfile) -> list[dict]:
    """全量校验三路音频，返回以 outfile 父目录为锚点的样本。"""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Scanning MC-LibriMix audio requires soundfile") from exc

    inventories = [_audio_inventory(path) for path in (mix_dir, s1_dir, s2_dir)]
    keys = set(inventories[0])
    if any(set(inventory) != keys for inventory in inventories[1:]):
        raise ValueError("Mixture, s1 and s2 WAV IDs must match exactly")
    anchor = Path(outfile).expanduser().absolute().parent.resolve()
    records = []
    for key in sorted(keys):
        speakers = speaker_ids(key)
        paths = [inventory[key] for inventory in inventories]
        info = [sf.info(str(path)) for path in paths]
        shapes = [(item.frames, item.samplerate, item.channels) for item in info]
        if any(any(value <= 0 for value in shape) for shape in shapes):
            raise ValueError(f"Audio frames, sample rate and channels must be positive: {key}")
        if len(set(shapes)) != 1:
            raise ValueError(f"Audio frames/sample rate/channels mismatch for {key}: {shapes}")
        relative = [Path(os.path.relpath(path, anchor)).as_posix() for path in paths]
        records.append({
            "key": key,
            "spk": speakers,
            "mix": {"default": [relative[0]]},
            "src": {speakers[0]: [relative[1]], speakers[1]: [relative[2]]},
        })
    return records


def write_samples(mix_dir, s1_dir, s2_dir, outfile) -> int:
    """验证完成后原子发布单个索引，绝不覆盖已有文件。"""
    output = Path(outfile).expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(output)
    records = scan_records(mix_dir, s1_dir, s2_dir, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(records)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate MC-LibriMix audio and write a WeSep samples index.",
        allow_abbrev=False,
    )
    parser.add_argument("--mix-dir", required=True)
    parser.add_argument("--s1-dir", required=True)
    parser.add_argument("--s2-dir", required=True)
    parser.add_argument("--outfile", required=True)
    args = parser.parse_args(argv)
    count = write_samples(args.mix_dir, args.s1_dir, args.s2_dir, args.outfile)
    print(f"Saved {count} samples to {args.outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
