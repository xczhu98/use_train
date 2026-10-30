#!/usr/bin/env python3
# Copyright (c) WeSep contributors.
# SPDX-License-Identifier: Apache-2.0
# 修改声明：将上游 shell 编排迁为 Python，增加完整校验和 staging 发布。
"""为已生成的 MC-LibriMix 音频和空间 cue 准备 WeSep 资源索引。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from .build_spatial_cues import SPATIAL_FIELDS, SPLITS
from .scan_librimix import scan_records


def _unique_mapping(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_cues(index_path, records):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Validating MC-LibriMix spatial cues requires numpy") from exc
    with index_path.open(encoding="utf-8") as stream:
        index = json.load(stream, object_pairs_hook=_unique_mapping)
    expected = {f"{record['key']}::{speaker}" for record in records for speaker in record["spk"]}
    if not isinstance(index, dict) or set(index) != expected:
        raise ValueError(f"Spatial cue keys must exactly match all sample speakers: {index_path}")
    paths = set()
    for key, item in index.items():
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"Spatial cue must contain a payload path: {key}")
        raw = item["path"]
        if not raw or raw != raw.strip() or Path(raw).is_absolute():
            raise ValueError(f"Spatial cue path must be non-empty and relative to its index: {key}")
        path = (index_path.parent / raw).resolve()
        if path.suffix != ".npy" or not path.is_file():
            raise ValueError(f"Spatial cue must point to an existing numeric NPY: {path}")
        if path in paths:
            raise ValueError(f"Spatial cue payload must be unique per mixture speaker: {path}")
        paths.add(path)
        payload = np.load(path, allow_pickle=False)
        if payload.dtype.names is None or set(payload.dtype.names) != set(SPATIAL_FIELDS) or payload.shape != (1,):
            raise ValueError(f"Spatial cue must be a single-point structured array of {SPATIAL_FIELDS}: {path}")
        for field in SPATIAL_FIELDS:
            value = payload[field]
            if value.dtype != np.dtype("float32"):
                raise ValueError(f"Spatial cue field must be float32: {path}:{field}")
            number = float(value[0])
            if not math.isfinite(number):
                raise ValueError(f"Spatial cue field must be finite: {path}:{field}")
            if field == "distance" and number <= 0:
                raise ValueError(f"Spatial cue distance must be positive: {path}")
            if field == "azimuth" and abs(number) > float(np.float32(2 * math.pi)):
                raise ValueError(f"Spatial cue azimuth must use radians in [-2*pi, 2*pi]: {path}")
            if field == "elevation" and abs(number) > float(np.float32(math.pi / 2)):
                raise ValueError(f"Spatial cue elevation must use radians in [-pi/2, pi/2]: {path}")
    return len(index)


def _empty_destination(destination):
    if destination.is_symlink():
        raise ValueError(f"Data output cannot be a symlink: {destination}")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Data output must not exist or must be empty: {destination}")


def prepare_data(
    dataset_root, data, mix_condition="reverb", target_condition="early",
    splits=("train-100", "dev", "test"),
) -> dict:
    """完整校验所选 split，在相邻 staging 内写索引并一次发布。"""
    if mix_condition not in ("early", "reverb") or target_condition not in ("early", "reverb"):
        raise ValueError("Mixture and target conditions must be 'early' or 'reverb'")
    if isinstance(splits, str):
        raise ValueError("splits must be a sequence of split names, not a string")
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or any(split not in SPLITS for split in splits):
        raise ValueError(f"splits must contain distinct names from {SPLITS}")
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    destination = Path(data).expanduser().absolute()
    _empty_destination(destination)
    destination = destination.resolve()
    if destination == root or root in destination.parents:
        raise ValueError("Data output must be outside the original dataset root")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Writing MC-LibriMix cue configuration requires PyYAML") from exc

    prepared = {}
    counts = {}
    for split in splits:
        split_root = root / split
        records = scan_records(
            split_root / f"mix_{mix_condition}",
            split_root / f"s1_{target_condition}",
            split_root / f"s2_{target_condition}",
            destination / split / "samples.jsonl",
        )
        cue_index = split_root / "cues" / "spatial.json"
        cue_count = _validate_cues(cue_index, records)
        resource = Path(os.path.relpath(cue_index, destination / split)).as_posix()
        cue_config = {
            "cues": {
                "spatial": {
                    "type": "npy",
                    "format": "fields",
                    "guaranteed": True,
                    "fields": ["azimuth", "elevation"],
                    "policy": {"type": "fixed", "key": "mix_spk_id", "resource": resource},
                },
            },
        }
        prepared[split] = (records, cue_config)
        counts[split] = {"samples": len(records), "cues": cue_count}

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        for split, (records, cue_config) in prepared.items():
            output = staging / split
            output.mkdir()
            with (output / "samples.jsonl").open("x", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            (output / "raw.list").write_text("samples.jsonl\n", encoding="utf-8")
            with (output / "cues.yaml").open("x", encoding="utf-8") as stream:
                yaml.safe_dump(cue_config, stream, sort_keys=False)

        for split, (records, cue_config) in prepared.items():
            output = staging / split
            with (output / "samples.jsonl").open(encoding="utf-8") as stream:
                staged_records = [json.loads(line) for line in stream]
            if staged_records != records:
                raise ValueError(f"Staged sample index verification failed: {split}")
            for record in staged_records:
                for group in (record["mix"], record["src"]):
                    for paths in group.values():
                        for path in paths:
                            if not (output / path).is_file():
                                raise FileNotFoundError(f"Staged sample audio path cannot resolve: {path}")
            if (output / "raw.list").read_text(encoding="utf-8") != "samples.jsonl\n":
                raise ValueError(f"Staged raw.list verification failed: {split}")
            with (output / "cues.yaml").open(encoding="utf-8") as stream:
                staged_config = yaml.safe_load(stream)
            if staged_config != cue_config:
                raise ValueError(f"Staged cue configuration verification failed: {split}")
            resource = staged_config["cues"]["spatial"]["policy"]["resource"]
            _validate_cues((output / resource).resolve(), staged_records)
        _empty_destination(destination)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "samples": sum(count["samples"] for count in counts.values()),
        "cues": sum(count["cues"] for count in counts.values()),
        "splits": counts,
        "data": str(destination),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare WeSep indexes for validated MC-LibriMix audio and cues.",
        allow_abbrev=False,
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--mix-condition", choices=("early", "reverb"), default="reverb")
    parser.add_argument("--target-condition", choices=("early", "reverb"), default="early")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    args = parser.parse_args(argv)
    stats = prepare_data(
        args.dataset_root, args.data, args.mix_condition, args.target_condition, args.splits,
    )
    print(json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
