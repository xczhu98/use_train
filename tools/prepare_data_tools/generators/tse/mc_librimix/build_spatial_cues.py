#!/usr/bin/env python3
# Copyright (c) WeSep contributors.
# SPDX-License-Identifier: Apache-2.0
# 修改声明：迁入 use_train，改用无 pickle 的结构化 NPY（azimuth/elevation/distance
# 三个 float32 命名字段、单点静态 cue，与 tasks/tse/data/offline_dataset.py 的
# spatial 读取合同一致）、严格校验及相对路径索引。
"""从仿真场景导出按 mixture 和说话人区分的空间 cue。"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import tempfile

from .scan_librimix import speaker_ids


SPLITS = ("train-100", "dev", "test")
SPATIAL_FIELDS = ("azimuth", "elevation", "distance")


def read_scenarios(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"split", "mixture_id", "speaker_1", "speaker_2"}
        required.update(
            f"source_{index}_{field}"
            for index in (1, 2)
            for field in ("azimuth_deg", "elevation_deg", "distance")
        )
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Scenario manifest is missing required cue columns: {path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Duplicate scenario manifest columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Scenario manifest is empty: {path}")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed scenario manifest rows: {path}")
    return rows


def export_spatial_cues(scenario_manifest, output_root, split="all") -> dict:
    """导出弧度角和米制距离，允许目标根目录已有音频及 metadata。"""
    if split not in (*SPLITS, "all"):
        raise ValueError(f"Invalid split: {split}")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Exporting MC-LibriMix spatial cues requires numpy") from exc

    rows = read_scenarios(scenario_manifest)
    root = Path(output_root).expanduser().absolute()
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError(f"Output root must be a directory, not a symlink: {root}")
    indexes = {}
    payloads = {}
    seen = set()
    for row in rows:
        row_split = row["split"]
        if row_split not in SPLITS:
            raise ValueError(f"Invalid scenario split: {row_split}")
        key = row["mixture_id"]
        speakers = speaker_ids(key)
        if [row["speaker_1"], row["speaker_2"]] != speakers:
            raise ValueError(f"Scenario speaker IDs disagree with mixture ID: {key}")
        identity = (row_split, key)
        if identity in seen:
            raise ValueError(f"Duplicate scenario: {identity}")
        seen.add(identity)
        for index, speaker in enumerate(speakers, 1):
            azimuth = float(row[f"source_{index}_azimuth_deg"])
            elevation = float(row[f"source_{index}_elevation_deg"])
            distance = float(row[f"source_{index}_distance"])
            if not all(math.isfinite(value) for value in (azimuth, elevation, distance)):
                raise ValueError(f"Spatial cue values must be finite: {key}::{speaker}")
            if not abs(azimuth) <= 360 or not -90 <= elevation <= 90 or distance <= 0:
                raise ValueError(f"Invalid azimuth, elevation or distance: {key}::{speaker}")
            values = (math.radians(azimuth), math.radians(elevation), distance)
            if any(abs(value) > np.finfo(np.float32).max for value in values):
                raise ValueError(f"Spatial cue exceeds float32 range: {key}::{speaker}")
            if split != "all" and row_split != split:
                continue
            name = f"{key}__{speaker}.npy"
            relative = Path(row_split) / "cues" / "spatial_npy" / name
            payload = np.empty(1, dtype=[(field, "<f4") for field in SPATIAL_FIELDS])
            payload[0] = values
            payloads[relative] = payload
            spatial_index = indexes.setdefault(row_split, {})
            spatial_index[f"{key}::{speaker}"] = {"path": f"spatial_npy/{name}"}
    if not indexes:
        raise ValueError(f"No scenarios found for split: {split}")

    outputs = list(payloads) + [Path(name) / "cues" / "spatial.json" for name in indexes]
    for relative in outputs:
        target = root / relative
        if os.path.lexists(target):
            raise FileExistsError(f"Spatial cue output already exists: {target}")
        for parent in target.parents:
            if parent == root.parent:
                break
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise ValueError(f"Cue output parent must be a directory: {parent}")

    root.parent.mkdir(parents=True, exist_ok=True)
    published = []
    created_dirs = []
    try:
        with tempfile.TemporaryDirectory(prefix=f".{root.name}.cues-", dir=root.parent) as temporary:
            staging = Path(temporary)
            for relative, payload in sorted(payloads.items()):
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    np.save(stream, payload)
            for name, index in sorted(indexes.items()):
                target = staging / name / "cues" / "spatial.json"
                with target.open("x", encoding="utf-8") as stream:
                    json.dump(index, stream, indent=2, sort_keys=True, allow_nan=False)
                    stream.write("\n")
            for relative in outputs:
                target = root / relative
                missing = []
                parent = target.parent
                while not parent.exists():
                    missing.append(parent)
                    parent = parent.parent
                for directory in reversed(missing):
                    directory.mkdir()
                    created_dirs.append(directory)
                os.link(staging / relative, target)
                published.append(target)
    except BaseException:
        for target in reversed(published):
            target.unlink()
        for directory in reversed(created_dirs):
            directory.rmdir()
        raise
    return {
        "mixtures": len(payloads) // 2,
        "cues": len(payloads),
        "splits": {
            name: {"mixtures": len(index) // 2, "cues": len(index)}
            for name, index in sorted(indexes.items())
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export numeric NPY spatial cues from MC-LibriMix scenarios.",
        allow_abbrev=False,
    )
    parser.add_argument("--scenario-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    args = parser.parse_args(argv)
    stats = export_spatial_cues(args.scenario_manifest, args.output_root, args.split)
    print(json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
