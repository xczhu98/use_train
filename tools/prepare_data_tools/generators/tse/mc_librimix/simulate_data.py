#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Derived from WeSep (https://github.com/wenet-e2e/wesep), local/simulate_data.sh.
# Modifications Copyright 2026 use_train contributors.
# 修改：shell 编排迁为 Python，完整验证后原子发布，不支持跳过或覆盖。
"""在同文件系统的临时目录生成并发布 MC-LibriMix 资源。"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from . import TOOL_VERSION, UPSTREAM_COMMIT, UPSTREAM_URL
from .simulation.create_scenarios import build_scenarios, load_config, write_scenarios
from .simulation.simulate_librimix import expected_paths, generate_waveforms


def _empty_destination(path):
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise FileExistsError(f"Output must be absent or an empty directory: {path}")


def _validate_assets(config, rows, root):
    try:
        import numpy as np
        import soundfile as sf
    except (ImportError, OSError) as error:
        raise RuntimeError("numpy and soundfile are required to validate generated assets") from error
    expected_wavs, cue_count = set(), 0
    by_split = {}
    for row in rows:
        by_split.setdefault(row["split"], []).append(row)
        paths = expected_paths(root / row["split"], f"{row['mixture_id']}.wav", config["outputs"])
        expected_wavs.update(paths)
        frames = set()
        for path in paths:
            info = sf.info(path)
            if (info.frames <= 0 or info.samplerate != config["audio"]["sample_rate"]
                    or info.channels != len(config["array"]["mic_positions"])
                    or info.subtype != "PCM_16"):
                raise ValueError(f"Invalid generated WAV: {path}")
            frames.add(info.frames)
        if len(frames) != 1:
            raise ValueError(f"Unaligned generated WAVs: {row['mixture_id']}")
    if set(root.glob("*/*/*.wav")) != expected_wavs:
        raise ValueError("Unexpected or missing generated WAV files")
    for split, scenarios in by_split.items():
        index_path = root / split / "cues" / "spatial.json"
        with index_path.open(encoding="utf-8") as stream:
            index = json.load(stream)
        expected_keys = {f"{r['mixture_id']}::{r[f'speaker_{i}']}"
                         for r in scenarios for i in (1, 2)}
        if not isinstance(index, dict) or set(index) != expected_keys:
            raise ValueError(f"Spatial cue keys do not match scenarios: {split}")
        for row in scenarios:
            for i in (1, 2):
                key = f"{row['mixture_id']}::{row[f'speaker_{i}']}"
                item = index[key]
                if not isinstance(item, dict) or set(item) != {"path"} or not isinstance(item["path"], str):
                    raise ValueError(f"Invalid spatial cue index entry: {key}")
                relative = Path(item["path"])
                expected = Path("spatial_npy") / f"{row['mixture_id']}__{row[f'speaker_{i}']}.npy"
                if relative.is_absolute() or relative != expected:
                    raise ValueError(f"Spatial cue must use its canonical relative path: {key}")
                values = {
                    "azimuth": math.radians(float(row[f"source_{i}_azimuth_deg"])),
                    "elevation": math.radians(float(row[f"source_{i}_elevation_deg"])),
                    "distance": float(row[f"source_{i}_distance"]),
                }
                payload = np.load(index_path.parent / relative, allow_pickle=False)
                if payload.dtype.names is None or set(payload.dtype.names) != set(values) or payload.shape != (1,):
                    raise ValueError(f"Invalid spatial cue fields: {key}")
                for field, value in values.items():
                    data = payload[field]
                    if (data.dtype != np.float32 or data.size != 1
                            or not np.isfinite(data).all()
                            or not np.allclose(data[0], value, rtol=1e-6, atol=1e-7)):
                        raise ValueError(f"Invalid spatial cue value: {key}/{field}")
                cue_count += 1
    return {"scenarios": len(rows), "wav_files": len(expected_wavs), "spatial_cues": cue_count}


def run(config_path, librimix_root, output_root, split="all", *, backend=None):
    output = Path(output_root).absolute()
    _empty_destination(output)
    config_path = Path(config_path)
    config_bytes = config_path.read_bytes()
    config = load_config(config_path)
    if config_path.read_bytes() != config_bytes:
        raise RuntimeError("Simulation config changed while being loaded")
    rows = build_scenarios(config, librimix_root, split)
    from .build_spatial_cues import export_spatial_cues

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        manifest = staging / "metadata" / "scenarios_v1.csv.gz"
        write_scenarios(rows, manifest)
        waveform_stats = generate_waveforms(config, librimix_root, manifest, staging,
                                           split, backend=backend)
        cue_stats = export_spatial_cues(manifest, staging, split)
        counts = _validate_assets(config, rows, staging)
        summary = {
            "tool": "mc_librimix",
            "tool_version": TOOL_VERSION,
            "upstream_url": UPSTREAM_URL,
            "upstream_commit": UPSTREAM_COMMIT,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "config": config,
            "seed": config["dataset"]["seed"],
            "split": split,
            "counts": counts,
            "waveforms": waveform_stats,
            "cues": cue_stats,
        }
        with (staging / "metadata" / "generation_summary.json").open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        _empty_destination(output)
        os.rename(staging, output)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate and atomically publish MC-LibriMix audio and spatial cues.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--librimix-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="all")
    args = parser.parse_args(argv)
    summary = run(args.config, args.librimix_root, args.output_root, args.split)
    print(json.dumps(summary["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
