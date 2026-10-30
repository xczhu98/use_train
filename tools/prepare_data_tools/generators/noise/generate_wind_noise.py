#!/usr/bin/env python3
"""风噪（wind noise）仿真数据生成工具。

基于 Mirabilii & Habets (IWAENC 2022) 的单通道风噪生成器
（vendored 于 ``_wind_noise_backend.py``，MIT License），按
``se.generate_simulated_rirs`` 的范式批量生成风噪 wav，并输出训练框架
可直接加载的 4 列 ``wind_noise_sources.scp``（``uid fs num_samples path``，路径相对于
manifest 文件目录），对应训练配置中的 ``data.train.windnoise_scp``；历史绝对路径仍兼容。

Usage:
    python -m tools.prepare_data_tools.generators.noise.generate_wind_noise \
        --output-dir /data/noise/wind_noise_16k \
        --fs 16000 --num_data 200 --duration 15 --workers 16

    # 多采样率（对齐 URGENT2026 口径）
    python -m tools.prepare_data_tools.generators.noise.generate_wind_noise \
        --output-dir /data/noise/wind_noise_multi \
        --fs 8000 16000 24000 48000 --num_data 200 --duration 15
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import shlex
import stat
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from core.utils.file_transaction import write_text_files_transactionally  # noqa: E402
from tools.prepare_data_tools._common import parse_float_pair, setup_logging  # noqa: E402
from tools.shared.scp_io import serialize_manifest_relative_path  # noqa: E402
from tools.prepare_data_tools.generators.noise._wind_noise_backend import (  # noqa: E402
    WindNoiseGenerator,
)

logger = setup_logging("generate_wind_noise")


def _make_uid(fs: int, gustiness: float, idx: int) -> str:
    """构造包含关键元信息的 uid / 文件名。"""
    return f"wind_fs{fs}_gust{gustiness:.1f}_{idx:08d}"


def _worker_generate(
    task: dict[str, Any],
) -> tuple[str, int, int, str, float] | None:
    """单任务：采样 gustiness、生成风噪、保存 wav。

    返回 ``(uid, fs, num_samples, abs_path, gustiness)``；前 4 列即 SCP 条目。
    """
    try:
        fs = int(task["fs"])
        idx = int(task["idx"])
        duration = float(task["duration"])
        gust_lo, gust_hi = task["gustiness_range"]
        seed = int(task["seed"])

        # gustiness 与生成器本体使用同一种子来源，保证相同 --seed 完全可复现
        rng = np.random.default_rng(seed)
        gustiness = float(rng.uniform(gust_lo, gust_hi))

        wn = WindNoiseGenerator(
            fs=fs,
            duration=duration,
            generate=True,
            gustiness=gustiness,
            start_seed=seed,
        )
        signal, _wind_profile = wn.generate_wind_noise()
        signal = np.asarray(signal, dtype=np.float32)

        uid = _make_uid(fs, gustiness, idx)
        shards = max(1, int(task["shards"]))
        out_path = (
            Path(task["out_dir"])
            / "wind_noise"
            / f"fs{fs}"
            / f"shard_{idx % shards:04d}"
            / f"{uid}.wav"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), signal, samplerate=fs, subtype="FLOAT")

        return (uid, fs, int(signal.shape[0]), str(out_path.resolve()), gustiness)
    except Exception as e:
        logger.error("Failed to generate wind noise fs=%s #%s: %s", task["fs"], task["idx"], e)
        return None


def _make_tasks(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    """惰性构造任务；idx 跨采样率全局递增，保证每条样本种子唯一。"""
    idx = 0
    for fs in args.fs:
        for _ in range(args.num_data):
            yield {
                "fs": fs,
                "idx": idx,
                "duration": args.duration,
                "gustiness_range": args.gustiness_range,
                "seed": args.seed + idx,
                "out_dir": str(args.out_dir.resolve()),
                "shards": args.shards,
            }
            idx += 1


def _scp_payload(
    entries: list[tuple[str, int, int, str, float]], scp_path: Path
) -> str:
    """序列化 4 列 SCP（gustiness 仅用于元信息统计）。"""
    content = "\n".join(
        f"{shlex.quote(uid)} {fs} {frames} {shlex.quote(relative_path)}"
        for uid, fs, frames, path, _gustiness in entries
        for relative_path in [serialize_manifest_relative_path(path, scp_path)]
    )
    return content + ("\n" if content else "")


def _validate_output_root(path: Path) -> Path:
    """词法绝对化输出路径，并拒绝叶节点或已有 ancestor symlink。"""
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"output path must not contain symbolic links: {current}")
        if current != absolute and not stat.S_ISDIR(mode):
            raise ValueError(f"output path ancestor is not a directory: {current}")
    if absolute.exists() and not absolute.is_dir():
        raise ValueError(f"output path exists and is not a directory: {absolute}")
    return absolute


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate simulated wind noise datasets (Mirabilii & Habets, IWAENC 2022).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 16kHz、200 条、每条 15 秒（URGENT2026 train 口径）
  python -m tools.prepare_data_tools.generators.noise.generate_wind_noise \
      --output-dir /data/noise/wind_noise_16k \
      --fs 16000 --num_data 200 --duration 15

  # 多采样率
  python -m tools.prepare_data_tools.generators.noise.generate_wind_noise \
      --output-dir /data/noise/wind_noise_multi \
      --fs 8000 16000 24000 48000 --num_data 200
        """,
    )
    parser.add_argument("--output-dir", "--out_dir", dest="out_dir", required=True, type=Path, help="输出目录（默认不允许已存在）")
    parser.add_argument("--fs", type=int, nargs="+", default=[16000], help="目标采样率列表（默认 16000）")
    parser.add_argument("--num_data", type=int, default=200, help="每个采样率生成的条数（默认 200）")
    parser.add_argument("--duration", type=float, default=15.0, help="每条时长（秒，默认 15）")
    parser.add_argument(
        "--gustiness_range",
        type=parse_float_pair,
        default=(3.0, 10.0),
        help="阵风强度范围，如 '3,10'（1=恒定风速，值越大阵风越强）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（每条样本 seed = seed + idx）")
    parser.add_argument("--workers", type=int, default=16, help="并行进程数（默认 16）")
    parser.add_argument("--shards", type=int, default=16, help="每个采样率下的分片数（默认 16）")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许写入已存在的非空输出目录",
    )
    args = parser.parse_args(argv)

    if any(fs <= 0 for fs in args.fs):
        parser.error("--fs values must all be > 0")
    if len(set(args.fs)) != len(args.fs):
        parser.error("--fs values must be unique")
    if args.num_data <= 0:
        parser.error("--num_data must be > 0")
    if args.workers <= 0:
        parser.error("--workers must be > 0")
    if args.shards <= 0:
        parser.error("--shards must be > 0")
    if not math.isfinite(args.duration) or args.duration <= 0.0:
        parser.error("--duration must be finite and > 0")
    gust_lo, gust_hi = args.gustiness_range
    if not all(math.isfinite(value) for value in (gust_lo, gust_hi)) or not (
        1.0 <= gust_lo <= gust_hi
    ):
        parser.error("--gustiness_range must be finite and satisfy 1 <= lo <= hi")
    try:
        out_dir = _validate_output_root(args.out_dir)
    except ValueError as exc:
        parser.error(str(exc))
    args.out_dir = out_dir
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        logger.error(
            "%s already exists and is not empty. Use --overwrite or delete it first.",
            out_dir,
        )
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = _make_tasks(args)
    total_tasks = len(args.fs) * max(0, args.num_data)
    logger.info(
        "Generating %d wind noise files (fs=%s, num_data=%d, duration=%.1fs) with %d workers",
        total_tasks,
        args.fs,
        args.num_data,
        args.duration,
        args.workers,
    )

    start_time = time.time()
    entries: list[tuple[str, int, int, str, float] | None] = []
    with mp.Pool(max(1, args.workers)) as pool:
        for entry in pool.imap_unordered(_worker_generate, tasks):
            entries.append(entry)

    valid = [e for e in entries if e is not None]
    logger.info("Done: %d/%d succeeded", len(valid), total_tasks)

    # 去重并按 uid 排序
    seen: set[str] = set()
    unique_entries: list[tuple[str, int, int, str, float]] = []
    for e in valid:
        if e[0] not in seen:
            seen.add(e[0])
            unique_entries.append(e)
    unique_entries.sort(key=lambda x: x[0])
    if len(unique_entries) != total_tasks:
        logger.error(
            "Failed: %d/%d wind-noise files not generated",
            total_tasks - len(unique_entries),
            total_tasks,
        )
        return 1

    # 元信息汇总
    summary: dict[str, Any] = {
        "out_dir": str(out_dir),
        "num_data_per_fs": args.num_data,
        "duration": args.duration,
        "gustiness_range": list(args.gustiness_range),
        "seed": args.seed,
        "total_files": len(unique_entries),
        "per_fs": {},
    }
    for fs in args.fs:
        fs_entries = [e for e in unique_entries if e[1] == fs]
        gusts = [e[4] for e in fs_entries]
        summary["per_fs"][f"fs{fs}"] = {
            "count": len(fs_entries),
            "n_samples": fs_entries[0][2] if fs_entries else 0,
            "gustiness_min": min(gusts) if gusts else None,
            "gustiness_max": max(gusts) if gusts else None,
            "gustiness_mean": float(np.mean(gusts)) if gusts else None,
        }
    scp_path = out_dir / "wind_noise_sources.scp"
    scp_payload = _scp_payload(unique_entries, scp_path)
    summary_path = out_dir / "wind_noise_meta.json"
    summary_payload = json.dumps(
        summary, ensure_ascii=False, indent=2, allow_nan=False
    )
    write_text_files_transactionally(
        [
            (scp_path, lambda handle: handle.write(scp_payload)),
            (summary_path, lambda handle: handle.write(summary_payload)),
        ],
        overwrite=True,
    )
    logger.info("Wrote %s with %d entries", scp_path, len(unique_entries))
    logger.info("Wrote summary: %s", summary_path)

    elapsed = time.time() - start_time
    logger.info(
        "All done. Total: %d files in %.2fs (%.4fs per file)",
        len(unique_entries),
        elapsed,
        elapsed / max(len(unique_entries), 1),
    )
    return 0 if len(unique_entries) == total_tasks else 1


if __name__ == "__main__":
    raise SystemExit(main())
