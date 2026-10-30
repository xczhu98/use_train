#!/usr/bin/env python3
"""合并多个 ``real_farend_echo_pair_json`` manifest（支持真实 + 仿真混合）。

用法::

    python -m tools.prepare_data_tools.generators.echo.merge_farend_echo_pair_manifests \\
        --inputs real_manifest.json synthetic_manifest.json \\
        --output merged_manifest.json

各源按样本数等权合并（去重后 shuffle）。加权采样建议在训练时通过 separate JSONs 实现。
输出条目统一写成按源 manifest 目录解析后的绝对路径，避免相对条目在新 manifest 下错位。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from core.utils.file_transaction import write_text_files_transactionally  # noqa: E402


def load_manifest(path: Path) -> list[str]:
    """读取唯一支持的 canonical stereo echo-pair manifest。"""
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: manifest root must be an object")
    unknown = set(obj) - {"pairs", "meta"}
    if unknown:
        raise ValueError(f"{path}: unknown manifest keys: {sorted(unknown)!r}")
    if "meta" in obj and not isinstance(obj["meta"], dict):
        raise ValueError(f"{path}: manifest meta must be an object")
    if "pairs" not in obj or not isinstance(obj["pairs"], list):
        raise ValueError(f"{path}: manifest must contain a pairs list")
    pairs = obj["pairs"]
    if not pairs:
        raise ValueError(f"Empty manifest: {path}")
    entries: list[str] = []
    for index, item in enumerate(pairs):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"{path}: pairs[{index}] must be a non-empty stereo path string"
            )
        entries.append(item.strip())
    return entries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="合并多个 farend/echo pair manifest。"
    )
    ap.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="输入 JSON 清单文件（一个或多个）",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="输出合并后的 JSON 路径",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle 随机种子",
    )
    ap.add_argument(
        "--dedup",
        action="store_true",
        default=True,
        help="显式开启去重（默认已开启，此 flag 仅用于明确意图）",
    )
    ap.add_argument(
        "--no-dedup",
        dest="dedup",
        action="store_false",
        help="关闭去重",
    )
    args = ap.parse_args(argv)

    all_entries: list[tuple[Path, str]] = []  # (base_dir, entry)
    source_metas: list[dict] = []

    for p in args.inputs:
        entries = load_manifest(p)
        base_dir = p.parent
        all_entries.extend((base_dir, e) for e in entries)
        source_metas.append({
            "path": str(p.resolve()),
            "count": len(entries),
        })
        print(f"Loaded {len(entries):>6} pairs from {p}")

    # Deduplicate
    if args.dedup:
        seen: set[str] = set()
        deduped: list[tuple[Path, str]] = []
        for base_dir, e in all_entries:
            # 相对路径按所属 manifest 所在目录解析，避免依赖 CWD
            key = (base_dir / e).resolve().as_posix()
            if key not in seen:
                seen.add(key)
                deduped.append((base_dir, e))
        removed = len(all_entries) - len(deduped)
        if removed:
            print(f"Removed {removed} duplicate entries")
        all_entries = deduped

    rng = random.Random(args.seed)
    # 合并后源 manifest 的 base_dir 丢失，保留相对路径会被 loader 按新 manifest 的 parent
    # 解析而错位；这里按所属源 manifest 目录解析成绝对路径后再写出。
    entries_only = [(base_dir / e).resolve().as_posix() for base_dir, e in all_entries]
    rng.shuffle(entries_only)

    out_obj = {
        "pairs": entries_only,
        "meta": {
            "merged": True,
            "sources": source_metas,
            "total_pairs": len(all_entries),
            "seed": args.seed,
        },
    }

    payload = json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n"
    write_text_files_transactionally(
        [(args.output, lambda handle: handle.write(payload))], overwrite=True
    )
    print(f"Wrote {len(all_entries)} pairs to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        pass