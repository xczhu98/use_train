#!/usr/bin/env python3
"""扫描目录下（含子目录）全部音频，生成 ``real_farend_echo_pair_json`` 用的清单。

默认与 ``DynamicMixingDataset`` 的 **立体声单文件** 模式一致：``pairs`` 中每项为 **一条路径字符串**
（等价于同一文件 ch0=echo、ch1=farend），见 ``tasks.aec.data.dynamic_dataset.parse_real_farend_echo_pair_file``。

默认会完整解码并过滤 NaN/Inf、零长度、过静音、幅度越界、DC 偏移的音频。

用法::

    python -m tools.prepare_data_tools.generators.echo.build_real_farend_echo_pair_manifest /data/corpus/echo_corpus_10s
    python -m tools.prepare_data_tools.generators.echo.build_real_farend_echo_pair_manifest /data/corpus --out /data/corpus/manifest.json
    python -m tools.prepare_data_tools.generators.echo.build_real_farend_echo_pair_manifest /data/corpus --relative
    python -m tools.prepare_data_tools.generators.echo.build_real_farend_echo_pair_manifest /data/corpus --exts .wav .flac
    python -m tools.prepare_data_tools.generators.echo.build_real_farend_echo_pair_manifest /data/corpus --no_validate

按 ``--val_ratio`` / ``--seed`` 可将 ``pairs`` 随机划分为 train/val 两份：划分开启时默认输出
``real_farend_echo_pair_train.json`` / ``real_farend_echo_pair_val.json``
（自定义 ``--out`` 时在其 stem 后加 ``_train`` / ``_val``）。

在 YAML 里设置::

    real_farend_echo_pair_json: /path/to/manifest.json

注意：训练侧 ``real_farend_echo_pair_json`` 字段必须指向 **.json** 文件
（``tasks.aec.data.dynamic_dataset.parse_real_farend_echo_pair_file`` 强制 .json 后缀），
因此划分产物同样保留 JSON 格式。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from tools.shared.file_scan import scan_audio_files  # noqa: E402
from tools.shared.audio_io import load_audio  # noqa: E402
from core.utils.file_transaction import write_text_files_transactionally  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


def collect_audio(root: Path, exts: tuple[str, ...]) -> list[Path]:
    return scan_audio_files(root, exts=exts, recursive=True)


def _validate_one(args: tuple[str, float, float, float]) -> tuple[str, bool]:
    path, max_peak, min_mean_abs, dc_offset_threshold = args
    try:
        load_audio(path, max_peak=max_peak, min_mean_abs=min_mean_abs, dc_offset_threshold=dc_offset_threshold)
        return path, True
    except Exception:
        return path, False


def filter_valid_audio(
    paths: list[Path],
    max_peak: float,
    min_mean_abs: float,
    dc_offset_threshold: float,
    workers: int = 8,
) -> tuple[list[Path], int]:
    """Return (valid_paths, skipped_count) after multi-threaded full-decode validation."""
    str_paths = [str(p) for p in paths]
    jobs = [(p, max_peak, min_mean_abs, dc_offset_threshold) for p in str_paths]
    valid_mask = [False] * len(paths)
    skipped = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_validate_one, job): i for i, job in enumerate(jobs)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="validating"):
            i = futures[fut]
            _, ok = fut.result()
            valid_mask[i] = ok
            if not ok:
                skipped += 1

    valid_paths = [p for p, ok in zip(paths, valid_mask) if ok]
    return valid_paths, skipped


def to_json_path(p: Path, *, relative_to: Path | None) -> str:
    if relative_to is not None:
        try:
            rel = p.resolve().relative_to(relative_to.resolve())
            s = rel.as_posix()
        except ValueError:
            s = p.resolve().as_posix()
    else:
        s = p.resolve().as_posix()
    return s


def dump_manifest(path: Path, entries: list[str]) -> None:
    """经共享文件事务写出 manifest（staging + fsync + 原子 replace）。"""
    obj = {"pairs": entries}
    payload = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    write_text_files_transactionally(
        [(path, lambda handle: handle.write(payload))], overwrite=True
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="递归收集音频文件，写入 real_farend_echo_pair_json 清单（默认：每条为立体声单路径）。"
    )
    ap.add_argument(
        "root_dir",
        nargs="+",
        type=Path,
        help="一个或多个根目录（可合并多个语料；输出路径默认取第一个根）",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="输出 .json 路径；默认 <root_dir>/real_farend_echo_pair_manifest.json；"
             "开启 --val_ratio 划分时默认输出 real_farend_echo_pair_train.json / real_farend_echo_pair_val.json",
    )
    ap.add_argument(
        "--relative",
        action="store_true",
        help="写入相对 root_dir 的路径（便于整包迁移）；默认写绝对路径",
    )
    ap.add_argument(
        "--exts",
        nargs="+",
        default=[".wav"],
        metavar=".EXT",
        help="收集的音频扩展名（默认：.wav）；可指定多个，例如 --exts .wav .flac",
    )
    ap.add_argument(
        "--no_validate",
        action="store_true",
        help="关闭完整解码校验（默认开启，过滤 NaN/Inf、零长度、过静音、幅度越界、DC 偏移）",
    )
    ap.add_argument(
        "--validate_max_peak",
        type=float,
        default=1.05,
        help="验证时最大允许峰值（默认：1.05）",
    )
    ap.add_argument(
        "--validate_min_mean_abs",
        type=float,
        default=1e-6,
        help="验证时最小平均幅度；低于此值视为过静音（默认：1e-6）",
    )
    ap.add_argument(
        "--validate_dc_offset_threshold",
        type=float,
        default=0.5,
        help="验证时最大允许 DC 偏移（默认：0.5）",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="验证时并行线程数（默认：8）",
    )
    ap.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="验证集比例 [0,1)，0 表示不划分（默认：0.05）",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="train/val 划分随机种子（默认：42）",
    )
    args = ap.parse_args(argv)
    if not (0 <= args.val_ratio < 1):
        raise SystemExit(f"--val_ratio 必须在 [0,1) 范围内: {args.val_ratio}")
    roots = args.root_dir
    root = roots[0]
    out = args.out
    if out is None:
        out = root / "real_farend_echo_pair_manifest.json"

    exts = tuple(e if e.startswith(".") else f".{e}" for e in args.exts)
    files: list[Path] = []
    for r in roots:
        files.extend(collect_audio(r, exts))
    if not files:
        raise SystemExit(f"未找到任何 {exts} 音频: {[str(r.resolve()) for r in roots]}")

    if not args.no_validate:
        files, skipped = filter_valid_audio(
            files,
            max_peak=args.validate_max_peak,
            min_mean_abs=args.validate_min_mean_abs,
            dc_offset_threshold=args.validate_dc_offset_threshold,
            workers=max(1, args.workers),
        )
        print(f"validation: total={len(files) + skipped}, kept={len(files)}, skipped={skipped}")
        if not files:
            raise SystemExit(f"验证后无有效音频: {root.resolve()}")

    rel_base = root.resolve() if args.relative else None
    pairs = [to_json_path(p, relative_to=rel_base) for p in files]

    if args.val_ratio > 0:
        n_val = max(1, int(len(pairs) * args.val_ratio))
        if n_val >= len(pairs):
            raise SystemExit(
                f"val 集规模 {n_val} 不小于总样本数 {len(pairs)}，请调小 --val_ratio 或增大语料"
            )
        rng = random.Random(args.seed)
        rng.shuffle(pairs)
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]
        if args.out is None:
            out_train = root / "real_farend_echo_pair_train.json"
            out_val = root / "real_farend_echo_pair_val.json"
        else:
            out_train = out.with_name(f"{out.stem}_train{out.suffix}")
            out_val = out.with_name(f"{out.stem}_val{out.suffix}")
        dump_manifest(out_train, train_pairs)
        dump_manifest(out_val, val_pairs)
        print(f"file_count_total={len(pairs)}")
        print(f"train_count={len(train_pairs)}")
        print(f"val_count={len(val_pairs)}")
        print(f"wrote: {out_train.resolve()}")
        print(f"wrote: {out_val.resolve()}")
    else:
        dump_manifest(out, pairs)
        print(f"file_count={len(pairs)}")
        print(f"wrote: {out.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        pass