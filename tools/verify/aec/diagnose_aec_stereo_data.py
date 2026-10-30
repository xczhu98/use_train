#!/usr/bin/env python3
"""诊断 AEC stereo echo 数据集的通道质量。

扫描所有 stereo WAV，检测以下问题：
  - ch0 (echo)  全零 / 能量极低
  - ch1 (farend) 全零 / 能量极低
  - 单声道文件（被错误保存为 stereo 或实际只有 1ch）
  - 双通道都静音

用法::

    # 基于 JSON manifest（推荐，与训练配置一致）
    python tools/verify/aec/diagnose_aec_stereo_data.py \
        --json data/urgent26/echo_manifest.json \
        --output-dir ./aec_diagnosis_report

    # 直接扫描目录（自动递归找 .wav）
    python tools/verify/aec/diagnose_aec_stereo_data.py \
        --dir  D:\\BaiduNetdiskDownload\\echo_corpus_10s \
        --output-dir ./aec_diagnosis_report

    # 只输出报告，不复制文件
    python tools/verify/aec/diagnose_aec_stereo_data.py \
        --json data/urgent26/echo_manifest.json \
        --output-dir ./report --no-copy

    # 删除问题文件（dry-run 预览，不实际删除）
    python tools/verify/aec/diagnose_aec_stereo_data.py \
        --json data/urgent26/echo_manifest.json \
        --output-dir ./report --delete-bad

    # 真正删除问题文件（危险操作！）
    python tools/verify/aec/diagnose_aec_stereo_data.py \
        --json data/urgent26/echo_manifest.json \
        --output-dir ./report --delete-bad --no-dry-run

输出结构::

    aec_diagnosis_report/
        report.json           # 详细统计
        report.txt            # 人类可读摘要
        manifest_clean.json   # 剔除问题文件后的新 manifest（仅 --json 模式）
        deleted_files.txt     # 被删除的文件列表（仅 --delete-bad 模式）
        normal/               # 正常文件（复制）
        silent_ch0/           # ch0 (echo)  静音
        silent_ch1/           # ch1 (farend) 静音
        silent_both/          # 双通道静音
        mono/                 # 实际只有 1 个通道
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.shared.file_scan import scan_audio_files  # noqa: E402

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
SILENT_THRESHOLD = 1e-6  # 与 dynamic_dataset 保持一致


def _max_abs(arr: np.ndarray) -> float:
    return float(np.max(np.abs(arr))) if arr.size > 0 else 0.0


def _fmt_e(v: float) -> str:
    return f"{v:.2e}"


# ------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------
class Category:
    NORMAL = "normal"
    SILENT_CH0 = "silent_ch0"
    SILENT_CH1 = "silent_ch1"
    SILENT_BOTH = "silent_both"
    MONO = "mono"
    READ_ERROR = "read_error"


CATEGORIES = [
    Category.NORMAL,
    Category.SILENT_CH0,
    Category.SILENT_CH1,
    Category.SILENT_BOTH,
    Category.MONO,
    Category.READ_ERROR,
]

CATEGORY_LABELS = {
    Category.NORMAL: "正常（双通道有能量）",
    Category.SILENT_CH0: "ch0 (echo)  静音",
    Category.SILENT_CH1: "ch1 (farend) 静音",
    Category.SILENT_BOTH: "双通道静音",
    Category.MONO: "单声道文件（< 2ch）",
    Category.READ_ERROR: "读取失败",
}


# ------------------------------------------------------------------
# Core scan
# ------------------------------------------------------------------
def scan_file(path: Path) -> dict:
    """扫描单个 stereo 文件，返回诊断结果字典。"""
    result = {
        "path": str(path.resolve()),
        "rel_path": str(path),
        "category": Category.READ_ERROR,
        "channels": 0,
        "frames": 0,
        "samplerate": 0,
        "duration_sec": 0.0,
        "ch0_max": 0.0,
        "ch1_max": 0.0,
        "note": "",
    }
    try:
        data, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        result["note"] = f"读取失败: {exc}"
        return result

    nch = int(data.shape[1])
    nframes = int(data.shape[0])
    result["channels"] = nch
    result["frames"] = nframes
    result["samplerate"] = int(sr)
    result["duration_sec"] = round(nframes / sr, 3) if sr > 0 else 0.0

    if nch < 2:
        result["category"] = Category.MONO
        if nch == 1:
            result["ch0_max"] = _max_abs(data[:, 0])
        result["note"] = f"只有 {nch} 个通道，需要至少 2ch"
        return result

    ch0 = data[:, 0]
    ch1 = data[:, 1]
    m0 = _max_abs(ch0)
    m1 = _max_abs(ch1)
    result["ch0_max"] = m0
    result["ch1_max"] = m1

    silent_0 = m0 < SILENT_THRESHOLD
    silent_1 = m1 < SILENT_THRESHOLD

    if silent_0 and silent_1:
        result["category"] = Category.SILENT_BOTH
        result["note"] = f"ch0={_fmt_e(m0)}, ch1={_fmt_e(m1)} 均低于阈值"
    elif silent_0:
        result["category"] = Category.SILENT_CH0
        result["note"] = f"ch0 (echo)  静音 {_fmt_e(m0)}, ch1 (farend)={_fmt_e(m1)}"
    elif silent_1:
        result["category"] = Category.SILENT_CH1
        result["note"] = f"ch1 (farend) 静音 {_fmt_e(m1)}, ch0 (echo)={_fmt_e(m0)}"
    else:
        result["category"] = Category.NORMAL
        result["note"] = f"ch0={_fmt_e(m0)}, ch1={_fmt_e(m1)}"

    return result


# ------------------------------------------------------------------
# Manifest parsing (matches DynamicMixingDataset)
# ------------------------------------------------------------------
def parse_manifest(json_path: Path) -> list[Path]:
    text = json_path.read_text(encoding="utf-8")
    obj = json.loads(text)
    pairs = obj.get("pairs", [])
    if not pairs:
        raise SystemExit(f"manifest 中无 'pairs' 字段: {json_path}")

    paths: list[Path] = []
    base = json_path.parent.resolve()
    for item in pairs:
        if isinstance(item, str):
            p = Path(item)
            if not p.is_absolute():
                p = base / p
            paths.append(p.resolve())
        elif isinstance(item, (list, tuple)) and len(item) >= 1:
            # stereo mode: first path == second path
            p = Path(item[0])
            if not p.is_absolute():
                p = base / p
            paths.append(p.resolve())
        else:
            raise SystemExit(f"不支持的 manifest 条目类型: {type(item).__name__}")

    # deduplicate (stereo mode has duplicate paths)
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in paths:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def collect_wavs(root: Path) -> list[Path]:
    return scan_audio_files(root, exts=(".wav",), recursive=True)


# ------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------
def generate_text_report(stats: dict, total: int) -> str:
    lines = [
        "=" * 70,
        "AEC Stereo Echo 数据质量诊断报告",
        "=" * 70,
        f"总文件数: {total}",
        "",
        "按问题类型统计:",
        "-" * 50,
    ]
    for cat in CATEGORIES:
        cnt = stats["by_category"].get(cat, 0)
        pct = 100.0 * cnt / total if total > 0 else 0.0
        label = CATEGORY_LABELS.get(cat, cat)
        lines.append(f"  {cat:15s}  {cnt:6d} ({pct:5.1f}%)  {label}")

    lines += ["", "按子目录统计 (silent_ch1 占比):", "-" * 50]
    for subdir, sub in sorted(stats["by_subdir"].items(), key=lambda x: -x[1]["silent_ch1_rate"]):
        rate = sub["silent_ch1_rate"]
        total_sub = sub["total"]
        silent_ch1 = sub.get(Category.SILENT_CH1, 0)
        lines.append(
            f"  {subdir:40s}  {silent_ch1:4d}/{total_sub:4d}  ({rate*100:5.1f}%)"
        )

    lines += ["", "=" * 70]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="诊断 AEC stereo echo 数据集的通道质量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --json data/urgent26/echo_manifest.json --output-dir ./report
  %(prog)s --dir D:\\echo_corpus --output-dir ./report --no-copy
        """,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", type=Path, help="real_farend_echo_pair_json manifest 路径")
    src.add_argument("--dir", type=Path, help="直接扫描的目录（递归找 .wav）")
    ap.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    ap.add_argument(
        "--no-copy",
        action="store_true",
        help="只生成报告，不复制文件到分类文件夹",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行扫描的 worker 数（默认 1，设为 0 则用 CPU 核数）",
    )
    ap.add_argument(
        "--delete-bad",
        action="store_true",
        help="删除被诊断为问题文件的 WAV（默认 dry-run，需配合 --no-dry-run）",
    )
    ap.add_argument(
        "--no-dry-run",
        action="store_true",
        help="真正执行删除/修改操作（默认 dry-run 预览）",
    )
    args = ap.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect files
    if args.json:
        files = parse_manifest(args.json)
        print(f"从 manifest 读取: {args.json}")
    else:
        files = collect_wavs(args.dir)
        print(f"从目录扫描: {args.dir}")

    print(f"待扫描文件数: {len(files)}")
    if not files:
        raise SystemExit("没有文件需要扫描")

    # Scan
    results: list[dict] = []
    if args.workers != 1:
        import concurrent.futures
        _max_workers = args.workers if args.workers > 0 else None
        with concurrent.futures.ThreadPoolExecutor(max_workers=_max_workers) as ex:
            futures = {ex.submit(scan_file, p): p for p in files}
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())
    else:
        for i, p in enumerate(files, 1):
            results.append(scan_file(p))
            if i % 100 == 0 or i == len(files):
                print(f"  扫描进度: {i}/{len(files)}")

    # Aggregate stats
    stats = {
        "by_category": defaultdict(int),
        "by_subdir": defaultdict(lambda: defaultdict(int)),
    }
    for r in results:
        cat = r["category"]
        stats["by_category"][cat] += 1
        subdir = str(Path(r["path"]).parent)
        stats["by_subdir"][subdir]["total"] += 1
        stats["by_subdir"][subdir][cat] += 1

    # Compute silent_ch1_rate per subdir
    for subdir, sub in stats["by_subdir"].items():
        total_sub = sub["total"]
        silent_ch1 = sub.get(Category.SILENT_CH1, 0)
        sub["silent_ch1_rate"] = silent_ch1 / total_sub if total_sub > 0 else 0.0

    # Write JSON report
    report = {
        "total_files": len(results),
        "silent_threshold": SILENT_THRESHOLD,
        "by_category": dict(stats["by_category"]),
        "by_subdir": {
            k: {kk: vv for kk, vv in v.items() if not callable(vv)}
            for k, v in stats["by_subdir"].items()
        },
        "details": results,
    }
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 JSON 报告: {json_path}")

    # Write text report
    txt_path = out_dir / "report.txt"
    txt_path.write_text(generate_text_report(stats, len(results)), encoding="utf-8")
    print(f"写入文本报告: {txt_path}")

    # Collect bad files
    bad_results = [r for r in results if r["category"] != Category.NORMAL]
    bad_paths = [Path(r["path"]) for r in bad_results]

    # ------------------------------------------------------------------
    # Delete bad files (optional)
    # ------------------------------------------------------------------
    if args.delete_bad:
        dry_run = not args.no_dry_run
        mode_str = "【预览模式，不会实际删除】" if dry_run else "【真正删除】"
        print(f"\n{mode_str} 处理问题文件...")
        print(f"  待处理问题文件数: {len(bad_paths)}")

        deleted_log: list[str] = []
        deleted_count = 0
        for r in bad_results:
            src_p = Path(r["path"])
            cat = r["category"]
            line = f"[{cat}] {src_p}  ({r['note']})"
            if dry_run:
                deleted_log.append("[DRY-RUN] " + line)
            else:
                try:
                    src_p.unlink()
                    deleted_log.append("[DELETED] " + line)
                    deleted_count += 1
                except Exception as exc:
                    deleted_log.append(f"[FAILED]  {line}  error={exc}")

        # Write deleted log
        del_txt = out_dir / "deleted_files.txt"
        del_txt.write_text("\n".join(deleted_log) + "\n", encoding="utf-8")
        print(f"  写入删除日志: {del_txt}")
        if not dry_run:
            print(f"  实际删除成功: {deleted_count}/{len(bad_paths)}")

        # Write clean manifest (only when --json is used)
        if args.json and not dry_run:
            # Re-parse original manifest and filter out deleted files
            original_obj = json.loads(args.json.read_text(encoding="utf-8"))
            original_pairs = original_obj.get("pairs", [])
            bad_set = {str(p.resolve()) for p in bad_paths}
            base = args.json.parent.resolve()

            clean_pairs: list = []
            for item in original_pairs:
                if isinstance(item, str):
                    p = Path(item)
                    if not p.is_absolute():
                        p = base / p
                    if str(p.resolve()) not in bad_set:
                        clean_pairs.append(item)
                elif isinstance(item, (list, tuple)) and len(item) >= 1:
                    p = Path(item[0])
                    if not p.is_absolute():
                        p = base / p
                    if str(p.resolve()) not in bad_set:
                        clean_pairs.append(item)
                else:
                    clean_pairs.append(item)

            clean_manifest = {"pairs": clean_pairs}
            clean_path = out_dir / "manifest_clean.json"
            clean_path.write_text(
                json.dumps(clean_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"  写入过滤后 manifest: {clean_path}")
            print(f"    原始条目: {len(original_pairs)} -> 过滤后: {len(clean_pairs)}")

    # ------------------------------------------------------------------
    # Copy files to category folders (optional)
    # ------------------------------------------------------------------
    # When --delete-bad --no-dry-run is used, original files are already deleted.
    # Skip copying to avoid "No such file or directory" errors.
    if not args.no_copy and not (args.delete_bad and args.no_dry_run):
        print("\n复制文件到分类文件夹...")
        cat_dirs: dict[str, Path] = {}
        for cat in CATEGORIES:
            d = out_dir / cat
            d.mkdir(exist_ok=True)
            cat_dirs[cat] = d

        copied = defaultdict(int)
        for r in results:
            cat = r["category"]
            src_p = Path(r["path"])
            dst = cat_dirs[cat] / src_p.name
            # handle name collision
            if dst.exists():
                stem = src_p.stem
                suffix = src_p.suffix
                dst = cat_dirs[cat] / f"{stem}_{copied[cat]:04d}{suffix}"
            try:
                shutil.copy2(str(src_p), str(dst))
                copied[cat] += 1
            except Exception as exc:
                print(f"  复制失败 {src_p}: {exc}")

        for cat in CATEGORIES:
            cnt = copied[cat]
            if cnt > 0:
                print(f"  [{cat}] 复制了 {cnt} 个文件 -> {cat_dirs[cat]}")
    elif not args.no_copy and args.delete_bad and args.no_dry_run:
        print("\n跳过复制: 已使用 --delete-bad --no-dry-run，原始文件已删除。")

    # Summary to stdout
    print("\n" + "=" * 50)
    print("诊断摘要:")
    for cat in CATEGORIES:
        cnt = stats["by_category"].get(cat, 0)
        if cnt > 0:
            pct = 100.0 * cnt / len(results)
            print(f"  {CATEGORY_LABELS[cat]:20s}: {cnt:5d} ({pct:5.1f}%)")
    print("=" * 50)

    # Exit code: non-zero if any problem found
    problem_count = len(bad_results)
    if problem_count > 0:
        print(f"\n发现 {problem_count} 个问题文件，请查看报告并人工确认。")
        if args.delete_bad and not args.no_dry_run:
            print("提示: 当前为 dry-run 模式，文件并未实际删除。要真正删除，请追加 --no-dry-run")
        return 1
    else:
        print("\n所有文件均正常，未发现通道静音问题。")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        pass
