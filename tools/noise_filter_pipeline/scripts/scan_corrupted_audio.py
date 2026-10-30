#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扫描正式数据集注册表中异常/损坏的音频文件。

用法:
  cd /path/to/use_train
  python tools/noise_filter_pipeline/scripts/scan_corrupted_audio.py

  # 显式覆盖 registry（例如部署副本 registry_local.yaml）
  python tools/noise_filter_pipeline/scripts/scan_corrupted_audio.py \
    --registry tools/noise_filter_pipeline/config/noise.yaml

  # 扫描并直接删除异常文件（谨慎使用）
  python tools/noise_filter_pipeline/scripts/scan_corrupted_audio.py \
    --registry tools/noise_filter_pipeline/config/noise.yaml --delete

输出:
  - 每个数据集的总文件数、异常文件数、异常比例
  - 异常文件列表（默认保存到 ./corrupted_audio_files_<timestamp>.txt）
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import soundfile as sf
import yaml

# 把 tools 加入 path，以便 import noise_filter_pipeline
_USE_TRAIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_TOOLS_DIR = os.path.join(_USE_TRAIN_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from noise_filter_pipeline.core.registry import (
    AUDIO_EXTS,
    DEFAULT_REGISTRY_PATH,
    DatasetRegistry,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描数据集中异常/损坏的音频文件")
    parser.add_argument(
        "--registry",
        default=DEFAULT_REGISTRY_PATH,
        help="数据集注册表路径 (YAML)，默认便携模板 noise.yaml；部署时应指向本地副本",
    )
    parser.add_argument(
        "--min-size-kb",
        type=float,
        default=5.0,
        help="小于该大小的文件视为异常（单位 KB），默认 5",
    )
    parser.add_argument(
        "--min-duration-sec",
        type=float,
        default=0.5,
        help="时长小于该值的文件视为异常（单位秒），默认 0.5",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="并行扫描线程数，默认 16",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="异常文件列表输出路径；默认 ./corrupted_audio_files_<timestamp>.txt",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="只扫描指定 dataset id，默认扫描所有 enabled 数据集",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="扫描后直接删除异常文件（默认只列出，不删除）",
    )
    # --dry-run / --no-dry-run 互斥且共享同一 dest：默认 dry-run（安全方向），
    # 显式 --no-dry-run 才真正删除；两flag同时给出时 argparse 直接报错，
    # 不再静默取其一
    delete_mode = parser.add_mutually_exclusive_group()
    delete_mode.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="只模拟删除，不真正删除（默认行为）",
    )
    delete_mode.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="配合 --delete 时真正删除异常文件",
    )
    return parser.parse_args()


def check_file(fpath: Path, min_size_bytes: int, min_duration_sec: float) -> Tuple[bool, Optional[str]]:
    """检查单个文件是否异常。返回 (is_bad, reason)。"""
    try:
        size = fpath.stat().st_size
    except OSError as e:
        return True, f"无法获取文件信息: {e}"

    if size < min_size_bytes:
        return True, f"文件过小: {size} bytes (< {min_size_bytes} bytes)"

    try:
        info = sf.info(str(fpath))
    except Exception as e:
        return True, f"无法读取音频信息: {e}"

    if info.duration is not None and info.duration < min_duration_sec:
        return True, f"时长过短: {info.duration:.3f}s (< {min_duration_sec}s)"

    return False, None


def scan_dataset(
    dataset_id: str,
    base_path: str,
    pattern: str,
    min_size_bytes: int,
    min_duration_sec: float,
    workers: int,
) -> Tuple[int, List[Tuple[str, str]]]:
    """扫描单个数据集，返回 (总文件数, [(path, reason), ...])。"""
    base = Path(base_path)
    if not base.exists():
        logger.error("数据集 %s 路径不存在: %s", dataset_id, base_path)
        return 0, []

    files = sorted(base.glob(pattern))
    files = [f for f in files if f.suffix.lower() in AUDIO_EXTS]
    total = len(files)

    results: List[Tuple[bool, Optional[str]]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        results = list(
            executor.map(
                lambda p: check_file(p, min_size_bytes, min_duration_sec),
                files,
            )
        )

    bad = [(str(f), reason) for f, (is_bad, reason) in zip(files, results) if is_bad]
    return total, bad


def _is_within_root(path: str, root: str) -> bool:
    """判断 path 的真实路径是否位于 root 之内（同 policy_processor 删除保护口径）。"""
    real_root = os.path.realpath(root)
    try:
        under_root = os.path.commonpath([os.path.realpath(path), real_root]) == real_root
    except ValueError:
        # 不同盘符/路径风格无法比较时按不在根目录内处理
        under_root = False
    return under_root


def delete_bad_files(
    all_bad: List[Tuple[str, str, str]],
    *,
    dry_run: bool,
    dataset_roots: Dict[str, str],
) -> Tuple[int, int]:
    """删除异常文件。

    Args:
        all_bad: (dataset_id, path, reason) 元组列表。结构化直接传递，
            不做字符串解析——含 tab 的文件名不会再被截断而误删其他文件
        dry_run: True 时只打印不删除
        dataset_roots: dataset_id → 该数据集根目录。只删除根目录内的文件
            （对照 B4 policy_processor._safe_delete_audio）；根目录未知或
            路径不在其中的记录一律跳过并告警，避免误删 registry 之外的数据

    Returns:
        (attempted, removed)
    """
    attempted = 0
    removed = 0
    for dataset_id, path, _reason in all_bad:
        if not path:
            continue
        root = dataset_roots.get(dataset_id)
        if not root or not _is_within_root(path, root):
            logger.warning("跳过数据集 %s 根目录之外的异常文件: %s", dataset_id, path)
            continue
        attempted += 1
        if dry_run:
            logger.info("[dry-run] 将删除: %s", path)
            continue
        try:
            os.remove(path)
            removed += 1
            logger.info("已删除: %s", path)
        except OSError as e:
            logger.error("删除失败 %s: %s", path, e)
    return attempted, removed


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    registry_path = args.registry
    if not os.path.isabs(registry_path):
        registry_path = os.path.join(os.getcwd(), registry_path)

    registry = DatasetRegistry(registry_path)

    min_size_bytes = int(args.min_size_kb * 1024)
    # (dataset_id, path, reason) 结构化元组，删除时直接传递不做字符串解析；
    # 展示/落盘用的 "[id] path\treason" 字符串在输出处再格式化
    all_bad: List[Tuple[str, str, str]] = []
    summary: Dict[str, Dict[str, Any]] = {}
    # dataset_id → 已扫描根目录，供 --delete 的根目录约束使用
    dataset_roots: Dict[str, str] = {}

    for entry in registry.entries.values():
        if args.dataset and entry.id != args.dataset:
            continue
        if not entry.enabled:
            continue
        if entry.type != "directory":
            logger.info("跳过非 directory 数据集: %s (%s)", entry.id, entry.type)
            continue

        logger.info("扫描数据集 %s: %s", entry.id, entry.path)
        # 相对路径按 registry 文件所在目录解析（与 Stage A / validator
        # 同口径），避免按 CWD 解析导致路径错位
        dataset_path = registry.resolve_path(entry.path)
        dataset_roots[entry.id] = dataset_path
        total, bad = scan_dataset(
            entry.id,
            dataset_path,
            entry.input_pattern or "**/*.wav",
            min_size_bytes,
            args.min_duration_sec,
            args.workers,
        )
        summary[entry.id] = {"total": total, "bad": len(bad), "ratio": len(bad) / max(1, total)}
        for path, reason in bad:
            all_bad.append((entry.id, path, reason))

    # 打印汇总
    print("\n" + "=" * 80)
    print("扫描结果汇总")
    print("=" * 80)
    grand_total = 0
    grand_bad = 0
    for ds_id, info in sorted(summary.items()):
        grand_total += info["total"]
        grand_bad += info["bad"]
        print(
            f"{ds_id:30s}  total={info['total']:8d}  bad={info['bad']:8d}  "
            f"ratio={info['ratio']*100:6.2f}%"
        )
    print("-" * 80)
    print(
        f"{'TOTAL':30s}  total={grand_total:8d}  bad={grand_bad:8d}  "
        f"ratio={grand_bad/max(1, grand_total)*100:6.2f}%"
    )
    print("=" * 80)

    # 写出异常文件列表
    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"./corrupted_audio_files_{timestamp}.txt"

    if all_bad:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(f"[{ds}] {p}\t{r}" for ds, p, r in all_bad) + "\n")
        print(f"\n异常文件列表已保存: {output_path} ({len(all_bad)} 条)")
    else:
        print("\n未发现异常文件")

    # 删除异常文件
    if args.delete:
        dry_run = args.dry_run
        attempted, removed = delete_bad_files(
            all_bad, dry_run=dry_run, dataset_roots=dataset_roots
        )
        if dry_run:
            print(f"\n[dry-run] 共 {attempted} 个异常文件待删除，未实际删除")
        else:
            print(f"\n实际删除: {removed}/{attempted} 个异常文件")

    return 0


if __name__ == "__main__":
    sys.exit(main())
