#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将目录树内「文件名 / 文件夹名」中的空格、逗号、句号等不安全字符统一为下划线，
与 Step4 归档 sanitize 规则一致，便于 prepare_scp 四列格式。

规则（基于 utils.sanitize.sanitize_basename_for_rename，额外做中文转拼音）：
  1. _chinese_to_pinyin 先将中文字符转为无声调拼音，中间值仍保留空格、括号等
     非中文字符（如 "中文 音频(1)" -> "zhongwen yinpin(1)"），不保留中文；
  2. _sanitize_name 再仅保留 Unicode 字母数字、下划线、连字符，其余字符统一为
     下划线（上述示例最终为 "zhongwen_yinpin_1"）。

依赖：pypinyin（仅当名称包含中文时才需要 import）。

文件：只处理主文件名（stem），扩展名不变。

重命名顺序：自深向浅。

用法:
  python tools/noise_filter_pipeline/scripts/rename_paths_replace_spaces.py /data/noise_root
  PYTHONPATH=tools python -m noise_filter_pipeline.cli rename_spaces --root /data/noise_root --dry-run
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_CHINESE_CHARS = re.compile(r"[\u4e00-\u9fff]")


def _chinese_to_pinyin(text: str) -> str:
    """中文字符转拼音（无声调），非中文片段原样保留。

    例如 "中文 音频(1)" -> "zhongwen yinpin(1)"；此处空格和括号尚未
    sanitize，后续 _sanitize_name 才得到 "zhongwen_yinpin_1"。
    """
    if not _CHINESE_CHARS.search(text):
        return text
    try:
        from pypinyin import lazy_pinyin
    except ImportError as e:
        raise RuntimeError(
            "文件名包含中文，需要 pypinyin 转拼音：pip install pypinyin"
        ) from e
    # lazy_pinyin 逐段转换：中文转拼音，非中文原样保留
    return "".join(lazy_pinyin(text))


def _import_sanitize():
    _tools = Path(__file__).resolve().parent.parent.parent  # .../use_train/tools
    if str(_tools) not in sys.path:
        sys.path.insert(0, str(_tools))
    from noise_filter_pipeline.utils.sanitize import sanitize_basename_for_rename

    return sanitize_basename_for_rename


def _sanitize_name(name: str, is_file: bool) -> str:
    """完整清洗：先中文转拼音，再走 sanitize 规则。"""
    sanitize_basename_for_rename = _import_sanitize()
    return sanitize_basename_for_rename(_chinese_to_pinyin(name), is_file)


def collect_rename_plan(root: Path) -> list[tuple[Path, Path]]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"不是目录: {root}")

    candidates: list[Path] = []
    for p in root.rglob("*"):
        if not p.exists():
            continue
        new_name = _sanitize_name(p.name, p.is_file())
        if new_name == p.name:
            continue
        candidates.append(p)

    def depth(p: Path) -> int:
        try:
            return len(p.relative_to(root).parts)
        except ValueError:
            return len(p.parts)

    candidates.sort(key=depth, reverse=True)

    plan: list[tuple[Path, Path]] = []
    seen_dst: set[str] = set()
    for p in candidates:
        new_name = _sanitize_name(p.name, p.is_file())
        dst = p.parent / new_name
        key = str(dst.resolve())
        if key in seen_dst:
            log.warning("跳过（目标已在本轮计划中）: %s -> %s", p, dst)
            continue
        if dst.exists() and dst.resolve() != p.resolve():
            log.error("目标已存在，跳过: %s -> %s", p, dst)
            continue
        plan.append((p, dst))
        seen_dst.add(key)
    return plan


def run_rename(roots: list[Path], dry_run: bool = False) -> tuple[int, int]:
    ok, skip = 0, 0
    for root in roots:
        root = Path(root).expanduser()
        try:
            plan = collect_rename_plan(root)
        except ValueError as e:
            log.error("%s", e)
            skip += 1
            continue
        if not plan:
            log.info("无需重命名（名称已符合规则）: %s", root)
            continue
        log.info("目录 %s: 待重命名 %d 项", root, len(plan))
        for src, dst in plan:
            if dry_run:
                print(f"  [dry-run] {src} -> {dst}")
                ok += 1
            else:
                try:
                    src.rename(dst)
                    log.info("OK %s -> %s", src, dst)
                    ok += 1
                except OSError as e:
                    log.error("失败 %s: %s", src, e)
                    skip += 1
    return ok, skip


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="将目录树内文件/目录名中的空格、逗号、句号等特殊字符统一为下划线（与 Step4 归档规则一致）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "roots",
        nargs="*",
        type=str,
        help="根目录；也可用 --root",
    )
    p.add_argument(
        "--root",
        "-r",
        action="append",
        default=[],
        dest="roots_opt",
        help="根目录（可多次）",
    )
    p.add_argument("--dry-run", "-n", action="store_true", help="只打印不执行")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG 日志")
    return p


def run_cli(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    roots = [Path(x) for x in (args.roots or []) + (args.roots_opt or [])]
    if not roots:
        log.error("请指定目录: rename_spaces /path 或 --root /path")
        return 2
    for r in roots:
        if not r.exists():
            log.error("路径不存在: %s", r)
            return 2
    ok, skip = run_rename(roots, dry_run=args.dry_run)
    log.info("完成: 处理 %d 项, 跳过/失败 %d", ok, skip)
    return 0 if skip == 0 or args.dry_run else (0 if ok else 1)


if __name__ == "__main__":
    sys.exit(run_cli())
