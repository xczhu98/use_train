#!/usr/bin/env python3
"""目录名规整工具。

对指定路径下的所有文件夹名称进行递归规范化，默认采用严格 ASCII 规则：
仅保留 ``A-Z a-z 0-9 _ -``，其余字符（空格、中文、特殊符号等）统一替换。

用法::

    # 默认 dry-run：只打印/输出映射，不真正重命名
    python -m tools.ops.data.normalize_dir_names /data/my_dataset

    # 真正执行
    python -m tools.ops.data.normalize_dir_names /data/my_dataset --force

    # 保存映射 JSON
    python -m tools.ops.data.normalize_dir_names /data/my_dataset --force \
        --output-mapping rename_map.json

    # 不跳过隐藏目录
    python -m tools.ops.data.normalize_dir_names /data/my_dataset --force \
        --skip-hidden False
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("normalize_dir_names")

# 默认允许保留的字符集合：字母、数字、下划线、连字符
_DEFAULT_ALLOWED_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")
_FALLBACK_NAME = "dir"


def _validate_replacement(replacement: str) -> str:
    """校验替换符本身必须是合法的单字符。"""
    if len(replacement) != 1:
        raise argparse.ArgumentTypeError("--replacement 必须是单个字符")
    if not re.fullmatch(r"[A-Za-z0-9_-]", replacement):
        raise argparse.ArgumentTypeError(
            "--replacement 只能是字母、数字、下划线或连字符"
        )
    return replacement


def _normalize_name(
    name: str,
    *,
    replacement: str = "_",
    lowercase: bool = True,
) -> str:
    """将目录名规整为严格 ASCII 名称。

    步骤：
      1. 非法字符序列替换为 ``replacement``；
      2. 合并连续替换符；
      3. 去除首尾替换符；
      4. 按需转小写；
      5. 空结果回退为 ``_FALLBACK_NAME``。
    """
    # 替换非法字符
    new_name = _DEFAULT_ALLOWED_PATTERN.sub(replacement, name)
    # 合并连续替换符
    new_name = re.sub(re.escape(replacement) + "+", replacement, new_name)
    # 去除首尾替换符
    new_name = new_name.strip(replacement)
    if lowercase:
        new_name = new_name.lower()
    if not new_name:
        new_name = _FALLBACK_NAME
    return new_name


def _resolve_unique_name(parent: Path, base_name: str, replacement: str = "_") -> str:
    """在 parent 目录下生成不冲突的目标名称；必要时追加 ``_1``、``_2``。"""
    if not (parent / base_name).exists():
        return base_name
    n = 1
    while True:
        candidate = f"{base_name}{replacement}{n}"
        if not (parent / candidate).exists():
            return candidate
        n += 1


def _collect_dirs(
    root: Path,
    *,
    skip_hidden: bool = True,
) -> list[tuple[Path, str]]:
    """自底向上收集所有待处理目录。

    Returns:
        list of ``(parent_path, dir_name)``，已按自底向上排序，
        保证子目录在父目录之前处理。
    """
    entries: list[tuple[Path, str]] = []
    for dirpath, dirnames, _ in os.walk(root, topdown=False):
        current = Path(dirpath)
        if current == root:
            # 不处理输入根目录本身
            pass
        for name in list(dirnames):
            if skip_hidden and name.startswith("."):
                logger.debug("跳过隐藏目录: %s", current / name)
                continue
            entries.append((current, name))
    return entries


def _build_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="递归规整目录名称",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s /data/my_dataset                  # 仅预览
  %(prog)s /data/my_dataset --force          # 真正执行
  %(prog)s /data/my_dataset --force --output-mapping map.json
""",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="待处理的根目录",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="关闭 dry-run，真正执行重命名（默认仅预览）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="仅输出预览映射（默认开启）",
    )
    parser.add_argument(
        "--replacement",
        type=_validate_replacement,
        default="_",
        help="非法字符替换符（默认：_）",
    )
    parser.add_argument(
        "--lowercase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否将大写字母转小写（默认：开启）",
    )
    parser.add_argument(
        "--skip-hidden",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否跳过以 '.' 开头的隐藏目录（默认：开启）",
    )
    parser.add_argument(
        "--output-mapping",
        type=Path,
        default=None,
        dest="output_mapping",
        help="将重命名映射写入 JSON 文件（old_abs_path -> new_abs_path）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每个目录的处理详情",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> dict[str, str]:
    args = _build_args(argv)

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        logger.error("输入路径不是目录: %s", input_dir)
        raise SystemExit(1)

    dry_run = args.dry_run and not args.force
    if dry_run:
        logger.info("当前为 dry-run 模式，不会真正修改文件系统。使用 --force 执行重命名。")

    entries = _collect_dirs(input_dir, skip_hidden=args.skip_hidden)
    mapping: dict[str, str] = {}
    skipped = 0
    renamed = 0
    conflicts = 0

    for parent, old_name in entries:
        old_path = parent / old_name
        new_base = _normalize_name(
            old_name,
            replacement=args.replacement,
            lowercase=args.lowercase,
        )
        if new_base == old_name:
            skipped += 1
            if args.verbose:
                logger.info("无需修改: %s", old_path)
            continue

        unique_name = _resolve_unique_name(parent, new_base, replacement=args.replacement)
        new_path = parent / unique_name
        if unique_name != new_base:
            conflicts += 1
            logger.info("冲突解决: %s -> %s", old_path, new_path)
        else:
            logger.info("准备重命名: %s -> %s", old_path, new_path)

        mapping[str(old_path.absolute())] = str(new_path.absolute())
        renamed += 1

        if not dry_run:
            old_path.rename(new_path)

    mode = "预览" if dry_run else "已执行"
    logger.info(
        "%s完成：共 %d 个目录，跳过 %d 个，%s %d 个（冲突解决 %d 个）",
        mode,
        len(entries),
        skipped,
        mode,
        renamed,
        conflicts,
    )

    if args.output_mapping:
        args.output_mapping.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_mapping, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        logger.info("映射已保存: %s", args.output_mapping)

    return mapping


if __name__ == "__main__":
    main()
