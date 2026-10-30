#!/usr/bin/env python3
"""YAML 配置注释规范检查（只读）。

规范条文见 ``configs/README.md`` 的「YAML 注释规范」章节。本脚本实现其中可机械
检查的三条：

- ``L1`` 分节线（``# ----`` / ``# ====``）的缩进必须为 0，或与其后键同缩进；
- ``L2`` 注释块与它描述的键之间不得有空行，也不得悬空在文件末尾；
- ``L3`` 非分节线注释行的缩进必须等于其后键的缩进。

文件头（首个 YAML key 之前）豁免全部规则；``# yaml-comment-lint: off`` 与
``# yaml-comment-lint: on`` 之间的区间整体豁免（用于手工切换块等受保护结构）。

用法::

    python tools/verify/check_config_comments.py                 # 检查 configs/ 全部 YAML
    python tools/verify/check_config_comments.py <path> [...]    # 检查指定文件或目录
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "configs"

_DIVIDER_RE = re.compile(r"^[-=]{4,}$|^[-=]{4,}\s.*\s[-=]{4,}$")
_KEY_LINE_RE = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_.-]*\s*:")
_LINT_OFF = "yaml-comment-lint: off"
_LINT_ON = "yaml-comment-lint: on"


def _is_comment(line: str) -> bool:
    return line.lstrip().startswith("#")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _comment_body(line: str) -> str:
    return line.lstrip()[1:].strip()


def _is_divider(line: str) -> bool:
    return bool(_DIVIDER_RE.match(_comment_body(line)))


def _first_key_index(lines: Sequence[str]) -> int:
    """返回首个 YAML key 的行号；全部为注释时返回行数。"""
    for index, line in enumerate(lines):
        if line.strip() and not _is_comment(line):
            return index
    return len(lines)


def check_text(text: str, *, path: str = "<memory>") -> list[str]:
    """返回 ``["<line>: <message>", ...]`` 形式的违规列表，行号从 1 开始。"""
    lines = text.splitlines()
    header_end = _first_key_index(lines)
    violations: list[str] = []
    lint_off = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if _LINT_OFF in line:
            lint_off = True
        elif _LINT_ON in line:
            lint_off = False
        if (
            not _is_comment(line)
            or index < header_end
            or lint_off
        ):
            index += 1
            continue
        # 收集连续注释块
        block_start = index
        while index < len(lines) and _is_comment(lines[index]):
            index += 1
        block = list(enumerate(lines[block_start:index], start=block_start))
        # L2：注释块必须紧贴后续键，且不得落在文件末尾
        if index >= len(lines) or not lines[index].strip():
            violations.append(
                f"{block_start + 1}: 注释块与它描述的键之间有空行或已到文件末尾（L2）"
            )
        key_indent = _indent(lines[index]) if index < len(lines) else None
        for line_index, block_line in block:
            if _is_divider(block_line):
                if _indent(block_line) not in (0, key_indent):
                    violations.append(
                        f"{line_index + 1}: 分节线缩进 {_indent(block_line)} 既不是 0 "
                        f"也不等于其后键缩进 {key_indent}（L1）"
                    )
                continue
            if _indent(block_line) != key_indent:
                violations.append(
                    f"{line_index + 1}: 注释缩进 {_indent(block_line)} 不等于其后键"
                    f"缩进 {key_indent}（L3）"
                )
    return violations


def check_file(path: Path) -> list[str]:
    return check_text(path.read_text(encoding="utf-8"), path=str(path))


def iter_config_paths(root: Path) -> Iterator[Path]:
    yield from sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".yaml", ".yml"}
    )


def collect_paths(targets: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for target in targets:
        candidate = Path(target)
        if candidate.is_dir():
            paths.extend(iter_config_paths(candidate))
        elif candidate.is_file():
            paths.append(candidate)
        else:
            raise FileNotFoundError(f"no such file or directory: {target}")
    return sorted(dict.fromkeys(paths))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="*",
        default=None,
        help="待检查的 YAML 文件或目录，默认 configs/ 下全部 YAML",
    )
    args = parser.parse_args(argv)
    paths = collect_paths(args.paths) if args.paths else list(iter_config_paths(DEFAULT_ROOT))
    total = 0
    for path in paths:
        violations = check_file(path)
        try:
            label = path.relative_to(REPO_ROOT)
        except ValueError:
            label = path
        for violation in violations:
            print(f"{label}:{violation}")
        total += len(violations)
    print(f"[SUMMARY] files={len(paths)} violations={total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
