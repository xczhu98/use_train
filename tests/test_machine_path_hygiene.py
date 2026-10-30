"""Tracked 报告与相关源码中的机器路径卫生门禁。"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TRACKED_SCAN_ROOTS = (
    "docs/reports",
    "tasks/se/models/pase_family",
    "tests/test_pase_wavlm_parity.py",
    "tools/prepare_data_tools/generators",
)
# 用户明确允许 Noise Filter 保留默认机器路径；即使未来扩展扫描根也不得纳入。
_EXCLUDED_PREFIXES = ("tools/noise_filter_pipeline/",)
_TEXT_SUFFIXES = {".html", ".json", ".md", ".py", ".toml", ".yaml", ".yml"}
_MACHINE_PATH_PATTERNS = (
    re.compile(
        r"(?<![\w<~])/(?:data_[^/\\\s\"'`]+|root|home/[^/\\\s\"'`]+|"
        r"Users/[^/\\\s\"'`]+|mnt/[^/\\\s\"'`]+|workspace)"
        r"(?:[/\\][^\s\"'`]*)?"
    ),
    re.compile(r"(?<!\w)[A-Za-z]:\\[^\s\"'`]+"),
)

pytestmark = pytest.mark.unit


def _tracked_text_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", *_TRACKED_SCAN_ROOTS],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    paths = []
    for relative in result.stdout.decode().split("\0"):
        if not relative or relative.startswith(_EXCLUDED_PREFIXES):
            continue
        path = _PROJECT_ROOT / relative
        if path.suffix in _TEXT_SUFFIXES and path.is_file():
            paths.append(path)
    return paths


def test_tracked_reports_and_sources_do_not_embed_machine_paths():
    violations = []
    for path in _tracked_text_files():
        relative = path.relative_to(_PROJECT_ROOT)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for pattern in _MACHINE_PATH_PATTERNS:
                if match := pattern.search(line):
                    violations.append(f"{relative}:{line_number}: {match.group(0)}")

    assert not violations, "检测到真实机器路径：\n" + "\n".join(violations)
