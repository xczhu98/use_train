#!/usr/bin/env python3
"""只读检查训练数据 manifest 的格式与能力。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

# 支持从仓库根目录直接执行脚本。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.prepare_data_tools.manifest_contracts import (  # noqa: E402
    MANIFEST_PROFILES,
    PROFILE_DYNAMIC_SOURCE,
    ManifestInspection,
    ManifestInspectionError,
    inspect_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "只读检查严格四列 SCP 或 JSONL manifest 的训练能力，不打开记录内音频。"
            "dynamic-speech 是生产 speech 发布门禁，要求每条记录都有完整 length。"
        )
    )
    parser.add_argument("manifest", help="待检查的 manifest 文件")
    parser.add_argument(
        "--profile",
        choices=MANIFEST_PROFILES,
        default=PROFILE_DYNAMIC_SOURCE,
        help="目标使用契约（默认：dynamic-source；dynamic-speech 要求完整 length）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出稳定、可机器读取的 JSON",
    )
    return parser


def _format_text(result: ManifestInspection) -> str:
    rates = ", ".join(str(value) for value in result.sample_rates) or "none"
    capabilities = result.capabilities
    return "\n".join(
        [
            f"manifest: {result.manifest}",
            f"profile: {result.profile}",
            f"serialization_format: {result.serialization_format}",
            f"record_count: {result.record_count}",
            f"sample_rates: {rates}",
            f"has_metadata: {str(capabilities.has_metadata).lower()}",
            f"length_complete: {str(capabilities.length_complete).lower()}",
            f"category_complete: {str(capabilities.category_complete).lower()}",
            "strict_fields_complete: "
            f"{str(capabilities.strict_fields_complete).lower()}",
            f"yaml_hint: {result.yaml_hint}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """运行 CLI 并返回进程退出码。"""
    args = _build_parser().parse_args(argv)
    try:
        result = inspect_manifest(args.manifest, profile=args.profile)
    except ManifestInspectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
    else:
        print(_format_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
