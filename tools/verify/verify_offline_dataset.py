#!/usr/bin/env python3
"""验证 GenerationManifest v1/v2 与生产 OfflineDataset 可完整重载。

默认执行正式 full scan；``--probe N`` 只用于定位问题，输出永远不具发布资格。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.prepare_data_tools.generation_manifest import (  # noqa: E402
    MANIFEST_NAME,
    GenerationManifestError,
    load_generation_manifest,
)
from tools.prepare_data_tools.offline_reload_gate import (  # noqa: E402
    OfflineReloadError,
    run_offline_reload_gates,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation_root", help="GenerationManifest 所属 generation root")
    parser.add_argument(
        "--manifest",
        default=MANIFEST_NAME,
        help="generation root 相对 manifest 路径（默认 generation_manifest.json）",
    )
    parser.add_argument(
        "--probe",
        type=int,
        metavar="N",
        help="仅调试前 N 条；不能作为正式 publish gate",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="ALIAS=PATH",
        help="schema v2 外部 root 显式绑定；可重复",
    )
    parser.add_argument("--json", action="store_true", help="输出稳定 JSON")
    return parser.parse_args(argv)


def _parse_root_bindings(values: list[str]) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--root 必须使用 ALIAS=PATH: {value!r}")
        alias, path = value.split("=", 1)
        if not alias or not path:
            raise ValueError(f"--root alias/path 不得为空: {value!r}")
        if alias in bindings:
            raise ValueError(f"--root alias 重复: {alias!r}")
        bindings[alias] = path
    return bindings


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        root_bindings = _parse_root_bindings(args.root)
        manifest = load_generation_manifest(
            args.generation_root,
            args.manifest,
            root_bindings=root_bindings or None,
        )
        report = run_offline_reload_gates(
            args.generation_root,
            manifest=manifest,
            probe=args.probe,
            root_bindings=root_bindings or None,
        )
    except (GenerationManifestError, OfflineReloadError, OSError, ValueError) as exc:
        error = {
            "status": "failed",
            "generation_root": str(args.generation_root),
            "manifest": str(args.manifest),
            "error": str(exc),
        }
        if args.json:
            print(json.dumps(error, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        else:
            print(f"错误：离线数据集验证失败：{exc}", file=sys.stderr)
        return 1

    payload = {
        "status": "passed",
        "generation_root": str(Path(args.generation_root)),
        "manifest": str(args.manifest),
        **report.to_dict(),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    elif report.publish_eligible:
        scanned = sum(gate.scanned for gate in report.gates)
        print(
            f"验证通过：task={report.task}，已完成 full scan，"
            f"扫描 {scanned} 条，具备正式发布门禁资格。"
        )
    else:
        scanned = sum(gate.scanned for gate in report.gates)
        print(
            f"调试 probe 通过：task={report.task}，扫描 {scanned} 条；"
            "该结果不具正式发布门禁资格。"
        )
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
