#!/usr/bin/env python3
"""
验证 config/noise.yaml（dataset registry）中所有启用数据集的路径是否可访问。

用法：
  python tools/noise_filter_pipeline/scripts/validate_dataset_registry.py

显式覆盖 registry（例如部署副本 registry_local.yaml）：
  python tools/noise_filter_pipeline/scripts/validate_dataset_registry.py \
      --registry tools/noise_filter_pipeline/config/noise.yaml

或从 pipeline.yaml 自动推导注册表路径：
  python tools/noise_filter_pipeline/scripts/validate_dataset_registry.py \
      --config tools/noise_filter_pipeline/config/pipeline.yaml

路径约定：
  - `--config` 中相对的 registry_path 按仓库根目录解析。
  - registry 中条目的相对 path 按 registry 文件所在目录解析。
  - 显式 `--registry` 直接使用传入的相对或绝对路径。

输出：
  - 每个 enabled 数据集的路径状态
  - 能匹配到的样本数量（directory 类型）
  - 最终是否全部通过
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# 将 tools 加入 PYTHONPATH，以便 import noise_filter_pipeline
_TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROJECT_ROOT = os.path.dirname(_TOOLS_DIR)
_DEFAULT_PIPELINE_PATH = os.path.join(
    _TOOLS_DIR, "noise_filter_pipeline", "config", "pipeline.yaml"
)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from noise_filter_pipeline.core.pipeline_config import UnifiedPipelineConfig
# AUDIO_EXTS 从 core.registry 引用，避免 import stage_a_ingestion 引入
# soundfile/tqdm 等重量级依赖（validate 脚本只做纯路径/匹配检查）
from noise_filter_pipeline.core.registry import AUDIO_EXTS, DatasetRegistry


def _glob_files(directory: str, pattern: str) -> List[str]:
    """返回 directory 下匹配 pattern 的音频文件列表。

    按 AUDIO_EXTS 过滤扩展名，与 Stage A 的目录扫描口径一致（否则
    匹配数会混入 txt/md 等非音频文件，与 Stage A 实际 ingest 数不符）。
    """
    search_path = os.path.join(directory, pattern)
    return [
        p for p in glob.glob(search_path, recursive=True)
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in AUDIO_EXTS
    ]


def _check_directory_entry(entry: Any, registry: DatasetRegistry) -> Tuple[bool, str, Optional[int]]:
    """验证 directory 类型数据集。

    返回 (是否通过, 状态描述, 匹配文件数)
    """
    resolved = registry.resolve_path(entry.path)
    if not os.path.isdir(resolved):
        return False, f"目录不存在: {resolved}", None

    matched = _glob_files(resolved, entry.input_pattern)
    count = len(matched)
    if count == 0:
        return False, f"目录存在但无匹配文件 ({entry.input_pattern}): {resolved}", 0
    return True, f"OK，匹配 {count} 个文件", count


def _check_file_entry(entry: Any, registry: DatasetRegistry) -> Tuple[bool, str, Optional[int]]:
    """验证 json/csv 类型数据集。"""
    resolved = registry.resolve_path(entry.path)
    if not os.path.isfile(resolved):
        return False, f"文件不存在: {resolved}", None
    return True, "OK", None


def _check_huggingface_entry(entry: Any, registry: DatasetRegistry) -> Tuple[bool, str, Optional[int]]:
    """验证 huggingface 类型数据集。

    若 path 指向本地目录则检查存在性；否则仅检查字符串非空。
    """
    resolved = registry.resolve_path(entry.path)
    if os.path.isdir(resolved):
        return True, f"本地目录存在: {resolved}", None
    # 非本地路径，仅做格式提醒
    if "/" in entry.path:
        return True, f"HuggingFace 仓库名（未验证可下载）: {entry.path}", None
    return False, f"路径无效: {entry.path}", None


def _check_label_mapping(entry: Any, registry_dir: str) -> Tuple[bool, str]:
    """验证 label_mapping 文件是否存在（仅 label_source=mapping 的条目强制）。

    其它 label_source（filename/directory/none）即使残留 label_mapping 配置
    也不会被 Stage A 消费，不应导致验证失败。
    """
    if entry.label_source != "mapping":
        return True, "未使用 mapping 标签源"
    mapping_path = entry.label_mapping_path(registry_dir)
    if mapping_path is None:
        return True, "未配置"
    if os.path.isfile(mapping_path):
        return True, f"OK: {mapping_path}"
    return False, f"映射文件不存在: {mapping_path}"


def validate_registry(registry_path: str, base_dir: str) -> Dict[str, Any]:
    """执行验证并返回结构化结果。

    条目相对路径统一按 registry 文件所在目录解析（registry.resolve_path，
    与 Stage A 同口径）；base_dir 参数仅为兼容保留在报告里，不再参与
    条目路径解析。
    """
    registry = DatasetRegistry(registry_path)
    results: List[Dict[str, Any]] = []
    all_passed = True

    for entry in registry.list_all():
        if not entry.enabled:
            results.append({
                "id": entry.id,
                "enabled": False,
                "status": "skipped",
                "message": "数据集未启用",
            })
            continue

        if entry.type == "directory":
            passed, message, count = _check_directory_entry(entry, registry)
        elif entry.type in ("json", "csv"):
            passed, message, count = _check_file_entry(entry, registry)
        elif entry.type == "huggingface":
            passed, message, count = _check_huggingface_entry(entry, registry)
        else:
            passed, message, count = False, f"未知类型: {entry.type}", None

        mapping_passed, mapping_message = _check_label_mapping(entry, registry_dir=registry._dir)

        record = {
            "id": entry.id,
            "name": entry.name,
            "enabled": True,
            "type": entry.type,
            "path": registry.resolve_path(entry.path),
            "passed": passed and mapping_passed,
            "path_status": {"passed": passed, "message": message},
            "mapping_status": {"passed": mapping_passed, "message": mapping_message},
        }
        if count is not None:
            record["matched_files"] = count
        results.append(record)

        if not record["passed"]:
            all_passed = False

    return {
        "registry_path": registry_path,
        "base_dir": base_dir,
        "all_passed": all_passed,
        "datasets": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="验证 config/noise.yaml（dataset registry）数据集路径"
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="dataset registry YAML 路径；未指定时从正式 pipeline.yaml 推导",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="pipeline.yaml 路径，自动推导 registry"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="registry_path 相对该目录解析（默认仓库根）；条目路径仍按 registry 目录解析",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    args = parser.parse_args()

    if args.base_dir:
        base_dir = args.base_dir
    else:
        # pipeline.yaml 的 registry_path 相对仓库根解析，避免 tools/ 被重复拼接。
        base_dir = _PROJECT_ROOT

    registry_path: Optional[str] = args.registry
    if registry_path is None:
        if args.config:
            cfg = UnifiedPipelineConfig.from_yaml(args.config, active_stages=())
            registry_path = cfg.resolve_registry_path(base_dir)
        else:
            # 无显式 override 时读取 pipeline.yaml，再推导 noise.yaml。
            cfg = UnifiedPipelineConfig.from_yaml(
                _DEFAULT_PIPELINE_PATH, active_stages=()
            )
            registry_path = cfg.resolve_registry_path(base_dir)

    if not registry_path or not os.path.isfile(registry_path):
        print(f"错误: 注册表文件不存在: {registry_path}", file=sys.stderr)
        return 2

    report = validate_registry(registry_path, base_dir)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("=" * 70)
        print(f"数据集注册表验证: {registry_path}")
        print("=" * 70)
        for ds in report["datasets"]:
            status = "PASS" if ds.get("passed") else ("SKIP" if ds.get("status") == "skipped" else "FAIL")
            print(f"[{status}] {ds['id']} ({ds.get('name', '')})")
            if "path_status" in ds:
                print(f"       path: {ds['path_status']['message']}")
            if "mapping_status" in ds:
                print(f"       mapping: {ds['mapping_status']['message']}")
            if "matched_files" in ds:
                print(f"       matched_files: {ds['matched_files']}")
        print("=" * 70)
        print(f"结果: {'全部通过' if report['all_passed'] else '存在失败项'}")
        print("=" * 70)

    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
