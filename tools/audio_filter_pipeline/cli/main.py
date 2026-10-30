"""Audio Filter Pipeline — generation-3（v3）artifact 管线入口。

Step4 示例：``PYTHONPATH=tools python -m audio_filter_pipeline.cli --config
 tools/audio_filter_pipeline/config/audio_filter_zh_16k.yaml --step step4``。
Step6 的 shipped ``spk_filter.yaml`` 仅是 schema-valid fail-closed 模板；必须
先替换全部 speaker assets、版本与 digest，才可将其用于 ``--step all``。
``all`` 请求 Step6 依赖闭包，不执行 Step4。generation-2（v2）实现及其 CLI
已移除，本入口只保留 v3 参数并直接委托
``processors.stages_v3.cli.run_v3_cli``。脚本入口统一使用
``PYTHONPATH=tools python -m audio_filter_pipeline.cli``。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# 将 tools 目录与项目根目录加入 path，以便 import audio_filter_pipeline 与 tools.shared.*。
# 当前文件位于 audio_filter_pipeline/cli/；模块入口直接从此处启动。
_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = str(_PIPELINE_ROOT.parent)
_PROJECT_ROOT = str(_PIPELINE_ROOT.parents[1])
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 在 import 任何可能加载 TensorFlow 的模块之前：限制 TF 线程，减轻与 PyTorch 同进程时的 segfault
if os.environ.get("TF_NUM_INTEROP_THREADS") is None:
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
if os.environ.get("TF_NUM_INTRAOP_THREADS") is None:
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"


# 明确约定为「仓库相对」的第三方资产路径键：YAML 中这些键的相对值按仓库根解析。
_REPO_RELATIVE_CONFIG_PATH_KEYS = (
    'urgent_pk_checkpoint',
    'urgent_pk_source_dir',
    'scoreq_model_dir',
    'dnsmos_model_dir',
    'nisqa_model_path',
    'utmos_model_path',
)


def _resolve_repo_relative_config_paths(data: dict) -> dict:
    """将明确约定为仓库相对的第三方资产路径解析为绝对路径。

    仅处理显式列入白名单的 MOS 模型资产键，避免改变 input_paths、
    output_root 等手写路径原有语义。
    """
    resolved = dict(data)
    for key in _REPO_RELATIVE_CONFIG_PATH_KEYS:
        value = resolved.get(key)
        if not isinstance(value, str) or not value or os.path.isabs(value):
            continue
        resolved[key] = os.path.normpath(os.path.join(_PROJECT_ROOT, value))
    return resolved


def load_config(config_path: str) -> dict:
    """加载 YAML；文件缺失或格式错误时直接抛出。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError("配置文件顶层必须是 mapping")
    return _resolve_repo_relative_config_paths(data)


def parse_args():
    parser = argparse.ArgumentParser(
        # allow_abbrev=False：关闭 argparse 前缀补全，避免已移除的 `--output`
        # 被当作 `--output_root` 的缩写静默接受。
        allow_abbrev=False,
        description='Audio Filter Pipeline：generation-3（v3）artifact 管线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''\
当前 v3 用法（从 use_train 仓库根目录执行）：
  # Step4：shipped audio_filter_*.yaml 默认使用 SCOREQ + DNSMOS 严格 2/2
  PYTHONPATH=tools python -m audio_filter_pipeline.cli \\
    --config tools/audio_filter_pipeline/config/audio_filter_zh_16k.yaml --step step4

  # Step6：先复制/派生 spk_filter.yaml，并替换全部 speaker assets、版本与 digest
  # shipped spk_filter.yaml 是 schema-valid fail-closed 模板，不可直接运行
  PYTHONPATH=tools python -m audio_filter_pipeline.cli \\
    --config /path/to/validated_spk_filter.yaml --step all

仓库不随附默认配置，必须显式传入 --config。NISQA、UTMOS 与 URGENT-PK
资产字段是 opt-in；未被 selection_policies 选中时不会 preflight。当前 shipped
URGENT-PK source tree 含 urgentpk/model_mod.py symlink，安全 tree scanner 会
fail-closed 拒绝，因此 opt-in 仍被阻塞；checkpoint SHA-256 与 commit 匹配不足以放行。
v3 target Step 会按 DAG 验证并运行或恢复所需上游 artifact；Step5/Step6 会先执行
speaker asset preflight，失败时不会启动任何 stage、factory 或 GPU worker。
生产 v3 GPU 路径禁止静默降级 CPU；CPU downgrade 开关仅属于 direct/legacy API。
        ''',
    )
    parser.add_argument('--config', '-c', default=None,
                        help='[v3 current] 配置文件路径；spk_filter.yaml 是 schema-valid fail-closed 模板，必须替换 speaker assets 后使用')
    parser.add_argument('--step', default=None,
                        help='[v3 current] target：all（等价 step6，依赖闭包不含 Step4）或 step0..step6（按 DAG 验证并运行/恢复依赖）')
    parser.add_argument('--drop-steps', default=None,
                        help='[v3 current] 独立、非级联清理命令：只删除显式列出的 step0..step6 artifact 后退出；不会级联删除下游，也不会删除已导出的 FLAC/manifest，需另行运行目标 step 重建')
    parser.add_argument('--output_root', default=None,
                        help='[v3 current] 输出根目录（生产推荐使用仓库外绝对路径）')
    parser.add_argument('--step0_workers', type=int, default=None,
                        help='[v3 current] Step0 并发 worker 数，必须为正整数')
    parser.add_argument('--step3_gpu_ids', type=str, default=None,
                        help='[v3 current] Step3 MOS 使用的物理 GPU ID 列表，例如 0,1；use_gpu 时不能为空')
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument('--use_gpu', action='store_true', default=None,
                           help='[v3 current] 启用 GPU 加速；未指定时从 YAML 的 use_gpu 读取；失败时禁止静默降级 CPU')
    gpu_group.add_argument('--no_use_gpu', action='store_true',
                           help='[v3 current] 显式选择 CPU 执行，不是 GPU 失败后的 downgrade')
    return parser.parse_args()


def main():
    # 管线关键进度/告警大量走 logging（INFO/WARNING）：不配置时 root 默认
    # 只显示 WARNING+，INFO（各 stage 起止、行数、耗时、汇总）全部不可见。
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    if not args.config:
        print("错误: 请用 --config 显式指定配置文件")
        return 1

    args.config = os.path.realpath(os.path.abspath(args.config))
    if not os.path.isfile(args.config):
        print(f"错误: 指定的配置文件不存在: {args.config}")
        return 1
    try:
        config_dict = load_config(args.config)
    except FileNotFoundError:
        print(f"错误: 配置文件不存在或已被删除: {args.config}")
        return 1
    except (OSError, UnicodeDecodeError, yaml.YAMLError, TypeError) as exc:
        print(f"错误: 无法加载配置文件 {args.config}: {exc}")
        return 1

    try:
        from audio_filter_pipeline.processors.optional_runtime import (
            LocalAssetError,
            OptionalRuntimeConfigError,
        )
        from audio_filter_pipeline.processors.stages_v3.cli import run_v3_cli
        return run_v3_cli(
            args,
            config_dict,
            config_base_dir=os.path.dirname(args.config),
        )
    except (
        TypeError,
        ValueError,
        OptionalRuntimeConfigError,
        LocalAssetError,
    ) as exc:
        print(f"错误: generation-3 配置或本地资产无效: {exc}")
        return 1


if __name__ == '__main__':
    exit(main())
