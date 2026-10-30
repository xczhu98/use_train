import os
import sys
import argparse
import logging
import signal
from pathlib import Path


def _install_sigterm_handler() -> None:
    """把 SIGTERM 转为 SystemExit，让 finally/atexit 清理逻辑得以执行。

    默认 SIGTERM 直接杀死进程，Stage C/D 的 Qwen worker pool、VAD 进程池、
    Apollo/AudioTag daemon 的 stop() 都不会执行，子进程变孤儿。
    转成 SystemExit 后异常沿调用栈传播，各 finally 中的 pool.stop() 正常回收。
    """

    def _handler(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handler)

# 将项目根目录与 tools 目录加入 path，以便同时 import noise_filter_pipeline 和 tools.shared.*。
# 当前文件位于 noise_filter_pipeline/cli/；模块入口直接从此处启动。
_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = str(_PIPELINE_ROOT.parent)
_PROJECT_ROOT = str(_PIPELINE_ROOT.parents[1])
for _path in (_PROJECT_ROOT, _TOOLS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# 在 import 任何可能加载 TensorFlow 的模块之前：限制 TF 线程，减轻与 PyTorch 同进程时的 segfault
if os.environ.get("TF_NUM_INTEROP_THREADS") is None:
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
if os.environ.get("TF_NUM_INTRAOP_THREADS") is None:
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

from noise_filter_pipeline.core.pipeline_config import UnifiedPipelineConfig
from noise_filter_pipeline.core.stage_runner import StageRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统一音频源库构建平台 - Noise Filter Pipeline Stage A-D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例（从仓库根目录执行）:
  PYTHONPATH=tools python -m noise_filter_pipeline.cli --stage A,B,C,D --config tools/noise_filter_pipeline/config/pipeline.yaml
  PYTHONPATH=tools python -m noise_filter_pipeline.cli --stage B --config tools/noise_filter_pipeline/config/pipeline.yaml
  PYTHONPATH=tools python -m noise_filter_pipeline.cli rename_spaces --root ./runs/noise_root
        """,
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="配置文件路径 (YAML 格式)；默认使用 config/pipeline.yaml",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="输出根目录；覆盖配置文件中的 output_root",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="要执行的阶段：A|B|C|D 或逗号分隔列表（如 A,B）；"
             "配合 --dataset 时各 stage 均只处理指定数据集",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="只处理指定的数据集 id（逗号分隔，如 dcase2019_task1a,musan）；"
             "对 A/B/C/D 均生效：A 仅接入指定 registry 数据集；"
             "独立运行 B/C/D 时仅消费上游目录中匹配的 per-dataset manifest，"
             "缺 manifest 立即报错并列出上游实际范围。"
             "注意：Stage A 的 manifest 是覆盖写，过滤跑会重建为只含这些数据集；"
             "后续需要全量 manifest 时重跑一次不带 --dataset 的 Stage A 即可。",
    )
    return parser.parse_args()


def _parse_stage_list(value: str) -> list[str]:
    """解析并校验命令行 stage 计划：只接受连续且不重复的 A-D 序列。"""
    stages = [stage.strip().upper() for stage in value.split(",") if stage.strip()]
    if not stages:
        raise ValueError("--stage 不能为空")
    if any(stage not in {"A", "B", "C", "D"} for stage in stages):
        raise ValueError("--stage 仅支持 A、B、C、D")
    if len(stages) != len(set(stages)) or any(
        ord(current) != ord(previous) + 1
        for previous, current in zip(stages, stages[1:])
    ):
        raise ValueError("--stage 必须是严格连续且不重复的阶段列表（如 A,B,C 或 C,D）")
    return stages


def main() -> int:
    _install_sigterm_handler()
    # 管线的关键进度/告警大量走 logging（INFO/WARNING）：不配置时 root 默认
    # 只显示 WARNING+，INFO（启动、断点恢复、子阶段汇总）全部不可见
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    if args.stage:
        try:
            stages = _parse_stage_list(args.stage)
        except ValueError as e:
            print(f"错误: {e}")
            return 1

        if args.config:
            config = UnifiedPipelineConfig.from_yaml(
                args.config, active_stages=stages
            )
        else:
            default_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "config",
                "pipeline.yaml",
            )
            config = UnifiedPipelineConfig.from_yaml(
                default_path, active_stages=stages
            )

        if args.output:
            config.output_root = args.output

        dataset_filter = None
        if args.dataset:
            dataset_filter = [s.strip() for s in args.dataset.split(",") if s.strip()]
            if not dataset_filter:
                print("错误: --dataset 不能为空")
                return 1
            if "A" not in stages:
                # --dataset 对 B/C/D 同样生效：精确过滤上游目录中匹配的
                # per-dataset manifest（缺失即报错，见 stage_runner），
                # 此处仅提示本次输入范围，不再是"仅对 A 生效"的告警
                print(
                    "提示: --stage 不含 A，B/C/D 将仅消费上游目录中匹配 "
                    f"{dataset_filter} 的 per-dataset manifest（缺失即报错）。"
                )
            scope_note = "（Stage A manifest 将覆盖重建）" if "A" in stages else ""
            print(f"数据集过滤: 仅处理 {dataset_filter}{scope_note}")

        runner = StageRunner(config, dataset_filter=dataset_filter)
        try:
            manifests = runner.run_pipeline(
                output_root=config.output_root,
                stages=stages,
            )
            print("\n" + "=" * 60)
            print("统一管道执行完成")
            for stage_name, manifest_path in manifests.items():
                print(f"  Stage {stage_name}: {manifest_path}")
            print("=" * 60)
            return 0
        except Exception as e:
            print(f"错误: Stage 执行失败: {e}")
            import traceback

            traceback.print_exc()
            return 1

    print("错误: 请指定 --stage A|B|C|D 或使用 rename_spaces 子命令")
    return 1


if __name__ == "__main__":
    # 子命令：目录树内文件名/文件夹名中文转拼音，并 sanitize 不安全字符
    if len(sys.argv) > 1 and sys.argv[1] == "rename_spaces":
        import importlib.util

        # 子命令绕过 main()，同样需要默认日志配置，否则 INFO 不可见
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        _rs = _PIPELINE_ROOT / "scripts" / "rename_paths_replace_spaces.py"
        _spec = importlib.util.spec_from_file_location("_rename_paths_replace_spaces", _rs)
        _mod = importlib.util.module_from_spec(_spec)
        assert _spec.loader is not None
        _spec.loader.exec_module(_mod)
        sys.exit(_mod.run_cli(sys.argv[2:]))

    exit(main())
