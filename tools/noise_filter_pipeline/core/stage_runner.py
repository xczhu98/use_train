"""
Stage 调度器。

将 stage 名映射到对应 orchestrator，负责串行执行并管理输入/输出 manifest 路径。
"""
from __future__ import annotations

import json
import os
import logging
import tempfile
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from tqdm import tqdm

from .pipeline_config import UnifiedPipelineConfig
from .manifest_io import (
    _iter_manifest_files,
    atomic_replace_durable,
    count_manifest,
    link_or_copy_file,
)
from .manifest_per_dataset import reject_legacy_single_manifest

logger = logging.getLogger(__name__)


# Lazy import stage orchestrators to avoid circular imports.
STAGE_MODULES = {
    "A": "noise_filter_pipeline.stages.stage_a_ingestion",
    "B": "noise_filter_pipeline.stages.stage_b_curation",
    "C": "noise_filter_pipeline.stages.stage_c_single_event",
    "D": "noise_filter_pipeline.stages.stage_d_enhancement",
}


class StageRunner:
    """统一 stage 调度器。"""

    def __init__(self, config: UnifiedPipelineConfig, dataset_filter: Optional[List[str]] = None):
        self.config = config
        # 数据集过滤同时约束 Stage A 及 B/C/D 的 per-dataset 输入文件。
        self.dataset_filter = set(dataset_filter) if dataset_filter else None

    def _select_dataset_manifest_dir(
        self, path: str, previous_stage: str, selected_dir: str
    ) -> str:
        """在临时目录中创建仅含已选数据集 manifest 的 regular 文件。"""
        # --dataset 过滤要求上游输入为 per-dataset manifest 目录（stage 根目录）；
        # 单文件输入无法按数据集筛选，直接给出明确提示，而不是误导性地报
        # "缺少已选数据集 manifest、上游实际包含: 无"
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"--dataset 过滤要求 Stage {previous_stage} 的输入为 per-dataset manifest "
                f"目录（stage 根目录），当前输入不是目录: {path}；"
                "请确认上游已成功完成并传入其 stage 根目录"
            )
        files = _iter_manifest_files(path)
        selected = {
            dataset_id: fp
            for fp in files
            for dataset_id in self.dataset_filter or ()
            if fp.name == f"{previous_stage.lower()}_manifest_{dataset_id}.jsonl"
        }
        missing = sorted((self.dataset_filter or set()) - set(selected))
        if missing:
            available = sorted(
                fp.name.removeprefix(f"{previous_stage.lower()}_manifest_").removesuffix(".jsonl")
                for fp in files
                if fp.name.startswith(f"{previous_stage.lower()}_manifest_")
            )
            raise FileNotFoundError(
                f"Stage {previous_stage} 缺少已选数据集的输入 manifest: {', '.join(missing)}；"
                f"上游实际包含: {', '.join(available) or '无'}"
            )
        for fp in selected.values():
            link_or_copy_file(str(fp), os.path.join(selected_dir, fp.name))
        logger.info(
            "--dataset 输入范围: %s（%d 个 %s per-dataset manifest）",
            ", ".join(sorted(selected)), len(selected), previous_stage,
        )
        return selected_dir

    def _resolve_input_manifest(
        self,
        stage_name: str,
        output_root: str,
        input_manifest: Optional[str],
    ) -> Optional[str]:
        """自动推导输入 manifest；Stage A 返回 None。

        per-dataset 架构下，上游输出为 stage 根目录下的
        ``<stage>_manifest_<dataset_id>.jsonl`` 集合，故返回目录路径，
        下游 processor 以目录方式合并读取。
        """
        if input_manifest is not None:
            if os.path.isdir(input_manifest):
                self._reject_upstream_legacy(stage_name, input_manifest)
            return input_manifest
        if stage_name == "A":
            return None
        prev_stage = chr(ord(stage_name) - 1)
        path = os.path.join(output_root, f"stage_{prev_stage.lower()}")
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Stage {stage_name} 缺少输入 manifest 目录: {path}")
        # 上游完整性校验：stage 成功完成后才写 .complete marker（见 _write_stage_complete）。
        # per-dataset 输出直接写正式文件，失败会留下部分文件，目录存在不代表产出完整；
        # 缺 marker 时拒绝启动，避免下游静默消费不完整结果（如 B 失败后只重跑 C）。
        if not os.path.isfile(os.path.join(path, ".complete")):
            raise RuntimeError(
                f"Stage {stage_name} 的上游 {prev_stage} 未成功完成"
                f"（缺少 {path}/.complete），请先重跑 {prev_stage}"
            )
        self._reject_upstream_legacy(stage_name, path)
        return path

    @staticmethod
    def _reject_upstream_legacy(stage_name: str, path: str) -> None:
        """输入目录含遗留单文件 manifest 时显式拒绝，不再自动拆分/删除。

        per-dataset 架构只读取 ``<stage>_manifest_<dataset_id>.jsonl``；旧单文件
        不会被消费，自动拆分又会隐式改写用户数据，因此交给人工按报错指引处理。
        """
        reject_legacy_single_manifest(path, stage_label=f"Stage {stage_name} 输入目录 {path}")

    @staticmethod
    def _write_stage_complete(stage_dir: str, stage_name: str) -> None:
        """stage 成功完成后写入 .complete marker（原子替换）。

        语义：marker 存在 = 本 stage 最近一次运行成功结束、输出目录中的
        per-dataset manifest 集合完整。下游据此确认输入完整性。
        """
        marker = {
            "stage": stage_name,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp_path = os.path.join(stage_dir, ".complete.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(marker, f, ensure_ascii=False)
            # .complete 是下游启动的前置凭据：掉电丢失会让已完成 stage 重算
            f.flush()
            os.fsync(f.fileno())
        atomic_replace_durable(tmp_path, os.path.join(stage_dir, ".complete"))

    def run_stage(
        self,
        stage_name: str,
        output_root: str,
        input_manifest: Optional[str] = None,
        global_pbar: Optional[tqdm] = None,
    ) -> str:
        """
        执行单个 stage，返回输出 manifest 路径。

        Args:
            stage_name: A/B/C/D 之一。
            output_root: 整个 pipeline 的输出根目录。
            input_manifest: 上一个 stage 的输出 manifest；Stage A 可省略。
            global_pbar: 可选的全局 tqdm 进度条；orchestrator 负责更新 total/step。
        """
        if not output_root:
            raise ValueError("output_root 不能为空，请在配置文件或 --output 中指定输出根目录")
        stage_name = stage_name.upper()
        if stage_name not in self.config.stages:
            raise ValueError(f"未启用的 stage: {stage_name}")

        module_path = STAGE_MODULES.get(stage_name)
        if module_path is None:
            raise ValueError(f"未知 stage: {stage_name}")

        # 动态导入对应 orchestrator
        import importlib
        mod = importlib.import_module(module_path)
        orchestrator_class = getattr(mod, f"Stage{stage_name}Orchestrator")

        stage_dir = os.path.join(output_root, f"stage_{stage_name.lower()}")
        os.makedirs(stage_dir, exist_ok=True)

        input_manifest = self._resolve_input_manifest(stage_name, output_root, input_manifest)
        selected_input_dir = None
        if stage_name != "A" and self.dataset_filter is not None:
            previous_stage = chr(ord(stage_name) - 1)
            selected_input_dir = tempfile.TemporaryDirectory(
                prefix=f"stage_{stage_name.lower()}_selected_"
            )
            try:
                input_manifest = self._select_dataset_manifest_dir(
                    input_manifest, previous_stage, selected_input_dir.name
                )
            except Exception:
                # 选择失败（如指定数据集缺上游 manifest）时清理临时目录，避免泄露
                selected_input_dir.cleanup()
                raise

        # 重跑必须先撤销旧完成态：本次失败时下游不得继续信任上次产物。
        # 只删除 marker，不动旧 JSONL，便于失败后排查或由原子写入保护读取者。
        # 必须在输入解析与 dataset 选择成功之后、orchestrator.run 之前执行：
        # 纯 CLI 输入错误（缺上游 manifest / --dataset 拼错）不得撤销完好产物的完成态
        complete_path = os.path.join(stage_dir, ".complete")
        if os.path.exists(complete_path):
            os.remove(complete_path)

        output_manifest = os.path.join(stage_dir, f"{stage_name.lower()}_manifest.jsonl")

        logger.info("=" * 60)
        logger.info("运行 Stage %s", stage_name)
        logger.info("输入: %s", input_manifest or "dataset registry")
        # 输入范围提示：B/C/D 处理的是上游目录现有的 per-dataset 文件；
        # 若上次是增量 A（只含部分数据集），此处只处理这些数据集
        if stage_name != "A" and input_manifest and os.path.isdir(input_manifest):
            n_files = len(_iter_manifest_files(input_manifest))
            logger.info(
                "输入目录 %s：%d 个 per-dataset 文件（如需处理全部数据集请先全量运行 Stage A）",
                input_manifest, n_files,
            )
        logger.info("输出: %s", output_manifest)
        logger.info("=" * 60)

        # Stage A 自行统计文件数加入全局 total；Stage B 按输入记录数计入；
        # Stage C/D 由 _run_substage 按子阶段 eligible 数动态计入，
        # 避免多子阶段重复消费同一 total 导致进度条超过 100%
        if global_pbar is not None and input_manifest is not None and stage_name == "B":
            input_count = count_manifest(input_manifest)
            global_pbar.total = (global_pbar.total or 0) + input_count
            global_pbar.refresh()

        if stage_name == "A":
            orchestrator = orchestrator_class(self.config, dataset_ids=self.dataset_filter)
        else:
            orchestrator = orchestrator_class(self.config)
        try:
            final_manifest = orchestrator.run(
                input_manifest=input_manifest,
                output_dir=stage_dir,
                output_manifest=output_manifest,
                global_pbar=global_pbar,
            )
        finally:
            if selected_input_dir is not None:
                selected_input_dir.cleanup()

        # 各 stage 在自身收尾时已统计并记录条数（Stage A 用 write_manifest 返回值，
        # B/C/D 单遍扫描），此处不再为一条日志重复解析整个 manifest
        logger.info("Stage %s 完成，输出: %s", stage_name, final_manifest)
        # 成功完成标记：下游据此确认本 stage 输出目录中的 per-dataset manifest 完整
        self._write_stage_complete(stage_dir, stage_name)
        return final_manifest

    @staticmethod
    def _validate_pipeline_stages(stages: List[str]) -> List[str]:
        """校验管线阶段计划，防止非 CLI 调用绕过相邻上游校验。"""
        normalized = [str(stage).upper() for stage in stages]
        if not normalized:
            raise ValueError("stage 计划不能为空")
        if any(stage not in STAGE_MODULES for stage in normalized):
            raise ValueError("stage 计划仅支持 A、B、C、D")
        if len(normalized) != len(set(normalized)) or any(
            ord(current) != ord(previous) + 1
            for previous, current in zip(normalized, normalized[1:])
        ):
            raise ValueError("stage 计划必须是严格连续且不重复的阶段列表（如 A,B,C 或 C,D）")
        return normalized

    def run_pipeline(
        self,
        output_root: str,
        stages: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """顺序执行多个 stage，返回 stage -> manifest 映射。"""
        if not output_root:
            raise ValueError("output_root 不能为空，请在配置文件或 --output 中指定输出根目录")
        stages = self._validate_pipeline_stages(stages or self.config.stages)
        manifests: Dict[str, str] = {}
        last_manifest: Optional[str] = None

        stage_labels = ",".join(stages)
        with tqdm(
            total=0,
            desc=f"Pipeline {stage_labels}",
            unit="file",
            leave=True,
            dynamic_ncols=True,
        ) as global_pbar:
            for stage_name in stages:
                stage_name = stage_name.upper()
                last_manifest = self.run_stage(
                    stage_name,
                    output_root=output_root,
                    input_manifest=last_manifest,
                    global_pbar=global_pbar,
                )
                manifests[stage_name] = last_manifest

        return manifests
