"""
Stage B: Source Curation（噪声筛选）。

将源音频通过 B1→B2→B3→B4 模块化 Processor 链路进行纯化：
  B1 分片 → B2 静音过滤 → B3 VAD → B4 策略过滤

说明：
  - B1 带宽检测/重采样已被移除。后续 D1 Apollo 超分会把音频统一到 44.1kHz 并做带宽扩展，
    这里不再提前拒绝低带宽或重采样，避免重复劳动。
  - B5 AED 分类已删除；Stage C 改用 AudioTag 作为 AudioSet 分类步骤。
  - 每个 Processor 继承 core.processor_base.Processor，可单独开关与替换。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

from ..core.pipeline_config import UnifiedPipelineConfig
from ..core.manifest_io import read_manifest_strict
from ..core.manifest_per_dataset import (
    link_per_dataset_manifests,
    split_manifest_by_dataset,
)
from ..core.stats import write_stage_stats
from ..processors.chunking import ChunkingProcessor
from ..processors.silence_processor import SilenceFilterProcessor
from ..processors.vad_processor import VADProcessor
from ..processors.policy_processor import PolicyProcessor
from ..processors.stage_b_unified import StageBUnifiedProcessor

logger = logging.getLogger(__name__)


class StageBOrchestrator:
    """Stage B orchestrator。"""

    def __init__(self, config: UnifiedPipelineConfig):
        self.config = config
        self.stage_b = config.stage_b or {}

    def _build_sub_configs(self) -> Dict[str, Dict[str, Any]]:
        """构造 B1/B2/B3/B4 子配置，兼容旧扁平字段。"""
        cfg = self.stage_b

        # B1 分片
        b1 = dict(cfg.get("b1", {}))
        if "mode" not in b1:
            b1["mode"] = "hive" if cfg.get("enable_hive_chunking") else "adaptive"
        for old_key, new_key in [
            ("chunk_duration", "chunk_duration"),
            ("min_chunk_duration", "min_chunk_duration"),
        ]:
            if new_key not in b1 and old_key in cfg:
                b1[new_key] = cfg[old_key]
        if "hive_window_sec" not in b1:
            hive_cfg = cfg.get("hive_chunking", {})
            b1.setdefault("hive_window_sec", hive_cfg.get("window_sec", 10.0))
            b1.setdefault("hive_step_sec", hive_cfg.get("step_sec", 5.0))
            b1.setdefault("hive_energy_threshold", hive_cfg.get("energy_threshold", 0.0005))
            b1.setdefault("hive_min_tail_sec", hive_cfg.get("min_tail_sec", 5.0))
            b1.setdefault("hive_output_label_key", hive_cfg.get("output_label_key", "hive_chunk_source"))

        # B2 静音过滤
        b2 = dict(cfg.get("b2", {}))
        for old_key, new_key in [
            ("enable_silence_filter", "enable"),
            ("max_silence_ratio", "max_silence_ratio"),
            ("min_chunk_duration_filter", "min_chunk_duration_filter"),
            ("shorten_long_silences", "shorten_long_silences"),
            ("max_silence_duration", "max_silence_duration"),
            ("target_silence_duration", "target_silence_duration"),
            ("target_silence_min", "target_silence_min"),
            ("target_silence_max", "target_silence_max"),
            ("use_random_silence_duration", "use_random_silence_duration"),
        ]:
            if new_key not in b2 and old_key in cfg:
                b2[new_key] = cfg[old_key]

        # B3 VAD
        b3 = dict(cfg.get("b3", {}))
        for old_key, new_key in [
            ("enable_vad", "enable"),
            ("use_multi_vad", "use_multi_vad"),
            ("vad_types", "vad_types"),
            ("vad_type", "vad_type"),
            ("vad_model_dir", "vad_model_dir"),
            ("funasr_model_dir", "funasr_model_dir"),
            ("firered_model_dir", "firered_model_dir"),
            ("nemo_onnx_path", "nemo_onnx_path"),
            ("use_gpu", "use_gpu"),
            ("speech_threshold", "speech_threshold"),
            ("vad_speech_ratio_threshold", "vad_speech_ratio_threshold"),
        ]:
            if new_key not in b3 and old_key in cfg:
                b3[new_key] = cfg[old_key]

        # B4 策略过滤
        b4 = dict(cfg.get("b4", {}))
        for old_key, new_key in [
            ("keep_noise", "keep_noise"),
            ("keep_suspected", "keep_suspected"),
            ("keep_rejected", "keep_rejected"),
            ("vad_speech_ratio_threshold", "vad_speech_ratio_threshold"),
            ("speech_threshold", "speech_threshold"),
        ]:
            if new_key not in b4 and old_key in cfg:
                b4[new_key] = cfg[old_key]

        # 文件级并发 workers 透传
        num_workers = int(cfg.get("num_workers", 1))
        b1["num_workers"] = num_workers
        b2["num_workers"] = num_workers

        # 只透传到消费该字段的 B1。
        max_peak = float(cfg.get("max_peak", 1.2))
        b1["max_peak"] = max_peak

        return {
            "b1": b1,
            "b2": b2,
            "b3": b3,
            "b4": b4,
            "num_workers": num_workers,
            "temp_dir": cfg.get("temp_dir"),
            "max_peak": max_peak,
            "vad_batch_chunks": cfg.get("vad_batch_chunks", 2000),
            "output_format": cfg.get("output_format", "flac"),
            "output_shard": cfg.get("output_shard", False),
            "save_temp_chunks": cfg.get("save_temp_chunks", False),
            "enable_checkpoint": cfg.get("enable_checkpoint", True),
            "checkpoint_path": cfg.get("checkpoint_path"),
            "checkpoint_save_interval": cfg.get("checkpoint_save_interval", 100),
            "manifest_write_batch_size": cfg.get("manifest_write_batch_size", 1000),
        }

    @staticmethod
    def _empty_input_dataset_ids(input_manifest: str) -> set[str]:
        """返回本轮为空的上游 per-dataset manifest 对应的数据集。"""
        if not os.path.isdir(input_manifest):
            return set()
        return {
            path.name[len("a_manifest_"):-len(".jsonl")]
            for path in Path(input_manifest).glob("a_manifest_*.jsonl")
            if path.stat().st_size == 0
        }

    @staticmethod
    def _prune_empty_dataset_outputs(output_dir: str, empty_dataset_ids: set[str]) -> None:
        """删除本轮空输入数据集遗留的 Stage B 根 manifest。"""
        for dataset_id in empty_dataset_ids:
            path = Path(output_dir, f"b_manifest_{dataset_id}.jsonl")
            if path.is_file():
                path.unlink()
                logger.info("Stage B 清理本轮空输入数据集的遗留 manifest: %s", path.name)

    @staticmethod
    def _preflight_manifest(input_manifest: str) -> None:
        """完整严格扫描 manifest，且拒绝全局重复 record_id。"""
        for _ in read_manifest_strict(input_manifest, reject_duplicate_ids=True):
            pass

    def run(
        self,
        input_manifest: str,
        output_dir: str,
        output_manifest: str,
        global_pbar: Optional[tqdm] = None,
    ) -> str:
        sub_cfgs = self._build_sub_configs()
        empty_input_dataset_ids = self._empty_input_dataset_ids(input_manifest)

        if self.stage_b.get("unified", False):
            os.makedirs(output_dir, exist_ok=True)
            # 单文件 in-memory 合并流水线（输出 per-dataset manifest 到 stage_b/ 根）
            processor = StageBUnifiedProcessor(
                sub_cfgs,
                progress_callback=(global_pbar.update if global_pbar is not None else None),
            )
            processor.process(
                input_manifest,
                output_dir,
            )
            # 与 C/D 的 infra-failed 语义对齐：VAD 整批失败等可重试失败
            # 未写 checkpoint，必须中止 stage（下游不得消费不完整产物），
            # 修复后重跑自动重试；否则 stage 会"成功"完成但产出为空
            if processor.has_retryable_failures():
                raise RuntimeError(
                    "Stage B unified 存在可重试处理失败（如 VAD 整批失败），"
                    "相关记录未写 checkpoint，重跑可自动重试"
                )
            self._prune_empty_dataset_outputs(output_dir, empty_input_dataset_ids)
            stats = write_stage_stats(
                manifest_path=output_dir,
                stage_name="B",
                output_path=os.path.join(output_dir, "b_stats.json"),
            )
            # stats 的单遍扫描已含总数与聚合 passed（B4 passed 的记录其聚合状态
            # 即 passed），不再为一条日志重复解析整个 manifest
            logger.info(
                "Stage B unified 完成：%d 条记录，B4 通过 %d，输出 %s",
                stats["total_records"], stats["status"]["passed"], output_dir,
            )
            return output_dir

        # 创建 B1 产物前必须严格扫描输入，拒绝坏行和全局重复 ID。
        self._preflight_manifest(input_manifest)
        os.makedirs(output_dir, exist_ok=True)

        # B1 分片（直接从 Stage A 的原始音频开始）
        b1_manifest = ChunkingProcessor(sub_cfgs["b1"]).process(
            input_manifest,
            os.path.join(output_dir, "B1"),
        )
        self._preflight_manifest(b1_manifest)

        # B2 静音过滤
        b2_manifest = SilenceFilterProcessor(sub_cfgs["b2"]).process(
            b1_manifest,
            os.path.join(output_dir, "B2"),
        )
        self._preflight_manifest(b2_manifest)

        # B3 VAD
        vad_processor = VADProcessor(sub_cfgs["b3"])
        try:
            b3_manifest = vad_processor.process(
                b2_manifest,
                os.path.join(output_dir, "B3"),
            )
            self._preflight_manifest(b3_manifest)
        finally:
            # 释放 VAD 进程池，避免 worker 进程滞留
            vad_processor.close()
        # 与 unified 分支/C/D 的 infra-failed 语义对齐：VAD 整批失败等可重试
        # 失败未写 checkpoint，必须中止 stage（下游不得消费不完整产物），
        # 修复后重跑自动重试；否则 stage 会"成功"完成但产出为空
        if vad_processor.has_retryable_failures():
            raise RuntimeError(
                "Stage B 存在可重试处理失败（如 VAD 整批失败），"
                "相关记录未写 checkpoint，重跑可自动重试"
            )

        # B4 策略过滤（基于 VAD 结果）
        b4_manifest = PolicyProcessor(sub_cfgs["b4"]).process(
            b3_manifest,
            os.path.join(output_dir, "B4"),
        )

        # 将 B4 本轮产出的 merged 单文件拆分为 per-dataset 后链接到 stage 根目录；
        # 该 merged 文件是本 stage 自己刚写出的中间产物，拆分后删除。
        split_manifest_by_dataset(b4_manifest, b4_manifest, remove_legacy_base=True)
        link_per_dataset_manifests(
            src_dir=os.path.join(output_dir, "B4"),
            src_prefix="b4",
            dst_dir=output_dir,
            dst_prefix="b",
        )
        self._prune_empty_dataset_outputs(output_dir, empty_input_dataset_ids)
        stats = write_stage_stats(
            manifest_path=output_dir,
            stage_name="B",
            output_path=os.path.join(output_dir, "b_stats.json"),
        )

        # stats 的单遍扫描已含总数与聚合 passed（B4 passed 的记录其聚合状态
        # 即 passed），不再为一条日志重复解析整个 manifest
        logger.info(
            "Stage B 完成：%d 条记录，B4 通过 %d，输出 %s",
            stats["total_records"], stats["status"]["passed"], output_dir,
        )
        return output_dir
