"""
B4: 策略过滤 Processor。

仅基于 VAD 结果应用保留策略，决定每条记录的最终档位
（noise / suspected / rejected）与是否保留。

注：原 AED 标签相关过滤（人声 / 音乐 / babble）已随旧 B5 删除，现由 Stage C3 admission 处理。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterator, List, Optional

from ..core.manifest_io import AudioRecord, read_manifest_strict, write_manifest
from ..core.processor_base import Processor
from .policy_engine import PolicyEngine

logger = logging.getLogger(__name__)


class PolicyProcessor(Processor):
    """B4: 策略过滤 Processor。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, stage_name="B4")
        self.keep_noise = bool(config.get("keep_noise", True))
        self.keep_suspected = bool(config.get("keep_suspected", False))
        self.keep_rejected = bool(config.get("keep_rejected", False))
        self.vad_speech_ratio_threshold = float(config.get("vad_speech_ratio_threshold", 0.05))
        self.speech_threshold = config.get("speech_threshold")
        self.delete_rejected_audio = bool(config.get("delete_rejected_audio", False))
        # 允许删除的根目录：仅限 pipeline 输出目录内的文件，防止误删用户源音频。
        # 由 process() 或上层（如 Stage B unified）注入；未设置时一律跳过删除。
        self.allowed_delete_root: Optional[str] = config.get("output_root")

        self._engine = PolicyEngine({
            "vad_speech_ratio_threshold": self.vad_speech_ratio_threshold,
            "speech_threshold": self.speech_threshold,
            "keep_noise": self.keep_noise,
            "keep_suspected": self.keep_suspected,
            "keep_rejected": self.keep_rejected,
            "defer_speech_to_stage_c": bool(config.get("defer_speech_to_stage_c", False)),
        })

    def _record_to_chunk(self, record: AudioRecord) -> Dict[str, Any]:
        """从 AudioRecord 构造 PolicyEngine 可用的 chunk dict（VAD 结果来自 B3）。"""
        b3_stage = record.stages.get("B3", {})
        b3_data = b3_stage if isinstance(b3_stage, dict) else {}
        return {
            "vad_result": b3_data.get("vad_result") if isinstance(b3_data, dict) else None,
            # B3 按自身阈值算出的三档类别；为 rejected 时 B4 必须尊重，不得翻案
            "vad_category": b3_data.get("vad_category") if isinstance(b3_data, dict) else None,
        }

    def _process_record(self, record: AudioRecord) -> AudioRecord:
        """处理单条记录。"""
        # VAD 被禁用时，直接按 noise 通过
        b3_stage = record.stages.get("B3", {})
        if isinstance(b3_stage, dict) and b3_stage.get("enabled") is False:
            record.labels["final_category"] = "noise"
            record.mark_stage(
                self.stage_name,
                "passed",
                extra={"final_category": "noise", "vad_disabled": True},
            )
            return record

        chunk = self._record_to_chunk(record)
        pr = self._engine.apply_to_chunk(chunk)

        record.labels["final_category"] = pr.final_category
        status = "passed" if pr.kept else "rejected"
        record.mark_stage(
            self.stage_name,
            status,
            extra={
                "final_category": pr.final_category,
                "fail_count": pr.fail_count,
                "layer1_kept": pr.layer1_kept,
            },
        )
        if not pr.kept:
            record.rejected_reason = pr.rejected_reason or "b4_policy"
            if self.delete_rejected_audio:
                self._safe_delete_audio(record.audio.get("path"))
        return record

    def _safe_delete_audio(self, path: Optional[str]) -> None:
        """删除 rejected 音频；仅限 pipeline 输出根目录内的文件。

        历史 bug：rejected 记录的 path 可能仍指向用户源音频（如整段保留的
        chunk 直接复用原路径），无条件 os.remove 会删掉用户数据。这里只删
        输出根目录内的文件，其余一律跳过并记录日志。
        """
        if not path or not os.path.isfile(path):
            return
        root = self.allowed_delete_root
        if not root:
            logger.warning(
                "B4 delete_rejected_audio 已开启但未设置输出根目录，跳过删除: %s", path
            )
            return
        real_path = os.path.realpath(path)
        real_root = os.path.realpath(root)
        try:
            under_root = os.path.commonpath([real_path, real_root]) == real_root
        except ValueError:
            # 不同盘符/路径风格无法比较时按不在根目录内处理
            under_root = False
        if not under_root:
            logger.warning("B4 跳过删除输出目录之外的文件: %s", path)
            return
        try:
            os.remove(real_path)
        except OSError as exc:
            logger.debug("B4 删除 rejected 音频失败 %s: %s", path, exc)

    def process_records(self, records: Iterator[AudioRecord]) -> List[AudioRecord]:
        """内存版策略过滤：逻辑与 process() 一致但不落盘，供 Stage B unified 复用。"""
        output: List[AudioRecord] = []
        total = 0
        kept = 0
        discarded = 0
        category_counts: Dict[str, int] = {}

        for rec in records:
            total += 1
            if self.should_skip(rec):
                cat = rec.labels.get("final_category", "unknown")
                category_counts[cat] = category_counts.get(cat, 0) + 1
                output.append(rec)
                continue

            # prev_stage 门控（与 stage_orchestrator._make_stage_pass_checker
            # 口径对齐）：B3 未 passed（rejected/suspected/skipped）的记录没有
            # 有效 VAD 结果，策略层可能按 keep_* 配置放行，导致上游已淘汰的
            # 记录"复活"；统一标 skipped 透传。无 B3 键的旧记录视为可处理，
            # 保持兼容。
            b3 = rec.stages.get("B3")
            if b3 is not None and b3.get("status") != "passed":
                rec.mark_stage(
                    self.stage_name,
                    "skipped",
                    extra={"reason": "prev_stage_B3_not_passed"},
                )
                output.append(rec)
                continue

            processed = self._process_record(rec)
            cat = processed.labels.get("final_category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if processed.is_stage_passed(self.stage_name):
                kept += 1
            else:
                discarded += 1
            output.append(processed)

        logger.info(
            "B4 策略过滤完成：%d 条记录，保留 %d，丢弃 %d，类别分布 %s",
            total,
            kept,
            discarded,
            category_counts,
        )
        return output

    def process(self, input_manifest: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        # 删除权限收敛到本 stage 输出目录之内
        if not self.allowed_delete_root:
            self.allowed_delete_root = output_dir
        output_manifest = os.path.join(output_dir, "b4_manifest.jsonl")

        output_records = self.process_records(
            read_manifest_strict(input_manifest, reject_duplicate_ids=True)
        )
        count = write_manifest(iter(output_records), output_manifest)
        logger.info("B4 策略过滤输出 %d 条记录: %s", count, output_manifest)
        return output_manifest
