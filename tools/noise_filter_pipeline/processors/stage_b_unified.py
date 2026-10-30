"""
Stage B 单文件 in-memory 流水线。

把 B1 分片 → B2 静音过滤 → B3 VAD → B4 策略过滤合并为一次处理：
  1. 读入源音频；
  2. 在内存中分片并做静音检测/缩短；
  3. 每累积 vad_batch_chunks 条 chunk 就 flush 一个 batch；
  4. batch 内直接对内存数组跑 VAD；
  5. 策略过滤，保留的 chunk 从内存数组直接写出到最终目录。

特性：
  - 支持 checkpoint / resume，kill 后可从断点继续；
  - 默认不写中间临时 chunk 音频，避免数百万次无意义 I/O；
  - 最终输出格式可配置（默认 flac）。

B5 AED 分类已删除；Stage C 改用 AudioTag 作为 AudioSet 分类步骤。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from ..core.checkpoint import CheckpointTracker
from ..core.manifest_io import (
    AudioRecord,
    _iter_manifest_files,
    read_manifest_strict,
)
from ..core.manifest_per_dataset import PerDatasetManifestWriter, init_checkpointed_output
from ..core.processor_base import Processor, _process_in_threads
from tools.shared.audio_io import (
    copy_or_convert_audio,
    load_audio_robust,
    make_unique_output_path,
    save_audio_atomically,
)
from .chunking_core import cut_adaptive, cut_hive, cut_none
from .policy_processor import PolicyProcessor
from .silence_processor import SilenceProcessor
from .vad_processor import VADProcessor

logger = logging.getLogger(__name__)


# 单批 chunk 音频常驻内存的安全预算：可用物理内存的比例上限。
# 批边界不影响处理结果（只影响 I/O 节奏），因此内存紧张时下调批大小是安全的
_BATCH_MEMORY_BUDGET_FRACTION = 0.25
# 估算每 chunk 内存时的采样率上界与 float32 字节数（保守按最高常见采样率）
_BATCH_ESTIMATE_MAX_SR = 48000
_BATCH_ESTIMATE_BYTES_PER_SAMPLE = 4


def _available_memory_bytes() -> Optional[int]:
    """返回可用物理内存字节数；平台不支持时返回 None。"""
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return int(pages) * int(page_size)


def _clamp_batch_chunks_by_memory(
    configured: int,
    *,
    chunk_duration: float,
    budget_fraction: float = _BATCH_MEMORY_BUDGET_FRACTION,
) -> int:
    """按可用物理内存钳制批内 chunk 数，估算不出可用内存时原样返回。

    整批 chunk 音频数组常驻内存（默认 2000 × 10s × 48k × float32 ≈ 3.8GB，
    16k 源约 1.3GB），叠加写盘线程预取窗口；内存小于预算时按最坏情况
    （``_BATCH_ESTIMATE_MAX_SR``）下调批大小，避免 OOM。
    """
    available = _available_memory_bytes()
    if available is None or configured < 1:
        return configured
    bytes_per_chunk = max(
        1.0,
        float(chunk_duration) * _BATCH_ESTIMATE_MAX_SR * _BATCH_ESTIMATE_BYTES_PER_SAMPLE,
    )
    cap = int((available * budget_fraction) // bytes_per_chunk)
    if cap < 1:
        logger.warning(
            "可用内存 %dMB 低于单批最小预算，vad_batch_chunks 钳制为 1（原值 %d）",
            available // (1024 * 1024), configured,
        )
        return 1
    if cap < configured:
        logger.warning(
            "可用内存 %dMB 不足以容纳 vad_batch_chunks=%d 的整批音频（估算每 chunk %dKB），"
            "钳制为 %d；如需更大批量请释放内存或降低 chunk_duration",
            available // (1024 * 1024), configured, int(bytes_per_chunk / 1024), cap,
        )
        return cap
    return configured


def _chunk_source(
    audio: np.ndarray,
    sr: int,
    mode: str,
    chunk_duration: float,
    min_chunk_duration: float,
    hive_window_sec: float,
    hive_step_sec: float,
    hive_energy_threshold: float,
    hive_min_tail_sec: float,
) -> List[Dict[str, Any]]:
    """按 mode 分片，共享核心算法。"""
    if mode == "adaptive":
        return cut_adaptive(audio, sr, chunk_duration, min_chunk_duration)
    if mode == "hive":
        return cut_hive(
            audio, sr,
            hive_window_sec,
            hive_step_sec,
            hive_energy_threshold,
            hive_min_tail_sec,
        )
    return cut_none(audio, sr)


def _line_count(path: str) -> int:
    """快速统计 manifest 行数（支持单文件或 per-dataset 目录），
    用于 tqdm total；比 JSON parse 整个 manifest 更轻量。"""
    n = 0
    for p in _iter_manifest_files(path):
        try:
            with open(p, "rb") as f:
                for _ in f:
                    n += 1
        except OSError:
            continue
    return n


class StageBUnifiedProcessor(Processor):
    """Stage B 合并处理器：单文件 in-memory 流水线。"""

    def __init__(self, config: Dict[str, Any], progress_callback: Optional[Callable[[], None]] = None):
        super().__init__(config, stage_name="B", progress_callback=progress_callback)
        self.num_workers = int(config.get("num_workers", 1))
        # num_workers 只是并发度、不改变处理语义：不进入配置指纹
        # （stage_b_curation 默认 8，随机器/任务变化；进入指纹会导致换机器
        # 或改并发后 checkpoint 被误判失效、全量重算）。
        # init_checkpointed_output 与 stage_orchestrator marker 均用 self.config。
        # b3.silero_group_size 同样只影响调度/吞吐，不改变处理语义；复制嵌套
        # dict 后剔除，避免修改调用方传入的运行时 b3 配置。
        self.config = {
            k: (dict(v) if isinstance(v, dict) else v)
            for k, v in config.items()
            if k != "num_workers"
        }
        checkpoint_b3 = self.config.get("b3")
        if isinstance(checkpoint_b3, dict):
            checkpoint_b3.pop("silero_group_size", None)
        # 基础设施失败计数（VAD 整批失败等未写 checkpoint 的记录数），
        # 供 stage_b_curation 判定本轮是否有可重试失败并中止 stage；
        # 仅后台 flush 线程自增，无并发竞争
        self._retryable_failures = 0
        # Stage B unified 临时目录：默认使用输出目录下的子目录，避免 /dev/shm 内存不足
        self.temp_dir = config.get("temp_dir")

        self.b1_cfg = dict(config.get("b1", {}))
        self.b2_cfg = dict(config.get("b2", {}))
        self.b3_cfg = dict(config.get("b3", {}))
        self.b4_cfg = dict(config.get("b4", {}))

        self.mode = self.b1_cfg.get("mode", "adaptive")
        self.max_peak = float(config.get("max_peak", 1.2))
        # 每累积多少条 chunk 就 flush 一次（VAD + 策略 + 落盘），避免大数据量 OOM。
        # 默认 2000 在显存/内存和 VAD batch 效率之间取平衡。
        # 必须 >= 1：配 0/负数时每条 chunk 单独 flush 一次（VAD → 策略 → 落盘），
        # 批处理退化为逐条且频繁 I/O，性能灾难；钳制到 1 并告警
        raw_vad_batch_chunks = int(config.get("vad_batch_chunks", 2000))
        if raw_vad_batch_chunks < 1:
            logger.warning(
                "Stage B unified vad_batch_chunks=%d 非法（必须 >= 1），钳制为 1",
                raw_vad_batch_chunks,
            )
        self.vad_batch_chunks = max(1, raw_vad_batch_chunks)
        # Stage B unified 最终输出音频格式，默认 flac 以节省磁盘空间。
        self.output_format = str(config.get("output_format", "flac"))
        available_formats = {fmt.lower() for fmt in sf.available_formats().keys()}
        if self.output_format.lower() not in available_formats:
            raise ValueError(
                f"Stage B unified 不支持的 output_format: {self.output_format}; "
                f"soundfile 可用格式: {sorted(available_formats)}"
            )
        # 是否保存中间临时 chunk 音频；默认 false，因为 VAD/落盘都走内存数组，
        # 写临时 wav 会引入数百万次无意义 I/O。
        self.save_temp_chunks = bool(config.get("save_temp_chunks", False))
        # 最终输出音频是否按文件名 hash 前 2 位分桶（audio/<shard2>/<file>）。
        # 默认 false 保持存量平铺布局；新运行建议开启，避免单目录百万级文件
        # （重构设计文档第 2 步：stage_b 与 D2 sr_label_dataset 同套分桶语义）
        self.output_shard = bool(config.get("output_shard", False))
        # checkpoint / manifest 批量写入
        self.enable_checkpoint = bool(config.get("enable_checkpoint", True))
        self.checkpoint_path = config.get("checkpoint_path")
        self.checkpoint_save_interval = int(config.get("checkpoint_save_interval", 100))
        self.manifest_write_batch_size = int(config.get("manifest_write_batch_size", 1000))
        self.chunk_duration = float(self.b1_cfg.get("chunk_duration", 10.0))
        # 整批 chunk 音频常驻内存，按可用内存再钳一次（不影响结果，只影响批边界）
        self.vad_batch_chunks = _clamp_batch_chunks_by_memory(
            self.vad_batch_chunks, chunk_duration=self.chunk_duration
        )
        self.min_chunk_duration = float(self.b1_cfg.get("min_chunk_duration", 2.0))
        self.hive_window_sec = float(self.b1_cfg.get("hive_window_sec", 10.0))
        self.hive_step_sec = float(self.b1_cfg.get("hive_step_sec", 5.0))
        self.hive_energy_threshold = float(self.b1_cfg.get("hive_energy_threshold", 0.0005))
        self.hive_min_tail_sec = float(self.b1_cfg.get("hive_min_tail_sec", 5.0))
        self.hive_output_label_key = self.b1_cfg.get("hive_output_label_key", "hive_chunk_source")

        self._silence_processor = SilenceProcessor(
            max_silence_ratio=float(self.b2_cfg.get("max_silence_ratio", 0.9)),
            min_duration=float(self.b2_cfg.get("min_chunk_duration_filter", 0.5)),
            shorten_long_silences_flag=bool(self.b2_cfg.get("shorten_long_silences", True)),
            max_silence_duration=float(self.b2_cfg.get("max_silence_duration", 2.0)),
            target_silence_duration=float(self.b2_cfg.get("target_silence_duration", 1.0)),
            target_silence_min=float(self.b2_cfg.get("target_silence_min", 0.5)),
            target_silence_max=float(self.b2_cfg.get("target_silence_max", 1.5)),
            use_random_silence_duration=bool(self.b2_cfg.get("use_random_silence_duration", True)),
            abs_min_energy=self.b2_cfg.get("abs_min_energy"),
            silent_energy_threshold=float(self.b2_cfg.get("silent_energy_threshold", 0.005)),
            dc_offset_threshold=self.b2_cfg.get("dc_offset_threshold", 0.5),
            random_seed=self.b2_cfg.get("random_seed"),
        )

        self.b2_enable = bool(self.b2_cfg.get("enable", True))
        self.b3_enable = bool(self.b3_cfg.get("enable", True))

        self._vad_processor = VADProcessor(self.b3_cfg)
        self._policy_processor = PolicyProcessor(self.b4_cfg)

    def _make_chunk_record(
        self,
        parent: AudioRecord,
        chunk_info: Dict[str, Any],
        chunk_path: str,
    ) -> AudioRecord:
        """由 chunk 元信息构造 AudioRecord。"""
        is_original = chunk_info.get("is_original", False)
        chunk_idx = chunk_info.get("chunk_idx", 1)
        chunk_id = (parent.chunk_id or "source") if is_original else f"chunk_{chunk_idx:04d}"
        rec = AudioRecord(
            record_id=AudioRecord.make_record_id(parent.dataset_id, parent.source_id, chunk_id),
            dataset_id=parent.dataset_id,
            source_id=parent.source_id,
            chunk_id=chunk_id,
            split=parent.split,
            audio={
                "path": chunk_path,
                "original_path": parent.audio.get("original_path") or parent.audio.get("path"),
                "sr": chunk_info["sample_rate"],
                "duration": chunk_info["duration"],
                # chunking_core 已转单声道，不得继承父记录的声道数
                "channels": 1,
            },
            labels=dict(parent.labels),
            provenance=dict(parent.provenance),
        )
        # 深拷贝 stages/labels：chunk 记录生命周期独立于父记录，后续 mark_stage /
        # 标签修改不得反向影响父记录（dict() 浅拷贝会共享 stage entry / list 标签）
        rec.stages = {k: dict(v) for k, v in parent.stages.items()}
        rec.labels = {
            k: (list(v) if isinstance(v, list) else v)
            for k, v in parent.labels.items()
        }
        if self.mode == "hive":
            rec.labels[self.hive_output_label_key] = parent.record_id
        rec.mark_stage(
            "B1",
            "passed",
            extra={
                "start_sample": chunk_info.get("start_sample"),
                "end_sample": chunk_info.get("end_sample"),
                "energy": chunk_info.get("energy"),
                "is_original": is_original,
                "mode": self.mode,
            },
        )
        rec.mark_stage(
            "B2",
            "passed",
            extra={
                "silence_ratio": chunk_info.get("silence_ratio"),
                "original_duration": chunk_info.get("original_duration"),
            },
        )
        return rec

    def _b2_rejection_reason(self, result: Dict[str, Any]) -> str:
        """从 B2 结果派生稳定、可审计的拒绝原因。"""
        if result.get("dc_offset") is not None and result.get("dc_offset_rejected"):
            return "dc_offset"
        if result.get("is_silent"):
            return "silent"
        # 与拒绝阈值同口径：silence_ratio 越过 max_silence_ratio 即归因
        # silence_ratio（此前写死 >=1.0，0.9~1.0 之间的拒绝被误归 silence_filter）
        if result.get("silence_ratio", 0.0) >= self._silence_processor.max_silence_ratio:
            return "silence_ratio"
        return "silence_filter"

    def _split_and_filter_source(
        self,
        record: AudioRecord,
        tmp_audio_dir: str,
    ) -> List[Tuple[AudioRecord, Optional[np.ndarray], Optional[int]]]:
        """对单条源记录分片、静音过滤、写入临时 chunk 文件。

        返回列表元素为 (AudioRecord, audio_array, sr)。AudioRecord 中的 path
        指向临时 chunk 文件，audio_array 保留在内存中供后续 VAD 与最终落盘复用，
        避免反复读写磁盘。
        """
        audio_path = record.audio.get("path")
        if not audio_path or not os.path.isfile(audio_path):
            logger.warning("B unified record %s: 音频文件不存在 %s", record.record_id, audio_path)
            record.mark_stage("B1", "rejected", extra={"reason": "missing_audio"})
            record.rejected_reason = "b1_missing_audio"
            return [(record, None, None)]

        try:
            audio, sr = load_audio_robust(audio_path, max_peak=self.max_peak)
        except Exception as exc:  # noqa: BLE001
            logger.warning("B unified 读取音频失败 %s: %s", audio_path, exc)
            record.mark_stage("B1", "rejected", extra={"reason": "read_error"})
            record.rejected_reason = "b1_read_error"
            return [(record, None, None)]

        chunks = _chunk_source(
            audio, sr,
            mode=self.mode,
            chunk_duration=self.chunk_duration,
            min_chunk_duration=self.min_chunk_duration,
            hive_window_sec=self.hive_window_sec,
            hive_step_sec=self.hive_step_sec,
            hive_energy_threshold=self.hive_energy_threshold,
            hive_min_tail_sec=self.hive_min_tail_sec,
        )

        if not chunks:
            record.mark_stage("B1", "rejected", extra={"reason": "no_chunks"})
            record.rejected_reason = "b1_no_chunks"
            return [(record, None, None)]

        results: List[Tuple[AudioRecord, Optional[np.ndarray], Optional[int]]] = []
        for chunk_info in chunks:
            if self.b2_enable:
                # chunking_core 已保证 audio 为单声道，无需再次转换。
                chunk_audio = chunk_info["audio"]
                silence_result = self._silence_processor.process_audio(
                    chunk_audio,
                    chunk_info["sr"],
                    # 按源记录 + chunk 序号派生随机流：同一 chunk 跨运行可复现，
                    # 且与线程调度无关
                    rng_context=f"{record.record_id}#{chunk_info.get('chunk_idx', 1)}",
                )
                if not silence_result.get("passed", True):
                    rec = self._make_chunk_record(record, chunk_info, "")
                    reason = self._b2_rejection_reason(silence_result)
                    rec.mark_stage(
                        "B2", "rejected",
                        extra={
                            "reason": reason,
                            "silence_ratio": silence_result.get("silence_ratio"),
                            "is_silent": silence_result.get("is_silent"),
                            "original_duration": silence_result.get("original_duration"),
                        },
                    )
                    rec.rejected_reason = f"b2_{reason}"
                    results.append((rec, None, None))
                    continue
                chunk_info["audio"] = silence_result.get("processed_audio", chunk_audio)
                chunk_info["silence_ratio"] = silence_result.get("silence_ratio")
                chunk_info["original_duration"] = silence_result.get("original_duration")
                chunk_info["duration"] = float(len(chunk_info["audio"]) / chunk_info["sr"])

            stem = Path(audio_path).stem
            chunk_idx = chunk_info.get("chunk_idx", 1)
            # 用 source 路径稳定 hash 避免同名文件碰撞；同时按 hash 前 2 字符分片目录，降低单目录文件数
            source_hash = hashlib.sha1(str(audio_path).encode("utf-8")).hexdigest()[:12]
            hash_prefix = source_hash[:2]
            chunk_name = f"{record.dataset_id}_{source_hash}_{stem}_chunk_{chunk_idx:04d}.wav"
            tmp_path = os.path.join(tmp_audio_dir, record.dataset_id, hash_prefix, chunk_name)
            if self.save_temp_chunks:
                os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
                try:
                    save_audio_atomically(chunk_info["audio"], chunk_info["sr"], tmp_path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("B unified 写入临时 chunk 失败 %s: %s", tmp_path, exc)
                    continue

            rec = self._make_chunk_record(record, chunk_info, tmp_path)
            # chunk_info 在本函数返回后即丢弃，数组无需再 copy 一份（batch 内存减半）
            results.append((rec, chunk_info["audio"], chunk_info["sr"]))

        if not results:
            record.mark_stage("B1", "rejected", extra={"reason": "all_chunks_filtered"})
            record.rejected_reason = "b1_all_chunks_filtered"
            return [(record, None, None)]

        return results

    def _prepare_batch(
        self,
        batch_records: List[AudioRecord],
        batch_arrays: List[Optional[Tuple[np.ndarray, int]]],
        vad_mask: List[bool],
        final_audio_dir: str,
    ) -> Tuple[
        List[AudioRecord],
        List[Tuple[AudioRecord, str, str, Optional[np.ndarray], Optional[int]]],
    ]:
        """prepare 阶段：VAD → B4 → 生成写盘任务；不落盘、不写 manifest/checkpoint。

        vad_mask 标记需要 VAD 的槽位；B1/B2 rejected 与 checkpoint 命中记录为
        terminal 条目原样透传。返回（按输入顺序的记录列表，写盘任务列表）。
        """
        eligible_idx = [i for i, m in enumerate(vad_mask) if m]
        if self.b3_enable:
            if eligible_idx:
                sub_records = [batch_records[i] for i in eligible_idx]
                # 防御：batch_arrays 与 records 长度不一致（旧调用方可能传空列表）时，
                # 缺失槽位用空数组占位，保持与 VADProcessor 的按序对齐语义
                sub_arrays: List[Tuple[np.ndarray, int]] = []
                for i in eligible_idx:
                    entry = (
                        batch_arrays[i]
                        if i < len(batch_arrays)
                        else None
                    )
                    sub_arrays.append(
                        entry
                        if entry is not None
                        else (np.array([], dtype=np.float32), 16000)
                    )
                processed = self._vad_processor.process_records_with_arrays(
                    sub_records, sub_arrays
                )
                for rec, i in zip(processed, eligible_idx):
                    batch_records[i] = rec
        else:
            for i in eligible_idx:
                batch_records[i].mark_stage(
                    "B3", "passed", extra={"enabled": False}
                )

        if eligible_idx:
            policy_records = self._policy_processor.process_records(
                iter([batch_records[i] for i in eligible_idx])
            )
            for rec, i in zip(policy_records, eligible_idx):
                batch_records[i] = rec

        audio_by_id: Dict[str, Optional[Tuple[np.ndarray, int]]] = {}
        for i in eligible_idx:
            entry = (
                batch_arrays[i]
                if i < len(batch_arrays)
                else None
            )
            audio_by_id[batch_records[i].record_id] = entry
        output_ext = f".{self.output_format}"
        jobs: List[Tuple[AudioRecord, str, str, Optional[np.ndarray], Optional[int]]] = []
        for i in eligible_idx:
            rec = batch_records[i]
            b3_stage = rec.stages.get("B3", {})
            if (
                b3_stage.get("status") == "rejected"
                and b3_stage.get("reason") == "vad_batch_failed"
            ):
                # B3 基础设施失败：无论 keep_rejected 配置如何都不得写出音频
                # （infra 失败的音频未经 VAD 判定，放行等于把未检数据送入下游），
                # 统一按 rejected 处理；_commit_batch 不会为其写 checkpoint，
                # resume 时可重试
                if rec.stages.get("B4", {}).get("status") != "rejected":
                    rec.mark_stage("B4", "rejected", extra={"reason": "b3_vad_batch_failed"})
                rec.rejected_reason = "b3_vad_batch_failed"
                continue
            if not rec.is_stage_passed("B4"):
                continue
            src_path = rec.audio["path"]
            audio_entry = audio_by_id.get(rec.record_id)
            audio_arr, sr = audio_entry if audio_entry is not None else (None, None)
            # 输出文件名必须确定性：src_path 是随机临时 chunk 路径（每次运行不同），
            # 直接用作 hash 源会让崩溃 resume 后同一 chunk 写出第二个不同名文件，
            # audio 目录积累孤儿重复。改用 original_path + chunk_id 构造确定性伪路径：
            # 先 abspath 再拼 #chunk_ 后缀，hash 跨运行稳定，stem 仍取 original 的 stem。
            original_path = rec.audio.get("original_path") or src_path
            hash_src = f"{os.path.abspath(original_path)}#chunk_{rec.chunk_id}"
            dst_path = make_unique_output_path(
                final_audio_dir, hash_src, ext=output_ext, shard=self.output_shard
            )
            jobs.append((rec, src_path, dst_path, audio_arr, sr))
        return batch_records, jobs

    def _write_kept_audio(
        self,
        job: Tuple[AudioRecord, str, str, Optional[np.ndarray], Optional[int]],
    ) -> None:
        """单个最终音频写盘任务。

        基础设施异常（ENOSPC/EIO 等）直接抛出并经 Future.result() 传播中止
        batch，不得当作内容级拒绝写 checkpoint；仅缺内存音频且无临时文件可
        复制时按内容级 missing_memory_audio 拒绝。
        """
        rec, src_path, dst_path, audio_arr, sr = job
        if audio_arr is not None and sr is not None and audio_arr.size > 0:
            save_audio_atomically(audio_arr, sr, dst_path)
            rec.audio["path"] = dst_path
            rec.audio["sr"] = sr
            return
        logger.warning("B unified 记录 %s 缺少内存音频", rec.record_id)
        if self.save_temp_chunks and os.path.isfile(src_path):
            copy_or_convert_audio(
                src_path, dst_path, target_sr=None, max_peak=self.max_peak
            )
            rec.audio["path"] = dst_path
            return
        rec.mark_stage("B4", "rejected", extra={"reason": "missing_memory_audio"})
        rec.rejected_reason = "b4_missing_memory_audio"

    def _commit_batch(
        self,
        ordered_records: List[AudioRecord],
        output_writer: PerDatasetManifestWriter,
        stats: Dict[str, int],
        ckpt: CheckpointTracker,
    ) -> None:
        """commit 阶段：按输入顺序 append manifest、更新统计并标记 checkpoint。

        只由协调线程在写盘全部完成后调用；B1/B2 rejected 与 checkpoint 命中
        记录同样经此按序落盘，保证每个数据集 manifest 与输入逻辑顺序一致。
        """
        done_ids: List[str] = []
        for rec in ordered_records:
            b1_stage = rec.stages.get("B1", {})
            if b1_stage.get("status") == "rejected":
                # B1 拒绝的源记录直接透传（不消耗 total），与旧语义一致
                output_writer.append(rec)
                ckpt.mark_done(rec.record_id)
                stats["rejected_sources"] = stats.get("rejected_sources", 0) + 1
                continue
            output_writer.append(rec)
            stats["total"] = stats.get("total", 0) + 1
            if rec.is_stage_passed("B4"):
                stats["kept"] = stats.get("kept", 0) + 1
            else:
                stats["discarded"] = stats.get("discarded", 0) + 1
            if rec.stages.get("B2", {}).get("status") == "rejected":
                stats["b2_rejected"] = stats.get("b2_rejected", 0) + 1
            # 基础设施失败（VAD 整批失败）不写 checkpoint，resume 时可重试；
            # 与 C2/C3/D1 的 infra-failed 语义对齐。
            # 注意必须读 B3 阶段标记而非 rejected_reason——后者会被 B4 策略覆盖
            b3_stage = rec.stages.get("B3", {})
            if b3_stage.get("status") == "rejected" and b3_stage.get("reason") == "vad_batch_failed":
                self._retryable_failures += 1
                continue
            done_ids.append(rec.record_id)
        ckpt.mark_done_batch(done_ids)

    def _process_batch(
        self,
        batch_records: List[AudioRecord],
        batch_arrays: List[Tuple[np.ndarray, int]],
        final_audio_dir: str,
    ) -> List[AudioRecord]:
        """兼容入口：prepare + 文件级并发写盘，返回按输入顺序的记录。

        VAD 与策略过滤结果直接在内存中流转，不再像旧实现那样每个 batch
        写/读 4 份中间 JSONL（chunks/b3/b4/batch_output）。
        """
        ordered, jobs = self._prepare_batch(
            batch_records,
            list(batch_arrays),
            [True] * len(batch_records),
            final_audio_dir,
        )
        # FLAC 编码/写盘在 native 层释放 GIL。复用文件级 num_workers 做有界并发，
        # 但保持结果消费顺序不变，确保 manifest/checkpoint 仍按输入顺序写入。
        list(_process_in_threads(jobs, self._write_kept_audio, self.num_workers))
        return ordered

    def _flush_batch(
        self,
        batch_records: List[AudioRecord],
        batch_arrays: List[Tuple[np.ndarray, int]],
        final_audio_dir: str,
        output_writer: PerDatasetManifestWriter,
        stats: Dict[str, int],
        ckpt: CheckpointTracker,
    ) -> None:
        """兼容入口：串行执行 prepare → 写盘 → commit（供测试直接调用）。"""
        if not batch_records:
            return
        ordered, jobs = self._prepare_batch(
            batch_records,
            list(batch_arrays),
            [True] * len(batch_records),
            final_audio_dir,
        )
        for job in jobs:
            self._write_kept_audio(job)
        self._commit_batch(ordered, output_writer, stats, ckpt)

    def has_retryable_failures(self) -> bool:
        """是否有未写 checkpoint、可在下次运行重试的基础设施失败。

        与 C/D 的 has_retryable_failures 语义一致，供 stage_b_curation
        检测后中止 stage，避免"VAD 整批失败但 stage 照常成功"。
        """
        return self._retryable_failures > 0

    def process(self, input_manifest: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_manifest = os.path.join(output_dir, "b_manifest.jsonl")
        final_audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(final_audio_dir, exist_ok=True)
        # B4 删除 rejected 音频时只允许动输出目录内的文件，防止误删用户源音频
        self._policy_processor.allowed_delete_root = output_dir

        manifest_writer, ckpt, prev_records = init_checkpointed_output(
            output_manifest, output_dir, self.config, input_manifest,
            enable_checkpoint=self.enable_checkpoint,
            checkpoint_path=self.checkpoint_path,
            save_interval=self.checkpoint_save_interval,
            manifest_batch_size=self.manifest_write_batch_size,
            stage_label="Stage B unified",
        )

        # 临时目录默认落在输出目录（output_dir）下，可通过 temp_dir 配置到高速 SSD；
        # 不默认使用 /dev/shm，避免大规模数据时内存不足。
        temp_parent = self.temp_dir or output_dir
        os.makedirs(temp_parent, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="stage_b_unified_", dir=temp_parent) as tmpdir:
            tmp_audio_dir = os.path.join(tmpdir, "audio")
            os.makedirs(tmp_audio_dir, exist_ok=True)

            source_total = _line_count(input_manifest)
            source_count = 0

            # 跨 batch 两级流水线：write(N) 在后台写盘线程池执行，prepare/VAD(N+1)
            # 在前台执行，二者重叠；commit 严格按 batch 顺序由本线程串行完成，
            # manifest/checkpoint 顺序因此与输入逻辑顺序一致。
            # pending_write = (Future, ordered_records)：当前唯一后台写盘 batch；
            # 同一时刻最多 1 个写盘 batch + 1 个 prepare/VAD batch。
            batch_records: List[AudioRecord] = []
            batch_arrays: List[Optional[Tuple[np.ndarray, int]]] = []
            vad_mask: List[bool] = []
            eligible_count = 0
            stats: Dict[str, int] = {
                "total": 0, "kept": 0, "discarded": 0,
                "b2_rejected": 0, "rejected_sources": 0,
            }
            pending_write: Optional[Tuple[Future, List[AudioRecord]]] = None

            def run_write_jobs(
                jobs: List[Tuple[AudioRecord, str, str, Optional[np.ndarray], Optional[int]]],
            ) -> None:
                for job in jobs:
                    self._write_kept_audio(job)

            def log_batch_completion(ordered: List[AudioRecord]) -> None:
                # 与旧 _process_batch 的同名日志口径一致：total 只含 chunk
                # （B1 拒绝的源记录不计），供长时任务按批观测进度
                chunk_total = sum(
                    1 for r in ordered
                    if r.stages.get("B1", {}).get("status") != "rejected"
                )
                kept = sum(1 for r in ordered if r.is_stage_passed("B4"))
                logger.info(
                    "Stage B unified batch 完成：%d 条 chunk，保留 %d，丢弃 %d",
                    chunk_total, kept, chunk_total - kept,
                )

            def commit_with_log(ordered: List[AudioRecord]) -> None:
                self._commit_batch(ordered, manifest_writer, stats, ckpt)
                log_batch_completion(ordered)

            def try_commit_pending() -> None:
                nonlocal pending_write
                if pending_write is None:
                    return
                fut, ordered = pending_write
                # 写盘基础设施异常（ENOSPC/EIO 等）在此抛出并中止 stage；
                # 本批与后续批均不 commit / 不写 checkpoint，resume 可重试
                fut.result()
                commit_with_log(ordered)
                pending_write = None

            def flush_current(write_pool: ThreadPoolExecutor) -> None:
                nonlocal batch_records, batch_arrays, vad_mask, eligible_count, pending_write
                if not batch_records:
                    return
                ordered, jobs = self._prepare_batch(
                    batch_records, batch_arrays, vad_mask, final_audio_dir,
                )
                batch_records, batch_arrays, vad_mask = [], [], []
                eligible_count = 0
                # prepare(N+1) 期间 write(N) 已在后台运行；先提交上一批再排队本批，
                # 保证 batch 间 commit 顺序
                try_commit_pending()
                if jobs:
                    pending_write = (
                        write_pool.submit(run_write_jobs, jobs),
                        ordered,
                    )
                else:
                    commit_with_log(ordered)

            try:
                with ThreadPoolExecutor(
                    max_workers=max(1, self.num_workers),
                    thread_name_prefix="stage_b_write",
                ) as write_pool:
                    with manifest_writer:
                        # 注意：不在这里创建内部 tqdm，统一使用 StageRunner 传入的
                        # global_pbar 刷新，避免两个进度条同时写 ANSI 控制序列导致乱码。
                        for chunks in _process_in_threads(
                            read_manifest_strict(input_manifest, reject_duplicate_ids=True),
                            lambda rec: self._split_and_filter_source(rec, tmp_audio_dir),
                            self.num_workers,
                        ):
                            source_count += 1
                            if self.progress_callback is not None:
                                self.progress_callback()
                            # 每 1000 个源文件打印一次进度日志，便于长时任务观察
                            if source_count % 1000 == 0:
                                logger.info(
                                    "Stage B unified 已处理 %d/%d 源文件，当前 batch %d 条 chunk",
                                    source_count,
                                    source_total,
                                    len(batch_records),
                                )
                            for rec, audio_arr, sr in chunks:
                                if ckpt.is_done(rec.record_id):
                                    prev = prev_records.get(rec.record_id)
                                    if prev is not None:
                                        # checkpoint 命中：作为 terminal 条目进入当前
                                        # batch，commit 时按输入顺序落盘，不再越过
                                        # 尚未完成的更早 batch 直接 append
                                        batch_records.append(prev)
                                        batch_arrays.append(None)
                                        vad_mask.append(False)
                                        continue
                                    # checkpoint 命中但 prev_records 缺少该条目（如上次
                                    # manifest flush 不完整）：不得静默跳过，回退为重新处理，
                                    # 与 C2/C3/D1 的 resume 语义一致。

                                if rec.stages.get("B1", {}).get("status") == "rejected":
                                    # B1 拒绝的源记录透传到最终 manifest，避免进入 B3/B4
                                    # 误伤原始数据文件或被 B4 “复活”。
                                    batch_records.append(rec)
                                    batch_arrays.append(None)
                                    vad_mask.append(False)
                                    continue
                                if rec.stages.get("B2", {}).get("status") == "rejected":
                                    # B2 是内容级拒绝：保留审计记录，但不得继续进入 B3/B4。
                                    batch_records.append(rec)
                                    batch_arrays.append(None)
                                    vad_mask.append(False)
                                    continue

                                if not self.b3_enable:
                                    # B3 禁用时必须在写出前标记，否则落盘 manifest 缺 B3 状态
                                    rec.mark_stage("B3", "passed", extra={"enabled": False})
                                batch_records.append(rec)
                                if audio_arr is not None and sr is not None:
                                    batch_arrays.append((audio_arr, sr))
                                else:
                                    # 理论上不会到达这里（B1 passed 的 chunk 必有音频）
                                    batch_arrays.append((np.array([], dtype=np.float32), 16000))
                                vad_mask.append(True)
                                eligible_count += 1

                                # 封批条件：eligible 达到 vad_batch_chunks 立即 flush；
                                # 纯 rejected/resume 的 terminal 批也按同一上限封批，
                                # 避免 batch 无限增长
                                if (
                                    eligible_count >= self.vad_batch_chunks
                                    or len(batch_records) >= self.vad_batch_chunks
                                ):
                                    flush_current(write_pool)

                        # 尾部 batch
                        if batch_records:
                            flush_current(write_pool)
                        try_commit_pending()

                        ckpt.save(force=True)
            finally:
                # 释放 VAD 进程池与写盘线程池，避免 worker 进程/线程滞留
                self._vad_processor.close()

            logger.info(
                "Stage B unified 完成：%d 条源记录 -> %d 条 chunk，保留 %d，丢弃 %d（B2 拒绝 %d），B1 拒绝源 %d，输出 %s",
                source_count,
                stats["total"],
                stats["kept"],
                stats["discarded"],
                stats["b2_rejected"],
                stats["rejected_sources"],
                output_manifest,
            )

        return output_dir
