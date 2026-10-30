"""
B1: 统一音频分片 Processor。

合并原 HiveChunkingProcessor 与 audio_utils.split_audio 的能力，
支持三种分片模式：
  - adaptive: 无重叠固定时长分片，尾段不足 min_chunk_duration 丢弃。
  - hive:    10s 窗口 / 5s hop，丢弃低能量片段与末尾残余。
  - none:    不分片，原音频直接透传。

输出每条 chunk 作为一个 AudioRecord，继承源记录的标签与来源信息。
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ..core.manifest_io import AudioRecord, read_manifest, write_manifest
from ..core.processor_base import Processor, _process_in_threads
from tools.shared.audio_io import load_audio_robust, save_audio
from .chunking_core import cut_adaptive, cut_hive, cut_none

logger = logging.getLogger(__name__)


class ChunkingProcessor(Processor):
    """B1: 统一分片 Processor。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, stage_name="B1")
        self.num_workers = int(config.get("num_workers", 1))
        self.mode = config.get("mode", "adaptive")

        # adaptive 模式参数
        self.chunk_duration = float(config.get("chunk_duration", 10.0))
        self.min_chunk_duration = float(config.get("min_chunk_duration", 2.0))

        # hive 模式参数
        self.hive_window_sec = float(config.get("hive_window_sec", 10.0))
        self.hive_step_sec = float(config.get("hive_step_sec", 5.0))
        self.hive_energy_threshold = float(config.get("hive_energy_threshold", 0.0005))
        self.hive_min_tail_sec = float(config.get("hive_min_tail_sec", 5.0))
        self.hive_output_label_key = config.get("hive_output_label_key", "hive_chunk_source")
        self.max_peak = float(config.get("max_peak", 1.2))

        if self.mode not in {"adaptive", "hive", "none"}:
            raise ValueError(f"B2 不支持的 mode: {self.mode}")

    def _cut_audio(
        self,
        audio_path: str,
    ) -> tuple[List[Dict[str, Any]], Optional[Dict[str, str]]]:
        """读取音频并按当前 mode 分片。

        Returns:
            (chunks, error)。成功时 error 为 None；失败时返回空列表与
            ``{"reason": "...", "exception_type": "...", "exception_msg": "..."}``
            字典，用于在 stage extra 中记录可读的异常分类与原始信息。
        """
        if not os.path.isfile(audio_path):
            return [], {"reason": "missing_audio"}
        try:
            audio, sr = load_audio_robust(audio_path, max_peak=self.max_peak)
        except ValueError as exc:
            logger.warning("B2 音频质量检查失败 %s: %s", audio_path, exc)
            return [], {
                "reason": "quality_check_failed",
                "exception_type": type(exc).__name__,
                "exception_msg": str(exc),
            }
        except OSError as exc:
            logger.warning("B2 读取音频失败(OS) %s: %s", audio_path, exc)
            return [], {
                "reason": "read_error_os",
                "exception_type": type(exc).__name__,
                "exception_msg": str(exc),
            }
        except RuntimeError as exc:
            logger.warning("B2 读取音频失败(Runtime) %s: %s", audio_path, exc)
            return [], {
                "reason": "read_error_runtime",
                "exception_type": type(exc).__name__,
                "exception_msg": str(exc),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("B2 读取音频失败 %s: %s", audio_path, exc)
            return [], {
                "reason": f"read_error_{type(exc).__name__}",
                "exception_type": type(exc).__name__,
                "exception_msg": str(exc),
            }

        if self.mode == "adaptive":
            return cut_adaptive(audio, sr, self.chunk_duration, self.min_chunk_duration), None
        if self.mode == "hive":
            return cut_hive(
                audio, sr,
                self.hive_window_sec,
                self.hive_step_sec,
                self.hive_energy_threshold,
                self.hive_min_tail_sec,
            ), None
        return cut_none(audio, sr), None

    def _save_chunks(
        self,
        audio_path: str,
        output_dir: str,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """将 chunk 元信息落盘，并补充 path 字段。"""
        stem = Path(audio_path).stem
        ext = Path(audio_path).suffix or ".wav"
        # 源路径稳定短 hash：避免不同源文件同名（同 stem）时 chunk 互相覆盖
        source_hash = hashlib.sha1(os.path.abspath(audio_path).encode("utf-8")).hexdigest()[:8]
        results: List[Dict[str, Any]] = []

        for chunk_info in chunks:
            chunk_idx = chunk_info.get("chunk_idx", 1)
            if chunk_info.get("is_original"):
                # 整段保留：直接复用原路径不写拷贝
                # （hive 短音频此前会写出无人引用的孤儿副本）
                output_path = audio_path
            elif self.mode == "hive":
                output_path = os.path.join(output_dir, f"{stem}_{source_hash}_chunk{chunk_idx}{ext}")
            else:
                output_path = os.path.join(output_dir, f"{stem}_{source_hash}_chunk_{chunk_idx:04d}.wav")

            if output_path != audio_path:
                try:
                    save_audio(chunk_info["audio"], chunk_info["sr"], output_path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("B2 写入 chunk 失败 %s: %s", output_path, exc)
                    continue

            chunk_info["path"] = output_path
            results.append(chunk_info)

        return results

    def _make_chunk_record(
        self,
        parent: AudioRecord,
        chunk_idx: int,
        chunk_info: Dict[str, Any],
    ) -> AudioRecord:
        """从 chunk 元信息构造新的 AudioRecord。"""
        is_original = chunk_info.get("is_original", False)
        chunk_id = (parent.chunk_id or "source") if is_original else f"chunk_{chunk_idx:04d}"
        new_record = AudioRecord(
            record_id=AudioRecord.make_record_id(parent.dataset_id, parent.source_id, chunk_id),
            dataset_id=parent.dataset_id,
            source_id=parent.source_id,
            chunk_id=chunk_id,
            split=parent.split,
            audio={
                "path": chunk_info["path"],
                "original_path": parent.audio.get("original_path") or parent.audio.get("path"),
                "sr": chunk_info["sample_rate"],
                "duration": chunk_info["duration"],
                # chunking_core 已转单声道，不得继承父记录的声道数
                "channels": 1,
            },
            labels=dict(parent.labels),
            provenance=dict(parent.provenance),
        )
        if self.mode == "hive":
            new_record.labels[self.hive_output_label_key] = parent.record_id
        new_record.stages = dict(parent.stages)
        new_record.mark_stage(
            self.stage_name,
            "passed",
            extra={
                "start_sample": chunk_info.get("start_sample"),
                "end_sample": chunk_info.get("end_sample"),
                "energy": chunk_info.get("energy"),
                "is_original": is_original,
                "mode": self.mode,
            },
        )
        return new_record

    def _chunk_record(
        self,
        record: AudioRecord,
        output_audio_dir: str,
    ) -> Iterator[AudioRecord]:
        """对单条记录执行分片，产出 chunk 记录。"""
        audio_path = record.audio.get("path")
        if not audio_path or not os.path.isfile(audio_path):
            logger.warning("B1 record %s: 音频文件不存在 %s", record.record_id, audio_path)
            record.mark_stage(self.stage_name, "rejected", extra={"reason": "missing_audio"})
            record.rejected_reason = "b1_missing_audio"
            yield record
            return

        dataset_dir = os.path.join(output_audio_dir, record.dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)

        chunks, cut_error = self._cut_audio(audio_path)
        if chunks:
            chunks = self._save_chunks(audio_path, dataset_dir, chunks)

        if not chunks:
            reason = cut_error.get("reason") if cut_error else "no_chunks"
            extra: Dict[str, Any] = {"reason": reason}
            if cut_error:
                extra["exception_type"] = cut_error.get("exception_type")
                extra["exception_msg"] = cut_error.get("exception_msg")
            record.mark_stage(self.stage_name, "rejected", extra=extra)
            record.rejected_reason = f"b1_{reason}"
            yield record
            return

        # 单条 chunk 且保留原路径时，直接在原记录上标记
        if len(chunks) == 1 and chunks[0].get("is_original"):
            record.mark_stage(self.stage_name, "passed", extra={
                "is_original": True,
                "mode": self.mode,
            })
            yield record
            return

        for idx, chunk_info in enumerate(chunks, start=1):
            yield self._make_chunk_record(record, idx, chunk_info)

    def process(self, input_manifest: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_manifest = os.path.join(output_dir, "b1_manifest.jsonl")
        output_audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(output_audio_dir, exist_ok=True)

        total_sources = 0
        total_chunks = 0

        def _chunk_one(rec: AudioRecord) -> List[AudioRecord]:
            # 必须在 worker 线程内消费 generator：_chunk_record 是含 yield 的
            # 生成器，直接返回只创建对象、函数体不执行，读音频/写盘（最重的
            # I/O）会在主线程消费时才发生，num_workers 并行形同虚设
            return list(self._chunk_record(rec, output_audio_dir))

        def record_generator() -> Iterator[AudioRecord]:
            nonlocal total_sources, total_chunks
            for chunks in _process_in_threads(read_manifest(input_manifest), _chunk_one, self.num_workers):
                total_sources += 1
                for chunk in chunks:
                    total_chunks += 1
                    yield chunk

        count = write_manifest(record_generator(), output_manifest)
        logger.info(
            "B1 分片完成：%d 条源音频 -> %d 条记录（mode=%s, workers=%d），输出 %s",
            total_sources,
            total_chunks,
            self.mode,
            self.num_workers,
            output_manifest,
        )
        return output_manifest
