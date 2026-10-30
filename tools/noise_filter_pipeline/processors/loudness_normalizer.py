"""
D2: 响度归一化与质量检查。

对增强后音频做 RMS 归一化、峰值限制、时长过滤，并写入最终源库。
对应 Hive Stage D 中的后处理环节。

`target_sr` 为可选配置。默认不设置，直接保留输入采样率（当前 pipeline.yaml
中 D1 output_sr 配置为 32kHz）。仅在需要强制输出到某一采样率时才启用。
"""
from __future__ import annotations

import logging
import os
import re
import stat
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

from ..core.manifest_io import (
    AudioRecord,
    read_manifest_with_progress,
)
from ..core.manifest_per_dataset import init_checkpointed_output
from ..core.processor_base import Processor, _process_in_threads
from ..utils.sanitize import sanitize_basename_for_rename
from tools.shared.audio_io import (
    load_audio_robust,
    make_unique_output_path,
    resample_audio,
    save_audio_atomically,
)

logger = logging.getLogger(__name__)

_DATASET_LABEL_NAMING_VERSION = 1
_DATASET_LABEL_SHARD_SIZE = 10000
_DATASET_LABEL_SOURCE_IDENTITY_VERSION = 1
_DATASET_LABEL_SOURCE_IDENTITY_KEY = "dataset_label_source_identity"


def _strict_value_equal(left: Any, right: Any) -> bool:
    """递归执行类型和值均一致的比较，避免 1 与 1.0 混淆源身份。"""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return (
            left.keys() == right.keys()
            and all(_strict_value_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_value_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _dataset_label_category(rec: AudioRecord) -> str:
    """读取 C3 原始 balance_category；缺失或非法时 fail-closed。"""
    c3_result = rec.stages.get("C3") if isinstance(rec.stages, dict) else None
    if not isinstance(c3_result, dict) or c3_result.get("status") != "passed":
        raise ValueError("dataset_label layout requires a passed C3 stage")
    category = c3_result.get("balance_category")
    if not isinstance(category, str) or not category or category != category.strip():
        raise ValueError("dataset_label layout requires C3 balance_category")
    return category


def register_dataset_label_namespace(
    rec: AudioRecord,
    dataset_namespace: Dict[str, str],
    category_namespaces: Dict[str, Dict[str, str]],
) -> None:
    """登记 dataset/category 清洗命名空间，并拒绝非单射映射。"""
    raw_dataset = rec.dataset_id or "unknown"
    dataset_dir = sanitize_basename_for_rename(raw_dataset, is_file=False)
    previous_dataset = dataset_namespace.get(dataset_dir)
    if previous_dataset is not None and previous_dataset != raw_dataset:
        raise ValueError(
            "dataset_label dataset_id 清洗冲突: "
            f"{previous_dataset!r} 与 {raw_dataset!r} -> {dataset_dir!r}"
        )
    dataset_namespace[dataset_dir] = raw_dataset

    raw_category = _dataset_label_category(rec)
    category_dir = sanitize_basename_for_rename(raw_category, is_file=False)
    category_namespace = category_namespaces.setdefault(raw_dataset, {})
    previous_category = category_namespace.get(category_dir)
    if previous_category is not None and previous_category != raw_category:
        raise ValueError(
            "dataset_label C3 balance_category 清洗冲突: "
            f"dataset_id={raw_dataset!r}, {previous_category!r} 与 "
            f"{raw_category!r} -> {category_dir!r}"
        )
    category_namespace[category_dir] = raw_category


def _sr_dir_name(sr: int) -> str:
    """采样率目录名（与 audio_filter 的 sr-first 约定一致）：
    如 ``sr_16k``、``sr_48k``；不足 1kHz 或非整千时保留完整数字。"""
    sr = int(sr)
    if sr >= 1000 and sr % 1000 == 0:
        return f"sr_{sr // 1000}k"
    return f"sr_{sr}"


class LoudnessNormalizerProcessor(Processor):
    """D2: 响度归一化与质检 Processor。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, stage_name="D2")
        self.num_workers = int(config.get("num_workers", 1))
        # num_workers 只是并发度、不改变处理语义；output_label_key 是已废弃的
        # C2 top-1 目录配置。两者均不进入配置指纹，目录类别固定来自 C3。
        # init_checkpointed_output 与 stage_orchestrator marker 均用 self.config。
        self.config = {
            k: v for k, v in config.items()
            if k not in {"num_workers", "output_label_key"}
        }
        shard_size = config.get("dataset_label_shard_size", _DATASET_LABEL_SHARD_SIZE)
        if (
            isinstance(shard_size, bool)
            or not isinstance(shard_size, int)
            or shard_size != _DATASET_LABEL_SHARD_SIZE
        ):
            raise ValueError(
                "D2 dataset_label_shard_size 必须为正式值 10000"
            )
        self.dataset_label_shard_size = shard_size
        self.config["dataset_label_shard_size"] = self.dataset_label_shard_size
        # target_rms: 归一化目标 RMS。0.1 ≈ -20 dBFS RMS，是语音/噪声库常用的
        # 归一化目标电平（留足峰值余量防削波）；无标准强制，可按数据分布调整
        self.target_rms = config.get("target_rms", 0.1)
        self.max_peak = config.get("max_peak", 0.95)
        self.min_duration = config.get("min_duration", 2.0)
        self.max_duration = config.get("max_duration", 30.0)
        for name, value in (
            ("target_rms", self.target_rms),
            ("max_peak", self.max_peak),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"D2 {name} 非法: {value!r}，必须为有限正数")
        for name, value in (
            ("min_duration", self.min_duration),
            ("max_duration", self.max_duration),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"D2 {name} 非法: {value!r}，必须为有限非负数")
        if self.max_duration <= self.min_duration:
            raise ValueError(
                "D2 duration 范围非法：max_duration 必须大于 min_duration"
            )
        # output_sr 未设置时使用 target_sr。
        self.output_sr = config.get("output_sr")
        if self.output_sr is None:
            self.output_sr = config.get("target_sr")
        # 必须严格是正 int；bool/浮点/字符串不能隐式转换后进入重采样链。
        if self.output_sr is not None and (
            type(self.output_sr) is not int or self.output_sr <= 0
        ):
            raise ValueError(
                f"D2 output_sr 非法: {self.output_sr!r}，必须为正整数或 null"
            )
        # output_layout: 输出音频目录布局。
        #   "sr_label_dataset"：audio/sr_16k/<C3 balance category>/<dataset_id>/<shard2>/xxx.flac，
        #       按准入后的噪声类别优先组织（训练侧按类型采样是高频消费模式），
        #       shard2 为文件名 hash 前 2 位分桶（00..ff），避免单目录文件数爆炸；
        #   "sr_dataset_label"：audio/sr_16k/<dataset_id>/<C3 balance category>/xxx.flac，
        #       在 sr_dataset 基础上再加准入类别层级，便于按噪声类别取用；
        #   "dataset_label"：audio/<dataset_id>/<C3 balance category>/shard_<ordinal//10000:05d>/<ordinal%10000:04d>.<format>，
        #       ordinal 按输入 manifest 中每个 dataset 的记录顺序分配；
        #   "sr_dataset"：audio/sr_16k/<dataset_id>/xxx.flac；
        #   "flat"：旧行为，全部平铺到 audio/ 下
        self.output_layout = config.get("output_layout", "sr_dataset")
        if self.output_layout not in (
            "flat", "sr_dataset", "sr_dataset_label", "sr_label_dataset", "dataset_label"
        ):
            # 非法取值若静默回退会让输出落到非预期布局，尽早报错
            raise ValueError(
                f"D2 非法 output_layout: {self.output_layout!r}，"
                "支持 flat / sr_dataset / sr_dataset_label / sr_label_dataset / dataset_label"
            )
        self.output_format = config.get("output_format", "flac")
        if not isinstance(self.output_format, str) or not re.fullmatch(
            r"[A-Za-z0-9]+", self.output_format
        ):
            raise ValueError(
                f"D2 output_format 非法: {self.output_format!r}，"
                "必须为仅含字母和数字的安全扩展名 token"
            )
        # checkpoint / resume：kill 后从断点继续，避免重跑全部音频
        self.enable_checkpoint = bool(config.get("enable_checkpoint", True))
        self.checkpoint_path = config.get("checkpoint_path")
        self.checkpoint_save_interval = int(config.get("checkpoint_save_interval", 100))
        self.manifest_write_batch_size = int(config.get("manifest_write_batch_size", 1000))
        # 基础设施失败计数（status=failed 且未写 checkpoint 的记录数），
        # 供 orchestrator 判定本轮是否有可重试失败（与 C 系列 processor 同语义）。
        # _process_one 在 _process_in_threads 多线程下执行，+= 非原子，
        # 必须加锁——计数丢失会让 has_retryable_failures 误报 False，
        # 含 failed 记录的不完整产物被标记完成供下游消费
        self._retryable_failures = 0
        self._retryable_failures_lock = threading.Lock()

    def has_retryable_failures(self) -> bool:
        """是否有未写 checkpoint、可在下次运行重试的基础设施失败。"""
        return self._retryable_failures > 0

    @staticmethod
    def _is_readable_audio(path: Optional[str]) -> bool:
        """校验产物是 regular file、非空且音频头信息可解析。"""
        if not path:
            return False
        try:
            file_stat = os.lstat(path)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size <= 0:
                return False
            info = sf.info(path)
        except (OSError, RuntimeError, ValueError):
            return False
        return info.frames > 0 and info.samplerate > 0

    @staticmethod
    def _rms(audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio ** 2)))

    @staticmethod
    def _input_fingerprint(path: Optional[str]) -> Optional[Dict[str, Any]]:
        """仅用 regular file 的 lstat 字段构造轻量指纹，不读取文件内容。"""
        if not path:
            return None
        try:
            file_stat = os.lstat(path)
        except OSError:
            return None
        if not stat.S_ISREG(file_stat.st_mode):
            return None
        return {
            "dev": file_stat.st_dev,
            "ino": file_stat.st_ino,
            "mode": stat.S_IFMT(file_stat.st_mode),
            "size": file_stat.st_size,
            "mtime_ns": file_stat.st_mtime_ns,
            "ctime_ns": file_stat.st_ctime_ns,
        }

    def _dataset_label_source_identity(self, rec: AudioRecord) -> Dict[str, Any]:
        """从当前原始输入记录构造不含全音频 hash 的源身份快照。"""
        input_path = rec.audio.get("path")
        return {
            "version": _DATASET_LABEL_SOURCE_IDENTITY_VERSION,
            "source_id": rec.source_id,
            "chunk_id": rec.chunk_id,
            "split": rec.split,
            "audio": {
                "path": input_path,
                "sr": rec.audio.get("sr"),
                "duration": rec.audio.get("duration"),
                "fingerprint": self._input_fingerprint(input_path),
            },
        }

    def _record_input_fingerprint(
        self, rec: AudioRecord, input_path: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """复用 dataset_label 源身份中的 stat，避免 D2 再做一次文件探测。"""
        if self.output_layout == "dataset_label":
            identity = rec.provenance.get(_DATASET_LABEL_SOURCE_IDENTITY_KEY)
            if isinstance(identity, dict):
                audio_identity = identity.get("audio")
                if (
                    isinstance(audio_identity, dict)
                    and audio_identity.get("path") == input_path
                    and isinstance(audio_identity.get("fingerprint"), dict)
                ):
                    return audio_identity["fingerprint"]
        return self._input_fingerprint(input_path)

    def _set_dataset_label_provenance(self, rec: AudioRecord, ordinal: int) -> None:
        """写入 ordinal 与原始源身份；已有字段冲突时立即 fail-fast。"""
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("D2 dataset_label_ordinal 必须为非负整数")
        expected = {
            "dataset_label_naming_version": _DATASET_LABEL_NAMING_VERSION,
            "dataset_label_ordinal": ordinal,
            "dataset_label_shard_size": self.dataset_label_shard_size,
            _DATASET_LABEL_SOURCE_IDENTITY_KEY: self._dataset_label_source_identity(rec),
        }
        provenance = rec.provenance
        if not isinstance(provenance, dict):
            raise ValueError("D2 dataset_label provenance 必须为对象")
        for field, value in expected.items():
            if field in provenance and not _strict_value_equal(provenance[field], value):
                raise ValueError(
                    f"D2 {field} 不匹配: {provenance[field]!r} != {value!r}"
                )
        provenance.update(expected)

    def _has_dataset_label_provenance(self, rec: AudioRecord, ordinal: int) -> bool:
        """严格校验旧 checkpoint 是否含当前版本及当前 dataset ordinal。"""
        provenance = rec.provenance
        if not isinstance(provenance, dict):
            return False
        expected = {
            "dataset_label_naming_version": _DATASET_LABEL_NAMING_VERSION,
            "dataset_label_ordinal": ordinal,
            "dataset_label_shard_size": self.dataset_label_shard_size,
        }
        return all(
            field in provenance
            and type(provenance[field]) is type(value)
            and provenance[field] == value
            for field, value in expected.items()
        )

    def _dataset_label_ordinal(self, rec: AudioRecord) -> int:
        """严格读取 dataset_label provenance 中的当前 ordinal。"""
        provenance = rec.provenance
        if not isinstance(provenance, dict):
            raise ValueError("D2 dataset_label provenance 必须为对象")
        if not self._has_dataset_label_provenance(
            rec, provenance.get("dataset_label_ordinal")
        ):
            raise ValueError("D2 dataset_label provenance 缺少有效版本、ordinal 或 shard_size")
        ordinal = provenance["dataset_label_ordinal"]
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("D2 dataset_label_ordinal 必须为非负整数")
        return ordinal

    def _can_restore_checkpoint_record(
        self,
        rec: AudioRecord,
        prev: AudioRecord,
        ordinal: int,
    ) -> bool:
        """判断 dataset_label checkpoint 是否绑定当前原始输入并可恢复。

        当前记录与旧记录必须完整匹配源身份快照（含轻量文件指纹），不能因为
        Apollo 的中间文件已清理而跳过检查；旧版本或任何身份字段缺失都会重算。
        """
        if self.output_layout != "dataset_label":
            return False
        if rec.record_id != prev.record_id or rec.dataset_id != prev.dataset_id:
            return False
        try:
            if _dataset_label_category(rec) != _dataset_label_category(prev):
                return False
        except ValueError:
            return False
        if not self._has_dataset_label_provenance(rec, ordinal):
            return False
        if not self._has_dataset_label_provenance(prev, ordinal):
            return False
        current_identity = rec.provenance.get(_DATASET_LABEL_SOURCE_IDENTITY_KEY)
        previous_identity = prev.provenance.get(_DATASET_LABEL_SOURCE_IDENTITY_KEY)
        if (
            not isinstance(current_identity, dict)
            or not isinstance(previous_identity, dict)
            or not _strict_value_equal(current_identity, previous_identity)
        ):
            return False
        for identity in (current_identity, previous_identity):
            audio_identity = identity.get("audio")
            if (
                not isinstance(audio_identity, dict)
                or not isinstance(audio_identity.get("fingerprint"), dict)
            ):
                # None 指纹无法证明 checkpoint 绑定了稳定的 regular file，
                # 例如 symlink/特殊文件/缺失输入，必须重新走处理路径。
                return False

        d2_stage = prev.stages.get("D2", {})
        if not isinstance(d2_stage, dict):
            return False
        output_sr = self.output_sr or rec.audio.get("sr")
        if not isinstance(output_sr, int) or isinstance(output_sr, bool):
            return False
        expected_path = self._output_path_for(
            output_sr, rec, rec.audio.get("path"), ordinal
        )
        if prev.audio.get("path") != expected_path:
            return False
        if d2_stage.get("status") == "rejected":
            return True
        return self._is_readable_audio(prev.audio.get("path"))

    def _can_restore_non_dataset_checkpoint_record(self, prev: AudioRecord) -> bool:
        """保持非 dataset_label 布局的既有 checkpoint 恢复语义。"""
        d2_stage = prev.stages.get("D2", {})
        input_fingerprint = d2_stage.get("input_fingerprint")
        src_path = d2_stage.get("input_path") or prev.audio.get("path")
        current_fingerprint = self._input_fingerprint(src_path)
        if input_fingerprint is None:
            return False
        if current_fingerprint is None or input_fingerprint != current_fingerprint:
            return False
        if d2_stage.get("status") == "rejected":
            return True
        return self._is_readable_audio(prev.audio.get("path"))

    def _normalize(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int, Dict[str, Any]]:
        info: Dict[str, Any] = {
            "original_rms": self._rms(audio),
            "original_peak": float(np.max(np.abs(audio))),
            "original_sr": sr,
        }

        if not np.isfinite(audio).all():
            # NaN/Inf 会让 rms 计算得到 NaN 并绕过 rms<=0 检查，
            # 必须显式拦截，避免把 NaN 音频写盘
            info["rejected_reason"] = "nan_or_inf"
            return audio, sr, info

        if self.output_sr and sr != self.output_sr:
            # 使用 resample_audio，内部优先 SoX（rate -v -I），SoX 不可用时 fallback 到 soxr/librosa/scipy
            audio = resample_audio(audio, sr, self.output_sr)
            sr = self.output_sr

        duration = audio.shape[0] / sr
        info["duration"] = duration

        if duration < self.min_duration or duration > self.max_duration:
            info["rejected_reason"] = "duration_out_of_range"
            return audio, sr, info

        rms = self._rms(audio)
        if rms <= 0:
            # 全零音频无有效内容，直接拒绝。
            info["rejected_reason"] = "zero_rms"
            return audio, sr, info
        audio = audio * (self.target_rms / rms)

        peak = np.max(np.abs(audio))
        if peak > self.max_peak:
            audio = audio * (self.max_peak / peak)

        info["normalized_rms"] = self._rms(audio)
        info["normalized_peak"] = float(np.max(np.abs(audio)))
        return audio, sr, info

    def _label_dir_name(self, rec: AudioRecord) -> str:
        """返回 C3 选定的平衡类别目录名；缺失或非法时 fail-closed。"""
        return sanitize_basename_for_rename(
            _dataset_label_category(rec), is_file=False
        )

    def _prepare_output_audio(self, output_dir: str) -> None:
        """初始化最终音频根目录。"""
        self._output_audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(self._output_audio_dir, exist_ok=True)

    def _output_dir_for(self, out_sr: int, rec: AudioRecord) -> str:
        """按 output_layout 计算输出音频目录。"""
        if self.output_layout == "flat":
            return self._output_audio_dir
        dataset_dir = sanitize_basename_for_rename(rec.dataset_id or "unknown", is_file=False)
        if self.output_layout == "dataset_label":
            return os.path.join(self._output_audio_dir, dataset_dir, self._label_dir_name(rec))

        parts = [self._output_audio_dir, _sr_dir_name(out_sr)]
        if self.output_layout == "sr_label_dataset":
            # label 优先于 dataset：训练侧按噪声类型跨数据集采样是高频消费模式；
            # shard2 分桶在 _process_one 的 make_unique_output_path(shard=True) 完成
            parts.append(self._label_dir_name(rec))
            parts.append(dataset_dir)
        else:
            parts.append(dataset_dir)
            if self.output_layout == "sr_dataset_label":
                parts.append(self._label_dir_name(rec))
        return os.path.join(*parts)

    def _output_path_for(
        self,
        out_sr: int,
        rec: AudioRecord,
        input_path: str,
        ordinal: Optional[int] = None,
    ) -> str:
        """返回最终音频路径；dataset_label 严格使用 provenance ordinal。"""
        output_dir = self._output_dir_for(out_sr, rec)
        if self.output_layout == "dataset_label":
            if ordinal is None:
                ordinal = self._dataset_label_ordinal(rec)
            if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
                raise ValueError("D2 dataset_label_ordinal 必须为非负整数")
            shard = ordinal // self.dataset_label_shard_size
            slot = ordinal % self.dataset_label_shard_size
            return os.path.join(
                output_dir,
                f"shard_{shard:05d}",
                f"{slot:04d}.{self.output_format}",
            )
        return make_unique_output_path(
            output_dir,
            input_path,
            ext=f".{self.output_format}",
            shard=self.output_layout == "sr_label_dataset",
        )

    def _process_one(self, rec: AudioRecord) -> AudioRecord:
        if self.should_skip(rec):
            return rec

        input_path = rec.audio["path"]
        ordinal = self._dataset_label_ordinal(rec) if self.output_layout == "dataset_label" else None
        input_fingerprint = self._record_input_fingerprint(rec, input_path)
        if input_fingerprint is None:
            if self.output_layout == "dataset_label":
                rec.audio["path"] = self._output_path_for(
                    rec.audio.get("sr") or 0, rec, input_path, ordinal
                )
            reason = (
                "missing_audio"
                if not input_path or not os.path.isfile(input_path)
                else "input_fingerprint_unavailable"
            )
            rec.mark_stage(
                "D2", "rejected",
                extra={
                    "reason": reason,
                    "input_path": input_path,
                    "input_fingerprint": input_fingerprint,
                },
            )
            rec.rejected_reason = f"d2_{reason}"
            return rec

        try:
            # 与 D1+D2 合并模式口径一致：max_peak=1.2，避免 MP3 解码正常越界被误杀
            audio, sr = load_audio_robust(input_path, max_peak=1.2)
            # 读取期间若源文件发生替换或修改，放弃本次结果并交给 resume 重试。
            if self._input_fingerprint(input_path) != input_fingerprint:
                raise OSError(f"input changed while loading: {input_path}")
            audio, out_sr, info = self._normalize(audio, sr)

            if "rejected_reason" in info:
                if self.output_layout == "dataset_label":
                    rec.audio["path"] = self._output_path_for(
                        out_sr, rec, input_path, ordinal
                    )
                rec.mark_stage(
                    "D2", "rejected",
                    extra={
                        **info,
                        "input_path": input_path,
                        "input_fingerprint": input_fingerprint,
                    },
                )
                rec.rejected_reason = f"d2_{info['rejected_reason']}"
                return rec

            output_path = self._output_path_for(
                out_sr, rec, input_path, ordinal if ordinal is not None else 0
            )
            save_audio_atomically(audio, out_sr, output_path)

            rec.audio["path"] = output_path
            rec.audio["sr"] = out_sr
            rec.audio["duration"] = info["duration"]
            rec.audio["rms"] = info["normalized_rms"]
            rec.audio["peak"] = info["normalized_peak"]
            # 输入指纹随 passed 记录落盘：resume 恢复时比对输入未变才复用，
            # 上游重算（路径不变、内容变化）时自动失效重处理
            rec.mark_stage(
                "D2", "passed",
                extra={**info, "input_path": input_path, "input_fingerprint": input_fingerprint},
            )
        except OSError as e:
            # 基础设施失败（磁盘满/挂载掉线/音频后端 I/O 等）：标 failed
            # （非终态 rejected），不写 checkpoint，resume 时自动重试
            logger.warning("D2 处理 %s 失败（基础设施，可重试）: %s", input_path, e)
            rec.mark_stage("D2", "failed", extra={"reason": str(e), "retryable": True})
            with self._retryable_failures_lock:
                self._retryable_failures += 1
        except Exception as e:
            # 内容级失败（解码异常/数据非法等）：内容级终态，重试无意义
            logger.warning("D2 处理 %s 失败: %s", input_path, e)
            if self.output_layout == "dataset_label":
                rec.audio["path"] = self._output_path_for(
                    rec.audio.get("sr") or 0, rec, input_path, ordinal
                )
            rec.mark_stage(
                "D2", "rejected",
                extra={
                    "reason": str(e),
                    "input_path": input_path,
                    "input_fingerprint": input_fingerprint,
                },
            )
            rec.rejected_reason = "d2_exception"
        return rec

    def process(
        self,
        input_manifest: str,
        output_dir: str,
        fingerprint_input: Optional[str] = None,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_manifest = os.path.join(output_dir, "d2_manifest.jsonl")
        self._prepare_output_audio(output_dir)

        manifest_writer, ckpt, prev_records = init_checkpointed_output(
            output_manifest, output_dir, self.config, input_manifest,
            enable_checkpoint=self.enable_checkpoint,
            checkpoint_path=self.checkpoint_path,
            save_interval=self.checkpoint_save_interval,
            manifest_batch_size=self.manifest_write_batch_size,
            stage_label="Stage D2",
            fingerprint_input=fingerprint_input,
        )

        # 流式分片处理：避免把整个 manifest 全量读成 list。
        # 每批读取 batch_size 条，用线程池并行处理后追加写出。
        stream_batch_size = max(64, self.num_workers * 4)
        count = 0
        seen_record_ids: set[str] = set()
        dataset_ordinals: Dict[str, int] = {}
        dataset_namespaces: Dict[str, str] = {}
        category_namespaces: Dict[str, Dict[str, str]] = {}

        def _flush_batch(batch: List[AudioRecord]) -> None:
            nonlocal count
            processed = list(_process_in_threads(batch, self._process_one, self.num_workers))
            count += len(processed)
            manifest_writer.extend(processed)
            # 内容级结论（passed/rejected）都写 checkpoint：D2 内容级判定是终态，
            # resume 重跑不会改变结论；基础设施失败（status=failed）不写，
            # resume 时自动重试（与 C 系列 infra-failed 可重试语义一致）
            ckpt.mark_done_batch([
                r.record_id for r in processed
                if r.stages.get("D2", {}).get("status") != "failed"
            ])

        try:
            with manifest_writer:
                batch: List[AudioRecord] = []
                for rec in read_manifest_with_progress(input_manifest, self.stage_name):
                    if rec.record_id in seen_record_ids:
                        raise ValueError(
                            f"D2 duplicate record_id: {rec.record_id!r}"
                        )
                    seen_record_ids.add(rec.record_id)
                    ordinal = None
                    if self.output_layout == "dataset_label":
                        register_dataset_label_namespace(
                            rec, dataset_namespaces, category_namespaces
                        )
                        dataset_key = rec.dataset_id or "unknown"
                        ordinal = dataset_ordinals.get(dataset_key, 0)
                        dataset_ordinals[dataset_key] = ordinal + 1
                        # 主线程在 checkpoint 判断和线程池提交前分配 ordinal，
                        # rejected/failed 也因此消耗位置并可留下空洞。
                        self._set_dataset_label_provenance(rec, ordinal)
                    if ckpt.is_done(rec.record_id):
                        prev = prev_records.get(rec.record_id)
                        if self.output_layout == "dataset_label":
                            if (
                                prev is not None
                                and ordinal is not None
                                and self._can_restore_checkpoint_record(rec, prev, ordinal)
                            ):
                                manifest_writer.append(prev)
                                count += 1
                                continue
                        elif prev is not None and self._can_restore_non_dataset_checkpoint_record(prev):
                            manifest_writer.append(prev)
                            count += 1
                            continue
                        # 上次输出缺失、旧 checkpoint 缺少顺序元数据或 ordinal 不匹配：重处理
                    batch.append(rec)
                    if len(batch) >= stream_batch_size:
                        _flush_batch(batch)
                        batch = []
                if batch:
                    _flush_batch(batch)
                ckpt.save(force=True)
        finally:
            self.close()

        logger.info(
            "D2 处理完成，共 %d 条（workers=%d，checkpoint=%s），输出 %s",
            count,
            self.num_workers,
            ckpt.checkpoint_path,
            output_manifest,
        )
        if self.has_retryable_failures():
            logger.warning(
                "%s 有 %d 条基础设施失败记录（status=failed，未写 checkpoint），重跑可自动重试。",
                self.stage_name, self._retryable_failures,
            )
        return output_dir
