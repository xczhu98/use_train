"""
B3: VAD Processor。

对通过静音过滤的音频批量运行 VAD（单模型或多模型投票），
将语音占比、fail_count、vad_category 写入 AudioRecord.stages["B3"]，
不修改音频文件本身。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.exceptions import InfraAbortError
from ..core.manifest_io import AudioRecord, append_manifest, read_manifest, write_manifest
from ..core.processor_base import Processor
from .vad import create_vad_detector, MultiVADDetector
from .vad.multi import _ENERGY_VAD_TYPES

logger = logging.getLogger(__name__)

# MultiVADDetector 支持的 VAD 类型全集（与 processors/vad/multi.py 的
# supported_vad_types 保持一致）。在 __init__ 提前校验，配置错误 fail-fast，
# 避免被预加载的容错 except 吞掉后整批静默降级为 vad_batch_failed。
_SUPPORTED_VAD_TYPES = frozenset(
    {"firered", "silero", "webrtc", "funasr", "nemo_onnx", "nemo", "energy"}
)


def _sample_record_ids(records: List[AudioRecord], limit: int = 5) -> List[str]:
    """取记录 ID 样例，用于日志定位异常 batch。"""
    return [rec.record_id for rec in records[:limit]]


class VADProcessor(Processor):
    """B3: VAD 检测 Processor。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, stage_name="B3")
        self.enable = config.get("enable", True)
        self.use_multi_vad = bool(config.get("use_multi_vad", True))
        self.vad_types = config.get("vad_types", ["silero", "funasr", "firered"])
        # vad_types 必须为非空 list：字符串会被 set() 按字符拆散（"silero" -> {s,i,l,e,r,o}）
        # 导致全部判为未知类型；空列表/None 则会让 MultiVADDetector 零 detector
        # 运行——fail_count 恒为 0，所有记录被静默判 noise 放行。归一化后立即
        # 校验，配置错误 fail-fast
        if isinstance(self.vad_types, str):
            self.vad_types = [self.vad_types]
        if not isinstance(self.vad_types, list) or not self.vad_types:
            raise ValueError(
                f"vad_types 必须为非空列表，当前值: {self.vad_types!r}"
            )
        unknown_vad_types = set(self.vad_types) - _SUPPORTED_VAD_TYPES
        if unknown_vad_types:
            raise ValueError(
                f"B3 不支持的 VAD 类型: {sorted(unknown_vad_types)}，"
                f"支持: {sorted(_SUPPORTED_VAD_TYPES)}"
            )
        self.vad_type = config.get("vad_type", "silero")
        # 各 VAD 模型目录独立配置；缺省时回退到旧的统一键 vad_model_dir
        # （三个 VAD 的目录结构互不兼容，共用一个键会互相喂错路径）
        self.vad_model_dir = config.get("vad_model_dir")
        self.firered_model_dir = config.get("firered_model_dir") or self.vad_model_dir
        self.funasr_model_dir = config.get("funasr_model_dir") or self.vad_model_dir
        self.nemo_onnx_path = config.get("nemo_onnx_path")
        self.use_gpu = bool(config.get("use_gpu", False))
        self.speech_threshold = float(config.get("speech_threshold", 0.4))
        self.vad_speech_ratio_threshold = float(config.get("vad_speech_ratio_threshold", 0.05))
        self.batch_size = int(config.get("batch_size", 32))
        raw_silero_group_size = config.get("silero_group_size", 16)
        if (
            isinstance(raw_silero_group_size, bool)
            or not isinstance(raw_silero_group_size, int)
            or raw_silero_group_size <= 0
        ):
            raise ValueError(
                "silero_group_size 必须为正整数，"
                f"当前值: {raw_silero_group_size!r}"
            )
        self.silero_group_size = raw_silero_group_size
        self.strict_vad_models = bool(config.get("strict_vad_models", False))
        self.reject_on_any_speech = bool(config.get("reject_on_any_speech", False))
        # VAD 批量检测的进程池大小；>1 时子进程并行 FunASR 帧级循环
        self.num_workers = max(1, int(config.get("num_workers", 1)))
        # worker 数超过 CPU 核数没有意义：每个 worker 都会跑满一个核做帧级
        # 循环，超限只会徒增模型加载内存与 CUDA context，此处钳制并告警
        cpu_count = os.cpu_count() or 1
        if self.num_workers > cpu_count:
            logger.warning(
                "B3 VAD num_workers=%d 超过 CPU 核数 %d，钳制为 %d",
                self.num_workers, cpu_count, cpu_count,
            )
            self.num_workers = cpu_count
        # VAD 内部临时文件目录（MultiVAD 共享临时 wav），默认系统临时目录
        self.temp_dir = config.get("temp_dir")
        # 是否在 manifest 的 stages.B3.vad_result 中保留每帧概率（frame_probs）。
        # frame_probs 占单条记录约 95% 体积（3 模型 × 10s 帧级浮点序列，~38KB/条），
        # 下游唯一消费方是 B4 用不同 speech_threshold 重算 speech_ratio——配置已要求
        # b3/b4 阈值一致（重算结果与存储值相同），默认 False 剥离；
        # 需要离线用新阈值重算 B4 时再开。B4 重算会回退到存储的 speech_ratio。
        self.keep_frame_probs = bool(config.get("keep_frame_probs", False))

        # 基础设施失败计数（VAD 整批失败等未写 checkpoint 的记录数），
        # 供 stage_b_curation 判定 staged 链路（unified: false）本轮是否有
        # 可重试失败并中止 stage；对齐 StageBUnifiedProcessor 的语义，
        # flush/批处理路径单线程自增，无并发竞争
        self._retryable_failures = 0

        self._detector: Optional[Any] = None
        if self.enable:
            # 主线程预加载模型，避免第一个 batch 时触发懒加载造成延迟；
            # 配置类错误（ValueError，含 vad_types 非法值）已在上方 fail-fast，
            # 此处只降级模型加载类异常（缺模型/环境问题），推理时再重试
            try:
                self._get_detector()
            except ValueError:
                raise
            except Exception as e:
                logger.warning("B3 VAD 模型预加载失败，将在推理时重试: %s", e)

    def _get_detector(self) -> Any:
        if self._detector is None:
            if self.use_multi_vad:
                self._detector = MultiVADDetector(
                    vad_types=self.vad_types,
                    firered_model_dir=self.firered_model_dir,
                    funasr_model_dir=self.funasr_model_dir,
                    nemo_model_path=self.vad_model_dir,
                    nemo_onnx_path=self.nemo_onnx_path,
                    use_gpu=self.use_gpu,
                    speech_threshold=self.speech_threshold,
                    speech_ratio_threshold=self.vad_speech_ratio_threshold,
                    strict_vad_models=self.strict_vad_models,
                    reject_on_any_speech=self.reject_on_any_speech,
                    temp_dir=self.temp_dir,
                    num_workers=self.num_workers,
                    silero_group_size=self.silero_group_size,
                    keep_frame_probs=self.keep_frame_probs,
                )
            else:
                # 单 VAD 模式按类型选对应目录，避免喂错结构
                single_model_dir = {
                    "firered": self.firered_model_dir,
                    "funasr": self.funasr_model_dir,
                }.get(self.vad_type, self.vad_model_dir)
                self._detector = create_vad_detector(
                    self.vad_type,
                    model_dir=single_model_dir,
                    onnx_path=self.nemo_onnx_path,
                    use_gpu=self.use_gpu,
                    threshold=self.speech_threshold,
                    temp_dir=self.temp_dir,
                )
        return self._detector

    def close(self) -> None:
        """释放 VAD 资源（进程池等）。长任务结束或异常退出前应调用。"""
        if self._detector is not None and hasattr(self._detector, "shutdown"):
            try:
                self._detector.shutdown()
            except Exception as e:  # noqa: BLE001
                logger.warning("B3 VAD detector shutdown 失败: %s", e)

    # 兼容别名：历史上对外暴露的是 shutdown()
    def shutdown(self) -> None:
        self.close()

    def has_retryable_failures(self) -> bool:
        """是否有未写 checkpoint、可在下次运行重试的基础设施失败。

        与 StageBUnifiedProcessor / C/D 的 has_retryable_failures 语义一致，
        供 stage_b_curation 在 staged 链路检测后中止 stage，避免"VAD 整批
        失败但 stage 照常成功"。
        """
        return self._retryable_failures > 0

    def _categorize_from_vad(self, vad_result: Any) -> str:
        """从 VAD 结果推导三档类别。"""
        if hasattr(vad_result, "final_category"):
            return str(vad_result.final_category)
        speech_ratio = getattr(vad_result, "speech_ratio", 0.0)
        return "rejected" if speech_ratio > self.vad_speech_ratio_threshold else "noise"

    def _detect_batch(self, paths: List[str]) -> List[Any]:
        # 注意：file-mode（staged 链路 B3）下 num_workers 不生效——MultiVAD
        # 的进程池并行只作用于数组路径（detect_batch_arrays，unified 链路
        # 使用）；文件路径模式在父进程串行 detect。需要 VAD 并行请使用
        # unified 链路或增大 vad_batch_chunks 以提高批效率。
        detector = self._get_detector()
        if hasattr(detector, "detect_batch"):
            return detector.detect_batch(paths)
        return [detector.detect(p) for p in paths]

    def _detect_batch_arrays(
        self,
        audio_arrays: List[Tuple[np.ndarray, int]],
    ) -> List[Any]:
        """对内存中的音频数组批量执行 VAD。"""
        detector = self._get_detector()
        if hasattr(detector, "detect_batch_arrays"):
            return detector.detect_batch_arrays(audio_arrays)
        if hasattr(detector, "detect_array"):
            return [detector.detect_array(audio, sr) for audio, sr in audio_arrays]
        raise RuntimeError("当前 VAD detector 不支持数组输入")

    def _normalize_vad_results(
        self,
        records: List[AudioRecord],
        vad_results: List[Any],
        context: str,
    ) -> List[Any]:
        """防御性校验：VAD 结果数必须与输入记录数一致。

        不一致时补齐 ``None``，让上层按 ``vad_failed`` 处理，避免索引越界
        或静默吞掉记录。
        """
        if len(vad_results) == len(records):
            return vad_results
        logger.error(
            "B3 VAD 结果数不匹配 [%s]: records=%d, vad_results=%d, sample_ids=%s",
            context,
            len(records),
            len(vad_results),
            _sample_record_ids(records),
        )
        if len(vad_results) > len(records):
            return vad_results[: len(records)]
        return vad_results + [None] * (len(records) - len(vad_results))

    def _vad_result_extra(self, vad_result: Any) -> Dict[str, Any]:
        """序列化 vad_result 供 stages.B3.extra 落盘；keep_frame_probs=False 时
        剥离 frame_probs（占记录体积 ~95%，见 __init__ 配置注释）。"""
        d = getattr(vad_result, "get_result_dict", lambda: {})()
        if self.keep_frame_probs or not isinstance(d, dict):
            return d
        d = dict(d)
        ind_list = d.get("individual_results")
        if isinstance(ind_list, list):
            d["individual_results"] = [
                {k: v for k, v in ind.items() if k != "frame_probs"}
                if isinstance(ind, dict) else ind
                for ind in ind_list
            ]
        d.pop("frame_probs", None)
        return d

    def _apply_vad_results(
        self,
        pending: List[AudioRecord],
        vad_results: List[Any],
        output_manifest: Optional[str],
        passed: List[int],
        rejected: List[int],
    ) -> None:
        """将 VAD 结果应用到 pending 记录并写出。

        ``output_manifest`` 为 None 时只做内存标记（供 Stage B unified 等
        调用方自行写出），不追加 manifest。

        ``vad_result`` 缺失（None）表示该记录 VAD 失败。严格模式下拒绝，
        非严格模式下标记为 suspected；注意 B4 的 prev_stage 门控会把所有
        B3 未 passed 记录（含 suspected/rejected/skipped）一律标 skipped
        ——B4 的 keep_rejected/keep_suspected 仅对 B3 passed 记录生效，
        B3 失败/未判定记录不会经 B4 策略放行到下游。本方法也不会直接
        passed，避免损坏/异常数据被误放行。
        """
        result_idx = 0
        for rec in pending:
            if result_idx < len(vad_results):
                vad_result = vad_results[result_idx]
                result_idx += 1
            else:
                vad_result = None

            if vad_result is None:
                # strict：直接拒绝；非 strict：标为 suspected（B4 门控会将其
                # skipped——keep_rejected/keep_suspected 仅对 B3 passed 记录
                # 生效，suspected 不会被策略放行）。
                status = "rejected" if self.strict_vad_models else "suspected"
                rec.mark_stage(
                    self.stage_name,
                    status,
                    extra={"reason": "vad_failed"},
                )
                if status == "rejected":
                    rec.rejected_reason = "b3_vad_failed"
                    rejected[0] += 1
                # suspected 不计入 passed，也不计入 rejected，由 B4 策略处理
                if output_manifest is not None:
                    append_manifest(iter([rec]), output_manifest)
                continue

            # strict 模式：单 VAD 退化为 energy fallback（模型缺失/加载失败）
            # 时与 MultiVAD 口径对齐（任一指定 VAD 退化 fallback 即 rejected），
            # 直接判 rejected，不得按正常结果放行
            if (
                self.strict_vad_models
                and str(getattr(vad_result, "vad_type", "")).lower() in _ENERGY_VAD_TYPES
            ):
                rec.mark_stage(
                    self.stage_name,
                    "rejected",
                    extra={
                        "reason": "vad_model_fallback",
                        "vad_result": self._vad_result_extra(vad_result),
                    },
                )
                rec.rejected_reason = "b3_vad_model_fallback"
                rejected[0] += 1
                if output_manifest is not None:
                    append_manifest(iter([rec]), output_manifest)
                continue

            vad_category = self._categorize_from_vad(vad_result)
            rec.mark_stage(
                self.stage_name,
                "passed",
                extra={
                    "vad_category": vad_category,
                    "vad_result": self._vad_result_extra(vad_result),
                },
            )
            passed[0] += 1
            if output_manifest is not None:
                append_manifest(iter([rec]), output_manifest)

    def _flush_pending(
        self,
        pending: List[AudioRecord],
        pending_paths: List[str],
        output_manifest: str,
        passed: List[int],
        rejected: List[int],
    ) -> None:
        """对累积的 micro-batch 执行 VAD 并逐条写出结果（文件路径模式）。"""
        if not pending:
            return

        valid_paths = [p for p in pending_paths if p and os.path.isfile(p)]
        vad_results: List[Any] = []
        if valid_paths:
            try:
                vad_results = self._detect_batch(valid_paths)
            except InfraAbortError:
                # 基础设施中止信号必须透传，不得降级为逐批失败
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "B3 VAD 批量检测失败 [file-mode]: batch_size=%d, sample_ids=%s, error=%s(%s)",
                    len(pending),
                    _sample_record_ids(pending),
                    type(exc).__name__,
                    exc,
                )
                # 整批失败属于基础设施故障（与 strict 无关）：统一标记
                # vad_batch_failed，与 array-mode 口径一致——使 checkpoint
                # 排除逻辑覆盖到它（不写 checkpoint，下次运行可重试），
                # 而不是在非 strict 模式标 suspected 放行后被 B4 拒绝并永久沉淀。
                # 归因与正常路径对齐：仅对文件存在的记录标记 vad_batch_failed
                # 并计入 _retryable_failures；文件缺失属内容级问题（VAD 尚未
                # 接触这些记录），标 missing_audio（rejected），不占可重试名额。
                for rec, path in zip(pending, pending_paths):
                    if not path or not os.path.isfile(path):
                        rec.mark_stage(
                            self.stage_name, "rejected", extra={"reason": "missing_audio"}
                        )
                        rec.rejected_reason = "b3_missing_audio"
                    else:
                        rec.mark_stage(
                            self.stage_name,
                            "rejected",
                            extra={"reason": "vad_batch_failed", "error": str(exc)},
                        )
                        rec.rejected_reason = "b3_vad_batch_failed"
                        self._retryable_failures += 1
                    rejected[0] += 1
                    append_manifest(iter([rec]), output_manifest)
                pending.clear()
                pending_paths.clear()
                return

        valid_pending: List[AudioRecord] = []
        for rec, path in zip(pending, pending_paths):
            if not path or not os.path.isfile(path):
                rec.mark_stage(self.stage_name, "rejected", extra={"reason": "missing_audio"})
                rec.rejected_reason = "b3_missing_audio"
                rejected[0] += 1
                append_manifest(iter([rec]), output_manifest)
            else:
                valid_pending.append(rec)

        vad_results = self._normalize_vad_results(valid_pending, vad_results, "file-mode")
        self._apply_vad_results(valid_pending, vad_results, output_manifest, passed, rejected)

        pending.clear()
        pending_paths.clear()

    def process_records_with_arrays(
        self,
        records: List[AudioRecord],
        audio_arrays: List[Tuple[np.ndarray, int]],
    ) -> List[AudioRecord]:
        """直接对内存中的音频数组执行 VAD，返回处理后的记录。

        用于 Stage B unified 等希望避免临时 chunk 落盘的场景。
        """
        if not records:
            return []
        if not self.enable:
            for rec in records:
                rec.mark_stage(self.stage_name, "passed", extra={"enabled": False})
            return records

        vad_results: List[Any] = []
        if audio_arrays:
            try:
                vad_results = self._detect_batch_arrays(audio_arrays)
            except InfraAbortError:
                # 基础设施中止信号必须透传，不得降级为逐批失败
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "B3 VAD 数组批量检测失败 [array-mode]: batch_size=%d, sample_ids=%s, error=%s(%s)",
                    len(records),
                    _sample_record_ids(records),
                    type(exc).__name__,
                    exc,
                )
                # 整批失败属于基础设施故障（与 strict 无关）：统一标记
                # vad_batch_failed，使 Stage B unified 的 checkpoint 排除逻辑
                # 覆盖到它（不写 checkpoint，下次运行可重试），而不是在非 strict
                # 模式标 suspected/vad_failed 后被 B4 拒绝并永久 checkpoint。
                for rec in records:
                    rec.mark_stage(
                        self.stage_name,
                        "rejected",
                        extra={"reason": "vad_batch_failed", "error": str(exc)},
                    )
                    rec.rejected_reason = "b3_vad_batch_failed"
                self._retryable_failures += len(records)
                return records

        vad_results = self._normalize_vad_results(records, vad_results, "array-mode")
        passed = [0]
        rejected = [0]
        self._apply_vad_results(records, vad_results, None, passed, rejected)
        return records

    def process(self, input_manifest: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_manifest = os.path.join(output_dir, "b3_manifest.jsonl")
        write_manifest(iter([]), output_manifest)

        total = 0
        passed = [0]
        rejected = [0]

        pending: List[AudioRecord] = []
        pending_paths: List[str] = []

        for rec in read_manifest(input_manifest):
            total += 1
            if self.should_skip(rec):
                append_manifest(iter([rec]), output_manifest)
                continue
            # prev_stage 门控（与 stage_orchestrator._make_stage_pass_checker
            # 口径对齐）：B2 未 passed（静音过滤 rejected 等）的记录不得再跑
            # VAD——其 chunk 音频仍在盘上，VAD 判 noise 后会被 B4 放行"复活"。
            # 无 B2 键的旧记录视为可处理，保持兼容。
            b2 = rec.stages.get("B2")
            if b2 is not None and b2.get("status") != "passed":
                rec.mark_stage(
                    self.stage_name,
                    "skipped",
                    extra={"reason": "prev_stage_B2_not_passed"},
                )
                append_manifest(iter([rec]), output_manifest)
                continue
            if not self.enable:
                rec.mark_stage(self.stage_name, "passed", extra={"enabled": False})
                passed[0] += 1
                append_manifest(iter([rec]), output_manifest)
                continue

            pending.append(rec)
            pending_paths.append(rec.audio.get("path"))
            if len(pending) >= self.batch_size:
                self._flush_pending(pending, pending_paths, output_manifest, passed, rejected)

        self._flush_pending(pending, pending_paths, output_manifest, passed, rejected)

        logger.info(
            "B3 VAD 检测完成：%d 条记录，通过 %d，拒绝 %d，输出 %s",
            total,
            passed[0],
            rejected[0],
            output_manifest,
        )
        return output_manifest
