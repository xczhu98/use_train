"""多 VAD 共识检测与 spawn 进程池 worker（从 processors/vad.py 拆出）。

注意：本文件位于 processors/vad/ 下，比原 processors/vad.py 深一层，
对 core 包的相对 import 相应多一级（``from ...core...``）。
"""
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import soundfile as sf

from ...core.exceptions import InfraAbortError
from .bootstrap import _suppress_stdout_stderr
from .wrappers import (
    FallbackVAD,
    FireRedVADWrapper,
    FunASRVADWrapper,
    NeMoONNXVADWrapper,
    NeMoVADWrapper,
    SileroVADWrapper,
    VADResult,
    WebRTCVADWrapper,
)

logger = logging.getLogger(__name__)


# 使用能量阈值的 fallback VAD 类型标识
_ENERGY_VAD_TYPES = {"energy", "fallback"}


def _write_shared_wav(audio: np.ndarray, temp_dir: Optional[str]) -> str:
    """创建共享临时 WAV；创建、关闭或写入失败均视为基础设施错误。"""
    import tempfile

    fd: Optional[int] = None
    path: Optional[str] = None
    try:
        fd, path = tempfile.mkstemp(
            suffix=".wav", prefix="vad_shared_", dir=temp_dir
        )
        os.close(fd)
        fd = None
        sf.write(path, audio.astype(np.float32, copy=False), 16000)
        return path
    except Exception as exc:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass
        raise InfraAbortError(f"共享 VAD 临时 WAV 写入失败: {exc}") from exc


class MultiVADResult:
    """多 VAD 结果：仅用 fail_count 三档（与 06_statistics_and_copy 对齐）。"""
    def __init__(
        self,
        results: List[VADResult],
        vad_types: List[str] = None,
        speech_ratio_threshold: float = 0.05,
        strict_vad_models: bool = False,
        reject_on_any_speech: bool = False,
        failed_vad_types: Optional[List[str]] = None,
    ):
        self.results = results
        self.vad_types = vad_types or [getattr(r, 'vad_type', f'vad_{i}') for i, r in enumerate(results)]
        self.speech_ratio_threshold = float(speech_ratio_threshold)
        self.strict_vad_models = bool(strict_vad_models)
        self.reject_on_any_speech = bool(reject_on_any_speech)
        # 推理中抛异常被跳过的指定 VAD；strict 模式下与 energy fallback 同等视为失败
        self.failed_vad_types = list(failed_vad_types or [])
        self._compute_consensus()

    def _is_real_model(self, result: VADResult) -> bool:
        """判断该 VAD 结果是否来自真实模型（非 energy fallback）。"""
        return getattr(result, 'vad_type', 'unknown').lower() not in _ENERGY_VAD_TYPES

    def _compute_consensus(self):
        speech_ratios = [r.speech_ratio for r in self.results]
        thresh = self.speech_ratio_threshold
        # fail_count = 满足「speech_ratio > threshold」的 VAD 个数（与 06 一致）
        self.fail_count = sum(1 for r in speech_ratios if r > thresh)
        self.total_vads = len(self.results)
        self.avg_speech_ratio = np.mean(speech_ratios) if speech_ratios else 0.0
        self.max_speech_ratio = max(speech_ratios) if speech_ratios else 0.0
        self.min_speech_ratio = min(speech_ratios) if speech_ratios else 0.0

        # strict 模式下：任一指定 VAD 退化为 energy fallback 或推理抛异常被跳过，
        # 都视为「模型失败」，直接判 rejected（对齐配置注释「任一指定 VAD 模型
        # 缺失/失败即拒绝」）
        if self.strict_vad_models and (self.results or self.failed_vad_types):
            self.real_model_count = sum(1 for r in self.results if self._is_real_model(r))
            self.has_fallback = (
                self.real_model_count < len(self.results) or bool(self.failed_vad_types)
            )
        else:
            self.real_model_count = len(self.results)
            self.has_fallback = False

        # reject_on_any_speech：只要有一个 VAD 认为含语音即拒绝
        if self.reject_on_any_speech and self.fail_count > 0:
            self.final_category = "rejected"
            self.confidence = self.max_speech_ratio
        # 三档仅由 fail_count 决定；strict fallback 时强制 rejected
        elif self.has_fallback:
            self.final_category = "rejected"
            self.confidence = 1.0
        elif self.fail_count == 0:
            self.final_category = "noise"
            self.confidence = 1.0 - self.max_speech_ratio
        elif self.fail_count == 1:
            self.final_category = "suspected"
            self.confidence = self.max_speech_ratio
        else:
            self.final_category = "rejected"
            self.confidence = self.max_speech_ratio

    def get_result_dict(self) -> Dict:
        vad_results = {}
        for i, r in enumerate(self.results):
            name = self.vad_types[i] if i < len(self.vad_types) else getattr(r, 'vad_type', f'vad_{i}')
            vad_results[name] = r.speech_ratio
        def _to_list(x):
            if x is None:
                return None
            if hasattr(x, 'tolist'):
                return x.tolist()
            return list(x) if not isinstance(x, list) else x
        return {
            'individual_results': [
                {
                    'vad_type': r.vad_type,
                    'has_speech': r.has_speech,
                    'speech_ratio': r.speech_ratio,
                    'confidence': r.confidence,
                    'frame_probs': _to_list(getattr(r, 'frame_probs', None)),
                    'is_real_model': self._is_real_model(r),
                }
                for r in self.results
            ],
            'vad_results': vad_results,
            'fail_count': self.fail_count,
            'speech_ratio_threshold': self.speech_ratio_threshold,
            'strict_vad_models': self.strict_vad_models,
            'reject_on_any_speech': self.reject_on_any_speech,
            'real_model_count': getattr(self, 'real_model_count', len(self.results)),
            'has_fallback': getattr(self, 'has_fallback', False),
            'failed_vad_types': list(getattr(self, 'failed_vad_types', [])),
            'consensus': {
                'total_vads': self.total_vads,
                'avg_speech_ratio': self.avg_speech_ratio,
                'max_speech_ratio': self.max_speech_ratio,
                'min_speech_ratio': self.min_speech_ratio
            },
            'final': {
                'category': self.final_category,
                'confidence': self.confidence
            }
        }


class MultiVADDetector:
    def __init__(self,
                 vad_types: List[str] = None,
                 firered_model_dir: str = None,
                 funasr_model_dir: str = None,
                 nemo_model_path: str = None,
                 nemo_onnx_path: str = None,
                 use_gpu: bool = False,
                 speech_threshold: float = 0.4,
                 speech_ratio_threshold: float = 0.05,
                 strict_vad_models: bool = False,
                 reject_on_any_speech: bool = False,
                 temp_dir: Optional[str] = None,
                 num_workers: int = 1,
                 silero_group_size: int = 16,
                 keep_frame_probs: bool = False):

        if vad_types is None:
            vad_types = ["firered", "webrtc", "silero", "funasr", "nemo", "energy"]

        self.vad_types = vad_types
        self.detectors = {}
        self.speech_ratio_threshold = speech_ratio_threshold
        self.strict_vad_models = bool(strict_vad_models)
        self.reject_on_any_speech = bool(reject_on_any_speech)
        self.temp_dir = temp_dir
        if (
            isinstance(silero_group_size, bool)
            or not isinstance(silero_group_size, int)
            or silero_group_size <= 0
        ):
            raise ValueError(
                "silero_group_size 必须为正整数，"
                f"当前值: {silero_group_size!r}"
            )
        self.silero_group_size = silero_group_size
        self.keep_frame_probs = bool(keep_frame_probs)
        # VAD 批量检测的进程池大小；>1 时每个子进程独立加载一套模型，
        # 绕过 GIL 并行 FunASR 的帧级 Python 循环
        self.num_workers = max(1, int(num_workers))
        self._pool: Optional[ProcessPoolExecutor] = None
        # 进程池连续基础设施失败计数：BrokenProcessPool 时重建一次并重试，
        # 连续失败达到上限则中止整个 stage，避免静默降级吞掉全部数据
        self._consecutive_pool_failures = 0
        self._max_consecutive_pool_failures = 3
        # 进程池 worker 重建 detector 所需的构造参数（num_workers 强制为 1，避免递归建池）
        self._ctor_kwargs = dict(
            vad_types=vad_types,
            firered_model_dir=firered_model_dir,
            funasr_model_dir=funasr_model_dir,
            nemo_model_path=nemo_model_path,
            nemo_onnx_path=nemo_onnx_path,
            use_gpu=use_gpu,
            speech_threshold=speech_threshold,
            speech_ratio_threshold=speech_ratio_threshold,
            strict_vad_models=strict_vad_models,
            reject_on_any_speech=reject_on_any_speech,
            temp_dir=temp_dir,
            num_workers=1,
            silero_group_size=silero_group_size,
            keep_frame_probs=keep_frame_probs,
        )

        supported_vad_types = {
            "firered", "silero", "webrtc", "funasr", "nemo_onnx", "nemo", "energy",
        }
        unknown_vad_types = set(vad_types) - supported_vad_types
        if unknown_vad_types:
            raise ValueError(f"不支持的 VAD 类型: {sorted(unknown_vad_types)}")

        if self.num_workers <= 1:
            # 单进程模式保持即时加载（构造即预检模型可用性，行为不变）
            self._build_detectors()
        # num_workers > 1 时父进程 detectors 懒加载：进程池路径由 worker
        # 按 _ctor_kwargs 各自构造，父进程只有在非池化 detect/detect_array
        # 首次使用时才加载，不白占一份模型内存/显存

    def _build_detectors(self) -> None:
        """按 _ctor_kwargs 构造全部 VAD wrapper（worker 内同样走此路径）。"""
        kw = self._ctor_kwargs
        for vad_type in self.vad_types:
            if vad_type == "firered":
                self.detectors[vad_type] = FireRedVADWrapper(
                    model_dir=kw.get("firered_model_dir"),
                    use_gpu=kw.get("use_gpu", False),
                    speech_threshold=kw.get("speech_threshold", 0.4)
                )
            elif vad_type == "silero":
                self.detectors[vad_type] = SileroVADWrapper(
                    use_gpu=kw.get("use_gpu", False),
                    threshold=kw.get("speech_threshold", 0.4)
                )
            elif vad_type == "webrtc":
                self.detectors[vad_type] = WebRTCVADWrapper()
            elif vad_type == "funasr":
                self.detectors[vad_type] = FunASRVADWrapper(
                    model_dir=kw.get("funasr_model_dir"),
                    use_gpu=kw.get("use_gpu", False),
                    threshold=kw.get("speech_threshold", 0.4),
                    temp_dir=self.temp_dir,
                )
            elif vad_type == "nemo_onnx":
                self.detectors[vad_type] = NeMoONNXVADWrapper(
                    onnx_path=kw.get("nemo_onnx_path"),
                    threshold=kw.get("speech_threshold", 0.4),
                    use_gpu=kw.get("use_gpu", False)
                )
            elif vad_type == "nemo":
                self.detectors[vad_type] = NeMoVADWrapper(
                    model_path=kw.get("nemo_model_path"),
                    use_gpu=kw.get("use_gpu", False),
                    threshold=kw.get("speech_threshold", 0.4)
                )
            elif vad_type == "energy":
                self.detectors[vad_type] = FallbackVAD()

    def _ensure_detectors(self) -> None:
        """非池化路径首次使用时才构造 detectors（懒加载入口）。"""
        if not self.detectors:
            self._build_detectors()

    def detect(self, audio_path: str) -> MultiVADResult:
        """对音频运行所有 VAD。为避免多次 I/O 解码同一文件，先将音频加载到 16k 数组，
        写出单个临时 16k wav 供各 VAD 共用（FireRed、FunASR 等需要文件路径的 VAD 直接读取，
        无需重复解码）。"""
        self._ensure_detectors()
        try:
            # 统一加载并重采样到 16k
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.astype(np.float64).mean(axis=1)
            else:
                audio = audio.astype(np.float64)
            if sr != 16000:
                # VAD 对重采样质量不敏感，直接用 soxr 纯内存重采样；
                # shared 的 resample_audio 走 SoX 路径，每次都要写临时
                # 文件并 fork sox 子进程，开销过大
                import soxr
                audio = soxr.resample(audio.astype(np.float32, copy=False), sr, 16000)
        except Exception as e:
            logger.error("MultiVAD shared audio load failed for %s: %s", audio_path, e)
            results, used_vad_types = self._run_detectors(path=audio_path)
        else:
            # 临时 WAV 是文件型 detector 的基础设施依赖；写失败不得回退原路径
            # 并伪装成内容级 detector 结果。
            tmp_path = _write_shared_wav(audio, getattr(self, "temp_dir", None))
            try:
                results, used_vad_types = self._run_detectors(path=tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return MultiVADResult(
            results,
            vad_types=used_vad_types,
            speech_ratio_threshold=self.speech_ratio_threshold,
            strict_vad_models=self.strict_vad_models,
            reject_on_any_speech=self.reject_on_any_speech,
            failed_vad_types=[t for t in self.detectors if t not in used_vad_types],
        )

    def detect_array(self, audio: np.ndarray, sr: int) -> MultiVADResult:
        """直接从 numpy 数组运行所有 VAD，尽量避免文件 I/O。

        音频先统一重采样到 16k；只有 detect_array 需桥接文件路径的 detector
        （FireRed/FunASR，以 needs_wav_file 标记）才共用一份共享临时 wav，
        其余 detector 直接接收数组，减少不必要的磁盘写。
        """
        self._ensure_detectors()
        audio = audio.astype(np.float64)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            # VAD 对重采样质量不敏感，直接用 soxr 纯内存重采样；
            # shared 的 resample_audio 走 SoX 路径，每个非 16k chunk
            # 都要写临时文件并 fork sox 子进程，开销过大
            import soxr
            audio = soxr.resample(audio.astype(np.float32, copy=False), sr, 16000)
            sr = 16000

        # 只有 detect_array 需要桥接文件路径的 detector（FireRed/FunASR，
        # 以 needs_wav_file 类属性标记）才触发共享临时 wav 的写出；
        # 其余 detector 直接接收数组，减少不必要的磁盘写
        need_path = any(
            getattr(d, "needs_wav_file", False) for d in self.detectors.values()
        )
        tmp_path = _write_shared_wav(audio, getattr(self, "temp_dir", None)) if need_path else None
        try:
            results, used_vad_types = self._run_detectors(
                audio_array=audio.astype(np.float32), sr=16000, path=tmp_path
            )
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return MultiVADResult(
            results,
            vad_types=used_vad_types,
            speech_ratio_threshold=self.speech_ratio_threshold,
            strict_vad_models=self.strict_vad_models,
            reject_on_any_speech=self.reject_on_any_speech,
            failed_vad_types=[t for t in self.detectors if t not in used_vad_types],
        )

    def _run_detectors(
        self,
        audio_array: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
        path: Optional[str] = None,
    ) -> Tuple[List[VADResult], List[str]]:
        """对所有 detector 执行推理，优先使用数组输入，否则回退到文件路径。"""
        results: List[VADResult] = []
        used_vad_types: List[str] = []
        with _suppress_stdout_stderr():
            for vad_type, detector in self.detectors.items():
                try:
                    if audio_array is not None and hasattr(detector, "detect_array"):
                        if getattr(detector, "needs_wav_file", False) and path is not None:
                            # 桥接文件路径的 detector：传入共享临时 wav，
                            # 避免每个 detector 各写一份临时文件
                            result = detector.detect_array(
                                audio_array, sr or 16000, shared_wav_path=path
                            )
                        else:
                            result = detector.detect_array(audio_array, sr or 16000)
                    elif path is not None:
                        result = detector.detect(path)
                    else:
                        logger.error("MultiVAD 未提供音频输入")
                        continue
                    results.append(result)
                    used_vad_types.append(vad_type)
                except Exception as e:
                    logger.error("Error with %s: %s", vad_type, e)
                    continue
        return results, used_vad_types

    def detect_batch(self, audio_paths: List[str]) -> List[MultiVADResult]:
        """串行逐文件检测，与 detect 行为一致。"""
        return [self.detect(path) for path in audio_paths]

    def _create_pool(self) -> ProcessPoolExecutor:
        """创建 VAD 批量检测进程池。

        spawn：子进程干净重建 CUDA/模型上下文，避免 fork 继承父进程状态。
        多 GPU 时按 worker 序号 round-robin 绑定 CUDA_VISIBLE_DEVICES 物理
        id，避免全部 worker 挤在同一张卡。
        """
        ctx = multiprocessing.get_context("spawn")
        # 物理 GPU id 列表必须在父进程确定：子进程设置 CUDA_VISIBLE_DEVICES
        # 后 torch 的设备计数缓存可能不再反映真实拓扑。
        # 父进程 CVD 受限（如 "2,3"）时 device_count() 返回的是可见卡数，
        # worker 序号必须映射回物理 id 列表，否则会被当作物理卡号绑定到
        # 错误的卡（可能抢占其他任务的 GPU）。
        gpu_ids: List[str] = []
        if self._ctor_kwargs.get("use_gpu"):
            try:
                import torch
                n_gpu = torch.cuda.device_count()
            except Exception:
                n_gpu = 0
            if n_gpu > 1:
                cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
                if cvd:
                    gpu_ids = [t.strip() for t in cvd.split(",") if t.strip()]
                else:
                    gpu_ids = [str(i) for i in range(n_gpu)]
        # worker 序号计数器：各子进程在 initializer 内自增领取序号
        worker_counter = ctx.Value("i", 0)
        return ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=ctx,
            initializer=_vad_worker_init,
            initargs=(self._ctor_kwargs, worker_counter, gpu_ids),
        )

    def _detect_batch_arrays_pooled(
        self,
        groups: List[List[Tuple[np.ndarray, int]]],
    ) -> List[MultiVADResult]:
        """通过进程池执行已预处理、已分组的数组；池损坏时重建一次并重试。

        BrokenProcessPool 属于基础设施故障：先丢弃旧池、重建新池重试一次；
        重建失败或连续多批失败则抛出 RuntimeError 中止 stage，避免把
        基础设施故障降级为逐批的内容级拒绝。
        """
        if self._pool is None:
            self._pool = self._create_pool()
        try:
            grouped = list(self._pool.map(_vad_worker_detect_arrays, groups, chunksize=1))
            self._consecutive_pool_failures = 0
            return [r for g in grouped for r in g]
        except BrokenProcessPool:
            self._consecutive_pool_failures += 1
            logger.error(
                "VAD 进程池损坏（连续第 %d 批），关闭旧池并尝试重建",
                self._consecutive_pool_failures,
            )
            self.shutdown()
            if self._consecutive_pool_failures >= self._max_consecutive_pool_failures:
                # 基础设施中止：抛 InfraAbortError，由上层透传以中止 stage，
                # 避免被 batch 级 blanket except 吞掉降级为逐批失败
                raise InfraAbortError(
                    f"VAD 进程池连续 {self._consecutive_pool_failures} 批基础设施失败，"
                    "中止 Stage B；请检查子进程崩溃原因（如 OOM / CUDA 错误）"
                )
        # 重建一次并重试；重建或重试再失败则把异常抛给上层（重建失败即中止）
        self._pool = self._create_pool()
        try:
            grouped = list(self._pool.map(_vad_worker_detect_arrays, groups, chunksize=1))
            self._consecutive_pool_failures = 0
            return [r for g in grouped for r in g]
        except BrokenProcessPool:
            self._consecutive_pool_failures += 1
            self.shutdown()
            if self._consecutive_pool_failures >= self._max_consecutive_pool_failures:
                # 基础设施中止：抛 InfraAbortError，由上层透传以中止 stage，
                # 避免被 batch 级 blanket except 吞掉降级为逐批失败
                raise InfraAbortError(
                    f"VAD 进程池连续 {self._consecutive_pool_failures} 批基础设施失败，"
                    "中止 Stage B；请检查子进程崩溃原因（如 OOM / CUDA 错误）"
                )
            raise

    @staticmethod
    def _prepare_audio_arrays(
        audio_arrays: List[Tuple[np.ndarray, int]],
    ) -> List[Tuple[np.ndarray, int]]:
        """在分组/IPC 前统一为 C-contiguous mono 16k float32。"""
        prepared: List[Tuple[np.ndarray, int]] = []
        for audio, sr in audio_arrays:
            array = np.asarray(audio)
            if array.ndim > 1:
                array = array.astype(np.float64, copy=False).mean(axis=1)
            if sr != 16000:
                import soxr
                array = soxr.resample(
                    array.astype(np.float32, copy=False), sr, 16000
                )
            prepared.append((np.ascontiguousarray(array, dtype=np.float32), 16000))
        return prepared

    def detect_batch_arrays(
        self,
        audio_arrays: List[Tuple[np.ndarray, int]],
    ) -> List[MultiVADResult]:
        """对内存数组按配置分组检测，单进程与进程池均保持输入顺序。"""
        if not audio_arrays:
            return []
        prepared = self._prepare_audio_arrays(audio_arrays)
        group_size = getattr(self, "silero_group_size", 16)
        groups = [
            prepared[i: i + group_size]
            for i in range(0, len(prepared), group_size)
        ]
        if self.num_workers <= 1:
            return [
                result
                for group in groups
                for result in self.detect_arrays_group(group)
            ]
        return self._detect_batch_arrays_pooled(groups)

    # 支持 lockstep 批量前向的 detector 类型（有状态模型，batch 维=并行序列）
    _GROUP_LOCKSTEP_TYPES = ("silero",)

    def detect_arrays_group(
        self,
        audio_arrays: List[Tuple[np.ndarray, int]],
    ) -> List[MultiVADResult]:
        """对一组内存数组批量检测，结果与逐条 detect_array 一致。

        支持 lockstep 的 detector（silero：batch 维为并行序列的有状态模型）
        整组一次前向；其余 detector 逐条执行。失败语义与 _run_detectors
        一致：单个 detector 失败不影响其他 detector 与其他音频。
        """
        self._ensure_detectors()
        prepared = self._prepare_audio_arrays(audio_arrays)
        n = len(prepared)
        prepped = [audio for audio, _ in prepared]

        # per_detector[vad_type][k] = 第 k 条音频的 VADResult（失败为 None）
        per_detector: Dict[str, List[Optional[VADResult]]] = {}
        need_path = any(
            getattr(detector, "needs_wav_file", False)
            for detector in self.detectors.values()
        )
        shared_paths: List[Optional[str]] = [None] * n
        try:
            if need_path:
                for k, audio in enumerate(prepped):
                    try:
                        shared_paths[k] = _write_shared_wav(audio, getattr(self, "temp_dir", None))
                    except InfraAbortError as exc:
                        raise InfraAbortError(
                            f"共享 VAD 临时 WAV 写入失败 (chunk {k}): {exc}"
                        ) from exc

            with _suppress_stdout_stderr():
                for vad_type, detector in self.detectors.items():
                    det_results: List[Optional[VADResult]] = [None] * n
                    if vad_type in self._GROUP_LOCKSTEP_TYPES and hasattr(detector, "detect_arrays_batch"):
                        try:
                            det_results = list(detector.detect_arrays_batch(prepped, 16000))
                        except Exception as e:
                            # 整组失败回退到逐条，保留单条容错（与 _run_detectors 语义一致）
                            logger.error("%s lockstep 批量失败，回退逐条: %s", vad_type, e)
                            for k in range(n):
                                try:
                                    det_results[k] = detector.detect_array(prepped[k], 16000)
                                except Exception as e2:
                                    logger.error("Error with %s (chunk %d): %s", vad_type, k, e2)
                    else:
                        for k in range(n):
                            try:
                                if getattr(detector, "needs_wav_file", False):
                                    det_results[k] = detector.detect_array(
                                        prepped[k], 16000,
                                        shared_wav_path=shared_paths[k],
                                    )
                                else:
                                    det_results[k] = detector.detect_array(prepped[k], 16000)
                            except Exception as e:
                                logger.error("Error with %s (chunk %d): %s", vad_type, k, e)
                    per_detector[vad_type] = det_results
        finally:
            for tmp_path in shared_paths:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        out: List[MultiVADResult] = []
        for k in range(n):
            results_k: List[VADResult] = []
            used_k: List[str] = []
            for vad_type in self.detectors:
                r = per_detector.get(vad_type, [None] * n)[k]
                if r is not None:
                    results_k.append(r)
                    used_k.append(vad_type)
            out.append(MultiVADResult(
                results_k,
                vad_types=used_k,
                speech_ratio_threshold=self.speech_ratio_threshold,
                strict_vad_models=self.strict_vad_models,
                reject_on_any_speech=self.reject_on_any_speech,
                failed_vad_types=[t for t in self.detectors if t not in used_k],
            ))
        return out

    def shutdown(self) -> None:
        """显式关闭进程池；不调用时由解释器退出时的 atexit 兜底。"""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None


# ---------------------------------------------------------------------------
# VAD 进程池 worker（模块级定义，spawn 子进程可 pickle/import）
# ---------------------------------------------------------------------------

_VAD_WORKER_DETECTOR: Optional[MultiVADDetector] = None


def _vad_worker_init(
    ctor_kwargs: Dict[str, Any],
    worker_counter=None,
    gpu_ids: Optional[List[str]] = None,
) -> None:
    """子进程启动时加载一套独立的 VAD 模型（每个 worker 只加载一次）。

    多 GPU（gpu_ids 多于 1 个）时按 worker 序号 round-robin 绑定
    CUDA_VISIBLE_DEVICES 物理 id（gpu_ids[worker_index % len]），避免
    所有 worker 的模型与 CUDA context 挤在同一张卡。必须在加载模型
    （初始化 CUDA context）之前设置才生效。
    """
    # 父死看门狗：pool worker 阻塞在 CPython 内部 call queue 时无法自查
    # 父进程状态，父进程被杀后会变孤儿；父死即自行退出
    from ...core.process_guard import start_parent_watchdog

    start_parent_watchdog(os.getppid(), name="vad-worker-watchdog")
    global _VAD_WORKER_DETECTOR
    if gpu_ids and len(gpu_ids) > 1:
        worker_index = 0
        if worker_counter is not None:
            with worker_counter.get_lock():
                worker_index = worker_counter.value
                worker_counter.value += 1
        gpu_id = gpu_ids[worker_index % len(gpu_ids)]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.debug("VAD worker #%d 绑定物理 GPU %s", worker_index, gpu_id)
    _VAD_WORKER_DETECTOR = MultiVADDetector(**ctor_kwargs)


def _vad_worker_detect_arrays(
    payload: List[Tuple[np.ndarray, int]],
) -> List[MultiVADResult]:
    """子进程内执行一组 VAD，并在 IPC 返回前按配置剥离大字段。"""
    results = _VAD_WORKER_DETECTOR.detect_arrays_group(payload)
    if not getattr(_VAD_WORKER_DETECTOR, "keep_frame_probs", False):
        for grouped in results:
            for result in grouped.results:
                result.frame_probs = None
    return results
