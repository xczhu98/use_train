"""Step1 与筛选策略无关的纯音频分析事实映射。"""

from __future__ import annotations

import logging
import math
import numbers
import os
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import replace
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from ...config import PipelineConfig, STEP1_VAD_MAX_WORKERS
from ...utils import bandwidth as bandwidth_module
from ...utils.algorithm_identity import analyzer_algorithm_payload
from ...utils.artifacts import ArtifactIndex, PipelineStep
from ...utils.bandwidth import (
    BandwidthAnalysisParameters,
    estimate_audio_bandwidth,
    get_optimal_sample_rate,
)
from ...utils.resample import resample_audio
from .. import vad_filter as vad_module
from ..segmentation import (
    STEP1_STABLE_SILENCE_PARAMETERS,
    _STABLE_SILENCE_ALGORITHM,
    _STABLE_SILENCE_WORKING_SET_MAX_BYTES,
    detect_stable_silence_intervals,
)
from ..signal_metrics import SignalMetrics, compute_signal_metrics
from ..vad_filter import SpeechInterval, VadAnalysis, VadAnalysisParameters
from .base import ArtifactStage, StageRunContext
from .ten_vad import TenVadProcessor, TenVadRuntimeOptions
from .ingest import (
    STEP0_ROW_FIELDS,
    _add_exception_note,
    _shutdown_executor_bounded,
)
from . import source_io as source_io_module
from .source_io import DecodedSource, decode_verified_source


logger = logging.getLogger(__name__)


STEP1_ANALYSIS_FIELDS = (
    "source_start_sample",
    "source_end_sample",
    "signal_finite",
    "signal_zero_ratio",
    "signal_peak",
    "signal_clipping_ratio",
    "signal_dc_offset",
    "signal_rms_dbfs",
    "signal_num_samples",
    "bandwidth_hz",
    "estimated_sr",
    "vad_speech_ratio",
    "vad_timeline_sample_rate",
    "vad_timeline_num_samples",
    "vad_timeline",
    "stable_silence_timeline",
)
STEP1_ROW_FIELDS = STEP0_ROW_FIELDS + STEP1_ANALYSIS_FIELDS
STEP1_ANALYSIS_CHANNEL_POLICY = "first_channel_index_0"
STEP1_ANALYSIS_MAX_INFLIGHT_BYTES = 4 * 1024 * 1024 * 1024
_STEP1_FLOAT32_ITEMSIZE = np.dtype(np.float32).itemsize
_STEP1_FLOAT64_ITEMSIZE = np.dtype(np.float64).itemsize
_STEP1_SIGNAL_METRICS_BLOCK_SAMPLES = 262_144


def _first_channel_audio(decoded: DecodedSource) -> np.ndarray:
    """返回 Step1 统一消费的第 1 通道（Python 索引 0）mono 连续数组。

    DecodedSource.audio 本身就是只驻留 channel 0 的一维 float32 数组。"""
    return np.ascontiguousarray(decoded.audio, dtype=np.float32)


def _is_lowercase_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _step1_working_set_bytes(
    upstream_row: Mapping[str, Any],
    *,
    source_decode_parameters: source_io_module.SourceDecodeParameters,
    bandwidth_parameters: BandwidthAnalysisParameters,
    vad_parameters: VadAnalysisParameters,
    raw_source_spooled: bool = True,
) -> int:
    """保守估算一个 Step1 worker 的 source 级峰值工作集。"""

    if not isinstance(upstream_row, Mapping):
        raise ValueError("Step1 upstream row must be a mapping")
    if not isinstance(source_decode_parameters, source_io_module.SourceDecodeParameters):
        raise ValueError("Step1 source decode parameters are invalid")
    if not isinstance(bandwidth_parameters, BandwidthAnalysisParameters):
        raise ValueError("Step1 bandwidth parameters are invalid")
    if not isinstance(vad_parameters, VadAnalysisParameters):
        raise ValueError("Step1 VAD parameters are invalid")
    if type(raw_source_spooled) is not bool:
        raise ValueError("Step1 raw_source_spooled must be a Python bool")

    def positive_int(field: str) -> int:
        value = upstream_row.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Step1 upstream row {field} must be a positive Python integer")
        return value

    source_size_bytes = upstream_row.get("source_size_bytes")
    if type(source_size_bytes) is not int or source_size_bytes < 0:
        raise ValueError("Step1 upstream row source_size_bytes must be a non-negative Python integer")
    frames = positive_int("frames")
    channels = positive_int("channels")
    sample_rate = positive_int("sample_rate")
    if not (
        source_decode_parameters.min_source_sample_rate
        <= sample_rate
        <= source_decode_parameters.max_source_sample_rate
    ):
        raise ValueError("Step1 upstream row sample_rate is outside the source decode range")
    fft_size = bandwidth_parameters.fft_size
    if type(fft_size) is not int or fft_size <= 0:
        raise ValueError("Step1 bandwidth fft_size must be a positive Python integer")
    if type(vad_parameters.analysis_sample_rate) is not int or vad_parameters.analysis_sample_rate <= 0:
        raise ValueError("Step1 VAD analysis_sample_rate must be a positive Python integer")

    # 驻留解码量只统计 channel 0 mono（frames × float32）；多通道源的其余
    # 通道在解码块内 transient 出现并立即释放，由块长约束而非本护栏约束。
    decoded_bytes = frames * _STEP1_FLOAT32_ITEMSIZE
    if decoded_bytes > source_decode_parameters.max_decoded_bytes:
        raise ValueError("Step1 source resource limit exceeded: decoded waveform")
    if sample_rate == vad_parameters.analysis_sample_rate:
        vad_capacity_samples = frames
    else:
        vad_capacity_samples = (
            frames * vad_parameters.analysis_sample_rate + sample_rate - 1
        ) // sample_rate + 1
    vad_capacity_bytes = vad_capacity_samples * _STEP1_FLOAT32_ITEMSIZE
    if vad_capacity_bytes > source_decode_parameters.max_vad_analysis_bytes:
        raise ValueError("Step1 VAD analysis resource limit exceeded")

    hop_size = fft_size // 4
    if hop_size <= 0:
        raise ValueError("Step1 bandwidth fft_size must be at least 4")
    frame_count = 1 if frames < fft_size else 1 + (frames - fft_size) // hop_size
    fft_block_frames = min(frame_count, bandwidth_module._FFT_FRAME_BLOCK)
    fft_bins = fft_size // 2 + 1
    bandwidth_full_bytes = frames * _STEP1_FLOAT64_ITEMSIZE * 2
    bandwidth_frame_state_bytes = frame_count * np.dtype(np.int64).itemsize * 2
    bandwidth_scratch_bytes = (
        fft_size * _STEP1_FLOAT64_ITEMSIZE
        + fft_block_frames * fft_size * _STEP1_FLOAT64_ITEMSIZE * 2
        + fft_block_frames * fft_bins * _STEP1_FLOAT64_ITEMSIZE * 3
        + fft_block_frames * fft_bins * np.dtype(np.bool_).itemsize
    )
    signal_metrics_bytes = (
        min(frames, _STEP1_SIGNAL_METRICS_BLOCK_SAMPLES) * _STEP1_FLOAT64_ITEMSIZE
    )
    raw_spool_bytes = 1024 * 1024 if raw_source_spooled else 0
    return (
        raw_spool_bytes
        + decoded_bytes
        + frames * _STEP1_FLOAT32_ITEMSIZE
        + vad_capacity_bytes * 4
        + bandwidth_full_bytes
        + bandwidth_frame_state_bytes
        + bandwidth_scratch_bytes
        + signal_metrics_bytes
        + _STABLE_SILENCE_WORKING_SET_MAX_BYTES
    )


def _validate_decoded(decoded: DecodedSource) -> None:
    if not isinstance(decoded, DecodedSource):
        raise ValueError("decoded must be a DecodedSource")

    audio = decoded.audio
    if not isinstance(audio, np.ndarray):
        raise ValueError("decoded audio must be a NumPy ndarray")
    if audio.dtype != np.dtype(np.float32):
        raise ValueError("decoded audio must have float32 dtype")
    if audio.ndim != 1:
        raise ValueError("decoded audio must be the one-dimensional channel-0 mono waveform")
    if not audio.flags.c_contiguous:
        raise ValueError("decoded audio must be C-contiguous")

    for field in ("sample_rate", "frames", "channels"):
        value = getattr(decoded, field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"decoded {field} must be a positive Python integer")
    if audio.shape != (decoded.frames,):
        raise ValueError("decoded audio shape must match frames")
    if decoded.frames == 0:
        raise ValueError("decoded audio must not be empty")

    for field in ("format", "subtype"):
        value = getattr(decoded, field)
        if type(value) is not str or not value.strip():
            raise ValueError(f"decoded {field} must be a non-empty string")
    if type(decoded.source_size_bytes) is not int or decoded.source_size_bytes < 0:
        raise ValueError("decoded source size must be a non-negative Python integer")
    if not _is_lowercase_sha256(decoded.source_sha256):
        raise ValueError("decoded source sha256 must be 64 lowercase hexadecimal characters")
    if decoded.vad_capacity_samples is not None and (
        type(decoded.vad_capacity_samples) is not int
        or decoded.vad_capacity_samples <= 0
    ):
        raise ValueError(
            "decoded VAD capacity must be a positive Python integer or None"
        )


def _finite_real(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite real number")
    return result


def _validated_signal_metrics(decoded: DecodedSource) -> SignalMetrics:
    try:
        metrics = compute_signal_metrics(_first_channel_audio(decoded))
    except Exception as exc:
        raise ValueError(f"signal metrics analysis failed: {exc}") from exc
    if not isinstance(metrics, SignalMetrics):
        raise ValueError("signal metrics must be a SignalMetrics result")
    if type(metrics.num_samples) is not int or metrics.num_samples != decoded.frames:
        raise ValueError("signal metrics num_samples does not match first-channel audio")
    if metrics.finite is not True:
        raise ValueError("signal metrics require finite decoded audio")

    for field in ("zero_ratio", "peak", "clipping_ratio", "dc_offset"):
        if getattr(metrics, field) is None:
            raise ValueError(f"signal metrics {field} must not be None")
        _finite_real(getattr(metrics, field), f"signal metrics {field}")
    if metrics.rms_dbfs is not None:
        _finite_real(metrics.rms_dbfs, "signal metrics rms_dbfs")
    return metrics


def _validated_bandwidth(
    decoded: DecodedSource,
    parameters: BandwidthAnalysisParameters,
) -> tuple[float, int | None]:
    try:
        result = estimate_audio_bandwidth(
            _first_channel_audio(decoded),
            sr=decoded.sample_rate,
            fft_size=parameters.fft_size,
            threshold_db=parameters.threshold_db,
        )
    except Exception as exc:
        raise ValueError(f"bandwidth analysis failed: {exc}") from exc
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("bandwidth analysis must return sample rate and bandwidth")
    returned_sr, raw_bandwidth = result
    if type(returned_sr) is not int or returned_sr != decoded.sample_rate:
        raise ValueError("bandwidth sample rate must match source sample rate")

    bandwidth_hz = _finite_real(raw_bandwidth, "bandwidth_hz")
    if bandwidth_hz < 0.0 or bandwidth_hz > decoded.sample_rate / 2:
        raise ValueError("bandwidth_hz must be within source Nyquist bounds")

    if bandwidth_hz == 0.0:
        # 零带宽是全静音、纯直流或无可检测频带的有效分析事实；此时没有
        # 可推导的有效采样率，使用 None 显式保留该语义。
        return bandwidth_hz, None
    try:
        estimated_sr = get_optimal_sample_rate(bandwidth_hz, decoded.sample_rate)
    except Exception as exc:
        raise ValueError(f"bandwidth analysis failed: {exc}") from exc
    if (
        type(estimated_sr) is not int
        or estimated_sr <= 0
        or estimated_sr > decoded.sample_rate
    ):
        raise ValueError("optimal sample rate must be a positive source-bounded Python integer")
    return bandwidth_hz, estimated_sr


def _validated_vad(
    decoded: DecodedSource,
    vad_processor: Any,
    parameters: VadAnalysisParameters,
) -> VadAnalysis:
    if parameters.analysis_sample_rate != vad_module.MODEL_SR:
        raise ValueError(
            f"VAD analysis_sample_rate must be pinned to {vad_module.MODEL_SR}"
        )
    if (
        type(decoded.vad_capacity_samples) is not int
        or decoded.vad_capacity_samples <= 0
    ):
        raise ValueError("decoded VAD capacity must be a positive Python integer")
    mono_audio = _first_channel_audio(decoded)
    if not bool(np.all(np.isfinite(mono_audio))):
        raise ValueError("VAD waveform must be finite")

    if decoded.sample_rate == parameters.analysis_sample_rate:
        vad_audio = mono_audio
    else:
        try:
            vad_audio = resample_audio(
                mono_audio,
                decoded.sample_rate,
                parameters.analysis_sample_rate,
                method=parameters.resample_method,
            )
        except Exception as exc:
            raise ValueError(f"VAD resampling failed: {exc}") from exc
        vad_audio = np.ascontiguousarray(vad_audio, dtype=np.float32)
    if not bool(np.all(np.isfinite(vad_audio))):
        raise ValueError("VAD waveform must remain finite after float32 conversion")

    if vad_audio.size > decoded.vad_capacity_samples:
        raise ValueError(
            "VAD resampled waveform exceeds decoded VAD capacity"
        )

    try:
        analysis = vad_processor.analyze(vad_audio, parameters.analysis_sample_rate)
    except Exception as exc:
        raise ValueError(f"VAD analysis failed: {exc}") from exc

    if not isinstance(analysis, VadAnalysis):
        raise ValueError("VAD analysis must return VadAnalysis")
    speech_ratio = _finite_real(analysis.speech_ratio, "VAD speech_ratio")
    if not 0.0 <= speech_ratio <= 1.0:
        raise ValueError("VAD speech_ratio must be within [0, 1]")
    if type(analysis.sample_rate) is not int:
        raise ValueError("VAD sample_rate must be a Python integer")
    if analysis.sample_rate != vad_module.MODEL_SR:
        raise ValueError(f"VAD sample_rate must be {vad_module.MODEL_SR}")
    if type(analysis.num_samples) is not int or analysis.num_samples < 0:
        raise ValueError("VAD num_samples must be a non-negative Python integer")
    expected_num_samples = int(vad_audio.size)
    if analysis.num_samples != expected_num_samples:
        if decoded.sample_rate == parameters.analysis_sample_rate:
            raise ValueError("VAD num_samples must match decoded frames")
        raise ValueError("VAD num_samples must match resampled waveform length")
    if type(analysis.timeline) is not tuple:
        raise ValueError("VAD timeline must be a tuple")

    previous_end = 0
    for interval in analysis.timeline:
        if not isinstance(interval, SpeechInterval):
            raise ValueError("VAD timeline entries must be SpeechInterval")
        if type(interval.start_sample) is not int or type(interval.end_sample) is not int:
            raise ValueError("VAD interval bounds must be Python integers")
        if not 0 <= interval.start_sample < interval.end_sample <= analysis.num_samples:
            raise ValueError("VAD interval bounds are outside the analysis domain")
        if interval.start_sample < previous_end:
            raise ValueError("VAD timeline must be ordered and non-overlapping")
        probability = _finite_real(interval.mean_probability, "VAD mean_probability")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("VAD mean_probability must be within [0, 1]")
        previous_end = interval.end_sample
    return analysis


def analyze_decoded_source(
    decoded: DecodedSource,
    vad_processor: Any,
    *,
    bandwidth_parameters: BandwidthAnalysisParameters | None = None,
    vad_parameters: VadAnalysisParameters | None = None,
) -> dict[str, Any]:
    """以第一通道（Python 索引 0）分析完整源音频，返回 JSON-safe 事实。"""

    bandwidth_parameters = (
        bandwidth_parameters or bandwidth_module.STEP1_BANDWIDTH_PARAMETERS
    )
    vad_parameters = vad_parameters or vad_module.STEP1_VAD_PARAMETERS
    _validate_decoded(decoded)
    metrics = _validated_signal_metrics(decoded)
    bandwidth_hz, estimated_sr = _validated_bandwidth(decoded, bandwidth_parameters)
    vad = _validated_vad(decoded, vad_processor, vad_parameters)
    stable_silence = detect_stable_silence_intervals(
        _first_channel_audio(decoded)[:, None],
        decoded.sample_rate,
        parameters=STEP1_STABLE_SILENCE_PARAMETERS,
    )

    timeline = [
        {
            "start_sample": int(interval.start_sample),
            "end_sample": int(interval.end_sample),
            "mean_probability": round(float(interval.mean_probability), 6),
        }
        for interval in vad.timeline
    ]
    return {
        "source_start_sample": 0,
        "source_end_sample": int(decoded.frames),
        "signal_finite": bool(metrics.finite),
        "signal_zero_ratio": float(metrics.zero_ratio),
        "signal_peak": float(metrics.peak),
        "signal_clipping_ratio": float(metrics.clipping_ratio),
        "signal_dc_offset": float(metrics.dc_offset),
        "signal_rms_dbfs": (
            None if metrics.rms_dbfs is None else float(metrics.rms_dbfs)
        ),
        # artifact contract 仍记录完整 decoded frame×channel 元素数；指标数值本身
        # 已由第一通道 waveform 计算。
        "signal_num_samples": int(decoded.frames * decoded.channels),
        "bandwidth_hz": float(bandwidth_hz),
        "estimated_sr": estimated_sr,
        "vad_speech_ratio": round(float(vad.speech_ratio), 6),
        "vad_timeline_sample_rate": int(vad.sample_rate),
        "vad_timeline_num_samples": int(vad.num_samples),
        "vad_timeline": timeline,
        "stable_silence_timeline": [
            {"start_sample": start, "end_sample": end}
            for start, end in stable_silence
        ],
    }


_STEP1_WORKER_VAD_PROCESSOR: Any = None
_STEP1_WORKER_SOURCE_DECODE_PARAMETERS: Any = None
_STEP1_WORKER_BANDWIDTH_PARAMETERS: BandwidthAnalysisParameters | None = None
_STEP1_WORKER_VAD_PARAMETERS: VadAnalysisParameters | None = None


def _init_step1_analyze_worker(
    ten_vad_options: TenVadRuntimeOptions,
    source_decode_parameters: Any,
    bandwidth_parameters: BandwidthAnalysisParameters,
    vad_parameters: VadAnalysisParameters,
) -> None:
    """在每个 CPU child 中一次性创建并缓存 TEN VAD runtime。"""

    # 强制 BLAS/OpenMP 线程数为 1：不继承外层训练进程的线程设置，避免 N 个
    # worker 各自开满全核线程池而超订。变量须在 runtime 创建前设置；对已经
    # 初始化的 native 线程池，仍需以实际 worker 线程数和吞吐基准确认效果。
    for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_env] = "1"

    global _STEP1_WORKER_VAD_PROCESSOR
    global _STEP1_WORKER_SOURCE_DECODE_PARAMETERS
    global _STEP1_WORKER_BANDWIDTH_PARAMETERS
    global _STEP1_WORKER_VAD_PARAMETERS
    _STEP1_WORKER_SOURCE_DECODE_PARAMETERS = source_decode_parameters
    _STEP1_WORKER_BANDWIDTH_PARAMETERS = bandwidth_parameters
    _STEP1_WORKER_VAD_PARAMETERS = vad_parameters
    _STEP1_WORKER_VAD_PROCESSOR = TenVadProcessor(
        options=ten_vad_options,
        analysis_parameters=vad_parameters,
    )


def _step1_analysis_row(
    upstream_row: Mapping[str, Any],
    *,
    vad_processor: Any,
    source_decode_parameters: Any,
    bandwidth_parameters: BandwidthAnalysisParameters,
    vad_parameters: VadAnalysisParameters,
) -> Mapping[str, Any]:
    decoded = decode_verified_source(
        upstream_row,
        parameters=source_decode_parameters,
        vad_target_sample_rate=vad_parameters.analysis_sample_rate,
    )
    facts = analyze_decoded_source(
        decoded,
        vad_processor,
        bandwidth_parameters=bandwidth_parameters,
        vad_parameters=vad_parameters,
    )
    if not isinstance(facts, Mapping) or set(facts) != set(STEP1_ANALYSIS_FIELDS):
        raise ValueError("Step1 analysis fields do not match the exact contract")
    values = {**upstream_row, **facts}
    return {field: values[field] for field in STEP1_ROW_FIELDS}


def _analyze_step1_row_worker(upstream_row: Mapping[str, Any]) -> Mapping[str, Any]:
    """计算单条 Step1 row；artifact 发布始终由父进程负责。"""

    if _STEP1_WORKER_VAD_PROCESSOR is None:
        raise RuntimeError("Step1 CPU worker was not initialized")
    assert _STEP1_WORKER_BANDWIDTH_PARAMETERS is not None
    assert _STEP1_WORKER_VAD_PARAMETERS is not None
    return _step1_analysis_row(
        upstream_row,
        vad_processor=_STEP1_WORKER_VAD_PROCESSOR,
        source_decode_parameters=_STEP1_WORKER_SOURCE_DECODE_PARAMETERS,
        bandwidth_parameters=_STEP1_WORKER_BANDWIDTH_PARAMETERS,
        vad_parameters=_STEP1_WORKER_VAD_PARAMETERS,
    )


class Step1AnalyzeStage(ArtifactStage):
    """发布与 selection policy 无关的完整源音频分析事实。"""

    step = PipelineStep.STEP1
    upstream_step = PipelineStep.STEP0
    progress_mode = "self_managed"
    progress_unit = "source"

    def __init__(
        self,
        *,
        ten_vad_options: TenVadRuntimeOptions = TenVadRuntimeOptions(),
        workers: int = 1,
        max_pending: int = 1,
        executor_factory: Callable[..., Any] = ProcessPoolExecutor,
    ):
        if type(workers) is not int or not 0 < workers <= STEP1_VAD_MAX_WORKERS:
            raise ValueError(
                "Step1 workers must be a positive Python integer "
                f"<= {STEP1_VAD_MAX_WORKERS}"
            )
        if type(max_pending) is not int or not workers <= max_pending <= workers * 2:
            raise ValueError(
                "Step1 max_pending must be a Python integer in "
                "[workers, workers * 2]"
            )
        self.ten_vad_options = ten_vad_options
        self.workers = workers
        self.max_pending = max_pending
        self.executor_factory = executor_factory
        self.source_decode_parameters = (
            source_io_module.STEP1_SOURCE_DECODE_PARAMETERS
        )
        self.bandwidth_parameters = bandwidth_module.STEP1_BANDWIDTH_PARAMETERS
        base_vad_parameters = vad_module.STEP1_VAD_PARAMETERS
        self.vad_parameters = replace(
            base_vad_parameters, speech_threshold=ten_vad_options.threshold
        )

    def config_payload(self, config: PipelineConfig) -> Mapping[str, Any]:
        return {
            "row_contract": {"version": 7, "fields": list(STEP1_ROW_FIELDS)},
            # 手工语义版本必须叠加算法源码哈希：只改 algorithm 实现而不改
            # 这里的字面量时，旧 Step1 artifact 仍会失配而重算。
            "analysis_algorithm_source": analyzer_algorithm_payload(),
            "source_decode": {
                "version": 5,
                "descriptor_bound": True,
                "verify_source_size": True,
                "verify_source_sha256": True,
                "raw_snapshot": "bounded_disk_spool_shared_with_hash_and_decode",
                "dtype": "float32",
                "retained_layout": "first_channel_mono_1d",
                "decode_block_frames": (
                    source_io_module._STEP1_DECODE_BLOCK_FRAMES
                ),
                "max_decoded_bytes": (
                    self.source_decode_parameters.max_decoded_bytes
                ),
                "max_raw_source_bytes": (
                    self.source_decode_parameters.max_raw_source_bytes
                ),
            },
            "analysis_resources": {
                "version": 3,
                "channel_policy": STEP1_ANALYSIS_CHANNEL_POLICY,
                "max_inflight_bytes": STEP1_ANALYSIS_MAX_INFLIGHT_BYTES,
                "inflight_reservation": "raw_decoded_vad_fft_signal_stable_silence",
                "max_vad_analysis_bytes": (
                    self.source_decode_parameters.max_vad_analysis_bytes
                ),
                "min_source_sample_rate": (
                    self.source_decode_parameters.min_source_sample_rate
                ),
                "max_source_sample_rate": (
                    self.source_decode_parameters.max_source_sample_rate
                ),
                "vad_capacity_rule": "ceil_ratio_plus_one_non_identity",
            },
            "signal_metrics": {
                "version": 2,
                "channel_policy": STEP1_ANALYSIS_CHANNEL_POLICY,
            },
            "bandwidth": {
                "version": 4,
                "channel_policy": STEP1_ANALYSIS_CHANNEL_POLICY,
                "fft_size": self.bandwidth_parameters.fft_size,
                "threshold_db": self.bandwidth_parameters.threshold_db,
                "short_nonempty_window": "zero_pad_to_fft_size",
                "zero_bandwidth_estimated_sr": None,
            },
            "vad": {
                "version": 5,
                "channel_policy": STEP1_ANALYSIS_CHANNEL_POLICY,
                "native_cpu": True,
                "native_handler_lifecycle": "per_source",
                "asset": None,
                "hop_size": self.ten_vad_options.hop_size,
                "native_threshold": self.ten_vad_options.threshold,
                "workers": self.workers,
                "max_pending": self.max_pending,
                "analysis_sample_rate": self.vad_parameters.analysis_sample_rate,
                "resample_method": self.vad_parameters.resample_method,
                "speech_threshold": self.vad_parameters.speech_threshold,
                "min_speech_ms": self.vad_parameters.min_speech_ms,
                "max_silence_gap_ms": self.vad_parameters.max_silence_gap_ms,
                "padding_ms": self.vad_parameters.padding_ms,
            },
            "stable_silence": {
                "version": 3,
                "source_domain": "original_sample_rate",
                "channel_policy": STEP1_ANALYSIS_CHANNEL_POLICY,
                "working_set_max_bytes": _STABLE_SILENCE_WORKING_SET_MAX_BYTES,
                "algorithm": _STABLE_SILENCE_ALGORITHM,
                "frame_ms": STEP1_STABLE_SILENCE_PARAMETERS.frame_ms,
                "rms_threshold_dbfs": (
                    STEP1_STABLE_SILENCE_PARAMETERS.rms_threshold_dbfs
                ),
                "peak_threshold_dbfs": (
                    STEP1_STABLE_SILENCE_PARAMETERS.peak_threshold_dbfs
                ),
                "min_silence_ms": (
                    STEP1_STABLE_SILENCE_PARAMETERS.min_silence_ms
                ),
                "guard_ms": STEP1_STABLE_SILENCE_PARAMETERS.guard_ms,
                "interval": "half_open",
            },
            "waveform_policy": {
                "full_source": True,
                "trim": False,
                "concatenate": False,
            },
        }

    def _get_vad_processor(self) -> Any:
        return TenVadProcessor(
            options=self.ten_vad_options,
            analysis_parameters=self.vad_parameters,
        )

    def _validate_upstream_row(
        self,
        upstream_row: Mapping[str, Any],
        *,
        context: StageRunContext,
        source_keys: set[str],
        expected_fields: set[str],
    ) -> None:
        if not isinstance(upstream_row, Mapping):
            raise ValueError("Step0 upstream row must be a mapping")
        if set(upstream_row) != expected_fields:
            raise ValueError("Step0 upstream row fields do not match the exact contract")
        if upstream_row["dataset_id"] != context.dataset_id:
            raise ValueError("Step0 upstream row dataset_id mismatch")
        source_key = upstream_row["source_key"]
        if type(source_key) is not str or not source_key:
            raise ValueError("Step0 upstream row source_key must be a non-empty string")
        if source_key in source_keys:
            raise ValueError(f"duplicate source_key in Step1 artifact: {source_key!r}")
        source_keys.add(source_key)

    def _raw_source_spooled(self, context: StageRunContext) -> bool:
        return context.config.dataset_specs()[context.dataset_id].source_layout != "streaming_flac_aggregate"

    def _serial_rows(
        self,
        context: StageRunContext,
        upstream_rows: Iterable[Mapping[str, Any]],
    ) -> Iterable[Mapping[str, Any]]:
        vad_processor = self._get_vad_processor()
        source_keys: set[str] = set()
        expected_fields = set(STEP0_ROW_FIELDS)
        raw_source_spooled = self._raw_source_spooled(context)
        progress = tqdm(
            total=None,
            unit="source",
            desc=f"step1:{context.dataset_id}",
            file=sys.stdout,
            disable=None,
        )
        try:
            for upstream_row in upstream_rows:
                self._validate_upstream_row(
                    upstream_row,
                    context=context,
                    source_keys=source_keys,
                    expected_fields=expected_fields,
                )
                reservation = _step1_working_set_bytes(
                    upstream_row,
                    source_decode_parameters=self.source_decode_parameters,
                    bandwidth_parameters=self.bandwidth_parameters,
                    vad_parameters=self.vad_parameters,
                    raw_source_spooled=raw_source_spooled,
                )
                if reservation > STEP1_ANALYSIS_MAX_INFLIGHT_BYTES:
                    raise ValueError(
                        "Step1 source working set exceeds the global inflight byte limit"
                    )
                yield _step1_analysis_row(
                    upstream_row,
                    vad_processor=vad_processor,
                    source_decode_parameters=self.source_decode_parameters,
                    bandwidth_parameters=self.bandwidth_parameters,
                    vad_parameters=self.vad_parameters,
                )
                progress.update(1)
        finally:
            progress.close()

    def _pooled_rows(
        self,
        context: StageRunContext,
        upstream_rows: Iterable[Mapping[str, Any]],
        total: Optional[int],
    ) -> Iterable[Mapping[str, Any]]:
        source_keys: set[str] = set()
        expected_fields = set(STEP0_ROW_FIELDS)
        raw_source_spooled = self._raw_source_spooled(context)
        pending: dict[int, tuple[Future[Mapping[str, Any]], int]] = {}
        next_submit = 0
        next_yield = 0
        upstream = iter(upstream_rows)
        staged_row: Mapping[str, Any] | None = None
        staged_reservation = 0
        inflight_bytes = 0
        exhausted = False
        logger.info(
            "Step1 dataset=%s CPU pool workers=%d max_pending=%d max_inflight_bytes=%d native_threads=1",
            context.dataset_id,
            self.workers,
            self.max_pending,
            STEP1_ANALYSIS_MAX_INFLIGHT_BYTES,
        )
        executor = self.executor_factory(
            max_workers=self.workers,
            initializer=_init_step1_analyze_worker,
            initargs=(
                self.ten_vad_options,
                self.source_decode_parameters,
                self.bandwidth_parameters,
                self.vad_parameters,
            ),
        )
        progress = tqdm(
            total=total,
            unit="source",
            desc=f"step1:{context.dataset_id}",
            file=sys.stdout,
            disable=None,
        )
        completed: dict[int, Mapping[str, Any]] = {}
        completed_normally = False
        primary_error: BaseException | None = None
        try:
            while True:
                while len(pending) < self.max_pending:
                    if staged_row is None:
                        if exhausted:
                            break
                        try:
                            upstream_row = next(upstream)
                        except StopIteration:
                            exhausted = True
                            break
                        self._validate_upstream_row(
                            upstream_row,
                            context=context,
                            source_keys=source_keys,
                            expected_fields=expected_fields,
                        )
                        staged_reservation = _step1_working_set_bytes(
                            upstream_row,
                            source_decode_parameters=self.source_decode_parameters,
                            bandwidth_parameters=self.bandwidth_parameters,
                            vad_parameters=self.vad_parameters,
                            raw_source_spooled=raw_source_spooled,
                        )
                        if staged_reservation > STEP1_ANALYSIS_MAX_INFLIGHT_BYTES:
                            raise ValueError(
                                "Step1 source working set exceeds the global inflight byte limit"
                            )
                        staged_row = upstream_row
                    if inflight_bytes + staged_reservation > STEP1_ANALYSIS_MAX_INFLIGHT_BYTES:
                        break
                    future = executor.submit(_analyze_step1_row_worker, staged_row)
                    pending[next_submit] = (future, staged_reservation)
                    inflight_bytes += staged_reservation
                    staged_row = None
                    staged_reservation = 0
                    next_submit += 1
                if not pending:
                    if staged_row is not None:
                        raise RuntimeError("Step1 inflight scheduler could not submit a fitting source")
                    completed_normally = True
                    return
                futures = tuple(future for future, _reservation in pending.values())
                if all(callable(getattr(future, "done", None)) for future in futures):
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                else:
                    # 轻量测试 executor 可能只实现 result/cancel；真实
                    # ProcessPool Future 仍走 FIRST_COMPLETED 完成通知路径。
                    done = {futures[0]}
                for future in done:
                    ordinal = next(
                        index
                        for index, (item, _reservation) in pending.items()
                        if item is future
                    )
                    _completed, reservation = pending.pop(ordinal)
                    inflight_bytes -= reservation
                    row = future.result()
                    if not isinstance(row, Mapping) or tuple(row) != STEP1_ROW_FIELDS:
                        raise ValueError("Step1 worker row fields do not match the exact contract")
                    completed[ordinal] = row
                    progress.update(1)
                while next_yield in completed:
                    yield completed.pop(next_yield)
                    next_yield += 1
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            cleanup_error: BaseException | None = None
            try:
                progress.close()
            except BaseException as exc:
                cleanup_error = exc
            if completed_normally and cleanup_error is None:
                try:
                    # 正常路径可以等待全部 worker，保持资源回收完整。
                    executor.shutdown(wait=True, cancel_futures=True)
                except TypeError:
                    executor.shutdown(wait=True)
                except BaseException as exc:
                    cleanup_error = exc
            else:
                _shutdown_executor_bounded(
                    executor,
                    pending_futures=(future for future, _reservation in pending.values()),
                    stage=f"step1-analyze:{context.dataset_id}",
                    workers=self.workers,
                    primary_error=primary_error or cleanup_error,
                )
            if cleanup_error is not None:
                if primary_error is not None:
                    _add_exception_note(primary_error, f"Step1 pool cleanup failed: {cleanup_error!r}")
                else:
                    raise cleanup_error

    def produce_rows(
        self,
        context: StageRunContext,
        upstream_index: Optional[ArtifactIndex],
        upstream_rows: Iterable[Mapping[str, Any]],
    ) -> Iterable[Mapping[str, Any]]:
        if upstream_index is None or upstream_index.step != PipelineStep.STEP0.value:
            raise ValueError("Step1 analysis requires a Step0 upstream artifact")
        if self.workers == 1:
            yield from self._serial_rows(context, upstream_rows)
            return
        yield from self._pooled_rows(context, upstream_rows, upstream_index.row_count)
