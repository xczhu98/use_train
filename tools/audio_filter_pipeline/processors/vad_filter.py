"""Silero VAD 段级过滤处理器与纯时间线后处理。

用 Silero VAD 单次逐帧分析同时生成整段语音帧占比（speech ratio）和不可变
语音时间线 facts，供 ``vad_filter`` Stage 消费。所有接口都只分析输入，绝不
裁剪或修改 waveform，语音停顿保持原样。

模型：``third_party_scripts/SileroVAD/silero_vad.jit``，通过 ``torch.jit.load``
本地加载（不走 torch.hub / 网络）。路径解析与 ``processors/mos.py`` 一致：
优先 ``USE_TRAIN_THIRD_PARTY_DIR``，否则 ``third_party_scripts/SileroVAD/``。

模型说明（本地 silero_vad.jit 为 VADRNNJITMerge 变体）：
- 输入固定 16kHz × 512 采样（32ms，官方文档称 ~30ms 粒度）的单帧；
- 每帧输出一个语音概率；状态保存在模块内部，逐帧顺序推理（帧间状态必须
  顺序传递，不能整段 batch），每条音频开始前必须调用 ``reset_states()``。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import numbers
import os
import threading
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from ..utils.resample import resample_audio, resample_method_choices

logger = logging.getLogger(__name__)

# 权重路径解析：与 processors/mos.py 同一约定
_USE_TRAIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
THIRD_PARTY_DIR = os.environ.get("USE_TRAIN_THIRD_PARTY_DIR") or os.path.join(_USE_TRAIN_ROOT, "third_party_scripts")
DEFAULT_VAD_MODEL_PATH = os.path.join(THIRD_PARTY_DIR, "SileroVAD", "silero_vad.jit")

# Silero VAD 输入采样率与单帧采样点数（512 @ 16kHz = 32ms/帧）
MODEL_SR = 16000
WINDOW_SAMPLES = 512

# 单帧语音概率判为语音的阈值（Silero 官方默认 0.5）
SPEECH_PROB_THRESHOLD = 0.5
_DIRECT_RESAMPLE_METHODS = frozenset(resample_method_choices()) - {"auto"}


@dataclass(frozen=True)
class VadAnalysisParameters:
    """Step1 VAD 执行与 config digest 共用的不可变参数。"""

    analysis_sample_rate: int = MODEL_SR
    resample_method: str = "soxr_hq"
    frame_samples: int = WINDOW_SAMPLES
    speech_threshold: float = SPEECH_PROB_THRESHOLD
    min_speech_ms: int = 100
    max_silence_gap_ms: int = 200
    padding_ms: int = 100

    def __post_init__(self) -> None:
        if type(self.resample_method) is not str or not self.resample_method.strip():
            raise TypeError("resample_method 必须是 exact non-empty string")
        if self.resample_method not in _DIRECT_RESAMPLE_METHODS:
            raise ValueError(
                "resample_method 必须是明确受支持的本地 direct method"
            )


STEP1_VAD_PARAMETERS = VadAnalysisParameters()


@dataclass(frozen=True)
class SpeechInterval:
    """同一分析采样域内的不可变半开语音样本区间。"""

    start_sample: int
    end_sample: int
    mean_probability: float


@dataclass(frozen=True)
class VadAnalysis:
    """单次 16kHz VAD 分析产生的不可变 ratio 与时间线 facts。"""

    speech_ratio: float
    timeline: Tuple[SpeechInterval, ...]
    sample_rate: int
    num_samples: int


def speech_timeline(
    probabilities: np.ndarray,
    *,
    total_samples: int,
    sample_rate: int = MODEL_SR,
    frame_samples: int = WINDOW_SAMPLES,
    speech_threshold: float = SPEECH_PROB_THRESHOLD,
    min_speech_ms: int = 100,
    max_silence_gap_ms: int = 200,
    padding_ms: int = 100,
) -> Tuple[SpeechInterval, ...]:
    """将固定帧语音概率转换为不可变的半开样本区间。"""

    if not isinstance(probabilities, np.ndarray):
        raise TypeError("probabilities 必须是 NumPy ndarray")
    if probabilities.ndim != 1:
        raise ValueError("probabilities 必须是一维数组")
    if not np.issubdtype(probabilities.dtype, np.floating):
        raise TypeError("probabilities 必须是 real floating 数组")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities 必须全部有限")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities 必须位于 [0.0, 1.0]")

    def _require_int(name: str, value: int, *, positive: bool) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} 必须是非 bool 整数")
        if (positive and value <= 0) or (not positive and value < 0):
            requirement = "正整数" if positive else "非负整数"
            raise ValueError(f"{name} 必须是{requirement}")
        return int(value)

    total_samples = _require_int("total_samples", total_samples, positive=False)
    sample_rate = _require_int("sample_rate", sample_rate, positive=True)
    frame_samples = _require_int("frame_samples", frame_samples, positive=True)
    min_speech_ms = _require_int("min_speech_ms", min_speech_ms, positive=False)
    max_silence_gap_ms = _require_int(
        "max_silence_gap_ms", max_silence_gap_ms, positive=False
    )
    padding_ms = _require_int("padding_ms", padding_ms, positive=False)

    if isinstance(speech_threshold, (bool, np.bool_)) or not isinstance(
        speech_threshold, numbers.Real
    ):
        raise TypeError("speech_threshold 必须是非 bool real 数")
    if not math.isfinite(speech_threshold) or not 0.0 <= speech_threshold <= 1.0:
        raise ValueError("speech_threshold 必须有限且位于 [0.0, 1.0]")

    if total_samples > len(probabilities) * frame_samples:
        raise ValueError("total_samples 超过 probability frames 覆盖范围")
    if probabilities.size == 0:
        return ()

    min_speech_samples = min_speech_ms * sample_rate // 1000
    max_gap_samples = max_silence_gap_ms * sample_rate // 1000
    padding_samples = padding_ms * sample_rate // 1000

    # 先提取 raw speech runs，再按 clamp 后的时长做 min speech 过滤。
    # run 边界由 numpy 一次性定位，只对 run 做 Python 循环：Step1 每个文件都
    # 调用一次，帧级 Python 循环在 10 万文件量级下是主要开销。
    speech_frames = probabilities >= speech_threshold
    padded = np.concatenate((
        np.zeros(1, dtype=bool), speech_frames, np.zeros(1, dtype=bool),
    ))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    retained = []
    for run_start, run_end in zip(edges[0::2], edges[1::2]):
        start = min(int(run_start) * frame_samples, total_samples)
        end = min(int(run_end) * frame_samples, total_samples)
        if end > start and end - start >= min_speech_samples:
            run_probabilities = probabilities[run_start:run_end]
            retained.append(
                [start, end, float(run_probabilities.sum(dtype=np.float64)), int(run_end - run_start)]
            )

    # 仅合并已通过 min speech 的 runs，桥接 silence 不参与概率均值。
    merged = []
    for start, end, probability_sum, frame_count in retained:
        if merged and start - merged[-1][1] <= max_gap_samples:
            merged[-1][1] = end
            merged[-1][2] += probability_sum
            merged[-1][3] += frame_count
        else:
            merged.append([start, end, probability_sum, frame_count])

    # Padding 后做 union；重叠或相触区间对应的 retained speech frames 一并聚合。
    unioned = []
    for start, end, probability_sum, frame_count in merged:
        padded_start = max(0, start - padding_samples)
        padded_end = min(total_samples, end + padding_samples)
        if unioned and padded_start <= unioned[-1][1]:
            unioned[-1][1] = max(unioned[-1][1], padded_end)
            unioned[-1][2] += probability_sum
            unioned[-1][3] += frame_count
        else:
            unioned.append([padded_start, padded_end, probability_sum, frame_count])

    return tuple(
        SpeechInterval(start, end, float(probability_sum / frame_count))
        for start, end, probability_sum, frame_count in unioned
    )


class SileroVadProcessor:
    """Silero VAD：一次逐帧推理生成 ratio 与 16kHz 时间线 facts。

    - ``frame_probabilities`` 返回每个 32ms 帧的语音概率；
    - ``analyze`` 同时返回 legacy frame-count ratio 与完整时间线；
    - ``speech_ratio`` 保持概率 >= ``SPEECH_PROB_THRESHOLD`` 的帧占比语义；
    - 所有接口绝不裁剪或修改 waveform。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        resample_method: str = "auto",
        analysis_parameters: VadAnalysisParameters = STEP1_VAD_PARAMETERS,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_path = model_path or DEFAULT_VAD_MODEL_PATH
        self.resample_method = resample_method
        self.analysis_parameters = analysis_parameters
        self._model = _load_vad_model(self.model_path, self.device)

    def _to_16k(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """任意采样率输入重采样到配置的 Silero 分析域。"""
        parameters = getattr(self, "analysis_parameters", STEP1_VAD_PARAMETERS)
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if sample_rate != parameters.analysis_sample_rate:
            wav = resample_audio(
                wav,
                sample_rate,
                parameters.analysis_sample_rate,
                method=self.resample_method,
            )
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        return wav

    def _frame_probabilities_16k(self, wav: np.ndarray) -> np.ndarray:
        """对已转换到 Silero 分析域的 waveform 顺序运行一次全部帧。"""
        if wav.size == 0:
            return np.zeros(0, dtype=np.float32)

        parameters = getattr(self, "analysis_parameters", STEP1_VAD_PARAMETERS)
        n_frames = max(1, math.ceil(wav.size / parameters.frame_samples))
        padded = np.zeros(n_frames * parameters.frame_samples, dtype=np.float32)
        padded[: wav.size] = wav
        frames = torch.from_numpy(padded).view(n_frames, parameters.frame_samples).to(
            self.device
        )

        # 该 jit 模型状态保存在模块内部，每条音频开始前必须重置。
        # 模型按线程独立实例化（见文件底部缓存说明），不存在跨线程共享，
        # 因此 reset_states() 到逐帧推理结束无需互斥锁。
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()

        # 逐帧顺序前向（该模型是状态ful RNN，帧间状态必须顺序传递；
        # 曾尝试整段 batch 一次前向，但 batch 各帧独立、丢失跨帧状态，
        # 会拉低持续语音的帧概率并打破 speech_like 回归测试，故保持逐帧）。
        # 优化点：不在循环内逐帧 .item()（每次都会 GPU 同步），
        # 而是先收集输出 tensor，最后一次性 .cpu()。
        outs = []
        for i in range(n_frames):
            outs.append(
                self._model(
                    frames[i : i + 1], parameters.analysis_sample_rate
                ).reshape(-1)[0]
            )
        return torch.stack(outs).detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def frame_probabilities(
        self, waveform: Union[np.ndarray, torch.Tensor], sample_rate: int
    ) -> np.ndarray:
        """逐帧（32ms）语音概率，shape (n_frames,)。"""
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        wav = self._to_16k(waveform, int(sample_rate))
        return self._frame_probabilities_16k(wav)

    @torch.inference_mode()
    def analyze(
        self, waveform: Union[np.ndarray, torch.Tensor], sample_rate: int
    ) -> VadAnalysis:
        """单次推理生成 legacy ratio 与 16kHz 分析域时间线 facts。"""
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        wav = self._to_16k(waveform, int(sample_rate))
        parameters = getattr(self, "analysis_parameters", STEP1_VAD_PARAMETERS)
        num_samples = int(wav.size)
        probs = self._frame_probabilities_16k(wav)
        ratio = (
            0.0
            if probs.size == 0
            else float((probs >= parameters.speech_threshold).mean())
        )
        timeline = speech_timeline(
            probs,
            total_samples=num_samples,
            sample_rate=parameters.analysis_sample_rate,
            frame_samples=parameters.frame_samples,
            speech_threshold=parameters.speech_threshold,
            min_speech_ms=parameters.min_speech_ms,
            max_silence_gap_ms=parameters.max_silence_gap_ms,
            padding_ms=parameters.padding_ms,
        )
        return VadAnalysis(
            ratio, timeline, int(parameters.analysis_sample_rate), num_samples
        )

    def speech_ratio(
        self, waveform: Union[np.ndarray, torch.Tensor], sample_rate: int
    ) -> float:
        """整段语音帧占比：概率 >= 0.5 的帧数 / 总帧数。空音频返回 0.0。"""
        probs = self.frame_probabilities(waveform, sample_rate)
        if probs.size == 0:
            return 0.0
        return float((probs >= SPEECH_PROB_THRESHOLD).mean())


# ── 每线程模型/处理器缓存 ──
# silero_vad.jit 是有状态 RNN（状态保存在模块内部，靠 reset_states() 复位），
# 同一模型实例绝不能跨线程并发推理，否则 reset/前向交错导致 speech_ratio 静默算错。
# 因此模型与处理器都按线程缓存：每个线程持有独立的 jit 模块实例（权重仅 ~2MB，
# 线程数有限，内存开销可忽略），从根上消除竞态，无需互斥锁串行化推理。
_THREAD_LOCAL = threading.local()

# 进程级加载日志节流：仅记录"某 (path, device) 是否已打过 info 日志"，
# 不是缓存共享——模型缓存仍是每线程独立实例（见上方注释）。
# 否则 step1 线程池下每个线程首次加载都打一条 info，日志刷屏。
_LOGGED_MODEL_KEYS: set = set()
_LOGGED_MODEL_KEYS_LOCK = threading.Lock()


def _thread_cache(name: str) -> Dict:
    """取当前线程的命名缓存字典（不存在则创建）。"""
    cache = getattr(_THREAD_LOCAL, name, None)
    if cache is None:
        cache = {}
        setattr(_THREAD_LOCAL, name, cache)
    return cache


def _load_vad_model(model_path: str, device: torch.device) -> torch.jit.ScriptModule:
    """加载并缓存 Silero VAD TorchScript 模型（每线程独立实例）。"""
    key = (os.path.abspath(model_path), str(device))
    cache = _thread_cache("_model_cache")
    if key not in cache:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Silero VAD 权重缺失: {model_path}")
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        cache[key] = model
        # 进程级首次加载打 info，后续线程的重复加载降级为 debug（日志节流，
        # 不影响每线程独立实例的缓存语义）
        with _LOGGED_MODEL_KEYS_LOCK:
            first_in_process = key not in _LOGGED_MODEL_KEYS
            _LOGGED_MODEL_KEYS.add(key)
        if first_in_process:
            logger.info("Silero VAD 模型已加载: %s (device=%s)", model_path, device)
        else:
            logger.debug(
                "Silero VAD 模型已加载（线程 %s 的独立实例）: %s (device=%s)",
                threading.get_ident(), model_path, device,
            )
    return cache[key]


def get_vad_processor(
    device: str = "auto",
    resample_method: str = "auto",
    model_path: Optional[str] = None,
    use_cache: bool = True,
    analysis_parameters: VadAnalysisParameters = STEP1_VAD_PARAMETERS,
) -> SileroVadProcessor:
    """获取（并按线程缓存）SileroVadProcessor 实例。

    每个线程拿到独立的 processor + 独立的有状态 RNN 模型实例，
    多线程并发推理互不影响；``use_cache=False`` 时每次新建实例。
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (
        device,
        resample_method,
        model_path or DEFAULT_VAD_MODEL_PATH,
        analysis_parameters,
    )
    if not use_cache:
        return SileroVadProcessor(
            model_path=model_path,
            device=device,
            resample_method=resample_method,
            analysis_parameters=analysis_parameters,
        )
    cache = _thread_cache("_processor_cache")
    processor = cache.get(key)
    if processor is None:
        processor = SileroVadProcessor(
            model_path=model_path,
            device=device,
            resample_method=resample_method,
            analysis_parameters=analysis_parameters,
        )
        cache[key] = processor
    return processor


def vad_model_available() -> bool:
    """检查默认 Silero VAD 权重是否就绪。"""
    return os.path.isfile(DEFAULT_VAD_MODEL_PATH)
