"""
B2 静音处理：静音比例计算、静音 chunk 判定、长静音缩短。

本模块是本管道静音检测的唯一实现（帧能量 RMS 口径）。
tools/shared/audio_io.py 中的 detect_silence / calculate_silence_ratio /
shorten_silence 是另一套历史实现（阈值系数与末帧覆盖口径不同），
processors/audio_utils.py 已停止 re-export，新代码一律使用本模块。
"""
import logging
import os
import random
from typing import Any, List, Tuple, Dict, Optional, Iterator

import numpy as np
import soundfile as sf

from ..core.manifest_io import AudioRecord, read_manifest, write_manifest
from ..core.processor_base import Processor, _process_in_threads
from tools.shared.audio_io import make_unique_output_path

logger = logging.getLogger(__name__)

# 帧能量分块上限：一次物化整份 (帧数, 帧长) 平方矩阵时峰值内存与音频总时长
# 成正比（10 分钟 16k 音频约 15GB），且 B2/unified 多线程并发时成倍放大；
# 按块推进把峰值内存限制在 (块帧数, 帧长) 量级
_ENERGY_BLOCK_FRAMES = 4096


def compute_energy_per_frame(audio: np.ndarray, sr: int,
                              frame_duration: float = 0.025,
                              hop_duration: float = 0.010) -> Tuple[List[float], List[float]]:
    frame_samples = int(frame_duration * sr)
    hop_samples = int(hop_duration * sr)
    frame_starts = range(0, len(audio) - frame_samples, hop_samples)
    frame_count = len(frame_starts)
    if frame_count == 0:
        return [], []

    # 只计算完整帧，且不包含恰好位于尾部边界的帧。
    # 沿采样轴取窗；二维输入仍按旧实现对帧内所有通道一起求 RMS。
    windows = np.lib.stride_tricks.sliding_window_view(audio, frame_samples, axis=0)
    # 块大小对齐 hop_samples：每帧起始偏移与全量切片逐帧一致，数值结果不变
    block_frames = max(hop_samples, _ENERGY_BLOCK_FRAMES - _ENERGY_BLOCK_FRAMES % hop_samples)
    block_energies = []
    for block_start in range(0, frame_count, block_frames):
        block_end = min(block_start + block_frames, frame_count)
        starts = np.arange(block_start, block_end) * hop_samples
        frames = windows[starts]
        energy_axes = tuple(range(1, frames.ndim))
        block_energies.append(np.sqrt(np.mean(frames ** 2, axis=energy_axes)))
    energies = block_energies[0] if len(block_energies) == 1 else np.concatenate(block_energies)
    times = (np.arange(frame_count) * hop_samples / sr).tolist()
    return times, energies.tolist()


def _frame_silence_metrics(
    times: List[float],
    energies: List[float],
    sr: int,
    frame_duration: float,
    threshold: Optional[float] = None,
    abs_min_threshold: Optional[float] = None,
    active_threshold: Optional[float] = None,
) -> Tuple[List[Tuple[float, float]], int]:
    """单次遍历帧结果，生成静音区域并统计活跃帧数。"""
    if not energies:
        return [], 0

    if threshold is None:
        mean_energy = np.mean(energies)
        # 根据采样率微调系数：窄带信号能量更集中，阈值稍低；宽带信号噪声地板更高，阈值稍高
        if sr <= 8000:
            coeff = 0.25
        elif sr >= 48000:
            coeff = 0.35
        else:
            coeff = 0.30
        threshold = mean_energy * coeff

    # 绝对最小阈值保护
    if abs_min_threshold is None:
        abs_min_threshold = 0.001
    threshold = max(threshold, abs_min_threshold)

    silence_regions = []
    in_silence = False
    silence_start = 0
    active_frames = 0

    for time, energy in zip(times, energies):
        if active_threshold is not None and energy > active_threshold:
            active_frames += 1
        if energy < threshold and not in_silence:
            silence_start = time
            in_silence = True
        elif energy >= threshold and in_silence:
            silence_regions.append((silence_start, time))
            in_silence = False

    if in_silence:
        # times 是帧起始时间，末帧覆盖 frame_duration，尾段静音边界不能用 hop_duration
        silence_regions.append((silence_start, times[-1] + frame_duration))

    return silence_regions, active_frames


def _silence_regions_from_frames(
    times: List[float],
    energies: List[float],
    sr: int,
    frame_duration: float,
    threshold: Optional[float] = None,
    abs_min_threshold: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """从已计算的帧能量构造静音区域，避免重复计算帧 RMS。"""
    silence_regions, _ = _frame_silence_metrics(
        times, energies, sr, frame_duration, threshold, abs_min_threshold
    )
    return silence_regions


def detect_silence_regions(audio: np.ndarray, sr: int,
                           frame_duration: float = 0.025,
                           hop_duration: float = 0.010,
                           threshold: Optional[float] = None,
                           abs_min_threshold: Optional[float] = None) -> List[Tuple[float, float]]:
    """检测静音区域。

    Args:
        audio: 音频数组。
        sr: 采样率。
        frame_duration: 帧长（秒）。
        hop_duration: 帧移（秒）。
        threshold: 能量阈值；None 时自动计算（mean(energies) * 0.3，并按采样率微调）。
        abs_min_threshold: 绝对最小阈值，防止极低能量音频的阈值过低。
                           若为 None 则使用默认值 0.001（16bit PCM RMS 极小值）。
    """
    times, energies = compute_energy_per_frame(audio, sr, frame_duration, hop_duration)
    return _silence_regions_from_frames(
        times, energies, sr, frame_duration, threshold, abs_min_threshold
    )


def calculate_silence_ratio(audio: np.ndarray, sr: int,
                           frame_duration: float = 0.025,
                           hop_duration: float = 0.010,
                           abs_min_threshold: Optional[float] = None) -> float:
    silence_regions = detect_silence_regions(audio, sr, frame_duration, hop_duration,
                                              abs_min_threshold=abs_min_threshold)
    
    if not silence_regions:
        return 0.0
    
    total_silence = sum(end - start for start, end in silence_regions)
    total_duration = len(audio) / sr
    
    return total_silence / total_duration if total_duration > 0 else 0.0


def is_silent_chunk(audio: np.ndarray, sr: int,
                   frame_duration: float = 0.025,
                   hop_duration: float = 0.010,
                   energy_threshold: float = 0.005) -> bool:
    """判断 chunk 是否整体静音（活跃帧占比 < 5%）。

    Args:
        energy_threshold: 帧 RMS 绝对阈值。默认 0.005 对正常电平音频合适；
            对低电平有效噪声场景（如远场底噪、小声环境音）容易误杀，
            建议调低到 0.001-0.002（通过 config 的 silent_energy_threshold 配置）。
    """
    times, energies = compute_energy_per_frame(audio, sr, frame_duration, hop_duration)
    
    if not energies:
        return True
    
    active_frames = sum(1 for e in energies if e > energy_threshold)
    active_ratio = active_frames / len(energies)
    
    return active_ratio < 0.05


def shorten_long_silences(audio: np.ndarray, sr: int,
                          max_silence_duration: float = 2.0,
                          target_silence_duration: float = 1.0,
                          target_silence_range: Tuple[float, float] = (0.5, 1.5),
                          use_random: bool = True,
                          frame_duration: float = 0.025,
                          hop_duration: float = 0.010,
                          rng: Optional[random.Random] = None,
                          silence_regions: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
    """缩短过长的静默段
    
    Args:
        audio: 输入音频数组
        sr: 采样率
        max_silence_duration: 超过此值的静默段会被缩短（秒）
        target_silence_duration: 默认目标静默时长（秒）
        target_silence_range: 随机范围 (min, max)，默认 0.5-1.5 秒
        use_random: 是否使用随机目标时长
        frame_duration: 帧长（秒）
        hop_duration: 帧移（秒）
        rng: 可选的私有随机数生成器；未提供时保持历史全局随机语义。
        silence_regions: 可选的预计算静音区域，供同一音频的调用方复用帧扫描结果。
    """
    rng = rng or random
    if silence_regions is None:
        silence_regions = detect_silence_regions(audio, sr, frame_duration, hop_duration)
    
    if not silence_regions:
        return audio
    
    result_segments = []
    last_pos = 0
    
    for start, end in silence_regions:
        duration = end - start
        
        if duration > max_silence_duration:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            # 随机选择目标静默时长（如果启用）
            if use_random:
                actual_target = rng.uniform(target_silence_range[0], target_silence_range[1])
            else:
                actual_target = target_silence_duration
            
            # 确保不超过原静默段的一半
            half_original = duration / 2
            actual_target = min(actual_target, half_original)
            
            keep_start_samples = int(actual_target / 2 * sr)
            keep_end_samples = int(actual_target / 2 * sr)

            # 在裁剪边界各加 5ms 线性 fade：seg_start 尾部 fade_out、seg_end 头部
            # fade_in，两段无重叠直接拼接（非重叠式 crossfade），避免拼接 pop 声
            crossfade_samples = min(int(0.005 * sr), keep_start_samples, keep_end_samples)

            seg_start = audio[last_pos:start_sample + keep_start_samples]
            seg_end = audio[end_sample - keep_end_samples:end_sample]

            # 首段长静音（last_pos == 0）同样需要 fade：fade 落在拼接边界而非文件头，
            # 跳过会导致首个拼接点产生 pop 声
            if crossfade_samples > 0 and len(seg_start) > crossfade_samples and len(seg_end) > crossfade_samples:
                # 先拷贝再施加 fade：seg_start/seg_end 是入参 audio 的 numpy view，
                # 直接赋值会原地污染调用方的音频缓冲区
                seg_start = seg_start.copy()
                seg_end = seg_end.copy()
                fade_in = np.linspace(0, 1, crossfade_samples, dtype=np.float64)
                fade_out = np.linspace(1, 0, crossfade_samples, dtype=np.float64)
                seg_start[-crossfade_samples:] = seg_start[-crossfade_samples:] * fade_out
                seg_end[:crossfade_samples] = seg_end[:crossfade_samples] * fade_in

            result_segments.append(seg_start)
            result_segments.append(seg_end)
            
            last_pos = end_sample
        else:
            result_segments.append(audio[last_pos:int(end * sr)])
            last_pos = int(end * sr)
    
    if last_pos < len(audio):
        result_segments.append(audio[last_pos:])
    
    return np.concatenate(result_segments) if result_segments else audio


class SilenceProcessor:
    def __init__(self,
                 max_silence_ratio: float = 0.9,
                 min_duration: float = 0.5,
                 shorten_long_silences_flag: bool = True,
                 max_silence_duration: float = 2.0,
                 target_silence_duration: float = 1.0,
                 target_silence_min: float = 0.5,
                 target_silence_max: float = 1.5,
                 use_random_silence_duration: bool = True,
                 abs_min_energy: Optional[float] = None,
                 silent_energy_threshold: float = 0.005,
                 dc_offset_threshold: Optional[float] = 0.5,
                 random_seed: Optional[int] = None):
        # 注意量纲：silent_energy_threshold 是 RMS（sqrt(mean(x²))），
        # 与 chunking_core 的 hive_energy_threshold（MSE 量纲）不同，不可直接比较
        self.max_silence_ratio = max_silence_ratio
        self.min_duration = min_duration
        self.shorten_long_silences_flag = shorten_long_silences_flag
        self.max_silence_duration = max_silence_duration
        self.target_silence_duration = target_silence_duration
        self.target_silence_range = (target_silence_min, target_silence_max)
        self.use_random_silence_duration = use_random_silence_duration
        # 不持有共享 random.Random 实例：process_audio 被多线程并发调用
        # （_process_in_threads），共享 RNG 会交错消费随机流导致结果不可复现；
        # 每次调用按 seed + record 上下文 per-call 派生独立随机流
        self._random_seed = random_seed
        self.abs_min_energy = abs_min_energy
        self.silent_energy_threshold = silent_energy_threshold
        # |mean| 上限：超过即判为解码/格式伪影（有/无符号误读等），
        # 与 D2 load_audio 的质检口径一致；None 关闭检查
        self.dc_offset_threshold = dc_offset_threshold
    
    def process_audio(
        self,
        audio: np.ndarray,
        sr: int,
        rng_context: Optional[str] = None,
    ) -> Dict:
        """静音检测与可选 shorten 均在内存中完成，不读写磁盘。用于 Step1 加速：先判静音再决定是否写盘。

        rng_context：可选的记录上下文（如 record_id）。配置 random_seed 时按
        ``seed:rng_context`` per-call 派生独立随机流——同一记录跨运行结果
        一致，且与线程调度无关（共享 RNG 实例的多线程交错会破坏可复现性）。
        """
        # 统一 float32：float64 会让整批 chunk 以双倍体积 pickle 进 VAD 进程池
        if len(audio.shape) > 1:
            audio = audio.astype(np.float32).mean(axis=1)
        else:
            audio = audio.astype(np.float32)
        duration = len(audio) / sr
        # DC offset 检查：|mean| 超阈值说明是解码/格式伪影（有/无符号误读、
        # 文件半段损坏等），不是有效噪声。与 D2 load_audio 质检同一口径，
        # 提前到 B2 拦截，避免坏 chunk 白烧下游 VAD/LLM 算力
        dc_offset = float(np.mean(audio)) if len(audio) else 0.0
        dc_rejected = (
            self.dc_offset_threshold is not None
            and abs(dc_offset) > self.dc_offset_threshold
        )
        # B2 的三个帧级判定共享一次 RMS 计算；shorten 使用默认绝对阈值时
        # 与 silence_ratio 的静音区域完全相同，可直接复用，否则保持各自阈值语义。
        times, energies = compute_energy_per_frame(audio, sr)
        silence_regions, active_frames = _frame_silence_metrics(
            times, energies, sr, frame_duration=0.025,
            abs_min_threshold=self.abs_min_energy,
            active_threshold=self.silent_energy_threshold,
        )
        total_silence = sum(end - start for start, end in silence_regions)
        silence_ratio = total_silence / duration if silence_regions and duration > 0 else 0.0
        is_silent = not energies or active_frames / len(energies) < 0.05
        passed = not dc_rejected and not is_silent and silence_ratio < self.max_silence_ratio and duration >= self.min_duration
        processed_audio = audio
        if passed and self.shorten_long_silences_flag:
            # per-call 派生随机流：seed 未配置时保持历史全局随机语义（rng=None）
            rng = None
            if self._random_seed is not None:
                rng = random.Random(f"{self._random_seed}:{rng_context or ''}")
            shortening_regions = silence_regions
            if self.abs_min_energy not in (None, 0.001):
                shortening_regions = _silence_regions_from_frames(
                    times, energies, sr, frame_duration=0.025,
                    abs_min_threshold=None,
                )
            processed_audio = shorten_long_silences(
                audio, sr,
                max_silence_duration=self.max_silence_duration,
                target_silence_duration=self.target_silence_duration,
                target_silence_range=self.target_silence_range,
                use_random=self.use_random_silence_duration,
                rng=rng,
                silence_regions=shortening_regions,
            )
        result = {
            'original_duration': duration,
            'sample_rate': sr,
            'silence_ratio': silence_ratio,
            'is_silent': is_silent,
            'dc_offset': dc_offset,
            'dc_offset_rejected': dc_rejected,
            'passed': passed,
        }
        if passed:
            result['processed_audio'] = processed_audio
        return result

    def process(self, audio_path: str,
                output_path: Optional[str] = None) -> Dict:
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        result = self.process_audio(audio, sr)
        result['audio_path'] = audio_path
        if result.get('passed') and output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, result['processed_audio'], sr)
            result['output_path'] = output_path
        return result
    
class SilenceFilterProcessor(Processor):
    """B2: 静音过滤 Processor。

    对输入音频检测静音比例，过滤静音过多或过短的片段；
    对通过者可选缩短过长静音段后落盘。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, stage_name="B2")
        self.num_workers = int(config.get("num_workers", 1))
        self.enable = config.get("enable", True)
        self.max_silence_ratio = float(config.get("max_silence_ratio", 0.9))
        self.min_chunk_duration_filter = float(config.get("min_chunk_duration_filter", 0.5))
        self.shorten_long_silences_flag = bool(config.get("shorten_long_silences", True))
        self.max_silence_duration = float(config.get("max_silence_duration", 2.0))
        self.target_silence_duration = float(config.get("target_silence_duration", 1.0))
        self.target_silence_min = float(config.get("target_silence_min", 0.5))
        self.target_silence_max = float(config.get("target_silence_max", 1.5))
        self.use_random_silence_duration = bool(config.get("use_random_silence_duration", True))
        self.abs_min_energy = config.get("abs_min_energy")
        # 帧 RMS 绝对阈值：低电平有效噪声（远场底噪等）场景建议调低到 0.001-0.002
        self.silent_energy_threshold = float(config.get("silent_energy_threshold", 0.005))
        # |mean| 上限：超过即判为解码/格式伪影，None 关闭检查
        self.dc_offset_threshold = config.get("dc_offset_threshold", 0.5)

        self._silence_processor = SilenceProcessor(
            max_silence_ratio=self.max_silence_ratio,
            min_duration=self.min_chunk_duration_filter,
            shorten_long_silences_flag=self.shorten_long_silences_flag,
            max_silence_duration=self.max_silence_duration,
            target_silence_duration=self.target_silence_duration,
            target_silence_min=self.target_silence_min,
            target_silence_max=self.target_silence_max,
            use_random_silence_duration=self.use_random_silence_duration,
            abs_min_energy=self.abs_min_energy,
            silent_energy_threshold=self.silent_energy_threshold,
            dc_offset_threshold=self.dc_offset_threshold,
            random_seed=config.get("random_seed"),
        )

    def _process_record(
        self,
        record: AudioRecord,
        output_audio_dir: str,
    ) -> AudioRecord:
        """处理单条记录。"""
        audio_path = record.audio.get("path")
        if not audio_path or not os.path.isfile(audio_path):
            record.mark_stage(self.stage_name, "rejected", extra={"reason": "missing_audio"})
            record.rejected_reason = "b2_missing_audio"
            return record

        try:
            audio, sr = sf.read(audio_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("B2 读取音频失败 %s: %s", audio_path, exc)
            record.mark_stage(self.stage_name, "rejected", extra={"reason": "read_error"})
            record.rejected_reason = "b2_read_error"
            return record

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        result = self._silence_processor.process_audio(audio, sr, rng_context=record.record_id)
        if not result.get("passed", True):
            record.mark_stage(
                self.stage_name,
                "rejected",
                extra={
                    "reason": "silence_filter",
                    "silence_ratio": result.get("silence_ratio"),
                    "is_silent": result.get("is_silent"),
                },
            )
            record.rejected_reason = "b2_silence_filter"
            return record

        dataset_dir = os.path.join(output_audio_dir, record.dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)
        # 输出名带源路径短 hash：同一数据集下不同目录的同名文件不会互相覆盖
        output_path = make_unique_output_path(dataset_dir, audio_path, suffix="_b2")
        processed_audio = result.get("processed_audio", audio)
        try:
            sf.write(output_path, processed_audio, sr)
        except Exception as exc:  # noqa: BLE001
            logger.warning("B2 写入音频失败 %s: %s", output_path, exc)
            record.mark_stage(self.stage_name, "rejected", extra={"reason": "write_error"})
            record.rejected_reason = "b2_write_error"
            return record

        record.audio["path"] = output_path
        record.audio["sr"] = sr
        record.audio["duration"] = float(len(processed_audio) / sr)
        # 上文已转单声道，channels 必须同步，避免下游按错误声道数解读
        record.audio["channels"] = 1
        record.mark_stage(
            self.stage_name,
            "passed",
            extra={
                "silence_ratio": result.get("silence_ratio"),
                "original_duration": result.get("original_duration"),
            },
        )
        return record

    def process(self, input_manifest: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_manifest = os.path.join(output_dir, "b2_manifest.jsonl")
        output_audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(output_audio_dir, exist_ok=True)

        total = 0
        passed = 0
        rejected = 0

        def _process_one(rec: AudioRecord) -> AudioRecord:
            if self.should_skip(rec):
                return rec
            # B1 未通过的记录不得进入 B2，防止后续阶段重新放行。
            b1 = rec.stages.get("B1")
            if b1 is not None and b1.get("status") != "passed":
                rec.mark_stage(
                    self.stage_name,
                    "skipped",
                    extra={"reason": "prev_stage_B1_not_passed"},
                )
                return rec
            if not self.enable:
                rec.mark_stage(self.stage_name, "passed", extra={"enabled": False})
                return rec
            return self._process_record(rec, output_audio_dir)

        def record_generator() -> Iterator[AudioRecord]:
            nonlocal total, passed, rejected
            # 流式读取 + 线程池处理，避免百万级文件全量驻留内存
            for rec in _process_in_threads(read_manifest(input_manifest), _process_one, self.num_workers):
                total += 1
                if rec.is_stage_passed(self.stage_name):
                    passed += 1
                else:
                    rejected += 1
                yield rec

        count = write_manifest(record_generator(), output_manifest)
        logger.info(
            "B2 静音过滤完成：%d 条记录，通过 %d，拒绝 %d（workers=%d），输出 %s",
            total,
            passed,
            rejected,
            self.num_workers,
            output_manifest,
        )
        return output_manifest
