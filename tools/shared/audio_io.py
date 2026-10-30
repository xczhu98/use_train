"""Shared audio I/O and basic processing utilities.

Used by both audio_filter_pipeline and noise_filter_pipeline.
Keeps a single, well-tested implementation of common operations like loading,
saving, resampling, splitting and silence detection.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)
_SOXR_AVAILABLE: bool | None = None
_SOX_AVAILABLE: bool | None = None


def _ffmpeg_timeout_sec() -> int:
    """ffmpeg 子进程超时（秒），可用 ``USE_TRAIN_FFMPEG_TIMEOUT_SEC`` 覆盖。"""
    raw = os.environ.get("USE_TRAIN_FFMPEG_TIMEOUT_SEC", "7200")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"USE_TRAIN_FFMPEG_TIMEOUT_SEC must be an integer, got {raw!r}"
        ) from exc


def _sox_timeout_sec() -> int:
    """SoX 子进程超时（秒），可用 ``USE_TRAIN_SOX_TIMEOUT_SEC`` 覆盖。"""
    raw = os.environ.get("USE_TRAIN_SOX_TIMEOUT_SEC", "600")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"USE_TRAIN_SOX_TIMEOUT_SEC must be an integer, got {raw!r}"
        ) from exc


class AudioValidationError(ValueError):
    """load_audio 质量检查失败的结构化异常。

    ``reason`` 为机器可读的失败类别（zero_length/nan_inf/peak_overflow/
    too_silent/dc_offset/read_error/invalid_sr），供调用方精确归类，
    避免解析中文错误消息。仍是 ``ValueError`` 子类，兼容既有 except。
    """

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


def _soxr_available() -> bool:
    global _SOXR_AVAILABLE
    if _SOXR_AVAILABLE is not None:
        return _SOXR_AVAILABLE
    try:
        import soxr  # noqa: F401
        _SOXR_AVAILABLE = True
    except ImportError:
        _SOXR_AVAILABLE = False
    return _SOXR_AVAILABLE


def _sox_available() -> bool:
    global _SOX_AVAILABLE
    if _SOX_AVAILABLE is not None:
        return _SOX_AVAILABLE
    sox_path = os.environ.get("SOX_BINARY", "sox")
    try:
        proc = subprocess.run(
            [sox_path, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5,
        )
        _SOX_AVAILABLE = proc.returncode == 0
    except Exception:  # noqa: BLE001
        _SOX_AVAILABLE = False
    return _SOX_AVAILABLE


def _resample_with_sox(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """使用 SoX 命令行做高质量重采样（rate -v -I）。"""
    sox_path = os.environ.get("SOX_BINARY", "sox")
    import tempfile

    with tempfile.TemporaryDirectory(prefix="sox_resample_") as tmpdir:
        in_path = os.path.join(tmpdir, "input.wav")
        out_path = os.path.join(tmpdir, "output.wav")
        # 使用 FLOAT subtype 避免 PCM_16 量化损失，保持重采样保真度
        sf.write(in_path, audio, orig_sr, subtype="FLOAT")
        cmd = [
            sox_path,
            in_path,
            out_path,
            "rate",
            "-v",  # very high quality
            "-I",  # linear phase
            str(target_sr),
        ]
        timeout_sec = _sox_timeout_sec()
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            # 坏输入可能让 SoX 无限挂起；超时归入重采样失败，由调用方降级到 scipy。
            raise RuntimeError(
                f"SoX 重采样超时（>{timeout_sec}s）: {orig_sr} -> {target_sr}"
            ) from exc
        out_audio, out_sr = sf.read(out_path, dtype="float32")
        if out_sr != target_sr:
            raise RuntimeError(f"SoX 输出采样率 {out_sr} 与目标 {target_sr} 不一致")
        # SoX 逐通道处理，多声道保持 (samples, channels) 原样返回，不做静默混单声道
        return out_audio.astype(np.float32)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """重采样到目标采样率。

    优先级：soxr（纯内存）→ librosa → SoX 子进程（rate -v -I）→ scipy。
    多声道输入约定为 soundfile 布局 ``(samples, channels)``：
    soxr>=1.0 直接接受该布局；librosa 沿最后一轴重采样，
    多声道时先转置为 ``(channels, samples)`` 处理后再转回。
    """
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(
            f"resample_audio: orig_sr and target_sr must be positive, "
            f"got orig_sr={orig_sr}, target_sr={target_sr}"
        )
    if orig_sr == target_sr:
        return audio
    multichannel = audio.ndim > 1
    if _soxr_available():
        try:
            import soxr
            # soxr>=1.0 的 2D 输入约定为 (samples, channels)，与 soundfile 一致
            return soxr.resample(audio, orig_sr, target_sr)
        except Exception as e:
            logger.warning("soxr resample failed: %s, falling back to librosa", e)
    try:
        import librosa
        # librosa 沿最后一轴重采样，多声道转置为 (channels, samples)
        work = audio.T if multichannel else audio
        out = librosa.resample(
            work.astype(np.float64), orig_sr=orig_sr, target_sr=target_sr
        ).astype(np.float32)
        return out.T if multichannel else out
    except Exception as e:
        logger.warning("librosa resample failed: %s, falling back to SoX", e)
    if _sox_available():
        try:
            return _resample_with_sox(audio, orig_sr, target_sr)
        except Exception as e:
            logger.warning("SoX resample failed: %s, falling back to scipy", e)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(orig_sr, target_sr)
    # resample_poly 默认沿 axis=0，与 soundfile 的 (samples, channels) 布局一致
    return resample_poly(audio.astype(np.float64), target_sr // g, orig_sr // g).astype(np.float32)


def _decode_with_ffmpeg(audio_path: str) -> Tuple[np.ndarray, int]:
    """用 ffmpeg 解码（wav 字节流管道）并读回。

    libsndfile/libmpg123 对损坏 MP3 帧容错较差（可能直接抛错），ffmpeg 对坏帧
    按跳过/静音处理且 stderr 由子进程捕获，不会污染主进程日志。
    """
    import io

    timeout_sec = _ffmpeg_timeout_sec()
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", audio_path, "-f", "wav", "-"],
            capture_output=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        # 坏文件可能让 ffmpeg 无限挂起；归类为解码失败，由 load_audio 统一按 read_error 处理。
        raise ValueError(
            f"ffmpeg 解码超时（>{timeout_sec}s）: {audio_path}"
        ) from exc
    if proc.returncode != 0 or not proc.stdout:
        tail = proc.stderr.decode("utf-8", errors="ignore")[-300:]
        raise ValueError(f"ffmpeg 解码失败 (code {proc.returncode}): {tail}")
    audio, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    return audio, int(sr)


def load_audio(
    audio_path: str,
    max_peak: float = 1.05,
    min_mean_abs: float = 1e-6,
    dc_offset_threshold: float = 0.5,
    return_stats: bool = False,
) -> Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, Dict[str, Any]]]:
    """加载音频并做硬质量检查。

    返回 ``(audio, sr)``；若 ``return_stats=True`` 则返回 ``(audio, sr, stats)``。
    stats 包含 ``duration/rms_dbfs/peak_dbfs/dc_offset/zero_ratio``。
    """
    try:
        audio, sr = sf.read(audio_path, dtype="float32")
    except Exception as e:
        # libsndfile/libmpg123 对损坏 MP3 帧容错差：用 ffmpeg 兜底解码，
        # 只有兜底也失败才判为不可读
        try:
            audio, sr = _decode_with_ffmpeg(audio_path)
            logger.warning("soundfile 解码失败，已用 ffmpeg 兜底解码 %s: %s", audio_path, e)
        except Exception as e2:
            raise AudioValidationError(
                f"无法读取音频文件 {audio_path}: soundfile({e}); ffmpeg({e2})",
                reason="read_error",
            ) from e2

    if audio.size == 0:
        raise AudioValidationError(f"音频文件为空: {audio_path}", reason="zero_length")
    if sr <= 0:
        raise AudioValidationError(f"非法采样率 {sr}: {audio_path}", reason="invalid_sr")

    # 质量检查在混音前对原始多声道数据整体取 max，
    # 避免一声道削波、另一声道反相时 mean 混音后检查被绕过
    if not np.isfinite(audio).all():
        raise AudioValidationError(f"音频包含 NaN/Inf: {audio_path}", reason="nan_inf")

    peak = float(np.max(np.abs(audio)))
    if peak > max_peak:
        raise AudioValidationError(
            f"音频幅度越界 peak={peak:.3f} > {max_peak}: {audio_path}", reason="peak_overflow"
        )

    mean_abs = float(np.mean(np.abs(audio)))
    if mean_abs < min_mean_abs:
        raise AudioValidationError(
            f"音频过静音 mean_abs={mean_abs:.3e} < {min_mean_abs}: {audio_path}", reason="too_silent"
        )

    dc_offset = float(np.mean(audio))
    if abs(dc_offset) > dc_offset_threshold:
        raise AudioValidationError(
            f"音频 DC offset={dc_offset:.3f} > {dc_offset_threshold}: {audio_path}",
            reason="dc_offset",
        )

    # 混音只用于返回值与统计量
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration = float(audio.size / sr)
    # 使用真 RMS。
    rms = float(np.sqrt(np.mean(audio ** 2)))
    rms_dbfs = 20.0 * np.log10(rms + 1e-12)
    peak_dbfs = 20.0 * np.log10(peak + 1e-12)
    zero_ratio = float(np.mean(np.abs(audio) < 1e-12))
    stats = {
        "duration": duration,
        "rms_dbfs": float(rms_dbfs),
        "peak_dbfs": float(peak_dbfs),
        "dc_offset": dc_offset,
        "zero_ratio": zero_ratio,
    }

    if return_stats:
        return audio, sr, stats
    return audio, sr


def save_audio(
    audio: np.ndarray,
    sr: int,
    output_path: str,
    subtype: Optional[str] = None,
):
    """保存音频到文件。

    写入前检查 NaN/Inf 并将样本 clip 到 [-1, 1]：libsndfile 在 float -> PCM16
    时是截断而非 clip，|x|>1.0 的样本会回绕成爆音。
    ``subtype`` 默认 None，保持 soundfile 按扩展名推断的默认格式
    （.wav 为 PCM_16），不改变下游现有输出格式。
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if not np.isfinite(audio).all():
        raise ValueError(f"音频包含 NaN/Inf，拒绝写入: {output_path}")
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(output_path, audio, sr, subtype=subtype)


def convert_audio_format(
    input_path: str,
    output_path: str,
    target_sr: Optional[int] = None,
    target_channels: int = 1,
    target_dtype: str = "PCM_16",
):
    """使用 ffmpeg 转换音频格式/采样率/通道数。"""
    valid_dtypes = {"PCM_16", "PCM_24", "PCM_32"}
    if target_dtype not in valid_dtypes:
        raise ValueError(
            f"target_dtype must be one of {valid_dtypes}, got {target_dtype}"
        )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # PCM 编码只能写入 WAV 容器，必须校验输出扩展名。
    if Path(output_path).suffix.lower() != ".wav":
        raise ValueError(
            f"convert_audio_format: PCM 输出仅支持 .wav 容器, got {output_path}"
        )

    cmd = ["ffmpeg", "-y", "-i", input_path]

    if target_sr:
        cmd.extend(["-ar", str(target_sr)])

    if target_channels:
        cmd.extend(["-ac", str(target_channels)])

    if target_dtype == "PCM_16":
        cmd.extend(["-acodec", "pcm_s16le", "-f", "wav"])
    elif target_dtype == "PCM_24":
        cmd.extend(["-acodec", "pcm_s24le", "-f", "wav"])
    elif target_dtype == "PCM_32":
        cmd.extend(["-acodec", "pcm_s32le", "-f", "wav"])

    cmd.append(output_path)

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            timeout=_ffmpeg_timeout_sec(),
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        logger.error("ffmpeg failed for %s -> %s:\n%s", input_path, output_path, stderr)
        raise


def split_audio(
    audio: np.ndarray,
    sr: int,
    chunk_duration: float = 10.0,
    min_chunk_duration: float = 2.0,
) -> List[Tuple[np.ndarray, float, float]]:
    """按固定步长分片，无重叠。

    分片规则：步长 = chunk_duration；尾段若时长 < min_chunk_duration 则不产出（直接丢弃）。
    整文件时长 < min_chunk_duration 时返回空列表。
    """
    chunk_samples = int(chunk_duration * sr)
    min_samples = int(min_chunk_duration * sr)

    chunks = []
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i : i + chunk_samples]

        if len(chunk) >= min_samples:
            start_time = i / sr
            end_time = (i + len(chunk)) / sr
            chunks.append((chunk, start_time, end_time))

    return chunks


def detect_silence(
    audio: np.ndarray,
    sr: int,
    frame_duration: float = 0.025,
    hop_duration: float = 0.010,
    energy_threshold: float = 0.01,
) -> List[Tuple[float, float]]:
    """基于能量检测静音段。"""
    frame_samples = int(frame_duration * sr)
    hop_samples = int(hop_duration * sr)

    if frame_samples <= 0 or hop_samples <= 0 or len(audio) < frame_samples:
        return []

    # 向量化计算逐帧 RMS 能量；range 终点 +1 确保覆盖最后一帧
    starts = np.arange(0, len(audio) - frame_samples + 1, hop_samples)
    windows = np.lib.stride_tricks.sliding_window_view(audio, frame_samples)[starts]
    rms = np.sqrt(np.mean(windows ** 2, axis=1))

    threshold = max(float(np.mean(rms)) * 0.5, energy_threshold)

    silence_regions = []
    in_silence = False
    silence_start = 0.0

    for start_sample, energy in zip(starts.tolist(), rms.tolist()):
        time = start_sample / sr
        if energy < threshold and not in_silence:
            silence_start = time
            in_silence = True
        elif energy >= threshold and in_silence:
            silence_regions.append((silence_start, time))
            in_silence = False

    if in_silence:
        silence_regions.append((silence_start, len(audio) / sr))

    return silence_regions


def calculate_silence_ratio(
    audio: np.ndarray,
    sr: int,
    frame_duration: float = 0.025,
    hop_duration: float = 0.010,
    energy_threshold: float = 0.01,
) -> float:
    silence_regions = detect_silence(audio, sr, frame_duration, hop_duration, energy_threshold)
    total_silence = sum(end - start for start, end in silence_regions)
    total_duration = len(audio) / sr
    return total_silence / total_duration if total_duration > 0 else 0.0


def is_pure_silence(
    audio: np.ndarray,
    sr: int,
    silence_threshold: float = 0.95,
    energy_threshold: float = 0.005,
) -> bool:
    silence_ratio = calculate_silence_ratio(audio, sr, energy_threshold=energy_threshold)
    return silence_ratio > silence_threshold


def shorten_silence(
    audio: np.ndarray,
    sr: int,
    max_silence_duration: float = 1.0,
    min_silence_duration: float = 0.3,
    energy_threshold: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """将超过 max_silence_duration 的静音段缩短并返回新数组。

    对过长的静音段保留随机时长 [min_silence_duration, max_silence_duration*0.5]，
    删除其余静音样本，因此返回数组长度可能短于输入。

    ``rng`` 为 None 时使用全局 ``np.random``（保持历史行为）；
    需要可复现结果时传入 ``np.random.default_rng(seed)``。
    """
    silence_regions = detect_silence(audio, sr, energy_threshold=energy_threshold)

    if not silence_regions:
        return audio.copy()

    # 默认沿用全局随机源，传入 rng 时可复现
    uniform = rng.uniform if rng is not None else np.random.uniform

    segments = []
    prev_end_sample = 0
    for start, end in silence_regions:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        # 保留当前静音段之前的非静音部分
        if start_sample > prev_end_sample:
            segments.append(audio[prev_end_sample:start_sample])

        duration = end - start
        if duration > max_silence_duration:
            new_duration = float(
                uniform(min_silence_duration, max_silence_duration * 0.5)
            )
            keep_samples = max(1, int(new_duration * sr))
            segments.append(audio[start_sample : start_sample + keep_samples])
        else:
            segments.append(audio[start_sample:end_sample])
        prev_end_sample = end_sample

    # 追加最后一个静音段之后的非静音尾部
    if prev_end_sample < len(audio):
        segments.append(audio[prev_end_sample:])

    if not segments:
        return audio.copy()
    return np.concatenate(segments)


class AudioChunk:
    """代表音频中的一个 chunk，附带元信息。"""

    def __init__(
        self,
        audio: np.ndarray,
        sr: int,
        start_time: float,
        end_time: float,
        source_path: str = "",
    ):
        self.audio = audio
        self.sr = sr
        self.start_time = start_time
        self.end_time = end_time
        self.source_path = source_path
        self.duration = end_time - start_time
        # detect_silence 结果缓存（按 energy_threshold 区分），
        # 避免 silence_ratio / is_silent 各自完整重算一遍
        self._silence_cache: Dict[float, List[Tuple[float, float]]] = {}

    def _detect_silence(self, energy_threshold: float) -> List[Tuple[float, float]]:
        if energy_threshold not in self._silence_cache:
            self._silence_cache[energy_threshold] = detect_silence(
                self.audio, self.sr, energy_threshold=energy_threshold
            )
        return self._silence_cache[energy_threshold]

    def _silence_ratio(self, energy_threshold: float) -> float:
        regions = self._detect_silence(energy_threshold)
        total_silence = sum(end - start for start, end in regions)
        total_duration = len(self.audio) / self.sr
        return total_silence / total_duration if total_duration > 0 else 0.0

    @property
    def silence_ratio(self) -> float:
        # 与 calculate_silence_ratio 默认参数保持一致
        return self._silence_ratio(energy_threshold=0.01)

    @property
    def is_silent(self) -> bool:
        # 与 is_pure_silence 默认参数保持一致
        return self._silence_ratio(energy_threshold=0.005) > 0.95

    def save(self, output_path: str):
        save_audio(self.audio, self.sr, output_path)


def process_audio_chunks(
    audio_path: str,
    chunk_duration: float = 10.0,
    min_chunk_duration: float = 2.0,
    filter_silent: bool = True,
    max_silence_ratio: float = 0.9,
) -> List[AudioChunk]:
    """加载音频并切分，可选过滤静音 chunk。"""
    try:
        audio, sr = load_audio(audio_path)
    except Exception as exc:
        logger.warning("跳过无法读取的音频 %s: %s", audio_path, exc)
        return []

    chunks = split_audio(audio, sr, chunk_duration, min_chunk_duration)

    result_chunks = []
    for chunk, start, end in chunks:
        audio_chunk = AudioChunk(chunk, sr, start, end, audio_path)
        if filter_silent and audio_chunk.is_silent:
            continue
        if audio_chunk.silence_ratio > max_silence_ratio:
            continue
        result_chunks.append(audio_chunk)

    return result_chunks


def get_audio_info(audio_path: str, *, compute_silence: bool = False) -> Dict[str, Any]:
    """获取音频文件元信息。

    默认使用 ``soundfile.info`` 快速读取，不加载音频数据。
    如需 ``silence_ratio``，请将 ``compute_silence`` 设为 True，此时会完整加载音频。
    """
    info = sf.info(audio_path)
    result = {
        "path": audio_path,
        "sample_rate": info.samplerate,
        "duration": info.duration,
        "channels": info.channels,
        "samples": info.frames,
        "silence_ratio": None,
    }
    if compute_silence:
        audio, sr = load_audio(audio_path)
        result["silence_ratio"] = calculate_silence_ratio(audio, sr)
    return result


_UNSAFE_STEM_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_stem(stem: str) -> str:
    """清洗文件名片段，只保留安全字符。"""
    cleaned = _UNSAFE_STEM_CHARS.sub("_", stem).strip("_.")
    return cleaned or "unnamed"


def make_unique_output_path(
    output_dir: str,
    src_path: str,
    suffix: str = "",
    ext: Optional[str] = None,
    shard: bool = False,
) -> str:
    """生成带源路径哈希的确定性输出路径。"""
    stem, src_ext = os.path.splitext(os.path.basename(src_path))
    used_ext = ext if ext else src_ext
    short_hash = hashlib.sha256(os.path.abspath(src_path).encode("utf-8")).hexdigest()[:16]
    name = f"{_sanitize_stem(stem)}{suffix}_{short_hash}{used_ext}"
    if shard:
        shard2 = hashlib.sha256(name.encode("utf-8")).hexdigest()[:2]
        return os.path.join(output_dir, shard2, name)
    return os.path.join(output_dir, name)


def load_audio_robust(
    audio_path: str,
    max_peak: float = 1.2,
    normalize_peak: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """加载音频；峰值越界时归一化后重试。"""
    try:
        return load_audio(audio_path, max_peak=max_peak)
    except ValueError as exc:
        msg = str(exc)
        if "幅度越界" not in msg and "peak" not in msg.lower():
            raise

    logger.warning("音频峰值越界，尝试 peak normalization: %s", audio_path)
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.size == 0:
        raise ValueError(f"音频文件为空: {audio_path}")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    peak = float(np.max(np.abs(audio)))
    if peak <= 0 or not np.isfinite(peak):
        raise ValueError(f"音频峰值异常: {audio_path}")
    audio = audio * (normalize_peak / peak)
    logger.warning("已把 %s 从 peak=%.3f 归一化到 %.3f", audio_path, peak, normalize_peak)
    return audio.astype(np.float32), int(sr)


def save_audio_atomically(audio: np.ndarray, sr: int, dst_path: str) -> None:
    """先写同目录临时文件，再原子替换最终音频。"""
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(dst_path)}.", suffix=os.path.splitext(dst_path)[1],
        dir=os.path.dirname(dst_path) or ".",
    )
    os.close(fd)
    try:
        save_audio(audio, sr, tmp_path)
        os.replace(tmp_path, dst_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def copy_or_convert_audio(
    src_path: str,
    dst_path: str,
    target_sr: Optional[int] = None,
    max_peak: Optional[float] = None,
) -> None:
    """复制音频到目标路径，必要时转换采样率。"""
    audio, sr = load_audio_robust(src_path, max_peak=max_peak or 1.2)
    if target_sr is not None and sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(dst_path)}.", suffix=os.path.splitext(dst_path)[1],
        dir=os.path.dirname(dst_path) or ".",
    )
    os.close(fd)
    try:
        save_audio(audio, sr, tmp_path)
        os.replace(tmp_path, dst_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
