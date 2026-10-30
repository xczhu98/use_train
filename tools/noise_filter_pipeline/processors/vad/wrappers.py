"""VAD wrapper 与工厂（从 processors/vad.py 拆出）。

注意：本文件位于 processors/vad/ 下，比原 processors/vad.py 深一层，
所有基于 ``__file__`` 的 use_train 根 / third_party 路径推导均已多上一层
``os.path.dirname``。
"""
import hashlib
import logging
import os
import re
import sys
import tempfile
import urllib.request
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlsplit

import numpy as np
import soundfile as sf

from .bootstrap import (
    _get_default_nemo_onnx_path,
    _get_or_download_fireredvad_root,
    _suppress_stdout_stderr,
)

logger = logging.getLogger(__name__)


def _load_audio_for_vad(path: str):
    """用 soundfile 读取 VAD 输入，避免 torchaudio 新版对 torchcodec 的运行时依赖。"""
    import torch

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    return torch.from_numpy(np.ascontiguousarray(audio.T)), int(sample_rate)


def _pcm16_roundtrip_arrays(audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """返回与 libsndfile PCM16 WAV 写读一致的 int16 与 float32 数组。"""
    array = np.asarray(audio, dtype=np.float32)
    array = np.nan_to_num(array, nan=-1.0, posinf=1.0, neginf=-1.0)
    scaled = np.floor(np.clip(array, -1.0, 1.0) * 32768.0)
    pcm16 = np.clip(scaled, -32768, 32767).astype(np.int16)
    quantized = pcm16.astype(np.float32) / 32768.0
    return pcm16, quantized


def _is_array_capability_error(exc: Exception) -> bool:
    """旧第三方版本不支持数组输入时允许一次能力降级。"""
    return isinstance(exc, (TypeError, AttributeError, NotImplementedError))


class VADResult:
    def __init__(self, has_speech: bool, speech_timestamps: List[Tuple[float, float]],
                 speech_ratio: float, confidence: float, vad_type: str = "unknown",
                 frame_probs: Optional[List[float]] = None):
        self.has_speech = has_speech
        self.speech_timestamps = speech_timestamps
        self.speech_ratio = speech_ratio
        self.confidence = confidence
        self.vad_type = vad_type
        # 每帧语音概率，用于后续用不同 speech_threshold 重算 speech_ratio（仅部分 VAD 如 Silero 有）
        self.frame_probs = frame_probs if frame_probs is not None else None

    def __repr__(self):
        return f"VADResult({self.vad_type}, has_speech={self.has_speech}, speech_ratio={self.speech_ratio:.2f})"

    def get_result_dict(self) -> Dict:
        """序列化为 dict，供 B3 写入 manifest 与 B4 策略引擎重算。"""
        return {
            'vad_type': self.vad_type,
            'has_speech': self.has_speech,
            'speech_ratio': self.speech_ratio,
            'confidence': self.confidence,
            'frame_probs': list(self.frame_probs) if self.frame_probs is not None else None,
        }


class FireRedVADWrapper:
    """FireRedVAD wrapper；新版走 PCM16 tuple，旧版回退 16k PCM16 WAV。"""
    needs_wav_file = False

    def __init__(self, model_dir: str = None, use_gpu: bool = False,
                 speech_threshold: float = 0.4, temp_dir: Optional[str] = None, **kwargs):
        self.use_gpu = use_gpu
        self.speech_threshold = speech_threshold
        self.model_dir = model_dir
        self.temp_dir = temp_dir
        self.vad = None
        self._array_input_supported: Optional[bool] = None
        self._load_model(**kwargs)

    def _load_model(self, smooth_window_size: int = 5,
                    min_speech_frame: int = 20,
                    max_speech_frame: int = 2000,
                    min_silence_frame: int = 20,
                    merge_silence_frame: int = 0,
                    extend_speech_frame: int = 0,
                    chunk_max_frame: int = 30000):
        try:
            import torch
            from fireredvad import FireRedVad, FireRedVadConfig
            # FireRedVAD calls .cuda() when use_gpu=True; avoid crash when CUDA unavailable
            use_gpu = bool(self.use_gpu and getattr(torch, 'cuda', None) and torch.cuda.is_available())
            vad_config = FireRedVadConfig(
                use_gpu=use_gpu,
                smooth_window_size=smooth_window_size,
                speech_threshold=self.speech_threshold,
                min_speech_frame=min_speech_frame,
                max_speech_frame=max_speech_frame,
                min_silence_frame=min_silence_frame,
                merge_silence_frame=merge_silence_frame,
                extend_speech_frame=extend_speech_frame,
                chunk_max_frame=chunk_max_frame)

            if not self.model_dir:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                processors_dir = os.path.dirname(current_dir)
                pipeline_root = os.path.dirname(processors_dir)
                use_train_root = os.path.dirname(os.path.dirname(pipeline_root))
                third_party = os.environ.get("USE_TRAIN_THIRD_PARTY_DIR") or os.path.join(use_train_root, "third_party_scripts")
                potential_paths = [
                    os.path.join(third_party, "FireRedVAD", "VAD"),
                    os.path.join(use_train_root, "pretrained_models", "FireRedVAD", "VAD"),
                    os.path.join(processors_dir, "pretrained_models", "FireRedVAD", "VAD"),
                ]
                for path in potential_paths:
                    if os.path.exists(path) and os.path.isdir(path):
                        self.model_dir = path
                        break
                if not self.model_dir:
                    root = _get_or_download_fireredvad_root()
                    if root:
                        self.model_dir = os.path.join(root, "VAD")
                if self.model_dir:
                    logger.debug("Using FireRedVAD model at %s", self.model_dir)
            if self.model_dir:
                self.vad = FireRedVad.from_pretrained(self.model_dir, vad_config)
            else:
                logger.warning("FireRedVAD model directory not specified")

        except ImportError as e:
            logger.warning("FireRedVAD 不可用，将使用备用 VAD。原因: %s", e)
            if "fireredvad" in str(e).lower():
                logger.warning("请用【当前运行所用的 Python】安装: %s -m pip install fireredvad[gpu]", sys.executable)
            self.vad = None
        except Exception as e:  # noqa: BLE001
            # 权重损坏/OOM 等非导入异常同样走 fallback：置 None 让 detect 走
            # 统一 fallback+strict 语义，避免异常传播后每个 batch 重试加载
            logger.warning(
                "FireRedVAD 模型加载失败，将使用备用 VAD。原因: %s(%s)",
                type(e).__name__, e,
            )
            self.vad = None

    @staticmethod
    def _build_result(result, probs) -> VADResult:
        if not isinstance(result, dict):
            result = result if hasattr(result, "__getitem__") else {}
        timestamps = result.get("timestamps", [])
        duration = float(result.get("dur", 0.0))
        total_speech = sum(float(end) - float(start) for start, end in timestamps)
        speech_ratio = (total_speech / duration) if duration > 0 else 0.0
        frame_probs = None
        confidence = 0.0
        if probs is not None:
            values = probs.tolist() if hasattr(probs, "tolist") else list(probs)
            frame_probs = [float(p) for p in values]
            confidence = max(frame_probs) if frame_probs else 0.0
        return VADResult(
            bool(timestamps), timestamps, speech_ratio, confidence,
            "firered_vad", frame_probs=frame_probs,
        )

    def detect(self, audio_path: str) -> VADResult:
        if self.vad is None:
            return FallbackVAD().detect(audio_path)
        try:
            with _suppress_stdout_stderr():
                result, probs = self.vad.detect(audio_path)
            return self._build_result(result, probs)
        except Exception as e:
            logger.debug("Error in FireRedVAD detection: %s", e)
            return FallbackVAD().detect(audio_path)

    def detect_array(self, audio: np.ndarray, sr: int,
                     shared_wav_path: Optional[str] = None) -> VADResult:
        """优先 PCM16 tuple；旧版本不支持时缓存能力并回退临时 WAV。"""
        if self.vad is None:
            return FallbackVAD().detect_array(audio, sr)
        if sr != 16000:
            import soxr
            audio = soxr.resample(audio.astype(np.float32, copy=False), sr, 16000)
            sr = 16000
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        pcm16, quantized = _pcm16_roundtrip_arrays(audio)
        supported = getattr(self, "_array_input_supported", None)
        if supported is not False:
            try:
                with _suppress_stdout_stderr():
                    result, probs = self.vad.detect((pcm16, sr))
                self._array_input_supported = True
                return self._build_result(result, probs)
            except Exception as exc:
                if not _is_array_capability_error(exc):
                    logger.debug("Error in FireRedVAD direct array detection: %s", exc)
                    return FallbackVAD().detect_array(quantized, sr)
                self._array_input_supported = False
                logger.debug("FireRedVAD 版本不支持数组输入，回退临时 WAV: %s", exc)
        if shared_wav_path is not None:
            return self.detect(shared_wav_path)
        import tempfile
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=self.temp_dir) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, pcm16, sr, subtype="PCM_16")
            return self.detect(tmp_path)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Silero VAD 模型获取：本地优先；下载必须 https + SHA-256（参照
# scripts/download_nemo_vad_onnx.py 的 https-only+SHA-256 范式）。
# 远端 torch.hub 仓库加载会执行仓库代码，已整体移除。
# ---------------------------------------------------------------------------
_SILERO_VAD_URL_ENV = "SILERO_VAD_URL"
_SILERO_VAD_SHA256_ENV = "SILERO_VAD_SHA256"
_SILERO_VAD_DEFAULT_URL = "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.jit"
_SILERO_VAD_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
_SILERO_DOWNLOAD_TIMEOUT_SECONDS = 60
_SILERO_MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024


def _silero_trusted_sha256() -> Optional[str]:
    """读取可信 SHA-256；未配置返回 None（调用方必须拒绝下载）。"""
    value = os.environ.get(_SILERO_VAD_SHA256_ENV, "").strip()
    if not value:
        return None
    if _SILERO_VAD_SHA256_RE.fullmatch(value) is None:
        raise ValueError(
            f"{_SILERO_VAD_SHA256_ENV} 必须是 64 位十六进制 SHA-256，当前值非法"
        )
    return value.lower()


def _validate_silero_download_url(url: str) -> None:
    """拒绝非 https / 无 hostname / 非法端口的下载地址。"""
    try:
        parsed = urlsplit(url)
        # 访问 port 以便同时拒绝非法端口等 malformed URL
        parsed.port
        hostname = parsed.hostname
    except ValueError as exc:
        raise ValueError(f"Silero VAD 下载地址无效: {exc}") from exc
    if parsed.scheme.lower() != "https":
        raise ValueError("Silero VAD 下载地址必须使用 https scheme")
    if not hostname:
        raise ValueError("Silero VAD 下载地址必须包含 hostname")


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """确保不会跟随 https → http 的降级重定向。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_silero_download_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download_silero_vad(url: str, dst_path: str) -> None:
    """下载到同目录临时文件 → 校验 SHA-256 → 原子发布。

    未配置可信摘要（``SILERO_VAD_SHA256``）时拒绝下载：不要在无法从本地证据
    得到可信摘要的情况下猜测/硬编码校验值。
    """
    trusted_sha256 = _silero_trusted_sha256()
    if trusted_sha256 is None:
        raise RuntimeError(
            f"拒绝下载 Silero VAD：未配置可信摘要 {_SILERO_VAD_SHA256_ENV}。"
            "请将校验过的 silero_vad.jit 放到 third_party_scripts/SileroVAD/，"
            f"或同时设置 {_SILERO_VAD_URL_ENV} 与 {_SILERO_VAD_SHA256_ENV} 后重试"
        )
    _validate_silero_download_url(url)
    target_dir = os.path.dirname(os.path.abspath(dst_path))
    os.makedirs(target_dir, exist_ok=True)
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=".silero_vad.", suffix=".download.tmp", dir=target_dir
        )
        os.close(fd)
        request = urllib.request.Request(url)
        opener = urllib.request.build_opener(_HttpsOnlyRedirectHandler())
        logger.info("Downloading Silero VAD model from %s ...", url)
        digest = hashlib.sha256()
        size = 0
        with opener.open(request, timeout=_SILERO_DOWNLOAD_TIMEOUT_SECONDS) as response:
            declared = response.headers.get("Content-Length")
            if declared is not None:
                if int(declared) > _SILERO_MAX_DOWNLOAD_BYTES:
                    raise ValueError("Silero VAD 下载文件超过大小上限")
            with open(tmp_path, "wb") as file_obj:
                while True:
                    chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > _SILERO_MAX_DOWNLOAD_BYTES:
                        raise ValueError("Silero VAD 下载文件超过大小上限")
                    file_obj.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != trusted_sha256:
            raise RuntimeError(
                "Silero VAD 下载文件 SHA-256 与 "
                f"{_SILERO_VAD_SHA256_ENV} 不匹配"
            )
        os.replace(tmp_path, dst_path)
        tmp_path = None
        logger.info("Silero VAD model saved to %s", dst_path)
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


class SileroVADWrapper:
    """Silero VAD wrapper：本地模型优先，下载需 https + SHA-256，无远端代码加载。

    ``strict_vad_models=True`` 时模型不可用直接抛 RuntimeError（fail-fast）；
    默认 False 保留 energy fallback，但留下明确 warning。注意当前
    ``MultiVADDetector``/``vad_processor`` 未把配置里的 strict_vad_models
    传入本类（超出本次改动范围），它们在结果层按 fallback 的 vad_type 判定失败。
    """

    def __init__(self, use_gpu: bool = False, threshold: float = 0.5,
                 strict_vad_models: bool = False):
        self.use_gpu = use_gpu
        self.threshold = threshold
        self.strict_vad_models = bool(strict_vad_models)
        self.model = None
        # 模型实际所在设备（加载时统一维护），_detect_waveform 据此搬运输入，
        # 避免 device mismatch
        self._device = "cpu"
        self._load_model()

    def _handle_load_failure(self, reason: str) -> None:
        """strict 下 fail-fast；非 strict 保持 energy fallback 并留下明确告警。"""
        if self.strict_vad_models:
            raise RuntimeError(
                "Silero VAD 模型不可用（strict_vad_models=true，禁止回退 energy VAD）: "
                f"{reason}"
            )
        logger.warning(
            "Silero VAD 不可用，将回退 energy VAD（strict_vad_models=false）: %s",
            reason,
        )
        self.model = None

    def _candidate_model_paths(self, torch) -> List[str]:
        """本地模型候选路径：third_party_scripts 优先，其次 torch hub 缓存目录。"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        use_train_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
        third_party = os.environ.get("USE_TRAIN_THIRD_PARTY_DIR") or os.path.join(use_train_root, "third_party_scripts")
        return [
            os.path.join(third_party, "SileroVAD", "silero_vad.jit"),
            os.path.join(torch.hub.get_dir(), "checkpoints", "silero_vad.jit"),
        ]

    def _resolve_model_path(self, torch) -> str:
        """返回本地可用模型路径；缺失时按 https+SHA-256 规则下载到缓存目录。"""
        candidates = self._candidate_model_paths(torch)
        for path in candidates:
            if os.path.isfile(path):
                return path
        cache_path = candidates[-1]
        url = (os.environ.get(_SILERO_VAD_URL_ENV) or _SILERO_VAD_DEFAULT_URL).strip()
        _download_silero_vad(url, cache_path)
        return cache_path

    def _load_model(self):
        try:
            import torch
        except Exception as e:  # noqa: BLE001
            self._handle_load_failure(f"import torch failed: {type(e).__name__}: {e}")
            return
        try:
            model_path = self._resolve_model_path(torch)
        except Exception as e:  # noqa: BLE001
            self._handle_load_failure(f"{type(e).__name__}: {e}")
            return
        try:
            # 只加载本地已校验文件；不做 torch.hub 远端仓库代码执行
            self.model = torch.jit.load(model_path, map_location="cpu")
            self.model.eval()
            if self.use_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()
                self._device = "cuda"
            logger.debug("Silero VAD loaded from %s", model_path)
        except Exception as e:  # noqa: BLE001
            self.model = None
            self._handle_load_failure(
                f"torch.jit.load failed for {model_path}: {type(e).__name__}: {e}"
            )

    def _detect_waveform(self, waveform: "torch.Tensor", sr: int) -> VADResult:
        """对已经加载并归一化到 16k 单声道的 waveform 做 Silero VAD。"""
        import torch

        if sr != 16000:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000

        # 仅对多声道输入做平均。
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze()
        # 模型在 cuda 上时输入必须同设备（否则 TorchScript 报 device mismatch
        # 并静默落入 energy fallback）；设备以加载时记录的 _device 为准，
        # jit 与 torch.hub 两条加载路径均已维护该字段
        device = getattr(self, "_device", "cpu")
        if not waveform.is_cuda and device != "cpu":
            waveform = waveform.to(device)

        speech_probs = []
        window_size_samples = 512
        stride_samples = window_size_samples // 2

        # Silero JIT 的状态必须按时间顺序逐窗传递，避免跨音频状态污染。
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()
        with torch.no_grad():
            for i in range(0, len(waveform) - window_size_samples + 1, stride_samples):
                p = self.model(waveform[i:i + window_size_samples].unsqueeze(0), 16000)
                speech_probs.append(float(p.squeeze()))

        if not speech_probs:
            return VADResult(False, [], 0.0, 0.0, "silero", frame_probs=None)
        return self._probs_to_result(speech_probs, len(waveform) / sr)

    # 窗口参数：512 样本窗，50% overlap（与 detect_arrays_batch 共享）
    _WINDOW_SAMPLES = 512
    _STRIDE_SAMPLES = 256

    def _probs_to_result(self, speech_probs: List[float], duration: float) -> VADResult:
        """由逐窗概率构建 VADResult（timestamps / speech_ratio / frame_probs）。"""
        stride_samples = self._STRIDE_SAMPLES
        speech_probs = np.array(speech_probs)
        is_speech = speech_probs > self.threshold

        speech_timestamps = []
        in_speech = False
        speech_start = 0.0

        for i, speech in enumerate(is_speech):
            time_sec = (i * stride_samples) / 16000
            if speech and not in_speech:
                speech_start = time_sec
                in_speech = True
            elif not speech and in_speech:
                speech_timestamps.append((speech_start, time_sec))
                in_speech = False
        if in_speech:
            speech_timestamps.append((speech_start, duration))

        total_speech = sum(end - start for start, end in speech_timestamps)
        speech_ratio = total_speech / duration if duration > 0 else 0.0
        frame_probs_list = speech_probs.tolist()
        return VADResult(
            has_speech=len(speech_timestamps) > 0,
            speech_timestamps=speech_timestamps,
            speech_ratio=speech_ratio,
            confidence=np.mean(speech_probs),
            vad_type="silero",
            frame_probs=frame_probs_list
        )

    def detect_arrays_batch(self, audios: List[np.ndarray], sr: int) -> List[VADResult]:
        """对多条数组做 lockstep 批量检测（batch 维 = 并行音频序列）。

        silero jit 的 batch 维语义就是"N 条并行序列各推进一个时间步"、
        _state 按序列独立演化，因此把 K 条音频按时间步 lockstep 喂入与
        逐条 detect_array 顺序前向的结果完全一致（短序列补零窗口占位，
        不参与输出）。前向次数从 sum(windows) 降到 max(windows)。
        """
        import torch

        # 与 _detect_waveform 相同的预处理口径：mono + 重采样到 16k + 设备对齐
        waves = []
        for audio in audios:
            w = torch.from_numpy(audio.astype(np.float32))
            if sr != 16000:
                import torchaudio
                w = torchaudio.functional.resample(w, sr, 16000)
            if w.dim() > 1 and w.shape[0] > 1:
                w = w.mean(dim=0, keepdim=True)
            w = w.squeeze()
            device = getattr(self, "_device", "cpu")
            if not w.is_cuda and device != "cpu":
                w = w.to(device)
            waves.append(w)

        ws, stride = self._WINDOW_SAMPLES, self._STRIDE_SAMPLES
        n_windows = [max(0, (len(w) - ws) // stride + 1) for w in waves]
        results: List[Optional[VADResult]] = [None] * len(waves)
        active = [i for i, n in enumerate(n_windows) if n > 0]
        for i in range(len(waves)):
            if i not in active:
                results[i] = VADResult(False, [], 0.0, 0.0, "silero", frame_probs=None)
        if not active:
            return results

        if hasattr(self.model, "reset_states"):
            self.model.reset_states()
        max_n = max(n_windows[i] for i in active)
        probs: Dict[int, List[float]] = {i: [] for i in active}
        with torch.no_grad():
            for t in range(max_n):
                batch = torch.stack([
                    waves[i][t * stride: t * stride + ws]
                    if t < n_windows[i]
                    else torch.zeros(ws, dtype=waves[i].dtype, device=waves[i].device)
                    for i in active
                ])
                out = self.model(batch, 16000).squeeze(-1).tolist()
                for j, i in enumerate(active):
                    if t < n_windows[i]:
                        probs[i].append(float(out[j]))

        for i in active:
            results[i] = self._probs_to_result(probs[i], len(waves[i]) / 16000)
        return results

    def detect(self, audio_path: str) -> VADResult:
        if self.model is None:
            return FallbackVAD().detect(audio_path)

        try:
            waveform, sr = _load_audio_for_vad(audio_path)
            return self._detect_waveform(waveform, sr)
        except Exception as e:
            logger.error("Error in Silero VAD: %s", e)
            return FallbackVAD().detect(audio_path)

    def detect_array(self, audio: np.ndarray, sr: int) -> VADResult:
        """直接从 numpy 数组做 Silero VAD。"""
        if self.model is None:
            return FallbackVAD().detect_array(audio, sr)

        try:
            import torch
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            waveform = torch.from_numpy(audio.astype(np.float32))
            return self._detect_waveform(waveform, sr)
        except Exception as e:
            logger.error("Error in Silero VAD array: %s", e)
            return FallbackVAD().detect_array(audio, sr)


class WebRTCVADWrapper:
    def __init__(self, mode: int = 3, sample_rate: int = 16000):
        self.mode = mode
        self.sample_rate = sample_rate
        self.frame_duration = 30

    def _detect_int16(self, audio: np.ndarray, sr: int) -> VADResult:
        """对已经归一化到 16k 单声道的 float32 音频做 WebRTC VAD。"""
        import webrtcvad

        vad = webrtcvad.Vad(mode=self.mode)

        audio = (audio * 32767).astype(np.int16)
        frame_samples = int(sr * self.frame_duration / 1000)
        frames = [audio[i:i+frame_samples] for i in range(0, len(audio), frame_samples)]

        speech_timestamps = []
        frame_probs_list = []
        in_speech = False
        speech_start = 0.0

        for i, frame in enumerate(frames):
            if len(frame) != frame_samples:
                continue
            is_speech = vad.is_speech(frame.tobytes(), sr)
            frame_probs_list.append(1.0 if is_speech else 0.0)
            time_sec = i * self.frame_duration / 1000
            if is_speech and not in_speech:
                speech_start = time_sec
                in_speech = True
            elif not is_speech and in_speech:
                speech_timestamps.append((speech_start, time_sec))
                in_speech = False
        if in_speech:
            speech_timestamps.append((speech_start, len(audio) / sr))

        total_speech = sum(end - start for start, end in speech_timestamps)
        duration = len(audio) / sr
        speech_ratio = total_speech / duration if duration > 0 else 0.0

        return VADResult(
            has_speech=len(speech_timestamps) > 0,
            speech_timestamps=speech_timestamps,
            speech_ratio=speech_ratio,
            confidence=1.0 if speech_ratio > 0.1 else 0.0,
            vad_type="webrtc",
            frame_probs=frame_probs_list if frame_probs_list else None
        )

    def detect(self, audio_path: str) -> VADResult:
        try:
            audio, sr = sf.read(audio_path)
            if sr != 16000:
                audio = self._resample(audio, sr, 16000)
                sr = 16000
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            return self._detect_int16(audio, sr)
        except ImportError:
            logger.warning("webrtcvad not installed")
            return FallbackVAD().detect(audio_path)
        except Exception as e:
            logger.debug("Error in WebRTC VAD: %s", e)
            return FallbackVAD().detect(audio_path)

    def detect_array(self, audio: np.ndarray, sr: int) -> VADResult:
        """直接从 numpy 数组做 WebRTC VAD。"""
        try:
            import webrtcvad
            if sr != 16000:
                audio = self._resample(audio, sr, 16000)
                sr = 16000
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            return self._detect_int16(audio, sr)
        except ImportError:
            logger.warning("webrtcvad not installed")
            return FallbackVAD().detect_array(audio, sr)
        except Exception as e:
            logger.debug("Error in WebRTC VAD array: %s", e)
            return FallbackVAD().detect_array(audio, sr)

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr <= 0 or len(audio) == 0:
            raise ValueError("_resample: orig_sr must be > 0 and audio non-empty")
        from tools.shared.audio_io import resample_audio
        return resample_audio(audio.astype(np.float64), orig_sr, target_sr)


class FunASRVADWrapper:
    """FunASR FSMN-VAD wrapper；逐条数组输入，旧版回退 PCM16 WAV。"""
    needs_wav_file = False

    def __init__(self, model_dir: str = None, use_gpu: bool = False,
                 threshold: float = 0.5, temp_dir: Optional[str] = None, **kwargs):
        self.use_gpu = use_gpu
        self.threshold = threshold
        self.model_dir = model_dir
        self.temp_dir = temp_dir
        self.model = None
        self._array_input_supported: Optional[bool] = None
        self._load_model()

    def _load_model(self):
        try:
            from funasr import AutoModel

            model_dir = self.model_dir
            if model_dir and not os.path.isdir(model_dir):
                # 配置的本地目录不存在时回退到 hub 默认模型，避免直接崩溃
                logger.warning(
                    "FunASR model_dir 不存在: %s，回退到 hub 默认模型", model_dir
                )
                model_dir = None
            if model_dir:
                # 显式指定 model_dir 时必须用它作为本地模型路径，
                # 不能再传 hub 模型名（否则 model_dir 被静默忽略）
                try:
                    self.model = AutoModel(
                        model=model_dir,
                        device="cuda" if self.use_gpu else "cpu",
                        disable_update=True,
                    )
                except Exception as e:  # noqa: BLE001
                    # 权重损坏/OOM 等加载失败置 None，走统一 fallback+strict
                    # 语义，避免异常传播后每个 batch 重试加载
                    logger.warning(
                        "Failed to load FunASR model from %s: %s(%s)",
                        model_dir, type(e).__name__, e,
                    )
                    self.model = None
            else:
                logger.warning("FunASR model directory not specified, using default")
                try:
                    self.model = AutoModel(
                        model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                        device="cuda" if self.use_gpu else "cpu",
                        disable_update=True,
                    )
                except Exception as e:
                    logger.warning("Failed to load FunASR model: %s", e)
                    self.model = None
        except ImportError:
            logger.warning("funasr not installed")
            self.model = None

    def _build_result(self, audio: np.ndarray, sr: int, generate_result) -> VADResult:
        """由 FunASR generate 结果构造 VADResult。"""
        if isinstance(generate_result, dict):
            value = generate_result.get('value', [])
        else:
            value = generate_result[0].get('value', []) if generate_result else []

        speech_timestamps = []
        for item in value:
            # FunASR FSMN VAD 的 value 项为 [start_ms, end_ms] 列表。
            if isinstance(item, dict):
                start = item.get('start', 0) / 1000.0
                end = item.get('end', 0) / 1000.0
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                start = float(item[0]) / 1000.0
                end = float(item[1]) / 1000.0
            else:
                continue
            if end > start:
                speech_timestamps.append((start, end))

        duration = len(audio) / sr
        total_speech = sum(end - start for start, end in speech_timestamps)
        speech_ratio = total_speech / duration if duration > 0 else 0.0
        frame_step = 0.02
        n_frames = max(1, int(duration / frame_step))
        # 帧中心单调递增：按起点排序并维护已开始区间的最大终点，
        # 将逐帧遍历所有 speech segment 降为单次线性扫描。
        sorted_timestamps = sorted(speech_timestamps, key=lambda item: item[0])
        next_segment = 0
        max_end = float("-inf")
        frame_probs_list = []
        for i in range(n_frames):
            t_center = (i + 0.5) * frame_step
            while (
                next_segment < len(sorted_timestamps)
                and sorted_timestamps[next_segment][0] <= t_center
            ):
                max_end = max(max_end, sorted_timestamps[next_segment][1])
                next_segment += 1
            # 使用 >= 保留旧实现 start <= t_center <= end 的闭区间语义。
            frame_probs_list.append(1.0 if max_end >= t_center else 0.0)

        return VADResult(
            has_speech=len(speech_timestamps) > 0,
            speech_timestamps=speech_timestamps,
            speech_ratio=speech_ratio,
            confidence=1.0 if speech_ratio > 0.1 else 0.0,
            vad_type="funasr",
            frame_probs=frame_probs_list
        )

    def detect(self, audio_path: str) -> VADResult:
        if self.model is None:
            return FallbackVAD().detect(audio_path)

        try:
            audio, sr = sf.read(audio_path)

            # 立体声必须先 downmix 为单声道（与 detect_array 口径一致）：
            # FireRedVAD/FunASR 均要求 16k 单声道输入
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            if sr != 16000:
                audio = self._resample(audio, sr, 16000)
                sr = 16000

            import tempfile
            tmp_path = None
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=self.temp_dir) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, audio, sr)

            try:
                result = self.model.generate(tmp_path, batch_size_s=300, hotword="")[0]
                return self._build_result(audio, sr, result)
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        except Exception as e:
            logger.error("Error in FunASR VAD: %s", e)
            return FallbackVAD().detect(audio_path)

    def detect_array(self, audio: np.ndarray, sr: int,
                     shared_wav_path: Optional[str] = None) -> VADResult:
        """优先逐条内存输入；旧版本不支持时缓存能力并回退 PCM16 WAV。"""
        if self.model is None:
            return FallbackVAD().detect_array(audio, sr)
        if sr != 16000:
            audio = self._resample(audio, sr, 16000)
            sr = 16000
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        pcm16, quantized = _pcm16_roundtrip_arrays(audio)
        supported = getattr(self, "_array_input_supported", None)
        if supported is not False:
            try:
                result = self.model.generate(
                    input=quantized, batch_size_s=300, hotword=""
                )[0]
                self._array_input_supported = True
                return self._build_result(quantized, sr, result)
            except Exception as exc:
                if not _is_array_capability_error(exc):
                    logger.error("Error in FunASR direct array detection: %s", exc)
                    return FallbackVAD().detect_array(quantized, sr)
                self._array_input_supported = False
                logger.debug("FunASR 版本不支持数组输入，回退临时 WAV: %s", exc)
        if shared_wav_path is not None:
            result = self.model.generate(
                shared_wav_path, batch_size_s=300, hotword=""
            )[0]
            return self._build_result(quantized, sr, result)
        import tempfile
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, dir=self.temp_dir
            ) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, pcm16, sr, subtype="PCM_16")
            result = self.model.generate(
                tmp_path, batch_size_s=300, hotword=""
            )[0]
            return self._build_result(quantized, sr, result)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr <= 0 or len(audio) == 0:
            raise ValueError("_resample: orig_sr must be > 0 and audio non-empty")
        # VAD 对重采样质量不敏感，直接用 soxr 纯内存重采样；
        # shared 的 resample_audio 走 SoX 路径，每个非 16k chunk
        # 都要写临时文件并 fork sox 子进程，开销过大
        import soxr
        return soxr.resample(audio.astype(np.float32, copy=False), orig_sr, target_sr)


class NeMoVADWrapper:
    def __init__(self, model_path: str = None, use_gpu: bool = False,
                 threshold: float = 0.5, **kwargs):
        self.use_gpu = use_gpu
        self.threshold = threshold
        self.model_path = model_path
        self.model = None
        # 模型实际所在设备（_detect_waveform 据此搬运输入，避免 device mismatch
        # 后落入 energy fallback；与 SileroVADWrapper 的 _device 口径一致）
        self._device = "cpu"
        self._load_model()

    def _load_model(self):
        try:
            import torch
            from nemo.collections.asr.models import EncDecClassificationModel

            try:
                self.model = EncDecClassificationModel.from_pretrained(model_name="vad_multilingual_marblenet")
            except Exception:
                try:
                    self.model = EncDecClassificationModel.from_pretrained(model_name="vad_telephony_marblenet")
                except Exception as e:
                    logger.warning("Failed to load NeMo VAD model: %s", e)
                    self.model = None
            # use_gpu 且 CUDA 可用时将模型放到 GPU。
            if (
                self.model is not None
                and self.use_gpu
                and torch.cuda.is_available()
            ):
                self.model = self.model.to("cuda")
                self._device = "cuda"
        except ImportError:
            logger.warning("nemo not installed")
            self.model = None

    def _detect_waveform(self, waveform: "torch.Tensor", sr: int) -> VADResult:
        """对已经加载并归一化到 16k 单声道的 waveform 做 NeMo VAD。"""
        import torch

        if sr != 16000:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 模型在 cuda 上时输入必须同设备（否则 VADIterator 报 device
        # mismatch，被外层 except 捕获后静默落入 energy fallback）；设备以
        # 加载时记录的 _device 为准，与 SileroVADWrapper 口径一致
        device = getattr(self, "_device", "cpu")
        if not waveform.is_cuda and device != "cpu":
            waveform = waveform.to(device)

        try:
            from nemo.collections.asr.parts.utils.offline_clustering import VADIterator
            vad_iterator = VADIterator(self.model)
            hypothesis = vad_iterator.transcribe(waveform)

            speech_timestamps = []
            if hasattr(hypothesis, 'timestamps'):
                for ts in hypothesis.timestamps:
                    if hasattr(ts, 'start') and hasattr(ts, 'end'):
                        speech_timestamps.append((ts.start, ts.end))

            duration = len(waveform[0]) / sr
            total_speech = sum(end - start for start, end in speech_timestamps)
            speech_ratio = total_speech / duration if duration > 0 else 0.0

            return VADResult(
                has_speech=len(speech_timestamps) > 0,
                speech_timestamps=speech_timestamps,
                speech_ratio=speech_ratio,
                confidence=1.0 if speech_ratio > 0.1 else 0.0,
                vad_type="nemo"
            )
        except Exception as e:
            logger.error("NeMo VAD inference failed: %s", e)
            raise

    def detect(self, audio_path: str) -> VADResult:
        if self.model is None:
            return FallbackVAD().detect(audio_path)

        try:
            waveform, sr = _load_audio_for_vad(audio_path)
            return self._detect_waveform(waveform, sr)
        except Exception as e:
            logger.error("Error in NeMo VAD: %s", e)
            return FallbackVAD().detect(audio_path)

    def detect_array(self, audio: np.ndarray, sr: int) -> VADResult:
        """直接从 numpy 数组做 NeMo VAD。"""
        if self.model is None:
            return FallbackVAD().detect_array(audio, sr)

        try:
            import torch
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            return self._detect_waveform(waveform, sr)
        except Exception as e:
            logger.error("Error in NeMo VAD array: %s", e)
            return FallbackVAD().detect_array(audio, sr)


# ---------------------------------------------------------------------------
# NeMo VAD ONNX（参考 AudioPipeline 04_filter_by_speech_vad.py，不依赖 nemo_toolkit）
# ---------------------------------------------------------------------------

class _NeMoMelPreprocessor:
    """NeMo VAD 所需的 Log-Mel 预处理器，仅用 PyTorch/torchaudio，不依赖 NeMo。"""

    def __init__(self, sr: int = 16000, n_fft: int = 512, win_length: int = 400,
                 hop_length: int = 160, n_mels: int = 80):
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self._mel = None
        self._window = None

    def _ensure_buffers(self, device):
        if self._mel is not None:
            return
        try:
            import torch
            import torchaudio
            self._mel = torchaudio.transforms.MelScale(
                n_mels=self.n_mels, sample_rate=self.sr, n_stft=self.n_fft // 2 + 1
            ).to(device)
            self._window = torch.hann_window(self.win_length, periodic=False, device=device)
        except ImportError:
            raise ImportError("NeMo ONNX VAD 需要 torch 与 torchaudio")

    def __call__(self, audio_signal):
        import torch
        device = audio_signal.device
        self._ensure_buffers(device)
        stft = torch.stft(
            audio_signal, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, center=True, window=self._window,
            return_complex=True,
        )
        magnitudes = stft.abs().pow(2)
        mel_spec = self._mel(magnitudes)
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-9))
        return log_mel


class NeMoONNXVADWrapper:
    """NeMo VAD ONNX 版，不依赖 nemo_toolkit。参考 AudioPipeline 04_filter_by_speech_vad.py。"""

    def __init__(self, onnx_path: str = None, threshold: float = 0.5, use_gpu: bool = False, **kwargs):
        self.threshold = threshold
        self.use_gpu = use_gpu
        self.session = None
        self.preprocessor = None
        self.device = None
        path = onnx_path or _get_default_nemo_onnx_path()
        if path and os.path.isfile(path):
            self._load_model(path)
        else:
            if "nemo_onnx" in str(onnx_path or "").lower() or onnx_path is None:
                logger.warning("NeMo ONNX VAD: 未找到模型，请将 ONNX 放到 third_party_scripts/NeMo-VAD-onnx/ 或设置 nemo_onnx_path")
            self.session = None

    def _load_model(self, onnx_path: str):
        try:
            import torch
            import onnxruntime
            self.device = torch.device("cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.use_gpu else ["CPUExecutionProvider"]
            self.session = onnxruntime.InferenceSession(onnx_path, providers=providers)
            self.preprocessor = _NeMoMelPreprocessor()
        except ImportError as e:
            logger.warning("NeMo ONNX VAD 需要 onnxruntime 与 torch: %s", e)
            self.session = None
        except Exception as e:
            logger.warning("NeMo ONNX VAD 加载失败 %s: %s", onnx_path, e)
            self.session = None

    def _detect_waveform(self, waveform: "torch.Tensor", sr: int) -> VADResult:
        """对已经加载并归一化到 16k 单声道的 waveform 做 NeMo ONNX VAD。"""
        import torch

        if sr != 16000:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.to(self.device)
        processed = self.preprocessor(waveform)
        input_names = [inp.name for inp in self.session.get_inputs()]
        ort_inputs = {input_names[0]: processed.cpu().numpy()}
        if len(input_names) > 1:
            length = torch.tensor([processed.shape[2]], dtype=torch.long).cpu().numpy()
            ort_inputs[input_names[1]] = length
        logits = self.session.run(None, ort_inputs)[0]
        # NeMo VAD 为二分类（非语音/语音），取第 1 列作为语音概率；
        # 输出类数不符时说明模型不符合预期，直接报错而非静默取错列
        if logits.shape[-1] != 2:
            raise RuntimeError(
                f"NeMo VAD 期望 2 类输出，实际 {logits.shape[-1]} 类，请检查模型"
            )
        probs = torch.softmax(torch.from_numpy(logits), dim=-1)[0, :, 1]
        vad_labels = (probs > self.threshold)
        speech_ratio = vad_labels.float().sum().item() / max(1, len(vad_labels))
        return VADResult(
            has_speech=speech_ratio > 0,
            speech_timestamps=[],
            speech_ratio=float(speech_ratio),
            confidence=1.0 if speech_ratio > 0.1 else 0.0,
            vad_type="nemo_onnx"
        )

    def detect(self, audio_path: str) -> VADResult:
        if self.session is None or self.preprocessor is None:
            return FallbackVAD().detect(audio_path)
        try:
            waveform, sr = _load_audio_for_vad(audio_path)
            return self._detect_waveform(waveform, sr)
        except Exception as e:
            logger.debug("NeMo ONNX VAD 推理失败: %s", e)
            return FallbackVAD().detect(audio_path)

    def detect_array(self, audio: np.ndarray, sr: int) -> VADResult:
        """直接从 numpy 数组做 NeMo ONNX VAD。"""
        if self.session is None or self.preprocessor is None:
            return FallbackVAD().detect_array(audio, sr)
        try:
            import torch
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            return self._detect_waveform(waveform, sr)
        except Exception as e:
            logger.debug("NeMo ONNX VAD array 推理失败: %s", e)
            return FallbackVAD().detect_array(audio, sr)


class FallbackVAD:
    def __init__(self, energy_threshold: float = 0.02):
        self.energy_threshold = energy_threshold

    def _detect_array_impl(self, audio: np.ndarray, sr: int) -> VADResult:
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        frame_size = int(sr * 0.025)
        hop_size = int(sr * 0.010)

        energy = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            energy.append(np.sqrt(np.mean(frame ** 2)))

        if not energy:
            return VADResult(False, [], 0.0, 0.0, "energy")

        threshold = np.mean(energy) * 1.5
        threshold = max(threshold, self.energy_threshold)

        is_speech = [e > threshold for e in energy]

        speech_timestamps = []
        in_speech = False
        speech_start = 0.0

        for i, speech in enumerate(is_speech):
            time_sec = i * hop_size / sr
            if speech and not in_speech:
                speech_start = time_sec
                in_speech = True
            elif not speech and in_speech:
                speech_timestamps.append((speech_start, time_sec))
                in_speech = False

        if in_speech:
            speech_timestamps.append((speech_start, len(audio) / sr))

        total_speech = sum(end - start for start, end in speech_timestamps)
        duration = len(audio) / sr
        speech_ratio = total_speech / duration if duration > 0 else 0.0

        return VADResult(
            has_speech=len(speech_timestamps) > 0,
            speech_timestamps=speech_timestamps,
            speech_ratio=speech_ratio,
            confidence=1.0 if speech_ratio > 0.1 else 0.0,
            vad_type="energy"
        )

    def detect(self, audio_path: str) -> VADResult:
        audio, sr = sf.read(audio_path)
        return self._detect_array_impl(audio, sr)

    def detect_array(self, audio: np.ndarray, sr: int) -> VADResult:
        """直接从 numpy 数组做能量 VAD。"""
        return self._detect_array_impl(audio, sr)


def create_vad_detector(vad_type: str = "firered", **kwargs) -> object:
    """创建单个 VAD 检测器。

    统一在此做参数名归一：调用方传 canonical 名称（model_dir / onnx_path /
    use_gpu / threshold / strict_vad_models），各 wrapper 的参数名差异
    （speech_threshold、model_path 等）在此映射，避免 kwargs 被 **kwargs
    静默吞掉。strict_vad_models 当前只被 SileroVADWrapper 消费。
    """
    model_dir = kwargs.get("model_dir")
    onnx_path = kwargs.get("onnx_path") or kwargs.get("nemo_onnx_path")
    use_gpu = kwargs.get("use_gpu", False)
    threshold = kwargs.get("threshold", kwargs.get("speech_threshold", 0.5))

    temp_dir = kwargs.get("temp_dir")
    if vad_type == "firered":
        # FireRed 的参数名是 speech_threshold 而非 threshold
        return FireRedVADWrapper(model_dir=model_dir, use_gpu=use_gpu, speech_threshold=threshold, temp_dir=temp_dir)
    elif vad_type == "webrtc":
        # WebRTCVADWrapper 不接受额外 kwargs，必须显式构造
        return WebRTCVADWrapper()
    elif vad_type == "silero":
        return SileroVADWrapper(
            use_gpu=use_gpu,
            threshold=threshold,
            strict_vad_models=bool(kwargs.get("strict_vad_models", False)),
        )
    elif vad_type == "funasr":
        return FunASRVADWrapper(model_dir=model_dir, use_gpu=use_gpu, threshold=threshold, temp_dir=temp_dir)
    elif vad_type == "nemo_onnx":
        return NeMoONNXVADWrapper(onnx_path=onnx_path, threshold=threshold, use_gpu=use_gpu)
    elif vad_type == "nemo":
        # NeMoVADWrapper 的参数名是 model_path 而非 model_dir
        return NeMoVADWrapper(model_path=model_dir, use_gpu=use_gpu, threshold=threshold)
    elif vad_type == "energy":
        # 显式选择能量 VAD（对齐 MultiVAD 的 "energy" 类型）
        return FallbackVAD()
    else:
        # 未知类型不得静默退化为 FallbackVAD（对齐 MultiVADDetector 的
        # ValueError 语义）：配置拼写错误必须尽早暴露，否则 strict 部署
        # 会在模型名写错时全程用能量 VAD 跑完整个 stage
        raise ValueError(f"不支持的 VAD 类型: {vad_type}")
