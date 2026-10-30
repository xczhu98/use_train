"""Local MOS model wrappers: DNSMOS, NISQA, UTMOS, SCOREQ.

All models load from ``third_party_scripts/`` under the project root.
No network download required — all weights are bundled locally.

Usage:
    from core.metrics.mos_models import load_dnsmos, load_nisqa, load_utmos, load_scoreq

    dnsmos  = load_dnsmos()       # callable(audio_1d, sr) -> dict
    nisqa   = load_nisqa()        # callable(audio_1d, sr) -> dict
    utmos   = load_utmos()        # callable(audio_1d, sr) -> dict
    scoreq_ = load_scoreq()       # callable(audio_1d, sr) -> dict

Author: Xincheng Zhu
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from core.utils.safe_load import safe_torch_load

_log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_THIRD_PARTY = _PROJECT_ROOT / "third_party_scripts"


def _resolve_third_party_dir() -> Path:
    """Resolve third_party_scripts directory.

    Priority: env var USE_TRAIN_THIRD_PARTY_DIR > project_root/third_party_scripts
    """
    env = os.environ.get("USE_TRAIN_THIRD_PARTY_DIR")
    if env:
        return Path(env)
    return _THIRD_PARTY


def _import_vendored_package(package_name: str, package_root: Path):
    """按显式路径加载 vendored 顶层包，避免把调用方目录注入 ``sys.path``。

    只让 ``package_name`` 自身可 import，目录内其它模块不再获得可导入性，因此
    环境变量/调用方传入的第三方目录不构成任意模块注入面。vendored NISQA 没有
    ``__init__.py``（PEP 420 命名空间包），此时注册带 ``__path__`` 的命名空间模块。
    """
    import importlib.util
    import sys
    import types

    if package_name in sys.modules:
        return sys.modules[package_name]
    root = Path(package_root)
    if not root.is_dir():
        raise ImportError(f"vendored package directory not found: {root}")
    init_file = root / "__init__.py"
    if init_file.is_file():
        spec = importlib.util.spec_from_file_location(
            package_name,
            init_file,
            submodule_search_locations=[str(root)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load vendored package: {init_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(package_name, None)
            raise
        return module
    module = types.ModuleType(package_name)
    module.__path__ = [str(root)]
    sys.modules[package_name] = module
    return module


from core.metrics.utils import _resample_np


# ────────────────────────────────────────────────────────────────
# DNSMOS (ONNX, 16 kHz, sig/bak/ovr scores)
# ────────────────────────────────────────────────────────────────

_DNSMOS_SR = 16000
_DNSMOS_INPUT_LEN = 9.01


class _DNSMOSScorer:
    """DNSMOS P.835 local scorer (ONNX)."""

    def __init__(self, model_path: str):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self._p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        self._p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        self._p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

    def __call__(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        if np.asarray(audio).size == 0:
            return {"OVRL": float("nan"), "SIG": float("nan"), "BAK": float("nan")}
        if sr != _DNSMOS_SR:
            audio = _resample_np(audio, sr, _DNSMOS_SR)

        len_samples = int(_DNSMOS_INPUT_LEN * _DNSMOS_SR)
        if len(audio) < len_samples:
            repeats = (len_samples // len(audio)) + 1
            audio = np.tile(audio, repeats)[:len_samples]

        num_hops = int(np.floor(len(audio) / _DNSMOS_SR) - _DNSMOS_INPUT_LEN) + 1
        sigs, baks, ovrs = [], [], []

        for idx in range(num_hops):
            seg = audio[idx * _DNSMOS_SR: int((idx + _DNSMOS_INPUT_LEN) * _DNSMOS_SR)]
            if len(seg) < len_samples:
                continue
            feat = seg.astype("float32")[np.newaxis, :]
            raw_sig, raw_bak, raw_ovr = self._sess.run(None, {"input_1": feat})[0][0]
            sigs.append(float(self._p_sig(raw_sig)))
            baks.append(float(self._p_bak(raw_bak)))
            ovrs.append(float(self._p_ovr(raw_ovr)))

        result = {
            "OVRL": float(np.mean(ovrs)) if ovrs else float("nan"),
            "SIG": float(np.mean(sigs)) if sigs else float("nan"),
            "BAK": float(np.mean(baks)) if baks else float("nan"),
        }
        # 非有限值是模型/推理失败信号；由 batch 聚合器计 failure 并跳过。
        return {k: (v if np.isfinite(v) else float("nan")) for k, v in result.items()}


def load_dnsmos(model_dir: Optional[str] = None) -> _DNSMOSScorer:
    """Load DNSMOS model from local ONNX file.

    Args:
        model_dir: path to DNSMOS directory containing ``DNSMOS/sig_bak_ovr.onnx``.
                   Default: ``third_party_scripts/DNSMOS``.
    """
    base = Path(model_dir) if model_dir else _resolve_third_party_dir() / "DNSMOS"
    onnx_path = base / "DNSMOS" / "sig_bak_ovr.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"DNSMOS model not found: {onnx_path}\n"
            "Expected: third_party_scripts/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
        )
    _log.info(f"Loading DNSMOS from {onnx_path}")
    return _DNSMOSScorer(str(onnx_path))


# ────────────────────────────────────────────────────────────────
# UTMOS (local weights + SpeechMOS source, 16 kHz)
# ────────────────────────────────────────────────────────────────

class _UTMOSScorer:
    """UTMOS22 strong learner MOS estimator.

    Loads from ``third_party_scripts/UTMOS/`` (local, no network).
    Reference: https://github.com/tarepan/SpeechMOS
    """

    _MAX_PADDED_SAMPLES_PER_RUN = 4_800_000

    def __init__(self, model_dir: str, device: str = "cpu"):
        _import_vendored_package(
            "speechmos", Path(model_dir) / "SpeechMOS" / "speechmos"
        )

        from speechmos.utmos22.strong.model import UTMOS22Strong

        self._model = UTMOS22Strong()
        weights_path = os.path.join(model_dir, "utmos22_strong_step7459_v1.pt")
        state_dict = safe_torch_load(
            weights_path,
            map_location="cpu",
            purpose="load bundled UTMOS weights",
        )
        self._model.load_state_dict(state_dict)
        self._device = device
        self._model = self._model.to(device=device)
        self._model.eval()

    def __call__(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        import torch
        if sr != 16000:
            audio = _resample_np(audio, sr, 16000)
            sr = 16000
        wave = torch.from_numpy(audio).float().unsqueeze(0).to(device=self._device)
        with torch.inference_mode():
            score = self._model(wave, sr)
        val = float(score.cpu().item())
        if not np.isfinite(val):
            _log.warning("UTMOS returned non-finite score")
            val = float("nan")
        return {"MOS": val}

    def batch(
        self,
        items: list[tuple[np.ndarray, int]],
        batch_size: int = 16,
    ) -> list[dict[str, float]]:
        """按 16 kHz 预处理后的确切长度分桶，避免无 mask 的 BLSTM 读取 padding。"""
        import torch

        if not items:
            return []
        buckets = {}
        for index, (audio, sr) in enumerate(items):
            if sr != 16000:
                audio = _resample_np(audio, sr, 16000)
            waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32))
            buckets.setdefault(int(waveform.size(0)), []).append((index, waveform))

        results: list[dict[str, float]] = [None] * len(items)  # type: ignore[list-item]
        for exact_length, bucket in buckets.items():
            safe_batch_size = max(
                1, self._MAX_PADDED_SAMPLES_PER_RUN // max(exact_length, 1),
            )
            max_items = min(max(1, int(batch_size)), safe_batch_size)
            for start in range(0, len(bucket), max_items):
                chunk = bucket[start:start + max_items]
                waveforms = torch.stack([waveform for _, waveform in chunk]).to(self._device)
                with torch.inference_mode():
                    scores = self._model(waveforms, 16000)
                for batch_index, (original_index, _) in enumerate(chunk):
                    value = float(scores[batch_index].cpu().item())
                    results[original_index] = {
                        "MOS": value if np.isfinite(value) else float("nan"),
                    }
        return results


def load_utmos(model_dir: Optional[str] = None, device: str = "cpu") -> _UTMOSScorer:
    """Load UTMOS model from local weights.

    Expected files in ``third_party_scripts/UTMOS/``:
      - ``utmos22_strong_step7459_v1.pt`` (~392 MB)
      - ``SpeechMOS/`` (source code directory)

    Args:
        model_dir: path to UTMOS directory. Default: ``third_party_scripts/UTMOS``.
        device: "cpu" or "cuda".
    """
    base = Path(model_dir) if model_dir else _resolve_third_party_dir() / "UTMOS"
    weights = base / "utmos22_strong_step7459_v1.pt"
    src_dir = base / "SpeechMOS" / "speechmos"
    if not weights.exists():
        raise FileNotFoundError(
            f"UTMOS weights not found: {weights}\n"
            "Expected: third_party_scripts/UTMOS/utmos22_strong_step7459_v1.pt"
        )
    if not src_dir.exists():
        raise FileNotFoundError(
            f"UTMOS source not found: {src_dir}\n"
            "Expected: third_party_scripts/UTMOS/SpeechMOS/speechmos/"
        )
    _log.info(f"Loading UTMOS from {base}")
    return _UTMOSScorer(str(base), device=device)


# ────────────────────────────────────────────────────────────────
# NISQA (PyTorch, any SR, MOS + sub-dimensions)
# ────────────────────────────────────────────────────────────────


def _load_nisqa_checkpoint(weights_path: str) -> dict:
    """安全加载 NISQA 专有 checkpoint，并校验其模型字段。"""
    checkpoint = safe_torch_load(
        weights_path,
        map_location="cpu",
        purpose="load bundled NISQA weights",
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"NISQA checkpoint {weights_path} 不是 dict 格式")
    if not isinstance(checkpoint.get("args"), dict):
        raise ValueError(f"NISQA checkpoint {weights_path} 缺少 dict 类型的 args")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(
            f"NISQA checkpoint {weights_path} 缺少 dict 类型的 model_state_dict"
        )
    return checkpoint


class _NISQAScorer:
    """NISQA local scorer (PyTorch, batch-of-one via temp WAV).

    The PyTorch model is loaded once in __init__ and reused for all
    subsequent calls.  Only the lightweight dataset (mel-spectrogram
    extraction from a single WAV) is recreated per call.

    Sample-rate handling:
      Input audio at any SR is resampled to the model's expected ``ms_sr``
      (from checkpoint, typically 48 kHz) before mel-spectrogram extraction,
      ensuring mel filter parameters match correctly and avoiding empty
      mel-band warnings from librosa.

    All NISQA verbose stdout output and third-party warnings are suppressed.
    The bundled checkpoint is loaded safely before the vendored constructor is
    invoked; the constructor only builds the architecture with pretrained
    loading disabled.
    """

    def __init__(self, nisqa_root: str, weights_path: str):
        import io
        import sys
        import tempfile
        import warnings

        import soundfile as sf

        _import_vendored_package("nisqa", Path(nisqa_root) / "nisqa")
        from nisqa.NISQA_model import nisqaModel  # noqa: N813

        checkpoint = _load_nisqa_checkpoint(weights_path)
        with tempfile.TemporaryDirectory(prefix="nisqa_init_") as tmpdir:
            dummy_path = os.path.join(tmpdir, "dummy.wav")
            sf.write(dummy_path, np.zeros(16000, dtype="float32"), 16000)

            # 复用 checkpoint 的完整模型参数，但禁止第三方构造器再次自行 torch.load。
            nisqa_args = dict(checkpoint["args"])
            nisqa_args.update({
                "mode": "predict_file",
                "pretrained_model": False,
                "ms_channel": 1,
                "deg": dummy_path,
                "output_dir": tmpdir,
                "ms_max_segments": 2500,  # 默认 1300 对长样本会失败；提升至 2500 以覆盖当前测试集
            })

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _orig = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    self._model = nisqaModel(nisqa_args)
                finally:
                    sys.stdout = _orig

        missing_keys, unexpected_keys = self._model.model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"NISQA checkpoint state_dict 不匹配 ({weights_path}): "
                f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
            )

        self._model.args["tr_parallel"] = False
        # Get target sample rate (NISQA internally uses 48 kHz)
        # Handle both dict and argparse.Namespace
        try:
            if hasattr(self._model.args, 'get'):
                # It's a dict
                ms_sr = self._model.args.get("ms_sr", 48000)
            else:
                # It's probably argparse.Namespace
                ms_sr = getattr(self._model.args, 'ms_sr', 48000)
            self._target_sr = int(ms_sr) if ms_sr is not None else 48000
        except Exception:
            # Fallback to 48000
            self._target_sr = 48000

        # nisqaModel is not thread-safe: shared state (args, ds_val) must not be touched concurrently.
        self._infer_lock = threading.Lock()

    def __call__(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        import io
        import sys
        import tempfile
        import warnings

        import soundfile as sf

        if sr != self._target_sr:
            audio = _resample_np(audio, sr, self._target_sr)
            sr = self._target_sr

        with self._infer_lock:
            with tempfile.TemporaryDirectory(prefix="nisqa_") as tmpdir:
                os.makedirs(tmpdir, exist_ok=True)
                wav_path = os.path.join(tmpdir, "input.wav")
                sf.write(wav_path, audio, sr)

                self._model.args["deg"] = wav_path
                self._model.args["output_dir"] = tmpdir

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _orig = sys.stdout
                    sys.stdout = io.StringIO()
                    try:
                        self._model._loadDatasetsFile()
                        df = self._model.predict()
                    finally:
                        sys.stdout = _orig

            if df is None or len(df) == 0:
                raise RuntimeError("NISQA predict() returned empty dataframe")

            row = df.iloc[0]
            cols = set(df.columns.astype(str))

            def _pick(*names: str, default: float = float("nan")) -> float:
                for n in names:
                    if n in cols:
                        v = float(row[n])
                        return v if np.isfinite(v) else default
                return default

            # NISQA (MOS-only) adds mos_pred only; NISQA_DIM adds noi_pred, col_pred, ...
            raw = {
                "MOS": _pick("mos_pred", "mos"),
                "NOI": _pick("noi_pred", "NOI"),
                "COL": _pick("col_pred", "COL"),
                "DIS": _pick("dis_pred", "DIS"),
                "LOUD": _pick("loud_pred", "LOUD"),
            }
            if not np.isfinite(raw["MOS"]):
                raise RuntimeError(
                    "NISQA output missing mos_pred; "
                    f"columns={sorted(cols)}"
                )
            # Missing sub-dimensions (MOS-only checkpoint): keep NaN for CSV/summary, not 0
            return {
                k: (v if np.isfinite(v) else float("nan"))
                for k, v in raw.items()
            }


def load_nisqa(model_dir: Optional[str] = None) -> _NISQAScorer:
    """Load NISQA model from local PyTorch weights.

    Args:
        model_dir: path to NISQA directory containing ``weights/nisqa.tar``.
                   Default: ``third_party_scripts/NISQA``.
    """
    base = Path(model_dir) if model_dir else _resolve_third_party_dir() / "NISQA"
    weights = base / "weights" / "nisqa.tar"
    if not weights.exists():
        raise FileNotFoundError(
            f"NISQA weights not found: {weights}\n"
            "Expected: third_party_scripts/NISQA/weights/nisqa.tar"
        )
    _log.info(f"Loading NISQA from {weights}")
    return _NISQAScorer(str(base), str(weights))


# ────────────────────────────────────────────────────────────────
# SCOREQ (pip package, ONNX, 16 kHz, NeurIPS 2024)
# ────────────────────────────────────────────────────────────────

_SCOREQ_ONNX_FILES = {
    ("natural", "nr"): "adapt_nr_telephone.onnx",
    ("natural", "ref"): "fixed_nmr_telephone.onnx",
    ("synthetic", "nr"): "adapt_nr_synthetic.onnx",
    ("synthetic", "ref"): "fixed_nmr_synthetic.onnx",
}


def _scoreq_mono_audio(audio: np.ndarray) -> np.ndarray:
    """将单条 time-first ``(T, C)`` 波形降混为一维。"""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=1, dtype=np.float32)
    raise ValueError(f"SCOREQ expects 1D or time-first (T, C) audio, got {audio.shape}")


class _SCOREQScorer:
    """SCOREQ NR-MOS scorer (local ONNX, no network)."""

    _PADDING_MULTIPLE = 320
    # SSL conv feature extractor 要求 padded 长度 >= 640（40 ms @16k）；
    # 更短的输入会在 conv_layers.6 退化成非法 shape 触发 ONNX INVALID_ARGUMENT。
    _MIN_PADDED_SAMPLES = 640
    _MAX_SAMPLES_PER_RUN = 1_000_000
    _MAX_SAMPLES_PER_ITEM = 160000
    _GPU_SESSION_MAX_RUNS = 10
    _GPU_MEM_LIMIT_BYTES = 16 * 1024 * 1024 * 1024

    def __init__(self, onnx_path: str, use_gpu: bool = False, session_path_lease=None):
        self._onnx_path = onnx_path
        self._onnx_filename = os.path.basename(onnx_path)
        self._use_gpu = bool(use_gpu)
        self._session_path_lease = session_path_lease
        self._runs_since_reset = 0
        self._broken_error = None
        self._resamplers = {}
        self._create_session()

    def bind_session_path_lease(self, lease) -> None:
        """为后续 GPU arena reset 绑定 owned snapshot 的 loader lease。"""
        self._session_path_lease = lease

    def _build_session(self, onnx_path: str):
        import onnxruntime as ort

        if not self._use_gpu:
            self._device_str = "CPU"
            return ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"],
            )

        options = ort.SessionOptions()
        options.add_session_config_entry(
            "gpu.mem_limit", str(self._GPU_MEM_LIMIT_BYTES),
        )
        options.add_session_config_entry(
            "gpu.arena_extend_strategy", "kSameAsRequested",
        )
        session = ort.InferenceSession(
            onnx_path,
            options,
            providers=[
                ("CUDAExecutionProvider", {
                    "arena_extend_strategy": "kSameAsRequested",
                    "gpu_mem_limit": self._GPU_MEM_LIMIT_BYTES,
                }),
                "CPUExecutionProvider",
            ],
        )
        providers = tuple(session.get_providers())
        if not providers or providers[0] != "CUDAExecutionProvider":
            raise RuntimeError("SCOREQ CUDA-primary provider registration failed")
        self._device_str = "GPU"
        return session

    def _create_session(self) -> None:
        lease = self._session_path_lease
        if lease is None:
            self._sess = self._build_session(self._onnx_path)
            return
        with lease() as root_path:
            self._sess = self._build_session(
                os.path.join(root_path, self._onnx_filename),
            )

    def _reset_gpu_session(self) -> None:
        try:
            self._create_session()
        except Exception as exc:
            self._broken_error = exc
            self._sess = None
            raise RuntimeError("SCOREQ GPU session reset failed; scorer is broken") from exc
        self._runs_since_reset = 0

    def _run(self, batch: np.ndarray):
        broken_error = getattr(self, "_broken_error", None)
        if broken_error is not None:
            raise RuntimeError("SCOREQ scorer is broken after session reset failure") from broken_error
        use_gpu = bool(getattr(self, "_use_gpu", False))
        if use_gpu and getattr(self, "_runs_since_reset", 0) >= self._GPU_SESSION_MAX_RUNS:
            self._reset_gpu_session()
        input_name = self._sess.get_inputs()[0].name
        if use_gpu:
            self._runs_since_reset = getattr(self, "_runs_since_reset", 0) + 1
        return self._sess.run(None, {input_name: batch})

    def __call__(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        import math
        import torch
        import torch.nn.functional as F

        mono = _scoreq_mono_audio(audio)
        wave = torch.from_numpy(mono).float().unsqueeze(0)
        if sr != 16000:
            import torchaudio
            if sr not in self._resamplers:
                self._resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
            wave = self._resamplers[sr](wave)
        if wave.size(-1) > self._MAX_SAMPLES_PER_ITEM:
            _log.debug("SCOREQ __call__: input exceeds 10s cap and is truncated")
            wave = wave[..., :self._MAX_SAMPLES_PER_ITEM]

        size = wave.size(-1)
        padded_size = max(
            math.ceil(size / self._PADDING_MULTIPLE) * self._PADDING_MULTIPLE,
            self._MIN_PADDED_SAMPLES,
        )
        if padded_size > size:
            wave = F.pad(wave, (0, padded_size - size), value=0)
        score = self._run(wave.numpy())[0].item()
        value = float(score)
        if not np.isfinite(value):
            _log.warning("SCOREQ returned non-finite score")
            value = float("nan")
        return {"MOS": value}

    def run_batch(self, list_of_audio_sr) -> list:
        """按截断/最小补零（``_MIN_PADDED_SAMPLES``）对齐后的确切长度分桶，避免跨样本 padding 改变模型语义。"""
        import math

        prepared = []
        for index, (audio, sr) in enumerate(list_of_audio_sr):
            audio = _scoreq_mono_audio(audio)
            if sr != 16000:
                if sr not in self._resamplers:
                    import torchaudio
                    self._resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
                import torch
                tensor = torch.from_numpy(audio).float().unsqueeze(0)
                audio = self._resamplers[sr](tensor).numpy().squeeze(0)
            if int(audio.size) > self._MAX_SAMPLES_PER_ITEM:
                _log.debug("SCOREQ run_batch: input exceeds 10s cap and is truncated")
                audio = audio[:self._MAX_SAMPLES_PER_ITEM]
            padded_size = max(
                math.ceil(int(audio.size) / self._PADDING_MULTIPLE) * self._PADDING_MULTIPLE,
                self._MIN_PADDED_SAMPLES,
            )
            if padded_size > audio.size:
                audio = np.pad(audio, (0, padded_size - audio.size), mode="constant")
            prepared.append((index, np.asarray(audio, dtype=np.float32)))
        if not prepared:
            return []

        buckets = {}
        for index, audio in prepared:
            buckets.setdefault(int(audio.size), []).append((index, audio))

        values = [float("nan")] * len(prepared)
        batch_failed = False
        for padded_size, bucket in buckets.items():
            max_items = max(1, self._MAX_SAMPLES_PER_RUN // max(padded_size, 1))
            for start in range(0, len(bucket), max_items):
                chunk = bucket[start:start + max_items]
                batch = np.stack([audio for _, audio in chunk]).astype(np.float32, copy=False)
                out = np.asarray(self._run(batch)[0])
                if out.ndim >= 2 and out.shape[1] > 0:
                    scores = out[:, 0]
                else:
                    scores = out.reshape(-1)
                if scores.size != len(chunk):
                    _log.warning(
                        "SCOREQ batch output size mismatch: expected %d, got %d; marking this batch failed",
                        len(chunk), scores.size,
                    )
                    batch_failed = True
                    continue
                for (index, _), value in zip(chunk, scores):
                    values[index] = float(value)

        if batch_failed:
            return [{"MOS": float("nan")} for _ in list_of_audio_sr]
        return [
            {"MOS": value if np.isfinite(value) else float("nan")}
            for value in values
        ]


def load_scoreq(
    model_dir: Optional[str] = None,
    data_domain: str = "natural",
    mode: str = "nr",
    use_gpu: bool = False,
) -> _SCOREQScorer:
    """Load SCOREQ model from local ONNX file.

    Expected file in ``third_party_scripts/SCOREQ/``:
      - ``adapt_nr_telephone.onnx`` (~378 MB, for natural NR mode)

    Args:
        model_dir: path to SCOREQ directory. Default: ``third_party_scripts/SCOREQ``.
        data_domain: "natural" or "synthetic".
        mode: "nr" (no-reference) or "ref".
        use_gpu: if True, use CUDAExecutionProvider when available (requires onnxruntime-gpu).
    """
    base = Path(model_dir) if model_dir else _resolve_third_party_dir() / "SCOREQ"
    fname = _SCOREQ_ONNX_FILES.get((data_domain, mode))
    if fname is None:
        raise ValueError(
            f"Invalid SCOREQ config: data_domain={data_domain}, mode={mode}. "
            f"Valid: {list(_SCOREQ_ONNX_FILES.keys())}"
        )
    onnx_path = base / fname
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"SCOREQ model not found: {onnx_path}\n"
            f"Expected: third_party_scripts/SCOREQ/{fname}\n"
            "Download from: https://zenodo.org/records/15739280"
        )
    _log.info(f"Loading SCOREQ from {onnx_path}")
    return _SCOREQScorer(str(onnx_path), use_gpu=use_gpu)
