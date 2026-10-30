"""AECMOS wrapper for AEC evaluation.

Loads ONNX models from ``third_party_scripts/AECMOS/``.
AECMOS evaluates enhanced audio in the context of AEC scenarios
using three inputs: loopback (lpb), microphone (mic), and enhanced (enh).

Outputs:
  - echo_mos: echo cancellation quality (0-5)
  - deg_mos:  other degradation quality (0-5)

Usage:
    from core.metrics.aecmos import load_aecmos, compute_aecmos

    estimator = load_aecmos("Run_1663915512_Stage_0")
    echo_mos, deg_mos = compute_aecmos(
        estimator, lpb_audio, mic_audio, enh_audio, talk_type="dt"
    )

Talk types:
  - "st"  : far-end single talk  (farend-singletalk)
  - "nst" : near-end single talk (nearend-singletalk)
  - "dt"  : double talk          (prob_target_echo / prob_target_noise_echo)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

_log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_AECMOS_DIR = _PROJECT_ROOT / "third_party_scripts" / "AECMOS"

# Default 16kHz model with scenario markers (recommended)
_DEFAULT_MODEL = "Run_1663915512_Stage_0.onnx"

# Model configs
_MODEL_CONFIGS = {
    "Run_1663915512_Stage_0": {
        "sr": 16000,
        "dft_size": 512,
        "need_scenario_marker": True,
        "hidden_size": (4, 1, 64),
    },
    "Run_1663829550_Stage_0": {
        "sr": 16000,
        "dft_size": 512,
        "need_scenario_marker": False,
        "hidden_size": (4, 1, 64),
    },
    "Run_1668423760_Stage_0": {
        "sr": 48000,
        "dft_size": 1536,
        "need_scenario_marker": True,
        "hidden_size": (4, 1, 96),
    },
}


class AECMOSEstimator:
    """Local AECMOS ONNX inference wrapper."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.max_len = 20  # seconds
        self.hop_fraction = 0.5

        model_name = self.model_path.stem
        matched = False
        for key, cfg in _MODEL_CONFIGS.items():
            if key in model_name:
                self.sampling_rate = cfg["sr"]
                self.dft_size = cfg["dft_size"]
                self.need_scenario_marker = cfg["need_scenario_marker"]
                self.hidden_size = cfg["hidden_size"]
                matched = True
                break
        if not matched:
            raise ValueError(f"Not a supported AECMOS model: {model_name}")

        # Lazy-load ONNX session
        self._ort_session = None

    def _get_session(self):
        if self._ort_session is None:
            import onnxruntime as ort
            self._ort_session = ort.InferenceSession(str(self.model_path))
        return self._ort_session

    def _mel_transform(self, sample: np.ndarray, sr: int) -> np.ndarray:
        import librosa
        mel_spec = librosa.feature.melspectrogram(
            y=sample,
            sr=sr,
            n_fft=self.dft_size + 1,
            hop_length=int(self.hop_fraction * self.dft_size),
            n_mels=160,
        )
        mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def _align_audio(
        self,
        lpb_audio: np.ndarray,
        mic_audio: np.ndarray,
        enh_audio: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        min_len = min(len(lpb_audio), len(mic_audio), len(enh_audio))
        if min_len == 0:
            raise ValueError("Empty audio input")
        return lpb_audio[:min_len], mic_audio[:min_len], enh_audio[:min_len]

    def run(
        self,
        talk_type: str,
        lpb_audio: np.ndarray,
        mic_audio: np.ndarray,
        enh_audio: np.ndarray,
    ) -> tuple[float, float]:
        """Run AECMOS inference.

        已知限制（C2）：本方法**不做重采样**，mel 变换直接按模型采样率
        （``self.sampling_rate``）提取特征。输入音频的实际采样率必须与模型
        采样率一致；非匹配输入不会报错，但结果会静默失真（频谱按错误
        采样率解释）。

        Args:
            talk_type: "st" | "nst" | "dt"
            lpb_audio: loopback (far-end) signal, 必须已是模型采样率
            mic_audio: microphone signal, 必须已是模型采样率
            enh_audio: enhanced (model output) signal, 必须已是模型采样率

        Returns:
            (echo_mos, deg_mos)
        """
        # Align lengths
        lpb_sig, mic_sig, enh_sig = self._align_audio(lpb_audio, mic_audio, enh_audio)

        # Truncate to max_len seconds
        seg_nb_samples = self.max_len * self.sampling_rate
        if len(lpb_sig) > seg_nb_samples:
            lpb_sig = lpb_sig[:seg_nb_samples]
            mic_sig = mic_sig[:seg_nb_samples]
            enh_sig = enh_sig[:seg_nb_samples]

        # Mel transform
        lpb_sig = self._mel_transform(lpb_sig, self.sampling_rate)
        mic_sig = self._mel_transform(mic_sig, self.sampling_rate)
        enh_sig = self._mel_transform(enh_sig, self.sampling_rate)

        # Scenario marker
        if self.need_scenario_marker:
            if talk_type == "nst":
                ne_st, fe_st = 1, 0
            elif talk_type == "st":
                ne_st, fe_st = 0, 1
            else:
                ne_st, fe_st = 0, 0

            mic_sig = np.concatenate(
                (mic_sig, np.ones((20, mic_sig.shape[1])) * (1 - fe_st), np.zeros((20, mic_sig.shape[1]))),
                axis=0,
            )
            lpb_sig = np.concatenate(
                (lpb_sig, np.ones((20, lpb_sig.shape[1])) * (1 - ne_st), np.zeros((20, lpb_sig.shape[1]))),
                axis=0,
            )
            enh_sig = np.concatenate(
                (enh_sig, np.ones((20, enh_sig.shape[1])), np.zeros((20, enh_sig.shape[1]))),
                axis=0,
            )

        # Stack and batch
        feats = np.stack((lpb_sig, mic_sig, enh_sig)).astype(np.float32)
        feats = np.expand_dims(feats, axis=0)

        # ONNX inference
        ort_session = self._get_session()
        input_name = ort_session.get_inputs()[0].name

        import torch
        h0 = torch.zeros(self.hidden_size, dtype=torch.float32).detach().numpy()
        result = ort_session.run([], {input_name: feats, "h0": h0})
        result = result[0]

        echo_mos = float(result[0])
        deg_mos = float(result[1])
        return echo_mos, deg_mos


def load_aecmos(model_name: Optional[str] = None) -> AECMOSEstimator:
    """Load AECMOS estimator.

    Args:
        model_name: e.g. "Run_1663915512_Stage_0" (16kHz, with scenario marker).
                    If None, uses the default recommended model.
    """
    if model_name is None:
        model_name = _DEFAULT_MODEL.replace(".onnx", "")
    else:
        # model_name 来自 CLI/评估配置，只接受纯文件名，禁止 "../" 越出 AECMOS 目录。
        model_name = Path(model_name).name
    if model_name in ("", ".", ".."):
        raise ValueError(f"invalid AECMOS model name: {model_name!r}")
    model_path = _AECMOS_DIR / f"{model_name}.onnx"
    if not model_path.exists():
        available = [p.name for p in _AECMOS_DIR.glob("*.onnx")]
        raise FileNotFoundError(
            f"AECMOS model not found: {model_path}. Available: {available}"
        )
    _log.info(f"Loaded AECMOS model: {model_path.name}")
    return AECMOSEstimator(model_path)


def compute_aecmos(
    estimator: AECMOSEstimator,
    lpb_audio: np.ndarray,
    mic_audio: np.ndarray,
    enh_audio: np.ndarray,
    talk_type: str = "dt",
) -> dict[str, float]:
    """Compute AECMOS scores for a single audio triplet.

    Args:
        estimator: Loaded AECMOSEstimator
        lpb_audio: loopback signal (1-D numpy array)
        mic_audio: microphone signal (1-D numpy array)
        enh_audio: enhanced signal (1-D numpy array)
        talk_type: "st" | "nst" | "dt"

    Returns:
        {"AECMOS_ECHO": echo_mos, "AECMOS_DEG": deg_mos}
    """
    echo_mos, deg_mos = estimator.run(talk_type, lpb_audio, mic_audio, enh_audio)
    return {"AECMOS_ECHO": echo_mos, "AECMOS_DEG": deg_mos}
