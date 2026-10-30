"""Top-level StuPASE inference model: DeWavLM-R -> CFM (flow matching) -> VocosVocoder.

Ported from ``stupase/models/stupase.py``. This composition is inference-only
(the CFM's ODE sampling is not differentiable) -- it is registered under
MODEL_REGISTRY purely for eval/inference pipelines, never used as a
"generator" in any training trainer (see flow_train_wrapper.py /
mel_gan_wrapper.py for the training-time model wrappers).
"""

import torch

from core.utils.registry import MODEL_REGISTRY
from tasks.se.models.base_model import BaseSEModel
from tasks.se.models.pase_family.common.utils import (
    align_waveform_length,
    scale_waveform_to_reference_peak,
)
from tasks.se.models.pase_family.common.wavlm_frontend import WavLM_feat

from .flow.cfm import CFM
from .vocos_vocoder import VocosVocoder


@MODEL_REGISTRY.register("StuPASE")
class StuPASE(BaseSEModel):
    """StuPASE generative speech enhancement model (inference-only).

    Args:
        dewavlm_ckpt_path: Path to DeWavLM-R.pt (dereverberated WavLM encoder).
        cfm_ckpt_path: Path to CFM.pt (flow-matching acoustic module).
        vocoder_ckpt_path: Path to Vocoder_Mel-16k.pt (Mel vocoder).
        output_layer: WavLM layer to use as the phonetic condition.
    """

    def __init__(
        self,
        dewavlm_ckpt_path: str = "",
        cfm_ckpt_path: str = "",
        vocoder_ckpt_path: str = "",
        output_layer: int = 24,
        **kwargs,
    ):
        super().__init__()
        for key in ("n_fft", "hop_length", "win_length", "window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"StuPASE got unexpected keyword arguments: {sorted(kwargs)}")
        if not dewavlm_ckpt_path or not cfm_ckpt_path or not vocoder_ckpt_path:
            raise ValueError("StuPASE inference requires dewavlm_ckpt_path, cfm_ckpt_path, and vocoder_ckpt_path")
        self.dewavlm = WavLM_feat(
            wavlm_ckpt_path=dewavlm_ckpt_path,
            output_layer=output_layer,
            load_pretrained=bool(dewavlm_ckpt_path),
            frozen=True,
        )
        # 三个 ckpt 路径已在上面 fail-closed；不保留「缺 ckpt 时不构造」的不可达回退
        self.cfm = CFM.from_pretrained(cfm_ckpt_path)
        self.vocoder = VocosVocoder.from_pretrained(vocoder_ckpt_path)

    @torch.no_grad()
    def forward(
        self,
        noisy_wav: torch.Tensor,
        lengths: torch.Tensor = None,
        *,
        fs=None,
        steps: int = 8,
        cfg_strength: float = 0.5,
        sway_sampling_coef: float = -1.0,
        normalize_output: bool = True,
    ) -> torch.Tensor:
        """Args: noisy_wav (B, L) or (B, 1, L). Returns: enhanced_wav (B, L).

        normalize_output: 是否把输出峰值按样本归一化到输入峰值，对齐官方
            ``stupase/inference/inference.py:84-85``；长音频分块推理时由外层
            传 False，在拼接后统一归一化。
        """
        x = noisy_wav
        if x.ndim == 3:
            x = x.squeeze(1)

        n_samples = x.shape[-1]
        sr = int(fs) if fs is not None else 16000

        rep = self.dewavlm(x, sr=sr, lengths=lengths)  # enhanced phonetic representations
        y, _ = self.cfm.sample(
            cond=x, cond_noisy=x, text=rep, no_ref_audio=True, vocoder=self.vocoder,
            steps=steps, cfg_strength=cfg_strength, sway_sampling_coef=sway_sampling_coef,
        )
        y = y.squeeze(1)
        y = align_waveform_length(y, n_samples)

        if normalize_output:
            # 官方 inference/inference.py:84-85：输出峰值归一化到输入峰值
            y = scale_waveform_to_reference_peak(y, x)

        return y
