"""StuPASE Phase-2 flow-matching training model.

Wraps a frozen ``WavLM_feat`` (DeWavLM-R, providing the phonetic condition
from the *noisy* audio) and a trainable ``CFM`` (flow-matching acoustic
module). Mirrors source ``stupase/train/train_flow.py``'s per-step logic
exactly: ``feat = encoder(noisy, sr=...); loss, cond, pred = cfm(clean, noisy, feat)``.

Used with ``tasks.se.trainer.dual_input_loss_trainer.DualInputLossTrainer``.
"""

import logging
from collections.abc import Mapping

import torch
import torchaudio

from core.utils.registry import MODEL_REGISTRY
from tasks.se.models.base_model import BaseSEModel
from tasks.se.models.pase_family.common.utils import peak_normalize_to
from tasks.se.models.pase_family.common.wavlm_frontend import WavLM_feat

from .flow.cfm import CFM

logger = logging.getLogger(__name__)


@MODEL_REGISTRY.register("StuPASEFlowTrainModel")
class StuPASEFlowTrainModel(BaseSEModel):
    """DeWavLM-R (frozen) + CFM (trainable) flow-matching training model.

    Args:
        encoder_ckpt_path: Path to the DeWavLM-R checkpoint (frozen phonetic encoder).
        encoder_output_layer: WavLM layer used as the phonetic condition.
        cfm_kwargs: Keyword args forwarded to ``CFM.__init__`` (dit_kwargs,
            mel_spec_kwargs, odeint_kwargs, sigma, audio_drop_prob, etc.).
    """

    input_domain = "waveform"
    output_domain = "loss"
    forward_kind = "loss_returning"
    frontend_owner = "model"

    # validation 验证音频：官方 train_flow.py:245-300（StuPASE Phase 2）。
    val_audio_stage = "flow"
    val_audio_vocoder_roles = ("vocoder",)
    val_audio_sample_rate = 16000

    def __init__(
        self,
        encoder_ckpt_path: str = "",
        encoder_output_layer: int = 24,
        cfm_kwargs: dict = None,
        feature_norm: str = "batch",
        **kwargs,
    ):
        super().__init__()
        for key in ("n_fft", "hop_length", "win_length", "window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"StuPASEFlowTrainModel got unexpected keyword arguments: {sorted(kwargs)}")
        self.encoder = WavLM_feat(
            wavlm_ckpt_path=encoder_ckpt_path,
            output_layer=encoder_output_layer,
            load_pretrained=bool(encoder_ckpt_path),
            frozen=True,
            feature_norm=feature_norm,
        )
        self.cfm = CFM(**(cfm_kwargs or {}))
        self._sample_rate = 16000

    def forward(
        self,
        noisy_wav: torch.Tensor,
        clean_wav: torch.Tensor,
        lengths: torch.Tensor = None,
        *,
        fs=None,
    ) -> torch.Tensor:
        sr = int(fs) if fs is not None else self._sample_rate
        target_sr = self.cfm.mel_spec.target_sample_rate
        if sr != target_sr:
            noisy_wav = torchaudio.functional.resample(noisy_wav, sr, target_sr)
            clean_wav = torchaudio.functional.resample(clean_wav, sr, target_sr)
            if lengths is not None:
                # int32 lengths × 16k 在 48k/4s 输入下会溢出，先升 int64。
                lengths = torch.div(
                    torch.as_tensor(
                        lengths, device=noisy_wav.device, dtype=torch.long
                    )
                    * target_sr
                    + sr
                    - 1,
                    sr,
                    rounding_mode="floor",
                )
        with torch.no_grad():
            feat = self.encoder(noisy_wav, sr=target_sr, lengths=lengths)  # (B, Tp, D)
        mel_lengths = None
        if lengths is not None:
            mel_lengths = torch.div(
                lengths, self.cfm.mel_spec.hop_length, rounding_mode="floor"
            ) + 1
            mel_lengths = mel_lengths.clamp_min(1)
        loss, _cond, _pred = self.cfm(clean_wav, noisy_wav, feat, lens=mel_lengths)
        return loss

    def val_audio_outputs(
        self,
        noisy_wav: torch.Tensor,
        clean_wav: torch.Tensor,
        lengths: torch.Tensor = None,
        *,
        fs=None,
        vocoders: Mapping[str, torch.nn.Module] = None,
        sample_settings=None,
        forward_dict=None,
        raw_enhanced: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """官方 ``train_flow.py:245-300`` 的验证音频中间量。

        ``recon = vocoder(mel_spec(clean))``；``enhanced = cfm.sample(cond=noisy,
        cond_noisy=noisy, text=encoder(noisy), no_ref_audio=True, vocoder=vocoder,
        **sample_settings)``。``recon``/``enhanced`` 各自峰值归一到 0.9。
        ``sample_settings`` 必须显式给出（官方 cfg_train_flow.yaml 的
        steps/cfg_strength/sway_sampling_coef）。
        """
        if not vocoders or "vocoder" not in vocoders:
            raise ValueError(
                "StuPASEFlowTrainModel val audio requires vocoders['vocoder']"
            )
        if sample_settings is None:
            raise ValueError(
                "StuPASEFlowTrainModel val audio requires "
                "trainer.val_audio.sample_settings"
            )
        vocoder = vocoders["vocoder"]
        sr = int(fs) if fs is not None else self._sample_rate
        target_sr = self.cfm.mel_spec.target_sample_rate
        noisy = noisy_wav
        clean = clean_wav
        if sr != target_sr:
            noisy = torchaudio.functional.resample(noisy_wav, sr, target_sr)
            clean = torchaudio.functional.resample(clean_wav, sr, target_sr)
            if lengths is not None:
                lengths = torch.div(
                    torch.as_tensor(
                        lengths, device=noisy.device, dtype=torch.long
                    )
                    * target_sr
                    + sr
                    - 1,
                    sr,
                    rounding_mode="floor",
                )
        feat = self.encoder(noisy, sr=target_sr, lengths=lengths)
        clean_mel = self.cfm.mel_spec(clean)
        recon = vocoder(clean_mel)
        enhanced, _ = self.cfm.sample(
            cond=noisy,
            cond_noisy=noisy,
            text=feat,
            no_ref_audio=True,
            vocoder=vocoder,
            **dict(sample_settings),
        )
        return {
            "noisy": noisy,
            "clean": clean,
            "recon": peak_normalize_to(recon),
            "enh_epoch{epoch:03d}": peak_normalize_to(enhanced),
        }
