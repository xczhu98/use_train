"""DeWavLM distillation model for PASE Phase-2 / StuPASE Phase-0 training.

Two independent ``WavLM_feat`` instances: a frozen teacher (stays at its
initial pretrained weights for the whole training run) and a trainable
student. The student is fed noisy audio, the teacher is fed clean audio, and
the training objective is MSE between their same-layer features — this
matches source ``pase/train/train_dewavlm.py`` (and StuPASE's analogous
``train_dewavlm.py`` for "DeWavLM-R").

Used with ``tasks.se.trainer.dual_input_loss_trainer.DualInputLossTrainer``, whose
``_forward_and_compute_loss`` calls ``model(noisy, clean, lengths)`` directly.

``use_plc_frontend=True``（UniPASE DeWavLM-Omni 阶段）把学生换成
``unipase.wavlm_frontend_plc.WavLM_feat_PLC``（enable_plc=True），前向时学生
以 ``mask=True``（丢包区域用 mask_emb 替换 CNN 特征后做掩码重建）编码带噪
输入，对齐官方 ``unipase/train/train_dewavlm.py:215-218``；teacher 侧官方
同样用 PLC 版类但 ``mask=False``，与普通 ``WavLM_feat`` 行为一致，故保持
不变。
"""

import logging
from collections.abc import Mapping

import torch
import torch.nn.functional as F
import torchaudio

from core.utils.registry import MODEL_REGISTRY
from tasks.se.models.base_model import BaseSEModel

from .wavlm_frontend import WavLM_feat

logger = logging.getLogger(__name__)


@MODEL_REGISTRY.register("DeWavLMDistillModel")
class DeWavLMDistillModel(BaseSEModel):
    """DeWavLM representation-distillation model (frozen teacher / trainable student).

    Args:
        teacher_ckpt_path: Path to the WavLM checkpoint used to initialize the
            frozen teacher (typically ``WavLM-Large.pt``).
        student_ckpt_path: Path to the WavLM checkpoint used to initialize the
            trainable student (typically also ``WavLM-Large.pt``, or a
            previous-phase checkpoint for warm-starting).
        output_layer: Which transformer layer to compare (int; source uses a
            single layer, 24, for this stage — not a multi-layer list).
        use_plc_frontend: True 时学生使用 ``WavLM_feat_PLC``（enable_plc=True，
            UniPASE DeWavLM-Omni），forward 对学生开启 ``mask=True``；False 时
            保持 PASE/StuPASE 原行为（普通 WavLM_feat，无掩码）。
    """

    input_domain = "waveform"
    output_domain = "loss"
    forward_kind = "loss_returning"
    frontend_owner = "model"

    # validation 验证音频：官方 train_dewavlm.py:260-290（StuPASE Phase 0 /
    # UniPASE Stage 1）把 student 特征经冻结 vocoder 还原为 noisy/clean 两路波形。
    val_audio_stage = "distill"
    val_audio_vocoder_roles = ("vocoder",)
    val_audio_sample_rate = 16000

    def __init__(
        self,
        teacher_ckpt_path: str = "",
        student_ckpt_path: str = "",
        output_layer: int = 24,
        use_plc_frontend: bool = False,
        feature_norm: str = "per_sample",
        **kwargs,
    ):
        super().__init__()
        for key in ("n_fft", "hop_length", "win_length", "window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"DeWavLMDistillModel got unexpected keyword arguments: {sorted(kwargs)}")
        self.teacher = WavLM_feat(
            wavlm_ckpt_path=teacher_ckpt_path,
            output_layer=output_layer,
            load_pretrained=bool(teacher_ckpt_path),
            frozen=True,
            feature_norm=feature_norm,
        )
        if use_plc_frontend:
            # 延迟导入以避免循环依赖。
            from ..unipase.wavlm_frontend_plc import WavLM_feat_PLC

            student_cls = lambda **kw: WavLM_feat_PLC(enable_plc=True, **kw)  # noqa: E731
        else:
            student_cls = WavLM_feat
        self.student = student_cls(
            wavlm_ckpt_path=student_ckpt_path,
            output_layer=output_layer,
            load_pretrained=bool(student_ckpt_path),
            frozen=False,
            feature_norm=feature_norm,
        )
        self.use_plc_frontend = use_plc_frontend

    def forward(
        self,
        noisy_wav: torch.Tensor,
        clean_wav: torch.Tensor,
        lengths: torch.Tensor = None,
        *,
        fs=None,
    ) -> torch.Tensor:
        # 官方 train_dewavlm.py:215-218：学生 mask=True（PLC 掩码），teacher mask=False
        student_feat = self.student(
            noisy_wav, sr=fs, mask=self.use_plc_frontend, lengths=lengths
        )
        with torch.no_grad():
            teacher_feat = self.teacher(clean_wav, sr=fs, lengths=lengths)
        return F.mse_loss(student_feat, teacher_feat)

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
        """官方 ``train_dewavlm.py:260-290`` 的验证音频中间量。

        ``noisy_feat = student(noisy)``（UniPASE Stage 1 的 PLC 学生为 mask=True，
        StuPASE Phase 0 无 PLC），``clean_feat = student(clean, mask=False)``，
        两路特征各经冻结 vocoder 还原；返回文件名模板 → 波形。vocoder 输入布局
        为 ``(B, D, T)``，与官方 ``vocoder(feat.transpose(1, 2))`` 一致。
        """
        if not vocoders or "vocoder" not in vocoders:
            raise ValueError(
                "DeWavLMDistillModel val audio requires vocoders['vocoder']"
            )
        vocoder = vocoders["vocoder"]
        noisy_feat = self.student(
            noisy_wav, sr=fs, mask=self.use_plc_frontend, lengths=lengths
        )
        with torch.no_grad():
            clean_feat = self.student(clean_wav, sr=fs, mask=False, lengths=lengths)
        noisy_recon = vocoder(noisy_feat.transpose(1, 2))
        clean_recon = vocoder(clean_feat.transpose(1, 2))
        sr = int(fs) if fs is not None else self.val_audio_sample_rate
        noisy_out = noisy_wav
        clean_out = clean_wav
        if sr != self.val_audio_sample_rate:
            noisy_out = torchaudio.functional.resample(
                noisy_wav, orig_freq=sr, new_freq=self.val_audio_sample_rate
            )
            clean_out = torchaudio.functional.resample(
                clean_wav, orig_freq=sr, new_freq=self.val_audio_sample_rate
            )
        return {
            "noisy": noisy_out,
            "clean": clean_out,
            "enh_epoch{epoch:03d}": noisy_recon,
            "ref_epoch{epoch:03d}": clean_recon,
        }
