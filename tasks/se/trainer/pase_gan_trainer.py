"""PASEGANTrainer: waveform-domain MPD+MBD GAN trainer for PASE / StuPASE.

``GANTrainer`` (the base class) assumes a MetricGAN-style discriminator that
scores ``(clean_mag, clean_pha, enhanced_mag, enhanced_pha)`` spectrogram
tensors and returns a single perceptual score. PASE's ``PASEDiscriminator``
(MPD+MBD, BigVGAN/Vocos-style) instead scores raw waveforms and returns
*lists* of per-sub-discriminator outputs/feature-maps, and PASE itself is a
pure waveform-in/waveform-out model (no framework STFT).

Rather than duplicating ``GANTrainer._train_epoch``'s accumulation/AMP/DDP/
logging plumbing, this subclass repurposes the 4-tuple slots that
``_extract_gan_mag_pha`` normally fills with (mag, pha) spectra to instead
carry ``(clean_wav, clean_wav, enhanced_wav, enhanced_wav)`` — the waveform
duplicated into both slots so that ``GANTrainer._train_step_discriminator``'s
unconditional ``enh_pha.detach()`` call stays valid (it would raise
``AttributeError`` on ``None``). The duplicated 2nd/4th positional args are
ignored by the overridden loss-computation hooks below.

Usage (YAML):
    model:
      generator: {type: PASE, args: {...}}
      discriminator: {type: PASEDiscriminator, args: {...}}
    trainer:
      type: PASEGANTrainer
      adversarial_weight: 1.0   # required: real adv/feat weights come from lamda_adv/lamda_feat
      lamda_adv: 2.0            # PASE Phase-3 uses 2.0; StuPASE Mel-vocoder uses 1.0
      lamda_feat: 1.0
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import torch
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP

from core.losses.pase_family import (
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from core.utils.safe_load import safe_torch_load
from tasks.se.models.pase_family.common.utils import resample_waveform_and_lengths
from tasks.se.trainer.gan_trainer import GANTrainer
from tasks.se.trainer.val_audio import ValAudioWriter

logger = logging.getLogger(__name__)


_GAN_VAL_AUDIO_STAGES = frozenset(
    {"vocoder", "adapter", "postnet", "gap_postnet", "pase_dual"}
)


def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    """沿 DDP / torch.compile 包装解包到原始模型。"""
    while True:
        if isinstance(module, DDP):
            module = module.module
        elif hasattr(module, "_orig_mod"):
            module = module._orig_mod
        else:
            return module


def _forward_accepts_fs(module: torch.nn.Module) -> bool:
    """解包 DDP / torch.compile 包装后，检测 ``module.forward`` 是否声明 ``fs``。

    DDP 与 ``torch.compile`` 的 forward 签名都是 ``(*args, **kwargs)``，直接
    反射会漏判；因此沿 ``.module`` / ``_orig_mod`` 解包到原始模型。
    """
    return "fs" in inspect.signature(_unwrap_module(module).forward).parameters


def load_official_gan_warm_start(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    ckpt_path: str,
) -> dict:
    """恢复官方 train_vocoder.py 的 G/D/optimizer warm-start 状态。

    官方 checkpoint key 为 ``model`` / ``discriminator`` / ``optimizer_g`` /
    ``optimizer_d``（``train_vocoder.py:190-210``）。四个 key 缺一即 fail closed；
    scheduler 不在 checkpoint 内（官方构造新 scheduler）。
    """
    ckpt = safe_torch_load(ckpt_path)
    required = ("model", "discriminator", "optimizer_g", "optimizer_d")
    missing = [key for key in required if key not in ckpt]
    if missing:
        raise KeyError(
            f"official GAN warm-start checkpoint {ckpt_path} missing keys {missing}; "
            f"found {sorted(ckpt.keys())}"
        )
    _unwrap_module(generator).load_state_dict(ckpt["model"], strict=True)
    _unwrap_module(discriminator).load_state_dict(ckpt["discriminator"], strict=True)
    optimizer_g.load_state_dict(ckpt["optimizer_g"])
    optimizer_d.load_state_dict(ckpt["optimizer_d"])
    return ckpt


class PASEGANTrainer(GANTrainer):
    """GAN trainer for waveform-native PASE/StuPASE generators with MPD+MBD discriminators."""

    def __init__(self, **kw: Any):
        super().__init__(**kw)
        tc = self.config.get("trainer", {})
        self.lamda_adv = float(tc.get("lamda_adv", 2.0))
        self.lamda_feat = float(tc.get("lamda_feat", 1.0))
        warm_start_ckpt = tc.get("official_warm_start_ckpt")
        if warm_start_ckpt:
            load_official_gan_warm_start(
                self.generator,
                self.discriminator,
                self.optim_g,
                self.optim_d,
                str(warm_start_ckpt),
            )
            if self.rank == 0:
                logger.info(
                    "Loaded official GAN warm start (G/D/optimizers) from %s",
                    warm_start_ckpt,
                )
        # 固定输出率 wrapper（Mel vocoder / UniPASE Vocoder/PostNet / GAP PostNet）
        # 接受 fs 并按 batch 实际采样率预处理输入；不支持 fs 的 generator 保持原调用。
        raw_generator = _unwrap_module(self.generator)
        self._generator_accepts_fs = "fs" in inspect.signature(raw_generator.forward).parameters
        output_sr = getattr(raw_generator, "output_sample_rate", None)
        if output_sr is not None and (type(output_sr) is not int or output_sr <= 0):
            raise ValueError(
                "generator.output_sample_rate must be an exact positive integer, "
                f"got {output_sr!r}"
            )
        self._generator_output_sample_rate = output_sr
        self._val_audio = ValAudioWriter.from_config(
            self.config,
            rank=self.rank,
            device=self.device,
            sample_rate=self.sample_rate,
        )
        if self._val_audio is not None:
            stage = getattr(raw_generator, "val_audio_stage", None)
            if stage not in _GAN_VAL_AUDIO_STAGES:
                raise ValueError(
                    f"{type(raw_generator).__name__} does not support "
                    f"trainer.val_audio (stage={stage!r}); supported stages: "
                    f"{sorted(_GAN_VAL_AUDIO_STAGES)}"
                )
            self._val_audio.validate_roles(
                getattr(raw_generator, "val_audio_vocoder_roles", ())
            )

    def _forward_generator(self, batch_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Pure waveform pipeline: no framework STFT.

        generator 声明了 ``fs`` 时传入 batch 实际采样率（多采样率训练），
        否则维持 ``generator(noisy, lengths)`` 调用。固定输出率 generator
        （声明 ``output_sample_rate``）还会把 clean/lengths 对齐到输出率，
        避免多采样率输入下 enhanced 与 clean 的采样率/长度不一致。
        """
        input_sr = self._batch_fs_hz(batch_dict.get("fs"), batch_dict["noisy"].shape[0])
        if self._generator_accepts_fs:
            enhanced = self.generator(batch_dict["noisy"], batch_dict["lengths"], fs=input_sr)
        else:
            enhanced = self.generator(batch_dict["noisy"], batch_dict["lengths"])
        output_sr = self._generator_output_sample_rate
        if output_sr is not None and input_sr != output_sr:
            clean, lengths = resample_waveform_and_lengths(
                batch_dict["clean"], batch_dict.get("lengths"), input_sr, output_sr
            )
            # 固定输出率阶段：loss context 会用 noisy - clean 构造 observation，
            # 必须把 noisy 一并对齐到输出率，否则尺寸不一致。
            noisy = torchaudio.functional.resample(
                batch_dict["noisy"], orig_freq=input_sr, new_freq=output_sr
            )
            return {
                **batch_dict,
                "clean": clean,
                "noisy": noisy,
                "lengths": lengths,
                "fs": output_sr,
                "enhanced": enhanced,
            }
        return {**batch_dict, "enhanced": enhanced}

    def _extract_gan_mag_pha(
        self, fwd: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Repurpose the 4-tuple slots to carry waveforms instead of mag/pha spectra.

        Both slots per side are set to the same waveform tensor (rather than
        ``None``) because ``GANTrainer._train_step_discriminator`` unconditionally
        calls ``.detach()`` on the 2nd/4th slot.
        """
        clean = fwd["clean"]
        enhanced = fwd["enhanced"]
        return clean, clean, enhanced, enhanced

    def _compute_discriminator_loss(
        self,
        clean_wav: torch.Tensor,
        _clean_dup: torch.Tensor,
        enh_wav_detached: torch.Tensor,
        _enh_dup: torch.Tensor,
        real_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y_d_rs, y_d_gs, _fmap_rs, _fmap_gs = self.discriminator(clean_wav, enh_wav_detached)
        loss_d, _r_losses, _g_losses = discriminator_loss(y_d_rs, y_d_gs)
        return loss_d

    def _compute_generator_adversarial_loss(
        self,
        clean_wav: torch.Tensor,
        _clean_dup: torch.Tensor,
        enh_wav: torch.Tensor,
        _enh_dup: torch.Tensor,
    ) -> torch.Tensor:
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(clean_wav, enh_wav)
        loss_adv, _gen_losses = generator_loss(y_d_gs)
        loss_feat = feature_loss(fmap_rs, fmap_gs)
        return self.lamda_adv * loss_adv + self.lamda_feat * loss_feat

    def _compute_validation_loss(self, forward_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """官方 GAN validation：reconstruction + adv + feature matching。

        ``GANTrainer`` 的通用 validation 只算 reconstruction；PASE 家族官方
        ``_validation_epoch`` 与训练 generator loss 同式（如
        ``unipase/train/train_adapter.py``、``gap-urgenet/train/train_postnet.py``），
        因此这里在重建项上追加同一组对抗/特征匹配项，避免 best checkpoint 判据
        与官方不一致。
        """
        loss_rec = super()._compute_validation_loss(forward_dict)
        clean_mag, clean_pha, enh_mag, enh_pha = self._extract_gan_mag_pha(forward_dict)
        loss_adv = self._compute_generator_adversarial_loss(
            clean_mag, clean_pha, enh_mag, enh_pha
        )
        return loss_rec + self.adversarial_weight * loss_adv

    def _validation_epoch(self, epoch: int):
        """Discriminator 在 validation 期间切 eval，结束后恢复 train。"""
        self.discriminator.eval()
        try:
            return super()._validation_epoch(epoch)
        finally:
            self.discriminator.train()

    def _maybe_save_val_audio(self, batch, epoch: int) -> None:
        """官方各阶段 ``_validation_epoch`` 的 rank-0 验证音频写入。"""
        if self._val_audio is None or batch.context is None:
            return
        forward_dict = batch.context["forward_dict"]
        fs = self._batch_fs_hz(forward_dict.get("fs"), forward_dict["clean"].shape[0])
        self._val_audio.capture(
            _unwrap_module(self.generator),
            forward_dict["noisy"],
            forward_dict["clean"],
            forward_dict["lengths"],
            fs=fs,
            epoch=epoch,
            step=int(batch.context["step"]),
            start_epoch=self.start_epoch,
            model_kwargs={
                "forward_dict": forward_dict,
                "raw_enhanced": batch.context["raw_enhanced"],
            },
        )
