"""UniPASEAdapterGANTrainer: UniPASE Stage 3（VocosAdapter）特征域 GAN trainer。

对齐官方 ``unipase/train/train_adapter.py:240-277`` 的单步逻辑：

    target, _   = encoder(clean, mask=False)      # clean L1 特征（MSE 目标）
    input_a, input_p = encoder(noisy, mask=True)  # noisy L1 / PLC 后 L24
    output      = adapter(input_a, input_p)       # 增强 L1 特征 (B, T, 1024)
    loss_g = 200 * MSE(output, target) + 1 * LSGAN_adv + 1 * feature_matching
    loss_d = LSGAN discriminator loss

与 ``PASEGANTrainer`` 的差异仅在「判别器/重建损失作用在 WavLM 特征而非波形」：

- generator 是 ``UniPASEAdapterGenerator``，其 ``forward(noisy, clean)`` 返回
  ``(enh_feat, target_feat)``；本 trainer 覆写 ``_forward_generator`` 把 clean
  一并传入（``GANTrainer._unpack_batch`` 的 batch_dict 本就含 clean，零侵入）。
- 判别器是 ``UniPASEMultiScaleDiscriminator``（特征域多尺度 1D 判别器），其
  输出结构与 ``PASEDiscriminator`` 相同（score/feature-map 列表），因此
  ``core.losses.pase_family`` 的 ``discriminator_loss`` /
  ``generator_loss`` / ``feature_loss`` 可直接复用——三者与官方
  ``unipase/utils/loss_gan.py:217-257`` 逐式一致（同源 BigVGAN LSGAN）。
- 验证阶段输出是特征而非波形：``_sanitize_model_output`` 的 clamp(-1, 1)
  会破坏特征分布（layer_norm 后的特征天然超出 [-1,1]），因此本 trainer
  将其退化为仅 nan_to_num；``_compute_validation_loss`` 由 ``PASEGANTrainer``
  统一为特征重建 + adv + feature matching（与官方 train_adapter.py 的验证式一致）。
- 验证音频由父类 ``PASEGANTrainer._maybe_save_val_audio`` 驱动；本阶段的
  ``UniPASEAdapterGenerator.val_audio_outputs`` 复用 validation 前向的
  ``target_feat``/``enhanced``，并用两个冻结 vocoder 还原 recon/enh0/enhanced。

用法（YAML）::

    model:
      generator: {type: UniPASEAdapterGenerator, args: {...}}
      discriminator: {type: UniPASEMultiScaleDiscriminator, args: {in_dim: 1024, ...}}
    trainer:
      type: tasks.se.trainer.unipase_adapter_gan_trainer.UniPASEAdapterGANTrainer
      adversarial_weight: 1.0
      lamda_recons: 200.0   # 官方 cfg_train_adapter.yaml coeff.recons
      lamda_adv: 1.0        # 官方 coeff.adv
      lamda_feat: 1.0       # 官方 coeff.feat
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from tasks.se.trainer.pase_gan_trainer import PASEGANTrainer


class UniPASEAdapterGANTrainer(PASEGANTrainer):
    """特征域（WavLM L1）GAN trainer：MSE×200 + LSGAN adv×1 + feat×1（官方默认）。"""

    SUPPORTS_WAVEFORM_VAL_METRICS = False
    VALIDATION_OUTPUT_DOMAIN = "wavlm_feature"

    def __init__(self, **kw: Any):
        super().__init__(**kw)
        tc = self.config.get("trainer", {})
        self.lamda_recons = float(tc.get("lamda_recons", 200.0))

    # ------------------------------------------------------------------
    # 前向：generator 需要同时看到 noisy 与 clean（clean L1 特征作 MSE 目标）
    # ------------------------------------------------------------------
    def _forward_generator(self, batch_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # 官方 train_adapter.py 传 sr=fs；不传会静默按 16k 处理（review P2-6）
        fs = self._batch_fs_hz(batch_dict.get("fs"), batch_dict["noisy"].shape[0])
        enh_feat, target_feat = self.generator(
            batch_dict["noisy"], batch_dict["clean"], batch_dict["lengths"], fs=fs
        )
        return {**batch_dict, "enhanced": enh_feat, "target_feat": target_feat}

    def _extract_gan_mag_pha(
        self, fwd: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """复用 4 元组槽位承载特征（语义同 PASEGANTrainer：2/4 槽位仅占位）。"""
        target = fwd["target_feat"]
        enhanced = fwd["enhanced"]
        return target, target, enhanced, enhanced

    # ------------------------------------------------------------------
    # 损失：重建项为特征 MSE × lamda_recons（官方 train_adapter.py:245）
    # ------------------------------------------------------------------
    def _compute_reconstruction_loss(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor,
        forward_dict: dict[str, torch.Tensor] | None = None,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        # clean 槽位是波形，特征 MSE 目标在 forward_dict["target_feat"]
        if forward_dict is None or "target_feat" not in forward_dict:
            # fail closed：缺 target 时若拿波形当目标会产生难以理解的 broadcast 报错
            raise ValueError(
                "UniPASEAdapterGANTrainer 需要 forward_dict['target_feat'] 作为特征 MSE 目标"
            )
        return self.lamda_recons * F.mse_loss(enhanced, forward_dict["target_feat"])

    # ------------------------------------------------------------------
    # 输出 sanitize：特征是 layer_norm 分布，clamp(-1,1) 会破坏数值；
    # 仅保留 nan_to_num（波形 clamp 语义对本 stage 不适用）
    # ------------------------------------------------------------------
    def _sanitize_model_output(self, wav: torch.Tensor) -> torch.Tensor:
        out = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
        n_nonfinite = int((~torch.isfinite(wav)).sum().item())
        self._accumulate_output_sanitize(n_nonfinite, 0)
        return out
