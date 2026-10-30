"""GAN Trainer for speech enhancement and AEC (MetricGAN-style).

Supports alternating generator/discriminator training with multi-model
and multi-optimizer infrastructure.

Usage:
    # YAML config
    model:
      generator:
        type: GTCRN
        args: {...}
      discriminator:
        type: MetricDiscriminator
        args: {...}

    optimizer:
      generator:
        type: AdamW
        args: {lr: 0.001}
      discriminator:
        type: AdamW
        args: {lr: 0.0005}

    trainer:
      type: GANTrainer
      discriminator_steps: 1
      generator_steps: 1  # 当前仅支持 1；>1 需每个子步重新 forward（未实现），__init__ 会拒绝
      adversarial_weight: 0.05
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from core.trainer.metrics_trainer import (
    _direct_spectral_loss_kwargs,
    validate_waveform_lengths,
)
from core.utils.ddp_error import is_unrecoverable_cuda_exception

from tasks.se.data.batch import SEBatch
from tasks.se.trainer.metrics_trainer import (
    SETrainerBase,
    _loss_requires_noisy_reference,
    _loss_requires_sample_rate,
    _se_loss_kwargs,
    _unpack_se_batch,
)

logger = logging.getLogger(__name__)


def _extract_mag_pha(spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract magnitude and phase from real spectrum (B, F, T, 2)."""
    if spec.is_complex():
        mag = spec.abs()
        pha = torch.angle(spec)
    else:
        re, im = spec[..., 0], spec[..., 1]
        mag = torch.sqrt(re**2 + im**2 + 1e-8)
        pha = torch.atan2(im, re)
    return mag, pha


class GANTrainer(SETrainerBase):
    """MetricGAN-style trainer with alternating G/D steps.

    Expects ``model`` and ``optimizer`` to be dicts with at least
    ``"generator"`` and ``"discriminator"`` keys. DDP training requires
    ``distributed.find_unused_parameters=true`` because each alternating backward
    pass freezes the other model. G 子步通过冻结 D 的底层 module 前向，保留对 G
    输入的梯度，但不重新启动 D 的 reducer。非有限 loss 仅沿当前子步真实图回退。

    Subclasses (e.g. ``MetricGANCompositeTrainer``, ``AECGANTrainer``) should override
    ``_unpack_batch`` and ``_forward_generator`` to adapt to specific tasks.
    """

    # validation 验证音频写入器；未启用时为 None（直接构造/测试路径也安全）。
    _val_audio = None

    def __init__(self, **kw: Any):
        # BaseTrainer handles dict models/optimizers automatically
        super().__init__(**kw)

        # Validate multi-model / multi-optimizer setup
        if not self._is_multi_model:
            raise ValueError(
                "GANTrainer requires a dict of models (generator + discriminator). "
                "Pass model={'generator': gen, 'discriminator': disc}"
            )
        if not self._is_multi_optimizer:
            raise ValueError(
                "GANTrainer requires a dict of optimizers (generator + discriminator). "
                "Pass optimizer={'generator': optim_g, 'discriminator': optim_d}"
            )

        required_models = {"generator", "discriminator"}
        missing_models = required_models - set(self.models.keys())
        if missing_models:
            raise ValueError(f"GANTrainer missing models: {missing_models}. Got: {list(self.models.keys())}")

        required_optims = {"generator", "discriminator"}
        missing_optims = required_optims - set(self.optimizers.keys())
        if missing_optims:
            raise ValueError(f"GANTrainer missing optimizers: {missing_optims}. Got: {list(self.optimizers.keys())}")

        self.generator = self.models["generator"]
        self.discriminator = self.models["discriminator"]
        self.optim_g = self.optimizers["generator"]
        self.optim_d = self.optimizers["discriminator"]

        if self.world_size > 1 and not self.config.get("distributed", {}).get(
            "find_unused_parameters", False
        ):
            raise ValueError(
                "GANTrainer requires distributed.find_unused_parameters=true under "
                "DDP because alternating G/D steps freeze the other wrapper."
            )

        # Only generator gets EMA updates
        self._ema_targets = {"generator"}

        # Training hyperparameters from config
        tc = self.config.get("trainer", {})
        self.discriminator_steps = int(tc.get("discriminator_steps", 1))
        if self.discriminator_steps < 1:
            raise ValueError(
                f"discriminator_steps={self.discriminator_steps} is invalid; "
                "Set trainer.discriminator_steps to an integer >= 1."
            )
        self.generator_steps = int(tc.get("generator_steps", 1))
        # generator_steps>1 会对同一个 enhanced 计算图重复 backward（forward 每个
        # micro-batch 只做一次），第二个子步必抛 "backward through the graph a
        # second time"。多步 G 更新需要每个子步重新 forward，目前未实现，
        # 因此在初始化时显式拒绝，避免训练中途崩溃。
        if self.generator_steps != 1:
            raise ValueError(
                f"generator_steps={self.generator_steps} is not supported: "
                "multiple generator sub-steps would backward through the same "
                "graph without a re-forward (not implemented). "
                "Set trainer.generator_steps: 1."
            )
        self.adversarial_weight = float(tc.get("adversarial_weight", 0.05))
        self._log_interval_d = int(tc.get("log_interval_discriminator", 0)) or self.log_interval

        logger.info(
            "GANTrainer initialized: D_steps=%d, G_steps=%d, adv_weight=%.3f",
            self.discriminator_steps, self.generator_steps, self.adversarial_weight,
        )

    # ------------------------------------------------------------------
    # Hooks for subclasses (SE vs AEC)
    # ------------------------------------------------------------------
    def _unpack_batch(self, batch: SEBatch) -> dict[str, torch.Tensor]:
        """Unpack DataLoader batch into tensors.

        复用 SE 的结构化批校验入口，
        使 GAN 循环和 ``SETrainerBase`` 共用同一批合同实现，非法类型或字段
        在 ``_unpack_se_batch`` / ``SEBatch.validate()`` 边界 fail-fast。
        """
        se_batch = _unpack_se_batch(batch)
        out: dict[str, torch.Tensor] = {
            "clean": se_batch.clean,
            "noisy": se_batch.noisy,
            "fs": se_batch.sr,
            "lengths": se_batch.lengths,
        }
        for key in ("is_pure_noise", "is_pure_echo", "is_pure_echo_noise"):
            value = getattr(se_batch, key)
            if value is not None:
                out[key] = value
        return out

    def _validate_batch_metadata(
        self, batch: Any, epoch: int, batch_idx: int
    ) -> None:
        if isinstance(batch, dict):
            validate_waveform_lengths(
                batch["lengths"],
                batch_size=batch["clean"].shape[0],
                max_length=batch["clean"].shape[-1],
                context=f"GAN train E{epoch} B{batch_idx}",
            )

    def _sanitize_batch_inputs(
        self, batch: Any, epoch: int, batch_idx: int
    ) -> Any:
        if not isinstance(batch, dict):
            return batch
        if getattr(self, "_strict_checks", False):
            self._assert_batch_valid(batch["clean"], batch["noisy"], batch["lengths"])
        nonfinite_fields = [
            key
            for key in ("clean", "noisy")
            if key in batch
            and isinstance(batch[key], torch.Tensor)
            and not torch.isfinite(batch[key]).all()
        ]
        for key in nonfinite_fields:
            batch[key] = torch.nan_to_num(
                batch[key], nan=0.0, posinf=0.0, neginf=0.0
            )
        if nonfinite_fields and self.rank == 0:
            logger.warning(
                "[GAN E%s B%s] NaN/Inf input fields %s, non-finite values zeroed out",
                epoch,
                batch_idx,
                ", ".join(nonfinite_fields),
            )
        return batch

    def _forward_generator(self, batch_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Generator forward pass. Subclasses override for task-specific logic.

        Default assumes spectrum-in/spectrum-out with framework STFT:
        ``noisy`` waveform → STFT → model → iSTFT → ``enhanced`` waveform.

        Compatible with GTCRN / BSRNN-style ``forward(noisy_spec, lengths)``.
        AEC models that use dict inputs should override this method.
        """
        stft = self.stft_module
        if stft is None:
            raise RuntimeError("GANTrainer default _forward_generator requires stft_module")

        noisy = batch_dict["noisy"]
        lengths = batch_dict["lengths"]
        if noisy.ndim == 3:
            raise ValueError(
                "default GANTrainer does not support spatial SE noisy input "
                "with shape [B,C,T]"
            )

        noisy_spec = stft(noisy)
        enhanced_spec = self.generator(noisy_spec, lengths)
        enhanced = stft.inverse(enhanced_spec, noisy.shape[-1])

        return {
            **batch_dict,
            "noisy_spec": noisy_spec,
            "enhanced_spec": enhanced_spec,
            "enhanced": enhanced,
        }

    def _sanitize_forward_output(
        self, forward_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """保留 generator 原始预测谱；sanitize 波形后刷新判别器使用的谱。"""
        if "enhanced_spec" in forward_dict:
            forward_dict["loss_enhanced_spectra"] = forward_dict["enhanced_spec"]
        forward_dict["enhanced"] = self._sanitize_model_output(forward_dict["enhanced"])
        if "enhanced_spec" in forward_dict and self.stft_module is not None:
            forward_dict["enhanced_spec"] = self.stft_module(forward_dict["enhanced"])
        return forward_dict

    def _get_discriminator_real_scores(
        self, forward_dict: dict[str, torch.Tensor]
    ) -> torch.Tensor | None:
        """Hook: return real perceptual scores for discriminator training.

        Subclasses (e.g. MetricGANCompositeTrainer) override to inject
        real-time PESQ or other metric scores. Default returns None.
        """
        return None

    def _reconstruction_loss_kwargs(
        self,
        forward_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """按活动 loss capability 从增广后的 forward dict 组装 SE 上下文。"""
        enhanced = forward_dict["enhanced"]
        clean = forward_dict["clean"]
        lengths = forward_dict["lengths"]
        noisy = forward_dict.get("noisy")
        inner_loss = getattr(self.loss_fn, "loss", self.loss_fn)
        needs_noisy = _loss_requires_noisy_reference(self.loss_fn) or bool(
            getattr(inner_loss, "any_ss_enabled", False)
        )
        if needs_noisy and noisy is None:
            raise ValueError(
                "GAN reconstruction loss requires forward_dict['noisy'] because "
                "an active loss component requires a noisy/undesired reference"
            )

        sample_rate_value = forward_dict.get("fs")
        if sample_rate_value is None:
            sample_rate_value = forward_dict.get("sr")
        if _loss_requires_sample_rate(self.loss_fn) and sample_rate_value is None:
            raise ValueError(
                "GAN reconstruction loss requires forward_dict['fs'] or "
                "forward_dict['sr'] because an active loss component requires "
                "the actual batch sample rate"
            )
        sample_rate = self._batch_fs_hz(sample_rate_value, clean.shape[0])
        batch = SEBatch(
            clean=clean,
            noisy=noisy,
            sr=sample_rate_value,
            lengths=lengths,
            is_pure_noise=forward_dict.get("is_pure_noise"),
            is_pure_echo=forward_dict.get("is_pure_echo"),
            is_pure_echo_noise=forward_dict.get("is_pure_echo_noise"),
        )
        out = _se_loss_kwargs(
            self.loss_fn,
            batch,
            enhanced,
            sample_rate,
            noisy=noisy,
            clean=clean,
        )
        out.update(
            _direct_spectral_loss_kwargs(
                self.loss_fn,
                getattr(self, "stft_module", None),
                clean,
                forward_dict.get("loss_enhanced_spectra"),
                lengths=lengths,
            )
        )
        return out

    def _compute_reconstruction_loss(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor,
        forward_dict: dict[str, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Reconstruction loss (existing supervised loss).

        基类按活动 loss capability 从 ``forward_dict`` 注入 SE 上下文；子类仍可
        覆盖本 hook，并继续收到原有的 ``forward_dict`` / pure flags。
        """
        if forward_dict is not None:
            kwargs = self._reconstruction_loss_kwargs(forward_dict)
        return self.loss_fn(enhanced, clean, lengths, **kwargs)

    def _compute_discriminator_loss(
        self,
        clean_mag: torch.Tensor,
        clean_pha: torch.Tensor,
        enhanced_mag: torch.Tensor,
        enhanced_pha: torch.Tensor,
        real_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Discriminator loss: MSE against real scores (MetricGAN style).

        Args:
            real_scores: Optional real perceptual scores (e.g. PESQ) for the batch.
                        If None, uses all-ones for real and zeros for fake.
        """
        score_real = self.discriminator(clean_mag, clean_pha, clean_mag, clean_pha)
        score_fake = self.discriminator(clean_mag, clean_pha, enhanced_mag, enhanced_pha)

        if real_scores is not None:
            # MetricGAN: D learns to score clean as 1.0 and enhanced as its actual PESQ
            loss_d_real = F.mse_loss(score_real, torch.ones_like(score_real))
            loss_d_fake = F.mse_loss(score_fake, real_scores)
        else:
            loss_d_real = F.mse_loss(score_real, torch.ones_like(score_real))
            loss_d_fake = F.mse_loss(score_fake, torch.zeros_like(score_fake))

        return loss_d_real + loss_d_fake

    def _compute_generator_adversarial_loss(
        self,
        clean_mag: torch.Tensor,
        clean_pha: torch.Tensor,
        enhanced_mag: torch.Tensor,
        enhanced_pha: torch.Tensor,
    ) -> torch.Tensor:
        """Generator adversarial loss: fool discriminator into scoring high."""
        score = self.discriminator(clean_mag, clean_pha, enhanced_mag, enhanced_pha)
        return F.mse_loss(score, torch.ones_like(score))

    def _gan_skip_optimizer_steps(self) -> bool:
        """清理 GAN 两侧梯度，且只在 G 侧推进一次 scaler bookkeeping。"""
        prepared_d = self._prepare_optimizer_step("discriminator", skip_step=True)
        prepared_g = self._prepare_optimizer_step("generator", skip_step=True)
        self._discard_prepared_optimizer_step(
            prepared_d, update_scaler=False
        )
        self._discard_prepared_optimizer_step(
            prepared_g, update_scaler=True
        )
        return False

    def _gan_optimizer_step(self, *, step_allowed: bool = True) -> bool:
        """先 prepare D/G 并做全局 gate，再按 D→G 原子提交一对 step。"""
        window_eligible = bool(
            step_allowed
            and getattr(self, "_gan_window_d_finite", True)
            and getattr(self, "_gan_window_g_finite", True)
        )
        prepared_d = self._prepare_optimizer_step(
            "discriminator",
            skip_step=not window_eligible,
        )
        prepared_g = self._prepare_optimizer_step(
            "generator",
            skip_step=not window_eligible,
        )
        from core.trainer.optimizer_step import paired_optimizer_gate_values

        local_gate = paired_optimizer_gate_values(
            (prepared_d, prepared_g),
            window_eligible=window_eligible,
        )
        eligible, amp_finite = self._allreduce_bool_ands(
            local_gate,
            phase="gan_optimizer_step_prepare",
        )
        if not eligible:
            self._discard_prepared_optimizer_step(
                prepared_d,
                update_scaler=False,
                amp_overflow=not amp_finite,
            )
            self._discard_prepared_optimizer_step(
                prepared_g,
                update_scaler=True,
                amp_overflow=not amp_finite,
            )
            return False

        self._commit_prepared_optimizer_step(
            prepared_d,
            update_scaler=False,
            increment_step=False,
        )
        self._commit_prepared_optimizer_step(
            prepared_g,
            update_scaler=True,
            increment_step=True,
        )
        return True

    # ------------------------------------------------------------------
    # Training loop hooks
    # ------------------------------------------------------------------
    def _extract_gan_mag_pha(
        self, fwd: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """从 forward dict 中提取 clean/enhanced 的 mag/pha。"""
        if "enhanced_spec" in fwd:
            enh_mag, enh_pha = _extract_mag_pha(fwd["enhanced_spec"])
        else:
            if self.stft_module is None:
                raise RuntimeError("GANTrainer needs stft_module or enhanced_spec")
            enh_spec = self.stft_module(fwd["enhanced"])
            enh_mag, enh_pha = _extract_mag_pha(enh_spec)

        if "clean_spec" in fwd:
            clean_mag, clean_pha = _extract_mag_pha(fwd["clean_spec"])
        else:
            if self.stft_module is None:
                raise RuntimeError("GANTrainer needs stft_module or clean_spec")
            clean_spec = self.stft_module(fwd["clean"])
            clean_mag, clean_pha = _extract_mag_pha(clean_spec)

        return clean_mag, clean_pha, enh_mag, enh_pha

    def _prepare_discriminator_losses(
        self,
        clean_mag: torch.Tensor,
        clean_pha: torch.Tensor,
        enh_mag: torch.Tensor,
        enh_pha: torch.Tensor,
        real_scores: torch.Tensor | None,
        accum: int,
    ) -> list[torch.Tensor]:
        """在 DDP backward 前构建 D loss，供跨 rank 异常同步。"""
        if not hasattr(self, "_gan_window_d_finite"):
            self._gan_window_d_finite = True
        losses: list[torch.Tensor] = []
        _g_req = [parameter.requires_grad for parameter in self.generator.parameters()]
        self.generator.requires_grad_(False)
        try:
            for _ in range(self.discriminator_steps):
                with self._capture_forward_graph(self.discriminator) as forward_graph:
                    loss_d = self._compute_discriminator_loss(
                        clean_mag,
                        clean_pha,
                        enh_mag.detach(),
                        enh_pha.detach(),
                        real_scores=real_scores,
                    )
                    loss_d = loss_d / accum / self.discriminator_steps
                    if not torch.isfinite(loss_d).all():
                        self._gan_window_d_finite = False
                        if self.rank == 0:
                            logger.warning(
                                "[step %s] NaN/Inf in loss_d, using zero D loss",
                                self.n_steps,
                            )
                        loss_d = self._graph_preserving_zero_loss(loss_d, outputs=forward_graph)
                    losses.append(loss_d)
        finally:
            for parameter, requires_grad in zip(self.generator.parameters(), _g_req):
                parameter.requires_grad_(requires_grad)
        return losses

    def _train_step_discriminator(
        self,
        clean_mag: torch.Tensor,
        clean_pha: torch.Tensor,
        enh_mag: torch.Tensor,
        enh_pha: torch.Tensor,
        real_scores: torch.Tensor | None,
        accum: int,
        *,
        prepared_losses: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """对已同步预检的 discriminator loss 执行 backward。"""
        losses = prepared_losses
        if losses is None:
            losses = self._prepare_discriminator_losses(
                clean_mag, clean_pha, enh_mag, enh_pha, real_scores, accum
            )
        loss_d_total = torch.zeros(1, device=self.device)
        for loss_d in losses:
            self.scaler.scale(loss_d).backward()
            loss_d_total += loss_d.detach()
        return loss_d_total

    def _prepare_generator_losses(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor,
        clean_mag: torch.Tensor,
        clean_pha: torch.Tensor,
        enh_mag: torch.Tensor,
        enh_pha: torch.Tensor,
        fwd: dict[str, torch.Tensor],
        accum: int,
    ) -> list[torch.Tensor]:
        """在 DDP backward 前构建已冻结 D 的 G loss，供跨 rank 异常同步。"""
        if not hasattr(self, "_gan_window_g_finite"):
            self._gan_window_g_finite = True
        losses: list[torch.Tensor] = []
        for _ in range(self.generator_steps):
            loss_kwargs = {
                key: fwd[key]
                for key in ("is_pure_noise", "is_pure_echo", "is_pure_echo_noise")
                if key in fwd
            }
            try:
                loss_rec = self._compute_reconstruction_loss(
                    enhanced, clean, lengths, forward_dict=fwd, **loss_kwargs
                )
            except TypeError as exc:
                if "unexpected keyword argument" in str(exc):
                    loss_rec = self._compute_reconstruction_loss(enhanced, clean, lengths)
                else:
                    raise
            loss_adv = self._compute_generator_adversarial_loss(
                clean_mag, clean_pha, enh_mag, enh_pha,
            )
            loss_g = (
                (loss_rec + self.adversarial_weight * loss_adv)
                / accum
                / self.generator_steps
            )
            if not torch.isfinite(loss_g).all():
                self._gan_window_g_finite = False
                if self.rank == 0:
                    logger.warning(
                        "[step %s] NaN/Inf in loss_g, using zero G loss",
                        self.n_steps,
                    )
                # fwd 保留本 batch 的 G 图，不重用已 backward 的 D 子步输出。
                loss_g = self._graph_preserving_zero_loss(loss_g, outputs=fwd)
            losses.append(loss_g)
        return losses

    def _train_step_generator(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor,
        clean_mag: torch.Tensor,
        clean_pha: torch.Tensor,
        enh_mag: torch.Tensor,
        enh_pha: torch.Tensor,
        fwd: dict[str, torch.Tensor],
        accum: int,
        *,
        prepared_losses: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """对已同步预检的 generator loss 执行 backward。"""
        losses = prepared_losses
        if losses is None:
            _d_req = [parameter.requires_grad for parameter in self.discriminator.parameters()]
            self.discriminator.requires_grad_(False)
            discriminator_wrapper = self.discriminator
            if isinstance(self.discriminator, torch.nn.parallel.DistributedDataParallel):
                self.discriminator = self.discriminator.module
            try:
                losses = self._prepare_generator_losses(
                    enhanced, clean, lengths, clean_mag, clean_pha, enh_mag, enh_pha,
                    fwd, accum,
                )
            finally:
                self.discriminator = discriminator_wrapper
                for parameter, requires_grad in zip(
                    self.discriminator.parameters(), _d_req
                ):
                    parameter.requires_grad_(requires_grad)
        loss_g_total = torch.zeros(1, device=self.device)
        for loss_g in losses:
            self.scaler.scale(loss_g).backward()
            loss_g_total += loss_g.detach()
        return loss_g_total

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def _train_epoch(
        self,
        epoch: int,
        step_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> Optional[float]:
        # 与 MetricsCapableTrainer 对齐：直接调用本方法时也必须经过 DDP max-step
        # 边界，不能让某个 rank 先进入 batch 循环。
        if self._sync_max_steps_reached():
            self._train_epoch_completed = False
            self._last_completed_batch_cursor = int(
                getattr(self, "_resume_skip_batches", 0)
            )
            return None

        running_loss_g = torch.zeros(1, device=self.device)
        running_loss_d = torch.zeros(1, device=self.device)
        committed_batches = 0
        accum = self.accumulation_steps
        accum_count = 0
        window_loss_g = torch.zeros(1, device=self.device)
        window_loss_d = torch.zeros(1, device=self.device)
        window_batches = 0
        stopped_by_max_steps = False
        stopped_by_shutdown = False
        completed_batch_cursor = 0
        self._train_epoch_completed = False
        self._gan_window_d_finite = True
        self._gan_window_g_finite = True
        # A23：与 SE/AEC 一致，每个 train epoch 开始清零输出 sanitize 计数
        self._reset_output_sanitize_stats()

        def _commit_train_step_window() -> None:
            """只把真实 paired step 对应的 window 纳入 epoch/log loss。"""
            nonlocal committed_batches, window_batches
            if window_batches <= 0:
                return
            running_loss_g.add_(window_loss_g)
            running_loss_d.add_(window_loss_d)
            committed_batches += window_batches
            self._train_log_loss_win.append(
                window_loss_g.item() / window_batches
            )
            window_loss_g.zero_()
            window_loss_d.zero_()
            window_batches = 0

        def _discard_train_step_window() -> None:
            """丢弃未提交 window 的 loss，避免污染 epoch 平均值与日志。"""
            nonlocal window_batches
            window_loss_g.zero_()
            window_loss_d.zero_()
            window_batches = 0

        pbar = self._make_train_progress_bar(epoch)

        # A22：从 middle checkpoint resume 时跳过本 epoch 已完整完成的 batch
        # （语义同 MetricsCapableTrainer._train_epoch；GAN 当前不产生 middle
        # checkpoint，此处保持训练循环对 cursor 的一致处理）
        skip_batches = self._resume_skip_batches if epoch == self.start_epoch else 0
        # 立即停机时必须保留已有 resume cursor；不能回退为 0 或跳过当前 epoch。
        completed_batch_cursor = int(skip_batches)
        if skip_batches > 0 and self.rank == 0:
            logger.info(
                f"  Resuming epoch {epoch} from middle checkpoint: "
                f"skipping first {skip_batches} batches"
            )

        shutdown_synced_at_last_batch = False
        for bi, raw_batch in pbar:
            if bi < skip_batches:
                continue
            shutdown_synced_at_last_batch = False
            local_prepare_error: Exception | None = None
            prepared = None
            try:
                prepared = self._prepare_batch(raw_batch, epoch, bi)
            except Exception as exc:
                # 致命 CUDA 故障（OOM / device-side assert 等）不得进入 abort
                # collective：NCCL 可能已失效，直接交给 launcher 更快也更安全。
                if is_unrecoverable_cuda_exception(exc):
                    raise
                local_prepare_error = exc
            prepare_error_context = self._sync_train_batch_exception(
                local_prepare_error,
                phase="train_prepare_batch_error",
            )
            if prepare_error_context is not None:
                raise RuntimeError(
                    f"[E{epoch} B{bi}] batch preparation failed: "
                    f"{prepare_error_context}"
                ) from local_prepare_error
            if local_prepare_error is not None:
                raise local_prepare_error
            assert prepared is not None
            batch_dict, fs_val, _n_placeholders = prepared

            if accum_count == 0:
                self._gan_window_d_finite = True
                self._gan_window_g_finite = True
            is_last_accum = (accum_count + 1 >= accum)
            local_forward_error: Exception | None = None
            fwd = None
            forward_sync_ctx = (
                contextlib.nullcontext()
                if (is_last_accum or self.world_size <= 1)
                else self._no_sync_all()
            )
            try:
                with forward_sync_ctx, torch.amp.autocast(
                    self.device.type, enabled=self.use_amp, dtype=self.amp_dtype
                ):
                    fwd = self._sanitize_forward_output(
                        self._forward_generator(batch_dict)
                    )
                    clean_mag, clean_pha, enh_mag, enh_pha = self._extract_gan_mag_pha(
                        fwd
                    )
                    real_scores = self._get_discriminator_real_scores(fwd)
                    # 此段只覆盖 generator forward / 谱特征准备；D/G loss 各自
                    # 在对应 backward 前同步异常，保持既有 D→G forward/backward 时序。
            except Exception as exc:
                if is_unrecoverable_cuda_exception(exc):
                    raise
                local_forward_error = exc
            forward_error_context = self._sync_train_batch_exception(
                local_forward_error,
                phase="train_forward_loss_error",
            )
            if forward_error_context is not None or local_forward_error is not None:
                # 当前 graph 已放弃（本 rank forward 失败，或已同步到其他 rank
                # 的失败）：此时没有 backward，可立即修复被污染的持久状态。
                self._repair_persistent_state_after_abandoned_graph()
            if forward_error_context is not None:
                raise RuntimeError(
                    f"[E{epoch} B{bi}] forward/loss failed: {forward_error_context}"
                ) from local_forward_error
            if local_forward_error is not None:
                raise local_forward_error
            assert fwd is not None

            # 与 core 训练骨架一致：forward 污染了模型持久状态（如 BatchNorm
            # running buffers）时本 window 梯度不可信，按非有限 loss 处理，
            # 跳过成对 optimizer step；buffer 修复推迟到 backward 之后。
            if self._has_persistent_state_repair_pending():
                self._gan_window_d_finite = False
                self._gan_window_g_finite = False
                if self.rank == 0:
                    logger.warning(
                        "[step %s] persistent model state repair pending, "
                        "dropping the paired optimizer step",
                        self.n_steps,
                    )

            sync_ctx = (
                contextlib.nullcontext()
                if (is_last_accum or self.world_size <= 1)
                else self._no_sync_all()
            )
            with self._persistent_state_repair_guard(), sync_ctx:
                local_discriminator_error: Exception | None = None
                prepared_d_losses = None
                try:
                    with torch.amp.autocast(
                        self.device.type, enabled=self.use_amp, dtype=self.amp_dtype
                    ):
                        prepared_d_losses = self._prepare_discriminator_losses(
                            clean_mag, clean_pha, enh_mag, enh_pha, real_scores, accum
                        )
                except Exception as exc:
                    if is_unrecoverable_cuda_exception(exc):
                        raise
                    local_discriminator_error = exc
                discriminator_error_context = self._sync_train_batch_exception(
                    local_discriminator_error,
                    phase="train_discriminator_loss_error",
                )
                if discriminator_error_context is not None:
                    raise RuntimeError(
                        f"[E{epoch} B{bi}] discriminator loss failed: "
                        f"{discriminator_error_context}"
                    ) from local_discriminator_error
                if local_discriminator_error is not None:
                    raise local_discriminator_error
                assert prepared_d_losses is not None
                loss_d_total = self._train_step_discriminator(
                    clean_mag,
                    clean_pha,
                    enh_mag,
                    enh_pha,
                    real_scores,
                    accum,
                    prepared_losses=prepared_d_losses,
                )

                d_requires_grad = [
                    parameter.requires_grad for parameter in self.discriminator.parameters()
                ]
                self.discriminator.requires_grad_(False)
                discriminator_wrapper = self.discriminator
                if isinstance(self.discriminator, torch.nn.parallel.DistributedDataParallel):
                    self.discriminator = self.discriminator.module
                try:
                    local_generator_error: Exception | None = None
                    prepared_g_losses = None
                    try:
                        with torch.amp.autocast(
                            self.device.type, enabled=self.use_amp, dtype=self.amp_dtype
                        ):
                            prepared_g_losses = self._prepare_generator_losses(
                                fwd["enhanced"],
                                fwd["clean"],
                                fwd["lengths"],
                                clean_mag,
                                clean_pha,
                                enh_mag,
                                enh_pha,
                                fwd,
                                accum,
                            )
                    except Exception as exc:
                        if is_unrecoverable_cuda_exception(exc):
                            raise
                        local_generator_error = exc
                    generator_error_context = self._sync_train_batch_exception(
                        local_generator_error,
                        phase="train_generator_loss_error",
                    )
                    if generator_error_context is not None:
                        raise RuntimeError(
                            f"[E{epoch} B{bi}] generator loss failed: "
                            f"{generator_error_context}"
                        ) from local_generator_error
                    if local_generator_error is not None:
                        raise local_generator_error
                    assert prepared_g_losses is not None
                    loss_g_total = self._train_step_generator(
                        fwd["enhanced"],
                        fwd["clean"],
                        fwd["lengths"],
                        clean_mag,
                        clean_pha,
                        enh_mag,
                        enh_pha,
                        fwd,
                        accum,
                        prepared_losses=prepared_g_losses,
                    )
                finally:
                    self.discriminator = discriminator_wrapper
                    for parameter, requires_grad in zip(
                        self.discriminator.parameters(), d_requires_grad
                    ):
                        parameter.requires_grad_(requires_grad)

            # backward 已消费完 running statistics 的版本，此时修复被污染的
            # 持久 buffer 不会再触发 inplace version 错误（与 core 训练骨架一致）。
            self._flush_persistent_state_repair()
            self._maybe_log_step_diagnostics(
                batch_dict,
                {
                    "loss_was_finite": bool(
                        self._gan_window_d_finite and self._gan_window_g_finite
                    ),
                },
                epoch,
                bi,
            )

            accum_count += 1
            window_batches += 1
            window_loss_g.add_(loss_g_total * accum)
            window_loss_d.add_(loss_d_total * accum)

            max_steps_reached = False
            if is_last_accum:
                shutdown_synced_at_last_batch = True
                if self._sync_shutdown_signum():
                    stopped_by_shutdown = True
                    for opt in self.optimizers.values():
                        opt.zero_grad(set_to_none=True)
                    _discard_train_step_window()
                    accum_count = 0
                    if self.rank == 0:
                        logger.info(
                            f"  Shutdown requested, dropping GAN window ending "
                            f"at batch {bi}."
                        )
                else:
                    stepped = self._gan_optimizer_step(
                        step_allowed=(
                            self._gan_window_d_finite and self._gan_window_g_finite
                        )
                    )
                    accum_count = 0
                    completed_batch_cursor = bi + 1
                    if stepped:
                        _commit_train_step_window()
                        if step_callback is not None:
                            step_callback(epoch, self.n_steps, completed_batch_cursor)
                        self._log_train_loss_after_step()
                        if self.rank == 0:
                            pbar.set_postfix(
                                loss_g=f"{loss_g_total.item() * accum:.4f}",
                                loss_d=f"{loss_d_total.item() * accum:.4f}",
                                step=self.n_steps,
                            )
                    else:
                        _discard_train_step_window()
                    # paired optimizer boundary 后同步 max_steps；即使本窗口跳步也要
                    # 让全部 rank 执行同一 collective 序列并验证 n_steps 一致。
                    max_steps_reached = self._sync_max_steps_reached()
            if stopped_by_shutdown:
                break
            save_batch_interval = int(getattr(self, "save_batch_interval", 0))
            if (
                save_batch_interval > 0
                and (bi + 1) % save_batch_interval == 0
            ):
                self._save_middle_checkpoint(epoch, bi)
            if max_steps_reached:
                stopped_by_max_steps = True
                if self.rank == 0:
                    logger.info(
                        f"  Reached max_steps={self.max_steps}, stopping epoch."
                    )
                break

        # Flush any remaining accumulated gradients at epoch end
        if not stopped_by_max_steps and not stopped_by_shutdown and not shutdown_synced_at_last_batch:
            stopped_by_shutdown = bool(self._sync_shutdown_signum())
        if accum_count > 0:
            if stopped_by_max_steps or stopped_by_shutdown:
                if self.rank == 0:
                    logger.info(
                        "  max_steps/shutdown reached: dropping %d residual accumulated batches",
                        accum_count,
                    )
                for opt in self.optimizers.values():
                    opt.zero_grad(set_to_none=True)
                _discard_train_step_window()
            else:
                def _skip_tail_window() -> bool:
                    _discard_train_step_window()
                    return self._gan_skip_optimizer_steps()

                def _step_tail_window() -> bool:
                    stepped = self._gan_optimizer_step(
                        step_allowed=(
                            self._gan_window_d_finite and self._gan_window_g_finite
                        )
                    )
                    if stepped:
                        _commit_train_step_window()
                    else:
                        _discard_train_step_window()
                    return stepped

                self._handle_epoch_tail_gradients(
                    accum_count,
                    accum,
                    epoch,
                    # 让 GAN helper 统一处理 D/G paired eligibility；非有限 loss
                    # 仍会在 step_fn 中成对清理，而不是由 tail 单边放行。
                    step_allowed=True,
                    step_callback=step_callback,
                    batch_cursor=bi + 1,
                    label="gan",
                    models=list(self.models.values()),
                    zero_grad_fn=_skip_tail_window,
                    step_fn=_step_tail_window,
                )
            accum_count = 0

        loader_batches = getattr(pbar, "total", None)
        self._last_completed_batch_cursor = int(completed_batch_cursor)
        self._train_epoch_completed = not (stopped_by_max_steps or stopped_by_shutdown) or (
            loader_batches is not None
            and completed_batch_cursor >= int(loader_batches)
        )

        avg_loss_g = running_loss_g.item() / max(committed_batches, 1)
        avg_loss_d = running_loss_d.item() / max(committed_batches, 1)
        if self.world_size > 1:
            # Allreduce sum and count separately for correct weighted average
            # when ranks have different batch counts
            loss_g_sum = self._allreduce_sum(
                running_loss_g.item(), phase="allreduce_train_loss_g_sum"
            )
            loss_d_sum = self._allreduce_sum(
                running_loss_d.item(), phase="allreduce_train_loss_d_sum"
            )
            n_global = self._allreduce_sum(
                float(committed_batches), phase="allreduce_train_loss_count"
            )
            avg_loss_g = (loss_g_sum / n_global) if n_global > 0 else float("nan")
            avg_loss_d = (loss_d_sum / n_global) if n_global > 0 else float("nan")
        if self.rank == 0:
            logger.info(
                f"Epoch {epoch} avg loss: G={avg_loss_g:.4f}, D={avg_loss_d:.4f}"
            )
        # A23：输出 sanitize/clamp 触发时按 epoch 汇总上报（每 epoch 最多一条）
        self._log_output_sanitize_summary(epoch, "[Train]")
        return avg_loss_g

    # ------------------------------------------------------------------
    # DDP helpers
    # ------------------------------------------------------------------
    def _compute_validation_loss(self, forward_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute validation reconstruction loss through the polymorphic hook."""
        return self._compute_reconstruction_loss(
            forward_dict["enhanced"],
            forward_dict["clean"],
            forward_dict["lengths"],
            forward_dict=forward_dict,
        )

    def _maybe_save_val_audio(self, batch: Any, epoch: int) -> None:
        """验证音频保存钩子；基础 GAN trainer 不写盘，PASE 家族子类覆盖。"""
        return None

    # ------------------------------------------------------------------
    # Validation (only generator)
    # ------------------------------------------------------------------
    def _validation_epoch(self, epoch: int) -> Optional[float]:
        """Generator-only validation through the shared batch-loop adapter."""
        from core.trainer.validation_orchestration import (
            ValidationBatchResult,
            run_validation_batch_loop,
        )

        trainer = self
        self.generator.eval()
        self._reset_output_sanitize_stats()
        do_metrics = self._advance_and_should_run_metrics()

        class _GANValidationAdapter:
            def prepare_batch(self, raw_batch, current_epoch, loop_count):
                batch_dict = trainer._unpack_batch(raw_batch)
                validate_waveform_lengths(
                    batch_dict["lengths"],
                    batch_size=batch_dict["clean"].shape[0],
                    max_length=batch_dict["clean"].shape[-1],
                    context=f"GAN validation E{current_epoch} B{loop_count}",
                )
                for key in list(batch_dict.keys()):
                    value = batch_dict[key]
                    if torch.is_tensor(value):
                        batch_dict[key] = value.to(
                            trainer.device, non_blocking=True
                        )
                return trainer._sanitize_batch_inputs(
                    batch_dict, current_epoch, loop_count
                )

            def forward_batch(self, batch_dict, current_epoch, loop_count):
                def _forward_generator():
                    with torch.amp.autocast(
                        trainer.device.type, enabled=trainer.use_amp, dtype=trainer.amp_dtype
                    ):
                        raw_forward = trainer._forward_generator(batch_dict)
                        # 验证音频需要 sanitize 前（未 clamp）的原始波形；仅在启用
                        # 且 rank 0 时暂存，不改变任何 loss/collective。
                        raw_enhanced = (
                            raw_forward["enhanced"]
                            if trainer.rank == 0 and trainer._val_audio is not None
                            else None
                        )
                        forward_dict = trainer._sanitize_forward_output(raw_forward)
                        return forward_dict, raw_enhanced

                forward_dict, raw_enhanced = trainer._run_synced_validation_subphase(
                    _forward_generator,
                    phase="validation_model_output_error",
                    context="GAN validation model/output preflight failed",
                )
                with torch.amp.autocast(
                    trainer.device.type, enabled=trainer.use_amp, dtype=trainer.amp_dtype
                ):
                    loss = trainer._compute_validation_loss(
                        forward_dict
                    )
                clean = forward_dict["clean"]
                context = None
                if trainer.rank == 0 and trainer._val_audio is not None:
                    context = {
                        "forward_dict": forward_dict,
                        "raw_enhanced": raw_enhanced,
                        "step": int(loop_count),
                    }
                return ValidationBatchResult(
                    loss=loss,
                    enhanced=forward_dict["enhanced"],
                    clean=clean,
                    lengths=forward_dict["lengths"],
                    fs=forward_dict.get("fs"),
                    context=context,
                )

            def loss_weight(self, batch):
                return int(batch.clean.shape[0])

            def after_loss(self, batch, **kwargs):
                if kwargs.get("is_ddp_padding", False):
                    return None
                return trainer._maybe_save_val_audio(batch, epoch)

            def collect_metric_pairs(self, batch, pairs, total_pairs):
                fs_val = trainer._batch_fs_hz(
                    batch.fs, batch.clean.shape[0]
                )
                return trainer._append_val_metric_pairs(
                    batch.enhanced,
                    batch.clean,
                    batch.lengths,
                    fs_val,
                    pairs,
                    total_pairs,
                )

            def on_max_batches(self, current_epoch, max_batches):
                return None

        loop_result = run_validation_batch_loop(
            self,
            epoch,
            _GANValidationAdapter(),
            collect_metrics=do_metrics,
        )

        # 保持原 GAN float64 _allreduce_sum collective 序列与 phase 名不变；
        # 中止时以零贡献参与，partial loss 不得进入 best/history。
        aborted = bool(loop_result.aborted)
        if aborted:
            loop_result.loss_sum.zero_()
            local_sample_count = 0
        else:
            local_sample_count = loop_result.sample_count
        loss_sum = loop_result.loss_sum.item()
        if self.world_size > 1:
            loss_sum = self._allreduce_sum(
                loss_sum, phase="val_loss_allreduce"
            )
            n_global = self._allreduce_sum(
                float(local_sample_count), phase="val_count_allreduce"
            )
        else:
            n_global = float(local_sample_count)
        avg_loss, val_x = self._val_avg_and_curve_x(
            loss_sum, n_global, epoch
        )

        metrics_aborted = self._run_validation_metrics_from_pairs(
            epoch,
            loop_result.metric_pairs,
            do_metrics,
            val_x=val_x,
            validation_aborted=aborted,
        )
        aborted = aborted or bool(metrics_aborted)
        if aborted:
            self._vm_epoch_count -= 1
        else:
            if self.rank == 0:
                if n_global > 0:
                    self._log_scalar("val/loss", avg_loss, val_x)
                logger.info(
                    f"  [Val @ epoch {epoch}] avg_loss={avg_loss:.4f} | "
                    f"forward {loop_result.forward_seconds:.1f}s"
                )
            self._log_output_sanitize_summary(epoch, "[Val]")
        return None if aborted else avg_loss
