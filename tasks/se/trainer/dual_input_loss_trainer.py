"""DualInputLossTrainer: trainer for models that compute their own scalar loss.

Used for PASE Phase-2 DeWavLM distillation and StuPASE's flow-matching (CFM)
training. Both require the *trainable* model to see ``noisy`` and ``clean``
simultaneously (``student_feat=model(noisy)`` vs. ``teacher_feat=teacher(clean)``
for distillation; CFM's own ``forward()`` already returns a scalar
conditional-flow-matching loss). Standard ``SETrainer``/``MetricsCapableTrainer``
never passes ``clean`` into ``model.forward()``, and its ``_validation_epoch``
unconditionally ``clamp(-1, 1)``s the model output and asserts
``enhanced.shape == clean.shape`` — both wrong when the model's output is a
scalar training loss rather than an enhanced waveform.

Model contract: ``forward(noisy_wav, clean_wav, lengths=None) -> scalar_loss``
(or a tuple whose first element is the scalar loss).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

import torch

from core.trainer.metrics_trainer import validate_waveform_lengths
from tasks.se.trainer.metrics_trainer import _unpack_se_batch
from tasks.se.trainer.se_trainer import SETrainer
from tasks.se.trainer.val_audio import ValAudioWriter

logger = logging.getLogger(__name__)

_DUAL_VAL_AUDIO_STAGES = frozenset({"distill", "flow"})


class DualInputLossTrainer(SETrainer):
    """Trainer for dual-input, scalar-loss-returning models (distillation / flow matching)."""

    SUPPORTS_WAVEFORM_VAL_METRICS = False
    VALIDATION_OUTPUT_DOMAIN = "scalar_loss"

    # validation 验证音频写入器；未启用时为 None（直接构造/测试路径也安全）。
    _val_audio: ValAudioWriter | None = None

    def __init__(self, **kw: Any):
        super().__init__(**kw)
        self._val_audio = ValAudioWriter.from_config(
            self.config,
            rank=self.rank,
            device=self.device,
            sample_rate=self.sample_rate,
        )
        if self._val_audio is not None:
            raw_model = self._raw_model()
            stage = getattr(raw_model, "val_audio_stage", None)
            if stage not in _DUAL_VAL_AUDIO_STAGES:
                raise ValueError(
                    f"{type(raw_model).__name__} does not support trainer.val_audio "
                    f"(stage={stage!r}); supported stages: "
                    f"{sorted(_DUAL_VAL_AUDIO_STAGES)}"
                )
            self._val_audio.validate_roles(
                getattr(raw_model, "val_audio_vocoder_roles", ())
            )

    def _run_synced_validation_phase(
        self,
        operation: Callable[[], Any],
        *,
        phase: str,
        context: str,
    ) -> Any:
        """普通异常跨 rank 同步；不可恢复 CUDA 故障立即交给 launcher。"""
        return self._run_synced_validation_subphase(
            operation, phase=phase, context=context
        )

    def _prepare_scalar_validation_batch(
        self,
        raw_batch: Any,
        epoch: int,
        loop_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """校验并搬运 scalar-loss validation batch，不包含模型前向。"""
        se_batch = _unpack_se_batch(raw_batch)
        validate_waveform_lengths(
            se_batch.lengths,
            batch_size=se_batch.clean.shape[0],
            max_length=se_batch.clean.shape[-1],
            context=f"Dual-input validation E{epoch} B{loop_count}",
        )
        clean = se_batch.clean.to(self.device, non_blocking=True)
        noisy = se_batch.noisy.to(self.device, non_blocking=True)
        lengths = se_batch.lengths.to(self.device, non_blocking=True)
        se_batch.clean = clean
        se_batch.noisy = noisy
        se_batch.lengths = lengths
        se_batch = self._sanitize_batch_inputs(se_batch, epoch, loop_count)
        clean = se_batch.clean
        noisy = se_batch.noisy
        lengths = se_batch.lengths
        fs_val = self._batch_fs_hz(se_batch.sr, clean.shape[0])
        return clean, noisy, lengths, fs_val

    @staticmethod
    def _require_scalar_loss(loss: Any) -> torch.Tensor:
        """校验模型契约：loss 必须是零维 Tensor，拒绝隐式 batch reduction。"""
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            shape = tuple(loss.shape) if isinstance(loss, torch.Tensor) else None
            raise RuntimeError(
                "DualInputLossTrainer model must return a scalar loss tensor, "
                f"got {type(loss).__name__} with shape {shape}"
            )
        return loss

    def _validation_forward(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor,
        fs_val: int,
    ) -> torch.Tensor:
        """验证前向 hook：返回 scalar loss。

        子类（如 RT-SEMamba 的谱域 KD）覆盖本 hook 即可复用共享验证循环，
        无需整段复制 ``_validation_epoch``；调用时处于 autocast 与
        ``torch.inference_mode()`` 内。
        """
        kwargs = {"fs": fs_val} if self._model_accepts_fs else {}
        loss = self.model(noisy, clean, lengths, **kwargs)
        if isinstance(loss, tuple):
            loss = loss[0]
        return self._require_scalar_loss(loss)

    def _forward_and_compute_loss(
        self,
        batch: Any,
        epoch: int,
        batch_idx: int,
        fs_val: int,
    ) -> tuple[torch.Tensor, bool, dict[str, Any]]:
        # Only pass ``fs`` when the model's forward accepts it (audit R-02).
        # ``self._model_accepts_fs`` is a bool set by ``MetricsCapableTrainer``
        # via ``"fs" in inspect.signature(model.forward).parameters``.
        kwargs = {"fs": fs_val} if self._model_accepts_fs else {}
        loss = self.model(batch.noisy, batch.clean, batch.lengths, **kwargs)
        if isinstance(loss, tuple):
            loss = loss[0]
        loss = self._require_scalar_loss(loss)
        loss_was_finite = bool(torch.isfinite(loss).item())
        return loss, loss_was_finite, {"loss_was_finite": loss_was_finite}

    def _validation_epoch(self, epoch: int) -> Optional[float]:
        """Validation for scalar-loss models: no clamp/shape-assert/audio-metric collection."""
        total_loss_gpu = torch.zeros(1, device=self.device)
        n = 0
        n_loops = 0
        aborted = False
        t_forward_start = time.monotonic()

        # A23：validation 统一 inference_mode（与 SE/AEC/GAN 一致）
        with self._ema_context(), torch.inference_mode():
            for raw_batch in self._iter_loader_with_synced_fetch(
                self.val_loader, phase="dual_validation_loader"
            ):
                # 每个 forward 前同步停机请求，避免任一 rank 单独退出验证循环。
                if self._sync_shutdown_signum():
                    aborted = True
                    break
                n_loops += 1
                # DDP 尾部对齐复制批：forward 照常执行（维持 buffer 广播同步），
                # 其 loss 贡献不计入（验证集每样本只计一次）。
                _is_ddp_pad = self._val_iter_is_ddp_padding(n_loops)
                clean, noisy, lengths, fs_val = self._run_synced_validation_phase(
                    lambda: self._prepare_scalar_validation_batch(
                        raw_batch, epoch, n_loops
                    ),
                    phase="validation_prepare_error",
                    context=(
                        f"[Dual-input validation E{epoch} B{n_loops}] "
                        "batch preparation failed"
                    ),
                )

                def _forward_validation_loss() -> torch.Tensor:
                    with torch.amp.autocast(
                        self.device.type, enabled=self.use_amp, dtype=self.amp_dtype
                    ):
                        # 与训练路径一致：使用 batch 实际采样率。
                        return self._validation_forward(noisy, clean, lengths, fs_val)

                loss = self._run_synced_validation_phase(
                    _forward_validation_loss,
                    phase="validation_forward_error",
                    context=(
                        f"[Dual-input validation E{epoch} B{n_loops}] "
                        "forward failed"
                    ),
                )

                total_loss_gpu, n, _ = self._accumulate_val_loss(
                    loss, total_loss_gpu, n, _is_ddp_pad, epoch, n_loops,
                    weight=int(clean.shape[0]),
                )

                # 官方验证音频落盘：loss 计算与 DDP padding 判定之后、仅 rank 0。
                if not _is_ddp_pad:
                    self._maybe_save_val_audio_dual(
                        noisy, clean, lengths, fs_val, epoch, n_loops
                    )

                if self.max_val_batches > 0 and n_loops >= self.max_val_batches:
                    break
            # 最后一批 forward 或 max_val_batches 截断后收到的信号也必须传播。
            if not aborted:
                aborted = bool(self._sync_shutdown_signum())

        t_forward_sec = time.monotonic() - t_forward_start
        if aborted:
            # 中止轮次不允许 partial validation 影响日志、best 或 history；仍执行固定归约。
            total_loss_gpu = torch.zeros(1, device=self.device)
            n = 0
        loss_sum, n_global = self._allreduce_val_loss_sum_count(total_loss_gpu, n)
        if aborted:
            if self.rank == 0:
                logger.info("  [Val] epoch %s aborted by shutdown request", epoch)
            return None

        avg, val_x = self._val_avg_and_curve_x(loss_sum, n_global, epoch)
        if self.rank == 0:
            if n_global > 0:
                self._log_scalar("val/loss", avg, val_x)
            logger.info(f"  [Val] epoch {epoch} | loss {avg:.4f} | forward {t_forward_sec:.1f}s")

        return avg

    def _maybe_save_val_audio_dual(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor,
        fs_val: int,
        epoch: int,
        step: int,
    ) -> None:
        """按 ``trainer.val_audio`` 写官方验证音频；未启用时零开销 no-op。"""
        if self._val_audio is None:
            return
        self._val_audio.capture(
            self._raw_model(),
            noisy,
            clean,
            lengths,
            fs=fs_val,
            epoch=epoch,
            step=step,
            start_epoch=self.start_epoch,
        )
