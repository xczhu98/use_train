"""MeanFlowSE 专属 complex-STFT 训练器。

训练路径把 noisy/clean waveform 转为框架 STFT，并通过 ``self.model``（而非
解包后的 raw model）调用双谱 MeanFlow objective，确保 DDP wrapper 参与前向。
验证路径执行一步 ``euler_mf`` 生成增强谱，再 iSTFT 供现有客观指标骨架使用。
"""

from __future__ import annotations

from typing import Any

import torch

from core.trainer.metrics_trainer import validate_waveform_lengths
from tasks.se.trainer.metrics_trainer import _unpack_se_batch
from tasks.se.trainer.se_trainer import SETrainer


class MeanFlowSETrainer(SETrainer):
    """训练返回 scalar objective、验证返回增强 waveform 的 task-local Trainer。"""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if self.stft_module is None:
            raise ValueError("MeanFlowSETrainer requires a top-level stft configuration")
        train_cfg = self.config.get("data", {}).get("train", {})
        if bool(train_cfg.get("repeat_to_max_duration", False)):
            raise ValueError(
                "MeanFlowSETrainer requires repeat_to_max_duration=false; "
                "short training pairs are zero-padded by the task-local trainer"
            )
        self.recipe_num_samples = int(train_cfg.get("max_duration", 0))
        if self.recipe_num_samples <= 0:
            raise ValueError("MeanFlowSETrainer requires data.train.max_duration > 0")

    def _training_progress(self, epoch: int | None = None) -> float:
        epochs = max(int(self.epochs), 1)
        epoch = int(self.current_epoch if epoch is None else epoch)
        return min(max((float(epoch) - 1.0) / epochs, 0.0), 1.0)

    @staticmethod
    def _normalize_pair(
        noisy: torch.Tensor, clean: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """按官方 recipe 用 noisy peak 对每个配对样本做 joint normalization。"""
        peak = noisy.abs().amax(dim=1, keepdim=True)
        scale = torch.where(peak > 0.0, peak, torch.ones_like(peak))
        return noisy / scale, clean / scale, scale

    def _pad_training_pair(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把有效配对波形对称零补齐到官方固定训练画布。"""
        if noisy.shape != clean.shape or noisy.ndim != 2:
            raise ValueError(
                "MeanFlowSE training pair must have matching (B,T) shapes, got "
                f"noisy={tuple(noisy.shape)}, clean={tuple(clean.shape)}"
            )
        if lengths.shape != (noisy.shape[0],):
            raise ValueError(
                f"MeanFlowSE lengths must have shape ({noisy.shape[0]},), "
                f"got {tuple(lengths.shape)}"
            )
        canvas = self.recipe_num_samples
        padded_noisy = noisy.new_zeros((noisy.shape[0], canvas))
        padded_clean = clean.new_zeros((clean.shape[0], canvas))
        # 逐条 .item() 会为每个样本触发一次 GPU→CPU 同步；批量物化一次即可。
        length_values = torch.stack(tuple(lengths.unbind())).detach().cpu().tolist()
        for index, length_value in enumerate(length_values):
            length = int(length_value)
            if length <= 0 or length > noisy.shape[1] or length > canvas:
                raise ValueError(
                    "MeanFlowSE training length must satisfy 0 < length <= "
                    f"min(batch_samples, recipe_num_samples), got {length}"
                )
            left = (canvas - length) // 2
            padded_noisy[index, left : left + length] = noisy[index, :length]
            padded_clean[index, left : left + length] = clean[index, :length]
        full_lengths = lengths.new_full(lengths.shape, canvas)
        return padded_noisy, padded_clean, full_lengths

    def _forward_and_compute_loss(
        self,
        batch: Any,
        epoch: int,
        batch_idx: int,
        fs_val: int,
    ) -> tuple[torch.Tensor, bool, dict[str, Any]]:
        del batch_idx
        noisy, clean, lengths = self._pad_training_pair(
            batch.noisy, batch.clean, batch.lengths
        )
        noisy, clean, _ = self._normalize_pair(noisy, clean)
        noisy_spec = self.stft_module(noisy)
        clean_spec = self.stft_module(clean)
        output = self.model(
            noisy_spec,
            lengths,
            fs=fs_val,
            clean_spec=clean_spec,
            progress=self._training_progress(epoch),
        )
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(
                "MeanFlowSE training forward must return (scalar_loss, diagnostics)"
            )
        loss, diagnostics = output
        if loss.ndim != 0:
            raise RuntimeError(f"MeanFlowSE objective must be scalar, got shape {tuple(loss.shape)}")
        loss_was_finite = bool(torch.isfinite(loss).item())
        info: dict[str, Any] = {
            "loss_was_finite": loss_was_finite,
            "diagnostics": diagnostics,
        }
        return loss, loss_was_finite, info

    def _maybe_log_step_diagnostics(
        self,
        batch: Any,
        loss_info: dict[str, Any],
        epoch: int,
        batch_idx: int,
    ) -> None:
        del batch, epoch, batch_idx
        if self.rank != 0 or self.n_steps <= 0 or self.n_steps % self.log_interval:
            return
        diagnostics = loss_info.get("diagnostics")
        if not isinstance(diagnostics, dict):
            return
        # 前四个 key 为 MeanFlowSE 诊断；后三个为 MeanSE 诊断（两模型共用本 trainer）。
        for name in (
            "loss_cfm", "loss_mf", "mf_weight", "mf_skipped",
            "loss_meanflow", "train_flow", "mean_interval",
        ):
            value = diagnostics.get(name)
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                self._log_scalar(f"train/meanflowse/{name}", float(value.item()), self.n_steps)

    def _val_prepare_batch(
        self,
        raw_batch: Any,
        epoch: int,
        n_loops: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """在 DDP forward 前完成 MeanFlowSE 波形与谱输入准备。"""
        batch = _unpack_se_batch(raw_batch)
        validate_waveform_lengths(
            batch.lengths,
            batch_size=batch.clean.shape[0],
            max_length=batch.clean.shape[-1],
            context=f"MeanFlowSE validation E{epoch}",
        )
        clean = batch.clean.to(self.device, non_blocking=True)
        noisy = batch.noisy.to(self.device, non_blocking=True)
        lengths = batch.lengths.to(self.device, non_blocking=True)
        batch.clean = clean
        batch.noisy = noisy
        batch.lengths = lengths
        batch = self._sanitize_batch_inputs(batch, epoch, n_loops)
        clean = batch.clean
        noisy = batch.noisy
        lengths = batch.lengths
        fs_val = self._batch_fs_hz(batch.sr, clean.shape[0])

        noisy_normalized, clean_normalized, scale = self._normalize_pair(noisy, clean)
        clean_spec = self.stft_module(clean_normalized)
        noisy_spec = self.stft_module(noisy_normalized)
        return clean, noisy, lengths, fs_val, scale, clean_spec, noisy_spec

    def _val_forward_prepared(
        self,
        prepared_batch: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        epoch: int,
        n_loops: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Any]:
        """消费已同步通过 prepare 门禁的 MeanFlowSE 谱输入。"""
        del n_loops
        clean, noisy, lengths, fs_val, scale, clean_spec, noisy_spec = prepared_batch
        with torch.amp.autocast(
            self.device.type, enabled=self.use_amp, dtype=self.amp_dtype
        ):
            objective = self.model(
                noisy_spec,
                lengths,
                fs=fs_val,
                clean_spec=clean_spec,
                progress=self._training_progress(epoch),
            )
            enhanced_spec = self.model(noisy_spec, lengths, fs=fs_val)
        if not isinstance(objective, tuple) or len(objective) != 2:
            raise RuntimeError("MeanFlowSE validation forward must return (loss, diagnostics)")
        loss, _ = objective
        if loss.ndim != 0:
            raise RuntimeError(f"MeanFlowSE validation objective must be scalar, got {loss.shape}")
        if not isinstance(enhanced_spec, torch.Tensor):
            raise RuntimeError("MeanFlowSE inference forward must return an enhanced spectrum tensor")
        if enhanced_spec.shape != clean_spec.shape:
            raise RuntimeError(
                "MeanFlowSE enhanced spectrum shape mismatch: "
                f"{tuple(enhanced_spec.shape)} vs {tuple(clean_spec.shape)}"
            )
        enhanced = self.stft_module.inverse(enhanced_spec, noisy.shape[-1]) * scale
        enhanced = self._sanitize_model_output(enhanced)
        if self._strict_checks:
            assert enhanced.shape == clean.shape, (
                f"Model output shape mismatch: enhanced={tuple(enhanced.shape)} "
                f"clean={tuple(clean.shape)}"
            )
        return loss, enhanced, clean, lengths, fs_val, None

    def _val_forward_loss(
        self,
        raw_batch: Any,
        epoch: int,
        n_loops: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Any]:
        """兼容旧调用方：顺序组合 prepare 与 prepared-forward 两阶段。"""
        prepared = self._val_prepare_batch(raw_batch, epoch, n_loops)
        return self._val_forward_prepared(prepared, epoch, n_loops)
