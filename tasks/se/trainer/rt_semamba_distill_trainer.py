"""RT-SEMamba progressive KD 的双输入谱域 trainer。

继承 ``DualInputLossTrainer`` 的 optimizer/DDP/AMP/accumulation/epoch-tail 语义，
只覆盖 task-specific forward 与 validation：先用框架 ``stft_module`` 把 noisy/clean
waveform 转为复谱，再将当前 optimizer step 显式传给 loss-returning KD wrapper。
"""

from __future__ import annotations

from typing import Any

import torch

from tasks.se.trainer.dual_input_loss_trainer import DualInputLossTrainer


class RTSEMambaDistillTrainer(DualInputLossTrainer):
    """驱动 ``RT_SEMambaDistill`` 的谱域 progressive KD trainer。"""

    def _distill_forward(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor,
        fs_val: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.stft_module is None:
            raise RuntimeError("RTSEMambaDistillTrainer requires a top-level stft block")
        noisy_spec = self.stft_module(noisy)
        clean_spec = self.stft_module(clean)
        kwargs: dict[str, Any] = {"step": int(self.n_steps)}
        if self._model_accepts_fs:
            kwargs["fs"] = fs_val
        output = self.model(noisy_spec, clean_spec, lengths, **kwargs)
        if isinstance(output, tuple):
            loss = output[0]
            diagnostics = output[1] if len(output) > 1 and isinstance(output[1], dict) else {}
        else:
            loss = output
            diagnostics = {}
        if loss.ndim != 0:
            raise RuntimeError(
                f"RT_SEMambaDistill must return scalar loss, got shape {tuple(loss.shape)}"
            )
        return loss, diagnostics

    def _forward_and_compute_loss(
        self,
        batch: Any,
        epoch: int,
        batch_idx: int,
        fs_val: int,
    ) -> tuple[torch.Tensor, bool, dict[str, Any]]:
        loss, diagnostics = self._distill_forward(
            batch.noisy, batch.clean, batch.lengths, fs_val
        )
        loss_was_finite = bool(torch.isfinite(loss).item())
        info: dict[str, Any] = {
            "loss_was_finite": loss_was_finite,
            "components": diagnostics,
        }
        return loss, loss_was_finite, info

    def _validation_forward(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor,
        fs_val: int,
    ) -> torch.Tensor:
        """验证复用训练侧谱域 KD 前向；验证循环本身由父类共享实现。"""
        loss, _ = self._distill_forward(noisy, clean, lengths, fs_val)
        return loss
