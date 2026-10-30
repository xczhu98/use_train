"""Speech Separation trainer hooks on the shared training skeleton."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from core.trainer.metrics_trainer import MetricsCapableTrainer
from core.metrics.separation import build_separation_metric_pairs
from core.data.source_separation_batch import SSBatch
from tasks.ss.model_io import SSModelRuntime, resolve_ss_execution_plan


class SSTrainer(MetricsCapableTrainer):
    """Train fixed-source-count separators with a task-local loss contract."""

    def __init__(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        sample_rate: int = 16000,
        stft_module: Any = None,
        **kwargs: Any,
    ):
        super().__init__(
            train_loader=train_loader,
            val_loader=val_loader,
            sample_rate=sample_rate,
            stft_module=stft_module,
            **kwargs,
        )
        plan = resolve_ss_execution_plan(
            self.config,
            self.model,
            dataset_contract=getattr(self.train_dataset, "contract", None),
            loss=self.loss_fn,
        )
        if type(self.sample_rate) is not int or self.sample_rate != plan.sample_rate:
            raise ValueError(
                "SSTrainer sample_rate does not match execution plan: "
                f"{self.sample_rate!r} != {plan.sample_rate}"
            )
        val_contract = getattr(getattr(self.val_loader, "dataset", None), "contract", None)
        if val_contract is not None:
            resolve_ss_execution_plan(
                self.config,
                self.model,
                dataset_contract=val_contract,
                loss=self.loss_fn,
            )
        self.model_runtime = SSModelRuntime(
            self.model,
            plan=plan,
            stft_module=self.stft_module,
        )

    def _unpack_batch(self, raw_batch: SSBatch) -> SSBatch:
        if not isinstance(raw_batch, SSBatch):
            raise TypeError("SSTrainer requires an SSBatch from ss_collate_fn")
        return raw_batch

    def _validate_batch_metadata(self, batch: SSBatch, epoch: int, batch_idx: int) -> None:
        del epoch, batch_idx
        batch.validate()

    def _batch_size_from_batch(self, batch: SSBatch) -> int:
        if not isinstance(batch, SSBatch):
            raise TypeError("SSTrainer batch-size hook requires SSBatch")
        return int(batch.mixture.shape[0])

    def _forward_sources(self, batch: SSBatch, fs_val: int) -> torch.Tensor:
        estimated = self.model_runtime.separate(
            batch.mixture,
            batch.lengths,
            fs_val,
        ).sources
        return estimated

    def _compute_loss(self, batch: SSBatch, estimated: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(
            estimated,
            batch.sources,
            lengths=batch.lengths,
            source_mask=batch.source_mask,
        )

    def _forward_and_compute_loss(self, batch, epoch, batch_idx, fs_val):
        del epoch, batch_idx
        estimated = self._forward_sources(batch, fs_val)
        loss = self._compute_loss(batch, estimated)
        return loss, bool(torch.isfinite(loss).all().item()), {
            "enhanced_safe": estimated,
        }

    def _val_prepare_batch(self, raw_batch, epoch, n_loops):
        batch, fs_val, _ = self._prepare_batch(raw_batch, epoch, n_loops)
        return batch, fs_val

    def _val_forward_prepared(self, prepared_batch, epoch, n_loops):
        del epoch, n_loops
        batch, fs_val = prepared_batch
        estimated = self._run_synced_validation_subphase(
            lambda: self._forward_sources(batch, fs_val),
            phase="validation_model_output_error",
            context="SS validation model/output preflight failed",
        )
        loss = self._compute_loss(batch, estimated)
        return (
            loss,
            estimated,
            batch.sources,
            batch.lengths,
            fs_val,
            {"source_mask": batch.source_mask},
        )

    def _val_forward_loss(self, raw_batch, epoch, n_loops):
        """兼容仍直接调用旧 validation hook 的任务外测试与工具。"""
        prepared = self._val_prepare_batch(raw_batch, epoch, n_loops)
        return self._val_forward_prepared(prepared, epoch, n_loops)

    def _collect_val_metric_pairs(
        self,
        enhanced,
        clean,
        lengths,
        fs_val,
        pairs,
        total_pairs_for_metrics,
        extra,
        ctx,
    ):
        del extra
        source_mask = ctx.get("source_mask") if isinstance(ctx, dict) else None
        if not isinstance(source_mask, torch.Tensor):
            raise RuntimeError("SS validation metric context requires source_mask")
        remaining = max(0, self._vm_max_metrics - total_pairs_for_metrics)
        new_pairs, _ = build_separation_metric_pairs(
            enhanced,
            clean,
            lengths,
            source_mask,
            fs_val,
            max_pairs=remaining,
        )
        pairs.extend(new_pairs)
        return total_pairs_for_metrics + len(new_pairs)
