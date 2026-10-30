"""FN-SSL trainer vertical slice."""

from __future__ import annotations

from typing import Any

import torch

from core.trainer.metrics_trainer import MetricsCapableTrainer
from tasks.ssl.data import SSLBatch
from tasks.ssl.model_io import validate_ssl_model_output
from tasks.ssl.runtime import SSLRuntimeInputs


class SSLTrainer(MetricsCapableTrainer):
    """把 manifest 产生的 IPD/DOA/VAD batch 接入共享训练生命周期。"""

    _train_epoch_tail_label = "ssl"

    def __init__(self, *args: Any, **kwargs: Any):
        config = kwargs.get("config") or {}
        val_metrics = (config.get("trainer") or {}).get("val_metrics") or {}
        if bool(val_metrics.get("enabled", False)):
            raise ValueError(
                "SSL does not support waveform trainer.val_metrics; "
                "set trainer.val_metrics.enabled=false"
            )
        loss_fn = kwargs.get("loss_fn")
        inner_loss = getattr(loss_fn, "loss", loss_fn)
        if not bool(getattr(inner_loss, "SSL_SPATIAL_LOSS", False)):
            raise ValueError(
                "SSLTrainer requires a loss declaring SSL_SPATIAL_LOSS capability "
                "for IPD targets and named doa/vad supervision"
            )
        super().__init__(*args, **kwargs)

    def _unpack_batch(self, raw_batch: Any) -> SSLBatch:
        if not isinstance(raw_batch, SSLBatch):
            raise TypeError("SSLTrainer requires SSLBatch")
        return raw_batch

    def _validate_batch_metadata(self, batch: SSLBatch, epoch: int, batch_idx: int) -> None:
        del epoch, batch_idx
        batch.validate()

    def _batch_size_from_batch(self, batch: SSLBatch) -> int:
        if not isinstance(batch, SSLBatch):
            raise TypeError("SSLTrainer requires SSLBatch")
        return int(batch.features.shape[0])

    def _get_batch_fs(self, batch: SSLBatch) -> int:
        if batch.sample_rates is None:
            raise ValueError("SSL batch must carry sample_rates")
        if not torch.equal(batch.sample_rates, batch.sample_rates[:1].expand_as(batch.sample_rates)):
            raise ValueError("SSL batch must use one sample rate")
        sample_rate = int(batch.sample_rates[0].item())
        if type(self.sample_rate) is not int or sample_rate != self.sample_rate:
            raise ValueError(
                "SSL batch sample rate must match configured data.sample_rate: "
                f"batch={sample_rate}, configured={self.sample_rate}"
            )
        return sample_rate

    def _forward_prediction(self, runtime: SSLRuntimeInputs):
        raw_output = self.model(**runtime.model_kwargs(self.model))
        fields = validate_ssl_model_output(
            self.model,
            raw_output,
            expected_shape=runtime.target_dpipd.shape,
            expected_device=runtime.features.device,
        )
        inner_loss = getattr(self.loss_fn, "loss", self.loss_fn)
        validate_loss_contract = getattr(inner_loss, "validate_contract", None)
        if callable(validate_loss_contract):
            validate_loss_contract(fields, **runtime.loss_kwargs())
        return fields

    def _compute_ssl_loss(self, prediction, runtime: SSLRuntimeInputs) -> torch.Tensor:
        """调用 loss 时原样传递 canonical geometry/timebase 字段。"""
        inner_loss = getattr(self.loss_fn, "loss", self.loss_fn)
        if not callable(inner_loss):
            raise TypeError("SSL loss wrapper must expose a callable loss")
        kwargs = runtime.loss_kwargs()
        if getattr(inner_loss, "SSL_COMPOSITE_LOSS", False):
            return inner_loss(prediction, **kwargs)
        if isinstance(prediction, dict) and getattr(
            inner_loss,
            "DOA_OBJECTIVE_IMPLEMENTED",
            getattr(self.loss_fn, "DOA_OBJECTIVE_IMPLEMENTED", False),
        ):
            kwargs["doa_prediction"] = prediction.get("doa_vector")
        return inner_loss(prediction, **kwargs)

    def _forward_and_compute_loss(
        self, batch: SSLBatch, epoch: int, batch_idx: int, fs_val: int
    ) -> tuple[torch.Tensor, bool, dict[str, Any]]:
        del epoch, batch_idx, fs_val
        runtime = SSLRuntimeInputs._from_validated_batch(batch)
        prediction = self._run_synced_validation_subphase(
            lambda: self._forward_prediction(runtime),
            phase="train_model_output_error",
            context="SSL training model/output preflight failed",
        )
        loss = self._compute_ssl_loss(prediction, runtime)
        return loss, bool(torch.isfinite(loss).all().item()), {"enhanced_safe": prediction}

    def _val_prepare_batch(self, raw_batch, epoch, n_loops):
        batch, fs_val, _ = self._prepare_batch(raw_batch, epoch, n_loops)
        return batch, fs_val, SSLRuntimeInputs._from_validated_batch(batch)

    def _val_forward_prepared(self, prepared_batch, epoch, n_loops):
        del epoch, n_loops
        batch, fs_val, runtime = prepared_batch
        prediction = self._run_synced_validation_subphase(
            lambda: self._forward_prediction(runtime),
            phase="validation_model_output_error",
            context="SSL validation model/output preflight failed",
        )
        loss = self._compute_ssl_loss(prediction, runtime)
        return loss, prediction, runtime.target_dpipd, runtime.output_lengths, fs_val, batch

    def _val_forward_loss(
        self, raw_batch: SSLBatch, epoch: int, n_loops: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Any]:
        prepared = self._val_prepare_batch(raw_batch, epoch, n_loops)
        return self._val_forward_prepared(prepared, epoch, n_loops)
