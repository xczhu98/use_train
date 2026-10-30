"""MetricsCapableTrainer 的致命 CUDA 异常不得进入 DDP abort collective 回归测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.trainer.metrics_trainer import (
    MetricsCapableTrainer,
    _is_unrecoverable_cuda_exception,
)
from tasks.se.data.batch import SEBatch
from tasks.se.trainer.se_trainer import SETrainer


pytestmark = pytest.mark.unit


class _TinySEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, noisy: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.linear(noisy)


class _MSELoss(nn.Module):
    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor,
        **_kwargs,
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(enhanced, clean)


def _build_trainer(tmp_path: Path) -> SETrainer:
    model = _TinySEModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = SEBatch(
        clean=torch.ones(2, 4),
        noisy=torch.zeros(2, 4),
        sr=torch.full((2,), 16000, dtype=torch.long),
        lengths=torch.full((2,), 4, dtype=torch.long),
    )
    return SETrainer(
        rank=0,
        world_size=1,
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0),
        loss_fn=_MSELoss(),
        config={
            "model": {"type": "TinySEModel", "args": {}},
            "trainer": {
                "epochs": 1,
                "exp_path": str(tmp_path),
                "validation_epoch_interval": 0,
                "save_curve_interval": 0,
            },
            "meta": {"use_amp": False},
            "data": {"placeholder_num_samples": 4},
        },
        train_loader=DataLoader([batch], batch_size=None),
        val_loader=DataLoader([batch], batch_size=None),
        sample_rate=16000,
    )


@pytest.mark.parametrize(
    "exc",
    [
        torch.cuda.OutOfMemoryError("CUDA out of memory"),
        RuntimeError("CUDA error: device-side assert triggered"),
        RuntimeError("CUDA error: device-side error"),
    ],
)
def test_unrecoverable_cuda_errors_are_not_ddp_abort_candidates(exc):
    assert _is_unrecoverable_cuda_exception(exc)


def test_oom_prepare_skips_abort_sync_and_reaches_base_oom_handler(tmp_path, monkeypatch):
    trainer = _build_trainer(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        trainer,
        "_prepare_batch",
        lambda *_args: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("OOM")),
    )
    monkeypatch.setattr(
        trainer,
        "_sync_train_batch_exception",
        lambda *_args, **_kwargs: pytest.fail("OOM must not enter abort sync"),
    )
    monkeypatch.setattr(
        trainer,
        "_clear_cuda_cache",
        lambda reason: calls.append(reason),
    )

    with pytest.raises(torch.cuda.OutOfMemoryError):
        trainer.train()

    assert calls == ["OOM recovery (train epoch 1)"]


def test_device_side_error_in_forward_skips_abort_sync(tmp_path, monkeypatch):
    trainer = _build_trainer(tmp_path)
    phases: list[str] = []

    def sync_error(_error, *, phase):
        phases.append(phase)
        return None

    monkeypatch.setattr(trainer, "_sync_train_batch_exception", sync_error)
    monkeypatch.setattr(
        trainer,
        "_forward_and_compute_loss",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError("CUDA error: device-side assert triggered")
        ),
    )

    with pytest.raises(RuntimeError, match="device-side assert"):
        trainer._train_epoch(1)

    assert phases == ["train_prepare_batch_error"]
