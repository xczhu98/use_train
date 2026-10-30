"""core trainer 的 DDP 异常边界、窗口统计及 CPU/Gloo 回归测试。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

from core.trainer.base_trainer import BaseTrainer
from core.trainer.metrics_trainer import MetricsCapableTrainer
from tasks.se.data.batch import SEBatch
from tasks.se.trainer.se_trainer import SETrainer


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


def _build_se_trainer(tmp_path: Path, *, accumulation_steps: int = 1) -> SETrainer:
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
                "accumulation_steps": accumulation_steps,
                "validation_epoch_interval": 0,
                "log_interval": 100,
                "save_batch_interval": 0,
                "save_curve_interval": 0,
            },
            "meta": {"use_amp": False},
            "data": {"placeholder_num_samples": 4},
        },
        train_loader=DataLoader([batch, batch, batch, batch], batch_size=None),
        val_loader=DataLoader([batch], batch_size=None),
        sample_rate=16000,
    )


@pytest.mark.unit
def test_train_batch_exception_sync_collects_local_and_remote_errors(monkeypatch):
    trainer = object.__new__(MetricsCapableTrainer)
    trainer.rank = 0
    trainer.world_size = 2
    trainer._nccl_flow_log = lambda _phase: None

    def gather(messages, local_text):
        messages[:] = [local_text, "rank 1: ValueError: remote metadata invalid"]

    monkeypatch.setattr(dist, "all_gather_object", gather)

    context = trainer._sync_train_batch_exception(
        ValueError("local metadata invalid"), phase="test_prepare"
    )

    assert context == (
        "rank 0: ValueError: local metadata invalid; "
        "rank 1: ValueError: remote metadata invalid"
    )


@pytest.mark.unit
def test_train_epoch_syncs_prepare_and_forward_errors_before_backward(tmp_path, monkeypatch):
    trainer = _build_se_trainer(tmp_path)
    phases: list[str] = []

    def sync_prepare(_error, *, phase):
        phases.append(phase)
        return "rank 1: ValueError: invalid remote batch" if phase.endswith("prepare_batch_error") else None

    monkeypatch.setattr(trainer, "_sync_train_batch_exception", sync_prepare)
    with pytest.raises(RuntimeError, match="batch preparation failed.*invalid remote batch"):
        trainer._train_epoch(0)
    assert phases == ["train_prepare_batch_error"]

    phases.clear()

    def sync_forward(_error, *, phase):
        phases.append(phase)
        return "rank 1: RuntimeError: remote loss failed" if phase.endswith("forward_loss_error") else None

    monkeypatch.setattr(trainer, "_sync_train_batch_exception", sync_forward)
    with pytest.raises(RuntimeError, match="forward/loss failed.*remote loss failed"):
        trainer._train_epoch(0)
    assert phases == ["train_prepare_batch_error", "train_forward_loss_error"]


@pytest.mark.unit
def test_unstepped_accumulation_window_does_not_contribute_epoch_loss(tmp_path, monkeypatch):
    trainer = _build_se_trainer(tmp_path, accumulation_steps=2)
    trainer._optimizer_step = lambda *, skip_step=False: False

    epoch_loss = trainer._train_epoch(0)

    assert epoch_loss is None
    assert list(trainer._train_log_loss_win) == []


@pytest.mark.unit
def test_cpu_ddp_omits_cuda_device_arguments(tmp_path, monkeypatch):
    import core.trainer.base_trainer as base_trainer_module

    captured_kwargs: list[dict] = []

    class _CpuDDP(nn.Module):
        def __init__(self, module: nn.Module, **kwargs) -> None:
            super().__init__()
            self.module = module
            captured_kwargs.append(kwargs)

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(base_trainer_module, "DDP", _CpuDDP)
    monkeypatch.setattr(
        BaseTrainer,
        "_select_or_allocate_checkpoint_run_dir",
        classmethod(lambda cls, _base, **_kwargs: (tmp_path / "checkpoints" / "run", False)),
    )
    model = _TinySEModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    BaseTrainer(
        rank=0,
        world_size=2,
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0),
        loss_fn=_MSELoss(),
        config={"trainer": {"exp_path": str(tmp_path)}, "meta": {}, "data": {}},
    )

    assert len(captured_kwargs) == 1
    assert "device_ids" not in captured_kwargs[0]
    assert "output_device" not in captured_kwargs[0]


@pytest.mark.unit
def test_bootstrap_cpu_gloo_does_not_select_cuda_device(tmp_path, monkeypatch):
    import core.trainer.entry_components as entry_components
    import core.utils.config as config_module

    class _Entry:
        logger = logging.getLogger("test_bootstrap_cpu_gloo")

        def parse_args(self):
            return argparse.Namespace(
                config=str(tmp_path / "config.yaml"),
                log_level="INFO",
                log_file=None,
                strict=False,
                max_steps=None,
                exp_path=None,
            )

        def _validate_config(self, _cfg):
            return None

        def _apply_strict_mode(self, _cfg, _strict, _rank):
            return None

        def seed_everything(self, _seed, _rank, deterministic=False):
            return None

    (tmp_path / "config.yaml").touch()
    calls: list[tuple[str, object]] = []
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(
        config_module,
        "load_merged_config",
        lambda _path: {"distributed": {"backend": "gloo"}, "meta": {}},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        dist,
        "init_process_group",
        lambda *, backend, timeout: calls.append((backend, timeout)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda _rank: pytest.fail("CPU/Gloo bootstrap must not call cuda.set_device"),
    )

    _args, _cfg, _rank, _local_rank, world_size = entry_components.bootstrap_entry(_Entry())

    assert world_size == 2
    assert calls and calls[0][0] == "gloo"


@pytest.mark.unit
def test_cpu_bfloat16_autocast_covers_train_and_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    trainer = _build_se_trainer(tmp_path)
    trainer.use_amp = True
    trainer.amp_dtype = torch.bfloat16
    output_dtypes = []
    handle = trainer.model.register_forward_hook(
        lambda _model, _args, output: output_dtypes.append(output.dtype)
    )
    try:
        before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
        train_loss = trainer._train_epoch(0)
        with torch.no_grad():
            val_loss, *_ = trainer._val_forward_loss(next(iter(trainer.val_loader)), 0, 1)
        assert train_loss is not None
        assert torch.isfinite(val_loss)
        assert output_dtypes == [torch.bfloat16] * 5
        assert trainer.n_steps == 4
        assert any(
            not torch.equal(parameter, original)
            for parameter, original in zip(trainer.model.parameters(), before)
        )
    finally:
        handle.remove()
