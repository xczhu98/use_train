"""真实 Gloo、BSRNN unused 参数与异常恢复。"""
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from conftest import spawn_ddp_workers
from core.losses.composite import StrictFiniteLossError
from core.trainer.metrics_trainer import MetricsCapableTrainer
from tasks.se.models.discriminative.bsrnn import BSRNN

pytestmark = [pytest.mark.ddp, pytest.mark.subprocess]


class _SpectrumTrainer(MetricsCapableTrainer):
    def _unpack_batch(self, batch):
        return batch

    def _forward_and_compute_loss(self, batch, epoch, batch_idx, fs_val):
        output = self.model(batch["input"], fs=16000)
        if self.rank == 1 and batch_idx == 1 and epoch == 2:
            if self.strict_failure:
                raise StrictFiniteLossError("rank-local strict failure")
            return output.new_tensor(float("nan"), requires_grad=True), False, {}
        return output.square().mean(), True, {}


def _bsrnn_worker(rank, path, accumulation, strict):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{path}/init", rank=rank,
                            world_size=2, timeout=timedelta(seconds=50))
    trainer = None
    try:
        torch.manual_seed(19)
        model = BSRNN(num_channel=4, num_layer=1, target_fs=48000, n_fft=2048)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = DataLoader([{"input": torch.randn(1, 1025, 3, 2), "fs": 16000}] * 4,
                            batch_size=None)
        loss_fn = torch.nn.MSELoss()
        loss_fn.STRICT_FINITE_CHECKS = strict
        trainer = _SpectrumTrainer(
            rank=rank, world_size=2, model=model, optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1),
            loss_fn=loss_fn, train_loader=loader, val_loader=loader,
            config={"model": {"type": "BSRNN", "args": {}},
                    "meta": {"use_amp": False}, "data": {},
                    "distributed": {"find_unused_parameters": True},
                    "trainer": {"exp_path": str(Path(path) / "run"),
                                "strict_checks": strict,
                                "accumulation_steps": accumulation, "save_curve_interval": 0,
                                "validation_epoch_interval": 0, "log_interval": 100}})
        trainer.strict_failure = strict
        observed = []
        handles = [p.register_hook(lambda grad, i=i: observed.append(i))
                   for i, p in enumerate(trainer.model.parameters())]
        trainer._train_epoch(1)
        assert len(list(model.parameters())) - len(set(observed)) == 100
        for handle in handles:
            handle.remove()
        if strict:
            with pytest.raises(RuntimeError, match="strict finite loss failed"):
                trainer._train_epoch(2)
        else:
            trainer._train_epoch(2)
        trainer._train_epoch(3)
        assert trainer.n_steps == (8 if strict else 12) // accumulation
        for parameter in trainer.model.parameters():
            other = parameter.detach().clone()
            dist.broadcast(other, 0)
            torch.testing.assert_close(parameter, other)
        assert not trainer.model._forward_hooks
    finally:
        if trainer is not None and trainer.writer is not None:
            trainer.writer.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("accumulation,strict", [(1, False), (2, False), (2, True)])
def test_real_bsrnn_unused_nan_and_next_iteration(tmp_path, accumulation, strict):
    spawn_ddp_workers(_bsrnn_worker, args=(str(tmp_path), accumulation, strict), timeout_sec=80)


def _exception_worker(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{path}/init", rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        trainer = object.__new__(MetricsCapableTrainer)
        trainer.rank, trainer.world_size, trainer.device = rank, 2, torch.device("cpu")
        trainer._nccl_flow_log = lambda phase: None
        gather = dist.all_gather_object
        calls = []

        def record(*args, **kwargs):
            calls.append(1)
            return gather(*args, **kwargs)

        dist.all_gather_object = record
        for phase in ("prepare", "forward", "validation"):
            with torch.inference_mode():
                assert trainer._sync_train_batch_exception(None, phase=phase) is None
                assert calls == []
                context = trainer._sync_train_batch_exception(
                    ValueError("remote failure") if rank else None, phase=phase)
            assert context == "rank 1: ValueError: remote failure"
            assert calls == [1]
            calls.clear()
    finally:
        dist.destroy_process_group()


def test_success_exception_boundary_uses_tensor_only(tmp_path):
    spawn_ddp_workers(_exception_worker, args=(str(tmp_path),), timeout_sec=60)


def _gan_worker(rank, path, bad_side, accumulation):
    from test_shared_trainer_audit_fixes import _build_audit_gan
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{path}/init", rank=rank,
                            world_size=2, timeout=timedelta(seconds=40))
    trainer = None
    try:
        trainer = _build_audit_gan(Path(path) / "run", rank=rank, world_size=2,
                                   accumulation=accumulation)
        trainer._train_epoch(1)
        trainer.bad_side, trainer.bad_rank = bad_side, 1
        before = [p.detach().clone() for model in trainer.models.values() for p in model.parameters()]
        trainer._train_epoch(2)
        for old, current in zip(before, [p for m in trainer.models.values() for p in m.parameters()]):
            torch.testing.assert_close(old, current)
        trainer.bad_side = None
        trainer._train_epoch(3)
        assert trainer.n_steps == 4 // accumulation
        for model in trainer.models.values():
            assert not model._forward_hooks
            for parameter in model.parameters():
                other = parameter.detach().clone()
                dist.broadcast(other, 0)
                torch.testing.assert_close(parameter, other)
    finally:
        if trainer is not None and trainer.writer is not None:
            trainer.writer.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("bad_side", ["generator", "discriminator"])
@pytest.mark.parametrize("accumulation", [1, 2])
def test_gan_rank_local_nan_recovers_with_frozen_peer(tmp_path, bad_side, accumulation):
    spawn_ddp_workers(_gan_worker, args=(str(tmp_path), bad_side, accumulation), timeout_sec=70)
