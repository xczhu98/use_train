"""共享训练发布、恢复和真实 AMP 尾窗回归。"""
from __future__ import annotations

import copy

import pytest
import torch

from core.trainer.base_trainer import BaseTrainer
from core.trainer.entry import UnifiedTrainingEntry
from test_middle_checkpoint_resume import _build_trainer, _close_log_handlers

pytestmark = pytest.mark.unit


@pytest.fixture
def trainers(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    instances = []

    def build(name, **kwargs):
        trainer = _build_trainer(tmp_path / name, n_train=10, **kwargs)
        instances.append(trainer)
        trainer.validation_steps_interval = 1
        trainer.validation_epoch_interval = 0
        trainer.save_top_k = 2
        trainer.save_curve_interval = 0
        return trainer

    yield build
    for trainer in instances:
        if trainer.writer is not None:
            trainer.writer.close()
    _close_log_handlers(tmp_path)


@pytest.mark.parametrize("remove_candidate", [False, True])
def test_resumed_negative_best_is_not_republished_by_worse_score(trainers, remove_candidate):
    original = trainers("original")
    original._train_epoch(1)
    original._validation_epoch = lambda epoch: -15.0
    original._maybe_validate_and_save_step_best(1, original.n_steps, 5)
    alias = original.ckpt_dir / "best_model.tar"
    state = BaseTrainer._safe_load(alias)
    resumed = trainers("resumed", resume=True, resume_from=alias)
    resumed.ckpt_dir = original.ckpt_dir
    if remove_candidate:
        for path in original.ckpt_dir.glob("best_step=*.tar"):
            path.unlink()
    with torch.no_grad():
        next(resumed.model.parameters()).add_(1)
    resumed._validation_epoch = lambda epoch: -5.0
    resumed._maybe_validate_and_save_step_best(2, resumed.n_steps + 1, 1)
    preserved = BaseTrainer._safe_load(alias)
    assert preserved["best_score"] == resumed.best_score == -15.0
    for key, value in state["model"].items():
        torch.testing.assert_close(preserved["model"][key], value)
    if not remove_candidate:
        assert resumed._step_best_list[0][1] == -15.0
    resumed._validation_epoch = lambda epoch: -20.0
    resumed._maybe_validate_and_save_step_best(2, resumed.n_steps + 2, 2)
    assert BaseTrainer._safe_load(alias)["best_score"] == -20.0


def test_entry_preview_accepts_committed_accumulation_tail(trainers):
    original = trainers("original", accum=2)
    original._train_epoch(1)
    original._validation_epoch = lambda epoch: -1.0
    original._maybe_validate_and_save_step_best(1, original.n_steps, 5)
    path = original.ckpt_dir / "best_model.tar"
    assert UnifiedTrainingEntry._checkpoint_start_epoch(original.config, path) == 1
    resumed = trainers("resumed", accum=2, resume=True, resume_from=path)
    assert resumed._resume_skip_batches == 5
    resumed._train_epoch(1)
    assert resumed.processed == []
    original._train_epoch(2)
    resumed._train_epoch(2)
    for a, b in zip(original.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(a, b)


def test_real_cpu_scaler_tail_overflow_backs_off_and_recovers(trainers):
    trainer = trainers("amp", accum=2)
    trainer.use_amp = True
    trainer.amp_dtype = torch.float16
    trainer.scaler = torch.amp.GradScaler("cpu", init_scale=65536.0)
    before = copy.deepcopy(trainer.model.state_dict())
    trainer._log_train_loss_after_step = lambda: None
    for index in range(3):
        loss = trainer.model(torch.ones(1, 16), torch.tensor([16])).sum()
        trainer.scaler.scale(loss).backward()
        next(trainer.model.parameters()).grad.fill_(float("inf"))
        assert not trainer._handle_epoch_tail_gradients(1, 2, index, step_allowed=True)
        assert trainer.scaler.get_scale() == 65536.0 / 2 ** (index + 1)
        assert trainer.n_steps == 0
        assert not trainer.optimizer.state
    for key, value in before.items():
        torch.testing.assert_close(trainer.model.state_dict()[key], value)
    trainer.scaler.scale(trainer.model(torch.ones(1, 16), torch.tensor([16])).sum()).backward()
    assert trainer._handle_epoch_tail_gradients(1, 2, 4, step_allowed=True)
    assert trainer.n_steps == 1


from tasks.se.trainer.gan_trainer import GANTrainer
from torch.utils.data import DataLoader


class _UnusedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.unused = torch.nn.Parameter(torch.ones(3))

    def forward(self, value):
        return self.linear(value)


class _AuditGAN(GANTrainer):
    bad_side = None
    bad_rank = 0

    def _unpack_batch(self, batch):
        return batch

    def _forward_generator(self, batch):
        return {**batch, "enhanced": self.generator(batch["noisy"])}

    def _extract_gan_mag_pha(self, fwd):
        return fwd["clean"], fwd["clean"], fwd["enhanced"], fwd["enhanced"]

    def _compute_discriminator_loss(self, cm, cp, em, ep, real_scores=None):
        result = self.discriminator(cm).square().mean() + self.discriminator(em).square().mean()
        if self.bad_side == "discriminator" and self.rank == self.bad_rank:
            return result.new_tensor(float("nan"), requires_grad=True)
        return result

    def _compute_reconstruction_loss(self, enhanced, clean, lengths, **kwargs):
        return (enhanced - clean).square().mean()

    def _compute_generator_adversarial_loss(self, cm, cp, em, ep):
        result = self.discriminator(em).square().mean()
        if self.bad_side == "generator" and self.rank == self.bad_rank:
            return result.new_tensor(float("nan"), requires_grad=True)
        return result

    def _train_step_discriminator(self, *args, **kwargs):
        before = [None if p.grad is None else p.grad.clone() for p in self.generator.parameters()]
        result = super()._train_step_discriminator(*args, **kwargs)
        for old, param in zip(before, self.generator.parameters()):
            assert (old is None and param.grad is None) or torch.equal(old, param.grad)
        return result

    def _train_step_generator(self, *args, **kwargs):
        before = [None if p.grad is None else p.grad.clone() for p in self.discriminator.parameters()]
        result = super()._train_step_generator(*args, **kwargs)
        for old, param in zip(before, self.discriminator.parameters()):
            assert (old is None and param.grad is None) or torch.equal(old, param.grad)
        return result


def _build_audit_gan(path, *, rank=0, world_size=1, accumulation=1, batches=2):
    torch.manual_seed(51)
    models = {name: _UnusedLinear() for name in ("generator", "discriminator")}
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=0.001) for name, model in models.items()}
    batch = {"noisy": torch.randn(2, 4), "clean": torch.zeros(2, 4),
             "lengths": torch.tensor([4, 4]), "fs": 16000}
    loader = DataLoader([batch] * batches, batch_size=None)
    return _AuditGAN(rank=rank, world_size=world_size, model=models, optimizer=optimizers,
                     scheduler={name: torch.optim.lr_scheduler.StepLR(opt, 1, 0.5)
                                for name, opt in optimizers.items()},
                     loss_fn=torch.nn.MSELoss(), train_loader=loader, val_loader=loader,
                     config={"meta": {"use_amp": False}, "distributed": {"find_unused_parameters": True},
                             "trainer": {"exp_path": str(path), "accumulation_steps": accumulation,
                                         "strict_checks": False, "log_interval": 100,
                                         "lr_schedule_per_step": True,
                                         "validation_epoch_interval": 0, "save_curve_interval": 0}})


@pytest.mark.parametrize("bad_side", ["discriminator", "generator"])
def test_gan_real_scaler_tail_overflow_discards_pair_once(tmp_path, bad_side, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    trainer = _build_audit_gan(tmp_path, accumulation=2, batches=1)
    trainer.use_amp, trainer.amp_dtype = True, torch.float16
    trainer.scaler = torch.amp.GradScaler("cpu", init_scale=16.0)
    initial = {key: copy.deepcopy(model.state_dict()) for key, model in trainer.models.items()}
    handle = trainer.models[bad_side].linear.weight.register_hook(
        lambda grad: torch.full_like(grad, float("inf")))
    try:
        for epoch in range(3):
            trainer._train_epoch(epoch)
            assert trainer.scaler.get_scale() == 16.0 / 2 ** (epoch + 1)
            assert trainer.n_steps == 0
            for key, model in trainer.models.items():
                assert not trainer.optimizers[key].state
                assert trainer.schedulers[key].last_epoch == 0
                for name, value in initial[key].items():
                    torch.testing.assert_close(model.state_dict()[name], value)
        handle.remove()
        trainer._train_epoch(4)
        assert trainer.n_steps == 1
    finally:
        handle.remove()
        if trainer.writer is not None:
            trainer.writer.close()
        _close_log_handlers(tmp_path)


def test_se_real_epoch_tail_scaler_backoff_preserves_state(trainers):
    trainer = trainers("epoch_amp", accum=6)
    trainer.use_amp, trainer.amp_dtype = True, torch.float16
    trainer.scaler = torch.amp.GradScaler("cpu", init_scale=16.0)
    before = copy.deepcopy(trainer.model.state_dict())
    scheduler = copy.deepcopy(trainer.scheduler.state_dict())
    handle = next(trainer.model.parameters()).register_hook(
        lambda grad: torch.full_like(grad, float("inf")))
    try:
        for epoch in range(3):
            trainer._train_epoch(epoch)
            assert trainer.n_steps == 0
            assert not trainer.optimizer.state
            assert trainer.scheduler.state_dict() == scheduler
            assert trainer.scaler.get_scale() == 16.0 / 2 ** (epoch + 1)
            assert trainer.scaler.state_dict()["_growth_tracker"] == 0
            for key, value in before.items():
                torch.testing.assert_close(trainer.model.state_dict()[key], value)
        handle.remove()
        trainer._train_epoch(4)
        assert trainer.n_steps == 1
    finally:
        handle.remove()


@pytest.mark.parametrize("patch", [
    {"batch_cursor": 3}, {"batch_cursor": 7}, {"batch_cursor": 5.0},
    {"batch_cursor": True}, {"batch_cursor": 5, "batch": 3},
])
def test_resume_rejects_invalid_cursor_before_live_state_changes(trainers, patch):
    trainer = trainers("invalid", accum=2)
    trainer._train_epoch(1)
    state = copy.deepcopy(trainer._build_state(1, batch_cursor=5, epoch_incomplete=True))
    state.update(patch)
    before = copy.deepcopy(trainer.model.state_dict())
    for value in state["model"].values():
        value.add_(20)
    path = trainer.ckpt_dir / "invalid.tar"
    torch.save(state, path)
    trainer._resume_from = str(path)
    with pytest.raises((ValueError, TypeError)):
        trainer._resume_checkpoint()
    for key, value in before.items():
        torch.testing.assert_close(trainer.model.state_dict()[key], value)
