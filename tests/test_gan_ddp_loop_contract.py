"""GANTrainer `_train_epoch` 本地边界与 middle checkpoint 回归测试。

本文件构造 world_size=1 的轻量 trainer，只覆盖单进程控制流（batch 异常同步
seam 由 stub 替换），不验证多 rank collective 语义；真实 DDP 一致性由
tests/test_ddp_collective_consistency.py 覆盖。
"""

from __future__ import annotations

import pytest
import torch

from tasks.se.trainer.gan_trainer import GANTrainer


pytestmark = [pytest.mark.unit]


def _new_loop_trainer() -> GANTrainer:
    """构造只覆盖 `_train_epoch` 边界的轻量 GAN trainer（单进程，不建进程组）。"""
    trainer = object.__new__(GANTrainer)
    trainer.device = torch.device("cpu")
    trainer.rank = 1  # 仅作为非默认哨兵值，本文件不启动真实 DDP
    trainer.world_size = 1
    trainer.use_amp = False
    trainer.amp_dtype = torch.float32
    trainer.accumulation_steps = 1
    trainer.start_epoch = 1
    trainer._resume_skip_batches = 0
    trainer._train_log_loss_win = []
    trainer.n_steps = 1
    trainer.max_steps = None
    trainer.discriminator = torch.nn.Linear(1, 1)
    trainer.models = {}
    trainer.optimizers = {}
    trainer._reset_output_sanitize_stats = lambda: None
    trainer._make_train_progress_bar = lambda _epoch: [(0, None)]
    trainer._sync_max_steps_reached = lambda: False
    trainer._sync_shutdown_signum = lambda: False
    trainer._log_train_loss_after_step = lambda: None
    trainer._log_output_sanitize_summary = lambda *_args: None
    trainer._gan_optimizer_step = lambda **_kwargs: True
    trainer._gan_skip_optimizer_steps = lambda: False
    trainer._handle_epoch_tail_gradients = lambda *_args, **_kwargs: False
    return trainer


@pytest.mark.unit
def test_gan_train_epoch_short_circuits_at_synced_max_steps():
    trainer = object.__new__(GANTrainer)
    calls: list[str] = []
    trainer._sync_max_steps_reached = lambda: calls.append("sync") or True
    trainer._resume_skip_batches = 3

    assert trainer._train_epoch(4) is None
    assert calls == ["sync"]
    assert trainer._train_epoch_completed is False
    assert trainer._last_completed_batch_cursor == 3


@pytest.mark.unit
@pytest.mark.parametrize(
    ("prepare_error", "forward_error", "expected_phase", "expected_message"),
    [
        (
            ValueError("invalid local batch"),
            None,
            "train_prepare_batch_error",
            "batch preparation failed",
        ),
        (
            None,
            RuntimeError("invalid local forward"),
            "train_forward_loss_error",
            "forward/loss failed",
        ),
    ],
)
def test_gan_train_epoch_syncs_recoverable_batch_errors(
    prepare_error,
    forward_error,
    expected_phase,
    expected_message,
):
    trainer = _new_loop_trainer()
    phases: list[str] = []

    def prepare(*_args):
        if prepare_error is not None:
            raise prepare_error
        return {}, 16000, 0

    def forward(_batch):
        if forward_error is not None:
            raise forward_error
        raise AssertionError("forward should not succeed in this test")

    def sync(local_error, *, phase):
        phases.append(phase)
        if phase == expected_phase:
            return f"rank 0: {type(local_error).__name__}: remote failure"
        return None

    trainer._prepare_batch = prepare
    trainer._forward_generator = forward
    trainer._sync_train_batch_exception = sync

    with pytest.raises(RuntimeError, match=expected_message):
        trainer._train_epoch(1)
    expected_phases = (
        [expected_phase]
        if prepare_error is not None
        else ["train_prepare_batch_error", expected_phase]
    )
    assert phases == expected_phases


@pytest.mark.parametrize("stage", ["prepare", "forward"])
def test_gan_train_epoch_fatal_cuda_error_skips_abort_sync(stage):
    """致命 CUDA 故障必须直接抛出，不得进入跨 rank abort collective。"""
    trainer = _new_loop_trainer()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("CUDA error: device-side assert triggered")

    trainer._prepare_batch = (
        _boom if stage == "prepare" else (lambda *_args: ({}, 16000, 0))
    )
    trainer._forward_generator = _boom if stage == "forward" else (lambda _batch: {})
    phases: list[str] = []
    trainer._sync_train_batch_exception = (
        lambda _error, *, phase: phases.append(phase) or None
    )

    with pytest.raises(RuntimeError, match="device-side assert"):
        trainer._train_epoch(1)

    fatal_phase = (
        "train_prepare_batch_error"
        if stage == "prepare"
        else "train_forward_loss_error"
    )
    assert fatal_phase not in phases


def test_gan_train_epoch_saves_middle_checkpoint_at_batch_interval():
    trainer = _new_loop_trainer()
    trainer.save_batch_interval = 1
    saved: list[tuple[int, int]] = []
    trainer._save_middle_checkpoint = lambda epoch, batch: saved.append((epoch, batch))
    trainer._prepare_batch = lambda *_args: ({}, 16000, 0)
    trainer._sync_train_batch_exception = lambda *_args, **_kwargs: None
    trainer._forward_generator = lambda batch: {
        **batch,
        "enhanced": torch.zeros(1, 1),
        "clean": torch.zeros(1, 1),
        "lengths": torch.ones(1, dtype=torch.long),
    }
    trainer._sanitize_forward_output = lambda forward: forward
    trainer._extract_gan_mag_pha = lambda _forward: (torch.zeros(1, 1, 1),) * 4
    trainer._get_discriminator_real_scores = lambda _forward: None
    trainer._prepare_discriminator_losses = lambda *_args, **_kwargs: []
    trainer._prepare_generator_losses = lambda *_args, **_kwargs: []
    trainer._train_step_discriminator = lambda *_args, **_kwargs: torch.zeros(1)
    trainer._train_step_generator = lambda *_args, **_kwargs: torch.zeros(1)

    trainer._train_epoch(1)

    assert saved == [(1, 0)]
