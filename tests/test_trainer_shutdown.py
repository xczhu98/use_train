"""信号优雅停机回归测试：flag-only handler 与训练主流程安全边界退出。

覆盖契约：
- 信号 handler 只写本地 flag（entry 与 trainer），不在异步上下文中执行
  checkpoint 保存、process group 销毁或 sys.exit；
- 第二次信号直接 ``os._exit(128 + signum)`` 强制退出，避免被卡死的主流程
  无限拖延；
- ``_sync_shutdown_signum`` 作为固定序列 collective 同步各 rank 退出判定；
- ``_train_epoch`` 仅在 optimizer/epoch-tail boundary 同步停机：未提交的
  accumulation window 整体丢弃，cursor 对齐到最后已提交边界；
- validation 在每次 forward 前和 terminal boundary 固定同步，不记录 partial 结果；
- ``train()`` 在 epoch 边界保存带 batch cursor 的 checkpoint 后退出循环。
"""

from __future__ import annotations

import logging
import os
import signal
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from core.trainer.base_trainer import BaseTrainer
from core.trainer.checkpoint_state import (
    resume_batch_cursor,
    resume_start_epoch,
    validate_resume_progress,
)
from core.trainer.entry import UnifiedTrainingEntry
from core.trainer.metrics_trainer import MetricsCapableTrainer
from core.trainer.validation_orchestration import (
    ValidationBatchResult,
    run_validation_batch_loop,
)
from core.utils.safe_load import safe_torch_load
from tasks.se.data.batch import SEBatch, SEBatchItem
from tasks.se.trainer.se_trainer import SETrainer


class _TrainerStub:
    """记录停机请求；信号 handler 绝不允许触发 checkpoint 保存。"""

    def __init__(self) -> None:
        self.requested: list[int] = []
        self.save_calls = 0
        self.current_epoch = 3

    def request_shutdown(self, signum: int) -> None:
        self.requested.append(int(signum))

    def _save_checkpoint(self, *args, **kwargs):
        self.save_calls += 1
        raise AssertionError("signal handler must not save checkpoints")


def _entry_stub() -> UnifiedTrainingEntry:
    class _ConcreteEntry(UnifiedTrainingEntry):
        def build_loaders(self, cfg, rank, local_rank, world_size):
            raise NotImplementedError

        def build_model(self, cfg, rank):
            raise NotImplementedError

        def get_trainer_cls(self):
            raise NotImplementedError

    entry = object.__new__(_ConcreteEntry)
    entry._shutdown_requested = False
    entry.logger = logging.getLogger("test_trainer_shutdown")
    return entry


def test_signal_handler_only_sets_flags():
    entry = _entry_stub()
    trainer = _TrainerStub()

    entry._graceful_shutdown(signal.SIGTERM, None, trainer, rank=0, world_size=1)

    assert entry._shutdown_requested is True
    assert trainer.requested == [int(signal.SIGTERM)]
    assert trainer.save_calls == 0


def test_second_signal_force_exits(monkeypatch):
    entry = _entry_stub()
    entry._shutdown_requested = True
    exits: list[int] = []
    monkeypatch.setattr(os, "_exit", lambda code: exits.append(code))

    entry._graceful_shutdown(signal.SIGTERM, None, _TrainerStub(), rank=0, world_size=1)

    assert exits == [128 + int(signal.SIGTERM)]


def test_training_cleanup_failure_preserves_primary_exception(monkeypatch):
    entry = _entry_stub()

    class _FailingTrainer:
        def train(self):
            raise RuntimeError("primary")

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        dist,
        "destroy_process_group",
        lambda: (_ for _ in ()).throw(RuntimeError("cleanup")),
    )

    with pytest.raises(RuntimeError, match="primary"):
        entry._run_trainer_main_loop(_FailingTrainer(), world_size=2)


def test_training_cleanup_failure_preserves_signal_exit_code(monkeypatch):
    entry = _entry_stub()

    class _ShutdownTrainer:
        _shutdown_signal = int(signal.SIGTERM)

        def train(self):
            return None

        def shutdown_requested(self):
            return True

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        dist,
        "destroy_process_group",
        lambda: (_ for _ in ()).throw(RuntimeError("cleanup")),
    )

    with pytest.raises(SystemExit) as exc_info:
        entry._run_trainer_main_loop(_ShutdownTrainer(), world_size=2)

    assert exc_info.value.code == 128 + int(signal.SIGTERM)


def test_keyboard_interrupt_cleanup_failure_preserves_exit_code(monkeypatch):
    entry = _entry_stub()
    monkeypatch.setattr(entry, "_ensure_free_master_port", lambda: None)
    monkeypatch.setattr(
        entry,
        "load_and_setup",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        dist,
        "destroy_process_group",
        lambda: (_ for _ in ()).throw(RuntimeError("cleanup")),
    )

    with pytest.raises(SystemExit) as exc_info:
        entry.run()

    assert exc_info.value.code == 130


def _base_trainer_stub() -> BaseTrainer:
    trainer = object.__new__(BaseTrainer)
    trainer._shutdown_requested = False
    trainer._shutdown_signal = None
    trainer.world_size = 1
    trainer.device = torch.device("cpu")
    return trainer


def test_sync_shutdown_signum_single_rank_local_only():
    trainer = _base_trainer_stub()
    assert trainer._sync_shutdown_signum() == 0

    trainer.request_shutdown(signal.SIGINT)

    assert trainer._sync_shutdown_signum() == int(signal.SIGINT)
    assert trainer.shutdown_requested()


def test_sync_shutdown_signum_allreduce_max_propagates_to_local_flags(monkeypatch):
    trainer = _base_trainer_stub()
    trainer.world_size = 2

    def fake_all_reduce(tensor, *, op):
        assert op is dist.ReduceOp.MAX
        tensor.fill_(int(signal.SIGTERM))  # 模拟另一 rank 收到 SIGTERM

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

    assert trainer._sync_shutdown_signum() == int(signal.SIGTERM)
    assert trainer.shutdown_requested()
    assert trainer._shutdown_signal == int(signal.SIGTERM)


# ---------------------------------------------------------------------------
# _train_epoch / train() 安全边界行为（单 rank，CPU）
# ---------------------------------------------------------------------------


class _TinySEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, noisy: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.conv(noisy.unsqueeze(1)).squeeze(1)


class _ShutdownInjectingLoss(nn.Module):
    """第 ``inject_at_call`` 次调用时向 trainer 写入停机请求（模拟信号）。"""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.trainer: SETrainer | None = None
        self.inject_at_call = 2

    def forward(self, enhanced, clean, lengths, **kwargs):
        self.calls += 1
        if self.trainer is not None and self.calls == self.inject_at_call:
            self.trainer.request_shutdown(signal.SIGTERM)
        return torch.nn.functional.mse_loss(enhanced, clean)


class _SyntheticSEDataset(Dataset):
    def __init__(self, n_samples: int = 8, sr: int = 16000, length: int = 1024) -> None:
        self.n_samples = n_samples
        self.sr = sr
        self.length = length

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        torch.manual_seed(42 + idx)
        clean = torch.randn(self.length) * 0.3
        noise = torch.randn(self.length) * 0.1
        return SEBatchItem(clean=clean, noisy=clean + noise, sr=self.sr, length=self.length)


def _collate(items):
    clean = torch.stack([it.clean for it in items])
    noisy = torch.stack([it.noisy for it in items])
    sr = torch.tensor([it.sr for it in items], dtype=torch.long)
    lengths = torch.tensor([it.length for it in items], dtype=torch.long)
    return SEBatch(clean=clean, noisy=noisy, sr=sr, lengths=lengths)


def _build_trainer(
    tmp_path: Path,
    *,
    epochs: int = 3,
    inject_at_call: int = 2,
    validation_epoch_interval: int = 0,
    accumulation_steps: int = 1,
) -> tuple[SETrainer, _ShutdownInjectingLoss]:
    config = {
        "model": {"type": "TinySEModel", "args": {}},
        "trainer": {
            "epochs": epochs,
            "gradient_clip_value": 5.0,
            "accumulation_steps": accumulation_steps,
            "validation_epoch_interval": validation_epoch_interval,
            "log_interval": 100,
            "exp_path": str(tmp_path),
            "strict_checks": False,
            "save_interval": 0,
            "save_batch_interval": 0,
            "save_curve_interval": 0,
        },
        "meta": {"seed": 42, "use_amp": False},
        "data": {"placeholder_num_samples": 1024},
    }
    torch.manual_seed(0)
    model = _TinySEModel()
    loss_fn = _ShutdownInjectingLoss()
    loss_fn.inject_at_call = inject_at_call
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = _SyntheticSEDataset(n_samples=8)
    dl = DataLoader(ds, batch_size=2, collate_fn=_collate)
    trainer = SETrainer(
        rank=0,
        world_size=1,
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0),
        loss_fn=loss_fn,
        config=config,
        train_loader=dl,
        val_loader=dl,
        sample_rate=16000,
    )
    loss_fn.trainer = trainer
    return trainer, loss_fn


def test_train_epoch_stops_at_batch_boundary_on_shutdown(tmp_path):
    trainer, loss_fn = _build_trainer(tmp_path)

    trainer._train_epoch(epoch=0)

    # 第 2 次 loss 调用（batch 1）注入信号；该 optimizer boundary 同步后
    # 丢弃尚未提交的窗口，cursor 保持在上一已提交边界。
    assert loss_fn.calls == 2
    assert trainer.n_steps == 1
    assert trainer._train_epoch_completed is False
    assert trainer._last_completed_batch_cursor == 1
    assert trainer.shutdown_requested()


def test_shutdown_sync_only_runs_on_optimizer_boundaries(tmp_path):
    """accumulation>1 时不得在每个 microbatch 顶部阻塞同步。"""
    trainer, loss_fn = _build_trainer(
        tmp_path,
        accumulation_steps=2,
        inject_at_call=3,
    )
    sync_calls = 0
    original_sync = trainer._sync_shutdown_signum

    def counted_sync():
        nonlocal sync_calls
        sync_calls += 1
        return original_sync()

    trainer._sync_shutdown_signum = counted_sync
    trainer._train_epoch(epoch=0)

    # 第一个 window 两个 microbatch 后同步一次并提交；信号在第二个 window
    # 首个 microbatch 到达，该窗口完成 backward 后在边界同步并整体丢弃。
    assert loss_fn.calls == 4
    assert trainer.n_steps == 1
    assert trainer._last_completed_batch_cursor == 2
    assert sync_calls == 2


def test_train_saves_cursor_checkpoint_and_exits_on_shutdown(tmp_path):
    trainer, loss_fn = _build_trainer(tmp_path, epochs=3)

    trainer.train()

    # 第 2 个窗口内收到信号，该窗口在 optimizer boundary 整体丢弃；checkpoint
    # cursor 只指向第 1 个已提交 batch，恢复后完整重做被丢弃窗口。
    assert trainer.n_steps == 1
    assert trainer.current_epoch == 1
    assert trainer._shutdown_signal == int(signal.SIGTERM)
    ckpts = list(tmp_path.glob("checkpoints/*/model_0001.pth"))
    assert len(ckpts) == 1, f"missing shutdown checkpoint under {tmp_path}/checkpoints"
    state = safe_torch_load(
        ckpts[0],
        allow_unsafe=True,
        purpose="inspect shutdown cursor checkpoint in regression test",
    )
    assert state["batch_cursor"] == 1
    assert state["epoch_incomplete"] is True
    assert state["epoch"] == 1


def test_shutdown_after_final_batch_finishes_epoch_finalization(tmp_path):
    """最后训练窗口提交后收到信号，仍完成 scheduler 与完整 checkpoint。"""
    trainer, loss_fn = _build_trainer(
        tmp_path,
        epochs=3,
        inject_at_call=10_000,
        validation_epoch_interval=1,
    )
    original_train_epoch = trainer._train_epoch

    def request_after_completed_train_epoch(epoch, *, step_callback=None):
        result = original_train_epoch(epoch, step_callback=step_callback)
        assert trainer._train_epoch_completed is True
        trainer.request_shutdown(signal.SIGTERM)
        return result

    trainer._train_epoch = request_after_completed_train_epoch
    trainer.train()

    # 四个训练窗口均提交；validation 在首个 fixed boundary 观察到信号并零
    # forward 中止，但 scheduler 与完整 epoch checkpoint 仍精确提交一次。
    assert loss_fn.calls == 4
    assert trainer.n_steps == 4
    assert trainer.current_epoch == 1
    assert trainer.scheduler.last_epoch == 1
    ckpts = list(tmp_path.glob("checkpoints/*/model_0001.pth"))
    assert len(ckpts) == 1
    state = safe_torch_load(
        ckpts[0],
        allow_unsafe=True,
        purpose="inspect fully finalized shutdown checkpoint in regression test",
    )
    assert "epoch_incomplete" not in state
    assert "batch_cursor" not in state
    assert state["scheduler"]["last_epoch"] == 1
    assert resume_start_epoch(state, trainer.accumulation_steps) == 2


def test_zero_batch_cursor_resumes_same_epoch_when_marked_incomplete():
    """cursor=0 需配合 incomplete 标记恢复，并兼容 legacy batch 字段。"""
    assert resume_start_epoch(
        {"epoch": 3, "batch_cursor": 0, "epoch_incomplete": True}, 4
    ) == 3
    assert resume_start_epoch({"epoch": 3, "batch": 0, "batch_cursor": 0}, 4) == 3
    assert resume_start_epoch({"epoch": 3, "batch_cursor": 0}, 4) == 4
    assert resume_start_epoch({"epoch": 3}, 4) == 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epoch", True),
        ("epoch", 1.0),
        ("epoch", "1"),
        ("epoch", -1),
        ("n_steps", False),
        ("n_steps", 2.0),
        ("n_steps", "2"),
        ("n_steps", -1),
        ("batch_cursor", True),
        ("batch_cursor", 2.0),
        ("batch_cursor", "2"),
        ("batch_cursor", -1),
        ("batch", False),
        ("batch", 1.0),
        ("batch", "1"),
        ("batch", -1),
        ("epoch_incomplete", 1),
        ("epoch_incomplete", "true"),
    ],
)
def test_resume_progress_rejects_non_exact_or_negative_values(field, value):
    checkpoint = {"epoch": 3, "n_steps": 2}
    checkpoint[field] = value
    with pytest.raises((TypeError, ValueError), match=field):
        validate_resume_progress(checkpoint, accumulation_steps=2, train_loader_len=4)


def test_resume_progress_requires_aligned_cursor_and_incomplete_progress_field():
    with pytest.raises(ValueError, match="aligned"):
        validate_resume_progress(
            {"epoch": 3, "n_steps": 2, "batch_cursor": 1},
            accumulation_steps=2,
            train_loader_len=4,
        )
    with pytest.raises(RuntimeError, match="progress field"):
        validate_resume_progress(
            {"epoch": 3, "n_steps": 2, "epoch_incomplete": True},
            accumulation_steps=2,
            train_loader_len=4,
        )


def test_resume_progress_loader_bounds_and_legacy_alignment():
    checkpoint = {
        "epoch": 3,
        "n_steps": 2,
        "batch_cursor": 4,
        "epoch_incomplete": True,
    }
    validate_resume_progress(checkpoint, accumulation_steps=2, train_loader_len=4)
    assert resume_batch_cursor(checkpoint, 2) == 4

    with pytest.raises(ValueError, match="train loader length"):
        validate_resume_progress(
            {**checkpoint, "batch_cursor": 6},
            accumulation_steps=2,
            train_loader_len=4,
        )

    legacy = {"epoch": 3, "n_steps": 2, "batch": 2}
    validate_resume_progress(legacy, accumulation_steps=2, train_loader_len=4)
    assert resume_batch_cursor(legacy, 2) == 2


def test_resume_progress_rejects_legacy_batch_at_loader_length():
    valid = {"epoch": 3, "n_steps": 2, "batch": 3}
    validate_resume_progress(valid, accumulation_steps=2, train_loader_len=4)

    with pytest.raises(ValueError, match="legacy batch 4.*train loader length 4"):
        validate_resume_progress(
            {**valid, "batch": 4},
            accumulation_steps=2,
            train_loader_len=4,
        )


def test_resume_progress_requires_consistent_legacy_batch_and_cursor():
    checkpoint = {
        "epoch": 3,
        "n_steps": 2,
        "batch": 2,
        "batch_cursor": 2,
        "epoch_incomplete": True,
    }
    validate_resume_progress(checkpoint, accumulation_steps=2, train_loader_len=4)

    with pytest.raises(ValueError, match="inconsistent.*batch_cursor=4.*legacy batch=2"):
        validate_resume_progress(
            {**checkpoint, "batch_cursor": 4},
            accumulation_steps=2,
            train_loader_len=4,
        )


def test_shutdown_before_first_new_batch_preserves_resume_cursor(tmp_path):
    trainer, _loss_fn = _build_trainer(tmp_path)
    trainer.start_epoch = 1
    trainer._resume_skip_batches = 2
    trainer.request_shutdown(signal.SIGTERM)

    trainer._train_epoch(epoch=1)

    assert trainer.n_steps == 0
    assert trainer._train_epoch_completed is False
    assert trainer._last_completed_batch_cursor == 2


@pytest.mark.parametrize(
    ("sync_results", "expected_forwards", "expected_sync_calls", "aborted"),
    [
        ([signal.SIGTERM, 0], 0, 2, True),
        ([0, signal.SIGTERM, 0], 1, 3, True),
        ([0, 0, 0], 2, 3, False),
    ],
)
def test_validation_loop_syncs_shutdown_at_fixed_boundaries(
    sync_results, expected_forwards, expected_sync_calls, aborted
):
    """每批前及 terminal boundary 固定同步；中止时不得本地独立 break。"""
    trainer = object.__new__(MetricsCapableTrainer)
    trainer.device = torch.device("cpu")
    trainer.val_loader = [object(), object()]
    trainer.max_val_batches = 0
    trainer._vm_max_metrics = 4
    trainer._ema_context = lambda: __import__("contextlib").nullcontext()
    trainer._val_iter_is_ddp_padding = lambda _count: False
    trainer._accumulate_val_loss = (
        lambda loss, running, count, *_args, weight=1, **_kwargs: (
            running + loss * weight,
            count + weight,
            True,
        )
    )
    sync_iter = iter(sync_results)
    sync_calls = 0

    def sync_shutdown():
        nonlocal sync_calls
        sync_calls += 1
        return next(sync_iter)

    trainer._sync_shutdown_signum = sync_shutdown
    trainer._sync_train_batch_exception = lambda *_args, **_kwargs: None
    forwards = 0

    class _Adapter:
        def prepare_batch(self, raw_batch, epoch, loop_count):
            return raw_batch

        def forward_batch(self, raw_batch, epoch, loop_count):
            nonlocal forwards
            forwards += 1
            value = torch.tensor([float(loop_count)])
            return ValidationBatchResult(value, value, value, torch.tensor([1]), 16000)

        def loss_weight(self, batch):
            return 1

        def after_loss(self, batch, **kwargs):
            return None

        def collect_metric_pairs(self, batch, pairs, total_pairs):
            pairs.append((np.zeros(1), np.ones(1), 16000))
            return total_pairs + 1

        def on_max_batches(self, epoch, max_batches):
            return None

    result = run_validation_batch_loop(
        trainer, 1, _Adapter(), collect_metrics=True
    )

    assert forwards == expected_forwards
    assert sync_calls == expected_sync_calls
    assert result.aborted is aborted
    assert result.metric_pairs == ([] if aborted else result.metric_pairs)


@pytest.mark.parametrize(
    "phase",
    ["prepare", "forward_loss", "loss_weight", "after_loss", "metric_pair_collection"],
)
def test_validation_loop_synchronizes_each_callback_failure(phase):
    trainer = object.__new__(MetricsCapableTrainer)
    trainer.device = torch.device("cpu")
    trainer.val_loader = [object()]
    trainer.max_val_batches = 0
    trainer._vm_max_metrics = 1
    trainer._ema_context = lambda: __import__("contextlib").nullcontext()
    trainer._val_iter_is_ddp_padding = lambda _count: False
    trainer._sync_shutdown_signum = lambda: 0
    trainer._accumulate_val_loss = lambda loss, running, count, *_args, weight=1, **_kwargs: (
        running + loss,
        count + weight,
        True,
    )

    def sync_error(local_error, *, phase: str):
        if phase == f"validation_{target_phase}_error":
            return (
                f"rank 1: ValueError: injected {phase}"
                if local_error is None
                else str(local_error)
            )
        return None

    trainer._sync_train_batch_exception = sync_error
    target_phase = phase

    class _Adapter:
        def _maybe_fail(self, current):
            if current == phase:
                raise ValueError(f"injected {current}")

        def prepare_batch(self, raw_batch, epoch, loop_count):
            self._maybe_fail("prepare")
            return raw_batch

        def forward_batch(self, raw_batch, epoch, loop_count):
            self._maybe_fail("forward_loss")
            value = torch.tensor([1.0])
            return ValidationBatchResult(value, value, value, torch.tensor([1]), 16000)

        def loss_weight(self, batch):
            self._maybe_fail("loss_weight")
            return 1

        def after_loss(self, batch, **kwargs):
            self._maybe_fail("after_loss")

        def collect_metric_pairs(self, batch, pairs, total_pairs):
            self._maybe_fail("metric_pair_collection")
            return total_pairs

        def on_max_batches(self, epoch, max_batches):
            return None

    with pytest.raises(RuntimeError, match=f"validation {phase} failed"):
        run_validation_batch_loop(trainer, 1, _Adapter(), collect_metrics=True)


@pytest.mark.parametrize("abort_phase", ["metrics", "extra"])
def test_full_validation_rolls_back_when_post_forward_phase_aborts(
    tmp_path, abort_phase
):
    """metrics/ERLE 阶段停机也必须让完整 validation 原子返回 None。"""
    trainer, _loss_fn = _build_trainer(tmp_path, inject_at_call=10_000)
    events = []
    trainer._log_val_loss_summary = lambda *_args: events.append("loss")
    trainer._val_log_extra_metrics = lambda *_args: events.append("extra_log")
    trainer._log_output_sanitize_summary = lambda *_args: events.append("sanitize")
    trainer._run_validation_metrics_from_pairs = (
        lambda *_args, **_kwargs: abort_phase == "metrics"
    )
    trainer._val_run_extra_metrics = (
        lambda *_args, **_kwargs: abort_phase == "extra"
    )
    vm_count_before = trainer._vm_epoch_count

    result = trainer._validation_epoch_impl(epoch=1)

    assert result is None
    assert trainer._vm_epoch_count == vm_count_before
    assert events == []
    assert trainer.val_metrics_epochs == []
    assert trainer.val_metrics_history == {}


pytestmark = [pytest.mark.unit]
