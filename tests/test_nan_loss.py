"""验证非有限 loss 的处理语义。

- 训练中遇到 NaN/Inf loss 时不应崩溃；
- NaN batch 不应导致运行平均 loss 变成 NaN；
- 全 NaN accumulation window 不应更新模型参数或优化器状态。
"""

from __future__ import annotations

import copy
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tasks.se.data.batch import SEBatch, SEBatchItem
from tasks.se.trainer.se_trainer import SETrainer


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


class _NaNLoss(nn.Module):
    """总是返回 NaN 的 loss，用于验证 trainer 的容错路径。"""

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.tensor(float("nan"), device=enhanced.device, requires_grad=True)


class _FiniteThenNaNLoss(nn.Module):
    """同一 accumulation window 中先返回 finite、再返回 NaN。"""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor, **kwargs) -> torch.Tensor:
        self.calls += 1
        if self.calls == 1:
            return F.mse_loss(enhanced, clean)
        return torch.tensor(float("nan"), device=enhanced.device, requires_grad=True)


class _StubEMA:
    """记录真实 optimizer step 触发的 EMA 更新次数。"""

    def __init__(self) -> None:
        self.n_updates = 0

    def update(self) -> None:
        self.n_updates += 1


class _SyntheticSEDataset(Dataset):
    def __init__(self, n_samples: int = 4, sr: int = 16000, length: int = 1024) -> None:
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


def _build_nan_trainer(
    exp_dir: Path,
    accumulation_steps: int,
    loss_fn: nn.Module | None = None,
) -> SETrainer:
    config = {
        "model": {"type": "TinySEModel", "args": {}},
        "trainer": {
            "epochs": 1,
            "max_steps": 2,
            "gradient_clip_value": 5.0,
            "accumulation_steps": accumulation_steps,
            "validation_epoch_interval": 0,
            "log_interval": 1,
            "lr_schedule_per_step": True,
            "exp_path": str(exp_dir),
            "max_val_batches": 1,
            "strict_checks": False,
        },
        "meta": {"seed": 42},
        "data": {"placeholder_num_samples": 1024},
    }
    model = _TinySEModel()
    loss_fn = loss_fn or _NaNLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=0.1
    )
    ds = _SyntheticSEDataset(n_samples=4)
    dl = DataLoader(ds, batch_size=2, collate_fn=_collate)
    trainer = SETrainer(
        rank=0,
        world_size=1,
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5),
        loss_fn=loss_fn,
        config=config,
        train_loader=dl,
        val_loader=dl,
        sample_rate=16000,
    )
    trainer.ema = _StubEMA()
    # 先执行一次真实更新，建立 AdamW 的 exp_avg、exp_avg_sq 和 step 状态；
    # 后续全 NaN window 若错误调用 optimizer.step，weight decay/momentum 会暴露该回归。
    batch = next(iter(dl))
    noisy = batch.noisy.to(trainer.device)
    lengths = batch.lengths.to(trainer.device)
    enhanced = trainer.model(noisy, lengths)
    trainer.scaler.scale(enhanced.square().mean()).backward()
    assert trainer._optimizer_step()
    return trainer


def _assert_optimizer_state_equal(before: dict, after: dict) -> None:
    assert before.keys() == after.keys()
    assert before["param_groups"] == after["param_groups"]
    assert before["state"].keys() == after["state"].keys()
    for param_id, before_state in before["state"].items():
        after_state = after["state"][param_id]
        assert before_state.keys() == after_state.keys()
        for key, before_value in before_state.items():
            after_value = after_state[key]
            if torch.is_tensor(before_value):
                assert torch.equal(before_value, after_value), (
                    f"optimizer state changed for parameter {param_id}, field {key}"
                )
            else:
                assert before_value == after_value


@pytest.mark.parametrize(
    "accumulation_steps",
    [2, 3],
    ids=["full-window", "epoch-tail"],
)
def test_all_nan_window_does_not_update_params_or_optimizer_state(accumulation_steps: int):
    """全 NaN full/tail window 不得触发 AdamW、EMA、scheduler 或 n_steps。"""
    tmpdir = tempfile.mkdtemp()
    try:
        trainer = _build_nan_trainer(Path(tmpdir), accumulation_steps)
        params_before = [p.detach().clone() for p in trainer.model.parameters()]
        optimizer_before = copy.deepcopy(trainer.optimizer.state_dict())
        scheduler_before = trainer.scheduler.last_epoch
        ema_before = trainer.ema.n_updates
        n_steps_before = trainer.n_steps

        loss = trainer._train_epoch(epoch=0)

        assert loss == 0.0, f"expected average loss 0 for all-NaN epoch, got {loss}"
        assert all(
            torch.equal(before, after.detach())
            for before, after in zip(params_before, trainer.model.parameters())
        ), "model params changed after all-NaN window"
        _assert_optimizer_state_equal(optimizer_before, trainer.optimizer.state_dict())
        assert trainer.scheduler.last_epoch == scheduler_before
        assert trainer.ema.n_updates == ema_before
        assert trainer.n_steps == n_steps_before
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.parametrize(
    "accumulation_steps",
    [2, 3],
    ids=["full-window", "epoch-tail"],
)
def test_finite_and_nan_same_window_still_steps(accumulation_steps: int):
    """同一窗口含 finite loss 时允许真实 step，不把 NaN batch 当作整窗跳步。"""
    tmpdir = tempfile.mkdtemp()
    try:
        trainer = _build_nan_trainer(
            Path(tmpdir), accumulation_steps, loss_fn=_FiniteThenNaNLoss()
        )
        params_before = [p.detach().clone() for p in trainer.model.parameters()]
        scheduler_before = trainer.scheduler.last_epoch
        ema_before = trainer.ema.n_updates
        n_steps_before = trainer.n_steps

        loss = trainer._train_epoch(epoch=0)

        assert loss is not None and loss > 0.0
        assert any(
            not torch.equal(before, after.detach())
            for before, after in zip(params_before, trainer.model.parameters())
        ), "finite loss window did not update model params"
        assert trainer.scheduler.last_epoch == scheduler_before + 1
        assert trainer.ema.n_updates == ema_before + 1
        assert trainer.n_steps == n_steps_before + 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


pytestmark = [
    pytest.mark.unit,
    pytest.mark.usefixtures("preserve_tensorboard_module_state"),
]
