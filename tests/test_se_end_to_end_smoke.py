"""SE 端到端训练 smoke-test。

用合成数据构造最小训练循环，验证重构后 SETrainer 能完成
forward / backward / optimizer step / checkpoint save / resume。
"""

from __future__ import annotations
import pytest

import logging
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tasks.se.data.batch import SEBatch, SEBatchItem
from tasks.se.trainer.se_trainer import SETrainer


def _close_log_handlers(exp_dir: Path) -> None:
    """关闭并移除指向 exp_dir 下 train.log 的 FileHandler，避免 Windows 临时目录清理失败。"""
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                path = Path(h.baseFilename).resolve()
                if str(exp_dir.resolve()) in str(path):
                    h.close()
                    root.removeHandler(h)
            except Exception:
                # 清理失败不应掩盖测试失败，但要留下可诊断的记录。
                logging.debug("failed to close log handler %s", h, exc_info=True)


class _TinySEModel(nn.Module):
    """极简 wav-in/wav-out SE 模型，仅用于 smoke-test。"""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, noisy: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # noisy: (B, T)
        x = noisy.unsqueeze(1)
        x = self.conv(x)
        return x.squeeze(1)


class _SimpleWavLoss(nn.Module):
    """吸收 CompositeLoss 额外 kwargs 的简单 MSE loss。"""

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor, **kwargs) -> torch.Tensor:
        return F.mse_loss(enhanced, clean)


class _RecordingWavLoss(_SimpleWavLoss):
    """记录 train/validation 传入的逐样本有效长度。"""

    def __init__(self) -> None:
        super().__init__()
        self.lengths_seen: list[torch.Tensor] = []

    def forward(self, enhanced, clean, lengths, **kwargs) -> torch.Tensor:
        self.lengths_seen.append(lengths.detach().clone())
        return super().forward(enhanced, clean, lengths, **kwargs)


class _SyntheticSEDataset(Dataset):
    """返回合成 (clean, noisy) 波形对的 Dataset。"""

    def __init__(self, n_samples: int = 4, sr: int = 16000, length: int = 1024) -> None:
        self.n_samples = n_samples
        self.sr = sr
        self.length = length
        self.rng = torch.Generator().manual_seed(42)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        torch.manual_seed(42 + idx)
        clean = torch.randn(self.length) * 0.3
        noise = torch.randn(self.length) * 0.1
        noisy = clean + noise
        return SEBatchItem(clean=clean, noisy=noisy, sr=self.sr, length=self.length)


def _collate_se_batch(items):
    """将 SEBatchItem 列表拼成 SEBatch。"""
    clean = torch.stack([it.clean for it in items])
    noisy = torch.stack([it.noisy for it in items])
    sr = torch.tensor([it.sr for it in items], dtype=torch.long)
    lengths = torch.tensor([it.length for it in items], dtype=torch.long)
    return SEBatch(clean=clean, noisy=noisy, sr=sr, lengths=lengths)


def _build_se_trainer(
    exp_dir: Path,
    device: str = "cpu",
    loss_fn: nn.Module | None = None,
) -> SETrainer:
    """构造一个只跑少量 step 的 SETrainer。"""
    config = {
        "model": {"type": "TinySEModel", "args": {}},
        "trainer": {
            "epochs": 1,
            "max_steps": 4,
            "gradient_clip_value": 5.0,
            "accumulation_steps": 1,
            "validation_epoch_interval": 0,
            "log_interval": 1,
            "exp_path": str(exp_dir),
            "max_val_batches": 1,
        },
        "meta": {"seed": 42},
        "data": {"placeholder_num_samples": 1024},
    }

    model = _TinySEModel()
    loss_fn = loss_fn or _SimpleWavLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    train_ds = _SyntheticSEDataset(n_samples=4)
    val_ds = _SyntheticSEDataset(n_samples=2)
    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=_collate_se_batch)
    val_loader = DataLoader(val_ds, batch_size=2, collate_fn=_collate_se_batch)

    return SETrainer(
        rank=0,
        world_size=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        sample_rate=16000,
    )


def test_se_train_and_validation_pass_batch_lengths(tmp_path):
    """SE train/validation 必须向 loss 透传同一组逐样本长度。"""
    exp_dir = tmp_path / "run"
    try:
        loss_fn = _RecordingWavLoss()
        trainer = _build_se_trainer(exp_dir, loss_fn=loss_fn)
        batch = next(iter(trainer.train_loader))
        batch.lengths = torch.tensor([901, 733], dtype=torch.long)
        expected = batch.lengths.clone()
        for field_name in batch.__dataclass_fields__:
            value = getattr(batch, field_name)
            if torch.is_tensor(value):
                setattr(batch, field_name, value.to(trainer.device))

        trainer._forward_and_compute_loss(batch, epoch=0, batch_idx=0, fs_val=16000)
        trainer._val_forward_loss(batch, epoch=0, n_loops=0)

        assert len(loss_fn.lengths_seen) == 2
        assert all(torch.equal(value.cpu(), expected) for value in loss_fn.lengths_seen)
    finally:
        _close_log_handlers(exp_dir)


def test_se_train_two_steps(tmp_path):
    """跑两个训练 step，断言 loss 有限且模型参数被更新。"""
    exp_dir = tmp_path / "run"
    try:
        trainer = _build_se_trainer(exp_dir)

        # 记录训练前参数
        param_before = next(trainer.model.parameters()).detach().clone()

        loss = trainer._train_epoch(epoch=0)

        assert loss is not None
        assert torch.isfinite(torch.tensor(loss)), f"SE smoke loss is non-finite: {loss}"

        param_after = next(trainer.model.parameters()).detach().clone()
        assert not torch.allclose(param_before, param_after), "SE model parameters did not update"
    finally:
        _close_log_handlers(exp_dir)


def test_se_checkpoint_save_resume(tmp_path):
    """保存 checkpoint 后能正确 resume。"""
    exp_dir = tmp_path / "run"
    try:
        trainer = _build_se_trainer(exp_dir)
        trainer._train_epoch(epoch=0)
        trainer._save_checkpoint(epoch=0, is_best=False)

        ckpt_path = list((exp_dir / "checkpoints").rglob("*.tar"))
        assert len(ckpt_path) > 0, "SE checkpoint not saved"

        # 重新构造 trainer 并 resume
        trainer2 = _build_se_trainer(exp_dir)
        trainer2._resume_checkpoint()
        # 简单验证 resume 后模型与前一个 trainer 一致
        for p1, p2 in zip(trainer.model.parameters(), trainer2.model.parameters()):
            torch.testing.assert_close(p1, p2)
    finally:
        _close_log_handlers(exp_dir)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        # 每个用例独立 exp 目录，避免第二个用例 resume 到前一个用例的 checkpoint。
        test_se_train_two_steps(Path(tmp) / "train-steps")
        test_se_checkpoint_save_resume(Path(tmp) / "checkpoint-resume")
    print("SE end-to-end smoke tests passed.")

pytestmark = [pytest.mark.integration]
