"""DualInputLossTrainer smoke-test.

Verifies the scalar-loss dual-input training loop (used for PASE Phase-2
DeWavLM distillation and StuPASE's flow-matching training) completes a
train epoch and a validation epoch with finite loss, without the
clamp(-1,1)/shape-assert/audio-metric-collection logic that a plain
SETrainer would apply (which would be wrong here since the model's output
is a scalar loss, not an enhanced waveform). Uses a tiny stand-in
distillation model (frozen "teacher" + trainable "student" linear layers)
rather than the real WavLM-based DeWavLMDistillModel, mirroring
tests/test_se_end_to_end_smoke.py's approach.
"""

from __future__ import annotations
import pytest

import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tasks.se.data.batch import SEBatch, SEBatchItem
from tasks.se.trainer.dual_input_loss_trainer import DualInputLossTrainer


class _TinyDualInputModel(nn.Module):
    """Frozen teacher / trainable student stand-in for DeWavLMDistillModel."""

    def __init__(self) -> None:
        super().__init__()
        self.teacher = nn.Linear(64, 32)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.student = nn.Linear(64, 32)

    def forward(self, noisy_wav: torch.Tensor, clean_wav: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Fake "feature extraction": project fixed-length waveform chunks through the linear layers.
        student_feat = self.student(noisy_wav[:, :64])
        with torch.no_grad():
            teacher_feat = self.teacher(clean_wav[:, :64])
        return F.mse_loss(student_feat, teacher_feat)


class _SyntheticDistillDataset(Dataset):
    def __init__(self, n_samples: int = 4, sr: int = 16000, length: int = 128) -> None:
        self.n_samples = n_samples
        self.sr = sr
        self.length = length

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        torch.manual_seed(42 + idx)
        clean = torch.randn(self.length) * 0.3
        noisy = clean + torch.randn(self.length) * 0.1
        return SEBatchItem(clean=clean, noisy=noisy, sr=self.sr, length=self.length)


def _collate(items):
    clean = torch.stack([it.clean for it in items])
    noisy = torch.stack([it.noisy for it in items])
    sr = torch.tensor([it.sr for it in items], dtype=torch.long)
    lengths = torch.tensor([it.length for it in items], dtype=torch.long)
    return SEBatch(clean=clean, noisy=noisy, sr=sr, lengths=lengths)


class _ScalarPassthroughLoss(nn.Module):
    def forward(self, enhanced, clean, lengths=None, **kwargs):
        return enhanced


def _build_trainer(exp_dir: Path) -> DualInputLossTrainer:
    model = _TinyDualInputModel()
    train_ds = _SyntheticDistillDataset(n_samples=4)
    val_ds = _SyntheticDistillDataset(n_samples=2)
    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=2, collate_fn=_collate)

    return DualInputLossTrainer(
        rank=0,
        world_size=1,
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        scheduler=torch.optim.lr_scheduler.ConstantLR(torch.optim.Adam(model.parameters(), lr=1e-3), factor=1.0),
        loss_fn=_ScalarPassthroughLoss(),
        config={
            "trainer": {
                "epochs": 1,
                "gradient_clip_value": 5.0,
                "exp_path": str(exp_dir),
                "val_metrics": {"enabled": False},
            },
            "meta": {"seed": 42, "allow_no_stft": True},
        },
        train_loader=train_loader,
        val_loader=val_loader,
        sample_rate=16000,
    )


def test_dual_input_train_epoch():
    tmpdir = tempfile.mkdtemp()
    try:
        exp_dir = Path(tmpdir)
        trainer = _build_trainer(exp_dir)

        param_before = next(trainer.model.student.parameters()).detach().clone()
        loss = trainer._train_epoch(epoch=0)

        assert loss is not None
        assert torch.isfinite(torch.tensor(loss)), f"loss is non-finite: {loss}"

        param_after = next(trainer.model.student.parameters()).detach().clone()
        assert not torch.allclose(param_before, param_after), "student parameters did not update"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_dual_input_validation_epoch():
    tmpdir = tempfile.mkdtemp()
    try:
        exp_dir = Path(tmpdir)
        trainer = _build_trainer(exp_dir)
        val_loss = trainer._validation_epoch(epoch=0)
        assert val_loss is not None
        assert torch.isfinite(torch.tensor(val_loss)), f"val loss is non-finite: {val_loss}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


pytestmark = [pytest.mark.external]
