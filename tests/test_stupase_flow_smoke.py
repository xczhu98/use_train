"""StuPASE flow-matching (StuP2) smoke-test: DualInputLossTrainer + StuPASEFlowTrainModel.

Builds the real StuPASEFlowTrainModel (frozen WavLM encoder + trainable
CFM/DiT) with a shrunk WavLM transformer (2 layers, 32-dim) so the test runs
fast, then verifies DualInputLossTrainer completes a train epoch and a
validation epoch with finite loss.
"""

from __future__ import annotations
import pytest

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tasks.se.data.batch import SEBatch, SEBatchItem
from tasks.se.models.pase_family.stupase.flow_train_wrapper import StuPASEFlowTrainModel
from tasks.se.models.pase_family.common.wavlm.WavLM import WavLM
from tasks.se.trainer.dual_input_loss_trainer import DualInputLossTrainer


def _build_tiny_flow_model() -> StuPASEFlowTrainModel:
    model = StuPASEFlowTrainModel(
        encoder_output_layer=1,
        cfm_kwargs={
            "dit_kwargs": dict(dim=32, depth=1, heads=2, dim_head=16, ff_mult=2, mel_dim=20, text_in_dim=32, text_dim=16, conv_layers=1),
            "mel_spec_kwargs": dict(n_fft=256, hop_length=64, win_length=256, n_mel_channels=20, target_sample_rate=16000),
        },
    )
    # Shrink the WavLM encoder transformer for test speed (WavLM_feat has no
    # public hook for a tiny from-scratch config; override cfg + rebuild the
    # transformer directly, matching the framework's own Large-config path).
    cfg = model.encoder.cfg
    cfg.encoder_layers = 2
    cfg.encoder_embed_dim = 32
    cfg.encoder_ffn_embed_dim = 64
    cfg.encoder_attention_heads = 2
    model.encoder.wavlm = WavLM(cfg)
    for p in model.encoder.wavlm.parameters():
        p.requires_grad = False
    return model


class _SyntheticFlowDataset(Dataset):
    def __init__(self, n_samples: int = 4, sr: int = 16000, length: int = 4000) -> None:
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
    model = _build_tiny_flow_model()
    train_ds = _SyntheticFlowDataset(n_samples=4)
    val_ds = _SyntheticFlowDataset(n_samples=2)
    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=2, collate_fn=_collate)

    return DualInputLossTrainer(
        rank=0,
        world_size=1,
        model=model,
        optimizer=torch.optim.Adam(model.cfm.parameters(), lr=1e-3),
        scheduler=torch.optim.lr_scheduler.ConstantLR(torch.optim.Adam(model.cfm.parameters(), lr=1e-3), factor=1.0),
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


def test_stupase_flow_train_epoch(tmp_path):
    exp_dir = tmp_path / "run"
    trainer = _build_trainer(exp_dir)

    param_before = [p.detach().clone() for p in trainer.model.cfm.parameters()]
    loss = trainer._train_epoch(epoch=0)

    assert loss is not None
    assert torch.isfinite(torch.tensor(loss)), f"loss is non-finite: {loss}"

    param_after = list(trainer.model.cfm.parameters())
    any_changed = any(
        not torch.allclose(before, after.detach())
        for before, after in zip(param_before, param_after)
    )
    assert any_changed, "no CFM parameters were updated"

    # Encoder must stay frozen.
    assert all(p.grad is None for p in trainer.model.encoder.parameters())


def test_stupase_flow_validation_epoch(tmp_path):
    exp_dir = tmp_path / "run"
    trainer = _build_trainer(exp_dir)
    val_loss = trainer._validation_epoch(epoch=0)
    assert val_loss is not None
    assert torch.isfinite(torch.tensor(val_loss)), f"val loss is non-finite: {val_loss}"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        # 每个用例独立 exp 目录，保持与 pytest tmp_path 一致的隔离。
        test_stupase_flow_train_epoch(Path(tmp) / "train-epoch")
        test_stupase_flow_validation_epoch(Path(tmp) / "validation-epoch")
    print("StuPASE flow-matching smoke tests passed.")

pytestmark = [pytest.mark.unit, pytest.mark.external]
