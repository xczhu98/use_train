"""SSTrainer 真实构造与最小前后向合同。"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from core.losses.pit import PITLoss
from tasks.ss.data import ss_collate_fn
from tasks.ss.data.config import build_ss_dataset_contract
from tasks.ss.trainer.ss_trainer import SSTrainer


class _TinySSDataset(Dataset):
    def __init__(self) -> None:
        self.contract = build_ss_dataset_contract(sample_rate=8000, num_sources=2)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        del index
        mixture = torch.linspace(-0.75, 0.75, 32)
        return {
            "uid": "tiny",
            "mixture": mixture,
            "sources": [mixture * 0.5, -mixture * 0.25],
            "length": mixture.numel(),
            "sr": 8000,
            "source_ids": ["speaker-a", "speaker-b"],
        }


class _TinySeparator(torch.nn.Module):
    input_domain = "waveform"
    output_domain = "waveform"
    sample_rate = 8000

    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, mixture, lengths, *, fs=None):
        del lengths, fs
        first = mixture * self.gain
        return {"est_sources": torch.stack((first, -first), dim=1)}


def _build_trainer(tmp_path) -> SSTrainer:
    dataset = _TinySSDataset()
    loader = DataLoader(dataset, batch_size=1, collate_fn=ss_collate_fn)
    model = _TinySeparator()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = {
        "meta": {"seed": 3, "use_amp": False, "use_ema": False},
        "model": {"type": "_TinySeparator", "args": {}},
        "data": {"mode": "offline_ss", "sample_rate": 8000},
        "ss": {"num_sources": 2},
        "trainer": {
            "epochs": 1,
            "exp_path": str(tmp_path),
            "validation_epoch_interval": 0,
            "validation_steps_interval": 0,
            "save_top_k": 0,
            "strict_checks": True,
        },
    }
    return SSTrainer(
        rank=0,
        world_size=1,
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0),
        loss_fn=PITLoss(pairwise_loss="PairwiseL1Loss"),
        config=config,
        train_loader=loader,
        val_loader=loader,
        train_dataset=dataset,
        sample_rate=8000,
    )


@pytest.mark.unit
def test_ss_trainer_minimal_prepare_forward_loss_backward_smoke(tmp_path):
    trainer = _build_trainer(tmp_path)
    raw_batch = next(iter(trainer.train_loader))
    batch, fs_val, placeholders = trainer._prepare_batch(raw_batch, 1, 0)
    loss, finite, info = trainer._forward_and_compute_loss(batch, 1, 0, fs_val)

    assert placeholders == 0
    assert fs_val == 8000
    assert finite is True
    assert info["enhanced_safe"].abs().max() > 1.0
    loss.backward()
    assert trainer._raw_model().gain.grad is not None
    assert torch.isfinite(trainer._raw_model().gain.grad)


@pytest.mark.unit
def test_ss_trainer_validation_uses_prepare_and_forward_phases(tmp_path):
    trainer = _build_trainer(tmp_path)
    raw_batch = next(iter(trainer.val_loader))

    prepared = trainer._val_prepare_batch(raw_batch, epoch=2, n_loops=1)
    phased = trainer._val_forward_prepared(prepared, epoch=2, n_loops=1)
    legacy = trainer._val_forward_loss(raw_batch, epoch=2, n_loops=1)

    assert prepared[0].mixture.shape == (1, 32)
    assert prepared[1] == 8000
    assert torch.allclose(phased[0], legacy[0])
    assert torch.equal(phased[1], legacy[1])


@pytest.mark.unit
def test_ss_trainer_rejects_constructor_sample_rate_mismatch(tmp_path):
    dataset = _TinySSDataset()
    loader = DataLoader(dataset, batch_size=1, collate_fn=ss_collate_fn)
    model = _TinySeparator()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = {
        "meta": {"use_amp": False, "use_ema": False},
        "model": {"type": "_TinySeparator", "args": {}},
        "data": {"mode": "offline_ss", "sample_rate": 8000},
        "ss": {"num_sources": 2},
        "trainer": {
            "exp_path": str(tmp_path),
            "validation_epoch_interval": 0,
            "save_top_k": 0,
        },
    }
    with pytest.raises(ValueError, match="SSTrainer sample_rate"):
        SSTrainer(
            rank=0,
            world_size=1,
            model=model,
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0),
            loss_fn=PITLoss(pairwise_loss="PairwiseL1Loss"),
            config=config,
            train_loader=loader,
            val_loader=loader,
            train_dataset=dataset,
            sample_rate=16000,
        )
