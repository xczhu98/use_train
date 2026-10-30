"""PASEGANTrainer smoke-test.

Verifies the waveform-domain MPD+MBD GAN training loop (PASEGANTrainer)
completes one epoch with finite G/D loss and updates generator parameters.
Uses tiny stand-in generator/discriminator modules (matching PASE/
PASEDiscriminator's forward signatures) rather than the real WavLM/Vocos
stack, mirroring tests/test_gan_smoke.py's approach -- this test targets the
new PASEGANTrainer plumbing, not WavLM/Vocos numerical fidelity (covered
separately by tests/test_pase_wavlm_parity.py and the max_steps integration
run in the migration plan's verification checklist).
"""

from __future__ import annotations
import pytest

import shutil
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from core.data.collate import collate_fn as _collate

from tasks.se.trainer.pase_gan_trainer import PASEGANTrainer


class _TinyPASEGeneratorDataset(Dataset):
    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        t = torch.randn(512)
        return (t, t, 16000, torch.tensor(512))


class _TinyPASEGenerator(nn.Module):
    """Waveform-in/waveform-out stand-in for PASE (encoder+decoder collapsed to one conv)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)

    def forward(self, noisy: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.conv(noisy.unsqueeze(1)).squeeze(1)


class _TinySubDiscriminator(nn.Module):
    """One MPD/MBD-style sub-discriminator: waveform -> (score, [feature_maps])."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(4, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        f1 = torch.relu(self.conv1(x))
        out = self.conv2(f1)
        return out.flatten(1), [f1, out]


class _TinyPASEDiscriminator(nn.Module):
    """Stand-in for PASEDiscriminator: two sub-discriminators, same return shape as CombinedDiscriminator."""

    def __init__(self) -> None:
        super().__init__()
        self.discs = nn.ModuleList([_TinySubDiscriminator(), _TinySubDiscriminator()])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discs:
            yr, fr = d(y)
            yg, fg = d(y_hat)
            y_d_rs.append(yr)
            fmap_rs.append(fr)
            y_d_gs.append(yg)
            fmap_gs.append(fg)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class _TinyReconLoss(nn.Module):
    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.nn.functional.mse_loss(enhanced, clean)


def test_pase_gan_train_epoch():
    ds = _TinyPASEGeneratorDataset()
    dl = DataLoader(ds, batch_size=2, collate_fn=_collate)

    gen = _TinyPASEGenerator()
    disc = _TinyPASEDiscriminator()
    exp_dir = tempfile.mkdtemp()
    try:
        trainer = PASEGANTrainer(
            rank=0,
            world_size=1,
            model={"generator": gen, "discriminator": disc},
            optimizer={
                "generator": torch.optim.Adam(gen.parameters(), lr=1e-3),
                "discriminator": torch.optim.Adam(disc.parameters(), lr=1e-3),
            },
            scheduler=None,
            loss_fn=_TinyReconLoss(),
            train_loader=dl,
            val_loader=dl,
            config={
                "trainer": {
                    "exp_path": exp_dir,
                    "epochs": 1,
                    "discriminator_steps": 1,
                    "generator_steps": 1,
                    "adversarial_weight": 1.0,
                    "lamda_adv": 2.0,
                    "lamda_feat": 1.0,
                    "val_metrics": {"enabled": False},
                },
                "meta": {"allow_no_stft": True},
            },
            sample_rate=16000,
            stft_module=None,
        )

        param_before = next(trainer.generator.parameters()).detach().clone()
        loss = trainer._train_epoch(1)
        assert loss is not None and torch.isfinite(torch.tensor(loss)).item()
        param_after = next(trainer.generator.parameters()).detach().clone()
        assert not torch.allclose(param_before, param_after), "generator parameters did not update"
    finally:
        shutil.rmtree(exp_dir, ignore_errors=True)


pytestmark = [pytest.mark.external]
