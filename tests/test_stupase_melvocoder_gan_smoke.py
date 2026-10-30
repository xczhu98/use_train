"""StuPASE Mel-vocoder GAN smoke-test (StuP1: PASEGANTrainer + StuPASEMelVocoderGenerator).

Mirrors tests/test_pase_gan_smoke.py's pattern, verifying PASEGANTrainer
also drives StuPASE's unconditional clean-to-clean vocoder-autoencoder GAN
training correctly. Uses tiny stand-in generator/discriminator (not the real
Mel_feat/VocosVocoder stack) for speed, consistent with the other smoke
tests in this suite.
"""

from __future__ import annotations
import pytest

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from core.data.collate import collate_fn as _collate

from tasks.se.trainer.pase_gan_trainer import PASEGANTrainer


class _TinyCleanOnlyDataset(Dataset):
    """noisy == clean, matching StuP1's unconditional autoencoder training recipe."""

    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        t = torch.randn(512)
        return (t, t, 16000, torch.tensor(512))


class _TinyMelVocoderGenerator(nn.Module):
    """Stand-in for StuPASEMelVocoderGenerator: waveform -> fake-mel -> waveform."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(4, 1, kernel_size=3, padding=1)

    def forward(self, wav: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = wav.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        return self.conv2(x).squeeze(1)


class _TinySubDiscriminator(nn.Module):
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


class _TinyCombinedDiscriminator(nn.Module):
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


def test_stupase_melvocoder_gan_train_epoch(tmp_path):
    ds = _TinyCleanOnlyDataset()
    dl = DataLoader(ds, batch_size=2, collate_fn=_collate)

    gen = _TinyMelVocoderGenerator()
    disc = _TinyCombinedDiscriminator()
    exp_dir = tmp_path / "run"
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
                "exp_path": str(exp_dir),
                "epochs": 1,
                "discriminator_steps": 1,
                "generator_steps": 1,
                "adversarial_weight": 1.0,
                "lamda_adv": 1.0,   # StuP1 uses 1.0, not PASE's 2.0
                "lamda_feat": 1.0,
                "val_metrics": {"enabled": False},
            },
            "meta": {"allow_no_stft": True},
        },
        sample_rate=16000,
        stft_module=None,
    )

    loss = trainer._train_epoch(1)
    assert loss is not None and torch.isfinite(torch.tensor(loss)).item()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        test_stupase_melvocoder_gan_train_epoch(Path(tmp))
    print("StuPASE Mel-vocoder GAN smoke test passed.")

pytestmark = [pytest.mark.unit, pytest.mark.external]
