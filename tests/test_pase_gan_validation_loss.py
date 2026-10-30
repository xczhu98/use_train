"""PASE-family GAN validation loss 合同测试。"""

from __future__ import annotations

import pytest
import torch

from tasks.se.trainer.pase_gan_trainer import PASEGANTrainer
from tasks.se.trainer.unipase_adapter_gan_trainer import UniPASEAdapterGANTrainer

pytestmark = [pytest.mark.unit]


def _forward_dict() -> dict[str, torch.Tensor]:
    return {
        "enhanced": torch.zeros(1, 8),
        "clean": torch.zeros(1, 8),
        "lengths": torch.tensor([8]),
    }


def test_pase_gan_validation_includes_adv_and_feature_matching(monkeypatch):
    trainer = PASEGANTrainer.__new__(PASEGANTrainer)
    trainer.adversarial_weight = 1.0
    trainer._compute_reconstruction_loss = lambda *args, **kwargs: torch.tensor(2.0)
    trainer._extract_gan_mag_pha = lambda fwd: (None, None, None, None)
    trainer._compute_generator_adversarial_loss = lambda *args: torch.tensor(3.0)

    loss = trainer._compute_validation_loss(_forward_dict())

    assert float(loss) == pytest.approx(5.0)


def test_unipase_adapter_validation_reuses_pase_gan_formula():
    trainer = UniPASEAdapterGANTrainer.__new__(UniPASEAdapterGANTrainer)
    trainer.adversarial_weight = 1.0
    trainer.lamda_recons = 200.0
    trainer._extract_gan_mag_pha = lambda fwd: (None, None, None, None)
    trainer._compute_generator_adversarial_loss = lambda *args: torch.tensor(1.5)
    forward = _forward_dict()
    forward["target_feat"] = torch.ones(1, 8)

    loss = trainer._compute_validation_loss(forward)

    assert float(loss) == pytest.approx(200.0 + 1.5)


def test_load_official_gan_warm_start_restores_g_d_and_optimizers(tmp_path):
    import torch.nn as nn

    from tasks.se.trainer.pase_gan_trainer import load_official_gan_warm_start

    gen = nn.Linear(2, 2)
    disc = nn.Linear(2, 2)
    opt_g = torch.optim.AdamW(gen.parameters(), lr=1e-3)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-3)
    gen(torch.ones(1, 2)).sum().backward()
    opt_g.step()
    disc(torch.ones(1, 2)).sum().backward()
    opt_d.step()
    ckpt_path = tmp_path / "model_350.tar"
    torch.save(
        {
            "model": gen.state_dict(),
            "discriminator": disc.state_dict(),
            "optimizer_g": opt_g.state_dict(),
            "optimizer_d": opt_d.state_dict(),
        },
        ckpt_path,
    )

    gen2 = nn.Linear(2, 2)
    disc2 = nn.Linear(2, 2)
    opt_g2 = torch.optim.AdamW(gen2.parameters(), lr=1e-3)
    opt_d2 = torch.optim.AdamW(disc2.parameters(), lr=1e-3)
    load_official_gan_warm_start(gen2, disc2, opt_g2, opt_d2, str(ckpt_path))

    for key, value in gen.state_dict().items():
        torch.testing.assert_close(gen2.state_dict()[key], value)
    for key, value in disc.state_dict().items():
        torch.testing.assert_close(disc2.state_dict()[key], value)
    assert len(opt_g2.state_dict()["state"]) == len(opt_g.state_dict()["state"])
    assert len(opt_d2.state_dict()["state"]) == len(opt_d.state_dict()["state"])


def test_load_official_gan_warm_start_rejects_missing_keys(tmp_path):
    import torch.nn as nn

    from tasks.se.trainer.pase_gan_trainer import load_official_gan_warm_start

    ckpt_path = tmp_path / "bad.tar"
    torch.save({"model": {}}, ckpt_path)
    gen = nn.Linear(2, 2)
    disc = nn.Linear(2, 2)
    with pytest.raises(KeyError, match="missing keys"):
        load_official_gan_warm_start(
            gen,
            disc,
            torch.optim.AdamW(gen.parameters()),
            torch.optim.AdamW(disc.parameters()),
            str(ckpt_path),
        )
