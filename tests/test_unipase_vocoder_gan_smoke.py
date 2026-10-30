"""UniPASE Stage 2 (Vocoder GAN) smoke-test.

用 tiny WavLM（2 层 / 32 维）+ tiny VocosVocoder 驱动真实的
``UniPASEVocoderGenerator`` 与 ``PASEDiscriminator``，经 ``PASEGANTrainer``
跑 1 个 train epoch + 1 个 val epoch（clean-only，noisy == clean，对齐官方
train_vocoder.py 的无条件自编码配方）。

无需官方仓库 / 真实 DeWavLM-Omni，CI 可直接运行。
"""

from __future__ import annotations
import pytest

import shutil
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from core.data.collate import collate_fn as _collate

from core.losses.pase_family import PaseMultiScaleMelLoss
from tasks.se.models.pase_family.pase import PASEDiscriminator
from tasks.se.models.pase_family.unipase.gan_wrappers import UniPASEVocoderGenerator
from tasks.se.trainer.pase_gan_trainer import PASEGANTrainer

_TINY_WAVLM_CFG = dict(
    encoder_layers=2,
    encoder_embed_dim=32,
    encoder_ffn_embed_dim=64,
    encoder_attention_heads=2,
    layer_norm_first=True,
    gru_rel_pos=True,
    relative_position_embedding=True,
)

_TINY_VOCODER_KW = dict(
    input_channels=32,
    dim=32,
    intermediate_dim=64,
    num_layers=2,
    num_res=2,
    num_attn=1,
    n_fft=1280,
    hop_length=320,
)

_N_SAMPLES = 6400  # 0.4s @16k


def _make_tiny_wavlm_ckpt(path: Path) -> None:
    from tasks.se.models.pase_family.common.wavlm.WavLM import WavLM, WavLMConfig

    torch.manual_seed(0)
    model = WavLM(WavLMConfig(dict(_TINY_WAVLM_CFG)))
    torch.save({"cfg": dict(_TINY_WAVLM_CFG), "model": model.state_dict()}, path)


class _TinyCleanOnlyDataset(Dataset):
    """noisy == clean，对齐官方 dataloader_clean.py 的 clean-only 配方。"""

    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        torch.manual_seed(100 + idx)
        t = torch.randn(_N_SAMPLES) * 0.3
        return (t, t, 16000, torch.tensor(_N_SAMPLES))


def _build_trainer(exp_dir: str, ckpt: Path) -> PASEGANTrainer:
    gen = UniPASEVocoderGenerator(encoder_ckpt_path=str(ckpt), **_TINY_VOCODER_KW)
    disc = PASEDiscriminator(mpd_reshapes=[2, 3], mbd_fft_sizes=[512, 256])
    ds = _TinyCleanOnlyDataset()
    dl = DataLoader(ds, batch_size=2, collate_fn=_collate)
    loss_fn = PaseMultiScaleMelLoss(
        sampling_rate=16000,
        n_mels=[10, 20, 40, 80],
        window_lengths=[128, 256, 512, 1024],
        weight=15.0,  # 官方 coeff.recons
    )
    return PASEGANTrainer(
        rank=0,
        world_size=1,
        model={"generator": gen, "discriminator": disc},
        optimizer={
            "generator": torch.optim.Adam(gen.parameters(), lr=1e-3),
            "discriminator": torch.optim.Adam(disc.parameters(), lr=1e-3),
        },
        scheduler=None,
        loss_fn=loss_fn,
        train_loader=dl,
        val_loader=dl,
        config={
            "trainer": {
                "exp_path": exp_dir,
                "epochs": 1,
                "discriminator_steps": 1,
                "generator_steps": 1,
                "adversarial_weight": 1.0,
                "lamda_adv": 0.5,   # 官方 cfg_train_vocoder.yaml coeff.adv
                "lamda_feat": 0.5,  # 官方 coeff.feat
                "val_metrics": {"enabled": False},
            },
            "meta": {"allow_no_stft": True},
        },
        sample_rate=16000,
        stft_module=None,
    )


def test_unipase_vocoder_gan_train_epoch():
    tmpdir = Path(tempfile.mkdtemp())
    try:
        ckpt = tmpdir / "tiny_wavlm.pt"
        _make_tiny_wavlm_ckpt(ckpt)
        trainer = _build_trainer(str(tmpdir / "exp"), ckpt)

        # encoder 必须冻结、vocoder 可训练
        assert all(not p.requires_grad for p in trainer.generator.encoder.parameters())
        voc_param = next(trainer.generator.vocoder.parameters())
        before = voc_param.detach().clone()

        loss = trainer._train_epoch(1)
        assert loss is not None and torch.isfinite(torch.tensor(loss)).item()
        assert not torch.allclose(before, voc_param.detach()), "vocoder did not update"

        val_loss = trainer._validation_epoch(1)
        assert val_loss is not None and torch.isfinite(torch.tensor(val_loss)).item()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


pytestmark = [pytest.mark.external]
