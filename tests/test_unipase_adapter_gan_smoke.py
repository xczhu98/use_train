"""UniPASE Stage 3 (Adapter GAN) smoke-test.

用 tiny WavLM（2 层 / 32 维）+ tiny VocosAdapter 驱动真实的
``UniPASEAdapterGenerator`` 与 ``UniPASEMultiScaleDiscriminator``，经
``UniPASEAdapterGANTrainer`` 跑 1 个 train epoch + 1 个 val epoch。验证：
  - generator 输出 (B,T,32) 特征，判别器工作在特征域；
  - noisy 带 20ms 零化丢包（PLC mask 路径）；
  - 重建项为 200*MSE(enh_feat, clean_L1_feat)（loss_fn 为 passthrough，
    实际重建损失由 trainer 内部计算）。

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

from core.losses.pase_family import ScalarPassthroughLoss
from tasks.se.models.pase_family.unipase.adapter_discriminators import (
    MultiScaleDiscriminator,
)
from tasks.se.models.pase_family.unipase.gan_wrappers import UniPASEAdapterGenerator
from tasks.se.trainer.unipase_adapter_gan_trainer import UniPASEAdapterGANTrainer

_TINY_WAVLM_CFG = dict(
    encoder_layers=2,
    encoder_embed_dim=32,
    encoder_ffn_embed_dim=64,
    encoder_attention_heads=2,
    layer_norm_first=True,
    gru_rel_pos=True,
    relative_position_embedding=True,
)

_TINY_ADAPTER_KW = dict(
    input_channels=32,
    dim=32,
    intermediate_dim=64,
    num_layers=2,
    num_res=2,
    num_attn=1,
    output_channels=32,
)

_N_SAMPLES = 6400  # 0.4s @16k


def _make_tiny_wavlm_ckpt(path: Path) -> None:
    from tasks.se.models.pase_family.common.wavlm.WavLM import WavLM, WavLMConfig

    torch.manual_seed(0)
    model = WavLM(WavLMConfig(dict(_TINY_WAVLM_CFG)))
    torch.save({"cfg": dict(_TINY_WAVLM_CFG), "model": model.state_dict()}, path)


class _TinyNoisyCleanDataset(Dataset):
    """noisy = clean + 噪声 + 20ms 零化丢包（触发 PLC mask 路径）。"""

    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        torch.manual_seed(200 + idx)
        clean = torch.randn(_N_SAMPLES) * 0.3
        noisy = clean + torch.randn(_N_SAMPLES) * 0.05
        packet = 320  # 20ms @16k
        noisy[2 * packet : 3 * packet] = 0.0
        return (clean, noisy, 16000, torch.tensor(_N_SAMPLES))


def _build_trainer(exp_dir: str, ckpt: Path) -> UniPASEAdapterGANTrainer:
    gen = UniPASEAdapterGenerator(
        encoder_ckpt_path=str(ckpt), enable_plc=True, **_TINY_ADAPTER_KW
    )
    #  tiny 模型只有 2 层 transformer：官方 [1, 24] -> [1, 2]
    gen.encoder.output_layer = [1, 2]
    disc = MultiScaleDiscriminator(in_dim=32, sub_dims=[8, 16])
    ds = _TinyNoisyCleanDataset()
    dl = DataLoader(ds, batch_size=2, collate_fn=_collate)
    return UniPASEAdapterGANTrainer(
        rank=0,
        world_size=1,
        model={"generator": gen, "discriminator": disc},
        optimizer={
            "generator": torch.optim.Adam(gen.parameters(), lr=1e-3),
            "discriminator": torch.optim.Adam(disc.parameters(), lr=1e-3),
        },
        scheduler=None,
        loss_fn=ScalarPassthroughLoss(),
        train_loader=dl,
        val_loader=dl,
        config={
            "trainer": {
                "exp_path": exp_dir,
                "epochs": 1,
                "discriminator_steps": 1,
                "generator_steps": 1,
                "adversarial_weight": 1.0,
                "lamda_recons": 200.0,  # 官方 cfg_train_adapter.yaml coeff.recons
                "lamda_adv": 1.0,       # 官方 coeff.adv
                "lamda_feat": 1.0,      # 官方 coeff.feat
                "val_metrics": {"enabled": False},
            },
            "meta": {"allow_no_stft": True},
        },
        sample_rate=16000,
        stft_module=None,
    )


def test_unipase_adapter_generator_forward_shapes():
    """wrapper 输出 (B,T,D) 增强特征 + clean L1 目标特征。"""
    tmpdir = Path(tempfile.mkdtemp())
    try:
        ckpt = tmpdir / "tiny_wavlm.pt"
        _make_tiny_wavlm_ckpt(ckpt)
        gen = UniPASEAdapterGenerator(
            encoder_ckpt_path=str(ckpt), enable_plc=True, **_TINY_ADAPTER_KW
        )
        gen.encoder.output_layer = [1, 2]
        noisy = torch.randn(2, _N_SAMPLES) * 0.3
        noisy[:, 640:960] = 0.0  # 20ms 丢包
        clean = torch.randn(2, _N_SAMPLES) * 0.3
        enh_feat, target_feat = gen(noisy, clean, None)
        assert enh_feat.ndim == 3 and enh_feat.shape[-1] == 32
        assert enh_feat.shape == target_feat.shape
        assert not target_feat.requires_grad, "target feat 必须在 no_grad 下计算"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_unipase_adapter_gan_train_epoch():
    tmpdir = Path(tempfile.mkdtemp())
    try:
        ckpt = tmpdir / "tiny_wavlm.pt"
        _make_tiny_wavlm_ckpt(ckpt)
        trainer = _build_trainer(str(tmpdir / "exp"), ckpt)

        assert all(not p.requires_grad for p in trainer.generator.encoder.parameters())
        adp_param = next(trainer.generator.adapter.parameters())
        before = adp_param.detach().clone()

        loss = trainer._train_epoch(1)
        assert loss is not None and torch.isfinite(torch.tensor(loss)).item()
        assert not torch.allclose(before, adp_param.detach()), "adapter did not update"

        val_loss = trainer._validation_epoch(1)
        assert val_loss is not None and torch.isfinite(torch.tensor(val_loss)).item()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


pytestmark = [pytest.mark.external]
