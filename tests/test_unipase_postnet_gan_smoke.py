"""UniPASE Stage 4 (PostNet GAN, 48kHz) smoke-test.

用 tiny WavLM（2 层 / 32 维）+ tiny VocosAdapter/VocosVocoder（随机权重经
from_pretrained 加载，走真实 ckpt 路径）+ tiny TFGridNet 驱动真实的
``UniPASEPostNetGenerator``，经 ``PASEGANTrainer`` 在 48k 数据上跑
1 个 train epoch + 1 个 val epoch。验证：
  - 冻结 16k 链路 (encoder/adapter/vocoder) requires_grad 全 False；
  - TFGridNet.forward（训练模式全时域残差，非 infer()）输出 48k 波形；
  - mel/adv/feat 损失在 48k 波形上有限且 PostNet 参数更新。

无需官方仓库 / 真实 UniPASE 权重，CI 可直接运行。
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
from tasks.se.models.pase_family.unipase.adapter import VocosAdapter
from tasks.se.models.pase_family.unipase.gan_wrappers import UniPASEPostNetGenerator
from tasks.se.models.pase_family.unipase.vocoder import VocosVocoder
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

_TINY_ADAPTER_KW = dict(
    input_channels=32,
    dim=32,
    intermediate_dim=64,
    num_layers=2,
    num_res=2,
    num_attn=1,
    output_channels=32,
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

_TINY_POSTNET_KW = dict(
    n_imics=1,
    n_layers=1,
    lstm_hidden_units=8,
    attn_n_head=2,
    attn_qk_output_channel=2,
    emb_dim=8,
)

_SR = 48000
_N_SAMPLES = 9600  # 0.2s @48k


def _make_tiny_wavlm_ckpt(path: Path) -> None:
    from tasks.se.models.pase_family.common.wavlm.WavLM import WavLM, WavLMConfig

    torch.manual_seed(0)
    model = WavLM(WavLMConfig(dict(_TINY_WAVLM_CFG)))
    torch.save({"cfg": dict(_TINY_WAVLM_CFG), "model": model.state_dict()}, path)


def _make_tiny_module_ckpt(path: Path, module: torch.nn.Module, cfg: dict) -> None:
    """按 from_pretrained 的 {"cfg", "generator"} 约定落盘随机权重 ckpt。"""
    torch.save({"cfg": dict(cfg), "generator": module.state_dict()}, path)


class _TinyNoisyCleanDataset48k(Dataset):
    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        torch.manual_seed(300 + idx)
        clean = torch.randn(_N_SAMPLES) * 0.3
        noisy = clean + torch.randn(_N_SAMPLES) * 0.05
        packet = 960  # 20ms @48k
        noisy[2 * packet : 3 * packet] = 0.0
        return (clean, noisy, _SR, torch.tensor(_N_SAMPLES))


def _build_generator(tmpdir: Path) -> UniPASEPostNetGenerator:
    wavlm_ckpt = tmpdir / "tiny_wavlm.pt"
    _make_tiny_wavlm_ckpt(wavlm_ckpt)
    adapter_ckpt = tmpdir / "tiny_adapter.pt"
    _make_tiny_module_ckpt(adapter_ckpt, VocosAdapter(**_TINY_ADAPTER_KW), _TINY_ADAPTER_KW)
    vocoder_ckpt = tmpdir / "tiny_vocoder.pt"
    _make_tiny_module_ckpt(vocoder_ckpt, VocosVocoder(**_TINY_VOCODER_KW), _TINY_VOCODER_KW)

    gen = UniPASEPostNetGenerator(
        encoder_ckpt_path=str(wavlm_ckpt),
        adapter_ckpt_path=str(adapter_ckpt),
        vocoder_ckpt_path=str(vocoder_ckpt),
        sr_in=_SR,
        enable_plc=True,
        postnet_kwargs=dict(_TINY_POSTNET_KW),
    )
    # tiny 模型只有 2 层 transformer：官方 [1, 24] -> [1, 2]
    gen.encoder.output_layer = [1, 2]
    return gen


def test_unipase_postnet_generator_forward_shape():
    """48k in -> 48k out，冻结链路无梯度，postnet 输出保留梯度。"""
    tmpdir = Path(tempfile.mkdtemp())
    try:
        gen = _build_generator(tmpdir)
        noisy = torch.randn(2, _N_SAMPLES) * 0.3
        noisy[:, 1920:2880] = 0.0  # 20ms 丢包 @48k
        out = gen(noisy, None)
        assert out.shape == (2, _N_SAMPLES), out.shape
        assert out.requires_grad, "postnet 输出必须带梯度（训练模式 forward）"
        assert all(not p.requires_grad for p in gen.encoder.parameters())
        assert all(not p.requires_grad for p in gen.adapter.parameters())
        assert all(not p.requires_grad for p in gen.decoder.parameters())
        assert any(p.requires_grad for p in gen.postnet.parameters())
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_unipase_postnet_gan_train_epoch():
    tmpdir = Path(tempfile.mkdtemp())
    try:
        gen = _build_generator(tmpdir)
        disc = PASEDiscriminator(mpd_reshapes=[2, 3], mbd_fft_sizes=[512, 256])
        ds = _TinyNoisyCleanDataset48k()
        dl = DataLoader(ds, batch_size=2, collate_fn=_collate)
        loss_fn = PaseMultiScaleMelLoss(
            sampling_rate=_SR,
            n_mels=[10, 20, 40, 80],
            window_lengths=[128, 256, 512, 1024],
            weight=15.0,  # 官方 coeff.recons
        )
        trainer = PASEGANTrainer(
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
                    "exp_path": str(tmpdir / "exp"),
                    "epochs": 1,
                    "discriminator_steps": 1,
                    "generator_steps": 1,
                    "adversarial_weight": 1.0,
                    "lamda_adv": 0.5,   # 官方 cfg_train_postnet.yaml coeff.adv
                    "lamda_feat": 0.5,  # 官方 coeff.feat
                    "val_metrics": {"enabled": False},
                },
                "meta": {"allow_no_stft": True},
            },
            sample_rate=_SR,
            stft_module=None,
        )

        pn_param = next(trainer.generator.postnet.parameters())
        before = pn_param.detach().clone()
        loss = trainer._train_epoch(1)
        assert loss is not None and torch.isfinite(torch.tensor(loss)).item()
        assert not torch.allclose(before, pn_param.detach()), "postnet did not update"

        val_loss = trainer._validation_epoch(1)
        assert val_loss is not None and torch.isfinite(torch.tensor(val_loss)).item()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


pytestmark = [pytest.mark.external]
