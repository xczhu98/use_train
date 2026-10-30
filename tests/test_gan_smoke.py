"""GANTrainer 最小 smoke-test。

验证 MetricGAN 风格的交替 G/D 训练循环在重构后仍能完成一个 epoch。
"""

from __future__ import annotations

import shutil
import tempfile

import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from core.data.collate import collate_fn as _collate


from core.losses.build import _StrictLossWrapper
from core.losses.composite import CompositeLoss
from core.losses.mbrcl import MultiBandRelativeContrastiveLoss
from core.stft import StftIstftModule
from tasks.se.trainer.gan_trainer import GANTrainer


class _TinyGANDataset(Dataset):
    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        t = torch.randn(512)
        return (t, t, 16000, torch.tensor(512))


class _TinyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class _TinyDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(
        self,
        cm: torch.Tensor,
        cp: torch.Tensor,
        em: torch.Tensor,
        ep: torch.Tensor,
    ) -> torch.Tensor:
        return (cm.mean(dim=(1, 2), keepdim=True) + em.mean(dim=(1, 2), keepdim=True)) * self.w


class _TinyLoss(nn.Module):
    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(enhanced, clean)


@pytest.mark.external
def test_gan_train_epoch():
    """写真实 exp_dir 的整轮 GAN smoke；仅显式 `-m external` 时运行。"""
    ds = _TinyGANDataset()
    dl = DataLoader(ds, batch_size=2, collate_fn=_collate)

    gen = _TinyGenerator()
    disc = _TinyDiscriminator()
    stft = StftIstftModule(n_fft=256, hop_length=128, win_length=256)
    exp_dir = tempfile.mkdtemp()
    try:
        trainer = GANTrainer(
            rank=0,
            world_size=1,
            model={"generator": gen, "discriminator": disc},
            optimizer={
                "generator": torch.optim.Adam(gen.parameters(), lr=1e-3),
                "discriminator": torch.optim.Adam(disc.parameters(), lr=1e-3),
            },
            scheduler=None,
            loss_fn=_TinyLoss(),
            train_loader=dl,
            val_loader=dl,
            config={
                "trainer": {
                    "exp_path": exp_dir,
                    "epochs": 1,
                    "discriminator_steps": 1,
                    "generator_steps": 1,
                    "adversarial_weight": 0.05,
                    "val_metrics": {"enabled": False},
                },
                "stft": {"n_fft": 256, "hop_length": 128},
            },
            sample_rate=16000,
            stft_module=stft,
        )
        loss = trainer._train_epoch(1)
        assert loss is not None and torch.isfinite(torch.tensor(loss)).item()
    finally:
        shutil.rmtree(exp_dir, ignore_errors=True)


def _new_mbrcl() -> MultiBandRelativeContrastiveLoss:
    return MultiBandRelativeContrastiveLoss(
        n_fft=64, hop_length=16, win_length=64, num_bands=4
    )


def _mbrcl_loss(kind: str):
    if kind == "single":
        return _new_mbrcl()
    if kind == "strict":
        return _StrictLossWrapper(_new_mbrcl())
    if kind == "composite":
        return CompositeLoss(
            components=[
                {
                    "type": "MultiBandRelativeContrastiveLoss",
                    "weight": 1.0,
                    "args": {
                        "n_fft": 64,
                        "hop_length": 16,
                        "win_length": 64,
                        "num_bands": 4,
                    },
                },
                {"type": "SISNRLoss", "weight": 0.1},
            ]
        )
    raise AssertionError(kind)


def _loss_trainer(loss_fn) -> GANTrainer:
    trainer = object.__new__(GANTrainer)
    trainer.sample_rate = 16000
    trainer.loss_fn = loss_fn
    return trainer


def _reconstruction_forward_dict() -> dict[str, torch.Tensor | int]:
    clean = torch.randn(2, 128)
    return {
        "enhanced": (0.8 * clean).requires_grad_(),
        "clean": clean,
        # 模拟增强后写回 forward_dict 的 noisy。
        "noisy": clean + 0.25,
        "lengths": torch.tensor([128, 112]),
        "fs": 24000,
        "is_pure_noise": torch.tensor([False, True]),
        "is_pure_echo": torch.zeros(2, dtype=torch.bool),
        "is_pure_echo_noise": torch.zeros(2, dtype=torch.bool),
    }


@pytest.mark.unit
@pytest.mark.parametrize("loss_kind", ["single", "strict", "composite"])
def test_gan_mbrcl_reconstruction_train_and_validation_use_forward_metadata(loss_kind):
    loss_fn = _mbrcl_loss(loss_kind)
    trainer = _loss_trainer(loss_fn)
    forward_dict = _reconstruction_forward_dict()
    enhanced = forward_dict["enhanced"]
    inner = getattr(loss_fn, "loss", loss_fn)
    mbrcl = inner.losses[0] if isinstance(inner, CompositeLoss) else inner
    mbrcl_calls: list[tuple[tuple, dict]] = []
    hooks = [
        mbrcl.register_forward_pre_hook(
            lambda _module, args, kwargs: mbrcl_calls.append((args, dict(kwargs))),
            with_kwargs=True,
        )
    ]
    sisnr_calls: list[tuple] = []
    if isinstance(inner, CompositeLoss):
        hooks.append(
            inner.losses[1].register_forward_pre_hook(
                lambda _module, args: sisnr_calls.append(args)
            )
        )

    train_loss = trainer._compute_reconstruction_loss(
        enhanced,
        forward_dict["clean"],
        forward_dict["lengths"],
        forward_dict=forward_dict,
    )
    val_loss = trainer._compute_validation_loss(forward_dict)
    for hook in hooks:
        hook.remove()

    assert torch.isfinite(train_loss) and torch.isfinite(val_loss)
    assert len(mbrcl_calls) == 2
    assert all(args[2] is forward_dict["lengths"] for args, _ in mbrcl_calls)
    first_args, first_kwargs = mbrcl_calls[0]
    assert first_args[0].shape[0] == 2
    assert first_kwargs["sample_rate"] == 24000
    assert torch.equal(first_kwargs["noisy"], forward_dict["noisy"])
    assert not {"is_pure_noise", "is_pure_echo", "is_pure_echo_noise"} & first_kwargs.keys()
    if loss_kind == "composite":
        assert sisnr_calls[0][0].shape[0] == 1
    assert 24000 in mbrcl._runtime_band_matrices


@pytest.mark.unit
def test_gan_zero_weight_mbrcl_and_plain_loss_receive_no_extra_kwargs():
    zero_weight = CompositeLoss(
        components=[
            {"type": "L1WavLoss", "weight": 1.0},
            {
                "type": "MultiBandRelativeContrastiveLoss",
                "weight": 0.0,
                "args": {
                    "n_fft": 64,
                    "hop_length": 16,
                    "win_length": 64,
                    "num_bands": 4,
                },
            },
        ]
    )
    forward_dict = _reconstruction_forward_dict()
    forward_dict.pop("noisy")
    forward_dict.pop("fs")
    forward_dict.pop("is_pure_noise")
    forward_dict.pop("is_pure_echo")
    forward_dict.pop("is_pure_echo_noise")
    assert torch.isfinite(_loss_trainer(zero_weight)._compute_validation_loss(forward_dict))

    trainer = _loss_trainer(_TinyLoss())
    assert trainer._reconstruction_loss_kwargs(forward_dict) == {}
    assert torch.isfinite(trainer._compute_validation_loss(forward_dict))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("missing_key", "match"),
    [
        ("noisy", r"requires forward_dict\['noisy'\]"),
        ("fs", r"requires forward_dict\['fs'\].*forward_dict\['sr'\]"),
    ],
)
def test_gan_mbrcl_missing_reconstruction_metadata_fails_fast(missing_key, match):
    forward_dict = _reconstruction_forward_dict()
    forward_dict.pop(missing_key)
    with pytest.raises(ValueError, match=match):
        _loss_trainer(_new_mbrcl())._compute_validation_loss(forward_dict)


pytestmark = [pytest.mark.usefixtures("preserve_tensorboard_module_state")]
