"""Tests for SE trainer loss metadata kwargs assembly.

``is_pure_*`` 只向 CompositeLoss 注入；noisy 与 runtime sample rate 则按活动
component capability 注入，且普通 loss 不接收额外 kwargs。
"""
import pytest

import torch
import torch.nn as nn

from core.losses.composite import CompositeLoss
from core.losses.mbrcl import MultiBandRelativeContrastiveLoss
from core.utils.registry import LOSS_REGISTRY
from tasks.se.data.batch import SEBatch
from tasks.se.trainer.metrics_trainer import _composite_pure_flag_kwargs, _se_loss_kwargs


@LOSS_REGISTRY.register()
class _DummySEKwargsLoss(nn.Module):
    """Dummy loss that accepts **kwargs (like PESQ/STOI-aware components)."""

    def forward(self, enhanced, clean, lengths, **kwargs):
        return torch.zeros((), device=enhanced.device, dtype=enhanced.dtype)


class _FakeWrappedLoss:
    """Dummy wrapper exposing the inner loss via ``loss`` attribute."""

    def __init__(self, inner):
        self.loss = inner


class _FakeNonCompositeLoss:
    """A loss that is neither CompositeLoss nor wrapped."""


def _make_composite() -> CompositeLoss:
    return CompositeLoss(components=[{"type": "_DummySEKwargsLoss", "weight": 1.0}])


def _make_batch(with_flags: bool = True) -> SEBatch:
    kwargs = {}
    if with_flags:
        kwargs = {
            "is_pure_noise": torch.tensor([True, False]),
            "is_pure_echo": torch.tensor([False, True]),
            "is_pure_echo_noise": torch.tensor([False, False]),
        }
    return SEBatch(
        clean=torch.randn(2, 1600),
        noisy=torch.randn(2, 1600),
        sr=torch.tensor([16000, 16000]),
        lengths=torch.tensor([1600, 1600], dtype=torch.int32),
        **kwargs,
    )


def test_pure_flags_forwarded_for_composite():
    out = _composite_pure_flag_kwargs(_make_composite(), _make_batch(), torch.device("cpu"))
    assert out["is_pure_noise"].tolist() == [True, False]
    assert out["is_pure_echo"].tolist() == [False, True]
    assert out["is_pure_echo_noise"].tolist() == [False, False]


def test_pure_flags_forwarded_for_wrapped_composite():
    loss_fn = _FakeWrappedLoss(_make_composite())
    out = _composite_pure_flag_kwargs(loss_fn, _make_batch(), torch.device("cpu"))
    assert out["is_pure_noise"].tolist() == [True, False]


def test_non_composite_loss_gets_no_flags():
    out = _composite_pure_flag_kwargs(
        _FakeNonCompositeLoss(), _make_batch(), torch.device("cpu")
    )
    assert out == {}


def test_missing_flags_return_empty():
    out = _composite_pure_flag_kwargs(
        _make_composite(), _make_batch(with_flags=False), torch.device("cpu")
    )
    assert out == {}


def test_flags_moved_to_requested_device():
    batch = _make_batch()
    out = _composite_pure_flag_kwargs(_make_composite(), batch, torch.device("cpu"))
    for v in out.values():
        assert v.device == torch.device("cpu")


def _new_mbrcl() -> MultiBandRelativeContrastiveLoss:
    return MultiBandRelativeContrastiveLoss(
        n_fft=64, hop_length=16, win_length=64, num_bands=4
    )


def test_single_mbrcl_receives_runtime_context_without_pure_flags():
    loss_fn = _new_mbrcl()
    batch = _make_batch()
    enhanced = (0.8 * batch.clean).requires_grad_()

    kwargs = _se_loss_kwargs(loss_fn, batch, enhanced, 24000)

    assert set(kwargs) == {"noisy", "sample_rate"}
    assert torch.equal(kwargs["noisy"], batch.noisy)
    assert kwargs["sample_rate"] == 24000
    loss = loss_fn(enhanced, batch.clean, batch.lengths, **kwargs)
    assert torch.isfinite(loss)
    assert 24000 in loss_fn._runtime_band_matrices


def test_composite_mbrcl_keeps_full_batch_while_sisnr_uses_speech_subset():
    loss_fn = CompositeLoss(
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
    batch = _make_batch()
    batch.is_pure_echo.zero_()
    enhanced = (0.8 * batch.clean).requires_grad_()
    mbrcl_calls: list[tuple[tuple, dict]] = []
    sisnr_calls: list[tuple] = []
    mbrcl_hook = loss_fn.losses[0].register_forward_pre_hook(
        lambda _module, args, kwargs: mbrcl_calls.append((args, dict(kwargs))),
        with_kwargs=True,
    )
    sisnr_hook = loss_fn.losses[1].register_forward_pre_hook(
        lambda _module, args: sisnr_calls.append(args)
    )

    loss = loss_fn(
        enhanced,
        batch.clean,
        batch.lengths,
        **_se_loss_kwargs(loss_fn, batch, enhanced, 24000),
    )
    mbrcl_hook.remove()
    sisnr_hook.remove()

    assert torch.isfinite(loss)
    mbrcl_args, mbrcl_kwargs = mbrcl_calls[0]
    assert mbrcl_args[0].shape[0] == 2
    assert torch.equal(mbrcl_kwargs["noisy"], batch.noisy)
    assert mbrcl_kwargs["sample_rate"] == 24000
    assert sisnr_calls[0][0].shape[0] == 1


pytestmark = [pytest.mark.unit]
