"""Tests for AECTrainer CompositeLoss kwargs assembly (issue 5 fix).

Ensures:
- pure-branch flags are forwarded only when the inner loss is a CompositeLoss.
- state / state_probs are forwarded only when a StateCrossEntropyLoss component is
  present; otherwise they are kept out of loss kwargs.
- undesired_ref / undesired_enhanced are forwarded only when an effective-SS
  component or FWNegSISDRLoss has non-zero weight.
"""
import pytest

import torch
import torch.nn as nn

from tasks.aec.trainer.aec_trainer import (
    _aec_loss_kwargs,
    _composite_ss_undesired_kwargs,
)
from core.losses.composite import CompositeLoss
from core.losses.mbrcl import MultiBandRelativeContrastiveLoss  # noqa: F401
from core.trainer.metrics_trainer import (
    _loss_has_active_capability,
    _waveform_loss_kwargs,
)
from core.utils.registry import LOSS_REGISTRY

_TEST_LOSS_NAMES = ("_DummyKwargsLoss",)


@pytest.fixture(autouse=True, scope="module")
def _unregister_test_losses():
    """模块结束后注销本文件注册的临时 loss，避免污染全局 LOSS_REGISTRY。

    ``Registry`` 目前没有公开的注销 API，测试直接操作内部映射。
    """
    yield
    for name in _TEST_LOSS_NAMES:
        LOSS_REGISTRY._registry.pop(name, None)


@LOSS_REGISTRY.register()
class _DummyKwargsLoss(nn.Module):
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
    return CompositeLoss(components=[{"type": "_DummyKwargsLoss", "weight": 1.0}])


def test_pure_flags_forwarded_for_composite():
    noisy = torch.randn(2, 16000)
    clean = torch.randn(2, 16000)
    enhanced = torch.randn(2, 16000)
    data = {
        "is_pure_noise": torch.tensor([True, False]),
        "is_pure_echo": torch.tensor([False, True]),
        "is_pure_echo_noise": torch.tensor([False, False]),
        "state": torch.zeros(2, 10, dtype=torch.long),
        "state_probs": torch.randn(2, 10, 8),
    }

    loss_fn = _make_composite()
    out = _composite_ss_undesired_kwargs(loss_fn, noisy, clean, enhanced, data)

    # No SS kwargs because the dummy CompositeLoss has no SS-enabled component.
    assert "undesired_ref" not in out
    assert "undesired_enhanced" not in out
    assert out["is_pure_noise"].tolist() == [True, False]
    assert out["is_pure_echo"].tolist() == [False, True]
    assert out["is_pure_echo_noise"].tolist() == [False, False]
    assert "state" not in out
    assert "state_probs" not in out


def test_pure_flags_forwarded_for_wrapped_composite():
    noisy = torch.randn(2, 16000)
    clean = torch.randn(2, 16000)
    enhanced = torch.randn(2, 16000)
    data = {
        "is_pure_noise": torch.tensor([True, True]),
        "state": torch.zeros(2, 10, dtype=torch.long),
    }

    loss_fn = _FakeWrappedLoss(_make_composite())
    out = _composite_ss_undesired_kwargs(loss_fn, noisy, clean, enhanced, data)
    assert out["is_pure_noise"].tolist() == [True, True]
    assert "state" not in out


def test_non_composite_loss_gets_no_flags():
    noisy = torch.randn(2, 16000)
    clean = torch.randn(2, 16000)
    enhanced = torch.randn(2, 16000)
    data = {
        "is_pure_noise": torch.tensor([True, False]),
        "state": torch.zeros(2, 10, dtype=torch.long),
    }
    loss_fn = _FakeNonCompositeLoss()
    out = _composite_ss_undesired_kwargs(loss_fn, noisy, clean, enhanced, data)
    assert out == {}


def test_missing_data_returns_empty():
    noisy = torch.randn(2, 16000)
    clean = torch.randn(2, 16000)
    enhanced = torch.randn(2, 16000)
    loss_fn = _make_composite()
    out = _composite_ss_undesired_kwargs(loss_fn, noisy, clean, enhanced, data=None)
    assert out == {}


def test_state_kwargs_forwarded_only_for_state_ce():
    noisy = torch.randn(2, 16000)
    clean = torch.randn(2, 16000)
    enhanced = torch.randn(2, 16000)
    data = {
        "state": torch.zeros(2, 10, dtype=torch.long),
        "state_probs": torch.randn(2, 10, 8),
    }

    loss_fn = CompositeLoss(
        components=[{"type": "StateCrossEntropyLoss", "weight": 0.001}]
    )
    out = _composite_ss_undesired_kwargs(loss_fn, noisy, clean, enhanced, data)
    assert out["state"] is data["state"]
    assert out["state_probs"] is data["state_probs"]


def test_state_kwargs_suppressed_without_state_ce():
    noisy = torch.randn(2, 16000)
    clean = torch.randn(2, 16000)
    enhanced = torch.randn(2, 16000)
    data = {
        "state": torch.zeros(2, 10, dtype=torch.long),
        "state_probs": torch.randn(2, 10, 8),
    }

    loss_fn = CompositeLoss(components=[{"type": "L1WavLoss", "weight": 1.0}])
    out = _composite_ss_undesired_kwargs(loss_fn, noisy, clean, enhanced, data)
    assert "state" not in out
    assert "state_probs" not in out


def test_undesired_ref_forwarded_for_fw_neg_sisdr():
    noisy = torch.randn(2, 16000)
    clean = torch.randn(2, 16000)
    enhanced = torch.randn(2, 16000)

    loss_fn = CompositeLoss(
        components=[
            {
                "type": "FWNegSISDRLoss",
                "weight": 0.2,
                "args": {
                    "n_fft": 320,
                    "hop_length": 160,
                    "n_bins": 18,
                    "scale": "mel",
                    "weights_type": "log_snr_speech_norm",
                },
            }
        ]
    )
    out = _composite_ss_undesired_kwargs(loss_fn, noisy, clean, enhanced)

    assert "undesired_ref" in out
    assert "undesired_enhanced" in out
    assert torch.allclose(out["undesired_ref"], noisy - clean)
    assert torch.allclose(out["undesired_enhanced"], noisy - enhanced)


def test_undesired_ref_suppressed_without_fw_neg_sisdr():
    noisy = torch.randn(2, 16000)
    clean = torch.randn(2, 16000)
    enhanced = torch.randn(2, 16000)

    loss_fn = CompositeLoss(components=[{"type": "L1WavLoss", "weight": 1.0}])
    out = _composite_ss_undesired_kwargs(loss_fn, noisy, clean, enhanced)

    assert "undesired_ref" not in out
    assert "undesired_enhanced" not in out


def _mbrcl_component(weight: float) -> dict:
    return {
        "type": "MultiBandRelativeContrastiveLoss",
        "weight": weight,
        "args": {
            "sample_rate": 32000,
            "n_fft": 64,
            "hop_length": 16,
            "num_bands": 4,
        },
    }


def test_shared_waveform_kwargs_support_wrapped_active_mbrcl():
    noisy = torch.randn(2, 320)
    clean = torch.randn_like(noisy)
    enhanced = torch.randn_like(noisy)
    wrapped = _FakeWrappedLoss(CompositeLoss(components=[_mbrcl_component(-0.5)]))

    out = _waveform_loss_kwargs(wrapped, noisy, clean, enhanced, 32000)

    assert _loss_has_active_capability(wrapped, "REQUIRES_NOISY_REFERENCE")
    assert out["noisy"] is noisy
    assert out["sample_rate"] == 32000


def test_zero_weight_fw_neg_sisdr_does_not_build_undesired_waveforms():
    noisy = torch.randn(2, 320)
    clean = torch.randn_like(noisy)
    enhanced = torch.randn_like(noisy)
    loss_fn = CompositeLoss(
        components=[
            {
                "type": "FWNegSISDRLoss",
                "weight": 0.0,
                "args": {"n_fft": 64, "hop_length": 32},
            },
            {"type": "L1WavLoss", "weight": 1.0},
        ]
    )

    out = _waveform_loss_kwargs(loss_fn, noisy, clean, enhanced, 32000)

    assert "undesired_ref" not in out
    assert "undesired_enhanced" not in out


def test_aec_loss_kwargs_inject_mic_and_sample_rate_for_active_mbrcl():
    mic = torch.randn(2, 320)
    nearend = torch.randn_like(mic)
    nearend_estimate = torch.randn_like(mic)
    loss_fn = CompositeLoss(components=[_mbrcl_component(1.0)])

    out = _aec_loss_kwargs(
        loss_fn,
        mic,
        nearend,
        nearend_estimate,
        sample_rate=32000,
    )

    assert out["noisy"] is mic
    assert out["sample_rate"] == 32000
    assert torch.isfinite(loss_fn(nearend_estimate, nearend, **out))


def test_aec_loss_kwargs_ignore_zero_weight_mbrcl_capabilities():
    mic = torch.randn(2, 320)
    nearend = torch.randn_like(mic)
    nearend_estimate = torch.randn_like(mic)
    loss_fn = CompositeLoss(
        components=[
            _mbrcl_component(0.0),
            {"type": "L1WavLoss", "weight": 1.0},
        ]
    )

    out = _aec_loss_kwargs(
        loss_fn,
        mic,
        nearend,
        nearend_estimate,
        sample_rate=32000,
    )

    assert "noisy" not in out
    assert "sample_rate" not in out


def test_aec_loss_kwargs_do_not_inject_context_for_plain_loss():
    mic = torch.randn(2, 320)
    nearend = torch.randn_like(mic)
    nearend_estimate = torch.randn_like(mic)
    loss_fn = CompositeLoss(components=[{"type": "L1WavLoss", "weight": 1.0}])

    out = _aec_loss_kwargs(
        loss_fn,
        mic,
        nearend,
        nearend_estimate,
        sample_rate=32000,
    )

    assert "noisy" not in out
    assert "sample_rate" not in out


pytestmark = [pytest.mark.unit]
