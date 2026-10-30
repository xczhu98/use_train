"""GAP-URGENet PostNet aware loss 与论文归一化回归测试。"""

import pytest
import torch
import torch.nn as nn

from core.losses.pase_family import GAPPostNetAwareLoss, PaseMultiScaleMelLoss
from tasks.se.models.pase_family.gap_urgenet.postnet import TFGridNet

pytestmark = [pytest.mark.unit]


class _MeanProxy(nn.Module):
    def forward(self, enhanced, clean=None, lengths=None):
        return enhanced.mean()


def _mel_kwargs():
    return {
        "sampling_rate": 48000,
        "n_mels": [5],
        "window_lengths": [32],
        "weight": 1.0,
    }


def test_aware_loss_zero_weights_preserve_mel_value_and_gradient():
    torch.manual_seed(0)
    clean = torch.randn(2, 128)
    enhanced_a = torch.randn(2, 128, requires_grad=True)
    enhanced_b = enhanced_a.detach().clone().requires_grad_(True)
    baseline = PaseMultiScaleMelLoss(**_mel_kwargs())
    aware = GAPPostNetAwareLoss(**_mel_kwargs())

    loss_a = baseline(enhanced_a, clean)
    loss_b = aware(enhanced_b, clean)
    loss_a.backward()
    loss_b.backward()

    torch.testing.assert_close(loss_b, loss_a)
    torch.testing.assert_close(enhanced_b.grad, enhanced_a.grad)
    assert aware.pesq_loss is None and aware.utmos_loss is None


def test_aware_loss_combines_enabled_proxies(monkeypatch):
    monkeypatch.setattr(GAPPostNetAwareLoss, "_build_pesq_loss", lambda self: _MeanProxy())
    monkeypatch.setattr(GAPPostNetAwareLoss, "_build_utmos_loss", lambda self: _MeanProxy())
    enhanced = torch.full((1, 128), 0.25, requires_grad=True)
    clean = torch.zeros_like(enhanced)
    kwargs = _mel_kwargs()
    kwargs["sampling_rate"] = 16000
    loss_fn = GAPPostNetAwareLoss(**kwargs, pesq_weight=0.5, utmos_weight=0.25)

    actual = loss_fn(enhanced, clean)
    mel = loss_fn.mel_loss(enhanced, clean)
    torch.testing.assert_close(actual, mel + 0.75 * enhanced.mean())
    actual.backward()
    assert enhanced.grad is not None and torch.isfinite(enhanced.grad).all()


@pytest.mark.parametrize(
    ("mode", "expect_centered"),
    [("legacy_centered", True), ("paper_variance", False)],
)
def test_postnet_normalization_contract(monkeypatch, mode, expect_centered):
    model = TFGridNet(n_imics=2, n_layers=0, emb_dim=4, normalization_mode=mode)
    captured = {}

    def fake_stft(x, *args, **kwargs):
        captured["x"] = x.detach()
        return torch.zeros(x.shape[0], 769, 2, device=x.device, dtype=torch.complex64)

    monkeypatch.setattr(torch, "stft", fake_stft)
    monkeypatch.setattr(torch, "istft", lambda *args, **kwargs: torch.zeros(1, 1536))
    x = torch.arange(1, 1 + 2 * 1536, dtype=torch.float32).reshape(1, 2, 1536)
    model(x)

    normalized = captured["x"].reshape_as(x)
    std = x.std(dim=(1, 2), keepdim=True) + 1e-12
    expected = (x - x.mean(dim=(1, 2), keepdim=True)) / std if expect_centered else x / std
    torch.testing.assert_close(normalized, expected)


def test_postnet_rejects_unknown_normalization_mode():
    with pytest.raises(ValueError, match="normalization_mode"):
        TFGridNet(normalization_mode="unknown")
