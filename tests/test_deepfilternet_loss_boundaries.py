"""DeepFilterNet 谱损失的 STFT 边界与梯度回归测试。"""

import pytest
import torch

from core.losses.model_specific.deepfilternet import (
    DeepFilterNetMultiResSpecLoss,
    DeepFilterNetOverAttenuationLoss,
)


def _loss_cases():
    return (
        pytest.param(
            DeepFilterNetMultiResSpecLoss,
            {
                "fft_sizes": (64,),
                "hop_factor": 0.25,
                "sample_rate": 16000,
                "max_freq_hz": None,
            },
            id="multi-res-spec",
        ),
        pytest.param(
            DeepFilterNetOverAttenuationLoss,
            {"n_fft": 64, "hop_length": 16, "win_length": 64},
            id="over-attenuation",
        ),
    )


def _loss_and_grad(loss_fn, enhanced, clean, lengths=None):
    enhanced = enhanced.clone().unsqueeze(0).requires_grad_(True)
    clean = clean.clone().unsqueeze(0)
    loss = loss_fn(enhanced, clean, lengths)
    loss.backward()
    return loss.detach(), enhanced.grad.detach()


@pytest.mark.parametrize(("loss_cls", "loss_kwargs"), _loss_cases())
@pytest.mark.parametrize(
    "valid_len", [1, 7, 96], ids=["one-sample", "short", "regular"]
)
@pytest.mark.parametrize("extra", [1, 31], ids=["extra-1", "pad-minus-1"])
def test_deepfilternet_spectral_losses_short_padding_matches_trim(
    loss_cls,
    loss_kwargs,
    valid_len,
    extra,
):
    """极短输入可前反向，轻量 padding 与 trim 的 loss/有效梯度一致。"""
    clean_valid = torch.linspace(0.25, 0.85, valid_len)
    enhanced_valid = 0.8 * clean_valid + torch.linspace(0.03, -0.02, valid_len)
    loss_fn = loss_cls(**loss_kwargs)

    trim_loss, trim_grad = _loss_and_grad(loss_fn, enhanced_valid, clean_valid)
    padded_loss, padded_grad = _loss_and_grad(
        loss_fn,
        torch.cat([enhanced_valid, torch.zeros(extra)]),
        torch.cat([clean_valid, torch.zeros(extra)]),
        torch.tensor([valid_len]),
    )

    assert torch.isfinite(trim_loss) and torch.isfinite(padded_loss)
    assert torch.isfinite(trim_grad).all() and torch.isfinite(padded_grad).all()
    torch.testing.assert_close(padded_loss, trim_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        padded_grad[:, :valid_len], trim_grad, rtol=1e-5, atol=1e-7
    )
    assert torch.count_nonzero(padded_grad[:, valid_len:]) == 0


def test_deepfilternet_multi_res_spec_preserves_nonzero_error_formula():
    generator = torch.Generator().manual_seed(61)
    clean = torch.randn(2, 96, generator=generator)
    enhanced = clean + 0.1 * torch.randn(2, 96, generator=generator)
    loss_fn = DeepFilterNetMultiResSpecLoss(
        fft_sizes=(64,),
        hop_factor=0.25,
        sample_rate=16000,
        max_freq_hz=None,
    )

    actual = loss_fn(enhanced, clean)
    kwargs = {
        "n_fft": 64,
        "hop_length": 16,
        "win_length": 32,
        "window": torch.hann_window(32),
        "center": True,
        "return_complex": True,
    }
    enhanced_mag = torch.stft(enhanced, **kwargs).abs().clamp(min=1e-8)
    clean_mag = torch.stft(clean, **kwargs).abs().clamp(min=1e-8)
    expected = (enhanced_mag - clean_mag).abs().mean()

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_deepfilternet_over_attenuation_preserves_nonzero_error_formula():
    generator = torch.Generator().manual_seed(62)
    clean = torch.randn(2, 96, generator=generator)
    enhanced = 0.8 * clean
    loss_fn = DeepFilterNetOverAttenuationLoss(
        n_fft=64, hop_length=16, win_length=64
    )

    actual = loss_fn(enhanced, clean)
    kwargs = {
        "n_fft": 64,
        "hop_length": 16,
        "win_length": 64,
        "window": torch.hann_window(64),
        "center": True,
        "return_complex": True,
    }
    enhanced_mag = torch.stft(enhanced, **kwargs).abs().clamp(min=1e-8)
    clean_mag = torch.stft(clean, **kwargs).abs().clamp(min=1e-8)
    expected = (clean_mag - enhanced_mag).clamp(min=0.0) / (clean_mag + 1e-8)

    torch.testing.assert_close(actual, expected.mean(), rtol=1e-6, atol=1e-7)


def test_deepfilternet_over_attenuation_perfect_reconstruction_has_zero_gradient():
    generator = torch.Generator().manual_seed(63)
    clean = torch.randn(1, 96, generator=generator)
    enhanced = clean.clone().requires_grad_(True)
    loss_fn = DeepFilterNetOverAttenuationLoss(
        n_fft=64, hop_length=16, win_length=64
    )

    loss = loss_fn(enhanced, clean)
    loss.backward()

    assert loss.item() == 0.0
    assert enhanced.grad is not None
    assert torch.isfinite(enhanced.grad).all()
    assert torch.count_nonzero(enhanced.grad) == 0


pytestmark = [pytest.mark.unit]
