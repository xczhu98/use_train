"""基础频域 loss 的 centered-STFT 短输入与 padding 回归测试。"""

import pytest
import torch

from core.losses.freq_domain import MultiResL1SpecLoss
from core.losses.spectral_distance import (
    LogMagnitudeSTFTLoss,
    SpectralConvergenceLoss,
)


def _loss_cases():
    return (
        pytest.param(
            SpectralConvergenceLoss,
            {"n_fft": 64, "hop_length": 16, "win_length": 64},
            id="spectral-convergence",
        ),
        pytest.param(
            LogMagnitudeSTFTLoss,
            {"n_fft": 64, "hop_length": 16, "win_length": 64},
            id="log-magnitude",
        ),
        pytest.param(
            MultiResL1SpecLoss,
            {
                "window_sz": (64,),
                "hop_sz": (16,),
                "normalize_variance": True,
            },
            id="multi-res-l1-spec",
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
def test_centered_stft_losses_short_padding_matches_trim(
    loss_cls,
    loss_kwargs,
    valid_len,
    extra,
):
    """极短输入可前反向，轻量 padding 与 trim 的 loss/有效梯度一致。"""
    clean_valid = torch.linspace(0.25, 0.85, valid_len)
    enhanced_valid = 1.1 * clean_valid + torch.linspace(0.03, -0.02, valid_len)
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


def test_spectral_convergence_preserves_normal_formula():
    generator = torch.Generator().manual_seed(51)
    clean = torch.randn(2, 96, generator=generator)
    enhanced = clean + 0.1 * torch.randn(2, 96, generator=generator)
    window = torch.hann_window(64)
    kwargs = {
        "n_fft": 64,
        "hop_length": 16,
        "win_length": 64,
        "window": window,
        "return_complex": True,
    }

    enhanced_mag = torch.stft(enhanced, **kwargs).abs().clamp_min(1e-7)
    clean_mag = torch.stft(clean, **kwargs).abs().clamp_min(1e-7)
    expected = (
        torch.linalg.vector_norm(clean_mag - enhanced_mag, dim=(1, 2))
        / torch.linalg.vector_norm(clean_mag, dim=(1, 2)).clamp_min(1e-8)
    ).mean()
    actual = SpectralConvergenceLoss(64, 16, 64)(enhanced, clean)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("target_scale", [0.0, 1e-12])
def test_spectral_convergence_zero_target_uses_stable_log_magnitude_fallback(
    target_scale,
):
    generator = torch.Generator().manual_seed(54)
    enhanced = torch.randn(2, 96, generator=generator).requires_grad_(True)
    clean = target_scale * torch.randn(2, 96, generator=generator)
    loss_fn = SpectralConvergenceLoss(64, 16, 64)

    actual = loss_fn(enhanced, clean)
    expected = LogMagnitudeSTFTLoss(64, 16, 64)(enhanced, clean)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert torch.isfinite(actual) and actual < 100.0
    actual.backward()
    assert torch.isfinite(enhanced.grad).all()
    assert enhanced.grad.abs().max() < 100.0


@pytest.mark.parametrize("target_scale", [0.0, 1e-12])
def test_spectral_convergence_zero_target_has_suppression_descent_direction(
    target_scale,
):
    generator = torch.Generator().manual_seed(55)
    clean = target_scale * torch.randn(1, 96, generator=generator)
    enhanced = torch.randn(1, 96, generator=generator).requires_grad_(True)
    loss_fn = SpectralConvergenceLoss(64, 16, 64)

    loss = loss_fn(enhanced, clean)
    loss.backward()
    reduced_loss = loss_fn(0.5 * enhanced.detach(), clean)

    assert reduced_loss < loss.detach()
    assert torch.sum(enhanced.grad * (-enhanced.detach())) < 0.0


@pytest.mark.parametrize("target_scale", [0.0, 1.0])
def test_spectral_convergence_perfect_point_has_finite_zero_gradient(target_scale):
    generator = torch.Generator().manual_seed(56)
    clean = target_scale * torch.randn(2, 96, generator=generator)
    enhanced = clean.clone().requires_grad_(True)
    loss = SpectralConvergenceLoss(64, 16, 64)(enhanced, clean)

    loss.backward()

    assert loss == 0.0
    assert torch.isfinite(enhanced.grad).all()
    assert torch.count_nonzero(enhanced.grad) == 0


def test_log_magnitude_preserves_normal_formula():
    generator = torch.Generator().manual_seed(52)
    clean = torch.randn(2, 96, generator=generator)
    enhanced = clean + 0.1 * torch.randn(2, 96, generator=generator)
    window = torch.hann_window(64)
    kwargs = {
        "n_fft": 64,
        "hop_length": 16,
        "win_length": 64,
        "window": window,
        "return_complex": True,
    }

    enhanced_mag = torch.stft(enhanced, **kwargs).abs().clamp_min(1e-7)
    clean_mag = torch.stft(clean, **kwargs).abs().clamp_min(1e-7)
    expected = (
        torch.log(enhanced_mag + 1e-8) - torch.log(clean_mag + 1e-8)
    ).abs().mean()
    actual = LogMagnitudeSTFTLoss(64, 16, 64)(enhanced, clean)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_multi_res_l1_spec_preserves_normal_formula():
    generator = torch.Generator().manual_seed(53)
    clean = torch.randn(2, 96, generator=generator)
    enhanced = clean + 0.1 * torch.randn(2, 96, generator=generator)
    eps = 1e-6
    weight = 0.4

    clean_normalized = clean / torch.sqrt(torch.var(clean, dim=1, keepdim=True, unbiased=False) + eps)
    enhanced_normalized = enhanced / torch.sqrt(
        torch.var(enhanced, dim=1, keepdim=True, unbiased=False) + eps
    )
    scaling = (
        torch.sum(enhanced_normalized * clean_normalized, dim=-1, keepdim=True)
        / (torch.sum(enhanced_normalized.square(), dim=-1, keepdim=True) + eps)
    )
    enhanced_scaled = enhanced_normalized * scaling
    time_loss = torch.sum((enhanced_scaled - clean_normalized).abs(), dim=-1)
    kwargs = {
        "n_fft": 64,
        "hop_length": 16,
        "win_length": 64,
        "window": torch.hann_window(64),
        "center": True,
        "return_complex": True,
    }
    clean_mag = torch.stft(clean_normalized, **kwargs).abs()
    enhanced_mag = torch.stft(enhanced_scaled, **kwargs).abs()
    spec_loss = torch.sum((enhanced_mag - clean_mag).abs(), dim=(1, 2))
    expected = (weight * time_loss + (1.0 - weight) * spec_loss).mean()

    actual = MultiResL1SpecLoss(
        window_sz=(64,),
        hop_sz=(16,),
        eps=eps,
        time_domain_weight=weight,
        normalize_variance=True,
    )(enhanced, clean)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


pytestmark = [pytest.mark.unit]
