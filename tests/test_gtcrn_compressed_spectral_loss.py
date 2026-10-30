"""GTCRN compressed spectral loss 的波形边界回归测试。"""

import pytest
import torch

from core.losses.hybrid import GTCRNCompressedSpectralLoss


pytestmark = [pytest.mark.unit]


def _small_loss(**kwargs) -> GTCRNCompressedSpectralLoss:
    return GTCRNCompressedSpectralLoss(
        n_fft=16,
        hop_length=4,
        win_length=16,
        **kwargs,
    )


def test_gtcrn_compressed_spectral_centered_short_input_is_differentiable():
    """center=True 的短波形不应因 reflect padding 大于输入而崩溃。"""
    clean = torch.tensor([[0.2, -0.4, 0.7, -0.1]])
    enhanced = (clean + torch.tensor([[0.05, -0.02, 0.01, 0.03]])).requires_grad_(
        True
    )

    loss = _small_loss()(enhanced, clean)
    loss.backward()

    assert torch.isfinite(loss)
    assert enhanced.grad is not None
    assert torch.isfinite(enhanced.grad).all()
    assert torch.count_nonzero(enhanced.grad) > 0


def test_gtcrn_compressed_spectral_one_sample_padding_matches_trim():
    """仅多 1 个 padding 样点时也必须与按 lengths 逐条 trim 一致。"""
    generator = torch.Generator().manual_seed(37)
    clean_valid = torch.randn(17, generator=generator)
    enhanced_valid = clean_valid + 0.1 * torch.randn(17, generator=generator)
    loss_fn = _small_loss()

    trimmed = enhanced_valid.unsqueeze(0).clone().requires_grad_(True)
    trimmed_loss = loss_fn(trimmed, clean_valid.unsqueeze(0))
    trimmed_grad = torch.autograd.grad(trimmed_loss, trimmed)[0]

    padded = torch.cat([enhanced_valid, torch.tensor([13.0])]).unsqueeze(0)
    padded = padded.requires_grad_(True)
    padded_clean = torch.cat([clean_valid, torch.tensor([-11.0])]).unsqueeze(0)
    padded_loss = loss_fn(padded, padded_clean, torch.tensor([17]))
    padded_grad = torch.autograd.grad(padded_loss, padded)[0]

    assert torch.allclose(padded_loss, trimmed_loss, rtol=1e-6, atol=1e-7)
    assert torch.allclose(padded_grad[:, :17], trimmed_grad, rtol=1e-5, atol=1e-7)
    assert torch.count_nonzero(padded_grad[:, 17:]) == 0


def test_gtcrn_compressed_spectral_perfect_reconstruction_has_finite_zero_gradient():
    """完美重建在短输入 fallback 上返回图零与有限零梯度。"""
    clean = torch.tensor([[0.3, -0.2, 0.8]])
    enhanced = clean.clone().requires_grad_(True)

    loss = _small_loss()(enhanced, clean, torch.tensor([3]))
    loss.backward()

    assert loss.item() == 0.0
    assert enhanced.grad is not None
    assert torch.isfinite(enhanced.grad).all()
    assert torch.count_nonzero(enhanced.grad) == 0


@pytest.mark.parametrize(("field", "value"), [("lamda_ri", -0.1), ("lamda_mag", -1.0)])
def test_gtcrn_compressed_spectral_rejects_negative_loss_weights(field, value):
    """RI 与 magnitude loss 权重必须是有限非负实数。"""
    with pytest.raises(ValueError, match=field):
        _small_loss(**{field: value})
