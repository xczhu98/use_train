"""MultiResSTFTLoss 数值稳定性与 STFT 边界回归测试。"""

import pytest
import torch

from core.losses.freq_domain import MultiResSTFTLoss


def _single_resolution_loss() -> MultiResSTFTLoss:
    return MultiResSTFTLoss(
        fft_sizes=(64,),
        hop_sizes=(16,),
        win_lengths=(64,),
    )


def test_multi_res_stft_perfect_reconstruction_has_finite_zero_gradient():
    """SC 的 sqrt(0) 不得让完美重建产生 NaN 梯度。"""
    generator = torch.Generator().manual_seed(41)
    clean = torch.randn(1, 128, generator=generator)
    enhanced = clean.clone().requires_grad_(True)

    loss = _single_resolution_loss()(enhanced, clean)
    loss.backward()

    assert loss.item() == 0.0
    assert enhanced.grad is not None
    assert torch.isfinite(enhanced.grad).all()
    assert torch.count_nonzero(enhanced.grad) == 0


def test_multi_res_stft_nonzero_error_preserves_original_formula():
    """非零误差仍严格使用 SC + log-mag L1 原公式。"""
    generator = torch.Generator().manual_seed(42)
    clean = torch.randn(1, 128, generator=generator)
    enhanced = clean + 0.1 * torch.randn(1, 128, generator=generator)

    actual = _single_resolution_loss()(enhanced, clean)
    stft_args = {
        "n_fft": 64,
        "hop_length": 16,
        "win_length": 64,
        "window": torch.hann_window(64),
        "return_complex": True,
        "center": True,
    }
    enhanced_mag = torch.stft(enhanced, **stft_args).abs().clamp(min=1e-8)
    clean_mag = torch.stft(clean, **stft_args).abs().clamp(min=1e-8)
    expected = (
        torch.sqrt(torch.sum((clean_mag - enhanced_mag).square()))
        / torch.sqrt(torch.sum(clean_mag.square()))
        + (torch.log(enhanced_mag) - torch.log(clean_mag)).abs().mean()
    )

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("num_samples", [1, 2, 8, 32])
def test_multi_res_stft_centered_stft_accepts_very_short_inputs(num_samples):
    """center=True 在输入不长于 reflect pad 时仍应稳定前反向。"""
    clean = torch.linspace(0.25, 1.0, num_samples).unsqueeze(0)
    enhanced = (clean + 0.1).requires_grad_(True)

    loss = _single_resolution_loss()(enhanced, clean)
    loss.backward()

    assert torch.isfinite(loss)
    assert enhanced.grad is not None
    assert torch.isfinite(enhanced.grad).all()


def test_multi_res_stft_one_sample_batch_padding_matches_trimmed_context():
    """仅多一个 batch padding 样本也须保留 trim 后完整 reflect 上下文。"""
    generator = torch.Generator().manual_seed(43)
    valid_length = 96
    clean_valid = torch.randn(valid_length, generator=generator)
    enhanced_valid = clean_valid + 0.05 * torch.randn(valid_length, generator=generator)
    loss_fn = _single_resolution_loss()

    trimmed_enhanced = enhanced_valid.unsqueeze(0).requires_grad_(True)
    trimmed_loss = loss_fn(trimmed_enhanced, clean_valid.unsqueeze(0))
    trimmed_loss.backward()

    padded_enhanced = torch.cat([enhanced_valid, torch.zeros(1)]).unsqueeze(0)
    padded_enhanced.requires_grad_(True)
    padded_clean = torch.cat([clean_valid, torch.zeros(1)]).unsqueeze(0)
    padded_loss = loss_fn(
        padded_enhanced,
        padded_clean,
        torch.tensor([valid_length]),
    )
    padded_loss.backward()

    assert torch.allclose(padded_loss, trimmed_loss, rtol=1e-6, atol=1e-7)
    assert torch.allclose(
        padded_enhanced.grad[:, :valid_length],
        trimmed_enhanced.grad,
        rtol=1e-5,
        atol=1e-7,
    )
    assert torch.count_nonzero(padded_enhanced.grad[:, valid_length:]) == 0


def test_multi_res_stft_zero_target_uses_single_log_magnitude_with_gradient():
    """零 target 关闭无定义的 SC，但保留一次 log-mag residual suppression。"""
    generator = torch.Generator().manual_seed(44)
    enhanced = torch.randn(1, 128, generator=generator).requires_grad_(True)
    clean = torch.zeros_like(enhanced)

    actual = _single_resolution_loss()(enhanced, clean)
    stft_args = {
        "n_fft": 64,
        "hop_length": 16,
        "win_length": 64,
        "window": torch.hann_window(64),
        "return_complex": True,
        "center": True,
    }
    enhanced_mag = torch.stft(enhanced, **stft_args).abs().clamp(min=1e-8)
    expected = (
        torch.log(enhanced_mag) - torch.log(torch.full_like(enhanced_mag, 1e-8))
    ).abs().mean()

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)
    actual.backward()
    assert enhanced.grad is not None
    assert torch.isfinite(enhanced.grad).all()
    assert torch.count_nonzero(enhanced.grad) > 0


def test_multi_res_stft_all_zero_has_finite_zero_loss_and_gradient():
    """target/output 全零时，SC 与 floored log-mag 均为 graph-connected zero。"""
    enhanced = torch.zeros(1, 128, requires_grad=True)
    clean = torch.zeros_like(enhanced)

    loss = _single_resolution_loss()(enhanced, clean)
    loss.backward()

    assert loss.item() == 0.0
    assert enhanced.grad is not None
    assert torch.isfinite(enhanced.grad).all()
    assert torch.count_nonzero(enhanced.grad) == 0


pytestmark = [pytest.mark.unit]
