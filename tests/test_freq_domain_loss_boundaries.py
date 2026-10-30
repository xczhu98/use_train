"""频域损失的公式、短输入与有效边界回归测试。"""

import pytest
import torch

from core.losses.freq_domain import MagnitudePlusComplexL1Loss, MultiResFreqLoss


def _signals(length: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260901 + length)
    clean = torch.randn(length, generator=generator)
    enhanced = clean + 0.08 * torch.randn(length, generator=generator)
    return enhanced, clean


def _loss_and_grad(
    loss_fn,
    enhanced_valid: torch.Tensor,
    clean_valid: torch.Tensor,
    right_padding: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if right_padding is None:
        enhanced = enhanced_valid.unsqueeze(0).requires_grad_(True)
        clean = clean_valid.unsqueeze(0)
        lengths = None
    else:
        enhanced = torch.cat(
            [enhanced_valid, enhanced_valid.new_zeros(right_padding)]
        ).unsqueeze(0)
        enhanced.requires_grad_(True)
        clean = torch.cat(
            [clean_valid, clean_valid.new_zeros(right_padding)]
        ).unsqueeze(0)
        lengths = torch.tensor([clean_valid.numel()])

    loss = loss_fn(enhanced, clean, lengths)
    loss.backward()
    return loss.detach(), enhanced.grad.detach()


def _assert_padded_matches_trimmed(loss_fn, valid_length: int, extra: int) -> None:
    enhanced, clean = _signals(valid_length)
    trimmed_loss, trimmed_grad = _loss_and_grad(loss_fn, enhanced, clean, None)
    padded_loss, padded_grad = _loss_and_grad(loss_fn, enhanced, clean, extra)

    torch.testing.assert_close(padded_loss, trimmed_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        padded_grad[:, :valid_length], trimmed_grad, rtol=1e-5, atol=1e-7
    )
    assert torch.count_nonzero(padded_grad[:, valid_length:]) == 0


def test_multi_res_freq_preserves_unpadded_formula() -> None:
    enhanced, clean = _signals(384)
    enhanced = enhanced.unsqueeze(0)
    clean = clean.unsqueeze(0)
    loss_fn = MultiResFreqLoss(fft_sizes=(64, 128))

    actual = loss_fn(enhanced, clean)
    expected = actual.new_zeros(())
    for n_fft in loss_fn.fft_sizes:
        kwargs = {
            "n_fft": n_fft,
            "hop_length": n_fft // 2,
            "win_length": n_fft,
            "window": getattr(loss_fn, f"_win_{n_fft}"),
            "center": True,
            "return_complex": True,
        }
        enhanced_spec = torch.stft(enhanced, **kwargs)
        clean_spec = torch.stft(clean, **kwargs)
        enhanced_mag = enhanced_spec.abs().clamp(1e-8)
        clean_mag = clean_spec.abs().clamp(1e-8)
        complex_mse = (
            torch.view_as_real(enhanced_spec) - torch.view_as_real(clean_spec)
        ).square().mean()
        relative_mag = (
            (enhanced_mag - clean_mag) / (enhanced_mag + clean_mag + 1e-7)
        ).square().mean()
        expected = expected + complex_mse + relative_mag
    expected = expected / len(loss_fn.fft_sizes)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_magnitude_plus_complex_l1_preserves_unpadded_formula() -> None:
    enhanced, clean = _signals(1024)
    enhanced = enhanced.unsqueeze(0)
    clean = clean.unsqueeze(0)
    loss_fn = MagnitudePlusComplexL1Loss(
        n_fft=512, hop_length=128, win_length=512
    )

    actual = loss_fn(enhanced, clean)
    kwargs = {
        "n_fft": 512,
        "hop_length": 128,
        "win_length": 512,
        "window": loss_fn._window,
        "center": True,
        "return_complex": True,
    }
    clean_spec = torch.stft(clean, **kwargs)
    enhanced_spec = torch.stft(enhanced, **kwargs)
    expected = (
        (clean_spec.abs() - enhanced_spec.abs()).abs()
        + (clean_spec - enhanced_spec).abs()
    ).mean()

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("extra", [1, 1023], ids=["extra-1", "pad-minus-1"])
def test_multi_res_freq_default_geometry_ignores_batch_padding(extra: int) -> None:
    _assert_padded_matches_trimmed(MultiResFreqLoss(), valid_length=1280, extra=extra)


@pytest.mark.parametrize("extra", [1, 255], ids=["extra-1", "pad-minus-1"])
def test_magnitude_plus_complex_l1_production_geometry_ignores_batch_padding(
    extra: int,
) -> None:
    loss_fn = MagnitudePlusComplexL1Loss(
        n_fft=512, hop_length=128, win_length=512
    )
    _assert_padded_matches_trimmed(loss_fn, valid_length=640, extra=extra)


@pytest.mark.parametrize(
    ("loss_fn", "extra"),
    [
        pytest.param(MultiResFreqLoss(), 1, id="multi-res-freq-extra-1"),
        pytest.param(MultiResFreqLoss(), 1023, id="multi-res-freq-pad-minus-1"),
        pytest.param(
            MagnitudePlusComplexL1Loss(
                n_fft=512, hop_length=128, win_length=512
            ),
            1,
            id="magnitude-plus-complex-l1-extra-1",
        ),
        pytest.param(
            MagnitudePlusComplexL1Loss(
                n_fft=512, hop_length=128, win_length=512
            ),
            255,
            id="magnitude-plus-complex-l1-pad-minus-1",
        ),
    ],
)
def test_frequency_losses_accept_one_sample_with_light_padding(
    loss_fn, extra: int
) -> None:
    enhanced = torch.tensor([0.7])
    clean = torch.tensor([0.5])

    trimmed_loss, trimmed_grad = _loss_and_grad(loss_fn, enhanced, clean, None)
    padded_loss, padded_grad = _loss_and_grad(loss_fn, enhanced, clean, extra)

    assert torch.isfinite(padded_loss)
    assert torch.isfinite(padded_grad).all()
    torch.testing.assert_close(padded_loss, trimmed_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(padded_grad[:, :1], trimmed_grad, rtol=1e-5, atol=1e-7)
    assert torch.count_nonzero(padded_grad[:, 1:]) == 0


pytestmark = [pytest.mark.unit]
