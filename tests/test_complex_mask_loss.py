"""Complex ideal ratio mask regression loss tests."""

import pytest
import torch

from core.losses.build import build_loss
from core.losses.complex_mask import ComplexIdealRatioMaskLoss

pytestmark = pytest.mark.unit


def _spectra(waveform, *, n_fft, hop_length, win_length):
    window = torch.hann_window(win_length, device=waveform.device)
    return torch.view_as_real(
        torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            normalized=False,
            return_complex=True,
        )
    )


def test_ideal_enhanced_spectrum_has_zero_compressed_cirm_error():
    n_fft, hop_length = 64, 16
    clean = torch.randn(2, 512)
    noisy = torch.stack((clean + 0.2 * torch.randn_like(clean), clean), dim=1)
    clean_spectrum = _spectra(clean, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    loss = ComplexIdealRatioMaskLoss(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        reference_channel=1,
    )

    value = loss(
        clean,
        clean,
        noisy=noisy,
        enhanced_spectra=clean_spectrum,
        target_spectra=clean_spectrum,
    )

    assert value.ndim == 0
    assert value.item() == pytest.approx(0.0, abs=1e-10)


def test_cirm_loss_backpropagates_through_direct_spectra_and_masks_lengths():
    n_fft, hop_length = 64, 16
    clean = torch.randn(2, 513)
    noisy = torch.stack((torch.randn_like(clean), torch.randn_like(clean)), dim=1)
    target_spectrum = _spectra(clean, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    prediction = (target_spectrum + 0.05 * torch.randn_like(target_spectrum)).requires_grad_()
    loss = ComplexIdealRatioMaskLoss(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        reference_channel=1,
    )

    value = loss(
        prediction.new_zeros(clean.shape),
        clean,
        torch.tensor([513, 401], dtype=torch.int32),
        noisy=noisy,
        enhanced_spectra=prediction,
        target_spectra=target_spectrum,
    )
    value.backward()

    assert value.ndim == 0 and torch.isfinite(value)
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert prediction.grad[1, :, 30:, :].abs().sum() == 0


def test_build_loss_injects_framework_stft_and_composite_forwards_noisy_reference():
    n_fft, hop_length = 64, 16
    stft = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": n_fft,
        "window": "hann",
        "center": True,
        "normalized": False,
    }
    loss = build_loss(
        {
            "type": "CompositeLoss",
            "components": [{"type": "ComplexIdealRatioMaskLoss", "weight": 1.0}],
            "strict": True,
        },
        stft_cfg=stft,
    )
    clean = torch.randn(1, 512)
    noisy = torch.stack((clean + torch.randn_like(clean), clean), dim=1)
    spectra = _spectra(clean, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    enhanced_spectra = (spectra + 0.02 * torch.randn_like(spectra)).requires_grad_()

    value = loss(
        clean,
        clean,
        torch.tensor([512], dtype=torch.int32),
        noisy=noisy,
        enhanced_spectra=enhanced_spectra,
        target_spectra=spectra,
    )
    value.backward()

    component = loss.loss.losses[0]
    assert isinstance(component, ComplexIdealRatioMaskLoss)
    assert (component.n_fft, component.hop_length, component.win_length) == (64, 16, 64)
    assert torch.isfinite(value)
    assert enhanced_spectra.grad is not None and torch.isfinite(enhanced_spectra.grad).all()


def test_se_loss_capability_uses_model_reference_channel_before_cirm_context():
    from tasks.se.data.batch import SEBatch
    from tasks.se.trainer.metrics_trainer import _se_loss_kwargs

    n_fft, hop_length = 64, 16
    clean = torch.randn(1, 256)
    noisy = torch.stack((torch.randn_like(clean), clean + 0.1 * torch.randn_like(clean)), dim=1)
    batch = SEBatch(clean=clean, noisy=noisy, sr=torch.tensor(8000), lengths=torch.tensor([256]))
    loss = build_loss(
        {
            "type": "ComplexIdealRatioMaskLoss",
            "args": {"reference_channel": 1},
            "strict": True,
        },
        stft_cfg={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": n_fft,
            "window": "hann",
            "center": True,
            "normalized": False,
        },
    )
    spectra = _spectra(clean, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    enhanced_spectra = (spectra + 0.01 * torch.randn_like(spectra)).requires_grad_()
    kwargs = _se_loss_kwargs(
        loss,
        batch,
        clean,
        8000,
        reference_channel=1,
    )
    kwargs.update(enhanced_spectra=enhanced_spectra, target_spectra=spectra)
    value = loss(clean, clean, batch.lengths, **kwargs)
    value.backward()

    assert kwargs["noisy"].shape == clean.shape
    torch.testing.assert_close(kwargs["noisy"], noisy[:, 1])
    assert torch.isfinite(value)
    assert enhanced_spectra.grad is not None and torch.isfinite(enhanced_spectra.grad).all()


def test_cirm_loss_fails_without_spectra_or_noisy_reference():
    loss = ComplexIdealRatioMaskLoss(n_fft=64, hop_length=16, win_length=64)
    waveform = torch.zeros(1, 128)
    with pytest.raises(ValueError, match="requires enhanced_spectra and target_spectra"):
        loss(waveform, waveform, noisy=waveform)
    with pytest.raises(ValueError, match="requires a noisy waveform reference"):
        loss(
            waveform,
            waveform,
            enhanced_spectra=torch.zeros(1, 33, 9, 2),
            target_spectra=torch.zeros(1, 33, 9, 2),
        )
