"""PASE multi-scale mel loss 的 lengths 与短输入回归测试。"""

import pytest
import torch
from librosa.filters import mel as librosa_mel_fn
from scipy import signal

from core.losses.pase_family import PaseMultiScaleMelLoss


_CURRENT_PASE_ARGS = {
    "sampling_rate": 16000,
    "n_mels": [5, 10, 20, 40, 80, 160, 320],
    "window_lengths": [32, 64, 128, 256, 512, 1024, 2048],
    "mag_weight": 0.0,
    "log_weight": 1.0,
    "pow": 1.0,
    "weight": 15.0,
    "window_type": "hann",
}


def _loss_and_grad(
    loss_fn: PaseMultiScaleMelLoss,
    enhanced_valid: torch.Tensor,
    clean_valid: torch.Tensor,
    extra: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if extra is None:
        enhanced = enhanced_valid.unsqueeze(0).requires_grad_(True)
        clean = clean_valid.unsqueeze(0)
        lengths = None
    else:
        enhanced_padding = torch.linspace(-3.0, 2.0, extra)
        clean_padding = torch.linspace(4.0, -2.0, extra)
        enhanced = torch.cat([enhanced_valid, enhanced_padding])
        enhanced = enhanced.unsqueeze(0).requires_grad_(True)
        clean = torch.cat([clean_valid, clean_padding]).unsqueeze(0)
        lengths = torch.tensor([enhanced_valid.numel()])

    loss = loss_fn(enhanced, clean, lengths)
    loss.backward()
    return loss.detach(), enhanced.grad.detach()


@pytest.mark.parametrize("sampling_rate", [16000, 48000], ids=["16k", "48k"])
@pytest.mark.parametrize("valid_length", [1, 7], ids=["one-sample", "short"])
@pytest.mark.parametrize("extra", [1, 1023], ids=["extra-1", "pad-minus-1"])
def test_pase_current_multiscale_mel_short_padding_matches_trim(
    sampling_rate: int, valid_length: int, extra: int
) -> None:
    """当前 PASE 七尺度配置对短输入应与 trim 的 loss/有效梯度一致。"""
    clean_valid = torch.linspace(0.25, 0.85, valid_length)
    enhanced_valid = 1.1 * clean_valid + torch.linspace(0.03, -0.02, valid_length)
    args = {**_CURRENT_PASE_ARGS, "sampling_rate": sampling_rate}
    loss_fn = PaseMultiScaleMelLoss(**args)

    trim_loss, trim_grad = _loss_and_grad(
        loss_fn, enhanced_valid, clean_valid, extra=None
    )
    padded_loss, padded_grad = _loss_and_grad(
        loss_fn, enhanced_valid, clean_valid, extra=extra
    )

    assert torch.isfinite(trim_loss) and torch.isfinite(padded_loss)
    assert torch.isfinite(trim_grad).all() and torch.isfinite(padded_grad).all()
    torch.testing.assert_close(padded_loss, trim_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        padded_grad[:, : enhanced_valid.numel()], trim_grad, rtol=1e-5, atol=1e-7
    )
    assert torch.count_nonzero(padded_grad[:, enhanced_valid.numel() :]) == 0


def test_pase_multiscale_mel_preserves_unpadded_formula() -> None:
    """无 padding 时仍使用原始 mel、log/magnitude 权重与尺度求和公式。"""
    generator = torch.Generator().manual_seed(61)
    clean = torch.randn(2, 1, 128, generator=generator)
    enhanced = clean + 0.08 * torch.randn(2, 1, 128, generator=generator)
    n_fft = 64
    hop_length = n_fft // 4
    n_mels = 12
    clamp_eps = 1e-5
    pow_value = 0.7
    log_weight = 0.8
    mag_weight = 0.3
    weight = 2.5

    window = torch.from_numpy(signal.get_window("hann", n_fft)).float()
    mel_basis = torch.from_numpy(
        librosa_mel_fn(sr=16000, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=None)
    )

    def _mel(wav: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            wav.reshape(-1, wav.shape[-1]),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        magnitude = spec.abs().reshape(2, 1, n_fft // 2 + 1, -1)
        return (magnitude.transpose(2, -1) @ mel_basis.T).transpose(-1, 2)

    enhanced_mel = _mel(enhanced)
    clean_mel = _mel(clean)
    log10 = torch.log(torch.tensor(10.0))
    enhanced_logmel = (
        torch.log(enhanced_mel.clamp(min=clamp_eps).pow(pow_value)) / log10
    )
    clean_logmel = torch.log(clean_mel.clamp(min=clamp_eps).pow(pow_value)) / log10
    expected = weight * (
        log_weight * torch.nn.functional.l1_loss(enhanced_logmel, clean_logmel)
        + mag_weight * torch.nn.functional.l1_loss(enhanced_mel, clean_mel)
    )

    actual = PaseMultiScaleMelLoss(
        sampling_rate=16000,
        n_mels=[n_mels],
        window_lengths=[n_fft],
        clamp_eps=clamp_eps,
        mag_weight=mag_weight,
        log_weight=log_weight,
        pow=pow_value,
        weight=weight,
    )(enhanced, clean)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


pytestmark = [pytest.mark.unit]
