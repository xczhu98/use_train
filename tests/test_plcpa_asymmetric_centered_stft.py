"""PLCPA-ASYM centered-STFT 边界与公式回归测试。"""

import pytest
import torch
import torch.nn.functional as F

from core.losses.plcpa_asymmetric import PLCPAAsymmetricLoss


@pytest.mark.parametrize("center", [True, False], ids=["centered", "noncentered"])
def test_plcpa_asymmetric_preserves_unmasked_formula(center):
    """正常长度、无 lengths 时保持 PLCPA-ASYM 的逐点平均公式。"""
    generator = torch.Generator().manual_seed(46)
    clean = torch.randn(2, 96, generator=generator)
    enhanced = clean + 0.1 * torch.randn(2, 96, generator=generator)
    loss_fn = PLCPAAsymmetricLoss(
        n_fft=64,
        hop_length=16,
        win_length=64,
        compress_factor=0.3,
        scale_mag=0.7,
        scale_phase=0.4,
        scale_asym=0.2,
        center=center,
    )

    actual = loss_fn(enhanced, clean)
    stft_kwargs = {
        "n_fft": 64,
        "hop_length": 16,
        "win_length": 64,
        "window": torch.hann_window(64),
        "center": center,
        "return_complex": True,
        "onesided": True,
    }
    enhanced_spec = torch.stft(enhanced, **stft_kwargs)
    clean_spec = torch.stft(clean, **stft_kwargs)
    enhanced_mag = enhanced_spec.abs().clamp_min(loss_fn.eps)
    clean_mag = clean_spec.abs().clamp_min(loss_fn.eps)
    enhanced_mag_c = enhanced_mag ** loss_fn.p
    clean_mag_c = clean_mag ** loss_fn.p
    enhanced_c = enhanced_spec / enhanced_mag ** (1.0 - loss_fn.p)
    clean_c = clean_spec / clean_mag ** (1.0 - loss_fn.p)
    expected = (
        loss_fn.scale_mag * (clean_mag_c - enhanced_mag_c).square().mean()
        + loss_fn.scale_phase * (clean_c - enhanced_c).abs().square().mean()
        + loss_fn.scale_asym * F.relu(clean_mag_c - enhanced_mag_c).square().mean()
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("center", "valid_len"),
    [
        pytest.param(True, 1, id="centered-extremely-short"),
        pytest.param(True, 48, id="centered-regular"),
        pytest.param(False, 96, id="noncentered-regression"),
    ],
)
@pytest.mark.parametrize("extra", [1, 31], ids=["extra-1", "pad-minus-1"])
def test_plcpa_asymmetric_padding_matches_trim(center, valid_len, extra):
    """PLCPA 的 trim/轻量 padding 应保持 loss 与有效区梯度等价。"""
    generator = torch.Generator().manual_seed(47)
    clean_valid = torch.randn(valid_len, generator=generator)
    enhanced_valid = clean_valid + 0.1 * torch.randn(valid_len, generator=generator)
    lengths = torch.tensor([valid_len])
    loss_fn = PLCPAAsymmetricLoss(
        n_fft=64,
        hop_length=16,
        win_length=64,
        center=center,
    )

    def loss_and_grad(right_padding):
        enhanced = torch.cat(
            [enhanced_valid, torch.zeros(right_padding)]
        ).unsqueeze(0).requires_grad_(True)
        clean = torch.cat([clean_valid, torch.zeros(right_padding)]).unsqueeze(0)
        loss = loss_fn(enhanced, clean, lengths)
        loss.backward()
        return loss.detach(), enhanced.grad.detach()

    trim_loss, trim_grad = loss_and_grad(0)
    padded_loss, padded_grad = loss_and_grad(extra)

    assert torch.isfinite(trim_loss) and torch.isfinite(padded_loss)
    assert torch.isfinite(trim_grad).all() and torch.isfinite(padded_grad).all()
    torch.testing.assert_close(padded_loss, trim_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        padded_grad[:, :valid_len], trim_grad, rtol=1e-5, atol=1e-7
    )
    assert torch.count_nonzero(padded_grad[:, valid_len:]) == 0


pytestmark = [pytest.mark.unit]
