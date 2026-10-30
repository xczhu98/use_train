"""GAPHybridLoss 的 length-aware 边界与公式回归测试。"""

import pytest
import torch

from core.losses.pase_family import GAPHybridLoss


def _small_loss() -> GAPHybridLoss:
    return GAPHybridLoss(
        sample_rate=1000,
        fft_len=0.064,
        hop_len=0.016,
        win_len=0.064,
    )


def _loss_and_grad(
    loss_fn: GAPHybridLoss,
    enhanced_valid: torch.Tensor,
    clean_valid: torch.Tensor,
    total: int,
):
    valid_len = clean_valid.numel()
    enhanced = torch.cat(
        [enhanced_valid, torch.full((total - valid_len,), 3.0)]
    ).unsqueeze(0)
    enhanced.requires_grad_(True)
    clean = torch.cat(
        [clean_valid, torch.full((total - valid_len,), -2.0)]
    ).unsqueeze(0)
    loss = loss_fn(enhanced, clean, torch.tensor([valid_len]))
    loss.backward()
    return loss.detach(), enhanced.grad.detach()


def test_gap_hybrid_loss_ignores_waveform_padding_in_loss_and_gradient():
    """谱项和 signed SDR 均只能读取 lengths 内的有效信号。"""
    generator = torch.Generator().manual_seed(71)
    clean = torch.randn(96, generator=generator)
    enhanced = clean + 0.1 * torch.randn(96, generator=generator)
    loss_fn = _small_loss()

    trim_loss, trim_grad = _loss_and_grad(loss_fn, enhanced, clean, 96)
    padded_loss, padded_grad = _loss_and_grad(loss_fn, enhanced, clean, 143)

    torch.testing.assert_close(padded_loss, trim_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        padded_grad[:, :96], trim_grad, rtol=1e-5, atol=1e-7
    )
    assert torch.count_nonzero(padded_grad[:, 96:]) == 0


@pytest.mark.parametrize("valid_len", [1, 7], ids=["one-sample", "short"])
@pytest.mark.parametrize("extra", [1, 31], ids=["extra-1", "pad-minus-1"])
def test_gap_hybrid_loss_short_input_is_stable_and_padding_invariant(
    valid_len,
    extra,
):
    """短于 centered-STFT reflect pad 的输入也须稳定且 trim 等价。"""
    clean = torch.linspace(0.2, 0.8, valid_len)
    enhanced = 1.1 * clean + torch.linspace(0.03, -0.01, valid_len)
    loss_fn = _small_loss()

    trim_loss, trim_grad = _loss_and_grad(
        loss_fn, enhanced, clean, valid_len
    )
    padded_loss, padded_grad = _loss_and_grad(
        loss_fn, enhanced, clean, valid_len + extra
    )

    assert torch.isfinite(trim_loss) and torch.isfinite(padded_loss)
    assert torch.isfinite(trim_grad).all() and torch.isfinite(padded_grad).all()
    torch.testing.assert_close(padded_loss, trim_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        padded_grad[:, :valid_len], trim_grad, rtol=1e-5, atol=1e-7
    )
    assert torch.count_nonzero(padded_grad[:, valid_len:]) == 0


def _official_gap_hybrid_formula(
    enhanced: torch.Tensor,
    clean: torch.Tensor,
    loss_fn: GAPHybridLoss,
) -> torch.Tensor:
    """独立复现官方 HybridLoss 的压缩谱与带符号 SDR 公式。"""
    n_fft = int(loss_fn.fft_len * loss_fn.sample_rate)
    n_hop = int(loss_fn.hop_len * loss_fn.sample_rate)
    n_win = int(loss_fn.win_len * loss_fn.sample_rate)
    window = torch.hann_window(n_win)
    pred_stft = torch.stft(
        enhanced, n_fft, n_hop, n_win, window, return_complex=True
    )
    true_stft = torch.stft(
        clean, n_fft, n_hop, n_win, window, return_complex=True
    )
    pred_mag = pred_stft.abs().clamp(loss_fn.eps)
    true_mag = true_stft.abs().clamp(loss_fn.eps)
    pred_stft_c = pred_stft / pred_mag ** (1 - loss_fn.c)
    true_stft_c = true_stft / true_mag ** (1 - loss_fn.c)
    real_loss = (pred_stft_c.real - true_stft_c.real).square().mean()
    imag_loss = (pred_stft_c.imag - true_stft_c.imag).square().mean()
    mag_loss = (pred_mag**loss_fn.c - true_mag**loss_fn.c).square().mean()
    sdr = -2 * torch.log10(
        torch.norm(enhanced, dim=-1, keepdim=True)
        / torch.norm(enhanced - clean, dim=-1, keepdim=True).clamp(loss_fn.eps)
        + loss_fn.eps
    ).mean()
    return (
        loss_fn.lamda_ri * (real_loss + imag_loss)
        + loss_fn.lamda_mag * mag_loss
        + loss_fn.lamda_sdr * sdr
    )


@pytest.mark.parametrize("perfect", [False, True], ids=["normal", "perfect"])
def test_gap_hybrid_loss_preserves_official_signed_formula(perfect):
    """full-valid 正常点和完美点均保持官方带符号 SDR/谱公式。"""
    generator = torch.Generator().manual_seed(72)
    clean = torch.randn(2, 96, generator=generator)
    if perfect:
        enhanced = clean.clone()
    else:
        enhanced = clean + 0.1 * torch.randn(2, 96, generator=generator)
    loss_fn = _small_loss()

    expected = _official_gap_hybrid_formula(enhanced, clean, loss_fn)
    actual = loss_fn(enhanced, clean)

    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_gap_hybrid_loss_runtime_sample_rate_overrides_constructor():
    """多采样率训练时 batch sample_rate 必须覆盖构造默认值。"""
    generator = torch.Generator().manual_seed(73)
    clean = torch.randn(2, 96, generator=generator)
    enhanced = clean + 0.1 * torch.randn(2, 96, generator=generator)
    runtime = _small_loss()

    assert GAPHybridLoss.REQUIRES_SAMPLE_RATE is True
    actual = runtime(enhanced, clean, sample_rate=2000)
    reference = GAPHybridLoss(
        sample_rate=2000, fft_len=0.064, hop_len=0.016, win_len=0.064
    )(enhanced, clean)

    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)

    from core.trainer.metrics_trainer import _waveform_loss_kwargs

    kwargs = _waveform_loss_kwargs(runtime, None, clean, enhanced, 2000)
    assert kwargs["sample_rate"] == 2000


@pytest.mark.parametrize("bad", [True, 0, -1, 1.5, "16000"])
def test_gap_hybrid_loss_rejects_invalid_runtime_sample_rate(bad):
    loss_fn = _small_loss()
    with pytest.raises((TypeError, ValueError)):
        loss_fn(torch.zeros(1, 16), torch.zeros(1, 16), sample_rate=bad)


pytestmark = [pytest.mark.unit]
