"""Multi-Band Relative Contrastive Loss（MBRCL）单元测试。"""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from core.losses.build import _StrictLossWrapper, build_loss
from core.losses.mbrcl import MultiBandRelativeContrastiveLoss
from core.losses.time_domain import L1WavLoss
from core.utils.registry import LOSS_REGISTRY
from tasks.se.data.batch import SEBatch
from tasks.se.trainer.metrics_trainer import (
    SETrainerBase,
    _loss_requires_noisy_reference,
    _loss_requires_sample_rate,
    _se_loss_kwargs,
)

pytestmark = [pytest.mark.unit]


def _small_loss(**kwargs) -> MultiBandRelativeContrastiveLoss:
    defaults = {
        "sample_rate": 16000,
        "n_fft": 64,
        "hop_length": 16,
        "win_length": 64,
        "window": "hann",
        "center": True,
        "num_bands": 4,
    }
    defaults.update(kwargs)
    return MultiBandRelativeContrastiveLoss(**defaults)


@pytest.mark.parametrize(
    ("sample_rate", "stft_cfg", "explicit_args", "expected_hop"),
    [
        (
            24000,
            {
                "n_fft": 320,
                "hop_length": 160,
                "win_length": 320,
                "window": "hamming",
                "center": False,
            },
            {"hop_length": 160},
            160,
        ),
        (
            48000,
            {
                "n_fft": 511,
                "hop_length": 127,
                "win_length": 511,
                "window": "hamming",
                "center": True,
            },
            {},
            127,
        ),
    ],
)
def test_registry_build_injects_analysis_context_and_preserves_explicit_args(
    sample_rate, stft_cfg, explicit_args, expected_hop
):
    assert LOSS_REGISTRY.get("MultiBandRelativeContrastiveLoss") is MultiBandRelativeContrastiveLoss
    loss = build_loss(
        {
            "type": "MultiBandRelativeContrastiveLoss",
            "args": {**explicit_args, "num_bands": 8, "margin": 0.1},
        },
        stft_cfg=stft_cfg,
        sample_rate=sample_rate,
    )
    assert loss.sample_rate == sample_rate
    assert (loss.n_fft, loss.hop_length, loss.win_length) == (
        stft_cfg["n_fft"],
        expected_hop,
        stft_cfg["win_length"],
    )
    assert loss.window_name == stft_cfg["window"]
    assert loss.center is stft_cfg["center"]
    assert loss.margin == pytest.approx(0.1)


def test_missing_geometry_or_effective_sample_rate_fails_clearly():
    with pytest.raises(ValueError, match=r"requires analysis STFT geometry.*n_fft"):
        MultiBandRelativeContrastiveLoss()

    loss = build_loss(
        {"type": "MultiBandRelativeContrastiveLoss", "args": {"num_bands": 4}},
        stft_cfg={"n_fft": 64, "hop_length": 16},
    )
    assert (loss.win_length, loss.window_name, loss.center) == (64, "hann", True)

    enhanced = torch.randn(1, 128)
    clean = torch.randn_like(enhanced)
    noisy = torch.randn_like(enhanced)
    with pytest.raises(ValueError, match=r"requires sample_rate.*construction.*forward"):
        loss(enhanced, clean, noisy=noisy)
    assert torch.isfinite(loss(enhanced, clean, noisy=noisy, sample_rate=16000))


def test_algorithm_defaults_remain_independent_of_analysis_context():
    loss = MultiBandRelativeContrastiveLoss(
        sample_rate=16000,
        n_fft=512,
        hop_length=128,
        win_length=512,
        window="hann",
        center=False,
    )

    assert loss.num_bands == 16
    assert loss.margin == pytest.approx(0.06)
    assert loss.eps == pytest.approx(1e-8)
    assert loss.band_weight == pytest.approx(0.5)
    assert loss.envelope_weight == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"sample_rate": 0}, ValueError),
        ({"n_fft": 31, "win_length": 32}, ValueError),
        ({"window": "blackman"}, ValueError),
        ({"num_bands": 1000}, ValueError),
        ({"margin": -0.1}, ValueError),
        ({"eps": 0.0}, ValueError),
        ({"band_weight": 0.0, "envelope_weight": 0.0}, ValueError),
        ({"center": 1}, TypeError),
    ],
)
def test_constructor_rejects_invalid_parameters(kwargs, error):
    with pytest.raises(error):
        _small_loss(**kwargs)


def test_noisy_reference_is_required_and_validated():
    loss = _small_loss()
    enhanced = torch.randn(2, 128)
    clean = torch.randn_like(enhanced)

    with pytest.raises(ValueError, match="requires noisy reference"):
        loss(enhanced, clean)
    with pytest.raises(ValueError, match="match enhanced shape"):
        loss(enhanced, clean, noisy=torch.randn(2, 127))
    with pytest.raises(TypeError, match="must be floating point"):
        loss(enhanced, clean, noisy=torch.ones(2, 128, dtype=torch.int64))
    with pytest.raises(ValueError, match="center=False requires"):
        _small_loss(center=False)(
            enhanced[:, :32], clean[:, :32], noisy=torch.randn(2, 32)
        )


def test_formula_oracle_uses_band_sum_softplus_envelope_and_weights():
    loss = _small_loss(num_bands=2, margin=0.3, band_weight=0.5, envelope_weight=0.2)
    clean = torch.tensor([[[1.0, 4.0, 9.0], [4.0, 9.0, 16.0]]])
    enhanced = torch.tensor([[[1.5, 3.0, 10.0], [5.0, 8.0, 18.0]]])
    noisy = torch.tensor([[[4.0, 8.0, 18.0], [9.0, 18.0, 32.0]]])

    actual = loss._loss_from_band_energies(enhanced, clean, noisy)
    d_enh = (enhanced.log() - clean.log()).abs().mean(dim=-1)
    d_noisy = (noisy.log() - clean.log()).abs().mean(dim=-1)
    expected_band = F.softplus(d_enh - d_noisy + 0.3).sum(dim=1)
    expected_env = (
        torch.diff(enhanced.sqrt(), dim=-1) - torch.diff(clean.sqrt(), dim=-1)
    ).abs().sum(dim=(1, 2)) / 2.0
    expected = 0.5 * expected_band + 0.2 * expected_env

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    # 明确防止误把论文的 band sum 实现成 band mean。
    assert not torch.allclose(actual, 0.5 * expected_band / 2.0 + 0.2 * expected_env)


def test_enhanced_closer_to_clean_has_lower_controlled_loss():
    loss = _small_loss(num_bands=2, envelope_weight=0.0)
    clean = torch.ones(1, 2, 3)
    noisy = torch.full_like(clean, 9.0)
    close = torch.full_like(clean, 1.1)
    far = torch.full_like(clean, 8.0)

    assert loss._loss_from_band_energies(close, clean, noisy).item() < loss._loss_from_band_energies(
        far, clean, noisy
    ).item()


def test_short_centered_silence_and_backward_are_finite_and_use_one_stft():
    loss = _small_loss()
    enhanced = torch.zeros(2, 8, requires_grad=True)
    clean = torch.zeros_like(enhanced)
    noisy = torch.zeros_like(enhanced)

    original_stft = torch.stft
    with patch("core.losses.mbrcl.torch.stft", wraps=original_stft) as stft_mock:
        value = loss(enhanced, clean, torch.tensor([1, 8]), noisy=noisy)
    stft_mock.assert_called_once()
    _, stft_kwargs = stft_mock.call_args
    assert stft_kwargs["n_fft"] == loss.n_fft
    assert stft_kwargs["hop_length"] == loss.hop_length
    assert stft_kwargs["win_length"] == loss.win_length
    torch.testing.assert_close(stft_kwargs["window"], loss._window)
    assert stft_kwargs["center"] is loss.center
    assert value.dtype == torch.float32
    assert torch.isfinite(value)
    value.backward()
    assert enhanced.grad is not None
    assert torch.isfinite(enhanced.grad).all()


def test_target_zero_band_reducer_rewards_energy_reduction_from_noisy():
    loss = _small_loss(
        num_bands=2,
        band_weight=1.0,
        envelope_weight=0.0,
    )
    clean = torch.zeros(1, 2, 4)
    noisy = torch.full_like(clean, 9.0)
    unchanged = noisy.clone()
    reduced = torch.ones_like(clean)

    unchanged_loss = loss._loss_from_band_energies(unchanged, clean, noisy)
    reduced_loss = loss._loss_from_band_energies(reduced, clean, noisy)

    assert reduced.sum() < noisy.sum()
    assert reduced_loss.item() < unchanged_loss.item()


def test_target_zero_envelope_reducer_rewards_smoother_lower_energy():
    loss = _small_loss(
        num_bands=2,
        band_weight=0.0,
        envelope_weight=1.0,
    )
    clean = torch.zeros(1, 2, 6)
    noisy = torch.full_like(clean, 9.0)
    fluctuating = torch.tensor(
        [[[1.0, 9.0, 1.0, 9.0, 1.0, 9.0]]]
    ).expand_as(clean)
    smooth = torch.full_like(clean, 4.0)

    fluctuating_loss = loss._loss_from_band_energies(fluctuating, clean, noisy)
    smooth_loss = loss._loss_from_band_energies(smooth, clean, noisy)

    assert smooth.sum() < fluctuating.sum()
    assert smooth_loss.item() < fluctuating_loss.item()


def test_target_zero_waveform_band_loss_rewards_suppression():
    loss = _small_loss(
        center=False,
        num_bands=4,
        band_weight=1.0,
        envelope_weight=0.0,
    )
    phase = 2.0 * torch.pi * 1000.0 * torch.arange(512) / 16000.0
    noisy = torch.sin(phase).unsqueeze(0)
    clean = torch.zeros_like(noisy)

    unchanged_loss = loss(noisy, clean, noisy=noisy)
    suppressed_loss = loss(0.25 * noisy, clean, noisy=noisy)

    assert suppressed_loss.item() < unchanged_loss.item()


def test_target_zero_waveform_envelope_loss_rewards_smoothing_at_lower_energy():
    loss = _small_loss(
        center=False,
        num_bands=4,
        band_weight=0.0,
        envelope_weight=1.0,
    )
    phase = 2.0 * torch.pi * 1000.0 * torch.arange(512) / 16000.0
    carrier = torch.sin(phase).unsqueeze(0)
    block_envelope = torch.tensor([0.2, 1.0]).repeat(4).repeat_interleave(64)
    fluctuating = carrier * block_envelope
    smooth_amplitude = 0.95 * block_envelope.square().mean().sqrt()
    smooth = carrier * smooth_amplitude
    clean = torch.zeros_like(carrier)
    noisy = 1.5 * carrier

    fluctuating_loss = loss(fluctuating, clean, noisy=noisy)
    smooth_loss = loss(smooth, clean, noisy=noisy)

    assert smooth.square().sum() < fluctuating.square().sum()
    assert smooth_loss.item() < fluctuating_loss.item()


@pytest.mark.parametrize(
    ("band_weight", "envelope_weight"),
    [(1.0, 0.0), (0.5, 0.2)],
    ids=["band-only", "full-mbrcl"],
)
def test_target_zero_nonzero_pure_output_gradient_descends_residual_energy(
    band_weight, envelope_weight
):
    loss = _small_loss(
        center=False,
        band_weight=band_weight,
        envelope_weight=envelope_weight,
    )
    phase = 2.0 * torch.pi * 1000.0 * torch.arange(512) / 16000.0
    noisy = torch.sin(phase).unsqueeze(0)
    clean = torch.zeros_like(noisy)
    enhanced = (0.4 * noisy).requires_grad_()

    value = loss(
        enhanced,
        clean,
        noisy=noisy,
        is_pure_noise=torch.ones(1, dtype=torch.bool),
    )
    grad = torch.autograd.grad(value, enhanced)[0]

    assert torch.isfinite(value)
    assert torch.isfinite(grad).all()
    # dot(grad, enhanced) > 0，因此沿 -grad 更新会一阶降低残余能量。
    assert torch.dot(grad.flatten(), enhanced.detach().flatten()).item() > 0.0


def _assert_target_zero_log_amplitude_scan(
    device: torch.device, autocast_dtype: torch.dtype | None = None
) -> None:
    loss = _small_loss(center=False).to(device)
    samples = torch.arange(512, device=device, dtype=torch.float32)
    waveform = torch.sin(2.0 * torch.pi * 700.0 * samples / 16000.0)
    waveform += 0.5 * torch.sin(2.0 * torch.pi * 2300.0 * samples / 16000.0)
    waveform /= waveform.abs().amax()
    amplitudes = torch.logspace(-4, 0, steps=9, device=device)
    noisy = waveform.expand(amplitudes.numel(), -1)
    clean = torch.zeros_like(noisy)
    enhanced = (amplitudes[:, None] * noisy).requires_grad_()
    pure = torch.ones(amplitudes.numel(), device=device, dtype=torch.bool)

    if autocast_dtype is None:
        value = loss(enhanced, clean, noisy=noisy, is_pure_noise=pure)
    else:
        with torch.autocast(device.type, dtype=autocast_dtype):
            value = loss(enhanced, clean, noisy=noisy, is_pure_noise=pure)
    grad = torch.autograd.grad(value, enhanced)[0]

    assert torch.isfinite(value)
    assert torch.isfinite(grad).all()
    per_sample_direction = (grad * enhanced.detach()).sum(dim=1)
    # 每个振幅样本都应沿 -grad 降低输出振幅，而非仅由 batch 聚合掩盖反向样本。
    assert torch.all(per_sample_direction > 0.0), per_sample_direction


def test_target_zero_log_amplitude_scan_is_finite_on_cpu_fp32():
    _assert_target_zero_log_amplitude_scan(torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_target_zero_log_amplitude_scan_is_finite_under_cuda_autocast():
    _assert_target_zero_log_amplitude_scan(
        torch.device("cuda"), autocast_dtype=torch.float16
    )


def test_target_zero_all_zero_equals_default_softplus_floor_and_has_zero_gradient():
    loss = MultiBandRelativeContrastiveLoss(
        sample_rate=16000,
        n_fft=512,
        hop_length=128,
        win_length=512,
        window="hann",
        center=False,
    )
    enhanced = torch.zeros(1, 1024, requires_grad=True)
    clean = torch.zeros_like(enhanced)
    noisy = torch.zeros_like(enhanced)

    actual = loss(enhanced, clean, noisy=noisy)
    expected = (
        actual.new_tensor(loss.band_weight * loss.num_bands)
        * F.softplus(actual.new_tensor(loss.margin))
    )

    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    assert actual.item() == pytest.approx(5.7887769)
    actual.backward()
    assert enhanced.grad is not None
    assert torch.count_nonzero(enhanced.grad) == 0


def test_target_zero_exact_zero_has_zero_gradient_and_softplus_floor():
    loss = _small_loss(center=False)
    phase = 2.0 * torch.pi * 1000.0 * torch.arange(512) / 16000.0
    noisy = torch.sin(phase).unsqueeze(0)
    clean = torch.zeros_like(noisy)
    enhanced = torch.zeros_like(noisy, requires_grad=True)

    actual = loss(
        enhanced,
        clean,
        noisy=noisy,
        is_pure_noise=torch.ones(1, dtype=torch.bool),
    )
    grad = torch.autograd.grad(actual, enhanced)[0]

    noisy_spectra = torch.stft(
        noisy,
        n_fft=loss.n_fft,
        hop_length=loss.hop_length,
        win_length=loss.win_length,
        window=loss._window,
        center=loss.center,
        return_complex=True,
        onesided=True,
    )
    noisy_energy = torch.einsum(
        "bf,nft->nbt", loss._band_matrix, noisy_spectra.abs().square()
    )
    d_noisy = (
        torch.log(noisy_energy + loss.eps) - math.log(loss.eps)
    ).abs().mean(dim=-1)
    expected = loss.band_weight * F.softplus(
        loss.margin - d_noisy
    ).sum(dim=1).mean()

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert torch.count_nonzero(grad) == 0


def test_enhanced_zero_clean_nonzero_is_high_loss_but_has_zero_power_gradient():
    loss = _small_loss(center=False, band_weight=1.0, envelope_weight=0.0)
    samples = torch.arange(512, dtype=torch.float32)
    clean = torch.sin(2.0 * torch.pi * 1000.0 * samples / 16000.0).unsqueeze(0)
    noisy = clean + 0.5 * torch.sin(
        2.0 * torch.pi * 2500.0 * samples / 16000.0
    ).unsqueeze(0)
    enhanced = torch.zeros_like(clean, requires_grad=True)

    actual = loss(enhanced, clean, noisy=noisy)
    grad = torch.autograd.grad(actual, enhanced)[0]

    assert actual.item() > loss.num_bands * F.softplus(
        actual.new_tensor(loss.margin)
    ).item()
    # 论文使用 |STFT(enhanced)|²；enhanced=0 是高 loss 边界，但功率公式导数为零。
    # 此测试固定论文语义，不擅自用线性幅度或额外项改写公式来制造梯度。
    assert torch.count_nonzero(grad) == 0


def test_single_loss_all_pure_target_zero_batch_is_finite_and_backwardable():
    loss = _small_loss()
    assert "REQUIRES_SPEECH_REFERENCE" not in type(loss).__dict__
    enhanced = torch.randn(3, 128, requires_grad=True)
    clean = torch.zeros_like(enhanced)
    noisy = torch.randn_like(enhanced)
    lengths = torch.tensor([128, 112, 96])

    value = loss(
        enhanced,
        clean,
        lengths,
        noisy=noisy,
        is_pure_noise=torch.tensor([True, False, False]),
        is_pure_echo=torch.tensor([False, True, False]),
        is_pure_echo_noise=torch.tensor([False, False, True]),
    )

    assert value.item() > 0.0
    assert torch.isfinite(value)
    value.backward()
    assert enhanced.grad is not None
    assert torch.isfinite(enhanced.grad).all()


def _expected_band_matrix(sample_rate: int, n_fft: int, num_bands: int) -> torch.Tensor:
    frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    frequency_mels = 2595.0 * torch.log10(1.0 + frequencies / 700.0)
    nyquist_mel = 2595.0 * math.log10(1.0 + (sample_rate / 2.0) / 700.0)
    mel_edges = torch.linspace(0.0, nyquist_mel, num_bands + 1)
    expected_index = torch.bucketize(frequency_mels, mel_edges[1:-1], right=True)
    return F.one_hot(expected_index, num_classes=num_bands).T.float()


def test_odd_fft_uses_exact_rfft_bin_frequencies():
    loss = _small_loss(n_fft=63, win_length=63)
    frequencies = torch.fft.rfftfreq(63, d=1.0 / loss.sample_rate)

    assert frequencies[-1].item() < loss.sample_rate / 2.0
    assert torch.equal(
        loss._band_matrix,
        _expected_band_matrix(loss.sample_rate, loss.n_fft, loss.num_bands),
    )


def _independent_forward_oracle(
    loss: MultiBandRelativeContrastiveLoss,
    enhanced: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    signals = torch.cat((enhanced, clean, noisy), dim=0).float()
    window = (
        torch.hamming_window(loss.win_length)
        if loss.window_name == "hamming"
        else torch.hann_window(loss.win_length)
    )
    spectra = torch.stft(
        signals,
        n_fft=loss.n_fft,
        hop_length=loss.hop_length,
        win_length=loss.win_length,
        window=window,
        center=loss.center,
        return_complex=True,
        onesided=True,
    )
    energies = torch.einsum(
        "bf,nft->nbt",
        _expected_band_matrix(sample_rate, loss.n_fft, loss.num_bands),
        spectra.abs().square(),
    )
    enhanced_energy, clean_energy, noisy_energy = energies.chunk(3, dim=0)
    log_clean = torch.log(clean_energy + loss.eps)
    d_enhanced = (
        torch.log(enhanced_energy + loss.eps) - log_clean
    ).abs().mean(dim=-1)
    d_noisy = (
        torch.log(noisy_energy + loss.eps) - log_clean
    ).abs().mean(dim=-1)
    band_term = F.softplus(d_enhanced - d_noisy + loss.margin).sum(dim=1)
    envelope_term = (
        torch.diff(torch.sqrt(enhanced_energy + loss.eps), dim=-1)
        - torch.diff(torch.sqrt(clean_energy + loss.eps), dim=-1)
    ).abs().sum(dim=(1, 2)) / (enhanced_energy.shape[-1] - 1)
    return (loss.band_weight * band_term + loss.envelope_weight * envelope_term).mean()


def test_runtime_sample_rate_overrides_constructor_and_caches_matching_bands():
    loss = _small_loss(
        sample_rate=16000,
        n_fft=255,
        hop_length=64,
        win_length=255,
        center=False,
    )
    generator = torch.Generator().manual_seed(13)
    enhanced = torch.randn(1, 512, generator=generator, requires_grad=True)
    clean = torch.randn(1, 512, generator=generator)
    noisy = torch.randn(1, 512, generator=generator)

    value_24k = loss(enhanced, clean, noisy=noisy, sample_rate=24000)
    oracle_24k = _independent_forward_oracle(loss, enhanced, clean, noisy, 24000)
    oracle_16k = _independent_forward_oracle(loss, enhanced, clean, noisy, 16000)
    matrix_24k = loss._band_matrix_for_sample_rate(24000)
    matrix_16k = loss._band_matrix_for_sample_rate(16000)

    assert torch.equal(matrix_24k, _expected_band_matrix(24000, 255, 4))
    assert torch.equal(matrix_16k, _expected_band_matrix(16000, 255, 4))
    assert not torch.equal(matrix_24k, matrix_16k)
    torch.testing.assert_close(value_24k, oracle_24k, atol=1e-6, rtol=1e-6)
    assert not torch.isclose(value_24k, oracle_16k, atol=1e-5, rtol=1e-5)
    assert matrix_24k.device.type == "cpu"
    assert matrix_24k.dtype == torch.float32
    assert loss._band_matrix_for_sample_rate(24000).data_ptr() == matrix_24k.data_ptr()
    assert torch.isfinite(value_24k)
    value_24k.backward()
    assert torch.isfinite(enhanced.grad).all()


@pytest.mark.parametrize("module_dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_module_dtype_conversion_keeps_fp32_power_projection_compatible(module_dtype):
    loss = _small_loss().to(dtype=module_dtype)
    enhanced = torch.randn(1, 128, requires_grad=True)
    clean = torch.randn_like(enhanced)
    noisy = torch.randn_like(enhanced)

    value = loss(enhanced, clean, noisy=noisy)

    assert value.dtype == torch.float32
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(enhanced.grad).all()


def test_runtime_sample_rate_cache_builds_once_under_concurrency():
    loss = _small_loss(sample_rate=16000, n_fft=255, hop_length=64, win_length=255)
    original_build = loss._build_band_matrix
    build_count = 0
    count_lock = threading.Lock()

    def counted_build(sample_rate):
        nonlocal build_count
        with count_lock:
            build_count += 1
        time.sleep(0.02)
        return original_build(sample_rate)

    with patch.object(loss, "_build_band_matrix", side_effect=counted_build):
        with ThreadPoolExecutor(max_workers=8) as executor:
            matrices = list(
                executor.map(loss._band_matrix_for_sample_rate, [24000] * 16)
            )

    assert build_count == 1
    assert len({matrix.data_ptr() for matrix in matrices}) == 1
    assert matrices[0].device.type == "cpu"
    assert matrices[0].dtype == torch.float32
    assert all("runtime" not in key and "lock" not in key for key in loss.state_dict())


def test_realistic_empty_mel_band_reports_geometry_at_construction_and_runtime():
    with pytest.raises(
        ValueError,
        match=r"empty bands=.*sample_rate=16000, n_fft=32, rfft_bins=17.*num_bands=16",
    ):
        _small_loss(n_fft=32, hop_length=8, win_length=32, num_bands=16)

    loss = _small_loss(
        sample_rate=16000,
        n_fft=16,
        hop_length=4,
        win_length=16,
        num_bands=5,
    )
    waveform = torch.randn(1, 64)
    with pytest.raises(
        ValueError,
        match=r"empty bands=.*sample_rate=48000, n_fft=16, rfft_bins=9.*num_bands=5",
    ):
        loss(waveform, waveform, noisy=waveform, sample_rate=48000)


def _padding_case(
    total: int, geometry: dict[str, object]
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = 181
    generator = torch.Generator().manual_seed(7)
    clean_valid = torch.randn(valid, generator=generator)
    noisy_valid = clean_valid + 0.3 * torch.randn(valid, generator=generator)
    enhanced_valid = clean_valid + 0.1 * torch.randn(valid, generator=generator)

    def padded(x: torch.Tensor) -> torch.Tensor:
        suffix = 10.0 * torch.randn(total - valid, generator=generator)
        return torch.cat((x, suffix)).unsqueeze(0)

    enhanced = padded(enhanced_valid).requires_grad_(True)
    loss = _small_loss(**geometry)(
        enhanced,
        padded(clean_valid),
        torch.tensor([valid]),
        noisy=padded(noisy_valid),
    )
    loss.backward()
    return loss.detach(), enhanced.grad.detach()


@pytest.mark.parametrize(
    "geometry",
    [
        {
            "n_fft": 64,
            "hop_length": 16,
            "win_length": 64,
            "window": "hann",
            "center": True,
        },
        {
            "n_fft": 63,
            "hop_length": 21,
            "win_length": 63,
            "window": "hamming",
            "center": False,
        },
    ],
    ids=["centered-even-hann", "uncentered-odd-hamming"],
)
def test_two_geometries_forward_backward_and_padding_invariance(geometry):
    reference_loss, reference_grad = _padding_case(256, geometry)
    padded_loss, padded_grad = _padding_case(512, geometry)

    assert torch.allclose(padded_loss, reference_loss, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        padded_grad[:, :181], reference_grad[:, :181], atol=1e-6, rtol=1e-5
    )
    assert torch.count_nonzero(reference_grad[:, 181:]) == 0
    assert torch.count_nonzero(padded_grad[:, 181:]) == 0


def test_composite_sisnr_uses_non_pure_while_mbrcl_and_mrstft_use_full_batch():
    loss = build_loss(
        {
            "type": "CompositeLoss",
            "components": [
                {"type": "SISNRLoss", "weight": 1.0},
                {
                    "type": "MultiBandRelativeContrastiveLoss",
                    "weight": 1.0,
                    "args": {"num_bands": 4},
                },
                {
                    "type": "MultiResSTFTLoss",
                    "weight": 1.0,
                    "args": {
                        "fft_sizes": [64],
                        "hop_sizes": [16],
                        "win_lengths": [64],
                    },
                },
            ],
        },
        stft_cfg={
            "n_fft": 64,
            "hop_length": 16,
            "win_length": 64,
            "window": "hann",
            "center": True,
        },
        sample_rate=16000,
    )
    generator = torch.Generator().manual_seed(41)
    clean = torch.randn(2, 256, generator=generator)
    clean[1].zero_()
    noisy = clean + 0.7 * torch.randn(2, 256, generator=generator)
    enhanced = (0.6 * noisy).requires_grad_()
    pure = torch.tensor([False, True])

    actual = loss(enhanced, clean, noisy=noisy, is_pure_noise=pure)
    expected_sisnr = loss.losses[0](enhanced[:1], clean[:1])
    expected_mbrcl = loss.losses[1](enhanced, clean, noisy=noisy)
    expected_mrstft = loss.losses[2](enhanced, clean)
    subset_mbrcl = loss.losses[1](enhanced[:1], clean[:1], noisy=noisy[:1])
    subset_mrstft = loss.losses[2](enhanced[:1], clean[:1])

    assert torch.allclose(actual, expected_sisnr + expected_mbrcl + expected_mrstft)
    assert not torch.allclose(expected_mbrcl, subset_mbrcl)
    assert not torch.allclose(expected_mrstft, subset_mrstft)
    assert torch.isfinite(actual)
    actual.backward()
    assert enhanced.grad is not None
    assert torch.isfinite(enhanced.grad).all()


def test_mbrcl_is_not_speech_reference_dependent_by_capability_or_class_name():
    from core.losses.composite import _SPEECH_REF_DEPENDENT_CLASS_NAMES

    assert "REQUIRES_SPEECH_REFERENCE" not in MultiBandRelativeContrastiveLoss.__dict__
    assert "MultiBandRelativeContrastiveLoss" not in _SPEECH_REF_DEPENDENT_CLASS_NAMES


def test_strict_wrapper_capability_and_se_kwargs_only_inject_active_context():
    class ExplicitFalseCapabilityLoss(L1WavLoss):
        REQUIRES_NOISY_REFERENCE = False
        REQUIRES_SAMPLE_RATE = False

    def build_composite(mbrcl_weight: float):
        return build_loss(
            {
                "type": "CompositeLoss",
                "components": [
                    {"type": "L1WavLoss", "weight": 1.0},
                    {
                        "type": "MultiBandRelativeContrastiveLoss",
                        "weight": mbrcl_weight,
                        "args": {"num_bands": 4},
                    },
                ],
            },
            stft_cfg={"n_fft": 64, "hop_length": 16},
        )

    mbrcl = _small_loss()
    inactive_composite = build_composite(0.0)
    active_composite = build_composite(1.0)
    explicit_false = ExplicitFalseCapabilityLoss()

    assert _loss_requires_noisy_reference(mbrcl)
    assert _loss_requires_sample_rate(mbrcl)
    assert _loss_requires_noisy_reference(_StrictLossWrapper(mbrcl))
    assert not _loss_requires_noisy_reference(explicit_false)
    assert not _loss_requires_sample_rate(_StrictLossWrapper(explicit_false))
    assert not _loss_requires_noisy_reference(inactive_composite)
    assert not _loss_requires_sample_rate(inactive_composite)
    assert _loss_requires_noisy_reference(_StrictLossWrapper(active_composite))
    assert _loss_requires_sample_rate(_StrictLossWrapper(active_composite))

    batch = SEBatch(
        clean=torch.randn(2, 128),
        noisy=torch.randn(2, 128),
        sr=torch.tensor([24000, 24000]),
        lengths=torch.tensor([128, 112]),
        is_pure_noise=torch.tensor([False, True]),
        is_pure_echo=torch.zeros(2, dtype=torch.bool),
        is_pure_echo_noise=torch.zeros(2, dtype=torch.bool),
    )
    enhanced = torch.randn_like(batch.clean, requires_grad=True)
    assert _se_loss_kwargs(L1WavLoss(), batch, enhanced, 24000) == {}
    kwargs = _se_loss_kwargs(mbrcl, batch, enhanced, 24000)
    assert kwargs["noisy"] is batch.noisy
    assert kwargs["sample_rate"] == 24000
    assert "is_pure_noise" not in kwargs

    strict_mbrcl = _StrictLossWrapper(mbrcl)
    strict_kwargs = _se_loss_kwargs(strict_mbrcl, batch, enhanced, 24000)
    assert "is_pure_noise" not in strict_kwargs
    actual = strict_mbrcl(enhanced, batch.clean, batch.lengths, **strict_kwargs)
    expected = mbrcl(
        enhanced,
        batch.clean,
        batch.lengths,
        noisy=batch.noisy,
        sample_rate=24000,
    )
    assert torch.allclose(actual, expected)


def test_se_trainer_train_and_validation_pass_actual_batch_sample_rate():
    class ScaleModel(torch.nn.Module):
        def forward(self, noisy, lengths):
            return 0.5 * noisy

    trainer = object.__new__(SETrainerBase)
    trainer.loss_fn = build_loss(
        {
            "type": "CompositeLoss",
            "components": [
                {
                    "type": "MultiBandRelativeContrastiveLoss",
                    "weight": 1.0,
                    "args": {"num_bands": 4},
                }
            ],
        },
        stft_cfg={"n_fft": 64, "hop_length": 16},
    )
    trainer.model = ScaleModel()
    trainer.stft_module = None
    trainer._model_accepts_fs = False
    trainer._model_accepts_return_lsnr = False
    trainer._model_accepts_return_alpha = False
    trainer._use_lsnr_loss = False
    trainer._loss_accepts_lsnr = False
    trainer._loss_needs_predicted_alpha = False
    trainer._strict_checks = False
    trainer._sanitize_model_output = lambda value: value
    trainer.device = torch.device("cpu")
    trainer.use_amp = False
    trainer.amp_dtype = torch.float32
    trainer.sample_rate = 16000

    batch_24k = SEBatch(
        clean=torch.randn(1, 128),
        noisy=torch.randn(1, 128),
        sr=torch.tensor([24000]),
        lengths=torch.tensor([128]),
    )
    train_loss, finite, _ = trainer._forward_and_compute_loss(
        batch_24k, epoch=0, batch_idx=0, fs_val=24000
    )
    assert finite and torch.isfinite(train_loss)

    batch_8k = SEBatch(
        clean=torch.randn(1, 128),
        noisy=torch.randn(1, 128),
        sr=torch.tensor([8000]),
        lengths=torch.tensor([128]),
    )
    val_loss, *_, fs_val, _ = trainer._val_forward_loss(batch_8k, epoch=0, n_loops=0)
    mbrcl = trainer.loss_fn.losses[0]
    assert fs_val == 8000
    assert torch.isfinite(val_loss)
    assert 24000 in mbrcl._runtime_band_matrices
    assert 8000 in mbrcl._runtime_band_matrices


def test_envelope_is_graph_connected_zero_for_fewer_than_two_frames():
    loss = _small_loss(num_bands=2, band_weight=0.0, envelope_weight=1.0)
    enhanced = torch.ones(1, 2, 1, requires_grad=True)
    clean = torch.ones_like(enhanced)
    noisy = torch.full_like(enhanced, 2.0)

    value = loss._loss_from_band_energies(enhanced, clean, noisy).mean()
    assert value.item() == 0.0
    value.backward()
    assert enhanced.grad is not None
    assert torch.count_nonzero(enhanced.grad) == 0


def _forward_value_and_grad(
    loss: MultiBandRelativeContrastiveLoss,
    enhanced: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    enhanced = enhanced.detach().clone().requires_grad_(True)
    value = loss(enhanced, clean, noisy=noisy)
    grad = torch.autograd.grad(value, enhanced)[0]
    return value.detach(), grad.detach()


def test_mixed_waveform_dtypes_match_explicit_fp32_baseline():
    generator = torch.Generator().manual_seed(19)
    enhanced = torch.randn(2, 128, generator=generator, dtype=torch.float64)
    clean = torch.randn(2, 128, generator=generator, dtype=torch.float16)
    noisy = torch.randn(2, 128, generator=generator, dtype=torch.bfloat16)
    loss = _small_loss()

    actual, actual_grad = _forward_value_and_grad(loss, enhanced, clean, noisy)
    expected, expected_grad = _forward_value_and_grad(
        loss,
        enhanced.float(),
        clean.float(),
        noisy.float(),
    )

    assert actual.dtype == torch.float32
    assert actual_grad.dtype == enhanced.dtype
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_grad.float(), expected_grad, atol=1e-6, rtol=1e-6)


def test_cpu_autocast_keeps_mbrcl_fp32_and_matches_fp32_baseline():
    generator = torch.Generator().manual_seed(23)
    enhanced = torch.randn(2, 128, generator=generator)
    clean = torch.randn(2, 128, generator=generator)
    noisy = torch.randn(2, 128, generator=generator)
    loss = _small_loss()

    expected, expected_grad = _forward_value_and_grad(loss, enhanced, clean, noisy)
    actual_enhanced = enhanced.detach().clone().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = loss(actual_enhanced, clean, noisy=noisy)
    actual_grad = torch.autograd.grad(actual, actual_enhanced)[0]

    assert actual.dtype == torch.float32
    assert actual_grad.dtype == torch.float32
    assert torch.allclose(actual.detach(), expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_grad, expected_grad, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_amp_keeps_mbrcl_fp32_and_matches_fp32_baseline():
    generator = torch.Generator(device="cuda").manual_seed(29)
    enhanced = torch.randn(2, 128, generator=generator, device="cuda")
    clean = torch.randn(2, 128, generator=generator, device="cuda")
    noisy = torch.randn(2, 128, generator=generator, device="cuda")
    loss = _small_loss().cuda()

    expected, expected_grad = _forward_value_and_grad(loss, enhanced, clean, noisy)
    actual_enhanced = enhanced.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.float16):
        actual = loss(actual_enhanced, clean, noisy=noisy)
    actual_grad = torch.autograd.grad(actual, actual_enhanced)[0]

    assert actual.dtype == torch.float32
    assert actual_grad.dtype == torch.float32
    assert torch.allclose(actual.detach(), expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_grad, expected_grad, atol=1e-6, rtol=1e-6)


def test_extremely_short_centered_full_lengths_match_implicit_full_valid():
    generator = torch.Generator().manual_seed(31)
    enhanced = torch.randn(2, 8, generator=generator)
    clean = torch.randn(2, 8, generator=generator)
    noisy = torch.randn(2, 8, generator=generator)
    loss = _small_loss()

    implicit, implicit_grad = _forward_value_and_grad(loss, enhanced, clean, noisy)
    explicit_enhanced = enhanced.detach().clone().requires_grad_(True)
    explicit = loss(
        explicit_enhanced,
        clean,
        torch.full((2,), enhanced.shape[-1], dtype=torch.long),
        noisy=noisy,
    )
    explicit_grad = torch.autograd.grad(explicit, explicit_enhanced)[0]

    assert implicit.abs().item() > 0.0
    assert torch.count_nonzero(implicit_grad) > 0
    assert torch.allclose(implicit, explicit.detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(implicit_grad, explicit_grad, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("operation", ["pickle", "deepcopy"])
def test_pickle_and_deepcopy_recreate_runtime_cache_and_lock(operation):
    import copy
    import pickle

    loss = _small_loss(n_fft=255, hop_length=64, win_length=255)
    source_matrix = loss._band_matrix_for_sample_rate(24000)

    if operation == "pickle":
        clone = pickle.loads(pickle.dumps(loss))
    else:
        clone = copy.deepcopy(loss)

    assert loss._runtime_band_matrices[24000] is source_matrix
    assert clone._runtime_band_matrices == {}
    assert clone._runtime_band_matrices_lock is not loss._runtime_band_matrices_lock
    rebuilt = clone._band_matrix_for_sample_rate(24000)
    assert torch.equal(rebuilt, source_matrix)
    assert rebuilt is not source_matrix


@pytest.mark.parametrize(
    "corrupt_cache",
    [
        None,
        [],
        {24000: None},
        {24000: torch.empty(1)},
        {24000: torch.empty(4, 128, dtype=torch.float64)},
    ],
)
def test_corrupt_runtime_cache_is_rebuilt(corrupt_cache):
    loss = _small_loss(n_fft=255, hop_length=64, win_length=255)
    loss._runtime_band_matrices = corrupt_cache

    with patch.object(loss, "_build_band_matrix", wraps=loss._build_band_matrix) as build:
        matrix = loss._band_matrix_for_sample_rate(24000)

    assert build.call_count == 1
    assert torch.equal(matrix, _expected_band_matrix(24000, 255, 4))
    assert loss._runtime_band_matrices == {24000: matrix}
    assert matrix.device.type == "cpu"
    assert matrix.dtype == torch.float32


def test_matching_stft_context_is_declared_on_mbrcl_class():
    assert (
        MultiBandRelativeContrastiveLoss.__dict__["REQUIRES_MATCHING_STFT_CONTEXT"]
        is True
    )
