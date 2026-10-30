"""GTCRN 压缩谱与 log-ratio 损失 direct-spectrum 集成回归测试。"""

import pytest
import torch

from core.losses.composite import CompositeLoss
from core.losses.hybrid import GTCRNCompressedSpectralLoss, GTCRNLogRatioLoss
from core.losses.utils import compute_spec_frame_lengths
from core.stft import StftIstftModule
from core.trainer.metrics_trainer import _direct_spectral_loss_kwargs


pytestmark = [pytest.mark.unit]


def _oracle(
    predicted: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    compress_factor: float = 0.3,
    lamda_ri: float = 30.0,
    lamda_mag: float = 70.0,
) -> torch.Tensor:
    predicted = torch.view_as_complex(predicted.float().contiguous())
    target = torch.view_as_complex(target.float().contiguous())
    predicted_mag = torch.sqrt(predicted.real.square() + predicted.imag.square() + 1e-12)
    target_mag = torch.sqrt(target.real.square() + target.imag.square() + 1e-12)
    predicted_compressed = predicted / predicted_mag ** (1.0 - compress_factor)
    target_compressed = target / target_mag ** (1.0 - compress_factor)

    frame_lengths = torch.div(lengths, hop_length, rounding_mode="floor") + 1
    frame_index = torch.arange(predicted.shape[-1], device=predicted.device)
    mask = (frame_index[None, :] < frame_lengths[:, None]).to(predicted.real.dtype).unsqueeze(1)
    denominator = (mask.sum() * predicted.shape[1]).clamp(min=1.0)
    real_loss = (((predicted_compressed.real - target_compressed.real) ** 2) * mask).sum() / denominator
    imag_loss = (((predicted_compressed.imag - target_compressed.imag) ** 2) * mask).sum() / denominator
    magnitude_loss = (((predicted_mag**compress_factor - target_mag**compress_factor) ** 2) * mask).sum() / denominator
    return lamda_ri * (real_loss + imag_loss) + lamda_mag * magnitude_loss


def _inconsistent_spectra() -> tuple[
    GTCRNCompressedSpectralLoss,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(17)
    n_fft, hop_length, samples = 32, 8, 96
    stft = StftIstftModule(n_fft, hop_length, n_fft, window="hann", center=True)
    clean = torch.randn(2, samples)
    lengths = torch.tensor([samples, samples - 13], dtype=torch.long)
    clean[1, lengths[1] :] = 0.0
    target_spectra = stft(clean)

    # 任意复谱通常不满足 STFT consistency；iSTFT→STFT 会投影到一致谱子空间。
    predicted_spectra = torch.randn_like(target_spectra).requires_grad_()
    enhanced = stft.inverse(predicted_spectra, samples)
    reanalyzed = stft(enhanced)
    assert not torch.allclose(predicted_spectra, reanalyzed)

    loss_fn = GTCRNCompressedSpectralLoss(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        center=True,
    )
    return loss_fn, enhanced, clean, lengths, predicted_spectra, target_spectra


def test_gtcrn_direct_spectrum_matches_oracle_not_waveform_reanalysis():
    loss_fn, enhanced, clean, lengths, predicted_spectra, target_spectra = _inconsistent_spectra()

    direct = loss_fn(
        enhanced,
        clean,
        lengths,
        enhanced_spectra=predicted_spectra,
        target_spectra=target_spectra,
    )
    oracle = _oracle(
        predicted_spectra,
        target_spectra,
        lengths,
        n_fft=loss_fn.n_fft,
        hop_length=loss_fn.hop_length,
    )
    waveform_reanalysis = loss_fn(enhanced, clean, lengths)

    assert torch.allclose(direct, oracle, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(direct, waveform_reanalysis, rtol=1e-4, atol=1e-4)
    direct.backward()
    assert predicted_spectra.grad is not None


def test_composite_routes_direct_spectra_only_to_gtcrn_component():
    loss_fn, enhanced, clean, lengths, predicted_spectra, target_spectra = _inconsistent_spectra()
    composite = CompositeLoss(
        components=[
            {
                "type": "GTCRNCompressedSpectralLoss",
                "weight": 1.0,
                "args": {
                    "n_fft": loss_fn.n_fft,
                    "hop_length": loss_fn.hop_length,
                    "win_length": loss_fn.win_length,
                },
            },
            {"type": "L1WavLoss", "weight": 0.25},
        ]
    )

    actual = composite(
        enhanced,
        clean,
        lengths,
        enhanced_spectra=predicted_spectra,
        target_spectra=target_spectra,
    )
    expected = _oracle(
        predicted_spectra,
        target_spectra,
        lengths,
        n_fft=loss_fn.n_fft,
        hop_length=loss_fn.hop_length,
    ) + 0.25 * composite.losses[1](enhanced, clean, lengths)
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_gtcrn_without_spectrum_context_keeps_waveform_adapter():
    loss_fn, enhanced, clean, lengths, _, _ = _inconsistent_spectra()
    waveform_loss = loss_fn(enhanced, clean, lengths)
    assert torch.isfinite(waveform_loss)


@pytest.mark.parametrize("missing", ["enhanced_spectra", "target_spectra"])
def test_gtcrn_rejects_partial_spectrum_context(missing: str):
    loss_fn, enhanced, clean, lengths, predicted_spectra, target_spectra = _inconsistent_spectra()
    kwargs = {
        "enhanced_spectra": predicted_spectra,
        "target_spectra": target_spectra,
    }
    kwargs.pop(missing)
    with pytest.raises(ValueError, match="must be provided together"):
        loss_fn(enhanced, clean, lengths, **kwargs)


def test_gtcrn_direct_spectrum_masks_padded_frames():
    loss_fn, enhanced, clean, lengths, predicted_spectra, target_spectra = _inconsistent_spectra()
    baseline = loss_fn(
        enhanced,
        clean,
        lengths,
        enhanced_spectra=predicted_spectra,
        target_spectra=target_spectra,
    )
    changed = predicted_spectra.detach().clone()
    valid_frames = int(lengths[1].item()) // loss_fn.hop_length + 1
    changed[1, :, valid_frames:, :] += 1000.0
    masked = loss_fn(
        enhanced,
        clean,
        lengths,
        enhanced_spectra=changed,
        target_spectra=target_spectra,
    )
    assert torch.allclose(masked, baseline, rtol=1e-6, atol=1e-6)


def test_gtcrn_composite_keeps_pure_rows_in_full_batch():
    loss_fn, enhanced, clean, lengths, predicted_spectra, target_spectra = _inconsistent_spectra()
    composite = CompositeLoss(
        components=[
            {
                "type": "GTCRNCompressedSpectralLoss",
                "weight": 1.0,
                "args": {
                    "n_fft": loss_fn.n_fft,
                    "hop_length": loss_fn.hop_length,
                    "win_length": loss_fn.win_length,
                },
            }
        ]
    )
    pure_flags = torch.tensor([True, False])
    actual = composite(
        enhanced,
        clean,
        lengths,
        enhanced_spectra=predicted_spectra,
        target_spectra=target_spectra,
        is_pure_noise=pure_flags,
    )
    expected = _oracle(
        predicted_spectra,
        target_spectra,
        lengths,
        n_fft=loss_fn.n_fft,
        hop_length=loss_fn.hop_length,
    )
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_direct_spectral_context_uses_framework_stft_for_target():
    loss_fn, _, clean, _, predicted_spectra, _ = _inconsistent_spectra()
    framework_stft = StftIstftModule(
        loss_fn.n_fft,
        loss_fn.hop_length,
        loss_fn.win_length,
        window=loss_fn.window_name,
        center=loss_fn.center,
    )
    kwargs = _direct_spectral_loss_kwargs(
        loss_fn,
        framework_stft,
        clean,
        predicted_spectra,
    )
    assert kwargs["enhanced_spectra"] is predicted_spectra
    assert torch.equal(kwargs["target_spectra"], framework_stft(clean))


def test_direct_spectral_context_rejects_stft_geometry_mismatch():
    loss_fn, _, clean, _, predicted_spectra, _ = _inconsistent_spectra()
    mismatched_stft = StftIstftModule(
        loss_fn.n_fft,
        loss_fn.hop_length // 2,
        loss_fn.win_length,
        window=loss_fn.window_name,
        center=loss_fn.center,
    )
    with pytest.raises(ValueError, match="must match the framework STFT"):
        _direct_spectral_loss_kwargs(
            loss_fn,
            mismatched_stft,
            clean,
            predicted_spectra,
        )


def test_direct_spectral_context_rejects_ss_mixing_without_undesired_spectra():
    loss_fn, _, clean, _, predicted_spectra, _ = _inconsistent_spectra()
    composite = CompositeLoss(
        components=[
            {
                "type": "GTCRNCompressedSpectralLoss",
                "weight": 1.0,
                "enable_ss_undesired_energy": True,
                "args": {
                    "n_fft": loss_fn.n_fft,
                    "hop_length": loss_fn.hop_length,
                    "win_length": loss_fn.win_length,
                },
            }
        ]
    )
    framework_stft = StftIstftModule(
        loss_fn.n_fft,
        loss_fn.hop_length,
        loss_fn.win_length,
        window=loss_fn.window_name,
        center=loss_fn.center,
    )
    with pytest.raises(ValueError, match="does not support enable_ss_undesired_energy"):
        _direct_spectral_loss_kwargs(
            composite,
            framework_stft,
            clean,
            predicted_spectra,
        )


def test_direct_spectrum_target_loss_and_valid_gradient_ignore_batch_padding():
    """真实 reflect 边界、loss 与有效预测谱梯度均不得依赖 batch canvas。"""
    torch.manual_seed(1701)
    valid, n_fft, hop_length = 17, 16, 4
    valid_frames = valid // hop_length + 1
    loss_fn = GTCRNCompressedSpectralLoss(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        center=True,
    )
    framework_stft = StftIstftModule(
        n_fft,
        hop_length,
        n_fft,
        window="hann",
        center=True,
    )
    clean_valid = torch.randn(1, valid)
    predicted_valid = torch.randn(1, n_fft // 2 + 1, valid_frames, 2)
    trim_target = framework_stft(clean_valid)
    reference_loss = None
    reference_gradient = None

    for extra in (0, 1, n_fft // 2):
        clean = torch.cat((clean_valid, torch.full((1, extra), 23.0)), dim=-1)
        lengths = torch.tensor([valid], dtype=torch.long)
        frame_count = clean.shape[-1] // hop_length + 1
        predicted = torch.cat(
            (
                predicted_valid,
                torch.randn(
                    1,
                    n_fft // 2 + 1,
                    frame_count - valid_frames,
                    2,
                ),
            ),
            dim=-2,
        ).requires_grad_()

        kwargs = _direct_spectral_loss_kwargs(
            loss_fn,
            framework_stft,
            clean,
            predicted,
            lengths=lengths,
        )
        assert torch.allclose(
            kwargs["target_spectra"][..., :valid_frames, :],
            trim_target,
            rtol=1e-6,
            atol=1e-7,
        )

        enhanced = torch.zeros_like(clean)
        loss = loss_fn(enhanced, clean, lengths, **kwargs)
        gradient = torch.autograd.grad(loss, predicted)[0]
        if reference_loss is None:
            reference_loss = loss.detach()
            reference_gradient = gradient.detach()
        else:
            assert torch.allclose(loss, reference_loss, rtol=1e-6, atol=1e-7)
            assert torch.allclose(
                gradient[..., :valid_frames, :],
                reference_gradient,
                rtol=1e-5,
                atol=1e-7,
            )
        assert torch.count_nonzero(gradient[..., valid_frames:, :]) == 0


def _frame_lengths_for(
    spectra: torch.Tensor,
    lengths: torch.Tensor | None,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    """与 loss 内部一致的有效帧数：lengths=None 取全部谱帧，否则按框架契约计算。"""
    if lengths is None:
        return torch.full(
            (spectra.shape[0],),
            spectra.shape[-2],
            device=spectra.device,
            dtype=torch.long,
        )
    return compute_spec_frame_lengths(
        lengths,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
    ).long()


def _compressed_spectral_component_oracle(
    predicted: torch.Tensor,
    target: torch.Tensor,
    frame_lengths: torch.Tensor,
) -> torch.Tensor:
    """30/70 压缩 RI 与 magnitude 组件 oracle（与合成窗无关，两种窗语义共用）。"""
    predicted = torch.view_as_complex(predicted.float().contiguous())
    target = torch.view_as_complex(target.float().contiguous())
    pred_mag = torch.sqrt(predicted.real.square() + predicted.imag.square() + 1e-12)
    true_mag = torch.sqrt(target.real.square() + target.imag.square() + 1e-12)
    frame_index = torch.arange(predicted.shape[-1], device=predicted.device)
    frame_mask = (frame_index[None, :] < frame_lengths[:, None]).to(
        predicted.real.dtype
    ).unsqueeze(1)
    denominator = (frame_mask.sum() * predicted.shape[1]).clamp(min=1.0)
    real_loss = (
        ((predicted.real / pred_mag**0.7) - (target.real / true_mag**0.7)).square()
        * frame_mask
    ).sum() / denominator
    imag_loss = (
        ((predicted.imag / pred_mag**0.7) - (target.imag / true_mag**0.7)).square()
        * frame_mask
    ).sum() / denominator
    mag_loss = (
        (pred_mag**0.3 - true_mag**0.3).square() * frame_mask
    ).sum() / denominator
    return 30.0 * (real_loss + imag_loss) + 70.0 * mag_loss


def _ratio_component_oracle(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    frame_lengths: torch.Tensor,
    synthesis_window: torch.Tensor,
) -> torch.Tensor:
    """逐行裁剪谱 → 指定合成窗 ``length=None`` iSTFT → 官方投影公式。

    投影保持官方乘除顺序 ``dot*target/(target_energy+1e-8)``，ratio 分母与
    ``-log10`` 的 eps 位置不变；输出长 <1 的行（center=True 偶数 n_fft 单帧）
    与实现一致跳过，不贡献质量；聚合保持 per-sample mean。
    """
    ratio_losses = []
    for row, frame_count in enumerate(frame_lengths.tolist()):
        frame_count = int(frame_count)
        if hop_length * (frame_count - 1) + n_fft % 2 < 1:
            continue
        pred_wav = torch.istft(
            torch.view_as_complex(
                predicted[row : row + 1, :, :frame_count].float().contiguous()
            ),
            n_fft,
            hop_length,
            win_length,
            window=synthesis_window,
            center=True,
            normalized=False,
            onesided=True,
            length=None,
        )
        true_wav = torch.istft(
            torch.view_as_complex(
                target[row : row + 1, :, :frame_count].float().contiguous()
            ),
            n_fft,
            hop_length,
            win_length,
            window=synthesis_window,
            center=True,
            normalized=False,
            onesided=True,
            length=None,
        )
        projected = (
            (true_wav * pred_wav).sum(dim=-1, keepdim=True)
            * true_wav
            / (true_wav.square().sum(dim=-1, keepdim=True) + 1e-8)
        )
        ratio = projected.square().sum(dim=-1) / (
            (pred_wav - projected).square().sum(dim=-1) + 1e-8
        )
        ratio_losses.append(-torch.log10(ratio + 1e-8))
    return torch.cat(ratio_losses).mean()


def _production_hybrid_loss_oracle(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    lengths: torch.Tensor | None,
    window: str = "hann",
) -> tuple[torch.Tensor, torch.Tensor]:
    """生产 oracle：合成窗与 loss 相同的普通窗（不 sqrt），length=None iSTFT。

    在 compute_spec_frame_lengths 契约下重建输出长不大于有效样本数，
    实现侧的 sample_lengths 时间掩码恒为全 1，oracle 无需额外掩码。
    """
    frame_lengths = _frame_lengths_for(
        predicted, lengths, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )
    if window == "hann":
        synthesis_window = torch.hann_window(
            win_length, device=predicted.device, dtype=predicted.dtype
        )
    elif window == "hamming":
        synthesis_window = torch.hamming_window(
            win_length, device=predicted.device, dtype=predicted.dtype
        )
    else:
        raise ValueError(f"unsupported window {window!r}")
    spectral = _compressed_spectral_component_oracle(predicted, target, frame_lengths)
    ratio = _ratio_component_oracle(
        predicted,
        target,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        frame_lengths=frame_lengths,
        synthesis_window=synthesis_window,
    )
    return spectral, ratio


def _official_evidence_hybrid_oracle(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    lengths: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """官方证据 oracle：保留官方 sqrt-Hann 合成窗，其余公式与生产 oracle 相同。"""
    frame_lengths = _frame_lengths_for(
        predicted, lengths, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )
    synthesis_window = torch.hann_window(
        win_length, device=predicted.device, dtype=predicted.dtype
    ).sqrt()
    spectral = _compressed_spectral_component_oracle(predicted, target, frame_lengths)
    ratio = _ratio_component_oracle(
        predicted,
        target,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        frame_lengths=frame_lengths,
        synthesis_window=synthesis_window,
    )
    return spectral, ratio


def _hybrid_composite(
    n_fft: int, hop_length: int, win_length: int, window: str = "hann"
) -> CompositeLoss:
    geometry = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "window": window,
    }
    return CompositeLoss(
        components=[
            {
                "type": "GTCRNCompressedSpectralLoss",
                "weight": 1.0,
                "args": geometry,
            },
            {"type": "GTCRNLogRatioLoss", "weight": 1.0, "args": geometry},
        ]
    )


def _hybrid_components(
    n_fft: int, hop_length: int, win_length: int, window: str = "hann"
) -> tuple[GTCRNCompressedSpectralLoss, GTCRNLogRatioLoss]:
    geometry = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "window": window,
    }
    return GTCRNCompressedSpectralLoss(**geometry), GTCRNLogRatioLoss(**geometry)


def _sqrt_hann_stft(waveform: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    spectrum = torch.stft(
        waveform,
        n_fft,
        hop_length,
        n_fft,
        window=torch.hann_window(n_fft, device=waveform.device).sqrt(),
        return_complex=True,
    )
    return torch.view_as_real(spectrum)


@pytest.mark.parametrize(
    ("n_fft", "hop_length", "window", "case"),
    [
        pytest.param(512, 256, "hann", "inconsistent", id="hann-512-inconsistent"),
        pytest.param(
            512, 256, "hann", "high_amplitude", id="hann-512-high-amplitude-raw"
        ),
        pytest.param(320, 160, "hann", "inconsistent", id="hann-320"),
        pytest.param(640, 320, "hann", "inconsistent", id="hann-640"),
        pytest.param(2048, 512, "hann", "inconsistent", id="hann-2048"),
        pytest.param(320, 160, "hamming", "inconsistent", id="hamming-320"),
        pytest.param(512, 256, "hamming", "inconsistent", id="hamming-512"),
    ],
)
def test_gtcrn_direct_spectra_match_production_oracle_value_and_gradients(
    n_fft: int, hop_length: int, window: str, case: str
):
    """生产语义：hann/hamming 普通同窗下消费 raw 不一致谱，组件值与双侧梯度对齐生产 oracle。"""
    generator = torch.Generator().manual_seed(n_fft + hop_length + len(window))
    output_length = n_fft + 2 * hop_length + 37
    shape = (2, n_fft // 2 + 1, output_length // hop_length + 1, 2)
    predicted_base = torch.randn(shape, generator=generator)
    target_base = torch.randn(shape, generator=generator)
    if case == "high_amplitude":
        # 高幅场景模拟 trainer 的 sanitize/clamp 结果。
        predicted_base = predicted_base * 25.0

    predicted = predicted_base.clone().requires_grad_()
    target = target_base.clone().requires_grad_()
    expected_components = _production_hybrid_loss_oracle(
        predicted,
        target,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        lengths=None,
        window=window,
    )

    # 波形参数故意与 raw 谱无关，直谱路径不得消费它们。
    enhanced = torch.zeros(2, output_length)
    clean = torch.ones(2, output_length) * 0.125
    for loss_fn, expected in zip(
        _hybrid_components(n_fft, hop_length, n_fft, window), expected_components
    ):
        expected_grad = torch.autograd.grad(
            expected, (predicted, target), retain_graph=True
        )
        actual_predicted = predicted_base.clone().requires_grad_()
        actual_target = target_base.clone().requires_grad_()
        actual = loss_fn(
            enhanced,
            clean,
            None,
            enhanced_spectra=actual_predicted,
            target_spectra=actual_target,
        )
        actual_grad = torch.autograd.grad(actual, (actual_predicted, actual_target))

        assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-5)
        for got, want in zip(actual_grad, expected_grad):
            assert torch.allclose(got, want, rtol=3e-4, atol=2e-6)

    actual_total = _hybrid_composite(n_fft, hop_length, n_fft, window)(
        enhanced,
        clean,
        None,
        enhanced_spectra=predicted_base,
        target_spectra=target_base,
    )
    assert torch.allclose(
        actual_total, sum(expected_components), rtol=2e-5, atol=2e-5
    )


def test_gtcrn_direct_spectra_sqrt_hann_window_swap_matches_official_evidence():
    """官方证据：sqrt-Hann 分析前端生成的谱 + 替换 sqrt-Hann 合成窗。

    把 loss 的 ``_window`` buffer 替换为 ``torch.hann_window(win).sqrt()`` 后，
    组件值、双侧梯度与 composite 总和须逐项对齐官方 HybridLoss 翻译版
    （sqrt-Hann 前端），证明生产实现与官方仅窗不同、公式一致。
    """
    n_fft, hop_length = 512, 256
    generator = torch.Generator().manual_seed(n_fft + hop_length)
    output_length = n_fft + 2 * hop_length + 37
    predicted_base = _sqrt_hann_stft(
        torch.randn(2, output_length, generator=generator), n_fft, hop_length
    )
    target_base = _sqrt_hann_stft(
        torch.randn(2, output_length, generator=generator), n_fft, hop_length
    )

    predicted = predicted_base.clone().requires_grad_()
    target = target_base.clone().requires_grad_()
    expected_components = _official_evidence_hybrid_oracle(
        predicted,
        target,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        lengths=None,
    )
    sqrt_window = torch.hann_window(n_fft).sqrt()

    enhanced = torch.zeros(2, output_length)
    clean = torch.ones(2, output_length) * 0.125
    for loss_fn, expected in zip(
        _hybrid_components(n_fft, hop_length, n_fft), expected_components
    ):
        # 同名赋值直接替换已注册的非持久 buffer，模拟官方 sqrt-Hann 合成。
        loss_fn._window = sqrt_window.clone()
        expected_grad = torch.autograd.grad(
            expected, (predicted, target), retain_graph=True
        )
        actual_predicted = predicted_base.clone().requires_grad_()
        actual_target = target_base.clone().requires_grad_()
        actual = loss_fn(
            enhanced,
            clean,
            None,
            enhanced_spectra=actual_predicted,
            target_spectra=actual_target,
        )
        actual_grad = torch.autograd.grad(actual, (actual_predicted, actual_target))

        assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-5)
        for got, want in zip(actual_grad, expected_grad):
            assert torch.allclose(got, want, rtol=3e-4, atol=2e-6)

    composite = _hybrid_composite(n_fft, hop_length, n_fft)
    for component in composite.losses:
        component._window = sqrt_window.clone()
    actual_total = composite(
        enhanced,
        clean,
        None,
        enhanced_spectra=predicted_base,
        target_spectra=target_base,
    )
    assert torch.allclose(
        actual_total, sum(expected_components), rtol=2e-5, atol=2e-5
    )


def test_gtcrn_log_ratio_direct_consistent_spectra_match_waveform_projection():
    """普通 Hann 一致谱 + 生产普通 Hann 合成：ratio 值须与波形域投影一致。

    防回退：纯 torch 语义下若把合成窗改回 sqrt-Hann，``torch.istft`` 的包络
    归一化会按 ``sum(hann**1.5)/sum(hann) ≈ 0.85`` 系统性缩放内部采样，
    重建明显失真，不得再与波形域一致。
    """
    n_fft, hop_length, samples = 512, 256, 4096
    generator = torch.Generator().manual_seed(4242)
    clean = torch.randn(2, samples, generator=generator)
    enhanced = torch.randn(2, samples, generator=generator)
    framework_stft = StftIstftModule(
        n_fft, hop_length, n_fft, window="hann", center=True
    )
    predicted_spectra = framework_stft(enhanced)
    target_spectra = framework_stft(clean)
    loss_fn = GTCRNLogRatioLoss(n_fft=n_fft, hop_length=hop_length, win_length=n_fft)

    direct = loss_fn(
        torch.zeros(2, samples),
        torch.ones(2, samples),
        None,
        enhanced_spectra=predicted_spectra,
        target_spectra=target_spectra,
    )
    waveform_reference = loss_fn(enhanced, clean, None)
    assert torch.allclose(direct, waveform_reference, rtol=1e-5, atol=1e-5)

    # 生产普通同窗合成可精确重建原波形。
    plain_reconstruction = torch.istft(
        torch.view_as_complex(predicted_spectra.float().contiguous()),
        n_fft,
        hop_length,
        n_fft,
        window=torch.hann_window(n_fft),
        center=True,
        length=None,
    )
    assert torch.allclose(plain_reconstruction, enhanced, rtol=1e-4, atol=1e-5)

    # sqrt-Hann 合成在一致谱上产生不可忽略的包络归一化失真。
    sqrt_reconstruction = torch.istft(
        torch.view_as_complex(predicted_spectra.float().contiguous()),
        n_fft,
        hop_length,
        n_fft,
        window=torch.hann_window(n_fft).sqrt(),
        center=True,
        length=None,
    )
    assert sqrt_reconstruction.shape == enhanced.shape
    assert not torch.allclose(sqrt_reconstruction, enhanced, rtol=1e-3, atol=1e-3)


def test_gtcrn_direct_spectra_variable_lengths_match_component_oracles():
    """变长谱按有效 TF 全局加权，ratio 按样本均值（普通合成窗 oracle），且各自梯度匹配。"""
    n_fft, hop_length, output_length = 320, 160, 997
    lengths = torch.tensor([997, 641, 997, 487])
    shape = (4, n_fft // 2 + 1, output_length // hop_length + 1, 2)
    generator = torch.Generator().manual_seed(20260901)
    predicted_base = torch.randn(shape, generator=generator)
    target_base = torch.randn(shape, generator=generator)
    for row, sample_length in enumerate(lengths.tolist()):
        frame_count = int(
            compute_spec_frame_lengths(
                lengths[row : row + 1],
                n_fft=n_fft,
                win_length=n_fft,
                hop_length=hop_length,
                center=True,
            ).item()
        )
        predicted_base[row, :, frame_count:] = 1e4 + row
        target_base[row, :, frame_count:] = -1e4 - row

    def evaluate(
        loss_fn: GTCRNCompressedSpectralLoss | GTCRNLogRatioLoss,
        order: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted = predicted_base.index_select(0, order).clone().requires_grad_()
        target = target_base.index_select(0, order).clone().requires_grad_()
        ordered_lengths = lengths.index_select(0, order)
        loss = loss_fn(
            torch.zeros(4, output_length),
            torch.ones(4, output_length) * 0.25,
            ordered_lengths,
            enhanced_spectra=predicted,
            target_spectra=target,
        )
        pred_grad, true_grad = torch.autograd.grad(loss, (predicted, target))
        valid_frame_lengths = compute_spec_frame_lengths(
            ordered_lengths,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            center=True,
        )
        for row, frame_count in enumerate(valid_frame_lengths.tolist()):
            assert torch.count_nonzero(pred_grad[row, :, frame_count:]) == 0
            assert torch.count_nonzero(true_grad[row, :, frame_count:]) == 0
        return loss, pred_grad, true_grad

    expected_pred = predicted_base.clone().requires_grad_()
    expected_true = target_base.clone().requires_grad_()
    expected_components = _production_hybrid_loss_oracle(
        expected_pred,
        expected_true,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        lengths=lengths,
    )
    identity = torch.arange(4)
    permutation = torch.tensor([2, 0, 3, 1])
    for loss_fn, expected in zip(
        _hybrid_components(n_fft, hop_length, n_fft), expected_components
    ):
        expected_grad = torch.autograd.grad(
            expected, (expected_pred, expected_true), retain_graph=True
        )
        actual, pred_grad, true_grad = evaluate(loss_fn, identity)
        assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-5)
        assert torch.allclose(pred_grad, expected_grad[0], rtol=3e-4, atol=2e-6)
        assert torch.allclose(true_grad, expected_grad[1], rtol=3e-4, atol=2e-6)

        permuted, perm_pred_grad, perm_true_grad = evaluate(loss_fn, permutation)
        assert torch.allclose(permuted, actual, rtol=1e-6, atol=1e-6)
        assert torch.allclose(
            perm_pred_grad,
            pred_grad.index_select(0, permutation),
            rtol=3e-4,
            atol=2e-6,
        )
        assert torch.allclose(
            perm_true_grad,
            true_grad.index_select(0, permutation),
            rtol=3e-4,
            atol=2e-6,
        )


def test_official_hybrid_oracle_length_none_output_follows_frame_geometry():
    """官方 ``torch.istft(length=None)`` 输出长度只由 frame 几何决定。"""
    n_fft, hop_length, frame_count = 320, 160, 7
    spectra = torch.randn(2, n_fft // 2 + 1, frame_count, 2)
    waveform = torch.istft(
        torch.view_as_complex(spectra.contiguous()),
        n_fft,
        hop_length,
        n_fft,
        window=torch.hann_window(n_fft).sqrt(),
        length=None,
    )
    assert waveform.shape == (2, hop_length * (frame_count - 1))


def test_gtcrn_log_ratio_single_frame_even_fft_returns_connected_finite_zero():
    """center=True 偶数 n_fft 单帧的 length=None 重建输出为空。

    空组被跳过、不贡献质量，结果须为 graph-connected 的有限零，
    且对预测/目标谱的梯度均为零（目标未入图时梯度为 None）。
    """
    n_fft, hop_length = 320, 160
    generator = torch.Generator().manual_seed(99)
    predicted = torch.randn(
        2, n_fft // 2 + 1, 1, 2, generator=generator
    ).requires_grad_()
    target = torch.randn(
        2, n_fft // 2 + 1, 1, 2, generator=generator
    ).requires_grad_()
    loss = GTCRNLogRatioLoss(n_fft=n_fft, hop_length=hop_length, win_length=n_fft)(
        torch.zeros(2, n_fft),
        torch.ones(2, n_fft),
        None,
        enhanced_spectra=predicted,
        target_spectra=target,
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    pred_grad, true_grad = torch.autograd.grad(
        loss, (predicted, target), allow_unused=True
    )
    assert pred_grad is not None
    assert torch.count_nonzero(pred_grad) == 0
    assert true_grad is None or torch.count_nonzero(true_grad) == 0


def test_gtcrn_direct_spectra_reject_insufficient_frame_context():
    """lengths 所需有效帧数超过谱帧数仍须 fail-fast。"""
    n_fft, hop_length, output_length = 320, 160, 997
    spectra = torch.randn(2, n_fft // 2 + 1, 3, 2)
    with pytest.raises(ValueError, match="frame|length"):
        _hybrid_composite(n_fft, hop_length, n_fft)(
            torch.zeros(2, output_length),
            torch.ones(2, output_length),
            torch.tensor([997, 641]),
            enhanced_spectra=spectra,
            target_spectra=spectra.clone(),
        )


def test_gtcrn_log_ratio_length_none_uses_all_spectrum_frames():
    """fixed-shape 官方路径不得从无关 waveform canvas 反推有效帧。"""
    n_fft, hop_length, frame_count = 320, 160, 7
    generator = torch.Generator().manual_seed(8128)
    predicted_base = torch.randn(
        2, n_fft // 2 + 1, frame_count, 2, generator=generator
    )
    target_base = torch.randn(
        2, n_fft // 2 + 1, frame_count, 2, generator=generator
    )
    predicted = predicted_base.clone().requires_grad_()
    target = target_base.clone().requires_grad_()
    expected = _production_hybrid_loss_oracle(
        predicted,
        target,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        lengths=None,
    )[1]
    expected_grad = torch.autograd.grad(expected, (predicted, target))
    loss_fn = GTCRNLogRatioLoss(
        n_fft=n_fft, hop_length=hop_length, win_length=n_fft
    )

    for canvas_length in (n_fft, 2000):
        actual_predicted = predicted_base.clone().requires_grad_()
        actual_target = target_base.clone().requires_grad_()
        actual = loss_fn(
            torch.zeros(2, canvas_length),
            torch.ones(2, canvas_length),
            None,
            enhanced_spectra=actual_predicted,
            target_spectra=actual_target,
        )
        actual_grad = torch.autograd.grad(
            actual, (actual_predicted, actual_target)
        )
        assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-5)
        for got, want in zip(actual_grad, expected_grad):
            assert torch.allclose(got, want, rtol=3e-4, atol=2e-6)


def test_gtcrn_log_ratio_rejects_non_nola_geometry_at_construction():
    with pytest.raises(ValueError, match="NOLA"):
        GTCRNLogRatioLoss(n_fft=320, hop_length=320, win_length=320)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {
                "n_fft": 320,
                "hop_length": 160,
                "win_length": 320,
                "window": "hamming",
                "center": False,
            },
            id="hamming-center-false",
        ),
        pytest.param(
            {"n_fft": 512, "hop_length": 256, "win_length": 512, "normalized": True},
            id="normalized-true",
        ),
    ],
)
def test_gtcrn_log_ratio_construction_accepts_extended_geometries(kwargs):
    """hamming+center=False 与 normalized=True 满足框架 NOLA 契约，构造须放行。"""
    loss_fn = GTCRNLogRatioLoss(**kwargs)
    assert loss_fn.window_name == kwargs.get("window", "hann")
    assert loss_fn.center == kwargs.get("center", True)
    assert loss_fn.normalized == kwargs.get("normalized", False)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {
                "n_fft": 320,
                "hop_length": 160,
                "win_length": 320,
                "window": "hann",
                "center": False,
            },
            id="hann-center-false",
        ),
        pytest.param(
            {"n_fft": 512, "hop_length": 256, "win_length": 128},
            id="hop-exceeds-win",
        ),
        pytest.param(
            {"n_fft": 256, "hop_length": 128, "win_length": 512},
            id="win-exceeds-nfft",
        ),
    ],
)
def test_gtcrn_log_ratio_construction_rejects_invalid_geometries(kwargs):
    """hann+center=False、hop>win、win>n_fft 违反框架契约，构造期 fail-fast。"""
    with pytest.raises(ValueError):
        GTCRNLogRatioLoss(**kwargs)


def test_gtcrn_log_ratio_allows_odd_fft_single_frame_output():
    """奇数 n_fft 的单帧 length=None iSTFT 仍会产生一个有效样本。"""
    n_fft, hop_length = 321, 160
    predicted = torch.randn(1, n_fft // 2 + 1, 1, 2, requires_grad=True)
    target = torch.randn(1, n_fft // 2 + 1, 1, 2, requires_grad=True)
    loss = GTCRNLogRatioLoss(
        n_fft=n_fft, hop_length=hop_length, win_length=n_fft
    )(
        torch.zeros(1, 1),
        torch.ones(1, 1),
        None,
        enhanced_spectra=predicted,
        target_spectra=target,
    )
    gradients = torch.autograd.grad(loss, (predicted, target))
    assert torch.isfinite(loss)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_gtcrn_log_ratio_silent_direct_target_keeps_finite_zero_graph_under_amp():
    n_fft, hop_length, frame_count = 320, 160, 4
    predicted = torch.randn(
        2, n_fft // 2 + 1, frame_count, 2, requires_grad=True
    )
    target = torch.zeros_like(predicted, requires_grad=True)
    loss_fn = GTCRNLogRatioLoss(
        n_fft=n_fft, hop_length=hop_length, win_length=n_fft
    )

    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = loss_fn(
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            None,
            enhanced_spectra=predicted,
            target_spectra=target,
        )
    gradients = torch.autograd.grad(loss, (predicted, target))

    assert loss.dtype == torch.float32
    assert loss.item() == 0.0
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients)
