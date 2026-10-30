"""GTCRN direct-spectrum Trainer 接线回归测试。"""

import pytest
import torch

from core.losses.composite import CompositeLoss
from core.stft import StftIstftModule
from tasks.se.trainer.gan_trainer import GANTrainer


class _GANDirectRatioStub(GANTrainer):
    def _sanitize_model_output(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).clamp(
            -1.0, 1.0
        )


def test_gan_gtcrn_components_share_raw_spectra_and_ratio_ignores_sanitized_waveform():
    """两个 GTCRN 分量共享 raw 谱；ratio 不回退到 sanitize 后波形。"""
    torch.manual_seed(20260901)
    n_fft, hop_length, samples = 32, 8, 96
    stft = StftIstftModule(n_fft, hop_length, n_fft, window="hann", center=True)
    trainer = _GANDirectRatioStub.__new__(_GANDirectRatioStub)
    trainer.sample_rate = 16000
    trainer.stft_module = stft
    trainer.loss_fn = CompositeLoss(
        components=[
            {
                "type": "GTCRNCompressedSpectralLoss",
                "weight": 1.0,
                "args": {
                    "n_fft": n_fft,
                    "hop_length": hop_length,
                    "win_length": n_fft,
                },
            },
            {
                "type": "GTCRNLogRatioLoss",
                "weight": 1.0,
                "args": {
                    "n_fft": n_fft,
                    "hop_length": hop_length,
                    "win_length": n_fft,
                },
            },
        ]
    )
    clean = torch.randn(2, samples)
    lengths = torch.tensor([samples, samples - 13], dtype=torch.long)
    raw_enhanced_spectra = (20.0 * torch.randn_like(stft(clean))).requires_grad_()
    raw_waveform = stft.inverse(raw_enhanced_spectra, samples)
    forward_dict = {
        "enhanced": raw_waveform,
        "enhanced_spec": raw_enhanced_spectra,
        "clean": clean,
        "noisy": clean + 0.1,
        "lengths": lengths,
        "fs": 16000,
    }

    trainer._sanitize_forward_output(forward_dict)
    assert forward_dict["loss_enhanced_spectra"] is raw_enhanced_spectra
    assert not torch.equal(forward_dict["enhanced"], raw_waveform)

    calls: list[tuple[tuple, dict]] = []
    hooks = [
        component.register_forward_pre_hook(
            lambda _module, args, kwargs: calls.append((args, dict(kwargs))),
            with_kwargs=True,
        )
        for component in trainer.loss_fn.losses
    ]
    try:
        actual = trainer._compute_reconstruction_loss(
            forward_dict["enhanced"],
            clean,
            lengths,
            forward_dict=forward_dict,
        )
    finally:
        for hook in hooks:
            hook.remove()

    assert torch.isfinite(actual)
    assert len(calls) == 2
    first_kwargs, ratio_kwargs = calls[0][1], calls[1][1]
    assert first_kwargs["enhanced_spectra"] is raw_enhanced_spectra
    assert ratio_kwargs["enhanced_spectra"] is raw_enhanced_spectra
    assert first_kwargs["target_spectra"] is ratio_kwargs["target_spectra"]
    assert calls[0][0][2] is lengths and calls[1][0][2] is lengths

    raw_ratio = trainer.loss_fn.losses[1](
        forward_dict["enhanced"],
        clean,
        lengths,
        enhanced_spectra=raw_enhanced_spectra,
        target_spectra=ratio_kwargs["target_spectra"],
    )
    sanitized_waveform_ratio = trainer.loss_fn.losses[1](
        forward_dict["enhanced"], clean, lengths
    )
    assert not torch.allclose(raw_ratio, sanitized_waveform_ratio, rtol=1e-4, atol=1e-4)


pytestmark = [pytest.mark.unit]
