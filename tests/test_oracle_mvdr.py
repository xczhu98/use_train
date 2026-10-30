"""Torch oracle MVDR unit contracts."""

import pytest
import torch

from evaluate.oracle_beamformer import oracle_mvdr

pytestmark = pytest.mark.unit


def _snr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    error = estimate - reference
    return float(10.0 * torch.log10(reference.square().sum() / error.square().sum()))


def test_single_channel_mvdr_is_identity_and_preserves_unbatched_shape():
    mixture = torch.randn(1, 4096, generator=torch.Generator().manual_seed(1))
    target = torch.randn(1, 4096, generator=torch.Generator().manual_seed(2))

    enhanced = oracle_mvdr(mixture, target, n_fft=256, hop_length=128)

    assert enhanced.shape == (4096,)
    torch.testing.assert_close(enhanced, mixture[0], atol=2e-5, rtol=2e-5)


def test_mvdr_suppresses_a_spatially_distinct_interferer_and_batches():
    generator = torch.Generator().manual_seed(3)
    target_source = torch.randn(8192, generator=generator)
    interference = torch.randn(8192, generator=torch.Generator().manual_seed(4))
    target = torch.stack((target_source, 0.5 * target_source), dim=0)
    interference_image = torch.stack((interference, -interference), dim=0)
    mixture = target + interference_image

    enhanced = oracle_mvdr(
        mixture.unsqueeze(0),
        target.unsqueeze(0),
        n_fft=256,
        hop_length=64,
        reference_channel=0,
    )

    assert enhanced.shape == (1, 8192)
    assert _snr(target_source, enhanced[0]) > _snr(target_source, mixture[0]) + 15.0
    assert torch.isfinite(enhanced).all()


def test_mvdr_reference_channel_controls_distortionless_output():
    source = torch.randn(4096, generator=torch.Generator().manual_seed(5))
    target = torch.stack((source, 2.0 * source), dim=0)

    enhanced = oracle_mvdr(
        target,
        target,
        n_fft=256,
        hop_length=128,
        reference_channel=1,
    )

    assert _snr(target[1], enhanced) > 80.0


def test_mvdr_rejects_shape_and_parameter_errors():
    mixture = torch.ones(2, 1024)
    with pytest.raises(ValueError, match="must match mixture"):
        oracle_mvdr(mixture, torch.ones(1, 1024), n_fft=256)
    with pytest.raises(ValueError, match="reference_channel"):
        oracle_mvdr(mixture, mixture, n_fft=256, reference_channel=2)
    with pytest.raises(ValueError, match="diagonal_loading"):
        oracle_mvdr(mixture, mixture, n_fft=256, diagonal_loading=0.0)
