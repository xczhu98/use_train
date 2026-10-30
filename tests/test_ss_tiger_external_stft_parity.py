import pytest
import torch

from core.stft import build_stft_istft
from tasks.ss.models._third_party.tiger import TIGER as ReferenceTIGER
from tasks.ss.models.tiger import TIGER


@pytest.mark.unit
def test_tiger_external_stft_matches_internal_reference_path():
    args = {
        "out_channels": 8,
        "in_channels": 16,
        "num_blocks": 1,
        "upsampling_depth": 2,
        "win": 640,
        "stride": 160,
        "num_sources": 2,
        "sample_rate": 16000,
    }
    torch.manual_seed(42)
    reference = ReferenceTIGER(**args).eval()
    external = TIGER(
        **args,
        n_fft=640,
        hop_length=160,
        win_length=640,
        window="hann",
        center=True,
    ).eval()
    external.load_state_dict(reference.state_dict())
    mixture = torch.randn(1, 1, 1280)
    stft = build_stft_istft(
        {
            "n_fft": 640,
            "hop_length": 160,
            "win_length": 640,
            "window": "hann",
            "center": True,
        }
    )
    with torch.no_grad():
        reference_output = reference(mixture)[:, 0]
        external_output = external(
            stft(mixture[:, 0]), torch.tensor([mixture.shape[-1]])
        )["est_sources_spectra"]
        flat = external_output.reshape(2, *external_output.shape[2:])
        external_waveform = stft.inverse(flat, mixture.shape[-1]).reshape(1, 2, -1)
    assert external_waveform.shape == reference_output.shape
    assert torch.allclose(external_waveform, reference_output, atol=2e-4, rtol=2e-4)
