import pytest
import torch

from core.stft import build_stft_istft
from tasks.ss.inference.extractor import SSExtractor
from tasks.ss.model_io import SSModelRuntime, resolve_ss_execution_plan
from tasks.ss.models.spatialnet import SpatialNet


class _WaveformSeparator(torch.nn.Module):
    input_domain = "waveform"
    output_domain = "waveform"

    def forward(self, mixture, lengths, *, fs=None):
        del lengths, fs
        return {"est_sources": torch.stack((mixture, -mixture), dim=1)}


class _SpectrumSeparator(torch.nn.Module):
    input_domain = "spectrum"
    output_domain = "spectrum"

    def forward(self, mixture_spectra, lengths, *, fs=None):
        del lengths, fs
        return {
            "est_sources_spectra": torch.stack(
                (mixture_spectra, -mixture_spectra), dim=1
            )
        }


class _WrongLengthSeparator(_WaveformSeparator):
    def forward(self, mixture, lengths, *, fs=None):
        del lengths, fs
        return {"est_sources": torch.stack((mixture[:, :-1], -mixture[:, :-1]), dim=1)}


class _NonFiniteSeparator(_WaveformSeparator):
    def forward(self, mixture, lengths, *, fs=None):
        del lengths, fs
        sources = torch.stack((mixture, -mixture), dim=1)
        sources[:, 0, 0] = float("nan")
        return {"est_sources": sources}


def _config(stft=None):
    value = {
        "data": {"mode": "offline_ss", "sample_rate": 16000},
        "ss": {"num_sources": 2},
    }
    if stft is not None:
        value["stft"] = stft
    return value


@pytest.mark.unit
def test_ss_model_runtime_unifies_waveform_model_output():
    mixture = torch.randn(2, 64)
    model = _WaveformSeparator()
    runtime = SSModelRuntime(
        model,
        plan=resolve_ss_execution_plan(_config(), model),
    )
    result = runtime.separate(mixture, torch.tensor([64, 48]), sample_rate=16000)
    assert result.sources.shape == (2, 2, 64)
    assert result.source_spectra is None
    assert torch.equal(result.sources[:, 0], mixture)


@pytest.mark.unit
def test_ss_model_runtime_unifies_external_stft_output():
    stft_cfg = {
        "n_fft": 64,
        "hop_length": 32,
        "win_length": 64,
        "window": "hamming",
        "center": True,
    }
    stft = build_stft_istft(stft_cfg)
    mixture = torch.randn(2, 128)
    model = _SpectrumSeparator()
    runtime = SSModelRuntime(
        model,
        plan=resolve_ss_execution_plan(_config(stft_cfg), model),
        stft_module=stft,
    )
    result = runtime.separate(mixture, torch.tensor([128, 96]), sample_rate=16000)
    assert result.sources.shape == (2, 2, 128)
    assert result.source_spectra.shape[:2] == (2, 2)


@pytest.mark.unit
def test_ss_model_runtime_rejects_missing_spectrum_frontend():
    model = _SpectrumSeparator()
    with pytest.raises(ValueError, match="stft_module"):
        SSModelRuntime(
            model,
            plan=resolve_ss_execution_plan(
                _config(
                    {
                        "n_fft": 64,
                        "hop_length": 32,
                        "win_length": 64,
                        "window": "hamming",
                        "center": True,
                    }
                ),
                model,
            ),
        )


@pytest.mark.unit
def test_ss_extractor_uses_model_capability_without_model_io_string():
    model = _WaveformSeparator()
    extractor = SSExtractor(
        model,
        plan=resolve_ss_execution_plan(_config(), model),
    )
    mixture = torch.randn(1, 64)
    output = extractor(mixture, sample_rate=16000)
    assert output.shape == (1, 2, 64)


@pytest.mark.unit
def test_ss_model_runtime_rejects_runtime_sample_rate_mismatch():
    model = _WaveformSeparator()
    runtime = SSModelRuntime(model, plan=resolve_ss_execution_plan(_config(), model))
    with pytest.raises(ValueError, match="runtime sample_rate"):
        runtime.separate(torch.randn(1, 64), torch.tensor([64]), sample_rate=8000)


@pytest.mark.unit
def test_ss_model_runtime_rejects_wrong_output_length():
    model = _WrongLengthSeparator()
    runtime = SSModelRuntime(model, plan=resolve_ss_execution_plan(_config(), model))
    with pytest.raises(ValueError, match="time axis must exactly match"):
        runtime.separate(torch.randn(1, 64), torch.tensor([64]), sample_rate=16000)


@pytest.mark.unit
def test_ss_model_runtime_rejects_nonfinite_output():
    model = _NonFiniteSeparator()
    runtime = SSModelRuntime(model, plan=resolve_ss_execution_plan(_config(), model))
    with pytest.raises(ValueError, match="finite"):
        runtime.separate(torch.randn(1, 64), torch.tensor([64]), sample_rate=16000)


@pytest.mark.unit
def test_ss_model_runtime_rejects_spatial_input_for_mono_spectrum_model():
    stft_cfg = {
        "n_fft": 64,
        "hop_length": 32,
        "win_length": 64,
        "window": "hamming",
        "center": True,
    }
    model = _SpectrumSeparator()
    runtime = SSModelRuntime(
        model,
        plan=resolve_ss_execution_plan(_config(stft_cfg), model),
        stft_module=build_stft_istft(stft_cfg),
    )
    with pytest.raises(ValueError, match="microphone axis"):
        runtime.separate(
            torch.randn(1, 2, 128), torch.tensor([128]), sample_rate=16000
        )


@pytest.mark.unit
def test_spatialnet_runtime_supports_multichannel_and_c1_legacy():
    stft_cfg = {
        "n_fft": 64,
        "hop_length": 32,
        "win_length": 64,
        "window": "hamming",
        "center": True,
    }
    model = SpatialNet(
        n_channel=2,
        num_sources=2,
        dim_hidden=8,
        dim_ffn=16,
        dim_squeeze=4,
        num_layers=1,
        num_heads=2,
        num_freqs=33,
        conv_groups=(1, 1),
    ).eval()
    stft = build_stft_istft(stft_cfg)
    runtime = SSModelRuntime(
        model,
        plan=resolve_ss_execution_plan(_config(stft_cfg), model),
        stft_module=stft,
    )
    result = runtime.separate(
        torch.randn(1, 2, 128), torch.tensor([128]), sample_rate=16000
    )
    assert result.sources.shape == (1, 2, 128)
    assert result.source_spectra.shape[1:] == (2, 33, 5, 2)

    mono_model = SpatialNet(
        n_channel=1,
        num_sources=2,
        dim_hidden=8,
        dim_ffn=16,
        dim_squeeze=4,
        num_layers=1,
        num_heads=2,
        num_freqs=33,
        conv_groups=(1, 1),
    ).eval()
    mono_output = mono_model(stft(torch.randn(1, 128)))
    assert mono_output["est_sources_spectra"].shape == (1, 2, 33, 5, 2)
