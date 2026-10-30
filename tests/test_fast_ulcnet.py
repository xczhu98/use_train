"""Fast-ULCNet 官方结构、CRM 与 Comfi-FastGRNN 契约测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import core.losses  # noqa: F401  注册共享 loss
import tasks.se.models  # noqa: F401  注册 SE lazy map
from core.losses.build import build_loss
from core.models.build import build_model
from core.utils.config import load_merged_config
from tasks.se.models.discriminative.fast_ulcnet import FastULCNet
from tasks.se.models.modules.fast_ulcnet_impl.comfi_fast_grnn import ComfiFastGRNN
from tasks.se.models.modules.fast_ulcnet_impl.crm import ComplexRatioMask
from tools.verify.verify_fast_ulcnet_state_drift import _validate_frame_counts


pytestmark = pytest.mark.unit


def _stft_config() -> dict:
    return {
        "n_fft": 512,
        "hop_length": 128,
        "win_length": 512,
        "window": "hann",
        "center": True,
    }


def test_registry_build_and_official_parameter_count():
    model = build_model({"type": "FastULCNet", "args": {}}, stft_cfg=_stft_config())

    assert isinstance(model, FastULCNet)
    assert (model.n_fft, model.hop_length, model.win_length) == (512, 128, 512)
    assert model.window == "hann"
    assert model.center is True
    assert sum(parameter.numel() for parameter in model.parameters()) == 338_547


def test_offline_config_stft_fields_are_consumed_by_model_runtime():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs/se/voicebank_demand/fast_ulcnet_offline.yaml"
    )
    config = load_merged_config(config_path)

    model = build_model(config["model"], stft_cfg=config["stft"])

    assert model.n_fft == config["stft"]["n_fft"] == 512
    assert model.hop_length == config["stft"]["hop_length"] == 128
    assert model.win_length == config["stft"]["win_length"] == 512
    assert model.window == config["stft"]["window"] == "hann"
    assert model.center is config["stft"]["center"] is True
    assert model.compression_factor == config["model"]["args"]["compression_factor"]
    assert model.freq_rnn.hidden_size == config["model"]["args"][
        "bidirectional_frnn_units"
    ]
    assert model.sub_band_rnn1.hidden_size == config["model"]["args"][
        "sub_band_rnn_units"
    ]
    assert model.crm_layer.masking_mode == config["model"]["args"]["CRM_type"]

    loss = build_loss(config["loss"], stft_cfg=config["stft"])
    assert loss.n_fft == config["stft"]["n_fft"] == 512
    assert loss.hop_length == config["stft"]["hop_length"] == 128
    assert loss.win_length == config["stft"]["win_length"] == 512


def test_long_sequence_cli_requires_explicit_full_model_memory_override():
    with pytest.raises(ValueError, match="confirmation threshold"):
        _validate_frame_counts(
            whole_segment_frames=1001,
            temporal_frames=1,
            skip_whole_segment=False,
            allow_large_whole_segment=False,
        )

    _validate_frame_counts(
        whole_segment_frames=1001,
        temporal_frames=10_000,
        skip_whole_segment=True,
        allow_large_whole_segment=False,
    )
    _validate_frame_counts(
        whole_segment_frames=1001,
        temporal_frames=1,
        skip_whole_segment=False,
        allow_large_whole_segment=True,
    )


def test_constructor_rejects_unknown_kwargs_and_non_257_architecture():
    with pytest.raises(TypeError, match="unexpected keyword argument.*compression_facter"):
        FastULCNet(compression_facter=0.3)
    with pytest.raises(ValueError, match="fixed 257-bin architecture"):
        FastULCNet(n_fft=256, win_length=256)


@pytest.mark.parametrize("frequency_bins", [256, 258])
def test_forward_rejects_non_257_bin_spectrum(frequency_bins):
    model = FastULCNet().eval()
    spectrum = torch.zeros(1, frequency_bins, 2, 2)

    with pytest.raises(ValueError, match="exactly 257 frequency bins"):
        model(spectrum)


def test_power_law_preprocessing_and_decompression_match_official_formula():
    torch.manual_seed(0)
    model = FastULCNet(compression_factor=0.3)
    spectrum = torch.complex(torch.randn(4, 8), torch.randn(4, 8))
    spectrum[0, :5] = torch.complex(
        torch.tensor([0.0, -1e-30, -1e-12, -1.0, 4.0]),
        torch.tensor([0.0, 1e-24, -1e-9, 2.0, -3.0]),
    )

    magnitude, phase, comp_real, comp_imag = model._feature_preprocessing(spectrum)
    expected_real = torch.sign(spectrum.real) * torch.abs(spectrum.real).pow(0.3)
    expected_imag = torch.sign(spectrum.imag) * torch.abs(spectrum.imag).pow(0.3)

    assert torch.allclose(comp_real, expected_real, atol=0.0, rtol=1e-6)
    assert torch.allclose(comp_imag, expected_imag, atol=0.0, rtol=1e-6)
    assert torch.allclose(
        magnitude, torch.sqrt(expected_real**2 + expected_imag**2), atol=0.0, rtol=1e-6
    )
    assert torch.allclose(
        phase, torch.atan2(expected_imag, expected_real), atol=1e-7, rtol=1e-6
    )
    assert comp_real[0, 0].item() == 0.0
    assert comp_imag[0, 0].item() == 0.0
    assert torch.allclose(
        model._power_law_decompression(torch.complex(comp_real, comp_imag)),
        spectrum,
        atol=1e-20,
        rtol=1e-6,
    )


def test_forward_preserves_bft_real_imag_layout():
    class _PassThroughCRM(torch.nn.Module):
        def forward(self, real, imag, m_real, m_imag):
            assert real.shape == imag.shape == (1, 2, 257)
            return torch.complex(real, imag)

    model = FastULCNet(compression_factor=1.0).eval()
    model.crm_layer = _PassThroughCRM()
    spectrum = torch.arange(1 * 257 * 2 * 2, dtype=torch.float32).reshape(
        1, 257, 2, 2
    )

    with torch.inference_mode():
        enhanced = model(spectrum)

    assert torch.equal(enhanced, spectrum)


def test_crm_e_matches_official_magnitude_phase_formula():
    real = torch.tensor([[3.0, -2.0]])
    imag = torch.tensor([[4.0, 1.0]])
    mask_real = torch.tensor([[0.5, -0.25]])
    mask_imag = torch.tensor([[0.75, 0.4]])
    layer = ComplexRatioMask("E")

    actual = layer(real, imag, mask_real, mask_imag)
    safe_mask_real = mask_real + layer.eps
    expected_magnitude = torch.tanh(
        torch.sqrt(safe_mask_real**2 + mask_imag**2)
    ) * torch.sqrt(real**2 + imag**2)
    expected_phase = torch.atan2(imag, real) + torch.atan2(
        mask_imag, safe_mask_real
    )
    expected = torch.polar(expected_magnitude, expected_phase)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_crm_c_matches_complex_multiplication():
    real = torch.tensor([[1.0, 2.0]])
    imag = torch.tensor([[3.0, 4.0]])
    mask_real = torch.tensor([[0.5, -1.0]])
    mask_imag = torch.tensor([[2.0, 0.25]])

    actual = ComplexRatioMask("C")(real, imag, mask_real, mask_imag)
    expected = torch.complex(
        real * mask_real - imag * mask_imag,
        real * mask_imag + imag * mask_real,
    )

    assert torch.equal(actual, expected)


def test_crm_r_scales_real_and_imag_components_independently():
    real = torch.tensor([[1.0, 2.0]])
    imag = torch.tensor([[3.0, 4.0]])
    mask_real = torch.tensor([[0.5, -1.0]])
    mask_imag = torch.tensor([[2.0, 0.25]])

    actual = ComplexRatioMask("R")(real, imag, mask_real, mask_imag)
    expected = torch.complex(real * mask_real, imag * mask_imag)

    assert torch.equal(actual, expected)


def test_comfi_fast_grnn_matches_explicit_recurrence():
    layer = ComfiFastGRNN(input_size=1, hidden_size=1, batch_first=True)
    cell = layer.cells_fwd[0]
    with torch.no_grad():
        cell.w_matrix.fill_(0.4)
        cell.u_matrix.fill_(-0.2)
        cell.bias_gate.fill_(0.1)
        cell.bias_update.fill_(-0.3)
        cell.zeta.fill_(0.7)
        cell.nu.fill_(-2.0)
        cell.gamma.fill_(0.8)
        cell.lambd.fill_(0.25)

    inputs = torch.tensor([[[0.5], [-1.0], [0.25]]])
    actual, hidden = layer(inputs)
    expected_steps = []
    previous = torch.zeros(1, 1)
    for step in inputs.unbind(dim=1):
        projection = step @ cell.w_matrix + previous @ cell.u_matrix
        gate = torch.sigmoid(projection + cell.bias_gate)
        update = torch.tanh(projection + cell.bias_update)
        fast_hidden = gate * previous + (
            torch.sigmoid(cell.zeta) * (1 - gate) + torch.sigmoid(cell.nu)
        ) * update
        previous = torch.clamp(cell.gamma, 0.0, 1.0) * fast_hidden + (
            1 - torch.clamp(cell.gamma, 0.0, 1.0)
        ) * cell.lambd
        expected_steps.append(previous)
    expected = torch.stack(expected_steps, dim=1)

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert torch.allclose(hidden[0], expected[:, -1], atol=1e-7, rtol=1e-6)


def test_spectrum_forward_backward_is_finite():
    torch.manual_seed(0)
    model = FastULCNet()
    spectrum = torch.randn(1, 257, 3, 2, requires_grad=True)

    enhanced = model(spectrum)
    loss = enhanced.square().mean()
    loss.backward()

    assert enhanced.shape == spectrum.shape
    assert torch.isfinite(enhanced).all()
    assert spectrum.grad is not None
    assert torch.isfinite(spectrum.grad).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_zero_spectrum_forward_backward_has_finite_gradients():
    model = FastULCNet()
    spectrum = torch.zeros(1, 257, 3, 2, requires_grad=True)

    enhanced = model(spectrum)
    enhanced.sum().backward()

    assert torch.equal(enhanced, torch.zeros_like(enhanced))
    assert spectrum.grad is not None
    assert torch.equal(spectrum.grad, torch.zeros_like(spectrum.grad))
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_upstream_provenance_and_mit_licenses_are_present():
    module_dir = (
        Path(__file__).resolve().parents[1]
        / "tasks/se/models/modules/fast_ulcnet_impl"
    )
    provenance = (module_dir / "PROVENANCE").read_text(encoding="utf-8")
    fast_ulcnet_license = (module_dir / "LICENSE.Fast-ULCNet").read_text(
        encoding="utf-8"
    )
    crm_license = (module_dir / "LICENSE.CRM_pytorch").read_text(encoding="utf-8")

    assert "f1dcfb48df4aed05812c590696d306fb221bd52a" in provenance
    assert "21d457cb740dc50f72c5d1fdda68fa2cc332b77a" in provenance
    assert "Copyright (c) 2025 Nicolás Arrieta Larraza" in fast_ulcnet_license
    assert "Copyright (c) 2024 Your Name" in crm_license
    assert "MIT License" in fast_ulcnet_license
    assert "MIT License" in crm_license
