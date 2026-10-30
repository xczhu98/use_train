"""DCCRN 复数层、注册与框架谱接口契约测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import tasks.se.models  # noqa: F401  安装 SE lazy map
from core.models.build import build_model
from core.utils.config import load_merged_config
from tasks.se.models.discriminative.dccrn import DCCRN
from tasks.se.models.modules.dccrn_impl import (
    ComplexBatchNorm,
    ComplexConv2d,
    ComplexConvTranspose2d,
    NaiveComplexLSTM,
    apply_complex_mask,
    complex_cat,
)

pytestmark = pytest.mark.unit


def _small_model(**kwargs) -> DCCRN:
    defaults = {
        "n_fft": 64,
        "hop_length": 16,
        "win_length": 64,
        "rnn_units": 8,
        "kernel_num": (4, 4, 4, 4, 4),
    }
    defaults.update(kwargs)
    return DCCRN(**defaults)


def test_complex_conv_matches_explicit_complex_multiplication():
    layer = ComplexConv2d(2, 2, kernel_size=(1, 1), causal=True)
    with torch.no_grad():
        layer.real_conv.weight.fill_(2.0)
        layer.imag_conv.weight.fill_(-0.5)
        layer.real_conv.bias.fill_(0.25)
        layer.imag_conv.bias.fill_(-0.75)
    real = torch.tensor([[[[1.0, -2.0], [0.5, 3.0]]]])
    imag = torch.tensor([[[[4.0, 0.25], [-1.0, 2.0]]]])

    actual_real, actual_imag = layer(torch.cat((real, imag), dim=1)).chunk(2, dim=1)
    expected_real = F.conv2d(real, layer.real_conv.weight, layer.real_conv.bias)
    expected_real = expected_real - F.conv2d(
        imag, layer.imag_conv.weight, layer.imag_conv.bias
    )
    expected_imag = F.conv2d(real, layer.imag_conv.weight, layer.imag_conv.bias)
    expected_imag = expected_imag + F.conv2d(
        imag, layer.real_conv.weight, layer.real_conv.bias
    )

    assert torch.equal(actual_real, expected_real)
    assert torch.equal(actual_imag, expected_imag)


def test_complex_transposed_conv_matches_explicit_complex_multiplication():
    layer = ComplexConvTranspose2d(2, 2, kernel_size=(1, 1))
    with torch.no_grad():
        layer.real_conv.weight.fill_(1.5)
        layer.imag_conv.weight.fill_(0.25)
        layer.real_conv.bias.fill_(-0.5)
        layer.imag_conv.bias.fill_(0.75)
    real = torch.tensor([[[[1.0, -2.0]]]])
    imag = torch.tensor([[[[3.0, 0.5]]]])

    actual_real, actual_imag = layer(torch.cat((real, imag), dim=1)).chunk(2, dim=1)
    expected_real = F.conv_transpose2d(
        real, layer.real_conv.weight, layer.real_conv.bias
    ) - F.conv_transpose2d(imag, layer.imag_conv.weight, layer.imag_conv.bias)
    expected_imag = F.conv_transpose2d(
        real, layer.imag_conv.weight, layer.imag_conv.bias
    ) + F.conv_transpose2d(imag, layer.real_conv.weight, layer.real_conv.bias)

    assert torch.equal(actual_real, expected_real)
    assert torch.equal(actual_imag, expected_imag)


def test_complex_cat_preserves_real_then_imag_channel_layout():
    first = torch.tensor([[[[1.0]], [[10.0]]]])
    second = torch.tensor([[[[2.0]], [[3.0]], [[20.0]], [[30.0]]]])

    actual = complex_cat((first, second), axis=1)

    assert torch.equal(
        actual,
        torch.tensor([[[[1.0]], [[2.0]], [[3.0]], [[10.0]], [[20.0]], [[30.0]]]]),
    )


def test_naive_complex_lstm_matches_four_real_lstm_paths():
    torch.manual_seed(0)
    layer = NaiveComplexLSTM(input_size=4, hidden_size=6)
    real = torch.randn(3, 2, 2)
    imag = torch.randn(3, 2, 2)

    actual_real, actual_imag = layer((real, imag))
    expected_real = layer.real_lstm(real)[0] - layer.imag_lstm(imag)[0]
    expected_imag = layer.real_lstm(imag)[0] + layer.imag_lstm(real)[0]

    assert torch.allclose(actual_real, expected_real)
    assert torch.allclose(actual_imag, expected_imag)


def test_complex_batch_norm_whitens_uncorrelated_components():
    torch.manual_seed(1)
    layer = ComplexBatchNorm(2, affine=False, track_running_stats=False)
    inputs = torch.randn(64, 2, 4, 3)

    output = layer(inputs)
    real, imag = output.chunk(2, dim=1)

    assert torch.allclose(real.mean(), torch.tensor(0.0), atol=2e-6)
    assert torch.allclose(imag.mean(), torch.tensor(0.0), atol=2e-6)
    assert torch.allclose(real.square().mean(), torch.tensor(1.0), atol=2e-4)
    assert torch.allclose(imag.square().mean(), torch.tensor(1.0), atol=2e-4)
    assert torch.allclose((real * imag).mean(), torch.tensor(0.0), atol=2e-4)


@pytest.mark.parametrize("masking_mode", ["E", "C", "R"])
def test_apply_complex_mask_matches_reference_formula(masking_mode):
    noisy = torch.tensor([[[[3.0, 4.0], [-2.0, 1.0]]]])
    mask_real = torch.tensor([[[0.5, -0.25]]])
    mask_imag = torch.tensor([[[0.75, 0.4]]])

    actual = apply_complex_mask(noisy, mask_real, mask_imag, masking_mode)
    noisy_complex = torch.view_as_complex(noisy)
    if masking_mode == "E":
        expected_magnitude = torch.tanh(
            torch.sqrt(mask_real.square() + mask_imag.square())
        ) * torch.sqrt(noisy_complex.abs().square() + 1e-8)
        expected_phase = torch.angle(noisy_complex) + torch.atan2(mask_imag, mask_real)
        expected = torch.polar(expected_magnitude, expected_phase)
    elif masking_mode == "C":
        expected = noisy_complex * torch.complex(mask_real, mask_imag)
    else:
        expected = torch.complex(
            noisy_complex.real * mask_real, noisy_complex.imag * mask_imag
        )

    assert torch.allclose(torch.view_as_complex(actual), expected, atol=1e-6, rtol=1e-6)


def test_registry_build_consumes_framework_stft_contract():
    model = build_model(
        {
            "type": "DCCRN",
            "args": {
                "rnn_units": 8,
                "kernel_num": [4, 4, 4, 4, 4],
            },
        },
        stft_cfg={
            "n_fft": 64,
            "hop_length": 16,
            "win_length": 64,
            "window": "hann",
            "center": True,
        },
    )

    assert isinstance(model, DCCRN)
    assert (model.n_fft, model.hop_length, model.win_length) == (64, 16, 64)
    assert model.window == "hann"
    assert model.center is True


def test_offline_config_builds_canonical_dccrn_cl():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs/se/voicebank_demand/dccrn_offline.yaml"
    )
    config = load_merged_config(config_path)

    model = build_model(config["model"], stft_cfg=config["stft"])

    assert isinstance(model, DCCRN)
    assert model.n_fft == 512
    assert model.core.masking_mode == "E"
    assert model.core.use_clstm is True
    assert len(model.core.encoder) == len(model.core.decoder) == 6


def test_constructor_and_forward_fail_closed_on_invalid_contracts():
    with pytest.raises(TypeError, match="unexpected keyword argument.*rnn_unit"):
        DCCRN(rnn_unit=256)
    with pytest.raises(ValueError, match="divisible"):
        DCCRN(n_fft=400)
    model = _small_model()
    with pytest.raises(ValueError, match=r"\(B,F,T,2\)"):
        model(torch.randn(1, 33, 3))
    with pytest.raises(ValueError, match="expects F=33"):
        model(torch.randn(1, 32, 3, 2))


@pytest.mark.parametrize("use_clstm", [True, False])
def test_spectrum_forward_backward_preserves_shape_and_finiteness(use_clstm):
    torch.manual_seed(2)
    model = _small_model(use_clstm=use_clstm)
    spectrum = torch.randn(1, 33, 3, 2, requires_grad=True)

    enhanced = model(spectrum, lengths=torch.tensor([3]), fs=16000)
    enhanced.square().mean().backward()

    assert enhanced.shape == spectrum.shape
    assert torch.isfinite(enhanced).all()
    assert spectrum.grad is not None
    assert torch.isfinite(spectrum.grad).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_zero_spectrum_has_finite_output_and_gradients():
    model = _small_model()
    spectrum = torch.zeros(1, 33, 3, 2, requires_grad=True)

    enhanced = model(spectrum)
    enhanced.sum().backward()

    # E 模式沿用上游 ``sqrt(real²+imag²+eps)`` 幅度地板，零谱输出可为微小有限值。
    assert torch.isfinite(enhanced).all()
    assert enhanced.abs().max() < 1e-3
    assert spectrum.grad is not None
    assert torch.isfinite(spectrum.grad).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_upstream_provenance_and_apache_license_are_present():
    module_dir = (
        Path(__file__).resolve().parents[1] / "tasks/se/models/modules/dccrn_impl"
    )
    provenance = (module_dir / "PROVENANCE").read_text(encoding="utf-8")
    license_text = (module_dir / "LICENSE").read_text(encoding="utf-8")

    assert "2e20e122d4b752cd8299450dae5181ff2bd3c29d" in provenance
    assert "dc_crn.py" in provenance
    assert "complexnn.py" in provenance
    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
