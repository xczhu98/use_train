"""RWSA-MambaUNet 官方结构与框架 wrapper 契约测试。"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

pytestmark = pytest.mark.unit


try:
    rwsa_wrapper = importlib.import_module(
        "tasks.se.models.gan.generators.rwsa_mamba_unet"
    )
except ImportError as exc:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, ModuleNotFoundError) and current.name:
            missing_root = current.name.split(".", 1)[0]
            if missing_root in {"mamba_ssm", "causal_conv1d"}:
                pytest.skip(
                    f"RWSA-MambaUNet 可选依赖不可用: {exc}",
                    allow_module_level=True,
                )
        current = current.__cause__
    raise

RWSAMambaUNetGenerator = rwsa_wrapper.RWSAMambaUNetGenerator


class _SpyRWSAModel(nn.Module):
    """记录 wrapper 传入的官方幅相，并原样返回。"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.last_magnitude = None
        self.last_phase = None

    def forward(self, magnitude, phase):
        self.last_magnitude = magnitude.detach().clone()
        self.last_phase = phase.detach().clone()
        enhanced_complex = torch.stack(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=-1
        )
        return magnitude, phase, enhanced_complex


@pytest.fixture
def spy_model(monkeypatch):
    monkeypatch.setattr(rwsa_wrapper, "MambAttentionSEUNet", _SpyRWSAModel)
    return RWSAMambaUNetGenerator()


def test_wrapper_uses_official_abs_angle_and_pads_time_to_multiple_of_four(spy_model):
    values = torch.tensor(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [-3.0, 4.0],
            [-3.0, -4.0],
            [3.0, -4.0],
        ],
        dtype=torch.float32,
    )
    noisy_spec = values.view(1, 1, 5, 2).expand(1, 256, 5, 2).clone()

    enhanced = spy_model(noisy_spec)

    padded = torch.nn.functional.pad(noisy_spec, (0, 0, 0, 3))
    padded_complex = torch.view_as_complex(padded.contiguous())
    assert spy_model.model.last_magnitude.shape == (1, 256, 8)
    torch.testing.assert_close(
        spy_model.model.last_magnitude,
        torch.abs(padded_complex).pow(spy_model.compress_factor),
    )
    torch.testing.assert_close(spy_model.model.last_phase, torch.angle(padded_complex))
    assert enhanced.shape == noisy_spec.shape
    torch.testing.assert_close(enhanced, noisy_spec, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("time_frames,expected_padded_frames", [(4, 4), (5, 8), (6, 8), (7, 8)])
def test_wrapper_preserves_shape_after_time_padding(spy_model, time_frames, expected_padded_frames):
    noisy_spec = torch.randn(2, 256, time_frames, 2)

    enhanced = spy_model(noisy_spec)

    assert spy_model.model.last_phase.shape == (2, 256, expected_padded_frames)
    assert enhanced.shape == noisy_spec.shape
    assert torch.isfinite(enhanced).all()


def test_wrapper_explicitly_consumes_stft_contract(monkeypatch):
    monkeypatch.setattr(rwsa_wrapper, "MambAttentionSEUNet", _SpyRWSAModel)

    model = RWSAMambaUNetGenerator(
        n_fft=510,
        hop_length=120,
        win_length=510,
        window="hann",
        center=False,
    )

    assert model.n_fft == 510
    assert model.hop_length == 120
    assert model.win_length == 510
    assert model.window == "hann"
    assert model.center is False
    assert model.model.cfg["stft_cfg"] == {
        "n_fft": 510,
        "hop_size": 120,
        "win_size": 510,
        "compress_factor": 0.3,
    }


def test_wrapper_rejects_unknown_kwargs_before_optional_model_construction(monkeypatch):
    monkeypatch.setattr(rwsa_wrapper, "MambAttentionSEUNet", _SpyRWSAModel)

    with pytest.raises(TypeError, match="unexpected keyword argument 'typo'"):
        RWSAMambaUNetGenerator(typo=True)


@pytest.mark.parametrize(
    "kwargs,error_type,error_match",
    [
        ({"n_fft": 512}, ValueError, r"F=n_fft//2\+1.*divisible by 8"),
        ({"win_length": 511}, ValueError, "must not exceed n_fft"),
        ({"compress_factor": 0.0}, ValueError, "compress_factor must be positive"),
        ({"hid_feature": 18}, ValueError, "hid_feature must be divisible by 8"),
        ({"num_tfmamba": 0}, ValueError, "num_tfmamba must be a positive integer"),
        ({"window": ""}, ValueError, "window must be a non-empty string"),
        ({"center": 1}, TypeError, "center must be bool"),
    ],
)
def test_wrapper_rejects_invalid_constructor_contract(
    monkeypatch, kwargs, error_type, error_match
):
    monkeypatch.setattr(rwsa_wrapper, "MambAttentionSEUNet", _SpyRWSAModel)

    with pytest.raises(error_type, match=error_match):
        RWSAMambaUNetGenerator(**kwargs)


@pytest.mark.parametrize(
    "noisy_spec,error_type,error_match",
    [
        (torch.randn(1, 256, 4), ValueError, "shape.*B, F, T, 2"),
        (torch.randn(1, 255, 4, 2), ValueError, "F=255, expected 256"),
        (torch.ones(1, 256, 4, 2, dtype=torch.int32), TypeError, "floating point"),
        (torch.empty(0, 256, 4, 2), ValueError, "must be non-empty"),
        (torch.empty(1, 256, 0, 2), ValueError, "must be non-empty"),
    ],
)
def test_wrapper_rejects_invalid_spectrum(spy_model, noisy_spec, error_type, error_match):
    with pytest.raises(error_type, match=error_match):
        spy_model(noisy_spec)


def test_wrapper_rejects_non_tensor_input(spy_model):
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        spy_model([[[[0.0, 0.0]]]])


@pytest.mark.parametrize(
    "variant,hid_feature,num_tfmamba,expected_dims",
    [
        ("XS", 16, 2, [16, 32, 48]),
        ("S", 16, 4, [16, 32, 48]),
        ("M", 24, 4, [24, 48, 72]),
    ],
)
def test_official_variants_and_resolution_wise_attention_identity(
    variant, hid_feature, num_tfmamba, expected_dims
):
    model = RWSAMambaUNetGenerator(
        hid_feature=hid_feature,
        num_tfmamba=num_tfmamba,
    ).model

    assert model.dim == expected_dims, variant
    assert len(model.TSMamba1_encoder) == num_tfmamba
    assert len(model.TSMamba2_encoder) == num_tfmamba
    assert len(model.TSMamba_middle) == num_tfmamba
    assert len(model.TSMamba2_decoder) == num_tfmamba
    assert len(model.TSMamba1_decoder) == num_tfmamba

    for index in range(num_tfmamba):
        level1_attention = model.shared_attention_lvl1[index]
        level2_attention = model.shared_attention_lvl2[index]
        assert model.TSMamba1_encoder[index].attention is level1_attention
        assert model.TSMamba1_decoder[index].attention is level1_attention
        assert model.TSMamba2_encoder[index].attention is level2_attention
        assert model.TSMamba2_decoder[index].attention is level2_attention
        assert level1_attention is not level2_attention
        assert model.TSMamba_middle[index].attention is not level1_attention
        assert model.TSMamba_middle[index].attention is not level2_attention


def test_training_config_matches_official_s_variant_and_epoch_scheduler():
    config_path = (
        Path(__file__).parents[1]
        / "configs/se/urgent2026/gan/rwsa_mamba_unet_metricgan_dynamic.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["stft"] == {
        "n_fft": 510,
        "hop_length": 120,
        "win_length": 510,
        "window": "hann",
        "center": True,
    }
    assert config["model"]["generator"]["args"]["hid_feature"] == 16
    assert config["model"]["generator"]["args"]["num_tfmamba"] == 4
    assert config["scheduler"]["generator"] == {
        "type": "ExponentialLR",
        "args": {"gamma": 0.99},
    }
    assert config["trainer"]["lr_schedule_per_step"] is False
