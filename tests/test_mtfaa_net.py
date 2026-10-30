"""MTFAA-Net 注册、频谱接口与共享 mask head 契约测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tasks.se.models
from tasks.aec.models.modules.mtfaa_mask import MTFAAMaskHead
from core.models.build import build_model
from tasks.se.models.discriminative.mtfaa_net import MTFAANet


pytestmark = pytest.mark.unit


def _stft_config() -> dict:
    return {
        "n_fft": 512,
        "hop_length": 128,
        "win_length": 512,
        "window": "hann",
        "center": True,
    }


def _tiny_model_args() -> dict:
    return {
        "channels": [8, 8],
        "filter_bands": 16,
        "bottleneck_layers": 1,
        "tfcm_layers": 1,
    }


def test_registry_build_consumes_stft_context():
    assert tasks.se.models._MODEL_REGISTRY_MAP["MTFAANet"] == (
        "tasks.se.models.discriminative.mtfaa_net"
    )
    model = build_model(
        {"type": "MTFAANet", "args": _tiny_model_args()},
        stft_cfg=_stft_config(),
    )

    assert isinstance(model, MTFAANet)
    assert (model.n_fft, model.hop_length, model.win_length) == (512, 128, 512)
    assert model.window == "hann"
    assert model.center is True


def test_constructor_rejects_unknown_kwargs():
    with pytest.raises(TypeError, match="unexpected keyword argument.*tfcm_layres"):
        MTFAANet(tfcm_layres=1)


def test_spectrum_forward_preserves_layout_and_is_finite():
    torch.manual_seed(0)
    model = MTFAANet(**_tiny_model_args()).eval()
    noisy_spectrum = torch.randn(1, 257, 4, 2)

    with torch.inference_mode():
        enhanced = model(noisy_spectrum, fs=16000)

    assert enhanced.shape == noisy_spectrum.shape
    assert torch.isfinite(enhanced).all()


def test_forward_rejects_wrong_frequency_bins_and_sample_rate():
    model = MTFAANet(**_tiny_model_args())
    with pytest.raises(ValueError, match="257 frequency bins"):
        model(torch.randn(1, 129, 3, 2))
    with pytest.raises(ValueError, match="configured for 16000 Hz"):
        model(torch.randn(1, 257, 3, 2), fs=48000)


def test_shared_mask_head_shape_and_causal_time_contract():
    torch.manual_seed(1)
    head = MTFAAMaskHead(4).eval()
    features = torch.randn(1, 4, 6, 17)
    spectrum = torch.randn(1, 17, 6, 2)

    perturbed_features = features.clone()
    perturbed_spectrum = spectrum.clone()
    perturbed_features[:, :, 4:] += 100.0
    perturbed_spectrum[:, :, 4:] += 100.0

    with torch.inference_mode():
        output = head(features, spectrum)
        perturbed_output = head(perturbed_features, perturbed_spectrum)

    assert output.shape == spectrum.shape
    torch.testing.assert_close(output[:, :, :4], perturbed_output[:, :, :4])


def test_provenance_keeps_unofficial_implementation_warning_and_license():
    module_dir = (
        Path(__file__).resolve().parents[1] / "tasks/se/models/modules/mtfaa_impl"
    )
    provenance = (module_dir / "PROVENANCE").read_text(encoding="utf-8")
    license_text = (module_dir / "LICENSE").read_text(encoding="utf-8")

    assert "eb3b1f33d7c5178f238076938c99acaec9e2e904" in provenance
    assert "Implementation details may not be consistent with the paper" in provenance
    assert "不继承论文或参考仓库报告的指标" in provenance
    assert "MIT License" in license_text
    assert "Copyright (c) 2022 Shimin Zhang" in license_text
