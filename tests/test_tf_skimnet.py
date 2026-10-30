"""TF-SkiMNet registry、结构、因果性与来源边界测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tasks.se.models  # noqa: F401  安装 SE lazy registry map
from core.models.build import build_model
from core.utils.config import load_merged_config
from tasks.se.models.discriminative.tf_skimnet import TFSkiMNet


pytestmark = pytest.mark.unit


def _small_stft() -> dict[str, object]:
    return {
        "n_fft": 16,
        "hop_length": 8,
        "win_length": 16,
        "window": "hann",
        "center": True,
        "normalized": False,
    }


def _small_model(**overrides: object) -> TFSkiMNet:
    args: dict[str, object] = {
        **_small_stft(),
        "num_blocks": 2,
        "hidden_size": 4,
        "encoder_kernel_size": 3,
        "local_frequency_kernel": 3,
        "local_temporal_kernel": 3,
        "local_temporal_groups": 2,
    }
    args.update(overrides)
    return TFSkiMNet(**args)


def test_lazy_registry_builds_default_paper_structure() -> None:
    model = build_model(
        {"type": "TFSkiMNet", "args": {}},
        stft_cfg={
            "n_fft": 320,
            "hop_length": 160,
            "win_length": 320,
            "window": "hann",
            "center": True,
            "normalized": False,
        },
    )

    assert isinstance(model, TFSkiMNet)
    assert model.num_blocks == 6
    assert model.hidden_size == 32
    assert len(model.backbone.blocks) == 6
    assert len(model.backbone.hidden_state_networks) == 5
    assert len(model.backbone.cell_state_networks) == 5
    assert sum(parameter.numel() for parameter in model.parameters()) == 495_842


def test_voicebank_config_consumes_stft_and_skim_recurrence_parameters() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs/se/voicebank_demand/tf_skimnet_causal_offline_mag_complex_l1.yaml"
    )
    config = load_merged_config(config_path)

    model = build_model(config["model"], stft_cfg=config["stft"])

    assert isinstance(model, TFSkiMNet)
    assert (model.n_fft, model.hop_length, model.win_length) == (320, 160, 320)
    assert model.window == "hann"
    assert model.center is True
    assert model.normalized is False
    assert len(model.backbone.blocks) == config["model"]["args"]["num_blocks"] == 6
    assert model.backbone.hidden_size == config["model"]["args"]["hidden_size"] == 32


def test_blocks_keep_frequency_inplace_and_use_expected_recurrence_directions() -> None:
    model = _small_model(num_blocks=3)
    backbone = model.backbone

    assert len(backbone.blocks) == 3
    assert len(backbone.hidden_state_networks) == 2
    assert len(backbone.cell_state_networks) == 2
    for block in backbone.blocks:
        assert block.local_frequency.conv.kernel_size == (1, 3)
        assert block.local_frequency.conv.stride == (1, 1)
        assert block.global_frequency.lstm.bidirectional is True
        assert block.local_temporal is not None
        assert block.local_temporal.conv.conv.kernel_size == (3,)
        assert block.local_temporal.conv.conv.groups == 2
    for transition in [
        *backbone.hidden_state_networks,
        *backbone.cell_state_networks,
    ]:
        assert transition.lstm.bidirectional is False
        assert transition.lstm.input_size == 8
        assert transition.lstm.hidden_size == 8


def test_skim_memory_and_local_temporal_paths_are_configurable() -> None:
    model = _small_model(
        num_blocks=3,
        use_global_temporal=False,
        use_local_temporal=False,
    )

    assert model.backbone.use_global_temporal is False
    assert len(model.backbone.hidden_state_networks) == 0
    assert len(model.backbone.cell_state_networks) == 0
    assert all(block.local_temporal is None for block in model.backbone.blocks)


def test_forward_backward_preserves_complex_spectrum_layout() -> None:
    torch.manual_seed(0)
    model = _small_model()
    spectrum = torch.randn(2, 9, 5, 2, requires_grad=True)

    enhanced = model(spectrum, lengths=torch.tensor([40, 32]), fs=16000)
    enhanced.square().mean().backward()

    assert enhanced.shape == spectrum.shape
    assert torch.isfinite(enhanced).all()
    assert spectrum.grad is not None
    assert torch.isfinite(spectrum.grad).all()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_future_frames_do_not_change_prefix_output() -> None:
    torch.manual_seed(1)
    model = _small_model(num_blocks=2).eval()
    spectrum = torch.randn(1, 9, 7, 2)
    changed = spectrum.clone()
    changed[:, :, 4:, :] = torch.randn_like(changed[:, :, 4:, :])

    with torch.inference_mode():
        reference = model(spectrum)
        perturbed = model(changed)

    assert torch.allclose(reference[:, :, :4], perturbed[:, :, :4], atol=1e-6, rtol=1e-6)


def test_constructor_and_forward_fail_closed_on_invalid_contracts() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument.*hidden_szie"):
        TFSkiMNet(hidden_szie=32)
    with pytest.raises(ValueError, match="divisible"):
        _small_model(hidden_size=5)
    with pytest.raises(ValueError, match="positive odd integer"):
        _small_model(local_temporal_kernel=4)

    model = _small_model()
    with pytest.raises(ValueError, match="spectrum input"):
        model(torch.randn(1, 9, 7))
    with pytest.raises(ValueError, match="expects 9 frequency bins"):
        model(torch.randn(1, 8, 7, 2))
    with pytest.raises(ValueError, match="lengths must have shape"):
        model(torch.randn(1, 9, 7, 2), lengths=torch.ones(2))
    with pytest.raises(ValueError, match="requires fs=16000"):
        model(torch.randn(1, 9, 7, 2), fs=48000)


def test_provenance_records_unlicensed_model_only_boundary() -> None:
    provenance_path = (
        Path(__file__).resolve().parents[1]
        / "tasks/se/models/modules/tf_skimnet_impl/PROVENANCE"
    )
    provenance = provenance_path.read_text(encoding="utf-8")

    assert "2dcb11b84cc042ddda3733fbcba10e1695dee016" in provenance
    assert "f79bb4648d65ad2323476b703c62f4ada952f40a" in provenance
    assert "无许可证" in provenance
    assert "仅模型级移植" in provenance
    assert "不声称复现论文" in provenance
