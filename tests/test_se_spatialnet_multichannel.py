import pytest
import torch

from tasks.se.models import build_model


@pytest.mark.unit
def test_spatialnet_se_builds_and_projects_multichannel_spectrum_to_mono():
    model = build_model(
        {
            "type": "SpatialNet_SE",
            "args": {
                "n_fft": 64,
                "n_channel": 2,
                "num_layers": 1,
                "dim_hidden": 8,
                "dim_ffn": 16,
                "dim_squeeze": 4,
                "num_heads": 2,
                "conv_groups": (1, 1),
            },
        }
    )
    output = model(torch.randn(2, 2, 33, 5, 2))
    assert output.shape == (2, 33, 5, 2)


@pytest.mark.unit
def test_spatialnet_se_keeps_mono_layout_compatible():
    model = build_model(
        {
            "type": "SpatialNet_SE",
            "args": {
                "n_fft": 64,
                "num_layers": 1,
                "dim_hidden": 8,
                "dim_ffn": 16,
                "dim_squeeze": 4,
                "num_heads": 2,
                "conv_groups": (1, 1),
            },
        }
    )
    assert model(torch.randn(2, 33, 5, 2)).shape == (2, 33, 5, 2)
