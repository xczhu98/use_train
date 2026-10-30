"""真实 gpuRIR fixed-array integration gates."""

import importlib.util

import numpy as np
import pytest
import torch

from core.data.simulation.rir_provider import OnlineRIRProvider, SpatialRIRResult
from core.data.simulation.rir_service import close_online_rir_services
from core.data.simulation_configs import SimulationConfigs


pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable for real gpuRIR gate"),
    pytest.mark.skipif(importlib.util.find_spec("gpuRIR") is None, reason="gpuRIR unavailable"),
]


@pytest.mark.parametrize(
    ("profile", "channels"),
    (
        ("linear_2ch_80mm", 2),
        ("linear_4ch_40mm", 4),
        ("linear_6ch_40mm", 6),
    ),
)
def test_real_gpurir_fixed_array_profiles(profile, channels):
    cfg = SimulationConfigs.from_dict(
        {
            "rir_backend": "online",
            "spatial_enabled": True,
            "array_profile": profile,
            "reference_channel": channels - 1,
                        "gpurir_room_dim_range_m": [[3.5, 3.5], [3.5, 3.5], [2.8, 2.8]],
            "gpurir_rt60_range_sec": [0.15, 0.15],
            "gpurir_source_distance_range_m": [0.6, 0.6],
            "gpurir_timeout_sec": 30.0,
        },
        strict=True,
    )
    provider = OnlineRIRProvider.from_simulation(cfg)
    try:
        result = provider.sample_spatial_scene(
            8000, 2, np.random.default_rng(17), normalize=True
        )
        assert isinstance(result, SpatialRIRResult)
        assert result.rirs.shape[:2] == (2, channels)
        assert result.geometry["array_profile"] == profile
        assert result.geometry["reference_channel"] == channels - 1
        assert np.asarray(result.details["source_loc"]).shape == (2, 3)
        assert result.details["room_dim"] == pytest.approx([3.5, 3.5, 2.8])
        assert result.details["rt60"] == pytest.approx(0.15)
        assert result.generator_version == "gpuRIR-online-v4"
        assert result.provenance["timeout_sec"] == pytest.approx(30.0)
        assert result.rir_sha256 == result.details["rir_sha256"]
    finally:
        close_online_rir_services()
