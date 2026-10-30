"""真实 gpuRIR fixed-array integration gates；不会以 fake backend 计入真实通过。"""

import numpy as np
import pytest

from core.data.simulation.gpurir_backend import generate_rir_group
from core.data.simulation.rir_config import OnlineRIRConfig
from core.data.simulation.rir_provider import OnlineRIRProvider
from core.data.simulation.rir_geometry import sample_rir_scene
from core.data.simulation_configs import SimulationConfigs


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize(
    ("profile", "channels"),
    [("linear_2ch_80mm", 2), ("linear_4ch_40mm", 4), ("linear_6ch_40mm", 6)],
)
def test_real_gpurir_array_profiles_return_k_c_l(profile, channels):
    cfg = SimulationConfigs.from_dict(
        {
            "rir_backend": "online",
            "spatial_enabled": True,
            "array_profile": profile,
            "reference_channel": 0,
            "gpurir_room_dim_range_m": [[3.0, 3.0], [3.0, 3.0], [2.5, 2.5]],
            "gpurir_rt60_range_sec": [0.15, 0.15],
            "gpurir_source_distance_range_m": [0.5, 0.5],
            "gpurir_timeout_sec": 30.0,
        },
        strict=True,
    )
    provider = OnlineRIRProvider.from_simulation(cfg)
    try:
        result = provider.sample_spatial_scene(8000, 3, np.random.default_rng(17), normalize=True)
        assert result.canonical_rirs.shape[0:2] == (3, channels)
        assert np.isfinite(result.canonical_rirs).all()
        assert result.geometry["array_profile"] == profile
        assert result.geometry["reference_channel"] == 0
        assert result.provenance["backend"] == "gpuRIR"
        assert result.generator_version == "gpuRIR-online-v4"
        assert result.rir_sha256 == result.content_hash
        facts = result.facts
        assert np.asarray(facts["room_dim"]).shape == (3,)
        assert len(facts["source_loc"]) == 3
        assert facts["provenance"]["timeout_sec"] == pytest.approx(30.0)
        assert facts["provenance"]["requested_timeout_sec"] == pytest.approx(30.0)
        assert 0.0 < facts["provenance"]["deadline_timeout_sec"] < 30.0
        assert 0.15 <= facts["rt60"] <= 0.15
    finally:
        from core.data.simulation.rir_service import close_online_rir_services

        close_online_rir_services()


@pytest.mark.unit
def test_gpurir_timeout_is_explicit_before_backend_work():
    cfg = OnlineRIRConfig.from_simulation(SimulationConfigs())
    plan = sample_rir_scene(cfg, 8000, 1, np.random.default_rng(2))

    class Backend:
        def beta_SabineEstimation(self, room, rt60):
            raise AssertionError("timeout gate must run before gpuRIR calls")

    with pytest.raises(TimeoutError, match="exceeded timeout"):
        generate_rir_group(plan, Backend(), timeout=0.0)


@pytest.mark.unit
def test_gpurir_unavailable_is_fail_fast(monkeypatch):
    from core.data.simulation import rir_service

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    cfg = SimulationConfigs.from_dict(
        {
            "rir_backend": "online",
            "spatial_enabled": True,
                        "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "gpurir_device": 0,
        },
        strict=True,
    )
    with pytest.raises(rir_service.OnlineRIRError, match="outside CUDA_VISIBLE_DEVICES"):
        rir_service.acquire_rir_client(OnlineRIRConfig.from_simulation(cfg))
