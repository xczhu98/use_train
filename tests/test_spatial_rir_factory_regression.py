"""统一 spatial RIR factory 的 backend 与 source identity 回归。"""

import numpy as np
import pytest

from core.data.simulation.rir_provider import (
    OnlineRIRProvider,
    build_spatial_rir_provider,
)
from core.data.simulation_configs import SimulationConfigs


pytestmark = pytest.mark.unit


def test_factory_routes_online_provider_and_preserves_source_uids():
    cfg = SimulationConfigs.from_dict(
        {
            "rir_backend": "online",
            "spatial_enabled": True,
                        "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
        },
        strict=True,
    )
    observed = {}

    class Client:
        def generate(self, plan):
            observed["source_ids"] = tuple(plan["source_ids"])
            return np.ones((2, 2, plan["num_samples"]), dtype=np.float32), {
                **plan,
                "array_profile": "linear_2ch_80mm",
                "rir_sha256": "raw-fixture",
            }

    provider = build_spatial_rir_provider(cfg, client=Client())
    assert isinstance(provider, OnlineRIRProvider)
    result = provider.get_rirs(
        8000,
        2,
        np.random.default_rng(0),
        source_uids=("target", "interferer"),
    )

    assert observed["source_ids"] == ("target", "interferer")
    assert result.rirs.shape[:2] == (2, 2)
    assert result.provenance["provider_contract"] == "[K,C,L]"
    assert result.facts["rir_sha256"] == result.content_hash
