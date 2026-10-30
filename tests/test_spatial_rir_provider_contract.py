"""统一 spatial RIR provider interface 合同测试。"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from core.data.simulation.rir_provider import (
    OnlineRIRProvider,
    SpatialRIRResult,
    OfflineRIRProvider,
    build_spatial_rir_provider,
)
from core.data.simulation.rir_config import OnlineRIRConfig
from core.data.simulation_configs import SimulationConfigs


pytestmark = pytest.mark.unit


def test_factory_is_the_single_supplied_provider_construction_seam(tmp_path):
    value = np.ones((2, 3), dtype=np.float32)
    np.save(tmp_path / "a.npy", value)
    manifest = tmp_path / "rirs.json"
    import hashlib
    digest = hashlib.sha256(value.astype("<f4").tobytes()).hexdigest()
    manifest.write_text(json.dumps({"entries": {"a": {"full": "a.npy", "direct": "a.npy", "early": "a.npy"}}, "metadata": {
        "scene_uid": "scene-fixture", "actor_order": ["a"], "room": {"room_uid": "room-fixture"},
        "geometry": {"positions_m": [[-0.04, 0, 0], [0.04, 0, 0]]}, "generator_version": "fixture-v1",
        "track_identities": {"a": {"full": digest, "direct": digest, "early": digest}},
    }}), encoding="utf-8")
    cfg = SimpleNamespace(
        spatial_enabled=True,
        rir_backend="offline",
        array_profile="linear_2ch_80mm",
        reference_channel=0,
    )

    provider = build_spatial_rir_provider(
        cfg, rir_scp=manifest, geometry={"array_profile": "linear_2ch_80mm"}
    )

    assert isinstance(provider, OfflineRIRProvider)
    result = provider.get_rirs(8000, 1, None, source_uids=("a",))
    assert result.geometry["array_profile"] == "linear_2ch_80mm"
    assert result.content_hash == result.rir_sha256


def test_factory_requires_supplied_manifest_even_for_debug_provider():
    class Provider:
        def get_rirs(self, sample_rate, source_count, rng, *, source_uids=(), normalize=True):
            return np.ones((source_count, 2, 3), dtype=np.float32)

    cfg = SimpleNamespace(
        spatial_enabled=True,
        rir_backend="offline",
        array_profile="linear_2ch_80mm",
        reference_channel=0,
    )

    with pytest.raises(ValueError, match="rir_scp"):
        build_spatial_rir_provider(cfg, provider=Provider())


def test_supplied_provider_returns_typed_identity_and_geometry(tmp_path):
    bank = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    np.save(tmp_path / "a.npy", bank[0])
    np.save(tmp_path / "b.npy", bank[1])
    manifest = tmp_path / "rirs.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": {"a": "a.npy", "b": "b.npy"},
                "metadata": {
                    "geometry": {"array_profile": "linear_2ch_80mm"},
                    "generator_version": "fixture-v3",
                    "provenance": {"source": "unit-fixture"},
                },
            }
        ),
        encoding="utf-8",
    )
    result = OfflineRIRProvider(manifest, geometry={"positions_m": [[-0.04, 0, 0], [0.04, 0, 0]]}).get_rirs(
        8000, 2, None, source_uids=("a", "b")
    )
    assert isinstance(result, SpatialRIRResult)
    assert result.rirs.shape == (2, 2, 3)
    assert result.geometry["array_profile"] == "linear_2ch_80mm"
    assert result.generator_version == "fixture-v3"
    assert result.content_hash == result.rir_sha256 == result.rir_hash
    assert result.facts["provider_contract"] == "[K,C,L]"


def test_typed_result_preserves_legacy_mono_layout_and_canonical_projection():
    result = SpatialRIRResult(
        np.ones((2, 4), dtype=np.float32), {}, {}, "legacy-v1"
    )
    assert result.rirs.shape == (2, 4)
    assert result.legacy_rirs.shape == (2, 4)
    assert result.canonical_rirs.shape == (2, 1, 4)
    wave, facts = result
    assert wave.shape == (2, 4)
    assert facts["provider_contract"] == "[K,L]"


def test_online_provider_uses_same_typed_contract_without_cuda_client():
    config = OnlineRIRConfig.from_simulation(
        SimulationConfigs.from_dict(
            {
                "rir_backend": "online",
                "spatial_enabled": True,
                                "array_profile": "linear_2ch_80mm",
                "reference_channel": 0,
            },
            strict=True,
        )
    )

    class Client:
        def generate(self, plan):
            wave = np.ones(
                (1, 2, plan["num_samples"]), dtype=np.float32
            )
            return wave, {**plan, "rir_sha256": "raw-fixture"}

    result = OnlineRIRProvider(config, Client()).get_rirs(
        8000, 1, np.random.default_rng(1)
    )
    assert isinstance(result, SpatialRIRResult)
    assert result.rirs.shape[0:2] == (1, 2)
    assert result.geometry["array_profile"] == "linear_2ch_80mm"
    assert result.provenance["provider_contract"] == "[K,C,L]"
    assert result.facts["raw_rir_sha256"] == "raw-fixture"


def test_result_rejects_hash_tampering():
    with pytest.raises(ValueError, match="content_hash"):
        SpatialRIRResult(
            np.ones((1, 1, 2), dtype=np.float32), {}, {}, "fixture-v1", "bad"
        )


def test_supplied_provider_reads_exact_full_direct_early_track_set(tmp_path):
    full = np.array([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)
    direct = np.array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32)
    early = np.array([[1.0, 0.25, 0.0], [0.5, 0.125, 0.0]], dtype=np.float32)
    for name, values in (("full", full), ("direct", direct), ("early", early)):
        np.save(tmp_path / f"a_{name}.npy", np.stack([values[0], values[0] * 0.5]))
        np.save(tmp_path / f"b_{name}.npy", np.stack([values[1], values[1] * 0.5]))
    manifest = tmp_path / "tracks.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": {
                    "a": {"full": "a_full.npy", "direct": "a_direct.npy", "early": "a_early.npy"},
                    "b": {"full": "b_full.npy", "direct": "b_direct.npy", "early": "b_early.npy"},
                },
                "metadata": {
                    "geometry": {"array_profile": "linear_2ch_80mm"},
                    "provenance": {"scene_id": "scene-1", "source_order": ["a", "b"]},
                    "generator_version": "fixture-v4",
                },
            }
        ),
        encoding="utf-8",
    )
    result = OfflineRIRProvider(manifest, geometry={"positions_m": [[-0.04, 0, 0], [0.04, 0, 0]]}).get_rirs(
        8000, 2, None, source_uids=("a", "b")
    )
    assert result.full.rirs.shape == result.direct.rirs.shape == result.early.rirs.shape == (2, 2, 3)
    np.testing.assert_array_equal(result.direct.rirs[:, 0], direct)
    assert result.facts["direct_path_exact"] is True
    assert result.facts["track_set"]["direct"] == result.direct.content_hash
    assert result.canonical_rirs.shape == result.full.canonical_rirs.shape


def test_supplied_provider_keeps_full_only_manifest_compatible(tmp_path):
    bank = np.ones((2, 2, 3), dtype=np.float32)
    np.save(tmp_path / "a.npy", bank[0])
    np.save(tmp_path / "b.npy", bank[1])
    manifest = tmp_path / "full.json"
    manifest.write_text(json.dumps({"entries": {"a": "a.npy", "b": "b.npy"}}), encoding="utf-8")
    result = OfflineRIRProvider(manifest, geometry={"positions_m": [[-0.04, 0, 0], [0.04, 0, 0]]}).get_rirs(
        8000, 2, None, source_uids=("a", "b")
    )
    assert isinstance(result, SpatialRIRResult)
    assert result.rirs.shape == (2, 2, 3)


def test_supplied_provider_fails_fast_when_track_set_lacks_direct(tmp_path):
    bank = np.ones((2, 2, 3), dtype=np.float32)
    for key, index in (("a", 0), ("b", 1)):
        np.save(tmp_path / f"{key}.npy", bank[index])
    manifest = tmp_path / "missing_direct.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": {
                    "a": {"full": "a.npy", "early": "a.npy"},
                    "b": {"full": "b.npy", "early": "b.npy"},
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing 'direct' track"):
        OfflineRIRProvider(manifest, geometry={"positions_m": [[-0.04, 0, 0], [0.04, 0, 0]]}).get_rirs(
            8000, 2, None, source_uids=("a", "b")
        )


def test_public_supplied_metadata_excludes_entry_bookkeeping(tmp_path):
    from core.data.spatial import load_offline_rir_bank_with_metadata

    np.save(tmp_path / "a.npy", np.ones((1, 3), dtype=np.float32))
    manifest = tmp_path / "public.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": {"a": "a.npy"},
                "metadata": {
                    "generator_version": "fixture-v5",
                    "_entry_modes": {"a": "spoofed"},
                    "_entry_tracks": {"a": {"direct": "spoofed.npy"}},
                },
            }
        ),
        encoding="utf-8",
    )
    _, metadata = load_offline_rir_bank_with_metadata(
        manifest, ("a",), geometry={"positions_m": [[0.0, 0.0, 0.0]]}
    )
    assert metadata == {"generator_version": "fixture-v5"}
    assert "_entry_modes" not in metadata
    assert "_entry_tracks" not in metadata
