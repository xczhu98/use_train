"""Shared noise source policy contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from core.data.dynamic_dataset_base import BaseDynamicMixingDataset
from core.data.noise_source_policy import (
    NoiseApplicationRole,
    NoiseCacheProvenance,
    NoiseCategory,
    NoiseSourceFamily,
    NoiseSourcePolicy,
)
from core.data.simulation.resource_store import ResourceStore


pytestmark = pytest.mark.unit


def test_noise_source_policy_serializes_family_category_role_and_cache_provenance():
    policy = NoiseSourcePolicy(
        family=NoiseSourceFamily.MUSIC,
        category=NoiseCategory(category_id=9, category_name="Music"),
        application_role=NoiseApplicationRole.TSE_ENROLLMENT_ADDITIVE,
        cache_provenance=NoiseCacheProvenance.SHARED_AUDIO_CACHE_UNKNOWN,
        uid="music-1",
        path="/tmp/music.wav",
    )

    assert policy.to_dict() == {
        "family": "music",
        "category": {"id": 9, "name": "Music"},
        "application_role": "tse_enrollment_additive",
        "cache_provenance": "shared_audio_cache_unknown",
        "uid": "music-1",
        "path": "/tmp/music.wav",
    }


def test_noise_source_policy_rejects_invalid_category_identity():
    with pytest.raises(ValueError, match="category_id"):
        NoiseCategory(category_id=-1, category_name="Music")


def _reuse_dataset() -> BaseDynamicMixingDataset:
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.strict_load = True
    dataset._noise_manifest_metadata = {
        "ordinary-1": {"category_id": 3, "balance_category": "Traffic"},
        "music-1": {"category_id": 9, "balance_category": "Music"},
    }
    dataset._noise_single_reuse_cache = None
    dataset._noise_dual_reuse_cache = None
    dataset._wind_reuse_cache = None
    dataset.resource_store = ResourceStore(strict_load=True)
    return dataset


@pytest.mark.parametrize(
    ("family", "uid", "path"),
    [
        ("ordinary", "ordinary-1", "/noise/ordinary.wav"),
        ("music", "music-1", "/noise/music.wav"),
    ],
)
def test_additive_reuse_keeps_real_policy_identity(family, uid, path):
    dataset = _reuse_dataset()

    def read_one(_sr, _duration, _rng):
        dataset._record_noise_source(family=family, uid=uid, path=path)
        return np.ones((1, 4), dtype=np.float32)

    dataset._read_one_additive_noise_scp_only = read_one
    first_record = {"noise_sources": [], "noise_source_policies": [], "noise_assets": []}
    dataset._active_acoustic_plan_record = first_record
    dataset._mix_acoustic_read_additive("single", 16000, 4, np.random.default_rng(1), True)

    hit_record = {"noise_sources": [], "noise_source_policies": [], "noise_assets": []}
    dataset._active_acoustic_plan_record = hit_record
    dataset._mix_acoustic_read_additive("single", 16000, 4, np.random.default_rng(2), True)

    policy = hit_record["noise_source_policies"][-1]
    source = hit_record["noise_sources"][-1]
    assert policy.family.value == family
    assert policy.category.to_dict() == {
        "id": 3 if family == "ordinary" else 9,
        "name": "Traffic" if family == "ordinary" else "Music",
    }
    assert policy.uid == uid
    assert policy.path == path
    assert policy.cache_provenance is NoiseCacheProvenance.DATASET_REUSE_CACHE
    assert source.family == family
    assert source.uid == uid
    assert source.path == path
    assert source.cache_hit is True


def test_wind_reuse_keeps_real_policy_identity(monkeypatch):
    dataset = _reuse_dataset()
    dataset.wind_noises = {16000: {"wind-1": None}}
    dataset.wn_flt = {"wind-1": "/noise/wind.wav"}
    dataset._select_sample = lambda *_args, **_kwargs: "wind-1"
    dataset.resource_store = ResourceStore(
        strict_load=True,
        noise_reader=lambda *_args, **_kwargs: np.ones((1, 4), dtype=np.float32),
    )

    first_record = {"noise_sources": [], "noise_source_policies": [], "noise_assets": []}
    dataset._active_acoustic_plan_record = first_record
    dataset._mix_acoustic_get_wind_sample(16000, 4, np.random.default_rng(1), True)

    hit_record = {"noise_sources": [], "noise_source_policies": [], "noise_assets": []}
    dataset._active_acoustic_plan_record = hit_record
    dataset._mix_acoustic_get_wind_sample(16000, 4, np.random.default_rng(2), True)

    policy = hit_record["noise_source_policies"][-1]
    source = hit_record["noise_sources"][-1]
    assert policy.family is NoiseSourceFamily.WIND
    assert policy.uid == "wind-1"
    assert policy.path == "/noise/wind.wav"
    assert policy.cache_provenance is NoiseCacheProvenance.DATASET_REUSE_CACHE
    assert source.family == "wind"
    assert source.uid == "wind-1"
    assert source.path == "/noise/wind.wav"
    assert source.cache_hit is True


def test_legacy_reuse_tuple_does_not_create_fake_identity():
    dataset = _reuse_dataset()
    dataset._noise_single_reuse_cache = (16000, np.ones((1, 4), dtype=np.float32))
    record = {"noise_sources": [], "noise_source_policies": [], "noise_assets": []}
    dataset._active_acoustic_plan_record = record

    sample = dataset._mix_acoustic_read_additive(
        "single", 16000, 4, np.random.default_rng(1), True
    )

    np.testing.assert_array_equal(sample, np.ones((1, 4), dtype=np.float32))
    assert record["noise_sources"] == []
    assert record["noise_source_policies"] == []


def test_set_epoch_clears_reuse_tuples_with_identity_sidecars(monkeypatch):
    dataset = _reuse_dataset()
    cached = (16000, np.ones((1, 4), dtype=np.float32), ())
    dataset._rir_reuse_cache = None
    dataset._noise_only_reuse_cache = cached
    dataset._noise_single_reuse_cache = cached
    dataset._noise_dual_reuse_cache = cached
    dataset._wind_reuse_cache = cached
    dataset._clear_extra_epoch_caches = lambda: None
    dataset._worker_epoch = np.zeros(1, dtype=np.int64)
    dataset._worker_base_seed = np.zeros(1, dtype=np.int64)
    dataset._hp_taps = None

    dataset.set_epoch(1, base_seed=7)

    assert dataset._noise_only_reuse_cache is None
    assert dataset._noise_single_reuse_cache is None
    assert dataset._noise_dual_reuse_cache is None
    assert dataset._wind_reuse_cache is None
