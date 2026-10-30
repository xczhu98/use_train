"""Tests for the TSE-local stateless EnrollmentAugmentor."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from core.data.dataset_build import expand_simulation_cfg_tse_dataset_keys
from test_multisource_scene_simulation import sources
from tasks.tse.data.enrollment_augment import EnrollmentAugmentor


def test_dynamic_mapping_is_hoisted_before_strict_simulation_parse():
    section = {
        "simulation_cfg": {
            "enrollment_augment": {"enabled": True, "prob_eq": 0.5},
            "prob_reverberation": 0.0,
        }
    }
    expand_simulation_cfg_tse_dataset_keys(section)
    assert section["enrollment_augment"] == {"enabled": True, "prob_eq": 0.5}
    assert "enrollment_augment" not in section["simulation_cfg"]
    assert section["simulation_cfg"] == {"prob_reverberation": 0.0}


@pytest.mark.parametrize(
    "config, error",
    [
        ({"unknown": True}, "unknown keys"),
        ({"enabled": 1}, "must be a bool"),
        ({"prob_eq": 1.1}, "must be in"),
        ({"prob_eq": float("nan")}, "must be finite"),
        ({"eq_db_range": [1.0]}, "two-item"),
        ({"eq_db_range": [2.0, -2.0]}, "low <= high"),
        ({"rir_backend": "pool"}, "rir_backend"),
        ({"rir_backend": "online", "rir_scp": "x.scp"}, "does not take rir_scp"),
    ],
)
def test_config_rejects_invalid_mapping(config, error):
    with pytest.raises((TypeError, ValueError), match=error):
        EnrollmentAugmentor.from_config(config)


def test_disabled_and_zero_probability_do_not_consume_rng():
    waveform = np.linspace(-0.2, 0.2, 4096, dtype=np.float32)
    for config in (None, {"enabled": True, "prob_eq": 0.0}):
        augmentor = EnrollmentAugmentor.from_config(config)
        rng = np.random.default_rng(91)
        expected = np.random.default_rng(91)
        output, facts = augmentor.apply(waveform, 16000, rng)
        np.testing.assert_array_equal(output, waveform)
        assert not np.shares_memory(output, waveform)
        assert rng.bit_generator.state == expected.bit_generator.state
        assert facts["applied"] is False
        json.dumps(facts, allow_nan=False)


def test_enabled_eq_preserves_input_length_finiteness_and_determinism():
    waveform = (
        np.sin(np.arange(4096, dtype=np.float32) * 0.031)
        + 0.1 * np.sin(np.arange(4096, dtype=np.float32) * 0.19)
    ).astype(np.float32)
    original = waveform.copy()
    augmentor = EnrollmentAugmentor.from_config(
        {"enabled": True, "prob_eq": 1.0, "eq_db_range": [-9.0, -9.0]}
    )
    first, facts = augmentor.apply(waveform, 16000, np.random.default_rng(7))
    second, _ = augmentor.apply(waveform, 16000, np.random.default_rng(7))

    np.testing.assert_array_equal(waveform, original)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32
    assert first.ndim == 1 and first.shape == waveform.shape
    assert np.isfinite(first).all()
    assert not np.array_equal(first, waveform)
    assert facts["operation"] == "random_eq"
    assert facts["applied"] is True
    json.dumps(facts, allow_nan=False)


@pytest.mark.parametrize(
    "waveform",
    [np.ones((1, 4), dtype=np.float32), np.ones(4, dtype=np.float64)],
)
def test_apply_rejects_non_contract_input(waveform):
    with pytest.raises(ValueError, match="1-D float32"):
        EnrollmentAugmentor.from_config({"enabled": True}).apply(
            waveform, 16000, np.random.default_rng(0)
        )


def test_full_chain_preserves_finite_fixed_length():
    waveform = np.sin(np.arange(3200, dtype=np.float32) * 0.05).astype(np.float32)
    augmentor = EnrollmentAugmentor.from_config(
        {
            "enabled": True,
            "prob_eq": 0.0,
            "target_duration_seconds": 0.1,
            "target_db": -12.0,
            "fit_policy": "fixed",
            "prob_reverb": 1.0,
            "prob_noise": 1.0,
            "snr_range": [8.0, 8.0],
            "prob_eq_post": 1.0,
            "prob_lowpass": 1.0,
            "eq_db_range": [-3.0, -3.0],
            "lowpass_cutoff_range": [3000.0, 3000.0],
        }
    )
    rir = np.zeros((1, 32), dtype=np.float32)
    rir[0, 0] = 1.0
    noise = np.full((1, 1600), 0.05, dtype=np.float32)
    augmented, facts = augmentor.apply(
        waveform,
        16000,
        np.random.default_rng(3),
        rir_provider=lambda _sr, _rng: (rir, {"path": "rir.wav"}),
        noise_provider=lambda _sr, _length, _rng: (noise, {"path": "noise.wav"}),
    )
    assert augmented.shape == waveform.shape
    assert augmented.dtype == np.float32 and np.isfinite(augmented).all()
    assert {"rms", "rir", "noise", "eq_post", "lowpass"}.issubset(facts["operations"])
    assert facts["peak_guard_rescaled"] in {True, False}
    json.dumps(facts, allow_nan=False)


def test_disabled_dynamic_preparation_keeps_optional_rng_cursor():
    augmentor=EnrollmentAugmentor.from_config(None).for_dynamic(duration_seconds=.1,target_db=-20.,fit_policy='fixed')
    rng=np.random.default_rng(71);expected=np.random.default_rng(71)
    output,facts=augmentor.apply(np.ones(1600,dtype=np.float32)*.1,16000,rng)
    assert rng.bit_generator.state==expected.bit_generator.state
    assert output.shape==(1600,) and facts['operations']==['rms']


def test_rir_backend_online_uses_online_provider(monkeypatch, sources):
    from test_tse_seven_scenes import dataset_for

    captured = {}

    class FakeProvider:
        def sample_scene(self, sample_rate, source_count, rng, *, normalize, distance_range=None):
            captured["sample_rate"] = sample_rate
            captured["source_count"] = source_count
            captured["normalize"] = normalize
            captured["distance_range"] = distance_range
            wave = np.zeros((1, 64), np.float32)
            wave[0, 10] = 1.0
            return wave, {"group_uid": "fake-online", "dist": [0.1], "rir_sha256": "fake"}

    monkeypatch.setattr(
        "core.data.simulation.rir_provider.OnlineRIRProvider.from_simulation",
        classmethod(lambda cls, cfg: FakeProvider()),
    )
    ds = dataset_for(
        sources,
        "prob_target_only",
        simulation_cfg={"gpurir_enrollment_distance_range_m": [0.06, 0.18]},
        enrollment_augment={
            "enabled": True,
            "prob_eq": 0.0,
            "prob_reverb": 1.0,
            "rir_backend": "online",
        },
    )
    sample = ds[0]
    cue = sample["targets"][0]["cues"].speaker_waveform
    assert cue is not None and cue.present and cue.value is not None
    assert cue.length == len(cue.value)
    enroll = np.asarray(cue.value)
    assert captured["source_count"] == 1
    assert captured["normalize"] is True
    assert captured["distance_range"] == (0.06, 0.18)
    assert np.isfinite(enroll).all() and np.any(enroll)


def test_rir_backend_online_without_provider_fails_closed(monkeypatch, sources):
    from test_tse_seven_scenes import dataset_for

    monkeypatch.setattr(
        "core.data.simulation.rir_provider.OnlineRIRProvider.from_simulation",
        classmethod(lambda cls, cfg: None),
    )
    with pytest.raises(ValueError, match="requires the online RIR provider"):
        dataset_for(
            sources,
            "prob_target_only",
            enrollment_augment={
                "enabled": True,
                "prob_reverb": 1.0,
            "rir_backend": "online",
            },
        )


pytestmark = [pytest.mark.unit]
