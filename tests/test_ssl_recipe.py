"""SSL 官方动态语音配方的有限 profile、VAD 与 overlap 合同。"""

from __future__ import annotations

import numpy as np
import pytest

from tasks.ssl.data.recipe import (
    SSL_SIMULATION_PROFILES,
    apply_overlap_policy,
    clean_speech_with_webrtcvad,
    resolve_ssl_recipe,
)


pytestmark = pytest.mark.unit


def test_official_profiles_are_registered_and_unknown_values_fail_fast():
    assert SSL_SIMULATION_PROFILES == {
        "fnssl_official",
        "ipdnet_official",
        "ipdnet2_official",
    }
    assert resolve_ssl_recipe("fnssl_official").source_duration_seconds == 4.79
    with pytest.raises(ValueError, match="unsupported SSL simulation_profile"):
        resolve_ssl_recipe("arbitrary_profile")
    with pytest.raises(ValueError, match="require a registered"):
        resolve_ssl_recipe(None, vad_cleaning="none")


def test_recipe_overrides_cannot_change_registered_semantics():
    with pytest.raises(ValueError, match="does not match registered profile"):
        resolve_ssl_recipe("fnssl_official", overlap_policy="random_zero_mask_train")
    recipe = resolve_ssl_recipe(
        "ipdnet_official",
        vad_cleaning="webrtcvad_progressive",
        segment_sampling="chapter_random",
        overlap_policy="random_zero_mask_train",
    )
    assert recipe.to_provenance()["target_contract"] == "per_source_dpipd_pit"


def test_noise_source_boundary_is_explicit():
    from tasks.ssl.data.dynamic_dataset import _classify_ssl_noise_source

    assert _classify_ssl_noise_source("data/ssl/noise92/noise92.scp") == "noise92_scp"
    assert _classify_ssl_noise_source("data/ssl/noise/train.scp") == "other_noise_scp"
    assert _classify_ssl_noise_source(None) == "none"


def test_progressive_webrtc_vad_preserves_waveform_canvas_and_records_mode():
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    speech = (0.1 * np.sin(2 * np.pi * 220.0 * time)).astype(np.float32)
    cleaned, mask, facts = clean_speech_with_webrtcvad(speech, sample_rate)
    assert cleaned.shape == speech.shape
    assert mask.shape == speech.shape
    assert facts["algorithm"] == "progressive_aggressiveness"
    assert facts["attempted_modes"][0] == 3
    assert facts["attempted_modes"][-1] == facts["selected_mode"]
    assert facts["selected_mode"] in {1, 2, 3}


def test_ipdnet_overlap_policy_is_train_only_and_deterministic():
    speech = np.ones(16000, dtype=np.float32)
    train, train_facts = apply_overlap_policy(
        speech,
        policy="random_zero_mask_train",
        split="train",
        tracks=2,
        sample_rate=16000,
        rng=np.random.default_rng(4),
    )
    validation, validation_facts = apply_overlap_policy(
        speech,
        policy="random_zero_mask_train",
        split="validation",
        tracks=2,
        sample_rate=16000,
        rng=np.random.default_rng(4),
    )
    assert train_facts["applied"] is True
    assert 0 < train_facts["mask_samples"] <= 32000
    assert np.count_nonzero(train == 0.0) == train_facts["mask_samples"]
    assert validation_facts["applied"] is False
    np.testing.assert_array_equal(validation, speech)
