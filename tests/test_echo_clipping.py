"""Unit tests for tasks/aec/data/echo_clipping.py."""

from __future__ import annotations

import numpy as np
import pytest

from tasks.aec.data.echo_clipping import (
    apply_absolute_rail_clip,
    apply_peak_relative_clip,
    apply_playback_overload,
    normalize_echo_clipping_config,
    sample_peak_relative_clip_params,
    select_echo_clip_path,
    ser_db_from_power_ratio,
    summarize_absolute_rail_facts,
)


def test_normalize_echo_clipping_config_returns_canonical_defaults():
    assert normalize_echo_clipping_config(None) == {
        "enabled": False,
        "prob": 0.25,
        "path_probs": {
            "post_ser_distortion": 0.5,
            "pre_ser_overload": 0.3,
            "synced_ref_clip": 0.2,
        },
        "paths": {
            "post_ser_distortion": {
                "variant_probs": {"hard": 0.5, "tanh": 0.5},
                "peak_relative_threshold_range": (0.5, 0.95),
            },
            "pre_ser_overload": {
                "variant_probs": {"fixed_rail": 0.5, "tanh": 0.5},
                "overload_boost_db_range": (3.0, 18.0),
                "rail_dbfs_range": (-6.0, -1.0),
            },
            "synced_ref_clip": {
                "variant_probs": {"fixed_rail": 0.5, "tanh": 0.5},
                "rail_dbfs_range": (-6.0, -1.0),
            },
        },
    }


def test_normalize_echo_clipping_config_accepts_target_canonical_config():
    raw = {
        "enabled": True,
        "prob": 1,
        "path_probs": {
            "post_ser_distortion": 1,
            "pre_ser_overload": 2,
            "synced_ref_clip": 3,
        },
        "paths": {
            "post_ser_distortion": {
                "variant_probs": {"hard": 1, "tanh": 0},
                "peak_relative_threshold_range": [0.25, 0.75],
            },
            "pre_ser_overload": {
                "variant_probs": {"fixed_rail": 1, "tanh": 2},
                "overload_boost_db_range": [6, 12],
                "rail_dbfs_range": [-9, -3],
            },
            "synced_ref_clip": {
                "variant_probs": {"fixed_rail": 0, "tanh": 1},
                "rail_dbfs_range": [-12, -6],
            },
        },
    }

    assert normalize_echo_clipping_config(raw) == {
        "enabled": True,
        "prob": 1.0,
        "path_probs": {
            "post_ser_distortion": 1.0,
            "pre_ser_overload": 2.0,
            "synced_ref_clip": 3.0,
        },
        "paths": {
            "post_ser_distortion": {
                "variant_probs": {"hard": 1.0, "tanh": 0.0},
                "peak_relative_threshold_range": (0.25, 0.75),
            },
            "pre_ser_overload": {
                "variant_probs": {"fixed_rail": 1.0, "tanh": 2.0},
                "overload_boost_db_range": (6.0, 12.0),
                "rail_dbfs_range": (-9.0, -3.0),
            },
            "synced_ref_clip": {
                "variant_probs": {"fixed_rail": 0.0, "tanh": 1.0},
                "rail_dbfs_range": (-12.0, -6.0),
            },
        },
    }


@pytest.mark.parametrize("enabled", [0, 1, "true", None, [], {}])
def test_normalize_echo_clipping_config_rejects_non_boolean_enabled(enabled):
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        normalize_echo_clipping_config({"enabled": enabled})


@pytest.mark.parametrize("prob", [True, False, "0.5", None, [], {}])
def test_normalize_echo_clipping_config_rejects_non_numeric_prob(prob):
    with pytest.raises(ValueError, match="prob must be a finite probability"):
        normalize_echo_clipping_config({"prob": prob})


@pytest.mark.parametrize("prob", [np.nan, np.inf, -np.inf, -0.001, 1.001])
def test_normalize_echo_clipping_config_rejects_non_finite_or_out_of_range_prob(prob):
    with pytest.raises(ValueError, match=r"prob must be in \[0, 1\]"):
        normalize_echo_clipping_config({"prob": prob})


@pytest.mark.parametrize(
    "raw,match",
    [
        ({"mystery": 1}, "contains unknown keys"),
        ({"path_probs": {"unknown_path": 1}}, "unknown path"),
        ({"paths": {"unknown_path": {}}}, "unknown path"),
        (
            {"paths": {"post_ser_distortion": {"mystery": 1}}},
            "contains unknown keys",
        ),
        (
            {
                "paths": {
                    "post_ser_distortion": {"variant_probs": {"fixed_rail": 1}}
                }
            },
            "unknown variant",
        ),
        (
            {
                "paths": {
                    "pre_ser_overload": {"variant_probs": {"hard": 1}}
                }
            },
            "unknown variant",
        ),
        (
            {
                "paths": {
                    "post_ser_distortion": {
                        "peak_relative_threshold_range": [0.0, 0.5]
                    }
                }
            },
            "lower bound must be > 0",
        ),
        (
            {
                "paths": {
                    "pre_ser_overload": {"overload_boost_db_range": [-1, 2]}
                }
            },
            "must be non-negative",
        ),
        (
            {"paths": {"synced_ref_clip": {"rail_dbfs_range": [-3, 1]}}},
            "must be <= 0 dBFS",
        ),
        (
            {"paths": {"synced_ref_clip": {"rail_dbfs_range": [-1, -3]}}},
            "lo <= hi",
        ),
    ],
)
def test_normalize_echo_clipping_config_rejects_unknown_or_invalid_values(raw, match):
    with pytest.raises(ValueError, match=match):
        normalize_echo_clipping_config(raw)


@pytest.mark.parametrize(
    "legacy_key,legacy_value",
    [
        ("variant_probs", {"hard": 1.0}),
        ("peak_relative_threshold_range", [0.4, 0.8]),
        ("overload_boost_db_range", [3.0, 9.0]),
        ("overload_dbfs_threshold_range", [-8.0, -2.0]),
    ],
)
def test_normalize_echo_clipping_config_rejects_removed_flat_schema(
    legacy_key, legacy_value
):
    with pytest.raises(ValueError, match="contains unknown keys"):
        normalize_echo_clipping_config({legacy_key: legacy_value})


def test_normalize_echo_clipping_config_rejects_finite_weights_with_overflowing_sum():
    with pytest.raises(ValueError, match="sum to a finite value > 0"):
        normalize_echo_clipping_config(
            {
                "path_probs": {
                    "post_ser_distortion": np.finfo(np.float64).max,
                    "pre_ser_overload": np.finfo(np.float64).max,
                }
            }
        )


@pytest.mark.parametrize("path", ["pre_ser_overload", "synced_ref_clip"])
@pytest.mark.parametrize("rail_dbfs", [-1000.0, -10000.0])
def test_normalize_echo_clipping_config_rejects_rail_linear_underflow(
    path, rail_dbfs
):
    with pytest.raises(ValueError, match="linear rail"):
        normalize_echo_clipping_config(
            {"paths": {path: {"rail_dbfs_range": [rail_dbfs, rail_dbfs]}}}
        )


@pytest.mark.parametrize("boost_db", [800.0, 10000.0])
def test_normalize_echo_clipping_config_rejects_unsafe_float32_boost(boost_db):
    with pytest.raises(ValueError, match="float32"):
        normalize_echo_clipping_config(
            {
                "paths": {
                    "pre_ser_overload": {
                        "overload_boost_db_range": [boost_db, boost_db]
                    }
                }
            }
        )


def test_apply_absolute_rail_clip_reports_true_rail_hit_facts():
    signal = np.array([-1.0, -0.25, 0.25, 1.0], dtype=np.float32)
    rail_dbfs = 20.0 * np.log10(0.5)
    facts = {}

    out, variant, sampled_rail_dbfs = apply_absolute_rail_clip(
        signal,
        (rail_dbfs, rail_dbfs),
        {"fixed_rail": 1.0},
        np.random.default_rng(9),
        facts_out=facts,
    )

    np.testing.assert_allclose(out, [-0.5, -0.25, 0.25, 0.5], atol=1e-7)
    assert variant == "fixed_rail"
    assert sampled_rail_dbfs == pytest.approx(rail_dbfs)
    assert facts == {
        "sample_count": 4,
        "clipped_sample_count": 2,
        "clipped_sample_ratio": 0.5,
        "rail_hit": True,
    }


def test_apply_absolute_rail_clip_distinguishes_tanh_change_from_rail_hit():
    signal = np.array([-0.4, -0.25, 0.25, 0.4], dtype=np.float32)
    rail_dbfs = 20.0 * np.log10(0.5)
    facts = {}

    out, variant, sampled_rail_dbfs = apply_absolute_rail_clip(
        signal,
        (rail_dbfs, rail_dbfs),
        {"tanh": 1.0},
        np.random.default_rng(10),
        facts_out=facts,
    )

    expected = 0.5 * np.tanh(signal / 0.5)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)
    assert variant == "tanh"
    assert sampled_rail_dbfs == pytest.approx(rail_dbfs)
    assert facts == {
        "sample_count": 4,
        "clipped_sample_count": 0,
        "clipped_sample_ratio": 0.0,
        "rail_hit": False,
    }
    assert not np.array_equal(out, signal)


def test_summarize_absolute_rail_facts_keeps_synced_routes_and_aggregate():
    summary = summarize_absolute_rail_facts(
        {
            "sample_count": 4,
            "clipped_sample_count": 2,
            "clipped_sample_ratio": 0.5,
            "rail_hit": True,
        },
        {
            "sample_count": 2,
            "clipped_sample_count": 0,
            "clipped_sample_ratio": 0.0,
            "rail_hit": False,
        },
    )

    assert summary == {
        "echo_rail_hit": True,
        "farend_rail_hit": False,
        "rail_hit": True,
        "clipped_sample_count": 2,
        "rail_sample_count": 6,
        "clipped_sample_ratio": pytest.approx(1.0 / 3.0),
    }


def test_apply_absolute_rail_clip_rejects_linear_rail_underflow():
    with pytest.raises(ValueError, match="linear rail"):
        apply_absolute_rail_clip(
            np.ones(4, dtype=np.float32),
            (-10000.0, -10000.0),
            {"fixed_rail": 1.0},
            np.random.default_rng(13),
        )


def test_apply_absolute_rail_clip_rejects_non_finite_output(monkeypatch):
    monkeypatch.setattr(
        "tasks.aec.data.echo_clipping.sox_clip",
        lambda signal, threshold_db: np.full_like(signal, np.nan),
    )

    with pytest.raises(ValueError, match="non-finite output"):
        apply_absolute_rail_clip(
            np.ones(4, dtype=np.float32),
            (-6.0, -6.0),
            {"tanh": 1.0},
            np.random.default_rng(14),
        )


def test_select_echo_clip_path_never_triggers_at_zero_prob():
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert select_echo_clip_path(rng, 0.0, {"a": 1.0}) is None


def test_select_echo_clip_path_is_deterministic_for_seeded_rng():
    path_probs = {
        "post_ser_distortion": 0.5,
        "pre_ser_overload": 0.3,
        "synced_ref_clip": 0.2,
    }
    first_rng = np.random.default_rng(20260828)
    second_rng = np.random.default_rng(20260828)

    first = [select_echo_clip_path(first_rng, 1.0, path_probs) for _ in range(100)]
    second = [select_echo_clip_path(second_rng, 1.0, path_probs) for _ in range(100)]

    assert first == second
    assert set(first) == set(path_probs)


@pytest.mark.parametrize(
    "path_probs,heavy_path",
    [
        ({"light": 1.0, "heavy": 3.0}, "heavy"),
        ({"light": 3.0, "heavy": 1.0}, "light"),
    ],
)
def test_select_echo_clip_path_applies_path_prob_weights_statistically(
    path_probs, heavy_path
):
    rng = np.random.default_rng(20260828)
    draws = [select_echo_clip_path(rng, 1.0, path_probs) for _ in range(4000)]
    heavy_fraction = draws.count(heavy_path) / len(draws)

    # 3:1 权重应稳定落在理论 75% 附近，并随权重翻转而翻转。
    assert 0.72 <= heavy_fraction <= 0.78


@pytest.mark.parametrize(
    "weights",
    [
        {"hard": -1.0, "tanh": 2.0},
        {"hard": np.nan},
        {"hard": np.inf},
        {"hard": 0.0},
    ],
)
def test_sample_peak_relative_clip_params_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="variant_probs weights"):
        sample_peak_relative_clip_params(
            (0.5, 0.95), weights, np.random.default_rng(0)
        )


@pytest.mark.parametrize(
    "threshold_range",
    [(np.nan, 0.95), (0.5, np.inf), (0.95, 0.5)],
)
def test_sample_peak_relative_clip_params_rejects_invalid_range(threshold_range):
    with pytest.raises(ValueError, match="threshold_frac_range"):
        sample_peak_relative_clip_params(
            threshold_range, {"hard": 1.0}, np.random.default_rng(0)
        )


def test_apply_peak_relative_clip_silence_short_circuits():
    rng = np.random.default_rng(0)
    signal = np.zeros(1600, dtype=np.float32)
    out, variant = apply_peak_relative_clip(signal, (0.5, 0.95), {"hard": 1.0}, rng)
    np.testing.assert_array_equal(out, signal)
    assert variant == "none"


@pytest.mark.parametrize(
    "variant,expected",
    [
        ("hard", [-1.0, -0.5, 0.0, 0.5, 1.0]),
        (
            "tanh",
            [-0.9640276, -0.46211717, 0.0, 0.46211717, 0.9640276],
        ),
    ],
)
def test_apply_peak_relative_clip_post_variants_follow_per_sample_numeric_contract(
    variant, expected
):
    signal = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)

    out, sampled_variant = apply_peak_relative_clip(
        signal,
        (0.5, 0.5),
        {variant: 1.0},
        np.random.default_rng(1),
    )

    assert sampled_variant == variant
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_apply_peak_relative_clip_rejects_non_finite_input(value):
    signal = np.array([0.0, value, 1.0], dtype=np.float32)

    with pytest.raises(ValueError, match="signal contains non-finite values"):
        apply_peak_relative_clip(
            signal,
            (0.5, 0.5),
            {"hard": 1.0},
            np.random.default_rng(2),
        )


def test_apply_peak_relative_clip_rejects_extreme_tanh_threshold():
    threshold_frac = np.nextafter(0.0, 1.0)

    with pytest.raises(ValueError, match="threshold_db"):
        apply_peak_relative_clip(
            np.ones(4, dtype=np.float32),
            (threshold_frac, threshold_frac),
            {"tanh": 1.0},
            np.random.default_rng(2),
        )


def test_apply_peak_relative_clip_rejects_non_finite_output(monkeypatch):
    monkeypatch.setattr(
        "tasks.aec.data.echo_clipping.sox_clip",
        lambda signal, threshold_db: np.full_like(signal, np.inf),
    )

    with pytest.raises(ValueError, match="non-finite output"):
        apply_peak_relative_clip(
            np.ones(4, dtype=np.float32),
            (0.5, 0.5),
            {"tanh": 1.0},
            np.random.default_rng(2),
        )


def test_apply_peak_relative_clip_reuses_sampled_params_across_different_peaks():
    rng = np.random.default_rng(3)
    threshold_frac, variant = sample_peak_relative_clip_params(
        (0.4, 0.8), {"hard": 1.0}, rng
    )
    low_peak = np.array([-1.0, -0.2, 0.5, 1.0], dtype=np.float32)
    high_peak = low_peak * 3.0

    low_out, low_variant = apply_peak_relative_clip(
        low_peak,
        (0.4, 0.8),
        {"hard": 1.0},
        rng,
        threshold_frac=threshold_frac,
        variant=variant,
    )
    high_out, high_variant = apply_peak_relative_clip(
        high_peak,
        (0.4, 0.8),
        {"hard": 1.0},
        rng,
        threshold_frac=threshold_frac,
        variant=variant,
    )

    assert low_variant == high_variant == variant == "hard"
    np.testing.assert_allclose(high_out, low_out * 3.0)
    assert float(np.max(np.abs(low_out))) == pytest.approx(threshold_frac)
    assert float(np.max(np.abs(high_out))) == pytest.approx(3.0 * threshold_frac)


def test_apply_peak_relative_clip_legacy_call_keeps_rng_order_and_result():
    signal = np.linspace(-1.0, 1.0, 101, dtype=np.float32)
    legacy_rng = np.random.default_rng(4)
    split_rng = np.random.default_rng(4)

    legacy_out, legacy_variant = apply_peak_relative_clip(
        signal, (0.4, 0.8), {"hard": 0.5, "tanh": 0.5}, legacy_rng
    )
    threshold_frac, variant = sample_peak_relative_clip_params(
        (0.4, 0.8), {"hard": 0.5, "tanh": 0.5}, split_rng
    )
    split_out, split_variant = apply_peak_relative_clip(
        signal,
        (0.4, 0.8),
        {"hard": 0.5, "tanh": 0.5},
        split_rng,
        threshold_frac=threshold_frac,
        variant=variant,
    )

    np.testing.assert_array_equal(legacy_out, split_out)
    assert legacy_variant == split_variant == variant
    assert legacy_rng.random() == split_rng.random()


def test_apply_peak_relative_clip_explicit_params_keep_silence_and_rng_unchanged():
    signal = np.zeros(100, dtype=np.float32)
    rng = np.random.default_rng(5)
    expected_rng = np.random.default_rng(5)

    out, variant = apply_peak_relative_clip(
        signal,
        (0.4, 0.8),
        {"hard": 1.0},
        rng,
        threshold_frac=0.6,
        variant="hard",
    )

    np.testing.assert_array_equal(out, signal)
    assert variant == "none"
    assert rng.random() == expected_rng.random()


@pytest.mark.parametrize("threshold_frac", [0.39, 0.81, np.nan])
def test_apply_peak_relative_clip_rejects_invalid_explicit_threshold_frac(threshold_frac):
    with pytest.raises(ValueError, match="threshold_frac"):
        apply_peak_relative_clip(
            np.ones(100, dtype=np.float32),
            (0.4, 0.8),
            {"hard": 1.0},
            np.random.default_rng(6),
            threshold_frac=threshold_frac,
            variant="hard",
        )


def test_apply_peak_relative_clip_rejects_unknown_sampled_variant_even_for_silence():
    with pytest.raises(ValueError, match="unknown variant"):
        apply_peak_relative_clip(
            np.zeros(100, dtype=np.float32),
            (0.5, 0.5),
            {"hard": 1.0},
            np.random.default_rng(7),
            threshold_frac=0.5,
            variant="soft",
        )


def test_apply_peak_relative_clip_rejects_unknown_variant():
    rng = np.random.default_rng(8)
    signal = np.ones(100, dtype=np.float32)
    with pytest.raises(ValueError):
        apply_peak_relative_clip(signal, (0.5, 0.5), {"soft": 1.0}, rng)


def test_apply_playback_overload_silence_short_circuits():
    rng = np.random.default_rng(0)
    echo = np.zeros(1600, dtype=np.float32)
    out, boost_db, variant = apply_playback_overload(echo, (3.0, 18.0), (-6.0, -1.0), {"hard": 1.0}, rng)
    np.testing.assert_array_equal(out, echo)
    assert boost_db is None
    assert variant == "none"


@pytest.mark.parametrize(
    "variant_probs,expected_variant",
    [({"hard": 1.0}, "hard"), ({"fixed_rail": 1.0}, "fixed_rail")],
)
def test_apply_playback_overload_preserves_legacy_or_canonical_variant_label(
    variant_probs, expected_variant
):
    _, _, variant = apply_playback_overload(
        np.ones(4, dtype=np.float32),
        (0.0, 0.0),
        (-6.0, -6.0),
        variant_probs,
        np.random.default_rng(15),
    )

    assert variant == expected_variant


def test_apply_playback_overload_rejects_both_hard_aliases():
    with pytest.raises(ValueError, match="hard.*fixed_rail"):
        apply_playback_overload(
            np.ones(4, dtype=np.float32),
            (0.0, 0.0),
            (-6.0, -6.0),
            {"hard": 1.0, "fixed_rail": 1.0},
            np.random.default_rng(16),
        )


def test_apply_playback_overload_rejects_unsafe_float32_boost():
    with pytest.raises(ValueError, match="float32"):
        apply_playback_overload(
            np.ones(4, dtype=np.float32),
            (800.0, 800.0),
            (-6.0, -6.0),
            {"fixed_rail": 1.0},
            np.random.default_rng(17),
        )


def test_apply_playback_overload_rejects_non_finite_output(monkeypatch):
    monkeypatch.setattr(
        "tasks.aec.data.echo_clipping.apply_absolute_rail_clip",
        lambda *args, **kwargs: (
            np.full(4, np.nan, dtype=np.float32),
            "fixed_rail",
            -6.0,
        ),
    )

    with pytest.raises(ValueError, match="non-finite output"):
        apply_playback_overload(
            np.ones(4, dtype=np.float32),
            (0.0, 0.0),
            (-6.0, -6.0),
            {"fixed_rail": 1.0},
            np.random.default_rng(18),
        )


def test_apply_playback_overload_distinguishes_boost_only_from_rail_hit():
    rng = np.random.default_rng(3)
    t = np.linspace(0, 1, 1600, dtype=np.float32)
    echo = (0.05 * np.sin(2 * np.pi * 5 * t)).astype(np.float32)
    facts = {}
    out, boost_db, variant = apply_playback_overload(
        echo,
        (3.0, 18.0),
        (-6.0, -1.0),
        {"fixed_rail": 1.0},
        rng,
        facts_out=facts,
    )
    assert boost_db is not None
    assert 3.0 <= boost_db <= 18.0
    assert variant == "fixed_rail"
    # ``applied`` 表示相对原 waveform 有变化，``rail_hit`` 单独表示越 rail。
    assert facts == {
        "applied": True,
        "boost_db": boost_db,
        "rail_dbfs": pytest.approx(facts["rail_dbfs"]),
        "rail": pytest.approx(10 ** (facts["rail_dbfs"] / 20.0)),
        "sample_count": echo.size,
        "clipped_sample_count": 0,
        "clipped_sample_ratio": 0.0,
        "rail_hit": False,
    }
    assert -6.0 <= facts["rail_dbfs"] <= -1.0
    assert float(np.max(np.abs(out))) > float(np.max(np.abs(echo)))


def test_apply_playback_overload_reports_true_rail_hit():
    echo = np.array([-0.4, 0.4], dtype=np.float32)
    facts = {}

    out, boost_db, variant = apply_playback_overload(
        echo,
        (6.0, 6.0),
        (-6.0, -6.0),
        {"fixed_rail": 1.0},
        np.random.default_rng(19),
        facts_out=facts,
    )

    assert boost_db == 6.0
    assert variant == "fixed_rail"
    assert facts["applied"] is True
    assert facts["rail_hit"] is True
    assert facts["clipped_sample_count"] == 2
    assert facts["clipped_sample_ratio"] == 1.0
    np.testing.assert_allclose(np.abs(out), facts["rail"], rtol=1e-6)


def test_apply_playback_overload_reports_noop_from_unchanged_waveform():
    echo = np.array([-0.25, 0.25], dtype=np.float32)
    facts = {}

    out, boost_db, variant = apply_playback_overload(
        echo,
        (0.0, 0.0),
        (0.0, 0.0),
        {"fixed_rail": 1.0},
        np.random.default_rng(11),
        facts_out=facts,
    )

    np.testing.assert_array_equal(out, echo)
    assert boost_db == 0.0
    assert variant == "fixed_rail"
    assert facts == {
        "applied": False,
        "boost_db": 0.0,
        "rail_dbfs": 0.0,
        "rail": 1.0,
        "sample_count": 2,
        "clipped_sample_count": 0,
        "clipped_sample_ratio": 0.0,
        "rail_hit": False,
    }


def test_apply_playback_overload_silence_reports_noop_facts():
    echo = np.zeros(8, dtype=np.float32)
    facts = {"applied": True}

    out, boost_db, variant = apply_playback_overload(
        echo,
        (3.0, 18.0),
        (-6.0, -1.0),
        {"fixed_rail": 1.0},
        np.random.default_rng(12),
        facts_out=facts,
    )

    np.testing.assert_array_equal(out, echo)
    assert (boost_db, variant) == (None, "none")
    assert facts == {
        "applied": False,
        "boost_db": None,
        "rail_dbfs": None,
        "rail": None,
        "sample_count": 8,
        "clipped_sample_count": 0,
        "clipped_sample_ratio": 0.0,
        "rail_hit": False,
    }


def test_ser_db_from_power_ratio_equal_power_is_zero_db():
    rng = np.random.default_rng(4)
    clean = rng.normal(size=1600).astype(np.float32)
    assert ser_db_from_power_ratio(clean, clean) == pytest.approx(0.0, abs=1e-4)


def test_ser_db_from_power_ratio_louder_echo_is_negative():
    rng = np.random.default_rng(5)
    clean = rng.normal(size=1600).astype(np.float32)
    echo = clean * 10.0  # echo RMS is 10x clean RMS -> -20 dB SER
    assert ser_db_from_power_ratio(clean, echo) == pytest.approx(-20.0, abs=1e-3)

pytestmark = [pytest.mark.unit]
