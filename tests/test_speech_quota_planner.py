"""Contract tests for the bounded speech quota planner."""

from __future__ import annotations

from dataclasses import dataclass, fields
from fractions import Fraction
from itertools import product
import json
import math
from types import MappingProxyType

import pytest

import core.data.sampling as sampling
from core.data.sampling import (
    SpeechLeafMetadata,
    SpeechPlan,
    SpeechSamplingConfig,
    SpeakerSamplingConfig,
    _dataset_cap,
    _dataset_sr_options,
    make_speech_plan,
)
from core.data.speech_quota_solver import SolverBudget, SolverBudgetExceeded


@dataclass(frozen=True)
class Cell:
    language: str
    dataset_id: str
    batching_sr: int
    uids: tuple[str, ...]
    effective_seconds: float


def cell(language, dataset, sr, count, seconds):
    return Cell(language, dataset, sr, tuple(f"{language}-{dataset}-{sr}-{i}" for i in range(count)), seconds)


def test_make_speech_plan_without_overrides_preserves_legacy_defaults():
    normalized = sampling._normalize_speech_inputs(None)
    assert normalized.language == sampling.LanguageSamplingConfig((
        ("en", Fraction(0)),
        ("other", Fraction(0)),
        ("zh", Fraction(1)),
    ))
    assert normalized.dataset == sampling.DatasetSamplingConfig(
        alpha=Fraction(1, 2), max_ratio=Fraction(3, 10)
    )
    assert type(normalized.dataset.alpha) is Fraction
    assert type(normalized.dataset.max_ratio) is Fraction

    plan = make_speech_plan(
        [
            cell("zh", "a", 16000, 10, 4.0),
            cell("zh", "b", 16000, 10, 9.0),
            cell("zh", "c", 16000, 10, 16.0),
            cell("zh", "d", 16000, 10, 25.0),
        ],
        requested=10,
        world_size=1,
        batch_size=1,
    )

    assert plan.actual == 10
    assert plan.language_quotas == {"zh": 10}
    assert plan.global_dataset_quotas == {"a": 2, "b": 2, "c": 3, "d": 3}
    assert plan.summary["controllers"] == {
        "language_enabled": True,
        "dataset_enabled": True,
    }


def test_direct_overrides_with_speech_config_are_rejected():
    dataset_config = sampling.SpeechSamplingConfig(
        dataset=sampling.DatasetSamplingConfig(Fraction(4, 5), Fraction(2, 5))
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        sampling._normalize_speech_inputs(dataset_config, dataset_alpha=0.2)
    with pytest.raises(ValueError, match="cannot be combined"):
        sampling._normalize_speech_inputs(dataset_config, max_dataset_ratio=0.6)
    with pytest.raises(ValueError, match="cannot be combined"):
        sampling._normalize_speech_inputs(
            sampling.SpeechSamplingConfig(
                language=sampling.LanguageSamplingConfig(
                    probs=(
                        ("en", Fraction(1, 4)),
                        ("other", Fraction(0)),
                        ("zh", Fraction(3, 4)),
                    )
                ),
                dataset=sampling.DatasetSamplingConfig(
                    Fraction(4, 5), Fraction(2, 5)
                ),
            ),
            language_probs={"en": 3, "zh": 1},
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        make_speech_plan(
            [cell("zh", "a", 16000, 4, 1.0)],
            requested=2,
            world_size=1,
            batch_size=1,
            sampling_config=dataset_config,
            language_probs={"zh": 1},
        )


@pytest.mark.parametrize(
    ("dataset_alpha", "max_dataset_ratio", "expected"),
    [
        (0.2, None, sampling.DatasetSamplingConfig(Fraction(1, 5), Fraction(3, 10))),
        (None, 0.6, sampling.DatasetSamplingConfig(Fraction(1, 2), Fraction(3, 5))),
        (0.2, 0.6, sampling.DatasetSamplingConfig(Fraction(1, 5), Fraction(3, 5))),
    ],
)
def test_direct_dataset_controls_alone_fill_missing_default(
    dataset_alpha, max_dataset_ratio, expected
):
    normalized = sampling._normalize_speech_inputs(
        None,
        dataset_alpha=dataset_alpha,
        max_dataset_ratio=max_dataset_ratio,
    )

    assert normalized == sampling.SpeechSamplingConfig(dataset=expected)


def test_public_planner_preserves_arbitrarily_small_exact_dataset_ratio():
    tiny_ratio = Fraction(1, 10**1000)

    plan = make_speech_plan(
        [cell("zh", "a", 16000, 1, 1.0)],
        requested=1,
        world_size=1,
        batch_size=1,
        language_probs={"zh": 1},
        max_dataset_ratio=tiny_ratio,
    )

    assert plan.actual == 1
    assert plan.global_dataset_quotas == {"a": 1}
    assert plan.summary["datasets"]["a"]["cap"] == 1


@pytest.mark.parametrize(
    ("probs", "match"),
    [
        (
            (("en", Fraction(-1)), ("other", Fraction(2)), ("zh", Fraction(0))),
            "non-negative",
        ),
        (
            (("en", Fraction(0)), ("other", Fraction(0)), ("zh", Fraction(0))),
            "positive",
        ),
        ((("custom", Fraction(1)),), "unknown"),
        ((("zh", float("inf")),), "finite"),
    ],
)
def test_direct_dataclass_language_probs_are_strictly_validated(probs, match):
    with pytest.raises(ValueError, match=match):
        sampling._normalize_speech_inputs(
            sampling.SpeechSamplingConfig(
                language=sampling.LanguageSamplingConfig(probs=probs)
            )
        )


def test_direct_dataclass_language_probs_are_normalized_for_planner_quota():
    plan = make_speech_plan(
        [cell("zh", "a", 16000, 3, 1.0), cell("en", "b", 16000, 3, 1.0)],
        requested=4,
        world_size=1,
        batch_size=1,
        sampling_config=sampling.SpeechSamplingConfig(
            language=sampling.LanguageSamplingConfig(
                probs=(
                    ("en", Fraction(1, 4)),
                    ("other", Fraction(0)),
                    ("zh", Fraction(3, 4)),
                )
            )
        ),
    )

    assert plan.language_quotas == {"en": 1, "zh": 3}


def test_exact_fraction_language_tie_uses_stable_key_order():
    plan = make_speech_plan(
        [cell("en", "a", 16000, 3, 1.0), cell("zh", "b", 16000, 3, 1.0)],
        requested=3,
        world_size=1,
        batch_size=1,
        sampling_config=sampling.SpeechSamplingConfig(
            language=sampling.LanguageSamplingConfig(
                probs=(
                    ("en", Fraction(1, 6)),
                    ("other", Fraction(0)),
                    ("zh", Fraction(5, 6)),
                )
            )
        ),
    )

    assert plan.language_quotas == {"en": 1, "zh": 2}


def test_no_speaker_fast_path_preserves_legacy_float_hamilton_tie():
    plan = make_speech_plan(
        [
            cell("en", "en", 16000, 1, 1.0),
            cell("other", "other", 16000, 2, 1.0),
            cell("zh", "zh", 16000, 1, 1.0),
        ],
        requested=2,
        world_size=1,
        batch_size=1,
        language_probs={"en": 0.1, "other": 0.7, "zh": 0.2},
    )

    assert plan.language_quotas == {"en": 0, "other": 1, "zh": 1}


def test_bounded_hamilton_preserves_legacy_float_remainder_order():
    assert sampling._bounded_hamilton(
        {"a": 1.0, "b": 0.2}, {"a": 8, "b": 10}, 3
    ) == {"a": 3, "b": 0}


def test_direct_language_probs_stably_normalize_overflowing_weights():
    plan = make_speech_plan(
        [cell("zh", "a", 16000, 2, 1.0), cell("en", "b", 16000, 2, 1.0)],
        requested=4,
        world_size=1,
        batch_size=1,
        language_probs={"zh": 1e308, "en": 1e308},
    )

    assert plan.language_quotas == {"en": 2, "zh": 2}


def test_language_only_does_not_apply_dataset_caps():
    plan = make_speech_plan(
        [cell("zh", "a", 16000, 10, 100.0), cell("en", "b", 16000, 10, 1.0)],
        requested=10,
        world_size=1,
        batch_size=1,
        language_probs={"zh": 0.6, "en": 0.4},
    )

    assert plan.language_quotas == {"en": 4, "zh": 6}
    assert plan.global_dataset_quotas == {}
    assert plan.cell_quotas[("zh", "a", 16000)] == 6
    assert plan.cell_quotas[("en", "b", 16000)] == 4


def test_dataset_weights_stably_aggregate_overflowing_effective_seconds():
    plan = make_speech_plan(
        [
            cell("zh", "a", 16000, 6, 1e308),
            cell("zh", "b", 16000, 3, 1e308),
            cell("zh", "b", 48000, 3, 1e308),
        ],
        requested=6,
        world_size=1,
        batch_size=1,
        dataset_alpha=1,
        max_dataset_ratio=1,
    )

    assert plan.global_dataset_quotas == {"a": 2, "b": 4}
    assert math.isfinite(plan.summary["datasets"]["b"]["effective_seconds"])
    assert plan.summary["datasets"]["b"]["scaled_effective_seconds"] == 2.0
    assert plan.summary["datasets"]["b"]["effective_seconds_scale"] == 1e308


def test_dataset_weight_dynamic_range_does_not_leak_key_error():
    leaves = [
        make_leaf("en", "tiny", 16000, [0], Fraction.from_float(5e-324)),
        make_leaf("en", "large", 16000, [1], Fraction.from_float(1e308)),
        make_leaf("zh", "large", 16000, [2], Fraction.from_float(1e308)),
    ]

    try:
        plan = make_speech_plan(
            leaves,
            requested=2,
            world_size=1,
            batch_size=1,
            language_probs={"en": 0.5, "zh": 0.5},
            dataset_alpha=1,
            max_dataset_ratio=1,
        )
    except ValueError as exc:
        assert "dataset" in str(exc) or "infeasible" in str(exc)
    else:
        assert plan.actual > 0


def test_dataset_only_uses_one_global_pool_across_languages():
    plan = make_speech_plan(
        [
            cell("zh", "shared", 16000, 10, 9.0),
            cell("en", "shared", 16000, 10, 7.0),
            cell("zh", "other", 16000, 10, 4.0),
        ],
        requested=10,
        world_size=1,
        batch_size=1,
        dataset_alpha=0.5,
        max_dataset_ratio=0.6,
    )

    assert plan.language_quotas == {}
    assert plan.global_dataset_quotas == {"other": 4, "shared": 6}
    assert sum(
        quota
        for (_language, dataset, _sr), quota in plan.cell_quotas.items()
        if dataset == "shared"
    ) == 6
    assert plan.summary["datasets"]["shared"]["cap"] == 6


def test_joint_controls_satisfy_language_and_global_dataset_margins():
    cells = [
        cell("zh", "a", 16000, 8, 8.0),
        cell("zh", "b", 16000, 2, 2.0),
        cell("en", "a", 16000, 2, 2.0),
        cell("en", "b", 16000, 8, 8.0),
    ]
    plan = make_speech_plan(
        cells,
        requested=10,
        world_size=1,
        batch_size=1,
        language_probs={"zh": 0.6, "en": 0.4},
        dataset_alpha=0,
        max_dataset_ratio=0.5,
    )

    assert plan.language_quotas == {"en": 4, "zh": 6}
    assert plan.global_dataset_quotas == {"a": 5, "b": 5}
    assert sum(q for (lang, _ds, _sr), q in plan.cell_quotas.items() if lang == "zh") == 6
    assert sum(q for (lang, _ds, _sr), q in plan.cell_quotas.items() if lang == "en") == 4
    assert sum(q for (_lang, ds, _sr), q in plan.cell_quotas.items() if ds == "a") == 5
    assert sum(q for (_lang, ds, _sr), q in plan.cell_quotas.items() if ds == "b") == 5


def test_joint_plan_is_deterministic_under_reversed_cell_order():
    cells = [
        cell("zh", "a", 16000, 8, 8.0),
        cell("zh", "b", 48000, 2, 2.0),
        cell("en", "a", 48000, 2, 2.0),
        cell("en", "b", 16000, 8, 8.0),
    ]
    kwargs = dict(
        requested=8,
        world_size=2,
        batch_size=1,
        language_probs={"zh": 0.5, "en": 0.5},
        dataset_alpha=0,
        max_dataset_ratio=0.5,
    )

    assert make_speech_plan(cells, **kwargs).cell_quotas == make_speech_plan(
        list(reversed(cells)), **kwargs
    ).cell_quotas


def test_summary_reports_independent_controller_states():
    cells = [cell("zh", "a", 16000, 4, 1.0), cell("en", "b", 16000, 4, 1.0)]

    language = make_speech_plan(
        cells, requested=4, world_size=1, batch_size=1, language_probs={"zh": 1}
    )
    dataset = make_speech_plan(
        cells, requested=4, world_size=1, batch_size=1,
        dataset_alpha=0, max_dataset_ratio=1,
    )
    both = make_speech_plan(
        cells, requested=4, world_size=1, batch_size=1,
        language_probs={"zh": 0.5, "en": 0.5},
        dataset_alpha=0, max_dataset_ratio=0.5,
    )

    assert language.summary["controllers"] == {
        "language_enabled": True, "dataset_enabled": False
    }
    assert dataset.summary["controllers"] == {
        "language_enabled": False, "dataset_enabled": True
    }
    assert both.summary["controllers"] == {
        "language_enabled": True, "dataset_enabled": True
    }


def test_dataset_only_capacity_includes_all_languages():
    plan = make_speech_plan(
        [cell("zh", "shared", 16000, 2, 2.0), cell("other", "shared", 16000, 3, 3.0)],
        requested=5,
        world_size=1,
        batch_size=1,
        dataset_alpha=0,
        max_dataset_ratio=1,
    )

    assert plan.actual == 5
    assert plan.global_dataset_quotas == {"shared": 5}
    assert plan.summary["datasets"]["shared"]["capacity"] == 5
    assert sum(plan.cell_quotas.values()) == 5


def test_joint_transport_projects_to_largest_ddp_aligned_feasible_actual():
    plan = make_speech_plan(
        [cell("zh", "a", 16000, 2, 2.0), cell("en", "b", 16000, 8, 8.0)],
        requested=8,
        world_size=2,
        batch_size=1,
        language_probs={"zh": 0.5, "en": 0.5},
        dataset_alpha=0,
        max_dataset_ratio=0.5,
    )

    assert plan.actual == 4
    assert plan.language_quotas == {"en": 2, "zh": 2}
    assert plan.global_dataset_quotas == {"a": 2, "b": 2}


def test_dataset_weights_and_caps_use_alpha_and_global_actual_denominator():
    plan = make_speech_plan(
        [
            cell("zh", "a", 16000, 10, 4.0),
            cell("zh", "b", 16000, 10, 9.0),
            cell("zh", "c", 16000, 10, 16.0),
            cell("zh", "d", 16000, 10, 25.0),
        ],
        requested=10,
        world_size=1,
        batch_size=1,
        language_probs={"zh": 1},
        dataset_alpha=0.5,
        max_dataset_ratio=Fraction(3, 10),
    )

    datasets = plan.summary["datasets"]
    assert [datasets[name]["weight"] for name in ("a", "b", "c", "d")] == [2.0, 3.0, 4.0, 5.0]
    assert all(datasets[name]["cap"] == 3 for name in datasets)
    assert plan.language_quotas == {"zh": 10}
    assert plan.global_dataset_quotas == {"a": 2, "b": 2, "c": 3, "d": 3}


def test_small_libraries_never_receive_more_than_unique_uid_capacity():
    plan = make_speech_plan(
        [cell("zh", name, 16000, 1, 1.0) for name in ("a", "b", "c", "d")],
        requested=10,
        world_size=1,
        batch_size=1,
        language_probs={"zh": 1},
    )

    assert plan.actual == 4
    assert sum(plan.cell_quotas.values()) == 4
    assert all(quota <= 1 for quota in plan.cell_quotas.values())


def test_feasible_multiple_languages_match_hamilton_hard_targets():
    plan = make_speech_plan(
        [
            *[cell("zh", name, 16000, 10, 1.0) for name in ("a", "b", "c", "d")],
            *[cell("en", name, 16000, 10, 1.0) for name in ("a", "b", "c", "d")],
        ],
        requested=10,
        world_size=1,
        batch_size=1,
        language_probs={"zh": 0.6, "en": 0.4},
    )

    assert plan.actual == 10
    assert plan.language_quotas == {"en": 4, "zh": 6}


def test_ddp_sr_granularity_projects_actual_down_to_aligned_candidate():
    plan = make_speech_plan(
        [
            *[cell("zh", name, 16000, 10, 1.0) for name in ("a", "b", "c", "d")],
            *[cell("zh", name, 48000, 10, 1.0) for name in ("e", "f", "g", "h")],
        ],
        requested=13,
        world_size=2,
        batch_size=2,
        drop_last=True,
        language_probs={"zh": 1},
    )

    assert plan.actual == 12
    assert all(quota % 4 == 0 for quota in plan.sr_quotas.values())


def test_dataset_caps_shrink_actual_and_recompute_language_targets():
    plan = make_speech_plan(
        [
            *[cell("zh", name, 16000, 10, 1.0) for name in ("a", "b", "c")],
        ],
        requested=10,
        world_size=1,
        batch_size=1,
        language_probs={"zh": 1},
        max_dataset_ratio=0.3,
    )

    assert plan.actual == 9
    assert plan.language_quotas == {"zh": 9}
    assert all(quota <= 3 for quota in plan.global_dataset_quotas.values())


def test_cell_order_does_not_change_quota_assignment():
    cells = [
        cell("zh", "a", 16000, 4, 4.0), cell("zh", "b", 48000, 4, 9.0),
        cell("zh", "c", 16000, 4, 16.0), cell("zh", "d", 48000, 4, 25.0),
    ]
    kwargs = dict(requested=8, world_size=1, batch_size=1, language_probs={"zh": 1})

    assert make_speech_plan(cells, **kwargs).cell_quotas == make_speech_plan(list(reversed(cells)), **kwargs).cell_quotas


@pytest.mark.parametrize(
    ("cells", "probs", "match"),
    [([], {"zh": 1}, "no eligible cells"), ([cell("zh", "a", 16000, 1, 1.0)], {"zh": 1}, "infeasible")],
)
def test_positive_probability_without_legal_capacity_is_rejected(cells, probs, match):
    with pytest.raises(ValueError, match=match):
        make_speech_plan(cells, requested=2, world_size=2, batch_size=1, language_probs=probs)


def test_quotas_conserve_actual_without_duplicate_capacity_overruns():
    cells = [
        *[cell("zh", name, 16000, 3, 1.0) for name in ("a", "b", "c", "d")],
        *[cell("en", name, 48000, 2, 1.0) for name in ("a", "b", "c", "d")],
    ]
    plan = make_speech_plan(cells, requested=12, world_size=1, batch_size=1, language_probs={"zh": 0.5, "en": 0.5})

    capacities = {(item.language, item.dataset_id, item.batching_sr): len(item.uids) for item in cells}
    assert sum(plan.cell_quotas.values()) == plan.actual
    assert sum(plan.language_quotas.values()) == plan.actual
    assert plan.global_dataset_quotas == {}
    assert sum(plan.sr_quotas.values()) == plan.actual
    assert all(quota <= capacities[key] for key, quota in plan.cell_quotas.items())


def test_joint_sr_compensation_preserves_both_margins():
    cells = [
        cell("zh", "a", 16000, 2, 100.0),
        cell("zh", "a", 48000, 2, 1.0),
        cell("en", "b", 16000, 2, 1.0),
        cell("en", "b", 48000, 2, 100.0),
    ]
    plan = make_speech_plan(
        cells,
        requested=4,
        world_size=2,
        batch_size=1,
        language_probs={"zh": 0.5, "en": 0.5},
        dataset_alpha=0,
        max_dataset_ratio=0.5,
    )

    assert plan.language_quotas == {"en": 2, "zh": 2}
    assert plan.global_dataset_quotas == {"a": 2, "b": 2}
    assert all(quota % 2 == 0 for quota in plan.sr_quotas.values())
    assert sum(q for (lang, _ds, _sr), q in plan.cell_quotas.items() if lang == "zh") == 2
    assert sum(q for (_lang, ds, _sr), q in plan.cell_quotas.items() if ds == "a") == 2


def test_sr_compensation_exchanges_same_dataset_cells_without_changing_quotas():
    cells = []
    for dataset in ("a", "b", "c"):
        cells += [cell("zh", dataset, 16000, 1, 100.0), cell("zh", dataset, 48000, 1, 1.0)]
    cells += [cell("zh", "d", 16000, 1, 1.0), cell("zh", "d", 48000, 1, 100.0)]
    kwargs = dict(requested=4, world_size=2, batch_size=1, language_probs={"zh": 1}, max_dataset_ratio=0.3)

    plan = make_speech_plan(cells, **kwargs)
    assert plan.actual == 4
    assert plan.sr_quotas == {16000: 2, 48000: 2}
    assert plan.cell_quotas == make_speech_plan(list(reversed(cells)), **kwargs).cell_quotas


def test_non_monotonic_feasible_sizes_keep_larger_ddp_candidate():
    plan = make_speech_plan(
        [cell("zh", "a", 16000, 2, 1.0), cell("zh", "b", 48000, 2, 1.0)],
        requested=4, world_size=2, batch_size=1, language_probs={"zh": 1}, max_dataset_ratio=0.3,
    )
    assert plan.actual == 4
    assert plan.sr_quotas == {16000: 2, 48000: 2}


@pytest.mark.parametrize("cells", [
    [cell("zh", "a", 16000, 1, 1.0), cell("zh", "a", 16000, 1, 1.0)],
    [Cell("zh", "a", 16000, ("same",), 1.0), Cell("zh", "b", 48000, ("same",), 1.0)],
    [Cell("zh", "a", 16000, ("same", "same"), 1.0)],
])
def test_duplicate_cell_keys_and_uids_are_rejected(cells):
    with pytest.raises(ValueError, match="cell"):
        make_speech_plan(cells, requested=1, world_size=1, batch_size=1, language_probs={"zh": 1})


@pytest.mark.parametrize("bad", [
    Cell("", "a", 16000, ("u",), 1.0), Cell("zh", "", 16000, ("u",), 1.0),
    Cell("zh", "a", True, ("u",), 1.0), Cell("zh", "a", 16000, (), 1.0),
    Cell("zh", "a", 16000, ("",), 1.0), Cell("zh", "a", 16000, ("u",), -1.0),
])
def test_invalid_cells_are_rejected_with_cell_index(bad):
    with pytest.raises(ValueError, match="cell 0"):
        make_speech_plan([bad], requested=1, world_size=1, batch_size=1, language_probs={"zh": 1})


@pytest.mark.parametrize("probs", [{1: 1}, {"": 1}, {"  ": 1}])
def test_language_probability_keys_must_be_nonempty_strings(probs):
    with pytest.raises(ValueError, match="keys"):
        make_speech_plan([cell("zh", "a", 16000, 1, 1.0)], requested=1, world_size=1, batch_size=1, language_probs=probs)


def test_summary_includes_zero_quota_input_sample_rates():
    plan = make_speech_plan([cell("zh", "a", 16000, 2, 100.0), cell("zh", "b", 48000, 1, 1.0)], requested=2, world_size=1, batch_size=1, language_probs={"zh": 1}, max_dataset_ratio=1)
    assert 48000 in plan.summary["sample_rates"]
    assert plan.summary["sample_rates"][48000]["quota"] == 0


def test_global_dataset_caps_project_joint_plan_using_actual_denominator():
    cells = [cell("zh", name, 16000, 20, 1.0) for name in ("a", "b")]
    cells += [cell("en", f"d{index}", 16000, 20, 1.0) for index in range(10)]

    plan = make_speech_plan(
        cells, requested=10, world_size=2, batch_size=1,
        language_probs={"zh": 0.9, "en": 0.1}, max_dataset_ratio=0.3,
    )
    assert plan.actual == 2
    assert plan.language_quotas == {"en": 0, "zh": 2}
    assert plan.global_dataset_quotas["a"] == 1
    assert plan.global_dataset_quotas["b"] == 1
    assert all(details["cap"] == 1 for details in plan.summary["datasets"].values())


def test_sr_allocation_finds_cross_dataset_solution_without_local_greedy_exchange():
    plan = make_speech_plan(
        [
            cell("zh", "a", 16000, 1, 1.0), cell("zh", "a", 24000, 1, 100.0),
            cell("zh", "b", 16000, 1, 1.0), cell("zh", "b", 48000, 1, 100.0),
        ],
        requested=2, world_size=2, batch_size=1, language_probs={"zh": 1}, max_dataset_ratio=0.3,
    )
    assert plan.actual == 2
    assert plan.sr_quotas[16000] == 2
    assert all(quota % 2 == 0 for quota in plan.sr_quotas.values())


def test_effective_requested_ignores_zero_probability_language_capacity():
    plan = make_speech_plan(
        [cell("zh", "a", 16000, 2, 1.0), cell("en", "large", 16000, 100, 1.0)],
        requested=100, world_size=1, batch_size=1,
        language_probs={"zh": 1, "en": 0}, max_dataset_ratio=1,
    )
    assert plan.effective_requested == 2
    assert plan.actual == 2
    assert plan.language_quotas == {"zh": 2}


def test_zero_effective_seconds_are_rejected_without_division_error():
    with pytest.raises(ValueError, match="cell 0.*effective_seconds"):
        make_speech_plan(
            [cell("zh", "a", 16000, 1, 0.0)], requested=1, world_size=1, batch_size=1,
            language_probs={"zh": 1},
        )


def test_direct_language_probs_reject_custom_language_keys():
    with pytest.raises(ValueError, match="unknown"):
        make_speech_plan(
            [cell("zh", "a", 16000, 1, 1.0)], requested=1, world_size=1, batch_size=1,
            language_probs={"custom": 1},
        )


def test_complete_sr_solver_finds_feasible_plan_beyond_local_exchange():
    plan = make_speech_plan(
        [
            cell("zh", "d0", 24000, 4, 16.0), cell("zh", "d0", 8000, 2, 1.0),
            cell("zh", "d1", 48000, 2, 1.0), cell("zh", "d1", 8000, 2, 100.0),
            cell("zh", "d2", 16000, 4, 2.0), cell("zh", "d2", 8000, 5, 16.0),
        ],
        requested=4, world_size=2, batch_size=1, language_probs={"zh": 1},
        dataset_alpha=1, max_dataset_ratio=1,
    )
    assert plan.actual == 4
    assert all(quota % 2 == 0 for quota in plan.sr_quotas.values())


def test_candidate_enumeration_reaches_smaller_feasible_ddp_size():
    plan = make_speech_plan(
        [
            cell("zh", "d0", 16000, 2, 4.0), cell("zh", "d0", 8000, 1, 16.0),
            cell("zh", "d1", 8000, 4, 16.0), cell("zh", "d1", 24000, 2, 100.0),
            cell("zh", "d2", 16000, 4, 2.0), cell("zh", "d2", 48000, 6, 2.0),
        ],
        requested=9, world_size=3, batch_size=1, language_probs={"zh": 1},
        dataset_alpha=1, max_dataset_ratio=0.4,
    )
    assert plan.actual == 6
    assert all(quota % 3 == 0 for quota in plan.sr_quotas.values())


def test_dataset_sr_options_match_exhaustive_oracle():
    cases = [
        ((1, 1), 2),
        ((2, 1), 2),
        ((2, 2), 3),
    ]
    for capacities, granularity in cases:
        entries = [
            {"key": ("zh", "a", 8000), "sr": 8000, "capacity": capacities[0]},
            {"key": ("zh", "a", 16000), "sr": 16000, "capacity": capacities[1]},
        ]
        for target_a in range(sum(capacities) + 1):
            expected = {
                (values[0] % granularity,)
                for values in product(*(range(capacity + 1) for capacity in capacities))
                if sum(values) == target_a
            }
            options = _dataset_sr_options(
                entries,
                {entry["key"]: 0 for entry in entries[:-1]} | {entries[-1]["key"]: target_a},
                [8000, 16000], granularity,
            )
            assert set(options) == expected


def test_fraction_dataset_cap_uses_exact_ceiling_without_float_rounding():
    assert _dataset_cap(100, Fraction(1, 75), 525) == 7
    plan = make_speech_plan(
        [cell("zh", f"d{index}", 16000, 10, 1.0) for index in range(75)],
        requested=525, world_size=1, batch_size=1, language_probs={"zh": 1},
        max_dataset_ratio=Fraction(1, 75),
    )
    assert plan.actual == 525
    assert all(quota <= 7 for quota in plan.global_dataset_quotas.values())


def test_single_sr_many_groups_skip_sr_residue_planner(monkeypatch):
    def unexpected_compensation(*_args, **_kwargs):
        raise AssertionError("single-SR plans must not invoke residue compensation")

    monkeypatch.setattr(sampling, "_compensate_sr", unexpected_compensation)
    plan = make_speech_plan(
        [cell("zh", f"d{index}", 16000, 10, 1.0) for index in range(100)],
        requested=1000,
        world_size=1,
        batch_size=1,
        dataset_alpha=0,
        max_dataset_ratio=1,
    )

    assert plan.actual == 1000
    assert len(plan.global_dataset_quotas) == 100


def test_large_actual_with_few_keys_is_not_rejected_by_search_budget(monkeypatch):
    prepared = [
        {
            "key": ("zh", "a", 16000),
            "language": "zh",
            "dataset": "a",
            "sr": 16000,
            "capacity": 100_000_000,
            "seconds": 1.0,
        }
    ]
    monkeypatch.setattr(sampling, "_prepare_cells", lambda _cells: prepared)

    plan = make_speech_plan(
        [],
        requested=100_000_000,
        world_size=1,
        batch_size=1,
        language_probs={"zh": 1},
    )

    assert plan.actual == 100_000_000
    assert plan.cell_quotas == {("zh", "a", 16000): 100_000_000}


def test_transport_projection_preserves_reviewer_maximum_aligned_actual():
    cells = [
        cell("en", "a", 16000, 5, 1.0),
        cell("en", "b", 16000, 6, 9.0),
        cell("other", "b", 16000, 1, 2.0),
        cell("zh", "a", 16000, 1, 9.0),
        cell("zh", "b", 16000, 1, 9.0),
    ]
    config = SpeechSamplingConfig(
        language=sampling.LanguageSamplingConfig((
            ("en", Fraction(1, 20)),
            ("other", Fraction(1, 20)),
            ("zh", Fraction(9, 10)),
        )),
        dataset=sampling.DatasetSamplingConfig(
            alpha=Fraction(1), max_ratio=Fraction(1, 2)
        ),
    )
    entries = sampling._prepare_cells(cells)
    probs = dict(config.language.probs)
    context = sampling._prepare_search_context(entries, probs, 1.0)
    projected = sampling._project_transport_actual(
        context, probs, config.dataset.max_ratio, 12, 3
    )

    assert sampling._allocate_controlled(
        entries, 12, probs, 1.0, config.dataset.max_ratio, 3, context
    )[0]
    assert projected == 12
    plan = make_speech_plan(
        cells,
        requested=12,
        world_size=3,
        batch_size=1,
        sampling_config=config,
    )
    assert plan.actual == 12


def test_transport_projection_differential_oracle_checks_every_skipped_candidate():
    languages = ("en", "other", "zh")
    datasets = ("a", "b")
    pair_order = tuple(product(languages, datasets))
    probability_options = (
        {"en": Fraction(1, 20), "other": Fraction(1, 20), "zh": Fraction(9, 10)},
        {"en": Fraction(1, 3), "other": Fraction(1, 3), "zh": Fraction(1, 3)},
    )
    checked_skipped = 0

    for capacities in product((0, 1, 2), repeat=len(pair_order)):
        if any(
            sum(
                capacities[index]
                for index, pair in enumerate(pair_order)
                if pair[0] == language
            ) == 0
            for language in languages
        ):
            continue
        entries = [
            {
                "key": (language, dataset, 16000),
                "language": language,
                "dataset": dataset,
                "sr": 16000,
                "capacity": capacity,
                "seconds": float((1, 2, 3, 5, 7, 11)[index]),
                "seconds_fraction": Fraction((1, 2, 3, 5, 7, 11)[index]),
            }
            for index, ((language, dataset), capacity) in enumerate(
                zip(pair_order, capacities)
            )
            if capacity
        ]
        for probs, alpha, ratio, granularity in product(
            probability_options,
            (0.0, 1.0),
            (Fraction(1, 2), Fraction(1)),
            (1, 2, 3),
        ):
            context = sampling._prepare_search_context(entries, probs, alpha)
            capacity = sum(capacities)
            projected = capacity - capacity % granularity
            upper = sampling._project_transport_actual(
                context, probs, ratio, projected, granularity
            )
            for actual in range(upper + granularity, projected + 1, granularity):
                checked_skipped += 1
                assert not sampling._allocate_controlled(
                    entries, actual, probs, alpha, ratio, granularity, context
                )[0], (
                    f"projection skipped feasible actual={actual}: upper={upper}, "
                    f"projected={projected}, capacities={capacities}, "
                    f"probs={probs}, alpha={alpha}, ratio={ratio}, "
                    f"granularity={granularity}"
                )

    assert checked_skipped > 0


def test_transport_projection_differential_oracle_covers_float_and_three_datasets():
    checked_skipped = 0
    cases = (
        (
            ("en", "zh"),
            ("a", "b", "c"),
            (
                {"en": Fraction(1, 4), "zh": Fraction(3, 4)},
                {"en": 0.2, "zh": 0.8},
            ),
        ),
        (
            ("en", "other", "zh"),
            ("a", "b", "c"),
            (
                {
                    "en": Fraction(1, 20),
                    "other": Fraction(1, 20),
                    "zh": Fraction(9, 10),
                },
                {"en": 0.1, "other": 0.2, "zh": 0.7},
            ),
        ),
    )

    for languages, datasets, probability_options in cases:
        pair_order = tuple(product(languages, datasets))
        for capacities in product((0, 1), repeat=len(pair_order)):
            if any(
                sum(
                    capacities[index]
                    for index, pair in enumerate(pair_order)
                    if pair[0] == language
                ) == 0
                for language in languages
            ) or any(
                sum(
                    capacities[index]
                    for index, pair in enumerate(pair_order)
                    if pair[1] == dataset
                ) == 0
                for dataset in datasets
            ):
                continue
            entries = [
                {
                    "key": (language, dataset, 16000),
                    "language": language,
                    "dataset": dataset,
                    "sr": 16000,
                    "capacity": capacity,
                    "seconds": float(index + 1),
                    "seconds_fraction": Fraction(index + 1),
                }
                for index, ((language, dataset), capacity) in enumerate(
                    zip(pair_order, capacities)
                )
                if capacity
            ]
            for probs, alpha, ratio, granularity in product(
                probability_options,
                (0.0, 0.5, 1.0),
                (Fraction(1, 4), Fraction(1, 2), Fraction(1)),
                (1, 2, 3),
            ):
                context = sampling._prepare_search_context(entries, probs, alpha)
                capacity = sum(capacities)
                projected = capacity - capacity % granularity
                upper = sampling._project_transport_actual(
                    context, probs, ratio, projected, granularity
                )
                for actual in range(
                    upper + granularity, projected + 1, granularity
                ):
                    checked_skipped += 1
                    assert not sampling._allocate_controlled(
                        entries,
                        actual,
                        probs,
                        alpha,
                        ratio,
                        granularity,
                        context,
                    )[0], (
                        f"projection skipped feasible actual={actual}: "
                        f"upper={upper}, projected={projected}, "
                        f"capacities={capacities}, probs={probs}, "
                        f"alpha={alpha}, ratio={ratio}, "
                        f"granularity={granularity}"
                    )

    assert checked_skipped > 0


def test_transport_preflight_skips_proven_infeasible_candidate_range(monkeypatch):
    original = sampling._allocate_controlled
    attempted_actuals = []

    def track_attempt(entries, actual, *args, **kwargs):
        attempted_actuals.append(actual)
        return original(entries, actual, *args, **kwargs)

    monkeypatch.setattr(sampling, "_allocate_controlled", track_attempt)
    plan = make_speech_plan(
        [
            cell("en", "a", 16000, 5000, 3.0),
            cell("zh", "b", 16000, 5000, 1.0),
        ],
        requested=5000,
        world_size=1,
        batch_size=1,
        language_probs={"en": 0.5, "zh": 0.5},
        dataset_alpha=1,
        max_dataset_ratio=1,
    )

    assert plan.actual == 3
    assert attempted_actuals == list(range(attempted_actuals[0], 2, -1))
    assert len(attempted_actuals) < 20
    candidates = plan.summary["joint_solver"]["candidates"]
    assert candidates == (
        {
            "actual": attempted_actuals[0],
            "lowest_actual": 4,
            "count": len(attempted_actuals) - 1,
            "reason": "transport_structure",
        },
        {"actual": 3, "lowest_actual": 3, "count": 1, "reason": None},
    )


def test_transport_preflight_projects_ratio_binding_mismatch(monkeypatch):
    original = sampling._allocate_controlled
    attempted_actuals = []

    def track_attempt(entries, actual, *args, **kwargs):
        attempted_actuals.append(actual)
        return original(entries, actual, *args, **kwargs)

    monkeypatch.setattr(sampling, "_allocate_controlled", track_attempt)
    plan = make_speech_plan(
        [
            cell("en", "a", 16000, 5000, 9.0),
            cell("zh", "b", 16000, 5000, 1.0),
        ],
        requested=5000,
        world_size=1,
        batch_size=1,
        language_probs={"en": 0.9, "zh": 0.1},
        dataset_alpha=1,
        max_dataset_ratio=0.5,
    )

    assert plan.actual == 1
    assert len(attempted_actuals) < 20
    assert attempted_actuals == list(range(attempted_actuals[0], 0, -1))


def test_transport_search_preserves_maximum_with_language_capacity_binding():
    plan = make_speech_plan(
        [
            cell("en", "a", 16000, 5000, 4.0),
            cell("other", "b", 16000, 1, 3.0),
            cell("zh", "b", 16000, 5000, 3.0),
        ],
        requested=5000,
        world_size=1,
        batch_size=1,
        sampling_config=SpeechSamplingConfig(
            language=sampling.LanguageSamplingConfig((
                ("en", Fraction(1, 5)),
                ("other", Fraction(3, 5)),
                ("zh", Fraction(1, 5)),
            )),
            dataset=sampling.DatasetSamplingConfig(
                alpha=Fraction(1), max_ratio=Fraction(1)
            ),
        ),
    )

    assert plan.actual == 9


def test_transport_search_preserves_maximum_with_saturated_language():
    plan = make_speech_plan(
        [
            cell("en", "a", 16000, 100, 3.0),
            cell("en", "b", 16000, 2, 1.0),
            cell("other", "a", 16000, 5000, 1.0),
            cell("other", "b", 16000, 2500, 9.0),
            cell("zh", "a", 16000, 2500, 3.0),
        ],
        requested=5000,
        world_size=1,
        batch_size=1,
        sampling_config=SpeechSamplingConfig(
            language=sampling.LanguageSamplingConfig((
                ("en", Fraction(19, 64)),
                ("other", Fraction(15, 64)),
                ("zh", Fraction(15, 32)),
            )),
            dataset=sampling.DatasetSamplingConfig(
                alpha=Fraction(0), max_ratio=Fraction(1)
            ),
        ),
    )

    assert plan.actual == 11


def test_transport_preflight_projects_finite_cross_edge_capacity(monkeypatch):
    original = sampling._allocate_controlled
    attempted_actuals = []

    def track_attempt(entries, actual, *args, **kwargs):
        attempted_actuals.append(actual)
        return original(entries, actual, *args, **kwargs)

    monkeypatch.setattr(sampling, "_allocate_controlled", track_attempt)
    plan = make_speech_plan(
        [
            cell("en", "a", 16000, 5000, 9.0),
            cell("en", "b", 16000, 1, 1.0),
            cell("zh", "a", 16000, 1, 9.0),
            cell("zh", "b", 16000, 5000, 1.0),
        ],
        requested=5000,
        world_size=1,
        batch_size=1,
        language_probs={"en": 0.5, "zh": 0.5},
        dataset_alpha=1,
        max_dataset_ratio=1,
    )

    assert plan.actual == 3
    assert len(attempted_actuals) < 20
    assert attempted_actuals == list(range(attempted_actuals[0], 2, -1))


def test_transport_search_preserves_maximum_with_dataset_ratio_saturation():
    plan = make_speech_plan(
        [
            cell("en", "a", 16000, 4798, 10.0),
            cell("en", "b", 16000, 2081, 1.0),
            cell("other", "b", 16000, 5212, 1.0),
            cell("zh", "a", 16000, 7285, 10.0),
            cell("zh", "b", 16000, 824, 1.0),
        ],
        requested=10_000,
        world_size=1,
        batch_size=1,
        sampling_config=SpeechSamplingConfig(
            language=sampling.LanguageSamplingConfig((
                ("en", Fraction(5, 8)),
                ("other", Fraction(3, 16)),
                ("zh", Fraction(3, 16)),
            )),
            dataset=sampling.DatasetSamplingConfig(
                alpha=Fraction(1), max_ratio=Fraction(3, 4)
            ),
        ),
    )

    assert plan.actual == 8529


def test_dataset_only_total_cap_projection_avoids_linear_search(monkeypatch):
    original = sampling._allocate_controlled
    attempted_actuals = []
    prepared = [
        {
            "key": ("zh", "small", 16000),
            "language": "zh",
            "dataset": "small",
            "sr": 16000,
            "capacity": 1,
            "seconds": 1.0,
        },
        {
            "key": ("zh", "large", 16000),
            "language": "zh",
            "dataset": "large",
            "sr": 16000,
            "capacity": 200_000,
            "seconds": 200_000.0,
        },
    ]

    def track_attempt(entries, actual, *args, **kwargs):
        attempted_actuals.append(actual)
        return original(entries, actual, *args, **kwargs)

    monkeypatch.setattr(sampling, "_prepare_cells", lambda _cells: prepared)
    monkeypatch.setattr(sampling, "_allocate_controlled", track_attempt)
    plan = make_speech_plan(
        [],
        requested=200_000,
        world_size=1,
        batch_size=1,
        dataset_alpha=0,
        max_dataset_ratio=Fraction(1, 2),
    )

    assert plan.actual == 3
    assert attempted_actuals == [3]


def test_dataset_only_total_cap_projection_handles_multiple_physical_caps_and_granularity(
    monkeypatch,
):
    original = sampling._allocate_controlled
    attempted_actuals = []
    capacities = (1, 1, 200)
    prepared = [
        {
            "key": ("zh", f"d{index}", 16000),
            "language": "zh",
            "dataset": f"d{index}",
            "sr": 16000,
            "capacity": capacity,
            "seconds": float(index + 1),
        }
        for index, capacity in enumerate(capacities)
    ]

    def track_attempt(entries, actual, *args, **kwargs):
        attempted_actuals.append(actual)
        return original(entries, actual, *args, **kwargs)

    monkeypatch.setattr(sampling, "_prepare_cells", lambda _cells: prepared)
    monkeypatch.setattr(sampling, "_allocate_controlled", track_attempt)
    plan = make_speech_plan(
        [],
        requested=200,
        world_size=2,
        batch_size=1,
        dataset_alpha=0,
        max_dataset_ratio=Fraction(1, 3),
    )

    assert plan.actual == 4
    assert attempted_actuals == [4]


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [(Fraction(2, 5), 3), (Fraction(2, 3), 5)],
)
def test_dataset_only_total_cap_projection_covers_ratio_total_paths(
    ratio, expected
):
    entries = [
        {
            "key": ("zh", "small", 16000),
            "language": "zh",
            "dataset": "small",
            "sr": 16000,
            "capacity": 1,
            "seconds": 1.0,
            "seconds_fraction": Fraction(1),
        },
        {
            "key": ("zh", "large", 16000),
            "language": "zh",
            "dataset": "large",
            "sr": 16000,
            "capacity": 200,
            "seconds": 2.0,
            "seconds_fraction": Fraction(2),
        },
    ]
    context = sampling._prepare_search_context(entries, None, 0.0)

    assert sampling._project_transport_actual(
        context, None, ratio, 200, 1
    ) == expected


def test_dataset_total_cap_projection_large_denominator_has_bounded_steps(monkeypatch):
    original = sampling._fraction_ceiling
    ceiling_calls = 0

    def count_ceiling(value):
        nonlocal ceiling_calls
        ceiling_calls += 1
        return original(value)

    monkeypatch.setattr(sampling, "_fraction_ceiling", count_ceiling)
    denominator = 100_000
    projected = 10**30 - (10**30 % 2)

    assert sampling._project_dataset_total_cap(
        (10**30, 10**30, 1),
        Fraction(denominator - 1, 2 * denominator),
        projected,
        2,
    ) == 2 * denominator - 2
    assert ceiling_calls < 100


@pytest.mark.parametrize(
    ("capacities", "ratios"),
    [
        ((), (Fraction(1), Fraction(1, 97))),
        ((0,), (Fraction(1), Fraction(1, 97))),
        ((0, 1), (Fraction(1, 7), Fraction(1))),
        ((1, 2), (Fraction(1, 10), Fraction(1, 2), Fraction(1))),
        ((0, 1, 4), (Fraction(2, 5), Fraction(2, 3), Fraction(999, 2000))),
        ((1, 2, 4), (Fraction(1, 10), Fraction(1, 2), Fraction(1))),
    ],
)
def test_dataset_total_cap_helper_matches_exhaustive_aligned_oracle(
    capacities, ratios
):
    for ratio, granularity, projected in product(ratios, range(1, 6), range(0, 31)):
        projected -= projected % granularity
        expected = max(
            (
                actual
                for actual in range(0, projected + 1, granularity)
                if actual <= sum(
                    min(capacity, sampling._fraction_ceiling(ratio * actual))
                    for capacity in capacities
                )
            ),
            default=0,
        )
        assert sampling._project_dataset_total_cap(
            capacities, ratio, projected, granularity
        ) == expected, (
            f"capacities={capacities}, ratio={ratio}, "
            f"projected={projected}, granularity={granularity}"
        )


def test_dataset_only_projection_differential_oracle_checks_every_skipped_candidate():
    checked_skipped = 0
    checked_ratio_total_at_least_one = 0
    for dataset_count, capacities, ratio, granularity in product(
        (1, 2, 3),
        product((1, 2, 4), repeat=3),
        (
            Fraction(1, 10),
            Fraction(3, 10),
            Fraction(2, 5),
            Fraction(1, 2),
            Fraction(2, 3),
            Fraction(3, 4),
            Fraction(1),
        ),
        (1, 2, 3),
    ):
        selected_capacities = capacities[:dataset_count]
        entries = [
            {
                "key": ("zh", f"d{index}", 16000),
                "language": "zh",
                "dataset": f"d{index}",
                "sr": 16000,
                "capacity": capacity,
                "seconds": float(index + 1),
                "seconds_fraction": Fraction(index + 1),
            }
            for index, capacity in enumerate(selected_capacities)
        ]
        context = sampling._prepare_search_context(entries, None, 0.0)
        capacity = sum(selected_capacities)
        projected = capacity - capacity % granularity
        feasible_by_total_cap = [
            actual
            for actual in range(granularity, projected + 1, granularity)
            if actual <= sum(
                min(capacity, sampling._fraction_ceiling(ratio * actual))
                for capacity in selected_capacities
            )
        ]
        expected = max(feasible_by_total_cap, default=0)
        upper = sampling._project_transport_actual(
            context, None, ratio, projected, granularity
        )
        assert upper == expected, (
            f"dataset projection is not exact: upper={upper}, expected={expected}, "
            f"projected={projected}, capacities={selected_capacities}, "
            f"ratio={ratio}, granularity={granularity}"
        )
        if dataset_count * ratio >= 1 and upper < projected:
            checked_ratio_total_at_least_one += 1
        for actual in range(upper + granularity, projected + 1, granularity):
            checked_skipped += 1
            assert not sampling._allocate_controlled(
                entries, actual, None, 0.0, ratio, granularity, context
            )[0], (
                f"dataset projection skipped feasible actual={actual}: "
                f"upper={upper}, projected={projected}, "
                f"capacities={selected_capacities}, ratio={ratio}, "
                f"granularity={granularity}"
            )

    assert checked_skipped > 0
    assert checked_ratio_total_at_least_one > 0


def test_candidate_failure_summary_is_constant_size_for_long_search(monkeypatch):
    original = sampling._allocate_controlled

    def fail_until_three(entries, actual, *args, **kwargs):
        if actual > 3:
            return False, "language_marginal", {}
        return original(entries, actual, *args, **kwargs)

    monkeypatch.setattr(sampling, "_allocate_controlled", fail_until_three)
    plan = make_speech_plan(
        [cell("zh", "large", 16000, 5000, 5000.0)],
        requested=5000,
        world_size=1,
        batch_size=1,
        language_probs={"zh": 1},
    )

    assert plan.actual == 3
    assert plan.summary["joint_solver"]["version"] == 2
    assert plan.summary["joint_solver"]["candidates"] == (
        {
            "actual": 5000,
            "lowest_actual": 4,
            "count": 4997,
            "reason": "language_capacity",
        },
        {"actual": 3, "lowest_actual": 3, "count": 1, "reason": None},
    )


def test_infeasible_search_reports_stable_last_failure_stage():
    with pytest.raises(
        ValueError,
        match=(
            r"no positive actual.*controllers=language\+dataset.*"
            r"projected=8.*granularity=8.*candidates_tried=1.*"
            r"last_failure=transport_structure"
        ),
    ):
        make_speech_plan(
            [cell("zh", "a", 16000, 6, 6.0), cell("en", "b", 16000, 6, 6.0)],
            requested=8,
            world_size=8,
            batch_size=1,
            language_probs={"zh": 0.75, "en": 0.25},
            dataset_alpha=0,
            max_dataset_ratio=0.5,
        )


def test_sr_residue_dp_fails_fast_when_budget_is_exceeded(monkeypatch):
    monkeypatch.setattr(sampling, "_SR_DP_TRANSITION_BUDGET", 1)
    with pytest.raises(
        ValueError,
        match=(
            r"SR-residue planner budget exceeded.*"
            r"estimated states=.*transitions=.*materialized_entries=.*"
            r"controllers=language\+dataset.*projected=2.*granularity=1.*"
            r"candidates_tried=1.*last_failure=sr_residue_budget"
        ),
    ):
        make_speech_plan(
            [cell("zh", "a", 8000, 2, 1.0), cell("zh", "a", 16000, 2, 1.0)],
            requested=2, world_size=1, batch_size=1, language_probs={"zh": 1}, max_dataset_ratio=1,
        )


def test_sr_residue_dp_fails_fast_on_weighted_materialization_budget(monkeypatch):
    monkeypatch.setattr(sampling, "_SR_DP_MATERIALIZATION_BUDGET", 1)
    cells = [
        cell("zh", f"d{dataset}", 8000, 1, 1.0)
        for dataset in range(4)
    ] + [
        cell("zh", f"d{dataset}", 16000, 1, 1.0)
        for dataset in range(4)
    ]
    with pytest.raises(ValueError, match="SR-residue planner budget exceeded.*epoch quota"):
        make_speech_plan(
            cells, requested=4, world_size=1, batch_size=1,
            language_probs={"zh": 1}, max_dataset_ratio=1,
        )


def test_dataset_margin_is_global_across_languages():
    # 冻结设计中的全局 dataset 列边际：dataset 配额跨 language 合并，
    # cap 分母为全局 actual，而不是每个 language 内各自按比例分配。
    leaves = [
        cell("en", "large", 16000, 3, 90.0),
        cell("en", "small", 16000, 3, 10.0),
        cell("zh", "large", 16000, 3, 90.0),
        cell("zh", "small", 16000, 3, 10.0),
    ]
    plan = make_speech_plan(
        leaves,
        requested=8,
        world_size=1,
        batch_size=1,
        language_probs={"en": 0.5, "zh": 0.5},
        dataset_alpha=1.0,
        max_dataset_ratio=0.75,
    )

    assert plan.language_quotas == {"en": 4, "zh": 4}
    assert plan.global_dataset_quotas == {"large": 6, "small": 2}
    assert sum(q for (lang, ds, sr), q in plan.cell_quotas.items() if ds == "large") == 6


def test_global_dataset_cap_uses_global_actual_denominator():
    """判别性 case：cap 分母必须是全局 actual，而非 per-language target。

    big 同时存在于 zh（capacity 8，seconds 90）与 en（capacity 2，
    seconds 10），language_probs {zh: 0.8, en: 0.2}，actual=10：
    - 全局 cap 语义：cap = ceil(0.3 * 10) = 3；
    - per-language cap 语义：cap = ceil(0.3 * 8) + ceil(0.3 * 2) = 3 + 1 = 4。
    big 的 effective_seconds 显著大于 filler（90 vs 1），alpha=1 下
    big 权重占优，per-language 语义必然把 big 顶到 3+1=4；因此断言
    ``global_dataset_quotas["big"] <= 3`` 在 per-language 语义下必失败。
    """
    leaves = [
        cell("zh", "big", 16000, 8, 90.0),
        cell("en", "big", 16000, 2, 10.0),
        cell("zh", "f1", 16000, 3, 1.0),
        cell("zh", "f2", 16000, 3, 1.0),
        cell("zh", "f3", 16000, 2, 1.0),
        cell("en", "e1", 16000, 2, 1.0),
    ]
    plan = make_speech_plan(
        leaves,
        requested=10,
        world_size=1,
        batch_size=1,
        language_probs={"zh": 0.8, "en": 0.2},
        dataset_alpha=1,
        max_dataset_ratio=0.3,
    )

    assert plan.actual == 10
    assert plan.language_quotas == {"en": 2, "zh": 8}
    assert plan.global_dataset_quotas["big"] <= 3


def test_alternate_transport_flow_avoids_false_sr_shrink():
    # 两 language × 两 dataset × 两 SR 的 transportation 图：旧 fast path 的
    # 确定性首解（max-flow BFS 顺序）把 1 单位送到 (en,a) 的 16k，SR totals
    # 为 {16000: 2, 48000: 6}，对 granularity=4 不对齐；joint solver 联合
    # 搜索找到 alternate flow（两轴边际不变，SR totals {16000: 0, 48000: 8}
    # 全对齐），actual=8 不得 shrink。
    leaves = [
        cell("en", "a", 16000, 1, 1.0),
        cell("en", "b", 48000, 4, 1.0),
        cell("zh", "a", 48000, 4, 1.0),
        cell("zh", "b", 16000, 4, 1.0),
    ]
    pair_capacity = {("en", "a"): 1, ("en", "b"): 4, ("zh", "a"): 4, ("zh", "b"): 4}
    first_flow = sampling._allocate_joint_transport(
        pair_capacity, {"en": 4, "zh": 4}, {"a": 4, "b": 4}
    )
    first_sr = {
        16000: first_flow[("en", "a")] + first_flow[("zh", "b")],
        48000: first_flow[("en", "b")] + first_flow[("zh", "a")],
    }
    alternate_flow = {("en", "a"): 0, ("en", "b"): 4, ("zh", "a"): 4, ("zh", "b"): 0}
    alternate_sr = {
        16000: alternate_flow[("en", "a")] + alternate_flow[("zh", "b")],
        48000: alternate_flow[("en", "b")] + alternate_flow[("zh", "a")],
    }
    assert any(total % 4 for total in first_sr.values())
    assert {
        language: sum(q for (lang, _ds), q in alternate_flow.items() if lang == language)
        for language in ("en", "zh")
    } == {"en": 4, "zh": 4}
    assert {
        dataset: sum(q for (_lang, ds), q in alternate_flow.items() if ds == dataset)
        for dataset in ("a", "b")
    } == {"a": 4, "b": 4}
    assert all(total % 4 == 0 for total in alternate_sr.values())

    plan = make_speech_plan(
        leaves,
        requested=8,
        world_size=2,
        batch_size=2,
        drop_last=True,
        language_probs={"en": 0.5, "zh": 0.5},
        dataset_alpha=0,
        max_dataset_ratio=1,
    )
    assert plan.actual == 8
    assert plan.language_quotas == {"en": 4, "zh": 4}
    assert plan.global_dataset_quotas == {"a": 4, "b": 4}
    assert all(total % 4 == 0 for total in plan.sr_quotas.values())


# ---------------------------------------------------------------------------
# Task 4: public leaf/plan contract 与 joint solver 集成
# ---------------------------------------------------------------------------


def make_leaf(language, dataset, sr, indices, seconds, speaker=None):
    return SpeechLeafMetadata(
        language=language,
        dataset_id=dataset,
        speaker=speaker,
        batching_sr=sr,
        uid_indices=tuple(indices),
        effective_seconds=Fraction(str(seconds)),
    )


def make_leaves(*specs):
    """spec: ``(language, dataset, sr, count, seconds[, speaker])``；自动分配
    全局唯一的连续 ``uid_indices``。"""
    leaves, start = [], 0
    for spec in specs:
        language, dataset, sr, count, seconds = spec[:5]
        speaker = spec[5] if len(spec) > 5 else None
        leaves.append(
            make_leaf(language, dataset, sr, range(start, start + count), seconds, speaker)
        )
        start += count
    return leaves


def speaker_config(cap, language=None, dataset=None):
    return SpeechSamplingConfig(
        language=language, dataset=dataset, speaker=SpeakerSamplingConfig(cap)
    )


def _stable_json(value):
    def normalize(item):
        if isinstance(item, (MappingProxyType, dict)):
            return {key: normalize(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalize(val) for val in item]
        return item

    return json.dumps(normalize(value), sort_keys=True, default=str)


def _compat_summary_json(plan):
    """legacy-facing summary（剔除 Task 4 新增的 ``joint_solver`` 子对象）。"""
    return _stable_json(
        {key: val for key, val in plan.summary.items() if key != "joint_solver"}
    )


def test_public_leaf_and_plan_field_contracts():
    """Step 1：SpeechLeafMetadata / SpeechPlan 的 public fields 精确集合。"""
    leaf = make_leaf("zh", "a", 16000, [0, 1], 2.0)
    assert [field.name for field in fields(leaf)] == [
        "language", "dataset_id", "speaker", "batching_sr",
        "uid_indices", "effective_seconds",
    ]
    plan = make_speech_plan(
        [leaf], requested=2, world_size=1, batch_size=1, language_probs={"zh": 1}
    )
    assert [field.name for field in fields(plan)] == [
        "requested", "effective_requested", "actual",
        "language_quotas", "global_dataset_quotas", "cell_quotas",
        "leaf_quotas", "speaker_quotas", "sr_quotas", "summary",
    ]
    assert plan.leaf_quotas == {("zh", "a", None, 16000): 2}
    assert plan.cell_quotas == {("zh", "a", 16000): 2}
    assert plan.speaker_quotas == {}
    assert plan.global_dataset_quotas == {}


@pytest.mark.parametrize(
    "effective_seconds",
    [Fraction(1, 10**1000), Fraction(10**1000, 1)],
)
def test_leaf_effective_seconds_rejects_unsupported_float_range(effective_seconds):
    leaf = SpeechLeafMetadata(
        language="zh",
        dataset_id="a",
        speaker=None,
        batching_sr=16000,
        uid_indices=(0,),
        effective_seconds=effective_seconds,
    )

    with pytest.raises(ValueError, match="leaf 0.*effective_seconds"):
        make_speech_plan(
            [leaf], requested=1, world_size=1, batch_size=1,
            language_probs={"zh": 1},
        )


def test_merged_cell_seconds_overflow_has_cell_context():
    leaves = [
        make_leaf("zh", "a", 16000, [0], 10**308, ("a", "s1")),
        make_leaf("zh", "a", 16000, [1], 10**308, ("a", "s2")),
    ]

    with pytest.raises(
        ValueError,
        match=r"cell \('zh', 'a', 16000\).*effective_seconds.*supported float range",
    ):
        make_speech_plan(
            leaves,
            requested=1,
            world_size=1,
            batch_size=1,
            language_probs={"zh": 1},
        )


def test_leaf_input_validation_is_strict():
    good = make_leaf("zh", "a", 16000, [0], 1.0, ("a", "s1"))
    # 跨 leaf 重复 uid_index
    with pytest.raises(ValueError, match="duplicate uid_index"):
        make_speech_plan(
            [good, make_leaf("zh", "b", 16000, [0], 1.0)],
            requested=1, world_size=1, batch_size=1, language_probs={"zh": 1},
        )
    # speaker 非 canonical 二元组 / speaker dataset 与 leaf dataset 不一致
    with pytest.raises(ValueError, match="speaker"):
        make_speech_plan(
            [make_leaf("zh", "a", 16000, [0], 1.0, "s1")],
            requested=1, world_size=1, batch_size=1, language_probs={"zh": 1},
        )
    with pytest.raises(ValueError, match="speaker"):
        make_speech_plan(
            [make_leaf("zh", "a", 16000, [0], 1.0, ("b", "s1"))],
            requested=1, world_size=1, batch_size=1, language_probs={"zh": 1},
        )
    # 空 uid_indices / 非正 effective_seconds / 非法 sr
    with pytest.raises(ValueError, match="uid_indices"):
        make_speech_plan(
            [make_leaf("zh", "a", 16000, [], 1.0)],
            requested=1, world_size=1, batch_size=1, language_probs={"zh": 1},
        )
    with pytest.raises(ValueError, match="effective_seconds"):
        make_speech_plan(
            [make_leaf("zh", "a", 16000, [0], 0.0)],
            requested=1, world_size=1, batch_size=1, language_probs={"zh": 1},
        )
    with pytest.raises(ValueError, match="batching_sr"):
        make_speech_plan(
            [make_leaf("zh", "a", 0, [0], 1.0)],
            requested=1, world_size=1, batch_size=1, language_probs={"zh": 1},
        )
    # 重复 leaf key
    with pytest.raises(ValueError, match="duplicate leaf key"):
        make_speech_plan(
            [good, make_leaf("zh", "a", 16000, [1], 1.0, ("a", "s1"))],
            requested=1, world_size=1, batch_size=1, language_probs={"zh": 1},
        )
    # cell 与 leaf 混合输入
    with pytest.raises(ValueError, match="mixed"):
        make_speech_plan(
            [good, cell("zh", "b", 16000, 1, 1.0)],
            requested=1, world_size=1, batch_size=1, language_probs={"zh": 1},
        )
    # speaker controller 启用但 leaf 全部缺失 identity
    with pytest.raises(ValueError, match="speaker.*metadata|identity"):
        make_speech_plan(
            [make_leaf("zh", "a", 16000, [0, 1], 2.0)],
            requested=2, world_size=1, batch_size=1,
            sampling_config=speaker_config(2),
        )


def test_leaf_input_global_dataset_cap_uses_global_actual_denominator():
    """Step 2：leaf 输入下 dataset cap 分母仍为全局 actual（跨 language）。"""
    leaves = make_leaves(
        ("zh", "big", 16000, 8, 90.0),
        ("en", "big", 16000, 2, 10.0),
        ("zh", "f1", 16000, 3, 1.0),
        ("zh", "f2", 16000, 3, 1.0),
        ("zh", "f3", 16000, 2, 1.0),
        ("en", "e1", 16000, 2, 1.0),
    )
    plan = make_speech_plan(
        leaves, requested=10, world_size=1, batch_size=1,
        language_probs={"zh": 0.8, "en": 0.2},
        dataset_alpha=1, max_dataset_ratio=0.3,
    )
    assert plan.actual == 10
    assert plan.language_quotas == {"en": 2, "zh": 8}
    assert plan.global_dataset_quotas["big"] <= math.ceil(Fraction(3, 10) * 10)
    assert set(plan.leaf_quotas) == {
        ("zh", "big", None, 16000), ("en", "big", None, 16000),
        ("zh", "f1", None, 16000), ("zh", "f2", None, 16000),
        ("zh", "f3", None, 16000), ("en", "e1", None, 16000),
    }


def test_speaker_only_mode_caps_identities_without_margins():
    """Step 3：仅 speaker 时无 exact margin，从自然目标出发取最大可行 actual。"""
    leaves = make_leaves(
        ("zh", "a", 16000, 5, 1.0, ("a", "s1")),
        ("zh", "b", 16000, 5, 1.0, ("b", "s2")),
    )
    plan = make_speech_plan(
        leaves, requested=10, world_size=1, batch_size=1,
        sampling_config=speaker_config(2),
    )
    assert plan.actual == 4
    assert plan.speaker_quotas == {("a", "s1"): 2, ("b", "s2"): 2}
    assert plan.language_quotas == {}
    assert plan.global_dataset_quotas == {}
    assert sum(plan.cell_quotas.values()) == 4
    assert plan.summary["joint_solver"]["speaker"]["cap"] == 2
    assert plan.summary["joint_solver"]["fast_path"] is False


def test_speaker_capacity_projection_reaches_feasible_candidate_beyond_old_budget():
    leaves = make_leaves(
        ("zh", "a", 16000, 2001, 1.0, ("a", "s1")),
    )

    plan = make_speech_plan(
        leaves, requested=2001, world_size=1, batch_size=1,
        sampling_config=speaker_config(1),
    )

    assert plan.effective_requested == 1
    assert plan.actual == 1
    assert len(plan.summary["joint_solver"]["candidates"]) == 1
    assert all(
        candidate["reason"] != "candidate_search_budget"
        for candidate in plan.summary["joint_solver"]["candidates"]
    )


def test_speaker_language_projection_reaches_reviewer_maximum_with_default_budget():
    """P1-1 reviewer 原图：高 requested 必须直接抵达最大可行 1789。"""
    leaves = make_leaves(
        ("en", "d", 16000, 4830, 1.0, ("d", "en-1")),
        ("en", "d", 16000, 4830, 1.0, ("d", "en-2")),
        ("zh", "d", 16000, 4830, 1.0, ("d", "zh-1")),
    )
    config = speaker_config(
        1610,
        language=sampling.LanguageSamplingConfig((
            ("en", Fraction(1, 10)),
            ("zh", Fraction(9, 10)),
        )),
    )

    plan = make_speech_plan(
        leaves,
        requested=4830,
        world_size=1,
        batch_size=1,
        sampling_config=config,
    )

    assert plan.effective_requested == 4830
    assert plan.actual == 1789
    assert dict(plan.language_quotas) == {"en": 179, "zh": 1610}
    assert dict(plan.speaker_quotas) == {
        ("d", "en-1"): 0,
        ("d", "en-2"): 179,
        ("d", "zh-1"): 1610,
    }
    assert plan.summary["joint_solver"]["candidates"] == (
        {
            "actual": 1789,
            "lowest_actual": 1789,
            "count": 1,
            "reason": None,
        },
    )
    stats = plan.summary["joint_solver"]["solver_stats"]
    assert stats["states"] < 1_000
    assert stats["transitions"] < 1_000


def _exhaustive_speaker_margin_feasible(
    leaves, actual, language_quotas, dataset_quotas, speaker_cap, granularity
):
    for quotas in product(*(range(leaf.capacity + 1) for leaf in leaves)):
        if sum(quotas) != actual:
            continue
        if language_quotas is not None and any(
            sum(
                quota
                for leaf, quota in zip(leaves, quotas)
                if leaf.key[0] == language
            ) != target
            for language, target in language_quotas.items()
        ):
            continue
        if dataset_quotas is not None and any(
            sum(
                quota
                for leaf, quota in zip(leaves, quotas)
                if leaf.key[1] == dataset
            ) != target
            for dataset, target in dataset_quotas.items()
        ):
            continue
        if any(
            sum(
                quota
                for leaf, quota in zip(leaves, quotas)
                if leaf.key[2] == speaker
            ) > speaker_cap
            for speaker in {leaf.key[2] for leaf in leaves}
        ):
            continue
        sr_totals = {
            sr: sum(
                quota
                for leaf, quota in zip(leaves, quotas)
                if leaf.key[3] == sr
            )
            for sr in {leaf.key[3] for leaf in leaves}
        }
        if all(total % granularity == 0 for total in sr_totals.values()):
            return True
    return False


def test_speaker_margin_projection_exhaustive_oracle_proves_skipped_candidates():
    """language/dataset/联合轴跳过的小图 candidate 均由穷举证明不可行。"""
    checked_skipped = 0
    specs = (
        ("en", "a", ("a", "s1")),
        ("zh", "a", ("a", "s1")),
        ("en", "b", ("b", "s2")),
        ("zh", "b", ("b", "s3")),
    )
    controller_options = (
        ({"en": Fraction(1, 4), "zh": Fraction(3, 4)}, None),
        (None, 0.0),
        ({"en": Fraction(1, 2), "zh": Fraction(1, 2)}, 0.0),
    )
    for capacities, (probs, alpha), granularity in product(
        product((0, 1, 2), repeat=4), controller_options, (1, 2)
    ):
        if probs is not None and (
            capacities[0] + capacities[2] == 0
            or capacities[1] + capacities[3] == 0
        ):
            continue
        if alpha is not None and (
            capacities[0] + capacities[1] == 0
            or capacities[2] + capacities[3] == 0
        ):
            continue
        entries = [
            {
                "key": (language, dataset, speaker, 16000),
                "language": language,
                "dataset": dataset,
                "speaker": speaker,
                "sr": 16000,
                "capacity": capacity,
                "seconds": 1.0,
                "seconds_fraction": Fraction(1),
            }
            for (language, dataset, speaker), capacity in zip(specs, capacities)
            if capacity
        ]
        speaker_cap = 1
        identity_capacities = {}
        for entry in entries:
            identity = entry["speaker"]
            identity_capacities[identity] = (
                identity_capacities.get(identity, 0) + entry["capacity"]
            )
        effective = sum(
            min(capacity, speaker_cap)
            for capacity in identity_capacities.values()
        )
        projected = effective - effective % granularity
        legacy = [
            sampling._legacy_entry(entry)
            for entry in sampling._merge_leaf_entries(entries)
        ]
        context = sampling._prepare_search_context(legacy, probs, alpha)
        ratio = Fraction(1) if alpha is not None else None
        upper = sampling._project_speaker_margin_actual(
            entries,
            context,
            probs,
            ratio,
            speaker_cap,
            projected,
            granularity,
        )
        solver_leaves = [
            sampling.SolverLeaf(entry["key"], entry["capacity"], Fraction())
            for entry in entries
        ]
        for actual in range(upper + granularity, projected + 1, granularity):
            checked_skipped += 1
            language_quotas = (
                sampling._bounded_hamilton(
                    probs, context["language_capacities"], actual
                )
                if probs is not None
                else None
            )
            dataset_quotas = None
            if alpha is not None:
                dataset_caps = {
                    key: sampling._dataset_cap(capacity, ratio, actual)
                    for key, capacity in context["dataset_capacities"].items()
                }
                dataset_quotas = sampling._bounded_hamilton(
                    context["dataset_weights"], dataset_caps, actual
                )
            margins_exist = (
                (probs is None or language_quotas is not None)
                and (alpha is None or dataset_quotas is not None)
            )
            assert not margins_exist or not _exhaustive_speaker_margin_feasible(
                solver_leaves,
                actual,
                language_quotas,
                dataset_quotas,
                speaker_cap,
                granularity,
            ), (
                f"projection skipped feasible actual={actual}: upper={upper}, "
                f"capacities={capacities}, probs={probs}, alpha={alpha}, "
                f"granularity={granularity}"
            )

    assert checked_skipped > 0


@pytest.mark.parametrize(
    ("language_on", "dataset_on", "speaker_on"),
    [(False, False, True)] + [
        combo for combo in product([False, True], repeat=3) if any(combo)
    ][1:],
)
def test_all_controller_combinations(language_on, dataset_on, speaker_on):
    """Step 3：7 种非空组合下 margin 字段空/非空与 speaker cap 语义正确。"""
    from core.data.sampling import DatasetSamplingConfig, LanguageSamplingConfig

    leaves = make_leaves(
        ("zh", "a", 16000, 4, 2.0, ("a", "s1")),
        ("en", "b", 16000, 4, 2.0, ("b", "s2")),
    )
    config = SpeechSamplingConfig(
        language=(
            LanguageSamplingConfig(
                probs=(("en", Fraction(1, 2)), ("other", Fraction(0)), ("zh", Fraction(1, 2)))
            )
            if language_on else None
        ),
        dataset=DatasetSamplingConfig(Fraction(0), Fraction(1)) if dataset_on else None,
        speaker=SpeakerSamplingConfig(3) if speaker_on else None,
    )
    plan = make_speech_plan(
        leaves, requested=8, world_size=1, batch_size=1, sampling_config=config
    )
    assert (plan.language_quotas != {}) == language_on
    assert (plan.global_dataset_quotas != {}) == dataset_on
    assert (plan.speaker_quotas != {}) == speaker_on
    if speaker_on:
        assert all(quota <= 3 for quota in plan.speaker_quotas.values())
    assert sum(plan.cell_quotas.values()) == plan.actual
    assert plan.summary["joint_solver"]["fast_path"] is not speaker_on


def test_same_raw_speaker_in_different_datasets_has_independent_caps():
    """Step 4：同裸 speaker label 跨 dataset 是独立 identity，各自吃满 cap。"""
    leaves = make_leaves(
        ("zh", "a", 16000, 4, 1.0, ("a", "s1")),
        ("zh", "b", 16000, 4, 1.0, ("b", "s1")),
    )
    plan = make_speech_plan(
        leaves, requested=4, world_size=1, batch_size=1,
        sampling_config=speaker_config(2),
    )
    assert plan.speaker_quotas == {("a", "s1"): 2, ("b", "s1"): 2}
    assert plan.actual == 4


def test_same_canonical_speaker_across_languages_shares_one_cap():
    """Step 4：同一 canonical identity 跨 language 共享一个 absolute cap。"""
    leaves = make_leaves(
        ("zh", "a", 16000, 3, 1.0, ("a", "s1")),
        ("en", "a", 16000, 3, 1.0, ("a", "s1")),
    )
    plan = make_speech_plan(
        leaves, requested=4, world_size=1, batch_size=1,
        sampling_config=speaker_config(2),
    )
    assert plan.speaker_quotas[("a", "s1")] <= 2
    assert plan.actual == 2


def test_speaker_cap_shrinks_to_largest_feasible_ddp_candidate():
    """Step 5：requested=11, ws=2, bs=2, drop_last → 搜 8（不可行）再 4（可行）。"""
    leaves = make_leaves(("zh", "a", 16000, 10, 1.0, ("a", "s1")))
    plan = make_speech_plan(
        leaves, requested=11, world_size=2, batch_size=2, drop_last=True,
        sampling_config=speaker_config(5),
    )
    assert plan.actual == 4
    assert plan.speaker_quotas == {("a", "s1"): 4}


def test_margins_are_recomputed_per_candidate_not_clipped():
    """Step 5：actual=8 的 margin 不可行时，最终 margin 必须对 4 重算（和为 4）。"""
    leaves = make_leaves(
        ("zh", "a", 16000, 6, 1.0, ("a", "s1")),
        ("en", "b", 16000, 6, 1.0, ("b", "s2")),
    )
    config = SpeechSamplingConfig(
        language=sampling.LanguageSamplingConfig(
            probs=(("en", Fraction(1, 2)), ("other", Fraction(0)), ("zh", Fraction(1, 2)))
        ),
        speaker=SpeakerSamplingConfig(3),
    )
    plan = make_speech_plan(
        leaves, requested=11, world_size=2, batch_size=2, drop_last=True,
        sampling_config=config,
    )
    # actual=8：language margin {zh:4, en:4}，zh 单 identity cap=3 → 不可行；
    # actual=4：margin 重算为 {zh:2, en:2}（若从 8 裁剪和不可能为 4）
    assert plan.actual == 4
    assert plan.language_quotas == {"en": 2, "zh": 2}
    assert sum(plan.language_quotas.values()) == 4


def test_solver_budget_exceeded_propagates_unwrapped():
    """Step 6：solver 预算耗尽原样上抛 SolverBudgetExceeded，不包装为 shrink。"""
    leaves = make_leaves(
        ("zh", "a", 16000, 4, 1.0, ("a", "s1")),
        ("en", "b", 16000, 4, 1.0, ("b", "s2")),
    )
    with pytest.raises(SolverBudgetExceeded, match="budget exceeded"):
        make_speech_plan(
            leaves, requested=4, world_size=1, batch_size=1,
            sampling_config=speaker_config(2),
            budget=SolverBudget(max_states=1),
        )


def test_speaker_capacity_shortfall_records_speaker_capacity_reason():
    """Step 6：speaker cap 总容量不足按 speaker_capacity 记录 reason code。"""
    leaves = make_leaves(("zh", "a", 16000, 8, 1.0, ("a", "s1")))
    with pytest.raises(ValueError, match="last_failure=speaker_capacity"):
        make_speech_plan(
            leaves, requested=8, world_size=4, batch_size=1,
            sampling_config=speaker_config(2),
        )


def test_speaker_structural_conflict_projects_to_largest_feasible_candidate():
    """speaker-adjusted language capacity 直接排除 8/7，保留最大可行 6。

    actual=8 时 margin {zh:6, en:2}，zh 侧两个 identity 各 cap 2（合计最多
    4 < 6）；actual=7 的 zh Hamilton quota 5 也超过 4。必要上界因此直接投影
    到 6，不消耗 solver budget 逐候选重判；actual=6 的 margin {zh:4,en:2}
    恰好可行。
    """
    leaves = make_leaves(
        ("zh", "a", 16000, 4, 1.0, ("a", "s1")),
        ("zh", "a", 16000, 4, 1.0, ("a", "s2")),
        ("en", "b", 16000, 4, 1.0, ("b", "s3")),
        ("en", "b", 16000, 4, 1.0, ("b", "s4")),
    )
    config = SpeechSamplingConfig(
        language=sampling.LanguageSamplingConfig(
            probs=(
                ("en", Fraction(1, 4)),
                ("other", Fraction(0)),
                ("zh", Fraction(3, 4)),
            )
        ),
        speaker=SpeakerSamplingConfig(2),
    )
    plan = make_speech_plan(
        leaves, requested=8, world_size=1, batch_size=1, sampling_config=config
    )
    assert plan.actual == 6
    assert plan.language_quotas == {"en": 2, "zh": 4}
    assert plan.summary["joint_solver"]["candidates"] == (
        {"actual": 6, "lowest_actual": 6, "count": 1, "reason": None},
    )


def test_speaker_summary_reports_binding_usage_and_estimated_primary_seconds():
    """speaker summary 同时报告 binding、usage 和 leaf 平均时长估算。"""
    leaves = make_leaves(
        ("zh", "a", 16000, 2, 2.0, ("a", "s1")),
        ("zh", "b", 16000, 8, 8.0, ("b", "s2")),
    )
    plan = make_speech_plan(
        leaves, requested=3, world_size=1, batch_size=1,
        sampling_config=speaker_config(2),
    )
    assert plan.speaker_quotas == {("a", "s1"): 1, ("b", "s2"): 2}
    speaker = plan.summary["joint_solver"]["speaker"]
    assert speaker["cap"] == 2
    assert dict(speaker["usage"]) == dict(plan.speaker_quotas)
    assert dict(speaker["binding"]) == {("a", "s1"): False, ("b", "s2"): True}
    assert speaker["primary_samples"] == 3
    assert "primary_seconds" not in speaker
    assert speaker["estimated_primary_seconds"] == pytest.approx(3.0)


def test_speaker_summary_seconds_are_leaf_average_estimate_for_heterogeneous_uids():
    """leaf 汇总 1s/5s UID 时，planner 只能报告 quota×平均值 3s。"""
    uid_seconds = (Fraction(1), Fraction(5))
    leaf = SpeechLeafMetadata(
        language="zh",
        dataset_id="a",
        speaker=("a", "s1"),
        batching_sr=16000,
        uid_indices=(0, 1),
        effective_seconds=sum(uid_seconds, Fraction()),
    )

    plan = make_speech_plan(
        [leaf], requested=1, world_size=1, batch_size=1,
        sampling_config=speaker_config(1),
    )

    speaker = plan.summary["joint_solver"]["speaker"]
    assert speaker["primary_samples"] == 1
    assert speaker["estimated_primary_seconds"] == pytest.approx(3.0)


def test_speaker_summary_seconds_overflow_has_summary_field_context():
    leaves = make_leaves(
        ("zh", "a", 16000, 1, 10**308, ("a", "s1")),
        ("zh", "b", 16000, 1, 10**308, ("b", "s2")),
    )

    with pytest.raises(
        ValueError,
        match="speaker summary estimated_primary_seconds.*supported float range",
    ):
        make_speech_plan(
            leaves,
            requested=2,
            world_size=1,
            batch_size=1,
            sampling_config=speaker_config(1),
        )


def test_speaker_summary_seconds_underflow_has_summary_field_context():
    leaf = SpeechLeafMetadata(
        language="zh",
        dataset_id="a",
        speaker=("a", "s1"),
        batching_sr=16000,
        uid_indices=(0, 1),
        effective_seconds=Fraction(1, 2**1074),
    )

    with pytest.raises(
        ValueError,
        match="speaker summary estimated_primary_seconds.*supported float range",
    ):
        make_speech_plan(
            [leaf], requested=1, world_size=1, batch_size=1,
            sampling_config=speaker_config(1),
        )


# Task 4 Step 7 baseline fixtures：由 Task 0-2 冻结实现（旧 cell 输入、无
# speaker controller）逐项捕获；新实现必须逐位一致。
BASELINE_FIXTURES = {
    "language_only": {
        "cells": [
            *[cell("zh", name, 16000, 10, 1.0) for name in ("a", "b", "c", "d")],
            *[cell("en", name, 16000, 10, 1.0) for name in ("a", "b", "c", "d")],
        ],
        "kwargs": dict(
            requested=10, world_size=1, batch_size=1,
            language_probs={"zh": 0.6, "en": 0.4},
        ),
        "language_quotas": {"en": 4, "zh": 6},
        "global_dataset_quotas": {},
        "cell_quotas": {
            ("en", "a", 16000): 1, ("en", "b", 16000): 1,
            ("en", "c", 16000): 1, ("en", "d", 16000): 1,
            ("zh", "a", 16000): 2, ("zh", "b", 16000): 2,
            ("zh", "c", 16000): 1, ("zh", "d", 16000): 1,
        },
        "sr_quotas": {16000: 10},
        "summary_json": '{"actual": 10, "controllers": {"dataset_enabled": false, "language_enabled": true}, "datasets": {}, "effective_requested": 10, "languages": {"en": {"capacity": 40, "quota": 4, "target": 4}, "zh": {"capacity": 40, "quota": 6, "target": 6}}, "requested": 10, "sample_rates": {"16000": {"capacity": 80, "granularity": 1, "quota": 10}}, "shrink_reason": null, "sr_compensation": []}',
    },
    "joint_sr_compensation": {
        "cells": [
            cell("zh", "a", 16000, 2, 100.0), cell("zh", "a", 48000, 2, 1.0),
            cell("en", "b", 16000, 2, 1.0), cell("en", "b", 48000, 2, 100.0),
        ],
        "kwargs": dict(
            requested=4, world_size=2, batch_size=1,
            language_probs={"zh": 0.5, "en": 0.5},
            dataset_alpha=0, max_dataset_ratio=0.5,
        ),
        "language_quotas": {"en": 2, "zh": 2},
        "global_dataset_quotas": {"a": 2, "b": 2},
        "cell_quotas": {
            ("en", "b", 16000): 0, ("en", "b", 48000): 2,
            ("zh", "a", 16000): 0, ("zh", "a", 48000): 2,
        },
        "sr_quotas": {16000: 0, 48000: 4},
        "summary_json": '{"actual": 4, "controllers": {"dataset_enabled": true, "language_enabled": true}, "datasets": {"a": {"cap": 2, "capacity": 4, "effective_seconds": 101.0, "effective_seconds_scale": 100.0, "quota": 2, "real_target": 2.0, "scaled_effective_seconds": 1.01, "scaled_weight": 1.0, "weight": 1.0}, "b": {"cap": 2, "capacity": 4, "effective_seconds": 101.0, "effective_seconds_scale": 100.0, "quota": 2, "real_target": 2.0, "scaled_effective_seconds": 1.01, "scaled_weight": 1.0, "weight": 1.0}}, "effective_requested": 4, "languages": {"en": {"capacity": 4, "quota": 2, "target": 2}, "zh": {"capacity": 4, "quota": 2, "target": 2}}, "requested": 4, "sample_rates": {"16000": {"capacity": 4, "granularity": 2, "quota": 0}, "48000": {"capacity": 4, "granularity": 2, "quota": 4}}, "shrink_reason": null, "sr_compensation": [{"cell": ["zh", "a", 16000], "from": 2, "to": 0}, {"cell": ["zh", "a", 48000], "from": 0, "to": 2}]}',
    },
    "dataset_only": {
        "cells": [
            cell("zh", "shared", 16000, 10, 9.0),
            cell("en", "shared", 16000, 10, 7.0),
            cell("zh", "other", 16000, 10, 4.0),
        ],
        "kwargs": dict(
            requested=10, world_size=1, batch_size=1,
            dataset_alpha=0.5, max_dataset_ratio=0.6,
        ),
        "language_quotas": {},
        "global_dataset_quotas": {"other": 4, "shared": 6},
        "cell_quotas": {
            ("zh", "other", 16000): 4,
            ("en", "shared", 16000): 3, ("zh", "shared", 16000): 3,
        },
        "sr_quotas": {16000: 10},
        "summary_json": '{"actual": 10, "controllers": {"dataset_enabled": true, "language_enabled": false}, "datasets": {"other": {"cap": 6, "capacity": 10, "effective_seconds": 4.0, "effective_seconds_scale": 9.0, "quota": 4, "real_target": 3.3333333333333335, "scaled_effective_seconds": 0.4444444444444444, "scaled_weight": 0.6666666666666666, "weight": 2.0}, "shared": {"cap": 6, "capacity": 20, "effective_seconds": 16.0, "effective_seconds_scale": 9.0, "quota": 6, "real_target": 6.666666666666667, "scaled_effective_seconds": 1.7777777777777777, "scaled_weight": 1.3333333333333333, "weight": 4.0}}, "effective_requested": 10, "languages": {}, "requested": 10, "sample_rates": {"16000": {"capacity": 30, "granularity": 1, "quota": 10}}, "shrink_reason": null, "sr_compensation": []}',
    },
}


@pytest.mark.parametrize("fixture_name", sorted(BASELINE_FIXTURES))
def test_no_speaker_fast_path_is_bit_for_bit_with_legacy_baseline(fixture_name):
    """Step 7：无 speaker fast path 对 Task 0 baseline 逐位一致。"""
    fixture = BASELINE_FIXTURES[fixture_name]
    plan = make_speech_plan(fixture["cells"], **fixture["kwargs"])
    assert dict(plan.language_quotas) == fixture["language_quotas"]
    assert dict(plan.global_dataset_quotas) == fixture["global_dataset_quotas"]
    assert dict(plan.cell_quotas) == fixture["cell_quotas"]
    assert dict(plan.sr_quotas) == fixture["sr_quotas"]
    assert plan.leaf_quotas == {
        (lang, ds, None, sr): quota
        for (lang, ds, sr), quota in fixture["cell_quotas"].items()
    }
    assert plan.speaker_quotas == {}
    assert plan.summary["joint_solver"]["fast_path"] is True
    assert _compat_summary_json(plan) == fixture["summary_json"]


def test_no_speaker_plan_is_deterministic_under_reversed_leaf_order():
    """Step 7：leaf 输入顺序反转后结果完全一致。"""
    leaves = make_leaves(
        ("zh", "a", 16000, 8, 8.0),
        ("zh", "b", 48000, 2, 2.0),
        ("en", "a", 48000, 2, 2.0),
        ("en", "b", 16000, 8, 8.0),
    )
    kwargs = dict(
        requested=8, world_size=2, batch_size=1,
        language_probs={"zh": 0.5, "en": 0.5},
        dataset_alpha=0, max_dataset_ratio=0.5,
    )
    forward = make_speech_plan(leaves, **kwargs)
    reversed_plan = make_speech_plan(list(reversed(leaves)), **kwargs)
    assert forward.cell_quotas == reversed_plan.cell_quotas
    assert forward.leaf_quotas == reversed_plan.leaf_quotas
    assert _stable_json(forward.summary) == _stable_json(reversed_plan.summary)


def test_alternate_flow_candidate_is_found_by_joint_solver_fallback():
    """Step 7 关联：无 speaker 时旧 fast path 的 SR 失败由 joint solver 接管，
    在本 fixture 上找到 alternate flow 而不是错误 shrink。"""
    leaves = make_leaves(
        ("en", "a", 16000, 1, 1.0),
        ("en", "b", 48000, 4, 1.0),
        ("zh", "a", 48000, 4, 1.0),
        ("zh", "b", 16000, 4, 1.0),
    )
    plan = make_speech_plan(
        leaves, requested=8, world_size=2, batch_size=2, drop_last=True,
        language_probs={"en": 0.5, "zh": 0.5},
        dataset_alpha=0, max_dataset_ratio=1,
    )
    assert plan.actual == 8
    assert plan.language_quotas == {"en": 4, "zh": 4}
    assert plan.global_dataset_quotas == {"a": 4, "b": 4}
    assert all(total % 4 == 0 for total in plan.sr_quotas.values())
    assert plan.summary["joint_solver"]["solver_used"] is True


pytestmark = [pytest.mark.unit]
