"""Deterministic contracts for the read-only noise sampling audit."""

import json
import sys
from types import SimpleNamespace

import pytest

from core.data.dynamic_dataset_base import BaseDynamicMixingDataset
from tools.verify import audit_noise_sampling
from tools.verify.audit_noise_sampling import audit_dataset


def _complete_noise_sampling_state(dataset):
    """补齐手工 fixture 与生产初始化一致的噪声派生索引。"""
    dataset._noise_category_id_by_name = {
        name: category_id
        for category_id, name in dataset._noise_category_name_by_id.items()
    }
    natural_positions_by_fs = {}
    natural_uid_position_by_fs = {}
    for fs, uids in dataset._noise_natural_uids_by_fs.items():
        positions_by_category = {}
        uid_positions = {}
        for position, uid in enumerate(uids):
            category_id = dataset._noise_category_id_by_uid[uid]
            positions_by_category.setdefault(category_id, []).append(position)
            uid_positions[uid] = position
        natural_positions_by_fs[fs] = {
            category_id: tuple(positions)
            for category_id, positions in positions_by_category.items()
        }
        natural_uid_position_by_fs[fs] = uid_positions
    dataset._noise_natural_positions_by_fs = natural_positions_by_fs
    dataset._noise_natural_uid_position_by_fs = natural_uid_position_by_fs
    dataset._noise_category_tier_candidates_cache_uid_refs = 0
    dataset._noise_natural_tier_uids_cache_uid_refs = 0


class _ClassifiedDataset(BaseDynamicMixingDataset):
    def __init__(self):
        self.target_sr = 16000
        self.balanced_noise_sampling_enabled = True
        self.noise_source = {
            16000: {
                "vehicle-a": "/not-read/vehicle-a.wav",
                "vehicle-b": "/not-read/vehicle-b.wav",
                "thunder": "/not-read/thunder.wav",
            }
        }
        self.music_source = {}
        self.wind_noises = {}
        self.sim_cfg = SimpleNamespace(
            noise_mix_prob=1.0, prob_wind_noise_sidechain=0.0
        )
        self._noise_sampling_config = SimpleNamespace(
            balance_mode="temperature", alpha=0.5
        )
        self._noise_categories_complete = True
        self._noise_category_tier_candidates_cache = {}
        self._noise_category_order_cache = {}
        self._noise_temperature_base_weights_cache = {}
        self._noise_natural_tier_uids_cache = {}
        self._noise_category_uids_by_fs = {
            16000: {1: ("vehicle-a", "vehicle-b"), 2: ("thunder",)}
        }
        self._noise_natural_uids_by_fs = {
            16000: ("vehicle-a", "vehicle-b", "thunder")
        }
        self._noise_category_id_by_uid = {
            "vehicle-a": 1,
            "vehicle-b": 1,
            "thunder": 2,
        }
        self._noise_category_name_by_id = {1: "Vehicle", 2: "Thunder"}
        self._noise_category_weight_by_id = {1: 2.0, 2: 5.0}
        _complete_noise_sampling_state(self)


class _HigherTierNaturalDataset(BaseDynamicMixingDataset):
    def __init__(self):
        self.target_sr = 24000
        self.balanced_noise_sampling_enabled = False
        self.noise_source = {
            32000: {"vehicle": "/not-read/vehicle.wav"},
            48000: {
                "thunder-a": "/not-read/thunder-a.wav",
                "thunder-b": "/not-read/thunder-b.wav",
            },
        }
        self.music_source = {}
        self.wind_noises = {}
        self.sim_cfg = SimpleNamespace(
            noise_mix_prob=1.0, prob_wind_noise_sidechain=0.0
        )
        self._noise_sampling_config = None
        self._noise_categories_complete = True
        self._noise_category_tier_candidates_cache = {}
        self._noise_category_order_cache = {}
        self._noise_temperature_base_weights_cache = {}
        self._noise_natural_tier_uids_cache = {}
        self._noise_category_uids_by_fs = {
            32000: {1: ("vehicle",)},
            48000: {2: ("thunder-a", "thunder-b")},
        }
        self._noise_natural_uids_by_fs = {
            32000: ("vehicle",),
            48000: ("thunder-a", "thunder-b"),
        }
        self._noise_category_id_by_uid = {
            "vehicle": 1,
            "thunder-a": 2,
            "thunder-b": 2,
        }
        self._noise_category_name_by_id = {1: "Vehicle", 2: "Thunder"}
        _complete_noise_sampling_state(self)


class _LegacyDataset:
    target_sr = 16000
    balanced_noise_sampling_enabled = False
    noise_source = {16000: {"a": "/not-read/a.wav", "b": "/not-read/b.wav"}}
    wind_noises = {}
    sim_cfg = SimpleNamespace(noise_mix_prob=1.0, prob_wind_noise_sidechain=0.0)

    @staticmethod
    def _select_sample(sr, source_by_fs, rng):
        uids = sorted(source_by_fs[sr])
        return uids[int(rng.integers(0, len(uids)))]


def test_audit_is_byte_stable_and_reports_runtime_helper_theory():
    dataset = _ClassifiedDataset()
    first = audit_dataset(dataset, draws=50, seed=7)
    second = audit_dataset(dataset, draws=50, seed=7)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["effective_mode"] == "temperature"
    assert first["alpha"] == 0.5
    assert first["category_available"] is True
    assert first["source_count"] == {"dual": 50}
    assert first["same_category_violation"] == 0
    theory = first["selector_theory"]
    tier, candidates, weights = dataset._temperature_candidates_and_weights(16000)
    assert theory["tier"] == tier == "exact"
    assert theory["categories"]["Vehicle"]["candidate_count"] == len(candidates[1])
    assert theory["categories"]["Vehicle"]["weight"] == weights[1]
    assert theory["categories"]["Thunder"]["weight"] == weights[2]
    assert sum(item["share"] for item in theory["categories"].values()) == pytest.approx(1.0)
    coverage = first["uid_coverage"]
    assert coverage["max_repeat"] == max(coverage["selection_histogram"].values())
    assert coverage["mean_repeat"] == pytest.approx(
        coverage["selections"] / coverage["unique"]
    )


def test_audit_reports_category_equal_and_natural_uid_shares():
    dataset = _ClassifiedDataset()
    dataset._noise_sampling_config = SimpleNamespace(
        balance_mode="category", alpha=0.5
    )
    category = audit_dataset(dataset, draws=5, seed=1)["selector_theory"]
    assert category["categories"]["Vehicle"]["weight"] == 1.0
    assert category["categories"]["Thunder"]["weight"] == 1.0
    assert category["categories"]["Vehicle"]["share"] == 0.5

    dataset.balanced_noise_sampling_enabled = False
    dataset._noise_sampling_config = None
    natural = audit_dataset(dataset, draws=5, seed=1)["selector_theory"]
    assert natural["categories"]["Vehicle"]["weight"] == 2.0
    assert natural["categories"]["Thunder"]["weight"] == 1.0
    assert natural["categories"]["Vehicle"]["share"] == pytest.approx(2 / 3)


def test_natural_theory_aggregates_all_higher_sr_like_runtime_second_source():
    dataset = _HigherTierNaturalDataset()

    report = audit_dataset(dataset, draws=20, seed=3)
    theory = report["selector_theory"]

    assert theory["tier"] == "higher"
    assert theory["categories"] == {
        "Vehicle": {
            "category_id": 1,
            "candidate_count": 1,
            "weight": 1.0,
            "share": 1.0,
            "uid_share": 1.0,
        }
    }

    second = report["natural_second_source_theory"]
    assert second["first_tier"] == "higher"
    assert second["by_first_category"]["Vehicle"] == {
        "excluded_category_id": 1,
        "tier": "higher",
        "categories": {
            "Thunder": {
                "category_id": 2,
                "candidate_count": 2,
                "category_share": 1.0,
                "uid_share": 0.5,
            }
        },
    }
    runtime_second = dataset._noise_natural_tier_view(
        "higher", (32000, 48000), exclude_category_id=1
    )
    assert runtime_second == ("thunder-a", "thunder-b")


def test_alpha_sweep_reports_tier_local_reuse_and_keeps_difficulty_at_alpha_one():
    report = audit_dataset(_ClassifiedDataset(), draws=50, seed=7)
    sweep = report["alpha_sweep"]

    assert sweep["availability"] == "available"
    assert sweep["tier"] == "exact"
    assert sweep["selection_budget"] == 100
    assert list(sweep["alphas"]) == ["0.5", "0.7", "0.85", "1"]
    assert sweep["alpha_one_includes_difficulty"] is True
    alpha_one = sweep["alphas"]["1"]
    vehicle = alpha_one["categories"]["Vehicle"]
    thunder = alpha_one["categories"]["Thunder"]
    assert vehicle["weight"] == pytest.approx(2 * 2.0)
    assert thunder["weight"] == pytest.approx(1 * 5.0)
    assert thunder["category_share"] == pytest.approx(5 / 9)
    assert thunder["raw_difficulty_prior_share"] == pytest.approx(5 / 7)
    assert thunder["difficulty_share"] == thunder["raw_difficulty_prior_share"]
    assert "difficulty_share" in next(iter(alpha_one["deprecated_fields"]))
    exposure = alpha_one["difficulty_exposure_by_value"]
    assert exposure["2"]["selection_exposure_share"] == pytest.approx(4 / 9)
    assert exposure["5"]["selection_exposure_share"] == pytest.approx(5 / 9)
    assert sum(item["selection_exposure_share"] for item in exposure.values()) == pytest.approx(1.0)
    direct_unique = (
        2 * (1 - (1 - 2 / 9) ** sweep["selection_budget"])
        + 1 * (1 - (1 - 5 / 9) ** sweep["selection_budget"])
    )
    assert alpha_one["expected_unique"] == pytest.approx(direct_unique)
    assert alpha_one["expected_unique"] <= sweep["selection_budget"]
    assert alpha_one["expected_repeats"] == pytest.approx(
        sweep["selection_budget"] - alpha_one["expected_unique"]
    )
    assert sum(
        item["expected_unique_uid_exposure_share"] for item in exposure.values()
    ) == pytest.approx(1.0)
    assert alpha_one["rare_reuse"] == {
        "Thunder": {
            "candidate_count": 1,
            "expected_uid_count": thunder["expected_uid_count"],
            "expected_unique": thunder["expected_unique"],
            "expected_repeats": thunder["expected_repeats"],
        }
    }
    assert alpha_one["max_expected_uid_count"] == pytest.approx(
        thunder["expected_uid_count"]
    )


def test_alpha_sweep_accepts_custom_alphas_and_rejects_invalid_values():
    report = audit_dataset(_ClassifiedDataset(), draws=5, seed=1, alphas=(0.7, 1.0))
    assert list(report["alpha_sweep"]["alphas"]) == ["0.7", "1"]

    for invalid in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite values"):
            audit_dataset(_ClassifiedDataset(), draws=5, seed=1, alphas=(invalid,))


def test_alpha_sweep_rejects_nonfinite_or_nonpositive_difficulty():
    for invalid in (0.0, float("nan"), float("inf")):
        dataset = _ClassifiedDataset()
        dataset._noise_category_weight_by_id[1] = invalid
        with pytest.raises(ValueError, match="difficulty must be finite and positive"):
            audit_dataset(dataset, draws=5, seed=1)


@pytest.mark.parametrize("sample_rate", [0, -1, 16000.5, True])
def test_audit_rejects_invalid_processing_sample_rate(sample_rate):
    with pytest.raises(ValueError, match="sample_rate must be a positive integer"):
        audit_dataset(_ClassifiedDataset(), draws=5, seed=1, sample_rate=sample_rate)


def test_audit_reports_category_unavailable_for_ordinary_scp():
    report = audit_dataset(_LegacyDataset(), draws=20, seed=3)

    assert report["effective_mode"] == "natural"
    assert report["category_available"] is False
    assert report["selector_theory"] == {
        "availability": "unavailable",
        "tier": None,
        "categories": {},
    }
    assert report["alpha_sweep"] == {
        "availability": "unavailable",
        "reason": "category metadata incomplete",
    }
    assert report["first_category"] == {"unavailable": 20}
    assert report["second_category"] == {"unavailable": 20}
    assert report["family"] == {"ordinary": 40}
    assert sum(report["uid_coverage"]["selection_histogram"].values()) == 40


def test_cli_json_out_creates_parent_and_records_source_identity(
    tmp_path, monkeypatch, capsys
):
    config = tmp_path / "config.yaml"
    output = tmp_path / "nested" / "audit.json"
    monkeypatch.setattr(audit_noise_sampling, "load_merged_config", lambda _: {})
    monkeypatch.setattr(
        audit_noise_sampling, "_build_dataset", lambda config, split: _LegacyDataset()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_noise_sampling.py",
            str(config),
            "--split",
            "validation",
            "--draws",
            "2",
            "--json-out",
            str(output),
        ],
    )

    audit_noise_sampling.main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert output.parent.is_dir()
    assert report["source"] == {
        "config_path": str(config.resolve()),
        "split": "validation",
    }
    assert json.loads(capsys.readouterr().out) == report


pytestmark = pytest.mark.unit
