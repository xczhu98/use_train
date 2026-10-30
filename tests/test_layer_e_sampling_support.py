"""Layer E 无放回抽样必须满足引用增强的正权重支持数。"""

from copy import deepcopy

import pytest

from core.data.simulation_configs import SimulationConfigs


pytestmark = [pytest.mark.unit]


def _payload(counts, *, weights=(0.0, 1.0), enabled=True):
    names = ["bandwidth_limitation", "clipping"]
    defaults = SimulationConfigs()
    augmentations = {name: deepcopy(defaults.augmentations[name]) for name in names}
    for name, weight in zip(names, weights):
        augmentations[name]["weight"] = weight
    return {
        "prob_channel_enabled": enabled,
        "augmentations_name": names,
        "augmentations": augmentations,
        "num_augmentations": counts,
    }


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("enabled", [True, False])
def test_rejects_count_above_positive_weight_support(strict, enabled):
    with pytest.raises(
        ValueError,
        match="num_augmentations.*positive-probability count 2.*1 positive-weight",
    ):
        SimulationConfigs.from_dict(_payload({2: 1.0}, enabled=enabled), strict=strict)


@pytest.mark.parametrize("probability", [1e-12, 1e-300])
def test_tiny_positive_count_probability_still_requires_support(probability):
    with pytest.raises(ValueError, match="count 2.*1 positive-weight"):
        SimulationConfigs.from_dict(
            _payload({1: 1.0 - probability, 2: probability}), strict=True
        )


@pytest.mark.parametrize("counts", [{0: 1.0}, {1: 1.0, 99: 0.0}])
def test_zero_counts_and_zero_probability_counts_preserve_distribution(counts):
    cfg = SimulationConfigs.from_dict(_payload(counts), strict=True)
    assert cfg.num_augmentations == counts
    assert cfg.augmentations_name == ["bandwidth_limitation", "clipping"]
    assert cfg.augmentations["bandwidth_limitation"]["weight"] == 0.0


@pytest.mark.parametrize("weight", [1.0, 1e-300])
def test_positive_weight_support_includes_tiny_weights(weight):
    cfg = SimulationConfigs.from_dict(
        _payload({2: 1.0}, weights=(weight, 1.0)), strict=True
    )
    assert cfg.num_augmentations == {2: 1.0}
    assert cfg.augmentations["bandwidth_limitation"]["weight"] == weight


@pytest.mark.parametrize("count", [0, 1])
def test_all_zero_referenced_weights_keep_original_error(count):
    with pytest.raises(ValueError, match="augmentation weights.*greater than 0"):
        SimulationConfigs.from_dict(
            _payload({count: 1.0}, weights=(0.0, 0.0)), strict=True
        )


def test_unreferenced_positive_weights_do_not_supply_sampling_support():
    payload = _payload({2: 1.0})
    payload["augmentations"]["extension"] = {"weight": 100.0}
    with pytest.raises(ValueError, match="count 2.*1 positive-weight"):
        SimulationConfigs.from_dict(payload, strict=True)


def test_default_distribution_and_one_level_augmentation_merge_are_preserved():
    defaults = SimulationConfigs.from_dict({}, strict=True)
    clipping = deepcopy(defaults.augmentations["clipping"])
    clipping["weight"] = 0.0
    payload = {"augmentations": {"clipping": clipping}}
    original = deepcopy(payload)
    cfg = SimulationConfigs.from_dict(payload, strict=True)
    assert cfg.num_augmentations == {0: 0.25, 1: 0.40, 2: 0.20, 3: 0.15}
    assert cfg.augmentations_name == defaults.augmentations_name
    assert cfg.augmentations["clipping"] == clipping
    for name in ("bandwidth_limitation", "codec", "packet_loss"):
        assert cfg.augmentations[name] == defaults.augmentations[name]
    assert payload == original
    assert defaults.augmentations["clipping"]["weight"] == 1.0
    with pytest.raises(ValueError, match="clipping.*clipping_min_quantile"):
        SimulationConfigs.from_dict(
            {"augmentations": {"clipping": {"weight": 0.0}}}, strict=True
        )


@pytest.mark.parametrize(
    "counts",
    [
        {0: -0.1, 1: 1.1},
        {0: 1.1},
        {0: float("nan")},
        {0: float("inf")},
        {0: float("-inf")},
        {0: True},
        {0: False, 1: 1.0},
        {0: 0.0},
        {0: 0.4, 1: 0.4},
    ],
)
def test_invalid_count_probabilities_remain_rejected(counts):
    with pytest.raises(ValueError, match="num_augmentations.*probabilities"):
        SimulationConfigs.from_dict(_payload(counts), strict=True)


@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_real_layer_e_sampling_with_zero_weight(count, seed):
    import numpy as np

    from core.data.dynamic_dataset_base import BaseDynamicMixingDataset

    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = SimulationConfigs.from_dict(_payload({count: 1.0}), strict=True)
    dataset._aug_names = list(dataset.sim_cfg.augmentations_name)
    dataset._aug_weights = np.array(
        [dataset.sim_cfg.augmentations[name]["weight"] for name in dataset._aug_names],
        dtype=np.float64,
    )
    dataset._aug_weights /= dataset._aug_weights.sum()
    waveform = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    original = waveform.copy()
    result = dataset._apply_channel_degradation_result(
        waveform, 16000, len(waveform), np.random.default_rng(seed), False
    )
    assert result.enabled is True
    assert result.num_augmentations_drawn == count
    assert result.waveform.shape == waveform.shape
    assert np.isfinite(result.waveform).all()
    np.testing.assert_array_equal(waveform, original)
    if count == 0:
        assert result.selected == ()
        assert result.operations == ()
        assert result.skip_reason == "draw_not_selected"
        assert result.observation_changed is False
        np.testing.assert_array_equal(result.waveform, original)
    else:
        assert result.selected == ("clipping",)
        assert result.applied == ("clipping",)
        assert result.skipped == ()
        assert result.operation_applied("clipping") is True
        assert result.observation_changed is True
        assert result.waveform.min() > original.min()
        assert result.waveform.max() < original.max()
